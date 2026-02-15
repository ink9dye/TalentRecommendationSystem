import sqlite3
import os
import pandas as pd
import time
from config import KGATAX_TRAIN_DATA_DIR


class KGIndexBuilder:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.txt_path = os.path.join(output_dir, "kg_final.txt")
        self.db_path = os.path.join(output_dir, "kg_index.db")

    def build(self):
        """将 3200 万条三元组结构化到 SQLite，并建立覆盖索引"""
        if not os.path.exists(self.txt_path):
            print(f"[Error] 未发现源文件: {self.txt_path}")
            return

        print(f"[*] 启动 KG 索引构建流程...")
        start_time = time.time()

        # 1. 初始化数据库环境
        # 使用 WAL 模式和关闭同步以极大提升写入速度
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = OFF;
            PRAGMA cache_size = 100000; -- 100MB 缓存

            DROP TABLE IF EXISTS kg_triplets;
            CREATE TABLE kg_triplets (
                h INTEGER,
                r INTEGER,
                t INTEGER
            );
        """)

        # 2. 流式分块导入 (保护 32GB 内存不爆)
        print(f"[*] 正在从 TXT 导入数据到 SQLite (分块处理)...")
        chunksize = 2000000  # 每次处理 200 万行
        total_rows = 0

        # 使用 pandas 快速解析文本并写入数据库
        for chunk in pd.read_csv(self.txt_path, sep=' ', names=['h', 'r', 't'], chunksize=chunksize):
            # 这里的自环拦截是最后的防线
            chunk = chunk[chunk['h'] != chunk['t']]
            chunk.to_sql("kg_triplets", conn, if_exists="append", index=False)
            total_rows += len(chunk)
            print(f"  - 已处理 {total_rows / 1000000:.1f} M 条边...")

        # 3. 核心步骤：构建覆盖索引 (Covering Index)
        # 覆盖索引允许 SQLite 直接从索引树返回数据，无需“回表”查询原始数据页，速度极快
        print(f"[*] 正在构建双向覆盖索引 (这可能需要 1-3 分钟)...")
        conn.executescript("""
                           -- 加速正向查询: 根据头节点找邻居
                           CREATE INDEX IF NOT EXISTS idx_h_lookup ON kg_triplets(h, r, t);
                           -- 加速反向查询: 根据尾节点找邻居
                           CREATE INDEX IF NOT EXISTS idx_t_lookup ON kg_triplets(t, r, h);
                           ANALYZE; -- 更新统计信息让查询优化器更聪明
                           """)

        duration = time.time() - start_time
        print(f"\n[OK] 索引库构建完成！")
        print(f" - 最终有效边数: {total_rows}")
        print(f" - 索引文件路径: {self.db_path}")
        print(f" - 总耗时: {duration:.2f} 秒")

        conn.close()


def main():
    # 使用与您的训练生成器一致的输出目录
    builder = KGIndexBuilder(KGATAX_TRAIN_DATA_DIR)

    print("=" * 50)
    print(" 知识图谱离线索引构建工具 (KGAT 训练加速版) ")
    print("=" * 50)

    try:
        builder.build()
        print("\n提示: 现在您可以在 DataLoaderKGAT 中通过 SQL 语句执行秒级子图采样了。")
    except Exception as e:
        print(f"\n[CRITICAL] 构建失败: {str(e)}")


if __name__ == "__main__":
    main()