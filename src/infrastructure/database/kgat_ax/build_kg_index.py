import sqlite3
import os
import pandas as pd
import time
from config import KGATAX_TRAIN_DATA_DIR
from src.infrastructure.database.kgat_ax.pipeline_state import write_stage_done


class KGIndexBuilder:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.txt_path = os.path.join(output_dir, "kg_final.txt")
        self.db_path = os.path.join(output_dir, "kg_index.db")

    def build(self):
        """
        将 3200 万条三元组结构化到 SQLite，并建立覆盖索引。
        修改点：支持权重字段 'w'，适配加权拓扑收割逻辑。
        """
        # 1. 检查源文本文件是否存在
        if not os.path.exists(self.txt_path):
            print(f"[Error] 未发现源文件: {self.txt_path}")
            return

        # 2. 检查索引数据库是否已存在且完整
        if os.path.exists(self.db_path):
            try:
                check_conn = sqlite3.connect(self.db_path)
                cursor = check_conn.cursor()
                # 校验表是否存在以及是否包含权重字段 w
                table_info = cursor.execute("PRAGMA table_info(kg_triplets)").fetchall()
                has_weight = any(col[1] == 'w' for col in table_info)

                if table_info and has_weight:
                    count = cursor.execute("SELECT count(*) FROM kg_triplets").fetchone()[0]
                    if count > 0:
                        print(f"[*] 发现已存在的加权索引库 ({self.db_path})，包含 {count} 条数据。跳过构建。")
                        check_conn.close()
                        write_stage_done(2, {"skipped": True, "reason": "existing_db"})
                        return
                check_conn.close()
                print("[*] 现有索引库版本过旧或损坏，准备重新构建...")
            except Exception as e:
                print(f"[*] 校验异常 ({str(e)})，准备重新构建...")

        # 3. 启动构建流程
        print(f"[*] 启动加权 KG 索引构建流程...")
        start_time = time.time()

        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = OFF;
            PRAGMA cache_size = 100000; 

            DROP TABLE IF EXISTS kg_triplets;
            -- 核心修改：增加 w REAL 字段以存储权重
            CREATE TABLE kg_triplets (
                h INTEGER,
                r INTEGER,
                t INTEGER,
                w REAL
            );
        """)

        # 4. 流式分块导入 (适配 generate_training_data.py 的 4 列输出格式)
        print(f"[*] 正在从 TXT 导入加权数据到 SQLite...")
        chunksize = 2000000
        total_rows = 0

        # 核心修改：names 参数增加 'w'
        for chunk in pd.read_csv(self.txt_path, sep=' ', names=['h', 'r', 't', 'w'], chunksize=chunksize):
            # 拦截自环边
            chunk = chunk[chunk['h'] != chunk['t']]
            chunk.to_sql("kg_triplets", conn, if_exists="append", index=False)
            total_rows += len(chunk)
            print(f"  - 已处理 {total_rows / 1000000:.1f} M 条加权边...")

        # 5. 构建加权覆盖索引 (Covering Index)
        # 将 w 字段纳入索引，确保数据读取（Index Only Scan）无需回表，大幅提升采样速度
        print(f"[*] 正在构建双向加权覆盖索引...")
        conn.executescript("""
                           -- 正向覆盖索引: (h, r, t, w)
                           CREATE INDEX IF NOT EXISTS idx_h_lookup ON kg_triplets(h, r, t, w);
                           -- 反向覆盖索引: (t, r, h, w)
                           CREATE INDEX IF NOT EXISTS idx_t_lookup ON kg_triplets(t, r, h, w);
                           ANALYZE;
                           """)

        duration = time.time() - start_time
        print(f"\n[OK] 加权索引库构建完成！")
        print(f" - 最终有效边数: {total_rows}")
        print(f" - 索引文件路径: {self.db_path}")
        print(f" - 总耗时: {duration:.2f} 秒")

        conn.close()
        write_stage_done(2, {"skipped": False})

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