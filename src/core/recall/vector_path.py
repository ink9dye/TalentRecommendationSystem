import faiss
import json
import sqlite3
import time
import numpy as np
from config import ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH, DB_PATH, JOB_INDEX_PATH, JOB_MAP_PATH


class VectorPath:
    """
    向量路召回：实现基于 SBERT 的语义召回
    """

    def __init__(self, recall_limit=300):
        self.search_k = 500  # 检索深度
        self.recall_limit = recall_limit

        # 1. 加载论文摘要索引
        self.index = faiss.read_index(ABSTRACT_INDEX_PATH)
        with open(ABSTRACT_MAP_PATH, 'r', encoding='utf-8') as f:
            self.id_map = json.load(f)

        # 2. 加载岗位描述索引
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_id_map = json.load(f)

    def recall(self, query_vector, verbose=False):
        """
        召回主逻辑
        :param query_vector: 输入向量
        :param verbose: 是否打印中间调试信息（岗位锚定、论文概要等）
        """
        start_t = time.time()
        conn = sqlite3.connect(DB_PATH)

        try:
            # --- 步骤 1: 岗位锚定 (仅在 verbose 为 True 时执行查询和打印) ---
            if verbose:
                _, j_indices = self.job_index.search(query_vector, 3)
                j_ids = [self.job_id_map[idx] for idx in j_indices[0] if 0 <= idx < len(self.job_id_map)]

                if j_ids:
                    job_placeholders = ','.join(['?'] * len(j_ids))
                    jobs = conn.execute(
                        f"SELECT job_name, description FROM jobs WHERE securityId IN ({job_placeholders})",
                        j_ids).fetchall()
                    print(f"\n[向量路中间过程 - 匹配岗位需求]")
                    for name, desc in jobs:
                        print(f" - 岗位: {name} | 描述: {desc[:60]}...")

            # --- 步骤 2: 语义检索相似论文 ---
            _, indices = self.index.search(query_vector, self.search_k)
            work_ids = [self.id_map[idx] for idx in indices[0] if 0 <= idx < len(self.id_map)]

            if not work_ids:
                return [], (time.time() - start_t) * 1000

            # --- 步骤 3: 打印相似论文详细信息 (仅在 verbose 为 True 时执行) ---
            if verbose:
                work_ids_limit = work_ids[:5]
                placeholders = ','.join(['?'] * len(work_ids_limit))
                paper_query = f"""
                    SELECT w.title, w.concepts_text, a.full_text_en 
                    FROM works w 
                    LEFT JOIN abstracts a ON w.work_id = a.work_id 
                    WHERE w.work_id IN ({placeholders})
                """
                papers = conn.execute(paper_query, work_ids_limit).fetchall()
                print(f"\n[向量路中间过程 - 语义相关论文概要]")
                for title, tags, abstract_text in papers:
                    print(f" - 标题: {title}")
                    print(f"   关键词: {tags if tags else 'N/A'}")
                    print(f"   摘要: {abstract_text[:100] if abstract_text else 'N/A'}...")

            # --- 步骤 4: 获取对应的作者 ID ---
            all_work_placeholders = ','.join(['?'] * len(work_ids))
            author_query = f"SELECT DISTINCT author_id FROM authorships WHERE work_id IN ({all_work_placeholders})"
            author_ids = [row[0] for row in conn.execute(author_query, work_ids).fetchall()]

        finally:
            conn.close()

        duration = (time.time() - start_t) * 1000
        return author_ids[:self.recall_limit], duration


if __name__ == "__main__":
    # 实例化向量路召回组件，设置召回上限为300
    v_path = VectorPath(recall_limit=300)

    print("\n" + "=" * 60)
    print("🚀 向量路召回独立测试控制台")
    print("请输入查询向量 (JSON列表格式) 进行语义召回")
    print("=" * 60)

    try:
        while True:
            raw_input = input("\n请粘贴稠密向量 (输入 'q' 退出):\n>> ").strip()

            # 检查退出条件
            if not raw_input or raw_input.lower() == 'q':
                print("退出测试模式")
                break

            try:
                # 解析JSON格式的向量列表
                vector_list = json.loads(raw_input)

                # 转换为numpy数组并调整维度
                query_vec = np.array([vector_list]).astype('float32')

                # 执行L2归一化以适应Faiss的内积索引
                faiss.normalize_L2(query_vec)

                # 执行召回，开启verbose模式显示中间过程
                print("\n[执行召回中...]")
                author_ids, duration = v_path.recall(query_vec, verbose=True)

                # 输出结果
                print(f"\n[召回结果报告]")
                print(f"- 召回耗时: {duration:.2f} ms")
                print(f"- 召回作者数量: {len(author_ids)}")

                if author_ids:
                    print(f"- 前5位作者ID: {author_ids[:5]}")
                    print(f"- 全部作者ID列表:")
                    # 每行显示10个ID，提高可读性
                    for i in range(0, len(author_ids), 10):
                        print(f"  {author_ids[i:i + 10]}")
                else:
                    print("- 未找到相关作者")

                print("-" * 30)

            except json.JSONDecodeError:
                print("[错误] JSON格式解析失败，请确保输入的是有效的JSON列表格式")
                print("示例: [0.1, -0.2, 0.3, ...]")
            except Exception as e:
                print(f"[错误] 召回过程发生异常: {str(e)}")
                import traceback

                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n\n[!] 测试被用户中断")

    print("[*] 向量路召回测试结束")