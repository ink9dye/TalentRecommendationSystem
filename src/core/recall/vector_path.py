import faiss
import json
import sqlite3
import time
import numpy as np
from config import ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH, DB_PATH, JOB_INDEX_PATH, JOB_MAP_PATH


class VectorPath:
    """
    向量路召回：实现基于 SBERT 的语义召回
    逻辑：输入向量 -> 锚定岗位描述(打印) -> 检索相似摘要 -> 获取作者 ID
    """

    def __init__(self, recall_limit=100):
        self.search_k = 500  # 检索深度
        self.recall_limit = recall_limit

        # 1. 加载论文摘要索引 (用于语义召回)
        self.index = faiss.read_index(ABSTRACT_INDEX_PATH)
        with open(ABSTRACT_MAP_PATH, 'r', encoding='utf-8') as f:
            self.id_map = json.load(f)

        # 2. 加载岗位描述索引 (仅用于中间过程打印，辅助观察召回了哪些岗位)
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_id_map = json.load(f)

    def recall(self, query_vector):
        start_t = time.time()
        conn = sqlite3.connect(DB_PATH)

        try:
            # --- 步骤 1: 岗位锚定打印 (解释向量当前匹配到了哪些真实岗位) ---
            _, j_indices = self.job_index.search(query_vector, 3)
            j_ids = [self.job_id_map[idx] for idx in j_indices[0] if 0 <= idx < len(self.job_id_map)]

            job_placeholders = ','.join(['?'] * len(j_ids))
            jobs = conn.execute(f"SELECT job_name, description FROM jobs WHERE securityId IN ({job_placeholders})",
                                j_ids).fetchall()

            print(f"\n[向量路中间过程 - 匹配岗位需求]")
            for name, desc in jobs:
                print(f" - 岗位: {name} | 描述: {desc[:60]}...")

            # --- 步骤 2: 语义检索相似论文 ---
            _, indices = self.index.search(query_vector, self.search_k)
            work_ids = [self.id_map[idx] for idx in indices[0] if 0 <= idx < len(self.id_map)]

            if not work_ids:
                return [], (time.time() - start_t) * 1000

            # --- 步骤 3: 打印相似论文详细信息 (Top 5) ---
            # 修正：从 works 获取标题/标签，从 abstracts 获取摘要正文
            work_ids_limit = work_ids[:5]
            placeholders = ','.join(['?'] * len(work_ids_limit))

            # 这里使用了 LEFT JOIN 确保即使某些论文没摘要也能显示标题
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
    v_path = VectorPath(recall_limit=100)

    print("\n" + "=" * 60)
    print("🚀 向量路测试模式 (岗位锚定 + 摘要检索)")
    print("=" * 60)

    try:
        while True:
            raw_input = input("\n请粘贴向量数据 (list格式) 或输入 'q' 退出:\n>> ").strip()
            if raw_input.lower() in ['q', 'exit']: break
            if not raw_input: continue

            try:
                vector_list = json.loads(raw_input)
                query_vec = np.array([vector_list]).astype('float32')
                faiss.normalize_L2(query_vec)

                id_list, search_time = v_path.recall(query_vec)

                print(f"\n[结果报告]")
                print(f"- 耗时: {search_time:.2f} ms")
                print(f"- 召回候选人数量: {len(id_list)}")
                print("-" * 30)
                print(f"前10位作者ID: {id_list[:10]}")
                print("-" * 30)

            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"[错误] 处理失败: {e}")

    except KeyboardInterrupt:
        print("\n已退出")