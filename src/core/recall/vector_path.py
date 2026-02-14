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

    def __init__(self, recall_limit=200):
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
    # 测试模式下可以开启 verbose=True
    v_path = VectorPath(recall_limit=200)
    # ... 剩下的测试代码调用时改为 v_path.recall(query_vec, verbose=True)