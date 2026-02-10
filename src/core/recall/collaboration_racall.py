import sqlite3
import pandas as pd
from config import DB_PATH
import time


class CollaborativeRecallPath:
    """
    协同路召回：基于 DeepICF 的协作关系挖掘
    职责：利用学者间的署名权重、时序衰减及共同合作者发现相关人才
    """

    def __init__(self, top_k=100):
        self.db_path = DB_PATH
        self.top_k = top_k  # 对应文档建议保留前 100 名协作候选人

    def recall(self, seed_author_ids, timeout=0.5):
        if not seed_author_ids: return []

        start_time = time.time()
        deadline = start_time + timeout
        aggregated_results = {}

        # 将种子作者切片，例如每 20 个一组进行扩展
        chunk_size = 20
        conn = sqlite3.connect(self.db_path)

        for i in range(0, len(seed_author_ids), chunk_size):
            if time.time() > deadline:
                break

            chunk = seed_author_ids[i:i + chunk_size]
            placeholders = ','.join(['?'] * len(chunk))

            # 使用预计算的协作相似度 S_total [cite: 3, 4]
            query = f"""
                SELECT target_author_id, SUM(collaboration_score) as score
                FROM scholar_collaboration
                WHERE source_author_id IN ({placeholders})
                AND target_author_id NOT IN ({placeholders})
                GROUP BY target_author_id
            """

            # 排除自身逻辑
            params = chunk + chunk
            df = pd.read_sql_query(query, conn, params=params)

            for _, row in df.iterrows():
                tid, s = row['target_author_id'], row['score']
                aggregated_results[tid] = aggregated_results.get(tid, 0) + s

        conn.close()

        # 排序并返回已发现的最强协作候选人
        sorted_res = sorted(aggregated_results.items(), key=lambda x: x[1], reverse=True)
        return [r[0] for r in sorted_res[:self.top_k]]