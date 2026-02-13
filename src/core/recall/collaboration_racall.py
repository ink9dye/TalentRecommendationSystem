import sqlite3
import pandas as pd
import time
# 确保从你的配置中导入 COLLAB_DB_PATH
from config import COLLAB_DB_PATH


class CollaborativeRecallPath:
    """
    协同路召回：基于预计算的学者协作索引
    职责：利用学者间的署名权重、时序衰减及共同合作者发现相关人才
    """

    def __init__(self, top_k=100):
        # 注意：这里应指向协作度索引数据库，而非原始业务库
        self.collab_db_path = COLLAB_DB_PATH
        self.top_k = top_k

    def recall(self, seed_author_ids, timeout=1.0):
        """
        根据种子作者 ID 召回协作关系最紧密的候选人
        """
        if not seed_author_ids:
            return []

        start_time = time.time()
        deadline = start_time + timeout
        aggregated_results = {}

        # 建立连接
        conn = sqlite3.connect(self.collab_db_path)

        try:
            # 分批处理种子作者，防止 SQL 占位符过多
            chunk_size = 50
            for i in range(0, len(seed_author_ids), chunk_size):
                if time.time() > deadline:
                    break

                chunk = seed_author_ids[i:i + chunk_size]
                placeholders = ','.join(['?'] * len(chunk))

                # 修改点 1：字段名匹配 (aid1, aid2, score)
                # 修改点 2：双向查询。因为索引存储时保证了 aid1 < aid2
                # 我们需要查 (seed as aid1 -> target is aid2) 和 (seed as aid2 -> target is aid1)
                query = f"""
                    SELECT aid2 as target_id, score FROM scholar_collaboration 
                    WHERE aid1 IN ({placeholders})
                    UNION ALL
                    SELECT aid1 as target_id, score FROM scholar_collaboration 
                    WHERE aid2 IN ({placeholders})
                """

                df = pd.read_sql_query(query, conn, params=chunk + chunk)

                for _, row in df.iterrows():
                    tid, s = row['target_id'], row['score']
                    # 过滤掉种子作者自身
                    if tid in seed_author_ids:
                        continue
                    aggregated_results[tid] = aggregated_results.get(tid, 0) + s

        finally:
            conn.close()

        # 排序并返回得分最高的 Top K 候选人
        sorted_res = sorted(aggregated_results.items(), key=lambda x: x[1], reverse=True)
        return [r[0] for r in sorted_res[:self.top_k]]