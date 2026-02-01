import sqlite3
from neo4j import GraphDatabase
from tqdm import tqdm
import os
import time

# --- 基础配置信息 ---
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Aa1278389701"
NEO4J_DATABASE = "talent-graph"

# 协作相似度索引存放路径
INDEX_PATH = r"E:\PythonProject\TalentRecommendationSystem\data\build_index\scholar_collaboration.db"

class ScholarSimilarityIndexer:
    """
    学者协作相似度索引构建器
    优化策略：移除分数阈值，依靠 [0..100] 剪枝确保每个学者尽可能获得 100 名候选人。
    """
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self._init_sqlite()

    def _init_sqlite(self):
        """初始化本地索引库"""
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        self.conn = sqlite3.connect(INDEX_PATH)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("DROP TABLE IF EXISTS scholar_collaboration")
        self.conn.execute("""
                          CREATE TABLE scholar_collaboration
                          (
                              aid1  TEXT,
                              aid2  TEXT,
                              score REAL,
                              PRIMARY KEY (aid1, aid2)
                          )
                          """)

    def build_index(self):
        """执行全量协作挖掘"""
        print("--- 启动：学者协作相似度索引构建 (全量召回优化版) ---")

        # 1. 获取所有学者 ID
        with self.driver.session(database=NEO4J_DATABASE) as session:
            author_ids = [record["id"] for record in session.run("MATCH (a:Author) RETURN a.id as id")]

        print(f"[*] 监测到 {len(author_ids)} 名学者，正在构建深度协作网络...")

        # 2. 分批处理
        INNER_BATCH = 80
        batch_to_sqlite = []

        # 【Cypher 逻辑变更】
        # 移除了 WHERE direct_score > 0.2，改为纯排序截断
        # 确保只要有合作关系，就会被记录在案（最多 100 人）
        # 【算法升级：双边对称归一化版】
        # 1. 计算一跳分数并执行双边归一化: Score / (sqrt(d1)*sqrt(d2))
        # 2. 计算二跳分数并执行路径规范化: (Score1*Score2) / (d_bridge * sqrt(d1)*sqrt(d2))
        # 3. 最终融合权重 alpha = 0.3
        query = """
                        MATCH (a1:Author)-[r1:AUTHORED]->(w:Work)<-[r2:AUTHORED]-(a2:Author)
                        WHERE a1.id IN $id_list AND a1.id < a2.id

                        WITH a1, a2, 
                             sum(r1.weight * r2.weight * log(w.citations + 2.71828)) AS raw_direct

                        WITH a1, a2, 
                             raw_direct / (sqrt(a1.h_index + 1) * sqrt(a2.h_index + 1)) AS normalized_direct

                        // 仅对最有潜力的前 100 名候选人执行昂贵的二跳计算
                        ORDER BY normalized_direct DESC
                        WITH a1, collect({a2: a2, nd: normalized_direct})[0..100] AS top_candidates
                        UNWIND top_candidates AS candidate
                        WITH a1, candidate.a2 AS a2, candidate.nd AS normalized_direct

                        OPTIONAL MATCH (a1)-[r_a1b:AUTHORED]->(w1:Work)<-[r_ba1:AUTHORED]-(bridge:Author)-[r_ba2:AUTHORED]->(w2:Work)<-[r_a2b:AUTHORED]-(a2)
                        WHERE bridge <> a1 AND bridge <> a2

                        WITH a1, a2, normalized_direct, bridge,
                             (r_a1b.weight * r_ba1.weight * log(w1.citations + 2.71828)) AS s1,
                             (r_ba2.weight * r_a2b.weight * log(w2.citations + 2.71828)) AS s2

                        WITH a1, a2, normalized_direct,
                             sum((s1 * s2) / ((bridge.h_index + 1) * sqrt(a1.h_index + 1) * sqrt(a2.h_index + 1))) AS normalized_indirect

                        RETURN a1.id AS aid1, a2.id AS aid2, 
                               (normalized_direct + 0.3 * coalesce(normalized_indirect, 0)) AS final_score
                        """

        for i in tqdm(range(0, len(author_ids), INNER_BATCH), desc="全局协作挖掘"):
            current_ids = author_ids[i: i + INNER_BATCH]

            try:
                with self.driver.session(database=NEO4J_DATABASE) as session:
                    result = session.run(query, id_list=current_ids)
                    for record in result:
                        batch_to_sqlite.append((record['aid1'], record['aid2'], record['final_score']))
                        batch_to_sqlite.append((record['aid2'], record['aid1'], record['final_score']))
            except Exception as e:
                with open("indexing_errors.log", "a") as f:
                    f.write(f"Batch {i} failed: {str(e)}\n")
                continue

            if len(batch_to_sqlite) >= 2000:
                self._save_batch(batch_to_sqlite)
                batch_to_sqlite = []

        if batch_to_sqlite:
            self._save_batch(batch_to_sqlite)

        print("正在建立 SQLite 索引...")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_aid1 ON scholar_collaboration(aid1)")
        print(f"--- 构建完成，索引存储于: {INDEX_PATH} ---")

    def _save_batch(self, batch):
        self.conn.executemany("INSERT OR REPLACE INTO scholar_collaboration VALUES (?, ?, ?)", batch)
        self.conn.commit()

    def close(self):
        self.driver.close()
        self.conn.close()

if __name__ == "__main__":
    indexer = ScholarSimilarityIndexer()
    try:
        start_time = time.time()
        indexer.build_index()
        print(f"总耗时: {(time.time() - start_time)/3600:.2f} 小时")
    finally:
        indexer.close()