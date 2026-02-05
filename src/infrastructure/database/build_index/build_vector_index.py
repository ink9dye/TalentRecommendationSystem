import sqlite3
import json
import numpy as np
import faiss
import time
import os
from neo4j import GraphDatabase
from datetime import datetime
from tqdm import tqdm

# --- 配置 ---
DB_PATH = r"E:\PythonProject\TalentRecommendationSystem\data\academic_dataset_v5.db"
INDEX_DIR = r"E:\PythonProject\TalentRecommendationSystem\data\build_index"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Aa1278389701"
NEO4J_DATABASE = "talent-graph"

BATCH_SIZE = 2000
SLEEP_TIME = 0.1


class KnowledgeGraphBuilder:
    def __init__(self, reference_year=None):
        # 增加连接池配置和超时设置
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=60.0,  # 连接超时增加到60秒
            max_transaction_retry_time=30.0,  # 自动重试时间
            keep_alive=True  # 保持长连接
        )
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
        self.db_name = NEO4J_DATABASE
        self.reference_year = reference_year if reference_year else datetime.now().year

    def close(self):
        self.conn.close()
        self.driver.close()

    def init_db(self):
        """创建约束，确保 MATCH 性能"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (w:Work) REQUIRE w.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Vocabulary) REQUIRE v.id IS UNIQUE"
        ]
        with self.driver.session(database=self.db_name) as session:
            for cmd in constraints:
                session.run(cmd)

    def load_nodes(self):
        """优化版导入节点：流式读取 + 分批提交，彻底解决超时问题"""
        cursor = self.conn.cursor()

        tasks = [
            {
                "name": "Author",
                "sql": "SELECT author_id, name, h_index, cited_by_count FROM authors",
                "cypher": "UNWIND $data AS row MERGE (a:Author {id: row.author_id}) SET a.name = row.name, a.h_index = row.h_index, a.citations = row.cited_by_count"
            },
            {
                "name": "Work",
                "sql": "SELECT work_id, title, year, citation_count FROM works",
                "cypher": "UNWIND $data AS row MERGE (w:Work {id: row.work_id}) SET w.title = row.title, w.year = row.year, w.citations = row.citation_count"
            },
            {
                "name": "Job",
                "sql": "SELECT securityId, job_name, company, keyword FROM jobs",
                "cypher": "UNWIND $data AS row MERGE (j:Job {id: row.securityId}) SET j.name = row.job_name, j.company = row.company, j.category = row.keyword"
            },
            {
                "name": "Vocabulary",
                "sql": "SELECT id, term, entity_type FROM vocabulary",
                "cypher": "UNWIND $data AS row MERGE (v:Vocabulary {id: row.id}) SET v.term = row.term, v.type = row.entity_type"
            }
        ]

        for task in tasks:
            print(f"--- 正在处理节点任务: {task['name']} ---")
            cursor.execute(task['sql'])

            batch = []
            # 使用 tqdm 监控进度（虽然 fetchone 不知道总数，我们可以通过总行数计算）
            # 先查一下总行数
            cursor_count = self.conn.cursor()
            cursor_count.execute(f"SELECT count(*) FROM ({task['sql']})")
            total_rows = cursor_count.fetchone()[0]

            with tqdm(total=total_rows, desc=f"导入 {task['name']}") as pbar:
                while True:
                    row = cursor.fetchone()
                    if row is None:
                        break

                    batch.append(dict(row))

                    # 达到批次大小，立即发送给 Neo4j
                    if len(batch) >= BATCH_SIZE:
                        self._safe_execute_cypher(task['cypher'], batch)
                        batch = []  # 重置批次内存
                        pbar.update(BATCH_SIZE)
                        time.sleep(SLEEP_TIME)  # 核心：给 CPU 和数据库喘息时间

                # 提交最后一批不满 BATCH_SIZE 的数据
                if batch:
                    self._safe_execute_cypher(task['cypher'], batch)
                    pbar.update(len(batch))

    def _safe_execute_cypher(self, cypher, data):
        """封装一个带重试逻辑的执行函数，应对网络抖动"""
        try:
            with self.driver.session(database=self.db_name) as session:
                session.run(cypher, {"data": data})
        except Exception as e:
            print(f"\n[写入失败] 正在重试... 错误原因: {e}")
            time.sleep(1)  # 等待1秒后重试一次
            with self.driver.session(database=self.db_name) as session:
                session.run(cypher, {"data": data})

    def load_relationships(self):
        """
        构建 AUTHORED 关系并注入权重逻辑。
        对应算法：$weight = e^{-0.1 \cdot \Delta t} \cdot pos\_weight$
        """
        cursor = self.conn.cursor()
        print("--- 启动：构建 AUTHORED 关系并计算时序权重 ---")

        # 1. 预先获取总行数用于进度条展示
        cursor.execute("SELECT count(*) FROM authorships")
        total_count = cursor.fetchone()[0]

        # 2. 流式查询（JOIN works 表获取年份用于计算衰减）
        rels_query = """
                     SELECT a.author_id, a.work_id, w.year, a.pos_index, a.is_corresponding
                     FROM authorships a
                              JOIN works w ON a.work_id = w.work_id \
                     """
        cursor.execute(rels_query)

        batch_data = []
        cypher_query = """
        UNWIND $data AS row
        MATCH (a:Author {id: row.aid}), (w:Work {id: row.wid})
        MERGE (a)-[r:AUTHORED]->(w)
        SET r.weight = row.weight
        """

        with tqdm(total=total_count, desc="处理署名关系权重") as pbar:
            while True:
                row = cursor.fetchone()
                if row is None:
                    break

                # 权重逻辑计算
                year = row['year'] if row['year'] else self.reference_year
                delta_t = max(0, self.reference_year - year)

                # 时序衰减权重计算
                time_decay = np.exp(-0.1 * delta_t)
                # 通讯作者或第一作者加权 (pos_index=0)
                pos_weight = 1.2 if (row['pos_index'] == 0 or row['is_corresponding'] == 1) else 1.0

                batch_data.append({
                    "aid": row['author_id'],
                    "wid": row['work_id'],
                    "weight": float(time_decay * pos_weight)
                })

                # 达到批次大小则安全写入
                if len(batch_data) >= BATCH_SIZE:
                    self._safe_execute_cypher(cypher_query, batch_data)
                    batch_data = []
                    pbar.update(BATCH_SIZE)
                    time.sleep(SLEEP_TIME)

            # 提交剩余部分
            if batch_data:
                self._safe_execute_cypher(cypher_query, batch_data)
                pbar.update(len(batch_data))

    def build_semantic_bridge(self, min_edges=3):
        """
        使用 Faiss 建立语义桥梁。
        解决中英文术语对应障碍，确保岗位技能点与学术词汇的连通。
        """
        print(f"--- 启动：构建语义桥梁 (技能 -> 词汇) | 目标边数: {min_edges} ---")

        # 1. 加载 Faiss 索引和映射
        index_path = os.path.join(INDEX_DIR, "vocabulary.faiss")
        map_path = os.path.join(INDEX_DIR, "vocabulary_mapping.json")

        if not os.path.exists(index_path) or not os.path.exists(map_path):
            print("【错误】未找到 Faiss 索引或 ID 映射文件，请先运行索引构建脚本！")
            return

        index = faiss.read_index(index_path)
        with open(map_path, 'r', encoding='utf-8') as f:
            all_ids = json.load(f)

        # 2. 获取所有技能节点（需要作为起点建立关联的节点）
        cursor = self.conn.cursor()
        skills = cursor.execute("SELECT id, term FROM vocabulary WHERE entity_type = 'skill'").fetchall()

        bridge_batch = []
        cypher_query = """
        UNWIND $data AS row
        MATCH (v1:Vocabulary {id: row.from_id}), (v2:Vocabulary {id: row.to_id})
        MERGE (v1)-[r:SIMILAR_TO]->(v2)
        SET r.score = row.score
        """

        for s in tqdm(skills, desc="向量检索建立语义边"):
            s_id_str = str(s['id'])
            if s_id_str not in all_ids:
                continue

            try:
                # 获取该技能在 Faiss 中的向量并检索相似词
                idx_in_faiss = all_ids.index(s_id_str)
                vector = index.reconstruct(idx_in_faiss).reshape(1, -1)
                distances, indices = index.search(vector, 15)  # 获取前 15 个候选

                valid_neighbors_count = 0
                for dist, neighbor_idx in zip(distances[0], indices[0]):
                    neighbor_id_str = all_ids[neighbor_idx]

                    if neighbor_id_str == s_id_str:
                        continue  # 跳过自身

                    bridge_batch.append({
                        "from_id": int(s_id_str),
                        "to_id": int(neighbor_id_str),
                        "score": float(dist)
                    })

                    valid_neighbors_count += 1
                    if valid_neighbors_count >= min_edges:
                        break

                # 分批写入 Neo4j
                if len(bridge_batch) >= BATCH_SIZE:
                    self._safe_execute_cypher(cypher_query, bridge_batch)
                    bridge_batch = []
                    time.sleep(SLEEP_TIME)

            except Exception as e:
                # 记录单个词条的错误，但不中断整个任务
                continue

        # 提交最后一批
        if bridge_batch:
            self._safe_execute_cypher(cypher_query, bridge_batch)

    def run_pipeline(self):
        self.init_db()
        self.load_nodes()
        self.load_relationships()
        self.build_semantic_bridge()
        print("--- 知识图谱构建完成 ---")


if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    try:
        builder.run_pipeline()
    finally:
        builder.close()