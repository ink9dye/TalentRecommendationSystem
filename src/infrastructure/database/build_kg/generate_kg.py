import sqlite3
import json
import numpy as np
import faiss
import time
import os
from neo4j import GraphDatabase
from datetime import datetime
from tqdm import tqdm

# --- 核心配置 ---
DB_PATH = r"E:\PythonProject\TalentRecommendationSystem\data\academic_dataset_v5.db"
INDEX_DIR = r"E:\PythonProject\TalentRecommendationSystem\data\build_index"
FEATURE_JSON = os.path.join(INDEX_DIR, "feature_index.json")
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Aa1278389701"
NEO4J_DATABASE = "talent-graph"

# --- 性能优化参数 ---
BATCH_SIZE = 2000
SLEEP_TIME = 0.1

class KnowledgeGraphBuilder:
    def __init__(self, reference_year=None):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
        self.db_name = NEO4J_DATABASE
        # 设置基准年份用于计算时序衰减
        self.reference_year = reference_year if reference_year else datetime.now().year
        print(f"--- 知识图谱构建启动 | 目标库: {self.db_name} | 基准年份: {self.reference_year} ---")

    def close(self):
        self.conn.close()
        self.driver.close()

    def init_db(self):
        """初始化唯一性约束，确保 MATCH 效率并防止重复 """
        print("正在初始化数据库约束...")
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (w:Work) REQUIRE w.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Vocabulary) REQUIRE v.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE"
        ]
        with self.driver.session(database=self.db_name) as session:
            for cmd in constraints:
                session.run(cmd)

    def load_nodes(self):
        """分批导入全量节点数据"""
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
                "name": "Institution",
                "sql": "SELECT inst_id, name, cited_by_count FROM institutions",
                "cypher": "UNWIND $data AS row MERGE (i:Institution {id: row.inst_id}) SET i.name = row.name, i.citations = row.cited_by_count"
            },
            {
                "name": "Source", # 校准：使用 display_name
                "sql": "SELECT source_id, display_name FROM sources",
                "cypher": "UNWIND $data AS row MERGE (s:Source {id: row.source_id}) SET s.name = row.display_name"
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
            print(f"正在读取 {task['name']} 数据...")
            cursor.execute(task['sql'])
            all_rows = cursor.fetchall()
            data = [dict(r) for r in all_rows]

            for i in tqdm(range(0, len(data), BATCH_SIZE), desc=f"导入 {task['name']} 节点"):
                batch = data[i: i + BATCH_SIZE]
                with self.driver.session(database=self.db_name) as session:
                    session.run(task['cypher'], {"data": batch})
                if SLEEP_TIME > 0: time.sleep(SLEEP_TIME)

    def load_relationships(self):
        """构建 AUTHORED, AFFILIATED_WITH, PUBLISHED_IN 等关系"""
        cursor = self.conn.cursor()

        # 1. 构建 AUTHORED 署名关系 (含时序衰减权重)
        print("构建 AUTHORED 署名关系...")
        rels = cursor.execute("""
            SELECT a.author_id, a.work_id, w.year, a.pos_index, a.is_corresponding
            FROM authorships a JOIN works w ON a.work_id = w.work_id
        """).fetchall()

        batch_data = []
        cypher_authored = """
        UNWIND $data AS row
        MATCH (a:Author {id: row.aid}), (w:Work {id: row.wid})
        MERGE (a)-[r:AUTHORED]->(w)
        SET r.weight = row.weight
        """
        for r in tqdm(rels, desc="计算权重"):
            # 指数衰减函数: $e^{-0.1 \cdot \Delta t}$
            delta_t = max(0, self.reference_year - (r['year'] if r['year'] else self.reference_year))
            time_decay = np.exp(-0.1 * delta_t)
            # 加权逻辑：第一作者或通讯作者权重更高
            pos_weight = 1.2 if (r['pos_index'] == 0 or r['is_corresponding'] == 1) else 1.0

            batch_data.append({
                "aid": r['author_id'], "wid": r['work_id'],
                "weight": float(time_decay * pos_weight)
            })
            if len(batch_data) >= BATCH_SIZE:
                self._execute_cypher(cypher_authored, batch_data)
                batch_data = []
        if batch_data: self._execute_cypher(cypher_authored, batch_data)

        # 2. 校准：构建 AFFILIATED_WITH 关系 (作者-机构)
        print("构建 AFFILIATED_WITH 隶属关系...")
        # 使用字段 last_known_institution_id
        aff_data = cursor.execute("""
            SELECT author_id, last_known_institution_id AS inst_id 
            FROM authors 
            WHERE last_known_institution_id IS NOT NULL
        """).fetchall()
        cypher_aff = """
        UNWIND $data AS row
        MATCH (a:Author {id: row.author_id}), (i:Institution {id: row.inst_id})
        MERGE (a)-[:AFFILIATED_WITH]->(i)
        """
        self._batch_process(aff_data, cypher_aff, "隶属关系")

        # 3. 校准：构建 PUBLISHED_IN 关系 (作品-来源)
        print("构建 PUBLISHED_IN 发表关系...")
        # 从 authorships 提取作品与渠道的对应关系
        pub_data = cursor.execute("""
            SELECT DISTINCT work_id, source_id 
            FROM authorships 
            WHERE source_id IS NOT NULL
        """).fetchall()
        cypher_pub = """
        UNWIND $data AS row
        MATCH (w:Work {id: row.work_id}), (s:Source {id: row.source_id})
        MERGE (w)-[:PUBLISHED_IN]->(s)
        """
        self._batch_process(pub_data, cypher_pub, "发表关系")

    def inject_normalized_features(self):
        """注入归一化特征，用于 KGAT-AX 的全息嵌入层 """
        if not os.path.exists(FEATURE_JSON):
            print(f"[!] 警告: 未找到 {FEATURE_JSON}，请先运行 build_feature_index.py")
            return

        print("正在注入归一化特征指标...")
        with open(FEATURE_JSON, 'r', encoding='utf-8') as f:
            features = json.load(f)

        # 注入学者特征 (norm_h, norm_citations)
        author_cypher = """
        UNWIND $data AS row 
        MATCH (a:Author {id: row.id}) 
        SET a.norm_h = row.h_index, a.norm_citations = row.cited_by_count
        """
        self._batch_process_features(features['author'], author_cypher, "作者特征")

        # 注入机构特征 (norm_works, norm_citations)
        inst_cypher = """
        UNWIND $data AS row 
        MATCH (i:Institution {id: row.id}) 
        SET i.norm_works = row.works_count, i.norm_citations = row.cited_by_count
        """
        self._batch_process_features(features['institution'], inst_cypher, "机构特征")

    def build_semantic_bridge(self, min_edges=3):
        """构建岗位技能 -> 学术词汇的语义桥梁 """
        print(f"--- 启动语义桥梁构建 | 目标：每个 Industry 节点连接 {min_edges} 个相似词 ---")
        try:
            index = faiss.read_index(f"{INDEX_DIR}/vocabulary.faiss")
            with open(f"{INDEX_DIR}/vocabulary_mapping.json", 'r', encoding='utf-8') as f:
                all_ids = json.load(f)
        except Exception as e:
            print(f"[!] 索引加载失败: {e}"); return

        skills = self.conn.cursor().execute("SELECT id, term FROM vocabulary WHERE entity_type = 'industry'").fetchall()
        bridge_batch = []
        cypher_bridge = """
        UNWIND $data AS row
        MATCH (v1:Vocabulary {id: row.from_id}), (v2:Vocabulary {id: row.to_id})
        MERGE (v1)-[r:SIMILAR_TO]->(v2)
        SET r.score = row.score
        """
        for s in tqdm(skills, desc="向量空间检索"):
            s_id_str = str(s['id'])
            if s_id_str not in all_ids: continue
            try:
                idx_in_faiss = all_ids.index(s_id_str)
                vector = index.reconstruct(idx_in_faiss).reshape(1, -1)
                D, I = index.search(vector, 15)
                valid_count = 0
                for dist, neighbor_idx in zip(D[0], I[0]):
                    neighbor_id_str = all_ids[neighbor_idx]
                    if neighbor_id_str == s_id_str: continue
                    bridge_batch.append({"from_id": int(s_id_str), "to_id": int(neighbor_id_str), "score": float(dist)})
                    valid_count += 1
                    if valid_count >= min_edges: break
                if len(bridge_batch) >= BATCH_SIZE:
                    self._execute_cypher(cypher_bridge, bridge_batch)
                    bridge_batch = []
            except: continue
        if bridge_batch: self._execute_cypher(cypher_bridge, bridge_batch)

    def _execute_cypher(self, query, data):
        with self.driver.session(database=self.db_name) as session:
            session.run(query, {"data": data})
        if SLEEP_TIME > 0: time.sleep(SLEEP_TIME)

    def _batch_process(self, rows, cypher, desc):
        data = [dict(r) for r in rows]
        for i in tqdm(range(0, len(data), BATCH_SIZE), desc=desc):
            self._execute_cypher(cypher, data[i:i + BATCH_SIZE])

    def _batch_process_features(self, feature_map, cypher, desc):
        batch = []
        for node_id, attrs in tqdm(feature_map.items(), desc=desc):
            attrs['id'] = node_id
            batch.append(attrs)
            if len(batch) >= BATCH_SIZE:
                self._execute_cypher(cypher, batch)
                batch = []
        if batch: self._execute_cypher(cypher, batch)

    def run_pipeline(self):
        start_time = time.time()
        self.init_db()
        self.load_nodes()
        self.load_relationships()
        self.inject_normalized_features()
        self.build_semantic_bridge()
        print(f"\n--- 任务完成! 总耗时: {item_time(time.time() - start_time)} ---")

def item_time(seconds):
    m, s = divmod(seconds, 60); h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    try:
        builder.run_pipeline()
    finally:
        builder.close()