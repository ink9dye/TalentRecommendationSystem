import sqlite3
import json
import numpy as np
import faiss
import time
import os
from neo4j import GraphDatabase
from datetime import datetime
from tqdm import tqdm

# ==========================================
# 模块说明：知识图谱拓扑构建器 (Topology Builder)
# ==========================================
# 核心职责：
# 1. 建立实体间的连接（Relation），而非计算最终分数。
# 2. 注入“原子权重”（Atomic Weights）：如署名顺位、发表年份。
# 3. 这种设计允许我们在查询（Query Time）时，根据当前时间动态计算“时序衰减”，
#    而不是在建图时写死一个静态分数。
# ==========================================

# --- 核心配置 ---
# SQLite: 存储元数据（论文标题、作者名、摘要等具体文本），作为“事实源头”
DB_PATH = r"E:\PythonProject\TalentRecommendationSystem\data\academic_dataset_v5.db"
# FAISS/Index: 存储向量索引和预计算的特征，辅助语义对齐
INDEX_DIR = r"E:\PythonProject\TalentRecommendationSystem\data\build_index"

# Neo4j: 仅存储“骨架”（拓扑结构），用于路径游走和关系发现
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Aa1278389701"
NEO4J_DATABASE = "talent-graph"

# 性能调优：
# BATCH_SIZE=2000：Neo4j 官方推荐的写入批次大小，平衡事务内存开销与网络 RTT
BATCH_SIZE = 2000
SLEEP_TIME = 0.05


class KnowledgeGraphBuilder:
    def __init__(self):
        """
        [初始化] KG 构建器
        --------------------------------
        作用：
        1. 建立与 Neo4j 的长连接，配置 1 小时的连接寿命以防止大数据量导入时断连。
        2. 连接 SQLite 数据源。
        """
        # 配置长连接与超时，防止大规模写入中断
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=3600  # 保持连接存活 1 小时
        )
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row  # 允许通过列名访问 SQLite 结果
        self.db_name = NEO4J_DATABASE
        print(f"--- 拓扑驱动型 KG 构建启动 | 时间: {datetime.now()} ---")

    def close(self):
        """[资源释放] 关闭数据库连接，防止句柄泄露"""
        self.conn.close()
        self.driver.close()

    def init_db(self):
        """
        [Schema 定义] 初始化约束与索引
        --------------------------------
        作用：
        1. 数据完整性：创建 UNIQUE CONSTRAINT，防止同一个作者或论文被重复创建。
        2. 查询加速：为 name, year, term 创建索引。
           - 特别注意：为 Work(year) 建索引是为了支持论文中的 '时序衰减' 算法，
             让我们能快速过滤出 '近5年' 或 '近10年' 的论文。
        """
        print("正在初始化数据库约束与索引...")
        commands = [
            # --- 1. 唯一性约束 (Schema Consistency) ---
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (w:Work) REQUIRE w.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Vocabulary) REQUIRE v.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE",

            # --- 2. 性能索引 (Query Optimization) ---
            "CREATE INDEX IF NOT EXISTS FOR (a:Author) ON (a.name)",
            # 关键索引：后续计算 exp(-0.1 * (Now - pub_year)) 时需要用到 year
            "CREATE INDEX IF NOT EXISTS FOR (w:Work) ON (w.year)",
            "CREATE INDEX IF NOT EXISTS FOR (v:Vocabulary) ON (v.term)"
        ]
        with self.driver.session(database=self.db_name) as session:
            for cmd in commands:
                session.run(cmd)

    def load_nodes(self):
        """
        [ETL - 节点层] 导入六类核心实体
        --------------------------------
        作用：从 SQLite 读取清洗好的数据，写入 Neo4j 节点。
        逻辑：
        - 仅写入用于“图搜索”的轻量级属性（如 id, name, h-index）。
        - 复杂的长文本（如 full_text）留在 SQLite 中，不在图里存，节省内存。
        """
        cursor = self.conn.cursor()

        # 定义节点加载任务清单
        tasks = [
            {
                "name": "Author",  # 学者节点
                "sql": "SELECT author_id, name, h_index, cited_by_count, works_count FROM authors",
                "cypher": "UNWIND $data AS row MERGE (a:Author {id: row.author_id}) SET a.name = row.name, a.h_index = row.h_index, a.citations = row.cited_by_count, a.works_count = row.works_count"
            },
            {
                "name": "Work",  # 论文节点
                "sql": "SELECT work_id, title, year, citation_count FROM works",
                "cypher": "UNWIND $data AS row MERGE (w:Work {id: row.work_id}) SET w.title = row.title, w.year = row.year, w.citations = row.citation_count"
            },
            {
                "name": "Institution",  # 机构节点
                "sql": "SELECT inst_id, name, works_count, cited_by_count FROM institutions",
                "cypher": "UNWIND $data AS row MERGE (i:Institution {id: row.inst_id}) SET i.name = row.name, i.works_count = row.works_count, i.citations = row.cited_by_count"
            },
            {
                "name": "Source",  # 期刊/会议节点
                "sql": "SELECT source_id, display_name, type, works_count, cited_by_count FROM sources",
                "cypher": "UNWIND $data AS row MERGE (s:Source {id: row.source_id}) SET s.name = row.display_name, s.type = row.type, s.citations = row.cited_by_count"
            },
            {
                "name": "Vocabulary",  # 概念节点（学术Topic + 行业Skill）
                "sql": "SELECT id, term, entity_type FROM vocabulary",
                "cypher": "UNWIND $data AS row MERGE (v:Vocabulary {id: row.id}) SET v.term = row.term, v.type = row.entity_type"
            },
            {
                "name": "Job",  # 岗位节点
                "sql": "SELECT securityId, job_name, company, salary FROM jobs",
                "cypher": "UNWIND $data AS row MERGE (j:Job {id: row.securityId}) SET j.name = row.job_name, j.company = row.company, j.salary = row.salary"
            }
        ]

        # 循环执行每个任务
        for task in tasks:
            cursor.execute(task['sql'])
            rows = [dict(r) for r in cursor.fetchall()]
            # 使用通用批处理函数写入
            self._batch_process(rows, task['cypher'], f"导入节点: {task['name']}")

    def load_relationships(self):
        """
        [ETL - 关系层] 构建拓扑关联 (代码核心)
        --------------------------------
        作用：定义图谱的“骨架”，并注入算法所需的“原子因子”。
        关键改进：
        1. 不计算最终分数，只存 `pos_weight` 和 `pub_year`。
        2. 让 Neo4j 回归纯粹的拓扑存储，计算逻辑下放给查询语句。
        """
        cursor = self.conn.cursor()

        # ========================================================
        # 1. AUTHORED (作者 -> 论文)
        # 核心逻辑：这是计算“协同强度”的基础。
        # ========================================================
        print("构建核心拓扑: AUTHORED (注入 pos_weight 和 pub_year)...")

        # SQL Join：一次性获取 关系+权重+年份，避免后续查询 N+1 问题
        rels = cursor.execute("""
                              SELECT a.author_id, a.work_id, a.pos_index, a.is_corresponding,a.is_alphabetical,w.year
                              FROM authorships a
                                       JOIN works w ON a.work_id = w.work_id
                              """).fetchall()

        # Cypher 逻辑：
        # - pos_weight: 署名权重。一作/通讯权重高(1.2)，其他普通(1.0)。
        # - pub_year:   论文年份。用于后续查询时动态计算衰减权重。
        cypher_authored = """
        UNWIND $data AS row
        MATCH (a:Author {id: row.aid}), (w:Work {id: row.wid})
        MERGE (a)-[r:AUTHORED]->(w)
        SET r.pos_weight = row.pos_w,  
            r.pub_year = row.year      
        """

        batch = []
        for r in tqdm(rels, desc="处理协作边"):
            # 1. 初始化基础权重
            base_w = 1.0

            # 2. 第一作者判定逻辑 (pos_index=1 且非字母序)
            is_first_author = (r['pos_index'] == 1 and r['is_alphabetical'] == 0)

            # 3. 通讯作者判定逻辑
            is_corresponding = (r['is_corresponding'] == 1)

            # 4. 权重叠加逻辑：
            # 如果是第一作者，加 0.2
            if is_first_author:
                base_w += 0.2

            # 如果是通讯作者，再加 0.2
            if is_corresponding:
                base_w += 0.2

            # 最终结果：
            # 普通作者 = 1.0
            # 仅第一作者 = 1.2
            # 仅通讯作者 = 1.2
            # 第一作者且通讯作者 = 1.4
            pos_w = base_w

            # 年份处理
            year = int(r['year']) if r['year'] else 2000

            batch.append({
                "aid": r['author_id'], "wid": r['work_id'],
                "pos_w": float(pos_w), "year": year
            })

            if len(batch) >= BATCH_SIZE:
                self._execute_cypher(cypher_authored, batch)
                batch = []
        if batch: self._execute_cypher(cypher_authored, batch)

        # ========================================================
        # 2. TAGGED (论文 -> 学术概念)
        # 核心逻辑：论文中包含了 concepts_text，将其映射到 Vocabulary 节点
        # ========================================================
        print("构建语义关联: TAGGED...")
        # 模糊匹配：concepts_text LIKE '%term%'
        tag_data = cursor.execute("""
                                  SELECT w.work_id, v.id as vocab_id
                                  FROM works w
                                           JOIN vocabulary v ON w.concepts_text LIKE '%' || v.term || '%'
                                  WHERE v.entity_type = 'topic'
                                  """).fetchall()
        self._batch_process(tag_data,
                            "UNWIND $data AS row MATCH (w:Work {id: row.wid}), (v:Vocabulary {id: row.vid}) MERGE (w)-[:TAGGED]->(v)",
                            "建立 TAGGED 关系", key_map={'work_id': 'wid', 'vocab_id': 'vid'})

        # ========================================================
        # 3. AFFILIATED_WITH (作者 -> 机构)
        # ========================================================
        aff_data = cursor.execute(
            "SELECT author_id, last_known_institution_id as inst_id FROM authors WHERE last_known_institution_id IS NOT NULL").fetchall()
        self._batch_process(aff_data,
                            "UNWIND $data AS row MATCH (a:Author {id: row.aid}), (i:Institution {id: row.iid}) MERGE (a)-[:AFFILIATED_WITH]->(i)",
                            "建立 AFFILIATED_WITH 关系", key_map={'author_id': 'aid', 'inst_id': 'iid'})

        # ========================================================
        # 4. PUBLISHED_IN (论文 -> 期刊)
        # ========================================================
        source_rels = cursor.execute("SELECT work_id, source_id FROM works WHERE source_id IS NOT NULL").fetchall()
        self._batch_process(source_rels,
                            "UNWIND $data AS row MATCH (w:Work {id: row.wid}), (s:Source {id: row.sid}) MERGE (w)-[:PUBLISHED_IN]->(s)",
                            "建立 PUBLISHED_IN 关系", key_map={'work_id': 'wid', 'source_id': 'sid'})

        # ========================================================
        # 5. REQUIRES (岗位 -> 行业技能)
        # 核心逻辑：这是连接“企业侧”与“学术侧”的入口。
        # 解析 Job 描述中的 "Java, AI" 等字符串，连接到 Vocabulary 节点。
        # ========================================================
        print("构建需求关联: REQUIRES (Job -> Vocabulary)...")
        jobs = cursor.execute("SELECT securityId, skills FROM jobs WHERE skills IS NOT NULL").fetchall()

        # 内存优化：预加载所有 industry 词汇到字典，避免循环内频繁查询 SQL
        vocab_map = {row['term'].lower(): row['id'] for row in
                     cursor.execute("SELECT term, id FROM vocabulary WHERE entity_type='industry'")}

        job_batch = []
        cypher_req = "UNWIND $data AS row MATCH (j:Job {id: row.jid}), (v:Vocabulary {id: row.vid}) MERGE (j)-[:REQUIRES]->(v)"

        for job in tqdm(jobs, desc="处理岗位技能"):
            if not job['skills']: continue
            # 清洗：兼容中英文逗号，去除空格，统一小写
            skills = [s.strip().lower() for s in job['skills'].replace('，', ',').split(',')]
            for skill in skills:
                if skill in vocab_map:
                    # 只有当技能在 Vocabulary 表中存在时才建立关系
                    job_batch.append({"jid": job['securityId'], "vid": vocab_map[skill]})

            if len(job_batch) >= BATCH_SIZE:
                self._execute_cypher(cypher_req, job_batch)
                job_batch = []
        if job_batch: self._execute_cypher(cypher_req, job_batch)

    def build_semantic_bridge(self, threshold=0.75):
        """
        [跨域对齐] 构建语义桥梁 SIMILAR_TO
        --------------------------------
        作用：解决“中文应用词”与“英文学术词”不匹配的问题。
        逻辑：
        1. 读取 FAISS 向量索引（里面存了所有词汇的 SBERT 向量）。
        2. 对每个 Industry 词（如 '人工智能'），在向量空间中搜最相似的 Topic 词（如 'Artificial Intelligence'）。
        3. 如果相似度 > threshold，则在图谱中建立一条 [SIMILAR_TO] 边。
        结果：实现了 Job -> '人工智能' -> (SIMILAR_TO) -> 'AI' -> (TAGGED) -> Paper 的路径连通。
        """
        print(f"--- 建立词汇语义桥梁 (阈值: {threshold}) ---")
        try:
            # 加载预训练好的向量索引
            index = faiss.read_index(f"{INDEX_DIR}/vocabulary.faiss")
            with open(f"{INDEX_DIR}/vocabulary_mapping.json", 'r', encoding='utf-8') as f:
                all_ids = json.load(f)  # 向量索引对应的 ID 列表
        except Exception as e:
            print(f"跳过语义桥梁构建 (缺少索引文件): {e}")
            return

        # 仅获取“行业词”作为查询源
        vocab_nodes = self.conn.cursor().execute("SELECT id FROM vocabulary WHERE entity_type = 'industry'").fetchall()

        batch = []
        # Cypher: 建立相似度边，并存储相似度分数
        cypher = "UNWIND $data AS row MATCH (v1:Vocabulary {id: row.f}), (v2:Vocabulary {id: row.t}) MERGE (v1)-[r:SIMILAR_TO]->(v2) SET r.score = row.s"

        for v in tqdm(vocab_nodes, desc="向量空间语义对齐"):
            vid_str = str(v['id'])
            if vid_str not in all_ids: continue

            # 从 FAISS 恢复向量并搜索
            idx = all_ids.index(vid_str)
            vec = index.reconstruct(idx).reshape(1, -1)
            D, I = index.search(vec, 5)  # Top-5 近邻搜索

            for dist, n_idx in zip(D[0], I[0]):
                if dist < threshold: continue  # 过滤低置信度匹配
                nid_str = all_ids[n_idx]
                if nid_str == vid_str: continue  # 排除自己连自己

                # 记录：从 Industry Vocabulary -> Academic Vocabulary 的连接
                batch.append({"f": int(vid_str), "t": int(nid_str), "s": float(dist)})

            if len(batch) >= BATCH_SIZE:
                self._execute_cypher(cypher, batch)
                batch = []
        if batch: self._execute_cypher(cypher, batch)

    def _execute_cypher(self, query, data):
        """[工具函数] 执行 Cypher 语句，封装 Session 管理与异常捕获，增加了简单的重试逻辑"""
        for _ in range(3):  # 最多重试3次
            try:
                with self.driver.session(database=self.db_name) as session:
                    session.run(query, {"data": data})
                return  # 成功则退出重试
            except Exception as e:
                print(f"Cypher Error: {e}, 正在重试...")
                time.sleep(1)

    def _batch_process(self, rows, cypher, desc, key_map=None):
        """
        [工具函数] 通用批量处理器
        --------------------------------
        作用：将 SQL 查出的 rows 切分为小批次（Batch），逐批发送给 Neo4j。
        key_map: 可选参数，用于将 SQL 字段名映射为 Cypher 参数名（如 author_id -> aid）。
        """
        data = []
        for r in rows:
            item = dict(r)
            if key_map:
                # 字典推导式：只保留 key_map 中定义的字段，并重命名
                item = {key_map[k]: v for k, v in item.items() if k in key_map}
            data.append(item)

        # 使用 tqdm 显示进度条
        for i in tqdm(range(0, len(data), BATCH_SIZE), desc=desc):
            self._execute_cypher(cypher, data[i:i + BATCH_SIZE])

    def run_pipeline(self):
        """
        [主流水线] 顺序执行所有构建步骤
        """
        start_time = time.time()
        self.init_db()  # 1. 建立索引 (基础架构)
        self.load_nodes()  # 2. 建立节点 (导入实体)
        self.load_relationships()  # 3. 建立关系 (核心：注入权重与时序)
        self.build_semantic_bridge()  # 4. 建立桥梁 (跨域语义对齐)
        print(f"\n--- KG 构建完成! 总耗时: {int(time.time() - start_time)}s ---")


if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    try:
        builder.run_pipeline()
    finally:
        builder.close()  # 确保程序退出时关闭连接