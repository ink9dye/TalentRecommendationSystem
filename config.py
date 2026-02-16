import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. 基础配置信息 ---
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Aa1278389701"
NEO4J_DATABASE = "talent-graph"

# --- 2. 核心数据目录 ---
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- 3. 索引存储目录 ---
INDEX_DIR = os.path.join(DATA_DIR, "build_index")
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)

# --- 4. 数据库路径 ---
DB_PATH = os.path.join(DATA_DIR, "academic_dataset_v5.db")

# --- 5. 三大向量索引具体路径 ---
VOCAB_INDEX_PATH = os.path.join(INDEX_DIR, "vocabulary.faiss")
VOCAB_MAP_PATH = os.path.join(INDEX_DIR, "vocabulary_mapping.json")

ABSTRACT_INDEX_PATH = os.path.join(INDEX_DIR, "abstract.faiss")
ABSTRACT_MAP_PATH = os.path.join(INDEX_DIR, "abstract_mapping.json")

JOB_INDEX_PATH = os.path.join(INDEX_DIR, "job_description.faiss")
JOB_MAP_PATH = os.path.join(INDEX_DIR, "job_description_mapping.json")

# --- 6. 其他索引 ---
FEATURE_INDEX_PATH = os.path.join(INDEX_DIR, "feature_index.json")
COLLAB_DB_PATH = os.path.join(INDEX_DIR, "scholar_collaboration.db")

# --- 7. SBERT 模型本地存放路径 ---
SBERT_DIR = os.path.join(DATA_DIR, "build_sbert")
if not os.path.exists(SBERT_DIR):
    os.makedirs(SBERT_DIR)

# --- 8. 知识图谱构建 SQL 语句 ---
SQL_QUERIES = {
    # 基础实体同步：确保包含通用的 id (用于 MERGE) 和业务特有的 aid, wid 等
    "SYNC_AUTHORS": "SELECT author_id as id, author_id as aid, name, h_index, works_count, cited_by_count as citations, last_updated FROM authors WHERE last_updated > ? ORDER BY last_updated ASC",
    "SYNC_INSTITUTIONS": "SELECT inst_id as id, inst_id as iid, name, works_count, cited_by_count as citations, last_updated FROM institutions WHERE last_updated > ? ORDER BY last_updated ASC",
    "SYNC_SOURCES": "SELECT source_id as id, source_id as sid, display_name as name, type, works_count, cited_by_count as citations, last_updated FROM sources WHERE last_updated > ? ORDER BY last_updated ASC",
    "SYNC_WORKS": "SELECT work_id as id, work_id as wid, title as name, title, year, citation_count as citations, concepts_text, keywords_text FROM works WHERE year > ? ORDER BY year ASC",
    "SYNC_JOBS": "SELECT securityId as id, securityId as jid, job_name as name, skills, description, crawl_time FROM jobs WHERE crawl_time > ? ORDER BY crawl_time ASC",

    # 修正：vocabulary 使用 voc_id 作为主键和游标
    "GET_ALL_VOCAB": "SELECT voc_id as id, voc_id, voc_id as vid, term as name, term, entity_type FROM vocabulary WHERE voc_id > ? ORDER BY voc_id ASC",

    "SYNC_JOB_SKILLS": "SELECT securityId as jid, skills, crawl_time FROM jobs WHERE crawl_time > ? AND skills IS NOT NULL ORDER BY crawl_time ASC",

    # 修正：authorships 使用 ship_id 替代原本的 id 字段进行游标过滤
    "SYNC_AUTHORED_TOPOLOGY": """
                              SELECT aship.ship_id   as sync_id,
                                     aship.author_id as aid,
                                     aship.work_id   as wid,
                                     aship.inst_id   as iid,
                                     aship.source_id as sid,
                                     aship.pos_index,
                                     aship.is_corresponding,
                                     aship.is_alphabetical,
                                     w.year
                              FROM authorships aship
                                       JOIN works w ON aship.work_id = w.work_id
                              WHERE aship.ship_id > ?
                              ORDER BY aship.ship_id ASC
                              """
}

# --- 9. Neo4j Cypher 模板 ---
CYPHER_TEMPLATES = {
    "INIT_SCHEMA": [
        # 全局唯一约束
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (w:Work) REQUIRE w.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Vocabulary) REQUIRE v.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE",

        # 原生业务 ID 索引：支持跨库反查
        "CREATE INDEX IF NOT EXISTS FOR (v:Vocabulary) ON (v.voc_id)",
        "CREATE INDEX IF NOT EXISTS FOR (a:Author) ON (a.aid)",
        "CREATE INDEX IF NOT EXISTS FOR (w:Work) ON (w.wid)",
        "CREATE INDEX IF NOT EXISTS FOR (i:Institution) ON (i.iid)",
        "CREATE INDEX IF NOT EXISTS FOR (s:Source) ON (s.sid)",
        "CREATE INDEX IF NOT EXISTS FOR (j:Job) ON (j.jid)",
        "CREATE INDEX IF NOT EXISTS FOR (v:Vocabulary) ON (v.term)",
        "CREATE INDEX IF NOT EXISTS FOR (a:Author) ON (a.h_index)"
    ],

    # 节点合并：确保原生 ID (wid, aid, voc_id 等) 被写入属性
    "MERGE_WORK": "UNWIND $data AS row MERGE (n:Work {id: row.id}) SET n.wid = row.wid, n.title = row.title, n.name = row.name, n.year = row.year, n.citations = row.citations",
    "MERGE_AUTHOR": "UNWIND $data AS row MERGE (n:Author {id: row.id}) SET n.aid = row.aid, n.name = row.name, n.h_index = row.h_index, n.works_count = row.works_count, n.citations = row.citations",
    "MERGE_INSTITUTION": "UNWIND $data AS row MERGE (i:Institution {id: row.id}) SET i.iid = row.iid, i.name = row.name, i.works_count = row.works_count, i.citations = row.citations",
    "MERGE_SOURCE": "UNWIND $data AS row MERGE (s:Source {id: row.id}) SET s.sid = row.sid, s.name = row.name, s.type = row.type, s.works_count = row.works_count, s.citations = row.citations",
    "MERGE_JOB": "UNWIND $data AS row MERGE (j:Job {id: row.id}) SET j.jid = row.jid, j.name = row.name, j.skills = row.skills, j.description = row.description",

    # 存储 voc_id 原生属性
    "MERGE_VOCAB": "UNWIND $data AS row MERGE (v:Vocabulary {id: row.id}) SET v.voc_id = row.voc_id, v.vid = row.vid, v.term = toLower(row.term), v.type = row.entity_type",

    # 复杂关系构建：对齐 pos_index 属性名
    "LINK_AUTHORED_COMPLEX": """
        UNWIND $data AS row
        MATCH (a:Author {id: row.aid}), (w:Work {id: row.wid})
        MERGE (a)-[r:AUTHORED]->(w)
        SET r.pos_index = row.pos_index, 
            r.pub_year = row.year,
            r.is_corresponding = row.is_corresponding
        WITH a, w, row
        WHERE row.iid IS NOT NULL
        MATCH (i:Institution {id: row.iid})
        MERGE (w)-[:PRODUCED_BY]->(i)
        WITH w, row
        WHERE row.sid IS NOT NULL
        MATCH (s:Source {id: row.sid})
        MERGE (w)-[:PUBLISHED_IN]->(s)
    """,
    "LINK_SIMILAR": "UNWIND $data AS row MATCH (v1:Vocabulary {id: row.f}), (v2:Vocabulary {id: row.t}) MERGE (v1)-[r:SIMILAR_TO]->(v2) SET r.score = row.s",
    "LINK_WORK_VOCAB": "UNWIND $data AS row MATCH (w:Work {id: row.wid}), (v:Vocabulary {term: row.term}) MERGE (w)-[:HAS_TOPIC]->(v)",
    "LINK_JOB_VOCAB": "UNWIND $data AS row MATCH (j:Job {id: row.jid}), (v:Vocabulary {term: row.term}) MERGE (j)-[:REQUIRE_SKILL]->(v)"
}

# --- 10. 整合配置字典 ---
CONFIG_DICT = {
    "DB_PATH": DB_PATH,
    "INDEX_DIR": INDEX_DIR,
    "VOCAB_INDEX_PATH": VOCAB_INDEX_PATH,
    "VOCAB_MAP_PATH": VOCAB_MAP_PATH,
    "NEO4J_URI": NEO4J_URI,
    "NEO4J_USER": NEO4J_USER,
    "NEO4J_PASSWORD": NEO4J_PASSWORD,
    "NEO4J_DATABASE": NEO4J_DATABASE,
    "BATCH_SIZE": 5000,
    "SBERT_DIR": SBERT_DIR
}

# --- 11. SQLite 索引初始化脚本 ---
# 重点：此处新增了针对“慢得绝望”问题的覆盖索引
SQL_INIT_SCRIPTS = [
    # 基础同步加速索引
    "CREATE INDEX IF NOT EXISTS idx_author_updated ON authors(last_updated)",
    "CREATE INDEX IF NOT EXISTS idx_work_year ON works(year)",
    "CREATE INDEX IF NOT EXISTS idx_inst_updated ON institutions(last_updated)",
    "CREATE INDEX IF NOT EXISTS idx_source_updated ON sources(last_updated)",
    "CREATE INDEX IF NOT EXISTS idx_job_crawl ON jobs(crawl_time)",
    "CREATE INDEX IF NOT EXISTS idx_aship_id ON authorships(ship_id)",
    "CREATE INDEX IF NOT EXISTS idx_vocab_voc_id ON vocabulary(voc_id)",

    # 关键覆盖索引 (Covering Index)
    # 1. 特征覆盖索引：解决生成训练集和精排特征加载时的磁盘 I/O 瓶颈
    "CREATE INDEX IF NOT EXISTS idx_author_ax_covering ON authors(author_id, h_index, cited_by_count, works_count)",

    # 2. 拓扑覆盖索引：解决向量路从论文 ID 找作者时的全表扫描瓶颈
    "CREATE INDEX IF NOT EXISTS idx_aship_work_author ON authorships(work_id, author_id, inst_id, source_id)",

    # 3. 反向拓扑覆盖索引：解决协同路计算和图谱全量导出的性能瓶颈
    "CREATE INDEX IF NOT EXISTS idx_aship_author_work ON authorships(author_id, work_id)"
]

# --- 12. KGATAX 训练数据的存放路径 ---
KGATAX_TRAIN_DATA_DIR = os.path.join(DATA_DIR, "kgatax_train_data")
if not os.path.exists(KGATAX_TRAIN_DATA_DIR):
    os.makedirs(KGATAX_TRAIN_DATA_DIR)