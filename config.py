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

# --- 5. 向量索引具体路径 ---
VOCAB_INDEX_PATH = os.path.join(INDEX_DIR, "vocabulary.faiss")
VOCAB_MAP_PATH = os.path.join(INDEX_DIR, "vocabulary_mapping.json")

ABSTRACT_INDEX_PATH = os.path.join(INDEX_DIR, "abstract.faiss")
ABSTRACT_MAP_PATH = os.path.join(INDEX_DIR, "abstract_mapping.json")

JOB_INDEX_PATH = os.path.join(INDEX_DIR, "job_description.faiss")
JOB_MAP_PATH = os.path.join(INDEX_DIR, "job_description_mapping.json")



# --- 6. 其他索引 ---
FEATURE_INDEX_PATH = os.path.join(INDEX_DIR, "feature_index.json")
COLLAB_DB_PATH = os.path.join(INDEX_DIR, "scholar_collaboration.db")
VOCAB_STATS_DB_PATH = os.path.join(INDEX_DIR, 'vocab_stats.db')

# --- 7. SBERT 模型本地存放路径 ---
SBERT_MODEL_NAME = 'Alibaba-NLP/gte-multilingual-base'

# 核心修改：在 build_sbert 下增加一个以模型名命名的子文件夹
# 使用 .split('/')[-1] 得到 'gte-multilingual-base'
SBERT_DIR = os.path.join(DATA_DIR, "build_sbert", SBERT_MODEL_NAME.split('/')[-1])

# 确保多层目录都能自动创建
if not os.path.exists(SBERT_DIR):
    os.makedirs(SBERT_DIR)


# --- 8. 知识图谱构建 SQL 语句 ---
SQL_QUERIES = {
    # 基础实体同步
    "SYNC_AUTHORS": "SELECT author_id as id, name, h_index, works_count, cited_by_count as citations, last_updated FROM authors WHERE last_updated > ? ORDER BY last_updated ASC",
    "SYNC_INSTITUTIONS": "SELECT inst_id as id, name, works_count, cited_by_count as citations, last_updated FROM institutions WHERE last_updated > ? ORDER BY last_updated ASC",
    "SYNC_SOURCES": "SELECT source_id as id, display_name as name, type, works_count, cited_by_count as citations, last_updated FROM sources WHERE last_updated > ? ORDER BY last_updated ASC",
    "SYNC_WORKS": "SELECT work_id as id, title as name, title, year, citation_count as citations, concepts_text, keywords_text, domain_ids FROM works WHERE year > ? ORDER BY year ASC",
    "SYNC_JOBS": "SELECT securityId as id, job_name as name, skills, description, crawl_time, domain_ids FROM jobs WHERE crawl_time > ? ORDER BY crawl_time ASC",
    "GET_ALL_VOCAB": "SELECT voc_id as id, term as name, term, entity_type FROM vocabulary WHERE voc_id > ? ORDER BY voc_id ASC",
    "SYNC_JOB_SKILLS": "SELECT securityId as id, skills, crawl_time FROM jobs WHERE crawl_time > ? AND skills IS NOT NULL ORDER BY crawl_time ASC",

    # [新增] 用于标题回溯打标的元数据查询 (包含 title 字段)
    "GET_WORK_METADATA_FOR_TAGGING": "SELECT work_id as id, title, concepts_text, keywords_text FROM works",

    # [新增] 共现统计查询：从临时映射表中统计在同一篇 Work 下出现的词对频率
    "GET_VOCAB_CO_OCCURRENCE": """
                               SELECT a.term as term_a, b.term as term_b, COUNT(a.work_id) as freq
                               FROM work_terms_temp a
                                        JOIN work_terms_temp b ON a.work_id = b.work_id
                               WHERE a.term < b.term -- 避免重复计算 (A,B) 与 (B,A)，且防止自关联
                               GROUP BY a.term, b.term
                               HAVING freq > 1 -- 过滤仅共现 1 次的微弱噪声
                               """,

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
        "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (w:Work) REQUIRE w.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Vocabulary) REQUIRE v.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE",
        "CREATE INDEX IF NOT EXISTS FOR (w:Work) ON (w.domain_ids)",
        "CREATE INDEX IF NOT EXISTS FOR (j:Job) ON (j.domain_ids)",
        "CREATE INDEX IF NOT EXISTS FOR (v:Vocabulary) ON (v.term)",
        "CREATE INDEX IF NOT EXISTS FOR (a:Author) ON (a.h_index)",
        # [新增] 为共现关系建立索引，加速后续路径计算
        "CREATE INDEX IF NOT EXISTS FOR ()-[r:CO_OCCURRED_WITH]-() ON (r.weight)"
    ],

    "MERGE_WORK": "UNWIND $data AS row MERGE (n:Work {id: row.id}) SET n.title = row.title, n.name = row.name, n.year = row.year, n.citations = row.citations, n.domain_ids = row.domain_ids",
    "MERGE_AUTHOR": "UNWIND $data AS row MERGE (n:Author {id: row.id}) SET n.name = row.name, n.h_index = row.h_index, n.works_count = row.works_count, n.citations = row.citations",
    "MERGE_INSTITUTION": "UNWIND $data AS row MERGE (i:Institution {id: row.id}) SET i.name = row.name, i.works_count = row.works_count, i.citations = row.citations",
    "MERGE_SOURCE": "UNWIND $data AS row MERGE (s:Source {id: row.id}) SET s.name = row.name, s.type = row.type, s.works_count = row.works_count, s.citations = row.citations",
    "MERGE_JOB": "UNWIND $data AS row MERGE (j:Job {id: row.id}) SET j.name = row.name, j.skills = row.skills, j.description = row.description, j.domain_ids = row.domain_ids",
    "MERGE_VOCAB": "UNWIND $data AS row MERGE (v:Vocabulary {id: row.id}) SET v.term = toLower(row.term), v.name = toLower(row.term), v.type = row.entity_type",

    "LINK_AUTHORED_COMPLEX": """
        UNWIND $data AS row
        MATCH (a:Author {id: row.aid}), (w:Work {id: row.wid})
        MERGE (a)-[r:AUTHORED]->(w)
        SET r.pos_index = row.pos_index, 
            r.pub_year = row.year,
            r.is_corresponding = row.is_corresponding,
            r.pos_weight = row.pos_w
        WITH a, w, row
        WHERE row.iid IS NOT NULL
        MATCH (i:Institution {id: row.iid})
        MERGE (w)-[:PRODUCED_BY]->(i)
        WITH w, row
        WHERE row.sid IS NOT NULL
        MATCH (s:Source {id: row.sid})
        MERGE (w)-[:PUBLISHED_IN]->(s)
    """,

    # [核心修改] LINK_WORK_VOCAB 保持不变，但其输入数据将包含通过标题回溯扫描出的 term
    "LINK_WORK_VOCAB": "UNWIND $data AS row MATCH (w:Work {id: row.id}), (v:Vocabulary {term: row.term}) MERGE (w)-[:HAS_TOPIC]->(v)",

    # [新增] 共现权重模板：建立学术词汇间的横向关联
    "MERGE_CO_OCCURRENCE": """
        UNWIND $data AS row
        MATCH (v1:Vocabulary {term: row.term_a}), (v2:Vocabulary {term: row.term_b})
        MERGE (v1)-[r:CO_OCCURRED_WITH]-(v2)
        SET r.weight = row.freq
    """,

    "LINK_SIMILAR": "UNWIND $data AS row MATCH (v1:Vocabulary {id: row.f}), (v2:Vocabulary {id: row.t}) MERGE (v1)-[r:SIMILAR_TO]->(v2) SET r.score = row.s",
    "LINK_JOB_VOCAB": "UNWIND $data AS row MATCH (j:Job {id: row.id}), (v:Vocabulary {term: row.term}) MERGE (j)-[:REQUIRE_SKILL]->(v)"
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
    "BATCH_SIZE": 1000,
    "SBERT_DIR": SBERT_DIR,
    "SBERT_MODEL_NAME": SBERT_MODEL_NAME
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


# --- 13. 业务领域映射表 (新增) ---
DOMAIN_MAP = {
    "1": "计算机科学", "2": "医学", "3": "政治学", "4": "工程学", "5": "物理学",
    "6": "材料科学", "7": "生物学", "8": "地理学", "9": "化学", "10": "商学",
    "11": "社会学", "12": "哲学", "13": "环境科学", "14": "数学", "15": "心理学",
    "16": "地质学", "17": "经济学"
}
# 反向查找表：支持“计算机科学” -> "1"
NAME_TO_DOMAIN_ID = {v: k for k, v in DOMAIN_MAP.items()}