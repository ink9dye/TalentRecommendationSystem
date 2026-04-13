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

# QueryEncoder 共振词表快照（随主库 mtime 失效；首次运行从 DB 构建并写入）
HARDCORE_LEXICON_SNAPSHOT_PATH = os.path.join(INDEX_DIR, "query_encoder_hardcore_lexicon.json")
# 标签路领域向量快照（随 DOMAIN_MAP / SBERT config.json 失效；首次在 LabelRecallPath 内编码并写入）
LABEL_DOMAIN_VECTORS_NPZ_PATH = os.path.join(INDEX_DIR, "label_domain_vectors.npz")
LABEL_DOMAIN_VECTORS_META_PATH = os.path.join(INDEX_DIR, "label_domain_vectors_meta.json")

# --- 6.1 标签路召回参数（学术词过滤与 SIMILAR_TO 约束）---
VOCAB_P95_PAPER_COUNT = 800   # 学术词 paper_count 上限，过滤泛词（如 machine learning / algorithm）
SIMILAR_TO_TOP_K = 3         # 每个锚点词最多扩展的学术词数量
SIMILAR_TO_MIN_SCORE = 0.65   # SIMILAR_TO 边权重下限，防止扩散过强

# --- 6.2 三层领域（Stage2B/Stage3 topic_align，见 README 修订版方案）---
TOPIC_ALIGN_SUBFIELD = 0.65   # 仅 subfield 对齐时的层级分
TOPIC_ALIGN_FIELD = 0.35     # 仅 field 对齐时的层级分
TOPIC_ALIGN_NONE = 0.10      # 层级无命中时的层级分（用于 primary 惩罚，不在此一票否决）
# primary 打分中 hierarchy_norm 的保守映射（仅 _hierarchy_norm 使用，不影响 topic_align）
HIERARCHY_NORM_TOPIC = 0.75    # topic 命中
HIERARCHY_NORM_SUBFIELD = 0.45 # subfield 命中
HIERARCHY_NORM_FIELD = 0.20    # field 命中
HIERARCHY_NORM_NONE = 0.05     # 无命中
# hierarchy 的 identity 联动上限：anchor_identity_score < 阈值时，hierarchy 乘折扣再进 primary（避免 propulsion/simula/control flow 等 topic 命中但本义不对的词拿满 hier）
HIERARCHY_IDENTITY_THRESHOLD = 0.50   # identity 低于此视为“本义不对齐”
HIERARCHY_IDENTITY_DISCOUNT = 0.50    # 上述情况下 hierarchy 乘数（0.5 = 最多只当一半用）
# 补充锚点来源权重：JD 向量补充的锚点在 primary 打分时乘此值，避免 Robotics/Robot control 把 Telerobotics 等顶到最前
ANCHOR_SOURCE_WEIGHT_JD_SUPPLEMENT = 0.65   # jd_vector_supplement 锚点的 source_weight（0.6~0.75）
TOPIC_WEIGHT_PRIMARY = 0.10   # primary 的 topic 权重
TOPIC_WEIGHT_DENSE = 0.25    # dense_expansion 的 topic 权重
TOPIC_WEIGHT_CLUSTER = 0.35  # cluster_expansion 的 topic 权重
TOPIC_WEIGHT_COOC = 0.25     # cooc_expansion 的 topic 权重
TOPIC_MIN_ALIGN = 0.20       # expansion 低对齐软门限（仅 expansion 生效）
TOPIC_LOW_ALIGN_PENALTY = 0.50  # 低对齐 expansion 额外惩罚系数
# Stage3 按 source_type 降权：cluster/cooc 视为噪声，最终分乘以下系数
CLUSTER_EXPANSION_PENALTY = 0.75   # cluster_expansion 最终分乘数
COOC_EXPANSION_PENALTY = 0.65      # cooc_expansion 最终分乘数

# 6.3 领域拟合（domain_fit）与 Stage2 门控
DOMAIN_FIT_WEIGHTS = (0.4, 0.3, 0.2, 0.1)  # domain, field, subfield, topic
DOMAIN_FIT_MIN_PRIMARY = 0.45      # Stage2A：domain_fit 低于此禁止做 primary（提高以压缩跨领域词）
DOMAIN_FIT_MIN_PRIMARY_BROAD = 0.40  # Stage2A：broad_concept 锚点（如 generic_task_term）做 primary 时 domain_fit 下限，强降权
DOMAIN_FIT_MIN_EXPANSION = 0.20   # Stage2B：扩展词 domain_fit 低于此不进词池
DOMAIN_SPAN_MAX_EXPANSION = 12    # Stage2B：扩展词 domain_span 超过此不进词池（跨域过大）

# 高歧义锚点（acronym / generic_task_term）做 primary 时 identity 下限，高于普通 PRIMARY_MIN_IDENTITY
PRIMARY_MIN_IDENTITY_HIGH_AMBIGUITY = 0.72

# Stage2B 高可信 primary：参与 dense/cluster/cooc 扩散须同时满足 identity、domain_fit、source、domain_span 等结构约束（不依赖词面黑名单）
DOMAIN_FIT_HIGH_CONFIDENCE = 0.55   # 高可信 primary 的 domain_fit 下限（提高以仅主场词参与扩散）
TRUSTED_SOURCE_TYPES_FOR_DIFFUSION = ("similar_to", "jd_vector", "conditioned_vec")  # 召回时 gte+JD 上下文(conditioned_vec) 与 similar_to/jd_vector 均可参与扩散

# Stage3 final_score 公式：source_weight / domain_gate / role_penalty
SOURCE_WEIGHT_SIMILAR_TO = 1.0    # primary 来自 similar_to
SOURCE_WEIGHT_JD_VECTOR = 0.95   # primary 来自 jd_vector
SOURCE_WEIGHT_CONDITIONED_VEC = 0.95   # primary 来自 conditioned_vec（召回时 gte+JD 上下文检索）
SOURCE_WEIGHT_DENSE = 0.85       # dense_expansion
SOURCE_WEIGHT_CLUSTER = 0.75    # cluster_expansion
SOURCE_WEIGHT_COOC = 0.70       # cooc_expansion
DOMAIN_GATE_MIN = 0.5           # domain_gate = DOMAIN_GATE_MIN + (1-DOMAIN_GATE_MIN)*domain_fit

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
        # 标签路召回：按 id 查找节点（与 UNIQUE 约束等价，显式列出便于与手动建库一致）
        "CREATE INDEX IF NOT EXISTS FOR (v:Vocabulary) ON (v.id)",
        "CREATE INDEX IF NOT EXISTS FOR (j:Job) ON (j.id)",
        "CREATE INDEX IF NOT EXISTS FOR (w:Work) ON (w.id)",
        "CREATE INDEX IF NOT EXISTS FOR (a:Author) ON (a.id)",
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
    "SBERT_MODEL_NAME": SBERT_MODEL_NAME,
    "HARDCORE_LEXICON_SNAPSHOT_PATH": HARDCORE_LEXICON_SNAPSHOT_PATH,
    "LABEL_DOMAIN_VECTORS_NPZ_PATH": LABEL_DOMAIN_VECTORS_NPZ_PATH,
    "LABEL_DOMAIN_VECTORS_META_PATH": LABEL_DOMAIN_VECTORS_META_PATH,
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
    "CREATE INDEX IF NOT EXISTS idx_aship_author_work ON authorships(author_id, work_id)",
    # 注：vocabulary_topic_stats / vocabulary_domain_stats 在 VOCAB_STATS_DB_PATH（vocab_stats.db），
    # 不在主库；对应 voc_id 索引见 build_vocab_stats_index._prepare_db。
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
