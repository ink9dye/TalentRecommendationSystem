import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 基础配置信息 ---
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Aa1278389701"
NEO4J_DATABASE = "talent-graph"
# 2. 核心数据目录
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 3. 索引存储目录
INDEX_DIR = os.path.join(DATA_DIR, "build_index")
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)

# 4. 数据库路径
DB_PATH = os.path.join(DATA_DIR, "academic_dataset_v5.db")

# 5. 三大向量索引具体路径 (对应你 tex 文档中的设计)
# 词汇索引 (跨语言对齐用)
VOCAB_INDEX_PATH = os.path.join(INDEX_DIR, "vocabulary.faiss")
VOCAB_MAP_PATH = os.path.join(INDEX_DIR, "vocabulary_mapping.json")

# 摘要索引 (向量路召回用)
ABSTRACT_INDEX_PATH = os.path.join(INDEX_DIR, "abstract.faiss")
ABSTRACT_MAP_PATH = os.path.join(INDEX_DIR, "abstract_mapping.json")

# 岗位描述索引 (标签路/需求对齐用)
JOB_INDEX_PATH = os.path.join(INDEX_DIR, "job_description.faiss")
JOB_MAP_PATH = os.path.join(INDEX_DIR, "job_description_mapping.json")

# 6. 其他索引
FEATURE_INDEX_PATH = os.path.join(INDEX_DIR, "feature_index.json")
COLLAB_DB_PATH = os.path.join(INDEX_DIR, "scholar_collaboration.db")

# 7. SBERT 模型本地存放路径
SBERT_DIR = os.path.join(DATA_DIR, "build_sbert")
if not os.path.exists(SBERT_DIR):
    os.makedirs(SBERT_DIR)

