import os

# 1. 基础身份配置
EMAIL = "2022337621072@mails.zstu.edu.cn"

# 2. 路径配置
# 获取当前 db_config.py 所在的目录 (即 .../use_openalex/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置数据库路径为上一级目录 (即 .../use_openalex/../academic_dataset_v5.db)
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "academic_dataset_v5.db"))

# 如果你还需要一个专门存放导出数据的文件夹，也可以设在同级
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "academic_dataset_v5_exports"))
os.makedirs(DATA_DIR, exist_ok=True)

# 3. 领域配置（统一由 src.utils.domain_config 维护，此处仅兼容 re-export）
from src.utils.domain_config import OPENALEX_FIELDS as FIELDS

# 4. API 性能配置
BASE_DELAY = 0.1
BATCH_SIZE_AUTHORS = 25
MAX_WORKERS = 3