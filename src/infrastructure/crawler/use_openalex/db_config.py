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

# 3. 领域 ID 配置
FIELDS = {
    "1": ("Computer_science", "C41008148"),
    "2": ("Medicine", "C71924100"),
    "3": ("Political_science", "C17744445"),
    "4": ("Engineering", "C127413603"),
    "5": ("Physics", "C121332964"),
    "6": ("Materials_science", "C192562407"),
    "7": ("Biology", "C86803240"),
    "8": ("Geography", "C205649164"),
    "9": ("Chemistry", "C185592680"),
    "10": ("Business", "C144133560"),
    "11": ("Sociology", "C144024400"),
    "12": ("Philosophy", "C138885662"),
    "13": ("Environmental_science", "C39432304"),
    "14": ("Mathematics", "C33923547"),
    "15": ("Psychology", "C15744967"),
    "16": ("Geology", "C127313418"),
    "17": ("Economics", "C162324750")
}

# 4. API 性能配置
BASE_DELAY = 0.1
BATCH_SIZE_AUTHORS = 25
MAX_WORKERS = 3