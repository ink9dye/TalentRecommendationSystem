# -*- coding: utf-8 -*-
"""
领域配置统一模块：17 个业务领域的唯一数据源。

所有领域相关常量由此模块导出，避免在 config、total_recall、label_path 等处重复定义。
config 仅做兼容性 re-export（DOMAIN_MAP、NAME_TO_DOMAIN_ID），其余模块直接从本模块导入。
"""
import re
from typing import Dict, Iterable, List, Dict as _Dict, Pattern, Tuple

# --- 唯一数据源：领域 ID -> 中文名、英文名、OpenAlex concept ID、时间衰减 ---
# name：中文名，前端/业务展示；
# name_en：英文名，与 OpenAlex display_name 对齐（空格小写形式见 NAME_EN_TO_DOMAIN_ID）；
# openalex_concept_id：OpenAlex 概念 ID，爬虫/API 使用；
# decay_rate：半衰期（年），供标签路/向量路论文时序衰减使用。
DOMAIN_TABLE: Dict[str, Dict[str, object]] = {
    "1":  {"name": "计算机科学", "name_en": "Computer_science", "openalex_concept_id": "C41008148", "decay_rate": 4},
    "2":  {"name": "医学",       "name_en": "Medicine", "openalex_concept_id": "C71924100", "decay_rate": 5},
    "3":  {"name": "政治学",     "name_en": "Political_science", "openalex_concept_id": "C17744445", "decay_rate": 8},
    "4":  {"name": "工程学",     "name_en": "Engineering", "openalex_concept_id": "C127413603", "decay_rate": 6},
    "5":  {"name": "物理学",     "name_en": "Physics", "openalex_concept_id": "C121332964", "decay_rate": 10},
    "6":  {"name": "材料科学",   "name_en": "Materials_science", "openalex_concept_id": "C192562407", "decay_rate": 8},
    "7":  {"name": "生物学",     "name_en": "Biology", "openalex_concept_id": "C86803240", "decay_rate": 6},
    "8":  {"name": "地理学",     "name_en": "Geography", "openalex_concept_id": "C205649164", "decay_rate": 10},
    "9":  {"name": "化学",       "name_en": "Chemistry", "openalex_concept_id": "C185592680", "decay_rate": 8},
    "10": {"name": "商学",       "name_en": "Business", "openalex_concept_id": "C144133560", "decay_rate": 6},
    "11": {"name": "社会学",     "name_en": "Sociology", "openalex_concept_id": "C144024400", "decay_rate": 8},
    "12": {"name": "哲学",       "name_en": "Philosophy", "openalex_concept_id": "C138885662", "decay_rate": 18},
    "13": {"name": "环境科学",   "name_en": "Environmental_science", "openalex_concept_id": "C39432304", "decay_rate": 8},
    "14": {"name": "数学",       "name_en": "Mathematics", "openalex_concept_id": "C33923547", "decay_rate": 16},
    "15": {"name": "心理学",     "name_en": "Psychology", "openalex_concept_id": "C15744967", "decay_rate": 8},
    "16": {"name": "地质学",     "name_en": "Geology", "openalex_concept_id": "C127313418", "decay_rate": 13},
    "17": {"name": "经济学",     "name_en": "Economics", "openalex_concept_id": "C162324750", "decay_rate": 8},
}

# --- 派生导出（供各模块使用）---
DOMAIN_MAP: Dict[str, str] = {k: v["name"] for k, v in DOMAIN_TABLE.items()}
NAME_TO_DOMAIN_ID: Dict[str, str] = {v["name"]: k for k, v in DOMAIN_TABLE.items()}
DOMAIN_DECAY_RATES: Dict[str, float] = {k: v["decay_rate"] for k, v in DOMAIN_TABLE.items()}

# 供 OpenAlex 爬虫使用：领域 ID -> (英文名, OpenAlex concept ID)，与原 db_config.FIELDS 同形
OPENALEX_FIELDS: Dict[str, Tuple[str, str]] = {
    k: (v["name_en"], v["openalex_concept_id"]) for k, v in DOMAIN_TABLE.items()
}

# 归一化英文名 -> 领域 ID（供 OpenAlex display_name 匹配等使用）
NAME_EN_TO_DOMAIN_ID: Dict[str, str] = {
    v["name_en"].replace("_", " ").lower(): k for k, v in DOMAIN_TABLE.items()
}

# 默认半衰期（未在表中配置的领域使用，单位：年）
DEFAULT_DECAY_RATE = 10.0


# --- 标题降权规则（统一供 Label / Vector 使用）---

TEXT_DECAY_RULES: List[_Dict[str, object]] = [
    {
        "id": "survey",
        "pattern": re.compile(r"(survey|overview|review|handbook|textbook)", re.IGNORECASE),
        "factor": 0.1,
    },
    {
        "id": "data",
        "pattern": re.compile(r"(data from:|dataset:|supplementary data)", re.IGNORECASE),
        "factor": 0.05,
    },
]

