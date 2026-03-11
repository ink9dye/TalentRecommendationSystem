# -*- coding: utf-8 -*-
"""
领域配置统一模块：17 个业务领域的唯一数据源。

所有领域相关常量由此模块导出，避免在 config、total_recall、label_path 等处重复定义。
config 仅做兼容性 re-export（DOMAIN_MAP、NAME_TO_DOMAIN_ID），其余模块直接从本模块导入。
"""
from typing import Dict

# --- 唯一数据源：领域 ID -> 中文名、时序衰减率 ---
# 中文名：前端/业务展示；衰减率：标签路论文时序衰减。仅用 domain_id 做过滤，不做 Query 文本增强。
DOMAIN_TABLE: Dict[str, Dict[str, object]] = {
    "1":  {"name": "计算机科学", "decay_rate": 0.90},
    "2":  {"name": "医学",       "decay_rate": 0.94},
    "3":  {"name": "政治学",     "decay_rate": 0.95},
    "4":  {"name": "工程学",     "decay_rate": 0.92},
    "5":  {"name": "物理学",     "decay_rate": 0.95},
    "6":  {"name": "材料科学",   "decay_rate": 0.95},
    "7":  {"name": "生物学",     "decay_rate": 0.95},
    "8":  {"name": "地理学",     "decay_rate": 0.95},
    "9":  {"name": "化学",       "decay_rate": 0.95},
    "10": {"name": "商学",       "decay_rate": 0.95},
    "11": {"name": "社会学",     "decay_rate": 0.95},
    "12": {"name": "哲学",       "decay_rate": 0.99},
    "13": {"name": "环境科学",   "decay_rate": 0.95},
    "14": {"name": "数学",       "decay_rate": 0.98},
    "15": {"name": "心理学",     "decay_rate": 0.95},
    "16": {"name": "地质学",     "decay_rate": 0.95},
    "17": {"name": "经济学",     "decay_rate": 0.95},
}

# --- 派生导出（供各模块使用）---
DOMAIN_MAP: Dict[str, str] = {k: v["name"] for k, v in DOMAIN_TABLE.items()}
NAME_TO_DOMAIN_ID: Dict[str, str] = {v["name"]: k for k, v in DOMAIN_TABLE.items()}
DOMAIN_DECAY_RATES: Dict[str, float] = {k: v["decay_rate"] for k, v in DOMAIN_TABLE.items()}

# 默认衰减率（未在表中配置的领域使用）
DEFAULT_DECAY_RATE = 0.95
