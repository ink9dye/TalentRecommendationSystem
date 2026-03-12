# -*- coding: utf-8 -*-
"""
领域配置统一模块：17 个业务领域的唯一数据源。

所有领域相关常量由此模块导出，避免在 config、total_recall、label_path 等处重复定义。
config 仅做兼容性 re-export（DOMAIN_MAP、NAME_TO_DOMAIN_ID），其余模块直接从本模块导入。
"""
import re
from typing import Dict, Iterable, List, Dict as _Dict, Pattern

# --- 唯一数据源：领域 ID -> 中文名、时间衰减参数 ---
# 中文名：前端/业务展示；
# decay_rate：各领域的“时间半衰期”（half-life，单位：年），供标签路/向量路论文时序衰减使用。
DOMAIN_TABLE: Dict[str, Dict[str, object]] = {
    # decay_rate 含义：半衰期（年）。例如 6 表示 6 年前的论文权重约为当前的一半。
    "1":  {"name": "计算机科学", "decay_rate": 6},   # CS / AI：更新快，6 年半衰期
    "2":  {"name": "医学",       "decay_rate": 7},   # 医学：7 年
    "3":  {"name": "政治学",     "decay_rate": 10},  # 政治/政策：10 年
    "4":  {"name": "工程学",     "decay_rate": 8},   # 工程/控制/机器人：8 年
    "5":  {"name": "物理学",     "decay_rate": 12},  # 物理：12 年
    "6":  {"name": "材料科学",   "decay_rate": 10},  # 材料：10 年
    "7":  {"name": "生物学",     "decay_rate": 8},   # 生物：8 年
    "8":  {"name": "地理学",     "decay_rate": 12},  # 地理：12 年
    "9":  {"name": "化学",       "decay_rate": 10},  # 化学：10 年
    "10": {"name": "商学",       "decay_rate": 8},   # 商学/管理：8 年
    "11": {"name": "社会学",     "decay_rate": 10},  # 社会学：10 年
    "12": {"name": "哲学",       "decay_rate": 20},  # 哲学/人文：20 年
    "13": {"name": "环境科学",   "decay_rate": 10},  # 环境科学：10 年
    "14": {"name": "数学",       "decay_rate": 18},  # 数学：18 年
    "15": {"name": "心理学",     "decay_rate": 10},  # 心理学：10 年
    "16": {"name": "地质学",     "decay_rate": 15},  # 地质：15 年
    "17": {"name": "经济学",     "decay_rate": 10},  # 经济学：10 年
}

# --- 派生导出（供各模块使用）---
DOMAIN_MAP: Dict[str, str] = {k: v["name"] for k, v in DOMAIN_TABLE.items()}
NAME_TO_DOMAIN_ID: Dict[str, str] = {v["name"]: k for k, v in DOMAIN_TABLE.items()}
DOMAIN_DECAY_RATES: Dict[str, float] = {k: v["decay_rate"] for k, v in DOMAIN_TABLE.items()}

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

