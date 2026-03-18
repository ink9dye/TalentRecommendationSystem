# -*- coding: utf-8 -*-
"""
领域/文本时间衰减：半衰期与标题规则。
"""

from datetime import datetime
from typing import Iterable

from src.utils.domain_config import DOMAIN_DECAY_RATES, DEFAULT_DECAY_RATE, TEXT_DECAY_RULES


def apply_text_decay(title: str) -> float:
    """根据标题内容计算文本类型衰减系数。命中多个规则时系数相乘；未命中时为 1.0。"""
    if not title:
        return 1.0
    decay = 1.0
    for rule in TEXT_DECAY_RULES:
        pat = rule["pattern"]
        fac = rule["factor"]
        if pat.search(title):
            decay *= fac
    return decay


def get_decay_rate_for_domains(domain_ids: Iterable[str]) -> float:
    """
    给定一组领域 ID（如 {"1","4","14"}），返回统一使用的“时间半衰期”（单位：年）。
    注意：DOMAIN_DECAY_RATES 中的值含义已从“底数”改为 half-life。
    """
    ids = {str(d).strip() for d in domain_ids if str(d).strip()}
    if not ids:
        return DEFAULT_DECAY_RATE
    half_lives = [DOMAIN_DECAY_RATES.get(d, DEFAULT_DECAY_RATE) for d in ids]
    return max(half_lives)


def compute_time_decay(year: int, domain_ids: Iterable[str]) -> float:
    """统一的时间衰减接口：weight = 0.5 ** (age / half_life)。"""
    try:
        y = int(year) if year is not None else 2000
    except (TypeError, ValueError):
        y = 2000
    current_year = datetime.now().year
    age = max(0, current_year - y)
    half_life = get_decay_rate_for_domains(domain_ids)
    if half_life <= 0:
        return 1.0
    return 0.5 ** (age / half_life)
