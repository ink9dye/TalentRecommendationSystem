# -*- coding: utf-8 -*-
"""
薄转发层：保持 from src.utils.tools import ... 兼容，实际实现位于 utils 下独立模块。
- 时序衰减：src.utils.decay
- 技能清洗：src.utils.skill_clean
- 句子残片过滤：src.utils.text_filters
"""

from src.utils.decay import (
    apply_text_decay,
    get_decay_rate_for_domains,
    compute_time_decay,
)
from src.utils.skill_clean import (
    SKILL_SPLIT_PATTERN,
    PUNCTUATION_PREFIX_PATTERN,
    PREFIX_PATTERN,
    SUFFIX_PATTERN,
    JD_NOISE_PATTERN,
    FORBIDDEN,
    ROLE_WORDS,
    ORG_WORDS,
    CATEGORY_WORDS,
    SHORT_WHITELIST,
    DIGIT_WHITELIST,
    JD_SENTENCE_PATTERN,
    GENERIC_JD_PREFIXES,
    GENERIC_JD_SUFFIXES,
    GENERIC_JD_VERB_HIGH_PERF_PREFIXES,
    FRAGMENT_ACTION_PREFIXES,
    FRAGMENT_SOFT_SUFFIXES,
    FRAGMENT_EVALUATIVE,
    split_space_terms,
    normalize_skill,
    is_generic_jd_fragment,
    is_bad_skill,
    extract_skills,
)

__all__ = [
    "apply_text_decay",
    "get_decay_rate_for_domains",
    "compute_time_decay",
    "SKILL_SPLIT_PATTERN",
    "PUNCTUATION_PREFIX_PATTERN",
    "PREFIX_PATTERN",
    "SUFFIX_PATTERN",
    "JD_NOISE_PATTERN",
    "FORBIDDEN",
    "ROLE_WORDS",
    "ORG_WORDS",
    "CATEGORY_WORDS",
    "SHORT_WHITELIST",
    "DIGIT_WHITELIST",
    "JD_SENTENCE_PATTERN",
    "GENERIC_JD_PREFIXES",
    "GENERIC_JD_SUFFIXES",
    "GENERIC_JD_VERB_HIGH_PERF_PREFIXES",
    "FRAGMENT_ACTION_PREFIXES",
    "FRAGMENT_SOFT_SUFFIXES",
    "FRAGMENT_EVALUATIVE",
    "split_space_terms",
    "normalize_skill",
    "is_generic_jd_fragment",
    "is_bad_skill",
    "extract_skills",
]
