# -*- coding: utf-8 -*-
"""
技能清洗与 JD 文本归一化：分隔符、前后缀、垃圾词、残片规则。
"""

import re
import unicodedata

# -------------------------------------------------
# 1. skill 分隔符
# -------------------------------------------------

SKILL_SPLIT_PATTERN = re.compile(r"[，,、/|;；+&\\。！!：:]")

# -------------------------------------------------
# 1.1 标点/列表符前缀清洗（含 bullet ●▪•· 等）
# -------------------------------------------------
PUNCTUATION_PREFIX_PATTERN = re.compile(
    r"^[\s.()（）●▪•·\u2022\u2023\u2043\u00b7]+"
)

# -------------------------------------------------
# 2. JD 句式前缀
# -------------------------------------------------
PREFIX_PATTERN = re.compile(
    r"^(发表|要求|需|需要|熟悉|了解|掌握|精通|具备|能够|可以|负责|参与|从事|有|具有|擅长|使用|应用|在)"
)

# -------------------------------------------------
# 3. JD 句式尾缀（长词必须放前面）
# -------------------------------------------------
SUFFIX_PATTERN = re.compile(
    r"(开发经验|项目经验|相关经验|实践经验|研究经验|经验|能力|背景|优先|编程|框架|专家|工程师|研究员|总裁|总监|经理|产品|职称|为主|技术员|顾问|领军人物|企业|行业|工作|证书|职位|主任|招聘|优秀论文|论文|竞赛|奖|等)$"
)

# -------------------------------------------------
# 4. JD 垃圾描述关键词
# -------------------------------------------------
JD_NOISE_PATTERN = re.compile(
    r"(专业|学历|论文|竞赛|获奖|五险|出差|销售|工作语言|团队|沟通|表达|责任心|执行力|学习能力|抗压|汇报|学校|负责人|相关|经验|经验者)"
)

# -------------------------------------------------
# 5. 明确无意义词
# -------------------------------------------------
FORBIDDEN = {
    "不限",
    "其他",
    "相关",
    "方向",
    "经验",
    "能力",
    "背景",
    "周末双休",
    "年终奖",
    "企业",
    "行业核心期刊发表过文章",
    "小学生",
    "五险一金",
    "五险",
    "一金",
    "社保",
    "公积金"
}

# -------------------------------------------------
# 6. 职级 / 机构 / 类别（污染词）
# -------------------------------------------------
ROLE_WORDS = {
    "高端人才",
    "教授",
    "副教授",
    "助理教授",
    "讲师",
    "研究员",
    "往届生",
    "应届生",
    "职称",
    "应聘者",
    "博士",
    "硕士",
    "研究生",
    "本科",
    "中共党员"
}

ORG_WORDS = {
    "公立学校",
    "事业单位",
    "科研机构",
    "甲方公司",
    "双休",
}

CATEGORY_WORDS = {
    "职业与技能培训类",
    "技术类",
    "管理类",
}

# -------------------------------------------------
# 7. 短词白名单（避免误删）
# -------------------------------------------------
SHORT_WHITELIST = {
    "c",
    "c++",
    "go",
    "r",
    "py",
    "js",
    "sql"
}

DIGIT_WHITELIST = {
    "5g", "3d", "ros2",
    "python3", "java8", "java11", "c++11", "c++14", "c++17", "c++20",
    "vue2", "vue3", "react16", "react18", "angular2+", "angular4+",
    "tensorflow2", "pytorch1", "hadoop2", "hadoop3", "spark2", "spark3",
    "http2", "http3", "ipv4", "ipv6", "4k", "8k", "1080p", "720p",
    "2d", "3d", "4d", "2.5d", "2.4g", "5.8g", "2.4ghz", "5ghz"
}

JD_SENTENCE_PATTERN = re.compile(
    r"(发表|参加|参与|获得|承担).*(论文|竞赛|比赛|奖)"
)

# -------------------------------------------------
# 7.1 泛用 JD 描述：前缀/后缀 + 残片结构规则
# -------------------------------------------------
GENERIC_JD_PREFIXES = (
    "对系统",
    "建立高",
    "技术追踪",
)
GENERIC_JD_SUFFIXES = (
    "技术文档",
    "有高要求",
)
GENERIC_JD_VERB_HIGH_PERF_PREFIXES = ("建立高性能", "构建高性能", "实现高性能")

FRAGMENT_ACTION_PREFIXES = ("推动", "提升", "形成", "开展", "确保")
FRAGMENT_SOFT_SUFFIXES = ("鲁棒性", "可执行性", "系统化思维", "沟通协作能力")
FRAGMENT_EVALUATIVE = re.compile(r"(高要求|优秀|扎实|良好)")

# -------------------------------------------------
# skill 标准化
# -------------------------------------------------


def split_space_terms(term):
    tokens = term.split()
    if 3 <= len(tokens) <= 6 and all(len(t) <= 10 for t in tokens):
        return tokens
    return [term]


def normalize_skill(term: str):
    term = term.strip().lower()
    term = unicodedata.normalize("NFKC", term)
    term = re.sub(r'[\u200b\u200c\u200d]', '', term)
    term = PUNCTUATION_PREFIX_PATTERN.sub("", term)
    term = PREFIX_PATTERN.sub("", term)
    if term.startswith("熟练使用"):
        term = term[4:].strip()
    term = SUFFIX_PATTERN.sub("", term)
    term = re.sub(r'[\(（].*?[\)）]', '', term)
    term = re.sub(r'\s+', ' ', term).strip()
    term = term.strip(".,;；。:：")
    return term


def is_generic_jd_fragment(term: str) -> bool:
    """判断是否为泛用 JD 叙述残片，而非可独立映射学术词的技术概念。"""
    if not term or len(term) <= 5:
        return False
    if term.endswith("的") and len(term) >= 6:
        return True
    if term.count("的") >= 2:
        return True
    if len(term) > 18 and "的" in term:
        return True
    if any(term.startswith(p) for p in FRAGMENT_ACTION_PREFIXES):
        if len(term) >= 8 or FRAGMENT_EVALUATIVE.search(term) or "的" in term:
            return True
    if FRAGMENT_EVALUATIVE.search(term) and (len(term) >= 8 or term.endswith("的")):
        return True
    if any(term.endswith(s) for s in FRAGMENT_SOFT_SUFFIXES):
        if len(term) >= 10 or "的" in term:
            return True
    return False


def is_bad_skill(term: str):
    if not term:
        return True
    if term in FORBIDDEN:
        return True
    if term in ROLE_WORDS:
        return True
    if term in ORG_WORDS:
        return True
    if term in CATEGORY_WORDS:
        return True
    if JD_NOISE_PATTERN.search(term):
        return True
    if "能力" in term and len(term) > 6:
        return True
    if term.endswith("类"):
        return True
    if JD_SENTENCE_PATTERN.search(term):
        return True
    if any(term.startswith(p) for p in GENERIC_JD_PREFIXES):
        return True
    if any(term.endswith(s) for s in GENERIC_JD_SUFFIXES):
        return True
    if any(term.startswith(p) for p in GENERIC_JD_VERB_HIGH_PERF_PREFIXES):
        return True
    if is_generic_jd_fragment(term):
        return True
    if re.search(r"\d", term) and term not in DIGIT_WHITELIST:
        if not re.search(r'^[a-z\+]+[\d\.\+\-]+$', term, re.IGNORECASE):
            return True
    if len(term) > 25:
        return True
    if len(term) <= 1 and term not in SHORT_WHITELIST:
        return True
    if term.isdigit():
        return True
    if "(" in term or ")" in term or "（" in term or "）" in term:
        return True
    return False


def extract_skills(text: str):
    if not text:
        return []
    parts = SKILL_SPLIT_PATTERN.split(text)
    skills = []
    for p in parts:
        if not p or not p.strip():
            continue
        term = normalize_skill(p)
        if not term:
            continue
        for sub_term in split_space_terms(term):
            if is_bad_skill(sub_term):
                continue
            skills.append(sub_term)
    return list(set(skills))
