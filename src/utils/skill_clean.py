# -*- coding: utf-8 -*-
"""
技能清洗与 JD 文本归一化：分隔符、前后缀、垃圾词、残片规则。
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional

try:
    from src.utils.text_filters.sentence_fragment_filter import is_sentence_fragment
except ImportError:
    def is_sentence_fragment(text: str) -> bool:
        return False

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
    r"^(发表|要求|需|需要|熟悉|了解|掌握|精通|具备|能够|可以|负责|参与|从事|有|具有|擅长|使用|应用|在|"
    r"利用|调研|对|研发与优化|建立|推动|开展|持续关注|包括但不限于|针对|确保)"
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
    r"(专业|学历|论文|竞赛|获奖|五险|出差|销售|工作语言|团队|沟通|表达|责任心|执行力|学习能力|抗压|汇报|学校|负责人|相关|经验|经验者|"
    r"任职要求|岗位职责|核心要求|加分项|基本要求|工作职责|职位描述)"
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

FRAGMENT_ACTION_PREFIXES = ("推动", "提升", "形成", "开展", "确保", "调研")
FRAGMENT_SOFT_SUFFIXES = ("鲁棒性", "可执行性", "平滑性", "系统化思维", "沟通协作能力")
FRAGMENT_EVALUATIVE = re.compile(r"(高要求|优秀|扎实|良好)")

# 泛 JD 说明壳：整段多为叙述/目标/流程，不宜作技术锚点（仅对含中文的 term 启用，避免误伤纯英文技能词）
JD_META_SHELL_PATTERN = re.compile(
    r"(前沿研究|一致性优化|全流程开发|进行调研|进行约束|仿真到实机|平台构建与验证|"
    r"及规划领域|领域的前沿)"
)

# 极短纯泛化词（非具体技术栈名）
GENERIC_SHORT_SKILL_NOISE = frozenset(
    {
        "高性能",
        "系统架构",
    }
)

# 「工具名 + 等…说明尾巴」：整段不进 cleaned_skills，由 extract_skills 尝试剥离前导拉丁工具 token
JD_TOOL_LISTING_SHELL_PATTERNS = (
    re.compile(r"^([a-z][a-z0-9+\-./]*)\s+等机器人常用开发库", re.IGNORECASE),
    re.compile(r"^([a-z][a-z0-9+\-./]*)等机器人常用开发库", re.IGNORECASE),
    re.compile(r"^([a-z][a-z0-9+\-./]*)\s+等平台搭建仿真环境", re.IGNORECASE),
    re.compile(r"^([a-z][a-z0-9+\-./]*)等平台搭建仿真环境", re.IGNORECASE),
)
JD_TOOL_LISTING_TAIL_RE = re.compile(r"(等机器人常用开发库|等平台搭建仿真环境)")

# JD 目录级标题前缀（任职要求● 掌握... → 取●后的内容再归一化）
JD_HEADER_PREFIX = re.compile(
    r"^(任职要求|岗位职责|核心要求|加分项|基本要求|工作职责|职位描述)[\s●▪•·]*",
    re.IGNORECASE,
)

# -------------------------------------------------
# skill 标准化
# -------------------------------------------------

# 英文技术向 token：字母开头，可含数字与常见技术符号（避免把叙述词表当白名单）
_TOKEN_TECH_SHAPED = re.compile(r"^[a-zA-Z][a-zA-Z0-9+\-./]*$")


def _has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def _has_latin_skill_token(s: str) -> bool:
    """含 2+ 连续拉丁字母时倾向视为技术缩写/栈名，避免把「包括但不限于 mpc」类条误杀。"""
    return bool(re.search(r"[a-z]{2,}", s, re.IGNORECASE))


def _peel_leading_tool_from_listing_shell(term: str) -> Optional[List[str]]:
    """
    识别「前导拉丁工具/缩写 + 等机器人常用开发库 / 等平台搭建仿真环境」类说明壳。
    返回 [工具小写 token] 供单独准入；不匹配则返回 None（走原有整段判定）。
    """
    if not term or not _has_cjk(term):
        return None
    t = term.strip()
    for pat in JD_TOOL_LISTING_SHELL_PATTERNS:
        m = pat.match(t)
        if not m:
            continue
        tok = (m.group(1) or "").strip().lower()
        if not tok or not _token_is_tech_shaped(tok):
            continue
        return [tok]
    return None


def _token_is_tech_shaped(tok: str) -> bool:
    """形式特征：像技术词汇的英文 token（非停用词表）。"""
    if len(tok) < 2 or len(tok) > 18:
        return False
    return bool(_TOKEN_TECH_SHAPED.match(tok))


def _should_preserve_spaced_term_whole(term: str) -> bool:
    """
    短语保留优先：空格分隔的整体是否应作为一条技能保留，而不是拆子词或退化为首 token。

    基于形式特征（token 数、长度、字符集、是否叙述残片），不写死具体技术名词。
    """
    if not term or " " not in term:
        return False
    # 中英混排或含中文的空格短语：整体保留，避免把「Python 开发」等拆碎
    if _has_cjk(term):
        return True
    tokens = term.split()
    # 2~5 个英文技术向 token：覆盖 motion control、robot arm control、isaac sim … framework 等整行产品名
    if not (2 <= len(tokens) <= 5):
        return False
    if len(term) > 60:
        return False
    if not all(_token_is_tech_shaped(t) for t in tokens):
        return False
    if is_generic_jd_fragment(term):
        return False
    if is_sentence_fragment(term):
        return False
    return True


def _should_degrade_long_spaced_term_to_first_token(term: str) -> bool:
    """
    仅在「明显不像可整体保留的短技术短语」的长串上，允许退化为首词。
    用于替代原先「len>15 且含空格则一律首词」的强规则。
    """
    if not term or " " not in term:
        return False
    if _should_preserve_spaced_term_whole(term):
        return False
    tokens = term.split()
    if _has_cjk(term):
        return False
    # 若仍符合「全技术向 token」且未过长，应由 preserve 整行保留，此处不退化
    if (
        2 <= len(tokens) <= 5
        and len(term) <= 60
        and all(_token_is_tech_shaped(t) for t in tokens)
    ):
        return False
    # 叙述性长串：词数多、总长过大，或夹杂非技术形态的 token
    if len(tokens) >= 6:
        return True
    if len(term) > 60:
        return True
    if len(tokens) >= 4 and not all(_token_is_tech_shaped(t) for t in tokens):
        return True
    if any(len(t) > 18 for t in tokens):
        return True
    return False


def split_space_terms(term):
    """
    短语保留优先：始终返回整段 term，不再按空格拆成多个子 skill。
    旧版对 3~6 个短 token 一律拆分的策略已移除，避免子词覆盖原短语。
    多词是否退化为首词仅在 extract_skills 中由 _should_degrade_* 判定。
    """
    return [term]


def normalize_skill(term: str):
    term = term.strip().lower()
    term = unicodedata.normalize("NFKC", term)
    term = re.sub(r'[\u200b\u200c\u200d]', '', term)
    term = JD_HEADER_PREFIX.sub("", term)
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
    # 软指标 / 质量叙述：「稳定性与可执行性」等不以「的」结尾，原 10 字门槛过严
    if "与可执行性" in term and len(term) >= 6:
        return True
    if "系统的平滑性" in term:
        return True
    if "扎实" in term and "工程实现" in term:
        return True
    if any(term.startswith(p) for p in FRAGMENT_ACTION_PREFIXES):
        if len(term) >= 8 or FRAGMENT_EVALUATIVE.search(term) or "的" in term:
            return True
    if FRAGMENT_EVALUATIVE.search(term) and (len(term) >= 8 or term.endswith("的")):
        return True
    if FRAGMENT_EVALUATIVE.search(term) and len(term) >= 6 and "工程" in term:
        return True
    if any(term.endswith(s) for s in FRAGMENT_SOFT_SUFFIXES):
        if len(term) >= 6 or "的" in term:
            return True
    return False


def _is_jd_duty_algorithm_shell(term: str) -> bool:
    """
    长职责/研发壳：以「算法开发/算法研发」收尾或含「全身控制算法」叙述块。
    刻意不泛化到「控制算法」「最优控制」等短技术锚点。
    """
    if not term or not _has_cjk(term):
        return False
    if len(term) >= 10 and term.endswith("算法开发"):
        return True
    if len(term) >= 10 and term.endswith("算法研发"):
        return True
    if "全身控制算法" in term and len(term) >= 12:
        return True
    if len(term) >= 12 and term.endswith("模块"):
        return True
    return False


def _is_jd_slogan_innovation_shell(term: str) -> bool:
    """口号式：知识沉淀、技术创新并形成…"""
    if not term or not _has_cjk(term):
        return False
    if "形成知识沉淀" in term:
        return True
    if "技术创新" in term and "并形成" in term:
        return True
    return False


def _is_evaluative_math_foundation(term: str) -> bool:
    """素质向：扎实的数学基础（不误伤「线性代数」「数值方法」等独立技能名）。"""
    if not term or not _has_cjk(term):
        return False
    if "数学基础" not in term:
        return False
    return bool(re.search(r"(扎实|优秀|良好|深厚).{0,4}数学基础", term))


def is_bad_skill(term: str):
    if not term:
        return True
    if is_sentence_fragment(term):
        return True
    if len(term) > 20 and "的" in term:
        return True
    if any(p in term for p in ["等相关", "等工具", "等库", "等常见"]):
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
    if term in GENERIC_SHORT_SKILL_NOISE:
        return True
    if _has_cjk(term) and JD_META_SHELL_PATTERN.search(term):
        return True
    if term.startswith("包括") and not _has_latin_skill_token(term):
        return True
    if _is_jd_slogan_innovation_shell(term):
        return True
    if _is_jd_duty_algorithm_shell(term):
        return True
    if _is_evaluative_math_foundation(term):
        return True
    if _has_cjk(term) and _has_latin_skill_token(term) and JD_TOOL_LISTING_TAIL_RE.search(term):
        return True
    if is_generic_jd_fragment(term):
        return True
    if re.search(r"\d", term) and term not in DIGIT_WHITELIST:
        if not re.search(r'^[a-z\+]+[\d\.\+\-]+$', term, re.IGNORECASE):
            return True
    if len(term) > 25:
        # 与「短语保留优先」一致：已判定为可整体保留的英文多词技术短语，允许略超原 25 字上限
        if not (" " in term and _should_preserve_spaced_term_whole(term) and len(term) <= 60):
            return True
    if len(term) <= 1 and term not in SHORT_WHITELIST:
        return True
    if term.isdigit():
        return True
    if "(" in term or ")" in term or "（" in term or "）" in term:
        return True
    return False


def extract_skills(
    text: str,
    *,
    fragment_stats: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if not text:
        return []
    parts = SKILL_SPLIT_PATTERN.split(text)
    skills = []
    for p in parts:
        if not p or not p.strip():
            continue
        raw_p = p.strip()
        term = normalize_skill(p)
        if not term:
            continue
        # 删除「长英文多词只保留首 token」的强规则；仅对明显叙述长串退化为首词
        if len(term) > 15 and " " in term:
            if _should_degrade_long_spaced_term_to_first_token(term):
                first_token = term.split()[0].strip()
                if first_token and len(first_token) <= 20:
                    term = first_token
        for sub_term in split_space_terms(term):
            if not sub_term:
                continue
            # normalize 会剥掉「对/推动」等 JD 前缀，残片规则需在归一化子项与原始分片上同时判定
            frag_sub = is_sentence_fragment(sub_term)
            frag_raw = is_sentence_fragment(raw_p)
            if frag_sub or frag_raw:
                if fragment_stats is not None:
                    fragment_stats["sentence_fragment_removed_count"] = (
                        int(fragment_stats.get("sentence_fragment_removed_count") or 0) + 1
                    )
                    samples = fragment_stats.setdefault("sentence_fragment_removed_samples", [])
                    if len(samples) < 5:
                        sample = sub_term if frag_sub else raw_p
                        if sample and sample not in samples:
                            samples.append(sample)
                continue
            peeled = _peel_leading_tool_from_listing_shell(sub_term)
            if peeled is not None:
                for tok in peeled:
                    if tok and not is_bad_skill(tok):
                        skills.append(tok)
                continue
            if is_bad_skill(sub_term):
                continue
            skills.append(sub_term)
    return list(set(skills))
