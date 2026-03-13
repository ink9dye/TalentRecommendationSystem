import re
import unicodedata
from datetime import datetime
from typing import Iterable

from src.utils.domain_config import DOMAIN_DECAY_RATES, DEFAULT_DECAY_RATE, TEXT_DECAY_RULES

# -------------------------------------------------
# 1. skill 分隔符
# -------------------------------------------------


SKILL_SPLIT_PATTERN = re.compile(r"[，,、/|;；+&\\。！!：:]")

# -------------------------------------------------
# 1.1 新增：标点前缀清洗
# -------------------------------------------------
PUNCTUATION_PREFIX_PATTERN = re.compile(
    r"^[.()（）]+"
)

# -------------------------------------------------
# 2. JD 句式前缀
# -------------------------------------------------

PREFIX_PATTERN = re.compile(
    r"^(发表|要求|需|需要|熟悉|了解|掌握|精通|具备|能够|可以|负责|参与|从事|有|具有|擅长|使用|应用)"
)
# -------------------------------------------------
# 3. JD 句式尾缀（长词必须放前面）
# -------------------------------------------------

SUFFIX_PATTERN = re.compile(
    r"(开发经验|项目经验|相关经验|实践经验|研究经验|经验|能力|背景|优先|编程|框架|专家|工程师|研究员|总裁|总监|经理|产品|职称|为主|技术员|顾问|领军人物|企业|行业|工作|证书|职位|主任|招聘|优秀论文|论文|竞赛|奖)$"
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
    "五险一金",  # 新增
    "五险",  # 新增
    "一金",  # 新增
    "社保",  # 新增
    "公积金"  # 新增
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
    "r",  # 新增
    "py",  # 新增
    "js",  # 新增
    "sql"  # 新增
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
# skill 标准化
# -------------------------------------------------
def split_space_terms(term):
    tokens = term.split()

    # 至少3个词才拆
    if 3 <= len(tokens) <= 6 and all(len(t) <= 10 for t in tokens):
        return tokens

    return [term]

def normalize_skill(term: str):
    term = term.strip().lower()
    term = unicodedata.normalize("NFKC", term)
    term = re.sub(r'[\u200b\u200c\u200d]', '', term)
    # 新增：去除开头的点号括号
    term = PUNCTUATION_PREFIX_PATTERN.sub("", term)

    # 去前缀
    term = PREFIX_PATTERN.sub("", term)

    # 去后缀
    term = SUFFIX_PATTERN.sub("", term)

    # 新增：去除括号及括号内容
    term = re.sub(r'[\(（].*?[\)）]', '', term)

    # 新增：去除多余的空白字符
    term = re.sub(r'\s+', ' ', term).strip()

    term = term.strip(".,;；。:：")
    return term


# -------------------------------------------------
# 判断是否是垃圾 skill
# -------------------------------------------------


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

    # 删除类别词
    if term.endswith("类"):
        return True



    if JD_SENTENCE_PATTERN.search(term):
        return True

    # 删除数字条件，但保留一些专业术语
    if re.search(r"\d", term) and term not in DIGIT_WHITELIST:
        # 新增：如果是版本号模式（如python3.7, c++11），允许通过
        if not re.search(r'^[a-z\+]+[\d\.\+\-]+$', term, re.IGNORECASE):
            return True

    # 删除过长句子
    if len(term) > 25:
        return True

    # 删除过短词
    if len(term) <= 1 and term not in SHORT_WHITELIST:
        return True

    # 新增：删除纯数字
    if term.isdigit():
        return True

    # 新增：删除括号碎片
    if "(" in term or ")" in term or "（" in term or "）" in term:
        return True

    return False


# -------------------------------------------------
# 主函数：从 JD 提取 skill
# -------------------------------------------------

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


def apply_text_decay(title: str) -> float:
    """
    根据标题内容计算文本类型衰减系数。
    命中多个规则时系数相乘；未命中时为 1.0。
    """
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
    # 策略：取半衰期最长的那条（衰减最慢），保证多领域时不“错杀”长寿命学科的经典成果
    return max(half_lives)


def compute_time_decay(year: int, domain_ids: Iterable[str]) -> float:
    """
    统一的时间衰减接口：基于“半衰期”模型。

    weight = 0.5 ** (age / half_life)

    - year: 论文发表年份，缺失或异常时退化为 2000 年。
    - domain_ids: 当前查询涉及的领域 ID 集合，用于从 DOMAIN_DECAY_RATES 中选取 half-life。
    """
    try:
        y = int(year) if year is not None else 2000
    except (TypeError, ValueError):
        y = 2000

    current_year = datetime.now().year
    age = max(0, current_year - y)
    half_life = get_decay_rate_for_domains(domain_ids)
    # half_life<=0 时视为不衰减，返回 1.0
    if half_life <= 0:
        return 1.0
    return 0.5 ** (age / half_life)
