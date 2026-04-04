import json
import math
import os
import re
import sqlite3
from typing import Dict, Any, Set, List, Tuple, Optional

import faiss
import numpy as np

from config import DB_PATH, DATA_DIR, VOCAB_P95_PAPER_COUNT, ANCHOR_SOURCE_WEIGHT_JD_SUPPLEMENT
from src.utils.tools import extract_skills
from src.core.recall.label_means.label_debug import debug_print

# backbone_score 权重：in_jd_context / is_task_like 权重大于 job_freq，避免图热词再次主导
BACKBONE_W_JOB_FREQ = 0.15
BACKBONE_W_COV_J = 0.2
BACKBONE_W_SPAN = 0.1
BACKBONE_W_IN_JD = 0.35
BACKBONE_W_TASK_LIKE = 0.35
BACKBONE_W_SIM = 0.2  # query 向量相似度（可选）
BACKBONE_FLOOR_FOR_BAODI = 0.5  # 层0 保底词的最低得分，确保参与排序后有机会进 TopN

# JD 进入 SBERT 的统一字符切片：与历史 anchor_ctx 中 query_for_ctx[:800] 一致（不 strip），
# 供 supplement 与 prefill 共用同一次 forward，避免 [:500]+strip 与 [:800] 各编一次。
LABEL_JD_ENCODE_MAX_CHARS = 800


def canonical_jd_text_for_encode(query_text: str) -> str:
    """与 Stage1 条件化锚点 JD 向量使用的切片一致。"""
    return (query_text or "")[:LABEL_JD_ENCODE_MAX_CHARS]


# 缩写词典 key 集合缓存，供 classify_anchor_type 判断 acronym
_ABBR_KEYS_CACHE: Set[str] | None = None


def _load_abbr_keys() -> Set[str]:
    """加载 data/industrial_abbr_expansion.json 的缩写 key 集合（小写），用于锚点类型 acronym 判断。"""
    global _ABBR_KEYS_CACHE
    if _ABBR_KEYS_CACHE is not None:
        return _ABBR_KEYS_CACHE
    path = os.path.join(DATA_DIR, "industrial_abbr_expansion.json")
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _ABBR_KEYS_CACHE = {str(k).strip().lower() for k in (data or {}).keys()}
        else:
            _ABBR_KEYS_CACHE = set()
    except Exception:
        _ABBR_KEYS_CACHE = set()
    return _ABBR_KEYS_CACHE


def classify_anchor_type(term: str) -> str:
    """
    对锚点（技能/概念词）打类型标签，供 Stage2/Stage3 按类型分策略使用。
    返回: acronym | canonical_academic_like | application_term | generic_task_term | unknown
    """
    if not term or len((term or "").strip()) < 2:
        return "unknown"
    t = (term or "").strip().lower()
    if t in _load_abbr_keys():
        return "acronym"
    generic_task_words = {
        "算法", "模型", "方法", "系统", "开发", "技术", "学习", "研究",
        "research", "algorithm", "model", "method", "system", "framework",
        "learning", "optimization",
    }
    if t in generic_task_words:
        return "generic_task_term"
    application_kws = ("应用", "开发", "工程", "implementation", "application", "engineering")
    if any(k in t for k in application_kws):
        return "application_term"
    return "canonical_academic_like"


def clean_job_skills(skills_text: str) -> Set[str]:
    """
    复用全局 JD 技能抽取逻辑，将岗位技能字段统一走 src.utils.tools.extract_skills，
    返回小写去重集合，便于与图谱 term 对齐。
    """
    if not skills_text or not isinstance(skills_text, str):
        return set()
    return set(extract_skills(str(skills_text)))


def _in_jd_context(term_text: str, cleaned_terms: Set[str] | None) -> bool:
    """
    判断图谱中的短 term 是否与当前 JD 清洗得到的技能短语存在“语境重合”：
      - 先看精确命中；
      - 再看 term 是否作为子串出现在任一 cleaned term 中（例如 '路径规划' 命中
        '轨迹规划与全身控制算法开发'）。
    仅使用简单的子串规则，避免过早引入复杂相似度。
    """
    if not term_text or not cleaned_terms:
        return False
    t = term_text.lower().strip()
    if not t:
        return False
    if t in cleaned_terms:
        return True
    # 过短的 term（单字）不做子串匹配，避免噪声放大
    if len(t) <= 1:
        return False
    for jt in cleaned_terms:
        if not jt:
            continue
        if t in jt:
            return True
    return False


def _is_task_like(term_text: str) -> bool:
    """
    轻量判断 term 是否更像“任务骨干”而非泛 soft-skill：
      - 命中运动学/动力学/轨迹/规划/仿真/识别/估计/控制/优化/参数等关键词之一。
    """
    if not term_text:
        return False
    t = term_text.lower()
    task_kws = [
        "运动学",
        "动力学",
        "trajectory",
        "轨迹",
        "planning",
        "规划",
        "仿真",
        "simulation",
        "识别",
        "estimation",
        "估计",
        "control",
        "控制",
        "优化",
        "optimization",
        "参数",
    ]
    return any(k in t for k in task_kws)


# ---------- 短语中心性（无硬编码词表，用于锚点重排） ----------
PHRASE_SPECIFICITY_MIN_LEN = 2
CONTEXT_WINDOW_CHAR = 120
# 支撑项凸组合权重（与 backbone 做「带底座」合成，避免四因子连乘把主线锚压到 0.0x）
ANCHOR_SUPPORT_W_SPEC = 0.30
ANCHOR_SUPPORT_W_CTX = 0.25
ANCHOR_SUPPORT_W_TASK = 0.25
ANCHOR_SUPPORT_W_LOCAL = 0.20
# final_anchor_score = backbone * (FLOOR_FRAC + SPREAD_FRAC * support_mean)，support_mean∈[0,1] → 乘子∈[FLOOR, FLOOR+SPREAD]
ANCHOR_FINAL_BACKBONE_FLOOR_FRAC = 0.72
ANCHOR_FINAL_SUPPORT_SPREAD_FRAC = 0.28
# [Stage1-Step2 anchor collapse audit]：对照 collapse_ratio，排查「路径规划被压穿」类问题
ANCHOR_COLLAPSE_AUDIT = True

# --- Stage1 角色接入（字符串与 stage1_domain_anchors.coarse_classify 输出保持一致）---
_JCR_CORE = "core_concept_candidate"
_JCR_OBJECT_TASK = "object_or_task_candidate"
_JCR_TOOL_FW = "tool_or_framework_candidate"
_JCR_GENERIC_RISKY = "generic_risky_candidate"


def _lookup_coarse_for_graph_term(term: str, jd_coarse_roles: Dict[str, str]) -> Optional[str]:
    """图谱 term 对齐 JD 粗分：精确匹配或最长键包含关系。"""
    if not jd_coarse_roles:
        return None
    tl = (term or "").strip().lower()
    if not tl:
        return None
    if tl in jd_coarse_roles:
        v = jd_coarse_roles[tl]
        if v == _JCR_GENERIC_RISKY and _is_concrete_skill_phrase(tl):
            return _JCR_OBJECT_TASK if _is_task_like(tl) else _JCR_CORE
        return v
    best_k, best_v = None, None
    best_len = -1
    for k, v in jd_coarse_roles.items():
        if not k:
            continue
        if tl in k and len(k) > best_len:
            best_k, best_v, best_len = k, v, len(k)
        elif k in tl and len(k) > best_len:
            best_k, best_v, best_len = k, v, len(k)
    if best_v == _JCR_GENERIC_RISKY and _is_concrete_skill_phrase(tl):
        return _JCR_OBJECT_TASK if _is_task_like(tl) else _JCR_CORE
    return best_v


def _competition_tier_from_coarse(coarse: Optional[str]) -> int:
    """0=主竞争池(core/object_task)，1=辅/工具/泛化；与 Top-N 分层配额一致。"""
    if coarse in (_JCR_CORE, _JCR_OBJECT_TASK):
        return 0
    if coarse in (_JCR_TOOL_FW, _JCR_GENERIC_RISKY):
        return 1
    return 1


def _backbone_role_multiplier(coarse: Optional[str]) -> float:
    """在 backbone→spread 之前折损辅/泛锚，使旧 backbone 不再单独主导排序。"""
    if coarse in (_JCR_CORE, _JCR_OBJECT_TASK):
        return 1.0
    if coarse == _JCR_TOOL_FW:
        return 0.48
    if coarse == _JCR_GENERIC_RISKY:
        return 0.22
    return 0.55


def _build_term_meta_lower(anchor_skills: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    if not anchor_skills:
        return m
    for _vid, info in anchor_skills.items():
        t = (info.get("term") or "").strip().lower()
        if t and t not in m:
            m[t] = info
    return m


# 主线技术槽位（对象/任务链/真机 pipeline），用于 groundedness，非「坏词表」
_MOTION_MAINLINE_SLOTS = re.compile(
    r"(robot|robotic|manipulator|\barm\b|机械臂|机器人|双臂|真机|"
    r"motion|trajectory|trajectories|planning|\bplan\b|\bcontrol\b|controller|servo|"
    r"kinematic|kinematics|dynamics|estimation|"
    r"轨迹|路径|规划|控制|运动|运控|规控|全身|状态估计|动力学|运动学|"
    r"仿真到实机|sim-to-real|simulation-to-real|sim2real)",
    re.I,
)
# JD 中常见「加分/偏好」段标题（结构），用于侧向技术相对位置
_BONUS_OR_PREF_SECTION = re.compile(
    r"(加分项|加分\s*[:：]|优先\s*条件|优先\s*[:：]|nice\s*to\s*have|optional)",
    re.I,
)
_LEARNING_OR_RL_TOPIC_TAIL = re.compile(r"(学习|learning)$", re.I)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# 对象槽 / 任务链槽：用于判断「完整技能短语」而非具体岗位词表
_OBJECT_SLOT = re.compile(
    r"(机器人|机械臂|双臂|真机|robot|robotic|manipulator|\barm\b)",
    re.I,
)
_TASK_CHAIN_SLOT = re.compile(
    r"(运动控制|路径规划|轨迹规划|运动规划|全身控制|状态估计|动力学|运动学|运控|规控|"
    r"仿真到实机|motion|planning|trajectory|kinematic|kinematics|dynamics|estimation|\bcontrol\b)",
    re.I,
)
# 研究取向 / 范式尾词（侧向技术，非黑名单）
_RESEARCH_FLAVOR_TAIL = re.compile(r"(智能|范式|理论)$")
_HYPHEN_RESEARCH_STYLE = re.compile(r"-(based|centric|oriented|agnostic|driven)$", re.I)


def _has_cjk_chars(s: str) -> bool:
    return bool(s and re.search(r"[\u4e00-\u9fff]", s))


def _is_concrete_skill_phrase(term: str) -> bool:
    """
    结构判定：完整、可独立作主锚的技能短语（对象+任务 或 足够长且多槽位），
    用于避免「仅因子串被更长句包含」误判为 risky，也不靠点名具体词。
    """
    if not term or not str(term).strip():
        return False
    t = str(term).strip()
    tl = t.lower()
    cjk = len(_CJK_RE.findall(t))
    ntok = len(tl.split())
    if cjk >= 6:
        # 长中文但以学习/learning 结尾且无对象槽：偏研究侧向，不强制判「完整主技能短语」
        if _LEARNING_OR_RL_TOPIC_TAIL.search(tl) and not _OBJECT_SLOT.search(t):
            return False
        return True
    if cjk >= 4 and _OBJECT_SLOT.search(t) and _TASK_CHAIN_SLOT.search(t):
        return True
    # 典型任务链复合词（路径规划、运动控制 等），无对象词也可作为主技能短语
    if cjk >= 4 and _TASK_CHAIN_SLOT.search(t):
        return True
    if len(_MOTION_MAINLINE_SLOTS.findall(t)) >= 2:
        return True
    if cjk >= 5 and _MOTION_MAINLINE_SLOTS.search(t):
        return True
    if ntok >= 2 and not _has_cjk_chars(t) and _is_task_like(tl):
        return True
    return False


def _is_research_side_branch_term(term: str) -> bool:
    """侧向/研究风格：缺主线槽位时的学习尾、范式尾、hyphen 研究形容词等（结构）。"""
    if not term:
        return False
    t = term.strip()
    tl = t.lower()
    if _is_concrete_skill_phrase(t):
        return False
    if _has_mainline_grounding(t):
        return False
    cjk = len(_CJK_RE.findall(t))
    if _HYPHEN_RESEARCH_STYLE.search(tl):
        return True
    if _LEARNING_OR_RL_TOPIC_TAIL.search(tl) and cjk <= 8:
        return True
    if cjk <= 5 and _RESEARCH_FLAVOR_TAIL.search(t):
        return True
    if cjk >= 6 and _LEARNING_OR_RL_TOPIC_TAIL.search(tl):
        return True
    return False


def _has_mainline_grounding(term: str) -> bool:
    return bool(term and _MOTION_MAINLINE_SLOTS.search(term))


def _cjk_len(term: str) -> int:
    if not term:
        return 0
    return len(_CJK_RE.findall(term))


def _is_subsumed_by_longer_jd_phrase(term: str, all_phrases: List[str]) -> bool:
    """更长 JD 短语包含本词；完整技术短语不因「被包住」单独判废。"""
    if not term or not all_phrases:
        return False
    t = term.strip().lower()
    if _is_concrete_skill_phrase(t):
        return False
    for p in all_phrases:
        pl = (p or "").strip().lower()
        if not pl or pl == t or len(pl) <= len(t):
            continue
        if t in pl:
            return True
    if " " not in t and re.match(r"^[a-z]+$", t):
        for p in all_phrases:
            pl = (p or "").strip().lower()
            if len(pl) <= len(t) + 1:
                continue
            if re.search(r"(?:^|\s)" + re.escape(t) + r"(?:\s|$)", pl):
                return True
    return False


def _term_after_bonus_section(term: str, raw_text: str) -> bool:
    if not raw_text or not term:
        return False
    m = _BONUS_OR_PREF_SECTION.search(raw_text)
    if not m:
        return False
    pos = raw_text.lower().find(term.strip().lower())
    if pos < 0:
        return False
    return pos >= m.start()


def _compute_mainline_rerank_factors(
    term: str,
    raw_text: str,
    all_phrases: List[str],
    coarse: Optional[str],
    specificity: float,
) -> Dict[str, Any]:
    """
    tier0 内部：主线/具体性/侧向技术 的结构化重排因子（无具体词黑名单）。
    含 object_task_bonus / concrete_phrase_bonus / subsumed_risky_penalty 等可解释字段。
    """
    tl = (term or "").strip()
    tll = tl.lower()
    grounded = _has_mainline_grounding(tl)
    concrete = _is_concrete_skill_phrase(tl)
    research_side = _is_research_side_branch_term(tl)
    ntok = len(tll.split())
    cjk = _cjk_len(tl)
    sub_raw = _is_subsumed_by_longer_jd_phrase(tl, all_phrases)
    learning_tail = bool(_LEARNING_OR_RL_TOPIC_TAIL.search(tll))
    short_en_algo = bool(re.match(r"^[a-z]{2,5}$", tll))

    # 子串冗余：仅对非完整短语生效（可解释字段）
    subsumed_risky_penalty = 0.0
    if sub_raw and coarse in (_JCR_CORE, _JCR_OBJECT_TASK):
        subsumed_risky_penalty = 0.22 if not concrete else 0.0

    # --- generic_main_penalty：主池内偏泛；完整技能短语大幅减轻 ---
    g_pen = 0.0
    if coarse in (_JCR_CORE, _JCR_OBJECT_TASK) and not grounded:
        if concrete:
            g_pen = max(g_pen, 0.04)
        if cjk == 2:
            g_pen = max(g_pen, 0.42)
        elif cjk == 3:
            g_pen = max(g_pen, 0.26)
        elif cjk >= 4 and cjk <= 5 and learning_tail and not concrete:
            g_pen = max(g_pen, 0.28)
        if ntok == 1 and re.match(r"^[a-z]+$", tll) and len(tll) <= 9:
            g_pen = max(g_pen, 0.28)
        if learning_tail and cjk <= 8 and not concrete:
            g_pen = max(g_pen, 0.22)
        if sub_raw and not concrete:
            g_pen = max(g_pen, 0.16)
        if specificity < 0.52 and not concrete:
            g_pen = max(g_pen, 0.1)
        if short_en_algo and not grounded:
            g_pen = max(g_pen, 0.22)
    g_pen = min(0.48, g_pen)

    # 对象+任务绑定 bonus（分型前移的数值影子）
    object_task_bonus = 0.0
    if concrete or (_OBJECT_SLOT.search(tl) and _TASK_CHAIN_SLOT.search(tl)):
        object_task_bonus = min(0.18, 0.06 + 0.04 * min(3, len(_MOTION_MAINLINE_SLOTS.findall(tl))))

    # mainline_bonus：槽位命中
    ml_b = 0.0
    if grounded:
        n_hits = len(_MOTION_MAINLINE_SLOTS.findall(tl))
        ml_b = min(0.2, 0.055 * max(1, n_hits))
    ml_b = min(0.22, ml_b + object_task_bonus * 0.85)

    # concrete_phrase_bonus：具体短语优先（与 coarse 一致）
    concrete_phrase_bonus = 0.0
    if concrete:
        concrete_phrase_bonus = min(0.2, 0.08 + 0.025 * min(cjk, 10))
    c_b = concrete_phrase_bonus
    c_b += 0.1 * max(0.0, float(specificity) - 0.48)
    c_b += 0.028 * min(ntok, 5)
    if cjk >= 6:
        c_b += 0.04
    c_b = min(0.26, c_b)

    # --- side_branch_penalty：加分段 / 研究风格 / RL 话题无落地 ---
    sb_p = 0.0
    if _term_after_bonus_section(tl, raw_text):
        sb_p = max(sb_p, 0.28)
    if research_side:
        sb_p = max(sb_p, 0.32)
    if not grounded and learning_tail and not concrete:
        sb_p = max(sb_p, 0.26)
    if not grounded and short_en_algo:
        sb_p = max(sb_p, 0.22)
    if coarse in (_JCR_CORE, _JCR_OBJECT_TASK) and concrete:
        sb_p *= 0.45
    sb_p = min(0.48, sb_p)

    eff_g = min(0.5, g_pen + subsumed_risky_penalty * 0.85)

    composite_mult = (
        (1.0 - eff_g)
        * (1.0 + min(0.24, ml_b))
        * (1.0 - min(0.48, sb_p))
        * (1.0 + min(0.26, c_b))
    )
    return {
        "generic_main_penalty": g_pen,
        "subsumed_risky_penalty": subsumed_risky_penalty,
        "object_task_bonus": object_task_bonus,
        "mainline_bonus": ml_b,
        "concrete_phrase_bonus": concrete_phrase_bonus,
        "side_branch_penalty": sb_p,
        "concreteness_bonus": c_b,
        "composite_mult": composite_mult,
    }


def _select_tiered_anchor_rows(
    anchor_scored_rows: List[Dict[str, Any]],
    top_n: int,
    raw_text: str = "",
    all_phrases: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    主竞争池（tier0）先占满名额，不足再用 tier1 补位。
    tier0 内按 tier0_rerank_score 排序（在 score_after_role_penalty 上叠主线/具体性/侧向因子），
    避免「仿真」「强化学习」等泛主锚仅靠旧分排在最前。
    """
    phrases = list(all_phrases or [])
    for r in anchor_scored_rows:
        coarse = r.get("coarse_jd_role")
        spec = float(r.get("specificity") or 0.5)
        if _competition_tier_from_coarse(coarse) == 0:
            term = (r.get("term") or "").strip()
            fac = _compute_mainline_rerank_factors(term, raw_text, phrases, coarse, spec)
            r["generic_main_penalty"] = fac["generic_main_penalty"]
            r["subsumed_risky_penalty"] = fac["subsumed_risky_penalty"]
            r["object_task_bonus"] = fac["object_task_bonus"]
            r["mainline_bonus"] = fac["mainline_bonus"]
            r["concrete_phrase_bonus"] = fac["concrete_phrase_bonus"]
            r["side_branch_penalty"] = fac["side_branch_penalty"]
            r["concreteness_bonus"] = fac["concreteness_bonus"]
            r["composite_mult"] = fac["composite_mult"]
            base = float(r.get("score_after_role_penalty") or 0.0)
            r["tier0_rerank_score"] = base * fac["composite_mult"]
        else:
            r["generic_main_penalty"] = 0.0
            r["subsumed_risky_penalty"] = 0.0
            r["object_task_bonus"] = 0.0
            r["mainline_bonus"] = 0.0
            r["concrete_phrase_bonus"] = 0.0
            r["side_branch_penalty"] = 0.0
            r["concreteness_bonus"] = 0.0
            r["composite_mult"] = 1.0
            r["tier0_rerank_score"] = float(r.get("score_after_role_penalty") or 0.0)

    tier0 = [r for r in anchor_scored_rows if _competition_tier_from_coarse(r.get("coarse_jd_role")) == 0]
    tier1 = [r for r in anchor_scored_rows if _competition_tier_from_coarse(r.get("coarse_jd_role")) == 1]
    tier0.sort(key=lambda x: -float(x.get("tier0_rerank_score") or 0.0))
    tier1.sort(key=lambda x: -float(x.get("tier0_rerank_score") or 0.0))
    out = tier0[:top_n]
    if len(out) < top_n:
        out.extend(tier1[: max(0, top_n - len(out))])
    return out


def compute_phrase_specificity(phrase: str, all_phrases: List[str]) -> float:
    """
    短语特异性：越具体、越少被其它长短语包含则越高。无词表。
    """
    if not phrase or not all_phrases:
        return 0.5
    phrase = phrase.strip().lower()
    if not phrase:
        return 0.5
    n_tokens = len(phrase.split())
    # 被多少其它（更长或等长）短语包含
    num_containing = 0
    for p in all_phrases:
        if not p or p.strip().lower() == phrase:
            continue
        p_lower = p.strip().lower()
        if len(p_lower) >= len(phrase) and phrase in p_lower:
            num_containing += 1
    total = max(len(all_phrases), 1)
    contain_ratio = num_containing / total
    # 越少被包含、token 越多，特异性越高
    spec = 0.3 + 0.15 * min(n_tokens, 6) + 0.55 * max(0, 1.0 - contain_ratio)
    return min(1.0, max(0.0, spec))


def compute_phrase_context_richness(phrase: str, raw_text: str, cleaned_terms: List[str]) -> float:
    """
    上下文丰富度：该短语在 JD 中所在窗口内，与多少其它技能短语共现。无词表。
    """
    if not phrase or not raw_text or not cleaned_terms:
        return 0.5
    phrase_lower = phrase.strip().lower()
    text_lower = raw_text.lower()
    pos = text_lower.find(phrase_lower)
    if pos < 0:
        return 0.3
    start = max(0, pos - CONTEXT_WINDOW_CHAR)
    end = min(len(raw_text), pos + len(phrase) + CONTEXT_WINDOW_CHAR)
    window = raw_text[start:end].lower()
    count = 0
    for t in cleaned_terms:
        if not t or t.strip().lower() == phrase_lower:
            continue
        if t.strip().lower() in window:
            count += 1
    return min(1.0, max(0.0, 0.2 + 0.8 * min(count / 5.0, 1.0)))


def compute_anchor_taskness(phrase: str, raw_text: str, cleaned_terms: List[str]) -> float:
    """
    任务性：与上下文丰富度同源，描述“是否处于任务描述密集区”。无词表。
    """
    return compute_phrase_context_richness(phrase, raw_text, cleaned_terms)


def compute_local_phrase_cluster_support(phrase: str, raw_text: str, cleaned_terms: List[str]) -> float:
    """
    局部短语簇支持：与 context_richness 同源，表示相邻技能短语支持度。无词表。
    """
    return compute_phrase_context_richness(phrase, raw_text, cleaned_terms)


def _print_stage1_step2_anchor_collapse_audit(
    rows: List[Dict[str, Any]], label: Any = None, limit: int = 28
) -> None:
    """窄表：短语支撑均值 + 相对 backbone 的塌陷比，解释锚点序为何偏。"""
    if not ANCHOR_COLLAPSE_AUDIT or not rows:
        return
    if label is not None and (
        not getattr(label, "verbose", False) or getattr(label, "silent", False)
    ):
        return
    print(f"\n{'-' * 80}\n[Stage1-Step2 anchor collapse audit] JD 工业锚 · 凸组合 final（非四连乘）\n{'-' * 80}")
    print(
        f"  公式: final = bb × ({ANCHOR_FINAL_BACKBONE_FLOOR_FRAC:.2f} + "
        f"{ANCHOR_FINAL_SUPPORT_SPREAD_FRAC:.2f} × support_mean); "
        f"support_mean = {ANCHOR_SUPPORT_W_SPEC:.2f}·spec+{ANCHOR_SUPPORT_W_CTX:.2f}·ctx+"
        f"{ANCHOR_SUPPORT_W_TASK:.2f}·task+{ANCHOR_SUPPORT_W_LOCAL:.2f}·local"
    )
    hdr = "rk | anchor           | bb    | spec | ctx  | task | loc  | smean | s_post | cratio"
    print(hdr)
    print("-" * len(hdr))
    for i, row in enumerate(rows[:limit], start=1):
        t = (row.get("term") or "")[:16]
        print(
            f"{i:2d} | {t:16} | {row.get('backbone_score', 0):5.2f} | "
            f"{row.get('specificity', 0):4.2f} | {row.get('context_richness', 0):4.2f} | "
            f"{row.get('taskness', 0):4.2f} | {row.get('local_cluster_support', 0):4.2f} | "
            f"{row.get('support_mean', 0):5.2f} | {row.get('score_after_role_penalty', row.get('final_anchor_score', 0)):5.2f} | "
            f"{row.get('collapse_ratio', 0):5.2f}"
        )


def extract_anchor_skills(label, target_job_ids, query_vector=None, total_j=None) -> Dict[str, Any]:
    """
    复用 LabelRecallPath._extract_anchor_skills 的逻辑（移动到独立模块，便于后续拆分）。
    label 需提供：graph, total_job_count, vocab_to_idx, all_vocab_vectors, debug_info, 以及相关阈值常量。
    """
    total_j = float(total_j or 0)
    if total_j <= 0:
        total_j = label.total_job_count

    # 优先使用基于当前 JD 文本抽取的技能短语集合（由 stage1_domain_anchors 预先挂载），
    # 保证锚点过滤与本次查询语境强绑定；若不存在则回退到岗位 skills 字段的清洗结果。
    cleaned_terms = getattr(label, "_jd_cleaned_terms", None)
    if cleaned_terms:
        cleaned_terms = {str(t).lower() for t in cleaned_terms}
    else:
        cleaned_terms = set()
        try:
            cursor = label.graph.run(
                "MATCH (j:Job) WHERE j.id IN $j_ids RETURN j.skills AS skills",
                j_ids=target_job_ids[: label.ANCHOR_JOBS_TOP_K],
            )
            for row in cursor:
                if row.get("skills"):
                    cleaned_terms |= clean_job_skills(str(row["skills"]))
        except Exception:
            cleaned_terms = set()
        if not cleaned_terms:
            cleaned_terms = None

    debug_print(1, "\n" + "-" * 80 + "\n[Step2] 锚点选择\n" + "-" * 80, label)
    sample_cleaned = list(cleaned_terms)[:50] if cleaned_terms else []
    if getattr(label, "verbose", False):
        print(f"[Step2 Debug] JD 清洗后技能短语样本({len(sample_cleaned)}): {sample_cleaned}")

    cypher1 = """
    MATCH (j:Job) WHERE j.id IN $j_ids
    MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
    WITH v.id AS vid, v.term AS term, count(DISTINCT j.id) AS job_freq
    RETURN vid, term, job_freq
    ORDER BY job_freq DESC
    """
    rows = []
    try:
        for r in label.graph.run(cypher1, j_ids=target_job_ids[: label.ANCHOR_JOBS_TOP_K]):
            if r.get("term") and len(str(r.get("term") or "")) > 1:
                rows.append(dict(r))
    except Exception:
        rows = []

    if not rows:
        stats = {
            "before_melt": 0,
            "after_melt": 0,
            "after_top30": 0,
            "melted_sample": [],
            "jd_cleaned_terms_sample": list(cleaned_terms)[:50] if cleaned_terms else [],
        }
        label.debug_info.anchor_melt_stats = stats
        label._last_anchor_melt_stats = stats
        return {}

    v_ids = list({int(r["vid"]) for r in rows})
    global_count = {}
    try:
        cypher2 = """
        UNWIND $v_ids AS vid
        MATCH (v:Vocabulary {id: vid})<-[:REQUIRE_SKILL]-(j:Job)
        RETURN vid, count(j) AS cnt
        """
        for r in label.graph.run(cypher2, v_ids=v_ids):
            global_count[int(r["vid"])] = int(r.get("cnt") or 0)
    except Exception:
        pass

    # 额外加载锚点候选的领域跨度（domain_span），用于对“长尾高纯度技术词”做统计型保活
    stats_span: dict[int, int] = {}
    try:
        if v_ids:
            ph = ",".join("?" * len(v_ids))
            rows_stats = label.stats_conn.execute(
                f"SELECT voc_id, domain_span FROM vocabulary_domain_stats WHERE voc_id IN ({ph})",
                v_ids,
            ).fetchall()
            for s in rows_stats:
                stats_span[int(s[0])] = int(s[1] or 0)
    except Exception:
        stats_span = {}

    melt_threshold = float(label.ANCHOR_MELT_COV_J)
    terms_before_melt = [r.get("term") or "" for r in rows]
    melted_terms: List[Tuple[str, float]] = []
    dropped_terms: List[Tuple[str, str, Any]] = []

    # ---------- 层1：噪声硬过滤 ----------
    # 只丢弃：过短、极端全行业泛词(cov_j >= melt_threshold)
    n_original_rows = len(rows)
    candidates_for_score: List[Tuple[Dict, float, int | None, bool, bool]] = []
    for r in rows:
        vid = int(r.get("vid"))
        g = global_count.get(vid, 0)
        cov_j = (g / total_j) if total_j else 0
        term_text = (r.get("term") or "").strip()
        if len(term_text) <= 1:
            dropped_terms.append((term_text, "length", len(term_text)))
            continue
        # Stage1 早期粗分型：噪声/偏好类 JD 短语不进入 REQUIRE_SKILL 主候选池（由 recall._jd_coarse_roles 注入）
        jd_coarse_roles = getattr(label, "_jd_coarse_roles", None) or {}
        if jd_coarse_roles:
            cr = jd_coarse_roles.get(term_text.lower().strip())
            if cr in ("noise_fragment_candidate", "venue_or_preference_candidate"):
                dropped_terms.append((term_text, "jd_coarse_role", cr))
                continue
        if cov_j >= melt_threshold:
            melted_terms.append((term_text, round(cov_j, 4)))
            dropped_terms.append((term_text, "cov_j", round(cov_j, 4)))
            continue
        job_freq = int(r.get("job_freq") or 0)
        span = stats_span.get(vid)
        in_jd = cleaned_terms is not None and _in_jd_context(term_text, cleaned_terms)
        task_like = _is_task_like(term_text)
        candidates_for_score.append((r, cov_j, span, in_jd, task_like))

    # ---------- 层0：保底标记 ----------
    # 满足 in_jd + is_task_like + 非超级泛词(已过层1) 的 term 记为保底，后续 backbone_score 给下限
    def _is_baodi(in_jd: bool, task_like: bool) -> bool:
        return bool(in_jd and task_like)

    # ---------- 层2：统一 backbone_score，一次排序 + 一次 TopN ----------
    q = np.asarray(query_vector, dtype=np.float32).flatten() if query_vector is not None else None
    scored: List[Tuple[float, Dict]] = []
    for r, cov_j, span, in_jd, task_like in candidates_for_score:
        job_freq = int(r.get("job_freq") or 0)
        span_val = span if span is not None else 10
        sim = 0.0
        if q is not None and q.size > 0:
            vid_str = str(r.get("vid"))
            idx = label.vocab_to_idx.get(vid_str)
            if idx is not None:
                try:
                    sim = float(np.dot(label.all_vocab_vectors[idx], q))
                except Exception:
                    pass
        score = (
            BACKBONE_W_JOB_FREQ * math.log(1 + job_freq)
            + BACKBONE_W_COV_J * (1.0 - min(1.0, cov_j))
            + BACKBONE_W_SPAN * (1.0 / (1 + span_val))
            + (BACKBONE_W_IN_JD if in_jd else 0.0)
            + (BACKBONE_W_TASK_LIKE if task_like else 0.0)
            + BACKBONE_W_SIM * max(0.0, sim)
        )
        if _is_baodi(in_jd, task_like):
            score = max(score, BACKBONE_FLOOR_FOR_BAODI)
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    raw_text = (getattr(label, "_jd_raw_text", None) or "") or ""
    all_phrases = list(cleaned_terms) if cleaned_terms else []
    jd_coarse_roles = getattr(label, "_jd_coarse_roles", None) or {}
    anchor_scored_rows: List[Dict[str, Any]] = []
    for bb_score, r in scored:
        term = (r.get("term") or "").strip()
        coarse = _lookup_coarse_for_graph_term(term, jd_coarse_roles)
        if not coarse:
            coarse = (
                _JCR_OBJECT_TASK
                if (_is_concrete_skill_phrase(term) or _is_task_like(term))
                else _JCR_CORE
            )
        bb_mult = _backbone_role_multiplier(coarse)
        bb_eff = float(bb_score) * bb_mult
        spec = ctx = task = local = 0.5
        if raw_text and all_phrases and term:
            spec = compute_phrase_specificity(term, all_phrases)
            ctx = compute_phrase_context_richness(term, raw_text, all_phrases)
            task = compute_anchor_taskness(term, raw_text, all_phrases)
            local = compute_local_phrase_cluster_support(term, raw_text, all_phrases)
        # 凸组合支撑均值 + backbone 底座：避免 (0.2+0.8·spec)×… 四连乘把主线锚整体打穿
        support_mean = (
            ANCHOR_SUPPORT_W_SPEC * spec
            + ANCHOR_SUPPORT_W_CTX * ctx
            + ANCHOR_SUPPORT_W_TASK * task
            + ANCHOR_SUPPORT_W_LOCAL * local
        )
        spread = (
            ANCHOR_FINAL_BACKBONE_FLOOR_FRAC
            + ANCHOR_FINAL_SUPPORT_SPREAD_FRAC * support_mean
        )
        score_before_role_penalty = float(bb_score) * spread
        score_after_role_penalty = float(bb_eff) * spread
        collapse_ratio = score_after_role_penalty / max(bb_score, 1e-6)
        anchor_scored_rows.append({
            "tid": r.get("vid"),
            "term": term or r.get("term"),
            "backbone_score": bb_score,
            "coarse_jd_role": coarse,
            "specificity": spec,
            "context_richness": ctx,
            "taskness": task,
            "local_cluster_support": local,
            "support_mean": support_mean,
            "support_spread": spread,
            "collapse_ratio": collapse_ratio,
            "score_before_role_penalty": score_before_role_penalty,
            "score_after_role_penalty": score_after_role_penalty,
            # tier0_rerank_score / final 在 _select_tiered_anchor_rows 内写入
            "final_anchor_score": score_after_role_penalty,
            "_row": r,
        })
    # 先按角色分层键排序，再分层取 Top-N（避免 aux/risky 仅靠高分插队）
    anchor_scored_rows.sort(
        key=lambda x: (
            _competition_tier_from_coarse(x.get("coarse_jd_role")),
            -float(x.get("score_after_role_penalty") or 0.0),
        )
    )
    _print_stage1_step2_anchor_collapse_audit(anchor_scored_rows, label)
    top_n = int(getattr(label, "ANCHOR_FINAL_TOP_K", 20))
    tiered_pick = _select_tiered_anchor_rows(anchor_scored_rows, top_n, raw_text, all_phrases)
    for row in anchor_scored_rows:
        tr = row.get("tier0_rerank_score")
        if tr is not None:
            row["final_anchor_score"] = float(tr)
    picked_set = {id(x) for x in tiered_pick}
    borderline_rejected = [x for x in anchor_scored_rows if id(x) not in picked_set][: top_n + 10]
    rows = [x["_row"] for x in tiered_pick]

    terms_after_melt = [r.get("term") or "" for r, *_ in candidates_for_score]
    stats = {
        "before_melt": n_original_rows,
        "after_layer1": len(candidates_for_score),
        "melted_sample": melted_terms[:25],
        "melt_threshold": melt_threshold,
        "terms_before_melt": terms_before_melt,
        "terms_after_melt": terms_after_melt,
        "cleaned_terms_sample": list(cleaned_terms)[:50] if cleaned_terms else [],
        "dropped_terms": dropped_terms[:50],
        "after_topN": len(rows),
        "terms_after_topN": [r.get("term") or "" for r in rows],
        "tiered_topN_used": True,
        "anchor_scored_rows": anchor_scored_rows[:50],
        "borderline_rejected": borderline_rejected,
    }
    label.debug_info.anchor_melt_stats = stats
    label._last_anchor_melt_stats = stats

    n_cov = sum(1 for _, reason, _ in dropped_terms if reason == "cov_j")
    n_len = sum(1 for _, reason, _ in dropped_terms if reason == "length")
    debug_print(1, f"[Step2] cleaned_skills 总数={len(all_phrases) or 0}", label)
    debug_print(1, f"[Step2] REQUIRE_SKILL 原始 rows={n_original_rows}，层1过滤后候选={len(candidates_for_score)}；cov_j 砍掉 {n_cov} 个，length 砍掉 {n_len} 个", label)
    debug_print(3, "[Step2 Reject Samples] term | reason | value", label)
    for term, reason, value in dropped_terms[:15]:
        debug_print(3, f"  {term[:30]} | {reason} | {value}", label)

    debug_print(2, "[Step2 Anchor Score Breakdown] term | bb | spec | ctx | task | loc | smean | s_pre | s_post | cr | rk", label)
    for i, row in enumerate(anchor_scored_rows[:20], 1):
        t = (row.get("term") or "")[:22]
        debug_print(2, (
            f"  {i:>2} {t:<22} | "
            f"bb={row.get('backbone_score', 0):.3f} | "
            f"spec={row.get('specificity', 0):.3f} | "
            f"ctx={row.get('context_richness', 0):.3f} | "
            f"task={row.get('taskness', 0):.3f} | "
            f"loc={row.get('local_cluster_support', 0):.3f} | "
            f"sm={row.get('support_mean', 0):.3f} | "
            f"s_pre={row.get('score_before_role_penalty', 0):.3f} | "
            f"s_post={row.get('score_after_role_penalty', 0):.3f} | "
            f"cr={row.get('collapse_ratio', 0):.3f} | "
            f"rk={i}"
        ), label)

    debug_print(2, "[Step2 Borderline Rejected] term | final | reason", label)
    for row in borderline_rejected[:10]:
        t = (row.get("term") or "")[:30]
        debug_print(2, f"  {t:<30} | {row.get('score_after_role_penalty', 0):.3f} | tier_or_quota_cutoff", label)

    if not rows:
        debug_print(1, "[Step2] 层2 排序+TopN 后无锚点可用。", label)
        return {}

    all_phrases_for_spec = list(cleaned_terms) if cleaned_terms else []
    anchors = {}
    for i, x in enumerate(tiered_pick, 1):
        r = x["_row"]
        vid_str = str(r.get("vid"))
        term = r.get("term")
        spec = x.get("specificity", compute_phrase_specificity(term or "", all_phrases_for_spec) if all_phrases_for_spec else 0.5)
        anchors[vid_str] = {
            "term": term,
            "specificity": spec,
            "coarse_jd_role": x.get("coarse_jd_role"),
            "final_anchor_score": x.get("tier0_rerank_score", x.get("final_anchor_score")),
            "tier0_rerank_score": x.get("tier0_rerank_score"),
            "score_before_role_penalty": x.get("score_before_role_penalty"),
            "score_after_role_penalty": x.get("score_after_role_penalty"),
            "generic_main_penalty": x.get("generic_main_penalty"),
            "subsumed_risky_penalty": x.get("subsumed_risky_penalty"),
            "object_task_bonus": x.get("object_task_bonus"),
            "mainline_bonus": x.get("mainline_bonus"),
            "concrete_phrase_bonus": x.get("concrete_phrase_bonus"),
            "side_branch_penalty": x.get("side_branch_penalty"),
            "concreteness_bonus": x.get("concreteness_bonus"),
            "composite_mult": x.get("composite_mult"),
            "backbone_score": x.get("backbone_score"),
            "context_richness": x.get("context_richness"),
            "taskness": x.get("taskness"),
            "local_cluster_support": x.get("local_cluster_support"),
            "support_mean": x.get("support_mean"),
            "collapse_ratio": x.get("collapse_ratio"),
            "anchor_source": "skill_direct",
            "anchor_source_weight": 1.0,
        }

    debug_print(1, f"[Step2] 最终 industrial anchors 数量: {len(anchors)}", label)
    debug_print(2, "[Step2 Final] tid | term | s_post | tier0 | g_pen | sub | ot | side | conc | final", label)
    for vid_str, info in list(anchors.items())[:30]:
        t = (info.get("term") or "")[:30]
        debug_print(2, (
            f"  {vid_str} | {t:<20} | {info.get('score_after_role_penalty', 0):.3f} | "
            f"{info.get('tier0_rerank_score', info.get('final_anchor_score', 0)):.3f} | "
            f"{info.get('generic_main_penalty', 0):.2f} | {info.get('subsumed_risky_penalty', 0):.2f} | "
            f"{info.get('object_task_bonus', 0):.2f} | {info.get('side_branch_penalty', 0):.2f} | "
            f"{info.get('concreteness_bonus', 0):.2f} | {info.get('final_anchor_score', 0):.3f}"
        ), label)
    return anchors


# ---------- 条件化锚点表示（带 JD 上下文，无词表） ----------
LOCAL_CONTEXT_WINDOW = 100
CO_ANCHOR_TOP_K = 5
# 主线优先：conditioned 仅轻度偏移，降低 JD 整体向量比重，避免 ctx 漂到错误义项
CONDITIONED_W_ANCHOR_HIGH_SPEC = 0.68
CONDITIONED_W_LOCAL_HIGH_SPEC = 0.17
CONDITIONED_W_CO_HIGH_SPEC = 0.10
CONDITIONED_W_JD_HIGH_SPEC = 0.05
CONDITIONED_W_ANCHOR_LOW_SPEC = 0.68
CONDITIONED_W_LOCAL_LOW_SPEC = 0.17
CONDITIONED_W_CO_LOW_SPEC = 0.10
CONDITIONED_W_JD_LOW_SPEC = 0.05


def build_anchor_local_context(
    anchor_term: str,
    raw_text: str,
    cleaned_terms: List[str],
    window: int = LOCAL_CONTEXT_WINDOW,
    anchor_skills: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    锚点在 JD 中的局部短语窗口（前后各 window 字符内出现的其它技能短语）。
    若传入 anchor_skills（已含 coarse_jd_role / is_context_only），则过滤：
    context_only 不参与；两枚 generic_risky 互不抬高。
    """
    if not anchor_term or not raw_text:
        return []
    anchor_lower = anchor_term.strip().lower()
    text_lower = raw_text.lower()
    pos = text_lower.find(anchor_lower)
    if pos < 0:
        return []
    start = max(0, pos - window)
    end = min(len(raw_text), pos + len(anchor_term) + window)
    snippet = raw_text[start:end].lower()
    meta_map = _build_term_meta_lower(anchor_skills)
    cur_meta = meta_map.get(anchor_lower) if anchor_skills else None
    cur_gr = (cur_meta or {}).get("coarse_jd_role") == _JCR_GENERIC_RISKY
    out = []
    for t in cleaned_terms:
        if not t or t.strip().lower() == anchor_lower:
            continue
        tl = t.strip().lower()
        if tl not in snippet:
            continue
        om = meta_map.get(tl) if anchor_skills else None
        if (om or {}).get("is_context_only"):
            continue
        if cur_gr and (om or {}).get("coarse_jd_role") == _JCR_GENERIC_RISKY:
            continue
        out.append(t.strip())
    return out[:10]


def collect_co_anchor_terms(
    anchor_term: str,
    selected_anchor_terms: List[str],
    raw_text: str,
    top_k: int = CO_ANCHOR_TOP_K,
    anchor_skills: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    与当前锚点在 JD 中共现的其它锚点词（同窗口内出现）。
    角色过滤策略同 build_anchor_local_context。
    """
    if not anchor_term or not raw_text or not selected_anchor_terms:
        return []
    anchor_lower = anchor_term.strip().lower()
    text_lower = raw_text.lower()
    pos = text_lower.find(anchor_lower)
    if pos < 0:
        return []
    start = max(0, pos - LOCAL_CONTEXT_WINDOW)
    end = min(len(raw_text), pos + len(anchor_term) + LOCAL_CONTEXT_WINDOW)
    snippet = raw_text[start:end].lower()
    meta_map = _build_term_meta_lower(anchor_skills)
    cur_meta = meta_map.get(anchor_lower) if anchor_skills else None
    cur_gr = (cur_meta or {}).get("coarse_jd_role") == _JCR_GENERIC_RISKY
    out = []
    for t in selected_anchor_terms:
        if not t or t.strip().lower() == anchor_lower:
            continue
        tl = t.strip().lower()
        if tl not in snippet:
            continue
        om = meta_map.get(tl) if anchor_skills else None
        if (om or {}).get("is_context_only"):
            continue
        if cur_gr and (om or {}).get("coarse_jd_role") == _JCR_GENERIC_RISKY:
            continue
        out.append(t.strip())
    return out[:top_k]


def encode_text_with_optional_cache(
    encoder: Any, text: str, encode_cache: Optional[Dict[str, np.ndarray]]
) -> Optional[np.ndarray]:
    """与 encoder.encode(text) 同语义；encode_cache 命中时复用向量（无损）。"""
    if not text:
        return None
    if encode_cache is not None and hasattr(encoder, "lookup_or_encode"):
        return encoder.lookup_or_encode(text, encode_cache)
    v, _ = encoder.encode(text)
    return v


# S1 anchor_ctx：单次 encode_batch 过大时按块切分；每行仍走 QueryEncoder.encode_batch 路径，
# 与「把 unique_raws 一次性传入 encode_batch」在 SentenceTransformer 下逐样本一致（无跨样本交互）。
_ANCHOR_CTX_PREFILL_ENCODE_CHUNK = 128


def prefill_encode_cache_for_anchor_ctx(
    encoder: Any,
    anchor_skills: Dict[str, Any],
    raw_text: str,
    encode_cache: Dict[str, np.ndarray],
) -> None:
    """
    在 build_conditioned_anchor_representation 循环前，收集本 JD 下所有待编码短文本，
    按原文去重后 encode_batch 写入 encode_cache（键与 lookup_or_encode 一致）。
    数值与逐条 encode/lookup_or_encode 等价，显著减少 S1 anchor_ctx 的模型前向次数。

    注意：encode_cache 允许为空 dict（例如 JD 片段编码失败时）；仍会预填锚点/局部/共现串，
    不得因「缓存尚空」整段跳过，否则退化为每锚点多次 forward。
    """
    if encode_cache is None or not raw_text or not anchor_skills or encoder is None:
        return
    if not hasattr(encoder, "encode_batch"):
        return

    cleaned_list = [info.get("term") or "" for info in (anchor_skills or {}).values() if info.get("term")]
    raws_ordered: List[str] = []
    if raw_text:
        jd_head = canonical_jd_text_for_encode(raw_text)
        if jd_head:
            # 已在 Stage1 单次编码写入缓存时跳过 batch，避免 JD 再进 encode_batch
            if jd_head not in encode_cache:
                raws_ordered.append(jd_head)

    for _vid, info in anchor_skills.items():
        term = (info.get("term") or "").strip()
        if not term:
            continue
        raws_ordered.append(term[:200])
        other_terms = [
            info2.get("term") or ""
            for _vid2, info2 in (anchor_skills or {}).items()
            if str(info2.get("term", "")).strip().lower() != term.strip().lower()
        ]
        local_phrases = build_anchor_local_context(term, raw_text, cleaned_list, anchor_skills=anchor_skills)
        co_terms = collect_co_anchor_terms(term, other_terms, raw_text, anchor_skills=anchor_skills)
        if local_phrases:
            raws_ordered.append(" ; ".join(local_phrases[:5])[:300])
        if co_terms:
            raws_ordered.append(" ; ".join(co_terms[:5])[:300])

    # 按原文去重，保持首次出现顺序（与逐条 encode 的缓存键一致）
    unique_raws: List[str] = []
    seen_enh: Set[str] = set()
    for raw in raws_ordered:
        if not raw:
            continue
        if raw in seen_enh:
            continue
        seen_enh.add(raw)
        unique_raws.append(raw)

    if not unique_raws:
        return

    chunk_sz = max(1, int(_ANCHOR_CTX_PREFILL_ENCODE_CHUNK))
    for off in range(0, len(unique_raws), chunk_sz):
        chunk = unique_raws[off : off + chunk_sz]
        batch = encoder.encode_batch(chunk)
        if batch.shape[0] != len(chunk):
            return
        for i, raw in enumerate(chunk):
            row = np.asarray(batch[i], dtype=np.float32).reshape(1, -1).copy()
            encode_cache[raw] = row


def build_conditioned_anchor_representation(
    anchor_term: str,
    anchor_info: Dict[str, Any],
    anchor_skills: Dict[str, Any],
    raw_text: str,
    encoder: Any,
    jd_vec_precomputed: Optional[np.ndarray] = None,
    encode_cache: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    构造条件化锚点表示：anchor_vec + local_phrase_vec + co_anchor_vec + jd_vec 加权组合。
    泛锚点（specificity 低）更多依赖上下文，具体锚点更多依赖自身。无词表。
    """
    out: Dict[str, Any] = {
        "anchor_vec": None,
        "local_phrase_vec": None,
        "co_anchor_vec": None,
        "jd_vec": None,
        "conditioned_vec": None,
        "local_phrases": [],
        "co_anchor_terms": [],
    }
    if not anchor_term or not raw_text or encoder is None:
        return out
    cleaned_list = [info.get("term") or "" for info in (anchor_skills or {}).values() if info.get("term")]
    try:
        v_anchor_t = encode_text_with_optional_cache(encoder, anchor_term.strip()[:200], encode_cache)
        if v_anchor_t is None:
            return out
        v_anchor = np.asarray(v_anchor_t, dtype=np.float32).flatten()
    except Exception:
        return out
    if jd_vec_precomputed is not None:
        v_jd = np.asarray(jd_vec_precomputed, dtype=np.float32).flatten()
    else:
        v_jd_t = encode_text_with_optional_cache(
            encoder, canonical_jd_text_for_encode(raw_text), encode_cache
        )
        if v_jd_t is None:
            v_jd = v_anchor
        else:
            v_jd = np.asarray(v_jd_t, dtype=np.float32).flatten()
    local_phrases = build_anchor_local_context(anchor_term, raw_text, cleaned_list, anchor_skills=anchor_skills)
    other_terms = [info.get("term") or "" for vid, info in (anchor_skills or {}).items() if str(info.get("term", "")).strip().lower() != anchor_term.strip().lower()]
    co_terms = collect_co_anchor_terms(anchor_term, other_terms, raw_text, anchor_skills=anchor_skills)
    out["local_phrases"] = local_phrases
    out["co_anchor_terms"] = co_terms
    try:
        if local_phrases:
            text_local = " ; ".join(local_phrases[:5])
            v_local = encode_text_with_optional_cache(encoder, text_local[:300], encode_cache)
            out["local_phrase_vec"] = (
                np.asarray(v_local, dtype=np.float32).flatten() if v_local is not None else v_anchor.copy()
            )
        else:
            out["local_phrase_vec"] = v_anchor.copy()
        if co_terms:
            text_co = " ; ".join(co_terms[:5])
            v_co = encode_text_with_optional_cache(encoder, text_co[:300], encode_cache)
            out["co_anchor_vec"] = (
                np.asarray(v_co, dtype=np.float32).flatten() if v_co is not None else v_anchor.copy()
            )
        else:
            out["co_anchor_vec"] = v_anchor.copy()
    except Exception:
        out["local_phrase_vec"] = v_anchor.copy()
        out["co_anchor_vec"] = v_anchor.copy()
    out["anchor_vec"] = v_anchor
    out["jd_vec"] = v_jd
    local_strength = min(1.0, len(local_phrases) / 5.0) if local_phrases else 0.0
    co_strength = min(1.0, len(co_terms) / 5.0) if co_terms else 0.0
    if local_strength < 0.2 and co_strength < 0.2:
        conditioned = np.asarray(v_anchor, dtype=np.float32).flatten()
        norm = np.linalg.norm(conditioned)
        if norm > 1e-9:
            conditioned = conditioned / norm
        out["conditioned_vec"] = conditioned
        out["w_anchor"], out["w_local"], out["w_co"], out["w_jd"] = 1.0, 0.0, 0.0, 0.0
        return out
    specificity = float(anchor_info.get("specificity", 0.6))
    if specificity >= 0.8:
        w_a, w_l, w_c, w_j = CONDITIONED_W_ANCHOR_HIGH_SPEC, CONDITIONED_W_LOCAL_HIGH_SPEC, CONDITIONED_W_CO_HIGH_SPEC, CONDITIONED_W_JD_HIGH_SPEC
    else:
        w_a, w_l, w_c, w_j = CONDITIONED_W_ANCHOR_LOW_SPEC, CONDITIONED_W_LOCAL_LOW_SPEC, CONDITIONED_W_CO_LOW_SPEC, CONDITIONED_W_JD_LOW_SPEC
    out["w_anchor"], out["w_local"], out["w_co"], out["w_jd"] = w_a, w_l, w_c, w_j
    conditioned = w_a * v_anchor + w_l * out["local_phrase_vec"] + w_c * out["co_anchor_vec"] + w_j * v_jd
    norm = np.linalg.norm(conditioned)
    if norm > 1e-9:
        conditioned = conditioned / norm
    out["conditioned_vec"] = conditioned
    return out


def supplement_anchors_from_jd_vector(
    label,
    query_text,
    anchor_skills,
    total_j=None,
    top_k=None,
    active_domain_ids=None,
    jd_encode_cache: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """
    复用 LabelRecallPath._supplement_anchors_from_jd_vector 的逻辑（移动到独立模块）。
    label 需提供：vocab_index/stats_conn/graph/_query_encoder/_load_vocab_meta/_vocab_meta 以及阈值常量。
    jd_encode_cache：可选，若已含 canonical_jd_text_for_encode(query_text) 的共振键则复用向量，避免重复 JD forward。
    """
    if not query_text or not getattr(label, "vocab_index", None):
        label.debug_info.supplement_anchors = []
        label.debug_info.supplement_anchors_report = []
        label._last_supplement_anchors = label.debug_info.supplement_anchors
        label._last_supplement_anchors_report = label.debug_info.supplement_anchors_report
        return

    total_j = float(total_j or 0) or label.total_job_count
    label._load_vocab_meta()
    encoder = label._query_encoder
    # 与 anchor_ctx 统一为 canonical_jd_text_for_encode（前 800、不 strip），与单次编码缓存对齐
    jd_snippet = canonical_jd_text_for_encode(query_text)
    if getattr(label, "verbose", False):
        print(f"[Bridge Debug] semantic_query_text 片段: {jd_snippet[:120]}")
    if not jd_snippet:
        label.debug_info.supplement_anchors = []
        label.debug_info.supplement_anchors_report = []
        label._last_supplement_anchors = label.debug_info.supplement_anchors
        label._last_supplement_anchors_report = label.debug_info.supplement_anchors_report
        return

    # 与 anchor_ctx 共用共振缓存键时走 lookup_or_encode，否则单独 encode（数值与未缓存时一致）
    if jd_encode_cache is not None and hasattr(encoder, "lookup_or_encode"):
        v_jd = encoder.lookup_or_encode(jd_snippet, jd_encode_cache)
    else:
        v_jd, _ = encoder.encode(jd_snippet)
    if v_jd is None:
        label.debug_info.supplement_anchors = []
        label.debug_info.supplement_anchors_report = []
        label._last_supplement_anchors = label.debug_info.supplement_anchors
        label._last_supplement_anchors_report = label.debug_info.supplement_anchors_report
        return
    v_jd = np.asarray(v_jd, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(v_jd)
    k = min(int(top_k or label.SUPPLEMENT_TOP_K), 30)
    scores, labels = label.vocab_index.search(v_jd, k)
    added = []
    label.debug_info.supplement_anchors_report = []
    melt_threshold = float(label.ANCHOR_MELT_COV_J)
    v_ids_to_check = []
    candidates = []
    for score, tid in zip(scores[0], labels[0]):
        tid = int(tid)
        if tid <= 0:
            continue
        if str(tid) in anchor_skills:
            continue
        term, etype = label._vocab_meta.get(tid, ("", ""))
        etype = (etype or "").lower()
        if not term or len(term) < 2:
            continue
        if etype and etype not in label.SUPPLEMENT_ALLOW_ENTITY_TYPES:
            label.debug_info.supplement_anchors_report.append(
                {"tid": tid, "term": term, "etype": etype, "score": float(score), "reason": "etype_not_allowed"}
            )
            continue
        candidates.append((tid, term, etype, float(score)))
        v_ids_to_check.append(tid)

    if not v_ids_to_check:
        label.debug_info.supplement_anchors = []
        label.debug_info.supplement_anchors_report = []
        label._last_supplement_anchors = label.debug_info.supplement_anchors
        label._last_supplement_anchors_report = label.debug_info.supplement_anchors_report
        return

    global_count = {}
    stats_map = {}
    try:
        cypher = """
        UNWIND $v_ids AS vid
        MATCH (v:Vocabulary {id: vid})<-[:REQUIRE_SKILL]-(j:Job)
        RETURN vid, count(j) AS cnt
        """
        for r in label.graph.run(cypher, v_ids=v_ids_to_check):
            global_count[int(r["vid"])] = int(r.get("cnt") or 0)
    except Exception:
        global_count = {vid: 0 for vid in v_ids_to_check}

    try:
        ph = ",".join("?" * len(v_ids_to_check))
        rows = label.stats_conn.execute(
            f"SELECT voc_id, work_count, domain_dist FROM vocabulary_domain_stats WHERE voc_id IN ({ph})",
            v_ids_to_check,
        ).fetchall()
        for r in rows:
            stats_map[int(r[0])] = (int(r[1] or 0), r[2])
    except Exception:
        stats_map = {}

    active_domains = set(str(d) for d in (active_domain_ids or []))
    ranked = []
    for tid, term, etype, score in candidates:
        g = global_count.get(tid, 0)
        cov_j = (g / total_j) if total_j else 0.0
        reason = None
        if cov_j >= melt_threshold:
            reason = "cov_j_melt"
        row = stats_map.get(tid)
        degree_w = 0
        domain_ratio = 0.0
        if row:
            degree_w, dist_json = row
            try:
                dist = json.loads(dist_json) if isinstance(dist_json, str) else (dist_json or {})
            except (TypeError, ValueError):
                dist = {}
            expanded = label._expand_domain_dist(dist)
            degree_w_expanded = sum(expanded.values())
            if active_domains:
                target_degree_w = sum(expanded.get(str(d), 0) for d in active_domains)
            else:
                target_degree_w = degree_w_expanded
            domain_ratio = (target_degree_w / degree_w_expanded) if degree_w_expanded else 0.0
            if active_domains and domain_ratio < float(label.SUPPLEMENT_DOMAIN_RATIO_MIN):
                reason = "low_domain_ratio"
        else:
            reason = reason or "no_stats"

        report_row = {
            "tid": tid,
            "term": term,
            "etype": etype,
            "score": float(score),
            "cov_j": float(cov_j),
            "domain_ratio": float(domain_ratio),
            "reason": reason or "candidate",
        }
        label.debug_info.supplement_anchors_report.append(report_row)
        if reason is None:
            ranked.append(report_row)

    ranked.sort(key=lambda x: (x["domain_ratio"], x["score"]), reverse=True)
    for item in ranked[: label.SUPPLEMENT_MAX_ADD]:
        tid = item["tid"]
        term = item["term"]
        cov_j = item["cov_j"]
        # 向量补锚：先给可解释 proxy 分，最终由 refine 与主锚 mx 对齐天花板（不可默认压过 REQUIRE）
        sim_proxy = float(item["score"]) * (0.55 + 0.45 * float(item.get("domain_ratio") or 0.0))
        anchor_skills[str(tid)] = {
            "term": term,
            "anchor_source": "jd_vector_supplement",
            "anchor_source_weight": ANCHOR_SOURCE_WEIGHT_JD_SUPPLEMENT,
            "score_before_role_penalty": sim_proxy,
            "score_after_role_penalty": sim_proxy * 0.24,
            "final_anchor_score": sim_proxy * 0.24,
        }
        added.append((tid, term, round(item["score"], 4), round(cov_j, 4), round(item["domain_ratio"], 4)))

    label.debug_info.supplement_anchors = added
    label._last_supplement_anchors = label.debug_info.supplement_anchors
    label._last_supplement_anchors_report = label.debug_info.supplement_anchors_report
    label._cached_v_jd = v_jd

