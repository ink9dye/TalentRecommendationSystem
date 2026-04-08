import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# SQLite 变量占位上限保守值，避免 IN (...) 过长
_SQLITE_IN_CHUNK = 900

from src.core.recall.label_means import label_anchors
from src.core.recall.label_means.hierarchy_guard import parse_json_dist
from src.core.recall.label_means.label_debug import debug_print
from src.core.recall.label_means.label_anchors import _is_concrete_skill_phrase, _is_task_like, _load_abbr_keys
from src.utils.domain_utils import DomainProcessor
from src.utils.tools import extract_skills

# ---------------------------------------------------------------------------
# Stage1 早期粗分型（阶段 A）：在 extract_anchor_skills 之前产生，用于准入与分层。
# 粗分角色（非终态，后续经 refine_stage1_anchor_layers 修正）
# ---------------------------------------------------------------------------
JD_COARSE_CORE = "core_concept_candidate"
JD_COARSE_OBJECT_TASK = "object_or_task_candidate"
JD_COARSE_TOOL_FW = "tool_or_framework_candidate"
JD_COARSE_VENUE_PREF = "venue_or_preference_candidate"
JD_COARSE_GENERIC_RISKY = "generic_risky_candidate"
JD_COARSE_NOISE = "noise_fragment_candidate"

ANCHOR_ROLE_MAIN = "main_anchor_candidate"
ANCHOR_ROLE_AUX = "aux_anchor_candidate"
ANCHOR_ROLE_CONTEXT = "context_only_anchor"

# 向量补锚相对主锚峰值的天花板（结构比例，非词表）
SUPPLEMENT_TO_MAIN_SCORE_CAP_RATIO = 0.38
CONTEXT_ANCHOR_MAX_SCORE_RATIO = 0.06

# 偏好/资格/发表导向（结构：叙述性连接，非技术名词表）
_PREF_QUAL_PATTERN = re.compile(
    r"(发表|优先|顶会|期刊|会议|ccf|分区|任职|资格|学历|薪资|福利|五险一金|质量要求|招聘要求|任职要求)"
)
# 英文 hyphen 泛化后缀（learning-based 等）
_HYPHEN_GENERIC_EN = re.compile(r"-(based|centric|oriented|agnostic|driven|free)$", re.IGNORECASE)
# 单 token 英文技术形（库名/仿真器短名）
_EN_SINGLE_TECH = re.compile(r"^[a-z][a-z0-9+\-]{1,15}$", re.IGNORECASE)


def _has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def _is_subsumed_fragment(term: str, all_terms: Set[str]) -> bool:
    """同一 JD 清洗集合内，是否存在更长短语以本词为子串（结构冗余 → 偏泛化）。"""
    if not term or not all_terms:
        return False
    t = term.strip().lower()
    if not t:
        return False
    if _is_concrete_skill_phrase(t):
        return False
    for u in all_terms:
        if not u or u == t:
            continue
        ul = u.strip().lower()
        if len(ul) <= len(t):
            continue
        if t in ul:
            return True
    # 英文整词被更长 JD 短语包含（如 control ⊂ motion control）
    if " " not in t and re.match(r"^[a-z]+$", t):
        for u in all_terms:
            ul = u.strip().lower()
            if len(ul) <= len(t) + 1:
                continue
            if re.search(r"(?:^|\s)" + re.escape(t) + r"(?:\s|$)", ul):
                return True
    return False


def coarse_classify_jd_term(term: str, all_terms: Set[str], raw_text: str) -> str:
    """
    阶段 A：对单条 JD 技能短语做轻量粗分型（形式 + 与全集相对位置），不依赖大词表。
    """
    if not term or not str(term).strip():
        return JD_COARSE_NOISE
    t = str(term).strip().lower()
    raw = (raw_text or "").lower()

    if len(t) < 2:
        return JD_COARSE_NOISE

    # 偏好/资格/发表语境（整段技能字段里偶发混入）
    if _PREF_QUAL_PATTERN.search(t):
        return JD_COARSE_VENUE_PREF

    # 英文结构泛化
    if _HYPHEN_GENERIC_EN.search(t):
        return JD_COARSE_GENERIC_RISKY

    abbr_keys = _load_abbr_keys()
    ntok = len(t.split())

    # 完整、可独立作主线的技术短语：优先 object_task / core，避免后置 rerank 才「救」回来
    if _is_concrete_skill_phrase(t):
        return JD_COARSE_OBJECT_TASK if _is_task_like(t) else JD_COARSE_CORE

    # 极短 CJK（多为碎片或高歧义双字）
    if _has_cjk(t) and len(t) <= 2:
        return JD_COARSE_GENERIC_RISKY

    # 短片段被更长 JD 短语覆盖 → 泛化冗余（完整短语已在上方排除）
    if _is_subsumed_fragment(t, all_terms):
        return JD_COARSE_GENERIC_RISKY

    # 多词英文
    if ntok >= 2 and not _has_cjk(t):
        if _is_task_like(t):
            return JD_COARSE_OBJECT_TASK
        # 非任务向的多词 ASCII（常为平台/产品名）
        if all(_EN_SINGLE_TECH.match(tok) for tok in t.split()):
            return JD_COARSE_TOOL_FW
        return JD_COARSE_CORE

    # 单词英文
    if ntok == 1 and not _has_cjk(t):
        if t in abbr_keys and len(t) <= 5:
            return JD_COARSE_TOOL_FW
        if _EN_SINGLE_TECH.match(t) and not _is_task_like(t):
            return JD_COARSE_TOOL_FW
        if _is_task_like(t) and len(t) <= 6 and _is_subsumed_fragment(t, all_terms):
            return JD_COARSE_GENERIC_RISKY
        if _is_task_like(t):
            return JD_COARSE_OBJECT_TASK
        return JD_COARSE_CORE

    # 中文或其它
    if _has_cjk(t):
        if len(t) >= 4:
            return JD_COARSE_OBJECT_TASK if _is_task_like(t) else JD_COARSE_CORE
        return JD_COARSE_GENERIC_RISKY

    return JD_COARSE_CORE


def build_jd_coarse_roles_map(cleaned_terms: Optional[Set[str]], raw_text: str) -> Dict[str, str]:
    """对 JD 清洗短语全集粗分；key 为小写 term。"""
    out: Dict[str, str] = {}
    if not cleaned_terms:
        return out
    terms = {str(x).strip().lower() for x in cleaned_terms if x and str(x).strip()}
    for term in terms:
        out[term] = coarse_classify_jd_term(term, terms, raw_text or "")
    return out


def refine_stage1_anchor_layers(
    anchor_skills: Dict[str, Any],
    jd_coarse_roles: Dict[str, str],
    cleaned_terms: Optional[Set[str]],
    raw_text: str,
) -> None:
    """
    阶段 B：在 extract_anchor_skills + supplement 之后，结合来源与 REQUIRE 命中修正分型，
    写入 anchor_role / anchor_priority / is_primary_eligible / source_kind 等，并真正接管 final 分数与排序。

    REQUIRE 直出锚：final 以 extract 中 score_after_role_penalty 为基（已在 backbone 环节折损辅/泛）；
    向量补锚：对主锚峰值做天花板；context_only：禁止与主锚同量级竞争。
    """
    terms_set = {str(x).strip().lower() for x in (cleaned_terms or set()) if x}
    # 先写元数据
    for _vid, info in (anchor_skills or {}).items():
        term = (info.get("term") or "").strip()
        tl = term.lower()
        src = (info.get("anchor_source") or "skill_direct").strip()
        source_kind = "jd_vector_supplement" if src == "jd_vector_supplement" else "jd_require_skill_graph"

        coarse = jd_coarse_roles.get(tl) or coarse_classify_jd_term(term, terms_set, raw_text)

        # 图谱与 JD 粗分偶发标成 risky：完整技能短语拉回主池
        if coarse == JD_COARSE_GENERIC_RISKY and _is_concrete_skill_phrase(term):
            coarse = JD_COARSE_OBJECT_TASK if _is_task_like(term) else JD_COARSE_CORE

        if source_kind == "jd_vector_supplement":
            if coarse in (JD_COARSE_CORE, JD_COARSE_OBJECT_TASK):
                coarse = JD_COARSE_TOOL_FW
            elif coarse == JD_COARSE_TOOL_FW:
                pass
            else:
                coarse = JD_COARSE_GENERIC_RISKY

        primary_ok = (
            source_kind != "jd_vector_supplement"
            and coarse in (JD_COARSE_CORE, JD_COARSE_OBJECT_TASK)
            and coarse not in (JD_COARSE_VENUE_PREF, JD_COARSE_NOISE, JD_COARSE_GENERIC_RISKY)
        )
        if coarse == JD_COARSE_TOOL_FW:
            primary_ok = False
        if coarse in (JD_COARSE_VENUE_PREF, JD_COARSE_NOISE):
            primary_ok = False
        if coarse == JD_COARSE_GENERIC_RISKY:
            primary_ok = False

        if coarse in (JD_COARSE_VENUE_PREF, JD_COARSE_NOISE):
            role = ANCHOR_ROLE_CONTEXT
        elif coarse == JD_COARSE_GENERIC_RISKY or coarse == JD_COARSE_TOOL_FW or source_kind == "jd_vector_supplement":
            role = ANCHOR_ROLE_AUX
        elif primary_ok:
            role = ANCHOR_ROLE_MAIN
        else:
            role = ANCHOR_ROLE_AUX

        if role == ANCHOR_ROLE_CONTEXT:
            priority = 0.08
        elif role == ANCHOR_ROLE_AUX:
            priority = 0.42 if coarse == JD_COARSE_GENERIC_RISKY else 0.48
        else:
            priority = 1.0

        base_at = label_anchors.classify_anchor_type(term)
        at = _refine_anchor_type_for_layer(base_at, coarse, role, source_kind)

        info["coarse_jd_role"] = coarse
        info["anchor_role"] = role
        info["anchor_priority"] = priority
        info["source_kind"] = source_kind
        info["is_primary_eligible"] = role == ANCHOR_ROLE_MAIN
        info["is_expand_eligible"] = role != ANCHOR_ROLE_CONTEXT
        info["is_context_only"] = role == ANCHOR_ROLE_CONTEXT
        info["anchor_type"] = at

    def _ordered_base_score(info: Dict[str, Any]) -> float:
        """优先使用 extract 内 tier0 主线重排分，避免补锚天花板误用旧 s_post。"""
        return float(
            (info or {}).get("tier0_rerank_score")
            or (info or {}).get("final_anchor_score")
            or (info or {}).get("score_after_role_penalty")
            or 0.0
        )

    # 主锚峰值：用于补锚天花板与 context 上限（仅 REQUIRE 图、且主候选）
    mx_primary = 0.0
    for _vid, info in (anchor_skills or {}).items():
        if (info or {}).get("source_kind") != "jd_require_skill_graph":
            continue
        if not (info or {}).get("is_primary_eligible"):
            continue
        mx_primary = max(mx_primary, _ordered_base_score(info))
    if mx_primary <= 0.0:
        for _vid, info in (anchor_skills or {}).items():
            if (info or {}).get("source_kind") != "jd_require_skill_graph":
                continue
            mx_primary = max(mx_primary, _ordered_base_score(info))
    if mx_primary <= 0.0:
        mx_primary = 1.0

    # 第二遍：final 分数与解释字段（主线重排分已进入 tier0_rerank_score，此处只做补锚封顶 / context 帽）
    for _vid, info in (anchor_skills or {}).items():
        role = (info or {}).get("anchor_role")
        source_kind = (info or {}).get("source_kind")
        pre_mainline = float((info or {}).get("score_after_role_penalty") or 0.0)
        base_ordered = _ordered_base_score(info)
        raw_pre = (info or {}).get("score_before_role_penalty")
        pre = float(raw_pre) if raw_pre is not None else pre_mainline

        info["score_before_role_penalty"] = pre
        info["score_after_role_penalty"] = pre_mainline

        if source_kind == "jd_vector_supplement":
            cap = mx_primary * SUPPLEMENT_TO_MAIN_SCORE_CAP_RATIO
            final_v = min(base_ordered, cap)
            info["final_anchor_score"] = final_v
            info["supplement_score_cap"] = cap
        elif role == ANCHOR_ROLE_CONTEXT:
            info["final_anchor_score"] = min(base_ordered, mx_primary * CONTEXT_ANCHOR_MAX_SCORE_RATIO)
        else:
            info["final_anchor_score"] = base_ordered

        info["score_explanation"] = {
            "mx_primary_ref": mx_primary,
            "ordered_base": base_ordered,
            "penalty_chain": "tier0_rerank_then_refine_cap" if source_kind == "jd_vector_supplement" else "tier0_mainline_rerank",
        }

    # 主 / 辅 / 上下文 分桶排序：主锚优先，其次辅，上下文垫后
    if anchor_skills:
        def _bucket(info: Dict[str, Any]) -> int:
            if (info or {}).get("is_context_only"):
                return 2
            if (info or {}).get("is_primary_eligible"):
                return 0
            return 1

        sorted_items = sorted(
            anchor_skills.items(),
            key=lambda x: (_bucket(x[1]), -float((x[1] or {}).get("final_anchor_score") or 0.0)),
        )
        anchor_skills.clear()
        anchor_skills.update(sorted_items)


def _strip_context_only_from_anchor_skills(
    anchor_skills: Dict[str, Any],
    recall: Any,
) -> None:
    """
    context_only 不参与主锚池与后续 anchor_ctx / 画像；侧挂到 recall._stage1_context_only_anchors 便于审计。
    """
    if not anchor_skills:
        return
    ctx_only = {k: v for k, v in list(anchor_skills.items()) if (v or {}).get("is_context_only")}
    if not ctx_only:
        return
    for k in ctx_only:
        del anchor_skills[k]
    setattr(recall, "_stage1_context_only_anchors", ctx_only)


def _refine_anchor_type_for_layer(
    base_type: str,
    coarse: str,
    role: str,
    source_kind: str,
) -> str:
    """在 classify_anchor_type 之上叠一层流程型修正，使 Stage2 现有 anchor_type 语义仍可用。"""
    if role == ANCHOR_ROLE_CONTEXT or coarse in (JD_COARSE_VENUE_PREF, JD_COARSE_NOISE):
        return "generic_task_term"
    if coarse == JD_COARSE_GENERIC_RISKY:
        return "generic_task_term"
    if coarse == JD_COARSE_TOOL_FW or source_kind == "jd_vector_supplement":
        if base_type == "acronym":
            return base_type
        return "application_term"
    return base_type


@dataclass
class Stage1Result:
    """
    阶段 1 结构化结果壳，用于逐步解耦领域与锚点阶段的中间状态。
    含 jd_profile（四层领域画像）供 Stage2 层级守卫使用。
    """

    active_domains: Set[int]
    domain_regex: str
    anchor_skills: Dict[Any, Any]
    job_ids: List[int]
    job_previews: List[Dict[str, Any]]
    dominance: float
    anchor_debug: Dict[str, Any]
    jd_profile: Optional[Dict[str, Any]] = None  # domain/field/subfield/topic_weights, active_*, main_*


def split_jd_segments(query_text: str) -> List[str]:
    """按换行与句读切 JD 为短 segment，供 anchor-centric jd_snippet 使用。"""
    if not query_text:
        return []
    parts = re.split(r"[\n\r]+|[。；;！!？?●•|]+", query_text)
    return [p.strip() for p in parts if p and str(p).strip()]


def patch_stage1_anchor_ctx_extras(
    anchor_skills: Dict[str, Any],
    jd_terms_cleaned_list: List[str],
    raw_text: str,
) -> None:
    """
    用 extract_skills 已清洗的 JD 短语补 local 提示，并做极简 co-anchor（同 segment / 共现）。
    写入每项 _stage1_local_phrases / _stage1_co_anchor_terms，供 label_anchors 侧优先消费。
    """
    if not anchor_skills:
        return
    seen: Set[str] = set()
    ordered_phrases: List[str] = []
    for p in jd_terms_cleaned_list or []:
        pl = (p or "").strip().lower()
        if not pl or pl in seen:
            continue
        seen.add(pl)
        ordered_phrases.append((p or "").strip())

    text_l = (raw_text or "").lower()
    pool: List[Tuple[str, Dict[str, Any], str]] = []
    for vid, inf in anchor_skills.items():
        if (inf or {}).get("is_context_only"):
            continue
        t = (inf.get("term") or "").strip()
        if t:
            pool.append((str(vid), inf, t))

    for _vid, info, term in pool:
        al = term.lower()
        jd_snip = (info.get("jd_snippet") or "").strip()
        jd_snip_l = jd_snip.lower()
        local_cands: List[str] = []
        for phrase in ordered_phrases:
            pl = phrase.lower()
            if pl == al:
                continue
            if al in pl or pl in al:
                local_cands.append(phrase)
            elif jd_snip_l and pl in jd_snip_l:
                local_cands.append(phrase)
        local_cands.sort(key=lambda x: len(x), reverse=True)
        uniq: List[str] = []
        useen: Set[str] = set()
        for x in local_cands:
            xl = x.lower()
            if xl in useen:
                continue
            useen.add(xl)
            uniq.append(x)
            if len(uniq) >= 3:
                break
        info["_stage1_local_phrases"] = uniq

    for vid, info, term in pool:
        al = term.lower()
        seg_a = (info.get("jd_snippet") or "").strip()
        loc_a = (info.get("local_context") or "").lower()
        phr_a = (info.get("phrase_context") or "").lower()
        combo = (loc_a + " " + phr_a + " " + seg_a.lower()).strip()
        pos_self = text_l.find(al)
        scored: List[Tuple[int, int, float, str]] = []
        for vid2, info2, term2 in pool:
            if vid2 == vid:
                continue
            bl = term2.lower()
            if bl == al:
                continue
            seg_b = (info2.get("jd_snippet") or "").strip()
            same_seg = bool(seg_a and seg_b and seg_a == seg_b)
            co_txt = bool(bl and (bl in combo or (seg_a and bl in seg_a.lower())))
            if not (same_seg or co_txt):
                continue
            pos_o = text_l.find(bl)
            dist = 99999
            if pos_self >= 0 and pos_o >= 0:
                dist = abs(pos_o - pos_self)
            fin = float(info2.get("final_anchor_score") or 0.0)
            tier = 0 if same_seg else 1
            scored.append((tier, dist, -fin, term2))
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        info["_stage1_co_anchor_terms"] = [x[3] for x in scored[:3]]


def attach_anchor_contexts(
    anchor_skills: Dict[str, Any],
    query_text: str,
    window: int = 10,
) -> None:
    """
    为每个锚点增加 anchor-centric jd_snippet、local_context、phrase_context。
    按换行/句读切 segment，优先最短命中 segment；段内取锚点短窗为 local_context。
    原地修改 anchor_skills：jd_snippet / local_context / phrase_context。
    """
    if not query_text or not anchor_skills:
        return
    segments = split_jd_segments(query_text)
    if not segments:
        segments = [query_text.strip()] if query_text.strip() else []
    text_lower = query_text.lower()
    _max_snip = 200

    for _vid, info in anchor_skills.items():
        term = (info.get("term") or "").strip()
        if not term:
            info["local_context"] = ""
            info["phrase_context"] = ""
            info["jd_snippet"] = ""
            continue
        term_lower = term.lower()
        best_seg = ""
        indexed = [(i, s) for i, s in enumerate(segments) if term_lower in s.lower()]
        if indexed:
            best_seg = min(indexed, key=lambda x: (len(x[1].strip()), x[0]))[1].strip()
        else:
            pos = text_lower.find(term_lower)
            if pos >= 0:
                start = max(0, pos - 36)
                end = min(len(query_text), pos + len(term) + 100)
                best_seg = query_text[start:end].strip()
                if len(best_seg) > _max_snip:
                    cut = query_text[start : start + _max_snip]
                    sp = cut.rfind(" ")
                    best_seg = (cut[:sp] if sp > 20 else cut).strip()
            else:
                head = query_text[: min(120, len(query_text))].strip()
                best_seg = head

        if not best_seg:
            info["jd_snippet"] = ""
            info["local_context"] = ""
            info["phrase_context"] = ""
            continue

        info["jd_snippet"] = best_seg[:_max_snip].strip()
        seg_use = info["jd_snippet"]
        seg_l = seg_use.lower()
        pos_in_seg = seg_l.find(term_lower)
        if pos_in_seg >= 0:
            start = max(0, pos_in_seg - window)
            end = min(len(seg_use), pos_in_seg + len(term) + window)
            info["local_context"] = seg_use[start:end].strip()
            info["phrase_context"] = seg_use.strip()[: min(180, len(seg_use))]
        else:
            info["local_context"] = seg_use[: min(80, len(seg_use))].strip()
            info["phrase_context"] = seg_use.strip()[:120]


def _batch_load_vocabulary_stats_for_tids(
    recall,
    tids: Set[int],
) -> Tuple[Dict[int, Any], Dict[int, Any]]:
    """
    批量加载 vocabulary_topic_stats / vocabulary_domain_stats，与逐条 WHERE voc_id=? 结果一致。
    返回 (topic_row_by_voc_id, domain_dist_by_voc_id)；topic 行为整行 tuple，与 fetchone 相同列序。
    """
    topic_map: Dict[int, Any] = {}
    domain_dist_map: Dict[int, Any] = {}
    if not tids or not getattr(recall, "stats_conn", None):
        return topic_map, domain_dist_map
    conn = recall.stats_conn
    tid_list = sorted(int(t) for t in tids)
    for i in range(0, len(tid_list), _SQLITE_IN_CHUNK):
        chunk = tid_list[i : i + _SQLITE_IN_CHUNK]
        if not chunk:
            continue
        ph = ",".join("?" * len(chunk))
        try:
            rows = conn.execute(
                f"SELECT voc_id, field_id, subfield_id, topic_id, field_dist, subfield_dist, topic_dist, source "
                f"FROM vocabulary_topic_stats WHERE voc_id IN ({ph})",
                chunk,
            ).fetchall()
            for row in rows:
                topic_map[int(row[0])] = row
        except Exception:
            pass
        try:
            rows = conn.execute(
                f"SELECT voc_id, domain_dist FROM vocabulary_domain_stats WHERE voc_id IN ({ph})",
                chunk,
            ).fetchall()
            for row in rows:
                domain_dist_map[int(row[0])] = row[1]
        except Exception:
            pass
    return topic_map, domain_dist_map


def build_jd_hierarchy_profile(
    anchor_skills: Dict[str, Any],
    query_text: str,
    recall,
) -> Dict[str, Any]:
    """
    利用锚点对应的学术落点候选（SIMILAR_TO）与 vocabulary_topic_stats / vocabulary_domain_stats
    聚合成 JD 的四层领域画像，供 Stage2 层级守卫使用。
    返回 jd_profile: domain_weights, field_weights, subfield_weights, topic_weights,
    active_domains, active_fields, active_subfields, active_topics, main_*_id。
    """
    jd_profile: Dict[str, Any] = {
        "domain_weights": {},
        "field_weights": {},
        "subfield_weights": {},
        "topic_weights": {},
        "active_domains": [],
        "active_fields": [],
        "active_subfields": [],
        "active_topics": [],
        "main_domain_id": None,
        "main_field_id": None,
        "main_subfield_id": None,
        "main_topic_id": None,
    }
    if not getattr(recall, "graph", None) or not getattr(recall, "stats_conn", None):
        return jd_profile
    # 每个锚点 vid 查 SIMILAR_TO 取 top 学术词，用其 topic/domain 分布加权聚合
    SIMILAR_TOP_K = 5
    all_domain: Dict[str, float] = {}
    all_field: Dict[str, float] = {}
    all_subfield: Dict[str, float] = {}
    all_topic: Dict[str, float] = {}
    total_weight = 0.0
    # (tid, weight) 保留锚点×相似词的多重贡献
    tid_weight_pairs: List[Tuple[int, float]] = []

    anchor_vids: List[int] = []
    for vid_str, ainfo in (anchor_skills or {}).items():
        if (ainfo or {}).get("is_context_only"):
            continue
        try:
            anchor_vids.append(int(vid_str))
        except (TypeError, ValueError):
            continue
    # -------------------------------------------------------------------------
    # Neo4j：由「每锚点 1 次 MATCH + LIMIT k」改为「全锚点 1 次 IN + Python 侧每 src 取 top-k」
    # 等价性（指标/语义/数据）：
    #   - 过滤：与原 Cypher 相同 — v.id 锚定、r.score>=0.5、v2.type in concept/keyword。
    #   - 每锚点的 top-k：按 r.score DESC 取前 k 条；与 per-anchor ORDER BY score DESC LIMIT k
    #     一致，除非多条边 score  bitwise 完全相等（Neo4j 未保证并列次序），对加权聚合影响可忽略。
    #   - 返回的 (tid, w) 多重集合与原 N 次查询的并集一致（含同一 tid 被多锚点命中）。
    # -------------------------------------------------------------------------
    if anchor_vids:
        try:
            from itertools import groupby

            sim_rows = recall.graph.run(
                """
                MATCH (v:Vocabulary)-[r:SIMILAR_TO]->(v2:Vocabulary)
                WHERE v.id IN $vids AND r.score >= 0.5
                  AND coalesce(v2.type, 'concept') IN ['concept', 'keyword']
                RETURN v.id AS src_vid, v2.id AS tid, r.score AS s
                ORDER BY src_vid, s DESC
                """,
                vids=anchor_vids,
            ).data()
            for _src_key, group in groupby(
                sim_rows or [],
                key=lambda x: x.get("src_vid"),
            ):
                for i, r in enumerate(group):
                    if i >= SIMILAR_TOP_K:
                        break
                    tid = r.get("tid")
                    if tid is None:
                        continue
                    try:
                        tid_i = int(tid)
                    except (TypeError, ValueError):
                        continue
                    w = float(r.get("s") or 0.5)
                    src_vid = r.get("src_vid")
                    ap = 1.0
                    if src_vid is not None and anchor_skills:
                        ap = float((anchor_skills.get(str(int(src_vid))) or {}).get("anchor_priority") or 1.0)
                    tid_weight_pairs.append((tid_i, w * ap))
        except Exception:
            tid_weight_pairs = []

    unique_tids = {t for t, _ in tid_weight_pairs}
    topic_map, domain_dist_map = _batch_load_vocabulary_stats_for_tids(recall, unique_tids)

    for tid, w in tid_weight_pairs:
        row_t = topic_map.get(tid)
        dist_json = domain_dist_map.get(tid)
        row_d = (dist_json,) if dist_json is not None else None
        if row_t:
            # 批注：SELECT 列为 voc_id, field_id, subfield_id, topic_id, field_dist, subfield_dist, topic_dist, source
            # 必须用 [4][5][6] 取三层 JSON；误用 [3] 会把 topic_id 当成 field_dist，导致 JD 画像与词侧 key 全对不上、Stage4 overlap 恒为 0。
            fd = parse_json_dist(row_t[4])
            sd = parse_json_dist(row_t[5])
            td = parse_json_dist(row_t[6])
            for k, v in fd.items():
                all_field[k] = all_field.get(k, 0.0) + v * w
            for k, v in sd.items():
                all_subfield[k] = all_subfield.get(k, 0.0) + v * w
            for k, v in td.items():
                all_topic[k] = all_topic.get(k, 0.0) + v * w
        if row_d and row_d[0]:
            dd = parse_json_dist(row_d[0])
            for k, v in dd.items():
                all_domain[k] = all_domain.get(k, 0.0) + v * w
        total_weight += w
    if total_weight <= 0:
        return jd_profile
    def _norm(d: Dict[str, float]) -> Dict[str, float]:
        s = sum(d.values())
        return {k: v / s for k, v in d.items()} if s else d
    jd_profile["domain_weights"] = _norm(all_domain)
    jd_profile["field_weights"] = _norm(all_field)
    jd_profile["subfield_weights"] = _norm(all_subfield)
    jd_profile["topic_weights"] = _norm(all_topic)
    jd_profile["active_domains"] = list(jd_profile.get("active_domains") or []) or list(all_domain.keys())[:10]
    jd_profile["active_fields"] = list(all_field.keys())[:15]
    jd_profile["active_subfields"] = list(all_subfield.keys())[:20]
    jd_profile["active_topics"] = list(all_topic.keys())[:25]
    if all_domain:
        jd_profile["main_domain_id"] = max(all_domain, key=all_domain.get)
    if all_field:
        jd_profile["main_field_id"] = max(all_field, key=all_field.get)
    if all_subfield:
        jd_profile["main_subfield_id"] = max(all_subfield, key=all_subfield.get)
    if all_topic:
        jd_profile["main_topic_id"] = max(all_topic, key=all_topic.get)
    return jd_profile


def run_stage1(
    recall,
    query_vector,
    query_text: str | None = None,
    semantic_query_text: str | None = None,
    domain_id: str | None = None,
) -> Tuple[Set[int], str, Dict[str, Any], Dict[str, Any]]:
    """
    阶段 1：领域与锚点。

    逻辑与原 LabelRecallPath._stage1_domain_and_anchors 等价：
      1) 用 DomainDetector 或回退逻辑做岗位 / 领域探测；
      2) 用岗位 skills 抽取锚点（+ JD 语义补充）；
      3) 决定 active_domain_set 与 regex_str；
      4) 写入 recall._last_stage1_result 供后续调试；
      5) 返回 (active_domain_set, regex_str, anchor_skills, debug_1)。
    anchor_skills 中每项含 anchor_type / coarse_jd_role / anchor_role / is_primary_eligible /
    source_kind 等，由早期粗分型与生成后修正分型写入，供 Stage2/3 分策略（不切换候选源）。
    """
    job_ids: list[int] = []
    inferred_domains: list[int] = []
    dominance: float = 0.0
    job_previews: list[dict[str, Any]] = []
    anchor_debug: Dict[str, Any] = {}
    stage1_sub_ms: Dict[str, float] = {}
    _t_stage1 = time.perf_counter()

    # 0) 预清洗当前 JD 的技能短语，供 Step2 稀疏锚点保活使用
    # 使用 extract_skills 保持与 JD 技能抽取链路一致，只在本次查询作用域内生效。
    text_for_skills = semantic_query_text or query_text
    jd_terms_cleaned = None
    jd_terms_cleaned_list: List[str] = []
    if text_for_skills:
        try:
            jd_terms_cleaned_list = [s.strip() for s in extract_skills(text_for_skills) if s and str(s).strip()]
            jd_terms_cleaned = {s.lower() for s in jd_terms_cleaned_list}
        except Exception:
            jd_terms_cleaned = None
            jd_terms_cleaned_list = []
    # 每次调用都覆盖上一轮缓存，避免跨查询串扰
    setattr(recall, "_jd_cleaned_terms", jd_terms_cleaned)
    setattr(recall, "_jd_raw_text", text_for_skills or "")
    # 阶段 A：JD 短语早期粗分型 → recall._jd_coarse_roles，供 extract_anchor_skills 准入与后续分层
    jd_coarse_roles: Dict[str, str] = {}
    if jd_terms_cleaned:
        jd_coarse_roles = build_jd_coarse_roles_map(jd_terms_cleaned, text_for_skills or "")
    setattr(recall, "_jd_coarse_roles", jd_coarse_roles)
    _t0 = time.perf_counter()
    stage1_sub_ms["prep"] = (_t0 - _t_stage1) * 1000.0

    # JD 与 anchor_ctx / supplement 共用同一段文本切片 + 单次编码（原文键写入 jd_encode_cache）
    query_for_ctx = (semantic_query_text or query_text) or ""
    jd_canonical = label_anchors.canonical_jd_text_for_encode(query_for_ctx)
    encoder = getattr(recall, "_query_encoder", None)
    jd_encode_cache: Dict[str, np.ndarray] = {}
    setattr(recall, "_jd_query_vec_1d", None)
    if encoder and jd_canonical:
        try:
            encoder.lookup_or_encode(jd_canonical, jd_encode_cache)
            row = jd_encode_cache.get(jd_canonical)
            if row is not None:
                recall._jd_query_vec_1d = np.asarray(row, dtype=np.float32).flatten().copy()
        except Exception:
            pass

    # 1) 领域与岗位：优先使用 DomainDetector，缺失时回退到旧实现
    if getattr(recall, "domain_detector", None) is not None:
        active_set, _, debug = recall.domain_detector.detect(
            query_vector,
            query_text=query_text,
            user_domain=None,
        )
        sd = debug.get("stage1_debug", {}) if isinstance(debug, dict) else {}
        job_ids = sd.get("job_ids", []) or []
        inferred_domains = sd.get("candidate_domains", []) or list(active_set or [])
        dominance = sd.get("dominance", 0.0) or 0.0
        job_previews = sd.get("job_previews", []) or []
        anchor_debug = sd.get("anchor_debug", {}) or {}
    else:
        job_ids, inferred_domains, dominance = recall._detect_domain_context(query_vector)
        job_previews = recall._get_job_previews(job_ids)
        anchor_debug = recall._get_anchor_debug_stats(job_ids[:20], recall.total_job_count) if job_ids else {}
    _t1 = time.perf_counter()
    stage1_sub_ms["domain_detect"] = (_t1 - _t0) * 1000.0

    # 2) 工业锚点：岗位技能提取 + JD 语义补充
    anchor_skills = label_anchors.extract_anchor_skills(
        recall,
        job_ids,
        query_vector=query_vector,
        total_j=recall.total_job_count,
    )
    if (semantic_query_text or query_text) and anchor_skills is not None:
        label_anchors.supplement_anchors_from_jd_vector(
            recall,
            semantic_query_text or query_text,
            anchor_skills,
            total_j=recall.total_job_count,
            top_k=recall.JD_VOCAB_TOP_K,
            jd_encode_cache=jd_encode_cache,
        )
    _t2 = time.perf_counter()
    stage1_sub_ms["anchors"] = (_t2 - _t1) * 1000.0
    # 阶段 B：锚点生成后修正分型（来源 / 粗分角色 / 主辅上下文），并写回 anchor_type 供 Stage2 沿用
    if anchor_skills:
        refine_stage1_anchor_layers(
            anchor_skills,
            jd_coarse_roles,
            jd_terms_cleaned,
            query_for_ctx,
        )
        _strip_context_only_from_anchor_skills(anchor_skills, recall)
    if not anchor_skills:
        # 无锚点时，后续阶段直接短路
        stage1_sub_ms["total"] = (_t2 - _t_stage1) * 1000.0
        return set(), "", {}, {
            "job_ids": job_ids,
            "job_previews": job_previews,
            "anchor_debug": anchor_debug,
            "dominance": dominance,
            "stage1_sub_ms": stage1_sub_ms,
        }

    # anchor_type / anchor_role / is_primary_eligible 等已在 refine_stage1_anchor_layers 中写入

    # 调试：打印锚点及类型，便于排查
    if getattr(recall, "verbose", False) and not getattr(recall, "silent", False):
        print("\n【Stage1 锚点】tid | term | coarse | role | primary? | s_pre | s_post | final | src | type")
        for tid, info in anchor_skills.items():
            term = (info.get("term") or "")[:32]
            atype = info.get("anchor_type", "")
            cr = info.get("coarse_jd_role", "")
            ar = info.get("anchor_role", "")
            pe = info.get("is_primary_eligible", "")
            spre = info.get("score_before_role_penalty", "")
            spost = info.get("score_after_role_penalty", "")
            fn = info.get("final_anchor_score", "")
            sk = info.get("source_kind", "")
            print(f"  {tid} | {term} | {cr} | {ar} | {pe} | {spre} | {spost} | {fn} | {sk} | {atype}")

    industrial_kws = [v["term"] for v in anchor_skills.values()]

    # 3) 领域集合：如果用户指定 domain_id，则优先；否则使用推断领域 + 向量语义排序
    if domain_id and str(domain_id) != "0":
        active_domain_set: Set[int] = DomainProcessor.to_set(domain_id)
        if len(active_domain_set) > recall.ACTIVE_DOMAINS_TOP_K:
            active_domain_set = set(sorted(active_domain_set)[: recall.ACTIVE_DOMAINS_TOP_K])
    else:
        candidate_5 = inferred_domains
        if recall.domain_vectors and len(candidate_5) > recall.ACTIVE_DOMAINS_TOP_K:
            q = np.asarray(query_vector, dtype=np.float32).flatten()
            if q.size > 0:
                scores: list[tuple[int, float]] = []
                for d in candidate_5:
                    dv = recall.domain_vectors.get(str(d))
                    if dv is not None and dv.size == q.size:
                        sc = float(np.dot(q, dv))
                        scores.append((d, sc))
                scores.sort(key=lambda x: x[1], reverse=True)
                active_domain_set = set(d for d, _ in scores[: recall.ACTIVE_DOMAINS_TOP_K])
            else:
                active_domain_set = set(sorted(candidate_5)[: recall.ACTIVE_DOMAINS_TOP_K])
        else:
            active_domain_set = set(sorted(candidate_5)[: recall.ACTIVE_DOMAINS_TOP_K])

    regex_str = DomainProcessor.build_neo4j_regex(active_domain_set)
    _t3 = time.perf_counter()
    stage1_sub_ms["domain_regex"] = (_t3 - _t2) * 1000.0

    # 4) 锚点上下文化 + JD 四层领域画像（供 Stage2 层级守卫）
    attach_anchor_contexts(anchor_skills, query_for_ctx, window=12)
    patch_stage1_anchor_ctx_extras(anchor_skills, jd_terms_cleaned_list, query_for_ctx)
    # 条件化锚点表示：泛锚点带 JD 上下文，供 Stage2A 用 conditioned_vec 做落点打分
    if encoder and query_for_ctx and anchor_skills:
        # 复用 supplement 前写入的 JD 共振缓存，prefill 内跳过 JD 的 encode_batch 重复项
        encode_cache = jd_encode_cache
        label_anchors.prefill_encode_cache_for_anchor_ctx(
            encoder, anchor_skills, query_for_ctx, encode_cache
        )
        jd_vec_once = None
        try:
            jd_arr = label_anchors.encode_text_with_optional_cache(
                encoder, jd_canonical, encode_cache
            )
            if jd_arr is not None:
                jd_vec_once = np.asarray(jd_arr, dtype=np.float32).flatten().copy()
        except Exception:
            jd_vec_once = None
        anchor_ctx_count = 0
        for _vid, info in anchor_skills.items():
            term = (info.get("term") or "").strip()
            if not term:
                continue
            try:
                ctx = label_anchors.build_conditioned_anchor_representation(
                    term,
                    info,
                    anchor_skills,
                    query_for_ctx,
                    encoder,
                    jd_vec_precomputed=jd_vec_once,
                    encode_cache=encode_cache,
                )
                if ctx.get("conditioned_vec") is not None:
                    info["conditioned_vec"] = ctx["conditioned_vec"]
                    info["anchor_vec"] = ctx.get("anchor_vec")
                    info["local_phrase_vec"] = ctx.get("local_phrase_vec")
                    info["co_anchor_vec"] = ctx.get("co_anchor_vec")
                    info["jd_vec"] = ctx.get("jd_vec")
                    info["_anchor_ctx"] = ctx
                if anchor_ctx_count < 10:
                    _js = ctx.get("jd_snippet") or info.get("jd_snippet") or ""
                    _lc = ctx.get("local_context") or info.get("local_context") or ""
                    _pc = ctx.get("phrase_context") or info.get("phrase_context") or ""
                    debug_print(2, f"[Anchor Context] anchor={term!r}", recall)
                    debug_print(2, f"  jd_snippet={_js[:120]!r}", recall)
                    debug_print(2, f"  local_context={_lc[:120]!r}", recall)
                    debug_print(2, f"  phrase_context={_pc[:120]!r}", recall)
                    debug_print(2, f"  local_phrases={ctx.get('local_phrases', [])[:8]}", recall)
                    debug_print(2, f"  co_anchor_terms={ctx.get('co_anchor_terms', [])[:8]}", recall)
                    debug_print(2, (
                        "  weights="
                        f"{{anchor:{ctx.get('w_anchor', 0):.2f}, local:{ctx.get('w_local', 0):.2f}, "
                        f"co:{ctx.get('w_co', 0):.2f}, jd:{ctx.get('w_jd', 0):.2f}}}"
                    ), recall)
                    anchor_ctx_count += 1
            except Exception:
                pass
        stage1_ctx_summary: Dict[str, int] = {
            "anchors_total": len(anchor_skills),
            "anchors_with_jd_snippet": 0,
            "anchors_with_local_phrases": 0,
            "anchors_with_co_anchor_terms": 0,
            "anchors_with_nonzero_w_local": 0,
            "anchors_with_nonzero_w_co": 0,
            "anchors_with_nonzero_w_jd": 0,
        }
        for _v, inf in anchor_skills.items():
            ctxx = (inf or {}).get("_anchor_ctx") or {}
            lp = ctxx.get("local_phrases") or (inf or {}).get("_stage1_local_phrases") or []
            co = ctxx.get("co_anchor_terms") or (inf or {}).get("_stage1_co_anchor_terms") or []
            if str((inf or {}).get("jd_snippet") or "").strip():
                stage1_ctx_summary["anchors_with_jd_snippet"] += 1
            if lp:
                stage1_ctx_summary["anchors_with_local_phrases"] += 1
            if co:
                stage1_ctx_summary["anchors_with_co_anchor_terms"] += 1
            if float(ctxx.get("w_local") or 0) > 0:
                stage1_ctx_summary["anchors_with_nonzero_w_local"] += 1
            if float(ctxx.get("w_co") or 0) > 0:
                stage1_ctx_summary["anchors_with_nonzero_w_co"] += 1
            if float(ctxx.get("w_jd") or 0) > 0:
                stage1_ctx_summary["anchors_with_nonzero_w_jd"] += 1
        if getattr(recall, "verbose", False) and not getattr(recall, "silent", False):
            debug_print(2, f"[Stage1 anchor_ctx 汇总] {stage1_ctx_summary}", recall)
    else:
        stage1_ctx_summary = {
            "anchors_total": len(anchor_skills) if anchor_skills else 0,
            "anchors_with_jd_snippet": 0,
            "anchors_with_local_phrases": 0,
            "anchors_with_co_anchor_terms": 0,
            "anchors_with_nonzero_w_local": 0,
            "anchors_with_nonzero_w_co": 0,
            "anchors_with_nonzero_w_jd": 0,
        }
        if anchor_skills:
            for _v, inf in anchor_skills.items():
                if str((inf or {}).get("jd_snippet") or "").strip():
                    stage1_ctx_summary["anchors_with_jd_snippet"] += 1
                lp = (inf or {}).get("_stage1_local_phrases") or []
                co = (inf or {}).get("_stage1_co_anchor_terms") or []
                if lp:
                    stage1_ctx_summary["anchors_with_local_phrases"] += 1
                if co:
                    stage1_ctx_summary["anchors_with_co_anchor_terms"] += 1
            if getattr(recall, "verbose", False) and not getattr(recall, "silent", False):
                debug_print(2, f"[Stage1 anchor_ctx 汇总] {stage1_ctx_summary}", recall)
    _t4 = time.perf_counter()
    stage1_sub_ms["anchor_ctx"] = (_t4 - _t3) * 1000.0
    jd_profile = build_jd_hierarchy_profile(anchor_skills, query_for_ctx, recall)
    _t5 = time.perf_counter()
    stage1_sub_ms["jd_profile"] = (_t5 - _t4) * 1000.0
    jd_profile["active_domains"] = list(active_domain_set)

    debug_1: Dict[str, Any] = {
        "job_ids": job_ids,
        "job_previews": job_previews,
        "anchor_debug": anchor_debug,
        "dominance": dominance,
        "industrial_kws": industrial_kws,
        "anchor_skills": anchor_skills,
        "jd_profile": jd_profile,
        "stage1_sub_ms": stage1_sub_ms,
        "stage1_ctx_summary": stage1_ctx_summary,
    }
    stage1_sub_ms["finalize"] = (time.perf_counter() - _t5) * 1000.0
    stage1_sub_ms["total"] = (time.perf_counter() - _t_stage1) * 1000.0

    # 5) 回填 Stage1Result（含 jd_profile 供 Stage2 使用）
    recall._last_stage1_result = Stage1Result(
        active_domains=set(active_domain_set),
        domain_regex=regex_str,
        anchor_skills=dict(anchor_skills or {}),
        job_ids=list(job_ids),
        job_previews=list(job_previews),
        dominance=float(dominance),
        anchor_debug=dict(anchor_debug or {}),
        jd_profile=jd_profile,
    )

    return active_domain_set, regex_str, anchor_skills, debug_1

