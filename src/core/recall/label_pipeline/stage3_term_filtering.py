from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from src.core.recall.label_means import term_scoring
from src.core.recall.label_means.label_debug import debug_print
from src.core.recall.label_means.hierarchy_guard import (
    build_family_key,
    get_retrieval_role_from_term_role,
    should_drop_term,
    score_term_record,
)

# 软排序 + 轻过滤：至少保留 top_k，不再用阈值淘汰为 0
STAGE3_TOP_K = 20
STAGE3_DETAIL_DEBUG = True  # 打印每个词的 base_score / hierarchy_score / penalties / final_score / reject_reason
# Stage3 role 因子：mainline primary 稳一点，dense/conditioned_vec/side 略降，避免 Motion controller > motion control
STAGE3_MAINLINE_BOOST = 1.06
STAGE3_DENSE_PENALTY = 0.90
STAGE3_CONDITIONED_PENALTY = 0.94
STAGE3_SIDE_PENALTY = 0.95
# 标签路追踪：是否打印 source_type / anchor / similar_to 原始候选 / 被过滤原因 / final primary 胜出原因
LABEL_PATH_TRACE = True
# 单锚 + mainline_hits=1 + can_expand 的旁枝：从 support 降级 risky（结构信号，非词表）
STAGE3_SUPPORT_DEMOTE_FC_MAX = 0.38
STAGE3_SUPPORT_DEMOTE_PTC_MAX = 0.52
STAGE3_SUPPORT_DEMOTE_OBJ_MIN = 0.38
STAGE3_SUPPORT_DEMOTE_DRIFT_MIN = 0.62
STAGE3_SUPPORT_DEMOTE_DRIFT_PTC_MAX = 0.48
STAGE3_SUPPORT_DEMOTE_GENERIC_MIN = 0.52
STAGE3_SUPPORT_DEMOTE_POLY_MIN = 0.58
STAGE3_SUPPORT_DEMOTE_POLY_PTC_MAX = 0.50
# family 保送式 paper 选词：每 family 最多 1 primary + 1 support，cluster 默认不进 paper recall
PAPER_RECALL_MAX_TERMS = 12
# 与 select_terms_for_paper_recall 对齐的入稿闸门说明（逐项 term）
STAGE3_PAPER_GATE_DEBUG = True
# Stage3 窄表审计：final adjust / cross-anchor / support·risky / paper bridge（与 bucket reason 互补）
STAGE3_AUDIT_DEBUG = True
# Stage3 score_mult 单词明细 / 重点词 watch / rerank TopN（仅 stage3_build_score_map）
DEBUG_LABEL_PATH = True
# 非空时：下列审计只打印 term 字面完全匹配（strip 后）的行，避免刷屏；置空 set() 则按 top_k 全表
STAGE3_DEBUG_FOCUS_TERMS: Set[str] = {
    "Motion control",
    "motion control",
    "robot control",
    "Robot control",
    "digital control",
    "Digital control",
    "Robot manipulator",
    "Educational robotics",
    "route planning",
    "Route planning",
}

# 全局共识因子默认值（无硬编码）；Stage3 已彻底移除 cluster 依赖
STAGE3_CROSS_ANCHOR_DEFAULT = 1.0


def _stage3_normalized_source_types(rec: Dict[str, Any]) -> Set[str]:
    """合并 source_types 与 source_type / source 单值，供 conditioned_only 与 paper 门统一使用。"""
    source_types = rec.get("source_types") or set()
    if isinstance(source_types, (list, tuple)):
        source_types = set(source_types)
    else:
        source_types = set(source_types)
    st = (rec.get("source_type") or rec.get("source") or rec.get("origin") or "").strip().lower()
    if st:
        source_types.add(st)
    return source_types


def _stage3_is_conditioned_only(rec: Dict[str, Any]) -> bool:
    """
    conditioned_only（严格）：仅有 conditioned_vec，且无 similar_to、无 family_landing。
    与分桶 / bucket_reason_flags / paper 硬门同源。
    """
    st = _stage3_normalized_source_types(rec)
    has_similar = "similar_to" in st
    has_family = "family_landing" in st
    has_conditioned = "conditioned_vec" in st
    return bool(has_conditioned and (not has_similar) and (not has_family))


def _merge_stage3_duplicates(raw_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按 tid 聚合同一词的多个候选，保留多来源结构信息，供 Stage3 分层与准入使用。
    合并后保留：anchor_count, evidence_count, family_keys, source_types, parent_anchors, parent_primaries,
    mainline_hits, side_hits, best_stage2_score, best_anchor_identity, best_jd_align, best_context_continuity。
    最佳 seed 条目的整表字段会随 `obj[k]=v` 写入（含 Stage2A 的 primary_bucket、primary_reason、
    fallback_primary、mainline_candidate、can_expand_local 等），供 `_assign_stage3_bucket` / paper 门使用。
    """
    bucket: Dict[int, Dict[str, Any]] = {}
    for rec in raw_candidates:
        tid = rec.get("tid") or rec.get("vid")
        if tid is None:
            continue
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            continue
        obj = bucket.setdefault(tid, {
            "tid": tid,
            "term": rec.get("term") or "",
            "records": [],
            "best_identity_score": 0.0,
            "best_seed_score": 0.0,
            "best_anchor_identity": 0.0,
            "best_jd_align": 0.0,
            "best_context_continuity": 0.0,
            "parent_anchors": set(),
            "parent_primaries": set(),
            "source_types": set(),
            "term_roles": set(),
            "family_keys": set(),
            "anchor_count": 0,
            "evidence_count": 0,
            "mainline_hits": 0,
            "side_hits": 0,
            "has_primary_role": False,
            "has_support_role": False,
            "role_in_anchor": "",
            "can_expand": False,
            "retain_mode": "normal",
            "polysemy_risk": 0.0,
            "object_like_risk": 0.0,
            "generic_risk": 0.0,
        })
        obj["records"].append(rec)
        ident = float(rec.get("identity_score") or rec.get("sim_score") or 0.0)
        obj["best_identity_score"] = max(obj["best_identity_score"], ident)
        obj["best_anchor_identity"] = max(obj["best_anchor_identity"], ident)
        seed_sc = float(rec.get("score") or 0.0)
        obj["best_seed_score"] = max(obj["best_seed_score"], seed_sc)
        obj["best_jd_align"] = max(
            obj["best_jd_align"],
            float(rec.get("jd_candidate_alignment") or rec.get("jd_align") or 0.0),
        )
        obj["best_context_continuity"] = max(
            obj["best_context_continuity"],
            float(rec.get("context_continuity") or 0.0),
        )
        obj["polysemy_risk"] = max(obj.get("polysemy_risk", 0), float(rec.get("polysemy_risk") or 0))
        obj["object_like_risk"] = max(obj.get("object_like_risk", 0), float(rec.get("object_like_risk") or 0))
        obj["generic_risk"] = max(obj.get("generic_risk", 0), float(rec.get("generic_risk") or 0))
        pa = (rec.get("parent_anchor") or "").strip()
        pp = (rec.get("parent_primary") or "").strip()
        st = (rec.get("source_type") or rec.get("source") or rec.get("origin") or "").strip()
        tr = (rec.get("term_role") or "").strip()
        role_anchor = (rec.get("role_in_anchor") or "").strip().lower()
        if pa:
            obj["parent_anchors"].add(pa)
        if pp:
            obj["parent_primaries"].add(pp)
        if st:
            obj["source_types"].add(st)
        if tr:
            obj["term_roles"].add(tr)
        if role_anchor == "mainline":
            obj["mainline_hits"] = obj.get("mainline_hits", 0) + 1
        elif role_anchor:
            obj["side_hits"] = obj.get("side_hits", 0) + 1
        if tr == "primary" or st == "similar_to":
            obj["has_primary_role"] = True
        else:
            obj["has_support_role"] = True
        if rec.get("can_expand"):
            obj["can_expand"] = True
        rm = (rec.get("retain_mode") or "normal").strip().lower()
        if rm == "normal":
            obj["retain_mode"] = "normal"
        if role_anchor == "mainline":
            obj["role_in_anchor"] = "mainline"
        elif not obj.get("role_in_anchor"):
            obj["role_in_anchor"] = role_anchor or "side"
        if seed_sc >= obj.get("best_seed_score", -1.0):
            for k, v in rec.items():
                if k not in ("records", "parent_anchors", "parent_primaries", "source_types", "term_roles", "family_keys"):
                    obj[k] = v
    merged = []
    for tid, obj in bucket.items():
        obj["parent_anchors"] = sorted(obj["parent_anchors"])
        obj["parent_primaries"] = sorted(obj["parent_primaries"])
        obj["source_types"] = sorted(obj["source_types"])
        obj["term_roles"] = sorted(obj["term_roles"])
        obj["family_keys"] = sorted(obj.get("family_keys") or [])
        obj["anchor_count"] = len(obj["parent_anchors"])
        obj["evidence_count"] = len(obj["records"])
        obj["best_stage2_score"] = obj.get("best_stage2_score") or obj.get("best_seed_score") or 0.0
        if not obj.get("role_in_anchor") and obj.get("mainline_hits", 0) > 0:
            obj["role_in_anchor"] = "mainline"
        if not obj.get("role_in_anchor"):
            obj["role_in_anchor"] = "side" if obj.get("side_hits", 0) > 0 else ""
        merged.append(obj)
    merged.sort(key=lambda x: float(x.get("best_seed_score") or 0.0), reverse=True)
    return merged


def _classify_stage3_entry_groups(terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    候选分层：trusted_primary / secondary_primary / support_expansion。
    trusted_primary 不硬杀，只打 risk_flag；support_expansion 才重点审查。
    """
    for rec in terms:
        term_role = (rec.get("term_role") or "").strip().lower()
        if isinstance(rec.get("term_roles"), (list, set)):
            term_roles_set = set(rec["term_roles"]) if rec["term_roles"] else set()
            if "primary" in term_roles_set:
                term_role = "primary"
        role_in_anchor = (rec.get("role_in_anchor") or "").strip().lower()
        can_expand = bool(rec.get("can_expand"))
        retain_mode = (rec.get("retain_mode") or "normal").strip().lower()
        if (
            term_role == "primary"
            and role_in_anchor == "mainline"
            and can_expand
        ):
            rec["stage3_entry_group"] = "trusted_primary"
        elif term_role == "primary":
            rec["stage3_entry_group"] = "secondary_primary"
        else:
            rec["stage3_entry_group"] = "support_expansion"
    return terms


def _check_stage3_admission(
    term: Dict[str, Any],
    jd_profile: Any,
    active_domains: Any,
) -> Dict[str, Any]:
    """
    分层审查：trusted_primary 默认不 hard_drop；secondary 可降分不轻易杀；support_expansion 才严格。
    返回 hard_drop, reason, risk_flags, admission_strength。
    """
    group = term.get("stage3_entry_group") or "support_expansion"
    risk_flags: List[str] = []
    poly = float(term.get("polysemy_risk") or 0.0)
    obj = float(term.get("object_like_risk") or 0.0)
    generic = float(term.get("generic_risk") or 0.0)
    anchor_id = float(term.get("best_anchor_identity") or term.get("best_identity_score") or 0.0)
    jd_align = float(term.get("best_jd_align") or 0.0)
    ctx = float(term.get("best_context_continuity") or 0.0)
    if poly > 0.75:
        risk_flags.append("high_polysemy")
    if obj > 0.55:
        risk_flags.append("object_like")
    if generic > 0.55:
        risk_flags.append("too_generic")
    term["risk_flags"] = risk_flags

    if group == "trusted_primary":
        return {
            "hard_drop": False,
            "reason": "",
            "risk_flags": risk_flags,
            "admission_strength": "trusted",
        }
    if group == "secondary_primary":
        if anchor_id < 0.16 and jd_align < 0.72 and ctx < 0.48:
            return {
                "hard_drop": True,
                "reason": "secondary_too_weak",
                "risk_flags": risk_flags,
                "admission_strength": "weak",
            }
        return {
            "hard_drop": False,
            "reason": "",
            "risk_flags": risk_flags,
            "admission_strength": "secondary",
        }
    if group == "support_expansion":
        if anchor_id < 0.14 and ctx < 0.50:
            return {
                "hard_drop": True,
                "reason": "weak_evidence_noise",
                "risk_flags": risk_flags,
                "admission_strength": "support",
            }
        if obj > 0.70 and generic > 0.60:
            return {
                "hard_drop": True,
                "reason": "object_generic_noise",
                "risk_flags": risk_flags,
                "admission_strength": "support",
            }
        return {
            "hard_drop": False,
            "reason": "",
            "risk_flags": risk_flags,
            "admission_strength": "support",
        }
    return {
        "hard_drop": False,
        "reason": "",
        "risk_flags": risk_flags,
        "admission_strength": "support",
    }


def _compute_stage3_global_consensus(
    recall,
    candidates: List[Dict[str, Any]],
) -> None:
    """
    只计算非 cluster 的全局信号：cross_anchor_evidence（参与最终分）、semantic_drift_risk（仅 debug）。
    不再写入任何 cluster 相关字段。
    """
    if not candidates:
        return
    all_anchors: Set[str] = set()
    for rec in candidates:
        for a in rec.get("parent_anchors") or []:
            if a:
                all_anchors.add(a)
    n_anchors = max(1, len(all_anchors))
    for rec in candidates:
        anchor_count = int(rec.get("anchor_count") or 0)
        evidence_count = int(rec.get("evidence_count") or 0)
        outside = float(rec.get("outside_subfield_mass") or 0.0)
        topic_fit = rec.get("topic_fit")
        if topic_fit is None:
            topic_fit = rec.get("subfield_fit")
        if topic_fit is None:
            topic_fit = rec.get("field_fit")
        if topic_fit is None:
            topic_fit = rec.get("domain_fit")
        topic_fit = float(topic_fit or 0.5)
        anchor_part = min(1.0, anchor_count / max(2.0, n_anchors))
        evidence_part = min(1.0, evidence_count / 3.0)
        cross_anchor = 0.90 + 0.12 * anchor_part + 0.08 * evidence_part
        cross_anchor = min(1.10, max(0.90, cross_anchor))
        drift = (
            0.45 * min(1.0, outside)
            + 0.25 * (1.0 - topic_fit)
            + 0.30 * (1.0 / max(1.0, anchor_count))
        )
        drift = min(1.0, max(0.0, drift))
        rec["cross_anchor_evidence"] = cross_anchor
        rec["semantic_drift_risk"] = drift


def _build_family_buckets(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按 family_key 分桶。"""
    family_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        fk = rec.get("family_key") or build_family_key(rec)
        rec["family_key"] = fk
        family_buckets[fk].append(rec)
    return family_buckets


def _compute_family_centrality(rec: Dict[str, Any], family_buckets: Dict[str, List[Dict[str, Any]]]) -> float:
    """
    衡量 term 在 family 内是否为「中心项」：has_primary_role、seed score 靠前、锚点与证据数。
    """
    fk = rec.get("family_key") or build_family_key(rec)
    members = family_buckets.get(fk) or [rec]
    scores = [float(m.get("best_seed_score") or m.get("score") or 0.0) for m in members]
    max_score = max(scores) if scores else 1.0
    cur_score = float(rec.get("best_seed_score") or rec.get("score") or 0.0)
    rank_score = cur_score / max(1e-6, max_score)
    role_bonus = 1.0 if rec.get("has_primary_role") else 0.85
    anchor_bonus = min(1.0, 0.65 + 0.10 * int(rec.get("anchor_count") or 0))
    evidence_bonus = min(1.0, 0.70 + 0.10 * int(rec.get("evidence_count") or 0))
    centrality = (
        0.45 * rank_score
        + 0.25 * role_bonus
        + 0.15 * anchor_bonus
        + 0.15 * evidence_bonus
    )
    return min(1.0, max(0.0, centrality))


def _apply_family_role_constraints(survivors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    防止 support 系统性压过 primary：每 family 至少保证一个 primary 可见；
    support 只有显著优于 primary 才允许压过，否则只能作为补充。
    """
    family_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in survivors:
        family_buckets[rec.get("family_key", "")].append(rec)
    for fk, members in family_buckets.items():
        primaries = [m for m in members if (m.get("retrieval_role") or "").lower() == "paper_primary"]
        supports = [m for m in members if (m.get("retrieval_role") or "").lower() == "paper_support"]
        best_primary = None
        if primaries:
            best_primary = max(primaries, key=lambda x: float(x.get("final_score") or 0.0))
            best_primary["final_score"] = float(best_primary.get("final_score") or 0.0) * 1.05
            best_primary["primary_visibility_boost"] = True
        if best_primary and supports:
            p_score = float(best_primary.get("final_score") or 0.0)
            for s in supports:
                s_score = float(s.get("final_score") or 0.0)
                if s_score < p_score * 1.10:
                    s["final_score"] = min(s_score, p_score * 0.98)
                    s["support_over_primary_suppressed"] = True
    return survivors


def _collect_risky_reasons(rec: Dict[str, Any]) -> List[str]:
    """Risky 理由：弱 family、高 drift+低 ptc、弱 topic 尾部；判定更严，避免正常主干词被误标。"""
    reasons: List[str] = []
    final_score = float(rec.get("final_score") or 0.0)
    drift = float(rec.get("semantic_drift_risk") or 0.0)
    ex = rec.get("stage3_explain") or {}
    ptc = float(ex.get("path_topic_consistency") or 0.0)
    fc = float(rec.get("family_centrality") or 0.0)
    if fc < 0.30:
        reasons.append("weak_family_centrality")
    if drift > 0.75 and ptc < 0.30:
        reasons.append("high_drift_risk")
    if ptc < 0.50 and final_score < 0.58:
        reasons.append("weak_topic_fit_tail")
    return reasons


def _collect_stage3_bucket_reason_flags(rec: Dict[str, Any]) -> List[str]:
    """判定链路：为何落入 core/support/risky，供 [Stage3 bucket reason] 打印。"""
    flags: List[str] = []
    mainline_hits = int(rec.get("mainline_hits") or 0)
    anchor_count = int(rec.get("anchor_count") or 0)
    can_expand = bool(rec.get("can_expand", False))
    identity_factor = float((rec.get("stage3_explain") or {}).get("identity_factor", 1.0) or 1.0)
    obj = float(rec.get("object_like_risk") or 0.0)
    generic = float(rec.get("generic_risk") or 0.0)
    primary_bucket = (rec.get("primary_bucket") or "").strip().lower()
    primary_reason = (rec.get("primary_reason") or "").strip().lower()
    fallback_primary = bool(rec.get("fallback_primary", False))
    mainline_candidate = bool(rec.get("mainline_candidate", False))
    can_expand_local = bool(rec.get("can_expand_local", can_expand))

    is_fallback_primary = (
        fallback_primary
        or primary_bucket == "primary_fallback_keep_no_expand"
        or primary_reason == "anchor_core_fallback"
    )

    is_locked_mainline = (
        primary_bucket == "primary_keep_no_expand"
        or primary_reason == "usable_mainline_no_expand"
        or (mainline_candidate and (not can_expand_local))
    )

    has_mainline_support = (mainline_hits >= 1) or can_expand or is_locked_mainline

    if not has_mainline_support and anchor_count > 0:
        flags.append("no_mainline_support")
    if (not has_mainline_support) and (not is_fallback_primary):
        flags.append("only_weak_keep_sources")
    if is_locked_mainline:
        flags.append("locked_mainline_no_expand")
    if is_fallback_primary:
        flags.append("fallback_primary")
    if anchor_count >= 2 and mainline_hits == 0:
        flags.append("cross_anchor_but_side_only")
    if _stage3_is_conditioned_only(rec):
        flags.append("conditioned_only")
    if identity_factor < 0.60:
        flags.append("identity_low_family")
    if obj >= 0.50:
        flags.append("object_like")
    if generic >= 0.50:
        flags.append("generic_like")
    if rec.get("stage3_support_demote_reason") == "single_anchor_expand_branch":
        flags.append("single_anchor_expand_branch_demote")
    return flags


def _bucket_stage3_terms(survivors: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    分为 core_terms、support_terms、risky_terms。
    若已有 stage3_bucket 则直接按桶分组；否则回退到原逻辑（强主干/结构型 core、support、risky）。
    """
    core_terms: List[Dict[str, Any]] = []
    support_terms: List[Dict[str, Any]] = []
    risky_terms: List[Dict[str, Any]] = []
    for rec in survivors:
        bucket = (rec.get("stage3_bucket") or "").strip().lower()
        if bucket == "core":
            core_terms.append(rec)
            continue
        if bucket == "support":
            support_terms.append(rec)
            continue
        if bucket == "risky":
            risky_terms.append(rec)
            continue
        ex = rec.get("stage3_explain") or {}
        final_score = float(rec.get("final_score") or 0.0)
        ptc = float(ex.get("path_topic_consistency") or 0.0)
        cross = float(ex.get("cross_anchor_factor") or rec.get("cross_anchor_evidence") or 1.0)
        role = (rec.get("retrieval_role") or "").lower()
        reasons = rec.get("risk_reasons") or []
        is_primary = role == "paper_primary"
        strong_primary_core = is_primary and final_score >= 0.66
        structured_primary_core = (
            is_primary and final_score >= 0.62 and ptc >= 0.55 and cross >= 0.94
        )
        if strong_primary_core or structured_primary_core:
            core_terms.append(rec)
            continue
        if final_score >= 0.56 and "high_drift_risk" not in reasons:
            support_terms.append(rec)
            continue
        risky_terms.append(rec)
    return core_terms, support_terms, risky_terms


def _debug_print_stage3_bucket_details(
    core_terms: List[Dict[str, Any]],
    support_terms: List[Dict[str, Any]],
    risky_terms: List[Dict[str, Any]],
    recall: Any,
) -> None:
    """打印 bucket 判定依据：term | bucket | final | ptc | cross | reasons。"""
    debug_print(2, "[Stage3 Bucket Details] term | bucket | final | ptc | cross | reasons", recall)
    for bucket_name, items in [
        ("core", core_terms),
        ("support", support_terms),
        ("risky", risky_terms),
    ]:
        for rec in items:
            explain = rec.get("stage3_explain") or {}
            final_score = float(rec.get("final_score") or 0.0)
            ptc = float(explain.get("path_topic_consistency") or 0.0)
            cross = float(
                explain.get("cross_anchor_factor")
                or rec.get("cross_anchor_evidence")
                or 1.0
            )
            reasons = rec.get("risk_reasons") or []
            term = (rec.get("term") or "")[:32]
            debug_print(
                2,
                f"  {term!r} | {bucket_name} | {final_score:.4f} | {ptc:.4f} | {cross:.4f} | {reasons}",
                recall,
            )


def _stage3_single_anchor_expand_branch_demote(term: Dict[str, Any]) -> bool:
    """
    单锚、仅 1 次 mainline 命中且 Stage2 可扩：若 family/topic 不在控制-规划主线中心，
    或 object/drift/generic/poly 结构偏支，则不应轻松进 support（压到 risky）。
    """
    anchor_count = int(term.get("anchor_count") or 0)
    mainline_hits = int(term.get("mainline_hits") or 0)
    can_expand = bool(term.get("can_expand", False))
    if anchor_count != 1 or mainline_hits < 1 or not can_expand:
        return False
    ex = term.get("stage3_explain") or {}
    fc = float(term.get("family_centrality") or ex.get("family_centrality") or 0.0)
    ptc = float(ex.get("path_topic_consistency") or 0.0)
    obj = float(term.get("object_like_risk") or 0.0)
    drift = float(term.get("semantic_drift_risk") or 0.0)
    generic = float(term.get("generic_risk") or 0.0)
    poly = float(term.get("polysemy_risk") or 0.0)
    if fc < STAGE3_SUPPORT_DEMOTE_FC_MAX and ptc < STAGE3_SUPPORT_DEMOTE_PTC_MAX:
        return True
    if obj >= STAGE3_SUPPORT_DEMOTE_OBJ_MIN:
        return True
    if drift >= STAGE3_SUPPORT_DEMOTE_DRIFT_MIN and ptc < STAGE3_SUPPORT_DEMOTE_DRIFT_PTC_MAX:
        return True
    if generic >= STAGE3_SUPPORT_DEMOTE_GENERIC_MIN:
        return True
    if poly >= STAGE3_SUPPORT_DEMOTE_POLY_MIN and ptc < STAGE3_SUPPORT_DEMOTE_POLY_PTC_MAX:
        return True
    return False


def _compute_stage3_risk_penalty(term: Dict[str, Any]) -> float:
    """
    风险仅降分不硬杀：trusted_primary 轻罚，support_expansion 重罚。
    """
    group = (term.get("stage3_entry_group") or "support_expansion")
    poly = float(term.get("polysemy_risk") or 0.0)
    obj = float(term.get("object_like_risk") or 0.0)
    generic = float(term.get("generic_risk") or 0.0)
    if group == "trusted_primary":
        penalty = 1.0 - 0.10 * poly - 0.06 * obj - 0.05 * generic
    elif group == "secondary_primary":
        penalty = 1.0 - 0.16 * poly - 0.08 * obj - 0.08 * generic
    else:
        penalty = 1.0 - 0.22 * poly - 0.14 * obj - 0.12 * generic
    return max(0.65, min(1.0, penalty))


def _assign_stage3_bucket(term: Dict[str, Any]) -> str:
    """
    Stage3 分桶（两函数硬门之一）：conditioned_only 永不得 core；并消费 Stage2A 透传语义。
    - is_fallback_primary：fallback 保线词，默认 risky（与正常 primary 区分）。
    - is_locked_mainline：primary_keep_no_expand / usable_mainline_no_expand 等「主线但不可扩」→ support，避免误伤 risky。
    - has_mainline_support：强主线（mainline_hits≥1 或 can_expand）或 locked mainline。
    """
    term.pop("stage3_support_demote_reason", None)
    anchor_count = int(term.get("anchor_count") or 0)
    if anchor_count < 1:
        anchor_count = 1
    mainline_hits = int(term.get("mainline_hits") or 0)
    can_expand = bool(term.get("can_expand", False))
    ex = term.get("stage3_explain") or {}
    identity_factor = float(ex.get("identity_factor", 1.0) or 1.0)

    primary_bucket = (term.get("primary_bucket") or "").strip().lower()
    primary_reason = (term.get("primary_reason") or "").strip().lower()
    fallback_primary = bool(term.get("fallback_primary", False))
    mainline_candidate = bool(term.get("mainline_candidate", False))
    can_expand_local = bool(term.get("can_expand_local", can_expand))

    is_fallback_primary = (
        fallback_primary
        or primary_bucket == "primary_fallback_keep_no_expand"
        or primary_reason == "anchor_core_fallback"
    )

    is_locked_mainline = (
        primary_bucket == "primary_keep_no_expand"
        or primary_reason == "usable_mainline_no_expand"
        or (mainline_candidate and (not can_expand_local))
    )

    strong_mainline_support = mainline_hits >= 1 or can_expand
    has_mainline_support = strong_mainline_support or is_locked_mainline

    st = _stage3_normalized_source_types(term)
    has_similar = "similar_to" in st
    has_family = "family_landing" in st
    has_conditioned = "conditioned_vec" in st
    conditioned_only = bool(has_conditioned and (not has_similar) and (not has_family))

    if is_fallback_primary:
        return "risky"

    if conditioned_only and anchor_count <= 1:
        return "support" if has_mainline_support else "risky"
    if conditioned_only and anchor_count >= 2:
        return "support"

    if is_locked_mainline:
        return "support"

    core_ok = (
        mainline_hits >= 1
        and can_expand
        and identity_factor >= 0.95
    )
    if core_ok:
        return "core"

    bucket = "support" if has_mainline_support else "risky"

    if bucket == "support" and _stage3_single_anchor_expand_branch_demote(term):
        term["stage3_support_demote_reason"] = "single_anchor_expand_branch"
        return "risky"
    return bucket


def _is_primary_like(rec: Dict[str, Any]) -> bool:
    """primary-like 主落点：只做软惩罚，不走严格 topic gate。"""
    source_type = (rec.get("source_type") or rec.get("source") or rec.get("origin") or "").strip().lower()
    term_role = (rec.get("term_role") or "").strip().lower()
    return (
        bool(rec.get("has_primary_role"))
        or term_role == "primary"
        or source_type == "similar_to"
        or int(rec.get("anchor_count") or 0) >= 2
    )


def _debug_print_stage3_tables(
    layered_terms: List[Dict[str, Any]],
    dropped_with_reason: List[Dict[str, Any]],
    survivors: List[Dict[str, Any]],
    recall: Any,
) -> None:
    """Stage3 汇总：条数 + 前3样本，不打印完整表。"""
    stage3_debug = getattr(term_scoring, "STAGE3_DEBUG", False)
    label_trace = LABEL_PATH_TRACE or stage3_debug
    if not (label_trace or stage3_debug):
        return
    entry_sample = [rec.get("term") for rec in layered_terms[:3]]
    drop_sample = [rec.get("term") for rec in dropped_with_reason[:3]]
    surv_sample = [rec.get("term") for rec in survivors[:3]]
    debug_print(2, f"[Stage3] entry 共 {len(layered_terms)} 条 前3: {entry_sample} | dropped {len(dropped_with_reason)} 前3: {drop_sample} | survivors {len(survivors)} 前3: {surv_sample}", recall)


def _debug_print_stage3_input(raw_candidates: List[Dict[str, Any]]) -> None:
    """Stage3 输入：条数 + 前3样本，不打印全表。"""
    if not raw_candidates:
        return
    sample = [(rec.get("term") or "", rec.get("source") or rec.get("origin") or "") for rec in raw_candidates[:3]]
    print(f"[stage3_input] 共 {len(raw_candidates)} 条 前3: {sample}")


def _write_term_maps(
    score_map: Dict[str, float],
    term_map: Dict[str, str],
    idf_map: Dict[str, float],
    term_role_map: Dict[str, str],
    term_source_map: Dict[str, str],
    parent_anchor_map: Dict[str, str],
    parent_primary_map: Dict[str, str],
    recall: Any,
    rec: Dict[str, Any],
    query_vector: Any = None,
) -> None:
    """将一条幸存词写入各 map 与 tag_purity_debug。"""
    tid_str = str(rec.get("tid", ""))
    if not tid_str or tid_str == "None":
        return
    score_map[tid_str] = float(rec["final_score"])
    term_map[tid_str] = rec.get("term") or ""
    term_role_map[tid_str] = rec.get("term_role") or "primary"
    term_source_map[tid_str] = rec.get("source") or rec.get("origin") or ""
    parent_anchor_map[tid_str] = rec.get("parent_anchor") or ""
    parent_primary_map[tid_str] = rec.get("parent_primary") or ""
    degree_w = int(rec.get("degree_w") or 0)
    total = float(getattr(recall, "total_work_count", 1e6) or 1e6)
    idf_map[tid_str] = term_scoring._smoothed_idf(
        degree_w,
        term_scoring._idf_backbone(total, int(rec.get("degree_w_expanded") or 0) or max(degree_w, 1)),
    )
    entry = {
        "tid": rec.get("tid"),
        "term": rec.get("term"),
        "term_role": rec.get("term_role"),
        "identity_score": rec.get("identity_score"),
        "quality_score": rec.get("quality_score"),
        "final_score": rec.get("final_score"),
        "degree_w": rec.get("degree_w"),
        "degree_w_expanded": rec.get("degree_w_expanded"),
        "source": rec.get("source") or rec.get("origin"),
        "domain_fit": rec.get("domain_fit"),
        "parent_anchor": rec.get("parent_anchor"),
        "parent_primary": rec.get("parent_primary"),
        "anchor_count": rec.get("anchor_count"),
        "evidence_count": rec.get("evidence_count"),
        "family_centrality": rec.get("family_centrality"),
        "path_topic_consistency": (rec.get("stage3_explain") or {}).get("path_topic_consistency"),
        "generic_penalty": (rec.get("stage3_explain") or {}).get("generic_penalty"),
        "cross_anchor_factor": (rec.get("stage3_explain") or {}).get("cross_anchor_factor"),
        "retrieval_role": rec.get("retrieval_role"),
    }
    if rec.get("stage3_explain"):
        entry.update(rec["stage3_explain"])
    debug_metrics = term_scoring.get_term_debug_metrics(recall, rec, query_vector)
    if debug_metrics:
        entry.update(debug_metrics)
    recall.debug_info.tag_purity_debug.append(entry)


def _write_term_maps_if_missing(
    score_map: Dict[str, float],
    term_map: Dict[str, str],
    idf_map: Dict[str, float],
    term_role_map: Dict[str, str],
    term_source_map: Dict[str, str],
    parent_anchor_map: Dict[str, str],
    parent_primary_map: Dict[str, str],
    recall: Any,
    rec: Dict[str, Any],
    query_vector: Any = None,
) -> None:
    """仅当 tid 尚未在 score_map 时写入各 map，避免 paper_terms 与 top_survivors 重复写入。"""
    tid_str = str(rec.get("tid", ""))
    if not tid_str or tid_str == "None" or tid_str in score_map:
        return
    _write_term_maps(
        score_map, term_map, idf_map, term_role_map, term_source_map,
        parent_anchor_map, parent_primary_map, recall, rec, query_vector,
    )


def _stage3_audit_filter_rows(rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """审计表行裁剪：若 STAGE3_DEBUG_FOCUS_TERMS 非空则只保留 term 命中集合的行。"""
    if not rows:
        return []
    if STAGE3_DEBUG_FOCUS_TERMS:
        rows = [r for r in rows if (r.get("term") or "").strip() in STAGE3_DEBUG_FOCUS_TERMS]
    return rows[:top_k]


def _print_stage3_final_adjust_audit(cands: List[Dict[str, Any]], top_k: int = 12) -> None:
    """
    [Stage3 final adjust audit]
    对照 stage3_build_score_map：pre_adjust（进入第二段乘子前）→ risk_mult × cross_bonus → post_adjust（当前 final_score）。
    依赖 rec['_stage3_pre_adjust_score']（由 stage3_build_score_map 写入）；缺失时用 post/(risk×cross) 反推。
    """
    if not STAGE3_AUDIT_DEBUG or not cands:
        return
    ranked = sorted(cands, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    ranked = _stage3_audit_filter_rows(ranked, top_k)
    if not ranked:
        print("\n" + "-" * 80 + "\n[Stage3 final adjust audit] (no rows after focus filter)\n" + "-" * 80)
        return

    print("\n" + "-" * 80)
    print("[Stage3 final adjust audit]")
    print("-" * 80)
    print(
        "term                         | bucket   | pre_adj  | risk_mlt | cross_bn | post_adj | "
        "anchor_count | mainline_hits | can_exp | cond_only | flags"
    )
    for rec in ranked:
        term = (rec.get("term") or "")[:28]
        bucket = (rec.get("stage3_bucket") or "")[:8]
        ex = rec.get("stage3_explain") or {}
        risk_mult = float(ex.get("mainline_risk_penalty") or 1.0)
        cross_bonus = float(ex.get("cross_anchor_score_bonus") or 1.0)
        post_adjust = float(rec.get("final_score") or 0.0)
        pre_adjust = rec.get("_stage3_pre_adjust_score")
        if pre_adjust is None:
            denom = risk_mult * cross_bonus
            pre_adjust = post_adjust / denom if denom else post_adjust
        else:
            pre_adjust = float(pre_adjust or 0.0)
        anchor_count = int(rec.get("anchor_count") or 0)
        mainline_hits = int(rec.get("mainline_hits") or 0)
        can_expand = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
        conditioned_only = _stage3_is_conditioned_only(rec)
        flags = list(rec.get("bucket_reason_flags") or [])
        print(
            f"{term:28} | {bucket:8} | {pre_adjust:8.4f} | {risk_mult:8.4f} | {cross_bonus:8.4f} | {post_adjust:8.4f} | "
            f"{anchor_count:^12d} | {mainline_hits:^13d} | {str(can_expand):^7} | {str(conditioned_only):^9} | {flags}"
        )


def _print_stage3_cross_anchor_audit(cands: List[Dict[str, Any]], top_k: int = 40) -> None:
    """
    [Stage3 cross-anchor audit]
    只列 anchor_count>=2：看跨锚词拿到的 cross_anchor_score_bonus 与桶、主线证据是否一致。
    """
    if not STAGE3_AUDIT_DEBUG or not cands:
        return
    focus = [c for c in cands if int(c.get("anchor_count") or 0) >= 2]
    focus = sorted(
        focus,
        key=lambda x: (int(x.get("anchor_count") or 0), float(x.get("final_score") or 0.0)),
        reverse=True,
    )
    focus = _stage3_audit_filter_rows(focus, top_k)
    if not focus:
        print("\n" + "-" * 80 + "\n[Stage3 cross-anchor audit] (no multi-anchor rows / focus empty)\n" + "-" * 80)
        return

    print("\n" + "-" * 80)
    print("[Stage3 cross-anchor audit]")
    print("-" * 80)
    print(
        "term                     | anchor_count | anchors                  | evidence_cnt | "
        "mainline_hits | can_exp | bucket   | cross_bn   | flags"
    )
    for rec in focus:
        term = (rec.get("term") or "")[:24]
        anchor_count = int(rec.get("anchor_count") or 0)
        anchors = rec.get("parent_anchors") or []
        if isinstance(anchors, (list, tuple, set)):
            anchors_show = ",".join(str(a) for a in list(anchors)[:4])
        else:
            anchors_show = str(anchors)[:24]
        evidence_count = int(rec.get("evidence_count") or 0)
        mainline_hits = int(rec.get("mainline_hits") or 0)
        can_expand = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
        bucket = (rec.get("stage3_bucket") or "")[:8]
        ex = rec.get("stage3_explain") or {}
        cross_bonus = float(ex.get("cross_anchor_score_bonus") or 1.0)
        flags = list(rec.get("bucket_reason_flags") or [])
        print(
            f"{term:24} | {anchor_count:^12d} | {anchors_show[:24]:24} | {evidence_count:^12d} | "
            f"{mainline_hits:^13d} | {str(can_expand):^7} | {bucket:8} | {cross_bonus:10.4f} | {flags}"
        )


def _print_stage3_support_risky_concise(cands: List[Dict[str, Any]], top_k: int = 20) -> None:
    """[Stage3 support/risky concise] 只列 support / risky，快速扫可疑项。"""
    if not STAGE3_AUDIT_DEBUG or not cands:
        return
    focus = [
        c for c in cands
        if (c.get("stage3_bucket") or "").strip().lower() in ("support", "risky")
    ]
    focus = sorted(focus, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    focus = _stage3_audit_filter_rows(focus, top_k)
    if not focus:
        print("\n" + "-" * 80 + "\n[Stage3 support/risky concise] (no rows / focus empty)\n" + "-" * 80)
        return

    print("\n" + "-" * 80)
    print("[Stage3 support/risky concise]")
    print("-" * 80)
    print(
        "term                     | bucket   | final_score | mainline_hits | can_exp | "
        "cond_only | weak_keep | xa>=2"
    )
    for rec in focus:
        term = (rec.get("term") or "")[:24]
        bucket = (rec.get("stage3_bucket") or "")[:8]
        final_score = float(rec.get("final_score") or 0.0)
        mainline_hits = int(rec.get("mainline_hits") or 0)
        can_expand = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
        conditioned_only = _stage3_is_conditioned_only(rec)
        reason_flags = set(rec.get("bucket_reason_flags") or [])
        weak_keep_only = "only_weak_keep_sources" in reason_flags
        cross_anchor = int(rec.get("anchor_count") or 0) >= 2
        print(
            f"{term:24} | {bucket:8} | {final_score:11.4f} | {mainline_hits:^13d} | {str(can_expand):^7} | "
            f"{str(conditioned_only):^9} | {str(weak_keep_only):^9} | {str(cross_anchor):^5}"
        )


def _print_stage3_to_paper_bridge(cands: List[Dict[str, Any]], top_k: int = 20) -> None:
    """
    [Stage3->Paper bridge]
    在 select_terms_for_paper_recall 为每条写好 selected / reject 之后调用：
    bucket 与 paper 去留同一行对照。
    """
    if not STAGE3_AUDIT_DEBUG or not cands:
        return
    ranked = sorted(cands, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    ranked = _stage3_audit_filter_rows(ranked, top_k)
    if not ranked:
        print("\n" + "-" * 80 + "\n[Stage3->Paper bridge] (no rows after focus filter)\n" + "-" * 80)
        return

    print("\n" + "-" * 80)
    print("[Stage3->Paper bridge]")
    print("-" * 80)
    print(
        "term                     | bucket   | final_score | selected_for_paper | select_reason          | "
        "reject_reason            | ml_hits | can_exp | source_type    | parent_anchor"
    )
    for rec in ranked:
        term = (rec.get("term") or "")[:24]
        bucket = (rec.get("stage3_bucket") or "")[:8]
        final_score = float(rec.get("final_score") or 0.0)
        selected = bool(rec.get("selected_for_paper", False))
        select_reason = str(rec.get("select_reason") or "")
        reject_reason = str(rec.get("paper_reject_reason") or "")
        mainline_hits = int(rec.get("mainline_hits") or 0)
        can_expand = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
        st = rec.get("source_type") or rec.get("source") or ""
        if not st and rec.get("source_types"):
            st = ",".join(sorted(str(x) for x in list(rec.get("source_types"))[:3]))
        source_type = str(st)[:14]
        parent_anchor = str(rec.get("parent_anchor") or "")[:18]
        print(
            f"{term:24} | {bucket:8} | {final_score:11.4f} | {str(selected):^18} | {select_reason[:22]:22} | "
            f"{reject_reason[:24]:24} | {mainline_hits:^7d} | {str(can_expand):^7} | {source_type:14} | {parent_anchor:18}"
        )


def _print_paper_term_selection_audit(ordered: List[Dict[str, Any]], limit: int = 45) -> None:
    """按 core→support→risky、同桶内 final_score 顺序，打印每条 paper 门结果（含未入选）。"""
    if not STAGE3_PAPER_GATE_DEBUG or not ordered:
        return
    print(f"\n{'-' * 80}\n[Stage3] Paper term selection gate (audit)\n{'-' * 80}")
    for rec in ordered[:limit]:
        term = (rec.get("term") or "")[:48]
        fs = float(rec.get("final_score") or 0.0)
        print(f"[paper_term_selection gate] term={term!r}")
        print(f"  bucket={rec.get('stage3_bucket')!r}")
        print(f"  final_score={fs:.4f}")
        print(f"  selected_for_paper={rec.get('selected_for_paper', False)}")
        print(f"  select_reason={rec.get('select_reason')!r}")
        print(f"  paper_reject_reason={repr(rec.get('paper_reject_reason') or '')}")
        print(f"  retrieval_role={rec.get('retrieval_role')!r}")
        print(f"  reason_flags={rec.get('bucket_reason_flags') or []}")


def select_terms_for_paper_recall(
    records: List[Dict[str, Any]],
    max_terms: int = 12,
) -> List[Dict[str, Any]]:
    """
    Paper 召回选词（两函数硬门之二）：
    - conditioned_only 且单锚 → 不进 paper（conditioned_only_single_anchor_block）。
    - fallback_primary（含 Stage2A primary_fallback / anchor_core_fallback）→ fallback_primary_block。
    - core 直接入选；support 需要“主线 grounding + 语义质量”；risky 不进。
    - 同 family_key 去重；遍历顺序 core→support→risky，同桶内按 final_score 降序。
    """
    if not records:
        return []

    def _bucket_sort_key(r: Dict[str, Any]) -> Tuple[int, float]:
        b = (r.get("stage3_bucket") or "").strip().lower()
        pri = 0 if b == "core" else (1 if b == "support" else 2)
        return pri, -float(r.get("final_score") or 0.0)

    ordered = sorted(records, key=_bucket_sort_key)
    used_family: Set[str] = set()
    selected: List[Dict[str, Any]] = []

    def _support_grounded_enough(rec: Dict[str, Any]) -> bool:
        """
        支撑词 paper 允许准入的“grounded support”门：
        - 避免 conditioned_only 抢占 paper 配额
        - 需要交叉锚证据（anchor_count & mainline_hits）托住
        - 用风险项做保守兜底（不再强依赖 context_continuity）
        """
        st = _stage3_normalized_source_types(rec)
        has_similar = "similar_to" in st
        has_family = "family_landing" in st
        has_conditioned = "conditioned_vec" in st
        conditioned_only = bool(has_conditioned and (not has_similar) and (not has_family))

        # fallback 只保留，不进 paper
        if bool(rec.get("fallback_primary", False)):
            return False

        if conditioned_only:
            return False

        final_score = float(rec.get("final_score") or 0.0)
        if final_score < 0.46:
            return False

        mainline_hits = int(rec.get("mainline_hits", 0) or 0)
        anchor_count = int(rec.get("anchor_count", 1) or 1)

        grounded_support_ok = (
            (anchor_count >= 2 and mainline_hits >= 1)
            or (mainline_hits >= 2)
        )
        if not grounded_support_ok:
            return False

        generic_risk = float(rec.get("generic_risk", 0.0) or 0.0)
        polysemy_risk = float(rec.get("polysemy_risk", 0.0) or 0.0)
        object_like_risk = float(rec.get("object_like_risk", 0.0) or 0.0)

        return (
            generic_risk <= 0.75
            and object_like_risk <= 0.50
            and polysemy_risk <= 0.65
        )

    for rec in ordered:
        anchor_count = int(rec.get("anchor_count") or 1) or 1
        mainline_hits = int(rec.get("mainline_hits") or 0)
        can_expand = bool(rec.get("can_expand", False))
        bucket = (rec.get("stage3_bucket") or "").strip().lower()

        primary_bucket = (rec.get("primary_bucket") or "").strip().lower()
        primary_reason = (rec.get("primary_reason") or "").strip().lower()
        fallback_primary = bool(rec.get("fallback_primary", False))
        mainline_candidate = bool(rec.get("mainline_candidate", False))
        can_expand_local = bool(rec.get("can_expand_local", can_expand))

        is_fallback_primary = (
            fallback_primary
            or primary_bucket == "primary_fallback_keep_no_expand"
            or primary_reason == "anchor_core_fallback"
        )

        is_locked_mainline = (
            primary_bucket == "primary_keep_no_expand"
            or primary_reason == "usable_mainline_no_expand"
            or (mainline_candidate and (not can_expand_local))
        )

        has_mainline_support = (
            (mainline_hits >= 1)
            or can_expand
            or is_locked_mainline
        )

        st = _stage3_normalized_source_types(rec)
        has_similar = "similar_to" in st
        has_family = "family_landing" in st
        has_conditioned = "conditioned_vec" in st
        conditioned_only = bool(has_conditioned and (not has_similar) and (not has_family))

        rec["selected_for_paper"] = False
        rec["select_reason"] = ""
        rec["paper_reject_reason"] = ""

        if conditioned_only and anchor_count <= 1:
            rec["paper_reject_reason"] = "conditioned_only_single_anchor_block"
            continue

        if is_fallback_primary:
            rec["paper_reject_reason"] = "fallback_primary_block"
            continue

        fk = rec.get("family_key") or build_family_key(rec)
        rec["family_key"] = fk

        sel = False
        reason = ""
        rrole = ""

        if bucket == "core":
            sel = True
            reason = "bucket_core_direct"
            rrole = "paper_primary"

        elif bucket == "support":
            support_ok = _support_grounded_enough(rec)
            if support_ok:
                sel = True
                reason = "support_mainline_or_expand_ok"
                rrole = "paper_support"
            else:
                rec["paper_reject_reason"] = "support_but_not_grounded"

        else:
            rec["paper_reject_reason"] = "risky_bucket_block"

        if not sel:
            continue

        if fk in used_family:
            rec["paper_reject_reason"] = "family_duplicate_block"
            continue

        rec["selected_for_paper"] = True
        rec["select_reason"] = reason
        rec["retrieval_role"] = rrole
        used_family.add(fk)
        selected.append(rec)
        if len(selected) >= max_terms:
            break

    # 与 bucket / final_score 同一批对象，在逐项写好 paper 字段后打横向对照表
    if STAGE3_AUDIT_DEBUG:
        _print_stage3_to_paper_bridge(ordered, top_k=20)
    _print_paper_term_selection_audit(ordered)
    return selected


def stage3_build_score_map(survivors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stage3 最终分数乘法修正（在 identity×risk×role 与 _assign_stage3_bucket 之后执行）：
    用 score_mult 压制 weak keep / conditioned_only / 无主线等；cross_bonus 仅在 core 或「强证据 support」上给条件跨锚加分；
    risky 永不拿跨锚正奖励；locked_mainline_no_expand 不参与跨锚正奖励，避免 keep_no_expand 仅靠多锚顶到前排。

    本次最小修正目标：
    1. 压低 conditioned_only + single_anchor 的 support 词
    2. conditioned_only 不再吃 support 的跨锚奖励
    3. 不改 Stage2，不改 paper gate，只改 Stage3 一个函数

    重排后重刷 bucket_reason_flags，使 [Stage3 bucket reason] / dominant factors 与最终序一致。
    """
    for rec in survivors:
        # 供 [Stage3 final adjust audit]：第二段乘子前的 final_score（identity×risk×role 后、×score_mult 前）
        rec["_stage3_pre_adjust_score"] = float(rec.get("final_score") or 0.0)

        reason_flags = set(rec.get("bucket_reason_flags") or [])
        bucket = (rec.get("stage3_bucket") or "").strip().lower()
        group = (rec.get("stage3_entry_group") or "").strip().lower()

        anchor_count = int(rec.get("anchor_count") or 1)
        mainline_hits = int(rec.get("mainline_hits") or 0)
        can_expand = bool(rec.get("can_expand", False))

        conditioned_only = _stage3_is_conditioned_only(rec)

        score_mult = 1.0

        # A. 风险压制（强于旧版 mainline_risk_penalty 链）
        if "no_mainline_support" in reason_flags:
            score_mult *= 0.72
        # 无主线证据且不可扩：在 A 基础上再压一档（「能进桶」≠「排位冲前」）
        if "no_mainline_support" in reason_flags and not can_expand:
            score_mult *= 0.82
        if "only_weak_keep_sources" in reason_flags:
            score_mult *= 0.72
        if "locked_mainline_no_expand" in reason_flags:
            score_mult *= 0.86
        if bucket == "risky":
            score_mult *= 0.82
        if group == "secondary_primary" and bucket != "core":
            score_mult *= 0.88

        # B. conditioned_only：本次唯一重点加强
        # 单锚 conditioned_only 在排序阶段就要明显压低，避免 Robotics / Robot control 的侧翼词顶到前排
        if conditioned_only:
            if anchor_count <= 1:
                # 原始基础压制
                score_mult *= 0.68

                # 单锚且主线命中弱，再压一档
                if mainline_hits <= 1:
                    score_mult *= 0.86

                # 单锚 conditioned_only 且位于 support，再压一档
                if bucket == "support":
                    score_mult *= 0.88

            elif mainline_hits == 0 and not can_expand:
                score_mult *= 0.80
            else:
                score_mult *= 0.90

        # C. 单锚、无主线命中、不可扩：与 weak 侧翼一致再压一档
        if anchor_count <= 1 and mainline_hits == 0 and not can_expand:
            score_mult *= 0.74

        # D. 跨锚奖励：按桶 + 主线证据条件发放
        # 本次修正：conditioned_only 不再拿 support 的跨锚正奖励
        cross_bonus = 1.0
        locked_no_xa_bonus = "locked_mainline_no_expand" in reason_flags
        if not locked_no_xa_bonus:
            if bucket == "core":
                if anchor_count >= 2:
                    cross_bonus *= 1.08
            elif bucket == "support":
                if (not conditioned_only) and anchor_count >= 2 and (mainline_hits >= 1 or can_expand):
                    cross_bonus *= 1.03
            # risky：保持 1.0

        # E. 跨锚出现但非强主线（无 mainline_hits、不可扩）：回收虚高
        if anchor_count >= 2 and mainline_hits == 0 and not can_expand:
            cross_bonus *= 0.92

        rec["final_score"] = float(rec.get("final_score") or 0.0) * score_mult * cross_bonus

        # --- DEBUG 1: Stage3 单词修正明细 ---
        if DEBUG_LABEL_PATH and (
            conditioned_only
            or bucket in ("support", "risky")
            or anchor_count >= 2
        ):
            term = rec.get("term", "")
            print(
                "[Stage3 score adjust] "
                f"term={term!r} | "
                f"bucket={bucket!r} group={group!r} | "
                f"anchors={anchor_count} mainline_hits={mainline_hits} can_expand={can_expand} | "
                f"conditioned_only={conditioned_only} | "
                f"score_mult={score_mult:.3f} cross_bonus={cross_bonus:.3f} | "
                f"final={float(rec.get('final_score') or 0.0):.4f}"
            )

        # --- DEBUG 2: 重点风险词专项 ---
        if DEBUG_LABEL_PATH:
            watch_terms = {
                "Educational robotics",
                "Telerobotics",
                "Robot manipulator",
                "robot control",
                "digital control",
                "Motion control",
            }
            term = rec.get("term", "")
            if term in watch_terms:
                print(
                    "[Stage3 watch] "
                    f"term={term!r} | "
                    f"bucket={bucket!r} | "
                    f"reason_flags={sorted(list(reason_flags))} | "
                    f"anchor_count={anchor_count} | "
                    f"mainline_hits={mainline_hits} | "
                    f"can_expand={can_expand} | "
                    f"conditioned_only={conditioned_only} | "
                    f"score_mult={score_mult:.3f} | "
                    f"cross_bonus={cross_bonus:.3f} | "
                    f"final={float(rec.get('final_score') or 0.0):.4f}"
                )

        ex = rec.get("stage3_explain") or {}
        ex["mainline_risk_penalty"] = score_mult
        ex["cross_anchor_score_bonus"] = cross_bonus
        ex["conditioned_only"] = conditioned_only
        rec["stage3_explain"] = ex

    survivors.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    for i, rec in enumerate(survivors):
        rec["stage3_rank"] = i
        rec["bucket_reason_flags"] = _collect_stage3_bucket_reason_flags(rec)

    # --- DEBUG 3: Stage3 排序后 TopN 总览 ---
    if DEBUG_LABEL_PATH:
        print("\n" + "-" * 80)
        print("[Stage3 rerank summary]")
        print("-" * 80)
        print("rank | term | bucket | conditioned_only | anchors | mainline_hits | can_expand | final")
        for i, rec in enumerate(survivors[:12], start=1):
            term = rec.get("term", "")
            bucket = rec.get("stage3_bucket", "")
            conditioned_only_row = _stage3_is_conditioned_only(rec)
            anchor_count_row = int(rec.get("anchor_count") or 1)
            mainline_hits_row = int(rec.get("mainline_hits") or 0)
            can_expand_row = bool(rec.get("can_expand", False))
            final_score = float(rec.get("final_score") or 0.0)
            print(
                f"{i:>4} | "
                f"{term[:28]:<28} | "
                f"{str(bucket):<7} | "
                f"{str(conditioned_only_row):<16} | "
                f"{anchor_count_row:^7} | "
                f"{mainline_hits_row:^13} | "
                f"{str(can_expand_row):<10} | "
                f"{final_score:.4f}"
            )

    return survivors


def _run_stage3_dual_gate(
    recall,
    raw_candidates: List[Dict[str, Any]],
    query_vector,
    anchor_vids=None,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], List[Dict[str, Any]]]:
    """
    Stage3 主流程：去重聚合 → 全局共识 → 轻硬过滤 + identity/topic gate → 唯一主分 score_term_record
    → family 角色约束 → risky/bucket → paper_terms 选词。彻底移除 cluster 依赖。
    """
    debug_print(1, "\n" + "-" * 80 + "\n[Stage3] Global Rerank\n" + "-" * 80, recall)
    score_map: Dict[str, float] = {}
    term_map: Dict[str, str] = {}
    idf_map: Dict[str, float] = {}
    term_role_map: Dict[str, str] = {}
    term_source_map: Dict[str, str] = {}
    parent_anchor_map: Dict[str, str] = {}
    parent_primary_map: Dict[str, str] = {}
    recall.debug_info.tag_purity_debug = []
    recall._last_tag_purity_debug = recall.debug_info.tag_purity_debug
    stage3_debug = getattr(term_scoring, "STAGE3_DEBUG", False)
    label_trace = LABEL_PATH_TRACE or stage3_debug
    survivors: List[Dict[str, Any]] = []
    dropped_with_reason: List[Dict[str, Any]] = []

    _debug_print_stage3_input(raw_candidates)
    merged_candidates = _merge_stage3_duplicates(raw_candidates)
    for i, rec in enumerate(merged_candidates):
        rec["stage2_rank"] = i
    _compute_stage3_global_consensus(recall, merged_candidates)
    family_buckets = _build_family_buckets(merged_candidates)
    for rec in merged_candidates:
        rec["family_centrality"] = _compute_family_centrality(rec, family_buckets)
    merged_candidates = _classify_stage3_entry_groups(merged_candidates)

    jd_profile = getattr(recall, "jd_profile", None)
    active_domains = getattr(recall, "active_domain_set", None)

    for rec in merged_candidates:
        rec["family_key"] = rec.get("family_key") or build_family_key(rec)
        if not rec.get("term_role") and rec.get("term_roles"):
            roles = list(rec["term_roles"]) if isinstance(rec["term_roles"], (list, set)) else [rec["term_roles"]]
            rec["term_role"] = "primary" if "primary" in roles else (roles[0] if roles else "")
        rec["term_role"] = rec.get("term_role") or ""
        rec["retrieval_role"] = rec.get("retrieval_role") or get_retrieval_role_from_term_role(rec.get("term_role"))

        drop, reject_reason = should_drop_term(rec)
        if drop:
            rec["reject_reason"] = reject_reason
            dropped_with_reason.append(rec)
            if STAGE3_DETAIL_DEBUG or stage3_debug:
                print(f"[Stage3 硬过滤] drop tid={rec.get('tid')} term={rec.get('term')!r} reason={reject_reason}")
            continue

        gate = _check_stage3_admission(rec, jd_profile, active_domains)
        if gate.get("hard_drop"):
            rec["reject_reason"] = gate.get("reason") or "admission"
            dropped_with_reason.append(rec)
            if label_trace:
                print(f"[Stage3 准入拒绝] tid={rec.get('tid')} term={rec.get('term')!r} reason={rec['reject_reason']}")
            continue

        primary_like = _is_primary_like(rec)
        if primary_like and label_trace:
            print(f"[Stage3 topic_gate] bypass primary-like tid={rec.get('tid')} term={rec.get('term')!r}")
        if not primary_like and not term_scoring.passes_topic_consistency(rec, None):
            rec["reject_reason"] = "topic_consistency"
            dropped_with_reason.append(rec)
            if label_trace:
                print(f"[Stage3 topic_gate] drop support tid={rec.get('tid')} term={rec.get('term')!r} reason=topic_consistency")
            continue

        rec["quality_score"] = term_scoring.score_term_expansion_quality(recall, rec)
        raw_final, explain = score_term_record(rec)
        identity_factor = term_scoring.compute_identity_factor(rec)
        risk_penalty = _compute_stage3_risk_penalty(rec)
        base_final = raw_final * identity_factor * risk_penalty
        role_factor = 1.0
        term_role = (rec.get("term_role") or "").strip().lower()
        role_in_anchor = (rec.get("role_in_anchor") or "").strip().lower()
        source_types = rec.get("source_types") or set()
        source_type_single = (rec.get("source_type") or rec.get("source") or "").strip().lower()
        if term_role == "primary" and role_in_anchor == "mainline":
            role_factor = STAGE3_MAINLINE_BOOST
        elif term_role == "dense_expansion" or "dense" in source_types or source_type_single == "dense":
            role_factor = STAGE3_DENSE_PENALTY
        elif "conditioned_vec" in source_types or source_type_single == "conditioned_vec":
            role_factor = STAGE3_CONDITIONED_PENALTY
        elif role_in_anchor == "side":
            role_factor = STAGE3_SIDE_PENALTY
        rec["final_score"] = base_final * role_factor
        explain["identity_factor"] = identity_factor
        explain["risk_penalty"] = risk_penalty
        explain["role_factor"] = role_factor
        rec["stage3_explain"] = explain
        rec["reject_reason"] = ""
        rec["stage3_bucket"] = _assign_stage3_bucket(rec)
        survivors.append(rec)

    survivors.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    for i, rec in enumerate(survivors):
        rec["stage3_rank"] = i
    survivors = _apply_family_role_constraints(survivors)
    survivors.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    for i, rec in enumerate(survivors):
        rec["stage3_rank"] = i

    for rec in survivors:
        rec["risk_reasons"] = _collect_risky_reasons(rec)
        rec["bucket_reason_flags"] = _collect_stage3_bucket_reason_flags(rec)

    survivors = stage3_build_score_map(survivors)

    # 窄表审计：final adjust → cross-anchor → support/risky（与 STAGE3_DEBUG_FOCUS_TERMS 配合可只看定点词）
    if STAGE3_AUDIT_DEBUG:
        _print_stage3_final_adjust_audit(survivors, top_k=12)
        _print_stage3_cross_anchor_audit(survivors, top_k=40)
        _print_stage3_support_risky_concise(survivors, top_k=20)

    core_terms_list, support_terms_list, risky_terms_list = _bucket_stage3_terms(survivors)
    if LABEL_PATH_TRACE or getattr(term_scoring, "STAGE3_DEBUG", False):
        for rec in (core_terms_list + support_terms_list + risky_terms_list)[:15]:
            term = (rec.get("term") or "")[:28]
            bucket = rec.get("stage3_bucket") or ""
            identity_factor = float((rec.get("stage3_explain") or {}).get("identity_factor", 1.0) or 1.0)
            anchor_count = int(rec.get("anchor_count") or 0)
            mainline_hits = int(rec.get("mainline_hits") or 0)
            can_expand = rec.get("can_expand", False)
            reason_flags = rec.get("bucket_reason_flags") or []
            print(
                f"[Stage3 bucket reason] term={term!r} | bucket={bucket!r} | identity_factor={identity_factor:.3f} | "
                f"anchor_count={anchor_count} mainline_hits={mainline_hits} can_expand={can_expand} | reason_flags={reason_flags}"
            )
        # 排序上升/下降主导因子：便于验证 mainline_support_factor 是否罚对
        for rec in survivors[:15]:
            term = (rec.get("term") or "")[:28]
            prev_rank = rec.get("stage2_rank")
            new_rank = rec.get("stage3_rank")
            explain = rec.get("stage3_explain") or {}
            up_factors = [k for k, v in explain.items() if isinstance(v, (int, float)) and (v > 1.0 or (k == "base_score" and v > 0.3))]
            down_factors = [k for k, v in explain.items() if isinstance(v, (int, float)) and 0 < v < 1.0]
            reason_flags = rec.get("bucket_reason_flags") or []
            missing_penalty = []
            if "no_mainline_support" in reason_flags or "only_weak_keep_sources" in reason_flags or "conditioned_only" in reason_flags:
                missing_penalty.append("mainline_support_factor")
            print(
                f"[Stage3 dominant factors] term={term!r}\n"
                f"  prev_rank={prev_rank} new_rank={new_rank}\n"
                f"  up_factors={up_factors}\n"
                f"  down_factors={down_factors}\n"
                f"  missing_penalty={missing_penalty}"
            )
    top_survivors = survivors[:STAGE3_TOP_K]
    paper_terms = select_terms_for_paper_recall(survivors, PAPER_RECALL_MAX_TERMS)

    for rec in top_survivors:
        _write_term_maps(
            score_map, term_map, idf_map, term_role_map, term_source_map,
            parent_anchor_map, parent_primary_map, recall, rec, query_vector,
        )
    for rec in paper_terms:
        _write_term_maps_if_missing(
            score_map, term_map, idf_map, term_role_map, term_source_map,
            parent_anchor_map, parent_primary_map, recall, rec, query_vector,
        )

    rerank_delta_rows = []
    for rec in survivors[:25]:
        rerank_delta_rows.append({
            "term": rec.get("term") or "",
            "stage2_rank": rec.get("stage2_rank", 0),
            "stage3_rank": rec.get("stage3_rank", 0),
            "delta": (rec.get("stage2_rank") or 0) - (rec.get("stage3_rank") or 0),
        })
    _debug_print_stage3_output(
        raw_candidates=merged_candidates,
        survivors=survivors,
        core_terms=core_terms_list,
        support_terms=support_terms_list,
        risky_terms=risky_terms_list,
        paper_terms=paper_terms,
        dropped_with_reason=dropped_with_reason,
        top_survivors=top_survivors,
        recall=recall,
        rerank_delta_rows=rerank_delta_rows,
        score_map=score_map,
        term_map=term_map,
        term_source_map=term_source_map,
        parent_anchor_map=parent_anchor_map,
        parent_primary_map=parent_primary_map,
    )
    recall.debug_info.dropped_with_reason = dropped_with_reason
    return score_map, term_map, idf_map, term_role_map, term_source_map, parent_anchor_map, parent_primary_map, paper_terms


def _debug_print_stage3_output(
    raw_candidates: List[Dict[str, Any]],
    survivors: List[Dict[str, Any]],
    core_terms: List[Dict[str, Any]],
    support_terms: List[Dict[str, Any]],
    risky_terms: List[Dict[str, Any]],
    paper_terms: List[Dict[str, Any]],
    dropped_with_reason: List[Dict[str, Any]],
    top_survivors: List[Dict[str, Any]],
    recall: Any,
    rerank_delta_rows: List[Dict[str, Any]],
    score_map: Dict[str, float],
    term_map: Dict[str, str],
    term_source_map: Dict[str, str],
    parent_anchor_map: Dict[str, str],
    parent_primary_map: Dict[str, str],
) -> None:
    """Stage3 调试输出：新列 anchor_count, evidence_count, family_centrality, path_topic_consistency, generic_penalty, cross_anchor_factor, retrieval_role；无 cluster 列。"""
    stage3_debug = getattr(term_scoring, "STAGE3_DEBUG", False)
    label_trace = LABEL_PATH_TRACE or stage3_debug
    _debug_print_stage3_tables(raw_candidates, dropped_with_reason, survivors, recall)
    debug_print(1, f"[Stage3] 输入候选总数={len(raw_candidates)} 幸存={len(survivors)} top_k={len(top_survivors)}", recall)
    debug_print(2, "[Stage3 Final Score Breakdown] term | stage2_rank | anchor_count | evidence_count | family_centrality | path_topic_consistency | generic_penalty | cross_anchor_factor | backbone_boost | object_like_penalty | bonus_term_penalty | retrieval_role | final", recall)
    for rec in survivors[:15]:
        t = (rec.get("term") or "")[:28]
        ex = rec.get("stage3_explain") or {}
        debug_print(2, (
            f"  {t:<28} | s2={rec.get('stage2_rank', 0):>2} | anc={rec.get('anchor_count', 0)} ev={rec.get('evidence_count', 0)} | "
            f"fc={float(rec.get('family_centrality') or 0):.3f} | ptc={float(ex.get('path_topic_consistency') or 0):.3f} | "
            f"gen={float(ex.get('generic_penalty') or 0):.3f} | cross={float(ex.get('cross_anchor_factor') or 0):.3f} | "
            f"bb={float(ex.get('backbone_boost') or 1):.3f} | obj={float(ex.get('object_like_penalty') or 1):.3f} | bonus={float(ex.get('bonus_term_penalty') or 1):.3f} | "
            f"{rec.get('retrieval_role', ''):14s} | final={float(rec.get('final_score') or 0):.3f}"
        ), recall)
    debug_print(2, "[Stage3 Rerank Delta] term | stage2_rank | stage3_rank | delta", recall)
    for row in rerank_delta_rows[:15]:
        t = (row.get("term") or "")[:28]
        debug_print(2, f"  {t:<28} | {row.get('stage2_rank', 0):>3} | {row.get('stage3_rank', 0):>3} | {row.get('delta', 0):+d}", recall)
    debug_print(2, "[Stage3 Buckets]", recall)
    debug_print(2, f"  core_terms={[r.get('term') for r in core_terms[:15]]}", recall)
    debug_print(2, f"  support_terms={[r.get('term') for r in support_terms[:15]]}", recall)
    debug_print(2, f"  risky_terms={[r.get('term') for r in risky_terms[:15]]}", recall)
    _debug_print_stage3_bucket_details(core_terms, support_terms, risky_terms, recall)
    debug_print(3, "[Stage3 Risky Term Reasons] term | reasons | final", recall)
    for r in risky_terms[:10]:
        t = (r.get("term") or "")[:28]
        debug_print(3, f"  {t:<28} | {r.get('risk_reasons', [])} | {r.get('final_score', 0):.3f}", recall)
    if STAGE3_DETAIL_DEBUG or stage3_debug:
        print(f"[paper_term_selection] family 保送式 选词数={len(paper_terms)} 明细: family_key | term | term_role | retrieval_role | final_score")
        for r in paper_terms[:30]:
            print(f"  {r.get('family_key','')} | {r.get('term','')!r} | {r.get('term_role','')} | {r.get('retrieval_role','')} | {r.get('final_score',0):.4f}")
        if len(paper_terms) > 30:
            print(f"  ... 共 {len(paper_terms)} 条")
        print(f"[Stage3] 去重后={len(raw_candidates)} 幸存={len(survivors)} top_k={len(top_survivors)} paper_terms={len(paper_terms)}")
        print("[Stage3 final_score 明细] term | base_score | path_topic_consistency | generic_penalty | cross_anchor_factor | backbone_boost | object_like_penalty | bonus_term_penalty | final_score | reject_reason")
        for i, rec in enumerate(top_survivors[:20]):
            term = (rec.get("term") or "")[:28]
            ex = rec.get("stage3_explain") or {}
            base_score = ex.get("base_score") or 0.0
            path_topic = ex.get("path_topic_consistency")
            gen_pen = ex.get("generic_penalty")
            cross_anchor = ex.get("cross_anchor_factor")
            backbone_boost = ex.get("backbone_boost")
            object_like_penalty = ex.get("object_like_penalty")
            bonus_term_penalty = ex.get("bonus_term_penalty")
            final_score = rec.get("final_score", 0) or 0.0
            reject = rec.get("reject_reason", "")
            _fmt = lambda x: f"{x:.3f}" if x is not None and isinstance(x, (int, float)) else str(x) if x is not None else "-"
            print(f"  {i+1} {term!r} | base={_fmt(base_score)} | path_topic={_fmt(path_topic)} | gen={_fmt(gen_pen)} | cross={_fmt(cross_anchor)} | bb={_fmt(backbone_boost)} | obj={_fmt(object_like_penalty)} | bonus={_fmt(bonus_term_penalty)} | final={final_score:.4f} | {reject!r}")
        if len(top_survivors) > 20:
            print(f"  ... 共 {len(top_survivors)} 条")
    if stage3_debug and score_map:
        print("[stage3_output] tid | term | source_type | parent_anchor | parent_primary | score")
        for i, (tid_str, sc) in enumerate(sorted(score_map.items(), key=lambda x: -x[1])[:25]):
            print(f"  {i+1} {tid_str} | {term_map.get(tid_str, '')!r} | {term_source_map.get(tid_str, '')} | {parent_anchor_map.get(tid_str, '')!r} | {parent_primary_map.get(tid_str, '')!r} | {sc:.3f}")
    if label_trace and dropped_with_reason:
        print("[标签路-被过滤原因] tid | term | source_type | parent_anchor | 原因")
        for r in dropped_with_reason[:50]:
            print(f"  {r.get('tid')} | {r.get('term','')!r} | {r.get('source') or r.get('origin','')} | {r.get('parent_anchor','')!r} | {r.get('reject_reason','')}")
        if len(dropped_with_reason) > 50:
            print(f"  ... 共 {len(dropped_with_reason)} 条被过滤")
    if label_trace and paper_terms:
        print("[标签路-final primary 为什么胜出] tid | term | source_type | parent_anchor | retrieval_role | 胜出原因")
        for r in paper_terms[:30]:
            print(f"  {r.get('tid')} | {r.get('term','')!r} | {r.get('source') or r.get('origin','')} | {r.get('parent_anchor','')!r} | {r.get('retrieval_role','')} | {r.get('win_reason','')}")
        if len(paper_terms) > 30:
            print(f"  ... 共 {len(paper_terms)} 条")


def run_stage3(
    recall,
    raw_candidates: List[Dict[str, Any]],
    query_vector,
    anchor_vids=None,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], List[Dict[str, Any]]]:
    """
    阶段 3：词权重。
    若候选含 term_role/identity_score（Stage2 新格式），走双闸门路径并返回 paper_terms（family 保送式选词）；否则走原有 _calculate_final_weights。
    返回 score_map, term_map, idf_map, term_role_map, term_source_map, parent_anchor_map, parent_primary_map, paper_terms。
    """
    if not raw_candidates:
        return {}, {}, {}, {}, {}, {}, {}, []

    use_dual_gate = (
        raw_candidates
        and ("term_role" in raw_candidates[0] or "identity_score" in raw_candidates[0])
    )
    paper_terms: List[Dict[str, Any]] = []
    if use_dual_gate:
        score_map, term_map, idf_map, term_role_map, term_source_map, parent_anchor_map, parent_primary_map, paper_terms = _run_stage3_dual_gate(
            recall, raw_candidates, query_vector, anchor_vids=anchor_vids
        )
    else:
        score_map, term_map, idf_map = recall._calculate_final_weights(
            raw_candidates, query_vector, anchor_vids=anchor_vids
        )
        term_role_map = {}
        term_source_map = {}
        parent_anchor_map = {}
        parent_primary_map = {}

    # verbose 调试：Stage3 已移除 cluster 列，改用 anchor_count / evidence_count / family_centrality / path_topic_consistency / generic_penalty / cross_anchor_factor / retrieval_role
    if recall.verbose and score_map and getattr(recall, "_last_tag_purity_debug", None):
        debug_by_tid = {str(d["tid"]): d for d in recall.debug_info.tag_purity_debug if d.get("tid") is not None}
        top_tids = sorted(score_map.keys(), key=lambda t: score_map.get(t, 0.0), reverse=True)

        def _fv(d, key, default=None):
            v = d.get(key, default)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return v

        def _f(v, w=6, ndec=4):
            if v is None:
                return "-".rjust(w)
            if isinstance(v, (int, float)):
                return f"{float(v):.{min(ndec, w-2)}f}".rjust(w)
            return str(v)[:w].rjust(w)

        head30 = top_tids[:30]
        lines = ["【Step 3 调试】Top20 学术词 | cos_sim(JD) | anchor_sim | final_weight"]
        for i, tid in enumerate(head30[:20], 1):
            term = term_map.get(tid, "")
            w = score_map.get(tid, 0.0)
            d = debug_by_tid.get(str(tid), {})
            cs = d.get("cos_sim")
            ans = d.get("anchor_sim")
            cs_s = f"{cs:.4f}" if cs is not None else "-"
            ans_s = f"{ans:.4f}" if ans is not None else "-"
            lines.append(f"  #{i:2d}  {term[:36]:36s}  cos={cs_s:6s}  anchor={ans_s:6s}  weight={w:.6f}")
        print("\n".join(lines))

        print("\n【观测面板 Stage3】Top30: term | anchor_count | evidence_count | family_centrality | path_topic_consistency | generic_penalty | cross_anchor_factor | retrieval_role | final_score")
        for i, tid in enumerate(head30, 1):
            d = debug_by_tid.get(str(tid), {})
            term = (d.get("term") or term_map.get(tid, ""))[:24]
            anc = d.get("anchor_count")
            ev = d.get("evidence_count")
            fc = _f(d.get("family_centrality"), 6)
            ptc = _f(d.get("path_topic_consistency"), 6)
            gen = _f(d.get("generic_penalty"), 6)
            cross = _f(d.get("cross_anchor_factor"), 6)
            role = (d.get("retrieval_role") or "")[:14]
            final_score = score_map.get(tid, 0.0)
            print(f"  #{i:2d}  {term:24s}  anc={anc} ev={ev}  fc={fc} ptc={ptc} gen={gen} cross={cross}  {role:14s}  {final_score:.6f}")

        print("\n【观测面板 汇总】Top30: term | base_score | path_topic_consistency | generic_penalty | cross_anchor_factor | final_score")
        for i, tid in enumerate(head30, 1):
            d = debug_by_tid.get(str(tid), {})
            term = (d.get("term") or term_map.get(tid, ""))[:24]
            print(
                f"  #{i:2d}  {term:24s}  base={_f(d.get('base_score'),7):>7}  ptc={_f(d.get('path_topic_consistency'),6):>6}  "
                f"gen={_f(d.get('generic_penalty'),6):>6}  cross={_f(d.get('cross_anchor_factor'),6):>6}  final={score_map.get(tid, 0.0):.6f}"
            )

    return score_map, term_map, idf_map, term_role_map, term_source_map, parent_anchor_map, parent_primary_map, paper_terms

