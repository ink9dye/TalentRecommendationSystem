# -*- coding: utf-8 -*-
"""
Stage2：学术词扩展。
先 Stage2A 主落点（保守），再 Stage2B 仅围绕 primary 扩展；无缩写扩写表。
当前 Stage2A 候选来源仅跨类型 SIMILAR_TO。
"""
from typing import Any, Dict, List, Optional, Set

from src.core.recall.label_means import label_expansion
from src.core.recall.label_means.hierarchy_guard import get_retrieval_role_from_term_role
from src.core.recall.label_means.label_expansion import (
    PreparedAnchor,
    ExpandedTermCandidate,
    LABEL_EXPANSION_DEBUG,
    STAGE2_NOISY_DEBUG,
    _anchor_skills_to_prepared_anchors,
    stage2_generate_academic_terms,
)


def _expanded_to_raw_candidates(terms: List[ExpandedTermCandidate]) -> List[Dict[str, Any]]:
    """
    Stage2 -> Stage3：正交层级字段 + 必要标识；2A/2B 分桶结论须写顶层 dict，
    Stage3 去重聚合才会把 winning rec 的键并入合并记录（仅 _debug 不会带上）。
    """
    out = []
    for c in terms:
        rec = {
            "tid": c.vid,
            "term": c.term,
            "term_role": c.term_role,
            "identity_score": c.identity_score,
            "source": c.source,
            "origin": c.source,
            "sim_score": c.semantic_score,
            "degree_w": c.degree_w,
            "domain_span": c.domain_span,
            "target_degree_w": c.target_degree_w,
            "degree_w_expanded": getattr(c, "degree_w_expanded", 0) or 0,
            "cov_j": c.cov_j,
            "hit_count": c.hit_count,
            "src_vids": getattr(c, "src_vids", []) or [],
            "domain_fit": getattr(c, "domain_fit", 1.0),
            "parent_anchor": c.anchor_term,
            "parent_primary": getattr(c, "parent_primary", c.term) or c.term,
            "field_fit": float(getattr(c, "field_fit", 0) or 0),
            "subfield_fit": float(getattr(c, "subfield_fit", 0) or 0),
            "topic_fit": float(getattr(c, "topic_fit", 0) or 0) if getattr(c, "topic_fit", None) is not None else None,
            "path_match": float(getattr(c, "path_match", 0) or 0),
            "genericity_penalty": float(getattr(c, "genericity_penalty", 1.0) or 1.0),
        }
        if getattr(c, "cluster_id", None) is not None:
            rec["cluster_id"] = c.cluster_id
        if getattr(c, "outside_subfield_mass", None) is not None:
            rec["outside_subfield_mass"] = c.outside_subfield_mass
        rec["_debug"] = {
            "topic_align": getattr(c, "topic_align", 1.0),
            "topic_level": getattr(c, "topic_level", "missing"),
            "topic_confidence": getattr(c, "topic_confidence", 1.0),
            "outside_topic_mass": getattr(c, "outside_topic_mass", None),
            "topic_entropy": getattr(c, "topic_entropy", None),
            "main_subfield_match": getattr(c, "main_subfield_match", None),
            "landing_score": getattr(c, "landing_score", None),
            "mainline_preference": getattr(c, "mainline_preference", None),
            "mainline_rank": getattr(c, "mainline_rank", None),
            "anchor_internal_rank": getattr(c, "anchor_internal_rank", None),
            "survive_primary": getattr(c, "survive_primary", None),
            "can_expand": getattr(c, "can_expand", None),
            "sort_key_snapshot": getattr(c, "sort_key_snapshot", None),
            "role_in_anchor": getattr(c, "role_in_anchor", None),
            "cross_anchor_support": getattr(c, "cross_anchor_support", None),
            "seed_block_reason": getattr(c, "seed_block_reason", None),
            "has_family_evidence": getattr(c, "has_family_evidence", False),
            # 与顶层同步，便于只开 _debug 时的离线对照
            "primary_bucket": getattr(c, "primary_bucket", None) or "",
            "can_expand_from_2a": bool(getattr(c, "can_expand_from_2a", False)),
            "fallback_primary": bool(getattr(c, "fallback_primary", False)),
            "admission_reason": getattr(c, "admission_reason", "") or "",
            "reject_reason": getattr(c, "reject_reason", "") or "",
            "stage2b_seed_tier": getattr(c, "stage2b_seed_tier", None) or "none",
            "mainline_candidate": bool(getattr(c, "mainline_candidate", False)),
            "primary_reason": getattr(c, "primary_reason", "") or "",
        }
        rec["retrieval_role"] = get_retrieval_role_from_term_role(c.term_role)
        # Stage3 分层与准入所需字段（顶层透传，便于 classify_stage3_entry_groups / check_stage3_admission）
        rec["can_expand"] = getattr(c, "can_expand", False)
        rec["role_in_anchor"] = getattr(c, "role_in_anchor", "") or ""
        rec["retain_mode"] = getattr(c, "retain_mode", "normal") or "normal"
        rec["source_type"] = getattr(c, "source", "") or rec.get("source", "") or ""
        rec["polysemy_risk"] = float(getattr(c, "polysemy_risk", 0) or 0)
        rec["object_like_risk"] = float(getattr(c, "object_like_risk", 0) or 0)
        rec["generic_risk"] = float(getattr(c, "generic_risk", 0) or 0)
        rec["context_continuity"] = float(getattr(c, "context_continuity", 0) or 0)
        rec["jd_candidate_alignment"] = float(getattr(c, "jd_candidate_alignment", 0.5) or getattr(c, "jd_align", 0.5) or 0.5)
        # 批注：以下键必须在顶层 dict 中（不能只放在 _debug）；Stage3 去重合并只从 winning 行的
        # 顶层键拷贝到聚合 rec，供分桶观测、统一连续分特征与 paper 硬挡（fallback 等）使用。
        rec["primary_bucket"] = getattr(c, "primary_bucket", "") or ""
        rec["can_expand_from_2a"] = bool(getattr(c, "can_expand_from_2a", False))
        rec["fallback_primary"] = bool(getattr(c, "fallback_primary", False))
        rec["admission_reason"] = getattr(c, "admission_reason", "") or ""
        rec["reject_reason"] = getattr(c, "reject_reason", "") or ""
        rec["survive_primary"] = bool(getattr(c, "survive_primary", False))
        rec["stage2b_seed_tier"] = getattr(c, "stage2b_seed_tier", "none") or "none"
        rec["mainline_candidate"] = bool(getattr(c, "mainline_candidate", False))
        rec["primary_reason"] = getattr(c, "primary_reason", "") or ""
        _pao = getattr(c, "parent_anchor_obj", None)
        _fs_obj = float(getattr(_pao, "final_anchor_score", 0.0) or 0.0) if _pao is not None else 0.0
        _fs_c = float(getattr(c, "parent_anchor_final_score", 0.0) or 0.0)
        rec["parent_anchor_final_score"] = max(_fs_obj, _fs_c)
        _rk_obj = int(getattr(_pao, "step2_anchor_rank", 0) or 0) if _pao is not None else 0
        _rk_c = int(getattr(c, "parent_anchor_step2_rank", 0) or 0)
        _rk = _rk_c if _rk_c > 0 else _rk_obj
        if _rk > 0:
            rec["parent_anchor_step2_rank"] = _rk
        out.append(rec)
    if LABEL_EXPANSION_DEBUG and out:
        n = len(out)
        n_2a = sum(1 for r in out if r.get("can_expand_from_2a"))
        n_fb = sum(1 for r in out if r.get("fallback_primary"))
        buckets: Dict[str, int] = {}
        for r in out:
            pb = str(r.get("primary_bucket") or "") or "(empty)"
            buckets[pb] = buckets.get(pb, 0) + 1
        top_pb = sorted(buckets.items(), key=lambda x: -x[1])[:4]
        pb_s = ",".join(f"{k}:{v}" for k, v in top_pb)
        print(
            f"[Stage2->3 field audit] n={n} can_expand_from_2a={n_2a} fallback_primary={n_fb} "
            f"primary_bucket_top=[{pb_s}] （逐条前10需 STAGE2_NOISY_DEBUG）"
        )
        if STAGE2_NOISY_DEBUG:
            for rec in out[:10]:
                print(
                    f"  term={rec['term']!r} primary_bucket={rec.get('primary_bucket')!r} "
                    f"can_expand_from_2a={rec.get('can_expand_from_2a')!r} "
                    f"fallback_primary={rec.get('fallback_primary')!r}"
                )
    return out


def run_stage2(
    recall,
    anchor_skills: Dict[str, Any],
    active_domain_set: Set[int],
    regex_str: str,
    query_vector,
    query_text: str = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
    jd_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    阶段 2：学术词扩展。
    先转 anchor_skills 为 PreparedAnchor，再走 Stage2A 主落点 + Stage2B 仅围绕 primary 扩展；
    传入 jd_profile 时启用四层领域契合与泛词惩罚。
    返回 raw_candidates（含 term_role、identity_score、topic_align、层级 fit 等）供 Stage3 使用。
    """
    prepared_anchors = _anchor_skills_to_prepared_anchors(recall, anchor_skills)
    if not prepared_anchors:
        return []
    jd_f = jd_field_ids or getattr(recall, "jd_field_ids", None)
    jd_s = jd_subfield_ids or getattr(recall, "jd_subfield_ids", None)
    jd_t = jd_topic_ids or getattr(recall, "jd_topic_ids", None)
    if jd_profile:
        jd_f = jd_f or (set(jd_profile.get("active_fields") or []) if jd_profile.get("field_weights") else None)
        jd_s = jd_s or (set(jd_profile.get("active_subfields") or []) if jd_profile.get("subfield_weights") else None)
        jd_t = jd_t or (set(jd_profile.get("active_topics") or []) if jd_profile.get("topic_weights") else None)
    terms = stage2_generate_academic_terms(
        recall,
        prepared_anchors,
        active_domain_set=active_domain_set,
        domain_regex=regex_str,
        query_vector=query_vector,
        query_text=query_text,
        jd_field_ids=jd_f,
        jd_subfield_ids=jd_s,
        jd_topic_ids=jd_t,
        jd_profile=jd_profile,
    )
    return _expanded_to_raw_candidates(terms)
