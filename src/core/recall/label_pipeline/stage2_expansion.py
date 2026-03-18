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
    _anchor_skills_to_prepared_anchors,
    stage2_generate_academic_terms,
)


def _expanded_to_raw_candidates(terms: List[ExpandedTermCandidate]) -> List[Dict[str, Any]]:
    """Stage2 -> Stage3：只传 5 个正交层级字段 + 必要标识；冗余字段仅入 _debug 供排查。"""
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
        }
        rec["retrieval_role"] = get_retrieval_role_from_term_role(c.term_role)
        out.append(rec)
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
