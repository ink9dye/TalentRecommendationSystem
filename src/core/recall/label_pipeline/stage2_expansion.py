# -*- coding: utf-8 -*-
"""
Stage2：学术词扩展。
先 Stage2A 主落点（保守），再 Stage2B 仅围绕 primary 扩展；无缩写扩写表。
当前 Stage2A 候选来源仅跨类型 SIMILAR_TO。
"""
from typing import Any, Dict, List, Optional, Set

from src.core.recall.label_means import label_expansion
from src.core.recall.label_means.label_expansion import (
    PreparedAnchor,
    ExpandedTermCandidate,
    _anchor_skills_to_prepared_anchors,
    stage2_generate_academic_terms,
)


def _expanded_to_raw_candidates(terms: List[ExpandedTermCandidate]) -> List[Dict[str, Any]]:
    """将 Stage2 输出的 ExpandedTermCandidate 转为下游/Stage3 使用的 raw_candidates 字典列表。"""
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
        }
        rec["topic_align"] = getattr(c, "topic_align", 1.0)
        rec["topic_level"] = getattr(c, "topic_level", "missing")
        rec["topic_confidence"] = getattr(c, "topic_confidence", 1.0)
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
) -> List[Dict[str, Any]]:
    """
    阶段 2：学术词扩展。
    先转 anchor_skills 为 PreparedAnchor，再走 Stage2A 主落点 + Stage2B 仅围绕 primary 扩展；
    返回 raw_candidates（含 term_role、identity_score、topic_align）供 Stage3 双闸门使用。
    当前 Stage2A 仅 SIMILAR_TO 主落点，无其他候选源并轨。
    可选 jd_field_ids/jd_subfield_ids/jd_topic_ids 供三层领域 topic_align；也可从 recall 上读取。
    """
    prepared_anchors = _anchor_skills_to_prepared_anchors(recall, anchor_skills)
    if not prepared_anchors:
        return []
    jd_f = jd_field_ids or getattr(recall, "jd_field_ids", None)
    jd_s = jd_subfield_ids or getattr(recall, "jd_subfield_ids", None)
    jd_t = jd_topic_ids or getattr(recall, "jd_topic_ids", None)
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
    )
    return _expanded_to_raw_candidates(terms)
