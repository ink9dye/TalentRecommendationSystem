# -*- coding: utf-8 -*-
"""
层级化领域守卫与自动负向领域屏蔽：分布/纯度/熵/层级契合/泛词惩罚等工具。

仅修改标签路，不修改索引构建；利用 vocabulary_topic_stats、vocabulary_domain_stats、
vocabulary_cooc_domain_ratio、vocabulary_cluster/cluster_members。
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional, Set, Tuple

# ---------- 分布与统计 ----------


def parse_json_dist(text: str | None) -> Dict[str, float]:
    """解析 JSON 分布，返回 {id_str: prob}。"""
    if not text:
        return {}
    try:
        d = json.loads(text) if isinstance(text, str) else text
        return {str(k): float(v) for k, v in (d or {}).items()}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def compute_purity(dist: Dict[str, float]) -> float:
    """purity = max_i p_i。"""
    if not dist:
        return 0.0
    return max(float(v) for v in dist.values())


def compute_entropy(dist: Dict[str, float], eps: float = 1e-12) -> float:
    """H(p) = -sum_i p_i * log(p_i + eps)。"""
    if not dist:
        return 0.0
    total = sum(dist.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for v in dist.values():
        p = float(v) / total + eps
        h -= p * math.log(p)
    return h


def compute_dist_overlap(term_dist: Dict[str, float], jd_dist: Dict[str, float]) -> float:
    """overlap = sum_k min(p_term(k), p_jd(k))。"""
    if not term_dist or not jd_dist:
        return 0.0
    return sum(min(term_dist.get(k, 0.0), jd_dist.get(k, 0.0)) for k in set(term_dist) | set(jd_dist))


def compute_outside_mass(term_dist: Dict[str, float], jd_keys: Set[str]) -> float:
    """1 - sum_{k in jd_keys} p_term(k)。"""
    if not term_dist or not jd_keys:
        return 1.0
    inside = sum(term_dist.get(k, 0.0) for k in jd_keys)
    return max(0.0, 1.0 - inside)


# ---------- 层级 fit ----------


def compute_hierarchical_fit(
    term_info: Dict[str, Any],
    jd_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    计算候选词与 JD 的四层分布契合度。
    term_info 需含 field_dist, subfield_dist, topic_dist, domain_dist（均为 dict 或已解析）。
    jd_profile 需含 field_weights, subfield_weights, topic_weights, domain_weights,
    active_subfields, active_topics（set 或 list），及 main_field_id, main_subfield_id, main_topic_id 等可选。
    返回 domain_fit, field_fit, subfield_fit, topic_fit, main_*_match, outside_subfield_mass, outside_topic_mass。
    """
    def _dist(d: Any) -> Dict[str, float]:
        if isinstance(d, dict):
            return d
        return parse_json_dist(d) if d else {}

    field_dist = _dist(term_info.get("field_dist"))
    subfield_dist = _dist(term_info.get("subfield_dist"))
    topic_dist = _dist(term_info.get("topic_dist"))
    domain_dist = _dist(term_info.get("domain_dist"))

    jd_f = _dist(jd_profile.get("field_weights"))
    jd_s = _dist(jd_profile.get("subfield_weights"))
    jd_t = _dist(jd_profile.get("topic_weights"))
    jd_d = _dist(jd_profile.get("domain_weights"))

    active_s = set(jd_profile.get("active_subfields") or [])
    if not active_s and jd_s:
        active_s = set(jd_s.keys())
    active_t = set(jd_profile.get("active_topics") or [])
    if not active_t and jd_t:
        active_t = set(jd_t.keys())

    # 缺失 = 证据不足，不按 0 处理；缺 subfield 退化为 field，缺 field 退化为 domain，缺 topic 为 None
    domain_fit = compute_dist_overlap(domain_dist, jd_d) if jd_d else 1.0
    field_fit = compute_dist_overlap(field_dist, jd_f) if jd_f else domain_fit
    subfield_fit = compute_dist_overlap(subfield_dist, jd_s) if jd_s else field_fit
    topic_fit = compute_dist_overlap(topic_dist, jd_t) if (topic_dist and jd_t) else None

    main_field = jd_profile.get("main_field_id")
    main_subfield = jd_profile.get("main_subfield_id")
    main_topic = jd_profile.get("main_topic_id")

    main_field_match = 1.0 if (main_field and field_dist.get(str(main_field), 0) > 0) else 0.0
    main_subfield_match = 1.0 if (main_subfield and subfield_dist.get(str(main_subfield), 0) > 0) else 0.0
    main_topic_match = 1.0 if (main_topic and topic_dist.get(str(main_topic), 0) > 0) else 0.0

    outside_subfield_mass = compute_outside_mass(subfield_dist, active_s)
    outside_topic_mass = compute_outside_mass(topic_dist, active_t)

    return {
        "domain_fit": domain_fit,
        "field_fit": field_fit,
        "subfield_fit": subfield_fit,
        "topic_fit": topic_fit,
        "main_field_match": main_field_match,
        "main_subfield_match": main_subfield_match,
        "main_topic_match": main_topic_match,
        "outside_subfield_mass": outside_subfield_mass,
        "outside_topic_mass": outside_topic_mass,
    }


# ---------- 泛词抑制 ----------


def compute_generic_penalty(work_count: int, domain_span: int) -> float:
    """
    GenericPenalty = 1 / log(2 + work_count) * 1 / (1 + 0.25 * (domain_span - 1))
    压制论文量极大、跨域过宽的大词。
    """
    if work_count is None:
        work_count = 0
    if domain_span is None:
        domain_span = 0
    a = 1.0 / math.log(2.0 + max(0, work_count)) if work_count else 1.0
    b = 1.0 / (1.0 + 0.25 * max(0, domain_span - 1))
    return a * b


def compute_external_penalty(fit_info: Dict[str, Any]) -> float:
    """
    无硬编码黑名单的负向领域惩罚：若词主要质量集中在 JD 外 subfield/topic 则降权。
    缺失 topic_fit / outside_topic_mass 时不按 0 惩罚，跳过该项。
    """
    penalty = 1.0
    if fit_info.get("subfield_fit", 1.0) is not None and fit_info.get("subfield_fit", 1.0) < 0.15:
        penalty *= 0.4
    tf = fit_info.get("topic_fit")
    if tf is not None and tf < 0.05:
        penalty *= 0.6
    if fit_info.get("outside_subfield_mass", 0.0) is not None and fit_info.get("outside_subfield_mass", 0.0) > 0.80:
        penalty *= 0.35
    otm = fit_info.get("outside_topic_mass")
    if otm is not None and otm > 0.90:
        penalty *= 0.5
    return penalty


def compute_purity_bonus(subfield_dist: Dict[str, float], topic_dist: Dict[str, float]) -> float:
    """PurityBonus = (subfield_purity)^0.8 * (topic_purity)^1.0"""
    ps = compute_purity(subfield_dist) if subfield_dist else 1.0
    pt = compute_purity(topic_dist) if topic_dist else 1.0
    return (ps ** 0.8) * (pt ** 1.0)


def compute_entropy_penalty(
    subfield_dist: Dict[str, float],
    topic_dist: Dict[str, float],
    lambda1: float = 0.3,
    lambda2: float = 0.5,
) -> float:
    """EntropyPenalty = 1 / (1 + lambda1*subfield_entropy + lambda2*topic_entropy)"""
    es = compute_entropy(subfield_dist) if subfield_dist else 0.0
    et = compute_entropy(topic_dist) if topic_dist else 0.0
    return 1.0 / (1.0 + lambda1 * es + lambda2 * et)


# ---------- Landing 总评分 ----------


def hierarchy_gate(fit_info: Dict[str, Any]) -> float:
    """
    (0.1+0.9*D)^0.3 * (0.1+0.9*F)^0.7 * (0.1+0.9*S)^1.2 * (0.1+0.9*T)^1.7
    Topic > Subfield > Field > Domain
    """
    D = fit_info.get("domain_fit", 0.0) or 0.0
    F = fit_info.get("field_fit", 0.0) or 0.0
    S = fit_info.get("subfield_fit", 0.0) or 0.0
    T = fit_info.get("topic_fit", 0.0) or 0.0
    return (
        (0.1 + 0.9 * D) ** 0.3
        * (0.1 + 0.9 * F) ** 0.7
        * (0.1 + 0.9 * S) ** 1.2
        * (0.1 + 0.9 * T) ** 1.7
    )


def score_landing_candidate(
    candidate: Dict[str, Any],
    anchor: Dict[str, Any],
    jd_profile: Optional[Dict[str, Any]],
    semantic_score: float,
    context_score: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    LandingScore = BaseSemantic * ContextScore * HierarchyGate * PurityBonus * EntropyPenalty * ExternalPenalty * GenericPenalty
    若 jd_profile 为空则退化为 semantic_score * context_score * generic_penalty。
    """
    explain: Dict[str, Any] = {}
    base = max(0.0, float(semantic_score)) * max(0.0, min(1.0, float(context_score)))
    explain["base_semantic"] = semantic_score
    explain["context_score"] = context_score

    if not jd_profile:
        work_count = candidate.get("work_count") or 0
        domain_span = candidate.get("domain_span") or 0
        gp = compute_generic_penalty(work_count, domain_span)
        return base * gp, explain

    fit_info = candidate.get("fit_info")
    if not fit_info:
        return base, explain
    explain["fit_info"] = fit_info

    gate = hierarchy_gate(fit_info)
    purity_bonus = compute_purity_bonus(
        fit_info.get("subfield_dist") or {},
        fit_info.get("topic_dist") or {},
    )
    subfield_d = (candidate.get("subfield_dist") or candidate.get("fit_info") or {}).get("subfield_dist") or {}
    topic_d = (candidate.get("topic_dist") or candidate.get("fit_info") or {}).get("topic_dist") or {}
    if not subfield_d and isinstance(fit_info.get("subfield_dist"), dict):
        subfield_d = fit_info.get("subfield_dist") or {}
    if not topic_d and isinstance(fit_info.get("topic_dist"), dict):
        topic_d = fit_info.get("topic_dist") or {}
    entropy_pen = compute_entropy_penalty(subfield_d, topic_d)
    external_pen = compute_external_penalty(fit_info)
    work_count = candidate.get("work_count") or 0
    domain_span = candidate.get("domain_span") or 0
    generic_pen = compute_generic_penalty(work_count, domain_span)

    score = base * gate * purity_bonus * entropy_pen * external_pen * generic_pen
    explain["hierarchy_gate"] = gate
    explain["purity_bonus"] = purity_bonus
    explain["entropy_penalty"] = entropy_pen
    explain["external_penalty"] = external_pen
    explain["generic_penalty"] = generic_pen
    explain["landing_score"] = score
    return score, explain


# ---------- 扩展资格与扩展打分 ----------


# 扩散条件：只看结构可靠性，不看具体领域词
EXPAND_MULTI_SOURCE_MIN = 0.8
EXPAND_SUBFIELD_FIT_MIN = 0.18
EXPAND_TOPIC_FIT_MIN = 0.05
EXPAND_OUTSIDE_SUBFIELD_MASS_MAX = 0.80


def allow_primary_to_expand(primary_record: Dict[str, Any]) -> bool:
    """
    只看结构可靠性，不看具体领域词。多来源支持、层级 fit 达标才允许扩散。
    """
    multi_src = primary_record.get("multi_source_support")
    if multi_src is None:
        multi_src = compute_multi_source_support(primary_record)
    if float(multi_src or 0.0) < EXPAND_MULTI_SOURCE_MIN:
        return False
    if float(primary_record.get("subfield_fit") or 0.0) < EXPAND_SUBFIELD_FIT_MIN:
        return False
    if float(primary_record.get("topic_fit") or 0.0) < EXPAND_TOPIC_FIT_MIN:
        return False
    if float(primary_record.get("outside_subfield_mass") or 0.0) > EXPAND_OUTSIDE_SUBFIELD_MASS_MAX:
        return False
    return True


def field_gravity(subfield_fit: float, topic_fit: float) -> float:
    """(0.1 + 0.9*subfield_fit)^1.2 * (0.1 + 0.9*topic_fit)^1.6"""
    return (0.1 + 0.9 * max(0, subfield_fit)) ** 1.2 * (0.1 + 0.9 * max(0, topic_fit)) ** 1.6


def score_expansion_candidate(
    neighbor_candidate: Dict[str, Any],
    seed_candidate: Dict[str, Any],
    jd_profile: Optional[Dict[str, Any]],
    raw_neighbor_score: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    NeighborScore = RawNeighborScore * SeedReliability * FieldGravity * PurityBonus * EntropyPenalty * ExternalPenalty * GenericPenalty
    """
    explain: Dict[str, Any] = {}
    seed_rel = 1.0
    if allow_primary_to_expand(seed_candidate):
        seed_rel = 1.0
    else:
        seed_rel = 0.5
    explain["seed_reliability"] = seed_rel

    fit_info = neighbor_candidate.get("fit_info") or {}
    subfield_fit = fit_info.get("subfield_fit", 0.0) or 0.0
    topic_fit = fit_info.get("topic_fit", 0.0) or 0.0
    fg = field_gravity(subfield_fit, topic_fit)
    explain["field_gravity"] = fg

    subfield_d = fit_info.get("subfield_dist") or {}
    topic_d = fit_info.get("topic_dist") or {}
    if not subfield_d:
        subfield_d = neighbor_candidate.get("subfield_dist") or {}
    if not topic_d:
        topic_d = neighbor_candidate.get("topic_dist") or {}
    purity_bonus = compute_purity_bonus(subfield_d, topic_d)
    entropy_pen = compute_entropy_penalty(subfield_d, topic_d)
    external_pen = compute_external_penalty(fit_info)
    work_count = neighbor_candidate.get("work_count") or 0
    domain_span = neighbor_candidate.get("domain_span") or 0
    generic_pen = compute_generic_penalty(work_count, domain_span)

    score = raw_neighbor_score * seed_rel * fg * purity_bonus * entropy_pen * external_pen * generic_pen
    explain["purity_bonus"] = purity_bonus
    explain["entropy_penalty"] = entropy_pen
    explain["external_penalty"] = external_pen
    explain["generic_penalty"] = generic_pen
    explain["final_expand_score"] = score
    return score, explain


# ---------- 结构信号：来源稳定性、family、retrieval 角色 ----------


def compute_multi_source_support(rec: Dict[str, Any]) -> float:
    """
    只看结构来源，不看词面。来源越多、越稳定，support 越高。
    source_flags 为 dict 时按权重累加；否则用 source 字符串近似。
    """
    flags = rec.get("source_flags")
    if isinstance(flags, dict):
        score = 0.0
        if flags.get("exact"):
            score += 1.0
        if flags.get("phrase"):
            score += 1.0
        if flags.get("dense"):
            score += 0.6
        if flags.get("similar_to"):
            score += 0.5
        if flags.get("cooc"):
            score += 0.4
        if flags.get("cluster"):
            score += 0.2
        return score
    src = (rec.get("source") or rec.get("origin") or "").strip().lower()
    if not src:
        return 0.0
    if src in ("edge_only", "edge_and_ctx", "jd_vector", "similar_to"):
        return 0.5
    if src in ("dense_expansion", "dense"):
        return 0.6
    if src in ("cooc_expansion", "cooc"):
        return 0.4
    if src in ("cluster_expansion", "cluster"):
        return 0.2
    return 0.3


def build_family_key(rec: Dict[str, Any]) -> str:
    """
    完全按结构关系生成 family，不看具体词面。
    优先级：parent_primary > seed_group_id > cluster_id > parent_anchor > tid
    """
    pp = rec.get("parent_primary")
    if pp is not None and pp != "":
        return f"primary::{pp}"
    sg = rec.get("seed_group_id")
    if sg is not None:
        return f"seed::{sg}"
    cid = rec.get("cluster_id")
    if cid is not None:
        return f"cluster::{cid}"
    pa = rec.get("parent_anchor")
    if pa is not None and pa != "":
        return f"anchor::{pa}"
    tid = rec.get("tid")
    return f"self::{tid}"


def get_retrieval_role_from_term_role(term_role: str) -> str:
    """
    从 term_role 得到 retrieval_role：谁进 paper 召回、谁仅辅助。
    paper_primary=参与检索且权重最高；paper_support=参与检索权重次之；blocked=不参与检索。
    """
    role = (term_role or "").strip().lower()
    if role == "primary":
        return "paper_primary"
    if role in ("dense_expansion", "cooc_expansion"):
        return "paper_support"
    if role in ("cluster_expansion", "cluster"):
        return "blocked"
    return "paper_support"


def compute_family_centrality(rec: Dict[str, Any], family_members: list) -> float:
    """
    候选是否为 family 结构中心：多来源支持 + 能带出 support（作为其它 term 的 parent_primary）。
    """
    source_support = rec.get("multi_source_support") or rec.get("source_support") or 0.0
    if isinstance(source_support, (int, float)):
        source_support = float(source_support)
    else:
        source_support = compute_multi_source_support(rec)
    term_val = rec.get("term") or ""
    child_count = sum(1 for m in family_members if (m.get("parent_primary") or "") == term_val)
    return 0.6 * min(source_support / 2.0, 1.0) + 0.4 * min(child_count / 3.0, 1.0)


# ---------- Stage3 硬过滤与 term 打分 ----------


def _get_topic_like_fit(rec: Dict[str, Any]) -> float:
    """topic_fit 退化：topic_fit → subfield_fit → field_fit → domain_fit。"""
    topic_fit = rec.get("topic_fit")
    if topic_fit is None:
        topic_fit = rec.get("subfield_fit")
    if topic_fit is None:
        topic_fit = rec.get("field_fit")
    if topic_fit is None:
        topic_fit = rec.get("domain_fit")
    return float(topic_fit or 0.0)


def should_drop_term(record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    轻硬过滤：primary-like 主落点只能软惩罚，不能硬删。
    仅两条规则，且只对「非 primary-like」生效：
    1）明显错域离群：outside_subfield > 0.97 且 topic_fit < 0.02
    2）极弱证据噪声：anchor_count<=1 且 evidence_count<=1 且 path_topic<0.15
    """
    outside_subfield = float(record.get("outside_subfield_mass") or 0.0)
    topic_fit = _get_topic_like_fit(record)
    source_type = (record.get("source_type") or record.get("source") or record.get("origin") or "").strip().lower()
    term_role = (record.get("term_role") or "").strip().lower()
    is_primary_like = (
        bool(record.get("has_primary_role"))
        or term_role == "primary"
        or source_type == "similar_to"
        or int(record.get("anchor_count") or 0) >= 2
    )

    if outside_subfield > 0.97 and topic_fit < 0.02:
        if not is_primary_like:
            return True, "outside_subfield_and_no_topic_fit"

    anchor_count = int(record.get("anchor_count") or 0)
    evidence_count = int(record.get("evidence_count") or 0)
    path_topic = float(record.get("path_topic_consistency") or 0.0)
    if path_topic <= 0.0:
        subfield_fit = record.get("subfield_fit")
        if subfield_fit is None:
            subfield_fit = record.get("field_fit")
        if subfield_fit is None:
            subfield_fit = record.get("domain_fit")
        path_topic = 0.55 * topic_fit + 0.35 * float(subfield_fit or 0.0)
    if anchor_count <= 1 and evidence_count <= 1 and path_topic < 0.15 and not is_primary_like:
        return True, "weak_evidence_noise"
    return False, ""


def _hierarchy_score_for_term(record: Dict[str, Any], fit_info: Dict[str, Any]) -> float:
    """
    从 fit_info 算单一 hierarchy_score；缺失 topic 时不惩罚，用 domain/field/subfield 加权。
    """
    domain_fit = fit_info.get("domain_fit", 1.0) or 1.0
    field_fit = fit_info.get("field_fit", 1.0) or 1.0
    subfield_fit = fit_info.get("subfield_fit", 1.0) or 1.0
    topic_fit = fit_info.get("topic_fit")
    if topic_fit is not None:
        return 0.25 * domain_fit + 0.25 * field_fit + 0.25 * subfield_fit + 0.25 * topic_fit
    return 0.4 * domain_fit + 0.3 * field_fit + 0.3 * subfield_fit


def _get_subfield_like_fit(rec: Dict[str, Any]) -> float:
    """subfield_fit 退化：subfield_fit → field_fit → domain_fit。"""
    subfield_fit = rec.get("subfield_fit")
    if subfield_fit is None:
        subfield_fit = rec.get("field_fit")
    if subfield_fit is None:
        subfield_fit = rec.get("domain_fit")
    return float(subfield_fit or 0.5)


def _get_topic_like_fit_eff(rec: Dict[str, Any]) -> float:
    """topic_fit 退化：topic_fit → subfield_fit → field_fit → domain_fit。"""
    topic_fit = rec.get("topic_fit")
    if topic_fit is None:
        topic_fit = rec.get("subfield_fit")
    if topic_fit is None:
        topic_fit = rec.get("field_fit")
    if topic_fit is None:
        topic_fit = rec.get("domain_fit")
    return float(topic_fit or 0.5)


def score_term_record(record: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Stage3 唯一主分函数。path_topic_consistency 吸收 primary-like 身份；generic_penalty 只轻惩罚不杀穿主干。
    final = base_score * gate * cross_anchor_factor，gate = 0.75 + 0.15*path_topic_consistency + 0.10*generic_penalty。
    """
    semantic = float(record.get("identity_score") or record.get("sim_score") or 0.0)
    context = float(record.get("quality_score") or 0.0)
    subfield_fit_eff = _get_subfield_like_fit(record)
    topic_fit_eff = _get_topic_like_fit_eff(record)
    source_count = len(record.get("source_types") or [])
    multi_source_support = min(1.0, 0.5 + 0.25 * source_count)
    base_score = (
        0.35 * semantic
        + 0.20 * context
        + 0.20 * subfield_fit_eff
        + 0.15 * topic_fit_eff
        + 0.10 * multi_source_support
    )
    source_type = (record.get("source_type") or record.get("source") or record.get("origin") or "").strip().lower()
    is_primary_like = (
        bool(record.get("has_primary_role"))
        or (record.get("term_role") or "").strip().lower() == "primary"
        or source_type == "similar_to"
        or int(record.get("anchor_count") or 0) >= 2
    )
    primary_bonus = 1.0 if record.get("has_primary_role") else 0.0
    similar_bonus = 1.0 if source_type == "similar_to" else 0.0
    anchor_bonus = min(1.0, int(record.get("anchor_count") or 0) / 2.0)
    path_topic_consistency = (
        0.45 * topic_fit_eff
        + 0.30 * subfield_fit_eff
        + 0.15 * primary_bonus
        + 0.05 * similar_bonus
        + 0.05 * anchor_bonus
    )
    if is_primary_like:
        path_topic_consistency = max(path_topic_consistency, 0.45)
    path_topic_consistency = min(1.0, max(0.0, path_topic_consistency))
    record["path_topic_consistency"] = path_topic_consistency
    outside = float(record.get("outside_subfield_mass") or 0.0)
    generic_penalty = 1.00 - 0.12 * min(1.0, outside)
    if is_primary_like:
        generic_penalty = max(generic_penalty, 0.90)
    if int(record.get("anchor_count") or 0) >= 2:
        generic_penalty = max(generic_penalty, 0.92)
    generic_penalty = min(1.00, max(0.82, generic_penalty))
    cross_anchor_factor = float(record.get("cross_anchor_evidence") or 1.0)
    cross_anchor_factor = min(1.10, max(0.90, cross_anchor_factor))
    gate = 0.75 + 0.15 * path_topic_consistency + 0.10 * generic_penalty
    final_score = base_score * gate * cross_anchor_factor
    explain = {
        "base_score": base_score,
        "path_topic_consistency": path_topic_consistency,
        "generic_penalty": generic_penalty,
        "cross_anchor_factor": cross_anchor_factor,
        "final_score": final_score,
        "semantic_drift_risk": float(record.get("semantic_drift_risk") or 0.0),
        "outside_subfield_mass": outside,
        "outside_topic_mass": float(record.get("outside_topic_mass") or 0.0),
        "family_centrality": float(record.get("family_centrality") or 0.0),
    }
    return final_score, explain


def apply_family_rank_decay(records: list, rank_key: str = "final_score", family_key: str = "cluster_id") -> list:
    """
    对同一 family/cluster 内按 rank_key 排序后，第 rank 位乘 0.85^(rank-1)。
    返回新列表，每条 record 为副本并含 family_rank_decay 及更新后的 rank_key。
    """
    from collections import defaultdict
    by_family: Dict[Any, list] = defaultdict(list)
    for r in records:
        fid = r.get(family_key) or r.get("seed_group_id") or "_none"
        by_family[fid].append(r)
    out = []
    for _fid, group in by_family.items():
        sorted_group = sorted(group, key=lambda x: float(x.get(rank_key) or 0.0), reverse=True)
        for rank, rec in enumerate(sorted_group):
            decay = 0.85 ** max(0, rank)
            rec_copy = dict(rec) if isinstance(rec, dict) else rec
            if isinstance(rec_copy, dict):
                rec_copy["family_rank_decay"] = decay
                prev = float(rec_copy.get(rank_key) or 0.0)
                rec_copy[rank_key] = prev * decay
            out.append(rec_copy)
    return out
