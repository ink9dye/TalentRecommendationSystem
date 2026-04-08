# -*- coding: utf-8 -*-
"""
Stage2：学术词扩展。
先 Stage2A 主落点（保守），再 Stage2B 仅围绕 primary 扩展；无缩写扩写表。
当前 Stage2A 候选来源仅跨类型 SIMILAR_TO。
"""
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.recall.label_means import label_expansion
from src.core.recall.label_means.hierarchy_guard import get_retrieval_role_from_term_role
from src.core.recall.label_means.label_expansion import (
    PreparedAnchor,
    ExpandedTermCandidate,
    _anchor_skills_to_prepared_anchors,
    stage2_generate_academic_terms,
)


def _expanded_to_raw_candidates(terms: List[ExpandedTermCandidate]) -> List[Dict[str, Any]]:
    """
    Stage2 -> Stage3：单条 candidate record 字段分层。
    顶层仅保留标识 / local evidence / risk / confidence；强 provisional 语义统一下沉到 stage2_local_meta。
    少量与 meta 重复的键暂留顶层作 Stage3 merge 兼容（TODO: Stage3 迁移后删 legacy mirror）。
    """
    out = []
    for c in terms:
        _pao = getattr(c, "parent_anchor_obj", None)
        _fs_obj = float(getattr(_pao, "final_anchor_score", 0.0) or 0.0) if _pao is not None else 0.0
        _fs_c = float(getattr(c, "parent_anchor_final_score", 0.0) or 0.0)
        parent_anchor_final_score = max(_fs_obj, _fs_c)
        _rk_obj = int(getattr(_pao, "step2_anchor_rank", 0) or 0) if _pao is not None else 0
        _rk_c = int(getattr(c, "parent_anchor_step2_rank", 0) or 0)
        _rk = _rk_c if _rk_c > 0 else _rk_obj

        can_ex_2a = bool(getattr(c, "can_expand_from_2a", False))
        can_ex = bool(getattr(c, "can_expand", False))
        role_ia = getattr(c, "role_in_anchor", "") or ""

        # 正式契约：Stage2 内部 provisional / 排名 / bucket 痕迹（非下游 final fact）
        stage2_local_meta: Dict[str, Any] = {
            "primary_bucket": getattr(c, "primary_bucket", "") or "",
            "fallback_primary": bool(getattr(c, "fallback_primary", False)),
            "admission_reason": getattr(c, "admission_reason", "") or "",
            "reject_reason": getattr(c, "reject_reason", "") or "",
            "survive_primary": bool(getattr(c, "survive_primary", False)),
            "stage2b_seed_tier": getattr(c, "stage2b_seed_tier", "none") or "none",
            "mainline_candidate": bool(getattr(c, "mainline_candidate", False)),
            "primary_reason": getattr(c, "primary_reason", "") or "",
            "parent_primary": getattr(c, "parent_primary", c.term) or c.term,
            "parent_anchor_final_score": float(parent_anchor_final_score),
            "parent_anchor_step2_rank": int(_rk) if _rk > 0 else None,
            "anchor_internal_rank": getattr(c, "anchor_internal_rank", None),
            "can_expand_local": can_ex_2a,
            "role_in_anchor": role_ia,
            "seed_block_reason": getattr(c, "seed_block_reason", None),
            "has_family_evidence": bool(getattr(c, "has_family_evidence", False)),
        }

        _jd = float(getattr(c, "jd_candidate_alignment", 0.5) or getattr(c, "jd_align", 0.5) or 0.5)
        _ident = float(c.identity_score)
        _sim = float(c.semantic_score)
        _poly = float(getattr(c, "polysemy_risk", 0) or 0)
        _obj = float(getattr(c, "object_like_risk", 0) or 0)
        _gen = float(getattr(c, "generic_risk", 0) or 0)

        risk_flags: List[str] = []
        if _gen > 0.25:
            risk_flags.append("generic")
        if _poly > 0.2:
            risk_flags.append("polysemy")
        if _obj > 0.2:
            risk_flags.append("object_like")

        # TODO(Stage2): landing/expansion 置信度在 2A/2B 分层稳定后按来源精化；现用局部分数近似
        _land_sc = getattr(c, "landing_score", None)
        landing_confidence = float(_land_sc) if _land_sc is not None else max(0.0, min(1.0, max(_ident, _sim)))
        expansion_confidence = max(0.0, min(1.0, float(getattr(c, "context_continuity", 0) or 0) * 0.5 + _sim * 0.5))

        rec: Dict[str, Any] = {
            "tid": c.vid,
            "term": c.term,
            "anchor_term": c.anchor_term,
            "candidate_source": c.source,
            "term_role_local": c.term_role,
            "identity_score": c.identity_score,
            "sim_score": c.semantic_score,
            "degree_w": c.degree_w,
            "domain_span": c.domain_span,
            "target_degree_w": c.target_degree_w,
            "degree_w_expanded": getattr(c, "degree_w_expanded", 0) or 0,
            "cov_j": c.cov_j,
            "hit_count": c.hit_count,
            "src_vids": getattr(c, "src_vids", []) or [],
            "domain_fit": getattr(c, "domain_fit", 1.0),
            "field_fit": float(getattr(c, "field_fit", 0) or 0),
            "subfield_fit": float(getattr(c, "subfield_fit", 0) or 0),
            "topic_fit": float(getattr(c, "topic_fit", 0) or 0) if getattr(c, "topic_fit", None) is not None else None,
            "path_match": float(getattr(c, "path_match", 0) or 0),
            "genericity_penalty": float(getattr(c, "genericity_penalty", 1.0) or 1.0),
            "polysemy_risk": _poly,
            "object_like_risk": _obj,
            "generic_risk": _gen,
            "risk_flags": risk_flags,
            "jd_candidate_alignment": _jd,
            "jd_align": _jd,
            "context_continuity": float(getattr(c, "context_continuity", 0) or 0),
            "landing_confidence": landing_confidence,
            "expansion_confidence": expansion_confidence,
            "retrieval_role": get_retrieval_role_from_term_role(c.term_role),
            "stage2_local_meta": stage2_local_meta,
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
            "primary_bucket": stage2_local_meta["primary_bucket"],
            "can_expand_from_2a": can_ex_2a,
            "fallback_primary": stage2_local_meta["fallback_primary"],
            "admission_reason": stage2_local_meta["admission_reason"],
            "reject_reason": stage2_local_meta["reject_reason"],
            "stage2b_seed_tier": stage2_local_meta["stage2b_seed_tier"],
            "mainline_candidate": stage2_local_meta["mainline_candidate"],
            "primary_reason": stage2_local_meta["primary_reason"],
            "surface_sim": getattr(c, "surface_sim", None),
            "conditioned_sim": getattr(c, "conditioned_sim", None),
            "context_gain": getattr(c, "context_gain", None),
            "source_set": sorted(getattr(c, "source_set", None) or []),
            "has_dynamic_support": getattr(c, "has_dynamic_support", None),
            "has_static_support": getattr(c, "has_static_support", None),
            "dual_support": getattr(c, "dual_support", None),
        }

        # --- TODO: remove legacy top-level mirror after Stage3 migration to stage2_local_meta ---
        rec["term_role"] = c.term_role
        rec["source"] = c.source
        rec["origin"] = c.source
        rec["parent_anchor"] = c.anchor_term
        rec["parent_primary"] = stage2_local_meta["parent_primary"]
        rec["source_type"] = getattr(c, "source", "") or ""
        rec["can_expand"] = can_ex
        rec["can_expand_from_2a"] = can_ex_2a
        rec["can_expand_local"] = can_ex_2a
        rec["role_in_anchor"] = role_ia
        rec["retain_mode"] = getattr(c, "retain_mode", "normal") or "normal"
        rec["primary_bucket"] = stage2_local_meta["primary_bucket"]
        rec["fallback_primary"] = stage2_local_meta["fallback_primary"]
        rec["admission_reason"] = stage2_local_meta["admission_reason"]
        rec["reject_reason"] = stage2_local_meta["reject_reason"]
        rec["survive_primary"] = stage2_local_meta["survive_primary"]
        rec["stage2b_seed_tier"] = stage2_local_meta["stage2b_seed_tier"]
        rec["mainline_candidate"] = stage2_local_meta["mainline_candidate"]
        rec["primary_reason"] = stage2_local_meta["primary_reason"]
        rec["parent_anchor_final_score"] = float(parent_anchor_final_score)
        if _rk > 0:
            rec["parent_anchor_step2_rank"] = int(_rk)

        out.append(rec)
    # 须读 label_expansion 模块上的当前值：from ... import FLAG 会在 import 时绑定副本，
    # recall() 内 label_expansion.LABEL_EXPANSION_DEBUG = ... 无法更新此处旧绑定。
    if getattr(label_expansion, "LABEL_EXPANSION_DEBUG", False) and out:
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
        if getattr(label_expansion, "STAGE2_NOISY_DEBUG", False):
            for rec in out[:10]:
                print(
                    f"  term={rec['term']!r} primary_bucket={rec.get('primary_bucket')!r} "
                    f"can_expand_from_2a={rec.get('can_expand_from_2a')!r} "
                    f"fallback_primary={rec.get('fallback_primary')!r}"
                )
    if out:
        sample = out[0]
        print("\n[Stage2] sample candidate top-level keys:", sorted(sample.keys()))
        print(
            "[Stage2] sample candidate stage2_local_meta keys:",
            sorted((sample.get("stage2_local_meta") or {}).keys()),
        )
    return out


def _empty_candidate_graph() -> Dict[str, List[Any]]:
    """四类边键齐全的空图；无候选或失败时 fallback。"""
    return {
        "same_anchor_edges": [],
        "cross_anchor_support_edges": [],
        "family_edges": [],
        "provenance_edges": [],
    }


def _rec_tid(rec: Dict[str, Any]) -> Optional[int]:
    t = rec.get("tid")
    if t is None:
        return None
    try:
        return int(t)
    except (TypeError, ValueError):
        return None


def _rec_score_hint(rec: Dict[str, Any]) -> float:
    """同锚 pairwise 用：优先 identity / sim，其次 landing_confidence。"""
    try:
        a = float(rec.get("identity_score") or 0)
        b = float(rec.get("sim_score") or 0)
        return max(a, b)
    except (TypeError, ValueError):
        return 0.0


def build_stage2_candidate_graph(
    anchor_to_candidates: Dict[str, Dict[str, Any]],
    all_candidates: List[Dict[str, Any]],
    jd_profile: Optional[Dict[str, Any]] = None,
    active_domains: Optional[Set[int]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    candidate_graph 最小可用版（MUV）：四类边语义固定，数值与门控为保守近似，供 Stage3 / 后续精化。
    same_anchor：同锚候选竞争；cross_anchor_support：跨锚相容（强门控+限额）；
    family：显式 family 痕迹弱边；provenance：expansion→landing 父项（仅当能解析到父 tid）。
    jd_profile / active_domains 预留；本轮门控主要用 record 内 evidence。
    """
    _ = jd_profile
    _ = active_domains
    same_anchor_edges: List[Dict[str, Any]] = []
    cross_edges: List[Dict[str, Any]] = []
    family_edges: List[Dict[str, Any]] = []
    provenance_edges: List[Dict[str, Any]] = []

    dedup_sa: Set[Tuple[Any, ...]] = set()
    dedup_x: Set[Tuple[Any, ...]] = set()
    dedup_f: Set[Tuple[Any, ...]] = set()
    dedup_p: Set[Tuple[Any, ...]] = set()

    # ---------- same_anchor_edges：每锚 candidates pairwise（规模上限防炸） ----------
    _PAIRWISE_CAP = 14
    for anchor_id, node in (anchor_to_candidates or {}).items():
        cands = list(node.get("candidates") or [])
        if len(cands) < 2:
            continue
        cands_scored = sorted(cands, key=_rec_score_hint, reverse=True)[:_PAIRWISE_CAP]
        for ra, rb in combinations(cands_scored, 2):
            ta, tb = _rec_tid(ra), _rec_tid(rb)
            if ta is None or tb is None or ta == tb:
                continue
            t1, t2 = (ta, tb) if ta < tb else (tb, ta)
            key = (t1, t2, "same_anchor", str(anchor_id))
            if key in dedup_sa:
                continue
            dedup_sa.add(key)
            sa, sb = _rec_score_hint(ra), _rec_score_hint(rb)
            same_anchor_edges.append(
                {
                    "src_tid": ta,
                    "dst_tid": tb,
                    "src_term": str(ra.get("term") or ""),
                    "dst_term": str(rb.get("term") or ""),
                    "edge_type": "same_anchor",
                    "relation": "compete",
                    "anchor_id": str(anchor_id),
                    "score_hint": abs(sa - sb),
                    "reason": "same_anchor_pairwise",
                }
            )

    # ---------- cross_anchor_support_edges：跨锚保守 + 每 src_tid 最多 4 条 ----------
    flat: List[Tuple[str, Dict[str, Any]]] = []
    for aid, node in (anchor_to_candidates or {}).items():
        for r in node.get("candidates") or []:
            flat.append((str(aid), r))
    cross_out: Dict[int, int] = defaultdict(int)

    def _jd(r: Dict[str, Any]) -> float:
        return float(r.get("jd_align") or r.get("jd_candidate_alignment") or 0)

    def _support_score(ri: Dict[str, Any], rj: Dict[str, Any]) -> float:
        ff = (float(ri.get("field_fit") or 0) + float(rj.get("field_fit") or 0)) / 2.0
        sf = (float(ri.get("subfield_fit") or 0) + float(rj.get("subfield_fit") or 0)) / 2.0
        ti = ri.get("topic_fit")
        tj = rj.get("topic_fit")
        tf = (
            (float(ti) + float(tj)) / 2.0
            if ti is not None and tj is not None
            else max(float(ti or 0), float(tj or 0))
        )
        jd = (_jd(ri) + _jd(rj)) / 2.0
        sm = (float(ri.get("sim_score") or 0) + float(rj.get("sim_score") or 0)) / 2.0
        return (ff + sf + tf + jd + sm) / 5.0

    for i, (ai, ri) in enumerate(flat):
        ti = _rec_tid(ri)
        if ti is None:
            continue
        if cross_out[ti] >= 4:
            continue
        gi = float(ri.get("generic_risk") or 0)
        if gi > 0.42:
            continue
        partners: List[Tuple[float, int, str, Dict[str, Any]]] = []
        for j, (aj, rj) in enumerate(flat):
            if i == j or ai == aj:
                continue
            tj = _rec_tid(rj)
            if tj is None or ti == tj:
                continue
            if float(rj.get("generic_risk") or 0) > 0.42:
                continue
            term_i = str(ri.get("term") or "").strip().lower()
            term_j = str(rj.get("term") or "").strip().lower()
            if not term_i or not term_j or term_i == term_j:
                continue
            ff = (float(ri.get("field_fit") or 0) + float(rj.get("field_fit") or 0)) / 2.0
            jd = (_jd(ri) + _jd(rj)) / 2.0
            sm = (float(ri.get("sim_score") or 0) + float(rj.get("sim_score") or 0)) / 2.0
            if ff < 0.28 or jd < 0.4 or sm < 0.18:
                continue
            sc = _support_score(ri, rj)
            partners.append((sc, tj, aj, rj))
        partners.sort(key=lambda x: -x[0])
        for sc, tj, aj, rj in partners:
            if cross_out[ti] >= 4:
                break
            t_lo, t_hi = (ti, tj) if ti < tj else (tj, ti)
            key = (t_lo, t_hi, "cross_anchor_support")
            if key in dedup_x:
                continue
            dedup_x.add(key)
            cross_edges.append(
                {
                    "src_tid": ti,
                    "dst_tid": tj,
                    "src_term": str(ri.get("term") or ""),
                    "dst_term": str(rj.get("term") or ""),
                    "edge_type": "cross_anchor_support",
                    "relation": "support",
                    "anchor_id": f"{ai}|{aj}",
                    "score_hint": sc,
                    "reason": "conservative_fit_align",
                }
            )
            cross_out[ti] += 1

    # ---------- family_edges：弱 family（同锚、显式痕迹） ----------
    for anchor_id, node in (anchor_to_candidates or {}).items():
        cands = list(node.get("candidates") or [])
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                ri, rj = cands[i], cands[j]
                ti, tj = _rec_tid(ri), _rec_tid(rj)
                if ti is None or tj is None or ti == tj:
                    continue
                m_i = ri.get("stage2_local_meta") or {}
                m_j = rj.get("stage2_local_meta") or {}
                src_i = str(ri.get("candidate_source") or ri.get("source") or "").lower()
                src_j = str(rj.get("candidate_source") or rj.get("source") or "").lower()
                fam_i = bool(m_i.get("has_family_evidence") or ri.get("has_family_evidence"))
                fam_j = bool(m_j.get("has_family_evidence") or rj.get("has_family_evidence"))
                fl_i = "family_landing" in src_i
                fl_j = "family_landing" in src_j
                pp_i = str(m_i.get("parent_primary") or ri.get("parent_primary") or "").strip().lower()
                pp_j = str(m_j.get("parent_primary") or rj.get("parent_primary") or "").strip().lower()
                reason = ""
                if fl_i or fl_j:
                    reason = "family_landing_coexist"
                elif fam_i or fam_j:
                    reason = "has_family_evidence"
                elif pp_i and pp_i == pp_j:
                    reason = "same_parent_primary_family"
                else:
                    continue
                t_lo, t_hi = (ti, tj) if ti < tj else (tj, ti)
                key = (t_lo, t_hi, "family", str(anchor_id))
                if key in dedup_f:
                    continue
                dedup_f.add(key)
                family_edges.append(
                    {
                        "src_tid": ti,
                        "dst_tid": tj,
                        "src_term": str(ri.get("term") or ""),
                        "dst_term": str(rj.get("term") or ""),
                        "edge_type": "family",
                        "relation": "family_weak",
                        "anchor_id": str(anchor_id),
                        "score_hint": None,
                        "reason": reason,
                    }
                )

    # ---------- provenance_edges：expansion → landing 父候选（按 parent_primary 匹配 term） ----------
    for anchor_id, node in (anchor_to_candidates or {}).items():
        exp_list = list(node.get("expansion_candidates") or [])
        if not exp_list:
            continue
        pool = list(node.get("landing_candidates") or []) + list(node.get("candidates") or [])
        by_term: Dict[str, Dict[str, Any]] = {}
        for r in pool:
            tt = str(r.get("term") or "").strip().lower()
            if tt and tt not in by_term:
                by_term[tt] = r
        for child in exp_list:
            tc = _rec_tid(child)
            if tc is None:
                continue
            meta = child.get("stage2_local_meta") or {}
            pp = str(meta.get("parent_primary") or child.get("parent_primary") or "").strip().lower()
            if not pp:
                continue
            parent = by_term.get(pp)
            if parent is None:
                continue
            tp = _rec_tid(parent)
            if tp is None or tp == tc:
                continue
            key = (tp, tc, "provenance", str(anchor_id))
            if key in dedup_p:
                continue
            dedup_p.add(key)
            provenance_edges.append(
                {
                    "src_tid": tp,
                    "dst_tid": tc,
                    "src_term": str(parent.get("term") or ""),
                    "dst_term": str(child.get("term") or ""),
                    "edge_type": "provenance",
                    "relation": "expanded_from",
                    "anchor_id": str(anchor_id),
                    "score_hint": _rec_score_hint(parent),
                    "reason": "parent_primary_term_match",
                }
            )

    return {
        "same_anchor_edges": same_anchor_edges,
        "cross_anchor_support_edges": cross_edges,
        "family_edges": family_edges,
        "provenance_edges": provenance_edges,
    }


def _rec_goes_to_expansion_slot(rec: Dict[str, Any]) -> bool:
    """
    近似：是否放入 expansion_candidates（非 2A/2B 数据流拆开，仅为槽位与后续图/Stage3 铺垫）。
    分不清时 False（归入 landing）。TODO: 与 Stage2A/2B 显式来源字段对齐后收紧。
    """
    meta = rec.get("stage2_local_meta") or {}
    src = str(rec.get("candidate_source") or rec.get("source") or "").strip().lower()
    pb = str(meta.get("primary_bucket") or rec.get("primary_bucket") or "").strip().lower()
    if pb in ("dense_expansion", "cluster_expansion", "cooc_expansion"):
        return True
    if bool(rec.get("can_expand")) and not bool(rec.get("can_expand_from_2a")):
        return True
    if any(tok in src for tok in ("expansion", "stage2b", "cooc", "cluster_exp", "dense_exp")):
        return True
    return False


def _anchor_node_summary(
    candidates: List[Dict[str, Any]],
    landing: List[Dict[str, Any]],
    expansion: List[Dict[str, Any]],
) -> Dict[str, Any]:
    def _pb(r: Dict[str, Any]) -> str:
        return str((r.get("stage2_local_meta") or {}).get("primary_bucket") or r.get("primary_bucket") or "").strip().lower()

    # TODO(Stage2): primary_bucket 别名/多标签时收紧
    support_seed_count = sum(1 for r in candidates if _pb(r) == "support_seed")
    support_keep_count = sum(1 for r in candidates if _pb(r) == "support_keep")
    risky_keep_count = sum(1 for r in candidates if _pb(r) in ("risky_keep", "risky"))
    return {
        "candidate_count": len(candidates),
        "landing_count": len(landing),
        "expansion_count": len(expansion),
        "support_seed_count": support_seed_count,
        "support_keep_count": support_keep_count,
        "risky_keep_count": risky_keep_count,
    }


def _organize_anchor_to_candidates(all_candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    anchor_to_candidates：正式 anchor 节点视图（非 list 直挂），供 candidate_graph / Stage3 / report。
    外层 key 过渡方案：anchor_id 字段 > anchor_term > parent_anchor；与 PreparedAnchor.vid 统一留待后续。
    landing/expansion 槽位为基于 candidate_source、primary_bucket、can_expand 等的近似归类。
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in all_candidates:
        aid = rec.get("anchor_id")
        key = str(aid).strip() if aid is not None and str(aid).strip() else ""
        if not key:
            key = str(rec.get("anchor_term") or rec.get("parent_anchor") or "_unknown")
        groups[key].append(rec)

    out: Dict[str, Dict[str, Any]] = {}
    for anchor_key, cands in groups.items():
        landing: List[Dict[str, Any]] = []
        expansion: List[Dict[str, Any]] = []
        for rec in cands:
            if _rec_goes_to_expansion_slot(rec):
                expansion.append(rec)
            else:
                landing.append(rec)
        first = cands[0]
        anchor_term = str(first.get("anchor_term") or first.get("parent_anchor") or anchor_key)
        meta0 = first.get("stage2_local_meta") or {}
        anchor_source_term = str(
            first.get("parent_primary") or meta0.get("parent_primary") or anchor_term
        )
        summ = _anchor_node_summary(cands, landing, expansion)
        out[anchor_key] = {
            "anchor_id": anchor_key,
            "anchor_term": anchor_term,
            "anchor_source_term": anchor_source_term,
            "candidates": cands,
            "landing_candidates": landing,
            "expansion_candidates": expansion,
            "summary": summ,
        }
    return out


def _build_stage2_report_min(
    all_candidates: List[Dict[str, Any]],
    prepared_anchors: List,
    anchor_to_candidates: Dict[str, Dict[str, Any]],
    candidate_graph: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, Any]:
    """最小 report：与 anchor 节点 summary 对齐；含 graph_edge_counts。"""
    source_distribution: Dict[str, int] = {}
    for rec in all_candidates:
        src = str(rec.get("source") or rec.get("origin") or "").strip() or "(empty)"
        source_distribution[src] = source_distribution.get(src, 0) + 1

    per_anchor_stats: Dict[str, Any] = {}
    landing_candidate_count = 0
    expansion_candidate_count = 0
    for k, node in anchor_to_candidates.items():
        summ = (node or {}).get("summary") or {}
        cands = (node or {}).get("candidates") or []
        per_anchor_stats[k] = {
            "candidate_count": int(summ.get("candidate_count", len(cands))),
            "landing_count": int(summ.get("landing_count", 0)),
            "expansion_count": int(summ.get("expansion_count", 0)),
            "support_seed_count": int(summ.get("support_seed_count", 0)),
            "support_keep_count": int(summ.get("support_keep_count", 0)),
            "risky_keep_count": int(summ.get("risky_keep_count", 0)),
        }
        landing_candidate_count += per_anchor_stats[k]["landing_count"]
        expansion_candidate_count += per_anchor_stats[k]["expansion_count"]

    cg = candidate_graph if candidate_graph is not None else _empty_candidate_graph()
    graph_edge_counts = {
        "same_anchor_edges": len(cg.get("same_anchor_edges") or []),
        "cross_anchor_support_edges": len(cg.get("cross_anchor_support_edges") or []),
        "family_edges": len(cg.get("family_edges") or []),
        "provenance_edges": len(cg.get("provenance_edges") or []),
    }

    return {
        "anchor_count": len(prepared_anchors),
        "candidate_count": len(all_candidates),
        "landing_candidate_count": landing_candidate_count,
        "expansion_candidate_count": expansion_candidate_count,
        "source_distribution": source_distribution,
        "per_anchor_stats": per_anchor_stats,
        "graph_edge_counts": graph_edge_counts,
    }


def _print_anchor_node_debug(anchor_to_candidates: Dict[str, Dict[str, Any]]) -> None:
    if not anchor_to_candidates:
        return
    first_key = next(iter(anchor_to_candidates))
    first_node = anchor_to_candidates[first_key]
    print("\n[Stage2] anchor_to_candidates first key:", first_key)
    print("[Stage2] anchor node keys:", sorted(first_node.keys()))
    print("[Stage2] anchor node summary:", first_node.get("summary", {}))


def _print_candidate_graph_samples(candidate_graph: Dict[str, List[Any]]) -> None:
    print(
        "\n[Stage2] candidate_graph edge counts:",
        {k: len(v) for k, v in candidate_graph.items()},
    )
    for edge_type in ("same_anchor_edges", "cross_anchor_support_edges", "family_edges", "provenance_edges"):
        sample_edges = (candidate_graph.get(edge_type) or [])[:2]
        if sample_edges:
            print(f"[Stage2] sample {edge_type}:", sample_edges)


def _print_stage2_structured_output(stage2_output: Dict[str, Any]) -> None:
    """主链跑一次即可看到 Stage2 新 dict 契约（本轮最小可见输出）。边数量与样例见 _print_candidate_graph_samples。"""
    print("\n[Stage2] output keys:", list(stage2_output.keys()))
    print("[Stage2] all_candidates:", len(stage2_output["all_candidates"]))
    print("[Stage2] anchors:", len(stage2_output["anchor_to_candidates"]))
    rep = stage2_output["stage2_report"]
    print(
        "[Stage2] stage2_report summary:",
        {
            "anchor_count": rep.get("anchor_count"),
            "candidate_count": rep.get("candidate_count"),
            "graph_edge_counts": rep.get("graph_edge_counts"),
        },
    )


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
) -> Dict[str, Any]:
    """
    阶段 2：学术词扩展。
    先转 anchor_skills 为 PreparedAnchor，再走 Stage2A 主落点 + Stage2B 仅围绕 primary 扩展；
    传入 jd_profile 时启用四层领域契合与泛词惩罚。

    返回结构化 dict（run_stage2 返回契约改造；all_candidates 仍为原 _expanded_to_raw_candidates 列表语义）。
    """
    prepared_anchors = _anchor_skills_to_prepared_anchors(recall, anchor_skills, query_text=query_text)
    if not prepared_anchors:
        _eg = _empty_candidate_graph()
        out: Dict[str, Any] = {
            "all_candidates": [],
            "anchor_to_candidates": {},
            "candidate_graph": _eg,
            "stage2_report": _build_stage2_report_min([], [], {}, candidate_graph=_eg),
        }
        _print_stage2_structured_output(out)
        return out
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
    # 出口薄包装：all_candidates 仍为平铺；anchor_to_candidates 为正式 anchor 节点（含 landing/expansion 槽位近似）。
    all_candidates = _expanded_to_raw_candidates(terms)
    anchor_to_candidates = _organize_anchor_to_candidates(all_candidates)
    _print_anchor_node_debug(anchor_to_candidates)
    candidate_graph = build_stage2_candidate_graph(
        anchor_to_candidates,
        all_candidates,
        jd_profile=jd_profile,
        active_domains=active_domain_set,
    )
    _print_candidate_graph_samples(candidate_graph)
    stage2_output: Dict[str, Any] = {
        "all_candidates": all_candidates,
        "anchor_to_candidates": anchor_to_candidates,
        "candidate_graph": candidate_graph,
        "stage2_report": _build_stage2_report_min(
            all_candidates, prepared_anchors, anchor_to_candidates, candidate_graph=candidate_graph
        ),
    }
    _print_stage2_structured_output(stage2_output)
    return stage2_output
