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
# 标签路追踪：是否打印 source_type / anchor / similar_to 原始候选 / 被过滤原因 / final primary 胜出原因
LABEL_PATH_TRACE = True
# family 保送式 paper 选词：每 family 最多 1 primary + 1 support，cluster 默认不进 paper recall
PAPER_RECALL_MAX_TERMS = 12

# 全局共识因子默认值（无硬编码）
STAGE3_CROSS_ANCHOR_DEFAULT = 1.0
STAGE3_CLUSTER_COHESION_DEFAULT = 1.0
STAGE3_DRIFT_RISK_DEFAULT = 0.5


def _compute_stage3_global_consensus(
    recall,
    raw_candidates: List[Dict[str, Any]],
) -> None:
    """
    为每条候选计算 cross_anchor_evidence、cluster_cohesion、semantic_drift_risk，原地写入 rec。
    无词表，仅用候选池内统计与图谱结构。
    """
    if not raw_candidates:
        return
    tid_to_anchors: Dict[int, Set[str]] = defaultdict(set)
    tid_to_cluster: Dict[int, Any] = {}
    all_anchors: Set[str] = set()
    voc_to_clusters = getattr(recall, "voc_to_clusters", None) or {}
    for rec in raw_candidates:
        tid = rec.get("tid") or rec.get("vid")
        if tid is not None:
            try:
                tid = int(tid)
            except (TypeError, ValueError):
                tid = None
        if tid is not None:
            pa = (rec.get("parent_anchor") or "").strip()
            if pa:
                tid_to_anchors[tid].add(pa)
                all_anchors.add(pa)
            if voc_to_clusters:
                clusters = voc_to_clusters.get(tid) or []
                if clusters:
                    cid, _ = max(clusters, key=lambda x: (x[1], x[0]))
                    tid_to_cluster[tid] = cid
    n_candidates = len(raw_candidates)
    n_anchors = max(1, len(all_anchors))
    cluster_to_count: Dict[Any, int] = defaultdict(int)
    for tid, cid in tid_to_cluster.items():
        cluster_to_count[cid] += 1
    for rec in raw_candidates:
        tid = rec.get("tid") or rec.get("vid")
        try:
            tid = int(tid) if tid is not None else None
        except (TypeError, ValueError):
            tid = None
        anchors_for_tid = tid_to_anchors.get(tid) or set()
        cross_anchor = min(1.0, 0.3 + 0.7 * len(anchors_for_tid) / max(5, n_anchors)) if n_anchors else STAGE3_CROSS_ANCHOR_DEFAULT
        cid = tid_to_cluster.get(tid) if tid is not None else None
        same_cluster = cluster_to_count.get(cid, 0) if cid is not None else 0
        cohesion = min(1.0, 0.2 + 0.8 * (same_cluster / max(1, n_candidates))) if n_candidates else STAGE3_CLUSTER_COHESION_DEFAULT
        outside = float(rec.get("outside_subfield_mass") or 0.0)
        topic_fit = float(rec.get("topic_fit") or rec.get("subfield_fit") or 0.5)
        n_supp = max(1, len(anchors_for_tid))
        drift = 0.4 * min(1.0, outside) + 0.3 * (1.0 / n_supp) + 0.3 * (1.0 - topic_fit)
        drift = max(0.01, min(1.0, drift))
        rec["cross_anchor_evidence"] = cross_anchor
        rec["cluster_cohesion"] = cohesion
        rec["semantic_drift_risk"] = drift


def select_terms_for_paper_recall(
    records: List[Dict[str, Any]],
    max_terms: int = 12,
) -> List[Dict[str, Any]]:
    """
    按 family 分桶选词：每 family 至少保 1 个 primary、最多再保 1 个 support；cluster(blocked) 不进 paper recall。
    不再全局 top-k，避免单 family 占满名额。
    """
    if not records:
        return []
    family_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        fk = rec.get("family_key") or build_family_key(rec)
        rec["family_key"] = fk
        rec["retrieval_role"] = rec.get("retrieval_role") or get_retrieval_role_from_term_role(rec.get("term_role"))
        family_buckets[fk].append(rec)

    selected: List[Dict[str, Any]] = []
    for fk, members in family_buckets.items():
        members = sorted(members, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
        primarys = [m for m in members if (m.get("retrieval_role") or "").strip().lower() == "paper_primary"]
        if primarys:
            best = max(primarys, key=lambda x: float(x.get("final_score") or 0.0))
            best["win_reason"] = f"family {fk!r} 下 primary 中 final_score 最高({float(best.get('final_score') or 0):.4f})"
            selected.append(best)
        supports = [m for m in members if (m.get("retrieval_role") or "").strip().lower() == "paper_support"]
        if supports:
            best = max(supports, key=lambda x: float(x.get("final_score") or 0.0))
            best["win_reason"] = f"family {fk!r} 下 support 中 final_score 最高({float(best.get('final_score') or 0):.4f})"
            selected.append(best)

    selected = sorted(selected, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    return selected[:max_terms]


def _run_stage3_dual_gate(
    recall,
    raw_candidates: List[Dict[str, Any]],
    query_vector,
    anchor_vids=None,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], List[Dict[str, Any]]]:
    """
    Stage3 两层：第一层只做少量硬过滤；第二层保留按 final_score 排序的 top_k。
    再按 family 保送式选词得到 paper_terms（每 family 1 primary + 1 support，cluster 不进 paper recall）。
    返回 7 个 map + paper_terms。
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
    dropped_with_reason: List[Dict[str, Any]] = []  # 被过滤候选 + 原因，供标签路追踪

    if stage3_debug and raw_candidates:
        print("[stage3_input] tid | term | source_type | parent_anchor | parent_primary | score")
        for i, rec in enumerate(raw_candidates[:25]):
            tid = rec.get("tid")
            term = (rec.get("term") or "")[:24]
            st = (rec.get("source") or rec.get("origin") or "")
            pa = rec.get("parent_anchor") or ""
            pp = rec.get("parent_primary") or ""
            sc = rec.get("identity_score") or rec.get("sim_score") or 0
            print(f"  {i+1} {tid} | {term!r} | {st} | {pa!r} | {pp!r} | {sc:.3f}")
        if len(raw_candidates) > 25:
            print(f"  ... 共 {len(raw_candidates)} 条")

    _compute_stage3_global_consensus(recall, raw_candidates)
    for i, rec in enumerate(raw_candidates):
        rec["stage2_rank"] = i

    for rec in raw_candidates:
        drop, reject_reason = should_drop_term(rec)
        if drop:
            rec["reject_reason"] = reject_reason
            dropped_with_reason.append(rec)
            if STAGE3_DETAIL_DEBUG or stage3_debug:
                print(f"[Stage3 硬过滤] drop tid={rec.get('tid')} term={rec.get('term')!r} reason={reject_reason}")
            continue
        rec["reject_reason"] = ""
        if not term_scoring.passes_identity_gate(rec):
            rec["reject_reason"] = "identity_gate"
            dropped_with_reason.append(rec)
            if label_trace:
                print(f"[Stage3 被过滤] tid={rec.get('tid')} term={rec.get('term')!r} source_type={rec.get('source') or rec.get('origin')} parent_anchor={rec.get('parent_anchor')!r} reason=identity_gate")
            continue
        if not term_scoring.passes_topic_consistency(rec, None):
            rec["reject_reason"] = "topic_consistency"
            dropped_with_reason.append(rec)
            if label_trace:
                print(f"[Stage3 被过滤] tid={rec.get('tid')} term={rec.get('term')!r} source_type={rec.get('source') or rec.get('origin')} parent_anchor={rec.get('parent_anchor')!r} reason=topic_consistency")
            continue
        rec["quality_score"] = term_scoring.score_term_expansion_quality(recall, rec)
        rec["final_score"] = term_scoring.compose_term_final_score(rec)
        fit_info = {
            "domain_fit": rec.get("domain_fit"),
            "field_fit": rec.get("field_fit"),
            "subfield_fit": rec.get("subfield_fit"),
            "topic_fit": rec.get("topic_fit"),
            "outside_subfield_mass": rec.get("outside_subfield_mass"),
            "outside_topic_mass": rec.get("outside_topic_mass"),
            "main_subfield_match": rec.get("main_subfield_match"),
        }
        rec["fit_info"] = {k: v for k, v in fit_info.items() if v is not None}
        final_score, explain = score_term_record(rec)
        rec["final_score"] = final_score
        rec["stage3_explain"] = explain
        rec["family_key"] = build_family_key(rec)
        rec["retrieval_role"] = get_retrieval_role_from_term_role(rec.get("term_role"))
        survivors.append(rec)

    survivors.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    for i, rec in enumerate(survivors):
        rec["stage3_rank"] = i
    rerank_delta_rows = []
    for rec in survivors[:25]:
        s2 = rec.get("stage2_rank", 0)
        s3 = rec.get("stage3_rank", 0)
        rerank_delta_rows.append({"term": rec.get("term") or "", "stage2_rank": s2, "stage3_rank": s3, "delta": s2 - s3})
    risky_terms_list = [r for r in survivors if (float(r.get("semantic_drift_risk") or 0) >= 0.55) or (float(r.get("cross_anchor_evidence") or 1) < 0.5)]
    core_terms_list = [r for r in survivors if r not in risky_terms_list][: max(1, (len(survivors) * 4 // 10))]
    support_terms_list = [r for r in survivors if r not in risky_terms_list and r not in core_terms_list]
    for r in risky_terms_list:
        reasons = []
        if float(r.get("cross_anchor_evidence") or 1) < 0.5:
            reasons.append("single_anchor_or_few")
        if float(r.get("cluster_cohesion") or 1) < 0.4:
            reasons.append("low_cluster_cohesion")
        if float(r.get("semantic_drift_risk") or 0) > 0.6:
            reasons.append("high_drift_risk")
        r["risk_reasons"] = reasons
    top_survivors = survivors[: STAGE3_TOP_K]
    paper_terms = select_terms_for_paper_recall(survivors, PAPER_RECALL_MAX_TERMS)

    for rec in top_survivors:
        tid_str = str(rec.get("tid", ""))
        if not tid_str or tid_str == "None":
            continue
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
        }
        if rec.get("stage3_explain"):
            entry.update(rec["stage3_explain"])
        debug_metrics = term_scoring.get_term_debug_metrics(recall, rec, query_vector)
        if debug_metrics:
            entry.update(debug_metrics)
        tid = rec.get("tid")
        if tid is not None and hasattr(recall, "_get_cluster_factor_for_term"):
            try:
                entry["cluster_factor"] = recall._get_cluster_factor_for_term(int(tid))
            except (TypeError, ValueError):
                pass
        recall.debug_info.tag_purity_debug.append(entry)

    # 保证 paper_terms 中的 tid 均在 score_map/term_map 等中（供 Stage4/Stage5 使用）
    for rec in paper_terms:
        tid_str = str(rec.get("tid", ""))
        if not tid_str or tid_str == "None" or tid_str in score_map:
            continue
        score_map[tid_str] = float(rec.get("final_score") or 0.0)
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

    debug_print(1, f"[Stage3] 输入候选总数={len(raw_candidates)} 幸存={len(survivors)} top_k={len(top_survivors)}", recall)
    debug_print(2, "[Stage3 Final Score Breakdown] term | stage2_rank | cross_anchor | cohesion | drift_risk | final", recall)
    for rec in survivors[:15]:
        t = (rec.get("term") or "")[:28]
        debug_print(2, (
            f"  {t:<28} | s2={rec.get('stage2_rank', 0):>2} | "
            f"cross={rec.get('cross_anchor_evidence', 0):.3f} | cohes={rec.get('cluster_cohesion', 0):.3f} | "
            f"drift={rec.get('semantic_drift_risk', 0):.3f} | final={rec.get('final_score', 0):.3f}"
        ), recall)
    debug_print(2, "[Stage3 Rerank Delta] term | stage2_rank | stage3_rank | delta", recall)
    for row in rerank_delta_rows[:15]:
        t = (row.get("term") or "")[:28]
        debug_print(2, f"  {t:<28} | {row.get('stage2_rank', 0):>3} | {row.get('stage3_rank', 0):>3} | {row.get('delta', 0):+d}", recall)
    debug_print(2, "[Stage3 Buckets]", recall)
    debug_print(2, f"  core_terms={[r.get('term') for r in core_terms_list[:15]]}", recall)
    debug_print(2, f"  support_terms={[r.get('term') for r in support_terms_list[:15]]}", recall)
    debug_print(2, f"  risky_terms={[r.get('term') for r in risky_terms_list[:15]]}", recall)
    debug_print(3, "[Stage3 Risky Term Reasons] term | reasons | final", recall)
    for r in risky_terms_list[:10]:
        t = (r.get("term") or "")[:28]
        debug_print(3, f"  {t:<28} | {r.get('risk_reasons', [])} | {r.get('final_score', 0):.3f}", recall)

    if STAGE3_DETAIL_DEBUG or stage3_debug:
        print(f"[paper_term_selection] family 保送式 选词数={len(paper_terms)} 明细: family_key | term | term_role | retrieval_role | final_score")
        for r in paper_terms[:30]:
            print(f"  {r.get('family_key','')} | {r.get('term','')!r} | {r.get('term_role','')} | {r.get('retrieval_role','')} | {r.get('final_score',0):.4f}")
        if len(paper_terms) > 30:
            print(f"  ... 共 {len(paper_terms)} 条")
        print(f"[Stage3] 轻过滤+top_k 输入={len(raw_candidates)} 硬过滤后幸存={len(survivors)} 保留 top_k={len(top_survivors)} 输出词数={len(score_map)} paper_terms={len(paper_terms)}")
        print("[Stage3 final_score 明细] term | base_score | hierarchy_score | external_penalty | entropy_penalty | generic_penalty | final_score | reject_reason")
        for i, rec in enumerate(top_survivors[:20]):
            term = (rec.get("term") or "")[:28]
            ex = rec.get("stage3_explain") or {}
            base_score = ex.get("base_score", rec.get("final_score")) or 0.0
            hierarchy_score = ex.get("hierarchy_score")
            ext_pen = ex.get("external_penalty")
            ent_pen = ex.get("entropy_penalty")
            gen_pen = ex.get("generic_penalty")
            final_score = rec.get("final_score", 0) or 0.0
            reject = rec.get("reject_reason", "")
            _fmt = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else str(x)
            print(f"  {i+1} {term!r} | base={_fmt(base_score)} | hier={_fmt(hierarchy_score)} | ext={_fmt(ext_pen)} | ent={_fmt(ent_pen)} | gen={_fmt(gen_pen)} | final={final_score:.4f} | {reject!r}")
        if len(top_survivors) > 20:
            print(f"  ... 共 {len(top_survivors)} 条")

    if stage3_debug and score_map:
        print("[stage3_output] tid | term | source_type | parent_anchor | parent_primary | score")
        for i, (tid_str, sc) in enumerate(sorted(score_map.items(), key=lambda x: -x[1])[:25]):
            print(f"  {i+1} {tid_str} | {term_map.get(tid_str, '')!r} | {term_source_map.get(tid_str, '')} | {parent_anchor_map.get(tid_str, '')!r} | {parent_primary_map.get(tid_str, '')!r} | {sc:.3f}")

    # 标签路追踪：被过滤原因汇总
    if label_trace and dropped_with_reason:
        recall.debug_info.dropped_with_reason = dropped_with_reason
        print("[标签路-被过滤原因] tid | term | source_type | parent_anchor | 原因")
        for r in dropped_with_reason[:50]:
            print(f"  {r.get('tid')} | {r.get('term','')!r} | {r.get('source') or r.get('origin','')} | {r.get('parent_anchor','')!r} | {r.get('reject_reason','')}")
        if len(dropped_with_reason) > 50:
            print(f"  ... 共 {len(dropped_with_reason)} 条被过滤")
    else:
        recall.debug_info.dropped_with_reason = getattr(recall.debug_info, "dropped_with_reason", None) or []

    # 标签路追踪：final primary 为什么胜出
    if label_trace and paper_terms:
        print("[标签路-final primary 为什么胜出] tid | term | source_type | parent_anchor | retrieval_role | 胜出原因")
        for r in paper_terms[:30]:
            wr = r.get("win_reason", "")
            print(f"  {r.get('tid')} | {r.get('term','')!r} | {r.get('source') or r.get('origin','')} | {r.get('parent_anchor','')!r} | {r.get('retrieval_role','')} | {wr}")
        if len(paper_terms) > 30:
            print(f"  ... 共 {len(paper_terms)} 条")

    return score_map, term_map, idf_map, term_role_map, term_source_map, parent_anchor_map, parent_primary_map, paper_terms


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

    # 保留原有的 verbose 调试逻辑（从 _stage3_word_weights 搬运）
    if recall.verbose and score_map and getattr(recall, "_last_tag_purity_debug", None):
        debug_by_tid = {str(d["tid"]): d for d in recall.debug_info.tag_purity_debug if d.get("tid") is not None}
        top_tids = sorted(score_map.keys(), key=lambda t: score_map.get(t, 0.0), reverse=True)
        cluster_factors = getattr(recall, "_last_cluster_rank_factors", {}) or recall.debug_info.cluster_rank_factors or {}

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

        print("\n【观测面板 1/4】基础与语义 (Top30)")
        print("  #   term                            hits deg_w deg_exp tgt_w  raw_pur cap_pur  cos_sim anc_sim ta_sim  ca_sim  ta_adv  idf_term  cov_j")
        for i, tid in enumerate(head30, 1):
            d = debug_by_tid.get(str(tid), {})
            term = (d.get("term") or term_map.get(tid, ""))[:32]
            print(
                f"  {i:2d}  {term:32s}  {_f(d.get('hit_count'),4,0):>4} {_f(d.get('degree_w'),5,0):>5} "
                f"{_f(d.get('degree_w_expanded'),6,0):>6} {_f(d.get('target_degree_w'),5,0):>5} "
                f"{_f(d.get('raw_tag_purity'),7):>7} {_f(d.get('capped_tag_purity'),7):>7} {_f(d.get('cos_sim'),7):>7} "
                f"{_f(d.get('anchor_sim'),7):>7} {_f(d.get('task_anchor_sim'),6):>6} {_f(d.get('carrier_anchor_sim'),6):>6} "
                f"{_f(d.get('task_advantage'),6):>6} {_f(d.get('idf_term'),8):>8} {_f(d.get('cov_j'),6):>6}"
            )

        print("\n【观测面板 2/4】骨架与共鸣 (Top30)")
        print("  #   term                            sem_fac buck_f hit_t  pur_t  tiny_p  base_sc  res   anc_res anc_rat res_fac hit_fac conv_bonus")
        for i, tid in enumerate(head30, 1):
            d = debug_by_tid.get(str(tid), {})
            term = (d.get("term") or term_map.get(tid, ""))[:32]
            print(
                f"  {i:2d}  {term:32s}  {_f(d.get('semantic_factor'),6):>6} {_f(d.get('bucket_factor'),6):>6} "
                f"{_f(d.get('hits_term'),6):>6} {_f(d.get('purity_term'),6):>6} {_f(d.get('tiny_penalty'),6):>6} "
                f"{_f(d.get('base_score'),7):>7} {_f(d.get('resonance'),5):>5} {_f(d.get('anchor_resonance'),7):>7} "
                f"{_f(d.get('anchor_ratio'),6):>6} {_f(d.get('resonance_factor'),6):>6} {_f(d.get('hit_count_factor'),6):>6} "
                f"{_f(d.get('convergence_bonus'),9):>9}"
            )

        print("\n【观测面板 3/4】角色 (Top30)")
        print("  #   term                            tc_str  ab_str  car_str nz_str  nz_pen  norm_t  norm_a  norm_c  main_role        margin  main_p  aux_p   alpha  rp_wo_nz role_pen")
        for i, tid in enumerate(head30, 1):
            d = debug_by_tid.get(str(tid), {})
            term = (d.get("term") or term_map.get(tid, ""))[:32]
            main_role = (d.get("main_role") or "none")[:14]
            print(
                f"  {i:2d}  {term:32s}  {_f(d.get('task_core_strength'),6):>6} {_f(d.get('abstract_strength'),6):>6} "
                f"{_f(d.get('carrier_strength'),6):>6} {_f(d.get('noise_strength'),6):>6} {_f(d.get('noise_penalty'),6):>6} "
                f"{_f(d.get('norm_task'),6):>6} {_f(d.get('norm_abs'),6):>6} {_f(d.get('norm_car'),6):>6}  {main_role:14s}  "
                f"{_f(d.get('role_margin'),6):>6} {_f(d.get('main_penalty'),6):>6} {_f(d.get('aux_penalty_avg'),6):>6} "
                f"{_f(d.get('alpha'),5):>5} {_f(d.get('role_penalty_without_noise'),7):>7} {_f(d.get('role_penalty'),6):>6}"
            )

        print("\n【观测面板 4/4】最终公式与得分 (Top30)")
        print("  #   term                            anc_f  job_pen d_span d_span_pen cooc_s  cooc_p  span_pen pur_bon clu_fac  term_bb  idf_val  dyn_w    clu_rank  final_score")
        for i, tid in enumerate(head30, 1):
            d = debug_by_tid.get(str(tid), {})
            term = (d.get("term") or term_map.get(tid, ""))[:32]
            final_score = score_map.get(tid, 0.0)
            clu_rank = cluster_factors.get(str(tid), 1.0)
            print(
                f"  {i:2d}  {term:32s}  {_f(d.get('anchor_factor'),5):>5} {_f(d.get('job_penalty'),6):>6} "
                f"{_f(d.get('domain_span'),5,0):>5} {_f(d.get('domain_span_penalty'),9):>9} {_f(d.get('cooc_span'),6):>6} "
                f"{_f(d.get('cooc_purity'),6):>6} {_f(d.get('span_penalty'),7):>7} {_f(d.get('purity_bonus'),7):>7} "
                f"{_f(d.get('cluster_factor'),7):>7} {_f(d.get('term_backbone'),7):>7} {_f(d.get('idf_val'),7):>7} "
                f"{_f(d.get('dynamic_weight'),7):>7}  {_f(clu_rank,7):>7}  {final_score:.6f}"
            )

        print("\n【日志 1】Top term 簇信息：term | cluster_id | cluster_score | cluster_factor | task_anchor_sim | task_advantage")
        for i, tid in enumerate(head30[:20], 1):
            term = (term_map.get(tid, "") or "")[:28]
            d = debug_by_tid.get(str(tid), {})
            cluster_id = None
            cluster_score = None
            if getattr(recall, "voc_to_clusters", None):
                clusters = recall.voc_to_clusters.get(int(tid))
                if clusters:
                    cid, csc = max(clusters, key=lambda x: x[1])
                    cluster_id = cid
                    cluster_score = round(csc, 4)
            clu_fac = d.get("cluster_factor")
            if clu_fac is None and hasattr(recall, "_get_cluster_factor_for_term"):
                clu_fac = recall._get_cluster_factor_for_term(int(tid))
            print(
                f"  {i:2d}  {term:28s}  cid={cluster_id}  c_score={cluster_score}  "
                f"clu_fac={_f(clu_fac,5):>5}  ta_sim={_f(d.get('task_anchor_sim'),5):>5}  ta_adv={_f(d.get('task_advantage'),5):>5}"
            )

        if getattr(recall, "voc_to_clusters", None) and getattr(recall, "cluster_members", None):
            hit_cids = set()
            for tid in head30:
                clusters = recall.voc_to_clusters.get(int(tid))
                if clusters:
                    cid, _ = max(clusters, key=lambda x: x[1])
                    hit_cids.add(cid)
            print("\n【日志 3】命中簇前 10 成员：cluster_id | 前 10 个 member term (及 task_anchor_sim)")
            for cid in list(hit_cids)[:20]:
                mems = recall.cluster_members.get(int(cid)) or []
                line = [f"  cid={cid}:"]
                for mid in mems[:10]:
                    d = debug_by_tid.get(str(mid), {})
                    t = (d.get("term") or term_map.get(str(mid), "") or "")[:18]
                    ta_sim = _f(d.get("task_anchor_sim"), 5)
                    line.append(f"{t}({ta_sim})")
                print(" ".join(line))

        print("\n【观测面板 汇总】Top30: term | base_score | task_core | abstract | carrier | noise_pen | main_role | role_margin | role_penalty | ta-ca | task_adv | final_score")
        for i, tid in enumerate(head30, 1):
            d = debug_by_tid.get(str(tid), {})
            term = (d.get("term") or term_map.get(tid, ""))[:24]
            ta = _fv(d, "task_anchor_sim")
            ca = _fv(d, "carrier_anchor_sim")
            ta_ca = (ta - ca) if (ta is not None and ca is not None) else None
            print(
                f"  #{i:2d}  {term:24s}  base={_f(d.get('base_score'),7):>7}  tc={_f(d.get('task_core_strength'),5):>5}  "
                f"ab={_f(d.get('abstract_strength'),5):>5}  car={_f(d.get('carrier_strength'),5):>5}  "
                f"nz_pen={_f(d.get('noise_penalty'),5):>5}  role={(d.get('main_role') or 'none'):16s}  "
                f"margin={_f(d.get('role_margin'),5):>5}  r_pen={_f(d.get('role_penalty'),5):>5}  "
                f"ta-ca={_f(ta_ca,6):>6}  adv={_f(d.get('task_advantage'),5):>5}  final={score_map.get(tid, 0.0):.6f}"
            )

    return score_map, term_map, idf_map, term_role_map, term_source_map, parent_anchor_map, parent_primary_map, paper_terms

