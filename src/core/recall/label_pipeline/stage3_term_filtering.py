from typing import Any, Dict, List, Tuple

from src.core.recall.label_means import term_scoring


def _run_stage3_dual_gate(
    recall,
    raw_candidates: List[Dict[str, Any]],
    query_vector,
    anchor_vids=None,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float]]:
    """Stage3 双闸门路径：identity 闸门 -> topic 闸门 -> quality 分 -> 最终分 -> 质量闸门。"""
    score_map: Dict[str, float] = {}
    term_map: Dict[str, str] = {}
    idf_map: Dict[str, float] = {}
    recall.debug_info.tag_purity_debug = []
    recall._last_tag_purity_debug = recall.debug_info.tag_purity_debug
    stage3_debug = getattr(term_scoring, "STAGE3_DEBUG", False)
    n_identity_ok, n_final_ok = 0, 0

    for rec in raw_candidates:
        if not term_scoring.passes_identity_gate(rec):
            continue
        n_identity_ok += 1
        if not term_scoring.passes_topic_consistency(rec, None):
            continue
        rec["quality_score"] = term_scoring.score_term_expansion_quality(recall, rec)
        rec["final_score"] = term_scoring.compose_term_final_score(rec)
        if rec.get("final_score", 0.0) < term_scoring.FINAL_MIN_TERM_SCORE:
            continue
        n_final_ok += 1
        tid_str = str(rec.get("tid", ""))
        if not tid_str or tid_str == "None":
            continue
        score_map[tid_str] = float(rec["final_score"])
        term_map[tid_str] = rec.get("term") or ""
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
        }
        # 合并调试指标（task_anchor_sim、task_advantage、cluster_factor、base_score 等）供日志打印
        debug_metrics = term_scoring.get_term_debug_metrics(recall, rec, query_vector)
        if debug_metrics:
            entry.update(debug_metrics)
        tid = rec.get("tid")
        if tid is not None and hasattr(recall, "_get_cluster_factor_for_term"):
            try:
                entry["cluster_factor"] = recall._get_cluster_factor_for_term(int(tid))
            except (TypeError, ValueError):
                pass
        identity = float(rec.get("identity_score") or 0.0)
        quality = float(rec.get("quality_score") or 0.0)
        role = (rec.get("term_role") or "").strip().lower()
        if role == "primary":
            entry["base_score"] = 0.7 * identity + 0.3 * quality
        elif role in ("dense_expansion", "cluster_expansion"):
            entry["base_score"] = 0.4 * identity + 0.6 * quality
        elif role == "cooc_expansion":
            entry["base_score"] = 0.3 * identity + 0.7 * quality
        else:
            entry["base_score"] = 0.5 * identity + 0.5 * quality
        recall.debug_info.tag_purity_debug.append(entry)

    if stage3_debug:
        print(f"[Stage3] 双闸门汇总 输入={len(raw_candidates)} 通过identity={n_identity_ok} 通过final_score={n_final_ok} 输出词数={len(score_map)}")
    return score_map, term_map, idf_map


def run_stage3(
    recall,
    raw_candidates: List[Dict[str, Any]],
    query_vector,
    anchor_vids=None,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float]]:
    """
    阶段 3：词权重。
    若候选含 term_role/identity_score（Stage2 新格式），走双闸门路径；否则走原有 _calculate_final_weights。
    """
    if not raw_candidates:
        return {}, {}, {}

    use_dual_gate = (
        raw_candidates
        and ("term_role" in raw_candidates[0] or "identity_score" in raw_candidates[0])
    )
    if use_dual_gate:
        score_map, term_map, idf_map = _run_stage3_dual_gate(
            recall, raw_candidates, query_vector, anchor_vids=anchor_vids
        )
    else:
        score_map, term_map, idf_map = recall._calculate_final_weights(
            raw_candidates, query_vector, anchor_vids=anchor_vids
        )

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

    return score_map, term_map, idf_map

