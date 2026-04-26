import collections
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.recall.label_pipeline.stage4_paper_recall import TERM_MAX_AUTHOR_SHARE
from src.utils.time_features import (
    compute_author_time_features,
    compute_author_recency_by_latest,
)
from src.utils.tools import get_decay_rate_for_domains as _get_decay_rate_for_domains
from src.core.recall.works_to_authors import accumulate_author_scores
from src.core.recall.label_means import paper_scoring


def _label_recall_stdout_enabled(recall: Any) -> bool:
    """标签路仅在 LabelRecallPath.verbose 且非 silent 时向 stdout 打印 Stage5 审计表。"""
    if getattr(recall, "silent", False):
        return False
    return bool(getattr(recall, "verbose", False))


# True：打印 [Stage5 support dominance audit]；**默认 False**（主轴 paper_primary 主导时噪音大，可疑 support 再 env 打开）
STAGE5_SUPPORT_DOMINANCE_AUDIT = os.environ.get("STAGE5_SUPPORT_DOMINANCE_AUDIT", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
# True：打印 [Stage5 support-only author penalty audit]（纯 support 贡献且无 primary 论文托底 → 作者分乘子）
STAGE5_SUPPORT_ONLY_AUTHOR_PENALTY_ENABLED = True
STAGE5_SUPPORT_ONLY_PENALTY_AUDIT = True
# 条件：sup_only_papers≥1 ∧ sup_with_pri_papers==0 ∧ top_pri_c≈0；不动词级分、不碰 Stage3
SUPPORT_ONLY_TOP_PRI_EPS = 1e-12
SUPPORT_ONLY_SUP_SHARE_STRONG = 0.85
SUPPORT_ONLY_SUP_SHARE_MILD = 0.65
SUPPORT_ONLY_PENALTY_STRONG = 0.62
SUPPORT_ONLY_PENALTY_MILD = 0.78
# True：打印 [Stage5 term-cap audit]，确认单 term 作者占比上限是否作用在最终聚合前
STAGE5_TERM_CAP_AUDIT = True
# True：打印 [Stage5 author structure audit]（base×结构乘子→after，含分项 mult）
STAGE5_AUTHOR_STRUCTURE_AUDIT = True
# 单篇论文「作者 fan-out」：进入 accumulate_author_scores 前轻缩 p["score"]，削弱一篇拖出一串作者（与作者内递减 / term cap / 结构乘子正交叠加）。
PAPER_AUTHOR_FANOUT_PENALTY_ENABLED = True
PAPER_AUTHOR_FANOUT_SOFT_K = 4.0
PAPER_AUTHOR_FANOUT_MIN_FACTOR = 0.72
PAPER_AUTHOR_FANOUT_MAX_COUNT = 12
STAGE5_FANOUT_AUDIT = True
# True：打印 single-paper side-driven suppression **audit**（窄门识别；实际乘子见下方 penalty）
STAGE5_SINGLE_PAPER_SIDE_DRIVEN_SUPPRESSION_AUDIT = True
# True：对「单篇 + side-driven + 无主线托底」作者在 support-only 之后乘温和因子（不改论文分、不碰 Stage4）
STAGE5_SINGLE_PAPER_SIDE_DRIVEN_PENALTY_ENABLED = True
STAGE5_SINGLE_PAPER_SIDE_DRIVEN_PENALTY_AUDIT = True
# fringe 顶篇更强；非 primary 且非 fringe 时略轻（与 Stage4 角色一致）
STAGE5_SINGLE_PAPER_SIDE_DRIVEN_STRONG_PENALTY = 0.73
STAGE5_SINGLE_PAPER_SIDE_DRIVEN_MILD_PENALTY = 0.86
# True：打印 [Stage5 broad-mainline dominance audit]（宽主词/umbrella 独占；与 side-driven 正交；不改分）
STAGE5_BROAD_MAINLINE_DOMINANCE_AUDIT = os.environ.get("STAGE5_BROAD_MAINLINE_DOMINANCE_AUDIT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
# True：在 fanout 之后、accumulate_author_scores 之前，对多作者论文按图 pos_weight 排序叠加调和 Zipf（1,1/2,… 归一）× 图权重；全相等则等权 fallback（不猜顺序）
STAGE5_AUTHORSHIP_WEIGHTING_ENABLED = True
STAGE5_AUTHORSHIP_WEIGHTING_AUDIT = True
STAGE5_AUTHORSHIP_WEIGHTING_MAX_PREVIEW = 4

# 从 Stage4 透传的论文旁路字段，供 suppression audit 读取（不改 Stage4，仅 Stage5 合并时保留）
_STAGE5_PAPER_AUDIT_COPY_KEYS = (
    "hit_quality_class",
    "paper_evidence_role",
    "paper_evidence_quality_score",
    "paper_final_score_v2",
    "paper_old_score",
    "paper_score_final",
    "paper_score_protected",
    "job_axis_labels",
    "job_axis_primary_label",
    "job_axis_audit_reasons",
)

# 与 Stage4 job-axis audit 对齐：厚子轴（用于识别「多轴强作者」）
_STAGE5_THICK_JOB_AXES = frozenset(
    {
        "dynamics_kinematics",
        "planning_trajectory",
        "optimal_control",
        "estimation",
        "simulation_sim2real",
    }
)
STAGE5_BROAD_MAINLINE_DOM_SHARE_MIN = float(os.environ.get("STAGE5_BROAD_MAINLINE_DOM_SHARE_MIN", "0.85"))
STAGE5_BROAD_MAINLINE_SECOND_TERM_SHARE_MIN = float(os.environ.get("STAGE5_BROAD_MAINLINE_SECOND_TERM_SHARE_MIN", "0.18"))
STAGE5_BROAD_MAINLINE_MHR_EXCLUDE_MIN = float(os.environ.get("STAGE5_BROAD_MAINLINE_MHR_EXCLUDE_MIN", "0.42"))

# 薄候选：放宽「顶篇相对全局 max 论文分」门槛，避免极少作者/极少论文时全员被 AUTHOR_BEST_PAPER_MIN_RATIO 滤空（仅触发于小池，非全局放水）
STAGE5_THIN_POOL_MAX_AUTHORS = int(os.environ.get("STAGE5_THIN_POOL_MAX_AUTHORS", "3"))
STAGE5_THIN_POOL_MAX_PAPERS = int(os.environ.get("STAGE5_THIN_POOL_MAX_PAPERS", "4"))
STAGE5_THIN_BEST_PAPER_RATIO_SCALE = float(os.environ.get("STAGE5_THIN_BEST_PAPER_RATIO_SCALE", "0.58"))
STAGE5_THIN_BEST_PAPER_RATIO_FLOOR = float(os.environ.get("STAGE5_THIN_BEST_PAPER_RATIO_FLOOR", "0.014"))
# 统一 LTR-lite 最终作者重排头：线性组合 Stage5 聚合分 + Stage4 author_rerank + 结构/轴多样性 + 审计命中（不经训练器）
STAGE5_UNIFIED_LTR_LITE_ENABLED = os.environ.get("STAGE5_UNIFIED_LTR_LITE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
# 供稿链断点审计：写入 last_debug_info["stage5_supply_chain_audit"]，默认关闭（仅排查时开）
STAGE5_SUPPLY_CHAIN_AUDIT = os.environ.get("STAGE5_SUPPLY_CHAIN_AUDIT", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

_STAGE5_MAINLINE_HIT_QUALITY = frozenset(
    {"mainline_resonance", "mainline_plus_support", "single_hit_mainline"}
)


def _stage5_harmonic_zipf_fracs(n: int) -> List[float]:
    """调和型 Zipf：第 k 位比例 ∝ 1/k，归一化使分量和为 1（再与图 pos_weight 相乘）。"""
    if n <= 0:
        return []
    raw = [1.0 / float(k) for k in range(1, n + 1)]
    s = sum(raw)
    if s <= 1e-18:
        return [1.0 / float(n)] * n
    return [x / s for x in raw]


def _stage5_apply_authorship_zipf_weights_to_papers(
    papers_for_agg: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    在 fanout 已作用到 p['score'] 之后，仅改写各 author 的 pos_weight（不改论文分、不改 Stage4）。
    - 图关系 r.pos_weight 在 Stage4 已写入 paper.weight → 此处读 a['pos_weight']。
    - 多作者且图权重不全相同：按 (pos_weight, aid) 升序视为署名序，叠加调和 Zipf 后与图权重相乘。
    - 多作者且图权重全相同：fallback_equal，不叠加 Zipf（避免在顺序不明时误伤）。
    """
    by_wid: Dict[str, Any] = {}
    ctr = collections.Counter()
    tail_ratios: List[float] = []

    for p in papers_for_agg or []:
        wid = str(p.get("wid") or "")
        authors = p.get("authors") or []
        n = len(authors)
        if n == 0:
            ctr["empty"] += 1
            p["authorship_weight_mode"] = "empty"
            by_wid[wid] = {"mode": "empty", "authors_n": 0}
            continue
        if n == 1:
            ctr["single_author"] += 1
            a0 = authors[0]
            gw0 = max(0.0, float(a0.get("pos_weight") or 1.0))
            p["authorship_weight_mode"] = "single_author"
            by_wid[wid] = {
                "mode": "single_author",
                "authors_n": 1,
                "paper_score_after_fanout": float(p.get("score") or 0.0),
                "fanout_factor": float(p.get("fanout_factor") or 1.0),
                "preview": [(str(a0.get("aid")), gw0, 1.0, gw0)],
            }
            continue

        graph_ws = [max(0.0, float(x.get("pos_weight") or 1.0)) for x in authors]
        if all(g <= 1e-18 for g in graph_ws):
            graph_ws = [1.0] * n

        rounded = {round(g, 6) for g in graph_ws}
        if len(rounded) == 1:
            ctr["fallback_equal"] += 1
            p["authorship_weight_mode"] = "fallback_equal"
            by_wid[wid] = {
                "mode": "fallback_equal",
                "authors_n": n,
                "reason": "all_graph_pos_weight_equal",
                "paper_score_after_fanout": float(p.get("score") or 0.0),
                "fanout_factor": float(p.get("fanout_factor") or 1.0),
            }
            continue

        idx_order = sorted(range(n), key=lambda i: (graph_ws[i], str(authors[i].get("aid") or "")))
        z_fracs = _stage5_harmonic_zipf_fracs(n)
        new_w = [0.0] * n
        for rank, idx in enumerate(idx_order):
            new_w[idx] = max(1e-18, graph_ws[idx] * z_fracs[rank])

        for i, auth in enumerate(authors):
            auth["pos_weight_before_authorship"] = graph_ws[i]
            auth["pos_weight"] = new_w[i]

        tot_sw = sum(new_w)
        shares = [x / tot_sw for x in new_w]
        first_s = shares[idx_order[0]]
        last_s = shares[idx_order[-1]]
        if first_s > 1e-12:
            tail_ratios.append(last_s / first_s)

        ctr["zipf_ranked"] += 1
        p["authorship_weight_mode"] = "zipf_ranked"

        preview: List[Dict[str, Any]] = []
        for rank in range(min(STAGE5_AUTHORSHIP_WEIGHTING_MAX_PREVIEW, n)):
            idx = idx_order[rank]
            preview.append(
                {
                    "aid": str(authors[idx].get("aid")),
                    "graph_pos_weight": graph_ws[idx],
                    "zipf_frac_of_rank": z_fracs[rank],
                    "combined_pos_weight": new_w[idx],
                    "share_of_paper": shares[idx],
                }
            )

        by_wid[wid] = {
            "mode": "zipf_ranked",
            "authors_n": n,
            "paper_score_after_fanout": float(p.get("score") or 0.0),
            "score_before_fanout": float(p.get("score_before_fanout") or 0.0),
            "fanout_factor": float(p.get("fanout_factor") or 1.0),
            "tail_to_head_share_ratio": (last_s / first_s) if first_s > 1e-12 else None,
            "preview": preview,
        }

    out: Dict[str, Any] = {
        "by_wid": by_wid,
        "counters": dict(ctr),
        "tail_to_head_share_ratio_mean": float(np.mean(tail_ratios)) if tail_ratios else None,
        "tail_to_head_share_ratio_median": float(np.median(tail_ratios)) if tail_ratios else None,
    }
    return out


def _print_stage5_authorship_weighting_audit(
    papers_for_agg: List[Dict[str, Any]],
    stats: Dict[str, Any],
    recall: Any,
    detail_limit: int = 14,
) -> None:
    if not _label_recall_stdout_enabled(recall) or not STAGE5_AUTHORSHIP_WEIGHTING_AUDIT:
        return
    print("\n" + "-" * 80)
    print("[Stage5 authorship Zipf-style weighting audit]")
    print("-" * 80)
    if not papers_for_agg:
        print("papers=0")
        print("-" * 80 + "\n")
        return
    ctr = (stats or {}).get("counters") or {}
    print(
        f"papers_total={len(papers_for_agg)} "
        f"zipf_ranked={ctr.get('zipf_ranked', 0)} "
        f"fallback_equal={ctr.get('fallback_equal', 0)} "
        f"single_author={ctr.get('single_author', 0)} "
        f"empty_authors={ctr.get('empty', 0)}"
    )
    tr_mean = (stats or {}).get("tail_to_head_share_ratio_mean")
    tr_med = (stats or {}).get("tail_to_head_share_ratio_median")
    if tr_mean is not None:
        print(
            f"multi_author_zipf tail/head_share_ratio mean={tr_mean:.4f} median={tr_med:.4f} "
            "(last_author_share / first_author_share; lower = thinner tail)"
        )
    multi = [p for p in papers_for_agg if len(p.get("authors") or []) > 1]
    multi_sorted = sorted(multi, key=lambda x: len(x.get("authors") or []), reverse=True)[
        : max(detail_limit, 1)
    ]
    print(f"--- sample papers (multi-author first, max {detail_limit}) ---")
    print("wid | authors_n | mode | score_post_fanout | fanout_f | head/tail share ratio | preview(aid:share)")
    for p in multi_sorted:
        wid = str(p.get("wid") or "")
        au = p.get("authors") or []
        m = len(au)
        mode = str(p.get("authorship_weight_mode") or "?")
        s_pf = float(p.get("score") or 0.0)
        ff = float(p.get("fanout_factor") or 1.0)
        meta = ((stats or {}).get("by_wid") or {}).get(wid) or {}
        thr = meta.get("tail_to_head_share_ratio")
        pr = meta.get("preview") or []
        pr_s = []
        for row in pr[:3]:
            pr_s.append(
                f"{row.get('aid')!s}:{float(row.get('share_of_paper') or 0.0):.2f}"
            )
        print(
            f"{wid[:18]:18} | {m:^9} | {mode:16} | {s_pf:17.6f} | {ff:8.3f} | "
            f"{str(thr) if thr is not None else 'n/a':^20} | {', '.join(pr_s)}"
        )
    print("-" * 80 + "\n")


def _stage5_count_primary_evidence_papers(papers: List[Dict[str, Any]]) -> int:
    n = 0
    for p in papers or []:
        if not isinstance(p, dict):
            continue
        if str(p.get("paper_evidence_role") or "").strip() == "primary_evidence":
            n += 1
    return int(n)


def _stage5_top_paper_dict_for_audit(
    aid: str,
    author_record: Optional[Dict[str, Any]],
    paper_map: Dict[str, Dict[str, Any]],
    author_top_works: Dict[str, List[Tuple[str, float]]],
) -> Dict[str, Any]:
    papers = (author_record or {}).get("papers") or []
    plist = [p for p in papers if isinstance(p, dict)]
    if plist:

        def _paper_score(p: Dict[str, Any]) -> float:
            try:
                if p.get("paper_score_final") is not None:
                    return float(p.get("paper_score_final") or 0.0)
                return float(p.get("score") or 0.0)
            except (TypeError, ValueError):
                return 0.0

        top = max(plist, key=_paper_score)
        return dict(top)
    wl = author_top_works.get(aid) or author_top_works.get(str(aid)) or []
    if not wl:
        return {}
    wid = max(wl, key=lambda x: float(x[1] or 0.0))[0]
    wid_s = str(wid)
    return dict(paper_map.get(wid_s) or paper_map.get(wid) or {})


def _stage5_is_single_paper_side_driven_author(
    aid: str,
    author_record: Optional[Dict[str, Any]],
    scored_row: Dict[str, Any],
    paper_map: Dict[str, Dict[str, Any]],
    author_top_works: Dict[str, List[Tuple[str, float]]],
    term_cap_audit: Dict[str, Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    """
    极窄 suppression audit：单篇 + side hit + 无主轴占比 + 词权极端集中 + 无 primary evidence paper。
    仅用于日志与旁路字段，不参与排序乘子。
    """
    aid_s = str(aid)
    summary = (author_record or {}).get("author_payload_summary") if author_record else None
    summary = summary if isinstance(summary, dict) else {}

    top_paper = _stage5_top_paper_dict_for_audit(aid_s, author_record, paper_map, author_top_works)
    top_hqc = summary.get("top_paper_hit_quality_class")
    if top_hqc is None or str(top_hqc).strip() == "":
        top_hqc = str(top_paper.get("hit_quality_class") or "").strip() or None

    pc = (
        int(summary["paper_count"])
        if summary.get("paper_count") is not None
        else int(scored_row.get("paper_count") or 0)
    )

    if summary.get("mainline_support_ratio") is not None:
        mhr = float(summary["mainline_support_ratio"])
    else:
        if pc == 1 and top_hqc:
            mhr = 1.0 if top_hqc in _STAGE5_MAINLINE_HIT_QUALITY else 0.0
        else:
            mhr = 0.0

    dom_share_raw = summary.get("dominant_term_share")
    dom_f: Optional[float] = None
    if dom_share_raw is not None:
        try:
            dom_f = float(dom_share_raw)
        except (TypeError, ValueError):
            dom_f = None
    if dom_f is None:
        tc = term_cap_audit.get(aid_s) or term_cap_audit.get(aid)
        if tc:
            try:
                dom_f = float(tc.get("dominant_share_before") or 0.0)
            except (TypeError, ValueError):
                dom_f = None

    papers_for_pri = (author_record or {}).get("papers") or []
    pri_ct = _stage5_count_primary_evidence_papers(papers_for_pri)
    if pri_ct == 0 and top_paper:
        pri_ct = _stage5_count_primary_evidence_papers([top_paper])

    top_role = str(top_paper.get("paper_evidence_role") or "").strip()
    old_v = top_paper.get("paper_old_score")
    v2_v = top_paper.get("paper_final_score_v2")
    try:
        old_f = float(old_v) if old_v is not None else None
    except (TypeError, ValueError):
        old_f = None
    try:
        v2_f = float(v2_v) if v2_v is not None else None
    except (TypeError, ValueError):
        v2_f = None

    sup_cls = str(summary.get("author_support_class") or "").strip()
    reason = str(summary.get("author_reason_summary") or "").strip()

    meta: Dict[str, Any] = {
        "author_id": aid_s,
        "paper_count": pc,
        "mainline_support_ratio": mhr,
        "top_paper_hit_quality_class": top_hqc,
        "dominant_term_share": dom_f,
        "author_support_class": sup_cls or None,
        "author_reason_summary": reason or None,
        "author_primary_evidence_count": pri_ct,
        "author_top_paper_evidence_role": top_role or None,
        "author_top_paper_old_score": old_f,
        "author_top_paper_v2_score": v2_f,
        "audit_fail": None,
    }

    if pc != 1:
        meta["audit_fail"] = "paper_count!=1"
        return False, meta
    if top_hqc != "single_hit_side":
        meta["audit_fail"] = "top_paper_hit_quality_class"
        return False, meta
    if mhr > 0.05:
        meta["audit_fail"] = "mainline_support_ratio"
        return False, meta
    if dom_f is None or dom_f < 0.9:
        meta["audit_fail"] = "dominant_term_share"
        return False, meta
    if pri_ct > 0:
        meta["audit_fail"] = "author_primary_evidence_count"
        return False, meta
    if top_role == "primary_evidence":
        meta["audit_fail"] = "top_paper_primary_evidence"
        return False, meta

    return True, meta


def _stage5_single_paper_side_driven_penalty_factor(
    aid: str,
    author_record: Optional[Dict[str, Any]],
    scored_row: Dict[str, Any],
    paper_map: Dict[str, Dict[str, Any]],
    author_top_works: Dict[str, List[Tuple[str, float]]],
    term_cap_audit: Dict[str, Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """
    极窄门：在既有 suppression audit 命中前提下，再要求 Stage4 侧叙事为「side-driven」单篇托底；
    按顶篇 paper_evidence_role 分两档温和乘子（0.72 / 0.85），不误伤 single-paper mainline。
    """
    out: Dict[str, Any] = {"penalty_status": "disabled", "author_id": str(aid)}
    if not STAGE5_SINGLE_PAPER_SIDE_DRIVEN_PENALTY_ENABLED:
        return 1.0, out

    hit, meta = _stage5_is_single_paper_side_driven_author(
        aid, author_record, scored_row, paper_map, author_top_works, term_cap_audit
    )
    out = {**meta, "penalty_status": "no_penalty"}

    summary = (author_record or {}).get("author_payload_summary") if author_record else None
    summary = summary if isinstance(summary, dict) else {}

    if not hit:
        out["penalty_status"] = "audit_not_hit"
        return 1.0, out

    sup_cls = str(summary.get("author_support_class") or "").strip()
    if sup_cls != "single_paper_supported":
        out["penalty_status"] = "skip_support_class"
        out["author_support_class_seen"] = sup_cls or None
        return 1.0, out

    reason_s = str(summary.get("author_reason_summary") or "").lower()
    ml_ct = int(summary.get("mainline_paper_count") or 0)
    so_ct = int(summary.get("side_only_paper_count") or 0)
    side_driven_narrative = (
        "side-driven" in reason_s
        or "side driven" in reason_s
        or (so_ct >= 1 and ml_ct == 0)
    )
    if not side_driven_narrative:
        out["penalty_status"] = "skip_not_side_driven_narrative"
        out["author_reason_summary"] = summary.get("author_reason_summary")
        return 1.0, out

    top_role = str(meta.get("author_top_paper_evidence_role") or "").strip()
    if top_role == "primary_evidence":
        out["penalty_status"] = "skip_primary_evidence"
        return 1.0, out

    if top_role == "fringe_evidence":
        factor = float(STAGE5_SINGLE_PAPER_SIDE_DRIVEN_STRONG_PENALTY)
        tier = "strong_fringe_evidence"
    else:
        factor = float(STAGE5_SINGLE_PAPER_SIDE_DRIVEN_MILD_PENALTY)
        tier = "mild_non_primary"

    trigger_reasons = [
        "narrow_gate:single_paper_side_driven_suppression_audit_hit",
        "author_support_class=single_paper_supported",
        "side_driven_narrative_ok",
        f"penalty_tier={tier}",
        f"top_paper_evidence_role={top_role}",
    ]
    out.update(
        {
            "penalty_status": "applied",
            "penalty_factor": factor,
            "penalty_tier": tier,
            "trigger_reasons": trigger_reasons,
            "top_paper_hit_quality_class": meta.get("top_paper_hit_quality_class"),
            "top_paper_evidence_role": top_role,
            "dominant_term_share": meta.get("dominant_term_share"),
            "mainline_support_ratio": meta.get("mainline_support_ratio"),
            "author_primary_evidence_count": meta.get("author_primary_evidence_count"),
        }
    )
    return factor, out


def _print_stage5_single_paper_side_driven_penalty_audit(
    penalty_by_aid: Dict[str, Dict[str, Any]],
    author_record_by_aid: Dict[str, Dict[str, Any]],
    recall: Any,
) -> None:
    if not _label_recall_stdout_enabled(recall) or not STAGE5_SINGLE_PAPER_SIDE_DRIVEN_PENALTY_AUDIT:
        return
    rows = [
        v
        for v in (penalty_by_aid or {}).values()
        if isinstance(v, dict) and v.get("penalty_applied")
    ]
    if not rows:
        print("\n" + "-" * 80)
        print("[Stage5 single-paper side-driven author penalty audit]")
        print("-" * 80)
        print("penalty_applied_count=0")
        print("-" * 80 + "\n")
        return

    print("\n" + "-" * 80)
    print("[Stage5 single-paper side-driven author penalty audit]")
    print("-" * 80)
    print(f"penalty_applied_count={len(rows)}")
    print(
        "author_id | author_name | original_score | penalty_factor | score_after_penalty | "
        "top_hqc | top_evidence_role | dom_share | mhr | pri_ct | reasons"
    )
    print("-" * 80)
    for r in sorted(rows, key=lambda x: float(x.get("original_score") or 0.0), reverse=True):
        aid = str(r.get("author_id") or "")
        ar = author_record_by_aid.get(aid) or {}
        name = (
            (ar.get("name") if isinstance(ar.get("name"), str) else None)
            or (ar.get("author_name") if isinstance(ar.get("author_name"), str) else None)
            or ""
        )
        rs = r.get("trigger_reasons")
        rs_s = "; ".join(str(x) for x in (rs or []) if x)[:220]
        print(
            f"{aid:16} | {str(name)[:24]:24} | "
            f"{float(r.get('original_score') or 0.0):14.6f} | "
            f"{float(r.get('penalty_factor') or 1.0):14.4f} | "
            f"{float(r.get('score_after_penalty') or 0.0):19.6f} | "
            f"{str(r.get('top_paper_hit_quality_class') or '')[:16]} | "
            f"{str(r.get('top_paper_evidence_role') or '')[:18]} | "
            f"{r.get('dominant_term_share')} | "
            f"{r.get('mainline_support_ratio')} | "
            f"{r.get('author_primary_evidence_count')} | "
            f"{rs_s!r}"
        )
    print("-" * 80 + "\n")


def _stage5_apply_suppression_audit_fields(
    scored_authors: List[Dict[str, Any]],
    author_record_by_aid: Dict[str, Dict[str, Any]],
    paper_map: Dict[str, Dict[str, Any]],
    author_top_works: Dict[str, List[Tuple[str, float]]],
    term_cap_audit: Dict[str, Dict[str, Any]],
) -> None:
    for row in scored_authors:
        aid = str(row.get("aid") or "")
        ar = author_record_by_aid.get(aid)
        hit, meta = _stage5_is_single_paper_side_driven_author(
            aid, ar, row, paper_map, author_top_works, term_cap_audit
        )
        row["author_suppression_audit_hit"] = bool(hit)
        row["author_suppression_audit_reasons"] = dict(meta)
        row["author_primary_evidence_count"] = meta.get("author_primary_evidence_count")
        row["author_top_paper_evidence_role"] = meta.get("author_top_paper_evidence_role")
        row["author_top_paper_old_score"] = meta.get("author_top_paper_old_score")
        row["author_top_paper_v2_score"] = meta.get("author_top_paper_v2_score")


def _stage5_author_suppression_audit_compact_row(
    author_rec: Dict[str, Any], audit_meta: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "author_id": audit_meta.get("author_id") or author_rec.get("aid"),
        "paper_count": audit_meta.get("paper_count"),
        "mainline_support_ratio": audit_meta.get("mainline_support_ratio"),
        "dominant_term": (author_rec.get("author_payload_summary") or {}).get("dominant_term")
        if isinstance(author_rec.get("author_payload_summary"), dict)
        else None,
        "dominant_term_share": audit_meta.get("dominant_term_share"),
        "top_paper_hit_quality_class": audit_meta.get("top_paper_hit_quality_class"),
        "author_support_class": audit_meta.get("author_support_class"),
        "author_reason_summary": audit_meta.get("author_reason_summary"),
        "author_top_paper_evidence_role": audit_meta.get("author_top_paper_evidence_role"),
        "author_suppression_audit_reasons": audit_meta,
    }


def _print_stage5_single_paper_side_driven_suppression_audit(
    scored_authors: List[Dict[str, Any]],
    author_record_by_aid: Dict[str, Dict[str, Any]],
    recall: Any,
) -> None:
    if not _label_recall_stdout_enabled(recall) or not STAGE5_SINGLE_PAPER_SIDE_DRIVEN_SUPPRESSION_AUDIT:
        return
    if not scored_authors:
        return
    n = len(scored_authors)
    hits = [r for r in scored_authors if r.get("author_suppression_audit_hit")]
    hit_n = len(hits)
    ratio = float(hit_n) / float(n) if n else 0.0

    sup_ctr: collections.Counter = collections.Counter()
    hqc_ctr: collections.Counter = collections.Counter()
    for r in scored_authors:
        aid = str(r.get("aid") or "")
        s = (author_record_by_aid.get(aid) or {}).get("author_payload_summary") or {}
        if isinstance(s, dict):
            sup_ctr[str(s.get("author_support_class") or "unknown")] += 1
            hqc_ctr[str(s.get("top_paper_hit_quality_class") or "unknown")] += 1

    print("\n" + "-" * 80)
    print("[Stage5 single-paper side-driven suppression audit]")
    print("-" * 80)
    print(
        f"authors_total={n} audit_hit_count={hit_n} audit_hit_ratio={ratio:.4f} "
        f"support_class_counter={dict(sup_ctr)} top_paper_hit_quality_counter={dict(hqc_ctr)}"
    )
    print("--- audit hit examples (max 10) ---")
    for r in hits[:10]:
        aid = str(r.get("aid") or "")
        ar = author_record_by_aid.get(aid) or {}
        meta = r.get("author_suppression_audit_reasons") or {}
        print(f"  {_stage5_author_suppression_audit_compact_row(ar, meta)}")
    print("-" * 80 + "\n")


def _print_stage5_non_hit_good_author_examples(
    scored_authors: List[Dict[str, Any]],
    author_record_by_aid: Dict[str, Dict[str, Any]],
    recall: Any,
) -> None:
    if not _label_recall_stdout_enabled(recall) or not STAGE5_SINGLE_PAPER_SIDE_DRIVEN_SUPPRESSION_AUDIT:
        return
    if not scored_authors:
        return
    good: List[Dict[str, Any]] = []
    good_ids: Set[str] = set()
    for r in scored_authors:
        if r.get("author_suppression_audit_hit"):
            continue
        aid = str(r.get("aid") or "")
        s = (author_record_by_aid.get(aid) or {}).get("author_payload_summary") or {}
        if not isinstance(s, dict):
            continue
        pc = int(s.get("paper_count") or 0)
        mhr = float(s.get("mainline_support_ratio") or 0.0)
        thqc = str(s.get("top_paper_hit_quality_class") or "")
        if pc > 1 or mhr > 0.05 or thqc in _STAGE5_MAINLINE_HIT_QUALITY:
            good.append(r)
            good_ids.add(aid)
        if len(good) >= 5:
            break
    # 若不足 5，用未命中的高分作者补齐
    if len(good) < 5:
        for r in scored_authors:
            if r.get("author_suppression_audit_hit"):
                continue
            aid = str(r.get("aid") or "")
            if aid in good_ids:
                continue
            good.append(r)
            good_ids.add(aid)
            if len(good) >= 5:
                break

    print("-" * 80)
    print("[Stage5 non-hit good author examples]")
    print("-" * 80)
    print(f"showing={len(good)} (max 5)")
    for r in good[:5]:
        aid = str(r.get("aid") or "")
        s = (author_record_by_aid.get(aid) or {}).get("author_payload_summary") or {}
        if not isinstance(s, dict):
            s = {}
        row = {
            "author_id": aid,
            "paper_count": s.get("paper_count"),
            "mainline_support_ratio": s.get("mainline_support_ratio"),
            "dominant_term_share": s.get("dominant_term_share"),
            "top_paper_hit_quality_class": s.get("top_paper_hit_quality_class"),
            "author_support_class": s.get("author_support_class"),
            "author_reason_summary": s.get("author_reason_summary"),
        }
        print(f"  {row}")
    print("-" * 80 + "\n")


def _stage5_collect_axis_label_union_for_author(
    aid: str,
    author_record: Optional[Dict[str, Any]],
    paper_map: Dict[str, Dict[str, Any]],
    author_top_works: Dict[str, List[Tuple[str, float]]],
) -> Set[str]:
    wids: Set[str] = set()
    if author_record:
        for p in author_record.get("papers") or []:
            if isinstance(p, dict) and p.get("wid") is not None:
                wids.add(str(p.get("wid")))
    for wid, _ in author_top_works.get(aid, []) or []:
        wids.add(str(wid))
    for wid, _ in author_top_works.get(str(aid), []) or []:
        wids.add(str(wid))
    out: Set[str] = set()
    for wid in wids:
        info = paper_map.get(wid) or paper_map.get(str(wid)) or {}
        labs = info.get("job_axis_labels")
        if isinstance(labs, list):
            out |= {str(x) for x in labs}
    return out


def _stage5_second_term_share_in_top3(top_terms: List[Tuple[str, Any]]) -> float:
    """top_terms_by_contribution 中第二词占前三项贡献和的比例（结构信号，非词表）。"""
    if not top_terms or len(top_terms) < 2:
        return 0.0
    vals: List[float] = []
    for _t, c in top_terms[:3]:
        try:
            vals.append(float(c))
        except (TypeError, ValueError):
            vals.append(0.0)
    if len(vals) < 2:
        return 0.0
    s = sum(max(0.0, v) for v in vals) + 1e-12
    return float(max(0.0, vals[1]) / s)


def _stage5_author_has_multi_axis_strength(
    axis_union: Set[str],
    scored_row: Dict[str, Any],
    summary: Dict[str, Any],
) -> bool:
    """
    排除 genuinely multi-axis authors：厚轴组合、标签多样性、结构 multi-hit、第二条强词、或较高 mainline 占比。
    不依赖具体 JD 词表。
    """
    ls = {str(x) for x in axis_union if str(x) != "unclassified"}
    thick = ls & _STAGE5_THICK_JOB_AXES
    if len(thick) >= 2:
        return True
    if len(thick) >= 1 and len(ls) >= 2:
        return True

    tt = scored_row.get("top_terms_by_contribution") or []
    if isinstance(tt, list) and _stage5_second_term_share_in_top3(tt) >= STAGE5_BROAD_MAINLINE_SECOND_TERM_SHARE_MIN:
        return True

    try:
        mhr = float(summary.get("mainline_support_ratio") or 0.0)
    except (TypeError, ValueError):
        mhr = 0.0
    if mhr >= STAGE5_BROAD_MAINLINE_MHR_EXCLUDE_MIN:
        return True

    mtp = int(scored_row.get("multi_term_paper_count_struct") or 0)
    stc = int(scored_row.get("strong_term_count_struct") or 0)
    if mtp >= 1 and stc >= 2:
        return True

    return False


def _stage5_top_paper_is_broad_mainline_shell(
    top_paper: Dict[str, Any],
) -> bool:
    """
    top paper「像宽壳/薄主轴」：无厚轴标签时以 mainline 质量 + 角色兜底；有标签则要求无 THICK 轴。
    """
    if not top_paper:
        return False
    labs = top_paper.get("job_axis_labels")
    if isinstance(labs, list) and labs:
        ls = {str(x) for x in labs} - {"unclassified"}
        if not ls:
            return True
        if ls & _STAGE5_THICK_JOB_AXES:
            return False
        if ls <= {"control_core", "generic_robot_autonomous_shell"}:
            return True
        if "generic_robot_autonomous_shell" in ls:
            return True
        return False
    hqc = str(top_paper.get("hit_quality_class") or "").strip()
    if hqc in _STAGE5_MAINLINE_HIT_QUALITY:
        return True
    return False


def _stage5_is_broad_mainline_dominance_author(
    aid: str,
    author_record: Optional[Dict[str, Any]],
    scored_row: Dict[str, Any],
    paper_map: Dict[str, Dict[str, Any]],
    author_top_works: Dict[str, List[Tuple[str, float]]],
    term_cap_audit: Dict[str, Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    """
    Very narrow：单 term 支配度高 + 全作者层面无多轴强支撑 + top paper 像宽壳/薄主轴；
    与 single-paper side-driven 互斥（由调用方排除 suppression hit）。
    """
    aid_s = str(aid)
    summary = (author_record or {}).get("author_payload_summary") if author_record else None
    summary = summary if isinstance(summary, dict) else {}

    dom_share_raw = summary.get("dominant_term_share")
    dom_f: Optional[float] = None
    if dom_share_raw is not None:
        try:
            dom_f = float(dom_share_raw)
        except (TypeError, ValueError):
            dom_f = None
    if dom_f is None:
        tc = term_cap_audit.get(aid_s) or term_cap_audit.get(aid)
        if tc:
            try:
                dom_f = float(tc.get("dominant_share_before") or 0.0)
            except (TypeError, ValueError):
                dom_f = None

    top_paper = _stage5_top_paper_dict_for_audit(aid_s, author_record, paper_map, author_top_works)
    top_hqc = str(top_paper.get("hit_quality_class") or "").strip()

    axis_union = _stage5_collect_axis_label_union_for_author(aid_s, author_record, paper_map, author_top_works)
    multi_strong = _stage5_author_has_multi_axis_strength(axis_union, scored_row, summary)

    signals: Dict[str, Any] = {
        "dominant_term_share": dom_f,
        "dominant_term": summary.get("dominant_term"),
        "axis_union": sorted(axis_union),
        "multi_axis_strength_gate": bool(multi_strong),
        "top_paper_hit_quality_class": top_hqc or None,
        "top_paper_evidence_role": str(top_paper.get("paper_evidence_role") or "").strip() or None,
        "top_paper_job_axis_labels": top_paper.get("job_axis_labels"),
        "broad_shell_top_paper": bool(_stage5_top_paper_is_broad_mainline_shell(top_paper)),
    }

    if dom_f is None or dom_f < STAGE5_BROAD_MAINLINE_DOM_SHARE_MIN:
        return False, {**signals, "reason": "dom_share_below_threshold"}

    if multi_strong:
        return False, {**signals, "reason": "multi_axis_strength_excluded"}

    if top_hqc == "single_hit_side":
        return False, {**signals, "reason": "top_paper_single_hit_side_excluded"}

    if not _stage5_top_paper_is_broad_mainline_shell(top_paper):
        return False, {**signals, "reason": "top_paper_not_broad_shell_profile"}

    return True, {**signals, "reason": "broad_mainline_dominance_hit"}


def _stage5_apply_broad_mainline_dominance_audit_fields(
    scored_authors: List[Dict[str, Any]],
    author_record_by_aid: Dict[str, Dict[str, Any]],
    paper_map: Dict[str, Dict[str, Any]],
    author_top_works: Dict[str, List[Tuple[str, float]]],
    term_cap_audit: Dict[str, Dict[str, Any]],
) -> None:
    for row in scored_authors:
        aid = str(row.get("aid") or "")
        ar = author_record_by_aid.get(aid)
        if row.get("author_suppression_audit_hit"):
            row["broad_mainline_dominance_hit"] = False
            row["broad_mainline_dominance_reason"] = "skipped_overlaps_side_driven_suppression_audit"
            row["broad_mainline_dominance_signals"] = {}
            continue
        hit, sig = _stage5_is_broad_mainline_dominance_author(
            aid, ar, row, paper_map, author_top_works, term_cap_audit
        )
        row["broad_mainline_dominance_hit"] = bool(hit)
        row["broad_mainline_dominance_reason"] = str(sig.get("reason") or "")
        row["broad_mainline_dominance_signals"] = dict(sig)
        row["broad_mainline_dominant_term"] = (sig.get("dominant_term") if isinstance(sig, dict) else None) or (
            (ar or {}).get("author_payload_summary") or {}
        ).get("dominant_term")
        row["broad_mainline_axis_profile"] = sig.get("axis_union")
        row["broad_mainline_has_second_strong_axis"] = bool(sig.get("multi_axis_strength_gate"))


def _print_stage5_broad_mainline_dominance_audit(
    scored_authors: List[Dict[str, Any]],
    author_record_by_aid: Dict[str, Dict[str, Any]],
    paper_map: Dict[str, Dict[str, Any]],
    author_top_works: Dict[str, List[Tuple[str, float]]],
    recall: Any,
) -> None:
    if not _label_recall_stdout_enabled(recall) or not STAGE5_BROAD_MAINLINE_DOMINANCE_AUDIT:
        return
    if not scored_authors:
        return
    n = len(scored_authors)
    hits = [r for r in scored_authors if r.get("broad_mainline_dominance_hit")]
    hit_n = len(hits)
    ratio = float(hit_n) / float(n) if n else 0.0

    print("\n" + "-" * 80)
    print("[Stage5 broad-mainline dominance audit]")
    print("-" * 80)
    print(f"authors_total={n} audit_hit_count={hit_n} audit_hit_ratio={ratio:.4f}")
    print("--- audit hit rows (max 12, by final score) ---")
    hits_sorted = sorted(hits, key=lambda r: float(r.get("score") or 0.0), reverse=True)[:12]
    for r in hits_sorted:
        aid = str(r.get("aid") or "")
        ar = author_record_by_aid.get(aid) or {}
        s = ar.get("author_payload_summary") or {}
        if not isinstance(s, dict):
            s = {}
        top_paper = _stage5_top_paper_dict_for_audit(aid, ar, paper_map, author_top_works)
        sig = r.get("broad_mainline_dominance_signals") or {}
        print(
            f"  aid={aid!r} final_score={float(r.get('score') or 0.0):.6f} "
            f"dominant_term={s.get('dominant_term')!r} dominant_term_share={s.get('dominant_term_share')!r} "
            f"paper_count={s.get('paper_count')!r}"
        )
        pri = _stage5_count_primary_evidence_papers((ar.get("papers") or []))
        print(
            f"    primary_evidence_paper_count={pri} "
            f"top_paper_role={str(top_paper.get('paper_evidence_role') or '')!r} "
            f"top_paper_hqc={str(top_paper.get('hit_quality_class') or '')!r} "
            f"title={str(top_paper.get('title') or '')[:100]!r}"
        )
        print(
            f"    top_paper_job_axis_labels={top_paper.get('job_axis_labels')!r} "
            f"signals={sig!r}"
        )
    print("--- non-hit good author examples (multi-axis / not umbrella-dominated, max 5) ---")
    good: List[Dict[str, Any]] = []
    for r in sorted(scored_authors, key=lambda x: float(x.get("score") or 0.0), reverse=True):
        if r.get("broad_mainline_dominance_hit"):
            continue
        if r.get("author_suppression_audit_hit"):
            continue
        aid = str(r.get("aid") or "")
        ar = author_record_by_aid.get(aid) or {}
        au = _stage5_collect_axis_label_union_for_author(aid, ar, paper_map, author_top_works)
        summ = ar.get("author_payload_summary") or {}
        if not isinstance(summ, dict):
            summ = {}
        if _stage5_author_has_multi_axis_strength(au, r, summ):
            good.append(r)
        if len(good) >= 5:
            break
    if len(good) < 5:
        for r in sorted(scored_authors, key=lambda x: float(x.get("score") or 0.0), reverse=True):
            if r.get("broad_mainline_dominance_hit") or r in good:
                continue
            if r.get("author_suppression_audit_hit"):
                continue
            good.append(r)
            if len(good) >= 5:
                break
    for r in good[:5]:
        aid = str(r.get("aid") or "")
        ar = author_record_by_aid.get(aid) or {}
        summ = ar.get("author_payload_summary") or {}
        if not isinstance(summ, dict):
            summ = {}
        au = _stage5_collect_axis_label_union_for_author(aid, ar, paper_map, author_top_works)
        print(
            f"  aid={aid!r} final_score={float(r.get('score') or 0.0):.6f} "
            f"dominant_term_share={summ.get('dominant_term_share')!r} "
            f"axis_union={sorted(au)!r} multi_axis_strength={_stage5_author_has_multi_axis_strength(au, r, summ)}"
        )
    print("-" * 80 + "\n")


def _try_hit_terms_wid(
    paper_hit_terms: Dict[str, List[str]], wid: Any
) -> List[str]:
    if wid is None:
        return []
    h = paper_hit_terms.get(wid)
    if h is not None:
        return h
    return paper_hit_terms.get(str(wid), []) or []


def _print_stage5_paper_fanout_audit(
    papers_for_agg: List[Dict[str, Any]],
    paper_hit_terms: Dict[str, List[str]],
    top_n: int = 20,
    recall: Any = None,
) -> None:
    if not _label_recall_stdout_enabled(recall):
        return
    if not STAGE5_FANOUT_AUDIT or not papers_for_agg:
        return
    print("\n[Stage5 paper fanout audit]")
    print("wid | score_before | authors_n | fanout_factor | score_after | hit_terms")
    rows = sorted(
        papers_for_agg,
        key=lambda x: float(
            x.get("score_before_fanout") if x.get("score_before_fanout") is not None else x.get("score") or 0.0
        ),
        reverse=True,
    )[:top_n]
    for p in rows:
        wid = p.get("wid")
        authors = p.get("authors") or []
        ht = _try_hit_terms_wid(paper_hit_terms, wid)
        print(
            f"{wid} | "
            f"{float(p.get('score_before_fanout') or 0.0):.4f} | "
            f"{len(authors):^3} | "
            f"{float(p.get('fanout_factor') or 1.0):.3f} | "
            f"{float(p.get('score') or 0.0):.4f} | "
            f"{ht[:4]!r}"
        )


def _apply_term_max_author_share_cap(
    author_term_contrib_raw: Dict[str, Dict[str, float]],
    max_share: float,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]]]:
    """
    单 term 对作者总贡献占比上限（与 stage4_paper_recall.TERM_MAX_AUTHOR_SHARE 同源）。

    对每个 author：先求各 term 的原始贡献和 raw_total，再令每项 capped_t = min(raw_t, max_share * raw_total)，
    重算 capped_total。仅缩放分值，不改变「单 term 100% dominant_share」形状；打散单词单篇作者依赖后续 **结构乘子**（`_compute_author_structure_shape`）。
    """
    capped_out: Dict[str, Dict[str, float]] = {}
    audit: Dict[str, Dict[str, Any]] = {}
    cap = float(max_share)
    for aid, atc in (author_term_contrib_raw or {}).items():
        raw_terms = {str(t): float(v) for t, v in (atc or {}).items() if float(v) > 1e-18}
        total_raw = float(sum(raw_terms.values()))
        if total_raw <= 1e-18:
            capped_out[str(aid)] = {}
            continue
        limit = cap * total_raw if cap < 1.0 - 1e-12 else total_raw
        capped_terms = {t: min(v, limit) for t, v in raw_terms.items()}
        total_capped = float(sum(capped_terms.values()))
        dom_tid_raw, v_raw = max(raw_terms.items(), key=lambda x: x[1])
        dom_tid_cap, v_cap = (
            max(capped_terms.items(), key=lambda x: x[1]) if capped_terms else ("", 0.0)
        )
        audit[str(aid)] = {
            "raw_total": total_raw,
            "capped_total": total_capped,
            "dominant_term_id": dom_tid_raw,
            "dominant_term_raw": float(v_raw),
            "dominant_term_after_cap": float(v_cap),
            "dominant_share_before": float(v_raw / total_raw) if total_raw > 1e-18 else 0.0,
            "dominant_share_after": float(v_cap / total_capped) if total_capped > 1e-18 else 0.0,
        }
        capped_out[str(aid)] = capped_terms
    return capped_out, audit


def _wid_nonzero_term_counts(papers_for_agg: List[Dict[str, Any]]) -> Dict[str, int]:
    """每篇论文 term_weights 中非零项个数，用于 multi-hit（≥2 term）计数。"""
    out: Dict[str, int] = {}
    for p in papers_for_agg or []:
        wid = str(p.get("wid") or "")
        if not wid:
            continue
        tw = p.get("term_weights") or {}
        nz = sum(1 for w in tw.values() if float(w or 0.0) > 1e-9)
        out[wid] = nz
    return out


# 相对作者自身 max(term_contrib) 的「强词」比例；0.35 比 0.2 更易析出 st≥2，避免全员 st=1 导致乘子无区分度
_STRONG_REL_TO_MAX = 0.35


def _compute_author_structure_shape(
    atc: Dict[str, float],
    works: List[Tuple[str, float]],
    wid_n_terms: Dict[str, int],
    paper_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    作者层结构乘子（批注）：**分立 structure_factor** 先把「单词单篇、无 multi-hit」与「多强词 / multi-term 论文」拉开，
    再乘较轻的 **term_strength_mult × paper_evidence_mult × multi_hit_mult**，避免多乘子叠回去把 singleton 作者抬回前排。
    **st** = 贡献 ≥ 0.35·max(term_contrib) 的强词个数；**pc** / **mtp** / **single_term_papers** 见保留论文与 wid_n_terms。
    无词表、无 RL/robot 特判。
    """
    vals = [float(v) for v in (atc or {}).values() if float(v) > 1e-18]
    max_tc = max(vals) if vals else 0.0
    thr_t = _STRONG_REL_TO_MAX * max_tc
    strong_term_count = sum(1 for v in vals if v >= thr_t) if max_tc > 1e-18 else 0

    wid_piece: Dict[str, float] = {}
    for wid, piece in works or []:
        w = str(wid)
        wid_piece[w] = max(wid_piece.get(w, 0.0), float(piece))
    pcs = list(wid_piece.values())
    max_pc = max(pcs) if pcs else 0.0
    thr_p = _STRONG_REL_TO_MAX * max_pc
    strong_paper_count = sum(1 for pc in pcs if pc >= thr_p) if max_pc > 1e-18 else 0

    paper_count = len(wid_piece)
    mtp = int(sum(1 for wid in wid_piece if int(wid_n_terms.get(wid, 0) or 0) >= 2))
    single_term_papers = int(
        sum(1 for wid in wid_piece if int(wid_n_terms.get(wid, 0) or 0) < 2)
    )

    st = int(strong_term_count)
    sp = int(strong_paper_count)
    pc = int(paper_count)

    # 分立结构惩奖：单 term + 单篇 + 无 multi-hit → 更强压；单 term + 少量篇仍无 mtp → 中压；
    # 有 multi-term 论文或多强词 → 奖励区（略收敛 bonus 斜率，避免与平滑三项叠乘过度放大）。
    if st <= 1 and pc <= 1 and mtp == 0:
        structure_factor = 0.42
    elif st <= 1 and pc <= 2 and mtp == 0:
        structure_factor = 0.58
    elif st <= 1 and mtp == 0:
        structure_factor = 0.72
    elif mtp >= 1 or st >= 2:
        structure_factor = 1.00 + 0.06 * min(mtp, 3)
    else:
        structure_factor = 0.88

    term_strength_mult = 0.90 + 0.10 * min(st, 3) / 3.0
    paper_evidence_mult = 0.88 + 0.12 * min(pc, 4) / 4.0
    multi_hit_mult = 1.00 + 0.08 * min(mtp, 3)
    structure_mult_total = float(
        structure_factor * term_strength_mult * paper_evidence_mult * multi_hit_mult
    )

    best_paper_hit_terms_count = 0
    if wid_piece:
        best_wid = max(wid_piece.items(), key=lambda x: x[1])[0]
        meta = paper_map.get(best_wid) or paper_map.get(str(best_wid)) or {}
        tw = meta.get("term_weights") or {}
        if isinstance(tw, dict) and tw:
            best_paper_hit_terms_count = sum(
                1 for w in tw.values() if float(w or 0.0) > 1e-9
            )
        elif isinstance(meta.get("hits"), list):
            best_paper_hit_terms_count = len(
                [h for h in (meta.get("hits") or []) if h is not None]
            )

    out: Dict[str, Any] = {
        "strong_term_count": st,
        "strong_paper_count": sp,
        "multi_term_paper_count": mtp,
        "paper_count": pc,
        "single_term_papers": single_term_papers,
        "best_paper_hit_terms_count": int(best_paper_hit_terms_count),
        "structure_factor": float(structure_factor),
        "term_strength_mult": float(term_strength_mult),
        "paper_evidence_mult": float(paper_evidence_mult),
        "multi_hit_mult": float(multi_hit_mult),
        "structure_mult_total": float(structure_mult_total),
    }
    return out


def _print_stage5_term_cap_audit(
    scored_authors: List[Dict[str, Any]],
    term_cap_audit: Dict[str, Dict[str, Any]],
    term_map: Dict[str, str],
    top_k: int = 25,
    recall: Any = None,
) -> None:
    """一眼确认 TERM_MAX_AUTHOR_SHARE 是否参与重算作者 term 矩阵，而非仅写常量。"""
    if not _label_recall_stdout_enabled(recall):
        return
    if not STAGE5_TERM_CAP_AUDIT or not scored_authors:
        return
    print("\n" + "-" * 80)
    print(
        f"[Stage5 term-cap & structure audit] TERM_MAX_AUTHOR_SHARE={TERM_MAX_AUTHOR_SHARE} | "
        "author_id | raw_total | capped_total | st | st_papr | mtp | struct_f | mult_tot | "
        "dominant_term | dom_share_before | dom_share_after"
    )
    print("-" * 80)
    for row in scored_authors[:top_k]:
        aid = str(row.get("aid") or "")
        rec = term_cap_audit.get(aid) or {}
        dom_tid = str(rec.get("dominant_term_id") or "")
        dom_name = term_map.get(dom_tid) or ""
        if not dom_name and dom_tid.isdigit():
            dom_name = term_map.get(str(int(dom_tid))) or ""
        dom_name = (dom_name or dom_tid)[:22]
        print(
            f"{aid:16} | {float(rec.get('raw_total') or 0.0):9.4f} | {float(rec.get('capped_total') or 0.0):9.4f} | "
            f"{int(rec.get('strong_term_count') or 0):^2} | {int(rec.get('strong_paper_count') or 0):^7} | "
            f"{int(rec.get('multi_term_paper_count') or 0):^3} | "
            f"{float(rec.get('structure_factor') or 1.0):8.3f} | {float(rec.get('structure_mult_total') or 1.0):8.3f} | "
            f"{dom_name!r} | {float(rec.get('dominant_share_before') or 0.0):8.2%} | "
            f"{float(rec.get('dominant_share_after') or 0.0):8.2%}"
        )


def _print_stage5_author_structure_audit(
    scored_authors: List[Dict[str, Any]],
    author_structure_audit: Dict[str, Dict[str, Any]],
    top_k: int = 25,
    recall: Any = None,
) -> None:
    """
    结构乘子施加前后：base_score（×时间权重后、结构前）、st/pc/mtp、单篇论文数、最佳篇命中词数、
    struct_f 与三项平滑 mult、struct_tot、after。用于判断 Top 是否仍被「单词单篇」统治。
    """
    if not _label_recall_stdout_enabled(recall):
        return
    if not STAGE5_AUTHOR_STRUCTURE_AUDIT or not scored_authors:
        return
    print("\n" + "-" * 80)
    print(
        "[Stage5 author structure audit] "
        "author_id | base_score | st | pc | mtp | st_papr | singl_papr | best_hit | "
        "struct_f | t_mult | p_mult | m_mult | struct_tot | after"
    )
    print("-" * 80)
    for row in scored_authors[:top_k]:
        aid = str(row.get("aid") or "")
        r = author_structure_audit.get(aid) or {}
        print(
            f"{aid:16} | {float(r.get('final_before_structure') or 0.0):10.4f} | "
            f"{int(r.get('strong_term_count') or 0):^2} | {int(r.get('paper_count') or 0):^2} | "
            f"{int(r.get('multi_term_paper_count') or 0):^3} | "
            f"{int(r.get('strong_paper_count') or 0):^7} | {int(r.get('single_term_papers') or 0):^8} | "
            f"{int(r.get('best_paper_hit_terms_count') or 0):^8} | "
            f"{float(r.get('structure_factor') or 1.0):8.3f} | "
            f"{float(r.get('term_strength_mult') or 1.0):6.3f} | "
            f"{float(r.get('paper_evidence_mult') or 1.0):6.3f} | "
            f"{float(r.get('multi_hit_mult') or 1.0):6.3f} | "
            f"{float(r.get('structure_mult_total') or 1.0):10.4f} | "
            f"{float(r.get('final_after_structure') or 0.0):10.4f}"
        )


def _is_primary_supported(primary_count: int, supporting_count: int) -> bool:
    """护栏 5 条件：≥1 个 primary 或 ≥2 个 primary/supporting 为 primary_supported。"""
    if primary_count < 0 and supporting_count < 0:
        return True
    return primary_count >= 1 or (primary_count + supporting_count) >= 2


def aggregate_author_evidence_by_term_role(
    papers_for_agg: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    按作者区分 primary_supported 与 expansion_supported 的证据，便于可解释。
    返回: aid -> {
      "primary_supported_score": float,
      "primary_supported_wids": List[str],
      "expansion_supported_score": float,
      "expansion_supported_wids": List[str],
    }
    """
    out: Dict[str, Dict[str, Any]] = {}
    for p in papers_for_agg or []:
        primary_count = p.get("primary_count", -1)
        supporting_count = p.get("supporting_count", -1)
        is_primary = _is_primary_supported(primary_count, supporting_count)
        score = float(p.get("score") or 0.0)
        wid = p.get("wid")
        for author in p.get("authors") or []:
            aid = author.get("aid") if isinstance(author, dict) else author
            if aid is None or aid == "":
                continue
            if aid not in out:
                out[aid] = {
                    "primary_supported_score": 0.0,
                    "primary_supported_wids": [],
                    "expansion_supported_score": 0.0,
                    "expansion_supported_wids": [],
                }
            if is_primary:
                out[aid]["primary_supported_score"] += score
                if wid is not None:
                    out[aid]["primary_supported_wids"].append(wid)
            else:
                out[aid]["expansion_supported_score"] += score
                if wid is not None:
                    out[aid]["expansion_supported_wids"].append(wid)
    return out


def _compute_author_support_only_metrics(
    aid: str,
    author_term_contrib: Dict[str, Dict[str, float]],
    author_all_retained_works: Dict[str, List[Tuple[str, float]]],
    paper_map: Dict[str, Dict[str, Any]],
    debug_1: Dict[str, Any],
) -> Dict[str, Any]:
    """
    与 [Stage5 support dominance audit] 同源分解：term 侧 primary/support 份额 + 论文侧是否只有 support 命中。
    供 support-only 作者惩罚与审计复用（避免两处规则漂移）。
    """
    atc = author_term_contrib.get(aid, {}) or {}
    total = sum(float(v) for v in atc.values()) or 1e-12
    pri_vals: List[float] = []
    sup_vals: List[float] = []
    for tid_s, c in atc.items():
        c = float(c)
        if c <= 0:
            continue
        rr = _retrieval_role_for_paper_term(debug_1, str(tid_s))
        if rr == "paper_support":
            sup_vals.append(c)
        else:
            pri_vals.append(c)
    top_pri_c = max(pri_vals) if pri_vals else 0.0
    top_sup_c = max(sup_vals) if sup_vals else 0.0
    sup_sum = sum(sup_vals)
    sup_share = float(sup_sum / total)

    sup_only_papers = 0
    sup_with_pri_papers = 0
    for wid, piece in author_all_retained_works.get(aid, []):
        if float(piece or 0.0) <= 0:
            continue
        info = paper_map.get(str(wid)) or paper_map.get(wid) or {}
        hits = info.get("hits") or []
        has_p = any(
            (h.get("role") or "").strip().lower() == "paper_primary"
            for h in hits
            if isinstance(h, dict)
        )
        has_s = any(
            (h.get("role") or "").strip().lower() == "paper_support"
            for h in hits
            if isinstance(h, dict)
        )
        if has_p:
            sup_with_pri_papers += 1
        elif has_s:
            sup_only_papers += 1

    return {
        "sup_share": sup_share,
        "top_pri_c": float(top_pri_c),
        "top_sup_c": float(top_sup_c),
        "sup_only_papers": int(sup_only_papers),
        "sup_with_pri_papers": int(sup_with_pri_papers),
    }


def _support_only_author_penalty_value(m: Dict[str, Any]) -> float:
    """论文级：仅有 support-only 篇、无 primary 篇；词级：几乎无 primary 贡献。否则 1.0。"""
    if int(m.get("sup_only_papers") or 0) < 1:
        return 1.0
    if int(m.get("sup_with_pri_papers") or 0) != 0:
        return 1.0
    if float(m.get("top_pri_c") or 0.0) > SUPPORT_ONLY_TOP_PRI_EPS:
        return 1.0
    sup_share = float(m.get("sup_share") or 0.0)
    if sup_share >= SUPPORT_ONLY_SUP_SHARE_STRONG:
        return float(SUPPORT_ONLY_PENALTY_STRONG)
    if sup_share >= SUPPORT_ONLY_SUP_SHARE_MILD:
        return float(SUPPORT_ONLY_PENALTY_MILD)
    return 1.0


def _print_stage5_support_only_penalty_audit(
    rows: List[Dict[str, Any]], top_k: int = 25, recall: Any = None
) -> None:
    if not _label_recall_stdout_enabled(recall):
        return
    if not STAGE5_SUPPORT_ONLY_PENALTY_AUDIT or not rows:
        return
    print("\n" + "-" * 80)
    print(
        "[Stage5 support-only author penalty audit] "
        "author_id | raw_score | sup_share | top_pri_c | sup_only_papers | "
        "sup_with_pri_papers | penalty | after"
    )
    print("-" * 80)
    sorted_rows = sorted(rows, key=lambda r: float(r.get("raw_score") or 0.0), reverse=True)[:top_k]
    for r in sorted_rows:
        print(
            f"{str(r.get('aid')):16} | {float(r.get('raw_score') or 0.0):11.4f} | "
            f"{float(r.get('sup_share') or 0.0):8.3f} | {float(r.get('top_pri_c') or 0.0):9.4f} | "
            f"{int(r.get('sup_only_papers') or 0):^15} | {int(r.get('sup_with_pri_papers') or 0):^19} | "
            f"{float(r.get('penalty') or 1.0):7.3f} | {float(r.get('after') or 0.0):11.4f}"
        )


def _retrieval_role_for_paper_term(debug_1: Dict[str, Any], tid_s: str) -> str:
    """Stage3 入 paper 的 retrieval_role（paper_primary / paper_support）；缺省偏保守记 primary。"""
    tr = debug_1.get("term_retrieval_roles") or {}
    v = tr.get(int(tid_s)) if tid_s.isdigit() else None
    if v is None:
        v = tr.get(tid_s)
    if v is None:
        tm = debug_1.get("term_paper_meta") or {}
        meta = tm.get(int(tid_s)) if tid_s.isdigit() else None
        if meta is None:
            meta = tm.get(tid_s)
        if isinstance(meta, dict):
            v = meta.get("retrieval_role")
    s = (str(v or "paper_primary")).strip().lower()
    return s if s in ("paper_primary", "paper_support") else "paper_primary"


def _print_stage5_support_dominance_audit(
    scored_authors: List[Dict[str, Any]],
    author_term_contrib: Dict[str, Dict[str, float]],
    author_all_retained_works: Dict[str, List[Tuple[str, float]]],
    paper_map: Dict[str, Dict[str, Any]],
    term_map: Dict[str, str],
    debug_1: Dict[str, Any],
    top_k: int = 25,
    recall: Any = None,
) -> None:
    """
    看 Top 作者：总分里 primary vs support 分解、dominant term 角色、论文侧是否「独狼 support」结构。
    term 贡献来自 author_term_contrib（同词递减后、已做 TERM_MAX_AUTHOR_SHARE 截断、未乘时间权重）；final_score 为之后管线分。
    """
    if not _label_recall_stdout_enabled(recall):
        return
    if not STAGE5_SUPPORT_DOMINANCE_AUDIT or not scored_authors:
        return
    print("\n" + "-" * 80)
    print(
        "[Stage5 support dominance audit] "
        "author_id | final_score | top_pri_c | top_sup_c | sup_share | "
        "sup_only_papers | sup+w_pri_papers | dominant_term | dom_role"
    )
    print("-" * 80)
    for row in scored_authors[:top_k]:
        aid = str(row.get("aid") or "")
        final_score = float(row.get("score") or 0.0)
        atc = author_term_contrib.get(aid, {}) or {}
        m = _compute_author_support_only_metrics(
            aid, author_term_contrib, author_all_retained_works, paper_map, debug_1
        )
        top_pri = float(m["top_pri_c"])
        top_sup = float(m["top_sup_c"])
        sup_share = float(m["sup_share"])
        sup_only = int(m["sup_only_papers"])
        sup_with = int(m["sup_with_pri_papers"])
        dom_tid, dom_c = max(atc.items(), key=lambda x: float(x[1])) if atc else ("", 0.0)
        dom_term = term_map.get(str(dom_tid), str(dom_tid))[:32]
        dom_role = _retrieval_role_for_paper_term(debug_1, str(dom_tid))
        print(
            f"{aid:16} | {final_score:11.4f} | {top_pri:9.4f} | {top_sup:9.4f} | {sup_share:8.3f} | "
            f"{sup_only:^15} | {sup_with:^18} | {dom_term!r} | {dom_role}"
        )


# 已实现：时间权重 × structure_mult_total（structure_factor × term/paper/mhit 平滑项，见 _compute_author_structure_shape）。
# 预留：HierarchyConsistency、按 term_family 的 FamilyBalancePenalty。


def _stage5_effective_best_paper_min_ratio(recall: Any, n_authors: int, n_papers: int) -> float:
    base = float(getattr(recall, "AUTHOR_BEST_PAPER_MIN_RATIO", 0.05) or 0.05)
    thin = (int(n_authors) <= STAGE5_THIN_POOL_MAX_AUTHORS) or (int(n_papers) <= STAGE5_THIN_POOL_MAX_PAPERS)
    if not thin:
        return base
    return max(STAGE5_THIN_BEST_PAPER_RATIO_FLOOR, base * STAGE5_THIN_BEST_PAPER_RATIO_SCALE)


def _stage5_axis_label_diversity(author_rec: Optional[Dict[str, Any]]) -> int:
    """作者 payload 上 job_axis_labels 去重计数，作多轴覆盖 proxy（与 export 侧 axis_coverage 一致来源）。"""
    seen: Set[str] = set()
    for p in (author_rec or {}).get("papers") or []:
        if not isinstance(p, dict):
            continue
        labs = p.get("job_axis_labels")
        if isinstance(labs, list):
            for x in labs:
                sx = str(x).strip()
                if sx:
                    seen.add(sx)
    return len(seen)


def _stage5_apply_unified_ltr_lite_final_rerank(
    scored_authors: List[Dict[str, Any]],
    author_record_by_aid: Dict[str, Dict[str, Any]],
) -> None:
    """
    在 suppression / broad-mainline 等 **审计字段已就绪** 之后，对同一 JD 候选作者做一次可解释线性融合并重排。
    - 保留原聚合分为主信号；并入 Stage4 author_rerank_score（min-max 于本批候选内）。
    - 轻量奖励：mainline 占比、论文数、多轴、primary 证据、hq multi-hit、结构乘子区间。
    - 轻量惩罚：suppression audit 与 broad-mainline dominance audit（仅排序层，不回头改论文分）。
    写出 author_stage5_unified_score 与 stage5_pre_unified_score，供导出与训练样本使用。
    """
    if not scored_authors or not STAGE5_UNIFIED_LTR_LITE_ENABLED:
        return

    s4_vals: List[float] = []
    for row in scored_authors:
        aid = str(row.get("aid") or "")
        ar = author_record_by_aid.get(aid) or {}
        s4_vals.append(float(ar.get("author_rerank_score") or ar.get("final_score_reranked") or 0.0))
    mn_s4 = min(s4_vals)
    mx_s4 = max(s4_vals)

    raw_list: List[float] = []
    for row in scored_authors:
        aid = str(row.get("aid") or "")
        ar = author_record_by_aid.get(aid) or {}
        s5 = float(row.get("score") or 0.0)
        row["stage5_pre_unified_score"] = s5

        s4v = float(ar.get("author_rerank_score") or ar.get("final_score_reranked") or 0.0)
        if mx_s4 > mn_s4 + 1e-18:
            s4n = (s4v - mn_s4) / (mx_s4 - mn_s4)
        else:
            s4n = 0.5

        summ = ar.get("author_payload_summary") if isinstance(ar.get("author_payload_summary"), dict) else {}
        mhr = float(summ.get("mainline_support_ratio") or 0.0)
        pc = int(summ.get("paper_count") or row.get("paper_count") or 0)
        hq_m = int(summ.get("high_quality_multi_hit_count") or 0)
        pri = float(row.get("primary_supported_score") or 0.0)
        axis_div = float(_stage5_axis_label_diversity(ar))
        axis_norm = min(1.0, axis_div / 5.0)

        sup_hit = 1.0 if row.get("author_suppression_audit_hit") else 0.0
        broad_hit = 1.0 if row.get("broad_mainline_dominance_hit") else 0.0

        struct_tot = float(row.get("structure_mult_total") or 1.0)
        struct_norm = min(1.0, max(0.0, (struct_tot - 0.42) / 0.70))

        u = (
            0.46 * s5
            + 0.20 * s4n
            + 0.10 * mhr
            + 0.07 * min(1.0, math.log1p(max(0, pc)) / math.log1p(6.0))
            + 0.06 * axis_norm
            + 0.05 * min(1.0, float(hq_m) / 2.0)
            + 0.04 * min(1.0, pri * 3.0)
            + 0.02 * struct_norm
            - 0.08 * sup_hit
            - 0.06 * broad_hit
        )
        raw_list.append(u)
        row["author_stage5_unified_breakdown"] = {
            "s5_pre_unified": round(s5, 6),
            "s4_struct_rerank_norm": round(s4n, 6),
            "mainline_support_ratio": round(mhr, 6),
            "axis_label_n": int(axis_div),
            "hq_multi_hit": int(hq_m),
            "primary_supported_score": round(pri, 6),
            "structure_mult_total": round(struct_tot, 6),
            "suppression_audit": bool(sup_hit > 0.5),
            "broad_mainline_dom_audit": bool(broad_hit > 0.5),
            "unified_raw": round(u, 6),
        }

    mn_u = min(raw_list)
    mx_u = max(raw_list)
    for i, row in enumerate(scored_authors):
        u_raw = raw_list[i]
        if mx_u > mn_u + 1e-18:
            u_fin = (u_raw - mn_u) / (mx_u - mn_u)
        else:
            u_fin = u_raw
        row["author_stage5_unified_raw"] = float(u_raw)
        row["author_stage5_unified_score"] = float(u_fin)
        row["score"] = float(u_fin)
        bd = row.get("author_stage5_unified_breakdown")
        if isinstance(bd, dict):
            bd["unified_final_norm"] = round(float(u_fin), 6)

    scored_authors.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)


def _stage5_infer_supply_chain_first_breakpoint(
    *,
    n_author_papers_records: int,
    n_unique_aids_in: int,
    n_unique_wids_in: int,
    n_paper_map: int,
    n_papers_nonpos: int,
    n_papers_for_agg: int,
    n_papers_for_agg_zero_authors: int,
    n_author_term_aids: int,
    n_author_scores_after_term_recompute: int,
    n_author_scores_before_best_paper_filter: int,
    n_author_scores_after_best_paper_filter: int,
    n_scored_authors_pre_unified: int,
) -> str:
    """
    首个断点编码（字母序越靠前越早）：
    A* 入口 / 合并前 | B merge | C paper 贡献 | D term 矩阵 | E author 总分 | F best-paper 门槛 | G 组装 scored_authors
    """
    if n_author_papers_records <= 0:
        return "A_INPUT_AUTHOR_RECORDS_EMPTY"
    if n_unique_wids_in <= 0:
        return "A_INPUT_NO_WIDS_IN_AUTHOR_RECORDS"
    if n_paper_map <= 0:
        return "B_MERGED_PAPER_MAP_EMPTY"
    if n_papers_for_agg <= 0:
        if n_paper_map > 0 and n_papers_nonpos >= n_paper_map:
            return "C_ALL_PAPERS_DROPPED_NONPOSITIVE_SCORE"
        return "C_PAPERS_FOR_AGG_EMPTY_OTHER"
    if n_papers_for_agg_zero_authors >= n_papers_for_agg > 0:
        return "C_PAPERS_FOR_AGG_HAVE_NO_AUTHOR_EDGES"
    if n_author_term_aids <= 0:
        return "D_NO_AUTHOR_TERM_CONTRIB_NONPOSITIVE_RANK_OR_WEIGHT"
    if n_author_scores_after_term_recompute <= 0:
        return "E_AUTHOR_SCORES_EMPTY_AFTER_TERM_RECOMPUTE"
    if n_author_scores_after_best_paper_filter <= 0 < n_author_scores_before_best_paper_filter:
        return "F_BEST_PAPER_MIN_RATIO_REMOVED_ALL_AUTHORS"
    if n_scored_authors_pre_unified <= 0 < n_author_scores_after_best_paper_filter:
        return "G_SCORED_AUTHORS_EMPTY_BUT_AUTHOR_SCORES_NONEMPTY"
    if n_scored_authors_pre_unified <= 0:
        return "G_SCORED_AUTHORS_EMPTY"
    return "OK"


def run_stage5(
    recall,
    author_papers_list: List[Dict[str, Any]],
    score_map: Dict[str, float],
    term_map: Dict[str, str],
    active_domain_set: Set[int],
    dominance: float,
    debug_1: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    阶段 5：作者打分与排序。论文贡献 → 作者×词矩阵 → 同词递减 → TERM_MAX_AUTHOR_SHARE → 时间权重
    → 结构乘子（分立 structure_factor + st/paper/mtp 三连乘，强词阈 0.35·max）
    → **support-only 作者乘子**（有 **sup_only** 论文、无 **primary 同框** 论文、**top_pri_c≈0** 且 **sup_share** 高时压分）
    → **single-paper side-driven 窄门乘子**（在既有 suppression audit 与 Stage4「side-driven」叙事一致时，对顶篇 fringe 略强、其余非 primary 略轻；与 fanout/term cap/ broad-mainline audit 正交）
    → **authorship Zipf-style pos_weight**（fanout 之后、accumulate 之前：图 pos_weight 不全等时按序叠加调和 1/k×图权；全等则等权 fallback）
    → 新作比 → 归一化排序 → **薄候选下放宽顶篇相对门槛** → **统一 LTR-lite 最终作者重排头**（线性融合 Stage5 分、Stage4 author_rerank、轴/结构/审计）。
    返回 (author_id_list, last_debug_info)。
    """
    _lp_out = bool(not getattr(recall, "silent", False) and getattr(recall, "verbose", False))

    def _p(*args, **kwargs):
        if _lp_out:
            print(*args, **kwargs)

    score_map = score_map or {}
    term_map = term_map or {}
    active_domain_set = active_domain_set or set()
    industrial_kws = debug_1.get("industrial_kws", []) if debug_1 else []
    stage5_sub_ms: Dict[str, float] = {}
    t_stage5 = time.perf_counter()
    if not author_papers_list:
        recall.debug_info.work_count = 0
        recall.debug_info.author_count = 0
        recall.debug_info.recall_vocab_count = len(score_map)
        stage5_sub_ms["total"] = (time.perf_counter() - t_stage5) * 1000.0
        recall.last_debug_info = {
            "active_domains": [str(d) for d in sorted(active_domain_set)],
            "dominance": float(dominance),
            "industrial_kws": industrial_kws,
            "anchor_skills": debug_1.get("anchor_skills", {}) if debug_1 else {},
            "score_map": score_map,
            "term_map": term_map,
            "work_count": 0,
            "author_count": 0,
            "recall_vocab_count": len(score_map),
            "filter_closed_loop": (debug_1 or {}).get("filter_closed_loop") or {},
            "stage5_sub_ms": stage5_sub_ms,
        }
        if debug_1 and debug_1.get("stage1_sub_ms") is not None:
            recall.last_debug_info["stage1_sub_ms"] = debug_1["stage1_sub_ms"]
        recall.last_debug_info["stage5_supply_chain_audit"] = {
            "first_breakpoint": "A_INPUT_AUTHOR_RECORDS_EMPTY",
            "n_author_papers_records": 0,
            "n_unique_aids_in": 0,
            "n_unique_wids_in": 0,
            "n_merged_paper_map": 0,
            "n_papers_nonpos_drop": 0,
            "n_papers_for_agg": 0,
            "n_papers_for_agg_zero_authors": 0,
            "n_author_term_aids": 0,
            "n_author_scores_after_term_recompute": 0,
            "n_author_scores_before_best_paper_filter": 0,
            "n_author_scores_after_best_paper_filter": 0,
            "n_scored_authors_pre_unified": 0,
            "n_best_paper_filter_removed": 0,
        }
        return [], recall.last_debug_info

    industrial_kws = debug_1.get("industrial_kws", [])
    anchor_skills = debug_1.get("anchor_skills", {})
    term_role_map = debug_1.get("term_role_map") or {}
    term_confidence_map = debug_1.get("term_confidence_map") or {}
    term_uniqueness_map = debug_1.get("term_uniqueness_map") or {}
    context = {
        "score_map": score_map,
        "term_map": term_map,
        "term_role_map": term_role_map,
        "term_confidence_map": term_confidence_map,
        "term_uniqueness_map": term_uniqueness_map,
        "term_paper_meta": debug_1.get("term_paper_meta") or {},
        "anchor_kws": [k.lower() for k in industrial_kws],
        "active_domain_set": active_domain_set,
        "dominance": dominance,
        "decay_rate": _get_decay_rate_for_domains(active_domain_set),
        "query_vector": debug_1.get("query_vector"),
    }
    t0 = time.perf_counter()

    author_record_by_aid: Dict[str, Dict[str, Any]] = {
        str(r["aid"]): r for r in author_papers_list if r.get("aid") is not None
    }

    _in_aids: Set[str] = set()
    _in_wids: Set[str] = set()
    for _rec in author_papers_list:
        if _rec.get("aid") is not None:
            _in_aids.add(str(_rec["aid"]))
        for _pp in _rec.get("papers") or []:
            if isinstance(_pp, dict) and _pp.get("wid") is not None:
                _in_wids.add(str(_pp["wid"]))

    paper_map: Dict[str, Dict[str, Any]] = {}
    author_raw_paper_cnt: Dict[str, int] = collections.Counter()

    def _merge_hits(old_hits: List[Any], new_hits: List[Any]) -> List[Any]:
        """
        合并同一论文来自不同路径的 hits。
        - dict hit：按 vid 去重，并保留更高 idf（避免重复累加同 term）。
        - 非 dict hit：按值去重，保序追加。
        """
        merged_dict_hits: Dict[str, Dict[str, Any]] = {}
        merged_other_hits: List[Any] = []
        merged_other_seen: Set[str] = set()

        for hit in (old_hits or []) + (new_hits or []):
            if isinstance(hit, dict):
                vid_s = str(hit.get("vid") or "")
                if not vid_s:
                    key = repr(hit)
                    if key not in merged_other_seen:
                        merged_other_seen.add(key)
                        merged_other_hits.append(hit)
                    continue
                idf_new = float(hit.get("idf") or 0.0)
                prev = merged_dict_hits.get(vid_s)
                if prev is None:
                    merged_dict_hits[vid_s] = dict(hit)
                else:
                    idf_prev = float(prev.get("idf") or 0.0)
                    if idf_new > idf_prev:
                        merged_dict_hits[vid_s] = dict(hit)
            else:
                key = repr(hit)
                if key not in merged_other_seen:
                    merged_other_seen.add(key)
                    merged_other_hits.append(hit)

        return list(merged_dict_hits.values()) + merged_other_hits

    for record in author_papers_list:
        aid = record["aid"]
        for paper in record["papers"]:
            wid = paper["wid"]
            entry = paper_map.get(wid)
            if entry is None:
                entry = {
                    "wid": wid,
                    "hits": list(paper.get("hits") or []),
                    "title": paper["title"],
                    "year": paper["year"],
                    "domains": paper["domains"],
                    "authors": [],
                }
                paper_map[wid] = entry
            else:
                # 批注：同一 wid 可能从不同 term 路径进入，这里必须并集保留多 term 命中证据。
                entry["hits"] = _merge_hits(
                    list(entry.get("hits") or []),
                    list(paper.get("hits") or []),
                )
            entry["authors"].append(
                {
                    "aid": aid,
                    "pos_weight": float(paper.get("weight") or 1.0),
                }
            )
            for _pk in _STAGE5_PAPER_AUDIT_COPY_KEYS:
                if _pk in paper and paper[_pk] is not None:
                    entry[_pk] = paper[_pk]
            author_raw_paper_cnt[aid] += 1

    t1 = time.perf_counter()
    stage5_sub_ms["merge_paper_map"] = (t1 - t0) * 1000.0
    _p("\n[Stage5 merged paper multi-hit audit]")
    multi_hit_rows: List[Tuple[str, str, List[str]]] = []
    for wid, info in paper_map.items():
        hits = info.get("hits") or []
        if len(hits) >= 2:
            multi_hit_rows.append((wid, info.get("title") or "", list(hits)))
    _p(f"multi_hit_papers={len(multi_hit_rows)}")
    for wid, title, hits in multi_hit_rows[:20]:
        _p(f"  wid={wid} hits={hits} title='{title[:100]}'")

    context["_proximity_cache"] = {}

    if os.environ.get("LABEL_PROFILE_STAGE5", "").strip() in ("1", "true", "yes"):
        context["_paper_contrib_prof"] = {}

    papers_for_agg: List[Dict[str, Any]] = []
    paper_hit_terms: Dict[str, List[str]] = {}
    all_works_count = 0
    papers_nonpos_drop = 0

    for wid, info in paper_map.items():
        paper_struct = {
            "wid": wid,
            "hits": info["hits"],
            "title": info["title"],
            "year": info["year"],
            "domains": info["domains"],
        }
        out = paper_scoring.compute_contribution(recall, paper_struct, context)
        p_score, p_hits, p_rank_score, p_term_weights = out[0], out[1], out[2], out[3]
        primary_count = out[4] if len(out) > 4 else -1
        supporting_count = out[5] if len(out) > 5 else -1
        all_works_count += 1
        if p_score <= 0:
            papers_nonpos_drop += 1
            continue
        # 护栏 5：论文进入高优先级候选至少满足其一：≥1 个 primary term，或 ≥2 个 primary/supporting；
        # 纯 expansion 支撑的论文不允许排到 very top（压分）
        if primary_count >= 0 and supporting_count >= 0:
            if primary_count == 0 and supporting_count < 2:
                p_score = float(p_score) * 0.05
        paper_hit_terms[wid] = p_hits
        info["score"] = float(p_score)
        info["rank_score"] = float(p_rank_score or 0)
        info["term_weights"] = dict(p_term_weights or {})
        info["primary_count"] = primary_count
        info["supporting_count"] = supporting_count
        papers_for_agg.append(
            {
                "wid": wid,
                "score": float(p_score),
                "rank_score": float(p_rank_score or 0),
                "term_weights": dict(p_term_weights or {}),
                "authors": info["authors"],
                "primary_count": primary_count,
                "supporting_count": supporting_count,
            }
        )

    n_papers_for_agg_zero_authors = sum(1 for p in papers_for_agg if not (p.get("authors") or []))

    t2 = time.perf_counter()
    stage5_sub_ms["paper_contribution"] = (t2 - t1) * 1000.0
    _p("\n[Stage5 paper term_weights audit]")
    multi_term_weight_rows: List[Tuple[str, float, List[Tuple[str, float]]]] = []
    for p in papers_for_agg:
        tw = p.get("term_weights") or {}
        nz = [(tid, round(float(w), 6)) for tid, w in tw.items() if float(w) > 1e-9]
        if len(nz) >= 2:
            multi_term_weight_rows.append((p["wid"], float(p["score"]), nz[:10]))
    _p(f"multi_term_weight_papers={len(multi_term_weight_rows)}")
    for wid, score, nz in multi_term_weight_rows[:20]:
        _p(f"  wid={wid} score={score:.6f} term_weights={nz}")

    pc_prof = context.pop("_paper_contrib_prof", None)
    if isinstance(pc_prof, dict) and pc_prof:
        stage5_sub_ms["paper_contrib_detail_ms"] = {k: round(float(v), 2) for k, v in sorted(pc_prof.items())}
        _p(
            "[Label S5 paper_scoring 子项累计 ms] "
            + " ".join(f"{k}={round(float(v), 1)}ms" for k, v in sorted(pc_prof.items()))
        )

    if papers_for_agg:
        paper_scores = [p["score"] for p in papers_for_agg]
        try:
            tau = float(np.percentile(paper_scores, 95))
        except Exception:
            tau = 0.0

        if tau > 0:

            def _compress(s: float) -> float:
                return float(tau * math.tanh(s / tau))

            for p in papers_for_agg:
                p["score"] = _compress(float(p["score"]))

    t3 = time.perf_counter()
    stage5_sub_ms["percentile_compress"] = (t3 - t2) * 1000.0

    # fan-out：作者数 >1 时缩论文总分；n≤4 附近接近不压，长作者表有下.floor（见 PAPER_AUTHOR_FANOUT_*）。
    t_fan0 = time.perf_counter()
    for p in papers_for_agg:
        authors = p.get("authors") or []
        n_auth = min(len(authors), int(PAPER_AUTHOR_FANOUT_MAX_COUNT))
        s0 = float(p.get("score") or 0.0)
        p["score_before_fanout"] = s0
        if PAPER_AUTHOR_FANOUT_PENALTY_ENABLED and n_auth > 1:
            denom = max(float(n_auth), float(PAPER_AUTHOR_FANOUT_SOFT_K))
            fanout_factor = max(
                float(PAPER_AUTHOR_FANOUT_MIN_FACTOR),
                math.sqrt(float(PAPER_AUTHOR_FANOUT_SOFT_K) / denom),
            )
        else:
            fanout_factor = 1.0
        p["fanout_factor"] = float(fanout_factor)
        p["score"] = s0 * float(fanout_factor)
    t3_fan = time.perf_counter()
    stage5_sub_ms["author_fanout_penalty"] = (t3_fan - t_fan0) * 1000.0
    _print_stage5_paper_fanout_audit(papers_for_agg, paper_hit_terms, recall=recall)

    authorship_w_stats: Dict[str, Any] = {
        "enabled": bool(STAGE5_AUTHORSHIP_WEIGHTING_ENABLED),
        "skipped": True,
    }
    t_auth0 = time.perf_counter()
    if STAGE5_AUTHORSHIP_WEIGHTING_ENABLED and papers_for_agg:
        authorship_w_stats = _stage5_apply_authorship_zipf_weights_to_papers(papers_for_agg)
        authorship_w_stats["enabled"] = True
        authorship_w_stats["skipped"] = False
        _print_stage5_authorship_weighting_audit(papers_for_agg, authorship_w_stats, recall=recall)
    t_auth1 = time.perf_counter()
    stage5_sub_ms["authorship_zipf_weighting"] = (t_auth1 - t_auth0) * 1000.0

    t_accum0 = time.perf_counter()
    agg_result = accumulate_author_scores(papers_for_agg, top_k_per_author=3)
    author_scores = agg_result.author_scores
    author_top_works = agg_result.author_top_works
    paper_scores_by_wid = {p["wid"]: float(p["score"]) for p in papers_for_agg}
    author_evidence_by_term_role = aggregate_author_evidence_by_term_role(papers_for_agg)

    t3a = time.perf_counter()
    stage5_sub_ms["accumulate_authors"] = (t3a - t_accum0) * 1000.0

    term_paper_contrib: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for p in papers_for_agg:
        wid, s_final, r_score, tw = (
            p["wid"],
            p["score"],
            p.get("rank_score") or 1.0,
            p.get("term_weights") or {},
        )
        if r_score <= 0:
            continue
        for vid_s, w in tw.items():
            term_paper_contrib[vid_s].append((wid, (w / r_score) * s_final))

    t4 = time.perf_counter()
    stage5_sub_ms["term_paper_index"] = (t4 - t3a) * 1000.0

    # -------------------------
    # Stage5 稍稳版修正（核心）：
    # 1) 不再只看 author_top_works，而是从 papers_for_agg 全量构造 author->term hit list
    # 2) 再做“同词递减聚合”，抑制同一 term 多篇论文线性灌榜
    # -------------------------
    author_term_hit_lists: Dict[str, Dict[str, List[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    # 保留作者的全量 retained works，供后续 best_paper/过滤/时间特征复用
    author_all_retained_works: Dict[str, List[Tuple[str, float]]] = collections.defaultdict(list)
    for p in papers_for_agg:
        wid = p["wid"]
        s_final = float(p.get("score") or 0.0)
        r_score = float(p.get("rank_score") or 1.0)
        tw = p.get("term_weights") or {}
        authors = p.get("authors") or []
        if r_score <= 0 or s_final <= 0:
            continue

        for a in authors:
            aid = a.get("aid")
            pos_weight = float(a.get("pos_weight") or 1.0)
            if not aid:
                continue

            # 作者在该论文上的贡献片段（复用既有署名权重）
            author_piece = s_final * pos_weight
            if author_piece > 0:
                author_all_retained_works[str(aid)].append((wid, author_piece))

            # 将论文贡献按 term 权重分摊到 author->term 命中列表
            for vid_s, w in tw.items():
                frac = float(w) / float(r_score)
                term_piece = s_final * pos_weight * frac
                if term_piece <= 0:
                    continue
                author_term_hit_lists[str(aid)][str(vid_s)].append(term_piece)

    # 为兼容后续逻辑，切到 retained works 口径（不再仅 top3）
    author_top_works = author_all_retained_works

    author_term_contrib: Dict[str, Dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    TERM_REPEAT_DECAY = 0.60
    for aid, term_hits in author_term_hit_lists.items():
        for tid_s, hit_list in term_hits.items():
            hit_list = sorted(hit_list, reverse=True)
            agg = 0.0
            for i, x in enumerate(hit_list):
                agg += float(x) * (TERM_REPEAT_DECAY ** i)
            author_term_contrib[aid][tid_s] = agg

    t5a = time.perf_counter()
    stage5_sub_ms["term_contrib_matrix"] = (t5a - t4) * 1000.0

    # 批注：在「同词递减聚合」之后、重建作者总分之前，对每作者施加单 term 贡献占比上限（TERM_MAX_AUTHOR_SHARE），
    # 避免单个泛词/强词独占 author_term_contrib，进而统治最终排序；与 Stage4 词级降压互补、无需领域黑名单。
    author_term_contrib, term_cap_audit_records = _apply_term_max_author_share_cap(
        {str(aid): dict(terms) for aid, terms in author_term_contrib.items()},
        TERM_MAX_AUTHOR_SHARE,
    )

    t5 = time.perf_counter()
    stage5_sub_ms["term_max_author_share_cap"] = (t5 - t5a) * 1000.0

    # 用「占比上限后的」term 贡献重建 author 总分
    recomputed_author_scores: Dict[str, float] = {}
    for aid, atc in author_term_contrib.items():
        recomputed_author_scores[aid] = float(sum(float(v) for v in atc.values()))
    author_scores = recomputed_author_scores
    n_author_term_aids = len(author_term_contrib)
    n_author_scores_after_term_recompute = len(author_scores)

    # 每篇论文上有多少个非零 term（与 Stage4 multi-hit 对齐），供 multi_term_paper_count
    wid_n_terms = _wid_nonzero_term_counts(papers_for_agg)
    author_structure_audit: Dict[str, Dict[str, Any]] = {}
    support_only_penalty_rows: List[Dict[str, Any]] = []
    single_paper_side_penalty_by_aid: Dict[str, Dict[str, Any]] = {}

    if author_scores:
        years_by_author: Dict[str, List[Any]] = {}
        for aid in author_scores.keys():
            years: List[Any] = []
            for wid, _ in author_top_works.get(aid, []):
                meta = paper_map.get(wid, {})
                years.append(meta.get("year"))
            years_by_author[aid] = years

        for aid, base_score in list(author_scores.items()):
            years = years_by_author.get(aid, [])
            activity, momentum, time_weight = compute_author_time_features(years)
            recency_by_latest = compute_author_recency_by_latest(years)
            score = float(base_score) * float(time_weight) * float(recency_by_latest)
            author_scores[aid] = score

        # 批注：结构乘子 = structure_factor（按 st/pc/mtp 分段）× term_strength_mult × paper_evidence_mult × multi_hit_mult。
        for aid in list(author_scores.keys()):
            aid_s = str(aid)
            before_struct = float(author_scores[aid])
            atc_map = {str(k): float(v) for k, v in (author_term_contrib.get(aid, {}) or {}).items()}
            works = author_top_works.get(aid, [])
            shape = _compute_author_structure_shape(atc_map, works, wid_n_terms, paper_map)
            after_struct = before_struct * float(shape["structure_mult_total"])
            author_scores[aid] = after_struct
            out_rec = {
                **shape,
                "final_before_structure": before_struct,
                "final_after_structure": after_struct,
            }
            author_structure_audit[aid_s] = out_rec
            term_cap_audit_records.setdefault(aid_s, {}).update(out_rec)

        # 批注：结构乘子之后、新作比过滤与 max 归一化之前——只削「论文全是 support 线、词贡献几无 primary」的作者，不砍 term 分。
        t_sup_pen0 = time.perf_counter()
        if STAGE5_SUPPORT_ONLY_AUTHOR_PENALTY_ENABLED:
            for aid in list(author_scores.keys()):
                aid_s = str(aid)
                m = _compute_author_support_only_metrics(
                    aid_s,
                    author_term_contrib,
                    author_all_retained_works,
                    paper_map,
                    debug_1 or {},
                )
                pen = _support_only_author_penalty_value(m)
                raw_before = float(author_scores[aid])
                after = raw_before * float(pen)
                author_scores[aid] = after
                support_only_penalty_rows.append(
                    {
                        "aid": aid_s,
                        "raw_score": raw_before,
                        "sup_share": m["sup_share"],
                        "top_pri_c": m["top_pri_c"],
                        "sup_only_papers": m["sup_only_papers"],
                        "sup_with_pri_papers": m["sup_with_pri_papers"],
                        "penalty": pen,
                        "after": after,
                    }
                )
        stage5_sub_ms["support_only_author_penalty"] = (
            (time.perf_counter() - t_sup_pen0) * 1000.0
        )

        t_sp_pen0 = time.perf_counter()
        if STAGE5_SINGLE_PAPER_SIDE_DRIVEN_PENALTY_ENABLED:
            for aid in list(author_scores.keys()):
                aid_s = str(aid)
                ar = author_record_by_aid.get(aid_s)
                pc_fb = int(author_raw_paper_cnt.get(aid, len(author_top_works.get(aid_s, []))))
                scored_stub = {"paper_count": pc_fb}
                factor, pmeta = _stage5_single_paper_side_driven_penalty_factor(
                    aid_s,
                    ar,
                    scored_stub,
                    paper_map,
                    author_top_works,
                    term_cap_audit_records,
                )
                before_sp = float(author_scores[aid])
                if factor < 1.0 - 1e-15:
                    after_sp = before_sp * float(factor)
                    author_scores[aid] = after_sp
                    single_paper_side_penalty_by_aid[aid_s] = {
                        "penalty_applied": True,
                        "author_id": aid_s,
                        "original_score": before_sp,
                        "score_after_penalty": after_sp,
                        "penalty_factor": float(factor),
                        "trigger_reasons": pmeta.get("trigger_reasons") or [],
                        "top_paper_hit_quality_class": pmeta.get("top_paper_hit_quality_class"),
                        "top_paper_evidence_role": pmeta.get("top_paper_evidence_role"),
                        "dominant_term_share": pmeta.get("dominant_term_share"),
                        "mainline_support_ratio": pmeta.get("mainline_support_ratio"),
                        "author_primary_evidence_count": pmeta.get("author_primary_evidence_count"),
                    }
        stage5_sub_ms["single_paper_side_driven_author_penalty"] = (
            (time.perf_counter() - t_sp_pen0) * 1000.0
        )
    else:
        stage5_sub_ms["support_only_author_penalty"] = 0.0
        stage5_sub_ms["single_paper_side_driven_author_penalty"] = 0.0

    _print_stage5_support_only_penalty_audit(support_only_penalty_rows, recall=recall)
    _print_stage5_single_paper_side_driven_penalty_audit(
        single_paper_side_penalty_by_aid,
        author_record_by_aid,
        recall,
    )

    t6 = time.perf_counter()
    stage5_sub_ms["time_and_family"] = (t6 - t5) * 1000.0

    n_author_scores_before_best_paper_filter = len(author_scores)
    n_best_paper_filter_removed = 0
    if papers_for_agg and author_scores and author_top_works:
        paper_scores_by_wid = {p["wid"]: float(p["score"]) for p in papers_for_agg}
        max_paper = max(paper_scores_by_wid.values()) if paper_scores_by_wid else 0.0
        if max_paper > 0:
            eff_ratio = _stage5_effective_best_paper_min_ratio(
                recall, len(author_scores), len(papers_for_agg)
            )
            min_contrib = max_paper * eff_ratio
            to_remove = [
                aid
                for aid in author_scores
                if max(
                    (paper_scores_by_wid.get(wid, 0.0) for wid, _ in author_top_works.get(aid, [])),
                    default=0.0,
                )
                < min_contrib
            ]
            n_best_paper_filter_removed = len(to_remove)
            for aid in to_remove:
                author_scores.pop(aid, None)
    n_author_scores_after_best_paper_filter = len(author_scores)

    if author_scores:
        max_score = max(author_scores.values())
        if max_score > 0:
            for aid in author_scores:
                author_scores[aid] = author_scores[aid] / max_score

    t7 = time.perf_counter()
    stage5_sub_ms["filter_normalize"] = (t7 - t6) * 1000.0

    t8 = time.perf_counter()

    scored_authors: List[Dict[str, Any]] = []
    for aid, total_score in sorted(author_scores.items(), key=lambda x: x[1], reverse=True):
        works = author_top_works.get(aid, [])
        if not works:
            continue

        per_author_papers: List[Dict[str, Any]] = []
        for wid, contrib in works:
            meta = paper_map.get(wid, {})
            hits = paper_hit_terms.get(wid, [])
            per_author_papers.append(
                {
                    "title": meta.get("title"),
                    "year": meta.get("year"),
                    "contribution": round(contrib, 6),
                    "hits": hits,
                }
            )

        per_author_papers.sort(key=lambda x: x.get("contribution", 0.0), reverse=True)
        top_papers = per_author_papers[:3]
        best_paper = top_papers[0] if top_papers else None

        tag_counter = collections.Counter()
        for p in per_author_papers:
            tag_counter.update(p.get("hits") or [])
        tag_stats = [{"term": t, "count": c} for t, c in tag_counter.most_common(10)]

        paper_cnt_author = author_raw_paper_cnt.get(aid, len(per_author_papers))

        final_score = author_scores.get(aid, total_score)
        atc = author_term_contrib.get(aid, {})
        top_terms_contrib = sorted(
            [(term_map.get(tid, ""), round(float(c), 6)) for tid, c in atc.items() if c > 0],
            key=lambda x: -x[1],
        )[:5]

        evidence = author_evidence_by_term_role.get(aid, {})
        _srec = author_structure_audit.get(str(aid), {})
        _sp_pen = single_paper_side_penalty_by_aid.get(str(aid)) or {}
        scored_authors.append(
            {
                "aid": aid,
                "score": final_score,
                "raw_score": total_score,
                "single_paper_side_driven_penalty_factor": float(
                    _sp_pen.get("penalty_factor") or 1.0
                ),
                "score_before_single_paper_side_penalty": _sp_pen.get("original_score"),
                "single_paper_side_driven_penalty_applied": bool(_sp_pen.get("penalty_applied")),
                "single_paper_side_driven_penalty_reasons": _sp_pen.get("trigger_reasons") or [],
                "top_paper": best_paper,
                "paper_count": paper_cnt_author,
                "top_papers": top_papers,
                "tag_stats": tag_stats,
                "top_terms_by_contribution": top_terms_contrib,
                "primary_supported_score": round(evidence.get("primary_supported_score", 0.0), 6),
                "primary_supported_wids": evidence.get("primary_supported_wids", [])[:20],
                "expansion_supported_score": round(evidence.get("expansion_supported_score", 0.0), 6),
                "expansion_supported_wids": evidence.get("expansion_supported_wids", [])[:20],
                "structure_mult_total": round(float(_srec.get("structure_mult_total") or 1.0), 6),
                "structure_factor": round(float(_srec.get("structure_factor") or 1.0), 6),
                "strong_term_count_struct": int(_srec.get("strong_term_count") or 0),
                "multi_term_paper_count_struct": int(_srec.get("multi_term_paper_count") or 0),
                "paper_count_struct": int(_srec.get("paper_count") or 0),
            }
        )

    t9 = time.perf_counter()
    stage5_sub_ms["build_ranked_list"] = (t9 - t8) * 1000.0

    n_scored_authors_pre_unified = len(scored_authors)
    _fb = _stage5_infer_supply_chain_first_breakpoint(
        n_author_papers_records=len(author_papers_list),
        n_unique_aids_in=len(_in_aids),
        n_unique_wids_in=len(_in_wids),
        n_paper_map=len(paper_map),
        n_papers_nonpos=papers_nonpos_drop,
        n_papers_for_agg=len(papers_for_agg),
        n_papers_for_agg_zero_authors=n_papers_for_agg_zero_authors,
        n_author_term_aids=n_author_term_aids,
        n_author_scores_after_term_recompute=n_author_scores_after_term_recompute,
        n_author_scores_before_best_paper_filter=n_author_scores_before_best_paper_filter,
        n_author_scores_after_best_paper_filter=n_author_scores_after_best_paper_filter,
        n_scored_authors_pre_unified=n_scored_authors_pre_unified,
    )
    stage5_supply_chain_audit: Dict[str, Any] = {
        "first_breakpoint": _fb,
        "n_author_papers_records": len(author_papers_list),
        "n_unique_aids_in": len(_in_aids),
        "n_unique_wids_in": len(_in_wids),
        "n_merged_paper_map": len(paper_map),
        "n_papers_nonpos_drop": papers_nonpos_drop,
        "n_papers_for_agg": len(papers_for_agg),
        "n_papers_for_agg_zero_authors": n_papers_for_agg_zero_authors,
        "n_author_term_aids": n_author_term_aids,
        "n_author_scores_after_term_recompute": n_author_scores_after_term_recompute,
        "n_author_scores_before_best_paper_filter": n_author_scores_before_best_paper_filter,
        "n_author_scores_after_best_paper_filter": n_author_scores_after_best_paper_filter,
        "n_best_paper_filter_removed": n_best_paper_filter_removed,
        "n_scored_authors_pre_unified": n_scored_authors_pre_unified,
    }
    if STAGE5_SUPPLY_CHAIN_AUDIT and _label_recall_stdout_enabled(recall):
        print("\n" + "-" * 80)
        print("[Stage5 supply_chain_audit]")
        print(json.dumps(stage5_supply_chain_audit, ensure_ascii=False, indent=2))
        print("-" * 80 + "\n")

    scored_authors.sort(key=lambda x: x["score"], reverse=True)
    _stage5_apply_suppression_audit_fields(
        scored_authors,
        author_record_by_aid,
        paper_map,
        author_top_works,
        term_cap_audit_records,
    )
    _print_stage5_single_paper_side_driven_suppression_audit(scored_authors, author_record_by_aid, recall)
    _print_stage5_non_hit_good_author_examples(scored_authors, author_record_by_aid, recall)
    _stage5_apply_broad_mainline_dominance_audit_fields(
        scored_authors,
        author_record_by_aid,
        paper_map,
        author_top_works,
        term_cap_audit_records,
    )
    _print_stage5_broad_mainline_dominance_audit(
        scored_authors,
        author_record_by_aid,
        paper_map,
        author_top_works,
        recall,
    )
    _print_stage5_term_cap_audit(scored_authors, term_cap_audit_records, term_map, top_k=25, recall=recall)
    _print_stage5_author_structure_audit(scored_authors, author_structure_audit, top_k=25, recall=recall)
    _print_stage5_support_dominance_audit(
        scored_authors,
        author_term_contrib,
        author_top_works,
        paper_map,
        term_map,
        debug_1 or {},
        top_k=25,
        recall=recall,
    )
    _stage5_apply_unified_ltr_lite_final_rerank(scored_authors, author_record_by_aid)
    # 批注：看 Top 作者分数来自哪几篇论文（全局 paper_score vs 作者贡献份额），便于判断偏题是否 Stage4 混入。
    paper_scores_by_wid_dbg = {p["wid"]: float(p["score"]) for p in papers_for_agg}
    _p("\n[Stage5 top-author paper provenance]")
    for a in scored_authors[:20]:
        aid = str(a.get("aid"))
        works_sorted = sorted(
            author_top_works.get(aid, []),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        _p(f"author={aid} final_score={float(a.get('score') or 0.0):.4f}")
        for rank, (wid, author_piece) in enumerate(works_sorted[:3], 1):
            ps = float(paper_scores_by_wid_dbg.get(wid, 0.0))
            ht = paper_hit_terms.get(wid, [])
            _p(
                f"  #{rank} wid={wid} paper_score={ps:.4f} author_piece={author_piece:.4f} hit_terms={ht}"
            )

    sorted_terms = sorted(
        [(term_map.get(tid, ""), score_map.get(tid, 0.0)) for tid in score_map],
        key=lambda x: x[1],
        reverse=True,
    )

    t9b = time.perf_counter()
    stage5_sub_ms["sort_authors_and_terms"] = (t9b - t9) * 1000.0

    top20_term_debug: List[Dict[str, Any]] = []
    if score_map and getattr(recall, "_last_tag_purity_debug", None):
        debug_by_tid = {str(d["tid"]): d for d in recall.debug_info.tag_purity_debug if d.get("tid") is not None}
        rank_factors = getattr(recall, "_last_cluster_rank_factors", {}) or recall.debug_info.cluster_rank_factors or {}
        for tid in sorted(score_map.keys(), key=lambda t: score_map.get(t, 0.0), reverse=True)[:20]:
            row = debug_by_tid.get(str(tid), {}) or {}
            top20_term_debug.append(
                {
                    "tid": tid,
                    "term": term_map.get(tid, ""),
                    "final_weight": float(score_map.get(tid, 0.0)),
                    "idf_val": float(row.get("idf_val", 0.0) or 0.0),
                    "base_score": float(row.get("base_score", 0.0) or 0.0),
                    "role_penalty": float(row.get("role_penalty", 1.0) or 1.0),
                    "task_advantage": float(row.get("task_advantage", 0.0) or 0.0)
                    if row.get("task_advantage") is not None
                    else None,
                    "cluster_factor": float(row.get("cluster_factor", 1.0) or 1.0),
                    "cluster_rank_factor": float(rank_factors.get(str(tid), 1.0) or 1.0),
                    "degree_w": int(row.get("degree_w", 0) or 0),
                    "cov_j": float(row.get("cov_j", 0.0) or 0.0),
                    "domain_span": int(row.get("domain_span", 0) or 0),
                    "anchor_sim": float(row.get("anchor_sim", 0.0) or 0.0)
                    if row.get("anchor_sim") is not None
                    else None,
                    "task_anchor_sim": float(row.get("task_anchor_sim", 0.0) or 0.0)
                    if row.get("task_anchor_sim") is not None
                    else None,
                }
            )

    t10 = time.perf_counter()
    stage5_sub_ms["top20_term_debug"] = (t10 - t9b) * 1000.0

    filter_closed_loop = debug_1.get("filter_closed_loop") or {}
    similar_to_pass = recall.debug_info.similar_to_pass or []
    similar_to_raw = recall.debug_info.similar_to_raw_rows or []

    filter_closed_loop.setdefault("similar_to_raw_tids", [r.get("tid") for r in similar_to_raw])
    filter_closed_loop.setdefault("similar_to_pass_tids", [r.get("tid") for r in similar_to_pass])
    filter_closed_loop.setdefault(
        "final_term_ids_for_paper",
        sorted(score_map.keys(), key=lambda t: score_map.get(t, 0.0), reverse=True),
    )
    filter_closed_loop.setdefault("final_term_count", len(score_map))

    # 低价值闭环检查降级：contains_check（industrial_kw/anchor_tid/bridged_kw）
    # 当前瓶颈已转向 Stage4/Stage5 的贡献聚合，默认不再填充该大块重复信息。

    recall.debug_info.filter_closed_loop = filter_closed_loop
    recall.debug_info.recall_vocab_count = len(score_map)
    recall.debug_info.work_count = all_works_count
    recall.debug_info.author_count = len(scored_authors)
    recall.debug_info.top_terms_final_contrib = top20_term_debug
    recall.debug_info.top_samples = scored_authors[:50]

    # -------------------------
    # 精准审计块 2/3：
    # [Stage5 term->author contribution top]
    # 目的：看每个 term 是否集中砸向少数作者；降噪：仅 **paper term 前 4 × 每 term 作者前 5**，避免淹没结构审计。
    # -------------------------
    final_term_ids_for_paper = sorted(score_map.keys(), key=lambda t: score_map.get(t, 0.0), reverse=True)
    term_author_rows: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for aid, atc in author_term_contrib.items():
        # 作者总分取 term 贡献和，避免受后续归一化影响 share 观察
        author_total = float(sum(float(v) for v in atc.values()) or 0.0)
        if author_total <= 0:
            continue
        term_hit_counter: Dict[str, int] = collections.Counter()
        for wid, _ in author_top_works.get(aid, []):
            tw = (paper_map.get(wid) or {}).get("term_weights") or {}
            for vid_s, w in tw.items():
                if float(w or 0.0) > 0:
                    term_hit_counter[str(vid_s)] += 1
        for tid in final_term_ids_for_paper:
            tid_s = str(tid)
            term_contrib = float(atc.get(tid_s, 0.0) or 0.0)
            if term_contrib <= 0:
                continue
            term_name = term_map.get(tid, "") or term_map.get(tid_s, "") or tid_s
            term_author_rows[term_name].append(
                {
                    "author_id": aid,
                    "term_contrib": term_contrib,
                    "paper_hits": int(term_hit_counter.get(tid_s, 0)),
                    "author_total_score": author_total,
                }
            )

    _p("\n[Stage5 term->author contribution top] (top 4 paper terms × 5 authors/term)")
    for tid in final_term_ids_for_paper[:4]:
        tid_s = str(tid)
        term_name = term_map.get(tid, "") or term_map.get(tid_s, "") or tid_s
        rows = sorted(
            term_author_rows.get(term_name, []),
            key=lambda x: float(x.get("term_contrib") or 0.0),
            reverse=True,
        )[:5]
        _p(f"term='{term_name}' authors={len(term_author_rows.get(term_name, []))}")
        for i, r in enumerate(rows, 1):
            total = max(float(r.get("author_total_score") or 0.0), 1e-9)
            share = float(r.get("term_contrib") or 0.0) / total
            _p(
                f"  #{i} author={r.get('author_id')} "
                f"term_contrib={float(r.get('term_contrib') or 0.0):.6f} "
                f"paper_hits={int(r.get('paper_hits') or 0)} "
                f"share_in_author={share:.3f}"
            )

    # -------------------------
    # 精准审计块 3/3：
    # [Stage5 top-author term mix]
    # 目的：查看 Top 作者是否被单一 term 主导（dominant_share 是否过高）。
    # -------------------------
    _p("\n[Stage5 top-author term mix]")
    for a in scored_authors[:20]:
        aid = str(a.get("aid"))
        atc = author_term_contrib.get(aid, {})
        rec = term_cap_audit_records.get(aid) or {}
        # 保留论文数（按 wid 去重），与 multi_term 论文数同日志展示，区分「单词单篇」作者
        _works = author_top_works.get(aid, [])
        paper_count_dedup = len({str(w) for w, _ in _works})
        mtp = int(rec.get("multi_term_paper_count") or 0)
        contribs = sorted(
            ((term_map.get(tid, "") or term_map.get(str(tid), "") or str(tid), float(v)) for tid, v in atc.items() if float(v or 0.0) > 0),
            key=lambda kv: kv[1],
            reverse=True,
        )
        total = sum(v for _, v in contribs) or 1e-9
        top3 = contribs[:3]
        mix = [f"{term}={v:.6f}({v/total:.2%})" for term, v in top3]
        dom_share = (top3[0][1] / total) if top3 else 0.0
        _p(
            f"author={aid} final={float(a.get('score') or 0.0):.4f} "
            f"papers={paper_count_dedup} multi_term_papers={mtp} "
            f"dominant_share={dom_share:.2%} top3={' | '.join(mix)}"
        )

    t11 = time.perf_counter()
    stage5_sub_ms["filter_closed_loop_meta"] = (t11 - t10) * 1000.0
    stage5_sub_ms["total"] = (t11 - t_stage5) * 1000.0

    author_evidence_debug = {
        aid: author_evidence_by_term_role.get(aid, {})
        for a in scored_authors[:50]
        for aid in [a.get("aid")]
    }
    stage5_term_cap_audit_out: List[Dict[str, Any]] = []
    for a in scored_authors[:25]:
        aid = str(a.get("aid") or "")
        r = dict(term_cap_audit_records.get(aid) or {})
        dom_tid = str(r.get("dominant_term_id") or "")
        dom_nm = term_map.get(dom_tid) or ""
        if not dom_nm and dom_tid.isdigit():
            dom_nm = term_map.get(str(int(dom_tid))) or ""
        r["author_id"] = aid
        r["dominant_term"] = dom_nm or dom_tid
        stage5_term_cap_audit_out.append(r)
    stage5_author_structure_audit_out = [
        {"author_id": str(a.get("aid")), **(author_structure_audit.get(str(a.get("aid")), {}) or {})}
        for a in scored_authors[:50]
    ]

    # --- StepX：为解释层导出每位作者的 top 命中标签词（core/support） ---
    author_top_terms: Dict[str, List[Dict[str, Any]]] = {}
    try:
        for a in scored_authors[: min(120, len(scored_authors))]:
            aid = str(a.get("aid") or "")
            atc = author_term_contrib.get(aid, {}) or {}
            contribs = sorted(
                ((str(tid_s), float(v)) for tid_s, v in atc.items() if float(v or 0.0) > 0),
                key=lambda kv: kv[1],
                reverse=True,
            )[:6]
            if not contribs:
                continue
            total = sum(v for _, v in contribs) or 1e-12
            rows: List[Dict[str, Any]] = []
            for tid_s, v in contribs:
                term = term_map.get(tid_s) or ""
                if not term and tid_s.isdigit():
                    term = term_map.get(str(int(tid_s))) or ""
                rr = _retrieval_role_for_paper_term(debug_1 or {}, tid_s)
                bucket = "core" if rr == "paper_primary" else "support"
                rows.append(
                    {
                        "term": term or tid_s,
                        "tid": tid_s,
                        "bucket": bucket,
                        "role": rr,
                        "score": float(v),
                        "share": float(v / total) if total > 0 else 0.0,
                    }
                )
            if rows:
                author_top_terms[aid] = rows
    except Exception:
        author_top_terms = {}
    recall.last_debug_info = {
        "active_domains": [str(d) for d in sorted(active_domain_set)],
        "dominance": float(dominance),
        "industrial_kws": industrial_kws,
        "anchor_skills": anchor_skills,
        "score_map": score_map,
        "term_map": term_map,
        "term_role_map": term_role_map,
        "idf_map": {},
        "filter_closed_loop": filter_closed_loop,
        "top_terms_final_contrib": top20_term_debug,
        "top_samples": scored_authors[:50],
        "work_count": all_works_count,
        "author_count": len(scored_authors),
        "recall_vocab_count": len(score_map),
        "author_evidence_by_term_role": author_evidence_debug,
        "stage5_term_cap_audit": stage5_term_cap_audit_out,
        "stage5_author_structure_audit": stage5_author_structure_audit_out,
        "stage5_authorship_weighting": authorship_w_stats,
        "stage5_sub_ms": stage5_sub_ms,
        "stage5_supply_chain_audit": stage5_supply_chain_audit,
        "author_top_terms": author_top_terms,
    }
    if debug_1 and debug_1.get("stage1_sub_ms") is not None:
        recall.last_debug_info["stage1_sub_ms"] = debug_1["stage1_sub_ms"]
    s4 = getattr(getattr(recall, "debug_info", None), "stage4_sub_ms", None)
    if s4:
        recall.last_debug_info["stage4_sub_ms"] = dict(s4)

    return [a["aid"] for a in scored_authors], recall.last_debug_info

