import collections
import math
import os
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from src.core.recall.label_pipeline.stage4_paper_recall import TERM_MAX_AUTHOR_SHARE
from src.utils.time_features import (
    compute_author_time_features,
    compute_author_recency_by_latest,
)
from src.utils.tools import get_decay_rate_for_domains as _get_decay_rate_for_domains
from src.core.recall.works_to_authors import accumulate_author_scores
from src.core.recall.label_means import paper_scoring
from src.core.recall.label_means.simple_factors import is_label_jd_title_gate_disabled


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
    → 新作比 → 归一化排序。
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

    # 标题向量：供 JD 语义门控；LABEL_NO_JD_TITLE_GATE 开启时跳过（省 SQLite / encode）
    wids = list(paper_map.keys())
    paper_title_vec_by_wid: Dict[str, Any] = {}
    paper_title_vec_by_title: Dict[str, Any] = {}
    if not is_label_jd_title_gate_disabled():
        enc = getattr(recall, "_query_encoder", None)
        store = getattr(recall, "_work_title_emb_store", None)
        if store is not None and wids:
            paper_title_vec_by_wid.update(store.get_many([str(w) for w in wids]))

        if enc is not None and paper_map:
            missing = [w for w in wids if str(w) not in paper_title_vec_by_wid]
            titles_unique: List[str] = []
            seen_t: Set[str] = set()
            for w in missing:
                t = (paper_map[w].get("title") or "").strip()
                if not t or t in seen_t:
                    continue
                seen_t.add(t)
                titles_unique.append(t)
            tm: Dict[str, Any] = {}
            if titles_unique:
                if hasattr(enc, "encode_batch"):
                    batch_m = enc.encode_batch(titles_unique)
                    tm = {
                        titles_unique[i]: np.asarray(batch_m[i], dtype=np.float32).reshape(1, -1).copy()
                        for i in range(len(titles_unique))
                    }
                else:
                    for t in titles_unique:
                        v, _ = enc.encode(t)
                        if v is not None:
                            tm[t] = np.asarray(v, dtype=np.float32).copy()
            for w in missing:
                t = (paper_map[w].get("title") or "").strip()
                if t in tm:
                    paper_title_vec_by_wid[str(w)] = tm[t]
            for w in wids:
                t = (paper_map[w].get("title") or "").strip()
                if not t:
                    continue
                v = paper_title_vec_by_wid.get(str(w))
                if v is not None:
                    paper_title_vec_by_title[t] = v

    context["paper_title_vec_by_wid"] = paper_title_vec_by_wid
    context["paper_title_vec_by_title"] = paper_title_vec_by_title
    context["_proximity_cache"] = {}

    if os.environ.get("LABEL_PROFILE_STAGE5", "").strip() in ("1", "true", "yes"):
        context["_paper_contrib_prof"] = {}

    papers_for_agg: List[Dict[str, Any]] = []
    paper_hit_terms: Dict[str, List[str]] = {}
    all_works_count = 0

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

    agg_result = accumulate_author_scores(papers_for_agg, top_k_per_author=3)
    author_scores = agg_result.author_scores
    author_top_works = agg_result.author_top_works
    paper_scores_by_wid = {p["wid"]: float(p["score"]) for p in papers_for_agg}
    author_evidence_by_term_role = aggregate_author_evidence_by_term_role(papers_for_agg)

    t3a = time.perf_counter()
    stage5_sub_ms["accumulate_authors"] = (t3a - t3_fan) * 1000.0

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

    # 每篇论文上有多少个非零 term（与 Stage4 multi-hit 对齐），供 multi_term_paper_count
    wid_n_terms = _wid_nonzero_term_counts(papers_for_agg)
    author_structure_audit: Dict[str, Dict[str, Any]] = {}
    support_only_penalty_rows: List[Dict[str, Any]] = []

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
    else:
        stage5_sub_ms["support_only_author_penalty"] = 0.0

    _print_stage5_support_only_penalty_audit(support_only_penalty_rows, recall=recall)

    t6 = time.perf_counter()
    stage5_sub_ms["time_and_family"] = (t6 - t5) * 1000.0

    if papers_for_agg and author_scores and author_top_works:
        paper_scores_by_wid = {p["wid"]: float(p["score"]) for p in papers_for_agg}
        max_paper = max(paper_scores_by_wid.values()) if paper_scores_by_wid else 0.0
        if max_paper > 0:
            min_contrib = max_paper * float(recall.AUTHOR_BEST_PAPER_MIN_RATIO)
            to_remove = [
                aid
                for aid in author_scores
                if max(
                    (paper_scores_by_wid.get(wid, 0.0) for wid, _ in author_top_works.get(aid, [])),
                    default=0.0,
                )
                < min_contrib
            ]
            for aid in to_remove:
                author_scores.pop(aid, None)

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
        scored_authors.append(
            {
                "aid": aid,
                "score": final_score,
                "raw_score": total_score,
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

    scored_authors.sort(key=lambda x: x["score"], reverse=True)
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
        "stage5_sub_ms": stage5_sub_ms,
    }
    if debug_1 and debug_1.get("stage1_sub_ms") is not None:
        recall.last_debug_info["stage1_sub_ms"] = debug_1["stage1_sub_ms"]
    s4 = getattr(getattr(recall, "debug_info", None), "stage4_sub_ms", None)
    if s4:
        recall.last_debug_info["stage4_sub_ms"] = dict(s4)

    return [a["aid"] for a in scored_authors], recall.last_debug_info

