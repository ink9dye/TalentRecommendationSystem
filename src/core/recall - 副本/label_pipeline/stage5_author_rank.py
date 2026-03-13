import collections
import math
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from src.utils.time_features import (
    compute_author_time_features,
    compute_author_recency_by_latest,
)
from src.utils.tools import get_decay_rate_for_domains as _get_decay_rate_for_domains
from src.core.recall.works_to_authors import accumulate_author_scores


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
    阶段 5：作者打分与排序。
    基本搬运 LabelRecallPath._stage5_score_and_rank_authors 逻辑，保持返回 (author_ids, last_debug_info)。
    """
    industrial_kws = debug_1.get("industrial_kws", [])
    anchor_skills = debug_1.get("anchor_skills", {})
    context = {
        "score_map": score_map,
        "term_map": term_map,
        "anchor_kws": [k.lower() for k in industrial_kws],
        "active_domain_set": active_domain_set,
        "dominance": dominance,
        "decay_rate": _get_decay_rate_for_domains(active_domain_set),
        "query_vector": debug_1.get("query_vector"),
    }

    paper_map: Dict[str, Dict[str, Any]] = {}
    author_raw_paper_cnt: Dict[str, int] = collections.Counter()

    for record in author_papers_list:
        aid = record["aid"]
        for paper in record["papers"]:
            wid = paper["wid"]
            entry = paper_map.setdefault(
                wid,
                {
                    "wid": wid,
                    "hits": paper["hits"],
                    "title": paper["title"],
                    "year": paper["year"],
                    "domains": paper["domains"],
                    "authors": [],
                },
            )
            entry["authors"].append({"aid": aid, "pos_weight": float(paper.get("weight") or 1.0)})
            author_raw_paper_cnt[aid] += 1

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
        p_score, p_hits, p_rank_score, p_term_weights = recall._compute_contribution(paper_struct, context)
        all_works_count += 1
        if p_score <= 0:
            continue
        paper_hit_terms[wid] = p_hits
        info["score"] = float(p_score)
        info["rank_score"] = float(p_rank_score or 0)
        info["term_weights"] = dict(p_term_weights or {})
        papers_for_agg.append(
            {
                "wid": wid,
                "score": float(p_score),
                "rank_score": float(p_rank_score or 0),
                "term_weights": dict(p_term_weights or {}),
                "authors": info["authors"],
            }
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

    agg_result = accumulate_author_scores(papers_for_agg, top_k_per_author=3)
    author_scores = agg_result.author_scores
    author_top_works = agg_result.author_top_works
    paper_scores_by_wid = {p["wid"]: float(p["score"]) for p in papers_for_agg}

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

    author_term_contrib: Dict[str, Dict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    for aid, works in author_top_works.items():
        for wid, contrib in works:
            info = paper_map.get(wid, {})
            r_score = info.get("rank_score") or 1.0
            tw = info.get("term_weights") or {}
            if r_score <= 0:
                continue
            for vid_s, w in tw.items():
                author_term_contrib[aid][vid_s] += contrib * (w / r_score)

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
            }
        )

    scored_authors.sort(key=lambda x: x["score"], reverse=True)
    sorted_terms = sorted(
        [(term_map.get(tid, ""), score_map.get(tid, 0.0)) for tid in score_map],
        key=lambda x: x[1],
        reverse=True,
    )

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

    for kw in (debug_1.get("industrial_kws") or []):
        label = f"industrial_kw:{kw}"
        filter_closed_loop.setdefault("contains_check", {})
        filter_closed_loop["contains_check"][label] = any(kw in t[0] for t in sorted_terms[:50])

    for tid in (anchor_skills or {}).keys():
        label = f"anchor_tid:{tid}"
        filter_closed_loop.setdefault("contains_check", {})
        filter_closed_loop["contains_check"][label] = any(
            str(tid) == str(t_id) for t_id, _ in sorted_terms[:50]
        )

    recall.debug_info.filter_closed_loop = filter_closed_loop
    recall.debug_info.recall_vocab_count = len(score_map)
    recall.debug_info.work_count = all_works_count
    recall.debug_info.author_count = len(scored_authors)
    recall.debug_info.top_terms_final_contrib = top20_term_debug
    recall.debug_info.top_samples = scored_authors[:50]

    recall.last_debug_info = {
        "active_domains": [str(d) for d in sorted(active_domain_set)],
        "dominance": float(dominance),
        "industrial_kws": industrial_kws,
        "anchor_skills": anchor_skills,
        "score_map": score_map,
        "term_map": term_map,
        "idf_map": {},  # 目前 idf_map 结构壳，由 score_map/tag_purity_debug 中可推导
        "filter_closed_loop": filter_closed_loop,
        "top_terms_final_contrib": top20_term_debug,
        "top_samples": scored_authors[:50],
        "work_count": all_works_count,
        "author_count": len(scored_authors),
        "recall_vocab_count": len(score_map),
    }

    return [a["aid"] for a in scored_authors], recall.last_debug_info

