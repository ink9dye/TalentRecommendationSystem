import collections
import json
import math
import re
import sqlite3
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from config import DB_PATH, VOCAB_P95_PAPER_COUNT, SIMILAR_TO_TOP_K, SIMILAR_TO_MIN_SCORE
from src.utils.domain_utils import DomainProcessor


def expand_semantic_map(
    label,
    core_vids: List[int],
    anchor_skills: Dict[str, Any],
    domain_regex: Optional[str] = None,
    query_vector=None,
    query_text: Optional[str] = None,
    return_raw: bool = False,
):
    """
    从 LabelRecallPath._expand_semantic_map 迁移而来。
    label 需提供：_query_expansion_with_topology/_query_expansion_by_context_vector/_expand_with_clusters/
              _calculate_academic_resonance/_calculate_anchor_resonance/_get_cooccurrence_domain_metrics/
              _calculate_final_weights，以及 debug_info。
    """
    regex = domain_regex if domain_regex else ".*"

    raw_edge = query_expansion_with_topology(label, core_vids, regex)
    raw_ctx = (
        query_expansion_by_context_vector(label, anchor_skills, query_text, regex, topk_per_anchor=3)
        if query_text
        else []
    )

    edge_map = {rec["tid"]: rec for rec in raw_edge}
    ctx_map = {rec["tid"]: rec for rec in raw_ctx}
    all_tids = set(edge_map.keys()) | set(ctx_map.keys())
    raw_merged = []
    for tid in all_tids:
        rec_e = edge_map.get(tid)
        rec_c = ctx_map.get(tid)
        sim_edge = float(rec_e["sim_score"]) if rec_e else 0.0
        sim_ctx = float(rec_c["sim_score"]) if rec_c else 0.0
        src_vids = set()
        if rec_e:
            src_vids.update(rec_e.get("src_vids") or [])
        if rec_c:
            src_vids.update(rec_c.get("src_vids") or [])
        hit = len(src_vids)
        base = label.EDGE_WEIGHT * sim_edge + label.CTX_EDGE_WEIGHT * sim_ctx
        degree_w = int((rec_e or rec_c).get("degree_w", 0) or 0)
        if degree_w >= label.HIT_BONUS_DEGREE_GATE:
            bonus = 1.0
        else:
            hit_eff = min(hit, label.HIT_BONUS_HIT_CAP) if hit >= 1 else 1
            bonus = min(label.HIT_BONUS_CAP, 1.0 + label.HIT_BONUS_BETA * math.log(hit_eff))
        sim_merged = base * bonus
        rec = dict(rec_e or rec_c)
        rec["sim_score"] = sim_merged
        rec["src_vids"] = sorted(src_vids)
        rec["hit_count"] = hit
        raw_merged.append(rec)

    raw_results = raw_merged
    if not raw_results:
        return [] if return_raw else ({}, {}, {})

    raw_results = expand_with_clusters(label, raw_results, regex, topk_per_seed=7, weight_decay=0.2)

    tids = [r["tid"] for r in raw_results]
    resonance_map = calculate_academic_resonance(label, tids)
    for rec in raw_results:
        rec["resonance"] = resonance_map.get(rec["tid"], 0.0)

    first_layer_core = [r["tid"] for r in raw_results if r.get("hit_count", 0) >= 2]
    if not first_layer_core:
        first_layer_core = tids
    anchor_resonance_map = calculate_anchor_resonance(label, tids, first_layer_core)
    for rec in raw_results:
        rec["anchor_resonance"] = anchor_resonance_map.get(rec["tid"], 0.0)

    active_domain_ids = set(re.findall(r"\d+", regex)) if regex and regex != ".*" else set()
    cooc_metrics = get_cooccurrence_domain_metrics(label, raw_results, active_domain_ids)
    for rec in raw_results:
        tid_key = str(rec["tid"])
        rec["cooc_span"] = cooc_metrics.get(tid_key, {}).get("cooc_span", 0.0)
        rec["cooc_purity"] = cooc_metrics.get(tid_key, {}).get("cooc_purity", 0.0)

    label.debug_info.expansion_raw_results = raw_results
    if return_raw:
        return raw_results
    return label._calculate_final_weights(raw_results, query_vector)


def expand_with_clusters(label, raw_results, domain_regex, topk_per_seed=5, weight_decay=0.2):
    # 直接复用原实现：依赖 label.cluster_members/label.voc_to_clusters/label.vocab_to_idx/label.all_vocab_vectors 等
    if not getattr(label, "cluster_members", None) or not getattr(label, "voc_to_clusters", None):
        return raw_results

    active_domain_ids = set(re.findall(r"\d+", domain_regex)) if domain_regex and domain_regex != ".*" else set()

    seed_vids = [int(rec["tid"]) for rec in raw_results]
    seed_vids_set = set(seed_vids)

    seed_to_cluster = {}
    for vid in seed_vids:
        clusters = label.voc_to_clusters.get(int(vid))
        if not clusters:
            continue
        cid, cscore = max(clusters, key=lambda x: x[1])
        seed_to_cluster[int(vid)] = (cid, cscore)

    if not seed_to_cluster:
        return raw_results

    seed_sim_map = {}
    for rec in raw_results:
        try:
            vid = int(rec["tid"])
        except Exception:
            continue
        seed_sim_map[vid] = float(rec.get("sim_score", 1.0))

    CLUSTER_EXPAND_TOP_SEEDS = 15
    sorted_by_sim = sorted(raw_results, key=lambda r: float(r.get("sim_score", 0.0)), reverse=True)
    allowed_seed_vids = {int(r["tid"]) for r in sorted_by_sim[:CLUSTER_EXPAND_TOP_SEEDS]}

    expansion_log = []
    cluster_expanded = {}

    for rec in raw_results:
        try:
            vid = int(rec["tid"])
        except Exception:
            continue
        if vid not in seed_to_cluster or vid not in allowed_seed_vids:
            continue

        cid, _ = seed_to_cluster[vid]
        members = label.cluster_members.get(int(cid)) or []
        if not members:
            continue

        candidates = [m for m in members if m not in seed_vids_set]
        if not candidates:
            continue

        seed_idx = label.vocab_to_idx.get(str(vid))
        if seed_idx is None:
            continue
        seed_vec = label.all_vocab_vectors[seed_idx]

        sims = []
        for m in candidates:
            midx = label.vocab_to_idx.get(str(m))
            if midx is None:
                continue
            mvec = label.all_vocab_vectors[midx]
            sim_in_cluster = float(np.dot(seed_vec, mvec))
            sims.append((m, sim_in_cluster))

        CLUSTER_MIN_SIM = 0.6
        sims = [(m, s) for m, s in sims if s >= CLUSTER_MIN_SIM]
        if not sims:
            continue

        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:topk_per_seed]
        seed_sim = seed_sim_map.get(vid, 1.0)

        seed_term = rec.get("term") or (
            label._vocab_meta.get(vid, ("", ""))[0] if getattr(label, "_vocab_meta", None) else ""
        )
        for m, sim_in_cluster in top:
            contrib = weight_decay * seed_sim * sim_in_cluster
            if contrib <= 0:
                continue
            entry = cluster_expanded.setdefault(int(m), {"sim_score": 0.0, "support": 0, "seed_vids": set()})
            entry["sim_score"] = max(entry["sim_score"], contrib)
            entry["support"] += 1
            entry["seed_vids"].add(int(vid))
            expansion_log.append(
                {
                    "term_tid": int(m),
                    "seed_vid": vid,
                    "seed_term": seed_term or str(vid),
                    "sim_in_cluster": round(sim_in_cluster, 4),
                    "seed_sim": round(seed_sim, 4),
                    "contrib": round(contrib, 6),
                }
            )

    if not cluster_expanded:
        return raw_results

    new_vids = [vid for vid in cluster_expanded.keys() if vid not in seed_vids_set]
    if not new_vids:
        return raw_results

    term_map = {}
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        ph = ",".join("?" * len(new_vids))
        rows = conn.execute(
            f"SELECT voc_id, term, entity_type FROM vocabulary WHERE voc_id IN ({ph})", new_vids
        ).fetchall()
        for r in rows:
            if (r["entity_type"] or "").lower() != "concept":
                continue
            term_map[int(r["voc_id"])] = r["term"]

    stats_map = {}
    ph = ",".join("?" * len(new_vids))
    rows = label.stats_conn.execute(
        f"SELECT voc_id, work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id IN ({ph})",
        new_vids,
    ).fetchall()
    for r in rows:
        stats_map[int(r[0])] = (int(r[1]), int(r[2]), r[3])

    active_domains = set(active_domain_ids)
    for vid, agg in cluster_expanded.items():
        if vid not in term_map or vid not in stats_map:
            continue
        degree_w, domain_span, dist_json = stats_map[vid]
        if degree_w <= 0:
            continue
        try:
            dist = json.loads(dist_json) if isinstance(dist_json, str) else dist_json
        except (TypeError, ValueError):
            dist = {}
        expanded = expand_domain_dist(label, dist)
        degree_w_expanded = sum(expanded.values())
        if active_domains:
            target_degree_w = sum(expanded.get(str(d), 0) for d in active_domains)
        else:
            target_degree_w = degree_w_expanded

        domain_ratio = target_degree_w / degree_w_expanded if degree_w_expanded else 0.0
        T = float(VOCAB_P95_PAPER_COUNT)
        if degree_w > T:
            x = min(float(degree_w) / T, 4.0)
            size_penalty = (1.0 / x) ** 2
        else:
            size_penalty = 1.0
        eps = 0.05
        r = max(domain_ratio, eps)
        domain_penalty = r ** 2

        raw_results.append(
            {
                "tid": vid,
                "term": term_map[vid],
                "sim_score": agg["sim_score"] * size_penalty * domain_penalty,
                "hit_count": agg["support"],
                "seed_vids": sorted(list(agg.get("seed_vids") or [])),
                "degree_w": degree_w,
                "degree_w_expanded": degree_w_expanded,
                "target_degree_w": target_degree_w,
                "domain_span": domain_span,
                "cov_j": 0.0,
                "origin": "cluster",
            }
        )

    label.debug_info.cluster_expansion_log = expansion_log
    return raw_results


def calculate_academic_resonance(label, tids: List[int]) -> Dict[int, float]:
    cypher = """
    MATCH (v1:Vocabulary)-[r:CO_OCCURRED_WITH]-(v2:Vocabulary)
    WHERE v1.id IN $tids AND v2.id IN $tids
    RETURN v1.id AS vid, SUM(r.weight) AS resonance_score
    """
    results = label.graph.run(cypher, tids=tids).data()
    return {r["vid"]: float(r["resonance_score"]) for r in results}


def calculate_anchor_resonance(label, tids: List[int], first_layer_tids: List[int]) -> Dict[int, float]:
    if not first_layer_tids:
        return {tid: 0.0 for tid in tids}
    cypher = """
    MATCH (v1:Vocabulary)-[r:CO_OCCURRED_WITH]-(v2:Vocabulary)
    WHERE v1.id IN $tids AND v2.id IN $first_layer_tids
    RETURN v1.id AS vid, SUM(r.weight) AS anchor_resonance_score
    """
    try:
        results = label.graph.run(cypher, tids=tids, first_layer_tids=first_layer_tids).data()
        return {r["vid"]: float(r["anchor_resonance_score"]) for r in results}
    except Exception:
        return {tid: 0.0 for tid in tids}


def get_cooccurrence_domain_metrics(label, raw_results, active_domain_ids):
    # 直接复用原实现：依赖 label.stats_conn 与 DB_PATH 主库
    if not raw_results or not active_domain_ids:
        return {str(rec["tid"]): {"cooc_span": 0.0, "cooc_purity": 0.0} for rec in raw_results}

    cooc_purity_from_table = {}
    try:
        tids = [rec["tid"] for rec in raw_results]
        domain_list = [str(d) for d in active_domain_ids]
        if tids and domain_list:
            ph_t = ",".join("?" * len(tids))
            ph_d = ",".join("?" * len(domain_list))
            rows = label.stats_conn.execute(
                f"SELECT voc_id, SUM(ratio) AS cooc_purity FROM vocabulary_cooc_domain_ratio WHERE voc_id IN ({ph_t}) AND domain_id IN ({ph_d}) GROUP BY voc_id",
                tids + domain_list,
            ).fetchall()
            cooc_purity_from_table = {str(r[0]): float(r[1]) for r in rows}
    except Exception:
        pass

    try:
        terms = list({rec["term"] for rec in raw_results})
        terms_set = set(terms)
        placeholders = ",".join("?" * len(terms))
        sql_cooc = (
            f"SELECT term_a, term_b, freq FROM vocabulary_cooccurrence "
            f"WHERE term_a IN ({placeholders}) OR term_b IN ({placeholders})"
        )
        rows = label.stats_conn.execute(sql_cooc, terms + terms).fetchall()

        term_to_partners = collections.defaultdict(list)
        for term_a, term_b, freq in rows:
            if term_a in terms_set:
                term_to_partners[term_a].append((term_b, freq))
            if term_b in terms_set:
                term_to_partners[term_b].append((term_a, freq))

        partner_terms = set()
        for pairs in term_to_partners.values():
            for p, _ in pairs:
                partner_terms.add(p)

        default_out = {
            str(rec["tid"]): {"cooc_span": 0.0, "cooc_purity": cooc_purity_from_table.get(str(rec["tid"]), 0.0)}
            for rec in raw_results
        }
        if not partner_terms:
            return default_out

        partner_list = list(partner_terms)
        ph = ",".join("?" * len(partner_list))
        with sqlite3.connect(DB_PATH) as main_conn:
            main_conn.row_factory = sqlite3.Row
            main_rows = main_conn.execute(
                f"SELECT voc_id, term FROM vocabulary WHERE term IN ({ph})", partner_list
            ).fetchall()
        partner_term_to_vocid = {row["term"]: row["voc_id"] for row in main_rows}

        partner_voc_ids = list(partner_term_to_vocid.values())
        if not partner_voc_ids:
            return default_out

        ph2 = ",".join("?" * len(partner_voc_ids))
        stats_rows = label.stats_conn.execute(
            f"SELECT voc_id, work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id IN ({ph2})",
            partner_voc_ids,
        ).fetchall()
        vocid_to_stats = {r[0]: (r[1], r[2], r[3]) for r in stats_rows}

        out = {}
        for rec in raw_results:
            tid, term = rec["tid"], rec["term"]
            pairs = term_to_partners.get(term, [])
            cooc_span_sum = cooc_purity_sum = total_freq = 0.0
            for partner_term, freq in pairs:
                voc_id = partner_term_to_vocid.get(partner_term)
                if voc_id is None:
                    continue
                st = vocid_to_stats.get(voc_id)
                if not st:
                    continue
                work_count, domain_span, dist_json = st
                try:
                    dist = json.loads(dist_json) if isinstance(dist_json, str) else dist_json
                except (TypeError, ValueError):
                    dist = {}
                expanded = expand_domain_dist(label, dist)
                degree_w_exp = sum(expanded.values())
                target_degree = sum(expanded.get(str(d), 0) for d in active_domain_ids)
                target_ratio = (target_degree / degree_w_exp) if degree_w_exp else 0.0
                cooc_span_sum += domain_span * freq
                cooc_purity_sum += target_ratio * freq
                total_freq += freq
            if total_freq > 0:
                out[str(tid)] = {
                    "cooc_span": cooc_span_sum / total_freq,
                    "cooc_purity": cooc_purity_from_table.get(str(tid), cooc_purity_sum / total_freq),
                }
            else:
                out[str(tid)] = {"cooc_span": 0.0, "cooc_purity": cooc_purity_from_table.get(str(tid), 0.0)}
        return out
    except Exception:
        return {
            str(rec["tid"]): {"cooc_span": 0.0, "cooc_purity": cooc_purity_from_table.get(str(rec["tid"]), 0.0)}
            for rec in raw_results
        }


def expand_domain_dist(label, dist):
    if not dist:
        return {}
    out = {}
    for key, count in dist.items():
        if not key or not count:
            continue
        for d in DomainProcessor.to_set(key):
            out[d] = out.get(d, 0) + count
    return out


def load_vocab_meta(label) -> None:
    if getattr(label, "_vocab_meta", None) is not None:
        return
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT voc_id, term, entity_type FROM vocabulary").fetchall()
            label._vocab_meta = {int(r[0]): (r[1] or "", r[2] or "") for r in rows}
    except Exception:
        label._vocab_meta = {}


def query_expansion_by_context_vector(label, anchor_skills, query_text, regex, topk_per_anchor=5):
    if not query_text or not anchor_skills:
        return []
    load_vocab_meta(label)
    encoder = label._query_encoder
    jd_snippet = (query_text or "").strip()[:500]
    active_domains = set(re.findall(r"\d+", regex)) if regex and regex != ".*" else set()

    v_jd = encoder.model.encode(
        [jd_snippet],
        batch_size=1,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    v_jd = np.asarray(v_jd, dtype=np.float32).reshape(1, -1)

    if not hasattr(label, "_term_vec_cache") or label._term_vec_cache is None:
        label._term_vec_cache = {}

    ctx_src_vids = []
    terms_lower = []
    terms_raw = []
    for vid, info in anchor_skills.items():
        term = (info.get("term") or "").strip()
        if not term:
            continue
        try:
            src_vid = int(vid)
        except Exception:
            continue
        ctx_src_vids.append(src_vid)
        tkey = term.lower()
        terms_lower.append(tkey)
        terms_raw.append(term)

    if not terms_lower:
        return []

    to_encode = []
    to_encode_keys = []
    for tkey, term in zip(terms_lower, terms_raw):
        if tkey not in label._term_vec_cache:
            to_encode.append(term)
            to_encode_keys.append(tkey)

    if to_encode:
        new_vecs = encoder.model.encode(
            to_encode,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        new_vecs = np.asarray(new_vecs, dtype=np.float32)
        for k, vec in zip(to_encode_keys, new_vecs):
            label._term_vec_cache[k] = vec

    v_terms = np.stack([label._term_vec_cache[tkey] for tkey in terms_lower], axis=0).astype(np.float32)
    if v_terms.ndim == 1:
        v_terms = v_terms.reshape(1, -1)

    lam = float(label.CTX_MIX_LAMBDA)
    embs = lam * v_jd + (1.0 - lam) * v_terms
    faiss.normalize_L2(embs)

    k = min(topk_per_anchor * 3, 30)
    scores, labels_arr = label.vocab_index.search(embs, k)

    by_tid = {}
    for row_i, src_vid in enumerate(ctx_src_vids):
        for score, tid in zip(scores[row_i], labels_arr[row_i]):
            tid = int(tid)
            if tid <= 0 or tid == int(src_vid):
                continue
            meta = label._vocab_meta.get(tid, ("", ""))
            if meta[1] not in ("concept", "keyword"):
                continue
            ctx_sim = max(0.0, float(score))
            if tid not in by_tid:
                by_tid[tid] = {"ctx_sim": 0.0, "src_vids": set(), "term": meta[0] or ""}
            by_tid[tid]["ctx_sim"] = max(by_tid[tid]["ctx_sim"], ctx_sim)
            by_tid[tid]["src_vids"].add(int(src_vid))
            if not by_tid[tid]["term"] and meta[0]:
                by_tid[tid]["term"] = meta[0]

    tids = list(by_tid.keys())
    if not tids:
        return []

    results = []
    for tid in tids:
        row = label.stats_conn.execute(
            "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
            (tid,),
        ).fetchone()
        if not row:
            continue
        degree_w, domain_span, dist_json = row
        try:
            dist = json.loads(dist_json) if isinstance(dist_json, str) else dist_json
        except (TypeError, ValueError):
            dist = {}
        expanded = expand_domain_dist(label, dist)
        degree_w_expanded = sum(expanded.values())
        target_degree_w = sum(expanded.get(str(d), 0) for d in active_domains)
        domain_ratio = target_degree_w / degree_w_expanded if degree_w_expanded else 0.0

        rec = by_tid[tid]
        ctx_sim = float(rec.get("ctx_sim", 0.0) or 0.0)
        if ctx_sim < float(label.SEMANTIC_MIN):
            continue

        if degree_w <= 40:
            purity_min = 0.4
        else:
            purity_min = float(label.DOMAIN_PURITY_MIN)
        if active_domains and domain_ratio < purity_min:
            continue

        T = float(VOCAB_P95_PAPER_COUNT)
        if degree_w > T:
            x = min(float(degree_w) / T, 4.0)
            size_penalty = (1.0 / x) ** 2
        else:
            size_penalty = 1.0

        src_vids = sorted(rec["src_vids"])
        results.append(
            {
                "tid": tid,
                "term": rec["term"] or label._vocab_meta.get(tid, ("", None))[0],
                "sim_score": ctx_sim * size_penalty,
                "src_vids": src_vids,
                "hit_count": len(src_vids),
                "degree_w": degree_w,
                "degree_w_expanded": degree_w_expanded,
                "target_degree_w": target_degree_w,
                "domain_span": domain_span,
                "cov_j": 0.0,
                "origin": "context_vector",
            }
        )
    return results


def query_expansion_with_topology(label, v_ids, regex):
    if not v_ids:
        return []
    active_domains = set(re.findall(r"\d+", regex))
    params = {"v_ids": list(v_ids), "min_score": SIMILAR_TO_MIN_SCORE, "top_k": SIMILAR_TO_TOP_K}
    cypher = """
    UNWIND $v_ids AS vid
    MATCH (v:Vocabulary {id: vid})-[r:SIMILAR_TO]->(v_rel:Vocabulary)
    WHERE r.score >= $min_score
      AND coalesce(v_rel.type, 'concept') = 'concept'
    WITH vid, v_rel.id AS tid, v_rel.term AS term, r.score AS sim_score
    ORDER BY vid, sim_score DESC
    WITH vid, collect({tid: tid, term: term, sim_score: sim_score})[0..$top_k] AS top3
    UNWIND top3 AS c
    RETURN vid AS src_vid, c.tid AS tid, c.term AS term, c.sim_score AS sim_score
    """
    rows = label.graph.run(cypher, **params).data()
    if not rows:
        label.debug_info.similar_to_raw_rows = []
        label.debug_info.similar_to_agg = []
        label.debug_info.similar_to_pass = []
        return []

    label.debug_info.similar_to_raw_rows = [
        {"src_vid": r.get("src_vid"), "tid": r.get("tid"), "term": r.get("term"), "sim_score": float(r.get("sim_score", 0.0) or 0.0)}
        for r in rows
    ]

    pipeline = {
        "n_similar_to_rows": len(rows),
        "active_domains": list(active_domains),
        "n_unique_tids": 0,
        "n_no_stats": 0,
        "n_fail_degree_w": 0,
        "n_fail_target_degree_w": 0,
        "n_fail_domain_ratio": 0,
        "n_fail_degree_w_expanded_zero": 0,
        "n_final": 0,
        "sample_fail_no_stats": [],
        "sample_fail_degree": [],
        "sample_fail_target": [],
        "sample_fail_ratio": [],
        "fail_domain_ratio_details": [],
    }

    by_tid = {}
    for r in rows:
        tid = r["tid"]
        term = r["term"] or ""
        sim = float(r["sim_score"])
        if tid not in by_tid:
            by_tid[tid] = {"tid": tid, "term": term, "sim_score": sim, "src_vids": set(), "hit_count": 0, "origin": "similar_to"}
        by_tid[tid]["sim_score"] = max(by_tid[tid]["sim_score"], sim)
        src_vid = r.get("src_vid")
        if src_vid is not None:
            try:
                by_tid[tid]["src_vids"].add(int(src_vid))
            except Exception:
                pass

    for tid, rec in by_tid.items():
        rec["hit_count"] = len(rec.get("src_vids") or [])
        rec["src_vids"] = sorted(list(rec.get("src_vids") or []))

    tids = list(by_tid.keys())
    pipeline["n_unique_tids"] = len(tids)

    label.debug_info.similar_to_agg = [
        {
            "tid": v.get("tid"),
            "term": v.get("term", ""),
            "sim_score": float(v.get("sim_score", 0.0) or 0.0),
            "hit_count": int(v.get("hit_count", 0) or 0),
            "src_vids": v.get("src_vids", []),
        }
        for v in by_tid.values()
    ]

    results = []
    for tid in tids:
        row = label.stats_conn.execute(
            "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
            (tid,),
        ).fetchone()
        if not row:
            pipeline["n_no_stats"] += 1
            if len(pipeline["sample_fail_no_stats"]) < 5:
                pipeline["sample_fail_no_stats"].append(tid)
            continue
        degree_w, domain_span, dist_json = row
        if degree_w > VOCAB_P95_PAPER_COUNT:
            pipeline["n_fail_degree_w"] += 1
            if len(pipeline["sample_fail_degree"]) < 5:
                pipeline["sample_fail_degree"].append(tid)
        try:
            dist = json.loads(dist_json) if isinstance(dist_json, str) else dist_json
        except (TypeError, ValueError):
            dist = {}
        expanded = expand_domain_dist(label, dist)
        degree_w_expanded = sum(expanded.values())
        target_degree_w = sum(expanded.get(str(d), 0) for d in active_domains)
        domain_ratio = target_degree_w / degree_w_expanded if degree_w_expanded else 0.0
        if degree_w <= 40:
            purity_min = 0.4
        else:
            purity_min = float(label.DOMAIN_PURITY_MIN)
        if domain_ratio < purity_min:
            if degree_w_expanded == 0:
                pipeline["n_fail_degree_w_expanded_zero"] += 1
            pipeline["n_fail_domain_ratio"] += 1
            if len(pipeline["sample_fail_ratio"]) < 5:
                pipeline["sample_fail_ratio"].append(tid)
            continue

        T = float(VOCAB_P95_PAPER_COUNT)
        if degree_w > T:
            x = min(float(degree_w) / T, 4.0)
            size_penalty = (1.0 / x) ** 2
        else:
            size_penalty = 1.0

        pipeline["n_final"] += 1
        rec = by_tid[tid]
        rec["degree_w"] = degree_w
        rec["degree_w_expanded"] = degree_w_expanded
        rec["target_degree_w"] = target_degree_w
        rec["domain_span"] = domain_span
        rec["cov_j"] = 0.0
        rec["sim_score"] = float(rec.get("sim_score", 0.0) or 0.0) * size_penalty
        results.append(rec)

    label.debug_info.expansion_pipeline_stats = pipeline
    label.debug_info.similar_to_pass = [
        {
            "tid": r.get("tid"),
            "term": r.get("term", ""),
            "sim_score": float(r.get("sim_score", 0.0) or 0.0),
            "hit_count": int(r.get("hit_count", 0) or 0),
            "src_vids": r.get("src_vids", []),
            "degree_w": int(r.get("degree_w", 0) or 0),
            "degree_w_expanded": int(r.get("degree_w_expanded", 0) or 0),
            "target_degree_w": int(r.get("target_degree_w", 0) or 0),
            "domain_span": int(r.get("domain_span", 0) or 0),
        }
        for r in results
    ]
    return results

