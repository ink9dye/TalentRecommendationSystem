import collections
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np

from config import (
    DB_PATH,
    VOCAB_P95_PAPER_COUNT,
    SIMILAR_TO_TOP_K,
    SIMILAR_TO_MIN_SCORE,
    TOPIC_ALIGN_SUBFIELD,
    TOPIC_ALIGN_FIELD,
)
from src.utils.domain_utils import DomainProcessor

# ---------- Stage2/3 保守常量：先跑通再精调 ----------
LABEL_EXPANSION_DEBUG = True  # 调试时打印 Stage2A/2B 流程
PRIMARY_MIN_IDENTITY = 0.62
IDENTITY_MARGIN = 0.08
PRIMARY_MAX_PER_ANCHOR = 2  # 保守：每锚点最多 2 个 primary
DENSE_MAX_PER_PRIMARY = 4   # Stage2B 每个 primary 最多 dense 近邻数
CLUSTER_MAX_PER_PRIMARY = 3 # Stage2B 每个 primary 最多簇内支持词数
COOC_SUPPORT_MIN_FREQ = 2
COOC_MAX_PER_PRIMARY = 2


@dataclass
class PreparedAnchor:
    """Stage1 输出，Stage2 输入。无缩写扩写表时 expanded_forms 仅 [anchor]。"""
    anchor: str
    vid: int
    anchor_type: str = "unknown"
    expanded_forms: List[str] = field(default_factory=list)


@dataclass
class LandingCandidate:
    """Stage2A 落点候选。"""
    vid: int
    term: str
    source: str  # similar_to | jd_vector
    semantic_score: float
    anchor_vid: int = 0
    anchor_term: str = ""


@dataclass
class PrimaryLanding:
    """Stage2A 选出的主落点。"""
    vid: int
    term: str
    identity_score: float
    source: str
    anchor_vid: int
    anchor_term: str


@dataclass
class ExpandedTermCandidate:
    """Stage2 输出 / Stage3 输入，含 term_role 与三层领域 topic_align。"""
    vid: int
    term: str
    term_role: str  # primary | dense_expansion | cluster_expansion | cooc_expansion
    identity_score: float
    source: str
    anchor_vid: int
    anchor_term: str
    semantic_score: float = 0.0
    degree_w: int = 0
    domain_span: int = 0
    target_degree_w: int = 0
    degree_w_expanded: int = 0
    cov_j: float = 0.0
    src_vids: List[int] = field(default_factory=list)
    hit_count: int = 0
    topic_align: float = 1.0
    topic_level: str = "missing"
    topic_confidence: float = 1.0


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
        # 来源标记：供 Stage3 做来源可信度加权（无硬编码词表）
        if rec_e and rec_c:
            rec["source"] = "edge_and_ctx"
        elif rec_e:
            rec["source"] = "edge_only"
        else:
            rec["source"] = "ctx_only"
        raw_merged.append(rec)

    # Stage2 三路调试：raw_edge / raw_ctx / raw_merged 各 top20，统一含 tid, term, sim_score, source/origin, degree_w, domain_span
    _top = lambda lst, key="sim_score": sorted(lst, key=lambda r: float(r.get(key, 0.0) or 0.0), reverse=True)[:20]
    label.debug_info.stage2_raw_edge_top20 = [
        {
            "tid": r.get("tid"),
            "term": (r.get("term") or "")[:40],
            "sim_score": round(float(r.get("sim_score", 0) or 0), 4),
            "origin": "edge",
            "degree_w": int(r.get("degree_w", 0) or 0),
            "domain_span": int(r.get("domain_span", 0) or 0),
        }
        for r in _top(raw_edge)
    ]
    label.debug_info.stage2_raw_ctx_top20 = [
        {
            "tid": r.get("tid"),
            "term": (r.get("term") or "")[:40],
            "sim_score": round(float(r.get("sim_score", 0) or 0), 4),
            "origin": "ctx",
            "degree_w": int(r.get("degree_w", 0) or 0),
            "domain_span": int(r.get("domain_span", 0) or 0),
        }
        for r in _top(raw_ctx)
    ]
    label.debug_info.stage2_raw_merged_top20 = [
        {
            "tid": r.get("tid"),
            "term": (r.get("term") or "")[:40],
            "sim_score": round(float(r.get("sim_score", 0) or 0), 4),
            "source": r.get("source", ""),
            "degree_w": int(r.get("degree_w", 0) or 0),
            "domain_span": int(r.get("domain_span", 0) or 0),
        }
        for r in _top(raw_merged)
    ]

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

    # 种子 vid -> source，用于传播到 cluster 扩展出的新词
    source_rank = {"edge_and_ctx": 2, "edge_only": 1, "ctx_only": 0}
    seed_to_source = {}
    for rec in raw_results:
        try:
            vid = int(rec["tid"])
            seed_to_source[vid] = rec.get("source") or "edge_only"
        except Exception:
            continue

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

        # 继承种子中最好的 source，供 Stage3 来源可信度加权
        seed_vids_list = list(agg.get("seed_vids") or [])
        best_source = "cluster"
        for sid in seed_vids_list:
            s = seed_to_source.get(int(sid))
            if s and source_rank.get(s, -1) > source_rank.get(best_source, -1):
                best_source = s
        raw_results.append(
            {
                "tid": vid,
                "term": term_map[vid],
                "sim_score": agg["sim_score"] * size_penalty * domain_penalty,
                "hit_count": agg["support"],
                "seed_vids": sorted(seed_vids_list),
                "degree_w": degree_w,
                "degree_w_expanded": degree_w_expanded,
                "target_degree_w": target_degree_w,
                "domain_span": domain_span,
                "cov_j": 0.0,
                "origin": "cluster",
                "source": best_source,
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
    if getattr(label, "verbose", False):
        print(f"[Bridge Debug] query_expansion_by_context_vector 收到 query_text 片段: {jd_snippet[:120]}")
    active_domains = set(re.findall(r"\d+", regex)) if regex and regex != ".*" else set()

    # 同步主召回的文本预处理/桥接逻辑：优先走 QueryEncoder.encode，再做 L2 归一化，
    # 避免 ctx 扩展与主 query 的表示不一致。
    v_jd, _ = encoder.encode(jd_snippet)
    if v_jd is None:
        return []
    v_jd = np.asarray(v_jd, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(v_jd)

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


# ---------- Stage2A：学术落点（跨类型 SIMILAR_TO 为主，可选 JD 向量） ----------


def _anchor_skills_to_prepared_anchors(label, anchor_skills: Dict[str, Any]) -> List[PreparedAnchor]:
    """将现有 anchor_skills (vid -> {term}) 转为 List[PreparedAnchor]。"""
    load_vocab_meta(label)
    out = []
    for vid_str, info in (anchor_skills or {}).items():
        try:
            vid = int(vid_str)
        except (TypeError, ValueError):
            continue
        term = (info.get("term") or "").strip() or (label._vocab_meta.get(vid, ("", ""))[0])
        if not term:
            continue
        out.append(PreparedAnchor(anchor=term, vid=vid, expanded_forms=[term]))
    return out


def retrieve_academic_term_by_similar_to(label, anchor: PreparedAnchor) -> List[LandingCandidate]:
    """Stage2A 落点：从锚点（industry）查跨类型 SIMILAR_TO → 学术词。图内为带扩写向量相似度。"""
    load_vocab_meta(label)
    if not getattr(label, "graph", None):
        if LABEL_EXPANSION_DEBUG:
            print(f"[Stage2A] SIMILAR_TO 跳过 anchor={anchor.anchor!r} vid={anchor.vid}（无 graph）")
        return []
    params = {
        "anchor_vid": anchor.vid,
        "min_score": SIMILAR_TO_MIN_SCORE,
        "top_k": SIMILAR_TO_TOP_K,
    }
    cypher = """
    MATCH (v:Vocabulary {id: $anchor_vid})-[r:SIMILAR_TO]->(v_rel:Vocabulary)
    WHERE r.score >= $min_score
      AND coalesce(v_rel.type, 'concept') IN ['concept', 'keyword']
    RETURN v_rel.id AS tid, v_rel.term AS term, r.score AS sim_score
    ORDER BY sim_score DESC
    LIMIT $top_k
    """
    try:
        rows = label.graph.run(cypher, **params).data()
    except Exception as e:
        if LABEL_EXPANSION_DEBUG:
            print(f"[Stage2A] SIMILAR_TO 查询异常 anchor_vid={anchor.vid} anchor={anchor.anchor!r}: {e}")
        return []
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2A] SIMILAR_TO anchor_vid={anchor.vid} anchor={anchor.anchor!r} min_score={SIMILAR_TO_MIN_SCORE} top_k={SIMILAR_TO_TOP_K} -> 命中 {len(rows)} 条")
    out = []
    for r in rows:
        tid = r.get("tid")
        if tid is None:
            continue
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            continue
        term = (r.get("term") or "").strip() or (label._vocab_meta.get(tid, ("", ""))[0])
        sim = max(0.0, min(1.0, float(r.get("sim_score", 0.0) or 0.0)))
        out.append(
            LandingCandidate(
                vid=tid,
                term=term or str(tid),
                source="similar_to",
                semantic_score=sim,
                anchor_vid=anchor.vid,
                anchor_term=anchor.anchor,
            )
        )
    if LABEL_EXPANSION_DEBUG and out:
        for i, c in enumerate(out[:5]):
            print(f"[Stage2A]   落点[{i}] vid={c.vid} term={c.term!r} sim={c.semantic_score:.3f}")
        if len(out) > 5:
            print(f"[Stage2A]   ... 共 {len(out)} 个落点候选")
    return out


def collect_landing_candidates(label, anchor: PreparedAnchor) -> List[LandingCandidate]:
    """Stage2A：仅跨类型 SIMILAR_TO（+ 可选 JD 向量）。不做 exact/alias、不做 anchor→dense。"""
    similar_list = retrieve_academic_term_by_similar_to(label, anchor)
    by_vid = {c.vid: c for c in similar_list}
    cands = list(by_vid.values())
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2A] collect_landing_candidates anchor={anchor.anchor!r} -> {len(cands)} 个候选")
    return cands


def score_academic_identity(c: LandingCandidate) -> float:
    """身份分：similar_to 用边权；jd_vector 用 0.5+0.5*semantic_score。"""
    if c.source == "similar_to":
        return max(0.0, min(1.0, c.semantic_score))
    if c.source == "jd_vector":
        return 0.5 + 0.5 * max(0.0, min(1.0, c.semantic_score))
    return 0.5 + 0.5 * max(0.0, min(1.0, c.semantic_score))


def select_primary_academic_landings(
    candidates: List[LandingCandidate],
    anchor_vid: int,
    min_identity: float = PRIMARY_MIN_IDENTITY,
    identity_margin: float = IDENTITY_MARGIN,
    max_per_anchor: int = PRIMARY_MAX_PER_ANCHOR,
) -> List[PrimaryLanding]:
    """保守：只留 identity 高且领先足够的；每锚点最多 max_per_anchor 个。"""
    if not candidates:
        return []
    for c in candidates:
        setattr(c, "identity_score", score_academic_identity(c))
    sorted_c = sorted(candidates, key=lambda x: getattr(x, "identity_score", 0.0), reverse=True)
    out = []
    for i, c in enumerate(sorted_c):
        sc = getattr(c, "identity_score", 0.0)
        if sc < min_identity:
            continue
        if i >= 1:
            prev_score = getattr(sorted_c[i - 1], "identity_score", 0.0)
            if prev_score - sc < identity_margin:
                continue
        if len(out) >= max_per_anchor:
            break
        out.append(
            PrimaryLanding(
                vid=c.vid,
                term=c.term,
                identity_score=sc,
                source=c.source,
                anchor_vid=c.anchor_vid,
                anchor_term=c.anchor_term,
            )
        )
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2A] select_primary 候选数={len(candidates)} min_identity={min_identity} -> primary 数={len(out)}")
        for p in out:
            print(f"[Stage2A]   primary vid={p.vid} term={p.term!r} identity={p.identity_score:.3f} source={p.source}")
    return out


# ---------- Stage2B：学术侧补充（dense / 簇 / 共现，不再用 SIMILAR_TO 学术→学术） ----------


def expand_from_vocab_dense_neighbors(
    label,
    primary_landings: List[PrimaryLanding],
    top_k_per_primary: int = None,
) -> List[ExpandedTermCandidate]:
    """从词汇向量索引取 primary 的学术近邻。term_role=dense_expansion。"""
    top_k_per_primary = top_k_per_primary or DENSE_MAX_PER_PRIMARY
    if not primary_landings or not getattr(label, "vocab_index", None) or not getattr(label, "vocab_to_idx", None):
        return []
    if getattr(label, "all_vocab_vectors", None) is None:
        return []
    load_vocab_meta(label)
    seen = set(p.vid for p in primary_landings)
    out = []
    for p in primary_landings:
        idx = label.vocab_to_idx.get(str(p.vid))
        if idx is None:
            continue
        vec = np.asarray(label.all_vocab_vectors[idx], dtype=np.float32).reshape(1, -1)
        k = min(top_k_per_primary + 5, 30)
        scores, ids = label.vocab_index.search(vec, k)
        count = 0
        for score, tid in zip(scores[0], ids[0]):
            if count >= top_k_per_primary:
                break
            tid = int(tid)
            if tid <= 0 or tid in seen:
                continue
            meta = label._vocab_meta.get(tid, ("", ""))
            if meta[1] not in ("concept", "keyword", "") and meta[1]:
                continue
            sim = max(0.0, min(1.0, float(score)))
            if sim < 0.3:
                continue
            seen.add(tid)
            out.append(
                ExpandedTermCandidate(
                    vid=tid,
                    term=meta[0] or str(tid),
                    term_role="dense_expansion",
                    identity_score=sim,
                    source="dense",
                    anchor_vid=p.anchor_vid,
                    anchor_term=p.anchor_term,
                    semantic_score=sim,
                    src_vids=[p.vid],
                    hit_count=1,
                )
            )
            count += 1
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2B] expand_from_vocab_dense_neighbors primary数={len(primary_landings)} -> dense_expansion {len(out)} 个")
    return out


def expand_from_cluster_members(
    label,
    primary_landings: List[PrimaryLanding],
    max_per_primary: int = None,
) -> List[ExpandedTermCandidate]:
    """从簇内取同簇支持词。term_role=cluster_expansion。"""
    max_per_primary = max_per_primary or CLUSTER_MAX_PER_PRIMARY
    voc_to_clusters = getattr(label, "voc_to_clusters", None) or {}
    cluster_members = getattr(label, "cluster_members", None) or {}
    if not primary_landings or not voc_to_clusters or not cluster_members:
        return []
    load_vocab_meta(label)
    seen = set(p.vid for p in primary_landings)
    out = []
    for p in primary_landings:
        clusters = voc_to_clusters.get(p.vid)
        if not clusters:
            continue
        cid, _ = max(clusters, key=lambda x: x[1])
        members = cluster_members.get(cid) or []
        count = 0
        for vid in members:
            if count >= max_per_primary:
                break
            vid = int(vid)
            if vid in seen:
                continue
            meta = label._vocab_meta.get(vid, ("", ""))
            if meta[1] not in ("concept", "keyword", "") and meta[1]:
                continue
            seen.add(vid)
            out.append(
                ExpandedTermCandidate(
                    vid=vid,
                    term=meta[0] or str(vid),
                    term_role="cluster_expansion",
                    identity_score=0.5,
                    source="cluster",
                    anchor_vid=p.anchor_vid,
                    anchor_term=p.anchor_term,
                    semantic_score=0.0,
                    src_vids=[p.vid],
                    hit_count=1,
                )
            )
            count += 1
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2B] expand_from_cluster_members primary数={len(primary_landings)} -> cluster_expansion {len(out)} 个")
    return out


def expand_from_cooccurrence_support(label, primary_landings: List[PrimaryLanding]) -> List[ExpandedTermCandidate]:
    """共现高支持度学术词。term_role=cooc_expansion。term -> voc_id 通过 _vocab_meta 反查。"""
    if not primary_landings or not getattr(label, "stats_conn", None):
        return []
    load_vocab_meta(label)
    term_to_vid = {}
    if getattr(label, "_vocab_meta", None):
        for v, (t, _) in label._vocab_meta.items():
            if t and t not in term_to_vid:
                term_to_vid[t.strip()] = v
    vid_to_term = {p.vid: (p.term, p.anchor_vid, p.anchor_term) for p in primary_landings}
    out = []
    seen = set(p.vid for p in primary_landings)
    for vid, (term, anchor_vid, anchor_term) in vid_to_term.items():
        if not term:
            continue
        try:
            rows = label.stats_conn.execute(
                "SELECT term_a, term_b, freq FROM vocabulary_cooccurrence WHERE (term_a = ? OR term_b = ?) AND freq >= ?",
                (term, term, COOC_SUPPORT_MIN_FREQ),
            ).fetchall()
        except Exception:
            continue
        count = 0
        for row in rows:
            if count >= COOC_MAX_PER_PRIMARY:
                break
            ta, tb, freq = row[0], row[1], row[2]
            other = (tb if ta == term else ta) or ""
            if other == term or not other:
                continue
            vid_other = term_to_vid.get(other.strip())
            if vid_other is None:
                continue
            if vid_other in seen:
                continue
            seen.add(vid_other)
            meta = label._vocab_meta.get(vid_other, ("", ""))
            out.append(
                ExpandedTermCandidate(
                    vid=vid_other,
                    term=meta[0] or other,
                    term_role="cooc_expansion",
                    identity_score=0.4,  # 共现给予固定身份分以过闸门 0.35
                    source="cooc",
                    anchor_vid=anchor_vid,
                    anchor_term=anchor_term,
                    semantic_score=0.0,
                    src_vids=[vid],
                    hit_count=int(freq),
                )
            )
            count += 1
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2B] expand_from_cooccurrence_support primary数={len(primary_landings)} -> cooc_expansion {len(out)} 个")
    return out


# ---------- 三层领域：vocabulary_topic_stats 查表与 topic_align 计算 ----------


def _load_vocabulary_topic_stats(label, voc_id: int) -> Optional[Dict[str, Any]]:
    """按 voc_id 查询 vocabulary_topic_stats，无记录或表不存在返回 None。"""
    if not getattr(label, "stats_conn", None):
        return None
    try:
        row = label.stats_conn.execute(
            "SELECT field_id, subfield_id, topic_id, field_dist, subfield_dist, topic_dist, source FROM vocabulary_topic_stats WHERE voc_id=?",
            (voc_id,),
        ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    return {
        "field_id": row[0],
        "subfield_id": row[1],
        "topic_id": row[2],
        "field_dist": row[3],
        "subfield_dist": row[4],
        "topic_dist": row[5],
        "source": row[6] if len(row) > 6 else None,
    }


def _parse_dist(dist_raw: Any) -> Dict[str, float]:
    """解析 JSON 分布，返回 {id_str: prob}。"""
    if not dist_raw:
        return {}
    try:
        d = json.loads(dist_raw) if isinstance(dist_raw, str) else dist_raw
        return {str(k): float(v) for k, v in (d or {}).items()}
    except Exception:
        return {}


def _extract_ids_from_row(topic_row: Dict[str, Any], key_id: str, key_dist: str) -> Set[str]:
    """从主值或分布提取 ID 集合。"""
    out = set()
    main = topic_row.get(key_id)
    if main is not None and str(main).strip():
        for p in re.split(r"[|,\s]+", str(main).strip()):
            if p:
                out.add(p.strip())
    if not out:
        dist = _parse_dist(topic_row.get(key_dist))
        for k, v in dist.items():
            if v and float(v) > 0:
                out.add(k)
    return out


def _compute_hierarchy_match_score(
    topic_row: Dict[str, Any],
    jd_field_ids: Set[str],
    jd_subfield_ids: Set[str],
    jd_topic_ids: Set[str],
) -> Tuple[float, str]:
    """层级命中分与档位。自上而下：topic → subfield → field → none。"""
    if not jd_topic_ids and not jd_subfield_ids and not jd_field_ids:
        return 1.0, "missing"
    term_topic = _extract_ids_from_row(topic_row, "topic_id", "topic_dist")
    term_subfield = _extract_ids_from_row(topic_row, "subfield_id", "subfield_dist")
    term_field = _extract_ids_from_row(topic_row, "field_id", "field_dist")

    def has_overlap(a: Set[str], b: Set[str]) -> bool:
        return bool(a & b) if a and b else False

    if jd_topic_ids and has_overlap(term_topic, jd_topic_ids):
        return 1.0, "topic"
    if jd_subfield_ids and has_overlap(term_subfield, jd_subfield_ids):
        return TOPIC_ALIGN_SUBFIELD, "subfield"
    if jd_field_ids and has_overlap(term_field, jd_field_ids):
        return TOPIC_ALIGN_FIELD, "field"
    return 0.0, "none"


def _compute_topic_confidence(topic_row: Dict[str, Any]) -> float:
    """按 source 映射主题可信度。"""
    source = (topic_row.get("source") or "").strip().lower()
    if source == "direct":
        return 1.0
    if source == "direct+cooc":
        return 0.9
    if source == "cooc":
        return 0.7
    return 1.0


def _attach_topic_align(
    label,
    voc_id: int,
    jd_field_ids: Set[str],
    jd_subfield_ids: Set[str],
    jd_topic_ids: Set[str],
) -> Tuple[float, str, float]:
    """为单个 voc_id 计算 topic_align, topic_level, topic_confidence。缺表/无记录时返回 (1.0, 'missing', 1.0)。"""
    topic_row = _load_vocabulary_topic_stats(label, voc_id)
    if not topic_row:
        return 1.0, "missing", 1.0
    hierarchy_score, topic_level = _compute_hierarchy_match_score(
        topic_row, jd_field_ids, jd_subfield_ids, jd_topic_ids
    )
    confidence = _compute_topic_confidence(topic_row)
    topic_align = hierarchy_score * confidence
    return topic_align, topic_level, confidence


def merge_primary_and_support_terms(
    primary_landings: List[PrimaryLanding],
    dense_list: List[ExpandedTermCandidate],
    cluster_list: List[ExpandedTermCandidate],
    cooc_list: List[ExpandedTermCandidate],
    label,
    active_domains: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> List[ExpandedTermCandidate]:
    """合并 primary + dense_expansion + cluster_expansion + cooc_expansion，补全 degree_w 与 topic_align 供 Stage3。"""
    load_vocab_meta(label)
    active = active_domains or set()
    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))
    out = []
    for p in primary_landings:
        row = None
        if getattr(label, "stats_conn", None):
            row = label.stats_conn.execute(
                "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (p.vid,),
            ).fetchone()
        degree_w = int(row[0]) if row else 0
        domain_span = int(row[1]) if row and len(row) > 1 else 0
        dist = {}
        if row and len(row) > 2 and row[2]:
            try:
                dist = json.loads(row[2]) if isinstance(row[2], str) else row[2]
            except Exception:
                pass
        expanded = expand_domain_dist(label, dist)
        degree_w_expanded = sum(expanded.values())
        target_degree_w = sum(expanded.get(str(d), 0) for d in active)
        topic_align, topic_level, topic_conf = _attach_topic_align(label, p.vid, jd_f, jd_s, jd_t)
        out.append(
            ExpandedTermCandidate(
                vid=p.vid,
                term=p.term,
                term_role="primary",
                identity_score=p.identity_score,
                source=p.source,
                anchor_vid=p.anchor_vid,
                anchor_term=p.anchor_term,
                semantic_score=1.0,
                degree_w=degree_w,
                domain_span=domain_span,
                degree_w_expanded=degree_w_expanded,
                target_degree_w=target_degree_w,
                src_vids=[],
                hit_count=1,
                topic_align=topic_align,
                topic_level=topic_level,
                topic_confidence=topic_conf,
            )
        )
    for c in dense_list:
        row = None
        if getattr(label, "stats_conn", None):
            row = label.stats_conn.execute(
                "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (c.vid,),
            ).fetchone()
        if row:
            c.degree_w = int(row[0])
            c.domain_span = int(row[1]) if len(row) > 1 else 0
            try:
                dist = json.loads(row[2]) if isinstance(row[2], str) else row[2] if len(row) > 2 else {}
            except Exception:
                dist = {}
            expanded = expand_domain_dist(label, dist)
            c.degree_w_expanded = sum(expanded.values())
            c.target_degree_w = sum(expanded.get(str(d), 0) for d in active)
        c.topic_align, c.topic_level, c.topic_confidence = _attach_topic_align(label, c.vid, jd_f, jd_s, jd_t)
        out.append(c)
    for c in cluster_list:
        row = None
        if getattr(label, "stats_conn", None):
            row = label.stats_conn.execute(
                "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (c.vid,),
            ).fetchone()
        if row:
            c.degree_w = int(row[0])
            c.domain_span = int(row[1]) if len(row) > 1 else 0
            try:
                dist = json.loads(row[2]) if isinstance(row[2], str) else row[2] if len(row) > 2 else {}
            except Exception:
                dist = {}
            expanded = expand_domain_dist(label, dist)
            c.degree_w_expanded = sum(expanded.values())
            c.target_degree_w = sum(expanded.get(str(d), 0) for d in active)
        c.topic_align, c.topic_level, c.topic_confidence = _attach_topic_align(label, c.vid, jd_f, jd_s, jd_t)
        out.append(c)
    for c in cooc_list:
        row = None
        if getattr(label, "stats_conn", None):
            row = label.stats_conn.execute(
                "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (c.vid,),
            ).fetchone()
        if row:
            c.degree_w = int(row[0])
            c.domain_span = int(row[1]) if len(row) > 1 else 0
            try:
                dist = json.loads(row[2]) if isinstance(row[2], str) else row[2] if len(row) > 2 else {}
            except Exception:
                dist = {}
            expanded = expand_domain_dist(label, dist)
            c.degree_w_expanded = sum(expanded.values())
            c.target_degree_w = sum(expanded.get(str(d), 0) for d in active)
        c.topic_align, c.topic_level, c.topic_confidence = _attach_topic_align(label, c.vid, jd_f, jd_s, jd_t)
        out.append(c)
    if LABEL_EXPANSION_DEBUG:
        n_primary = len(primary_landings)
        n_dense, n_cluster, n_cooc = len(dense_list), len(cluster_list), len(cooc_list)
        print(f"[Stage2B] merge_primary_and_support_terms primary={n_primary} dense={n_dense} cluster={n_cluster} cooc={n_cooc} -> 合计 {len(out)} 项")
    return out


def stage2_generate_academic_terms(
    label,
    prepared_anchors: List[PreparedAnchor],
    active_domain_set: Optional[Set[int]] = None,
    domain_regex: Optional[str] = None,
    query_vector=None,
    query_text: Optional[str] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> List[ExpandedTermCandidate]:
    """
    Stage2 总入口：先 Stage2A 主落点（保守），再 Stage2B 仅围绕 primary 扩展。
    无主落点则不扩展。可选传入 jd_field_ids/jd_subfield_ids/jd_topic_ids 供三层领域 topic_align。
    """
    active_domains = set(int(x) for x in (active_domain_set or [])) if active_domain_set else set()
    if domain_regex and not active_domains:
        try:
            active_domains = set(int(x) for x in re.findall(r"\d+", domain_regex))
        except (ValueError, TypeError):
            pass
    all_terms = []
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2] stage2_generate_academic_terms 开始 锚点数={len(prepared_anchors)} active_domains={len(active_domains)}")
    for anchor in prepared_anchors:
        candidates = collect_landing_candidates(label, anchor)
        primary_landings = select_primary_academic_landings(
            candidates, anchor.vid, PRIMARY_MIN_IDENTITY, IDENTITY_MARGIN, PRIMARY_MAX_PER_ANCHOR
        )
        if not primary_landings:
            if LABEL_EXPANSION_DEBUG:
                print(f"[Stage2] 锚点 anchor={anchor.anchor!r} vid={anchor.vid} 无 primary，跳过")
            continue
        dense_list = expand_from_vocab_dense_neighbors(label, primary_landings)
        cluster_list = expand_from_cluster_members(label, primary_landings)
        cooc_list = expand_from_cooccurrence_support(label, primary_landings)
        merged = merge_primary_and_support_terms(
            primary_landings,
            dense_list,
            cluster_list,
            cooc_list,
            label,
            active_domains=active_domains,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
        )
        all_terms.extend(merged)
        if LABEL_EXPANSION_DEBUG:
            print(f"[Stage2] 锚点 anchor={anchor.anchor!r} 本锚合并后 +{len(merged)} 项，累计 {len(all_terms)} 项")
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2] stage2_generate_academic_terms 结束 总学术词数={len(all_terms)}")
    return all_terms

