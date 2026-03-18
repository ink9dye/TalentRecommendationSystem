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
    TOPIC_ALIGN_NONE,
    TRUSTED_SOURCE_TYPES_FOR_DIFFUSION,
)
from src.core.recall.label_means.hierarchy_guard import (
    compute_hierarchical_fit,
    compute_purity,
)
from src.utils.domain_utils import DomainProcessor
from src.core.recall.label_means.label_debug import debug_print

# ---------- Stage2/3 保守常量：单一决策链，无冗余阈值（详见 README） ----------
LABEL_EXPANSION_DEBUG = True  # 调试时打印 Stage2A/2B 流程
STAGE2_VERBOSE_DEBUG = True   # True 时输出 Stage2 详细工整表格，便于调试


def _stage2_header(title: str, char: str = "=") -> None:
    """Stage2 调试：打印一节标题，工整分隔。"""
    if not (LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG):
        return
    line = char * 72
    print(f"\n{line}\n  [Stage2] {title}\n{line}")


def _stage2_table(rows: List[List[str]], header: List[str], col_widths: Optional[List[int]] = None) -> None:
    """Stage2 调试：打印一张对齐的表格。"""
    if not (LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG) or not header:
        return
    if not col_widths:
        col_widths = [max(len(str(h)), 4) for h in header]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(f"  {fmt.format(*header)}")
    print("  " + "-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        row_str = [str(x)[:w] for x, w in zip(row, col_widths)]
        if len(row_str) < len(col_widths):
            row_str.extend([""] * (len(col_widths) - len(row_str)))
        print(f"  {fmt.format(*row_str[:len(col_widths)])}")
PRIMARY_MIN_IDENTITY = 0.62       # Stage2A 准入：identity 下限（与 PRIMARY_MIN_PATH_MATCH 共同构成准入）
PRIMARY_MAX_PER_ANCHOR = 2        # 每锚点最多 primary 数
PRIMARY_TOP_M_PER_ANCHOR = 5      # 每锚先保留 top-m 候选再准入与冲突消解
CONDITIONED_VEC_TOP_K = 12        # 每锚点 conditioned_vec 检索学术词 top-k，与 SIMILAR_TO 合并
SEED_MIN_IDENTITY = 0.65         # Stage2B seed 准入：唯一常量，不与其他阈值叠加
DENSE_MAX_PER_PRIMARY = 4
CLUSTER_MAX_PER_PRIMARY = 3
COOC_SUPPORT_MIN_FREQ = 2
COOC_MAX_PER_PRIMARY = 2

# ---------- Stage2A 准入：仅两个门槛 + source 折扣，不再按 source 分多套阈值 ----------
PRIMARY_MIN_HIERARCHY_MATCH = 0.30   # hierarchy_match = 0.4*topic + 0.4*path + 0.2*subfield，effective = hierarchy_match * source_factor
PRIMARY_MIN_PATH_MATCH = 0.35
CONDVEC_SOURCE_FACTOR = 0.85         # conditioned_vec 来源时 effective_hierarchy = hierarchy_match * 0.85；similar_to = 1.0
PRIMARY_RESCUE_CROSS_ANCHOR_MIN = 2  # rescue：多锚支持 + path/jd/semantic 足够时可准入
PRIMARY_RESCUE_PATH_MATCH_MIN = 0.45
PRIMARY_RESCUE_JD_ALIGN_MIN = 0.80
PRIMARY_RESCUE_SEMANTIC_MIN = 0.78
HIERARCHY_LEVEL_TOPIC_EXACT = "topic_exact"
HIERARCHY_LEVEL_SUBFIELD_MATCH = "subfield_match"
HIERARCHY_LEVEL_FIELD_ONLY = "field_only"
HIERARCHY_LEVEL_OFF_PATH = "off_path"
TOPIC_SPAN_PENALTY_FACTOR = 0.2   # 泛化软惩罚：topic_span_penalty = 1/(1 + factor * max(0, span-1))
DOMAIN_SPAN_EXTREME = 24          # support 扩散仅在极端异常时硬拒绝（>24）；泛化主要由 topic_span_penalty 表达
SUPPORT_MIN_DOMAIN_FIT = 0.20     # Stage2B support 准入：domain_fit 低于此不进词池（唯一 support 门槛）
# ---------- Primary 排序：只保留一套权重（PRIMARY_SCORE_W_*），不再使用 PRIMARY_W_* ----------
PRIMARY_SCORE_W_SEMANTIC = 0.22
PRIMARY_SCORE_W_IDENTITY = 0.18
PRIMARY_SCORE_W_JD_ALIGN = 0.18
PRIMARY_SCORE_W_CROSS_ANCHOR = 0.10
PRIMARY_SCORE_W_NEIGHBOR = 0.08
PRIMARY_SCORE_W_FIELD = 0.10
PRIMARY_SCORE_W_SUBFIELD = 0.18
PRIMARY_SCORE_W_TOPIC = 0.24
PRIMARY_SCORE_W_PATH = 0.18
PRIMARY_SCORE_W_SPECIFICITY = 0.10


@dataclass
class PreparedAnchor:
    """Stage1 输出，Stage2 输入。无缩写扩写表时 expanded_forms 仅 [anchor]。conditioned_vec 为 JD 上下文条件化表示。"""
    anchor: str
    vid: int
    anchor_type: str = "unknown"
    expanded_forms: List[str] = field(default_factory=list)
    conditioned_vec: Optional[np.ndarray] = None  # 条件化锚点向量，用于 Stage2A 落点打分
    source_type: str = "skill_direct"    # skill_direct | jd_vector_supplement
    source_weight: float = 1.0          # 补充锚点用较低权重，primary 打分时乘此值


@dataclass
class LandingCandidate:
    """Stage2A 落点候选。"""
    vid: int
    term: str
    source: str  # similar_to（当前 Stage2A 唯一来源）；jd_vector 预留
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
    domain_fit: float = 1.0


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
    domain_fit: float = 1.0
    parent_primary: str = ""  # 产生该词的 primary term（primary 自身为 term）


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


def _voc_id_to_term_lower(conn, voc_ids: List[int]) -> Dict[int, str]:
    """从主库 vocabulary 取 voc_id -> term_lower，与 build_vocab_stats 的清洗一致（strip+lower）。"""
    if not voc_ids:
        return {}
    ph = ",".join("?" * len(voc_ids))
    rows = conn.execute(
        f"SELECT voc_id, term FROM vocabulary WHERE voc_id IN ({ph})",
        voc_ids,
    ).fetchall()
    out = {}
    for (vid, term) in rows:
        t = (term or "").strip().lower()
        if t:
            out[int(vid)] = t
    return out


def calculate_academic_resonance(label, tids: List[int]) -> Dict[int, float]:
    """
    从 vocab_stats.db 的 vocabulary_cooccurrence 计算候选词集内部共现权重和（学术共鸣）。
    与 build_vocab_stats_index 的共现数据一致，不再依赖 Neo4j CO_OCCURRED_WITH。
    """
    out = {tid: 0.0 for tid in tids}
    if len(tids) < 2 or not getattr(label, "stats_conn", None):
        return out
    try:
        with sqlite3.connect(DB_PATH) as main_conn:
            tid_to_term = _voc_id_to_term_lower(main_conn, tids)
        if len(tid_to_term) < 2:
            return out
        term_to_tid = {t: vid for vid, t in tid_to_term.items()}
        terms = list(term_to_tid.keys())
        ph = ",".join("?" * len(terms))
        sql = (
            f"SELECT term_a, term_b, freq FROM vocabulary_cooccurrence "
            f"WHERE term_a IN ({ph}) AND term_b IN ({ph})"
        )
        rows = label.stats_conn.execute(sql, terms + terms).fetchall()
        for (ta, tb, freq) in rows:
            f = int(freq or 0)
            if ta in term_to_tid and tb in term_to_tid:
                vid_a, vid_b = term_to_tid[ta], term_to_tid[tb]
                if vid_a in out:
                    out[vid_a] += f
                if vid_b in out:
                    out[vid_b] += f
    except Exception:
        pass
    return out


def calculate_anchor_resonance(label, tids: List[int], first_layer_tids: List[int]) -> Dict[int, float]:
    """
    从 vocab_stats.db 的 vocabulary_cooccurrence 计算候选词与第一层学术词的共现权重和（锚点共鸣）。
    与 build_vocab_stats_index 的共现数据一致，不再依赖 Neo4j CO_OCCURRED_WITH。
    """
    out = {tid: 0.0 for tid in tids}
    if not first_layer_tids or not getattr(label, "stats_conn", None):
        return out
    try:
        with sqlite3.connect(DB_PATH) as main_conn:
            tid_to_term_cand = _voc_id_to_term_lower(main_conn, tids)
            tid_to_term_first = _voc_id_to_term_lower(main_conn, first_layer_tids)
        if not tid_to_term_cand or not tid_to_term_first:
            return out
        terms_cand = set(tid_to_term_cand.values())
        terms_first = set(tid_to_term_first.values())
        term_to_tid_cand = {t: vid for vid, t in tid_to_term_cand.items()}
        ph_c = ",".join("?" * len(terms_cand))
        ph_f = ",".join("?" * len(terms_first))
        sql = (
            f"SELECT term_a, term_b, freq FROM vocabulary_cooccurrence "
            f"WHERE (term_a IN ({ph_c}) AND term_b IN ({ph_f})) OR (term_a IN ({ph_f}) AND term_b IN ({ph_c}))"
        )
        params = list(terms_cand) + list(terms_first) + list(terms_first) + list(terms_cand)
        rows = label.stats_conn.execute(sql, params).fetchall()
        for (ta, tb, freq) in rows:
            f = int(freq or 0)
            if ta in terms_cand and tb in terms_first and ta in term_to_tid_cand:
                out[term_to_tid_cand[ta]] = out.get(term_to_tid_cand[ta], 0) + f
            elif ta in terms_first and tb in terms_cand and tb in term_to_tid_cand:
                out[term_to_tid_cand[tb]] = out.get(term_to_tid_cand[tb], 0) + f
    except Exception:
        pass
    return out


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

    # Context-Aware：用 JD 片段作为上下文拼接，使 embedding 向目标领域偏移，减少多义词误匹配
    context_snippet = (query_text or "").strip()[:100].replace("(", " ").replace(")", " ")
    to_encode = []
    to_encode_keys = []
    for tkey, term in zip(terms_lower, terms_raw):
        if tkey not in label._term_vec_cache:
            term_with_ctx = f"{term} ({context_snippet})" if context_snippet else term
            to_encode.append(term_with_ctx)
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


# 仅用于 debug 打印标签 [高歧义]，不参与准入或打分
HIGH_AMBIGUITY_ANCHOR_TYPES = frozenset({"acronym", "generic_task_term"})


def check_seed_eligibility(
    label,
    p: "PrimaryLanding",
    jd_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float, Optional[str]]:
    """
    Stage2B 唯一 seed 决策入口：identity/source 通过后，再判 weak_retain/错义/cooc 严格 bar。
    返回 (eligible, seed_score, block_reason)；block_reason 仅在 eligible=False 时有值。
    """
    identity = getattr(p, "identity_score", 0.0) or 0.0
    src = (getattr(p, "source", "") or "").strip().lower()
    trusted_set = {s.strip().lower() for s in TRUSTED_SOURCE_TYPES_FOR_DIFFUSION if s}
    if identity < SEED_MIN_IDENTITY:
        return False, 0.0, "seed_score_too_low"
    if trusted_set and src not in trusted_set:
        return False, 0.0, "source_not_trusted"
    retain_mode = getattr(p, "retain_mode", "normal") or "normal"
    suppress_seed = getattr(p, "suppress_seed", True)
    if retain_mode == "weak_retain" and suppress_seed:
        return False, 0.0, "weak_primary_no_expand"
    anchor_text = getattr(p, "anchor_term", "") or ""
    if is_semantic_mismatch_seed(anchor_text, getattr(p, "term", "") or ""):
        return False, 0.0, "semantic_mismatch_seed"
    topic_source = getattr(p, "topic_source", "missing") or "missing"
    if topic_source == "cooc":
        sem = float(getattr(p, "semantic_score", 0) or 0)
        jd_align = float(getattr(p, "jd_align", 0.5) or 0.5)
        aid = float(getattr(p, "anchor_identity_score", identity) or identity)
        if not (sem >= 0.84 and jd_align >= 0.82 and aid >= 0.35):
            return False, 0.0, "cooc_seed_blocked"
    primary_score = getattr(p, "primary_score", identity) or identity
    path_match = float(getattr(p, "path_match", 0) or 0)
    genericity_penalty = float(getattr(p, "topic_span_penalty", 1.0) or 1.0)
    seed_score = primary_score * (0.7 + 0.3 * path_match) * genericity_penalty

    jd_align = float(getattr(p, "jd_align", 0.5) or 0.5)
    support_count = int(getattr(p, "cross_anchor_support_count", 1) or 1)
    blocked, block_reason = should_block_seed_expansion(
        anchor_text=anchor_text,
        primary_term=getattr(p, "term", "") or "",
        primary_score=primary_score,
        anchor_identity=getattr(p, "anchor_identity_score", identity) or identity,
        jd_align=jd_align,
        source_type=src,
        support_count=support_count,
        retain_mode=retain_mode,
        suppress_seed=suppress_seed,
    )
    if blocked:
        if LABEL_EXPANSION_DEBUG:
            print(f"[Stage2B] seed 禁扩 term={getattr(p, 'term', '')!r} anchor={anchor_text!r} reason={block_reason}")
        return False, 0.0, block_reason
    return True, seed_score, None


def _anchor_skills_to_prepared_anchors(label, anchor_skills: Dict[str, Any]) -> List[PreparedAnchor]:
    """将现有 anchor_skills (vid -> {term, anchor_type, conditioned_vec?, anchor_source?, anchor_source_weight?}) 转为 List[PreparedAnchor]。"""
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
        anchor_type = (info.get("anchor_type") or "unknown").strip().lower()
        conditioned = info.get("conditioned_vec")
        if conditioned is not None and hasattr(conditioned, "__len__"):
            conditioned = np.asarray(conditioned, dtype=np.float32).flatten()
        source_type = (info.get("anchor_source") or "skill_direct").strip()
        source_weight = float(info.get("anchor_source_weight", 1.0))
        out.append(
            PreparedAnchor(
                anchor=term,
                vid=vid,
                anchor_type=anchor_type,
                expanded_forms=[term],
                conditioned_vec=conditioned,
                source_type=source_type,
                source_weight=source_weight,
            )
        )
    return out


def _top_terms_by_vector(label, vec, k: int = 5) -> List[str]:
    """按余弦相似度返回与 vec 最接近的 k 个词（term 字符串），用于 Neighbor Compare。"""
    if vec is None or getattr(label, "vocab_to_idx", None) is None or getattr(label, "all_vocab_vectors", None) is None:
        return []
    try:
        v = np.asarray(vec, dtype=np.float32).flatten()
        mat = np.asarray(label.all_vocab_vectors, dtype=np.float32)
        if mat.ndim == 1:
            return []
        dots = np.dot(mat, v)
        norms = np.linalg.norm(mat, axis=1)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-9:
            return []
        sims = dots / (norms * norm_v + 1e-9)
        top_idx = np.argsort(sims)[::-1][:k]
        load_vocab_meta(label)
        idx_to_vid = {idx: vid_str for vid_str, idx in label.vocab_to_idx.items()}
        out = []
        for idx in top_idx:
            vid_str = idx_to_vid.get(int(idx))
            if vid_str is None:
                continue
            try:
                vid = int(vid_str)
            except (TypeError, ValueError):
                continue
            term = (label._vocab_meta.get(vid, ("", ""))[0] or "").strip() or vid_str
            out.append(term)
        return out
    except Exception:
        return []


def _compute_jd_candidate_alignment(label, vid: int, query_vector) -> float:
    """数据驱动：候选词与 JD 整体向量的余弦相似度，用于 primary 打分，无词表。"""
    if query_vector is None or getattr(label, "vocab_to_idx", None) is None or getattr(label, "all_vocab_vectors", None) is None:
        return 0.5
    idx = label.vocab_to_idx.get(str(vid))
    if idx is None:
        return 0.5
    try:
        term_vec = np.asarray(label.all_vocab_vectors[idx], dtype=np.float32).flatten()
        q = np.asarray(query_vector, dtype=np.float32).flatten()
        if term_vec.size != q.size or term_vec.size == 0:
            return 0.5
        cos_sim = float(np.dot(term_vec, q))
        cos_sim = max(-1.0, min(1.0, cos_sim))
        return 0.5 + 0.5 * max(0.0, cos_sim)
    except Exception:
        return 0.5


def retrieve_academic_term_by_similar_to(
    label,
    anchor: PreparedAnchor,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> List[LandingCandidate]:
    """Stage2A 落点：从锚点（industry）查跨类型 SIMILAR_TO → 学术词；仅保留与激活领域（及可选三级领域）一致的词。"""
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
    # 诊断：将本锚点命中的 similar_to 原始行写入 debug_info.similar_to_raw_rows（跨锚点累积）
    if rows and getattr(label, "debug_info", None) is not None:
        raw_list = getattr(label.debug_info, "similar_to_raw_rows", None)
        if raw_list is None:
            raw_list = []
            label.debug_info.similar_to_raw_rows = raw_list
        for r in rows:
            raw_list.append({
                "src_vid": anchor.vid,
                "tid": r.get("tid"),
                "term": r.get("term"),
                "sim_score": float(r.get("sim_score", 0.0) or 0.0),
            })
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
    if active_domain_set is not None or jd_field_ids or jd_subfield_ids or jd_topic_ids:
        n_before = len(out)
        kept = []
        dropped_with_reason = []
        for c in out:
            ok, reason = _term_in_active_domains_with_reason(
                label, c.vid,
                active_domain_set=active_domain_set,
                jd_field_ids=jd_field_ids,
                jd_subfield_ids=jd_subfield_ids,
                jd_topic_ids=jd_topic_ids,
            )
            if ok:
                kept.append(c)
                if reason and LABEL_EXPANSION_DEBUG:
                    print(f"[Stage2A] 保留（层级未命中）vid={c.vid} term={c.term!r} sim={c.semantic_score:.3f} reason={reason}")
            else:
                dropped_with_reason.append((c, reason))
        out = kept
        if LABEL_EXPANSION_DEBUG and n_before != len(out):
            print(f"[Stage2A] 领域过滤后 SIMILAR_TO 落点: {len(out)} 个（过滤前 {n_before}）")
            for c, reason in dropped_with_reason:
                print(f"[Stage2A]   领域过滤剔除 vid={c.vid} term={c.term!r} sim={c.semantic_score:.3f} 原因={reason}")
    if LABEL_EXPANSION_DEBUG and out:
        for i, c in enumerate(out[:5]):
            print(f"[Stage2A]   落点[{i}] vid={c.vid} term={c.term!r} sim={c.semantic_score:.3f}")
        if len(out) > 5:
            print(f"[Stage2A]   ... 共 {len(out)} 个落点候选")
    return out


def _retrieve_academic_terms_by_conditioned_vec(
    label,
    anchor: PreparedAnchor,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> List[LandingCandidate]:
    """Stage2A 召回时用 gte+JD 上下文：用锚点 conditioned_vec 在学术词索引中检索，仅返回 concept/keyword 且通过领域过滤的候选。"""
    if getattr(anchor, "conditioned_vec", None) is None:
        return []
    if not getattr(label, "vocab_index", None) or not getattr(label, "_vocab_meta", None):
        return []
    load_vocab_meta(label)
    try:
        vec = np.asarray(anchor.conditioned_vec, dtype=np.float32).flatten()
        if vec.size == 0:
            return []
        vec = vec.reshape(1, -1)
        faiss.normalize_L2(vec)
        k = min(CONDITIONED_VEC_TOP_K, getattr(label.vocab_index, "ntotal", 100))
        if k <= 0:
            return []
        scores, ids = label.vocab_index.search(vec, k)
    except Exception:
        return []
    out = []
    for score, tid in zip(scores[0], ids[0]):
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            continue
        if tid <= 0 or tid == getattr(anchor, "vid", -1):
            continue
        meta = label._vocab_meta.get(tid, ("", ""))
        if meta[1] not in ("concept", "keyword") and meta[1]:
            continue
        sim = max(0.0, min(1.0, float(score)))
        if sim < SIMILAR_TO_MIN_SCORE:
            continue
        if active_domain_set is not None or jd_field_ids or jd_subfield_ids or jd_topic_ids:
            if not _term_in_active_domains(
                label, tid,
                active_domain_set=active_domain_set,
                jd_field_ids=jd_field_ids,
                jd_subfield_ids=jd_subfield_ids,
                jd_topic_ids=jd_topic_ids,
            ):
                continue
        term = (meta[0] or "").strip() or str(tid)
        out.append(
            LandingCandidate(
                vid=tid,
                term=term,
                source="conditioned_vec",
                semantic_score=sim,
                anchor_vid=getattr(anchor, "vid", 0),
                anchor_term=getattr(anchor, "anchor", ""),
            )
        )
    if LABEL_EXPANSION_DEBUG and out:
        print(f"[Stage2A] conditioned_vec 检索 anchor={getattr(anchor, 'anchor', '')!r} -> {len(out)} 个学术词候选（与 SIMILAR_TO 合并）")
    return out


# ---------- Identity Gate：候选与锚点“本义”一致性，用于压制错义（propulsion/kinesics/simula 等） ----------
# 软闸门：identity_score -> gate 乘数，不硬删
IDENTITY_GATE_THRESHOLDS = [(0.75, 1.00), (0.55, 0.90), (0.35, 0.72)]  # (min_score, gate); else 0.45
# 错义/泛词惩罚：candidate 为这些词时 identity 压低（与锚点无稳定 lexical family 时不得过高）
IDENTITY_AMBIGUITY_TERMS = frozenset({
    "control", "robot", "robotics", "machine", "learning", "retrieval", "data", "crawling",
    "point", "point-to-point", "principle", "flow", "management", "digital", "automatic",
    "personal", "robot", "palo", "simula", "kinesics", "propulsion", "dynamism", "mechanics",
})
# 常见“锚点本义”英文对应（最小白名单，用于 boost；中文锚点可逐步加）
ANCHOR_IDENTITY_ALIASES: Dict[str, Set[str]] = {
    "动力学": {"dynamics", "dynamic", "mechanics", "kinetics"},
    "运动学": {"kinematics", "kinesiology"},
    "仿真": {"simulation", "simulate", "simulator"},
    "控制": {"control", "controller", "control engineering"},
    "运动控制": {"motion control", "movement control", "motion controller"},
    "机械臂": {"robotic arm", "robot arm", "manipulator"},
    "强化学习": {"reinforcement learning", "reinforcement", "rl"},
    "抓取": {"grasping", "grasp", "manipulation", "gripper"},
    "端到端": {"end-to-end", "end to end"},
}


def normalize_identity_surface(term: str) -> Dict[str, Any]:
    """
    轻量 identity 归一化，供 lexical identity 计算用。
    返回: raw, norm, tokens, token_set, head
    """
    if not isinstance(term, str):
        term = str(term or "")
    raw = term.strip()
    s = raw.lower().strip()
    # 去括号说明：control (management) -> control
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    # 连字符统一成空格：end-to-end -> end to end
    s = re.sub(r"-", " ", s)
    # 多空格压缩
    s = re.sub(r"\s+", " ", s).strip()
    norm = s
    tokens = [t for t in s.split() if t]
    token_set = set(tokens)
    head = tokens[0] if tokens else ""
    return {"raw": raw, "norm": norm, "tokens": tokens, "token_set": token_set, "head": head}


def compute_anchor_identity_score(
    anchor_term: str,
    candidate_term: str,
    anchor_type: Optional[str] = None,
) -> float:
    """
    候选与锚点是否“本义同一概念家族”的 0~1 分。
    用于压制：动力学->propulsion、运动学->kinesics、仿真->simula、抓取->Data retrieval、机械臂->Robot control 等错义。
    """
    a = normalize_identity_surface(anchor_term)
    c = normalize_identity_surface(candidate_term)
    atok = a["token_set"]
    ctok = c["token_set"]
    a_norm = a["norm"]
    c_norm = c["norm"]

    # 信号 1：完全词面/子串一致
    exact_or_substring = 0.0
    if a_norm and c_norm:
        if a_norm == c_norm:
            exact_or_substring = 1.0
        elif a_norm in c_norm or c_norm in a_norm:
            exact_or_substring = 0.85
        elif atok == ctok:
            exact_or_substring = 0.9

    # 信号 2：token overlap + 锚点英文别名字族
    inter = atok & ctok
    union = atok | ctok
    token_overlap_score = len(inter) / len(union) if union else 0.0
    for _anchor, aliases in ANCHOR_IDENTITY_ALIASES.items():
        an_norm = normalize_identity_surface(_anchor)["norm"]
        if an_norm and a_norm and (an_norm == a_norm or normalize_identity_surface(_anchor)["token_set"] == atok):
            for al in aliases:
                al_norm = normalize_identity_surface(al)["norm"]
                if al_norm and (al_norm == c_norm or al_norm in c_norm or c_norm in al_norm):
                    token_overlap_score = max(token_overlap_score, 0.7)
                    break
            break
    if exact_or_substring >= 0.85:
        token_overlap_score = max(token_overlap_score, 0.9)

    # 信号 3：head 一致性
    head_consistency_score = 0.0
    if a["head"] and a["head"] in ctok:
        head_consistency_score = 0.8
    elif c["head"] and c["head"] in atok:
        head_consistency_score = 0.6
    if exact_or_substring >= 0.85:
        head_consistency_score = max(head_consistency_score, 0.85)

    score = (
        0.45 * exact_or_substring
        + 0.35 * token_overlap_score
        + 0.20 * head_consistency_score
    )

    # 泛词/歧义惩罚
    generic_penalty = 1.0
    if ctok and len(ctok) <= 2:
        low_tokens = {t.lower() for t in c["tokens"]}
        if low_tokens & IDENTITY_AMBIGUITY_TERMS and not (atok & ctok):
            generic_penalty = 0.5
    ambiguity_penalty = 1.0
    c_head_lower = c["head"].lower() if c["head"] else ""
    if c_head_lower in IDENTITY_AMBIGUITY_TERMS and c_head_lower not in atok:
        ambiguity_penalty = 0.6
    if (c_norm in ("control (management)", "control flow", "data retrieval", "crawling",
                   "point-to-point", "end-to-end principle", "simula", "kinesics", "propulsion") and
            not (atok & ctok)):
        ambiguity_penalty = min(ambiguity_penalty, 0.35)

    score *= generic_penalty * ambiguity_penalty
    return max(0.0, min(1.0, score))


def _identity_gate_from_score(anchor_identity_score: float) -> float:
    """软闸门：0.75+ -> 1.0, 0.55+ -> 0.9, 0.35+ -> 0.72, else 0.45"""
    for thresh, gate in IDENTITY_GATE_THRESHOLDS:
        if anchor_identity_score >= thresh:
            return gate
    return 0.45


def collect_landing_candidates(
    label,
    anchor: PreparedAnchor,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
    jd_profile: Optional[Dict[str, Any]] = None,
    query_vector=None,
) -> List[LandingCandidate]:
    """Stage2A：跨类型 SIMILAR_TO + 召回时 gte+JD 上下文（conditioned_vec 检索学术词）；有 jd_profile 时做层级 fit 与 landing 打分。"""
    similar_list = retrieve_academic_term_by_similar_to(
        label, anchor,
        active_domain_set=active_domain_set,
        jd_field_ids=jd_field_ids,
        jd_subfield_ids=jd_subfield_ids,
        jd_topic_ids=jd_topic_ids,
    )
    by_vid = {c.vid: c for c in similar_list}
    # 召回时用 gte+JD 上下文：conditioned_vec 检索学术词，与 SIMILAR_TO 合并（同 vid 保留 SIMILAR_TO）
    ctx_list = _retrieve_academic_terms_by_conditioned_vec(
        label, anchor,
        active_domain_set=active_domain_set,
        jd_field_ids=jd_field_ids,
        jd_subfield_ids=jd_subfield_ids,
        jd_topic_ids=jd_topic_ids,
    )
    for c in ctx_list:
        if c.vid not in by_vid:
            by_vid[c.vid] = c
    cands = list(by_vid.values())
    for c in cands:
        c.domain_fit = _compute_domain_fit(
            label, c.vid,
            active_domain_set=active_domain_set,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
        )
    # 在 candidate 上保留层级状态，供 primary 打分惩罚（topic=1.0, subfield=0.65, field=0.35, none=0.10）
    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))
    for c in cands:
        if not jd_f and not jd_s and not jd_t:
            setattr(c, "hierarchy_score", 1.0)
            setattr(c, "hierarchy_level", "missing")
            setattr(c, "hierarchy_reason", "")
        else:
            topic_row = _load_vocabulary_topic_stats(label, c.vid)
            if not topic_row:
                setattr(c, "hierarchy_score", 1.0)
                setattr(c, "hierarchy_level", "missing")
                setattr(c, "hierarchy_reason", "")
            else:
                score, level = _compute_hierarchy_match_score(topic_row, jd_f, jd_s, jd_t)
                setattr(c, "hierarchy_score", score)
                setattr(c, "hierarchy_level", level)
                setattr(c, "hierarchy_reason", "topic_hierarchy_no_match" if score <= 0 else "")
    if jd_profile:
        jd_profile_for_fit = {k: v for k, v in jd_profile.items() if k != "active_domains"}
        jd_profile_for_fit["active_subfields"] = set(jd_profile.get("active_subfields") or [])
        jd_profile_for_fit["active_topics"] = set(jd_profile.get("active_topics") or [])
        filtered = []
        for c in cands:
            snap = get_vocab_hierarchy_snapshot(label, c.vid)
            term_info = {
                "field_dist": snap.get("field_dist") or {},
                "subfield_dist": snap.get("subfield_dist") or {},
                "topic_dist": snap.get("topic_dist") or {},
                "domain_dist": snap.get("domain_dist") or {},
            }
            fit_info = compute_hierarchical_fit(term_info, jd_profile_for_fit)
            setattr(c, "fit_info", fit_info)
            setattr(c, "work_count", snap.get("work_count") or 0)
            setattr(c, "domain_span", snap.get("domain_span") or 0)
            setattr(c, "subfield_fit", fit_info.get("subfield_fit", 0))
            setattr(c, "topic_fit", fit_info.get("topic_fit", 0))
            setattr(c, "outside_subfield_mass", fit_info.get("outside_subfield_mass", 0))
            setattr(c, "outside_topic_mass", fit_info.get("outside_topic_mass", 0))
            setattr(c, "topic_entropy", 0.0)
            fit_info["subfield_dist"] = snap.get("subfield_dist") or {}
            fit_info["topic_dist"] = snap.get("topic_dist") or {}
            # 内联 landing 排序分：仅用于 top-m 排序，不依赖 hierarchy_guard 旧体系
            df = fit_info.get("domain_fit") or 0.5
            span = int(snap.get("domain_span") or 0)
            genericity = 1.0 / (1.0 + TOPIC_SPAN_PENALTY_FACTOR * max(0, span - 1))
            land_score = (c.semantic_score or 0) * (0.5 + 0.5 * df) * genericity
            setattr(c, "landing_score", land_score)
            # 零硬编码：不再用 subfield_fit/topic_fit/outside_subfield_mass 硬门槛一票否决；
            # 层级信息仅参与 landing_score，最终由 primary_score（含 jd_align、neighborhood、isolation）统一排序
            filtered.append(c)
        cands = filtered
    # 数据驱动：候选与 JD 整体语义对齐（无词表）
    if query_vector is not None:
        for c in cands:
            jd_align = _compute_jd_candidate_alignment(label, c.vid, query_vector)
            setattr(c, "jd_candidate_alignment", jd_align)
    else:
        for c in cands:
            setattr(c, "jd_candidate_alignment", 0.5)
    # Identity Gate：候选与锚点本义一致性，软闸门 + 超低 identity 的 conditioned 候选降为 secondary_only
    anchor_term = getattr(anchor, "anchor", "") or ""
    anchor_type_opt = getattr(anchor, "anchor_type", None)
    for c in cands:
        aid = compute_anchor_identity_score(anchor_term, c.term or "", anchor_type_opt)
        setattr(c, "anchor_identity_score", aid)
        setattr(c, "identity_gate", _identity_gate_from_score(aid))
        if (getattr(c, "source", "") or "").strip().lower() == "conditioned_vec" and aid < 0.30:
            setattr(c, "primary_cap", "secondary_only")
        else:
            setattr(c, "primary_cap", None)
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2A] collect_landing_candidates anchor={anchor.anchor!r} -> {len(cands)} 个候选")
        print("[Stage2A 候选明细] tid | term | source | semantic_score | domain_fit | anchor_identity | identity_gate | landing_score | jd_align")
        for i, c in enumerate(cands[:25]):
            sem = getattr(c, "semantic_score", 0)
            df = getattr(c, "domain_fit", 1.0)
            aid = getattr(c, "anchor_identity_score", 0.5)
            gate = getattr(c, "identity_gate", 1.0)
            land = getattr(c, "landing_score", None)
            jd_a = getattr(c, "jd_candidate_alignment", None)
            cap = getattr(c, "primary_cap", None)
            cap_s = f" [{cap}]" if cap else ""
            print(f"  {i+1} {c.vid} | {c.term!r} | {c.source} | sem={sem:.3f} | df={df:.3f} | identity={aid:.3f} | gate={gate:.2f}{cap_s} | landing={land} | jd_align={jd_a}")
        if len(cands) > 25:
            print(f"[stage2a_candidates] ... 共 {len(cands)} 条")
    return cands


def _compute_neighborhood_and_isolation(label, flat_pool: List[Tuple[Any, LandingCandidate]]) -> None:
    """数据驱动：为每个候选计算与其它候选的邻域一致性及语义离群惩罚，无词表。原地写入 c.neighborhood_consistency、c.semantic_isolation_penalty。"""
    if not flat_pool or getattr(label, "vocab_to_idx", None) is None or getattr(label, "all_vocab_vectors", None) is None:
        for _, c in flat_pool:
            setattr(c, "neighborhood_consistency", 0.5)
            setattr(c, "semantic_isolation_penalty", 0.0)
        return
    vecs = {}
    for _, c in flat_pool:
        vid = c.vid
        if vid in vecs:
            continue
        idx = label.vocab_to_idx.get(str(vid))
        if idx is None:
            vecs[vid] = None
            continue
        try:
            v = np.asarray(label.all_vocab_vectors[idx], dtype=np.float32).flatten()
            vecs[vid] = v
        except Exception:
            vecs[vid] = None
    for _, c in flat_pool:
        v = vecs.get(c.vid)
        if v is None or v.size == 0:
            setattr(c, "neighborhood_consistency", 0.5)
            setattr(c, "semantic_isolation_penalty", 0.0)
            continue
        sims = []
        for (_, c2) in flat_pool:
            if c2.vid == c.vid:
                continue
            v2 = vecs.get(c2.vid)
            if v2 is None or v2.size != v.size:
                continue
            try:
                s = float(np.dot(v, v2))
                s = max(-1.0, min(1.0, s))
                sims.append(s)
            except Exception:
                pass
        if not sims:
            setattr(c, "neighborhood_consistency", 0.5)
            setattr(c, "semantic_isolation_penalty", 0.0)
            continue
        mean_sim = float(np.mean(sims))
        mean_sim = max(-1.0, min(1.0, mean_sim))
        setattr(c, "neighborhood_consistency", max(0.0, mean_sim))
        setattr(c, "semantic_isolation_penalty", max(0.0, 1.0 - mean_sim))


def _compute_conditioned_anchor_align_and_multi_anchor_support(
    label,
    flat_pool: List[Tuple[Any, LandingCandidate]],
    prepared_anchors: List[PreparedAnchor],
) -> None:
    """
    为每个候选设置 conditioned_anchor_align（与当前锚点条件化向量相似度）与 multi_anchor_support
    （与其它锚点条件化向量平均相似度）。无词表，纯向量与图结构。
    """
    if getattr(label, "vocab_to_idx", None) is None or getattr(label, "all_vocab_vectors", None) is None:
        for _, c in flat_pool:
            setattr(c, "conditioned_anchor_align", None)
            setattr(c, "multi_anchor_support", 0.5)
        return
    # 候选向量缓存
    vecs: Dict[int, Optional[np.ndarray]] = {}
    for _, c in flat_pool:
        if c.vid in vecs:
            continue
        idx = label.vocab_to_idx.get(str(c.vid))
        if idx is None:
            vecs[c.vid] = None
            continue
        try:
            v = np.asarray(label.all_vocab_vectors[idx], dtype=np.float32).flatten()
            vecs[c.vid] = v
        except Exception:
            vecs[c.vid] = None
    # 其它锚点条件化向量（按 vid）
    other_conditioned: Dict[int, np.ndarray] = {}
    for a in prepared_anchors:
        if a.conditioned_vec is None:
            continue
        try:
            v = np.asarray(a.conditioned_vec, dtype=np.float32).flatten()
            if v.size > 0:
                other_conditioned[a.vid] = v
        except Exception:
            pass
    for anchor, c in flat_pool:
        c_vec = vecs.get(c.vid)
        if c_vec is None or c_vec.size == 0:
            setattr(c, "conditioned_anchor_align", None)
            setattr(c, "multi_anchor_support", 0.5)
            continue
        # 当前锚点条件化对齐
        cond_align = None
        if getattr(anchor, "conditioned_vec", None) is not None:
            try:
                cv = np.asarray(anchor.conditioned_vec, dtype=np.float32).flatten()
                if cv.size == c_vec.size:
                    sim = float(np.dot(c_vec, cv))
                    sim = max(-1.0, min(1.0, sim))
                    cond_align = 0.5 + 0.5 * max(0.0, sim)
            except Exception:
                pass
        setattr(c, "conditioned_anchor_align", cond_align)
        # 多锚支持：与其它锚点条件化向量的平均相似度
        support_sims = []
        for a2_vid, a2_vec in other_conditioned.items():
            if a2_vid == anchor.vid:
                continue
            if a2_vec.size != c_vec.size:
                continue
            try:
                s = float(np.dot(c_vec, a2_vec))
                s = max(-1.0, min(1.0, s))
                support_sims.append(0.5 + 0.5 * max(0.0, s))
            except Exception:
                pass
        multi_support = float(np.mean(support_sims)) if support_sims else 0.5
        setattr(c, "multi_anchor_support", max(0.0, min(1.0, multi_support)))


def score_academic_identity(c: LandingCandidate) -> float:
    """身份分：当前 Stage2A 仅 similar_to，用边权；若未来接入 jd_vector 则用 0.5+0.5*semantic_score。"""
    if c.source == "similar_to":
        return max(0.0, min(1.0, c.semantic_score))
    if c.source == "jd_vector":
        return 0.5 + 0.5 * max(0.0, min(1.0, c.semantic_score))
    return 0.5 + 0.5 * max(0.0, min(1.0, c.semantic_score))


# ---------- Stage2B：学术侧补充（dense / 簇 / 共现，不再用 SIMILAR_TO 学术→学术） ----------


def expand_from_vocab_dense_neighbors(
    label,
    primary_landings: List[PrimaryLanding],
    top_k_per_primary: int = None,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
    jd_profile: Optional[Dict[str, Any]] = None,
) -> List[ExpandedTermCandidate]:
    """从词汇向量索引取 primary 的学术近邻；入口先审 seed，支撑词再过滤语义偏航。term_role=dense_expansion。"""
    top_k_per_primary = top_k_per_primary or DENSE_MAX_PER_PRIMARY
    if not primary_landings or not getattr(label, "vocab_index", None) or not getattr(label, "vocab_to_idx", None):
        return []
    if getattr(label, "all_vocab_vectors", None) is None:
        return []
    load_vocab_meta(label)
    seen = set(p.vid for p in primary_landings)
    out = []
    for p in primary_landings:
        ok, _, _ = check_seed_eligibility(label, p, jd_profile)
        if not ok:
            continue
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
            if active_domain_set is not None or jd_field_ids or jd_subfield_ids or jd_topic_ids:
                if not _term_in_active_domains(
                    label, tid,
                    active_domain_set=active_domain_set,
                    jd_field_ids=jd_field_ids,
                    jd_subfield_ids=jd_subfield_ids,
                    jd_topic_ids=jd_topic_ids,
                ):
                    continue
            domain_fit = _compute_domain_fit(
                label, tid,
                active_domain_set=active_domain_set,
                jd_field_ids=jd_field_ids,
                jd_subfield_ids=jd_subfield_ids,
                jd_topic_ids=jd_topic_ids,
            )
            if domain_fit < SUPPORT_MIN_DOMAIN_FIT:
                continue
            if _is_bad_support_for_anchor(getattr(p, "anchor_term", "") or "", meta[0] or ""):
                continue
            domain_span = 0
            if getattr(label, "stats_conn", None):
                row = label.stats_conn.execute(
                    "SELECT domain_span FROM vocabulary_domain_stats WHERE voc_id=?",
                    (tid,),
                ).fetchone()
                if row:
                    domain_span = int(row[0] or 0)
            if domain_span > DOMAIN_SPAN_EXTREME:
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
                    parent_primary=p.term,
                )
            )
            count += 1
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2B] expand_from_vocab_dense_neighbors primary数={len(primary_landings)} -> dense_expansion {len(out)} 个")
        for i, c in enumerate(out[:8]):
            print(f"[stage2b_expanded] tid={c.vid} term={c.term!r} source_type={c.source} parent_anchor={c.anchor_term!r} parent_primary={getattr(c, 'parent_primary', '')!r} score={c.identity_score:.3f}")
        if len(out) > 8:
            print(f"[stage2b_expanded] ... dense 共 {len(out)} 条")
    return out


def expand_from_cluster_members(
    label,
    primary_landings: List[PrimaryLanding],
    max_per_primary: int = None,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> List[ExpandedTermCandidate]:
    """从簇内取同簇支持词；仅保留与激活领域（及可选三级领域）一致的词。term_role=cluster_expansion。"""
    max_per_primary = max_per_primary or CLUSTER_MAX_PER_PRIMARY
    voc_to_clusters = getattr(label, "voc_to_clusters", None) or {}
    cluster_members = getattr(label, "cluster_members", None) or {}
    if not primary_landings or not voc_to_clusters or not cluster_members:
        return []
    load_vocab_meta(label)
    seen = set(p.vid for p in primary_landings)
    out = []
    for p in primary_landings:
        if getattr(p, "retain_mode", "normal") != "normal":
            continue
        if (getattr(p, "topic_source", "missing") or "missing") == "cooc":
            continue
        if (getattr(p, "primary_score", 0) or 0) < 0.40:
            continue
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
            if active_domain_set is not None or jd_field_ids or jd_subfield_ids or jd_topic_ids:
                if not _term_in_active_domains(
                    label, vid,
                    active_domain_set=active_domain_set,
                    jd_field_ids=jd_field_ids,
                    jd_subfield_ids=jd_subfield_ids,
                    jd_topic_ids=jd_topic_ids,
                ):
                    continue
            domain_fit = _compute_domain_fit(
                label, vid,
                active_domain_set=active_domain_set,
                jd_field_ids=jd_field_ids,
                jd_subfield_ids=jd_subfield_ids,
                jd_topic_ids=jd_topic_ids,
            )
            if domain_fit < SUPPORT_MIN_DOMAIN_FIT:
                continue
            domain_span = 0
            if getattr(label, "stats_conn", None):
                row = label.stats_conn.execute(
                    "SELECT domain_span FROM vocabulary_domain_stats WHERE voc_id=?",
                    (vid,),
                ).fetchone()
                if row:
                    domain_span = int(row[0] or 0)
            if domain_span > DOMAIN_SPAN_EXTREME:
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
                    parent_primary=p.term,
                )
            )
            count += 1
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2B] expand_from_cluster_members primary数={len(primary_landings)} -> cluster_expansion {len(out)} 个")
    return out


def expand_from_cooccurrence_support(
    label,
    primary_landings: List[PrimaryLanding],
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> List[ExpandedTermCandidate]:
    """共现高支持度学术词；仅允许 strong normal + direct 且 primary_score>=0.45 的 seed。term_role=cooc_expansion。"""
    if not primary_landings or not getattr(label, "stats_conn", None):
        return []
    load_vocab_meta(label)
    term_to_vid = {}
    if getattr(label, "_vocab_meta", None):
        for v, (t, _) in label._vocab_meta.items():
            if t and t not in term_to_vid:
                term_to_vid[t.strip()] = v
    strong_primaries = [
        p for p in primary_landings
        if getattr(p, "retain_mode", "normal") == "normal"
        and (getattr(p, "topic_source", "missing") or "missing") == "direct"
        and (getattr(p, "primary_score", 0) or 0) >= 0.45
    ]
    vid_to_term = {p.vid: (p.term, p.anchor_vid, p.anchor_term) for p in strong_primaries}
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
            if active_domain_set is not None or jd_field_ids or jd_subfield_ids or jd_topic_ids:
                if not _term_in_active_domains(
                    label, vid_other,
                    active_domain_set=active_domain_set,
                    jd_field_ids=jd_field_ids,
                    jd_subfield_ids=jd_subfield_ids,
                    jd_topic_ids=jd_topic_ids,
                ):
                    continue
            domain_fit = _compute_domain_fit(
                label, vid_other,
                active_domain_set=active_domain_set,
                jd_field_ids=jd_field_ids,
                jd_subfield_ids=jd_subfield_ids,
                jd_topic_ids=jd_topic_ids,
            )
            if domain_fit < SUPPORT_MIN_DOMAIN_FIT:
                continue
            domain_span = 0
            if getattr(label, "stats_conn", None):
                row = label.stats_conn.execute(
                    "SELECT domain_span FROM vocabulary_domain_stats WHERE voc_id=?",
                    (vid_other,),
                ).fetchone()
                if row:
                    domain_span = int(row[0] or 0)
            if domain_span > DOMAIN_SPAN_EXTREME:
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
                    parent_primary=term,
                )
            )
            count += 1
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2B] expand_from_cooccurrence_support primary数={len(primary_landings)} -> cooc_expansion {len(out)} 个")
    return out


# ---------- 领域/三级领域过滤：Stage2A/2B 仅保留与当前查询领域一致的词 ----------


def _term_in_active_domains(
    label,
    vid: int,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> bool:
    """
    判断词汇 vid 是否落在当前激活领域（及可选的三级领域）内，供 Stage2A/2B 过滤用。
    - 无 active_domain_set 时不做领域过滤，返回 True。
    - 有 active_domain_set 时查 vocabulary_domain_stats，要求 term 的 domain_dist 与激活领域有交集。
    - 若提供了 jd_field_ids/jd_subfield_ids/jd_topic_ids，则同时要求 vocabulary_topic_stats 至少 field 级命中。
    """
    active = set(int(x) for x in (active_domain_set or [])) if active_domain_set else set()
    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))

    if not active:
        domain_ok = True
    else:
        if not getattr(label, "stats_conn", None):
            domain_ok = True
        else:
            row = None
            try:
                row = label.stats_conn.execute(
                    "SELECT domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                    (vid,),
                ).fetchone()
            except Exception:
                domain_ok = True
            if not row or not row[0]:
                domain_ok = True
            else:
                try:
                    dist = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                except Exception:
                    dist = {}
                expanded = expand_domain_dist(label, dist or {})
                active_str = set(str(d) for d in active)
                term_domains = set(expanded.keys()) if expanded else set()
                domain_ok = bool(active_str & term_domains)
                # 主领域守卫：主领域（权重最大）必须在 active_domains 内，否则一票否决
                if domain_ok and expanded:
                    main_domain = max(expanded, key=expanded.get)
                    if main_domain not in active_str:
                        domain_ok = False

    if not domain_ok:
        return False
    if not jd_f and not jd_s and not jd_t:
        return True
    topic_row = _load_vocabulary_topic_stats(label, vid)
    if not topic_row:
        return True
    score, _ = _compute_hierarchy_match_score(topic_row, jd_f, jd_s, jd_t)
    # 与 _term_in_active_domains_with_reason 一致：层级不对齐不再一票否决，交由 primary 惩罚
    return True


def _term_in_active_domains_with_reason(
    label,
    vid: int,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    """
    同 _term_in_active_domains，但返回 (是否通过, 原因描述)。
    通过时 reason 为空或 "topic_hierarchy_no_match"（层级未命中仍保留进池，供日志与 primary 惩罚）；
    未通过时仅 "domain_no_match"。
    """
    active = set(int(x) for x in (active_domain_set or [])) if active_domain_set else set()
    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))

    if not active:
        domain_ok = True
    else:
        if not getattr(label, "stats_conn", None):
            domain_ok = True
        else:
            row = None
            try:
                row = label.stats_conn.execute(
                    "SELECT domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                    (vid,),
                ).fetchone()
            except Exception:
                domain_ok = True
            if not row or not row[0]:
                domain_ok = True
            else:
                try:
                    dist = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                except Exception:
                    dist = {}
                expanded = expand_domain_dist(label, dist or {})
                active_str = set(str(d) for d in active)
                term_domains = set(expanded.keys()) if expanded else set()
                domain_ok = bool(active_str & term_domains)
                if domain_ok and expanded:
                    main_domain = max(expanded, key=expanded.get)
                    if main_domain not in active_str:
                        domain_ok = False

    if not domain_ok:
        return (False, "domain_no_match")
    if not jd_f and not jd_s and not jd_t:
        return (True, "")
    topic_row = _load_vocabulary_topic_stats(label, vid)
    if not topic_row:
        return (True, "")
    score, level = _compute_hierarchy_match_score(topic_row, jd_f, jd_s, jd_t)
    # 不再在此处一票否决：topic_hierarchy_no_match 的候选保留进池，仅打上 reason 供日志与后续 primary 惩罚
    if score <= 0:
        return (True, "topic_hierarchy_no_match")
    return (True, "")


def _compute_domain_fit(
    label,
    vid: int,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> float:
    """
    为候选 term 计算领域拟合分，用于 Stage2A primary 门控与 Stage3 domain_gate。
    domain_fit = 0.4*domain_overlap + 0.3*field_overlap + 0.2*subfield_overlap + 0.1*topic_overlap。
    无表/无 JD 层级时对应 overlap 视为 1.0。
    """
    w_d, w_f, w_s, w_t = (0.4, 0.3, 0.2, 0.1)
    active = set(int(x) for x in (active_domain_set or [])) if active_domain_set else set()
    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))

    domain_overlap = 1.0
    if active and getattr(label, "stats_conn", None):
        try:
            row = label.stats_conn.execute(
                "SELECT domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (vid,),
            ).fetchone()
            if row and row[0]:
                dist = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                expanded = expand_domain_dist(label, dist or {})
                total = sum(expanded.values())
                if total > 0:
                    in_active = sum(expanded.get(str(d), 0) for d in active)
                    domain_overlap = in_active / total
                else:
                    domain_overlap = 1.0 if expanded and set(expanded.keys()) & {str(d) for d in active} else 0.0
        except Exception:
            pass

    field_overlap = 1.0
    subfield_overlap = 1.0
    topic_overlap = 1.0
    topic_row = _load_vocabulary_topic_stats(label, vid)
    if topic_row and (jd_f or jd_s or jd_t):
        term_f = _extract_ids_from_row(topic_row, "field_id", "field_dist")
        term_s = _extract_ids_from_row(topic_row, "subfield_id", "subfield_dist")
        term_t = _extract_ids_from_row(topic_row, "topic_id", "topic_dist")

        def _overlap(a: Set[str], b: Set[str]) -> float:
            if not b:
                return 1.0
            return 1.0 if (a & b) else 0.0

        field_overlap = _overlap(term_f, jd_f)
        subfield_overlap = _overlap(term_s, jd_s)
        topic_overlap = _overlap(term_t, jd_t)

    return float(
        w_d * domain_overlap
        + w_f * field_overlap
        + w_s * subfield_overlap
        + w_t * topic_overlap
    )


# ---------- 层级守卫：vocabulary 快照与层级 fit（仅标签路） ----------

_hierarchy_snapshot_cache: Dict[int, Dict[str, Any]] = {}


def get_vocab_hierarchy_snapshot(label, voc_id: int) -> Dict[str, Any]:
    """
    统一读取 vocabulary_topic_stats + vocabulary_domain_stats 为层级快照。
    返回 field_id, subfield_id, topic_id, field_dist, subfield_dist, topic_dist, domain_dist, domain_span, work_count（已解析）。
    """
    global _hierarchy_snapshot_cache
    if voc_id in _hierarchy_snapshot_cache:
        return _hierarchy_snapshot_cache[voc_id]
    out = {
        "voc_id": voc_id,
        "field_id": None,
        "subfield_id": None,
        "topic_id": None,
        "field_dist": {},
        "subfield_dist": {},
        "topic_dist": {},
        "domain_dist": {},
        "domain_span": 0,
        "work_count": 0,
    }
    if not getattr(label, "stats_conn", None):
        return out
    try:
        row_t = label.stats_conn.execute(
            "SELECT field_id, subfield_id, topic_id, field_dist, subfield_dist, topic_dist FROM vocabulary_topic_stats WHERE voc_id=?",
            (voc_id,),
        ).fetchone()
        if row_t:
            out["field_id"] = row_t[0]
            out["subfield_id"] = row_t[1]
            out["topic_id"] = row_t[2]
            out["field_dist"] = _parse_dist(row_t[3])
            out["subfield_dist"] = _parse_dist(row_t[4])
            out["topic_dist"] = _parse_dist(row_t[5])
        row_d = label.stats_conn.execute(
            "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
            (voc_id,),
        ).fetchone()
        if row_d:
            out["work_count"] = int(row_d[0] or 0)
            out["domain_span"] = int(row_d[1] or 0)
            out["domain_dist"] = _parse_dist(row_d[2])
    except Exception:
        pass
    _hierarchy_snapshot_cache[voc_id] = out
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
    """层级命中分与档位（仅用于 debug/explain）。准入以 compute_hierarchy_evidence 的连续 effective_* 为准，不做硬档位拒绝。"""
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


def _overlap_ratio(cand_set: Set[str], active_set: Set[str]) -> float:
    """JD 激活集合被候选覆盖的比例：|cand & active| / |active|，active 空时返回 0。"""
    if not active_set:
        return 0.0
    inter = len(cand_set & active_set)
    return inter / len(active_set) if inter else 0.0


def _compute_path_match(
    cand_fields: Set[str],
    cand_subfields: Set[str],
    cand_topics: Set[str],
    active_fields: Set[str],
    active_subfields: Set[str],
    active_topics: Set[str],
) -> float:
    """
    三层路径一致性：加权平均，避免几何平均强连坐导致 topic 不完整时 path 塌成 0.03。
    权重 0.2*field + 0.3*subfield + 0.5*topic；某层 active 为空则跳过并重新归一化。
    """
    w_field, w_sub, w_topic = 0.2, 0.3, 0.5
    total_w = 0.0
    score = 0.0
    if active_fields:
        score += w_field * (_overlap_ratio(cand_fields, active_fields) or 0.0)
        total_w += w_field
    if active_subfields:
        score += w_sub * (_overlap_ratio(cand_subfields, active_subfields) or 0.0)
        total_w += w_sub
    if active_topics:
        score += w_topic * (_overlap_ratio(cand_topics, active_topics) or 0.0)
        total_w += w_topic
    if total_w <= 0:
        return 0.0
    return score / total_w


def compute_hierarchy_evidence(
    label,
    voc_id: int,
    active_fields: Set[str],
    active_subfields: Set[str],
    active_topics: Set[str],
) -> Dict[str, Any]:
    """
    三层领域特征 + 来源可信度。准入/裁判使用 effective_*（overlap * hierarchy_reliability），
    cooc 补出的三级领域不再与 direct 同权。
    返回含 topic_source, hierarchy_reliability, effective_topic_overlap, effective_subfield_overlap, effective_path_match。
    """
    snap = get_vocab_hierarchy_snapshot(label, voc_id)
    topic_row = _load_vocabulary_topic_stats(label, voc_id) or {}
    topic_source = (topic_row.get("source") or "missing").strip().lower()
    if topic_source == "direct":
        hierarchy_reliability = 1.0
    elif topic_source == "direct+cooc":
        hierarchy_reliability = 0.8
    elif topic_source == "cooc":
        hierarchy_reliability = 0.5
    else:
        hierarchy_reliability = 0.4

    cand_fields = _extract_ids_from_row(
        {"field_id": snap.get("field_id"), "field_dist": snap.get("field_dist") or {}},
        "field_id",
        "field_dist",
    )
    cand_subfields = _extract_ids_from_row(
        {"subfield_id": snap.get("subfield_id"), "subfield_dist": snap.get("subfield_dist") or {}},
        "subfield_id",
        "subfield_dist",
    )
    cand_topics = _extract_ids_from_row(
        {"topic_id": snap.get("topic_id"), "topic_dist": snap.get("topic_dist") or {}},
        "topic_id",
        "topic_dist",
    )
    field_overlap = _overlap_ratio(cand_fields, active_fields) if active_fields else 0.0
    subfield_overlap = _overlap_ratio(cand_subfields, active_subfields) if active_subfields else 0.0
    topic_overlap = _overlap_ratio(cand_topics, active_topics) if active_topics else 0.0
    path_match = _compute_path_match(
        cand_fields, cand_subfields, cand_topics,
        active_fields, active_subfields, active_topics,
    )
    effective_topic = topic_overlap * hierarchy_reliability
    effective_subfield = subfield_overlap * hierarchy_reliability
    effective_path = path_match * hierarchy_reliability

    topic_dist = _parse_dist(snap.get("topic_dist")) or _parse_dist(topic_row.get("topic_dist"))
    topic_specificity = compute_purity(topic_dist) if topic_dist else 0.0
    domain_span = int(snap.get("domain_span") or 0)
    topic_span_penalty = 1.0 / (1.0 + TOPIC_SPAN_PENALTY_FACTOR * max(0, domain_span - 1))
    if topic_overlap >= 0.5 and active_topics:
        hierarchy_level = HIERARCHY_LEVEL_TOPIC_EXACT
    elif subfield_overlap >= 0.35 and active_subfields:
        hierarchy_level = HIERARCHY_LEVEL_SUBFIELD_MATCH
    elif field_overlap >= 0.2 and active_fields:
        hierarchy_level = HIERARCHY_LEVEL_FIELD_ONLY
    else:
        hierarchy_level = HIERARCHY_LEVEL_OFF_PATH
    return {
        "field_overlap": field_overlap,
        "subfield_overlap": subfield_overlap,
        "topic_overlap": topic_overlap,
        "path_match": path_match,
        "topic_specificity": topic_specificity,
        "topic_span_penalty": topic_span_penalty,
        "hierarchy_level": hierarchy_level,
        "topic_source": topic_source,
        "hierarchy_reliability": hierarchy_reliability,
        "effective_topic_overlap": effective_topic,
        "effective_subfield_overlap": effective_subfield,
        "effective_path_match": effective_path,
    }


def _lexical_term_sanity(term: str, meta: Any) -> bool:
    """术语合法性：非空、非纯数字、长度合理。"""
    if not term or not str(term).strip():
        return False
    t = str(term).strip()
    if t.isdigit():
        return False
    if len(t) > 120:
        return False
    return True


# ---------- 边界保留与禁扩（正确但 hierarchy/path 弱 → 保留降权；错误/窄义 → 禁扩或剔除） ----------

# 工程主干锚点/术语关键词，用于 should_retain_borderline_candidate 的 engineering_core 判断
_ENGINEERING_CORE_ANCHORS = frozenset({
    "控制", "运动控制", "传统控制", "路径规划", "强化学习", "仿真", "机器人", "机器人运动控制",
    "医疗机器人", "动力学", "动力学参数辨识", "抓取", "端到端", "机械臂", "robotics", "robot control",
    "motion control", "reinforcement learning", "simulation", "pathfinding", "automatic control",
})
_ENGINEERING_CORE_TERM_SUBSTR = frozenset({
    "control", "motion", "robot", "learning", "simulation", "path", "planning", "identification",
    "dynamics", "manipulator", "robotics", "reinforcement", "digital control", "feedback control",
})


def _is_engineering_core_anchor(anchor_text: str) -> bool:
    """锚点是否为控制/规划/学习/仿真/机器人等工程主干。"""
    if not anchor_text:
        return False
    a = (anchor_text or "").strip().lower()
    return any(k in a or a in k for k in _ENGINEERING_CORE_ANCHORS)


def _is_engineering_core_term(term_text: str) -> bool:
    """候选是否为明显工程术语（关键词子串）。"""
    if not term_text:
        return False
    t = (term_text or "").strip().lower()
    return any(k in t for k in _ENGINEERING_CORE_TERM_SUBSTR)


def should_retain_borderline_candidate(
    anchor_text: str,
    candidate_text: str,
    semantic_score: float,
    jd_align: float,
    anchor_identity: float,
    hierarchy_level: str,
    topic_overlap: float,
    subfield_overlap: float,
    path_match: float,
    source_type: str,
) -> Tuple[bool, Optional[str]]:
    """
    当 check_primary_admission 原本要 reject 时，判断是否属于“虽然 hierarchy/path 弱，但语义上该保留”的边界词。
    返回 (retain, retain_reason)；retain_reason 为 "strong_*" 时 retain_mode=strong、不 suppress_seed，否则 weak、suppress_seed=True。
    """
    cand = (candidate_text or "").strip()
    anchor = (anchor_text or "").strip().lower()
    src = (source_type or "").strip().lower()

    # 1. 主干语义强：semantic + jd_align 双高 → strong retain
    if semantic_score >= 0.82 and jd_align >= 0.80:
        return True, "strong_semantic_retain"

    # 2. 工程主干锚点 + 明显工程术语，path 弱也先保留
    if _is_engineering_core_anchor(anchor_text) and _is_engineering_core_term(cand):
        if semantic_score >= 0.78 and jd_align >= 0.76:
            return True, "engineering_core_retain"

    # 3. similar_to 时，非明显错义则允许弱保留
    if src == "similar_to":
        if semantic_score >= 0.80 and jd_align >= 0.76:
            return True, "similar_to_weak_retain"

    # 4. hierarchy 只是弱（非 off_path 硬冲突）时也可弱保留
    if hierarchy_level != HIERARCHY_LEVEL_OFF_PATH:
        if semantic_score >= 0.78 and jd_align >= 0.78:
            return True, "borderline_hierarchy_retain"

    # 5. 即使 off_path，双高也弱保留（避免 simulation / reinforcement learning 等误杀）
    if semantic_score >= 0.80 and jd_align >= 0.78:
        return True, "weak_off_path_retain"

    return False, None


def _is_bad_support_for_anchor(anchor_text: str, support_term: str) -> bool:
    """扩散支撑词是否明显偏义（与 is_semantic_mismatch_seed 同族逻辑，避免 抓取→retrieval、动力学→propulsion 等）。"""
    return is_semantic_mismatch_seed(anchor_text, support_term)


def is_semantic_mismatch_seed(anchor_text: str, primary_term: str) -> bool:
    """
    临时护栏：明显错义/偏义 seed 禁止扩散。
    动力学→propulsion、抓取→retrieval/crawling、端到端→principle、强化学习→仅 q-learning 等。
    """
    anchor = (anchor_text or "").strip().lower()
    term = (primary_term or "").strip().lower()
    if not term:
        return False

    if "动力学" in anchor or "dynamics" in anchor:
        if "propulsion" in term or "propulsor" in term:
            return True
    if "抓取" in anchor or "grasping" in anchor or "manipulation" in anchor:
        if any(x in term for x in ["retrieval", "crawler", "crawling"]):
            return True
    if "端到端" in anchor or "end-to-end" in anchor:
        if any(x in term for x in ["principle", "delay", "point-to-point", "end point"]):
            return True
    if "强化学习" in anchor or "reinforcement learning" in anchor:
        if term == "q-learning":
            return True
    return False


def _is_over_specific_subterm(anchor_text: str, primary_term: str) -> bool:
    """是否仅为锚点的过窄子项（如 q-learning 对 强化学习、robot hand 对 机械臂），不宜单独当 seed。"""
    anchor = (anchor_text or "").strip().lower()
    term = (primary_term or "").strip().lower()
    # 子项黑名单：锚点 -> 仅该词时禁扩
    over_specific = (
        ("机械臂", "robot hand"),
        ("机械臂", "robotic hand"),
        ("强化学习", "q-learning"),
        ("robot control", "machine control"),
        ("robot control", "servo control"),
        ("robotics", "telerobotics"),
    )
    for a_key, t_key in over_specific:
        if a_key in anchor or anchor in a_key:
            if t_key in term or term in t_key:
                return True
    return False


def should_block_seed_expansion(
    anchor_text: str,
    primary_term: str,
    primary_score: float,
    anchor_identity: float,
    jd_align: float,
    source_type: str,
    support_count: int,
    retain_mode: str,
    suppress_seed: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    surviving primary 是否应禁止扩散。
    返回 (block, block_reason)；block=True 时不应作为 seed。
    weak_retain 且 suppress_seed=True 时禁扩；suppress_seed=False 时可扩。
    """
    if retain_mode in ("weak", "weak_primary"):
        return True, "weak_primary_no_expand"
    if retain_mode == "weak_retain" and suppress_seed:
        return True, "weak_primary_no_expand"
    if anchor_identity < 0.20 and support_count < 2:
        return True, "low_identity_single_support"
    if is_semantic_mismatch_seed(anchor_text, primary_term):
        return True, "semantic_mismatch_seed"
    if _is_over_specific_subterm(anchor_text, primary_term):
        return True, "over_specific_without_head_term"
    src = (source_type or "").strip().lower()
    if src == "conditioned_vec" and anchor_identity < 0.35 and support_count < 2:
        return True, "weak_condvec_seed"
    return False, None


def check_primary_admission(
    anchor_text: str,
    anchor_meta: Any,
    candidate: LandingCandidate,
    hierarchy_evidence: Dict[str, Any],
    semantic_score: float,
    anchor_identity: float,
    jd_align: float,
    source_type: str,
    cross_anchor_support_count: int,
) -> Tuple[bool, List[str], bool, Dict[str, Any]]:
    """
    Stage2A 主落点准入：domain 强冲突才硬拒；topic/path 弱改为 weak_retain，不再误杀主干词。
    返回 (admitted, reasons, rescued, meta)；meta 含 retain_mode（normal|weak_retain）、suppress_seed、retain_reason。
    """
    src = (source_type or "").strip().lower()
    reasons: List[str] = []
    if not _lexical_term_sanity(getattr(candidate, "term", "") or "", None):
        return False, ["lexical_not_term"], False, {"retain_mode": "reject"}

    # 大领域强冲突：三层 overlap 全为 0 时硬拒
    field_o = hierarchy_evidence.get("field_overlap", 0) or 0
    sub_o = hierarchy_evidence.get("subfield_overlap", 0) or 0
    topic_o = hierarchy_evidence.get("topic_overlap", 0) or 0
    if field_o == 0 and sub_o == 0 and topic_o == 0:
        # 无 JD 时也为 0，用 semantic/jd 救回；有 JD 时全 0 即强冲突
        if semantic_score < 0.78 or jd_align < 0.76:
            return False, ["domain_conflict_strong"], False, {"retain_mode": "reject"}

    if anchor_identity < 0.20:
        if not (jd_align >= PRIMARY_RESCUE_JD_ALIGN_MIN and semantic_score >= PRIMARY_RESCUE_SEMANTIC_MIN):
            reasons.append("low_identity_not_rescued")
    is_generic = getattr(anchor_meta, "is_generic_anchor", False)
    if is_generic and src == "conditioned_vec" and cross_anchor_support_count < 2:
        reasons.append("generic_anchor_condvec_needs_multi_support")
    if reasons:
        return False, reasons, False, {"retain_mode": "reject"}

    ev_topic = hierarchy_evidence.get("effective_topic_overlap", hierarchy_evidence.get("topic_overlap", 0)) or 0
    ev_path = hierarchy_evidence.get("effective_path_match", hierarchy_evidence.get("path_match", 0)) or 0

    # 主干语义保护：双高直接弱保留且可做 seed
    if semantic_score >= 0.82 and jd_align >= 0.80:
        return True, [], False, {"retain_mode": "weak_retain", "suppress_seed": False, "retain_reason": "strong_semantic"}

    # hierarchy 很强，正常通过
    if ev_topic >= PRIMARY_MIN_HIERARCHY_MATCH and ev_path >= PRIMARY_MIN_PATH_MATCH:
        return True, [], False, {"retain_mode": "normal", "suppress_seed": False}

    # hierarchy 弱但 semantic/jd 足够，弱保留且默认不扩散
    if semantic_score >= 0.78 and jd_align >= 0.76:
        return True, ["hierarchy_weak"], False, {"retain_mode": "weak_retain", "suppress_seed": True, "retain_reason": "borderline_hierarchy"}

    return False, ["hierarchy_weak", "path_weak"], False, {"retain_mode": "reject"}


def _piecewise_identity_factor(anchor_identity: float) -> float:
    """极低 identity 时大幅下沉。"""
    if anchor_identity >= 0.5:
        return 1.0
    if anchor_identity >= 0.30:
        return 0.85
    if anchor_identity >= 0.15:
        return 0.60
    return 0.40


def compute_primary_score(
    candidate: LandingCandidate,
    semantic_score: float,
    anchor_identity: float,
    jd_align: float,
    cross_anchor_support_count: int,
    local_neighborhood_consistency: float,
    hierarchy_evidence: Dict[str, Any],
    source_type: str,
) -> float:
    """
    对已准入的候选排序；hierarchy 作强加分项，不做硬生杀；weak_retain 轻降权 0.85。
    """
    base = (
        0.35 * max(0, min(1, semantic_score))
        + 0.25 * max(0, min(1, jd_align))
        + 0.15 * max(0, min(1, anchor_identity))
        + 0.10 * min(cross_anchor_support_count / 2.0, 1.0)
        + 0.15 * max(0, min(1, local_neighborhood_consistency))
    )
    eff_sub = hierarchy_evidence.get("effective_subfield_overlap", hierarchy_evidence.get("subfield_overlap", 0)) or 0
    eff_topic = hierarchy_evidence.get("effective_topic_overlap", hierarchy_evidence.get("topic_overlap", 0)) or 0
    eff_path = hierarchy_evidence.get("effective_path_match", hierarchy_evidence.get("path_match", 0)) or 0
    spec = hierarchy_evidence.get("topic_specificity", 0) or 0
    hierarchy_score = 0.20 * eff_sub + 0.30 * eff_topic + 0.30 * eff_path + 0.20 * spec
    span_factor = hierarchy_evidence.get("topic_span_penalty", 1.0) or 1.0
    final = (base + hierarchy_score) * span_factor
    if getattr(candidate, "retain_mode", "normal") == "weak_retain":
        final *= 0.85
    return final


# 主词优先：锚点->主干表达给小幅加分，避免 robot hand 长期压过 robotic arm
_HEAD_TERM_BONUS_MAP = (
    ("机械臂", ("robotic arm", "robot arm")),
    ("运动控制", ("motion control",)),
    ("机器人运动控制", ("robot control", "motion control")),
    ("路径规划", ("path planning", "pathfinding")),
    ("强化学习", ("reinforcement learning",)),
    ("仿真", ("simulation",)),
    ("医疗机器人", ("medical robotics",)),
    ("传统控制", ("automatic control",)),
)


def _head_term_bonus(anchor_text: str, term: str) -> float:
    """锚点主干词匹配时加小 bonus，便于 robotic arm 压过 robot hand。"""
    if not anchor_text or not term:
        return 0.0
    anchor = (anchor_text or "").strip().lower()
    t = (term or "").strip().lower()
    for a_key, heads in _HEAD_TERM_BONUS_MAP:
        if a_key not in anchor and anchor not in a_key:
            continue
        for h in heads:
            if h in t or t in h:
                return 0.08
    return 0.0


def choose_better_term_with_hierarchy(
    a: LandingCandidate,
    b: LandingCandidate,
    anchor: PreparedAnchor,
) -> LandingCandidate:
    """锚点内冲突裁判：三层领域 + 主词优先 bonus，避免子项压主项。"""
    ev_a = getattr(a, "hierarchy_evidence", {}) or {}
    ev_b = getattr(b, "hierarchy_evidence", {}) or {}
    anchor_text = getattr(anchor, "anchor", "") or ""
    score_a = (
        0.30 * (getattr(a, "primary_score", 0) or 0)
        + 0.20 * (ev_a.get("effective_topic_overlap", ev_a.get("topic_overlap", 0)) or 0)
        + 0.20 * (ev_a.get("effective_path_match", ev_a.get("path_match", 0)) or 0)
        + 0.10 * ev_a.get("topic_specificity", 0)
        + 0.10 * max(0, getattr(a, "jd_candidate_alignment", 0.5) or 0.5)
        + 0.10 * getattr(a, "anchor_identity_score", 0.5)
    ) * ev_a.get("topic_span_penalty", 1.0)
    score_a += _head_term_bonus(anchor_text, getattr(a, "term", "") or "")
    score_b = (
        0.30 * (getattr(b, "primary_score", 0) or 0)
        + 0.20 * (ev_b.get("effective_topic_overlap", ev_b.get("topic_overlap", 0)) or 0)
        + 0.20 * (ev_b.get("effective_path_match", ev_b.get("path_match", 0)) or 0)
        + 0.10 * ev_b.get("topic_specificity", 0)
        + 0.10 * max(0, getattr(b, "jd_candidate_alignment", 0.5) or 0.5)
        + 0.10 * getattr(b, "anchor_identity_score", 0.5)
    ) * ev_b.get("topic_span_penalty", 1.0)
    score_b += _head_term_bonus(anchor_text, getattr(b, "term", "") or "")
    return a if score_a >= score_b else b


def _are_semantically_overlapping(a: LandingCandidate, b: LandingCandidate) -> bool:
    """是否语义重叠（同 vid 或 term 高度相似）。"""
    if a.vid == b.vid:
        return True
    ta = (getattr(a, "term", "") or "").strip().lower()
    tb = (getattr(b, "term", "") or "").strip().lower()
    if ta == tb:
        return True
    return False


def resolve_anchor_local_conflicts(
    anchor: PreparedAnchor,
    candidates: List[LandingCandidate],
    primary_top_k: int = PRIMARY_MAX_PER_ANCHOR,
) -> List[LandingCandidate]:
    """
    同一锚点下泛词 vs 具体词、错义词 vs 对义词的裁决；控制下 control flow 不压过 motion control，抓取下 Data retrieval 不压过 grasping 类。
    """
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda x: getattr(x, "primary_score", 0.0) or 0.0, reverse=True)
    kept: List[LandingCandidate] = []
    for cand in ranked:
        conflict = False
        for i, existing in enumerate(kept):
            if not _are_semantically_overlapping(cand, existing):
                continue
            winner = choose_better_term_with_hierarchy(cand, existing, anchor)
            if winner is existing:
                conflict = True
                if LABEL_EXPANSION_DEBUG:
                    debug_print(2, f"[Stage2A Conflict Drop] anchor={anchor.anchor!r} drop={cand.term!r} kept={existing.term!r}", None)
                break
            else:
                kept.pop(i)
                if LABEL_EXPANSION_DEBUG:
                    debug_print(2, f"[Stage2A Conflict Replace] anchor={anchor.anchor!r} old={existing.term!r} new={cand.term!r}", None)
                break
        if not conflict:
            kept.append(cand)
        if len(kept) >= primary_top_k:
            break
    return kept[:primary_top_k]


def log_primary_reject(
    anchor: Any,
    cand: LandingCandidate,
    hierarchy_evidence: Dict[str, Any],
    reasons: List[str],
) -> None:
    """Stage2A 准入拒绝日志，便于排查为何错词被挡。"""
    if not LABEL_EXPANSION_DEBUG:
        return
    topic = hierarchy_evidence.get("topic_overlap", 0)
    sub = hierarchy_evidence.get("subfield_overlap", 0)
    path = hierarchy_evidence.get("path_match", 0)
    anchor_text = getattr(anchor, "anchor", "") or ""
    term = getattr(cand, "term", "") or ""
    print(
        f"[Stage2A Reject] anchor={anchor_text!r} cand={term!r} "
        f"topic={topic:.3f} sub={sub:.3f} path={path:.3f} reasons={reasons}"
    )


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
        e = ExpandedTermCandidate(
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
            domain_fit=getattr(p, "domain_fit", 1.0),
            parent_primary=p.term,
        )
        if getattr(p, "subfield_fit", None) is not None:
            setattr(e, "subfield_fit", p.subfield_fit)
        if getattr(p, "topic_fit", None) is not None:
            setattr(e, "topic_fit", p.topic_fit)
        setattr(e, "field_fit", getattr(p, "field_fit", 0))
        setattr(e, "path_match", getattr(p, "path_match", 0))
        setattr(e, "genericity_penalty", getattr(p, "topic_span_penalty", 1.0))
        setattr(e, "retain_mode", getattr(p, "retain_mode", "normal"))
        setattr(e, "topic_source", getattr(p, "topic_source", "missing"))
        setattr(e, "seed_blocked", getattr(p, "seed_blocked", False))
        setattr(e, "seed_block_reason", getattr(p, "seed_block_reason", None))
        if getattr(p, "outside_subfield_mass", None) is not None:
            setattr(e, "outside_subfield_mass", p.outside_subfield_mass)
        if getattr(p, "outside_topic_mass", None) is not None:
            setattr(e, "outside_topic_mass", p.outside_topic_mass)
        if getattr(p, "topic_entropy", None) is not None:
            setattr(e, "topic_entropy", p.topic_entropy)
        if getattr(p, "landing_score", None) is not None:
            setattr(e, "landing_score", p.landing_score)
        if getattr(p, "fit_info", None) and p.fit_info.get("main_subfield_match") is not None:
            setattr(e, "main_subfield_match", p.fit_info.get("main_subfield_match"))
        if getattr(label, "voc_to_clusters", None):
            clusters = label.voc_to_clusters.get(int(p.vid)) or []
            if clusters:
                cid, _ = max(clusters, key=lambda x: x[1])
                setattr(e, "cluster_id", cid)
        out.append(e)
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
        c.domain_fit = _compute_domain_fit(label, c.vid, active_domain_set=active_domains, jd_field_ids=jd_field_ids, jd_subfield_ids=jd_subfield_ids, jd_topic_ids=jd_topic_ids)
        if c.domain_fit < SUPPORT_MIN_DOMAIN_FIT or (getattr(c, "domain_span", 0) or 0) > DOMAIN_SPAN_EXTREME:
            continue
        if getattr(label, "voc_to_clusters", None):
            clusters = label.voc_to_clusters.get(int(c.vid)) or []
            if clusters:
                cid, _ = max(clusters, key=lambda x: x[1])
                setattr(c, "cluster_id", cid)
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
        c.domain_fit = _compute_domain_fit(label, c.vid, active_domain_set=active_domains, jd_field_ids=jd_field_ids, jd_subfield_ids=jd_subfield_ids, jd_topic_ids=jd_topic_ids)
        if c.domain_fit < SUPPORT_MIN_DOMAIN_FIT or (getattr(c, "domain_span", 0) or 0) > DOMAIN_SPAN_EXTREME:
            continue
        if getattr(label, "voc_to_clusters", None):
            clusters = label.voc_to_clusters.get(int(c.vid)) or []
            if clusters:
                cid, _ = max(clusters, key=lambda x: x[1])
                setattr(c, "cluster_id", cid)
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
        c.domain_fit = _compute_domain_fit(label, c.vid, active_domain_set=active_domains, jd_field_ids=jd_field_ids, jd_subfield_ids=jd_subfield_ids, jd_topic_ids=jd_topic_ids)
        if c.domain_fit < SUPPORT_MIN_DOMAIN_FIT or (getattr(c, "domain_span", 0) or 0) > DOMAIN_SPAN_EXTREME:
            continue
        if getattr(label, "voc_to_clusters", None):
            clusters = label.voc_to_clusters.get(int(c.vid)) or []
            if clusters:
                cid, _ = max(clusters, key=lambda x: x[1])
                setattr(c, "cluster_id", cid)
        out.append(c)
    if LABEL_EXPANSION_DEBUG:
        n_primary = len(primary_landings)
        for i, c in enumerate(out[:15]):
            print(f"[stage2_merged] tid={c.vid} term={c.term!r} source_type={getattr(c,'source','')} parent_anchor={c.anchor_term!r} parent_primary={getattr(c,'parent_primary', c.term)!r} score={getattr(c,'identity_score',0):.3f} domain_fit={getattr(c,'domain_fit',1):.3f}")
        if len(out) > 15:
            print(f"[stage2_merged] ... 共 {len(out)} 条")
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
    jd_profile: Optional[Dict[str, Any]] = None,
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
    # 诊断：新 Stage2 流水线统一在此初始化 similar_to 相关 debug，供 stage5 / 诊断面板使用
    if getattr(label, "debug_info", None) is not None:
        label.debug_info.similar_to_raw_rows = []
        label.debug_info.similar_to_agg = []
        label.debug_info.similar_to_pass = []
    all_terms = []
    debug_print(1, "\n" + "-" * 80 + "\n[Stage2A] Primary Landing\n" + "-" * 80, label)
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2] stage2_generate_academic_terms 开始 锚点数={len(prepared_anchors)} active_domains={len(active_domains)}")
    # ---------- Stage2 总览（详细调试） ----------
    _stage2_header("总览")
    if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
        has_jd = jd_profile is not None
        jd_fields = len(jd_field_ids) if jd_field_ids else 0
        jd_sub = len(jd_subfield_ids) if jd_subfield_ids else 0
        jd_top = len(jd_topic_ids) if jd_topic_ids else 0
        print(f"  锚点数: {len(prepared_anchors)}  |  active_domains: {len(active_domains)}  |  有 JD: {has_jd}")
        print(f"  JD 维度: field={jd_fields}  subfield={jd_sub}  topic={jd_top}  |  PRIMARY_TOP_M_PER_ANCHOR={PRIMARY_TOP_M_PER_ANCHOR}")
        print(f"  锚点列表: {[getattr(a, 'anchor', a) for a in prepared_anchors]}")
    # ---------- 数据驱动两阶段选主：先每锚 top-m 候选，再全局邻域/离群算 primary_score，最后每锚取 top primary ----------
    anchor_cands_list: List[Tuple[PreparedAnchor, List[LandingCandidate]]] = []
    for anchor in prepared_anchors:
        candidates = collect_landing_candidates(
            label, anchor,
            active_domain_set=active_domains,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
            jd_profile=jd_profile,
            query_vector=query_vector,
        )
        for c in candidates:
            setattr(c, "identity_score", score_academic_identity(c))
        sort_key = lambda x: (getattr(x, "landing_score", None) is not None, getattr(x, "landing_score", 0.0) or getattr(x, "identity_score", 0.0))
        top_m = sorted(candidates, key=sort_key, reverse=True)[: PRIMARY_TOP_M_PER_ANCHOR]
        if top_m:
            anchor_cands_list.append((anchor, top_m))
    # ---------- Stage2A 每锚候选与 top-m ----------
    _stage2_header("Stage2A 每锚候选与 top-m", "-")
    if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
        _stage2_table(
            [
                [str(i + 1), getattr(a, "anchor", str(a))[:20], str(len(top_m)), ", ".join((c.term or "")[:12] for c in top_m[:6])]
                for i, (a, top_m) in enumerate(anchor_cands_list)
            ],
            ["#", "锚点", "top_m数", "top 候选(前6)"],
            col_widths=[4, 22, 10, 48],
        )
    for idx, (anchor, top_m) in enumerate(anchor_cands_list[:5]):
        raw_top = [c.term for c in top_m[:5]]
        conditioned_top = _top_terms_by_vector(label, getattr(anchor, "conditioned_vec", None), 5) if getattr(anchor, "conditioned_vec", None) is not None else []
        debug_print(2, f"[Stage2A Neighbor Compare] anchor={anchor.anchor!r}", label)
        debug_print(2, f"  raw_top={raw_top}", label)
        debug_print(2, f"  conditioned_top={conditioned_top}", label)
    flat_pool: List[Tuple[PreparedAnchor, LandingCandidate]] = []
    for anchor, cands in anchor_cands_list:
        for c in cands:
            flat_pool.append((anchor, c))
    if flat_pool:
        _compute_neighborhood_and_isolation(label, flat_pool)
        _compute_conditioned_anchor_align_and_multi_anchor_support(label, flat_pool, prepared_anchors)

    _stage2_header("Stage2A flat_pool（邻域/conditioned 已算）", "-")
    if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
        print(f"  flat_pool 总条数: {len(flat_pool)}")

    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))
    primary_landings_by_anchor: Dict[int, List[PrimaryLanding]] = {}
    evidence_table: List[Dict[str, Any]] = []

    if flat_pool:
        # 唯一 primary 链：准入(check_primary_admission) -> 打分(compute_primary_score) -> 冲突消解(resolve_anchor_local_conflicts) -> primary_landings
        by_term_cross: Dict[Tuple[int, str], Set[int]] = {}
        for a, c in flat_pool:
            key = (c.vid, (c.term or "").strip())
            if key not in by_term_cross:
                by_term_cross[key] = set()
            by_term_cross[key].add(a.vid)
        hier_cache: Dict[int, Dict[str, Any]] = {}
        for anchor, c in flat_pool:
            if c.vid not in hier_cache:
                hier_cache[c.vid] = compute_hierarchy_evidence(label, c.vid, jd_f, jd_s, jd_t)
            hier_ev = hier_cache[c.vid]
            cross_count = len(by_term_cross.get((c.vid, (c.term or "").strip()), set()))
            sem = getattr(c, "semantic_score", 0) or 0
            aid = getattr(c, "anchor_identity_score", 0.5) or 0.5
            jd_a = getattr(c, "jd_candidate_alignment", 0.5) or 0.5
            src = (getattr(c, "source", "") or "").strip()
            admitted, reasons, rescued, admission_meta = check_primary_admission(
                getattr(anchor, "anchor", ""),
                anchor,
                c,
                hier_ev,
                sem,
                aid,
                jd_a,
                src,
                cross_count,
            )
            if not admitted:
                setattr(c, "_primary_rejected", True)
                log_primary_reject(anchor, c, hier_ev, reasons)
            else:
                setattr(c, "_primary_rejected", False)
                setattr(c, "retain_mode", admission_meta.get("retain_mode", "normal"))
                setattr(c, "suppress_seed", admission_meta.get("suppress_seed", False))
                setattr(c, "retain_reason", admission_meta.get("retain_reason"))
                setattr(c, "cross_anchor_support_count", cross_count)
                setattr(c, "hierarchy_evidence", hier_ev)
                setattr(c, "topic_source", hier_ev.get("topic_source", "missing"))
                setattr(c, "identity_score", score_academic_identity(c))
                ps = compute_primary_score(
                    c,
                    sem,
                    aid,
                    jd_a,
                    cross_count,
                    getattr(c, "neighborhood_consistency", 0.5) or 0.5,
                    hier_ev,
                    src,
                )
                setattr(c, "primary_score", ps)
        _n_before_admission = len(flat_pool)
        flat_pool = [(a, c) for (a, c) in flat_pool if not getattr(c, "_primary_rejected", False)]
        _n_after_admission = len(flat_pool)
        _stage2_header("Stage2A 准入结果", "-")
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
            print(f"  准入前: {_n_before_admission}  条  ->  准入后: {_n_after_admission}  条  (拒绝: {_n_before_admission - _n_after_admission})")
        for anchor in prepared_anchors:
            pool_for_anchor = [c for (a, c) in flat_pool if a.vid == anchor.vid]
            primary_candidates = resolve_anchor_local_conflicts(anchor, pool_for_anchor)
            primary_landings_list = []
            for c in primary_candidates:
                p = PrimaryLanding(
                    vid=c.vid,
                    term=c.term,
                    identity_score=getattr(c, "identity_score", 0),
                    source=c.source,
                    anchor_vid=c.anchor_vid,
                    anchor_term=c.anchor_term,
                    domain_fit=getattr(c, "domain_fit", 1.0),
                )
                setattr(p, "primary_score", getattr(c, "primary_score", 0))
                setattr(p, "anchor_identity_score", getattr(c, "anchor_identity_score", 0.5))
                setattr(p, "identity_gate", getattr(c, "identity_gate", 1.0))
                setattr(p, "retain_mode", getattr(c, "retain_mode", "normal"))
                setattr(p, "topic_source", getattr(c, "topic_source", "missing"))
                setattr(p, "suppress_seed", getattr(c, "suppress_seed", False))
                setattr(p, "retain_reason", getattr(c, "retain_reason"))
                setattr(p, "cross_anchor_support_count", getattr(c, "cross_anchor_support_count", 1))
                setattr(p, "jd_align", getattr(c, "jd_candidate_alignment", 0.5))
                setattr(p, "semantic_score", getattr(c, "semantic_score", 0))
                ev = getattr(c, "hierarchy_evidence", {}) or {}
                setattr(p, "field_fit", ev.get("field_overlap", 0))
                setattr(p, "path_match", ev.get("path_match", 0))
                setattr(p, "topic_span_penalty", ev.get("topic_span_penalty", 1.0))
                if getattr(c, "fit_info", None) is not None:
                    setattr(p, "fit_info", c.fit_info)
                    setattr(p, "subfield_fit", getattr(c, "subfield_fit", 0))
                    setattr(p, "topic_fit", getattr(c, "topic_fit", 0))
                    setattr(p, "outside_subfield_mass", getattr(c, "outside_subfield_mass", 0))
                    setattr(p, "topic_entropy", getattr(c, "topic_entropy", 0))
                    setattr(p, "landing_score", getattr(c, "landing_score", 0))
                primary_landings_list.append(p)
            primary_landings_by_anchor[anchor.vid] = primary_landings_list
        _stage2_header("Stage2A 每锚 primary（冲突消解后）", "-")
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
            rows_summary = []
            for anchor in prepared_anchors:
                pool_for_anchor = [c for (a, c) in flat_pool if a.vid == anchor.vid]
                plist = primary_landings_by_anchor.get(anchor.vid, [])
                rows_summary.append([
                    getattr(anchor, "anchor", str(anchor))[:18],
                    str(len(pool_for_anchor)),
                    str(len(plist)),
                ])
            if rows_summary:
                _stage2_table(rows_summary, ["锚点", "准入候选数", "primary数"], col_widths=[20, 12, 12])
            for anchor in prepared_anchors:
                plist = primary_landings_by_anchor.get(anchor.vid, [])
                if not plist:
                    continue
                print(f"  --- 锚点 {getattr(anchor, 'anchor', anchor)!r} primary 明细 ---")
                _stage2_table(
                    [
                        [
                            str(p.vid),
                            (p.term or "")[:16],
                            f"{getattr(p, 'primary_score', 0):.3f}",
                            f"{getattr(p, 'path_match', 0):.2f}",
                            f"{getattr(p, 'topic_span_penalty', 1.0):.2f}",
                        ]
                        for p in plist[:15]
                    ],
                    ["tid", "term", "primary_score", "path_match", "topic_span_penalty"],
                    col_widths=[8, 18, 14, 12, 18],
                )
                if len(plist) > 15:
                    print(f"  ... 共 {len(plist)} 条 primary")
        for anchor, c in flat_pool:
            ev = getattr(c, "hierarchy_evidence", {}) or {}
            hierarchy_n = ev.get("topic_overlap", 0) * 0.4 + ev.get("path_match", 0) * 0.4 + ev.get("subfield_overlap", 0) * 0.2
            anchor_align_val = getattr(c, "conditioned_anchor_align", None) or getattr(c, "semantic_score", 0)
            evidence_table.append({
                "anchor": anchor.anchor,
                "anchor_vid": anchor.vid,
                "candidate": c.term,
                "tid": c.vid,
                "source": getattr(c, "source", "") or "",
                "semantic_score": getattr(c, "semantic_score", 0) or 0,
                "edge_affinity": getattr(c, "identity_score", 0),
                "anchor_align": anchor_align_val,
                "conditioned_anchor_align": getattr(c, "conditioned_anchor_align", None),
                "multi_anchor_support": getattr(c, "multi_anchor_support", 0.5),
                "jd_align": getattr(c, "jd_candidate_alignment", 0.5),
                "hierarchy_consistency": hierarchy_n,
                "neighborhood_consistency": getattr(c, "neighborhood_consistency", 0.5),
                "isolation_penalty": getattr(c, "semantic_isolation_penalty", 0),
                "polysemy_risk": getattr(c, "outside_subfield_mass", 0.5) or 0.5,
                "specificity_prior": 0.5,
                "anchor_identity_score": getattr(c, "anchor_identity_score", 0.5),
                "identity_gate": getattr(c, "identity_gate", 1.0),
                "base_primary_score": getattr(c, "primary_score", 0),
                "primary_score": getattr(c, "primary_score", 0),
            })
    if getattr(label, "debug_info", None) is not None:
        label.debug_info.stage2_anchor_evidence_table = evidence_table
        # 按 term 聚合：term | sources | similar_to_score | conditioned_score | final_primary_score（便于区分双路来源）
        by_term: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for row in evidence_table:
            tid, term = row["tid"], (row.get("candidate") or "").strip() or str(row["tid"])
            key = (tid, term)
            if key not in by_term:
                by_term[key] = {
                    "tid": tid,
                    "term": term,
                    "sources": set(),
                    "similar_to_score": None,
                    "conditioned_score": None,
                    "final_primary_score": 0.0,
                }
            s = (row.get("source") or "").strip().lower()
            sem = float(row.get("semantic_score") or 0)
            ps = float(row.get("primary_score") or 0)
            by_term[key]["sources"].add(s if s else "similar_to")
            if s == "similar_to":
                if by_term[key]["similar_to_score"] is None or sem > (by_term[key]["similar_to_score"] or 0):
                    by_term[key]["similar_to_score"] = sem
            elif s == "conditioned_vec":
                if by_term[key]["conditioned_score"] is None or sem > (by_term[key]["conditioned_score"] or 0):
                    by_term[key]["conditioned_score"] = sem
            if ps > (by_term[key]["final_primary_score"] or 0):
                by_term[key]["final_primary_score"] = ps
        term_breakdown = []
        for (tid, term), v in by_term.items():
            term_breakdown.append({
                "tid": tid,
                "term": v["term"],
                "sources": sorted(v["sources"]) if v["sources"] else [],
                "similar_to_score": v["similar_to_score"],
                "conditioned_score": v["conditioned_score"],
                "final_primary_score": v["final_primary_score"],
            })
        term_breakdown.sort(key=lambda x: -(x["final_primary_score"] or 0))
        label.debug_info.stage2a_term_source_breakdown = term_breakdown
    debug_print(2, "[Stage2A Primary Score Breakdown] tid | term | edge | cond_align | jd_align | hier | multi_anchor | neigh | specificity | poly_risk | isolation | final", label)
    for i, row in enumerate(evidence_table[:15], 1):
        cond = row.get("conditioned_anchor_align")
        cond_s = f"{cond:.3f}" if cond is not None else "-"
        debug_print(2, (
            f"  {i:>2} {row.get('tid')} | {(str(row.get('candidate') or ''))[:26]:<26} | "
            f"edge={row.get('edge_affinity', 0):.3f} | cond={cond_s} | jd={row.get('jd_align', 0):.3f} | "
            f"hier={row.get('hierarchy_consistency', 0):.3f} | multi={row.get('multi_anchor_support', 0):.3f} | "
            f"neigh={row.get('neighborhood_consistency', 0):.3f} | spec={row.get('specificity_prior', 0):.3f} | "
            f"poly={row.get('polysemy_risk', 0):.3f} | isol={row.get('isolation_penalty', 0):.3f} | "
            f"final={row.get('primary_score', 0):.3f}"
        ), label)
    by_term_cross: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for anchor, c in flat_pool:
        key = (c.vid, c.term)
        if key not in by_term_cross:
            by_term_cross[key] = {"term": c.term, "supported_by_anchors": [], "support_weight_sum": 0.0}
        by_term_cross[key]["supported_by_anchors"].append(anchor.anchor)
        by_term_cross[key]["support_weight_sum"] += getattr(c, "primary_score", 0.0)
    cross_list = sorted(by_term_cross.values(), key=lambda x: -x["support_weight_sum"])[:15]
    debug_print(2, "[Stage2A Cross-Anchor Evidence] term | support_count | support_weight_sum | anchors", label)
    for row in cross_list:
        anchors_preview = (row.get("supported_by_anchors") or [])[:5]
        debug_print(2, f"  {(row.get('term') or '')[:28]:<28} | cnt={len(row.get('supported_by_anchors') or []):>2} | sum={row.get('support_weight_sum', 0):.3f} | {anchors_preview}", label)
    if LABEL_EXPANSION_DEBUG and evidence_table:
        print("[Stage2A Identity Gate] anchor | candidate | source | identity | gate | base | final")
        for row in evidence_table[:30]:
            anc = (str(row.get("anchor") or ""))[:14]
            cand = (str(row.get("candidate") or ""))[:24]
            src = (row.get("source") or "")[:14]
            aid = row.get("anchor_identity_score", 0.5)
            gate = row.get("identity_gate", 1.0)
            base = row.get("base_primary_score", 0)
            final = row.get("primary_score", 0)
            print(f"  {anc:14s} | {cand:24s} | {src:14s} | id={aid:.3f} | gate={gate:.2f} | base={base:.3f} | final={final:.3f}")
        if len(evidence_table) > 30:
            print(f"  ... 共 {len(evidence_table)} 条")
        print("[Stage2 锚点-候选证据表] anchor | candidate(tid) | edge | cond_align | multi_anchor | jd_align | hier | neigh | isol | primary")
        for row in evidence_table[:40]:
            anc = (str(row.get("anchor") or ""))[:14]
            cand = (str(row.get("candidate") or ""))[:20]
            cond = row.get("conditioned_anchor_align")
            cond_s = f"{cond:.3f}" if cond is not None else "-"
            print(f"  {anc:14s} | {cand!r}({row.get('tid')}) | edge={row['edge_affinity']:.3f} | cond={cond_s} | multi={row.get('multi_anchor_support', 0.5):.3f} | jd={row['jd_align']:.3f} | hier={row['hierarchy_consistency']:.3f} | neigh={row['neighborhood_consistency']:.3f} | isol={row['isolation_penalty']:.3f} | primary={row['primary_score']:.3f}")
        if len(evidence_table) > 40:
            print(f"  ... 共 {len(evidence_table)} 条")

    for anchor in prepared_anchors:
        primary_landings = primary_landings_by_anchor.get(anchor.vid) or []
        if not primary_landings:
            if LABEL_EXPANSION_DEBUG:
                anchor_type_lower = (getattr(anchor, "anchor_type", "") or "").strip().lower()
                amb_tag = " [高歧义]" if anchor_type_lower in HIGH_AMBIGUITY_ANCHOR_TYPES else ""
                pool_for_anchor = [(a, c) for (a, c) in flat_pool if a.vid == anchor.vid]
                if not pool_for_anchor:
                    print(
                        f"[Stage2] 锚点 anchor={anchor.anchor!r} vid={anchor.vid}{amb_tag} 无 primary，跳过 | "
                        f"原因: 本锚点无候选进入 flat_pool"
                    )
                else:
                    print(
                        f"[Stage2] 锚点 anchor={anchor.anchor!r} vid={anchor.vid}{amb_tag} 无 primary，跳过 | "
                        f"原因: 本锚点有 {len(pool_for_anchor)} 个候选均未通过 check_primary_admission"
                    )
                    for _, c in pool_for_anchor[:5]:
                        print(f"[Stage2]   候选 tid={c.vid} term={c.term!r} identity={getattr(c, 'identity_score', 0):.3f} domain_fit={getattr(c, 'domain_fit', 0):.3f}")
                    if len(pool_for_anchor) > 5:
                        print(f"[Stage2]   ... 共 {len(pool_for_anchor)} 个候选均未通过")
            continue
        if LABEL_EXPANSION_DEBUG:
            print(f"[Stage2] 锚点 anchor={anchor.anchor!r} primary 数={len(primary_landings)}")
        # Stage2B：唯一 seed 决策入口 check_seed_eligibility -> (eligible, seed_score, block_reason)
        diffusion_primaries = []
        _seed_detail: List[Tuple[Any, bool, float]] = []
        for p in primary_landings:
            eligible, seed_score, block_reason = check_seed_eligibility(label, p, jd_profile)
            setattr(p, "seed_blocked", not eligible)
            setattr(p, "seed_block_reason", block_reason if not eligible else None)
            _seed_detail.append((p, eligible, seed_score))
            if eligible:
                setattr(p, "seed_score", seed_score)
                diffusion_primaries.append(p)
        if not diffusion_primaries and primary_landings:
            diffusion_primaries = sorted(primary_landings, key=lambda x: getattr(x, "primary_score", x.identity_score), reverse=True)[: max(1, len(primary_landings) // 2)]
        _stage2_header(f"Stage2B seed 明细 [锚点 {getattr(anchor, 'anchor', anchor)!r}]", "-")
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and _seed_detail:
            _stage2_table(
                [
                    [(p.term or "")[:20], "Y" if el else "N", f"{sc:.3f}" if el else "-"]
                    for (p, el, sc) in _seed_detail[:20]
                ],
                ["term", "eligible", "seed_score"],
                col_widths=[22, 10, 12],
            )
            print(f"  diffusion_primaries({len(diffusion_primaries)}): {[p.term for p in diffusion_primaries[:10]]}")
        debug_print(2, f"[Stage2B] anchor={anchor.anchor!r} primary 数={len(primary_landings)} seed 数={len(diffusion_primaries)}", label)
        debug_print(1, f"[Stage2B] seed 数={len(diffusion_primaries)}/{len(primary_landings)} 参与扩散（SEED_MIN_IDENTITY={SEED_MIN_IDENTITY}）", label)
        if diffusion_primaries:
            debug_print(2, f"[Stage2B] seed_terms={[p.term for p in diffusion_primaries[:10]]}", label)
        if LABEL_EXPANSION_DEBUG and (len(diffusion_primaries) != len(primary_landings) or not diffusion_primaries):
            print(f"[Stage2B] seed 数={len(diffusion_primaries)}/{len(primary_landings)}（identity≥{SEED_MIN_IDENTITY} & source∈可信，单一决策无二次审批）")
        dense_list = expand_from_vocab_dense_neighbors(
            label, diffusion_primaries,
            active_domain_set=active_domains,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
            jd_profile=jd_profile,
        )
        cluster_list = expand_from_cluster_members(
            label, diffusion_primaries,
            active_domain_set=active_domains,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
        )
        cooc_list = expand_from_cooccurrence_support(
            label, diffusion_primaries,
            active_domain_set=active_domains,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
        )
        _stage2_header("Stage2B 扩展汇总（dense / cluster / cooc）", "-")
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
            print(f"  dense_kept={len(dense_list)}  |  cluster_kept={len(cluster_list)}  |  cooc_kept={len(cooc_list)}  |  primary={len(primary_landings)}")
            if dense_list:
                print(f"  dense 前5: {[getattr(c, 'term', c) for c in dense_list[:5]]}")
            if cluster_list:
                print(f"  cluster 前5: {[getattr(c, 'term', c) for c in cluster_list[:5]]}")
            if cooc_list:
                print(f"  cooc 前5: {[getattr(c, 'term', c) for c in cooc_list[:5]]}")
        debug_print(2, (
            f"[Stage2B Expansion Summary] dense_kept={len(dense_list)} | "
            f"cluster_kept={len(cluster_list)} | cooc_kept={len(cooc_list)} | "
            f"primary={len(primary_landings)} -> merged 本锚"
        ), label)
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
        _stage2_header("Stage2B 本锚合并", "-")
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
            print(f"  primary={len(primary_landings)} + dense={len(dense_list)} + cluster={len(cluster_list)} + cooc={len(cooc_list)} -> merged={len(merged)}  |  累计 all_terms={len(all_terms)}")
        if LABEL_EXPANSION_DEBUG:
            print(f"[Stage2] 锚点 anchor={anchor.anchor!r} 本锚合并后 +{len(merged)} 项，累计 {len(all_terms)} 项")
    # 诊断：从 similar_to_raw_rows 聚合出 similar_to_agg；从最终 all_terms 中筛出 similar_to 来源的项写入 similar_to_pass
    if getattr(label, "debug_info", None) is not None:
        raw_rows = getattr(label.debug_info, "similar_to_raw_rows", None) or []
        by_tid_agg = {}
        for r in raw_rows:
            tid = r.get("tid")
            if tid is None:
                continue
            tid_key = int(tid) if isinstance(tid, (int, float)) or (isinstance(tid, str) and tid.isdigit()) else tid
            if tid_key not in by_tid_agg:
                by_tid_agg[tid_key] = {"tid": tid, "term": r.get("term", ""), "sim_score": 0.0, "src_vids": []}
            by_tid_agg[tid_key]["sim_score"] = max(by_tid_agg[tid_key]["sim_score"], float(r.get("sim_score", 0) or 0))
            src_vid = r.get("src_vid")
            if src_vid is not None and src_vid not in by_tid_agg[tid_key]["src_vids"]:
                by_tid_agg[tid_key]["src_vids"].append(src_vid)
        label.debug_info.similar_to_agg = [
            {"tid": v["tid"], "term": v["term"], "sim_score": v["sim_score"], "hit_count": len(v["src_vids"]), "src_vids": sorted(v["src_vids"])}
            for v in by_tid_agg.values()
        ]
        similar_to_vids = {c.vid for c in all_terms if (getattr(c, "source", "") or "").strip().lower() == "similar_to"}
        pass_list = []
        for tid_key, agg in by_tid_agg.items():
            if tid_key not in similar_to_vids:
                continue
            pass_list.append({
                "tid": agg["tid"],
                "term": agg["term"],
                "sim_score": float(agg["sim_score"]),
                "hit_count": len(agg["src_vids"]),
                "src_vids": agg["src_vids"],
                "degree_w": 0,
                "degree_w_expanded": 0,
                "target_degree_w": 0,
                "domain_span": 0,
            })
        label.debug_info.similar_to_pass = pass_list
    _stage2_header("Stage2 结束汇总", "=")
    if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
        by_role: Dict[str, int] = collections.Counter(getattr(t, "term_role", "?") for t in all_terms)
        print(f"  总学术词数: {len(all_terms)}")
        print(f"  按 term_role: {dict(by_role)}")
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2] stage2_generate_academic_terms 结束 总学术词数={len(all_terms)}")
    return all_terms

