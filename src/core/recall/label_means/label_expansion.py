import collections
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from types import SimpleNamespace
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
PRIMARY_TOP_M_PER_ANCHOR = 8      # 每锚先保留 top-m 候选，后面严判不早剪枝
CONDITIONED_VEC_TOP_K = 12        # 每锚点 conditioned_vec 检索学术词 top-k，与 SIMILAR_TO 合并
# Stage2A 采集阶段放宽 top_k，主线词可能排第 4～8
STAGE2A_COLLECT_BASE_TOP_K = 8   # base 视角取 6～8，严判在后
STAGE2A_COLLECT_CONDITIONED_TOP_K = 8
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
# ---------- Dense 最小修复补丁：support 锚点语义复核四道门（不靠硬编码） ----------
DENSE_SUPPORT_PRIMARY_CONSISTENCY_MIN = 0.72   # 候选与 parent primary 向量一致性
DENSE_SUPPORT_ANCHOR_CONSISTENCY_MIN = 0.70   # 候选与 anchor conditioned_vec 一致性
DENSE_SUPPORT_CONTEXT_STABILITY_MIN = 0.72    # 候选在 anchor context 邻域中的稳定性
DENSE_SUPPORT_FAMILY_SUPPORT_MIN = 0.68      # 候选对同锚 surviving primary family 的支撑
# 强冲突领域 ID：term 主领域在此集且与激活领域无交时返回 domain_conflict_strong（医学/社科/管理等），空集则所有 domain_no_match 仅做 soft retain
STRONG_CONFLICT_DOMAIN_IDS: Set[str] = set()
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

# ---------- Stage2A 四分桶：同源双视角落点稳定性裁判（详见 README） ----------
# 全局底线阈值（相对阈值在 calibrate_anchor_thresholds 中按锚点内分布计算）
STAGE2A_GLOBAL_FLOOR = SimpleNamespace(
    identity_low=0.18,
    identity_primary=0.35,
    identity_keep=0.40,
    identity_expand=0.55,
    view_stability_low=0.25,
    view_stability_primary=0.45,
    view_stability_keep=0.50,
    view_stability_expand=0.60,
    hierarchy_low=0.20,
    hierarchy_mid=0.35,
    hierarchy_expand=0.50,
    shift_quality_low=0.20,
    shift_quality_mid=0.40,
    shift_quality_expand=0.55,
    primary_keep_line=0.35,
    primary_expand_line=0.50,
    primary_keep=0.35,
    primary_expand=0.50,
    canonical_expand=0.45,
    ambiguity_high=0.70,
    ambiguity_mid=0.50,
    ambiguity_low=0.35,
    generic_mid=0.45,
    generic_low=0.35,
    branch_drift_high=0.65,
    branch_drift_mid=0.45,
    branch_drift_low=0.35,
    jd_align_mid=0.55,
    base_expand_line=0.55,
    max_reasonable_shift=0.25,
    max_reasonable_rank_gap=8,
    max_useful_gain=0.20,
    max_tolerable_drop=0.15,
    mainline_low=0.25,
    mainline_keep=0.40,
    mainline_expand=0.55,
    object_like_low=0.35,
)
# primary_score 权重（同源双视角统一打分）
STAGE2A_WEIGHTS = SimpleNamespace(
    base=0.28,
    identity=0.22,
    hierarchy=0.18,
    jd_align=0.14,
    view_stability=0.10,
    shift_quality=0.08,
    ambiguity_penalty=0.35,
    generic_penalty=0.30,
    branch_penalty=0.35,
)


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
    """Stage2A 落点候选。similar_to 初始近邻 + conditioned_vec 上下文纠偏。"""
    vid: int
    term: str
    source: str  # similar_to | conditioned_vec
    semantic_score: float
    anchor_vid: int = 0
    anchor_term: str = ""

    # ===== Stage2A 上下文纠偏 / 准入 / 打分 =====
    context_sim: float = 0.0                 # conditioned_vec 下的相似度；若无则 0
    context_supported: bool = False          # 是否得到上下文支持
    context_gap: float = 1.0                 # raw_sim - context_sim 的差，越大越可疑
    source_role: str = "seed_candidate"     # seed_candidate | context_fallback
    primary_eligible: bool = False
    primary_eligibility_reasons: List[str] = field(default_factory=list)

    # ===== 已显式化（原 setattr 动态塞入） =====
    anchor_identity_score: float = 0.5
    jd_candidate_alignment: float = 0.5
    neighborhood_consistency: float = 0.5
    local_neighborhood_consistency: float = 0.5
    cross_anchor_support_count: int = 1
    hierarchy_evidence: Dict[str, Any] = field(default_factory=dict)
    primary_score: float = 0.0
    retain_mode: str = "normal"
    suppress_seed: bool = False
    retain_reason: Optional[str] = None
    topic_source: str = "missing"
    identity_score: float = 0.0
    identity_gate: float = 1.0
    domain_fit: float = 1.0
    domain_reason: str = ""                  # domain_conflict_strong 等，供 check_primary_eligibility


@dataclass
class Stage2ACandidate:
    """Stage2A 统一候选对象：只做证据与组内相对排序，不做固定阈值淘汰。"""
    tid: int
    term: str
    source: str  # similar_to | conditioned_vec | alias_or_exact ...

    # 原始证据（仅用于组内相对比较）
    semantic_score: float = 0.0
    context_sim: float = 0.0
    jd_align: float = 0.0
    mainline_sim: Optional[float] = None
    cross_anchor_support: float = 0.0
    family_match: float = 0.0
    hierarchy_consistency: float = 0.0
    polysemy_risk: float = 0.0
    isolation_risk: float = 0.0

    # 组内相对排序用（percentile 0~1，由 assign_relative_scores_within_anchor 填充）
    relative_scores: Dict[str, float] = field(default_factory=dict)
    composite_rank_score: float = 0.0

    # 最终标签（由 select_primary_per_anchor 设置）
    survive_primary: bool = False
    can_expand: bool = False
    reject_reason: Optional[str] = None
    role: Optional[str] = None  # mainline | side | off | unknown


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
    Stage2B 唯一 seed 决策入口：只有「适合扩散的 primary」才能扩；无 fallback 兜底。
    返回 (eligible, seed_score, block_reason)。
    """
    identity = float(getattr(p, "identity_score", 0.0) or 0.0)
    anchor_identity = float(getattr(p, "anchor_identity_score", identity) or identity)
    primary_score = float(getattr(p, "primary_score", identity) or identity)
    jd_align = float(getattr(p, "jd_align", 0.5) or 0.5)
    src = (getattr(p, "source", "") or "").strip().lower()
    retain_mode = getattr(p, "retain_mode", "normal") or "normal"
    suppress_seed = bool(getattr(p, "suppress_seed", True))
    support_count = int(getattr(p, "cross_anchor_support_count", 1) or 1)
    anchor_text = getattr(p, "anchor_term", "") or ""
    primary_term = getattr(p, "term", "") or ""

    # 1) weak_retain / suppress_seed 一律不扩
    if retain_mode != "normal" or suppress_seed:
        return False, 0.0, "weak_primary_no_expand"

    # 2) source 必须可信
    trusted_set = {s.strip().lower() for s in (TRUSTED_SOURCE_TYPES_FOR_DIFFUSION or []) if s}
    if trusted_set and src not in trusted_set:
        return False, 0.0, "source_not_trusted"

    # 3) 语义错义 seed 直接禁
    if is_semantic_mismatch_seed(anchor_text, primary_term):
        return False, 0.0, "semantic_mismatch_seed"

    # 4) 不是所有 primary 都能扩
    if not is_primary_expandable(
        anchor_text=anchor_text,
        primary_term=primary_term,
        primary_score=primary_score,
        anchor_identity=anchor_identity,
        jd_align=jd_align,
        support_count=support_count,
        source_type=src,
    ):
        if LABEL_EXPANSION_DEBUG:
            print(f"[Stage2B] seed 禁扩 term={primary_term!r} anchor={anchor_text!r} reason=primary_not_expandable")
        return False, 0.0, "primary_not_expandable"

    # seed_expand_factor：过窄支线/设备词/单锚支持 惩罚
    seed_expand_factor = 1.0
    if is_narrow_method_term(primary_term):
        seed_expand_factor *= 0.75
    if is_device_or_object_term(primary_term):
        seed_expand_factor *= 0.65
    if support_count == 1:
        seed_expand_factor *= 0.85

    path_match = float(getattr(p, "path_match", 0.0) or 0.0)
    span_penalty = float(getattr(p, "topic_span_penalty", 1.0) or 1.0)
    seed_score = primary_score * (0.7 + 0.3 * path_match) * span_penalty * seed_expand_factor
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
    top_k: Optional[int] = None,
) -> List[LandingCandidate]:
    """Stage2A 落点：从锚点（industry）查跨类型 SIMILAR_TO → 学术词；仅保留与激活领域（及可选三级领域）一致的词。"""
    load_vocab_meta(label)
    if not getattr(label, "graph", None):
        if LABEL_EXPANSION_DEBUG:
            print(f"[Stage2A] SIMILAR_TO 跳过 anchor={anchor.anchor!r} vid={anchor.vid}（无 graph）")
        return []
    use_top_k = int(top_k) if top_k is not None else SIMILAR_TO_TOP_K
    params = {
        "anchor_vid": anchor.vid,
        "min_score": SIMILAR_TO_MIN_SCORE,
        "top_k": use_top_k,
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
        print(f"[Stage2A] SIMILAR_TO anchor_vid={anchor.vid} anchor={anchor.anchor!r} min_score={SIMILAR_TO_MIN_SCORE} top_k={use_top_k} -> 命中 {len(rows)} 条")
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
            elif reason == "domain_conflict_strong":
                dropped_with_reason.append((c, reason))
            else:
                # domain_no_match：上位通用主词（mechanics/simulation/route planning）保留但降权，进后续 admission
                setattr(c, "soft_domain_retain", True)
                setattr(c, "domain_fit", 0.85)
                kept.append(c)
                if LABEL_EXPANSION_DEBUG:
                    print(f"[Stage2A] 软保留（domain_no_match）vid={c.vid} term={c.term!r} sim={c.semantic_score:.3f}")
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
    similar_to_candidates: Optional[List[LandingCandidate]] = None,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> Tuple[List[LandingCandidate], Dict[int, Dict[str, float]]]:
    """
    Stage2A 上下文纠偏：
    1) 用 conditioned_vec 查学术词索引
    2) 返回 context_neighbors
    3) 同时给 similar_to 候选生成 context 重打分信号

    返回:
    - context_neighbors: 带上下文检索到的候选
    - rerank_signals: {vid: {"context_sim": x, "context_supported": 0|1, "context_gap": y}}
    """
    if getattr(anchor, "conditioned_vec", None) is None:
        return [], {}
    if not getattr(label, "vocab_index", None) or not getattr(label, "_vocab_meta", None):
        return [], {}
    load_vocab_meta(label)
    try:
        vec = np.asarray(anchor.conditioned_vec, dtype=np.float32).flatten()
        if vec.size == 0:
            return [], {}
        vec = vec.reshape(1, -1)
        faiss.normalize_L2(vec)
        k = min(STAGE2A_COLLECT_CONDITIONED_TOP_K, getattr(label.vocab_index, "ntotal", 100))
        if k <= 0:
            return [], {}
        scores, ids = label.vocab_index.search(vec, k)
    except Exception:
        return [], {}

    context_neighbors: List[LandingCandidate] = []
    context_score_map: Dict[int, float] = {}

    for score, tid in zip(scores[0], ids[0]):
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            continue
        if tid <= 0 or tid == getattr(anchor, "vid", -1):
            continue
        meta = label._vocab_meta.get(tid, ("", ""))
        term = (meta[0] or "").strip() or str(tid)
        vocab_type = meta[1] or ""
        if vocab_type not in ("concept", "keyword") and vocab_type:
            continue
        sim = max(0.0, min(1.0, float(score)))
        if sim < SIMILAR_TO_MIN_SCORE:
            continue
        ok, reason = _term_in_active_domains_with_reason(
            label, tid,
            active_domain_set=active_domain_set,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
        )
        if not ok:
            continue
        cand = LandingCandidate(
            vid=tid,
            term=term,
            source="conditioned_vec",
            semantic_score=sim,
            anchor_vid=getattr(anchor, "vid", 0),
            anchor_term=getattr(anchor, "anchor", ""),
        )
        cand.context_sim = sim
        cand.context_supported = sim >= max(SIMILAR_TO_MIN_SCORE, 0.78)
        cand.context_gap = 0.0
        cand.source_role = "context_fallback"
        context_neighbors.append(cand)
        context_score_map[tid] = sim

    rerank_signals: Dict[int, Dict[str, float]] = {}
    similar_to_candidates = similar_to_candidates or []
    for cand in similar_to_candidates:
        ctx_sim = context_score_map.get(cand.vid, 0.0)
        rerank_signals[cand.vid] = {
            "context_sim": ctx_sim,
            "context_supported": 1.0 if ctx_sim >= 0.78 else 0.0,
            "context_gap": max(0.0, float(getattr(cand, "semantic_score", 0.0) or 0.0) - ctx_sim),
        }

    if LABEL_EXPANSION_DEBUG and context_neighbors:
        print(
            f"[Stage2A] conditioned_vec 上下文纠偏 anchor={getattr(anchor, 'anchor', '')!r} "
            f"-> {len(context_neighbors)} 个候选（用于重打分/弱补位）"
        )
    return context_neighbors, rerank_signals


# ---------- Stage2A 极简版：3 个判断 + 1 个轻量排序，不堆参数 ----------

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# 极简版常量：仅用于 3 个判断，不参与总分生死
STAGE2A_ANCHOR_BASE_MIN = 0.55
STAGE2A_CTX_DROP_TOLERANCE = 0.08
STAGE2A_FAMILY_MATCH_MIN = 0.45
STAGE2A_MAINLINE_SIM_MAIN = 0.62
STAGE2A_MAINLINE_SIM_SIDE = 0.55
STAGE2A_MAINLINE_SIM_OFF = 0.52
STAGE2A_BONUS_SIM_SIDE = 0.60
STAGE2A_EXPAND_FAMILY_MIN = 0.60
STAGE2A_EXPAND_MAINLINE_MIN = 0.65
STAGE2A_EXPAND_OBJECT_RISK_MAX = 0.45


# ---------- Stage2A 极简重构：组内相对排序（无固定阈值） ----------


def _assign_rank_percentile(candidates: List["Stage2ACandidate"], key: str, out_key: str) -> None:
    """同锚点内按 key 排序，将百分位 [0,1] 写入 relative_scores[out_key]。"""
    if not candidates:
        return
    vals = [getattr(c, key, 0.0) for c in candidates]
    n = len(vals)
    order = sorted(range(n), key=lambda i: (vals[i], -i), reverse=True)
    rank_by_idx = {}
    for r, i in enumerate(order):
        rank_by_idx[i] = 1.0 - (r / (n - 1)) if n > 1 else 1.0
    for i, c in enumerate(candidates):
        c.relative_scores[out_key] = rank_by_idx.get(i, 0.0)


def _assign_rank_percentile_nullable(candidates: List["Stage2ACandidate"], key: str, out_key: str) -> None:
    """同锚点内按 key 排序，None 不参与排序，有值的给百分位；无值写 0.5。"""
    if not candidates:
        return
    valid = [(i, c) for i, c in enumerate(candidates) if getattr(c, key, None) is not None]
    if not valid:
        for c in candidates:
            c.relative_scores[out_key] = 0.5
        return
    n = len(valid)
    order = sorted(range(n), key=lambda j: (getattr(valid[j][1], key), -valid[j][0]), reverse=True)
    rank_by_orig = {valid[order[r]][0]: (1.0 - (r / (n - 1)) if n > 1 else 1.0) for r in range(n)}
    for i, c in enumerate(candidates):
        c.relative_scores[out_key] = rank_by_orig.get(i, 0.5)


def _assign_reverse_rank_percentile(candidates: List["Stage2ACandidate"], key: str, out_key: str) -> None:
    """风险项：值越大越差，按 key 升序排，最小的给最高分。"""
    if not candidates:
        return
    vals = [getattr(c, key, 0.0) for c in candidates]
    n = len(vals)
    order = sorted(range(n), key=lambda i: (vals[i], i))  # 升序，order[0]=最小
    rank_by_idx = {order[r]: (1.0 - (r / (n - 1)) if n > 1 else 1.0) for r in range(n)}
    for i, c in enumerate(candidates):
        c.relative_scores[out_key] = rank_by_idx.get(i, 0.0)


def assign_relative_scores_within_anchor(candidates: List["Stage2ACandidate"]) -> None:
    """把绝对值转为锚点内部相对位置，不依赖固定阈值。"""
    if not candidates:
        return
    _assign_rank_percentile(candidates, "semantic_score", "semantic_rank")
    _assign_rank_percentile(candidates, "context_sim", "context_rank")
    _assign_rank_percentile(candidates, "jd_align", "jd_rank")
    _assign_rank_percentile_nullable(candidates, "mainline_sim", "mainline_rank")
    _assign_rank_percentile(candidates, "cross_anchor_support", "cross_anchor_rank")
    _assign_rank_percentile(candidates, "family_match", "family_rank")
    _assign_rank_percentile(candidates, "hierarchy_consistency", "hier_rank")
    _assign_reverse_rank_percentile(candidates, "polysemy_risk", "polysemy_rank")
    _assign_reverse_rank_percentile(candidates, "isolation_risk", "isolation_rank")


def compose_anchor_internal_rank_score(cand: "Stage2ACandidate") -> float:
    """组内综合排序：证据均值，不加新参数。"""
    keys = [
        "semantic_rank", "context_rank", "jd_rank", "mainline_rank",
        "cross_anchor_rank", "family_rank", "hier_rank",
        "polysemy_rank", "isolation_rank",
    ]
    vals = [cand.relative_scores.get(k) for k in keys if cand.relative_scores.get(k) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def is_competitive_runner_up(
    best: "Stage2ACandidate",
    second: "Stage2ACandidate",
    candidates: List["Stage2ACandidate"],
) -> bool:
    """不用固定阈值：第二名是否与第一名形成「前二集团」而值得一起保留。"""
    if second.composite_rank_score <= 0:
        return False
    if len(candidates) <= 2:
        return True
    third = candidates[2]
    gap12 = best.composite_rank_score - second.composite_rank_score
    gap23 = second.composite_rank_score - third.composite_rank_score
    return gap23 >= gap12


def judge_expandability_relative(anchor: PreparedAnchor, cand: "Stage2ACandidate", candidates: List["Stage2ACandidate"]) -> bool:
    """是否可扩散：仅对 surviving primary 判断，用相对证据（强维度数 >= 弱维度数）。"""
    if not cand.survive_primary:
        return False
    dims = ["semantic_rank", "context_rank", "jd_rank", "mainline_rank", "cross_anchor_rank"]
    strong = sum(1 for k in dims if (cand.relative_scores.get(k) or 0) >= 0.5)
    weak = len(dims) - strong
    return strong >= weak


def build_stage2a_mainline_profile_centroid(
    label: Any,
    anchors: List[PreparedAnchor],
    semantic_query_text: Optional[str] = None,
) -> Dict[str, Any]:
    """构建主线画像仅用于偏好排序证据，不用于硬 reject。只输出 centroid。"""
    vecs: List[np.ndarray] = []
    for a in anchors:
        cv = getattr(a, "conditioned_vec", None)
        if cv is not None:
            try:
                v = np.asarray(cv, dtype=np.float32).flatten()
                if v.size > 0:
                    vecs.append(v)
            except Exception:
                pass
            continue
        anchor_vec = _get_candidate_vec_for_mainline(label, int(getattr(a, "vid", 0) or 0))
        if anchor_vec is not None:
            vecs.append(anchor_vec)
    if not vecs:
        return {"centroid": None}
    mat = np.array(vecs, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    centroid = np.mean(mat, axis=0).astype(np.float32)
    norm = np.linalg.norm(centroid)
    if norm > 1e-9:
        centroid = centroid / norm
    return {"centroid": centroid}


def compute_mainline_similarity(
    label: Any,
    cand: "Stage2ACandidate",
    mainline_profile: Dict[str, Any],
) -> Optional[float]:
    """候选与主线质心的相似度；无向量时返回 None（表示未知，不判死）。"""
    cand_vec = _get_candidate_vec_for_mainline(label, cand.tid)
    centroid = mainline_profile.get("centroid")
    if cand_vec is None or centroid is None:
        return None
    return float(_cos_sim_mainline(cand_vec, centroid))


def build_cross_anchor_index(
    per_anchor_candidates: Dict[int, List["Stage2ACandidate"]],
) -> Dict[int, List[int]]:
    """tid -> 支持该候选的 anchor vid 列表，供 cross_anchor_support 证据用。"""
    index: Dict[int, List[int]] = {}
    for anchor_vid, cands in per_anchor_candidates.items():
        for c in cands:
            tid = c.tid
            if tid not in index:
                index[tid] = []
            index[tid].append(anchor_vid)
    return index


def _compute_cross_anchor_support_score(tid: int, cross_anchor_index: Dict[int, List[int]]) -> float:
    """0~1：该 tid 被多少锚点支持，归一化到组内。"""
    support_list = cross_anchor_index.get(tid, [])
    if not support_list:
        return 0.0
    max_sup = max(len(v) for v in cross_anchor_index.values()) if cross_anchor_index else 1
    return len(support_list) / max(max_sup, 1)


def _compute_hierarchy_consistency_for_candidate(
    label: Any,
    tid: int,
    jd_field_ids: Optional[Set[str]],
    jd_subfield_ids: Optional[Set[str]],
    jd_topic_ids: Optional[Set[str]],
) -> float:
    """单值层级一致性，供组内相对排序用。"""
    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))
    if not jd_f and not jd_s and not jd_t:
        return 1.0
    ev = compute_hierarchy_evidence(label, tid, jd_f, jd_s, jd_t)
    path = float(ev.get("effective_path_match", 0) or 0)
    topic = float(ev.get("effective_topic_overlap", 0) or 0)
    sub = float(ev.get("effective_subfield_overlap", 0) or 0)
    return (0.4 * path + 0.35 * topic + 0.25 * sub)


def _compute_isolation_risk_for_candidates(
    label: Any,
    candidates: List["Stage2ACandidate"],
) -> None:
    """原地写入每个候选的 isolation_risk（与同组其他候选的语义平均相似度越低越高）。"""
    vocab_idx = getattr(label, "vocab_to_idx", None)
    all_vecs = getattr(label, "all_vocab_vectors", None)
    if not candidates or vocab_idx is None or all_vecs is None:
        for c in candidates:
            c.isolation_risk = 0.5
        return
    vecs: Dict[int, Optional[np.ndarray]] = {}
    for c in candidates:
        if c.tid in vecs:
            continue
        idx = label.vocab_to_idx.get(str(c.tid))
        if idx is None:
            vecs[c.tid] = None
            continue
        try:
            v = np.asarray(label.all_vocab_vectors[idx], dtype=np.float32).flatten()
            vecs[c.tid] = v
        except Exception:
            vecs[c.tid] = None
    for c in candidates:
        v = vecs.get(c.tid)
        if v is None or v.size == 0:
            c.isolation_risk = 0.5
            continue
        sims = []
        for c2 in candidates:
            if c2.tid == c.tid:
                continue
            v2 = vecs.get(c2.tid)
            if v2 is None or v2.size != v.size:
                continue
            try:
                s = float(np.dot(v, v2))
                s = max(-1.0, min(1.0, s))
                sims.append(s)
            except Exception:
                pass
        if not sims:
            c.isolation_risk = 0.5
            continue
        mean_sim = max(0.0, float(np.mean(sims)))
        c.isolation_risk = 1.0 - mean_sim


def enrich_stage2a_candidates(
    label: Any,
    anchor: PreparedAnchor,
    candidates: List["Stage2ACandidate"],
    anchors: List[PreparedAnchor],
    active_domains: Optional[Set[int]],
    semantic_query_text: Optional[str],
    cross_anchor_index: Dict[int, List[int]],
    query_vector: Optional[np.ndarray] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> List["Stage2ACandidate"]:
    """对每个候选补全证据；只算证据，不做 reject。最后按 composite_rank_score 降序。"""
    mainline_profile = build_stage2a_mainline_profile_centroid(label, anchors, semantic_query_text)
    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))

    for c in candidates:
        c.jd_align = _compute_jd_candidate_alignment(label, c.tid, query_vector) if query_vector is not None else 0.5
        c.mainline_sim = compute_mainline_similarity(label, c, mainline_profile)
        c.cross_anchor_support = _compute_cross_anchor_support_score(c.tid, cross_anchor_index)
        c.family_match = compute_anchor_identity_score(
            getattr(anchor, "anchor", "") or "",
            c.term or "",
            getattr(anchor, "anchor_type", None),
        )
        c.hierarchy_consistency = _compute_hierarchy_consistency_for_candidate(label, c.tid, jd_f, jd_s, jd_t)
        c.polysemy_risk = 1.0 - c.family_match

    _compute_isolation_risk_for_candidates(label, candidates)
    assign_relative_scores_within_anchor(candidates)
    for c in candidates:
        c.composite_rank_score = compose_anchor_internal_rank_score(c)
    return sorted(candidates, key=lambda x: x.composite_rank_score, reverse=True)


def select_primary_per_anchor(
    anchor: PreparedAnchor,
    candidates: List["Stage2ACandidate"],
) -> List["Stage2ACandidate"]:
    """锚点内保留相对最优者为 primary；再对 primary 单独判 can_expand。"""
    if not candidates:
        return []
    for c in candidates:
        c.survive_primary = False
        c.reject_reason = "not_selected_within_anchor"
    best = candidates[0]
    best.survive_primary = True
    best.role = "mainline"
    if len(candidates) >= 2:
        second = candidates[1]
        if is_competitive_runner_up(best, second, candidates):
            second.survive_primary = True
            second.role = "side"
    for c in candidates:
        if c.survive_primary:
            c.can_expand = judge_expandability_relative(anchor, c, candidates)
    return [c for c in candidates if c.survive_primary]


def _init_merged_primary(cand: "Stage2ACandidate", anchor_term: str) -> Dict[str, Any]:
    """merge_stage2a_primary 时每个 tid 的初始合并项。"""
    return {
        "tid": cand.tid,
        "term": cand.term or str(cand.tid),
        "support_anchors": [anchor_term],
        "support_count": 1,
        "best_score": cand.composite_rank_score,
        "can_expand": cand.can_expand,
    }


def merge_stage2a_primary(all_anchor_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """跨锚汇总：同一 academic term(tid) 合并证据，按 support_count、best_score 排序。"""
    merged: Dict[int, Dict[str, Any]] = {}
    for block in all_anchor_results:
        anchor = block["anchor"]
        anchor_term = getattr(anchor, "anchor", "") or str(getattr(anchor, "vid", ""))
        for cand in block.get("primary", []):
            if cand.tid not in merged:
                merged[cand.tid] = _init_merged_primary(cand, anchor_term)
            else:
                merged[cand.tid]["support_anchors"].append(anchor_term)
                merged[cand.tid]["support_count"] += 1
                merged[cand.tid]["best_score"] = max(merged[cand.tid]["best_score"], cand.composite_rank_score)
                merged[cand.tid]["can_expand"] = merged[cand.tid]["can_expand"] or cand.can_expand
    return sorted(
        merged.values(),
        key=lambda x: (x["support_count"], x["best_score"]),
        reverse=True,
    )


def landing_candidates_to_stage2a(landing_list: List[LandingCandidate]) -> List[Stage2ACandidate]:
    """将 collect_landing_candidates 的 LandingCandidate 列表转为 Stage2ACandidate，供极简 Stage2A 流水线用。"""
    out: List[Stage2ACandidate] = []
    for c in landing_list:
        out.append(Stage2ACandidate(
            tid=c.vid,
            term=(c.term or "").strip() or str(c.vid),
            source=(getattr(c, "source", "") or "similar_to").strip(),
            semantic_score=float(getattr(c, "semantic_score", 0) or 0),
            context_sim=float(getattr(c, "context_sim", 0) or 0),
        ))
    return out


def _score_anchor_family_match(cand: Dict[str, Any], anchor: PreparedAnchor, _context: Dict[str, Any]) -> float:
    """是否仍在 anchor 语义族里：alias / 近义 canonical family，非黑名单。"""
    anchor_term = getattr(anchor, "anchor", "") or ""
    candidate_term = (cand.get("term") or "").strip()
    if not candidate_term:
        return 0.0
    return compute_anchor_identity_score(anchor_term, candidate_term, getattr(anchor, "anchor_type", None))


def _score_similarity_to_anchor_group(
    cand: Dict[str, Any],
    mainline_profile: Dict[str, Any],
    group_key: str,
    label: Any,
) -> float:
    """candidate 与主线/支线 anchor 群的接近程度（用 centroid）。"""
    tid = cand.get("tid")
    if tid is None:
        return 0.0
    vec = _get_candidate_vec_for_mainline(label, int(tid))
    if vec is None:
        return 0.0
    centroid = mainline_profile.get("mainline_centroid" if group_key == "mainline" else "bonus_centroid")
    if centroid is None:
        return 0.0
    return _cos_sim_mainline(vec, centroid)


def judge_anchor_validity(cand: Dict[str, Any], anchor: PreparedAnchor, context: Dict[str, Any]) -> bool:
    """
    是不是当前 anchor 的合理学术落点。解决 dynamism / control flow / simula / kinesiology / kinesics。
    """
    base_score = float(cand.get("base_score", 0) or 0)
    ctx_score = float(cand.get("ctx_score", 0) or 0)
    family_match = _score_anchor_family_match(cand, anchor, context)
    if base_score < STAGE2A_ANCHOR_BASE_MIN:
        return False
    if ctx_score > 0 and (ctx_score + STAGE2A_CTX_DROP_TOLERANCE < base_score):
        return False
    if family_match < STAGE2A_FAMILY_MATCH_MIN:
        return False
    return True


def judge_mainline_role(cand: Dict[str, Any], mainline_profile: Dict[str, Any], label: Any) -> str:
    """
    属于 JD 主线、支线还是偏题。mainline / side / off。
    解决 medical robotics、reinforcement learning、digital control 等“能解释但不该主导”。
    """
    mainline_sim = _score_similarity_to_anchor_group(cand, mainline_profile, "mainline", label)
    bonus_sim = _score_similarity_to_anchor_group(cand, mainline_profile, "bonus", label)
    if mainline_sim >= STAGE2A_MAINLINE_SIM_MAIN:
        return "mainline"
    if bonus_sim >= STAGE2A_BONUS_SIM_SIDE and mainline_sim < STAGE2A_MAINLINE_SIM_SIDE:
        return "side"
    if mainline_sim >= STAGE2A_MAINLINE_SIM_OFF:
        return "side"
    return "off"


def judge_expandability(cand: Dict[str, Any], anchor: PreparedAnchor, mainline_profile: Dict[str, Any], label: Any) -> bool:
    """
    保留后能否作为 Stage2B 扩散种子。robotic arm 可扩，motion control 可考虑，RL/medical robotics 不该扩。
    """
    family_match = _score_anchor_family_match(cand, anchor, {})
    object_risk = compute_object_like_risk(cand)
    mainline_sim = _score_similarity_to_anchor_group(cand, mainline_profile, "mainline", label)
    if family_match < STAGE2A_EXPAND_FAMILY_MIN:
        return False
    if mainline_sim < STAGE2A_EXPAND_MAINLINE_MIN:
        return False
    if object_risk > STAGE2A_EXPAND_OBJECT_RISK_MAX:
        return False
    return True


def bucket_stage2a_candidate(anchor_valid: bool, mainline_role: str, expandable: bool) -> str:
    """极简分桶：先判是不是锚点落点，再判主线/支线/偏题，再判能不能扩。"""
    if not anchor_valid:
        return "reject"
    if mainline_role == "off":
        return "reject"
    if mainline_role == "side":
        return "primary_keep_no_expand"
    if mainline_role == "mainline" and expandable:
        return "primary_expandable"
    return "primary_keep_no_expand"


def score_stage2a_candidate(cand: Dict[str, Any], mainline_role: str, expandable: bool) -> float:
    """轻量排序分：只用于同 bucket 内排序与 anchor 内冲突消解，不决定生死。"""
    score = float(cand.get("base_score", 0) or 0)
    if mainline_role == "mainline":
        score += 0.12
    elif mainline_role == "side":
        score -= 0.03
    if expandable:
        score += 0.08
    return max(0.0, min(1.0, score))


# ---------- Stage2A 主线画像与合并（极简：仅 mainline/bonus 分型） ----------


def build_stage2a_mainline_profile(
    anchors: List[PreparedAnchor],
    jd_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    极简主线画像：只构建 mainline_anchor_ids、bonus_anchor_ids、mainline_centroid、bonus_centroid。
    前几名高分 anchor 作主线，明显 bonus 倾向的归 side；不做复杂聚类。
    """
    vecs_main: List[Tuple[float, np.ndarray]] = []
    vecs_bonus: List[Tuple[float, np.ndarray]] = []
    mainline_anchor_ids: List[int] = []
    bonus_anchor_ids: List[int] = []
    for a in anchors:
        cv = getattr(a, "conditioned_vec", None)
        if cv is None:
            continue
        try:
            v = np.asarray(cv, dtype=np.float32).flatten()
            if v.size == 0:
                continue
        except Exception:
            continue
        vid = int(getattr(a, "vid", 0) or 0)
        w = float(getattr(a, "source_weight", 1.0))
        if w >= 0.85:
            vecs_main.append((w, v))
            mainline_anchor_ids.append(vid)
        else:
            vecs_bonus.append((w, v))
            bonus_anchor_ids.append(vid)
    mainline_centroid = None
    if vecs_main:
        weights = np.array([x[0] for x in vecs_main], dtype=np.float32)
        mat = np.array([x[1] for x in vecs_main], dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        weights = weights / (weights.sum() + 1e-9)
        mainline_centroid = np.average(mat, axis=0, weights=weights).astype(np.float32)
        norm = np.linalg.norm(mainline_centroid)
        if norm > 1e-9:
            mainline_centroid = mainline_centroid / norm
    if mainline_centroid is None and vecs_main:
        mainline_centroid = np.asarray(vecs_main[0][1], dtype=np.float32).flatten()
        n = np.linalg.norm(mainline_centroid)
        if n > 1e-9:
            mainline_centroid = mainline_centroid / n
    bonus_centroid = None
    if vecs_bonus:
        weights_b = np.array([x[0] for x in vecs_bonus], dtype=np.float32)
        mat_b = np.array([x[1] for x in vecs_bonus], dtype=np.float32)
        if mat_b.ndim == 1:
            mat_b = mat_b.reshape(1, -1)
        weights_b = weights_b / (weights_b.sum() + 1e-9)
        bonus_centroid = np.average(mat_b, axis=0, weights=weights_b).astype(np.float32)
        norm_b = np.linalg.norm(bonus_centroid)
        if norm_b > 1e-9:
            bonus_centroid = bonus_centroid / norm_b
    return {
        "mainline_anchor_ids": set(mainline_anchor_ids),
        "bonus_anchor_ids": set(bonus_anchor_ids),
        "mainline_centroid": mainline_centroid,
        "bonus_centroid": bonus_centroid,
    }


def unify_same_source_views(
    anchor: PreparedAnchor,
    landing_candidates: List[LandingCandidate],
    jd_profile: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    极简：同一 tid 的 base / conditioned 合并成一条 candidate 记录。
    只输出：tid, term, base_score, ctx_score, base_hit, ctx_hit, base_rank, ctx_rank。
    """
    merged: Dict[int, Dict[str, Any]] = {}
    base_rank_by_vid: Dict[int, int] = {}
    ctx_rank_by_vid: Dict[int, int] = {}
    base_seen = 0
    ctx_seen = 0
    for c in landing_candidates:
        tid = c.vid
        rec = merged.get(tid)
        if rec is None:
            rec = {
                "tid": tid,
                "term": (c.term or "").strip() or str(tid),
                "base_score": 0.0,
                "base_rank": None,
                "base_hit": False,
                "ctx_score": 0.0,
                "ctx_rank": None,
                "ctx_hit": False,
            }
            merged[tid] = rec
        src = (getattr(c, "source", "") or "").strip().lower()
        if src == "similar_to":
            rec["base_score"] = max(rec["base_score"], float(getattr(c, "semantic_score", 0) or 0))
            base_seen += 1
            if tid not in base_rank_by_vid:
                base_rank_by_vid[tid] = base_seen
            rec["base_rank"] = base_rank_by_vid[tid]
            rec["base_hit"] = True
            ctx_sim = float(getattr(c, "context_sim", 0) or 0)
            if ctx_sim > 0:
                rec["ctx_score"] = max(rec["ctx_score"], ctx_sim)
                if not rec["ctx_hit"]:
                    ctx_seen += 1
                    ctx_rank_by_vid[tid] = ctx_seen
                rec["ctx_rank"] = ctx_rank_by_vid.get(tid)
                rec["ctx_hit"] = True
        elif src == "conditioned_vec":
            rec["ctx_score"] = max(rec["ctx_score"], float(getattr(c, "semantic_score", 0) or 0))
            ctx_seen += 1
            if tid not in ctx_rank_by_vid:
                ctx_rank_by_vid[tid] = ctx_seen
            rec["ctx_rank"] = ctx_rank_by_vid[tid]
            rec["ctx_hit"] = True
    return list(merged.values())


def _get_candidate_vec_for_mainline(label: Any, tid: int) -> Optional[np.ndarray]:
    """取候选词向量供 mainline/bonus alignment 用；无 label 或无向量时返回 None。"""
    idx = getattr(label, "vocab_to_idx", None)
    if idx is None:
        return None
    i = idx.get(str(tid)) if isinstance(idx, dict) else idx.get(tid)
    if i is None:
        return None
    all_vecs = getattr(label, "all_vocab_vectors", None)
    if all_vecs is None:
        return None
    try:
        vec = np.asarray(all_vecs[i], dtype=np.float32)
        return vec.flatten() if vec.ndim != 1 else vec
    except Exception:
        return None


def _cos_sim_mainline(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度，限制到 [0, 1]。"""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    if a.size != b.size or a.size == 0:
        return 0.0
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n <= 1e-9:
        return 0.0
    return float(max(0.0, min(1.0, np.dot(a, b) / n)))


def _score_cross_anchor_support(
    feat: Dict[str, Any],
    all_anchors: List[PreparedAnchor],
    current_anchor: PreparedAnchor,
    label: Any,
) -> float:
    """候选得到多个主线锚点向量支持的程度（cross_anchor_support）。"""
    return _score_co_anchor_identity_support(feat, all_anchors, current_anchor, label)


def _score_family_resonance_with_mainline(
    feat: Dict[str, Any], mainline_profile: Dict[str, Any], label: Any
) -> float:
    """候选与主线簇的家族共振：与 mainline clusters 的覆盖度。"""
    clusters = mainline_profile.get("mainline_clusters") or []
    tid = feat.get("tid")
    if tid is None or not clusters:
        return 0.0
    vec = _get_candidate_vec_for_mainline(label, int(tid))
    if vec is None:
        return 0.0
    scores = []
    for c in clusters:
        cent = c.get("centroid")
        if cent is not None:
            scores.append(_cos_sim_mainline(vec, cent) * float(c.get("weight", 1.0)))
    if not scores:
        return 0.0
    scores.sort(reverse=True)
    return sum(scores[:2]) / 2.0 if len(scores) >= 2 else scores[0]


def compute_mainline_alignment(
    feat: Dict[str, Any],
    mainline_profile: Dict[str, Any],
    label: Any,
    anchor: Optional[PreparedAnchor] = None,
    all_anchors: Optional[List[PreparedAnchor]] = None,
) -> float:
    """
    候选与 Stage1 主线 anchor 群的一致性：sim_mainline + cross_anchor_support + family_resonance。
    回答：更像主线还是更像支线/奖励项/泛词分支。
    """
    centroid = mainline_profile.get("mainline_centroid")
    tid = feat.get("tid")
    if tid is None:
        return 0.0
    vec = _get_candidate_vec_for_mainline(label, int(tid))
    if vec is None:
        return 0.0
    sim_mainline = _cos_sim_mainline(vec, centroid) if centroid is not None else 0.0
    cross_anchor_support = 0.0
    if anchor is not None and all_anchors:
        cross_anchor_support = _score_cross_anchor_support(feat, all_anchors, anchor, label)
    family_resonance = _score_family_resonance_with_mainline(feat, mainline_profile, label)
    return _clip01(
        0.45 * sim_mainline
        + 0.30 * cross_anchor_support
        + 0.25 * family_resonance
    )


def compute_bonus_branch_alignment(feat: Dict[str, Any], mainline_profile: Dict[str, Any], label: Any) -> float:
    """候选与 bonus 支线簇的贴合度；高则易为奖励项支线。"""
    bonus = mainline_profile.get("bonus_clusters") or []
    tid = feat.get("tid")
    if tid is None or not bonus:
        return 0.0
    vec = _get_candidate_vec_for_mainline(label, int(tid))
    if vec is None:
        return 0.0
    scores = []
    for c in bonus:
        cent = c.get("centroid")
        if cent is not None:
            scores.append(_cos_sim_mainline(vec, cent) * float(c.get("weight", 1.0)))
    return max(scores) if scores else 0.0


def compute_view_stability(feat: Dict[str, Any], T: Optional[SimpleNamespace] = None) -> float:
    """同源双视角稳定性；降权为辅助项，不再主导。"""
    base = float(feat.get("base_score", 0) or 0)
    ctx = float(feat.get("ctx_score", 0) or 0)
    base_hit = bool(feat.get("base_hit"))
    ctx_hit = bool(feat.get("ctx_hit"))
    if not base_hit and not ctx_hit:
        return 0.0
    overlap = 1.0 if (base_hit and ctx_hit) else 0.6
    score_gap = abs(base - ctx)
    rank_gap = 0.0
    br, cr = feat.get("base_rank"), feat.get("ctx_rank")
    if br is not None and cr is not None:
        rank_gap = abs(int(br) - int(cr))
    return _clip01(
        overlap
        * (1.0 - min(1.0, score_gap / 0.20) * 0.45)
        * (1.0 - min(1.0, rank_gap / 5.0) * 0.25)
    )


def _score_family_alias_match(anchor: PreparedAnchor, feat: Dict[str, Any], _jd_profile: Optional[Dict[str, Any]]) -> float:
    anchor_term = getattr(anchor, "anchor", "") or ""
    candidate_term = (feat.get("term") or "").strip()
    if not candidate_term:
        return 0.0
    return compute_anchor_identity_score(anchor_term, candidate_term, getattr(anchor, "anchor_type", None))


def _score_co_anchor_identity_support(
    feat: Dict[str, Any],
    all_anchors: List[PreparedAnchor],
    current_anchor: PreparedAnchor,
    label: Any,
) -> float:
    """候选与其它主线锚点的协同度（向量相似度平均）。"""
    tid = feat.get("tid")
    if tid is None:
        return 0.0
    cand_vec = _get_candidate_vec_for_mainline(label, int(tid))
    if cand_vec is None:
        return 0.0
    sims = []
    for a in all_anchors:
        if getattr(a, "vid", None) == getattr(current_anchor, "vid", None):
            continue
        cv = getattr(a, "conditioned_vec", None)
        if cv is None:
            continue
        try:
            v = np.asarray(cv, dtype=np.float32).flatten()
            if v.size == cand_vec.size:
                sims.append(_cos_sim_mainline(cand_vec, v))
        except Exception:
            pass
    return float(np.mean(sims)) if sims else 0.0


def compute_anchor_candidate_identity(
    anchor: PreparedAnchor,
    feat: Dict[str, Any],
    jd_profile: Optional[Dict[str, Any]],
    all_anchors: Optional[List[PreparedAnchor]] = None,
    label: Any = None,
) -> float:
    """候选是否仍保持锚点身份；显式加入与其它主线 anchor 的协同度。"""
    base_anchor_match = float(feat.get("base_score", 0) or 0)
    stable_overlap = min(float(feat.get("base_score", 0) or 0), float(feat.get("ctx_score", 0) or 0))
    alias_family_match = _score_family_alias_match(anchor, feat, jd_profile)
    co_anchor_identity = 0.0
    if all_anchors and label:
        co_anchor_identity = _score_co_anchor_identity_support(feat, all_anchors, anchor, label)
    return _clip01(
        0.35 * base_anchor_match
        + 0.20 * stable_overlap
        + 0.20 * alias_family_match
        + 0.25 * co_anchor_identity
    )


def _compute_topic_path_proximity(feat: Dict[str, Any], _jd_profile: Optional[Dict[str, Any]]) -> float:
    return _clip01(0.5 * (float(feat.get("subfield_fit", 0) or 0) + float(feat.get("topic_fit", 0) or 0)))


def compute_hierarchy_consistency(feat: Dict[str, Any], jd_profile: Optional[Dict[str, Any]]) -> float:
    """
    候选与 JD 四层层级一致性：field / subfield / topic / path_match 四段，
    高 topic 但低 path_match 时施加惩罚（单一 topic 命中、路径不落地视为可疑）。
    """
    field_fit = float(feat.get("field_fit", 0) or 0)
    subfield_fit = float(feat.get("subfield_fit", 0) or 0)
    topic_fit = float(feat.get("topic_fit", 0) or 0)
    path_match = float(feat.get("path_match", 0) or 0)
    if path_match <= 0 and (subfield_fit > 0 or topic_fit > 0):
        path_match = _compute_topic_path_proximity(feat, jd_profile)
    raw = (
        0.20 * field_fit
        + 0.25 * subfield_fit
        + 0.25 * topic_fit
        + 0.30 * path_match
    )
    mismatch_penalty = 0.0
    if topic_fit >= 0.70 and path_match < 0.25:
        mismatch_penalty = 0.25
    return _clip01(raw * (1.0 - mismatch_penalty))


def compute_context_shift_quality(
    feat: Dict[str, Any],
    mainline_alignment: float,
) -> float:
    """上下文偏移只有当让候选更贴主线时才算正收益。"""
    gain = float(feat.get("shift_gain", 0) or 0)
    drop = float(feat.get("shift_drop", 0) or 0)
    jd_align = float(feat.get("jd_align", 0) or 0)
    gain_term = min(1.0, gain / 0.08) if 0.08 > 0 else 0.0
    drop_term = min(1.0, drop / 0.08) if 0.08 > 0 else 0.0
    return _clip01(
        0.50 * gain_term
        + 0.35 * mainline_alignment
        + 0.15 * jd_align
        - 0.45 * drop_term
    )


def compute_ambiguity_risk(feat: Dict[str, Any], jd_profile: Optional[Dict[str, Any]]) -> float:
    view_stability = float(feat.get("view_stability", 0.5) or 0.5)
    hierarchy = float(feat.get("hierarchy", 0.5) or 0.5)
    return _clip01((1.0 - view_stability) * 0.5 + (1.0 - hierarchy) * 0.5)


def compute_generic_risk(feat: Dict[str, Any], jd_profile: Optional[Dict[str, Any]]) -> float:
    hierarchy = float(feat.get("hierarchy", 0.5) or 0.5)
    jd_align = float(feat.get("jd_align", 0.5) or 0.5)
    return _clip01((1.0 - hierarchy) * 0.6 + jd_align * 0.2)


def compute_branch_drift_risk(
    mainline_alignment: float,
    bonus_alignment: float,
    generic_risk: float,
    object_like_risk: float,
) -> float:
    """
    支线漂移风险：主线一致性越低越高；bonus/支线一致性越高越高；泛词/对象词风险越高越高。
    用于在 Stage2A 压住 medical robotics、reinforcement learning 等支线。
    """
    risk = 0.0
    risk += max(0.0, 0.60 - mainline_alignment) * 0.45
    risk += bonus_alignment * 0.30
    risk += generic_risk * 0.15
    risk += object_like_risk * 0.10
    return _clip01(risk)


def _lexical_object_like_score(term: str) -> float:
    """词形上更像具体对象/器官/部件而非 canonical 学术概念。"""
    if not term:
        return 0.0
    t = term.lower()
    object_like = ("hand", "arm", "finger", "leg", "joint", "organ", "surgical robot", "manipulator")
    if any(x in t for x in object_like):
        return 0.5
    if len(t.split()) >= 4:
        return 0.2
    return 0.0


def _neighborhood_specificity_score(feat: Dict[str, Any]) -> float:
    """邻域细粒度：占位，无 label 时返回 0。"""
    return 0.0


def _hierarchy_tail_specificity(feat: Dict[str, Any]) -> float:
    """层级末端细粒度：topic 过细、更像具体对象层级。占位返回 0。"""
    return 0.0


def compute_object_like_risk(feat: Dict[str, Any]) -> float:
    """
    比 anchor 更细粒度、更像具体对象/器官/部件而非 canonical academic term 则加风险。
    综合：lexical + neighborhood + hierarchy_tail。
    """
    term = (feat.get("term") or "").strip()
    if not term:
        return 0.0
    lexical = _lexical_object_like_score(term)
    neighborhood = _neighborhood_specificity_score(feat)
    hierarchy_tail = _hierarchy_tail_specificity(feat)
    return _clip01(0.40 * lexical + 0.35 * neighborhood + 0.25 * hierarchy_tail)


def compute_candidate_risks(feat: Dict[str, Any], jd_profile: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    """返回 (ambiguity_risk, generic_risk)；branch_drift_risk 由 compute_branch_drift_risk(mainline, bonus, generic, object_like) 单独算。"""
    return (
        compute_ambiguity_risk(feat, jd_profile),
        compute_generic_risk(feat, jd_profile),
    )


def compute_canonicalness(
    feat: Dict[str, Any],
    anchor: PreparedAnchor,
    alias_match_fn=None,
) -> float:
    """
    Canonical 学术表达偏好：alias 家族匹配 + cross_anchor 支持 + path_match。
    用于抬升 motion control、压制 movement control 等非标准表述。
    """
    alias_match = _score_family_alias_match(anchor, feat, None) if alias_match_fn is None else alias_match_fn(feat)
    cross_anchor = float(feat.get("cross_anchor_support", feat.get("identity", 0) or 0))
    if cross_anchor <= 0 and feat.get("identity") is not None:
        cross_anchor = float(feat.get("identity", 0) or 0) * 0.5
    path_match = float(feat.get("path_match", 0) or 0)
    if path_match <= 0:
        path_match = 0.5 * (float(feat.get("subfield_fit", 0) or 0) + float(feat.get("topic_fit", 0) or 0))
    return _clip01(0.40 * alias_match + 0.30 * cross_anchor + 0.30 * path_match)


def score_stage2a_primary(feat: Dict[str, Any], W: Optional[SimpleNamespace] = None) -> float:
    """
    主线优先 + canonical 落点：mainline_alignment / identity / canonicalness 提权，
    view_stability 降为辅助；三项风险惩罚。
    """
    base = (
        0.22 * float(feat.get("mainline_alignment", 0) or 0)
        + 0.18 * float(feat.get("identity", 0) or 0)
        + 0.14 * float(feat.get("canonicalness", 0) or 0)
        + 0.14 * float(feat.get("base_score", 0) or 0)
        + 0.12 * float(feat.get("hierarchy", 0) or 0)
        + 0.08 * float(feat.get("jd_align", 0) or 0)
        + 0.06 * float(feat.get("context_shift_quality", 0) or 0)
        + 0.06 * float(feat.get("view_stability", 0) or 0)
    )
    penalty = (
        (1.0 - 0.30 * float(feat.get("branch_drift_risk", 0) or 0))
        * (1.0 - 0.20 * float(feat.get("object_like_risk", 0) or 0))
        * (1.0 - 0.20 * float(feat.get("ambiguity_risk", 0) or 0))
    )
    return _clip01(base * max(0.0, penalty))


def decide_stage2a_bucket(feat: Dict[str, Any], T: SimpleNamespace) -> str:
    """
    四分桶收紧：reject 明显错义/支线漂移；primary_expandable 需主线强+canonical+风险低；
    primary_keep_no_expand 主线相关且可解释；其余 observe_only。
    """
    mainline = float(feat.get("mainline_alignment", 0) or 0)
    identity = float(feat.get("identity", 0) or 0)
    branch_drift = float(feat.get("branch_drift_risk", 0) or 0)
    primary_score = float(feat.get("primary_score", 0) or 0)
    canonicalness = float(feat.get("canonicalness", 0) or 0)
    object_like = float(feat.get("object_like_risk", 0) or 0)

    if (
        mainline < getattr(T, "mainline_low", 0.25)
        and identity < getattr(T, "identity_low", 0.18)
        and branch_drift > getattr(T, "branch_drift_high", 0.65)
    ):
        return "reject"

    if (
        mainline >= getattr(T, "mainline_expand", 0.55)
        and identity >= getattr(T, "identity_expand", 0.55)
        and canonicalness >= getattr(T, "canonical_expand", 0.45)
        and branch_drift < getattr(T, "branch_drift_low", 0.35)
        and object_like < getattr(T, "object_like_low", 0.35)
        and primary_score >= getattr(T, "primary_expand", 0.50)
    ):
        return "primary_expandable"

    if (
        mainline >= getattr(T, "mainline_keep", 0.40)
        and primary_score >= getattr(T, "primary_keep", 0.35)
    ):
        return "primary_keep_no_expand"

    return "observe_only"


def calibrate_anchor_thresholds(evaluated_candidates: List[Dict[str, Any]], global_floor: SimpleNamespace) -> SimpleNamespace:
    """按锚点内分布得到相对阈值，与全局底线取 max；含 mainline_*、object_like_low。"""
    if not evaluated_candidates:
        return global_floor
    identity_vals = sorted([float(c.get("identity", 0) or 0) for c in evaluated_candidates])
    view_vals = sorted([float(c.get("view_stability", 0) or 0) for c in evaluated_candidates])
    primary_vals = sorted([float(c.get("primary_score", 0) or 0) for c in evaluated_candidates])
    mainline_vals = sorted([float(c.get("mainline_alignment", 0) or 0) for c in evaluated_candidates])
    n = len(identity_vals)
    p25 = max(0, n * 25 // 100)
    p60 = max(0, min(n - 1, n * 60 // 100))
    p75 = max(0, min(n - 1, n * 75 // 100))
    p80 = max(0, min(n - 1, n * 80 // 100))
    T = SimpleNamespace()
    T.identity_low = max(getattr(global_floor, "identity_low", 0.18), identity_vals[p25] if identity_vals else 0)
    T.identity_primary = max(getattr(global_floor, "identity_primary", 0.35), identity_vals[p60] if identity_vals else 0)
    T.identity_keep = max(getattr(global_floor, "identity_keep", 0.40), identity_vals[p60] if identity_vals else 0)
    T.identity_expand = max(getattr(global_floor, "identity_expand", 0.55), identity_vals[p75] if identity_vals else 0)
    T.view_stability_low = max(getattr(global_floor, "view_stability_low", 0.25), view_vals[p25] if view_vals else 0)
    T.view_stability_primary = max(getattr(global_floor, "view_stability_primary", 0.45), view_vals[p60] if view_vals else 0)
    T.view_stability_keep = max(getattr(global_floor, "view_stability_keep", 0.50), view_vals[p60] if view_vals else 0)
    T.view_stability_expand = max(getattr(global_floor, "view_stability_expand", 0.60), view_vals[p75] if view_vals else 0)
    T.primary_keep_line = max(getattr(global_floor, "primary_keep_line", 0.35), primary_vals[p60] if primary_vals else 0)
    T.primary_expand_line = max(getattr(global_floor, "primary_expand_line", 0.50), primary_vals[p80] if primary_vals else 0)
    T.primary_keep = max(getattr(global_floor, "primary_keep", 0.35), primary_vals[p60] if primary_vals else 0)
    T.primary_expand = max(getattr(global_floor, "primary_expand", 0.50), primary_vals[p80] if primary_vals else 0)
    canonical_vals = sorted([float(c.get("canonicalness", 0) or 0) for c in evaluated_candidates])
    T.canonical_expand = max(getattr(global_floor, "canonical_expand", 0.45), canonical_vals[p75] if canonical_vals else 0)
    T.mainline_low = max(getattr(global_floor, "mainline_low", 0.25), mainline_vals[p25] if mainline_vals else 0)
    T.mainline_keep = max(getattr(global_floor, "mainline_keep", 0.40), mainline_vals[p60] if mainline_vals else 0)
    T.mainline_expand = max(getattr(global_floor, "mainline_expand", 0.55), mainline_vals[p75] if mainline_vals else 0)
    T.base_expand_line = getattr(global_floor, "base_expand_line", 0.55)
    T.hierarchy_low = getattr(global_floor, "hierarchy_low", 0.20)
    T.hierarchy_mid = getattr(global_floor, "hierarchy_mid", 0.35)
    T.hierarchy_expand = getattr(global_floor, "hierarchy_expand", 0.50)
    T.shift_quality_low = getattr(global_floor, "shift_quality_low", 0.20)
    T.shift_quality_mid = getattr(global_floor, "shift_quality_mid", 0.40)
    T.shift_quality_expand = getattr(global_floor, "shift_quality_expand", 0.55)
    T.ambiguity_high = getattr(global_floor, "ambiguity_high", 0.70)
    T.ambiguity_mid = getattr(global_floor, "ambiguity_mid", 0.50)
    T.ambiguity_low = getattr(global_floor, "ambiguity_low", 0.35)
    T.generic_mid = getattr(global_floor, "generic_mid", 0.45)
    T.generic_low = getattr(global_floor, "generic_low", 0.35)
    T.branch_drift_high = getattr(global_floor, "branch_drift_high", 0.65)
    T.branch_drift_mid = getattr(global_floor, "branch_drift_mid", 0.45)
    T.branch_drift_low = getattr(global_floor, "branch_drift_low", 0.35)
    T.object_like_low = getattr(global_floor, "object_like_low", 0.35)
    T.jd_align_mid = getattr(global_floor, "jd_align_mid", 0.55)
    T.max_reasonable_shift = getattr(global_floor, "max_reasonable_shift", 0.25)
    T.max_reasonable_rank_gap = getattr(global_floor, "max_reasonable_rank_gap", 8)
    T.max_useful_gain = getattr(global_floor, "max_useful_gain", 0.20)
    T.max_tolerable_drop = getattr(global_floor, "max_tolerable_drop", 0.15)
    return T


def evaluate_stage2a_candidate(
    anchor: PreparedAnchor,
    cand: Dict[str, Any],
    jd_profile: Optional[Dict[str, Any]],
    mainline_profile: Optional[Dict[str, Any]] = None,
    all_anchors: Optional[List[PreparedAnchor]] = None,
    label: Any = None,
    thresholds: Optional[SimpleNamespace] = None,
    weights: Optional[SimpleNamespace] = None,
) -> Dict[str, Any]:
    """
    极简 Stage2A：只做 3 个判断 + 分桶 + 轻量排序分。
    先判「是不是这个锚点的合理学术落点」→ 再判「是不是 JD 主线」→ 再判「能不能扩散」。
    """
    context = jd_profile if isinstance(jd_profile, dict) else {}
    anchor_valid = judge_anchor_validity(cand, anchor, context)
    if not anchor_valid:
        return {
            **cand,
            "anchor_valid": False,
            "mainline_role": "off",
            "expandable": False,
            "bucket": "reject",
            "score": float(cand.get("base_score", 0) or 0),
            "primary_score": float(cand.get("base_score", 0) or 0),
            "identity": _score_anchor_family_match(cand, anchor, context),
        }
    mainline_role = "off"
    if mainline_profile and label:
        mainline_role = judge_mainline_role(cand, mainline_profile, label)
    if mainline_role == "off":
        return {
            **cand,
            "anchor_valid": True,
            "mainline_role": "off",
            "expandable": False,
            "bucket": "reject",
            "score": float(cand.get("base_score", 0) or 0),
            "primary_score": float(cand.get("base_score", 0) or 0),
            "identity": _score_anchor_family_match(cand, anchor, context),
        }
    expandable = False
    if mainline_role == "mainline" and mainline_profile and label:
        expandable = judge_expandability(cand, anchor, mainline_profile, label)
    bucket = bucket_stage2a_candidate(anchor_valid=anchor_valid, mainline_role=mainline_role, expandable=expandable)
    score = score_stage2a_candidate(cand=cand, mainline_role=mainline_role, expandable=expandable)
    return {
        **cand,
        "anchor_valid": anchor_valid,
        "mainline_role": mainline_role,
        "expandable": expandable,
        "bucket": bucket,
        "score": score,
        "primary_score": score,
        "identity": _score_anchor_family_match(cand, anchor, context),
    }


def resolve_anchor_primary_candidates(
    anchor: PreparedAnchor,
    evaluated_candidates: List[Dict[str, Any]],
    jd_profile: Optional[Dict[str, Any]],
    mainline_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    极简冲突消解：只看 bucket 再 score。每个 anchor 最多 1 个 keep_no_expand + 1 个 expandable。
    """
    ordered = sorted(
        evaluated_candidates,
        key=lambda x: (
            x.get("bucket") == "primary_expandable",
            x.get("bucket") == "primary_keep_no_expand",
            float(x.get("score", x.get("primary_score", 0)) or 0),
        ),
        reverse=True,
    )
    rejects: List[Dict[str, Any]] = []
    keep_no_expand: List[Dict[str, Any]] = []
    expandable: List[Dict[str, Any]] = []
    for row in ordered:
        b = row.get("bucket", "reject")
        if b == "reject":
            rejects.append(row)
        elif b == "primary_keep_no_expand":
            keep_no_expand.append(row)
        elif b == "primary_expandable":
            expandable.append(row)
    keep_no_expand = sorted(keep_no_expand, key=lambda x: -(float(x.get("score", 0) or 0)))[:1]
    expandable = sorted(expandable, key=lambda x: -(float(x.get("score", 0) or 0)))[:1]
    return {
        "rejects": rejects,
        "observe_only": [],
        "primary_keep_no_expand": keep_no_expand,
        "primary_expandable": expandable,
        "debug_rows": rejects + keep_no_expand + expandable,
    }


def select_stage2b_seeds(bucketed: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """仅 primary_expandable 可进入 Stage2B 扩散。"""
    return [x for x in (bucketed.get("primary_expandable") or []) if x.get("expandable")]


# ---------- Identity Gate：候选与锚点"本义"一致性，用于压制错义（propulsion/kinesics/simula 等） ----------候选与锚点“本义”一致性，用于压制错义（propulsion/kinesics/simula 等） ----------
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
    "路径规划": {"route planning", "path planning", "motion planning", "trajectory planning"},
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
    候选与锚点是否「本义同一概念家族」的 0~1 分。
    这里的 identity 只用于「软区分本义/偏义」，不再当成中英映射任务里的硬门。
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
    alias_hit = False
    for _anchor, aliases in ANCHOR_IDENTITY_ALIASES.items():
        an_norm = normalize_identity_surface(_anchor)["norm"]
        if an_norm and a_norm and (an_norm == a_norm or normalize_identity_surface(_anchor)["token_set"] == atok):
            for al in aliases:
                al_norm = normalize_identity_surface(al)["norm"]
                if al_norm and (al_norm == c_norm or al_norm in c_norm or c_norm in al_norm):
                    token_overlap_score = max(token_overlap_score, 0.72)
                    alias_hit = True
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

    # 中英映射 / 锚点别名命中时，给一个较高的基础分，避免全体被压成 low identity
    base_floor = 0.0
    if alias_hit:
        base_floor = max(base_floor, 0.34)
    if a_norm and c_norm and any(t in c_norm for t in a_norm.split()):
        base_floor = max(base_floor, 0.26)
    if token_overlap_score >= 0.45 and head_consistency_score >= 0.6:
        base_floor = max(base_floor, 0.30)

    score = max(
        base_floor,
        0.40 * exact_or_substring
        + 0.35 * token_overlap_score
        + 0.25 * head_consistency_score
    )

    # 泛词/歧义惩罚：保留，但不再把正常中英映射一刀砍死
    generic_penalty = 1.0
    if ctok and len(ctok) <= 2:
        low_tokens = {t.lower() for t in c["tokens"]}
        if low_tokens & IDENTITY_AMBIGUITY_TERMS and not (atok & ctok):
            generic_penalty = 0.72
    ambiguity_penalty = 1.0
    c_head_lower = c["head"].lower() if c["head"] else ""
    if c_head_lower in IDENTITY_AMBIGUITY_TERMS and c_head_lower not in atok:
        ambiguity_penalty = 0.78
    if (c_norm in ("control (management)", "control flow", "data retrieval", "crawling",
                   "point-to-point", "end-to-end principle", "simula", "kinesics", "propulsion") and
            not (atok & ctok)):
        ambiguity_penalty = min(ambiguity_penalty, 0.45)

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
    """
    Stage2A:
    1) 先用 similar_to 给初始候选
    2) 再用 conditioned_vec 做上下文纠偏（重打分信号）
    3) 主池太弱时，才少量补 context_fallback
    """
    # A. 初始候选：similar_to
    similar_to_candidates = retrieve_academic_term_by_similar_to(
        label, anchor,
        active_domain_set=active_domain_set,
        jd_field_ids=jd_field_ids,
        jd_subfield_ids=jd_subfield_ids,
        jd_topic_ids=jd_topic_ids,
        top_k=STAGE2A_COLLECT_BASE_TOP_K,
    )
    # B. 上下文纠偏：conditioned_vec
    context_neighbors, rerank_signals = _retrieve_academic_terms_by_conditioned_vec(
        label, anchor,
        similar_to_candidates=similar_to_candidates,
        active_domain_set=active_domain_set,
        jd_field_ids=jd_field_ids,
        jd_subfield_ids=jd_subfield_ids,
        jd_topic_ids=jd_topic_ids,
    )
    # 供 Stage2B dense support gate 使用：锚点 context 邻域与得分
    setattr(anchor, "_context_neighbors", context_neighbors)
    setattr(anchor, "_context_score_map", {c.vid: getattr(c, "context_sim", getattr(c, "semantic_score", 0.0)) for c in context_neighbors})
    candidates: List[LandingCandidate] = []
    existing_vids: Set[int] = set()
    # C. similar_to 仍是主候选池，附加 context 信号
    for cand in similar_to_candidates:
        sig = rerank_signals.get(cand.vid, {})
        cand.context_sim = float(sig.get("context_sim", 0.0) or 0.0)
        cand.context_supported = bool(sig.get("context_supported", 0.0) >= 1.0)
        cand.context_gap = float(sig.get("context_gap", 1.0) or 1.0)
        cand.source_role = "seed_candidate"
        candidates.append(cand)
        existing_vids.add(cand.vid)
    # D. 只有主池太弱时，才从 context_neighbors 补 1~2 个 fallback
    if len(candidates) <= 1:
        added = 0
        for cand in context_neighbors:
            if cand.vid in existing_vids:
                continue
            if cand.context_sim < 0.82:
                continue
            candidates.append(cand)
            existing_vids.add(cand.vid)
            added += 1
            if added >= 2:
                break
    cands = candidates
    for c in cands:
        c.domain_fit = _compute_domain_fit(
            label, c.vid,
            active_domain_set=active_domain_set,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
        )
        if getattr(c, "soft_domain_retain", False):
            c.domain_fit = (getattr(c, "domain_fit", 1.0) or 1.0) * 0.85
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
    if STAGE2_VERBOSE_DEBUG:
        print("[Stage2A 候选明细] tid | term | source | semantic_score | context_sim | context_supported | context_gap")
        for i, c in enumerate(cands[:10], 1):
            print(
                f"  {i} {c.vid} | {c.term!r} | {c.source} "
                f"| sem={getattr(c, 'semantic_score', 0):.3f}"
                f" | ctx={getattr(c, 'context_sim', 0):.3f}"
                f" | ctx_ok={getattr(c, 'context_supported', False)}"
                f" | gap={getattr(c, 'context_gap', 1.0):.3f}"
            )
        if len(cands) > 10:
            print(f"  ... 共 {len(cands)} 条")
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
    """从词汇向量索引取 primary 的学术近邻；候选须过 support_expandable_for_anchor 四道门（primary/anchor/context/family）。"""
    top_k_per_primary = top_k_per_primary or DENSE_MAX_PER_PRIMARY
    if not primary_landings or not getattr(label, "vocab_index", None) or not getattr(label, "vocab_to_idx", None):
        return []
    if getattr(label, "all_vocab_vectors", None) is None:
        return []
    load_vocab_meta(label)
    seen = set(p.vid for p in primary_landings)
    out = []
    support_domain_min = max(SUPPORT_MIN_DOMAIN_FIT, 0.72)

    anchor_to_primaries: Dict[int, List[Any]] = {}
    for p in primary_landings:
        a_vid = getattr(p, "anchor_vid", 0)
        anchor_to_primaries.setdefault(a_vid, []).append(p)

    for p in primary_landings:
        ok, seed_score, _ = check_seed_eligibility(label, p, jd_profile)
        if not ok:
            continue
        max_keep = 2
        if seed_score >= 0.40 and int(getattr(p, "cross_anchor_support_count", 1) or 1) >= 2:
            max_keep = 3
        idx = label.vocab_to_idx.get(str(p.vid))
        if idx is None:
            continue
        vec = np.asarray(label.all_vocab_vectors[idx], dtype=np.float32).reshape(1, -1)
        k = min(top_k_per_primary + 6, 20)
        scores, ids = label.vocab_index.search(vec, k)
        kept = 0
        anchor_term = getattr(p, "anchor_term", "") or ""

        anchor_stub = SimpleNamespace(
            anchor_term=anchor_term,
            anchor_vid=getattr(p, "anchor_vid", 0),
            conditioned_vec=getattr(p, "anchor_conditioned_vec", None),
            _context_score_map=getattr(p, "_context_score_map", None) or {},
            _context_neighbors=getattr(p, "_context_neighbors", None) or [],
            _stage2b_anchor_primaries=anchor_to_primaries.get(getattr(p, "anchor_vid", 0), []),
        )

        for score, tid in zip(scores[0], ids[0]):
            if kept >= max_keep:
                break
            tid = int(tid)
            if tid <= 0 or tid in seen:
                continue
            meta = label._vocab_meta.get(tid, ("", ""))
            term = (meta[0] or "").strip() or str(tid)
            vocab_type = meta[1] or ""
            if vocab_type not in ("concept", "keyword", "") and vocab_type:
                continue
            sim = max(0.0, min(1.0, float(score)))
            if sim < 0.55:
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
            if domain_fit < support_domain_min:
                continue

            keep, keep_meta = support_expandable_for_anchor(
                label=label,
                anchor=anchor_stub,
                parent_primary=p,
                candidate_vid=tid,
                candidate_term=term,
            )
            if not keep:
                if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
                    print(
                        f"[Stage2B dense reject] anchor={anchor_stub.anchor_term!r} "
                        f"parent={getattr(p, 'term', '')!r} cand={term!r} "
                        f"reason={keep_meta.get('reason')}"
                    )
                continue

            if is_device_or_object_term(term) and not anchor_allows_device_expansion(anchor_term):
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
            keep_score = float(keep_meta.get("keep_score", sim))
            out.append(
                ExpandedTermCandidate(
                    vid=tid,
                    term=term,
                    term_role="dense_expansion",
                    identity_score=keep_score,
                    source="dense",
                    anchor_vid=p.anchor_vid,
                    anchor_term=p.anchor_term,
                    semantic_score=sim,
                    src_vids=[p.vid],
                    hit_count=1,
                    parent_primary=p.term,
                )
            )
            kept += 1
            if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
                print(
                    f"[stage2b_expanded] tid={tid} term={term!r} source_type=dense "
                    f"parent_anchor={p.anchor_term!r} parent_primary={p.term!r} "
                    f"score={sim:.3f} keep_score={keep_score:.3f}"
                )
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
    """Cluster 扩散当前默认关闭（全关），避免脏簇成员混入。term_role=cluster_expansion。"""
    return []


def expand_from_cooccurrence_support(
    label,
    primary_landings: List[PrimaryLanding],
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
    jd_profile: Optional[Dict[str, Any]] = None,
) -> List[ExpandedTermCandidate]:
    """共现支持词；仅允许强 normal + 多锚 + 高 jd_align 的 seed，support 须过 support_expandable_for_anchor，每 seed 最多 2 条。"""
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
        and float(getattr(p, "primary_score", 0) or 0) >= 0.70
        and float(getattr(p, "jd_align", 0) or 0) >= 0.82
        and int(getattr(p, "cross_anchor_support_count", 1) or 1) >= 2
    ]
    anchor_to_primaries: Dict[int, List[Any]] = {}
    for p in primary_landings:
        a_vid = getattr(p, "anchor_vid", 0)
        anchor_to_primaries.setdefault(a_vid, []).append(p)
    out = []
    seen = set(p.vid for p in primary_landings)
    cooc_min_freq = max(COOC_SUPPORT_MIN_FREQ, 2)
    for p in strong_primaries:
        term = getattr(p, "term", "") or ""
        if not term:
            continue
        try:
            rows = label.stats_conn.execute(
                "SELECT term_a, term_b, freq FROM vocabulary_cooccurrence WHERE (term_a = ? OR term_b = ?) AND freq >= ?",
                (term, term, cooc_min_freq),
            ).fetchall()
        except Exception:
            continue
        anchor_stub = SimpleNamespace(
            anchor_term=getattr(p, "anchor_term", "") or "",
            anchor_vid=getattr(p, "anchor_vid", 0),
            conditioned_vec=getattr(p, "anchor_conditioned_vec", None),
            _context_score_map=getattr(p, "_context_score_map", None) or {},
            _context_neighbors=getattr(p, "_context_neighbors", None) or [],
            _stage2b_anchor_primaries=anchor_to_primaries.get(getattr(p, "anchor_vid", 0), []),
        )
        kept = 0
        for row in rows:
            if kept >= 2:
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
            if (freq or 0) < 3:
                continue
            cooc_strength = min(1.0, float(freq or 0) / 5.0)
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
            if domain_fit < 0.75:
                continue
            keep, _ = support_expandable_for_anchor(
                label=label,
                anchor=anchor_stub,
                parent_primary=p,
                candidate_vid=int(vid_other),
                candidate_term=other,
            )
            if not keep:
                continue
            domain_span = 0
            if getattr(label, "stats_conn", None):
                r = label.stats_conn.execute(
                    "SELECT domain_span FROM vocabulary_domain_stats WHERE voc_id=?",
                    (vid_other,),
                ).fetchone()
                if r:
                    domain_span = int(r[0] or 0)
            if domain_span > DOMAIN_SPAN_EXTREME:
                continue
            seen.add(vid_other)
            meta = label._vocab_meta.get(vid_other, ("", ""))
            out.append(
                ExpandedTermCandidate(
                    vid=vid_other,
                    term=meta[0] or other,
                    term_role="cooc_expansion",
                    identity_score=cooc_strength,
                    source="cooc",
                    anchor_vid=getattr(p, "anchor_vid", 0),
                    anchor_term=getattr(p, "anchor_term", "") or "",
                    semantic_score=cooc_strength,
                    src_vids=[getattr(p, "vid", 0)],
                    hit_count=int(freq),
                    parent_primary=term,
                )
            )
            kept += 1
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
                # 强冲突：主领域在医学/社科/管理等且与激活领域无交时硬拒（见 README Stage2A 漏点修复）
                if not domain_ok and expanded and STRONG_CONFLICT_DOMAIN_IDS:
                    main_domain = max(expanded, key=expanded.get)
                    if str(main_domain) in STRONG_CONFLICT_DOMAIN_IDS:
                        return (False, "domain_conflict_strong")

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


def _get_vocab_vec(label: Any, vid: int) -> Optional[np.ndarray]:
    """取词汇向量，供 dense support gate 使用。"""
    idx = getattr(label, "vocab_to_idx", None)
    if idx is None:
        return None
    i = idx.get(str(vid)) if isinstance(idx, dict) else None
    if i is None:
        i = idx.get(vid)
    if i is None:
        return None
    all_vecs = getattr(label, "all_vocab_vectors", None)
    if all_vecs is None:
        return None
    try:
        vec = np.asarray(all_vecs[i], dtype=np.float32)
    except Exception:
        return None
    if vec.ndim != 1:
        vec = vec.flatten()
    return vec


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度，限制到 [0, 1]。"""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    if a.size != b.size or a.size == 0:
        return 0.0
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n <= 1e-9:
        return 0.0
    return float(max(0.0, min(1.0, np.dot(a, b) / n)))


def _estimate_context_support(
    label: Any,
    anchor: Any,
    candidate_vid: int,
    candidate_vec: Optional[np.ndarray] = None,
) -> float:
    """
    候选是否也出现在 anchor 的 conditioned top-k 附近。
    优先查缓存的 _context_score_map；没有则和 _context_neighbors 比局部相似度。
    """
    context_score_map = getattr(anchor, "_context_score_map", None) or {}
    if candidate_vid in context_score_map:
        try:
            return float(context_score_map[candidate_vid])
        except Exception:
            pass
    context_neighbors = getattr(anchor, "_context_neighbors", None) or []
    if not context_neighbors:
        return 0.0
    if candidate_vec is None:
        candidate_vec = _get_vocab_vec(label, candidate_vid)
    if candidate_vec is None:
        return 0.0
    best = 0.0
    for n in context_neighbors:
        n_vid = getattr(n, "vid", n) if not isinstance(n, int) else n
        n_vec = _get_vocab_vec(label, int(n_vid))
        if n_vec is None:
            continue
        best = max(best, _cos_sim(candidate_vec, n_vec))
    return best


def _estimate_anchor_family_support(
    label: Any,
    candidate_vec: np.ndarray,
    surviving_primaries: List[Any],
) -> float:
    """
    候选是否得到同锚 surviving primary family 的共同支持。
    防止只贴某一个旁支 primary（drift_penalty）。
    """
    if not surviving_primaries:
        return 0.0
    sims: List[float] = []
    for p in surviving_primaries:
        p_vid = getattr(p, "vid", None)
        if p_vid is None:
            continue
        p_vec = _get_vocab_vec(label, int(p_vid))
        if p_vec is None:
            continue
        sims.append(_cos_sim(candidate_vec, p_vec))
    if not sims:
        return 0.0
    mean_sim = sum(sims) / len(sims)
    max_sim = max(sims)
    min_sim = min(sims)
    drift_penalty = max(0.0, max_sim - min_sim)
    score = mean_sim - 0.25 * drift_penalty
    return max(0.0, min(1.0, score))


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


# ---------- Stage2B 双层门：seed gate（谁能扩）+ support gate（扩出来的词谁能留） ----------

# 禁止作为扩散 seed 的窄方法/器件支线词（做 primary 可保留，做 seed 会带偏 dense/cluster）
NARROW_METHOD_OR_BRANCH_TERMS = frozenset({
    "q-learning", "digital control", "automatic control", "route planning",
    "pathfinding", "servo control", "machine control", "instrument control",
    "radio control", "electronic control unit", "automatic train control",
    "automatic frequency control", "digitally controlled oscillator",
})

# 设备/对象/组件/应用支线词：dense 默认不进入 support，除非锚点本身允许
DEVICE_OBJECT_TERM_PATTERNS = (
    "oscillator", "radio control", "train control", "unit", "ecu",
    "instrument control", "comparator", "discharge machining", "frequency",
    "decimal", "design strategy", "two-sided market", "protocol", "resolution",
    "social computing", "social software", "electrical discharge",
    "frenet", "serret", "center frequency",
)


def is_narrow_method_term(primary_term: str) -> bool:
    """是否属于窄方法/支线词，不宜作为扩散 seed（seed_expand_factor 惩罚）。"""
    if not primary_term:
        return False
    t = (primary_term or "").strip().lower()
    if t in NARROW_METHOD_OR_BRANCH_TERMS:
        return True
    for k in NARROW_METHOD_OR_BRANCH_TERMS:
        if k in t or t in k:
            return True
    return False


def is_device_or_object_term(term: str) -> bool:
    """是否像设备/对象/组件/应用支线词，dense 默认不进入 support。"""
    if not term:
        return False
    t = (term or "").strip().lower()
    for pat in DEVICE_OBJECT_TERM_PATTERNS:
        if pat in t:
            return True
    return False


def anchor_allows_device_expansion(anchor_text: str) -> bool:
    """锚点是否允许设备类扩展（如「医疗机器人」可带出部分设备词）。当前保守：一律不允许。"""
    if not anchor_text:
        return False
    a = (anchor_text or "").strip().lower()
    # 若以后要对「医疗机器人」「手术机器人」等开放，在此加白名单
    return False


def is_primary_expandable(
    anchor_text: str,
    primary_term: str,
    primary_score: float,
    anchor_identity: float,
    jd_align: float,
    support_count: int,
    source_type: str,
) -> bool:
    """
    只有「适合扩散的 primary」才能扩。
    允许：本体主词（motion control, robot control, medical robotics, robotic arm, reinforcement learning）、
    多锚支持且上下文稳定的骨干词。
    禁止：q-learning、digital control、automatic control、route planning 等窄支线；
    任何 weak_retain + suppress_seed=True。
    """
    anchor = (anchor_text or "").strip().lower()
    term = (primary_term or "").strip().lower()
    if not term:
        return False
    # 窄方法/支线词禁止扩散
    if is_narrow_method_term(primary_term):
        return False
    # 本体主词或骨干词：identity 或 多锚+jd 稳定
    if support_count >= 2 and jd_align >= 0.78 and primary_score >= 0.35:
        return True
    if anchor_identity >= 0.40 and jd_align >= 0.75:
        return True
    # 主词白名单式放行
    expandable_heads = (
        "motion control", "robot control", "medical robotics", "robotic arm",
        "reinforcement learning", "path planning", "simulation", "control engineering",
    )
    if any(h in term for h in expandable_heads) and anchor_identity >= 0.22:
        return True
    return False


def support_expandable_for_anchor(
    label: Any,
    anchor: Any,
    parent_primary: Any,
    candidate_vid: int,
    candidate_term: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Dense support 最小修复 gate：不靠硬编码词表，只靠语义一致性 + 上下文稳定性 + family 支撑。
    四道门：primary_consistency、anchor_consistency、context_stability、family_support。
    返回 (keep, meta)；meta 含 reason、keep_score、各分量。
    """
    candidate_vec = _get_vocab_vec(label, candidate_vid)
    if candidate_vec is None:
        return False, {"reason": "missing_candidate_vec"}

    primary_vid = getattr(parent_primary, "vid", None)
    if primary_vid is None:
        return False, {"reason": "missing_primary_vid"}
    primary_vec = _get_vocab_vec(label, int(primary_vid))
    if primary_vec is None:
        return False, {"reason": "missing_primary_vec"}

    primary_consistency = _cos_sim(primary_vec, candidate_vec)

    anchor_consistency = 0.0
    conditioned = getattr(anchor, "conditioned_vec", None)
    if conditioned is not None:
        try:
            cv = np.asarray(conditioned, dtype=np.float32).flatten()
            if cv.size > 0:
                anchor_consistency = _cos_sim(cv, candidate_vec)
        except Exception:
            pass
    has_anchor_vec = conditioned is not None
    if not has_anchor_vec:
        anchor_consistency = 1.0  # 无 conditioned_vec 时跳过此项门控

    context_stability = _estimate_context_support(
        label=label,
        anchor=anchor,
        candidate_vid=candidate_vid,
        candidate_vec=candidate_vec,
    )
    ctx_map = getattr(anchor, "_context_score_map", None) or {}
    ctx_neighbors = getattr(anchor, "_context_neighbors", None) or []
    has_context_data = bool(ctx_neighbors) or (candidate_vid in ctx_map)
    if not has_context_data:
        context_stability = 1.0  # 无 context 数据时跳过此项门控

    surviving_primaries = getattr(anchor, "_stage2b_anchor_primaries", None) or []
    family_support = _estimate_anchor_family_support(
        label=label,
        candidate_vec=candidate_vec,
        surviving_primaries=surviving_primaries,
    )
    if not surviving_primaries:
        family_support = 1.0  # 无同锚 primary 时跳过此项门控

    if primary_consistency < DENSE_SUPPORT_PRIMARY_CONSISTENCY_MIN:
        return False, {
            "reason": "low_primary_consistency",
            "primary_consistency": primary_consistency,
        }
    if has_anchor_vec and anchor_consistency < DENSE_SUPPORT_ANCHOR_CONSISTENCY_MIN:
        return False, {
            "reason": "low_anchor_consistency",
            "anchor_consistency": anchor_consistency,
        }
    if has_context_data and context_stability < DENSE_SUPPORT_CONTEXT_STABILITY_MIN:
        return False, {
            "reason": "low_context_stability",
            "context_stability": context_stability,
        }
    if surviving_primaries and family_support < DENSE_SUPPORT_FAMILY_SUPPORT_MIN:
        return False, {
            "reason": "low_family_support",
            "family_support": family_support,
        }

    # 新增：dense support 对锚点的 identity 约束，抑制 robotic arm -> robotic hand 等 family 漂移
    anchor_term = getattr(anchor, "anchor_term", "") or getattr(anchor, "anchor", "") or ""
    anchor_identity_for_support = compute_anchor_identity_score(anchor_term, candidate_term)
    if anchor_identity_for_support < 0.16:
        return False, {"reason": "low_anchor_identity_support", "anchor_identity_for_support": anchor_identity_for_support}

    # 新增：family 漂移惩罚（候选只极贴某一 primary 而对整锚支撑不均衡时降分）
    drift_penalty = max(0.0, primary_consistency - family_support)

    keep_score = (
        0.26 * primary_consistency
        + 0.26 * anchor_consistency
        + 0.22 * context_stability
        + 0.16 * family_support
        + 0.10 * anchor_identity_for_support
    )
    keep_score = keep_score - 0.18 * drift_penalty

    if keep_score < 0.76:
        return False, {
            "reason": "keep_score_too_low",
            "keep_score": keep_score,
            "primary_consistency": primary_consistency,
            "anchor_consistency": anchor_consistency,
            "context_stability": context_stability,
            "family_support": family_support,
            "anchor_identity_for_support": anchor_identity_for_support,
            "drift_penalty": drift_penalty,
        }

    return True, {
        "reason": "ok",
        "keep_score": keep_score,
        "primary_consistency": primary_consistency,
        "anchor_consistency": anchor_consistency,
        "context_stability": context_stability,
        "family_support": family_support,
        "anchor_identity_for_support": anchor_identity_for_support,
        "drift_penalty": drift_penalty,
    }


def compute_context_consistency(c: LandingCandidate) -> float:
    """
    raw semantic_score 与 conditioned_vec context_sim 的一致性。
    越一致，说明这个词在「无上下文近邻」和「有上下文近邻」里都站得住。
    """
    raw_sim = max(0.0, min(1.0, float(getattr(c, "semantic_score", 0.0) or 0.0)))
    ctx_sim = max(0.0, min(1.0, float(getattr(c, "context_sim", 0.0) or 0.0)))
    if ctx_sim <= 0.0:
        return raw_sim * 0.65
    consistency = 1.0 - abs(raw_sim - ctx_sim)
    score = 0.45 * raw_sim + 0.45 * ctx_sim + 0.10 * consistency
    return max(0.0, min(1.0, score))


def check_primary_eligibility(
    anchor: PreparedAnchor,
    c: LandingCandidate,
    hier_ev: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """资格赛：只拦特别危险的，边界情况留给 admission 做 weak_retain。返回 (eligible, reasons)。"""
    reasons: List[str] = []
    aid = float(getattr(c, "anchor_identity_score", 0.0) or 0.0)
    ctx_sim = float(getattr(c, "context_sim", 0.0) or 0.0)
    source_role = (getattr(c, "source_role", "") or "").strip()

    # identity 不再作为普遍硬门；只有非常低且上下文也不支持时才拦
    if aid < 0.20 and ctx_sim < 0.80:
        reasons.append("identity_too_low")
    # similar_to 候选：上下文明显反对时再拦；边界情况留给 admission 做 weak_retain
    if c.source == "similar_to" and ctx_sim > 0.0 and ctx_sim < 0.70:
        reasons.append("context_not_supporting_similar_to")
    # context fallback：仍保守，但降低 identity 门槛
    if source_role == "context_fallback":
        if ctx_sim < 0.82:
            reasons.append("context_fallback_not_strong_enough")
        if aid < 0.32 and ctx_sim < 0.88:
            reasons.append("context_fallback_identity_weak")
    domain_reason = getattr(c, "domain_reason", "") or ""
    if domain_reason == "domain_conflict_strong":
        reasons.append("domain_conflict_strong")
    return len(reasons) == 0, reasons


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
    """Stage2A 主落点准入：上下文稳定性优先，identity 只做软约束。"""
    reasons: List[str] = []
    rescued = False
    meta: Dict[str, Any] = {
        "retain_mode": "reject",
        "suppress_seed": False,
        "retain_reason": None,
    }
    if not _lexical_term_sanity(getattr(candidate, "term", "") or "", None):
        return False, ["lexical_not_term"], rescued, meta

    eligible, eligibility_reasons = check_primary_eligibility(anchor_meta, candidate, hierarchy_evidence)
    if not eligible:
        return False, eligibility_reasons, rescued, meta

    path_match = float(hierarchy_evidence.get("path_match", 0.0) or 0.0)
    topic_overlap = float(hierarchy_evidence.get("topic_overlap", 0.0) or 0.0)
    subfield_overlap = float(hierarchy_evidence.get("subfield_overlap", 0.0) or 0.0)
    context_consistency = compute_context_consistency(candidate)
    ctx_sim = float(getattr(candidate, "context_sim", 0.0) or 0.0)
    source_role = (getattr(candidate, "source_role", "") or "").strip()

    # 主通道：优先看上下文稳定性，其次才看 identity
    if (
        semantic_score >= 0.80
        and jd_align >= 0.78
        and context_consistency >= 0.76
        and (anchor_identity >= 0.22 or ctx_sim >= 0.82)
    ):
        meta["retain_mode"] = "normal" if anchor_identity >= 0.30 else "weak_retain"
        meta["suppress_seed"] = anchor_identity < 0.30
        meta["retain_reason"] = "dual_space_supported"
        return True, reasons, rescued, meta

    # rescue：多锚 + path/jd/semantic 不差 + context 至少不反对
    if (
        cross_anchor_support_count >= PRIMARY_RESCUE_CROSS_ANCHOR_MIN
        and path_match >= PRIMARY_RESCUE_PATH_MATCH_MIN
        and jd_align >= max(0.76, PRIMARY_RESCUE_JD_ALIGN_MIN - 0.02)
        and semantic_score >= PRIMARY_RESCUE_SEMANTIC_MIN
        and context_consistency >= 0.72
    ):
        rescued = True
        meta["retain_mode"] = "weak_retain"
        meta["suppress_seed"] = True
        meta["retain_reason"] = "cross_anchor_context_rescue"
        return True, reasons, rescued, meta

    # context fallback：仍保守，但允许更低 identity 通过
    if source_role == "context_fallback":
        if ctx_sim >= 0.84 and jd_align >= 0.80 and anchor_identity >= 0.32:
            meta["retain_mode"] = "weak_retain"
            meta["suppress_seed"] = True
            meta["retain_reason"] = "context_fallback_supported"
            return True, reasons, rescued, meta

    # hierarchy 尚可 + context 边界可接受：弱保留，不直接全砍
    if topic_overlap >= PRIMARY_MIN_HIERARCHY_MATCH and subfield_overlap >= 0.25 and context_consistency >= 0.70:
        meta["retain_mode"] = "weak_retain"
        meta["suppress_seed"] = True
        meta["retain_reason"] = "hierarchy_supported_but_context_borderline"
        return True, ["context_borderline"], rescued, meta

    # 通用主词弱放行：simulation / mechanics / route planning 等合理但 hierarchy 弱的主词
    if (
        semantic_score >= 0.80
        and jd_align >= 0.78
        and context_consistency >= 0.68
        and anchor_identity >= 0.16
    ):
        meta["retain_mode"] = "weak_retain"
        meta["suppress_seed"] = True
        meta["retain_reason"] = "generic_main_term_supported"
        return True, ["generic_main_term"], rescued, meta

    reasons.append("dual_space_not_stable")
    return False, reasons, rescued, meta


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
    本义 + 双空间一致性做主干，hierarchy 只微调；不再让 path/topic/span 单独翻盘。
    """
    raw_sim = max(0.0, min(1.0, float(semantic_score or 0.0)))
    identity = max(0.0, min(1.0, float(anchor_identity or 0.0)))
    jd = max(0.0, min(1.0, float(jd_align or 0.0)))
    cross = max(0.0, min(1.0, min(int(cross_anchor_support_count or 0), 2) / 2.0))
    neigh = max(0.0, min(1.0, float(local_neighborhood_consistency or 0.0)))
    context_consistency = compute_context_consistency(candidate)

    base = (
        0.30 * identity
        + 0.26 * context_consistency
        + 0.24 * jd
        + 0.10 * raw_sim
        + 0.06 * cross
        + 0.04 * neigh
    )
    field_fit = max(0.0, min(1.0, float(hierarchy_evidence.get("field_overlap", 0.0) or 0.0)))
    subfield_fit = max(0.0, min(1.0, float(hierarchy_evidence.get("subfield_overlap", 0.0) or 0.0)))
    topic_fit = max(0.0, min(1.0, float(hierarchy_evidence.get("topic_overlap", 0.0) or 0.0)))
    path_match = max(0.0, min(1.0, float(hierarchy_evidence.get("path_match", 0.0) or 0.0)))
    topic_span_penalty = max(0.0, min(1.0, float(hierarchy_evidence.get("topic_span_penalty", 1.0) or 1.0)))
    topic_specificity = max(0.0, min(1.0, float(hierarchy_evidence.get("topic_specificity", 0.0) or 0.0)))
    hierarchy_bonus = (
        0.04 * field_fit
        + 0.05 * subfield_fit
        + 0.05 * topic_fit
        + 0.03 * path_match
        + 0.02 * topic_specificity
    )
    span_factor = 0.95 + 0.05 * topic_span_penalty
    final = (base + hierarchy_bonus) * span_factor
    if getattr(candidate, "retain_mode", "normal") == "weak_retain":
        final *= 0.92
    if getattr(candidate, "source", "") == "similar_to" and not getattr(candidate, "context_supported", False):
        final *= 0.90
    return max(0.0, min(1.0, final))


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
        print(f"  JD 维度: field={jd_fields}  subfield={jd_sub}  topic={jd_top}  |  Stage2A=组内相对选主（无固定阈值）")
        print(f"  锚点列表: {[getattr(a, 'anchor', a) for a in prepared_anchors]}")
    # ---------- Stage2A 极简重构：每锚点收集候选 → 证据 → 组内相对排序 → 选 primary → 仅 primary 再判 can_expand ----------
    per_anchor_candidates: Dict[int, List[Stage2ACandidate]] = {}
    for anchor in prepared_anchors:
        landing_list = collect_landing_candidates(
            label, anchor,
            active_domain_set=active_domains,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
            jd_profile=jd_profile,
            query_vector=query_vector,
        )
        stage2a_list = landing_candidates_to_stage2a(landing_list)
        per_anchor_candidates[anchor.vid] = stage2a_list

    cross_anchor_index = build_cross_anchor_index(per_anchor_candidates)
    all_anchor_results: List[Dict[str, Any]] = []
    primary_landings_by_anchor: Dict[int, List[PrimaryLanding]] = {}
    evidence_table: List[Dict[str, Any]] = []

    _stage2_header("Stage2A 极简：组内相对排序选主落点（无固定阈值）", "-")
    for anchor in prepared_anchors:
        candidates = per_anchor_candidates.get(anchor.vid, [])
        if not candidates:
            primary_landings_by_anchor[anchor.vid] = []
            all_anchor_results.append({"anchor": anchor, "candidates": [], "primary": []})
            continue
        enriched = enrich_stage2a_candidates(
            label, anchor, candidates, prepared_anchors, active_domains, query_text,
            cross_anchor_index,
            query_vector=query_vector,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
        )
        selected = select_primary_per_anchor(anchor, enriched)
        primary_landings_list = []
        for cand in selected:
            p = PrimaryLanding(
                vid=cand.tid,
                term=cand.term or str(cand.tid),
                identity_score=cand.family_match,
                source=cand.source,
                anchor_vid=anchor.vid,
                anchor_term=anchor.anchor,
                domain_fit=1.0,
            )
            setattr(p, "primary_score", cand.composite_rank_score)
            setattr(p, "anchor_identity_score", cand.family_match)
            setattr(p, "retain_mode", "normal" if cand.can_expand else "weak_retain")
            setattr(p, "suppress_seed", not cand.can_expand)
            setattr(p, "topic_source", "missing")
            setattr(p, "jd_align", cand.jd_align)
            setattr(p, "bucket", "primary_expandable" if cand.can_expand else "primary_keep_no_expand")
            setattr(p, "expandable", cand.can_expand)
            setattr(p, "field_fit", 0.0)
            setattr(p, "path_match", 0.0)
            setattr(p, "topic_span_penalty", 1.0)
            setattr(p, "subfield_fit", 0.0)
            setattr(p, "topic_fit", 0.0)
            setattr(p, "cross_anchor_support_count", max(1, len(cross_anchor_index.get(cand.tid, []))))
            primary_landings_list.append(p)
            _mainline_num = 0.7 if cand.role == "mainline" else (0.4 if cand.role == "side" else 0.0)
            evidence_table.append({
                "anchor": anchor.anchor,
                "anchor_vid": anchor.vid,
                "candidate": cand.term,
                "tid": cand.tid,
                "bucket": "primary_expandable" if cand.can_expand else "primary_keep_no_expand",
                "primary_score": cand.composite_rank_score,
                "mainline_alignment": _mainline_num,
                "expandable": cand.can_expand,
                "source": cand.source,
                "semantic_score": cand.semantic_score,
                "jd_align": cand.jd_align,
                "anchor_identity_score": cand.family_match,
                "identity_gate": 1.0,
                "base_primary_score": cand.composite_rank_score,
                "edge_affinity": cand.family_match,
                "conditioned_anchor_align": None,
                "multi_anchor_support": cand.cross_anchor_support,
                "hierarchy_consistency": cand.hierarchy_consistency,
                "neighborhood_consistency": 0.5,
                "isolation_penalty": cand.isolation_risk,
                "polysemy_risk": cand.polysemy_risk,
                "specificity_prior": 0.5,
            })
        primary_landings_by_anchor[anchor.vid] = primary_landings_list
        all_anchor_results.append({"anchor": anchor, "candidates": enriched, "primary": selected})

    final_primary_merged = merge_stage2a_primary(all_anchor_results)

    _stage2_header("Stage2A 每锚 primary（组内相对选主）", "-")
    if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
        rows_summary = []
        for a in prepared_anchors:
            plist = primary_landings_by_anchor.get(a.vid, [])
            n_cand = len(per_anchor_candidates.get(a.vid, []))
            rows_summary.append([
                getattr(a, "anchor", str(a))[:18],
                str(n_cand),
                str(len(plist)),
            ])
        if rows_summary:
            _stage2_table(rows_summary, ["锚点", "候选数", "primary数"], col_widths=[20, 12, 12])
        for a in prepared_anchors:
            plist = primary_landings_by_anchor.get(a.vid, [])
            if not plist:
                continue
            print(f"  --- 锚点 {getattr(a, 'anchor', a)!r} primary 明细 ---")
            _stage2_table(
                [
                    [
                        str(p.vid),
                        (p.term or "")[:16],
                        f"{getattr(p, 'primary_score', 0):.3f}",
                        getattr(p, "bucket", ""),
                        "Y" if getattr(p, "expandable", False) else "N",
                    ]
                    for p in plist[:15]
                ],
                ["tid", "term", "primary_score", "bucket", "expandable"],
                col_widths=[8, 18, 14, 22, 10],
            )
            if len(plist) > 15:
                print(f"  ... 共 {len(plist)} 条 primary")

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
    for row in evidence_table:
        key = (row["tid"], (row.get("candidate") or "").strip())
        if key not in by_term_cross:
            by_term_cross[key] = {"term": row.get("candidate", ""), "supported_by_anchors": [], "support_weight_sum": 0.0}
        by_term_cross[key]["supported_by_anchors"].append(row.get("anchor", ""))
        by_term_cross[key]["support_weight_sum"] += float(row.get("primary_score", 0) or 0)
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
                n_cand = len(per_anchor_candidates.get(anchor.vid, []))
                print(
                    f"[Stage2] 锚点 anchor={anchor.anchor!r} vid={anchor.vid}{amb_tag} 无 primary，跳过 | "
                    f"原因: 本锚点组内相对选主后无 primary（候选数={n_cand}）"
                )
            continue
        if LABEL_EXPANSION_DEBUG:
            print(f"[Stage2] 锚点 anchor={anchor.anchor!r} primary 数={len(primary_landings)}")
        # Stage2B：仅 primary_expandable 可进扩散（四分桶已决定，不再用 check_seed_eligibility 过滤）
        diffusion_primaries = [p for p in primary_landings if getattr(p, "expandable", False)]
        _seed_detail: List[Tuple[Any, bool, float]] = []
        for p in primary_landings:
            eligible = getattr(p, "expandable", False)
            setattr(p, "seed_blocked", not eligible)
            setattr(p, "seed_block_reason", None if eligible else "not_primary_expandable")
            seed_score = getattr(p, "primary_score", 0.5) or 0.5
            setattr(p, "seed_score", seed_score)
            _seed_detail.append((p, eligible, seed_score))
        # 仅 primary_expandable 参与扩散
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
        # 为 dense/cooc 的 support_expandable_for_anchor 提供锚点 context
        for p in diffusion_primaries:
            setattr(p, "anchor_conditioned_vec", getattr(anchor, "conditioned_vec", None))
            setattr(p, "_context_neighbors", getattr(anchor, "_context_neighbors", None) or [])
            setattr(p, "_context_score_map", getattr(anchor, "_context_score_map", None) or {})
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
            jd_profile=jd_profile,
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

