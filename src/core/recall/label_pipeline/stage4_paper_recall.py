import time
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.recall.label_means.label_anchors import canonical_jd_text_for_encode
from src.core.recall.label_pipeline import stage1_domain_anchors as _stage1_anchors
from src.utils.time_features import compute_paper_recency
from config import ABSTRACT_MAP_PATH, INDEX_DIR

# 层级守卫：单 term 最多贡献论文数，避免泛词占满 paper 池
TERM_MAX_PAPERS = 50
# 单 term 对某作者总贡献占比上限（Stage5 `_apply_term_max_author_share_cap`；与 paper_scoring 无关）
TERM_MAX_AUTHOR_SHARE = 0.25

# 词侧熔断：degree_w/total_w 超过此比例的泛词在 Cypher 内过滤
MELT_RATIO = 0.05
# 领域软奖励：论文 domain_ids 匹配目标领域时的乘数（小幅加成，不主导）
DOMAIN_BONUS_MATCH = 1.2
DOMAIN_BONUS_NO_MATCH = 1.0

# 全局 paper 池上限
GLOBAL_PAPER_LIMIT = 2000
# True：逐条打印 paper_map 合并；False：默认关闭（降噪，避免刷屏与大量重复 wid）
STAGE4_PAPER_MAP_WRITE_VERBOSE = False
# [Stage4 jd audit] 每条 term 打印前 K 条 pre_gate 行（默认 3；原为 10）
STAGE4_JD_AUDIT_TOP_K = 3
# 仅对「final_score 最高的前 K 个 paper_primary」打明细 jd audit（降噪）；True 则全量
STAGE4_JD_AUDIT_PRIMARY_DETAIL_K = int(os.environ.get("STAGE4_JD_AUDIT_PRIMARY_DETAIL_K", "3"))
STAGE4_JD_AUDIT_FULL = os.environ.get("STAGE4_JD_AUDIT_FULL", "").strip().lower() in ("1", "true", "yes")
# overlap survival 审计最大行数（高分存活 + rank 靠前被淘汰项；**默认 0 关闭**，问题已收窄到 term 级放大而非 overlap 有无）
STAGE4_OVERLAP_SURVIVAL_MAX_LINES = int(os.environ.get("STAGE4_OVERLAP_SURVIVAL_MAX_LINES", "0"))
# 三级领域共识软乘子：词侧 vocabulary_topic_stats × JD field/subfield/topic 画像，不对论文硬筛
STAGE4_HIERARCHY_CONSENSUS_ENABLED = os.environ.get("STAGE4_HIERARCHY_CONSENSUS_ENABLED", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
STAGE4_HIERARCHY_AUDIT_TOP_BY_SCORE = int(os.environ.get("STAGE4_HIERARCHY_AUDIT_TOP_BY_SCORE", "10"))
STAGE4_HIERARCHY_AUDIT_TOP_BY_DELTA = int(os.environ.get("STAGE4_HIERARCHY_AUDIT_TOP_BY_DELTA", "10"))
STAGE4_TOPIC_META_COVERAGE_AUDIT = os.environ.get("STAGE4_TOPIC_META_COVERAGE_AUDIT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
# 方案 B：bonus 以池内 consensus 中位数为锚，相对拉开（不是全局压到 ~0.8x）
STAGE4_HIERARCHY_BONUS_BETA = float(os.environ.get("STAGE4_HIERARCHY_BONUS_BETA", "12.0"))
STAGE4_HIERARCHY_BONUS_CLIP_LOW = float(os.environ.get("STAGE4_HIERARCHY_BONUS_CLIP_LOW", "0.82"))
STAGE4_HIERARCHY_BONUS_CLIP_HIGH = float(os.environ.get("STAGE4_HIERARCHY_BONUS_CLIP_HIGH", "1.15"))
STAGE4_HIERARCHY_DISTRIBUTION_TOP_N = int(os.environ.get("STAGE4_HIERARCHY_DISTRIBUTION_TOP_N", "10"))
# True：仅当 wid 命中里 **term_score×idf 加权** 有 ≥ 下列比例的 **Stage3·strong_main_axis_core** 时，才允许 hierarchy_bonus>1（与 bonus_core / support 解耦）
STAGE4_HIERARCHY_BONUS_POSITIVE_MAIN_AXIS_ONLY = os.environ.get(
    "STAGE4_HIERARCHY_BONUS_POSITIVE_MAIN_AXIS_ONLY", "1"
).strip().lower() not in ("0", "false", "no")
STAGE4_HIERARCHY_STRONG_AXIS_WEIGHT_FRAC = float(
    os.environ.get("STAGE4_HIERARCHY_STRONG_AXIS_WEIGHT_FRAC", "0.20")
)
STAGE4_HIERARCHY_TERM_GROUP_AUDIT = os.environ.get("STAGE4_HIERARCHY_TERM_GROUP_AUDIT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
# False：只打 [Stage4 hierarchy bonus distribution] 分位数摘要，不逐条 top_boosted/top_penalized（与 consensus audit 防重复）
STAGE4_HIERARCHY_BONUS_DISTRIBUTION_DETAIL = os.environ.get(
    "STAGE4_HIERARCHY_BONUS_DISTRIBUTION_DETAIL", ""
).strip().lower() in ("1", "true", "yes")
# True：打印 author payload 逐条 multi-hit；False：仅保留汇总计数（默认）
STAGE4_AUTHOR_PAYLOAD_AUDIT_VERBOSE = False
# 0：关闭。>0：每词沿 HAS_TOPIC 仅保留 year 降序前 N 篇（与 Python 侧 term_contrib 排序不同，可能改变结果；用于压行数时对照）
STAGE4_LAYER1_PER_V_CAP = int(os.environ.get("STAGE4_LAYER1_PER_V_CAP", "0"))


def _batch_jd_align_for_wids(
    wids: Set[str],
    paper_vecs: Any,
    paper_id_to_row: Dict[str, int],
    paper_norms: Optional[np.ndarray],
    query_vec_1d: Optional[np.ndarray],
    query_norm: float,
) -> Dict[str, float]:
    """
    wid -> jd_align，与逐行 np.dot(pv,q)/(||pv||*||q||) 等价；对行索引去重后向量化。
    """
    out: Dict[str, float] = {}
    if query_vec_1d is None or query_norm <= 1e-12:
        return {w: 0.5 for w in wids}
    if not isinstance(paper_vecs, np.ndarray) or paper_vecs.size == 0:
        return {w: 0.5 for w in wids}
    q = np.asarray(query_vec_1d, dtype=np.float32).flatten()
    idx_to_wids: Dict[int, List[str]] = defaultdict(list)
    for w in wids:
        ri = paper_id_to_row.get(str(w))
        if ri is None:
            out[w] = 0.5
            continue
        try:
            ri_i = int(ri)
        except (TypeError, ValueError):
            out[w] = 0.5
            continue
        if ri_i < 0 or ri_i >= len(paper_vecs):
            out[w] = 0.5
            continue
        idx_to_wids[ri_i].append(w)
    if not idx_to_wids:
        return out
    n_rows = int(paper_vecs.shape[0])
    U = np.array(sorted(idx_to_wids.keys()), dtype=np.int64)
    U = U[(U >= 0) & (U < n_rows)]
    if U.size == 0:
        for w in wids:
            if w not in out:
                out[w] = 0.5
        return out
    P = np.asarray(paper_vecs[U], dtype=np.float32)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    if paper_norms is not None and len(paper_norms) >= n_rows:
        pn = np.asarray(paper_norms[U], dtype=np.float64)
    else:
        pn = np.linalg.norm(P, axis=1)
    pn = np.where(pn <= 1e-12, 1e-12, pn)
    dots = P @ q
    cos = dots / (float(query_norm) * pn)
    jda = np.clip(0.5 * (cos + 1.0), 0.0, 1.0)
    for i, ri in enumerate(U):
        ri_i = int(ri)
        val = float(jda[i])
        for ww in idx_to_wids[ri_i]:
            out[ww] = val
    for w in wids:
        if w not in out:
            out[w] = 0.5
    return out


def _normalize_topic_dist(raw: Any) -> Dict[str, float]:
    """
    统一解析 vocabulary_topic_stats / jd_profile 中的分布：JSON 串或 dict，key→str，值归一化后和为 1。
    """
    if not raw:
        return {}
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            raw = json.loads(s)
        except Exception:
            return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in raw.items():
        ks = str(k).strip()
        if not ks:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0:
            out[ks] = fv
    tot = sum(out.values())
    if tot > 0:
        out = {k: v / tot for k, v in out.items()}
    return out


def _build_term_topic_meta_from_row(row: Any) -> Dict[str, Any]:
    """
    批注：批量 SQL 列序为 voc_id, field_id, subfield_id, topic_id, field_dist, subfield_dist, topic_dist, source。
    优先用 *_ dist（含 cooc 补全）；无 dist 再退主值 one-hot，避免「只有主 id 的词」在 overlap 里永远是 0。
    """
    if row is None:
        return {}
    try:
        field_id = row[1]
        subfield_id = row[2]
        topic_id = row[3]
        field_dist = _normalize_topic_dist(row[4])
        subfield_dist = _normalize_topic_dist(row[5])
        topic_dist = _normalize_topic_dist(row[6])
        source = row[7] if len(row) > 7 else None
    except (TypeError, IndexError):
        return {}
    if not field_dist and field_id is not None and str(field_id).strip():
        field_dist = {str(field_id).strip(): 1.0}
    if not subfield_dist and subfield_id is not None and str(subfield_id).strip():
        subfield_dist = {str(subfield_id).strip(): 1.0}
    if not topic_dist and topic_id is not None and str(topic_id).strip():
        topic_dist = {str(topic_id).strip(): 1.0}
    return {
        "field_dist": field_dist,
        "subfield_dist": subfield_dist,
        "topic_dist": topic_dist,
        "field_id": str(field_id).strip() if field_id is not None and str(field_id).strip() else None,
        "subfield_id": str(subfield_id).strip() if subfield_id is not None and str(subfield_id).strip() else None,
        "topic_id": str(topic_id).strip() if topic_id is not None and str(topic_id).strip() else None,
        "source": str(source).strip() if source else None,
    }


def _dist_overlap(a: Dict[str, float], b: Dict[str, float]) -> float:
    """分布点积 overlap（a、b 已归一化时等价于同一底层事件空间下的 Σ p·q）。"""
    if not a or not b:
        return 0.0
    return sum(float(a.get(k, 0.0)) * float(b.get(k, 0.0)) for k in a)


def _accumulate_weighted_dist(acc: Dict[str, float], dist: Dict[str, float], w: float) -> None:
    if w <= 0 or not dist:
        return
    for k, p in dist.items():
        acc[str(k)] = acc.get(str(k), 0.0) + w * float(p)


def compute_hierarchy_consensus_for_paper(
    paper_hits: List[Dict[str, Any]],
    term_topic_meta: Dict[int, Dict[str, Any]],
    jd_topic_profile: Dict[str, Dict[str, float]],
) -> Tuple[float, Dict[str, Any]]:
    """
    同一 wid：各命中词的三层 dist 按 term_score×idf 加权混合，再与 JD 同层归一化分布做点积，得 hierarchy_consensus。
    无层级信号时 consensus=0（后续由池内中位数相对映射 bonus，而不是整池 ×0.8）。
    """
    field_acc: Dict[str, float] = {}
    subfield_acc: Dict[str, float] = {}
    topic_acc: Dict[str, float] = {}
    total_w = 0.0

    for hit in paper_hits or []:
        if not isinstance(hit, dict):
            continue
        try:
            tid = int(hit.get("vid"))
        except (TypeError, ValueError):
            continue
        ts = float(hit.get("term_score") or 0.0)
        idf = float(hit.get("idf") or 0.0)
        w = max(0.0, ts * idf)
        if w <= 0:
            continue
        meta = term_topic_meta.get(tid) or {}
        fd = meta.get("field_dist") or {}
        sd = meta.get("subfield_dist") or {}
        td = meta.get("topic_dist") or {}
        if not fd and not sd and not td:
            continue
        total_w += w
        _accumulate_weighted_dist(field_acc, fd, w)
        _accumulate_weighted_dist(subfield_acc, sd, w)
        _accumulate_weighted_dist(topic_acc, td, w)

    if total_w <= 0:
        return 0.0, {
            "reason": "no_hits_with_topic_signal",
            "field_cons": 0.0,
            "subfield_cons": 0.0,
            "topic_cons": 0.0,
            "hierarchy_consensus": 0.0,
        }

    def _norm_mix(d: Dict[str, float]) -> Dict[str, float]:
        s = sum(d.values())
        return {k: v / s for k, v in d.items()} if s > 0 else {}

    mix_f = _norm_mix(field_acc)
    mix_s = _norm_mix(subfield_acc)
    mix_t = _norm_mix(topic_acc)

    jd_f = jd_topic_profile.get("field_dist") or {}
    jd_s = jd_topic_profile.get("subfield_dist") or {}
    jd_t = jd_topic_profile.get("topic_dist") or {}
    if not jd_f and not jd_s and not jd_t:
        return 0.0, {
            "reason": "empty_jd_topic_profile",
            "field_cons": 0.0,
            "subfield_cons": 0.0,
            "topic_cons": 0.0,
            "hierarchy_consensus": 0.0,
        }

    field_cons = _dist_overlap(mix_f, jd_f) if mix_f and jd_f else 0.0
    subfield_cons = _dist_overlap(mix_s, jd_s) if mix_s and jd_s else 0.0
    topic_cons = _dist_overlap(mix_t, jd_t) if mix_t and jd_t else 0.0
    hierarchy_consensus = 0.20 * field_cons + 0.35 * subfield_cons + 0.45 * topic_cons
    detail = {
        "field_cons": round(field_cons, 4),
        "subfield_cons": round(subfield_cons, 4),
        "topic_cons": round(topic_cons, 4),
        "hierarchy_consensus": round(hierarchy_consensus, 6),
    }
    return float(hierarchy_consensus), detail


def _hierarchy_bonus_from_delta(delta: float, beta: float, lo: float, hi: float) -> float:
    """方案 B：以 1.0 为中心，相对池内中位数拉开。"""
    return float(np.clip(1.0 + beta * delta, lo, hi))


def _hits_carry_paper_lane_tier(
    hits: List[Dict[str, Any]],
    get_term_meta: Any,
) -> bool:
    """任一条 hit 的 term_meta 含非空 paper_select_lane_tier 则视为 Stage3 新路径（可启用主轴门控）。"""
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        try:
            tid = int(hit.get("vid"))
        except (TypeError, ValueError):
            continue
        meta = get_term_meta(tid) or {}
        if str(meta.get("paper_select_lane_tier") or "").strip():
            return True
    return False


def _hierarchy_strong_axis_hit_weights(
    hits: List[Dict[str, Any]],
    get_term_meta: Any,
) -> Tuple[float, float, float]:
    """
    返回 (strong_main_axis_core 的 Σ term_score×idf , 全 hit 总权, strong 占比)。
    与 compute_hierarchy_consensus_for_paper 里权重定义一致。
    """
    total_w = 0.0
    strong_w = 0.0
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        try:
            tid = int(hit.get("vid"))
        except (TypeError, ValueError):
            continue
        ts = float(hit.get("term_score") or 0.0)
        idf = float(hit.get("idf") or 0.0)
        w = max(0.0, ts * idf)
        total_w += w
        meta = get_term_meta(tid) or {}
        if str(meta.get("paper_select_lane_tier") or "").strip() == "strong_main_axis_core":
            strong_w += w
    frac = (strong_w / total_w) if total_w > 1e-12 else 0.0
    return strong_w, total_w, float(frac)


def _clip_hierarchy_bonus_by_main_axis_mass(
    raw_bonus: float,
    hits: List[Dict[str, Any]],
    get_term_meta: Any,
) -> Tuple[float, str, Dict[str, Any]]:
    """
    方案 B（Stage4）：**bonus_core / support** 等不给 **>1** 的正向 hierarchy 放大；仅当本 wid 的 consensus 证据里
    **强主轴 core** 的加权占比 ≥ STAGE4_HIERARCHY_STRONG_AXIS_WEIGHT_FRAC 时保留原 clip 上沿。
    无 paper_select_lane_tier 的旧 term_meta → 不截断（保持旧行为）。
    """
    if not STAGE4_HIERARCHY_BONUS_POSITIVE_MAIN_AXIS_ONLY:
        return float(raw_bonus), "feature_off", {}
    if not _hits_carry_paper_lane_tier(hits, get_term_meta):
        return float(raw_bonus), "legacy_no_lane_tier_meta", {}
    sw, tw, frac = _hierarchy_strong_axis_hit_weights(hits, get_term_meta)
    extra = {
        "hierarchy_strong_axis_w": round(sw, 6),
        "hierarchy_hit_total_w": round(tw, 6),
        "hierarchy_strong_axis_mass_frac": round(frac, 6),
        "hierarchy_strong_axis_frac_gate": float(STAGE4_HIERARCHY_STRONG_AXIS_WEIGHT_FRAC),
    }
    if tw <= 1e-12:
        return float(raw_bonus), "no_weighted_hits", extra
    if frac >= float(STAGE4_HIERARCHY_STRONG_AXIS_WEIGHT_FRAC):
        return float(raw_bonus), "allow_positive_boost", extra
    capped = min(float(raw_bonus), 1.0)
    return capped, ("capped_non_strong_axis_mass" if capped < raw_bonus else "no_positive_raw"), extra


def _print_stage4_hierarchy_bonus_term_group_audit(
    by_wid: Dict[str, Dict[str, Any]],
    get_term_meta: Any,
    audit_print: bool = True,
) -> None:
    """
    [Stage4 hierarchy bonus by term-group audit]：按 **vid** 汇总进入 wid 池的论文上的 mean consensus / mean bonus，
    对照 Stage3 **paper_select_lane_tier** 与父锚强度，解释为何某 term 组级 bonus 偏高或偏低。
    """
    if not audit_print or not STAGE4_HIERARCHY_TERM_GROUP_AUDIT or not by_wid:
        return
    agg: Dict[int, Dict[str, float]] = defaultdict(
        lambda: {"papers": 0.0, "sum_c": 0.0, "sum_b": 0.0, "sum_b_raw": 0.0}
    )
    for _rec in by_wid.values():
        _det = _rec.get("hierarchy_consensus_detail") or {}
        _cons = float(_det.get("hierarchy_consensus") or 0.0)
        _b = float(_rec.get("hierarchy_consensus_bonus") or 1.0)
        _b_raw = float(_det.get("hierarchy_bonus_raw", _b))
        _seen: Set[int] = set()
        for _h in _rec.get("hits") or []:
            if not isinstance(_h, dict):
                continue
            try:
                _vid = int(_h.get("vid"))
            except (TypeError, ValueError):
                continue
            if _vid in _seen:
                continue
            _seen.add(_vid)
            a = agg[_vid]
            a["papers"] += 1.0
            a["sum_c"] += _cons
            a["sum_b"] += _b
            a["sum_b_raw"] += _b_raw
    if not agg:
        return
    print("\n" + "-" * 80)
    print(
        "[Stage4 hierarchy bonus by term-group audit] "
        "term | tier | papers_in_pool | mean_cons | mean_bonus | mean_bonus_raw | "
        "parent_anchor | pa_sc | rk | is_main_axis_core | tier_boost_note"
    )
    print("-" * 80)
    for _vid in sorted(agg.keys(), key=lambda v: -agg[v]["sum_b"]):
        meta = get_term_meta(_vid) or {}
        term = str(meta.get("term") or _vid)[:28]
        tier = str(meta.get("paper_select_lane_tier") or "-")[:22]
        n = max(1.0, agg[_vid]["papers"])
        mean_c = agg[_vid]["sum_c"] / n
        mean_b = agg[_vid]["sum_b"] / n
        mean_br = agg[_vid]["sum_b_raw"] / n
        panch = str(meta.get("parent_anchor") or "")[:18]
        pas = float(meta.get("parent_anchor_final_score") or 0.0)
        prk = meta.get("parent_anchor_step2_rank")
        rk_i = int(prk) if prk is not None else 999
        is_mac = str(meta.get("paper_select_lane_tier") or "") == "strong_main_axis_core"
        note = "mean_raw>mean ⇒ wid级正向被主轴门控截断" if mean_br > mean_b + 1e-4 else "-"
        print(
            f"{term:28} | {tier:22} | {int(n):^14} | {mean_c:9.4f} | {mean_b:10.4f} | {mean_br:14.4f} | "
            f"{panch:18} | {pas:5.2f} | {rk_i:2d} | {str(is_mac):^17} | {note}"
        )


def _audit_hierarchy_bonus_distribution(
    dist_rows: List[Dict[str, Any]], top_n: int, audit_print: bool = True
) -> None:
    if not audit_print or not dist_rows:
        return
    cons = np.array([float(r["consensus"]) for r in dist_rows], dtype=np.float64)
    bon = np.array([float(r["bonus"]) for r in dist_rows], dtype=np.float64)

    def _pct(x: np.ndarray, q: float) -> float:
        return float(np.percentile(x, q))

    print("\n[Stage4 hierarchy bonus distribution]")
    print(f"candidate_papers={len(dist_rows)} beta={STAGE4_HIERARCHY_BONUS_BETA} clip=[{STAGE4_HIERARCHY_BONUS_CLIP_LOW},{STAGE4_HIERARCHY_BONUS_CLIP_HIGH}]")
    print(
        f"hierarchy_consensus min={float(np.min(cons)):.4f} p25={_pct(cons, 25):.4f} "
        f"p50={_pct(cons, 50):.4f} p75={_pct(cons, 75):.4f} max={float(np.max(cons)):.4f}"
    )
    print(
        f"hierarchy_bonus min={float(np.min(bon)):.4f} p25={_pct(bon, 25):.4f} "
        f"p50={_pct(bon, 50):.4f} p75={_pct(bon, 75):.4f} max={float(np.max(bon)):.4f}"
    )
    if not STAGE4_HIERARCHY_BONUS_DISTRIBUTION_DETAIL:
        return
    _tn = max(0, top_n)
    boosted = sorted(dist_rows, key=lambda r: -float(r["bonus"]))[:_tn]
    penalized = sorted(dist_rows, key=lambda r: float(r["bonus"]))[:_tn]
    print(f"top_{_tn}_boosted (by hierarchy_bonus):")
    for i, r in enumerate(boosted, 1):
        print(
            f"  #{i} pid={r['wid']!r} bonus={float(r['bonus']):.4f} consensus={float(r['consensus']):.4f} "
            f"delta_vs_median={float(r.get('delta_vs_median', 0.0)):.4f} "
            f"paper_score_before={float(r.get('paper_score_before', 0.0)):.4f} title={str(r.get('title') or '')[:75]!r}"
        )
    print(f"top_{_tn}_penalized (by hierarchy_bonus):")
    for i, r in enumerate(penalized, 1):
        print(
            f"  #{i} pid={r['wid']!r} bonus={float(r['bonus']):.4f} consensus={float(r['consensus']):.4f} "
            f"delta_vs_median={float(r.get('delta_vs_median', 0.0)):.4f} "
            f"paper_score_before={float(r.get('paper_score_before', 0.0)):.4f} title={str(r.get('title') or '')[:75]!r}"
        )


def _jd_topic_profile_from_stage1(jd_src: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Stage1 的 weights 已是 dict，此处再做 key/str 与归一化，与词侧 dist 同空间可比。"""
    return {
        "field_dist": _normalize_topic_dist(jd_src.get("field_weights")),
        "subfield_dist": _normalize_topic_dist(jd_src.get("subfield_weights")),
        "topic_dist": _normalize_topic_dist(jd_src.get("topic_weights")),
    }


def _audit_jd_topic_profile(jd_topic_profile: Dict[str, Dict[str, float]], audit_print: bool = True) -> None:
    if not audit_print:
        return
    jd_f = jd_topic_profile.get("field_dist") or {}
    jd_s = jd_topic_profile.get("subfield_dist") or {}
    jd_t = jd_topic_profile.get("topic_dist") or {}

    def _top_entries(d: Dict[str, float], n: int = 5) -> List[Tuple[str, float]]:
        return sorted(d.items(), key=lambda x: -x[1])[:n]

    print("\n[Stage4 jd-topic-profile audit]")
    print(
        f"field_keys_count={len(jd_f)} subfield_keys_count={len(jd_s)} topic_keys_count={len(jd_t)} "
        f"top_field_keys={_top_entries(jd_f)} top_subfield_keys={_top_entries(jd_s)} top_topic_keys={_top_entries(jd_t)}"
    )


def get_term_role_weight(term_retrieval_roles: Optional[Dict[int, str]], vid: int) -> float:
    """按 retrieval_role 给权重：paper_primary=1.0，paper_support=0.7，blocked/其他=0.4。不看领域词。"""
    if not term_retrieval_roles:
        return 1.0
    role = (term_retrieval_roles.get(vid) or term_retrieval_roles.get(str(vid)) or "").strip().lower()
    if role == "paper_primary":
        return 1.0
    if role == "paper_support":
        return 0.7
    return 0.4


# --- Step2：Stage4 paper-level explanation schema（仅观测字段；不改排序/门控）---
STAGE4_PAPER_EXPLAIN_VERSION = "v1"

_STAGE4_PAPER_EXPLANATION_KEYS = (
    "paper_explain_version",
    "hit_terms",
    "hit_count",
    "hit_term_roles",
    "hit_parent_anchors",
    "hit_parent_primaries",
    "mainline_term_count",
    "side_term_count",
    "same_family_or_cross_family",
    "grounding_score",
    "jd_align",
    "hierarchy_consensus",
    "hierarchy_bonus",
    "domain_bonus",
    "recency_score",
    "paper_score_base",
    "paper_score_final",
    "paper_reason_summary",
    "hit_quality_class",
    "coherence_reason",
    "multi_hit_strength",
    "paper_global_pool_kept",
)


def _stage4_family_key_for_hit(hit: Dict[str, Any], get_term_meta: Any) -> str:
    fk = str(hit.get("family_key") or "").strip()
    if fk:
        return fk
    try:
        tid = int(hit.get("vid"))
    except (TypeError, ValueError):
        return ""
    meta = get_term_meta(tid) or {}
    return str(meta.get("family_key") or "").strip()


def _stage4_hit_mainline_like(hit: Dict[str, Any], get_term_meta: Any) -> bool:
    """保守：优先 lane_type（Step1）与 retrieval_role；不引入词面规则。"""
    lt = str(hit.get("lane_type") or "").strip().lower()
    if lt == "risky_bridge_coverage":
        return False
    if lt == "direct_primary":
        return True
    role = str(hit.get("role") or "").strip().lower()
    if lt == "support_coverage" and role == "paper_support":
        return False
    if role == "paper_primary":
        return True
    if role == "paper_support":
        return False
    try:
        tid = int(hit.get("vid"))
    except (TypeError, ValueError):
        return False
    meta = get_term_meta(tid) or {}
    r2 = str(meta.get("retrieval_role") or "").strip().lower()
    return r2 == "paper_primary"


def _stage4_same_family_or_cross_family(
    hits: List[Dict[str, Any]], get_term_meta: Any
) -> str:
    if len(hits) < 2:
        return "single_hit"
    keys = [_stage4_family_key_for_hit(h, get_term_meta) for h in hits if isinstance(h, dict)]
    non_empty = [k for k in keys if k]
    if not non_empty:
        return "mixed_or_unknown"
    u = set(non_empty)
    if len(u) <= 1:
        return "same_family"
    return "cross_family"


def _classify_stage4_paper_hit_quality(
    hits: List[Dict[str, Any]],
    get_term_meta: Any,
) -> Dict[str, Any]:
    """
    multi-hit 质量 + 主/侧计数（只读 hit 与 term_meta；不改变分数）。
    返回 materialize 所需子字段。
    """
    clean = [h for h in (hits or []) if isinstance(h, dict)]
    n = len(clean)
    ml = sum(1 for h in clean if _stage4_hit_mainline_like(h, get_term_meta))
    sl = n - ml
    sf = _stage4_same_family_or_cross_family(clean, get_term_meta)

    if n <= 0:
        return {
            "mainline_term_count": 0,
            "side_term_count": 0,
            "same_family_or_cross_family": "single_hit",
            "hit_quality_class": "side_only_or_accidental_multi_hit",
            "coherence_reason": "no hits",
            "multi_hit_strength": 0,
        }

    if n == 1:
        if ml == 1:
            hq = "single_hit_mainline"
            cr = "single mainline hit"
        else:
            hq = "single_hit_side"
            cr = "single support/side hit"
        mhs = 0
        return {
            "mainline_term_count": ml,
            "side_term_count": sl,
            "same_family_or_cross_family": sf,
            "hit_quality_class": hq,
            "coherence_reason": cr,
            "multi_hit_strength": mhs,
        }

    if ml >= 2:
        hq = "mainline_resonance"
        cr = "two or more mainline-like hits"
        mhs = 2
    elif ml >= 1 and sl >= 1:
        hq = "mainline_plus_support"
        cr = "mainline + support complement"
        mhs = 2
    else:
        hq = "side_only_or_accidental_multi_hit"
        cr = "two hits but both side/support-like"
        mhs = 1

    return {
        "mainline_term_count": ml,
        "side_term_count": sl,
        "same_family_or_cross_family": sf,
        "hit_quality_class": hq,
        "coherence_reason": cr,
        "multi_hit_strength": mhs,
    }


def _stage4_paper_reason_summary(
    cls: Dict[str, Any],
    grounding_max: float,
    hit_count: int,
) -> str:
    """短英文模板，便于日志与下游稳定展示。"""
    hq = str(cls.get("hit_quality_class") or "other")
    g = float(grounding_max or 0.0)
    strong_g = g >= 0.35
    if hit_count <= 1:
        if hq == "single_hit_mainline":
            return "single-hit mainline with strong grounding" if strong_g else "single-hit mainline paper"
        if hq == "single_hit_side":
            return "side-only single-hit paper"
        return "single-hit paper"
    if hq == "mainline_resonance":
        return "multi-hit mainline resonance paper"
    if hq == "mainline_plus_support":
        return "multi-hit mainline+support paper"
    if hq == "side_only_or_accidental_multi_hit":
        return "multi-hit but side-heavy evidence"
    return "multi-hit paper"


def _materialize_stage4_paper_explanation_fields(
    wid: str,
    rec: Dict[str, Any],
    *,
    get_term_meta: Any,
    global_pool_kept: bool,
) -> None:
    hits = [h for h in (rec.get("hits") or []) if isinstance(h, dict)]
    cls = _classify_stage4_paper_hit_quality(hits, get_term_meta)

    hit_terms = [str(h.get("term") or h.get("vid") or "") for h in hits]
    roles = [str(h.get("role") or "") for h in hits]
    pas = [str(h.get("parent_anchor") or "") for h in hits]
    pps = [str(h.get("parent_primary") or "") for h in hits]

    gvals = [float(h.get("grounding") or 0.0) for h in hits]
    jvals = [float(h.get("jd_align") or 0.0) for h in hits]
    grounding_max = max(gvals) if gvals else 0.0
    jd_mean = float(sum(jvals) / len(jvals)) if jvals else None

    det = rec.get("hierarchy_consensus_detail") or {}
    h_cons = det.get("hierarchy_consensus")
    try:
        h_cons_f = float(h_cons) if h_cons is not None else None
    except (TypeError, ValueError):
        h_cons_f = None

    h_bonus = rec.get("hierarchy_consensus_bonus")
    try:
        h_bonus_f = float(h_bonus) if h_bonus is not None else None
    except (TypeError, ValueError):
        h_bonus_f = None

    y = rec.get("year")
    recency = None
    if y is not None:
        try:
            recency = float(compute_paper_recency(y, None))
        except Exception:
            recency = None

    ps_b = rec.get("paper_score_base")
    try:
        ps_base = float(ps_b) if ps_b is not None else None
    except (TypeError, ValueError):
        ps_base = None
    ps_f = rec.get("paper_score")
    try:
        ps_final = float(ps_f) if ps_f is not None else 0.0
    except (TypeError, ValueError):
        ps_final = 0.0

    summ = _stage4_paper_reason_summary(cls, grounding_max, len(hits))

    rec["paper_explain_version"] = STAGE4_PAPER_EXPLAIN_VERSION
    rec["hit_terms"] = hit_terms
    rec["hit_count"] = int(len(hits))
    rec["hit_term_roles"] = roles
    rec["hit_parent_anchors"] = pas
    rec["hit_parent_primaries"] = pps
    rec["mainline_term_count"] = int(cls["mainline_term_count"])
    rec["side_term_count"] = int(cls["side_term_count"])
    rec["same_family_or_cross_family"] = str(cls["same_family_or_cross_family"])
    rec["grounding_score"] = float(grounding_max)
    rec["jd_align"] = jd_mean
    rec["hierarchy_consensus"] = h_cons_f
    rec["hierarchy_bonus"] = h_bonus_f
    rec["domain_bonus"] = None
    rec["recency_score"] = recency
    rec["paper_score_base"] = ps_base
    rec["paper_score_final"] = float(ps_final)
    rec["paper_reason_summary"] = summ
    rec["hit_quality_class"] = str(cls["hit_quality_class"])
    rec["coherence_reason"] = str(cls["coherence_reason"])
    rec["multi_hit_strength"] = int(cls["multi_hit_strength"])
    rec["paper_global_pool_kept"] = bool(global_pool_kept)


def _stage4_paper_compact_explanation_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "wid": rec.get("wid"),
        "title": str(rec.get("title") or "")[:100],
        "hit_terms": rec.get("hit_terms"),
        "hit_quality_class": rec.get("hit_quality_class"),
        "grounding_score": rec.get("grounding_score"),
        "jd_align": rec.get("jd_align"),
        "hierarchy_bonus": rec.get("hierarchy_bonus"),
        "paper_score_final": rec.get("paper_score_final"),
        "paper_reason_summary": rec.get("paper_reason_summary"),
    }


def _print_stage4_paper_explanation_summary(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    audit_print: bool,
) -> None:
    if not audit_print or not by_wid:
        return
    kept = [w for w in by_wid.keys() if w in selected_wids_set]
    kept_papers_count = len(kept)
    multi_hit_kept = [
        w
        for w in kept
        if int((by_wid[w] or {}).get("hit_count") or 0) >= 2
    ]
    multi_hit_papers_count = len(multi_hit_kept)

    kept_sorted = sorted(
        kept,
        key=lambda x: -float((by_wid.get(x) or {}).get("paper_score_final") or 0.0),
    )
    ex_kept = [_stage4_paper_compact_explanation_row(by_wid[w]) for w in kept_sorted[:8]]

    mh_sorted = sorted(
        multi_hit_kept,
        key=lambda x: -float((by_wid.get(x) or {}).get("paper_score_final") or 0.0),
    )
    ex_mh = [_stage4_paper_compact_explanation_row(by_wid[w]) for w in mh_sorted[:5]]

    rejected = [w for w in by_wid.keys() if w not in selected_wids_set]
    rej_sorted = sorted(
        rejected,
        key=lambda x: -float((by_wid.get(x) or {}).get("paper_score_final") or 0.0),
    )
    ex_rej = [_stage4_paper_compact_explanation_row(by_wid[w]) for w in rej_sorted[:8]]

    ctr: Counter = Counter()
    for w in kept:
        r = by_wid.get(w) or {}
        ctr[str(r.get("hit_quality_class") or "unknown")] += 1

    print("\n" + "-" * 80)
    print("[Stage4 paper explanation summary]")
    print("-" * 80)
    print(
        f"kept_papers_count={kept_papers_count} multi_hit_papers_count={multi_hit_papers_count} "
        f"hit_quality_counter={dict(ctr)}"
    )
    print("--- top kept paper examples (compact, max 8) ---")
    for row in ex_kept:
        print(f"  {row}")
    print("--- top multi-hit kept examples (compact, max 5) ---")
    for row in ex_mh:
        print(f"  {row}")
    print("--- top rejected pool examples (compact, max 8, by paper_score_final in wid pool) ---")
    for row in ex_rej:
        print(f"  {row}")
    print("-" * 80 + "\n")


# --- Step3：author payload evidence summary（供 Stage5 读；不改作者打分）---
STAGE4_AUTHOR_PAYLOAD_EXPLAIN_VERSION = "v1"

_AUTHOR_MAINLINE_PAPER_CLASSES = frozenset(
    {"mainline_resonance", "mainline_plus_support", "single_hit_mainline"}
)
_AUTHOR_SIDE_PAPER_CLASSES = frozenset(
    {"single_hit_side", "side_only_or_accidental_multi_hit"}
)
_AUTHOR_HIGH_QUALITY_MULTI_CLASSES = frozenset({"mainline_resonance", "mainline_plus_support"})


def _author_paper_score(p: Dict[str, Any]) -> float:
    v = p.get("paper_score_final")
    if v is None:
        v = p.get("score")
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _author_paper_hit_quality_class(p: Dict[str, Any]) -> str:
    return str(p.get("hit_quality_class") or "").strip()


def _author_paper_hit_count(p: Dict[str, Any]) -> int:
    hc = p.get("hit_count")
    if hc is not None:
        try:
            return int(hc)
        except (TypeError, ValueError):
            pass
    return len(p.get("hits") or [])


def _author_collect_hit_terms(p: Dict[str, Any]) -> List[str]:
    ht = p.get("hit_terms")
    if isinstance(ht, list) and ht:
        return [str(t).strip() for t in ht if str(t).strip()]
    out: List[str] = []
    for h in p.get("hits") or []:
        if not isinstance(h, dict):
            continue
        t = str(h.get("term") or h.get("vid") or "").strip()
        if t:
            out.append(t)
    return out


def _author_dominant_term_from_papers(papers: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[float]]:
    term_mass: Dict[str, float] = defaultdict(float)
    for p in papers:
        sc = _author_paper_score(p)
        for t in set(_author_collect_hit_terms(p)):
            term_mass[t] += sc
    if not term_mass:
        return None, None
    dom = max(term_mass.keys(), key=lambda k: term_mass[k])
    total = sum(_author_paper_score(p) for p in papers)
    share = float(term_mass[dom] / total) if total > 1e-12 else None
    return dom, share


def _summarize_author_payload_evidence(
    author_id: str,
    papers: List[Dict[str, Any]],
    author_rec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """只读 papers[] 上 Step2 已透传字段；不改列表与分数。若 author_rec 上已有 dominant_term/dom_share_before 则复用。"""
    plist = [p for p in (papers or []) if isinstance(p, dict)]
    pc = len(plist)
    scores = sorted((_author_paper_score(p) for p in plist), reverse=True)
    top_score = float(scores[0]) if scores else 0.0
    top3_mass = float(sum(scores[:3]))

    mainline_n = 0
    side_n = 0
    multi_n = 0
    hq_multi_n = 0
    side_multi_n = 0
    all_single_side = bool(plist) and all(
        _author_paper_hit_quality_class(p) == "single_hit_side" for p in plist
    )

    for p in plist:
        hqc = _author_paper_hit_quality_class(p)
        if hqc in _AUTHOR_MAINLINE_PAPER_CLASSES:
            mainline_n += 1
        if hqc in _AUTHOR_SIDE_PAPER_CLASSES:
            side_n += 1
        hcnt = _author_paper_hit_count(p)
        if hcnt >= 2:
            multi_n += 1
            if hqc in _AUTHOR_HIGH_QUALITY_MULTI_CLASSES:
                hq_multi_n += 1
            if hqc == "side_only_or_accidental_multi_hit":
                side_multi_n += 1

    dom_t, dom_share = _author_dominant_term_from_papers(plist)
    if author_rec and isinstance(author_rec, dict):
        ex_dom = author_rec.get("dominant_term")
        ex_sh = author_rec.get("dom_share_before")
        if ex_dom and ex_sh is not None:
            try:
                dom_t = str(ex_dom).strip() or dom_t
                dom_share = float(ex_sh)
            except (TypeError, ValueError):
                pass
    top_paper = max(plist, key=_author_paper_score) if plist else None
    top_hqc = _author_paper_hit_quality_class(top_paper) if top_paper else None
    if not top_hqc:
        top_hqc = None

    denom = max(pc, 1)
    return {
        "author_id": str(author_id),
        "author_explain_version": STAGE4_AUTHOR_PAYLOAD_EXPLAIN_VERSION,
        "paper_count": int(pc),
        "top_paper_score": float(top_score),
        "top3_paper_mass": float(top3_mass),
        "mainline_paper_count": int(mainline_n),
        "side_only_paper_count": int(side_n),
        "multi_hit_paper_count": int(multi_n),
        "high_quality_multi_hit_count": int(hq_multi_n),
        "side_only_multi_hit_count": int(side_multi_n),
        "mainline_support_ratio": float(mainline_n / denom),
        "side_support_ratio": float(side_n / denom),
        "dominant_term": dom_t,
        "dominant_term_share": dom_share,
        "top_paper_hit_quality_class": top_hqc,
        "_all_single_hit_side": bool(all_single_side),
    }


def _classify_author_support_profile(summary: Dict[str, Any]) -> None:
    """就地写入 author_support_class / author_reason_summary / risk / notes。"""
    pc = int(summary.get("paper_count") or 0)
    ml = int(summary.get("mainline_paper_count") or 0)
    so = int(summary.get("side_only_paper_count") or 0)
    mhr = float(summary.get("mainline_support_ratio") or 0.0)
    dom = str(summary.get("dominant_term") or "").strip()
    dom_share = summary.get("dominant_term_share")
    try:
        dom_sf = float(dom_share) if dom_share is not None else None
    except (TypeError, ValueError):
        dom_sf = None
    all_single_side = bool(summary.pop("_all_single_hit_side", False))

    notes: List[str] = []
    risk = False

    if pc <= 0:
        summary["author_support_class"] = "other"
        summary["author_reason_summary"] = "no papers in payload"
        summary["author_evidence_risk_flag"] = False
        summary["author_evidence_notes"] = []
        return

    if ml == 0:
        notes.append("no_mainline_paper")

    if pc == 1:
        sup = "single_paper_supported"
        if ml >= 1:
            reason = "single-paper mainline author"
        elif so >= 1:
            reason = "single-paper side-driven author"
            risk = True
        else:
            reason = "single-paper author"
            risk = True
        notes.append("one_paper_driven")
    elif ml > so and mhr >= 0.5:
        sup = "mainline_supported"
        reason = "mainline-supported author"
    elif ml > 0 and so > ml:
        sup = "mixed_but_side_heavy"
        reason = "mixed evidence but side-heavy author"
        risk = True
    elif so > ml:
        sup = "side_only_supported"
        if dom:
            reason = f"side-heavy author driven by {dom} papers"
        else:
            reason = "side-heavy author"
        risk = True
    elif ml > 0:
        sup = "mainline_supported"
        reason = "mainline-supported author"
    else:
        sup = "other"
        reason = "other evidence mix"
        risk = True

    if all_single_side:
        notes.append("only_single_hit_side_papers")
        risk = True
    if dom and "robotic arm" in dom.lower():
        notes.append("dominant_term_is_robotic_arm")
    if dom_sf is not None and dom_sf >= 0.65:
        notes.append("high_dominant_term_share")
        risk = True

    summary["author_support_class"] = sup
    summary["author_reason_summary"] = reason
    summary["author_evidence_risk_flag"] = bool(risk)
    summary["author_evidence_notes"] = notes[:3]


def _stage5_author_compact_explanation_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "author_id": summary.get("author_id"),
        "paper_count": summary.get("paper_count"),
        "mainline_paper_count": summary.get("mainline_paper_count"),
        "side_only_paper_count": summary.get("side_only_paper_count"),
        "multi_hit_paper_count": summary.get("multi_hit_paper_count"),
        "mainline_support_ratio": summary.get("mainline_support_ratio"),
        "dominant_term": summary.get("dominant_term"),
        "dominant_term_share": summary.get("dominant_term_share"),
        "top_paper_hit_quality_class": summary.get("top_paper_hit_quality_class"),
        "author_support_class": summary.get("author_support_class"),
        "author_reason_summary": summary.get("author_reason_summary"),
    }


def _print_stage5_author_payload_summary(
    author_payload: List[Dict[str, Any]],
    *,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    author_payload = author_payload or []
    authors_total = len(author_payload)
    sup_ctr: Counter = Counter()
    risk_n = 0
    for rec in author_payload:
        s = rec.get("author_payload_summary") or {}
        if not isinstance(s, dict):
            continue
        sup_ctr[str(s.get("author_support_class") or "unknown")] += 1
        if s.get("author_evidence_risk_flag"):
            risk_n += 1

    ranked = sorted(
        author_payload,
        key=lambda r: float((r.get("author_payload_summary") or {}).get("top_paper_score") or 0.0),
        reverse=True,
    )
    top_ex = [
        _stage5_author_compact_explanation_row(r.get("author_payload_summary") or {})
        for r in ranked[:12]
    ]
    side_pool = [
        r
        for r in ranked
        if str((r.get("author_payload_summary") or {}).get("author_support_class") or "")
        in ("side_only_supported", "mixed_but_side_heavy", "single_paper_supported")
    ]
    side_ex = [
        _stage5_author_compact_explanation_row(r.get("author_payload_summary") or {})
        for r in side_pool[:8]
    ]

    print("\n" + "-" * 80)
    print("[Stage5 author payload summary]")
    print("-" * 80)
    print(f"authors_total={authors_total} support_class_counter={dict(sup_ctr)} risk_flag_count={risk_n}")
    print("--- top author examples (by top_paper_score, max 12) ---")
    for row in top_ex:
        print(f"  {row}")
    print("--- side-heavy top author examples (max 8) ---")
    for row in side_ex:
        print(f"  {row}")
    print("-" * 80 + "\n")


# --- Step4：LTR-like 多特征后置重审（不训练模型；不改召回/解释字段定义）---
_STAGE4_PAPER_LTR_KEYS = ("paper_rerank_score", "paper_rerank_factor", "paper_rerank_notes")
STAGE4_PAPER_LTR_FACTOR_CLIP = (0.80, 1.12)
STAGE5_AUTHOR_LTR_FACTOR_CLIP = (0.78, 1.10)


def _compute_stage4_ltr_like_paper_rerank_score(rec: Dict[str, Any]) -> None:
    """
    基于 Step2 结构字段的温和联合调整；不覆盖 paper_score_final / paper_score（全局 cap 与审计仍看原分）。
    Stage5 聚合使用 paper_rerank_score（经 wid_to_hits_and_score → papers[].score）。
    """
    base = float(rec.get("paper_score_final") if rec.get("paper_score_final") is not None else rec.get("paper_score") or 0.0)
    hq = str(rec.get("hit_quality_class") or "").strip()
    ml = int(rec.get("mainline_term_count") or 0)
    sd = int(rec.get("side_term_count") or 0)
    mhs = int(rec.get("multi_hit_strength") or 0)
    g = float(rec.get("grounding_score") or 0.0)
    jd_v = rec.get("jd_align")
    try:
        jd_f = float(jd_v) if jd_v is not None else 0.5
    except (TypeError, ValueError):
        jd_f = 0.5
    hb_v = rec.get("hierarchy_bonus")
    try:
        hb_f = float(hb_v) if hb_v is not None else 1.0
    except (TypeError, ValueError):
        hb_f = 1.0

    delta = 0.0
    notes: List[str] = []
    if hq == "mainline_resonance":
        delta += 0.06
    elif hq == "mainline_plus_support":
        delta += 0.05
    elif hq == "single_hit_mainline":
        delta += 0.04
    elif hq == "single_hit_side":
        delta -= 0.045
        notes.append("side_hit_quality")
    elif hq == "side_only_or_accidental_multi_hit":
        delta -= 0.055
        notes.append("side_multi_hit_quality")

    delta += 0.014 * float(min(ml, 3))
    delta -= 0.009 * float(min(sd, 3))
    delta += 0.018 * float(min(mhs, 2))
    delta += float(np.clip((g - 0.26) * 0.07, -0.028, 0.028))
    delta += float(np.clip((jd_f - 0.52) * 0.055, -0.026, 0.026))
    delta += float(np.clip((hb_f - 1.0) * 0.12, -0.022, 0.022))

    lo, hi = STAGE4_PAPER_LTR_FACTOR_CLIP
    fac = float(np.clip(1.0 + delta, lo, hi))
    rec["paper_rerank_factor"] = fac
    rec["paper_rerank_score"] = float(base * fac)
    rec["paper_rerank_notes"] = notes[:3]


def _compute_stage5_ltr_like_author_rerank_score(auth_rec: Dict[str, Any], summary: Dict[str, Any]) -> None:
    """作者层结构重审；final_score_base 用 top_paper_score 作 Stage4 侧代理基分（Stage5 内仍会再聚合）。"""
    base = float(summary.get("top_paper_score") or 0.0)
    mhr = float(summary.get("mainline_support_ratio") or 0.0)
    ssr = float(summary.get("side_support_ratio") or 0.0)
    sup_cls = str(summary.get("author_support_class") or "")
    risk = bool(summary.get("author_evidence_risk_flag"))
    dom_share = summary.get("dominant_term_share")
    try:
        dom_sf = float(dom_share) if dom_share is not None else None
    except (TypeError, ValueError):
        dom_sf = None
    hq_m = int(summary.get("high_quality_multi_hit_count") or 0)
    so_m = int(summary.get("side_only_multi_hit_count") or 0)

    delta = 0.0
    notes: List[str] = []
    delta += 0.11 * mhr
    delta -= 0.09 * ssr

    if sup_cls == "mainline_supported":
        delta += 0.038
    elif sup_cls == "mixed_but_side_heavy":
        delta -= 0.028
        notes.append("mixed_side_heavy_profile")
    elif sup_cls == "side_only_supported":
        delta -= 0.048
        notes.append("side_only_profile")
    elif sup_cls == "single_paper_supported":
        delta -= 0.055
        notes.append("single_paper_profile")

    if risk:
        delta -= 0.038
        notes.append("risk_flag")

    if dom_sf is not None and sup_cls in (
        "side_only_supported",
        "mixed_but_side_heavy",
        "single_paper_supported",
    ):
        if dom_sf >= 0.72:
            delta -= 0.055 * float(min(1.0, max(0.0, (dom_sf - 0.52) / 0.48)))
            notes.append("high_dom_share_side_profile")

    delta += 0.022 * float(min(hq_m, 2))
    delta -= 0.016 * float(min(so_m, 2))

    lo, hi = STAGE5_AUTHOR_LTR_FACTOR_CLIP
    fac = float(np.clip(1.0 + delta, lo, hi))
    auth_rec["author_rerank_factor"] = fac
    auth_rec["final_score_base"] = base
    auth_rec["final_score_reranked"] = float(base * fac)
    auth_rec["author_rerank_score"] = float(base * fac)
    auth_rec["author_rerank_notes"] = notes[:3]


def _stage4_paper_ltr_compact_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "wid": rec.get("wid"),
        "hit_terms": rec.get("hit_terms"),
        "hit_quality_class": rec.get("hit_quality_class"),
        "paper_score_final": rec.get("paper_score_final"),
        "paper_rerank_factor": rec.get("paper_rerank_factor"),
        "paper_rerank_score": rec.get("paper_rerank_score"),
        "paper_rerank_notes": rec.get("paper_rerank_notes"),
    }


def _print_stage4_paper_ltr_rerank_summary(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    *,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    kept = [by_wid[w] for w in selected_wids_set if w in by_wid]
    if not kept:
        print("\n[Stage4 paper LTR-like rerank summary] kept_papers_count=0\n")
        return
    boosted = sorted(kept, key=lambda r: -float(r.get("paper_rerank_factor") or 1.0))[:8]
    penal = sorted(kept, key=lambda r: float(r.get("paper_rerank_factor") or 1.0))[:8]
    print("\n" + "-" * 80)
    print("[Stage4 paper LTR-like rerank summary]")
    print("-" * 80)
    print(f"kept_papers_count={len(kept)} factor_clip={STAGE4_PAPER_LTR_FACTOR_CLIP}")
    print("--- top boosted papers (by paper_rerank_factor, max 8) ---")
    for r in boosted:
        print(f"  {_stage4_paper_ltr_compact_row(r)}")
    print("--- top penalized papers (lowest factor, max 8) ---")
    for r in penal:
        print(f"  {_stage4_paper_ltr_compact_row(r)}")
    print("-" * 80 + "\n")


def _stage5_author_ltr_compact_row(auth_rec: Dict[str, Any]) -> Dict[str, Any]:
    s = auth_rec.get("author_payload_summary") or {}
    return {
        "author_id": auth_rec.get("aid"),
        "original_score": auth_rec.get("final_score_base"),
        "rerank_factor": auth_rec.get("author_rerank_factor"),
        "rerank_score": auth_rec.get("author_rerank_score"),
        "author_support_class": s.get("author_support_class"),
        "mainline_support_ratio": s.get("mainline_support_ratio"),
        "dominant_term": s.get("dominant_term"),
        "dominant_term_share": s.get("dominant_term_share"),
        "top_paper_hit_quality_class": s.get("top_paper_hit_quality_class"),
        "author_rerank_notes": auth_rec.get("author_rerank_notes"),
        "author_reason_summary": s.get("author_reason_summary"),
    }


def _print_stage5_author_ltr_rerank_summary(
    author_payload: List[Dict[str, Any]],
    pre_top20_ids: List[str],
    *,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    ap = author_payload or []
    post_top20 = [str(r.get("aid")) for r in ap[:20]]
    ncmp = min(len(pre_top20_ids), len(post_top20))
    changed = sum(1 for i in range(ncmp) if pre_top20_ids[i] != post_top20[i])
    boosted = sorted(ap, key=lambda r: -float(r.get("author_rerank_factor") or 1.0))[:12]
    penal = sorted(ap, key=lambda r: float(r.get("author_rerank_factor") or 1.0))[:12]
    print("\n" + "-" * 80)
    print("[Stage5 author LTR-like rerank summary]")
    print("-" * 80)
    print(f"authors_total={len(ap)} rerank_changed_top20_count={changed} factor_clip={STAGE5_AUTHOR_LTR_FACTOR_CLIP}")
    print("--- top boosted authors (by author_rerank_factor, max 12) ---")
    for r in boosted:
        print(f"  {_stage5_author_ltr_compact_row(r)}")
    print("--- top penalized authors (lowest factor, max 12) ---")
    for r in penal:
        print(f"  {_stage5_author_ltr_compact_row(r)}")
    print("-" * 80 + "\n")


# --- Step5：主线主题保护（后置方向层；不推翻 Step2~4 字段与 LTR 定义）---
_STAGE4_PAPER_MAINLINE_PROT_KEYS = (
    "paper_mainline_protection_factor",
    "paper_mainline_protection_notes",
    "paper_score_protected",
)
STAGE4_PAPER_MAINLINE_PROT_CLIP = (0.88, 1.08)
STAGE5_AUTHOR_MAINLINE_PROT_CLIP = (0.85, 1.08)


def _term_looks_object_or_carrier(term: str) -> bool:
    """极少稳定子串：仅作后置保护微调，不扩展词表体系。"""
    t = (term or "").strip().lower()
    if not t:
        return False
    needles = (
        "robotic arm",
        "robot arm",
        "robot hand",
        "robotic hand",
        "gripper",
        "end effector",
        "end-effector",
        "manipulator",
        "actuator",
        "quadcopter",
        "drone body",
        "hardware prototype",
    )
    return any(n in t for n in needles)


def _compute_stage4_mainline_protection_factor(rec: Dict[str, Any]) -> None:
    """
    paper_rerank_score 之上再乘温和保护因子；不修改 paper_rerank_score / paper_score_final。
    """
    prs = rec.get("paper_rerank_score")
    try:
        base = float(prs) if prs is not None else float(rec.get("paper_score_final") or rec.get("paper_score") or 0.0)
    except (TypeError, ValueError):
        base = 0.0

    hq = str(rec.get("hit_quality_class") or "").strip()
    ml = int(rec.get("mainline_term_count") or 0)
    sd = int(rec.get("side_term_count") or 0)
    g = float(rec.get("grounding_score") or 0.0)
    jd_v = rec.get("jd_align")
    try:
        jd_f = float(jd_v) if jd_v is not None else 0.5
    except (TypeError, ValueError):
        jd_f = 0.5

    hit_terms = rec.get("hit_terms") if isinstance(rec.get("hit_terms"), list) else []
    carrier_hit = any(_term_looks_object_or_carrier(str(x)) for x in hit_terms)

    delta = 0.0
    notes: List[str] = []

    if hq in ("mainline_resonance", "mainline_plus_support", "single_hit_mainline"):
        delta += 0.05 if hq == "mainline_resonance" else 0.042 if hq == "mainline_plus_support" else 0.034
        notes.append("mainline_hit_structure")
    if ml >= 1:
        delta += 0.012 * float(min(ml, 2))
    if jd_f >= 0.58 and ml >= 1:
        delta += 0.015
        notes.append("jd_align_with_mainline_terms")

    if hq == "single_hit_side":
        delta -= 0.038
        notes.append("single_side_hit")
    elif hq == "side_only_or_accidental_multi_hit":
        delta -= 0.048
        notes.append("side_multi_hit")

    if ml == 0 and sd >= 1 and hq in ("single_hit_side", "side_only_or_accidental_multi_hit"):
        delta -= 0.022
        notes.append("no_mainline_term_on_paper")

    if carrier_hit and hq in ("single_hit_side", "side_only_or_accidental_multi_hit"):
        delta -= 0.032
        notes.append("object_carrier_hit_pattern")

    if g < 0.22 and hq in ("single_hit_side", "side_only_or_accidental_multi_hit"):
        delta -= 0.018

    lo, hi = STAGE4_PAPER_MAINLINE_PROT_CLIP
    fac = float(np.clip(1.0 + delta, lo, hi))
    rec["paper_mainline_protection_factor"] = fac
    rec["paper_mainline_protection_notes"] = notes[:3]
    rec["paper_score_protected"] = float(base * fac)


def _compute_stage5_mainline_protection_factor(auth_rec: Dict[str, Any], summary: Dict[str, Any]) -> None:
    """在 final_score_reranked 上乘温和因子；保留 Step4 rerank 字段。"""
    try:
        base = float(auth_rec.get("final_score_reranked") or 0.0)
    except (TypeError, ValueError):
        base = 0.0

    sup_cls = str(summary.get("author_support_class") or "")
    mhr = float(summary.get("mainline_support_ratio") or 0.0)
    ssr = float(summary.get("side_support_ratio") or 0.0)
    risk = bool(summary.get("author_evidence_risk_flag"))
    dom = str(summary.get("dominant_term") or "").strip()
    dom_share = summary.get("dominant_term_share")
    try:
        dom_sf = float(dom_share) if dom_share is not None else None
    except (TypeError, ValueError):
        dom_sf = None
    top_hqc = str(summary.get("top_paper_hit_quality_class") or "").strip()
    pc = int(summary.get("paper_count") or 0)

    delta = 0.0
    notes: List[str] = []

    if sup_cls == "mainline_supported" or mhr > 0.01:
        delta += 0.045 * float(min(1.0, mhr * 2.2 + (0.25 if sup_cls == "mainline_supported" else 0.0)))
        notes.append("mainline_author_structure")
    if top_hqc in ("single_hit_mainline", "mainline_plus_support", "mainline_resonance"):
        delta += 0.022
        notes.append("top_paper_mainline_quality")

    if sup_cls == "side_only_supported":
        delta -= 0.048
        notes.append("side_only_supported_profile")
    elif sup_cls == "single_paper_supported":
        delta -= 0.042
        notes.append("single_paper_supported_profile")
    elif sup_cls == "mixed_but_side_heavy":
        delta -= 0.02
        notes.append("mixed_side_heavy_profile")

    if risk and mhr < 0.08:
        delta -= 0.028
        notes.append("risk_low_mainline")

    if (
        dom_sf is not None
        and dom_sf >= 0.78
        and _term_looks_object_or_carrier(dom)
        and sup_cls in ("side_only_supported", "single_paper_supported", "mixed_but_side_heavy")
    ):
        delta -= 0.055 * float(min(1.0, (dom_sf - 0.55) / 0.45))
        notes.append("high_share_object_carrier_dom")

    if pc <= 1 and ssr >= 0.85 and mhr < 0.05:
        delta -= 0.025
        notes.append("singleton_side_heavy")

    lo, hi = STAGE5_AUTHOR_MAINLINE_PROT_CLIP
    fac = float(np.clip(1.0 + delta, lo, hi))
    auth_rec["author_mainline_protection_factor"] = fac
    auth_rec["author_mainline_protection_notes"] = notes[:3]
    auth_rec["final_score_protected"] = float(base * fac)


def _stage4_paper_mainline_prot_compact_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "wid": rec.get("wid"),
        "hit_terms": rec.get("hit_terms"),
        "hit_quality_class": rec.get("hit_quality_class"),
        "paper_rerank_score": rec.get("paper_rerank_score"),
        "protection_factor": rec.get("paper_mainline_protection_factor"),
        "paper_score_protected": rec.get("paper_score_protected"),
        "protection_notes": rec.get("paper_mainline_protection_notes"),
    }


def _print_stage4_paper_mainline_protection_summary(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    *,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    kept = [by_wid[w] for w in selected_wids_set if w in by_wid]
    if not kept:
        print("\n[Stage4 paper mainline protection summary] kept_papers_count=0\n")
        return
    boosted = sorted(kept, key=lambda r: -float(r.get("paper_mainline_protection_factor") or 1.0))[:8]
    suppressed = sorted(kept, key=lambda r: float(r.get("paper_mainline_protection_factor") or 1.0))[:8]
    print("\n" + "-" * 80)
    print("[Stage4 paper mainline protection summary]")
    print("-" * 80)
    print(f"kept_papers_count={len(kept)} factor_clip={STAGE4_PAPER_MAINLINE_PROT_CLIP}")
    print("--- top protected papers (highest factor, max 8) ---")
    for r in boosted:
        print(f"  {_stage4_paper_mainline_prot_compact_row(r)}")
    print("--- top suppressed papers (lowest factor, max 8) ---")
    for r in suppressed:
        print(f"  {_stage4_paper_mainline_prot_compact_row(r)}")
    print("-" * 80 + "\n")


def _stage5_author_mainline_prot_compact_row(auth_rec: Dict[str, Any]) -> Dict[str, Any]:
    s = auth_rec.get("author_payload_summary") or {}
    return {
        "author_id": auth_rec.get("aid"),
        "author_support_class": s.get("author_support_class"),
        "dominant_term": s.get("dominant_term"),
        "dominant_term_share": s.get("dominant_term_share"),
        "final_score_reranked": auth_rec.get("final_score_reranked"),
        "protection_factor": auth_rec.get("author_mainline_protection_factor"),
        "final_score_protected": auth_rec.get("final_score_protected"),
        "protection_notes": auth_rec.get("author_mainline_protection_notes"),
    }


def _print_stage5_author_mainline_protection_summary(
    author_payload: List[Dict[str, Any]],
    pre_prot_top20_ids: List[str],
    *,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    ap = author_payload or []
    post_top20 = [str(r.get("aid")) for r in ap[:20]]
    ncmp = min(len(pre_prot_top20_ids), len(post_top20))
    changed = sum(1 for i in range(ncmp) if pre_prot_top20_ids[i] != post_top20[i])
    boosted = sorted(ap, key=lambda r: -float(r.get("author_mainline_protection_factor") or 1.0))[:12]
    suppressed = sorted(ap, key=lambda r: float(r.get("author_mainline_protection_factor") or 1.0))[:12]
    print("\n" + "-" * 80)
    print("[Stage5 author mainline protection summary]")
    print("-" * 80)
    print(f"authors_total={len(ap)} protected_top20_changed_count={changed} factor_clip={STAGE5_AUTHOR_MAINLINE_PROT_CLIP}")
    print("--- top protected authors (highest factor, max 12) ---")
    for r in boosted:
        print(f"  {_stage5_author_mainline_prot_compact_row(r)}")
    print("--- top suppressed authors (lowest factor, max 12) ---")
    for r in suppressed:
        print(f"  {_stage5_author_mainline_prot_compact_row(r)}")
    print("-" * 80 + "\n")


def run_stage4(
    recall,
    vocab_ids: List[int],
    regex_str: str,
    term_scores: Optional[Dict[int, float]] = None,
    term_retrieval_roles: Optional[Dict[int, str]] = None,
    term_meta: Optional[Dict[int, Dict[str, Any]]] = None,
    jd_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    阶段 4：二层论文召回。用带权学术词沿 HAS_TOPIC 拉论文，按 paper_score 全局排序后截断。

    - 取消论文层 domain 硬过滤，改为软奖励（匹配则乘 DOMAIN_BONUS_MATCH，否则 1.0）。
    - per-(vid,wid) 贡献链不变；按 wid 合并 hits 后乘 **hierarchy_consensus_bonus**（词侧 vocabulary_topic_stats 与 JD 三级 overlap 得 consensus；再相对**本批 wid 池** consensus **中位数**映射：clip(1+β·Δ, 默认 [0.82,1.15])，无硬过滤）。若 **term_meta** 含 **`paper_select_lane_tier`**（Stage3）：**仅当** 该 wid 上 **strong_main_axis_core** 的 Σ(term_score×idf) 占比 ≥ **`STAGE4_HIERARCHY_STRONG_AXIS_WEIGHT_FRAC`** 时才允许 bonus>1，否则 **cap 至 1.0**（bonus_core / support 不再吃组级正向放大）。见 **`[Stage4 hierarchy bonus by term-group audit]`**。
    - paper_score = Σ term_contrib × hierarchy_bonus（再全局排序截断），含 Stage3 词质量与 per-term 限流。
    - 词侧熔断放宽为 MELT_RATIO（默认 5%），避免合理词被误杀。

    输入:
      - vocab_ids: 参与检索的词汇 ID（即 final_term_ids_for_paper）。
      - regex_str: 领域正则，用于计算 domain_bonus；为空则不奖励。
      - term_scores: vid -> Stage3 的 final_score；若为 None 则按 1.0 处理。
      - term_meta: 可选，vid -> {term,parent_anchor,...,retrieval_role,**paper_select_lane_tier**,parent_anchor_final_score,rank,...}，用于 grounding 与 **hierarchy 正向 bonus 门控**（与 Stage3 **strong_main_axis_core** 对齐）。
      - jd_text: 可选，整段 JD 文本；用于 grounding 计算时的辅助关键词命中。

    返回: list of { 'aid': str, 'papers': [ { wid, hits, weight, title, year, domains }, ... ] }，供 Stage5 消费。
    """
    di = getattr(recall, "debug_info", None)

    _label_path_stdout = bool(
        not getattr(recall, "silent", False) and getattr(recall, "verbose", False)
    )

    def _lp(*args, **kwargs):
        if _label_path_stdout:
            print(*args, **kwargs)

    def _merge_hits(old_hits: List[Dict[str, Any]], new_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按 canonical vid 合并 hit，避免同 wid 下多次覆盖/重复。
        规则：同 vid 保留 idf 更高的一条（视为更强命中证据）。
        """
        by_vid: Dict[str, Dict[str, Any]] = {}
        for h in (old_hits or []) + (new_hits or []):
            if not isinstance(h, dict):
                continue
            vid_s = str(h.get("vid") or "")
            if not vid_s:
                continue
            prev = by_vid.get(vid_s)
            if prev is None:
                by_vid[vid_s] = dict(h)
                continue
            if float(h.get("idf") or 0.0) > float(prev.get("idf") or 0.0):
                by_vid[vid_s] = dict(h)
        return list(by_vid.values())

    def _save_sub(ms: Dict[str, float]) -> None:
        if di is not None:
            di.stage4_sub_ms = ms

    if not vocab_ids or not getattr(recall, "graph", None):
        _save_sub({})
        return []
    v_ids = [int(x) for x in vocab_ids if x is not None]
    if not v_ids:
        _save_sub({})
        return []
    total_w = float(getattr(recall, "total_work_count", 1e6) or 1e6)
    term_scores = term_scores or {}
    # 统一用 int key 查找
    def _term_score(vid: int) -> float:
        return float(term_scores.get(vid) or term_scores.get(str(vid)) or 1.0)

    def _get_term_meta(vid: int) -> Dict[str, Any]:
        if not term_meta:
            return {}
        return term_meta.get(vid) or term_meta.get(str(vid)) or {}

    def _compute_grounding_score(
        vid: int,
        title: str,
        domains: str,
    ) -> Dict[str, float]:
        """
        Stage4 的 paper grounding：用 paper 的 title/domains 对齐岗位主轴与 term 证据。
        返回：
          - grounding: 0~1（主轴/词面落地强度）
          - off_topic_penalty: 额外偏题惩罚（用于抑制泛命中论文池）
        """
        meta = _get_term_meta(vid)

        term_text = str(meta.get("term") or "").lower()
        parent_anchor = str(meta.get("parent_anchor") or "").lower()
        parent_primary = str(meta.get("parent_primary") or "").lower()
        retrieval_role = str(meta.get("retrieval_role") or "").lower()

        t = (title or "").lower()
        d = (domains or "").lower()
        _jd = (jd_text or "").lower()

        # 1) 词面命中
        lexical_hit = 1.0 if term_text and term_text in t else 0.0

        # 2) 父锚 / 父主词命中（主轴证据）
        anchor_axis_hit = 0.0
        if parent_anchor and parent_anchor in t:
            anchor_axis_hit += 0.5
        if parent_primary and parent_primary in t:
            anchor_axis_hit += 0.5
        anchor_axis_hit = min(anchor_axis_hit, 1.0)

        # 3) 机器人/控制主轴关键词命中（title/domains）
        axis_keywords = [
            "robot",
            "robotic",
            "manipulator",
            "motion control",
            "robot control",
            "dynamics",
            "kinematics",
            "trajectory",
            "planning",
            "optimal control",
            "state estimation",
            "simulation",
        ]
        axis_hit_cnt = sum(1 for kw in axis_keywords if (kw in t) or (kw in d))
        jd_axis_match = min(axis_hit_cnt / 4.0, 1.0)

        # 4) 偏题惩罚（交通/调度/泛 AI 等）
        off_topic_penalty = 1.0
        off_keywords = [
            "charging station",
            "vehicle routing",
            "traffic",
            "transportation",
            "logistics",
            "supply chain",
            "crystallization",
            "v2x",
            "cybersecurity",
        ]
        if any(kw in t for kw in off_keywords):
            # 加重泛交通/物流/工业流程偏题惩罚，优先清理“看似相关但主轴不对”的高分论文
            off_topic_penalty *= 0.55

        # RL 相关额外约束：如不是机器人/控制体系，则压
        if ("reinforcement learning" in term_text) or ("q-learning" in term_text):
            if not any(
                kw in t
                for kw in ["robot", "robotic", "control", "manipulator", "motion", "locomotion", "planning"]
            ):
                # RL 词必须更贴机器人/控制主轴，否则强压
                off_topic_penalty *= 0.45

        # route planning 相关额外约束：交通/船舶/公交定制路线等压（略严于旧版 0.40）
        if "route planning" in term_text:
            if any(
                kw in t
                for kw in [
                    "charging station",
                    "traffic",
                    "transportation",
                    "vehicle",
                    "bus",
                    "rescue route",
                    "ship",
                    "shipping",
                    "maritime",
                    "expressway",
                    "customized bus",
                    "intelligent ship",
                ]
            ):
                off_topic_penalty *= 0.32

        # robotic arm 相关额外约束：纯器件/结构而非控制也压一点
        if "robotic arm" in term_text:
            if not any(
                kw in t
                for kw in [
                    "control",
                    "trajectory",
                    "motion",
                    "dynamics",
                    "planning",
                    "kinematics",
                    "manipulation",
                    "grasp",
                ]
            ):
                off_topic_penalty *= 0.60

        # robot control：字面极泛，易误收「聊天/安全/LLM」类标题；对齐 RL/route/arm 的专项约束风格
        if "robot control" in term_text:
            pseudo_control_kw = [
                "chat control",
                "backdoor",
                "safety",
                "alignment",
                "llm",
                "large language",
                "language model",
                "prompt",
                "gpt",
            ]
            if any(kw in t for kw in pseudo_control_kw):
                off_topic_penalty *= 0.25
            control_axis_keywords = [
                "motion",
                "manipulator",
                "trajectory",
                "locomotion",
                "kinematics",
                "dynamics",
                "path planning",
                "rehabilitation robot",
                "mobile robot",
                "quadruped",
                "biped",
                "humanoid",
            ]
            if not any((kw in t) or (kw in d) for kw in control_axis_keywords):
                off_topic_penalty *= 0.55

        # supervised learning：易混入通用 ML/自监督综述；无机器人/控制主轴则压
        if "supervised learning" in term_text:
            if not any(
                kw in t or kw in d
                for kw in [
                    "robot",
                    "robotic",
                    "motion",
                    "control",
                    "manipulation",
                    "reinforcement",
                    "imitation learning",
                    "trajectory",
                    "locomotion",
                ]
            ):
                off_topic_penalty *= 0.50

        # 提高词面命中权重，降低“只沾主轴泛词”论文的 grounding 分
        grounding = (0.55 * lexical_hit + 0.20 * anchor_axis_hit + 0.25 * jd_axis_match)
        if retrieval_role == "paper_support":
            grounding *= 0.92

        grounding = max(0.0, min(1.0, float(grounding)))
        return {"grounding": grounding, "off_topic_penalty": float(off_topic_penalty)}

    def _ensure_stage4_resources() -> None:
        """
        Stage4 资源懒加载（最小补丁）：
        1) 论文摘要向量矩阵
        2) paper_id -> row_idx 映射
        3) 缓存到 recall 对象，避免重复加载
        """
        if bool(getattr(recall, "_stage4_vec_ready", False)):
            return

        vec_path = os.path.join(INDEX_DIR, "abstract_vectors.npy")
        map_path = ABSTRACT_MAP_PATH

        paper_vecs = None
        paper_id_to_row: Dict[str, int] = {}
        paper_norms = None

        try:
            if os.path.exists(vec_path):
                paper_vecs = np.load(vec_path, mmap_mode="r")
        except Exception:
            paper_vecs = None
            try:
                if os.path.exists(vec_path):
                    paper_vecs = np.load(vec_path)
            except Exception:
                paper_vecs = None

        try:
            if os.path.exists(map_path):
                with open(map_path, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)
                if isinstance(raw_map, list):
                    paper_id_to_row = {str(pid): i for i, pid in enumerate(raw_map)}
                elif isinstance(raw_map, dict):
                    paper_id_to_row = {str(pid): int(idx) for pid, idx in raw_map.items()}
        except Exception:
            paper_id_to_row = {}

        if isinstance(paper_vecs, np.ndarray) and paper_vecs.size > 0 and paper_vecs.ndim == 2:
            try:
                if isinstance(paper_vecs, np.memmap):
                    paper_norms = None
                else:
                    paper_norms = np.linalg.norm(paper_vecs, axis=1)
                    paper_norms = np.where(paper_norms <= 1e-12, 1e-12, paper_norms)
            except Exception:
                paper_norms = None

        setattr(recall, "_paper_abstract_vecs", paper_vecs)
        setattr(recall, "_paper_id_to_row", paper_id_to_row)
        setattr(recall, "_paper_abstract_norms", paper_norms)
        setattr(recall, "_stage4_vec_ready", True)

        dim = int(paper_vecs.shape[1]) if isinstance(paper_vecs, np.ndarray) and paper_vecs.ndim == 2 else 0
        _lp(f"[Stage4 vec ready] papers={len(paper_id_to_row)} dim={dim}")

    def _score_paper_for_term(
        vid: int,
        wid: str,
        title: str,
        domains: str,
        year: Any,
        idf_weight: float,
        domain_bonus: float,
        term_final: float,
        role_weight: float,
        query_vec_1d: Optional[np.ndarray],
        query_norm: float,
        jd_align_pre: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Stage4 单篇打分（最小侵入版）：
        在原有 term grounding + penalty 基础上，新增 JD↔摘要向量 jd_align 软融合。
        """
        tt = term_type_cache.get(vid)
        if tt is None:
            tt = _compute_term_type_factors(vid)
            term_type_cache[vid] = tt

        ground = _compute_grounding_score(vid, title, domains)
        term_grounding = float(ground["grounding"]) * float(tt["grounding_factor"])
        term_grounding = max(0.0, min(1.0, term_grounding))
        off_topic_penalty = float(ground["off_topic_penalty"])
        term_paper_role_factor = float(tt["paper_factor"])

        # 默认中性值：向量缺失时不一票否决（jd_align_pre 由批处理与逐行等价）
        jd_align = 0.50
        if jd_align_pre is not None:
            jd_align = max(0.0, min(1.0, float(jd_align_pre)))
        else:
            paper_vecs = getattr(recall, "_paper_abstract_vecs", None)
            paper_id_to_row = getattr(recall, "_paper_id_to_row", {}) or {}
            paper_norms = getattr(recall, "_paper_abstract_norms", None)
            row_idx = paper_id_to_row.get(str(wid))
            if (
                row_idx is not None
                and isinstance(paper_vecs, np.ndarray)
                and query_vec_1d is not None
                and query_norm > 1e-12
                and 0 <= int(row_idx) < len(paper_vecs)
            ):
                try:
                    ri = int(row_idx)
                    pv = np.asarray(paper_vecs[ri], dtype=np.float32).flatten()
                    if pv.size == query_vec_1d.size:
                        if paper_norms is not None and ri < len(paper_norms):
                            pn = float(paper_norms[ri])
                        else:
                            pn = float(np.linalg.norm(pv))
                        if pn > 1e-12:
                            cos = float(np.dot(query_vec_1d, pv) / (query_norm * pn))
                            jd_align = 0.5 * (cos + 1.0)
                            jd_align = max(0.0, min(1.0, jd_align))
                except Exception:
                    jd_align = 0.50

        # 关键融合（最小修正）：
        # 1) term_grounding 继续当主轴
        # 2) jd_align 只做弱辅助，不能“救活”term 不够像的论文
        # 3) term_grounding 很低时，JD 相似度影响进一步缩小
        jd_boost = 0.85 + 0.15 * jd_align
        if term_grounding < 0.20:
            jd_boost = 0.92 + 0.08 * jd_align

        hybrid_grounding = term_grounding * jd_boost
        hybrid_grounding = max(0.0, min(1.0, float(hybrid_grounding)))

        recency = compute_paper_recency(year, None)
        base_term_contrib = term_final * role_weight * idf_weight * domain_bonus * recency
        final_paper_score = (
            base_term_contrib
            * (0.10 + 0.90 * hybrid_grounding)
            * off_topic_penalty
            * term_paper_role_factor
        )

        # 批注：Stage3 已把词分成 paper_primary / paper_support；support 过门后仍易吃「JD 高 + tg 低」。
        # 不在硬门一刀砍死，而在分数上再压一档，避免与 core 词在 wid 聚合、作者榜里同台竞技。
        meta_sr = _get_term_meta(vid)
        retrieval_role_local = str(
            meta_sr.get("retrieval_role")
            or (term_retrieval_roles.get(vid) if term_retrieval_roles else None)
            or (term_retrieval_roles.get(str(vid)) if term_retrieval_roles else None)
            or ""
        ).strip().lower()
        if retrieval_role_local == "paper_support":
            final_paper_score *= 0.85
            if term_grounding < 0.10:
                final_paper_score *= 0.25
            elif term_grounding < 0.18:
                final_paper_score *= 0.45
            elif term_grounding < 0.28:
                final_paper_score *= 0.70

        return {
            "term_grounding": term_grounding,
            "jd_align": jd_align,
            "hybrid_grounding": hybrid_grounding,
            "offtopic_penalty": off_topic_penalty,
            "paper_factor": term_paper_role_factor,
            "year_factor": recency,
            "domain_bonus": domain_bonus,
            "idf_weight": idf_weight,
            "final_paper_score": final_paper_score,
        }

    def _compute_term_type_factors(vid: int) -> Dict[str, float]:
        """
        Stage4 最小 term-type 因子（无词表硬编码）：
        - 方法骨架词：grounding 略放宽、local cap 略放宽
        - 对象型词：grounding / paper 贡献 / local cap 略收紧
        """
        meta = _get_term_meta(vid)
        term_lc = str(meta.get("term") or "").strip().lower()
        retrieval_role = str(meta.get("retrieval_role") or "").strip().lower()
        if term_lc == "robot control":
            retrieval_role = "paper_support"
        stage3_bucket = str(meta.get("stage3_bucket") or "").strip().lower()
        can_expand = bool(meta.get("can_expand", False))
        term_role = str(meta.get("term_role") or "").strip().lower()

        object_like_penalty = float(meta.get("object_like_penalty", 1.0) or 1.0)
        bonus_term_penalty = float(meta.get("bonus_term_penalty", 1.0) or 1.0)
        generic_penalty = float(meta.get("generic_penalty", 1.0) or 1.0)
        object_like_risk = 1.0 - max(0.0, min(1.0, object_like_penalty))
        generic_risk = 1.0 - max(0.0, min(1.0, generic_penalty))
        bonus_risk = 1.0 - max(0.0, min(1.0, bonus_term_penalty))

        method_like = (
            can_expand
            or stage3_bucket == "core"
            or retrieval_role == "paper_primary"
            or term_role == "primary"
        )
        object_like = object_like_risk >= 0.35 or object_like_penalty <= 0.82

        grounding_factor = 1.00
        paper_factor = 1.00
        local_cap = TERM_MAX_PAPERS

        if method_like and not object_like:
            grounding_factor = 1.14
            local_cap = min(TERM_MAX_PAPERS, 15)
        elif object_like:
            # 对象型 term（继续收紧版）：保留召回，但进一步压制作者榜“对象词霸榜”
            # - grounding_factor 下调：降低仅靠主轴泛命中的通过强度
            # - paper_factor 下调：削弱进入作者聚合前的单篇贡献
            # - local_cap 收紧到 3：减少同一对象词向 Stage5 喂入的高分论文数量
            grounding_factor = 0.72
            paper_factor = 0.60 if retrieval_role == "paper_primary" else 0.70
            local_cap = min(TERM_MAX_PAPERS, 3)
        else:
            # 普通词：轻微收敛，防止泛词累积分过快
            if retrieval_role == "paper_primary":
                paper_factor = 0.93
            local_cap = min(TERM_MAX_PAPERS, 12)

        # 泛词/bonus 风险高时，再做轻微收敛（不做硬门）
        if generic_risk > 0.40:
            grounding_factor *= 0.96
        if bonus_risk > 0.35:
            paper_factor *= 0.96

        # 与 term_retrieval_roles 覆盖一致：umbrella「robot control」额外压低 paper_factor，削弱进入 Stage5 前的单 term 论文分。
        if term_lc == "robot control":
            paper_factor = min(float(paper_factor), 0.68)

        return {
            # 下限放宽到 0.70，允许对象词 grounding_factor=0.72 生效
            "grounding_factor": max(0.70, min(1.20, float(grounding_factor))),
            # 下限放宽到 0.55，允许对象词 primary 使用 0.60 真正生效
            "paper_factor": max(0.55, min(1.00, float(paper_factor))),
            # 下限放到 3，确保对象型 term 的收紧策略可真实生效
            "local_cap": int(max(3, min(TERM_MAX_PAPERS, local_cap))),
        }

    # ---------- 第一层：按 term 拉 (vid, wid, idf_weight, domain_bonus, year)，无论文层硬过滤 ----------
    params: Dict[str, Any] = {"v_ids": v_ids, "total_w": total_w}
    if regex_str and regex_str.strip():
        params["regex"] = regex_str.strip()
        domain_bonus_expr = (
            "CASE WHEN $regex IS NOT NULL AND size($regex) > 0 AND w.domain_ids =~ $regex "
            f"THEN {DOMAIN_BONUS_MATCH} ELSE {DOMAIN_BONUS_NO_MATCH} END"
        )
    else:
        domain_bonus_expr = str(DOMAIN_BONUS_NO_MATCH)

    _layer1_cap = max(0, min(STAGE4_LAYER1_PER_V_CAP, 200000))
    if _layer1_cap > 0:
        cypher_layer1 = (
            f"MATCH (v:Vocabulary) WHERE v.id IN $v_ids\n"
            f"WITH v, count {{ (v)<-[:HAS_TOPIC]-() }} AS degree_w\n"
            f"WHERE (degree_w * 1.0 / $total_w) < $melt_ratio\n"
            f"WITH v, log10($total_w / (degree_w + 1)) AS idf_weight\n"
            f"MATCH (v)<-[:HAS_TOPIC]-(w:Work)\n"
            f"WITH v, w, idf_weight, {domain_bonus_expr} AS domain_bonus, w.year AS year,\n"
            f"     coalesce(w.title, '') AS title, coalesce(w.domain_ids, '') AS domains\n"
            f"ORDER BY v.id, coalesce(year, 0) DESC, w.id\n"
            f"WITH v, idf_weight, collect({{w:w, db:domain_bonus, y:year, t:title, d:domains}})[0..{_layer1_cap}] AS items\n"
            "UNWIND items AS it\n"
            "RETURN v.id AS vid, it.w.id AS wid, idf_weight, it.db AS domain_bonus, it.y AS year, it.t AS title, it.d AS domains"
        )
    else:
        cypher_layer1 = f"""
    MATCH (v:Vocabulary) WHERE v.id IN $v_ids
    WITH v, count {{ (v)<-[:HAS_TOPIC]-() }} AS degree_w
    WHERE (degree_w * 1.0 / $total_w) < $melt_ratio
    WITH v, log10($total_w / (degree_w + 1)) AS idf_weight
    MATCH (v)<-[:HAS_TOPIC]-(w:Work)
    WITH v, w, idf_weight, {domain_bonus_expr} AS domain_bonus, w.year AS year,
         coalesce(w.title, '') AS title, coalesce(w.domain_ids, '') AS domains
    RETURN v.id AS vid, w.id AS wid, idf_weight, domain_bonus, year, title, domains
    """
    params["melt_ratio"] = MELT_RATIO

    sub_ms: Dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        cursor = recall.graph.run(cypher_layer1, **params)
        rows = list(cursor)
    except Exception:
        sub_ms["cypher1"] = (time.perf_counter() - t0) * 1000.0
        sub_ms["total"] = sub_ms["cypher1"]
        _save_sub(sub_ms)
        return []

    t1 = time.perf_counter()
    sub_ms["cypher1"] = (t1 - t0) * 1000.0
    _unique_w: Set[str] = set()
    for _rr in rows:
        if _rr.get("wid") is not None:
            _unique_w.add(str(_rr.get("wid")))
    sub_ms["cypher1_rows"] = float(len(rows))
    sub_ms["cypher1_unique_wids"] = float(len(_unique_w))

    if not rows:
        sub_ms["total"] = (time.perf_counter() - t0) * 1000.0
        _save_sub(sub_ms)
        return []

    # ---------- Python：recency、role_weight、grounding/off_topic、term_contrib，per-term 限流，再按 paper 聚合 ----------
    # 批注：Stage3 可能对「robot control」标 paper_primary；该词为控制类 umbrella，字面过宽。
    # Stage4 侧覆盖为 paper_support，使 role_weight 与 hits 角色与 motion control 等细项区分，避免泛词 Primary 灌作者榜。
    term_retrieval_roles = dict(term_retrieval_roles or {})
    for _vid0 in v_ids:
        try:
            _vi = int(_vid0)
        except (TypeError, ValueError):
            continue
        _m0 = _get_term_meta(_vi)
        if str(_m0.get("term") or "").strip().lower() == "robot control":
            term_retrieval_roles[_vi] = "paper_support"
            term_retrieval_roles[str(_vi)] = "paper_support"
    by_term: Dict[int, List[tuple]] = defaultdict(list)

    # -------------------------
    # Stage4 诊断：过滤漏斗/死因/样本
    # 只做计数与打印，不改变筛选与聚合逻辑
    # -------------------------
    # 批注：将单一硬门改为“双阈值门”，保留主命中严格性，同时允许高 JD 对齐的弱辅助命中进入。
    PRIMARY_GROUNDING_MIN = 0.12
    SECONDARY_GROUNDING_MIN = 0.06
    SECONDARY_JD_ALIGN_MIN = 0.78

    def _new_funnel() -> Dict[str, int]:
        return {
            "cypher_raw": 0,
            "after_year_filter": 0,
            "after_basic_meta": 0,
            "after_grounding_gate": 0,
            "after_offtopic_penalty_sort": 0,
            "after_local_cap": 0,
            "final_unique": 0,
        }

    def _new_reject_reason() -> Dict[str, int]:
        return {
            "low_grounding": 0,
            "off_topic_penalty_too_low": 0,
            "duplicate_dropped": 0,
            "local_cap_cut": 0,
            "global_cap_cut": 0,
        }

    term_funnel_counts: Dict[int, Dict[str, int]] = defaultdict(_new_funnel)
    term_reject_reason_counts: Dict[int, Dict[str, int]] = defaultdict(_new_reject_reason)
    term_low_grounding_samples: Dict[int, List[Dict[str, Any]]] = defaultdict(list)  # 每 term 最多 3
    term_local_cap_cut_samples: Dict[int, List[Dict[str, Any]]] = defaultdict(list)  # 每 term 最多 3

    # Stage4 摘要向量资源懒加载（只做一次）
    _ensure_stage4_resources()
    query_vec_1d: Optional[np.ndarray] = None
    query_norm: float = 0.0
    try:
        cached = getattr(recall, "_jd_query_vec_1d", None)
        if cached is not None:
            query_vec_1d = np.asarray(cached, dtype=np.float32).flatten()
            query_norm = float(np.linalg.norm(query_vec_1d))
    except Exception:
        query_vec_1d = None
        query_norm = 0.0
    if (query_vec_1d is None or query_norm <= 1e-12) and jd_text:
        try:
            enc = getattr(recall, "_query_encoder", None)
            if enc is not None:
                jd_can = canonical_jd_text_for_encode(jd_text)
                qv, _ = enc.encode(jd_can)
                if qv is not None:
                    query_vec_1d = np.asarray(qv, dtype=np.float32).flatten()
                    query_norm = float(np.linalg.norm(query_vec_1d))
        except Exception:
            query_vec_1d = None
            query_norm = 0.0

    jd_align_map = _batch_jd_align_for_wids(
        _unique_w,
        getattr(recall, "_paper_abstract_vecs", None),
        getattr(recall, "_paper_id_to_row", {}) or {},
        getattr(recall, "_paper_abstract_norms", None),
        query_vec_1d,
        query_norm,
    )

    term_capped_unique_wids: Dict[int, set] = defaultdict(set)
    wid_to_paper_meta: Dict[str, Dict[str, Any]] = {}  # wid -> {title, domains, year}
    term_type_cache: Dict[int, Dict[str, float]] = {}
    # 调试断点：cap 前/后按 term 的论文行，用于 overlap 与生存性分析
    term_rows_before_cap: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    term_rows_after_cap: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    # 审计用：grounding 门控前、评分链路上的各行（非 cap 后 kept）；命名见 [Stage4 pre-gate paper score audit]
    term_kept_paper_audit: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        vid = int(r["vid"])
        raw_wid = r["wid"]
        wid = str(raw_wid) if raw_wid is not None else None
        if wid is None:
            continue

        # -------- funnel: cypher_raw --------
        term_funnel_counts[vid]["cypher_raw"] += 1
        term_funnel_counts[vid]["after_year_filter"] += 1 if r.get("year") is not None else 0
        term_funnel_counts[vid]["after_basic_meta"] += 1

        idf_weight = float(r.get("idf_weight") or 0.0)
        domain_bonus = float(r.get("domain_bonus") or 1.0)
        year = r.get("year")
        title = str(r.get("title") or "")
        domains = str(r.get("domains") or "")
        wid_to_paper_meta[wid] = {"title": title, "domains": domains, "year": year}

        term_final = _term_score(vid)
        role_weight = get_term_role_weight(term_retrieval_roles, vid)
        score_detail = _score_paper_for_term(
            vid=vid,
            wid=wid,
            title=title,
            domains=domains,
            year=year,
            idf_weight=idf_weight,
            domain_bonus=domain_bonus,
            term_final=term_final,
            role_weight=role_weight,
            query_vec_1d=query_vec_1d,
            query_norm=query_norm,
            jd_align_pre=jd_align_map.get(wid),
        )
        term_contrib = float(score_detail["final_paper_score"])
        grounding = float(score_detail["term_grounding"])  # 硬门：先保证论文真的像这个 term
        hybrid_grounding = float(score_detail["hybrid_grounding"])  # 软排：JD 只参与排序
        jd_align = float(score_detail["jd_align"])
        off_topic_penalty = float(score_detail["offtopic_penalty"])
        # 记录“论文贡献前的因子分解”：只用于 debug 打印，不参与排序逻辑
        term_kept_paper_audit[vid].append(
            {
                "paper_id": wid,
                "title": title,
                "term_grounding": float(score_detail["term_grounding"]),
                "jd_align": float(score_detail["jd_align"]),
                "hybrid_grounding": float(score_detail["hybrid_grounding"]),
                "grounding": grounding,
                "gating_grounding": grounding,
                "hybrid_grounding": hybrid_grounding,
                "offtopic_penalty": off_topic_penalty,
                "paper_factor": float(score_detail["paper_factor"]),
                "year_factor": float(score_detail["year_factor"]),
                "domain_bonus": domain_bonus,
                "idf_weight": idf_weight,
                "final_paper_score": term_contrib,
            }
        )

        # 双阈值门：
        # - primary：grounding 必须 >= 0.12
        # - secondary：允许 grounding 较低但 jd_align 足够高的辅助命中进入
        allow_hit = False
        hit_level = "primary"
        if grounding >= PRIMARY_GROUNDING_MIN:
            allow_hit = True
            hit_level = "primary"
        elif grounding >= SECONDARY_GROUNDING_MIN and jd_align >= SECONDARY_JD_ALIGN_MIN:
            allow_hit = True
            hit_level = "secondary"

        if not allow_hit:
            term_reject_reason_counts[vid]["low_grounding"] += 1
            if len(term_low_grounding_samples[vid]) < 3:
                term_low_grounding_samples[vid].append(
                    {
                        "wid": wid,
                        "title": title,
                        "domains": domains,
                        "reason": "low_grounding",
                        "grounding": grounding,
                        "penalty": off_topic_penalty,
                        "jd_align": jd_align,
                    }
                )
            continue

        term_funnel_counts[vid]["after_grounding_gate"] += 1
        # 批注：secondary 命中保留但做轻降权，防止弱命中反客为主。
        effective_term_contrib = term_contrib * (0.65 if hit_level == "secondary" else 1.0)
        by_term[vid].append(
            (
                wid,
                effective_term_contrib,
                idf_weight,
                hit_level,
                grounding,
                jd_align,
            )
        )
        term_rows_before_cap[vid].append(
            {
                "pid": wid,
                "final_paper_score": float(effective_term_contrib),
            }
        )

    # 每个 term 最多保留 TERM_MAX_PAPERS 篇（按 term_contrib 降序）
    limited: List[tuple] = []
    for vid, triples in by_term.items():
        triples.sort(key=lambda x: -x[1])
        tt = term_type_cache.get(vid)
        if tt is None:
            tt = _compute_term_type_factors(vid)
            term_type_cache[vid] = tt
        local_cap = int(tt["local_cap"])

        term_funnel_counts[vid]["after_offtopic_penalty_sort"] = term_funnel_counts[vid][
            "after_grounding_gate"
        ]

        cut_cnt = max(0, len(triples) - local_cap)
        term_reject_reason_counts[vid]["local_cap_cut"] += cut_cnt

        # local cap samples（用于打印，不影响 limited）
        if cut_cnt > 0 and len(term_local_cap_cut_samples[vid]) < 3:
            cut_triples = triples[local_cap : local_cap + 3]
            for (cut_wid, *_rest) in cut_triples:
                meta = wid_to_paper_meta.get(cut_wid) or {"title": "", "domains": ""}
                if len(term_local_cap_cut_samples[vid]) < 3:
                    term_local_cap_cut_samples[vid].append(
                        {
                            "wid": cut_wid,
                            "title": meta.get("title") or "",
                            "domains": meta.get("domains") or "",
                            "reason": "local_cap_cut",
                        }
                    )

        kept_triples = triples[:local_cap]
        term_funnel_counts[vid]["after_local_cap"] = len(kept_triples)
        for (kept_wid, *_rest) in kept_triples:
            term_capped_unique_wids[vid].add(kept_wid)
            term_rows_after_cap[vid].append({"pid": kept_wid})

        for (wid, term_contrib, idf_weight, hit_level, grounding, jd_align) in triples[:local_cap]:
            limited.append((wid, vid, term_contrib, idf_weight, hit_level, grounding, jd_align))

        # 批注：输出每个 term 在关键层级的数量，优先判断是 grounding 还是 cap 在砍样本。
        meta = _get_term_meta(vid)
        term_name = str(meta.get("term") or meta.get("anchor") or meta.get("parent_primary") or vid)
        rr_audit = str(
            meta.get("retrieval_role")
            or term_retrieval_roles.get(vid)
            or term_retrieval_roles.get(str(vid))
            or ""
        ).strip().lower()
        if rr_audit == "paper_support" and kept_triples:
            grounds = [float(x[4]) for x in kept_triples]
            sec_n = sum(1 for x in kept_triples if str(x[3]) == "secondary")
            _lp(
                f"[Stage4 support grounding audit] term='{term_name}' "
                f"kept_total={len(kept_triples)} "
                f"tg_lt_0.10={sum(1 for g in grounds if g < 0.10)} "
                f"tg_lt_0.18={sum(1 for g in grounds if g < 0.18)} "
                f"tg_lt_0.28={sum(1 for g in grounds if g < 0.28)} "
                f"secondary_hits={sec_n}"
            )
        _lp(
            f"[Stage4 term stats] term='{term_name}' "
            f"raw={term_funnel_counts[vid].get('cypher_raw', 0)} "
            f"after_grounding={term_funnel_counts[vid].get('after_grounding_gate', 0)} "
            f"before_cap={len(term_rows_before_cap.get(vid, []))} "
            f"after_cap={len(term_rows_after_cap.get(vid, []))}"
        )
        rows_for_paper_map = kept_triples
        term = term_name
        _lp(f"[Stage4 paper_map source] term='{term}' source_stage='after_cap' rows={len(rows_for_paper_map)}")

    # 断点 1：local cap 前的 cross-term overlap
    _lp("\n[Stage4 overlap before local-cap]")
    term_pid_map_before_local_cap: Dict[int, set] = {
        vid: {str(r.get("pid")) for r in rows if r.get("pid") is not None}
        for vid, rows in term_rows_before_cap.items()
    }
    overlap_pairs: List[tuple] = []
    v_ids_sorted = sorted(term_pid_map_before_local_cap.keys())
    for i, vid_a in enumerate(v_ids_sorted):
        for vid_b in v_ids_sorted[i + 1 :]:
            inter = sorted(term_pid_map_before_local_cap[vid_a].intersection(term_pid_map_before_local_cap[vid_b]))
            if not inter:
                continue
            name_a = str((_get_term_meta(vid_a) or {}).get("term") or vid_a)
            name_b = str((_get_term_meta(vid_b) or {}).get("term") or vid_b)
            _lp(f"  {name_a} x {name_b} -> overlap={len(inter)} sample={inter[:10]}")
            overlap_pairs.append((vid_a, vid_b, inter))

    # 统计型打印：全局 multi-hit 潜力（cap 前）
    pid_term_count: Dict[str, set] = defaultdict(set)
    for vid, rows in term_rows_before_cap.items():
        term_name = str((_get_term_meta(vid) or {}).get("term") or vid)
        for r in rows:
            pid = str(r.get("pid") or "")
            if pid:
                pid_term_count[pid].add(term_name)
    multi_candidates = [(pid, sorted(list(ts))) for pid, ts in pid_term_count.items() if len(ts) >= 2]
    _lp(f"\n[Stage4 multi-hit potential before cap] count={len(multi_candidates)}")
    for pid, terms in multi_candidates[:20]:
        _lp(f"  pid='{pid}' terms={terms}")

    # 断点 2：overlap 论文在各 term 的 cap 前后生存情况（**默认关闭**：STAGE4_OVERLAP_SURVIVAL_MAX_LINES=0）
    cross_term_overlap_pids = {pid for (_, _, inter) in overlap_pairs for pid in inter}
    survival_entries: List[Dict[str, Any]] = []
    for vid, rows in term_rows_before_cap.items():
        term_name = str((_get_term_meta(vid) or {}).get("term") or vid)
        rows_sorted = sorted(rows, key=lambda x: float(x.get("final_paper_score") or 0.0), reverse=True)
        top_after_cap = {str(r.get("pid")) for r in term_rows_after_cap.get(vid, []) if r.get("pid") is not None}
        for rank, r in enumerate(rows_sorted, 1):
            pid = str(r.get("pid") or "")
            if not pid or pid not in cross_term_overlap_pids:
                continue
            sc = float(r.get("final_paper_score") or 0.0)
            surv = pid in top_after_cap
            survival_entries.append(
                {"term": term_name, "pid": pid, "rank": rank, "sc": sc, "surv": surv}
            )
    survived_high = [e for e in survival_entries if e["surv"]]
    survived_high.sort(key=lambda e: -e["sc"])
    danger_cut = [e for e in survival_entries if (not e["surv"]) and int(e["rank"]) <= 15]
    danger_cut.sort(key=lambda e: (int(e["rank"]), -e["sc"]))
    max_lines = max(0, STAGE4_OVERLAP_SURVIVAL_MAX_LINES)
    if max_lines > 0:
        _lp("\n[Stage4 overlap survival audit]")
        picked: List[Dict[str, Any]] = []
        seen_pk: Set[Tuple[str, str]] = set()
        for e in survived_high:
            if len(picked) >= max_lines:
                break
            pk = (e["pid"], e["term"])
            if pk in seen_pk:
                continue
            seen_pk.add(pk)
            picked.append(e)
        for e in danger_cut:
            if len(picked) >= max_lines:
                break
            pk = (e["pid"], e["term"])
            if pk in seen_pk:
                continue
            seen_pk.add(pk)
            picked.append(e)
        for e in picked:
            _lp(
                f"term='{e['term']}' pid='{e['pid']}' "
                f"rank_before_cap={e['rank']} "
                f"score={float(e['sc']):.3f} "
                f"survive_after_cap={e['surv']}"
            )

    # 按 wid 聚合：paper_score = Σ term_contrib，hits 合并为 canonical term 证据
    by_wid: Dict[str, Dict[str, Any]] = {}
    for (wid, vid, term_contrib, idf_weight, hit_level, grounding, jd_align) in limited:
        tt = term_type_cache.get(vid)
        if tt is None:
            tt = _compute_term_type_factors(vid)
            term_type_cache[vid] = tt
        meta = _get_term_meta(vid)
        retrieval_role = (
            term_retrieval_roles.get(vid)
            or term_retrieval_roles.get(str(vid))
            or meta.get("retrieval_role")
            or "paper_primary"
        )
        hit = {
            "vid": str(vid),
            # 批注：Stage4 不再二次清洗 term，直接使用 Stage3/canonical term。
            "term": str(meta.get("term") or ""),
            "idf": float(idf_weight),
            "role": str(retrieval_role),
            "term_score": float(_term_score(vid)),
            "paper_factor": float(tt.get("paper_factor") or 1.0),
            "hit_level": str(hit_level),
            "grounding": float(grounding),
            "jd_align": float(jd_align),
            # Step2 / Step1：供 wid 级 explanation 与 multi-hit 质量分类（不改打分）
            "parent_anchor": str(meta.get("parent_anchor") or ""),
            "parent_primary": str(meta.get("parent_primary") or ""),
            "lane_type": str(meta.get("lane_type") or ""),
            "family_key": str(meta.get("family_key") or ""),
        }
        if wid not in by_wid:
            wid_meta = wid_to_paper_meta.get(wid) or {"title": "", "domains": ""}
            by_wid[wid] = {
                "wid": wid,
                "title": wid_meta.get("title") or "",
                "year": wid_meta.get("year"),
                "domains": wid_meta.get("domains") or "",
                "paper_score": 0.0,
                "hits": [],
            }
        old_hits = list(by_wid[wid].get("hits") or [])
        old_terms = [str(h.get("term") or h.get("vid") or "") for h in old_hits if isinstance(h, dict)]
        by_wid[wid]["paper_score"] = float(by_wid[wid]["paper_score"]) + float(term_contrib)
        by_wid[wid]["hits"] = _merge_hits(by_wid[wid].get("hits") or [], [hit])
        new_hits = list(by_wid[wid].get("hits") or [])
        new_terms = [str(h.get("term") or h.get("vid") or "") for h in new_hits if isinstance(h, dict)]
        # 小样本调试：默认关；避免每条 wid 刷屏
        if STAGE4_PAPER_MAP_WRITE_VERBOSE:
            _lp(
                f"[Stage4 paper_map write] wid='{wid}' "
                f"incoming_term='{str(hit.get('term') or hit.get('vid') or '')}' "
                f"old_hit_count={len(old_hits)} new_hit_count={len(new_hits)} "
                f"old_terms={old_terms} new_terms={new_terms}"
            )

    # ---------- wid 级：三级领域共识软乘子（词 vocabulary_topic_stats × JD 画像；论文无需三级领域）----------
    jd_src = getattr(getattr(recall, "_last_stage1_result", None), "jd_profile", None) or {}
    jd_topic_profile = _jd_topic_profile_from_stage1(jd_src)
    _audit_jd_topic_profile(jd_topic_profile, audit_print=_label_path_stdout)
    has_jd_hier = any(jd_topic_profile.get(k) for k in ("field_dist", "subfield_dist", "topic_dist"))
    all_hit_vids: Set[int] = set()
    for _rec0 in by_wid.values():
        for _h in (_rec0.get("hits") or []):
            if not isinstance(_h, dict):
                continue
            try:
                all_hit_vids.add(int(_h.get("vid")))
            except (TypeError, ValueError):
                pass
    topic_load_tids: Set[int] = set(int(x) for x in v_ids) | all_hit_vids
    topic_map, _ = _stage1_anchors._batch_load_vocabulary_stats_for_tids(recall, topic_load_tids)
    term_topic_meta = {
        int(vid): _build_term_topic_meta_from_row(topic_map.get(int(vid))) for vid in topic_load_tids
    }

    if STAGE4_TOPIC_META_COVERAGE_AUDIT:
        _lp("\n[Stage4 topic-meta coverage audit]")
        for _ctid in sorted(topic_load_tids):
            _tname = str((_get_term_meta(_ctid) or {}).get("term") or _ctid)
            _row = topic_map.get(_ctid)
            _mm = term_topic_meta.get(_ctid) or {}
            _src = _mm.get("source") if _row is not None else "missing"
            if not _src:
                _src = "-"
            _lp(
                f"term={_tname!r} tid={_ctid} topic_meta_found={_row is not None} source={_src} "
                f"has_field_id={bool(_mm.get('field_id'))} has_subfield_id={bool(_mm.get('subfield_id'))} "
                f"has_topic_id={bool(_mm.get('topic_id'))} has_field_dist={bool(_mm.get('field_dist'))} "
                f"has_subfield_dist={bool(_mm.get('subfield_dist'))} has_topic_dist={bool(_mm.get('topic_dist'))}"
            )

    def _print_hier_line(
        _tag: str,
        _i: int,
        _wid: str,
        _rec: Dict[str, Any],
        _bef: float,
        _aft: float,
        _bon: float,
        _det: Dict[str, Any],
        _hits: List[Dict[str, Any]],
    ) -> None:
        _terms = [str(h.get("term") or h.get("vid") or "") for h in _hits if isinstance(h, dict)]
        _lp(
            f"{_tag} #{_i} pid={_wid!r} title={(str(_rec.get('title') or ''))[:80]!r} "
            f"hit_terms={_terms} "
            f"field_cons={_det.get('field_cons')} subfield_cons={_det.get('subfield_cons')} "
            f"topic_cons={_det.get('topic_cons')} hierarchy_consensus={_det.get('hierarchy_consensus')} "
            f"hierarchy_bonus={_bon:.4f} paper_score_before={_bef:.4f} paper_score_after={_aft:.4f}"
        )

    hierarchy_audit_rows: List[tuple] = []
    if STAGE4_HIERARCHY_CONSENSUS_ENABLED and has_jd_hier:
        # 批注：先收集本批 wid 的 raw consensus，再取中位数；bonus = clip(1 + β(cons−median), …)，避免 JD 重叠整体偏小时全员挤在 ~0.82。
        _pre: List[tuple] = []
        for _wid, _rec in by_wid.items():
            _before = float(_rec.get("paper_score") or 0.0)
            _hits = list(_rec.get("hits") or [])
            _cons, _det0 = compute_hierarchy_consensus_for_paper(_hits, term_topic_meta, jd_topic_profile)
            _pre.append((_wid, _rec, _before, _hits, float(_cons), dict(_det0)))
        _consensus_list = [t[4] for t in _pre]
        _median_c = float(np.median(np.array(_consensus_list, dtype=np.float64))) if _consensus_list else 0.0
        dist_rows: List[Dict[str, Any]] = []
        for _wid, _rec, _before, _hits, _cons, _det0 in _pre:
            _delta = float(_cons) - _median_c
            _raw_bonus = _hierarchy_bonus_from_delta(
                _delta,
                STAGE4_HIERARCHY_BONUS_BETA,
                STAGE4_HIERARCHY_BONUS_CLIP_LOW,
                STAGE4_HIERARCHY_BONUS_CLIP_HIGH,
            )
            # 批注：wid 级 consensus 已混合多词；仅当 strong_main_axis_core 的 hit 加权占比够高才保留 >1 正向 bonus，
            # 避免 bonus_core（如 RL）凭 topic 重合在组级再被抬高。
            _bonus, _tier_gate, _tier_extra = _clip_hierarchy_bonus_by_main_axis_mass(
                _raw_bonus, _hits, _get_term_meta
            )
            _after = _before * _bonus
            _det = {
                **_det0,
                **_tier_extra,
                "median_consensus": round(_median_c, 6),
                "beta": STAGE4_HIERARCHY_BONUS_BETA,
                "delta_vs_median": round(_delta, 6),
                "hierarchy_bonus_raw": round(_raw_bonus, 6),
                "hierarchy_tier_boost_gate": _tier_gate,
                "hierarchy_bonus": round(_bonus, 6),
            }
            _rec["paper_score_base"] = float(_before)
            _rec["paper_score"] = _after
            _rec["hierarchy_consensus_bonus"] = _bonus
            _rec["hierarchy_consensus_detail"] = _det
            hierarchy_audit_rows.append((_before, _after, _wid, _rec, _bonus, _det, _hits))
            dist_rows.append(
                {
                    "wid": _wid,
                    "title": _rec.get("title") or "",
                    "consensus": _cons,
                    "bonus": _bonus,
                    "delta_vs_median": _delta,
                    "paper_score_before": _before,
                }
            )
        _audit_hierarchy_bonus_distribution(
            dist_rows, STAGE4_HIERARCHY_DISTRIBUTION_TOP_N, audit_print=_label_path_stdout
        )
        _print_stage4_hierarchy_bonus_term_group_audit(
            by_wid, _get_term_meta, audit_print=_label_path_stdout
        )
        _by_delta = sorted(hierarchy_audit_rows, key=lambda x: -abs(x[1] - x[0]))[
            : max(0, STAGE4_HIERARCHY_AUDIT_TOP_BY_DELTA)
        ]
        # 批注：与 top_by_abs_score_delta 信息高度重叠时只保留后者，缩短日志。
        _lp("\n[Stage4 hierarchy consensus audit] top_by_abs_score_delta")
        for _i, (_bef, _aft, _wid, _rec, _bon, _det, _hits) in enumerate(_by_delta, 1):
            _print_hier_line("delta", _i, _wid, _rec, _bef, _aft, _bon, _det, _hits)
    else:
        for _rec in by_wid.values():
            _rec["hierarchy_consensus_bonus"] = 1.0
            _rec["hierarchy_consensus_detail"] = {"reason": "disabled_or_no_jd_profile"}
            _rec["paper_score_base"] = float(_rec.get("paper_score") or 0.0)

    # 全局按 paper_score 排序，取前 GLOBAL_PAPER_LIMIT（已乘 hierarchy_consensus_bonus）
    sorted_wids = sorted(
        by_wid.keys(),
        key=lambda w: -float((by_wid[w] or {}).get("paper_score") or 0.0),
    )[:GLOBAL_PAPER_LIMIT]
    selected_wids_set = set(sorted_wids)

    for _w, _rec in by_wid.items():
        _materialize_stage4_paper_explanation_fields(
            str(_w),
            _rec,
            get_term_meta=_get_term_meta,
            global_pool_kept=str(_w) in selected_wids_set,
        )
    _print_stage4_paper_explanation_summary(by_wid, selected_wids_set, audit_print=_label_path_stdout)

    for __w, __rec in by_wid.items():
        _compute_stage4_ltr_like_paper_rerank_score(__rec)
    _print_stage4_paper_ltr_rerank_summary(by_wid, selected_wids_set, audit_print=_label_path_stdout)

    for __w, __rec in by_wid.items():
        _compute_stage4_mainline_protection_factor(__rec)
    _print_stage4_paper_mainline_protection_summary(by_wid, selected_wids_set, audit_print=_label_path_stdout)

    # -------- final_unique / global_cap_cut 统计（按 term 级唯一 wid）--------
    for vid, capped_wids in term_capped_unique_wids.items():
        final_unique = len(capped_wids.intersection(selected_wids_set))
        term_funnel_counts[vid]["final_unique"] = final_unique
        term_reject_reason_counts[vid]["global_cap_cut"] = len(capped_wids) - final_unique
    t2 = time.perf_counter()
    sub_ms["python_agg"] = (t2 - t1) * 1000.0
    # -------------------------
    # Stage4 诊断打印（只打印“折叠明显”的 term）
    # -------------------------
    # retain_ratio = final_unique / cypher_raw，按最糟的少量 term 输出
    retain_list: List[tuple] = []
    for vid, f in term_funnel_counts.items():
        cypher_raw = f.get("cypher_raw", 0) or 0
        if cypher_raw <= 0:
            continue
        final_unique = f.get("final_unique", 0) or 0
        retain_ratio = final_unique / float(cypher_raw) if cypher_raw > 0 else 0.0
        retain_list.append((retain_ratio, -cypher_raw, vid))

    retain_list.sort()
    focus_vids = [vid for (_, __, vid) in retain_list[:6] if term_funnel_counts[vid].get("cypher_raw", 0) >= 10]

    if focus_vids:
        for vid in focus_vids:
            meta = _get_term_meta(vid)
            term_name = str(meta.get("term") or meta.get("anchor") or meta.get("parent_primary") or vid)
            role = (term_retrieval_roles.get(vid) or term_retrieval_roles.get(str(vid)) or "").strip().lower()

            f = term_funnel_counts[vid]
            r = term_reject_reason_counts[vid]

            _lp(
                f"[Stage4 term funnel] term='{term_name}' role='{role}' "
                f"cypher_raw={f['cypher_raw']} "
                f"after_year_filter={f['after_year_filter']} "
                f"after_basic_meta={f['after_basic_meta']} "
                f"after_grounding_gate={f['after_grounding_gate']} "
                f"after_offtopic_penalty_sort={f['after_offtopic_penalty_sort']} "
                f"after_local_cap={f['after_local_cap']} "
                f"final_unique={f['final_unique']}"
            )

            _lp(
                f"[Stage4 reject reason summary] term='{term_name}' "
                f"low_grounding={r['low_grounding']} "
                f"off_topic_penalty_too_low={r['off_topic_penalty_too_low']} "
                f"duplicate_dropped={r['duplicate_dropped']} "
                f"local_cap_cut={r['local_cap_cut']} "
                f"global_cap_cut={r['global_cap_cut']}"
            )

            # reject samples：low_grounding -> local_cap_cut -> global_cap_cut 依次补齐到 3
            reject_samples: List[Dict[str, Any]] = []
            reject_samples.extend(term_low_grounding_samples.get(vid, [])[:3])
            if len(reject_samples) < 3:
                reject_samples.extend(term_local_cap_cut_samples.get(vid, [])[: 3 - len(reject_samples)])

            if len(reject_samples) < 3:
                capped_wids = term_capped_unique_wids.get(vid, set()) or set()
                diff_wids = list(capped_wids - selected_wids_set)
                diff_wids = diff_wids[: 3 - len(reject_samples)]
                for w in diff_wids:
                    meta2 = wid_to_paper_meta.get(w) or {"title": "", "domains": ""}
                    g = _compute_grounding_score(vid, meta2.get("title") or "", meta2.get("domains") or "")
                    reject_samples.append(
                        {
                            "wid": w,
                            "title": meta2.get("title") or "",
                            "domains": meta2.get("domains") or "",
                            "reason": "global_cap_cut",
                            "grounding": g.get("grounding", 0.0),
                            "penalty": g.get("off_topic_penalty", 1.0),
                        }
                    )

            if reject_samples:
                for s in reject_samples[:3]:
                    wid = s.get("wid")
                    title = s.get("title") or ""
                    reason = s.get("reason") or ""
                    grounding_val = s.get("grounding", None)
                    penalty_val = s.get("penalty", None)
                    if grounding_val is None or penalty_val is None:
                        g = _compute_grounding_score(vid, title, s.get("domains") or "")
                        grounding_val = g.get("grounding", 0.0)
                        penalty_val = g.get("off_topic_penalty", 1.0)
                    grounding = float(grounding_val or 0.0)
                    penalty = float(penalty_val or 1.0)
                    _lp(
                        f"[Stage4 reject samples] term='{term_name}' "
                        f"pid='{wid}' title={title[:80]!r} reason='{reason}' "
                        f"grounding={grounding:.3f} penalty={penalty:.3f}"
                    )

            # kept papers：最终入选（selected_wids_set）里，取对该 term 贡献的 top3（按 paper_score）
            capped_wids = term_capped_unique_wids.get(vid, set()) or set()
            kept_wids = list(capped_wids.intersection(selected_wids_set))
            kept_wids.sort(key=lambda w: -float((by_wid.get(w) or {}).get("paper_score") or 0.0))
            kept_wids = kept_wids[:3]
            for i, w in enumerate(kept_wids, start=1):
                meta3 = wid_to_paper_meta.get(w) or {"title": "", "domains": ""}
                g = _compute_grounding_score(vid, meta3.get("title") or "", meta3.get("domains") or "")
                final_score = float((by_wid.get(w) or {}).get("paper_score") or 0.0)
                _lp(
                    f"[Stage4 kept papers] term='{term_name}' rank={i} pid='{w}' "
                    f"grounding={float(g.get('grounding') or 0.0):.3f} "
                    f"penalty={float(g.get('off_topic_penalty') or 1.0):.3f} "
                    f"final_paper_score={final_score:.3f} title={meta3.get('title')[:80]!r}"
                )

            # rejected papers sample：再取一些未入选且可用的样本（最多 3 条）
            rejected_wids: List[str] = []
            rejected_wids.extend([s.get("wid") for s in reject_samples if s.get("wid")][:3])
            if len(rejected_wids) < 3:
                extra = list((capped_wids - selected_wids_set) or [])[: 3 - len(rejected_wids)]
                rejected_wids.extend(extra)
            rejected_wids = [w for w in rejected_wids if w is not None][:3]

            for w in rejected_wids:
                meta3 = wid_to_paper_meta.get(w) or {"title": "", "domains": ""}
                g = _compute_grounding_score(vid, meta3.get("title") or "", meta3.get("domains") or "")
                final_score = float((by_wid.get(w) or {}).get("paper_score") or 0.0)
                # 是否 low_grounding：看 grounding 是否低于阈值
                gg = float(g.get("grounding") or 0.0)
                reason = "low_grounding" if gg < PRIMARY_GROUNDING_MIN else "pruned_after_terms"
                _lp(
                    f"[Stage4 rejected papers] term='{term_name}' pid='{w}' reason='{reason}' "
                    f"grounding={float(g.get('grounding') or 0.0):.3f} "
                    f"penalty={float(g.get('off_topic_penalty') or 1.0):.3f} "
                    f"final_paper_score={final_score:.3f} title={meta3.get('title')[:80]!r}"
                )

    # 精准审计块 1/3：
    # [Stage4 pre-gate paper score audit] 收集于 grounding 门之前，行数 = pre_cap 前的 scored_rows，不是 cap 后 kept。
    _lp("\n[Stage4 pre-gate paper score audit]")
    jd_audit_primary_vids: Set[int] = set()
    for _va in v_ids:
        _ma = _get_term_meta(_va)
        _rra = (
            str(_ma.get("retrieval_role") or (term_retrieval_roles or {}).get(_va) or "").strip().lower()
        )
        if _rra == "paper_primary":
            jd_audit_primary_vids.add(_va)
    _primary_ranked = sorted(jd_audit_primary_vids, key=_term_score, reverse=True)
    jd_audit_primary_detail_vids = set(_primary_ranked[: max(0, STAGE4_JD_AUDIT_PRIMARY_DETAIL_K)])
    for vid in v_ids:
        meta = _get_term_meta(vid)
        term_name = str(meta.get("term") or meta.get("anchor") or meta.get("parent_primary") or vid)
        rows_audit = term_kept_paper_audit.get(vid, [])
        rows_audit.sort(key=lambda x: float(x.get("final_paper_score") or 0.0), reverse=True)
        detail_jd = bool(STAGE4_JD_AUDIT_FULL or vid in jd_audit_primary_detail_vids)
        _lp(
            f"term='{term_name}' pre_gate_scored_rows={len(rows_audit)} "
            f"jd_audit_detail={detail_jd}"
        )
        if not detail_jd:
            continue
        for i, p in enumerate(rows_audit[:STAGE4_JD_AUDIT_TOP_K], 1):
            _lp(
                f"[Stage4 jd audit] #{i} term='{term_name}' pid='{p.get('paper_id')}' "
                f"tg={float(p.get('term_grounding') or 0.0):.3f} "
                f"jd={float(p.get('jd_align') or 0.5):.3f} "
                f"hybrid={float(p.get('hybrid_grounding') or p.get('grounding') or 0.0):.3f} "
                f"grounding={float(p.get('grounding') or 0.0):.3f} "
                f"penalty={float(p.get('offtopic_penalty') or 1.0):.3f} "
                f"paper_factor={float(p.get('paper_factor') or 1.0):.3f} "
                f"year_factor={float(p.get('year_factor') or 1.0):.3f} "
                f"domain_bonus={float(p.get('domain_bonus') or 1.0):.3f} "
                f"idf_weight={float(p.get('idf_weight') or 0.0):.3f} "
                f"final={float(p.get('final_paper_score') or 0.0):.3f} "
                f"title={str(p.get('title') or '')[:80]!r}"
            )

    if not sorted_wids:
        sub_ms["total"] = (time.perf_counter() - t0) * 1000.0
        _save_sub(sub_ms)
        return []

    # 审计 1：确认 Stage4 内部 wid 聚合后确实存在 multi-hit 论文
    _lp("\n[Stage4 merged wid multi-hit audit]")
    stage4_multi_hit_rows: List[tuple] = []
    for wid, rec in by_wid.items():
        hits = rec.get("hits") or []
        if len(hits) >= 2:
            terms = [str(h.get("term") or h.get("vid") or "") for h in hits if isinstance(h, dict)]
            stage4_multi_hit_rows.append((wid, rec.get("title") or "", terms))
    _lp(f"multi_hit_papers={len(stage4_multi_hit_rows)}")
    for wid, title, terms in stage4_multi_hit_rows[:20]:
        _lp(f"  wid={wid} hit_terms={terms} title='{(title or '')[:100]}'")
    # 断点 4A：merged paper_map 细节（每条多 hit 明细）
    _lp("\n[Stage4 merged wid multi-hit detail]")
    for wid, rec in by_wid.items():
        hits = rec.get("hits") or []
        if len(hits) < 2:
            continue
        terms = [str(h.get("term") or h.get("vid") or "") for h in hits if isinstance(h, dict)]
        _lp(
            f"wid='{wid}' hit_count={len(hits)} terms={terms} "
            f"title='{str(rec.get('title') or '')[:100]}'"
        )

    # ---------- 第二层：按 wid 查作者与论文元数据，按 aid 聚合为 author_papers_list ----------
    params2 = {"wids": sorted_wids}
    cypher_layer2 = """
    MATCH (w:Work) WHERE w.id IN $wids
    MATCH (w)<-[r:AUTHORED]-(a:Author)
    WITH a.id AS aid, w.id AS wid, r.pos_weight AS weight, w.title AS title, w.year AS year, w.domain_ids AS domains
    WITH aid, collect({wid: wid, weight: weight, title: title, year: year, domains: domains}) AS papers
    RETURN aid, papers
    """
    t3 = time.perf_counter()
    try:
        cursor2 = recall.graph.run(cypher_layer2, **params2)
        author_rows = list(cursor2)
    except Exception:
        sub_ms["cypher2"] = (time.perf_counter() - t3) * 1000.0
        sub_ms["total"] = (time.perf_counter() - t0) * 1000.0
        _save_sub(sub_ms)
        return []

    t4 = time.perf_counter()
    sub_ms["cypher2"] = (t4 - t3) * 1000.0

    # 为每篇 paper 挂上 Stage4 算好的 hits 与 score（供 Stage5 / debug 使用）
    wid_to_hits_and_score = {
        str(wid): (
            list((rec or {}).get("hits") or []),
            float(
                (rec or {}).get("paper_score_protected")
                if (rec or {}).get("paper_score_protected") is not None
                else (rec or {}).get("paper_rerank_score")
                if (rec or {}).get("paper_rerank_score") is not None
                else (rec or {}).get("paper_score") or 0.0
            ),
            (rec or {}).get("title"),
            (rec or {}).get("domains"),
            (rec or {}).get("year"),
        )
        for wid, rec in by_wid.items()
    }

    out: List[Dict[str, Any]] = []
    for rec in author_rows:
        aid = rec.get("aid")
        papers_raw = rec.get("papers") or []
        papers = []
        for p in papers_raw:
            wid = p.get("wid")
            if wid is None:
                continue
            wid_s = str(wid)
            hits, score, title_s4, domains_s4, year_s4 = wid_to_hits_and_score.get(wid_s, ([], 0.0, None, None, None))
            _br = by_wid.get(wid_s) or {}
            _paper_row: Dict[str, Any] = {
                "wid": wid_s,
                "hits": hits,
                "weight": p.get("weight"),
                "title": title_s4 if title_s4 is not None else p.get("title"),
                "year": year_s4 if year_s4 is not None else p.get("year"),
                "domains": domains_s4 if domains_s4 is not None else p.get("domains"),
                "score": score,
            }
            for _ek in _STAGE4_PAPER_EXPLANATION_KEYS:
                if _ek in _br:
                    _paper_row[_ek] = _br[_ek]
            for _lk in _STAGE4_PAPER_LTR_KEYS:
                if _lk in _br:
                    _paper_row[_lk] = _br[_lk]
            for _pk in _STAGE4_PAPER_MAINLINE_PROT_KEYS:
                if _pk in _br:
                    _paper_row[_pk] = _br[_pk]
            papers.append(_paper_row)
        if aid is not None and papers:
            out.append({
                "aid": str(aid),
                "papers": papers,
            })

    for _auth_rec in out:
        _apapers = _auth_rec.get("papers") or []
        _aid_s = str(_auth_rec.get("aid") or "")
        _asum = _summarize_author_payload_evidence(_aid_s, _apapers, _auth_rec)
        _classify_author_support_profile(_asum)
        _auth_rec["author_payload_summary"] = _asum
        _compute_stage5_ltr_like_author_rerank_score(_auth_rec, _asum)
        _compute_stage5_mainline_protection_factor(_auth_rec, _asum)

    _pre_top20_aids = [
        str(r.get("aid"))
        for r in sorted(
            out,
            key=lambda r: -float((r.get("author_payload_summary") or {}).get("top_paper_score") or 0.0),
        )[:20]
    ]
    _pre_prot_top20_aids = [
        str(r.get("aid"))
        for r in sorted(out, key=lambda r: -float(r.get("final_score_reranked") or 0.0))[:20]
    ]
    out.sort(key=lambda r: -float(r.get("final_score_protected") or 0.0))

    _print_stage5_author_payload_summary(out, audit_print=_label_path_stdout)
    _print_stage5_author_ltr_rerank_summary(out, _pre_top20_aids, audit_print=_label_path_stdout)
    _print_stage5_author_mainline_protection_summary(out, _pre_prot_top20_aids, audit_print=_label_path_stdout)

    sub_ms["build_list"] = (time.perf_counter() - t4) * 1000.0
    sub_ms["total"] = (time.perf_counter() - t0) * 1000.0
    _save_sub(sub_ms)

    # 审计 2：multi-hit 计数；逐条明细默认关闭（污染已在 term 侧定位时优先看 Stage3）
    _lp("\n[Stage4 author payload audit]")
    payload_multi_cnt = 0
    for rec in out:
        for p in rec.get("papers") or []:
            hits = p.get("hits") or []
            if len(hits) >= 2:
                payload_multi_cnt += 1
                if STAGE4_AUTHOR_PAYLOAD_AUDIT_VERBOSE:
                    aid = rec.get("aid")
                    terms = [str(h.get("term") or h.get("vid") or "") for h in hits if isinstance(h, dict)]
                    _lp(f"aid={aid} wid={p.get('wid')} hit_terms={terms}")
    _lp(f"author_payload_multi_hit_papers={payload_multi_cnt}")
    if STAGE4_AUTHOR_PAYLOAD_AUDIT_VERBOSE:
        _lp("\n[Stage4 author payload multi-hit detail]")
        for rec in out:
            aid = rec.get("aid")
            for p in rec.get("papers") or []:
                hits = p.get("hits") or []
                if len(hits) < 2:
                    continue
                terms = [str(h.get("term") or h.get("vid") or "") for h in hits if isinstance(h, dict)]
                _lp(f"aid='{aid}' wid='{p.get('wid')}' hit_count={len(hits)} terms={terms}")
    return out


# --- Step8: Stage4 prep bridge (term→paper selection from Stage3 ranked_terms; POST-LAYER) ---
from src.core.recall.label_pipeline.stage4_prep_bridge import (  # noqa: E402
    prepare_stage4_terms_from_stage3,
    select_terms_for_paper_recall,
)
