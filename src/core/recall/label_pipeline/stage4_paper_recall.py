import time
import json
import math
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
# Step2：evidence quality 混入 v2 分；True=按 paper_final_score_v2 重排 sorted_wids 与 term kept 诊断；False=仅审计
STAGE4_ENABLE_EVIDENCE_SCORE_MIGRATION = True
STAGE4_EVIDENCE_MIGRATION_WEIGHT_OLD = 0.85
STAGE4_EVIDENCE_MIGRATION_WEIGHT_EVIDENCE = 0.15
STAGE4_EVIDENCE_MIGRATION_TOPK_CHANGED = 15
# Step2.2：仅对「低旧分 + single_hit_side + 无主轴」的 fringe 收窄 evidence 混入，抑制 v2 名次误抬升
STAGE4_FRINGE_V2_LOCAL_WEIGHT_OLD = 0.93
STAGE4_FRINGE_V2_LOCAL_WEIGHT_EVIDENCE = 0.07
STAGE4_FRINGE_V2_LOW_OLD_THRESHOLD = 0.03
# Step2 收口：在 global kept 集合确定前，用 compete 分（v2 + 极窄 fringe 降权 / 主轴轻加成）重选 GLOBAL_PAPER_LIMIT
STAGE4_KEEP_COMPETE_ENABLED = True
STAGE4_PREKEPTFRINGE_GUARD_FACTOR = 0.88
STAGE4_PREKEPT_MAINLINE_KEEP_BONUS = 1.035
# keep 入口竞争：在 local_cap 截断之外，再纳入「已过 grounding、未进 local cap」的 (vid,wid) 行。
# <=0：纳入 local_cap 之后 remainder 全部（仅受 grounding 与 triples 长度限制）；>0：最多再纳入这么多条。
# 使 by_wid 大于「仅 kept 前 local_cap」时的唯一 wid 覆盖，避免 compete 候选与 global baseline 完全同构。
STAGE4_KEEP_COMPETE_EXTRA_PER_TERM = int(os.environ.get("STAGE4_KEEP_COMPETE_EXTRA_PER_TERM", "0"))
# Paper-topic-Paper 轻量社区：kept 竞争位点对孤立 term-only fringe 额外抑制、对 topic-cohesive 极轻加成（不改动 v2 主干公式）
STAGE4_COMMUNITY_ISOLATED_FRINGE_FACTOR = float(os.environ.get("STAGE4_COMMUNITY_ISOLATED_FRINGE_FACTOR", "0.90"))
STAGE4_COMMUNITY_COHESIVE_KEEP_BONUS = float(os.environ.get("STAGE4_COMMUNITY_COHESIVE_KEEP_BONUS", "1.05"))
# 收窄 JD overlap：field 单层高阈值；subfield/topic 单独阈值（避免 53/53 宽场沾边即 True）
STAGE4_JD_OVERLAP_TOPIC_MIN = float(os.environ.get("STAGE4_JD_OVERLAP_TOPIC_MIN", "0.032"))
STAGE4_JD_OVERLAP_SUBFIELD_MIN = float(os.environ.get("STAGE4_JD_OVERLAP_SUBFIELD_MIN", "0.042"))
STAGE4_JD_OVERLAP_FIELD_MIN = float(os.environ.get("STAGE4_JD_OVERLAP_FIELD_MIN", "0.14"))
STAGE4_JD_OVERLAP_FIELD_REQUIRES_SUBTOP_MIN = float(
    os.environ.get("STAGE4_JD_OVERLAP_FIELD_REQUIRES_SUBTOP_MIN", "0.018")
)
# 兼容旧名：仅作日志/对照，强判定用上面分项阈值
STAGE4_JD_TOPIC_OVERLAP_FLAG_MIN = float(os.environ.get("STAGE4_JD_TOPIC_OVERLAP_FLAG_MIN", "0.032"))
# 池内 neighborhood：频次阈值抬高，且以 subfield/topic 为主（field 高频易吸满池）
STAGE4_TOPIC_NEIGHBORHOOD_MIN_FREQ = int(os.environ.get("STAGE4_TOPIC_NEIGHBORHOOD_MIN_FREQ", "4"))
STAGE4_TOPIC_NEIGHBORHOOD_POOL_FREQ_FRAC = float(os.environ.get("STAGE4_TOPIC_NEIGHBORHOOD_POOL_FREQ_FRAC", "0.12"))
STAGE4_TOPIC_NEIGHBORHOOD_INCLUDE_FIELD_POOL = os.environ.get(
    "STAGE4_TOPIC_NEIGHBORHOOD_INCLUDE_FIELD_POOL", ""
).strip().lower() in ("1", "true", "yes")
# 邻居：禁止仅靠共享宽 field；需 subfield+topic 同签 或 topic mix 高重叠等
STAGE4_NEIGHBOR_TOPIC_MIX_MIN = float(os.environ.get("STAGE4_NEIGHBOR_TOPIC_MIX_MIN", "0.22"))
STAGE4_NEIGHBOR_SUBFIELD_MIX_MIN = float(os.environ.get("STAGE4_NEIGHBOR_SUBFIELD_MIX_MIN", "0.17"))
STAGE4_NEIGHBOR_TOPIC_MIX_PAIR_MIN = float(os.environ.get("STAGE4_NEIGHBOR_TOPIC_MIX_PAIR_MIN", "0.11"))
STAGE4_COMMUNITY_FULL_PAIRWISE_MAX_POOL = int(os.environ.get("STAGE4_COMMUNITY_FULL_PAIRWISE_MAX_POOL", "650"))
# topic_cohesive：至少 2 个强信号（邻居≥2 / 严格 JD / 严格 nh / hcons / community_score）
STAGE4_COHESIVE_STRONG_SIGNALS_MIN = int(os.environ.get("STAGE4_COHESIVE_STRONG_SIGNALS_MIN", "2"))
STAGE4_COHESIVE_NEIGHBOR_MIN_FOR_SIGNAL = int(os.environ.get("STAGE4_COHESIVE_NEIGHBOR_MIN_FOR_SIGNAL", "2"))
STAGE4_COHESIVE_HCONS_MIN = float(os.environ.get("STAGE4_COHESIVE_HCONS_MIN", "0.038"))
STAGE4_COHESIVE_DET_BLEND_MIN = float(os.environ.get("STAGE4_COHESIVE_DET_BLEND_MIN", "0.048"))
STAGE4_WEAK_NEIGHBOR_EXACT = int(os.environ.get("STAGE4_WEAK_NEIGHBOR_EXACT", "1"))
# Step2：mainline / multi-hit 解释层审计（修正口径后打印 top multi-hit；默认 ≥8 条）
STAGE4_MAINLINE_TERM_AUDIT_TOP_MULTI_HIT = int(os.environ.get("STAGE4_MAINLINE_TERM_AUDIT_TOP_MULTI_HIT", "12"))


def _stage4_top_keys_from_dist(d: Dict[str, float], n: int = 5) -> List[str]:
    if not d:
        return []
    items = sorted(d.items(), key=lambda kv: -float(kv[1]))[:n]
    return [str(k) for k, _ in items if k is not None and str(k).strip()]


def _stage4_jd_strict_neighborhood_key_set(jd_topic_profile: Dict[str, Dict[str, float]]) -> Set[str]:
    """收窄：以 JD 的 subfield/topic top keys 为主，field 仅少量 top，避免宽场键吸满池。"""
    out: Set[str] = set()
    jd_f = jd_topic_profile.get("field_dist") or {}
    jd_s = jd_topic_profile.get("subfield_dist") or {}
    jd_t = jd_topic_profile.get("topic_dist") or {}
    for k in _stage4_top_keys_from_dist(jd_s, 4):
        out.add("s:" + k)
    for k in _stage4_top_keys_from_dist(jd_t, 4):
        out.add("t:" + k)
    for k in _stage4_top_keys_from_dist(jd_f, 2):
        out.add("f:" + k)
    return out


def _stage4_jd_profile_overlap_flag_strict(fc: float, sc: float, tc: float) -> bool:
    """field 单命中需更高且伴随 sub/topic，否则仅靠 subfield 或 topic 达阈才算 overlap。"""
    if tc >= STAGE4_JD_OVERLAP_TOPIC_MIN:
        return True
    if sc >= STAGE4_JD_OVERLAP_SUBFIELD_MIN:
        return True
    if fc >= STAGE4_JD_OVERLAP_FIELD_MIN and max(sc, tc) >= STAGE4_JD_OVERLAP_FIELD_REQUIRES_SUBTOP_MIN:
        return True
    return False


def _stage4_paper_weighted_topic_mix(
    paper_hits: List[Dict[str, Any]],
    term_topic_meta: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float]:
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
        return {}, {}, {}, 0.0

    def _norm_mix(d: Dict[str, float]) -> Dict[str, float]:
        s = sum(d.values())
        return {k: v / s for k, v in d.items()} if s > 0 else {}

    return _norm_mix(field_acc), _norm_mix(subfield_acc), _norm_mix(topic_acc), float(total_w)


def _stage4_argmax_topic_key(dist: Dict[str, float]) -> Optional[str]:
    if not dist:
        return None
    return str(max(dist.items(), key=lambda kv: float(kv[1]))[0]).strip() or None


def _stage4_paper_signature_keys(fk: Optional[str], sk: Optional[str], tk: Optional[str]) -> Set[str]:
    s: Set[str] = set()
    if fk:
        s.add("f:" + str(fk))
    if sk:
        s.add("s:" + str(sk))
    if tk:
        s.add("t:" + str(tk))
    return s


def _stage4_is_topic_neighbor(
    sig_a: Tuple[Optional[str], Optional[str], Optional[str], float, Dict[str, float], Dict[str, float], Dict[str, float]],
    sig_b: Tuple[Optional[str], Optional[str], Optional[str], float, Dict[str, float], Dict[str, float], Dict[str, float]],
) -> bool:
    """
    收窄「真邻居」：禁止仅靠共享宽 field。
    - 同 subfield+topic 的 argmax；或
    - topic mix 高重叠；或
    - subfield+topic mix 同时达中等阈值。
    """
    _fk, sk, tk, tw, mf, ms, mt = sig_a
    _ofk, osk, otk, otw, omf, oms, omt = sig_b
    if tw <= 0 or otw <= 0:
        return False
    # 同 subfield+topic 仍须有 topic mix 可验证的重叠，避免元数据 argmax 塌缩成全团
    if sk and osk and tk and otk and sk == osk and tk == otk:
        tov_sig = _dist_overlap(mt, omt)
        if tov_sig >= STAGE4_NEIGHBOR_TOPIC_MIX_PAIR_MIN:
            return True
    tov = _dist_overlap(mt, omt)
    if tov >= STAGE4_NEIGHBOR_TOPIC_MIX_MIN:
        return True
    sov = _dist_overlap(ms, oms)
    if sov >= STAGE4_NEIGHBOR_SUBFIELD_MIX_MIN and tov >= STAGE4_NEIGHBOR_TOPIC_MIX_PAIR_MIN:
        return True
    return False


def _stage4_compute_paper_topic_neighbor_counts(
    wids: List[str],
    wid_sigs: Dict[str, Tuple[Optional[str], Optional[str], Optional[str], float, Dict[str, float], Dict[str, float], Dict[str, float]]],
) -> Dict[str, int]:
    n_pool = len(wids)
    out: Dict[str, int] = {w: 0 for w in wids}
    if n_pool <= 1:
        return out
    if n_pool <= STAGE4_COMMUNITY_FULL_PAIRWISE_MAX_POOL:
        for i, wa in enumerate(wids):
            sa = wid_sigs.get(wa)
            if sa is None:
                continue
            for wb in wids[i + 1 :]:
                sb = wid_sigs.get(wb)
                if sb is None:
                    continue
                if _stage4_is_topic_neighbor(sa, sb):
                    out[wa] += 1
                    out[wb] += 1
        return out
    # 大池：仅用 subfield+topic 复合签名桶计数，不用单键并集（避免 clique）
    ck_to_wids: Dict[str, Set[str]] = defaultdict(set)
    per_wid_ck: Dict[str, str] = {}
    for w in wids:
        tup = wid_sigs.get(w)
        if tup is None:
            per_wid_ck[w] = ""
            continue
        _fk, sk, tk, _tw, _mf, _ms, _mt = tup
        if sk and tk:
            ck = f"s:{sk}|t:{tk}"
            per_wid_ck[w] = ck
            ck_to_wids[ck].add(w)
        else:
            per_wid_ck[w] = ""
    for w in wids:
        ck = per_wid_ck.get(w) or ""
        if not ck:
            out[w] = 0
            continue
        neigh = set(ck_to_wids.get(ck, set()))
        neigh.discard(w)
        out[w] = len(neigh)
    return out


def _stage4_compute_paper_topic_community_score(
    rec: Dict[str, Any],
    *,
    neighbor_count: int,
    in_neighborhood: bool,
) -> float:
    """连续分：邻居、sub/topic 层 overlap 为主，field 轻权；无强支撑时明显偏低。"""
    det = rec.get("hierarchy_consensus_detail") or {}
    try:
        fc = float(det.get("field_cons") or 0.0)
        sc = float(det.get("subfield_cons") or 0.0)
        tc = float(det.get("topic_cons") or 0.0)
    except (TypeError, ValueError):
        fc = sc = tc = 0.0
    try:
        hcons = float(det.get("hierarchy_consensus") or 0.0)
    except (TypeError, ValueError):
        hcons = 0.0
    hc2 = rec.get("hierarchy_consensus")
    try:
        if hc2 is not None:
            hcons = max(hcons, float(hc2))
    except (TypeError, ValueError):
        pass
    try:
        nh = max(0, int(neighbor_count))
    except (TypeError, ValueError):
        nh = 0
    w_n = 0.26 * min(1.0, nh / 12.0)
    w_st = 0.40 * (0.48 * min(1.0, tc / 0.28) + 0.52 * min(1.0, sc / 0.24))
    w_f = 0.10 * min(1.0, fc / 0.40)
    w_nh = 0.12 * (1.0 if in_neighborhood else 0.0)
    w_hc = 0.12 * min(1.0, hcons / 0.14)
    return float(min(1.0, w_n + w_st + w_f + w_nh + w_hc))


def _stage4_det_layer_support_signal(rec: Dict[str, Any]) -> bool:
    """hierarchy_consensus_detail 的 field/subfield/topic 与 JD 混合强度（不作宽 OR）。"""
    det = rec.get("hierarchy_consensus_detail") or {}
    try:
        fc = float(det.get("field_cons") or 0.0)
        sc = float(det.get("subfield_cons") or 0.0)
        tc = float(det.get("topic_cons") or 0.0)
    except (TypeError, ValueError):
        fc = sc = tc = 0.0
    blend = 0.22 * fc + 0.38 * sc + 0.40 * tc
    return bool(blend >= STAGE4_COHESIVE_DET_BLEND_MIN)


def _stage4_count_cohesion_strong_signals(rec: Dict[str, Any]) -> int:
    """强信号计数（至多 5）：邻居≥2、严格 JD、严格 neighborhood、hcons、det 混合支撑。"""
    n = 0
    try:
        nh = int(rec.get("paper_topic_neighbor_count") or 0)
    except (TypeError, ValueError):
        nh = 0
    if nh >= STAGE4_COHESIVE_NEIGHBOR_MIN_FOR_SIGNAL:
        n += 1
    if rec.get("paper_jd_topic_profile_overlap_flag"):
        n += 1
    if rec.get("paper_topic_neighborhood_hit_flag"):
        n += 1
    det = rec.get("hierarchy_consensus_detail") or {}
    try:
        hcons = float(det.get("hierarchy_consensus") or 0.0)
    except (TypeError, ValueError):
        hcons = 0.0
    hc2 = rec.get("hierarchy_consensus")
    try:
        if hc2 is not None:
            hcons = max(hcons, float(hc2))
    except (TypeError, ValueError):
        pass
    if hcons >= STAGE4_COHESIVE_HCONS_MIN:
        n += 1
    if _stage4_det_layer_support_signal(rec):
        n += 1
    return int(n)


def _stage4_assign_candidate_source_role(rec: Dict[str, Any]) -> str:
    """至少三档：cohesive 需 ≥2 个强信号；weak 为单信号或恰 1 个邻居；其余 term_only。"""
    scount = _stage4_count_cohesion_strong_signals(rec)
    try:
        nh = int(rec.get("paper_topic_neighbor_count") or 0)
    except (TypeError, ValueError):
        nh = 0
    if scount >= STAGE4_COHESIVE_STRONG_SIGNALS_MIN:
        return "topic_cohesive_candidate"
    if scount == 1 or nh == STAGE4_WEAK_NEIGHBOR_EXACT:
        return "topic_weak_candidate"
    return "term_only_candidate"


def _stage4_should_apply_isolated_fringe_community_guard(rec: Dict[str, Any]) -> bool:
    if str(rec.get("hit_quality_class") or "").strip() != "single_hit_side":
        return False
    try:
        if int(rec.get("mainline_term_count") or 0) != 0:
            return False
    except (TypeError, ValueError):
        return False
    if int(rec.get("paper_topic_neighbor_count") or 0) != 0:
        return False
    if str(rec.get("candidate_source_role") or "") != "term_only_candidate":
        return False
    return True


def _stage4_should_apply_topic_cohesive_keep_bonus(rec: Dict[str, Any]) -> bool:
    """与 source 分层对齐：仅显式 topic_cohesive_candidate 触发（不扩权到 weak / 单邻居）。"""
    if _stage4_should_apply_isolated_fringe_community_guard(rec):
        return False
    return str(rec.get("candidate_source_role") or "") == "topic_cohesive_candidate"


def _stage4_collect_topic_cohesive_candidate_wids(
    by_wid: Dict[str, Dict[str, Any]],
    term_topic_meta: Dict[int, Dict[str, Any]],
    jd_topic_profile: Dict[str, Dict[str, float]],
) -> Tuple[Set[str], Dict[str, Any]]:
    """
    以当前 candidate pool 构造收窄后的 topic neighborhood（JD subfield/topic 为主 + 池内高频 s/t），
    严格邻居与多信号计数得到三档 candidate_source_role；不改 paper_score 主干。
    """
    jd_key_set = _stage4_jd_strict_neighborhood_key_set(jd_topic_profile)
    wid_sigs: Dict[str, Tuple[Optional[str], Optional[str], Optional[str], float, Dict[str, float], Dict[str, float], Dict[str, float]]] = {}
    key_freq: Counter = Counter()
    wids = [w for w, r in by_wid.items() if isinstance(r, dict)]

    for wid in wids:
        rec = by_wid[wid]
        hits = [h for h in (rec.get("hits") or []) if isinstance(h, dict)]
        mf, ms, mt, tw = _stage4_paper_weighted_topic_mix(hits, term_topic_meta)
        fk = _stage4_argmax_topic_key(mf)
        sk = _stage4_argmax_topic_key(ms)
        tk = _stage4_argmax_topic_key(mt)
        wid_sigs[wid] = (fk, sk, tk, tw, mf, ms, mt)
        for prefix, key in (("f", fk), ("s", sk), ("t", tk)):
            if key:
                key_freq[prefix + ":" + str(key)] += 1

    n_pool = max(1, len(wids))
    min_freq = max(
        STAGE4_TOPIC_NEIGHBORHOOD_MIN_FREQ,
        int(math.ceil(STAGE4_TOPIC_NEIGHBORHOOD_POOL_FREQ_FRAC * n_pool)),
    )
    pool_keys_st = {k for k, c in key_freq.items() if c >= min_freq and (k.startswith("s:") or k.startswith("t:"))}
    pool_keys_f = set()
    if STAGE4_TOPIC_NEIGHBORHOOD_INCLUDE_FIELD_POOL:
        pool_keys_f = {k for k, c in key_freq.items() if c >= min_freq + 2 and k.startswith("f:")}
    pool_neighborhood_keys: Set[str] = set(pool_keys_st) | pool_keys_f | jd_key_set

    neighbor_counts = _stage4_compute_paper_topic_neighbor_counts(wids, wid_sigs)

    role_ctr: Counter = Counter()
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    cohesive_set: Set[str] = set()
    for wid in wids:
        rec = by_wid[wid]
        fk, sk, tk, tw, mf, ms, mt = wid_sigs[wid]
        det = rec.get("hierarchy_consensus_detail") or {}
        try:
            fc = float(det.get("field_cons") or 0.0)
            sc = float(det.get("subfield_cons") or 0.0)
            tc = float(det.get("topic_cons") or 0.0)
        except (TypeError, ValueError):
            fc = sc = tc = 0.0
        jd_flag = _stage4_jd_profile_overlap_flag_strict(fc, sc, tc)
        nh = int(neighbor_counts.get(wid, 0))
        st_keys: Set[str] = set()
        if sk:
            st_keys.add("s:" + str(sk))
        if tk:
            st_keys.add("t:" + str(tk))
        in_nh_st = bool(st_keys & pool_neighborhood_keys)
        in_nh_field_secondary = bool(fk and (("f:" + str(fk)) in pool_neighborhood_keys) and max(sc, tc) >= 0.028)
        in_nh = bool(in_nh_st or in_nh_field_secondary)
        rec["paper_jd_topic_profile_overlap_flag"] = jd_flag
        rec["paper_topic_neighborhood_hit_flag"] = in_nh
        rec["paper_topic_neighbor_count"] = nh
        rec["paper_topic_community_score"] = _stage4_compute_paper_topic_community_score(
            rec, neighbor_count=nh, in_neighborhood=in_nh
        )
        role = _stage4_assign_candidate_source_role(rec)
        rec["candidate_source_role"] = role
        rec["paper_topic_cohesive_flag"] = role == "topic_cohesive_candidate"
        role_ctr[role] += 1
        if role == "topic_cohesive_candidate":
            cohesive_set.add(wid)
        if len(examples[role]) < 5:
            examples[role].append(
                {
                    "wid": wid,
                    "title": str(rec.get("title") or "")[:96],
                    "neighbor_count": nh,
                    "cohesion_strong_signals": _stage4_count_cohesion_strong_signals(rec),
                    "jd_overlap_flag": jd_flag,
                    "in_topic_neighborhood": in_nh,
                }
            )

    summary = {
        "candidate_pool_size": len(wids),
        "candidate_source_role_counter": dict(role_ctr),
        "topic_cohesive_candidate_count": int(role_ctr.get("topic_cohesive_candidate", 0)),
        "topic_weak_candidate_count": int(role_ctr.get("topic_weak_candidate", 0)),
        "term_only_candidate_count": int(role_ctr.get("term_only_candidate", 0)),
        "topic_neighborhood_key_count": len(pool_neighborhood_keys),
        "pool_neighborhood_min_freq_used": min_freq,
        "pool_cohesive_key_diversity": len({k for k, c in key_freq.items() if c >= min_freq}),
        "examples_by_role": {k: list(v) for k, v in examples.items()},
    }
    return cohesive_set, summary


def _print_stage4_candidate_source_summary(summary: Dict[str, Any], *, audit_print: bool) -> None:
    if not audit_print:
        return
    print("\n" + "-" * 80)
    print("[Stage4 candidate source summary]")
    print("-" * 80)
    print(
        f"candidate_pool_size={summary.get('candidate_pool_size')} "
        f"candidate_source_role_counter={summary.get('candidate_source_role_counter')} "
        f"topic_cohesive_candidate_count={summary.get('topic_cohesive_candidate_count')} "
        f"topic_weak_candidate_count={summary.get('topic_weak_candidate_count')} "
        f"term_only_candidate_count={summary.get('term_only_candidate_count')} "
        f"topic_neighborhood_key_count={summary.get('topic_neighborhood_key_count')} "
        f"pool_neighborhood_min_freq_used={summary.get('pool_neighborhood_min_freq_used')} "
        f"pool_cohesive_key_diversity={summary.get('pool_cohesive_key_diversity')}"
    )
    ex = summary.get("examples_by_role") or {}
    for role in ("topic_cohesive_candidate", "topic_weak_candidate", "term_only_candidate"):
        rows = ex.get(role) or []
        print(f"--- examples {role} (max 5) ---")
        if not rows:
            print("  (empty)")
        else:
            for row in rows:
                print(f"  {row}")
    print("-" * 80 + "\n")


def _print_stage4_paper_community_filtering_summary(
    stats: Optional[Dict[str, Any]],
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    *,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    print("\n" + "-" * 80)
    print("[Stage4 paper community filtering summary]")
    print("-" * 80)
    dist: Counter = Counter()
    for _w, rec in by_wid.items():
        if not isinstance(rec, dict):
            continue
        try:
            nc = int(rec.get("paper_topic_neighbor_count") or 0)
        except (TypeError, ValueError):
            nc = 0
        dist[nc] += 1
    print(f"paper_topic_neighbor_count_histogram={dict(sorted(dist.items()))}")
    if stats:
        print(
            f"isolated_fringe_community_guard_hit_count={stats.get('community_isolated_fringe_guard_hit_count')} "
            f"topic_cohesive_keep_bonus_hit_count={stats.get('community_topic_cohesive_keep_bonus_hit_count')}"
        )
    kept = [w for w in selected_wids_set if w in by_wid]
    k_cohesive = 0
    k_weak = 0
    k_isolated_term = 0
    k_n_neighbor_ge1 = 0
    for w in kept:
        r = by_wid.get(w) or {}
        role = str(r.get("candidate_source_role") or "")
        if role == "topic_cohesive_candidate":
            k_cohesive += 1
        if role == "topic_weak_candidate":
            k_weak += 1
        if role == "term_only_candidate":
            k_isolated_term += 1
        try:
            if int(r.get("paper_topic_neighbor_count") or 0) >= 1:
                k_n_neighbor_ge1 += 1
        except (TypeError, ValueError):
            pass
    print(
        f"kept_papers_count={len(kept)} "
        f"kept_topic_cohesive_candidate_count={k_cohesive} "
        f"kept_topic_weak_candidate_count={k_weak} "
        f"kept_term_only_candidate_count={k_isolated_term} "
        f"kept_neighbor_count_ge1={k_n_neighbor_ge1}"
    )
    print("-" * 80 + "\n")


def _stage4_collect_keep_compete_extra_rows_for_term(
    triples: List[tuple],
    vid: int,
    local_cap: int,
) -> List[tuple]:
    """
    local_cap 之后、仍在 triples 内（已过 grounding 排序）的项，作为 keep compete 扩面用的行。
    STAGE4_KEEP_COMPETE_EXTRA_PER_TERM：<=0 表示不裁剪条数（纳入 remainder 全部）；>0 表示最多再纳入这么多条。
    """
    if local_cap >= len(triples):
        return []
    tail = triples[local_cap:]
    n_extra = int(STAGE4_KEEP_COMPETE_EXTRA_PER_TERM)
    if n_extra > 0:
        tail = tail[:n_extra]
    out: List[tuple] = []
    for row in tail:
        (wid, term_contrib, idf_weight, hit_level, grounding, jd_align) = row[:6]
        out.append((wid, vid, term_contrib, idf_weight, hit_level, grounding, jd_align))
    return out


def _stage4_keep_compete_example_line(wid: str, rec: Any) -> str:
    """keep competition 日志：单条 wid 的 added/removed 样例行。"""
    if not isinstance(rec, dict):
        rec = {}
    title = str(rec.get("title") or "")[:120]
    try:
        old_s = float(rec.get("paper_score") or 0.0)
    except (TypeError, ValueError):
        old_s = 0.0
    try:
        v2 = float(rec.get("paper_final_score_v2") or 0.0)
    except (TypeError, ValueError):
        v2 = 0.0
    try:
        ck = float(rec.get("paper_compete_score_for_keep") or 0.0)
    except (TypeError, ValueError):
        ck = 0.0
    return (
        f"wid={wid} title={title!r} "
        f"candidate_source_role={rec.get('candidate_source_role')} "
        f"paper_topic_neighbor_count={rec.get('paper_topic_neighbor_count')} "
        f"hit_quality_class={rec.get('hit_quality_class')} "
        f"mainline_term_count={rec.get('mainline_term_count')} "
        f"hit_count={rec.get('hit_count')} "
        f"paper_evidence_role_preview={rec.get('paper_evidence_role_preview')} "
        f"paper_evidence_role={rec.get('paper_evidence_role')} "
        f"paper_old_score={old_s:.6f} "
        f"paper_final_score_v2={v2:.6f} "
        f"paper_compete_score_for_keep={ck:.6f}"
    )


def _robot_control_axis_hit_count(merged_lc: str) -> int:
    """仅用于 robot control 错域补丁：统计 title+domains 合并串中主轴偏好词命中数。"""
    axis_prefs = (
        "motion control",
        "robot motion",
        "trajectory optimization",
        "trajectory",
        "planning",
        "path planning",
        "motion",
        "kinematics",
        "dynamics",
        "robot dynamics",
        "locomotion",
        "state estimation",
        "optimal control",
        "mpc",
        "ilqr",
        "ddp",
        "controller design",
        "controller",
        "compliance",
        "whole-body",
        "whole body",
        "manipulator",
        "quadruped",
        "humanoid",
        "biped",
    )
    return sum(1 for kw in axis_prefs if kw in merged_lc)


def _robot_control_offtopic_penalty(merged_lc: str) -> Tuple[float, Dict[str, Any]]:
    """
    极窄：仅当 term 为「robot control」时在 grounding 中乘到 off_topic_penalty 上。
    第二轮收紧：强错域近否决式压分；chat/LLM/安全覆盖面加大；robot+control 缺主轴时区分度更强；强错域∩缺主轴可连乘。
    """
    mult = 1.0
    explain: Dict[str, Any] = {}
    axis_hits = _robot_control_axis_hit_count(merged_lc)
    explain["robot_control_axis_hit_count"] = int(axis_hits)

    # 第一层 A：医疗 / 康复 / 患者支持（近「一票否决」式乘子，较第一轮 0.18 显著更强）
    medical_kw = (
        "rehabilitation",
        "rehabilitative",
        "stroke",
        "patient",
        "patients",
        "healthcare",
        "clinical",
        "medical",
        "therapy",
        "therapeutic",
        "hospital",
        "nursing",
        "assistive",
        "disability",
        "elderly",
        "elder care",
        "elderly care",
        "elderly-care",
        "patient support",
        "care robot",
        "assistive robot",
    )
    medical_hit = any(k in merged_lc for k in medical_kw)
    if medical_hit:
        mult *= 0.06
        explain["robot_control_medical_penalty"] = True

    # 第一层 B：chat / LLM / alignment / safety / cyber（较第一轮 0.22 更强 + pattern 更全）
    chat_safety_kw = (
        "chat control",
        "backdoor",
        "llm",
        "large language",
        "large language model",
        "language model",
        "chatbot",
        "jailbreak",
        "safe alignment",
        "ai alignment",
        "gpt-",
        " gpt",
        "gpt ",
        "llama",
        "prompt injection",
        "cybersecurity",
        "cyber security",
        "cyberattack",
        "cyber attack",
        "cyber-attack",
        "adversarial attack",
        "network security",
        "computer security",
    )
    chat_hit = any(k in merged_lc for k in chat_safety_kw)
    if chat_hit:
        mult *= 0.08
        explain["robot_control_chat_safety_penalty"] = True
    # safety / alignment / security：与 LLM/对抗/网络语义绑定时强压，避免单独误伤「机器人安全关键」类表述
    _robot_motion_hint = any(
        k in merged_lc
        for k in (
            "locomotion",
            "manipulator",
            "trajectory",
            "kinematics",
            "dynamics",
            "quadruped",
            "humanoid",
            "motion control",
            "biped",
        )
    )
    if "safety" in merged_lc and any(
        k in merged_lc
        for k in (
            "llm",
            "language model",
            "chatbot",
            "jailbreak",
            "prompt",
            "adversarial",
            "alignment",
            "attack",
            "defense",
        )
    ):
        if not _robot_motion_hint:
            mult *= 0.08
            explain["robot_control_chat_safety_penalty"] = True
    if "alignment" in merged_lc and any(
        k in merged_lc for k in ("llm", "language model", "chatbot", "gpt", "prompt", "jailbreak", "safe")
    ):
        if not _robot_motion_hint:
            mult *= 0.09
            explain["robot_control_chat_safety_penalty"] = True
    if "security" in merged_lc and any(
        k in merged_lc for k in ("cyber", "network", "adversarial", "llm", "language model", "attack", "defense")
    ):
        if not _robot_motion_hint and "robot" not in merged_lc and "robotic" not in merged_lc:
            mult *= 0.10
            explain["robot_control_chat_safety_penalty"] = True
    if ("attack" in merged_lc or "defense" in merged_lc) and any(
        k in merged_lc for k in ("adversarial", "llm", "language model", "network security", "cyber")
    ):
        if "robot" not in merged_lc and "robotic" not in merged_lc:
            mult *= 0.08
            explain["robot_control_chat_safety_penalty"] = True

    strong_offtopic = bool(
        explain.get("robot_control_medical_penalty") or explain.get("robot_control_chat_safety_penalty")
    )
    if strong_offtopic:
        explain["robot_control_strong_offtopic_hit"] = True

    # 第二层：robot / robotic + control 同时出现，但主轴命中不足 → 较第一轮 0.88 明显更强，且 0 命中与 1 命中区分
    looks_generic_rc = ("robot" in merged_lc or "robotic" in merged_lc) and "control" in merged_lc
    if looks_generic_rc:
        if axis_hits == 0:
            mult *= 0.40
            explain["robot_control_axis_missing_penalty"] = True
        elif axis_hits == 1:
            mult *= 0.68
            explain["robot_control_axis_missing_penalty"] = True
        # axis_hits >= 2：不施加 axis_missing 惩罚

    # 第三层：强错域且（字面 robot+control 且主轴不足）→ 额外连乘收紧
    if strong_offtopic and looks_generic_rc and axis_hits < 2:
        mult *= 0.58
        explain["robot_control_strong_offtopic_hit"] = True

    explain["robot_control_offtopic_penalty"] = float(mult)
    return mult, explain


def _robot_control_motion_alias_support(
    term_text_lc: str,
    merged_lc: str,
    parent_anchor_raw: str,
    parent_primary_raw: str,
    rc_offtopic_mult: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    极窄：仅 term 为 robot control、且父锚处于「机器人运动控制/运动控制」语境时，
    若 paper 标题+domains 命中运动控制向短语，则在 grounding 上给小额加分（在 off-topic 乘子已确定之后计算）。
    rc_offtopic_mult 过低（强错域）时不启用，避免与 rehab/chat 压制对冲。
    """
    explain: Dict[str, Any] = {}
    if (term_text_lc or "").strip() != "robot control":
        return 0.0, explain
    parents_blob = f"{parent_anchor_raw}\n{parent_primary_raw}"
    if ("机器人运动控制" not in parents_blob) and ("运动控制" not in parents_blob):
        return 0.0, explain
    if rc_offtopic_mult < 0.14:
        explain["robot_motion_alias_skipped_strong_offtopic"] = True
        return 0.0, explain

    aliases = (
        "motion control",
        "robot motion control",
        "robot dynamics",
        "trajectory control",
        "whole-body control",
        "whole body control",
        "locomotion control",
        "quadruped control",
        "humanoid control",
        "manipulator control",
        "robot controller",
        "controller design",
        "compliance control",
        "optimal control",
        "state estimation",
    )
    hits = [a for a in aliases if a in merged_lc]
    if not hits:
        return 0.0, explain

    nh = min(len(hits), 4)
    bonus = min(0.05, 0.0125 * nh)
    explain["robot_motion_alias_support"] = True
    explain["robot_motion_alias_hit"] = True
    explain["robot_motion_alias_terms"] = hits[:12]
    explain["robot_motion_alias_bonus"] = float(bonus)
    return float(bonus), explain


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
    "paper_evidence_quality_score",
    "paper_evidence_quality_factors",
    "paper_evidence_role_preview",
    "paper_old_score",
    "paper_final_score_v2",
    "paper_old_score_rank",
    "paper_final_score_v2_rank",
    "paper_evidence_role",
    "paper_final_score_v2_before_limit",
    "paper_final_score_v2_after_limit",
    "fringe_v2_uplift_limited",
    "fringe_v2_local_weights",
    "fringe_v2_uplift_cap_applied",
    "paper_compete_score_for_keep",
    "pre_kept_side_only_fringe_guard_applied",
    "pre_kept_mainline_keep_bonus_applied",
    "candidate_source_role",
    "paper_topic_neighbor_count",
    "paper_topic_community_score",
    "paper_topic_cohesive_flag",
    "paper_jd_topic_profile_overlap_flag",
    "paper_topic_neighborhood_hit_flag",
    "community_isolated_fringe_community_guard_applied",
    "community_topic_cohesive_keep_bonus_applied",
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
    """
    是否计为「主轴/primary-like」命中（用于 mainline_term_count 与 multi-hit 分层）。

    优先读 Stage3/Stage4-prep 写入 **term_meta**（retrieval_role / paper_select_lane_tier / lane_type），
    不单独依赖 hit[\"role\"]：后者可能被 Stage4 为 umbrella 词覆盖（如压低泛词 Primary 权重），
    否则会与 prep 已选入 paper lane 的语义脱节。
    """
    try:
        tid = int(hit.get("vid"))
    except (TypeError, ValueError):
        return False
    meta = get_term_meta(tid) or {}
    tier = str(meta.get("paper_select_lane_tier") or "").strip().lower()
    if tier == "stage4_prep_core":
        return True
    lt = str(meta.get("lane_type") or hit.get("lane_type") or "").strip().lower()
    if lt == "risky_bridge_coverage":
        return False
    if lt == "direct_primary":
        return True
    role = str(meta.get("retrieval_role") or hit.get("role") or "").strip().lower()
    if lt == "support_coverage" and role == "paper_support":
        return False
    if role == "paper_primary":
        return True
    if role == "paper_support":
        return False
    r2 = str(meta.get("retrieval_role") or "").strip().lower()
    return r2 == "paper_primary"


def _stage4_hits_parent_primary_axis_aligned(
    hits: List[Dict[str, Any]], get_term_meta: Any
) -> bool:
    """多条命中是否共享同一 parent_primary（结构主轴簇），用于区分互补 multi-hit 与纯侧向 accidental。"""
    if len(hits) < 2:
        return False
    pps: List[str] = []
    for h in hits:
        if not isinstance(h, dict):
            return False
        try:
            tid = int(h.get("vid"))
        except (TypeError, ValueError):
            return False
        meta = get_term_meta(tid) or {}
        pp = str(meta.get("parent_primary") or h.get("parent_primary") or "").strip()
        if not pp:
            return False
        pps.append(pp.lower())
    return len(set(pps)) == 1


def _stage4_audit_resolve_mainline_side_lists(
    hits: List[Dict[str, Any]], get_term_meta: Any
) -> Tuple[List[str], List[str]]:
    """与 _stage4_hit_mainline_like 一致：按条拆 mainline / side 词面（不含 axis_boost 虚拟计数）。"""
    mains: List[str] = []
    sides: List[str] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        t = str(h.get("term") or h.get("vid") or "")
        if _stage4_hit_mainline_like(h, get_term_meta):
            mains.append(t)
        else:
            sides.append(t)
    return mains, sides


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

    - mainline 口径以 term_meta 为准（见 _stage4_hit_mainline_like）。
    - 若尚无 per-hit mainline、但多命中共享 parent_primary，则记一次「主轴簇互补」credit，
      将分层从纯 side_only 提升为 mainline_plus_support（不改动 evidence 公式字段）。
    """
    clean = [h for h in (hits or []) if isinstance(h, dict)]
    n = len(clean)
    ml_raw = sum(1 for h in clean if _stage4_hit_mainline_like(h, get_term_meta))
    ml = int(ml_raw)
    axis_aligned = bool(n >= 2 and ml_raw == 0 and _stage4_hits_parent_primary_axis_aligned(clean, get_term_meta))
    if axis_aligned:
        ml = 1
    sl = n - ml
    sf = _stage4_same_family_or_cross_family(clean, get_term_meta)
    detail: Dict[str, Any] = {
        "mainline_like_hits_raw_count": int(ml_raw),
        "axis_parent_primary_boost_applied": bool(axis_aligned),
    }

    if n <= 0:
        return {
            "mainline_term_count": 0,
            "side_term_count": 0,
            "same_family_or_cross_family": "single_hit",
            "hit_quality_class": "side_only_or_accidental_multi_hit",
            "coherence_reason": "no hits",
            "multi_hit_strength": 0,
            "hit_quality_detail": detail,
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
            "hit_quality_detail": detail,
        }

    if ml >= 2:
        hq = "mainline_resonance"
        cr = "two or more mainline-like hits"
        mhs = 2
    elif ml >= 1 and sl >= 1:
        hq = "mainline_plus_support"
        if axis_aligned and ml_raw == 0:
            cr = "axis-aligned complementary multi-hit (shared parent_primary among paper-lane terms)"
        else:
            cr = "mainline + support complement"
        mhs = 2
    else:
        hq = "side_only_or_accidental_multi_hit"
        if ml_raw == 0 and (not axis_aligned) and n >= 2:
            cr = "multi-hit with no mainline-like term and parent_primary not unified (accidental/side-heavy)"
        else:
            cr = "two hits but both side/support-like"
        detail["side_only_classification"] = "no_mainline_and_no_parent_primary_cluster"
        mhs = 1

    return {
        "mainline_term_count": ml,
        "side_term_count": sl,
        "same_family_or_cross_family": sf,
        "hit_quality_class": hq,
        "coherence_reason": cr,
        "multi_hit_strength": mhs,
        "hit_quality_detail": detail,
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
    hit_rr_prep: List[str] = []
    for h in hits:
        try:
            _tid = int(h.get("vid"))
            hit_rr_prep.append(str((get_term_meta(_tid) or {}).get("retrieval_role") or ""))
        except (TypeError, ValueError):
            hit_rr_prep.append("")
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
    rec["hit_term_retrieval_roles_prep"] = hit_rr_prep
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
    rec["paper_hit_quality_detail"] = dict(cls.get("hit_quality_detail") or {})
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


def _safe01(x: Any, default: float = 0.0) -> float:
    """将任意数值安全裁剪到 [0, 1]；无法解析时用 default。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    if v != v:  # NaN
        return float(default)
    return float(np.clip(v, 0.0, 1.0))


def _stage4_collect_paper_evidence_inputs(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    从当前 paper record 抽取 evidence quality 所需输入；不修改 rec，不触碰排序分数字段语义。
    """
    hits = [h for h in (rec.get("hits") or []) if isinstance(h, dict)]
    hit_terms = rec.get("hit_terms")
    if not isinstance(hit_terms, list):
        hit_terms = [str(h.get("term") or h.get("vid") or "") for h in hits]

    hc = rec.get("hit_count")
    if hc is None:
        try:
            hc = int(len(hits))
        except (TypeError, ValueError):
            hc = 0
    else:
        try:
            hc = int(hc)
        except (TypeError, ValueError):
            hc = len(hits)

    ml = rec.get("mainline_term_count")
    sl = rec.get("side_term_count")
    try:
        ml_i = int(ml) if ml is not None else 0
    except (TypeError, ValueError):
        ml_i = 0
    try:
        sl_i = int(sl) if sl is not None else 0
    except (TypeError, ValueError):
        sl_i = max(0, hc - ml_i)

    hq = str(rec.get("hit_quality_class") or "").strip()
    if not hq and hits:
        # 无 class 时从 coherence / hits 粗判，避免空串
        cr = str(rec.get("coherence_reason") or "").lower()
        if "mainline" in cr and "single" in cr:
            hq = "single_hit_mainline"
        elif "side" in cr or "support" in cr:
            hq = "single_hit_side"

    coh = str(rec.get("coherence_reason") or "").strip().lower()
    gs = _safe01(rec.get("grounding_score"), 0.0)
    ja = rec.get("jd_align")
    if ja is None:
        jf: Optional[float] = None
    else:
        try:
            jf = float(ja)
        except (TypeError, ValueError):
            jf = None
    hb = rec.get("hierarchy_bonus")
    try:
        hb_f = float(hb) if hb is not None else None
    except (TypeError, ValueError):
        hb_f = None

    return {
        "hit_terms": hit_terms,
        "hit_count": hc,
        "mainline_term_count": ml_i,
        "side_term_count": sl_i,
        "hit_quality_class": hq,
        "coherence_reason_lower": coh,
        "grounding_score": gs,
        "jd_align": jf,
        "hierarchy_bonus": hb_f,
    }


def _stage4_should_shrink_side_only_evidence_quality(inp: Dict[str, Any]) -> bool:
    """
    极窄条件：single_hit_side + 无主轴计数 + 确有侧向命中。
    仅用于压掉「高 jd_align 在无边主轴时误抬 evidence」的假阳性，不误伤 mainline 类。
    """
    hq = str(inp.get("hit_quality_class") or "").strip()
    if hq != "single_hit_side":
        return False
    try:
        ml = int(inp.get("mainline_term_count") or 0)
    except (TypeError, ValueError):
        ml = 0
    if ml != 0:
        return False
    try:
        sl = int(inp.get("side_term_count") or 0)
    except (TypeError, ValueError):
        sl = 0
    return sl >= 1


def _paper_evidence_quality_role_preview(
    score_01: float,
    inp: Dict[str, Any],
) -> str:
    """
    仅预览标签：primary / support / fringe；不参与排序。
    """
    hq = str(inp.get("hit_quality_class") or "").strip()
    ml = int(inp.get("mainline_term_count") or 0)
    sl = int(inp.get("side_term_count") or 0)
    hc = int(inp.get("hit_count") or 0)

    fringe = False
    if hq == "single_hit_side":
        fringe = True
    if hq == "side_only_or_accidental_multi_hit":
        fringe = True
    if ml == 0 and sl > 0 and hc >= 1:
        fringe = True
    if score_01 < 0.34:
        fringe = True

    primary_ok = (
        score_01 >= 0.58
        and hq != "single_hit_side"
        and ml >= 1
        and not fringe
    )

    if fringe:
        return "fringe_candidate"
    if primary_ok:
        return "primary_candidate"
    return "support_candidate"


def _compute_paper_evidence_quality_score(rec: Dict[str, Any]) -> Tuple[float, Dict[str, Any], str]:
    """
    旁路证据质量分（0~1）及分项；不读取/修改 paper_score_final / rerank 相关字段。
    """
    inp = _stage4_collect_paper_evidence_inputs(rec)
    hq = inp["hit_quality_class"] or "other"

    # 分项基底：类别主信号（无单项一票否决，最终仍与其它项加权混合）
    base_map = {
        "mainline_resonance": 0.90,
        "mainline_plus_support": 0.84,
        "single_hit_mainline": 0.72,
        "side_only_or_accidental_multi_hit": 0.40,
        "single_hit_side": 0.30,
        "other": 0.48,
    }
    base_hq = float(base_map.get(hq, base_map["other"]))

    g = _safe01(inp.get("grounding_score"), 0.0)
    jd_raw = inp.get("jd_align")
    ml = int(inp.get("mainline_term_count") or 0)
    sl = int(inp.get("side_term_count") or 0)
    hc = int(inp.get("hit_count") or 0)

    shrink_ev = _stage4_should_shrink_side_only_evidence_quality(inp)
    # 默认 jd 参与强度
    if jd_raw is None:
        jd_part = 0.55
    else:
        jd_part = _safe01(jd_raw, 0.55)
    # A：极窄 gating — single_hit_side + 无 mainline 时，不让高 jd_align 全强度抬分
    if shrink_ev:
        if jd_raw is None:
            jd_part = 0.35 * 0.55 + 0.65 * 0.5
        else:
            jn = _safe01(jd_raw, 0.55)
            jd_part = 0.35 * float(jn) + 0.65 * 0.5

    # B：同条件下的额外薄弱惩罚（raw_linear 上扣减，clip 前）
    fringe_side_hit_penalty = 0.08 if shrink_ev else 0.0

    mainline_bonus = 0.06 if ml >= 1 else 0.0
    multi_hit_bonus = 0.04 if hc >= 2 else 0.0

    side_only_penalty = 0.0
    if sl > 0 and ml == 0:
        side_only_penalty = 0.10

    coh = inp.get("coherence_reason_lower") or ""
    coherence_mainline_hint = 0.03 if ("mainline" in coh and "two or more" in coh) else 0.0
    if "mainline + support" in coh:
        coherence_mainline_hint = max(coherence_mainline_hint, 0.02)

    # hierarchy_bonus：仅作微弱形状信号（通常约 0.82~1.15），映射到 0~1 再加权
    hb = inp.get("hierarchy_bonus")
    hb_n = _safe01((float(hb) - 0.82) / 0.33, 0.5) if hb is not None else 0.5

    # 轻量线性合成后裁剪到 0~1（内部保留分项供审计）
    raw_linear = (
        0.42 * base_hq
        + 0.22 * g
        + 0.18 * jd_part
        + 0.08 * hb_n
        + mainline_bonus
        + multi_hit_bonus
        - side_only_penalty
        + coherence_mainline_hint
        - fringe_side_hit_penalty
    )
    score_01 = _safe01(raw_linear, 0.0)

    jd_align_raw_val: Optional[float]
    if jd_raw is None:
        jd_align_raw_val = None
    else:
        try:
            jd_align_raw_val = round(float(jd_raw), 4)
        except (TypeError, ValueError):
            jd_align_raw_val = None

    factors: Dict[str, Any] = {
        "base_hit_quality_mapped": round(base_hq, 4),
        "hit_quality_class": hq,
        "grounding_01": round(g, 4),
        "jd_align_raw": jd_align_raw_val,
        "jd_align_effective": round(jd_part, 4),
        "jd_align_was_missing": jd_raw is None,
        "side_only_quality_shrink_applied": bool(shrink_ev),
        "fringe_side_hit_penalty": round(fringe_side_hit_penalty, 4),
        "offaxis_control_phrase_penalty": 0.0,
        "mainline_bonus": round(mainline_bonus, 4),
        "multi_hit_bonus": round(multi_hit_bonus, 4),
        "side_only_penalty": round(side_only_penalty, 4),
        "coherence_mainline_hint": round(coherence_mainline_hint, 4),
        "hierarchy_bonus_norm": round(hb_n, 4),
        "raw_linear_before_clip": round(raw_linear, 6),
    }
    role = _paper_evidence_quality_role_preview(score_01, inp)
    return score_01, factors, role


def _stage4_attach_paper_evidence_quality_fields(rec: Dict[str, Any]) -> None:
    """在 explanation materialize 之后写回旁路字段；不改 paper_score_final / rerank。"""
    sc, fac, role = _compute_paper_evidence_quality_score(rec)
    rec["paper_evidence_quality_score"] = float(sc)
    rec["paper_evidence_quality_factors"] = fac
    rec["paper_evidence_role_preview"] = str(role)


def _stage4_paper_evidence_quality_compact_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    ht = rec.get("hit_terms")
    if isinstance(ht, list):
        ht_s = str(ht)[:120]
    else:
        ht_s = str(ht or "")[:120]
    return {
        "wid": rec.get("wid"),
        "title": str(rec.get("title") or "")[:80],
        "hit_terms": ht_s,
        "hit_quality_class": rec.get("hit_quality_class"),
        "mainline_term_count": rec.get("mainline_term_count"),
        "side_term_count": rec.get("side_term_count"),
        "grounding_score": rec.get("grounding_score"),
        "jd_align": rec.get("jd_align"),
        "paper_score_final": rec.get("paper_score_final"),
        "paper_evidence_quality_score": rec.get("paper_evidence_quality_score"),
        "paper_evidence_role_preview": rec.get("paper_evidence_role_preview"),
    }


def _print_stage4_paper_evidence_quality_summary(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    audit_print: bool,
) -> None:
    if not audit_print or not by_wid:
        return
    kept = [w for w in by_wid.keys() if w in selected_wids_set]
    kept_papers_count = len(kept)
    if kept_papers_count == 0:
        print("\n[Stage4 paper evidence quality summary] kept_papers_count=0\n")
        return

    scores: List[float] = []
    role_ctr: Counter = Counter()
    hit_ctr: Counter = Counter()

    for w in kept:
        r = by_wid.get(w) or {}
        v = r.get("paper_evidence_quality_score")
        try:
            scores.append(float(v) if v is not None else 0.0)
        except (TypeError, ValueError):
            scores.append(0.0)
        role_ctr[str(r.get("paper_evidence_role_preview") or "unknown")] += 1
        hit_ctr[str(r.get("hit_quality_class") or "unknown")] += 1

    smin = min(scores) if scores else 0.0
    smax = max(scores) if scores else 0.0
    smean = float(sum(scores) / len(scores)) if scores else 0.0

    print("\n" + "-" * 80)
    print("[Stage4 paper evidence quality summary]")
    print("-" * 80)
    print(
        f"kept_papers_count={kept_papers_count} "
        f"quality_role_preview_counter={dict(role_ctr)} "
        f"quality_score_range=min={smin:.4f} max={smax:.4f} mean={smean:.4f} "
        f"hit_quality_counter={dict(hit_ctr)}"
    )
    print("-" * 80 + "\n")


def _print_stage4_paper_evidence_quality_top_rows(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    audit_print: bool,
) -> None:
    if not audit_print or not by_wid:
        return
    kept = [w for w in by_wid.keys() if w in selected_wids_set]
    if not kept:
        print("\n[Stage4 paper evidence quality top rows] kept_papers_count=0\n")
        return

    kept_by_final = sorted(
        kept,
        key=lambda x: -float((by_wid.get(x) or {}).get("paper_score_final") or 0.0),
    )
    top_rows = [_stage4_paper_evidence_quality_compact_row(by_wid[w]) for w in kept_by_final[:8]]

    kept_by_eq = sorted(
        kept,
        key=lambda x: float((by_wid.get(x) or {}).get("paper_evidence_quality_score") or 0.0),
    )
    low_rows = [_stage4_paper_evidence_quality_compact_row(by_wid[w]) for w in kept_by_eq[:5]]

    print("-" * 80)
    print("[Stage4 paper evidence quality top rows]")
    print("-" * 80)
    print(f"--- top kept papers by paper_score_final (max {min(8, len(kept_by_final))}) ---")
    for row in top_rows:
        print(f"  {row}")
    print(f"--- lowest paper_evidence_quality_score among kept (max {min(5, len(kept_by_eq))}) ---")
    for row in low_rows:
        print(f"  {row}")
    print("-" * 80 + "\n")


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


def _stage4_paper_old_score_for_output(rec: Dict[str, Any]) -> float:
    """
    与 wid_to_hits_and_score → papers[].score 一致：protected > rerank > raw paper_score。
    作为 v2 混合的「旧分」口径，避免与 Stage5 实际读到的分脱节。
    """
    if rec.get("paper_score_protected") is not None:
        try:
            return float(rec.get("paper_score_protected") or 0.0)
        except (TypeError, ValueError):
            pass
    if rec.get("paper_rerank_score") is not None:
        try:
            return float(rec.get("paper_rerank_score") or 0.0)
        except (TypeError, ValueError):
            pass
    try:
        return float(rec.get("paper_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _stage4_should_limit_fringe_v2_uplift(rec: Dict[str, Any]) -> bool:
    """
    极窄：低 paper_old_score 的 single_hit_side + 无主轴，且预览为 fringe，
    避免 evidence(0~1) 把极低 old 在 v2 排名里不合理托升。
    """
    hq = str(rec.get("hit_quality_class") or "").strip()
    if hq != "single_hit_side":
        return False
    try:
        ml = int(rec.get("mainline_term_count") or 0)
    except (TypeError, ValueError):
        ml = 0
    if ml != 0:
        return False
    try:
        old = float(rec.get("paper_old_score") or 0.0)
    except (TypeError, ValueError):
        old = 0.0
    if old >= STAGE4_FRINGE_V2_LOW_OLD_THRESHOLD:
        return False
    pv = str(rec.get("paper_evidence_role_preview") or "").strip()
    if pv and pv != "fringe_candidate":
        return False
    return True


def _compute_stage4_local_v2_weights(rec: Dict[str, Any]) -> Tuple[float, float]:
    """全局仍为 0.85/0.15；仅极窄 fringe uplift 风险条目标记为 0.93/0.07。"""
    if _stage4_should_limit_fringe_v2_uplift(rec):
        return (float(STAGE4_FRINGE_V2_LOCAL_WEIGHT_OLD), float(STAGE4_FRINGE_V2_LOCAL_WEIGHT_EVIDENCE))
    return (float(STAGE4_EVIDENCE_MIGRATION_WEIGHT_OLD), float(STAGE4_EVIDENCE_MIGRATION_WEIGHT_EVIDENCE))


def _assign_stage4_paper_evidence_role_base(rec: Dict[str, Any]) -> str:
    """
    正式 evidence role（厚度门之前）：必须同时参考 v2、hit_quality_class、mainline_term_count；
    禁止将明显 single_hit_side 判为 primary。
    """
    hq = str(rec.get("hit_quality_class") or "").strip()
    try:
        ml = int(rec.get("mainline_term_count") or 0)
    except (TypeError, ValueError):
        ml = 0
    try:
        sl = int(rec.get("side_term_count") or 0)
    except (TypeError, ValueError):
        sl = 0
    try:
        v2 = float(rec.get("paper_final_score_v2") or 0.0)
    except (TypeError, ValueError):
        v2 = 0.0

    # fringe：硬规则优先
    if hq == "single_hit_side":
        return "fringe_evidence"
    if hq == "side_only_or_accidental_multi_hit" and ml == 0:
        return "fringe_evidence"
    if ml == 0 and sl >= 1 and hq in ("single_hit_side", "side_only_or_accidental_multi_hit"):
        return "fringe_evidence"
    if v2 < 0.38 and ml == 0 and hq != "single_hit_mainline":
        return "fringe_evidence"

    # primary：高分 + 主轴证据结构
    if ml >= 1 and hq in ("mainline_resonance", "mainline_plus_support", "single_hit_mainline") and v2 >= 0.52:
        return "primary_evidence"
    if ml >= 1 and v2 >= 0.62 and hq not in ("single_hit_side", "side_only_or_accidental_multi_hit"):
        return "primary_evidence"

    return "support_evidence"


def _attach_stage4_score_migration_fields(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    jd_align_map: Optional[Dict[str, float]] = None,
) -> None:
    """批量写回迁移字段；ranks 仅在全局 kept（selected_wids_set）内编号。"""
    for _w, rec in by_wid.items():
        if not isinstance(rec, dict):
            continue
        rec["paper_old_score"] = float(_stage4_paper_old_score_for_output(rec))
        old = float(rec["paper_old_score"])
        ev = _safe01(rec.get("paper_evidence_quality_score"), 0.0)
        wg_o = float(STAGE4_EVIDENCE_MIGRATION_WEIGHT_OLD)
        wg_e = float(STAGE4_EVIDENCE_MIGRATION_WEIGHT_EVIDENCE)
        v2_before = float(wg_o * old + wg_e * ev)
        wo, we = _compute_stage4_local_v2_weights(rec)
        v2_after = float(wo * old + we * ev)
        lim = _stage4_should_limit_fringe_v2_uplift(rec)
        rec["paper_final_score_v2_before_limit"] = float(v2_before)
        rec["paper_final_score_v2_after_limit"] = float(v2_after)
        rec["fringe_v2_uplift_limited"] = bool(lim)
        rec["fringe_v2_local_weights"] = f"{wo:.2f},{we:.2f}" if lim else None
        rec["fringe_v2_uplift_cap_applied"] = False
        rec["paper_final_score_v2"] = float(v2_after)
        rec["paper_evidence_role"] = str(_assign_stage4_paper_evidence_role(rec, jd_align_map))
        rec["paper_old_score_rank"] = None
        rec["paper_final_score_v2_rank"] = None

    kept = [w for w in selected_wids_set if w in by_wid]
    if not kept:
        return

    rank_old_l = sorted(
        kept,
        key=lambda w: -float((by_wid.get(w) or {}).get("paper_old_score") or 0.0),
    )
    rank_v2_l = sorted(
        kept,
        key=lambda w: -float((by_wid.get(w) or {}).get("paper_final_score_v2") or 0.0),
    )
    ro = {w: i + 1 for i, w in enumerate(rank_old_l)}
    rv = {w: i + 1 for i, w in enumerate(rank_v2_l)}
    for w in kept:
        r = by_wid.get(w)
        if isinstance(r, dict):
            r["paper_old_score_rank"] = int(ro[w])
            r["paper_final_score_v2_rank"] = int(rv[w])


def _stage4_score_migration_compact_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    fac = rec.get("paper_evidence_quality_factors") if isinstance(rec.get("paper_evidence_quality_factors"), dict) else {}
    return {
        "wid": rec.get("wid"),
        "title": str(rec.get("title") or "")[:80],
        "hit_quality_class": rec.get("hit_quality_class"),
        "mainline_term_count": rec.get("mainline_term_count"),
        "paper_old_score": rec.get("paper_old_score"),
        "paper_evidence_quality_score": rec.get("paper_evidence_quality_score"),
        "paper_final_score_v2_before_limit": rec.get("paper_final_score_v2_before_limit"),
        "paper_final_score_v2_after_limit": rec.get("paper_final_score_v2_after_limit"),
        "paper_final_score_v2": rec.get("paper_final_score_v2"),
        "paper_old_score_rank": rec.get("paper_old_score_rank"),
        "paper_final_score_v2_rank": rec.get("paper_final_score_v2_rank"),
        "fringe_v2_uplift_limited": rec.get("fringe_v2_uplift_limited"),
        "fringe_v2_local_weights": rec.get("fringe_v2_local_weights"),
        "fringe_v2_uplift_cap_applied": rec.get("fringe_v2_uplift_cap_applied"),
        "jd_align_raw": fac.get("jd_align_raw"),
        "jd_align_effective": fac.get("jd_align_effective"),
        "fringe_side_hit_penalty": fac.get("fringe_side_hit_penalty"),
        "side_only_quality_shrink_applied": fac.get("side_only_quality_shrink_applied"),
    }


def _print_stage4_score_migration_summary(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    *,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    kept = [w for w in selected_wids_set if w in by_wid]
    kc = len(kept)
    print("\n" + "-" * 80)
    print("[Stage4 score migration]")
    print("-" * 80)
    print(
        f"kept_papers_count={kc} migration_enabled={STAGE4_ENABLE_EVIDENCE_SCORE_MIGRATION} "
        f"mixing_weight_old={STAGE4_EVIDENCE_MIGRATION_WEIGHT_OLD} "
        f"mixing_weight_evidence={STAGE4_EVIDENCE_MIGRATION_WEIGHT_EVIDENCE}"
    )
    if kc == 0:
        print("--- no kept papers ---")
        print("-" * 80 + "\n")
        return

    K = min(STAGE4_EVIDENCE_MIGRATION_TOPK_CHANGED, kc)
    rank_old_l = sorted(
        kept,
        key=lambda w: -float((by_wid.get(w) or {}).get("paper_old_score") or 0.0),
    )
    rank_v2_l = sorted(
        kept,
        key=lambda w: -float((by_wid.get(w) or {}).get("paper_final_score_v2") or 0.0),
    )
    ro = {w: i + 1 for i, w in enumerate(rank_old_l)}
    rv = {w: i + 1 for i, w in enumerate(rank_v2_l)}

    topk = set(rank_old_l[:K])
    changed = sum(1 for w in topk if ro[w] != rv[w])
    print(f"changed_rank_count_topK(top{K})={changed}")

    by_delta = sorted(
        kept,
        key=lambda w: -abs(ro[w] - rv[w]),
    )[:8]
    print("--- old_vs_v2 rank delta examples (max 8 by |Δrank|) ---")
    for w in by_delta:
        r = by_wid.get(w) or {}
        print(f"  {_stage4_score_migration_compact_row(r)} delta_rank={abs(ro[w] - rv[w])}")
    print("-" * 80 + "\n")


def _stage4_evidence_role_example_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "wid": rec.get("wid"),
        "title": str(rec.get("title") or "")[:72],
        "hit_quality_class": rec.get("hit_quality_class"),
        "mainline_term_count": rec.get("mainline_term_count"),
        "paper_evidence_role": rec.get("paper_evidence_role"),
        "paper_final_score_v2": rec.get("paper_final_score_v2"),
    }


def _print_stage4_evidence_role_summary(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    *,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    kept_recs = [by_wid[w] for w in selected_wids_set if w in by_wid and isinstance(by_wid.get(w), dict)]
    ctr: Counter = Counter()
    for r in kept_recs:
        ctr[str(r.get("paper_evidence_role") or "unknown")] += 1

    print("\n" + "-" * 80)
    print("[Stage4 evidence role summary]")
    print("-" * 80)
    print(f"paper_evidence_role_counter={dict(ctr)}")

    def _top_for(role: str, n: int = 5) -> List[Dict[str, Any]]:
        pool = [r for r in kept_recs if str(r.get("paper_evidence_role") or "") == role]
        pool.sort(key=lambda r: -float(r.get("paper_final_score_v2") or 0.0))
        return [_stage4_evidence_role_example_row(r) for r in pool[:n]]

    for label, role in (
        ("primary_evidence", "primary_evidence"),
        ("support_evidence", "support_evidence"),
        ("fringe_evidence", "fringe_evidence"),
    ):
        ex = _top_for(role, 5)
        print(f"--- top {label} (max 5 by paper_final_score_v2) ---")
        if not ex:
            print("  (empty)")
        else:
            for row in ex:
                print(f"  {row}")
    print("-" * 80 + "\n")


def _stage4_should_apply_prekept_fringe_guard(rec: Dict[str, Any]) -> bool:
    """极窄：single_hit_side + 无主轴 + 单 hit + fringe 预览/角色，用于 kept 前降权。"""
    hq = str(rec.get("hit_quality_class") or "").strip()
    if hq != "single_hit_side":
        return False
    try:
        if int(rec.get("mainline_term_count") or 0) != 0:
            return False
    except (TypeError, ValueError):
        return False
    try:
        if int(rec.get("hit_count") or 0) != 1:
            return False
    except (TypeError, ValueError):
        return False
    pv = str(rec.get("paper_evidence_role_preview") or "").strip()
    re = str(rec.get("paper_evidence_role") or "").strip()
    if pv == "fringe_candidate":
        return True
    if re == "fringe_evidence":
        return True
    return False


def _stage4_should_apply_prekept_mainline_keep_bonus(rec: Dict[str, Any]) -> bool:
    """主轴/多命中/质量优于 single_hit_side 时极轻加成（与 fringe guard 互斥由调用顺序保证）。"""
    try:
        if int(rec.get("mainline_term_count") or 0) >= 1:
            return True
    except (TypeError, ValueError):
        pass
    try:
        if int(rec.get("hit_count") or 0) >= 2:
            return True
    except (TypeError, ValueError):
        pass
    hq = str(rec.get("hit_quality_class") or "").strip()
    if hq in ("mainline_resonance", "mainline_plus_support", "single_hit_mainline"):
        return True
    pv = str(rec.get("paper_evidence_role_preview") or "").strip()
    if pv and pv != "fringe_candidate":
        return True
    re = str(rec.get("paper_evidence_role") or "").strip()
    if re in ("primary_evidence", "support_evidence"):
        return True
    return False


def _compute_stage4_compete_score_for_keep(rec: Dict[str, Any]) -> float:
    """
    kept 全局池竞争分：在 paper_final_score_v2 上做极窄乘子；不取代 Step2 v2 公式本身。
    在 prekept fringe/mainline 之后叠乘极窄的 community-aware 因子（仅 kept 竞争位点）。
    """
    rec["pre_kept_side_only_fringe_guard_applied"] = False
    rec["pre_kept_mainline_keep_bonus_applied"] = False
    rec["community_isolated_fringe_community_guard_applied"] = False
    rec["community_topic_cohesive_keep_bonus_applied"] = False
    if not STAGE4_ENABLE_EVIDENCE_SCORE_MIGRATION or not STAGE4_KEEP_COMPETE_ENABLED:
        base = float(rec.get("paper_final_score_v2") or rec.get("paper_score") or 0.0)
        rec["paper_compete_score_for_keep"] = float(base)
        return float(base)
    base = float(rec.get("paper_final_score_v2") or 0.0)
    s = float(base)
    if _stage4_should_apply_prekept_fringe_guard(rec):
        s = float(base * STAGE4_PREKEPTFRINGE_GUARD_FACTOR)
        rec["pre_kept_side_only_fringe_guard_applied"] = True
    elif _stage4_should_apply_prekept_mainline_keep_bonus(rec):
        s = float(base * STAGE4_PREKEPT_MAINLINE_KEEP_BONUS)
        rec["pre_kept_mainline_keep_bonus_applied"] = True
    if _stage4_should_apply_isolated_fringe_community_guard(rec):
        s = float(s * STAGE4_COMMUNITY_ISOLATED_FRINGE_FACTOR)
        rec["community_isolated_fringe_community_guard_applied"] = True
    elif (not rec.get("pre_kept_mainline_keep_bonus_applied")) and _stage4_should_apply_topic_cohesive_keep_bonus(
        rec
    ):
        s = float(s * STAGE4_COMMUNITY_COHESIVE_KEEP_BONUS)
        rec["community_topic_cohesive_keep_bonus_applied"] = True
    rec["paper_compete_score_for_keep"] = float(s)
    return float(s)


def _stage4_apply_keep_compete_reselect(
    by_wid: Dict[str, Dict[str, Any]],
    selected_baseline: Set[str],
    sorted_baseline: List[str],
    *,
    audit_print: bool = False,
    jd_align_map: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    在首次 attach 之后调用：用 compete 分从全 wid 池重选 GLOBAL_PAPER_LIMIT，并二次 attach 刷新 rank。
    若未启用 migration / keep compete，返回 None（不改变 selected）。
    """
    if not STAGE4_ENABLE_EVIDENCE_SCORE_MIGRATION or not STAGE4_KEEP_COMPETE_ENABLED:
        return None
    if audit_print:
        print(
            "[Stage4 keep compete dataflow] "
            "baseline_selected_source=top_GLOBAL_PAPER_LIMIT_by_paper_score_after_hierarchy "
            f"baseline_selected_size={int(len(selected_baseline))} "
            f"candidate_pool=all_keys_in_by_wid candidate_pool_size={int(len(by_wid))} "
            "new_selected_will_be=top_GLOBAL_PAPER_LIMIT_by_minus_paper_compete_score_for_keep "
            "prerequisite=caller_must_have_run_attach_stage4_score_migration_fields "
            "(paper_final_score_v2 + paper_evidence_role present before compete loop)"
        )
    candidate_pool_size_before_reselect = int(len(by_wid))
    selected_wids_set_size_before_reselect = int(len(selected_baseline))
    fringe_n = 0
    bonus_n = 0
    comm_iso_n = 0
    comm_coh_n = 0
    for _w, rec in by_wid.items():
        if not isinstance(rec, dict):
            continue
        _compute_stage4_compete_score_for_keep(rec)
        if rec.get("pre_kept_side_only_fringe_guard_applied"):
            fringe_n += 1
        if rec.get("pre_kept_mainline_keep_bonus_applied"):
            bonus_n += 1
        if rec.get("community_isolated_fringe_community_guard_applied"):
            comm_iso_n += 1
        if rec.get("community_topic_cohesive_keep_bonus_applied"):
            comm_coh_n += 1

    if audit_print:
        print(
            "[Stage4 keep compete compute] "
            f"per_wid=_compute_stage4_compete_score_for_keep rows={candidate_pool_size_before_reselect} "
            "when_migration_on_base_reads=paper_final_score_v2 "
            "guards_may_use=paper_evidence_role_and_paper_evidence_role_preview"
        )

    all_wids = list(by_wid.keys())
    new_sorted = sorted(
        all_wids,
        key=lambda w: -float((by_wid.get(w) or {}).get("paper_compete_score_for_keep") or 0.0),
    )[:GLOBAL_PAPER_LIMIT]
    new_set = set(new_sorted)
    selected_wids_set_size_after_reselect = int(len(new_set))

    for _w, rec in by_wid.items():
        if isinstance(rec, dict):
            rec["paper_global_pool_kept"] = str(_w) in new_set

    if audit_print:
        print(
            "[Stage4 keep compete attach] "
            "second_attach=_attach_stage4_score_migration_fields(by_wid,new_selected_set) "
            "purpose=refresh_within_kept_ranks_paper_old_score_rank_paper_final_score_v2_rank "
            "does_not_change_compete_sort_order_already_fixed_above"
        )

    _attach_stage4_score_migration_fields(by_wid, new_set, jd_align_map)

    added = new_set - selected_baseline
    removed = selected_baseline - new_set
    _n_pool = int(len(by_wid))

    return {
        "baseline_selected": set(selected_baseline),
        "new_selected_set": new_set,
        "new_sorted_wids": new_sorted,
        "candidate_pool_size_before_reselect": candidate_pool_size_before_reselect,
        "selected_wids_set_size_before_reselect": selected_wids_set_size_before_reselect,
        "selected_wids_set_size_after_reselect": selected_wids_set_size_after_reselect,
        "keep_compete_pool_len_vs_global_limit": f"len(by_wid)={_n_pool} GLOBAL_PAPER_LIMIT={GLOBAL_PAPER_LIMIT}",
        "changed_kept_count_expected_zero_when_pool_fits_entirely_under_limit": bool(_n_pool <= GLOBAL_PAPER_LIMIT),
        "changed_kept_count": int(len(added) + len(removed)),
        "added_to_kept_count": int(len(added)),
        "removed_from_kept_count": int(len(removed)),
        "prekept_fringe_guard_hit_count": int(fringe_n),
        "prekept_mainline_keep_bonus_hit_count": int(bonus_n),
        "community_isolated_fringe_guard_hit_count": int(comm_iso_n),
        "community_topic_cohesive_keep_bonus_hit_count": int(comm_coh_n),
        "added_wids": sorted(
            added,
            key=lambda w: -float((by_wid.get(w) or {}).get("paper_compete_score_for_keep") or 0.0),
        )[:12],
        "removed_wids": sorted(
            removed,
            key=lambda w: -float((by_wid.get(w) or {}).get("paper_score") or 0.0),
        )[:12],
        "sorted_baseline_head": list(sorted_baseline[:8]),
    }


def _print_stage4_keep_competition_migration_summary(
    stats: Optional[Dict[str, Any]],
    by_wid: Dict[str, Dict[str, Any]],
    *,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    print("\n" + "-" * 80)
    print("[Stage4 keep competition summary]")
    print("-" * 80)
    print(
        "keep_compete_score_source="
        "paper_final_score_v2 * (optional prekept_fringe_guard="
        f"{STAGE4_PREKEPTFRINGE_GUARD_FACTOR} | optional prekept_mainline_bonus="
        f"{STAGE4_PREKEPT_MAINLINE_KEEP_BONUS}) * (optional community_isolated="
        f"{STAGE4_COMMUNITY_ISOLATED_FRINGE_FACTOR} | optional community_cohesive_bonus="
        f"{STAGE4_COMMUNITY_COHESIVE_KEEP_BONUS})"
    )
    if not stats:
        print("keep_compete_reselect=skipped (migration off or STAGE4_KEEP_COMPETE_ENABLED=False)")
        print("-" * 80 + "\n")
        return
    ns = stats["new_selected_set"]
    print(
        f"candidate_pool_size_before_reselect={stats.get('candidate_pool_size_before_reselect')} "
        f"selected_wids_set_size_before_reselect={stats.get('selected_wids_set_size_before_reselect')} "
        f"selected_wids_set_size_after_reselect={stats.get('selected_wids_set_size_after_reselect')}"
    )
    if stats.get("keep_compete_pool_len_vs_global_limit"):
        print(f"keep_compete_pool_len_vs_global_limit: {stats.get('keep_compete_pool_len_vs_global_limit')}")
    if stats.get("changed_kept_count_expected_zero_when_pool_fits_entirely_under_limit") is not None:
        print(
            f"changed_kept_count_expected_zero_when_pool_fits_entirely_under_limit="
            f"{stats.get('changed_kept_count_expected_zero_when_pool_fits_entirely_under_limit')} "
            "(when True: len(by_wid)<=GLOBAL_PAPER_LIMIT so global baseline often equals full wid pool as a set)"
        )
    print(
        f"kept_papers_count={len(ns)} changed_kept_count={stats['changed_kept_count']} "
        f"added_to_kept_count={stats.get('added_to_kept_count')} "
        f"removed_from_kept_count={stats.get('removed_from_kept_count')} "
        f"prekept_fringe_guard_hit_count={stats['prekept_fringe_guard_hit_count']} "
        f"prekept_mainline_keep_bonus_hit_count={stats['prekept_mainline_keep_bonus_hit_count']} "
        f"community_isolated_fringe_guard_hit_count={stats.get('community_isolated_fringe_guard_hit_count', 0)} "
        f"community_topic_cohesive_keep_bonus_hit_count={stats.get('community_topic_cohesive_keep_bonus_hit_count', 0)}"
    )
    if stats.get("keep_compete_extra_limited_row_count") is not None:
        print(
            f"keep_compete_extra_limited_row_count={stats.get('keep_compete_extra_limited_row_count')} "
            f"STAGE4_KEEP_COMPETE_EXTRA_PER_TERM={STAGE4_KEEP_COMPETE_EXTRA_PER_TERM}"
        )
    print("--- added to kept (examples, up to 5) ---")
    for w in (stats.get("added_wids") or [])[:5]:
        print("  " + _stage4_keep_compete_example_line(str(w), by_wid.get(w)))
    print("--- removed from kept (examples, up to 5) ---")
    for w in (stats.get("removed_wids") or [])[:5]:
        print("  " + _stage4_keep_compete_example_line(str(w), by_wid.get(w)))
    print("-" * 80 + "\n")


def _print_stage4_keep_compete_pool_expansion_audit(
    by_wid: Dict[str, Dict[str, Any]],
    baseline_selected_wids: Set[str],
    term_capped_all_wids: Set[str],
    keep_compete_wids_term_local_cap: Set[str],
    keep_compete_wids_extra_post_cap: Set[str],
    extra_rows_slice_total: int,
    keep_compete_extra_row_materialized_count: int,
    get_term_meta: Any,
    *,
    audit_print: bool,
) -> None:
    """
    Step 2.2：区分 baseline_selected、keep compete 可见的 by_wid 全键、以及 term cap 集合；
    仅打印，不改分、不改集合。
    """
    if not audit_print:
        return
    n_by = int(len(by_wid))
    n_base = int(len(baseline_selected_wids))
    n_cap_union = int(len(term_capped_all_wids))
    n_tag_cap = int(len(keep_compete_wids_term_local_cap))
    n_tag_extra = int(len(keep_compete_wids_extra_post_cap))
    overlap_cap_extra = int(len(keep_compete_wids_term_local_cap & keep_compete_wids_extra_post_cap))
    extra_only_wids = keep_compete_wids_extra_post_cap - keep_compete_wids_term_local_cap
    in_pool_not_in_term_capped = {str(w) for w in by_wid.keys()} - term_capped_all_wids
    new_pool_minus_baseline = n_by - n_base
    new_pool_minus_term_capped = n_by - n_cap_union
    impossible_set_change = n_by <= GLOBAL_PAPER_LIMIT and n_base == n_by
    print("\n" + "=" * 80)
    print("[Stage4 keep compete pool expansion audit]")
    print("=" * 80)
    print(
        f"baseline_selected_wids_size={n_base} "
        f"keep_compete_candidate_pool_by_wid_all_keys={n_by} "
        f"by_wid_minus_baseline_count={new_pool_minus_baseline}"
    )
    print(
        f"candidate_pool_size_before_merge_unique_wids_term_local_cap_tag={n_tag_cap} "
        f"candidate_pool_size_after_merge_by_wid_keys={n_by} "
        f"(merge aggregates cap+extra rows into same wid dict)"
    )
    print(
        f"extra_rows_grounded_remainder_total_across_terms={extra_rows_slice_total} "
        f"extra_rows_materialized_to_compete_limited_list={keep_compete_extra_row_materialized_count} "
        f"STAGE4_KEEP_COMPETE_EXTRA_PER_TERM={STAGE4_KEEP_COMPETE_EXTRA_PER_TERM} "
        f"(<=0: include full remainder per term; >0: cap rows)"
    )
    print(
        f"term_capped_unique_union_size={n_cap_union} "
        f"wids_in_by_wid_not_in_term_capped_union={len(in_pool_not_in_term_capped)} "
        f"by_wid_minus_term_capped_union={new_pool_minus_term_capped}"
    )
    print(
        "pool_split unique_wid_tags: "
        f"baseline_from_sorted_wids={n_base} "
        f"extra_from_term_post_local_cap_grounded_unique={n_tag_extra} "
        f"overlap_term_cap_and_extra_tags={overlap_cap_extra}"
    )
    if extra_rows_slice_total > 0 and keep_compete_extra_row_materialized_count < extra_rows_slice_total:
        print(
            "[Stage4 keep compete pool expansion audit] note: remainder rows exist but some not materialized "
            "because STAGE4_KEEP_COMPETE_EXTRA_PER_TERM>0 caps how many tail rows merge into limited."
        )
    elif extra_rows_slice_total == 0:
        print(
            "[Stage4 keep compete pool expansion audit] note: no grounded remainder beyond local_cap "
            "(len(triples)==local_cap for all terms); cannot expand pool via extra slice."
        )
    if impossible_set_change:
        print(
            "[Stage4 keep compete pool expansion audit] membership_change_expectation: "
            f"len(by_wid)={n_by} <= GLOBAL_PAPER_LIMIT={GLOBAL_PAPER_LIMIT} and baseline uses top-{GLOBAL_PAPER_LIMIT} "
            "by paper_score — global selected set typically equals all by_wid keys; "
            "changed_kept_count may stay 0 unless compete ordering swaps marginal ranks only (set equality)."
        )
    if not in_pool_not_in_term_capped and n_tag_extra > 0:
        print(
            "[Stage4 keep compete pool expansion audit] note: extra-post-cap wids still overlap term_capped_union "
            "or all extra wids also have term_local_cap tag (multi-term); "
            "wids_in_by_wid_not_in_term_capped may be 0."
        )
    print("--- sample: up to 5 wids tagged extra-only (post_cap grounded, not in term_local_cap wid tag set) ---")
    samp = sorted(
        list(extra_only_wids),
        key=lambda w: -float((by_wid.get(w) or {}).get("paper_score") or 0.0),
    )[:5]
    if not samp:
        print("  (empty — extra tail wid set equals term_local_cap wid set or no extra rows)")
    for w in samp:
        rec = by_wid.get(w) or {}
        hits = rec.get("hits") or []
        h0 = hits[0] if hits and isinstance(hits[0], dict) else {}
        try:
            _tv = int(h0.get("vid"))
        except (TypeError, ValueError):
            _tv = 0
        _meta = get_term_meta(_tv) if _tv else {}
        src_term = str(_meta.get("term") or _meta.get("anchor") or _tv or "")
        try:
            g0 = float(h0.get("grounding") or 0.0)
        except (TypeError, ValueError):
            g0 = 0.0
        try:
            psf = float(rec.get("paper_score_final") or rec.get("paper_score") or 0.0)
        except (TypeError, ValueError):
            psf = 0.0
        try:
            po = float(rec.get("paper_old_score") or 0.0)
        except (TypeError, ValueError):
            po = 0.0
        try:
            v2 = float(rec.get("paper_final_score_v2") or 0.0)
        except (TypeError, ValueError):
            v2 = 0.0
        print(
            f"  wid={w} source_term={src_term!r} source_stage=keep_compete_extra_post_cap_grounded "
            f"grounding={g0:.4f} paper_score_final_or_paper_score={psf:.6f} paper_old_score={po:.6f} "
            f"paper_final_score_v2={v2:.6f}"
        )
    print("=" * 80 + "\n")


def _print_stage4_v2_activation_audit(
    by_wid: Dict[str, Dict[str, Any]],
    keep_stats: Optional[Dict[str, Any]],
    final_selected_wids: Set[str],
    *,
    audit_print: bool,
) -> None:
    """
    Step 2.1：只读审计——标明 v2 / evidence_role / compete 分在 kept 竞争前后的生效位置，
    不改变任何分数、集合或排序。
    """
    if not audit_print:
        return
    print("\n" + "=" * 80)
    print("[Stage4 v2 activation audit]")
    print("=" * 80)
    print(
        "[Stage4 v2 activation audit] field_lifecycle (first_write / first_decision_in_pipeline_order)"
    )
    print(
        "  paper_evidence_quality_score: "
        "first_write=_stage4_attach_paper_evidence_quality_fields (calls _compute_paper_evidence_quality_score); "
        "runs_after=per-wid materialize in main loop; before=hierarchy/sorted_wids baseline."
    )
    print(
        "  paper_old_score / paper_final_score_v2 / paper_evidence_role: "
        "first_write=_attach_stage4_score_migration_fields (all wids in by_wid); "
        "first_attach_in_main=before _stage4_apply_keep_compete_reselect (baseline selected_wids_set for within-kept ranks)."
    )
    print(
        "  paper_compete_score_for_keep: "
        "first_write=_compute_stage4_compete_score_for_keep (inside keep reselect loop); "
        "first_decision=global sort key for new_selected_set uses this field (desc)."
    )
    print(
        "  paper_global_pool_kept: "
        "first_write=_stage4_apply_keep_compete_reselect after new_set computed; "
        "reflects membership in post-compete top GLOBAL_PAPER_LIMIT."
    )
    print(
        "[Stage4 v2 activation audit] kept_compete_set_sources "
        "(baseline selected_wids / candidate_pool / new_selected_set)"
    )
    print(
        "  baseline_selected_wids: "
        "set of top GLOBAL_PAPER_LIMIT wids by paper_score (after hierarchy_consensus_bonus), "
        "see main sorted_wids = sort(by_wid.keys(), -paper_score)[:GLOBAL_PAPER_LIMIT]."
    )
    print(
        "  candidate_pool_size_before_reselect: "
        "len(by_wid) — all paper rows aggregated into Stage4 by_wid (not only baseline)."
    )
    print(
        "  new_selected_set: "
        "top GLOBAL_PAPER_LIMIT wids by paper_compete_score_for_keep descending over full by_wid; "
        "then second _attach_stage4_score_migration_fields(by_wid, new_set) refreshes within-kept ranks only."
    )
    reselect_executed = keep_stats is not None
    # 源码顺序：主链先 _attach_stage4_score_migration_fields 再 _stage4_apply_keep_compete_reselect（见下方调用处）
    migration_fields_attached_before_keep_compete = True
    evidence_role_assigned_before_keep_compete = True
    keep_compete_reads_v2_score = bool(
        STAGE4_ENABLE_EVIDENCE_SCORE_MIGRATION and STAGE4_KEEP_COMPETE_ENABLED
    )
    pool_sz = int(keep_stats["candidate_pool_size_before_reselect"]) if keep_stats else int(len(by_wid))
    base_sz = int(keep_stats["selected_wids_set_size_before_reselect"]) if keep_stats else int(len(final_selected_wids))
    new_sz = int(keep_stats["selected_wids_set_size_after_reselect"]) if keep_stats else int(len(final_selected_wids))
    changed = int(keep_stats.get("changed_kept_count") or 0) if keep_stats else 0
    effective_membership_change = changed > 0
    print("[Stage4 v2 activation audit] compact_flags")
    print(f"  migration_fields_attached_before_keep_compete={migration_fields_attached_before_keep_compete}")
    print(f"  evidence_role_assigned_before_keep_compete={evidence_role_assigned_before_keep_compete}")
    print(f"  keep_compete_reads_v2_score={keep_compete_reads_v2_score}")
    print(f"  keep_compete_reselect_executed={reselect_executed}")
    print(f"  keep_compete_candidate_pool_size={pool_sz}")
    print(f"  keep_compete_baseline_selected_size={base_sz}")
    print(f"  keep_compete_new_selected_size={new_sz}")
    print(f"  keep_compete_effective_reselect={effective_membership_change}")
    if keep_compete_reads_v2_score and reselect_executed:
        print(
            "[Stage4 v2 activation audit] conclusion: paper_final_score_v2 is read inside "
            "_compute_stage4_compete_score_for_keep BEFORE new_selected_set is built; "
            "changed_kept_count=0 only means baseline_set==new_set under current scores, not that v2 was ignored. "
            "paper_evidence_role is assigned in the prior _attach_stage4_score_migration_fields and is used by "
            "prekept fringe/mainline/community guards in the same compete pass."
        )
    elif STAGE4_ENABLE_EVIDENCE_SCORE_MIGRATION and not reselect_executed:
        print(
            "[Stage4 v2 activation audit] conclusion: keep compete path skipped; "
            "v2/role attach still runs for reporting; they do not alter global kept via compete sort."
        )
    else:
        print(
            "[Stage4 v2 activation audit] conclusion: migration or keep compete disabled; "
            "compete score falls back per _compute_stage4_compete_score_for_keep."
        )
    baseline_set: Set[str] = (
        set(keep_stats["baseline_selected"]) if keep_stats else set(final_selected_wids)
    )
    new_set: Set[str] = (
        set(keep_stats["new_selected_set"]) if keep_stats else set(final_selected_wids)
    )
    all_w = [w for w in by_wid.keys() if isinstance(by_wid.get(w), dict)]
    all_w.sort(
        key=lambda w: -float((by_wid.get(w) or {}).get("paper_compete_score_for_keep") or 0.0),
    )
    sample_wids = all_w[:5]
    print("[Stage4 v2 activation audit] top_sample_by_paper_compete_score_for_keep (max 5)")
    for w in sample_wids:
        rec = by_wid.get(w) or {}
        try:
            o = float(rec.get("paper_old_score") or 0.0)
        except (TypeError, ValueError):
            o = 0.0
        try:
            v2 = float(rec.get("paper_final_score_v2") or 0.0)
        except (TypeError, ValueError):
            v2 = 0.0
        try:
            ck = float(rec.get("paper_compete_score_for_keep") or 0.0)
        except (TypeError, ValueError):
            ck = 0.0
        pgk = bool(rec.get("paper_global_pool_kept"))
        in_base = w in baseline_set
        in_new = w in new_set
        print(
            f"  wid={w} paper_old_score={o:.6f} paper_final_score_v2={v2:.6f} "
            f"paper_compete_score_for_keep={ck:.6f} paper_global_pool_kept={pgk} "
            f"in_baseline_selected={in_base} in_new_selected_set={in_new}"
        )
    print("=" * 80 + "\n")


def _print_stage4_mainline_term_audit(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    get_term_meta: Any,
    *,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    kept_multi: List[Tuple[str, Dict[str, Any], int]] = []
    for w in selected_wids_set:
        rec = by_wid.get(w)
        if not isinstance(rec, dict):
            continue
        try:
            hc = int(rec.get("hit_count") or 0)
        except (TypeError, ValueError):
            hc = 0
        if hc >= 2:
            kept_multi.append((str(w), rec, hc))
    kept_multi.sort(key=lambda x: -x[2])
    top_n = max(8, int(STAGE4_MAINLINE_TERM_AUDIT_TOP_MULTI_HIT))
    kept_multi = kept_multi[:top_n]

    print("\n" + "=" * 80)
    print("[Stage4 mainline term audit] top multi-hit papers in kept pool")
    print("=" * 80)
    if not kept_multi:
        print("(no multi-hit papers in kept set)")
        print("=" * 80 + "\n")
        return
    for i, (wid, rec, hc) in enumerate(kept_multi, 1):
        hits = [h for h in (rec.get("hits") or []) if isinstance(h, dict)]
        mains, sides = _stage4_audit_resolve_mainline_side_lists(hits, get_term_meta)
        htr = [str(h.get("role") or "") for h in hits]
        htrp = rec.get("hit_term_retrieval_roles_prep")
        if not isinstance(htrp, list):
            htrp = []
        hpp = [str(h.get("parent_primary") or "") for h in hits]
        print(f"--- #{i} wid={wid!r} hit_count={hc} ---")
        print(f"  hit_terms={rec.get('hit_terms')!r}")
        print(f"  hit_term_roles_stage4_hit_dict={htr!r}")
        print(f"  hit_term_retrieval_roles_prep_meta={htrp!r}")
        print(f"  hit_term_parent_primary={hpp!r}")
        print(f"  mainline_terms_resolved_mainline_like={mains!r}")
        print(f"  side_terms_resolved={sides!r}")
        print(
            f"  mainline_term_count={rec.get('mainline_term_count')!r} "
            f"side_term_count={rec.get('side_term_count')!r}"
        )
        print(f"  hit_quality_class={rec.get('hit_quality_class')!r}")
        print(f"  paper_hit_quality_detail={rec.get('paper_hit_quality_detail')!r}")
    print("=" * 80 + "\n")


def _print_stage4_multihit_classification_audit(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    *,
    audit_print: bool,
) -> None:
    """对仍被判为 side_only_or_accidental_multi_hit 的多命中论文打印结构化原因（flags）。"""
    if not audit_print:
        return
    kept = [w for w in selected_wids_set if w in by_wid]
    bad: List[Tuple[str, Dict[str, Any]]] = []
    for w in kept:
        rec = by_wid.get(w) or {}
        if not isinstance(rec, dict):
            continue
        try:
            hc = int(rec.get("hit_count") or 0)
        except (TypeError, ValueError):
            hc = 0
        if hc < 2:
            continue
        if str(rec.get("hit_quality_class") or "").strip() != "side_only_or_accidental_multi_hit":
            continue
        bad.append((str(w), rec))
    print("\n" + "-" * 80)
    print("[Stage4 multi-hit classification audit] side_only_or_accidental_multi_hit (with reason flags)")
    print("-" * 80)
    if not bad:
        print("(none in kept pool)")
        print("-" * 80 + "\n")
        return
    for i, (wid, rec) in enumerate(bad[:25], 1):
        det = rec.get("paper_hit_quality_detail") or {}
        print(
            f"  [{i}] wid={wid!r} hit_terms={rec.get('hit_terms')!r} "
            f"mainline_like_raw={det.get('mainline_like_hits_raw_count')!r} "
            f"axis_parent_primary_boost_applied={det.get('axis_parent_primary_boost_applied')!r} "
            f"coherence_reason={rec.get('coherence_reason')!r}"
        )
    print("-" * 80 + "\n")


def _print_stage4_kept_pool_quality_summary(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    *,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    kept = [w for w in selected_wids_set if w in by_wid]
    hqc: Counter = Counter()
    role_ctr: Counter = Counter()
    ml_pos = 0
    mh = 0
    sh_side = 0
    for w in kept:
        r = by_wid.get(w) or {}
        hqc[str(r.get("hit_quality_class") or "unknown")] += 1
        role_ctr[str(r.get("paper_evidence_role") or "unknown")] += 1
        try:
            if int(r.get("mainline_term_count") or 0) > 0:
                ml_pos += 1
        except (TypeError, ValueError):
            pass
        try:
            if int(r.get("hit_count") or 0) >= 2:
                mh += 1
        except (TypeError, ValueError):
            pass
        if str(r.get("hit_quality_class") or "") == "single_hit_side":
            sh_side += 1
    print("\n" + "-" * 80)
    print("[Stage4 kept pool quality summary]")
    print("-" * 80)
    print(
        f"kept_papers_count={len(kept)} hit_quality_counter={dict(hqc)} "
        f"paper_evidence_role_counter={dict(role_ctr)} "
        f"mainline_term_count_gt0={ml_pos} multi_hit_papers_count={mh} "
        f"single_hit_side_kept_count={sh_side}"
    )
    print("-" * 80 + "\n")


# --- Stage4 job-axis coverage audit（旁路观测；不参与排序/分数/gate）---
STAGE4_JOB_AXIS_COVERAGE_AUDIT = os.environ.get("STAGE4_JOB_AXIS_COVERAGE_AUDIT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
STAGE4_JOB_AXIS_COVERAGE_TOP_ROWS = int(os.environ.get("STAGE4_JOB_AXIS_COVERAGE_TOP_ROWS", "15"))

STAGE4_JOB_AXIS_ORDER: Tuple[str, ...] = (
    "control_core",
    "dynamics_kinematics",
    "planning_trajectory",
    "optimal_control",
    "estimation",
    "simulation_sim2real",
    "generic_robot_autonomous_shell",
)

# 旁路审计：主证据候选分（不参与排序/角色/gate；仅日志与字段观测）
STAGE4_PRIMARY_CANDIDATE_AUDIT_TOP_K = int(os.environ.get("STAGE4_PRIMARY_CANDIDATE_AUDIT_TOP_K", "12"))
# Step2：very narrow support->primary（仅改 evidence role；不改 v2/排序）
STAGE4_PRIMARY_PROMOTION_REL_GAP = float(os.environ.get("STAGE4_PRIMARY_PROMOTION_REL_GAP", "0.022"))
STAGE4_PRIMARY_PROMOTION_ABS_FLOOR = float(os.environ.get("STAGE4_PRIMARY_PROMOTION_ABS_FLOOR", "0.735"))
STAGE4_PRIMARY_PROMOTION_MAX = int(os.environ.get("STAGE4_PRIMARY_PROMOTION_MAX", "5"))
# Step3：极轻量 primary-aware micro rerank（paper_final_score_v3 = v2 * factor；不改 v2 原值）
STAGE4_PRIMARY_AWARE_RERANK_ENABLED = os.environ.get("STAGE4_PRIMARY_AWARE_RERANK_ENABLED", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
STAGE4_PRIMARY_AWARE_PROMOTED_BASE = float(os.environ.get("STAGE4_PRIMARY_AWARE_PROMOTED_BASE", "1.028"))
STAGE4_PRIMARY_AWARE_PROMOTED_MAX = float(os.environ.get("STAGE4_PRIMARY_AWARE_PROMOTED_MAX", "1.040"))
STAGE4_PRIMARY_AWARE_NATIVE_BASE = float(os.environ.get("STAGE4_PRIMARY_AWARE_NATIVE_BASE", "1.010"))
STAGE4_PRIMARY_AWARE_NATIVE_MAX = float(os.environ.get("STAGE4_PRIMARY_AWARE_NATIVE_MAX", "1.018"))
STAGE4_PRIMARY_AWARE_NATIVE_PCS_MIN = float(os.environ.get("STAGE4_PRIMARY_AWARE_NATIVE_PCS_MIN", "0.765"))
_STAGE4_PRIMARY_PROMOTION_ALLOWED_HQ: frozenset = frozenset(
    {"mainline_plus_support", "single_hit_mainline", "mainline_resonance"}
)
_STAGE4_PRIMARY_CAND_HIT_QUALITY_WEIGHT: Dict[str, float] = {
    "mainline_resonance": 1.0,
    "mainline_plus_support": 0.95,
    "single_hit_mainline": 0.82,
    "single_hit_side": 0.22,
    "side_only_or_accidental_multi_hit": 0.32,
}


def _stage4_primary_candidate_axis_weight(rec: Dict[str, Any]) -> float:
    """复用已有 job_axis_primary_label（若存在）；不调用 infer，避免在审计中改写路径。"""
    prim = str(rec.get("job_axis_primary_label") or "").strip()
    if not prim or prim == "unclassified":
        return 0.0
    if prim == "generic_robot_autonomous_shell":
        return -0.12
    return 0.10


def _compute_paper_primary_candidate_score(rec: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    旁路审计分：衡量「像不像主证据候选」；不参与任何裁决。
    返回 (0~1 分数, 分量摘要 dict)。
    """
    parts: Dict[str, Any] = {}
    try:
        ml = int(rec.get("mainline_term_count") or 0)
    except (TypeError, ValueError):
        ml = 0
    ml = max(0, min(6, ml))
    parts["mainline_norm"] = ml / 6.0

    hq = str(rec.get("hit_quality_class") or "").strip()
    parts["hq_w"] = float(_STAGE4_PRIMARY_CAND_HIT_QUALITY_WEIGHT.get(hq, 0.45))

    parts["jd_overlap"] = 1.0 if rec.get("paper_jd_topic_profile_overlap_flag") else 0.0
    parts["nh_flag"] = 1.0 if rec.get("paper_topic_neighborhood_hit_flag") else 0.0
    parts["cohesive"] = 1.0 if rec.get("paper_topic_cohesive_flag") else 0.0

    det = rec.get("hierarchy_consensus_detail") or {}
    try:
        hcons = float(det.get("hierarchy_consensus") or 0.0)
    except (TypeError, ValueError):
        hcons = 0.0
    try:
        hc2 = rec.get("hierarchy_consensus")
        if hc2 is not None:
            hcons = max(hcons, float(hc2))
    except (TypeError, ValueError):
        pass
    parts["hcons_norm"] = float(min(1.0, max(0.0, hcons / 0.20)))

    try:
        pcs = float(rec.get("paper_topic_community_score") or 0.0)
    except (TypeError, ValueError):
        pcs = 0.0
    parts["community_norm"] = float(min(1.0, max(0.0, pcs)))

    try:
        scount = int(_stage4_count_cohesion_strong_signals(rec))
    except Exception:
        scount = 0
    parts["strong_signals_norm"] = float(min(1.0, max(0.0, scount / 5.0)))

    parts["axis_w"] = float(_stage4_primary_candidate_axis_weight(rec))

    try:
        eq = float(rec.get("paper_evidence_quality_score") or 0.0)
    except (TypeError, ValueError):
        eq = 0.0
    parts["eq_norm"] = float(min(1.0, max(0.0, eq)))

    try:
        v2 = float(rec.get("paper_final_score_v2") or 0.0)
    except (TypeError, ValueError):
        v2 = 0.0
    parts["v2_norm"] = float(min(1.0, max(0.0, v2 / 0.25)))

    try:
        hc = int(rec.get("hit_count") or 0)
    except (TypeError, ValueError):
        hc = 0
    parts["multihit_norm"] = float(min(1.0, max(0.0, min(hc, 5) / 5.0)))

    raw = (
        0.17 * parts["mainline_norm"]
        + 0.20 * parts["hq_w"]
        + 0.09 * parts["jd_overlap"]
        + 0.06 * parts["nh_flag"]
        + 0.07 * parts["cohesive"]
        + 0.10 * parts["hcons_norm"]
        + 0.07 * parts["community_norm"]
        + 0.06 * parts["strong_signals_norm"]
        + parts["axis_w"]
        + 0.05 * parts["eq_norm"]
        + 0.04 * parts["v2_norm"]
        + 0.04 * parts["multihit_norm"]
    )
    score = float(max(0.0, min(1.0, raw)))
    parts["raw_before_clip"] = float(raw)
    return score, parts


def _paper_primary_candidate_compact_row(rec: Dict[str, Any], parts: Optional[Dict[str, Any]] = None) -> str:
    if parts is None:
        sc, comp = _compute_paper_primary_candidate_score(rec)
    else:
        comp = parts
        try:
            sc = float(rec.get("paper_primary_candidate_score") or 0.0)
        except (TypeError, ValueError):
            sc, comp = _compute_paper_primary_candidate_score(rec)
    wid = rec.get("wid")
    title = str(rec.get("title") or "")[:88]
    reasons = comp or {}
    comp_s = ",".join(
        f"{k}={reasons.get(k)}"
        for k in (
            "mainline_norm",
            "hq_w",
            "jd_overlap",
            "hcons_norm",
            "axis_w",
            "v2_norm",
        )
        if k in reasons
    )
    return (
        f"wid={wid!r} title={title!r} paper_evidence_role={rec.get('paper_evidence_role')!r} "
        f"hit_quality_class={rec.get('hit_quality_class')!r} mainline_term_count={rec.get('mainline_term_count')} "
        f"job_axis_primary_label={rec.get('job_axis_primary_label')!r} paper_final_score_v2={rec.get('paper_final_score_v2')} "
        f"paper_primary_candidate_score={sc:.6f} components=({comp_s})"
    )


def _stage4_ensure_job_axis_labels_for_kept(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    jd_align_map: Optional[Dict[str, float]],
) -> None:
    """升格依赖 job_axis_primary_label：缺失则复用既有 infer 旁路写入（不额外打轴审计块）。"""
    need = False
    for w in selected_wids_set:
        r = by_wid.get(w)
        if not isinstance(r, dict):
            continue
        if not str(r.get("job_axis_primary_label") or "").strip():
            need = True
            break
    if need:
        _attach_job_axis_audit_to_kept_papers(by_wid, selected_wids_set, jd_align_map)


def _stage4_compute_and_attach_primary_candidate_scores(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
) -> None:
    """与 primary candidate audit 同口径写入 paper_primary_candidate_score（供升格与日志复用）。"""
    for w in selected_wids_set:
        r = by_wid.get(w)
        if not isinstance(r, dict):
            continue
        sc, parts = _compute_paper_primary_candidate_score(r)
        r["paper_primary_candidate_score"] = float(sc)
        r["paper_primary_candidate_audit_components"] = dict(parts)


def _should_promote_support_to_primary(
    rec: Dict[str, Any],
    parts: Dict[str, Any],
    *,
    score_threshold: float,
) -> Tuple[bool, List[str]]:
    """Very narrow：仅 support_evidence；不改分，只判是否满足升格门。"""
    if str(rec.get("paper_evidence_role") or "").strip() != "support_evidence":
        return False, ["not_support_evidence"]
    hq = str(rec.get("hit_quality_class") or "").strip()
    if hq not in _STAGE4_PRIMARY_PROMOTION_ALLOWED_HQ:
        return False, [f"hit_quality_class_not_eligible:{hq}"]
    try:
        ml = int(rec.get("mainline_term_count") or 0)
    except (TypeError, ValueError):
        ml = 0
    if ml < 1:
        return False, ["mainline_term_count_lt_1"]
    if not rec.get("paper_jd_topic_profile_overlap_flag"):
        return False, ["jd_topic_profile_overlap_required"]
    ax = str(rec.get("job_axis_primary_label") or "").strip()
    if not ax or ax in ("unclassified", "generic_robot_autonomous_shell"):
        return False, [f"job_axis_not_substantive:{ax or 'empty'}"]
    try:
        cs = float(rec.get("paper_primary_candidate_score") or 0.0)
    except (TypeError, ValueError):
        cs = 0.0
    if cs < float(score_threshold):
        return False, [f"primary_candidate_score<{score_threshold:.6f} (got {cs:.6f})"]
    nh = float(parts.get("nh_flag") or 0.0)
    hcn = float(parts.get("hcons_norm") or 0.0)
    coh = float(parts.get("cohesive") or 0.0)
    try:
        nhit = int(rec.get("hit_count") or 0)
    except (TypeError, ValueError):
        nhit = 0
    struct_ok = (
        nh >= 1.0
        or hcn >= 0.30
        or coh >= 1.0
        or nhit >= 2
        or hq in ("mainline_plus_support", "mainline_resonance")
    )
    if not struct_ok:
        return False, ["structure_gate_failed(nh|hcons_norm>=0.30|cohesive|hit_count>=2|mainline_plus_support|mainline_resonance)"]
    return True, ["narrow_gate_pass"]


def _print_stage4_primary_promotion_summary(
    *,
    ctr_before: Counter,
    ctr_after: Counter,
    promoted_rows: List[Dict[str, Any]],
    near_miss_rows: List[Dict[str, Any]],
    prim_anchor_max: float,
    score_threshold: float,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    print("\n" + "-" * 80)
    print("[Stage4 primary promotion summary]")
    print("-" * 80)
    print(
        f"promotion_total={len(promoted_rows)} "
        f"prim_anchor_candidate_score_max={prim_anchor_max:.6f} "
        f"promotion_score_threshold=max(abs_floor={STAGE4_PRIMARY_PROMOTION_ABS_FLOOR:.4f}, "
        f"anchor_max_minus_rel_gap={prim_anchor_max:.6f}-{STAGE4_PRIMARY_PROMOTION_REL_GAP:.4f})={score_threshold:.6f} "
        f"max_promotions_cap={STAGE4_PRIMARY_PROMOTION_MAX}"
    )
    print(f"paper_evidence_role_counter_before_promotion={dict(ctr_before)}")
    print(f"paper_evidence_role_counter_after_promotion={dict(ctr_after)}")
    print(f"primary_evidence_total_after={ctr_after.get('primary_evidence', 0)}")
    print("--- promoted papers ---")
    if not promoted_rows:
        print("  (none)")
    else:
        for row in promoted_rows:
            print(
                f"  wid={row.get('wid')!r} title={str(row.get('title') or '')[:90]!r} "
                f"old_role={row.get('old_role')!r} new_role={row.get('new_role')!r} "
                f"hit_quality_class={row.get('hit_quality_class')!r} mainline_term_count={row.get('mainline_term_count')} "
                f"job_axis_primary_label={row.get('job_axis_primary_label')!r} "
                f"paper_final_score_v2={row.get('paper_final_score_v2')} "
                f"paper_primary_candidate_score={row.get('paper_primary_candidate_score')} "
                f"promotion_reason={row.get('promotion_reason')}"
            )
    print("--- near-miss support (not promoted, top by paper_primary_candidate_score) ---")
    if not near_miss_rows:
        print("  (none)")
    else:
        for row in near_miss_rows:
            print(
                f"  wid={row.get('wid')!r} score={row.get('paper_primary_candidate_score')} "
                f"hit_quality_class={row.get('hit_quality_class')!r} "
                f"why_not={row.get('why_not')}"
            )
    print("-" * 80 + "\n")


def _apply_stage4_support_to_primary_promotion(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    jd_align_map: Optional[Dict[str, float]],
    *,
    audit_print: bool,
) -> Dict[str, Any]:
    """
    Very narrow：support_evidence -> primary_evidence；不改 v2/排序。
    依赖已写入的 paper_primary_candidate_score / audit_components。
    """
    kept_recs: List[Dict[str, Any]] = [
        by_wid[w] for w in selected_wids_set if w in by_wid and isinstance(by_wid.get(w), dict)
    ]
    for r in kept_recs:
        r["paper_primary_promoted"] = False
        r.pop("paper_primary_promotion_reason", None)
        r.pop("paper_primary_promotion_score", None)

    ctr_before: Counter = Counter(str(r.get("paper_evidence_role") or "unknown") for r in kept_recs)

    prim_anchor_max = max(
        (
            float(r.get("paper_primary_candidate_score") or 0.0)
            for r in kept_recs
            if str(r.get("paper_evidence_role") or "").strip() == "primary_evidence"
        ),
        default=0.0,
    )
    out: Dict[str, Any] = {
        "ctr_before": ctr_before,
        "ctr_after": ctr_before,
        "promoted": [],
        "near_miss": [],
        "prim_anchor_max": float(prim_anchor_max),
        "score_threshold": 0.0,
    }
    if prim_anchor_max <= 0.0:
        _print_stage4_primary_promotion_summary(
            ctr_before=ctr_before,
            ctr_after=ctr_before,
            promoted_rows=[],
            near_miss_rows=[],
            prim_anchor_max=0.0,
            score_threshold=0.0,
            audit_print=audit_print,
        )
        return out

    score_threshold = float(
        max(STAGE4_PRIMARY_PROMOTION_ABS_FLOOR, prim_anchor_max - STAGE4_PRIMARY_PROMOTION_REL_GAP)
    )
    out["score_threshold"] = score_threshold

    scored: List[Tuple[float, Dict[str, Any], bool, List[str]]] = []
    for r in kept_recs:
        if str(r.get("paper_evidence_role") or "").strip() != "support_evidence":
            continue
        parts = r.get("paper_primary_candidate_audit_components")
        if not isinstance(parts, dict):
            _, parts = _compute_paper_primary_candidate_score(r)
            r["paper_primary_candidate_audit_components"] = dict(parts)
        ok, why = _should_promote_support_to_primary(r, parts, score_threshold=score_threshold)
        try:
            cs = float(r.get("paper_primary_candidate_score") or 0.0)
        except (TypeError, ValueError):
            cs = 0.0
        scored.append((cs, r, ok, why))

    scored.sort(key=lambda t: -t[0])
    promoted_rows: List[Dict[str, Any]] = []
    promoted_wids: Set[str] = set()

    for cs, r, ok, why in scored:
        if not ok:
            continue
        if len(promoted_rows) >= int(STAGE4_PRIMARY_PROMOTION_MAX):
            break
        old_role = str(r.get("paper_evidence_role") or "")
        r["paper_evidence_role"] = "primary_evidence"
        r["paper_primary_promoted"] = True
        pr = list(why) + [
            f"score_threshold={score_threshold:.6f}",
            f"prim_anchor_max={prim_anchor_max:.6f}",
        ]
        r["paper_primary_promotion_reason"] = pr
        r["paper_primary_promotion_score"] = float(cs)
        w = r.get("wid")
        if w is not None:
            promoted_wids.add(str(w))
        promoted_rows.append(
            {
                "wid": w,
                "title": r.get("title"),
                "old_role": old_role,
                "new_role": "primary_evidence",
                "hit_quality_class": r.get("hit_quality_class"),
                "mainline_term_count": r.get("mainline_term_count"),
                "job_axis_primary_label": r.get("job_axis_primary_label"),
                "paper_final_score_v2": r.get("paper_final_score_v2"),
                "paper_primary_candidate_score": cs,
                "promotion_reason": pr,
            }
        )

    ctr_after: Counter = Counter(str(r.get("paper_evidence_role") or "unknown") for r in kept_recs)
    out["ctr_after"] = ctr_after
    out["promoted"] = promoted_rows

    near_miss_rows: List[Dict[str, Any]] = []
    for cs, r, ok, why in sorted(scored, key=lambda t: -t[0]):
        w = r.get("wid")
        if w is not None and str(w) in promoted_wids:
            continue
        near_miss_rows.append(
            {
                "wid": w,
                "paper_primary_candidate_score": cs,
                "hit_quality_class": r.get("hit_quality_class"),
                "why_not": why if not ok else ["eligible_but_cap_or_order"],
            }
        )
        if len(near_miss_rows) >= 5:
            break

    out["near_miss"] = near_miss_rows
    _print_stage4_primary_promotion_summary(
        ctr_before=ctr_before,
        ctr_after=ctr_after,
        promoted_rows=promoted_rows,
        near_miss_rows=near_miss_rows,
        prim_anchor_max=float(prim_anchor_max),
        score_threshold=score_threshold,
        audit_print=audit_print,
    )
    return out


def _stage4_paper_effective_sort_score(rec: Optional[Dict[str, Any]]) -> float:
    """Step3 后 term 内 / 全局重排：优先 paper_final_score_v3（未写入时退回 v2）。"""
    if not isinstance(rec, dict):
        return 0.0
    if rec.get("paper_final_score_v3") is not None:
        try:
            return float(rec.get("paper_final_score_v3") or 0.0)
        except (TypeError, ValueError):
            pass
    return float(rec.get("paper_final_score_v2") or 0.0)


def _compute_stage4_primary_aware_rerank_factor(rec: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    极窄乘子：仅 primary_evidence；promoted 优先；原生高 candidate 次之；不 penalize。
    """
    if str(rec.get("paper_evidence_role") or "").strip() != "primary_evidence":
        return 1.0, ["not_primary_evidence"]
    hq = str(rec.get("hit_quality_class") or "").strip()
    if hq == "single_hit_side":
        return 1.0, ["blocked_single_hit_side"]
    if str(rec.get("paper_evidence_role_preview") or "").strip() == "fringe_candidate":
        return 1.0, ["blocked_fringe_preview"]
    if str(rec.get("job_axis_primary_label") or "").strip() == "generic_robot_autonomous_shell":
        return 1.0, ["blocked_generic_shell_axis"]
    try:
        pcs = float(rec.get("paper_primary_candidate_score") or 0.0)
    except (TypeError, ValueError):
        pcs = 0.0

    promoted = bool(rec.get("paper_primary_promoted"))
    if promoted:
        extra = min(
            float(STAGE4_PRIMARY_AWARE_PROMOTED_MAX - STAGE4_PRIMARY_AWARE_PROMOTED_BASE),
            max(0.0, (pcs - 0.74) * 0.09),
        )
        f = float(min(STAGE4_PRIMARY_AWARE_PROMOTED_MAX, STAGE4_PRIMARY_AWARE_PROMOTED_BASE + extra))
        return f, ["promoted_primary_micro_boost", f"pcs={pcs:.4f}"]

    if pcs >= STAGE4_PRIMARY_AWARE_NATIVE_PCS_MIN:
        span = float(STAGE4_PRIMARY_AWARE_NATIVE_MAX - STAGE4_PRIMARY_AWARE_NATIVE_BASE)
        extra = min(span, max(0.0, (pcs - STAGE4_PRIMARY_AWARE_NATIVE_PCS_MIN) * 0.15))
        f = float(min(STAGE4_PRIMARY_AWARE_NATIVE_MAX, STAGE4_PRIMARY_AWARE_NATIVE_BASE + extra))
        return f, ["native_primary_candidate_micro_boost", f"pcs={pcs:.4f}"]

    return 1.0, ["primary_no_micro_boost"]


def _apply_stage4_primary_aware_rerank(
    by_wid: Dict[str, Dict[str, Any]],
    sorted_wids: List[str],
    selected_wids_set: Set[str],
    *,
    audit_print: bool,
) -> Tuple[List[str], Dict[str, Any]]:
    """写入 v3/factor；按 v3 重排 sorted_wids（禁用时 v3=v2、顺序不变）。"""
    for w, r in by_wid.items():
        if not isinstance(r, dict):
            continue
        try:
            v2 = float(r.get("paper_final_score_v2") or 0.0)
        except (TypeError, ValueError):
            v2 = 0.0
        if not STAGE4_PRIMARY_AWARE_RERANK_ENABLED:
            r["paper_primary_aware_rerank_factor"] = 1.0
            r["paper_final_score_v3"] = float(v2)
            r["paper_primary_aware_rerank_reasons"] = ["rerank_disabled_v3_equals_v2"]
            continue
        if w not in selected_wids_set:
            r["paper_primary_aware_rerank_factor"] = 1.0
            r["paper_final_score_v3"] = float(v2)
            r["paper_primary_aware_rerank_reasons"] = ["outside_selected_kept_pool"]
            continue
        fac, rs = _compute_stage4_primary_aware_rerank_factor(r)
        r["paper_primary_aware_rerank_factor"] = float(fac)
        r["paper_primary_aware_rerank_reasons"] = list(rs)
        r["paper_final_score_v3"] = float(v2 * fac)

    kept = [w for w in selected_wids_set if w in by_wid]
    order_v2 = sorted(kept, key=lambda w: -float((by_wid.get(w) or {}).get("paper_final_score_v2") or 0.0))
    order_v3 = sorted(kept, key=lambda w: -_stage4_paper_effective_sort_score(by_wid.get(w)))
    rv3 = {w: i + 1 for i, w in enumerate(order_v3)}
    for w in kept:
        br = by_wid.get(w)
        if isinstance(br, dict):
            br["paper_final_score_v3_rank"] = int(rv3[w])

    new_sorted = sorted(
        list(sorted_wids),
        key=lambda w: -_stage4_paper_effective_sort_score(by_wid.get(w) or {}),
    )

    changed_top = {10: 0, 20: 0}
    for k in (10, 20):
        for i in range(min(k, len(order_v2), len(order_v3))):
            if order_v2[i] != order_v3[i]:
                changed_top[k] += 1

    top_boosted = sorted(
        (by_wid[w] for w in kept if float((by_wid.get(w) or {}).get("paper_primary_aware_rerank_factor") or 1.0) > 1.0),
        key=lambda r: -float(r.get("paper_primary_aware_rerank_factor") or 1.0),
    )[:12]

    rank_moves: List[Dict[str, Any]] = []
    r2 = {w: i + 1 for i, w in enumerate(order_v2)}
    for w in kept:
        o2 = r2.get(w)
        o3 = rv3.get(w)
        if o2 != o3 and o2 is not None and o3 is not None:
            rank_moves.append(
                {
                    "wid": w,
                    "title": (by_wid.get(w) or {}).get("title"),
                    "rank_v2": o2,
                    "rank_v3": o3,
                    "paper_final_score_v2": (by_wid.get(w) or {}).get("paper_final_score_v2"),
                    "factor": (by_wid.get(w) or {}).get("paper_primary_aware_rerank_factor"),
                    "paper_final_score_v3": (by_wid.get(w) or {}).get("paper_final_score_v3"),
                }
            )
    rank_moves.sort(key=lambda x: min(x.get("rank_v2") or 999, x.get("rank_v3") or 999))
    rank_moves = rank_moves[:20]

    stats: Dict[str, Any] = {
        "enabled": bool(STAGE4_PRIMARY_AWARE_RERANK_ENABLED),
        "total_kept": len(kept),
        "changed_top10_positions": changed_top[10],
        "changed_top20_positions": changed_top[20],
        "promoted_boosted_count": sum(
            1
            for w in kept
            if bool((by_wid.get(w) or {}).get("paper_primary_promoted"))
            and float((by_wid.get(w) or {}).get("paper_primary_aware_rerank_factor") or 1.0) > 1.0
        ),
        "order_v2": order_v2,
        "order_v3": order_v3,
        "top_boosted": top_boosted,
        "rank_moves": rank_moves,
        "by_wid_for_print": by_wid,
    }
    _print_stage4_primary_aware_rerank_summary(stats, audit_print=audit_print)
    return new_sorted, stats


def _print_stage4_primary_aware_rerank_summary(stats: Dict[str, Any], *, audit_print: bool) -> None:
    if not audit_print:
        return
    order_v2: List[str] = list(stats.get("order_v2") or [])
    order_v3: List[str] = list(stats.get("order_v3") or [])
    top10_changed = any(
        order_v2[i] != order_v3[i] for i in range(min(10, len(order_v2), len(order_v3)))
    )
    print("\n" + "-" * 80)
    print("[Stage4 primary-aware rerank summary]")
    print("-" * 80)
    print(
        f"rerank_enabled={stats.get('enabled')} total_papers_kept={stats.get('total_kept')} "
        f"changed_top10_position_mismatches={stats.get('changed_top10_positions')} "
        f"changed_top20_position_mismatches={stats.get('changed_top20_positions')} "
        f"factor_clip_range=[1.0,{max(STAGE4_PRIMARY_AWARE_PROMOTED_MAX, STAGE4_PRIMARY_AWARE_NATIVE_MAX):.4f}] "
        f"primary_promoted_boosted_count={stats.get('promoted_boosted_count')} "
        f"top10_order_changed={top10_changed}"
    )
    print("note=only_boost_no_explicit_penalize; v2_unchanged; v3=v2*factor")
    print("--- top boosted papers (factor>1) ---")
    for r in stats.get("top_boosted") or []:
        if not isinstance(r, dict):
            continue
        print(
            f"  wid={r.get('wid')!r} title={str(r.get('title') or '')[:88]!r} "
            f"paper_evidence_role={r.get('paper_evidence_role')!r} paper_primary_promoted={r.get('paper_primary_promoted')} "
            f"hit_quality_class={r.get('hit_quality_class')!r} "
            f"paper_final_score_v2={r.get('paper_final_score_v2')} "
            f"paper_primary_aware_rerank_factor={r.get('paper_primary_aware_rerank_factor')} "
            f"paper_final_score_v3={r.get('paper_final_score_v3')} "
            f"reasons={r.get('paper_primary_aware_rerank_reasons')}"
        )
    print("--- top penalized papers ---")
    print("  (none; boost-only micro-rerank, no explicit penalize)")
    print("--- rank moved rows (subset, v2 vs v3) ---")
    for row in stats.get("rank_moves") or []:
        print(
            f"  wid={row.get('wid')!r} rank_v2={row.get('rank_v2')} rank_v3={row.get('rank_v3')} "
            f"title={str(row.get('title') or '')[:72]!r} v2={row.get('paper_final_score_v2')} "
            f"factor={row.get('factor')} v3={row.get('paper_final_score_v3')}"
        )
    print("--- top 10 by paper_final_score_v2 (before) ---")
    by_wid = stats.get("by_wid_for_print")
    if isinstance(by_wid, dict):
        for i, w in enumerate(order_v2[:10], 1):
            r = by_wid.get(w) or {}
            print(
                f"  #{i} wid={w!r} v2={r.get('paper_final_score_v2')} "
                f"title={str(r.get('title') or '')[:80]!r}"
            )
    else:
        for i, w in enumerate(order_v2[:10], 1):
            print(f"  #{i} wid={w!r}")
    print("--- top 10 by paper_final_score_v3 (after) ---")
    if isinstance(by_wid, dict):
        for i, w in enumerate(order_v3[:10], 1):
            r = by_wid.get(w) or {}
            print(
                f"  #{i} wid={w!r} v3={r.get('paper_final_score_v3')} v2={r.get('paper_final_score_v2')} "
                f"factor={r.get('paper_primary_aware_rerank_factor')} "
                f"title={str(r.get('title') or '')[:80]!r}"
            )
    else:
        for i, w in enumerate(order_v3[:10], 1):
            print(f"  #{i} wid={w!r}")
    print("-" * 80 + "\n")


def _print_stage4_primary_candidate_audit(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    *,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    kept_recs: List[Dict[str, Any]] = [
        by_wid[w] for w in selected_wids_set if w in by_wid and isinstance(by_wid.get(w), dict)
    ]
    n = len(kept_recs)
    top_k = max(1, min(STAGE4_PRIMARY_CANDIDATE_AUDIT_TOP_K, max(n, 1)))

    for r in kept_recs:
        if isinstance(r.get("paper_primary_candidate_audit_components"), dict) and r.get(
            "paper_primary_candidate_score"
        ) is not None:
            continue
        sc, parts = _compute_paper_primary_candidate_score(r)
        r["paper_primary_candidate_score"] = float(sc)
        r["paper_primary_candidate_audit_components"] = dict(parts)

    ctr: Counter = Counter()
    for r in kept_recs:
        ctr[str(r.get("paper_evidence_role") or "unknown")] += 1

    ranked = sorted(
        kept_recs,
        key=lambda x: -float(x.get("paper_primary_candidate_score") or 0.0),
    )
    non_primary = [r for r in ranked if str(r.get("paper_evidence_role") or "").strip() != "primary_evidence"]
    support_pool = [r for r in non_primary if str(r.get("paper_evidence_role") or "").strip() == "support_evidence"]
    support_sorted = sorted(
        support_pool,
        key=lambda x: -float(x.get("paper_primary_candidate_score") or 0.0),
    )
    support_like_primary = support_sorted[:8]

    prim_cs_max = max(
        (
            float(r.get("paper_primary_candidate_score") or 0.0)
            for r in kept_recs
            if str(r.get("paper_evidence_role") or "").strip() == "primary_evidence"
        ),
        default=0.0,
    )
    # 旁路门槛：与池中 primary 的审计分差距 ≤0.022 视为「很像 primary」（无 primary 时退化为固定下限）
    _floor = max(0.65, float(prim_cs_max) - 0.022) if prim_cs_max > 0 else 0.72
    support_near_primary_count = sum(
        1
        for r in support_pool
        if float(r.get("paper_primary_candidate_score") or 0.0) >= _floor
    )

    print("\n" + "-" * 80)
    print("[Stage4 primary candidate audit]")
    print("-" * 80)
    print(
        f"total_papers_considered={n} top_k_show={top_k} "
        f"paper_evidence_role_counter={dict(ctr)} "
        f"primary_candidate_score_max_among_primary_evidence={prim_cs_max:.6f} "
        f"support_like_primary_score_floor={_floor:.6f}"
    )
    print(f"--- top {top_k} by paper_primary_candidate_score ---")
    for r in ranked[:top_k]:
        p = r.get("paper_primary_candidate_audit_components")
        if not isinstance(p, dict):
            _, p = _compute_paper_primary_candidate_score(r)
        print("  " + _paper_primary_candidate_compact_row(r, parts=p if isinstance(p, dict) else None))

    print("--- support_like_primary (support_evidence: high primary-candidate vs pool primary) ---")
    print(
        f"support_evidence_total={len(support_pool)} "
        f"support_like_primary_count(score>={_floor:.4f})={support_near_primary_count}"
    )
    _sup_show = min(8, len(support_like_primary))
    print(f"top_support_primary_like_examples (max {_sup_show}) ---")
    for r in support_like_primary[:8]:
        p = r.get("paper_primary_candidate_audit_components")
        if not isinstance(p, dict):
            _, p = _compute_paper_primary_candidate_score(r)
        print("  " + _paper_primary_candidate_compact_row(r, parts=p if isinstance(p, dict) else None))
    print("-" * 80 + "\n")


def _normalize_axis_text(*parts: Any) -> str:
    """Lowercase, single-spaced blob for cheap substring checks."""
    buf: List[str] = []
    for p in parts:
        if p is None:
            continue
        if isinstance(p, (list, tuple, set)):
            for x in p:
                buf.append(str(x))
        else:
            buf.append(str(p))
    s = " ".join(buf).strip().lower()
    if not s:
        return ""
    out: List[str] = []
    for ch in s:
        if ch.isspace():
            if out and out[-1] != " ":
                out.append(" ")
        else:
            out.append(ch)
    return "".join(out).strip()


def _axis_padded_blob(rec: Dict[str, Any], jd_align_map: Optional[Dict[str, float]]) -> Tuple[str, str]:
    """(padded_blob, wid) for matching; blob has leading/trailing spaces."""
    wid = str(rec.get("wid") or "")
    title = rec.get("title")
    domains = rec.get("domains")
    hits = rec.get("hits") if isinstance(rec.get("hits"), list) else []
    hit_terms: List[str] = []
    if isinstance(rec.get("hit_terms"), list):
        hit_terms = [str(x) for x in rec.get("hit_terms") or []]
    else:
        for h in hits:
            if isinstance(h, dict):
                hit_terms.append(str(h.get("term") or h.get("vid") or ""))
    abs_blob = ""
    for k in ("abstract", "abstract_snippet", "snippet", "paper_abstract"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            abs_blob = v
            break
    ja = rec.get("jd_align")
    if ja is None and jd_align_map is not None and wid:
        try:
            ja = jd_align_map.get(wid)
        except Exception:
            ja = None
    ja_s = f" jd_align={float(ja):.4f} " if ja is not None else ""
    raw = _normalize_axis_text(title, domains, hit_terms, abs_blob, ja_s)
    if not raw:
        return "", wid
    return f" {raw} ", wid


def _axis_match_add(
    reasons: Dict[str, List[str]],
    axis: str,
    reason: str,
    labels: Set[str],
) -> None:
    labels.add(axis)
    reasons.setdefault(axis, []).append(reason)


def _infer_job_axis_labels_for_paper(
    rec: Dict[str, Any],
    jd_align_map: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], Optional[str], Dict[str, List[str]]]:
    """
    轻量多标签推断；仅用于审计。
    返回 (sorted labels, primary_label, axis -> reason strings)。
    """
    labels: Set[str] = set()
    reasons: Dict[str, List[str]] = {}
    blob, _wid = _axis_padded_blob(rec, jd_align_map)
    if not blob:
        return ["unclassified"], "unclassified", {"unclassified": ["no_text_signal"]}

    hqc = str(rec.get("top_paper_hit_quality_class") or rec.get("hit_quality_class") or "").strip().lower()

    def _hit(pat: str, tag: str, axis: str) -> bool:
        if pat in blob:
            _axis_match_add(reasons, axis, f"{tag}:{pat.strip()}", labels)
            return True
        return False

    # control_core（偏方法/理论控制，而非单独泛词 control）
    for pat in (
        " robust control ",
        " sliding mode ",
        " lyapunov ",
        " h-infinity ",
        " feedback control ",
        " nonlinear control ",
        " adaptive control ",
        " tracking control ",
        " disturbance observer ",
        " impedance control ",
        " hybrid control ",
        " passivity ",
        " backstepping ",
        " small-gain ",
        " gain scheduling ",
        " linear quadratic regulator ",
        " lqr ",
        " loop shaping ",
    ):
        _hit(pat, "title_or_text", "control_core")

    # dynamics / kinematics
    for pat in (
        " inverse dynamics ",
        " kinematics ",
        " forward kinematics ",
        " jacobian ",
        " manipulator dynamics ",
        " rigid body ",
        " lagrangian ",
        " euler-lagrange ",
        " operational space ",
        " joint space ",
        " dynamics ",
        " dynamic model ",
        " equations of motion ",
    ):
        _hit(pat, "title_or_text", "dynamics_kinematics")

    # planning / trajectory
    for pat in (
        " motion planning ",
        " trajectory optimization ",
        " trajectory generation ",
        " trajectory tracking ",
        " path planning ",
        " collision avoidance ",
        " rrt ",
        " rapidly-exploring ",
        " prm ",
        " sampling-based planning ",
        " coverage path ",
        " kinodynamic ",
    ):
        _hit(pat, "title_or_text", "planning_trajectory")

    # optimal control / MPC
    for pat in (
        " optimal control ",
        " model predictive ",
        " mpc ",
        " ilqr ",
        " iterative linear quadratic ",
        " differential dynamic programming ",
        " ddp ",
        " receding horizon ",
        " constrained optimization ",
    ):
        _hit(pat, "title_or_text", "optimal_control")

    # estimation / observers（localization 仅在机器人/控制上下文中记一条）
    for pat in (
        " state estimation ",
        " kalman ",
        " ekf ",
        " ukf ",
        " particle filter ",
        " observer ",
        " moving horizon estimation ",
    ):
        _hit(pat, "title_or_text", "estimation")
    if (" localization " in blob or " slam " in blob) and any(
        x in blob for x in (" robot ", " robotic ", " autonomous ", " manipulator ", " uav ", " quadrotor ", " mobile robot ")
    ):
        _axis_match_add(reasons, "estimation", "title_or_text:localization/slam_in_robot_context", labels)

    # simulation / sim2real
    for pat in (
        " simulation ",
        " simulator ",
        " gazebo ",
        " mujoco ",
        " pybullet ",
        " isaac ",
        " sim2real ",
        " sim-to-real ",
        " domain randomization ",
        " digital twin ",
        " physics engine ",
        " friction ",
        " system identification ",
        " transfer from sim ",
    ):
        _hit(pat, "title_or_text", "simulation_sim2real")

    # hit_terms / quality class 弱信号（仅当正文未命中对应轴时补充少量线索）
    ht_join = _normalize_axis_text(rec.get("hit_terms"))
    if ht_join:
        ht_pad = f" {ht_join} "
        if "robot control" in ht_pad and "control_core" not in labels:
            _axis_match_add(reasons, "control_core", "hit_terms:robot_control_umbrella", labels)
        for kw, ax in (
            ("dynamics", "dynamics_kinematics"),
            ("kinematic", "dynamics_kinematics"),
            ("planning", "planning_trajectory"),
            ("trajectory", "planning_trajectory"),
            ("mpc", "optimal_control"),
            ("estimation", "estimation"),
        ):
            if kw in ht_pad and ax not in labels:
                _axis_match_add(reasons, ax, f"hit_terms:{kw}", labels)

    if hqc in ("mainline_resonance", "mainline_plus_support", "single_hit_mainline") and not labels:
        _axis_match_add(reasons, "control_core", f"quality_class:{hqc}", labels)

    substantive: Set[str] = {
        "control_core",
        "dynamics_kinematics",
        "planning_trajectory",
        "optimal_control",
        "estimation",
        "simulation_sim2real",
    }
    has_substantive = bool(substantive.intersection(labels))

    # generic shell：宽应用词 + 缺少厚子轴（有任一实质轴则不打）
    broad_ok = any(
        p in blob
        for p in (
            " robot ",
            " robotic ",
            " autonomous ",
            " navigation ",
            " framework ",
            " middleware ",
            " ros ",
            " system architecture ",
            " shell ",
        )
    )
    if broad_ok and not has_substantive:
        _axis_match_add(
            reasons,
            "generic_robot_autonomous_shell",
            "pattern:broad_robot_autonomous_nav_framework_without_substantive_axis",
            labels,
        )
    elif broad_ok and has_substantive:
        # 有厚轴则移除 generic 标签（避免误判强论文）
        if "generic_robot_autonomous_shell" in labels:
            labels.discard("generic_robot_autonomous_shell")
            reasons.pop("generic_robot_autonomous_shell", None)

    if not labels:
        labels.add("unclassified")
        reasons.setdefault("unclassified", []).append("no_axis_rule_matched")

    ordered = [a for a in STAGE4_JOB_AXIS_ORDER if a in labels]
    rest = sorted(labels.difference(STAGE4_JOB_AXIS_ORDER))
    out_labels = ordered + rest

    primary: Optional[str] = None
    for a in STAGE4_JOB_AXIS_ORDER:
        if a in labels:
            primary = a
            break
    if primary is None:
        primary = out_labels[0] if out_labels else "unclassified"

    return out_labels, primary, reasons


def _attach_job_axis_audit_to_kept_papers(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    jd_align_map: Optional[Dict[str, float]],
) -> None:
    """旁路写入 audit 字段；任何异常吞掉，不中断主流程。"""
    for w in selected_wids_set:
        rec = by_wid.get(w)
        if not isinstance(rec, dict):
            continue
        try:
            labs, prim, rsn = _infer_job_axis_labels_for_paper(rec, jd_align_map=jd_align_map)
            rec["job_axis_labels"] = labs
            rec["job_axis_primary_label"] = prim
            rec["job_axis_audit_reasons"] = {k: list(v) for k, v in rsn.items()}
        except Exception:
            rec["job_axis_labels"] = ["unclassified"]
            rec["job_axis_primary_label"] = "unclassified"
            rec["job_axis_audit_reasons"] = {"unclassified": ["audit_infer_failed_safe_fallback"]}


def _summarize_job_axis_coverage(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
) -> Dict[str, Any]:
    kept = [w for w in selected_wids_set if w in by_wid]
    n = max(len(kept), 1)
    axis_counts: Counter = Counter()
    multi_axis = 0
    generic_n = 0
    for w in kept:
        rec = by_wid.get(w) or {}
        labs = rec.get("job_axis_labels")
        if not isinstance(labs, list):
            continue
        labs_non_uc = [str(x) for x in labs if str(x) != "unclassified"]
        if len(labs_non_uc) >= 2:
            multi_axis += 1
        for a in labs:
            axis_counts[str(a)] += 1
        if "generic_robot_autonomous_shell" in labs:
            generic_n += 1
    per_axis_pct = {a: 100.0 * axis_counts.get(a, 0) / float(n) for a in STAGE4_JOB_AXIS_ORDER}
    per_axis_pct["unclassified"] = 100.0 * axis_counts.get("unclassified", 0) / float(n)
    low_axes = [a for a in STAGE4_JOB_AXIS_ORDER if axis_counts.get(a, 0) == 0]
    weak_axes = [a for a in STAGE4_JOB_AXIS_ORDER if 0 < axis_counts.get(a, 0) < max(1, int(0.05 * n))]
    return {
        "kept_total": len(kept),
        "axis_counts": dict(axis_counts),
        "per_axis_pct": per_axis_pct,
        "multi_axis_papers": multi_axis,
        "generic_shell_count": generic_n,
        "generic_shell_pct": 100.0 * generic_n / float(n),
        "axes_absent": low_axes,
        "axes_weak": weak_axes,
    }


def _print_stage4_job_axis_coverage_summary(
    summary: Dict[str, Any],
    *,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    print("\n" + "-" * 80)
    print("[Stage4 job-axis coverage summary]")
    print("-" * 80)
    kt = int(summary.get("kept_total") or 0)
    print(f"kept_papers_total={kt}")
    ac = summary.get("axis_counts") or {}
    pct = summary.get("per_axis_pct") or {}
    for ax in STAGE4_JOB_AXIS_ORDER:
        c = int(ac.get(ax, 0) or 0)
        p = float(pct.get(ax, 0.0) or 0.0)
        print(f"  axis={ax} papers={c} pct_of_kept={p:.1f}%")
    uc = int(ac.get("unclassified", 0) or 0)
    ucp = float(pct.get("unclassified", 0.0) or 0.0)
    print(f"  axis=unclassified papers={uc} pct_of_kept={ucp:.1f}%")
    print(
        f"multi_axis_papers={int(summary.get('multi_axis_papers') or 0)} "
        f"generic_robot_autonomous_shell_count={int(summary.get('generic_shell_count') or 0)} "
        f"generic_robot_autonomous_shell_pct={float(summary.get('generic_shell_pct') or 0.0):.1f}%"
    )
    print(f"axes_absent_count0={summary.get('axes_absent')}")
    print(f"axes_weak_low_coverage={summary.get('axes_weak')}")
    print("-" * 80 + "\n")


def _print_stage4_job_axis_coverage_top_rows(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    *,
    audit_print: bool,
    top_n: int,
) -> None:
    if not audit_print:
        return
    kept = [w for w in selected_wids_set if w in by_wid]
    if not kept:
        print("\n" + "-" * 80)
        print("[Stage4 job-axis coverage top rows]")
        print("-" * 80)
        print("(no kept papers)")
        print("-" * 80 + "\n")
        return

    def _final_score_w(w: str) -> float:
        r = by_wid.get(w) or {}
        v = r.get("paper_final_score_v2")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        try:
            return float(r.get("paper_score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    ranked = sorted(kept, key=lambda w: -_final_score_w(w))
    ranked = ranked[: max(1, top_n)]
    print("\n" + "-" * 80)
    print("[Stage4 job-axis coverage top rows]")
    print("-" * 80)
    for i, w in enumerate(ranked, 1):
        r = by_wid.get(w) or {}
        title = str(r.get("title") or "")[:100]
        hqc = str(r.get("top_paper_hit_quality_class") or r.get("hit_quality_class") or "")
        ht = r.get("hit_terms")
        if not isinstance(ht, list):
            ht = []
        labs = r.get("job_axis_labels")
        rsn = r.get("job_axis_audit_reasons")
        rs_short = ""
        if isinstance(rsn, dict) and rsn:
            parts: List[str] = []
            for ax, msgs in list(rsn.items())[:5]:
                if isinstance(msgs, list) and msgs:
                    parts.append(f"{ax}:[{msgs[0]}]")
            rs_short = "; ".join(parts)
        fs = _final_score_w(w)
        print(
            f"  #{i} wid={w!r} paper_final_score_v2_or_fallback={fs:.6f} hit_quality_class={hqc!r} "
            f"title={title!r}"
        )
        print(f"      hit_terms={ht!r}")
        print(f"      job_axis_labels={labs!r}")
        print(f"      audit_reasons_short={rs_short!r}")
    print("-" * 80 + "\n")


# --- primary_evidence 厚度门（温和收紧 role；不改 v2 / evidence 分公式）---
STAGE4_PRIMARY_EVIDENCE_THICK_GATE = os.environ.get("STAGE4_PRIMARY_EVIDENCE_THICK_GATE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
STAGE4_PRIMARY_THICK_JD_ALIGN_FALLBACK = float(os.environ.get("STAGE4_PRIMARY_THICK_JD_ALIGN_FALLBACK", "0.78"))
STAGE4_PRIMARY_THICK_TOPIC_CONS_STRONG = float(os.environ.get("STAGE4_PRIMARY_THICK_TOPIC_CONS_STRONG", "0.075"))
STAGE4_PRIMARY_THICK_TOPIC_SUBFIELD_PAIR_TC = float(os.environ.get("STAGE4_PRIMARY_THICK_TOPIC_SUBFIELD_PAIR_TC", "0.055"))
STAGE4_PRIMARY_THICK_TOPIC_SUBFIELD_PAIR_SC = float(os.environ.get("STAGE4_PRIMARY_THICK_TOPIC_SUBFIELD_PAIR_SC", "0.028"))
STAGE4_PRIMARY_THICK_TOPIC_MODERATE = float(os.environ.get("STAGE4_PRIMARY_THICK_TOPIC_MODERATE", "0.042"))
STAGE4_PRIMARY_THICK_JD_WITH_TOPIC_MODERATE = float(os.environ.get("STAGE4_PRIMARY_THICK_JD_WITH_TOPIC_MODERATE", "0.76"))


def _paper_axis_combo_is_primary_like(labs: Set[str]) -> bool:
    """条件 C：双轴共支撑（不含纯 generic_shell 组合）。"""
    if "generic_robot_autonomous_shell" in labs:
        thick = {"dynamics_kinematics", "planning_trajectory", "optimal_control", "estimation", "simulation_sim2real"}
        if not (labs & thick) and labs <= {"generic_robot_autonomous_shell", "control_core", "unclassified"}:
            return False
    cc = "control_core" in labs
    dk = "dynamics_kinematics" in labs
    pt = "planning_trajectory" in labs
    oc = "optimal_control" in labs
    es = "estimation" in labs
    sim = "simulation_sim2real" in labs
    if cc and dk:
        return True
    if cc and pt:
        return True
    if cc and oc:
        return True
    if cc and es:
        return True
    if pt and oc:
        return True
    if sim and cc:
        return True
    return False


def _axis_profile_generic_shell_dominant(labs_set: Set[str]) -> bool:
    """存在 generic_robot_autonomous_shell，且无任何厚子轴（动力学/规划/最优/估计/仿真）。"""
    ls = {str(x) for x in labs_set if str(x) != "unclassified"}
    if "generic_robot_autonomous_shell" not in ls:
        return False
    thick_axes = {
        "dynamics_kinematics",
        "planning_trajectory",
        "optimal_control",
        "estimation",
        "simulation_sim2real",
    }
    if ls & thick_axes:
        return False
    return True


def _paper_jd_topic_alignment_primary_like(rec: Dict[str, Any]) -> Tuple[bool, str]:
    """条件 D：subfield/topic 级与 JD 的混合强度（非仅靠 field）。"""
    det = rec.get("hierarchy_consensus_detail")
    if not isinstance(det, dict):
        return False, "D:missing_hierarchy_consensus_detail"
    try:
        tc = float(det.get("topic_cons") or 0.0)
        sc = float(det.get("subfield_cons") or 0.0)
    except (TypeError, ValueError):
        tc, sc = 0.0, 0.0
    if tc >= STAGE4_PRIMARY_THICK_TOPIC_CONS_STRONG:
        return True, f"D:topic_cons>={STAGE4_PRIMARY_THICK_TOPIC_CONS_STRONG:.3f}"
    if tc >= STAGE4_PRIMARY_THICK_TOPIC_SUBFIELD_PAIR_TC and sc >= STAGE4_PRIMARY_THICK_TOPIC_SUBFIELD_PAIR_SC:
        return True, "D:topic+subfield_pair_strong"
    try:
        ja = float(rec.get("jd_align") or 0.0)
    except (TypeError, ValueError):
        ja = 0.0
    if tc >= STAGE4_PRIMARY_THICK_TOPIC_MODERATE and ja >= STAGE4_PRIMARY_THICK_JD_WITH_TOPIC_MODERATE:
        return True, "D:topic_moderate_plus_high_jd_align"
    return False, "D:insufficient_topic_subfield_signal"


def _decide_primary_evidence_thickness(
    rec: Dict[str, Any],
    jd_align_map: Optional[Dict[str, float]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    primary 厚度门：满足 A/B/C/D 之一，或在非 generic-shell-dominant 下用 F（jd+hit_quality）兜底。
    generic_shell_dominant 时禁止仅靠 F 抬 primary。
    """
    detail: Dict[str, Any] = {
        "passed": False,
        "satisfied": None,
        "codes": [],
        "labels_used": [],
        "blocked": None,
    }
    hq = str(rec.get("hit_quality_class") or "").strip()
    try:
        ml = int(rec.get("mainline_term_count") or 0)
    except (TypeError, ValueError):
        ml = 0
    try:
        v2 = float(rec.get("paper_final_score_v2") or 0.0)
    except (TypeError, ValueError):
        v2 = 0.0

    labs: List[str]
    try:
        raw_labs = rec.get("job_axis_labels")
        if isinstance(raw_labs, list) and raw_labs:
            labs = [str(x) for x in raw_labs]
        else:
            labs, _, _ = _infer_job_axis_labels_for_paper(rec, jd_align_map)
    except Exception:
        labs = ["unclassified"]
    detail["labels_used"] = list(labs)
    labs_set = set(labs)

    shell_dom = _axis_profile_generic_shell_dominant(labs_set)

    # A：主轴 term 数
    if ml >= 2:
        detail["passed"] = True
        detail["satisfied"] = "A:mainline_term_count_ge_2"
        detail["codes"].append("A")
        return True, detail

    # B：命中质量厚
    if hq == "mainline_plus_support":
        detail["passed"] = True
        detail["satisfied"] = "B:mainline_plus_support"
        detail["codes"].append("B")
        return True, detail

    # C：轴组合
    if _paper_axis_combo_is_primary_like(labs_set):
        detail["passed"] = True
        detail["satisfied"] = "C:axis_pair_or_combo"
        detail["codes"].append("C")
        return True, detail

    # D：JD topic/subfield 对齐
    d_ok, d_msg = _paper_jd_topic_alignment_primary_like(rec)
    if d_ok:
        detail["passed"] = True
        detail["satisfied"] = d_msg
        detail["codes"].append("D")
        return True, detail

    # F：无轴/瘦轴时的温和兜底（term 集瘦时避免卡死）；generic_shell_dominant 禁止单靠 F
    try:
        ja = float(rec.get("jd_align") or 0.0)
    except (TypeError, ValueError):
        ja = 0.0
    if not shell_dom and ml >= 1 and ja >= STAGE4_PRIMARY_THICK_JD_ALIGN_FALLBACK:
        if hq in ("mainline_resonance", "single_hit_mainline") and v2 >= 0.48:
            detail["passed"] = True
            detail["satisfied"] = f"F:jd_align_ge_{STAGE4_PRIMARY_THICK_JD_ALIGN_FALLBACK:.2f}_resonance_or_mainline"
            detail["codes"].append("F")
            return True, detail

    detail["passed"] = False
    if shell_dom:
        detail["blocked"] = "generic_shell_dominant_need_ABCD_or_non_shell_F_context"
    else:
        detail["blocked"] = "thickness_gate_failed_need_A_B_C_D_or_F"
    return False, detail


def _assign_stage4_paper_evidence_role(
    rec: Dict[str, Any],
    jd_align_map: Optional[Dict[str, float]] = None,
) -> str:
    """base primary → 厚度门；失败降为 support_evidence（不改分数）。"""
    base = _assign_stage4_paper_evidence_role_base(rec)
    if base != "primary_evidence":
        rec["primary_evidence_gate_detail"] = None
        return base
    if not STAGE4_PRIMARY_EVIDENCE_THICK_GATE:
        rec["primary_evidence_gate_detail"] = {"passed": True, "skipped": "STAGE4_PRIMARY_EVIDENCE_THICK_GATE_off"}
        return "primary_evidence"
    ok, detail = _decide_primary_evidence_thickness(rec, jd_align_map)
    rec["primary_evidence_gate_detail"] = detail
    if ok:
        return "primary_evidence"
    return "support_evidence"


def _print_stage4_primary_evidence_gate_audit(
    by_wid: Dict[str, Dict[str, Any]],
    selected_wids_set: Set[str],
    *,
    audit_print: bool,
) -> None:
    if not audit_print:
        return
    kept = [w for w in selected_wids_set if w in by_wid]
    if not kept:
        print("\n[Stage4 primary evidence gate audit] kept_papers_count=0\n")
        return
    primary_before = 0
    primary_after = 0
    support_after = 0
    fringe_after = 0
    demoted: List[Tuple[str, Dict[str, Any], str, str]] = []
    for w in kept:
        rec = by_wid[w]
        if not isinstance(rec, dict):
            continue
        b = _assign_stage4_paper_evidence_role_base(rec)
        f = str(rec.get("paper_evidence_role") or "")
        if b == "primary_evidence":
            primary_before += 1
        if f == "primary_evidence":
            primary_after += 1
        elif f == "support_evidence":
            support_after += 1
        elif f == "fringe_evidence":
            fringe_after += 1
        if b == "primary_evidence" and f == "support_evidence":
            det = rec.get("primary_evidence_gate_detail") or {}
            blk = str(det.get("blocked") or det.get("satisfied") or "")
            demoted.append((w, rec, b, blk))
    print("\n" + "-" * 80)
    print("[Stage4 primary evidence gate audit]")
    print("-" * 80)
    print(
        f"primary_before_count={primary_before} primary_after_count={primary_after} "
        f"support_after_count={support_after} fringe_after_count={fringe_after} "
        f"demoted_from_primary_to_support={len(demoted)}"
    )
    if demoted:
        print("--- demoted papers (base_primary -> support) ---")
        for w, rec, _b, blk in demoted[:40]:
            title = str(rec.get("title") or "")[:100]
            try:
                v2 = float(rec.get("paper_final_score_v2") or 0.0)
            except (TypeError, ValueError):
                v2 = 0.0
            hqc = str(rec.get("hit_quality_class") or "")
            try:
                ml = int(rec.get("mainline_term_count") or 0)
            except (TypeError, ValueError):
                ml = 0
            labs = rec.get("job_axis_labels")
            if not isinstance(labs, list):
                labs = (rec.get("primary_evidence_gate_detail") or {}).get("labels_used")
            det = rec.get("primary_evidence_gate_detail") or {}
            sat = det.get("satisfied")
            print(
                f"  wid={w!r} title={title!r} paper_final_score_v2={v2:.6f} "
                f"hit_quality_class={hqc!r} mainline_term_count={ml} job_axis_labels={labs!r}"
            )
            print(
                f"    role_base=primary_evidence role_final=support_evidence "
                f"downgrade_reason={blk!r} gate_detail_satisfied={sat!r}"
            )
    kept_prim: List[Tuple[str, Dict[str, Any]]] = []
    for w in kept:
        r = by_wid.get(w) or {}
        if str(r.get("paper_evidence_role") or "") == "primary_evidence":
            kept_prim.append((w, r))
    kept_prim.sort(key=lambda t: -float((t[1].get("paper_final_score_v2") or 0.0)))
    if kept_prim:
        print("--- sample retained primary (why still thick enough, max 8) ---")
        for w, rec in kept_prim[:8]:
            title = str(rec.get("title") or "")[:90]
            det = rec.get("primary_evidence_gate_detail") or {}
            print(
                f"  wid={w!r} title={title!r} satisfied={det.get('satisfied')!r} "
                f"codes={det.get('codes')!r} labels={det.get('labels_used')!r}"
            )
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
    ) -> Dict[str, Any]:
        """
        Stage4 的 paper grounding：用 paper 的 title/domains 对齐岗位主轴与 term 证据。
        返回：
          - grounding: 0~1（主轴/词面落地强度）
          - off_topic_penalty: 额外偏题惩罚（用于抑制泛命中论文池）
          - 可选 robot_control_grounding_explain：仅 term 为 robot control 时的错域压制说明
          - 可选 robot_motion_alias_explain：robot control + 机器人运动控制父锚且命中 motion 向 alias 时的加分说明
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
        rc_explain: Dict[str, Any] = {}
        rc_mult = 1.0
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

        # robot control：字面极泛；专项错域压制 + 缺主轴轻压（仅本 term，见 _robot_control_offtopic_penalty）
        if "robot control" in term_text:
            merged_lc = (t + " " + d).strip()
            rc_mult, rc_explain = _robot_control_offtopic_penalty(merged_lc)
            off_topic_penalty *= rc_mult

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

        # robot control + 机器人运动控制父锚：off-topic 已乘入 rc_mult 后，对命中 motion 向 alias 的 paper 给小额 grounding 加分（不改动主公式结构）
        alias_explain: Dict[str, Any] = {}
        if "robot control" in term_text:
            merged_alias = (t + " " + d).strip()
            ab, alias_explain = _robot_control_motion_alias_support(
                term_text,
                merged_alias,
                str(meta.get("parent_anchor") or ""),
                str(meta.get("parent_primary") or ""),
                float(rc_mult),
            )
            if ab > 0:
                grounding = min(1.0, float(grounding) + ab)

        grounding = max(0.0, min(1.0, float(grounding)))
        out_gs: Dict[str, Any] = {"grounding": grounding, "off_topic_penalty": float(off_topic_penalty)}
        if rc_explain:
            out_gs["robot_control_grounding_explain"] = rc_explain
        if alias_explain:
            out_gs["robot_motion_alias_explain"] = alias_explain
        return out_gs

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
    ) -> Dict[str, Any]:
        """
        Stage4 单篇打分（最小侵入版）：
        在原有 term grounding + penalty 基础上，新增 JD↔摘要向量 jd_align 软融合。
        """
        tt = term_type_cache.get(vid)
        if tt is None:
            tt = _compute_term_type_factors(vid)
            term_type_cache[vid] = tt

        ground = _compute_grounding_score(vid, title, domains)
        rc_gex = ground.get("robot_control_grounding_explain")
        rm_alias_ex = ground.get("robot_motion_alias_explain")
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

        out_score: Dict[str, Any] = {
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
        if rc_gex is not None:
            out_score["robot_control_grounding_explain"] = rc_gex
        if rm_alias_ex is not None:
            out_score["robot_motion_alias_explain"] = rm_alias_ex
        return out_score

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
        _audit_row: Dict[str, Any] = {
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
        if score_detail.get("robot_control_grounding_explain") is not None:
            _audit_row["robot_control_grounding_explain"] = score_detail["robot_control_grounding_explain"]
        if score_detail.get("robot_motion_alias_explain") is not None:
            _audit_row["robot_motion_alias_explain"] = score_detail["robot_motion_alias_explain"]
        term_kept_paper_audit[vid].append(_audit_row)

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
    limited_row_tag: List[str] = []
    keep_compete_extra_limited_row_count = 0
    keep_compete_wids_term_local_cap: Set[str] = set()
    keep_compete_wids_extra_post_cap: Set[str] = set()
    extra_rows_slice_total = 0
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
            _ws = str(wid)
            limited.append((wid, vid, term_contrib, idf_weight, hit_level, grounding, jd_align))
            limited_row_tag.append("term_local_cap")
            keep_compete_wids_term_local_cap.add(_ws)

        _extra_keep_rows = _stage4_collect_keep_compete_extra_rows_for_term(triples, vid, local_cap)
        extra_rows_slice_total += max(0, len(triples) - local_cap)
        keep_compete_extra_limited_row_count += len(_extra_keep_rows)
        for _er in _extra_keep_rows:
            limited.append(_er)
            limited_row_tag.append("keep_compete_extra_post_cap_grounded")
            try:
                keep_compete_wids_extra_post_cap.add(str(_er[0]))
            except (TypeError, ValueError, IndexError):
                pass

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

    term_capped_all_wids: Set[str] = set()
    for _tcset in term_capped_unique_wids.values():
        term_capped_all_wids |= {str(x) for x in _tcset}

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
    for _li, (wid, vid, term_contrib, idf_weight, hit_level, grounding, jd_align) in enumerate(limited):
        _row_tag = (
            limited_row_tag[_li]
            if _li < len(limited_row_tag)
            else "unknown"
        )
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
                "keep_compete_materialize_tags": set(),
            }
        _kt = by_wid[wid].get("keep_compete_materialize_tags")
        if not isinstance(_kt, set):
            _kt = set()
            by_wid[wid]["keep_compete_materialize_tags"] = _kt
        _kt.add(str(_row_tag))
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
        _stage4_attach_paper_evidence_quality_fields(_rec)

    for __w, __rec in by_wid.items():
        _compute_stage4_ltr_like_paper_rerank_score(__rec)

    for __w, __rec in by_wid.items():
        _compute_stage4_mainline_protection_factor(__rec)

    _, _stage4_topic_pool_summary = _stage4_collect_topic_cohesive_candidate_wids(
        by_wid, term_topic_meta, jd_topic_profile
    )
    _print_stage4_candidate_source_summary(_stage4_topic_pool_summary, audit_print=_label_path_stdout)

    # Step2：先算 v2/role（相对 paper_score 初筛 kept 的 rank），再按 compete 重选 global kept，使 evidence/v2 真正参与「谁能进池」
    _attach_stage4_score_migration_fields(by_wid, selected_wids_set, jd_align_map)
    _print_stage4_keep_compete_pool_expansion_audit(
        by_wid,
        selected_wids_set,
        term_capped_all_wids,
        keep_compete_wids_term_local_cap,
        keep_compete_wids_extra_post_cap,
        extra_rows_slice_total,
        keep_compete_extra_limited_row_count,
        _get_term_meta,
        audit_print=_label_path_stdout,
    )
    _keep_compete_stats = _stage4_apply_keep_compete_reselect(
        by_wid, selected_wids_set, sorted_wids, audit_print=_label_path_stdout, jd_align_map=jd_align_map
    )
    if _keep_compete_stats is not None:
        _keep_compete_stats["keep_compete_extra_limited_row_count"] = int(
            keep_compete_extra_limited_row_count
        )
        selected_wids_set = _keep_compete_stats["new_selected_set"]
        sorted_wids = list(_keep_compete_stats["new_sorted_wids"])

    _print_stage4_primary_evidence_gate_audit(by_wid, selected_wids_set, audit_print=_label_path_stdout)

    _print_stage4_keep_competition_migration_summary(
        _keep_compete_stats, by_wid, audit_print=_label_path_stdout
    )
    _print_stage4_v2_activation_audit(
        by_wid, _keep_compete_stats, selected_wids_set, audit_print=_label_path_stdout
    )
    _print_stage4_paper_community_filtering_summary(
        _keep_compete_stats, by_wid, selected_wids_set, audit_print=_label_path_stdout
    )
    _print_stage4_mainline_term_audit(
        by_wid, selected_wids_set, _get_term_meta, audit_print=_label_path_stdout
    )
    _print_stage4_multihit_classification_audit(by_wid, selected_wids_set, audit_print=_label_path_stdout)
    _print_stage4_kept_pool_quality_summary(by_wid, selected_wids_set, audit_print=_label_path_stdout)

    if STAGE4_JOB_AXIS_COVERAGE_AUDIT and _label_path_stdout:
        _attach_job_axis_audit_to_kept_papers(by_wid, selected_wids_set, jd_align_map)
        _job_axis_cov = _summarize_job_axis_coverage(by_wid, selected_wids_set)
        _print_stage4_job_axis_coverage_summary(_job_axis_cov, audit_print=True)
        _print_stage4_job_axis_coverage_top_rows(
            by_wid,
            selected_wids_set,
            audit_print=True,
            top_n=max(1, STAGE4_JOB_AXIS_COVERAGE_TOP_ROWS),
        )

    _print_stage4_paper_explanation_summary(by_wid, selected_wids_set, audit_print=_label_path_stdout)
    _print_stage4_paper_evidence_quality_summary(by_wid, selected_wids_set, audit_print=_label_path_stdout)
    _print_stage4_paper_evidence_quality_top_rows(by_wid, selected_wids_set, audit_print=_label_path_stdout)
    _print_stage4_paper_ltr_rerank_summary(by_wid, selected_wids_set, audit_print=_label_path_stdout)
    _print_stage4_paper_mainline_protection_summary(by_wid, selected_wids_set, audit_print=_label_path_stdout)
    _print_stage4_score_migration_summary(by_wid, selected_wids_set, audit_print=_label_path_stdout)
    # Step2：旁路主证据候选分 + very narrow support->primary（仅改 evidence role）
    _stage4_ensure_job_axis_labels_for_kept(by_wid, selected_wids_set, jd_align_map)
    _stage4_compute_and_attach_primary_candidate_scores(by_wid, selected_wids_set)
    _apply_stage4_support_to_primary_promotion(by_wid, selected_wids_set, jd_align_map, audit_print=_label_path_stdout)
    _print_stage4_evidence_role_summary(by_wid, selected_wids_set, audit_print=_label_path_stdout)
    _print_stage4_primary_candidate_audit(by_wid, selected_wids_set, audit_print=_label_path_stdout)
    sorted_wids, _stage4_primary_aware_rerank_stats = _apply_stage4_primary_aware_rerank(
        by_wid, sorted_wids, selected_wids_set, audit_print=_label_path_stdout
    )

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
            if STAGE4_ENABLE_EVIDENCE_SCORE_MIGRATION:
                kept_wids.sort(
                    key=lambda w: -_stage4_paper_effective_sort_score(by_wid.get(w) or {})
                )
            else:
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
