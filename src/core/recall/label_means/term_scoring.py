import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import (
    TOPIC_WEIGHT_PRIMARY,
    TOPIC_WEIGHT_DENSE,
    TOPIC_WEIGHT_CLUSTER,
    TOPIC_WEIGHT_COOC,
    TOPIC_MIN_ALIGN,
    TOPIC_LOW_ALIGN_PENALTY,
    CLUSTER_EXPANSION_PENALTY,
    COOC_EXPANSION_PENALTY,
    SOURCE_WEIGHT_SIMILAR_TO,
    SOURCE_WEIGHT_JD_VECTOR,
    SOURCE_WEIGHT_DENSE,
    SOURCE_WEIGHT_CLUSTER,
    SOURCE_WEIGHT_COOC,
    DOMAIN_GATE_MIN,
)
from src.core.recall.label_means import advanced_metrics as label_means_adv


def _vector_based_alignment_factors(
    label,
    tid_str: str,
    query_vector,
) -> Tuple[float, float]:
    """
    仅用向量相似度得到任务对齐因子与语境因子，无关键词。
    task_alignment: 0.7~1.0 由 cos_sim(term, query) 决定；
    context_factor: 0.85~1.0 由 anchor_sim(term, anchors) 决定。
    """
    task_f, ctx_f = 1.0, 1.0
    idx = label.vocab_to_idx.get(tid_str) if getattr(label, "vocab_to_idx", None) else None
    if idx is None:
        return task_f, ctx_f
    try:
        term_vec = np.asarray(label.all_vocab_vectors[idx], dtype=np.float32).flatten()
    except Exception:
        return task_f, ctx_f
    if query_vector is not None and term_vec.size > 0:
        q = np.asarray(query_vector, dtype=np.float32).flatten()
        if q.size == term_vec.size:
            cos_sim = float(np.dot(term_vec, q))
            cos_sim = max(-1.0, min(1.0, cos_sim))
            task_f = 0.7 + 0.3 * max(0.0, cos_sim)
    anchor_vecs = getattr(label, "_anchor_vectors", None)
    if anchor_vecs is not None and anchor_vecs.size > 0 and term_vec.size > 0:
        try:
            anchor_vecs = np.asarray(anchor_vecs, dtype=np.float32)
            if anchor_vecs.ndim == 1:
                anchor_vecs = anchor_vecs.reshape(1, -1)
            sims = np.dot(anchor_vecs, term_vec)
            anchor_sim = float(np.mean(sims))
            anchor_sim = max(-1.0, min(1.0, anchor_sim))
            ctx_f = 0.85 + 0.15 * max(0.0, anchor_sim)
        except Exception:
            pass
    return task_f, ctx_f


def _source_credibility(rec: Dict[str, Any]) -> float:
    """
    按 Stage2 来源做可信度加权，无硬编码词表。
    edge_and_ctx=1.0, edge_only=0.95, ctx_only=0.72, cluster 继承种子或 0.95。
    """
    s = (rec.get("source") or rec.get("origin") or "").strip().lower()
    if s == "edge_and_ctx":
        return 1.0
    if s == "edge_only":
        return 0.95
    if s == "ctx_only":
        return 0.72
    # cluster 或其它：继承自种子时已在 expand_with_clusters 中设为 edge_and_ctx/edge_only/ctx_only
    return 0.95


def calculate_final_weights(
    label,
    raw_results: List[Dict[str, Any]],
    query_vector,
    anchor_vids: Optional[List[int]] = None,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float]]:
    """
    统一封装 LabelRecallPath._calculate_final_weights 的逻辑，作为 label_means 层的词级评分入口。

    参数:
      - label: 提供 total_work_count / vocab_to_idx / all_vocab_vectors / debug_info 等属性的宿主对象
      - raw_results: 语义扩展阶段的候选列表
      - query_vector: 当前 JD 向量
      - anchor_vids: 本次 JD 的锚点 vocab id 列表，用于锚点距离门控

    返回:
      - score_map: tid(str) -> 动态权重
      - term_map: tid(str) -> term 文本
      - idf_map: tid(str) -> 词级 idf_val（平滑后的 IDF）
    """
    score_map: Dict[str, float] = {}
    term_map: Dict[str, str] = {}
    idf_map: Dict[str, float] = {}

    required = ("degree_w", "cov_j", "domain_span", "target_degree_w")

    # 初始化调试容器
    label.debug_info.tag_purity_debug = []
    label._last_tag_purity_debug = label.debug_info.tag_purity_debug

    # --- 预计算锚点向量，供锚点距离门控使用 ---
    label._anchor_vectors = None
    label._task_anchor_vectors = None
    label._carrier_anchor_vectors = None
    if (
        anchor_vids
        and getattr(label, "vocab_to_idx", None) is not None
        and getattr(label, "all_vocab_vectors", None) is not None
    ):
        idxs: List[int] = []
        for vid in anchor_vids:
            i = label.vocab_to_idx.get(str(vid))
            if i is not None:
                idxs.append(i)
        if idxs:
            vecs = np.asarray(label.all_vocab_vectors[idxs], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms > 1e-9, norms, 1.0)
            anchor_vecs = vecs / norms
            label._anchor_vectors = anchor_vecs

            if query_vector is not None:
                try:
                    q = np.asarray(query_vector, dtype=np.float32).flatten()
                    if q.size > 0:
                        qn = np.linalg.norm(q)
                        if qn > 1e-9:
                            q = q / qn
                            sims = np.dot(anchor_vecs, q)
                            n_anchor = anchor_vecs.shape[0]
                            if n_anchor < 4:
                                label._task_anchor_vectors = anchor_vecs
                            else:
                                median_sim = float(np.median(sims))
                                task_mask = sims >= median_sim
                                carrier_mask = sims < median_sim
                                if np.any(task_mask):
                                    label._task_anchor_vectors = anchor_vecs[task_mask]
                                if np.any(carrier_mask):
                                    label._carrier_anchor_vectors = anchor_vecs[carrier_mask]
                except Exception:
                    label._task_anchor_vectors = None
                    label._carrier_anchor_vectors = None

    # --- 预计算语义分位数，用于 bucket_factor ---
    label._semantic_p20 = None
    label._semantic_p80 = None
    if query_vector is not None and raw_results:
        sims: List[float] = []
        try:
            q = np.asarray(query_vector, dtype=np.float32).flatten()
            if q.size > 0:
                for rec in raw_results:
                    tid = rec.get("tid")
                    if tid is None:
                        continue
                    idx = label.vocab_to_idx.get(str(tid))
                    if idx is None:
                        continue
                    try:
                        term_vec = label.all_vocab_vectors[idx]
                    except Exception:
                        continue
                    cos_sim = float(np.dot(term_vec, q))
                    if cos_sim >= float(label.SEMANTIC_MIN):
                        sims.append(cos_sim)
        except Exception:
            sims = []

        if sims:
            sims.sort()
            n = len(sims)

            def _pick_percentile(p: float) -> float:
                if n == 1:
                    return sims[0]
                k = int(p * (n - 1))
                k = max(0, min(n - 1, k))
                return sims[k]

            if n >= 5:
                label._semantic_p20 = _pick_percentile(0.2)
                label._semantic_p80 = _pick_percentile(0.8)

    # --- 主循环：逐 term 计算动态权重 ---
    for rec in raw_results:
        tid_str = str(rec["tid"])
        term_text = rec.get("term") or ""
        if all(rec.get(k) is not None for k in required):
            dynamic_weight, idf_val = _apply_word_quality_penalty(label, rec, query_vector)
        else:
            degree_w = rec.get("degree_w") or 1
            sim_score = rec.get("sim_score") or 0.0
            dynamic_weight = sim_score / math.log(1.0 + degree_w) if degree_w else 0.0
            idf_val = math.log10(label.total_work_count / (degree_w + 1))

        # 仅用向量相似度调节：任务对齐(cos_sim)、语境(anchor_sim)，无关键词规则
        if dynamic_weight > 0.0:
            task_f, ctx_f = _vector_based_alignment_factors(label, tid_str, query_vector)
            dynamic_weight *= task_f * ctx_f
        # Stage2 来源可信度：edge_and_ctx > edge_only > ctx_only
        dynamic_weight *= _source_credibility(rec)

        score_map[tid_str] = dynamic_weight
        term_map[tid_str] = term_text
        idf_map[tid_str] = idf_val

    _apply_cluster_rank_decay(label, score_map)
    return score_map, term_map, idf_map


def _genericity_penalty(rec: Dict[str, Any]) -> float:
    """
    对“跨领域且高频”的大泛词做轻量降权，同时避免过度伤害细分技术词。

    仅依赖通用统计特征：
      - domain_span: 领域跨度，越大越接近“万金油”；
      - work_count/degree_w_expanded: 论文覆盖规模，越大越容易成为通用词。

    额外对形如 "*control*" 的广义控制词，在缺少明显机器人/轨迹/最优控制修饰时，做一档通用惩罚。
    """
    term_text = (rec.get("term") or "").strip()
    if not term_text:
        return 1.0

    t_low = term_text.lower()
    try:
        span = int(rec.get("domain_span") or 0)
    except (TypeError, ValueError):
        span = 0
    span = max(1, span)

    # work_count 在部分链路中可能缺失，回退到 degree_w_expanded / degree_w
    try:
        work_count = int(rec.get("work_count") or rec.get("degree_w_expanded") or rec.get("degree_w") or 0)
    except (TypeError, ValueError):
        work_count = 0

    penalty = 1.0

    # 1) 纯统计角度：跨很多领域、且论文量巨大的词，易为通用大词
    if span >= 6 and work_count >= 200:
        penalty *= 0.4
    elif span >= 4 and work_count >= 100:
        penalty *= 0.6

    # 2) 形态角度：大类“control”词，缺少明显机器人/轨迹/最优控制等技术前缀时再降一档
    if "control" in t_low:
        robotics_hints = (
            "robot",
            "uav",
            "ugv",
            "arm",
            "manipulator",
            "kinematic",
            "trajectory",
            "motion",
            "mpc",
            "lqr",
            "ilqr",
            "ddp",
            "state estimation",
            "real-time",
            "realtime",
        )
        if not any(h in t_low for h in robotics_hints):
            penalty *= 0.5

    # 避免数值完全归零，保留一定可逆空间
    return max(0.1, float(penalty))


def get_term_debug_metrics(
    label, rec: Dict[str, Any], query_vector
) -> Optional[Dict[str, Any]]:
    """
    仅计算并返回单条 rec 的调试指标（task_anchor_sim、carrier_anchor_sim、anchor_factor、term_backbone 等），
    不写入 tag_purity_debug。供 Stage3 双闸门路径合并到 debug 条目用。
    """
    try:
        degree_w = int(rec.get("degree_w") or 0)
        degree_w_expanded = int(rec.get("degree_w_expanded") or 0) or max(degree_w, 1)
        target_degree_w = int(rec.get("target_degree_w") or 0)
        domain_span = max(1, int(rec.get("domain_span") or 1))
        cov_j = float(rec.get("cov_j") or 0.0)

        raw_tag_purity = (target_degree_w / degree_w_expanded) if degree_w_expanded else 0.0
        tag_purity = min(1.0, raw_tag_purity)
        purity_term = tag_purity

        total = float(getattr(label, "total_work_count", 1e6) or 1e6)
        idf_term = _idf_backbone(total, degree_w_expanded)
        if (rec.get("source") or "").strip().lower() == "ctx_only":
            idf_term *= 0.85
        idf_val = _smoothed_idf(degree_w, idf_term)

        cos_sim = 0.5
        idx = getattr(label, "vocab_to_idx", None) and label.vocab_to_idx.get(str(rec.get("tid")))
        if idx is not None and query_vector is not None:
            av = getattr(label, "all_vocab_vectors", None)
            if av is not None:
                term_vec = av[idx]
                cos_sim = float(np.dot(np.asarray(term_vec).flatten(), np.asarray(query_vector).flatten()))

        if cos_sim < float(getattr(label, "SEMANTIC_MIN", 0.0)):
            return {"idf_val": round(idf_val, 6)}

        semantic_factor = math.pow(max(0.0, cos_sim), float(getattr(label, "SEMANTIC_POWER", 1.0)))
        max_anchor_sim = None
        task_anchor_sim = None
        carrier_anchor_sim = None
        anchor_vecs = getattr(label, "_anchor_vectors", None)
        task_anchor_vecs = getattr(label, "_task_anchor_vectors", None)
        carrier_anchor_vecs = getattr(label, "_carrier_anchor_vectors", None)
        if idx is not None:
            try:
                av = getattr(label, "all_vocab_vectors", None)
                if av is not None:
                    tvec = np.asarray(av[idx], dtype=np.float32).flatten()
                    tn = np.linalg.norm(tvec)
                    if tn > 1e-9:
                        t_unit = tvec / tn
                        if anchor_vecs is not None and anchor_vecs.size > 0:
                            max_anchor_sim = float(np.max(np.dot(anchor_vecs, t_unit)))
                        if task_anchor_vecs is not None and task_anchor_vecs.size > 0:
                            task_anchor_sim = float(np.max(np.dot(task_anchor_vecs, t_unit)))
                        if carrier_anchor_vecs is not None and carrier_anchor_vecs.size > 0:
                            carrier_anchor_sim = float(np.max(np.dot(carrier_anchor_vecs, t_unit)))
            except Exception:
                pass

        ta = task_anchor_sim if task_anchor_sim is not None else max_anchor_sim
        ca = carrier_anchor_sim if carrier_anchor_sim is not None else max_anchor_sim
        anchor_factor = _anchor_factor(
            ta, ca,
            base=getattr(label, "ANCHOR_BASE", 0.35),
            gain=getattr(label, "ANCHOR_GAIN", 0.65),
        )
        size_penalty = label_means_adv.size_penalty_factor(degree_w, degree_w_expanded)
        term_backbone = semantic_factor * idf_term * purity_term * size_penalty * anchor_factor

        task_advantage = None
        if task_anchor_sim is not None and carrier_anchor_sim is not None:
            task_advantage = round(task_anchor_sim - carrier_anchor_sim, 6)

        return {
            "anchor_sim": round(max_anchor_sim, 6) if max_anchor_sim is not None else None,
            "task_anchor_sim": round(task_anchor_sim, 6) if task_anchor_sim is not None else None,
            "carrier_anchor_sim": round(carrier_anchor_sim, 6) if carrier_anchor_sim is not None else None,
            "task_advantage": task_advantage,
            "idf_val": round(idf_val, 6),
            "anchor_factor": round(anchor_factor, 6),
            "term_backbone": round(term_backbone, 6),
            "cos_sim": round(cos_sim, 6),
        }
    except Exception:
        return None


def _apply_word_quality_penalty(label, rec: Dict[str, Any], query_vector):
    """
    从 LabelRecallPath._apply_word_quality_penalty 迁移而来，内部使用 label 访问资源与超参。
    """
    degree_w = rec["degree_w"]
    cov_j = rec["cov_j"]
    domain_span = max(1, rec["domain_span"])
    degree_w_expanded = rec.get("degree_w_expanded")
    if degree_w_expanded is None:
        degree_w_expanded = degree_w

    raw_tag_purity = (rec["target_degree_w"] / degree_w_expanded) if degree_w_expanded else 0.0
    tag_purity = min(1.0, raw_tag_purity)
    purity_term = tag_purity

    idf_term = _idf_backbone(label.total_work_count, degree_w_expanded)
    # ctx_only 仅来自 JD/锚点向量扩展，对 IDF 做轻量折损，避免大泛词靠 idf 拉高
    if (rec.get("source") or "").strip().lower() == "ctx_only":
        idf_term *= 0.85
    idf_val = _smoothed_idf(degree_w, idf_term)

    cos_sim = 0.5
    idx = label.vocab_to_idx.get(str(rec["tid"]))
    if idx is not None and query_vector is not None:
        term_vec = label.all_vocab_vectors[idx]
        cos_sim = float(np.dot(term_vec, query_vector.flatten()))

    if cos_sim < float(label.SEMANTIC_MIN):
        return 0.0, idf_val

    semantic_factor = math.pow(max(0.0, cos_sim), float(label.SEMANTIC_POWER))

    max_anchor_sim = None
    task_anchor_sim = None
    carrier_anchor_sim = None
    anchor_vecs = getattr(label, "_anchor_vectors", None)
    task_anchor_vecs = getattr(label, "_task_anchor_vectors", None)
    carrier_anchor_vecs = getattr(label, "_carrier_anchor_vectors", None)
    if idx is not None:
        try:
            tvec = np.asarray(label.all_vocab_vectors[idx], dtype=np.float32).flatten()
            tn = np.linalg.norm(tvec)
            if tn > 1e-9:
                t_unit = tvec / tn
                if anchor_vecs is not None and anchor_vecs.size > 0:
                    max_anchor_sim = float(np.max(np.dot(anchor_vecs, t_unit)))
                if task_anchor_vecs is not None and task_anchor_vecs.size > 0:
                    task_anchor_sim = float(np.max(np.dot(task_anchor_vecs, t_unit)))
                if carrier_anchor_vecs is not None and carrier_anchor_vecs.size > 0:
                    carrier_anchor_sim = float(np.max(np.dot(carrier_anchor_vecs, t_unit)))
        except Exception:
            pass

    ta = task_anchor_sim if task_anchor_sim is not None else max_anchor_sim
    ca = carrier_anchor_sim if carrier_anchor_sim is not None else max_anchor_sim
    anchor_factor = _anchor_factor(ta, ca, base=getattr(label, "ANCHOR_BASE", 0.35), gain=getattr(label, "ANCHOR_GAIN", 0.65))

    size_penalty = label_means_adv.size_penalty_factor(degree_w, degree_w_expanded)
    sim_term = semantic_factor
    term_backbone = sim_term * idf_term * purity_term * size_penalty * anchor_factor

    extra_factor = label_means_adv.term_extra_factors(
        rec=rec,
        cos_sim=cos_sim,
        degree_w=degree_w,
        degree_w_expanded=degree_w_expanded,
        cov_j=cov_j,
        domain_span=domain_span,
        tag_purity=tag_purity,
        task_anchor_sim=task_anchor_sim,
        carrier_anchor_sim=carrier_anchor_sim,
        max_anchor_sim=max_anchor_sim,
    )

    generic_penalty = _genericity_penalty(rec)
    dynamic_weight = term_backbone * extra_factor * generic_penalty

    if label.debug_info.tag_purity_debug is not None:
        try:
            label.debug_info.tag_purity_debug.append(
                {
                    "tid": rec.get("tid"),
                    "term": (rec.get("term") or "")[:40],
                    "hit_count": int(rec.get("hit_count", 1) or 1),
                    "degree_w": degree_w,
                    "degree_w_expanded": degree_w_expanded,
                    "target_degree_w": rec.get("target_degree_w"),
                    "raw_tag_purity": round(raw_tag_purity, 6),
                    "capped_tag_purity": round(tag_purity, 6),
                    "cos_sim": round(cos_sim, 6),
                    "anchor_sim": round(max_anchor_sim, 6) if max_anchor_sim is not None else None,
                    "task_anchor_sim": round(task_anchor_sim, 6) if task_anchor_sim is not None else None,
                    "carrier_anchor_sim": round(carrier_anchor_sim, 6)
                    if carrier_anchor_sim is not None
                    else None,
                    "idf_term": round(idf_term, 6),
                    "cov_j": round(float(cov_j), 6) if cov_j is not None else None,
                    "idf_val": round(idf_val, 6),
                    "anchor_factor": round(anchor_factor, 6),
                    "semantic_factor": round(semantic_factor, 6),
                    "purity_term": round(purity_term, 6),
                    "term_backbone": round(term_backbone, 6),
                    "generic_penalty": round(generic_penalty, 6),
                    "dynamic_weight": round(dynamic_weight, 6),
                    "source": (rec.get("source") or rec.get("origin") or "").strip(),
                }
            )
        except Exception:
            pass

    return dynamic_weight, idf_val


def _apply_cluster_rank_decay(label, score_map: Dict[str, float]) -> None:
    """
    从 LabelRecallPath._apply_cluster_rank_decay 迁移而来，对每个 cluster 内部按 score 排序做 head-tail 衰减。
    """
    if not getattr(label, "voc_to_clusters", None) or not score_map:
        return

    label.debug_info.cluster_rank_factors = {}

    debug_list = label.debug_info.tag_purity_debug or []
    anchor_sim_by_tid: Dict[str, float] = {}
    task_sim_by_tid: Dict[str, float] = {}
    for d in debug_list:
        tid = d.get("tid")
        if tid is None:
            continue
        tid_str = str(tid)
        if d.get("anchor_sim") is not None:
            anchor_sim_by_tid[tid_str] = float(d["anchor_sim"])
        if d.get("task_anchor_sim") is not None:
            task_sim_by_tid[tid_str] = float(d["task_anchor_sim"])

    cluster_to_tids: Dict[int, List[str]] = {}
    for tid_str in score_map.keys():
        try:
            tid = int(tid_str)
        except (TypeError, ValueError):
            continue
        clusters = label.voc_to_clusters.get(tid)
        if not clusters:
            continue
        cid, _ = max(clusters, key=lambda x: x[1])
        cluster_to_tids.setdefault(cid, []).append(tid_str)

    for cid, tids in cluster_to_tids.items():
        if len(tids) <= 3:
            for tid_str in tids:
                label.debug_info.cluster_rank_factors[tid_str] = 1.0
            continue

        if len(tids) >= 5:
            tids_sorted = sorted(
                tids,
                key=lambda x: (
                    score_map.get(x, 0.0),
                    task_sim_by_tid.get(x, 0.0),
                ),
                reverse=True,
            )
        else:
            tids_sorted = sorted(tids, key=lambda x: score_map.get(x, 0.0), reverse=True)

        for rank, tid_str in enumerate(tids_sorted):
            anchor_sim = anchor_sim_by_tid.get(tid_str)
            if anchor_sim is not None and anchor_sim >= 0.9:
                factor = 1.0
            elif rank <= 4:
                factor = 1.0
            else:
                factor = 0.85 ** (rank - 4)

            score_map[tid_str] *= factor
            label.debug_info.cluster_rank_factors[tid_str] = float(factor)


def _idf_backbone(total_work_count: float, degree_w_expanded: float) -> float:
    """
    标准 IDF（IR 口径）：idf = log(1 + total_work / (1 + paper_count))。
    """
    return math.log(1.0 + float(total_work_count) / (1.0 + float(degree_w_expanded)))


def _smoothed_idf(degree_w: int, idf_backbone_val: float) -> float:
    """
    平滑 IDF：小词/大词略做约束，中等区间略微加权。
    """
    if degree_w < 10:
        return 1.0
    if degree_w < 50:
        t = (float(degree_w) - 10.0) / 40.0
        return 1.0 + 0.5 * max(0.0, min(1.0, t))
    return 0.9


def _anchor_factor(ta: float, ca: float, base: float, gain: float) -> float:
    """
    锚点距离门控因子：task_anchor_sim 优先，其次 carrier_anchor_sim。
    等价于 LabelRecallPath._anchor_factor 的逻辑，只是参数从 label 提取。
    """
    if ta is None and ca is None:
        return 1.0
    ta_clamped = max(0.0, min(1.0, ta or 0.0))
    ca_clamped = max(0.0, min(1.0, ca or 0.0))
    # 原先 ANCHOR_BASE 固定为 0.35，gain 分成 0.65 + 0.15
    return (
        base
        + gain * math.pow(ta_clamped, 1.5)
        + 0.15 * math.pow(ca_clamped, 1.1)
    )


# ---------- Stage3 双闸门与两分制（先跑通，公式从简） ----------
STAGE3_DEBUG = True  # 调试时打印身份闸门未通过、最终分
PRIMARY_MIN_IDENTITY_GATE = 0.62
DENSE_CLUSTER_MIN_IDENTITY_GATE = 0.45  # dense_expansion / cluster_expansion 共用
COOC_MIN_IDENTITY_GATE = 0.35
FINAL_MIN_TERM_SCORE = 0.15
EXPANSION_ROLES = frozenset({"dense_expansion", "cluster_expansion", "cooc_expansion"})


def passes_identity_gate(rec: Dict[str, Any]) -> bool:
    """按 term_role 判身份闸门。支持 primary | dense_expansion | cluster_expansion | cooc_expansion。"""
    role = (rec.get("term_role") or "").strip().lower()
    identity = float(rec.get("identity_score") or 0.0)
    tid = rec.get("tid") or rec.get("vid") or ""
    term = (rec.get("term") or "")[:24]
    if not role:
        return True
    threshold = None
    if role == "primary":
        threshold = PRIMARY_MIN_IDENTITY_GATE
        passed = identity >= threshold
    elif role == "dense_expansion" or role == "cluster_expansion":
        threshold = DENSE_CLUSTER_MIN_IDENTITY_GATE
        passed = identity >= threshold
    elif role == "cooc_expansion":
        threshold = COOC_MIN_IDENTITY_GATE
        passed = identity >= threshold
    else:
        threshold = COOC_MIN_IDENTITY_GATE
        passed = identity >= threshold
    if STAGE3_DEBUG and not passed:
        print(f"[Stage3] identity_gate 未通过 tid={tid} term={term!r} role={role} identity={identity:.3f} < {threshold}")
    return passed


def passes_topic_consistency(rec: Dict[str, Any], active_domains: Optional[Any] = None) -> bool:
    """Topic 一致性闸门。先跑通阶段直接通过，后续再接 vocabulary_topic_stats。"""
    return True


def score_term_expansion_quality(label, rec: Dict[str, Any]) -> float:
    """质量分：仅衡量「作为召回 term 好不好用」。先简化为 idf 骨架 + 语义/纯度。"""
    tid = rec.get("tid")
    degree_w = int(rec.get("degree_w") or 0)
    degree_w_expanded = int(rec.get("degree_w_expanded") or 0) or max(degree_w, 1)
    target_degree_w = int(rec.get("target_degree_w") or 0)
    total_work_count = float(getattr(label, "total_work_count", 1e6) or 1e6)
    idf_backbone = _idf_backbone(total_work_count, degree_w_expanded)
    idf_val = _smoothed_idf(degree_w, idf_backbone)
    purity = target_degree_w / degree_w_expanded if degree_w_expanded else 0.0
    sim = max(0.0, float(rec.get("sim_score") or 0.0))
    quality = idf_val * (0.5 + 0.5 * purity) * (0.5 + 0.5 * min(1.0, sim))
    return max(0.0, min(1.0, quality))


def get_topic_weight_by_role(term_role: str) -> float:
    """按 term_role 返回三层领域权重（乘性融合用）。"""
    role = (term_role or "").strip().lower()
    if role == "primary":
        return TOPIC_WEIGHT_PRIMARY
    if role == "dense_expansion":
        return TOPIC_WEIGHT_DENSE
    if role == "cluster_expansion":
        return TOPIC_WEIGHT_CLUSTER
    if role == "cooc_expansion":
        return TOPIC_WEIGHT_COOC
    return 0.0


def _get_source_weight(rec: Dict[str, Any]) -> float:
    """按来源 source 返回 source_weight。"""
    s = (rec.get("source") or rec.get("origin") or "").strip().lower()
    if s == "similar_to":
        return SOURCE_WEIGHT_SIMILAR_TO
    if s == "jd_vector":
        return SOURCE_WEIGHT_JD_VECTOR
    if s == "dense":
        return SOURCE_WEIGHT_DENSE
    if s == "cluster":
        return SOURCE_WEIGHT_CLUSTER
    if s == "cooc":
        return SOURCE_WEIGHT_COOC
    return 1.0


def _get_domain_gate(rec: Dict[str, Any]) -> float:
    """domain_gate：三层领域匹配，乘回 final_score。缺 domain_fit 时退化为 DOMAIN_GATE_MIN，避免漏传反而无惩罚。"""
    default_fit = DOMAIN_GATE_MIN  # 缺失或 None 时保守默认，不用 1.0
    raw = rec.get("domain_fit", default_fit)
    if raw is None:
        raw = default_fit
    domain_fit = float(raw)
    return DOMAIN_GATE_MIN + (1.0 - DOMAIN_GATE_MIN) * max(0.0, min(1.0, domain_fit))


def _get_role_penalty(rec: Dict[str, Any]) -> float:
    """role_penalty：任务核心/抽象/载体/噪声。暂无细分时按 term_role 给权。"""
    role = (rec.get("term_role") or "").strip().lower()
    if role == "primary":
        return 1.0
    if role == "dense_expansion":
        return 0.95
    if role == "cluster_expansion":
        return 0.90
    if role == "cooc_expansion":
        return 0.85
    return 1.0


def _get_expansion_penalty(rec: Dict[str, Any]) -> float:
    """expansion_penalty：对 cluster/cooc 额外惩罚。"""
    role = (rec.get("term_role") or "").strip().lower()
    if role == "cluster_expansion":
        return CLUSTER_EXPANSION_PENALTY
    if role == "cooc_expansion":
        return COOC_EXPANSION_PENALTY
    return 1.0


def compose_term_final_score(rec: Dict[str, Any]) -> float:
    """
    final_score = base_score * source_weight * domain_gate * task_consistency * role_penalty * expansion_penalty
    其中：source_weight 看来源；domain_gate 看三层领域匹配；task_consistency 看 JD 语义+强锚点共振；
    role_penalty 看是否任务核心/抽象/载体/噪声；expansion_penalty 对 cluster/cooc 额外惩罚。
    """
    identity = float(rec.get("identity_score") or 0.0)
    quality = float(rec.get("quality_score") or 0.0)
    role = (rec.get("term_role") or "").strip().lower()
    if role == "primary":
        base_score = 0.7 * identity + 0.3 * quality
    elif role == "dense_expansion" or role == "cluster_expansion":
        base_score = 0.4 * identity + 0.6 * quality
    elif role == "cooc_expansion":
        base_score = 0.3 * identity + 0.7 * quality
    else:
        base_score = 0.5 * identity + 0.5 * quality

    source_weight = _get_source_weight(rec)
    domain_gate = _get_domain_gate(rec)
    topic_align = float(rec["topic_align"]) if "topic_align" in rec else 1.0
    topic_weight = get_topic_weight_by_role(role)
    task_consistency = 1.0 - topic_weight + topic_weight * topic_align
    if role in EXPANSION_ROLES and topic_align < TOPIC_MIN_ALIGN:
        task_consistency *= TOPIC_LOW_ALIGN_PENALTY
    role_penalty = _get_role_penalty(rec)
    expansion_penalty = _get_expansion_penalty(rec)

    final_score = base_score * source_weight * domain_gate * task_consistency * role_penalty * expansion_penalty

    if STAGE3_DEBUG and final_score < FINAL_MIN_TERM_SCORE:
        tid = rec.get("tid") or rec.get("vid") or ""
        term = (rec.get("term") or "")[:24]
        print(
            f"[Stage3] final_score 低于阈值 tid={tid} term={term!r} role={role} score={final_score:.3f} < {FINAL_MIN_TERM_SCORE} "
            f"(base={base_score:.3f} domain_gate={domain_gate:.3f} task_cons={task_consistency:.3f})"
        )
    return final_score

