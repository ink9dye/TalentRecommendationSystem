import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.recall.label_means import advanced_metrics as label_means_adv


def _task_core_text_factor(term: str) -> float:
    """
    仅基于 term 文本形态，对“任务骨干”类词做轻量增强，避免完全依赖复杂指标。
    目标是优先扶起运动学/动力学/规划/控制/估计/仿真等核心技术词。
    """
    if not term:
        return 1.0
    t = term.lower()

    core_hints = [
        # 英文任务骨干
        "kinematic",
        "dynamic",
        "motion",
        "trajectory",
        "planning",
        "optimization",
        "optimal control",
        "mpc",
        "lqr",
        "ilqr",
        "ddp",
        "state estimation",
        "estimation",
        "observer",
        "whole-body",
        "whole body",
        "sim-to-real",
        "simulation",
        # 中文任务骨干
        "运动学",
        "动力学",
        "轨迹",
        "规划",
        "最优控制",
        "状态估计",
        "全身控制",
        "仿真",
        "仿真到实机",
    ]
    if any(h in t for h in core_hints):
        return 1.2

    # 次一级：robot + control/kinematics/trajectory 组合
    if "robot" in t and any(x in t for x in ["control", "kinematic", "trajectory"]):
        return 1.1

    return 1.0


def _robot_entity_text_penalty(term: str) -> float:
    """
    对“泛机器人实体词”（robot hand / modular robot / UGV / Robot vision 等）
    做一档纯形态降权，避免其在缺乏任务骨干修饰时占据 Top。
    """
    if not term:
        return 1.0
    t = term.lower()

    robot_like = any(x in t for x in ["robot", "robotic", "robotics", "vehicle", "arm", "hand", "ugv", "uav"])
    task_like = any(
        x in t
        for x in [
            "kinematic",
            "dynamic",
            "motion",
            "trajectory",
            "planning",
            "optimization",
            "control",
            "estimation",
            "whole-body",
            "whole body",
            "sim-to-real",
            "仿真",
            "运动学",
            "动力学",
            "轨迹",
            "规划",
            "最优控制",
            "状态估计",
        ]
    )

    # 机器人/载具实体，但缺少任务骨干修饰 → 视作泛实体
    if robot_like and not task_like:
        return 0.4

    # 明显噪声控制词
    noisy_control = [
        "industrial control system",
        "emotional control",
        "psychological control",
        "chemical control",
    ]
    if any(x in t for x in noisy_control):
        return 0.3

    return 1.0


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

        # 纯文本形态上的轻量调整：优先扶起任务骨干词，压制泛机器人实体词，
        # 避免完全依赖复杂统计/向量指标。
        if dynamic_weight > 0.0:
            dynamic_weight *= _task_core_text_factor(term_text)
            dynamic_weight *= _robot_entity_text_penalty(term_text)

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

