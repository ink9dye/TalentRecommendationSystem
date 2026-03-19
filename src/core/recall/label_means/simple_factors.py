import math
from typing import Iterable, Any, Optional

import numpy as np

from src.utils.tools import apply_text_decay


def survey_decay_factor(hit_count: int, raw_title: str) -> float:
    """
    综述 / 文本类型统一衰减因子。
    等价于 LabelRecallPath._survey_decay_factor 中的实现：
      - hit_count > 1 时采用 1 / hit_count^2；
      - 再乘以 apply_text_decay(title)。
    """
    survey_decay = (1.0 / math.pow(hit_count, 2)) if hit_count > 1 else 1.0
    text_decay = apply_text_decay(raw_title or "")
    return survey_decay * text_decay


def coverage_norm_factor(hit_count: int) -> float:
    """
    命中标签数量归一化因子。
    等价于 1 / log(2 + hit_count)，hit_count<=0 时为 1.0。
    """
    if hit_count > 0:
        return 1.0 / math.log(2.0 + hit_count)
    return 1.0


def paper_cluster_bonus(cluster_ids: Iterable[Any]) -> float:
    """
    论文跨 topic cluster 奖励因子。
    等价于 log1p(cluster_count)。
    """
    if not cluster_ids:
        return 1.0
    cluster_ids = list(cluster_ids)
    cluster_count = len(cluster_ids)
    return math.log1p(cluster_count) if cluster_count > 0 else 1.0


def paper_jd_semantic_gate_factor(
    raw_title: str,
    jd_vec,
    encoder,
    paper_vec_precomputed: Optional[np.ndarray] = None,
) -> float:
    """
    论文标题与 JD 的语义相似度门控因子。
    等价于 LabelRecallPath._paper_jd_semantic_gate_factor 中的逻辑：
      - cos < 0.3 -> 0.1
      - 0.3 <= cos < 0.5 -> 0.4
      - 否则 1.0
    paper_vec_precomputed: 与 encoder.encode(raw_title) 同分布的标题向量（可 batch 预计算），传入则不再 encode。
    """
    if jd_vec is None or not raw_title or encoder is None:
        return 1.0
    try:
        if paper_vec_precomputed is not None:
            paper_vec = paper_vec_precomputed
            if getattr(paper_vec, "ndim", 0) == 1:
                paper_vec = paper_vec.reshape(1, -1)
        else:
            paper_vec, _ = encoder.encode(raw_title)
        if paper_vec is None or paper_vec.size == 0:
            return 1.0
        jd_flat = np.asarray(jd_vec, dtype=np.float32).flatten()
        pf = np.asarray(paper_vec, dtype=np.float32).flatten()
        jd_norm = np.linalg.norm(jd_flat)
        pf_norm = np.linalg.norm(pf)
        if jd_norm <= 1e-9 or pf_norm <= 1e-9:
            return 1.0
        cos = float(np.dot(jd_flat / jd_norm, pf / pf_norm))
        cos = max(-1.0, min(1.0, cos))
        if cos < 0.3:
            return 0.1
        if cos < 0.5:
            return 0.4
        return 1.0
    except Exception:
        return 1.0

