import math
from typing import Iterable, Any

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

