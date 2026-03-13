from datetime import datetime
from typing import Iterable, List, Optional, Tuple
import math


def _safe_year(year: Optional[int], default: int = 2000) -> int:
    """
    将 year 安全转换为合理的公历年份。
    """
    try:
        y = int(year)
    except (TypeError, ValueError):
        return default

    current_year = datetime.now().year
    if y < 1900 or y > current_year:
        return default
    return y


def compute_paper_recency(year: Optional[int],
                          domain_ids: Optional[Iterable[str]] = None) -> float:
    """
    论文层时间权重（recency），使用阶梯式 bucket，而不是指数衰减。

    当前简单按年龄分段：
      0-3 年  -> 1.0
      3-6 年  -> 0.6
      6-10 年 -> 0.3
      10+ 年  -> 0.05

    参数 domain_ids 目前未使用，保留是为了后续支持“按领域定制 bucket”时保持接口兼容。
    """
    _ = domain_ids  # 占位，避免未使用参数告警

    y = _safe_year(year, default=2000)
    current_year = datetime.now().year
    age = max(0, current_year - y)

    if age <= 3:
        return 1.0
    elif age <= 6:
        return 0.5
    elif age <= 10:
        return 0.2
    else:
        # 10 年以上论文仅保留极小权重，显著提升近期工作的相对贡献度
        return 0.01


def compute_author_time_features(years: List[int]) -> Tuple[float, float, float]:
    """
    作者层时间特征：同时刻画活跃度 (activity) 与科研动量 (momentum)。

    输入:
      years: 该作者所有论文的年份列表（如 [2015, 2018, 2019, 2021, 2023]）

    输出:
      activity: 1 + log(1 + papers_last_5y)
      momentum: 剪裁后的 (papers_last_3y + 1) / (papers_prev_3y + 1)
      time_weight: 推荐的整体时间权重 = activity * log(1 + momentum)
    """
    if not years:
        return 1.0, 1.0, 1.0

    current_year = datetime.now().year

    cleaned: List[int] = []
    for y in years:
        try:
            yi = int(y)
        except (TypeError, ValueError):
            continue
        if 1900 <= yi <= current_year:
            cleaned.append(yi)

    if not cleaned:
        return 1.0, 1.0, 1.0

    last_5_start = current_year - 4  # 含当前年，共 5 年
    last_3_start = current_year - 2
    prev_3_start = current_year - 5
    prev_3_end = current_year - 3

    papers_last_5y = sum(1 for y in cleaned if y >= last_5_start)
    papers_last_3y = sum(1 for y in cleaned if y >= last_3_start)
    papers_prev_3y = sum(1 for y in cleaned if prev_3_start <= y <= prev_3_end)

    # 1) 活跃度：最近 5 年发文越多，越高；无近作作者保持 1.0，不做额外惩罚
    activity = 1.0 + math.log1p(papers_last_5y)

    # 2) 科研动量：最近 3 年 vs 前 3 年，加 1 防止除零，再做上限裁剪
    raw_momentum = (papers_last_3y + 1.0) / (papers_prev_3y + 1.0)
    momentum = min(3.0, raw_momentum)

    # 3) 推荐的整体时间权重：动量再做一次 log 压缩，避免放大过猛
    momentum_bonus = math.log1p(momentum)
    time_weight = activity * momentum_bonus

    return activity, momentum, time_weight


def compute_author_recency_by_latest(years: List[int]) -> float:
    """
    基于“最近代表作年份”的平滑时间权重。

    设计目标（age = 当前年 - 最近作品年份）：
      - age <= 10  年: 权重接近 1.0
      - age ≈ 10  年: 权重约 0.56
      - age ≈ 20  年: 权重约 0.10
      - age ≈ 30+ 年: 截断到下限 0.01，避免完全抹除历史贡献

    公式： recent_factor = max(0.01, 10 ^ -((age / 20) ^ 2))
    """
    if not years:
        return 1.0

    current_year = datetime.now().year

    cleaned: List[int] = []
    for y in years:
        try:
            yi = int(y)
        except (TypeError, ValueError):
            continue
        if 1900 <= yi <= current_year:
            cleaned.append(yi)

    if not cleaned:
        return 1.0

    latest_year = max(cleaned)
    age = max(0, current_year - latest_year)

    exponent = - (age / 20.0) ** 2
    recent_factor = 10.0 ** exponent

    if recent_factor < 0.01:
        recent_factor = 0.01

    return recent_factor

