import math


def term_resonance_factor(resonance: float, anchor_resonance: float) -> float:
    """
    词级学术共鸣因子接口（占位实现）。

    第三轮目标：复杂指标先统一置为 1，保留主骨架因子（sim/idf/purity/size_penalty/anchor_factor）。
    未来若要恢复真实逻辑，可在此重写：
      - anchor_resonance > 0 时 1+log1p(resonance)，否则 0.1 等。
    """
    return 1.0


def term_convergence_bonus(hit_count_factor: float, hit_count: int, resonance_factor: float) -> float:
    """
    词级收敛奖励接口（占位实现）。

    真实逻辑原为：hit_count_factor * log1p(hit_count) * resonance_factor。
    目前统一返回 1.0，方便只观察主骨架行为。
    """
    return 1.0


def term_cooc_span_penalty(cooc_span: float) -> float:
    """
    词级共现领域跨度惩罚接口（占位实现）。

    真实逻辑原为：1 / (1 + log1p(cooc_span))。
    目前统一返回 1.0。
    """
    return 1.0


def term_cooc_purity_bonus(cooc_purity: float) -> float:
    """
    词级共现目标领域纯度奖励接口（占位实现）。

    真实逻辑原为：1 + log1p(cooc_purity)。
    目前统一返回 1.0。
    """
    return 1.0


def cluster_gating_factor(tid: int, cluster_factors) -> float:
    """
    簇级 gating 接口（占位实现）。

    目前不对 cluster 做额外 gating，统一返回 1.0。
    若未来需要引入 task-specific cluster 调整，可在此重写。
    """
    return 1.0


def size_penalty_factor(degree_w: int, degree_w_expanded: int) -> float:
    """
    词级 size_penalty 接口（占位实现）。

    真实逻辑可以用 degree_w / P95 或小词削顶来定义。
    当前一律返回 1.0。
    """
    return 1.0


def term_extra_factors(
    rec: dict,
    cos_sim: float,
    degree_w: int,
    degree_w_expanded: int,
    cov_j: float,
    domain_span: int,
    tag_purity: float,
    task_anchor_sim,
    carrier_anchor_sim,
    max_anchor_sim,
) -> float:
    """
    所有“非主骨架”词级修饰项的统一接口（占位实现）。

    包括但不限于：
      - 共鸣（resonance/anchor_resonance）及其硬/软门控
      - 命中次数相关的共振奖励（log(1+hit_count) 等）
      - 岗位泛词惩罚（job_penalty）
      - 领域跨度惩罚（domain_span_penalty）
      - 共现 cooc_span / cooc_purity
      - 角色分层（task_core/abstract/carrier/noise/role_penalty）
      - cluster_gating 及簇内 rank 衰减等

    当前统一返回 1.0，只保留主骨架五因子作用。
    后续可以在此逐条恢复对应维度的逻辑。
    """
    return 1.0

