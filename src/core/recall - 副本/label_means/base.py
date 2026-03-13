from dataclasses import dataclass, field
from typing import Any, Dict, Set


@dataclass
class LabelContext:
    """
    召回主流程的查询级上下文。
    当前仅作为结构壳使用，方便后续在 label_means 内部统一访问 query/领域等信息。
    """

    query_text: str
    query_vector: Any
    active_domains: Set[int]
    dominance: float
    score_map: Dict[str, float] = field(default_factory=dict)
    term_map: Dict[str, str] = field(default_factory=dict)

