# -*- coding: utf-8 -*-
"""
统一候选池结构：CandidateRecord、CandidatePool、PoolDebugSummary。

供总召回（total_recall）构建候选主表与证据明细，供训练与精排直接读取特征，
避免在训练/精排/解释阶段重复查库。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CandidateRecord:
    """
    候选池中一位作者的统一记录（一人一条）。
    除候选池分与来源外，含作者静态指标、query-author 交叉特征与辅助标记，供训练与精排直接使用。
    """
    # --- 基础标识 ---
    author_id: str
    author_name: Optional[str] = None

    # --- 路径来源 ---
    from_vector: bool = False
    from_label: bool = False
    from_collab: bool = False
    path_count: int = 0

    # --- 各路表现 ---
    vector_rank: Optional[int] = None
    label_rank: Optional[int] = None
    collab_rank: Optional[int] = None
    vector_score_raw: Optional[float] = None
    label_score_raw: Optional[float] = None
    collab_score_raw: Optional[float] = None

    # --- 融合分 ---
    rrf_score: float = 0.0
    multi_path_bonus: float = 0.0
    pair_path_bonus: float = 0.0
    candidate_pool_score: float = 0.0
    is_multi_path_hit: bool = False

    # --- 作者静态指标（KGAT-AX/训练用，第一版可 None）---
    h_index: Optional[float] = None
    works_count: Optional[int] = None
    cited_by_count: Optional[int] = None
    recent_works_count: Optional[int] = None
    recent_citations: Optional[int] = None
    institution_level: Optional[float] = None
    top_work_quality: Optional[float] = None

    # --- query-author 交叉（精排用，第一版可粗算）---
    topic_similarity: Optional[float] = None
    skill_coverage_ratio: Optional[float] = None
    domain_consistency: Optional[float] = None
    paper_hit_strength: Optional[float] = None
    recent_activity_match: Optional[float] = None

    # --- 标签路摘要特征（供硬过滤与精排）---
    label_term_count: int = 0
    label_core_term_count: int = 0
    label_support_term_count: int = 0
    label_risky_term_count: int = 0
    label_best_term_score: float = 0.0

    # --- 候选池辅助标记 ---
    bucket_type: str = ""
    passed_hard_filter: bool = True
    dominant_recall_path: Optional[str] = None
    hard_filter_reasons: List[str] = field(default_factory=list)
    bucket_reasons: Optional[str] = None

    # --- 证据（可为 None）---
    vector_evidence: Optional[Dict[str, Any]] = None
    label_evidence: Optional[Dict[str, Any]] = None
    collab_evidence: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """便于序列化或送入精排/训练。"""
        return {
            "author_id": self.author_id,
            "author_name": self.author_name,
            "from_vector": self.from_vector,
            "from_label": self.from_label,
            "from_collab": self.from_collab,
            "path_count": self.path_count,
            "vector_rank": self.vector_rank,
            "label_rank": self.label_rank,
            "collab_rank": self.collab_rank,
            "vector_score_raw": self.vector_score_raw,
            "label_score_raw": self.label_score_raw,
            "collab_score_raw": self.collab_score_raw,
            "rrf_score": self.rrf_score,
            "multi_path_bonus": self.multi_path_bonus,
            "pair_path_bonus": self.pair_path_bonus,
            "candidate_pool_score": self.candidate_pool_score,
            "label_term_count": self.label_term_count,
            "label_core_term_count": self.label_core_term_count,
            "label_support_term_count": self.label_support_term_count,
            "label_risky_term_count": self.label_risky_term_count,
            "label_best_term_score": self.label_best_term_score,
            "is_multi_path_hit": self.is_multi_path_hit,
            "h_index": self.h_index,
            "works_count": self.works_count,
            "cited_by_count": self.cited_by_count,
            "recent_works_count": self.recent_works_count,
            "recent_citations": self.recent_citations,
            "institution_level": self.institution_level,
            "top_work_quality": self.top_work_quality,
            "topic_similarity": self.topic_similarity,
            "skill_coverage_ratio": self.skill_coverage_ratio,
            "domain_consistency": self.domain_consistency,
            "paper_hit_strength": self.paper_hit_strength,
            "recent_activity_match": self.recent_activity_match,
            "bucket_type": self.bucket_type,
            "passed_hard_filter": self.passed_hard_filter,
            "dominant_recall_path": self.dominant_recall_path,
            "hard_filter_reasons": self.hard_filter_reasons,
            "bucket_reasons": self.bucket_reasons,
            "vector_evidence": self.vector_evidence,
            "label_evidence": self.label_evidence,
            "collab_evidence": self.collab_evidence,
        }


@dataclass
class PoolDebugSummary:
    """候选池各阶段统计，供调参与排查。"""
    v_raw_count: int = 0
    l_raw_count: int = 0
    c_raw_count: int = 0
    v_after_quota: int = 0
    l_after_quota: int = 0
    c_after_quota: int = 0
    before_dedup_count: int = 0
    after_dedup_count: int = 0
    hard_filtered_count: int = 0
    bucket_a_count: int = 0
    bucket_b_count: int = 0
    bucket_c_count: int = 0
    bucket_d_count: int = 0
    bucket_e_count: int = 0
    bucket_f_count: int = 0
    final_pool_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v_raw_count": self.v_raw_count,
            "l_raw_count": self.l_raw_count,
            "c_raw_count": self.c_raw_count,
            "v_after_quota": self.v_after_quota,
            "l_after_quota": self.l_after_quota,
            "c_after_quota": self.c_after_quota,
            "before_dedup_count": self.before_dedup_count,
            "after_dedup_count": self.after_dedup_count,
            "hard_filtered_count": self.hard_filtered_count,
            "bucket_a_count": self.bucket_a_count,
            "bucket_b_count": self.bucket_b_count,
            "bucket_c_count": self.bucket_c_count,
            "bucket_d_count": self.bucket_d_count,
            "bucket_e_count": self.bucket_e_count,
            "bucket_f_count": self.bucket_f_count,
            "final_pool_size": self.final_pool_size,
        }


@dataclass
class CandidatePool:
    """
    一轮查询的候选池结果，显式三块：
    - candidate_records：候选主表，一人一条
    - candidate_evidence_rows：证据明细表，一人多条
    - pool_debug_summary：统计信息
    """
    query_text: str = ""
    applied_domains: Optional[str] = None
    candidate_records: List[CandidateRecord] = field(default_factory=list)
    candidate_evidence_rows: List[Dict[str, Any]] = field(default_factory=list)
    pool_debug_summary: Optional[PoolDebugSummary] = None
    path_costs: Optional[Dict[str, float]] = None
    domain_debug: Optional[Dict[str, Any]] = None
