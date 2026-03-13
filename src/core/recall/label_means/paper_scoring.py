import math
from typing import Any, Dict, List, Tuple

import numpy as np

from src.utils.domain_utils import DomainProcessor
from src.utils.time_features import compute_paper_recency
from src.core.recall.label_means.simple_factors import (
    survey_decay_factor,
    coverage_norm_factor,
    paper_cluster_bonus,
    paper_jd_semantic_gate_factor,
)


def compute_contribution(
    recall,
    paper: Dict[str, Any],
    context: Dict[str, Any],
) -> Tuple[float, List[str], float, Dict[str, float]]:
    """
    论文级贡献度计算（从 LabelRecallPath._compute_contribution 迁移），
    作为 label_means 层的统一入口。

    参数:
      - recall: 提供 voc_to_clusters/_is_retracted/_get_domain_purity_factor/_calculate_proximity/_query_encoder 等方法与属性的宿主对象
      - paper: 论文结构 {wid, hits, title, year, domains}
      - context: 上下文，至少需要:
          * score_map: term 分数
          * term_map: tid -> term
          * active_domain_set: 当前激活领域集合
          * dominance: 领域主导度
          * query_vector: JD 向量（可选，用于 paper-JD 语义 gate）

    返回:
      - score: 论文最终得分
      - hit_terms: 命中标签的 term 列表
      - rank_score: 仅由标签权重累加得到的基础分
      - term_weights: {vid_s: 加权分}，用于后续作者拆分与 debug
    """
    raw_title = (paper.get("title") or "")

    # 1. 撤稿拦截
    if _is_retracted(raw_title):
        return 0.0, [], 0.0, {}

    # 2. 领域纯度降权
    domain_coeff = _get_domain_purity_factor(
        paper.get("domains"),
        context["active_domain_set"],
        context["dominance"],
    )
    if domain_coeff <= 0:
        return 0.0, [], 0.0, {}

    # 3. 标签匹配与动态权重累加
    score_map: Dict[str, float] = context["score_map"]
    term_map: Dict[str, str] = context["term_map"]

    rank_score = 0.0
    term_weights: Dict[str, float] = {}
    valid_hids: List[int] = []
    hit_terms: List[str] = []

    for hit in paper.get("hits", []):
        vid_s = str(hit["vid"])
        if vid_s in score_map:
            w = float(score_map[vid_s]) * float(hit.get("idf", 0.0))
            rank_score += w
            term_weights[vid_s] = w
            valid_hids.append(hit["vid"])
            hit_terms.append(term_map.get(vid_s, ""))

    if rank_score == 0:
        return 0.0, [], 0.0, {}

    # 4. 综述/文本类型衰减
    hit_count = len(valid_hids)
    survey_decay = survey_decay_factor(hit_count, raw_title)

    # 5. 语义紧密度加成
    proximity = _calculate_proximity(
        recall.vocab_to_idx,
        recall.all_vocab_vectors,
        valid_hids,
    )
    proximity_bonus = math.pow(1.0 + proximity, hit_count)

    # 6. 时间衰减
    time_decay = compute_paper_recency(paper.get("year", 2000), context["active_domain_set"])

    # 6.5 跨簇奖励
    cluster_ids = set()
    for vid in valid_hids:
        try:
            clusters = recall.voc_to_clusters.get(int(vid), [])
        except Exception:
            clusters = []
        if clusters:
            cid, _ = max(clusters, key=lambda x: x[1])
            cluster_ids.add(cid)
    cluster_bonus = paper_cluster_bonus(cluster_ids)

    # 7. 命中标签数量归一化
    coverage_norm = coverage_norm_factor(hit_count)

    # 7.1 组合主干因子
    score = (
        rank_score
        * coverage_norm
        * cluster_bonus
        * proximity_bonus
        * domain_coeff
        * time_decay
        * survey_decay
    )

    # 7.5 paper-level JD semantic gate
    jd_vec = context.get("query_vector")
    gate_factor = paper_jd_semantic_gate_factor(
        raw_title,
        jd_vec,
        getattr(recall, "_query_encoder", None),
    )
    score *= gate_factor

    return float(score), hit_terms, float(rank_score), term_weights


def _is_retracted(title: str) -> bool:
    """
    撤稿拦截：识别论文是否为撤稿通知。
    """
    return bool(title) and "retraction" in title.lower()


def _get_domain_purity_factor(paper_domains_raw, active_set, dominance: float) -> float:
    """
    领域专注度计算（Purity Engine），从 LabelRecallPath._get_domain_purity_factor 迁移。
    """
    paper_domains = DomainProcessor.to_set(paper_domains_raw)

    intersect = paper_domains.intersection(active_set)

    if paper_domains and not intersect:
        return 0.0

    purity_ratio = 1.0
    if paper_domains:
        purity_ratio = len(intersect) / len(paper_domains)

    base_score = 1.0 + (dominance * 5.0) if intersect else 0.5
    return base_score * math.pow(purity_ratio, 6)


def _calculate_proximity(vocab_to_idx: Dict[str, int], all_vocab_vectors, hit_ids: List[int]) -> float:
    """
    计算命中标签在向量空间中的平均余弦相似度（语义紧密度），从 LabelRecallPath._calculate_proximity 迁移。
    """
    if len(hit_ids) < 2:
        return 0.5
    idxs = [vocab_to_idx.get(str(vid)) for vid in hit_ids if str(vid) in vocab_to_idx]
    idxs = [i for i in idxs if i is not None]
    if len(idxs) < 2:
        return 0.5

    vecs = all_vocab_vectors[idxs]
    sim_matrix = np.dot(vecs, vecs.T)
    return float(np.mean(sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)]))

