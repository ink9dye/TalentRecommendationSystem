from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


class AuthorScoreResult:
    """
    统一封装“论文 → 作者”层面的聚合结果，供各路召回共用。

    - author_scores: 作者总分（aid -> float），可直接用于排序。
    - author_top_works: 每个作者贡献度最高的若干篇论文（aid -> [(wid, contrib_score), ...]），
      便于后续做代表作展示与调试。
    """

    def __init__(
        self,
        author_scores: Dict[str, float],
        author_top_works: Dict[str, List[Tuple[str, float]]],
    ) -> None:
        self.author_scores = author_scores
        self.author_top_works = author_top_works

    def sorted_authors(self, limit: Optional[int] = None) -> List[str]:
        """
        按作者总分从高到低返回作者 ID 列表，可选截断前 limit 名。
        """
        items = sorted(self.author_scores.items(), key=lambda x: x[1], reverse=True)
        if limit is not None and limit > 0:
            items = items[:limit]
        return [aid for aid, _ in items]


def accumulate_author_scores(
    papers: Iterable[dict],
    top_k_per_author: Optional[int] = None,
) -> AuthorScoreResult:
    """
    将“论文级别的贡献分”按作者拆分并聚合为作者总分。

    统一入参格式（无论向量路 / 标签路）：
        papers = [
            {
                "wid": "W123",           # 论文 ID
                "score": 0.0123,         # 该论文在本次查询下的基础贡献度（已含 time_decay 等，但尚未按作者拆分）
                "authors": [
                    {"aid": "A1", "pos_weight": 1.0},
                    {"aid": "A2", "pos_weight": 0.7},
                    ...
                ],
            },
            ...
        ]

    拆分规则：
        对于单篇论文 w，设其基础贡献度为 score_w，
        全体作者的签名权重为 {pos_weight_i}，则：
            frac_i = pos_weight_i / Σ_j pos_weight_j
            contrib_{i, w} = score_w * frac_i

        若某篇论文 pos_weight 全为 0，则退化为 1/n 均分。

    :param top_k_per_author:
        - 若为正整数：仅保留每个作者贡献度最高的前 K 篇论文参与累计（例如向量路可用 3）。
        - 若为 None：累加该作者所有正贡献论文（例如标签路当前行为）。
    :return: AuthorScoreResult
    """
    # 先按 (作者, 论文) 维度保存拆分后的贡献分，便于后续做 per-author TopK
    author_work_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

    for p in papers:
        base_score = float(p.get("score", 0.0) or 0.0)
        if base_score <= 0:
            continue

        wid = str(p.get("wid") or "")
        authors = p.get("authors") or []
        if not wid or not authors:
            continue

        # 收集当前论文内所有作者的 pos_weight
        weights: List[Tuple[str, float]] = []
        total_pos = 0.0
        for a in authors:
            aid = a.get("aid")
            if aid is None:
                continue
            w = float(a.get("pos_weight", 1.0) or 0.0)
            w = max(0.0, w)
            weights.append((str(aid), w))
            total_pos += w

        if not weights:
            continue

        # 若 pos_weight 全为 0，则退化为 1/n 均分
        if total_pos <= 0:
            n = len(weights)
            if n <= 0:
                continue
            share = base_score / n
            for aid, _ in weights:
                prev = author_work_scores[aid].get(wid, 0.0)
                author_work_scores[aid][wid] = prev + share
            continue

        # 正常情况：按 pos_weight / Σpos_weight 分摊本篇论文贡献
        for aid, w in weights:
            frac = w / total_pos
            contrib = base_score * frac
            if contrib <= 0:
                continue
            prev = author_work_scores[aid].get(wid, 0.0)
            author_work_scores[aid][wid] = prev + contrib

    # 对每个作者做 TopK 聚合
    # 关键改动（第一优先）：
    # - 不再线性累加同一作者的多篇论文贡献，改为按作者内排名递减累计。
    # - 目的：抑制“同一 term 下多篇高分论文线性吃满”导致的作者榜失衡。
    DECAY_BY_RANK = [1.00, 0.55, 0.30, 0.18, 0.10]
    author_scores: Dict[str, float] = {}
    author_top_works: Dict[str, List[Tuple[str, float]]] = {}

    for aid, work_map in author_work_scores.items():
        items = list(work_map.items())  # (wid, contrib_score)
        items.sort(key=lambda x: x[1], reverse=True)

        if top_k_per_author is not None and top_k_per_author > 0:
            items = items[:top_k_per_author]

        weighted_items: List[Tuple[str, float]] = []
        total = 0.0
        for i, (wid, score) in enumerate(items):
            if i < len(DECAY_BY_RANK):
                factor = DECAY_BY_RANK[i]
            else:
                factor = 0.06
            adj = float(score) * float(factor)
            if adj <= 0:
                continue
            weighted_items.append((wid, adj))
            total += adj
        if total <= 0:
            continue

        author_scores[aid] = total
        author_top_works[aid] = weighted_items

    return AuthorScoreResult(author_scores, author_top_works)

