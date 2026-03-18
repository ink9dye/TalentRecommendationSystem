import sqlite3
import torch
from typing import List, Optional, Any
from py2neo import Graph
from config import DB_PATH
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
# 引入工具类
from src.utils.domain_utils import DomainProcessor
from src.core.ranking.rank_scorer import RankScorer
from src.core.ranking.rank_explainer import RankExplainer

# 精排三阶段参数（与 README 6.1～6.5 对齐）
PRE_RANK_MAX = 200
LAMBDA_POOL = 0.2
LAMBDA_KGAT = 0.6
LAMBDA_STABILITY = 0.2


class RankingEngine:
    def __init__(self, model, dataloader):
        """
        RankingEngine 初始化：对齐全量 ID 空间并挂载推理子组件。
        """
        self.device = torch.device("cpu")
        self.dataloader = dataloader

        # 1. 初始化图数据库连接，用于路径回溯
        self.graph = Graph(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            name=NEO4J_DATABASE
        )

        # 2. 核心修复：从 dataloader 获取已持久化的映射表
        self.raw_to_int = getattr(dataloader, 'entity_to_int', {})
        self.raw_user_to_int = getattr(dataloader, 'user_to_int', {})

        # 3. 初始化评分组件
        self.scorer = RankScorer(model, dataloader, self.device)

        # 4. 初始化解释器
        self.explainer = RankExplainer(
            model,
            self.graph,
            self.raw_to_int,
            self.device,
            dataloader=self.dataloader
        )

    def execute_rank(
        self,
        real_job_ids,
        candidate_raw_ids,
        filter_domain=None,
        candidate_pool=None,
        lambda_pool=LAMBDA_POOL,
        lambda_kgat=LAMBDA_KGAT,
        lambda_stability=LAMBDA_STABILITY,
    ):
        """
        三阶段精排：候选池预排序 → KGAT-AX 深度精排 → 最终稳定融合。
        支持两种输入：
        - candidate_pool 存在时：使用 candidate_records 做预排序与 rule_stability，再融合 kgatax 分；
        - 否则：仅用 candidate_raw_ids 顺序作召回序，与 KGAT 分按原 40/60 融合（兼容旧逻辑）。
        """
        # --- 解析输入：优先使用候选池 ---
        records = getattr(candidate_pool, "candidate_records", None) if candidate_pool else None
        evidence_rows = getattr(candidate_pool, "candidate_evidence_rows", None) if candidate_pool else None
        if records:
            author_ids_from_pool = [r.author_id for r in records]
            candidate_raw_ids = candidate_raw_ids or author_ids_from_pool
        else:
            records = None
            evidence_rows = None

        # --- 1. 领域并集预过滤 ---
        if records:
            active_records = self._filter_records_by_domain(records, filter_domain)
            active_candidates = [r.author_id for r in active_records]
        else:
            active_candidates = candidate_raw_ids
            if filter_domain and filter_domain != "0":
                active_candidates = self._filter_by_domain_union(candidate_raw_ids, filter_domain)
            active_records = None

        if filter_domain and filter_domain != "0" and not records:
            print(f"[Ranking] 领域并集过滤完成: {len(candidate_raw_ids)} -> {len(active_candidates)} (Pattern: {filter_domain})")

        if not active_candidates:
            print("[Warning] 领域过滤后无可匹配候选人，精排流程提前终止。")
            return []

        # --- 2. 候选池预排序（仅当有 records 且数量超过 PRE_RANK_MAX 时压缩）---
        if active_records and len(active_records) > PRE_RANK_MAX:
            active_records = self._pre_rank(active_records, top_n=PRE_RANK_MAX)
            active_candidates = [r.author_id for r in active_records]
            print(f"[Ranking] 预排序压缩: 保留 top {len(active_candidates)} 进入 KGAT 精排")

        # --- 3. KGAT-AX 深度精排得分（有 candidate_records 且模型启用四分支时使用 calc_score_v2）---
        kgat_scores = self.scorer.compute_scores(
            real_job_ids, active_candidates, candidate_records=active_records
        )

        # --- 4. 候选池基础分与 rule_stability（有 records 时）---
        n = len(active_candidates)
        pool_scores = torch.zeros(n, device=kgat_scores.device)
        rule_stability_scores = torch.zeros(n, device=kgat_scores.device)
        author_to_record = {}
        if active_records:
            for r in active_records:
                author_to_record[r.author_id] = r
            for i, aid in enumerate(active_candidates):
                r = author_to_record.get(aid)
                if r is not None:
                    pool_scores[i] = float(r.candidate_pool_score or 0.0)
                    rule_stability_scores[i] = self._compute_rule_stability(r)
        else:
            recall_scores = torch.linspace(1.0, 0.0, steps=n).to(kgat_scores.device)
            pool_scores = recall_scores

        # --- 5. 归一化 ---
        def _norm(x):
            xmin, xmax = x.min(), x.max()
            return (x - xmin) / (xmax - xmin + 1e-8)
        kgat_norm = _norm(kgat_scores)
        pool_norm = _norm(pool_scores)
        if rule_stability_scores.abs().sum() > 1e-8:
            stab_norm = _norm(rule_stability_scores)
        else:
            stab_norm = torch.zeros_like(rule_stability_scores)

        # --- 6. 最终稳定融合 ---
        final_fusion_scores = (
            lambda_pool * pool_norm + lambda_kgat * kgat_norm + lambda_stability * stab_norm
        )

        # --- 7. 选取 Top 100 ---
        top_k = min(100, n)
        top_val, top_idx = torch.topk(final_fusion_scores, top_k)

        results = []
        for i in range(top_k):
            original_idx = top_idx[i].item()
            raw_aid = str(active_candidates[original_idx])
            rec = author_to_record.get(raw_aid) if author_to_record else None

            stats = self._fetch_sqlite_stats(raw_aid)
            exp_kw = {"author_id": raw_aid, "job_raw_ids": real_job_ids}
            if rec is not None:
                exp_kw["candidate_record"] = rec
            if evidence_rows is not None:
                exp_kw["candidate_evidence_rows"] = [e for e in evidence_rows if e.get("author_id") == raw_aid]
            exp_data = self.explainer.explain(**exp_kw)

            details = {
                "kgat_score": round(float(kgat_norm[original_idx].item()), 4),
                "recall_score": round(float(pool_norm[original_idx].item()), 4),
                "match_type": exp_data.get("match_type"),
            }
            if rule_stability_scores.abs().sum() > 1e-8:
                details["rule_stability"] = round(float(stab_norm[original_idx].item()), 4)
            if rec is not None and rec.candidate_pool_score is not None:
                details["candidate_pool_score"] = round(float(rec.candidate_pool_score), 4)

            results.append({
                "rank": i + 1,
                "author_id": raw_aid,
                "name": stats.get("name"),
                "score": round(float(top_val[i].item()), 4),
                "representative_work": {
                    "work_id": exp_data.get("work_id"),
                    "title": exp_data.get("key_evidence_work"),
                    "link": exp_data.get("work_url"),
                    "published_at": exp_data.get("source"),
                },
                "recommendation_reason": exp_data.get("summary"),
                "metrics": stats,
                "collaboration": exp_data.get("collaborators"),
                "details": details,
            })

        return results

    def _pre_rank(self, records: List[Any], top_n: int = PRE_RANK_MAX) -> List[Any]:
        """
        候选池预排序：用轻量特征对 records 打分，保留 top_n。
        特征：candidate_pool_score、from_label、from_vector、from_collab、path_count、
        bucket_type、domain_consistency、paper_hit_strength、recent_activity_match。
        """
        bucket_rank = {"A": 4, "B": 3, "C": 2, "D": 1, "": 0}
        scored = []
        for r in records:
            base = float(r.candidate_pool_score or 0.0)
            path_bonus = (1.0 if r.from_label else 0.0) * 0.1 + (1.0 if r.from_vector else 0.0) * 0.08 + (1.0 if r.from_collab else 0.0) * 0.05
            path_bonus += min((r.path_count or 0) * 0.03, 0.15)
            domain = float(r.domain_consistency or 0.0)
            paper = float(r.paper_hit_strength or 0.0)
            activity = float(r.recent_activity_match or 0.0)
            bucket = bucket_rank.get((r.bucket_type or "").strip().upper(), 0) * 0.05
            pre_score = base + path_bonus + 0.1 * domain + 0.1 * paper + 0.05 * activity + bucket
            scored.append((pre_score, r))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_n]]

    def _compute_rule_stability(self, record: Any) -> float:
        """
        rule_stability = multi_path_bonus + label_support_bonus - collab_only_penalty
        - low_activity_penalty - weak_paper_evidence_penalty（README 6.5）
        """
        s = 0.0
        s += float(record.multi_path_bonus or 0.0)
        if getattr(record, "from_label", False):
            s += 0.1
        if getattr(record, "from_collab", False) and not getattr(record, "from_vector", False) and not getattr(record, "from_label", False):
            s -= 0.15
        if (getattr(record, "recent_activity_match", None) or 1.0) < 0.2:
            s -= 0.08
        if (getattr(record, "paper_hit_strength", None) or 1.0) < 0.2:
            s -= 0.08
        return max(-0.3, min(0.5, s))

    def _filter_records_by_domain(self, records: List[Any], pattern: Optional[str]):
        """对 candidate_records 做领域并集过滤，保留与 pattern 有交集的作者。"""
        if not pattern or pattern == "0":
            return list(records)
        author_ids = [r.author_id for r in records]
        valid_ids = set(self._filter_by_domain_union(author_ids, pattern))
        return [r for r in records if r.author_id in valid_ids]

    def _filter_by_domain_union(self, author_ids: list, pattern: str):
        """
        【升级版】利用 DomainProcessor 实现精排前的领域硬过滤。
        """
        if not pattern or pattern == "0":
            return author_ids

        try:
            conn = sqlite3.connect(DB_PATH)
            placeholders = ','.join(['?'] * len(author_ids))

            # 解析目标领域为 Set 提高查找效率
            target_set = DomainProcessor.to_set(pattern)

            # 获取候选人在数据库中的领域分布
            sql = f"""
                SELECT a.author_id, w.domain_ids 
                FROM authorships a 
                JOIN works w ON a.work_id = w.work_id 
                WHERE a.author_id IN ({placeholders})
            """
            rows = conn.execute(sql, author_ids).fetchall()
            conn.close()

            # 使用字典聚合作者的领域信息（一个作者可能对应多篇论文）
            valid_authors = set()
            for aid, d_ids in rows:
                # 调用工具类判定交集
                if DomainProcessor.has_intersect(d_ids, target_set):
                    valid_authors.add(aid)

            # 维持原有的召回顺序返回（召回越靠前，在列表中的位置越靠前）
            return [aid for aid in author_ids if aid in valid_authors]
        except Exception as e:
            print(f"[Error] 精排领域并集修剪失败: {e}")
            return author_ids

    def _fetch_sqlite_stats(self, raw_id):
        """
        获取基础学术指标
        """
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT name, h_index, cited_by_count, works_count FROM authors WHERE author_id=?",
            (raw_id,)
        ).fetchone()
        conn.close()

        if row:
            return {
                "name": row[0],
                "h_index": row[1],
                "citations": row[2],
                "total_papers": row[3]
            }
        return {"name": "未知", "h_index": 0, "citations": 0, "total_papers": 0}