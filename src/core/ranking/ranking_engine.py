import sqlite3
import torch
from py2neo import Graph
from config import DB_PATH
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
# 引入工具类
from src.utils.domain_utils import DomainProcessor
from src.core.ranking.rank_scorer import RankScorer
from src.core.ranking.rank_explainer import RankExplainer


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

    def execute_rank(self, real_job_ids, candidate_raw_ids, filter_domain=None):
        """
        【1:1 平衡版】执行精排与召回等权重融合重排。
        """
        # --- 1. 领域并集预过滤 (改用 DomainProcessor) ---
        active_candidates = candidate_raw_ids
        if filter_domain and filter_domain != "0":
            active_candidates = self._filter_by_domain_union(candidate_raw_ids, filter_domain)
            print(f"[Ranking] 领域并集过滤完成: {len(candidate_raw_ids)} -> {len(active_candidates)} (Pattern: {filter_domain})")

        if not active_candidates:
            print("[Warning] 领域过滤后无可匹配候选人，精排流程提前终止。")
            return []

        # --- 2. 批量获取精排模型得分 ---
        kgat_scores = self.scorer.compute_scores(real_job_ids, active_candidates)

        # --- 3. 生成“召回顺序分” ---
        recall_len = len(active_candidates)
        recall_scores = torch.linspace(1.0, 0.0, steps=recall_len).to(kgat_scores.device)

        # --- 4. 对精排原始分进行归一化 ---
        kgat_min = kgat_scores.min()
        kgat_max = kgat_scores.max()
        kgat_norm = (kgat_scores - kgat_min) / (kgat_max - kgat_min + 1e-8)

        # --- 5. 执行权重融合 (40% 模型分 + 60% 召回基础分) ---
        final_fusion_scores = 0.4 * kgat_norm + 0.6 * recall_scores

        # --- 6. 选取 Top 100 ---
        top_k = min(100, recall_len)
        top_val, top_idx = torch.topk(final_fusion_scores, top_k)

        results = []
        for i in range(top_k):
            original_idx = top_idx[i].item()
            raw_aid = str(active_candidates[original_idx])

            stats = self._fetch_sqlite_stats(raw_aid)
            exp_data = self.explainer.explain(raw_aid, real_job_ids)

            results.append({
                "rank": i + 1,
                "author_id": raw_aid,
                "name": stats.get('name'),
                "score": round(float(top_val[i].item()), 4),
                "representative_work": {
                    "work_id": exp_data.get("work_id"),
                    "title": exp_data.get("key_evidence_work"),
                    "link": exp_data.get("work_url"),
                    "published_at": exp_data.get("source")
                },
                "recommendation_reason": exp_data.get("summary"),
                "metrics": stats,
                "collaboration": exp_data.get("collaborators"),
                "details": {
                    "kgat_score": round(float(kgat_norm[original_idx].item()), 4),
                    "recall_score": round(float(recall_scores[original_idx].item()), 4),
                    "match_type": exp_data.get("match_type")
                }
            })

        return results

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