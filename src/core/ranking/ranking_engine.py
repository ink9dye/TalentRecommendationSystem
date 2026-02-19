import sqlite3
import torch
from py2neo import Graph
from config import DB_PATH
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
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
        # raw_to_int 包含所有实体 (a_, w_, v_ 等前缀)
        self.raw_to_int = getattr(dataloader, 'entity_to_int', {})
        # raw_user_to_int 专指岗位 ID
        self.raw_user_to_int = getattr(dataloader, 'user_to_int', {})

        # 3. 初始化评分组件
        self.scorer = RankScorer(model, dataloader, self.device)

        # 4. 初始化解释器：传入 dataloader 以便访问 aux_info_all (AX 特征)
        # 注意：这要求 RankExplainer 的 __init__ 也需要接收 dataloader 参数
        self.explainer = RankExplainer(
            model,
            self.graph,
            self.raw_to_int,
            self.device,
            dataloader=self.dataloader
        )
    def execute_rank(self, real_job_ids, candidate_raw_ids):
        # 1. 批量评分 (使用聚合后的 TOP-3 锚点)
        scores = self.scorer.compute_scores(real_job_ids, candidate_raw_ids)

        # 2. 选取 Top 100
        top_k = min(100, len(candidate_raw_ids))
        top_val, top_idx = torch.topk(scores, top_k)

        results = []
        # 使用第 1 个锚点作为证据链生成的参考上下文
        reference_job_id = real_job_ids[0]

        for i in range(top_k):
            raw_aid = str(candidate_raw_ids[top_idx[i].item()])
            results.append({
                "rank": i + 1,
                "author_id": raw_aid,
                "score": round(float(top_val[i].item()), 4),
                "metrics": self._fetch_sqlite_stats(raw_aid),
                # 为每一个人生成证据链
                "evidence_chain": self.explainer.explain(raw_aid, reference_job_id)
            })
        return results

    def _fetch_sqlite_stats(self, raw_id):
        """获取学术指标并修复键名映射"""
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT name, h_index, cited_by_count FROM authors WHERE author_id=?", (raw_id,)).fetchone()
        conn.close()
        return {"name": row[0], "h_index": row[1], "citations": row[2]} if row else {}