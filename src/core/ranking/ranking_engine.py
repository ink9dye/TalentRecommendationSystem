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
        """
        执行精排与召回 1:1 权重融合重排逻辑。
        """
        # 1. 批量获取精排模型原始得分 (基于 KGATAX 向量空间)
        # 调用已优化为局部计算的 scorer.compute_scores，返回长度为 500 的 Tensor
        kgat_scores = self.scorer.compute_scores(real_job_ids, candidate_raw_ids)

        # 2. 生成“召回顺序分” (Recall Rank Score)
        # 逻辑：召回列表中第 1 名(index 0) 得 1.0 分，最后一名得 0.0 分
        recall_len = len(candidate_raw_ids)
        recall_scores = torch.linspace(1.0, 0.0, steps=recall_len).to(kgat_scores.device)

        # 3. 对精排原始分进行归一化 (Min-Max Normalization)
        # 确保 KGAT 分数区间被压缩到 [0, 1]，使其能与召回顺序分在同一量级对等融合
        kgat_min = kgat_scores.min()
        kgat_max = kgat_scores.max()
        # 加 1e-8 防止除以 0
        kgat_norm = (kgat_scores - kgat_min) / (kgat_max - kgat_min + 1e-8)

        # 4. 执行 1:1 权重融合
        # 最终得分 = 50% 语义精排能力 + 50% 召回相关性顺序
        final_fusion_scores = 0.5 * kgat_norm + 0.5 * recall_scores

        # 5. 选取融合后的 Top 100
        top_k = min(100, recall_len)
        top_val, top_idx = torch.topk(final_fusion_scores, top_k)

        results = []
        # 使用第 1 个锚点岗位作为解释器的背景参考
        reference_job_id = real_job_ids[0]

        for i in range(top_k):
            # 获取在原始候选名单中的索引
            original_idx = top_idx[i].item()
            raw_aid = str(candidate_raw_ids[original_idx])

            results.append({
                "rank": i + 1,
                "author_id": raw_aid,
                "score": round(float(top_val[i].item()), 4),  # 融合后的最终分
                "kgat_score": round(float(kgat_norm[original_idx].item()), 4),  # 归一化后的精排贡献
                "recall_score": round(float(recall_scores[original_idx].item()), 4),  # 召回顺序贡献
                "metrics": self._fetch_sqlite_stats(raw_aid),
                "evidence_chain": self.explainer.explain(raw_aid, real_job_ids)
            })

        return results
    def _fetch_sqlite_stats(self, raw_id):
        """获取学术指标并修复键名映射"""
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT name, h_index, cited_by_count FROM authors WHERE author_id=?", (raw_id,)).fetchone()
        conn.close()
        return {"name": row[0], "h_index": row[1], "citations": row[2]} if row else {}