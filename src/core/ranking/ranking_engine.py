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

    def execute_rank(self, real_job_ids, candidate_raw_ids, filter_domain=None):
        """
        【1:1 平衡版】执行精排与召回等权重融合重排。
        融合逻辑：50% KGATAX 模型分 (内含乘法门控增益) + 50% 召回顺序分。
        """
        # --- 1. 领域并集预过滤 (保持业务红线) ---
        active_candidates = candidate_raw_ids
        if filter_domain:
            active_candidates = self._filter_by_domain_union(candidate_raw_ids, filter_domain)
            print(f"[Ranking] 领域并集过滤完成: {len(candidate_raw_ids)} -> {len(active_candidates)} (Pattern: {filter_domain})")

        if not active_candidates:
            print("[Warning] 领域过滤后无可匹配候选人，精排流程提前终止。")
            return []

        # --- 2. 批量获取精排模型得分 ---
        # 核心：此时得分已包含 holographic_fusion 乘法门控对大牛的能量放大
        kgat_scores = self.scorer.compute_scores(real_job_ids, active_candidates)

        # --- 3. 生成“召回顺序分” (Recall Rank Score) ---
        # 逻辑：为召回池中的 500 人生成从 1.0 到 0.0 的线性梯度分
        recall_len = len(active_candidates)
        recall_scores = torch.linspace(1.0, 0.0, steps=recall_len).to(kgat_scores.device)

        # --- 4. 对精排原始分进行归一化 ---
        kgat_min = kgat_scores.min()
        kgat_max = kgat_scores.max()
        kgat_norm = (kgat_scores - kgat_min) / (kgat_max - kgat_min + 1e-8)

        # --- 5. 执行权重融合 ---
        final_fusion_scores = 0.4 * kgat_norm + 0.6 * recall_scores

        # --- 6. 选取融合后的 Top 100 ---
        top_k = min(100, recall_len)
        top_val, top_idx = torch.topk(final_fusion_scores, top_k)

        results = []
        for i in range(top_k):
            # 获取在当前活跃候选名单中的索引
            original_idx = top_idx[i].item()
            raw_aid = str(active_candidates[original_idx])

            # 获取包含 total_papers 的基础统计指标
            stats = self._fetch_sqlite_stats(raw_aid)

            # 调用解释器获取证据链：包含 work_id, work_url 和 summary
            exp_data = self.explainer.explain(raw_aid, real_job_ids)

            # --- 构造符合前端期待的深度结果对象 ---
            results.append({
                "rank": i + 1,
                "author_id": raw_aid,
                "name": stats.get('name'),
                "score": round(float(top_val[i].item()), 4),  # 1:1 融合后的最终分

                # 核心需求 1：代表论文详细信息
                "representative_work": {
                    "work_id": exp_data.get("work_id"),
                    "title": exp_data.get("key_evidence_work"),
                    "link": exp_data.get("work_url"),
                    "published_at": exp_data.get("source")
                },

                # 核心需求 2：清晰的推荐理由
                "recommendation_reason": exp_data.get("summary"),

                # 包含 H-index, total_papers 和总引用的统计指标
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
        【新增辅助方法】利用领域并集正则(如 "1|17|2") 在精排前修剪 500 人候选池。
        """
        import re
        try:
            conn = sqlite3.connect(DB_PATH)  # DB_PATH 来自 config
            placeholders = ','.join(['?'] * len(author_ids))

            # 获取候选人在数据库中的所有领域分布
            sql = f"""
                SELECT DISTINCT a.author_id, w.domain_ids 
                FROM authorships a 
                JOIN works w ON a.work_id = w.work_id 
                WHERE a.author_id IN ({placeholders})
            """
            rows = conn.execute(sql, author_ids).fetchall()
            conn.close()

            valid_authors = set()
            # 编译正则：pattern 形如 "1|17|2|18"
            # 逻辑：只要作者产出的论文领域中包含并集里的任何一个 ID，即视为合格
            # 添加 r 前缀，变为 fr-string
            regex = re.compile(fr"(^|,|\|)({pattern})(,|$|\|)")

            for aid, d_ids in rows:
                if d_ids and regex.search(str(d_ids)):
                    valid_authors.add(aid)

            # 维持原有的召回顺序返回
            return [aid for aid in author_ids if aid in valid_authors]
        except Exception as e:
            print(f"[Error] 精排领域并集修剪失败: {e}")
            return author_ids  # 失败则退回全量，防止流程中断

    def _fetch_sqlite_stats(self, raw_id):
        """
        获取学术指标：包含姓名、H-index、总引用以及新增的总论文数
        """
        import sqlite3
        from config import DB_PATH

        conn = sqlite3.connect(DB_PATH)
        # SQL 查询中加入 works_count 字段以获取人才的总论文产出
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
                "total_papers": row[3]  # 新增字段：总论文数
            }
        return {}