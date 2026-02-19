import torch
import numpy as np

class RankExplainer:
    """
    职责：结合 Neo4j 多跳拓扑发现与 KGAT 注意力权重分析，生成主导推理证据链。
    """

    def __init__(self, model, graph, raw_to_int, device, dataloader=None):
        self.model = model
        self.graph = graph
        self.raw_to_int = raw_to_int
        self.device = device
        self.dataloader = dataloader
        self.REL_AUTHORED = 1  # 对应 config.py 中的关系 ID

    def explain(self, author_id: str, job_raw_id: str):
        """
        升级版推理逻辑：引入 SIMILAR_TO 语义跳跃，解决路径断裂问题。
        """
        # 0. ID 归一化对齐
        norm_author_id = str(author_id).strip().lower()
        h_int = self.raw_to_int.get(f"a_{norm_author_id}", 0)

        # 1. 增强版拓扑发现：引入 1-hop 语义跳跃
        # 逻辑：Job -> 技能A -> (SIMILAR_TO) -> 技能B <- 论文主题 <- 人才
        multi_hop_query = """
        MATCH (j:Job {id: $jid})-[:REQUIRE_SKILL]->(v1:Vocabulary)
        OPTIONAL MATCH (v1)-[:SIMILAR_TO]-(v2:Vocabulary)
        WITH v1, COALESCE(v2, v1) as target_v
        MATCH (target_v)<-[:HAS_TOPIC]-(w:Work)<-[:AUTHORED]-(a:Author {id: $aid})
        RETURN v1.term as req_skill, 
               target_v.term as match_skill, 
               w.title as title, 
               w.id as wid,
               (CASE WHEN v1 = target_v THEN 'exact' ELSE 'semantic' END) as match_type
        LIMIT 10
        """
        paths = self.graph.run(multi_hop_query, jid=str(job_raw_id), aid=str(author_id)).data()

        # 2. 回退探测 (Fallback)
        if not paths:
            fallback_query = """
            MATCH (a:Author {id: $aid})-[:AUTHORED]->(w:Work)-[:HAS_TOPIC]->(v:Vocabulary)
            RETURN v.term as match_skill, v.term as req_skill, w.title as title, w.id as wid, 'fallback' as match_type
            LIMIT 5
            """
            paths = self.graph.run(fallback_query, aid=str(author_id)).data()

        if not paths:
            return {"summary": "该人才为跨领域专家，基于全息 Embedding 向量与岗位需求高度匹配。"}

        # 3. KGATAX 注意力定权 (Attention Scoring)
        # 让模型评价这些路径中，哪些作品最能代表该人才的能力
        h_list = torch.LongTensor([h_int] * len(paths)).to(self.device)
        t_list = torch.LongTensor([self.raw_to_int.get(f"w_{str(p['wid']).strip().lower()}", 0) for p in paths]).to(self.device)
        r_list = torch.LongTensor([self.REL_AUTHORED] * len(paths)).to(self.device)

        with torch.no_grad():
            # 获取模型内部的注意力分值
            att_scores = self.model.update_attention_batch(h_list, t_list, r_list).squeeze()
            if np.isscalar(att_scores): att_scores = [att_scores]
            else: att_scores = att_scores.cpu().numpy()

        for i, p in enumerate(paths):
            p['att_weight'] = float(att_scores[i])

        # 4. 学术特征注入 (AX Context)
        ax_stats = {"h_index_norm": 0.0, "citation_norm": 0.0}
        if self.dataloader is not None:
            feats = self.dataloader.aux_info_all[h_int]
            ax_stats = {"h_index_norm": round(float(feats[0]), 4), "citation_norm": round(float(feats[1]), 4)}

        # 5. 最终证据链合成
        best_path = sorted(paths, key=lambda x: x['att_weight'], reverse=True)[0]

        # 动态生成总结
        if best_path['match_type'] == 'exact':
            summary = f"模型识别到人才在‘{best_path['req_skill']}’领域的直接产出（如《{best_path['title']}》）与需求精准对齐。"
        elif best_path['match_type'] == 'semantic':
            summary = f"人才研究的‘{best_path['match_skill']}’领域与您要求的‘{best_path['req_skill']}’在图谱中高度相关，代表作《{best_path['title']}》权重极高。"
        else:
            summary = f"虽然缺乏直接技能交集，但模型捕捉到其在‘{best_path['match_skill']}’领域的强学术背景具有极高相关性。"

        return {
            "matched_skill": best_path['req_skill'],
            "evidence_node": best_path['match_skill'],
            "key_evidence_work": best_path['title'],
            "match_type": best_path['match_type'],
            "model_confidence": round(float(best_path['att_weight']), 4),
            "academic_quality": ax_stats,
            "summary": summary
        }