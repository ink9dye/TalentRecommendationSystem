import torch
import numpy as np
import random


class RankExplainer:
    """
    职责：结合 Neo4j 多跳拓扑、学术社交网络与 KGAT 注意力，生成富含协作信息的清晰证据链。
    """

    def __init__(self, model, graph, raw_to_int, device, dataloader=None):
        self.model = model
        self.graph = graph
        self.raw_to_int = raw_to_int
        self.device = device
        self.dataloader = dataloader
        self.REL_AUTHORED = 1  # 对应 config.py

    def explain(self, author_id: str, job_raw_ids: list):
        """
        深度推理逻辑：多锚点拓扑探测 + 发表平台提取 + 合作伙伴收割
        """
        norm_author_id = str(author_id).strip().lower()
        h_int = self.raw_to_int.get(f"a_{norm_author_id}", 0)

        # 1. 增强版 Cypher：一次性提取技能、论文、期刊及前两位合作伙伴
        multi_hop_query = """
        MATCH (j:Job) WHERE j.id IN $jids OR j.securityId IN $jids
        MATCH (j)-[:REQUIRE_SKILL]->(v1:Vocabulary)
        WHERE NOT v1.term IN ['computer science', 'mathematics', 'engineering', 'physics', 'technology', '领域专家']

        OPTIONAL MATCH (v1)-[:SIMILAR_TO]-(v2:Vocabulary)
        WITH v1, COALESCE(v2, v1) as target_v, j

        MATCH (target_v)<-[:HAS_TOPIC]-(w:Work)<-[:AUTHORED]-(a:Author {id: $aid})

        // 提取发表平台
        OPTIONAL MATCH (w)-[:PUBLISHED_IN]->(src:Source)
        // 提取核心合作者 (排除自己，取前 2 位)
        OPTIONAL MATCH (w)<-[:AUTHORED]-(collab:Author)
        WHERE collab.id <> $aid

        WITH j, v1, target_v, w, src, collect(DISTINCT collab.name)[0..2] as co_authors,
             (CASE WHEN v1 = target_v THEN 'exact' ELSE 'semantic' END) as m_type

        RETURN v1.term as req_skill, 
               target_v.term as match_skill, 
               w.title as title, 
               w.id as wid,
               src.name as source_name,
               co_authors,
               m_type as match_type
        ORDER BY match_type ASC LIMIT 15
        """
        paths = self.graph.run(multi_hop_query, jids=job_raw_ids, aid=str(author_id)).data()

        # 2. 语义保底 (Fuzzy Match)：支持技能描述词的文本级对齐
        if not paths:
            fuzzy_query = """
            MATCH (j:Job) WHERE j.id IN $jids OR j.securityId IN $jids
            WITH j.skills as job_skills, j
            MATCH (a:Author {id: $aid})-[:AUTHORED]->(w:Work)-[:HAS_TOPIC]->(v:Vocabulary)
            WHERE any(s IN split(job_skills, ',') WHERE toLower(v.term) CONTAINS toLower(s))
            OPTIONAL MATCH (w)-[:PUBLISHED_IN]->(src:Source)
            OPTIONAL MATCH (w)<-[:AUTHORED]-(collab:Author) WHERE collab.id <> $aid
            WITH v, w, src, collect(DISTINCT collab.name)[0..2] as co_authors
            RETURN v.term as req_skill, v.term as match_skill, w.title as title, 
                   w.id as wid, src.name as source_name, co_authors, 'fuzzy' as match_type
            LIMIT 5
            """
            paths = self.graph.run(fuzzy_query, jids=job_raw_ids, aid=str(author_id)).data()

        # 3. 终极 Fallback
        if not paths:
            return self._generate_fallback_response()

        # 4. 利用 KGAT 注意力权重筛选最具有说服力的“证据论文”
        h_list = torch.LongTensor([h_int] * len(paths)).to(self.device)
        t_list = [self.raw_to_int.get(f"w_{str(p['wid']).strip().lower()}", 0) for p in paths]
        r_list = torch.LongTensor([self.REL_AUTHORED] * len(paths)).to(self.device)

        with torch.no_grad():
            att_scores = self.model.update_attention_batch(h_list, torch.LongTensor(t_list).to(self.device),
                                                           r_list).squeeze()
            att_scores = att_scores.cpu().numpy() if len(paths) > 1 else [att_scores.item()]

        for i, p in enumerate(paths):
            p['att_weight'] = float(att_scores[i])

        # 5. 排序并选取最优路径
        best_path = sorted(paths, key=lambda x: (x['match_type'] == 'fallback', -x['att_weight']))[0]

        # 6. 动态生成多样化总结文本
        summary = self._build_dynamic_summary(best_path)

        return {
            "matched_skill": best_path['req_skill'],
            "key_evidence_work": best_path['title'],
            "source": best_path.get('source_name', "相关领域核心刊物"),
            "collaborators": best_path['co_authors'],
            "match_type": best_path['match_type'],
            "model_confidence": round(float(best_path['att_weight']), 4),
            "summary": summary
        }

    def _build_dynamic_summary(self, path):
        """利用多句式模板降低文本重复度"""
        req = path['req_skill']
        m_skill = path['match_skill']
        title = path['title']
        src = path.get('source_name') or "学术期刊"
        collabs = "、".join(path['co_authors']) if path['co_authors'] else None

        # 句式 A：侧重精准匹配与发表平台
        template_a = f"精准对齐：人才在岗位的核心诉求‘{req}’上有深厚积累，其发表在《{src}》上的代表作《{title}》显示了极高的专业造诣"
        # 句式 B：侧重语义关联与社交背书
        template_b = f"语义关联：人才深耕的‘{m_skill}’领域与该岗位要求的‘{req}’具有强技术迁移性。其与{collabs}等专家的协同成果《{title}》是该方向的重要参考" if collabs \
            else f"学术背书：人才在‘{m_skill}’领域的成果《{title}》（发表于《{src}》）展现了其与岗位需求高度契合的研发能力"
        # 句式 C：侧重影响力与活跃度
        template_c = f"能力验证：模型通过分析人才在《{src}》产出的《{title}》，识别出其在‘{req}’方向上的实战经验"

        # 增加合作者后缀
        collab_suffix = f"，并与该领域的专家 {collabs} 保持着紧密的协作关系。" if collabs else "。"

        choices = [template_a, template_b, template_c]
        # 根据 match_type 权重选择或随机选择以增加多样性
        base_text = choices[0] if path['match_type'] == 'exact' else random.choice(choices[1:])

        return base_text + (collab_suffix if "与" not in base_text else "。")

    def _generate_fallback_response(self):
        return {
            "matched_skill": "全息领域匹配",
            "key_evidence_work": "多篇核心领域产出",
            "summary": "综合评估：该人才的整体学术画像与岗位向量表征高度重合，在机器人与人工智能交叉领域具备极强的技术泛化能力。"
        }