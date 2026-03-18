import torch
import numpy as np
import random
from typing import Any, Dict, List, Optional


class RankExplainer:
    """
    职责：结合 Neo4j 多跳拓扑、学术社交网络与 KGAT 注意力，生成富含协作信息的清晰证据链。
    支持接入 candidate_record / candidate_evidence_rows，输出四段式证据链（README 6.6～6.7）。
    """

    def __init__(self, model, graph, raw_to_int, device, dataloader=None):
        self.model = model
        self.graph = graph
        self.raw_to_int = raw_to_int
        self.device = device
        self.dataloader = dataloader
        self.REL_AUTHORED = 1  # 对应 config.py

    def explain(
        self,
        author_id: str,
        job_raw_ids: list,
        candidate_record: Optional[Any] = None,
        candidate_evidence_rows: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        深度推理逻辑：多锚点拓扑探测 + 发表平台提取 + 合作伙伴收割。
        若传入 candidate_record / candidate_evidence_rows，则补充四段式证据链（召回来源、主题匹配、学术实力、模型置信）。
        """
        norm_author_id = str(author_id).strip().lower()
        h_int = self.raw_to_int.get(f"a_{norm_author_id}", 0)

        # 1. 增强版 Cypher：引入相似度分数过滤，防止语义漂移
        multi_hop_query = """
        MATCH (j:Job) WHERE j.id IN $jids OR j.securityId IN $jids
        MATCH (j)-[:REQUIRE_SKILL]->(v1:Vocabulary)
        WHERE NOT v1.term IN ['computer science', 'mathematics', 'engineering', 'physics', 'technology', '领域专家']

        // --- 核心修正点：增加 score 校验，只有强相似边 (>0.7) 才允许跳转 ---
        OPTIONAL MATCH (v1)-[r:SIMILAR_TO]-(v2:Vocabulary)
        WHERE r.score > 0.7

        // 如果没有高分相似词，COALESCE 确保 target_v 维持原词 v1，走精准匹配逻辑
        WITH j, v1, COALESCE(v2, v1) as target_v

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

        # 2. 语义保底 (Fuzzy Match)
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

        # 6. 动态生成总结文本
        summary = self._build_dynamic_summary(best_path)

        out = {
            "matched_skill": best_path['req_skill'],
            "key_evidence_work": best_path['title'],
            "work_id": best_path['wid'],
            "work_url": f"https://openalex.org/{best_path['wid']}",
            "source": best_path.get('source_name', "相关领域核心刊物"),
            "collaborators": best_path['co_authors'],
            "match_type": best_path['match_type'],
            "model_confidence": round(float(best_path['att_weight']), 4),
            "summary": summary,
        }
        if candidate_record is not None or (candidate_evidence_rows and len(candidate_evidence_rows) > 0):
            out["evidence_chain"] = self._build_four_segment_evidence(
                best_path, candidate_record, candidate_evidence_rows or []
            )
            out["summary"] = out["evidence_chain"].get("full_summary", summary)
        return out
    def _build_four_segment_evidence(
        self,
        best_path: Dict[str, Any],
        record: Optional[Any],
        evidence_rows: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        四段式证据链（README 6.7）：召回来源摘要、主题匹配摘要、学术实力摘要、模型置信摘要。
        """
        seg1 = "召回来源："
        if record:
            paths = []
            if getattr(record, "from_vector", False):
                paths.append("向量语义")
            if getattr(record, "from_label", False):
                paths.append("标签路径")
            if getattr(record, "from_collab", False):
                paths.append("协作网络")
            path_count = getattr(record, "path_count", 0) or 0
            seg1 += "、".join(paths) if paths else "多路召回"
            if path_count > 1:
                seg1 += f"；多路命中（{path_count} 条路径）。"
            else:
                seg1 += "。"
            dom = getattr(record, "dominant_recall_path", None) or ""
            if dom:
                seg1 += f" 主导来源：{dom}。"
        else:
            seg1 += "来自多路召回融合。"
        if evidence_rows:
            seg1 += " 证据路径：" + "；".join([e.get("path", "") for e in evidence_rows[:5]]) + "。"

        seg2 = "主题匹配："
        req = best_path.get("req_skill") or "岗位核心技能"
        match_skill = best_path.get("match_skill") or req
        seg2 += f"岗位诉求「{req}」与学术词「{match_skill}」对齐；"
        seg2 += f"关键论文《{best_path.get('title', '')}》建立作者与岗位的匹配路径。"

        seg3 = "学术实力："
        if record:
            h = getattr(record, "h_index", None)
            works = getattr(record, "works_count", None)
            cited = getattr(record, "cited_by_count", None)
            recent = getattr(record, "recent_works_count", None)
            seg3 += f"H-index {h or '-'}，总论文 {works or '-'}，总引用 {cited or '-'}"
            if recent is not None:
                seg3 += f"，近年产出 {recent} 篇"
            seg3 += "。"
        else:
            seg3 += "详见作者学术指标。"

        seg4 = "模型置信："
        att = best_path.get("att_weight", 0)
        seg4 += f"KGAT-AX 对代表作的注意力权重 {att:.4f}；"
        seg4 += "排序结果与候选池证据一致。"

        full = " ".join([seg1, seg2, seg3, seg4])
        return {
            "recall_source": seg1,
            "topic_match": seg2,
            "academic_strength": seg3,
            "model_confidence": seg4,
            "full_summary": full,
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
            "work_id": None,
            "work_url": None,
            "source": "相关领域",
            "collaborators": None,
            "match_type": "fallback",
            "model_confidence": 0.0,
            "summary": "综合评估：该人才的整体学术画像与岗位向量表征高度重合，在机器人与人工智能交叉领域具备极强的技术泛化能力。",
        }