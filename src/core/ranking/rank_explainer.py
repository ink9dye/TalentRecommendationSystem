import torch
import numpy as np
import random
import re
from typing import Any, Dict, List, Optional, Tuple


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
        query_text: Optional[str] = None,
    ):
        """
        深度推理逻辑：多锚点拓扑探测 + 发表平台提取 + 合作伙伴收割。
        若传入 candidate_record / candidate_evidence_rows：
        - 优先使用候选池中已有证据（vector_evidence / label_evidence / collab_evidence 等）生成代表作与解释，保证与召回一致；
        - 仅在证据缺失/不可用时，才触发 Neo4j 多跳检索作为兜底。
        """
        # 0) 候选池证据优先：从 candidate_record / candidate_evidence_rows 中挑选代表作
        pool_rows = self._filter_evidence_rows_for_author(author_id, candidate_evidence_rows or [])
        chosen_work = self._select_representative_work_from_pool(candidate_record, pool_rows)
        if chosen_work is not None:
            wid = chosen_work.get("wid")
            title = chosen_work.get("title") or "候选池代表作"
            # 对单篇代表作计算注意力权重作为 model_confidence（不再为“找证据”而额外跑 Neo4j 多跳）
            att_w = self._safe_attention_weight_for_work(author_id, wid) if wid else 0.0
            # 仍补充精排侧图谱证据：用于说明“为何精排把他排前”（与召回证据互补）
            kg_best = self._try_graph_evidence_best_path(author_id, job_raw_ids)
            out = {
                "matched_skill": chosen_work.get("matched_skill") or "候选池证据一致性",
                "key_evidence_work": title,
                "work_id": wid,
                "work_url": (f"https://openalex.org/{wid}" if wid else None),
                "source": chosen_work.get("source") or "候选池证据",
                "collaborators": chosen_work.get("collaborators"),
                "match_type": chosen_work.get("match_type") or "pool_evidence",
                "model_confidence": round(float(att_w or 0.0), 4),
                "summary": "",
            }
            out["evidence_chain"] = self._build_four_segment_evidence_from_pool(
                candidate_record,
                pool_rows,
                chosen_work,
                att_w,
                query_text=query_text,
                kg_best_path=kg_best,
            )
            out["summary"] = out["evidence_chain"].get("full_summary") or self._build_pool_summary_fallback(
                candidate_record,
                chosen_work,
                att_w,
                query_text=query_text,
                kg_best_path=kg_best,
            )
            return out

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
        # 默认 summary/full_summary 走更“推荐说明”的风格（不暴露内部字段）
        summary = self._build_user_facing_summary(
            record=candidate_record,
            chosen_work={"title": best_path.get("title"), "wid": best_path.get("wid")},
            kg_best_path=best_path,
            query_text=None,
        )

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
            out["evidence_chain"] = self._build_four_segment_evidence(best_path, candidate_record, candidate_evidence_rows or [])
            out["summary"] = out["evidence_chain"].get("full_summary", summary)
        return out

    def _filter_evidence_rows_for_author(
        self, author_id: str, rows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        aid = str(author_id).strip().lower()
        out: List[Dict[str, Any]] = []
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            ra = str(r.get("author_id") or "").strip().lower()
            if not ra or ra != aid:
                continue
            out.append(r)
        return out

    def _select_representative_work_from_pool(
        self, record: Optional[Any], evidence_rows: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        从候选池证据中选一篇代表作。
        目标：解释与召回一致；尽量不触发 Neo4j 查询。
        """
        # 优先：CandidateRecord.vector_evidence.top_papers（结构最稳定、且含 wid/title/score）
        vec_ev = getattr(record, "vector_evidence", None) if record is not None else None
        chosen = self._pick_from_vector_evidence(vec_ev)
        if chosen is not None:
            return chosen

        # 其次：evidence_rows 中 path=vector/label/collab 的 evidence（尽量从中拼出 wid/title）
        for prefer in ("vector", "label", "collab"):
            for row in evidence_rows or []:
                if str(row.get("path") or "").strip().lower() != prefer:
                    continue
                ev = row.get("evidence")
                # vector evidence 与 CandidateRecord 同结构
                if prefer == "vector":
                    chosen = self._pick_from_vector_evidence(ev)
                    if chosen is not None:
                        return chosen
                # label/collab：结构不稳定，只做极保守尝试
                chosen = self._pick_from_generic_evidence(ev, match_type=f"{prefer}_pool_evidence")
                if chosen is not None:
                    return chosen
        return None

    def _pick_from_vector_evidence(self, vector_evidence: Any) -> Optional[Dict[str, Any]]:
        if not vector_evidence or not isinstance(vector_evidence, dict):
            return None
        top_papers = vector_evidence.get("top_papers") or []
        if not isinstance(top_papers, list) or not top_papers:
            return None

        def _paper_score(p: Dict[str, Any]) -> float:
            try:
                return float(p.get("score_clause_aware") or p.get("score_hybrid") or p.get("score_dense") or 0.0)
            except Exception:
                return 0.0

        best = None
        best_s = -1.0
        for p in top_papers:
            if not isinstance(p, dict):
                continue
            wid = p.get("wid")
            title = (p.get("title") or "").strip()
            if not wid or not title:
                continue
            s = _paper_score(p)
            if s > best_s:
                best_s = s
                best = p
        if best is None:
            return None
        return {
            "wid": best.get("wid"),
            "title": best.get("title"),
            "source": "向量路 evidence",
            "collaborators": None,
            "match_type": "vector_pool_evidence",
            "matched_skill": "向量语义命中",
        }

    def _pick_from_generic_evidence(self, evidence: Any, match_type: str) -> Optional[Dict[str, Any]]:
        """
        保守解析：尽量从 evidence 中抽到 {wid,title}。
        仅用于 label/collab 的弱兜底，避免强行猜结构导致误导。
        """
        if not evidence:
            return None
        if isinstance(evidence, dict):
            # 常见：{"work_id":..,"title":..} 或 {"wid":..,"title":..}
            wid = evidence.get("wid") or evidence.get("work_id")
            title = evidence.get("title")
            if wid and title:
                return {
                    "wid": wid,
                    "title": title,
                    "source": "候选池证据",
                    "collaborators": None,
                    "match_type": match_type,
                    "matched_skill": "候选池证据命中",
                }
            # 常见：{"top_papers":[...]}（兼容 vector_evidence 形式）
            if "top_papers" in evidence:
                return self._pick_from_vector_evidence(evidence)
        if isinstance(evidence, list):
            # list[dict] 中尝试找第一条具备 wid/title 的项
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                wid = item.get("wid") or item.get("work_id")
                title = item.get("title")
                if wid and title:
                    return {
                        "wid": wid,
                        "title": title,
                        "source": "候选池证据",
                        "collaborators": None,
                        "match_type": match_type,
                        "matched_skill": "候选池证据命中",
                    }
        return None

    def _safe_attention_weight_for_work(self, author_id: str, work_id: Any) -> float:
        try:
            norm_author_id = str(author_id).strip().lower()
            h_int = self.raw_to_int.get(f"a_{norm_author_id}", 0)
            wid = str(work_id).strip().lower()
            t_int = self.raw_to_int.get(f"w_{wid}", 0)
            if h_int == 0 or t_int == 0:
                return 0.0
            h_list = torch.LongTensor([h_int]).to(self.device)
            t_list = torch.LongTensor([t_int]).to(self.device)
            r_list = torch.LongTensor([self.REL_AUTHORED]).to(self.device)
            with torch.no_grad():
                s = self.model.update_attention_batch(h_list, t_list, r_list).squeeze()
                return float(s.item())
        except Exception:
            return 0.0

    def _build_four_segment_evidence_from_pool(
        self,
        record: Optional[Any],
        evidence_rows: List[Dict[str, Any]],
        chosen_work: Dict[str, Any],
        att_weight: float,
        query_text: Optional[str] = None,
        kg_best_path: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        四段式证据链（对齐候选池）：召回来源摘要、主题匹配摘要、学术实力摘要、模型置信摘要。
        与 _build_four_segment_evidence 的区别：主题匹配段不强依赖 Neo4j 的 req_skill/match_skill。
        """
        # 1) 用户可读 summary/full_summary（默认前端展示用）
        display_summary = self._build_user_facing_summary(
            record=record,
            chosen_work=chosen_work,
            kg_best_path=kg_best_path,
            query_text=query_text,
        )

        # 2) 分段（仍保留原 key，供前端兼容；分段也必须是用户可读文本）
        seg_recall = self._build_user_facing_segment_recall(record)
        seg_topic = self._build_user_facing_segment_paper_and_path(record, chosen_work, kg_best_path)
        seg_profile = self._build_user_facing_segment_profile(record)
        seg_model = self._build_user_facing_segment_model(record)

        # 3) 调试摘要（不进入默认 display_summary/full_summary）
        debug_summary = self._build_debug_summary(
            record=record,
            chosen_work=chosen_work,
            kg_best_path=kg_best_path,
            query_text=query_text,
            att_weight=att_weight,
        )

        # 4) 可选 bullets（前端若要做卡片式展示，可直接用）
        evidence_bullets = self._build_user_facing_evidence_bullets(
            record=record, chosen_work=chosen_work, kg_best_path=kg_best_path
        )

        return {
            "recall_source": seg_recall,
            "topic_match": seg_topic,
            "academic_strength": seg_profile,
            "model_confidence": seg_model,
            "full_summary": display_summary,
            # 新增字段（不破坏旧前端）
            "display_summary": display_summary,
            "debug_summary": debug_summary,
            "evidence_bullets": evidence_bullets,
        }

    def _build_pool_summary_fallback(
        self,
        record: Optional[Any],
        chosen_work: Dict[str, Any],
        att_weight: float,
        query_text: Optional[str] = None,
        kg_best_path: Optional[Dict[str, Any]] = None,
    ) -> str:
        # fallback 也必须走“用户可读”逻辑，避免出现 KGAT 注意力、内部字段等调试信息
        return self._build_user_facing_summary(
            record=record,
            chosen_work=chosen_work,
            kg_best_path=kg_best_path,
            query_text=query_text,
        )

    # ---------------------------------------------------------------------
    # User-facing summary helpers (no internal/debug fields)
    # ---------------------------------------------------------------------

    def _build_user_facing_summary(
        self,
        record: Optional[Any],
        chosen_work: Dict[str, Any],
        kg_best_path: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
    ) -> str:
        """
        生成默认展示的自然语言推荐理由（面向用户，不暴露内部调试字段/数值）。
        结构：
        - 推荐原因（为何值得看）
        - 代表论文证据（代表作与方向关系）
        - 作者画像（H-index、相关论文、近5年论文）
        - 可选谨慎句（论文少/近年少/相关度一般）
        """
        from_v = bool(getattr(record, "from_vector", False)) if record else False
        from_l = bool(getattr(record, "from_label", False)) if record else False
        from_c = bool(getattr(record, "from_collab", False)) if record else False

        title = str((chosen_work or {}).get("title") or "").strip() or "代表作"

        # 概念词：优先 label_evidence core，其次 matched_skill，再次图谱 match_skill
        core_terms = self._extract_core_terms(record, max_n=3)
        if not core_terms:
            ms = str((chosen_work or {}).get("matched_skill") or "").strip()
            if ms and ms not in ("候选池证据一致性", "向量语义命中", "候选池证据命中", "候选池证据一致"):
                core_terms = [ms]
        if not core_terms:
            try:
                msk = str((kg_best_path or {}).get("match_skill") or "").strip()
                if msk:
                    core_terms = [msk]
            except Exception:
                core_terms = []
        concept_phrase = "、".join(core_terms[:3]) if core_terms else "岗位技术方向"

        # 相关度档位（仅用于措辞，不展示“代表论文相关度：xx”字段）
        band = self._vector_best_paper_score_band(record)

        # 推荐原因句
        if from_v and from_l:
            s1 = (
                "该作者同时被语义检索和概念标签路径命中，说明其论文内容与岗位描述在文本语义和技术概念上都有交集。"
            )
        elif from_v:
            s1 = "该作者主要由语义检索召回，代表论文与岗位文本存在一定语义相似性。"
        elif from_l:
            s1 = f"该作者主要由概念标签路径召回，系统在其论文证据中发现与「{concept_phrase}」相关的技术线索。"
        elif from_c:
            s1 = "该作者主要由协作网络召回，说明其与当前方向的相关作者群体存在合作关联。"
        else:
            s1 = "系统在候选池中发现该作者与岗位需求存在一定关联，建议进一步查看其代表作与研究方向。"

        # 代表论文证据句（不放 OpenAlex id）
        if band in ("medium", "low"):
            rel_phrase = "存在一定相关性"
        else:
            rel_phrase = "相关"
        s2 = f"其代表作《{title}》与「{concept_phrase}」方向{rel_phrase}，可作为该作者与岗位技术需求关联的主要证据。"

        # 作者画像句
        h = getattr(record, "h_index", None) if record else None
        works = getattr(record, "works_count", None) if record else None
        recent = getattr(record, "recent_works_count", None) if record else None
        h_s = "-" if h in (None, "") else str(h)
        w_s = "-" if works in (None, "") else str(works)
        r_s = "-" if recent in (None, "") else str(recent)
        s3 = f"作者画像显示：H-index 为 {h_s}，相关论文 {w_s} 篇，近 5 年相关论文 {r_s} 篇。"

        # 谨慎句（可选）
        caution = self._build_caution_sentence(
            works=works, recent=recent, band=band, from_vector=from_v, from_label=from_l
        )

        # 若 band 较低/中等，用更保守的首句收尾（但不写“中等/较低”）
        if caution:
            return " ".join([s1, s2, s3, caution])
        return " ".join([s1, s2, s3])

    def _build_user_facing_segment_recall(self, record: Optional[Any]) -> str:
        from_v = bool(getattr(record, "from_vector", False)) if record else False
        from_l = bool(getattr(record, "from_label", False)) if record else False
        from_c = bool(getattr(record, "from_collab", False)) if record else False
        if from_v and from_l:
            return "该作者同时被语义检索和概念标签路径命中，证据来源相对稳定。"
        if from_v:
            return "该作者由语义检索召回，说明其代表论文与岗位文本存在一定语义相似性。"
        if from_l:
            return "该作者由概念标签路径召回，说明其研究概念与岗位技术线索存在交集。"
        if from_c:
            return "该作者由协作网络召回，说明其与相关作者群体存在合作关联。"
        return "该作者在候选池中被筛选出来，建议进一步查看其代表作与研究方向。"

    def _build_user_facing_segment_paper_and_path(
        self,
        record: Optional[Any],
        chosen_work: Dict[str, Any],
        kg_best_path: Optional[Dict[str, Any]],
    ) -> str:
        title = str((chosen_work or {}).get("title") or "").strip() or "代表作"
        core_terms = self._extract_core_terms(record, max_n=3)
        if not core_terms:
            try:
                msk = str((kg_best_path or {}).get("match_skill") or "").strip()
                if msk:
                    core_terms = [msk]
            except Exception:
                core_terms = []
        concept_phrase = "、".join(core_terms[:3]) if core_terms else "岗位技术方向"
        return f"代表作《{title}》与「{concept_phrase}」方向相关，可作为该作者与岗位技术需求关联的主要证据。"

    def _build_user_facing_segment_profile(self, record: Optional[Any]) -> str:
        h = getattr(record, "h_index", None) if record else None
        works = getattr(record, "works_count", None) if record else None
        recent = getattr(record, "recent_works_count", None) if record else None
        h_s = "-" if h in (None, "") else str(h)
        w_s = "-" if works in (None, "") else str(works)
        r_s = "-" if recent in (None, "") else str(recent)
        return f"作者画像显示：H-index 为 {h_s}，相关论文 {w_s} 篇，近 5 年相关论文 {r_s} 篇。"

    def _build_user_facing_segment_model(self, record: Optional[Any]) -> str:
        # 不输出 KGAT 注意力等内部数值；用更“结果导向”的表述
        if record is not None and (getattr(record, "from_vector", False) or getattr(record, "from_label", False) or getattr(record, "from_collab", False)):
            return "精排阶段将该候选保留在当前排序结果中，说明图结构与候选池证据没有明显冲突。"
        return "精排阶段未发现与候选池证据明显冲突的信号。"

    def _build_user_facing_evidence_bullets(
        self,
        record: Optional[Any],
        chosen_work: Dict[str, Any],
        kg_best_path: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        out: List[str] = []
        try:
            title = str((chosen_work or {}).get("title") or "").strip()
            if title:
                out.append(f"代表作：《{title}》")
        except Exception:
            pass
        try:
            core = self._extract_core_terms(record, max_n=3)
            if core:
                out.append("命中核心概念：" + "、".join(core))
        except Exception:
            pass
        try:
            if isinstance(kg_best_path, dict):
                req = str(kg_best_path.get("req_skill") or "").strip()
                msk = str(kg_best_path.get("match_skill") or "").strip()
                if req and msk:
                    out.append(f"技术线索：{req} → {msk}")
        except Exception:
            pass
        return out

    def _extract_core_terms(self, record: Optional[Any], max_n: int = 3) -> List[str]:
        if record is None:
            return []
        lab = getattr(record, "label_evidence", None)
        terms = []
        if isinstance(lab, dict):
            terms = lab.get("terms") or lab.get("term_list") or []
        elif isinstance(lab, list):
            terms = lab
        out: List[str] = []
        if isinstance(terms, list):
            for t in terms:
                if not isinstance(t, dict):
                    continue
                if (t.get("bucket") or "").strip().lower() != "core":
                    continue
                nm = str(t.get("term") or "").strip()
                if not nm:
                    continue
                if nm not in out:
                    out.append(nm)
                if len(out) >= int(max_n):
                    break
        return out

    def _vector_best_paper_score_band(self, record: Optional[Any]) -> str:
        """
        将 best_paper_score 映射为档位，仅用于措辞（不直接暴露给用户）。
        return: "high" | "medium" | "low" | ""
        """
        try:
            vec = getattr(record, "vector_evidence", None) if record is not None else None
            if not isinstance(vec, dict):
                return ""
            summ = vec.get("summary") or {}
            best = summ.get("best_paper_score")
            if best is None:
                return ""
            b = float(best)
            if b >= 0.72:
                return "high"
            if b >= 0.55:
                return "medium"
            return "low"
        except Exception:
            return ""

    def _build_caution_sentence(
        self,
        *,
        works: Any,
        recent: Any,
        band: str,
        from_vector: bool,
        from_label: bool,
    ) -> str:
        parts: List[str] = []
        try:
            if recent is not None and int(recent) == 0:
                parts.append("不过，近 5 年相关论文数量较少，需进一步确认近期活跃度。")
        except Exception:
            pass
        try:
            if works is not None and float(works) <= 1:
                parts.append("其相关论文数量有限，建议结合论文内容进一步人工核验。")
        except Exception:
            pass
        if band in ("medium", "low"):
            # 避免“高度匹配”措辞
            parts.append("整体上可作为候选补充关注，建议结合岗位细节进一步核验。")
        if (from_vector and from_label) and not parts:
            # 稳定性描述（不过度夸大）
            parts.append("由于同时命中语义与概念证据，证据来源相对稳定。")
        # 只取一句，避免冗长
        return parts[0] if parts else ""

    # ---------------------------------------------------------------------
    # Debug helpers (can include internal fields)
    # ---------------------------------------------------------------------

    def _build_debug_summary(
        self,
        *,
        record: Optional[Any],
        chosen_work: Dict[str, Any],
        kg_best_path: Optional[Dict[str, Any]],
        query_text: Optional[str],
        att_weight: float,
    ) -> str:
        """
        内部排障用摘要：允许包含召回来源、dominant_recall_path、query_type_coverage、
        best_paper_score 档位、核心概念、KGAT 注意力、图谱路径等。
        注意：不得并入默认 display_summary/full_summary。
        """
        parts: List[str] = []
        try:
            if record is not None:
                src = []
                if getattr(record, "from_vector", False):
                    src.append("vector")
                if getattr(record, "from_label", False):
                    src.append("label")
                if getattr(record, "from_collab", False):
                    src.append("collab")
                parts.append("sources=" + "+".join(src) if src else "sources=none")
                dom = getattr(record, "dominant_recall_path", None)
                if dom:
                    parts.append(f"dominant={dom}")
        except Exception:
            pass
        try:
            vec = getattr(record, "vector_evidence", None) if record is not None else None
            if isinstance(vec, dict):
                summ = vec.get("summary") or {}
                qcov = summ.get("query_type_coverage") or []
                if qcov:
                    parts.append("query_type_coverage=" + ",".join([str(x) for x in qcov[:6]]))
                best = summ.get("best_paper_score")
                if best is not None:
                    parts.append(f"best_paper_score={float(best):.4f}")
                    band = self._vector_best_paper_score_band(record)
                    if band:
                        parts.append(f"best_paper_band={band}")
        except Exception:
            pass
        try:
            core = self._extract_core_terms(record, max_n=3)
            if core:
                parts.append("core_terms=" + ",".join(core))
        except Exception:
            pass
        try:
            title = str((chosen_work or {}).get("title") or "").strip()
            wid = (chosen_work or {}).get("wid")
            if title:
                parts.append(f"rep_title={title[:80]}")
            if wid:
                parts.append(f"rep_wid={wid}")
        except Exception:
            pass
        try:
            if isinstance(kg_best_path, dict):
                req = kg_best_path.get("req_skill")
                msk = kg_best_path.get("match_skill")
                mt = kg_best_path.get("match_type")
                if req and msk:
                    parts.append(f"graph_path={req}->{msk}({mt})")
        except Exception:
            pass
        try:
            parts.append(f"kgat_attention={float(att_weight or 0.0):.4f}")
        except Exception:
            pass
        return " | ".join(parts) if parts else ""

    def _try_graph_evidence_best_path(
        self, author_id: str, job_raw_ids: list
    ) -> Optional[Dict[str, Any]]:
        """
        在“候选池证据可用”的前提下，补一条精排侧图谱路径证据（轻量）。
        目的：回答“精排为什么认为他更匹配”，与召回证据互补。
        """
        if not getattr(self, "graph", None) or not job_raw_ids:
            return None
        try:
            q = """
            MATCH (j:Job) WHERE j.id IN $jids OR j.securityId IN $jids
            MATCH (j)-[:REQUIRE_SKILL]->(v1:Vocabulary)
            WHERE NOT v1.term IN ['computer science', 'mathematics', 'engineering', 'physics', 'technology', '领域专家']
            OPTIONAL MATCH (v1)-[r:SIMILAR_TO]-(v2:Vocabulary)
            WHERE r.score > 0.7
            WITH v1, COALESCE(v2, v1) as target_v
            MATCH (target_v)<-[:HAS_TOPIC]-(w:Work)<-[:AUTHORED]-(a:Author {id: $aid})
            RETURN v1.term as req_skill,
                   target_v.term as match_skill,
                   w.title as title,
                   w.id as wid,
                   (CASE WHEN v1 = target_v THEN 'exact' ELSE 'semantic' END) as match_type
            ORDER BY match_type ASC
            LIMIT 5
            """
            rows = self.graph.run(q, jids=job_raw_ids, aid=str(author_id)).data()
            if not rows:
                return None
            # 选第一条即可（exact 优先，其次 semantic）
            r0 = rows[0] if isinstance(rows, list) else None
            return r0 if isinstance(r0, dict) else None
        except Exception:
            return None

    def _extract_query_keywords(self, text: str, top_k: int = 6) -> List[str]:
        """
        轻量关键词抽取：不依赖外部模型，尽量从 query_text 中抽到可展示的词。
        - 英文：按词切分，去停用词，保留较长 token
        - 中文：按连续汉字片段切分，保留长度>=2
        """
        if not text:
            return []
        t = text.strip()
        if not t:
            return []
        # 统一小写仅用于英文匹配；中文保持原样即可
        low = t.lower()
        stop = {
            "the", "and", "or", "to", "of", "in", "for", "with", "on", "at", "by", "from",
            "a", "an", "as", "is", "are", "be", "this", "that", "it", "we", "you", "our",
            "岗位", "需求", "负责", "要求", "需要", "优先", "相关", "经验", "能力", "方向", "项目",
        }
        tokens: List[str] = []
        # 英文/数字/连字符 token
        for w in re.findall(r"[a-zA-Z][a-zA-Z0-9\-_/]{1,}", low):
            w2 = w.strip("-_/")
            if len(w2) < 4:
                continue
            if w2 in stop:
                continue
            tokens.append(w2)
        # 中文片段
        for s in re.findall(r"[\u4e00-\u9fff]{2,}", t):
            if s in stop:
                continue
            tokens.append(s)
        if not tokens:
            return []
        freq: Dict[str, int] = {}
        for w in tokens:
            freq[w] = freq.get(w, 0) + 1
        ranked = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
        return [w for w, _ in ranked[: max(1, int(top_k))]]

    def _match_keywords_in_title(self, keywords: List[str], title: str, top_k: int = 4) -> List[str]:
        if not keywords or not title:
            return []
        tlow = title.lower()
        hits: List[str] = []
        for kw in keywords:
            if not kw:
                continue
            if re.search(r"[\u4e00-\u9fff]", kw):
                if kw in title:
                    hits.append(kw)
            else:
                if kw.lower() in tlow:
                    hits.append(kw)
            if len(hits) >= top_k:
                break
        return hits
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

        # 默认展示不暴露 KGAT 注意力等内部数值
        seg4 = "精排阶段将该候选保留在当前排序结果中，说明图结构证据与召回证据没有明显冲突。"

        # full_summary 也保持用户可读，不拼接内部字段
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