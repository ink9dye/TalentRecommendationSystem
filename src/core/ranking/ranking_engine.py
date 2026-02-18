import torch
import sqlite3
import json
import os
import numpy as np
from py2neo import Graph
from typing import List, Dict
from config import DB_PATH, KGATAX_TRAIN_DATA_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE


class RankingEngine:
    def __init__(self, model, dataloader):
        """
        KGAT-AX 强化版精排引擎：融合模型注意力机制与图谱路径回溯
        """
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device("cpu")

        # 1. 建立 Neo4j 连接
        self.graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), name=NEO4J_DATABASE)

        # 2. 载入 ID 映射
        self.id_map_path = os.path.join(KGATAX_TRAIN_DATA_DIR, "id_map.json")
        self._load_id_mapping()

        # 3. 关系类型定义（需与 generate_training_data.py 一致）
        self.REL_AUTHORED = 1
        self.REL_HAS_TOPIC = 4

    def _load_id_mapping(self):
        with open(self.id_map_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        self.raw_to_int = mapping['entity']
        self.int_to_raw = {int(v): k for k, v in mapping['entity'].items()}
        self.raw_user_to_int = {k: v for k, v in mapping['entity'].items() if
                                not k.startswith(('a_', 'w_', 'v_', 'i_', 's_'))}

    def execute_rank(self, job_raw_id: str, candidate_raw_ids: List[str]) -> List[Dict]:
        """
        精排核心流程
        """
        self.model.eval()

        user_int_id = self.raw_user_to_int.get(job_raw_id, 0)
        item_int_ids = [self.raw_to_int.get(f"a_{aid}", 0) for aid in candidate_raw_ids]

        with torch.no_grad():
            u_tensor = torch.LongTensor([user_int_id]).to(self.device)
            i_tensor = torch.LongTensor(item_int_ids).to(self.device)
            aux_info = self.dataloader.aux_info_all.to(self.device)

            # 执行模型打分
            scores = self.model.calc_score(u_tensor, i_tensor, aux_info)
            scores = scores.squeeze(0)

        top_k = min(100, len(item_int_ids))
        top_val, top_idx = torch.topk(scores, top_k)

        ranked_results = []
        for i in range(top_k):
            idx = top_idx[i].item()
            auth_raw_id = candidate_raw_ids[idx]

            stats = self._fetch_sqlite_stats(auth_raw_id)
            # 调用模型驱动的证据回溯函数
            evidence = self._get_kgat_driven_evidence(f"a_{auth_raw_id}", job_raw_id)

            ranked_results.append({
                "rank": i + 1,
                "author_id": auth_raw_id,
                "name": stats['name'],
                "score": round(float(top_val[i]), 4),
                "metrics": stats,
                "evidence_chain": evidence
            })

        return ranked_results

    def _get_kgat_driven_evidence(self, auth_prefixed_id: str, job_raw_id: str) -> dict:
        """
        核心升级：先查询 Neo4j 候选路径，再通过模型 Attention 进行权重筛选
        """
        author_id = auth_prefixed_id.replace("a_", "")

        # 1. 初始 Neo4j 查询：获取所有可能的“岗位-技能-作品-人才”链路
        path_query = """
        MATCH (j:Job {id: $jid})-[:REQUIRE_SKILL]->(v:Vocabulary)<-[:HAS_TOPIC]-(w:Work)<-[:AUTHORED]-(a:Author {id: $aid})
        RETURN v.term as skill, v.id as vid, w.title as title, w.id as wid
        """
        paths = self.graph.run(path_query, jid=job_raw_id, aid=author_id).data()

        evidence = {
            "matched_skills": [],
            "representative_works": [],
            "notable_collaborators": [],
            "summary": "多维学术特征深度匹配。"
        }

        if not paths:
            return evidence

        # 2. 转换 ID 并计算模型注意力
        # 我们重点检查 (Author)-[AUTHORED]->(Work) 这一跳的权重
        h_list, t_list, r_list = [], [], []
        auth_int = self.raw_to_int.get(auth_prefixed_id, 0)

        for p in paths:
            work_int = self.raw_to_int.get(f"w_{p['wid']}", 0)
            h_list.append(auth_int)
            t_list.append(work_int)
            r_list.append(self.REL_AUTHORED)

        with torch.no_grad():
            # 调用模型内部的注意力计算
            # 这反映了模型对该人才名下各作品的“重视程度”
            att_scores = self.model.update_attention_batch(
                torch.LongTensor(h_list).to(self.device),
                torch.LongTensor(t_list).to(self.device),
                torch.LongTensor(r_list).to(self.device)
            ).squeeze().cpu().numpy()

        # 3. 结果融合：将注意力分值挂载回路径
        # 若只有一条路径，att_scores 可能是标量，需处理
        if np.isscalar(att_scores): att_scores = [att_scores]

        for i, p in enumerate(paths):
            p['att'] = float(att_scores[i])

        # 4. 根据注意力分值进行精选排序
        sorted_paths = sorted(paths, key=lambda x: x['att'], reverse=True)

        # 仅取模型关注度最高的前 2 个技能和前 3 篇论文
        top_skills = list(dict.fromkeys([p['skill'] for p in sorted_paths]))[:2]
        top_works = [p['title'] for p in sorted_paths][:3]

        evidence["matched_skills"] = top_skills
        evidence["representative_works"] = top_works

        # 5. 回溯高阶合作者（保持 Neo4j 辅助展示）
        collab_query = """
        MATCH (a:Author {id: $aid})-[:AUTHORED]->(w:Work)<-[:AUTHORED]-(peer:Author)
        WHERE peer.h_index > 25 AND a.id <> peer.id
        RETURN DISTINCT peer.name as peer_name, peer.h_index as ph ORDER BY ph DESC LIMIT 2
        """
        res_c = self.graph.run(collab_query, aid=author_id).data()
        evidence["notable_collaborators"] = [f"{r['peer_name']}(H-{r['ph']})" for r in res_c]

        # 6. 生成基于权重的总结
        if top_skills:
            evidence[
                "summary"] = f"模型识别到其在“{top_skills[0]}”领域的学术贡献（Attention权重最大），共关联 {len(top_works)} 篇高权重代表作。"

        return evidence

    def _fetch_sqlite_stats(self, raw_id: str) -> Dict:
        """从 SQLite 获取学术指标"""
        conn = sqlite3.connect(DB_PATH)
        # 从数据库中查询作者的原始学术数据
        row = conn.execute("SELECT name, h_index, cited_by_count, works_count FROM authors WHERE author_id=?",
                           (raw_id,)).fetchone()
        conn.close()

        # --- 核心修复：将 'cited' 修改为 'citations' 以对齐 talent_app.py 的预期 ---
        if row:
            return {
                "name": row[0],
                "h_index": row[1],
                "citations": row[2],  # 修改此处键名
                "works": row[3]
            }

        return {
            "name": "Unknown",
            "h_index": 0,
            "citations": 0,
            "works": 0
        }