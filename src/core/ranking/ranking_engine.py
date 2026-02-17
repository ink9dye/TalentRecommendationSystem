import torch
import sqlite3
import json
import os
from py2neo import Graph
from typing import List, Dict
from config import DB_PATH, KGATAX_TRAIN_DATA_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class RankingEngine:
    def __init__(self, model, dataloader):
        """
        KGAT-AX 精排引擎：实现特征融合打分与图谱证据回溯
        :param model: 训练好的 KGAT 模型，需包含 aux_embed_layer
        :param dataloader: 数据加载器，提供 aux_info_all 特征矩阵
        """
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device("cpu")  # 推理建议使用 CPU 环境以匹配 32GB 内存

        # 建立 Neo4j 连接：用于实时证据链回溯 [cite: 2]
        self.graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        # 载入 ID 映射表：实现模型 Int ID 与 业务 Raw ID 的双向转换
        self.id_map_path = os.path.join(KGATAX_TRAIN_DATA_DIR, "id_map.json")
        self._load_id_mapping()

    def _load_id_mapping(self):
        with open(self.id_map_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            # {整数ID: 原始字符串ID}
            self.int_to_raw = {int(v): k for k, v in mapping['entity'].items()}

    def execute_rank(self, user_id: int, candidate_ids: List[int]) -> List[Dict]:
        """
        执行精排：打分(AX融合) -> 筛选(Top 100) -> 溯源(Neo4j)
        """
        self.model.eval()

        # 1. 执行 KGAT-AX 模型打分
        # 模型内部通过 holographic_fusion 实现 e_enhanced = e_topo * e_attr
        with torch.no_grad():
            u_tensor = torch.LongTensor([user_id]).to(self.device)
            i_tensor = torch.LongTensor(candidate_ids).to(self.device)
            # 计算融合了学术特征(AX)的预测分值 [cite: 1, 145]
            scores = self.model.calc_score(u_tensor, i_tensor, self.dataloader.aux_info_all.to(self.device))
            scores = scores.squeeze(0)

        # 2. 选取 Top 100 人才
        top_k = min(100, len(candidate_ids))
        top_val, top_idx = torch.topk(scores, top_k)

        results = []
        for i in range(top_k):
            idx = top_idx[i].item()
            auth_int_id = candidate_ids[idx]
            raw_id_prefixed = self.int_to_raw.get(auth_int_id, "")

            # 剥离前缀以便查询数据库 (例如 "a_A50637..." -> "A50637...")
            raw_id = raw_id_prefixed.replace("a_", "") if raw_id_prefixed.startswith("a_") else raw_id_prefixed

            # 3. 挂载 SQLite 原始学术指标 (用于前端点击详情展示)
            stats = self._fetch_sqlite_stats(raw_id)

            # 4. 生成增强版证据链：利用 Neo4j 回溯协作与技能路径
            evidence = self._get_graph_evidence(raw_id_prefixed, user_id)

            results.append({
                "author_id": raw_id,
                "score": round(float(top_val[i]), 4),
                "metrics": stats,
                "evidence": evidence
            })

        return results

    def _get_graph_evidence(self, auth_raw_id: str, job_int_id: int) -> str:
        """
        利用图谱拓扑结构回溯推荐理由，体现协作路与标签路的双重逻辑
        """
        job_raw_id = self.int_to_raw.get(job_int_id, "")
        evidence_list = []

        # 策略 A：技能路径回溯 (标签路逻辑)
        # 匹配 (Job)-[:REQUIRE_SKILL]->(Vocab)<-[:HAS_TOPIC]-(Work)<-[:AUTHORED]-(Author)
        semantic_cypher = """
        MATCH (j:Job {id: $jid})-[:REQUIRE_SKILL]->(v:Vocabulary)<-[:HAS_TOPIC]-(w:Work)<-[:AUTHORED]-(a:Author {id: $aid})
        RETURN v.term as skill, count(w) as work_count
        ORDER BY work_count DESC LIMIT 1
        """
        sem_res = self.graph.run(semantic_cypher, jid=job_raw_id, aid=auth_raw_id).data()
        if sem_res:
            evidence_list.append(
                f"【专业对齐】深耕“{sem_res[0]['skill']}”方向，有 {sem_res[0]['work_count']} 篇相关代表作。")

        # 策略 B：协作影响力回溯 (协同路逻辑)
        # 挖掘该学者与高影响力(H-index > 20)专家的合作记录
        collab_cypher = """
        MATCH (a:Author {id: $aid})-[:AUTHORED]->(w:Work)<-[:AUTHORED]-(peer:Author)
        WHERE peer.h_index > 20 AND a.id <> peer.id
        RETURN peer.name as peer_name, w.title as work_title
        ORDER BY w.citations DESC LIMIT 1
        """
        collab_res = self.graph.run(collab_cypher, aid=auth_raw_id).data()
        if collab_res:
            evidence_list.append(
                f"【学术圈层】曾与专家 {collab_res[0]['peer_name']} 合作发表高引论文《{collab_res[0]['work_title']}》。")

        return " | ".join(evidence_list) if evidence_list else "基于学术拓扑与多维特征的语义匹配推荐。"

    def _fetch_sqlite_stats(self, raw_id: str) -> Dict:
        """查询原始学术属性，补全人才画像 """
        stats = {"name": "Unknown", "h_index": 0, "cited": 0, "works": 0}
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            query = "SELECT name, h_index, cited_by_count, works_count FROM authors WHERE author_id = ?"
            row = cursor.execute(query, (raw_id,)).fetchone()
            conn.close()
            if row:
                stats = {"name": row[0], "h_index": row[1], "cited": row[2], "works": row[3]}
        except:
            pass
        return stats