import os
import re
import json
import sqlite3
import faiss
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from py2neo import Graph
from config import (
    DB_PATH, KGATAX_TRAIN_DATA_DIR, FEATURE_INDEX_PATH,
    JOB_INDEX_PATH, JOB_MAP_PATH, NEO4J_URI, NEO4J_USER,
    NEO4J_PASSWORD, NEO4J_DATABASE
)
from src.core.recall.total_recall import TotalRecallSystem


class KGATAXTrainingGenerator:
    def __init__(self):
        self.output_dir = KGATAX_TRAIN_DATA_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 1. 初始化 Neo4j 连接
        self.graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), name=NEO4J_DATABASE)

        # 2. 载入岗位索引
        print(f"[*] 正在载入预计算岗位索引: {JOB_INDEX_PATH}")
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_id_to_idx = {sid: i for i, sid in enumerate(json.load(f))}

        # 3. 召回系统
        self.recall_system = TotalRecallSystem()
        self.recall_system.v_path.verbose = False
        self.recall_system.l_path.verbose = False

        # 4. 载入学术质量映射
        self.author_quality_map = {}
        self._preload_author_quality_from_index()

        # 5. ID 映射管理
        self.user_to_int = {}
        self.entity_to_int = {}
        self.user_counter = 0
        self.entity_counter = 0
        self.ENTITY_OFFSET = 0

    def _preload_author_quality_from_index(self):
        if not os.path.exists(FEATURE_INDEX_PATH):
            return
        with open(FEATURE_INDEX_PATH, 'r', encoding='utf-8') as f:
            feature_bundle = json.load(f)
        author_features = feature_bundle.get('author', {})
        for aid, feats in author_features.items():
            h_norm = feats.get('h_index', 0.0)
            c_norm = feats.get('cited_by_count', 0.0)
            w_norm = feats.get('works_count', 0.0)
            self.author_quality_map[str(aid)] = 0.4 * h_norm + 0.4 * c_norm + 0.2 * w_norm

    def get_user_id(self, raw_id):
        raw_id = str(raw_id)
        if raw_id not in self.user_to_int:
            self.user_to_int[raw_id] = self.user_counter
            self.user_counter += 1
        return self.user_to_int[raw_id]

    def get_ent_id(self, raw_id):
        raw_id = str(raw_id).strip().lower()
        if raw_id not in self.entity_to_int:
            self.entity_to_int[raw_id] = self.entity_counter
            self.entity_counter += 1
        return self.entity_to_int[raw_id] + self.ENTITY_OFFSET

    def generate_refined_train_data(self, train_size=3000, test_size=300):
        """
        任务 1：生成精排训练样本
        返回：被抽样用于训练的岗位 ID 列表 (锚点)
        """
        print(f"\n>>> 任务 1: 生成混合排名精排数据...")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        all_jobs = conn.execute("SELECT securityId, job_name, description, skills FROM jobs").fetchall()

        # 抽样 3300 个基准岗位
        sampled = random.sample(all_jobs, min(len(all_jobs), train_size + test_size))
        # 记录基准岗位原始 ID
        sampled_job_ids = [str(j['securityId']) for j in sampled]

        for j in sampled: self.get_user_id(j['securityId'])
        self.ENTITY_OFFSET = self.user_counter

        train_lines, test_lines = [], []
        for job in tqdm(sampled[:train_size], desc="Generating Train"):
            line = self._process_single_job(job)
            if line: train_lines.append(line)

        for job in tqdm(sampled[train_size:], desc="Generating Test"):
            line = self._process_single_job(job)
            if line: test_lines.append(line)

        train_path = os.path.join(self.output_dir, "train.txt")
        test_path = os.path.join(self.output_dir, "test.txt")
        with open(train_path, "w", encoding='utf-8') as f:
            f.write("\n".join(train_lines))
        with open(test_path, "w", encoding='utf-8') as f:
            f.write("\n".join(test_lines))
        conn.close()

        print(f"[OK] 任务 1 完成，共生成 {len(train_lines) + len(test_lines)} 条阶梯样本。")
        return sampled_job_ids  # 返回锚点 ID 供下一步使用

    def _process_single_job(self, job):
        job_raw_id = str(job['securityId'])
        if job_raw_id not in self.job_id_to_idx: return None

        query_text = f"{job['job_name'] or ''} {job['description'] or ''} {job['skills'] or ''}"
        recall_results = self.recall_system.execute(query_text)
        candidates = recall_results.get('final_top_500', [])
        if len(candidates) < 100: return None

        # 融合排序与抽样
        recall_ranks = {str(aid): i for i, aid in enumerate(candidates)}
        quality_list = sorted([(str(aid), self.author_quality_map.get(str(aid), 0.0)) for aid in candidates],
                              key=lambda x: x[1], reverse=True)
        quality_ranks = {aid: i for i, (aid, _) in enumerate(quality_list)}

        fused = sorted([(str(aid), 0.5 * recall_ranks[str(aid)] + 0.5 * quality_ranks[str(aid)]) for aid in candidates],
                       key=lambda x: x[1])

        num_cand = len(fused)
        pos_ids = [str(self.get_ent_id(f"a_{a[0]}")) for a in fused[:min(100, num_cand)]]
        fair_ids = [str(self.get_ent_id(f"a_{a[0]}")) for a in
                    random.sample(fused[min(100, num_cand):min(400, num_cand)],
                                  min(100, max(0, min(400, num_cand) - 100)))]
        neutral_ids = [str(self.get_ent_id(f"a_{a[0]}")) for a in fused[min(400, num_cand):]]

        potential_pool = list(self.author_quality_map.keys())
        easy_neg_ids = [str(self.get_ent_id(f"a_{aid}")) for aid in random.sample(potential_pool, 100)]

        u_id = self.get_user_id(job_raw_id)
        return f"{u_id};{','.join(pos_ids)};{','.join(fair_ids)};{','.join(neutral_ids)};{','.join(easy_neg_ids)}"

    def generate_kg_topology(self, sampled_job_ids: list):
        """
        任务 2：全量加权拓扑收割
        修正点：提取 pos_weight 和 score，并将权重持久化到 kg_final.txt 中。
        """
        print(f"\n>>> 任务 2: 执行全量 ID 登记与加权拓扑收割...")

        # 1. 登记全量岗位 ID (保持原逻辑)
        all_job_res = self.graph.run("MATCH (j:Job) RETURN j.id as jid").data()
        for res in tqdm(all_job_res, desc="Registering All Jobs"):
            self.get_user_id(str(res['jid']))

        self.ENTITY_OFFSET = self.user_counter

        # 2. 导出锚点 (保持原逻辑)
        anchor_path = os.path.join(self.output_dir, "trained_anchors.json")
        with open(anchor_path, "w", encoding='utf-8') as f:
            json.dump(sampled_job_ids, f, indent=4, ensure_ascii=False)

        kg_triplets = []  # 现在存储 (h, r, t, weight)
        rel_map = {"AUTHORED": 1, "PRODUCED_BY": 2, "PUBLISHED_IN": 3, "HAS_TOPIC": 4, "REQUIRE_SKILL": 5,
                   "SIMILAR_TO": 6}

        for rel_name, rel_id in rel_map.items():
            print(f"[*] 正在收割加权关系: {rel_name} (ID: {rel_id})")
            h_l, t_l = self._get_labels(rel_name)
            if not h_l: continue

            # --- 核心修改：增加对 weight (pos_weight) 和 score 的提取 ---
            query = f"""
            MATCH (n:{h_l})-[r:{rel_name}]->(m:{t_l}) 
            RETURN n.id as hid, m.id as tid, r.pos_weight as weight, r.score as score
            """
            res = self.graph.run(query).data()

            for r in tqdm(res, desc=f"Mapping {rel_name}", leave=False):
                h_raw, t_raw = str(r['hid']), str(r['tid'])

                # 处理 ID 映射
                if rel_id == 5:
                    h_int = self.get_user_id(h_raw)
                else:
                    h_int = self.get_ent_id(f"{self._get_prefix_by_label(h_l)}{h_raw}")
                t_int = self.get_ent_id(f"{self._get_prefix_by_label(t_l)}{t_raw}")

                # --- 核心修改：确定该边的最终权重 ---
                # 1. 如果是 AUTHORED，取 pos_weight (内含时序衰减)
                # 2. 如果是 SIMILAR_TO，取 score
                # 3. 其他默认 1.0
                edge_w = r.get('weight') or r.get('score') or 1.0

                kg_triplets.append((h_int, rel_id, t_int, round(float(edge_w), 4)))

        # 3. 持久化输出加权三元组
        print(f"[*] 正在写入加权拓扑文件 kg_final.txt...")
        # 去重时保留权重
        df = pd.DataFrame(kg_triplets, columns=['h', 'r', 't', 'w']).drop_duplicates(subset=['h', 'r', 't'])
        final_list = df[df['h'] != df['t']].values.tolist()

        with open(os.path.join(self.output_dir, "kg_final.txt"), "w", encoding='utf-8') as f:
            for h, r, t, w in final_list:
                # 格式改为: h r t w
                f.write(f"{int(h)} {int(r)} {int(t)} {w}\n")

        self._save_mapping()
        print(f"[OK] 拓扑构建完成，包含权重信息。最终边数: {len(final_list)}")

    def _get_labels(self, rel):
        m = {"AUTHORED": ("Author", "Work"), "PRODUCED_BY": ("Work", "Institution"), "PUBLISHED_IN": ("Work", "Source"),
             "HAS_TOPIC": ("Work", "Vocabulary"), "REQUIRE_SKILL": ("Job", "Vocabulary"),
             "SIMILAR_TO": ("Vocabulary", "Vocabulary")}
        return m.get(rel, (None, None))

    def _get_prefix_by_label(self, label):
        return {"Author": "a_", "Work": "w_", "Institution": "i_", "Source": "s_", "Vocabulary": "v_", "Job": ""}.get(
            label, "")

    def _save_mapping(self):
        full_mapping = {**self.user_to_int}
        for k, v in self.entity_to_int.items(): full_mapping[k] = v + self.ENTITY_OFFSET
        with open(os.path.join(self.output_dir, "id_map.json"), "w", encoding='utf-8') as f:
            json.dump({"entity": full_mapping, "offset": self.ENTITY_OFFSET,
                       "total_nodes": self.user_counter + self.entity_counter,
                       "user_count": self.user_counter, "entity_count": self.entity_counter}, f, indent=4,
                      ensure_ascii=False)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    gen = KGATAXTrainingGenerator()
    # 1. 运行任务 1 并获取 3300 锚点 ID
    trained_anchors = gen.generate_refined_train_data(train_size=3000, test_size=300)
    # 2. 运行任务 2 执行全量映射与收割
    gen.generate_kg_topology(sampled_job_ids=trained_anchors)