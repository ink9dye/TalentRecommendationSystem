import os
import re
import json
import sqlite3
import faiss
import numpy as np
import pandas as pd
import random
import logging
from tqdm import tqdm
from config import (
    DB_PATH, KGATAX_TRAIN_DATA_DIR, FEATURE_INDEX_PATH,
    JOB_INDEX_PATH, JOB_MAP_PATH,VOCAB_INDEX_PATH,VOCAB_MAP_PATH
)
from src.core.recall.total_recall import TotalRecallSystem


class KGATAXTrainingGenerator:
    def __init__(self):
        self.output_dir = KGATAX_TRAIN_DATA_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 1. 载入预计算的岗位描述向量索引 (核心优化)
        print(f"[*] 正在载入预计算岗位索引: {JOB_INDEX_PATH}")
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_id_to_idx = {sid: i for i, sid in enumerate(json.load(f))}

        # 2. 依然需要召回系统来处理召回逻辑
        self.recall_system = TotalRecallSystem()
        self.recall_system.v_path.verbose = False
        self.recall_system.l_path.verbose = False

        # 3. 载入学术质量索引
        self.author_quality_map = {}
        self._preload_author_quality_from_index()

        # 4. ID 映射管理
        self.user_to_int = {}
        self.entity_to_int = {}
        self.user_counter = 0
        self.entity_counter = 0
        self.ENTITY_OFFSET = 0
        self.stats = {"self_loops": 0, "total_triplets": 0, "user_connectivity": 0.0}

    def _preload_author_quality_from_index(self):
        """加载 build_feature_index.py 生成的归一化指标"""
        if not os.path.exists(FEATURE_INDEX_PATH):
            print(f"[!] 错误: 未发现特征索引，请运行 build_feature_index.py")
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
        raw_id = str(raw_id)
        if raw_id not in self.entity_to_int:
            self.entity_to_int[raw_id] = self.entity_counter
            self.entity_counter += 1
        return self.entity_to_int[raw_id] + self.ENTITY_OFFSET

    def _process_single_job(self, job):
        """核心精排逻辑：使用预计算向量进行极速召回"""
        job_raw_id = str(job['securityId'])

        # 优化：直接从 Faiss 获取该岗位的向量，跳过 SBERT 编码
        if job_raw_id not in self.job_id_to_idx:
            return None

        # 1. 执行召回 (利用已有的 recall_system 获取 500 名候选人)
        # 注意：这里我们仍调用 execute，但 recall_system 内部应能识别向量输入或文本
        query_text = f"{job['job_name'] or ''} {job['description'] or ''} {job['skills'] or ''}"
        recall_results = self.recall_system.execute(query_text)
        candidates = recall_results.get('final_top_500', [])
        if len(candidates) < 480: return None

        # 2. 混合重排逻辑：召回序(50%) + 质量序(50%)
        recall_ranks = {str(aid): i for i, aid in enumerate(candidates)}
        quality_list = [(str(aid), self.author_quality_map.get(str(aid), 0.0)) for aid in candidates]
        quality_list.sort(key=lambda x: x[1], reverse=True)
        quality_ranks = {aid: i for i, (aid, _) in enumerate(quality_list)}

        fused_scored = []
        for aid in candidates:
            aid_str = str(aid)
            fused_rank = 0.5 * recall_ranks[aid_str] + 0.5 * quality_ranks[aid_str]
            fused_scored.append((aid_str, fused_rank))
        fused_scored.sort(key=lambda x: x[1])

        try:
            # 3. 四级梯度抽样
            pos_ids = [str(self.get_ent_id(a[0])) for a in fused_scored[:100]]
            fair_ids = [str(self.get_ent_id(a[0])) for a in random.sample(fused_scored[100:400], 100)]
            neutral_ids = [str(self.get_ent_id(a[0])) for a in fused_scored[400:500]]

            cand_set = set(str(aid) for aid in candidates)
            potential_pool = list(self.author_quality_map.keys())
            easy_neg_raw = random.sample([aid for aid in potential_pool if aid not in cand_set], 100)
            easy_neg_ids = [str(self.get_ent_id(aid)) for aid in easy_neg_raw]

            u_id = self.get_user_id(job_raw_id)
            return f"{u_id};{','.join(pos_ids)};{','.join(fair_ids)};{','.join(neutral_ids)};{','.join(easy_neg_ids)}"
        except Exception:
            return None

    def generate_refined_train_data(self, train_size=1000, test_size=100):
        print(f"\n>>> 任务 1: 生成混合排名精排数据 (Faiss 加速模式)...")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        all_jobs = conn.execute("SELECT securityId, job_name, description, skills FROM jobs").fetchall()

        sampled = random.sample(all_jobs, min(len(all_jobs), train_size + test_size))
        for j in sampled: self.get_user_id(j['securityId'])
        self.ENTITY_OFFSET = self.user_counter

        train_lines, test_lines = [], []
        for job in tqdm(sampled[:train_size], desc="Train Jobs"):
            line = self._process_single_job(job)
            if line: train_lines.append(line)
        for job in tqdm(sampled[train_size:], desc="Test Jobs"):
            line = self._process_single_job(job)
            if line: test_lines.append(line)

        with open(os.path.join(self.output_dir, "train.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(train_lines))
        with open(os.path.join(self.output_dir, "test.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(test_lines))
        conn.close()

    def generate_kg_topology(self):
        """
        构建全量知识图谱拓扑。
        职责：整合学术、语义、岗位及向量桥接的所有关系，生成统一的整数三元组文件。
        关系类型定义：
        1: [Author]-AUTHORED->[Work]
        2: [Work]-PRODUCED_BY->[Institution]
        3: [Work]-PUBLISHED_IN->[Source]
        4: [Work]-HAS_TOPIC->[Vocab]
        5: [Job]-REQUIRE_SKILL->[Vocab]
        6: [Vocab]-SIMILAR_TO->[Vocab] (向量桥接)
        """
        print("\n>>> 任务 2: 构建全量知识图谱拓扑 (包含学术/语义/岗位/向量桥接)...")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        kg_triplets = []

        # --- A. 学术核心拓扑 (Author, Work, Inst, Source) ---
        aship_rows = conn.execute("""
                                  SELECT author_id, work_id, inst_id, source_id
                                  FROM authorships
                                  """).fetchall()

        for r in tqdm(aship_rows, desc="Relating Authors/Works/Insts/Sources"):
            h_auth = self.get_ent_id(f"a_{r['author_id']}")
            t_work = self.get_ent_id(f"w_{r['work_id']}")

            # 1. Author -[1]-> Work
            kg_triplets.append((h_auth, 1, t_work))

            # 2. Work -[2]-> Institution
            if r['inst_id']:
                t_inst = self.get_ent_id(f"i_{r['inst_id']}")
                kg_triplets.append((t_work, 2, t_inst))

            # 3. Work -[3]-> Source (Journal/Conf)
            if r['source_id']:
                t_src = self.get_ent_id(f"s_{r['source_id']}")
                kg_triplets.append((t_work, 3, t_src))

        # --- B. 语义映射 (Work-Topic) ---
        work_rows = conn.execute("SELECT work_id, concepts_text FROM works WHERE concepts_text IS NOT NULL").fetchall()
        for row in tqdm(work_rows, desc="Relating Works to Topics"):
            h_work = self.get_ent_id(f"w_{row['work_id']}")
            for term in re.split(r'[|;,]', row['concepts_text']):
                clean_term = term.strip().lower()
                if clean_term:
                    t_vocab = self.get_ent_id(f"v_{clean_term}")
                    kg_triplets.append((h_work, 4, t_vocab))

        # --- C. 岗位技能 (Job-Skill) ---
        # 注意：Job 在此模型中通常作为 User 空间处理
        job_rows = conn.execute("SELECT securityId, skills FROM jobs WHERE skills IS NOT NULL").fetchall()
        for row in tqdm(job_rows, desc="Relating Jobs to Skills"):
            raw_jid = str(row['securityId'])
            if raw_jid in self.user_to_int:
                h_job = self.user_to_int[raw_jid]  # 这里的 ID 是 User 空间的
                skills = re.split(r'[,，;；/ \t\n]', row['skills'])
                for skill in skills:
                    clean_skill = skill.strip().lower()
                    if clean_skill:
                        t_vocab = self.get_ent_id(f"v_{clean_skill}")
                        kg_triplets.append((h_job, 5, t_vocab))

        # --- D. 向量空间桥接 (Vocab-Vocab SIMILAR_TO) ---
        if os.path.exists(VOCAB_INDEX_PATH):
            print("[*] 正在通过 Faiss 索引构建语义桥接三元组...")
            v_index = faiss.read_index(VOCAB_INDEX_PATH)

            # 重点：此时 v_map 必须存储词汇的文本内容（Term），而不是数据库 ID
            with open(VOCAB_MAP_PATH, 'r', encoding='utf-8') as f:
                v_map = json.load(f)

                # 遍历向量库映射表
            for idx, raw_v_term in enumerate(tqdm(v_map, desc="Vector Bridging")):
                # 从 Faiss 索引中重建该词汇的向量
                vec = v_index.reconstruct(idx).reshape(1, -1)
                # 检索最相似的 5 个邻居
                scores, indices = v_index.search(vec, 5)

                # 修正点 1：统一使用 "v_" + "清洗后的文本" 作为 Key，对齐 Section B 和 C
                clean_source_term = str(raw_v_term).strip().lower()
                h_vocab = self.get_ent_id(f"v_{clean_source_term}")

                for neighbor_idx in indices[0]:
                    # 过滤无效索引及自环
                    if neighbor_idx == -1 or neighbor_idx == idx:
                        continue

                    # 修正点 2：邻居节点也使用文本内容生成 ID
                    neighbor_raw_term = v_map[neighbor_idx]
                    clean_neighbor_term = str(neighbor_raw_term).strip().lower()

                    t_vocab = self.get_ent_id(f"v_{clean_neighbor_term}")

                    # 建立 6 号关系：SIMILAR_TO
                    kg_triplets.append((h_vocab, 6, t_vocab))

        # --- E. 数据清洗与持久化 ---
        print(f"[*] 原始边总数: {len(kg_triplets)}，开始去重与清洗...")
        df = pd.DataFrame(kg_triplets, columns=['h', 'r', 't']).drop_duplicates()
        # 过滤掉自环
        kg_triplets = df[df['h'] != df['t']].values.tolist()

        self.stats["total_triplets"] = len(kg_triplets)

        with open(os.path.join(self.output_dir, "kg_final.txt"), "w", encoding='utf-8') as f:
            for h, r, t in kg_triplets:
                f.write(f"{h} {r} {t}\n")

        # 保存映射表 (包含 offset 逻辑)
        full_mapping = {**self.user_to_int}
        for k, v in self.entity_to_int.items():
            full_mapping[k] = v + self.ENTITY_OFFSET

        with open(os.path.join(self.output_dir, "id_map.json"), "w", encoding='utf-8') as f:
            json.dump({
                "entity": full_mapping,
                "user_count": self.user_counter,
                "entity_count": self.entity_counter,
                "offset": self.ENTITY_OFFSET,
                "total_nodes": self.user_counter + self.entity_counter,
                "stats": self.stats
            }, f, indent=4, ensure_ascii=False)

        conn.close()
        print(f"[OK] 全量图谱构建完成。有效三元组数量: {len(kg_triplets)}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    gen = KGATAXTrainingGenerator()
    gen.generate_refined_train_data(train_size=1000, test_size=100)
    gen.generate_kg_topology()