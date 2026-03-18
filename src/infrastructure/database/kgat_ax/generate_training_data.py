"""
KGAT-AX 训练数据生成器。

职责（对齐 README 修改计划 5.5）：
- 训练样本入口优先基于 candidate_pool.candidate_records；final_top_500 仅作兼容回退。
- 分层正负样本：Strong/Weak Positive，EasyNeg/FieldNeg/HardNeg/CollabNeg。
- 可选导出四分支字段到 train_four_branch.json / test_four_branch.json，供 DataLoader 与四分支模型使用。
"""
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

# 分层正负样本标签（README 5.5）
LABEL_STRONG_POS = "strong_pos"
LABEL_WEAK_POS = "weak_pos"
LABEL_HARD_NEG = "hard_neg"
LABEL_COLLAB_NEG = "collab_neg"
LABEL_FIELD_NEG = "field_neg"
LABEL_EASY_NEG = "easy_neg"


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
        任务 1：生成精排训练样本（优先基于 candidate_pool.candidate_records，分层正负样本，README 5.5）。
        同时写入 train_four_branch.json / test_four_branch.json（四分支字段），供 DataLoader 与四分支模型使用。
        返回：被抽样用于训练的岗位 ID 列表 (锚点)。
        """
        print(f"\n>>> 任务 1: 生成混合排名精排数据（候选池入口 + 分层正负样本 + 四分支导出）...")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        all_jobs = conn.execute("SELECT securityId, job_name, description, skills, domain_ids FROM jobs").fetchall()
        sampled = random.sample(all_jobs, min(len(all_jobs), train_size + test_size))
        sampled_job_ids = [str(j['securityId']) for j in sampled]

        train_lines, test_lines = [], []
        train_four_branch, test_four_branch = {}, {}

        for job in tqdm(sampled[:train_size], desc="Generating Train"):
            line, aux = self._process_single_job(job)
            if line:
                train_lines.append(line)
                if aux:
                    train_four_branch.update(aux)

        for job in tqdm(sampled[train_size:], desc="Generating Test"):
            line, aux = self._process_single_job(job)
            if line:
                test_lines.append(line)
                if aux:
                    test_four_branch.update(aux)

        train_path = os.path.join(self.output_dir, "train.txt")
        test_path = os.path.join(self.output_dir, "test.txt")
        with open(train_path, "w", encoding='utf-8') as f:
            f.write("\n".join(train_lines))
        with open(test_path, "w", encoding='utf-8') as f:
            f.write("\n".join(test_lines))

        if train_four_branch:
            with open(os.path.join(self.output_dir, "train_four_branch.json"), "w", encoding='utf-8') as f:
                json.dump(train_four_branch, f, ensure_ascii=False, indent=2)
        if test_four_branch:
            with open(os.path.join(self.output_dir, "test_four_branch.json"), "w", encoding='utf-8') as f:
                json.dump(test_four_branch, f, ensure_ascii=False, indent=2)

        conn.close()
        print(f"[OK] 任务 1 完成，共生成 {len(train_lines) + len(test_lines)} 条阶梯样本；四分支条目 train={len(train_four_branch)} test={len(test_four_branch)}。")
        return sampled_job_ids

    def _process_single_job(self, job):
        """
        处理单个岗位：优先使用 candidate_pool.candidate_records，分层正负样本（README 5.5）；
        不足时回退到 final_top_500 + 原四级梯度。返回 (train_line, four_branch_aux_dict 或 None)。
        """
        job_raw_id = str(job['securityId'])
        if job_raw_id not in self.job_id_to_idx:
            return None, None

        query_text = f"{job['job_name'] or ''} {job['description'] or ''} {job['skills'] or ''}"
        use_domain = random.random() < 0.75
        target_domain = job['domain_ids'] if use_domain else None
        recall_results = self.recall_system.execute(query_text, domain_id=target_domain)

        pool = recall_results.get("candidate_pool")
        records = getattr(pool, "candidate_records", None) if pool else None

        if records and len(records) >= 100:
            line, aux = self._build_from_candidate_pool(job_raw_id, records)
            return line, aux
        # 回退：使用 final_top_500，保持原四级梯度逻辑
        candidates = recall_results.get("final_top_500", [])
        if len(candidates) < 100:
            return None, None

        recall_ranks = {str(aid): i for i, aid in enumerate(candidates)}
        quality_list = sorted(
            [(str(aid), self.author_quality_map.get(str(aid), 0.0)) for aid in candidates],
            key=lambda x: x[1], reverse=True,
        )
        quality_ranks = {aid: i for i, (aid, _) in enumerate(quality_list)}
        fused = sorted(
            [(str(aid), 0.6 * recall_ranks[str(aid)] + 0.4 * quality_ranks[str(aid)]) for aid in candidates],
            key=lambda x: x[1],
        )
        num_cand = len(fused)
        pos_ids = [str(self.get_ent_id(f"a_{a[0]}")) for a in fused[: min(100, num_cand)]]
        fair_ids = [
            str(self.get_ent_id(f"a_{a[0]}"))
            for a in random.sample(
                fused[min(100, num_cand) : min(400, num_cand)],
                min(100, max(0, min(400, num_cand) - 100)),
            )
        ]
        neutral_ids = [str(self.get_ent_id(f"a_{a[0]}")) for a in fused[min(400, num_cand) :]]
        potential_pool = list(self.author_quality_map.keys())
        easy_neg_ids = [str(self.get_ent_id(f"a_{aid}")) for aid in random.sample(potential_pool, 100)]
        u_id = self.get_user_id(job_raw_id)
        line = f"{u_id};{','.join(pos_ids)};{','.join(fair_ids)};{','.join(neutral_ids)};{','.join(easy_neg_ids)}"
        return line, None

    def _classify_record(self, r, rank_by_score):
        """将一条 CandidateRecord 分为 Strong Positive / Weak Positive / HardNeg / CollabNeg / FieldNeg（README 5.5）。"""
        if not getattr(r, "passed_hard_filter", True):
            return LABEL_FIELD_NEG
        bucket = (getattr(r, "bucket_type") or "").strip().upper()
        from_label = getattr(r, "from_label", False)
        from_collab = getattr(r, "from_collab", False)
        path_count = getattr(r, "path_count", 0) or 0

        if (bucket == "A" and (from_label or path_count >= 2)):
            return LABEL_STRONG_POS
        if bucket in ("A", "B") and (from_label or path_count >= 2 or getattr(r, "from_vector", False)):
            return LABEL_WEAK_POS
        if from_collab and path_count == 1 and not getattr(r, "from_label", False) and not getattr(r, "from_vector", False):
            return LABEL_COLLAB_NEG
        if (from_label or path_count >= 2) and rank_by_score.get(r.author_id, 999) > 100:
            return LABEL_HARD_NEG
        return LABEL_FIELD_NEG

    def _build_from_candidate_pool(self, job_raw_id, records):
        """基于 candidate_records 构建分层正负样本并生成 train 行与四分支侧车。"""
        u_id = self.get_user_id(job_raw_id)
        pool_aids = {r.author_id for r in records}
        max_score = max((r.candidate_pool_score or 0.0) for r in records) or 1.0
        rank_by_score = {}
        for i, r in enumerate(sorted(records, key=lambda x: -(x.candidate_pool_score or 0.0))):
            rank_by_score[r.author_id] = i

        strong_pos, weak_pos, hard_neg, collab_neg, field_neg = [], [], [], [], []
        for r in records:
            label = self._classify_record(r, rank_by_score)
            if label == LABEL_STRONG_POS:
                strong_pos.append(r)
            elif label == LABEL_WEAK_POS:
                weak_pos.append(r)
            elif label == LABEL_HARD_NEG:
                hard_neg.append(r)
            elif label == LABEL_COLLAB_NEG:
                collab_neg.append(r)
            else:
                field_neg.append(r)

        pos_cap = min(100, len(strong_pos) + len(weak_pos))
        pos_pool = sorted(strong_pos + weak_pos, key=lambda x: -(x.candidate_pool_score or 0.0))
        pos_records = pos_pool[:pos_cap]
        if len(pos_records) < 100:
            all_by_score = sorted(records, key=lambda x: -(x.candidate_pool_score or 0.0))
            pos_set = {r.author_id for r in pos_records}
            for r in all_by_score:
                if len(pos_records) >= 100:
                    break
                if r.author_id not in pos_set:
                    pos_records.append(r)
                    pos_set.add(r.author_id)
        rest_weak = [r for r in pos_pool if r not in pos_records]
        fair_records = (rest_weak + hard_neg)[:100]
        neutral_records = field_neg[:150]
        easy_neg_aids = [aid for aid in random.sample(list(self.author_quality_map.keys()), 200) if aid not in pool_aids][:100]
        if len(easy_neg_aids) < 100:
            easy_neg_aids += [r.author_id for r in collab_neg[: 100 - len(easy_neg_aids)]]

        pos_ids = [str(self.get_ent_id(f"a_{r.author_id}")) for r in pos_records]
        fair_ids = [str(self.get_ent_id(f"a_{r.author_id}")) for r in fair_records]
        neutral_ids = [str(self.get_ent_id(f"a_{r.author_id}")) for r in neutral_records]
        easy_neg_ids = [str(self.get_ent_id(f"a_{aid}")) for aid in easy_neg_aids]
        if len(easy_neg_ids) < 100:
            easy_neg_ids += [str(self.get_ent_id(f"a_{r.author_id}")) for r in collab_neg[: 100 - len(easy_neg_ids)]]

        four_branch = {}
        for eid, r in zip(pos_ids, pos_records):
            four_branch[f"u{u_id}_i{eid}"] = self._export_four_branch_row(r, max_score)
        for eid, r in zip(fair_ids, fair_records):
            four_branch[f"u{u_id}_i{eid}"] = self._export_four_branch_row(r, max_score)
        for eid, r in zip(neutral_ids, neutral_records):
            four_branch[f"u{u_id}_i{eid}"] = self._export_four_branch_row(r, max_score)
        for aid in easy_neg_aids:
            eid = str(self.get_ent_id(f"a_{aid}"))
            four_branch[f"u{u_id}_i{eid}"] = self._export_four_branch_row(None, max_score)
        for r in collab_neg[: max(0, 100 - len(easy_neg_aids))]:
            eid = str(self.get_ent_id(f"a_{r.author_id}"))
            four_branch[f"u{u_id}_i{eid}"] = self._export_four_branch_row(r, max_score)

        line = f"{u_id};{','.join(pos_ids)};{','.join(fair_ids)};{','.join(neutral_ids)};{','.join(easy_neg_ids)}"
        return line, four_branch

    def _export_four_branch_row(self, record, max_pool_score):
        """导出一条 (job, author) 的四分支特征列表：recall 13 维、author_aux 12 维、interaction 8 维。"""
        if record is None:
            return {"recall": [0.0] * 13, "author_aux": [0.0] * 12, "interaction": [0.0] * 8}
        max_rank = 500.0
        r = record
        v_rank = getattr(r, "vector_rank", None) or 0
        l_rank = getattr(r, "label_rank", None) or 0
        c_rank = getattr(r, "collab_rank", None) or 0
        recall = [
            1.0 if getattr(r, "from_vector", False) else 0.0,
            1.0 if getattr(r, "from_label", False) else 0.0,
            1.0 if getattr(r, "from_collab", False) else 0.0,
            (getattr(r, "path_count", 0) or 0) / 3.0,
            1.0 - min(v_rank / max_rank, 1.0) if v_rank else 0.0,
            1.0 - min(l_rank / max_rank, 1.0) if l_rank else 0.0,
            1.0 - min(c_rank / max_rank, 1.0) if c_rank else 0.0,
            (float(r.candidate_pool_score or 0.0) / max_pool_score) if max_pool_score else 0.0,
            {"A": 0.25, "B": 0.5, "C": 0.75, "D": 1.0}.get((getattr(r, "bucket_type") or "D").strip().upper(), 0.5),
            1.0 if getattr(r, "is_multi_path_hit", False) else 0.0,
            float(getattr(r, "vector_score_raw") or 0.0),
            float(getattr(r, "label_score_raw") or 0.0),
            float(getattr(r, "collab_score_raw") or 0.0),
        ]
        h = getattr(r, "h_index", None)
        w = getattr(r, "works_count", None)
        c = getattr(r, "cited_by_count", None)
        author_aux = [
            np.log1p(h) if h is not None else 0.0,
            np.log1p(c) if c is not None else 0.0,
            np.log1p(w) if w is not None else 0.0,
        ] + [0.0] * 9
        interaction = [
            float(getattr(r, "topic_similarity") or 0.0),
            float(getattr(r, "skill_coverage_ratio") or 0.0),
            float(getattr(r, "domain_consistency") or 0.0),
            float(getattr(r, "paper_hit_strength") or 0.0),
            float(getattr(r, "recent_activity_match") or 0.0),
            0.0,
            0.0,
            0.0,
        ]
        return {"recall": recall, "author_aux": author_aux, "interaction": interaction}

    def generate_kg_topology(self, sampled_job_ids: list):
        """
        任务 2：全量加权拓扑收割
        修正点：提取 pos_weight 和 score，并将权重持久化到 kg_final.txt 中。
        """
        print(f"\n>>> 任务 2: 执行全量 ID 登记与加权拓扑收割...")



        #  导出锚点 (保持原逻辑)
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

    # 第一步：锁定全局 ID 偏移量 (从 Neo4j 读取全量岗位)
    print("\n[*] 正在从 Neo4j 锁定全局 ID 偏移量 (执行人口普查)...")
    # 这里的查询必须覆盖所有可能的 Job 节点
    all_job_res = gen.graph.run("MATCH (j:Job) RETURN j.id as jid").data()

    for res in tqdm(all_job_res, desc="登记全量岗位 ID"):
        # 强制将 Neo4j 里的所有 Job ID 映射到 User 空间 (0 ~ OFFSET-1)
        gen.get_user_id(str(res['jid']))

    # 核心锁：一旦赋值，后面绝不能再改 self.ENTITY_OFFSET
    gen.ENTITY_OFFSET = gen.user_counter
    print(f"[OK] ENTITY_OFFSET 已锁定为: {gen.ENTITY_OFFSET}")

    # 第二步：生成精排样本 (此时使用已锁定的 OFFSET)
    trained_anchors = gen.generate_refined_train_data(train_size=3000, test_size=300)

    # 第三步：执行全量拓扑收割
    gen.generate_kg_topology(sampled_job_ids=trained_anchors)