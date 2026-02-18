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

        # 1. 初始化 Neo4j 连接：作为核心关系的“收割”来源
        self.graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), name=NEO4J_DATABASE)

        # 2. 载入预计算的岗位描述向量索引
        print(f"[*] 正在载入预计算岗位索引: {JOB_INDEX_PATH}")
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_id_to_idx = {sid: i for i, sid in enumerate(json.load(f))}

        # 3. 召回系统（用于生成 CF 阶段的负采样样本）
        self.recall_system = TotalRecallSystem()
        self.recall_system.v_path.verbose = False
        self.recall_system.l_path.verbose = False

        # 4. 载入学术质量索引 (AX 特征基础)
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
            # 融合指标作为样本抽样的权重参考
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

    def harvest_neo4j_relations(self):
        """
        核心任务：从 Neo4j 提取所有桥接边，确保训练数据不再是孤岛。
        1. SIMILAR_TO (Relation 6): 打通工业词与学术词的关联。
        2. REQUIRE_SKILL (Relation 5): 补全岗位与技能的逻辑连接。
        """
        print("\n>>> 正在从 Neo4j 收割关键语义桥接边 (Relation 5 & 6)...")
        extra_triplets = []

        # A. 提取词汇间的语义桥接 (对应 builder.py 中的相似度关联)
        # 使用 term 而非 id，以匹配 generate_kg_topology 中的 ID 映射策略
        bridge_query = "MATCH (v1:Vocabulary)-[:SIMILAR_TO]->(v2:Vocabulary) RETURN v1.term AS f, v2.term AS t"
        results = self.graph.run(bridge_query).data()
        for res in results:
            h = self.get_ent_id(f"v_{res['f']}")
            t = self.get_ent_id(f"v_{res['t']}")
            extra_triplets.append((h, 6, t))

        # B. 提取岗位与技能的关联
        job_skill_query = "MATCH (j:Job)-[:REQUIRE_SKILL]->(v:Vocabulary) RETURN j.id AS jid, v.term AS vterm"
        results = self.graph.run(job_skill_query).data()
        for res in results:
            raw_jid = str(res['jid'])
            if raw_jid in self.user_to_int:
                h_job = self.user_to_int[raw_jid] # 岗位在 User 空间
                t_vocab = self.get_ent_id(f"v_{res['vterm']}")
                extra_triplets.append((h_job, 5, t_vocab))

        print(f"[OK] 成功从 Neo4j 捕获 {len(extra_triplets)} 条关联边。")
        return extra_triplets

    def _process_single_job(self, job):
        """
        核心精排逻辑：使用预计算向量进行极速召回
        修改点：
        1. 降低候选人门槛（从 480 降至 100），防止样本量不足导致文件为空。
        2. 增加调试输出，方便定位为何岗位被跳过。
        """
        job_raw_id = str(job['securityId'])

        if job_raw_id not in self.job_id_to_idx:
            # print(f"[Debug] 跳过岗位 {job_raw_id}: 无预计算向量")
            return None

        # 1. 执行召回 (利用已有的 recall_system 获取候选人)
        query_text = f"{job['job_name'] or ''} {job['description'] or ''} {job['skills'] or ''}"
        recall_results = self.recall_system.execute(query_text)
        candidates = recall_results.get('final_top_500', [])

        # --- 核心修改：降低阈值 ---
        # 如果当前数据量较小，建议设为 100 甚至更低，确保训练能跑通
        if len(candidates) < 100:
            # print(f"[Debug] 跳过岗位 {job_raw_id}: 召回人数不足 ({len(candidates)} < 100)")
            return None

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
            # 3. 四级梯度抽样（确保 ID 空间对齐）
            # 根据实际召回人数动态调整抽样上限
            num_cand = len(fused_scored)
            pos_end = min(100, num_cand)
            fair_end = min(400, num_cand)

            pos_ids = [str(self.get_ent_id(f"a_{a[0]}")) for a in fused_scored[:pos_end]]
            # 这里的 random.sample 需保护，防止范围越界
            fair_sample_size = min(100, max(0, fair_end - pos_end))
            fair_ids = [str(self.get_ent_id(f"a_{a[0]}")) for a in
                        random.sample(fused_scored[pos_end:fair_end], fair_sample_size)]
            neutral_ids = [str(self.get_ent_id(f"a_{a[0]}")) for a in fused_scored[fair_end:]]

            # 处理易负样本（Easy Negatives）
            cand_set = set(str(aid) for aid in candidates)
            potential_pool = list(self.author_quality_map.keys())
            easy_neg_raw = random.sample([aid for aid in potential_pool if aid not in cand_set], 100)
            easy_neg_ids = [str(self.get_ent_id(f"a_{aid}")) for aid in easy_neg_raw]

            u_id = self.get_user_id(job_raw_id)

            return f"{u_id};{','.join(pos_ids)};{','.join(fair_ids)};{','.join(neutral_ids)};{','.join(easy_neg_ids)}"
        except Exception as e:
            print(f"[Error] 岗位 {job_raw_id} 抽样失败: {e}")
            return None

    def generate_refined_train_data(self, train_size=1000, test_size=100):
        """
        生成混合排名精排数据
        修改点：增加绝对路径打印和有效行数统计。
        """
        print(f"\n>>> 任务 1: 生成混合排名精排数据 (Faiss 加速模式)...")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        all_jobs = conn.execute("SELECT securityId, job_name, description, skills FROM jobs").fetchall()

        # 采样任务
        sampled = random.sample(all_jobs, min(len(all_jobs), train_size + test_size))
        for j in sampled: self.get_user_id(j['securityId'])
        self.ENTITY_OFFSET = self.user_counter

        train_lines, test_lines = [], []

        # 处理训练集
        for job in tqdm(sampled[:train_size], desc="Generating Train Samples"):
            line = self._process_single_job(job)
            if line: train_lines.append(line)

        # 处理测试集
        for job in tqdm(sampled[train_size:], desc="Generating Test Samples"):
            line = self._process_single_job(job)
            if line: test_lines.append(line)

        # 确保文件即使为空也会被创建，并打印路径
        train_path = os.path.abspath(os.path.join(self.output_dir, "train.txt"))
        test_path = os.path.abspath(os.path.join(self.output_dir, "test.txt"))

        with open(train_path, "w", encoding='utf-8') as f:
            f.write("\n".join(train_lines))
        with open(test_path, "w", encoding='utf-8') as f:
            f.write("\n".join(test_lines))

        conn.close()

        print(f"[OK] 任务 1 完成：")
        print(f"  - 有效训练岗位: {len(train_lines)} (目标: {train_size})")
        print(f"  - 有效测试岗位: {len(test_lines)} (目标: {test_size})")
        print(f"  - 文件已存至: {train_path}")

    def generate_kg_topology(self):
        """
        全量收割模式：直接从 Neo4j 导出全量 12.8M 条边。
        这能确保训练集的边数与 Neo4j 数据库物理状态完全对齐。
        """
        print(f"\n>>> 任务 2: 正在从 Neo4j 全量收割 12.8M 条边...")
        kg_triplets = []

        # 1. 定义关系类型到模型编号的映射 (1-6)
        # 严格对应你论文和数据加载器中的关系定义
        rel_map = {
            "AUTHORED": 1,  # Author -> Work
            "PRODUCED_BY": 2,  # Work -> Institution
            "PUBLISHED_IN": 3,  # Work -> Source
            "HAS_TOPIC": 4,  # Work -> Vocabulary
            "REQUIRE_SKILL": 5,  # Job -> Vocabulary
            "SIMILAR_TO": 6  # Vocabulary -> Vocabulary
        }

        # 2. 遍历关系类型执行收割
        for rel_name, rel_id in rel_map.items():
            print(f"[*] 正在收割关系类型: {rel_name} (Code: {rel_id})...")

            # 根据关系名确定头尾节点的标签，用于精准查询
            # 这里的逻辑必须与你 builder.py 构建图谱时的标签完全一致
            if rel_name == "AUTHORED":
                h_label, t_label = "Author", "Work"
            elif rel_name == "PRODUCED_BY":
                h_label, t_label = "Work", "Institution"
            elif rel_name == "PUBLISHED_IN":
                h_label, t_label = "Work", "Source"
            elif rel_name == "HAS_TOPIC":
                h_label, t_label = "Work", "Vocabulary"
            elif rel_name == "REQUIRE_SKILL":
                h_label, t_label = "Job", "Vocabulary"
            elif rel_name == "SIMILAR_TO":
                h_label, t_label = "Vocabulary", "Vocabulary"
            else:
                continue

            # 执行 Cypher 全量查询
            # 备注：对于 12M 数据，py2neo 的 data() 会占用一定内存，但 32GB 环境可以承受
            cypher = f"MATCH (n:{h_label})-[r:{rel_name}]->(m:{t_label}) RETURN n.id as hid, m.id as tid"
            results = self.graph.run(cypher).data()

            for res in tqdm(results, desc=f"Mapping {rel_name}", leave=False):
                h_raw_id = str(res['hid'])
                t_raw_id = str(res['tid'])

                # --- 关键 ID 映射逻辑 ---
                # A. 处理头节点
                if rel_id == 5:
                    # Job 节点被视为 User，使用 get_user_id (0-offset)
                    h_int = self.get_user_id(h_raw_id)
                else:
                    # 其他节点属于 Entity 空间 (offset 以后)
                    # 自动根据标签补全前缀，确保 ID 空间唯一性
                    prefix = self._get_prefix_by_label(h_label)
                    h_int = self.get_ent_id(f"{prefix}{h_raw_id}")

                # B. 处理尾节点 (全量归属于 Entity 空间)
                t_prefix = self._get_prefix_by_label(t_label)
                t_int = self.get_ent_id(f"{t_prefix}{t_raw_id}")

                kg_triplets.append((h_int, rel_id, t_int))

        # 3. 终极去重与自环清洗
        print(f"[*] 收割完毕，正在执行最终清洗...")
        df = pd.DataFrame(kg_triplets, columns=['h', 'r', 't']).drop_duplicates()
        final_list = df[df['h'] != df['t']].values.tolist()

        # 4. 持久化输出
        with open(os.path.join(self.output_dir, "kg_final.txt"), "w", encoding='utf-8') as f:
            for h, r, t in final_list:
                f.write(f"{h} {r} {t}\n")

        self._save_mapping()
        print(f"[OK] 拓扑构建完成。最终边数: {len(final_list)}，已成功对齐 Neo4j。")

    def _get_prefix_by_label(self, label):
        """辅助函数：根据节点标签返回 ID 前缀"""
        prefixes = {
            "Author": "a_",
            "Work": "w_",
            "Institution": "i_",
            "Source": "s_",
            "Vocabulary": "v_",
            "Job": ""  # Job 在该系统中通常直接作为 User ID
        }
        return prefixes.get(label, "")

    def _save_mapping(self):
        full_mapping = {**self.user_to_int}
        for k, v in self.entity_to_int.items():
            full_mapping[k] = v + self.ENTITY_OFFSET
        with open(os.path.join(self.output_dir, "id_map.json"), "w", encoding='utf-8') as f:
            json.dump({
                "entity": full_mapping,
                "offset": self.ENTITY_OFFSET,
                "total_nodes": self.user_counter + self.entity_counter,
                "user_count": self.user_counter,
                "entity_count": self.entity_counter
            }, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    gen = KGATAXTrainingGenerator()
    gen.generate_refined_train_data(train_size=3000, test_size=300)
    gen.generate_kg_topology()