import os
import re
import gc
import torch
import json
import sqlite3
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from config import DB_PATH, KGATAX_TRAIN_DATA_DIR
from src.core.recall.total_recall import TotalRecallSystem


class KGATAXTrainingGenerator:
    def __init__(self):
        self.output_dir = KGATAX_TRAIN_DATA_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.recall_system = TotalRecallSystem()
        self.recall_system.v_path.verbose = False
        self.recall_system.l_path.verbose = False

        # --- ID 分区管理 ---
        self.user_to_int = {}  # Job 映射 (User)
        self.entity_to_int = {}  # 其他映射 (Entity)
        self.user_counter = 0
        self.entity_counter = 0
        self.ENTITY_OFFSET = 1000000  # 预留足够的空间给 User

        self.relation_to_int = {
            "interact": 0, "out_interact": 1, "authored": 2, "produced_by": 3,
            "published_in": 4, "has_topic": 5, "similar_to": 6, "require_skill": 7
        }

        # 质量统计哨兵
        self.stats = {
            "self_loops": 0,
            "total_triplets": 0,
            "user_connectivity": 0.0
        }

    def get_user_id(self, raw_id):
        raw_id = str(raw_id)
        if raw_id not in self.user_to_int:
            self.user_to_int[raw_id] = self.user_counter
            self.user_counter += 1
        return self.user_to_int[raw_id]

    def get_ent_id(self, raw_id):
        """核心修复：确保返回字典中固定的 ID，而非累加器当前值"""
        raw_id = str(raw_id)
        if raw_id not in self.entity_to_int:
            self.entity_to_int[raw_id] = self.entity_counter
            self.entity_counter += 1
        return self.entity_to_int[raw_id]

    def generate_refined_train_data(self, sample_size=1000):
        print(f"\n>>> 任务 1: 生成精排交互数据 (目标采样: {sample_size})")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        all_jobs = conn.execute("SELECT securityId, job_name, description FROM jobs").fetchall()
        jobs = random.sample(all_jobs, sample_size) if len(all_jobs) > sample_size else all_jobs

        train_lines, test_lines = [], []
        processed_success = 0

        for job in tqdm(jobs, desc="Generating Training Samples"):
            job_raw_id = str(job['securityId'])
            query_text = f"{job['job_name'] or ''} {job['description'] or ''}"

            recall_results = self.recall_system.execute(query_text)
            candidates = recall_results.get('final_top_500', [])

            if not candidates or len(candidates) < 20:
                continue

            placeholders = ','.join(['?'] * len(candidates))
            author_stats = conn.execute(
                f"SELECT author_id, h_index, cited_by_count FROM authors WHERE author_id IN ({placeholders})",
                candidates
            ).fetchall()

            scored_list = []
            for r in author_stats:
                h_idx = float(r['h_index'] or 0)
                citations = float(r['cited_by_count'] or 0)
                quality_score = np.log1p(h_idx) * np.log1p(citations)
                scored_list.append((str(r['author_id']), quality_score))

            scored_list.sort(key=lambda x: x[1], reverse=True)

            try:
                # 转换为全局 ID 并校验
                pos_authors = [str(self.get_ent_id(a[0]) + self.ENTITY_OFFSET) for a in scored_list[:15]]
                u_id = self.get_user_id(job_raw_id)

                if u_id >= self.ENTITY_OFFSET:
                    continue

                line = f"{u_id} " + " ".join(pos_authors)
                if random.random() > 0.2:
                    train_lines.append(line)
                else:
                    test_lines.append(line)
                processed_success += 1
            except Exception:
                continue

        with open(os.path.join(self.output_dir, "train.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(train_lines))
        with open(os.path.join(self.output_dir, "test.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(test_lines))
        conn.close()

    def generate_kg_topology(self):
        print("\n>>> 任务 2: 构建知识图谱拓扑 (带质量校验)")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        kg_triplets = []

        # A. Author - Work - Institution (拓扑核心)
        aship_rows = conn.execute("SELECT author_id, work_id, inst_id FROM authorships").fetchall()
        for r in aship_rows:
            h = self.get_ent_id(r['author_id']) + self.ENTITY_OFFSET
            t_w = self.get_ent_id(r['work_id']) + self.ENTITY_OFFSET

            if h != t_w:  # 自环拦截 1
                kg_triplets.append((h, 2, t_w))
            else:
                self.stats["self_loops"] += 1

            if r['inst_id']:
                t_i = self.get_ent_id(r['inst_id']) + self.ENTITY_OFFSET
                if t_w != t_i:  # 自环拦截 2
                    kg_triplets.append((t_w, 3, t_i))

        # B. Work - Topic
        work_rows = conn.execute("SELECT work_id, concepts_text FROM works WHERE concepts_text IS NOT NULL").fetchall()
        for row in work_rows:
            h = self.get_ent_id(row['work_id']) + self.ENTITY_OFFSET
            for term in row['concepts_text'].split('|'):
                clean_term = term.strip().lower()
                if not clean_term: continue
                t = self.get_ent_id(f"v_{clean_term}") + self.ENTITY_OFFSET
                if h != t:
                    kg_triplets.append((h, 5, t))

        # C. Job - Skill (User - Entity 连接)
        job_rows = conn.execute("SELECT securityId, skills FROM jobs WHERE skills IS NOT NULL").fetchall()
        for row in job_rows:
            h = self.get_user_id(row['securityId'])
            skills = re.split(r'[,，;；/ \t\n]', row['skills'])
            for skill in skills:
                clean_skill = skill.strip().lower()
                if not clean_skill: continue
                t = self.get_ent_id(f"v_{clean_skill}") + self.ENTITY_OFFSET
                kg_triplets.append((h, 7, t))

        # 执行静态全量扫描
        self._perform_final_sanity_check(kg_triplets)

        with open(os.path.join(self.output_dir, "kg_final.txt"), "w") as f:
            for h, r, t in kg_triplets: f.write(f"{h} {r} {t}\n")

        # 导出最终映射及质量统计
        full_mapping = {**self.user_to_int}
        for k, v in self.entity_to_int.items():
            full_mapping[k] = v + self.ENTITY_OFFSET

        with open(os.path.join(self.output_dir, "id_map.json"), "w") as f:
            json.dump({
                "entity": full_mapping,
                "user_count": self.user_counter,
                "offset": self.ENTITY_OFFSET,
                "quality_stats": self.stats
            }, f)
        conn.close()

    def _perform_final_sanity_check(self, triplets):
        """三元组全量静态分析，杜绝无效数据进入训练器"""
        print("\n[Sanity Check] 执行数据合规性扫描...")
        df = pd.DataFrame(triplets, columns=['h', 'r', 't'])

        # 1. 物理检查自环
        actual_loops = (df['h'] == df['t']).sum()
        if actual_loops > 0:
            print(f"  [!] 发现 {actual_loops} 条残余自环，执行物理隔离。")
            triplets[:] = [tri for tri in triplets if tri[0] != tri[2]]

        # 2. 检查 Job 节点在 KG 中的覆盖度
        user_nodes_in_kg = set(df[df['r'] == 7]['h'].unique())
        self.stats["user_connectivity"] = len(user_nodes_in_kg) / (self.user_counter + 1e-9)
        print(f"  [*] 岗位覆盖率: {self.stats['user_connectivity']:.2%}")

        # 3. 检查最大物理 ID 是否越界
        global_max = df[['h', 't']].max().max()
        expected_max = self.entity_counter + self.ENTITY_OFFSET
        if global_max >= expected_max:
            print(f"  [!] ID 空间警告: 物理最大 ID {global_max} 接近/超过 预期最大值 {expected_max}")

        self.stats["total_triplets"] = len(triplets)


if __name__ == "__main__":
    import time
    import logging
    import warnings

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    print("\n" + "★" * 50)
    print(" KGATAX 训练数据生成流水线 (质量增强版) ".center(44))
    print("★" * 50)

    try:
        generator = KGATAXTrainingGenerator()

        # 任务 1
        generator.generate_refined_train_data(sample_size=1000)

        # 任务 2
        generator.generate_kg_topology()

        print("\n" + "★" * 50)
        print(f"【生成报告】")
        print(f"● 排除自环数: {generator.stats['self_loops']}")
        print(f"● 有效三元组: {generator.stats['total_triplets']}")
        print(f"● 岗位连通率: {generator.stats['user_connectivity']:.2%}")
        print("★" * 50 + "\n")

    except Exception as e:
        print(f"\n[Error] 流水线执行崩溃: {str(e)}")