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

# 导入统一配置
from config import (
    DB_PATH,
    KGATAX_TRAIN_DATA_DIR
)
# 导入您的全量召回系统
from src.core.recall.total_recall import TotalRecallSystem


class KGATAXTrainingGenerator:
    """
    针对两阶段架构优化的训练数据生成器
    核心逻辑：利用岗位描述执行模拟召回 -> 质量过滤筛选正样本 -> 构造精排对比对
    """

    def __init__(self):
        self.output_dir = KGATAX_TRAIN_DATA_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 1. 初始化召回系统
        print("[*] 正在初始化全量召回系统，准备模拟生产环境...")
        self.recall_system = TotalRecallSystem()

        # --- 性能优化：强制关闭内部组件的冗余打印与额外查询 ---
        self.recall_system.v_path.verbose = False

        # 2. 建立全局 ID 映射表 (CKG 统一空间)
        self.entity_to_int = {}
        self.relation_to_int = {
            "interact": 0,
            "out_interact": 1,
            "authored": 2,
            "produced_by": 3,
            "published_in": 4,
            "has_topic": 5,
            "similar_to": 6,
            "require_skill": 7
        }
        self.int_counter = 0

    def get_int_id(self, raw_id):
        raw_id = str(raw_id)
        if raw_id not in self.entity_to_int:
            self.entity_to_int[raw_id] = self.int_counter
            self.int_counter += 1
        return self.entity_to_int[raw_id]

    def generate_refined_train_data(self, sample_size=1000):
        """
        利用岗位详细描述生成精排训练样本
        优化点：引入岗位采样和静默召回，显著提升生成速度
        """
        print(f"\n>>> 任务 1: 生成精排交互数据 (采样规模: {sample_size})")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # 开启 WAL 模式加速大规模读取
        conn.execute("PRAGMA journal_mode=WAL;")

        all_jobs = conn.execute("SELECT securityId, job_name, description FROM jobs").fetchall()

        # --- 采样逻辑：精排模型训练不需要全量 1.1w 岗位，采样 1000 个即可 ---
        if len(all_jobs) > sample_size:
            jobs = random.sample(all_jobs, sample_size)
        else:
            jobs = all_jobs

        train_lines = []
        test_lines = []

        for job in tqdm(jobs, desc="Generating Training Samples"):
            job_raw_id = job['securityId']
            query_text = f"{job['job_name']} {job['description'] or ''}"

            # 1. 执行静默召回 (verbose=False)
            recall_results = self.recall_system.execute(query_text)
            candidates = recall_results.get('final_top_500', [])

            if len(candidates) < 20:
                continue

            # 2. 从数据库拉取 AX 指标进行二次排序
            placeholders = ','.join(['?'] * len(candidates))
            author_stats = conn.execute(
                f"SELECT author_id, h_index, cited_by_count FROM authors WHERE author_id IN ({placeholders})",
                candidates
            ).fetchall()

            scored_authors = []
            for row in author_stats:
                h_idx = row['h_index'] or 0
                citations = row['cited_by_count'] or 0
                quality_score = h_idx * np.log1p(citations)
                scored_authors.append((row['author_id'], quality_score))

            scored_authors.sort(key=lambda x: x[1], reverse=True)

            # 3. 构造对比样本 (Top 15 为正)
            pos_authors = [str(self.get_int_id(a[0])) for a in scored_authors[:15]]

            if pos_authors:
                job_int_id = self.get_int_id(job_raw_id)
                line = f"{job_int_id} " + " ".join(pos_authors)

                if random.random() > 0.2:
                    train_lines.append(line)
                else:
                    test_lines.append(line)

        # 保存考卷
        with open(os.path.join(self.output_dir, "train.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(train_lines))
        with open(os.path.join(self.output_dir, "test.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(test_lines))

        conn.close()

    def generate_kg_topology(self):
        """
        生成知识图谱三元组数据 (kg_final.txt)
        """
        print("\n>>> 任务 2: 构建知识图谱拓扑与语义连接")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        kg_triplets = []

        print("[*] 解析学术协作拓扑...")
        aship_rows = conn.execute("SELECT author_id, work_id, inst_id FROM authorships").fetchall()
        for r in aship_rows:
            kg_triplets.append((self.get_int_id(r['author_id']), 2, self.get_int_id(r['work_id'])))
            if r['inst_id']:
                kg_triplets.append((self.get_int_id(r['work_id']), 3, self.get_int_id(r['inst_id'])))

        print("[*] 解析语义锚定关系...")
        work_rows = conn.execute("SELECT work_id, concepts_text FROM works").fetchall()
        for row in work_rows:
            if row['concepts_text']:
                h_work = self.get_int_id(row['work_id'])
                for term in row['concepts_text'].split('|'):
                    kg_triplets.append((h_work, 5, self.get_int_id(f"v_{term.strip().lower()}")))

        job_rows = conn.execute("SELECT securityId, skills FROM jobs WHERE skills IS NOT NULL").fetchall()
        for row in job_rows:
            h_job = self.get_int_id(row['securityId'])
            for skill in re.split(r'[,，;；/]', row['skills']):
                kg_triplets.append((h_job, 7, self.get_int_id(f"v_{skill.strip().lower()}")))

        print(f"[*] 正在保存 {len(kg_triplets)} 条三元组边...")
        with open(os.path.join(self.output_dir, "kg_final.txt"), "w", encoding='utf-8') as f:
            for h, r, t in kg_triplets:
                f.write(f"{h} {r} {t}\n")

        mapping = {"entity": self.entity_to_int, "relation": self.relation_to_int}
        with open(os.path.join(self.output_dir, "id_map.json"), "w", encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)

        conn.close()
        print(f"\n[成功] 训练数据准备完成！节点数: {len(self.entity_to_int)}")


if __name__ == "__main__":
    generator = KGATAXTrainingGenerator()
    # 采样 1000 个岗位生成精排训练集
    generator.generate_refined_train_data(sample_size=1000)
    generator.generate_kg_topology()