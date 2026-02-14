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

        # --- 性能优化：强制关闭内部组件的冗余打印，确保 tqdm 界面整洁 ---
        # 此时召回系统已内部预载向量矩阵
        self.recall_system.v_path.verbose = False
        self.recall_system.l_path.verbose = False

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
        优化点：强制命中 idx_author_metrics_covering 覆盖索引，极速提取特征
        """
        print(f"\n>>> 任务 1: 生成精排交互数据 (采样规模: {sample_size})")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # 开启高性能读取模式
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA cache_size = -1000000;") # 1GB 缓存
        # 必须执行 ANALYZE 确保命中最新覆盖索引
        conn.execute("ANALYZE;")

        all_jobs = conn.execute("SELECT securityId, job_name, description FROM jobs").fetchall()

        if len(all_jobs) > sample_size:
            jobs = random.sample(all_jobs, sample_size)
        else:
            jobs = all_jobs

        train_lines = []
        test_lines = []

        for job in tqdm(jobs, desc="Generating Training Samples"):
            job_raw_id = job['securityId']
            query_text = f"{job['job_name']} {job['description'] or ''}"

            # 1. 执行静默召回 (底层已优化至 1.1s)
            recall_results = self.recall_system.execute(query_text)
            candidates = recall_results.get('final_top_500', [])

            if len(candidates) < 20:
                continue

            # 2. 极速提取 AX 指标：强制命中覆盖索引 idx_author_metrics_covering
            # 字段顺序必须与索引定义一致：(author_id, h_index, cited_by_count, works_count)
            placeholders = ','.join(['?'] * len(candidates))
            author_stats = conn.execute(
                f"""SELECT author_id, h_index, cited_by_count 
                    FROM authors INDEXED BY idx_author_metrics_covering 
                    WHERE author_id IN ({placeholders})""",
                candidates
            ).fetchall()

            scored_authors = []
            for row in author_stats:
                h_idx = row['h_index'] or 0
                citations = row['cited_by_count'] or 0
                # 改进评分公式：双对数平滑，防止引用极值干扰精排模型学习
                quality_score = np.log1p(h_idx) * np.log1p(citations)
                scored_authors.append((row['author_id'], quality_score))

            scored_authors.sort(key=lambda x: x[1], reverse=True)

            # 3. 构造对比样本 (Top 15 为正样本)
            pos_authors = [str(self.get_int_id(a[0])) for a in scored_authors[:15]]

            if pos_authors:
                job_int_id = self.get_int_id(job_raw_id)
                line = f"{job_int_id} " + " ".join(pos_authors)

                if random.random() > 0.2:
                    train_lines.append(line)
                else:
                    test_lines.append(line)

        # 保存结果
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

        # A. 学术协作关系 (Author - Work - Institution)
        print("[*] 正在解析作者-论文-机构链路...")
        aship_rows = conn.execute("SELECT author_id, work_id, inst_id FROM authorships").fetchall()
        for r in aship_rows:
            # authored: 2
            kg_triplets.append((self.get_int_id(r['author_id']), 2, self.get_int_id(r['work_id'])))
            if r['inst_id']:
                # produced_by: 3
                kg_triplets.append((self.get_int_id(r['work_id']), 3, self.get_int_id(r['inst_id'])))

        # B. 论文语义标签 (Work - Vocabulary)
        print("[*] 正在解析论文语义标签 (has_topic)...")
        work_rows = conn.execute("SELECT work_id, concepts_text FROM works WHERE concepts_text IS NOT NULL").fetchall()
        for row in work_rows:
            h_work = self.get_int_id(row['work_id'])
            for term in row['concepts_text'].split('|'):
                # has_topic: 5
                kg_triplets.append((h_work, 5, self.get_int_id(f"v_{term.strip().lower()}")))

        # C. 岗位技能需求 (Job - Vocabulary)
        print("[*] 正在解析岗位技能要求 (require_skill)...")
        job_rows = conn.execute("SELECT securityId, skills FROM jobs WHERE skills IS NOT NULL").fetchall()
        for row in job_rows:
            h_job = self.get_int_id(row['securityId'])
            # 兼容多种分隔符的技能切分
            skills = re.split(r'[,，;；/ \t\n]', row['skills'])
            for skill in skills:
                if skill.strip():
                    # require_skill: 7
                    kg_triplets.append((h_job, 7, self.get_int_id(f"v_{skill.strip().lower()}")))

        # 保存三元组
        print(f"[*] 正在保存 {len(kg_triplets)} 条三元组边...")
        with open(os.path.join(self.output_dir, "kg_final.txt"), "w", encoding='utf-8') as f:
            for h, r, t in kg_triplets:
                f.write(f"{h} {r} {t}\n")

        # 导出全局映射表
        mapping = {"entity": self.entity_to_int, "relation": self.relation_to_int}
        with open(os.path.join(self.output_dir, "id_map.json"), "w", encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)

        conn.close()
        print(f"\n[成功] 训练数据准备完成！节点总数: {len(self.entity_to_int)}")


if __name__ == "__main__":
    generator = KGATAXTrainingGenerator()
    # 模拟真实推荐场景，采样 1000 个岗位生成精排正样本对
    generator.generate_refined_train_data(sample_size=1000)
    # 生成全局知识图谱拓扑
    generator.generate_kg_topology()