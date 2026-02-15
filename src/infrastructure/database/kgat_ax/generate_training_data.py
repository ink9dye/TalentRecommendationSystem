import os
import re
import gc
import torch
import json
import sqlite3
import numpy as np
import pandas as pd
import random
import logging
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

        # --- 核心优化：预载入学者质量分到内存 ---
        self.author_quality_map = {}
        self._preload_author_quality()

        # --- ID 压缩映射管理 ---
        self.user_to_int = {}  # 原始岗位 ID -> 压缩 User ID (从 0 开始)
        self.entity_to_int = {}  # 原始实体 ID -> 相对压缩偏移量
        self.user_counter = 0
        self.entity_counter = 0
        self.ENTITY_OFFSET = 0  # 动态计算：等于参与任务的所有岗位总数

        self.relation_to_int = {
            "interact": 0, "out_interact": 1, "authored": 2, "produced_by": 3,
            "published_in": 4, "has_topic": 5, "similar_to": 6, "require_skill": 7
        }

        # 质量统计哨兵
        self.stats = {
            "self_loops": 0,
            "total_triplets": 0,
            "user_connectivity": 0.0,
            "train_jobs": 0,
            "test_jobs": 0
        }

    def _preload_author_quality(self):
        """
        从预计算的特征索引加载学者质量分，确保训练正样本筛选与模型特征分布一致。
        不再进行二次 log1p 计算，直接利用索引中的归一化数值。
        """
        import json
        from config import FEATURE_INDEX_PATH  # 确保 config 中定义了此路径

        print("\n[*] 正在从预计算索引载入学者质量分到内存...")

        if not os.path.exists(FEATURE_INDEX_PATH):
            print(f"[!] 警告: 未发现特征索引 {FEATURE_INDEX_PATH}，将回退至数据库查询模式。")
            # 如果索引不存在，则执行原有的 SQL 逻辑（带对数平滑）
            self._preload_author_quality_fallback()
            return

        with open(FEATURE_INDEX_PATH, 'r', encoding='utf-8') as f:
            feature_bundle = json.load(f)

        # 提取作者特征字典
        author_features = feature_bundle.get('author', {})

        for aid, feats in tqdm(author_features.items(), desc="Mapping Quality Scores"):
            # 获取预计算好的对数平滑归一化指标
            h_idx = feats.get('h_index', 0.0)
            citations = feats.get('cited_by_count', 0.0)

            # 核心逻辑修改：采用加法融合或保持乘法
            # 由于是归一化后的 [0, 1] 数值，加法能更好地抑制极值干扰，保证样本均衡
            self.author_quality_map[str(aid)] = h_idx + citations

        print(f"[OK] 成功从索引载入 {len(self.author_quality_map)} 名学者特征。")
        print(f"[*] 质量分分布参考 - 均值: {np.mean(list(self.author_quality_map.values())):.4f}\n")

    def get_user_id(self, raw_id):
        """获取 User ID (0 到 ENTITY_OFFSET - 1)"""
        raw_id = str(raw_id)
        if raw_id not in self.user_to_int:
            self.user_to_int[raw_id] = self.user_counter
            self.user_counter += 1
        return self.user_to_int[raw_id]

    def get_ent_id(self, raw_id):
        """获取 Entity ID (从 ENTITY_OFFSET 开始)"""
        raw_id = str(raw_id)
        if raw_id not in self.entity_to_int:
            self.entity_to_int[raw_id] = self.entity_counter
            self.entity_counter += 1
        # 返回物理 ID = 相对 ID + 偏移量
        return self.entity_to_int[raw_id] + self.ENTITY_OFFSET

    def _process_single_job(self, job, conn):
        """利用内存字典执行极速精排"""
        job_raw_id = str(job['securityId'])
        query_text = f"{job['job_name'] or ''} {job['description'] or ''}"

        # 1. 执行召回系统 (向量+标签+协同)
        recall_results = self.recall_system.execute(query_text)
        candidates = recall_results.get('final_top_500', [])

        if not candidates or len(candidates) < 20:
            return None

        # 2. 从内存字典快速获取分值 (取代原有的 SQL 查询)
        scored_list = []
        for aid in candidates:
            # O(1) 内存寻址，不再受磁盘 I/O 限制
            score = self.author_quality_map.get(str(aid), 0.0)
            scored_list.append((str(aid), score))

        # 3. 根据分值重排并取前 15 名作为正样本
        scored_list.sort(key=lambda x: x[1], reverse=True)

        try:
            pos_authors = [str(self.get_ent_id(a[0])) for a in scored_list[:15]]
            u_id = self.get_user_id(job_raw_id)
            return f"{u_id} " + " ".join(pos_authors)
        except Exception:
            return None

    def generate_refined_train_data(self, train_size=1000, test_size=100):
        """实现 ID 压缩与动态偏移，确保训练集和测试集隔离"""
        print(f"\n>>> 任务 1: 生成精排交互数据 (训练:{train_size}, 测试:{test_size})")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # 1. 抽取所有需要的岗位并预注册 ID
        all_jobs_raw = conn.execute("SELECT securityId, job_name, description FROM jobs").fetchall()

        if len(all_jobs_raw) < (train_size + test_size):
            sampled_jobs = all_jobs_raw
            train_pool = sampled_jobs[:int(len(sampled_jobs) * 0.9)]
            test_pool = sampled_jobs[int(len(sampled_jobs) * 0.9):]
        else:
            sampled_jobs = random.sample(all_jobs_raw, train_size + test_size)
            train_pool = sampled_jobs[:train_size]
            test_pool = sampled_jobs[train_size:]

        # 预先确立 ENTITY_OFFSET 边界
        for job in sampled_jobs:
            self.get_user_id(job['securityId'])

        self.ENTITY_OFFSET = self.user_counter
        print(f"[*] 已注册 {self.ENTITY_OFFSET} 个岗位。实体 ID 将从 {self.ENTITY_OFFSET} 开始映射。")

        train_lines, test_lines = [], []

        # 2. 生成交互行 (利用进程内 conn 进行必要的简单查询)
        for job in tqdm(train_pool, desc="Processing Train Jobs"):
            line = self._process_single_job(job, conn)
            if line: train_lines.append(line)

        for job in tqdm(test_pool, desc="Processing Test Jobs"):
            line = self._process_single_job(job, conn)
            if line: test_lines.append(line)

        self.stats["train_jobs"] = len(train_lines)
        self.stats["test_jobs"] = len(test_lines)

        with open(os.path.join(self.output_dir, "train.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(train_lines))
        with open(os.path.join(self.output_dir, "test.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(test_lines))

        conn.close()

    def generate_kg_topology(self):
        """构建知识图谱拓扑并持久化 ID 映射字典"""
        print("\n>>> 任务 2: 构建知识图谱拓扑 (动态压缩版)")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        kg_triplets = []

        # A. Author - Work - Institution 映射
        aship_rows = conn.execute("SELECT author_id, work_id, inst_id FROM authorships").fetchall()
        for r in tqdm(aship_rows, desc="Mapping Authorships"):
            h = self.get_ent_id(r['author_id'])
            t_w = self.get_ent_id(r['work_id'])
            if h != t_w:
                kg_triplets.append((h, 2, t_w))
            else:
                self.stats["self_loops"] += 1

            if r['inst_id']:
                t_i = self.get_ent_id(r['inst_id'])
                if t_w != t_i: kg_triplets.append((t_w, 3, t_i))

        # B. Work - Topic 映射
        work_rows = conn.execute("SELECT work_id, concepts_text FROM works WHERE concepts_text IS NOT NULL").fetchall()
        for row in tqdm(work_rows, desc="Mapping Topics"):
            h = self.get_ent_id(row['work_id'])
            for term in row['concepts_text'].split('|'):
                clean_term = term.strip().lower()
                if not clean_term: continue
                t = self.get_ent_id(f"v_{clean_term}")
                if h != t: kg_triplets.append((h, 5, t))

        # C. Job - Skill 连接 (跨 User-Entity 空间)
        job_rows = conn.execute("SELECT securityId, skills FROM jobs WHERE skills IS NOT NULL").fetchall()
        for row in job_rows:
            raw_id = str(row['securityId'])
            if raw_id in self.user_to_int:
                h = self.user_to_int[raw_id]
                skills = re.split(r'[,，;；/ \t\n]', row['skills'])
                for skill in skills:
                    clean_skill = skill.strip().lower()
                    if not clean_skill: continue
                    t = self.get_ent_id(f"v_{clean_skill}")
                    kg_triplets.append((h, 7, t))

        self._perform_final_sanity_check(kg_triplets)

        with open(os.path.join(self.output_dir, "kg_final.txt"), "w") as f:
            for h, r, t in kg_triplets: f.write(f"{h} {r} {t}\n")

        # 整合映射表并保存
        full_mapping = {**self.user_to_int}
        for k, v in self.entity_to_int.items():
            full_mapping[k] = v + self.ENTITY_OFFSET

        with open(os.path.join(self.output_dir, "id_map.json"), "w") as f:
            json.dump({
                "entity": full_mapping,
                "user_count": self.user_counter,
                "entity_count": self.entity_counter,
                "offset": self.ENTITY_OFFSET,
                "total_nodes": self.user_counter + self.entity_counter,
                "quality_stats": self.stats
            }, f)
        conn.close()

    def _perform_final_sanity_check(self, triplets):
        """执行数据扫描，处理自环并计算连通率"""
        df = pd.DataFrame(triplets, columns=['h', 'r', 't'])
        actual_loops = (df['h'] == df['t']).sum()
        if actual_loops > 0:
            triplets[:] = [tri for tri in triplets if tri[0] != tri[2]]

        user_nodes_in_kg = set(df[df['r'] == 7]['h'].unique())
        self.stats["user_connectivity"] = len(user_nodes_in_kg) / (self.user_counter + 1e-9)
        self.stats["total_triplets"] = len(triplets)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    print("\n" + "★" * 50)
    print(" KGATAX 训练数据生成流水线 ".center(44))


    try:
        generator = KGATAXTrainingGenerator()
        # 处理 10,000 个训练岗位和 1,000 个测试岗位
        generator.generate_refined_train_data(train_size=10000, test_size=1000)
        generator.generate_kg_topology()

        print(f"【生成报告】")
        print(f"岗位/用户总数: {generator.user_counter}")
        print(f"实体总数: {generator.entity_counter}")
        print(f"ID 偏移量: {generator.ENTITY_OFFSET}")
        print(f"有效三元组: {generator.stats['total_triplets']}")
        print(f"岗位 KG 连通率: {generator.stats['user_connectivity']:.2%}")


    except Exception as e:
        import traceback

        traceback.print_exc()