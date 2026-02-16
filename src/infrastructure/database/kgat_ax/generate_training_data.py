import os
import re
import json
import sqlite3
import numpy as np
import pandas as pd
import random
import logging
from tqdm import tqdm
from config import DB_PATH, KGATAX_TRAIN_DATA_DIR, FEATURE_INDEX_PATH
from src.core.recall.total_recall import TotalRecallSystem


class KGATAXTrainingGenerator:
    def __init__(self):
        self.output_dir = KGATAX_TRAIN_DATA_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.recall_system = TotalRecallSystem()
        self.recall_system.v_path.verbose = False
        self.recall_system.l_path.verbose = False

        # --- 1. 载入 build_feature_index.py 生成的归一化索引 ---
        self.author_quality_map = {}
        self._preload_author_quality_from_index()

        # --- 2. ID 映射管理 ---
        self.user_to_int = {}
        self.entity_to_int = {}
        self.user_counter = 0
        self.entity_counter = 0
        self.ENTITY_OFFSET = 0

    def _preload_author_quality_from_index(self):
        """直接使用已有的归一化特征进行专家加权"""
        if not os.path.exists(FEATURE_INDEX_PATH):
            print(f"[!] 错误: 未发现特征索引，请运行 build_feature_index.py")
            return

        with open(FEATURE_INDEX_PATH, 'r', encoding='utf-8') as f:
            feature_bundle = json.load(f)

        author_features = feature_bundle.get('author', {})
        for aid, feats in author_features.items():
            # 使用归一化后的指标均衡权重
            h_norm = feats.get('h_index', 0.0)
            c_norm = feats.get('cited_by_count', 0.0)
            w_norm = feats.get('works_count', 0.0)
            # 专家权重公式：0.4/0.4/0.2
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
        """
        核心精排逻辑：召回序(50%) + 质量序(50%) 混合重排
        采样梯度：100正、100尚可、100中性、100无关
        """
        job_raw_id = str(job['securityId'])
        # 使用你提供的 jobs 表字段进行文本拼接
        query_text = f"{job['job_name'] or ''} {job['description'] or ''} {job['skills'] or ''}"

        # 1. 召回原始 500 人 (获取召回名次分)
        recall_results = self.recall_system.execute(query_text)
        candidates = recall_results.get('final_top_500', [])
        if len(candidates) < 480: return None

        recall_ranks = {str(aid): i for i, aid in enumerate(candidates)}

        # 2. 计算质量名次分
        quality_list = [(str(aid), self.author_quality_map.get(str(aid), 0.0)) for aid in candidates]
        quality_list.sort(key=lambda x: x[1], reverse=True)
        quality_ranks = {aid: i for i, (aid, _) in enumerate(quality_list)}

        # 3. 混合排名融合 (名次即分值)
        fused_scored = []
        for aid in candidates:
            aid_str = str(aid)
            # 融合分 = 0.5 * 召回序 + 0.5 * 质量序
            fused_rank = 0.5 * recall_ranks[aid_str] + 0.5 * quality_ranks[aid_str]
            fused_scored.append((aid_str, fused_rank))

        fused_scored.sort(key=lambda x: x[1])

        try:
            # --- 四级梯度阶梯抽样 ---
            # A. 正面 (1-100名)
            pos_ids = [str(self.get_ent_id(a[0])) for a in fused_scored[:100]]
            # B. 尚可 (101-400名抽样100)
            fair_pool = fused_scored[100:400]
            sampled_fair = random.sample(fair_pool, 100)
            fair_ids = [str(self.get_ent_id(a[0])) for a in sampled_fair]
            # C. 中性/硬负 (401-500名)
            neutral_pool = fused_scored[400:500]
            neutral_ids = [str(self.get_ent_id(a[0])) for a in neutral_pool]
            # D. 纯粹负面 (全局随机100)
            cand_set = set(str(aid) for aid in candidates)
            all_known = list(self.author_quality_map.keys())
            potential_negs = [aid for aid in all_known if aid not in cand_set]
            easy_neg_raw = random.sample(potential_negs, 100) if len(potential_negs) >= 100 else []
            easy_neg_ids = [str(self.get_ent_id(aid)) for aid in easy_neg_raw]

            u_id = self.get_user_id(job_raw_id)
            # 最终存储格式：User;Pos;Fair;Neutral;EasyNeg
            return f"{u_id};{','.join(pos_ids)};{','.join(fair_ids)};{','.join(neutral_ids)};{','.join(easy_neg_ids)}"
        except Exception:
            return None

    def generate_refined_train_data(self, train_size=1000, test_size=100):
        """核心修复：从 jobs 表读取全字段，并确立 ID 边界"""
        print(f"\n>>> 任务: 生成混合排名精排数据 (混合重排模式)...")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        # 修正：读取所有必要字段 [cite: 3]
        all_jobs = conn.execute("SELECT securityId, job_name, description, skills FROM jobs").fetchall()

        sampled = random.sample(all_jobs, min(len(all_jobs), train_size + test_size))
        for j in sampled: self.get_user_id(j['securityId'])
        self.ENTITY_OFFSET = self.user_counter

        train_lines, test_lines = [], []
        train_pool = sampled[:train_size]
        test_pool = sampled[train_size:]

        for job in tqdm(train_pool, desc="Train Jobs"):
            line = self._process_single_job(job)
            if line: train_lines.append(line)
        for job in tqdm(test_pool, desc="Test Jobs"):
            line = self._process_single_job(job)
            if line: test_lines.append(line)

        with open(os.path.join(self.output_dir, "train.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(train_lines))
        with open(os.path.join(self.output_dir, "test.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(test_lines))
        conn.close()
        print(f"[OK] 训练集与测试集生成完毕。")


if __name__ == "__main__":
    gen = KGATAXTrainingGenerator()
    # 按照你的要求：1000训练，100测试
    gen.generate_refined_train_data(train_size=1000, test_size=100)