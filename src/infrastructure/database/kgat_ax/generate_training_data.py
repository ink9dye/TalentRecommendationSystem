import os
import re
import gc
import torch
import json
import sqlite3
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

# 导入统一配置
from config import (
    DB_PATH, INDEX_DIR,
    JOB_INDEX_PATH, JOB_MAP_PATH,
    ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH,
    VOCAB_INDEX_PATH, VOCAB_MAP_PATH,  # 新增：词汇索引路径
    KGATAX_TRAIN_DATA_DIR
)


class TrainingDataGenerator:
    def __init__(self):
        self.output_dir = KGATAX_TRAIN_DATA_DIR
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 1. 加载映射和索引
        print("[*] 正在加载向量索引...")
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        self.abs_index = faiss.read_index(ABSTRACT_INDEX_PATH)
        self.vocab_index = faiss.read_index(VOCAB_INDEX_PATH)  # 加载词汇索引

        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_raw_ids = json.load(f)
        with open(ABSTRACT_MAP_PATH, 'r', encoding='utf-8') as f:
            self.abs_work_ids = json.load(f)
        with open(VOCAB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.vocab_raw_ids = json.load(f)  # 加载词汇映射

        # 2. 建立全局 ID 映射表
        self.entity_to_int = {}
        self.relation_to_int = {"interact": 0, "out_interact": 1}
        self.int_counter = 0

    def get_int_id(self, raw_id):
        if raw_id not in self.entity_to_int:
            self.entity_to_int[raw_id] = self.int_counter
            self.int_counter += 1
        return self.entity_to_int[raw_id]

    def _map_work_to_author(self, conn):
        """建立 Work_ID 到 Author_ID 的映射"""
        print("[*] 正在建立作品-作者映射...")
        cursor = conn.cursor()
        rows = cursor.execute("SELECT work_id, author_id FROM authorships").fetchall()
        mapping = {}
        for r in rows:
            mapping[r[0]] = r[1]
        return mapping

    def generate_cf_data(self):
        """利用向量相似度生成 CF 交互数据 (train.txt)"""
        print("\n>>> 任务 1: 生成协同过滤交互数据 (Job-Author)")
        conn = sqlite3.connect(DB_PATH)
        work_to_author = self._map_work_to_author(conn)

        # 检索每个 Job 最相似的 20 篇论文摘要
        D, I = self.abs_index.search(self.job_index.reconstruct_n(0, len(self.job_raw_ids)), 20)

        train_lines = []
        for i, neighbors in enumerate(tqdm(I, desc="Mapping Jobs to Authors")):
            job_raw_id = self.job_raw_ids[i]
            job_int_id = self.get_int_id(job_raw_id)

            matched_authors = []
            for work_idx in neighbors:
                work_raw_id = self.abs_work_ids[work_idx]
                author_raw_id = work_to_author.get(work_raw_id)
                if author_raw_id:
                    matched_authors.append(str(self.get_int_id(author_raw_id)))

            if matched_authors:
                train_lines.append(f"{job_int_id} " + " ".join(matched_authors))

        with open(os.path.join(self.output_dir, "train.txt"), "w") as f:
            f.write("\n".join(train_lines))

        with open(os.path.join(self.output_dir, "test.txt"), "w") as f:
            f.write("\n".join(train_lines[-int(len(train_lines) * 0.1):]))

        conn.close()

    def generate_kg_data(self):
        """生成知识图谱三元组数据 (包含动态词汇相似度)"""
        print("\n>>> 任务 2: 生成知识图谱三元组数据")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        rel_map = {
            "authored": 2, "produced_by": 3,
            "published_in": 4, "has_topic": 5,
            "similar_to": 6, "require_skill": 7
        }
        self.relation_to_int.update(rel_map)
        kg_triplets = []

        # --- 1. 拓扑路 ---
        print("[*] 解析拓扑关系 (Author/Work/Inst/Source)...")
        rows = cursor.execute("SELECT author_id, work_id, inst_id, source_id FROM authorships").fetchall()
        for r in rows:
            h_auth = self.get_int_id(r['author_id'])
            t_work = self.get_int_id(r['work_id'])
            kg_triplets.append((h_auth, rel_map['authored'], t_work))
            if r['inst_id']:
                kg_triplets.append((t_work, rel_map['produced_by'], self.get_int_id(r['inst_id'])))
            if r['source_id']:
                kg_triplets.append((t_work, rel_map['published_in'], self.get_int_id(r['source_id'])))

        # --- 2. 语义路 ---
        print("[*] 解析语义桥接 (Work-Vocabulary)...")
        work_rows = cursor.execute("SELECT work_id, concepts_text, keywords_text FROM works").fetchall()
        for row in work_rows:
            raw_text = f"{row['concepts_text'] or ''}|{row['keywords_text'] or ''}"
            terms = set([t.strip().lower() for t in re.split(r'[|;,]', raw_text) if t.strip()])
            h_work = self.get_int_id(row['work_id'])
            for term in terms:
                kg_triplets.append((h_work, rel_map['has_topic'], self.get_int_id(f"vocab_{term}")))

        # --- 3. 需求路 ---
        print("[*] 解析需求对齐 (Job-Vocabulary)...")
        job_rows = cursor.execute("SELECT securityId, skills FROM jobs WHERE skills IS NOT NULL").fetchall()
        for row in job_rows:
            skills = set([s.strip().lower() for s in re.split(r'[,，;；]', row['skills']) if s.strip()])
            h_job = self.get_int_id(row['securityId'])
            for skill in skills:
                kg_triplets.append((h_job, rel_map['require_skill'], self.get_int_id(f"vocab_{skill}")))

        # --- 4. 关联路：利用 vocabulary.faiss 动态检索替代数据库查询 ---
        print("[*] 正在利用 Faiss 动态生成词汇关联 (Vocab-Vocab)...")
        # 提取所有词向量并检索相似词 (取 Top 6，排除第一个自己)
        k_neighbors = 6
        vocab_vectors = self.vocab_index.reconstruct_n(0, len(self.vocab_raw_ids))
        D, I = self.vocab_index.search(vocab_vectors, k_neighbors)

        for i, neighbors in enumerate(tqdm(I, desc="Calculating Vocab Similarities")):
            h_vocab_raw = self.vocab_raw_ids[i]
            h_id = self.get_int_id(f"vocab_{h_vocab_raw}")  # 注意：保持 ID 命名规则一致

            for nb_idx in neighbors:
                # 跳过自己 (索引 i) 或无效结果 (-1)
                if nb_idx == i or nb_idx == -1:
                    continue

                t_vocab_raw = self.vocab_raw_ids[nb_idx]
                t_id = self.get_int_id(f"vocab_{t_vocab_raw}")
                kg_triplets.append((h_id, rel_map['similar_to'], t_id))

        # --- 关键：二进制转换与存储 ---
        print(f"[*] 正在转换 {len(kg_triplets)} 条边为二进制张量...")
        kg_np = np.array(kg_triplets, dtype=np.int32)

        del kg_triplets
        gc.collect()

        torch.save(torch.from_numpy(kg_np), os.path.join(self.output_dir, "kg_final.pt"))
        with open(os.path.join(self.output_dir, "kg_final.txt"), "w") as f:
            f.write("Binary format saved in kg_final.pt")

        conn.close()
        print(f"[成功] 二进制三元组已保存。实体总数: {len(self.entity_to_int)}")

    def save_id_mapping(self):
        """保存 ID 映射表供推理阶段使用"""
        mapping = {"entity": self.entity_to_int, "relation": self.relation_to_int}
        with open(os.path.join(self.output_dir, "id_map.json"), "w", encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)
        print(f"\n[成功] 数据准备完成！实体总数: {len(self.entity_to_int)}")


if __name__ == "__main__":
    generator = TrainingDataGenerator()
    generator.generate_cf_data()
    generator.generate_kg_data()
    generator.save_id_mapping()