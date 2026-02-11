import os
import re
import json
import sqlite3
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

# 导入你的统一配置
from config import (
    DB_PATH, INDEX_DIR,
    JOB_INDEX_PATH, JOB_MAP_PATH,
    ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH,KGATAX_TRAIN_DATA_DIR
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

        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_raw_ids = json.load(f)
        with open(ABSTRACT_MAP_PATH, 'r', encoding='utf-8') as f:
            self.abs_work_ids = json.load(f)

        # 2. 建立全局 ID 映射表 (模型只认 0, 1, 2...)
        self.entity_to_int = {}
        self.relation_to_int = {"interact": 0, "out_interact": 1}  # 预留交互关系 [cite: 413]
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
        # 这里的映射逻辑决定了 CF 交互的目标
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
        # 向量路初步筛选，作为模型训练的“正样本”参考
        D, I = self.abs_index.search(self.job_index.reconstruct_n(0, len(self.job_raw_ids)), 20)

        train_lines = []
        for i, neighbors in enumerate(tqdm(I, desc="Mapping Jobs to Authors")):
            job_raw_id = self.job_raw_ids[i]
            job_int_id = self.get_int_id(job_raw_id)

            # 找到对应的作者 ID
            matched_authors = []
            for work_idx in neighbors:
                work_raw_id = self.abs_work_ids[work_idx]
                author_raw_id = work_to_author.get(work_raw_id)
                if author_raw_id:
                    matched_authors.append(str(self.get_int_id(author_raw_id)))

            if matched_authors:
                # 格式: User_ID Item_ID1 Item_ID2...
                train_lines.append(f"{job_int_id} " + " ".join(matched_authors))

        with open(os.path.join(self.output_dir, "train.txt"), "w") as f:
            f.write("\n".join(train_lines))

        # 简单处理 test.txt：取最后 10% 作为测试集
        with open(os.path.join(self.output_dir, "test.txt"), "w") as f:
            f.write("\n".join(train_lines[-int(len(train_lines) * 0.1):]))

        conn.close()

    def generate_kg_data(self):
        """
        核心优化版：生成千万级知识图谱三元组数据
        """
        print("\n>>> 任务 2: 生成知识图谱三元组数据 (二进制优化版)")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 1. 关系映射定义
        rel_map = {
            "authored": 2, "produced_by": 3,
            "published_in": 4, "has_topic": 5,
            "similar_to": 6, "require_skill": 7
        }
        self.relation_to_int.update(rel_map)

        # 使用 numpy 数组的高效存储
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

        # --- 4. 关联路 ---
        print("[*] 解析词汇增强 (Vocab-Vocab)...")
        try:
            sim_rows = cursor.execute("SELECT from_id, to_id FROM vocabulary_similarity").fetchall()
            for r in sim_rows:
                kg_triplets.append((self.get_int_id(f"vocab_id_{r['from_id']}"),
                                    rel_map['similar_to'],
                                    self.get_int_id(f"vocab_id_{r['to_id']}")))
        except sqlite3.OperationalError:
            print("提示: 未发现词汇相似度表。")

        # --- 关键：二进制转换与存储 ---
        print(f"[*] 正在转换 {len(kg_triplets)} 条边为二进制张量...")
        kg_np = np.array(kg_triplets, dtype=np.int32)

        # 释放内存
        del kg_triplets
        gc.collect()

        # 保存 .pt 文件供 DataLoaderKGAT 加载
        torch.save(torch.from_numpy(kg_np), os.path.join(self.output_dir, "kg_final.pt"))

        # 兼容性空文件
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