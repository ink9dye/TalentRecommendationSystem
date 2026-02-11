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
        从 SQLite 导出三元组并转换为整数 ID (kg_final.txt)。
        完整补全了：
        1. 拓扑路: Author-Work, Work-Institution, Work-Source
        2. 语义路: Work-Vocabulary (HAS_TOPIC)
        3. 需求路: Job-Vocabulary (REQUIRE_SKILL)
        4. 关联路: Vocabulary-Vocabulary (SIMILAR_TO)
        """
        import re
        print("\n>>> 任务 2: 生成知识图谱三元组数据")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        kg_lines = []

        # 定义关系 ID 映射，严格对齐 KG 构建逻辑
        rel_map = {
            "authored": 2,  # Author -> Work
            "produced_by": 3,  # Work -> Institution
            "published_in": 4,  # Work -> Source
            "has_topic": 5,  # Work -> Vocabulary
            "similar_to": 6,  # Vocabulary -> Vocabulary
            "require_skill": 7  # Job -> Vocabulary (补全岗位需求链路)
        }
        self.relation_to_int.update(rel_map)

        # 1. 基础拓扑：解析作者、作品、机构、渠道间的关系
        print("[*] 正在解析作者、作品、机构、渠道间的拓扑关系...")
        rows = cursor.execute("""
                              SELECT author_id, work_id, inst_id, source_id
                              FROM authorships
                              """).fetchall()

        for r in rows:
            h_auth = self.get_int_id(r['author_id'])
            t_work = self.get_int_id(r['work_id'])
            # (Author)-[authored]->(Work)
            kg_lines.append(f"{h_auth} {rel_map['authored']} {t_work}")

            # (Work)-[produced_by]->(Institution)
            if r['inst_id']:
                t_inst = self.get_int_id(r['inst_id'])
                kg_lines.append(f"{t_work} {rel_map['produced_by']} {t_inst}")

            # (Work)-[published_in]->(Source)
            if r['source_id']:
                t_src = self.get_int_id(r['source_id'])
                kg_lines.append(f"{t_work} {rel_map['published_in']} {t_src}")

        # 2. 语义桥接：解析论文与词汇关联 (Work)-[has_topic]->(Vocabulary)
        print("[*] 正在解析作品关键词与语义词汇关联...")
        work_rows = cursor.execute("SELECT work_id, concepts_text, keywords_text FROM works").fetchall()
        for row in work_rows:
            # 采用与 build_work_semantic_links 一致的正则拆分逻辑
            raw_text = f"{row['concepts_text'] or ''}|{row['keywords_text'] or ''}"
            terms = set([t.strip().lower() for t in re.split(r'[|;,]', raw_text) if t.strip()])

            h_work = self.get_int_id(row['work_id'])
            for term in terms:
                t_vocab = self.get_int_id(f"vocab_{term}")
                kg_lines.append(f"{h_work} {rel_map['has_topic']} {t_vocab}")

        # 3. 需求对齐：解析岗位与技能关联 (Job)-[require_skill]->(Vocabulary)
        print("[*] 正在解析岗位需求与技能词汇关联...")
        # 对应 SYNC_JOB_SKILLS 逻辑
        job_rows = cursor.execute("SELECT securityId, skills FROM jobs WHERE skills IS NOT NULL").fetchall()
        for row in job_rows:
            # 采用与 build_job_skill_links 一致的正则拆分逻辑
            skills = set([s.strip().lower() for s in re.split(r'[,，;；]', row['skills']) if s.strip()])

            h_job = self.get_int_id(row['securityId'])
            for skill in skills:
                t_vocab = self.get_int_id(f"vocab_{skill}")
                kg_lines.append(f"{h_job} {rel_map['require_skill']} {t_vocab}")

        # 4. 词汇增强：解析词汇相似度关联 (Vocabulary)-[similar_to]->(Vocabulary)
        print("[*] 正在解析词汇间的语义相似度关系...")
        try:
            sim_rows = cursor.execute("SELECT from_id, to_id FROM vocabulary_similarity").fetchall()
            for r in sim_rows:
                h_v = self.get_int_id(f"vocab_id_{r['from_id']}")
                t_v = self.get_int_id(f"vocab_id_{r['to_id']}")
                kg_lines.append(f"{h_v} {rel_map['similar_to']} {t_v}")
        except sqlite3.OperationalError:
            print("提示: SQLite 中未发现相似度关联表，将跳过相似度三元组导出。")

        # 写入训练用的 kg_final.txt
        with open(os.path.join(self.output_dir, "kg_final.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(kg_lines))

        conn.close()
        print(f"[成功] kg_final.txt 已生成，共包含 {len(kg_lines)} 条结构化三元组。")

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