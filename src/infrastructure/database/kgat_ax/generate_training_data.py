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
        self.ENTITY_OFFSET = 1000000  # 预留足够的空间给 User，或在最后动态合并

        self.relation_to_int = {
            "interact": 0, "out_interact": 1, "authored": 2, "produced_by": 3,
            "published_in": 4, "has_topic": 5, "similar_to": 6, "require_skill": 7
        }

    def validate_database_readiness(self, conn):
        """生成前校验：检查数据库关键字段"""
        print("[Check] 正在校验数据库完整性...")

        # 1. 检查 H-index 覆盖率
        total_authors = conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
        null_h = conn.execute("SELECT COUNT(*) FROM authors WHERE h_index IS NULL").fetchone()[0]
        if null_h / total_authors > 0.3:
            print(f"警告: {null_h}/{total_authors} 的作者缺少 H-index。")

        # 2. 检查 Job 描述
        empty_jobs = conn.execute("SELECT COUNT(*) FROM jobs WHERE description IS NULL OR description = ''").fetchone()[
            0]
        if empty_jobs > 0:
            print(f"警告: 有 {empty_jobs} 个岗位缺少描述，召回系统可能无法正常工作。")

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
        return self.entity_counter - 1  # 返回当前实体的相对索引

    def generate_refined_train_data(self, sample_size=1000):
        """
        利用岗位描述生成精排训练样本，增加工业级健壮性检测
        核心改进：前置校验、索引强制检查、ID 合规性验证
        """
        print(f"\n>>> 任务 1: 生成精排交互数据 (目标采样: {sample_size})")

        # 连接数据库并进行前置准备
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("ANALYZE;")

        # --- 1. 前置数据库健康校验 ---
        print("[Check] 正在执行数据库准备状态校验...")
        total_authors = conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
        if total_authors == 0:
            raise ValueError("错误: 数据库中没有作者数据，请先运行爬虫/导入程序。")

        null_h = conn.execute("SELECT COUNT(*) FROM authors WHERE h_index IS NULL").fetchone()[0]
        if null_h / total_authors > 0.5:
            print(f"警告: 超过 50% ({null_h}/{total_authors}) 的作者缺失 H-index，质量评分可能失效。")

        # --- 2. 岗位样本准备 ---
        all_jobs = conn.execute("SELECT securityId, job_name, description FROM jobs").fetchall()
        if not all_jobs:
            raise ValueError("错误: 数据库 jobs 表为空。")

        jobs = random.sample(all_jobs, sample_size) if len(all_jobs) > sample_size else all_jobs

        train_lines, test_lines = [], []
        processed_success = 0

        # --- 3. 循环生成与模拟召回 ---
        for job in tqdm(jobs, desc="Generating Training Samples"):
            # 强制类型转换并处理空值
            job_raw_id = str(job['securityId'])
            job_name = str(job['job_name'] or "")
            job_desc = str(job['description'] or "")
            query_text = f"{job_name} {job_desc}"

            # 执行召回系统（三路召回模拟生产环境）
            recall_results = self.recall_system.execute(query_text)
            candidates = recall_results.get('final_top_500', [])

            # 数量检测：候选人太少不足以构造精排对
            if not candidates or len(candidates) < 20:
                continue

            # 批量获取 AX 学术特征：强制使用覆盖索引优化 I/O
            placeholders = ','.join(['?'] * len(candidates))
            try:
                # 严格对应索引顺序：(author_id, h_index, cited_by_count)
                author_stats = conn.execute(
                    f"""SELECT author_id, h_index, cited_by_count 
                        FROM authors INDEXED BY idx_author_metrics_covering 
                        WHERE author_id IN ({placeholders})""",
                    candidates
                ).fetchall()
            except sqlite3.Error as e:
                print(f"SQL 查询失败 (可能是索引未对齐): {e}")
                continue

            # --- 4. 质量评分与正样本构建 (基于专家经验公式) ---
            scored_list = []
            for r in author_stats:
                # 双对数平滑，防止引用极值干扰梯度
                h_idx = float(r['h_index'] or 0)
                citations = float(r['cited_by_count'] or 0)
                quality_score = np.log1p(h_idx) * np.log1p(citations)
                scored_list.append((str(r['author_id']), quality_score))

            # 按分数倒序排列
            scored_list.sort(key=lambda x: x[1], reverse=True)

            # 转换为全局图谱空间 ID (User ID 在 [0, OFFSET), Entity ID 在 [OFFSET, ...))
            try:
                pos_authors = [str(self.get_ent_id(a[0]) + self.ENTITY_OFFSET) for a in scored_list[:15]]
                if pos_authors:
                    u_id = self.get_user_id(job_raw_id)
                    # 校验：确保 User ID 确实小于 OFFSET
                    if u_id >= self.ENTITY_OFFSET:
                        raise IndexError(f"User ID {u_id} 溢出至实体空间，请增大 ENTITY_OFFSET。")

                    line = f"{u_id} " + " ".join(pos_authors)

                    # 训练/测试集 8:2 随机划分
                    if random.random() > 0.2:
                        train_lines.append(line)
                    else:
                        test_lines.append(line)
                    processed_success += 1
            except Exception as e:
                print(f"ID 转换异常: {e}")
                continue

        # --- 5. 结果持久化与统计校验 ---
        with open(os.path.join(self.output_dir, "train.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(train_lines))
        with open(os.path.join(self.output_dir, "test.txt"), "w", encoding='utf-8') as f:
            f.write("\n".join(test_lines))

        conn.close()

        print(f"\n[OK] 任务 1 完成统计:")
        print(f" - 成功生成的岗位样本: {processed_success}")
        print(f" - 训练集行数: {len(train_lines)}")
        print(f" - 测试集行数: {len(test_lines)}")
        if processed_success < (sample_size * 0.5):
            print("警告: 成功率低于 50%，请检查召回系统匹配质量或数据库数据。")

    def generate_kg_topology(self):
        print("\n>>> 任务 2: 构建知识图谱拓扑")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        kg_triplets = []

        # A. Author - Work - Institution
        aship_rows = conn.execute("SELECT author_id, work_id, inst_id FROM authorships").fetchall()
        for r in aship_rows:
            h = self.get_ent_id(r['author_id']) + self.ENTITY_OFFSET
            t_w = self.get_ent_id(r['work_id']) + self.ENTITY_OFFSET
            kg_triplets.append((h, 2, t_w))
            if r['inst_id']:
                t_i = self.get_ent_id(r['inst_id']) + self.ENTITY_OFFSET
                kg_triplets.append((t_w, 3, t_i))

        # B. Work - Topic
        work_rows = conn.execute("SELECT work_id, concepts_text FROM works WHERE concepts_text IS NOT NULL").fetchall()
        for row in work_rows:
            h = self.get_ent_id(row['work_id']) + self.ENTITY_OFFSET
            for term in row['concepts_text'].split('|'):
                t = self.get_ent_id(f"v_{term.strip().lower()}") + self.ENTITY_OFFSET
                kg_triplets.append((h, 5, t))

        # C. Job - Skill (User - Entity 连接)
        job_rows = conn.execute("SELECT securityId, skills FROM jobs WHERE skills IS NOT NULL").fetchall()
        for row in job_rows:
            h = self.get_user_id(row['securityId'])
            skills = re.split(r'[,，;；/ \t\n]', row['skills'])
            for skill in skills:
                if skill.strip():
                    t = self.get_ent_id(f"v_{skill.strip().lower()}") + self.ENTITY_OFFSET
                    kg_triplets.append((h, 7, t))

        with open(os.path.join(self.output_dir, "kg_final.txt"), "w") as f:
            for h, r, t in kg_triplets: f.write(f"{h} {r} {t}\n")

        # 导出最终映射：合并 user 和 entity 字典
        full_mapping = {**self.user_to_int}
        for k, v in self.entity_to_int.items():
            full_mapping[k] = v + self.ENTITY_OFFSET

        with open(os.path.join(self.output_dir, "id_map.json"), "w") as f:
            json.dump({"entity": full_mapping, "user_count": self.user_counter, "offset": self.ENTITY_OFFSET}, f)
        conn.close()


if __name__ == "__main__":
    import time
    import logging
    import warnings

    # 1. 环境静默化配置：防止内层模型进度条刷屏
    # 关闭 SentenceTransformer 的 tqdm 进度条
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 屏蔽不必要的警告
    warnings.filterwarnings("ignore")

    # 2. 日志系统配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(KGATAX_TRAIN_DATA_DIR, "generator.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    print("\n" + "★" * 50)
    print(" KGATAX 训练数据生成流水线 (工业增强版) ".center(44))
    print("★" * 50)

    total_start = time.time()

    try:
        # 3. 前置预检：检测核心资源是否到位
        print(f"[*] 执行前置预检中...")
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"数据库文件缺失: {DB_PATH}")

        # 4. 初始化生成器 (加载模型与索引)
        print(f"[*] 正在初始化全量召回系统 (预加载模型中，请稍候)...")
        generator = KGATAXTrainingGenerator()

        # 5. 执行数据校验：检测 ID 分区空间是否足够
        # 预估 Job 数量如果接近 ENTITY_OFFSET 则报错
        if generator.user_counter >= generator.ENTITY_OFFSET:
            raise OverflowError(
                f"Job 数量已接近 OFFSET ({generator.ENTITY_OFFSET})，请调大 config.py 中的 ENTITY_OFFSET")

        # 6. 任务 1: 生成精排正负样本
        t1_start = time.time()
        # 推荐：先用 50 条做跑通测试，正式生产用 1000+
        generator.generate_refined_train_data(sample_size=1000)
        t1_end = time.time()
        print(f"[OK] 任务 1 (精排数据) 生成完毕，耗时: {t1_end - t1_start:.2f}s")

        # 7. 任务 2: 构建知识图谱拓扑
        t2_start = time.time()
        generator.generate_kg_topology()
        t2_end = time.time()
        print(f"[OK] 任务 2 (KG 拓扑) 生成完毕，耗时: {t2_end - t2_start:.2f}s")

        # 8. 最终输出统计与类型自检
        total_duration = time.time() - total_start
        print("\n" + "★" * 50)
        print(f"【生成报告】")
        print(f"● 总执行耗时: {total_duration:.2f} 秒")
        print(f"● 节点总数 (Entities): {generator.entity_counter}")
        print(f"● 用户总数 (Users/Jobs): {generator.user_counter}")
        print(f"● 输出目录: {generator.output_dir}")

        # 文件完整性检测
        required_files = ["train.txt", "test.txt", "kg_final.txt", "id_map.json"]
        missing = [f for f in required_files if not os.path.exists(os.path.join(generator.output_dir, f))]

        if not missing:
            print(f"✔ 所有必要文件已成功生成并校验完毕。")
        else:
            print(f"✘ 警告：部分文件生成失败: {missing}")
        print("★" * 50 + "\n")

    except KeyboardInterrupt:
        print("\n[!] 用户中断了生成流程。")
    except Exception as e:
        logger.error(f"流水线执行崩溃: {str(e)}", exc_info=True)
        print(f"\n[Error] 详细错误已记录至 generator.log")
    finally:
        # 强制垃圾回收
        gc.collect()