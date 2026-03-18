import torch
import os
import glob
import sqlite3
import re
import numpy as np

# 直接从你的 config 中引入所有配置项
from config import (
    KGATAX_TRAIN_DATA_DIR,
    DB_PATH,
    DOMAIN_MAP,
    NAME_TO_DOMAIN_ID
)

from src.infrastructure.database.kgat_ax.model import KGAT
from src.infrastructure.database.kgat_ax.data_loader import DataLoaderKGAT
from src.infrastructure.database.kgat_ax.kgat_parser.parser_kgat import parse_kgat_args
from src.core.recall.total_recall import TotalRecallSystem
from src.core.ranking.ranking_engine import RankingEngine


class TotalCore:
    def __init__(self):
        """系统初始化：负责核心组件加载与环境对齐"""
        print("[*] 正在初始化全量人才推荐核心引擎...", flush=True)
        self.args = parse_kgat_args()
        self.device = torch.device("cpu")  # 精排推荐使用CPU以确保稳定性

        # 1. 初始化数据环境
        self._init_data_env()

        # 2. 初始化核心模型
        self.model = self._load_trained_model()

        # 3. 挂载子系统 (召回与精排)
        self.recall_subsystem = TotalRecallSystem()
        self.ranking_engine = RankingEngine(self.model, self.dataloader)

        print(f"[OK] 引擎初始化完成。节点偏移量: {self.args.ENTITY_OFFSET}")

    def _init_data_env(self):
        """数据路径与参数对齐：直接读取 config 路径"""
        self.args.data_dir = os.path.dirname(KGATAX_TRAIN_DATA_DIR)
        self.args.data_name = os.path.basename(KGATAX_TRAIN_DATA_DIR)

        self.dataloader = DataLoaderKGAT(self.args, self._get_silent_logger())
        self.args.ENTITY_OFFSET = self.dataloader.ENTITY_OFFSET
        self.args.n_aux_features = getattr(self.args, 'n_aux_features', 3)

    def _load_trained_model(self):
        """加载训练好的最优权重"""
        model = KGAT(
            self.args, self.dataloader.n_users,
            self.dataloader.n_users_entities, self.dataloader.n_relations,
            self.dataloader.A_in
        ).to(self.device)

        # 基于训练目录寻找权重
        weight_path = os.path.join(KGATAX_TRAIN_DATA_DIR, 'weights')
        best_files = glob.glob(os.path.join(weight_path, "best_model_epoch_*.pth"))

        if best_files:
            # 选取 Epoch 最大的模型作为最优模型
            best_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]), reverse=True)
            target = best_files[0]
            ckpt = torch.load(target, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"[OK] 载入权重: {os.path.basename(target)}")

        return model

    def _get_union_domain_regex(self, job_ids: list):
        """获取锚点岗位的领域并集作为正则过滤条件"""
        try:
            conn = sqlite3.connect(DB_PATH)
            placeholders = ','.join(['?'] * len(job_ids))
            sql = f"SELECT domain_ids FROM jobs WHERE securityId IN ({placeholders})"
            rows = conn.execute(sql, job_ids).fetchall()
            conn.close()

            domain_set = set()
            for row in rows:
                if row[0]:
                    # 鲁棒解析：处理 1|2 或 1,2 格式
                    parts = re.split(r'[,|]', str(row[0]))
                    domain_set.update([p.strip() for p in parts if p.strip()])

            return "|".join(sorted(list(domain_set))) if domain_set else None
        except Exception as e:
            print(f"[Warning] 精排阶段领域并集提取失败: {e}")
            return None

    def suggest(self, text_query: str, manual_domain_id: str = None):
        """
        核心工作流：语义导航 -> 多路召回 -> 领域硬过滤 -> 1:1 深度精排
        """
        # 0. 规范化输入领域 (支持中文名或 ID 转换)
        target_domain = NAME_TO_DOMAIN_ID.get(manual_domain_id, manual_domain_id)
        if target_domain in ["0", "", "None", None]:
            target_domain = None

        # --- Step 1: 语义导航 (寻找 3 个最贴近需求的岗位锚点) ---
        query_vec, _ = self.recall_subsystem.encoder.encode(text_query)
        _, indices = self.recall_subsystem.v_path.job_index.search(query_vec, 3)
        real_job_ids = [self.recall_subsystem.v_path.job_id_map[idx] for idx in indices[0]]

        if not real_job_ids:
            print("[Error] 语义导航未能锁定任何岗位锚点。")
            return []

        print(f"[*] 语义导航完成，锁定精排锚点: {real_job_ids}")

        # --- Step 2: 多路召回 (向量+标签+协同)，得到候选池与名单 ---
        recall_res = self.recall_subsystem.execute(text_query, domain_id=target_domain)
        candidate_pool = recall_res.get("candidate_pool")
        candidates = recall_res.get("final_top_500", [])
        if candidate_pool and getattr(candidate_pool, "candidate_records", None):
            candidates = [r.author_id for r in candidate_pool.candidate_records]

        if not candidates:
            print("[Warning] 召回阶段未获得候选人。")
            return []

        # --- Step 3: 确定精排过滤策略 ---
        # 逻辑：手动输入则强制执行手动领域，否则自动提取 3 个锚点的并集
        filter_pattern = target_domain if target_domain else self._get_union_domain_regex(real_job_ids)

        mode_label = f"手动指定: {target_domain}" if target_domain else f"自动并集: {filter_pattern}"
        print(f"[*] 精排阶段启动 (模式: {mode_label})")

        # --- Step 4: 三阶段精排（预排序 + KGAT-AX + 稳定融合），传入候选池以启用 rule_stability 与四段式证据 ---
        final_results = self.ranking_engine.execute_rank(
            real_job_ids,
            candidates,
            filter_domain=filter_pattern,
            candidate_pool=candidate_pool,
        )

        return final_results

    def _get_silent_logger(self):
        class S:
            def info(self, m): pass
            def error(self, m): print(f"Error: {m}")
        return S()


if __name__ == "__main__":
    # 交互式测试入口
    core = TotalCore()

    print("\n" + "=" * 115)
    print("🚀 人才推荐核心系统 (Core Engine) - 交互式控制台")
    print("-" * 115)
    f_list = list(DOMAIN_MAP.items())
    for i in range(0, len(f_list), 6):
        print(" | ".join([f"{k}:{v}" for k, v in f_list[i:i + 6]]))
    print("=" * 115)

    try:
        # 1. 领域锁定
        domain_choice = input("\n请选择领域编号 (1-17, 0或回车开启自动并集模式): ").strip()
        current_field = DOMAIN_MAP.get(domain_choice, "全领域 (自动并集模式)")
        print(f"[*] 当前锁定领域：{current_field}")

        while True:
            # 2. 需求输入
            user_input = input(f"\n[{current_field}] 请输入岗位需求或技术描述 (q退出): ").strip()
            if not user_input or user_input.lower() == 'q':
                break

            # 3. 执行全流程
            results = core.suggest(user_input, manual_domain_id=domain_choice)

            # 4. 格式化输出 Top 10 结果
            print("\n" + "=" * 80)
            print(f"📊 精排推荐结果列表 (Top 10)")
            print("=" * 80)

            for item in results[:10]:
                print(f"Rank {item['rank']:<2} | {item['name']} (ID: {item['author_id']})")
                print(f" > 综合得分: {item['score']} [召回: {item['details']['recall_score']} | 精排: {item['details']['kgat_score']} ]")

                work = item['representative_work']
                print(f" > 代表论文: 《{work['title']}》")
                print(f" > OpenAlex: {work['link']}")

                print(f" > 推荐理由: {item['recommendation_reason']}")

                m = item['metrics']
                print(f" > 学术指标: H-Index {m.get('h_index', 0)} | 总论文 {m.get('total_papers', 0)} | 总引用 {m.get('citations', 0)}")
                print("-" * 80)

            print(f"[*] 本次匹配共找到符合条件的专家 {len(results)} 名。")

    except KeyboardInterrupt:
        print("\n[!] 系统安全退出。")