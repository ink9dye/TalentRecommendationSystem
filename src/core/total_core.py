import torch
import os
import glob
import numpy as np

from src.infrastructure.database.kgat_ax.model import KGAT
from src.infrastructure.database.kgat_ax.data_loader import DataLoaderKGAT
from src.infrastructure.database.kgat_ax.kgat_parser.parser_kgat import parse_kgat_args
from config import KGATAX_TRAIN_DATA_DIR

from src.core.recall.total_recall import TotalRecallSystem
from src.core.ranking.ranking_engine import RankingEngine


class TotalCore:
    def __init__(self):
        """系统初始化：仅负责基础组件的加载"""
        self.args = parse_kgat_args()
        self.device = torch.device("cpu")  # 精排推荐使用CPU

        # 1. 初始化数据环境
        self._init_data_env()

        # 2. 初始化核心模型
        self.model = self._load_trained_model()

        # 3. 挂载子系统 (解耦后的召回与精排)
        self.recall_subsystem = TotalRecallSystem()
        self.ranking_engine = RankingEngine(self.model, self.dataloader)

        print(f"[Debug] Entity Offset: {self.args.ENTITY_OFFSET}")
        print(f"[Debug] Total Nodes in Model: {self.model.global_max_id}")

    def _init_data_env(self):
        """数据与参数初始化"""
        self.args.data_dir = os.path.dirname(KGATAX_TRAIN_DATA_DIR)
        self.args.data_name = os.path.basename(KGATAX_TRAIN_DATA_DIR)

        self.dataloader = DataLoaderKGAT(self.args, self._get_silent_logger())
        self.args.ENTITY_OFFSET = self.dataloader.ENTITY_OFFSET
        self.args.n_aux_features = getattr(self.args, 'n_aux_features', 3)

    def _load_trained_model(self):
        """模型加载逻辑逻辑提取到独立方法"""
        # 显式传入 dataloader 中的 A_in 邻接矩阵进行初始化
        model = KGAT(
            self.args, self.dataloader.n_users,
            self.dataloader.n_users_entities, self.dataloader.n_relations,
            self.dataloader.A_in
        )

        # 寻找并加载最佳权重
        weight_path = os.path.join(KGATAX_TRAIN_DATA_DIR, 'weights')
        best_files = glob.glob(os.path.join(weight_path, "best_model_epoch_*.pth"))

        # 建议使用文件名中的 Epoch 数字排序，以获取真正的最新/最佳模型
        target = None
        if best_files:
            best_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]), reverse=True)
            target = best_files[0]

        if target:
            # 1. weights_only=False：解决 numpy 标量导致的 UnpicklingError
            ckpt = torch.load(target, map_location='cpu', weights_only=False)

            # 2. strict=False：忽略权重文件中的 "A_in" 键，防止 Unexpected key 报错
            # 只要模型参数（Embedding、Linear层等）匹配，加载就会成功
            model.load_state_dict(ckpt['model_state_dict'], strict=False)

            print(f"[OK] 载入权重: {os.path.basename(target)} (训练进度: Epoch {ckpt.get('epoch', '??')})")

        return model

    def suggest(self, text_query: str):
        """
        核心工作流：语义导航 -> 多路召回 -> 深度精排 (KGATAX 主导版)
        """
        # --- Step 1: 语义导航 (Semantic Navigation) ---
        # 目的：将非结构化描述映射到图谱中已有的“锚点岗位”
        # 使用向量检索（FAISS）找到与当前描述最接近的 3 个历史岗位 ID
        query_vec, _ = self.recall_subsystem.encoder.encode(text_query)
        _, indices = self.recall_subsystem.v_path.job_index.search(query_vec, 3)

        # 获取这 3 个岗位在数据库中的原始 ID (securityId)
        real_job_ids = [self.recall_subsystem.v_path.job_id_map[idx] for idx in indices[0]]

        if not real_job_ids:
            print("[Error] 语义导航未能锁定任何岗位锚点，请检查 job_index 是否完整。")
            return []

        print(f"[*] 语义导航完成，锁定 Top-3 Job 锚点: {real_job_ids}")

        # --- Step 2: 多路召回 (Multi-channel Recall) ---
        # 目的：快速从全量人才库中筛选出 500 个高潜力候选人
        # 这里的 execute 通常包含了语义向量召回、标签召回和图谱关系召回的融合
        recall_res = self.recall_subsystem.execute(text_query)
        candidates = recall_res.get('final_top_500', [])

        if not candidates:
            print("[Warning] 召回阶段未获得候选人。")
            return []

        # --- Step 3: 深度精排与证据链生成 (KGATAX Reranking) ---
        # 目的：利用 KGAT 的全息 Embedding 进行多锚点均值融合评分，并生成证据路径
        # 逻辑流程：
        # 1. 提取 real_job_ids 的 Embedding 并计算均值中心
        # 2. 计算 500 人相对于该中心的 KGATAX 匹配得分
        # 3. 筛选 Top 100
        # 4. 针对每个人，回溯其与锚点岗位之间的知识路径（证据链）

        print(f"[*] 进入精排阶段，对 {len(candidates)} 名候选人执行拓扑评估...")

        # 调用 RankingEngine 的执行方法
        # 传入多锚点 ID 以平滑语义偏置，返回带证据链的 Top 100 结果
        final_results = self.ranking_engine.execute_rank(real_job_ids, candidates)

        print(f"[OK] 流程结束。已生成 100 名候选人的精排列表及对应证据链。")
        return final_results

    def _get_silent_logger(self):
        class S:
            def info(self, m): pass

            def error(self, m): print(f"Error: {m}")

        return S()


if __name__ == "__main__":
    app = TotalCore()
    print(app.suggest("寻找计算机视觉专家"))