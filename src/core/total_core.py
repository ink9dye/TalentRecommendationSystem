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

    def suggest(self, text_query: str, manual_domain_id: str = None):
        """
        核心工作流：语义导航(仅精排锚定) -> 可选领域召回 -> 深度精排
        :param text_query: 原始需求描述
        :param manual_domain_id: 只有显式输入时，才会触发召回路径的硬过滤
        """
        # --- Step 1: 语义导航 (仅用于确定精排阶段的坐标中心) ---
        query_vec, _ = self.recall_subsystem.encoder.encode(text_query)
        _, indices = self.recall_subsystem.v_path.job_index.search(query_vec, 3)

        # 这些锚点 ID 仅用于后续计算 Ranking Score，不用于生成 domain 滤镜
        real_job_ids = [self.recall_subsystem.v_path.job_id_map[idx] for idx in indices[0]]

        if not real_job_ids:
            print("[Error] 语义导航未能锁定任何岗位锚点。")
            return []

        print(f"[*] 语义导航完成，锁定精排锚点: {real_job_ids}")

        # --- Step 2: 多路召回 (严格遵循“无输入不假设”) ---
        # 除非 manual_domain_id 有值，否则向量路将执行全库语义召回
        # 标签路会根据自身逻辑锚定技能，但不会被强制施加领域硬过滤
        recall_res = self.recall_subsystem.execute(text_query, domain_id=manual_domain_id)
        candidates = recall_res.get('final_top_500', [])

        if not candidates:
            print("[Warning] 召回阶段未获得候选人。")
            return []

        # --- Step 3: 深度精排与证据链生成 ---
        # 利用锚点岗位的 Embedding 均值对 500 名候选人进行重排
        print(f"[*] 进入精排阶段，对 {len(candidates)} 名候选人执行拓扑评估...")

        final_results = self.ranking_engine.execute_rank(real_job_ids, candidates)

        print(f"[OK] 流程结束。已生成 100 名候选人的精排列表。")
        return final_results

    def _get_silent_logger(self):
        class S:
            def info(self, m): pass

            def error(self, m): print(f"Error: {m}")

        return S()


if __name__ == "__main__":
    app = TotalCore()
    print(app.suggest("寻找计算机视觉专家"))