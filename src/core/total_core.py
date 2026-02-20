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
        核心工作流：语义导航(仅精排锚定) -> 可选领域召回 -> 深度精排(1:1 融合重排)
        :param text_query: 原始需求描述
        :param manual_domain_id: 只有显式输入时，才会触发召回路径的硬过滤
        """
        # --- Step 1: 语义导航 (确定精排阶段的坐标中心) ---
        # 即使召回阶段没选领域，精排也需要找到对应的 Job 锚点来计算向量距离
        query_vec, _ = self.recall_subsystem.encoder.encode(text_query)
        _, indices = self.recall_subsystem.v_path.job_index.search(query_vec, 3)

        # 获取 TOP-3 岗位 ID，用于后续 Scorer 计算 Embedding 均值
        real_job_ids = [self.recall_subsystem.v_path.job_id_map[idx] for idx in indices[0]]

        if not real_job_ids:
            print("[Error] 语义导航未能锁定任何岗位锚点。")
            return []

        print(f"[*] 语义导航完成，锁定精排锚点: {real_job_ids}")

        # --- Step 2: 多路召回 (遵循“无输入不假设”原则) ---
        # 召回系统会返回按 RRF 融合排序后的 500 名候选人
        recall_res = self.recall_subsystem.execute(text_query, domain_id=manual_domain_id)
        candidates = recall_res.get('final_top_500', [])

        if not candidates:
            print("[Warning] 召回阶段未获得候选人。")
            return []

        # --- Step 3: 深度精排与 1:1 权重融合 ---
        # 这里调用的 execute_rank 内部已改为：
        # 1. 局部 Embedding 计算 (解决 187万节点 OOM 问题)
        # 2. 0.5 * KGAT 精排得分 + 0.5 * 召回顺序得分
        print(f"[*] 进入局部精排阶段，对 {len(candidates)} 名候选人执行 1:1 权重融合评估...")

        # 传入 real_job_ids (锚点) 和 candidates (召回列表)
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
    print(app.suggest("运动学与动力学算法研发：负责机器人运动学、动力学建模以及运动控制算法的设计与优化；建立⾼性能、可扩展的机器⼈运动控制与状态估计模块。轨迹规划与全⾝控制算法开发：研发与优化机器人轨迹生成与全⾝控制算法，包括但不限于RRT/PRM/CHOMP/MPC/iLQR等；针对复杂场景进行约束优化、时序规划与碰撞规避设计，确保系统的平滑性、稳定性与可执行性。仿真平台构建与验证：利用Isaac Sim/Gazebo/MuJoCo等平台搭建仿真环境，进行算法快速验证与评估；推动仿真到实机的一致性优化，包括动力学一致性、摩擦模型校准、控制频率匹配等。系统集成与性能调优：参与机器人控制系统全流程开发，从底层控制到高层规划架构；开展实时控制性能调优（延迟、抖动、稳定性分析），提升系统在复杂任务下的执行效率与鲁棒性。技术追踪与创新：持续关注运动控制、机器人动力学建模及规划领域的前沿研究，对新算法进行调研、实现与落地；推动技术创新并形成知识沉淀。"))