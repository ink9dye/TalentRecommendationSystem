import torch
import os
import sys
import glob

# 1. 导入基础定义与配置
from src.infrastructure.database.kgat_ax.model import KGAT
from src.infrastructure.database.kgat_ax.data_loader import DataLoaderKGAT
from src.infrastructure.database.kgat_ax.kgat_parser.parser_kgat import parse_kgat_args
from config import KGATAX_TRAIN_DATA_DIR

# 2. 导入子系统
from src.core.recall.total_recall import TotalRecallSystem
from src.core.ranking.ranking_engine import RankingEngine


# src/core/talent_app.py

class TalentApp:
    def __init__(self):
        self.args = parse_kgat_args()
        self.args.data_dir = os.path.dirname(KGATAX_TRAIN_DATA_DIR)
        self.args.data_name = os.path.basename(KGATAX_TRAIN_DATA_DIR)

        print(f"[*] 启动路径校验: {KGATAX_TRAIN_DATA_DIR}")

        # 1. 启动数据加载器
        self.dataloader = DataLoaderKGAT(self.args, self._get_silent_logger())

        # --- 【核心修复：同步偏移量】 ---
        # KGAT 模型内部需要此参数来区分 User 空间和 Entity 空间
        self.args.ENTITY_OFFSET = self.dataloader.ENTITY_OFFSET

        # 如果你之前在训练时手动设置了特征数，这里也建议同步一下
        self.args.n_aux_features = getattr(self.args, 'n_aux_features', 3)

        # 2. 此时初始化模型，就不会报 'ENTITY_OFFSET' 缺失错误了
        self.model = KGAT(
            self.args,
            self.dataloader.n_users,
            self.dataloader.n_users_entities,
            self.dataloader.n_relations,
            self.dataloader.A_in
        )
        self._auto_load_best_weights()
        self.model.eval()

        self.recall_subsystem = TotalRecallSystem()
        self.ranking_engine = RankingEngine(self.model, self.dataloader)

    def _auto_load_best_weights(self):
        """
        自动探测逻辑：优先找序号最大的 best 模型，没找到则找最新的 checkpoint
        """
        weight_path = os.path.join(KGATAX_TRAIN_DATA_DIR, 'weights')

        # 1. 搜寻所有 best 模型文件
        best_files = glob.glob(os.path.join(weight_path, "best_model_epoch_*.pth"))

        target_file = None
        if best_files:
            # 根据文件名中的数字（Epoch）降序排列，取最大的一个
            best_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]), reverse=True)
            target_file = best_files[0]
        else:
            # 2. 如果没找到 best，退而求其次寻找最新的 checkpoint
            latest_path = os.path.join(weight_path, "latest_checkpoint.pth")
            if os.path.exists(latest_path):
                target_file = latest_path

        if target_file and os.path.exists(target_file):
            ckpt = torch.load(target_file, map_location='cpu', weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(
                f"[OK] 自动探测并加载模型: {os.path.basename(target_file)} (训练进度: Epoch {ckpt.get('epoch', '??')})")
        else:
            raise FileNotFoundError(f"在 {weight_path} 下未发现任何可用权重，请先运行 pipeline.py 进行训练。")

    def suggest(self, text_query, job_id="job_001"):
        """
        执行端到端推荐
        """
        # 1. 执行召回
        recall_res = self.recall_subsystem.execute(text_query)
        candidates = recall_res['final_top_500']

        # 2. 执行精排与证据回溯
        final_list = self.ranking_engine.execute_rank(job_id, candidates)
        return final_list

    def _get_silent_logger(self):
        class S:
            def info(self, m): pass

            def error(self, m): print(f"Error: {m}")

        return S()


if __name__ == "__main__":
    # 现在无需传参，直接启动
    try:
        app = TalentApp()
        query = "寻找擅长计算机视觉和图像生成的学者，有顶会论文发表经验"

        print(f"\n[*] 正在分析需求: {query}")
        results = app.suggest(query)

        # 打印 Top 3 结果
        for res in results[:3]:
            print(f"\n[排名 {res['rank']}] {res['name']} (得分: {res['score']})")
            print(f" ├─ 学术地位: H-index {res['metrics']['h_index']} | 总引 {res['metrics']['citations']}")
            print(f" ├─ 推荐理由: {res['evidence_chain']['summary']}")
            if res['evidence_chain']['representative_works']:
                print(f" └─ 代表作: {res['evidence_chain']['representative_works'][0]}")
    except Exception as e:
        print(f"\n[错误] 系统启动失败: {e}")