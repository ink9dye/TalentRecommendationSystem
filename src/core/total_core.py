import torch
import os
import glob
import sqlite3
import re
import numpy as np

from src.infrastructure.database.kgat_ax.model import KGAT
from src.infrastructure.database.kgat_ax.data_loader import DataLoaderKGAT
from src.infrastructure.database.kgat_ax.kgat_parser.parser_kgat import parse_kgat_args
from config import KGATAX_TRAIN_DATA_DIR, DB_PATH  # 确保从 config 引入数据库路径

from src.core.recall.total_recall import TotalRecallSystem
from src.core.ranking.ranking_engine import RankingEngine


class TotalCore:
    # 建立名称到 ID 的映射，方便前端直接传输中文名
    DOMAIN_NAME_TO_ID = {
        "计算机科学": "1", "医学": "2", "政治学": "3", "工程学": "4", "物理学": "5",
        "材料科学": "6", "生物学": "7", "地理学": "8", "化学": "9", "商学": "10",
        "社会学": "11", "哲学": "12", "环境科学": "13", "数学": "14", "心理学": "15",
        "地质学": "16", "经济学": "17"
    }

    def __init__(self):
        """系统初始化：负责核心组件加载与环境对齐"""
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
        """模型加载逻辑：同步 A_in 矩阵并加载最优权重"""
        model = KGAT(
            self.args, self.dataloader.n_users,
            self.dataloader.n_users_entities, self.dataloader.n_relations,
            self.dataloader.A_in
        )

        weight_path = os.path.join(KGATAX_TRAIN_DATA_DIR, 'weights')
        best_files = glob.glob(os.path.join(weight_path, "best_model_epoch_*.pth"))

        if best_files:
            best_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]), reverse=True)
            target = best_files[0]
            ckpt = torch.load(target, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"[OK] 载入权重: {os.path.basename(target)} (训练进度: Epoch {ckpt.get('epoch', '??')})")

        return model

    def _get_union_domain_regex(self, job_ids: list):
        """
        【关键逻辑】从数据库获取 3 个锚点岗位的领域标签并取并集。
        能将 "1|17" 和 "2|18" 正确拼接为 "1|2|17|18"
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            placeholders = ','.join(['?'] * len(job_ids))
            sql = f"SELECT domain_ids FROM jobs WHERE securityId IN ({placeholders})"
            rows = conn.execute(sql, job_ids).fetchall()
            conn.close()

            domain_set = set()
            for row in rows:
                if row[0]:
                    # 鲁棒性解析：支持逗号和竖线分隔符的混合处理
                    parts = re.split(r'[,|]', str(row[0]))
                    domain_set.update([p.strip() for p in parts if p.strip()])

            # 返回拼接后的并集字符串，例如 "1|2|17|18"
            return "|".join(sorted(list(domain_set))) if domain_set else None
        except Exception as e:
            print(f"[Warning] 精排阶段领域并集提取失败: {e}")
            return None

    def suggest(self, text_query: str, manual_domain_id: str = None):
        """
        核心工作流：语义导航 -> 广度召回 -> 精排并集过滤 -> 1:1 融合重排
        :param text_query: 原始需求描述
        :param manual_domain_id: 前端手动输入的领域 (若为空则执行并集模式)
        """
        # 0. 规范化手动输入的领域 ID
        target_domain = self.DOMAIN_NAME_TO_ID.get(manual_domain_id, manual_domain_id)
        if target_domain in ["0", ""]: target_domain = None

        # --- Step 1: 语义导航 (寻找 3 个锚点岗位) ---
        query_vec, _ = self.recall_subsystem.encoder.encode(text_query)
        _, indices = self.recall_subsystem.v_path.job_index.search(query_vec, 3)
        real_job_ids = [self.recall_subsystem.v_path.job_id_map[idx] for idx in indices[0]]

        if not real_job_ids:
            print("[Error] 语义导航未能锁定任何岗位锚点。")
            return []

        print(f"[*] 语义导航完成，锁定精排锚点: {real_job_ids}")

        # --- Step 2: 多路召回 (广度模式) ---
        # 核心：此处仅传 target_domain。如果用户没选，则 domain_id 为 None，执行全库召回
        recall_res = self.recall_subsystem.execute(text_query, domain_id=target_domain)
        candidates = recall_res.get('final_top_500', [])

        if not candidates:
            print("[Warning] 召回阶段未获得候选人。")
            return []

        # --- Step 3: 深度精排阶段 (执行并集过滤) ---
        # 【核心修改点】仅在此时生成用于精排过滤的并集正则
        filter_pattern = None
        if not target_domain:
            filter_pattern = self._get_union_domain_regex(real_job_ids)
            if filter_pattern:
                print(f"[*] 精排阶段激活领域并集修剪模式: {filter_pattern}")
        else:
            filter_pattern = target_domain  # 如果手动选了，就以手动的为准

        print(f"[*] 进入局部精排阶段，对 {len(candidates)} 名候选人执行 1:1 权重融合评估...")

        # 传入 real_job_ids, candidates 以及拼接好的 filter_domain 正则
        final_results = self.ranking_engine.execute_rank(
            real_job_ids,
            candidates,
            filter_domain=filter_pattern
        )

        print(f"[OK] 流程结束。已生成 {len(final_results)} 名候选人的精排列表。")
        return final_results

    def _get_silent_logger(self):
        class S:
            def info(self, m): pass
            def error(self, m): print(f"Error: {m}")
        return S()


if __name__ == "__main__":
    app = TotalCore()
    # JD 示例文本：涵盖了运动学、动力学以及仿真平台要求
    query_text = ("运动学与动力学算法研发：负责机器人运动学、动力学建模以及运动控制算法的设计与优化；建立⾼性能、可扩展的机器⼈运动控制与状态估计模块。轨迹规划与全⾝控制算法开发：研发与优化机器人轨迹生成与全⾝控制算法，包括但不限于RRT/PRM/CHOMP/MPC/iLQR等；针对复杂场景进行约束优化、时序规划与碰撞规避设计，确保系统的平滑性、稳定性与可执行性。仿真平台构建与验证：利用Isaac Sim/Gazebo/MuJoCo等平台搭建仿真环境，进行算法快速验证与评估；推动仿真到实机的一致性优化，包括动力学一致性、摩擦模型校准、控制频率匹配等。系统集成与性能调优：参与机器人控制系统全流程开发，从底层控制到高层规划架构；开展实时控制性能调优（延迟、抖动、稳定性分析），提升系统在复杂任务下的执行效率与鲁棒性。技术追踪与创新：持续关注运动控制、机器人动力学建模及规划领域的前沿研究，对新算法进行调研、实现与落地；推动技术创新并形成知识沉淀。")

    # 执行建议：系统将自动执行语义导航、全量召回、并集过滤及 1:1 融合评分
    results = app.suggest(query_text)

    # 以列表形式展示结果
    print("\n" + "=" * 80)
    print(f"🚀 人才推荐结果列表 (已根据岗位领域并集进行精排过滤)")
    print("=" * 80)

    # 遍历 Top 10 名候选人，展示你期待的核心信息
    for item in results[:10]:
        print(f"Rank {item['rank']:<2} | {item['name']} (ID: {item['author_id']})")
        print(
            f" > 综合得分: {item['score']} [召回得分: {item['details']['recall_score']} | 精排得分: {item['details']['kgat_score']} ]")

        # 展示代表论文及其 OpenAlex 链接
        work = item['representative_work']
        print(f" > 代表论文: 《{work['title']}》")
        print(f" > 论文 ID : {work['work_id']} | 预览链接: {work['link']}")

        # 展示明确的推荐理由
        print(f" > 推荐理由: {item['recommendation_reason']}")

        # 展示学术指标 (H-Index, 引用量)
        m = item['metrics']
        print(f" > 学术指标: H-Index {m.get('h_index', 0)} | 总引用 {m.get('citations', 0)}")

        print("-" * 80)

    print(f"[*] 推荐结束。共找到符合条件的候选人 {len(results)} 名。")