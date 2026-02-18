# algorithms/kgat_ax/__init__.py
from src.infrastructure.database.kgat_ax.model import KGAT
from src.infrastructure.database.kgat_ax.trainer import train as train_kgat_ax
import torch

class KGAT_AX_Predictor:
    """供 app/services/ 使用的单例预测器"""
    def __init__(self, model_path, args, n_users, n_entities, n_relations):
        self.model = KGAT(args, n_users, n_entities, n_relations)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def rank(self, user_id, candidate_ids, aux_info_all):
        """执行精排"""
        with torch.no_grad():
            scores = self.model.calc_score(user_id, candidate_ids, aux_info_all)
        return scores