"""
精排评分组件：KGAT-AX 空间向量运算与分值计算。

- 无候选池或模型未启用四分支时：多锚点均值融合 + AX 特征注入（calc_cf_embeddings_subset）。
- 有 candidate_records 且模型启用四分支时：使用 calc_score_v2，融合 Graph / Author / Recall / Interaction 四塔得分（README 5.3～5.4）。
"""
import torch
import numpy as np
from typing import List, Optional, Any


def _record_to_four_branch(record: Any, max_pool_score: float = 1.0):
    """将一条 CandidateRecord 转为四分支特征向量（与 generate_training_data 导出格式一致）。"""
    max_rank = 500.0
    recall = [
        1.0 if getattr(record, "from_vector", False) else 0.0,
        1.0 if getattr(record, "from_label", False) else 0.0,
        1.0 if getattr(record, "from_collab", False) else 0.0,
        (getattr(record, "path_count", 0) or 0) / 3.0,
        1.0 - min((getattr(record, "vector_rank", None) or 0) / max_rank, 1.0),
        1.0 - min((getattr(record, "label_rank", None) or 0) / max_rank, 1.0),
        1.0 - min((getattr(record, "collab_rank", None) or 0) / max_rank, 1.0),
        (float(getattr(record, "candidate_pool_score", 0) or 0) / max_pool_score) if max_pool_score else 0.0,
        {"A": 0.25, "B": 0.5, "C": 0.75, "D": 1.0}.get((getattr(record, "bucket_type") or "D").strip().upper(), 0.5),
        1.0 if getattr(record, "is_multi_path_hit", False) else 0.0,
        float(getattr(record, "vector_score_raw") or 0.0),
        float(getattr(record, "label_score_raw") or 0.0),
        float(getattr(record, "collab_score_raw") or 0.0),
    ]
    h, w, c = getattr(record, "h_index", None), getattr(record, "works_count", None), getattr(record, "cited_by_count", None)
    author_aux = [
        np.log1p(h) if h is not None else 0.0,
        np.log1p(c) if c is not None else 0.0,
        np.log1p(w) if w is not None else 0.0,
    ] + [0.0] * 9
    interaction = [
        float(getattr(record, "topic_similarity") or 0.0),
        float(getattr(record, "skill_coverage_ratio") or 0.0),
        float(getattr(record, "domain_consistency") or 0.0),
        float(getattr(record, "paper_hit_strength") or 0.0),
        float(getattr(record, "recent_activity_match") or 0.0),
        0.0, 0.0, 0.0,
    ]
    return recall, author_aux, interaction


class RankScorer:
    """
    职责：执行 KGATAX 空间向量运算与分值计算。
    无候选池时：多锚点均值融合 + AX 特征注入；有候选池且模型四分支时：使用 calc_score_v2 融合四塔得分。
    """

    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def compute_scores(
        self,
        real_job_ids: list,
        candidate_raw_ids: list,
        candidate_records: Optional[List[Any]] = None,
    ):
        """
        执行精排评分。若传入 candidate_records 且模型启用四分支，则使用 calc_score_v2 融合四塔；
        否则使用图塔 + AX 的 calc_cf_embeddings_subset 得分。
        """
        self.model.eval()

        job_int_ids = [self.dataloader.user_to_int.get(str(jid).strip(), 0) for jid in real_job_ids]
        item_int_ids = []
        for aid in candidate_raw_ids:
            key = f"a_{str(aid).strip().lower()}"
            idx = self.dataloader.entity_to_int.get(key, 0)
            item_int_ids.append(idx)

        use_four = getattr(self.model, "_use_four_tower", lambda: False)() and candidate_records is not None and len(candidate_records) == len(candidate_raw_ids)

        if use_four:
            with torch.no_grad():
                aux_info_all = self.dataloader.aux_info_all.to(self.device)
                max_score = max((getattr(r, "candidate_pool_score", 0) or 0.0) for r in candidate_records) or 1.0
                B1, B2 = len(job_int_ids), len(item_int_ids)
                n_recall = getattr(self.model, "n_recall_features", 13)
                n_author = getattr(self.model, "n_author_aux", 12)
                n_inter = getattr(self.model, "n_interaction_features", 8)
                recall_list = []
                author_aux_list = []
                interaction_list = []
                for r in candidate_records:
                    rec, auth, inter = _record_to_four_branch(r, max_score)
                    recall_list.append(rec[:n_recall])
                    author_aux_list.append(auth[:n_author])
                    interaction_list.append(inter[:n_inter])
                author_aux_item = torch.tensor(author_aux_list, dtype=torch.float32, device=self.device)
                recall_features = torch.tensor(recall_list, dtype=torch.float32, device=self.device).unsqueeze(0).expand(B1, -1, -1)
                interaction_features = torch.tensor(interaction_list, dtype=torch.float32, device=self.device).unsqueeze(0).expand(B1, -1, -1)
                job_t = torch.LongTensor(job_int_ids).to(self.device)
                item_t = torch.LongTensor(item_int_ids).to(self.device)
                out = self.model.calc_score_v2(
                    job_t, item_t, aux_info_all,
                    author_aux_item=author_aux_item,
                    recall_features=recall_features,
                    interaction_features=interaction_features,
                )
                s = out["final_score"]
                if s.dim() == 2 and s.size(0) == 1:
                    combined_scores = s.squeeze(0)
                elif s.dim() == 2:
                    combined_scores = s.mean(dim=0)
                else:
                    combined_scores = s
        else:
            all_active_ids = torch.LongTensor(job_int_ids + item_int_ids).to(self.device)
            with torch.no_grad():
                aux_info_subset = self.dataloader.aux_info_all[all_active_ids].to(self.device)
                active_embeds = self.model.calc_cf_embeddings_subset(all_active_ids, aux_info_subset)
                n_jobs = len(job_int_ids)
                job_embeds = active_embeds[:n_jobs]
                item_embeds = active_embeds[n_jobs:]
                u_e_avg = torch.mean(job_embeds, dim=0, keepdim=True)
                combined_scores = torch.matmul(u_e_avg, item_embeds.transpose(0, 1)).squeeze(0)

        min_s, max_s = combined_scores.min(), combined_scores.max()
        print(f"[Scorer Logic] 采用「四分支融合」模式" if use_four else "[Scorer Logic] 采用「语义主导-学术择优」模式")
        print(f"[Scorer Debug] 锚点:{len(job_int_ids)} | 候选人:{len(item_int_ids)} | 分值区间: [{min_s:.4f} ~ {max_s:.4f}]")
        return combined_scores