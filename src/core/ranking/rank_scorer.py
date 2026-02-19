import torch


class RankScorer:
    """
    职责：执行 KGATAX 空间向量运算与分值计算。
    核心逻辑：多锚点均值融合 + AX 特征注入 + ID 归一化。
    """

    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device


    def compute_scores(self, real_job_ids: list, candidate_raw_ids: list):
        """
        执行深度精排评分，并集成全链路诊断逻辑。
        """
        self.model.eval()

        # 1. 岗位锚点 ID 转换诊断
        # 统计有多少个锚点岗位在 ID 空间中找到了对应位置
        job_int_ids = [self.dataloader.user_to_int.get(str(jid).strip(), 0) for jid in real_job_ids]
        valid_jobs = len([x for x in job_int_ids if x > 0])
        print(f"[Scorer Debug] 锚点映射: 有效={valid_jobs}/{len(real_job_ids)} (IDs: {job_int_ids})")

        # 2. 人才候选人 ID 转换诊断
        # 强制小写归一化
        item_int_ids = []
        for aid in candidate_raw_ids:
            key = f"a_{str(aid).strip().lower()}"
            idx = self.dataloader.entity_to_int.get(key, 0)
            item_int_ids.append(idx)

        valid_items = len([x for x in item_int_ids if x > 0])
        print(f"[Scorer Debug] 人才映射: 有效={valid_items}/500")

        with torch.no_grad():
            # 3. 学术特征 (AX) 完整性自检
            aux_info = self.dataloader.aux_info_all.to(self.device)
            # 检查特征矩阵是否全是 0 (如果是 0，说明 DataLoader 的 lower() 没改对)
            aux_sum = aux_info[item_int_ids].sum().item()
            print(f"[Scorer Debug] AX 特征强度: 总和={aux_sum:.4f} (若为0则特征加载失败)")

            # 4. 计算 KGATAX 全息 Embedding
            all_embeds = self.model.calc_cf_embeddings(aux_info)

            # 监控 Embedding 向量是否存在“坍缩” (Standard Deviation 过小)
            print(f"[Scorer Debug] Embedding 状态: Mean={all_embeds.mean():.4f}, Std={all_embeds.std():.4f}")

            # 5. 执行张量运算
            job_indices = torch.LongTensor(job_int_ids).to(self.device)
            item_indices = torch.LongTensor(item_int_ids).to(self.device)

            # 多锚点均值融合
            u_e_multi = all_embeds[job_indices]
            u_e_avg = torch.mean(u_e_multi, dim=0, keepdim=True)

            i_e = all_embeds[item_indices]

            # 计算点积得分
            scores = torch.matmul(u_e_avg, i_e.transpose(0, 1)).squeeze(0)

            # 6. 分值区分度监控
            # 如果 Max 与 Min 差距极小，说明模型没有辨别力
            print(
                f"[Scorer Debug] 分值区间: [{scores.min():.4f} ~ {scores.max():.4f}] | 极差={scores.max() - scores.min():.4f}")

            return scores