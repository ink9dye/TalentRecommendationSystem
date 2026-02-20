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
        【修复版】执行局部精排评分，仅针对 500 名候选人，杜绝内存溢出。
        """
        self.model.eval()

        # 1. 岗位锚点 ID 转换诊断 (Job Anchors)
        job_int_ids = [self.dataloader.user_to_int.get(str(jid).strip(), 0) for jid in real_job_ids]
        valid_jobs = len([x for x in job_int_ids if x > 0])

        # 2. 人才候选人 ID 转换 (Candidate Authors)
        item_int_ids = []
        for aid in candidate_raw_ids:
            # 确保对齐 DataLoader 中的 a_ 前缀和小写逻辑
            key = f"a_{str(aid).strip().lower()}"
            idx = self.dataloader.entity_to_int.get(key, 0)
            item_int_ids.append(idx)

        # 3. 构造活跃节点子集 (Active Node Subset)
        # 我们只计算这 500+3 个节点的 Embedding
        all_active_ids = torch.LongTensor(job_int_ids + item_int_ids).to(self.device)

        with torch.no_grad():
            # 4. 局部学术特征 (AX) 提取
            # 关键：不再加载 187 万行的 aux_info，只切片出需要的 500 多行
            aux_info_subset = self.dataloader.aux_info_all[all_active_ids].to(self.device)

            # 5. 调用模型新增的局部计算接口 calc_cf_embeddings_subset
            # 注意：需确保 model.py 中已添加该方法
            active_embeds = self.model.calc_cf_embeddings_subset(all_active_ids, aux_info_subset)

            # 6. 拆分岗位与人才的 Embedding 空间
            n_jobs = len(job_int_ids)
            job_embeds = active_embeds[:n_jobs]  # 前面是 Job 锚点
            item_embeds = active_embeds[n_jobs:]  # 后面是候选人

            # 7. 执行张量运算：多锚点均值融合
            # 聚合 3 个锚点岗位的特征，形成一个统一的“理想人选”向量
            u_e_avg = torch.mean(job_embeds, dim=0, keepdim=True)

            # 8. 计算点积得分
            # 结果维度: [1, 500] -> squeeze -> [500]
            scores = torch.matmul(u_e_avg, item_embeds.transpose(0, 1)).squeeze(0)

            # 9. 诊断输出
            print(f"[Scorer Debug] 局部计算完成 | 锚点:{valid_jobs} | 候选人:{len(item_int_ids)}")
            print(
                f"[Scorer Debug] 分值区间: [{scores.min():.4f} ~ {scores.max():.4f}] | 极差={scores.max() - scores.min():.4f}")

            return scores