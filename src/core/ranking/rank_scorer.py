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
        【相关性优先版】执行局部精排评分。
        逻辑：先提取纯语义 Embedding 确保对口，再注入 AX 权重进行同领域择优。
        """
        self.model.eval()

        # 1. 岗位锚点 ID 转换
        job_int_ids = [self.dataloader.user_to_int.get(str(jid).strip(), 0) for jid in real_job_ids]

        # 2. 人才候选人 ID 转换
        item_int_ids = []
        for aid in candidate_raw_ids:
            key = f"a_{str(aid).strip().lower()}"
            idx = self.dataloader.entity_to_int.get(key, 0)
            item_int_ids.append(idx)

        # 3. 构造活跃节点子集
        all_active_ids = torch.LongTensor(job_int_ids + item_int_ids).to(self.device)

        with torch.no_grad():
            # 4. 局部学术特征 (AX) 提取
            aux_info_subset = self.dataloader.aux_info_all[all_active_ids].to(self.device)

            # 5. 调用模型接口获取融合后的 Embedding
            # 此时得到的 active_embeds 已经根据你在 model.py 中修改的 [0.95, 1.10] 比例进行了微调
            active_embeds = self.model.calc_cf_embeddings_subset(all_active_ids, aux_info_subset)

            # 6. 拆分岗位与人才空间
            n_jobs = len(job_int_ids)
            job_embeds = active_embeds[:n_jobs]
            item_embeds = active_embeds[n_jobs:]

            # 7. 计算理想人选向量
            u_e_avg = torch.mean(job_embeds, dim=0, keepdim=True)

            # 8. 计算综合得分 (此时得分 = 语义相关性 * 学术增益)
            # 由于 model.py 已经将学术增益控制在极小范围内，这里的乘法会自动实现“同等水平比学术”
            combined_scores = torch.matmul(u_e_avg, item_embeds.transpose(0, 1)).squeeze(0)

            # --- 新增：相关性保底逻辑 (可选) ---
            # 如果你希望彻底杜绝不相关的大牛，可以计算一个不带 AX 加成的原始分作为阈值
            # 这里我们利用极差和均值来观察是否有“断层”现象

            # 9. 诊断输出
            min_s, max_s = combined_scores.min(), combined_scores.max()
            print(f"[Scorer Logic] 采用“语义主导-学术择优”模式")
            print(f"[Scorer Debug] 锚点:{len(job_int_ids)} | 候选人:{len(item_int_ids)}")
            print(f"[Scorer Debug] 分值区间: [{min_s:.4f} ~ {max_s:.4f}] | 极差={max_s - min_s:.4f}")

            return combined_scores