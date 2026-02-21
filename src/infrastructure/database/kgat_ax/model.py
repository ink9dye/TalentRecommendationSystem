import torch
import torch.nn as nn
import torch.nn.functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):
    """
    轻量化的聚合器层，仅负责邻居信息聚合与非线性变换
    """

    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
        # 修复点：super 必须指向当前类 Aggregator
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        if self.aggregator_type == 'gcn':
            self.linear = nn.Linear(self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.linear.weight)
        elif self.aggregator_type == 'graphsage':
            self.linear = nn.Linear(self.in_dim * 2, self.out_dim)
            nn.init.xavier_uniform_(self.linear.weight)
        elif self.aggregator_type == 'bi-interaction':
            self.linear1 = nn.Linear(self.in_dim, self.out_dim)
            self.linear2 = nn.Linear(self.in_dim, self.out_dim)
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.xavier_uniform_(self.linear2.weight)
        else:
            raise NotImplementedError

    def forward(self, ego_embeddings, A_in):
        # 执行图卷积消息传递
        side_embeddings = torch.sparse.mm(A_in, ego_embeddings)

        if self.aggregator_type == 'gcn':
            embeddings = ego_embeddings + side_embeddings
            embeddings = self.activation(self.linear(embeddings))
        elif self.aggregator_type == 'graphsage':
            embeddings = torch.cat([ego_embeddings, side_embeddings], dim=1)
            embeddings = self.activation(self.linear(embeddings))
        elif self.aggregator_type == 'bi-interaction':
            sum_embeddings = self.activation(self.linear1(ego_embeddings + side_embeddings))
            bi_embeddings = self.activation(self.linear2(ego_embeddings * side_embeddings))
            embeddings = bi_embeddings + sum_embeddings
        return self.message_dropout(embeddings)


class KGAT(nn.Module):
    """
    KGAT-AX 主模型：整合学术质量 (AX) 乘法门控与多层图卷积聚合。
    """
    def __init__(self, args, n_users, n_total_nodes, n_relations, A_in=None):
        """
        修正版构造函数：直接使用 n_total_nodes 对齐 Embedding 空间，防止维度漂移。
        """
        super(KGAT, self).__init__()
        self.n_users = n_users
        self.n_relations = n_relations
        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        self.aggregation_type = args.aggregation_type
        self.ENTITY_OFFSET = args.ENTITY_OFFSET

        # 解析层配置
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))
        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        # 1. 核心 AX 映射层：将 3 维学术特征映射到 Embedding 空间
        self.n_aux_features = getattr(args, 'n_aux_features', 3)
        self.aux_embed_layer = nn.Linear(self.n_aux_features, self.embed_dim)
        nn.init.xavier_uniform_(self.aux_embed_layer.weight)

        # 2. 类型偏置层 (用于区分 Job 与 Author 节点)
        self.type_embed = nn.Embedding(2, self.embed_dim)
        nn.init.xavier_uniform_(self.type_embed.weight)

        # 3. 核心 Embedding 空间：对齐全局节点数
        self.global_max_id = n_total_nodes
        self.entity_user_embed = nn.Embedding(self.global_max_id, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)

        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))
        self.W_a = nn.Parameter(torch.Tensor(self.relation_dim, 1))

        nn.init.xavier_uniform_(self.entity_user_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)
        nn.init.xavier_uniform_(self.W_a)

        # 4. 堆叠聚合器层
        self.aggregator_layers = nn.ModuleList([
            Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type)
            for k in range(self.n_layers)
        ])

        if A_in is not None:
            if A_in.shape[0] != self.global_max_id:
                raise ValueError(f"维度不匹配: 邻接矩阵({A_in.shape[0]}) != global_max_id({self.global_max_id})")
            self.register_buffer('A_in', A_in)
        else:
            self.A_in = None

    # src/infrastructure/database/kgat_ax/model.py

    def holographic_fusion(self, entity_embed, aux_info, node_ids=None):
        """
        【统一对齐版】全息融合逻辑：语义相关性主导，学术影响力微调。
        作用：确保人才推荐首先基于专业对口（基准模长），再由学术影响力（AX）实现 15% 以内的排名微调。
        """
        # 1. 学术特征平滑处理：使用 log1p 压制极端数值（如百万级引用），防止其产生压倒性的权重
        # 这能确保 H-index 100 和 H-index 20 的差距在经过变换后处于合理区间
        smooth_aux = torch.log1p(aux_info)

        # 2. 构造“平局决胜”门控 (Tie-breaking Gate)：
        # 修改点：将增益区间控制在 [1.0, 1.15] 之间
        # 1.0 为相关性基准：保证即便是学术新人也不会在对口领域被无故降权
        # 0.15 为最高增益：学术泰斗最高获得 15% 的分值提升，不足以使其在不相关领域超过专业对口者
        gate_weight = torch.sigmoid(self.aux_embed_layer(smooth_aux)) * 0.15 + 1.0

        # 3. 乘法门控融合：将学术增益注入语义向量
        # 此时，AX 特征仅在语义距离（点积）接近的情况下，决定最终的 Rank 顺序
        fused_embed = entity_embed * gate_weight

        # 4. 强制执行 L2 归一化：维持向量空间的模长稳定，确保推理时余弦相似度的准确性
        fused_embed = F.normalize(fused_embed, p=2, dim=-1)

        # 5. 叠加身份标签（Job 0 / Author 1）：利用 ENTITY_OFFSET 区分节点类型
        if node_ids is not None:
            type_labels = (node_ids >= self.ENTITY_OFFSET).long()
            fused_embed += self.type_embed(type_labels)

        return fused_embed

    def calc_cf_embeddings(self, aux_info_all):
        """带身份标识与 AX 增强的消息传递层"""
        device = self.entity_user_embed.weight.device
        all_ids = torch.arange(self.global_max_id).to(device)

        # 基础 Embedding + 融合
        ego_embed = self.holographic_fusion(self.entity_user_embed.weight, aux_info_all, all_ids)
        all_embed = [ego_embed]

        # 图卷积聚合
        for layer in self.aggregator_layers:
            ego_embed = layer(ego_embed, self.A_in)
            all_embed.append(F.normalize(ego_embed, p=2, dim=1))

        return torch.cat(all_embed, dim=1)

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids, aux_info_all):
        all_embed = self.calc_cf_embeddings(aux_info_all)
        u_e, p_e, n_e = all_embed[user_ids], all_embed[item_pos_ids], all_embed[item_neg_ids]

        pos_score = torch.sum(u_e * p_e, 1)
        neg_score = torch.sum(u_e * n_e, 1)

        cf_loss = torch.mean(F.softplus(neg_score - pos_score))
        l2_loss = _L2_loss_mean(u_e) + _L2_loss_mean(p_e) + _L2_loss_mean(n_e)
        return cf_loss + self.cf_l2loss_lambda * l2_loss

    def calc_kg_loss(self, h, r, pos_t, neg_t, h_aux, pos_t_aux, neg_t_aux):
        h_e = self.holographic_fusion(self.entity_user_embed(h), h_aux, h)
        p_t_e = self.holographic_fusion(self.entity_user_embed(pos_t), pos_t_aux, pos_t)
        n_t_e = self.holographic_fusion(self.entity_user_embed(neg_t), neg_t_aux, neg_t)

        r_e = self.relation_embed(r)
        W_r = self.trans_M[r]

        r_h = torch.bmm(h_e.unsqueeze(1), W_r).squeeze(1)
        r_p = torch.bmm(p_t_e.unsqueeze(1), W_r).squeeze(1)
        r_n = torch.bmm(n_t_e.unsqueeze(1), W_r).squeeze(1)

        pos_score = torch.sum(torch.pow(r_h + r_e - r_p, 2), 1)
        neg_score = torch.sum(torch.pow(r_h + r_e - r_n, 2), 1)

        kg_loss = torch.mean(F.softplus(pos_score - neg_score))
        l2_loss = _L2_loss_mean(r_h) + _L2_loss_mean(r_e) + _L2_loss_mean(r_p) + _L2_loss_mean(r_n)
        return kg_loss + self.kg_l2loss_lambda * l2_loss

    def update_attention_batch(self, h_list, t_list, r_list):
        h_e, t_e, r_e = self.entity_user_embed(h_list), self.entity_user_embed(t_list), self.relation_embed(r_list)
        W_r = self.trans_M[r_list]

        h_e = torch.bmm(h_e.unsqueeze(1), W_r).squeeze(1)
        t_e = torch.bmm(t_e.unsqueeze(1), W_r).squeeze(1)

        att_score = torch.sum((h_e + r_e) * torch.tanh(t_e), dim=1, keepdim=True)
        att_score = torch.sigmoid(torch.matmul(att_score, self.W_a.mean(0, keepdim=True).T))
        return att_score

    def forward(self, *input, mode):
        if mode == 'train_cf': return self.calc_cf_loss(*input)
        elif mode == 'train_kg': return self.calc_kg_loss(*input)
        elif mode == 'predict': return self.calc_score(*input)
        elif mode == 'update_att': return self.update_attention_batch(*input)

    def calc_score(self, user_ids, item_ids, aux_info_all):
        all_embed = self.calc_cf_embeddings(aux_info_all)
        return torch.matmul(all_embed[user_ids], all_embed[item_ids].transpose(0, 1))

    def calc_cf_embeddings_subset(self, node_ids, aux_info_subset):
        """
        【优化版】局部特征提取：平衡语义相关性与学术影响力。
        核心逻辑：先确定专业对口（Semantic Match），再由学术指标做小幅优选（Refinement）。
        """
        # 1. 提取基础 Embedding (这部分决定了“是否对口”)
        ego_embed = self.entity_user_embed(node_ids)

        # 2. 执行学术特征平滑处理
        # 使用 log1p 压制极端数值（如部分大佬百万级的引用量），避免拉大差距
        smooth_aux = torch.log1p(aux_info_subset)

        # 3. 构造学术门控 (AX Gate)
        # 修改点：缩放系数0.2，位移0.8
        gate_weight = torch.sigmoid(self.aux_embed_layer(smooth_aux)) * 0.2 + 0.8

        # 4. 执行融合：语义向量 * 微调权重
        fused_embed = ego_embed * gate_weight

        # 5. 强制执行 L2 归一化，确保余弦相似度计算的稳定性
        fused_embed = F.normalize(fused_embed, p=2, dim=-1)

        # 6. 叠加类型偏置 (Job vs Author)
        type_labels = (node_ids >= self.ENTITY_OFFSET).long()
        fused_embed += self.type_embed(type_labels)

        return fused_embed