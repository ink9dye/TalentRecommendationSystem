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
    KGAT-AX 主模型：管理全局 Embedding 与 多层聚合
    """

    def __init__(self, args, n_users, n_total_nodes, n_relations, A_in=None):
        super(KGAT, self).__init__()
        self.n_users = n_users
        self.n_relations = n_relations
        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        self.aggregation_type = args.aggregation_type
        self.ENTITY_OFFSET = args.ENTITY_OFFSET

        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))
        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        # 1. AX 特征融合层
        self.n_aux_features = getattr(args, 'n_aux_features', 3)
        self.aux_embed_layer = nn.Linear(self.n_aux_features, self.embed_dim)
        nn.init.xavier_uniform_(self.aux_embed_layer.weight)

        # 2. 类型偏置层
        self.type_embed = nn.Embedding(2, self.embed_dim)
        nn.init.xavier_uniform_(self.type_embed.weight)

        # --- 核心修改点：维度对齐 ---
        # 直接使用 data.n_users_entities (1,870,366)，不再手动计算
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
            # 校验邻接矩阵维度是否与 Embedding 层严格一致
            if A_in.shape[0] != self.global_max_id:
                raise ValueError(f"维度不匹配: 邻接矩阵({A_in.shape[0]}) != global_max_id({self.global_max_id})")
            self.register_buffer('A_in', A_in)


class KGAT(nn.Module):
    def __init__(self, args, n_users, n_total_nodes, n_relations, A_in=None):
        """
        修正版构造函数：直接使用 n_total_nodes 对齐 Embedding 空间
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

        # 1. 核心 AX 映射层
        self.n_aux_features = getattr(args, 'n_aux_features', 3)
        self.aux_embed_layer = nn.Linear(self.n_aux_features, self.embed_dim)
        nn.init.xavier_uniform_(self.aux_embed_layer.weight)

        # 2. 类型偏置层
        self.type_embed = nn.Embedding(2, self.embed_dim)
        nn.init.xavier_uniform_(self.type_embed.weight)

        # 3. 核心 Embedding 空间：直接使用传入的总节点数，防止维度漂移
        self.global_max_id = n_total_nodes
        self.entity_user_embed = nn.Embedding(self.global_max_id, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)

        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))
        self.W_a = nn.Parameter(torch.Tensor(self.relation_dim, 1))

        nn.init.xavier_uniform_(self.entity_user_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)
        nn.init.xavier_uniform_(self.W_a)

        # 4. 聚合器层
        self.aggregator_layers = nn.ModuleList([
            Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type)
            for k in range(self.n_layers)
        ])

        if A_in is not None:
            # 校验邻接矩阵维度是否与 Embedding 层严格匹配
            if A_in.shape[0] != self.global_max_id:
                raise ValueError(f"维度不匹配: 邻接矩阵({A_in.shape[0]}) != global_max_id({self.global_max_id})")
            self.register_buffer('A_in', A_in)
        else:
            self.A_in = None

    # src/infrastructure/database/kgat_ax/model.py

    def holographic_fusion(self, entity_embed, aux_info, node_ids=None):
        """
        【深度修复版】非线性门控融合：增强高权重专家的区分度
        """
        # 1. 使用 Sigmoid 门控：将学术特征映射为 [0.5, 1.5] 的缩放系数
        # 学术指标越高，gate_weight 越接近 1.5；反之越接近 0.5
        gate_weight = torch.sigmoid(self.aux_embed_layer(aux_info)) + 0.5

        # 2. 采用乘法门控融合 (Multiplicative Gating)
        # 这会直接放大/缩小语义向量的强度，让学术大牛在点积计算中占优
        fused_embed = entity_embed * gate_weight

        # 3. 强制执行 L2 归一化，保持向量空间稳定性
        fused_embed = F.normalize(fused_embed, p=2, dim=-1)

        # 4. 叠加身份标签（Job vs Author）
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
        【核心修复】局部特征提取，解决 187万节点导致的内存溢出。
        node_ids: 待计算的节点 ID 序列 (Job 锚点 + Author 候选人)
        aux_info_subset: 对应这些节点的 AX 特征张量
        """
        # 1. 提取基础 Embedding
        ego_embed = self.entity_user_embed(node_ids)

        # 2. 执行全息融合 (残差加法 + 归一化)
        # 这里的计算量仅为 node_ids 的长度，不再分配全量 187万 的内存
        aux_bias = torch.tanh(self.aux_embed_layer(aux_info_subset))
        fused_embed = ego_embed + aux_bias

        # 3. 归一化处理，防止模长爆炸
        fused_embed = F.normalize(fused_embed, p=2, dim=-1)

        # 4. 叠加类型偏置
        type_labels = (node_ids >= self.ENTITY_OFFSET).long()
        fused_embed += self.type_embed(type_labels)

        return fused_embed