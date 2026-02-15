import torch
import torch.nn as nn
import torch.nn.functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, aggregator_type):
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
        side_embeddings = torch.matmul(A_in, ego_embeddings)
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
    def __init__(self, args, n_users, n_entities, n_relations, A_in=None):
        super(KGAT, self).__init__()
        self.n_users = n_users
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        self.aggregation_type = args.aggregation_type
        self.ENTITY_OFFSET = args.ENTITY_OFFSET  # 关键偏移量

        # 解析卷积层配置
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))
        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        # 1. 核心 AX 映射层：增强学术特征到语义空间的对齐能力
        self.n_aux_features = getattr(args, 'n_aux_features', 3)
        self.aux_embed_layer = nn.Linear(self.n_aux_features, self.embed_dim)
        nn.init.xavier_uniform_(self.aux_embed_layer.weight)

        # 2. 类型偏置层 (Type Embedding)：解决节点类型分不清的核心插件
        # 0: User (Job/岗位), 1: Entity (Author/Work/人才等)
        self.type_embed = nn.Embedding(2, self.embed_dim)
        nn.init.xavier_uniform_(self.type_embed.weight)

        # 3. 核心 Embedding 空间
        self.global_max_id = args.ENTITY_OFFSET + n_entities
        self.entity_user_embed = nn.Embedding(self.global_max_id, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)

        # TransR 空间投影矩阵
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
            if A_in.shape[0] != self.global_max_id:
                raise ValueError(f"维度不匹配: 邻接矩阵({A_in.shape[0]}) != global_max_id({self.global_max_id})")
            self.register_buffer('A_in', A_in)
        else:
            self.A_in = None

    def holographic_fusion(self, entity_embed, aux_info, node_ids=None):
        """
        全息融合层：利用归一化学术特征作为缩放门控。
        适配更新：针对 build_feature_index.py 产生的对数平滑特征进行非线性对齐。
        """
        # 1. 应用学术特征缩放 (AX 增强)
        # 索引更新后，aux_info 已经是 [0, 1] 的归一化数值。
        # 使用 tanh 将其映射到语义空间，并通过 +1.0 偏置确保基础特征不被抹除。
        # 这样，学术表现优异（特征趋近1）的节点会在 Embedding 空间获得显著增强。
        res_weight = torch.tanh(self.aux_embed_layer(aux_info)) + 1.0
        fused_embed = entity_embed * res_weight

        # 2. 注入类型偏置 (Type Bias)
        # 索引更新明确了 ID 分区：[0, OFFSET) 为岗位，[OFFSET, MAX) 为实体。
        if node_ids is not None:
            # 动态生成类型 ID：岗位(Job)为 0，人才/论文等(Entity)为 1
            # 这一步能打破岗位节点与候选人节点在图谱结构上的语义对称性。
            type_labels = (node_ids >= self.ENTITY_OFFSET).long()

            # 注入身份 Embedding，帮助模型显式学习“推荐方向” (Job -> Author)
            fused_embed += self.type_embed(type_labels)

        return fused_embed

    def calc_cf_embeddings(self, aux_info_all):
        """带身份标识与 AX 增强的消息传递层"""
        device = self.entity_user_embed.weight.device
        all_ids = torch.arange(self.global_max_id).to(device)

        # 第一步：基础 Embedding + 身份偏置 + 学术融合
        ego_embed = self.holographic_fusion(self.entity_user_embed.weight, aux_info_all, all_ids)
        all_embed = [ego_embed]

        # 第二步：图卷积聚合
        for layer in self.aggregator_layers:
            ego_embed = layer(ego_embed, self.A_in)
            all_embed.append(F.normalize(ego_embed, p=2, dim=1))

        return torch.cat(all_embed, dim=1)

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids, aux_info_all):
        """计算带健壮性校验的 BPR 损失"""
        all_embed = self.calc_cf_embeddings(aux_info_all)

        u_e = all_embed[user_ids]
        p_e = all_embed[item_pos_ids]
        n_e = all_embed[item_neg_ids]

        pos_score = torch.sum(u_e * p_e, 1)
        neg_score = torch.sum(u_e * n_e, 1)

        cf_loss = torch.mean(F.softplus(neg_score - pos_score))
        l2_loss = _L2_loss_mean(u_e) + _L2_loss_mean(p_e) + _L2_loss_mean(n_e)

        return cf_loss + self.cf_l2loss_lambda * l2_loss

    def calc_kg_loss(self, h, r, pos_t, neg_t, h_aux, pos_t_aux, neg_t_aux):
        """KG 损失：同样注入类型偏置以保持特征一致性"""
        h_e = self.entity_user_embed(h)
        p_t_e = self.entity_user_embed(pos_t)
        n_t_e = self.entity_user_embed(neg_t)

        h_e = self.holographic_fusion(h_e, h_aux, h)
        p_t_e = self.holographic_fusion(p_t_e, pos_t_aux, pos_t)
        n_t_e = self.holographic_fusion(n_t_e, neg_t_aux, neg_t)

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
        h_e = self.entity_user_embed(h_list)
        t_e = self.entity_user_embed(t_list)
        r_e = self.relation_embed(r_list)
        W_r = self.trans_M[r_list]

        # 注意力计算中也需考虑节点身份
        # 这里的身份信息已隐含在 entity_user_embed 的训练过程中
        h_e = torch.bmm(h_e.unsqueeze(1), W_r).squeeze(1)
        t_e = torch.bmm(t_e.unsqueeze(1), W_r).squeeze(1)

        att_score = torch.sum((h_e + r_e) * torch.tanh(t_e), dim=1, keepdim=True)
        att_score = torch.sigmoid(torch.matmul(att_score, self.W_a.mean(0, keepdim=True).T))
        return att_score

    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        elif mode == 'train_kg':
            return self.calc_kg_loss(*input)
        elif mode == 'predict':
            return self.calc_score(*input)
        elif mode == 'update_att':
            return self.update_attention_batch(*input)

    def calc_score(self, user_ids, item_ids, aux_info_all):
        all_embed = self.calc_cf_embeddings(aux_info_all)
        u_e = all_embed[user_ids]
        i_e = all_embed[item_ids]
        return torch.matmul(u_e, i_e.transpose(0, 1))