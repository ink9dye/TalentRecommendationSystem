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
        self.conv_dim_list = [args.embed_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))
        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        # 核心 AX 映射层
        self.n_aux_features = getattr(args, 'n_aux_features', 3)
        self.aux_embed_layer = nn.Linear(self.n_aux_features, self.embed_dim)
        nn.init.xavier_uniform_(self.aux_embed_layer.weight)

        self.entity_user_embed = nn.Embedding(self.n_entities + self.n_users, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))

        # 初始化注意力向量
        self.W_a = nn.Parameter(torch.Tensor(self.relation_dim, 1))

        nn.init.xavier_uniform_(self.entity_user_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)
        nn.init.xavier_uniform_(self.W_a)

        self.aggregator_layers = nn.ModuleList([
            Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type)
            for k in range(self.n_layers)
        ])

        # 适配新版 Sparse Tensor
        self.A_in = nn.Parameter(torch.sparse_coo_tensor(
            size=(self.n_users + self.n_entities, self.n_users + self.n_entities),
            dtype=torch.float32
        ))
        if A_in is not None: self.A_in.data = A_in
        self.A_in.requires_grad = False

    def holographic_fusion(self, entity_embed, aux_info):
        """全息嵌入融合：将归一化后的 H-index 等指标注入实体表示"""
        aux_vector = torch.tanh(self.aux_embed_layer(aux_info))
        return torch.mul(entity_embed, 1 + aux_vector)

    def calc_cf_embeddings(self, aux_info_all):
        """带 AX 增强的消息传递层"""
        ego_embed = self.holographic_fusion(self.entity_user_embed.weight, aux_info_all)
        all_embed = [ego_embed]

        for layer in self.aggregator_layers:
            ego_embed = layer(ego_embed, self.A_in)
            all_embed.append(F.normalize(ego_embed, p=2, dim=1))

        return torch.cat(all_embed, dim=1)

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids, aux_info_all):
        all_embed = self.calc_cf_embeddings(aux_info_all)
        u_e, p_e, n_e = all_embed[user_ids], all_embed[item_pos_ids], all_embed[item_neg_ids]
        pos_score, neg_score = torch.sum(u_e * p_e, 1), torch.sum(u_e * n_e, 1)
        cf_loss = torch.mean((-1.0) * F.logsigmoid(pos_score - neg_score))
        l2_loss = _L2_loss_mean(u_e) + _L2_loss_mean(p_e) + _L2_loss_mean(n_e)
        return cf_loss + self.cf_l2loss_lambda * l2_loss

    def calc_kg_loss(self, h, r, pos_t, neg_t, h_aux, pos_t_aux, neg_t_aux):
        """全息增强三元组损失"""
        h_e, p_t_e, n_t_e = self.entity_user_embed(h), self.entity_user_embed(pos_t), self.entity_user_embed(neg_t)
        h_e = self.holographic_fusion(h_e, h_aux)
        p_t_e = self.holographic_fusion(p_t_e, pos_t_aux)
        n_t_e = self.holographic_fusion(n_t_e, neg_t_aux)

        r_e, W_r = self.relation_embed(r), self.trans_M[r]
        r_h = torch.bmm(h_e.unsqueeze(1), W_r).squeeze(1)
        r_p = torch.bmm(p_t_e.unsqueeze(1), W_r).squeeze(1)
        r_n = torch.bmm(n_t_e.unsqueeze(1), W_r).squeeze(1)

        pos_score = torch.sum(torch.pow(r_h + r_e - r_p, 2), 1)
        neg_score = torch.sum(torch.pow(r_h + r_e - r_n, 2), 1)
        kg_loss = torch.mean((-1.0) * F.logsigmoid(neg_score - pos_score))
        l2_loss = _L2_loss_mean(r_h) + _L2_loss_mean(r_e) + _L2_loss_mean(r_p) + _L2_loss_mean(r_n)
        return kg_loss + self.kg_l2loss_lambda * l2_loss

    def update_attention_batch(self, h_list, t_list, r_list):
        """计算注意力得分，用于更新邻接矩阵权重"""
        h_e = self.entity_user_embed(h_list)
        t_e = self.entity_user_embed(t_list)
        r_e = self.relation_embed(r_list)
        W_r = self.trans_M[r_list]

        h_e = torch.bmm(h_e.unsqueeze(1), W_r).squeeze(1)
        t_e = torch.bmm(t_e.unsqueeze(1), W_r).squeeze(1)

        att_score = torch.sum((h_e + r_e) * torch.tanh(t_e), dim=1, keepdim=True)
        att_score = torch.sigmoid(torch.matmul(att_score, self.W_a.mean(0, keepdim=True).T))
        return att_score

    def forward(self, *input, mode):
        if mode == 'train_cf': return self.calc_cf_loss(*input)
        if mode == 'train_kg': return self.calc_kg_loss(*input)
        if mode == 'predict': return self.calc_score(*input)
        if mode == 'update_att': return self.update_attention_batch(*input)

    def calc_score(self, user_ids, item_ids, aux_info_all):
        all_embed = self.calc_cf_embeddings(aux_info_all)
        return torch.matmul(all_embed[user_ids], all_embed[item_ids].transpose(0, 1))