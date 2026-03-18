# -*- coding: utf-8 -*-
"""
KGAT-AX 精排模型：两阶段推荐架构中的「第二阶段深度精排器」。

设计依据：README 修改计划「KGAT-AX 在系统中的重新定位与结构升级」「KGAT-AX v2 修改方案」。
核心定位：对总召回输出的统一候选池进行精细排序，而非全库检索。
支持四分支融合：Graph Tower + Author Tower + Recall Tower + Interaction Tower，
并保留分项输出 s_graph / s_author / s_recall / s_interaction / final_score 便于消融与解释。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _L2_loss_mean(x):
    """L2 正则项：对 embedding 各维平方和求平均，用于 CF/KG 损失中的权重衰减。"""
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


# ---------------------------------------------------------------------------
# 四分支输入字段维度常量（与 README 5.2 / 5.3 对齐，便于与 data_loader 一致）
# ---------------------------------------------------------------------------
# Author Tower：README 建议的 12 维作者显式指标（log/归一化后）
AUTHOR_AUX_COLUMNS = [
    "author_h_index_log",
    "author_citations_log",
    "author_works_log",
    "author_recent_5y_works_log",
    "author_recent_5y_citations_log",
    "best_paper_citations_log",
    "top3_paper_citations_mean_log",
    "paper_recency_score",
    "best_institution_authority",
    "avg_institution_authority",
    "best_source_authority",
    "avg_source_authority",
]
N_AUTHOR_AUX_DEFAULT = len(AUTHOR_AUX_COLUMNS)  # 12

# Recall Tower：线上可复算的召回来源特征（来自 CandidatePool）
# 示例：from_vector, from_label, from_collab, path_count, vector_rank_norm, ...
N_RECALL_FEATURES_DEFAULT = 13

# Interaction Tower：query-author 交叉特征（岗位-作者匹配）
# 示例：topic_similarity, skill_coverage_ratio, domain_consistency, ...
N_INTERACTION_FEATURES_DEFAULT = 8


class Aggregator(nn.Module):
    """
    轻量化的图卷积聚合器：仅负责邻居信息聚合与非线性变换。
    支持 GCN / GraphSAGE / Bi-Interaction 三种聚合方式，用于 KGAT 的图传播层。
    """

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
        """执行一跳消息传递：side_embeddings = A_in @ ego_embeddings，再按类型聚合。"""
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
    KGAT-AX 主模型（含 v2 四分支扩展）。

    - 保留原有 KGAT 主干：图传播 + 3 维 AX 全息融合，用于 Graph Tower。
    - 可选扩展（README 5.2 / 5.3）：
      - Author Tower：作者显式指标（如 12 维 author_aux）→ 学术实力/活跃度/权威性；
      - Recall Tower：召回来源特征 → 多路命中、排名、分桶等；
      - Interaction Tower：query-author 交叉特征 → 主题匹配、技能覆盖等。
    - 融合层：concat(graph_repr_u, graph_repr_i, author_repr_i, recall_repr_ui, interaction_repr_ui) → MLP → final_score。
    - 当 n_author_aux / n_recall_features / n_interaction_features 均为 0 时，行为与旧版一致（仅图 + 3 维 AX）。
    """

    def __init__(self, args, n_users, n_total_nodes, n_relations, A_in=None):
        """
        参数：
            args: 需包含 embed_dim, relation_dim, conv_dim_list, n_aux_features(默认3),
                  以及可选的 n_author_aux, n_recall_features, n_interaction_features（为 0 则不加对应塔）.
            n_users: 用户（岗位）数量。
            n_total_nodes: 图中总节点数（含 Job + Entity）。
            n_relations: 关系类型数。
            A_in: 归一化邻接矩阵（稀疏），用于图传播。
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

        # 图塔输出拼接后的维度（用于 v2 的 graph_proj）
        self._graph_concat_dim = sum(self.conv_dim_list)

        # ---------- 1. Graph Tower：原有 AX 映射与图结构 ----------
        # 1.1 核心 AX 映射层：将 3 维学术特征映射到 Embedding 空间（用于全息融合）
        self.n_aux_features = getattr(args, 'n_aux_features', 3)
        self.aux_embed_layer = nn.Linear(self.n_aux_features, self.embed_dim)
        nn.init.xavier_uniform_(self.aux_embed_layer.weight)

        # 1.2 类型偏置层：区分 Job(0) 与 Author(1)，便于模型区分节点角色
        self.type_embed = nn.Embedding(2, self.embed_dim)
        nn.init.xavier_uniform_(self.type_embed.weight)

        # 1.3 实体/用户 Embedding 与关系 Embedding（TransR 等用）
        self.global_max_id = n_total_nodes
        self.entity_user_embed = nn.Embedding(self.global_max_id, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))
        self.W_a = nn.Parameter(torch.Tensor(self.relation_dim, 1))
        nn.init.xavier_uniform_(self.entity_user_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)
        nn.init.xavier_uniform_(self.W_a)

        # 1.4 堆叠聚合器层（图传播）
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

        # ---------- 2. 四分支 v2 扩展（可选） ----------
        self.n_author_aux = getattr(args, 'n_author_aux', 0)
        self.n_recall_features = getattr(args, 'n_recall_features', 0)
        self.n_interaction_features = getattr(args, 'n_interaction_features', 0)

        # 2.1 图表示投影：将多层 concat 的图嵌入投影到统一 tower 维度，便于与其它塔拼接
        self.graph_proj = nn.Linear(self._graph_concat_dim, self.embed_dim)
        nn.init.xavier_uniform_(self.graph_proj.weight)

        # 2.2 Author Tower：作者显式指标 → embed_dim（README 5.2 Author Tower）
        if self.n_author_aux > 0:
            self.author_tower = nn.Sequential(
                nn.Linear(self.n_author_aux, self.embed_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            )
            nn.init.xavier_uniform_(self.author_tower[0].weight)
        else:
            self.author_tower = None

        # 2.3 Recall Tower：召回来源特征 → embed_dim（README 5.2 Recall Tower）
        if self.n_recall_features > 0:
            self.recall_tower = nn.Sequential(
                nn.Linear(self.n_recall_features, self.embed_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            )
            nn.init.xavier_uniform_(self.recall_tower[0].weight)
        else:
            self.recall_tower = None

        # 2.4 Interaction Tower：query-author 交叉特征 → embed_dim（README 5.2 Interaction Tower）
        if self.n_interaction_features > 0:
            self.interaction_tower = nn.Sequential(
                nn.Linear(self.n_interaction_features, self.embed_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            )
            nn.init.xavier_uniform_(self.interaction_tower[0].weight)
        else:
            self.interaction_tower = None

        # 2.5 融合层：5 部分拼接 [graph_u, graph_i, author_i, recall_ui, interaction_ui] → final_score（README 5.3）
        self.fusion_mlp = nn.Sequential(
            nn.Linear(5 * self.embed_dim, self.embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, 1),
        )
        nn.init.xavier_uniform_(self.fusion_mlp[0].weight)
        nn.init.xavier_uniform_(self.fusion_mlp[3].weight)

    def _use_four_tower(self):
        """是否启用四分支融合（任一额外塔存在即启用）。"""
        return self.n_author_aux > 0 or self.n_recall_features > 0 or self.n_interaction_features > 0

    def holographic_fusion(self, entity_embed, aux_info, node_ids=None):
        """
        【统一对齐版】全息融合：语义相关性主导，学术影响力微调。
        作用：确保人才推荐首先基于专业对口（基准模长），再由学术影响力（AX）实现约 15% 以内的排名微调。
        用于 Graph Tower 内部的节点表示。
        """
        smooth_aux = torch.log1p(aux_info)
        gate_weight = torch.sigmoid(self.aux_embed_layer(smooth_aux)) * 0.15 + 1.0
        fused_embed = entity_embed * gate_weight
        fused_embed = F.normalize(fused_embed, p=2, dim=-1)
        if node_ids is not None:
            type_labels = (node_ids >= self.ENTITY_OFFSET).long()
            fused_embed = fused_embed + self.type_embed(type_labels)
        return fused_embed

    def calc_cf_embeddings(self, aux_info_all):
        """
        带身份标识与 AX 增强的图传播：得到所有节点的多层拼接表示。
        返回形状 [n_total_nodes, graph_concat_dim]，用于 CF 打分与 v2 的 graph_proj 输入。
        """
        device = self.entity_user_embed.weight.device
        all_ids = torch.arange(self.global_max_id).to(device)
        ego_embed = self.holographic_fusion(self.entity_user_embed.weight, aux_info_all, all_ids)
        all_embed = [ego_embed]
        for layer in self.aggregator_layers:
            ego_embed = layer(ego_embed, self.A_in)
            all_embed.append(F.normalize(ego_embed, p=2, dim=1))
        return torch.cat(all_embed, dim=1)

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids, aux_info_all):
        """CF 阶段 BPR 损失：正样本得分高于负样本。"""
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
        """KG 阶段 TransR 风格损失：正三元组得分高于负三元组。"""
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
        """批量更新图注意力权重（用于 TransR 边权重等）。"""
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
        """统一入口：按 mode 分发到 CF 损失、KG 损失、预测或注意力更新。"""
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        elif mode == 'train_kg':
            return self.calc_kg_loss(*input)
        elif mode == 'predict':
            return self.calc_score(*input)
        elif mode == 'update_att':
            return self.update_attention_batch(*input)
        raise ValueError(f"Unknown mode: {mode}")

    def calc_score(self, user_ids, item_ids, aux_info_all):
        """
        精排得分（兼容旧接口）：仅使用 Graph Tower，返回 [len(user_ids), len(item_ids)] 的得分矩阵。
        当未启用四分支时，线上/训练均使用此接口即可。
        """
        all_embed = self.calc_cf_embeddings(aux_info_all)
        return torch.matmul(all_embed[user_ids], all_embed[item_ids].transpose(0, 1))

    def calc_score_v2(
        self,
        user_ids,
        item_ids,
        aux_info_all,
        author_aux_item=None,
        recall_features=None,
        interaction_features=None,
    ):
        """
        四分支精排得分（README 5.3 / 5.4）：返回 final_score 与分项 s_graph, s_author, s_recall, s_interaction。

        参数：
            user_ids: [B1] 岗位 id。
            item_ids: [B2] 候选作者 id。
            aux_info_all: [n_total_nodes, n_aux_features] 全图 AX 特征（用于图塔）。
            author_aux_item: [B2, n_author_aux] 可选，候选作者的 12 维显式指标。
            recall_features: [B1, B2, n_recall] 可选，每对 (job, author) 的召回来源特征。
            interaction_features: [B1, B2, n_interaction] 可选，每对 (job, author) 的交叉特征。

        返回：
            dict: final_score [B1, B2], s_graph [B1, B2], s_author [B1, B2], s_recall [B1, B2], s_interaction [B1, B2]。
        若未启用四分支或未传入额外特征，则退化为图塔点积，分项中缺失的用 0 填充。
        """
        device = self.entity_user_embed.weight.device
        all_embed = self.calc_cf_embeddings(aux_info_all)
        graph_concat = all_embed
        graph_repr = self.graph_proj(graph_concat)
        graph_repr_u = graph_repr[user_ids]
        graph_repr_i = graph_repr[item_ids]
        B1, B2 = graph_repr_u.size(0), graph_repr_i.size(0)

        zero_embed = torch.zeros(B2, self.embed_dim, device=device, dtype=graph_repr.dtype)
        if self.author_tower is not None and author_aux_item is not None:
            author_repr_i = self.author_tower(author_aux_item)
        else:
            author_repr_i = zero_embed

        if self.recall_tower is not None and recall_features is not None:
            recall_flat = recall_features.view(-1, recall_features.size(-1))
            recall_repr_flat = self.recall_tower(recall_flat)
            recall_repr_ui = recall_repr_flat.view(B1, B2, self.embed_dim)
        else:
            recall_repr_ui = torch.zeros(B1, B2, self.embed_dim, device=device, dtype=graph_repr.dtype)

        if self.interaction_tower is not None and interaction_features is not None:
            inter_flat = interaction_features.view(-1, interaction_features.size(-1))
            inter_repr_flat = self.interaction_tower(inter_flat)
            interaction_repr_ui = inter_repr_flat.view(B1, B2, self.embed_dim)
        else:
            interaction_repr_ui = torch.zeros(B1, B2, self.embed_dim, device=device, dtype=graph_repr.dtype)

        # 分项标量（便于消融与解释）：s_graph 为图塔点积，其余为对应塔表示的 L2 范数或线性投影
        s_graph = torch.matmul(graph_repr_u, graph_repr_i.transpose(0, 1))
        s_author = author_repr_i.sum(dim=-1).unsqueeze(0).expand(B1, B2)
        s_recall = recall_repr_ui.sum(dim=-1)
        s_interaction = interaction_repr_ui.sum(dim=-1)

        # 融合：对每对 (u,i) 拼接 5 部分后过 MLP
        graph_u_expand = graph_repr_u.unsqueeze(1).expand(-1, B2, -1)
        graph_i_expand = graph_repr_i.unsqueeze(0).expand(B1, -1, -1)
        author_i_expand = author_repr_i.unsqueeze(0).expand(B1, -1, -1)
        fusion_input = torch.cat(
            [graph_u_expand, graph_i_expand, author_i_expand, recall_repr_ui, interaction_repr_ui],
            dim=-1,
        )
        final_score = self.fusion_mlp(fusion_input).squeeze(-1)

        return {
            "final_score": final_score,
            "s_graph": s_graph,
            "s_author": s_author,
            "s_recall": s_recall,
            "s_interaction": s_interaction,
        }

    def calc_cf_embeddings_subset(self, node_ids, aux_info_subset):
        """
        【优化版】局部图表示：仅对给定节点计算带 AX 门控的嵌入（用于线上精排子图）。
        逻辑：先确定专业对口（语义），再由学术指标做小幅优选；与 holographic_fusion 一致。
        """
        ego_embed = self.entity_user_embed(node_ids)
        smooth_aux = torch.log1p(aux_info_subset)
        gate_weight = torch.sigmoid(self.aux_embed_layer(smooth_aux)) * 0.2 + 0.8
        fused_embed = ego_embed * gate_weight
        fused_embed = F.normalize(fused_embed, p=2, dim=-1)
        type_labels = (node_ids >= self.ENTITY_OFFSET).long()
        fused_embed = fused_embed + self.type_embed(type_labels)
        return fused_embed
