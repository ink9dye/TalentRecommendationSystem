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

        # 2. 核心修复：基于物理最大索引开辟 Embedding 空间
        # 确保 global_max_id 覆盖 [0, user_count) 和 [OFFSET, OFFSET + n_entities)
        self.global_max_id = args.ENTITY_OFFSET + n_entities

        # 初始化实体与用户 Embedding (物理维度: global_max_id)
        self.entity_user_embed = nn.Embedding(self.global_max_id, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)

        # TransR 空间投影矩阵: [关系数, 实体空间维度, 关系空间维度]
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))

        # 3. 初始化注意力向量与各层权重
        self.W_a = nn.Parameter(torch.Tensor(self.relation_dim, 1))

        nn.init.xavier_uniform_(self.entity_user_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.trans_M)
        nn.init.xavier_uniform_(self.W_a)

        # 4. 构建聚合器层
        self.aggregator_layers = nn.ModuleList([
            Aggregator(self.conv_dim_list[k], self.conv_dim_list[k + 1], self.mess_dropout[k], self.aggregation_type)
            for k in range(self.n_layers)
        ])

        # 5. 适配 Sparse Tensor 邻接矩阵
        if A_in is not None:
            # 哨兵校验：确保矩阵维度与模型声明一致
            if A_in.shape[0] != self.global_max_id:
                raise ValueError(f"维度不匹配: 邻接矩阵维度({A_in.shape[0]}) != global_max_id({self.global_max_id})")

            # 注册为 buffer，跟随 model.to(device) 自动移动
            self.register_buffer('A_in', A_in)
        else:
            self.A_in = None

    def holographic_fusion(self, entity_embed, aux_info):
        """
        全息融合层：利用学术特征作为缩放门控
        res_weight 范围 (0, 2)，其中 1.0 表示学术表现平平，保持原样
        """
        # 确保 aux_info 的 device 与 entity_embed 一致
        res_weight = torch.tanh(self.aux_embed_layer(aux_info)) + 1.0
        return entity_embed * res_weight

    def calc_cf_embeddings(self, aux_info_all):
        """带 AX 增强的消息传递层"""
        # 第一步：基础 Embedding 与 学术特征融合
        ego_embed = self.holographic_fusion(self.entity_user_embed.weight, aux_info_all)
        all_embed = [ego_embed]

        # 第二步：多层图卷积聚合
        for layer in self.aggregator_layers:
            ego_embed = layer(ego_embed, self.A_in)
            all_embed.append(F.normalize(ego_embed, p=2, dim=1))

        # 第三步：多层特征拼接（Symmetry Breaking）
        return torch.cat(all_embed, dim=1)

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids, aux_info_all):
        """
        计算协同过滤 BPR 损失，增加工业级健壮性检测
        """
        # --- 1. 维度与类型哨兵检查 ---
        # 确保输入 ID 向量长度完全一致
        if not (user_ids.shape[0] == item_pos_ids.shape[0] == item_neg_ids.shape[0]):
            raise RuntimeError(f"训练 Batch 维度不匹配: "
                               f"User({user_ids.shape[0]}), "
                               f"Pos({item_pos_ids.shape[0]}), "
                               f"Neg({item_neg_ids.shape[0]})")

        # --- 2. 消息传递与特征融合 ---
        # 计算包含 AX 学术特征增强的全量 Embedding
        all_embed = self.calc_cf_embeddings(aux_info_all)

        # --- 3. 索引安全校验 ---
        # 防止由于 id_map.json 配置错误导致的 Embedding 索引越界
        max_id = all_embed.shape[0]
        if user_ids.max() >= max_id or item_pos_ids.max() >= max_id or item_neg_ids.max() >= max_id:
            raise IndexError(f"输入 ID 超出 Embedding 矩阵范围 (MaxID: {max_id})。请检查 ID 分区逻辑。")

        # --- 4. 提取对应的向量表示 ---
        u_e = all_embed[user_ids]
        p_e = all_embed[item_pos_ids]
        n_e = all_embed[item_neg_ids]

        # --- 5. 评分计算与 BPR 损失 ---
        # 计算用户与正/负样本的向量点积得分
        pos_score = torch.sum(u_e * p_e, 1)
        neg_score = torch.sum(u_e * n_e, 1)

        # 使用 Softplus 计算平滑的 BPR 损失（等价于 -log_sigmoid(pos - neg)）
        cf_loss = torch.mean(F.softplus(neg_score - pos_score))

        # --- 6. 正则化与数值自检 ---
        # 计算 L2 范数均值以防止过拟合
        l2_loss = _L2_loss_mean(u_e) + _L2_loss_mean(p_e) + _L2_loss_mean(n_e)

        total_loss = cf_loss + self.cf_l2loss_lambda * l2_loss

        # 最终哨兵：检测梯度爆炸或数据污染导致的 NaN
        if torch.isnan(total_loss):
            raise ValueError("检测到 CF Loss 为 NaN！请检查学习率配置或 aux_info_all 的归一化状态。")

        return total_loss

    def calc_kg_loss(self, h, r, pos_t, neg_t, h_aux, pos_t_aux, neg_t_aux):
        """全息增强三元组损失 (TransR 空间投影)"""
        h_e = self.entity_user_embed(h)
        p_t_e = self.entity_user_embed(pos_t)
        n_t_e = self.entity_user_embed(neg_t)

        # 融合 AX 特征到三元组学习中
        h_e = self.holographic_fusion(h_e, h_aux)
        p_t_e = self.holographic_fusion(p_t_e, pos_t_aux)
        n_t_e = self.holographic_fusion(n_t_e, neg_t_aux)

        r_e = self.relation_embed(r)
        W_r = self.trans_M[r]

        # 投影到关系子空间
        r_h = torch.bmm(h_e.unsqueeze(1), W_r).squeeze(1)
        r_p = torch.bmm(p_t_e.unsqueeze(1), W_r).squeeze(1)
        r_n = torch.bmm(n_t_e.unsqueeze(1), W_r).squeeze(1)

        # TransR 欧式距离得分
        pos_score = torch.sum(torch.pow(r_h + r_e - r_p, 2), 1)
        neg_score = torch.sum(torch.pow(r_h + r_e - r_n, 2), 1)

        kg_loss = torch.mean(F.softplus(pos_score - neg_score))
        l2_loss = _L2_loss_mean(r_h) + _L2_loss_mean(r_e) + _L2_loss_mean(r_p) + _L2_loss_mean(r_n)
        return kg_loss + self.kg_l2loss_lambda * l2_loss

    def update_attention_batch(self, h_list, t_list, r_list):
        """计算注意力得分（知识感知注意力机制）"""
        h_e = self.entity_user_embed(h_list)
        t_e = self.entity_user_embed(t_list)
        r_e = self.relation_embed(r_list)
        W_r = self.trans_M[r_list]

        h_e = torch.bmm(h_e.unsqueeze(1), W_r).squeeze(1)
        t_e = torch.bmm(t_e.unsqueeze(1), W_r).squeeze(1)

        # 注意力权重公式：A(h,r,t) = (Wh + Wr) * tanh(Wt)
        att_score = torch.sum((h_e + r_e) * torch.tanh(t_e), dim=1, keepdim=True)
        # 映射到 [0, 1] 空间并应用 W_a
        att_score = torch.sigmoid(torch.matmul(att_score, self.W_a.mean(0, keepdim=True).T))
        return att_score

    def forward(self, *input, mode):
        """统一分发层，处理变长参数"""
        if mode == 'train_cf':
            # 期望输入: (user_ids, item_pos_ids, item_neg_ids, aux_info_all)
            return self.calc_cf_loss(*input)
        elif mode == 'train_kg':
            # 期望输入: (h, r, pos_t, neg_t, h_aux, pos_t_aux, neg_t_aux)
            return self.calc_kg_loss(*input)
        elif mode == 'predict':
            # 期望输入: (user_ids, item_ids, aux_info_all)
            return self.calc_score(*input)
        elif mode == 'update_att':
            # 期望输入: (h_list, t_list, r_list)
            return self.update_attention_batch(*input)

    def calc_score(self, user_ids, item_ids, aux_info_all):
        """推理阶段：计算用户与候选实体的匹配度"""
        # 注意：在大型评估中，推荐在 trainer.py 中预计算好 all_embed 后直接矩阵乘法
        # 此处保留接口供单次推理使用
        all_embed = self.calc_cf_embeddings(aux_info_all)
        u_e = all_embed[user_ids]
        i_e = all_embed[item_ids]
        return torch.matmul(u_e, i_e.transpose(0, 1))