import os
import gc
import random
import collections
import torch
import sqlite3
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

class DataLoaderBase(object):
    def __init__(self, args, logging):
        self.args = args
        # 兼容 config.py 中的路径定义
        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.kg_file = os.path.join(self.data_dir, "kg_final.txt")

        # 1. 加载协同过滤数据
        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(os.path.join(self.data_dir, 'test.txt'))
        self.statistic_cf()

        # 2. 加载学术指标 (AX)
        self.load_auxiliary_info()

    def load_auxiliary_info(self):
        """最终修正版：确保特征矩阵大小与全局映射完全对齐，并适配大内存数据读取"""
        map_path = os.path.join(self.data_dir, "id_map.json")
        with open(map_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        entity_to_int = mapping['entity']

        # [cite_start]以映射表长度为准初始化矩阵，彻底解决 IndexError [cite: 149]
        num_total_entities = len(entity_to_int)
        features = np.zeros((num_total_entities, self.args.n_aux_features), dtype=np.float32)

        # [cite_start]从 SQLite 加载真实学术指标 [cite: 156]
        conn = sqlite3.connect(self.args.db_path)
        authors_df = pd.read_sql("SELECT author_id, h_index, cited_by_count, works_count FROM authors", conn)
        insts_df = pd.read_sql("SELECT inst_id, cited_by_count, works_count FROM institutions", conn)
        conn.close()

        # [cite_start]执行归一化 [cite: 157]
        for df in [authors_df, insts_df]:
            for col in ['cited_by_count', 'works_count']:
                if col in df.columns:
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-9)
        authors_df['h_index'] = (authors_df['h_index'] - authors_df['h_index'].min()) / \
                                (authors_df['h_index'].max() - authors_df['h_index'].min() + 1e-9)

        # [cite_start]填充特征：利用 entity_to_int 找到模型内部索引执行全息增强 [cite: 157]
        for _, row in authors_df.iterrows():
            aid_str = str(row['author_id'])
            if aid_str in entity_to_int:
                idx = entity_to_int[aid_str]
                features[idx] = [row['h_index'], row['cited_by_count'], row['works_count']]

        for _, row in insts_df.iterrows():
            iid_str = str(row['inst_id'])
            if iid_str in entity_to_int:
                idx = entity_to_int[iid_str]
                features[idx] = [0.0, row['cited_by_count'], row['works_count']]

        self.aux_info_all = torch.from_numpy(features)
        # 更新类属性，确保后续逻辑同步
        self.n_users_entities = num_total_entities

        # [cite_start]显式清理大数据对象，释放 RAM [cite: 151]
        del authors_df, insts_df, mapping
        gc.collect()

    def load_cf(self, filename):
        user_ids, item_ids = [], []
        with open(filename, 'r') as f:
            for line in f.readlines():
                l = line.strip().split(' ')
                if len(l) == 0: continue
                u_id = int(l[0])
                for i_id in l[1:]:
                    user_ids.append(u_id)
                    item_ids.append(int(i_id))

        user_dict = collections.defaultdict(list)
        for u, i in zip(user_ids, item_ids):
            user_dict[u].append(i)

        return (np.array(user_ids), np.array(item_ids)), user_dict

    def statistic_cf(self):
        self.n_users = max(self.cf_train_data[0]) + 1
        self.n_items = max(self.cf_train_data[1]) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])
        self.n_users_entities = self.n_users + self.n_items

    def load_kg(self, filename):
        # [cite_start]使用空格作为分隔符读取文本三元组 [cite: 358]
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python', on_bad_lines='skip')
        return kg_data.drop_duplicates()

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        exist_heads = list(kg_dict.keys())
        batch_head = random.sample(exist_heads, batch_size) if batch_size <= len(exist_heads) else \
            [random.choice(exist_heads) for _ in range(batch_size)]

        h_ids, r_ids, pos_t_ids, neg_t_ids = [], [], [], []
        for h in batch_head:
            t, r = random.choice(kg_dict[h])
            h_ids.append(h)
            r_ids.append(r)
            pos_t_ids.append(t)

            # 负采样逻辑
            while True:
                neg_t = random.randint(0, highest_neg_idx - 1)
                if neg_t not in [x[0] for x in kg_dict[h]]:
                    break
            neg_t_ids.append(neg_t)

        h_ids, r_ids = torch.LongTensor(h_ids), torch.LongTensor(r_ids)
        pos_t_ids, neg_t_ids = torch.LongTensor(pos_t_ids), torch.LongTensor(neg_t_ids)

        # [cite_start]提取对应的 AX 指标特征 [cite: 10]
        h_aux = self.aux_info_all[h_ids]
        pos_t_aux = self.aux_info_all[pos_t_ids]
        neg_t_aux = self.aux_info_all[neg_t_ids]

        return h_ids, r_ids, pos_t_ids, neg_t_ids, h_aux, pos_t_aux, neg_t_aux

    def generate_cf_batch(self, user_dict, batch_size):
        exist_users = list(user_dict.keys())
        batch_user = random.sample(exist_users, batch_size) if batch_size <= len(exist_users) else \
            [random.choice(exist_users) for _ in range(batch_size)]

        batch_pos_item, batch_neg_item = [], []
        for u in batch_user:
            pos_items = user_dict[u]
            batch_pos_item.append(random.choice(pos_items))
            while True:
                neg_item = random.randint(0, self.n_items - 1)
                if neg_item not in pos_items:
                    break
            batch_neg_item.append(neg_item)

        return torch.LongTensor(batch_user), torch.LongTensor(batch_pos_item), torch.LongTensor(batch_neg_item)


class DataLoaderKGAT(DataLoaderBase):
    def __init__(self, args, logging):
        super().__init__(args, logging)
        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)

        self.laplacian_type = args.laplacian_type
        self.create_adjacency_dict()
        self.create_laplacian_dict()

    def construct_data(self, kg_data):
        """
        显式处理 ID 偏移并严格对齐全息增强所需的矩阵边界
        [cite_start]遵循 KGAT-AX 标准，集成协同过滤交互与知识图谱结构 [cite: 7, 339, 403]
        """
        # [cite_start]1. 构建逆向三元组增加图稠密度 [cite: 358]
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True)

        # [cite_start]2. 重新映射 ID 以区分 User 和 Entity [cite: 405]
        # 为 interact 交互关系预留关系 ID 0 和 1
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1

        # --- 维度对齐核心修复 ---
        # 基础实体数取目前 KG 数据中的最大编号 + 1
        self.n_entities = max(kg_data['h'].max(), kg_data['t'].max()) + 1

        # [cite_start]映射交互数据，为 User ID 增加偏移量 (Offset) 以集成 CKG [cite: 339, 404]
        # [cite_start]偏移量通常设为 n_entities，使用户 ID 紧随实体 ID 之后 [cite: 405]
        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in
                                self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in
                               self.test_user_dict.items()}

        # [cite_start]3. 将 CF 交互边作为特殊关系注入协作知识图谱 (CKG) [cite: 339, 403]
        cf_h = list(self.train_user_dict.keys())
        cf_t = [random.choice(self.train_user_dict[u]) for u in cf_h]
        cf_r = [0] * len(cf_h)
        cf_df = pd.DataFrame({'h': cf_h, 'r': cf_r, 't': cf_t})

        self.kg_train_data = pd.concat([kg_data, cf_df], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # === 核心对齐补丁：确保模型 Embedding 层与 AX 特征矩阵维度严格相等 ===
        # [cite_start]必须取特征矩阵 aux_info_all 的实际行数作为最终总节点数 [cite: 157]
        # [cite_start]这防止了 holographic_fusion 执行 Hadamard 积时出现维度不匹配错误 [cite: 10, 96]
        self.n_users_entities = self.aux_info_all.shape[0]
        # [cite_start]反向推算 n_entities，确保 trainer.py 初始化模型时分配正确的 Embedding 大小 [cite: 421, 70]
        self.n_entities = self.n_users_entities - self.n_users

        # [cite_start]构造字典用于递归消息传递与注意力计算 [cite: 71, 358]
        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)
        h_list, t_list, r_list = [], [], []

        for row in self.kg_train_data.itertuples():
            h, r, t = row.h, row.r, row.t
            h_list.append(h); t_list.append(t); r_list.append(r)
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        # [cite_start]转换为 LongTensor 供模型更新注意力权重模式 (update_att) 使用 [cite: 359, 451]
        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def convert_coo2tensor(self, coo):
        indices = np.vstack((coo.row, coo.col))
        return torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(coo.data), torch.Size(coo.shape))

    def create_adjacency_dict(self):

        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows = [e[0] for e in ht_list]
            cols = [e[1] for e in ht_list]
            vals = [1.0] * len(rows)
            # 使用修正后的 n_users_entities 确保不越界
            adj = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
            self.adjacency_dict[r] = adj

    def create_laplacian_dict(self):

        def random_walk_norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))
            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0
            return sp.diags(d_inv).dot(adj).tocoo()

        # 计算并叠加归一化矩阵
        self.laplacian_dict = {r: random_walk_norm_lap(adj) for r, adj in self.adjacency_dict.items()}
        A_in = sum(self.laplacian_dict.values())
        self.A_in = self.convert_coo2tensor(A_in.tocoo())

        # [cite_start]内存防御：计算完 A_in 后清理原始邻接矩阵 [cite: 151]
        self.adjacency_dict.clear()
        self.laplacian_dict.clear()
        gc.collect()

    def print_info(self, logging):
        logging.info(f'n_users: {self.n_users} | n_entities: {self.n_entities}')
        logging.info(f'n_relations: {self.n_relations} | n_kg_train: {self.n_kg_train}')
        logging.info(f'Final n_users_entities (Matrix Dim): {self.n_users_entities}')