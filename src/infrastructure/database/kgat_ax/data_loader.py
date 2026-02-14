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
        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.kg_file = os.path.join(self.data_dir, "kg_final.txt")

        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(os.path.join(self.data_dir, 'test.txt'))
        self.statistic_cf()
        self.load_auxiliary_info()

    def load_auxiliary_info(self):
        map_path = os.path.join(self.data_dir, "id_map.json")
        with open(map_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        entity_to_int = mapping['entity']

        num_total_entities = len(entity_to_int)
        features = np.zeros((num_total_entities, self.args.n_aux_features), dtype=np.float32)

        conn = sqlite3.connect(self.args.db_path)
        authors_df = pd.read_sql("SELECT author_id, h_index, cited_by_count, works_count FROM authors", conn)
        insts_df = pd.read_sql("SELECT inst_id, cited_by_count, works_count FROM institutions", conn)
        conn.close()

        # data_loader.py 约 35 行
        for df in [authors_df, insts_df]:
            for col in ['cited_by_count', 'works_count']:
                if col in df.columns:
                    # 关键：先对引用数取 log，再做归一化，防止“马太效应”淹没其他学者
                    df[col] = np.log1p(df[df[col] > 0][col])
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-9)
        authors_df['h_index'] = (authors_df['h_index'] - authors_df['h_index'].min()) / \
                                (authors_df['h_index'].max() - authors_df['h_index'].min() + 1e-9)

        for _, row in authors_df.iterrows():
            aid_str = str(row['author_id'])
            if aid_str in entity_to_int:
                features[entity_to_int[aid_str]] = [row['h_index'], row['cited_by_count'], row['works_count']]

        for _, row in insts_df.iterrows():
            iid_str = str(row['inst_id'])
            if iid_str in entity_to_int:
                features[entity_to_int[iid_str]] = [0.0, row['cited_by_count'], row['works_count']]

        self.aux_info_all = torch.from_numpy(features)
        self.n_users_entities = num_total_entities
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
        # 既然使用了统一 ID 映射，直接从 mapping 获取总数
        self.n_users_entities = self.aux_info_all.shape[0]
        # 这里的 n_users 和 n_items 仅用于兼容旧接口，实际应使用统一索引
        self.n_users = max(self.train_user_dict.keys()) + 1
        self.n_items = self.n_users_entities - self.n_users

    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python', on_bad_lines='skip')
        return kg_data.drop_duplicates()

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        """向量化加速版：一次性生成批量负样本，极大提升 CPU 效率"""
        exist_heads = list(kg_dict.keys())
        batch_head = np.random.choice(exist_heads, batch_size)

        h_ids, r_ids, pos_t_ids = [], [], []
        for h in batch_head:
            t, r = random.choice(kg_dict[h])
            h_ids.append(h)
            r_ids.append(r)
            pos_t_ids.append(t)

        # 批量负采样，忽略极低概率的碰撞
        neg_t_ids = np.random.randint(0, highest_neg_idx, size=(batch_size,))

        return (torch.LongTensor(h_ids), torch.LongTensor(r_ids),
                torch.LongTensor(pos_t_ids), torch.LongTensor(neg_t_ids),
                self.aux_info_all[h_ids], self.aux_info_all[pos_t_ids], self.aux_info_all[neg_t_ids])

    def generate_cf_batch(self, user_dict, batch_size):
        """向量化加速版：减少 Python 原生循环次数"""
        exist_users = list(user_dict.keys())
        batch_user = np.random.choice(exist_users, batch_size)

        batch_pos_item = [random.choice(user_dict[u]) for u in batch_user]
        batch_neg_item = np.random.randint(0, self.n_users_entities, size=(batch_size,))

        return torch.LongTensor(batch_user), torch.LongTensor(batch_pos_item), torch.LongTensor(batch_neg_item)

class DataLoaderKGAT(DataLoaderBase):
    def __init__(self, args, logging):
        super().__init__(args, logging)
        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)
        self.create_adjacency_dict()
        self.create_laplacian_dict()

    def construct_data(self, kg_data):
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy().rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True)

        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1

        self.train_user_dict = {k: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

        self.n_users_entities = self.aux_info_all.shape[0]
        self.n_users = max(self.train_user_dict.keys()) + 1
        self.n_entities = self.n_users_entities - self.n_users

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)
        h_list, t_list, r_list = [], [], []

        for row in kg_data.itertuples():
            h, r, t = row.h, row.r, row.t
            h_list.append(h); t_list.append(t); r_list.append(r)
            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list, self.t_list, self.r_list = torch.LongTensor(h_list), torch.LongTensor(t_list), torch.LongTensor(r_list)
        self.n_kg_train = len(kg_data)

    def create_adjacency_dict(self):
        self.adjacency_dict = {}
        for r, ht_list in self.train_relation_dict.items():
            rows, cols = [e[0] for e in ht_list], [e[1] for e in ht_list]
            self.adjacency_dict[r] = sp.coo_matrix(([1.0]*len(rows), (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))

    # data_loader.py 约 138 行开始
    def create_laplacian_dict(self):
        def norm_lap(adj):
            rowsum = np.array(adj.sum(axis=1))
            d_inv = np.power(rowsum, -1.0).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            # 核心：必须显式转为 float32，节省一半空间
            return sp.diags(d_inv).dot(adj).astype(np.float32).tocoo()

        print("[*] 正在分步构建并转换拉普拉斯矩阵...")
        total_adj = None
        # 迭代累加并手动释放内存
        for r in list(self.adjacency_dict.keys()):
            adj = self.adjacency_dict[r]
            lap = norm_lap(adj)
            if total_adj is None:
                total_adj = lap
            else:
                total_adj = total_adj + lap

            # 极重要：处理完一个关系就删一个，手动 GC
            self.adjacency_dict[r] = None
            del adj, lap
            gc.collect()

        print("[*] 正在将 Scipy 转换为 Torch 稀疏张量...")
        # 转换后立即释放 total_adj
        self.A_in = self.convert_coo2tensor(total_adj)
        del total_adj
        gc.collect()

    def convert_coo2tensor(self, coo):
        indices = np.vstack((coo.row, coo.col))
        return torch.sparse_coo_tensor(torch.LongTensor(indices), torch.FloatTensor(coo.data), torch.Size(coo.shape))

    def print_info(self, logging):
        logging.info(f'n_users: {self.n_users} | n_entities: {self.n_entities} | n_kg_train: {self.n_kg_train}')