import os
import gc
import random
import collections
import torch
import json
import sqlite3
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
from config import KGATAX_TRAIN_DATA_DIR, FEATURE_INDEX_PATH


class DataLoaderKGAT(object):
    """
    适配四级梯度精排逻辑的 KGAT-AX 数据加载器。
    职责：解析分层训练集、加载归一化学术特征 (AX)、执行索引驱动的子图采样 。
    """

    def __init__(self, args, logging):
        self.args = args
        self.logging = logging

        # 路径初始化
        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.test_file = os.path.join(self.data_dir, 'test.txt')
        self.kg_file = os.path.join(self.data_dir, "kg_final.txt")
        self.test_batch_size = getattr(args, 'test_batch_size', 1024)

        # 1. 载入全局映射与分区信息（奠定 ID 空间基础）
        self.load_id_mapping()

        # 2. 载入协同过滤交互数据 (支持四级梯度解析)
        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
        self.n_cf_train = len(self.cf_train_data[0])
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)

        # 3. 提取学术增强特征 (AX) - 直接对接 build_feature_index.py 的产物
        self.load_auxiliary_info()

        # 4. 构建图结构 - 利用 build_kg_index.py 建立的 SQLite 索引执行子图采样
        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)

        # 5. 构建拉普拉斯矩阵 (含自动缓存逻辑)
        self.create_laplacian_dict()

        # 6. 最终自检
        self.sanity_check()

    def load_id_mapping(self):
        """加载 ID 压缩映射，确立 User 与 Entity 的物理边界"""
        map_path = os.path.join(self.data_dir, "id_map.json")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"未发现映射文件: {map_path}，请先运行 generate_training_data.py。")

        with open(map_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        self.n_users_entities = mapping.get("total_nodes")
        self.n_users = mapping.get("user_count", 0)
        self.n_entities = mapping.get("entity_count", 0)
        self.ENTITY_OFFSET = mapping.get("offset", 0)

        self.logging.info(
            f"[*] ID 空间映射成功: User(0-{self.ENTITY_OFFSET - 1}), Entity({self.ENTITY_OFFSET}-{self.n_users_entities - 1})")

    def load_cf(self, filename):
        """
        解析四级梯度格式
        格式: uid;pos_ids;fair_ids;neutral_ids;easy_ids
        """
        u, i = [], []
        if not hasattr(self, 'tiered_cf_dict'):
            self.tiered_cf_dict = {}

        user_pos_dict = collections.defaultdict(list)
        count = 0

        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(';')
                if len(parts) < 5: continue

                uid = int(parts[0])
                # 解析各级分档列表
                pos_ids = [int(x) for x in parts[1].split(',') if x]
                fair_ids = [int(x) for x in parts[2].split(',') if x]
                neutral_ids = [int(x) for x in parts[3].split(',') if x]
                easy_ids = [int(x) for x in parts[4].split(',') if x]

                user_pos_dict[uid] = pos_ids
                for iid in pos_ids:
                    u.append(uid)
                    i.append(iid)

                # 存入采样字典，供训练时执行阶梯博弈
                self.tiered_cf_dict[uid] = {
                    'fair': fair_ids,
                    'neutral': neutral_ids,
                    'easy': easy_ids
                }
                count += 1

        self.logging.info(f"[*] 已载入 {count} 个岗位的分层训练样本带。")
        return (np.array(u), np.array(i)), user_pos_dict

    def generate_cf_batch(self, user_dict, batch_size):
        """
        实现阶梯负采样策略：强化模型对混合排名中后 400 名“近义负样本”的分辨力。
        """
        valid_users = list(user_dict.keys())
        if not valid_users: return None

        u = np.random.choice(valid_users, batch_size)
        p = [random.choice(user_dict[usr]) for usr in u]

        n = []
        for usr in u:
            tiers = self.tiered_cf_dict.get(usr, {})
            rand = random.random()

            # 采样权重：40% 尚可, 40% 中性(硬负), 20% 无关(易负)
            if rand < 0.4 and tiers.get('fair'):
                n.append(random.choice(tiers['fair']))
            elif rand < 0.8 and tiers.get('neutral'):
                n.append(random.choice(tiers['neutral']))
            elif tiers.get('easy'):
                n.append(random.choice(tiers['easy']))
            else:
                n.append(random.randint(self.ENTITY_OFFSET, self.n_users_entities - 1))

        return torch.LongTensor(u), torch.LongTensor(p), torch.LongTensor(n)

    def load_auxiliary_info(self):
        """
        加载经 build_feature_index.py 处理后的归一化特征。
        修改点：在查找映射表时，为原始作者 ID 添加 'a_' 前缀，以匹配 id_map.json 中的键名。
        """
        global_max_id = self.n_users_entities
        # 默认三维特征：h_index, cited_by, works_count
        features = np.zeros((global_max_id, self.args.n_aux_features), dtype=np.float32)

        if not os.path.exists(FEATURE_INDEX_PATH):
            self.logging.error(f"[!] 未发现特征索引: {FEATURE_INDEX_PATH}")
            self.aux_info_all = torch.from_numpy(features)
            return

        with open(FEATURE_INDEX_PATH, 'r', encoding='utf-8') as f:
            feature_bundle = json.load(f)

        map_path = os.path.join(self.data_dir, "id_map.json")
        with open(map_path, 'r', encoding='utf-8') as f_map:
            mapping = json.load(f_map)

        # e2i 里的键现在是 "a_123", "w_456" 等格式
        e2i = mapping['entity']

        author_features = feature_bundle.get('author', {})
        for raw_aid, feat_dict in author_features.items():
            # --- 核心修改：添加前缀以匹配 id_map.json 中的 entity 键 ---
            prefixed_aid = f"a_{raw_aid}"

            if prefixed_aid in e2i:
                idx = e2i[prefixed_aid]
                if idx < global_max_id:
                    features[idx] = [
                        feat_dict.get('h_index', 0.0),
                        feat_dict.get('cited_by_count', 0.0),
                        feat_dict.get('works_count', 0.0)
                    ]

        self.aux_info_all = torch.from_numpy(features)
        self.logging.info(f"[*] AX 归一化特征加载完成（已适配 a_ 前缀映射）。")
        del feature_bundle, e2i
        gc.collect()

    def load_kg(self, filename):
        """索引驱动子图采样：利用 build_kg_index.py 生成的离线数据库加速 """
        db_path = os.path.join(self.data_dir, "kg_index.db")
        if not os.path.exists(db_path):
            self.logging.error(f"[!] 未发现 KG 索引库: {db_path}，请先运行 build_kg_index.py")
            return pd.DataFrame(columns=['h', 'r', 't'])

        self.logging.info(f"[*] 正在执行索引驱动采样...")
        seeds = list(set(self.cf_train_data[0].tolist()) | set(self.cf_train_data[1].tolist()))
        conn = sqlite3.connect(db_path)

        all_edges = []
        chunk_size = 500
        for i in range(0, len(seeds), chunk_size):
            batch = seeds[i:i + chunk_size]
            placeholders = ','.join(['?'] * len(batch))
            # 1-hop 采样：提取当前训练任务相关的局部拓扑
            sql = f"SELECT h, r, t FROM kg_triplets WHERE h IN ({placeholders}) OR t IN ({placeholders})"
            all_edges.extend(conn.execute(sql, batch + batch).fetchall())

        conn.close()
        df = pd.DataFrame(all_edges, columns=['h', 'r', 't']).drop_duplicates()
        self.logging.info(f"[*] 采样获得子图边数: {len(df)}")
        return df

    def construct_data(self, kg_data):
        """构建正向与反向关系，建立 TransR 空间映射基础 """
        if kg_data.empty: return
        n_rel = max(kg_data['r']) + 1
        inv_kg = kg_data.copy().rename({'h': 't', 't': 'h'}, axis=1)
        inv_kg['r'] += n_rel
        kg_data = pd.concat([kg_data, inv_kg], ignore_index=True)

        self.n_relations = max(kg_data['r']) + 1
        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in kg_data.itertuples(index=False):
            self.train_kg_dict[row.h].append((row.t, row.r))
            self.train_relation_dict[row.r].append((row.h, row.t))

        self.h_list = torch.LongTensor(kg_data['h'].values)
        self.t_list = torch.LongTensor(kg_data['t'].values)
        self.r_list = torch.LongTensor(kg_data['r'].values)
        self.n_kg_train = len(kg_data)

    def create_laplacian_dict(self):
        """生成归一化拉普拉斯矩阵，用于图卷积聚合消息传递"""
        cache_file = os.path.join(self.data_dir, 'adj_matrix_cache.npz')
        if os.path.exists(cache_file):
            adj = sp.load_npz(cache_file)
            self.A_in = self.convert_coo2tensor(adj)
            return

        rows, cols, datas = [], [], []
        for r, ht in self.train_relation_dict.items():
            for h, t in ht:
                rows.append(h);
                cols.append(t);
                datas.append(1.0)

        adj = sp.coo_matrix((datas, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
        rowsum = np.array(adj.sum(axis=1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        norm_adj = sp.diags(d_inv).dot(adj).tocoo()
        sp.save_npz(cache_file, norm_adj)
        self.A_in = self.convert_coo2tensor(norm_adj)

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        """执行知识图谱三元组采样，用于 TransR 损失计算"""
        h_batch = np.random.choice(list(kg_dict.keys()), batch_size)
        h, r, p, n = [], [], [], []
        for head in h_batch:
            t, rel = random.choice(kg_dict[head])
            h.append(head);
            r.append(rel);
            p.append(t)
            n.append(random.randint(self.ENTITY_OFFSET, self.n_users_entities - 1))
        return torch.LongTensor(h), torch.LongTensor(r), torch.LongTensor(p), torch.LongTensor(n), \
            self.aux_info_all[h], self.aux_info_all[p], self.aux_info_all[n]

    def convert_coo2tensor(self, coo):
        idx = torch.LongTensor(np.vstack((coo.row, coo.col)))
        return torch.sparse_coo_tensor(idx, torch.FloatTensor(coo.data), torch.Size(coo.shape))

    def sanity_check(self):
        """数据质量诊断：防止 ID 越界或分区冲突"""
        self.logging.info(f"\n{'=' * 20} DATA SANITY CHECK {'=' * 20}")
        if self.A_in._indices().max().item() >= self.n_users_entities:
            raise ValueError("ID 越界！请检查 ENTITY_OFFSET。")
        self.logging.info(f"节点总空间: {self.n_users_entities}")
        self.logging.info(f"交互边数量: {self.n_cf_train}")
        self.logging.info(f"KG 三元组数量: {self.n_kg_train}")
        self.logging.info("[Success] 静态数据校验通过！")