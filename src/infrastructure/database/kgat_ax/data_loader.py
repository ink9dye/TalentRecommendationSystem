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
    适配加权拓扑与四级梯度精排逻辑的 KGAT-AX 数据加载器。
    职责：加载归一化学术特征 (AX)、执行加权子图采样、构建加权拉普拉斯矩阵。
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

        # 1. 载入全局映射与分区信息
        self.load_id_mapping()

        # 2. 载入分层协同过滤交互数据
        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
        self.n_cf_train = len(self.cf_train_data[0])
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)

        # 3. 提取学术增强特征 (AX)
        self.load_auxiliary_info()

        # 4. 执行加权索引驱动采样
        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)

        # 5. 构建加权拉普拉斯矩阵
        self.create_laplacian_dict()

        # 6. 最终自检
        self.sanity_check()

    def load_id_mapping(self):
        """加载 ID 映射，确立 User 与 Entity 边界"""
        map_path = os.path.join(self.data_dir, "id_map.json")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"未发现映射文件: {map_path}。")

        with open(map_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        self.entity_to_int = mapping.get("entity", {})
        self.user_to_int = {k: v for k, v in self.entity_to_int.items() if
                            not k.startswith(('a_', 'w_', 'v_', 'i_', 's_'))}

        self.n_users_entities = mapping.get("total_nodes")
        self.n_users = mapping.get("user_count", 0)
        self.ENTITY_OFFSET = mapping.get("offset", 0)
        self.n_relations = 0  # 待 construct_data 初始化

    def load_cf(self, filename):
        """解析四级梯度格式"""
        u, i = [], []
        if not hasattr(self, 'tiered_cf_dict'):
            self.tiered_cf_dict = {}

        user_pos_dict = collections.defaultdict(list)
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) < 5: continue
                uid = int(parts[0])
                pos_ids = [int(x) for x in parts[1].split(',') if x]
                user_pos_dict[uid] = pos_ids
                for iid in pos_ids:
                    u.append(uid);
                    i.append(iid)
                self.tiered_cf_dict[uid] = {
                    'fair': [int(x) for x in parts[2].split(',') if x],
                    'neutral': [int(x) for x in parts[3].split(',') if x],
                    'easy': [int(x) for x in parts[4].split(',') if x]
                }
        return (np.array(u), np.array(i)), user_pos_dict

    def generate_cf_batch(self, user_dict, batch_size):
        """阶梯负采样逻辑"""
        valid_users = [u for u in user_dict.keys() if len(user_dict[u]) > 0]
        u = np.random.choice(valid_users, batch_size)
        p = [random.choice(user_dict[usr]) for usr in u]
        n = []
        for usr in u:
            tiers = self.tiered_cf_dict.get(usr, {})
            rand = random.random()
            if rand < 0.4:
                target_list = tiers.get('fair') or tiers.get('neutral') or tiers.get('easy')
            elif rand < 0.8:
                target_list = tiers.get('neutral') or tiers.get('easy')
            else:
                target_list = tiers.get('easy')

            n.append(random.choice(target_list) if target_list else random.randint(self.ENTITY_OFFSET,
                                                                                   self.n_users_entities - 1))
        return torch.LongTensor(u), torch.LongTensor(p), torch.LongTensor(n)

    def load_auxiliary_info(self):
        """加载 AX 学术特征并执行归一化对齐"""
        features = np.zeros((self.n_users_entities, self.args.n_aux_features), dtype=np.float32)
        if not os.path.exists(FEATURE_INDEX_PATH):
            self.aux_info_all = torch.from_numpy(features);
            return

        with open(FEATURE_INDEX_PATH, 'r', encoding='utf-8') as f:
            feature_bundle = json.load(f)

        match_count = 0
        author_features = feature_bundle.get('author', {})
        for raw_aid, feat_dict in author_features.items():
            key = f"a_{str(raw_aid).strip().lower()}"
            if key in self.entity_to_int:
                idx = self.entity_to_int[key]
                features[idx] = [feat_dict.get('h_index', 0.), feat_dict.get('cited_by_count', 0.),
                                 feat_dict.get('works_count', 0.)]
                match_count += 1

        self.aux_info_all = torch.from_numpy(features)
        self.logging.info(f"[*] AX 特征加载完成，命中 {match_count} 位作者。")

    def load_kg(self, filename):
        """
        加权索引驱动采样 (定向 2-Hop 语义增强版)
        核心修改：在 1-Hop 基础上，提取 Vocab 节点并补全 SIMILAR_TO (r=6) 关系，确保关系总数对齐为 14。
        """
        db_path = os.path.join(self.data_dir, "kg_index.db")
        if not os.path.exists(db_path):
            self.logging.error(f"[!] 找不到加权索引库: {db_path}")
            return pd.DataFrame()

        self.logging.info(f"[*] 执行加权子图采样 (正在织入语义桥梁)...")

        # 种子节点包含训练集中的所有 User(Job) 和 Item(Author)
        seeds = list(set(self.cf_train_data[0].tolist()) | set(self.cf_train_data[1].tolist()))
        conn = sqlite3.connect(db_path)
        all_edges = []
        vocab_nodes = set()  # 用于存储子图中出现的词汇/技能节点
        chunk_size = 500

        # --- 阶段 1: 1-Hop 基础采样 ---
        for i in range(0, len(seeds), chunk_size):
            batch = seeds[i:i + chunk_size]
            placeholders = ','.join(['?'] * len(batch))

            # 提取带权重 w 的直接关联边
            sql = f"SELECT h, r, t, w FROM kg_triplets WHERE h IN ({placeholders}) OR t IN ({placeholders})"
            res = conn.execute(sql, batch + batch).fetchall()
            all_edges.extend(res)

            # 记录子图中遇到的词汇节点：r=4 (Work->Vocab), r=5 (Job->Vocab)
            # 注意：在这些关系中，t 节点是 Vocabulary
            for h, r, t, w in res:
                if r in [4, 5]:
                    vocab_nodes.add(t)

        # --- 阶段 2: 定向 2-Hop 语义补全 (针对 SIMILAR_TO 关系) ---
        if vocab_nodes:
            self.logging.info(f"[*] 发现 {len(vocab_nodes)} 个词汇节点，正在注入 SIMILAR_TO 语义关系...")
            v_list = list(vocab_nodes)

            for i in range(0, len(v_list), chunk_size):
                batch = v_list[i:i + chunk_size]
                placeholders = ','.join(['?'] * len(batch))

                # 关键：只拉取关系 ID 为 6 (SIMILAR_TO) 且两端都在子图中的边
                # 这样既能补全语义，又不会导致子图规模爆炸
                sql = f"SELECT h, r, t, w FROM kg_triplets WHERE r=6 AND h IN ({placeholders}) AND t IN ({placeholders})"
                bridge_res = conn.execute(sql, batch + batch).fetchall()
                all_edges.extend(bridge_res)

        conn.close()

        # 转换为 DataFrame 并去重
        df_final = pd.DataFrame(all_edges, columns=['h', 'r', 't', 'w']).drop_duplicates()

        # 验证是否包含关系 6
        if not df_final.empty and 6 in df_final['r'].unique():
            self.logging.info(f"[OK] 语义桥梁注入成功，子图包含 SIMILAR_TO 关系。")
        else:
            self.logging.warning(f"[!] 警告：子图中仍未发现 SIMILAR_TO 关系，请检查 kg_final.txt 数据质量。")

        return df_final

    def construct_data(self, kg_data):
        """
        构建带权关系字典。
        核心修改：将权重 w 存入字典以供后续拉普拉斯矩阵构建。
        """
        if kg_data.empty: return
        n_rel = max(kg_data['r']) + 1
        inv_kg = kg_data.copy().rename({'h': 't', 't': 'h'}, axis=1)
        inv_kg['r'] += n_rel
        kg_data = pd.concat([kg_data, inv_kg], ignore_index=True)

        self.n_relations = int(max(kg_data['r']) + 1)
        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in kg_data.itertuples(index=False):
            # 存储格式变更为 (目标, 关系, 权重)
            self.train_kg_dict[row.h].append((row.t, row.r, row.w))
            self.train_relation_dict[row.r].append((row.h, row.t, row.w))

        self.h_list = torch.LongTensor(kg_data['h'].values)
        self.t_list = torch.LongTensor(kg_data['t'].values)
        self.r_list = torch.LongTensor(kg_data['r'].values)
        self.n_kg_train = len(kg_data)

    def create_laplacian_dict(self):
        """
        生成归一化加权拉普拉斯矩阵。
        核心修改：使用真实权重 w 填充邻接矩阵，实现时序衰减物理隔离。
        """
        cache_file = os.path.join(self.data_dir, 'weighted_adj_cache.npz')
        if os.path.exists(cache_file):
            adj = sp.load_npz(cache_file)
            self.A_in = self.convert_coo2tensor(adj);
            return

        rows, cols, datas = [], [], []
        for r, htw in self.train_relation_dict.items():
            for h, t, w in htw:
                rows.append(h);
                cols.append(t)
                # 核心点：不再是 1.0，而是使用 builder.py 计算的 pos_weight
                datas.append(float(w))

        adj = sp.coo_matrix((datas, (rows, cols)), shape=(self.n_users_entities, self.n_users_entities))
        # 归一化处理
        rowsum = np.array(adj.sum(axis=1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        norm_adj = sp.diags(d_inv).dot(adj).tocoo()

        sp.save_npz(cache_file, norm_adj)
        self.A_in = self.convert_coo2tensor(norm_adj)

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        """执行带权三元组采样"""
        h_batch = np.random.choice(list(kg_dict.keys()), batch_size)
        h, r, p, n = [], [], [], []
        for head in h_batch:
            # 适配 (t, r, w) 三元组结构
            t, rel, weight = random.choice(kg_dict[head])
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
        """数据质量诊断"""
        self.logging.info(f"\n{'=' * 10} 加权数据校验 {'=' * 10}")
        self.logging.info(f"节点空间: {self.n_users_entities} | 关系数: {self.n_relations}")
        self.logging.info(f"KG 边数: {self.n_kg_train} | AX 特征维度: {self.aux_info_all.shape}")
        if self.A_in._indices().max().item() >= self.n_users_entities:
            raise ValueError("ID 越界！")
        self.logging.info("[Success] 加权数据载入流程完成。")