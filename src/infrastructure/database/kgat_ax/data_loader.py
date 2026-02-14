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
from tqdm import tqdm


class DataLoaderKGAT(object):
    """
    针对 KGAT-AX 架构优化的数据加载器
    职责：ID映射加载、三元组构建、邻接矩阵生成（含缓存）及学术特征提取
    """

    def __init__(self, args, logging):
        self.args = args
        self.logging = logging

        # 路径初始化
        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.kg_file = os.path.join(self.data_dir, "kg_final.txt")
        self.test_batch_size = getattr(args, 'test_batch_size', 1024)

        # 1. 载入全局映射与分区信息（核心：奠定 ID 空间基础）
        self.load_id_mapping()

        # 2. 载入协同过滤交互数据
        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
        self.n_cf_train = len(self.cf_train_data[0])
        self.cf_test_data, self.test_user_dict = self.load_cf(os.path.join(self.data_dir, 'test.txt'))

        # 3. 提取学术增强特征 (AX) - 适配 database.py 的覆盖索引
        self.load_auxiliary_info()

        # 4. 构建图结构
        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)

        # 5. 构建拉普拉斯矩阵 (含自动缓存逻辑，避免重复计算耗时)
        self.create_laplacian_dict()

        # 6. 最终自检（杜绝 ID 冲突导致的训练崩溃）
        self.sanity_check()

    def load_id_mapping(self):
        """从 JSON 加载全局 ID 映射，确保 User(Job) 与 Entity 边界严谨"""
        map_path = os.path.join(self.data_dir, "id_map.json")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"未发现映射文件: {map_path}，请先运行数据生成脚本。")

        with open(map_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        self.entity_to_int = mapping['entity']
        self.n_users_entities = len(self.entity_to_int)
        # 必须是 Job 节点的数量，用于区分 CF 和 KG 空间
        self.n_users = mapping.get("user_count", 0)
        self.n_entities = self.n_users_entities - self.n_users
        # 获取实体起始偏移量
        self.ENTITY_OFFSET = mapping.get("offset", 0)

    def load_auxiliary_info(self):
        """
        提取特征：利用覆盖索引提取特征。
        修正物理维度：解决由 ENTITY_OFFSET 导致的索引越界问题。
        """
        # 【核心修复】：物理矩阵的大小必须能装下最大的物理索引
        # 你的实体 ID 范围是 [OFFSET, OFFSET + n_entities)
        global_max_id = self.ENTITY_OFFSET + self.n_entities
        features = np.zeros((global_max_id, self.args.n_aux_features), dtype=np.float32)

        conn = sqlite3.connect(self.args.db_path)
        # 增加缓存，加速大规模读取
        conn.execute("PRAGMA cache_size = -100000")

        # 提取作者特征: 严格对应 database.py 索引顺序 (author_id, h_index, cited_by_count, works_count)
        author_query = "SELECT author_id, h_index, cited_by_count, works_count FROM authors INDEXED BY idx_author_metrics_covering"

        # 使用 chunk 避免内存溢出
        for chunk in pd.read_sql(author_query, conn, chunksize=100000):
            for row in chunk.itertuples(index=False):
                aid_str = str(row.author_id)
                # 【关键修复】：只处理在 id_map.json 中定义的实体（即参与训练/测试的 ID）
                if aid_str in self.entity_to_int:
                    idx = self.entity_to_int[aid_str]
                    # 安全边界检查：确保 idx 落在刚才开辟的 [0, global_max_id) 空间内
                    if idx < global_max_id:
                        features[idx] = [
                            np.log1p(row.h_index or 0),
                            np.log1p(row.cited_by_count or 0),
                            np.log1p(row.works_count or 0)
                        ]
        conn.close()

        # 归一化：将特征缩放到 [0, 1] 区间，防止训练时梯度爆炸
        f_min, f_max = features.min(axis=0), features.max(axis=0)
        self.aux_info_all = torch.from_numpy((features - f_min) / (f_max - f_min + 1e-9))

        self.logging.info(f"[*] AX 特征提取完成，物理矩阵维度已对齐至: {global_max_id}")
        gc.collect()

    def create_laplacian_dict(self):
        """构建归一化邻接矩阵：修正维度逻辑以兼容 OFFSET 分区"""
        cache_file = os.path.join(self.data_dir, 'adj_matrix_cache.npz')
        if os.path.exists(cache_file):
            self.logging.info(f"[*] 发现矩阵缓存，正在秒速加载...")
            adj = sp.load_npz(cache_file)
            self.A_in = self.convert_coo2tensor(adj)
            return

        self.logging.info("[*] 正在构建邻接矩阵...")
        rows, cols, datas = [], [], []
        for r, ht in self.train_relation_dict.items():
            for h, t in ht:
                rows.append(h)
                cols.append(t)
                datas.append(1.0)

        # 【核心修复】：矩阵维度必须覆盖最大物理索引
        # 你的最大索引是 ENTITY_OFFSET + n_entities，约为 2839682
        global_max_id = self.ENTITY_OFFSET + self.n_entities

        # 1. 声明足够大的矩阵尺寸
        adj = sp.coo_matrix(
            (datas, (rows, cols)),
            shape=(global_max_id, global_max_id)
        )

        # 2. 拉普拉斯归一化: D^-1 * A
        rowsum = np.array(adj.sum(axis=1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        norm_adj = sp.diags(d_inv).dot(adj).astype(np.float32).tocoo()

        self.logging.info(f"[*] 正在保存缓存...")
        sp.save_npz(cache_file, norm_adj)
        self.A_in = self.convert_coo2tensor(norm_adj)

    def construct_data(self, kg_data):
        """对齐 KG 三元组并添加逆关系边"""
        # 添加逆关系 (h, r, t) -> (t, r + n_rel, h)
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

    def generate_cf_batch(self, user_dict, batch_size):
        """CF 训练批次生成：(User, Pos_Item, Neg_Item)"""
        u = np.random.choice(list(user_dict.keys()), batch_size)
        p = [random.choice(user_dict[usr]) for usr in u]
        # 负采样范围为全实体空间
        n = np.random.randint(0, self.n_users_entities, batch_size)
        return torch.LongTensor(u), torch.LongTensor(p), torch.LongTensor(n)

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        """KG 训练批次生成：带 AX 辅助信息的分发"""
        h_batch = np.random.choice(list(kg_dict.keys()), batch_size)
        h, r, p, n = [], [], [], []
        for head in h_batch:
            t, rel = random.choice(kg_dict[head])
            h.append(head);
            r.append(rel);
            p.append(t)
            # 修正：负采样必须落在 Entity 空间，避开 User 节点
            n.append(random.randint(self.ENTITY_OFFSET, self.n_users_entities - 1))

        return torch.LongTensor(h), torch.LongTensor(r), torch.LongTensor(p), torch.LongTensor(n), \
            self.aux_info_all[h], self.aux_info_all[p], self.aux_info_all[n]

    def convert_coo2tensor(self, coo):
        """Scipy COO 转换为 PyTorch Sparse Tensor"""
        idx = torch.LongTensor(np.vstack((coo.row, coo.col)))
        return torch.sparse_coo_tensor(idx, torch.FloatTensor(coo.data), torch.Size(coo.shape))

    def load_cf(self, filename):
        u, i = [], []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) < 2: continue
                uid = int(parts[0])
                for iid in parts[1:]:
                    u.append(uid);
                    i.append(int(iid))
        udict = collections.defaultdict(list)
        for user, item in zip(u, i): udict[user].append(item)
        return (np.array(u), np.array(i)), udict

    def load_kg(self, filename):
        return pd.read_csv(filename, sep=' ', names=['h', 'r', 't'])

    def sanity_check(self):
        """深度数据质量校验：防止模型在第一轮就因为低级错误崩溃"""
        self.logging.info(f"\n{'=' * 20} DATA SANITY CHECK {'=' * 20}")

        # 1. 范围校验：防止 Embedding 越界
        # A_in._indices() 拿到的是 Sparse Tensor 的坐标映射
        max_id_in_adj = self.A_in._indices().max().item()
        global_max_id = self.ENTITY_OFFSET + self.n_entities

        if max_id_in_adj >= global_max_id:
            raise ValueError(f"ID 越界: 邻接矩阵最大 ID({max_id_in_adj}) >= 预设最大 ID({global_max_id})")

        # 2. 空间分区校验
        if self.ENTITY_OFFSET < self.n_users:
            raise ValueError(f"分区重叠: ENTITY_OFFSET({self.ENTITY_OFFSET}) 必须大于 UserCount({self.n_users})")

        # 3. 特征值校验
        if torch.isnan(self.aux_info_all).any():
            raise ValueError("特征矩阵中包含 NaN！请检查数据库中的学术指标是否有非数值项。")

        # 4. 交互数据（训练集）合法性校验
        # 确保 train.txt 里的所有 item_id 都在实体空间 [OFFSET, global_max_id)
        train_items = self.cf_train_data[1]
        if train_items.min() < self.ENTITY_OFFSET:
            raise ValueError(f"交互数据错误: 发现 Item ID {train_items.min()} 小于偏移量 {self.ENTITY_OFFSET}")

        self.logging.info(f"节点总空间: {global_max_id}")
        self.logging.info(f"交互边数量: {self.n_cf_train}")
        self.logging.info(f"KG 三元组数量: {self.n_kg_train}")
        self.logging.info("[Success] 静态数据校验通过！")
        self.logging.info(f"{'=' * 50}\n")