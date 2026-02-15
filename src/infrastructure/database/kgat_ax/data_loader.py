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
from config import KGATAX_TRAIN_DATA_DIR


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
        """从 JSON 加载全局 ID 映射，支持动态压缩 ID 空间"""
        map_path = os.path.join(self.data_dir, "id_map.json")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"未发现映射文件: {map_path}，请先运行 generate_training_data.py。")

        with open(map_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        self.entity_to_int = mapping['entity']
        # 【修复】：直接从映射文件获取总节点数，不再通过 len(entity_to_int) 推算
        self.n_users_entities = mapping.get("total_nodes", len(self.entity_to_int))
        self.n_users = mapping.get("user_count", 0)
        self.n_entities = mapping.get("entity_count", 0)
        self.ENTITY_OFFSET = mapping.get("offset", 0)

        self.logging.info(
            f"[*] ID 空间映射成功: User(0-{self.ENTITY_OFFSET - 1}), Entity({self.ENTITY_OFFSET}-{self.n_users_entities - 1})")

    def load_auxiliary_info(self):
        """
        提取特征：直接加载预计算好的特征索引文件。
        适配：build_feature_index.py 生成的对数平滑归一化特征。
        """
        import json
        # 【核心逻辑】：物理矩阵的大小必须能装下最大的物理索引
        global_max_id = self.ENTITY_OFFSET + self.n_entities
        features = np.zeros((global_max_id, self.args.n_aux_features), dtype=np.float32)

        # 1. 加载预计算好的特征索引 (由 build_feature_index.py 生成)
        # 路径通常在 config 中定义，这里假设通过 args 传入
        feature_index_path = getattr(self.args, 'feature_index_path', 'data/feature_index.json')

        if not os.path.exists(feature_index_path):
            self.logging.error(f"[!] 未发现特征索引文件: {feature_index_path}，请先运行 build_feature_index.py")
            # 兜底逻辑：保持全零特征，防止程序崩溃
            self.aux_info_all = torch.from_numpy(features)
            return

        with open(feature_index_path, 'r', encoding='utf-8') as f:
            feature_bundle = json.load(f)

        author_features = feature_bundle.get('author', {})
        inst_features = feature_bundle.get('institution', {})

        self.logging.info(f"[*] 正在从 JSON 映射特征到物理矩阵...")

        # 2. 映射作者特征
        # 注意：id_map.json 中的 entity_to_int 存储的是压缩后的 ID
        for raw_aid, feat_dict in author_features.items():
            if raw_aid in self.entity_to_int:
                idx = self.entity_to_int[raw_aid]
                if idx < global_max_id:
                    # 直接使用索引中预处理好的：h_index, cited_by_count, works_count
                    features[idx] = [
                        feat_dict.get('h_index', 0.0),
                        feat_dict.get('cited_by_count', 0.0),
                        feat_dict.get('works_count', 0.0)
                    ]

        # 3. 映射机构特征 (如果你的 KG 包含机构节点，也需对齐)
        for raw_iid, feat_dict in inst_features.items():
            if raw_iid in self.entity_to_int:
                idx = self.entity_to_int[raw_iid]
                if idx < global_max_id:
                    # 机构特征通常只有发文和引用
                    features[idx] = [
                        0.0,  # h_index 占位
                        feat_dict.get('cited_by_count', 0.0),
                        feat_dict.get('works_count', 0.0)
                    ]

        # 4. 转换为 Tensor
        # 【重要】：不再进行二次归一化，直接使用预计算好的 0-1 值
        self.aux_info_all = torch.from_numpy(features)

        self.logging.info(f"[*] AX 特征提取完成。来源: {feature_index_path}")
        self.logging.info(f"[*] 特征矩阵均值: {features.mean():.4f}, 最大值: {features.max():.4f}")

        del feature_bundle
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
        """优化后的负采样：确保完全锁定在人才/实体空间"""
        valid_users = [u for u, items in user_dict.items() if len(items) > 0]
        if not valid_users:
            return None

        u = np.random.choice(valid_users, batch_size)
        p = [random.choice(user_dict[usr]) for usr in u]

        # 【关键】：负采样范围必须是 [ENTITY_OFFSET, n_users_entities)
        # 这确保了岗位永远不会被当做推荐结果，从而强化 Type Embedding 的效果
        n = np.random.randint(self.ENTITY_OFFSET, self.n_users_entities, batch_size)

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
        """
        利用外部 SQLite 索引执行秒级子图采样。
        职责：从 3200万条边中精准提取与当前训练集(Job/User)相关的 1-hop 和 2-hop 拓扑。
        """
        # 直接引用 config 中的路径，保持全局一致性

        db_path = os.path.join(KGATAX_TRAIN_DATA_DIR, "kg_index.db")

        if not os.path.exists(db_path):
            self.logging.error(f"[!] 未发现索引库: {db_path}，请先运行 build_kg_index.py")
            return

        self.logging.info(f"[*] 正在执行索引驱动采样，目标库: {db_path}")

        # 1. 确定核心种子节点 (训练集涉及的 Job 和人)
        seeds = list(set(self.cf_train_data[0].tolist()) | set(self.cf_train_data[1].tolist()))
        conn = sqlite3.connect(db_path)

        try:
            # 2. 提取一阶关联 (1-hop)：双向命中种子
            self.logging.info(f"[*] 正在提取一阶邻居 (种子数: {len(seeds)})...")
            hop1_edges = []
            chunk_size = 500  # 避免 SQL 占位符超限

            for i in range(0, len(seeds), chunk_size):
                batch_seeds = seeds[i:i + chunk_size]
                placeholders = ','.join(['?'] * len(batch_seeds))

                # 利用 idx_h_lookup 和 idx_t_lookup 覆盖索引进行 Index-Only Scan
                sql = f"SELECT h, r, t FROM kg_triplets WHERE h IN ({placeholders}) OR t IN ({placeholders})"
                hop1_edges.extend(conn.execute(sql, batch_seeds + batch_seeds).fetchall())

            # 提取一阶涉及的所有节点，准备二阶扩展
            hop1_nodes = set()
            for h, r, t in hop1_edges:
                hop1_nodes.add(h)
                hop1_nodes.add(t)

            self.logging.info(f"[*] 一阶提取完成：得到 {len(hop1_edges)} 条边")

            # 3. 提取二阶辐射 (2-hop)：贪婪外延
            self.logging.info(f"[*] 正在执行二阶贪婪外延...")
            hop2_edges = []
            max_limit = 1800000 - len(hop1_edges)  # 32GB 内存安全阈值

            hop1_nodes_list = list(hop1_nodes)
            for i in range(0, len(hop1_nodes_list), chunk_size):
                if len(hop2_edges) > max_limit:
                    break

                batch_nodes = hop1_nodes_list[i:i + chunk_size]
                placeholders = ','.join(['?'] * len(batch_nodes))

                # 只查询头节点在圈内的边，捕捉所有外延学术背景
                sql = f"SELECT h, r, t FROM kg_triplets WHERE h IN ({placeholders})"
                hop2_edges.extend(conn.execute(sql, batch_nodes).fetchall())

            # 4. 结果整合与去重
            all_edges = hop1_edges + hop2_edges
            all_kg_df = pd.DataFrame(all_edges, columns=['h', 'r', 't']).drop_duplicates()

            # 5. 最终自检与兜底：若子图过于稀疏则补全随机边
            if len(all_kg_df) < 100000:
                self.logging.warning(f"[!] 连通性不足 ({len(all_kg_df)}条)，注入全局随机采样作为环境噪音...")
                random_edges = conn.execute("SELECT h, r, t FROM kg_triplets LIMIT 200000").fetchall()
                all_kg_df = pd.concat(
                    [all_kg_df, pd.DataFrame(random_edges, columns=['h', 'r', 't'])]).drop_duplicates()

            self.all_kg_data = all_kg_df.values
            self.n_kg_train = len(self.all_kg_data)
            self.logging.info(f"[*] 最终 KG 训练规模: {self.n_kg_train} 条边")

        finally:
            conn.close()
            gc.collect()  # 显式回收 SQL 结果集占用的内存

        return all_kg_df

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