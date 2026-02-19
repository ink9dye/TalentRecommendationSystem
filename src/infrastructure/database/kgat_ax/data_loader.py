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
        """
        加载 ID 压缩映射，确立 User 与 Entity 的物理边界。
        修改点：显式存储 entity_to_int 和 user_to_int 映射表。
        """
        map_path = os.path.join(self.data_dir, "id_map.json")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"未发现映射文件: {map_path}，请先运行 generate_training_data.py。")

        with open(map_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        # --- 核心修改：持久化映射字典，供精排引擎（RankingEngine）使用 ---
        # 1. 存储全量实体映射 (包含 a_, w_, v_ 等前缀的作者、作品、技能)
        self.entity_to_int = mapping.get("entity", {})

        # 2. 提取用户（岗位）映射
        # 在 generate_training_data 中，岗位 ID 是直接以原始字符串存储的，没有 a_ 等前缀
        self.user_to_int = {k: v for k, v in self.entity_to_int.items() if
                            not k.startswith(('a_', 'w_', 'v_', 'i_', 's_'))}

        # 为了兼容 RankScorer 的旧命名习惯，额外提供一个别名
        self.raw_to_int = self.entity_to_int
        self.raw_user_to_int = self.user_to_int

        # 3. 基础统计信息提取
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
        实现阶梯负采样策略：修复 IndexError 并强化退避机制。
        """
        # 1. 核心修复：只在确实有正样本的用户中选择，防止 random.choice 报错
        valid_users = [u for u in user_dict.keys() if len(user_dict[u]) > 0]

        if not valid_users:
            self.logging.warning("所有用户均无正样本，无法生成 Batch")
            return None

        # 随机选择 batch_size 个用户
        u = np.random.choice(valid_users, batch_size)

        # 2. 安全采样正样本
        p = [random.choice(user_dict[usr]) for usr in u]

        n = []
        for usr in u:
            tiers = self.tiered_cf_dict.get(usr, {})
            rand = random.random()

            # 3. 带有退避逻辑的阶梯采样 (4:4:2 比例)
            # 优先采样分层负样本，若该层缺失则向下退避
            if rand < 0.4:
                # 尝试采样 'fair' (前 100-400 名)
                target_list = tiers.get('fair', [])
                if not target_list: target_list = tiers.get('neutral', [])
                if not target_list: target_list = tiers.get('easy', [])
            elif rand < 0.8:
                # 尝试采样 'neutral' (前 400-500 名，硬负样本)
                target_list = tiers.get('neutral', [])
                if not target_list: target_list = tiers.get('easy', [])
            else:
                # 采样 'easy' (完全无关的人才)
                target_list = tiers.get('easy', [])

            # 4. 终极退避：如果分层字典里啥都没有，就在全局人才空间随机抓一个
            if target_list:
                n.append(random.choice(target_list))
            else:
                n.append(random.randint(self.ENTITY_OFFSET, self.n_users_entities - 1))

        return torch.LongTensor(u), torch.LongTensor(p), torch.LongTensor(n)

    def load_auxiliary_info(self):
        """
        加载经 build_feature_index.py 处理后的归一化特征。
        职责：为作者节点注入 h-index 等学术质量指标。
        修正点：加入 ID 归一化对齐与加载命中率监控。
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

        # 从 id_map.json 加载 ID 空间定义
        map_path = os.path.join(self.data_dir, "id_map.json")
        with open(map_path, 'r', encoding='utf-8') as f_map:
            mapping = json.load(f_map)

        # e2i 里的键是 "a_123", "w_456" 等全小写格式
        e2i = mapping.get('entity', {})
        author_features = feature_bundle.get('author', {})

        match_count = 0
        # 仅用于调试：记录前几个匹配失败的 ID
        mismatch_samples = []

        for raw_aid, feat_dict in author_features.items():
            # --- 核心修复：添加前缀并执行小写化，严格对齐 generate_training_data 逻辑 ---
            prefixed_aid = f"a_{str(raw_aid).strip().lower()}"

            if prefixed_aid in e2i:
                idx = e2i[prefixed_aid]
                if idx < global_max_id:
                    features[idx] = [
                        feat_dict.get('h_index', 0.0),
                        feat_dict.get('cited_by_count', 0.0),
                        feat_dict.get('works_count', 0.0)
                    ]
                    match_count += 1
            else:
                if len(mismatch_samples) < 3:
                    mismatch_samples.append(prefixed_aid)

        self.aux_info_all = torch.from_numpy(features)

        # --- 增加调试日志 ---
        self.logging.info(f"[*] AX 学术特征预载入完成:")
        self.logging.info(f"    - 特征库作者数: {len(author_features)}")
        self.logging.info(f"    - 成功映射人数: {match_count}")

        if match_count == 0 and len(author_features) > 0:
            self.logging.error(f"[关键错误] AX 特征完全匹配失败！")
            self.logging.error(f"    - 尝试匹配的 ID 样例: {mismatch_samples}")
            # 获取 id_map 中的样例进行对比
            id_map_samples = list(e2i.keys())[:3]
            self.logging.error(f"    - id_map.json 中的 ID 样例: {id_map_samples}")
        elif match_count < len(author_features) * 0.1:
            self.logging.warning(f"[警告] AX 特征匹配率极低 ({match_count}/{len(author_features)})，请检查 ID 格式。")

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