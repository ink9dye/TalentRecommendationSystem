# -*- coding: utf-8 -*-
"""
KGAT-AX 训练与评估用数据加载器。

职责（对齐 README 修改计划，已实现）：
- 加载归一化学术特征（AX）：支持 3 维兼容格式与 12 维 Author Tower 格式（README 5.2）。
- 加载四级梯度 CF 与加权 KG，执行加权子图采样、构建加权拉普拉斯矩阵。
- 训练样本入口已与 candidate_pool.candidate_records 对齐（generate_training_data 优先基于候选池）。
- 支持加载 train_four_branch.json / test_four_branch.json 四分支侧车，供四分支模型训练与评估使用。
"""

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

# ---------------------------------------------------------------------------
# 作者显式指标列名（与 model.AUTHOR_AUX_COLUMNS 一致，README 5.2）
# 当前 feature_index 仅产出 3 维（h_index, cited_by_count, works_count），
# 扩展 build_feature_index 后可产出 12 维，此处顺序需与模型 Author Tower 输入一致。
# ---------------------------------------------------------------------------
AUTHOR_AUX_COLUMNS_V2 = [
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
N_AUTHOR_AUX_V2 = len(AUTHOR_AUX_COLUMNS_V2)


class DataLoaderKGAT(object):
    """
    适配加权拓扑与四级梯度精排逻辑的 KGAT-AX 数据加载器。

    - 支持 3 维 AX（兼容旧版）与 12 维 Author Tower 特征（README 5.2）；
    - 加权子图采样、加权拉普拉斯矩阵；
    - 训练样本与 candidate_pool.candidate_records 对齐（README 5.5）；
    - 可选加载四分支侧车 train_four_branch.json / test_four_branch.json，供四分支模型使用。
    """

    def __init__(self, args, logging):
        self.args = args
        self.logging = logging

        # 路径初始化
        self.data_dir = os.path.join(args.data_dir, args.data_name)
        # 训练入口：优先池监督 JSONL；不存在则回退 legacy train/test.txt
        self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.test_file = os.path.join(self.data_dir, 'test.txt')
        self.train_pool_supervised_file = os.path.join(self.data_dir, "train_pool_supervised.jsonl")
        self.dev_pool_supervised_file = os.path.join(self.data_dir, "dev_pool_supervised.jsonl")
        self.test_pool_supervised_file = os.path.join(self.data_dir, "test_pool_supervised.jsonl")
        self.kg_file = os.path.join(self.data_dir, "kg_final.txt")
        self.test_batch_size = getattr(args, 'test_batch_size', 1024)

        # 1. 载入全局映射与分区信息（User / Entity 边界）
        self.load_id_mapping()

        # 2. 载入训练/评估交互数据
        #    - 池监督模式：从 *_pool_supervised.jsonl 构造 (u, pos_item) 样本与 tiered_cf_dict；
        #      并提供 gold-only 评估字典，避免 weak 样本污染指标。
        #    - legacy 模式：沿用 train.txt/test.txt（四级梯度）。
        self.pool_supervised_mode = (
            os.path.exists(self.train_pool_supervised_file)
            and os.path.exists(self.dev_pool_supervised_file)
            and os.path.exists(self.test_pool_supervised_file)
        )
        self.pos_sample_weight = {}  # (uid, iid) -> float(sample_weight)
        self.pool_supervised_stats = {}
        if self.pool_supervised_mode:
            self.logging.info(
                "[*] 检测到池监督 JSONL：优先使用 train/dev/test_pool_supervised.jsonl 作为训练入口。"
            )
            (
                self.cf_train_data,
                self.train_user_dict,
                self.train_user_dict_gold,
                self.tiered_cf_dict,
                self.pos_sample_weight,
                self.pool_supervised_stats,
            ) = self.load_cf_from_pool_supervised(
                train_path=self.train_pool_supervised_file,
                dev_path=self.dev_pool_supervised_file,
                test_path=self.test_pool_supervised_file,
                debug_train_limit=int(os.environ.get("KGATAX_DEBUG_TRAIN_LIMIT", "0") or "0"),
            )
            self.n_cf_train = len(self.cf_train_data[0])
            # 评估：all vs gold-only
            self.test_user_dict = self.pool_supervised_stats.get("test_user_dict_all", {})
            self.test_user_dict_gold = self.pool_supervised_stats.get("test_user_dict_gold", {})
            self.dev_user_dict = self.pool_supervised_stats.get("dev_user_dict_all", {})
            self.dev_user_dict_gold = self.pool_supervised_stats.get("dev_user_dict_gold", {})
            self.train_user_dict_all = self.train_user_dict
            self.train_user_dict_gold = self.train_user_dict_gold
        else:
            # legacy：载入分层协同过滤交互数据（四级梯度：pos / fair / neutral / easy）
            self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
            self.n_cf_train = len(self.cf_train_data[0])
            self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)

        # 3. 提取学术增强特征（AX）：3 维或 12 维，由 args.n_aux_features / n_author_aux 决定
        self.load_auxiliary_info()

        # 4. 载入四分支侧车（若存在且模型启用四分支）
        self.four_branch_train = {}
        self.four_branch_test = {}
        self._load_four_branch_sidecar()

        # 5. 执行加权索引驱动采样（1-Hop + 2-Hop SIMILAR_TO 语义桥）
        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)

        # 6. 构建加权拉普拉斯矩阵（用于图传播）
        self.create_laplacian_dict()

        # 7. 自检
        self.sanity_check()

    def _load_four_branch_sidecar(self):
        """加载 train_four_branch.json / test_four_branch.json（若存在），供四分支模型使用。"""
        use_four = (
            getattr(self.args, 'n_recall_features', 0) > 0
            or getattr(self.args, 'n_author_aux', 0) > 0
            or getattr(self.args, 'n_interaction_features', 0) > 0
        )
        if not use_four:
            return
        for name, key in [('train_four_branch.json', 'four_branch_train'), ('test_four_branch.json', 'four_branch_test')]:
            path = os.path.join(self.data_dir, name)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    setattr(self, key, json.load(f))
                self.logging.info(f"[*] 已加载四分支侧车: {name}，共 {len(getattr(self, key))} 条。")

    def load_id_mapping(self):
        """
        加载 ID 映射，确立 User（岗位）与 Entity（作者等）边界。
        id_map.json 需包含：entity, total_nodes, user_count, offset。
        """
        map_path = os.path.join(self.data_dir, "id_map.json")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"未发现映射文件: {map_path}。")

        with open(map_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        self.entity_to_int = mapping.get("entity", {})
        self.user_to_int = {
            k: v for k, v in self.entity_to_int.items()
            if not k.startswith(('a_', 'w_', 'v_', 'i_', 's_'))
        }
        self.n_users_entities = mapping.get("total_nodes")
        self.n_users = mapping.get("user_count", 0)
        self.ENTITY_OFFSET = mapping.get("offset", 0)
        self.n_relations = 0

    def load_cf(self, filename):
        """
        解析四级梯度 CF 文件格式：
        每行：uid;pos_ids;fair_neg;neutral_neg;easy_neg
        用于阶梯负采样（Strong Positive > Fair > Neutral > EasyNeg）。
        """
        u, i = [], []
        if not hasattr(self, 'tiered_cf_dict'):
            self.tiered_cf_dict = {}

        user_pos_dict = collections.defaultdict(list)
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) < 5:
                    continue
                uid = int(parts[0])
                pos_ids = [int(x) for x in parts[1].split(',') if x]
                user_pos_dict[uid] = pos_ids
                for iid in pos_ids:
                    u.append(uid)
                    i.append(iid)
                self.tiered_cf_dict[uid] = {
                    'fair': [int(x) for x in parts[2].split(',') if x],
                    'neutral': [int(x) for x in parts[3].split(',') if x],
                    'easy': [int(x) for x in parts[4].split(',') if x],
                }
        return (np.array(u), np.array(i)), user_pos_dict

    def _normalize_label_grade(self, raw_label_grade):
        """
        训练用 label_grade 归一到 {0,1,2}，并返回 (normalized, raw_int_or_none, out_of_range_flag)。
        规则：
        - raw >= 2 -> 2
        - raw == 1 -> 1
        - else -> 0
        """
        if raw_label_grade is None:
            return 0, None, True
        try:
            r = int(raw_label_grade)
        except (TypeError, ValueError):
            return 0, None, True
        if r >= 2:
            return 2, r, False
        if r == 1:
            return 1, r, False
        if r == 0:
            return 0, r, False
        # 负数等异常 → 归零但计为异常
        return 0, r, True

    def load_cf_from_pool_supervised(
        self,
        *,
        train_path: str,
        dev_path: str,
        test_path: str,
        debug_train_limit: int = 0,
    ):
        """
        从 pool_supervised.jsonl 构造 CF 训练样本与评估字典。

        - 训练样本 (u,i) 取 train_y==1 的行（gold+weak），并记录 sample_weight 供训练加权。
        - tiered_cf_dict：候选池来自所有出现的 author_id（含负类），fair/neutral 置空，easy=候选池。
        - 评估字典：all（train_y==1）与 gold-only（label_source=='gold' 且 train_y==1）两套。
        """
        def _read_jsonl(path: str, limit: int = 0):
            rows = []
            if not os.path.exists(path):
                return rows
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    rows.append(obj)
                    if limit and len(rows) >= limit:
                        break
            return rows

        train_rows = _read_jsonl(train_path, limit=debug_train_limit if debug_train_limit > 0 else 0)
        dev_rows = _read_jsonl(dev_path)
        test_rows = _read_jsonl(test_path)

        raw_grade_counts = collections.Counter()
        norm_grade_counts = collections.Counter()
        out_of_range = 0
        label_source_counts = collections.Counter()
        w_min = None
        w_max = None

        def _uid_from_security(security_id: str):
            return int(self.user_to_int.get(str(security_id)))

        def _iid_from_author(author_id: str):
            key = f"a_{str(author_id).strip().lower()}"
            return int(self.entity_to_int.get(key))

        # train core containers
        u_list, i_list = [], []
        train_user_all = collections.defaultdict(list)
        train_user_gold = collections.defaultdict(list)
        tiered = {}
        pos_w = {}

        # helper for dict building
        def _consume_rows(rows, *, build_train: bool):
            nonlocal w_min, w_max, out_of_range
            for obj in rows:
                qid = obj.get("query_id")
                _ = qid  # 保留字段，当前仅用于诊断/一致性
                sec = obj.get("securityId")
                aid = obj.get("author_id")
                if sec is None or aid is None:
                    continue
                if str(sec) not in self.user_to_int:
                    continue
                a_key = f"a_{str(aid).strip().lower()}"
                if a_key not in self.entity_to_int:
                    continue

                uid = int(self.user_to_int[str(sec)])
                iid = int(self.entity_to_int[a_key])

                raw_grade = obj.get("label_grade")
                raw_grade_counts[str(raw_grade)] += 1
                norm_grade, raw_int, is_oor = self._normalize_label_grade(raw_grade)
                norm_grade_counts[int(norm_grade)] += 1
                if is_oor:
                    out_of_range += 1

                # 强制重算 train_y，避免导出侧的历史遗留
                train_y = 1 if norm_grade in (1, 2) else 0

                src = str(obj.get("label_source") or "")
                label_source_counts[src] += 1

                try:
                    sw = float(obj.get("sample_weight", 1.0))
                except (TypeError, ValueError):
                    sw = 1.0
                if w_min is None or sw < w_min:
                    w_min = sw
                if w_max is None or sw > w_max:
                    w_max = sw

                # tiered candidates: 用 easy 填充（候选池内精排），fair/neutral 留空
                t = tiered.get(uid)
                if t is None:
                    t = {"fair": [], "neutral": [], "easy": []}
                    tiered[uid] = t
                if iid not in t["easy"]:
                    t["easy"].append(iid)

                if train_y == 1:
                    if build_train:
                        u_list.append(uid)
                        i_list.append(iid)
                        pos_w[(uid, iid)] = sw
                    train_user_all[uid].append(iid)
                    if src == "gold":
                        train_user_gold[uid].append(iid)

        _consume_rows(train_rows, build_train=True)
        # dev/test 只用于评估 dict（不参与 cf_train_data）
        dev_user_all = collections.defaultdict(list)
        dev_user_gold = collections.defaultdict(list)
        test_user_all = collections.defaultdict(list)
        test_user_gold = collections.defaultdict(list)

        def _consume_eval(rows, out_all, out_gold):
            for obj in rows:
                sec = obj.get("securityId")
                aid = obj.get("author_id")
                if sec is None or aid is None:
                    continue
                if str(sec) not in self.user_to_int:
                    continue
                a_key = f"a_{str(aid).strip().lower()}"
                if a_key not in self.entity_to_int:
                    continue
                uid = int(self.user_to_int[str(sec)])
                iid = int(self.entity_to_int[a_key])
                norm_grade, _, _ = self._normalize_label_grade(obj.get("label_grade"))
                train_y = 1 if norm_grade in (1, 2) else 0
                if train_y != 1:
                    continue
                out_all[uid].append(iid)
                if str(obj.get("label_source") or "") == "gold":
                    out_gold[uid].append(iid)

        _consume_eval(dev_rows, dev_user_all, dev_user_gold)
        _consume_eval(test_rows, test_user_all, test_user_gold)

        # 统计输出（供 trainer dry run 验收）
        stats = {
            "raw_label_grade_counts": dict(raw_grade_counts),
            "normalized_label_grade_counts": dict(norm_grade_counts),
            "raw_label_grade_out_of_range_count": int(out_of_range),
            "label_source_counts": dict(label_source_counts),
            "sample_weight_min": float(w_min) if w_min is not None else None,
            "sample_weight_max": float(w_max) if w_max is not None else None,
            "dev_gold_sample_count": int(sum(len(v) for v in dev_user_gold.values())),
            "test_gold_sample_count": int(sum(len(v) for v in test_user_gold.values())),
            "loaded_train_samples": int(len(u_list)),
            "loaded_dev_samples": int(sum(len(v) for v in dev_user_all.values())),
            "loaded_test_samples": int(sum(len(v) for v in test_user_all.values())),
            "train_gold_samples": int(sum(len(v) for v in train_user_gold.values())),
            "train_weak_samples": int(
                sum(len(v) for v in train_user_all.values()) - sum(len(v) for v in train_user_gold.values())
            ),
            "test_user_dict_all": dict(test_user_all),
            "test_user_dict_gold": dict(test_user_gold),
            "dev_user_dict_all": dict(dev_user_all),
            "dev_user_dict_gold": dict(dev_user_gold),
        }

        self.logging.info(
            "[池监督统计] raw_label_grade_counts=%s | normalized_label_grade_counts=%s | out_of_range=%s",
            stats["raw_label_grade_counts"],
            stats["normalized_label_grade_counts"],
            stats["raw_label_grade_out_of_range_count"],
        )
        self.logging.info(
            "[池监督统计] label_source_counts=%s | sample_weight_min=%.4f sample_weight_max=%.4f",
            stats["label_source_counts"],
            stats["sample_weight_min"] or -1.0,
            stats["sample_weight_max"] or -1.0,
        )
        self.logging.info(
            "[池监督统计] loaded_train/dev/test_samples=%s/%s/%s | dev_gold=%s test_gold=%s",
            stats["loaded_train_samples"],
            stats["loaded_dev_samples"],
            stats["loaded_test_samples"],
            stats["dev_gold_sample_count"],
            stats["test_gold_sample_count"],
        )

        # cf_train_data: (u,i)
        return (np.array(u_list), np.array(i_list)), train_user_all, train_user_gold, tiered, pos_w, stats

    def generate_cf_batch(self, user_dict, batch_size):
        """
        阶梯负采样：以一定概率从 fair / neutral / easy 中采负例，
        使模型学习「正样本 > 难负样本 > 易负样本」的排序。
        """
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
            n.append(
                random.choice(target_list)
                if target_list
                else random.randint(self.ENTITY_OFFSET, self.n_users_entities - 1)
            )
        u_t = torch.LongTensor(u)
        p_t = torch.LongTensor(p)
        n_t = torch.LongTensor(n)
        # 池监督：附带每个 (u,pos) 的样本权重；legacy 不返回权重
        if getattr(self, "pool_supervised_mode", False):
            w = []
            for uu, pp in zip(u_t.tolist(), p_t.tolist()):
                w.append(float(self.pos_sample_weight.get((int(uu), int(pp)), 1.0)))
            return u_t, p_t, n_t, torch.FloatTensor(w)
        return u_t, p_t, n_t

    def get_four_branch_for_batch(self, u_tensor, p_tensor, n_tensor, use_train=True):
        """
        根据四分支侧车返回当前 batch 的 author_aux / recall / interaction 特征（若有）。
        用于训练时四分支融合损失。返回 (author_aux_p, author_aux_n, recall_up, recall_un, interaction_up, interaction_un)，
        均为 float32 tensor；若侧车不存在或 key 缺失则对应为 None。
        """
        sidecar = self.four_branch_train if use_train else self.four_branch_test
        if not sidecar:
            return None, None, None, None, None, None
        u_list = u_tensor.tolist()
        p_list = p_tensor.tolist()
        n_list = n_tensor.tolist()
        B = len(u_list)
        n_recall = getattr(self.args, 'n_recall_features', 13)
        n_author = getattr(self.args, 'n_author_aux', 12)
        n_inter = getattr(self.args, 'n_interaction_features', 8)
        author_aux_p = np.zeros((B, n_author), dtype=np.float32)
        author_aux_n = np.zeros((B, n_author), dtype=np.float32)
        recall_up = np.zeros((B, n_recall), dtype=np.float32)
        recall_un = np.zeros((B, n_recall), dtype=np.float32)
        interaction_up = np.zeros((B, n_inter), dtype=np.float32)
        interaction_un = np.zeros((B, n_inter), dtype=np.float32)
        for i in range(B):
            key_p = f"u{u_list[i]}_i{p_list[i]}"
            key_n = f"u{u_list[i]}_i{n_list[i]}"
            row_p = sidecar.get(key_p, {})
            row_n = sidecar.get(key_n, {})
            for j, name in enumerate(['recall', 'author_aux', 'interaction']):
                arr_p = row_p.get(name, [])
                arr_n = row_n.get(name, [])
                if name == 'recall' and len(arr_p) >= n_recall:
                    recall_up[i] = arr_p[:n_recall]
                if name == 'recall' and len(arr_n) >= n_recall:
                    recall_un[i] = arr_n[:n_recall]
                if name == 'author_aux' and len(arr_p) >= n_author:
                    author_aux_p[i] = arr_p[:n_author]
                if name == 'author_aux' and len(arr_n) >= n_author:
                    author_aux_n[i] = arr_n[:n_author]
                if name == 'interaction' and len(arr_p) >= n_inter:
                    interaction_up[i] = arr_p[:n_inter]
                if name == 'interaction' and len(arr_n) >= n_inter:
                    interaction_un[i] = arr_n[:n_inter]
        return (
            torch.from_numpy(author_aux_p),
            torch.from_numpy(author_aux_n),
            torch.from_numpy(recall_up),
            torch.from_numpy(recall_un),
            torch.from_numpy(interaction_up),
            torch.from_numpy(interaction_un),
        )

    def load_auxiliary_info(self):
        """
        加载 AX 学术特征并做维度对齐。

        - 若 args.n_aux_features == 3：仅使用 h_index, cited_by_count, works_count（与现有 feature_index 一致）。
        - 若 args.n_aux_features == 12（或 n_author_aux == 12）：使用 README 5.2 的 AUTHOR_AUX_COLUMNS；
          当前 build_feature_index 仍只产出 3 维，缺失维用 0 填充；待特征管线扩展后可在此处按 12 列读取。
        """
        n_aux = getattr(self.args, 'n_aux_features', 3)
        features = np.zeros((self.n_users_entities, n_aux), dtype=np.float32)

        if not os.path.exists(FEATURE_INDEX_PATH):
            self.aux_info_all = torch.from_numpy(features)
            self.logging.info("[*] 未找到特征索引，AX 特征置零。")
            return

        with open(FEATURE_INDEX_PATH, 'r', encoding='utf-8') as f:
            feature_bundle = json.load(f)

        author_features = feature_bundle.get('author', {})
        match_count = 0

        for raw_aid, feat_dict in author_features.items():
            key = f"a_{str(raw_aid).strip().lower()}"
            if key not in self.entity_to_int:
                continue
            idx = self.entity_to_int[key]
            if n_aux == 3:
                features[idx] = [
                    feat_dict.get('h_index', 0.0),
                    feat_dict.get('cited_by_count', 0.0),
                    feat_dict.get('works_count', 0.0),
                ]
            else:
                for i in range(min(n_aux, N_AUTHOR_AUX_V2)):
                    if i == 0:
                        val = feat_dict.get('h_index', 0.0)
                    elif i == 1:
                        val = feat_dict.get('cited_by_count', 0.0)
                    elif i == 2:
                        val = feat_dict.get('works_count', 0.0)
                    else:
                        val = feat_dict.get(AUTHOR_AUX_COLUMNS_V2[i], 0.0)
                    features[idx, i] = val
            match_count += 1

        self.aux_info_all = torch.from_numpy(features)
        self.logging.info(f"[*] AX 特征加载完成，维度 {n_aux}，命中 {match_count} 位作者。")

    def load_kg(self, filename):
        """
        加权索引驱动采样（定向 2-Hop 语义增强）：
        - 阶段 1：以 CF 种子节点做 1-Hop 采样，带权重 w；
        - 阶段 2：对子图中的 Vocabulary 节点补全 SIMILAR_TO(r=6) 边，织入语义桥梁。
        关系总数与 builder 约定一致（含反向边）。
        """
        db_path = os.path.join(self.data_dir, "kg_index.db")
        if not os.path.exists(db_path):
            self.logging.error(f"[!] 找不到加权索引库: {db_path}")
            return pd.DataFrame()

        self.logging.info("[*] 执行加权子图采样 (正在织入语义桥梁)...")

        seeds = list(set(self.cf_train_data[0].tolist()) | set(self.cf_train_data[1].tolist()))
        conn = sqlite3.connect(db_path)
        all_edges = []
        vocab_nodes = set()
        chunk_size = 500

        for i in range(0, len(seeds), chunk_size):
            batch = seeds[i : i + chunk_size]
            placeholders = ','.join(['?'] * len(batch))
            sql = f"SELECT h, r, t, w FROM kg_triplets WHERE h IN ({placeholders}) OR t IN ({placeholders})"
            res = conn.execute(sql, batch + batch).fetchall()
            all_edges.extend(res)
            for h, r, t, w in res:
                if r in [4, 5]:
                    vocab_nodes.add(t)

        if vocab_nodes:
            self.logging.info(f"[*] 发现 {len(vocab_nodes)} 个词汇节点，正在注入 SIMILAR_TO 语义关系...")
            v_list = list(vocab_nodes)
            for i in range(0, len(v_list), chunk_size):
                batch = v_list[i : i + chunk_size]
                placeholders = ','.join(['?'] * len(batch))
                sql = f"SELECT h, r, t, w FROM kg_triplets WHERE r=6 AND h IN ({placeholders}) AND t IN ({placeholders})"
                bridge_res = conn.execute(sql, batch + batch).fetchall()
                all_edges.extend(bridge_res)

        conn.close()
        df_final = pd.DataFrame(all_edges, columns=['h', 'r', 't', 'w']).drop_duplicates()

        if not df_final.empty and 6 in df_final['r'].unique():
            self.logging.info("[OK] 语义桥梁注入成功，子图包含 SIMILAR_TO 关系。")
        else:
            self.logging.warning("[!] 警告：子图中仍未发现 SIMILAR_TO 关系，请检查 kg_final.txt 数据质量。")

        return df_final

    def construct_data(self, kg_data):
        """
        构建带权关系字典与反向边，将权重 w 存入供拉普拉斯矩阵使用。
        存储格式：(目标节点, 关系 id, 权重 w)。
        """
        if kg_data.empty:
            return
        n_rel = max(kg_data['r']) + 1
        inv_kg = kg_data.copy().rename(columns={'h': 't', 't': 'h'})
        inv_kg['r'] += n_rel
        kg_data = pd.concat([kg_data, inv_kg], ignore_index=True)

        self.n_relations = int(max(kg_data['r']) + 1)
        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in kg_data.itertuples(index=False):
            self.train_kg_dict[row.h].append((row.t, row.r, row.w))
            self.train_relation_dict[row.r].append((row.h, row.t, row.w))

        self.h_list = torch.LongTensor(kg_data['h'].values)
        self.t_list = torch.LongTensor(kg_data['t'].values)
        self.r_list = torch.LongTensor(kg_data['r'].values)
        self.n_kg_train = len(kg_data)

    def create_laplacian_dict(self):
        """
        生成归一化加权拉普拉斯矩阵。
        使用真实边权 w（来自 builder 的 pos_weight 等），实现时序衰减与语义权重的物理隔离。
        """
        cache_file = os.path.join(self.data_dir, 'weighted_adj_cache.npz')
        if os.path.exists(cache_file):
            adj = sp.load_npz(cache_file)
            self.A_in = self.convert_coo2tensor(adj)
            return

        rows, cols, datas = [], [], []
        for r, htw in self.train_relation_dict.items():
            for h, t, w in htw:
                rows.append(h)
                cols.append(t)
                datas.append(float(w))

        adj = sp.coo_matrix(
            (datas, (rows, cols)),
            shape=(self.n_users_entities, self.n_users_entities),
        )
        rowsum = np.array(adj.sum(axis=1))
        d_inv = np.power(rowsum, -1.0).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        norm_adj = sp.diags(d_inv).dot(adj).tocoo()

        sp.save_npz(cache_file, norm_adj)
        self.A_in = self.convert_coo2tensor(norm_adj)

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        """
        带权三元组采样：从 kg_dict 中采 (h, r, t, w)，负样本随机采尾实体。
        返回 h, r, p, n 及对应 aux 向量，供 calc_kg_loss 使用。
        """
        h_batch = np.random.choice(list(kg_dict.keys()), batch_size)
        h, r, p, n = [], [], [], []
        for head in h_batch:
            t, rel, weight = random.choice(kg_dict[head])
            h.append(head)
            r.append(rel)
            p.append(t)
            n.append(random.randint(self.ENTITY_OFFSET, self.n_users_entities - 1))
        return (
            torch.LongTensor(h),
            torch.LongTensor(r),
            torch.LongTensor(p),
            torch.LongTensor(n),
            self.aux_info_all[h],
            self.aux_info_all[p],
            self.aux_info_all[n],
        )

    def convert_coo2tensor(self, coo):
        """将 scipy 稀疏 COO 矩阵转为 PyTorch sparse tensor。"""
        idx = torch.LongTensor(np.vstack((coo.row, coo.col)))
        return torch.sparse_coo_tensor(
            idx,
            torch.FloatTensor(coo.data),
            torch.Size(coo.shape),
        )

    def sanity_check(self):
        """数据质量诊断：节点空间、关系数、边数、AX 维度、ID 边界。"""
        self.logging.info(f"\n{'=' * 10} 加权数据校验 {'=' * 10}")
        self.logging.info(f"节点空间: {self.n_users_entities} | 关系数: {self.n_relations}")
        self.logging.info(f"KG 边数: {self.n_kg_train} | AX 特征维度: {self.aux_info_all.shape}")
        if self.A_in._indices().max().item() >= self.n_users_entities:
            raise ValueError("ID 越界！")
        self.logging.info("[Success] 加权数据载入流程完成。")
