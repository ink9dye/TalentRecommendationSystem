import sys
import os
import random
from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.optim as optim
import logging

# ====================================================
# 【全模块封闭化：全部使用相对路径引用】
# ====================================================
from .model import KGAT
from .data_loader import DataLoaderKGAT

# 从内部的 utils 和 parser 文件夹导入
from .utils.metrics import calc_metrics_at_k
from .utils.log_helper import create_log_id, logging_config
from .utils.model_helper import load_model, early_stopping
from .parser.parser_kgat import parse_kgat_args


def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()
    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    # KGAT-AX：获取全图辅助信息执行精排预测
    aux_info_all = dataloader.aux_info_all.to(device)

    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)
            with torch.no_grad():
                batch_scores = model(batch_user_ids, item_ids, aux_info_all, mode='predict')

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict,
                                              batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return None, metrics_dict


def train(args):
    # 随机种子设定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = DataLoaderKGAT(args, logging)

    # 构造 KGAT-AX 模型
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations, data.A_in)
    model.to(device)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_epoch, best_recall = -1, 0
    Ks = eval(args.Ks)
    k_min = min(Ks)

    # 权重保存路径
    weight_path = os.path.join(os.path.dirname(__file__), 'weights')
    if not os.path.exists(weight_path): os.makedirs(weight_path)

    for epoch in range(1, args.n_epoch + 1):
        model.train()

        # --- 1. CF 训练 (全息融合注入全图 AX 数据) ---
        aux_info_all = data.aux_info_all.to(device)
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1
        for _ in range(n_cf_batch):
            u, p, n = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            cf_loss = model(u.to(device), p.to(device), n.to(device), aux_info_all, mode='train_cf')
            cf_optimizer.zero_grad();
            cf_loss.backward();
            cf_optimizer.step()

        # --- 2. KG 训练 (注入增强三元组 AX 特征) ---
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1
        for _ in range(n_kg_batch):
            batch_data = data.generate_kg_batch(data.train_kg_dict, data.kg_batch_size, data.n_users_entities)
            h, r, pt, nt, h_aux, pt_aux, nt_ax = [d.to(device) for d in batch_data]
            kg_loss = model(h, r, pt, nt, h_aux, pt_aux, nt_ax, mode='train_kg')
            kg_optimizer.zero_grad();
            kg_loss.backward();
            kg_optimizer.step()

        # --- 3. 更新图注意力系数 ---
        model(data.h_list.to(device), data.t_list.to(device), data.r_list.to(device),
              list(data.laplacian_dict.keys()), mode='update_att')

        # --- 4. 评估与保存 ---
        if (epoch % args.evaluate_every) == 0:
            _, metrics_dict = evaluate(model, data, Ks, device)
            logging.info(f'Epoch {epoch:04d} Recall@{k_min}: {metrics_dict[k_min]["recall"]:.4f}')

            if metrics_dict[k_min]['recall'] > best_recall:
                best_recall = metrics_dict[k_min]['recall']
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(weight_path, 'best_model.pth'))


if __name__ == '__main__':
    args = parse_kgat_args()
    train(args)