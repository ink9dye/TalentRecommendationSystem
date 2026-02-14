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

# 导入核心组件
from model import KGAT
from data_loader import DataLoaderKGAT

# 引用工具模块
from kgat_utils.metrics import calc_metrics_at_k
from kgat_utils.log_helper import create_log_id, logging_config
from kgat_utils.model_helper import save_model, early_stopping

# 引用参数解析器
from kgat_parser.parser_kgat import parse_kgat_args

# 【严格引用 config】确保路径与 generate_training_data 保持一致
from config import KGATAX_TRAIN_DATA_DIR, DB_PATH


def evaluate(model, dataloader, Ks, device):
    """
    精排评估函数：利用全息增强后的 Embedding 进行 NDCG 和 Recall 计算
    """
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()
    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    # 在全图统一 ID 模式下，item_ids 为全图实体索引
    n_items = dataloader.n_users_entities
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    aux_info_all = dataloader.aux_info_all.to(device)

    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating') as pbar:
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
    return metrics_dict


def train(args):
    """
    训练主函数：协同训练 CF Loss 与 KG Loss
    """
    args.save_dir = KGATAX_TRAIN_DATA_DIR
    args.data_dir = os.path.dirname(KGATAX_TRAIN_DATA_DIR)
    args.data_name = os.path.basename(KGATAX_TRAIN_DATA_DIR)

    if not hasattr(args, 'db_path') or args.db_path == 'datasets/talent_system.db':
        args.db_path = DB_PATH

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. 加载数据
    data = DataLoaderKGAT(args, logging)

    # 2. 构造模型
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations, data.A_in)
    model.to(device)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    recall_list = []
    best_recall = 0
    Ks = eval(args.Ks)
    k_min = Ks[0]

    weight_path = os.path.join(args.save_dir, 'weights')
    if not os.path.exists(weight_path): os.makedirs(weight_path)

    # 3. 迭代训练
    for epoch in range(1, args.n_epoch + 1):
        model.train()
        aux_info_all = data.aux_info_all.to(device)

        # --- 第一阶段：CF 训练进度条 ---
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1
        avg_cf_loss = 0
        cf_pbar = tqdm(total=n_cf_batch, desc=f'Epoch {epoch:03d} [CF]', leave=False)

        for _ in range(n_cf_batch):
            u, p, n = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            cf_loss = model(u.to(device), p.to(device), n.to(device), aux_info_all, mode='train_cf')

            cf_optimizer.zero_grad()
            cf_loss.backward()
            cf_optimizer.step()
            avg_cf_loss += cf_loss.item()
            cf_pbar.update(1)
        cf_pbar.close()

        # --- 第二阶段：KG 训练进度条 ---
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1
        avg_kg_loss = 0
        kg_pbar = tqdm(total=n_kg_batch, desc=f'Epoch {epoch:03d} [KG]', leave=False)

        for _ in range(n_kg_batch):
            batch_data = data.generate_kg_batch(data.train_kg_dict, data.kg_batch_size, data.n_users_entities)
            h, r, pt, nt, h_aux, pt_aux, nt_ax = [d.to(device) for d in batch_data]

            kg_loss = model(h, r, pt, nt, h_aux, pt_aux, nt_ax, mode='train_kg')

            kg_optimizer.zero_grad()
            kg_loss.backward()
            kg_optimizer.step()
            avg_kg_loss += kg_loss.item()
            kg_pbar.update(1)
        kg_pbar.close()

        # --- 第三阶段：更新注意力 ---
        with torch.no_grad():
            model(data.h_list.to(device), data.t_list.to(device), data.r_list.to(device), mode='update_att')

        logging.info(
            f'Epoch {epoch:04d} | CF Loss: {avg_cf_loss / n_cf_batch:.4f} | KG Loss: {avg_kg_loss / n_kg_batch:.4f}')

        # --- 第四阶段：评估 ---
        if (epoch % args.evaluate_every) == 0:
            metrics_dict = evaluate(model, data, Ks, device)
            current_recall = metrics_dict[k_min]['recall']
            recall_list.append(current_recall)

            best_recall_tmp, should_stop = early_stopping(recall_list, args.stopping_steps)

            logging.info(
                f'Evaluation: Recall@{k_min}: {current_recall:.4f} | NDCG@{k_min}: {metrics_dict[k_min]["ndcg"]:.4f}')

            if current_recall >= best_recall:
                best_recall = current_recall
                save_model(model, weight_path, epoch)
                logging.info(f'[*] Best model updated and saved at epoch {epoch}')

            if should_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break


if __name__ == '__main__':
    args = parse_kgat_args()
    train(args)