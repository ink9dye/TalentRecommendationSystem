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
# 【全模块封闭化：相对路径引用与工具集成】
# ====================================================
from .model import KGAT
from .data_loader import DataLoaderKGAT

# 引用你提供的 utils 模块
from .utils.metrics import calc_metrics_at_k
from .utils.log_helper import create_log_id, logging_config
from .utils.model_helper import save_model, early_stopping

# 引用参数解析器
from .parser.parser_kgat import parse_kgat_args

# 【严格引用 config】确保存储路径唯一性
from config import KGATAX_TRAIN_DATA_DIR


def evaluate(model, dataloader, Ks, device):
    """
    精排评估函数：利用全息增强后的 Embedding 进行 NDCG 和 Recall 计算
    """
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()
    user_ids = list(test_user_dict.keys())
    # 分批次处理以防止显存溢出
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    # KGAT-AX 核心：获取全图辅助学术指标执行精排预测
    aux_info_all = dataloader.aux_info_all.to(device)

    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)
            with torch.no_grad():
                # 预测阶段同样需要 aux_info 进行特征对齐
                batch_scores = model(batch_user_ids, item_ids, aux_info_all, mode='predict')

            batch_scores = batch_scores.cpu()
            # 调用 metrics.py 中的核心评估函数
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict,
                                              batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)

    # 聚合所有 Batch 的结果并取均值
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return metrics_dict


def train(args):
    """
    训练主函数：协同训练 CF Loss 与 KG Loss
    """
    # 1. 强制使用 config 中的路径，覆盖命令行输入以防出错
    args.save_dir = KGATAX_TRAIN_DATA_DIR
    args.data_dir = os.path.dirname(KGATAX_TRAIN_DATA_DIR)
    args.data_name = os.path.basename(KGATAX_TRAIN_DATA_DIR)

    # 2. 随机种子设定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 3. 日志系统初始化
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. 加载数据 (DataLoader 会读取 kgatax_train_data 下的 train.txt/id_map.json)
    data = DataLoaderKGAT(args, logging)

    # 5. 构造 KGAT-AX 模型
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations, data.A_in)
    model.to(device)

    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    recall_list = []  # 用于早停监控
    best_recall = 0
    Ks = eval(args.Ks)
    k_min = Ks[0]  # 监控 Recall@20

    # 权重保存目录锁定在 kgatax_train_data/weights/
    weight_path = os.path.join(args.save_dir, 'weights')
    if not os.path.exists(weight_path): os.makedirs(weight_path)

    # 6. 迭代训练循环
    for epoch in range(1, args.n_epoch + 1):
        model.train()

        # --- 第一阶段：CF 训练 (全息融合注入全图 AX 数据) ---
        aux_info_all = data.aux_info_all.to(device)
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1
        avg_cf_loss = 0
        for _ in range(n_cf_batch):
            u, p, n = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            # 注入学术指标
            cf_loss = model(u.to(device), p.to(device), n.to(device), aux_info_all, mode='train_cf')

            cf_optimizer.zero_grad()
            cf_loss.backward()
            cf_optimizer.step()
            avg_cf_loss += cf_loss.item()

        # --- 第二阶段：KG 训练 (注入三元组特异性 AX 特征) ---
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1
        avg_kg_loss = 0
        for _ in range(n_kg_batch):
            batch_data = data.generate_kg_batch(data.train_kg_dict, data.kg_batch_size, data.n_users_entities)
            h, r, pt, nt, h_aux, pt_aux, nt_ax = [d.to(device) for d in batch_data]

            # 计算受 AX 增强影响的 KG Loss
            kg_loss = model(h, r, pt, nt, h_aux, pt_aux, nt_ax, mode='train_kg')

            kg_optimizer.zero_grad()
            kg_loss.backward()
            kg_optimizer.step()
            avg_kg_loss += kg_loss.item()

        # --- 第三阶段：显式更新图注意力权重 ---
        with torch.no_grad():
            model(data.h_list.to(device), data.t_list.to(device), data.r_list.to(device), mode='update_att')

        logging.info(
            f'Epoch {epoch:04d} | CF Loss: {avg_cf_loss / n_cf_batch:.4f} | KG Loss: {avg_kg_loss / n_kg_batch:.4f}')

        # --- 第四阶段：评估与早停保护 ---
        if (epoch % args.evaluate_every) == 0:
            metrics_dict = evaluate(model, data, Ks, device)
            current_recall = metrics_dict[k_min]['recall']
            recall_list.append(current_recall)

            # 调用 model_helper 执行早停逻辑
            best_recall_tmp, should_stop = early_stopping(recall_list, args.stopping_steps)

            logging.info(
                f'Evaluation: Recall@{k_min}: {current_recall:.4f} | NDCG@{k_min}: {metrics_dict[k_min]["ndcg"]:.4f}')

            # 如果当前 Recall 是历史最高，保存模型
            if current_recall >= best_recall:
                best_recall = current_recall
                save_model(model, weight_path, epoch)
                logging.info(f'[*] Best model updated and saved at epoch {epoch}')

            if should_stop:
                logging.info(f"Early stopping at epoch {epoch} (Best Recall: {best_recall:.4f})")
                break


if __name__ == '__main__':
    args = parse_kgat_args()
    train(args)