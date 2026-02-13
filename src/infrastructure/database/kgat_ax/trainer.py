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
    # 分批次处理以防止内存/显存溢出
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    # KGAT-AX 核心：获取全图辅助学术指标执行精排预测 [cite: 10, 222]
    # aux_info_all 在模型前向传播中用于对实体进行 holographic fusion
    aux_info_all = dataloader.aux_info_all.to(device)

    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)
            with torch.no_grad():
                # 预测阶段调用 calc_score 模式，传入辅助信息 Tensor
                batch_scores = model(batch_user_ids, item_ids, aux_info_all, mode='predict')

            batch_scores = batch_scores.cpu()
            # 执行 Recall@K 和 NDCG@K 评估 [cite: 557, 656]
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
    训练主函数：协同训练 CF Loss 与 KG Loss [cite: 508, 514]
    """
    # 1. 强制对齐路径，确保读取 generate_training_data 生成的文本文件
    args.save_dir = KGATAX_TRAIN_DATA_DIR
    args.data_dir = os.path.dirname(KGATAX_TRAIN_DATA_DIR)
    args.data_name = os.path.basename(KGATAX_TRAIN_DATA_DIR)

    # 确保 db_path 默认指向学术数据库
    if not hasattr(args, 'db_path') or args.db_path == 'datasets/talent_system.db':
        args.db_path = DB_PATH

    # 2. 随机种子设定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 3. 日志系统初始化
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)

    # 针对 Intel 核显环境：由于无 CUDA，默认回退至 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 4. 加载数据：此步骤会调用 load_kg 读取文本格式三元组并从 SQLite 提取 AX 指标 [cite: 49, 133]
    data = DataLoaderKGAT(args, logging)

    # 5. 构造 KGAT-AX 模型：包含注意力机制和全息融合层 [cite: 7, 10]
    # A_in 将以稀疏 Tensor 形式参与计算 [cite: 71, 440]
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations, data.A_in)
    model.to(device)

    # 定义优化器 [cite: 173, 514]
    cf_optimizer = optim.Adam(model.parameters(), lr=args.lr)
    kg_optimizer = optim.Adam(model.parameters(), lr=args.lr)

    recall_list = []
    best_recall = 0
    Ks = eval(args.Ks)
    k_min = Ks[0]

    weight_path = os.path.join(args.save_dir, 'weights')
    if not os.path.exists(weight_path): os.makedirs(weight_path)

    # 6. 迭代训练循环
    for epoch in range(1, args.n_epoch + 1):
        model.train()

        # --- 第一阶段：CF 训练 (利用全息融合注入全图 AX 数据) [cite: 157, 492] ---
        aux_info_all = data.aux_info_all.to(device)
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1
        avg_cf_loss = 0
        for _ in range(n_cf_batch):
            u, p, n = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
            # 计算融合了学术指标的协同过滤损失 [cite: 144, 504]
            cf_loss = model(u.to(device), p.to(device), n.to(device), aux_info_all, mode='train_cf')

            cf_optimizer.zero_grad()
            cf_loss.backward()
            cf_optimizer.step()
            avg_cf_loss += cf_loss.item()

        # --- 第二阶段：KG 训练 (注入三元组特异性 AX 特征) [cite: 105, 436] ---
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1
        avg_kg_loss = 0
        for _ in range(n_kg_batch):
            # generate_kg_batch 返回三元组及对应的辅助特征 [cite: 159]
            batch_data = data.generate_kg_batch(data.train_kg_dict, data.kg_batch_size, data.n_users_entities)
            h, r, pt, nt, h_aux, pt_aux, nt_ax = [d.to(device) for d in batch_data]

            # 计算知识图谱三元组损失 [cite: 105, 436]
            kg_loss = model(h, r, pt, nt, h_aux, pt_aux, nt_ax, mode='train_kg')

            kg_optimizer.zero_grad()
            kg_loss.backward()
            kg_optimizer.step()
            avg_kg_loss += kg_loss.item()

        # --- 第三阶段：显式更新图注意力权重 [cite: 8, 359] ---
        # 注意：此处基于 CKG 结构递归传播嵌入 [cite: 358, 421]
        with torch.no_grad():
            model(data.h_list.to(device), data.t_list.to(device), data.r_list.to(device), mode='update_att')

        logging.info(
            f'Epoch {epoch:04d} | CF Loss: {avg_cf_loss / n_cf_batch:.4f} | KG Loss: {avg_kg_loss / n_kg_batch:.4f}')

        # --- 第四阶段：评估与早停保护 [cite: 176, 576] ---
        if (epoch % args.evaluate_every) == 0:
            metrics_dict = evaluate(model, data, Ks, device)
            current_recall = metrics_dict[k_min]['recall']
            recall_list.append(current_recall)

            # 执行早停逻辑，防止过拟合 [cite: 176, 576]
            best_recall_tmp, should_stop = early_stopping(recall_list, args.stopping_steps)

            logging.info(
                f'Evaluation: Recall@{k_min}: {current_recall:.4f} | NDCG@{k_min}: {metrics_dict[k_min]["ndcg"]:.4f}')

            # 如果当前召回率达到最优，保存权重 [cite: 12]
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