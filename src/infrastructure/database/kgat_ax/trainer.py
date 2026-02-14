import sys
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import logging
import scipy.sparse as sp

from model import KGAT
from data_loader import DataLoaderKGAT
from kgat_utils.metrics import calc_metrics_at_k
from kgat_utils.log_helper import create_log_id, logging_config
from kgat_utils.model_helper import save_model, early_stopping
from kgat_parser.parser_kgat import parse_kgat_args
from config import KGATAX_TRAIN_DATA_DIR, DB_PATH


def check_args_and_env(args):
    """前置参数检查：踩坑点提前预防"""
    print("\n[System Check] 正在执行环境预检...")

    # 1. 路径检查
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"错误: 数据库文件不存在于 {DB_PATH}")

    # 2. 内存风险预警 (针对 31.7GB 内存)
    if hasattr(args, 'embed_dim') and args.embed_dim > 128:
        print("警告: Embedding 维度较高，建议在 A_in 矩阵构建时监控内存使用。")

    # 3. 参数校验
    try:
        ks = eval(args.Ks)
        if not isinstance(ks, list): raise ValueError
    except:
        raise ValueError("错误: 参数 Ks 必须是列表格式，例如 '[20, 50]'")

    # 4. 显卡检查
    if torch.cuda.is_available():
        print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: 未检测到可用 GPU，将使用 CPU 训练，速度会非常慢。")
    print("[System Check] 预检通过！\n")


def evaluate(model, dataloader, Ks, device):
    """
    高性能评估函数
    重构逻辑：预计算 Embedding + 批量点积，彻底消除重复的图聚合计算
    """
    model.eval()
    test_user_dict = dataloader.test_user_dict
    user_ids = list(test_user_dict.keys())

    # 1. 显存优化：一次性将辅助特征搬运到 GPU
    aux_info_all = dataloader.aux_info_all.to(device)

    # 2. 核心加速：预计算所有节点的最终 Embedding
    # 不再在下方的 tqdm 循环里调用 model(...)，因为 model 内部包含耗时的图卷积
    with torch.no_grad():
        all_embed = model.calc_cf_embeddings(aux_info_all)

        # 3. 分批计算 User-Item 评分
    # 将 User 分批以防止 scores 矩阵 (Batch_U x All_Items) 撑爆显存
    batch_size = dataloader.test_batch_size
    user_batches = [user_ids[i:i + batch_size] for i in range(0, len(user_ids), batch_size)]

    metrics = {k: {'recall': [], 'ndcg': []} for k in Ks}
    item_ids_vec = np.arange(dataloader.n_users_entities)

    for batch_u in tqdm(user_batches, desc='Eval', leave=False):
        batch_u_gpu = torch.LongTensor(batch_u).to(device)

        with torch.no_grad():
            # 极速矩阵点积计算得分
            # scores 形状: [len(batch_u), n_users_entities]
            scores = torch.matmul(all_embed[batch_u_gpu], all_embed.transpose(0, 1)).cpu()

        # 计算 Top-K 指标
        batch_m = calc_metrics_at_k(scores, dataloader.train_user_dict, test_user_dict,
                                    np.array(batch_u), item_ids_vec, Ks)

        for k in Ks:
            metrics[k]['recall'].append(batch_m[k]['recall'])
            metrics[k]['ndcg'].append(batch_m[k]['ndcg'])

    return {k: {m: np.mean(v) for m, v in val.items()} for k, val in metrics.items()}


def train(args):
    # 初始化路径与环境
    args.save_dir = KGATAX_TRAIN_DATA_DIR
    args.data_dir = os.path.dirname(KGATAX_TRAIN_DATA_DIR)
    args.data_name = os.path.basename(KGATAX_TRAIN_DATA_DIR)
    if not hasattr(args, 'db_path'): args.db_path = DB_PATH

    check_args_and_env(args)
    logging_config(folder=args.save_dir, name='log_fast', no_console=False)

    # --- 修改 1: 强制使用 CPU ---
    device = torch.device("cpu")
    print("[*] 运行模式: 强制 CPU 训练")

    # --- 数据加载 ---
    data = DataLoaderKGAT(args, logging)
    args.ENTITY_OFFSET = data.ENTITY_OFFSET

    # 确保模型在 CPU 上
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations, data.A_in).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- 断点续训与权重管理 ---
    start_epoch = 1
    best_recall = 0
    recall_list = []
    weight_path = os.path.join(args.save_dir, 'weights')
    if not os.path.exists(weight_path): os.makedirs(weight_path)

    latest_models = sorted([f for f in os.listdir(weight_path) if f.endswith('.pth')])
    if latest_models:
        last_model = os.path.join(weight_path, latest_models[-1])
        try:
            checkpoint = torch.load(last_model, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"成功加载现有权重: {last_model}")
        except Exception as e:
            print(f"权重加载失败，将从头开始训练。错误: {e}")

    Ks = eval(args.Ks)

    # --- 训练主循环 ---
    for epoch in range(start_epoch, args.n_epoch + 1):
        model.train()
        # 确保辅助特征在 CPU 上
        aux_all_cpu = data.aux_info_all.to(device)

        # --- 修改 2: 在 CF 阶段加入 tqdm 进度条 ---
        n_cf = data.n_cf_train // args.cf_batch_size + 1
        loss_cf = 0
        cf_pbar = tqdm(range(n_cf), desc=f'Epoch {epoch} [CF Phase]', leave=False)
        for _ in cf_pbar:
            u, p, n = data.generate_cf_batch(data.train_user_dict, args.cf_batch_size)
            # 所有 tensor 都在 CPU 上
            loss = model(u.to(device), p.to(device), n.to(device), aux_all_cpu, mode='train_cf')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_cf += loss.item()
            # 动态显示当前 Batch 的 Loss
            cf_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # --- 修改 3: 在 KG 阶段加入 tqdm 进度条 ---
        n_kg = data.n_kg_train // args.kg_batch_size + 1
        loss_kg = 0
        kg_pbar = tqdm(range(n_kg), desc=f'Epoch {epoch} [KG Phase]', leave=False)
        for _ in kg_pbar:
            h, r, pt, nt, ha, pa, na = [d.to(device) for d in
                                        data.generate_kg_batch(data.train_kg_dict, args.kg_batch_size,
                                                               data.n_users_entities)]
            loss = model(h, r, pt, nt, ha, pa, na, mode='train_kg')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_kg += loss.item()
            kg_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 3. 注意力更新
        if epoch == 1 or epoch % 10 == 0:
            with torch.no_grad():
                model(data.h_list.to(device), data.t_list.to(device), data.r_list.to(device), mode='update_att')

        # 日志记录
        avg_cf_loss = loss_cf / n_cf
        avg_kg_loss = loss_kg / n_kg
        logging.info(f'Epoch {epoch} | CF Loss: {avg_cf_loss:.4f} | KG Loss: {avg_kg_loss:.4f}')

        # 4. 评估与早停
        if epoch % args.evaluate_every == 0:
            # evaluate 函数内部也应确保使用 CPU
            res = evaluate(model, data, Ks, device)
            curr_recall = res[Ks[0]]['recall']
            logging.info(f"Eval @{Ks[0]}: Recall: {curr_recall:.4f}, NDCG: {res[Ks[0]]['ndcg']:.4f}")

            recall_list.append(curr_recall)
            _, stop = early_stopping(recall_list, args.stopping_steps)

            if curr_recall > best_recall:
                best_recall = curr_recall
                save_model(model, weight_path, epoch)
                print(f"[*] 发现更佳 Recall: {best_recall:.4f}，模型已保存。")

            if stop:
                logging.info(f"触发早停，训练提前结束。")
                break

if __name__ == '__main__':
    train(parse_kgat_args())