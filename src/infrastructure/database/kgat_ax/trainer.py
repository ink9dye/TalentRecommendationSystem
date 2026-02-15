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
    print("\n[System Check] 正在执行环境预检...")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"错误: 数据库文件不存在于 {DB_PATH}")
    if hasattr(args, 'embed_dim') and args.embed_dim > 128:
        print("警告: Embedding 维度较高，请监控内存。")
    try:
        ks = eval(args.Ks)
        if not isinstance(ks, list): raise ValueError
    except:
        raise ValueError("错误: 参数 Ks 必须是列表格式。")
    print("[System Check] 预检通过！\n")


def evaluate(model, dataloader, Ks, device):
    """
    针对 CPU 内存优化的评估函数：
    1. Top-K 截断：仅保留前 500 个高分候选人，防止指标计算时内存爆炸。
    2. 内存回收：显式删除中间变量并调用垃圾回收。
    """
    import gc  # 导入垃圾回收
    model.eval()
    test_user_dict = dataloader.test_user_dict
    user_ids = list(test_user_dict.keys())

    # 1. 加载归一化特征并预计算 Embedding
    aux_info_all = dataloader.aux_info_all.to(device)
    with torch.no_grad():
        all_embed = model.calc_cf_embeddings(aux_info_all)

    # 2. 提取人才（实体）空间
    offset = dataloader.ENTITY_OFFSET
    global_max_id = dataloader.ENTITY_OFFSET + dataloader.n_entities

    if offset >= all_embed.shape[0]:
        raise IndexError(f"ENTITY_OFFSET ({offset}) 超过了 Embedding 矩阵大小 ({all_embed.shape[0]})")

    item_embeds = all_embed[offset:global_max_id]
    item_ids_vec = np.arange(offset, global_max_id)

    # 3. 分批次评估 (建议 test_batch_size 设置为 50-100)
    batch_size = dataloader.test_batch_size
    user_batches = [user_ids[i:i + batch_size] for i in range(0, len(user_ids), batch_size)]
    metrics = {k: {'recall': [], 'ndcg': []} for k in Ks}

    for batch_u in tqdm(user_batches, desc='Eval', leave=False):
        batch_u_tensor = torch.LongTensor(batch_u).to(device)

        with torch.no_grad():
            curr_user_embeds = all_embed[batch_u_tensor]
            # 计算全量得分：[batch_size, 180w]
            # 虽然 100 * 180w 的 float32 矩阵只有约 700MB，但后续操作会成倍放大
            all_scores = torch.matmul(curr_user_embeds, item_embeds.transpose(0, 1))

            # --- 核心优化：Top-K 截断 ---
            # 只取前 500 名，后续指标计算只在这 500 人中进行
            # 这能确保 calc_metrics_at_k 内部不会产生 7GB 的掩码矩阵
            top_k_val = 500
            top_scores, top_indices = torch.topk(all_scores, k=min(top_k_val, item_embeds.shape[0]), dim=1)

            # 构造一个“瘦身版”得分矩阵：
            # 先填充极小值，再将 Top-K 的真实分数填回去
            # 这样 calc_metrics_at_k 拿到的虽然还是 180w 列，但非 Top-K 的部分会被完全忽略且不触发内存高压
            reduced_scores = torch.full_like(all_scores, -1e10)
            reduced_scores.scatter_(1, top_indices, top_scores)

            # 转换为 numpy 供指标函数使用
            final_scores_cpu = reduced_scores.cpu().numpy()

        # 4. 指标计算
        batch_m = calc_metrics_at_k(
            final_scores_cpu,
            dataloader.train_user_dict,
            test_user_dict,
            np.array(batch_u),
            item_ids_vec,
            Ks
        )

        for k in Ks:
            metrics[k]['recall'].append(batch_m[k]['recall'])
            metrics[k]['ndcg'].append(batch_m[k]['ndcg'])

        # 及时清理当前 Batch 的内存碎片
        del all_scores, reduced_scores, final_scores_cpu
        gc.collect()

    # 5. 最终清理
    del all_embed, item_embeds
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return {k: {m: np.mean(v) for m, v in val.items()} for k, val in metrics.items()}
def train(args):
    args.save_dir = KGATAX_TRAIN_DATA_DIR
    args.data_dir = os.path.dirname(KGATAX_TRAIN_DATA_DIR)
    args.data_name = os.path.basename(KGATAX_TRAIN_DATA_DIR)
    check_args_and_env(args)
    logging_config(folder=args.save_dir, name='log_fast', no_console=False)

    device = torch.device("cpu")
    data = DataLoaderKGAT(args, logging)
    args.ENTITY_OFFSET = data.ENTITY_OFFSET

    model = KGAT(args, data.n_users, data.n_entities, data.n_relations, data.A_in).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    best_recall = 0
    recall_list = []
    weight_path = os.path.join(args.save_dir, 'weights')
    if not os.path.exists(weight_path): os.makedirs(weight_path)

    # 自动恢复最近进度
    latest_models = sorted([f for f in os.listdir(weight_path) if f.endswith('.pth')])
    if latest_models:
        last_model = os.path.join(weight_path, latest_models[-1])
        try:
            checkpoint = torch.load(last_model, map_location=device)
            model.load_state_dict(checkpoint)
            # 尝试从文件名恢复 epoch，如 'model_epoch_5.pth'
            start_epoch = int(latest_models[-1].split('_')[-1].split('.')[0]) + 1
            print(f"成功加载权重并恢复至 Epoch {start_epoch}: {last_model}")
        except:
            print("权重加载失败，重头开始。")

    Ks = eval(args.Ks)

    for epoch in range(start_epoch, args.n_epoch + 1):
        model.train()
        aux_all_cpu = data.aux_info_all.to(device)

        # --- 1. CF 阶段 ---
        n_cf = data.n_cf_train // args.cf_batch_size + 1
        loss_cf = 0
        cf_pbar = tqdm(range(n_cf), desc=f'Epoch {epoch} [CF Phase]', leave=False)
        for _ in cf_pbar:
            batch = data.generate_cf_batch(data.train_user_dict, args.cf_batch_size)
            if batch is None: continue  # 极端空样本预防
            u, p, n = batch

            loss = model(u.to(device), p.to(device), n.to(device), aux_all_cpu, mode='train_cf')

            optimizer.zero_grad()
            # 【核心修改】：放大 CF 信号权重（如 5.0），强制模型从 0 突破到 1
            (loss * 5.0).backward()
            optimizer.step()
            loss_cf += loss.item()
            cf_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # --- 2. KG 阶段 ---
        n_kg = data.n_kg_train // args.kg_batch_size + 1
        # 削弱 KG 训练频率：如果 KG Loss 太低，可以每轮只训练部分 batch
        n_kg = n_kg // 2 + 1
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

        if epoch == 1 or epoch % 10 == 0:
            with torch.no_grad():
                model(data.h_list.to(device), data.t_list.to(device), data.r_list.to(device), mode='update_att')

        logging.info(f'Epoch {epoch} | CF Loss: {loss_cf / n_cf:.4f} | KG Loss: {loss_kg / n_kg:.4f}')

        if epoch % args.evaluate_every == 0:
            res = evaluate(model, data, Ks, device)
            curr_recall = res[Ks[0]]['recall']
            logging.info(f"Eval @{Ks[0]}: Recall: {curr_recall:.4f}, NDCG: {res[Ks[0]]['ndcg']:.4f}")
            recall_list.append(curr_recall)
            _, stop = early_stopping(recall_list, args.stopping_steps)
            if curr_recall > best_recall:
                best_recall = curr_recall
                save_model(model, weight_path, epoch)
            if stop: break


if __name__ == '__main__':
    train(parse_kgat_args())