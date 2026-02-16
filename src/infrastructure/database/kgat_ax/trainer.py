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
    """环境与参数合规性预检"""
    print("\n[System Check] 正在执行环境预检...")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"错误: 数据库文件不存在于 {DB_PATH}")
    if hasattr(args, 'embed_dim') and args.embed_dim > 128:
        print("警告: Embedding 维度较高，请监控内存利用率。")
    try:
        ks = eval(args.Ks)
        if not isinstance(ks, list): raise ValueError
    except:
        raise ValueError("错误: 参数 Ks 必须是列表格式，例如 [20, 50, 100]。")
    print("[System Check] 预检通过！\n")


def evaluate(model, dataloader, Ks, device, use_test_set=False):
    """
    针对 500 人精排场景优化的评估函数。
    支持拟合度自检（训练集）与泛化力评估（测试集）。
    """
    import gc
    model.eval()

    # 切换评估目标
    if use_test_set:
        eval_user_dict = dataloader.test_user_dict
        mode_str = "Test Set (Generalization Capability)"
    else:
        eval_user_dict = dataloader.train_user_dict
        mode_str = "Train Set (Fitting/Rule Reproduction)"

    user_ids = list(eval_user_dict.keys())
    # 抽样评估，防止全量 185w 节点检索导致内存溢出
    if len(user_ids) > 1000:
        user_ids = random.sample(user_ids, 1000)

    # 核心：获取注入了 AX 特征的全量 Embedding
    aux_info_all = dataloader.aux_info_all.to(device)
    with torch.no_grad():
        all_embed = model.calc_cf_embeddings(aux_info_all)

    offset = dataloader.ENTITY_OFFSET
    global_max_id = dataloader.ENTITY_OFFSET + dataloader.n_entities
    item_embeds = all_embed[offset:global_max_id]
    item_ids_vec = np.arange(offset, global_max_id)

    batch_size = dataloader.test_batch_size
    user_batches = [user_ids[i:i + batch_size] for i in range(0, len(user_ids), batch_size)]
    metrics = {k: {'recall': [], 'ndcg': []} for k in Ks}

    print(f"\n[*] 启动评估: {mode_str}")
    for batch_u in tqdm(user_batches, desc='Evaluating', leave=False):
        batch_u_tensor = torch.LongTensor(batch_u).to(device)
        with torch.no_grad():
            curr_user_embeds = all_embed[batch_u_tensor]
            # 计算相似度矩阵
            all_scores = torch.matmul(curr_user_embeds, item_embeds.transpose(0, 1))

            # 模拟精排池：仅保留 Top 500 的分数进入指标计算
            top_k_val = 500
            top_scores, top_indices = torch.topk(all_scores, k=min(top_k_val, item_embeds.shape[0]), dim=1)
            reduced_scores = torch.full_like(all_scores, -1e10)
            reduced_scores.scatter_(1, top_indices, top_scores)
            final_scores_cpu = reduced_scores.cpu().numpy()

        # 计算 Recall 和 NDCG
        batch_m = calc_metrics_at_k(
            torch.from_numpy(final_scores_cpu),
            dataloader.train_user_dict,
            eval_user_dict,
            np.array(batch_u),
            item_ids_vec,
            Ks
        )

        for k in Ks:
            metrics[k]['recall'].append(batch_m[k]['recall'])
            metrics[k]['ndcg'].append(batch_m[k]['ndcg'])

        del all_scores, reduced_scores, final_scores_cpu
        gc.collect()

    del all_embed, item_embeds
    gc.collect()
    return {k: {m: np.mean(v) for m, v in val.items()} for k, val in metrics.items()}


def save_checkpoint(model, optimizer, epoch, best_recall, args, path, is_best=False):
    """持久化 Checkpoint 机制，支持断点续传"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_recall': best_recall,
        'args': args
    }
    if is_best:
        file_path = os.path.join(path, f"best_model_epoch_{epoch}.pth")
    else:
        file_path = os.path.join(path, "latest_checkpoint.pth")

    torch.save(checkpoint, file_path)
    print(f"[*] {'[最佳]' if is_best else '[同步]'} 模型已保存: {file_path}")


def train(args):
    # 配置初始化
    args.save_dir = KGATAX_TRAIN_DATA_DIR
    args.data_dir = os.path.dirname(KGATAX_TRAIN_DATA_DIR)
    args.data_name = os.path.basename(KGATAX_TRAIN_DATA_DIR)
    check_args_and_env(args)
    logging_config(folder=args.save_dir, name='log_kgatax', no_console=False)

    device = torch.device("cpu")  # 建议在 CPU 上调试，32GB RAM 可支持小型 Batch
    data = DataLoaderKGAT(args, logging)
    args.ENTITY_OFFSET = data.ENTITY_OFFSET

    # 模型与优化器初始化
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations, data.A_in).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    best_recall = 0
    recall_list = []
    weight_path = os.path.join(args.save_dir, 'weights')
    if not os.path.exists(weight_path): os.makedirs(weight_path)

    # 自动断点恢复逻辑
    latest_models = sorted([f for f in os.listdir(weight_path) if f.endswith('.pth')],
                           key=lambda x: os.path.getmtime(os.path.join(weight_path, x)))
    if latest_models:
        last_model = os.path.join(weight_path, latest_models[-1])
        try:
            checkpoint = torch.load(last_model, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_recall = checkpoint.get('best_recall', 0)
            logging.info(f"成功载入历史权重，接续 Epoch {start_epoch}, 历史最佳 Recall: {best_recall:.4f}")
        except Exception as e:
            logging.warning(f"权重加载异常，将冷启动。原因: {e}")

    Ks = eval(args.Ks)

    for epoch in range(start_epoch, args.n_epoch + 1):
        model.train()
        # 预加载全量学术特征到计算设备
        aux_all_device = data.aux_info_all.to(device)

        # --- 1. CF 阶段：学习阶梯排名规律 ---
        n_cf = data.n_cf_train // args.cf_batch_size + 1
        loss_cf = 0
        cf_pbar = tqdm(range(n_cf), desc=f'Epoch {epoch} [CF Phase]', leave=False)
        for _ in cf_pbar:
            # 内部执行分层负采样：Pos > Fair > Neutral > EasyNeg
            batch = data.generate_cf_batch(data.train_user_dict, args.cf_batch_size)
            if batch is None: continue
            u, p, n = batch

            # 计算带 AX 注入的 BPR Loss
            loss = model.calc_cf_loss(u.to(device), p.to(device), n.to(device), aux_all_device)

            optimizer.zero_grad()
            # 增加精排阶段的梯度敏感度
            (loss * 2.0).backward()
            optimizer.step()
            loss_cf += loss.item()
            cf_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # --- 2. KG 阶段：学习人才关系网（协作/领域） ---
        n_kg = (data.n_cf_train // args.kg_batch_size) + 1
        loss_kg = 0
        kg_pbar = tqdm(range(n_kg), desc=f'Epoch {epoch} [KG Phase]', leave=False)
        for _ in kg_pbar:
            h, r, pt, nt, ha, pa, na = [d.to(device) for d in
                                        data.generate_kg_batch(data.train_kg_dict, args.kg_batch_size,
                                                               data.n_users_entities)]
            # 计算 TransR 图谱损失
            loss = model.calc_kg_loss(h, r, pt, nt, ha, pa, na)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_kg += loss.item()
            kg_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 定期更新图注意力权重
        if epoch == 1 or epoch % 10 == 0:
            with torch.no_grad():
                model.update_attention_batch(data.h_list.to(device), data.t_list.to(device), data.r_list.to(device))

        logging.info(f'Epoch {epoch} | CF Loss: {loss_cf / n_cf:.4f} | KG Loss: {loss_kg / n_kg:.4f}')

        # 核心：评估前先保存，防御内存崩溃导致的数据丢失
        save_checkpoint(model, optimizer, epoch, best_recall, args, weight_path, is_best=False)

        # --- 3. 评估阶段 ---
        if epoch % args.evaluate_every == 0:
            # 默认进行训练集拟合度校验
            res = evaluate(model, data, Ks, device, use_test_set=False)
            curr_recall = res[Ks[0]]['recall']
            logging.info(f"Eval (Fitting) @{Ks[0]}: Recall: {curr_recall:.4f}, NDCG: {res[Ks[0]]['ndcg']:.4f}")

            # 选做：进行测试集泛化验证
            res_test = evaluate(model, data, Ks, device, use_test_set=True)
            logging.info(f"Eval (Generalization) @{Ks[0]}: Recall: {res_test[Ks[0]]['recall']:.4f}")

            recall_list.append(curr_recall)
            _, stop = early_stopping(recall_list, args.stopping_steps)

            if curr_recall > best_recall:
                best_recall = curr_recall
                save_checkpoint(model, optimizer, epoch, best_recall, args, weight_path, is_best=True)

            if stop:
                logging.info("达到早停条件，训练结束。")
                break


if __name__ == '__main__':
    train(parse_kgat_args())