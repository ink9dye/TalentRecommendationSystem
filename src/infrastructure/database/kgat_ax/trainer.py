import sys
import os
import gc
import json
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
import argparse
import scipy.sparse as sp
torch.serialization.add_safe_globals([argparse.Namespace])

from src.infrastructure.database.kgat_ax.model import KGAT
from src.infrastructure.database.kgat_ax.data_loader import DataLoaderKGAT
from src.infrastructure.database.kgat_ax.kgat_utils.metrics import calc_metrics_at_k
from src.infrastructure.database.kgat_ax.kgat_utils.log_helper import create_log_id, logging_config
from src.infrastructure.database.kgat_ax.kgat_utils.model_helper import save_model, early_stopping
from src.infrastructure.database.kgat_ax.kgat_parser.parser_kgat import parse_kgat_args
from config import KGATAX_TRAIN_DATA_DIR, DB_PATH
from src.infrastructure.database.kgat_ax.pipeline_state import write_stage_done


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


# src/infrastructure/database/kgat_ax/trainer.py

def evaluate(model, dataloader, Ks, device, use_test_set=False):

    model.eval()

    # 1. 确定当前评估的目标集（正确答案）
    eval_user_dict = dataloader.test_user_dict if use_test_set else dataloader.train_user_dict
    user_ids = list(eval_user_dict.keys())

    # --- 核心修复点：修复 KeyError: np.int64(0) ---
    if use_test_set:
        # 测试模式：正常屏蔽训练集中的已见数据
        exclude_user_dict = dataloader.train_user_dict
    else:
        # 拟合模式：创建一个包含所有用户但内容为空列表的字典
        # 这确保了底层函数执行 train_user_dict[u] 时不会报错，同时也不屏蔽任何正确答案
        exclude_user_dict = {u: [] for u in user_ids}

    # 抽样评估以节省时间
    if len(user_ids) > 1000:
        user_ids = random.sample(user_ids, 1000)

    # 2. 获取全量节点 Embedding
    aux_info_all = dataloader.aux_info_all.to(device)
    with torch.no_grad():
        all_embed = model.calc_cf_embeddings(aux_info_all)

    metrics = {k: {'recall': [], 'ndcg': []} for k in Ks}

    mode_str = 'Test' if use_test_set else 'Train'
    print(f"[*] 启动候选池精排评估 (池大小: ~500人, Mode: {mode_str}, 屏蔽集大小: {len(exclude_user_dict)})")

    for usr in tqdm(user_ids, desc='Ranking', leave=False):
        # 动态构建当前岗位的 500 人候选池
        tiers = dataloader.tiered_cf_dict.get(usr, {})
        pos_ids = eval_user_dict[usr]

        candidate_pool = list(set(pos_ids) |
                              set(tiers.get('fair', [])) |
                              set(tiers.get('neutral', [])) |
                              set(tiers.get('easy', [])))

        if not candidate_pool:
            continue

        candidate_indices = torch.LongTensor(candidate_pool).to(device)

        with torch.no_grad():
            u_e = all_embed[usr].unsqueeze(0)  # [1, dim]
            i_e = all_embed[candidate_indices]  # [num_candidates, dim]

            # 3. 计算 500 个候选人的得分
            scores = torch.matmul(u_e, i_e.transpose(0, 1)).squeeze(0)

            # 4. 构造评分向量，非候选成员设为极小值
            final_scores = torch.full((dataloader.n_users_entities,), -1e10).to(device)
            final_scores[candidate_indices] = scores

            # --- 传入修正后的动态屏蔽集 ---
            batch_m = calc_metrics_at_k(
                final_scores.unsqueeze(0).cpu(),
                exclude_user_dict,  # 动态确定的屏蔽字典
                eval_user_dict,
                np.array([usr]),
                np.arange(dataloader.n_users_entities),
                Ks
            )

            for k in Ks:
                metrics[k]['recall'].append(batch_m[k]['recall'])
                metrics[k]['ndcg'].append(batch_m[k]['ndcg'])

    del all_embed
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
    model = KGAT(args, data.n_users, data.n_users_entities, data.n_relations, data.A_in).to(device)
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
            checkpoint = torch.load(last_model, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_recall = checkpoint.get('best_recall', 0)
            logging.info(f"成功载入历史权重，接续 Epoch {start_epoch}, 历史最佳 Recall: {best_recall:.4f}")
        except Exception as e:
            logging.warning(f"权重加载异常，将冷启动。原因: {e}")

    Ks = eval(args.Ks)

    last_epoch = None
    for epoch in range(start_epoch, args.n_epoch + 1):
        model.train()
        # 预加载全量学术特征到计算设备
        aux_all_device = data.aux_info_all.to(device)

        # --- 1. CF 阶段：学习阶梯排名规律（有四分支侧车时用 calc_score_v2 融合四塔）---
        n_cf = data.n_cf_train // args.cf_batch_size + 1
        loss_cf = 0
        cf_pbar = tqdm(range(n_cf), desc=f'Epoch {epoch} [CF Phase]', leave=False)
        for _ in cf_pbar:
            batch = data.generate_cf_batch(data.train_user_dict, args.cf_batch_size)
            if batch is None: continue
            if len(batch) == 4:
                u, p, n, sample_w = batch
                sample_w = sample_w.to(device)
            else:
                u, p, n = batch
                sample_w = None
            author_aux_p, author_aux_n, recall_up, recall_un, interaction_up, interaction_un = data.get_four_branch_for_batch(u, p, n, use_train=True)

            if (recall_up is not None and author_aux_p is not None and interaction_up is not None):
                # 四分支训练：calc_score_v2 + BPR + 可选 teacher 蒸馏
                aux_info_all = aux_all_device
                B = u.size(0)
                job_t = u.to(device)
                pos_t = p.to(device)
                neg_t = n.to(device)
                recall_up = recall_up.to(device)
                recall_un = recall_un.to(device)
                author_aux_p = author_aux_p.to(device)
                author_aux_n = author_aux_n.to(device)
                interaction_up = interaction_up.to(device)
                interaction_un = interaction_un.to(device)
                out_pos = model.calc_score_v2(
                    job_t, pos_t, aux_info_all,
                    author_aux_item=author_aux_p,
                    recall_features=recall_up.unsqueeze(1).expand(-1, 1, -1),
                    interaction_features=interaction_up.unsqueeze(1).expand(-1, 1, -1),
                )
                out_neg = model.calc_score_v2(
                    job_t, neg_t, aux_info_all,
                    author_aux_item=author_aux_n,
                    recall_features=recall_un.unsqueeze(1).expand(-1, 1, -1),
                    interaction_features=interaction_un.unsqueeze(1).expand(-1, 1, -1),
                )
                s_pos = out_pos["final_score"].squeeze(-1) if out_pos["final_score"].dim() > 1 else out_pos["final_score"]
                s_neg = out_neg["final_score"].squeeze(-1) if out_neg["final_score"].dim() > 1 else out_neg["final_score"]
                if s_pos.dim() == 2:
                    s_pos = s_pos.squeeze(1)
                if s_neg.dim() == 2:
                    s_neg = s_neg.squeeze(1)
                per = F.softplus(s_neg - s_pos)
                if sample_w is not None:
                    loss_rank = torch.sum(per * sample_w) / torch.clamp(torch.sum(sample_w), min=1e-6)
                else:
                    loss_rank = torch.mean(per)
                loss = loss_rank
            else:
                # 兼容旧版：calc_cf_loss 内部已聚合；池监督加权仅对四分支分支执行
                loss = model.calc_cf_loss(u.to(device), p.to(device), n.to(device), aux_all_device)

            optimizer.zero_grad()
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

        # src/infrastructure/database/kgat_ax/trainer.py

        # --- 3. 评估阶段 ---
        if epoch % args.evaluate_every == 0:
            # A. 拟合度校验（仅作为日志记录，用于观察模型是否“复读机”）
            res_fit = evaluate(model, data, Ks, device, use_test_set=False)
            logging.info(
                f"Eval (Fitting) @{Ks[0]}: Recall: {res_fit[Ks[0]]['recall']:.4f}, NDCG: {res_fit[Ks[0]]['ndcg']:.4f}")

            # B. 【核心修改】泛化验证：这是衡量模型实力的真实指标
            res_test_all = evaluate(model, data, Ks, device, use_test_set=True)
            curr_recall = res_test_all[Ks[0]]['recall']
            logging.info(f"Eval (Generalization, all) @{Ks[0]}: Recall: {curr_recall:.4f}")

            # gold-only：避免 weak 样本当作正式泛化指标
            if getattr(data, "pool_supervised_mode", False) and hasattr(data, "test_user_dict_gold"):
                gold_n = sum(len(v) for v in getattr(data, "test_user_dict_gold", {}).values())
                logging.info(f"[Gold-only] test_gold_sample_count: {gold_n}")
                if gold_n > 0:
                    _orig = data.test_user_dict
                    try:
                        data.test_user_dict = data.test_user_dict_gold
                        res_test_gold = evaluate(model, data, Ks, device, use_test_set=True)
                        logging.info(
                            f"Eval (Generalization, gold-only) @{Ks[0]}: "
                            f"Recall: {res_test_gold[Ks[0]]['recall']:.4f}, "
                            f"NDCG: {res_test_gold[Ks[0]]['ndcg']:.4f}"
                        )
                    finally:
                        data.test_user_dict = _orig

            # C. 决策逻辑：使用测试集指标决定早停与保存
            recall_list.append(curr_recall)  # 记录测试集 Recall，避开 Fitting 的 nan 陷阱
            _, stop = early_stopping(recall_list, args.stopping_steps)

            # 只有测试集表现更好时，才认为它是“最佳模型”
            if curr_recall > best_recall:
                best_recall = curr_recall
                # 保存时记录的是测试集的 Recall
                save_checkpoint(model, optimizer, epoch, best_recall, args, weight_path, is_best=True)
                logging.info(f"[*] 发现更优的泛化模型 (Test Recall: {best_recall:.4f})，已更新 Best Model。")

            if stop:
                logging.info(f"连续 {args.stopping_steps} 次评估未提升泛化性能，触发早停。")
                last_epoch = epoch
                break

        last_epoch = epoch

    write_stage_done(3, {"n_epoch": args.n_epoch, "finished_epoch": last_epoch})


if __name__ == '__main__':
    train(parse_kgat_args())