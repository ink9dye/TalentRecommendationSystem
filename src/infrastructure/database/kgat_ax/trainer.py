import sys
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import logging

from model import KGAT
from data_loader import DataLoaderKGAT
from kgat_utils.metrics import calc_metrics_at_k
from kgat_utils.log_helper import create_log_id, logging_config
from kgat_utils.model_helper import save_model, early_stopping
from kgat_parser.parser_kgat import parse_kgat_args
from config import KGATAX_TRAIN_DATA_DIR, DB_PATH


def evaluate(model, dataloader, Ks, device):
    model.eval()
    test_user_dict = dataloader.test_user_dict
    user_ids = list(test_user_dict.keys())
    # 缩小评估时的 Batch 以节省内存
    user_batches = [torch.LongTensor(user_ids[i:i + dataloader.test_batch_size]) for i in
                    range(0, len(user_ids), dataloader.test_batch_size)]
    item_ids = torch.arange(dataloader.n_users_entities, dtype=torch.long).to(device)
    aux_info_all = dataloader.aux_info_all.to(device)

    metrics = {k: {'recall': [], 'ndcg': []} for k in Ks}
    for batch_u in tqdm(user_batches, desc='Eval', leave=False):
        with torch.no_grad():
            scores = model(batch_u.to(device), item_ids, aux_info_all, mode='predict').cpu()
        batch_m = calc_metrics_at_k(scores, dataloader.train_user_dict, test_user_dict, batch_u.numpy(),
                                    item_ids.cpu().numpy(), Ks)
        for k in Ks:
            metrics[k]['recall'].append(batch_m[k]['recall'])
            metrics[k]['ndcg'].append(batch_m[k]['ndcg'])

    return {k: {m: np.mean(v) for m, v in val.items()} for k, val in metrics.items()}


def train(args):
    args.save_dir, args.data_dir = KGATAX_TRAIN_DATA_DIR, os.path.dirname(KGATAX_TRAIN_DATA_DIR)
    args.data_name = os.path.basename(KGATAX_TRAIN_DATA_DIR)
    if not hasattr(args, 'db_path'): args.db_path = DB_PATH

    logging_config(folder=args.save_dir, name='log_fast', no_console=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = DataLoaderKGAT(args, logging)
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations, data.A_in).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_recall, recall_list = 0, []
    Ks = eval(args.Ks)

    for epoch in range(1, args.n_epoch + 1):
        model.train()
        aux_all = data.aux_info_all.to(device)

        # --- CF Training ---
        n_cf = data.n_cf_train // args.cf_batch_size + 1
        loss_cf = 0
        for _ in tqdm(range(n_cf), desc=f'Ep{epoch} CF', leave=False):
            u, p, n = data.generate_cf_batch(data.train_user_dict, args.cf_batch_size)
            loss = model(u.to(device), p.to(device), n.to(device), aux_all, mode='train_cf')
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()
            loss_cf += loss.item()

        # --- KG Training ---
        n_kg = data.n_kg_train // args.kg_batch_size + 1
        loss_kg = 0
        for _ in tqdm(range(n_kg), desc=f'Ep{epoch} KG', leave=False):
            h, r, pt, nt, ha, pa, na = [d.to(device) for d in
                                        data.generate_kg_batch(data.train_kg_dict, args.kg_batch_size,
                                                               data.n_users_entities)]
            loss = model(h, r, pt, nt, ha, pa, na, mode='train_kg')
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()
            loss_kg += loss.item()

        # --- 延迟注意力更新：显著提升速度 ---
        if epoch % 10 == 0:
            with torch.no_grad():
                model(data.h_list.to(device), data.t_list.to(device), data.r_list.to(device), mode='update_att')

        logging.info(f'Epoch {epoch} | CF Loss: {loss_cf / n_cf:.4f} | KG Loss: {loss_kg / n_kg:.4f}')

        if epoch % args.evaluate_every == 0:
            res = evaluate(model, data, Ks, device)
            logging.info(f"Eval @{Ks[0]}: Recall: {res[Ks[0]]['recall']:.4f}, NDCG: {res[Ks[0]]['ndcg']:.4f}")
            best_recall_tmp, stop = early_stopping(recall_list, args.stopping_steps)
            if res[Ks[0]]['recall'] > best_recall:
                best_recall = res[Ks[0]]['recall']
                save_model(model, os.path.join(args.save_dir, 'weights'), epoch)
            if stop: break


if __name__ == '__main__':
    train(parse_kgat_args())