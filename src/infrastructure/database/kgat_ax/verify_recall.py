import torch
import json
import numpy as np
import os
import random
import argparse
from tqdm import tqdm
from trainer import evaluate
from data_loader import DataLoaderKGAT
from model import KGAT
from kgat_parser.parser_kgat import parse_kgat_args
from config import KGATAX_TRAIN_DATA_DIR


class SimpleLogger:
    def info(self, msg): print(f"[INFO] {msg}")

    def error(self, msg): print(f"[ERROR] {msg}")

    def warning(self, msg): print(f"[WARN] {msg}")


def run_offline_verification():
    torch.serialization.add_safe_globals([argparse.Namespace])
    args = parse_kgat_args()

    args.data_dir = os.path.dirname(KGATAX_TRAIN_DATA_DIR)
    args.data_name = os.path.basename(KGATAX_TRAIN_DATA_DIR)

    device = torch.device("cpu")  # 评估建议使用 CPU 或大显存 GPU
    weight_path = os.path.join(KGATAX_TRAIN_DATA_DIR, 'weights', 'latest_checkpoint.pth')

    if not os.path.exists(weight_path):
        print(f"[Error] 找不到权重文件: {weight_path}")
        return

    dataloader = DataLoaderKGAT(args, SimpleLogger())
    args.ENTITY_OFFSET = dataloader.ENTITY_OFFSET

    model = KGAT(
        args,
        dataloader.n_users,
        dataloader.n_users_entities,
        dataloader.n_relations,
        dataloader.A_in
    ).to(device)

    try:
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"\n[*] 成功载入 Epoch {checkpoint['epoch']} 权重")
    except Exception as e:
        print(f"[CRITICAL] 权重加载失败: {str(e)}")
        return

    print("\n" + "=" * 20 + " 深度诊断中 " + "=" * 20)

    # A. 检查 ID 空间
    test_pos_ids = set()
    for ids in dataloader.test_user_dict.values():
        test_pos_ids.update(ids)

    print(f"[*] 测试集唯一正样本 (人才) 数: {len(test_pos_ids)}")
    wrong_id_count = sum(1 for i in test_pos_ids if i < dataloader.ENTITY_OFFSET)
    if wrong_id_count > 0:
        print(f"[严重警告] 发现 {wrong_id_count} 个测试样本 ID 错误地落在了 User 空间！")
    else:
        print(f"[OK] ID 偏移校验通过。")

    # B. 【关键修改】提取严格的人才候选池
    map_path = os.path.join(KGATAX_TRAIN_DATA_DIR, "id_map.json")
    with open(map_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    entity_map = mapping['entity']
    # 仅保留以 'a_' 开头的人才 ID，剔除 v_ (技能), w_ (作品) 等
    final_author_list = [int_id for raw_id, int_id in entity_map.items() if str(raw_id).startswith('a_')]

    # 校验测试集正样本是否都在这个池子里
    missing = test_pos_ids - set(final_author_list)
    if missing:
        print(f"[警告] 过滤逻辑遗漏了 {len(missing)} 个正样本，已强制补回。")
        final_author_list = list(set(final_author_list) | missing)

    print(f"[*] 最终评估池规模 (仅限人才): {len(final_author_list)} 人")
    # 注入 dataloader 供 evaluate 函数使用
    dataloader.author_ids = final_author_list

    # C. 原始得分趋势检查
    test_users = list(dataloader.test_user_dict.keys())
    if test_users:
        u_sample = random.choice(test_users)
        p_sample = dataloader.test_user_dict[u_sample][0]
        # 随机负样本也限制在人才池内
        n_sample = random.choice(final_author_list)

        with torch.no_grad():
            aux_all = dataloader.aux_info_all.to(device)
            all_embed = model.calc_cf_embeddings(aux_all)
            p_score = torch.matmul(all_embed[u_sample], all_embed[p_sample].reshape(-1, 1)).item()
            n_score = torch.matmul(all_embed[u_sample], all_embed[n_sample]).item()
            print(f"[*] 抽样得分 (User {u_sample}): 正样本 {p_score:.4f} vs 随机人才负样本 {n_score:.4f}")

    print("=" * 50 + "\n")

    # --- 5. 执行评估 ---
    print(">>> 开始执行精简池评估...")
    res_train = evaluate(model, dataloader, [20, 100], device, use_test_set=False)
    for k, v in res_train.items():
        print(f"Train @{k:3d} | Recall: {v['recall']:.4f} | NDCG: {v['ndcg']:.4f}")

    res_test = evaluate(model, dataloader, [20, 100], device, use_test_set=True)
    for k, v in res_test.items():
        print(f"Test  @{k:3d} | Recall: {v['recall']:.4f} | NDCG: {v['ndcg']:.4f}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run_offline_verification()