import torch
import json
import numpy as np
import os
import random
import argparse
import sqlite3
from tqdm import tqdm
from trainer import evaluate
from data_loader import DataLoaderKGAT
from model import KGAT
from kgat_parser.parser_kgat import parse_kgat_args
from config import KGATAX_TRAIN_DATA_DIR, DB_PATH
from src.core.recall.total_recall import TotalRecallSystem  #


class SimpleLogger:
    def info(self, msg): print(f"[INFO] {msg}")

    def error(self, msg): print(f"[ERROR] {msg}")

    def warning(self, msg): print(f"[WARN] {msg}")


def run_offline_verification():
    torch.serialization.add_safe_globals([argparse.Namespace])
    args = parse_kgat_args()

    args.data_dir = os.path.dirname(KGATAX_TRAIN_DATA_DIR)
    args.data_name = os.path.basename(KGATAX_TRAIN_DATA_DIR)

    device = torch.device("cpu")  # 评估建议使用 CPU
    weight_path = os.path.join(KGATAX_TRAIN_DATA_DIR, 'weights', 'latest_checkpoint.pth')

    if not os.path.exists(weight_path):
        print(f"[Error] 找不到权重文件: {weight_path}")
        return

    # 1. 载入数据与模型
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

    # A. 检查 ID 空间 (确保正样本全部落在人才 a_ 空间)
    test_pos_ids = set()
    for ids in dataloader.test_user_dict.values():
        test_pos_ids.update(ids)

    print(f"[*] 测试集唯一正样本 (人才) 数: {len(test_pos_ids)}")
    wrong_id_count = sum(1 for i in test_pos_ids if i < dataloader.ENTITY_OFFSET)
    if wrong_id_count > 0:
        print(f"[严重警告] 发现 {wrong_id_count} 个测试样本 ID 错误地落在了 User 空间！")
    else:
        print(f"[OK] ID 偏移校验通过。")

    # B. 提取严格的人才候选池 (用于模拟全库排序)
    map_path = os.path.join(KGATAX_TRAIN_DATA_DIR, "id_map.json")
    with open(map_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    entity_map = mapping['entity']
    final_author_list = [int_id for raw_id, int_id in entity_map.items() if str(raw_id).startswith('a_')]
    dataloader.author_ids = final_author_list
    print(f"[*] 全球人才总池规模: {len(final_author_list)} 人")

    # --- C. 【核心新增】75/25 混合模式分拆评估 ---
    # 目的：验证模型在“领域硬过滤”与“全库搜索”两种策略下的排序表现
    print("\n" + "=" * 15 + " 75/25 策略分拆评估 " + "=" * 15)

    # 初始化召回系统以模拟实时检索
    recall_system = TotalRecallSystem()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    test_users = list(dataloader.test_user_dict.keys())
    sample_users = random.sample(test_users, min(len(test_users), 200))  # 抽样评估

    domain_aware_metrics = {'recall': [], 'ndcg': []}
    general_metrics = {'recall': [], 'ndcg': []}

    for u_int in tqdm(sample_users, desc="Split Mode Testing"):
        # 反查 Job 的原始描述与 Domain IDs
        raw_job_id = [k for k, v in dataloader.user_to_int.items() if v == u_int][0]
        job = conn.execute("SELECT job_name, description, skills, domain_ids FROM jobs WHERE securityId=?",
                           (raw_job_id,)).fetchone()
        if not job: continue

        query_text = f"{job['job_name']} {job['description']} {job['skills']}"
        pos_ids = set(dataloader.test_user_dict[u_int])

        # 模拟 75% 场景：带 Domain 过滤（批评估与训练数据一致：is_training=True，跳过领域 prompt 二次编码）
        res_d = recall_system.execute(query_text, domain_id=job["domain_ids"], is_training=True)
        cand_d = [dataloader.entity_to_int.get(f"a_{aid}") for aid in res_d['final_top_500'] if
                  f"a_{aid}" in dataloader.entity_to_int]

        # 模拟 25% 场景：全库搜索 (domain_id=None)；domain 策略不同须单独召回，无法与 res_d 共用
        res_g = recall_system.execute(query_text, domain_id=None, is_training=True)
        cand_g = [dataloader.entity_to_int.get(f"a_{aid}") for aid in res_g['final_top_500'] if
                  f"a_{aid}" in dataloader.entity_to_int]

        # 评分对比逻辑 (利用本地简易 Rank 逻辑)
        def score_and_eval(candidates):
            if not candidates: return 0, 0
            # 这里调用 model 内部的 calc_score
            with torch.no_grad():
                u_e = model.calc_cf_embeddings(dataloader.aux_info_all.to(device))[u_int].unsqueeze(0)
                i_e = model.calc_cf_embeddings(dataloader.aux_info_all.to(device))[candidates]
                scores = torch.matmul(u_e, i_e.transpose(0, 1)).squeeze(0)
                top_idx = torch.argsort(scores, descending=True)[:100].cpu().numpy()
                ranked_ids = [candidates[i] for i in top_idx]

                # 计算命中率
                hits = [1 if rid in pos_ids else 0 for rid in ranked_ids]
                recall = sum(hits) / len(pos_ids) if pos_ids else 0
                return recall, hits  # 简化版指标

        rec_d, _ = score_and_eval(cand_d)
        rec_g, _ = score_and_eval(cand_g)

        domain_aware_metrics['recall'].append(rec_d)
        general_metrics['recall'].append(rec_g)

    conn.close()

    print(f"\n[策略对比报告]")
    print(f"- 精准模式 (Domain-Aware 75%): 平均 Recall@100: {np.mean(domain_aware_metrics['recall']):.4f}")
    print(f"- 通用模式 (General Search 25%): 平均 Recall@100: {np.mean(general_metrics['recall']):.4f}")
    print("=" * 50 + "\n")

    # --- 2. 标准静态评估 (基于生成的 test.txt) ---
    print(">>> 开始执行 test.txt 混合池评估...")
    res_test = evaluate(model, dataloader, [20, 100], device, use_test_set=True)
    for k, v in res_test.items():
        print(f"Test  @{k:3d} | Recall: {v['recall']:.4f} | NDCG: {v['ndcg']:.4f}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run_offline_verification()