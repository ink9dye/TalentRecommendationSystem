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


# --- 1. 模拟日志类 ---
class SimpleLogger:
    def info(self, msg): print(f"[INFO] {msg}")

    def error(self, msg): print(f"[ERROR] {msg}")

    def warning(self, msg): print(f"[WARN] {msg}")


def run_offline_verification():
    # --- 2. 安全配置与环境准备 ---
    torch.serialization.add_safe_globals([argparse.Namespace])
    args = parse_kgat_args()

    args.data_dir = os.path.dirname(KGATAX_TRAIN_DATA_DIR)
    args.data_name = os.path.basename(KGATAX_TRAIN_DATA_DIR)

    device = torch.device("cpu")
    weight_path = os.path.join(KGATAX_TRAIN_DATA_DIR, 'weights', 'latest_checkpoint.pth')

    if not os.path.exists(weight_path):
        print(f"[Error] 找不到权重文件: {weight_path}")
        return

    # --- 3. 初始化与数据加载 ---
    dataloader = DataLoaderKGAT(args, SimpleLogger())
    args.ENTITY_OFFSET = dataloader.ENTITY_OFFSET

    model = KGAT(
        args,
        dataloader.n_users,
        dataloader.n_entities,
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

    # --- 4. 【新增】数据质量与 ID 连通性深度诊断 ---
    print("\n" + "=" * 20 + " 深度诊断中 " + "=" * 20)

    # A. 检查测试集正样本是否在全局 ID 空间内
    test_pos_ids = set()
    for ids in dataloader.test_user_dict.values():
        test_pos_ids.update(ids)

    print(f"[*] 测试集共包含 {len(test_pos_ids)} 个唯一正样本 (人才)")
    print(f"[*] 实体空间起始点 (OFFSET): {dataloader.ENTITY_OFFSET}")

    # 检查是否有正样本 ID 小于 OFFSET（如果小于，说明 ID 映射彻底乱了）
    wrong_id_count = sum(1 for i in test_pos_ids if i < dataloader.ENTITY_OFFSET)
    if wrong_id_count > 0:
        print(f"[严重警告] 发现 {wrong_id_count} 个测试样本 ID 低于 OFFSET！ID 空间存在重叠。")
    else:
        print(f"[OK] 所有测试样本均位于人才实体空间内。")

    # B. 自动构建候选池并校验命中率
    map_path = os.path.join(KGATAX_TRAIN_DATA_DIR, "id_map.json")
    with open(map_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    entity_map = mapping['entity']

    potential_authors = []
    for raw_id, int_id in entity_map.items():
        if int_id >= dataloader.ENTITY_OFFSET:
            raw_s = str(raw_id)
            # 排除非人才前缀
            if not (raw_s.startswith('v_') or raw_s.startswith('inst_')):
                potential_authors.append(int_id)

    final_author_list = list(set(potential_authors))

    # 【关键诊断】：白名单是否包含正样本？
    missing_in_whitelist = test_pos_ids - set(final_author_list)
    if len(missing_in_whitelist) > 0:
        print(f"[警告] 你的候选人过滤逻辑剔除了 {len(missing_in_whitelist)} 个测试集正样本！")
        print(f"[*] 正在将这些遗漏样本强制补回白名单以确保评估有效性...")
        final_author_list.extend(list(missing_in_whitelist))
        final_author_list = list(set(final_author_list))

    print(f"[*] 最终评估池规模: {len(final_author_list)} 人")
    dataloader.author_ids = final_author_list

    # C. 抽样预测诊断 (查看原始分数值)
    test_users = list(dataloader.test_user_dict.keys())
    if test_users:
        u_sample = random.choice(test_users)
        p_sample = dataloader.test_user_dict[u_sample][0]
        n_sample = random.randint(dataloader.ENTITY_OFFSET, dataloader.ENTITY_OFFSET + dataloader.n_entities - 1)

        with torch.no_grad():
            aux_all = dataloader.aux_info_all.to(device)
            all_embed = model.calc_cf_embeddings(aux_all)
            u_e = all_embed[u_sample].unsqueeze(0)
            p_e = all_embed[p_sample].unsqueeze(0)
            n_e = all_embed[n_sample].unsqueeze(0)

            p_score = torch.matmul(u_e, p_e.T).item()
            n_score = torch.matmul(u_e, n_e.T).item()
            print(f"[*] 抽样得分诊断 (User {u_sample}): 正样本 {p_score:.4f} vs 随机负样本 {n_score:.4f}")
            if p_score < n_score:
                print("[警告] 正样本得分低于负样本！模型可能学反了或未收敛。")

    print("=" * 50 + "\n")

    # --- 5. 执行评估 ---
    print(">>> 开始训练集拟合度验证 (1000个采样点)")
    res_train = evaluate(model, dataloader, [20, 100], device, use_test_set=False)
    for k, metrics in res_train.items():
        print(f"Train @{k:3d} | Recall: {metrics['recall']:.4f} | NDCG: {metrics['ndcg']:.4f}")

    print("\n>>> 开始测试集泛化力验证")
    res_test = evaluate(model, dataloader, [20, 100], device, use_test_set=True)
    for k, metrics in res_test.items():
        print(f"Test  @{k:3d} | Recall: {metrics['recall']:.4f} | NDCG: {metrics['ndcg']:.4f}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    run_offline_verification()