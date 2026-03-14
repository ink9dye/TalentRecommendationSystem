# -*- coding: utf-8 -*-
"""
阶段 1 诊断脚本：查指定词在学术词向量空间中的 TopN 邻居，
用于判断语义漂移是否来自 embedding（如 MPC 的邻居是否大量出现 AGC、Well control 等）。

用法：在项目根目录执行
  python -m src.core.recall.diagnose_embedding_neighbors
或
  python src/core/recall/diagnose_embedding_neighbors.py
"""
from __future__ import annotations

import os
import sys

import faiss
import numpy as np
import sqlite3

# 确保项目根在 path 中
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from config import DB_PATH, VOCAB_INDEX_PATH
from src.core.recall.label_means.infra import LabelMeansInfra
from src.core.recall.input_to_vector import QueryEncoder


# 诊断用词：核心算法/概念 + 易漂移词
QUERY_TERMS_CORE = [
    "MPC",
    "iLQR",
    "RRT",
    "robot kinematics",
    "trajectory optimization",
]
QUERY_TERMS_DRIFT = [
    "Well control",
    "Automatic gain control",
    "robotic hand",
]
TOP_N = 20


def load_vocid_to_term(db_path: str) -> dict[str, str]:
    """从 vocabulary 表加载 voc_id -> term。"""
    vocid_to_term = {}
    try:
        with sqlite3.connect(db_path) as conn:
            for row in conn.execute("SELECT voc_id, term FROM vocabulary"):
                vid, term = row[0], (row[1] or "").strip()
                vocid_to_term[str(vid)] = term
    except Exception as e:
        print(f"[Warning] 加载 vocabulary 失败: {e}")
    return vocid_to_term


def run_diagnosis():
    print("=" * 80)
    print("【阶段 1 诊断】学术词向量空间 TopN 邻居")
    print("目的：判断语义漂移是否来自 embedding（MPC/iLQR 邻居是否偏到 AGC、Well control 等）")
    print("=" * 80)

    # 1. 加载与主流程一致的资源（若全量 init 失败则仅加载学术词索引，便于无 Neo4j 时诊断）
    print("\n[*] 加载编码器与学术词向量库...")
    encoder = QueryEncoder()
    vocab_index = None
    vocab_to_idx = {}
    try:
        infra = LabelMeansInfra()
        infra.init_resources()
        vocab_index = infra.vocab_index
        vocab_to_idx = infra.vocab_to_idx
    except Exception as e:
        print(f"[Warning] 全量资源初始化失败 ({e})，尝试仅加载学术词 FAISS 与映射...")
    if not vocab_index:
        try:
            vocab_index = faiss.read_index(VOCAB_INDEX_PATH)
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("SELECT voc_id FROM vocabulary ORDER BY voc_id ASC").fetchall()
                vocab_to_idx = {str(r[0]): i for i, r in enumerate(rows)}
        except Exception as e2:
            print(f"[Error] 学术词索引加载失败: {e2}，请先构建向量索引。")
            return
    if not vocab_to_idx:
        print("[Error] 未加载到 vocab_to_idx，请检查 DB 与向量索引。")
        return

    # 2. index -> voc_id（FAISS 返回的是索引下标）
    idx_to_vocid = {int(idx): vid for vid, idx in vocab_to_idx.items()}
    vocid_to_term = load_vocid_to_term(DB_PATH)
    print(f"[OK] 学术词数量: {len(vocab_to_idx)}，已加载 term 映射: {len(vocid_to_term)} 条")

    # 3. 对所有待查词做 encode + search
    all_terms = QUERY_TERMS_CORE + QUERY_TERMS_DRIFT
    results_per_query = {}

    for query in all_terms:
        vec, _ = encoder.encode(query)
        if vec is None:
            print(f"[Skip] 编码为空: {query}")
            results_per_query[query] = []
            continue
        vec = np.asarray(vec, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        faiss.normalize_L2(vec)
        k = min(TOP_N + 5, vocab_index.ntotal)  # 多取几个以防去重
        scores, labels = vocab_index.search(vec, k)
        rows = []
        seen = set()
        for (s, idx) in zip(scores[0], labels[0]):
            if idx < 0:
                continue
            vid = idx_to_vocid.get(int(idx))
            if vid is None:
                continue
            term = vocid_to_term.get(vid, "?")
            if term in seen:
                continue
            seen.add(term)
            rows.append((vid, term, float(s)))
            if len(rows) >= TOP_N:
                break
        results_per_query[query] = rows

    # 4. 打印表格
    print("\n" + "=" * 80)
    print("【核心算法/概念】在学术词向量空间中的 Top{} 邻居".format(TOP_N))
    print("=" * 80)
    for query in QUERY_TERMS_CORE:
        rows = results_per_query.get(query, [])
        print("\n--- 查询: \"{}\" ---".format(query))
        if not rows:
            print("  (无结果)")
            continue
        print("  {:>4} | {:>8} | {:>8} | {}".format("Rank", "voc_id", "cos_sim", "neighbor term"))
        for r, (vid, term, sim) in enumerate(rows, 1):
            term_show = (term[:48] + "..") if len(term) > 50 else term
            print("  {:>4} | {:>8} | {:>8.4f} | {}".format(r, vid, sim, term_show))

    print("\n" + "=" * 80)
    print("【易漂移词】在学术词向量空间中的 Top{} 邻居".format(TOP_N))
    print("=" * 80)
    for query in QUERY_TERMS_DRIFT:
        rows = results_per_query.get(query, [])
        print("\n--- 查询: \"{}\" ---".format(query))
        if not rows:
            print("  (无结果)")
            continue
        print("  {:>4} | {:>8} | {:>8} | {}".format("Rank", "voc_id", "cos_sim", "neighbor term"))
        for r, (vid, term, sim) in enumerate(rows, 1):
            term_show = (term[:48] + "..") if len(term) > 50 else term
            print("  {:>4} | {:>8} | {:>8.4f} | {}".format(r, vid, sim, term_show))

    # 5. 简要结论提示
    print("\n" + "=" * 80)
    print("【诊断结论建议】")
    print("  - 若 MPC / iLQR 的邻居里大量出现 Automatic gain control、Well control、robotic hand → 漂移很可能来自 embedding，可考虑策略 4（换模型/微调）。")
    print("  - 若 robot kinematics、trajectory optimization 的邻居偏「运动学/轨迹/控制」→ embedding 尚可，优先做阶段 2 的阈值/权重。")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnosis()
