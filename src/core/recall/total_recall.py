import time
import json
import numpy as np
import faiss
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from src.core.recall.input_to_vector import QueryEncoder
from src.core.recall.vector_path import VectorPath
from src.core.recall.label_path import LabelRecallPath
from src.core.recall.collaboration_path import CollaborativeRecallPath
from config import DB_PATH

# 限制底层模型并行，防止多线程冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class TotalRecallSystem:
    # --- 全 17 领域精简语义锚点 ---
    DOMAIN_PROMPTS = {
        "1": "CS, IT, Software", "2": "Medicine, Biology", "3": "Politics, Law",
        "4": "Engineering, Manufacturing", "5": "Physics, Energy", "6": "Material Science",
        "7": "Biology, Life", "8": "Geography, Earth", "9": "Chemistry",
        "10": "Business, Management", "11": "Sociology", "12": "Philosophy",
        "13": "Environment", "14": "Mathematics", "15": "Psychology",
        "16": "Geology", "17": "Economics"
    }

    def __init__(self):
        print("[*] 正在初始化全量召回系统 (Training-Safe Mode)...", flush=True)
        self.encoder = QueryEncoder()
        self.v_path = VectorPath(recall_limit=500)
        self.l_path = LabelRecallPath(recall_limit=500)
        self.c_path = CollaborativeRecallPath(recall_limit=500)
        self.executor = ThreadPoolExecutor(max_workers=3)

    def _get_author_works(self, author_id, top_n=2):
        """利用知识图谱获取作者贡献度最高的代表作"""
        try:
            graph = self.l_path.graph
            cypher = """
            MATCH (a:Author {id: $aid})-[r:AUTHORED]->(w:Work)
            RETURN w.title AS title, w.year AS year, r.pos_weight AS weight
            ORDER BY r.pos_weight DESC LIMIT $limit
            """
            return graph.run(cypher, aid=author_id, limit=top_n).data()
        except Exception:
            return []

    def execute(self, query_text, domain_id=None, is_training=False):
        """
        执行多路召回
        :param domain_id: 为 None 时，内部路径将跳过所有硬过滤逻辑
        """
        start_time = time.time()

        # --- 1. 领域过滤项处理 ---
        target_domains = []
        if domain_id and str(domain_id).strip():
            if isinstance(domain_id, list):
                target_domains = [str(d) for d in domain_id]
            elif isinstance(domain_id, str):
                target_domains = [d.strip() for d in domain_id.split('|') if d.strip()]
            else:
                target_domains = [str(domain_id)]

        # --- 2. Query 处理策略 ---
        final_query = query_text
        if target_domains and not is_training:
            primary_d = target_domains[0]
            if primary_d in self.DOMAIN_PROMPTS:
                bias = self.DOMAIN_PROMPTS[primary_d]
                final_query = f"{query_text} | Area: {bias}"

        query_vec, _ = self.encoder.encode(final_query)
        faiss.normalize_L2(query_vec)

        # --- 3. 并行分发 ---
        # 注意：如果 target_domains 为空，路径内部应识别并跳过 SQL/Cypher 的 WHERE 过滤子句
        future_v = self.executor.submit(self.v_path.recall, query_vec, target_domains=target_domains)
        future_l = self.executor.submit(self.l_path.recall, query_vec, target_domains=target_domains)

        v_list, v_cost = future_v.result()
        l_list, l_cost = future_l.result()

        # 协同路基于前两路 Top100 种子进行扩散
        seeds = list(set(v_list[:100] + l_list[:100]))
        c_list, c_cost = self.c_path.recall(seeds)

        # --- 4. 融合并保留名次信息 ---
        final_list, rank_map = self._fuse_results(v_list, l_list, c_list)

        return {
            "final_top_500": final_list,
            "rank_map": rank_map,  # 包含每路名次的字典
            "total_ms": (time.time() - start_time) * 1000,
            "applied_domains": target_domains,
            "details": {"v_cost": v_cost, "l_cost": l_cost, "cost_c": c_cost}
        }

    def _fuse_results(self, v_res, l_res, c_res):
        """RRF 融合，并记录原始路径名次"""
        rrf_k = 60
        scores = {}
        # 记录每个作者在各路的名次：{aid: {'v': rank, 'l': rank, 'c': rank}}
        rank_map = {}

        paths = [("v", v_res, 5), ("l", l_res, 4), ("c", c_res, 3)]

        for p_tag, res_list, weight in paths:
            for rank, aid in enumerate(res_list):
                # 累加 RRF 分数
                score = weight * (1.0 / (rrf_k + rank + 1))
                scores[aid] = scores.get(aid, 0) + score

                # 记录名次（1-based）
                if aid not in rank_map:
                    rank_map[aid] = {'v': '-', 'l': '-', 'c': '-'}
                rank_map[aid][p_tag] = rank + 1

        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_candidates[:500]], rank_map


if __name__ == "__main__":
    system = TotalRecallSystem()
    fields = {
        "1": "计算机科学", "2": "医学", "3": "政治学", "4": "工程学", "5": "物理学",
        "6": "材料科学", "7": "生物学", "8": "地理学", "9": "化学", "10": "商学",
        "11": "社会学", "12": "哲学", "13": "环境科学", "14": "数学", "15": "心理学",
        "16": "地质学", "17": "经济学"
    }

    print("\n" + "=" * 115)
    print("🚀 人才推荐系统 - 全量召回测试 (领域硬过滤开关模式)")
    print("-" * 115)
    f_list = list(fields.items())
    for i in range(0, len(f_list), 6):
        print(" | ".join([f"{k}:{v}" for k, v in f_list[i:i + 6]]))
    print("=" * 115)

    try:
        domain_choice = input("\n请选择领域编号 (直接回车则搜索【默认全领域】): ").strip()
        current_field = fields.get(domain_choice, "全领域")
        print(
            f"[*] 当前锁定上下文：{current_field} {'(已禁用领域硬过滤)' if current_field == '全领域' else '(已激活硬过滤)'}")

        while True:
            user_input = input(f"\n[{current_field}] 输入关键词 (q退出): ").strip()
            if not user_input or user_input.lower() == 'q': break

            results = system.execute(user_input, domain_id=domain_choice)
            candidates = results['final_top_500']
            rank_map = results['rank_map']

            print(
                f"\n[任务概览] 总耗时: {results['total_ms']:.2f} ms | 向量路: {results['details']['v_cost']:.1f}ms | 标签路: {results['details']['l_cost']:.1f}ms")
            print("-" * 115)
            # 增加各路名次打印 (V:向量路, L:标签路, C:协同路)
            print(f"{'综合排名':<6} | {'作者 ID':<10} | {'各路名次 (V/L/C)':<15} | {'图谱代表作 (权重)'}")
            print("-" * 115)

            for rank, aid in enumerate(candidates[:20], 1):
                rm = rank_map[aid]
                path_ranks = f"V:{rm['v']:>3} L:{rm['l']:>3} C:{rm['c']:>3}"

                works = system._get_author_works(aid, top_n=1)
                info = f"《{works[0]['title'][:50]}...》({works[0]['weight']:.3f})" if works else "无数据"

                print(f"#{rank:<5} | {aid:<10} | {path_ranks:<15} | {info}")

            print("-" * 115)

    except KeyboardInterrupt:
        print("\n系统已安全退出")