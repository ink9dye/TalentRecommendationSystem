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
        # 增加召回上限，为后续精排留足空间
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
        :param domain_id: 输入 1-17 的字符串，若为 "0" 或 None 则视为全领域搜索
        """
        start_time = time.time()

        # --- 1. 领域过滤逻辑预处理 ---
        # 统一处理 domain_id：如果是 "0" 或 空字符串，设为 None 以触发各路径的“全召回”逻辑
        processed_domain = None
        if domain_id and str(domain_id).strip() not in ["0", ""]:
            processed_domain = str(domain_id).strip()

        # --- 2. 语义增强 (Query Expansion) ---
        final_query = query_text
        if processed_domain and not is_training:
            # 只有在指定了有效领域时，才在 Query 后面挂载语义锚点，增强向量路的准确性
            if processed_domain in self.DOMAIN_PROMPTS:
                bias = self.DOMAIN_PROMPTS[processed_domain]
                final_query = f"{query_text} | Area: {bias}"

        # 向量转换
        query_vec, _ = self.encoder.encode(final_query)
        faiss.normalize_L2(query_vec)

        # --- 3. 并行召回分发 ---
        # 将 processed_domain 传递给底层，底层 LabelRecallPath 会据此拼接 Cypher
        # 注意：这里参数名对齐为你修改后的 domain_ids
        future_v = self.executor.submit(self.v_path.recall, query_vec, target_domains=processed_domain)
        future_l = self.executor.submit(self.l_path.recall, query_vec, domain_id=processed_domain)

        v_list, v_cost = future_v.result()
        l_list, l_cost = future_l.result()

        # --- 4. 协同路扩散 (基于向量路和标签路的最优种子) ---
        seeds = list(set(v_list[:100] + l_list[:100]))
        c_list, c_cost = self.c_path.recall(seeds)

        # --- 5. RRF 结果融合 ---
        final_list, rank_map = self._fuse_results(v_list, l_list, c_list)

        return {
            "final_top_500": final_list,
            "rank_map": rank_map,
            "total_ms": (time.time() - start_time) * 1000,
            "applied_domains": processed_domain if processed_domain else "All Fields",
            "details": {"v_cost": v_cost, "l_cost": l_cost, "cost_c": c_cost}
        }

    def _fuse_results(self, v_res, l_res, c_res):
        """RRF (Reciprocal Rank Fusion) 融合"""
        rrf_k = 60
        scores = {}
        rank_map = {}

        # 赋予不同召回路径不同的基础权重
        paths = [("v", v_res, 1.2), ("l", l_res, 1.0), ("c", c_res, 0.6)]

        for p_tag, res_list, weight in paths:
            for rank, aid in enumerate(res_list):
                score = weight * (1.0 / (rrf_k + rank + 1))
                scores[aid] = scores.get(aid, 0) + score

                if aid not in rank_map:
                    rank_map[aid] = {'v': '-', 'l': '-', 'c': '-'}
                rank_map[aid][p_tag] = rank + 1

        # --- multi-route bonus：命中多条路径的作者额外加分，强化 V∩L∩C 交集 ---
        # 1. 统计每个作者被几条路径命中（1/2/3）
        path_count = {}
        for aid, ranks in rank_map.items():
            cnt = 0
            if ranks["v"] != "-":
                cnt += 1
            if ranks["l"] != "-":
                cnt += 1
            if ranks["c"] != "-":
                cnt += 1
            path_count[aid] = cnt

        # 2. 在 RRF 分基础上增加一个与 path_count 成正比的小 bonus
        alpha = 0.02  # 可按效果微调：增大则更偏好多路交集，减小则更接近原始 RRF
        final_score = {}
        for aid, base in scores.items():
            pc = path_count.get(aid, 1)
            bonus = alpha * (pc - 1)  # 1 路: +0, 2 路: +alpha, 3 路: +2alpha
            final_score[aid] = base + bonus

        sorted_candidates = sorted(final_score.items(), key=lambda x: x[1], reverse=True)
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
    print("🚀 人才推荐系统 - 生产级全量召回集成版")
    print("-" * 115)
    f_list = list(fields.items())
    for i in range(0, len(f_list), 6):
        print(" | ".join([f"{k}:{v}" for k, v in f_list[i:i + 6]]))
    print("=" * 115)

    try:
        # 1. 领域选择（一次锁定，多次查询）
        domain_choice = input("\n请选择领域编号 (1-17, 0或回车跳过): ").strip()
        current_field = fields.get(domain_choice, "全领域")
        print(f"[*] 当前系统环境：{current_field} {'(硬过滤已激活)' if domain_choice in fields else '(全领域广度搜索)'}")

        while True:
            # 2. 需求输入
            user_input = input(f"\n[{current_field}] 请输入岗位需求或技术关键词 (q退出): ").strip()
            if not user_input or user_input.lower() == 'q':
                break

            # 3. 执行系统召回
            results = system.execute(user_input, domain_id=domain_choice)
            candidates = results['final_top_500']
            rank_map = results['rank_map']

            # 4. 打印报告
            print(f"\n[召回报告] 耗时: {results['total_ms']:.2f}ms | 路径耗时: V={results['details']['v_cost']:.1f}ms, L={results['details']['l_cost']:.1f}ms, C={results['details']['cost_c']:.1f}ms")
            print("-" * 115)
            print(f"{'综合排名':<6} | {'作者 ID':<10} | {'各路名次 (V/L/C)':<15} | {'知识图谱核心作 (权重)'}")
            print("-" * 115)

            for rank, aid in enumerate(candidates[:50], 1):
                rm = rank_map[aid]
                # 格式化各路名次显示
                v_rank = f"{rm['v']}" if rm['v'] != '-' else "-"
                l_rank = f"{rm['l']}" if rm['l'] != '-' else "-"
                c_rank = f"{rm['c']}" if rm['c'] != '-' else "-"
                path_ranks = f"V:{v_rank:>3} L:{l_rank:>3} C:{c_rank:>3}"

                # 获取代表作信息
                works = system._get_author_works(aid, top_n=1)
                if works:
                    work_title = works[0]['title']
                    if len(work_title) > 60: work_title = work_title[:57] + "..."
                    info = f"《{work_title}》({works[0]['weight']:.3f})"
                else:
                    info = "暂无图谱论文数据"

                print(f"#{rank:<5} | {aid:<10} | {path_ranks:<15} | {info}")

            print("-" * 115)
            print(f"[*] 已召回 {len(candidates)} 名候选人，上方显示前 50 名综合最优解。")

    except KeyboardInterrupt:
        print("\n[!] 系统安全退出。")