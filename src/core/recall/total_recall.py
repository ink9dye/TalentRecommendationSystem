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
    # --- 全 17 领域精简语义锚点 (用于引导向量偏移，避免喧宾夺主) ---
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

    def _get_author_works(self, author_id, top_n=3):
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
        :param query_text: 原始业务需求描述
        :param domain_id: 领域编号
        :param is_training: 是否为训练模式。为 True 时禁用所有 Prompt 增强，保证模型训练纯度
        """
        start_time = time.time()

        # --- Query 处理策略：平衡原始意图与领域偏置 ---
        final_query = query_text
        if domain_id and not is_training and domain_id in self.DOMAIN_PROMPTS:
            bias = self.DOMAIN_PROMPTS[domain_id]
            # 强化策略：重复原始 Query 增加 Attention 权重，Area 词置于末尾作为滤镜
            final_query = f"{query_text}, {query_text} | Area: {bias}"
            # print(f"[*] 引导式检索已激活 (非训练模式)")

        # 向量编码
        query_vec, encode_duration = self.encoder.encode(final_query)
        faiss.normalize_L2(query_vec)

        # 任务并行化
        future_v = self.executor.submit(self.v_path.recall, query_vec)
        future_l = self.executor.submit(self.l_path.recall, query_vec)

        v_list, v_cost = future_v.result()
        l_list, l_cost = future_l.result()

        # 社交路协同扩展
        seeds = list(set(v_list[:80] + l_list[:80]))
        c_list, c_cost = self.c_path.recall(seeds)

        # RRF 结果融合
        final_list, _ = self._fuse_results(v_list, l_list, c_list)

        return {
            "final_top_500": final_list,
            "total_ms": (time.time() - start_time) * 1000,
            "details": {"v_cost": v_cost, "l_cost": l_cost, "c_cost": c_cost}
        }

    def _fuse_results(self, v_res, l_res, c_res):
        """采用 Reciprocal Rank Fusion 进行多路融合"""
        rrf_k = 60
        scores = {}
        # 向量路与标签路各占 1.0 权重，协同路作为社交补充占 0.5
        weights = {"vector": 1.0, "label": 1.0, "collab": 0.5}
        for path, res_list in [("vector", v_res), ("label", l_res), ("collab", c_res)]:
            w = weights[path]
            for rank, aid in enumerate(res_list):
                score = w * (1.0 / (rrf_k + rank + 1))
                scores[aid] = scores.get(aid, 0) + score
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_candidates[:500]], scores


if __name__ == "__main__":
    system = TotalRecallSystem()
    fields = {
        "1": "计算机科学", "2": "医学", "3": "政治学", "4": "工程学", "5": "物理学",
        "6": "材料科学", "7": "生物学", "8": "地理学", "9": "化学", "10": "商学",
        "11": "社会学", "12": "哲学", "13": "环境科学", "14": "数学", "15": "心理学",
        "16": "地质学", "17": "经济学"
    }

    print("\n" + "=" * 85)
    print("🚀 人才推荐系统 - 全量召回测试 (支持 17 领域解耦检索)")
    print("-" * 85)
    f_list = list(fields.items())
    for i in range(0, len(f_list), 6):
        print(" | ".join([f"{k}:{v}" for k, v in f_list[i:i + 6]]))
    print("=" * 85)

    try:
        # 交互层：选择测试领域
        domain_choice = input("\n请选择领域编号 (直接回车则搜索【默认全领域】): ").strip()
        current_field = fields.get(domain_choice, "全领域")
        print(f"[*] 当前锁定搜索上下文：{current_field}")

        while True:
            user_input = input(f"\n[{current_field}] 请输入业务需求关键词 (输入 'q' 退出): \n>> ").strip()
            if not user_input or user_input.lower() == 'q': break

            # 执行召回
            results = system.execute(user_input, domain_id=domain_choice, is_training=False)
            candidates = results['final_top_500']

            print(f"\n[任务概览] 总响应时间: {results['total_ms']:.2f} ms")
            print("-" * 110)
            print(f"{'排名':<4} | {'作者 ID':<12} | {'图谱代表作 (pos_weight)'}")
            print("-" * 110)

            # 打印 Top 20 及其高权重论文
            for rank, aid in enumerate(candidates[:20], 1):
                works = system._get_author_works(aid, top_n=2)
                if works:
                    # 显示标题和计算出的贡献权重
                    info = " / ".join([f"《{w['title'][:50]}...》({w['weight']:.3f})" for w in works])
                else:
                    info = "⚠️ 图谱无关联数据"
                print(f"{rank:<4} | {aid:<12} | {info}")

            print("-" * 110)

    except KeyboardInterrupt:
        print("\n系统已安全退出")