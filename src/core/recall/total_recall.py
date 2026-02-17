import time
import json
import numpy as np
import faiss
import os
from concurrent.futures import ThreadPoolExecutor
from src.core.recall.input_to_vector import QueryEncoder
from src.core.recall.vector_path import VectorPath
from src.core.recall.label_path import LabelRecallPath
from src.core.recall.collaboration_path import CollaborativeRecallPath

# 限制底层 OMP 线程，防止与 Python 线程池冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class TotalRecallSystem:
    """
    全量召回调度调度系统
    核心架构：向量路 + 标签路 并行执行 -> 协同路补充 -> RRF 融合输出 Top 500
    """

    def __init__(self):
        print("[*] 正在初始化全量召回系统，请稍候...", flush=True)
        # 1. 初始化各路组件
        self.encoder = QueryEncoder()
        self.v_path = VectorPath(recall_limit=500)  # 向量路
        self.l_path = LabelRecallPath(recall_limit=500)  # 标签路
        self.c_path = CollaborativeRecallPath(recall_limit=500)  # 协同路

        # 2. 初始化并行执行器
        self.executor = ThreadPoolExecutor(max_workers=3)

    def execute(self, query_text):
        start_time = time.time()

        # 1. 编码 (提取编码耗时)
        query_vec, encode_duration = self.encoder.encode(query_text)
        faiss.normalize_L2(query_vec)
        encode_ms = encode_duration * 1000

        # 2. 并行执行向量路与标签路
        future_v = self.executor.submit(self.v_path.recall, query_vec)
        future_l = self.executor.submit(self.l_path.recall, query_vec)

        # 获取子路径返回的 (list, ms_cost)
        v_list, v_cost = future_v.result()
        l_list, l_cost = future_l.result()

        # 3. 协同路扩展
        seeds = list(set(v_list + l_list))
        c_list, c_cost = self.c_path.recall(seeds)

        # 4. 融合
        final_list, _ = self._fuse_results(v_list, l_list, c_list)

        return {
            "final_top_500": final_list,
            "total_ms": (time.time() - start_time) * 1000,
            "details": {
                "encode_cost": encode_ms,
                "v_cost": v_cost,
                "l_cost": l_cost,
                "c_cost": c_cost
            }
        }

    def _fuse_results(self, v_res, l_res, c_res):
        """
        Reciprocal Rank Fusion (RRF) 结果融合
        """
        rrf_k = 60
        scores = {}

        # 权重分配：向量与标签 1.0，协同作为补充设为 0.5
        weights = {"vector": 1.0, "label": 1.0, "collab": 0.5}

        for path_name, res_list in [("vector", v_res), ("label", l_res), ("collab", c_res)]:
            w = weights[path_name]
            for rank, aid in enumerate(res_list):
                score = w * (1.0 / (rrf_k + rank + 1))
                scores[aid] = scores.get(aid, 0) + score

        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_candidates[:500]], scores


if __name__ == "__main__":
    system = TotalRecallSystem()

    print("\n" + "=" * 60)
    print("🚀 人才推荐系统 - 三路并行召回控制台")
    print("输入招聘需求执行端到端召回")
    print("=" * 60)

    try:
        while True:
            user_input = input("\n请输入搜索关键词 (或输入 'q' 退出): \n>> ").strip()
            if not user_input or user_input.lower() == 'q': break

            print(f"[*] 正在为需求执行多路召回策略...")
            results = system.execute(user_input)
            details = results['details']

            print(f"\n[召回任务完成]")
            # --- 新增分项耗时展示 ---
            print(f" ├─ 语义编码耗时: {details['encode_cost']:.2f} ms")
            print(f" ├─ 向量路召回:   {details['v_cost']:.2f} ms")
            print(f" ├─ 标签路召回:   {details['l_cost']:.2f} ms")
            print(f" └─ 协同路扩展:   {details['c_cost']:.2f} ms")
            print(f"- 总响应时间: {results['total_ms']:.2f} ms")
            print(f"- 最终融合候选池: {len(results['final_top_500'])} 人")

            print("-" * 30)
            print("【精排候选人 Top 10 预览】:")
            print(results['final_top_500'][:10])
            print("-" * 30)

    except KeyboardInterrupt:
        print("\n系统已关闭")