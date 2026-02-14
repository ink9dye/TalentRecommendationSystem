import time
import json
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor
from src.core.recall.input_to_vector import QueryEncoder
from src.core.recall.vector_path import VectorPath
from src.core.recall.label_path import LabelRecallPath
from src.core.recall.collaboration_path import CollaborativeRecallPath


class TotalRecallSystem:
    """
    全量召回调度调度系统
    核心架构：向量路 + 标签路 并行执行 -> 协同路补充 -> RRF 融合输出 Top 500
    """

    def __init__(self):
        print("[*] 正在初始化全量召回系统，请稍候...", flush=True)
        # 1. 初始化各路组件
        self.encoder = QueryEncoder()
        self.v_path = VectorPath(recall_limit=300)  # 向量路，扩大池子以备精排
        self.l_path = LabelRecallPath(recall_limit=300)  # 标签路
        self.c_path = CollaborativeRecallPath(recall_limit=200)  # 协同路

        # 2. 初始化并行执行器
        self.executor = ThreadPoolExecutor(max_workers=2)

    def execute(self, query_text):
        start_time = time.time()

        # 1. 编码
        query_vec, encode_duration = self.encoder.encode(query_text)
        faiss.normalize_L2(query_vec)

        # 2. 并行执行向量路与标签路
        future_v = self.executor.submit(self.v_path.recall, query_vec)
        future_l = self.executor.submit(self.l_path.recall, query_vec)

        v_list, v_cost = future_v.result()
        l_list, l_cost = future_l.result()

        # 3. 协同路扩展
        seeds = list(set(v_list + l_list))
        # 确保 collaboration_path.py 的 recall 也是返回 (ids, cost)
        c_list, c_cost = self.c_path.recall(seeds)

        # 4. 融合
        final_list, _ = self._fuse_results(v_list, l_list, c_list)

        return {
            "final_top_500": final_list,
            "total_ms": (time.time() - start_time) * 1000,
            "details": {"v_cost": v_cost, "l_cost": l_cost, "c_cost": c_cost}
        }

    def _fuse_results(self, v_res, l_res, c_res):
        """
        Reciprocal Rank Fusion (RRF) 结果融合
        通过各路顺位倒数之和进行重排
        """
        rrf_k = 60  # RRF 常数
        scores = {}

        # 权重分配：向量与标签 1.0，协同作为补充设为 0.8
        weights = {"vector": 1.0, "label": 1.0, "collab": 0.5}

        for path_name, res_list in [("vector", v_res), ("label", l_res), ("collab", c_res)]:
            w = weights[path_name]
            for rank, aid in enumerate(res_list):
                # RRF 公式：score = sum( w / (k + rank) )
                score = w * (1.0 / (rrf_k + rank + 1))
                scores[aid] = scores.get(aid, 0) + score

        # 按融合分倒序排列
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_candidates[:500]], scores


if __name__ == "__main__":
    system = TotalRecallSystem()

    print("\n" + "=" * 60)
    print("🚀 人才推荐系统 - 三路并行召回控制台")
    print("输入招聘需求（如：Java开发、云计算架构师）执行端到端召回")
    print("=" * 60)

    try:
        while True:
            user_input = input("\n请输入搜索关键词 (或输入 'q' 退出): \n>> ").strip()
            if not user_input or user_input.lower() == 'q': break

            print(f"[*] 正在为需求执行多路召回策略...")
            results = system.execute(user_input)

            print(f"\n[召回任务完成]")
            print(f"- 总响应时间: {results['total_ms']:.2f} ms")
            print(f"- 最终融合候选池: {len(results['final_top_500'])} 人")

            # 打印融合后的前 10 名
            print("-" * 30)
            print("【精排候选人 Top 10 预览】:")
            print(results['final_top_500'][:10])
            print("-" * 30)

    except KeyboardInterrupt:
        print("\n系统已关闭")