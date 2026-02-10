import time
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from vector_recall import VectorRecallPath
from label_racll import LabelRecallPath
from collaboration_racall import CollaborativeRecallPath


class TotalRecall:
    """
    全量召回调度器：整合三路召回并控制超时
    """

    def __init__(self):
        # 初始化三路召回组件
        self.v_path = VectorRecallPath()
        self.l_path = LabelRecallPath()
        self.c_path = CollaborativeRecallPath()
        # 维持 3 个线程池位点，对应架构中的并发设计
        self.executor = ThreadPoolExecutor(max_workers=3)

    def execute(self, query_text, job_id, debug=False):
        """
        执行完整召回流程
        Args:
            query_text: 用户/岗位需求文本
            job_id: 岗位 ID
            debug: 是否开启调试模式返回详细得分
        """
        start_time = time.time()
        # 分配时间片：前两路并行 500ms，协同路后续 500ms
        timeout_limit = 0.5

        # --- 阶段 1: 向量路 + 标签路 并行启动 ---
        future_v = self.executor.submit(self.v_path.recall, query_text, timeout=timeout_limit)
        future_l = self.executor.submit(self.l_path.recall, job_id, timeout=timeout_limit)

        # 获取向量路结果 (ID 列表)
        try:
            v_res = future_v.result(timeout=timeout_limit + 0.1)
        except Exception as e:
            logging.error(f"Vector path error: {e}")
            v_res = []

        # 获取标签路结果 (ID 列表)
        try:
            l_res = future_l.result(timeout=timeout_limit + 0.1)
        except Exception as e:
            logging.error(f"Label path error: {e}")
            l_res = []

        # --- 阶段 2: 协同路 启动 ---
        # 提取去重后的作者 ID 作为种子进行协作挖掘
        seeds = list(set(v_res + l_res))
        c_res = []
        if seeds:
            try:
                # 协同路执行基于 DeepICF 的协作扩展 [cite: 2, 4]
                future_c = self.executor.submit(self.c_path.recall, seeds, timeout=timeout_limit)
                c_res = future_c.result(timeout=timeout_limit + 0.1)
            except Exception as e:
                logging.error(f"Collaborative path error: {e}")
                c_res = []

        # --- 结果融合 (Fusion) ---
        # 优先级：标签路(精确) > 向量路(语义) > 协同路(扩展)
        final_candidates = self._fuse_results(l_res, v_res, c_res)

        elapsed = time.time() - start_time
        return {
            "final": final_candidates,
            "details": {
                "vector": v_res,
                "label": l_res,
                "collab": c_res
            },
            "time": elapsed
        }

    def _fuse_results(self, label_list, vector_list, collab_list):
        """
        使用 RRF (Reciprocal Rank Fusion) 算法融合多路名次
        """
        rrf_k = 60  # 标准常数，防止前几名权重过大
        scores = {}

        # 定义各路权重 (如果需要)
        weights = {"label": 1.0, "vector": 1.0, "collab": 0.5}

        for path_name, res_list in [("label", label_list), ("vector", vector_list), ("collab", collab_list)]:
            w = weights[path_name]
            for rank, aid in enumerate(res_list):
                # rank 从 0 开始，所以公式是 1 / (k + rank + 1)
                score = w * (1.0 / (rrf_k + rank + 1))
                scores[aid] = scores.get(aid, 0) + score

        # 根据计算出的 RRF 总分排序
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 返回前 500 名进入精排
        return [item[0] for item in sorted_candidates[:500]], scores


# --- 本地调试模块 ---
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # 初始化
    tr = TotalRecall()

    # 模拟输入
    QUERY = "计算机视觉 算法工程师"
    JOB_ID = "j_001"

    print("\n" + "=" * 60)
    print(f"🚀 Talent Recommendation System - Total Recall Debug")
    print("=" * 60)
    print(f"Input Query: {QUERY}")
    print(f"Job ID: {JOB_ID}\n")

    # 执行召回
    results = tr.execute(QUERY, JOB_ID)

    # 打印各路结果
    for path_name in ["label", "vector", "collab"]:
        ids = results["details"][path_name]
        print(f"[{path_name.upper()} PATH] Found: {len(ids)} candidates")
        # 打印前 5 个展示
        if ids:
            print(f"  Top 5 IDs: {ids[:5]}")
        else:
            print("  (Timed out or no matches)")
        print("-" * 30)

    # 打印最终汇总
    final = results["final"]
    print(f"\n[FINAL CANDIDATES] Total: {len(final)}")
    print(f"Sample: {final[:10]}")
    print(f"Total Elapsed Time: {results['time']:.4f}s")

    if results['time'] > 1.0:
        print("⚠️ Warning: Recall exceeded 1000ms SLA!")
    else:
        print("✅ Recall within 1000ms SLA.")
    print("=" * 60 + "\n")