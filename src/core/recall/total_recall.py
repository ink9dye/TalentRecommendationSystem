import time
import logging
import sys
import io
from concurrent.futures import ThreadPoolExecutor
from vector_recall import VectorRecallPath
from label_racll import LabelRecallPath
from collaboration_racall import CollaborativeRecallPath

# 解决 Windows 环境下可能的输出乱码
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class TotalRecall:
    """
    全量召回调度器：整合三路召回并控制超时
    """

    def __init__(self):
        print("[*] 正在初始化全量召回系统，请稍候...")

        # 1. 初始化向量路 (SBERT + Faiss)
        try:
            self.v_path = VectorRecallPath()
            print("[OK] 向量路组件加载成功")
        except Exception as e:
            logging.error(f"向量路加载失败: {e}")
            self.v_path = None

        # 2. 初始化标签路 (Neo4j)
        try:
            self.l_path = LabelRecallPath()
            if self.l_path.graph is None:
                raise ConnectionError("Neo4j 服务未连接")
            print("[OK] 标签路组件加载成功")
        except Exception as e:
            logging.warning(f"标签路加载失败 (将跳过该路): {e}")
            self.l_path = None

        # 3. 初始化协同路 (SQLite Index)
        try:
            self.c_path = CollaborativeRecallPath()
            print("[OK] 协同路组件加载成功")
        except Exception as e:
            logging.error(f"协同路加载失败: {e}")
            self.c_path = None

        # 维持并发线程池
        self.executor = ThreadPoolExecutor(max_workers=3)

    def execute(self, query_text, job_id):
        """
        执行完整召回流程
        """
        start_time = time.time()
        timeout_limit = 0.8  # 考虑到 Neo4j 响应，适当放宽时间片

        # --- 阶段 1: 向量路 + 标签路 并行启动 ---
        v_res, l_res = [], []

        # 提交向量召回任务
        future_v = self.executor.submit(self.v_path.recall, query_text, timeout=timeout_limit) if self.v_path else None
        # 提交标签召回任务
        future_l = self.executor.submit(self.l_path.recall, job_id, timeout=timeout_limit) if self.l_path else None

        # 获取向量路结果
        if future_v:
            try:
                v_res = future_v.result(timeout=timeout_limit + 0.2)
            except Exception as e:
                logging.error(f"Vector path execution error: {e}")

        # 获取标签路结果
        if future_l:
            try:
                l_res = future_l.result(timeout=timeout_limit + 0.2)
            except Exception as e:
                logging.error(f"Label path execution error: {e}")

        # --- 阶段 2: 协同路 启动 ---
        # 提取去重后的作者 ID 作为种子进行协作挖掘
        seeds = list(set(v_res + l_res))
        c_res = []
        if seeds and self.c_path:
            try:
                future_c = self.executor.submit(self.c_path.recall, seeds, timeout=timeout_limit)
                c_res = future_c.result(timeout=timeout_limit + 0.2)
            except Exception as e:
                logging.error(f"Collaborative path execution error: {e}")

        # --- 结果融合 (Fusion) ---
        # 优先级：标签路(精确) > 向量路(语义) > 协同路(扩展)
        final_candidates, scores = self._fuse_results(l_res, v_res, c_res)

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
        rrf_k = 60
        scores = {}
        # 定义权重：标签路和向量路由于准确度高，权重设为 1.0；协同路作为补充设为 0.5
        weights = {"label": 1.0, "vector": 1.0, "collab": 0.5}

        for path_name, res_list in [("label", label_list), ("vector", vector_list), ("collab", collab_list)]:
            w = weights[path_name]
            for rank, aid in enumerate(res_list):
                score = w * (1.0 / (rrf_k + rank + 1))
                scores[aid] = scores.get(aid, 0) + score

        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_candidates[:500]], scores


# --- 交互式本地调试模块 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    # 1. 预初始化系统
    tr = TotalRecall()

    print("\n" + "=" * 60)
    print(f"🚀 人才推荐系统 - 全量召回调试控制台")
    print("=" * 60)

    try:
        while True:
            # 2. 提示输入文本
            print("\n请输入搜索关键词 (或输入 'q' 退出):")
            user_input = input(">> ").strip()

            if user_input.lower() == 'q':
                print("程序已退出。")
                break

            if not user_input:
                continue

            # 模拟一个 Job ID（实际应用中通常由前端传入）
            test_job_id = "j_001"

            print(f"[*] 正在执行召回逻辑...")
            results = tr.execute(user_input, test_job_id)

            # 3. 打印各路结果
            print("-" * 30)
            for path_name in ["label", "vector", "collab"]:
                ids = results["details"][path_name]
                count = len(ids)
                print(f"[{path_name.upper()} 路径] 找到: {count} 个候选人")
                if ids:
                    print(f"  前 10 个 ID: {ids[:10]}")
                print("-" * 30)

            # 4. 打印最终汇总
            final = results["final"]
            print(f"\n[最终融合候选人] 总计: {len(final)}")
            print(f"样本展示 (前 10): {final[:10]}")
            print(f"总耗时: {results['time']:.4f}s")

            if results['time'] > 1.0:
                print("⚠️ 警告: 召回耗时超过 1000ms SLA!")
            else:
                print("✅ 召回性能达标。")
            print("=" * 60)

    except KeyboardInterrupt:
        print("\n检测到强制退出，正在关闭...")