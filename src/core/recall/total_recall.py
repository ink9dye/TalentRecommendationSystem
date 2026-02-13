import time
import logging
import sys
import io
import traceback
import faiss  # 导入用于归一化
from concurrent.futures import ThreadPoolExecutor
from vector_path import VectorRecallPath
from label_path import LabelRecallPath
from collaboration_path import CollaborativeRecallPath

if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class TotalRecall:
    def __init__(self):
        print("[*] 正在初始化全量召回系统，请稍候...", flush=True)

        try:
            self.v_path = VectorRecallPath()
            print("[OK] 向量路组件加载成功")
        except Exception as e:
            logging.error(f"向量路加载失败: {e}")
            self.v_path = None

        try:
            self.l_path = LabelRecallPath()
            print("[OK] 标签路组件加载成功")
        except Exception as e:
            logging.warning(f"标签路加载失败: {e}")
            self.l_path = None

        try:
            self.c_path = CollaborativeRecallPath()
            print("[OK] 协同路组件加载成功")
        except Exception as e:
            logging.error(f"协同路加载失败: {e}")
            self.c_path = None

        self.executor = ThreadPoolExecutor(max_workers=3)

    def execute(self, query_text):
        """
        全量召回执行逻辑
        优化：单次编码 + 向量分发，解决 TimeoutError
        """
        start_time = time.time()
        v_res, l_res, c_res = [], [], []

        try:
            # 1. 核心优化：在主线程进行单次语义编码，避免多线程竞争 GIL
            print(f"[*] 正在生成语义向量...", flush=True)
            query_vector = self.v_path.model.encode([query_text]).astype('float32')
            faiss.normalize_L2(query_vector)

            # 2. 语义锚定：获取标签 ID (复用向量)
            top_vocab_ids = self.v_path.get_top_vocab(query_vector=query_vector)
            print(f"[*] 锚定标签 ID: {top_vocab_ids}")

            # 3. 并行执行向量路与标签路 (放宽超时至 2.0s 确保稳定)
            # 传递 query_vector 避免 v_path.recall 再次编码
            future_v = self.executor.submit(self.v_path.recall, query_vector=query_vector)
            future_l = self.executor.submit(self.l_path.recall_by_vocab_ids, top_vocab_ids)

            try:
                v_res = future_v.result(timeout=2.0)
            except Exception:
                print("Vector Path Error Detail:")
                traceback.print_exc()

            try:
                l_res = future_l.result(timeout=2.0)
            except Exception:
                print("Label Path Error Detail:")
                traceback.print_exc()

            # 4. 协同路扩展
            seeds = list(set(v_res + l_res))
            if seeds and self.c_path:
                c_res = self.c_path.recall(seeds)

        except Exception as e:
            logging.error(f"Execution failed: {e}")
            traceback.print_exc()

        final_candidates, _ = self._fuse_results(l_res, v_res, c_res)

        return {
            "final": final_candidates,
            "details": {"vector": v_res, "label": l_res, "collab": c_res},
            "time": time.time() - start_time
        }

    def _fuse_results(self, label_list, vector_list, collab_list):
        rrf_k = 60
        scores = {}
        weights = {"label": 1.0, "vector": 1.0, "collab": 0.5}

        for path_name, res_list in [("label", label_list), ("vector", vector_list), ("collab", collab_list)]:
            w = weights[path_name]
            for rank, aid in enumerate(res_list):
                score = w * (1.0 / (rrf_k + rank + 1))
                scores[aid] = scores.get(aid, 0) + score

        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_candidates[:500]], scores


if __name__ == "__main__":
    tr = TotalRecall()
    print("\n" + "=" * 60)
    print(f"🚀 人才推荐系统 - 全量召回调试控制台")
    print("=" * 60)

    try:
        while True:
            user_input = input("\n请输入搜索关键词 (或输入 'q' 退出):\n>> ").strip()
            if user_input.lower() == 'q': break
            if not user_input: continue

            print(f"[*] 正在执行召回逻辑...")
            results = tr.execute(user_input)

            for path_name in ["label", "vector", "collab"]:
                ids = results["details"][path_name]
                print(f"[{path_name.upper()} 路径] 找到: {len(ids)} 个候选人")

            print(f"\n[最终融合候选人] 总计: {len(results['final'])}")
            print(f"总耗时: {results['time']:.4f}s")
            print("=" * 60)
    except KeyboardInterrupt:
        print("\n正在关闭...")