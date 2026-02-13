import os
import faiss
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from config import SBERT_DIR


class QueryEncoder:
    """
    语义编码器：将自然语言需求转化为 384 维稠密向量
    """

    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        if not os.path.exists(SBERT_DIR):
            os.makedirs(SBERT_DIR)

        # 核心：模型加载只在 __init__ 中执行一次
        print(f"[*] 正在初始化编码器并加载 SBERT 模型 (目录: {SBERT_DIR})...", flush=True)
        start_load = time.time()
        self.model = SentenceTransformer(model_name, cache_folder=SBERT_DIR)
        print(f"[OK] 模型加载完毕，耗时: {time.time() - start_load:.4f}s")

    def encode(self, text):
        if not text:
            return None

        # 推理阶段：复用已加载的模型
        start_encode = time.time()

        # 开启 normalize_embeddings 确保输出向量模长为 1，直接适配 Faiss 的内积检索
        vector = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')

        duration = time.time() - start_encode
        return vector, duration


if __name__ == "__main__":
    # 配置 NumPy 确保打印全量数值，不使用省略号
    np.set_printoptions(threshold=np.inf, suppress=True)

    # 实例化对象（执行一次性加载）
    encoder = QueryEncoder()

    print("\n" + "=" * 50)
    print("🚀 语义向量转换独立测试模块")
    print("输入 'q' 或 'exit' 退出")
    print("=" * 50)

    try:
        while True:
            user_input = input("\n请输入搜索关键词/需求描述: ").strip()

            if user_input.lower() in ['q', 'exit']:
                break
            if not user_input:
                continue

            # 执行转换
            vec, cost_ms = encoder.encode(user_input)

            # 格式化输出
            print(f"\n[结果反馈]")
            print(f"- 处理耗时: {cost_ms * 1000:.2f} ms")  # 转换为毫秒展示
            print(f"- 向量维度: {vec.shape}")

            # 打印完整向量（转换为列表展示更整洁）
            full_vector = vec[0].tolist()
            # print(f"- 完整向量内容 (前5维): {full_vector[:5]} ... (后5维): {full_vector[-5:]}")

            # 如果你需要查看每一个维度的数值，请取消下面这一行的注释
            print(f"- 稠密向量全量数据: \n{full_vector}")

            print("-" * 50)

    except KeyboardInterrupt:
        print("\n[!] 用户强制退出")
    finally:
        print("[*] 编码器已关闭")