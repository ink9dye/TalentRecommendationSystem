import os
os.environ["HF_HUB_OFFLINE"] = "1"
import sqlite3
import re
import time
import collections
import numpy as np
from sentence_transformers import SentenceTransformer
from config import SBERT_DIR, DB_PATH, SBERT_MODEL_NAME


class QueryEncoder:
    """
    语义编码器：SentenceTransformer 版（与 build_vector_index 一致）
    集成核心技术：【动态自共振增强 (Dynamic Resonance)】
    """

    def __init__(self):

        print(f"[*] 正在初始化本地编码器: {SBERT_DIR}...", flush=True)
        start_load = time.time()

        self.model = SentenceTransformer(
            SBERT_DIR,
            trust_remote_code=True,
            device="cpu"
        )

        self.model.max_seq_length = 1024
        self.model.eval()

        self.hardcore_lexicon = self._build_dynamic_lexicon()

        print(f"[OK] 动态特征库加载完毕 (核心词条: {len(self.hardcore_lexicon)})")
        print(f"[*] 语义编码器就绪，耗时: {time.time() - start_load:.4f}s")
    def _build_dynamic_lexicon(self):
        """（保持原有的统计学过滤逻辑不变）"""
        lexicon = set()
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT skills FROM jobs WHERE skills IS NOT NULL")
            all_skills_data = [row[0].lower() for row in cursor.fetchall()]
            total_docs = len(all_skills_data)
            all_words = []
            for s in all_skills_data:
                words = re.findall(r'[a-zA-Z\u4e00-\u9fa5]{2,}', s)
                all_words.extend(words)
            word_counts = collections.Counter(all_words)
            cursor.execute("SELECT term FROM vocabulary")
            vocab_terms = {row[0].strip().lower() for row in cursor.fetchall()}
            upper_limit = total_docs * 0.03
            lower_limit = 3
            for word, freq in word_counts.items():
                if word in vocab_terms:
                    if lower_limit <= freq <= upper_limit:
                        lexicon.add(word)
            conn.close()
        except Exception as e:
            print(f"[Warning] 动态词库自动构建失败: {e}")
        return lexicon

    def _apply_dynamic_resonance(self, text):
        """（保持原有的信号放大逻辑不变）"""
        if not text or not self.hardcore_lexicon: return text
        jd_words = re.findall(r'[a-zA-Z0-9\u4e00-\u9fa5]+', text.lower())
        hit_terms = [w for w in jd_words if w in self.hardcore_lexicon]
        if hit_terms:
            unique_hits = list(set(hit_terms))
            resonance_string = " ".join([f"{t} {t} {t}" for t in unique_hits])
            return f"{text} {resonance_string}"
        return text

    def encode(self, text):
        """
        执行向量化（SentenceTransformer，与 build_vector_index 向量空间一致）
        返回 (vector, duration)，vector 为 (1, dim) 的 float32 数组，已 L2 归一化。
        """
        if not text: return None

        enhanced_text = self._apply_dynamic_resonance(text)
        start_encode = time.time()

        vector = self.model.encode(
            [enhanced_text],
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype('float32')

        duration = time.time() - start_encode
        return vector, duration


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, suppress=True)
    encoder = QueryEncoder()
    print("\n" + "=" * 60)
    print("🚀 动态自共振编码器 (SentenceTransformer) 测试模式")
    print("=" * 60)

    try:
        while True:
            user_input = input("\n请输入岗位描述 (输入 q 退出): ").strip()
            if user_input.lower() in ['q', 'exit']: break
            if not user_input: continue

            vec, cost_s = encoder.encode(user_input)
            print(f"\n[处理结果]")
            print(f"- 耗时: {cost_s * 1000:.2f} ms")
            print(f"- 向量维度: {vec.shape}")
            print(f"- 实际编码文本: {encoder._apply_dynamic_resonance(user_input)}")
            print("-" * 60)
    except KeyboardInterrupt:
        print("\n[!] 退出")
