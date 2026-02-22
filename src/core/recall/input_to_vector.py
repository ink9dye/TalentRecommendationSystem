import os
import sqlite3
import re
import time
import json
import collections
import numpy as np
import faiss  # 用于后续高效向量检索的库
from sentence_transformers import SentenceTransformer
from config import SBERT_DIR, DB_PATH, SBERT_MODEL_NAME # 增加导入


class QueryEncoder:
    """
    语义编码器：将自然语言需求转化为 768 维稠密向量。
    核心技术：【动态自共振增强 (Dynamic Resonance)】
    原理：通过重复出现核心技术词，人为提高 Transformer 模型在注意力机制（Attention）中对这些词的权重。
    """

    def __init__(self, model_name=None):
        """
        初始化编码器
        :param model_name: 预训练模型名。设置为 None 时自动加载 config.py 中的 SBERT_MODEL_NAME (BGE-M3)。
        """
        # 1. 确保缓存目录存在
        if not os.path.exists(SBERT_DIR):
            os.makedirs(SBERT_DIR)

        # 2. 核心修正：将默认值设为 None，确保在不传参时默认启用 BGE-M3 (768维)
        # 这对于处理你定义的 17 个学科领域的跨语言对齐至关重要
        self.active_model_name = model_name if model_name else SBERT_MODEL_NAME

        print(f"[*] 正在初始化编码器并加载 SBERT 模型: {self.active_model_name}...", flush=True)
        start_load = time.time()

        # 3. 加载 SBERT 模型（自动下载或从本地 SBERT_DIR 加载）
        self.model = SentenceTransformer(self.active_model_name, cache_folder=SBERT_DIR,
            trust_remote_code=True)

        # 4. 执行动态词库构建（这是本类的精华：让模型具备“行业常识”）
        self.hardcore_lexicon = self._build_dynamic_lexicon()

        print(f"[OK] 动态特征库加载完毕 (核心词条: {len(self.hardcore_lexicon)})")
        print(f"[*] 语义编码器就绪，当前维度: 768，耗时: {time.time() - start_load:.4f}s")

    def _build_dynamic_lexicon(self):
        """
        【统计学过滤逻辑】
        利用 DF (Document Frequency) 逆文档频率的思想，从数据库中提取具有“高区分度”的技术词。
        """
        lexicon = set()
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # A. 提取全量岗位技能库：这是我们计算词频的基数
            cursor.execute("SELECT skills FROM jobs WHERE skills IS NOT NULL")
            all_skills_data = [row[0].lower() for row in cursor.fetchall()]
            total_docs = len(all_skills_data)

            # 词频统计
            all_words = []
            for s in all_skills_data:
                # 正则解析：匹配英文单词和中文字符，过滤单字（如 'a', 'c' 等噪声）和纯数字
                words = re.findall(r'[a-zA-Z\u4e00-\u9fa5]{2,}', s)
                all_words.extend(words)

            word_counts = collections.Counter(all_words)

            # B. 引入标准词库：从 vocabulary 表中获取基准术语，确保提取的是“技术词”而非“日常词”
            cursor.execute("SELECT term FROM vocabulary")
            vocab_terms = {row[0].strip().lower() for row in cursor.fetchall()}

            # C. 统计筛选阈值（黄金分割逻辑）：
            # 1. 上限 (Upper Bound): 8%。如果一个词出现太频繁（如 "Python", "Team"），它就不具备区分度了。
            # 2. 下限 (Lower Bound): 3次。防止偶然出现的拼写错误进入词库。
            upper_limit = total_docs * 0.03
            lower_limit = 3

            for word, freq in word_counts.items():
                # 交叉验证：必须是标准术语 + 频率适中
                if word in vocab_terms:
                    if lower_limit <= freq <= upper_limit:
                        lexicon.add(word)

            conn.close()
        except Exception as e:
            print(f"[Warning] 动态词库自动构建失败，回退至空配置: {e}")

        return lexicon

    def _apply_dynamic_resonance(self, text):
        """
        【信号放大逻辑】
        如果原始 JD 为: "寻找一名 Java 开发人员"
        经过共振增强可能变为: "寻找一名 Java 开发人员 java java java"
        这会让词嵌入向量向 "Java" 的语义空间剧烈偏移。
        """
        if not text or not self.hardcore_lexicon:
            return text

        # 分词处理
        jd_words = re.findall(r'[a-zA-Z0-9\u4e00-\u9fa5]+', text.lower())

        # 识别文本中命中的核心硬核词
        hit_terms = []
        for w in jd_words:
            if w in self.hardcore_lexicon:
                hit_terms.append(w)

        if hit_terms:
            unique_hits = list(set(hit_terms))
            # 这里的重复 3 次是一个经验超参数，旨在不破坏整体语义的前提下通过冗余信息拉升权重
            resonance_string = " ".join([f"{t} {t} {t}" for t in unique_hits])
            return f"{text} {resonance_string}"

        return text

    def encode(self, text):
        """
        执行最终的向量化转换
        """
        if not text:
            return None

        # 步骤 1: 文本增强（注入自共振信号）
        enhanced_text = self._apply_dynamic_resonance(text)

        start_encode = time.time()

        # 步骤 2: 推理计算
        # normalize_embeddings=True 使向量长度归一化为 1。
        # 这样两个向量的点积(Inner Product)就等价于余弦相似度(Cosine Similarity)。
        vector = self.model.encode(
            [enhanced_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')  # Faiss 默认使用 float32 提高计算效率

        duration = time.time() - start_encode
        return vector, duration



if __name__ == "__main__":


    # 配置 NumPy 确保打印全量数值（防止中间出现省略号）
    np.set_printoptions(threshold=np.inf, suppress=True)

    # 实例化对象
    encoder = QueryEncoder()

    print("\n" + "=" * 60)
    print("🚀 动态自共振编码器 (Dynamic Resonance) 测试模式")
    print("=" * 60)

    try:
        while True:
            user_input = input("\n请输入岗位描述 (输入 q 退出): ").strip()

            if user_input.lower() in ['q', 'exit']:
                break
            if not user_input:
                continue

            # 执行编码
            vec, cost_ms = encoder.encode(user_input)

            print(f"\n[处理结果]")
            print(f"- 耗时: {cost_ms * 1000:.2f} ms")
            print(f"- 动态识别出的增强词典规模: {len(encoder.hardcore_lexicon)}")
            print(f"- 向量维度: {vec.shape}")

            # --- 核心修改：打印可直接复制的向量数值 ---
            # vec 的形状是 (1, 768)，通过 [0] 取出具体的向量，tolist() 转为 Python 列表
            vector_json = json.dumps(vec[0].tolist())
            print(f"- 转化后的向量 (JSON 格式，可直接复制):")
            print(vector_json)
            # ---------------------------------------

            # 如果想看增强后的文本（仅调试用）
            print(f"\n- 实际编码文本: \n{encoder._apply_dynamic_resonance(user_input)}")

            print("-" * 60)

    except KeyboardInterrupt:
        print("\n[!] 退出")