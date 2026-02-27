import os
import sqlite3
import re
import time
import json
import collections
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer
from optimum.intel import OVModelForFeatureExtraction  # 引入 OpenVINO 推理引擎
from config import SBERT_DIR, DB_PATH, SBERT_MODEL_NAME


class QueryEncoder:
    """
    语义编码器：OpenVINO 加速版
    集成核心技术：【动态自共振增强 (Dynamic Resonance)】
    """

    def __init__(self, model_name=None):
        # 1. 路径校准：指向固化后的模型子文件夹
        model_abs_path = os.path.abspath(SBERT_DIR)
        self.active_model_name = model_name if model_name else SBERT_MODEL_NAME

        print(f"[*] 正在初始化 OpenVINO 线上编码器: {self.active_model_name}...", flush=True)
        start_load = time.time()

        # 2. [修正] 加载分词器：增加本地文件强制选项
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_abs_path,
            trust_remote_code=True,
            local_files_only=True # 确保不再尝试联网
        )

        # 3. [核心修正] 加载 OpenVINO 模型
        # 必须显式设置 export=False，因为我们直接读取索引脚本转换好的“熟食”
        self.model = OVModelForFeatureExtraction.from_pretrained(
            model_abs_path,
            device="CPU",
            compile=True,       # 线上推理必须 compile 以换取最低延迟
            export=False,       # [关键] 强制关闭自动导出，避开架构识别报错
            local_files_only=True,
            trust_remote_code=True,
            task="feature-extraction" # 显式任务对齐
        )

        # 4. 执行动态词库构建（保留你的原创精华）
        self.hardcore_lexicon = self._build_dynamic_lexicon()

        print(f"[OK] 动态特征库加载完毕 (核心词条: {len(self.hardcore_lexicon)})")
        print(f"[*] 语义编码器就绪，已直接加载硬件加速模型，耗时: {time.time() - start_load:.4f}s")

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

    def _mean_pooling(self, model_output, attention_mask):
        """【关键】与离线端完全对齐的平均池化逻辑"""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, text):
        """
        执行最终的向量化转换（OpenVINO 硬件加速版）
        """
        if not text: return None

        # 步骤 1: 文本增强
        enhanced_text = self._apply_dynamic_resonance(text)
        start_encode = time.time()

        # 步骤 2: 推理计算
        inputs = self.tokenizer([enhanced_text], padding=True, truncation=True, max_length=1024, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 步骤 3: 池化与归一化（确保与 Faiss 索引库中的向量空间完全一致）
        embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        vector = embeddings.cpu().numpy().astype('float32')
        duration = time.time() - start_encode
        return vector, duration


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, suppress=True)
    encoder = QueryEncoder()
    print("\n" + "=" * 60)
    print("🚀 动态自共振编码器 (OpenVINO 加速版) 测试模式")
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