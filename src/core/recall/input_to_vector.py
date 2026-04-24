import os
os.environ["HF_HUB_OFFLINE"] = "1"
import time
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from config import SBERT_DIR


class QueryEncoder:
    """
    语义编码器：SentenceTransformer 版（与 build_vector_index 一致）
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

        print(f"[*] 语义编码器就绪，耗时: {time.time() - start_load:.4f}s")
        # 原文 → (1, dim) 向量；与 lookup_or_encode / encode_cache 键一致，供 encode / batch 去重
        self._embed_dedup_cache: Dict[str, np.ndarray] = {}

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Stable encoding path for local gte model.

        Why: In some environments, remote-code model forward may produce invalid internal position_ids.
        We pass explicit position_ids to keep behavior stable. Pooling follows the model's
        sentence-transformers config (gte-multilingual-base uses CLS pooling).
        """
        if not texts:
            dim = int(self.model.get_sentence_embedding_dimension())
            return np.zeros((0, dim), dtype=np.float32)

        # sentence-transformers layout: [0]=Transformer, [1]=Pooling, ...
        transformer = self.model[0]
        auto_model = getattr(transformer, "auto_model", None)
        tokenizer = getattr(transformer, "tokenizer", None)
        if auto_model is None or tokenizer is None:
            # Fallback to library default if internal structure differs.
            vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)

        with torch.no_grad():
            batch = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=int(getattr(self.model, "max_seq_length", 1024) or 1024),
                return_tensors="pt",
            )
            seq_len = int(batch["input_ids"].shape[1])
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch["input_ids"].shape[0], -1)
            outputs = auto_model(**batch, position_ids=position_ids)
            last_hidden = outputs.last_hidden_state  # [bs, seq, dim]
            cls = last_hidden[:, 0, :]  # CLS pooling
            cls = F.normalize(cls, p=2, dim=1)
            return cls.cpu().numpy().astype(np.float32)

    def clear_embed_dedup_cache(self) -> None:
        """单次召回入口清空，避免跨查询无限增长；键为原文，与 encode/lookup 一致。"""
        self._embed_dedup_cache.clear()

    def encode(self, text):
        """
        执行向量化（SentenceTransformer，与 build_vector_index 向量空间一致）
        返回 (vector, duration)，vector 为 (1, dim) 的 float32 数组，已 L2 归一化。
        """
        if not text:
            return None, 0.0

        if text in self._embed_dedup_cache:
            return self._embed_dedup_cache[text].copy(), 0.0

        start_encode = time.time()

        vector = self._encode_texts([text]).astype("float32")

        duration = time.time() - start_encode
        self._embed_dedup_cache[text] = vector
        return vector, duration

    def lookup_or_encode(self, text: str, cache: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        同 encode 的文本预处理与模型行为，按原文去重缓存。
        用于单条 JD 内多锚点/多论文标题复用，数值与未缓存的 encode 一致。
        """
        if not text:
            return None
        if text in cache:
            vec = cache[text]
            if text not in self._embed_dedup_cache:
                self._embed_dedup_cache[text] = vec
            return vec
        if text in self._embed_dedup_cache:
            vector = self._embed_dedup_cache[text]
            cache[text] = vector
            return vector
        start_encode = time.time()
        vector = self._encode_texts([text]).astype("float32")
        _ = time.time() - start_encode
        cache[text] = vector
        self._embed_dedup_cache[text] = vector
        return vector

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量编码，每行与对同一字符串单独 encode 等价（同一归一化）。
        返回 shape (n, dim) 的 float32；texts 为空时返回 shape (0, dim) 的空数组。
        命中实例级 _embed_dedup_cache 时跳过该行前向；batch 内相同原文只 forward 一次。
        """
        if not texts:
            dim = int(self.model.get_sentence_embedding_dimension())
            return np.zeros((0, dim), dtype=np.float32)
        dim = int(self.model.get_sentence_embedding_dimension())
        n = len(texts)
        out = np.zeros((n, dim), dtype=np.float32)
        missing_indices: List[int] = []
        missing_texts: List[str] = []
        for i, t in enumerate(texts):
            if t in self._embed_dedup_cache:
                row = self._embed_dedup_cache[t]
                out[i, :] = np.asarray(row, dtype=np.float32).reshape(-1)[:dim]
            else:
                missing_indices.append(i)
                missing_texts.append(t)
        if not missing_indices:
            return out
        unique_order: List[str] = []
        seen: Set[str] = set()
        for t in missing_texts:
            if t not in seen:
                seen.add(t)
                unique_order.append(t)
        batch_vecs = self._encode_texts(unique_order).astype(np.float32)
        for j, t in enumerate(unique_order):
            self._embed_dedup_cache[t] = batch_vecs[j : j + 1].copy()
        text_to_row = {t: batch_vecs[j] for j, t in enumerate(unique_order)}
        for idx, t in zip(missing_indices, missing_texts):
            out[idx, :] = text_to_row[t]
        return out


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, suppress=True)
    encoder = QueryEncoder()
    print("\n" + "=" * 60)
    print("🚀 语义编码器 (SentenceTransformer) 测试模式")
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
            print(f"- 实际编码文本: {user_input}")
            print("-" * 60)
    except KeyboardInterrupt:
        print("\n[!] 退出")
