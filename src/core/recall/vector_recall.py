import faiss
import numpy as np
import json
import os
import sqlite3
import time
from sentence_transformers import SentenceTransformer

# 导入全局配置
from config import (
    CONFIG_DICT,
    ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH,
    VOCAB_INDEX_PATH, VOCAB_MAP_PATH,
    SBERT_DIR, DB_PATH
)


class VectorRecallPath:
    """
    向量路召回：基于 SBERT + Faiss 的语义召回
    职责：1. 实现从需求文本到候选作者 ID 的转换；2. 为标签路提供语义锚定节点 (Vocabulary ID)
    """

    def __init__(self, top_k=None):
        # 1. 从配置字典获取全局参数
        self.config = CONFIG_DICT
        self.top_k = top_k or self.config.get("BATCH_SIZE", 300)

        # 2. 加载 SBERT 模型
        MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
        if not os.path.exists(SBERT_DIR):
            os.makedirs(SBERT_DIR)

        print(f"[*] 正在从缓存加载 SBERT 模型: {MODEL_NAME}")
        # 显式指定 cache_folder 确保模型加载路径正确
        self.model = SentenceTransformer(MODEL_NAME, cache_folder=SBERT_DIR)

        # 3. 加载 摘要(Abstract) 向量索引与映射
        self._load_index_resource()

    def _load_index_resource(self):
        """加载 Faiss 索引和对应的 ID 映射文件"""
        # 加载摘要索引
        if not os.path.exists(ABSTRACT_INDEX_PATH):
            raise FileNotFoundError(f"未找到摘要向量索引: {ABSTRACT_INDEX_PATH}")
        self.abstract_index = faiss.read_index(ABSTRACT_INDEX_PATH)

        with open(ABSTRACT_MAP_PATH, 'r', encoding='utf-8') as f:
            self.abstract_id_map = json.load(f)

        # 加载词汇(Vocabulary)索引，用于标签路的语义锚定
        if not os.path.exists(VOCAB_INDEX_PATH):
            print(f"[Warning] 未找到词汇向量索引: {VOCAB_INDEX_PATH}，标签路锚定将受限")
            self.vocab_index = None
        else:
            self.vocab_index = faiss.read_index(VOCAB_INDEX_PATH)
            with open(VOCAB_MAP_PATH, 'r', encoding='utf-8') as f:
                self.vocab_id_map = json.load(f)

    def get_top_vocab(self, query_text, top_n=3):
        """
        [新增] 语义锚定接口：获取与输入文本最匹配的 Top-N 个词汇节点 ID
        用于支撑标签路的 (Vocab)-[:HAS_TOPIC]-(Work) 推理路径
        """
        if self.vocab_index is None:
            return []

        # A. 编码并强制转换类型为 float32
        query_vector = self.model.encode([query_text]).astype('float32')
        faiss.normalize_L2(query_vector)

        # B. 检索 Top-N
        _, indices = self.vocab_index.search(query_vector, top_n)

        v_ids = []
        for idx in indices[0]:
            if idx == -1: continue
            # 将 Faiss 内部索引映射为数据库中的 Vocabulary ID
            str_idx = str(idx)
            if str_idx in self.vocab_id_map:
                v_ids.append(self.vocab_id_map[str_idx])

        return v_ids

    def _get_authors_by_works(self, work_ids, deadline):
        """从 SQLite 获取 Work 关联的作者"""
        if not work_ids: return []

        conn = sqlite3.connect(DB_PATH)
        author_ids = []
        seen_authors = set()
        batch_size = 50

        for i in range(0, len(work_ids), batch_size):
            if time.time() > deadline:
                break

            batch = work_ids[i:i + batch_size]
            placeholders = ','.join(['?'] * len(batch))
            query = f"SELECT author_id FROM authorships WHERE work_id IN ({placeholders})"

            cursor = conn.execute(query, batch)
            for row in cursor.fetchall():
                aid = row[0]
                if aid not in seen_authors:
                    author_ids.append(aid)
                    seen_authors.add(aid)

        conn.close()
        return author_ids

    def recall(self, query_text, timeout=0.5):
        """
        执行向量路召回 (从论文摘要维度)
        """
        start_time = time.time()
        deadline = start_time + timeout

        # A. 语义编码 (强制 float32 以防 Faiss 崩溃)
        query_vector = self.model.encode([query_text]).astype('float32')
        faiss.normalize_L2(query_vector)

        # B. 向量检索
        _, indices = self.abstract_index.search(query_vector, self.top_k)

        # C. 映射为 Work ID
        candidate_work_ids = []
        for idx in indices[0]:
            if idx == -1: continue
            str_idx = str(idx)
            if str_idx in self.abstract_id_map:
                candidate_work_ids.append(self.abstract_id_map[str_idx])

        # D. 关系映射获取作者
        candidate_author_ids = self._get_authors_by_works(candidate_work_ids, deadline)

        return candidate_author_ids[:self.top_k]