import faiss
import numpy as np
import json
import os
import sqlite3
import time
from sentence_transformers import SentenceTransformer

# 导入全局配置
# 确保项目根目录在 PYTHONPATH 中，或者使用相对导入
from config import CONFIG_DICT, ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH, SBERT_DIR, DB_PATH


class VectorRecallPath:
    """
    向量路召回：基于 SBERT + Faiss 的语义召回
    职责：实现从需求文本到候选作者 ID 的高效转换
    """

    def __init__(self, top_k=None):
        # 1. 从配置字典获取全局参数
        self.config = CONFIG_DICT
        self.top_k = top_k or self.config.get("BATCH_SIZE", 300)  # 默认回退到 300

        # 2. 加载本地 SBERT 模型 (384 维映射)
        if not os.path.exists(SBERT_DIR):
            raise FileNotFoundError(f"SBERT 模型路径不存在: {SBERT_DIR}")
        self.model = SentenceTransformer(SBERT_DIR)

        # 3. 加载 Faiss 离线索引
        if not os.path.exists(ABSTRACT_INDEX_PATH):
            raise FileNotFoundError(f"未找到摘要向量索引: {ABSTRACT_INDEX_PATH}")
        self.index = faiss.read_index(ABSTRACT_INDEX_PATH)

        # 4. 加载 WorkID 映射表
        if not os.path.exists(ABSTRACT_MAP_PATH):
            raise FileNotFoundError(f"未找到 ID 映射文件: {ABSTRACT_MAP_PATH}")
        with open(ABSTRACT_MAP_PATH, 'r', encoding='utf-8') as f:
            self.id_map = json.load(f)

    def _get_authors_by_works(self, work_ids, deadline):
        """分批查询作者，并在到达截止时间时中断"""
        if not work_ids: return []

        conn = sqlite3.connect(DB_PATH)
        author_ids = set()
        batch_size = 50  # 每次查询 50 个作品

        for i in range(0, len(work_ids), batch_size):
            # 检查是否超时
            if time.time() > deadline:
                break

            batch = work_ids[i:i + batch_size]
            placeholders = ','.join(['?'] * len(batch))
            query = f"SELECT DISTINCT author_id FROM authorships WHERE work_id IN ({placeholders})"

            cursor = conn.execute(query, batch)
            author_ids.update([row[0] for row in cursor.fetchall()])

        conn.close()
        return list(author_ids)

    def recall(self, query_text, timeout=0.5):
        start_time = time.time()
        deadline = start_time + timeout

        # A. 语义编码 (SBERT)
        query_vector = self.model.encode([query_text]).astype('float32')

        # B. 向量检索 (Faiss)
        _, indices = self.index.search(query_vector, self.top_k)
        candidate_work_ids = [self.id_map[str(idx)] for idx in indices[0] if idx != -1]

        # C. 关系映射 (带超时检查的增量查询)
        # 即使只查了一半的作品，也会返回这一半对应的作者
        candidate_author_ids = self._get_authors_by_works(candidate_work_ids, deadline)

        return candidate_author_ids[:self.top_k]