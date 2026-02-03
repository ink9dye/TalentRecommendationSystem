import faiss
import numpy as np
import json
import os
import sqlite3
from sentence_transformers import SentenceTransformer

# --- 路径动态初始化 ---
# 获取当前脚本文件的绝对路径 (src/core/recall/vector_recall.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 向上追溯三级目录到达项目根目录 (TalentRecommendationSystem)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

# 所有的相对路径现在都基于 PROJECT_ROOT
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sbert_models", "paraphrase-multilingual-MiniLM-L12-v2")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "faiss", "abstract_vector.index")
MAPPING_PATH = os.path.join(PROJECT_ROOT, "data", "faiss", "abstract_mapping.json")
SQLITE_DB_PATH = os.path.join(PROJECT_ROOT, "data", "sqlite", "academic_dataset_v5.db")


class VectorRecallPath:
    """
    向量路召回：基于 SBERT + Faiss 的语义召回 [cite: 1]
    职责：实现从需求文本到候选作者 ID 的高效转换
    """

    def __init__(self, top_k=300):
        # 1. 加载 SBERT 模型 (README 提到的 384 维映射) [cite: 2, 3]
        self.model = SentenceTransformer(MODEL_PATH)
        self.top_k = top_k

        # 2. 加载 Faiss 离线索引
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"未找到摘要向量索引，请检查路径: {FAISS_INDEX_PATH}")

        self.index = faiss.read_index(FAISS_INDEX_PATH)

        # 3. 加载 WorkID 映射表
        with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
            self.id_map = json.load(f)

    def _get_authors_by_works(self, work_ids):
        """通过论文 ID 从 SQLite 中查询对应的作者 ID [cite: 2, 3]"""
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()

        # 对应架构中通过作品关联作者的步骤 [cite: 1]
        placeholders = ','.join(['?'] * len(work_ids))
        query = f"SELECT DISTINCT author_id FROM authorships WHERE work_id IN ({placeholders})"

        cursor.execute(query, work_ids)
        author_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return author_ids

    def recall(self, query_text):
        """执行召回流程 """
        # A. 语义编码
        query_vector = self.model.encode([query_text]).astype('float32')

        # B. 向量检索
        _, indices = self.index.search(query_vector, self.top_k)

        # C. ID 转换
        candidate_work_ids = [self.id_map[str(idx)] for idx in indices[0] if idx != -1]

        # D. 关系映射
        candidate_author_ids = self._get_authors_by_works(candidate_work_ids)

        return candidate_author_ids[:self.top_k]