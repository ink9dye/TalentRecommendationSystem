import sqlite3
import json
import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 1. 解决控制台输出乱码
if sys.platform.startswith('win'):
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 2. 导入统一配置
from config import (
    DB_PATH, INDEX_DIR, SBERT_DIR,
    VOCAB_INDEX_PATH, VOCAB_MAP_PATH,
    ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH,
    JOB_INDEX_PATH, JOB_MAP_PATH
)

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
BATCH_SIZE = 128  # 向量化批处理大小


class VectorIndexGenerator:
    def __init__(self):
        # 核心修改：通过 cache_folder 指定模型下载/加载路径
        print(f"[*] 正在加载 SBERT 模型: {MODEL_NAME}")
        print(f"[*] 模型存储位置: {SBERT_DIR}")
        self.model = SentenceTransformer(MODEL_NAME, cache_folder=SBERT_DIR)

        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row

        if not os.path.exists(INDEX_DIR):
            os.makedirs(INDEX_DIR)

    def _save_index(self, name, embeddings, ids, index_path, map_path):
        """保存 Faiss 索引和 ID 映射，显式指定编码"""
        dimension = embeddings.shape[1]
        # 使用 Inner Product (内积) 索引，用于余弦相似度
        index = faiss.IndexFlatIP(dimension)

        # 归一化以支持余弦相似度
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        # 保存二进制索引文件
        faiss.write_index(index, index_path)

        # 解决文件乱码：显式指定 encoding='utf-8'
        with open(map_path, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 确保存入原始中文字符
            json.dump(ids, f, ensure_ascii=False, indent=4)

        print(f"[成功] 索引已保存: {os.path.basename(index_path)} | 记录数: {len(ids)}")

    def build_vocabulary_index(self):
        """1. 语义向量检索索引 (Vocabulary)"""
        print("\n>>> 任务 1: 构建词汇向量索引 (Vocabulary)")
        cursor = self.conn.cursor()
        rows = cursor.execute("SELECT voc_id, term FROM vocabulary").fetchall()

        texts = [row['term'] for row in rows]
        ids = [str(row['voc_id']) for row in rows]

        if not texts: return

        print(f"正在编码 {len(texts)} 个词汇...")
        embeddings = self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
        self._save_index("vocabulary", np.array(embeddings).astype('float32'), ids, VOCAB_INDEX_PATH, VOCAB_MAP_PATH)

    def build_abstract_index(self):
        """2. 摘要向量检索索引 (Abstracts)"""
        print("\n>>> 任务 2: 构建论文摘要向量索引 (Abstracts)")
        cursor = self.conn.cursor()
        # 只取有文本的摘要
        rows = cursor.execute("""
                              SELECT work_id, full_text_en
                              FROM abstracts
                              WHERE full_text_en IS NOT NULL
                                AND full_text_en != ''
                              """).fetchall()

        texts = [row['full_text_en'] for row in rows]
        ids = [row['work_id'] for row in rows]

        if not texts:
            print("警告: 未发现可用的摘要文本，请确保已运行摘要还原程序。")
            return

        print(f"正在编码 {len(texts)} 篇论文摘要...")
        embeddings = self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
        self._save_index("abstract", np.array(embeddings).astype('float32'), ids, ABSTRACT_INDEX_PATH,
                         ABSTRACT_MAP_PATH)

    def build_job_description_index(self):
        """3. 描述向量检索索引 (Jobs)"""
        print("\n>>> 任务 3: 构建岗位描述向量索引 (Jobs)")
        cursor = self.conn.cursor()
        # 结合岗位名和描述
        rows = cursor.execute("SELECT securityId, job_name, description FROM jobs").fetchall()

        texts = [f"{row['job_name']} {row['description']}" for row in rows]
        ids = [row['securityId'] for row in rows]

        if not texts: return

        print(f"正在编码 {len(texts)} 个岗位描述...")
        embeddings = self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
        self._save_index("job_description", np.array(embeddings).astype('float32'), ids, JOB_INDEX_PATH, JOB_MAP_PATH)

    def run_all(self):
        self.build_vocabulary_index()
        self.build_abstract_index()
        self.build_job_description_index()
        print("\n[所有索引构建完成]")
        print(f"模型存放于: {SBERT_DIR}")
        print(f"索引存放于: {INDEX_DIR}")


if __name__ == "__main__":
    generator = VectorIndexGenerator()
    generator.run_all()