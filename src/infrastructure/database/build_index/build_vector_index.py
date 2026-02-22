import sqlite3
import json
import os
import sys
import numpy as np
import faiss
import logging
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
        print(f"[*] 正在加载 SBERT 模型: {MODEL_NAME}")
        print(f"[*] 模型存储位置: {SBERT_DIR}")
        self.model = SentenceTransformer(MODEL_NAME, cache_folder=SBERT_DIR)

        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row

        if not os.path.exists(INDEX_DIR):
            os.makedirs(INDEX_DIR)

    def _save_index(self, name, embeddings, ids, index_path, map_path, use_id_map=False):
        """
        双模式索引保存（增强版）：
        在保存 .faiss 索引的同时，备份原始向量为 .npy 文件，
        以解决后续召回阶段 reconstruct not implemented 的限制。
        """
        dimension = embeddings.shape[1]

        # 1. 创建底层 HNSW 索引核心
        # M=32, efConstruction=200 保证了 55w 规模下的检索精度与速度
        sub_index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        sub_index.hnsw.efConstruction = 200

        # 2. 归一化以支持余弦相似度
        faiss.normalize_L2(embeddings)

        # --- 【核心新增】：保存原始向量快照 ---
        # 路径规则：将 .faiss 后缀替换为 _vectors.npy
        vec_save_path = index_path.replace('.faiss', '_vectors.npy')
        np.save(vec_save_path, embeddings)
        print(f"[*] 原始向量已备份至: {os.path.basename(vec_save_path)}")
        # ------------------------------------

        if use_id_map:
            # --- 模式 A: IndexIDMap (Vocabulary 专用) ---
            # 这种结构在搜索时极快，但无法通过索引反推原始向量
            index = faiss.IndexIDMap(sub_index)
            ids_np = np.array(ids).astype('int64')

            print(f"[*] 正在将向量写入 {name} 的 HNSW 原生 ID 索引...")
            index.add_with_ids(embeddings, ids_np)

            with open(map_path, 'w', encoding='utf-8') as f:
                json.dump({"info": "Native IDMap used for continuous integer IDs"}, f)
        else:
            # --- 模式 B: 物理索引 + JSON Map (Abstracts/Jobs 专用) ---
            print(f"[*] 正在将向量写入 {name} 的 HNSW 物理索引...")
            sub_index.add(embeddings)
            index = sub_index

            with open(map_path, 'w', encoding='utf-8') as f:
                json.dump(ids, f, ensure_ascii=False, indent=4)

        # 3. 保存二进制索引文件
        faiss.write_index(index, index_path)
        print(f"[成功] 索引已保存: {os.path.basename(index_path)} | 记录数: {len(ids)}")

    def build_vocabulary_index(self):
        """1. 词汇索引 - 适配连续整数 ID"""
        print("\n>>> 任务 1: 构建词汇向量索引 (Vocabulary)")
        cursor = self.conn.cursor()
        rows = cursor.execute("SELECT voc_id, term FROM vocabulary").fetchall()

        texts = [row['term'] for row in rows]
        ids = [int(row['voc_id']) for row in rows]

        if not texts: return

        print(f"正在编码 {len(texts)} 个词汇...")
        # 加上进度条
        embeddings = self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

        self._save_index(
            "vocabulary",
            np.array(embeddings).astype('float32'),
            ids,
            VOCAB_INDEX_PATH,
            VOCAB_MAP_PATH,
            use_id_map=True
        )

    def build_abstract_index(self):
        """2. 摘要索引 - CPU 多核提速 + 解决假死版"""
        print("\n>>> 任务 2: 构建论文摘要向量索引 (Abstracts)")
        cursor = self.conn.cursor()

        # 添加读取提示，55万数据读取会有几秒延迟
        print("[*] 正在从 SQLite 读取摘要数据（55万条规模，请稍候）...")
        rows = cursor.execute("""
                              SELECT work_id, full_text_en
                              FROM abstracts
                              WHERE full_text_en IS NOT NULL
                                AND full_text_en != ''
                              """).fetchall()

        texts = [row['full_text_en'] for row in rows]
        ids = [str(row['work_id']) for row in rows]

        if not texts:
            print("警告: 未发现可用的摘要文本。")
            return

        print(f"[OK] 数据加载成功，共计 {len(texts)} 条，准备开始多核编码...")

        # --- 核心改进：带进度条的多进程编码 ---
        try:
            pool = self.model.start_multi_process_pool()
            # 加上 show_progress_bar=True 以确保能看到 Batches 进度
            embeddings = self.model.encode(
                texts,
                pool=pool,
                batch_size=BATCH_SIZE,
                show_progress_bar=True
            )
            self.model.stop_multi_process_pool(pool)
        except Exception as e:
            print(f"[!] 多进程启动异常 ({e})，正在回退到单进程编码...")
            embeddings = self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

        self._save_index(
            "abstract",
            embeddings.astype('float32'),
            ids,
            ABSTRACT_INDEX_PATH,
            ABSTRACT_MAP_PATH
        )

    def build_job_description_index(self):
        """3. 岗位描述索引"""
        print("\n>>> 任务 3: 构建岗位描述向量索引 (Jobs)")
        cursor = self.conn.cursor()
        rows = cursor.execute("SELECT securityId, job_name, description FROM jobs").fetchall()

        texts = [f"{row['job_name']} {row['description']}" for row in rows]
        ids = [str(row['securityId']) for row in rows]

        if not texts: return

        print(f"正在编码 {len(texts)} 个岗位描述...")
        # 加上进度条
        embeddings = self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
        self._save_index("job_description", np.array(embeddings).astype('float32'), ids, JOB_INDEX_PATH, JOB_MAP_PATH)

    def run_all(self):
        # 如果任务 1 已经跑完，可以在这里注释掉以节省时间
        # self.build_vocabulary_index()
        self.build_abstract_index()
        self.build_job_description_index()
        print("\n[所有索引构建完成]")
        print(f"模型存放于: {SBERT_DIR}")
        print(f"索引存放于: {INDEX_DIR}")


if __name__ == "__main__":
    # Windows 环境下必须在 __main__ 下运行生成器，防止多进程报错
    generator = VectorIndexGenerator()

    # 运行词汇索引任务 ---

    # generator.build_vocabulary_index()

    # --- 如果你还需要更新岗位索引，可以取消下面这行的注释 ---
    # generator.build_job_description_index()

    # --- 暂时注释掉 run_all()，防止误触发全量更新 ---
    # generator.run_all()

