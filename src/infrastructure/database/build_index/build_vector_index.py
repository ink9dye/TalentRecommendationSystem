import sqlite3
import json
import os
import sys
import numpy as np
import faiss
import torch
import gc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 1. 解决控制台输出乱码
if sys.platform.startswith('win'):
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 2. 导入统一配置
from config import (
    DB_PATH, INDEX_DIR, SBERT_DIR, SBERT_MODEL_NAME,
    VOCAB_INDEX_PATH, VOCAB_MAP_PATH,
    ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH,
    JOB_INDEX_PATH, JOB_MAP_PATH
)

# --- 针对 i5-1240P 的终极稳定配置 ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 12代 i5 建议使用 8 线程，平衡 P-Core 性能与系统开销
torch.set_num_threads(8)

MODEL_NAME = SBERT_MODEL_NAME
BATCH_SIZE = 16  # 1024 长度下，建议保持在 16 以免内存抖动
SHARD_SIZE = 50000  # 5万条一个分片
TEMP_SHARD_DIR = os.path.join(INDEX_DIR, "shards")


class StableVectorGenerator:
    def __init__(self):
        print(f"[*] 正在加载本地 SBERT 模型: {MODEL_NAME}")
        self.model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=SBERT_DIR,
            trust_remote_code=True,
            device="cpu"
        )

        # 【关键：精度平衡点】
        # 基于 SQL 探测结果，1024 可覆盖 99.45% 的论文数据
        self.model.max_seq_length = 1024
        self.model.eval()

        print("[*] 正在进行模型预热...")
        self.model.encode(["warm up text"], batch_size=1)
        print("[+] 预热完成，准备进入正式编码环节。")

        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row

        for d in [INDEX_DIR, TEMP_SHARD_DIR]:
            if not os.path.exists(d): os.makedirs(d)

    def _smart_trim(self, text):
        """
        针对超长文本的智能清洗策略：
        如果超过 4000 字符，保留前 2000 字和后 2000 字，
        通过“头尾拼接”最大程度保留论文的背景和结论信息。
        """
        if not text: return ""
        if len(text) > 4000:
            # 拼接头部和尾部，中间用空格隔开
            return text[:2000] + " " + text[-2000:]
        return text

    def _save_index(self, name, embeddings, ids, index_path, map_path, use_id_map=False):
        """通用索引保存逻辑（带向量备份）"""
        dimension = embeddings.shape[1]
        sub_index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        sub_index.hnsw.efConstruction = 200
        faiss.normalize_L2(embeddings)

        # 备份原始向量
        vec_save_path = index_path.replace('.faiss', '_vectors.npy')
        np.save(vec_save_path, embeddings)
        print(f"[*] 原始向量已备份: {os.path.basename(vec_save_path)}")

        if use_id_map:
            index = faiss.IndexIDMap(sub_index)
            index.add_with_ids(embeddings, np.array(ids).astype('int64'))
            with open(map_path, 'w', encoding='utf-8') as f:
                json.dump({"info": "Native IDMap used"}, f)
        else:
            sub_index.add(embeddings)
            index = sub_index
            with open(map_path, 'w', encoding='utf-8') as f:
                json.dump(ids, f, ensure_ascii=False, indent=4)

        faiss.write_index(index, index_path)
        print(f"[成功] {name} 索引已保存。记录数: {len(ids)}")

    def build_job_description_index(self):
        """2. 岗位描述索引 (Jobs)"""
        print("\n>>> 任务 1: 构建岗位描述向量索引 (Jobs)")
        cursor = self.conn.cursor()
        rows = cursor.execute("SELECT securityId, job_name, description FROM jobs").fetchall()

        # 应用智能清洗
        texts = [self._smart_trim(f"{row['job_name']} {row['description']}") for row in rows]
        ids = [str(row['securityId']) for row in rows]

        if not texts: return
        embeddings = self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
        self._save_index("job_description", np.array(embeddings).astype('float32'), ids,
                         JOB_INDEX_PATH, JOB_MAP_PATH)

    def build_abstract_index(self, start_shard=None, end_shard=None):
        """3. 摘要索引 (并行分片模式 - 支持指定范围)"""
        print("\n>>> 任务 2: 构建论文摘要向量索引 (并行模式)")
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM abstracts WHERE full_text_en IS NOT NULL AND full_text_en != ''")
        total_count = cursor.fetchone()[0]

        if total_count == 0: return
        num_shards = (total_count + SHARD_SIZE - 1) // SHARD_SIZE

        # --- 并行范围逻辑 ---
        # 如果未指定范围，默认处理全量分片
        s_idx = start_shard if start_shard is not None else 0
        e_idx = end_shard if end_shard is not None else num_shards
        print(f"[*] 任务分配: 处理分片 {s_idx + 1} 到 {e_idx} (总分片数: {num_shards})")

        # 只跑指定范围内的分片
        for i in range(s_idx, e_idx):
            shard_vec_path = os.path.join(TEMP_SHARD_DIR, f"shard_{i}.npy")
            shard_id_path = os.path.join(TEMP_SHARD_DIR, f"shard_{i}_ids.json")

            if os.path.exists(shard_vec_path):
                print(f"[-] 分片 {i + 1}/{num_shards} 已存在，跳过。")
                continue

            start_offset = i * SHARD_SIZE
            print(f"\n[*] [当前进程] 正在处理分片 {i + 1}/{num_shards} (Offset: {start_offset})")
            cursor.execute(f"""
                SELECT work_id, full_text_en FROM abstracts 
                WHERE full_text_en IS NOT NULL AND full_text_en != ''
                LIMIT {SHARD_SIZE} OFFSET {start_offset}
            """)
            shard_rows = cursor.fetchall()

            texts = [self._smart_trim(r['full_text_en']) for r in shard_rows]
            ids = [str(r['work_id']) for r in shard_rows]
            del shard_rows

            embeddings = self.model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            np.save(shard_vec_path, embeddings.astype('float32'))
            with open(shard_id_path, 'w', encoding='utf-8') as f:
                json.dump(ids, f)

            del texts, ids, embeddings
            gc.collect()

        # --- 安全合并检查 ---
        # 只有当磁盘上已经集齐了全部 12 个分片时，才执行最后的合并操作
        print(f"\n[*] 进程 {s_idx + 1}-{e_idx} 完成阶段性任务，正在检查全部分片是否就绪...")
        all_completed = all(os.path.exists(os.path.join(TEMP_SHARD_DIR, f"shard_{j}.npy")) for j in range(num_shards))

        if all_completed:
            self._merge_abstract_shards(num_shards)
        else:
            print(f"[!] 尚有其他分片未完成，请等待另一个窗口结束后自动或手动合并。")

    def _merge_abstract_shards(self, num_shards):
        """合并所有分片并构建最终 Faiss 索引"""
        print("\n[*] 检测到所有分片已完成，正在从磁盘执行全量合并...")
        all_vecs, all_ids = [], []
        for i in tqdm(range(num_shards), desc="读取分片"):
            vec_path = os.path.join(TEMP_SHARD_DIR, f"shard_{i}.npy")
            id_path = os.path.join(TEMP_SHARD_DIR, f"shard_{i}_ids.json")

            all_vecs.append(np.load(vec_path))
            with open(id_path, 'r', encoding='utf-8') as f:
                all_ids.extend(json.load(f))

        # 调用原有的通用保存函数
        self._save_index("abstract", np.vstack(all_vecs), all_ids, ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH)
        print("\n[!!!] 55万条论文摘要全量索引构建圆满完成！")

    def run_all(self):
        # 按照从易到难顺序执行
        # self.build_vocabulary_index()
        # self.build_job_description_index()
        self.build_abstract_index()


if __name__ == "__main__":
    generator = StableVectorGenerator()
    generator.run_all()