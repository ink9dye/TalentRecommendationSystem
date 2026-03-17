import sqlite3
import json
import os
import sys
import numpy as np
import faiss
import torch
import gc
import math
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 1. 解决控制台输出乱码
if sys.platform.startswith('win'):
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 2. 导入统一配置
from config import (
    DB_PATH, DATA_DIR, INDEX_DIR, SBERT_DIR, SBERT_MODEL_NAME,
    VOCAB_INDEX_PATH, VOCAB_MAP_PATH,
    ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH,
    JOB_INDEX_PATH, JOB_MAP_PATH
)

# --- 针对 i5-1240P 的终极稳定配置 ---
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# [必须保留] 开启 PyTorch 底层的 Intel MKL-DNN 加速
torch.backends.mkldnn.enabled = True
# 回归单进程，由底层 MKL 统一调度 8 个线程，告别内存总线互抢
torch.set_num_threads(8)

MODEL_NAME = SBERT_MODEL_NAME
BATCH_SIZE = 16  # 单进程下 16 是最甜点的配置
SHARD_SIZE = 10000  # 1万条一个分片，极速且安全
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


        self.model.max_seq_length = 1024
        self.model.eval()

        print("[*] 正在进行模型预热...")
        with torch.no_grad():
            self.model.encode(["warm up text"], batch_size=1)
        print("[+] 预热完成。")

        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row

        for d in [INDEX_DIR, TEMP_SHARD_DIR]:
            if not os.path.exists(d): os.makedirs(d)

    def _smart_trim(self, text):
        if not text: return ""
        if len(text) > 4000:
            return text[:2000] + " " + text[-2000:]
        return text

    def _save_index(self, name, embeddings, ids, index_path, map_path, use_id_map=False):
        dimension = embeddings.shape[1]
        sub_index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        sub_index.hnsw.efConstruction = 200
        faiss.normalize_L2(embeddings)

        vec_save_path = index_path.replace('.faiss', '_vectors.npy')
        np.save(vec_save_path, embeddings)

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
        print(f"[成功] {name} 索引已保存。")

    def _load_term_to_abbreviation_map(self):
        """从 data/industrial_abbr_expansion.json 构建 归一化全称 -> [缩写] 的反向映射，供 Alias Embedding 使用。"""
        path = os.path.join(DATA_DIR, "industrial_abbr_expansion.json")
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            abbr_to_expansions = json.load(f)
        term_to_abbrs = {}
        for abbr, expansions in abbr_to_expansions.items():
            if not isinstance(expansions, list):
                expansions = [expansions]
            for exp in expansions:
                key = exp.strip().lower().replace("-", " ")
                if key not in term_to_abbrs:
                    term_to_abbrs[key] = []
                if abbr not in term_to_abbrs[key]:
                    term_to_abbrs[key].append(abbr)
        return term_to_abbrs

    def build_vocabulary_index(self):
        print("\n>>> 任务 1: 构建词汇表向量索引 (Vocabulary)")
        cursor = self.conn.cursor()
        try:
            rows = cursor.execute(
                "SELECT voc_id, term FROM vocabulary WHERE term IS NOT NULL AND term != ''").fetchall()
            # Alias Embedding：有缩写时用 "term | abbr"，否则仍用 term（README：向量索引支持缩写与全称双匹配）
            term_to_abbrs = self._load_term_to_abbreviation_map()
            texts = []
            for row in rows:
                term = row['term']
                key = term.strip().lower().replace("-", " ")
                abbrs = term_to_abbrs.get(key, [])
                if abbrs:
                    texts.append(f"{term} | {' | '.join(abbrs)}")
                else:
                    texts.append(term)
            ids = [int(row['voc_id']) for row in rows]
            if not texts:
                return

            with torch.no_grad():
                embeddings = self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
            self._save_index(
                "vocabulary",
                np.array(embeddings).astype('float32'),
                ids,
                VOCAB_INDEX_PATH,
                VOCAB_MAP_PATH,
                use_id_map=True
            )
        except Exception as e:
            print(f"[!] 词汇表建索异常: {e}")

    def build_job_description_index(self):
        print("\n>>> 任务 2: 构建岗位描述向量索引 (Jobs)")
        cursor = self.conn.cursor()
        rows = cursor.execute("SELECT securityId, job_name, description FROM jobs").fetchall()
        texts = [self._smart_trim(f"{row['job_name']} {row['description']}") for row in rows]
        ids = [str(row['securityId']) for row in rows]
        if not texts: return

        with torch.no_grad():
            embeddings = self.model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
        self._save_index("job_description", np.array(embeddings).astype('float32'), ids, JOB_INDEX_PATH, JOB_MAP_PATH)

    def build_abstract_index(self):
        print("\n>>> 任务 3: 构建论文摘要向量索引 (单进程全量切片计算)")
        cursor = self.conn.cursor()

        # 【核心救命修改】：一次性把数据拉到内存，彻底消灭 OFFSET 慢查询
        print("[*] 正在拉取全量数据库记录到内存...")
        cursor.execute(
            "SELECT work_id, full_text_en FROM abstracts WHERE full_text_en IS NOT NULL AND full_text_en != ''")
        all_rows = cursor.fetchall()

        total_count = len(all_rows)
        if total_count == 0: return

        num_shards = math.ceil(total_count / SHARD_SIZE)
        print(f"[*] 摘要总条数: {total_count}，总分片数: {num_shards} (每分片1万条)")

        for i in range(num_shards):
            shard_vec_path = os.path.join(TEMP_SHARD_DIR, f"shard_{i}.npy")
            shard_id_path = os.path.join(TEMP_SHARD_DIR, f"shard_{i}_ids.json")

            if os.path.exists(shard_vec_path):
                print(f"[-] 分片 {i + 1}/{num_shards} 已存在，跳过。")
                continue

            start_offset = i * SHARD_SIZE
            end_offset = min((i + 1) * SHARD_SIZE, total_count)
            print(f"\n[*] 正在处理分片 {i + 1}/{num_shards} (第 {start_offset} 到 {end_offset} 条)")

            # 纯内存切片，微秒级极速
            shard_rows = all_rows[start_offset:end_offset]
            texts = [self._smart_trim(r['full_text_en']) for r in shard_rows]
            ids = [str(r['work_id']) for r in shard_rows]

            with torch.no_grad():
                # 单进程下进度条不会乱跳，安心开启
                embeddings = self.model.encode(
                    texts,
                    batch_size=BATCH_SIZE,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )

            np.save(shard_vec_path, embeddings.astype('float32'))
            with open(shard_id_path, 'w', encoding='utf-8') as f:
                json.dump(ids, f)

            del texts, ids, embeddings, shard_rows
            gc.collect()

        del all_rows
        gc.collect()
        self._merge_abstract_shards(num_shards)

    def _merge_abstract_shards(self, num_shards):
        print("\n[*] 正在执行全量合并...")
        all_vecs, all_ids = [], []
        for i in tqdm(range(num_shards), desc="合并分片"):
            vec_path = os.path.join(TEMP_SHARD_DIR, f"shard_{i}.npy")
            id_path = os.path.join(TEMP_SHARD_DIR, f"shard_{i}_ids.json")
            if not os.path.exists(vec_path): continue
            all_vecs.append(np.load(vec_path))
            with open(id_path, 'r', encoding='utf-8') as f:
                all_ids.extend(json.load(f))

        if all_vecs:
            self._save_index("abstract", np.vstack(all_vecs), all_ids, ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH)
            print("\n[!!!] 论文摘要全量索引构建圆满完成！")

    def run_all(self):
        self.build_vocabulary_index()
        # # self.build_job_description_index()
        # self.build_abstract_index()


if __name__ == "__main__":
    generator = StableVectorGenerator()
    generator.run_all()