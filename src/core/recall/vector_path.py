import faiss
import json
import sqlite3
import time
import numpy as np
from config import ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH, DB_PATH


class VectorPath:
    """
    向量路召回：实现基于 SBERT 的语义召回
    职责：语义向量 -> 相似文献 -> 作者 ID 列表
    """

    def __init__(self, recall_limit=100):
        # 搜索深度设为 500 以应对作者去重后的减员，确保能填满 100 人
        self.search_k = 500
        self.recall_limit = recall_limit

        # 初始化加载资源
        self.index = faiss.read_index(ABSTRACT_INDEX_PATH)
        with open(ABSTRACT_MAP_PATH, 'r', encoding='utf-8') as f:
            self.id_map = json.load(f)

    def recall(self, query_vector):
        """
        执行召回并返回 ID 列表
        """
        # 1. Faiss 检索
        _, indices = self.index.search(query_vector, self.search_k)

        # 2. 映射 Work IDs (基于行号索引)
        work_ids = [
            self.id_map[idx]
            for idx in indices[0]
            if 0 <= idx < len(self.id_map)
        ]

        if not work_ids:
            return []

        # 3. SQLite 关联映射
        conn = sqlite3.connect(DB_PATH)
        try:
            placeholders = ','.join(['?'] * len(work_ids))
            # 使用 DISTINCT 确保 ID 不重复
            query = f"SELECT DISTINCT author_id FROM authorships WHERE work_id IN ({placeholders})"
            cursor = conn.execute(query, work_ids)
            # 提取第一列 ID
            author_ids = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

        # 4. 返回精确长度的 ID 列表
        return author_ids[:self.recall_limit]


if __name__ == "__main__":
    v_path = VectorPath(recall_limit=100)

    print("\n" + "=" * 60)
    print("🚀 向量路测试：直接输出 100 个 Author ID")
    print("=" * 60)

    try:
        while True:
            raw_input = input("\n请粘贴向量数据 (list格式) 或输入 'q' 退出:\n>> ").strip()
            if raw_input.lower() in ['q', 'exit']: break
            if not raw_input: continue

            try:
                # 预处理输入向量
                vector_list = json.loads(raw_input)
                query_vec = np.array([vector_list]).astype('float32')
                faiss.normalize_L2(query_vec)

                # 执行召回获取 ID 列表
                start_t = time.time()
                id_list = v_path.recall(query_vec)
                duration = (time.time() - start_t) * 1000

                # 最终输出展示
                print(f"\n[召回结果 - 耗时 {duration:.2f}ms]")
                print(f"ID 列表长度: {len(id_list)}")
                print("-" * 30)
                print(id_list)  # 打印完整的 100 个 ID 列表
                print("-" * 30)

            except Exception as e:
                print(f"[错误] 处理失败: {e}")

    except KeyboardInterrupt:
        print("\n已退出")