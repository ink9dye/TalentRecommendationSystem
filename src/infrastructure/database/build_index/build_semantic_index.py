import sqlite3
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util
from database import DatabaseManager


class SemanticIndexEngine:
    def __init__(self, db_path, model_path):
        self.db = DatabaseManager(db_path)
        # 加载本地 384 维 SBERT 模型
        self.model = SentenceTransformer(model_path)

    def update_vocabulary_embeddings(self):
        """第一步：为 vocabulary 表中所有词汇生成并存储向量"""
        with self.db.connection() as conn:
            # 提取尚未向量化的词汇
            rows = conn.execute("SELECT id, term FROM vocabulary WHERE vector IS NULL").fetchall()

        if not rows: return

        print(f"正在为 {len(rows)} 个词汇生成 384 维向量...")
        ids = [r[0] for r in rows]
        terms = [r[1] for r in rows]
        embeddings = self.model.encode(terms, convert_to_tensor=False)

        with self.db.connection() as conn:
            for i, vector in zip(ids, embeddings):
                # 以二进制 BLOB 形式存入 SQLite
                conn.execute("UPDATE vocabulary SET vector = ? WHERE id = ?",
                             (sqlite3.Binary(pickle.dumps(vector)), i))
        print("向量预计算完成。")

    def generate_top_k_edges(self, k=3):
        """
        第二步：实现你的核心逻辑——每个节点至少建立 3 条语义关联边
        将“工业侧词汇”连接到最相似的 3 个“学术侧词汇”
        """
        with self.db.connection() as conn:
            # 分别提取工业侧和学术侧的词汇及向量
            industry_rows = conn.execute(
                "SELECT term, vector FROM vocabulary WHERE entity_type = 'industry'").fetchall()
            academic_rows = conn.execute(
                "SELECT term, vector FROM vocabulary WHERE entity_type IN ('concept', 'keyword')").fetchall()

        if not industry_rows or not academic_rows:
            print("数据不足，无法构建关联。")
            return []

        # 解析向量
        ind_terms = [r[0] for r in industry_rows]
        ind_vecs = np.array([pickle.loads(r[1]) for r in industry_rows])

        aca_terms = [r[0] for r in academic_rows]
        aca_vecs = np.array([pickle.loads(r[1]) for r in academic_rows])

        # 计算相似度矩阵 (Cosine Similarity)
        # 结果维度: [工业词数, 学术词数]
        sim_matrix = util.cos_sim(ind_vecs, aca_vecs).numpy()

        semantic_edges = []
        print(f"正在为每个应用词计算 Top-{k} 学术关联...")

        for i, skill in enumerate(ind_terms):
            # 获取当前技能词对应的所有学术词相似度排名
            # argsort 会按分值从小到大排，我们取最后 k 个
            top_indices = sim_matrix[i].argsort()[-k:][::-1]

            for idx in top_indices:
                score = sim_matrix[i][idx]
                target_keyword = aca_terms[idx]

                # 记录关联：[来源词, 目标学术词, 相似度分值]
                semantic_edges.append({
                    "source": skill,
                    "target": target_keyword,
                    "weight": round(float(score), 4)
                })

        return semantic_edges


# --- 执行合并与索引构建 ---
if __name__ == "__main__":
    DB_PATH = 'TalentSystem.db'
    MODEL_PATH = r'/src/infrastructure/database/models\sbert_model'

    engine = SemanticIndexEngine(DB_PATH, MODEL_PATH)

    # 1. 完善 SQLite 中的向量字段
    engine.update_vocabulary_embeddings()

    # 2. 生成 Top-3 关联边
    edges = engine.generate_top_k_edges(k=3)

    # 3. 打印示例预览
    print("\n--- 语义边生成示例 (Top-3) ---")
    for edge in edges[:6]:
        print(f"[{edge['source']}] --({edge['weight']})--> [{edge['target']}]")