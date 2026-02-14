import faiss
import json
import time
import sqlite3
import numpy as np
from py2neo import Graph
from config import (
    CONFIG_DICT, JOB_INDEX_PATH, JOB_MAP_PATH,
    VOCAB_INDEX_PATH, VOCAB_MAP_PATH
)


class LabelRecallPath:
    """
    标签路召回极速版 (性能优化目标 < 500ms)
    逻辑：锚定岗位 -> 提取核心词 -> 向量预取 -> 三维评分排名
    """

    def __init__(self, recall_limit=100):
        self.recall_limit = recall_limit
        try:
            self.graph = Graph(
                CONFIG_DICT["NEO4J_URI"],
                auth=(CONFIG_DICT["NEO4J_USER"], CONFIG_DICT["NEO4J_PASSWORD"]),
                name=CONFIG_DICT["NEO4J_DATABASE"]
            )
            print("[OK] Neo4j 连接成功")
        except Exception as e:
            print(f"[Error] Neo4j 连接失败: {e}")

        # 1. 加载岗位索引
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_id_map = json.load(f)

        # 2. 加载词汇向量索引
        self.vocab_index = faiss.read_index(VOCAB_INDEX_PATH)
        with open(VOCAB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.vocab_id_map = json.load(f)
            self.vocab_to_idx = {str(vid): i for i, vid in enumerate(self.vocab_id_map)}

    def _get_skills_from_jobs(self, query_vector, top_n=5):
        """步骤 1 & 2: 提取核心学术词 (核心排名: 100)"""
        _, indices = self.job_index.search(query_vector, top_n)
        job_ids = [self.job_id_map[idx] for idx in indices[0] if 0 <= idx < len(self.job_id_map)]

        cypher = """
        MATCH (j:Job) WHERE j.id IN $j_ids
        MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
        RETURN v.id AS vid, v.term AS term
        """
        res = self.graph.run(cypher, j_ids=job_ids).to_data_frame()
        if res.empty: return []

        print(f" [链路跟踪] 提取核心学术词: {res['term'].unique().tolist()[:5]}")
        return [int(v) for v in res['vid'].unique().tolist()]

    def recall(self, query_vector):
        if self.graph is None: return [], 0
        start_t = time.time()

        # 1. 获取核心学术词
        core_ids = self._get_skills_from_jobs(query_vector)
        if not core_ids: return [], 0

        # 2. 联想词扩展
        expand_cypher = """
        MATCH (v:Vocabulary) WHERE v.id IN $v_ids
        OPTIONAL MATCH (v)-[:SIMILAR_TO]-(v_rel:Vocabulary)
        WITH v_rel LIMIT 100
        RETURN DISTINCT v_rel.id AS vid
        """
        expanded_res = self.graph.run(expand_cypher, v_ids=core_ids).to_data_frame()
        ext_ids = [int(v) for v in expanded_res['vid'].tolist() if v is not None and int(v) not in core_ids]

        all_ids = list(set(core_ids + ext_ids))

        # --- 性能优化：一次性预取所有相关向量，避免在循环中重复 I/O ---
        vocab_vec_cache = {}
        for vid in all_ids:
            s_vid = str(vid)
            if s_vid in self.vocab_to_idx:
                vocab_vec_cache[vid] = self.vocab_index.reconstruct(self.vocab_to_idx[s_vid])

        # 3. 极简 Neo4j 召回 (减轻数据库聚合负担)
        final_cypher = """
        MATCH (v:Vocabulary) 
        USING INDEX v:Vocabulary(id)
        WHERE v.id IN $all_v_ids
        MATCH (v)<-[:HAS_TOPIC]-(w:Work)
        WITH w, collect(DISTINCT v.id) AS hit_ids LIMIT 2000
        MATCH (w)<-[auth_r:AUTHORED]-(a:Author)
        RETURN a.id AS aid, 
               sum(auth_r.pos_weight) AS auth_w, 
               collect(DISTINCT {wid: w.id, title: w.title, hits: hit_ids}) AS papers
        """
        res = self.graph.run(final_cypher, all_v_ids=all_ids).to_data_frame()
        if res.empty: return [], (time.time() - start_t) * 1000

        # 4. 内存内高性能排名计算
        scored_authors = []
        for _, row in res.iterrows():
            author_score = 0.0
            best_hits = 0
            best_prox = 0.0

            for paper in row['papers']:
                h_ids = paper['hits']
                h_count = len(h_ids)

                # A. 关键词排名分 (100 vs 20)
                rank_score = sum([100 if vid in core_ids else 20 for vid in h_ids])

                # B. 向量内聚度 (使用缓存向量计算)
                proximity = 0.5
                if h_count >= 2:
                    paper_vecs = [vocab_vec_cache[vid] for vid in h_ids if vid in vocab_vec_cache]
                    if len(paper_vecs) >= 2:
                        sims = np.dot(paper_vecs, np.array(paper_vecs).T)
                        proximity = float(np.mean(sims[np.triu_indices(len(paper_vecs), k=1)]))

                # C. 单篇论文得分 = (基础分 + 数量平方) * (1 + 内聚度)
                paper_score = (rank_score + h_count ** 2) * (1 + proximity)
                author_score += paper_score

                best_hits = max(best_hits, h_count)
                best_prox = max(best_prox, proximity)

            scored_authors.append({
                'aid': row['aid'],
                'score': row['auth_w'] * author_score,
                'hits': best_hits,
                'prox': best_prox,
                'title': row['papers'][0]['title']
            })

        # 最终排序
        scored_authors.sort(key=lambda x: x['score'], reverse=True)
        duration = (time.time() - start_t) * 1000

        # 打印三维分析报告
        print(f"\n[综合三维排名报告 - TOP 5]")
        for i, auth in enumerate(scored_authors[:5]):
            print(f" {i + 1}. 作者: {auth['aid']} | 得分: {auth['score']:.2f}")
            print(f"    最高命中: {auth['hits']} | 向量内聚度: {auth['prox']:.4f} | 论文: {auth['title'][:50]}...")

        return [a['aid'] for a in scored_authors[:self.recall_limit]], duration


if __name__ == "__main__":
    l_path = LabelRecallPath(recall_limit=100)
    print("\n" + "=" * 60 + "\n🚀 标签路调试模式 (优化语法与采样性能)\n" + "=" * 60)

    try:
        while True:
            raw_input = input("\n请粘贴稠密向量:\n>> ").strip()
            if not raw_input or raw_input.lower() == 'q': break
            try:
                vector_list = json.loads(raw_input)
                query_vec = np.array([vector_list]).astype('float32')
                faiss.normalize_L2(query_vec)

                top_ids, search_time = l_path.recall(query_vec)

                print(f"\n[结果报告]")
                print(f"- 耗时: {search_time:.2f} ms")
                print(f"- 召回候选人数量: {len(top_ids)}")
                print("-" * 30)
            except Exception as e:
                import traceback

                traceback.print_exc()
    except KeyboardInterrupt:
        pass