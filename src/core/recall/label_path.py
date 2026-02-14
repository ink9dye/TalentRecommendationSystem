import faiss
import json
import time
import numpy as np
from py2neo import Graph
from config import (
    CONFIG_DICT, JOB_INDEX_PATH, JOB_MAP_PATH,
    VOCAB_INDEX_PATH, VOCAB_MAP_PATH
)


class LabelRecallPath:
    """
    标签路召回极限优化版
    核心修改：
    1. 移除 Pandas：使用 Neo4j 原生 Cursor 迭代，消除 DataFrame 序列化开销。
    2. 两段式评分：仅对前 100 名执行高能耗向量内聚度计算，余下执行快速路径。
    3. 预载入优化：维持向量内存矩阵以确保 Index-Only 级别的计算速度。
    """

    def __init__(self, recall_limit=200):
        self.recall_limit = recall_limit
        self.verbose = False
        try:
            self.graph = Graph(
                CONFIG_DICT["NEO4J_URI"],
                auth=(CONFIG_DICT["NEO4J_USER"], CONFIG_DICT["NEO4J_PASSWORD"]),
                name=CONFIG_DICT["NEO4J_DATABASE"]
            )
        except Exception as e:
            pass

        # 加载索引
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_id_map = json.load(f)

        self.vocab_index = faiss.read_index(VOCAB_INDEX_PATH)
        with open(VOCAB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.vocab_id_map = json.load(f)
            self.vocab_to_idx = {str(vid): i for i, vid in enumerate(self.vocab_id_map)}

        # 向量预载入矩阵
        print("[*] 正在预载入词汇向量矩阵以消除 I/O 瓶颈...", flush=True)
        ntotal = self.vocab_index.ntotal
        self.all_vocab_vectors = self.vocab_index.reconstruct_n(0, ntotal).astype('float32')
        print(f"[OK] 预载入 {ntotal} 个词汇向量完成")

    def _get_skills_from_jobs(self, query_vector, top_n=5):
        """原生 Cursor 模式获取核心词"""
        _, indices = self.job_index.search(query_vector, top_n)
        job_ids = [self.job_id_map[idx] for idx in indices[0] if 0 <= idx < len(self.job_id_map)]

        cypher = """
        MATCH (j:Job) WHERE j.id IN $j_ids
        MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
        RETURN DISTINCT v.id AS vid
        """
        # 使用 run().to_table() 或直接迭代比 to_data_frame() 快
        cursor = self.graph.run(cypher, j_ids=job_ids)
        return [int(record['vid']) for record in cursor]

    def _calculate_proximity_fast(self, hit_ids):
        """内聚度极速计算"""
        if len(hit_ids) < 2:
            return 0.5
        idxs = [self.vocab_to_idx[str(vid)] for vid in hit_ids if str(vid) in self.vocab_to_idx]
        if len(idxs) < 2:
            return 0.5
        vecs = self.all_vocab_vectors[idxs]
        sim_matrix = np.dot(vecs, vecs.T)
        n = sim_matrix.shape[0]
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        return float(np.mean(upper_tri))

    def recall(self, query_vector):
        if self.graph is None: return [], 0
        start_t = time.time()

        # 1. 获取核心词
        core_ids = self._get_skills_from_jobs(query_vector)
        if not core_ids: return [], 0
        core_ids_set = set(core_ids)

        # 2. 拓扑扩展 (原生迭代)
        expand_cypher = """
        MATCH (v:Vocabulary) WHERE v.id IN $v_ids
        OPTIONAL MATCH (v)-[:SIMILAR_TO]-(v_rel:Vocabulary)
        WITH v_rel LIMIT 100
        RETURN DISTINCT v_rel.id AS vid
        """
        ext_cursor = self.graph.run(expand_cypher, v_ids=core_ids)
        all_ids = list(core_ids_set | {int(rec['vid']) for rec in ext_cursor if rec['vid'] is not None})

        # 3. Neo4j 召回 (使用原生 Cursor 避免 DataFrame 开销)
        # 增加按照基础权重初步排序，方便后续执行两段式评分
        final_cypher = """
        MATCH (v:Vocabulary) 
        USING INDEX v:Vocabulary(id)
        WHERE v.id IN $all_v_ids
        MATCH (v)<-[:HAS_TOPIC]-(w:Work)
        WITH w, collect(DISTINCT v.id) AS hit_ids LIMIT 300 
        MATCH (w)<-[auth_r:AUTHORED]-(a:Author)
        RETURN a.id AS aid, 
               sum(auth_r.pos_weight) AS auth_w, 
               collect({wid: w.id, hits: hit_ids}) AS papers
        ORDER BY auth_w DESC
        """

        cursor = self.graph.run(final_cypher, all_v_ids=all_ids)

        # 4. 内存内分级评分排名
        scored_authors = []
        processed_count = 0

        # 直接迭代原生结果集
        for record in cursor:
            processed_count += 1
            author_total_score = 0.0

            # --- 优化点：两段式计算策略 ---
            # 仅对前 100 名高权重的作者执行矩阵运算，后续执行快速路径
            is_fast_path = processed_count > 100

            papers = record['papers']
            for paper in papers:
                h_ids = paper['hits']
                h_count = len(h_ids)
                rank_score = sum([100 if vid in core_ids_set else 20 for vid in h_ids])

                if is_fast_path:
                    proximity = 0.5  # 快速路径默认值
                else:
                    proximity = self._calculate_proximity_fast(h_ids)  # 精排路计算

                author_total_score += (rank_score + h_count ** 2) * (1 + proximity)

            scored_authors.append({
                'aid': record['aid'],
                'score': record['auth_w'] * author_total_score
            })

        # 最终排序
        scored_authors.sort(key=lambda x: x['score'], reverse=True)
        duration = (time.time() - start_t) * 1000

        if self.verbose:
            print(f"\n[原生迭代+两段式评分报告 - TOP 5]")
            for i, auth in enumerate(scored_authors[:5]):
                print(f" {i + 1}. 作者: {auth['aid']} | 得分: {auth['score']:.2f}")

        return [a['aid'] for a in scored_authors[:self.recall_limit]], duration


if __name__ == "__main__":
    l_path = LabelRecallPath(recall_limit=200)
    l_path.verbose = True

    try:
        while True:
            raw_input = input("\n请输入稠密向量 (JSON 格式):\n>> ").strip()
            if not raw_input or raw_input.lower() == 'q': break
            try:
                import json

                vector_list = json.loads(raw_input)
                query_vec = np.array([vector_list]).astype('float32')
                faiss.normalize_L2(query_vec)

                top_ids, search_time = l_path.recall(query_vec)

                print(f"\n[性能反馈]")
                print(f"- 标签路(原生加速版)耗时: {search_time:.2f} ms")
                print(f"- 召回候选人数: {len(top_ids)}")
                print("-" * 30)
            except Exception as e:
                import traceback

                traceback.print_exc()
    except KeyboardInterrupt:
        pass