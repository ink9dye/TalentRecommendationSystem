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
    标签路召回优化版：引入关键词分级排名、命中数量平方奖励及向量空间内聚度纠偏
    """

    def __init__(self, recall_limit=100):
        self.recall_limit = recall_limit
        try:
            self.graph = Graph(
                CONFIG_DICT["NEO4J_URI"],
                auth=(CONFIG_DICT["NEO4J_USER"], CONFIG_DICT["NEO4J_PASSWORD"]),
                name=CONFIG_DICT["NEO4J_DATABASE"]
            )
            self.graph.run("RETURN 1").evaluate()
            print("[OK] Neo4j 连接成功")
        except Exception as e:
            print(f"[Error] Neo4j 连接失败: {e}")

        # 1. 加载岗位 Faiss 索引 (用于需求锚定)
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_id_map = json.load(f)

        # 2. 加载学术词 Faiss 索引 (用于计算词汇间的向量相似度/内聚度)
        self.vocab_index = faiss.read_index(VOCAB_INDEX_PATH)
        with open(VOCAB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.vocab_id_map = json.load(f)
            # 建立 ID 到索引位置的映射，方便快速提取向量
            self.vocab_to_idx = {str(vid): i for i, vid in enumerate(self.vocab_id_map)}

    def _get_skills_from_jobs(self, query_vector, top_n=5):
        """步骤 1 & 2: 锚定岗位 -> 提取岗位直接关联的核心学术词汇"""
        _, indices = self.job_index.search(query_vector, top_n)
        job_ids = [self.job_id_map[idx] for idx in indices[0] if 0 <= idx < len(self.job_id_map)]

        cypher = """
        MATCH (j:Job) WHERE j.id IN $j_ids
        MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
        RETURN v.id AS vid, v.term AS term
        """
        res = self.graph.run(cypher, j_ids=job_ids).to_data_frame()
        if res.empty:
            return []

        print(f" [链路跟踪] 锚定岗位提取核心学术词: {res['term'].unique().tolist()[:5]}")
        return [int(v) for v in res['vid'].unique().tolist()]

    def _calculate_proximity(self, vocab_ids):
        """三维评分-维度C：利用向量索引计算命中心学术词之间的平均相似度(内聚度)"""
        if len(vocab_ids) < 2:
            return 0.5  # 孤立词给予中等基础系数加成

        # 获取 Faiss 索引中的位置
        idxs = [self.vocab_to_idx[str(vid)] for vid in vocab_ids if str(vid) in self.vocab_to_idx]
        if len(idxs) < 2:
            return 0.5

        # 提取向量 (build_vector_index.py 中已做 L2 归一化)
        vecs = np.array([self.vocab_index.reconstruct(i) for i in idxs])

        # 计算余弦相似度矩阵 (内积即余弦)
        sim_matrix = np.dot(vecs, vecs.T)

        # 取上三角（不含对角线）的平均值作为内聚分
        n = sim_matrix.shape[0]
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        return float(np.mean(upper_tri))

    def recall(self, query_vector):
        if self.graph is None:
            return [], 0

        # 1. 提取岗位直出的核心学术词 (核心排名分: 100)
        core_vocab_ids = self._get_skills_from_jobs(query_vector)
        if not core_vocab_ids:
            return [], 0

        # 2. 联想扩展词 (联想排名分: 20)
        expand_cypher = """
        MATCH (v:Vocabulary) WHERE v.id IN $v_ids
        OPTIONAL MATCH (v)-[:SIMILAR_TO]-(v_rel:Vocabulary)
        WITH v_rel LIMIT 100
        RETURN DISTINCT v_rel.id AS vid
        """
        expanded_res = self.graph.run(expand_cypher, v_ids=core_vocab_ids).to_data_frame()
        ext_vocab_ids = [int(v) for v in expanded_res['vid'].tolist() if v is not None and int(v) not in core_vocab_ids]

        all_vocab_ids = list(set(core_vocab_ids + ext_vocab_ids))

        # 3. Neo4j 初步检索：获取论文及其命中的原始 ID 集合
        final_cypher = """
        MATCH (v:Vocabulary) 
        USING INDEX v:Vocabulary(id)
        WHERE v.id IN $all_v_ids
        MATCH (v)<-[:HAS_TOPIC]-(w:Work)

        // 关键性能优化：采样剪枝并带回命中词详细信息用于后期计算
        WITH w, collect(DISTINCT v.id) AS hit_ids, collect(DISTINCT v.term) AS hit_terms LIMIT 2000

        MATCH (w)<-[auth_r:AUTHORED]-(a:Author)
        RETURN a.id AS aid, 
               sum(auth_r.pos_weight) AS base_auth_weight,
               collect(DISTINCT {wid: w.id, title: w.title, hit_ids: hit_ids, hit_terms: hit_terms}) AS work_samples
        """

        start_t = time.time()
        res = self.graph.run(final_cypher, all_v_ids=all_vocab_ids).to_data_frame()
        if res.empty:
            return [], (time.time() - start_t) * 1000

        # 4. 执行综合三维评分排名
        scored_authors = []
        for _, row in res.iterrows():
            author_total_score = 0.0
            max_hit_count = 0
            avg_proximity = 0.0

            for work in row['work_samples']:
                # 维度 A：排名权重 (核心词 100, 扩展词 20)
                rank_score = sum([100 if vid in core_vocab_ids else 20 for vid in work['hit_ids']])

                # 维度 B：数量奖励 (平方效应)
                hit_count = len(work['hit_ids'])
                qty_bonus = hit_count ** 2

                # 维度 C：向量近邻相似度加成
                proximity = self._calculate_proximity(work['hit_ids'])

                # 单篇论文综合分 = (排名基础分 + 数量奖励) * (1 + 相似度系数)
                work_score = (rank_score + qty_bonus) * (1 + proximity)
                author_total_score += work_score

                max_hit_count = max(max_hit_count, hit_count)
                avg_proximity += proximity

            # 最终得分 = 作者署名权重 * 累计论文表现
            final_score = row['base_auth_weight'] * author_total_score

            scored_authors.append({
                'aid': row['aid'],
                'score': final_score,
                'hits': max_hit_count,
                'prox': avg_proximity / len(row['work_samples']),
                'title': row['work_samples'][0]['title']
            })

        # 排序并取 Top 100
        scored_authors.sort(key=lambda x: x['score'], reverse=True)
        duration = (time.time() - start_t) * 1000

        # 5. 打印精准召回报告
        print(f"\n[综合三维排名报告 - 纠偏模式结果]")
        for i, auth in enumerate(scored_authors[:5]):
            print(f" {i + 1}. 作者: {auth['aid']} | 综合得分: {auth['score']:.2f}")
            print(f"    最高命中词数: {auth['hits']} | 向量近邻度(内聚性): {auth['prox']:.4f}")
            print(f"    代表作: {auth['title'][:60]}...")

        return [a['aid'] for a in scored_authors[:self.recall_limit]], duration


if __name__ == "__main__":
    l_path = LabelRecallPath(recall_limit=100)
    print("\n" + "=" * 60 + "\n🚀 标签路优化测试 (三维评分排名版)\n" + "=" * 60)

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
                print(f"- 召回候选人总数: {len(top_ids)}")
                print("-" * 30)
            except Exception as e:
                import traceback

                traceback.print_exc()
    except KeyboardInterrupt:
        pass