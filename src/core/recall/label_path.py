import faiss
import json
import time
import numpy as np
from py2neo import Graph
from config import CONFIG_DICT, JOB_INDEX_PATH, JOB_MAP_PATH


class LabelRecallPath:
    """
    标签路召回优化版：通过路径剪枝与索引强制约束提升检索性能
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

        # 预加载 Faiss 索引
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_id_map = json.load(f)

    def _anchor_job_to_vocabs(self, query_vector, top_n=5):
        """步骤 1: 向量空间 -> 岗位 ID -> 技能词 ID"""
        _, indices = self.job_index.search(query_vector, top_n)
        job_ids = [self.job_id_map[idx] for idx in indices[0] if 0 <= idx < len(self.job_id_map)]

        # 优化查询：j.id 匹配
        cypher = """
        MATCH (j:Job) WHERE j.id IN $j_ids
        MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
        RETURN v.id AS vid
        """
        res = self.graph.run(cypher, j_ids=job_ids).to_data_frame()
        if res.empty: return []
        return [int(v) for v in res['vid'].unique().tolist()]

    def recall(self, query_vector):
        if self.graph is None: return [], 0

        # 1. 获取核心词汇
        core_vocab_ids = self._anchor_job_to_vocabs(query_vector)
        if not core_vocab_ids: return [], 0

        # 2. 语义桥接扩展 (限制扩展数量防止组合爆炸)
        expand_cypher = """
        MATCH (v:Vocabulary) WHERE v.id IN $v_ids
        OPTIONAL MATCH (v)-[:SIMILAR_TO]-(v_rel:Vocabulary)
        WITH v, v_rel LIMIT 30
        RETURN DISTINCT v_rel.id AS vid
        """
        expanded_res = self.graph.run(expand_cypher, v_ids=core_vocab_ids).to_data_frame()

        # 合并词汇 ID
        all_vocab_ids = core_vocab_ids
        if not expanded_res.empty:
            all_vocab_ids += [int(v) for v in expanded_res['vid'].tolist() if v is not None]
        all_vocab_ids = list(set(all_vocab_ids))

        # 3. 最终召回：引入路径剪枝 (LIMIT 1000) 与 署名权重聚合
        #
        final_cypher = """
        MATCH (v:Vocabulary) 
        USING INDEX v:Vocabulary(id)
        WHERE v.id IN $v_ids

        // 关键性能优化：限制每个标签关联的 Work 采样量，避免处理数万个关系
        MATCH (v)<-[:HAS_TOPIC]-(w:Work)
        WITH w LIMIT 2000 

        MATCH (w)<-[r:AUTHORED]-(a:Author)
        // 计算基于署名排名和时序衰减的 pos_weight 总分
        RETURN a.id AS aid, sum(r.pos_weight) AS score
        ORDER BY score DESC LIMIT $limit
        """

        start_t = time.time()
        res = self.graph.run(final_cypher, v_ids=all_vocab_ids, limit=self.recall_limit).to_data_frame()
        duration = (time.time() - start_t) * 1000

        if res.empty: return [], duration

        # 返回 Top 100 的纯 ID 列表
        return res['aid'].tolist(), duration


if __name__ == "__main__":
    l_path = LabelRecallPath(recall_limit=100)
    print("\n" + "=" * 60 + "\n🚀 标签路优化测试 (目标：<1000ms)\n" + "=" * 60)

    try:
        while True:
            raw_input = input("\n请粘贴稠密向量:\n>> ").strip()
            if not raw_input or raw_input.lower() == 'q': break

            try:
                vector_list = json.loads(raw_input)
                query_vec = np.array([vector_list]).astype('float32')
                faiss.normalize_L2(query_vec)

                # 执行召回
                top_ids, search_time = l_path.recall(query_vec)

                print(f"\n[召回报告]")
                print(f"- 优化后推理耗时: {search_time:.2f} ms")
                print(f"- 召回候选人数: {len(top_ids)}")
                print("-" * 30)
                print("【Top 100 作者 ID 顺位列表】:")
                print(top_ids)
                print("-" * 30)

            except Exception as e:
                print(f"[Error] {e}")
    except KeyboardInterrupt:
        pass