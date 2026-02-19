import faiss
import json
import time
import numpy as np
import traceback
from py2neo import Graph
from config import (
    CONFIG_DICT, JOB_INDEX_PATH, JOB_MAP_PATH,
    VOCAB_INDEX_PATH, VOCAB_MAP_PATH
)


class LabelRecallPath:
    """
    标签路召回极限优化版 (作者贡献度分摊评分模型)

    核心逻辑：
    1. 拓扑检索：通过 Job 锚定核心词，在图谱中执行 (Vocab)<-[:HAS_TOPIC]-(Work)<-[:AUTHORED]-(Author) 路径扩展。
    2. 贡献分摊：作者总分 = Σ (单篇论文匹配质量 * 作者在该论文中的署名权重 pos_weight)。
    3. 两段式评分：仅对初步排序前 100 名执行高能耗向量内聚度计算，其余执行快速路径。
    4. 预载入优化：维持向量内存矩阵，确保 Index-Only 级别的计算速度。
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
            self.graph.run("RETURN 1").evaluate()
            print("[OK] Neo4j 连接成功")
        except Exception as e:
            print(f"[Error] Neo4j 连接失败: {e}")

        # 1. 加载 Faiss 索引及 ID 映射
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_id_map = json.load(f)

        self.vocab_index = faiss.read_index(VOCAB_INDEX_PATH)
        with open(VOCAB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.vocab_id_map = json.load(f)
            self.vocab_to_idx = {str(vid): i for i, vid in enumerate(self.vocab_id_map)}

        # 2. 向量预载入矩阵：消除召回时从 Faiss reconstruct 的 I/O 瓶颈
        print("[*] 正在预载入词汇向量矩阵以消除实时计算开销...", flush=True)
        ntotal = self.vocab_index.ntotal
        # 一次性将所有词汇向量加载到内存，支持 NumPy 高速索引
        self.all_vocab_vectors = self.vocab_index.reconstruct_n(0, ntotal).astype('float32')
        print(f"[OK] 预载入 {ntotal} 个词汇向量完成")

    def _get_skills_from_jobs(self, query_vector, top_n=5):
        """原生 Cursor 模式获取岗位关联的核心词"""
        _, indices = self.job_index.search(query_vector, top_n)
        job_ids = [self.job_id_map[idx] for idx in indices[0] if 0 <= idx < len(self.job_id_map)]

        cypher = """
        MATCH (j:Job) WHERE j.id IN $j_ids
        MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
        RETURN DISTINCT v.id AS vid
        """
        cursor = self.graph.run(cypher, j_ids=job_ids)
        return [int(record['vid']) for record in cursor]

    def _calculate_proximity_fast(self, hit_ids):
        """内聚度极速计算：利用内存矩阵执行矩阵乘法"""
        if len(hit_ids) < 2:
            return 0.5
        idxs = [self.vocab_to_idx[str(vid)] for vid in hit_ids if str(vid) in self.vocab_to_idx]
        if len(idxs) < 2:
            return 0.5

        # 直接从内存切片
        vecs = self.all_vocab_vectors[idxs]
        sim_matrix = np.dot(vecs, vecs.T)
        n = sim_matrix.shape[0]
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        return float(np.mean(upper_tri))

    def recall(self, query_vector, target_domains=None):
        """
        标签路召回：增加领域 ID 的可选过滤逻辑
        :param query_vector: 输入向量
        :param target_domains: 可选，领域 ID 列表，例如 ['1', '4']
        """
        if self.graph is None: return [], 0
        start_t = time.time()

        # 1. 锚定核心词并联想扩展 (保持原有逻辑)
        core_ids = self._get_skills_from_jobs(query_vector)
        if not core_ids: return [], 0
        core_ids_set = set(core_ids)

        expand_cypher = """
        MATCH (v:Vocabulary) WHERE v.id IN $v_ids
        OPTIONAL MATCH (v)-[:SIMILAR_TO]-(v_rel:Vocabulary)
        WITH v_rel LIMIT 100
        RETURN DISTINCT v_rel.id AS vid
        """
        ext_cursor = self.graph.run(expand_cypher, v_ids=core_ids)
        all_ids = list(core_ids_set | {int(rec['vid']) for rec in ext_cursor if rec['vid'] is not None})

        # --- 【核心修改】2. 构建带领域过滤的 Neo4j 召回逻辑 ---
        # 如果提供了 target_domains，则在 Cypher 中加入集合交叉验证逻辑
        domain_filter_clause = ""
        if target_domains:
            # 利用 any() 函数检查论文领域列表与岗位目标领域是否有交集
            domain_filter_clause = "AND any(d IN split(w.domain_ids, '|') WHERE d IN $target_domains)"

        final_cypher = f"""
        MATCH (v:Vocabulary) 
        USING INDEX v:Vocabulary(id)
        WHERE v.id IN $all_v_ids
        MATCH (v)<-[:HAS_TOPIC]-(w:Work)
        WHERE 1=1 {domain_filter_clause}  // <--- 动态注入领域过滤子句
        WITH w, collect(DISTINCT v.id) AS hit_ids LIMIT 500 
        MATCH (w)<-[auth_r:AUTHORED]-(a:Author)
        RETURN a.id AS aid, 
               sum(auth_r.pos_weight) AS raw_rank, 
               collect({{wid: w.id, hits: hit_ids, weight: auth_r.pos_weight}}) AS papers
        ORDER BY raw_rank DESC
        """

        # 执行查询，将 target_domains 传入 Cypher 参数
        cursor = self.graph.run(final_cypher, all_v_ids=all_ids, target_domains=target_domains or [])

        # 3. 内存内分级评分排名：执行作者贡献分摊逻辑 (保持原有逻辑)
        scored_authors = []
        processed_count = 0

        for record in cursor:
            processed_count += 1
            author_total_score = 0.0

            # --- 两段式计算策略优化 ---
            is_fast_path = processed_count > 100
            papers = record['papers']

            for paper in papers:
                h_ids = paper['hits']
                h_count = len(h_ids)
                rank_score = sum([100 if vid in core_ids_set else 20 for vid in h_ids])

                if is_fast_path:
                    proximity = 0.5
                else:
                    proximity = self._calculate_proximity_fast(h_ids)

                work_quality = (rank_score + h_count ** 2) * (1 + proximity)
                author_total_score += (work_quality * paper['weight'])

            scored_authors.append({
                'aid': record['aid'],
                'score': author_total_score
            })

        scored_authors.sort(key=lambda x: x['score'], reverse=True)
        duration = (time.time() - start_t) * 1000

        return [a['aid'] for a in scored_authors[:self.recall_limit]], duration


# ==============================================================================
# 调试主函数：支持交互式测试
# ==============================================================================
if __name__ == "__main__":
    # 实例化召回组件
    l_path = LabelRecallPath(recall_limit=200)
    l_path.verbose = True  # 开启调试输出

    print("\n" + "=" * 60)
    print("🚀 标签路召回独立调试控制台 (贡献度分配模型)")
    print("请输入 Query 向量 (JSON 列表格式) 进行搜索")
    print("=" * 60)

    try:
        while True:
            raw_input = input("\n请粘贴稠密向量 (输入 'q' 退出):\n>> ").strip()
            if not raw_input or raw_input.lower() == 'q':
                break

            try:
                # 解析向量
                vector_list = json.loads(raw_input)
                query_vec = np.array([vector_list]).astype('float32')

                # 执行 L2 归一化以适配 Faiss 的内积检索
                faiss.normalize_L2(query_vec)

                # 执行召回
                top_ids, search_time = l_path.recall(query_vec)

                print(f"\n[调试反馈]")
                print(f"- 召回耗时: {search_time:.2f} ms")
                print(f"- 召回人数: {len(top_ids)}")
                if top_ids:
                    print(f"- 首位作者 ID: {top_ids[0]}")
                print("-" * 30)

            except json.JSONDecodeError:
                print("[Error] JSON 格式错误，请确保粘贴的是合法的列表字符串。")
            except Exception as e:
                print(f"[Critical Error] {str(e)}")
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n[!] 调试已终止。")