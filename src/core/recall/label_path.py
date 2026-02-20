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
    标签路召回极限优化版 - 领域增强过滤版
    逻辑：描述向量 -> 领域限定岗位 -> 岗位Skill -> 知识扩展 -> 领域限定论文 -> 学者
    针对 Domain ID 匹配逻辑：支持复合 ID (如 "1,14")，输入 "1" 可匹配 "1" 或 "1,14"，但不匹配 "14"
    """

    def __init__(self, recall_limit=200):
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

        # 岗位索引
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_id_map = json.load(f)

        # 词汇索引及映射
        self.vocab_index = faiss.read_index(VOCAB_INDEX_PATH)
        with open(VOCAB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.vocab_id_map = json.load(f)
            self.vocab_to_idx = {str(vid): i for i, vid in enumerate(self.vocab_id_map)}

        print("[*] 正在预载入词汇向量矩阵...")
        ntotal = self.vocab_index.ntotal
        self.all_vocab_vectors = self.vocab_index.reconstruct_n(0, ntotal).astype('float32')
        print(f"[OK] 预载入 {ntotal} 个词汇向量完成")

    def _get_skills_from_top_jobs(self, query_vector, domain_ids=None, top_k=3):
        """步骤 1 & 2: 向量 -> 岗位 -> 工业Skill (包含领域过滤)"""
        # 初始检索范围稍大，以确保在过滤后仍能凑齐 top_k
        search_k = 100 if domain_ids else top_k
        _, indices = self.job_index.search(query_vector, search_k)
        candidate_job_ids = [self.job_id_map[idx] for idx in indices[0] if 0 <= idx < len(self.job_id_map)]

        if domain_ids and str(domain_ids) != "0":
            # 匹配逻辑：使用正则确保 ID 是独立的数字。例如输入 "1"，匹配 "1" 或 "1,14" 或 "14,1"
            # (?i) 代表不区分大小写, (^|,) 确保开头或逗号分隔
            regex_pattern = f"(^|,){domain_ids}(,|$)"

            cypher_filter = """
            MATCH (j:Job) WHERE j.id IN $j_ids
            AND j.domain_ids =~ $regex
            RETURN j.id AS jid
            LIMIT $top_k
            """
            cursor = self.graph.run(cypher_filter, j_ids=candidate_job_ids, regex=regex_pattern, top_k=top_k)
            target_job_ids = [record['jid'] for record in cursor]

            # 如果领域内岗位不足，从候选里按相关度补齐（不限领域）
            if len(target_job_ids) < top_k:
                for jid in candidate_job_ids:
                    if jid not in target_job_ids:
                        target_job_ids.append(jid)
                    if len(target_job_ids) >= top_k: break
        else:
            target_job_ids = candidate_job_ids[:top_k]

        cypher_skills = """
        MATCH (j:Job) WHERE j.id IN $j_ids
        MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
        RETURN DISTINCT v.id AS vid, v.term AS term
        """
        cursor = self.graph.run(cypher_skills, j_ids=target_job_ids)
        core_skills = {str(record['vid']): record['term'] for record in cursor}
        return core_skills

    def _calculate_proximity_fast(self, hit_ids):
        """计算命中词之间的语义紧凑度"""
        if len(hit_ids) < 2: return 0.5
        idxs = [self.vocab_to_idx.get(str(vid)) for vid in hit_ids if str(vid) in self.vocab_to_idx]
        if len(idxs) < 2: return 0.5
        vecs = self.all_vocab_vectors[idxs]
        sim_matrix = np.dot(vecs, vecs.T)
        n = sim_matrix.shape[0]
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        return float(np.mean(upper_tri))

    def recall(self, query_vector, domain_ids=None):
        if self.graph is None: return [], 0
        start_t = time.time()

        # 1. 获取核心工业词
        core_skills_map = self._get_skills_from_top_jobs(query_vector, domain_ids=domain_ids, top_k=3)
        if not core_skills_map: return [], 0

        core_vids = [int(vid) for vid in core_skills_map.keys()]

        # 2. 知识图谱联想扩展
        expand_cypher = """
        MATCH (v:Vocabulary) WHERE v.id IN $v_ids
        MATCH (v)-[:SIMILAR_TO]-(v_rel:Vocabulary)
        RETURN v.id AS source_id, v_rel.id AS rel_id, v_rel.term AS rel_term
        """
        ext_cursor = self.graph.run(expand_cypher, v_ids=core_vids)

        valid_score_map = {}
        vocab_term_map = {}

        for vid, term in core_skills_map.items():
            valid_score_map[vid] = 100
            vocab_term_map[vid] = term

        for rec in ext_cursor:
            rel_id_s = str(rec['rel_id'])
            if rel_id_s not in valid_score_map:
                valid_score_map[rel_id_s] = 60
                vocab_term_map[rel_id_s] = rec['rel_term']

        all_eligible_vids = [int(vid) for vid in valid_score_map.keys()]

        # 3. 执行最终召回：增加论文 Domain 匹配逻辑
        # 使用正则过滤 Work(论文) 的 domain_ids
        domain_filter_clause = ""
        params = {"all_v_ids": all_eligible_vids}

        if domain_ids and str(domain_ids) != "0":
            domain_filter_clause = "AND w.domain_ids =~ $regex"
            params["regex"] = f"(^|,){domain_ids}(,|$)"

        final_cypher = f"""
        MATCH (v:Vocabulary) WHERE v.id IN $all_v_ids
        MATCH (v)<-[:HAS_TOPIC]-(w:Work)
        WHERE 1=1 {domain_filter_clause}
        WITH w, collect(DISTINCT v.id) AS hit_ids LIMIT 1000 
        MATCH (w)<-[auth_r:AUTHORED]-(a:Author)
        RETURN a.id AS aid, 
               collect({{wid: w.id, hits: hit_ids, weight: auth_r.pos_weight, title: w.title, year: w.year}}) AS papers
        """
        cursor = self.graph.run(final_cypher, **params)

        scored_authors = []
        all_works_found = set()

        for record in cursor:
            author_total_score = 0.0
            papers = record['papers']
            paper_details = []

            for paper in papers:
                all_works_found.add(paper['wid'])
                h_ids = paper['hits']
                rank_score = 0
                hit_terms = []
                valid_hids_for_prox = []

                for vid in h_ids:
                    s_vid = str(vid)
                    if s_vid in valid_score_map:
                        s = valid_score_map[s_vid]
                        rank_score += s
                        hit_terms.append(f"{vocab_term_map[s_vid]}({int(s)})")
                        valid_hids_for_prox.append(vid)

                if rank_score == 0: continue

                proximity = self._calculate_proximity_fast(valid_hids_for_prox)
                work_quality = rank_score * (1 + proximity)
                w_val = paper['weight'] if paper['weight'] > 0 else 0.001
                contribution = work_quality * w_val
                author_total_score += contribution

                paper_details.append({
                    'title': paper['title'],
                    'year': paper.get('year', 'N/A'),
                    'weight': round(w_val, 4),
                    'contribution': round(contribution, 2),
                    'hits': hit_terms
                })

            if not paper_details: continue

            paper_details.sort(key=lambda x: x['contribution'], reverse=True)
            scored_authors.append({
                'aid': record['aid'],
                'score': author_total_score,
                'top_paper': paper_details[0],
                'paper_count': len(papers)
            })

        scored_authors.sort(key=lambda x: x['score'], reverse=True)
        duration = (time.time() - start_t) * 1000

        self.last_debug_info = {
            'anchor_skills': list(core_skills_map.values()),
            'work_count': len(all_works_found),
            'author_count': len(scored_authors),
            'top_samples': scored_authors[:10]
        }

        return [a['aid'] for a in scored_authors[:self.recall_limit]], duration


if __name__ == "__main__":
    l_path = LabelRecallPath(recall_limit=200)
    print("\n" + "=" * 80)
    print("🚀 标签路 (LabelPath) - 严格领域匹配模式 (正则过滤)")
    print("=" * 80)

    try:
        while True:
            # 输入 domain_ids, 如 "1"
            domain_ids = input("\n请输入 Domain ID (1-17, 0或回车跳过): ").strip()
            if not domain_ids: domain_ids = "0"

            raw_input = input("请输入向量 JSON 或 'q' 退出: ").strip()
            if not raw_input or raw_input.lower() == 'q': break

            try:
                vector_list = json.loads(raw_input)
                query_vec = np.array([vector_list]).astype('float32')
                faiss.normalize_L2(query_vec)

                top_ids, search_time = l_path.recall(query_vec, domain_ids=domain_ids)
                db = l_path.last_debug_info

                print(f"\n[分析报告 - 当前领域限制: {domain_ids}]")
                print(f" > 匹配岗位核心词: {db.get('anchor_skills')}")
                print(f" > 检索到论文数: {db.get('work_count')} | 有效学者数: {db.get('author_count')}")

                print(f"\n{'排名':<3} | {'作者 ID':<12} | {'总分':<8} | {'篇数':<4} | {'最高贡献论文明细'}")
                print("-" * 120)
                for i, item in enumerate(db.get('top_samples', []), 1):
                    tp = item.get('top_paper', {})
                    hits_str = ",".join(tp.get('hits', [])[:3])
                    print(f"#{i:<2} | {item['aid']:<12} | {int(item['score']):<8} | {item['paper_count']:<4} | "
                          f"贡献:{tp.get('contribution')} | W:{tp.get('weight')} | 命中:{hits_str}")
                    if i <= 3:
                        print(f"    └─ 代表作: {tp.get('title', 'N/A')[:85]}...")

                print(f"\n[性能] 召回耗时: {search_time:.2f} ms")
            except Exception:
                traceback.print_exc()
    except KeyboardInterrupt:
        print("\n[!] 调试已终止。")