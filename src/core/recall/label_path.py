import faiss
import json
import re
import os
import sqlite3
import time
import math
import collections
import numpy as np
import traceback
from datetime import datetime
from py2neo import Graph
from src.core.recall.input_to_vector import QueryEncoder
from src.utils.domain_utils import DomainProcessor
from config import (
    CONFIG_DICT, JOB_INDEX_PATH, JOB_MAP_PATH,
    VOCAB_INDEX_PATH, VOCAB_MAP_PATH, DB_PATH,VOCAB_STATS_DB_PATH
)

# --- 算法配置：不同领域（Domain ID）的年度时间价值衰减率 ---
# 1-计算机(0.90, 更新快), 4-工程(0.92), 12-临床医学(0.99, 较稳定)
DOMAIN_DECAY_RATES = {
    "1": 0.90, "4": 0.92, "2": 0.94, "12": 0.99, "14": 0.98, "default": 0.95
}


class LabelRecallPath:
    """
    【核心架构】解耦版标签路召回 - 结构化流水线
    逻辑：通过向量检索探测领域 -> 从岗位(Job)提取工业技能锚点 -> 知识图谱语义扩展 ->
          映射至学术词汇(Vocabulary) -> 召回论文(Work) -> 综合评分计算专家(Author)贡献度。
    """

    def __init__(self, recall_limit=200):
        self.recall_limit = recall_limit
        self.current_year = datetime.now().year
        self._init_resources()

        # 预载入统计数据，用于计算后续 IDF 与 熔断率
        self.total_work_count = self._get_node_count("Work")
        self.total_job_count = self._get_node_count("Job")

    def _init_resources(self):
        """
        【资源初始化】解决 Faiss ID 与 向量矩阵的同步问题
        1. Faiss 索引：仅用于快速 Top-K 检索。
        2. .npy 矩阵：存储原始归一化向量，用于计算词汇间的语义紧密度（Proximity）。
        3. SQLite 映射：确保矩阵行号(Index)与数据库 voc_id 严格对齐。
        """
        try:
            # A. 初始化图数据库连接
            self.graph = Graph(
                CONFIG_DICT["NEO4J_URI"],
                auth=(CONFIG_DICT["NEO4J_USER"], CONFIG_DICT["NEO4J_PASSWORD"]),
                name=CONFIG_DICT["NEO4J_DATABASE"]
            )

            # B. 加载岗位描述索引（用于第一阶段：领域探测）
            self.job_index = faiss.read_index(JOB_INDEX_PATH)
            with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
                self.job_id_map = json.load(f)

            # C. 加载词汇索引与向量快照
            self.vocab_index = faiss.read_index(VOCAB_INDEX_PATH)
            vec_path = VOCAB_INDEX_PATH.replace('.faiss', '_vectors.npy')
            if not os.path.exists(vec_path):
                raise FileNotFoundError(f"未发现向量快照: {vec_path}，请先运行 build_vector_index.py。")

            # 直接加载原始向量矩阵，避开 IndexIDMap 不支持 reconstruct 的局限
            self.all_vocab_vectors = np.load(vec_path).astype('float32')

            # D. 建立 { 'voc_id': 矩阵行下标 } 映射
            # 必须 ORDER BY voc_id 以匹配向量编码时的顺序
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("SELECT voc_id FROM vocabulary ORDER BY voc_id ASC").fetchall()
                self.vocab_to_idx = {str(r[0]): i for i, r in enumerate(rows)}
            self.stats_conn = sqlite3.connect(VOCAB_STATS_DB_PATH, check_same_thread=False)
            print("[OK] 标签路资源初始化完成")
        except Exception as e:
            print(f"[Error] 资源加载失败: {e}")
            self.graph = None

    def _get_node_count(self, label):
        """统计图谱节点总数，作为计算 IDF 的分母"""
        try:
            res = self.graph.run(f"MATCH (n:{label}) RETURN count(n) AS c").data()
            return float(res[0]['c']) if res else 1000000.0
        except:
            return 1000000.0

    # --- 第一阶段：环境与领域探测 ---
    def _detect_domain_context(self, query_vector):
        """
        【领域探测】通过用户 Query 在 Job 空间寻找最相关的行业分布
        逻辑：检索相似岗位 -> 统计其 domain_ids -> 确定当前搜索的“主战场”。
        """
        _, indices = self.job_index.search(query_vector, 10)
        candidate_ids = [self.job_id_map[idx] for idx in indices[0] if 0 <= idx < len(self.job_id_map)]

        domain_counter = collections.Counter()
        cursor = self.graph.run(
            "MATCH (j:Job) WHERE j.id IN $j_ids RETURN j.domain_ids AS d_ids",
            j_ids=candidate_ids
        )
        for row in cursor:
            if row['d_ids']:
                for d in str(row['d_ids']).split(','):
                    domain_counter[d.strip()] += 1

        inferred = [d for d, _ in domain_counter.most_common(3)]
        # dominance：主导领域在 Top10 中的占比，决定后续领域分值的加成强度
        dominance = (domain_counter.most_common(1)[0][1] / 10.0) if domain_counter else 0
        return candidate_ids, inferred, dominance


    # --- 第二阶段：锚点技能提取 ---
    def _extract_anchor_skills(self, target_job_ids):
        """
        【工业侧：岗位技能提取】
        目的：从匹配的岗位中找到“高含金量”的专业词。
        逻辑：
        1. 统计命中岗位(Top 3)所要求的所有技能(Vocabulary)。
        2. 计算该技能在全量岗位节点(total_j)中的覆盖率 cov_j。
        3. 1% 熔断：仅保留在全量 Job 中出现频率 < 1% 的专业技能，剔除“沟通、办公”等通用词。
        4. 扩容：将 LIMIT 设为 50，以完整捕捉复合型岗位中的多个技术维度。
        """
        cypher = """
        MATCH (j:Job) WHERE j.id IN $j_ids
        MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
        WITH v, (COUNT { (v)<-[:REQUIRE_SKILL]-() } * 1.0 / $total_j) AS cov_j

        // --- 约束 1：岗位侧 1% 熔断 (按 Job 总节点数计算) ---
        WHERE cov_j < 0.01 

        RETURN DISTINCT v.id AS vid, v.term AS term, cov_j
        ORDER BY cov_j ASC 
        LIMIT 50 // 扩大上限至 50，确保长 JD 中的所有核心技术点(如 RRT, MPC, MuJoCo)均能作为锚点
        """
        cursor = self.graph.run(cypher, j_ids=target_job_ids[:5], total_j=self.total_job_count)

        # 返回格式：{vid: {"term": "术语名称"}}
        return {str(r['vid']): {"term": r['term']} for r in cursor if len(r['term']) > 1}
    # --- 第三阶段：语义扩展 ---
    def _expand_semantic_map(self, core_vids, anchor_skills, domain_regex=None, query_vector=None):
        """
        【主函数】语义扩展引擎：引入“领域跨度”与“语义纯度”双重惩罚。
        逻辑：作为中转站，将检索到的图拓扑数据与原始 JD 向量（query_vector）传递给打分引擎。
        """
        regex = domain_regex if domain_regex else ".*"

        # 1. 辅助函数 A：执行图检索，获取包含领域跨度、纯度及击中数（hit_count）的原始数据
        # 这一步利用 Neo4j 跳转和 SQLite 索引实现极速查询
        raw_results = self._query_expansion_with_topology(core_vids, regex)

        # 2. 辅助函数 B：应用数学公式计算最终动态权重
        # 【修复关键】：补齐 query_vector 参数，以支持 SBERT 语义守门员逻辑
        return self._calculate_final_weights(raw_results, query_vector)

    def _query_expansion_with_topology(self, v_ids, regex):
        """
        【信号收敛版】辅助函数 A：统计“技术共振”击中数。
        逻辑：
        1. Neo4j 统计每个学术词被多少个不同的工业锚点(v)击中。
        2. SQLite 提供论文侧的全局领域统计。
        """
        # 1. 极速 Neo4j 跳转：统计 hit_count
        # DISTINCT v 确保即使物理上有重复边，逻辑上也只计算一次有效击中
        cypher = """
        MATCH (v:Vocabulary) WHERE v.id IN $v_ids
        MATCH (v)-[:SIMILAR_TO]-(v_rel:Vocabulary)
        RETURN v_rel.id AS tid, v_rel.term AS term,
               COUNT(DISTINCT v) AS hit_count, 
               (COUNT { (v_rel)<-[:REQUIRE_SKILL]-() } * 1.0 / $total_j) AS cov_j
        """
        params = {"v_ids": v_ids, "total_j": self.total_job_count}
        candidates = self.graph.run(cypher, **params).data()

        # 解析当前目标领域 ID
        active_domains = set(re.findall(r'\d+', regex))
        results = []

        # 2. 极速 SQLite 查表
        for cand in candidates:
            tid = cand['tid']
            row = self.stats_conn.execute(
                "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (tid,)
            ).fetchone()

            if not row: continue

            degree_w, domain_span, dist_json = row
            dist = json.loads(dist_json)
            target_degree_w = sum(dist.get(str(d), 0) for d in active_domains)

            if target_degree_w > 0:
                results.append({
                    'tid': tid,
                    'term': cand['term'],
                    'degree_w': degree_w,
                    'target_degree_w': target_degree_w,
                    'cov_j': cand['cov_j'],
                    'domain_span': domain_span,
                    'hit_count': cand['hit_count']  # <--- 传递信号收敛强度
                })

        return results

    def _calculate_final_weights(self, raw_results, query_vector):
        """
        【收敛加权版】权重计算调度器
        逻辑：将图谱收敛信号与向量空间语义信号进行融合。
        """
        score_map, term_map, idf_map = {}, {}, {}

        for rec in raw_results:
            tid = str(rec['tid'])

            # 1. 调用增强版数学引擎，应用多路信号收敛奖惩
            dynamic_weight, idf_val = self._apply_word_quality_penalty(rec, query_vector)

            # 2. 存储结果
            score_map[tid] = dynamic_weight
            term_map[tid] = rec['term']
            idf_map[tid] = idf_val

        return score_map, term_map, idf_map

    def _apply_word_quality_penalty(self, rec, query_vector):
        """
        【信号收敛+语义增强版】核心数学引擎
        降噪维度：语义守门员(SBERT) + 领域跨度(Span) + 领域纯度(Purity)
        加成维度：IDF + 技术共振奖励(Hit Count)
        """
        degree_w = rec['degree_w']
        cov_j = rec['cov_j']
        # 确保分母不为 0
        domain_span = max(1, rec['domain_span'])

        # 1. 计算语义纯度 (Tag Purity)
        # 逻辑：目标领域的产出占比，过滤“挂羊头卖狗肉”的词汇
        tag_purity = rec['target_degree_w'] / degree_w

        # 2. 计算基础学术稀缺度 (IDF)
        idf_val = math.log10(self.total_work_count / (degree_w + 1))

        # 3. 计算断崖式岗位惩罚 (Suppression of generic job terms)
        # 针对 0.5% 以上覆盖率的词进行指数级压制
        job_penalty = 1.0 + math.exp(300.0 * (cov_j - 0.005))

        # --- 4. 【新增：多路信号收敛奖励】 ---
        # 逻辑：一个学术词被越多的工业锚点同时命中，说明其确定性越高。
        # 使用 log1p 以获得边际效应递减的平滑增长。
        hit_count = rec.get('hit_count', 1)
        convergence_bonus = math.log1p(hit_count)

        # --- 5. 【新增：SBERT 语义守门员】 ---
        # 逻辑：利用预计算的向量矩阵，计算学术词与当前 JD 整体语义的余弦相似度。
        cos_sim = 0.5  # 默认中立分
        idx = self.vocab_to_idx.get(str(rec['tid']))
        if idx is not None and query_vector is not None:
            term_vec = self.all_vocab_vectors[idx]
            # 计算点积（归一化向量即为余弦相似度）
            cos_sim = float(np.dot(term_vec, query_vector.flatten()))

        # 应用非线性惩罚：pow(cos_sim, 6)
        # 这将导致相似度低于 0.7 的词（如加密算法之于机器人）权重呈断崖式下跌。
        semantic_factor = math.pow(max(0, cos_sim), 6)

        # 6. 【最终公式：立体的评价体系】
        # $$Weight = \frac{IDF}{JobPenalty} \times Purity^2 \times \frac{Bonus}{Span} \times SemanticFactor$$
        dynamic_weight = (
                (idf_val / job_penalty)
                * math.pow(tag_purity, 2)  # 纯度平方奖励垂直度
                / domain_span  # 跨度降权压制万金油
                * convergence_bonus  # 技术共振加成
                * semantic_factor  # SBERT 语义守门员
        )

        return dynamic_weight, idf_val

    # --- 第四阶段：向量紧密度计算 ---
    def _calculate_proximity(self, hit_ids):
        """
        【语义纯度验证】计算命中标签在向量空间中的平均余弦相似度
        逻辑：如果一个作者命中的标签彼此在语义上很近（如：SLAM 和 激光雷达），则证明专家在该细分领域非常专注，得分越高。
        """
        if len(hit_ids) < 2: return 0.5
        idxs = [self.vocab_to_idx.get(str(vid)) for vid in hit_ids if str(vid) in self.vocab_to_idx]
        if len(idxs) < 2: return 0.5

        # 提取向量并计算相似度矩阵
        vecs = self.all_vocab_vectors[idxs]
        sim_matrix = np.dot(vecs, vecs.T)
        # 取上三角（不含对角线）计算均值
        return float(np.mean(sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)]))

    # --- 第五阶段：核心打分引擎 ---
    def _is_retracted(self, title):
        """
        【辅助函数 1】撤稿拦截器
        逻辑：识别论文是否为撤稿通知，这类文档不具备人才评价价值。
        """
        return "retraction" in title.lower()

    def _get_domain_purity_factor(self, paper_domains_raw, active_set, dominance):
        """
        【辅助函数 2】领域专注度计算（Purity Engine）
        逻辑：
        1. 领域一票否决：与目标领域完全无交集的论文直接排除。
        2. 纯度惩罚（你的期待）：涉及的领域越多，非目标领域占比越大，得分越低。
        3. 4 次方加成：通过 math.pow(ratio, 4) 让“不专注”的论文分数呈断崖式下跌。
        """
        paper_domains = DomainProcessor.to_set(paper_domains_raw)

        # 计算交集：论文中属于目标领域的部分
        intersect = paper_domains.intersection(active_set)

        # 1. 领域硬约束：如果明确标注了领域但与目标集完全不交，分值为 0
        if paper_domains and not intersect:
            return 0.0

        # 2. 计算纯度比率 (Purity Ratio)
        # 逻辑：目标领域数 / 总涉及领域数。例如 {计算机, 工程, 教育} 找机器人，纯度为 2/3 = 0.66
        purity_ratio = 1.0
        if paper_domains:
            purity_ratio = len(intersect) / len(paper_domains)

        # 3. 基础领域分与纯度惩罚
        # 采用 4 次方惩罚：纯度 0.5 的论文（如 CS+教育），其系数仅剩 0.0625 倍
        base_score = 1.0 + (dominance * 5.0) if intersect else 0.5
        return base_score * math.pow(purity_ratio, 4)

    def _compute_contribution(self, paper, context):
        """
        【主评分函数】量化贡献度计算
        调度逻辑：拦截撤稿 -> 计算领域纯度 -> 累加标签权重 -> 应用时序与紧密度加成。
        """
        raw_title = (paper.get('title') or "")

        # 1. 撤稿拦截
        if self._is_retracted(raw_title):
            return 0, []

        # 2. 领域纯度降权：调用辅助函数计算基于“专注度”的领域系数
        # 解决了你担心的“涉及领域越多分越低”的问题
        domain_coeff = self._get_domain_purity_factor(
            paper.get('domains'),
            context['active_domain_set'],
            context['dominance']
        )
        if domain_coeff <= 0:
            return 0, []

        # 3. 标签匹配与动态权重累加
        # 这里的 score_map 已经包含了之前修改的“词级领域跨度惩罚”
        rank_score = 0
        valid_hids, hit_terms = [], []
        for hit in paper['hits']:
            vid_s = str(hit['vid'])
            if vid_s in context['score_map']:
                rank_score += context['score_map'][vid_s] * hit['idf']
                valid_hids.append(hit['vid'])
                hit_terms.append(context['term_map'][vid_s])

        if rank_score == 0:
            return 0, []

        # 4. 综述降权 (原样保留你的 1/n^2 逻辑)
        hit_count = len(valid_hids)
        survey_decay = (1.0 / math.pow(hit_count, 2)) if hit_count > 1 else 1.0
        if any(k in raw_title.lower() for k in ['survey', 'overview', 'review']):
            survey_decay *= 0.1

        # 5. 指数级紧密度加成 (1+prox)^n
        proximity = self._calculate_proximity(valid_hids)
        proximity_bonus = math.pow(1.0 + proximity, hit_count)

        # 6. 时序衰减与署名权重
        year_diff = max(0, self.current_year - int(paper.get('year', 2000)))
        time_decay = math.pow(context['decay_rate'], year_diff)
        auth_weight = float(paper.get('weight') or 0.001)

        # 最终组合
        score = rank_score * proximity_bonus * domain_coeff * time_decay * survey_decay * auth_weight
        return score, hit_terms

    def recall(self, query_vector, domain_ids=None):
        """
        【最终修复版】全链路调度：引入 SBERT 语义守门员与多路信号收敛。
        修改点：
        1. 修复参数传递链：显式将 query_vector 传入语义扩展模块。
        2. 确保 Step 4 能够激活底层的向量相似度校验与技术共振计算。
        """
        if not self.graph: return [], 0
        start_t = time.time()

        # 1. 岗位空间探测：探测当前 JD 所属的主战场领域
        job_ids, inferred_domains, dominance = self._detect_domain_context(query_vector)

        # 2. 锚点提取：获取工业侧硬核技能词
        anchor_skills = self._extract_anchor_skills(job_ids)
        if not anchor_skills:
            return [], 0
        industrial_kws = [v['term'] for v in anchor_skills.values()]

        # 3. 领域处理：确定最终过滤范围（正则字符串）
        active_domain_set = DomainProcessor.to_set(
            domain_ids if domain_ids and str(domain_ids) != "0" else inferred_domains)
        regex_str = DomainProcessor.build_neo4j_regex(active_domain_set)

        # 4. 语义扩展：【核心修复点】
        # 补齐 query_vector 参数，彻底解决 _calculate_final_weights 报错问题。
        # 此步骤现在集成：技术共振奖励(Hit Count) + SBERT 语义守门员(Semantic Factor)
        score_map, term_map, idf_map = self._expand_semantic_map(
            [int(k) for k in anchor_skills.keys()],
            anchor_skills,
            domain_regex=regex_str,
            query_vector=query_vector  # <--- 关键修复：补齐此参数传递
        )

        # 对诊断用的学术词进行去重展示
        academic_kws = list(set(term_map.values()))

        # 5. 图谱拓扑检索：通过学术词反查 Work 节点
        params = {"v_ids": [int(k) for k in score_map.keys()], "total_w": self.total_work_count}
        domain_clause = ""
        if regex_str:
            domain_clause = "AND w.domain_ids =~ $regex"
            params["regex"] = regex_str

        final_cypher = f"""
        MATCH (v:Vocabulary) WHERE v.id IN $v_ids
        WITH v, COUNT {{ (v)<-[:HAS_TOPIC]-() }} AS degree_w
        WHERE (degree_w * 1.0 / $total_w) < 0.01 
        WITH v, log10($total_w / (degree_w + 1)) AS idf_weight
        MATCH (v)<-[:HAS_TOPIC]-(w:Work) 
        WHERE 1=1 {domain_clause} 
        WITH w, collect({{vid: v.id, idf: idf_weight}}) AS hit_info LIMIT 2000
        MATCH (w)<-[auth_r:AUTHORED]-(a:Author)
        RETURN a.id AS aid, collect({{wid: w.id, hits: hit_info, weight: auth_r.pos_weight, 
                                     title: w.title, year: w.year, domains: w.domain_ids}}) AS papers
        """
        cursor = self.graph.run(final_cypher, **params)

        # 6. 打分与上下文构建
        first_domain = list(active_domain_set)[0] if active_domain_set else "default"
        context = {
            'score_map': score_map,
            'term_map': term_map,
            'anchor_kws': [k.lower() for k in industrial_kws],
            'active_domain_set': active_domain_set,
            'dominance': dominance,
            'decay_rate': DOMAIN_DECAY_RATES.get(first_domain, 0.95)
        }

        scored_authors, all_works_count = [], 0
        for record in cursor:
            author_total_score, best_paper = 0.0, None
            for paper in record['papers']:
                all_works_count += 1
                # 贡献度计算：内含 4 次方纯度惩罚与负向领域拦截
                p_score, p_hits = self._compute_contribution(paper, context)
                author_total_score += p_score
                if p_score > 0 and (not best_paper or p_score > best_paper['contribution']):
                    best_paper = {
                        'title': paper['title'], 'year': paper['year'],
                        'contribution': round(p_score, 4), 'hits': p_hits
                    }

            if author_total_score > 0:
                scored_authors.append({
                    'aid': record['aid'], 'score': author_total_score,
                    'top_paper': best_paper, 'paper_count': len(record['papers'])
                })

        # 7. 最终排序与诊断封装
        scored_authors.sort(key=lambda x: x['score'], reverse=True)
        self.last_debug_info = {
            'active_domains': list(active_domain_set),
            'dominance': f"{dominance * 100:.1f}%",
            'industrial_kws': industrial_kws,
            'academic_kws': academic_kws,
            'work_count': all_works_count,
            'author_count': len(scored_authors),
            'top_samples': scored_authors[:20]
        }

        return [a['aid'] for a in scored_authors[:self.recall_limit]], (time.time() - start_t) * 1000
if __name__ == "__main__":


    l_path = LabelRecallPath(recall_limit=200)
    encoder = QueryEncoder()


    print("\n" + "=" * 115)
    print("🚀 增强诊断版：标签路 (Label Path) 全链路追踪")
    print("=" * 115)

    try:
        domain_choice = input("\n请选择领域编号 (0跳过): ").strip() or "0"

        while True:
            user_input = input(f"\n请输入岗位需求 (q退出): ").strip()
            if not user_input or user_input.lower() == 'q': break

            query_vec, _ = encoder.encode(user_input)
            faiss.normalize_L2(query_vec)

            top_ids, search_time = l_path.recall(query_vec, domain_ids=domain_choice)

            # --- 核心诊断日志 ---
            db = l_path.last_debug_info
            print("\n" + "🔍 [深度诊断流水线]" + "-" * 98)

            # 1. 领域并集
            domains = db.get('active_domains', [])
            domain_str = " | ".join(domains) if domains else "未限制"
            print(f"【Step 1: 领域探测】目标领域并集: [{domain_str}] (置信度: {db.get('dominance')})")

            # 2. 工业词 (JD 侧)
            i_kws = db.get('industrial_kws', [])
            print(f"【Step 2: 工业锚点】从 JD 提取的原始词: {i_kws}")

            # 3. 学术词 (跳转后)
            a_kws = db.get('academic_kws', [])
            print(f"【Step 3: 语义扩展】通过 SIMILAR_TO 映射到的学术词: {a_kws}")

            # 诊断跳转失败
            if i_kws and not a_kws:
                print("   ⚠️ 错误: 工业词未能跳转到学术词！请检查 Vocabulary 节点间的 SIMILAR_TO 关系。")

            # 4. 论文检索
            w_count = db.get('work_count', 0)
            print(f"【Step 4: 论文检索】在上述学术词下检索到 {w_count} 篇论文 (已过领域并集过滤)。")

            if a_kws and w_count == 0:
                print("   ⚠️ 警告: 已有学术词但召回为 0。可能原因：")
                print(f"      - 这些学术词在图谱中没有 HAS_TOPIC 连向 Work 节点。")
                print(f"      - 论文的领域标签与 [{domain_str}] 完全不交叠。")

            # 5. 作者加分
            a_count = db.get('author_count', 0)
            print(f"【Step 5: 贡献评价】最终有效作者: {a_count} 名")

            # --- 结果展示 ---
            print("-" * 115)
            print(f"{'排名':<6} | {'作者 ID':<12} | {'综合得分':<15} | {'知识图谱核心作 (命中标签数)'}")
            print("-" * 115)

            for i, item in enumerate(db.get('top_samples', []), 1):
                tp = item.get('top_paper', {})
                title = tp.get('title', 'Unknown')[:55]
                hit_info = f"(命中 {len(tp.get('hits', []))} 标签: {', '.join(tp.get('hits', []))})"
                print(f"#{i:<5} | {item['aid']:<12} | {int(item['score']):<15} | 《{title}》")
                print(f"{' ':23} ┗━ {hit_info}")

            print("-" * 115)
            print(f"[*] 诊断完成。耗时: {search_time:.2f}ms")

    except Exception as e:
        print(f"运行出错: {e}")
        traceback.print_exc()