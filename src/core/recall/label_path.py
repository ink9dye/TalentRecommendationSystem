import faiss
import json
import re
import time
import math
import collections
import numpy as np
import traceback
from datetime import datetime
from py2neo import Graph
from config import (
    CONFIG_DICT, JOB_INDEX_PATH, JOB_MAP_PATH,
    VOCAB_INDEX_PATH, VOCAB_MAP_PATH, DB_PATH
)

# --- 配置常量 ---
DOMAIN_DECAY_RATES = {
    "1": 0.90, "4": 0.92, "2": 0.94, "12": 0.99, "14": 0.98, "default": 0.95
}

HARD_BLACKLIST = {
    "深度学习", "deep learning", "机器学习", "machine learning",
    "python", "算法", "algorithm", "c++", "pytorch", "tensorflow",
    "人工智能", "ai", "神经网络", "neural network"
}


class LabelRecallPath:
    """
    解耦版标签路召回 - 结构化流水线
    """

    def __init__(self, recall_limit=200):
        self.recall_limit = recall_limit
        self.current_year = datetime.now().year
        self._init_resources()

        # 预载入统计数据
        self.total_work_count = self._get_node_count("Work")
        self.total_job_count = self._get_node_count("Job")

    def _init_resources(self):
        """初始化连接与索引资源"""
        try:
            self.graph = Graph(
                CONFIG_DICT["NEO4J_URI"],
                auth=(CONFIG_DICT["NEO4J_USER"], CONFIG_DICT["NEO4J_PASSWORD"]),
                name=CONFIG_DICT["NEO4J_DATABASE"]
            )
            self.job_index = faiss.read_index(JOB_INDEX_PATH)
            with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
                self.job_id_map = json.load(f)

            self.vocab_index = faiss.read_index(VOCAB_INDEX_PATH)
            with open(VOCAB_MAP_PATH, 'r', encoding='utf-8') as f:
                self.vocab_id_map = json.load(f)
                self.vocab_to_idx = {str(vid): i for i, vid in enumerate(self.vocab_id_map)}

            self.all_vocab_vectors = self.vocab_index.reconstruct_n(0, self.vocab_index.ntotal).astype('float32')
            print("[OK] 资源初始化完成")
        except Exception as e:
            print(f"[Error] 资源加载失败: {e}")
            self.graph = None

    def _get_node_count(self, label):
        """统计节点总数"""
        try:
            res = self.graph.run(f"MATCH (n:{label}) RETURN count(n) AS c").data()
            return float(res[0]['c']) if res else 1000000.0
        except:
            return 1000000.0

    # --- 第一阶段：环境与领域探测 ---
    def _detect_domain_context(self, query_vector):
        """探测 Job 空间的领域分布"""
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
        dominance = (domain_counter.most_common(1)[0][1] / 10.0) if domain_counter else 0
        return candidate_ids, inferred, dominance

    # --- 第二阶段：锚点技能提取 ---
    def _extract_anchor_skills(self, target_job_ids):
        """从匹配的 Job 中提取满足 2% 熔断条件的技能锚点"""
        cypher = """
        MATCH (j:Job) WHERE j.id IN $j_ids
        MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
        WITH v, COUNT { (v)<-[:REQUIRE_SKILL]-() } AS degree_j
        OPTIONAL MATCH (v_topic:Vocabulary {term: v.term})
        WHERE EXISTS { (v_topic)<-[:HAS_TOPIC]-() }
        WITH v, degree_j, COUNT { (v_topic)<-[:HAS_TOPIC]-() } AS degree_w
        WITH v, (degree_j * 1.0 / $total_j) AS cov_j, (degree_w * 1.0 / $total_w) AS cov_w,
             log10($total_w / (degree_w + 1)) AS idf_val
        WHERE cov_j < 0.02 AND cov_w < 0.02
        RETURN DISTINCT v.id AS vid, v.term AS term, idf_val
        ORDER BY idf_val DESC LIMIT 15
        """
        cursor = self.graph.run(cypher, j_ids=target_job_ids[:3],
                                total_j=self.total_job_count, total_w=self.total_work_count)

        return {str(r['vid']): {"term": r['term'], "idf": r['idf_val']}
                for r in cursor if len(r['term']) > 1 and r['term'].lower() not in HARD_BLACKLIST}

    # --- 第三阶段：语义扩展 ---
    def _expand_semantic_map(self, core_vids, anchor_skills):
        """扩展 Vocabulary 路径 (同名/相似度)"""
        cypher = """
        MATCH (v:Vocabulary) WHERE v.id IN $v_ids 
        OPTIONAL MATCH (v_topic:Vocabulary {term: v.term}) WHERE EXISTS { (v_topic)<-[:HAS_TOPIC]-() }
        OPTIONAL MATCH (v)-[:SIMILAR_TO]-(v_rel:Vocabulary) WHERE EXISTS { (v_rel)<-[:HAS_TOPIC]-() }
        RETURN v.id AS source_id, collect(DISTINCT v_topic.id) AS t_ids, collect(DISTINCT v_rel.id) AS r_ids, v.term AS term
        """
        cursor = self.graph.run(cypher, v_ids=core_vids)

        score_map, term_map = {}, {}
        for rec in cursor:
            s_term = rec['term']
            for tid in rec['t_ids']:
                score_map[str(tid)], term_map[str(tid)] = 100, s_term
            for rid in rec['r_ids']:
                if str(rid) not in score_map:
                    score_map[str(rid)], term_map[str(rid)] = 60, s_term
        return score_map, term_map

    # --- 第四阶段：向量紧密度计算 ---
    def _calculate_proximity(self, hit_ids):
        """计算命中标签在向量空间中的平均余弦相似度"""
        if len(hit_ids) < 2: return 0.5
        idxs = [self.vocab_to_idx.get(str(vid)) for vid in hit_ids if str(vid) in self.vocab_to_idx]
        if len(idxs) < 2: return 0.5
        vecs = self.all_vocab_vectors[idxs]
        sim_matrix = np.dot(vecs, vecs.T)
        return float(np.mean(sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)]))

    # --- 第五阶段：核心打分引擎 ---
    import re

    import re
    import math

    def _compute_contribution(self, paper, context):
        """
        单篇论文贡献度计算 - 拓扑信任版
        逻辑：岗位向量驱动 -> 命中Job -> 获取Domain/Skill -> 匹配Work -> 为Author加分
        """
        raw_title = (paper.get('title') or "").lower()

        # --- 1. 领域过滤逻辑 (基于你的思路：根据论文domain决定是否使用) ---
        work_domains = str(paper.get('domains', '')).replace('|', ',').split(',')
        # 检查论文领域是否在岗位推断出的 active_domains 列表中
        has_domain_match = any(ad in work_domains for ad in context['active_domains'])

        # 决策逻辑：
        # A. 如果论文明确标了领域，但不在岗位相关的领域里 -> 跨行噪声，舍弃
        # B. 如果论文没标领域 (None/Empty) -> 可能是新论文或标注缺失，选择信任其 HAS_TOPIC 关系，放行
        if paper.get('domains') and not has_domain_match:
            return 0, []

        # 领域系数：匹配则大幅加成，缺失则给中性基础权重 (0.5)
        domain_coeff = 1.0 + (context['dominance'] * 5.0) if has_domain_match else 0.5

        # --- 2. 标签匹配与 IDF 权重 (你的思路核心：通过topic找到论文) ---
        rank_score = 0
        valid_hids = []
        hit_terms = []

        # context['score_map'] 存储了从 Job 节点溯源而来的所有 Vocabulary 及其权重
        for hit in paper['hits']:
            vid_s = str(hit['vid'])
            if vid_s in context['score_map']:
                # 使用 IDF 平方加权：iLQR/MPC/ROS2 等硬核算法词的 IDF 远高于常用词
                # 这确保了“含金量”高的技术点能给作者带来极高加分
                weight = context['score_map'][vid_s] * math.pow(hit['idf'], 2)
                rank_score += weight
                valid_hids.append(hit['vid'])
                # 记录具体命中的标签名称，方便后续分析
                hit_terms.append(f"{context['term_map'][vid_s]}")

        # 若该论文在图谱中并没有与岗位相关的技能词建立 HAS_TOPIC 关系，则不计分
        if rank_score == 0:
            return 0, []

        # --- 3. 标题辅助奖励 (非硬性门槛) ---
        # 即使标题没写词，只要有关系也能召回；若标题写了，属于“点题”，给小额加成
        noise_penalty = 1.0
        if context['anchor_kws']:
            meaningful_kws = [k for k in context['anchor_kws'] if len(k) > 2 or k in ['ros', 'mpc']]
            found_in_title = any(re.search(rf"(^|[^a-z0-9]){re.escape(akw)}($|[^a-z0-9])", raw_title)
                                 for akw in meaningful_kws)
            if found_in_title:
                noise_penalty = 1.2

        # --- 4. 时序衰减与综述降权 ---
        # 综述论文 (命中标签过多) 贡献度衰减
        hit_count = len(valid_hids)
        survey_decay = (1.0 / math.pow(hit_count, 2)) if hit_count > 1 else 1.0
        if any(k in raw_title for k in ['survey', 'overview', 'review']):
            survey_decay *= 0.1

        # 时序分：越新的研究贡献越高
        year_diff = max(0, self.current_year - int(paper.get('year', 2000)))
        time_decay = math.pow(context['decay_rate'], year_diff)

        # --- 5. 组合最终得分 ---
        # 标签空间紧密度：命中的几个词在向量空间越近，说明作者在该垂直领域越深钻
        proximity = self._calculate_proximity(valid_hids)

        # 署名权重：基于 authored 边的权重（一作/通讯权重高）
        auth_weight = paper['weight'] if paper['weight'] > 0 else 0.001

        # 最终计算公式
        score = rank_score * (1 + proximity) * domain_coeff * time_decay * survey_decay * auth_weight * noise_penalty

        return score, hit_terms

    # --- 主召回逻辑 ---
    def recall(self, query_vector, domain_ids=None):
        if not self.graph: return [], 0
        start_t = time.time()

        # 1. 岗位空间探测：文本向量 -> 找到最接近的 Job 节点
        job_ids, inferred_domains, dominance = self._detect_domain_context(query_vector)

        # 2. 工业词提取 (JD侧)
        # 这一步找到的是直接挂在 Job 上的原始标签
        anchor_skills = self._extract_anchor_skills(job_ids)
        if not anchor_skills:
            self.last_debug_info = {"author_count": 0, "work_count": 0, "industrial_kws": [], "academic_kws": []}
            return [], 0

        industrial_kws = [v['term'] for v in anchor_skills.values()]

        # 3. 确定核心上下文与领域并集
        # 如果用户没传 domain_ids，则使用 inferred_domains 列表
        active_domains = [str(domain_ids)] if domain_ids and str(domain_ids) != "0" else inferred_domains

        # 4. 语义扩展：工业词 -> SIMILAR_TO -> 学术 Topic
        # 这一步是解决“工业词搜不到论文”的关键桥梁
        score_map, term_map = self._expand_semantic_map([int(k) for k in anchor_skills.keys()], anchor_skills)
        academic_kws = list(term_map.values())

        # 5. 执行图谱检索 (Topic -> Work)
        params = {"v_ids": [int(k) for k in score_map.keys()], "total_w": self.total_work_count}

        domain_clause = ""
        if active_domains:
            # 构造领域正则，匹配并集
            regex_str = f"(^|,){'|'.join(map(str, active_domains))}(,|$)"
            domain_clause = "AND (w.domain_ids =~ $regex OR w.domain_ids IS NULL OR w.domain_ids = '')"
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

        # 6. 打分流水线 (使用 pos_weight 加分)
        context = {
            'score_map': score_map,
            'term_map': term_map,
            'anchor_kws': [k.lower() for k in industrial_kws],
            'active_domains': active_domains,
            'dominance': dominance,
            'decay_rate': DOMAIN_DECAY_RATES.get(active_domains[0] if active_domains else "default", 0.95)
        }

        scored_authors = []
        all_works_count = 0
        for record in cursor:
            author_total_score = 0.0
            best_paper = None

            for paper in record['papers']:
                all_works_count += 1
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

        # 7. 排序与深度诊断信息记录
        scored_authors.sort(key=lambda x: x['score'], reverse=True)

        self.last_debug_info = {
            'active_domains': active_domains,
            'dominance': f"{dominance * 100:.1f}%",
            'industrial_kws': industrial_kws,  # 诊断：JD提取的原始词
            'academic_kws': academic_kws,  # 诊断：扩展出来的学术词
            'work_count': all_works_count,  # 诊断：检索到的论文总数
            'author_count': len(scored_authors),
            'top_samples': scored_authors[:20]
        }

        return [a['aid'] for a in scored_authors[:self.recall_limit]], (time.time() - start_t) * 1000

if __name__ == "__main__":
    from src.core.recall.input_to_vector import QueryEncoder
    import numpy as np
    import traceback

    l_path = LabelRecallPath(recall_limit=200)
    encoder = QueryEncoder()

    fields = {
        "1": "计算机科学", "4": "工程学", "5": "物理学", "14": "数学"
    }

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