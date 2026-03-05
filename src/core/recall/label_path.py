# -*- coding: utf-8 -*-
"""
标签路召回模块（LabelRecallPath）。

实现「Job → 技能清洗 → Job Vocabulary → SIMILAR_TO(top3, sim≥0.65) → 学术词(paper_count≤277) →
score=sim/log(1+paper_count) → Paper → Author」的图谱召回。
  - Step1 技能清洗：过滤 HR 垃圾词（经验/竞赛/获奖/发表/能力/熟悉/了解、不限/其他、长度<2）。
  - Step2 锚点：仅保留清洗后词在图谱 REQUIRE_SKILL 中的技能，3% 熔断。
  - Step3 扩展：每锚点 SIMILAR_TO 最多 top3、边权≥0.65；学术词过滤 work_count≤277。
  - Step4 打分：vocab_score = sim / log(1+paper_count)，论文/作者分累加。
依赖 vocab_stats.db 的 vocabulary_domain_stats；共鸣/共现指标仍计算供调试，最终权重采用上述公式。
输入输出格式与原有接口一致。
"""
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
    VOCAB_INDEX_PATH, VOCAB_MAP_PATH, DB_PATH, VOCAB_STATS_DB_PATH,
    VOCAB_P95_PAPER_COUNT, SIMILAR_TO_TOP_K, SIMILAR_TO_MIN_SCORE,
)
from src.utils.domain_config import DOMAIN_DECAY_RATES, DEFAULT_DECAY_RATE


class LabelRecallPath:
    """
    【核心架构】解耦版标签路召回 - 结构化流水线

    逻辑：通过向量检索探测领域 -> 从岗位(Job)提取工业技能锚点 -> 知识图谱语义扩展 ->
          映射至学术词汇(Vocabulary) -> 召回论文(Work) -> 综合评分计算专家(Author)贡献度。

    语义扩展阶段的词级权重除原有维度外，还引入共现领域指标（依赖 vocab_stats.db 的
    vocabulary_cooccurrence + vocabulary_domain_stats）：
      - 与各种领域的词都共现 → 万金油 → 按共现伙伴平均领域跨度(cooc_span)降权；
      - 只跟特定领域的词共现 → 专精 → 按共现伙伴目标领域纯度(cooc_purity)加权；
      - 与本次要搜索的词汇有共现 → 单词协作 → 沿用 resonance 与 convergence_bonus 加权。
    """

    def __init__(self, recall_limit=200, verbose=False):
        self.recall_limit = recall_limit
        self.verbose = verbose
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

    def _get_job_previews(self, job_ids, max_snippet=200):
        """
        查询命中岗位的名称与描述片段，用于诊断「Top10 是否真是目标领域岗位」。
        返回: [{"id": id, "name": name, "description_snippet": desc[:max_snippet]}, ...]
        """
        if not job_ids or not self.graph:
            return []
        try:
            cursor = self.graph.run(
                "MATCH (j:Job) WHERE j.id IN $j_ids RETURN j.id AS id, j.name AS name, j.description AS desc",
                j_ids=job_ids[:20]
            )
            out = []
            for row in cursor:
                desc = (row.get('desc') or '') or ''
                if isinstance(desc, str) and len(desc) > max_snippet:
                    desc = desc[:max_snippet] + '...'
                out.append({
                    'id': row.get('id'),
                    'name': (row.get('name') or '')[:80],
                    'description_snippet': desc
                })
            return out
        except Exception:
            return []

    def _get_anchor_debug_stats(self, job_ids, total_j):
        """
        统计参与锚点提取的岗位的 REQUIRE_SKILL 数量，以及 1% 熔断前后词数、被熔断词样例。
        返回: {"per_job_skill_count": [...], "skills_before_melt": N, "skills_after_melt": M, "melted_terms_sample": [...]}
        """
        if not job_ids or not self.graph or total_j <= 0:
            return {}
        try:
            # 每个岗位的 REQUIRE_SKILL 数量
            cursor = self.graph.run(
                """MATCH (j:Job) WHERE j.id IN $j_ids
                   MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
                   WITH j.id AS jid, count(v) AS skill_count
                   RETURN jid, skill_count ORDER BY jid""",
                j_ids=job_ids[:20]
            )
            per_job = [{'jid': r['jid'], 'skill_count': r['skill_count']} for r in cursor]

            # 所有技能及 cov_j（不应用 3% 熔断），用于统计熔断前后
            cypher_all = """
            MATCH (j:Job) WHERE j.id IN $j_ids
            MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
            WITH v, (COUNT { (v)<-[:REQUIRE_SKILL]-() } * 1.0 / $total_j) AS cov_j
            RETURN v.id AS vid, v.term AS term, cov_j
            """
            rows = self.graph.run(cypher_all, j_ids=job_ids[:20], total_j=total_j).data()
            before_melt = len(rows)
            after_melt = len([r for r in rows if r['cov_j'] < 0.03 and len((r.get('term') or '')) > 1])
            melted = [r['term'] for r in rows if r['cov_j'] >= 0.03][:20]
            return {
                'per_job_skill_count': per_job,
                'skills_before_melt': before_melt,
                'skills_after_melt': after_melt,
                'melted_terms_sample': melted
            }
        except Exception:
            return {}

    def _clean_job_skills(self, skills_text):
        """
        【Step 1 技能清洗】从岗位技能文本中提取并过滤，去除 HR 描述词与泛词。
        规则：拆分后删除含「经验|竞赛|获奖|发表|能力|熟悉|了解」的片段、
        删除「不限」「其他」、删除长度 < 2，返回小写集合（与图谱 term 对齐）。
        """
        if not skills_text or not isinstance(skills_text, str):
            return set()
        stop_substrings = re.compile(r"经验|竞赛|获奖|发表|能力|熟悉|了解", re.I)
        forbidden = {"不限", "其他"}
        raw = re.split(r"[,，、；;|\s]+", skills_text)
        out = set()
        for s in raw:
            t = s.strip()
            if not t or len(t) < 2:
                continue
            if stop_substrings.search(t) or t in forbidden:
                continue
            out.add(t.lower())
        return out

    # --- 第二阶段：锚点技能提取 ---
    def _extract_anchor_skills(self, target_job_ids):
        """
        【工业侧：岗位技能提取】先做技能清洗，再 3% 熔断（cov_j < 0.03）。
        逻辑：从命中岗位取 raw skills -> 清洗 -> 仅保留在图谱 REQUIRE_SKILL 中且 term 在清洗结果中的词，上限 50。
        """
        cleaned_terms = set()
        try:
            cursor = self.graph.run(
                "MATCH (j:Job) WHERE j.id IN $j_ids RETURN j.skills AS skills",
                j_ids=target_job_ids[:20]
            )
            for row in cursor:
                if row.get("skills"):
                    cleaned_terms |= self._clean_job_skills(str(row["skills"]))
        except Exception:
            pass
        if not cleaned_terms:
            cleaned_terms = None  # 无清洗结果时不按词过滤，退化为原逻辑

        cypher = """
        MATCH (j:Job) WHERE j.id IN $j_ids
        MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
        WITH v, (COUNT { (v)<-[:REQUIRE_SKILL]-() } * 1.0 / $total_j) AS cov_j
        WHERE cov_j < 0.03
        WITH v.id AS vid, v.term AS term, cov_j
        RETURN DISTINCT vid, term, cov_j
        ORDER BY cov_j ASC
        LIMIT 50
        """
        cursor = self.graph.run(cypher, j_ids=target_job_ids[:20], total_j=self.total_job_count)
        rows = [r for r in cursor if r["term"] and len(r["term"]) > 1]
        if cleaned_terms is not None:
            rows = [r for r in rows if (r["term"] or "").lower() in cleaned_terms]
        return {str(r["vid"]): {"term": r["term"]} for r in rows}
    # --- 第三阶段：语义扩展 ---
    def _expand_semantic_map(self, core_vids, anchor_skills, domain_regex=None, query_vector=None):
        """
        【学术共鸣 + 共现领域版】语义扩展引擎。

        逻辑：工业锚点激发 -> 学术候选池生成 -> 学术共鸣(resonance)与共现领域指标(cooc_span/cooc_purity) ->
              最终打分。输入输出格式不变：仍返回 (score_map, term_map, idf_map)，key 为 tid 字符串。
        """
        regex = domain_regex if domain_regex else ".*"

        # 1. 辅助函数 A：执行图检索，获取初步的学术词候选（tid 列表）
        # 这一步通过 SIMILAR_TO 建立物理通路
        raw_results = self._query_expansion_with_topology(core_vids, regex)
        if not raw_results:
            return {}, {}, {}

        # --- 2. 【学术共鸣】候选词与本次要搜索的词汇的共现（单词协作）---
        tids = [r['tid'] for r in raw_results]
        resonance_map = self._calculate_academic_resonance(tids)
        for rec in raw_results:
            rec['resonance'] = resonance_map.get(rec['tid'], 0.0)

        # --- 2.5 【锚点共鸣】与“第一层学术词”的共现（工业词与学术词无论文共现，故用第一层学术词做参考）
        # 第一层学术词 = 本轮扩展结果；取 hit_count>=2 的作为“核心第一层”（多锚点共识），若无则用全部第一层
        first_layer_core = [r['tid'] for r in raw_results if r.get('hit_count', 0) >= 2]
        if not first_layer_core:
            first_layer_core = tids
        anchor_resonance_map = self._calculate_anchor_resonance(tids, first_layer_core)
        for rec in raw_results:
            rec['anchor_resonance'] = anchor_resonance_map.get(rec['tid'], 0.0)

        # --- 3. 【共现领域指标】为“万金油降权”与“领域专精加权”提供 cooc_span / cooc_purity ---
        # 从 domain_regex 解析出目标领域 ID 集合（与 _query_expansion_with_topology 内一致）
        active_domain_ids = set(re.findall(r'\d+', regex)) if regex and regex != ".*" else set()
        cooc_metrics = self._get_cooccurrence_domain_metrics(raw_results, active_domain_ids)
        for rec in raw_results:
            tid_key = str(rec['tid'])
            rec['cooc_span'] = cooc_metrics.get(tid_key, {}).get('cooc_span', 0.0)
            rec['cooc_purity'] = cooc_metrics.get(tid_key, {}).get('cooc_purity', 0.0)

        # 4. 应用数学公式计算最终动态权重（含共鸣、共现广度惩罚、共现纯度奖励）
        self._last_expansion_raw_results = raw_results
        return self._calculate_final_weights(raw_results, query_vector)

    def _calculate_academic_resonance(self, tids):
        """
        【学术逻辑层】计算候选词集内部的连通密度（与本次要搜索的词汇的共现 = 单词协作）。
        逻辑：如果 MPC 和 WBC 都在候选名单里，且它们在图谱中有强共现边，则两者都会获得“共鸣加成”。
        输出：{vid: resonance_score}，供下游 convergence_bonus 加权。
        """
        cypher = """
        MATCH (v1:Vocabulary)-[r:CO_OCCURRED_WITH]-(v2:Vocabulary)
        WHERE v1.id IN $tids AND v2.id IN $tids
        RETURN v1.id AS vid, SUM(r.weight) AS resonance_score
        """
        results = self.graph.run(cypher, tids=tids).data()
        return {r['vid']: float(r['resonance_score']) for r in results}

    def _calculate_anchor_resonance(self, tids, first_layer_tids):
        """
        【锚点共鸣】计算每个候选学术词与“第一层学术词”（first_layer_tids）在论文中的 CO_OCCURRED_WITH 权重之和。
        工业词与学术词在论文中无共现，故用第一层学术词（由锚点 SIMILAR_TO 得到）做共现参考；与核心第一层无共现的扩展词给 0.1 惩罚。
        输出：{tid: anchor_resonance_score}。
        """
        if not first_layer_tids:
            return {tid: 0.0 for tid in tids}
        cypher = """
        MATCH (v1:Vocabulary)-[r:CO_OCCURRED_WITH]-(v2:Vocabulary)
        WHERE v1.id IN $tids AND v2.id IN $first_layer_tids
        RETURN v1.id AS vid, SUM(r.weight) AS anchor_resonance_score
        """
        try:
            results = self.graph.run(cypher, tids=tids, first_layer_tids=first_layer_tids).data()
            return {r['vid']: float(r['anchor_resonance_score']) for r in results}
        except Exception:
            return {tid: 0.0 for tid in tids}

    def _get_cooccurrence_domain_metrics(self, raw_results, active_domain_ids):
        """
        【共现领域指标】从 vocab_stats.db 计算每个候选词的两项指标，
        用于后续“万金油降权”与“领域专精加权”，不改变输入/输出格式，仅向 raw_results 的调用方提供可注入的数值。

        指标含义：
          - cooc_span：共现伙伴的（按共现频次加权的）平均领域跨度 domain_span。
            越大表示该词常与“跨很多领域”的词一起出现 → 万金油 → 下游做降权。
          - cooc_purity：共现伙伴在目标领域上的（按共现频次加权的）论文占比。
            越高表示该词常与“只在本领域出现”的词共现 → 领域专精 → 下游做加权。

        数据来源：
          - cooc_purity：优先从 vocabulary_cooc_domain_ratio(voc_id, domain_id, ratio) 按目标领域 SUM(ratio) 查表，无表时回退到共现+领域统计计算。
          - cooc_span：vocabulary_cooccurrence + vocabulary_domain_stats（伙伴的 domain_span 按 freq 加权）。

        输入：
          - raw_results：语义扩展得到的候选列表，每项至少含 'tid', 'term'。
          - active_domain_ids：当前搜索的目标领域 ID 集合（如 {"1","4"}），用于计算 cooc_purity 的目标领域占比。

        输出：{tid_str: {"cooc_span": float, "cooc_purity": float}}，无共现或表不存在时对应项为 0。
        """
        if not raw_results or not active_domain_ids:
            return {str(rec['tid']): {"cooc_span": 0.0, "cooc_purity": 0.0} for rec in raw_results}

        # 优先从预计算的 vocabulary_cooc_domain_ratio 取 cooc_purity（按目标领域 SUM(ratio)）
        cooc_purity_from_table = {}
        try:
            tids = [rec["tid"] for rec in raw_results]
            domain_list = [str(d) for d in active_domain_ids]
            if tids and domain_list:
                ph_t = ",".join("?" * len(tids))
                ph_d = ",".join("?" * len(domain_list))
                rows = self.stats_conn.execute(
                    f"SELECT voc_id, SUM(ratio) AS cooc_purity FROM vocabulary_cooc_domain_ratio WHERE voc_id IN ({ph_t}) AND domain_id IN ({ph_d}) GROUP BY voc_id",
                    tids + domain_list,
                ).fetchall()
                cooc_purity_from_table = {str(r[0]): float(r[1]) for r in rows}
        except Exception:
            pass

        try:
            terms = list({rec['term'] for rec in raw_results})
            terms_set = set(terms)
            # 1. 从 vocabulary_cooccurrence 查出候选词各自的共现伙伴及频次
            placeholders = ','.join('?' * len(terms))
            sql_cooc = (
                f"SELECT term_a, term_b, freq FROM vocabulary_cooccurrence "
                f"WHERE term_a IN ({placeholders}) OR term_b IN ({placeholders})"
            )
            rows = self.stats_conn.execute(sql_cooc, terms + terms).fetchall()

            # 2. 构建 候选 term -> [(partner_term, freq), ...]
            term_to_partners = collections.defaultdict(list)
            for term_a, term_b, freq in rows:
                if term_a in terms_set:
                    term_to_partners[term_a].append((term_b, freq))
                if term_b in terms_set:
                    term_to_partners[term_b].append((term_a, freq))

            partner_terms = set()
            for pairs in term_to_partners.values():
                for p, _ in pairs:
                    partner_terms.add(p)
            default_out = {str(rec["tid"]): {"cooc_span": 0.0, "cooc_purity": cooc_purity_from_table.get(str(rec["tid"]), 0.0)} for rec in raw_results}
            if not partner_terms:
                return default_out

            # 3. 主库：伙伴 term -> voc_id
            partner_list = list(partner_terms)
            ph = ','.join('?' * len(partner_list))
            with sqlite3.connect(DB_PATH) as main_conn:
                main_conn.row_factory = sqlite3.Row
                main_rows = main_conn.execute(
                    f"SELECT voc_id, term FROM vocabulary WHERE term IN ({ph})", partner_list
                ).fetchall()
            partner_term_to_vocid = {row['term']: row['voc_id'] for row in main_rows}

            partner_voc_ids = list(partner_term_to_vocid.values())
            if not partner_voc_ids:
                return default_out

            # 4. vocabulary_domain_stats：voc_id -> (work_count, domain_span, domain_dist)
            ph2 = ','.join('?' * len(partner_voc_ids))
            stats_rows = self.stats_conn.execute(
                f"SELECT voc_id, work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id IN ({ph2})",
                partner_voc_ids,
            ).fetchall()
            vocid_to_stats = {}
            for r in stats_rows:
                vocid_to_stats[r[0]] = (r[1], r[2], r[3])  # work_count, domain_span, domain_dist

            # 5. 对每个候选词计算按共现频次加权的 cooc_span 与 cooc_purity
            out = {}
            for rec in raw_results:
                tid, term = rec['tid'], rec['term']
                pairs = term_to_partners.get(term, [])
                cooc_span_sum = cooc_purity_sum = total_freq = 0.0
                for partner_term, freq in pairs:
                    voc_id = partner_term_to_vocid.get(partner_term)
                    if voc_id is None:
                        continue
                    st = vocid_to_stats.get(voc_id)
                    if not st:
                        continue
                    work_count, domain_span, dist_json = st
                    try:
                        dist = json.loads(dist_json) if isinstance(dist_json, str) else dist_json
                    except (TypeError, ValueError):
                        dist = {}
                    target_degree = sum(dist.get(str(d), 0) for d in active_domain_ids)
                    target_ratio = (target_degree / work_count) if work_count else 0.0
                    cooc_span_sum += domain_span * freq
                    cooc_purity_sum += target_ratio * freq
                    total_freq += freq
                if total_freq > 0:
                    out[str(tid)] = {
                        "cooc_span": cooc_span_sum / total_freq,
                        "cooc_purity": cooc_purity_from_table.get(str(tid), cooc_purity_sum / total_freq),
                    }
                else:
                    out[str(tid)] = {
                        "cooc_span": 0.0,
                        "cooc_purity": cooc_purity_from_table.get(str(tid), 0.0),
                    }
            return out
        except Exception:
            # 共现表或主库不可用时不改变行为，cooc_span 置 0，cooc_purity 用表数据若有
            return {str(rec["tid"]): {"cooc_span": 0.0, "cooc_purity": cooc_purity_from_table.get(str(rec["tid"]), 0.0)} for rec in raw_results}

    def _query_expansion_with_topology(self, v_ids, regex):
        """
        【改造版】SIMILAR_TO 每锚点 top-K、权重下限 + 学术词 paper_count 上限。
        逻辑：每锚点取 top3 且 score>=0.65 -> 按 tid 聚合 max(sim)、hit_count ->
              vocabulary_domain_stats 过滤 work_count<=277，保留 target_degree_w>0，并写入 sim_score。
        """
        if not v_ids:
            return []
        active_domains = set(re.findall(r'\d+', regex))
        params = {
            "v_ids": list(v_ids),
            "min_score": SIMILAR_TO_MIN_SCORE,
            "top_k": SIMILAR_TO_TOP_K,
        }
        cypher = """
        UNWIND $v_ids AS vid
        MATCH (v:Vocabulary {id: vid})-[r:SIMILAR_TO]->(v_rel:Vocabulary)
        WHERE r.score >= $min_score
        WITH vid, v_rel.id AS tid, v_rel.term AS term, r.score AS sim_score
        ORDER BY vid, sim_score DESC
        WITH vid, collect({tid: tid, term: term, sim_score: sim_score})[0..$top_k] AS top3
        UNWIND top3 AS c
        RETURN c.tid AS tid, c.term AS term, c.sim_score AS sim_score
        """
        rows = self.graph.run(cypher, **params).data()
        if not rows:
            return []

        # 按 tid 聚合：取 max(sim_score)，hit_count = 被多少锚点命中
        by_tid = {}
        for r in rows:
            tid = r["tid"]
            term = r["term"] or ""
            sim = float(r["sim_score"])
            if tid not in by_tid:
                by_tid[tid] = {"tid": tid, "term": term, "sim_score": sim, "hit_count": 0}
            by_tid[tid]["sim_score"] = max(by_tid[tid]["sim_score"], sim)
            by_tid[tid]["hit_count"] += 1

        tids = list(by_tid.keys())
        results = []
        for tid in tids:
            row = self.stats_conn.execute(
                "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (tid,),
            ).fetchone()
            if not row:
                continue
            degree_w, domain_span, dist_json = row
            if degree_w > VOCAB_P95_PAPER_COUNT:
                continue
            try:
                dist = json.loads(dist_json) if isinstance(dist_json, str) else dist_json
            except (TypeError, ValueError):
                dist = {}
            target_degree_w = sum(dist.get(str(d), 0) for d in active_domains)
            if target_degree_w <= 0:
                continue
            rec = by_tid[tid]
            rec["degree_w"] = degree_w
            rec["target_degree_w"] = target_degree_w
            rec["domain_span"] = domain_span
            rec["cov_j"] = 0.0
            results.append(rec)
        return results

    def _calculate_final_weights(self, raw_results, query_vector):
        """
        【收敛加权版】权重计算调度器。
        若扩展结果含 sim_score（改造版 SIMILAR_TO + paper_count 过滤），则用 score = sim / log(1+paper_count)。
        """
        score_map, term_map, idf_map = {}, {}, {}

        for rec in raw_results:
            tid = str(rec["tid"])
            if "sim_score" in rec and "degree_w" in rec:
                # 改造版：score = sim / log(1 + paper_count)，惩罚泛词
                degree_w = rec["degree_w"]
                sim_score = rec["sim_score"]
                dynamic_weight = sim_score / math.log(1.0 + degree_w)
                idf_val = math.log10(self.total_work_count / (degree_w + 1))
            else:
                dynamic_weight, idf_val = self._apply_word_quality_penalty(rec, query_vector)
            score_map[tid] = dynamic_weight
            term_map[tid] = rec.get("term") or ""
            idf_map[tid] = idf_val

        return score_map, term_map, idf_map

    def _apply_word_quality_penalty(self, rec, query_vector):
        """
        【学术共鸣 + 共现领域 + 语义拦截】核心数学引擎。

        降噪维度：语义守门员(SBERT) + 领域跨度(Span) + 领域纯度(Purity) + 共鸣熔断(Resonance)
                 + 共现领域广度(cooc_span)：与多领域词都共现 → 万金油降权。
        加成维度：IDF + 技术共振(Hit Count) + 学术共鸣(Resonance) + 共现目标领域纯度(cooc_purity)：专精加权。
        输入 rec 需含 degree_w, cov_j, domain_span, target_degree_w, hit_count, resonance；可选 cooc_span, cooc_purity。
        输出 (dynamic_weight, idf_val) 格式不变。
        """
        degree_w = rec['degree_w']
        cov_j = rec['cov_j']
        domain_span = max(1, rec['domain_span'])

        # 1. 计算语义纯度 (Tag Purity)
        # 逻辑：目标领域的产出占比，过滤“挂羊头卖狗肉”的词汇
        tag_purity = rec['target_degree_w'] / degree_w

        # 2. 计算基础学术稀缺度 (IDF)
        idf_val = math.log10(self.total_work_count / (degree_w + 1))

        # 3. 计算断崖式岗位惩罚 (Suppression of generic job terms)
        job_penalty = 1.0 + math.exp(300.0 * (cov_j - 0.005))

        # --- 4. 【核心修改：学术共鸣与共振奖励 + hit_count 直接加权】 ---
        # 逻辑 A：hit_count 代表有多少工业锚点支撑这个词；单锚点(hit_count=1)多为 python 等扩出的噪音，直接降权
        hit_count = rec.get('hit_count', 1)
        hit_count_factor = 0.4 if hit_count == 1 else 1.0  # 单锚点 0.4x，多锚点共识保持 1.0

        # 逻辑 B：resonance 代表该词与当前其他候选词的共现；anchor_resonance 代表与第一层学术词的共现
        resonance = rec.get('resonance', 0.0)
        anchor_resonance = rec.get('anchor_resonance', 0.0)

        # 【共鸣熔断器】：必须与第一层学术词有共现才给满分，否则 0.1 惩罚
        if anchor_resonance > 0:
            resonance_factor = 1.0 + math.log1p(resonance)
        else:
            resonance_factor = 0.1

        # 综合收敛奖惩：技术簇共鸣 + hit_count 直接加权（多锚点共识词得分更高）
        convergence_bonus = hit_count_factor * math.log1p(hit_count) * resonance_factor

        # --- 5. SBERT 语义守门员 ---
        # 逻辑：计算学术词向量与当前 JD 整体语义的余弦相似度
        cos_sim = 0.5
        idx = self.vocab_to_idx.get(str(rec['tid']))
        if idx is not None and query_vector is not None:
            term_vec = self.all_vocab_vectors[idx]
            cos_sim = float(np.dot(term_vec, query_vector.flatten()))

        # 应用 6 次方非线性惩罚，实现对弱相关词的断崖式拦截
        semantic_factor = math.pow(max(0, cos_sim), 6)

        # --- 6. 【共现领域惩罚与奖励】基于 vocab_stats 的 vocabulary_cooccurrence ---
        # 降权：与各种领域的词都共现 → 万金油 → 共现伙伴平均领域跨度大 → cooc_span 大 → 乘小于 1 的因子
        cooc_span = rec.get('cooc_span', 0.0)
        span_penalty = 1.0 / (1.0 + math.log1p(cooc_span)) if cooc_span > 0 else 1.0
        # 加权：只跟特定领域的词共现 → 专精 → 共现伙伴目标领域占比高 → cooc_purity 大 → 乘大于 1 的因子
        cooc_purity = rec.get('cooc_purity', 0.0)
        purity_bonus = (1.0 + math.log1p(cooc_purity)) if cooc_purity > 0 else 1.0

        # 7. 【最终公式：立体的评价体系】
        # 在原有公式上再乘 span_penalty（万金油降权）与 purity_bonus（领域专精加权），输入输出格式不变
        dynamic_weight = (
                (idf_val / job_penalty)
                * math.pow(tag_purity, 4)  # 纯度 4 次方，整体压泛词（跨多领域词得分断崖式下降）
                / domain_span  # 跨度降权压制万金油
                * convergence_bonus  # 学术共鸣（与本次搜索词共现）加成
                * semantic_factor  # SBERT 语义守门员
                * span_penalty  # 共现伙伴跨多领域 → 降权
                * purity_bonus  # 共现伙伴集中目标领域 → 加权
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

    def recall(self, query_vector, domain_id=None, query_text=None):
        """
        全链路调度：领域探测 → 锚点提取 → 语义扩展（含共现领域指标）→ 图谱反查 → 作者打分。
        入参 domain_id 与向量路统一；工业词 3% 熔断（含查询词）；学术词 2% 熔断。返回 (author_id_list, duration_ms)。
        """
        if not self.graph: return [], 0
        start_t = time.time()

        # 1. 岗位空间探测：探测当前 JD 所属的主战场领域
        job_ids, inferred_domains, dominance = self._detect_domain_context(query_vector)
        job_previews = self._get_job_previews(job_ids)

        # 2. 锚点提取：工业词 3% 熔断（对查询词也熔断，python 等高频词会被熔掉）
        anchor_skills = self._extract_anchor_skills(job_ids)
        anchor_debug = self._get_anchor_debug_stats(job_ids[:20], self.total_job_count) if job_ids else {}
        if not anchor_skills:
            return [], 0
        industrial_kws = [v['term'] for v in anchor_skills.values()]

        # 3. 领域处理：确定最终过滤范围（正则字符串），domain_id 与向量路命名统一
        active_domain_set = DomainProcessor.to_set(
            domain_id if domain_id and str(domain_id) != "0" else inferred_domains)
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
        WHERE (degree_w * 1.0 / $total_w) < 0.02 
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
            'decay_rate': DOMAIN_DECAY_RATES.get(first_domain, DEFAULT_DECAY_RATE)
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
        # 学术词按权重排序，供主函数打印 Top15
        sorted_terms = sorted(
            [(term_map.get(tid, ''), score_map.get(tid, 0)) for tid in score_map],
            key=lambda x: x[1], reverse=True
        )
        self.last_debug_info = {
            'active_domains': list(active_domain_set),
            'dominance': f"{dominance * 100:.1f}%",
            'job_ids': job_ids,
            'job_previews': job_previews,
            'anchor_debug': anchor_debug,
            'industrial_kws': industrial_kws,
            'anchor_detail': [f"{k}={v['term']}" for k, v in anchor_skills.items()],
            'academic_kws': academic_kws,
            'detailed_kws': getattr(self, '_last_expansion_raw_results', []),
            'top_scored_terms': sorted_terms,
            'recall_vocab_count': len(score_map),
            'work_count': all_works_count,
            'author_count': len(scored_authors),
            'top_samples': scored_authors[:20]
        }

        return [a['aid'] for a in scored_authors[:self.recall_limit]], (time.time() - start_t) * 1000


if __name__ == "__main__":
    l_path = LabelRecallPath(recall_limit=200)
    encoder = QueryEncoder()

    try:
        domain_choice = input("\n请选择领域编号 (0跳过): ").strip() or "0"

        while True:
            user_input = input(f"\n请输入岗位需求 (q退出): ").strip()
            if not user_input or user_input.lower() == 'q':
                break

            query_vec, _ = encoder.encode(user_input)
            faiss.normalize_L2(query_vec)

            top_ids, search_time = l_path.recall(query_vec, domain_id=domain_choice)

            # --- 核心诊断日志 ---
            db = l_path.last_debug_info
            print("\n" + "🔍 [深度诊断流水线]" + "-" * 98)

            # 1. 领域探测（增强：打印命中的岗位 ID + Top10 岗位名称与描述片段）
            domains = db.get('active_domains', [])
            domain_str = " | ".join(domains) if domains else "未限制"
            print(f"【Step 1: 领域探测】目标领域并集: [{domain_str}] (置信度: {db.get('dominance')})")
            job_ids = db.get('job_ids', [])
            if job_ids:
                print(f"      命中岗位 ID (Top20): {[jid[:50]+'...' if len(jid)>50 else jid for jid in job_ids[:20]]}")
            job_previews = db.get('job_previews', [])
            if job_previews:
                print(f"      Top20 岗位名称与描述片段（用于判断是否匹配）:")
                for i, jp in enumerate(job_previews[:10], 1):
                    name = (jp.get('name') or '')[:60]
                    snippet = (jp.get('description_snippet') or '')[:120]
                    print(f"        #{i} 名称: {name}")
                    print(f"            描述: {snippet}...")

            # 2. 工业锚点（增强：每个岗位 REQUIRE_SKILL 数、熔断前/后、被熔断词样例）
            i_kws = db.get('industrial_kws', [])
            anchor_detail = db.get('anchor_detail', [])
            print(f"【Step 2: 工业锚点】从 JD 提取的核心技能 (共 {len(i_kws)} 个): {i_kws}")
            if anchor_detail:
                print(f"      锚点明细 (vid -> term): {anchor_detail[:15]}")
            anchor_debug = db.get('anchor_debug', {})
            if anchor_debug:
                per_job = anchor_debug.get('per_job_skill_count', [])
                before_melt = anchor_debug.get('skills_before_melt', 0)
                after_melt = anchor_debug.get('skills_after_melt', 0)
                melted_sample = anchor_debug.get('melted_terms_sample', [])
                print(f"      参与锚点提取的岗位 (Top5) 每岗 REQUIRE_SKILL 数: {per_job}")
                print(f"      3% 熔断: 熔断前 {before_melt} 个技能词，熔断后保留 {after_melt} 个；被熔断词样例: {melted_sample[:15]}")

            # 3. 学术语义扩展与“学术共鸣”校验
            print(f"【Step 3: 语义扩展与实证校验】学术词质量评估:")
            print(f"      {'学术词 (Term)':<25} | {'收敛(Hits)':<10} | {'共鸣(Resonance)':<15} | {'状态'}")
            print(f"      {'-' * 25} | {'-' * 10} | {'-' * 15} | {'-' * 10}")

            detailed_kws = db.get('detailed_kws', [])
            if not detailed_kws:
                print("      （无数据）")
            for kw in sorted(detailed_kws, key=lambda x: x.get('resonance', 0), reverse=True)[:15]:
                hits = kw.get('hit_count', 1)
                res = kw.get('resonance', 0)
                status = "✅ 核心簇" if res > 0 and hits > 1 else "⚠️ 孤立点"
                if res == 0:
                    status = "❌ 已熔断"
                term_str = (kw.get('term') or '')[:25]
                print(f"      {term_str:<25} | {hits:<10} | {int(res):<15} | {status}")

            # 3.5 参与召回的学术词及权重 Top15（便于发现 stub file、folklore 等噪音）
            top_scored_terms = db.get('top_scored_terms', [])
            if top_scored_terms:
                print(f"      参与召回的学术词权重 Top15:")
                for term, score in top_scored_terms[:15]:
                    print(f"        - {term[:40]:<40}  weight={score:.6f}")

            # 4. 论文与作者召回
            w_count = db.get('work_count', 0)
            a_count = db.get('author_count', 0)
            vocab_count = db.get('recall_vocab_count', 0)
            if vocab_count:
                print(f"【Step 4: 召回规模】参与检索的学术词数: {vocab_count}，检索到 {w_count} 篇学术论文，最终锁定 {a_count} 名垂直领域专家。")
            else:
                print(f"【Step 4: 召回规模】检索到 {w_count} 篇学术论文，最终锁定 {a_count} 名垂直领域专家。")

            # --- 专家排名展示（增强：前几条打印原始得分 + 代表作详情）---
            print("-" * 115)
            print(f"{'排名':<6} | {'作者 ID':<12} | {'综合得分':<15} | {'学术领域代表作 (命中标签)'}")
            print("-" * 115)

            for i, item in enumerate(db.get('top_samples', []), 1):
                tp = item.get('top_paper', {})
                title = (tp.get('title') or 'Unknown')[:55]
                hit_tags = ", ".join(tp.get('hits', []))
                raw_score = item.get('score', 0)
                contrib = tp.get('contribution', 0)
                print(f"#{i:<5} | {item['aid']:<12} | {int(raw_score):<15} | 《{title}》")
                print(f"{' ':23} ┗━ 核心命中: {hit_tags}")
                if i <= 5:
                    print(f"{' ':23}     [调试] 原始得分={raw_score:.6f}, 代表作贡献={contrib:.6f}")

            print("-" * 115)
            print(f"[*] 诊断完成。全链路耗时: {search_time:.2f}ms")

    except Exception as e:
        print(f"运行出错: {e}")
        traceback.print_exc()