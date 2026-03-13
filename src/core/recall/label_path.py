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
import sqlite3
import time
import math
import collections
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set
from datetime import datetime
from src.core.recall.input_to_vector import QueryEncoder
from src.core.recall.works_to_authors import accumulate_author_scores
from src.utils.domain_utils import DomainProcessor
from src.utils.domain_detector import DomainDetector
from config import (
    DB_PATH, VOCAB_P95_PAPER_COUNT, SIMILAR_TO_TOP_K, SIMILAR_TO_MIN_SCORE,
)
from src.utils.domain_config import (
    DOMAIN_MAP,
)
from src.utils.tools import get_decay_rate_for_domains
from src.utils.time_features import (
    compute_paper_recency,
    compute_author_time_features,
    compute_author_recency_by_latest
)
from src.core.recall.label_means.simple_factors import (
    survey_decay_factor,
    coverage_norm_factor,
    paper_cluster_bonus,
    paper_jd_semantic_gate_factor,
)
from src.core.recall.label_means import advanced_metrics as label_means_adv, label_anchors, label_expansion
from src.core.recall.label_means.infra import LabelMeansInfra
from src.core.recall.label_pipeline import (
    stage1_domain_anchors,
    stage2_expansion,
    stage3_term_filtering,
    stage4_paper_recall,
    stage5_author_rank,
)


@dataclass
class Stage1Result:
    """
    阶段 1 结构化结果壳，用于逐步解耦领域与锚点阶段的中间状态。
    当前仅在内部存储与调试使用，不改变外部调用签名。
    """
    active_domains: Set[int]
    domain_regex: str
    anchor_skills: Dict[Any, Any]
    job_ids: List[int]
    job_previews: List[Dict[str, Any]]
    dominance: float
    anchor_debug: Dict[str, Any]


@dataclass
class RecallContext:
    """
    全链路召回上下文壳，后续供 label_means 等模块统一访问查询级别信息。
    目前仅作为占位结构体存在，不改变已有评分逻辑的数据结构类型。
    """
    query_text: str
    query_vector: Any
    active_domains: Set[int]
    dominance: float
    term_score_map: Dict[str, float] = field(default_factory=dict)
    term_meta_map: Dict[str, Any] = field(default_factory=dict)


class RecallDebugInfo:
    """
    统一存放调试相关的中间状态，收拢原本散落在类上的 _last_* 字段。
    第一层拆解阶段仅做聚合，不改变调试信息的具体内容与使用方式。
    """

    def __init__(self) -> None:
        # 锚点熔断与语义重排过程的统计信息
        self.anchor_melt_stats: Dict[str, Any] = {}
        # JD 向量补充锚点的明细与报告
        self.supplement_anchors: List[Any] = []
        self.supplement_anchors_report: List[Dict[str, Any]] = []
        # 语义扩展阶段的原始候选与流水线统计
        self.expansion_raw_results: List[Dict[str, Any]] = []
        self.expansion_pipeline_stats: Dict[str, Any] = {}
        # SIMILAR_TO 各阶段调试信息
        self.similar_to_raw_rows: List[Dict[str, Any]] = []
        self.similar_to_agg: List[Dict[str, Any]] = []
        self.similar_to_pass: List[Dict[str, Any]] = []
        self.raw_candidate_tids: List[int] = []
        # 词级标签纯度与簇衰减调试信息
        self.tag_purity_debug: List[Dict[str, Any]] = []
        self.cluster_rank_factors: Dict[str, float] = {}
        self.cluster_expansion_log: List[Dict[str, Any]] = []


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

    # 领域探测：检索岗位数、候选领域数、最终领域数
    DETECT_JOBS_TOP_K = 20
    CANDIDATE_DOMAINS_TOP_K = 5
    ACTIVE_DOMAINS_TOP_K = 3
    # 锚点提取：先熔断(cov_j<3%) -> 频次 Top30 -> 语义重排 Top20；核心词补充：JD 向量搜 vocabulary TopK
    ANCHOR_JOBS_TOP_K = 20
    ANCHOR_FREQ_TOP_K = 30
    ANCHOR_FINAL_TOP_K = 20
    ANCHOR_MELT_COV_J = 0.03  # 熔断：全图岗位覆盖率 >= 3% 的技能不参与锚点
    JD_VOCAB_TOP_K = 20       # 用 JD 向量直接搜 vocabulary 的 top-K 作为补充锚点
    ANCHOR_SIM_MIN = 0.4      # 锚点语义重排的最小相似度阈值（低于直接丢弃）
    ANCHOR_MIN_JOB_FREQ = 2   # 锚点共识门槛：至少出现在 Top20 命中岗位中的 2 个岗位
    # 无硬编码过滤：term 与任意锚点的最大余弦相似度低于此则降权（防 ML 簇劫持）
    ANCHOR_TERM_SIM_MIN = 0.45
    # 作者过滤：最佳论文贡献低于全局最大论文贡献的此比例则不出现在排序中
    AUTHOR_BEST_PAPER_MIN_RATIO = 0.05
    # Step3 词权重：语义次方、锚点系数、span 惩罚指数（(1+domain_span)^exp 做分母，折中压制泛词）
    SEMANTIC_POWER = 3
    ANCHOR_BASE = 0.35
    ANCHOR_GAIN = 0.65
    SPAN_PENALTY_EXPONENT = 0.35

    def __init__(self, recall_limit=200, verbose=False):
        self.recall_limit = recall_limit
        self.verbose = verbose
        self.current_year = datetime.now().year
        # 统一调试信息容器：收拢原本散落的 _last_* 状态
        self.debug_info = RecallDebugInfo()
        # 底层资源由 label_means.infra 统一管理
        self.infra = LabelMeansInfra()
        self.infra.init_resources()

        # 将 Infra 中的资源别名到实例属性，尽量不改动下游逻辑
        self.graph = self.infra.graph
        self.job_index = self.infra.job_index
        self.job_id_map = self.infra.job_id_map
        self.vocab_index = self.infra.vocab_index
        self.all_vocab_vectors = self.infra.all_vocab_vectors
        self.vocab_to_idx = self.infra.vocab_to_idx
        self.stats_conn = self.infra.stats_conn
        self.cluster_members = self.infra.cluster_members
        self.voc_to_clusters = self.infra.voc_to_clusters
        self.cluster_centroids = self.infra.cluster_centroids

        # 预载入统计数据，用于计算后续 IDF 与 熔断率
        self.total_work_count = self.infra.get_node_count("Work")
        self.total_job_count = self.infra.get_node_count("Job")

        # 编码器单例：全流程共用，避免重复加载模型（领域向量、锚点补充、语境扩展、main 均复用）
        self._query_encoder = QueryEncoder()
        # 领域向量：用于从 Top5 候选领域中按 query 相似度选 Top3
        self.domain_vectors = {}
        self._build_domain_vectors()

        # 统一领域探测组件：基于 Job 索引 + Neo4j
        try:
            self.domain_detector = DomainDetector(
                label_path=None,
                graph=self.graph,
                job_index=self.job_index,
                job_id_map=self.job_id_map,
                detect_jobs_top_k=self.DETECT_JOBS_TOP_K,
                candidate_domains_top_k=self.CANDIDATE_DOMAINS_TOP_K,
                active_domains_top_k=self.ACTIVE_DOMAINS_TOP_K,
            )
        except Exception:
            self.domain_detector = None

    def _compute_cluster_task_factors(self, query_vector):
        """
        基于当前 JD 的 query_vector 和簇中心向量，计算 cluster_task_factor。
        簇只做轻微微调：raw = sim^1.5 后线性映射到 [0.9, 1.0]，不承担主区分力。
        """
        # 若簇中心或 query 向量不可用，则关闭簇 gating
        if getattr(self, "cluster_centroids", None) is None or query_vector is None:
            self._cluster_task_factors = {}
            return

        q = np.asarray(query_vector, dtype=np.float32).flatten()
        if q.size == 0:
            self._cluster_task_factors = {}
            return

        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            self._cluster_task_factors = {}
            return
        q = q / q_norm

        sims = np.dot(self.cluster_centroids, q)  # shape = (K,)
        sims = np.clip(sims, 0.0, 1.0)

        # 簇只做轻微微调：将 raw 映射到 [0.9, 1.0]，不承担主区分力
        factors = {}
        for cid, sim in enumerate(sims):
            raw = float(sim) ** 1.5
            factor = 0.9 + 0.1 * min(max(raw, 0.0), 1.0)
            factors[cid] = factor

        self._cluster_task_factors = factors

    def _get_cluster_factor_for_term(self, tid: int) -> float:
        """
        给定 term 的 voc_id，返回基于其所属簇的 cluster_task_factor：
          - 概念/keyword：取 score>=cutoff 的主簇；
          - industry 等其它：取 score>=cutoff 的 top2 按 score 加权平均；
          - 无簇/无 gating 信息：返回 1.0。
        """
        factors = getattr(self, "_cluster_task_factors", None)
        if not factors:
            return 1.0

        clusters = self.voc_to_clusters.get(int(tid))
        if not clusters:
            return 1.0

        cutoff = 0.1
        clusters = [(cid, sc) for cid, sc in clusters if sc >= cutoff]
        if not clusters:
            return 1.0

        et = None
        if hasattr(self, "_vocab_meta"):
            meta = self._vocab_meta.get(int(tid))
            if meta:
                et = (meta[1] or "").lower()

        # 学术概念/keyword：仅使用主簇
        if et in ("concept", "keyword"):
            cid, _ = max(clusters, key=lambda x: x[1])
            return float(factors.get(cid, 1.0))

        # 其它类型（如 industry）：对 top2 簇做 score 加权平均
        sorted_cs = sorted(clusters, key=lambda x: x[1], reverse=True)
        top = sorted_cs[:2]
        num = den = 0.0
        for cid, sc in top:
            f = float(factors.get(cid, 1.0))
            num += f * float(sc)
            den += float(sc)
        if den <= 0:
            return 1.0
        return num / den

    def _build_domain_vectors(self):
        """
        用领域中文名编码得到领域向量（与 QueryEncoder 同空间、已 L2 归一化），
        供 recall 时从 Top5 候选领域中按与 query 的余弦相似度选 Top3。
        使用 self._query_encoder 单例，不再在此处新建编码器。
        """
        try:
            encoder = self._query_encoder
            for domain_id, name in DOMAIN_MAP.items():
                vec, _ = encoder.encode(name)
                if vec is not None and vec.size > 0:
                    self.domain_vectors[str(domain_id)] = np.asarray(vec.flatten(), dtype=np.float32)
            if self.domain_vectors:
                print(f"[OK] 领域向量已构建 (共 {len(self.domain_vectors)} 个)")
        except Exception as e:
            print(f"[Warn] 领域向量构建失败: {e}，将退化为按候选领域顺序取前 3")

    # --- 第一阶段：环境与领域探测 ---
    def _detect_domain_context(self, query_vector):
        """
        【领域探测】通过用户 Query 在 Job 空间寻找最相关的行业分布
        逻辑：检索 Top20 相似岗位 -> 统计其 domain_ids -> 取 most_common(5) 作为候选领域；
              后续在 recall() 中再用 query 与 5 个领域向量算相似度取 Top3。
        """
        k = self.DETECT_JOBS_TOP_K
        _, indices = self.job_index.search(query_vector, k)
        candidate_ids = [self.job_id_map[idx] for idx in indices[0] if 0 <= idx < len(self.job_id_map)]

        domain_counter = collections.Counter()
        cursor = self.graph.run(
            "MATCH (j:Job) WHERE j.id IN $j_ids RETURN j.domain_ids AS d_ids",
            j_ids=candidate_ids
        )
        for row in cursor:
            if row['d_ids']:
                for d in DomainProcessor.to_set(row['d_ids']):
                    domain_counter[d] += 1

        # 候选领域 Top5，最终 Top3 在 recall() 中按 query–领域向量相似度选取
        n_candidate = self.CANDIDATE_DOMAINS_TOP_K
        inferred = [d for d, _ in domain_counter.most_common(n_candidate)]
        # dominance：主导领域在这 k 个岗位中的占比
        dominance = (domain_counter.most_common(1)[0][1] / float(k)) if domain_counter else 0
        return candidate_ids, inferred, dominance

    def _get_job_previews(self, job_ids, max_snippet=200):
        """
        查询命中岗位的名称与描述片段，用于诊断「Top20 是否真是目标领域岗位」。
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
        return label_anchors.clean_job_skills(skills_text)

    # --- 第二阶段：锚点技能提取（先熔断 -> Top30 -> Top20）---
    def _extract_anchor_skills(self, target_job_ids, query_vector=None, total_j=None):
        return label_anchors.extract_anchor_skills(self, target_job_ids, query_vector=query_vector, total_j=total_j)

    def _supplement_anchors_from_jd_vector(self, query_text, anchor_skills, total_j=None, top_k=None, active_domain_ids=None):
        return label_anchors.supplement_anchors_from_jd_vector(
            self,
            query_text,
            anchor_skills,
            total_j=total_j,
            top_k=top_k,
            active_domain_ids=active_domain_ids,
        )

    # --- 第三阶段：语义扩展 ---
    # 融合权重：相似边 0.4，语境向量 0.6；多次命中奖励：泛词(work_count大)不奖励，其余按 ln(min(hit,5)) 计算并封顶
    CTX_EDGE_WEIGHT = 0.6
    EDGE_WEIGHT = 0.4
    HIT_BONUS_BETA = 0.25
    HIT_BONUS_CAP = 1.5
    HIT_BONUS_HIT_CAP = 5
    HIT_BONUS_DEGREE_GATE = 200
    # 领域纯度阈值：目标领域产出占比下限（用于筛掉跨领域泛词）
    DOMAIN_PURITY_MIN = 0.5
    # ctx 路向量加权和参数：v_ctx = normalize(λ*v_jd + (1-λ)*v_term)
    CTX_MIX_LAMBDA = 0.7
    # 语义硬门槛：候选词与 query 的余弦相似度低于阈值时直接置 0（防止稀缺词以高 IDF 带偏）
    SEMANTIC_MIN = 0.4
    # 补充锚点入口控制：top-K、补充数量上限、允许的实体类型与领域纯度下限
    SUPPLEMENT_TOP_K = 10
    SUPPLEMENT_MAX_ADD = 6
    SUPPLEMENT_DOMAIN_RATIO_MIN = 0.6
    SUPPLEMENT_ALLOW_ENTITY_TYPES = {"concept", "keyword"}

    def _expand_semantic_map(self, core_vids, anchor_skills, domain_regex=None, query_vector=None, query_text=None, return_raw=False):
        return label_expansion.expand_semantic_map(
            self,
            core_vids=core_vids,
            anchor_skills=anchor_skills,
            domain_regex=domain_regex,
            query_vector=query_vector,
            query_text=query_text,
            return_raw=return_raw,
        )

    def _expand_with_clusters(self, raw_results, domain_regex, topk_per_seed=5, weight_decay=0.2):
        """
        使用概念簇对第一层学术词做局部扩展：
          1. 仅高置信 seed（按 sim_score 排序的前 N 个）才触发簇扩展。
          2. 对每个 seed 词，找到其所属簇（取 score 最高的一个）；簇内只保留 sim_in_cluster >= 0.6 的成员，再取 topk_per_seed 个。
          3. 扩展词的初始 sim_score 约为 seed_sim_score * sim_in_cluster * weight_decay。
        返回扩展后的 raw_results 列表，结构与原始列表一致（附加若干新 tid）。
        """
        # 若簇索引不可用，直接退化为原逻辑
        if not getattr(self, "cluster_members", None) or not getattr(self, "voc_to_clusters", None):
            return raw_results

        # 解析当前激活领域集合（与后续领域过滤保持一致）
        active_domain_ids = set(re.findall(r'\d+', domain_regex)) if domain_regex and domain_regex != ".*" else set()

        seed_vids = [int(rec["tid"]) for rec in raw_results]
        seed_vids_set = set(seed_vids)

        # seed -> 最优簇 (cluster_id, cluster_score)
        seed_to_cluster = {}
        for vid in seed_vids:
            clusters = self.voc_to_clusters.get(int(vid))
            if not clusters:
                continue
            # 取 score 最大的簇
            cid, cscore = max(clusters, key=lambda x: x[1])
            seed_to_cluster[int(vid)] = (cid, cscore)

        if not seed_to_cluster:
            return raw_results

        # 建立 seed -> sim_score 映射，若缺失则视为 1.0
        seed_sim_map = {}
        for rec in raw_results:
            try:
                vid = int(rec["tid"])
            except Exception:
                continue
            seed_sim_map[vid] = float(rec.get("sim_score", 1.0))

        # 仅高置信 seed 才触发簇扩展：按 sim_score 降序取前 N 个
        CLUSTER_EXPAND_TOP_SEEDS = 15
        sorted_by_sim = sorted(raw_results, key=lambda r: float(r.get("sim_score", 0.0)), reverse=True)
        allowed_seed_vids = {int(r["tid"]) for r in sorted_by_sim[:CLUSTER_EXPAND_TOP_SEEDS]}

        # 簇扩展来源表（用于日志 2）
        expansion_log = []

        # 聚合簇扩展出来的候选学术词
        # vid -> {"sim_score": float, "support": int, "seed_vids": set[int]}
        cluster_expanded = {}

        for rec in raw_results:
            try:
                vid = int(rec["tid"])
            except Exception:
                continue
            if vid not in seed_to_cluster:
                continue
            if vid not in allowed_seed_vids:
                continue

            cid, _ = seed_to_cluster[vid]
            members = self.cluster_members.get(int(cid)) or []
            if not members:
                continue

            # 排除自身与已在初始候选中的词
            candidates = [m for m in members if m not in seed_vids_set]
            if not candidates:
                continue

            seed_idx = self.vocab_to_idx.get(str(vid))
            if seed_idx is None:
                continue
            seed_vec = self.all_vocab_vectors[seed_idx]

            sims = []
            for m in candidates:
                midx = self.vocab_to_idx.get(str(m))
                if midx is None:
                    continue
                mvec = self.all_vocab_vectors[midx]
                sim_in_cluster = float(np.dot(seed_vec, mvec))
                sims.append((m, sim_in_cluster))

            # 只保留簇内相似度 ≥0.6 的成员，再取 topk
            CLUSTER_MIN_SIM = 0.6
            sims = [(m, s) for m, s in sims if s >= CLUSTER_MIN_SIM]
            if not sims:
                continue

            sims.sort(key=lambda x: x[1], reverse=True)
            top = sims[:topk_per_seed]
            seed_sim = seed_sim_map.get(vid, 1.0)

            seed_term = rec.get("term") or (self._vocab_meta.get(vid, ("", ""))[0] if getattr(self, "_vocab_meta", None) else "")
            for m, sim_in_cluster in top:
                contrib = weight_decay * seed_sim * sim_in_cluster
                if contrib <= 0:
                    continue
                entry = cluster_expanded.setdefault(
                    int(m), {"sim_score": 0.0, "support": 0, "seed_vids": set()}
                )
                entry["sim_score"] = max(entry["sim_score"], contrib)
                entry["support"] += 1
                entry["seed_vids"].add(int(vid))
                expansion_log.append({
                    "term_tid": int(m),
                    "seed_vid": vid,
                    "seed_term": seed_term or str(vid),
                    "sim_in_cluster": round(sim_in_cluster, 4),
                    "seed_sim": round(seed_sim, 4),
                    "contrib": round(contrib, 6),
                })

        if not cluster_expanded:
            return raw_results

        # 查扩展词的 term 与领域统计
        new_vids = [vid for vid in cluster_expanded.keys() if vid not in seed_vids_set]
        if not new_vids:
            return raw_results

        term_map = {}
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            ph = ",".join("?" * len(new_vids))
            # 仅保留 entity_type='concept' 的学术概念，过滤 industry 等工业实体
            rows = conn.execute(
                f"SELECT voc_id, term, entity_type FROM vocabulary WHERE voc_id IN ({ph})", new_vids
            ).fetchall()
            for r in rows:
                if (r["entity_type"] or "").lower() != "concept":
                    continue
                term_map[int(r["voc_id"])] = r["term"]

        stats_map = {}
        ph = ",".join("?" * len(new_vids))
        rows = self.stats_conn.execute(
            f"SELECT voc_id, work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id IN ({ph})",
            new_vids,
        ).fetchall()
        for r in rows:
            stats_map[int(r[0])] = (int(r[1]), int(r[2]), r[3])  # work_count, span, dist_json

        # 按与 _query_expansion_with_topology 相同的过滤逻辑进行领域过滤与熔断
        active_domains = set(active_domain_ids)
        for vid, agg in cluster_expanded.items():
            if vid not in term_map or vid not in stats_map:
                continue

            degree_w, domain_span, dist_json = stats_map[vid]
            if degree_w <= 0:
                continue

            try:
                dist = json.loads(dist_json) if isinstance(dist_json, str) else dist_json
            except (TypeError, ValueError):
                dist = {}
            expanded = self._expand_domain_dist(dist)
            degree_w_expanded = sum(expanded.values())

            if active_domains:
                target_degree_w = sum(expanded.get(str(d), 0) for d in active_domains)
            else:
                target_degree_w = degree_w_expanded

            # 领域纯度与大词惩罚：改为软上限（次方惩罚），不再直接过滤
            domain_ratio = target_degree_w / degree_w_expanded if degree_w_expanded else 0.0
            # 大词 size_penalty：degree_w 超过 P95 后按 (T/degree_w)^2 衰减，最多按 4 倍截断
            T = float(VOCAB_P95_PAPER_COUNT)
            if degree_w > T:
                x = min(float(degree_w) / T, 4.0)
                size_penalty = (1.0 / x) ** 2
            else:
                size_penalty = 1.0

            # 领域纯度次方惩罚：越偏离目标领域，惩罚越重
            eps = 0.05
            r = max(domain_ratio, eps)
            domain_penalty = r ** 2

            new_rec = {
                "tid": vid,
                "term": term_map[vid],
                "sim_score": agg["sim_score"] * size_penalty * domain_penalty,
                "hit_count": agg["support"],
                "seed_vids": sorted(list(agg.get("seed_vids") or [])),
                "degree_w": degree_w,
                "degree_w_expanded": degree_w_expanded,
                "target_degree_w": target_degree_w,
                "domain_span": domain_span,
                "cov_j": 0.0,
                "origin": "cluster",
            }
            raw_results.append(new_rec)

        # 日志 2：簇扩展来源表（便于判断坏词由谁带进）
        self.debug_info.cluster_expansion_log = expansion_log
        if self.verbose and expansion_log:
            print("\n【簇扩展来源表】term(tid) | seed_term(seed_vid) | sim_in_cluster | seed_sim | contrib")
            for entry in expansion_log[:50]:  # 最多 50 条
                exp_term = term_map.get(entry["term_tid"], str(entry["term_tid"]))
                print(f"  {exp_term}({entry['term_tid']})  <-  {entry['seed_term']}({entry['seed_vid']})  {entry['sim_in_cluster']:.4f}  {entry['seed_sim']:.4f}  {entry['contrib']:.6f}")
            if len(expansion_log) > 50:
                print(f"  ... 共 {len(expansion_log)} 条，仅显示前 50 条")

        return raw_results

    def _calculate_academic_resonance(self, tids):
        return label_expansion.calculate_academic_resonance(self, tids)

    def _calculate_anchor_resonance(self, tids, first_layer_tids):
        return label_expansion.calculate_anchor_resonance(self, tids, first_layer_tids)

    def _get_cooccurrence_domain_metrics(self, raw_results, active_domain_ids):
        return label_expansion.get_cooccurrence_domain_metrics(self, raw_results, active_domain_ids)

        """
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
                    expanded = self._expand_domain_dist(dist)
                    degree_w_exp = sum(expanded.values())
                    target_degree = sum(expanded.get(str(d), 0) for d in active_domain_ids)
                    target_ratio = (target_degree / degree_w_exp) if degree_w_exp else 0.0
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

    def _expand_domain_dist(self, dist):
        """
        将 domain_dist 中可能存在的复合 key（如 "2|7|9"）拆成单领域并合并计数。
        与 DomainProcessor.to_set 的拆分规则一致（支持 | , 空格），索引不变时在召回侧解析。
        返回: 单领域 ID -> 该领域下的 (论文-领域) 出现次数。
        """
        if not dist:
            return {}
        out = {}
        for key, count in dist.items():
            if not key or not count:
                continue
            for d in DomainProcessor.to_set(key):
                out[d] = out.get(d, 0) + count
        return out

    def _load_vocab_meta(self):
        """懒加载：voc_id -> (term, entity_type)，用于语境向量扩展时只保留学术词。"""
        if getattr(self, '_vocab_meta', None) is not None:
            return
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("SELECT voc_id, term, entity_type FROM vocabulary").fetchall()
                self._vocab_meta = {int(r[0]): (r[1] or "", r[2] or "") for r in rows}
        except Exception:
            self._vocab_meta = {}

    def _query_expansion_by_context_vector(self, anchor_skills, query_text, regex, topk_per_anchor=5):
        """
        用「JD 上下文 + 锚点」编码后在 vocabulary 向量索引中检索，只保留学术词，并与 topology 同结构的 stats 过滤。
        返回与 _query_expansion_with_topology 同结构的 rec 列表（tid, term, sim_score, src_vids, hit_count, degree_w, ...）。
        """
        if not query_text or not anchor_skills:
            return []
        self._load_vocab_meta()
        encoder = self._query_encoder
        jd_snippet = (query_text or "").strip()[:500]
        active_domains = set(re.findall(r'\d+', regex)) if regex and regex != ".*" else set()

        # --- ctx 路策略：禁用动态共振 + 向量加权和 + term 向量缓存 ---
        # 1) 编码一次 JD snippet（normalize_embeddings=True -> 已归一化）
        v_jd = encoder.model.encode(
            [jd_snippet],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        v_jd = np.asarray(v_jd, dtype=np.float32).reshape(1, -1)

        # 2) 收集 term，并准备缓存（term_lower -> vec）
        if not hasattr(self, "_term_vec_cache") or self._term_vec_cache is None:
            self._term_vec_cache = {}

        ctx_src_vids = []
        terms_lower = []
        terms_raw = []
        for vid, info in anchor_skills.items():
            term = (info.get("term") or "").strip()
            if not term:
                continue
            try:
                src_vid = int(vid)
            except Exception:
                continue
            ctx_src_vids.append(src_vid)
            tkey = term.lower()
            terms_lower.append(tkey)
            terms_raw.append(term)

        if not terms_lower:
            return []

        # 3) 批量编码未命中的 term（短文本，禁用动态共振）
        to_encode = []
        to_encode_keys = []
        for tkey, term in zip(terms_lower, terms_raw):
            if tkey not in self._term_vec_cache:
                to_encode.append(term)
                to_encode_keys.append(tkey)

        if to_encode:
            new_vecs = encoder.model.encode(
                to_encode,
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            new_vecs = np.asarray(new_vecs, dtype=np.float32)
            for k, vec in zip(to_encode_keys, new_vecs):
                self._term_vec_cache[k] = vec

        # 4) 按锚点顺序组装 term 向量矩阵
        v_terms = np.stack([self._term_vec_cache[tkey] for tkey in terms_lower], axis=0).astype(np.float32)
        if v_terms.ndim == 1:
            v_terms = v_terms.reshape(1, -1)

        # 5) 向量加权和生成 ctx 向量并归一化
        lam = float(self.CTX_MIX_LAMBDA)
        embs = lam * v_jd + (1.0 - lam) * v_terms
        faiss.normalize_L2(embs)

        # --- 批量检索 ---
        k = min(topk_per_anchor * 3, 30)
        scores, labels = self.vocab_index.search(embs, k)

        # --- 聚合：按 tid 聚合 max(ctx_sim) + src_vids ---
        by_tid = {}
        for row_i, src_vid in enumerate(ctx_src_vids):
            for score, tid in zip(scores[row_i], labels[row_i]):
                tid = int(tid)
                if tid <= 0 or tid == int(src_vid):
                    continue
                meta = self._vocab_meta.get(tid, ("", ""))
                if meta[1] not in ("concept", "keyword"):
                    continue
                ctx_sim = max(0.0, float(score))
                if tid not in by_tid:
                    by_tid[tid] = {"ctx_sim": 0.0, "src_vids": set(), "term": meta[0] or ""}
                by_tid[tid]["ctx_sim"] = max(by_tid[tid]["ctx_sim"], ctx_sim)
                by_tid[tid]["src_vids"].add(int(src_vid))
                if not by_tid[tid]["term"] and meta[0]:
                    by_tid[tid]["term"] = meta[0]

        tids = list(by_tid.keys())
        if not tids:
            return []

        results = []
        for tid in tids:
            row = self.stats_conn.execute(
                "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (tid,),
            ).fetchone()
            if not row:
                continue
            degree_w, domain_span, dist_json = row
            try:
                dist = json.loads(dist_json) if isinstance(dist_json, str) else dist_json
            except (TypeError, ValueError):
                dist = {}
            expanded = self._expand_domain_dist(dist)
            degree_w_expanded = sum(expanded.values())
            target_degree_w = sum(expanded.get(str(d), 0) for d in active_domains)
            domain_ratio = target_degree_w / degree_w_expanded if degree_w_expanded else 0.0

            # 首层 ctx 路硬过滤：语义相似度与领域纯度均需达标
            rec = by_tid[tid]
            ctx_sim = float(rec.get("ctx_sim", 0.0) or 0.0)
            if ctx_sim < float(self.SEMANTIC_MIN):
                continue

            if degree_w <= 40:
                purity_min = 0.4
            else:
                purity_min = float(self.DOMAIN_PURITY_MIN)
            if active_domains and domain_ratio < purity_min:
                continue

            # 大词 size_penalty：防止高频大词统治排序
            T = float(VOCAB_P95_PAPER_COUNT)
            if degree_w > T:
                x = min(float(degree_w) / T, 4.0)
                size_penalty = (1.0 / x) ** 2
            else:
                size_penalty = 1.0

            src_vids = sorted(rec["src_vids"])
            results.append({
                "tid": tid,
                "term": rec["term"] or self._vocab_meta.get(tid, ("", None))[0],
                "sim_score": ctx_sim * size_penalty,
                "src_vids": src_vids,
                "hit_count": len(src_vids),
                "degree_w": degree_w,
                "degree_w_expanded": degree_w_expanded,
                "target_degree_w": target_degree_w,
                "domain_span": domain_span,
                "cov_j": 0.0,
                "origin": "context_vector",
            })
        return results

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
          AND coalesce(v_rel.type, 'concept') = 'concept'
        WITH vid, v_rel.id AS tid, v_rel.term AS term, r.score AS sim_score
        ORDER BY vid, sim_score DESC
        WITH vid, collect({tid: tid, term: term, sim_score: sim_score})[0..$top_k] AS top3
        UNWIND top3 AS c
        RETURN vid AS src_vid, c.tid AS tid, c.term AS term, c.sim_score AS sim_score
        """
        rows = self.graph.run(cypher, **params).data()
        if not rows:
            # 记录空的 raw，便于诊断
            self.debug_info.similar_to_raw_rows = []
            self.debug_info.similar_to_agg = []
            self.debug_info.similar_to_pass = []
            return []

        # 缓存 SIMILAR_TO 原始扩展行（仅保留必要字段，避免 debug 体积过大）
        self.debug_info.similar_to_raw_rows = [
            {
                "src_vid": r.get("src_vid"),
                "tid": r.get("tid"),
                "term": r.get("term"),
                "sim_score": float(r.get("sim_score", 0.0) or 0.0),
            }
            for r in rows
        ]

        # --- 调试：语义扩展管道各阶段计数 ---
        pipeline = {
            "n_similar_to_rows": len(rows),
            "active_domains": list(active_domains),
            "n_unique_tids": 0,
            "n_no_stats": 0,
            "n_fail_degree_w": 0,
            "n_fail_target_degree_w": 0,
            "n_fail_domain_ratio": 0,
            "n_fail_degree_w_expanded_zero": 0,
            "n_final": 0,
            "sample_fail_no_stats": [],
            "sample_fail_degree": [],
            "sample_fail_target": [],
            "sample_fail_ratio": [],
            "fail_domain_ratio_details": [],
        }

        # 按 tid 聚合：取 max(sim_score)，hit_count = 被多少“不同锚点(src_vid)”命中
        by_tid = {}
        for r in rows:
            tid = r["tid"]
            term = r["term"] or ""
            sim = float(r["sim_score"])
            if tid not in by_tid:
                by_tid[tid] = {
                    "tid": tid,
                    "term": term,
                    "sim_score": sim,
                    "src_vids": set(),
                    "hit_count": 0,
                    "origin": "similar_to",
                }
            by_tid[tid]["sim_score"] = max(by_tid[tid]["sim_score"], sim)
            src_vid = r.get("src_vid")
            if src_vid is not None:
                try:
                    by_tid[tid]["src_vids"].add(int(src_vid))
                except Exception:
                    pass

        for tid, rec in by_tid.items():
            rec["hit_count"] = len(rec.get("src_vids") or [])
            # set 不可序列化，转 list 供 debug
            rec["src_vids"] = sorted(list(rec.get("src_vids") or []))

        tids = list(by_tid.keys())
        pipeline["n_unique_tids"] = len(tids)

        # 保存“去重聚合后的候选（尚未做 stats/领域过滤）”供诊断
        self.debug_info.similar_to_agg = [
            {
                "tid": v.get("tid"),
                "term": v.get("term", ""),
                "sim_score": float(v.get("sim_score", 0.0) or 0.0),
                "hit_count": int(v.get("hit_count", 0) or 0),
                "src_vids": v.get("src_vids", []),
            }
            for v in by_tid.values()
        ]

        results = []
        for tid in tids:
            row = self.stats_conn.execute(
                "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (tid,),
            ).fetchone()
            if not row:
                pipeline["n_no_stats"] += 1
                if len(pipeline["sample_fail_no_stats"]) < 5:
                    pipeline["sample_fail_no_stats"].append(tid)
                continue
            degree_w, domain_span, dist_json = row
            if degree_w > VOCAB_P95_PAPER_COUNT:
                pipeline["n_fail_degree_w"] += 1
                if len(pipeline["sample_fail_degree"]) < 5:
                    pipeline["sample_fail_degree"].append(tid)
            try:
                dist = json.loads(dist_json) if isinstance(dist_json, str) else dist_json
            except (TypeError, ValueError):
                dist = {}
            expanded = self._expand_domain_dist(dist)
            degree_w_expanded = sum(expanded.values())
            target_degree_w = sum(expanded.get(str(d), 0) for d in active_domains)
            # 领域纯度过滤（分档）：首层 SIMILAR_TO 采用硬过滤
            domain_ratio = target_degree_w / degree_w_expanded if degree_w_expanded else 0.0
            if degree_w <= 40:
                purity_min = 0.4
            else:
                purity_min = float(self.DOMAIN_PURITY_MIN)
            if domain_ratio < purity_min:
                if degree_w_expanded == 0:
                    pipeline["n_fail_degree_w_expanded_zero"] += 1
                pipeline["n_fail_domain_ratio"] += 1
                if len(pipeline["sample_fail_ratio"]) < 5:
                    pipeline["sample_fail_ratio"].append(tid)
                details = pipeline.get("fail_domain_ratio_details", [])
                if len(details) < 20:
                    all_ratio = {d: round(expanded.get(d, 0) / degree_w_expanded, 4) for d in expanded} if degree_w_expanded else {}
                    fail_reason = "degree_w_expanded=0" if degree_w_expanded == 0 else f"domain_ratio<{purity_min}"
                    details.append({
                        "tid": tid,
                        "term": by_tid[tid].get("term", ""),
                        "degree_w": degree_w,
                        "degree_w_expanded": degree_w_expanded,
                        "target_degree_w": target_degree_w,
                        "domain_ratio": round(domain_ratio, 4),
                        "target_domains_dist": {str(d): expanded.get(str(d), 0) for d in active_domains},
                        "all_domains_ratio": all_ratio,
                        "fail_reason": fail_reason,
                    })
                    pipeline["fail_domain_ratio_details"] = details
                # 领域纯度不过关：直接丢弃，不进入首层候选
                continue

            # 大词 size_penalty：防止高频大词统治排序
            T = float(VOCAB_P95_PAPER_COUNT)
            if degree_w > T:
                x = min(float(degree_w) / T, 4.0)
                size_penalty = (1.0 / x) ** 2
            else:
                size_penalty = 1.0

            pipeline["n_final"] += 1
            rec = by_tid[tid]
            rec["degree_w"] = degree_w
            rec["degree_w_expanded"] = degree_w_expanded
            rec["target_degree_w"] = target_degree_w
            rec["domain_span"] = domain_span
            rec["cov_j"] = 0.0
            # 仅保留大词 size_penalty，不再额外按领域纯度降权（低纯度已在上方丢弃）
            rec["sim_score"] = float(rec.get("sim_score", 0.0) or 0.0) * size_penalty
            results.append(rec)
        self.debug_info.expansion_pipeline_stats = pipeline

        # 保存“stats/领域过滤后”的第一层学术词（similar_to 通过项）供诊断
        self.debug_info.similar_to_pass = [
            {
                "tid": r.get("tid"),
                "term": r.get("term", ""),
                "sim_score": float(r.get("sim_score", 0.0) or 0.0),
                "hit_count": int(r.get("hit_count", 0) or 0),
                "src_vids": r.get("src_vids", []),
                "degree_w": int(r.get("degree_w", 0) or 0),
                "degree_w_expanded": int(r.get("degree_w_expanded", 0) or 0),
                "target_degree_w": int(r.get("target_degree_w", 0) or 0),
                "domain_span": int(r.get("domain_span", 0) or 0),
            }
            for r in results
        ]
        return results

    def _calculate_final_weights(self, raw_results, query_vector, anchor_vids=None):
        """
        【统一权重】所有候选词一律走 _apply_word_quality_penalty（含领域纯度、共鸣、语义守门、锚点距离门控等），
        无 sim_score/degree_w 分支例外，流程清晰、领域占比一致生效。
        anchor_vids：本次 JD 锚点 voc id 列表，用于无硬编码的锚点距离门控（与锚点均远的 term 降权）。
        """
        score_map, term_map, idf_map = {}, {}, {}
        required = ("degree_w", "cov_j", "domain_span", "target_degree_w")
        self.debug_info.tag_purity_debug = []
        # 保持与旧字段名的兼容，便于现有 getattr(self, "_last_tag_purity_debug", ...) 逻辑复用
        self._last_tag_purity_debug = self.debug_info.tag_purity_debug

        # --- 预计算锚点向量，供 _apply_word_quality_penalty 中锚点距离门控使用 ---
        # 同时基于本次 JD 语义，将锚点划分为 task_anchors 与 carrier_anchors 两个子集：
        # - task_anchors：更贴近“控制/规划/动力学”等任务意图；
        # - carrier_anchors：更贴近“机器人/机械臂/robotics”等载体语义。
        self._anchor_vectors = None
        self._task_anchor_vectors = None
        self._carrier_anchor_vectors = None
        if anchor_vids and getattr(self, "vocab_to_idx", None) is not None and getattr(self, "all_vocab_vectors", None) is not None:
            idxs = []
            for vid in anchor_vids:
                i = self.vocab_to_idx.get(str(vid))
                if i is not None:
                    idxs.append(i)
            if idxs:
                vecs = np.asarray(self.all_vocab_vectors[idxs], dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms > 1e-9, norms, 1.0)
                anchor_vecs = (vecs / norms)
                self._anchor_vectors = anchor_vecs

                # 基于本次 JD 的语义方向（query_vector）在锚点集合内部做一次 task/carrier 软划分：
                # 仅使用向量语义，不依赖人工词表，避免过度硬编码。
                if query_vector is not None:
                    try:
                        q = np.asarray(query_vector, dtype=np.float32).flatten()
                        if q.size > 0:
                            qn = np.linalg.norm(q)
                            if qn > 1e-9:
                                q = q / qn
                                sims = np.dot(anchor_vecs, q)
                                n_anchor = anchor_vecs.shape[0]
                                # 锚点数较少时不做拆分，统一作为 task_anchors 使用
                                if n_anchor < 4:
                                    self._task_anchor_vectors = anchor_vecs
                                else:
                                    median_sim = float(np.median(sims))
                                    task_mask = sims >= median_sim
                                    carrier_mask = sims < median_sim
                                    if np.any(task_mask):
                                        self._task_anchor_vectors = anchor_vecs[task_mask]
                                    if np.any(carrier_mask):
                                        self._carrier_anchor_vectors = anchor_vecs[carrier_mask]
                    except Exception:
                        # 语义划分失败时退化为单一锚点集合，不影响主流程
                        self._task_anchor_vectors = None
                        self._carrier_anchor_vectors = None

        # --- 预计算语义相似度分布，用于 Top20/中60/底20 分段加权 ---
        # 说明：
        #  - 仍然保留 SEMANTIC_MIN 作为硬门槛，只对 >= 阈值的词做分位数统计；
        #  - 若当次查询候选词过少或全部低于阈值，则退化为无分段（bucket_factor=1）。
        self._semantic_p20 = None
        self._semantic_p80 = None
        if query_vector is not None and raw_results:
            sims = []
            try:
                q = np.asarray(query_vector, dtype=np.float32).flatten()
                if q.size > 0:
                    for rec in raw_results:
                        tid = rec.get("tid")
                        if tid is None:
                            continue
                        idx = self.vocab_to_idx.get(str(tid))
                        if idx is None:
                            continue
                        try:
                            term_vec = self.all_vocab_vectors[idx]
                        except Exception:
                            continue
                        cos_sim = float(np.dot(term_vec, q))
                        if cos_sim >= float(self.SEMANTIC_MIN):
                            sims.append(cos_sim)
            except Exception:
                sims = []

            if sims:
                sims.sort()
                n = len(sims)
                # 简单的分位实现：使用排序后第 k 个元素近似 20% / 80%
                def _pick_percentile(p: float) -> float:
                    if n == 1:
                        return sims[0]
                    k = int(p * (n - 1))
                    k = max(0, min(n - 1, k))
                    return sims[k]

                # 仅当样本数足够大时才启用分段，以避免极小样本下的过度放大
                if n >= 5:
                    self._semantic_p20 = _pick_percentile(0.2)
                    self._semantic_p80 = _pick_percentile(0.8)

        for rec in raw_results:
            tid = str(rec["tid"])
            if all(rec.get(k) is not None for k in required):
                dynamic_weight, idf_val = self._apply_word_quality_penalty(rec, query_vector)
            else:
                # 安全回退：仅当数据不完整时（如某条扩展路径漏写字段）
                degree_w = rec.get("degree_w") or 1
                sim_score = rec.get("sim_score") or 0.0
                dynamic_weight = sim_score / math.log(1.0 + degree_w) if degree_w else 0.0
                idf_val = math.log10(self.total_work_count / (degree_w + 1))
            score_map[tid] = dynamic_weight
            term_map[tid] = rec.get("term") or ""
            idf_map[tid] = idf_val
        # 词级权重计算完毕后，在簇内做一次 head–tail 衰减，平衡主干与尾部 term
        self._apply_cluster_rank_decay(score_map)

        return score_map, term_map, idf_map

    def _select_terms_for_paper(self, score_map, term_map, max_terms=16):
        """
        从全部打分学术词中选出一小撮高质量词用于论文检索。
        仅依赖通用指标（weight / 领域纯度 / 语义相似度），避免硬编码具体词表。
        """
        if not score_map:
            return []

        debug_rows = {}
        for row in (getattr(self, "_last_tag_purity_debug", None) or self.debug_info.tag_purity_debug or []):
            tid = row.get("tid")
            if tid is not None:
                debug_rows[str(tid)] = row

        ranked_tids = sorted(
            score_map.keys(),
            key=lambda t: float(score_map.get(t, 0.0) or 0.0),
            reverse=True,
        )

        selected = []
        for tid in ranked_tids:
            tid_s = str(tid)
            row = debug_rows.get(tid_s, {})
            weight = float(score_map.get(tid, 0.0) or 0.0)
            if weight <= 0.0:
                continue

            # 领域纯度：优先使用 capped_tag_purity，退化到 raw_tag_purity
            tag_purity = float(row.get("capped_tag_purity") or row.get("raw_tag_purity") or 0.0)
            if tag_purity and tag_purity < 0.45:
                continue

            # 语义相似度：cos_sim / task_anchor_sim / anchor_sim 取最大值
            cos_sim = float(row.get("cos_sim") or 0.0)
            anchor_sim = float(row.get("task_anchor_sim") or row.get("anchor_sim") or 0.0)
            sim_val = max(cos_sim, anchor_sim)
            if sim_val and sim_val < float(self.SEMANTIC_MIN):
                continue

            selected.append(int(tid))
            if len(selected) >= int(max_terms):
                break

        if not selected:
            # 安全回退：若所有过滤条件都过于严格，则退化为分最高的前若干个
            fallback = []
            for tid in ranked_tids[: min(8, len(ranked_tids))]:
                fallback.append(int(tid))
            return fallback

        return selected

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
        # 与边路过滤同一口径：用展开后的总和作分母，purity 自然 ≤ 1
        degree_w_expanded = rec.get('degree_w_expanded')
        if degree_w_expanded is None:
            degree_w_expanded = degree_w

        # 1. 计算语义纯度 (Tag Purity)
        raw_tag_purity = (rec['target_degree_w'] / degree_w_expanded) if degree_w_expanded else 0.0
        tag_purity = min(1.0, raw_tag_purity)
        purity_term = tag_purity  # 作为主骨架中的 purity 因子

        # 2. IDF 主骨架（含 size_penalty）
        idf_term = self._idf_backbone(degree_w_expanded)
        idf_val = self._smoothed_idf(degree_w, idf_term)

        # 3. 语义相似度（sim）
        cos_sim = 0.5
        idx = self.vocab_to_idx.get(str(rec['tid']))
        if idx is not None and query_vector is not None:
            term_vec = self.all_vocab_vectors[idx]
            cos_sim = float(np.dot(term_vec, query_vector.flatten()))

        # 语义硬拦截
        if cos_sim < float(self.SEMANTIC_MIN):
            return 0.0, idf_val

        semantic_factor = math.pow(max(0.0, cos_sim), float(self.SEMANTIC_POWER))

        # 4. 轻量 anchor_factor（仅依赖锚点相似度）
        max_anchor_sim = None
        task_anchor_sim = None
        carrier_anchor_sim = None
        anchor_vecs = getattr(self, "_anchor_vectors", None)
        task_anchor_vecs = getattr(self, "_task_anchor_vectors", None)
        carrier_anchor_vecs = getattr(self, "_carrier_anchor_vectors", None)
        if idx is not None:
            try:
                tvec = np.asarray(self.all_vocab_vectors[idx], dtype=np.float32).flatten()
                tn = np.linalg.norm(tvec)
                if tn > 1e-9:
                    t_unit = tvec / tn
                    if anchor_vecs is not None and anchor_vecs.size > 0:
                        max_anchor_sim = float(np.max(np.dot(anchor_vecs, t_unit)))
                    if task_anchor_vecs is not None and task_anchor_vecs.size > 0:
                        task_anchor_sim = float(np.max(np.dot(task_anchor_vecs, t_unit)))
                    if carrier_anchor_vecs is not None and carrier_anchor_vecs.size > 0:
                        carrier_anchor_sim = float(np.max(np.dot(carrier_anchor_vecs, t_unit)))
            except Exception:
                pass

        ta = task_anchor_sim if task_anchor_sim is not None else max_anchor_sim
        ca = carrier_anchor_sim if carrier_anchor_sim is not None else max_anchor_sim
        anchor_factor = self._anchor_factor(ta, ca)

        # 5. 主骨架 five factors
        # 这里把 size_penalty 抽象成一个额外因子，但当前从 advanced_metrics 返回 1.0
        size_penalty = label_means_adv.size_penalty_factor(degree_w, degree_w_expanded)
        sim_term = semantic_factor
        term_backbone = sim_term * idf_term * purity_term * size_penalty * anchor_factor

        # 6. 其余所有非主骨架修饰项统一交给 advanced_metrics（当前返回 1.0）
        extra_factor = label_means_adv.term_extra_factors(
            rec=rec,
            cos_sim=cos_sim,
            degree_w=degree_w,
            degree_w_expanded=degree_w_expanded,
            cov_j=cov_j,
            domain_span=domain_span,
            tag_purity=tag_purity,
            task_anchor_sim=task_anchor_sim,
            carrier_anchor_sim=carrier_anchor_sim,
            max_anchor_sim=max_anchor_sim,
        )

        dynamic_weight = term_backbone * extra_factor
        return dynamic_weight, idf_val

    # ---------- 词级权重因子（第二轮：因子接口抽取，仅封装现有逻辑） ----------

    def _idf_backbone(self, degree_w_expanded: float) -> float:
        """
        标准 IDF（IR 口径）：idf = log(1 + total_work / (1 + paper_count))。
        """
        return math.log(1.0 + float(self.total_work_count) / (1.0 + float(degree_w_expanded)))

    def _smoothed_idf(self, degree_w: int, idf_backbone_val: float) -> float:
        """
        平滑 IDF：小词/大词略做约束，中等区间略微加权。
        逻辑与原实现完全一致，只做封装。
        """
        if degree_w < 10:
            return 1.0
        if degree_w < 50:
            t = (float(degree_w) - 10.0) / 40.0
            return 1.0 + 0.5 * max(0.0, min(1.0, t))
        return 0.9

    def _job_penalty_factor(self, cov_j: float) -> float:
        """
        岗位泛词惩罚因子：复用了原有 exp(300*(cov_j-0.005)) 公式。
        """
        return 1.0 + math.exp(300.0 * (float(cov_j) - 0.005))

    def _hit_count_factor(self, hit_count: int) -> float:
        """
        支持锚点数量因子：单锚点 0.4，多锚点 1.0。
        """
        return 0.4 if int(hit_count or 0) == 1 else 1.0

    def _resonance_factor(self, resonance: float, anchor_resonance: float) -> float:
        """
        共鸣因子：与第一层学术词有共现时 1+log1p(resonance)，否则 0.1。
        """
        return label_means_adv.term_resonance_factor(resonance, anchor_resonance)

    def _convergence_bonus(self, hit_count_factor: float, hit_count: int, resonance_factor: float) -> float:
        """
        收敛奖励：hit_count_factor * log1p(hit_count) * resonance_factor。
        """
        return label_means_adv.term_convergence_bonus(hit_count_factor, hit_count, resonance_factor)

    def _anchor_factor(self, ta: float, ca: float) -> float:
        """
        锚点距离门控因子：task_anchor_sim 优先，其次 carrier_anchor_sim。
        封装原有 0.20 + 0.65*ta^1.5 + 0.15*ca^1.1 逻辑。
        """
        if ta is None and ca is None:
            return 1.0
        ta_clamped = max(0.0, min(1.0, ta or 0.0))
        ca_clamped = max(0.0, min(1.0, ca or 0.0))
        return (
            0.20
            + 0.65 * math.pow(ta_clamped, 1.5)
            + 0.15 * math.pow(ca_clamped, 1.1)
        )

    def _cooc_span_penalty(self, cooc_span: float) -> float:
        """
        共现领域跨度惩罚：cooc_span 大 → 万金油 → 乘小于 1 的因子。
        等价于 1 / (1 + log1p(cooc_span))。
        """
        return label_means_adv.term_cooc_span_penalty(cooc_span)

    def _cooc_purity_bonus(self, cooc_purity: float) -> float:
        """
        共现目标领域纯度奖励：cooc_purity 大 → 专精 → 乘大于 1 的因子。
        等价于 1 + log1p(cooc_purity)。
        """
        return label_means_adv.term_cooc_purity_bonus(cooc_purity)

    def _domain_span_penalty(self, domain_span: int) -> float:
        """
        领域跨度惩罚：根据 (1 + domain_span)^SPAN_PENALTY_EXPONENT 计算。
        """
        return math.pow(1.0 + float(domain_span), float(self.SPAN_PENALTY_EXPONENT))

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
        3. 6 次方加成：通过 math.pow(ratio, 6) 让多领域均匀分布的论文分数呈断崖式下跌。
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
        # 采用 6 次方惩罚：多领域均匀分布的论文被压得更狠（无硬编码领域 id）
        base_score = 1.0 + (dominance * 5.0) if intersect else 0.5
        return base_score * math.pow(purity_ratio, 6)

    # ---------- 论文级评分因子（第二轮：因子接口抽取，仅封装现有逻辑） ----------

    def _survey_decay_factor(self, hit_count: int, raw_title: str) -> float:
        """
        综述 / 文本类型统一衰减因子。
        封装原有：hit_count^(-2) + apply_text_decay(title) 的组合逻辑。
        """
        return survey_decay_factor(hit_count, raw_title)

    def _coverage_norm_factor(self, hit_count: int) -> float:
        """
        命中标签数量归一化因子。
        封装原有：1 / log(2 + hit_count) 逻辑。
        """
        return coverage_norm_factor(hit_count)

    def _paper_cluster_bonus(self, cluster_ids) -> float:
        """
        论文跨 topic cluster 奖励因子。
        封装原有：log1p(cluster_count) 逻辑。
        """
        return paper_cluster_bonus(cluster_ids)

    def _paper_jd_semantic_gate_factor(self, raw_title: str, jd_vec, encoder) -> float:
        """
        论文标题与 JD 的语义相似度门控因子。
        封装原有：cos<0.3 ->0.1, cos<0.5->0.4, 否则 1.0 的分段逻辑。
        """
        return paper_jd_semantic_gate_factor(raw_title, jd_vec, encoder)

    def _compute_contribution(self, paper, context):
        """
        【主评分函数】量化贡献度计算
        调度逻辑：拦截撤稿 -> 计算领域纯度 -> 累加标签权重 -> 应用时序与紧密度加成。
        """
        raw_title = (paper.get('title') or "")

        # 1. 撤稿拦截
        if self._is_retracted(raw_title):
            return 0, [], 0.0, {}

        # 2. 领域纯度降权：调用辅助函数计算基于“专注度”的领域系数
        # 解决了你担心的“涉及领域越多分越低”的问题
        domain_coeff = self._get_domain_purity_factor(
            paper.get('domains'),
            context['active_domain_set'],
            context['dominance']
        )
        if domain_coeff <= 0:
            return 0, [], 0.0, {}

        # 3. 标签匹配与动态权重累加
        # 这里的 score_map 已经包含了之前修改的“词级领域跨度惩罚”
        rank_score = 0
        term_weights = {}  # vid_s -> score_map[vid]*idf，供调试与作者来源拆解
        valid_hids, hit_terms = [], []
        for hit in paper['hits']:
            vid_s = str(hit['vid'])
            if vid_s in context['score_map']:
                w = context['score_map'][vid_s] * hit['idf']
                rank_score += w
                term_weights[vid_s] = w
                valid_hids.append(hit['vid'])
                hit_terms.append(context['term_map'][vid_s])

        if rank_score == 0:
            return 0, [], 0.0, {}

        # 4. 综述降权 + 文本类型降权（统一规则）
        hit_count = len(valid_hids)
        survey_decay = self._survey_decay_factor(hit_count, raw_title)

        # 5. 指数级紧密度加成 (1+prox)^n
        proximity = self._calculate_proximity(valid_hids)
        proximity_bonus = math.pow(1.0 + proximity, hit_count)

        # 6. 时序衰减：统一调用 time_features.compute_paper_recency（阶梯式 bucket）
        time_decay = compute_paper_recency(paper.get('year', 2000), context['active_domain_set'])

        # 6.5 跨簇奖励：命中多个 topic cluster 的论文略微加成
        cluster_ids = set()
        for vid in valid_hids:
            try:
                clusters = self.voc_to_clusters.get(int(vid), [])
            except Exception:
                clusters = []
            if clusters:
                # 取该词所属得分最高的簇作为其主簇
                cid, _ = max(clusters, key=lambda x: x[1])
                cluster_ids.add(cid)
        cluster_bonus = self._paper_cluster_bonus(cluster_ids)

        # 7. 命中标签数量归一化：防止“命中标签越多 → 单论文贡献爆炸”主导作者排序
        # 采用次线性归一：log(2 + hit_count)，保证 1 标签与多标签论文在同一量级
        coverage_norm = self._coverage_norm_factor(hit_count)

        # 最终组合：按词权重 + 语义紧密度 + 领域纯度 + 时间/文本类型 + 命中数归一化 + 跨簇奖励 综合计算论文本身贡献度
        score = (
            rank_score
            * coverage_norm
            * cluster_bonus
            * proximity_bonus
            * domain_coeff
            * time_decay
            * survey_decay
        )

        # 7.5 paper-level JD semantic gate：论文标题与 JD 的语义相似度门控，压制跨领域噪声（如教育/农业推荐被误召）
        jd_vec = context.get("query_vector")
        gate_factor = self._paper_jd_semantic_gate_factor(
            raw_title,
            jd_vec,
            getattr(self, "_query_encoder", None),
        )
        score *= gate_factor

        return score, hit_terms, rank_score, term_weights

    # ---------- 五阶段流程（便于维护与修改） ----------

    def _stage1_domain_and_anchors(self, query_vector, query_text=None, domain_id=None):
        return stage1_domain_anchors.run_stage1(self, query_vector, query_text=query_text, domain_id=domain_id)

    def _stage2_expand_academic_terms(self, anchor_skills, active_domain_set, regex_str, query_vector, query_text=None):
        """
        阶段 2：学术词扩展。边路 + 语境向量路 + 簇扩展 + 共鸣/共现，返回候选列表（不计算最终词权）。
        返回: raw_candidates，每项含 tid, term, degree_w, target_degree_w, domain_span, cov_j, hit_count 等，供 stage3 统一公式。
        """
        # 基于当前 query_vector 预先计算一次簇级 gating 因子，供词权重阶段使用
        self._compute_cluster_task_factors(query_vector)
        return self._expand_semantic_map(
            [int(k) for k in anchor_skills.keys()],
            anchor_skills,
            domain_regex=regex_str,
            query_vector=query_vector,
            query_text=query_text,
            return_raw=True,
        )

    def _apply_cluster_rank_decay(self, score_map: dict) -> None:
        """
        对每个 cluster 内部按 score 排序，做 head-tail 衰减（仅同簇尾部微调）：
        - cluster_size <= 3: 不做衰减；
        - 若 anchor_sim >= 0.9: factor = 1.0；
        - 否则若 rank <= 4: factor = 1.0；
        - 否则: factor = 0.85 ** (rank - 4)。
        直接原地修改 score_map。
        注意：簇内 rank decay 只做「同簇尾部微调」，不负责纠正簇头；簇头是否该排前面由
        task_anchor_sim / task_advantage / main_role 决定。
        """
        if not getattr(self, "voc_to_clusters", None) or not score_map:
            return

        self.debug_info.cluster_rank_factors = {}

        debug_list = getattr(self, "_last_tag_purity_debug", None) or []
        anchor_sim_by_tid = {}
        task_sim_by_tid = {}
        for d in debug_list:
            tid = d.get("tid")
            if tid is None:
                continue
            tid_str = str(tid)
            if d.get("anchor_sim") is not None:
                anchor_sim_by_tid[tid_str] = float(d["anchor_sim"])
            if d.get("task_anchor_sim") is not None:
                task_sim_by_tid[tid_str] = float(d["task_anchor_sim"])

        cluster_to_tids = {}
        for tid_str in score_map.keys():
            try:
                tid = int(tid_str)
            except (TypeError, ValueError):
                continue
            clusters = self.voc_to_clusters.get(tid)
            if not clusters:
                continue
            cid, _ = max(clusters, key=lambda x: x[1])
            cluster_to_tids.setdefault(cid, []).append(tid_str)

        for cid, tids in cluster_to_tids.items():
            if len(tids) <= 3:
                for tid_str in tids:
                    self.debug_info.cluster_rank_factors[tid_str] = 1.0
                continue

            # 簇内排序：以 score 为主键，task_anchor_sim 为次关键字，弱化载体词对簇头的干扰
            if len(tids) >= 5:
                tids_sorted = sorted(
                    tids,
                    key=lambda x: (
                        score_map.get(x, 0.0),
                        task_sim_by_tid.get(x, 0.0),
                    ),
                    reverse=True,
                )
            else:
                tids_sorted = sorted(tids, key=lambda x: score_map.get(x, 0.0), reverse=True)
            for rank, tid_str in enumerate(tids_sorted):
                anchor_sim = anchor_sim_by_tid.get(tid_str)
                if anchor_sim is not None and anchor_sim >= 0.9:
                    factor = 1.0
                elif rank <= 4:
                    factor = 1.0
                else:
                    factor = 0.85 ** (rank - 4)

                score_map[tid_str] *= factor
                self.debug_info.cluster_rank_factors[tid_str] = float(factor)

    def _stage3_word_weights(self, raw_candidates, query_vector, anchor_vids=None):
        """
        阶段 3：词权重。统一走复杂公式（领域纯度、共鸣、语义守门、锚点距离门控等），无分支例外。
        返回: (score_map, term_map, idf_map)。anchor_vids 用于无硬编码的锚点距离门控。
        """
        return stage3_term_filtering.run_stage3(self, raw_candidates, query_vector, anchor_vids=anchor_vids)

    def _stage4_graph_search(self, vocab_ids, regex_str):
        """
        阶段 4：图检索。用学术词 ID 反查 Work 与 Author。
        返回: list of { 'aid': str, 'papers': [ { wid, hits, weight, title, year, domains }, ... ] }。
        """
        params = {"v_ids": vocab_ids, "total_w": self.total_work_count}
        domain_clause = ""
        if regex_str:
            domain_clause = "AND w.domain_ids =~ $regex"
            params["regex"] = regex_str
        final_cypher = f"""
        MATCH (v:Vocabulary) WHERE v.id IN $v_ids
        WITH v, COUNT {{ (v)<-[:HAS_TOPIC]-() }} AS degree_w
        WHERE (degree_w * 1.0 / $total_w) < 0.03 
        WITH v, log10($total_w / (degree_w + 1)) AS idf_weight
        MATCH (v)<-[:HAS_TOPIC]-(w:Work) 
        WHERE 1=1 {domain_clause} 
        WITH w, collect({{vid: v.id, idf: idf_weight}}) AS hit_info LIMIT 2000
        MATCH (w)<-[auth_r:AUTHORED]-(a:Author)
        RETURN a.id AS aid, collect({{wid: w.id, hits: hit_info, weight: auth_r.pos_weight, 
                                     title: w.title, year: w.year, domains: w.domain_ids}}) AS papers
        """
        cursor = self.graph.run(final_cypher, **params)
        return list(cursor)

    def _stage5_score_and_rank_authors(self, author_papers_list, score_map, term_map, active_domain_set, dominance, debug_1):
        """
        阶段 5：作者打分与排序。按论文贡献度聚合、排序、截断，并组装 last_debug_info。
        返回: (author_id_list, last_debug_info)。
        """
        industrial_kws = debug_1.get("industrial_kws", [])
        anchor_skills = debug_1.get("anchor_skills", {})
        context = {
            "score_map": score_map,
            "term_map": term_map,
            "anchor_kws": [k.lower() for k in industrial_kws],
            "active_domain_set": active_domain_set,
            "dominance": dominance,
            "decay_rate": get_decay_rate_for_domains(active_domain_set),
            "query_vector": debug_1.get("query_vector"),  # JD 向量，供 paper-level semantic gate
        }

        # ---------- 统一“论文 → 作者”聚合逻辑 ----------
        # 1) 先按论文维度重组成 {wid -> 论文信息 + 全体作者列表} 结构
        paper_map = {}  # wid -> {wid, hits, title, year, domains, authors: [{aid, pos_weight}, ...]}
        author_raw_paper_cnt = collections.Counter()  # aid -> 原始论文数量（用于 debug 中的 paper_count）

        for record in author_papers_list:
            aid = record["aid"]
            for paper in record["papers"]:
                wid = paper["wid"]
                entry = paper_map.setdefault(
                    wid,
                    {
                        "wid": wid,
                        "hits": paper["hits"],
                        "title": paper["title"],
                        "year": paper["year"],
                        "domains": paper["domains"],
                        "authors": [],
                    },
                )
                entry["authors"].append(
                    {"aid": aid, "pos_weight": float(paper.get("weight") or 1.0)}
                )
                author_raw_paper_cnt[aid] += 1

        # 2) 对每篇论文计算“论文本身”的贡献度（不含作者拆分），并为公共聚合函数准备输入
        papers_for_agg = []
        paper_hit_terms = {}  # wid -> 命中标签列表（用于后续 per-author 代表作与标签统计）
        all_works_count = 0

        for wid, info in paper_map.items():
            paper_struct = {
                "wid": wid,
                "hits": info["hits"],
                "title": info["title"],
                "year": info["year"],
                "domains": info["domains"],
            }
            p_score, p_hits, p_rank_score, p_term_weights = self._compute_contribution(paper_struct, context)
            all_works_count += 1
            if p_score <= 0:
                continue
            paper_hit_terms[wid] = p_hits
            info["score"] = float(p_score)
            info["rank_score"] = float(p_rank_score or 0)
            info["term_weights"] = dict(p_term_weights or {})
            papers_for_agg.append(
                {
                    "wid": wid,
                    "score": float(p_score),
                    "rank_score": float(p_rank_score or 0),
                    "term_weights": dict(p_term_weights or {}),
                    "authors": info["authors"],
                }
            )

        # 2.5 论文软上限：按本次查询下论文得分分布的 95 分位构造 τ，用 tanh 做平滑压缩
        if papers_for_agg:
            paper_scores = [p["score"] for p in papers_for_agg]
            try:
                tau = float(np.percentile(paper_scores, 95))
            except Exception:
                tau = 0.0

            if tau > 0:
                def _compress(s: float) -> float:
                    # τ * tanh(s/τ)：中小论文几乎不变，极强论文逐渐被压到 τ 附近
                    return float(tau * math.tanh(s / tau))

                for p in papers_for_agg:
                    p["score"] = _compress(float(p["score"]))

        # 3) 统一调用 works_to_authors 进行“论文 → 作者”分摊与聚合（仅保留每位作者贡献最高的 Top3 论文）
        agg_result = accumulate_author_scores(papers_for_agg, top_k_per_author=3)
        author_scores = agg_result.author_scores
        author_top_works = agg_result.author_top_works  # aid -> [(wid, contrib_score), ...]
        paper_scores_by_wid = {p["wid"]: float(p["score"]) for p in papers_for_agg}

        # 3.5) 每 term 对论文的贡献（用于 Top term 最终贡献表）与每作者来源 term 拆解
        term_paper_contrib = collections.defaultdict(list)  # tid -> [(wid, contrib), ...]
        for p in papers_for_agg:
            wid, s_final, r_score, tw = p["wid"], p["score"], p.get("rank_score") or 1.0, p.get("term_weights") or {}
            if r_score <= 0:
                continue
            for vid_s, w in tw.items():
                term_paper_contrib[vid_s].append((wid, (w / r_score) * s_final))

        author_term_contrib = collections.defaultdict(lambda: collections.defaultdict(float))  # aid -> tid -> contrib
        for aid, works in author_top_works.items():
            for wid, contrib in works:
                info = paper_map.get(wid, {})
                r_score = info.get("rank_score") or 1.0
                tw = info.get("term_weights") or {}
                if r_score <= 0:
                    continue
                for vid_s, w in tw.items():
                    author_term_contrib[aid][vid_s] += contrib * (w / r_score)

        # 4) 重建 per-author 调试与展示结构（代表作、标签统计等）
        # ---- 作者层时间特征：活跃度 + 动量（统一 time_features）----
        if author_scores:
            # 为每个作者收集其论文年份列表（基于 paper_map）
            years_by_author = {}
            for aid in author_scores.keys():
                years = []
                for wid, _ in author_top_works.get(aid, []):
                    meta = paper_map.get(wid, {})
                    years.append(meta.get("year"))
                years_by_author[aid] = years

            for aid, base_score in list(author_scores.items()):
                years = years_by_author.get(aid, [])
                # activity/momentum 由 time_features 统一计算
                activity, momentum, time_weight = compute_author_time_features(years)
                # 基于最近代表作年龄的平滑衰减也集中在 time_features 中
                recency_by_latest = compute_author_recency_by_latest(years)
                score = float(base_score) * float(time_weight) * float(recency_by_latest)

                author_scores[aid] = score

        # 作者过滤（数据驱动）：最佳论文贡献低于全局最大论文贡献一定比例的作者不参与排序
        if papers_for_agg and author_scores and author_top_works:
            paper_scores_by_wid = {p["wid"]: float(p["score"]) for p in papers_for_agg}
            max_paper = max(paper_scores_by_wid.values()) if paper_scores_by_wid else 0.0
            if max_paper > 0:
                min_contrib = max_paper * float(self.AUTHOR_BEST_PAPER_MIN_RATIO)
                to_remove = [
                    aid for aid in author_scores
                    if max(
                        (paper_scores_by_wid.get(wid, 0.0) for wid, _ in author_top_works.get(aid, [])),
                        default=0.0,
                    ) < min_contrib
                ]
                for aid in to_remove:
                    author_scores.pop(aid, None)

        # 对作者得分做一次归一化，使最高分为 1.0，便于诊断与对比，不改变排序结果
        if author_scores:
            max_score = max(author_scores.values())
            if max_score > 0:
                for aid in author_scores:
                    author_scores[aid] = author_scores[aid] / max_score

        scored_authors = []
        for aid, total_score in sorted(author_scores.items(), key=lambda x: x[1], reverse=True):
            works = author_top_works.get(aid, [])
            if not works:
                continue

            per_author_papers = []
            for wid, contrib in works:
                meta = paper_map.get(wid, {})
                hits = paper_hit_terms.get(wid, [])
                per_author_papers.append(
                    {
                        "title": meta.get("title"),
                        "year": meta.get("year"),
                        "contribution": round(contrib, 6),
                        "hits": hits,
                    }
                )

            # 多篇代表作：按贡献度排序，取前 3 作为代表作列表
            per_author_papers.sort(key=lambda x: x.get("contribution", 0.0), reverse=True)
            top_papers = per_author_papers[:3]
            best_paper = top_papers[0] if top_papers else None

            # 标签汇总：统计作者所有正贡献论文中的标签命中次数
            tag_counter = collections.Counter()
            for p in per_author_papers:
                tag_counter.update(p.get("hits") or [])
            tag_stats = [{"term": t, "count": c} for t, c in tag_counter.most_common(10)]

            # 作者总论文数（未截断，仅用于 debug 展示）
            paper_cnt_author = author_raw_paper_cnt.get(aid, len(per_author_papers))

            # 最终作者得分：论文贡献 + 作者时间特征（activity/momentum）
            final_score = author_scores.get(aid, total_score)
            # 作者来源 term 拆解：按贡献排序取前 5
            atc = author_term_contrib.get(aid, {})
            top_terms_contrib = sorted(
                [(term_map.get(tid, ""), round(float(c), 6)) for tid, c in atc.items() if c > 0],
                key=lambda x: -x[1],
            )[:5]

            scored_authors.append({
                "aid": aid,
                "score": final_score,
                "raw_score": total_score,
                "top_paper": best_paper,
                "paper_count": paper_cnt_author,
                "top_papers": top_papers,
                "tag_stats": tag_stats,
                "top_terms_by_contribution": top_terms_contrib,
            })

        scored_authors.sort(key=lambda x: x["score"], reverse=True)
        sorted_terms = sorted(
            [(term_map.get(tid, ""), score_map.get(tid, 0)) for tid in score_map],
            key=lambda x: x[1], reverse=True
        )
        # Top20 学术词调试：term / cos_sim(JB) / anchor_sim / final_weight，便于观察谁在抢权重
        top20_term_debug = []
        if score_map and getattr(self, "_last_tag_purity_debug", None):
            debug_by_tid = {str(d["tid"]): d for d in self.debug_info.tag_purity_debug if d.get("tid") is not None}
            rank_factors = getattr(self, "_last_cluster_rank_factors", {}) or self.debug_info.cluster_rank_factors or {}
            for tid in sorted(score_map.keys(), key=lambda t: score_map.get(t, 0.0), reverse=True)[:20]:
                row = debug_by_tid.get(str(tid), {}) or {}
                top20_term_debug.append({
                    "term": (term_map.get(tid, ""))[:50],
                    "cos_sim": row.get("cos_sim"),
                    "anchor_sim": row.get("anchor_sim"),
                    "idf_term": row.get("idf_term"),
                    "degree_w": row.get("degree_w"),
                    "degree_w_expanded": row.get("degree_w_expanded"),
                    "hit_count": row.get("hit_count"),
                    "cluster_rank_factor": round(float(rank_factors.get(str(tid), 1.0)), 6),
                    "cluster_factor": row.get("cluster_factor"),
                    "job_penalty": row.get("job_penalty"),
                    "domain_span": row.get("domain_span"),
                    "weight": round(score_map.get(tid, 0.0), 6),
                })
        academic_kws = list(set(term_map.values()))
        # 过滤闭环：三份集合 + 指定 tid 是否在最终集合
        similar_to_pass = getattr(self, "_last_similar_to_pass", []) or self.debug_info.similar_to_pass
        similar_to_pass_tids = sorted(set(r["tid"] for r in similar_to_pass if r.get("tid") is not None))
        similar_to_raw_tids = getattr(self, "_last_raw_candidate_tids", []) or self.debug_info.raw_candidate_tids
        # 精检：仅选取少量高质量学术词参与论文检索
        final_term_ids_for_paper = self._select_terms_for_paper(score_map, term_map, max_terms=16)
        debug_check_tids = [(2280, "supervised learning"), (4045, "robot learning"), (152, "control (management)")]
        filter_closed_loop = {
            "similar_to_raw_tids": similar_to_raw_tids[:50],
            "similar_to_pass_tids": similar_to_pass_tids[:50],
            "final_term_ids_for_paper": final_term_ids_for_paper[:50],
            "final_term_count": len(final_term_ids_for_paper),
            "contains_check": {f"{name}({tid})": tid in final_term_ids_for_paper for tid, name in debug_check_tids},
        }
        # Top term 最终贡献表（Top20）：term, tid, final_weight, main_role, role_penalty, paper_count_hit, top_paper_contrib, total_paper_contrib
        debug_by_tid = {
            str(d["tid"]): d
            for d in (getattr(self, "_last_tag_purity_debug", None) or self.debug_info.tag_purity_debug or [])
            if d.get("tid") is not None
        }
        top_tids_by_weight = sorted(score_map.keys(), key=lambda t: score_map.get(t, 0.0), reverse=True)[:20]
        top_terms_final_contrib = []
        for tid in top_tids_by_weight:
            tid_s = str(tid)
            lst = term_paper_contrib.get(tid_s, [])
            pc = len(lst)
            top_c = max((c for _, c in lst), default=0.0)
            total_c = sum(c for _, c in lst)
            row = debug_by_tid.get(tid_s, {})
            top_terms_final_contrib.append({
                "term": term_map.get(tid_s, ""),
                "tid": int(tid) if tid_s.isdigit() else tid_s,
                "final_weight": round(score_map.get(tid_s, 0.0), 6),
                "main_role": row.get("main_role") or "none",
                "role_penalty": row.get("role_penalty"),
                "paper_count_hit": pc,
                "top_paper_contrib": round(top_c, 6),
                "total_paper_contrib": round(total_c, 6),
                "task_anchor_sim": row.get("task_anchor_sim"),
                "carrier_anchor_sim": row.get("carrier_anchor_sim"),
                "task_advantage": row.get("task_advantage"),
            })

        self.last_debug_info = {
            "active_domains": list(active_domain_set),
            "dominance": f"{dominance * 100:.1f}%",
            "expansion_pipeline": getattr(self, "_last_expansion_pipeline_stats", None) or self.debug_info.expansion_pipeline_stats or None,
            "similar_to_raw": getattr(self, "_last_similar_to_raw_rows", []) or self.debug_info.similar_to_raw_rows or [],
            "similar_to_agg": getattr(self, "_last_similar_to_agg", []) or self.debug_info.similar_to_agg or [],
            "similar_to_pass": similar_to_pass,
            "regex_str": debug_1.get("regex_str", ""),
            "job_ids": debug_1.get("job_ids", []),
            "job_previews": debug_1.get("job_previews", []),
            "anchor_debug": debug_1.get("anchor_debug", {}),
            "anchor_melt_stats": getattr(self, "_last_anchor_melt_stats", None) or self.debug_info.anchor_melt_stats or None,
            "supplement_anchors": getattr(self, "_last_supplement_anchors", []) or self.debug_info.supplement_anchors or [],
            "industrial_kws": industrial_kws,
            "anchor_detail": [f"{k}={v['term']}" for k, v in anchor_skills.items()],
            "academic_kws": academic_kws,
            "detailed_kws": getattr(self, "_last_expansion_raw_results", []) or self.debug_info.expansion_raw_results or [],
            "top_scored_terms": sorted_terms,
            "top20_term_debug": top20_term_debug,
            "recall_vocab_count": len(score_map),
            "work_count": all_works_count,
            "author_count": len(scored_authors),
            "top_samples": scored_authors[:50],
            "filter_closed_loop": filter_closed_loop,
            "top_terms_final_contrib": top_terms_final_contrib,
        }
        return [a["aid"] for a in scored_authors], self.last_debug_info

    def recall(self, query_vector, domain_id=None, query_text=None):
        """
        全链路调度（五阶段）：领域与锚点 → 学术词扩展 → 词权重（统一复杂公式）→ 图检索 → 作者打分。
        入参 domain_id 与向量路统一。返回 (author_id_list, duration_ms)。
        """
        if not self.graph:
            return [], 0
        start_t = time.time()

        # 阶段 1：领域与锚点
        active_domain_set, regex_str, anchor_skills, debug_1 = self._stage1_domain_and_anchors(
            query_vector, query_text=query_text, domain_id=domain_id
        )
        if not anchor_skills:
            return [], (time.time() - start_t) * 1000

        # 阶段 2：学术词扩展（仅产出候选，不算权）
        raw_candidates = self._stage2_expand_academic_terms(
            anchor_skills, active_domain_set, regex_str, query_vector, query_text=query_text
        )
        if not raw_candidates:
            return [], (time.time() - start_t) * 1000
        self.debug_info.raw_candidate_tids = sorted(
            set(r.get("tid") for r in raw_candidates if r.get("tid") is not None)
        )
        self._last_raw_candidate_tids = self.debug_info.raw_candidate_tids

        # 阶段 3：词权重（统一走复杂公式，传入锚点 ID 供锚点距离门控）
        anchor_vids = [int(k) for k in anchor_skills.keys()] if anchor_skills else None
        score_map, term_map, idf_map = self._stage3_word_weights(raw_candidates, query_vector, anchor_vids=anchor_vids)

        # 阶段 4：图检索
        author_papers_list = self._stage4_graph_search(
            [int(k) for k in score_map.keys()], regex_str
        )

        # 阶段 5：作者打分与排序（debug_1 中补上 regex_str、query_vector 供 last_debug_info 与 paper semantic gate）
        debug_1["regex_str"] = regex_str
        debug_1["query_vector"] = query_vector
        dominance = debug_1.get("dominance", 0.0)
        author_ids, _ = self._stage5_score_and_rank_authors(
            author_papers_list, score_map, term_map, active_domain_set, dominance, debug_1
        )

        elapsed_ms = (time.time() - start_t) * 1000
        return author_ids[: self.recall_limit], elapsed_ms


if __name__ == "__main__":
    from src.core.recall.label_means.label_debug_cli import run_label_debug_cli

    run_label_debug_cli()