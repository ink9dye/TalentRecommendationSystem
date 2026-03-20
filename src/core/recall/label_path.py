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
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from src.core.recall.input_to_vector import QueryEncoder
from src.core.recall.works_to_authors import accumulate_author_scores
from src.utils.domain_utils import DomainProcessor
from src.utils.domain_detector import DomainDetector
from config import (
    DB_PATH,
    VOCAB_P95_PAPER_COUNT,
    SIMILAR_TO_TOP_K,
    SIMILAR_TO_MIN_SCORE,
    SBERT_DIR,
    LABEL_DOMAIN_VECTORS_NPZ_PATH,
    LABEL_DOMAIN_VECTORS_META_PATH,
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
from src.core.recall.label_means.label_encoder_snapshots import (
    save_domain_vectors,
    try_load_domain_vectors,
)
from src.core.recall.label_means.simple_factors import (
    survey_decay_factor,
    coverage_norm_factor,
    paper_cluster_bonus,
    paper_jd_semantic_gate_factor,
    is_label_jd_title_gate_disabled,
)
from src.core.recall.label_means import advanced_metrics as label_means_adv, label_anchors, label_expansion
from src.core.recall.label_means.infra import LabelMeansInfra
from src.core.recall.label_means import term_scoring
from src.core.recall.label_pipeline.stage1_domain_anchors import Stage1Result
from src.core.recall.label_pipeline import (
    stage1_domain_anchors,
    stage2_expansion,
    stage3_term_filtering,
    stage4_paper_recall,
    stage5_author_rank,
)


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
        # 标签路追踪：Stage3 被过滤候选及原因
        self.dropped_with_reason: List[Dict[str, Any]] = []
        # Stage4 子阶段耗时（毫秒），供 recall 汇总打印
        self.stage4_sub_ms: Dict[str, float] = {}


def _print_label_sub_stage_ms(stage_label: str, sub_ms: Optional[Dict[str, Any]] = None) -> None:
    """打印单个大阶段内的子计时（毫秒）。sub_ms 为 {名称: 毫秒}。"""
    if not sub_ms:
        return
    parts: list[str] = []
    for name, ms in sub_ms.items():
        if isinstance(ms, (int, float)):
            parts.append(f"{name}={ms:.0f}ms")
    if parts:
        print(f"[Label {stage_label} 子阶段耗时] " + " ".join(parts))


def _emit_label_pipeline_checkpoints(checkpoints, debug_1=None, do_print=True):
    """
    标签路必查日志：每阶段关键计数，用于定位「从哪一步开始跑偏」。
    do_print=False 时不输出（不打印模式下不显示）。
    """
    if not checkpoints or not do_print:
        return
    parts = []
    first_bad = None
    for c in checkpoints:
        stage = c.get("stage", "?")
        ok = c.get("ok", True)
        if stage == "S1":
            parts.append(f"S1 anchors={c.get('anchors', 0)} domains={c.get('active_domains', 0)}")
        elif stage == "S2":
            parts.append(f"S2 raw_candidates={c.get('raw_candidates', 0)}")
        elif stage == "S3":
            parts.append(f"S3 score_map_terms={c.get('score_map_terms', 0)}")
        elif stage == "S3_select":
            parts.append(f"S3_select final_term_ids={c.get('final_term_ids', 0)}")
        elif stage == "S4":
            parts.append(f"S4 authors={c.get('authors', 0)} papers={c.get('papers', 0)}")
        elif stage == "S5":
            parts.append(f"S5 ranked={c.get('ranked_authors', 0)}")
        else:
            parts.append(f"{stage}={c}")
        if first_bad is None and not ok:
            first_bad = stage
    line = " | ".join(parts)
    print(f"[Label必查] {line}")
    if first_bad is not None:
        print(f"[Label必查] 首次异常阶段: {first_bad}（此处开始跑偏，请优先排查）")
    if debug_1 is not None and isinstance(debug_1, dict):
        debug_1["pipeline_checkpoints"] = checkpoints


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
                total_job_count=float(self.total_job_count),
            )
        except Exception:
            self.domain_detector = None

        # 论文标题离线索引（可选）：存在 WORK_TITLE_EMB_DB_PATH 时加载
        try:
            from src.core.recall.label_means.work_title_emb_store import WorkTitleEmbeddingStore

            self._work_title_emb_store = WorkTitleEmbeddingStore.open_optional()
        except Exception:
            self._work_title_emb_store = None

        if is_label_jd_title_gate_disabled():
            print(
                "[LabelRecallPath] LABEL_NO_JD_TITLE_GATE 已启用："
                "论文标题↔JD 向量门控已关闭（gate=1.0，Stage5 不再为门控编码标题）",
                flush=True,
            )

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
        使用 self._query_encoder 单例；优先读快照（label_domain_vectors*.npz/meta），缺失则编码并写入。
        """
        try:
            loaded = try_load_domain_vectors(
                SBERT_DIR,
                DOMAIN_MAP,
                LABEL_DOMAIN_VECTORS_NPZ_PATH,
                LABEL_DOMAIN_VECTORS_META_PATH,
            )
            if loaded is not None:
                self.domain_vectors = loaded
                print(
                    f"[OK] 领域向量已从快照加载 (共 {len(self.domain_vectors)} 个) "
                    f"path={LABEL_DOMAIN_VECTORS_NPZ_PATH}",
                    flush=True,
                )
                return

            encoder = self._query_encoder
            for domain_id, name in DOMAIN_MAP.items():
                vec, _ = encoder.encode(name)
                if vec is not None and vec.size > 0:
                    self.domain_vectors[str(domain_id)] = np.asarray(vec.flatten(), dtype=np.float32)
            if self.domain_vectors:
                print(f"[OK] 领域向量已构建 (共 {len(self.domain_vectors)} 个)，已写入快照", flush=True)
                save_domain_vectors(
                    self.domain_vectors,
                    SBERT_DIR,
                    DOMAIN_MAP,
                    LABEL_DOMAIN_VECTORS_NPZ_PATH,
                    LABEL_DOMAIN_VECTORS_META_PATH,
                )
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

    def _supplement_anchors_from_jd_vector(
        self,
        query_text,
        anchor_skills,
        total_j=None,
        top_k=None,
        active_domain_ids=None,
        jd_encode_cache=None,
    ):
        return label_anchors.supplement_anchors_from_jd_vector(
            self,
            query_text,
            anchor_skills,
            total_j=total_j,
            top_k=top_k,
            active_domain_ids=active_domain_ids,
            jd_encode_cache=jd_encode_cache,
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
        【统一权重】所有候选词一律走 term_scoring.calculate_final_weights，
        保持原有签名与返回值不变，作为薄代理。
        """
        return term_scoring.calculate_final_weights(
            self,
            raw_results,
            query_vector,
            anchor_vids=anchor_vids,
        )

    SELECT_TAG_PURITY_MIN = 0.40
    SELECT_SEMANTIC_MIN = 0.38
    SELECT_CTX_ONLY_CAP = 5
    SELECT_MIN_PAPER_COUNT = 3
    SELECT_FALLBACK_TOP = 12

    def _select_terms_for_paper(self, score_map, term_map, max_terms=20):
        """
        从全部打分学术词中选出一小撮高质量词用于论文检索。
        依赖：weight、领域纯度、语义相似度、来源(ctx_only 上限)、图论文数下限、轻量覆盖(edge_and_ctx 至少 1)。
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
        ctx_only_count = 0
        edge_and_ctx_passed = []

        for tid in ranked_tids:
            tid_s = str(tid)
            row = debug_rows.get(tid_s, {})
            weight = float(score_map.get(tid, 0.0) or 0.0)
            if weight <= 0.0:
                continue

            final_score = row.get("final_score")
            has_legacy = (row.get("raw_tag_purity") is not None or row.get("capped_tag_purity") is not None)
            if has_legacy:
                tag_purity = float(row.get("capped_tag_purity") or row.get("raw_tag_purity") or 0.0)
                if tag_purity and tag_purity < getattr(self, "SELECT_TAG_PURITY_MIN", 0.40):
                    continue
                cos_sim = float(row.get("cos_sim") or 0.0)
                anchor_sim = float(row.get("task_anchor_sim") or row.get("anchor_sim") or 0.0)
                sim_val = max(cos_sim, anchor_sim)
                if sim_val and sim_val < getattr(self, "SELECT_SEMANTIC_MIN", 0.38):
                    continue
            else:
                pass

            degree_w = int(row.get("degree_w") or 0)
            if degree_w < getattr(self, "SELECT_MIN_PAPER_COUNT", 3):
                continue

            source = (row.get("source") or row.get("origin") or "").strip().lower()
            if source == "ctx_only":
                if ctx_only_count >= getattr(self, "SELECT_CTX_ONLY_CAP", 5):
                    continue
                ctx_only_count += 1

            if source == "edge_and_ctx":
                edge_and_ctx_passed.append((tid, weight))

            selected.append(int(tid))
            if len(selected) >= int(max_terms):
                break

        if selected and edge_and_ctx_passed:
            has_edge_and_ctx = any(
                (debug_rows.get(str(t)) or {}).get("source", "").strip().lower() == "edge_and_ctx"
                for t in selected
            )
            if not has_edge_and_ctx:
                best_tid = edge_and_ctx_passed[0][0]
                selected = selected[:-1] + [int(best_tid)]

        if not selected:
            fallback = []
            for tid in ranked_tids[: min(getattr(self, "SELECT_FALLBACK_TOP", 12), len(ranked_tids))]:
                fallback.append(int(tid))
            return fallback

        return selected

    def _apply_word_quality_penalty(self, rec, query_vector):
        """
        薄代理：词级权重计算已下沉至 label_means.term_scoring。
        仅为兼容保留此方法签名，实际逻辑委托给 term_scoring._apply_word_quality_penalty。
        """
        from src.core.recall.label_means import term_scoring

        return term_scoring._apply_word_quality_penalty(self, rec, query_vector)

    # ---------- 词级权重因子（第二轮：因子接口抽取，仅封装现有逻辑） ----------

    def _compute_contribution(self, paper, context):
        """
        薄代理：论文级贡献度计算已下沉至 label_means.paper_scoring。
        仅为兼容保留此方法签名，实际逻辑委托给 paper_scoring.compute_contribution。
        """
        from src.core.recall.label_means import paper_scoring

        return paper_scoring.compute_contribution(self, paper, context)

    # ---------- 五阶段流程（便于维护与修改） ----------

    def _stage1_domain_and_anchors(self, query_vector, query_text=None, semantic_query_text=None, domain_id=None):
        return stage1_domain_anchors.run_stage1(
            self,
            query_vector,
            query_text=query_text,
            semantic_query_text=semantic_query_text,
            domain_id=domain_id,
        )

    def _stage2_expand_academic_terms(self, anchor_skills, active_domain_set, regex_str, query_vector, query_text=None, jd_profile=None):
        """
        阶段 2：学术词扩展。边路 + 语境向量路 + 簇扩展 + 共鸣/共现，返回候选列表（不计算最终词权）。
        传入 jd_profile 时启用层级守卫与泛词抑制。
        """
        return stage2_expansion.run_stage2(
            self,
            anchor_skills=anchor_skills,
            active_domain_set=active_domain_set,
            regex_str=regex_str,
            query_vector=query_vector,
            query_text=query_text,
            jd_profile=jd_profile,
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

    def _build_term_uniqueness_map(self, score_map, active_domain_set):
        """
        按 vocabulary_domain_stats 为每个 term 计算领域纯度，供 Stage5 论文贡献度乘数使用。
        领域专属性强的词（如 robotic arm）得高分，通用词（如 RL）得低分。
        返回: Dict[str, float]，vid_s -> [0, 1] 的 term_uniqueness（缺统计时默认 1.0）。
        """
        if not score_map or not active_domain_set or not getattr(self, "stats_conn", None):
            return {}
        active = set(int(x) for x in active_domain_set)
        out = {}
        for vid_s in score_map:
            try:
                vid = int(vid_s)
            except (TypeError, ValueError):
                out[vid_s] = 1.0
                continue
            row = None
            try:
                row = self.stats_conn.execute(
                    "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                    (vid,),
                ).fetchone()
            except Exception:
                out[vid_s] = 1.0
                continue
            if not row or not row[2]:
                out[vid_s] = 1.0
                continue
            degree_w_expanded = 0.0
            target_degree_w = 0.0
            try:
                dist = json.loads(row[2]) if isinstance(row[2], str) else row[2]
                expanded = self._expand_domain_dist(dist or {})
                degree_w_expanded = sum(expanded.values())
                target_degree_w = sum(expanded.get(str(d), 0) for d in active)
            except Exception:
                out[vid_s] = 1.0
                continue
            if degree_w_expanded <= 0:
                out[vid_s] = 1.0
                continue
            ratio = target_degree_w / degree_w_expanded
            out[vid_s] = max(0.0, min(1.0, float(ratio)))
        return out

    def _build_term_confidence_map(self, term_role_map, term_source_map):
        """
        按来源给每个 term 可信度：exact/bridge primary 高(0.95)、similar_to primary 中(0.9)、dense 中(0.75)、cluster/cooc 低(0.6)。
        供 paper_term_contrib = term_weight * term_confidence * paper_match_strength 使用。
        """
        out = {}
        for tid_str, role in (term_role_map or {}).items():
            role = (role or "").strip().lower()
            source = (term_source_map or {}).get(tid_str, "").strip().lower()
            if role == "primary":
                if source in ("similar_to", "jd_vector", ""):
                    out[tid_str] = 0.95
                else:
                    out[tid_str] = 0.90
            elif role == "dense_expansion":
                out[tid_str] = 0.75
            elif role in ("cluster_expansion", "cooc_expansion"):
                out[tid_str] = 0.60
            else:
                out[tid_str] = 0.80
        return out

    def _stage3_word_weights(self, raw_candidates, query_vector, anchor_vids=None):
        """
        阶段 3：词权重。统一走复杂公式（领域纯度、共鸣、语义守门、锚点距离门控等），无分支例外。
        返回: (score_map, term_map, idf_map, term_role_map, term_source_map, parent_anchor_map, parent_primary_map)。
        """
        return stage3_term_filtering.run_stage3(self, raw_candidates, query_vector, anchor_vids=anchor_vids)

    def _stage4_graph_search(
        self,
        vocab_ids,
        regex_str,
        score_map=None,
        term_retrieval_roles=None,
        term_meta=None,
        jd_text=None,
    ):
        """
        阶段 4：图检索。用学术词 ID 反查 Work 与 Author，带 Stage3 词权与 retrieval_role（paper_primary / paper_support）参与 paper_score。
        返回: list of { 'aid': str, 'papers': [ { wid, hits, weight, title, year, domains }, ... ] }。
        """
        score_map = score_map or {}
        term_scores = {
            int(tid): float(score_map.get(str(tid)) or score_map.get(tid) or 0.0)
            for tid in (vocab_ids or [])
        }
        return stage4_paper_recall.run_stage4(
            self, vocab_ids, regex_str,
            term_scores=term_scores,
            term_retrieval_roles=term_retrieval_roles,
            term_meta=term_meta,
            jd_text=jd_text,
        )

    def _stage5_score_and_rank_authors(self, author_papers_list, score_map, term_map, active_domain_set, dominance, debug_1):
        """
        阶段 5：作者打分与排序。按论文贡献度聚合、排序、截断，并组装 last_debug_info。
        返回: (author_id_list, last_debug_info)。
        """
        return stage5_author_rank.run_stage5(
            self,
            author_papers_list,
            score_map,
            term_map,
            active_domain_set,
            dominance,
            debug_1,
        )


    def recall(self, query_vector, domain_id=None, query_text=None, semantic_query_text=None):
        """
        全链路调度（五阶段）：领域与锚点 → 学术词扩展 → 词权重（统一复杂公式）→ 图检索 → 作者打分。
        入参 domain_id 与向量路统一。返回 (author_id_list, duration_ms)。
        """
        if not self.graph:
            self.last_debug_info = {"pipeline_checkpoints": [], "active_domains": [], "dominance": 0.0}
            return [], 0
        start_t = time.time()

        # 统一由 verbose 控制各模块是否打印
        from src.core.recall.label_means import label_expansion
        label_expansion.LABEL_EXPANSION_DEBUG = self.verbose
        label_expansion.STAGE2_VERBOSE_DEBUG = self.verbose
        stage3_term_filtering.LABEL_PATH_TRACE = self.verbose
        stage3_term_filtering.STAGE3_DETAIL_DEBUG = self.verbose
        term_scoring.STAGE3_DEBUG = self.verbose

        # 阶段 1：领域与锚点
        active_domain_set, regex_str, anchor_skills, debug_1 = self._stage1_domain_and_anchors(
            query_vector,
            query_text=query_text,
            semantic_query_text=semantic_query_text,
            domain_id=domain_id,
        )
        _print_label_sub_stage_ms("S1", debug_1.get("stage1_sub_ms") if isinstance(debug_1, dict) else None)
        checkpoints = []
        checkpoints.append({"stage": "S1", "anchors": len(anchor_skills or {}), "active_domains": len(active_domain_set or set()), "ok": bool(anchor_skills)})
        t_s1 = time.time()
        s1_ms = (t_s1 - start_t) * 1000
        if not anchor_skills:
            total_ms = (time.time() - start_t) * 1000
            print(f"[Label 各阶段耗时] S1={s1_ms:.0f}ms 总={total_ms:.0f}ms")
            _emit_label_pipeline_checkpoints(checkpoints, None, self.verbose)
            self.last_debug_info = dict(debug_1) if isinstance(debug_1, dict) else {}
            self.last_debug_info["pipeline_checkpoints"] = checkpoints
            return [], total_ms

        # 阶段 2：学术词扩展（仅产出候选，不算权）；传入 jd_profile 供层级守卫
        jd_profile = getattr(self._last_stage1_result, "jd_profile", None) if getattr(self, "_last_stage1_result", None) else None
        raw_candidates = self._stage2_expand_academic_terms(
            anchor_skills,
            active_domain_set,
            regex_str,
            query_vector,
            query_text=semantic_query_text or query_text,
            jd_profile=jd_profile,
        )
        checkpoints.append({"stage": "S2", "raw_candidates": len(raw_candidates or []), "ok": bool(raw_candidates)})
        t_s2 = time.time()
        s2_ms = (t_s2 - t_s1) * 1000
        if not raw_candidates:
            total_ms = (time.time() - start_t) * 1000
            print(f"[Label 各阶段耗时] S1={s1_ms:.0f}ms S2={s2_ms:.0f}ms 总={total_ms:.0f}ms")
            _emit_label_pipeline_checkpoints(checkpoints, debug_1, self.verbose)
            self.last_debug_info = dict(debug_1) if isinstance(debug_1, dict) else {}
            self.last_debug_info["pipeline_checkpoints"] = checkpoints
            return [], total_ms
        debug_1["stage2_anchor_evidence_table"] = getattr(self.debug_info, "stage2_anchor_evidence_table", None) or []
        self.debug_info.raw_candidate_tids = sorted(
            set(r.get("tid") for r in raw_candidates if r.get("tid") is not None)
        )
        self._last_raw_candidate_tids = self.debug_info.raw_candidate_tids
        # 诊断：新 Stage2 产出的 raw_candidates 写入 expansion_raw_results，供 Stage3 来源回溯表显示 source（anchor/similar_to 等）
        self.debug_info.expansion_raw_results = raw_candidates

        # 诊断回填：若新 Stage2 未写入 similar_to_pass（或为空），用 raw_candidates 中 source=similar_to 的项补全，保证面板有数
        similar_to_pass = getattr(self.debug_info, "similar_to_pass", None) or []
        if not similar_to_pass and raw_candidates:
            from_raw = [
                {
                    "tid": r.get("tid"),
                    "term": r.get("term", ""),
                    "sim_score": float(r.get("sim_score") or r.get("identity_score") or 0.0),
                    "hit_count": len(r.get("src_vids") or []),
                    "src_vids": list(r.get("src_vids") or []),
                }
                for r in raw_candidates
                if (r.get("source") or r.get("origin") or "").strip().lower() == "similar_to"
            ]
            if from_raw:
                self.debug_info.similar_to_pass = from_raw

        # 标签路追踪：source anchor、similar_to 原始候选（便于定位从哪一步开始跑偏）
        _label_trace = self.verbose or getattr(stage3_term_filtering, "LABEL_PATH_TRACE", False) or getattr(term_scoring, "STAGE3_DEBUG", False)
        if _label_trace:
            na = len(anchor_skills or {})
            _keys = list(anchor_skills.keys())[:5] if anchor_skills else []
            _terms_short: List[str] = []
            for k in _keys:
                v = anchor_skills.get(k)
                if isinstance(v, dict):
                    _terms_short.append(str(v.get("term", v.get("skill", k)))[:22])
                else:
                    _terms_short.append(str(k)[:22])
            _noisy = bool(getattr(label_expansion, "STAGE2_NOISY_DEBUG", False))
            print(
                "[标签路-source anchor] 数量=%s 预览(term 前5)=%s（全量 debug_info；键值明细开 STAGE2_NOISY_DEBUG）"
                % (na, _terms_short)
            )
            if _noisy and anchor_skills:
                _anchors = list(anchor_skills.keys())[:30]
                _anchor_preview = []
                for k in _anchors:
                    v = anchor_skills.get(k)
                    if isinstance(v, dict):
                        _anchor_preview.append(f"{k}={v.get('term', v.get('skill', k))!r}")
                    else:
                        _anchor_preview.append(str(k))
                print("  [标签路-source anchor 明细] 前30: %s" % (_anchor_preview[:20],))
            raw_rows = getattr(self.debug_info, "similar_to_raw_rows", None) or []
            sims = [float(r.get("sim_score") or 0.0) for r in raw_rows]
            max_sim = max(sims) if sims else 0.0
            print(
                "[标签路-similar_to 原始候选] 条数=%s max_sim=%.3f（逐行见 debug_info；明细打印开 STAGE2_NOISY_DEBUG）"
                % (len(raw_rows), max_sim)
            )
            if _noisy:
                _sim_preview_n = 10
                for i, r in enumerate(raw_rows[:_sim_preview_n]):
                    print(
                        "  %s src_vid=%s tid=%s term=%r sim_score=%s"
                        % (
                            i + 1,
                            r.get("src_vid"),
                            r.get("tid"),
                            (r.get("term") or "")[:28],
                            r.get("sim_score"),
                        )
                    )
                if len(raw_rows) > _sim_preview_n:
                    print("  ... 省略 %s 条" % (len(raw_rows) - _sim_preview_n))
            agg = getattr(self.debug_info, "similar_to_agg", None) or []
            pass_list = getattr(self.debug_info, "similar_to_pass", None) or []
            print("[标签路-similar_to] 聚合=%s 领域通过=%s" % (len(agg), len(pass_list)))
            breakdown = getattr(self.debug_info, "stage2a_term_source_breakdown", None) or []
            if breakdown:
                if _noisy:
                    print(
                        "[Stage2A 双路来源] term | sources | similar_to_score | conditioned_score | final_primary_score"
                    )
                    for row in breakdown[:50]:
                        term = (row.get("term") or "")[:32]
                        srcs = ",".join(row.get("sources") or []) or "-"
                        sim_s = row.get("similar_to_score")
                        cond_s = row.get("conditioned_score")
                        sim_str = "%.3f" % sim_s if sim_s is not None else "-"
                        cond_str = "%.3f" % cond_s if cond_s is not None else "-"
                        prim = row.get("final_primary_score") or 0
                        print("  %s | %s | %s | %s | %.3f" % (term, srcs, sim_str, cond_str, prim))
                    if len(breakdown) > 50:
                        print("  ... 共 %s 条" % len(breakdown))
                else:
                    print(
                        "[Stage2A 双路来源] n=%s（逐行开 STAGE2_NOISY_DEBUG）" % len(breakdown)
                    )

        # 阶段 3：词权重（统一走复杂公式，传入锚点 ID 供锚点距离门控）
        anchor_vids = [int(k) for k in anchor_skills.keys()] if anchor_skills else None
        stage3_out = self._stage3_word_weights(
            raw_candidates, query_vector, anchor_vids=anchor_vids
        )
        if len(stage3_out) == 8:
            score_map, term_map, idf_map, term_role_map, term_source_map, parent_anchor_map, parent_primary_map, paper_terms = stage3_out
        else:
            score_map, term_map, idf_map, term_role_map, term_source_map, parent_anchor_map, parent_primary_map = stage3_out
            paper_terms = []
        checkpoints.append({"stage": "S3", "score_map_terms": len(score_map or {}), "ok": bool(score_map)})
        t_s3 = time.time()
        s3_ms = (t_s3 - t_s2) * 1000

        # 精检：优先 family 保送式 paper_terms（每 family 1 primary + 1 support）；否则回退到 _select_terms_for_paper
        if paper_terms:
            final_term_ids_for_paper = [int(r.get("tid")) for r in paper_terms if r.get("tid") is not None]
            term_scores_for_paper = {int(r["tid"]): float(r.get("final_score") or 0.0) for r in paper_terms if r.get("tid") is not None}
            term_retrieval_roles = {int(r["tid"]): (r.get("retrieval_role") or "paper_support") for r in paper_terms if r.get("tid") is not None}
            term_meta_for_stage4 = {
                int(r["tid"]): {
                    "term": r.get("term"),
                    "parent_anchor": r.get("parent_anchor"),
                    "parent_primary": r.get("parent_primary"),
                    "retrieval_role": r.get("retrieval_role"),
                    "stage3_bucket": r.get("stage3_bucket"),
                    "term_role": r.get("term_role"),
                    "can_expand": r.get("can_expand"),
                    "mainline_hits": int(r.get("mainline_hits") or 0),
                    "paper_select_lane_tier": r.get("paper_select_lane_tier"),
                    "parent_anchor_final_score": float(
                        r.get("best_parent_anchor_final_score")
                        or r.get("parent_anchor_final_score")
                        or 0.0
                    ),
                    "parent_anchor_step2_rank": r.get("best_parent_anchor_step2_rank")
                    if r.get("best_parent_anchor_step2_rank") is not None
                    else r.get("parent_anchor_step2_rank"),
                    "object_like_penalty": (r.get("stage3_explain") or {}).get("object_like_penalty"),
                    "bonus_term_penalty": (r.get("stage3_explain") or {}).get("bonus_term_penalty"),
                    "generic_penalty": (r.get("stage3_explain") or {}).get("generic_penalty"),
                }
                for r in paper_terms
                if r.get("tid") is not None
            }
            term_family_keys = {int(r["tid"]): (r.get("family_key") or "") for r in paper_terms if r.get("tid") is not None}
        else:
            final_term_ids_for_paper = self._select_terms_for_paper(score_map, term_map, max_terms=20)
            term_scores_for_paper = None
            term_retrieval_roles = None
            term_meta_for_stage4 = None
            term_family_keys = None
        checkpoints.append({"stage": "S3_select", "final_term_ids": len(final_term_ids_for_paper or []), "ok": bool(final_term_ids_for_paper)})
        if getattr(term_scoring, "STAGE3_DEBUG", False):
            print("[final_term_ids_for_paper] tid | term | term_role | retrieval_role | parent_primary | score")
            for i, tid in enumerate(final_term_ids_for_paper[:30], 1):
                tid_str = str(tid)
                term = term_map.get(tid_str, "")
                st = term_role_map.get(tid_str, "")
                rr = (term_retrieval_roles or {}).get(tid) or "-"
                pp = parent_primary_map.get(tid_str, "")
                sc = score_map.get(tid_str, 0.0)
                print(f"  {i} {tid} | {term!r} | {st} | {rr} | {pp!r} | {sc:.3f}")
            if len(final_term_ids_for_paper) > 30:
                print(f"  ... 共 {len(final_term_ids_for_paper)} 条")
        # 将闭环信息提前挂到 debug_1，供 stage5_author_rank 复用/补全
        filter_closed_loop = debug_1.get("filter_closed_loop") or {}
        filter_closed_loop["final_term_ids_for_paper"] = final_term_ids_for_paper
        filter_closed_loop["final_term_count"] = len(final_term_ids_for_paper)
        debug_1["filter_closed_loop"] = filter_closed_loop

        # term_confidence：exact/bridge primary 高、similar_to primary 中、cluster/cooc 低，供 paper_term_contrib
        term_confidence_map = self._build_term_confidence_map(term_role_map, term_source_map)
        term_uniqueness_map = self._build_term_uniqueness_map(score_map, active_domain_set)
        debug_1["term_role_map"] = term_role_map
        debug_1["term_source_map"] = term_source_map
        debug_1["term_confidence_map"] = term_confidence_map
        debug_1["term_uniqueness_map"] = term_uniqueness_map
        debug_1["parent_anchor_map"] = parent_anchor_map
        debug_1["parent_primary_map"] = parent_primary_map
        if term_family_keys is not None:
            debug_1["term_family_keys"] = term_family_keys

        # 阶段 4：图检索（传入 term 分数与 retrieval_role，primary 权重大、support 次之）
        author_papers_list = self._stage4_graph_search(
            final_term_ids_for_paper,
            regex_str,
            score_map=term_scores_for_paper if term_scores_for_paper is not None else score_map,
            term_retrieval_roles=term_retrieval_roles,
            term_meta=term_meta_for_stage4,
            jd_text=semantic_query_text or query_text,
        )
        _print_label_sub_stage_ms("S4", getattr(self.debug_info, "stage4_sub_ms", None) or {})
        n_papers = sum(len(p.get("papers") or []) for p in (author_papers_list or []))
        checkpoints.append({"stage": "S4", "authors": len(author_papers_list or []), "papers": n_papers, "ok": bool(author_papers_list)})
        t_s4 = time.time()
        s4_ms = (t_s4 - t_s3) * 1000

        # 调试：按 term 汇总；top 论文用 **去重 wid + 该篇 hit_terms 数**，避免同一 wid 重复刷屏
        if getattr(term_scoring, "STAGE3_DEBUG", False) and final_term_ids_for_paper:
            _debug_rows = {}
            for row in (getattr(self, "_last_tag_purity_debug", None) or getattr(self.debug_info, "tag_purity_debug", None) or []):
                tid = row.get("tid")
                if tid is not None:
                    _debug_rows[str(tid)] = row
            _papers_per_tid = collections.defaultdict(set)
            _authors_per_tid = collections.defaultdict(set)
            _wid_hitn_for_tid: Dict[int, Dict[Any, int]] = collections.defaultdict(dict)
            for ap in (author_papers_list or []):
                aid = ap.get("aid")
                for p in ap.get("papers") or []:
                    wid = p.get("wid")
                    hits = p.get("hits") or []
                    hit_vids = set()
                    for h in hits:
                        if isinstance(h, dict) and h.get("vid") is not None:
                            hit_vids.add(int(h["vid"]))
                    hit_n = len(hit_vids)
                    for h in hits:
                        if not isinstance(h, dict):
                            continue
                        vv = h.get("vid")
                        if vv is None:
                            continue
                        tid_k = int(vv)
                        _papers_per_tid[tid_k].add(wid)
                        if aid:
                            _authors_per_tid[tid_k].add(str(aid))
                        prev = _wid_hitn_for_tid[tid_k].get(wid)
                        if prev is None or hit_n > prev:
                            _wid_hitn_for_tid[tid_k][wid] = hit_n
            print(
                "[final_term_ids_for_paper] tid | term | source_type | parent_primary | "
                "papers_before_filter | papers_after_filter | authors_before_merge | "
                "top_unique_papers(wid:hit_terms_count,...)"
            )
            for i, tid in enumerate(final_term_ids_for_paper[:30], 1):
                tid_str = str(tid)
                term = term_map.get(tid_str, "")
                st = term_source_map.get(tid_str, "")
                pp = parent_primary_map.get(tid_str, "")
                row = _debug_rows.get(tid_str, {})
                papers_before = int(row.get("degree_w") or 0)
                papers_after = len(_papers_per_tid.get(int(tid), set()))
                authors_before = len(_authors_per_tid.get(int(tid), set()))
                items = sorted(
                    (_wid_hitn_for_tid.get(int(tid)) or {}).items(),
                    key=lambda x: (-x[1], str(x[0])),
                )[:10]
                top_str = ",".join(f"{w}:{n}" for w, n in items) if items else "-"
                print(f"  {i} {tid} | {term!r} | {st} | {pp!r} | {papers_before} | {papers_after} | {authors_before} | {top_str}")
            if len(final_term_ids_for_paper) > 30:
                print(f"  ... 共 {len(final_term_ids_for_paper)} 条")

        # 阶段 5：作者打分与排序（debug_1 中补上 regex_str、query_vector 供 last_debug_info 与 paper semantic gate）
        debug_1["regex_str"] = regex_str
        debug_1["query_vector"] = query_vector
        # paper_scoring：support/primary 附着衰减、主 line 浅开关、Stage5 审计用
        debug_1["term_paper_meta"] = term_meta_for_stage4 if term_meta_for_stage4 is not None else {}
        debug_1["term_retrieval_roles"] = term_retrieval_roles if term_retrieval_roles is not None else {}
        dominance = debug_1.get("dominance", 0.0)
        author_ids, last_debug_info = self._stage5_score_and_rank_authors(
            author_papers_list, score_map, term_map, active_domain_set, dominance, debug_1
        )
        checkpoints.append({"stage": "S5", "ranked_authors": len(author_ids or []), "ok": True})
        t_s5 = time.time()
        s5_ms = (t_s5 - t_s4) * 1000
        elapsed_ms = (t_s5 - start_t) * 1000
        debug_1["pipeline_checkpoints"] = checkpoints
        if last_debug_info is not None:
            last_debug_info["pipeline_checkpoints"] = checkpoints
        print(f"[Label 各阶段耗时] S1={s1_ms:.0f}ms S2={s2_ms:.0f}ms S3={s3_ms:.0f}ms S4={s4_ms:.0f}ms S5={s5_ms:.0f}ms 总={elapsed_ms:.0f}ms")
        _print_label_sub_stage_ms("S5", (last_debug_info or {}).get("stage5_sub_ms"))
        _emit_label_pipeline_checkpoints(checkpoints, debug_1, self.verbose)

        author_list = (author_ids or [])[: self.recall_limit]
        meta_list = [
            {
                "author_id": str(aid),
                "label_rank": i + 1,
                "label_score_raw": None,
                "label_evidence": None,
            }
            for i, aid in enumerate(author_list)
        ]
        return meta_list, elapsed_ms


if __name__ == "__main__":
    from src.core.recall.label_means.label_debug_cli import run_label_debug_cli

    run_label_debug_cli()