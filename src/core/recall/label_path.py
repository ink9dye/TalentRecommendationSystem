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
from src.core.recall.works_to_authors import accumulate_author_scores
from src.utils.domain_utils import DomainProcessor
from config import (
    CONFIG_DICT, JOB_INDEX_PATH, JOB_MAP_PATH,
    VOCAB_INDEX_PATH, VOCAB_MAP_PATH, DB_PATH, VOCAB_STATS_DB_PATH,
    VOCAB_P95_PAPER_COUNT, SIMILAR_TO_TOP_K, SIMILAR_TO_MIN_SCORE,
)
from src.utils.domain_config import (
    DOMAIN_DECAY_RATES,
    DEFAULT_DECAY_RATE,
    DOMAIN_MAP,
)
from src.utils.tools import apply_text_decay, get_decay_rate_for_domains, compute_time_decay


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

    def __init__(self, recall_limit=200, verbose=False):
        self.recall_limit = recall_limit
        self.verbose = verbose
        self.current_year = datetime.now().year
        self._init_resources()

        # 预载入统计数据，用于计算后续 IDF 与 熔断率
        self.total_work_count = self._get_node_count("Work")
        self.total_job_count = self._get_node_count("Job")

        # 编码器单例：全流程共用，避免重复加载模型（领域向量、锚点补充、语境扩展、main 均复用）
        self._query_encoder = QueryEncoder()
        # 领域向量：用于从 Top5 候选领域中按 query 相似度选 Top3
        self.domain_vectors = {}
        self._build_domain_vectors()

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

            # E. 概念簇缓存：cluster_id -> [voc_id]，voc_id -> [(cluster_id, score)]
            self.cluster_members = collections.defaultdict(list)
            self.voc_to_clusters = collections.defaultdict(list)
            try:
                cur = self.stats_conn.execute(
                    "SELECT cluster_id, voc_id FROM cluster_members"
                )
                for cid, vid in cur:
                    self.cluster_members[int(cid)].append(int(vid))
                cur = self.stats_conn.execute(
                    "SELECT voc_id, cluster_id, score FROM vocabulary_cluster"
                )
                for vid, cid, sc in cur:
                    self.voc_to_clusters[int(vid)].append((int(cid), float(sc)))
            except Exception:
                # 概念簇索引缺失时退化为无簇模式
                self.cluster_members = collections.defaultdict(list)
                self.voc_to_clusters = collections.defaultdict(list)

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

    # --- 第二阶段：锚点技能提取（先熔断 -> Top30 -> Top20）---
    def _extract_anchor_skills(self, target_job_ids, query_vector=None, total_j=None):
        """
        【工业侧：岗位技能提取】先熔断 -> Top30 -> Top20。
        逻辑：
          - 从 Top20 岗位的 REQUIRE_SKILL 得到 (vid, term, job_freq)；
          - 熔断：仅保留 cov_j < ANCHOR_MELT_COV_J 的技能（全图岗位覆盖率，编程语言等泛词被滤掉）；
          - 在熔断后按 job_freq 降序取 Top30；
          - 可选：用清洗出的 raw skills 过滤；
          - 若有 query_vector：语义重排后取 Top20；否则直接取前 20。
        """
        total_j = float(total_j or 0)
        if total_j <= 0:
            total_j = self.total_job_count

        cleaned_terms = set()
        try:
            cursor = self.graph.run(
                "MATCH (j:Job) WHERE j.id IN $j_ids RETURN j.skills AS skills",
                j_ids=target_job_ids[: self.ANCHOR_JOBS_TOP_K]
            )
            for row in cursor:
                if row.get("skills"):
                    cleaned_terms |= self._clean_job_skills(str(row["skills"]))
        except Exception:
            pass
        if not cleaned_terms:
            cleaned_terms = None

        # 1) Top20 岗位内所有 (vid, term, job_freq)，不先 limit
        cypher1 = """
        MATCH (j:Job) WHERE j.id IN $j_ids
        MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
        WITH v.id AS vid, v.term AS term, count(DISTINCT j.id) AS job_freq
        RETURN vid, term, job_freq
        ORDER BY job_freq DESC
        """
        rows = []
        try:
            for r in self.graph.run(cypher1, j_ids=target_job_ids[: self.ANCHOR_JOBS_TOP_K]):
                if r.get("term") and len(str(r.get("term") or "")) > 1:
                    rows.append(dict(r))
        except Exception:
            rows = []

        if not rows:
            self._last_anchor_melt_stats = {"before_melt": 0, "after_melt": 0, "after_top30": 0, "melted_sample": []}
            return {}

        # 2) 查全图每个 vid 的 global_job_count，算 cov_j，熔断
        v_ids = list({int(r["vid"]) for r in rows})
        global_count = {}
        try:
            cypher2 = """
            UNWIND $v_ids AS vid
            MATCH (v:Vocabulary {id: vid})<-[:REQUIRE_SKILL]-(j:Job)
            RETURN vid, count(j) AS cnt
            """
            for r in self.graph.run(cypher2, v_ids=v_ids):
                global_count[int(r["vid"])] = int(r.get("cnt") or 0)
        except Exception:
            pass

        melt_threshold = float(self.ANCHOR_MELT_COV_J)
        rows_with_cov = []
        melted_terms = []
        for r in rows:
            vid = int(r.get("vid"))
            g = global_count.get(vid, 0)
            cov_j = (g / total_j) if total_j else 0
            if cov_j >= melt_threshold:
                melted_terms.append((r.get("term") or "", round(cov_j, 4)))
                continue
            # 共识门槛：避免“单岗位偏航技能”触发学术扩展
            if int(r.get("job_freq") or 0) < int(self.ANCHOR_MIN_JOB_FREQ):
                continue
            rows_with_cov.append((r, cov_j))

        self._last_anchor_melt_stats = {
            "before_melt": len(rows),
            "after_melt": len(rows_with_cov),
            "melted_sample": melted_terms[:25],
            "melt_threshold": melt_threshold,
        }

        # 3) 熔断后按 job_freq 取 Top30
        rows_with_cov.sort(key=lambda x: (x[0].get("job_freq") or 0), reverse=True)
        rows = [x[0] for x in rows_with_cov[: self.ANCHOR_FREQ_TOP_K]]
        self._last_anchor_melt_stats["after_top30"] = len(rows)

        # 4) 可选：清洗词过滤
        if cleaned_terms is not None:
            rows = [r for r in rows if (r.get("term") or "").lower() in cleaned_terms]

        if not rows:
            return {}

        # 5) 语义重排取 Top20
        if query_vector is not None:
            q = np.asarray(query_vector, dtype=np.float32).flatten()
            scored = []
            for r in rows:
                vid = str(r.get("vid"))
                idx = self.vocab_to_idx.get(vid)
                sim = -1.0
                if idx is not None and q.size > 0:
                    try:
                        sim = float(np.dot(self.all_vocab_vectors[idx], q))
                    except Exception:
                        sim = -1.0
                scored.append((sim, int(r.get("job_freq") or 0), r))
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            # 相似度阈值：低于阈值直接丢弃（避免与当前 JD 偏离的岗位技能锚点混入）
            sim_min = float(self.ANCHOR_SIM_MIN)
            scored_keep = [x for x in scored if (x[0] is not None and float(x[0]) >= sim_min)]
            self._last_anchor_melt_stats["sim_min"] = sim_min
            self._last_anchor_melt_stats["sim_kept"] = len(scored_keep)
            self._last_anchor_melt_stats["sim_dropped"] = max(0, len(scored) - len(scored_keep))
            rows = [x[2] for x in scored_keep[: self.ANCHOR_FINAL_TOP_K]]
        else:
            rows = rows[: self.ANCHOR_FINAL_TOP_K]

        return {str(r.get("vid")): {"term": r.get("term")} for r in rows}

    def _supplement_anchors_from_jd_vector(self, query_text, anchor_skills, total_j=None, top_k=15):
        """
        用 JD 向量直接搜 vocabulary，取与当前 JD 语义最接近的 top_k 个词作为补充锚点（不硬编码强词）。
        补充词也做熔断：仅当 cov_j < ANCHOR_MELT_COV_J 时才并入。返回本次新增的 [(vid, term), ...] 供调试。
        """
        if not query_text or not getattr(self, "vocab_index", None):
            self._last_supplement_anchors = []
            return
        total_j = float(total_j or 0) or self.total_job_count
        self._load_vocab_meta()
        encoder = self._query_encoder
        jd_snippet = (query_text or "").strip()[:500]
        if not jd_snippet:
            self._last_supplement_anchors = []
            return

        v_jd = encoder.model.encode(
            [jd_snippet],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        v_jd = np.asarray(v_jd, dtype=np.float32).reshape(1, -1)
        k = min(int(top_k), 30)
        scores, labels = self.vocab_index.search(v_jd, k)
        added = []
        melt_threshold = float(self.ANCHOR_MELT_COV_J)
        v_ids_to_check = []
        candidates = []
        for score, tid in zip(scores[0], labels[0]):
            tid = int(tid)
            if tid <= 0:
                continue
            if str(tid) in anchor_skills:
                continue
            term, _ = self._vocab_meta.get(tid, ("", ""))
            if not term or len(term) < 2:
                continue
            candidates.append((tid, term, float(score)))
            v_ids_to_check.append(tid)

        if not v_ids_to_check:
            self._last_supplement_anchors = []
            return

        global_count = {}
        try:
            cypher = """
            UNWIND $v_ids AS vid
            MATCH (v:Vocabulary {id: vid})<-[:REQUIRE_SKILL]-(j:Job)
            RETURN vid, count(j) AS cnt
            """
            for r in self.graph.run(cypher, v_ids=v_ids_to_check):
                global_count[int(r["vid"])] = int(r.get("cnt") or 0)
        except Exception:
            global_count = {vid: 0 for vid in v_ids_to_check}

        for tid, term, score in candidates:
            g = global_count.get(tid, 0)
            cov_j = (g / total_j) if total_j else 0
            if cov_j >= melt_threshold:
                continue
            anchor_skills[str(tid)] = {"term": term}
            added.append((tid, term, round(score, 4), round(cov_j, 4)))

        self._last_supplement_anchors = added
        self._cached_v_jd = v_jd

    # --- 第三阶段：语义扩展 ---
    # 融合权重：相似边 0.2，语境向量 0.8；多次命中奖励：泛词(work_count大)不奖励，其余按 ln(min(hit,5)) 计算并封顶
    CTX_EDGE_WEIGHT = 0.8
    EDGE_WEIGHT = 0.2
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

    def _expand_semantic_map(self, core_vids, anchor_skills, domain_regex=None, query_vector=None, query_text=None, return_raw=False):
        """
        【学术共鸣 + 共现领域版】语义扩展引擎。

        逻辑：工业锚点激发 -> 边路(SIMILAR_TO)与语境向量路(JD+锚点)候选 -> 0.2/0.8 融合 + 多次命中奖励 ->
              学术候选池 -> 共鸣/共现指标 -> [若 return_raw=False] 最终打分。
        return_raw=True 时仅返回 raw_results（供阶段化流程中 stage2→stage3 拆分）；否则返回 (score_map, term_map, idf_map)。
        """
        regex = domain_regex if domain_regex else ".*"

        # 1. 边路：SIMILAR_TO 图检索
        raw_edge = self._query_expansion_with_topology(core_vids, regex)
        # 2. 语境向量路：JD 片段 + 锚点 编码后在 vocabulary 索引中检索（仅学术词）
        raw_ctx = self._query_expansion_by_context_vector(anchor_skills, query_text, regex, topk_per_anchor=5) if query_text else []

        # 3. 按 tid 合并：sim_merged = (0.2*sim_edge + 0.8*sim_ctx) * hit_bonus，src_vids 取并集
        edge_map = {rec["tid"]: rec for rec in raw_edge}
        ctx_map = {rec["tid"]: rec for rec in raw_ctx}
        all_tids = set(edge_map.keys()) | set(ctx_map.keys())
        raw_merged = []
        for tid in all_tids:
            rec_e = edge_map.get(tid)
            rec_c = ctx_map.get(tid)
            sim_edge = float(rec_e["sim_score"]) if rec_e else 0.0
            sim_ctx = float(rec_c["sim_score"]) if rec_c else 0.0
            src_vids = set()
            if rec_e:
                src_vids.update(rec_e.get("src_vids") or [])
            if rec_c:
                src_vids.update(rec_c.get("src_vids") or [])
            hit = len(src_vids)
            base = self.EDGE_WEIGHT * sim_edge + self.CTX_EDGE_WEIGHT * sim_ctx
            # 泛词 gate：work_count(=degree_w) 大的词不享受命中奖励；其余 hit 封顶到 5 防止爆炸
            degree_w = int((rec_e or rec_c).get("degree_w", 0) or 0)
            if degree_w >= self.HIT_BONUS_DEGREE_GATE:
                bonus = 1.0
            else:
                hit_eff = min(hit, self.HIT_BONUS_HIT_CAP) if hit >= 1 else 1
                bonus = min(self.HIT_BONUS_CAP, 1.0 + self.HIT_BONUS_BETA * math.log(hit_eff))
            sim_merged = base * bonus
            rec = dict(rec_e or rec_c)
            rec["sim_score"] = sim_merged
            rec["src_vids"] = sorted(src_vids)
            rec["hit_count"] = hit
            raw_merged.append(rec)

        raw_results = raw_merged
        if not raw_results:
            return [] if return_raw else ({}, {}, {})

        # 1.5 概念簇扩展：在簇内为每个 seed 找少量近邻学术词（top5，且簇内相似度≥0.6），作为次级扩展
        raw_results = self._expand_with_clusters(raw_results, regex, topk_per_seed=7)

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

        # 4. 应用数学公式计算最终动态权重（含共鸣、共现广度惩罚、共现纯度奖励）；若 return_raw 则仅返回候选列表供 stage3 统一算权
        self._last_expansion_raw_results = raw_results
        if return_raw:
            return raw_results
        return self._calculate_final_weights(raw_results, query_vector)

    def _expand_with_clusters(self, raw_results, domain_regex, topk_per_seed=5, weight_decay=0.4):
        """
        使用概念簇对第一层学术词做局部扩展：
          1. 对每个 seed 词，找到其所属簇（取 score 最高的一个）。
          2. 在该簇内只保留 sim_in_cluster >= 0.6 的成员，再按相似度取 topk_per_seed 个作为扩展词。
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

        return raw_results

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

            # 大词与领域纯度软惩罚（不再直接过滤）
            T = float(VOCAB_P95_PAPER_COUNT)
            if degree_w > T:
                x = min(float(degree_w) / T, 4.0)
                size_penalty = (1.0 / x) ** 2
            else:
                size_penalty = 1.0

            eps = 0.05
            r = max(domain_ratio, eps)
            domain_penalty = r ** 2
            rec = by_tid[tid]
            src_vids = sorted(rec["src_vids"])
            results.append({
                "tid": tid,
                "term": rec["term"] or self._vocab_meta.get(tid, ("", None))[0],
                "sim_score": rec["ctx_sim"] * size_penalty * domain_penalty,
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
            self._last_similar_to_raw_rows = []
            self._last_similar_to_agg = []
            self._last_similar_to_pass = []
            return []

        # 缓存 SIMILAR_TO 原始扩展行（仅保留必要字段，避免 debug 体积过大）
        self._last_similar_to_raw_rows = [
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
        self._last_similar_to_agg = [
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
            # 领域纯度过滤（分档）：由硬过滤改为软惩罚（次方形式）
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

            # 大词 size_penalty 与领域纯度 domain_penalty（软上限，次方惩罚）
            T = float(VOCAB_P95_PAPER_COUNT)
            if degree_w > T:
                x = min(float(degree_w) / T, 4.0)
                size_penalty = (1.0 / x) ** 2
            else:
                size_penalty = 1.0

            eps = 0.05
            r = max(domain_ratio, eps)
            domain_penalty = r ** 2

            pipeline["n_final"] += 1
            rec = by_tid[tid]
            rec["degree_w"] = degree_w
            rec["degree_w_expanded"] = degree_w_expanded
            rec["target_degree_w"] = target_degree_w
            rec["domain_span"] = domain_span
            rec["cov_j"] = 0.0
            # 将惩罚作用在 sim_score 上，避免大词/低纯度词完全统治排序
            rec["sim_score"] = float(rec.get("sim_score", 0.0) or 0.0) * size_penalty * domain_penalty
            results.append(rec)
        self._last_expansion_pipeline_stats = pipeline

        # 保存“stats/领域过滤后”的第一层学术词（similar_to 通过项）供诊断
        self._last_similar_to_pass = [
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

    def _calculate_final_weights(self, raw_results, query_vector):
        """
        【统一权重】所有候选词一律走 _apply_word_quality_penalty（含领域纯度、共鸣、语义守门等），
        无 sim_score/degree_w 分支例外，流程清晰、领域占比一致生效。
        仅当 rec 缺少必要字段时做安全回退，避免异常。
        """
        score_map, term_map, idf_map = {}, {}, {}
        required = ("degree_w", "cov_j", "domain_span", "target_degree_w")
        self._last_tag_purity_debug = []

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
        # 与边路过滤同一口径：用展开后的总和作分母，purity 自然 ≤ 1
        degree_w_expanded = rec.get('degree_w_expanded')
        if degree_w_expanded is None:
            degree_w_expanded = degree_w

        # 1. 计算语义纯度 (Tag Purity)，分母用 degree_w_expanded，与 target_degree_w 同口径，purity 自然 ≤ 1
        # 逻辑：目标领域的产出占比（展开后统计），过滤“挂羊头卖狗肉”的词汇
        raw_tag_purity = (rec['target_degree_w'] / degree_w_expanded) if degree_w_expanded else 0.0
        tag_purity = min(1.0, raw_tag_purity)

        # 2. 计算基础学术稀缺度 (IDF)（轻化版：压缩 + 开根号，避免极端放大/压制）
        idf_raw = math.log10(self.total_work_count / (degree_w + 1))
        # 将 IDF 压到 [0.2, 2.0] 区间，再做 0.5 次方，弱化影响
        idf_clamped = min(2.0, max(0.2, idf_raw))
        idf_val = math.pow(idf_clamped, 0.5)

        # 3. 计算断崖式岗位惩罚 (Suppression of generic job terms)
        job_penalty = 1.0 + math.exp(300.0 * (cov_j - 0.005))

        # --- 4. 【核心修改：学术共鸣与共振奖励 + hit_count 直接加权】 ---
        # 逻辑 A：hit_count 代表有多少工业锚点支撑这个词；单锚点(hit_count=1)多为 python 等扩出的噪音，直接降权
        hit_count = rec.get('hit_count', 1)
        hit_count_factor = 0.4 if hit_count == 1 else 1.0  # 单锚点 0.4x，多锚点共识保持 1.0

        # 逻辑 B：resonance 代表该词与当前其他候选词的共现；anchor_resonance 代表与第一层学术词的共现
        resonance = rec.get('resonance', 0.0)
        anchor_resonance = rec.get('anchor_resonance', 0.0)

        # 【强孤立点熔断】若既无学术共鸣、又无锚点共鸣且仅被单锚点击中，则视为噪声标签，直接舍弃
        if anchor_resonance <= 0 and resonance <= 0 and hit_count <= 1:
            return 0.0, idf_val

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

        # 记录供诊断（包含 cos_sim，用于后续调整 SEMANTIC_MIN 等阈值）
        if getattr(self, "_last_tag_purity_debug", None) is not None:
            self._last_tag_purity_debug.append({
                "tid": rec.get("tid"),
                "term": (rec.get("term") or "")[:40],
                "degree_w": degree_w,
                "degree_w_expanded": degree_w_expanded,
                "target_degree_w": rec.get("target_degree_w"),
                "raw_tag_purity": round(raw_tag_purity, 6),
                "capped_tag_purity": round(tag_purity, 6),
                "cos_sim": round(cos_sim, 6),
            })

        # 语义硬拦截：低相关词直接置 0，避免稀缺词（高 IDF）“以小博大”带偏召回
        if cos_sim < float(self.SEMANTIC_MIN):
            return 0.0, idf_val

        # 应用 3 次方非线性惩罚，实现对弱相关词的断崖式拦截
        semantic_factor = math.pow(max(0, cos_sim), 3)

        # 基于本次查询的语义相似度分布做 Top20% / 中60% / 底20% 分段加权：
        # - Top 20%: 语义最贴近 JD 的学术词，适度放大（×1.5）；
        # - 中间 60%: 语义相关但非核心，保持原权重（×1.0）；
        # - 底部 20%: 勉强过阈值的边缘词，略微压制（×0.5），避免“跑题但同领域”的词抢占过多权重。
        bucket_factor = 1.0
        p20 = getattr(self, "_semantic_p20", None)
        p80 = getattr(self, "_semantic_p80", None)
        if p20 is not None and p80 is not None:
            try:
                if cos_sim >= p80:
                    bucket_factor = 1.5
                elif cos_sim <= p20:
                    bucket_factor = 0.5
            except Exception:
                bucket_factor = 1.0
        semantic_factor *= bucket_factor

        # 基础骨架：sim * log(1+hits) * purity / log(1+paper_count)
        sim_term = semantic_factor
        hits_term = math.log1p(hit_count)
        purity_term = tag_purity
        freq_term = math.log1p(degree_w)
        term_backbone = (sim_term * hits_term * purity_term) / max(1e-6, freq_term)

        # --- 6. 【共现领域惩罚与奖励】基于 vocab_stats 的 vocabulary_cooccurrence ---
        # 降权：与各种领域的词都共现 → 万金油 → 共现伙伴平均领域跨度大 → cooc_span 大 → 乘小于 1 的因子
        cooc_span = rec.get('cooc_span', 0.0)
        span_penalty = 1.0 / (1.0 + math.log1p(cooc_span)) if cooc_span > 0 else 1.0
        # 加权：只跟特定领域的词共现 → 专精 → 共现伙伴目标领域占比高 → cooc_purity 大 → 乘大于 1 的因子
        cooc_purity = rec.get('cooc_purity', 0.0)
        purity_bonus = (1.0 + math.log1p(cooc_purity)) if cooc_purity > 0 else 1.0

        # 7. 【最终公式：立体的评价体系】
        # 骨架：sim * log(1+hits) * purity / log(1+paper_count)
        # 细节：轻化 IDF + 岗位惩罚 + 领域跨度 + 共鸣/共现修正
        dynamic_weight = (
                term_backbone
                * idf_val  # 轻化后的 IDF，只做微调
                / job_penalty
                / domain_span
                * convergence_bonus  # 学术共鸣（与本次搜索词共现）加成
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

        # 4. 综述降权 + 文本类型降权（统一规则）
        hit_count = len(valid_hids)
        survey_decay = (1.0 / math.pow(hit_count, 2)) if hit_count > 1 else 1.0
        # 标题文本降权：survey/overview/review/handbook + data from:/dataset:/supplementary data
        text_decay = apply_text_decay(raw_title)
        survey_decay *= text_decay

        # 5. 指数级紧密度加成 (1+prox)^n
        proximity = self._calculate_proximity(valid_hids)
        proximity_bonus = math.pow(1.0 + proximity, hit_count)

        # 6. 时序衰减：统一调用工具层 compute_time_decay（按领域配置 decay_rate）
        time_decay = compute_time_decay(paper.get('year', 2000), context['active_domain_set'])

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
        cluster_count = len(cluster_ids)
        cluster_bonus = math.log1p(cluster_count) if cluster_count > 0 else 1.0

        # 7. 命中标签数量归一化：防止“命中标签越多 → 单论文贡献爆炸”主导作者排序
        # 采用次线性归一：log(2 + hit_count)，保证 1 标签与多标签论文在同一量级
        if hit_count > 0:
            coverage_norm = 1.0 / math.log(2.0 + hit_count)
        else:
            coverage_norm = 1.0

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
        return score, hit_terms

    # ---------- 五阶段流程（便于维护与修改） ----------

    def _stage1_domain_and_anchors(self, query_vector, query_text=None, domain_id=None):
        """
        阶段 1：领域与锚点。确定目标领域、工业侧锚点技能。
        返回: (active_domain_set, regex_str, anchor_skills, debug_1)。
        debug_1 含 job_ids, job_previews, anchor_debug, dominance, industrial_kws, anchor_skills 等供阶段 5 诊断用。
        无锚点时 anchor_skills 为空 dict。
        """
        job_ids, inferred_domains, dominance = self._detect_domain_context(query_vector)
        job_previews = self._get_job_previews(job_ids)
        anchor_skills = self._extract_anchor_skills(
            job_ids, query_vector=query_vector, total_j=self.total_job_count
        )
        anchor_debug = self._get_anchor_debug_stats(job_ids[:20], self.total_job_count) if job_ids else {}
        if query_text and anchor_skills is not None:
            self._supplement_anchors_from_jd_vector(
                query_text, anchor_skills, total_j=self.total_job_count, top_k=self.JD_VOCAB_TOP_K
            )
        if not anchor_skills:
            return set(), "", {}, {"job_ids": job_ids, "job_previews": job_previews, "anchor_debug": anchor_debug, "dominance": dominance}

        industrial_kws = [v["term"] for v in anchor_skills.values()]
        if domain_id and str(domain_id) != "0":
            active_domain_set = DomainProcessor.to_set(domain_id)
            if len(active_domain_set) > self.ACTIVE_DOMAINS_TOP_K:
                active_domain_set = set(list(sorted(active_domain_set))[: self.ACTIVE_DOMAINS_TOP_K])
        else:
            candidate_5 = inferred_domains
            if self.domain_vectors and len(candidate_5) > self.ACTIVE_DOMAINS_TOP_K:
                q = np.asarray(query_vector, dtype=np.float32).flatten()
                if q.size > 0:
                    scores = []
                    for d in candidate_5:
                        dv = self.domain_vectors.get(str(d))
                        if dv is not None and dv.size == q.size:
                            sc = float(np.dot(q, dv))
                            scores.append((d, sc))
                    scores.sort(key=lambda x: x[1], reverse=True)
                    active_domain_set = set(x[0] for x in scores[: self.ACTIVE_DOMAINS_TOP_K])
                else:
                    active_domain_set = set(list(sorted(candidate_5))[: self.ACTIVE_DOMAINS_TOP_K])
            else:
                active_domain_set = set(list(sorted(candidate_5))[: self.ACTIVE_DOMAINS_TOP_K])
        regex_str = DomainProcessor.build_neo4j_regex(active_domain_set)
        debug_1 = {
            "job_ids": job_ids,
            "job_previews": job_previews,
            "anchor_debug": anchor_debug,
            "dominance": dominance,
            "industrial_kws": industrial_kws,
            "anchor_skills": anchor_skills,
        }
        return active_domain_set, regex_str, anchor_skills, debug_1

    def _stage2_expand_academic_terms(self, anchor_skills, active_domain_set, regex_str, query_vector, query_text=None):
        """
        阶段 2：学术词扩展。边路 + 语境向量路 + 簇扩展 + 共鸣/共现，返回候选列表（不计算最终词权）。
        返回: raw_candidates，每项含 tid, term, degree_w, target_degree_w, domain_span, cov_j, hit_count 等，供 stage3 统一公式。
        """
        return self._expand_semantic_map(
            [int(k) for k in anchor_skills.keys()],
            anchor_skills,
            domain_regex=regex_str,
            query_vector=query_vector,
            query_text=query_text,
            return_raw=True,
        )

    def _stage3_word_weights(self, raw_candidates, query_vector):
        """
        阶段 3：词权重。统一走复杂公式（领域纯度、共鸣、语义守门等），无分支例外。
        返回: (score_map, term_map, idf_map)。
        """
        if not raw_candidates:
            return {}, {}, {}
        return self._calculate_final_weights(raw_candidates, query_vector)

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
            p_score, p_hits = self._compute_contribution(paper_struct, context)
            all_works_count += 1
            if p_score <= 0:
                continue
            paper_hit_terms[wid] = p_hits
            info["score"] = float(p_score)
            papers_for_agg.append(
                {
                    "wid": wid,
                    "score": float(p_score),
                    "authors": info["authors"],
                }
            )

        # 3) 统一调用 works_to_authors 进行“论文 → 作者”分摊与聚合（仅保留每位作者贡献最高的 Top3 论文）
        agg_result = accumulate_author_scores(papers_for_agg, top_k_per_author=3)
        author_scores = agg_result.author_scores
        author_top_works = agg_result.author_top_works  # aid -> [(wid, contrib_score), ...]

        # 4) 重建 per-author 调试与展示结构（代表作、标签统计等）
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
                        "contribution": round(contrib, 4),
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

            # 作者总论文数（未截断，用于轻量奖励“持续贡献”）
            paper_cnt_author = author_raw_paper_cnt.get(aid, len(per_author_papers))
            # Author-level bonus：log(1 + paper_count_author)，在 Top3 代表作总分上做轻量放大
            author_bonus = math.log1p(paper_cnt_author)
            final_score = total_score * author_bonus

            scored_authors.append({
                "aid": aid,
                "score": final_score,
                "raw_score": total_score,
                "top_paper": best_paper,
                "paper_count": paper_cnt_author,
                "top_papers": top_papers,
                "tag_stats": tag_stats,
            })

        scored_authors.sort(key=lambda x: x["score"], reverse=True)
        sorted_terms = sorted(
            [(term_map.get(tid, ""), score_map.get(tid, 0)) for tid in score_map],
            key=lambda x: x[1], reverse=True
        )
        academic_kws = list(set(term_map.values()))
        self.last_debug_info = {
            "active_domains": list(active_domain_set),
            "dominance": f"{dominance * 100:.1f}%",
            "expansion_pipeline": getattr(self, "_last_expansion_pipeline_stats", None),
            "similar_to_raw": getattr(self, "_last_similar_to_raw_rows", []),
            "similar_to_agg": getattr(self, "_last_similar_to_agg", []),
            "similar_to_pass": getattr(self, "_last_similar_to_pass", []),
            "regex_str": debug_1.get("regex_str", ""),
            "job_ids": debug_1.get("job_ids", []),
            "job_previews": debug_1.get("job_previews", []),
            "anchor_debug": debug_1.get("anchor_debug", {}),
            "anchor_melt_stats": getattr(self, "_last_anchor_melt_stats", None),
            "supplement_anchors": getattr(self, "_last_supplement_anchors", []),
            "industrial_kws": industrial_kws,
            "anchor_detail": [f"{k}={v['term']}" for k, v in anchor_skills.items()],
            "academic_kws": academic_kws,
            "detailed_kws": getattr(self, "_last_expansion_raw_results", []),
            "top_scored_terms": sorted_terms,
            "recall_vocab_count": len(score_map),
            "work_count": all_works_count,
            "author_count": len(scored_authors),
            "top_samples": scored_authors[:50],
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

        # 阶段 3：词权重（统一走复杂公式）
        score_map, term_map, idf_map = self._stage3_word_weights(raw_candidates, query_vector)

        # 阶段 4：图检索
        author_papers_list = self._stage4_graph_search(
            [int(k) for k in score_map.keys()], regex_str
        )

        # 阶段 5：作者打分与排序（debug_1 中补上 regex_str 供 last_debug_info）
        debug_1["regex_str"] = regex_str
        dominance = debug_1.get("dominance", 0.0)
        author_ids, _ = self._stage5_score_and_rank_authors(
            author_papers_list, score_map, term_map, active_domain_set, dominance, debug_1
        )

        elapsed_ms = (time.time() - start_t) * 1000
        return author_ids[: self.recall_limit], elapsed_ms


if __name__ == "__main__":
    l_path = LabelRecallPath(recall_limit=200)
    encoder = l_path._query_encoder

    try:
        domain_choice = input("\n请选择领域编号 (0跳过): ").strip() or "0"

        while True:
            user_input = input(f"\n请输入岗位需求 (q退出): ").strip()
            if not user_input or user_input.lower() == 'q':
                break

            query_vec, _ = encoder.encode(user_input)
            faiss.normalize_L2(query_vec)

            top_ids, search_time = l_path.recall(query_vec, domain_id=domain_choice, query_text=user_input)

            # --- 核心诊断日志 ---
            db = l_path.last_debug_info
            print("\n" + "🔍 [深度诊断流水线]" + "-" * 98)

            # 1. 领域探测（增强：打印命中的岗位 ID + Top20 岗位名称与描述片段）
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
                max_show = 200  # 想全量可调大；避免控制台刷屏导致难定位
                print(f"      锚点明细 (vid -> term) (展示 {min(len(anchor_detail), max_show)}/{len(anchor_detail)}):")
                for s in anchor_detail[:max_show]:
                    print(f"        - {s}")
            anchor_debug = db.get('anchor_debug', {})
            if anchor_debug:
                per_job = anchor_debug.get('per_job_skill_count', [])
                before_melt = anchor_debug.get('skills_before_melt', 0)
                after_melt = anchor_debug.get('skills_after_melt', 0)
                melted_sample = anchor_debug.get('melted_terms_sample', [])
                print(f"      参与锚点提取的岗位 (Top5) 每岗 REQUIRE_SKILL 数: {per_job}")
                print(f"      3% 熔断(统计用): 熔断前 {before_melt} 个，熔断后 {after_melt} 个；被熔断词样例: {melted_sample[:15]}")
            # 先熔断 -> Top30 -> Top20 实际执行统计
            melt_stats = db.get('anchor_melt_stats')
            if melt_stats:
                print(f"      【先熔断】实际: 熔断前={melt_stats.get('before_melt', 0)} 熔断后={melt_stats.get('after_melt', 0)} "
                      f"Top30后={melt_stats.get('after_top30', 0)} 阈值={melt_stats.get('melt_threshold', 0.03)}")
                melted_sample = melt_stats.get('melted_sample', [])
                if melted_sample:
                    print(f"      被熔断词及 cov_j (最多 20 条): {melted_sample[:20]}")
            # 核心词补充（JD 向量搜 vocabulary）
            supp = db.get('supplement_anchors', [])
            if supp:
                print(f"      【核心词补充】JD 向量搜 vocabulary 新增 {len(supp)} 个锚点 (vid, term, sim, cov_j):")
                for x in supp[:20]:
                    print(f"        - vid={x[0]} term={x[1]} sim={x[2]} cov_j={x[3]}")
                if len(supp) > 20:
                    print(f"        ... 共 {len(supp)} 条")
            else:
                print(f"      【核心词补充】本次未新增锚点（已存在或熔断过滤）")
            anchor_detail_final = db.get('anchor_detail', [])
            print(f"      合并后锚点总数: {len(anchor_detail_final)} (岗位熔断Top20 + 核心词补充)")

            # 3. 学术语义扩展与“学术共鸣”校验
            # 2.5 语义扩展管道（Step 3 前置）：看清在哪个环节被筛光
            expansion_pipeline = db.get('expansion_pipeline')
            regex_str = db.get('regex_str', '')
            if expansion_pipeline:
                p = expansion_pipeline
                print(f"【Step 3 前置: 语义扩展管道】")
                print(f"      目标领域 ID: {p.get('active_domains', [])}")
                print(f"      领域正则(regex): {regex_str}")
                print(f"      SIMILAR_TO 原始行数: {p.get('n_similar_to_rows', 0)}")
                print(f"      去重后候选学术词数: {p.get('n_unique_tids', 0)}")

                # 打印 SIMILAR_TO 原始扩展：按“源锚点(src_vid)”分组，展示每个锚点的 top 扩展
                similar_to_raw = db.get('similar_to_raw', []) or []
                if similar_to_raw:
                    src_map = collections.defaultdict(list)  # src_vid -> [(tid, term, sim)]
                    for r in similar_to_raw:
                        src = r.get('src_vid')
                        if src is None:
                            continue
                        src_map[int(src)].append(
                            (r.get('tid'), r.get('term'), float(r.get('sim_score', 0.0) or 0.0))
                        )

                    # 用锚点明细做 src_vid -> src_term 映射（形如 "167211=wbc"）
                    src_term_map = {}
                    for s in db.get('anchor_detail', []) or []:
                        if isinstance(s, str) and "=" in s:
                            k, v = s.split("=", 1)
                            if k.strip().isdigit():
                                src_term_map[int(k.strip())] = v.strip()

                    print("      SIMILAR_TO 原始扩展（按源锚点分组，每个锚点展示 Top3）:")
                    for src_vid in sorted(src_map.keys()):
                        items = sorted(src_map[src_vid], key=lambda x: x[2], reverse=True)[:3]
                        src_term = src_term_map.get(src_vid, "")
                        head = f"        - src_vid={src_vid}" + (f"({src_term})" if src_term else "")
                        print(head)
                        for tid, term, sc in items:
                            print(f"            -> tid={tid}  term={term}  sim={sc:.4f}")

                # 打印聚合后的候选（去重后，带 hit_count 与 src_vids）
                sim_agg = db.get('similar_to_agg', []) or []
                if sim_agg:
                    top_show = 20
                    sim_agg_sorted = sorted(
                        sim_agg,
                        key=lambda x: (int(x.get('hit_count', 0) or 0), float(x.get('sim_score', 0.0) or 0.0)),
                        reverse=True
                    )
                    print(f"      SIMILAR_TO 聚合后候选 Top{top_show}（按 hits、sim_score 排序）:")
                    for r in sim_agg_sorted[:top_show]:
                        sc = float(r.get('sim_score', 0.0) or 0.0)
                        print(f"        - tid={r.get('tid')}  term={r.get('term')}  sim={sc:.4f}  hits={r.get('hit_count')}  src_vids={r.get('src_vids')}")

                # 打印“通过 stats/领域过滤”的第一层学术词
                sim_pass = db.get('similar_to_pass', []) or []
                if sim_pass:
                    print("      通过过滤的第一层学术词（similar_to_pass）:")
                    for r in sorted(
                        sim_pass,
                        key=lambda x: (int(x.get('hit_count', 0) or 0), float(x.get('sim_score', 0.0) or 0.0)),
                        reverse=True
                    )[:30]:
                        deg = int(r.get('degree_w', 0) or 0)
                        tgt = int(r.get('target_degree_w', 0) or 0)
                        deg_exp = int(r.get('degree_w_expanded', 0) or 0)
                        # 正确口径：用展开后的单领域计数总和作分母（兼容 domain_dist 中存在复合 key）
                        ratio_correct = (float(tgt) / float(deg_exp)) if deg_exp else 0.0
                        sc = float(r.get('sim_score', 0.0) or 0.0)
                        print(f"        - tid={r.get('tid')}  term={r.get('term')}  sim={sc:.4f}  hits={r.get('hit_count')}  src_vids={r.get('src_vids')}  "
                              f"degree_w={deg}  degree_w_exp={deg_exp}  target_w={tgt}  ratio_correct={ratio_correct:.3f}")
                print(f"      【边路 SIMILAR_TO 领域过滤】")
                print(f"      去重后候选学术词数: {p.get('n_unique_tids', 0)}")
                print(f"      无 vocabulary_domain_stats 行: {p.get('n_no_stats', 0)}  (样例 tid: {p.get('sample_fail_no_stats', [])[:5]})")
                print(f"      work_count > P95(277) 被筛: {p.get('n_fail_degree_w', 0)}  (样例: {p.get('sample_fail_degree', [])[:5]})")
                print(f"      degree_w_expanded=0 被筛: {p.get('n_fail_degree_w_expanded_zero', 0)}  (归入下方纯度不足明细)")
                print(f"      目标领域 0 篇(target_degree_w<=0) 被筛: {p.get('n_fail_target_degree_w', 0)}  (样例: {p.get('sample_fail_target', [])[:5]})")
                print(f"      领域纯度 < 50% 被筛 (含 expanded=0): {p.get('n_fail_domain_ratio', 0)}  (样例: {p.get('sample_fail_ratio', [])[:5]})")
                print(f"      最终通过边路领域过滤: {p.get('n_final', 0)}")
                fail_details = p.get("fail_domain_ratio_details", [])
                if fail_details:
                    print(f"      被筛词纯度明细 (最多展示 20 条):")
                    print(f"      {'term':<20} | {'tid':<8} | {'deg_w':<8} | {'deg_w_exp':<10} | {'target_w':<10} | {'ratio':<8} | {'原因':<18} | 目标领域分布")
                    print(f"      {'-' * 20} | {'-' * 8} | {'-' * 8} | {'-' * 10} | {'-' * 10} | {'-' * 8} | {'-' * 18} | ---------")
                    for x in fail_details:
                        term_s = (x.get("term") or "")[:20]
                        dist_s = x.get("target_domains_dist") or {}
                        dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(dist_s.items()))
                        deg = int(x.get("degree_w", 0) or 0)
                        deg_exp = int(x.get("degree_w_expanded", 0) or 0)
                        tgt = int(x.get("target_degree_w", 0) or 0)
                        ratio_correct = (float(tgt) / float(deg_exp)) if deg_exp else 0.0
                        reason = x.get("fail_reason", "")
                        print(f"      {term_s:<20} | {x.get('tid', ''):<8} | {deg:<8} | {deg_exp:<10} | {tgt:<10} | {ratio_correct:<8.4f} | {reason:<18} | {dist_str}")
                        all_ratio = x.get("all_domains_ratio") or {}
                        if all_ratio:
                            ratio_str = ", ".join(
                                f"{DOMAIN_MAP.get(d, d)}({d}):{all_ratio[d]*100:.1f}%" for d in sorted(all_ratio.keys(), key=lambda d: -all_ratio[d])
                            )
                            print(f"        各领域占比: {ratio_str}")
                print(f"      最终通过进入 Step 3 的学术词数: {p.get('n_final', 0)}")
            else:
                print(f"【Step 3 前置: 语义扩展管道】无数据（未执行或未记录）")

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

            # 3.6 Tag Purity 诊断（raw>1 时可见统计口径异常）
            tag_purity_debug = getattr(l_path, "_last_tag_purity_debug", []) or []
            over_one = [x for x in tag_purity_debug if x.get("raw_tag_purity", 0) > 1.0]
            print(f"      【Tag Purity 诊断】共 {len(tag_purity_debug)} 个词参与权重计算，其中 raw_tag_purity>1 的共 {len(over_one)} 个")
            if over_one:
                print(f"      raw_tag_purity>1 明细 (最多 20 条，口径=target_degree_w/degree_w_expanded):")
                print(f"      {'term':<20} | {'tid':<6} | {'deg_w':<8} | {'deg_exp':<8} | {'target_w':<10} | {'raw':<8} | {'capped':<6}")
                print(f"      {'-' * 20} | {'-' * 6} | {'-' * 8} | {'-' * 8} | {'-' * 10} | {'-' * 8} | {'-' * 6}")
                for x in sorted(over_one, key=lambda t: -t.get("raw_tag_purity", 0))[:20]:
                    term_s = (x.get("term") or "")[:20]
                    print(f"      {term_s:<20} | {x.get('tid', ''):<6} | {x.get('degree_w', 0):<8} | {x.get('degree_w_expanded', 0):<8} | {x.get('target_degree_w', 0):<10} | {x.get('raw_tag_purity', 0):<8.4f} | {x.get('capped_tag_purity', 0):<6.4f}")
            elif tag_purity_debug:
                print(f"      口径 target_degree_w/degree_w_expanded，purity 自然≤1；raw 范围: [{min(x.get('raw_tag_purity', 0) for x in tag_purity_debug):.4f}, {max(x.get('raw_tag_purity', 0) for x in tag_purity_debug):.4f}]")

            # 4. 论文与作者召回
            w_count = db.get('work_count', 0)
            a_count = db.get('author_count', 0)
            vocab_count = db.get('recall_vocab_count', 0)
            if vocab_count:
                print(f"【Step 4: 召回规模】参与检索的学术词数: {vocab_count}，检索到 {w_count} 篇学术论文，最终锁定 {a_count} 名垂直领域专家。")
            else:
                print(f"【Step 4: 召回规模】检索到 {w_count} 篇学术论文，最终锁定 {a_count} 名垂直领域专家。")

            # --- 专家排名展示（增强：展示前 50 名，多篇代表作与标签统计）---
            print("-" * 115)
            print(f"{'排名':<6} | {'作者 ID':<12} | {'综合得分':<18} | {'学术领域代表作 (命中标签)'}")
            print("-" * 115)

            for i, item in enumerate(db.get('top_samples', []), 1):
                raw_score = item.get('score', 0)
                score_str = f"{raw_score:.4f}" if isinstance(raw_score, (int, float)) else str(raw_score)
                paper_count = item.get('paper_count', 0)
                aid = item.get('aid', '')
                # 主行：作者整体信息
                print(f"#{i:<5} | {aid:<12} | {score_str:<18} | 论文数: {paper_count}")

                # 代表作（单篇贡献度最高的一篇）
                tp = item.get('top_paper', {}) or {}
                if tp:
                    title = (tp.get('title') or 'Unknown')[:55]
                    hit_tags = ", ".join(tp.get('hits', []))
                    contrib = tp.get('contribution', 0)
                    print(f"{' ':23} ┗━ 代表作: 《{title}》")
                    print(f"{' ':23}     年份={tp.get('year', 'Unknown')}, 贡献={contrib:.6f}, 命中标签: {hit_tags}")

                # 多篇代表作：按贡献度排序的前 3 篇
                top_papers = item.get('top_papers') or []
                if top_papers:
                    print(f"{' ':23}     多篇代表作 Top{len(top_papers)}:")
                    for p in top_papers:
                        p_title = (p.get('title') or 'Unknown')[:55]
                        p_hits = ", ".join(p.get('hits', []))
                        p_contrib = p.get('contribution', 0.0)
                        print(f"{' ':23}       ● [{p.get('year', 'Unknown')}] 《{p_title}》 | 贡献={p_contrib:.6f} | 命中: {p_hits}")

                # 标签汇总统计：Top10 标签及命中次数
                tag_stats = item.get('tag_stats') or []
                if tag_stats:
                    tags_str = ", ".join(f"{t.get('term', '')}({t.get('count', 0)})" for t in tag_stats)
                    print(f"{' ':23}     核心标签Top{len(tag_stats)}: {tags_str}")

                # 只展示前 50 名，避免控制台输出过长
                if i >= 50:
                    break

            print("-" * 115)
            print(f"[*] 诊断完成。全链路耗时: {search_time:.2f}ms")

    except Exception as e:
        print(f"运行出错: {e}")
        traceback.print_exc()