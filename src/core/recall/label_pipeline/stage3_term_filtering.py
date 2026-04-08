import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.recall.label_means import term_scoring
from src.core.recall.label_means.label_debug import debug_print
from src.core.recall.label_means.hierarchy_guard import (
    build_family_key,
    get_retrieval_role_from_term_role,
    should_drop_term,
    score_term_record,
)

# 软排序 + 轻过滤：至少保留 top_k，不再用阈值淘汰为 0
STAGE3_TOP_K = 20
STAGE3_DETAIL_DEBUG = False  # 默认关闭重型逐词细表（可按需打开）
# Stage3 role 因子：mainline primary 稳一点，dense/conditioned_vec/side 略降，避免 Motion controller > motion control
STAGE3_MAINLINE_BOOST = 1.06
# primary_expandable → stage3_bucket=core：原仅 identity≥0.95；加一条「非 conditioned_only + JD 对齐」结构化升舱（如 robot control）
STAGE3_EXPANDABLE_CORE_IDENTITY_HARD = 0.95
STAGE3_EXPANDABLE_CORE_JD_ALIGN_MIN = 0.50
STAGE3_DENSE_PENALTY = 0.90
STAGE3_CONDITIONED_PENALTY = 0.94
STAGE3_SIDE_PENALTY = 0.95
# 标签路追踪：是否打印 source_type / anchor / similar_to 原始候选 / 被过滤原因 / final primary 胜出原因
LABEL_PATH_TRACE = False
# 单锚 + mainline_hits=1 + can_expand 的旁枝：从 support 降级 risky（结构信号，非词表）
STAGE3_SUPPORT_DEMOTE_FC_MAX = 0.38
STAGE3_SUPPORT_DEMOTE_PTC_MAX = 0.52
STAGE3_SUPPORT_DEMOTE_OBJ_MIN = 0.38
STAGE3_SUPPORT_DEMOTE_DRIFT_MIN = 0.62
STAGE3_SUPPORT_DEMOTE_DRIFT_PTC_MAX = 0.48
STAGE3_SUPPORT_DEMOTE_GENERIC_MIN = 0.52
STAGE3_SUPPORT_DEMOTE_POLY_MIN = 0.58
STAGE3_SUPPORT_DEMOTE_POLY_PTC_MAX = 0.50
# family 保送式 paper 选词：每 family 最多 1 primary + 1 support，cluster 默认不进 paper recall
PAPER_RECALL_MAX_TERMS = 12
# Paper 入稿：统一分数相对截断（相对 Top1 与绝对下限取 max）；避免弱支线索性跟跑
PAPER_RECALL_DYNAMIC_FLOOR_REL = 0.62
PAPER_RECALL_DYNAMIC_FLOOR_ABS = 0.12
# --- 主 cutoff 之后的「尾部补位」（不改 gate / unified / paper_select_score 公式；最多再塞 1~2 词）---
# 触发：已选词很少（≤3）或 **support 槽无人**（selected_support==0）；near-miss 带来 heterogeneity，避免 term set 过窄。
PAPER_RECALL_TAIL_EXPAND_ENABLED = True
PAPER_RECALL_TAIL_EXPAND_MAX_EXTRA = 2
PAPER_RECALL_TAIL_EXPAND_DELTA_MAX = 0.045
PAPER_RECALL_TAIL_EXPAND_REQUIRE_CORE_OR_SEED = True
# 主 cutoff 后「结构纠偏」：历史 paper lane swap（已弃用；Stage3 仅做 Stage4 输入准备）
PAPER_RECALL_CORE_NEAR_MISS_SWAP_MAX_BELOW_FLOOR = 0.02
PAPER_RECALL_SUPPORT_SWAP_SCORE_MARGIN = 0.03
# --- Stage4 输入准备（paper_terms）：risky 极少量例外、support 最低 GC 子分（非 validation）---
STAGE4_PREP_RISKY_MAX = 1
STAGE4_PREP_RISKY_MIN_FINAL = 0.44
STAGE4_PREP_RISKY_MAX_RISK_DIM = 0.58
STAGE4_PREP_RISKY_MIN_LF_BB_SUM = 0.88
STAGE4_PREP_SUPPORT_MIN_LF = 0.30
STAGE4_PREP_SUPPORT_MIN_BB = 0.26
STAGE4_PREP_SUPPORT_MAX_RISK_DIM = 0.78
STAGE4_PREP_CORE_FLOOR_RELAX = 0.10  # core 相对 support 略放宽 dynamic_floor（仍看 paper_ready）
# Stage4 输入准备：互补语境（小 λ，避免 coverage 压过 paper_ready / 结构门）
STAGE4_PREP_COVERAGE_LAMBDA = 0.10
STAGE4_PREP_COVERAGE_PROMOTION_THRESH = 0.36  # gain 高于此计为 coverage_based_promotions（观测用）
# --- Stage4 prep：weak support contamination → penalized-admit（进 coverage 池，乘法压 paper_ready；非旧 soft-admit）---
STAGE4_PREP_SUPPORT_CONTAMINATION_ALLOW_COMPETE = True
STAGE4_PREP_SUPPORT_CONTAMINATION_PENALTY = 0.50  # 对 paper_ready 的乘法因子（<1，penalized-admit；靠 coverage 翻盘）
STAGE4_PREP_SUPPORT_CONTAMINATION_READY_MULT = STAGE4_PREP_SUPPORT_CONTAMINATION_PENALTY  # 别名，兼容旧引用
# --- Stage4 prep：默认 risky 外极少「结构可竞争」进池（coverage 前仍过滤；名额极小）---
STAGE4_PREP_RISKY_ALLOW_COMPETE = True
STAGE4_PREP_RISKY_COMPETE_MAX = 1  # 每个 JD 至多允许多少条 default-risky 进入 pool_risky（与 STAGE4_PREP_RISKY_MAX 选中配额独立）
STAGE4_PREP_RISKY_COMPETE_READY_MULT = 0.68
STAGE4_PREP_RISKY_COMPETE_MIN_FINAL = 0.36
STAGE4_PREP_RISKY_COMPETE_MAX_RISK_PEN = 0.64  # risk_penalty 子分上限（与 exception 的 STAGE4_PREP_RISKY_MAX_RISK_DIM 区分）
STAGE4_PREP_RISKY_COMPETE_MIN_BACKBONE = 0.22
STAGE4_PREP_RISKY_COMPETE_MIN_LOCAL_FIT = 0.26
STAGE4_PREP_RISKY_COMPETE_REQUIRE_META = True  # 需至少一处 merge/provenance/anchor_support 元信息（通用结构，非词表）
# 对 (1 - max_redundancy) 做轻度指数加压：压低近邻重复的有效 gain，抬高互补项相对优势
STAGE4_PREP_COVERAGE_GAIN_SHARPEN = 1.18
# 通用 redundancy 权重（v2：family / parent / topic / token 更敏感，仍 clip 到 1）
STAGE4_PREP_REDUND_W_FAMILY_EXACT = 0.58
STAGE4_PREP_REDUND_W_FAMILY_PREFIX = 0.30
STAGE4_PREP_REDUND_W_PP_EXACT = 0.34
STAGE4_PREP_REDUND_W_PP_SUB = 0.20
STAGE4_PREP_REDUND_W_PA_EXACT = 0.30
STAGE4_PREP_REDUND_W_PA_SUB = 0.18
STAGE4_PREP_REDUND_W_TOPIC_EXACT = 0.18
STAGE4_PREP_REDUND_W_TOPIC_SUB = 0.10
STAGE4_PREP_REDUND_W_TOKEN_JACCARD = 0.30
# paper 选词 lane-tier 扫描序：强主轴 core → support_lane（含近失配）→ other_eligible → bonus-like primary core 殿后（不靠词表）
PAPER_SELECT_STRONG_MAIN_AXIS_SCORE_MIN = 0.95
PAPER_SELECT_STRONG_MAIN_AXIS_MAX_RANK = 6
PAPER_SELECT_STRONG_MAIN_AXIS_ML_EXP_SCORE_MIN = 0.85
PAPER_SELECT_STRONG_MAIN_AXIS_ML_EXP_MAX_RANK = 8
# paper_readiness · 锚点先验（Stage3）：以 Step2 锚点分为「主证据」，名次只做轻平滑（避免 rank 主导把 robot control 等主线词打出 dynamic_floor）
PAPER_READINESS_ANCHOR_SCORE_NORM = 1.20  # anchor_score_norm = clip(score / 此值, 0, 1)
PAPER_READINESS_ANCHOR_PRIOR_W_SCORE = 0.88  # 先验里分值权重（主）
PAPER_READINESS_ANCHOR_PRIOR_W_RANK = 0.12  # 先验里名次平滑权重（辅）
PAPER_READINESS_ANCHOR_RANK_SMOOTH_LAMBDA = 0.04  # rank_smooth = 1/(1+λ·max(rank-1,0))
PAPER_READINESS_ANCHOR_READINESS_BLEND_LO = 0.82  # readiness *= LO + HI * anchor_prior
PAPER_READINESS_ANCHOR_READINESS_BLEND_HI = 0.18
# core + 已命中主线 + 可扩：paper_readiness 保底（结构信号，非词表）；防「主线可扩却被 floor 误杀」
PAPER_READINESS_CORE_MAINLINE_EXPAND_MIN = 0.78
# primary_support_seed + 主线 + 可扩 + 非 conditioned_only：较 core 略低的 readiness 底板，与 PAPER_SUPPORT_SEED_FACTOR 叠加后仍易竞争 paper
PAPER_READINESS_SUPPORT_SEED_MAINLINE_EXPAND_MIN = 0.76
# paper_select_score：在 final_score×paper_readiness 底分上 **加性轻注入**（仅 eligible 排序 + dynamic_floor；**不改 final_score**）
# 直接吃 Stage2A `primary_bucket`、主线命中、可扩、`parent_anchor_*`，抬高 JD 主轴词、压低弱桶/无主线 — 无量词黑名单。
PAPER_SELECT_SCORE_ANCHOR_NORM = 1.20
PAPER_SELECT_BUCKET_BONUS_EXPANDABLE = 0.055
PAPER_SELECT_BUCKET_BONUS_SUPPORT_SEED = 0.020
PAPER_SELECT_BUCKET_PENALTY_SUPPORT_KEEP = -0.010
PAPER_SELECT_BUCKET_PENALTY_RISKY_KEEP = -0.030
PAPER_SELECT_BONUS_MAINLINE_HIT = 0.020
PAPER_SELECT_PENALTY_MAINLINE_NONE = -0.015
PAPER_SELECT_BONUS_CAN_EXPAND = 0.020
PAPER_SELECT_PENALTY_NO_EXPAND = -0.010
PAPER_SELECT_RANK_BONUS_LE3 = 0.035
PAPER_SELECT_RANK_BONUS_LE6 = 0.020
PAPER_SELECT_RANK_BONUS_LE10 = 0.008
PAPER_SELECT_ANCHOR_SCORE_BONUS_COEF = 0.015  # × anchor_score_norm；过强可改 0.05（README 备选）
PAPER_SELECT_ROLE_BONUS_PRIMARY = 0.015
PAPER_SELECT_ROLE_PENALTY_PRIMARY_SIDE = -0.010
PAPER_SELECT_ROLE_PENALTY_DENSE_EXP = -0.020
# paper_select_score 结构 bonus：**仅缩放加性 bonus 的正部**；负部（bucket/risky 惩罚等）全额保留，避免误抬弱桶。
# 分档与 `paper_select_lane_tier` / `_is_strong_main_axis_core` / `_is_bonus_like_core` 一致（算分时尚未写 tier 字段亦可调用）。
PAPER_SELECT_BONUS_TIER_SCALE_STRONG_MAIN_AXIS = 1.0
PAPER_SELECT_BONUS_TIER_SCALE_BONUS_CORE = 0.52  # 未达强主轴门槛的 primary core（如 RL）：压低 +bonus
PAPER_SELECT_BONUS_TIER_SCALE_SUPPORT_LANE = 0.90  # support / near_miss
PAPER_SELECT_BONUS_TIER_SCALE_OTHER_ELIGIBLE = 0.80
# --- bonus_core paper-ready（仅 select_terms_for_paper_recall）：对 eligible 的 p_sel 再乘一层结构乘子；禁止 swap/tail 的 core-near-miss 抢救；无词表 ---
PAPER_BONUS_CORE_READINESS_RANK_THRESHOLD = 8  # rk>此值 → 乘 READINESS_RANK_MULT
PAPER_BONUS_CORE_READINESS_SCORE_THRESHOLD = 0.90  # a_sc<此值 → 乘 READINESS_SCORE_MULT
PAPER_BONUS_CORE_READINESS_RANK_MULT = 0.82
PAPER_BONUS_CORE_READINESS_SCORE_MULT = 0.82
PAPER_BONUS_CORE_READINESS_ML0_MULT = 0.85
PAPER_BONUS_CORE_READINESS_NO_EXPAND_MULT = 0.88
PAPER_BONUS_CORE_READINESS_FALLBACK_PRIMARY_MULT = 0.80
PAPER_BONUS_CORE_READINESS_CONDITIONED_ONLY_MULT = 0.88
# Paper 选词：候选池规模惩罚（log10、系数上限）。小 final N 时缓释「大池词抬高 paper_select_score 与 dynamic_floor」；
# 信号来自 `papers_before_filter` 或回落 `degree_w`（Stage3 词汇论文规模），**非**词表黑名单。
PAPER_SELECT_POOL_PENALTY_ENABLED = True
PAPER_SELECT_POOL_PENALTY_LOG_COEF = 0.015
PAPER_SELECT_POOL_PENALTY_MAX = 0.08
# Paper bridge：weak support 结构门（非词表拦词）。统一分只负责排序；是否进 paper 由本组常量做最后结构裁决。
STAGE3_PAPER_SUPPORT_BLOCK_ENABLED = True
STAGE3_PAPER_SUPPORT_MAX_ANCHOR_COUNT = 1
STAGE3_PAPER_SUPPORT_MAX_MAINLINE_HITS = 1
# 与 _paper_grounding_score 同尺度：低于则表示 grounding proxy 偏软，易混进「伪 support」
STAGE3_PAPER_SUPPORT_MIN_GROUNDING = 0.62
# 与 MIN_GROUNDING 联用：分不够高且 grounding 不够硬时，不能仅靠统一分混进 paper
STAGE3_PAPER_SUPPORT_MIN_FINAL_SCORE = 0.30
# conditioned-only 的 support 默认更严（易漂移）
STAGE3_PAPER_SUPPORT_BLOCK_COND_ONLY = True
# ---- support → paper：在满足 weak_support_contamination 时，**小配额软放行**（折扣 readiness），扩 eligible / final N；非词表。----
SUPPORT_TO_PAPER_ENABLED = True
SUPPORT_TO_PAPER_MAX = 3
SUPPORT_TO_PAPER_MAX_FRAC = 1.0 / 3.0  # 入选 support 同时受「最多 SUPPORT_TO_PAPER_MAX」与「不超过 max_terms 的 1/3」约束
PAPER_SUPPORT_SEED_FACTOR = 0.82  # support_seed_soft_admit：乘 readiness（略抬高，避免 0.72 与连乘双压过狠）
PAPER_SUPPORT_KEEP_FACTOR = 0.58  # primary_support_keep 且结构信号达标时软放行
# core 未过 JD 全局主轴门（rk∨a_sc）：**不**再硬拒；降级 support_pool，乘 readiness 后进 `paper_select_score` 竞争（配额仍走 support 槽）
PAPER_CORE_AXIS_NEAR_MISS_FACTOR = 0.72
# 未触 risky_side_block 的 risky：保守进 support_pool，略压 readiness，避免与主轴 core 同权
PAPER_RISKY_SUPPORT_LANE_FACTOR = 0.88
# 其它少见桶：与「非主轴」同哲学，轻压后进池
PAPER_FALLBACK_SUPPORT_LANE_FACTOR = 0.90
# Paper：risky + 无主线命中 + 不可扩 + side/expansion 角色 — 仅靠 unified 分过线仍会混入（如 motion controller）
STAGE3_PAPER_RISKY_SIDE_BLOCK_ENABLED = True
STAGE3_PAPER_RISKY_SIDE_TERM_ROLES = frozenset(
    {"primary_side", "dense_expansion", "cluster_expansion", "cooc_expansion"}
)
# Paper：JD 全局主轴门（非词表）。仅 stage3_bucket=core：**父锚 rank 与 a_sc 二选一达线即过**（OR），
# 避免 AND 把「分略低但名次尚可」或「名次靠后但分仍高」一并双杀；两侧皆弱仍拒（与 AND 相同）。
# 调参：单调改 MAX_RANK / MIN_SCORE；若 rk 与 a_sc 双不达线仍落选，需再放宽常数或引入其它 Stage3 结构信号。
STAGE3_PAPER_MAIN_AXIS_GATE_ENABLED = True
STAGE3_PAPER_MAIN_AXIS_MAX_RANK = 10
STAGE3_PAPER_MAIN_AXIS_MIN_SCORE = 0.90
# Stage3 第二段 — 统一连续分权重（无 bucket / primary_bucket 条件乘子；调参只改标量）
STAGE3_UNIFIED_W_SEED = 0.34
STAGE3_UNIFIED_W_ANCHOR_ID = 0.18
STAGE3_UNIFIED_W_JD_ALIGN = 0.16
STAGE3_UNIFIED_W_FAMILY_CENTRALITY = 0.10
STAGE3_UNIFIED_W_CROSS_ANCHOR = 0.10
STAGE3_UNIFIED_W_MAINLINE = 0.07
STAGE3_UNIFIED_W_EXPAND = 0.05
STAGE3_UNIFIED_W_DRIFT = 0.12
STAGE3_UNIFIED_W_GENERIC = 0.08
STAGE3_UNIFIED_W_POLY = 0.08
STAGE3_UNIFIED_W_OBJECT = 0.06
# Global Coherence Rerank：四分项主路径 + 小幅 legacy unified blend（TODO: 纯 GC 后将 STAGE3_GC_LEGACY_BLEND→0）
STAGE3_GC_LEGACY_BLEND = 0.12
STAGE3_GC_W_LOCAL_FIT = 0.38
STAGE3_GC_W_CROSS = 0.34
STAGE3_GC_W_BACKBONE = 0.28
STAGE3_GC_W_RISK = 0.42
# 与 select_terms_for_paper_recall 对齐的入稿闸门说明（逐项 term）
STAGE3_PAPER_GATE_DEBUG = False  # 默认降噪；需要逐项 gate 时再开
# Stage4 输入准备观测：全量池 / pre-coverage 挡板 / pairwise redundancy / 语境轴追踪（不改变选词结果）
STAGE3_PAPER_POOL_FULL_AUDIT = False
STAGE3_PRE_COVERAGE_BLOCK_AUDIT = False
STAGE3_CONTEXT_COVERAGE_PAIRWISE_AUDIT = False
STAGE3_CONTEXT_AXIS_TRACE_AUDIT = False
# 若为 True：STAGE3_AUDIT_DEBUG=True 时上述四项观测一并开启（单项 True 时无论 AUDIT 如何都打）
STAGE3_PAPER_OBS_WITH_AUDIT_DEBUG = False
# Stage4 prep：penalized-admit / risky compete 轻量审计（可与 STAGE3_PAPER_GATE_DEBUG 叠加）
STAGE3_PREP_PENALTY_COMPETE_AUDIT = False
STAGE3_PAPER_CUTOFF_AUDIT = True  # [Stage3 paper cutoff] 排名与截断原因
# 主 cutoff、swap 前：四行 KPI（strong/support 实取、bonus 实取、仍≥floor 的 support 余量）。用于一眼判断殿后序是否生效：
# 若 bonus_core_taken>0 且 support_lane_remaining_candidates>0，主扫描下属异常信号（应排查）；reason 见 bonus-core 审计。
STAGE3_PAPER_LANE_FILL_AUDIT = True
# bonus_core（宽 primary）逐条：想进与否、被谁占坑、挡因（含 support_lane_priority_not_filled）。
STAGE3_BONUS_CORE_BLOCKED_AUDIT = True
# [Stage3 bonus-core readiness audit]：bonus-like core 的结构 ready、p_sel 缩放前后、swap/tail near-miss 通道资格（与 STAGE3_PAPER_CUTOFF_AUDIT 联动）
STAGE3_BONUS_CORE_READINESS_AUDIT = True
# 与 [Stage3 paper centrality audit] 重叠度高；默认关，需拆 p_sel_base/bonus 时再开
STAGE3_ELIGIBLE_CORE_CLOSE_CALL_AUDIT = False
STAGE3_SUPPORT_CONTAMINATION_AUDIT = True  # Stage3→4 之间：可疑 support 摘要
STAGE3_UNIFIED_SCORE_DEBUG = True  # 连续分特征拆解（替代原「规则乘子」视角）
# 默认只打 TopN（paper 瓶颈在看 eligible / support 软放行；unified 仅辅）
STAGE3_UNIFIED_SCORE_DEBUG_TOP_K = 3
# Stage3 窄表审计：final adjust / cross-anchor / support·risky / paper bridge（与 bucket reason 互补）
STAGE3_AUDIT_DEBUG = True
# Guardrail：仅底线硬拒 vs 软标记（与 GC 主分分离）
STAGE3_GUARDRAIL_REJECT_AUDIT = True
STAGE3_GUARDRAIL_SOFT_AUDIT = True
STAGE3_GUARDRAIL_SOFT_AUDIT_MAX = 18
# Stage3 score_mult 单词明细 / 重点词 watch / rerank TopN（仅 stage3_build_score_map）
DEBUG_LABEL_PATH = True
STAGE3_BUCKET_FACTOR_DEBUG = False   # [Stage3 bucket reason] / [Stage3 dominant factors]
STAGE3_OUTPUT_BREAKDOWN_DEBUG = False  # Final Score Breakdown / Bucket Details / Risky Reasons / final_score 明细
STAGE3_OBSERVABILITY_PANEL_DEBUG = False  # 观测面板 Stage3 / 汇总
# 非空时：下列审计只打印 term 字面完全匹配（strip 后）的行，避免刷屏；置空 set() 则按 top_k 全表
STAGE3_DEBUG_FOCUS_TERMS: Set[str] = {
    "Motion control",
    "motion control",
    "robot control",
    "Robot control",
    "digital control",
    "Digital control",
    "Robot manipulator",
    "Educational robotics",
    "route planning",
    "Route planning",
}

# 全局共识因子默认值（无硬编码）；Stage3 已彻底移除 cluster 依赖
STAGE3_CROSS_ANCHOR_DEFAULT = 1.0
# tid 合并：多锚时 primary_bucket 取「最强证据」而非「单条最高 seed 分记录」整表覆盖，避免 anchor_count 与语义字段撕裂
STAGE3_PRIMARY_BUCKET_MERGE_PRIORITY: Dict[str, int] = {
    "primary_expandable": 5,
    "primary_support_seed": 4,
    "primary_support_keep": 3,
    "primary_keep_no_expand": 3,
    "risky_keep": 2,
    "primary_fallback_keep_no_expand": 1,
    "reject": 0,
    "": 0,
}
STAGE3_TERM_ROLE_MERGE_PRIORITY: Dict[str, int] = {
    "primary": 5,
    "primary_side": 3,
    "dense_expansion": 2,
    "cluster_expansion": 2,
    "cooc_expansion": 2,
    "": 0,
}
# 多锚 tid 合并窄表：回答「是 Stage3 真判 risky 还是 merge 把 motion control 带歪」
STAGE3_DUPLICATE_MERGE_AUDIT = True
# primary_expandable 却未成 core：精调 bucket 时对照 identity / jd / cap
STAGE3_CORE_MISS_AUDIT = True

_STAGE3_MERGE_SKIP_KEYS = frozenset(
    {
        "records",
        "parent_anchors",
        "parent_primaries",
        "source_types",
        "term_roles",
        "family_keys",
    }
)
_STAGE3_MERGE_STRUCTURE_FIELDS = frozenset(
    {
        "primary_bucket",
        "primary_reason",
        "mainline_candidate",
        "can_expand_local",
        "fallback_primary",
        "fallback_primary_all",
        "role_in_anchor",
        "term_role",
        "can_expand",
        "can_expand_from_2a",
    }
)
# Representative 行仅填充「非语义 / 非聚合」字段；聚合语义见 source_evidence_list 与 *_summary
_STAGE3_MERGE_NEVER_FROM_REPRESENTATIVE = (
    _STAGE3_MERGE_SKIP_KEYS
    | _STAGE3_MERGE_STRUCTURE_FIELDS
    | frozenset(
        {
            "stage2_local_meta",
            "source_evidence_list",
            "stage2_local_meta_list",
            "anchor_support_summary",
            "provenance_summary",
            "merge_debug_summary",
            "primary_bucket_summary",
            "role_in_anchor_set",
            "anchor_role_distribution",
            "term_role_distribution",
        }
    )
)


def _stage3_normalized_source_types(rec: Dict[str, Any]) -> Set[str]:
    """合并 source_types 与 source_type / source 单值，供 conditioned_only 与 paper 门统一使用。"""
    source_types = rec.get("source_types") or set()
    if isinstance(source_types, (list, tuple)):
        source_types = set(source_types)
    else:
        source_types = set(source_types)
    st = (rec.get("source_type") or rec.get("source") or rec.get("origin") or "").strip().lower()
    if st:
        source_types.add(st)
    return source_types


def _stage3_is_conditioned_only(rec: Dict[str, Any]) -> bool:
    """
    conditioned_only（严格）：仅有 conditioned_vec，且无 similar_to、无 family_landing。
    与分桶 / bucket_reason_flags / paper 硬门同源。
    """
    st = _stage3_normalized_source_types(rec)
    has_similar = "similar_to" in st
    has_family = "family_landing" in st
    has_conditioned = "conditioned_vec" in st
    return bool(has_conditioned and (not has_similar) and (not has_family))


def _stage3_clamp01(x: Any) -> float:
    try:
        v = float(x or 0.0)
    except (TypeError, ValueError):
        v = 0.0
    return max(0.0, min(1.0, v))


def _stage3_cross_anchor_strength(rec: Dict[str, Any]) -> float:
    """把跨锚信号压到 [0,1]：优先 cross_anchor_evidence（约 0.9~1.1），否则退回 explain.cross_anchor_factor。"""
    ex = rec.get("stage3_explain") or {}
    ca_raw = rec.get("cross_anchor_evidence")
    if ca_raw is None:
        ca_raw = ex.get("cross_anchor_factor")
    try:
        ca = float(ca_raw)
    except (TypeError, ValueError):
        ca = 1.0
    if 0.89 <= ca <= 1.11:
        return _stage3_clamp01((ca - 0.9) / 0.2)
    return _stage3_clamp01(ca)


def _compute_stage3_unified_continuous_score(rec: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Stage3 第二段：唯一主序分（连续、无 bucket 分支）。
    结构对齐 README 说明：正项为证据/主线/可扩，负项为漂移与泛化风险。
    """
    seed = _stage3_clamp01(rec.get("best_seed_score") or rec.get("score") or 0.0)
    anchor_id = _stage3_clamp01(rec.get("best_anchor_identity") or 0.0)
    jd_al = _stage3_clamp01(rec.get("best_jd_align") or 0.0)
    fc = _stage3_clamp01(rec.get("family_centrality") or 0.0)
    cross_s = _stage3_cross_anchor_strength(rec)
    mh = int(rec.get("mainline_hits") or 0)
    can_ex = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
    mainline_strength = _stage3_clamp01(min(1.0, mh / 2.0 + (0.35 if can_ex else 0.0)))
    expandability = 1.0 if can_ex else 0.0
    drift = _stage3_clamp01(rec.get("semantic_drift_risk"))
    gen = _stage3_clamp01(rec.get("generic_risk"))
    poly = _stage3_clamp01(rec.get("polysemy_risk"))
    obj = _stage3_clamp01(rec.get("object_like_risk"))

    pos = (
        STAGE3_UNIFIED_W_SEED * seed
        + STAGE3_UNIFIED_W_ANCHOR_ID * anchor_id
        + STAGE3_UNIFIED_W_JD_ALIGN * jd_al
        + STAGE3_UNIFIED_W_FAMILY_CENTRALITY * fc
        + STAGE3_UNIFIED_W_CROSS_ANCHOR * cross_s
        + STAGE3_UNIFIED_W_MAINLINE * mainline_strength
        + STAGE3_UNIFIED_W_EXPAND * expandability
    )
    neg = (
        STAGE3_UNIFIED_W_DRIFT * drift
        + STAGE3_UNIFIED_W_GENERIC * gen
        + STAGE3_UNIFIED_W_POLY * poly
        + STAGE3_UNIFIED_W_OBJECT * obj
    )
    raw = pos - neg
    unified = max(0.0, min(1.0, raw))
    breakdown = {
        "seed_score": seed,
        "anchor_identity": anchor_id,
        "jd_align": jd_al,
        "family_centrality": fc,
        "cross_anchor_strength": cross_s,
        "mainline_strength": mainline_strength,
        "expandability": expandability,
        "semantic_drift_risk": drift,
        "generic_risk": gen,
        "polysemy_risk": poly,
        "object_like_risk": obj,
        "unified_pos": pos,
        "unified_neg": neg,
        "unified_raw": raw,
    }
    return unified, breakdown


def _stage3_get_candidate_graph_from_recall(recall: Any) -> Dict[str, Any]:
    ctx = getattr(recall, "_stage3_stage2_context", None) if recall is not None else None
    if not isinstance(ctx, dict):
        return {}
    g = ctx.get("candidate_graph")
    return g if isinstance(g, dict) else {}


def _stage3_graph_cross_anchor_stats(tid: Any, graph: Dict[str, Any]) -> Tuple[int, float]:
    """当前 tid 在 cross_anchor_support_edges 上的度数与 score_hint 均值（candidate_graph 消费入口）。"""
    edges = graph.get("cross_anchor_support_edges") if graph else None
    if not isinstance(edges, list) or tid is None:
        return 0, 0.0
    try:
        t_int = int(tid)
    except (TypeError, ValueError):
        return 0, 0.0
    hints: List[float] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        try:
            st = int(e.get("src_tid"))
            dt = int(e.get("dst_tid"))
        except (TypeError, ValueError):
            continue
        if st == t_int or dt == t_int:
            try:
                hints.append(float(e.get("score_hint") or 0.0))
            except (TypeError, ValueError):
                hints.append(0.0)
    if not hints:
        return 0, 0.0
    return len(hints), float(sum(hints) / len(hints))


def _compute_stage3_local_fit(rec: Dict[str, Any]) -> Dict[str, Any]:
    """局部贴合：消费 source_evidence_list / explain / 弱 bucket 先验（不支配）。"""
    notes: List[str] = []
    ex = rec.get("stage3_explain") or {}
    evid = list(rec.get("source_evidence_list") or [])
    identities: List[float] = []
    for e in evid:
        if not isinstance(e, dict):
            continue
        identities.append(float(e.get("identity_score") or e.get("sim_score") or 0.0))
    if not identities:
        identities = [float(rec.get("best_identity_score") or rec.get("identity_score") or 0.0)]
    mean_id = sum(identities) / len(identities)
    max_id = max(identities)
    seed = _stage3_clamp01(rec.get("best_seed_score") or rec.get("score"))
    jd = _stage3_clamp01(rec.get("best_jd_align"))
    ptc = _stage3_clamp01(ex.get("path_topic_consistency"))
    qual = _stage3_clamp01(rec.get("quality_score"))
    fc = _stage3_clamp01(rec.get("family_centrality") or ex.get("family_centrality"))
    pbs = rec.get("primary_bucket_summary") or {}
    derived = (pbs.get("derived_primary_bucket") or "").strip().lower()
    pri = STAGE3_PRIMARY_BUCKET_MERGE_PRIORITY.get(derived, 0)
    bucket_prior = 0.04 * (pri / 5.0)
    core_fit = (
        0.20 * _stage3_clamp01(mean_id)
        + 0.16 * _stage3_clamp01(max_id)
        + 0.14 * seed
        + 0.16 * jd
        + 0.16 * ptc
        + 0.10 * qual
        + 0.08 * fc
    )
    score = _stage3_clamp01(core_fit + bucket_prior)
    if len(evid) >= 2:
        score = _stage3_clamp01(score + 0.02)
        notes.append("multi_source_evidence:+0.02")
    notes.append("primary_bucket_prior_weak:max+0.04")
    return {
        "score": score,
        "components": {
            "evidence_rows": len(evid),
            "identity_mean": mean_id,
            "identity_max": max_id,
            "seed_norm": seed,
            "jd_align": jd,
            "path_topic_consistency": ptc,
            "quality_score": qual,
            "family_centrality": fc,
            "derived_primary_bucket": derived,
            "bucket_prior_applied": bucket_prior,
        },
        "notes": notes,
    }


def _compute_stage3_cross_anchor_coherence(rec: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    """跨锚一致性：anchor_support_summary + merge_debug + candidate_graph cross_anchor_support_edges。"""
    notes: List[str] = []
    adb = rec.get("anchor_support_summary") or {}
    mdbg = rec.get("merge_debug_summary") or {}
    ac = int(adb.get("anchor_count") or rec.get("anchor_count") or 1)
    mac = int(adb.get("mainline_anchor_count") or 0)
    eac = int(adb.get("expandable_anchor_count") or 0)
    sac_only = int(adb.get("side_anchor_count") or 0)
    side_only_global = bool(mdbg.get("cross_anchor_side_only"))
    tid = rec.get("tid")
    edge_n, edge_mean = _stage3_graph_cross_anchor_stats(tid, graph)
    edge_boost = _stage3_clamp01(edge_mean) * min(1.0, edge_n / 3.0) if edge_n else 0.0

    if ac <= 1:
        score = _stage3_clamp01(0.38 + 0.28 * edge_boost)
        notes.append("single_anchor:base+graph")
    else:
        mainline_ratio = mac / max(1, ac)
        expand_ratio = eac / max(1, ac)
        score = (
            0.12
            + 0.38 * _stage3_clamp01(mainline_ratio)
            + 0.32 * _stage3_clamp01(expand_ratio)
            + 0.18 * edge_boost
        )
        score = _stage3_clamp01(score)
        notes.append("multi_anchor:mainline_ratio+expand_ratio+graph")

    cross_pen = 0.0
    if side_only_global:
        cross_pen += 0.38
        notes.append("cross_anchor_side_only_penalty:-0.38")
    if sac_only >= 2 and mac == 0 and ac >= 2:
        cross_pen += 0.14
        notes.append("multi_side_only_anchors_penalty:-0.14")
    score = _stage3_clamp01(score - cross_pen)
    return {
        "score": score,
        "components": {
            "anchor_count": ac,
            "mainline_anchor_count": mac,
            "expandable_anchor_count": eac,
            "side_only_anchor_count": sac_only,
            "cross_anchor_edge_count": edge_n,
            "cross_anchor_edge_hint_mean": edge_mean,
            "cross_anchor_side_only": side_only_global,
            "cross_anchor_side_only_penalty": min(0.38, 0.38 if side_only_global else 0.0)
            + (0.14 if (sac_only >= 2 and mac == 0 and ac >= 2) else 0.0),
        },
        "notes": notes,
    }


def _compute_stage3_backbone_alignment(rec: Dict[str, Any]) -> Dict[str, Any]:
    """主轴/JD 骨架贴合：mainline + expand + 多来源 parent anchor 统计 + 结构罚则。"""
    notes: List[str] = []
    ex = rec.get("stage3_explain") or {}
    mh = int(rec.get("mainline_hits") or 0)
    evid = list(rec.get("source_evidence_list") or [])
    exp_local_frac = (
        sum(1 for e in evid if isinstance(e, dict) and e.get("can_expand_local")) / max(1, len(evid))
    )
    can_ex = bool(rec.get("can_expand") or rec.get("can_expand_from_2a"))
    try:
        pas_raw = float(
            rec.get("best_parent_anchor_final_score") or rec.get("parent_anchor_final_score") or 0.0
        )
    except (TypeError, ValueError):
        pas_raw = 0.0
    a_sc = _stage3_clamp01(pas_raw / 1.2)
    rk = rec.get("best_parent_anchor_step2_rank")
    try:
        rk_i = int(rk) if rk is not None else 99
    except (TypeError, ValueError):
        rk_i = 99
    rank_prior = _stage3_clamp01(1.0 / (1.0 + 0.12 * max(0, rk_i - 1)))
    bb = (
        0.28 * _stage3_clamp01(min(1.0, mh / 2.0))
        + 0.24 * (1.0 if can_ex else 0.0)
        + 0.18 * _stage3_clamp01(exp_local_frac)
        + 0.18 * a_sc
        + 0.12 * rank_prior
    )
    if _stage3_is_conditioned_only(rec):
        bb = max(0.0, bb - 0.18)
        notes.append("conditioned_only:-0.18")
    flags = rec.get("bucket_reason_flags") or []
    if "locked_mainline_no_expand" in flags:
        bb = max(0.0, bb - 0.08)
        notes.append("locked_mainline_no_expand:-0.08")
    if "cross_anchor_but_side_only" in flags:
        bb = max(0.0, bb - 0.12)
        notes.append("cross_anchor_but_side_only:-0.12")
    bb = _stage3_clamp01(bb)
    return {
        "score": bb,
        "components": {
            "mainline_hits": mh,
            "expandable_support": float(can_ex),
            "expand_local_fraction": exp_local_frac,
            "anchor_score_prior": a_sc,
            "anchor_rank_prior": rank_prior,
            "best_parent_anchor_step2_rank": rk_i,
        },
        "notes": notes,
    }


def _compute_stage3_risk_penalty_dimension(rec: Dict[str, Any]) -> Dict[str, Any]:
    """风险 penalty 块：0~1，越高越差；显式混合冲突 / provenance / 结构信号。"""
    notes: List[str] = []
    gen = _stage3_clamp01(rec.get("generic_risk"))
    poly = _stage3_clamp01(rec.get("polysemy_risk"))
    obj = _stage3_clamp01(rec.get("object_like_risk"))
    drift = _stage3_clamp01(rec.get("semantic_drift_risk"))
    mdbg = rec.get("merge_debug_summary") or {}
    prov = rec.get("provenance_summary") or {}

    c_conflict = int(bool(mdbg.get("primary_bucket_conflict")))
    c_tr = int(bool(mdbg.get("term_role_conflict")))
    c_ria = int(bool(mdbg.get("role_in_anchor_conflict")))
    c_ms = int(bool(mdbg.get("mainline_side_conflict")))
    conflict_n = c_conflict + c_tr + c_ria + c_ms
    conflict_penalty = min(0.34, 0.085 * conflict_n)
    if c_conflict:
        notes.append("bucket_conflict")
    if c_ms:
        notes.append("mainline_side_conflict")

    ca_side_pen = 0.22 if mdbg.get("cross_anchor_side_only") else 0.0
    single_path_pen = 0.12 if prov.get("single_path_multi_anchor_weak_signal") else 0.0
    if prov.get("single_path_only") and int(prov.get("distinct_source_type_count") or 0) <= 1:
        single_path_pen += 0.06
        notes.append("single_path_narrow_provenance")

    score = _stage3_clamp01(
        0.20 * gen
        + 0.18 * poly
        + 0.18 * obj
        + 0.16 * drift
        + conflict_penalty
        + ca_side_pen
        + single_path_pen
    )
    return {
        "score": score,
        "components": {
            "generic_risk": gen,
            "polysemy_risk": poly,
            "object_like_risk": obj,
            "drift_risk": drift,
            "conflict_penalty": conflict_penalty,
            "single_path_penalty": single_path_pen,
            "cross_anchor_side_only_penalty": ca_side_pen,
            "conflict_event_count": conflict_n,
        },
        "notes": notes,
    }


def _build_stage3_global_coherence_score(rec: Dict[str, Any], recall: Any) -> Tuple[float, Dict[str, Any]]:
    """
    全局一致性主分：显式四分项 + 线性正项 + risk 扣分；再与 legacy unified 小幅 blend。
    TODO(Stage3 GC): admission / paper lane 进一步读 stage3_score_breakdown。
    """
    graph = _stage3_get_candidate_graph_from_recall(recall)
    lf = _compute_stage3_local_fit(rec)
    cx = _compute_stage3_cross_anchor_coherence(rec, graph)
    bb = _compute_stage3_backbone_alignment(rec)
    rk = _compute_stage3_risk_penalty_dimension(rec)

    pos = (
        STAGE3_GC_W_LOCAL_FIT * float(lf["score"])
        + STAGE3_GC_W_CROSS * float(cx["score"])
        + STAGE3_GC_W_BACKBONE * float(bb["score"])
    )
    gc_core = float(pos) - STAGE3_GC_W_RISK * float(rk["score"])
    gc_final = max(0.0, min(1.0, gc_core))

    unified_legacy, uni_bd = _compute_stage3_unified_continuous_score(rec)
    alpha = float(STAGE3_GC_LEGACY_BLEND)
    final = max(0.0, min(1.0, (1.0 - alpha) * gc_final + alpha * unified_legacy))

    breakdown: Dict[str, Any] = {
        "local_fit": lf,
        "cross_anchor_coherence": cx,
        "backbone_alignment": bb,
        "risk_penalty": rk,
        "combined_positive": pos,
        "gc_pre_blend": gc_final,
        "legacy_unified_score": unified_legacy,
        "legacy_unified_breakdown": uni_bd,
        "blend_alpha_legacy": alpha,
        "final_score": final,
        "used_candidate_graph_cross_edges": bool(graph.get("cross_anchor_support_edges")),
    }
    return final, breakdown


def _paper_recall_dynamic_floor(
    candidates: List[Dict[str, Any]],
    score_key: str = "final_score",
) -> float:
    """Paper 动态下限：Top1(score_key)×REL 与 ABS 取 max。paper 选词可与 final_score 同尺度或改用 paper_select_score。"""
    if not candidates:
        return PAPER_RECALL_DYNAMIC_FLOOR_ABS
    top = max(float(r.get(score_key) or 0.0) for r in candidates)
    return max(PAPER_RECALL_DYNAMIC_FLOOR_ABS, top * PAPER_RECALL_DYNAMIC_FLOOR_REL)


def _print_stage3_unified_breakdown(
    survivors: List[Dict[str, Any]],
    top_k: int = STAGE3_UNIFIED_SCORE_DEBUG_TOP_K,
) -> None:
    """打印连续分特征拆解。默认 Top 很少（见 `STAGE3_UNIFIED_SCORE_DEBUG_TOP_K`）；paper 优先看 pool / centrality。**STAGE3_DEBUG_FOCUS_TERMS** 非空时只打命中行。"""
    if not STAGE3_UNIFIED_SCORE_DEBUG or not survivors:
        return
    rows = sorted(survivors, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    rows = _stage3_audit_filter_rows(rows, top_k)
    if not rows:
        return
    print("\n" + "-" * 80)
    print("[Stage3 unified score breakdown] 连续分项 → final_score（无 bucket 乘子链）")
    print("-" * 80)
    for rec in rows:
        bd = rec.get("stage3_unified_breakdown") or {}
        term = (rec.get("term") or "")[:28]
        print(
            f"term={term!r} | seed={bd.get('seed_score', 0):.3f} anchor_id={bd.get('anchor_identity', 0):.3f} "
            f"jd_align={bd.get('jd_align', 0):.3f} fam_cent={bd.get('family_centrality', 0):.3f} "
            f"cross={bd.get('cross_anchor_strength', 0):.3f} mainline={bd.get('mainline_strength', 0):.3f} "
            f"expand={bd.get('expandability', 0):.3f}"
        )
        print(
            f"  risks: drift={bd.get('semantic_drift_risk', 0):.3f} generic={bd.get('generic_risk', 0):.3f} "
            f"poly={bd.get('polysemy_risk', 0):.3f} obj={bd.get('object_like_risk', 0):.3f} | "
            f"final={float(rec.get('final_score') or 0):.4f} bucket={rec.get('stage3_bucket')!r} (仅观测)"
        )


def _print_stage3_global_coherence_breakdown(
    survivors: List[Dict[str, Any]],
    top_k: int = STAGE3_UNIFIED_SCORE_DEBUG_TOP_K,
) -> None:
    """Global Coherence 四分项 + final（消费 merge aggregate + candidate_graph）。"""
    if not STAGE3_UNIFIED_SCORE_DEBUG or not survivors:
        return
    rows = sorted(survivors, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    rows = _stage3_audit_filter_rows(rows, top_k)
    if not rows:
        return
    print("\n" + "-" * 80)
    print("[Stage3 global coherence breakdown] local_fit | cross_anchor | backbone | risk_pen | gc_pre | blend | final")
    print("-" * 80)
    for rec in rows:
        sb = rec.get("stage3_score_breakdown") or {}
        term = (rec.get("term") or "")[:26]
        lf = float((sb.get("local_fit") or {}).get("score") or 0.0)
        cx = float((sb.get("cross_anchor_coherence") or {}).get("score") or 0.0)
        bb = float((sb.get("backbone_alignment") or {}).get("score") or 0.0)
        rk = float((sb.get("risk_penalty") or {}).get("score") or 0.0)
        gcp = float(sb.get("gc_pre_blend") or 0.0)
        fin = float(rec.get("final_score") or 0.0)
        graphed = sb.get("used_candidate_graph_cross_edges")
        print(
            f"term={term!r} | lf={lf:.3f} cx={cx:.3f} bb={bb:.3f} risk={rk:.3f} "
            f"gc_pre={gcp:.3f} blend_α={sb.get('blend_alpha_legacy')} | final={fin:.4f} graph_x={graphed}"
        )
        adb = rec.get("anchor_support_summary") or {}
        mdbg = rec.get("merge_debug_summary") or {}
        print(
            f"  aggregate_audit: anc={adb.get('anchor_count')} mac={adb.get('mainline_anchor_count')} "
            f"eac={adb.get('expandable_anchor_count')} side_only_global={mdbg.get('cross_anchor_side_only')} "
            f"prov_multi={ (rec.get('provenance_summary') or {}).get('multi_source')}"
        )


def _print_stage3_risky_coherence_audit(survivors: List[Dict[str, Any]], top_k: int = 12) -> None:
    """risky 候选：为何 risky（GC 视角，非只看 bucket 名）。"""
    if not STAGE3_AUDIT_DEBUG or not survivors:
        return
    risky = [r for r in survivors if (r.get("stage3_bucket") or "").strip().lower() == "risky"]
    if not risky:
        return
    risky.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    print("\n" + "-" * 80)
    print("[Stage3 risky coherence audit] term | final | risk块 | cross_side | conflicts | prov | reasons")
    print("-" * 80)
    for rec in risky[:top_k]:
        term = (rec.get("term") or "")[:22]
        fs = float(rec.get("final_score") or 0.0)
        sb = rec.get("stage3_score_breakdown") or {}
        rp = (sb.get("risk_penalty") or {}).get("components") or {}
        mdbg = rec.get("merge_debug_summary") or {}
        prov = rec.get("provenance_summary") or {}
        reasons = rec.get("risk_reasons") or []
        print(
            f"{term!r} | final={fs:.3f} | risk_pen={float((sb.get('risk_penalty') or {}).get('score') or 0):.3f} "
            f"| ca_side={mdbg.get('cross_anchor_side_only')} | "
            f"conf={rp.get('conflict_event_count')} | narrow_1path={prov.get('single_path_only')} | {reasons}"
        )


def _print_stage3_paper_cutoff_audit(
    ordered: List[Dict[str, Any]],
    floor: float,
    max_terms: int,
) -> None:
    """eligible 内按 paper 选词序：paper_select_score + family + 动态 floor，看谁进/谁被刷掉。"""
    if not STAGE3_PAPER_CUTOFF_AUDIT or not ordered:
        return
    print("\n" + "-" * 80)
    print(
        "[Stage3 paper cutoff] rank | term | final | p_ready | p_sel | family_key | selected | reason"
    )
    print(f"  dynamic_floor(on paper_select_score)={floor:.4f} max_terms={max_terms}")
    print("-" * 80)
    for i, rec in enumerate(ordered, start=1):
        term = (rec.get("term") or "")[:22]
        fs = float(rec.get("final_score") or 0.0)
        pr = float(rec.get("paper_readiness") or 0.0)
        pss = float(rec.get("paper_select_score") or 0.0)
        fk = str(rec.get("family_key") or "")[:18]
        sel = bool(rec.get("selected_for_paper", False))
        reason = str(rec.get("paper_cutoff_reason") or rec.get("select_reason") or "")
        print(
            f"{i:4d} | {term:22} | {fs:5.3f} | {pr:5.3f} | {pss:5.3f} | {fk:18} | {str(sel):8} | {reason}"
        )


def _print_stage3_paper_selection_narrow_audit(ordered: List[Dict[str, Any]]) -> None:
    """
    [Stage3 paper selection audit] 窄表：回答「bonus 型词是否仍靠 final_score 抬进 paper」。
    与 cutoff 表重叠度高：**仅 STAGE3_DEBUG_FOCUS_TERMS 非空** 时打印命中行，避免双表刷屏。
    """
    if not STAGE3_PAPER_CUTOFF_AUDIT or not ordered or not STAGE3_DEBUG_FOCUS_TERMS:
        return
    print("\n" + "-" * 80)
    print("[Stage3 paper selection audit] eligible · structural ranking (no blocklist)")
    print("-" * 80)
    hdr = (
        "term · fs · ready · p_sel · bucket · role · ml · anch · exp co pg sel · "
        "panch · a_sc · rk · a_n · r_sm · psf · sup_reason · outcome"
    )
    print(hdr)
    print("-" * 96)
    for rec in ordered:
        if (rec.get("term") or "").strip() not in STAGE3_DEBUG_FOCUS_TERMS:
            continue
        term = (rec.get("term") or "")[:18]
        fs = float(rec.get("final_score") or 0.0)
        ready = float(rec.get("paper_readiness") or 0.0)
        pss = float(rec.get("paper_select_score") or 0.0)
        b = (rec.get("stage3_bucket") or "")[:6]
        role = (rec.get("term_role") or "")[:10]
        ml = int(rec.get("mainline_hits") or 0)
        anch = int(rec.get("anchor_count") or 0)
        exp = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
        co = _stage3_is_conditioned_only(rec)
        pg = float(rec.get("paper_grounding") or _paper_grounding_score(rec))
        sel = bool(rec.get("selected_for_paper", False))
        if sel:
            outcome = str(rec.get("select_reason") or "selected")
        else:
            outcome = str(rec.get("paper_reject_reason") or rec.get("paper_cutoff_reason") or "")
        pas = float(
            rec.get("best_parent_anchor_final_score")
            or rec.get("parent_anchor_final_score")
            or 0.0
        )
        par = rec.get("best_parent_anchor_step2_rank")
        if par is None:
            par = rec.get("parent_anchor_step2_rank")
        par_i = int(par) if par is not None else 0
        panch = (rec.get("parent_anchor") or "")[:10]
        if not panch and rec.get("parent_anchors"):
            pa_list = rec.get("parent_anchors") or []
            panch = str(pa_list[0])[:10] if pa_list else ""
        a_n = float(rec.get("paper_anchor_score_norm") or 0.0)
        r_sm = float(rec.get("paper_anchor_rank_smooth") or 0.0)
        psf = float(rec.get("paper_support_factor") or 1.0)
        sr = str(rec.get("paper_support_reason") or "-")[:22]
        print(
            f"{term:18} | {fs:4.2f} | {ready:5.2f} | {pss:5.2f} | {b:6} | {role:10} | {ml:^2} | {anch:^4} | "
            f"{str(exp)[:1]:^3} | {str(co)[:1]:^5} | {pg:4.2f} | {str(sel)[:1]:^3} | "
            f"panch={panch:10} a_sc={pas:4.2f} r={par_i:^3} "
            f"a_n={a_n:4.2f} r_sm={r_sm:4.2f} | psf={psf:4.2f} | {sr:22} | {outcome[:18]}"
        )


def _print_stage3_paper_floor_block_audit(
    ordered: List[Dict[str, Any]],
    floor: float,
) -> None:
    """
    [Stage3 paper floor block audit]
    仅 core、未入选、且因 dynamic_floor 被拒：对齐「主线可扩却被 paper_select_score 压出 floor」类误杀排查。
    """
    if not STAGE3_PAPER_CUTOFF_AUDIT or not ordered:
        return
    blocks: List[Dict[str, Any]] = []
    for rec in ordered:
        if rec.get("selected_for_paper"):
            continue
        if (rec.get("stage3_bucket") or "").strip().lower() != "core":
            continue
        reason = str(rec.get("paper_reject_reason") or rec.get("select_reason") or "")
        cutoff = str(rec.get("paper_cutoff_reason") or "")
        if "below_dynamic_floor" not in reason and "below_dynamic_floor" not in cutoff:
            continue
        blocks.append(rec)
    if not blocks:
        return
    print("\n" + "-" * 80)
    print(
        "[Stage3 paper floor block audit] core · below_dynamic_floor（主场：robot control / route planning 等误杀）"
    )
    print(f"  dynamic_floor(paper_select_score)={floor:.4f}")
    print("-" * 80)
    print(
        "term          | final | ready | p_sel | floor | Δfloor | ml | exp | p_anchor      | pa_sc"
    )
    print("-" * 80)
    for rec in blocks:
        t = (rec.get("term") or "")[:14]
        fs = float(rec.get("final_score") or 0.0)
        rd = float(rec.get("paper_readiness") or 0.0)
        pss = float(rec.get("paper_select_score") or 0.0)
        delta = floor - pss
        ml = int(rec.get("mainline_hits") or 0)
        exp = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
        pa = str(rec.get("parent_anchor") or "")[:14]
        if not pa and rec.get("parent_anchors"):
            pa = str((rec.get("parent_anchors") or [""])[0])[:14]
        pas = float(
            rec.get("best_parent_anchor_final_score")
            or rec.get("parent_anchor_final_score")
            or 0.0
        )
        print(
            f"{t:14} | {fs:5.3f} | {rd:5.3f} | {pss:5.3f} | {floor:5.3f} | {delta:+6.3f} | "
            f"{ml:^2} | {str(exp)[:1]:^3} | {pa:14} | {pas:5.3f}"
        )


def _print_stage3_paper_centrality_audit(ordered: List[Dict[str, Any]], top_n: int = 12) -> None:
    """
    paper_select_score 结构化分解：一眼看 RL / robotic arm 仍在是因为 fs 高还是主线/父锚弱；含 **池规模** 便于对照 pool penalty。
    """
    if not STAGE3_PAPER_CUTOFF_AUDIT or not ordered:
        return
    rows = sorted(
        ordered,
        key=lambda x: float(x.get("paper_select_score") or 0.0),
        reverse=True,
    )[:top_n]
    print("\n" + "-" * 80)
    print(
        "[Stage3 paper centrality audit] top eligible by paper_select_score "
        "(base=L×R + tier-scaled struct bonus − pool; +b_pre=缩放前 struct 总和, sc=PAPER_SELECT_BONUS_TIER_SCALE_*)"
    )
    print("-" * 80)
    print(
        "term                  | p_bef | parent_anch  | a_sc | rk | primary_bucket | ml | ex | "
        "fs   | ready | base | +b_pre | sc | +bonus | p_sel"
    )
    print("-" * 80)
    for row in rows:
        term = str(row.get("term") or "")[:22]
        p_bef = int(
            row.get("paper_select_pool_size")
            or row.get("papers_before_filter")
            or row.get("degree_w")
            or 0
        )
        panch = str(row.get("parent_anchor") or "")[:12]
        if not panch and row.get("parent_anchors"):
            panch = str((row.get("parent_anchors") or [""])[0])[:12]
        a_sc = float(
            row.get("best_parent_anchor_final_score")
            or row.get("parent_anchor_final_score")
            or 0.0
        )
        rk = row.get("best_parent_anchor_step2_rank")
        if rk is None:
            rk = row.get("parent_anchor_step2_rank")
        rk_i = int(rk) if rk is not None else 999
        pb = str(row.get("primary_bucket") or "")[:14]
        ml = int(row.get("mainline_hits") or 0)
        ex = bool(row.get("can_expand_from_2a", False) or row.get("can_expand", False))
        fs = float(row.get("final_score") or 0.0)
        ready = float(row.get("paper_readiness") or 0.0)
        base = float(row.get("paper_select_score_base") or fs * ready)
        btot = float(row.get("paper_select_score_bonus_total") or 0.0)
        b_pre_v = row.get("paper_select_score_bonus_pre_tier")
        b_pre = float(b_pre_v) if b_pre_v is not None else btot
        b_sc = float(row.get("paper_select_bonus_tier_scale") or 1.0)
        psel = float(row.get("paper_select_score") or 0.0)
        print(
            f"{term:22} | {p_bef:5d} | {panch:12} | {a_sc:4.2f} | {rk_i:2d} | {pb:14} | {ml:^2} | "
            f"{'T' if ex else 'F':^2} | {fs:4.3f} | {ready:5.3f} | {base:5.3f} | {b_pre:+6.3f} | {b_sc:4.2f} | "
            f"{btot:+6.3f} | {psel:5.3f}"
        )


def _print_stage3_paper_pool_penalty_audit(ordered: List[Dict[str, Any]], top_n: int = 16) -> None:
    """eligible 内 paper 排序键分解：底分、加性 bonus、**池规模惩罚**、最终 p_sel（与 centrality 同序）。"""
    if not STAGE3_PAPER_CUTOFF_AUDIT or not ordered:
        return
    rows = sorted(
        ordered,
        key=lambda x: float(x.get("paper_select_score") or 0.0),
        reverse=True,
    )[:top_n]
    print("\n" + "-" * 80)
    print(
        "[Stage3 paper pool penalty audit] eligible · base + tier-scaled bonus − pool_penalty(log10 papers) → p_sel "
        f"(enabled={PAPER_SELECT_POOL_PENALTY_ENABLED}, coef={PAPER_SELECT_POOL_PENALTY_LOG_COEF}, "
        f"max={PAPER_SELECT_POOL_PENALTY_MAX})"
    )
    print("-" * 80)
    print(
        "term                  | papers_bef | base  | +b_pre | sc | +bonus | -pool_pen | p_sel"
    )
    print("-" * 80)
    for row in rows:
        term = str(row.get("term") or "")[:22]
        pbf = int(
            row.get("paper_select_pool_size")
            or row.get("papers_before_filter")
            or row.get("degree_w")
            or 0
        )
        base = float(row.get("paper_select_score_base") or 0.0)
        btot = float(row.get("paper_select_score_bonus_total") or 0.0)
        b_pre_v = row.get("paper_select_score_bonus_pre_tier")
        b_pre = float(b_pre_v) if b_pre_v is not None else btot
        b_sc = float(row.get("paper_select_bonus_tier_scale") or 1.0)
        ppen = float(row.get("paper_select_pool_penalty") or 0.0)
        psel = float(row.get("paper_select_score") or 0.0)
        print(
            f"{term:22} | {pbf:10d} | {base:5.3f} | {b_pre:+6.3f} | {b_sc:4.2f} | {btot:+6.3f} | "
            f"{-ppen:9.3f} | {psel:5.3f}"
        )


def _print_stage3_paper_lane_tier_audit(ordered: List[Dict[str, Any]]) -> None:
    """
    [Stage3 paper lane tier audit]：strong / support_lane / other_eligible / bonus_core 分层后的 **全局遍历序**
    （与截断扫描顺序一致；support_lane 段内 near_miss 先于普通 support，再按 p_sel）。
    """
    if not STAGE3_PAPER_CUTOFF_AUDIT or not ordered:
        return
    print("\n" + "-" * 80)
    print(
        "[Stage3 paper lane tier audit] tier-order scan (strong_main_axis_core → support_lane[near_miss≻] → "
        "other_eligible → bonus_core; "
        f"strong thresholds a_sc≥{PAPER_SELECT_STRONG_MAIN_AXIS_SCORE_MIN} rk≤{PAPER_SELECT_STRONG_MAIN_AXIS_MAX_RANK} "
        f"or ml+exp+a_sc≥{PAPER_SELECT_STRONG_MAIN_AXIS_ML_EXP_SCORE_MIN} rk≤{PAPER_SELECT_STRONG_MAIN_AXIS_ML_EXP_MAX_RANK})"
    )
    print("-" * 80)
    print(
        "term                  | tier                  | p_sel | a_sc | rk | ml | exp | lane   | parent_anchor"
    )
    print("-" * 80)
    for rec in ordered:
        term = str(rec.get("term") or "")[:22]
        tier = str(rec.get("paper_select_lane_tier") or "")[:22]
        psel = float(rec.get("paper_select_score") or 0.0)
        a_sc, rk_i = _paper_parent_anchor_score_and_rank(rec)
        ml = int(rec.get("mainline_hits") or 0)
        ex = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
        lane = str(rec.get("paper_recall_quota_lane") or "")[:7]
        panch = str(rec.get("parent_anchor") or "")[:14]
        if not panch and rec.get("parent_anchors"):
            panch = str((rec.get("parent_anchors") or [""])[0])[:14]
        print(
            f"{term:22} | {tier:22} | {psel:5.3f} | {a_sc:4.2f} | {rk_i:2d} | {ml:^2} | "
            f"{'T' if ex else 'F':^3} | {lane:7} | {panch:14}"
        )


def _format_stage3_blocked_by_terms(selected_terms: List[str], *, max_parts: int = 14) -> str:
    if not selected_terms:
        return "-"
    parts = [t for t in selected_terms if t][:max_parts]
    s = ", ".join(parts)
    if len([t for t in selected_terms if t]) > max_parts:
        s += ", …"
    return s or "-"


def _print_stage3_paper_lane_fill_audit(
    selected: List[Dict[str, Any]],
    support_lane_pool: List[Dict[str, Any]],
    floor: float,
) -> None:
    """
    [Stage3 paper lane fill audit] — 与 README 示例一致，仅 5 行（标题 + 4 个 KPI）。

    - strong / support_lane / bonus_core：主 cutoff 当期 **实际入选** 条数（按 paper_select_lane_tier）。
    - support_lane_remaining_candidates：**support_lane 池**里仍未进栏、且 p_sel≥dynamic_floor 的条数
      （合格 support 仍在池里等名额；与 bonus_core_taken 对照可判断是否「support 未吃满 bonus 却偷跑」）。

    批注：other_eligible 若入选会计入 max_terms，明细见同期 **[Stage3 paper selected composition]**；本块刻意保持极简。
    """
    if not STAGE3_PAPER_CUTOFF_AUDIT or not STAGE3_PAPER_LANE_FILL_AUDIT:
        return
    by_tier: Dict[str, int] = defaultdict(int)
    for rec in selected:
        by_tier[str(rec.get("paper_select_lane_tier") or "unknown")] += 1
    sup_rem_ge_floor = 0
    fl = float(floor)
    for r in support_lane_pool:
        if r.get("selected_for_paper"):
            continue
        if float(r.get("paper_select_score") or 0.0) >= fl - 1e-9:
            sup_rem_ge_floor += 1
    print("\n[Stage3 paper lane fill audit]")
    print(f"strong_main_axis_core={by_tier.get('strong_main_axis_core', 0)}")
    print(f"support_lane_taken={by_tier.get('support_lane', 0)}")
    print(f"bonus_core_taken={by_tier.get('bonus_core', 0)}")
    print(f"support_lane_remaining_candidates={sup_rem_ge_floor}")


def _print_stage3_bonus_core_blocked_audit(
    bonus_core_pool: List[Dict[str, Any]],
    selected_term_list: List[str],
    family_owner: Dict[str, str],
    support_remaining_ge_floor: int,
) -> None:
    """
    [Stage3 bonus-core audit] — 与 README 示例同形：表头一行 + 每条 bonus_core 一行。

    - blocked_by：主截断下**已按扫描序入选的 term 列表**（ comma 分隔），即「被谁挤掉名额」；
      family 去重挡板时为**同 family_key 已选中的词**。
    - support_lane_priority_not_filled：已 **past_paper_recall_max_terms**，且此时仍有 support 池
      p_sel≥floor 未进栏 — 说明名额被前排（含 support）占满，bonus 殿后排到仍无空位（非「RL 抢在 support 前」）。

    批注：打印时机为 **主 cutoff 后、swap/tail 前**；最终入选以 **[Stage3 paper selected composition]** 为准。
    """
    if not STAGE3_PAPER_CUTOFF_AUDIT or not STAGE3_BONUS_CORE_BLOCKED_AUDIT:
        return
    if not bonus_core_pool:
        return
    print("\n[Stage3 bonus-core audit]")
    print("term | p_sel | selected | blocked_by | reason")
    for rec in bonus_core_pool:
        term = str(rec.get("term") or "").strip() or "(empty)"
        pss = float(rec.get("paper_select_score") or 0.0)
        sel = bool(rec.get("selected_for_paper"))
        cutoff = str(rec.get("paper_cutoff_reason") or rec.get("paper_reject_reason") or "")
        fk = str(rec.get("family_key") or "").strip()
        if sel:
            blocked_by = "-"
            reason = "-"
        elif cutoff == "below_dynamic_floor":
            blocked_by = "-"
            reason = "below_dynamic_floor"
        elif cutoff in ("family_duplicate_block", "family_dup_or_below_cut"):
            blocker = family_owner.get(fk, "") if fk else ""
            blocked_by = (blocker or "?").strip()
            reason = "family_duplicate_block"
        elif cutoff == "support_quota_full":
            blocked_by = _format_stage3_blocked_by_terms(selected_term_list)
            reason = "support_quota_full"
        elif cutoff == "past_paper_recall_max_terms":
            blocked_by = _format_stage3_blocked_by_terms(selected_term_list)
            if support_remaining_ge_floor > 0:
                reason = "support_lane_priority_not_filled"
            else:
                reason = "max_terms_exhausted"
        else:
            blocked_by = _format_stage3_blocked_by_terms(selected_term_list)
            reason = cutoff or "unknown"
        sel_s = "True" if sel else "False"
        print(f"{term} | {pss:.3f} | {sel_s} | {blocked_by} | {reason}")


def _print_stage3_paper_selected_composition_audit(selected: List[Dict[str, Any]]) -> None:
    """[Stage3 paper selected composition]：最终入选按 paper_select_lane_tier 计数（含 tail 写入的 tier）。"""
    if not STAGE3_PAPER_CUTOFF_AUDIT:
        return
    if not selected:
        print("\n[Stage3 paper selected composition] selected_total=0")
        return
    bucket = defaultdict(int)
    for rec in selected:
        t = str(rec.get("paper_select_lane_tier") or "unknown")
        bucket[t] += 1
    parts = [f"{k}={bucket[k]}" for k in sorted(bucket.keys())]
    print(
        f"\n[Stage3 paper selected composition] selected_total={len(selected)} " + " ".join(parts)
    )


def _print_stage3_eligible_core_close_call_audit(
    ordered: List[Dict[str, Any]],
    top_n: int = 18,
) -> None:
    """
    eligible 内 **stage3_bucket=core** 按 paper_select_score 名次截取：显式 p_sel_base / bonus / total，
    便于判断截断附近是底分还是加性项（含 anchor_score_bonus）在抬位。
    """
    if (
        not STAGE3_PAPER_CUTOFF_AUDIT
        or not STAGE3_ELIGIBLE_CORE_CLOSE_CALL_AUDIT
        or not ordered
    ):
        return
    core_rows = [
        r
        for r in ordered
        if (r.get("stage3_bucket") or "").strip().lower() == "core"
    ]
    if not core_rows:
        return
    core_rows = core_rows[:top_n]
    print("\n" + "-" * 80)
    print(
        "[Stage3 eligible core close-call audit] "
        "eligible core by paper_select rank — p_sel_base / p_sel_bonus / p_sel_total vs final×ready"
    )
    print("-" * 80)
    print(
        "term              | final | ready | p_sel_base | p_sel_bonus | p_sel_total | "
        "parent_anch  | a_sc | rk"
    )
    print("-" * 80)
    for row in core_rows:
        term = str(row.get("term") or "")[:16]
        fs = float(row.get("final_score") or 0.0)
        ready = float(row.get("paper_readiness") or 0.0)
        pb = float(row.get("paper_select_score_base") or fs * ready)
        pbon = float(row.get("paper_select_score_bonus_total") or 0.0)
        ptot = float(row.get("paper_select_score") or 0.0)
        panch = str(row.get("parent_anchor") or "")[:12]
        if not panch and row.get("parent_anchors"):
            panch = str((row.get("parent_anchors") or [""])[0])[:12]
        a_sc = float(
            row.get("best_parent_anchor_final_score")
            or row.get("parent_anchor_final_score")
            or 0.0
        )
        rk_raw = row.get("best_parent_anchor_step2_rank")
        if rk_raw is None:
            rk_raw = row.get("parent_anchor_step2_rank")
        rk_i = int(rk_raw) if rk_raw is not None else 999
        print(
            f"{term:16} | {fs:5.3f} | {ready:5.3f} | {pb:10.3f} | {pbon:+11.3f} | "
            f"{ptot:11.3f} | {panch:12} | {a_sc:4.2f} | {rk_i:2d}"
        )


def _print_stage3_paper_main_axis_gate_audit(records: List[Dict[str, Any]]) -> None:
    """
    全量 input 顺序：每条是否因「非 JD 全局主轴父锚」在进 eligible 前被挡；对照 RL / robotic arm 等仍存活原因。
    """
    if not STAGE3_PAPER_CUTOFF_AUDIT or not records:
        return
    print("\n" + "-" * 80)
    print(
        "[Stage3 paper main-axis gate audit] core → 父锚 (rk≤MAX) ∨ (a_sc≥MIN) 过门；双不达仍拒；非 core 跳过本门"
    )
    print(
        f"  (STAGE3_PAPER_MAIN_AXIS_GATE_ENABLED={STAGE3_PAPER_MAIN_AXIS_GATE_ENABLED}, "
        f"MAX_RANK={STAGE3_PAPER_MAIN_AXIS_MAX_RANK}, MIN_SCORE={STAGE3_PAPER_MAIN_AXIS_MIN_SCORE})"
    )
    print("-" * 80)
    print(
        "term                  | bucket | parent_anchor  | a_sc | rk | axis_pass | reason"
    )
    print("-" * 80)
    for rec in records:
        term = str(rec.get("term") or "")[:22]
        bucket = str(rec.get("stage3_bucket") or "")[:6]
        panch = str(rec.get("parent_anchor") or "")[:12]
        if not panch and rec.get("parent_anchors"):
            panch = str((rec.get("parent_anchors") or [""])[0])[:12]
        a_sc = float(
            rec.get("best_parent_anchor_final_score")
            or rec.get("parent_anchor_final_score")
            or 0.0
        )
        rk_raw = rec.get("best_parent_anchor_step2_rank")
        if rk_raw is None:
            rk_raw = rec.get("parent_anchor_step2_rank")
        rk = int(rk_raw) if rk_raw is not None else 999
        ap = rec.get("paper_main_axis_gate_pass")
        if ap is None:
            axis_cell = "n/a"
            reason = str(rec.get("paper_reject_reason") or rec.get("paper_cutoff_reason") or "")[
                :28
            ]
        else:
            axis_cell = str(bool(ap))[:9]
            reason = str(rec.get("paper_main_axis_gate_reason") or "")[:28]
        print(
            f"{term:22} | {bucket:6} | {panch:12} | {a_sc:4.2f} | {rk:2d} | {axis_cell:9} | {reason}"
        )


def _print_stage3_paper_gate_summary(
    *,
    core_at_main_axis_gate: int,
    axis_direct_core: int,
    support_pool_n: int,
    merged_eligible_n: int,
    blocked_support: int,
    blocked_risky: int,
    support_soft_admit: int = 0,
    core_axis_near_miss_soft_admit: int = 0,
) -> None:
    """
    一眼区分：paper 入口收窄来自「前置硬挡」还是「合并后排序/配额/floor」。
    - axis_direct_core：过主轴门、进 primary 槽参与排序的 core 条数（合并前在 eligible 主干）。
    - support_pool_n：含 support 统一下池、core 主轴 near-miss 降级、risky 保守入池等（合并前）。
    merged_eligible_n：eligible.extend(support_pool) 后总长（readiness / paper_select_score 的输入规模）。
    """
    if not STAGE3_PAPER_CUTOFF_AUDIT:
        return
    print(
        f"[Stage3 paper gate summary] core_at_main_axis_gate={core_at_main_axis_gate} "
        f"axis_direct_core={axis_direct_core} "
        f"support_pool_n={support_pool_n} "
        f"merged_eligible_n={merged_eligible_n} "
        f"support_soft_admit(weak_sup)={support_soft_admit} "
        f"core_axis_near_miss_soft_admit={core_axis_near_miss_soft_admit} "
        f"blocked_main_axis_hard=0 "
        f"blocked_support={blocked_support} "
        f"blocked_risky={blocked_risky}"
    )


def _print_stage3_paper_gate_reject_audit(counts: Dict[str, int]) -> None:
    """进合并池之前的前置门：按 reject/cutoff 原因聚合计数（解释 survivors 为何没进 merged eligible）。"""
    if not STAGE3_PAPER_CUTOFF_AUDIT or not counts:
        return
    parts = [f"{k}={v}" for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    print(f"[Stage3 paper gate reject audit] " + " ".join(parts))


def _print_stage3_support_soft_admit_audit(rows: List[Dict[str, Any]]) -> None:
    """weak_support_contamination 路径：谁被软放进 eligible、谁仍被硬挡。"""
    if not STAGE3_PAPER_CUTOFF_AUDIT or not rows:
        return
    print("\n" + "-" * 80)
    print(
        "[Stage3 support soft-admit audit] weak_support_contamination 路径 · "
        f"SUPPORT_TO_PAPER_ENABLED={SUPPORT_TO_PAPER_ENABLED}"
    )
    print("-" * 80)
    hdr = (
        "term                 | pbucket        | ml | exp | psf  | p_bef | "
        "admitted | reason"
    )
    print(hdr)
    print("-" * 80)
    for r in rows:
        t = str(r.get("term") or "")[:20]
        pb = str(r.get("primary_bucket") or "")[:14]
        ml = int(r.get("mainline_hits") or 0)
        ex = "T" if r.get("can_expand") else "F"
        psf = float(r.get("paper_support_factor") or 0.0)
        pbf = int(r.get("papers_before") or 0)
        adm = str(bool(r.get("admitted")))[:5]
        rs = str(r.get("support_reason") or r.get("block_reason") or "")[:40]
        print(
            f"{t:20} | {pb:14} | {ml:^2} | {ex:^3} | {psf:4.2f} | {pbf:5d} | "
            f"{adm:8} | {rs}"
        )


def _print_stage3_paper_quota_audit(
    selected: List[Dict[str, Any]],
    max_terms: int,
    support_cap: int,
    support_quota_full_cnt: int,
) -> None:
    """最终入选里 core/support 计数与 support 配额触顶次数。"""
    if not STAGE3_PAPER_CUTOFF_AUDIT:
        return
    n_primary_lane = sum(
        1 for r in selected if (r.get("paper_recall_quota_lane") or "") == "primary"
    )
    n_support_lane = sum(
        1 for r in selected if (r.get("paper_recall_quota_lane") or "") == "support"
    )
    print(
        f"[Stage3 paper quota audit] max_terms={max_terms} support_cap={support_cap} "
        f"selected_total={len(selected)} selected_primary_lane={n_primary_lane} "
        f"selected_support_lane={n_support_lane} "
        f"support_quota_full_count={support_quota_full_cnt}"
    )


def _print_stage3_paper_near_miss_audit(
    ordered: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    floor: float,
) -> None:
    """
    未进 final_term_ids_for_paper 但 paper_select_score 贴近 floor 或末席入选分的 eligible 词。
    """
    if not STAGE3_PAPER_CUTOFF_AUDIT or not ordered:
        return
    sel_pss = [float(r.get("paper_select_score") or 0.0) for r in selected]
    min_sel = min(sel_pss) if sel_pss else 0.0
    fifth = float(selected[4].get("paper_select_score") or 0.0) if len(selected) >= 5 else 0.0
    rows: List[Tuple[float, Dict[str, Any], str, float]] = []
    for rec in ordered:
        if rec.get("selected_for_paper"):
            continue
        pss = float(rec.get("paper_select_score") or 0.0)
        near_ok = pss >= floor * 0.98
        if min_sel > 0:
            near_ok = near_ok or pss >= min_sel - 0.04
        if len(selected) >= 5 and fifth > 0:
            near_ok = near_ok or pss >= fifth - 0.04
        if not near_ok:
            continue
        reason = str(rec.get("paper_reject_reason") or rec.get("paper_cutoff_reason") or "")
        delta = (min_sel - pss) if min_sel > 0 else 0.0
        rows.append((pss, rec, reason, delta))
    if not rows:
        return
    rows.sort(key=lambda x: -x[0])
    print("\n" + "-" * 80)
    print("[Stage3 paper near-miss audit] eligible 未入选 · 贴近 floor/末席/top5 线")
    print(
        f"  floor={floor:.4f} min_selected_p_sel={min_sel:.4f} fifth_selected={fifth:.4f}"
    )
    print("-" * 80)
    pa_hdr = "parent_anch"
    print(
        f"term                 | bkt  | pbucket      | final | ready | p_sel  | Δ末席  | "
        f"a_sc | rk | {pa_hdr:^10} | sup_reason | reject/cutoff"
    )
    print("-" * 80)
    for pss, rec, reason, delta in rows[:24]:
        t = (rec.get("term") or "")[:20]
        bkt = str(rec.get("stage3_bucket") or "")[:4]
        pbk = str(rec.get("primary_bucket") or "")[:12]
        fs = float(rec.get("final_score") or 0.0)
        rd = float(rec.get("paper_readiness") or 0.0)
        pas = float(
            rec.get("best_parent_anchor_final_score")
            or rec.get("parent_anchor_final_score")
            or 0.0
        )
        pr = rec.get("best_parent_anchor_step2_rank")
        if pr is None:
            pr = rec.get("parent_anchor_step2_rank")
        pr_i = int(pr) if pr is not None else 0
        panch = str(rec.get("parent_anchor") or "")[:10]
        if not panch and rec.get("parent_anchors"):
            panch = str((rec.get("parent_anchors") or [""])[0])[:10]
        sup_rs = str(rec.get("paper_support_reason") or "")[:18]
        print(
            f"{t:20} | {bkt:4} | {pbk:12} | {fs:5.3f} | {rd:5.3f} | {pss:6.3f} | {delta:+6.3f} | "
            f"{pas:4.2f} | {pr_i:2d} | {panch:10} | {sup_rs:18} | {reason[:22]}"
        )


def _print_stage3_support_contamination_summary(cands: List[Dict[str, Any]]) -> None:
    """
    Stage3→4 之间：用论文召回前的结构信号标记「可疑 support 入稿」。
    （与 Stage4 的 tg 统计同源 proxy：paper_grounding 预估值 + anchor/ml。）
    """
    if not STAGE3_SUPPORT_CONTAMINATION_AUDIT or not cands:
        return
    suspicious: List[Dict[str, Any]] = []
    for rec in cands:
        if not rec.get("selected_for_paper"):
            continue
        b = (rec.get("stage3_bucket") or "").strip().lower()
        if b != "support":
            continue
        ac = int(rec.get("anchor_count") or 0)
        mh = int(rec.get("mainline_hits") or 0)
        pg = _paper_grounding_score(rec)
        if ac <= 1 and mh <= 1:
            suspicious.append(rec)
        elif pg < 0.22:
            suspicious.append(rec)
    if not suspicious:
        return
    print("\n" + "-" * 80)
    print("[Stage3 support contamination] 已选入 paper 的 support 词（结构可疑摘要）")
    print("-" * 80)
    for rec in suspicious:
        term = rec.get("term") or ""
        pg = _paper_grounding_score(rec)
        gating = str(rec.get("paper_support_gate") or "")
        print(
            f"term={term!r} anchor_count={int(rec.get('anchor_count') or 0)} "
            f"mainline_hits={int(rec.get('mainline_hits') or 0)} "
            f"paper_grounding_proxy={pg:.3f} final_score={float(rec.get('final_score') or 0):.4f} "
            f"paper_support_gate={gating!r} "
            f"decision=suspicious_support_term"
        )


# --- Stage3 merge：multi-anchor evidence aggregation（非 winner-snapshot）---
def _stage3_rec_bucket_norm(rec: Dict[str, Any]) -> str:
    meta = rec.get("stage2_local_meta") or {}
    return (meta.get("primary_bucket") or rec.get("primary_bucket") or "").strip().lower()


def _stage3_merge_source_evidence_slice(rec: Dict[str, Any]) -> Dict[str, Any]:
    meta = rec.get("stage2_local_meta") or {}
    rf = rec.get("risk_flags")
    if rf is None:
        rf_list: List[Any] = []
    elif isinstance(rf, list):
        rf_list = rf
    else:
        rf_list = [rf]
    st_raw = rec.get("source_type") or rec.get("source") or rec.get("origin") or ""
    st = str(st_raw).strip().lower()
    return {
        "parent_anchor": (rec.get("parent_anchor") or "").strip(),
        "anchor_term": (rec.get("parent_anchor") or "").strip(),
        "candidate_source": str(st_raw).strip(),
        "term_role_local": (rec.get("term_role") or "").strip(),
        "identity_score": float(rec.get("identity_score") or rec.get("sim_score") or 0.0),
        "sim_score": float(rec.get("sim_score") or rec.get("identity_score") or 0.0),
        "retrieval_role": (rec.get("retrieval_role") or "").strip(),
        "risk_flags": rf_list[:16],
        "source_type": st,
        "role_in_anchor": (rec.get("role_in_anchor") or "").strip().lower(),
        "can_expand_local": bool(rec.get("can_expand_local")),
        "primary_bucket_local": (meta.get("primary_bucket") or rec.get("primary_bucket") or "").strip(),
    }


def _stage3_merge_primary_bucket_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    cnt: Counter = Counter()
    for r in records:
        k = _stage3_rec_bucket_norm(r)
        cnt[k or ""] += 1
    distribution = dict(sorted(cnt.items(), key=lambda x: (-x[1], x[0])))
    nonempty = {bk: c for bk, c in cnt.items() if bk}
    if nonempty:
        derived = max(
            nonempty.keys(),
            key=lambda b: (STAGE3_PRIMARY_BUCKET_MERGE_PRIORITY.get(b, 0), nonempty[b]),
        )
    else:
        derived = ""
    return {"bucket_distribution": distribution, "derived_primary_bucket": derived}


def _stage3_merge_pick_display_primary_bucket(records: List[Dict[str, Any]], derived_lower: str) -> str:
    if not derived_lower:
        return ""
    for r in records:
        if _stage3_rec_bucket_norm(r) == derived_lower:
            meta = r.get("stage2_local_meta") or {}
            pb = (meta.get("primary_bucket") or r.get("primary_bucket") or "").strip()
            if pb:
                return pb
    return derived_lower


def _stage3_merge_pick_primary_reason(records: List[Dict[str, Any]], derived_lower: str) -> str:
    if not derived_lower:
        return ""
    for r in records:
        if _stage3_rec_bucket_norm(r) == derived_lower:
            pr = (r.get("primary_reason") or "").strip()
            if pr:
                return pr
    return ""


def _stage3_merge_anchor_support_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    anchor_to_roles: Dict[str, Set[str]] = defaultdict(set)
    expand_anchors: Set[str] = set()
    for r in records:
        pa = (r.get("parent_anchor") or "").strip()
        ra = (r.get("role_in_anchor") or "").strip().lower()
        if pa and ra:
            anchor_to_roles[pa].add(ra)
        ex = bool(r.get("can_expand") or r.get("can_expand_local") or r.get("can_expand_from_2a"))
        if ex and pa:
            expand_anchors.add(pa)
    anchor_roles: Dict[str, str] = {}
    for a, roles in anchor_to_roles.items():
        if "mainline" in roles and "side" in roles:
            anchor_roles[a] = "mixed"
        elif "mainline" in roles:
            anchor_roles[a] = "mainline"
        elif "side" in roles:
            anchor_roles[a] = "side"
        else:
            anchor_roles[a] = "other"
    all_anchors = sorted(anchor_to_roles.keys())
    mainline_anchor_count = len([a for a in anchor_to_roles if "mainline" in anchor_to_roles[a]])
    side_anchor_count = len(
        [a for a in anchor_to_roles if "mainline" not in anchor_to_roles[a] and "side" in anchor_to_roles[a]]
    )
    return {
        "anchor_count": len(all_anchors),
        "anchors": all_anchors,
        "mainline_anchor_count": mainline_anchor_count,
        "side_anchor_count": side_anchor_count,
        "expandable_anchor_count": len(expand_anchors),
        "anchor_roles": dict(sorted(anchor_roles.items())),
    }


def _stage3_merge_provenance_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    cnt: Counter = Counter()
    parent_nonempty = {
        (r.get("parent_anchor") or "").strip()
        for r in records
        if (r.get("parent_anchor") or "").strip()
    }
    for r in records:
        raw = r.get("source_type") or r.get("source") or r.get("origin") or ""
        key = str(raw).strip().lower() or "__unknown__"
        cnt[key] += 1
    return {
        "source_type_counts": dict(sorted(cnt.items(), key=lambda x: (-x[1], x[0]))),
        "distinct_source_type_count": len(cnt),
        "multi_source": len(cnt) > 1,
        "single_path_only": len(cnt) <= 1,
        "single_path_multi_anchor_weak_signal": bool(len(cnt) <= 1 and len(parent_nonempty) >= 2),
    }


def _stage3_merge_role_distribution(records: List[Dict[str, Any]]) -> Dict[str, int]:
    cnt: Counter = Counter()
    for r in records:
        tr = (r.get("term_role") or "").strip().lower()
        cnt[tr or ""] += 1
    return dict(sorted(cnt.items(), key=lambda x: (-x[1], x[0])))


def _stage3_merge_derived_term_role(
    records: List[Dict[str, Any]],
    term_role_distribution: Dict[str, int],
) -> str:
    has_mainline_evidence = any(
        (r.get("role_in_anchor") or "").strip().lower() == "mainline" for r in records
    )
    has_primary_like = any(
        (r.get("term_role") or "").strip().lower() == "primary"
        or (r.get("source_type") or r.get("source") or r.get("origin") or "").strip().lower() == "similar_to"
        for r in records
    )
    any_side = any((r.get("role_in_anchor") or "").strip().lower() == "side" for r in records)
    all_side_or_empty = all(
        (r.get("role_in_anchor") or "").strip().lower() in ("side", "") for r in records
    )
    if has_mainline_evidence or has_primary_like:
        return "primary"
    if any_side and all_side_or_empty and not has_mainline_evidence:
        return "primary_side"
    nonempty_roles = {k: v for k, v in term_role_distribution.items() if k}
    if nonempty_roles:
        return max(
            nonempty_roles.keys(),
            key=lambda k: (STAGE3_TERM_ROLE_MERGE_PRIORITY.get(k, 0), nonempty_roles[k]),
        )
    return ""


def _stage3_merge_debug_summary(
    records: List[Dict[str, Any]],
    anchor_sup: Dict[str, Any],
) -> Dict[str, Any]:
    buckets = [_stage3_rec_bucket_norm(r) for r in records]
    bucket_uniq = {b for b in buckets if b}
    roles_tr = {(r.get("term_role") or "").strip().lower() for r in records}
    roles_tr.discard("")
    ra_roles = {
        (r.get("role_in_anchor") or "").strip().lower()
        for r in records
        if (r.get("role_in_anchor") or "").strip()
    }
    expands = [bool(r.get("can_expand") or r.get("can_expand_local") or r.get("can_expand_from_2a")) for r in records]
    has_t = any(expands)
    has_f = any(not x for x in expands)
    risk_union: Set[Any] = set()
    for r in records:
        rf = r.get("risk_flags")
        if isinstance(rf, list):
            risk_union.update(rf)
        elif rf is not None:
            risk_union.add(rf)
    srcs = sorted(
        {
            str(r.get("source_type") or r.get("source") or r.get("origin") or "").strip()
            for r in records
            if (r.get("source_type") or r.get("source") or r.get("origin") or "").strip()
        }
    )
    sh = sum(1 for r in records if (r.get("role_in_anchor") or "").strip().lower() == "side")
    mac = int(anchor_sup.get("mainline_anchor_count") or 0)
    ac = int(anchor_sup.get("anchor_count") or 0)
    return {
        "original_record_count": len(records),
        "anchors": list(anchor_sup.get("anchors") or []),
        "sources": srcs,
        "primary_bucket_conflict": len(bucket_uniq) > 1,
        "term_role_conflict": len(roles_tr) > 1,
        "role_in_anchor_conflict": len(ra_roles) > 1,
        "mainline_side_conflict": ("mainline" in ra_roles and "side" in ra_roles),
        "can_expand_conflict": bool(has_t and has_f and len(records) > 1),
        "cross_anchor_side_only": bool(ac >= 2 and mac == 0 and sh > 0),
        "risk_flags_union": sorted(str(x) for x in risk_union),
        "legacy_compatibility_mirror_fields": [
            "primary_bucket",
            "primary_reason",
            "term_role",
            "role_in_anchor",
            "can_expand",
            "can_expand_from_2a",
            "can_expand_local",
            "fallback_primary",
            "mainline_candidate",
        ],
        "note": "TODO(Stage3 merge): legacy mirrors derived; global rerank 应消费 *_summary / *_list。",
    }


def _stage3_merge_has_primary_support_flags(records: List[Dict[str, Any]]) -> Tuple[bool, bool]:
    hp = False
    hs = False
    for r in records:
        tr = (r.get("term_role") or "").strip().lower()
        st = (r.get("source_type") or r.get("source") or r.get("origin") or "").strip().lower()
        if tr == "primary" or st == "similar_to":
            hp = True
        else:
            hs = True
    return hp, hs


def _stage3_finalize_merged_candidate(obj: Dict[str, Any], representative_rec: Any) -> None:
    """
    单 tid 多来源 → 一条 merged record：证据聚合 + legacy compatibility mirror（derived，非 winner 单条语义）。
    representative 仅补非语义字段；primary_bucket / term_role / can_expand 等一律由聚合推导。
    """
    records = list(obj.get("records") or [])
    obj["source_evidence_list"] = [_stage3_merge_source_evidence_slice(r) for r in records]
    obj["stage2_local_meta_list"] = [dict(r.get("stage2_local_meta") or {}) for r in records]

    pb_summary = _stage3_merge_primary_bucket_summary(records)
    derived_pb_lower = (pb_summary.get("derived_primary_bucket") or "").strip().lower()
    anchor_sup = _stage3_merge_anchor_support_summary(records)
    prov = _stage3_merge_provenance_summary(records)
    term_role_dist = _stage3_merge_role_distribution(records)
    dbg = _stage3_merge_debug_summary(records, anchor_sup)

    role_ia_set = sorted(
        {
            (r.get("role_in_anchor") or "").strip().lower()
            for r in records
            if (r.get("role_in_anchor") or "").strip()
        }
    )
    anchor_role_dist_flat: Dict[str, List[str]] = {}
    for r in records:
        pa = (r.get("parent_anchor") or "").strip()
        ra = (r.get("role_in_anchor") or "").strip().lower()
        if not pa or not ra:
            continue
        anchor_role_dist_flat.setdefault(pa, []).append(ra)
    for pa in list(anchor_role_dist_flat.keys()):
        anchor_role_dist_flat[pa] = sorted(set(anchor_role_dist_flat[pa]))

    can_expand = any(
        bool(r.get("can_expand") or r.get("can_expand_local") or r.get("can_expand_from_2a")) for r in records
    )
    can_expand_from_2a = any(bool(r.get("can_expand_from_2a")) for r in records)
    can_expand_local = any(bool(r.get("can_expand_local")) for r in records)
    mainline_candidate = any(bool(r.get("mainline_candidate")) for r in records)
    fallback_primary = bool(obj.get("fallback_primary_all", False))

    derived_term_role = _stage3_merge_derived_term_role(records, term_role_dist)
    mh = sum(1 for r in records if (r.get("role_in_anchor") or "").strip().lower() == "mainline")
    sh = sum(1 for r in records if (r.get("role_in_anchor") or "").strip().lower() == "side")
    if mh > 0:
        derived_role_in_anchor = "mainline"
    elif sh > 0:
        derived_role_in_anchor = "side"
    else:
        derived_role_in_anchor = ""

    hp, hs = _stage3_merge_has_primary_support_flags(records)

    if representative_rec:
        for k, v in representative_rec.items():
            if k in _STAGE3_MERGE_NEVER_FROM_REPRESENTATIVE:
                continue
            obj[k] = v

    obj["primary_bucket_summary"] = pb_summary
    obj["anchor_support_summary"] = anchor_sup
    obj["provenance_summary"] = prov
    obj["merge_debug_summary"] = dbg
    obj["term_role_distribution"] = term_role_dist
    obj["role_in_anchor_set"] = role_ia_set
    obj["anchor_role_distribution"] = dict(sorted(anchor_role_dist_flat.items()))

    obj["primary_bucket"] = _stage3_merge_pick_display_primary_bucket(records, derived_pb_lower)
    obj["primary_reason"] = _stage3_merge_pick_primary_reason(records, derived_pb_lower)
    obj["term_role"] = derived_term_role
    obj["role_in_anchor"] = derived_role_in_anchor
    obj["can_expand"] = can_expand
    obj["can_expand_from_2a"] = can_expand_from_2a
    obj["can_expand_local"] = can_expand_local
    obj["mainline_candidate"] = mainline_candidate
    obj["fallback_primary"] = fallback_primary
    obj["has_primary_role"] = hp
    obj["has_support_role"] = hs
    obj["stage3_merge_source"] = "evidence_aggregate_v1"
    # legacy 单字段：优先与 derived bucket 对齐的一条 meta，非 seed 冠军；完整证据见 stage2_local_meta_list
    legacy_meta: Dict[str, Any] = {}
    for r in records:
        if _stage3_rec_bucket_norm(r) == derived_pb_lower:
            m = r.get("stage2_local_meta")
            if isinstance(m, dict) and m:
                legacy_meta = dict(m)
                break
    if not legacy_meta:
        for r in records:
            m = r.get("stage2_local_meta")
            if isinstance(m, dict) and m:
                legacy_meta = dict(m)
                break
    obj["stage2_local_meta"] = legacy_meta
    obj.pop("fallback_primary_all", None)


def _print_stage3_duplicate_merge_audit(merged: List[Dict[str, Any]]) -> None:
    """多锚 tid：raw 来源分布 vs 聚合摘要 vs legacy derived mirrors（非单胜者叙事）。"""
    if not STAGE3_DUPLICATE_MERGE_AUDIT or not merged:
        return
    lines_printed = 0
    for obj in merged:
        n_anch = int(obj.get("anchor_count") or 0)
        if n_anch < 2:
            continue
        term = (obj.get("term") or "").strip()
        if lines_printed == 0:
            print("\n" + "-" * 80)
            print("[Stage3 duplicate merge audit] anchor_count>=2：来源分布 / 聚合摘要 / legacy mirror（derived）")
            print("-" * 80)
        lines_printed += 1
        recs = obj.get("records") or []
        raw_pb = [_stage3_rec_bucket_norm(r) or (r.get("primary_bucket") or "") for r in recs]
        raw_roles = [((r.get("role_in_anchor") or "")) for r in recs]
        raw_ce = [
            bool(r.get("can_expand") or r.get("can_expand_local") or r.get("can_expand_from_2a"))
            for r in recs
        ]
        pbs = obj.get("primary_bucket_summary") or {}
        adb = obj.get("anchor_support_summary") or {}
        mdbg = obj.get("merge_debug_summary") or {}
        prov = obj.get("provenance_summary") or {}
        print(f"term={term!r} tid={obj.get('tid')} records={len(recs)}")
        print(f"  parent_anchors={obj.get('parent_anchors')!r}")
        print(f"  primary_buckets_raw(norm/meta)={raw_pb!r}")
        print(f"  role_in_anchor_raw={raw_roles!r}")
        print(f"  can_expand_raw(any local)={raw_ce!r}")
        print(
            f"  bucket_distribution={pbs.get('bucket_distribution')!r} "
            f"derived_primary_bucket={pbs.get('derived_primary_bucket')!r}"
        )
        print(
            f"  anchor_roles={adb.get('anchor_roles')!r} "
            f"mainline_anchor_count={adb.get('mainline_anchor_count')} "
            f"side_anchor_count={adb.get('side_anchor_count')} "
            f"expandable_anchor_count={adb.get('expandable_anchor_count')}"
        )
        print(
            f"  provenance source_type_counts={prov.get('source_type_counts')!r} "
            f"multi_source={prov.get('multi_source')}"
        )
        print(
            f"  term_role_distribution={obj.get('term_role_distribution')!r} "
            f"role_in_anchor_set={obj.get('role_in_anchor_set')!r}"
        )
        print(
            f"  merge_conflicts: bucket={mdbg.get('primary_bucket_conflict')} "
            f"term_role={mdbg.get('term_role_conflict')} "
            f"ria={mdbg.get('role_in_anchor_conflict')} "
            f"mainline_side={mdbg.get('mainline_side_conflict')} "
            f"cross_anchor_side_only={mdbg.get('cross_anchor_side_only')}"
        )
        print(
            f"  legacy_mirror primary_bucket={obj.get('primary_bucket')!r} "
            f"can_expand={bool(obj.get('can_expand'))} "
            f"role_in_anchor={obj.get('role_in_anchor')!r} "
            f"term_role={obj.get('term_role')!r}"
        )
        print(
            f"  merge_source={obj.get('stage3_merge_source')!r} "
            f"fallback_primary={bool(obj.get('fallback_primary'))}"
        )
    if lines_printed:
        print("-" * 80 + "\n")


def _merge_stage3_duplicates(raw_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按 tid 聚合同一学术 candidate 的多条 Stage2 记录 → 单条 merged candidate（多锚证据聚合）。

    TODO(Stage3 merge)：由 winner-snapshot 迁移为 evidence-aggregate。
    - `source_evidence_list` / `stage2_local_meta_list` / `*_summary` 为后续 global coherence rerank 的主材料。
    - 顶层 `primary_bucket` / `term_role` / `can_expand` / `role_in_anchor` 等为 **legacy compatibility mirror**，
      一律由组内全体记录 **推导**，不再由单条 seed 冠军覆盖语义。
    - 非语义标量（如 src_vids）仍可从 seed 分最高的 representative 行补齐（见 _STAGE3_MERGE_NEVER_FROM_REPRESENTATIVE）。

    合并后主链仍消费：`anchor_count`, `evidence_count`, `family_keys`, `source_types`, `parent_anchors`,
    `parent_primaries`, `mainline_hits`, `side_hits`, `best_*` 及上述 mirror 字段。
    """
    bucket: Dict[int, Dict[str, Any]] = {}
    for rec in raw_candidates:
        tid = rec.get("tid") or rec.get("vid")
        if tid is None:
            continue
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            continue
        obj = bucket.setdefault(
            tid,
            {
                "tid": tid,
                "term": rec.get("term") or "",
                "records": [],
                "best_identity_score": 0.0,
                "best_seed_score": 0.0,
                "best_anchor_identity": 0.0,
                "best_jd_align": 0.0,
                "best_context_continuity": 0.0,
                "parent_anchors": set(),
                "parent_primaries": set(),
                "source_types": set(),
                "term_roles": set(),
                "family_keys": set(),
                "mainline_hits": 0,
                "side_hits": 0,
                "fallback_primary_all": True,
                "retain_mode": "normal",
                "polysemy_risk": 0.0,
                "object_like_risk": 0.0,
                "generic_risk": 0.0,
                "_merge_best_seed_sc": -1.0,
                "_merge_best_seed_rec": None,
            },
        )
        obj["records"].append(rec)
        ident = float(rec.get("identity_score") or rec.get("sim_score") or 0.0)
        obj["best_identity_score"] = max(obj["best_identity_score"], ident)
        obj["best_anchor_identity"] = max(obj["best_anchor_identity"], ident)
        seed_sc = float(rec.get("score") or 0.0)
        obj["best_seed_score"] = max(obj["best_seed_score"], seed_sc)
        obj["best_jd_align"] = max(
            obj["best_jd_align"],
            float(rec.get("jd_candidate_alignment") or rec.get("jd_align") or 0.0),
        )
        obj["best_context_continuity"] = max(
            obj["best_context_continuity"],
            float(rec.get("context_continuity") or 0.0),
        )
        obj["polysemy_risk"] = max(obj.get("polysemy_risk", 0), float(rec.get("polysemy_risk") or 0))
        obj["object_like_risk"] = max(obj.get("object_like_risk", 0), float(rec.get("object_like_risk") or 0))
        obj["generic_risk"] = max(obj.get("generic_risk", 0), float(rec.get("generic_risk") or 0))
        pafs = float(rec.get("parent_anchor_final_score") or 0.0)
        if pafs > float(obj.get("best_parent_anchor_final_score") or 0.0):
            obj["best_parent_anchor_final_score"] = pafs
        rk_pa = int(rec.get("parent_anchor_step2_rank") or 0)
        if rk_pa > 0:
            prev_rk = obj.get("best_parent_anchor_step2_rank")
            if prev_rk is None or rk_pa < int(prev_rk):
                obj["best_parent_anchor_step2_rank"] = rk_pa
        pa = (rec.get("parent_anchor") or "").strip()
        pp = (rec.get("parent_primary") or "").strip()
        st = (rec.get("source_type") or rec.get("source") or rec.get("origin") or "").strip()
        tr = (rec.get("term_role") or "").strip()
        role_anchor = (rec.get("role_in_anchor") or "").strip().lower()
        if pa:
            obj["parent_anchors"].add(pa)
        if pp:
            obj["parent_primaries"].add(pp)
        if st:
            obj["source_types"].add(st)
        if tr:
            obj["term_roles"].add(tr)
        fk = rec.get("family_key")
        if fk:
            obj["family_keys"].add(str(fk))
        if role_anchor == "mainline":
            obj["mainline_hits"] = obj.get("mainline_hits", 0) + 1
        elif role_anchor:
            obj["side_hits"] = obj.get("side_hits", 0) + 1
        rm = (rec.get("retain_mode") or "normal").strip().lower()
        if rm == "normal":
            obj["retain_mode"] = "normal"
        obj["fallback_primary_all"] = bool(obj.get("fallback_primary_all", True)) and bool(
            rec.get("fallback_primary", False)
        )

        cur_pb = _stage3_rec_bucket_norm(rec)
        prev_sc = float(obj.get("_merge_best_seed_sc", -1.0))
        if seed_sc > prev_sc:
            obj["_merge_best_seed_sc"] = seed_sc
            obj["_merge_best_seed_rec"] = rec
        elif seed_sc == prev_sc and seed_sc >= 0.0:
            old = obj.get("_merge_best_seed_rec") or {}
            old_pb = _stage3_rec_bucket_norm(old)
            if STAGE3_PRIMARY_BUCKET_MERGE_PRIORITY.get(cur_pb, 0) > STAGE3_PRIMARY_BUCKET_MERGE_PRIORITY.get(
                old_pb, 0
            ):
                obj["_merge_best_seed_rec"] = rec

    merged: List[Dict[str, Any]] = []
    for tid, obj in bucket.items():
        best_rec = obj.pop("_merge_best_seed_rec", None)
        obj.pop("_merge_best_seed_sc", None)
        obj["parent_anchors"] = sorted(obj["parent_anchors"])
        obj["parent_primaries"] = sorted(obj["parent_primaries"])
        obj["source_types"] = sorted(obj["source_types"])
        obj["term_roles"] = sorted(obj["term_roles"])
        obj["family_keys"] = sorted(obj.get("family_keys") or [])
        obj["anchor_count"] = len(obj["parent_anchors"])
        obj["evidence_count"] = len(obj["records"])
        obj["best_stage2_score"] = obj.get("best_stage2_score") or obj.get("best_seed_score") or 0.0
        _stage3_finalize_merged_candidate(obj, best_rec)
        merged.append(obj)
    merged.sort(key=lambda x: float(x.get("best_seed_score") or 0.0), reverse=True)
    _print_stage3_duplicate_merge_audit(merged)
    return merged


def _classify_stage3_entry_groups(terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    候选分层：trusted_primary / secondary_primary / support_expansion。
    trusted_primary 不硬杀，只打 risk_flag；support_expansion 才重点审查。
    """
    for rec in terms:
        term_role = (rec.get("term_role") or "").strip().lower()
        if isinstance(rec.get("term_roles"), (list, set)):
            term_roles_set = set(rec["term_roles"]) if rec["term_roles"] else set()
            if "primary" in term_roles_set:
                term_role = "primary"
        role_in_anchor = (rec.get("role_in_anchor") or "").strip().lower()
        can_expand = bool(rec.get("can_expand"))
        retain_mode = (rec.get("retain_mode") or "normal").strip().lower()
        if (
            term_role == "primary"
            and role_in_anchor == "mainline"
            and can_expand
        ):
            rec["stage3_entry_group"] = "trusted_primary"
        elif term_role == "primary":
            rec["stage3_entry_group"] = "secondary_primary"
        else:
            rec["stage3_entry_group"] = "support_expansion"
    return terms


def _stage3_guardrail_tid_ok(term: Dict[str, Any]) -> bool:
    tid = term.get("tid") if term.get("tid") is not None else term.get("vid")
    if tid is None:
        return False
    try:
        int(tid)
    except (TypeError, ValueError):
        return False
    return True


def _stage3_collect_guardrail_soft_flags(term: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    结构风险软标记 + 说明（不触发 hard_drop）：供 GC / risk_penalty / 审计。
    原 admission 中的 secondary/support 弱证据 veto 已降级为 soft / notes。
    """
    soft: List[str] = []
    notes: List[str] = []
    mdbg = term.get("merge_debug_summary") or {}
    prov = term.get("provenance_summary") or {}
    pbs = term.get("primary_bucket_summary") or {}

    if _stage3_is_conditioned_only(term):
        soft.append("conditioned_only")
    if mdbg.get("cross_anchor_side_only"):
        soft.append("cross_anchor_side_only")
    if mdbg.get("primary_bucket_conflict"):
        soft.append("merge_primary_bucket_conflict")
    if mdbg.get("term_role_conflict"):
        soft.append("merge_term_role_conflict")
    if mdbg.get("role_in_anchor_conflict"):
        soft.append("merge_role_in_anchor_conflict")
    if mdbg.get("mainline_side_conflict"):
        soft.append("merge_mainline_side_conflict")
    if prov.get("single_path_only"):
        soft.append("provenance_single_path_only")
    if prov.get("single_path_multi_anchor_weak_signal"):
        soft.append("single_path_multi_anchor_weak_signal")
    if prov.get("multi_source") is False and int(prov.get("distinct_source_type_count") or 0) <= 1:
        soft.append("provenance_narrow")

    derived = (pbs.get("derived_primary_bucket") or "").strip().lower()
    if derived in ("risky_keep", "reject"):
        soft.append("weak_derived_bucket_prior")

    group = term.get("stage3_entry_group") or "support_expansion"
    anchor_id = float(term.get("best_anchor_identity") or term.get("best_identity_score") or 0.0)
    jd_align = float(term.get("best_jd_align") or 0.0)
    ctx = float(term.get("best_context_continuity") or 0.0)
    obj = float(term.get("object_like_risk") or 0.0)
    generic = float(term.get("generic_risk") or 0.0)

    if group == "secondary_primary" and anchor_id < 0.16 and jd_align < 0.72 and ctx < 0.48:
        soft.append("legacy_secondary_weak_triple_low")
        notes.append("TODO(Stage3 guardrail): was secondary_too_weak hard_drop; now GC/risk_penalty")
    if group == "support_expansion" and anchor_id < 0.14 and ctx < 0.50:
        soft.append("legacy_support_anchor_ctx_weak")
        notes.append("TODO(Stage3 guardrail): was weak_evidence_noise hard_drop")
    if group == "support_expansion" and obj > 0.70 and generic > 0.60:
        soft.append("legacy_high_object_generic_combo")
        notes.append("TODO(Stage3 guardrail): was object_generic_noise hard_drop")

    return soft, notes


def _print_stage3_guardrail_reject_audit(rows: List[Dict[str, Any]]) -> None:
    if not STAGE3_GUARDRAIL_REJECT_AUDIT or not rows:
        return
    print("\n" + "-" * 80)
    print("[Stage3 guardrail reject] 仅底线硬拒（broken / 缺字段 / 空壳）")
    print("-" * 80)
    for r in rows[:40]:
        print(
            f"  tid={r.get('tid')} term={r.get('term')!r} reason={r.get('guardrail_reason')!r} "
            f"flags={r.get('guardrail_flags')!r}"
        )
    if len(rows) > 40:
        print(f"  ... 共 {len(rows)} 条")
    print("-" * 80 + "\n")


def _print_stage3_guardrail_soft_audit(samples: List[Dict[str, Any]]) -> None:
    if not (STAGE3_GUARDRAIL_SOFT_AUDIT and STAGE3_AUDIT_DEBUG) or not samples:
        return
    print("\n" + "-" * 80)
    print("[Stage3 guardrail soft] 已准入但带结构风险标记（由 GC / bucket / paper 后续处理）")
    print("-" * 80)
    for rec in samples[: int(STAGE3_GUARDRAIL_SOFT_AUDIT_MAX)]:
        term = (rec.get("term") or "")[:28]
        print(
            f"  tid={rec.get('tid')} term={term!r} soft={rec.get('stage3_guardrail_soft_flags')!r} "
            f"notes_n={len(rec.get('stage3_guardrail_notes') or [])}"
        )
    if len(samples) > int(STAGE3_GUARDRAIL_SOFT_AUDIT_MAX):
        print(f"  ... 省略 {len(samples) - int(STAGE3_GUARDRAIL_SOFT_AUDIT_MAX)} 条（仅展示前 {STAGE3_GUARDRAIL_SOFT_AUDIT_MAX}）")
    print("-" * 80 + "\n")


def _check_stage3_admission(
    term: Dict[str, Any],
    jd_profile: Any,
    active_domains: Any,
) -> Dict[str, Any]:
    """
    Stage3 **guardrail**：仅挡明显坏数据 / 结构不成立、无法进入主打分的记录。

    不再基于 primary_bucket / role / conditioned_only / cross-anchor side-only / 未来 paper 价值做 hard_drop；
    此类信号写入 `stage3_guardrail_soft_flags` / `stage3_guardrail_notes`，由 global coherence 与 post-hoc 角色处理。

    返回兼容字段：hard_drop, reason, risk_flags, admission_strength；
    扩展：is_admitted, guardrail_reason, guardrail_flags, guardrail_soft_flags, notes。

    jd_profile / active_domains 保留签名，当前 guardrail 不使用（避免域级 veto 越权）。
    """
    _ = jd_profile
    _ = active_domains

    def _pack(
        hard_drop: bool,
        reason: str,
        g_flags: List[str],
        risk_flags: List[str],
        soft: List[str],
        notes: List[str],
        strength: str,
    ) -> Dict[str, Any]:
        return {
            "hard_drop": hard_drop,
            "reason": reason,
            "risk_flags": risk_flags,
            "admission_strength": strength,
            "is_admitted": not hard_drop,
            "guardrail_reason": reason,
            "guardrail_flags": list(g_flags),
            "guardrail_soft_flags": list(soft),
            "notes": list(notes),
        }

    if not isinstance(term, dict):
        return _pack(True, "not_a_dict", ["not_a_dict"], [], [], [], "rejected")

    risk_flags: List[str] = []
    poly = float(term.get("polysemy_risk") or 0.0)
    obj = float(term.get("object_like_risk") or 0.0)
    generic = float(term.get("generic_risk") or 0.0)
    if poly > 0.75:
        risk_flags.append("high_polysemy")
    if obj > 0.55:
        risk_flags.append("object_like")
    if generic > 0.55:
        risk_flags.append("too_generic")
    term["risk_flags"] = risk_flags

    group = term.get("stage3_entry_group") or "support_expansion"
    strength_map = {
        "trusted_primary": "trusted",
        "secondary_primary": "secondary",
        "support_expansion": "support",
    }
    admission_strength = strength_map.get(group, "support")

    soft, gr_notes = _stage3_collect_guardrail_soft_flags(term)

    # --- A 类：底线硬拒 ---
    if not _stage3_guardrail_tid_ok(term):
        return _pack(True, "missing_or_invalid_tid", ["missing_or_invalid_tid"], risk_flags, soft, gr_notes, "rejected")

    term_str = str(term.get("term") or "").strip()
    if not term_str:
        return _pack(True, "empty_term", ["empty_term"], risk_flags, soft, gr_notes, "rejected")

    recs = term.get("records") or []
    evl = term.get("source_evidence_list") or []
    ev_c = int(term.get("evidence_count") or 0)
    if len(recs) == 0 and len(evl) == 0 and ev_c <= 0:
        return _pack(True, "empty_evidence_shell", ["empty_evidence_shell"], risk_flags, soft, gr_notes, "rejected")

    term["stage3_guardrail_soft_flags"] = soft
    term["stage3_guardrail_notes"] = gr_notes
    term["stage3_admission_mode"] = "guardrail_v1"

    return _pack(False, "", [], risk_flags, soft, gr_notes, admission_strength)


# 兼容旧命名：语义已收敛为 guardrail-only
def _stage3_guardrail_filter(
    term: Dict[str, Any],
    jd_profile: Any,
    active_domains: Any,
) -> Dict[str, Any]:
    """与 `_check_stage3_admission` 同义，便于搜索「guardrail」。"""
    return _check_stage3_admission(term, jd_profile, active_domains)


def _compute_stage3_global_consensus(
    recall,
    candidates: List[Dict[str, Any]],
) -> None:
    """
    只计算非 cluster 的全局信号：cross_anchor_evidence（参与最终分）、semantic_drift_risk（仅 debug）。
    不再写入任何 cluster 相关字段。
    """
    if not candidates:
        return
    all_anchors: Set[str] = set()
    for rec in candidates:
        for a in rec.get("parent_anchors") or []:
            if a:
                all_anchors.add(a)
    n_anchors = max(1, len(all_anchors))
    for rec in candidates:
        anchor_count = int(rec.get("anchor_count") or 0)
        evidence_count = int(rec.get("evidence_count") or 0)
        outside = float(rec.get("outside_subfield_mass") or 0.0)
        topic_fit = rec.get("topic_fit")
        if topic_fit is None:
            topic_fit = rec.get("subfield_fit")
        if topic_fit is None:
            topic_fit = rec.get("field_fit")
        if topic_fit is None:
            topic_fit = rec.get("domain_fit")
        topic_fit = float(topic_fit or 0.5)
        anchor_part = min(1.0, anchor_count / max(2.0, n_anchors))
        evidence_part = min(1.0, evidence_count / 3.0)
        cross_anchor = 0.90 + 0.12 * anchor_part + 0.08 * evidence_part
        cross_anchor = min(1.10, max(0.90, cross_anchor))
        drift = (
            0.45 * min(1.0, outside)
            + 0.25 * (1.0 - topic_fit)
            + 0.30 * (1.0 / max(1.0, anchor_count))
        )
        drift = min(1.0, max(0.0, drift))
        rec["cross_anchor_evidence"] = cross_anchor
        rec["semantic_drift_risk"] = drift


def _build_family_buckets(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按 family_key 分桶。"""
    family_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        fk = rec.get("family_key") or build_family_key(rec)
        rec["family_key"] = fk
        family_buckets[fk].append(rec)
    return family_buckets


def _compute_family_centrality(rec: Dict[str, Any], family_buckets: Dict[str, List[Dict[str, Any]]]) -> float:
    """
    衡量 term 在 family 内是否为「中心项」：has_primary_role、seed score 靠前、锚点与证据数。
    """
    fk = rec.get("family_key") or build_family_key(rec)
    members = family_buckets.get(fk) or [rec]
    scores = [float(m.get("best_seed_score") or m.get("score") or 0.0) for m in members]
    max_score = max(scores) if scores else 1.0
    cur_score = float(rec.get("best_seed_score") or rec.get("score") or 0.0)
    rank_score = cur_score / max(1e-6, max_score)
    role_bonus = 1.0 if rec.get("has_primary_role") else 0.85
    anchor_bonus = min(1.0, 0.65 + 0.10 * int(rec.get("anchor_count") or 0))
    evidence_bonus = min(1.0, 0.70 + 0.10 * int(rec.get("evidence_count") or 0))
    centrality = (
        0.45 * rank_score
        + 0.25 * role_bonus
        + 0.15 * anchor_bonus
        + 0.15 * evidence_bonus
    )
    return min(1.0, max(0.0, centrality))


def _apply_family_role_constraints(survivors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    防止 support 系统性压过 primary：每 family 至少保证一个 primary 可见；
    support 只有显著优于 primary 才允许压过，否则只能作为补充。
    """
    family_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in survivors:
        family_buckets[rec.get("family_key", "")].append(rec)
    for fk, members in family_buckets.items():
        primaries = [m for m in members if (m.get("retrieval_role") or "").lower() == "paper_primary"]
        supports = [m for m in members if (m.get("retrieval_role") or "").lower() == "paper_support"]
        best_primary = None
        if primaries:
            best_primary = max(primaries, key=lambda x: float(x.get("final_score") or 0.0))
            best_primary["final_score"] = float(best_primary.get("final_score") or 0.0) * 1.05
            best_primary["primary_visibility_boost"] = True
        if best_primary and supports:
            p_score = float(best_primary.get("final_score") or 0.0)
            for s in supports:
                s_score = float(s.get("final_score") or 0.0)
                if s_score < p_score * 1.10:
                    s["final_score"] = min(s_score, p_score * 0.98)
                    s["support_over_primary_suppressed"] = True
    return survivors


def _collect_risky_reasons(rec: Dict[str, Any]) -> List[str]:
    """Risky 理由：弱 family、高 drift+低 ptc、弱 topic 尾部；判定更严，避免正常主干词被误标。"""
    reasons: List[str] = []
    final_score = float(rec.get("final_score") or 0.0)
    drift = float(rec.get("semantic_drift_risk") or 0.0)
    ex = rec.get("stage3_explain") or {}
    ptc = float(ex.get("path_topic_consistency") or 0.0)
    fc = float(rec.get("family_centrality") or 0.0)
    if fc < 0.30:
        reasons.append("weak_family_centrality")
    if drift > 0.75 and ptc < 0.30:
        reasons.append("high_drift_risk")
    if ptc < 0.50 and final_score < 0.58:
        reasons.append("weak_topic_fit_tail")
    return reasons


def _collect_stage3_bucket_reason_flags(rec: Dict[str, Any]) -> List[str]:
    """判定链路：为何落入 core/support/risky，供 [Stage3 bucket reason] 打印。"""
    flags: List[str] = []
    mainline_hits = int(rec.get("mainline_hits") or 0)
    anchor_count = int(rec.get("anchor_count") or 0)
    can_expand = bool(rec.get("can_expand", False))
    identity_factor = float((rec.get("stage3_explain") or {}).get("identity_factor", 1.0) or 1.0)
    obj = float(rec.get("object_like_risk") or 0.0)
    generic = float(rec.get("generic_risk") or 0.0)
    primary_bucket = (rec.get("primary_bucket") or "").strip().lower()
    primary_reason = (rec.get("primary_reason") or "").strip().lower()
    fallback_primary = bool(rec.get("fallback_primary", False))
    mainline_candidate = bool(rec.get("mainline_candidate", False))
    can_expand_local = bool(rec.get("can_expand_local", can_expand))

    is_fallback_primary = (
        fallback_primary
        or primary_bucket == "primary_fallback_keep_no_expand"
        or primary_reason == "anchor_core_fallback"
    )

    is_locked_mainline = (
        primary_bucket in ("primary_keep_no_expand", "primary_support_keep")
        or primary_reason == "usable_mainline_no_expand"
        or (mainline_candidate and (not can_expand_local))
    )

    has_mainline_support = (mainline_hits >= 1) or can_expand or is_locked_mainline

    if not has_mainline_support and anchor_count > 0:
        flags.append("no_mainline_support")
    if (not has_mainline_support) and (not is_fallback_primary):
        flags.append("only_weak_keep_sources")
    if is_locked_mainline:
        flags.append("locked_mainline_no_expand")
    if is_fallback_primary:
        flags.append("fallback_primary")
    if anchor_count >= 2 and mainline_hits == 0:
        flags.append("cross_anchor_but_side_only")
    if _stage3_is_conditioned_only(rec):
        flags.append("conditioned_only")
    if identity_factor < 0.60:
        flags.append("identity_low_family")
    if obj >= 0.50:
        flags.append("object_like")
    if generic >= 0.50:
        flags.append("generic_like")
    if rec.get("stage3_support_demote_reason") == "single_anchor_expand_branch":
        flags.append("single_anchor_expand_branch_demote")
    return flags


def _bucket_stage3_terms(survivors: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    分为 core_terms、support_terms、risky_terms。
    若已有 stage3_bucket 则直接按桶分组；否则回退到原逻辑（强主干/结构型 core、support、risky）。
    """
    core_terms: List[Dict[str, Any]] = []
    support_terms: List[Dict[str, Any]] = []
    risky_terms: List[Dict[str, Any]] = []
    for rec in survivors:
        bucket = (rec.get("stage3_bucket") or "").strip().lower()
        if bucket == "core":
            core_terms.append(rec)
            continue
        if bucket == "support":
            support_terms.append(rec)
            continue
        if bucket == "risky":
            risky_terms.append(rec)
            continue
        ex = rec.get("stage3_explain") or {}
        final_score = float(rec.get("final_score") or 0.0)
        ptc = float(ex.get("path_topic_consistency") or 0.0)
        cross = float(ex.get("cross_anchor_factor") or rec.get("cross_anchor_evidence") or 1.0)
        role = (rec.get("retrieval_role") or "").lower()
        reasons = rec.get("risk_reasons") or []
        is_primary = role == "paper_primary"
        strong_primary_core = is_primary and final_score >= 0.66
        structured_primary_core = (
            is_primary and final_score >= 0.62 and ptc >= 0.55 and cross >= 0.94
        )
        if strong_primary_core or structured_primary_core:
            core_terms.append(rec)
            continue
        if final_score >= 0.56 and "high_drift_risk" not in reasons:
            support_terms.append(rec)
            continue
        risky_terms.append(rec)
    return core_terms, support_terms, risky_terms


def _debug_print_stage3_bucket_details(
    core_terms: List[Dict[str, Any]],
    support_terms: List[Dict[str, Any]],
    risky_terms: List[Dict[str, Any]],
    recall: Any,
) -> None:
    """打印 bucket 判定依据：term | bucket | final | ptc | cross | reasons。"""
    debug_print(2, "[Stage3 Bucket Details] term | bucket | final | ptc | cross | reasons", recall)
    for bucket_name, items in [
        ("core", core_terms),
        ("support", support_terms),
        ("risky", risky_terms),
    ]:
        for rec in items:
            explain = rec.get("stage3_explain") or {}
            final_score = float(rec.get("final_score") or 0.0)
            ptc = float(explain.get("path_topic_consistency") or 0.0)
            cross = float(
                explain.get("cross_anchor_factor")
                or rec.get("cross_anchor_evidence")
                or 1.0
            )
            reasons = rec.get("risk_reasons") or []
            term = (rec.get("term") or "")[:32]
            debug_print(
                2,
                f"  {term!r} | {bucket_name} | {final_score:.4f} | {ptc:.4f} | {cross:.4f} | {reasons}",
                recall,
            )


def _stage3_single_anchor_expand_branch_demote(term: Dict[str, Any]) -> bool:
    """
    单锚、仅 1 次 mainline 命中且 Stage2 可扩：若 family/topic 不在控制-规划主线中心，
    或 object/drift/generic/poly 结构偏支，则不应轻松进 support（压到 risky）。
    """
    anchor_count = int(term.get("anchor_count") or 0)
    mainline_hits = int(term.get("mainline_hits") or 0)
    can_expand = bool(term.get("can_expand", False))
    if anchor_count != 1 or mainline_hits < 1 or not can_expand:
        return False
    ex = term.get("stage3_explain") or {}
    fc = float(term.get("family_centrality") or ex.get("family_centrality") or 0.0)
    ptc = float(ex.get("path_topic_consistency") or 0.0)
    obj = float(term.get("object_like_risk") or 0.0)
    drift = float(term.get("semantic_drift_risk") or 0.0)
    generic = float(term.get("generic_risk") or 0.0)
    poly = float(term.get("polysemy_risk") or 0.0)
    if fc < STAGE3_SUPPORT_DEMOTE_FC_MAX and ptc < STAGE3_SUPPORT_DEMOTE_PTC_MAX:
        return True
    if obj >= STAGE3_SUPPORT_DEMOTE_OBJ_MIN:
        return True
    if drift >= STAGE3_SUPPORT_DEMOTE_DRIFT_MIN and ptc < STAGE3_SUPPORT_DEMOTE_DRIFT_PTC_MAX:
        return True
    if generic >= STAGE3_SUPPORT_DEMOTE_GENERIC_MIN:
        return True
    if poly >= STAGE3_SUPPORT_DEMOTE_POLY_MIN and ptc < STAGE3_SUPPORT_DEMOTE_POLY_PTC_MAX:
        return True
    return False


def _compute_stage3_risk_penalty(term: Dict[str, Any]) -> float:
    """
    风险仅降分不硬杀：trusted_primary 轻罚，support_expansion 重罚。
    """
    group = (term.get("stage3_entry_group") or "support_expansion")
    poly = float(term.get("polysemy_risk") or 0.0)
    obj = float(term.get("object_like_risk") or 0.0)
    generic = float(term.get("generic_risk") or 0.0)
    if group == "trusted_primary":
        penalty = 1.0 - 0.10 * poly - 0.06 * obj - 0.05 * generic
    elif group == "secondary_primary":
        penalty = 1.0 - 0.16 * poly - 0.08 * obj - 0.08 * generic
    else:
        penalty = 1.0 - 0.22 * poly - 0.14 * obj - 0.12 * generic
    return max(0.65, min(1.0, penalty))


def _assign_stage3_bucket(term: Dict[str, Any]) -> str:
    """
    Stage3 分桶：显式消费 Stage2A 五层 primary_bucket（expandable / support_seed / support_keep / risky_keep），
    与 conditioned_only、fallback 正交；support 子类强度不同（seed 可信 > keep 保留 > risky_keep 已默认 risky）。
    """
    term.pop("stage3_support_demote_reason", None)
    term.pop("stage3_core_cap_reason", None)
    term.pop("stage3_core_miss_reason", None)
    anchor_count = int(term.get("anchor_count") or 0)
    if anchor_count < 1:
        anchor_count = 1
    mainline_hits = int(term.get("mainline_hits") or 0)
    can_expand = bool(term.get("can_expand", False))
    ex = term.get("stage3_explain") or {}
    identity_factor = float(ex.get("identity_factor", 1.0) or 1.0)

    primary_bucket = (term.get("primary_bucket") or "").strip().lower()
    primary_reason = (term.get("primary_reason") or "").strip().lower()
    fallback_primary = bool(term.get("fallback_primary", False))
    mainline_candidate = bool(term.get("mainline_candidate", False))
    can_expand_local = bool(term.get("can_expand_local", can_expand))

    is_fallback_primary = (
        fallback_primary
        or primary_bucket == "primary_fallback_keep_no_expand"
        or primary_reason == "anchor_core_fallback"
    )

    st = _stage3_normalized_source_types(term)
    has_similar = "similar_to" in st
    has_family = "family_landing" in st
    has_conditioned = "conditioned_vec" in st
    conditioned_only = bool(has_conditioned and (not has_similar) and (not has_family))

    def _cap_core_if_conditioned_single_anchor(side: str) -> str:
        """
        单锚 + conditioned_only 不得挂 core：与 paper 硬门（conditioned_only_single_anchor_block）一致，
        避免 rerank 表里显示 core、 downstream 却被挡的口径割裂。
        """
        if side == "core" and conditioned_only and anchor_count <= 1:
            term["stage3_core_cap_reason"] = "conditioned_only_single_anchor"
            return "support"
        return side

    if is_fallback_primary:
        return "risky"

    if primary_bucket == "risky_keep":
        return "risky"

    # primary_expandable：core = 主线+可扩 ∧ (identity 硬杠 ∨ 非纯 conditioned + JD 对齐)；否则 support 并写 miss 原因供审计
    if primary_bucket == "primary_expandable":
        jd_align_core = float(
            term.get("best_jd_align")
            or term.get("jd_candidate_alignment")
            or term.get("jd_align")
            or ex.get("jd_align")
            or ex.get("jd_candidate_alignment")
            or 0.0
        )
        if mainline_hits >= 1 and can_expand:
            if identity_factor >= STAGE3_EXPANDABLE_CORE_IDENTITY_HARD:
                return _cap_core_if_conditioned_single_anchor("core")
            if (
                anchor_count >= 1
                and (not conditioned_only)
                and jd_align_core >= STAGE3_EXPANDABLE_CORE_JD_ALIGN_MIN
            ):
                return _cap_core_if_conditioned_single_anchor("core")
        if not (mainline_hits >= 1 and can_expand):
            term["stage3_core_miss_reason"] = "expandable_need_mainline_and_expand"
        else:
            if conditioned_only:
                term["stage3_core_miss_reason"] = "conditioned_only_no_identity_bypass"
            elif jd_align_core < STAGE3_EXPANDABLE_CORE_JD_ALIGN_MIN:
                term["stage3_core_miss_reason"] = "identity_low_and_jd_below_bypass"
            else:
                term["stage3_core_miss_reason"] = "identity_below_expandable_core_bar"
        return "support"

    # primary_support_seed：可信 support；证据不足 → risky
    if primary_bucket == "primary_support_seed":
        if mainline_hits >= 1 or can_expand:
            return "support"
        return "risky"

    # primary_support_keep / 旧 primary_keep_no_expand：支线保留 support，默认可弱于 seed
    if primary_bucket in ("primary_support_keep", "primary_keep_no_expand"):
        if conditioned_only and anchor_count <= 1:
            return "support" if (mainline_hits >= 1 or can_expand_local) else "risky"
        return "support"

    # ----- 无显式五层 bucket 时的回退链 -----
    is_locked_mainline = (
        primary_bucket in ("primary_keep_no_expand", "primary_support_keep")
        or primary_reason == "usable_mainline_no_expand"
        or (mainline_candidate and (not can_expand_local))
    )

    strong_mainline_support = mainline_hits >= 1 or can_expand
    has_mainline_support = strong_mainline_support or is_locked_mainline

    if conditioned_only and anchor_count <= 1:
        return "support" if has_mainline_support else "risky"
    if conditioned_only and anchor_count >= 2:
        return "support"

    if is_locked_mainline:
        return "support"

    core_ok = (
        mainline_hits >= 1
        and can_expand
        and identity_factor >= 0.95
    )
    if core_ok:
        return _cap_core_if_conditioned_single_anchor("core")

    bucket = "support" if has_mainline_support else "risky"

    if bucket == "support" and _stage3_single_anchor_expand_branch_demote(term):
        term["stage3_support_demote_reason"] = "single_anchor_expand_branch"
        return "risky"
    return bucket


def _is_primary_like(rec: Dict[str, Any]) -> bool:
    """primary-like 主落点：只做软惩罚，不走严格 topic gate。"""
    source_type = (rec.get("source_type") or rec.get("source") or rec.get("origin") or "").strip().lower()
    term_role = (rec.get("term_role") or "").strip().lower()
    return (
        bool(rec.get("has_primary_role"))
        or term_role == "primary"
        or source_type == "similar_to"
        or int(rec.get("anchor_count") or 0) >= 2
    )


def _debug_print_stage3_tables(
    layered_terms: List[Dict[str, Any]],
    dropped_with_reason: List[Dict[str, Any]],
    survivors: List[Dict[str, Any]],
    recall: Any,
) -> None:
    """Stage3 汇总：条数 + 前3样本，不打印完整表。"""
    stage3_debug = getattr(term_scoring, "STAGE3_DEBUG", False)
    label_trace = LABEL_PATH_TRACE or stage3_debug
    if not (label_trace or stage3_debug):
        return
    entry_sample = [rec.get("term") for rec in layered_terms[:3]]
    drop_sample = [rec.get("term") for rec in dropped_with_reason[:3]]
    surv_sample = [rec.get("term") for rec in survivors[:3]]
    debug_print(2, f"[Stage3] entry 共 {len(layered_terms)} 条 前3: {entry_sample} | dropped {len(dropped_with_reason)} 前3: {drop_sample} | survivors {len(survivors)} 前3: {surv_sample}", recall)


def _debug_print_stage3_input(raw_candidates: List[Dict[str, Any]]) -> None:
    """Stage3 输入：条数 + 前3样本，不打印全表。"""
    if not raw_candidates:
        return
    stage3_debug = getattr(term_scoring, "STAGE3_DEBUG", False)
    label_trace = LABEL_PATH_TRACE or stage3_debug
    if not (label_trace or stage3_debug):
        return
    sample = [(rec.get("term") or "", rec.get("source") or rec.get("origin") or "") for rec in raw_candidates[:3]]
    print(f"[stage3_input] 共 {len(raw_candidates)} 条 前3: {sample}")


def _write_term_maps(
    score_map: Dict[str, float],
    term_map: Dict[str, str],
    idf_map: Dict[str, float],
    term_role_map: Dict[str, str],
    term_source_map: Dict[str, str],
    parent_anchor_map: Dict[str, str],
    parent_primary_map: Dict[str, str],
    recall: Any,
    rec: Dict[str, Any],
    query_vector: Any = None,
) -> None:
    """将一条幸存词写入各 map 与 tag_purity_debug。"""
    tid_str = str(rec.get("tid", ""))
    if not tid_str or tid_str == "None":
        return
    score_map[tid_str] = float(rec["final_score"])
    term_map[tid_str] = rec.get("term") or ""
    term_role_map[tid_str] = rec.get("term_role") or "primary"
    term_source_map[tid_str] = rec.get("source") or rec.get("origin") or ""
    parent_anchor_map[tid_str] = rec.get("parent_anchor") or ""
    parent_primary_map[tid_str] = rec.get("parent_primary") or ""
    degree_w = int(rec.get("degree_w") or 0)
    total = float(getattr(recall, "total_work_count", 1e6) or 1e6)
    idf_map[tid_str] = term_scoring._smoothed_idf(
        degree_w,
        term_scoring._idf_backbone(total, int(rec.get("degree_w_expanded") or 0) or max(degree_w, 1)),
    )
    entry = {
        "tid": rec.get("tid"),
        "term": rec.get("term"),
        "term_role": rec.get("term_role"),
        "identity_score": rec.get("identity_score"),
        "quality_score": rec.get("quality_score"),
        "final_score": rec.get("final_score"),
        "degree_w": rec.get("degree_w"),
        "degree_w_expanded": rec.get("degree_w_expanded"),
        "source": rec.get("source") or rec.get("origin"),
        "domain_fit": rec.get("domain_fit"),
        "parent_anchor": rec.get("parent_anchor"),
        "parent_primary": rec.get("parent_primary"),
        "anchor_count": rec.get("anchor_count"),
        "evidence_count": rec.get("evidence_count"),
        "family_centrality": rec.get("family_centrality"),
        "path_topic_consistency": (rec.get("stage3_explain") or {}).get("path_topic_consistency"),
        "generic_penalty": (rec.get("stage3_explain") or {}).get("generic_penalty"),
        "cross_anchor_factor": (rec.get("stage3_explain") or {}).get("cross_anchor_factor"),
        "retrieval_role": rec.get("retrieval_role"),
    }
    if rec.get("stage3_explain"):
        entry.update(rec["stage3_explain"])
    debug_metrics = term_scoring.get_term_debug_metrics(recall, rec, query_vector)
    if debug_metrics:
        entry.update(debug_metrics)
    recall.debug_info.tag_purity_debug.append(entry)


def _write_term_maps_if_missing(
    score_map: Dict[str, float],
    term_map: Dict[str, str],
    idf_map: Dict[str, float],
    term_role_map: Dict[str, str],
    term_source_map: Dict[str, str],
    parent_anchor_map: Dict[str, str],
    parent_primary_map: Dict[str, str],
    recall: Any,
    rec: Dict[str, Any],
    query_vector: Any = None,
) -> None:
    """仅当 tid 尚未在 score_map 时写入各 map，避免 paper_terms 与 top_survivors 重复写入。"""
    tid_str = str(rec.get("tid", ""))
    if not tid_str or tid_str == "None" or tid_str in score_map:
        return
    _write_term_maps(
        score_map, term_map, idf_map, term_role_map, term_source_map,
        parent_anchor_map, parent_primary_map, recall, rec, query_vector,
    )


def _stage3_audit_filter_rows(rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """审计表行裁剪：若 STAGE3_DEBUG_FOCUS_TERMS 非空则只保留 term 命中集合的行。"""
    if not rows:
        return []
    if STAGE3_DEBUG_FOCUS_TERMS:
        rows = [r for r in rows if (r.get("term") or "").strip() in STAGE3_DEBUG_FOCUS_TERMS]
    return rows[:top_k]


def _print_stage3_final_adjust_audit(cands: List[Dict[str, Any]], top_k: int = 12) -> None:
    """
    [Stage3 final adjust audit]
    统一连续分模式下：pre_adj = identity×risk×role 链输出；post_adj = 连续分 final_score；
    risk_mlt / cross_bn 恒为 1.0（旧 bucket 乘子链已移除，仅保留列对齐旧日志脚本）。
    """
    if not STAGE3_AUDIT_DEBUG or not cands:
        return
    ranked = sorted(cands, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    ranked = _stage3_audit_filter_rows(ranked, top_k)
    if not ranked:
        print("\n" + "-" * 80 + "\n[Stage3 final adjust audit] (no rows after focus filter)\n" + "-" * 80)
        return

    print("\n" + "-" * 80)
    print("[Stage3 final adjust audit] (unified_continuous; risk×cross deprecated=1.0)")
    print("-" * 80)
    print(
        "term                         | bucket   | pre_adj  | risk_mlt | cross_bn | post_adj | "
        "anchor_count | mainline_hits | can_exp | cond_only | flags"
    )
    for rec in ranked:
        term = (rec.get("term") or "")[:28]
        bucket = (rec.get("stage3_bucket") or "")[:8]
        ex = rec.get("stage3_explain") or {}
        risk_mult = float(ex.get("mainline_risk_penalty") or 1.0)
        cross_bonus = float(ex.get("cross_anchor_score_bonus") or 1.0)
        post_adjust = float(rec.get("final_score") or 0.0)
        pre_adjust = rec.get("_stage3_pre_adjust_score")
        if pre_adjust is None:
            pre_adjust = post_adjust
        else:
            pre_adjust = float(pre_adjust or 0.0)
        anchor_count = int(rec.get("anchor_count") or 0)
        mainline_hits = int(rec.get("mainline_hits") or 0)
        can_expand = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
        conditioned_only = _stage3_is_conditioned_only(rec)
        flags = list(rec.get("bucket_reason_flags") or [])
        print(
            f"{term:28} | {bucket:8} | {pre_adjust:8.4f} | {risk_mult:8.4f} | {cross_bonus:8.4f} | {post_adjust:8.4f} | "
            f"{anchor_count:^12d} | {mainline_hits:^13d} | {str(can_expand):^7} | {str(conditioned_only):^9} | {flags}"
        )


def _print_stage3_cross_anchor_audit(cands: List[Dict[str, Any]], top_k: int = 40) -> None:
    """
    [Stage3 cross-anchor audit]
    只列 anchor_count>=2：看跨锚词拿到的 cross_anchor_score_bonus 与桶、主线证据是否一致。
    """
    if not STAGE3_AUDIT_DEBUG or not cands:
        return
    focus = [c for c in cands if int(c.get("anchor_count") or 0) >= 2]
    focus = sorted(
        focus,
        key=lambda x: (int(x.get("anchor_count") or 0), float(x.get("final_score") or 0.0)),
        reverse=True,
    )
    focus = _stage3_audit_filter_rows(focus, top_k)
    if not focus:
        print("\n" + "-" * 80 + "\n[Stage3 cross-anchor audit] (no multi-anchor rows / focus empty)\n" + "-" * 80)
        return

    print("\n" + "-" * 80)
    print("[Stage3 cross-anchor audit]")
    print("-" * 80)
    print(
        "term                     | anchor_count | anchors                  | evidence_cnt | "
        "mainline_hits | can_exp | bucket   | cross_bn   | flags"
    )
    for rec in focus:
        term = (rec.get("term") or "")[:24]
        anchor_count = int(rec.get("anchor_count") or 0)
        anchors = rec.get("parent_anchors") or []
        if isinstance(anchors, (list, tuple, set)):
            anchors_show = ",".join(str(a) for a in list(anchors)[:4])
        else:
            anchors_show = str(anchors)[:24]
        evidence_count = int(rec.get("evidence_count") or 0)
        mainline_hits = int(rec.get("mainline_hits") or 0)
        can_expand = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
        bucket = (rec.get("stage3_bucket") or "")[:8]
        ex = rec.get("stage3_explain") or {}
        cross_bonus = float(ex.get("cross_anchor_score_bonus") or 1.0)
        flags = list(rec.get("bucket_reason_flags") or [])
        print(
            f"{term:24} | {anchor_count:^12d} | {anchors_show[:24]:24} | {evidence_count:^12d} | "
            f"{mainline_hits:^13d} | {str(can_expand):^7} | {bucket:8} | {cross_bonus:10.4f} | {flags}"
        )


def _print_stage3_support_subtype_audit(cands: List[Dict[str, Any]], top_k: int = 24) -> None:
    """
    [Stage3 support subtype audit]
    bucket=support 时拆 primary_support_seed / primary_support_keep / conditioned_only 等，与 stage3_build_score_map 乘子对齐排查。
    """
    if not STAGE3_AUDIT_DEBUG or not cands:
        return
    focus = [
        c for c in cands
        if (c.get("stage3_bucket") or "").strip().lower() == "support"
    ]
    focus = sorted(focus, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    focus = _stage3_audit_filter_rows(focus, top_k)
    if not focus:
        print("\n" + "-" * 80 + "\n[Stage3 support subtype audit] (no support rows / focus empty)\n" + "-" * 80)
        return

    print("\n" + "-" * 80)
    print("[Stage3 support subtype audit]")
    print("-" * 80)
    print(
        "term                     | stage3_bucket | primary_bucket       | cond_only | anch | ml_hits | can_exp | final"
    )
    for rec in focus:
        term = (rec.get("term") or "")[:24]
        sb = (rec.get("stage3_bucket") or "")[:12]
        pb = (rec.get("primary_bucket") or "")[:20]
        co = _stage3_is_conditioned_only(rec)
        anch = int(rec.get("anchor_count") or 0)
        ml = int(rec.get("mainline_hits") or 0)
        ce = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
        fs = float(rec.get("final_score") or 0.0)
        print(
            f"{term:24} | {sb:13} | {pb:20} | {str(co):^9} | {anch:^4} | {ml:^7} | {str(ce):^7} | {fs:.4f}"
        )


def _print_stage3_support_risky_concise(cands: List[Dict[str, Any]], top_k: int = 20) -> None:
    """[Stage3 support/risky concise] 只列 support / risky，快速扫可疑项。"""
    if not STAGE3_AUDIT_DEBUG or not cands:
        return
    focus = [
        c for c in cands
        if (c.get("stage3_bucket") or "").strip().lower() in ("support", "risky")
    ]
    focus = sorted(focus, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    focus = _stage3_audit_filter_rows(focus, top_k)
    if not focus:
        print("\n" + "-" * 80 + "\n[Stage3 support/risky concise] (no rows / focus empty)\n" + "-" * 80)
        return

    print("\n" + "-" * 80)
    print("[Stage3 support/risky concise]")
    print("-" * 80)
    print(
        "term                     | bucket   | final_score | mainline_hits | can_exp | "
        "cond_only | weak_keep | xa>=2"
    )
    for rec in focus:
        term = (rec.get("term") or "")[:24]
        bucket = (rec.get("stage3_bucket") or "")[:8]
        final_score = float(rec.get("final_score") or 0.0)
        mainline_hits = int(rec.get("mainline_hits") or 0)
        can_expand = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
        conditioned_only = _stage3_is_conditioned_only(rec)
        reason_flags = set(rec.get("bucket_reason_flags") or [])
        weak_keep_only = "only_weak_keep_sources" in reason_flags
        cross_anchor = int(rec.get("anchor_count") or 0) >= 2
        print(
            f"{term:24} | {bucket:8} | {final_score:11.4f} | {mainline_hits:^13d} | {str(can_expand):^7} | "
            f"{str(conditioned_only):^9} | {str(weak_keep_only):^9} | {str(cross_anchor):^5}"
        )


def _print_stage3_core_miss_audit(cands: List[Dict[str, Any]], limit: int = 24) -> None:
    """[Stage3 core-miss audit]：2A 已标 primary_expandable 但未落 stage3 core，便于对照 identity/jd/cap。"""
    if not STAGE3_CORE_MISS_AUDIT or not cands:
        return
    rows: List[Dict[str, Any]] = []
    for rec in cands:
        pb = (rec.get("primary_bucket") or "").strip().lower()
        sb = (rec.get("stage3_bucket") or "").strip().lower()
        if pb == "primary_expandable" and sb != "core":
            rows.append(rec)
    if not rows:
        return
    rows.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    rows = rows[:limit]
    print("\n" + "-" * 80)
    print("[Stage3 core-miss audit] primary_expandable 但未 core（精调对照）")
    print("-" * 80)
    for rec in rows:
        ex = rec.get("stage3_explain") or {}
        ident = float(ex.get("identity_factor", 1.0) or 1.0)
        jd_al = float(
            rec.get("best_jd_align")
            or rec.get("jd_candidate_alignment")
            or rec.get("jd_align")
            or ex.get("jd_align")
            or 0.0
        )
        reason = (
            rec.get("stage3_core_miss_reason")
            or rec.get("stage3_core_cap_reason")
            or ""
        )
        co = _stage3_is_conditioned_only(rec)
        print(
            f"term={rec.get('term')!r} primary_bucket={rec.get('primary_bucket')!r} "
            f"stage3_bucket={rec.get('stage3_bucket')!r}"
        )
        print(
            f"  mainline_hits={int(rec.get('mainline_hits') or 0)} can_expand="
            f"{bool(rec.get('can_expand', False) or rec.get('can_expand_from_2a', False))} "
            f"anchor_count={int(rec.get('anchor_count') or 0)} conditioned_only={co}"
        )
        print(
            f"  identity_factor={ident:.3f} jd_align={jd_al:.3f} final_score="
            f"{float(rec.get('final_score') or 0.0):.4f} core_miss_or_cap={reason!r}"
        )


def _print_stage3_to_paper_bridge(cands: List[Dict[str, Any]], top_k: int = 20) -> None:
    """
    [Stage3->Paper bridge]
    在 select_terms_for_paper_recall 为每条写好 selected / reject 之后调用：
    bucket 与 paper 去留同一行对照。
    """
    if not STAGE3_AUDIT_DEBUG or not cands:
        return
    ranked = sorted(cands, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    ranked = _stage3_audit_filter_rows(ranked, top_k)
    if not ranked:
        print("\n" + "-" * 80 + "\n[Stage3->Paper bridge] (no rows after focus filter)\n" + "-" * 80)
        return

    print("\n" + "-" * 80)
    print("[Stage3->Paper bridge]")
    print("-" * 80)
    print(
        "term                     | bucket   | final_score | selected_for_paper | select_reason          | "
        "reject_reason            | ml_hits | can_exp | source_type    | parent_anchor"
    )
    for rec in ranked:
        term = (rec.get("term") or "")[:24]
        bucket = (rec.get("stage3_bucket") or "")[:8]
        final_score = float(rec.get("final_score") or 0.0)
        selected = bool(rec.get("selected_for_paper", False))
        select_reason = str(rec.get("select_reason") or "")
        reject_reason = str(rec.get("paper_reject_reason") or "")
        mainline_hits = int(rec.get("mainline_hits") or 0)
        can_expand = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
        st = rec.get("source_type") or rec.get("source") or ""
        if not st and rec.get("source_types"):
            st = ",".join(sorted(str(x) for x in list(rec.get("source_types"))[:3]))
        source_type = str(st)[:14]
        parent_anchor = str(rec.get("parent_anchor") or "")[:18]
        print(
            f"{term:24} | {bucket:8} | {final_score:11.4f} | {str(selected):^18} | {select_reason[:22]:22} | "
            f"{reject_reason[:24]:24} | {mainline_hits:^7d} | {str(can_expand):^7} | {source_type:14} | {parent_anchor:18}"
        )


def _print_paper_term_selection_audit(ordered: List[Dict[str, Any]], limit: int = 45) -> None:
    """按 core→support→risky、同桶内 final_score 顺序，打印每条 paper 门结果（含未入选）。"""
    if not STAGE3_PAPER_GATE_DEBUG or not ordered:
        return
    print(f"\n{'-' * 80}\n[Stage3] Paper term selection gate (audit)\n{'-' * 80}")
    for rec in ordered[:limit]:
        term = (rec.get("term") or "")[:48]
        fs = float(rec.get("final_score") or 0.0)
        can_expand = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
        fk_audit = rec.get("family_key") or build_family_key(rec)
        print(f"[paper_term_selection gate] term={term!r}")
        print(f"  bucket={rec.get('stage3_bucket')!r}")
        print(f"  primary_bucket={rec.get('primary_bucket')!r}")
        print(f"  term_role={rec.get('term_role')!r}")
        print(f"  final_score={fs:.4f}")
        print(f"  family_key={fk_audit!r}")
        print(f"  anchor_count={int(rec.get('anchor_count') or 0)}")
        print(f"  mainline_hits={int(rec.get('mainline_hits') or 0)}")
        print(f"  can_expand={can_expand}")
        print(f"  paper_grounding={float(rec.get('paper_grounding') or 0.0):.4f}")
        print(f"  risky_side_paper_block={bool(rec.get('risky_side_paper_block'))}")
        print(f"  paper_support_gate={rec.get('paper_support_gate')!r}")
        print(f"  selected_for_paper={rec.get('selected_for_paper', False)}")
        print(f"  select_reason={rec.get('select_reason')!r}")
        print(f"  paper_reject_reason={repr(rec.get('paper_reject_reason') or '')}")
        print(f"  retrieval_role={rec.get('retrieval_role')!r}")
        print(f"  reason_flags={rec.get('bucket_reason_flags') or []}")
        if rec.get("paper_final_score_need") is not None:
            print(f"  paper_final_score_need={float(rec.get('paper_final_score_need')):.4f}")
        if rec.get("paper_grounding_need") is not None:
            print(f"  paper_grounding_need={rec.get('paper_grounding_need')!r}")
        print(
            f"  paper_anchor_score_norm={float(rec.get('paper_anchor_score_norm') or 0):.4f} "
            f"paper_anchor_rank_smooth={float(rec.get('paper_anchor_rank_smooth') or 0):.4f} "
            f"paper_anchor_prior={float(rec.get('paper_anchor_prior') or 0):.4f}"
        )


def _paper_gate_is_fallback_primary(rec: Dict[str, Any]) -> bool:
    """与 select_terms_for_paper_recall 前置门一致的 fallback_primary 判定（供 tail expand 复用）。"""
    fb = bool(rec.get("fallback_primary", False))
    pb = (rec.get("primary_bucket") or "").strip().lower()
    pr = (rec.get("primary_reason") or "").strip().lower()
    return fb or pb == "primary_fallback_keep_no_expand" or pr == "anchor_core_fallback"


def _paper_parent_anchor_score_and_rank(rec: Dict[str, Any]) -> Tuple[float, int]:
    a_sc = float(
        rec.get("best_parent_anchor_final_score")
        or rec.get("parent_anchor_final_score")
        or 0.0
    )
    prk = rec.get("best_parent_anchor_step2_rank")
    if prk is None:
        prk = rec.get("parent_anchor_step2_rank")
    rk = int(prk) if prk is not None else 999
    return a_sc, rk


def _is_strong_main_axis_core(rec: Dict[str, Any]) -> bool:
    """primary lane 的 core：父锚够强或「主线+可扩+父锚靠前」——lane-tier 扫描序首段（先于 support_lane 与 bonus_core）。"""
    if (rec.get("stage3_bucket") or "").strip().lower() != "core":
        return False
    if str(rec.get("paper_recall_quota_lane") or "") != "primary":
        return False
    a_sc, rk = _paper_parent_anchor_score_and_rank(rec)
    ml = int(rec.get("mainline_hits") or 0)
    ex = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
    if a_sc >= float(PAPER_SELECT_STRONG_MAIN_AXIS_SCORE_MIN):
        return True
    if rk <= int(PAPER_SELECT_STRONG_MAIN_AXIS_MAX_RANK):
        return True
    if (
        ml >= 1
        and ex
        and a_sc >= float(PAPER_SELECT_STRONG_MAIN_AXIS_ML_EXP_SCORE_MIN)
        and rk <= int(PAPER_SELECT_STRONG_MAIN_AXIS_ML_EXP_MAX_RANK)
    ):
        return True
    return False


def _is_bonus_like_core(rec: Dict[str, Any]) -> bool:
    """primary lane 但未达强主轴门槛的 core（如宽学术词 / 方法学项）：lane-tier 全局序上在 support_lane 之后补位。"""
    if (rec.get("stage3_bucket") or "").strip().lower() != "core":
        return False
    if str(rec.get("paper_recall_quota_lane") or "") != "primary":
        return False
    return not _is_strong_main_axis_core(rec)


def _paper_bonus_core_readiness(rec: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    结构型 **bonus_core paper-ready** 乘子（无量词、不看字面）。

    - 仅当 **`stage3_bucket=core`** 且 **`_is_bonus_like_core`**（即后续 `paper_select_lane_tier=bonus_core`）时 `<1`；
    - 否则返回 `1.0`。
    - 用途：在 **`select_terms_for_paper_recall`** 中对 **`paper_select_score` 乘入**，与同函数内 **禁止**
      `core_near_miss` 的 **swap-in / tail-expand** 通道配套：保留 term 层可见性，收紧 **final_term_ids_for_paper** 入场。
    """
    dbg: Dict[str, Any] = {
        "applies": False,
        "a_sc": 0.0,
        "rk": 999,
        "ml": 0,
        "exp": False,
        "fb": False,
        "cond": False,
        "ready_factor": 1.0,
    }
    if (rec.get("stage3_bucket") or "").strip().lower() != "core":
        return 1.0, dbg
    if not _is_bonus_like_core(rec):
        return 1.0, dbg

    dbg["applies"] = True
    a_sc, rk = _paper_parent_anchor_score_and_rank(rec)
    ml = int(rec.get("mainline_hits") or 0)
    exp = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
    fb = _paper_gate_is_fallback_primary(rec)
    cond = _stage3_is_conditioned_only(rec)
    dbg.update(
        a_sc=float(a_sc),
        rk=int(rk),
        ml=ml,
        exp=exp,
        fb=fb,
        cond=cond,
    )

    ready = 1.0
    th_rk = int(PAPER_BONUS_CORE_READINESS_RANK_THRESHOLD)
    th_sc = float(PAPER_BONUS_CORE_READINESS_SCORE_THRESHOLD)
    if int(rk) > th_rk:
        ready *= float(PAPER_BONUS_CORE_READINESS_RANK_MULT)
    if float(a_sc) < th_sc:
        ready *= float(PAPER_BONUS_CORE_READINESS_SCORE_MULT)
    if ml <= 0:
        ready *= float(PAPER_BONUS_CORE_READINESS_ML0_MULT)
    if not exp:
        ready *= float(PAPER_BONUS_CORE_READINESS_NO_EXPAND_MULT)
    if fb:
        ready *= float(PAPER_BONUS_CORE_READINESS_FALLBACK_PRIMARY_MULT)
    if cond:
        ready *= float(PAPER_BONUS_CORE_READINESS_CONDITIONED_ONLY_MULT)

    dbg["ready_factor"] = float(ready)
    return float(ready), dbg


def _paper_select_bonus_tier_scale(rec: Dict[str, Any]) -> Tuple[float, str]:
    """
    返回 (scale, tier_key)，与后续写入的 paper_select_lane_tier 语义对齐。
    scale 乘在 **bonus 正部** 上；负部不缩放（见 _compute_paper_select_score）。
    """
    if _is_strong_main_axis_core(rec):
        return float(PAPER_SELECT_BONUS_TIER_SCALE_STRONG_MAIN_AXIS), "strong_main_axis_core"
    if _is_bonus_like_core(rec):
        return float(PAPER_SELECT_BONUS_TIER_SCALE_BONUS_CORE), "bonus_core"
    lane = str(rec.get("paper_recall_quota_lane") or "")
    if lane == "support":
        return float(PAPER_SELECT_BONUS_TIER_SCALE_SUPPORT_LANE), "support_lane"
    return float(PAPER_SELECT_BONUS_TIER_SCALE_OTHER_ELIGIBLE), "other_eligible"


def _paper_support_lane_scan_sort_key(rec: Dict[str, Any]) -> Tuple[int, float]:
    """
    support_lane 段内二级序：主轴近失配先进（JD 门双弱降级者），其余 support 随后；同级按 p_sel。
    """
    gate = str(rec.get("paper_support_gate") or "")
    reason = str(rec.get("paper_support_reason") or "")
    near_miss = (
        gate == "core_axis_near_miss_soft_admit"
        or reason == "core_axis_near_miss_soft_admit"
    )
    return (1 if near_miss else 0, float(rec.get("paper_select_score") or 0.0))


def _apply_paper_recall_tail_expand(
    selected: List[Dict[str, Any]],
    ordered: List[Dict[str, Any]],
    used_family: Set[str],
    floor: float,
    max_terms: int,
    selected_support: int,
) -> int:
    """
    **[已弃用 / 未再调用]**：尾部补位已内联至 `select_terms_for_paper_recall`（core near-miss 先于 support soft-admit）。
    保留本段仅供对照旧 **`tail_expand_after_big_pool_core`** 行为。

    主截断完成后：若 **词表过窄**（len(selected)≤3）或 **support 槽全空**（selected_support==0），
    从 near-miss 带再补 ≤MAX_EXTRA 槽。旧优先级：seed+ml+可扩 > core_axis_near_miss > 其余。
    仍 family 去重；不放宽 risky / fallback / conditioned_only。
    """
    if not PAPER_RECALL_TAIL_EXPAND_ENABLED or not ordered:
        return 0
    need_tail_expand = len(selected) <= 3 or int(selected_support) <= 0
    if not need_tail_expand:
        return 0
    cap_total = max_terms + int(PAPER_RECALL_TAIL_EXPAND_MAX_EXTRA)

    tail_candidates: List[Dict[str, Any]] = []
    for rec in ordered:
        if rec.get("selected_for_paper"):
            continue
        if _paper_gate_is_fallback_primary(rec):
            continue
        if _stage3_is_conditioned_only(rec):
            continue
        if (rec.get("stage3_bucket") or "").strip().lower() == "risky":
            continue

        p_sel = float(rec.get("paper_select_score") or 0.0)
        if p_sel < float(floor) - float(PAPER_RECALL_TAIL_EXPAND_DELTA_MAX):
            continue

        bkt = (rec.get("stage3_bucket") or "").strip().lower()
        pb = (rec.get("primary_bucket") or "").strip().lower()
        if PAPER_RECALL_TAIL_EXPAND_REQUIRE_CORE_OR_SEED:
            if not (bkt == "core" or pb == "primary_support_seed"):
                continue

        can_x = bool(rec.get("can_expand") or rec.get("can_expand_from_2a"))
        ml = int(rec.get("mainline_hits") or 0)
        sup_rs = str(rec.get("paper_support_reason") or "")
        # 批注：先抢「真 seed+主线+可扩」，再抢主轴 near-miss 降级 core，最后才是其它 core/seed
        if pb == "primary_support_seed" and ml >= 1 and can_x:
            pri = 3
        elif sup_rs.startswith("core_axis_near_miss"):
            pri = 2
        else:
            pri = 1
        rec["_tail_expand_priority"] = pri
        tail_candidates.append(rec)

    tail_candidates.sort(
        key=lambda x: (
            int(x.get("_tail_expand_priority") or 0),
            float(x.get("paper_select_score") or 0.0),
            float(x.get("final_score") or 0.0),
            int(x.get("mainline_hits") or 0),
            1 if bool(x.get("can_expand") or x.get("can_expand_from_2a")) else 0,
        ),
        reverse=True,
    )

    extra_n = 0
    for rec in tail_candidates:
        if extra_n >= int(PAPER_RECALL_TAIL_EXPAND_MAX_EXTRA):
            break
        if len(selected) >= cap_total:
            break
        fam = str(rec.get("family_key") or "").strip()
        if not fam:
            fam = build_family_key(rec)
            rec["family_key"] = fam
        if fam and fam in used_family:
            continue

        lane = str(rec.get("paper_recall_quota_lane") or "")
        rec["selected_for_paper"] = True
        rec["paper_reject_reason"] = ""
        rec["paper_cutoff_reason"] = "tail_expand_after_big_pool_core"
        rec["select_reason"] = "tail_expand_after_big_pool_core"
        rec["retrieval_role"] = "paper_primary" if lane == "primary" else "paper_support"
        used_family.add(fam)
        selected.append(rec)
        extra_n += 1
        rec.pop("_tail_expand_priority", None)

    for rec in tail_candidates:
        rec.pop("_tail_expand_priority", None)

    return extra_n


def _print_stage3_paper_swap_audit(
    did_swap: bool,
    swap_out_term: str,
    swap_in_term: str,
    out_psel: float,
    in_psel: float,
    floor: float,
) -> None:
    """主 cutoff 后 core near-miss ↔ suspicious support 换位的一行摘要（与 tail 触发无关）。"""
    if not STAGE3_PAPER_CUTOFF_AUDIT:
        return
    delta = float(out_psel) - float(in_psel)
    print(
        f"\n[Stage3 paper swap audit] swapped={did_swap} "
        f"out={swap_out_term!r} out_p_sel={out_psel:.3f} "
        f"in={swap_in_term!r} in_p_sel={in_psel:.3f} "
        f"delta={delta:+.3f} floor={floor:.3f} "
        f"reason='core_near_miss_replace_support_soft_admit'"
    )


def _paper_core_near_miss_rescue_channel_ok(
    rec: Dict[str, Any], floor: float, *, for_swap: bool
) -> bool:
    """
    主 cutoff 之后：是否满足 **core near-miss 抢救** 的结构性门槛（不含 family / 名额）。
    **bonus_core（`_is_bonus_like_core`）恒 False**，与 swap-in / tail 第一层一致。
    """
    if (rec.get("stage3_bucket") or "").strip().lower() != "core":
        return False
    if _is_bonus_like_core(rec):
        return False
    reason = str(rec.get("paper_cutoff_reason") or rec.get("paper_reject_reason") or "")
    if "below_dynamic_floor" not in reason:
        return False
    if int(rec.get("mainline_hits") or 0) < 1:
        return False
    if not bool(rec.get("can_expand") or rec.get("can_expand_from_2a")):
        return False
    p_sel = float(rec.get("paper_select_score") or 0.0)
    delta = (
        float(PAPER_RECALL_CORE_NEAR_MISS_SWAP_MAX_BELOW_FLOOR)
        if for_swap
        else float(PAPER_RECALL_TAIL_EXPAND_DELTA_MAX)
    )
    return p_sel >= float(floor) - delta


def _print_stage3_bonus_core_readiness_audit(
    ordered: List[Dict[str, Any]], floor: float,
) -> None:
    """bonus-like core：结构 ready 乘子、p_sel 缩放前后、core-near-miss 通道资格（对照 swap/tail 规则）。"""
    if not STAGE3_PAPER_CUTOFF_AUDIT or not STAGE3_BONUS_CORE_READINESS_AUDIT:
        return
    rows = [
        r
        for r in ordered
        if (r.get("paper_bonus_core_readiness_debug") or {}).get("applies")
    ]
    if not rows:
        return
    print("\n[Stage3 bonus-core readiness audit]")
    print(
        "term                  | tier      | a_sc | rk | ml | exp | fb | cond | ready | "
        "p_raw | p_adj | elig_sw | elig_te"
    )
    for rec in rows:
        term = str(rec.get("term") or "")[:22]
        tier = str(rec.get("paper_select_lane_tier") or "")[:9]
        d = rec.get("paper_bonus_core_readiness_debug") or {}
        p_raw = float(rec.get("paper_select_score_pre_bonus_core_readiness") or 0.0)
        p_adj = float(rec.get("paper_select_score") or 0.0)
        elig_sw = _paper_core_near_miss_rescue_channel_ok(rec, floor, for_swap=True)
        elig_te = _paper_core_near_miss_rescue_channel_ok(rec, floor, for_swap=False)
        print(
            f"{term:22s} | {tier:9s} | {float(d.get('a_sc', 0)):.2f} | {int(d.get('rk', 999)):4d} | "
            f"{int(d.get('ml', 0)):2d} | {str(bool(d.get('exp')))[:1]:3s} | "
            f"{str(bool(d.get('fb')))[:1]:3s} | {str(bool(d.get('cond')))[:1]:3s} | "
            f"{float(d.get('ready_factor', 1)):.3f} | {p_raw:.3f} | {p_adj:.3f} | "
            f"{str(bool(elig_sw)):5s} | {str(bool(elig_te)):5s}"
        )


def _print_stage3_paper_swap_pair_audit(
    core_nm_for_swap: List[Dict[str, Any]],
    sup_soft_for_swap_out: List[Dict[str, Any]],
    did_swap: bool,
    swap_in_term: str,
    swap_out_term: str,
    floor: float,
) -> None:
    """swap 候选对：lane、raw/adj p_sel、readiness、margin 是否允许换位（解释「为何换 / 为何没换」）。"""
    if not STAGE3_PAPER_CUTOFF_AUDIT:
        return
    margin_f = float(PAPER_RECALL_SUPPORT_SWAP_SCORE_MARGIN)
    print("\n[Stage3 paper swap pair audit]")
    if not core_nm_for_swap or not sup_soft_for_swap_out:
        print(
            f"  pair_available=False core_nm_n={len(core_nm_for_swap)} "
            f"sup_soft_out_n={len(sup_soft_for_swap_out)} floor={floor:.4f}"
        )
        return
    cin = core_nm_for_swap[0]
    cout = sup_soft_for_swap_out[0]
    cin_term = str(cin.get("term") or "")[:22]
    cout_term = str(cout.get("term") or "")[:22]
    in_lane = str(cin.get("paper_select_lane_tier") or "")
    out_lane = str(cout.get("paper_select_lane_tier") or "")
    in_raw = float(
        cin.get("paper_select_score_pre_bonus_core_readiness")
        or cin.get("paper_select_score")
        or 0.0
    )
    in_adj = float(cin.get("paper_select_score") or 0.0)
    out_eff = float(cout.get("paper_select_score") or 0.0)
    in_ready = float(cin.get("paper_bonus_core_readiness") or 1.0)
    margin_ok = in_adj >= out_eff - margin_f
    swap_ok = did_swap
    reason_bits: List[str] = []
    if not margin_ok:
        reason_bits.append("margin_fail")
    if _is_bonus_like_core(cin):
        reason_bits.append("bonus_core_not_in_swap_pool")
    reason_s = "|".join(reason_bits) if reason_bits else "ok"
    print(
        f"  in_term={cin_term!r} in_lane={in_lane!r} in_p_sel_raw={in_raw:.3f} "
        f"in_ready={in_ready:.3f} in_p_sel_adj={in_adj:.3f}\n"
        f"  out_term={cout_term!r} out_lane={out_lane!r} out_p_sel={out_eff:.3f} "
        f"margin_ok={margin_ok} (Δmargin={margin_f:.3f}) swap_ok={swap_ok} reason={reason_s!r}"
    )


def _print_stage3_tail_expand_audit(
    extra_n: int,
    floor: float,
    selected_before: int,
    need_tail_expand: bool,
    selected_support: int,
) -> None:
    if not STAGE3_PAPER_CUTOFF_AUDIT:
        return
    print(
        f"\n[Stage3 tail expand audit] enabled={PAPER_RECALL_TAIL_EXPAND_ENABLED} "
        f"trigger={need_tail_expand} (selected≤3 or support_lane_count==0) "
        f"extra_added={extra_n} floor={floor:.4f} selected_before={selected_before} "
        f"selected_support_before={selected_support} "
        f"Δ≤{PAPER_RECALL_TAIL_EXPAND_DELTA_MAX} max_extra={PAPER_RECALL_TAIL_EXPAND_MAX_EXTRA} "
        f"p_order=core_near_miss(1)>support_soft_admit(2)>other(3)"
    )


def _support_to_paper_quota_cap(max_terms: int) -> int:
    """最终 paper terms 中最多入选几条 stage3_bucket=support（与 max_terms 成比例）。"""
    if not SUPPORT_TO_PAPER_ENABLED or max_terms <= 0:
        return 0
    return min(SUPPORT_TO_PAPER_MAX, int(max_terms * SUPPORT_TO_PAPER_MAX_FRAC))


def _paper_select_pool_size_for_penalty(rec: Dict[str, Any]) -> int:
    """词汇侧论文池规模：优先 `papers_before_filter`，否则 `degree_w`；至少 1 避免 log(0)。"""
    for key in ("papers_before_filter", "degree_w"):
        v = rec.get(key)
        if v is None:
            continue
        try:
            return max(1, int(v))
        except (TypeError, ValueError):
            continue
    return 1


def _paper_grounding_score(rec: Dict[str, Any]) -> float:
    """0~1：主线命中、跨锚、ptc、family 综合；用于 Stage3 可疑 support 摘要等审计 proxy。"""
    ex = rec.get("stage3_explain") or {}
    ptc = float(ex.get("path_topic_consistency") or 0.0)
    fc = float(rec.get("family_centrality") or ex.get("family_centrality") or 0.0)
    mh = int(rec.get("mainline_hits") or 0)
    ac = int(rec.get("anchor_count") or 1)
    return float(
        min(
            1.0,
            0.18 * min(mh, 3)
            + 0.06 * max(0, ac - 1)
            + 0.44 * ptc
            + 0.32 * fc,
        )
    )


def _compute_paper_select_score(row: Dict[str, Any]) -> float:
    """
    仅用于 Stage3 **论文检索词**排序键（`paper_select_score`）。
    在 **`final_score × paper_readiness`** 底分上：**加性** Stage2 结构 bonus（经 **`PAPER_SELECT_BONUS_TIER_SCALE_*`**
    按 strong / bonus_core / support_lane / other 对 **bonus 正部**缩放）− **对数型候选池规模惩罚**（见 `PAPER_SELECT_POOL_PENALTY_*`）。
    **不修改** `final_score`，不回头改 Stage2。
    """
    # 与 tid 合并字段对齐，便于日志与外部工具读 `parent_anchor_final_score` / `parent_anchor_step2_rank`
    if row.get("parent_anchor_final_score") is None and row.get("best_parent_anchor_final_score") is not None:
        try:
            row["parent_anchor_final_score"] = float(row["best_parent_anchor_final_score"])
        except (TypeError, ValueError):
            pass
    if row.get("parent_anchor_step2_rank") is None and row.get("best_parent_anchor_step2_rank") is not None:
        try:
            row["parent_anchor_step2_rank"] = int(row["best_parent_anchor_step2_rank"])
        except (TypeError, ValueError):
            pass

    fs = float(row.get("final_score") or 0.0)
    ready = float(row.get("paper_readiness") or 0.0)

    parent_anchor_score = float(
        row.get("best_parent_anchor_final_score")
        or row.get("parent_anchor_final_score")
        or 0.0
    )
    parent_anchor_rank = int(
        row.get("best_parent_anchor_step2_rank")
        or row.get("parent_anchor_step2_rank")
        or 999
    )

    mainline_hits = int(row.get("mainline_hits") or 0)
    can_expand = bool(row.get("can_expand_from_2a", False) or row.get("can_expand", False))
    primary_bucket = (row.get("primary_bucket") or "").strip().lower()
    term_role = (row.get("term_role") or "").strip().lower()

    base = fs * ready

    bucket_bonus = 0.0
    if primary_bucket == "primary_expandable":
        bucket_bonus = PAPER_SELECT_BUCKET_BONUS_EXPANDABLE
    elif primary_bucket == "primary_support_seed":
        bucket_bonus = PAPER_SELECT_BUCKET_BONUS_SUPPORT_SEED
    elif primary_bucket == "primary_support_keep":
        bucket_bonus = PAPER_SELECT_BUCKET_PENALTY_SUPPORT_KEEP
    elif primary_bucket == "risky_keep":
        bucket_bonus = PAPER_SELECT_BUCKET_PENALTY_RISKY_KEEP
    elif primary_bucket in {"primary_keep_no_expand", "primary_fallback_keep_no_expand"}:
        bucket_bonus = PAPER_SELECT_BUCKET_PENALTY_SUPPORT_KEEP

    mainline_bonus = (
        PAPER_SELECT_BONUS_MAINLINE_HIT if mainline_hits > 0 else PAPER_SELECT_PENALTY_MAINLINE_NONE
    )
    expand_bonus = (
        PAPER_SELECT_BONUS_CAN_EXPAND if can_expand else PAPER_SELECT_PENALTY_NO_EXPAND
    )

    if parent_anchor_rank <= 3:
        anchor_rank_bonus = PAPER_SELECT_RANK_BONUS_LE3
    elif parent_anchor_rank <= 6:
        anchor_rank_bonus = PAPER_SELECT_RANK_BONUS_LE6
    elif parent_anchor_rank <= 10:
        anchor_rank_bonus = PAPER_SELECT_RANK_BONUS_LE10
    else:
        anchor_rank_bonus = 0.0

    anchor_score_norm = (
        max(0.0, min(1.0, parent_anchor_score / PAPER_SELECT_SCORE_ANCHOR_NORM))
        if parent_anchor_score > 0
        else 0.0
    )
    anchor_score_bonus = PAPER_SELECT_ANCHOR_SCORE_BONUS_COEF * anchor_score_norm

    role_bonus = 0.0
    if term_role == "primary":
        role_bonus = PAPER_SELECT_ROLE_BONUS_PRIMARY
    elif term_role == "primary_side":
        role_bonus = PAPER_SELECT_ROLE_PENALTY_PRIMARY_SIDE
    elif term_role == "dense_expansion":
        role_bonus = PAPER_SELECT_ROLE_PENALTY_DENSE_EXP

    bonus_total_raw = (
        bucket_bonus
        + mainline_bonus
        + expand_bonus
        + anchor_rank_bonus
        + anchor_score_bonus
        + role_bonus
    )
    tier_scale, tier_key = _paper_select_bonus_tier_scale(row)
    bonus_pos = max(0.0, float(bonus_total_raw))
    bonus_neg = min(0.0, float(bonus_total_raw))
    bonus_total = bonus_pos * float(tier_scale) + bonus_neg

    pool_size = _paper_select_pool_size_for_penalty(row)
    pool_penalty = 0.0
    if PAPER_SELECT_POOL_PENALTY_ENABLED:
        pool_penalty = min(
            float(PAPER_SELECT_POOL_PENALTY_MAX),
            float(PAPER_SELECT_POOL_PENALTY_LOG_COEF) * math.log10(float(pool_size)),
        )
    score = base + bonus_total - pool_penalty

    row["paper_select_pool_size"] = int(pool_size)
    row["paper_select_pool_penalty"] = float(pool_penalty)
    row["paper_select_score_base"] = float(base)
    row["paper_select_score_bonus_pre_tier"] = float(bonus_total_raw)
    row["paper_select_bonus_tier_scale"] = float(tier_scale)
    row["paper_select_bonus_tier_key"] = tier_key
    row["paper_select_score_bonus_total"] = float(bonus_total)

    return max(score, 0.0)


def _apply_paper_readiness_for_recall(rec: Dict[str, Any]) -> None:
    """
    Paper 选词序专用乘子（不改 final_score）：stage3 结构字段 + **Step2 锚点主线强度**（无词表）。
    - **锚点先验**：同时读 `best_parent_anchor_final_score` 与 `best_parent_anchor_step2_rank`；
      `anchor_score_norm` 主导，`rank_smooth=1/(1+λ·max(rank-1,0))` 仅轻量调和；再
      `readiness *= (blend_lo + blend_hi * anchor_prior)`，避免旧版「无分时 rank 单路倒数」压穿 paper。
    - **core + mainline_hits≥1 + 可扩**：`paper_readiness` 不低于 **`PAPER_READINESS_CORE_MAINLINE_EXPAND_MIN`**，减轻 dynamic_floor 对主轴词的误杀。
    - **primary_support_seed + mainline_hits≥1 + 可扩 + ¬conditioned_only**：不低于 **`PAPER_READINESS_SUPPORT_SEED_MAINLINE_EXPAND_MIN`**（随后在 step10 仍可 × **`PAPER_SUPPORT_SEED_FACTOR`**）。
    - **`paper_select_score`** 不在此函数内做简单乘法，而在写入 readiness 后交给 **`_compute_paper_select_score`**（底分×ready + Stage2 结构加性项）。
    """
    paper_readiness = 1.0
    bucket = (rec.get("stage3_bucket") or "").strip().lower()
    term_role = (rec.get("term_role") or "").strip().lower()
    mainline_hits = int(rec.get("mainline_hits") or 0)
    anchor_count = int(rec.get("anchor_count") or 0)
    can_expand = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
    conditioned_only = _stage3_is_conditioned_only(rec)
    pg_raw = rec.get("paper_grounding")
    paper_grounding = float(pg_raw) if pg_raw is not None else float(_paper_grounding_score(rec))

    # 1) bucket 软分层：core 最稳，support 次之，risky 最弱（观测桶，非硬拒）
    if bucket == "core":
        paper_readiness *= 1.00
    elif bucket == "support":
        paper_readiness *= 0.86
    else:
        paper_readiness *= 0.72

    # 2) 主线证据：命中越多越像「论文主检索词」
    paper_readiness *= 0.88 + 0.10 * min(mainline_hits, 2)

    # 3) 可扩性：能沿 Stage2A 主线继续扩的更偏主词而非纯 bonus
    paper_readiness *= 1.00 if can_expand else 0.84

    # 4) 多锚：跨锚更稳；单锚更易是旁枝/加分项
    paper_readiness *= 0.90 + 0.06 * min(anchor_count, 2)

    # 5) conditioned_only：已过单锚硬门者仍软压一层
    if conditioned_only:
        paper_readiness *= 0.80

    # 6) role：side / expansion 不应轻易压过 primary
    if term_role in {"primary_side", "dense_expansion", "cluster_expansion", "cooc_expansion"}:
        paper_readiness *= 0.82

    # 7) grounding proxy：越像可落 paper 项越保留
    paper_readiness *= 0.82 + 0.18 * paper_grounding

    # 8) 锚点主线先验：**以 parent 锚点分为准**，Step2 名次只做轻平滑；再线性混入 readiness（(rank) 不再单独主导整条乘子链）
    anchor_score = float(
        rec.get("best_parent_anchor_final_score")
        or rec.get("parent_anchor_final_score")
        or 0.0
    )
    anchor_rank = int(
        rec.get("best_parent_anchor_step2_rank")
        or rec.get("parent_anchor_step2_rank")
        or 999
    )
    anchor_score_norm = max(
        0.0,
        min(1.0, anchor_score / PAPER_READINESS_ANCHOR_SCORE_NORM),
    )
    rank_smooth = 1.0 / (
        1.0
        + PAPER_READINESS_ANCHOR_RANK_SMOOTH_LAMBDA * max(anchor_rank - 1, 0)
    )
    anchor_prior = (
        PAPER_READINESS_ANCHOR_PRIOR_W_SCORE * anchor_score_norm
        + PAPER_READINESS_ANCHOR_PRIOR_W_RANK * rank_smooth
    )
    anchor_prior_mult = (
        PAPER_READINESS_ANCHOR_READINESS_BLEND_LO
        + PAPER_READINESS_ANCHOR_READINESS_BLEND_HI * anchor_prior
    )
    paper_readiness *= anchor_prior_mult
    rec["paper_anchor_score_norm"] = float(anchor_score_norm)
    rec["paper_anchor_rank_smooth"] = float(rank_smooth)
    rec["paper_anchor_prior"] = float(anchor_prior)
    rec["paper_anchor_prior_mult"] = float(anchor_prior_mult)

    # 9) 主线可扩 core 保底：避免锚先验与连乘把「paper 主轴词」ready 压过低以至触发 below_dynamic_floor（非 RL 专惠，靠 bucket+ml+exp 结构）
    if bucket == "core" and mainline_hits >= 1 and can_expand:
        paper_readiness = max(paper_readiness, float(PAPER_READINESS_CORE_MAINLINE_EXPAND_MIN))

    # 9b) primary_support_seed + 主线 + 可扩 + 非 conditioned_only：readiness 保底（与 core 同类结构信号；助 soft_admit / tail 竞争）
    primary_bucket_rd = (rec.get("primary_bucket") or "").strip().lower()
    if (
        primary_bucket_rd == "primary_support_seed"
        and mainline_hits >= 1
        and can_expand
        and not conditioned_only
    ):
        paper_readiness = max(
            paper_readiness,
            float(PAPER_READINESS_SUPPORT_SEED_MAINLINE_EXPAND_MIN),
        )

    # 10) support 软放行：在 `select_terms_for_paper_recall` 已写入 `paper_support_factor`(<1) 时压低 readiness，再进入底分×bonus−pool
    psf = float(rec.get("paper_support_factor") or 1.0)
    if psf < 1.0 - 1e-9:
        paper_readiness *= psf

    rec["paper_readiness"] = float(paper_readiness)
    # paper 赛道最终排序键：底分×readiness + Stage2 主线/父锚/role 轻量加性项（见 _compute_paper_select_score）
    rec["paper_select_score"] = float(_compute_paper_select_score(rec))


def _stage4_paper_ready_score(rec: Dict[str, Any]) -> float:
    """
    Stage4 输入准备用轻量分数：只依赖 GC 四分项 + final_score + bucket 轻微先验。
    不参与 Stage3 主分、不模拟论文召回。
    """
    sb = rec.get("stage3_score_breakdown") or {}
    lf = float((sb.get("local_fit") or {}).get("score") or 0.0)
    cx = float((sb.get("cross_anchor_coherence") or {}).get("score") or 0.0)
    bb = float((sb.get("backbone_alignment") or {}).get("score") or 0.0)
    rk = float((sb.get("risk_penalty") or {}).get("score") or 0.0)
    fs = float(rec.get("final_score") or 0.0)
    bkt = (rec.get("stage3_bucket") or "").strip().lower()
    bucket_prior = 0.05 if bkt == "core" else (0.0 if bkt == "support" else -0.08)
    pos = (
        STAGE3_GC_W_LOCAL_FIT * lf
        + STAGE3_GC_W_CROSS * cx
        + STAGE3_GC_W_BACKBONE * bb
    )
    penalized = max(0.0, min(1.0, pos - STAGE3_GC_W_RISK * rk))
    blend = 0.52 * fs + 0.48 * penalized + bucket_prior
    return float(max(0.0, min(1.02, blend)))


def _stage4_prep_norm_str(s: Any) -> str:
    return str(s or "").strip().lower()


def _stage4_prep_topic_hint(rec: Dict[str, Any]) -> str:
    prov = rec.get("provenance_summary") or {}
    mdbg = rec.get("merge_debug_summary") or {}
    adb = rec.get("anchor_support_summary") or {}
    for key in ("dominant_topic", "topic", "topic_label", "subfield", "field"):
        v = prov.get(key)
        if v:
            return _stage4_prep_norm_str(v)
    for key in ("topic_hint", "topic", "subfield"):
        v = mdbg.get(key)
        if v:
            return _stage4_prep_norm_str(v)
    v2 = adb.get("topic_consistency_hint") or adb.get("path_topic_id")
    if v2:
        return _stage4_prep_norm_str(v2)
    return ""


def _stage4_prep_parent_anchor_str(rec: Dict[str, Any]) -> str:
    pa = _stage4_prep_norm_str(rec.get("parent_anchor"))
    if not pa and rec.get("parent_anchors"):
        try:
            pa = _stage4_prep_norm_str((rec.get("parent_anchors") or [""])[0])
        except (TypeError, IndexError, KeyError):
            pa = ""
    return pa


def _stage4_prep_term_tokens(term: Any) -> Set[str]:
    raw = _stage4_prep_norm_str(term).replace("/", " ").replace("-", " ")
    stop = {
        "the",
        "a",
        "an",
        "of",
        "and",
        "for",
        "in",
        "to",
        "on",
        "with",
    }
    return {t for t in raw.split() if len(t) > 1 and t not in stop}


def _stage4_prep_record_has_meta_signal(rec: Dict[str, Any]) -> bool:
    if rec.get("merge_debug_summary"):
        return True
    if rec.get("provenance_summary"):
        return True
    if rec.get("anchor_support_summary"):
        return True
    return False


def _stage4_prep_risky_compete_structural_ok(
    rec: Dict[str, Any],
    lf: float,
    bb: float,
    rk: float,
    fs: float,
) -> bool:
    if not STAGE4_PREP_RISKY_ALLOW_COMPETE:
        return False
    if fs < STAGE4_PREP_RISKY_COMPETE_MIN_FINAL:
        return False
    if rk > STAGE4_PREP_RISKY_COMPETE_MAX_RISK_PEN:
        return False
    if bb < STAGE4_PREP_RISKY_COMPETE_MIN_BACKBONE:
        return False
    if lf < STAGE4_PREP_RISKY_COMPETE_MIN_LOCAL_FIT:
        return False
    if STAGE4_PREP_RISKY_COMPETE_REQUIRE_META and not _stage4_prep_record_has_meta_signal(rec):
        return False
    return True


def _stage4_prep_pair_redundancy_core(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """单源 redundancy：0~1 clip；detail 供 breakdown / 日志。"""
    fk_a = _stage4_prep_norm_str(a.get("family_key"))
    fk_b = _stage4_prep_norm_str(b.get("family_key"))
    r = 0.0
    family_exact = bool(fk_a and fk_b and fk_a == fk_b)
    family_prefix_align = False
    if fk_a and fk_b:
        if family_exact:
            r += STAGE4_PREP_REDUND_W_FAMILY_EXACT
        else:
            n = min(len(fk_a), len(fk_b), 14)
            if n >= 6 and fk_a[:n] == fk_b[:n]:
                family_prefix_align = True
                r += STAGE4_PREP_REDUND_W_FAMILY_PREFIX
    pp_a = _stage4_prep_norm_str(a.get("parent_primary"))
    pp_b = _stage4_prep_norm_str(b.get("parent_primary"))
    parent_primary_exact = bool(pp_a and pp_b and pp_a == pp_b)
    parent_primary_substring = bool(
        pp_a and pp_b and not parent_primary_exact and (pp_a in pp_b or pp_b in pp_a)
    )
    if parent_primary_exact:
        r += STAGE4_PREP_REDUND_W_PP_EXACT
    elif parent_primary_substring:
        r += STAGE4_PREP_REDUND_W_PP_SUB
    pa_a = _stage4_prep_parent_anchor_str(a)
    pa_b = _stage4_prep_parent_anchor_str(b)
    parent_anchor_exact = bool(pa_a and pa_b and pa_a == pa_b)
    parent_anchor_substring = bool(
        pa_a and pa_b and not parent_anchor_exact and (pa_a in pa_b or pa_b in pa_a)
    )
    if parent_anchor_exact:
        r += STAGE4_PREP_REDUND_W_PA_EXACT
    elif parent_anchor_substring:
        r += STAGE4_PREP_REDUND_W_PA_SUB
    th_a = _stage4_prep_topic_hint(a)
    th_b = _stage4_prep_topic_hint(b)
    topic_hint_exact = bool(th_a and th_b and th_a == th_b)
    topic_substring = bool(
        th_a
        and th_b
        and not topic_hint_exact
        and min(len(th_a), len(th_b)) >= 4
        and (th_a in th_b or th_b in th_a)
    )
    th_contrib = 0.0
    if topic_hint_exact:
        th_contrib = STAGE4_PREP_REDUND_W_TOPIC_EXACT
        r += th_contrib
    elif topic_substring:
        th_contrib = STAGE4_PREP_REDUND_W_TOPIC_SUB
        r += th_contrib
    to_a = _stage4_prep_term_tokens(a.get("term"))
    to_b = _stage4_prep_term_tokens(b.get("term"))
    token_jaccard = 0.0
    tj_contrib = 0.0
    if to_a and to_b:
        inter = len(to_a & to_b)
        union = len(to_a | to_b)
        if union:
            token_jaccard = inter / float(union)
            tj_contrib = token_jaccard * STAGE4_PREP_REDUND_W_TOKEN_JACCARD
            r += tj_contrib
    r_raw_pre = float(r)
    r_final = float(max(0.0, min(1.0, r)))
    detail: Dict[str, Any] = {
        "same_family": family_exact,
        "family_prefix_align": family_prefix_align,
        "family_match": family_exact or family_prefix_align,
        "parent_primary_exact": parent_primary_exact,
        "parent_primary_substring": parent_primary_substring,
        "parent_primary_match": parent_primary_exact or parent_primary_substring,
        "parent_anchor_exact": parent_anchor_exact,
        "parent_anchor_substring": parent_anchor_substring,
        "parent_anchor_match": parent_anchor_exact or parent_anchor_substring,
        "topic_hint_match": topic_hint_exact,
        "topic_substring_match": topic_substring,
        "topic_match": topic_hint_exact or topic_substring,
        "topic_hint_available": bool(th_a or th_b),
        "token_jaccard": round(token_jaccard, 4),
        "family_contrib": round(
            (STAGE4_PREP_REDUND_W_FAMILY_EXACT if family_exact else 0.0)
            + (STAGE4_PREP_REDUND_W_FAMILY_PREFIX if family_prefix_align else 0.0),
            4,
        ),
        "parent_primary_contrib": round(
            (STAGE4_PREP_REDUND_W_PP_EXACT if parent_primary_exact else 0.0)
            + (STAGE4_PREP_REDUND_W_PP_SUB if parent_primary_substring else 0.0),
            4,
        ),
        "parent_anchor_contrib": round(
            (STAGE4_PREP_REDUND_W_PA_EXACT if parent_anchor_exact else 0.0)
            + (STAGE4_PREP_REDUND_W_PA_SUB if parent_anchor_substring else 0.0),
            4,
        ),
        "topic_contrib": round(th_contrib, 4),
        "token_jaccard_contrib": round(tj_contrib, 4),
        "additive_before_clip": round(r_raw_pre, 4),
        "redundancy": r_final,
        "one_pair_context_gain": float(max(0.0, min(1.0, 1.0 - r_final))),
    }
    return r_final, detail


def _stage4_prep_pair_redundancy(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """0~1：越高越「近邻 / 同簇」，用于压低互补 gain。"""
    r_final, _ = _stage4_prep_pair_redundancy_core(a, b)
    return float(r_final)


def _stage4_prep_pair_redundancy_breakdown(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> Dict[str, Any]:
    """与 `_stage4_prep_pair_redundancy` 同一核心；拆项供观测。"""
    _, detail = _stage4_prep_pair_redundancy_core(a, b)
    base_gain = float(max(0.0, min(1.0, 1.0 - float(detail["redundancy"]))))
    detail["context_coverage_gain_raw"] = round(base_gain, 4)
    detail["context_coverage_gain_shaped"] = round(
        float(max(0.0, min(1.0, base_gain ** float(STAGE4_PREP_COVERAGE_GAIN_SHARPEN)))), 4
    )
    return detail


def _stage3_paper_obs_enabled(specific_flag: bool) -> bool:
    return bool(specific_flag or (STAGE3_AUDIT_DEBUG and STAGE3_PAPER_OBS_WITH_AUDIT_DEBUG))


def _stage3_paper_obs_gc_four(rec: Dict[str, Any]) -> Dict[str, Any]:
    sb = rec.get("stage3_score_breakdown") or {}
    if not isinstance(sb, dict):
        return {
            "local_fit": "n/a",
            "cross_anchor_coherence": "n/a",
            "backbone_alignment": "n/a",
            "risk_penalty": "n/a",
        }

    def _s(block: str) -> Any:
        b = sb.get(block)
        if isinstance(b, dict) and "score" in b:
            try:
                return round(float(b.get("score") or 0.0), 4)
            except (TypeError, ValueError):
                return "n/a"
        return "n/a"

    return {
        "local_fit": _s("local_fit"),
        "cross_anchor_coherence": _s("cross_anchor_coherence"),
        "backbone_alignment": _s("backbone_alignment"),
        "risk_penalty": _s("risk_penalty"),
    }


def _stage3_paper_obs_parent_anchor_raw(rec: Dict[str, Any]) -> str:
    pa = rec.get("parent_anchor")
    if pa is not None and str(pa).strip():
        return str(pa).strip()
    if rec.get("parent_anchors"):
        try:
            return str((rec.get("parent_anchors") or [""])[0]).strip()
        except (TypeError, IndexError, KeyError):
            return ""
    return ""


def _stage3_paper_obs_soft_flags(rec: Dict[str, Any]) -> Any:
    gf = rec.get("stage3_guardrail_soft_flags")
    if gf:
        return gf
    return rec.get("bucket_reason_flags") or []


def _stage3_paper_obs_row_status(
    rec: Dict[str, Any],
    pool_ids: Set[int],
    selected_ids: Set[int],
) -> str:
    if rec.get("selected_for_paper"):
        return "selected_for_stage4"
    gp = str(rec.get("_stage4_prep_gate_phase") or "")
    if gp == "pre_coverage":
        return "blocked_pre_coverage"
    if id(rec) in pool_ids:
        if id(rec) in selected_ids:
            return "selected_for_stage4"
        return "rejected_after_coverage"
    if gp == "selection":
        return "rejected_after_coverage"
    if gp == "in_pool":
        return "entered_coverage"
    return "survived_stage3"


def _stage3_paper_obs_block_stage_row(
    rec: Dict[str, Any],
    pool_ids: Set[int],
    selected_ids: Set[int],
) -> Tuple[str, str]:
    """(block_stage, block_reason) 供 report / 全量表。"""
    if rec.get("selected_for_paper"):
        return "selected", ""
    gp = str(rec.get("_stage4_prep_gate_phase") or "")
    gr = str(rec.get("_stage4_prep_gate_reason") or rec.get("paper_reject_reason") or "")
    if gp == "pre_coverage":
        return "pre_coverage", gr
    if id(rec) in pool_ids and id(rec) not in selected_ids:
        return "post_selection", str(rec.get("paper_reject_reason") or rec.get("paper_cutoff_reason") or "")
    if gp == "selection":
        return "selection", gr
    if id(rec) in pool_ids:
        return "in_pool", ""
    return "not_in_pool", gr


def _print_stage3_paper_candidate_pool_full(
    records: List[Dict[str, Any]],
    *,
    recall: Any = None,
) -> None:
    print("\n" + "=" * 88)
    print("[Stage3 paper candidate pool full audit] all survivors in Stage4 prep scope")
    print("=" * 88)
    ctx_n = None
    if recall is not None:
        ctx = getattr(recall, "_stage3_stage2_context", None)
        if isinstance(ctx, dict):
            ac = ctx.get("all_candidates")
            if isinstance(ac, list):
                ctx_n = len(ac)
    if ctx_n is not None:
        print(
            f"note: stage2 all_candidates count (contract list) = {ctx_n} "
            f"| paper prep input = {len(records)} survivors"
        )
    else:
        print(
            f"note: stage2 all_candidates unavailable on recall | paper prep input = {len(records)} survivors"
        )
    rows = sorted(
        records,
        key=lambda r: float(r.get("final_score") or 0.0),
        reverse=True,
    )
    for r in rows:
        gc = _stage3_paper_obs_gc_four(r)
        pr_v = r.get("paper_ready_for_stage4")
        pr_s = f"{float(pr_v):.4f}" if pr_v is not None else "n/a"
        g = r.get("context_coverage_gain")
        g_s = f"{float(g):.3f}" if g is not None else "n/a"
        s4 = r.get("stage4_prep_score")
        s4_s = f"{float(s4):.4f}" if s4 is not None else "n/a"
        sel = r.get("select_reason") or ""
        rej = r.get("paper_reject_reason") or r.get("paper_cutoff_reason") or ""
        pa = r.get("parent_anchor") or ""
        if not pa and r.get("parent_anchors"):
            try:
                pa = (r.get("parent_anchors") or [""])[0]
            except (TypeError, IndexError, KeyError):
                pa = ""
        pbase = r.get("paper_ready_base_for_stage4")
        pbase_s = f"{float(pbase):.4f}" if pbase is not None else "n/a"
        print(
            f"term={str(r.get('term'))[:36]!r} tid={r.get('tid')} bucket={r.get('stage3_bucket')!r} "
            f"final={float(r.get('final_score') or 0):.4f} paper_ready={pr_s} paper_ready_base={pbase_s} gain={g_s} s4p={s4_s}\n"
            f"  penalty_kind={str(r.get('_stage4_prep_penalty_kind') or '')!r} "
            f"fk={str(r.get('family_key'))[:40]!r} pp={str(r.get('parent_primary'))[:36]!r} "
            f"pa={str(pa)[:36]!r} pool_lane={r.get('_stage4_prep_pool_lane')!r} "
            f"gate_phase={r.get('_stage4_prep_gate_phase')!r} gate_reason={r.get('_stage4_prep_gate_reason')!r}\n"
            f"  LF={gc['local_fit']} CX={gc['cross_anchor_coherence']} "
            f"BB={gc['backbone_alignment']} RK={gc['risk_penalty']}\n"
            f"  guardrail_soft_flags={_stage3_paper_obs_soft_flags(r)} "
            f"sel_reason={sel!r} reject={rej!r}"
        )
    print("=" * 88 + "\n")


def _print_stage3_pre_coverage_block_audit(records: List[Dict[str, Any]]) -> None:
    blocked = [r for r in records if str(r.get("_stage4_prep_gate_phase") or "") == "pre_coverage"]
    print("\n" + "=" * 88)
    print(f"[Stage3 pre-coverage block audit] n={len(blocked)} (hard_block only; excludes penalized_admit)")
    print("=" * 88)
    for r in sorted(blocked, key=lambda x: float(x.get("final_score") or 0.0), reverse=True):
        pr_v = r.get("paper_ready_for_stage4")
        pr_s = f"{float(pr_v):.4f}" if pr_v is not None else "n/a"
        pa = r.get("parent_anchor") or ""
        if not pa and r.get("parent_anchors"):
            try:
                pa = (r.get("parent_anchors") or [""])[0]
            except (TypeError, IndexError, KeyError):
                pa = ""
        br = str(r.get("_stage4_prep_gate_reason") or r.get("paper_cutoff_reason") or "")
        print(
            f"term={str(r.get('term'))[:36]!r} tid={r.get('tid')} bucket={r.get('stage3_bucket')!r} "
            f"final={float(r.get('final_score') or 0):.4f} paper_ready={pr_s}\n"
            f"  fk={str(r.get('family_key'))[:40]!r} pp={str(r.get('parent_primary'))[:36]!r} "
            f"pa={str(pa)[:36]!r}\n"
            f"  block_stage=pre_coverage block_reason={br!r}"
        )
    print("=" * 88 + "\n")


def _stage3_prep_penalty_compete_audit_enabled() -> bool:
    return bool(STAGE3_PREP_PENALTY_COMPETE_AUDIT or STAGE3_PAPER_GATE_DEBUG)


def _print_stage3_prep_penalty_audit(records: List[Dict[str, Any]]) -> None:
    """weak_support_contamination → penalized-admit：base vs 惩罚后 paper_ready、是否进池。"""
    rows = [
        r
        for r in records
        if str(r.get("_stage4_prep_penalty_kind") or "") == "penalized_admit"
        or str(r.get("paper_support_gate") or "") == "weak_support_contamination_penalized_admit"
    ]
    print("\n" + "=" * 88)
    print(f"[Stage3 pre-coverage penalty audit] weak_support penalized_admit n={len(rows)}")
    print("=" * 88)
    for r in sorted(rows, key=lambda x: float(x.get("final_score") or 0.0), reverse=True):
        base = r.get("paper_ready_base_for_stage4")
        pr = r.get("paper_ready_for_stage4")
        base_s = f"{float(base):.4f}" if base is not None else "n/a"
        pr_s = f"{float(pr):.4f}" if pr is not None else "n/a"
        in_pool = str(r.get("_stage4_prep_gate_phase") or "") == "in_pool"
        print(
            f"term={str(r.get('term'))[:36]!r} tid={r.get('tid')} bucket={r.get('stage3_bucket')!r}\n"
            f"  block_or_penalty_reason=weak_support_contamination_penalized_admit "
            f"base_paper_ready={base_s} penalized_paper_ready={pr_s} in_coverage_pool={in_pool}"
        )
    print("=" * 88 + "\n")


def _print_stage3_risky_compete_audit(records: List[Dict[str, Any]]) -> None:
    """risky default → coverage_eligible_risky（进池 / overflow）。"""
    rows = [
        r
        for r in records
        if str(r.get("_stage4_prep_pool_lane") or "") == "risky_compete"
        or str(r.get("paper_cutoff_reason") or "") == "risky_compete_pool_overflow"
    ]
    print("\n" + "=" * 88)
    print(f"[Stage3 risky compete audit] coverage_eligible_risky / overflow n={len(rows)}")
    print("=" * 88)
    for r in sorted(rows, key=lambda x: float(x.get("final_score") or 0.0), reverse=True):
        base = r.get("paper_ready_base_for_stage4")
        pr = r.get("paper_ready_for_stage4")
        base_s = f"{float(base):.4f}" if base is not None else "n/a"
        pr_s = f"{float(pr):.4f}" if pr is not None else "n/a"
        in_pool = str(r.get("_stage4_prep_gate_phase") or "") == "in_pool"
        reason = "risky_compete_pool_overflow" if not in_pool else "coverage_eligible_risky"
        print(
            f"term={str(r.get('term'))[:36]!r} tid={r.get('tid')} bucket={r.get('stage3_bucket')!r}\n"
            f"  status={reason!r} base_paper_ready={base_s} penalized_paper_ready={pr_s} in_coverage_pool={in_pool}"
        )
    print("=" * 88 + "\n")


def _print_stage3_context_coverage_pairwise(
    backbone: Optional[Dict[str, Any]],
    compare_rows: List[Dict[str, Any]],
    lam: float,
) -> None:
    if not backbone:
        print("\n[Stage3 context coverage pairwise audit] skipped: no backbone record\n")
        return
    print("\n" + "=" * 88)
    print("[Stage3 context coverage pairwise audit] vs backbone (pair); final gain uses max over selected set")
    print(
        f"backbone term={str(backbone.get('term'))!r} tid={backbone.get('tid')} λ={lam:.3f}"
    )
    print("=" * 88)
    for r in compare_rows:
        bd = _stage4_prep_pair_redundancy_breakdown(backbone, r)
        _pr_c = float(r.get("paper_ready_for_stage4") or 0.0)
        _g_final = float(r.get("context_coverage_gain") or 0.0)
        print(
            f"candidate={str(r.get('term'))[:36]!r} tid={r.get('tid')} bucket={r.get('stage3_bucket')!r}\n"
            f"  same_family={bd['same_family']} family_prefix={bd['family_prefix_align']} family_match={bd['family_match']} "
            f"pp_match={bd['parent_primary_match']} pa_match={bd['parent_anchor_match']} "
            f"topic_hint_match={bd['topic_hint_match']} topic_substring_match={bd.get('topic_substring_match')} "
            f"topic_match={bd.get('topic_match')} topic_hint_available={bd['topic_hint_available']}\n"
            f"  token_jaccard={bd['token_jaccard']} additive_pre_clip={bd['additive_before_clip']} "
            f"redundancy={bd['redundancy']:.4f} one_pair_gain_raw={bd.get('context_coverage_gain_raw')} "
            f"one_pair_gain_shaped={bd.get('context_coverage_gain_shaped')}\n"
            f"  paper_ready={_pr_c:.4f} context_coverage_gain_vs_selected_set={_g_final:.4f} "
            f"λ*gain={lam * _g_final:.4f} stage4_prep_score={_pr_c + lam * _g_final:.4f}"
            f"\n  pool_lane={r.get('_stage4_prep_pool_lane')!r}"
        )
    print("=" * 88 + "\n")


def _print_stage3_context_axis_trace_audit(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 88)
    print(f"[Stage3 context-axis trace audit] n={len(rows)} (structural keys only)")
    print("=" * 88)
    for row in rows:
        print(
            f"term={row.get('term')!r} tid={row.get('tid')} status={row.get('status')!r} "
            f"bucket={row.get('stage3_bucket')!r}\n"
            f"  pa={row.get('parent_anchor')!r} pp={row.get('parent_primary')!r} "
            f"fk={row.get('family_key')!r} topic_hint={row.get('topic_hint')!r}\n"
            f"  final={row.get('final_score')} paper_ready={row.get('paper_ready')} "
            f"block_stage={row.get('block_stage')!r} block_reason={row.get('block_reason')!r}"
        )
    print("=" * 88 + "\n")


def _compute_stage4_prep_context_coverage_gain(
    candidate: Dict[str, Any],
    selected: List[Dict[str, Any]],
    recall: Any = None,
) -> float:
    """
    轻量互补度：相对已选集合，与任一条越近 redundancy 越高，gain 越低。不读词表、不做组合优化。
    recall 预留；当前仅使用候选与已选 record 上的结构字段。
    第二轮：对 (1 - red_max) 做轻度指数 sharpen，压低近邻重复的有效增益。
    """
    _ = recall
    if not selected:
        return 0.0
    red_max = 0.0
    for s in selected:
        red_max = max(red_max, _stage4_prep_pair_redundancy(candidate, s))
    base = float(max(0.0, min(1.0, 1.0 - red_max)))
    return float(max(0.0, min(1.0, base ** float(STAGE4_PREP_COVERAGE_GAIN_SHARPEN))))


def select_terms_for_paper_recall(
    records: List[Dict[str, Any]],
    max_terms: int = 12,
    recall: Any = None,
    stage3_dropped_pre_paper: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Stage4 input prep (lightweight): after global coherence rerank, only choose terms to send to Stage4.

    Hard gates: conditioned_only+single anchor, fallback_primary, risky_side_block.
    weak_support_contamination：默认 penalized-admit（乘法压低 paper_ready，仍进 coverage 池竞争；非旧 soft-admit）。
    risky default：在通用结构门槛下可极少进入 coverage_eligible_risky 池。无 swap/tail/重型 lane。

    Selection: (1) one **backbone** core (highest paper_ready passing core_floor + family);
    (2) greedy fill with ``stage4_prep_score = paper_ready + λ * context_coverage_gain`` vs current selected;
    gain：结构 redundancy v2 + (1-red_max)^sharpen — no keyword lists.
    """
    if not records:
        return []

    gate_reject_counts: Dict[str, int] = defaultdict(int)
    pre_coverage_penalized_counts: Dict[str, int] = Counter()
    pool_core: List[Dict[str, Any]] = []
    pool_support: List[Dict[str, Any]] = []
    pool_risky_exc: List[Dict[str, Any]] = []
    risky_compete_staging: List[Dict[str, Any]] = []
    coverage_eligible_risky_count = 0

    for rec in records:
        anchor_count = int(rec.get("anchor_count") or 1) or 1
        primary_bucket = (rec.get("primary_bucket") or "").strip().lower()
        primary_reason = (rec.get("primary_reason") or "").strip().lower()
        fallback_primary = bool(rec.get("fallback_primary", False))
        is_fallback_primary = (
            fallback_primary
            or primary_bucket == "primary_fallback_keep_no_expand"
            or primary_reason == "anchor_core_fallback"
        )
        st = _stage3_normalized_source_types(rec)
        has_similar = "similar_to" in st
        has_family = "family_landing" in st
        has_conditioned = "conditioned_vec" in st
        conditioned_only = bool(has_conditioned and (not has_similar) and (not has_family))

        rec["selected_for_paper"] = False
        rec["select_reason"] = ""
        rec["paper_reject_reason"] = ""
        rec["paper_cutoff_reason"] = ""
        rec["risky_side_paper_block"] = False
        rec.pop("paper_select_lane_tier", None)
        rec["_stage4_prep_gate_phase"] = ""
        rec["_stage4_prep_gate_reason"] = ""
        rec["_stage4_prep_pool_lane"] = ""
        rec["_stage4_prep_penalty_kind"] = ""
        rec["paper_ready_base_for_stage4"] = None  # type: ignore[assignment]
        contamination_penalized = False

        if conditioned_only and anchor_count <= 1:
            rec["paper_reject_reason"] = "conditioned_only_single_anchor_block"
            rec["paper_cutoff_reason"] = "conditioned_only_single_anchor_block"
            gate_reject_counts["conditioned_only_single_anchor_block"] += 1
            rec["_stage4_prep_gate_phase"] = "pre_coverage"
            rec["_stage4_prep_gate_reason"] = "conditioned_only_single_anchor_block"
            continue

        if is_fallback_primary:
            rec["paper_reject_reason"] = "fallback_primary_block"
            rec["paper_cutoff_reason"] = "fallback_primary_block"
            gate_reject_counts["fallback_primary_block"] += 1
            rec["_stage4_prep_gate_phase"] = "pre_coverage"
            rec["_stage4_prep_gate_reason"] = "fallback_primary_block"
            continue

        paper_grounding = _paper_grounding_score(rec)
        rec["paper_grounding"] = paper_grounding
        bucket = (rec.get("stage3_bucket") or "").strip().lower()
        pb = (rec.get("primary_bucket") or "").strip().lower()
        anchor_count_gate = int(rec.get("anchor_count") or 0)
        mainline_hits = int(rec.get("mainline_hits") or 0)
        can_expand = bool(rec.get("can_expand", False) or rec.get("can_expand_from_2a", False))
        conditioned_only_gate = _stage3_is_conditioned_only(rec)
        final_score_gate = float(rec.get("final_score") or 0.0)
        weak_support_contamination = (
            STAGE3_PAPER_SUPPORT_BLOCK_ENABLED
            and bucket == "support"
            and pb in {"primary_support_seed", "primary_support_keep"}
            and (
                (
                    anchor_count_gate <= STAGE3_PAPER_SUPPORT_MAX_ANCHOR_COUNT
                    and mainline_hits <= STAGE3_PAPER_SUPPORT_MAX_MAINLINE_HITS
                    and paper_grounding < STAGE3_PAPER_SUPPORT_MIN_GROUNDING
                )
                or ((not can_expand) and paper_grounding < STAGE3_PAPER_SUPPORT_MIN_GROUNDING)
                or (
                    STAGE3_PAPER_SUPPORT_BLOCK_COND_ONLY
                    and conditioned_only_gate
                    and anchor_count_gate <= 2
                )
                or (
                    final_score_gate < STAGE3_PAPER_SUPPORT_MIN_FINAL_SCORE
                    and paper_grounding < STAGE3_PAPER_SUPPORT_MIN_GROUNDING
                )
            )
        )
        if weak_support_contamination:
            if (
                STAGE3_PAPER_SUPPORT_BLOCK_ENABLED
                and STAGE4_PREP_SUPPORT_CONTAMINATION_ALLOW_COMPETE
            ):
                contamination_penalized = True
                rec["paper_support_gate"] = "weak_support_contamination_penalized_admit"
                rec["_stage4_prep_penalty_kind"] = "penalized_admit"
                rec["paper_reject_reason"] = ""
                rec["paper_cutoff_reason"] = ""
                pre_coverage_penalized_counts["weak_support_contamination_penalized_admit"] += 1
            else:
                rec["paper_support_gate"] = "weak_support_contamination_block"
                rec["paper_reject_reason"] = "weak_support_contamination_block"
                rec["paper_cutoff_reason"] = "weak_support_contamination_block"
                gate_reject_counts["weak_support_contamination_block"] += 1
                rec["_stage4_prep_gate_phase"] = "pre_coverage"
                rec["_stage4_prep_gate_reason"] = "weak_support_contamination_block"
                continue

        term_role_gate = (rec.get("term_role") or "").strip().lower()
        risky_side_paper_block = (
            STAGE3_PAPER_RISKY_SIDE_BLOCK_ENABLED
            and bucket == "risky"
            and (not can_expand)
            and mainline_hits <= 0
            and term_role_gate in STAGE3_PAPER_RISKY_SIDE_TERM_ROLES
        )
        rec["risky_side_paper_block"] = bool(risky_side_paper_block)
        if risky_side_paper_block:
            rec["paper_support_gate"] = "risky_side_block"
            rec["paper_reject_reason"] = "risky_side_block"
            rec["paper_cutoff_reason"] = "risky_side_block"
            gate_reject_counts["risky_side_block"] += 1
            rec["_stage4_prep_gate_phase"] = "pre_coverage"
            rec["_stage4_prep_gate_reason"] = "risky_side_block"
            continue

        sb = rec.get("stage3_score_breakdown") or {}
        lf = float((sb.get("local_fit") or {}).get("score") or 0.0)
        bb = float((sb.get("backbone_alignment") or {}).get("score") or 0.0)
        rk = float((sb.get("risk_penalty") or {}).get("score") or 0.0)
        pr_base = float(_stage4_paper_ready_score(rec))
        rec["paper_ready_base_for_stage4"] = pr_base
        pr = pr_base
        if contamination_penalized:
            pr = pr_base * float(STAGE4_PREP_SUPPORT_CONTAMINATION_PENALTY)
        rec["paper_ready_for_stage4"] = pr
        rec["paper_select_score"] = pr

        fk = rec.get("family_key") or build_family_key(rec)
        rec["family_key"] = fk
        rec["paper_main_axis_gate_pass"] = True
        rec["paper_main_axis_gate_reason"] = "stage4_prep_no_main_axis_gate"

        if bucket == "core":
            rec["paper_support_gate"] = "stage4_prep_core"
            rec["paper_recall_quota_lane"] = "primary"
            rec["_stage4_prep_gate_phase"] = "in_pool"
            rec["_stage4_prep_gate_reason"] = ""
            rec["_stage4_prep_pool_lane"] = "core"
            pool_core.append(rec)
        elif bucket == "support":
            if lf < STAGE4_PREP_SUPPORT_MIN_LF and bb < STAGE4_PREP_SUPPORT_MIN_BB + 0.04:
                rec["paper_reject_reason"] = "support_paper_prep_low_lf_bb"
                rec["paper_cutoff_reason"] = "support_paper_prep_low_lf_bb"
                gate_reject_counts["support_paper_prep_low_lf_bb"] += 1
                rec["_stage4_prep_gate_phase"] = "pre_coverage"
                rec["_stage4_prep_gate_reason"] = "support_paper_prep_low_lf_bb"
                continue
            if lf < STAGE4_PREP_SUPPORT_MIN_LF or bb < STAGE4_PREP_SUPPORT_MIN_BB:
                rec["paper_reject_reason"] = "support_paper_prep_low_gc_subscores"
                rec["paper_cutoff_reason"] = "support_paper_prep_low_gc_subscores"
                gate_reject_counts["support_paper_prep_low_gc_subscores"] += 1
                rec["_stage4_prep_gate_phase"] = "pre_coverage"
                rec["_stage4_prep_gate_reason"] = "support_paper_prep_low_gc_subscores"
                continue
            if rk > STAGE4_PREP_SUPPORT_MAX_RISK_DIM:
                rec["paper_reject_reason"] = "support_paper_prep_high_risk_dim"
                rec["paper_cutoff_reason"] = "support_paper_prep_high_risk_dim"
                gate_reject_counts["support_paper_prep_high_risk_dim"] += 1
                rec["_stage4_prep_gate_phase"] = "pre_coverage"
                rec["_stage4_prep_gate_reason"] = "support_paper_prep_high_risk_dim"
                continue
            if not contamination_penalized:
                rec["paper_support_gate"] = "stage4_prep_support"
            rec["paper_recall_quota_lane"] = "support"
            rec["_stage4_prep_gate_phase"] = "in_pool"
            rec["_stage4_prep_gate_reason"] = ""
            rec["_stage4_prep_pool_lane"] = "support"
            pool_support.append(rec)
        elif bucket == "risky":
            fs = float(rec.get("final_score") or 0.0)
            if (
                fs >= STAGE4_PREP_RISKY_MIN_FINAL
                and rk <= STAGE4_PREP_RISKY_MAX_RISK_DIM
                and (lf + bb) >= STAGE4_PREP_RISKY_MIN_LF_BB_SUM
            ):
                rec["paper_support_gate"] = "stage4_prep_risky_exception"
                rec["paper_recall_quota_lane"] = "support"
                rec["_stage4_prep_gate_phase"] = "in_pool"
                rec["_stage4_prep_gate_reason"] = ""
                rec["_stage4_prep_pool_lane"] = "risky_exception"
                pool_risky_exc.append(rec)
            else:
                if _stage4_prep_risky_compete_structural_ok(rec, lf, bb, rk, fs):
                    risky_compete_staging.append(rec)
                else:
                    rec["paper_reject_reason"] = "risky_default_not_for_stage4_prep"
                    rec["paper_cutoff_reason"] = "risky_default_not_for_stage4_prep"
                    gate_reject_counts["risky_default_not_for_stage4_prep"] += 1
                    rec["_stage4_prep_gate_phase"] = "pre_coverage"
                    rec["_stage4_prep_gate_reason"] = "risky_default_not_for_stage4_prep"
        else:
            rec["paper_reject_reason"] = "paper_prep_bucket_not_core_support_risky"
            rec["paper_cutoff_reason"] = "paper_prep_bucket_not_core_support_risky"
            gate_reject_counts["paper_prep_bucket_not_core_support_risky"] += 1
            rec["_stage4_prep_gate_phase"] = "pre_coverage"
            rec["_stage4_prep_gate_reason"] = "paper_prep_bucket_not_core_support_risky"

    if risky_compete_staging:
        ranked_rc = sorted(
            risky_compete_staging,
            key=lambda r: (
                float(r.get("paper_ready_for_stage4") or 0.0),
                float(r.get("final_score") or 0.0),
            ),
            reverse=True,
        )
        admit_n = max(0, int(STAGE4_PREP_RISKY_COMPETE_MAX))
        admit = ranked_rc[:admit_n]
        spill = ranked_rc[admit_n:]
        for rec in spill:
            rec["paper_reject_reason"] = "risky_compete_pool_overflow"
            rec["paper_cutoff_reason"] = "risky_compete_pool_overflow"
            gate_reject_counts["risky_compete_pool_overflow"] += 1
            rec["_stage4_prep_gate_phase"] = "pre_coverage"
            rec["_stage4_prep_gate_reason"] = "risky_compete_pool_overflow"
        for rec in admit:
            base_pr = float(rec.get("paper_ready_for_stage4") or 0.0)
            rec["paper_ready_base_for_stage4"] = base_pr
            pr2 = base_pr * float(STAGE4_PREP_RISKY_COMPETE_READY_MULT)
            rec["paper_ready_for_stage4"] = pr2
            rec["paper_select_score"] = pr2
            rec["_stage4_prep_penalty_kind"] = "coverage_eligible_risky"
            rec["paper_support_gate"] = "stage4_prep_risky_compete"
            rec["paper_recall_quota_lane"] = "support"
            rec["_stage4_prep_gate_phase"] = "in_pool"
            rec["_stage4_prep_gate_reason"] = ""
            rec["_stage4_prep_pool_lane"] = "risky_compete"
            rec["paper_reject_reason"] = ""
            rec["paper_cutoff_reason"] = ""
            pool_risky_exc.append(rec)
            pre_coverage_penalized_counts["coverage_eligible_risky"] += 1
        coverage_eligible_risky_count = int(len(admit))

    floor_pool = pool_core + pool_support
    if floor_pool:
        top_pr = max(float(r.get("paper_ready_for_stage4") or 0.0) for r in floor_pool)
        floor = max(PAPER_RECALL_DYNAMIC_FLOOR_ABS, top_pr * PAPER_RECALL_DYNAMIC_FLOOR_REL)
    else:
        floor = float(PAPER_RECALL_DYNAMIC_FLOOR_ABS)

    core_floor = max(0.0, floor - float(STAGE4_PREP_CORE_FLOOR_RELAX))

    def _sort_prep_pool(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            rows,
            key=lambda r: (
                float(r.get("paper_ready_for_stage4") or 0.0),
                float(r.get("final_score") or 0.0),
            ),
            reverse=True,
        )

    sorted_core = _sort_prep_pool(pool_core)
    sorted_support = _sort_prep_pool(pool_support)
    sorted_risky = _sort_prep_pool(pool_risky_exc)

    pool_all_passed = pool_core + pool_support + pool_risky_exc
    used_family: Set[str] = set()
    selected: List[Dict[str, Any]] = []
    selected_ids: Set[int] = set()
    selected_support = 0
    selected_risky = 0
    support_cap = _support_to_paper_quota_cap(max_terms)
    family_dedup_drops = 0
    family_skip_terms: Set[str] = set()
    coverage_based_promotions = 0
    lam = float(STAGE4_PREP_COVERAGE_LAMBDA)

    def _mark_selected(
        rec: Dict[str, Any],
        role: str,
        reason: str,
        cutoff: str,
        gain: float,
        sps: float,
        detail: str,
    ) -> None:
        rec["selected_for_paper"] = True
        rec["select_reason"] = reason
        rec["paper_cutoff_reason"] = cutoff
        rec["paper_reject_reason"] = ""
        rec["retrieval_role"] = "paper_primary" if role == "core" else "paper_support"
        rec["paper_select_lane_tier"] = "stage4_prep_" + role
        rec["context_coverage_gain"] = float(gain)
        rec["stage4_prep_score"] = float(sps)
        rec["paper_select_score"] = float(sps)
        rec["stage4_prep_selection_detail"] = detail
        rec["_stage4_prep_gate_phase"] = "selected"
        rec["_stage4_prep_gate_reason"] = ""

    def _eligible_for_slot(
        rec: Dict[str, Any],
        bkt: str,
        pr: float,
        fk: str,
    ) -> Tuple[bool, str]:
        if fk in used_family:
            return False, "family_already_in_paper_set"
        if bkt == "core":
            if pr < core_floor:
                return False, "below_paper_ready_floor_core"
            return True, ""
        if bkt == "support":
            if support_cap > 0 and selected_support >= support_cap:
                return False, "support_quota_full"
            if pr < floor:
                return False, "below_paper_ready_floor"
            return True, ""
        if bkt == "risky":
            if selected_risky >= STAGE4_PREP_RISKY_MAX:
                return False, "risky_exception_quota_full"
            if support_cap > 0 and selected_support >= support_cap:
                return False, "support_quota_full_risky_counts_as_support"
            if pr < floor:
                return False, "below_paper_ready_floor"
            return True, ""
        return False, "paper_prep_bucket_not_core_support_risky"

    # --- A: 主骨架：最高分 core 一条（仍受 floor + family）---
    backbone: Optional[Dict[str, Any]] = None
    for rec in sorted_core:
        pr = float(rec.get("paper_ready_for_stage4") or 0.0)
        if pr < core_floor:
            rec["paper_reject_reason"] = "below_paper_ready_floor_core"
            rec["paper_cutoff_reason"] = "below_paper_ready_floor_core"
            rec["_stage4_prep_gate_phase"] = "selection"
            rec["_stage4_prep_gate_reason"] = "below_paper_ready_floor_core"
            continue
        fk = str(rec.get("family_key") or "").strip()
        if fk in used_family:
            rec["paper_reject_reason"] = "family_duplicate_block"
            rec["paper_cutoff_reason"] = "family_dup_stage4_prep"
            rec["_stage4_prep_gate_phase"] = "selection"
            rec["_stage4_prep_gate_reason"] = "family_duplicate_block"
            family_dedup_drops += 1
            continue
        used_family.add(fk)
        backbone = rec
        sps = pr
        _mark_selected(
            rec,
            "core",
            "stage4_prep_backbone_core",
            "stage4_prep_selected_backbone",
            0.0,
            sps,
            "backbone_core",
        )
        selected.append(rec)
        selected_ids.add(id(rec))
        break

    # --- B: 其余槽位：core 余量 + support + risky，按 stage4_prep_score 贪心（无 core 时首槽等价于 paper_ready max）---
    rest_cores = [r for r in sorted_core if id(r) not in selected_ids]
    candidates_fill = rest_cores + sorted_support + sorted_risky

    while len(selected) < max_terms and candidates_fill:
        best_rec: Optional[Dict[str, Any]] = None
        best_key = (-1.0, -1.0, -1.0)
        best_gain = 0.0
        best_detail = ""
        best_role = "support"
        best_reason = ""
        best_cutoff = ""

        for rec in candidates_fill:
            if id(rec) in selected_ids:
                continue
            bkt = (rec.get("stage3_bucket") or "").strip().lower()
            pr = float(rec.get("paper_ready_for_stage4") or 0.0)
            fk = str(rec.get("family_key") or "").strip()
            ok, why = _eligible_for_slot(rec, bkt, pr, fk)
            if not ok:
                if why == "family_already_in_paper_set":
                    family_skip_terms.add(str(rec.get("term") or ""))
                continue

            gain = _compute_stage4_prep_context_coverage_gain(rec, selected, recall)
            sps = pr + lam * gain
            fin = float(rec.get("final_score") or 0.0)
            key = (sps, pr, fin)

            if bkt == "core":
                detail = "complement_core"
                role = "core"
                cutoff = "stage4_prep_selected_core_complement"
                reason = "stage4_prep_core_complement"
            elif bkt == "support":
                detail = "complement_support"
                role = "support"
                cutoff = "stage4_prep_selected_support_coverage"
                reason = "stage4_prep_support_coverage"
            elif str(rec.get("_stage4_prep_pool_lane") or "") == "risky_compete":
                detail = "complement_risky_compete"
                role = "risky"
                cutoff = "stage4_prep_selected_risky_compete"
                reason = "stage4_prep_risky_compete_coverage"
            else:
                detail = "complement_risky_exception"
                role = "risky"
                cutoff = "stage4_prep_selected_risky_exception"
                reason = "stage4_prep_risky_exception_coverage"

            if best_rec is None or key > best_key:
                best_key = key
                best_rec = rec
                best_gain = gain
                best_detail = detail
                best_role = role
                best_reason = reason
                best_cutoff = cutoff

        if best_rec is None:
            break

        fk_b = str(best_rec.get("family_key") or "").strip()
        used_family.add(fk_b)
        if best_gain >= STAGE4_PREP_COVERAGE_PROMOTION_THRESH and best_detail.startswith("complement_"):
            coverage_based_promotions += 1

        _mark_selected(
            best_rec,
            best_role,
            best_reason,
            best_cutoff,
            best_gain,
            float(best_key[0]),
            best_detail,
        )
        selected.append(best_rec)
        selected_ids.add(id(best_rec))
        bkt_b = (best_rec.get("stage3_bucket") or "").strip().lower()
        if bkt_b in {"support", "risky"}:
            selected_support += 1
        if bkt_b == "risky":
            selected_risky += 1

    same_family_skips = len(family_skip_terms)

    for rec in pool_all_passed:
        if rec.get("selected_for_paper"):
            continue
        g = _compute_stage4_prep_context_coverage_gain(rec, selected, recall)
        pr = float(rec.get("paper_ready_for_stage4") or 0.0)
        rec["context_coverage_gain"] = float(g)
        rec["stage4_prep_score"] = float(pr + lam * g)

    for rec in records:
        if rec.get("selected_for_paper"):
            continue
        if not rec.get("paper_cutoff_reason"):
            rec["paper_cutoff_reason"] = "not_selected_stage4_prep"
            if not rec.get("paper_reject_reason"):
                rec["paper_reject_reason"] = "not_selected_stage4_prep"

    redundancy_rules = [
        f"weights: family_exact={STAGE4_PREP_REDUND_W_FAMILY_EXACT} prefix={STAGE4_PREP_REDUND_W_FAMILY_PREFIX}",
        f"parent_primary exact/sub={STAGE4_PREP_REDUND_W_PP_EXACT}/{STAGE4_PREP_REDUND_W_PP_SUB}",
        f"parent_anchor exact/sub={STAGE4_PREP_REDUND_W_PA_EXACT}/{STAGE4_PREP_REDUND_W_PA_SUB}",
        f"topic exact/sub={STAGE4_PREP_REDUND_W_TOPIC_EXACT}/{STAGE4_PREP_REDUND_W_TOPIC_SUB} "
        f"token_jaccard_coef={STAGE4_PREP_REDUND_W_TOKEN_JACCARD} clip=[0,1]",
    ]
    coverage_rules = [
        "redundancy v2: stronger family/parent/topic/token-jaccard (generic signals; no domain lexicon)",
        f"gain = (1 - max_pair_redundancy)^SHARPEN SHARPEN=STAGE4_PREP_COVERAGE_GAIN_SHARPEN; "
        f"stage4_prep_score = paper_ready + λ*gain λ=STAGE4_PREP_COVERAGE_LAMBDA",
        "weak_support_contamination: penalized-admit (ready mult) enters pool; risky default: tiny compete pool",
    ]
    selection_rules = [
        "hard_gates: conditioned_only_1anch, fallback_primary, risky_side_block; "
        "weak_support_contamination penalized-admit or legacy hard_block if ALLOW_COMPETE off",
        "backbone: first passing core by sorted paper_ready (floor+family); then greedy by stage4_prep_score",
        "support_cap / risky exception / risky_compete (cap STAGE4_PREP_RISKY_COMPETE_MAX) / family dedup",
        "paper_ready: _stage4_prep blend; complement uses shaped context_coverage_gain (Stage4 validates evidence)",
    ]
    sel_core_n = sum(1 for r in selected if (r.get("stage3_bucket") or "").lower() == "core")
    sel_sup_n = sum(1 for r in selected if (r.get("stage3_bucket") or "").lower() == "support")
    sel_risk_n = sum(1 for r in selected if (r.get("stage3_bucket") or "").lower() == "risky")
    coverage_risky_selected_count = sum(
        1 for r in selected if str(r.get("_stage4_prep_pool_lane") or "") == "risky_compete"
    )

    pool_ids_set = {id(r) for r in pool_all_passed}
    sel_ids_set = {id(r) for r in selected}
    pre_coverage_block_counts: Dict[str, int] = Counter()
    for _r in records:
        if str(_r.get("_stage4_prep_gate_phase") or "") == "pre_coverage":
            pre_coverage_block_counts[str(_r.get("_stage4_prep_gate_reason") or "unknown")] += 1

    obs_notes: List[str] = [
        "context_axis_trace uses structural fields only (no fixed semantic-axis classifier).",
        "Stage4 prep: weak_support_contamination → penalized-admit (STAGE4_PREP_SUPPORT_CONTAMINATION_*); "
        "risky default → coverage_eligible_risky (STAGE4_PREP_RISKY_*_COMPETE_*). Stage4 still validates papers.",
    ]
    if stage3_dropped_pre_paper is None:
        obs_notes.append(
            "stage3_dropped_pre_paper not passed; axis trace lists survivors only. "
            "dual_gate can pass dropped_with_reason into select_terms_for_paper_recall(..., stage3_dropped_pre_paper=...)."
        )
    else:
        obs_notes.append(
            f"stage3_dropped_pre_paper: {len(stage3_dropped_pre_paper)} rows; up to 40 appended to context_axis_trace."
        )

    pairwise_anchor = backbone if backbone is not None else (selected[0] if selected else None)
    pairwise_coverage_samples: List[Dict[str, Any]] = []
    if pairwise_anchor is not None:
        _picap = 48
        for _cand in list(sorted_support) + list(sorted_risky):
            if _picap <= 0:
                break
            _bd = _stage4_prep_pair_redundancy_breakdown(pairwise_anchor, _cand)
            pairwise_coverage_samples.append(
                {
                    "selected_backbone_term": pairwise_anchor.get("term"),
                    "selected_backbone_tid": pairwise_anchor.get("tid"),
                    "candidate_term": _cand.get("term"),
                    "candidate_tid": _cand.get("tid"),
                    "candidate_bucket": _cand.get("stage3_bucket"),
                    **_bd,
                    "context_coverage_gain_final_vs_set": float(_cand.get("context_coverage_gain") or 0),
                }
            )
            _picap -= 1

    context_axis_trace: List[Dict[str, Any]] = []
    for _r in sorted(records, key=lambda x: float(x.get("final_score") or 0.0), reverse=True):
        _pr = _r.get("paper_ready_for_stage4")
        _th = _stage4_prep_topic_hint(_r)
        bs, br = _stage3_paper_obs_block_stage_row(_r, pool_ids_set, sel_ids_set)
        context_axis_trace.append(
            {
                "term": _r.get("term"),
                "tid": _r.get("tid"),
                "parent_anchor": _stage3_paper_obs_parent_anchor_raw(_r) or "n/a",
                "parent_primary": _r.get("parent_primary") or "n/a",
                "family_key": _r.get("family_key") or "n/a",
                "topic_hint": _th or "n/a",
                "stage3_bucket": _r.get("stage3_bucket"),
                "final_score": float(_r.get("final_score") or 0.0),
                "paper_ready": float(_pr) if _pr is not None else "n/a",
                "status": _stage3_paper_obs_row_status(_r, pool_ids_set, sel_ids_set),
                "block_stage": bs,
                "block_reason": br,
            }
        )
    if stage3_dropped_pre_paper:
        for _dr in stage3_dropped_pre_paper[:40]:
            if not isinstance(_dr, dict):
                continue
            _dth = _stage4_prep_topic_hint(_dr)
            context_axis_trace.append(
                {
                    "term": _dr.get("term") or "n/a",
                    "tid": _dr.get("tid"),
                    "parent_anchor": _stage3_paper_obs_parent_anchor_raw(_dr) or "n/a",
                    "parent_primary": _dr.get("parent_primary") or "n/a",
                    "family_key": _dr.get("family_key") or "n/a",
                    "topic_hint": _dth or "n/a",
                    "stage3_bucket": _dr.get("stage3_bucket") or "n/a",
                    "final_score": float(_dr.get("final_score") or 0.0),
                    "paper_ready": "n/a",
                    "status": "dropped_stage3_pre_survivor",
                    "block_stage": "stage3_survivor_filter",
                    "block_reason": str(_dr.get("reject_reason") or ""),
                }
            )

    selected_backbone_term = str(backbone.get("term") or "") if backbone is not None else ""

    def _row_payload_sel(rec: Dict[str, Any]) -> Dict[str, Any]:
        gc = _stage3_paper_obs_gc_four(rec)
        bs, br = _stage3_paper_obs_block_stage_row(rec, pool_ids_set, sel_ids_set)
        pr_v = rec.get("paper_ready_for_stage4")
        pb_v = rec.get("paper_ready_base_for_stage4")
        return {
            "term": rec.get("term") or "",
            "tid": rec.get("tid"),
            "bucket": rec.get("stage3_bucket"),
            "final_score": float(rec.get("final_score") or 0.0),
            "paper_ready": float(pr_v) if pr_v is not None else "n/a",
            "paper_ready_base": float(pb_v) if pb_v is not None else "n/a",
            "_stage4_prep_penalty_kind": str(rec.get("_stage4_prep_penalty_kind") or ""),
            "pool_lane": str(rec.get("_stage4_prep_pool_lane") or ""),
            "context_coverage_gain": float(rec.get("context_coverage_gain") or 0.0),
            "stage4_prep_score": float(rec.get("stage4_prep_score") or rec.get("paper_select_score") or 0.0),
            "family_key": rec.get("family_key") or "",
            "parent_primary": rec.get("parent_primary") or "",
            "parent_anchor": _stage3_paper_obs_parent_anchor_raw(rec) or "n/a",
            "guardrail_soft_flags": _stage3_paper_obs_soft_flags(rec),
            "local_fit": gc["local_fit"],
            "cross_anchor_coherence": gc["cross_anchor_coherence"],
            "backbone_alignment": gc["backbone_alignment"],
            "risk_penalty": gc["risk_penalty"],
            "selection_reason": str(rec.get("select_reason") or ""),
            "selection_detail": str(rec.get("stage4_prep_selection_detail") or ""),
            "block_stage": bs,
            "block_reason": br,
        }

    def _row_payload_rej(rec: Dict[str, Any]) -> Dict[str, Any]:
        rr = str(rec.get("paper_reject_reason") or rec.get("paper_cutoff_reason") or "")
        gc = _stage3_paper_obs_gc_four(rec)
        bs, br = _stage3_paper_obs_block_stage_row(rec, pool_ids_set, sel_ids_set)
        pr_v = rec.get("paper_ready_for_stage4")
        s4v = rec.get("stage4_prep_score")
        if s4v is None:
            s4v = rec.get("paper_select_score")
        pb_v = rec.get("paper_ready_base_for_stage4")
        return {
            "term": rec.get("term") or "",
            "tid": rec.get("tid"),
            "bucket": rec.get("stage3_bucket"),
            "final_score": float(rec.get("final_score") or 0.0),
            "paper_ready": float(pr_v) if pr_v is not None else "n/a",
            "paper_ready_base": float(pb_v) if pb_v is not None else "n/a",
            "_stage4_prep_penalty_kind": str(rec.get("_stage4_prep_penalty_kind") or ""),
            "pool_lane": str(rec.get("_stage4_prep_pool_lane") or ""),
            "context_coverage_gain": float(rec.get("context_coverage_gain") or 0.0),
            "stage4_prep_score": float(s4v) if s4v is not None else "n/a",
            "family_key": rec.get("family_key") or "",
            "parent_primary": rec.get("parent_primary") or "",
            "parent_anchor": _stage3_paper_obs_parent_anchor_raw(rec) or "n/a",
            "local_fit": gc["local_fit"],
            "cross_anchor_coherence": gc["cross_anchor_coherence"],
            "backbone_alignment": gc["backbone_alignment"],
            "risk_penalty": gc["risk_penalty"],
            "reject_reason": rr,
            "block_stage": bs,
            "block_reason": br,
        }

    selected_terms = [_row_payload_sel(r) for r in selected]
    rejected_terms = [_row_payload_rej(r) for r in records if not r.get("selected_for_paper")]
    report: Dict[str, Any] = {
        "selected_count": len(selected),
        "core_selected": sel_core_n,
        "support_selected": sel_sup_n,
        "risky_selected": sel_risk_n,
        "rejected_count": len(rejected_terms),
        "family_dedup_drops": family_dedup_drops,
        "context_coverage_used": True,
        "coverage_lambda": lam,
        "coverage_rules": coverage_rules,
        "redundancy_rules": redundancy_rules,
        "coverage_based_promotions": coverage_based_promotions,
        "same_family_skips": same_family_skips,
        "candidate_pool_size_before_selection": len(records),
        "pre_coverage_penalized_counts": dict(pre_coverage_penalized_counts),
        "coverage_eligible_risky_count": coverage_eligible_risky_count,
        "coverage_risky_selected_count": coverage_risky_selected_count,
        "pre_coverage_block_counts": dict(pre_coverage_block_counts),
        "selected_backbone_term": selected_backbone_term,
        "pairwise_coverage_samples": pairwise_coverage_samples,
        "context_axis_trace": context_axis_trace[:120],
        "obs_notes": obs_notes,
        "paper_ready_floor": floor,
        "paper_ready_core_floor": core_floor,
        "support_cap": support_cap,
        "selection_rules": selection_rules,
        "gate_reject_counts": dict(gate_reject_counts),
        "selected_terms": selected_terms,
        "rejected_terms": rejected_terms[:80],
    }
    if recall is not None and getattr(recall, "debug_info", None) is not None:
        recall.debug_info.paper_lane_report = report  # type: ignore[attr-defined]

    if _stage3_paper_obs_enabled(STAGE3_PAPER_POOL_FULL_AUDIT):
        _print_stage3_paper_candidate_pool_full(records, recall=recall)
    if _stage3_paper_obs_enabled(STAGE3_PRE_COVERAGE_BLOCK_AUDIT):
        _print_stage3_pre_coverage_block_audit(records)
    if _stage3_paper_obs_enabled(STAGE3_CONTEXT_COVERAGE_PAIRWISE_AUDIT):
        _print_stage3_context_coverage_pairwise(
            pairwise_anchor, list(sorted_support) + list(sorted_risky), lam
        )
    if _stage3_paper_obs_enabled(STAGE3_CONTEXT_AXIS_TRACE_AUDIT):
        _print_stage3_context_axis_trace_audit(context_axis_trace[: min(len(context_axis_trace), 80)])

    if _stage3_prep_penalty_compete_audit_enabled():
        _print_stage3_prep_penalty_audit(records)
        _print_stage3_risky_compete_audit(records)

    _print_stage3_paper_gate_reject_audit(gate_reject_counts)
    do_audit = bool(
        STAGE3_PAPER_GATE_DEBUG or LABEL_PATH_TRACE or getattr(term_scoring, "STAGE3_DEBUG", False)
    )
    if do_audit:
        print("\n" + "-" * 80)
        print("[Stage3 paper term preparation audit] Stage4 input prep (not validation)")
        print("-" * 80)
        for r in selected[:20]:
            print(
                f"  SEL term={str(r.get('term'))[:32]!r} bucket={(r.get('stage3_bucket') or '')!r} "
                f"final={float(r.get('final_score') or 0):.4f} paper_ready={float(r.get('paper_ready_for_stage4') or 0):.4f} "
                f"s4p={float(r.get('stage4_prep_score') or 0):.4f} gain={float(r.get('context_coverage_gain') or 0):.3f} "
                f"reason={(r.get('select_reason') or '')!r}"
            )
        print("-" * 80)
        print("[Stage3 context coverage audit] support candidates vs final selected (complement, not validation)")
        print("-" * 80)
        for r in sorted_support[:24]:
            g = float(r.get("context_coverage_gain") or 0.0)
            pr = float(r.get("paper_ready_for_stage4") or 0.0)
            s4 = pr + lam * g
            pa = _stage4_prep_parent_anchor_str(r)[:20]
            pp = _stage4_prep_norm_str(r.get("parent_primary"))[:24]
            fk = _stage4_prep_norm_str(r.get("family_key"))[:28]
            st = "SEL" if r.get("selected_for_paper") else "skip"
            print(
                f"  {st} term={str(r.get('term'))[:28]!r} fk={fk!r} pp={pp!r} pa={pa!r} "
                f"pr={pr:.4f} gain={g:.3f} s4p={s4:.4f}"
            )
        rej_show = [r for r in records if not r.get("selected_for_paper")][:14]
        for r in rej_show:
            print(
                f"  REJ term={str(r.get('term'))[:32]!r} bucket={(r.get('stage3_bucket') or '')!r} "
                f"final={float(r.get('final_score') or 0):.4f} paper_ready={float(r.get('paper_ready_for_stage4') or 0):.4f} "
                f"why={(r.get('paper_reject_reason') or r.get('paper_cutoff_reason') or '')!r}"
            )
    print(
        f"\n[Stage3 paper lane summary] context_coverage_used=True λ={lam:.3f} promotions={coverage_based_promotions} "
        f"same_family_skips={same_family_skips} core={sel_core_n} support={sel_sup_n} risky_exc={sel_risk_n} "
        f"family_dedup_drops={family_dedup_drops} total_to_stage4={len(selected)} "
        f"floor={floor:.4f} core_floor={core_floor:.4f} support_cap={support_cap}"
    )
    if STAGE3_AUDIT_DEBUG:
        _print_stage3_to_paper_bridge(records, top_k=20)
    return selected

def stage3_build_score_map(survivors: List[Dict[str, Any]], recall: Any = None) -> List[Dict[str, Any]]:
    """
    Stage3 第二段：**Global Coherence Rerank** 主分（四分项）+ 小幅 legacy unified blend；
    `stage3_bucket` **仅在**本段 `final_score` 定稿后再赋值，仅作角色映射/观测。

    - **主路径**：`stage3_score_breakdown`（local_fit / cross_anchor_coherence / backbone_alignment / risk_penalty）
      消费 merge 聚合字段与 `candidate_graph.cross_anchor_support_edges`（经 recall 透传）。
    - **兼容**：`stage3_unified_breakdown` 仍为 legacy 连续分拆解；`mainline_risk_penalty` / `cross_anchor_score_bonus` 置 1.0 兼容旧表。
    - **家族约束**：统一分算完后 `_apply_family_role_constraints`。
    """
    for rec in survivors:
        rec["bucket_reason_flags"] = _collect_stage3_bucket_reason_flags(rec)

    for rec in survivors:
        rec["_stage3_pre_adjust_score"] = float(rec.get("final_score") or 0.0)
        final, gc_bd = _build_stage3_global_coherence_score(rec, recall)
        rec["final_score"] = final
        rec["stage3_score_breakdown"] = gc_bd
        rec["stage3_unified_breakdown"] = gc_bd.get("legacy_unified_breakdown") or {}
        ex = rec.get("stage3_explain") or {}
        ex["mainline_risk_penalty"] = 1.0
        ex["cross_anchor_score_bonus"] = 1.0
        ex["unified_continuous_score"] = float(gc_bd.get("legacy_unified_score") or 0.0)
        ex["global_coherence_pre_blend"] = float(gc_bd.get("gc_pre_blend") or 0.0)
        ex["local_fit_score"] = float((gc_bd.get("local_fit") or {}).get("score") or 0.0)
        ex["cross_anchor_coherence_score"] = float((gc_bd.get("cross_anchor_coherence") or {}).get("score") or 0.0)
        ex["backbone_alignment_score"] = float((gc_bd.get("backbone_alignment") or {}).get("score") or 0.0)
        ex["risk_penalty_score"] = float((gc_bd.get("risk_penalty") or {}).get("score") or 0.0)
        ex["conditioned_only"] = _stage3_is_conditioned_only(rec)
        rec["stage3_explain"] = ex

    survivors.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    survivors = _apply_family_role_constraints(survivors)
    survivors.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)

    for i, rec in enumerate(survivors):
        rec["stage3_rank"] = i
        rec["bucket_reason_flags"] = _collect_stage3_bucket_reason_flags(rec)
        rec["stage3_bucket"] = _assign_stage3_bucket(rec)

    if STAGE3_UNIFIED_SCORE_DEBUG:
        _print_stage3_unified_breakdown(survivors)
        _print_stage3_global_coherence_breakdown(survivors)
    if STAGE3_AUDIT_DEBUG:
        _print_stage3_risky_coherence_audit(survivors)

    if DEBUG_LABEL_PATH:
        print("\n" + "-" * 80)
        print("[Stage3 rerank summary] global_coherence + family role constraint (bucket post-hoc)")
        print("-" * 80)
        print(
            "rank | term | bucket | primary_bucket | cond_only | anch | ml_hits | can_exp | final"
        )
        for i, rec in enumerate(survivors[:12], start=1):
            term = rec.get("term", "")
            bucket = rec.get("stage3_bucket", "")
            pb_row = (rec.get("primary_bucket") or "")[:18]
            conditioned_only_row = _stage3_is_conditioned_only(rec)
            anchor_count_row = int(rec.get("anchor_count") or 1)
            mainline_hits_row = int(rec.get("mainline_hits") or 0)
            can_expand_row = bool(rec.get("can_expand", False))
            final_score = float(rec.get("final_score") or 0.0)
            print(
                f"{i:>4} | "
                f"{term[:22]:<22} | "
                f"{str(bucket):<6} | "
                f"{pb_row:<18} | "
                f"{str(conditioned_only_row):<9} | "
                f"{anchor_count_row:^4} | "
                f"{mainline_hits_row:^7} | "
                f"{str(can_expand_row):<7} | "
                f"{final_score:.4f}"
            )

    return survivors


def _run_stage3_dual_gate(
    recall,
    raw_candidates: List[Dict[str, Any]],
    query_vector,
    anchor_vids=None,
) -> Tuple[
    Dict[str, float],
    Dict[str, str],
    Dict[str, float],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    Stage3 主流程：去重聚合 → 全局共识 → 轻硬过滤 + identity/topic gate → 唯一主分 score_term_record
    → family 角色约束 → risky/bucket → paper_terms 选词。彻底移除 cluster 依赖。
    """
    debug_print(1, "\n" + "-" * 80 + "\n[Stage3] Global Rerank\n" + "-" * 80, recall)
    score_map: Dict[str, float] = {}
    term_map: Dict[str, str] = {}
    idf_map: Dict[str, float] = {}
    term_role_map: Dict[str, str] = {}
    term_source_map: Dict[str, str] = {}
    parent_anchor_map: Dict[str, str] = {}
    parent_primary_map: Dict[str, str] = {}
    recall.debug_info.tag_purity_debug = []
    recall._last_tag_purity_debug = recall.debug_info.tag_purity_debug
    stage3_debug = getattr(term_scoring, "STAGE3_DEBUG", False)
    label_trace = LABEL_PATH_TRACE or stage3_debug
    survivors: List[Dict[str, Any]] = []
    dropped_with_reason: List[Dict[str, Any]] = []

    _debug_print_stage3_input(raw_candidates)
    merged_candidates = _merge_stage3_duplicates(raw_candidates)
    for i, rec in enumerate(merged_candidates):
        rec["stage2_rank"] = i
    _compute_stage3_global_consensus(recall, merged_candidates)
    family_buckets = _build_family_buckets(merged_candidates)
    for rec in merged_candidates:
        rec["family_centrality"] = _compute_family_centrality(rec, family_buckets)
    merged_candidates = _classify_stage3_entry_groups(merged_candidates)

    jd_profile = getattr(recall, "jd_profile", None)
    active_domains = getattr(recall, "active_domain_set", None)
    topic_gate_bypass_primary_like_count = 0
    guardrail_reject_audit_rows: List[Dict[str, Any]] = []
    guardrail_soft_samples: List[Dict[str, Any]] = []

    for rec in merged_candidates:
        rec["family_key"] = rec.get("family_key") or build_family_key(rec)
        if not rec.get("term_role") and rec.get("term_roles"):
            roles = list(rec["term_roles"]) if isinstance(rec["term_roles"], (list, set)) else [rec["term_roles"]]
            rec["term_role"] = "primary" if "primary" in roles else (roles[0] if roles else "")
        rec["term_role"] = rec.get("term_role") or ""
        rec["retrieval_role"] = rec.get("retrieval_role") or get_retrieval_role_from_term_role(rec.get("term_role"))

        drop, reject_reason = should_drop_term(rec)
        if drop:
            rec["reject_reason"] = reject_reason
            dropped_with_reason.append(rec)
            if STAGE3_DETAIL_DEBUG or stage3_debug:
                print(f"[Stage3 硬过滤] drop tid={rec.get('tid')} term={rec.get('term')!r} reason={reject_reason}")
            continue

        gate = _check_stage3_admission(rec, jd_profile, active_domains)
        if gate.get("hard_drop"):
            rec["reject_reason"] = gate.get("reason") or "guardrail"
            rec["guardrail_flags"] = gate.get("guardrail_flags") or []
            dropped_with_reason.append(rec)
            guardrail_reject_audit_rows.append(
                {
                    "tid": rec.get("tid"),
                    "term": rec.get("term"),
                    "guardrail_reason": gate.get("guardrail_reason") or gate.get("reason"),
                    "guardrail_flags": gate.get("guardrail_flags"),
                }
            )
            if STAGE3_DETAIL_DEBUG or stage3_debug:
                print(
                    f"[Stage3 guardrail hard_drop] tid={rec.get('tid')} term={rec.get('term')!r} "
                    f"reason={rec['reject_reason']!r} flags={rec.get('guardrail_flags')!r}"
                )
            continue

        if rec.get("stage3_guardrail_soft_flags"):
            guardrail_soft_samples.append(rec)

        primary_like = _is_primary_like(rec)
        if primary_like and label_trace:
            if STAGE3_DEBUG_FOCUS_TERMS:
                _tk = (rec.get("term") or "").strip()
                if _tk in STAGE3_DEBUG_FOCUS_TERMS:
                    print(
                        f"[Stage3 topic_gate] bypass primary-like tid={rec.get('tid')} term={rec.get('term')!r}"
                    )
            else:
                topic_gate_bypass_primary_like_count += 1
        if not primary_like and not term_scoring.passes_topic_consistency(rec, None):
            rec["reject_reason"] = "topic_consistency"
            dropped_with_reason.append(rec)
            if label_trace:
                print(f"[Stage3 topic_gate] drop support tid={rec.get('tid')} term={rec.get('term')!r} reason=topic_consistency")
            continue

        rec["quality_score"] = term_scoring.score_term_expansion_quality(recall, rec)
        raw_final, explain = score_term_record(rec)
        identity_factor = term_scoring.compute_identity_factor(rec)
        risk_penalty = _compute_stage3_risk_penalty(rec)
        base_final = raw_final * identity_factor * risk_penalty
        role_factor = 1.0
        term_role = (rec.get("term_role") or "").strip().lower()
        role_in_anchor = (rec.get("role_in_anchor") or "").strip().lower()
        source_types = rec.get("source_types") or set()
        source_type_single = (rec.get("source_type") or rec.get("source") or "").strip().lower()
        if term_role == "primary" and role_in_anchor == "mainline":
            role_factor = STAGE3_MAINLINE_BOOST
        elif term_role == "dense_expansion" or "dense" in source_types or source_type_single == "dense":
            role_factor = STAGE3_DENSE_PENALTY
        elif "conditioned_vec" in source_types or source_type_single == "conditioned_vec":
            role_factor = STAGE3_CONDITIONED_PENALTY
        elif role_in_anchor == "side":
            role_factor = STAGE3_SIDE_PENALTY
        rec["final_score"] = base_final * role_factor
        explain["identity_factor"] = identity_factor
        explain["risk_penalty"] = risk_penalty
        explain["role_factor"] = role_factor
        rec["stage3_explain"] = explain
        rec["reject_reason"] = ""
        survivors.append(rec)

    _print_stage3_guardrail_reject_audit(guardrail_reject_audit_rows)
    _print_stage3_guardrail_soft_audit(guardrail_soft_samples)

    if label_trace and topic_gate_bypass_primary_like_count and not STAGE3_DEBUG_FOCUS_TERMS:
        print(
            f"[Stage3 topic_gate] bypass primary-like count={topic_gate_bypass_primary_like_count} "
            f"(明细仅 STAGE3_DEBUG_FOCUS_TERMS 非空时逐条打印)"
        )

    survivors.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    for i, rec in enumerate(survivors):
        rec["stage3_rank"] = i

    # 批注：家族 primary/support 微调挪入 stage3_build_score_map；stage3_bucket 仅在 GC 主分定稿后写入。

    survivors = stage3_build_score_map(survivors, recall)
    for rec in survivors:
        rec["risk_reasons"] = _collect_risky_reasons(rec)

    # 窄表审计：final adjust → cross-anchor → support/risky（与 STAGE3_DEBUG_FOCUS_TERMS 配合可只看定点词）
    if STAGE3_AUDIT_DEBUG:
        _print_stage3_final_adjust_audit(survivors, top_k=12)
        _print_stage3_cross_anchor_audit(survivors, top_k=40)
        _print_stage3_support_subtype_audit(survivors, top_k=24)
        _print_stage3_support_risky_concise(survivors, top_k=20)
        _print_stage3_core_miss_audit(survivors, limit=24)

    core_terms_list, support_terms_list, risky_terms_list = _bucket_stage3_terms(survivors)
    if STAGE3_BUCKET_FACTOR_DEBUG and (LABEL_PATH_TRACE or getattr(term_scoring, "STAGE3_DEBUG", False)):
        for rec in (core_terms_list + support_terms_list + risky_terms_list)[:15]:
            term = (rec.get("term") or "")[:28]
            bucket = rec.get("stage3_bucket") or ""
            identity_factor = float((rec.get("stage3_explain") or {}).get("identity_factor", 1.0) or 1.0)
            anchor_count = int(rec.get("anchor_count") or 0)
            mainline_hits = int(rec.get("mainline_hits") or 0)
            can_expand = rec.get("can_expand", False)
            reason_flags = rec.get("bucket_reason_flags") or []
            print(
                f"[Stage3 bucket reason] term={term!r} | bucket={bucket!r} | identity_factor={identity_factor:.3f} | "
                f"anchor_count={anchor_count} mainline_hits={mainline_hits} can_expand={can_expand} | reason_flags={reason_flags}"
            )
        # 排序上升/下降主导因子：便于验证 mainline_support_factor 是否罚对
        for rec in survivors[:15]:
            term = (rec.get("term") or "")[:28]
            prev_rank = rec.get("stage2_rank")
            new_rank = rec.get("stage3_rank")
            explain = rec.get("stage3_explain") or {}
            up_factors = [k for k, v in explain.items() if isinstance(v, (int, float)) and (v > 1.0 or (k == "base_score" and v > 0.3))]
            down_factors = [k for k, v in explain.items() if isinstance(v, (int, float)) and 0 < v < 1.0]
            reason_flags = rec.get("bucket_reason_flags") or []
            missing_penalty = []
            if "no_mainline_support" in reason_flags or "only_weak_keep_sources" in reason_flags or "conditioned_only" in reason_flags:
                missing_penalty.append("mainline_support_factor")
            print(
                f"[Stage3 dominant factors] term={term!r}\n"
                f"  prev_rank={prev_rank} new_rank={new_rank}\n"
                f"  up_factors={up_factors}\n"
                f"  down_factors={down_factors}\n"
                f"  missing_penalty={missing_penalty}"
            )
    top_survivors = survivors[:STAGE3_TOP_K]
    paper_terms = select_terms_for_paper_recall(
        survivors,
        PAPER_RECALL_MAX_TERMS,
        recall,
        stage3_dropped_pre_paper=dropped_with_reason,
    )

    for rec in top_survivors:
        _write_term_maps(
            score_map, term_map, idf_map, term_role_map, term_source_map,
            parent_anchor_map, parent_primary_map, recall, rec, query_vector,
        )
    for rec in paper_terms:
        _write_term_maps_if_missing(
            score_map, term_map, idf_map, term_role_map, term_source_map,
            parent_anchor_map, parent_primary_map, recall, rec, query_vector,
        )

    rerank_delta_rows = []
    for rec in survivors[:25]:
        rerank_delta_rows.append({
            "term": rec.get("term") or "",
            "stage2_rank": rec.get("stage2_rank", 0),
            "stage3_rank": rec.get("stage3_rank", 0),
            "delta": (rec.get("stage2_rank") or 0) - (rec.get("stage3_rank") or 0),
        })
    _debug_print_stage3_output(
        raw_candidates=merged_candidates,
        survivors=survivors,
        core_terms=core_terms_list,
        support_terms=support_terms_list,
        risky_terms=risky_terms_list,
        paper_terms=paper_terms,
        dropped_with_reason=dropped_with_reason,
        top_survivors=top_survivors,
        recall=recall,
        rerank_delta_rows=rerank_delta_rows,
        score_map=score_map,
        term_map=term_map,
        term_source_map=term_source_map,
        parent_anchor_map=parent_anchor_map,
        parent_primary_map=parent_primary_map,
    )
    _g = _stage3_get_candidate_graph_from_recall(recall)
    _xedges = _g.get("cross_anchor_support_edges") if isinstance(_g, dict) else None
    recall._stage3_last_gc_report = {
        "stage3_scoring_version": "global_coherence_four_block_v1",
        "final_formula": (
            "(1-α)*clamp(w_lf*lf+w_cx*cx+w_bb*bb - w_r*risk_pen,0,1) + α*legacy_unified; "
            "bucket post-hoc after final_score"
        ),
        "alpha_legacy_blend": STAGE3_GC_LEGACY_BLEND,
        "weights": {
            "local_fit": STAGE3_GC_W_LOCAL_FIT,
            "cross_anchor": STAGE3_GC_W_CROSS,
            "backbone": STAGE3_GC_W_BACKBONE,
            "risk_scale": STAGE3_GC_W_RISK,
        },
        "candidate_graph_cross_edges_non_empty": bool(isinstance(_xedges, list) and len(_xedges) > 0),
        "aggregate_fields_consumed": [
            "source_evidence_list",
            "stage2_local_meta_list",
            "anchor_support_summary",
            "provenance_summary",
            "primary_bucket_summary",
            "merge_debug_summary",
        ],
        "survivor_count": len(survivors),
    }
    if getattr(recall, "debug_info", None) is not None:
        recall.debug_info.stage3_last_gc_report = recall._stage3_last_gc_report  # type: ignore[attr-defined]
    recall.debug_info.dropped_with_reason = dropped_with_reason
    return (
        score_map,
        term_map,
        idf_map,
        term_role_map,
        term_source_map,
        parent_anchor_map,
        parent_primary_map,
        paper_terms,
        core_terms_list,
        support_terms_list,
        risky_terms_list,
    )


def _debug_print_stage3_output(
    raw_candidates: List[Dict[str, Any]],
    survivors: List[Dict[str, Any]],
    core_terms: List[Dict[str, Any]],
    support_terms: List[Dict[str, Any]],
    risky_terms: List[Dict[str, Any]],
    paper_terms: List[Dict[str, Any]],
    dropped_with_reason: List[Dict[str, Any]],
    top_survivors: List[Dict[str, Any]],
    recall: Any,
    rerank_delta_rows: List[Dict[str, Any]],
    score_map: Dict[str, float],
    term_map: Dict[str, str],
    term_source_map: Dict[str, str],
    parent_anchor_map: Dict[str, str],
    parent_primary_map: Dict[str, str],
) -> None:
    """Stage3 调试输出：新列 anchor_count, evidence_count, family_centrality, path_topic_consistency, generic_penalty, cross_anchor_factor, retrieval_role；无 cluster 列。"""
    stage3_debug = getattr(term_scoring, "STAGE3_DEBUG", False)
    label_trace = LABEL_PATH_TRACE or stage3_debug
    _debug_print_stage3_tables(raw_candidates, dropped_with_reason, survivors, recall)
    debug_print(1, f"[Stage3] 输入候选总数={len(raw_candidates)} 幸存={len(survivors)} top_k={len(top_survivors)}", recall)
    if STAGE3_OUTPUT_BREAKDOWN_DEBUG:
        debug_print(2, "[Stage3 Final Score Breakdown] term | stage2_rank | anchor_count | evidence_count | family_centrality | path_topic_consistency | generic_penalty | cross_anchor_factor | backbone_boost | object_like_penalty | bonus_term_penalty | retrieval_role | final", recall)
        for rec in survivors[:15]:
            t = (rec.get("term") or "")[:28]
            ex = rec.get("stage3_explain") or {}
            debug_print(2, (
                f"  {t:<28} | s2={rec.get('stage2_rank', 0):>2} | anc={rec.get('anchor_count', 0)} ev={rec.get('evidence_count', 0)} | "
                f"fc={float(rec.get('family_centrality') or 0):.3f} | ptc={float(ex.get('path_topic_consistency') or 0):.3f} | "
                f"gen={float(ex.get('generic_penalty') or 0):.3f} | cross={float(ex.get('cross_anchor_factor') or 0):.3f} | "
                f"bb={float(ex.get('backbone_boost') or 1):.3f} | obj={float(ex.get('object_like_penalty') or 1):.3f} | bonus={float(ex.get('bonus_term_penalty') or 1):.3f} | "
                f"{rec.get('retrieval_role', ''):14s} | final={float(rec.get('final_score') or 0):.3f}"
            ), recall)
        debug_print(2, "[Stage3 Rerank Delta] term | stage2_rank | stage3_rank | delta", recall)
        for row in rerank_delta_rows[:15]:
            t = (row.get("term") or "")[:28]
            debug_print(2, f"  {t:<28} | {row.get('stage2_rank', 0):>3} | {row.get('stage3_rank', 0):>3} | {row.get('delta', 0):+d}", recall)
        debug_print(2, "[Stage3 Buckets]", recall)
        debug_print(2, f"  core_terms={[r.get('term') for r in core_terms[:15]]}", recall)
        debug_print(2, f"  support_terms={[r.get('term') for r in support_terms[:15]]}", recall)
        debug_print(2, f"  risky_terms={[r.get('term') for r in risky_terms[:15]]}", recall)
        _debug_print_stage3_bucket_details(core_terms, support_terms, risky_terms, recall)
        debug_print(3, "[Stage3 Risky Term Reasons] term | reasons | final", recall)
        for r in risky_terms[:10]:
            t = (r.get("term") or "")[:28]
            debug_print(3, f"  {t:<28} | {r.get('risk_reasons', [])} | {r.get('final_score', 0):.3f}", recall)
    if STAGE3_OUTPUT_BREAKDOWN_DEBUG and (STAGE3_DETAIL_DEBUG or stage3_debug):
        print(f"[paper_term_selection] family 保送式 选词数={len(paper_terms)} 明细: family_key | term | term_role | retrieval_role | final_score")
        for r in paper_terms[:30]:
            print(f"  {r.get('family_key','')} | {r.get('term','')!r} | {r.get('term_role','')} | {r.get('retrieval_role','')} | {r.get('final_score',0):.4f}")
        if len(paper_terms) > 30:
            print(f"  ... 共 {len(paper_terms)} 条")
        print(f"[Stage3] 去重后={len(raw_candidates)} 幸存={len(survivors)} top_k={len(top_survivors)} paper_terms={len(paper_terms)}")
        print("[Stage3 final_score 明细] term | base_score | path_topic_consistency | generic_penalty | cross_anchor_factor | backbone_boost | object_like_penalty | bonus_term_penalty | final_score | reject_reason")
        for i, rec in enumerate(top_survivors[:20]):
            term = (rec.get("term") or "")[:28]
            ex = rec.get("stage3_explain") or {}
            base_score = ex.get("base_score") or 0.0
            path_topic = ex.get("path_topic_consistency")
            gen_pen = ex.get("generic_penalty")
            cross_anchor = ex.get("cross_anchor_factor")
            backbone_boost = ex.get("backbone_boost")
            object_like_penalty = ex.get("object_like_penalty")
            bonus_term_penalty = ex.get("bonus_term_penalty")
            final_score = rec.get("final_score", 0) or 0.0
            reject = rec.get("reject_reason", "")
            _fmt = lambda x: f"{x:.3f}" if x is not None and isinstance(x, (int, float)) else str(x) if x is not None else "-"
            print(f"  {i+1} {term!r} | base={_fmt(base_score)} | path_topic={_fmt(path_topic)} | gen={_fmt(gen_pen)} | cross={_fmt(cross_anchor)} | bb={_fmt(backbone_boost)} | obj={_fmt(object_like_penalty)} | bonus={_fmt(bonus_term_penalty)} | final={final_score:.4f} | {reject!r}")
        if len(top_survivors) > 20:
            print(f"  ... 共 {len(top_survivors)} 条")
    if STAGE3_OUTPUT_BREAKDOWN_DEBUG and stage3_debug and score_map:
        print("[stage3_output] tid | term | source_type | parent_anchor | parent_primary | score")
        for i, (tid_str, sc) in enumerate(sorted(score_map.items(), key=lambda x: -x[1])[:25]):
            print(f"  {i+1} {tid_str} | {term_map.get(tid_str, '')!r} | {term_source_map.get(tid_str, '')} | {parent_anchor_map.get(tid_str, '')!r} | {parent_primary_map.get(tid_str, '')!r} | {sc:.3f}")
    if STAGE3_OUTPUT_BREAKDOWN_DEBUG and label_trace and dropped_with_reason:
        print("[标签路-被过滤原因] tid | term | source_type | parent_anchor | 原因")
        for r in dropped_with_reason[:50]:
            print(f"  {r.get('tid')} | {r.get('term','')!r} | {r.get('source') or r.get('origin','')} | {r.get('parent_anchor','')!r} | {r.get('reject_reason','')}")
        if len(dropped_with_reason) > 50:
            print(f"  ... 共 {len(dropped_with_reason)} 条被过滤")
    if STAGE3_OUTPUT_BREAKDOWN_DEBUG and label_trace and paper_terms:
        print("[标签路-final primary 为什么胜出] tid | term | source_type | parent_anchor | retrieval_role | 胜出原因")
        for r in paper_terms[:30]:
            print(f"  {r.get('tid')} | {r.get('term','')!r} | {r.get('source') or r.get('origin','')} | {r.get('parent_anchor','')!r} | {r.get('retrieval_role','')} | {r.get('win_reason','')}")
        if len(paper_terms) > 30:
            print(f"  ... 共 {len(paper_terms)} 条")


def _stage2_candidate_graph_non_empty(candidate_graph: Dict[str, Any]) -> bool:
    """True when Stage2 graph carries at least one non-empty edge bucket (attach-only signal; scoring may still ignore it)."""
    if not candidate_graph:
        return False
    for key in (
        "same_anchor_edges",
        "cross_anchor_support_edges",
        "family_edges",
        "provenance_edges",
    ):
        edges = candidate_graph.get(key)
        if isinstance(edges, list) and len(edges) > 0:
            return True
        if isinstance(edges, dict) and len(edges) > 0:
            return True
    return False


def _legacy_selected_core_rows_from_maps(
    score_map: Dict[str, float], term_map: Dict[str, str]
) -> List[Dict[str, Any]]:
    """TODO(Stage3 contract): legacy _calculate_final_weights has no buckets; approximate core list as sorted survivors."""
    rows: List[Dict[str, Any]] = []
    for tid_str, sc in sorted(score_map.items(), key=lambda x: -float(x[1] or 0.0)):
        tid_raw: Any = tid_str
        try:
            tid_raw = int(tid_str)
        except (TypeError, ValueError):
            pass
        rows.append(
            {
                "tid": tid_raw,
                "term": term_map.get(tid_str, ""),
                "final_score": float(sc or 0.0),
            }
        )
    return rows


def run_stage3(
    recall,
    stage2_output: Dict[str, Any],
    query_vector,
    anchor_vids=None,
) -> Dict[str, Any]:
    """
    Stage3（Global Coherence Rerank）：消费 Stage2 结构化契约，返回结构化 dict。

    TODO(Stage3 contract): 本版为「壳层契约迁移」，打分/合并/准入仍以既有实现为主；
    candidate_graph / anchor_to_candidates 仅正式接入并下传，全局图重排留待后续迭代。

    入参 stage2_output 须含：all_candidates, anchor_to_candidates, candidate_graph, stage2_report。

    返回字段（稳定口径）：selected_core_terms, selected_support_terms, risky_terms,
    paper_terms, global_coherence_report；另保留 score_map / term_map / … 供主链与 Stage4 薄适配。
    """
    # --- unpack stage2 output ---
    if not isinstance(stage2_output, dict):
        raise TypeError(
            "run_stage3 expects stage2_output: dict (Stage2 contract), "
            f"got {type(stage2_output).__name__}"
        )
    _required_keys = (
        "all_candidates",
        "anchor_to_candidates",
        "candidate_graph",
        "stage2_report",
    )
    _missing = [k for k in _required_keys if k not in stage2_output]
    if _missing:
        raise KeyError(
            "stage2_output missing required Stage2 contract keys "
            f"{_missing}; required={list(_required_keys)}"
        )

    all_candidates: List[Dict[str, Any]] = stage2_output["all_candidates"]
    anchor_to_candidates: Dict[str, Any] = stage2_output["anchor_to_candidates"]
    candidate_graph: Dict[str, Any] = stage2_output["candidate_graph"]
    stage2_report: Dict[str, Any] = stage2_output["stage2_report"]

    if not isinstance(all_candidates, list):
        raise TypeError(
            f"stage2_output['all_candidates'] must be list, got {type(all_candidates).__name__}"
        )
    if not isinstance(anchor_to_candidates, dict):
        raise TypeError(
            f"stage2_output['anchor_to_candidates'] must be dict, got {type(anchor_to_candidates).__name__}"
        )
    if not isinstance(candidate_graph, dict):
        raise TypeError(
            f"stage2_output['candidate_graph'] must be dict, got {type(candidate_graph).__name__}"
        )
    if not isinstance(stage2_report, dict):
        raise TypeError(
            f"stage2_output['stage2_report'] must be dict, got {type(stage2_report).__name__}"
        )

    graph_non_empty = _stage2_candidate_graph_non_empty(candidate_graph)
    anchor_map_non_empty = bool(anchor_to_candidates)

    # 向下透传（供后续 _merge_stage3_duplicates / global rerank 使用；本版不在此消费图结构）
    _pass_through = {
        "anchor_to_candidates": anchor_to_candidates,
        "candidate_graph": candidate_graph,
        "stage2_report": stage2_report,
    }
    setattr(recall, "_stage3_stage2_context", _pass_through)
    if getattr(recall, "debug_info", None) is not None:
        recall.debug_info.stage2_contract_pass_through = _pass_through  # type: ignore[attr-defined]

    notes: List[str] = [
        "TODO(Stage3 contract): shell migration; merge/rerank still legacy unless dual_gate path.",
    ]
    candidate_count_in = len(all_candidates)

    if not all_candidates:
        empty_report = {
            "stage3_version": "contract_refactor_v1",
            "candidate_count_in": 0,
            "core_count": 0,
            "support_count": 0,
            "risky_count": 0,
            "paper_term_count": 0,
            "used_candidate_graph": graph_non_empty,
            "used_anchor_to_candidates": anchor_map_non_empty,
            "notes": notes + ["empty all_candidates"],
            "stage3_scoring_meta": None,
            "paper_lane_report": None,
        }
        return {
            "selected_core_terms": [],
            "selected_support_terms": [],
            "risky_terms": [],
            "paper_terms": [],
            "global_coherence_report": empty_report,
            "score_map": {},
            "term_map": {},
            "idf_map": {},
            "term_role_map": {},
            "term_source_map": {},
            "parent_anchor_map": {},
            "parent_primary_map": {},
        }

    # --- prepare candidates for stage3 ---
    raw_candidates = all_candidates

    use_dual_gate = bool(
        raw_candidates
        and ("term_role" in raw_candidates[0] or "identity_score" in raw_candidates[0])
    )
    selected_core_terms: List[Dict[str, Any]] = []
    selected_support_terms: List[Dict[str, Any]] = []
    risky_terms: List[Dict[str, Any]] = []
    paper_terms: List[Dict[str, Any]] = []

    # --- merge / filter / rerank with existing logic (temporary) ---
    if use_dual_gate:
        (
            score_map,
            term_map,
            idf_map,
            term_role_map,
            term_source_map,
            parent_anchor_map,
            parent_primary_map,
            paper_terms,
            selected_core_terms,
            selected_support_terms,
            risky_terms,
        ) = _run_stage3_dual_gate(recall, raw_candidates, query_vector, anchor_vids=anchor_vids)
    else:
        recall._stage3_last_gc_report = None
        score_map, term_map, idf_map = recall._calculate_final_weights(
            raw_candidates, query_vector, anchor_vids=anchor_vids
        )
        term_role_map = {}
        term_source_map = {}
        parent_anchor_map = {}
        parent_primary_map = {}
        # --- select core / support / risky（legacy：无 bucket，core≈全部加权幸存者，support/risky 空）---
        selected_core_terms = _legacy_selected_core_rows_from_maps(score_map, term_map)
        selected_support_terms = []
        risky_terms = []
        notes.append("legacy_path:_calculate_final_weights_no_stage3_buckets")

    # verbose 调试：Stage3 已移除 cluster 列，改用 anchor_count / evidence_count / family_centrality / path_topic_consistency / generic_penalty / cross_anchor_factor / retrieval_role
    if STAGE3_OBSERVABILITY_PANEL_DEBUG and recall.verbose and score_map and getattr(recall, "_last_tag_purity_debug", None):
        debug_by_tid = {str(d["tid"]): d for d in recall.debug_info.tag_purity_debug if d.get("tid") is not None}
        top_tids = sorted(score_map.keys(), key=lambda t: score_map.get(t, 0.0), reverse=True)

        def _fv(d, key, default=None):
            v = d.get(key, default)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return v

        def _f(v, w=6, ndec=4):
            if v is None:
                return "-".rjust(w)
            if isinstance(v, (int, float)):
                return f"{float(v):.{min(ndec, w-2)}f}".rjust(w)
            return str(v)[:w].rjust(w)

        head30 = top_tids[:30]
        lines = ["【Step 3 调试】Top20 学术词 | cos_sim(JD) | anchor_sim | final_weight"]
        for i, tid in enumerate(head30[:20], 1):
            term = term_map.get(tid, "")
            w = score_map.get(tid, 0.0)
            d = debug_by_tid.get(str(tid), {})
            cs = d.get("cos_sim")
            ans = d.get("anchor_sim")
            cs_s = f"{cs:.4f}" if cs is not None else "-"
            ans_s = f"{ans:.4f}" if ans is not None else "-"
            lines.append(f"  #{i:2d}  {term[:36]:36s}  cos={cs_s:6s}  anchor={ans_s:6s}  weight={w:.6f}")
        print("\n".join(lines))

        print("\n【观测面板 Stage3】Top30: term | anchor_count | evidence_count | family_centrality | path_topic_consistency | generic_penalty | cross_anchor_factor | retrieval_role | final_score")
        for i, tid in enumerate(head30, 1):
            d = debug_by_tid.get(str(tid), {})
            term = (d.get("term") or term_map.get(tid, ""))[:24]
            anc = d.get("anchor_count")
            ev = d.get("evidence_count")
            fc = _f(d.get("family_centrality"), 6)
            ptc = _f(d.get("path_topic_consistency"), 6)
            gen = _f(d.get("generic_penalty"), 6)
            cross = _f(d.get("cross_anchor_factor"), 6)
            role = (d.get("retrieval_role") or "")[:14]
            final_score = score_map.get(tid, 0.0)
            print(f"  #{i:2d}  {term:24s}  anc={anc} ev={ev}  fc={fc} ptc={ptc} gen={gen} cross={cross}  {role:14s}  {final_score:.6f}")

        print("\n【观测面板 汇总】Top30: term | base_score | path_topic_consistency | generic_penalty | cross_anchor_factor | final_score")
        for i, tid in enumerate(head30, 1):
            d = debug_by_tid.get(str(tid), {})
            term = (d.get("term") or term_map.get(tid, ""))[:24]
            print(
                f"  #{i:2d}  {term:24s}  base={_f(d.get('base_score'),7):>7}  ptc={_f(d.get('path_topic_consistency'),6):>6}  "
                f"gen={_f(d.get('generic_penalty'),6):>6}  cross={_f(d.get('cross_anchor_factor'),6):>6}  final={score_map.get(tid, 0.0):.6f}"
            )

    # --- build global_coherence_report & structured return ---
    _paper_rep = getattr(getattr(recall, "debug_info", None), "paper_lane_report", None)
    global_coherence_report: Dict[str, Any] = {
        "stage3_version": "contract_refactor_v1",
        "candidate_count_in": candidate_count_in,
        "core_count": len(selected_core_terms),
        "support_count": len(selected_support_terms),
        "risky_count": len(risky_terms),
        "paper_term_count": len(paper_terms),
        "used_candidate_graph": graph_non_empty,
        "used_anchor_to_candidates": anchor_map_non_empty,
        "notes": notes,
        "stage3_scoring_meta": getattr(recall, "_stage3_last_gc_report", None),
        "paper_lane_report": _paper_rep,
    }
    return {
        "selected_core_terms": selected_core_terms,
        "selected_support_terms": selected_support_terms,
        "risky_terms": risky_terms,
        "paper_terms": paper_terms,
        "global_coherence_report": global_coherence_report,
        "score_map": score_map,
        "term_map": term_map,
        "idf_map": idf_map,
        "term_role_map": term_role_map,
        "term_source_map": term_source_map,
        "parent_anchor_map": parent_anchor_map,
        "parent_primary_map": parent_primary_map,
    }

