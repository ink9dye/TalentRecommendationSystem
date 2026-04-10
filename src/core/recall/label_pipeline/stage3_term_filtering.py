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
from src.core.recall.label_pipeline.stage4_prep_bridge import prepare_stage4_terms_from_stage3

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
STAGE3_UNIFIED_SCORE_DEBUG = True  # 连续分特征拆解（替代原「规则乘子」视角）
# 默认只打 TopN（详细诊断时与 Stage4 prep 日志配合；unified 仅辅）
STAGE3_UNIFIED_SCORE_DEBUG_TOP_K = 3
# Stage3 窄表审计：final adjust / cross-anchor / support·risky（与 bucket reason 互补）
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
    "Movement control",
    "movement control",
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

# term-level aggregation（Step2）：正式壳标记；normalize 合并分支据此识别
STAGE3_AGGREGATE_SOURCE_V2 = "evidence_aggregate_v2"
STAGE3_AGGREGATE_SOURCE_MARKERS = frozenset({"evidence_aggregate_v1", STAGE3_AGGREGATE_SOURCE_V2})
STAGE3_AGGREGATION_DEBUG = True
STAGE3_AGGREGATION_DEBUG_MAX = 12
# Legacy bucket observability is compat/debug-only. Keep default OFF so it does not dominate Stage3 main logs.
STAGE3_LEGACY_BUCKET_AUDIT = False

# Stage3 入口契约：normalize 可观测性（仅日志；不影响打分）
STAGE3_INPUT_NORMALIZE_DEBUG = True
STAGE3_INPUT_NORMALIZE_DEBUG_MAX = 12
# 用于 legacy_fields_present：观测仍残留在顶层的「临时兼容镜像」键（非穷举所有顶层键）
STAGE3_LEGACY_MIRROR_KEYS_FOR_OBS = frozenset(
    {
        "primary_bucket",
        "primary_reason",
        "fallback_primary",
        "parent_primary",
        "parent_anchor_step2_rank",
        "parent_anchor_final_score",
        "can_expand",
        "can_expand_from_2a",
        "can_expand_local",
        "term_role",
        "anchor_internal_rank",
        "survive_primary",
        "admission_reason",
        "reject_reason",
        "stage2b_seed_tier",
        "mainline_candidate",
        "role_in_anchor",
    }
)


def _stage3_infer_local_role_from_primary_bucket(primary_bucket: str) -> str:
    """
    LEGACY COMPAT ONLY.

    说明：`primary_bucket` 属于 Stage2 的 provisional/legacy 半裁决语义（应下沉到 `stage2_debug_meta`）。
    Stage3 主链（normalize/aggregate/classify/guardrail/GC/unified/bucket/rerank）不得再以它作为正式输入语义。

    该函数仅用于旧数据/离线回放兼容场景的映射兜底；主链应优先依赖稳定字段（local_role/can_expand_local/mainline_* 等）。
    """
    pb = (primary_bucket or "").strip().lower()
    if pb in ("primary_expandable", "primary_keep_no_expand", "primary_fallback_keep_no_expand"):
        return "local_core"
    if pb in ("primary_support_seed", "primary_support_keep"):
        return "local_support"
    if pb in ("risky_keep", "risky"):
        return "local_risky"
    if "support" in pb:
        return "local_support"
    return "local_risky"


def _normalize_stage3_raw_stage2_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    单条 Stage2 raw candidate → Stage3 统一消费形状（就地补字段；不删兼容键）。

    读取顺序：stage2_local_meta → legacy 顶层 → 默认值。
    """
    meta: Dict[str, Any] = rec.get("stage2_local_meta") if isinstance(rec.get("stage2_local_meta"), dict) else {}

    def _f(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    pa_score = meta.get("parent_anchor_final_score")
    if pa_score is None:
        pa_score = rec.get("parent_anchor_final_score")
    best_pa = _f(pa_score, 0.0)

    pa_rank = meta.get("parent_anchor_step2_rank")
    if pa_rank is None:
        pa_rank = rec.get("parent_anchor_step2_rank")
    best_rk: Optional[int]
    try:
        best_rk = int(pa_rank) if pa_rank is not None and int(pa_rank) > 0 else None
    except (TypeError, ValueError):
        best_rk = None

    cel = meta.get("can_expand_local")
    if cel is None:
        cel = rec.get("can_expand_local")
    if cel is None:
        cel = bool(rec.get("can_expand_from_2a", False))
    can_expand_local = bool(cel)

    ria = meta.get("role_in_anchor")
    if ria is None:
        ria = rec.get("role_in_anchor")
    role_in_anchor = str(ria).strip() if ria is not None else ""

    mlc = meta.get("mainline_candidate")
    if mlc is None:
        mlc = rec.get("mainline_candidate")
    mainline_candidate = bool(mlc)

    hfe = meta.get("has_family_evidence")
    if hfe is None:
        hfe = rec.get("has_family_evidence")
    has_family_evidence = bool(hfe)

    seed_block = meta.get("seed_block_reason")
    if seed_block is None:
        seed_block = rec.get("seed_block_reason")

    jd_raw = rec.get("jd_candidate_alignment")
    if jd_raw is None:
        jd_raw = rec.get("jd_align")
    jd_align = max(0.0, min(1.0, _f(jd_raw if jd_raw is not None else 0.5, 0.5)))
    rec["jd_align"] = jd_align
    rec["jd_candidate_alignment"] = jd_align

    cs = rec.get("candidate_source")
    if cs is None or (isinstance(cs, str) and not str(cs).strip()):
        cs = rec.get("source") or rec.get("origin") or ""
    rec["candidate_source"] = cs

    ident = rec.get("identity_score")
    if ident is None:
        ident = rec.get("best_identity_score")
    rec["identity_score"] = _f(ident, 0.0)

    sim = rec.get("sim_score")
    if sim is None:
        sim = rec.get("semantic_score")
    if sim is not None:
        rec["sim_score"] = _f(sim, 0.0)

    # term_role：Stage3 主链的统一入口字段；优先读 Stage2 新契约字段 term_role_local
    if not (rec.get("term_role") or "").strip():
        trl = (rec.get("term_role_local") or "").strip()
        if trl:
            rec["term_role"] = trl
    tr = (rec.get("term_role") or "").strip()
    rec["retrieval_role"] = (rec.get("retrieval_role") or "").strip() or get_retrieval_role_from_term_role(tr)

    # local_role：Stage3 主链稳定入口字段（来自 Stage2 正式契约）。
    # 注意：不得再从 primary_bucket/primary_reason/fallback_primary 等 legacy 字段“反推”来驱动 Stage3 主裁决；
    # 若缺失，一律保守回落为 local_risky（避免误把未知提升为 core/support）。
    lr = (rec.get("local_role") or "").strip()
    rec["local_role"] = lr or "local_risky"

    rec["best_parent_anchor_final_score"] = best_pa
    rec["best_parent_anchor_step2_rank"] = best_rk
    rec["can_expand_local"] = can_expand_local
    rec["role_in_anchor"] = role_in_anchor
    rec["mainline_candidate"] = mainline_candidate
    rec["has_family_evidence"] = has_family_evidence
    rec["seed_block_reason"] = seed_block

    # legacy half-decision 字段：仅从 stage2_debug_meta 做兼容映射（Stage3 主链勿依赖其原始顶层镜像）
    dbg = rec.get("stage2_debug_meta") if isinstance(rec.get("stage2_debug_meta"), dict) else {}
    if not (rec.get("parent_primary") or "").strip():
        pp = dbg.get("parent_primary")
        if pp is not None:
            rec["parent_primary"] = str(pp)
    rec["stage3_input_schema_version"] = "v1"
    rec["stage3_record_normalized"] = True

    # --- Contract cut: prevent Stage3 mainline from accidentally consuming legacy/provisional keys ---
    # Legacy keys may still exist in `stage2_debug_meta` for compat/debug/Stage4 prep post-layer.
    for k in (
        "primary_bucket",
        "fallback_primary",
        "survive_primary",
        "admission_reason",
        "reject_reason",
        "stage2b_seed_tier",
        "primary_reason",
    ):
        rec.pop(k, None)
    return rec


def _normalize_stage3_merged_shell_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    tid 合并后的 shell：补全 normalize 标记与统一口径字段（merge 已写 best_* / can_expand_* 多数场景）。
    """
    try:
        jd = float(rec.get("best_jd_align") or rec.get("jd_candidate_alignment") or rec.get("jd_align") or 0.0)
    except (TypeError, ValueError):
        jd = 0.0
    jd = max(0.0, min(1.0, jd))
    rec["jd_align"] = jd
    rec["jd_candidate_alignment"] = jd

    st = rec.get("source_types")
    if isinstance(st, (list, tuple)) and st:
        rec["candidate_source"] = str(st[0])
    elif not (rec.get("candidate_source") or "").strip():
        rec["candidate_source"] = str(rec.get("source_type") or rec.get("source") or "")

    tr = (rec.get("term_role") or "").strip()
    rec["retrieval_role"] = (rec.get("retrieval_role") or "").strip() or get_retrieval_role_from_term_role(tr)

    # merged shell 的 local_role 由 aggregation 层写入；这里不允许从 legacy primary_bucket 反推。
    if not (rec.get("local_role") or "").strip():
        rec["local_role"] = "local_risky"

    meta_list = [m for m in (rec.get("stage2_local_meta_list") or []) if isinstance(m, dict)]
    has_fam = any(bool(m.get("has_family_evidence")) for m in meta_list)
    rec["has_family_evidence"] = bool(has_fam or rec.get("has_family_evidence"))

    sbr = None
    for m in meta_list:
        if m.get("seed_block_reason") is not None:
            sbr = m.get("seed_block_reason")
            break
    if sbr is None:
        lm = rec.get("stage2_local_meta")
        if isinstance(lm, dict):
            sbr = lm.get("seed_block_reason")
    rec["seed_block_reason"] = sbr

    if rec.get("best_parent_anchor_final_score") is None:
        rec["best_parent_anchor_final_score"] = float(rec.get("parent_anchor_final_score") or 0.0)
    if rec.get("best_parent_anchor_step2_rank") is None and rec.get("parent_anchor_step2_rank") is not None:
        try:
            rk = int(rec.get("parent_anchor_step2_rank"))
            rec["best_parent_anchor_step2_rank"] = rk if rk > 0 else None
        except (TypeError, ValueError):
            pass

    rec["stage3_input_schema_version"] = "v1"
    rec["stage3_record_normalized"] = True

    # Same contract cut for merged records (debug meta stays; top-level legacy mirrors removed).
    for k in (
        "primary_bucket",
        "fallback_primary",
        "survive_primary",
        "admission_reason",
        "reject_reason",
        "stage2b_seed_tier",
        "primary_reason",
    ):
        rec.pop(k, None)
    return rec


def _normalize_stage3_input_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Stage3 入口：单条 raw 或 merged shell 统一补全（主链优先读此处写入的字段）。"""
    if rec.get("stage3_merge_source") in STAGE3_AGGREGATE_SOURCE_MARKERS:
        return _normalize_stage3_merged_shell_record(rec)
    return _normalize_stage3_raw_stage2_record(rec)


def _normalize_stage3_input_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for rec in records:
        _normalize_stage3_input_record(rec)
    return records


def _print_stage3_input_normalize_debug(records: List[Dict[str, Any]], *, tag: str) -> None:
    """轻量验收日志：normalize 后前若干条（debug/audit；非主序分）。"""
    if not STAGE3_INPUT_NORMALIZE_DEBUG or not records:
        return
    n = min(int(STAGE3_INPUT_NORMALIZE_DEBUG_MAX), len(records))
    print("\n" + "-" * 80)
    print(f"[Stage3 input normalize] {tag} (showing {n}/{len(records)})")
    print("-" * 80)
    for rec in records[:n]:
        legacy_fields_seen_top_level = sorted(k for k in STAGE3_LEGACY_MIRROR_KEYS_FOR_OBS if k in rec)
        dbg = rec.get("stage2_debug_meta") if isinstance(rec.get("stage2_debug_meta"), dict) else {}
        legacy_keys_in_debug_meta = sorted(
            k for k in STAGE3_LEGACY_MIRROR_KEYS_FOR_OBS if k in dbg and dbg.get(k) is not None
        )
        print(
            f"  term={rec.get('term')!r} stage3_record_normalized={rec.get('stage3_record_normalized')!r} "
            f"schema_version={rec.get('stage3_input_schema_version')!r} "
            f"local_role={rec.get('local_role')!r} candidate_source={rec.get('candidate_source')!r} "
            f"best_parent_anchor_final_score={rec.get('best_parent_anchor_final_score')!r} "
            f"best_parent_anchor_step2_rank={rec.get('best_parent_anchor_step2_rank')!r} "
            f"can_expand_local={rec.get('can_expand_local')!r} "
            f"mainline_candidate={rec.get('mainline_candidate')!r} "
            f"legacy_fields_seen_top_level={legacy_fields_seen_top_level} "
            f"legacy_keys_in_stage2_debug_meta={legacy_keys_in_debug_meta}"
        )
    print("-" * 80 + "\n")


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
    """局部贴合：优先消费聚合层 `aggregation_*` / evidence_count；否则退回扫 source_evidence_list。"""
    notes: List[str] = []
    ex = rec.get("stage3_explain") or {}
    evid = list(rec.get("source_evidence_list") or [])
    if rec.get("stage3_aggregated"):
        mean_id = float(rec.get("aggregation_identity_mean") or 0.0)
        max_id = float(rec.get("aggregation_identity_max") or mean_id)
        evid_rows = int(rec.get("evidence_count") or len(evid))
    else:
        identities: List[float] = []
        for e in evid:
            if not isinstance(e, dict):
                continue
            identities.append(float(e.get("identity_score") or e.get("sim_score") or 0.0))
        if not identities:
            identities = [float(rec.get("best_identity_score") or rec.get("identity_score") or 0.0)]
        mean_id = sum(identities) / len(identities)
        max_id = max(identities)
        evid_rows = len(evid)
    seed = _stage3_clamp01(rec.get("best_seed_score") or rec.get("score"))
    jd = _stage3_clamp01(rec.get("best_jd_align"))
    ptc = _stage3_clamp01(ex.get("path_topic_consistency"))
    qual = _stage3_clamp01(rec.get("quality_score"))
    fc = _stage3_clamp01(rec.get("family_centrality") or ex.get("family_centrality"))
    # Stage3 主链不得再消费 Stage2 legacy/provisional 的 bucket/fallback/reason 语义做裁决或打分乘子；
    # 这里用稳定字段 local_role 提供极弱先验（可解释、跨阶段稳定）。
    lr = (rec.get("local_role") or "").strip().lower()
    local_role_prior = 0.03 if lr == "local_core" else (0.01 if lr == "local_support" else 0.0)
    core_fit = (
        0.20 * _stage3_clamp01(mean_id)
        + 0.16 * _stage3_clamp01(max_id)
        + 0.14 * seed
        + 0.16 * jd
        + 0.16 * ptc
        + 0.10 * qual
        + 0.08 * fc
    )
    score = _stage3_clamp01(core_fit + local_role_prior)
    if evid_rows >= 2:
        score = _stage3_clamp01(score + 0.02)
        notes.append("multi_source_evidence:+0.02")
    notes.append("local_role_prior_weak:max+0.03")
    return {
        "score": score,
        "components": {
            "evidence_rows": evid_rows,
            "identity_mean": mean_id,
            "identity_max": max_id,
            "seed_norm": seed,
            "jd_align": jd,
            "path_topic_consistency": ptc,
            "quality_score": qual,
            "family_centrality": fc,
            "local_role": lr,
            "local_role_prior_applied": local_role_prior,
        },
        "notes": notes,
    }


def _compute_stage3_cross_anchor_coherence(rec: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    """跨锚一致性：优先顶层聚合标量 / anchor_support_summary；merge_debug 取 cross_anchor_side_only。"""
    notes: List[str] = []
    adb = rec.get("anchor_support_summary") or {}
    mdbg = rec.get("merge_debug_summary") or {}
    ac = int(rec.get("anchor_count") or adb.get("anchor_count") or 1)
    mac = int(rec.get("mainline_anchor_count") or adb.get("mainline_anchor_count") or 0)
    eac = int(rec.get("expandable_anchor_count") or adb.get("expandable_anchor_count") or 0)
    sac_only = int(rec.get("side_anchor_count") or adb.get("side_anchor_count") or 0)
    if rec.get("stage3_aggregated"):
        side_only_global = bool(rec.get("cross_anchor_side_only"))
    else:
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
    if rec.get("stage3_aggregated"):
        exp_local_frac = float(rec.get("aggregation_expand_local_fraction") or 0.0)
    else:
        evid = list(rec.get("source_evidence_list") or [])
        exp_local_frac = (
            sum(1 for e in evid if isinstance(e, dict) and e.get("can_expand_local")) / max(1, len(evid))
        )
    # 主链：主轴可扩性以 normalize 后 can_expand_local 为先，legacy 仅兼容（audit/merge summary 仍可直接读顶层）
    can_ex = bool(rec.get("can_expand_local") or rec.get("can_expand") or rec.get("can_expand_from_2a"))
    try:
        pas_raw = float(
            rec.get("best_parent_anchor_final_score") or rec.get("parent_anchor_final_score") or 0.0
        )
    except (TypeError, ValueError):
        pas_raw = 0.0
    a_sc = _stage3_clamp01(pas_raw / 1.2)
    rk = rec.get("best_parent_anchor_step2_rank")
    if rk is None:
        rk = rec.get("parent_anchor_step2_rank")
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
    # 结构性负向：从 backbone 扣减等价并入 risk_penalty（GC 线性式下 w_bb·Δbb = w_r·Δr，主排序解释统一走 lf/cx/bb/risk）
    migrated_penalty_bb_equiv = 0.0
    if _stage3_is_conditioned_only(rec):
        migrated_penalty_bb_equiv += 0.18
        notes.append("conditioned_only:migrated_to_risk_penalty")
    flags = rec.get("bucket_reason_flags") or []
    if "locked_mainline_no_expand" in flags:
        migrated_penalty_bb_equiv += 0.08
        notes.append("locked_mainline_no_expand:migrated_to_risk_penalty")
    if "cross_anchor_but_side_only" in flags:
        migrated_penalty_bb_equiv += 0.12
        notes.append("cross_anchor_but_side_only:migrated_to_risk_penalty")
    bb = _stage3_clamp01(bb)
    return {
        "score": bb,
        "migrated_penalty_bb_equiv": float(migrated_penalty_bb_equiv),
        "components": {
            "mainline_hits": mh,
            "expandable_support": float(can_ex),
            "expand_local_fraction": exp_local_frac,
            "anchor_score_prior": a_sc,
            "anchor_rank_prior": rank_prior,
            "best_parent_anchor_step2_rank": rk_i,
            "migrated_penalty_bb_equiv": float(migrated_penalty_bb_equiv),
        },
        "notes": notes,
    }


def _compute_stage3_risk_penalty_dimension(
    rec: Dict[str, Any],
    *,
    migrated_bb_penalty_equiv: float = 0.0,
) -> Dict[str, Any]:
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

    ca_so = rec.get("cross_anchor_side_only") if rec.get("stage3_aggregated") else None
    if ca_so is None:
        ca_so = mdbg.get("cross_anchor_side_only")
    ca_side_pen = 0.22 if ca_so else 0.0
    single_path_pen = 0.12 if prov.get("single_path_multi_anchor_weak_signal") else 0.0
    if prov.get("single_path_only") and int(prov.get("distinct_source_type_count") or 0) <= 1:
        single_path_pen += 0.06
        notes.append("single_path_narrow_provenance")

    mig_bb = max(0.0, float(migrated_bb_penalty_equiv or 0.0))
    w_bb = float(STAGE3_GC_W_BACKBONE)
    w_r = max(float(STAGE3_GC_W_RISK), 1e-9)
    migrated_into_risk_units = (w_bb / w_r) * mig_bb
    if mig_bb > 1e-12:
        notes.append(f"backbone_struct_migrated:+{migrated_into_risk_units:.4f}_risk_units")

    score = _stage3_clamp01(
        0.20 * gen
        + 0.18 * poly
        + 0.18 * obj
        + 0.16 * drift
        + conflict_penalty
        + ca_side_pen
        + single_path_pen
        + migrated_into_risk_units
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
            "migrated_from_backbone_bb_equiv": mig_bb,
            "migrated_into_risk_units": migrated_into_risk_units,
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
    mig_bb = float(bb.get("migrated_penalty_bb_equiv") or 0.0)
    rk = _compute_stage3_risk_penalty_dimension(rec, migrated_bb_penalty_equiv=mig_bb)

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
    print(
        "[Stage3 unified score breakdown] legacy 连续分项 + α blend；主排序以 GC 四分项为准；"
        "bucket / family_role 仅后置标注"
    )
    print("-" * 80)
    for rec in rows:
        bd = rec.get("stage3_unified_breakdown") or {}
        term = (rec.get("term") or "")[:28]
        sb = rec.get("stage3_score_breakdown") or {}
        gcp = float(sb.get("gc_pre_blend") or 0.0)
        fam = sb.get("family_role_post_adjust") if isinstance(sb, dict) else None
        fam_s = ""
        if isinstance(fam, dict) and fam.get("mult") is not None:
            fam_s = f" fam_mult={float(fam.get('mult') or 1.0):.3f}({fam.get('note')})"
        print(
            f"term={term!r} | seed={bd.get('seed_score', 0):.3f} anchor_id={bd.get('anchor_identity', 0):.3f} "
            f"jd_align={bd.get('jd_align', 0):.3f} fam_cent={bd.get('family_centrality', 0):.3f} "
            f"cross={bd.get('cross_anchor_strength', 0):.3f} mainline={bd.get('mainline_strength', 0):.3f} "
            f"expand={bd.get('expandability', 0):.3f}"
        )
        print(
            f"  risks: drift={bd.get('semantic_drift_risk', 0):.3f} generic={bd.get('generic_risk', 0):.3f} "
            f"poly={bd.get('polysemy_risk', 0):.3f} obj={bd.get('object_like_risk', 0):.3f} | "
            f"gc_pre={gcp:.3f} final={float(rec.get('final_score') or 0):.4f} "
            f"bucket={rec.get('stage3_bucket')!r} (仅观测){fam_s}"
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
        rpc = (sb.get("risk_penalty") or {}).get("components") or {}
        mig_bb = float(rpc.get("migrated_from_backbone_bb_equiv") or 0.0)
        if mig_bb > 1e-9:
            print(
                f"  risk_block: backbone_struct_migrated bb_equiv={mig_bb:.3f} "
                f"-> +{float(rpc.get('migrated_into_risk_units') or 0.0):.4f} risk_units"
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



# --- Stage3 merge：multi-anchor evidence aggregation（非 winner-snapshot）---
def _stage3_rec_bucket_norm(rec: Dict[str, Any]) -> str:
    meta = rec.get("stage2_local_meta") or {}
    if meta.get("primary_bucket"):
        return str(meta.get("primary_bucket") or "").strip().lower()
    dbg = rec.get("stage2_debug_meta") or {}
    if isinstance(dbg, dict) and dbg.get("primary_bucket"):
        return str(dbg.get("primary_bucket") or "").strip().lower()
    return (rec.get("primary_bucket") or "").strip().lower()


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


def _stage3_compute_best_signal_mixed(records: List[Dict[str, Any]]) -> bool:
    """
    各 best_* 维度若由不同 raw 行取得最优值则为 True（聚合层可观测；不改变取 max 规则）。
    """
    if len(records) <= 1:
        return False

    def _idx_argmax(getter) -> int:
        best_i = 0
        best_v: Optional[float] = None
        for i, r in enumerate(records):
            v = getter(r)
            try:
                vf = float(v)
            except (TypeError, ValueError):
                vf = float("-inf")
            if best_v is None or vf > best_v:
                best_v = vf
                best_i = i
        return best_i

    def _score(r: Dict[str, Any]) -> float:
        try:
            return float(r.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _jd(r: Dict[str, Any]) -> float:
        try:
            return float(r.get("jd_candidate_alignment") or r.get("jd_align") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _ident(r: Dict[str, Any]) -> float:
        try:
            return float(r.get("identity_score") or r.get("sim_score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _pa(r: Dict[str, Any]) -> float:
        try:
            return float(r.get("best_parent_anchor_final_score") or r.get("parent_anchor_final_score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _rk(r: Dict[str, Any]) -> float:
        """越大越好：等价于 rank 越小越好。"""
        try:
            x = r.get("best_parent_anchor_step2_rank")
            if x is None:
                x = r.get("parent_anchor_step2_rank")
            v = int(x) if x is not None else 999
            if v <= 0:
                v = 999
            return float(-v)
        except (TypeError, ValueError):
            return float(-999)

    idxs = {
        _idx_argmax(_score),
        _idx_argmax(_jd),
        _idx_argmax(_ident),
        _idx_argmax(_pa),
        _idx_argmax(_rk),
    }
    return len(idxs) > 1


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
    # can_expand 语义统一收敛到 can_expand_local（Stage2 侧已瘦身；Stage3 merge 不再抬回 can_expand/can_expand_from_2a 顶层镜像）
    expands = [bool(r.get("can_expand_local")) for r in records]
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
        "best_signal_mixed_sources": _stage3_compute_best_signal_mixed(records),
        "risk_flags_union": sorted(str(x) for x in risk_union),
        "legacy_compatibility_mirror_fields": [
            # 仅用于观测：哪些旧语义“可能仍存在于 raw record 顶层”。
            # 注意：term-level merged record 顶层不再抬回 can_expand/can_expand_from_2a/fallback_primary/primary_reason 等镜像。
            "primary_bucket",
            "term_role",
            "role_in_anchor",
            "can_expand_local",
            "mainline_candidate",
            "parent_primary",
        ],
        "note": "Stage3 merge: 主链应消费 *_summary / *_list；避免 term-level merged record 再制造 legacy mirror 顶层键。",
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
    单 tid 多来源 → 一条 term-level aggregated record。

    分层（注释即契约）：
    - 原始证据：`records` / `source_evidence_list` / `stage2_local_meta_list`
    - 正式聚合字段：`primary_bucket_summary` / `anchor_support_summary` / `provenance_summary` /
      `merge_debug_summary` / `best_*` / `anchor_count` / `mainline_anchor_count` / `multi_source` 等
    - legacy mirror：仅允许保留极少数为了兼容/桥接仍需的字段；其余不再在 term-level merged record 顶层抬回。
      正式语义以 summary + 顶层聚合标量为准。
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

    can_expand_local = any(bool(r.get("can_expand_local")) for r in records)
    mainline_candidate = any(bool(r.get("mainline_candidate")) for r in records)
    # local_role：Stage3 主链稳定语义（来自 Stage2 正式契约字段 local_role），聚合为 term-level 主视角
    # 规则：local_core > local_support > local_risky（保守：缺失一律视为 risky）
    lr_cnt: Counter = Counter()
    for r in records:
        lr = str(r.get("local_role") or "").strip().lower() or "local_risky"
        lr_cnt[lr] += 1
    local_role_distribution = dict(sorted(lr_cnt.items(), key=lambda x: (-x[1], x[0])))
    if lr_cnt:
        local_role_derived = max(
            lr_cnt.keys(),
            key=lambda k: ({"local_core": 3, "local_support": 2, "local_risky": 1}.get(k, 0), lr_cnt[k]),
        )
    else:
        local_role_derived = "local_risky"

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

    idents = [float(r.get("identity_score") or r.get("sim_score") or 0.0) for r in records]
    obj["aggregation_identity_mean"] = sum(idents) / len(idents) if idents else 0.0
    obj["aggregation_identity_max"] = max(idents) if idents else 0.0
    n_exp = sum(1 for r in records if bool(r.get("can_expand_local")))
    obj["aggregation_expand_local_fraction"] = float(n_exp) / float(max(1, len(records)))

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

    # 正式 bucket 语义统一写入 primary_bucket_summary.derived_primary_bucket；
    # 为避免 Stage3 在 merged_post_merge 阶段“二次抬回顶层 legacy mirror”，不再写顶层 primary_bucket 镜像。
    obj["term_role"] = derived_term_role
    obj["role_in_anchor"] = derived_role_in_anchor
    obj["can_expand_local"] = can_expand_local
    obj["mainline_candidate"] = mainline_candidate
    obj["local_role"] = local_role_derived
    obj["local_role_distribution"] = local_role_distribution
    obj["has_primary_role"] = hp
    obj["has_support_role"] = hs
    obj["stage3_merge_source"] = STAGE3_AGGREGATE_SOURCE_V2
    obj["stage3_aggregated"] = True
    obj["stage3_aggregation_schema_version"] = "v2"

    # 聚合层一次写齐：主链 / GC 优先读顶层与 summary，避免再扫 source_evidence_list
    obj["mainline_anchor_count"] = int(anchor_sup.get("mainline_anchor_count") or 0)
    obj["side_anchor_count"] = int(anchor_sup.get("side_anchor_count") or 0)
    obj["expandable_anchor_count"] = int(anchor_sup.get("expandable_anchor_count") or 0)
    obj["multi_source"] = bool(prov.get("multi_source"))
    obj["source_type_counts"] = dict(prov.get("source_type_counts") or {})
    pas_sorted = list(obj.get("parent_anchors") or [])
    obj["parent_anchor_set"] = list(pas_sorted)

    obj["primary_bucket_conflict"] = bool(dbg.get("primary_bucket_conflict"))
    obj["term_role_conflict"] = bool(dbg.get("term_role_conflict"))
    obj["role_in_anchor_conflict"] = bool(dbg.get("role_in_anchor_conflict"))
    obj["mainline_side_conflict"] = bool(dbg.get("mainline_side_conflict"))
    obj["cross_anchor_side_only"] = bool(dbg.get("cross_anchor_side_only"))

    # 不再抬回 parent_anchor_final_score / parent_anchor_step2_rank 这类旧镜像；统一用 best_parent_anchor_*。

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
    # 兼容：term-level merged record 收纳一条 stage2_debug_meta（供后置 Stage4 prep 兼容注入；不参与 Stage3 主链打分）
    legacy_dbg: Dict[str, Any] = {}
    for r in records:
        d = r.get("stage2_debug_meta")
        if isinstance(d, dict) and d:
            legacy_dbg = dict(d)
            break
    obj["stage2_debug_meta"] = legacy_dbg
    obj.pop("fallback_primary_all", None)


def _print_stage3_aggregation_debug(merged: List[Dict[str, Any]]) -> None:
    """term-level aggregation 轻量验收（不改变打分）。"""
    if not STAGE3_AGGREGATION_DEBUG or not merged:
        return
    n = min(int(STAGE3_AGGREGATION_DEBUG_MAX), len(merged))
    print("\n" + "-" * 80)
    print(f"[Stage3 aggregation] (showing {n}/{len(merged)} term-level records)")
    print("-" * 80)
    for obj in merged[:n]:
        print(
            f"  term={obj.get('term')!r} tid={obj.get('tid')} stage3_aggregated={obj.get('stage3_aggregated')!r} "
            f"aggregation_schema_version={obj.get('stage3_aggregation_schema_version')!r} "
            f"records={len(obj.get('records') or [])} anchor_count={obj.get('anchor_count')} "
            f"mainline_anchor_count={obj.get('mainline_anchor_count')} "
            f"side_anchor_count={obj.get('side_anchor_count')} "
            f"expandable_anchor_count={obj.get('expandable_anchor_count')} "
            f"multi_source={obj.get('multi_source')!r} "
            f"local_role={obj.get('local_role')!r} "
            f"best_parent_anchor_final_score={obj.get('best_parent_anchor_final_score')!r} "
            f"best_parent_anchor_step2_rank={obj.get('best_parent_anchor_step2_rank')!r}"
        )
    print("-" * 80 + "\n")


def _stage3_term_bucket_accumulate_row(obj: Dict[str, Any], rec: Dict[str, Any]) -> None:
    """单条 raw 写入 tid 聚合壳（与历史 _merge_stage3_duplicates 内联逻辑一致）。"""
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
    pafs = float(rec.get("best_parent_anchor_final_score") or rec.get("parent_anchor_final_score") or 0.0)
    if pafs > float(obj.get("best_parent_anchor_final_score") or 0.0):
        obj["best_parent_anchor_final_score"] = pafs
    rk_src = rec.get("best_parent_anchor_step2_rank")
    if rk_src is None:
        rk_src = rec.get("parent_anchor_step2_rank")
    try:
        rk_pa = int(rk_src or 0)
    except (TypeError, ValueError):
        rk_pa = 0
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
    # fallback_primary 等 legacy half-decision 已下沉；Stage3 aggregation 不再依赖其顶层镜像。

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


def _build_stage3_aggregated_term_record(tid: Any, grouped_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    单 tid → 一条 term-level aggregated record（正式聚合对象）。
    输入须为已 normalize 的 Stage2 raw 列表（可多条）。
    """
    gr = list(grouped_records)
    if not gr:
        return {}
    first = gr[0]
    obj: Dict[str, Any] = {
        "tid": int(tid),
        "term": first.get("term") or "",
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
    }
    for rec in gr:
        _stage3_term_bucket_accumulate_row(obj, rec)

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
    return obj


def _aggregate_stage3_term_evidence(
    records: List[Dict[str, Any]],
    recall: Any = None,
) -> List[Dict[str, Any]]:
    """
    Stage3 term-level evidence aggregation 唯一正式入口。
    输入：已 normalize 的 raw Stage2 records；输出：每 tid 一条 aggregated term record。
    """
    _ = recall
    bucket: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        tid = rec.get("tid") or rec.get("vid")
        if tid is None:
            continue
        try:
            tid_i = int(tid)
        except (TypeError, ValueError):
            continue
        bucket[tid_i].append(rec)

    merged: List[Dict[str, Any]] = []
    for tid_i in sorted(bucket.keys()):
        merged.append(_build_stage3_aggregated_term_record(tid_i, bucket[tid_i]))
    merged.sort(key=lambda x: float(x.get("best_seed_score") or 0.0), reverse=True)
    _print_stage3_aggregation_debug(merged)
    _print_stage3_duplicate_merge_audit(merged)
    return merged


def _print_stage3_duplicate_merge_audit(merged: List[Dict[str, Any]]) -> None:
    """term-level aggregation 详细审计（多锚）：raw 分布 / 正式聚合字段 / legacy mirror。"""
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
            print(
                "[Stage3 duplicate merge audit] term-level aggregation 详细审计（anchor_count>=2；"
                "非临时 merge 调试）"
            )
            print("-" * 80)
        lines_printed += 1
        recs = obj.get("records") or []
        raw_roles = [((r.get("role_in_anchor") or "")) for r in recs]
        raw_ce = [bool(r.get("can_expand_local")) for r in recs]
        pbs = obj.get("primary_bucket_summary") or {}
        adb = obj.get("anchor_support_summary") or {}
        mdbg = obj.get("merge_debug_summary") or {}
        prov = obj.get("provenance_summary") or {}
        print(f"term={term!r} tid={obj.get('tid')} records={len(recs)}")
        print(f"  parent_anchors={obj.get('parent_anchors')!r}")
        print(f"  role_in_anchor_raw={raw_roles!r}")
        print(f"  can_expand_raw(any local)={raw_ce!r}")
        print(
            f"  local_role_distribution={obj.get('local_role_distribution')!r} "
            f"local_role={obj.get('local_role')!r}"
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
            f"  merged_top_level(unified): can_expand_local={bool(obj.get('can_expand_local'))} "
            f"role_in_anchor={obj.get('role_in_anchor')!r} "
            f"term_role={obj.get('term_role')!r}"
        )
        print(
            f"  aggregated_best_signals: best_seed_score={float(obj.get('best_seed_score') or 0):.4f} "
            f"best_anchor_identity={float(obj.get('best_anchor_identity') or 0):.4f} "
            f"best_jd_align={float(obj.get('best_jd_align') or 0):.4f} "
            f"best_signal_mixed_sources={mdbg.get('best_signal_mixed_sources')!r}"
        )
        print(
            f"  stage3_merge_source={obj.get('stage3_merge_source')!r} "
            f"stage3_aggregation_schema_version={obj.get('stage3_aggregation_schema_version')!r} "
            f"fallback_primary(debug_meta)={bool((obj.get('stage2_debug_meta') or {}).get('fallback_primary'))}"
        )
        if STAGE3_LEGACY_BUCKET_AUDIT:
            raw_pb = [_stage3_rec_bucket_norm(r) for r in recs]
            print("\n  [Stage3 legacy merge compat audit]")
            print(f"    primary_buckets_raw(norm/meta)={raw_pb!r}")
            print(
                f"    bucket_distribution={pbs.get('bucket_distribution')!r} "
                f"derived_primary_bucket={pbs.get('derived_primary_bucket')!r}"
            )
    if lines_printed:
        print("-" * 80 + "\n")


def _merge_stage3_duplicates(raw_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    兼容别名：历史入口名保留。实现已收口至 `_aggregate_stage3_term_evidence`（不重复打印 aggregation audit）。
    """
    return _aggregate_stage3_term_evidence(raw_candidates, None)


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
        # can_expand 语义统一：聚合后也只读 can_expand_local（避免 term-level merged record 再制造 can_expand/can_expand_from_2a 顶层镜像）
        if rec.get("stage3_aggregated"):
            can_expand = bool(rec.get("can_expand_local"))
        else:
            can_expand = bool(rec.get("can_expand_local") or rec.get("can_expand"))
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


def _stage3_compute_basic_risk_flags(term: Dict[str, Any]) -> List[str]:
    """
    Stage3 风险标签（仅用于解释与审计，不参与 guardrail 硬裁决）。

    注意：这一步（Step3）要求 Stage3 guardrail 收缩为「底线闸门 + 软标记层」，
    因此 risk_flags 只做暴露与后续风险分量，不做准入二次裁决。
    """
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
    return risk_flags


def _stage3_should_hard_reject_term(rec: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Stage3 guardrail（hard reject）= **底线守门**，不是第二裁决台。

    输出 `(should_reject, canonical_reason)`，`canonical_reason` 仅两类稳定口径，便于 summary：
    - `missing_required_stage3_fields`：非 dict / 缺 tid / 空 term / 空证据壳，无法形成合法 Stage3 record
    - `weak_evidence_noise`：仅保留 `hierarchy_guard.should_drop_term` 中的 **weak_evidence_noise** 底线项

    细分子类写入 `rec["stage3_guardrail_reject_detail"]`（仅 should_reject=True 时），供审计行展示。

    **不再** hard reject：
    - `outside_subfield_and_no_topic_fit`（改由 soft risk note + 连续分 / risky bucket / 下游处理）
    - `terminal_low_signal_no_recovery`（结构性“不喜欢”改由 risk_penalty / bucket / soft / Stage4 prep）
    """
    rec.pop("stage3_guardrail_reject_detail", None)

    if not isinstance(rec, dict):
        rec["stage3_guardrail_reject_detail"] = "not_a_dict"
        return True, "missing_required_stage3_fields"

    # 1) 缺核心字段优先：避免在非法 shell 上误触发 should_drop_term 的弱证据规则
    if not _stage3_guardrail_tid_ok(rec):
        rec["stage3_guardrail_reject_detail"] = "missing_or_invalid_tid"
        return True, "missing_required_stage3_fields"
    term_str = str(rec.get("term") or "").strip()
    if not term_str:
        rec["stage3_guardrail_reject_detail"] = "empty_term"
        return True, "missing_required_stage3_fields"
    recs = rec.get("records") or []
    evl = rec.get("source_evidence_list") or []
    ev_c = int(rec.get("evidence_count") or 0)
    if len(recs) == 0 and len(evl) == 0 and ev_c <= 0:
        rec["stage3_guardrail_reject_detail"] = "empty_evidence_shell"
        return True, "missing_required_stage3_fields"

    # 2) hierarchy 轻硬过滤：只保留 weak_evidence_noise；离群域宽改走 soft（Step4）
    try:
        drop, drop_reason = should_drop_term(rec)
    except Exception:
        drop, drop_reason = False, ""
    if drop:
        if drop_reason == "outside_subfield_and_no_topic_fit":
            pass  # 非底线 invalidity，不硬拒
        elif drop_reason == "weak_evidence_noise":
            rec["stage3_guardrail_reject_detail"] = "weak_evidence_noise"
            return True, "weak_evidence_noise"
        else:
            rec["stage3_guardrail_reject_detail"] = str(drop_reason or "should_drop_unknown")
            return True, "weak_evidence_noise"

    return False, ""


def _stage3_collect_guardrail_soft_flags(term: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    结构风险软标记 + 说明（不触发 hard_drop）：供 GC / risk_penalty / 审计。
    原 admission 中的 secondary/support 弱证据 veto 已降级为 soft / notes。
    """
    soft: List[str] = []
    notes: List[str] = []
    mdbg = term.get("merge_debug_summary") or {}
    prov = term.get("provenance_summary") or {}

    def _conflict_signal(key: str) -> bool:
        """聚合层顶层冲突位优先（与 merge_debug_summary 同步）；非聚合记录回落 mdbg。"""
        if term.get("stage3_aggregated"):
            v = term.get(key)
            if v is not None:
                return bool(v)
        return bool(mdbg.get(key))

    if _stage3_is_conditioned_only(term):
        soft.append("conditioned_only")
    # 与 should_drop_term 的离群规则对齐，但 Step4 起仅作 risk note（非 hard_drop）
    if not _is_primary_like(term):
        outside_subfield = float(term.get("outside_subfield_mass") or 0.0)
        topic_like = float(term.get("topic_fit") or 0.0)
        if topic_like <= 0.0:
            for k in ("subfield_fit", "field_fit", "domain_fit"):
                if term.get(k) is not None:
                    topic_like = float(term.get(k) or 0.0)
                    break
        if outside_subfield > 0.97 and topic_like < 0.02:
            soft.append("outside_subfield_extreme_low_topic")
    if _conflict_signal("cross_anchor_side_only"):
        soft.append("cross_anchor_side_only")
    # 冲突位来自 aggregation 正式输出；非字面 primary_bucket 桶判定
    if _conflict_signal("primary_bucket_conflict"):
        soft.append("merge_primary_bucket_conflict")
    if _conflict_signal("term_role_conflict"):
        soft.append("merge_term_role_conflict")
    if _conflict_signal("role_in_anchor_conflict"):
        soft.append("merge_role_in_anchor_conflict")
    if _conflict_signal("mainline_side_conflict"):
        soft.append("merge_mainline_side_conflict")
    # 混源不是硬裁决，仅作结构风险暴露
    if _conflict_signal("best_signal_mixed_sources"):
        soft.append("best_signal_mixed_sources")
    if prov.get("single_path_only"):
        soft.append("provenance_single_path_only")
    if prov.get("single_path_multi_anchor_weak_signal"):
        soft.append("single_path_multi_anchor_weak_signal")
    if prov.get("multi_source") is False and int(prov.get("distinct_source_type_count") or 0) <= 1:
        soft.append("provenance_narrow")

    # Stage3 主链不消费 legacy derived_primary_bucket；若 local_role 本身为 risky，则只做软标记（不硬拒）
    lr = (term.get("local_role") or "").strip().lower()
    if lr == "local_risky":
        soft.append("local_role_risky")

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


def _stage3_format_guardrail_soft_flags_for_log(flags: List[str]) -> List[str]:
    """
    Step3（解释口径收口）：soft flags 仍保留原 key（兼容/可检索），但对外展示降级为「risk_note」语义，
    避免在主解释里呈现为“第二套裁决系统”。
    """
    if not flags:
        return []
    rename = {
        # legacy / provisional conflict flags：仅作审计注记
        "merge_primary_bucket_conflict": "note_bucket_conflict",
        "merge_term_role_conflict": "note_term_role_conflict",
        "merge_role_in_anchor_conflict": "note_role_in_anchor_conflict",
        "merge_mainline_side_conflict": "note_mainline_side_conflict",
        # cross-anchor / provenance：解释上归入 coherence/audit
        "cross_anchor_side_only": "note_cross_anchor_side_only",
        "provenance_single_path_only": "note_provenance_single_path_only",
        "provenance_narrow": "note_provenance_narrow",
        # local_role：仅作风险注记（排序主因由连续分承担）
        "local_role_risky": "note_local_role_risky",
        # conditioned_only：仍重要，但作为风险注记展示（主解释用 risk_pen + cross-anchor/coherence）
        "conditioned_only": "note_conditioned_only",
        "outside_subfield_extreme_low_topic": "note_outside_subfield_extreme",
    }
    out: List[str] = []
    for f in flags:
        if not f:
            continue
        out.append(rename.get(f, f"note_{f}"))
    return out


def _print_stage3_guardrail_reject_audit(rows: List[Dict[str, Any]]) -> None:
    if not STAGE3_GUARDRAIL_REJECT_AUDIT or not rows:
        return
    print("\n" + "-" * 80)
    print(
        "[Stage3 guardrail reject] bottom-line only（invalid record / empty shell / weak_evidence_noise）；"
        "not a ranking decision"
    )
    print("-" * 80)
    for r in rows[:40]:
        det = r.get("guardrail_detail")
        det_s = f" detail={det!r}" if det else ""
        print(
            f"  tid={r.get('tid')} term={r.get('term')!r} reason={r.get('guardrail_reason')!r}{det_s} "
            f"flags={r.get('guardrail_flags')!r}"
        )
    if len(rows) > 40:
        print(f"  ... 共 {len(rows)} 条")
    print("-" * 80 + "\n")


def _print_stage3_guardrail_soft_audit(samples: List[Dict[str, Any]]) -> None:
    if not (STAGE3_GUARDRAIL_SOFT_AUDIT and STAGE3_AUDIT_DEBUG) or not samples:
        return
    print("\n" + "-" * 80)
    print("[Stage3 guardrail soft] 已准入但带 risk note（不参与主裁决；主解释看 continuous score + coherence）")
    print("-" * 80)
    for rec in samples[: int(STAGE3_GUARDRAIL_SOFT_AUDIT_MAX)]:
        term = (rec.get("term") or "")[:28]
        prov = rec.get("provenance_summary") or {}
        multi_source = prov.get("multi_source")
        if multi_source is None:
            multi_source = rec.get("multi_source")
        soft_notes = _stage3_format_guardrail_soft_flags_for_log(
            list(rec.get("stage3_guardrail_soft_flags") or [])
        )
        print(
            f"  tid={rec.get('tid')} term={term!r} risk_notes={soft_notes!r} "
            f"anc={int(rec.get('anchor_count') or 0)} ml_hits={int(rec.get('mainline_hits') or 0)} "
            f"can_exp={bool(rec.get('can_expand') or rec.get('can_expand_local'))} multi_source={bool(multi_source)} "
            f"notes_n={len(rec.get('stage3_guardrail_notes') or [])}"
        )
    if len(samples) > int(STAGE3_GUARDRAIL_SOFT_AUDIT_MAX):
        print(f"  ... 省略 {len(samples) - int(STAGE3_GUARDRAIL_SOFT_AUDIT_MAX)} 条（仅展示前 {STAGE3_GUARDRAIL_SOFT_AUDIT_MAX}）")
    print("-" * 80 + "\n")


def _print_stage3_guardrail_summary(
    hard_reject_reasons: List[str],
    soft_flag_lists: List[List[str]],
    survivor_count: int,
) -> None:
    """
    轻量汇总：验证 Stage3 是否从“多拒”变为“少拒多标记”。
    仅用于日志可观测性，不参与打分。
    """
    reasons = Counter([r for r in hard_reject_reasons if r])
    sf = Counter()
    for flags in soft_flag_lists:
        for f in flags or []:
            sf[f] += 1
    soft_flagged = sum(1 for flags in soft_flag_lists if flags)
    clean = max(0, int(survivor_count) - int(soft_flagged))
    print("\n" + "-" * 80)
    print("[Stage3 guardrail summary]")
    print("-" * 80)
    print(f"hard_reject_count={sum(reasons.values())}")
    print(f"soft_noted_survivors={soft_flagged} (risk notes only; ranking 主因见 score/gc breakdown)")
    print(f"clean_survivors={clean} (no soft notes)")
    print(f"hard_reject_reasons={dict(reasons)} (canonical bottom-line classes only)")
    top_soft = dict(sf.most_common(12))
    top_soft_notes = {
        k: v
        for k, v in Counter(
            [x for flags in soft_flag_lists for x in _stage3_format_guardrail_soft_flags_for_log(flags or [])]
        ).most_common(12)
    }
    print(f"top_soft_flags_raw={top_soft}")
    print(f"top_risk_notes={top_soft_notes}")
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

    risk_flags = _stage3_compute_basic_risk_flags(term)
    term["risk_flags"] = list(risk_flags)

    group = term.get("stage3_entry_group") or "support_expansion"
    strength_map = {
        "trusted_primary": "trusted",
        "secondary_primary": "secondary",
        "support_expansion": "support",
    }
    admission_strength = strength_map.get(group, "support")

    # 两段式协议：hard_reject → soft_flags → continue_to_gc
    should_reject, reason = _stage3_should_hard_reject_term(term)
    if should_reject:
        return _pack(True, reason, [reason], risk_flags, [], [], "rejected")

    soft, gr_notes = _stage3_collect_guardrail_soft_flags(term)
    term["stage3_guardrail_soft_flags"] = soft
    term["stage3_guardrail_notes"] = gr_notes
    term["stage3_admission_mode"] = "guardrail_v2"
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

    仅作用于 **GC 主分已定稿之后** 的轻量 tie/visibility 调整；写入 `stage3_score_breakdown.family_role_post_adjust`
    供审计，避免被误读为第二套与 lf/cx/bb/risk 无关的「隐式主分」。
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
            old_p = float(best_primary.get("final_score") or 0.0)
            new_p = old_p * 1.05
            best_primary["final_score"] = new_p
            best_primary["primary_visibility_boost"] = True
            sbp = best_primary.get("stage3_score_breakdown")
            if isinstance(sbp, dict):
                sbp["family_role_post_adjust"] = {
                    "before": old_p,
                    "after": new_p,
                    "mult": new_p / max(1e-9, old_p),
                    "note": "primary_visibility_boost",
                }
        if best_primary and supports:
            p_score = float(best_primary.get("final_score") or 0.0)
            for s in supports:
                s_score = float(s.get("final_score") or 0.0)
                if s_score < p_score * 1.10:
                    old_s = s_score
                    new_s = min(s_score, p_score * 0.98)
                    s["final_score"] = new_s
                    s["support_over_primary_suppressed"] = True
                    sbs = s.get("stage3_score_breakdown")
                    if isinstance(sbs, dict):
                        sbs["family_role_post_adjust"] = {
                            "before": old_s,
                            "after": new_s,
                            "mult": new_s / max(1e-9, old_s),
                            "note": "support_cap_vs_primary",
                        }
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

    # --- Stage3 mainline rule: stable evidence only (no legacy half-decision read ports) ---
    local_role = (rec.get("local_role") or "").strip().lower()
    can_expand_local = bool(
        rec.get("can_expand_local")
        if rec.get("can_expand_local") is not None
        else (rec.get("can_expand") if rec.get("can_expand") is not None else rec.get("can_expand_from_2a"))
    )
    identity_factor = float((rec.get("stage3_explain") or {}).get("identity_factor", 1.0) or 1.0)
    obj = float(rec.get("object_like_risk") or 0.0)
    generic = float(rec.get("generic_risk") or 0.0)
    mainline_candidate = bool(rec.get("mainline_candidate", False))

    is_locked_mainline = bool(mainline_candidate and (not can_expand_local))
    has_mainline_support = (mainline_hits >= 1) or can_expand_local or is_locked_mainline

    if not has_mainline_support and anchor_count > 0:
        flags.append("no_mainline_support")
    if is_locked_mainline:
        flags.append("locked_mainline_no_expand")
    if local_role == "local_risky" and not has_mainline_support:
        flags.append("local_role_risky_no_mainline")
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
    can_expand = bool(
        term.get("can_expand_local")
        if term.get("can_expand_local") is not None
        else (term.get("can_expand") if term.get("can_expand") is not None else term.get("can_expand_from_2a"))
    )
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
    Stage3 分桶（POST-HOC TAG）：在 `final_score` 已定稿并完成排序之后，对 term 打标签：
    core / support / risky。

    禁止：在 Stage3 主链中直接消费 Stage2 legacy/provisional 字段作为裁决输入：
    - primary_bucket / fallback_primary / primary_reason / survive_primary / admission_reason / reject_reason / stage2b_seed_tier

    说明：`primary_bucket_summary` 等 legacy 观测可以继续存在于 debug/compat/Stage4 prep 后置层，
    但 bucket 只应成为对 `final_score` 画像的轻量解释标签，而非第二套裁决系统。
    """
    term.pop("stage3_support_demote_reason", None)
    term.pop("stage3_core_cap_reason", None)
    term.pop("stage3_core_miss_reason", None)
    anchor_count = int(term.get("anchor_count") or 0)
    if anchor_count < 1:
        anchor_count = 1
    mainline_hits = int(term.get("mainline_hits") or 0)
    local_role = (term.get("local_role") or "").strip().lower()
    can_expand_local = bool(
        term.get("can_expand_local")
        if term.get("can_expand_local") is not None
        else (term.get("can_expand") if term.get("can_expand") is not None else term.get("can_expand_from_2a"))
    )
    conditioned_only = _stage3_is_conditioned_only(term)

    final_score = float(term.get("final_score") or 0.0)
    sb = term.get("stage3_score_breakdown") or {}
    risk_score = 0.0
    try:
        risk_score = float(((sb.get("risk_penalty") or {}).get("score")) or 0.0)
    except (TypeError, ValueError):
        risk_score = 0.0

    def _cap_core_if_conditioned_single_anchor(side: str) -> str:
        if side == "core" and conditioned_only and anchor_count <= 1:
            term["stage3_core_cap_reason"] = "conditioned_only_single_anchor"
            return "support"
        return side

    # 1) risky：低分或风险画像明显（保留标签，不是 hard reject）
    if final_score < 0.40:
        return "risky"
    if risk_score >= 0.78:
        return "risky"
    if local_role == "local_risky" and final_score < 0.55:
        return "risky"

    # 2) core：高分 + 风险较低 + 稳定主线证据（仅作为 post-hoc tag）
    if final_score >= 0.70 and risk_score <= 0.55 and (mainline_hits >= 1 or can_expand_local or local_role == "local_core"):
        return _cap_core_if_conditioned_single_anchor("core")

    # 3) support：其余中间带（避免把“只是没被拒掉”的低分项塞进 support）
    return "support"


_STAGE3_ROBOT_MOTION_BACKBONE_TERMS: Tuple[str, ...] = (
    "robot control",
    "motion control",
    "movement control",
)


def _stage3_robot_motion_backbone_term_rank(rec: Dict[str, Any]) -> Optional[int]:
    t = (rec.get("term") or "").strip().lower()
    try:
        return _STAGE3_ROBOT_MOTION_BACKBONE_TERMS.index(t)
    except ValueError:
        return None


def _stage3_robot_motion_rescue_anchor_match(rec: Dict[str, Any]) -> bool:
    chunks: List[str] = []
    pa = rec.get("parent_anchor")
    if pa is not None and str(pa).strip():
        chunks.append(str(pa))
    pp = rec.get("parent_primary")
    if pp is not None and str(pp).strip():
        chunks.append(str(pp))
    panchors = rec.get("parent_anchors")
    if isinstance(panchors, (list, tuple, set)):
        chunks.extend(str(x) for x in panchors if x is not None and str(x).strip())
    elif panchors is not None and str(panchors).strip():
        chunks.append(str(panchors))
    blob = "\n".join(chunks)
    return ("机器人运动控制" in blob) or ("运动控制" in blob)


def _stage3_robot_motion_rescue_excluded_by_risk(rec: Dict[str, Any]) -> bool:
    obj = float(rec.get("object_like_risk") or 0.0)
    if obj >= 0.50:
        return True
    drift = float(rec.get("semantic_drift_risk") or 0.0)
    ex = rec.get("stage3_explain") or {}
    ptc = float(ex.get("path_topic_consistency") or 0.0)
    if drift > 0.75 and ptc < 0.30:
        return True
    return False


def _stage3_is_robot_motion_backbone_rescue_candidate(rec: Dict[str, Any]) -> bool:
    if _stage3_robot_motion_backbone_term_rank(rec) is None:
        return False
    if (rec.get("stage3_bucket") or "").strip().lower() != "support":
        return False
    if not _stage3_robot_motion_rescue_anchor_match(rec):
        return False
    if _stage3_is_conditioned_only(rec):
        return False
    if (rec.get("local_role") or "").strip().lower() == "local_risky":
        return False
    mainline_hits = int(rec.get("mainline_hits") or 0)
    anchor_count = int(rec.get("anchor_count") or 0)
    if not (mainline_hits >= 1 or anchor_count >= 2):
        return False
    can_expand_local = bool(
        rec.get("can_expand_local")
        if rec.get("can_expand_local") is not None
        else (rec.get("can_expand") if rec.get("can_expand") is not None else rec.get("can_expand_from_2a"))
    )
    pb = (rec.get("primary_bucket") or "").strip().lower()
    if not (
        can_expand_local
        or pb == "primary_support_seed"
        or pb == "primary_support_keep"
    ):
        return False
    return True


def _stage3_try_promote_robot_motion_backbone_core(survivors: List[Dict[str, Any]]) -> None:
    """
    极窄 post-hoc：在「机器人运动控制 / 运动控制」父锚语境下，将组内至多 1 条 support 标为 core，
    便于 Stage4 拿到 backbone core。不改变 final_score / 主公式 / 排序。
    """
    for r in survivors:
        if _stage3_robot_motion_backbone_term_rank(r) is None:
            continue
        if (r.get("stage3_bucket") or "").strip().lower() == "core":
            return
    candidates: List[Dict[str, Any]] = []
    for r in survivors:
        if not _stage3_is_robot_motion_backbone_rescue_candidate(r):
            continue
        if _stage3_robot_motion_rescue_excluded_by_risk(r):
            continue
        candidates.append(r)
    if not candidates:
        return

    def _sort_key(rec: Dict[str, Any]) -> Tuple[int, int, int, float]:
        tr = _stage3_robot_motion_backbone_term_rank(rec) or 99
        mh = int(rec.get("mainline_hits") or 0)
        ce = bool(
            rec.get("can_expand_local")
            if rec.get("can_expand_local") is not None
            else (rec.get("can_expand") if rec.get("can_expand") is not None else rec.get("can_expand_from_2a"))
        )
        fs = float(rec.get("final_score") or 0.0)
        return (tr, -mh, -int(ce), -fs)

    chosen = sorted(candidates, key=_sort_key)[0]
    chosen["stage3_bucket"] = "core"
    chosen["stage3_core_rescue"] = True
    chosen["bucket_promote_reason"] = "robot_motion_backbone_rescue"
    flags = list(chosen.get("bucket_reason_flags") or [])
    if "robot_motion_backbone_rescue" not in flags:
        flags.append("robot_motion_backbone_rescue")
    chosen["bucket_reason_flags"] = flags
    ex = dict(chosen.get("stage3_explain") or {})
    ex["stage3_core_rescue_reason"] = "robot_motion_backbone_rescue"
    chosen["stage3_explain"] = ex


def _stage3_is_motion_control_multi_anchor_rescue_candidate(rec: Dict[str, Any]) -> bool:
    if (rec.get("term") or "").strip().lower() != "motion control":
        return False
    if (rec.get("stage3_bucket") or "").strip().lower() != "risky":
        return False
    if int(rec.get("anchor_count") or 0) < 2:
        return False
    if not _stage3_robot_motion_rescue_anchor_match(rec):
        return False
    if _stage3_is_conditioned_only(rec):
        return False
    if (rec.get("local_role") or "").strip().lower() == "local_risky":
        return False
    flags = set(rec.get("bucket_reason_flags") or [])
    if "no_mainline_support" not in flags or "cross_anchor_but_side_only" not in flags:
        return False
    if "object_like" in flags or "generic_like" in flags:
        return False
    if _stage3_robot_motion_rescue_excluded_by_risk(rec):
        return False
    rr = rec.get("risk_reasons") or []
    if isinstance(rr, (list, tuple)) and "high_drift_risk" in rr:
        return False
    return True


def _stage3_try_rescue_motion_control_from_risky(survivors: List[Dict[str, Any]]) -> None:
    """
    极窄 post-hoc：仅「motion control」在机器人运动控制主轴下、双锚 side-only + 无主线命中 被标 risky 时，
    拉回 support；不改 final_score / 排序 / 主链字段，不抬 core（与 robot_motion_backbone_rescue 独立）。
    """
    for rec in survivors:
        if not _stage3_is_motion_control_multi_anchor_rescue_candidate(rec):
            continue
        rec["stage3_bucket"] = "support"
        rec["stage3_support_rescue"] = True
        rec["bucket_promote_reason"] = "motion_control_multi_anchor_rescue"
        flags = list(rec.get("bucket_reason_flags") or [])
        if "motion_control_multi_anchor_rescue" not in flags:
            flags.append("motion_control_multi_anchor_rescue")
        rec["bucket_reason_flags"] = flags
        ex = dict(rec.get("stage3_explain") or {})
        ex["stage3_support_rescue_reason"] = "motion_control_multi_anchor_rescue"
        rec["stage3_explain"] = ex
        return


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
    pre_adj = 第一段 identity×risk×role 链（GC 前参考）；post_adj = GC 四分项 blend 后 final，可再经 family_role 轻调。
    risk_mlt / cross_bn 恒为 1.0（列对齐遗留）；主叙事请看 gc_pre 与 risk_penalty 块（含 backbone→risk 迁移）。
    """
    if not STAGE3_AUDIT_DEBUG or not cands:
        return
    ranked = sorted(cands, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    ranked = _stage3_audit_filter_rows(ranked, top_k)
    if not ranked:
        print("\n" + "-" * 80 + "\n[Stage3 final adjust audit] (no rows after focus filter)\n" + "-" * 80)
        return

    print("\n" + "-" * 80)
    print(
        "[Stage3 final adjust audit] post_adj = GC(lf,cx,bb,risk)×blend + 可选 family_role；"
        "bucket/flags 列仅后置标签"
    )
    print("-" * 80)
    print(
        "term                         | bucket   | pre_adj  | gc_pre   | fam_m | post_adj | "
        "risk_mlt | cross_bn | anchor_count | mainline_hits | can_exp | cond_only | flags"
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
        sb = rec.get("stage3_score_breakdown") or {}
        gc_pre = float(sb.get("gc_pre_blend") or 0.0) if isinstance(sb, dict) else 0.0
        fam_adj = sb.get("family_role_post_adjust") if isinstance(sb, dict) else None
        fam_m = 1.0
        if isinstance(fam_adj, dict) and fam_adj.get("mult") is not None:
            fam_m = float(fam_adj.get("mult") or 1.0)
        anchor_count = int(rec.get("anchor_count") or 0)
        mainline_hits = int(rec.get("mainline_hits") or 0)
        can_expand = bool(
            rec.get("can_expand_local")
            if rec.get("can_expand_local") is not None
            else (rec.get("can_expand") if rec.get("can_expand") is not None else rec.get("can_expand_from_2a"))
        )
        conditioned_only = _stage3_is_conditioned_only(rec)
        flags = list(rec.get("bucket_reason_flags") or [])
        print(
            f"{term:28} | {bucket:8} | {pre_adjust:8.4f} | {gc_pre:8.4f} | {fam_m:5.3f} | {post_adjust:8.4f} | "
            f"{risk_mult:8.4f} | {cross_bonus:8.4f} | {anchor_count:^12d} | {mainline_hits:^13d} | "
            f"{str(can_expand):^7} | {str(conditioned_only):^9} | {flags}"
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
        can_expand = bool(
            rec.get("can_expand_local")
            if rec.get("can_expand_local") is not None
            else (rec.get("can_expand") if rec.get("can_expand") is not None else rec.get("can_expand_from_2a"))
        )
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
    [Stage3 post-hoc bucket audit] support rows
    仅用于查看 post-hoc bucket=Support 的画像（final_score 已定稿后打标签），不作为主排序裁决。
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
        print("\n" + "-" * 80 + "\n[Stage3 post-hoc bucket audit] support rows (no rows / focus empty)\n" + "-" * 80)
        return

    print("\n" + "-" * 80)
    print("[Stage3 post-hoc bucket audit] support rows (tags only; not driving ranking)")
    print("-" * 80)
    print(
        "term                     | stage3_bucket | local_role   | cond_only | anch | ml_hits | can_exp | final"
    )
    for rec in focus:
        term = (rec.get("term") or "")[:24]
        sb = (rec.get("stage3_bucket") or "")[:12]
        lr = (rec.get("local_role") or "").strip().lower()[:10]
        co = _stage3_is_conditioned_only(rec)
        anch = int(rec.get("anchor_count") or 0)
        ml = int(rec.get("mainline_hits") or 0)
        ce = bool(
            rec.get("can_expand_local")
            if rec.get("can_expand_local") is not None
            else (rec.get("can_expand") if rec.get("can_expand") is not None else rec.get("can_expand_from_2a"))
        )
        fs = float(rec.get("final_score") or 0.0)
        print(
            f"{term:24} | {sb:13} | {lr:10} | {str(co):^9} | {anch:^4} | {ml:^7} | {str(ce):^7} | {fs:.4f}"
        )


def _print_stage3_support_risky_concise(cands: List[Dict[str, Any]], top_k: int = 20) -> None:
    """[Stage3 post-hoc bucket concise] 只列 support / risky（final_score 后置标签），快速扫画像。"""
    if not STAGE3_AUDIT_DEBUG or not cands:
        return
    focus = [
        c for c in cands
        if (c.get("stage3_bucket") or "").strip().lower() in ("support", "risky")
    ]
    focus = sorted(focus, key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    focus = _stage3_audit_filter_rows(focus, top_k)
    if not focus:
        print("\n" + "-" * 80 + "\n[Stage3 post-hoc bucket concise] (no rows / focus empty)\n" + "-" * 80)
        return

    print("\n" + "-" * 80)
    print("[Stage3 post-hoc bucket concise] (tags only; not driving ranking)")
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
        can_expand = bool(
            rec.get("can_expand_local")
            if rec.get("can_expand_local") is not None
            else (rec.get("can_expand") if rec.get("can_expand") is not None else rec.get("can_expand_from_2a"))
        )
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



def stage3_build_score_map(survivors: List[Dict[str, Any]], recall: Any = None) -> List[Dict[str, Any]]:
    """
    Stage3 第二段：**continuous-score-first Global Coherence Rerank** — `final_score` 主由
    local_fit / cross_anchor_coherence / backbone_alignment / risk_penalty 经线性式 + 小幅 legacy blend 形成；
    `stage3_bucket` **仅在**本段 `final_score` 定稿后再赋值（后置标签）。

    - **主路径**：`stage3_score_breakdown`；原 backbone 上结构性扣减等价并入 `risk_penalty`（GC 权重下与扣 bb 线性等价）。
    - **兼容**：`stage3_unified_breakdown` 为 legacy 连续分拆解，仅参与 α blend，不作 bucket 乘子链。
    - **家族约束**：GC 后再 `_apply_family_role_constraints`，并写入 `family_role_post_adjust` 供审计。
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
        print("[Stage3 rerank summary] continuous score + global coherence (bucket/flags are post-hoc notes)")
        print("-" * 80)
        print(
            "rank | term | lf | cx | bb | risk | gc_pre | final | graph_x | "
            "bucket(tag) | local_role(tag) | risk_notes"
        )
        for i, rec in enumerate(survivors[:12], start=1):
            term = rec.get("term", "")
            bucket = rec.get("stage3_bucket", "")
            lr_row = (rec.get("local_role") or "")[:10]
            conditioned_only_row = _stage3_is_conditioned_only(rec)
            anchor_count_row = int(rec.get("anchor_count") or 1)
            mainline_hits_row = int(rec.get("mainline_hits") or 0)
            can_expand_row = bool(
                rec.get("can_expand_local")
                if rec.get("can_expand_local") is not None
                else (rec.get("can_expand") if rec.get("can_expand") is not None else rec.get("can_expand_from_2a"))
            )
            final_score = float(rec.get("final_score") or 0.0)
            sb = rec.get("stage3_score_breakdown") or {}
            lf = float((sb.get("local_fit") or {}).get("score") or 0.0)
            cx = float((sb.get("cross_anchor_coherence") or {}).get("score") or 0.0)
            bb = float((sb.get("backbone_alignment") or {}).get("score") or 0.0)
            rk = float((sb.get("risk_penalty") or {}).get("score") or 0.0)
            gcp = float(sb.get("gc_pre_blend") or 0.0)
            graphed = bool(sb.get("used_candidate_graph_cross_edges"))
            risk_notes = _stage3_format_guardrail_soft_flags_for_log(
                list(rec.get("stage3_guardrail_soft_flags") or [])
            )
            print(
                f"{i:>4} | "
                f"{term[:22]:<22} | "
                f"{lf:>4.2f} | {cx:>4.2f} | {bb:>4.2f} | {rk:>4.2f} | {gcp:>5.3f} | {final_score:>6.4f} | {str(graphed):<7} | "
                f"{str(bucket):<10} | {lr_row:<12} | {str(risk_notes)[:60]}"
            )
            # 保留轻量结构列（不作为主解释）
            _ = conditioned_only_row
            _ = anchor_count_row
            _ = mainline_hits_row
            _ = can_expand_row

    return survivors


def _stage3_mainline_phase1_normalize_merge_aggregate(
    recall: Any,
    raw_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Phase 1 (mainline): normalize Stage2 rows → duplicate merge / term-level aggregate →
    global consensus + family context for downstream guardrail + scoring.
    """
    _normalize_stage3_input_records(raw_candidates)
    _print_stage3_input_normalize_debug(raw_candidates, tag="raw_pre_merge")
    _debug_print_stage3_input(raw_candidates)
    merged_candidates = _aggregate_stage3_term_evidence(raw_candidates, recall)
    _normalize_stage3_input_records(merged_candidates)
    _print_stage3_input_normalize_debug(merged_candidates, tag="merged_post_merge")
    for i, rec in enumerate(merged_candidates):
        rec["stage2_rank"] = i
    _compute_stage3_global_consensus(recall, merged_candidates)
    family_buckets = _build_family_buckets(merged_candidates)
    for rec in merged_candidates:
        rec["family_centrality"] = _compute_family_centrality(rec, family_buckets)
    merged_candidates = _classify_stage3_entry_groups(merged_candidates)
    return merged_candidates


def _run_stage3_execute_mainline(
    recall: Any,
    raw_candidates: List[Dict[str, Any]],
    query_vector: Any,
    anchor_vids: Any,
    notes: List[str],
) -> Tuple[
    Dict[str, float],
    Dict[str, str],
    Dict[str, float],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    Phase 1–5 orchestration entry: full dual_gate mainline or legacy `_calculate_final_weights`.
    Returns the same 11-tuple as `_run_stage3_dual_gate` / legacy shim for `run_stage3`.
    """
    use_dual_gate = bool(
        raw_candidates
        and ("term_role" in raw_candidates[0] or "identity_score" in raw_candidates[0])
    )
    if use_dual_gate:
        return _run_stage3_dual_gate(recall, raw_candidates, query_vector, anchor_vids=anchor_vids)
    recall._stage3_last_gc_report = None
    score_map, term_map, idf_map = recall._calculate_final_weights(
        raw_candidates, query_vector, anchor_vids=anchor_vids
    )
    term_role_map: Dict[str, str] = {}
    term_source_map: Dict[str, str] = {}
    parent_anchor_map: Dict[str, str] = {}
    parent_primary_map: Dict[str, str] = {}
    selected_core_terms = _legacy_selected_core_rows_from_maps(score_map, term_map)
    selected_support_terms: List[Dict[str, Any]] = []
    risky_terms: List[Dict[str, Any]] = []
    notes.append("legacy_path:_calculate_final_weights_no_stage3_buckets")
    stage3_result: Dict[str, Any] = {
        "ranked_terms": selected_core_terms,
        "dropped_terms": [],
        "stage3_survivors": len(selected_core_terms),
        "stage3_dropped": 0,
        "boundary_source": "stage3_ranked_terms",
        "guardrail_summary": {
            "hard_reject_total": 0,
            "hard_reject_by_reason": {},
            "soft_flagged_survivors": 0,
        },
    }
    return (
        score_map,
        term_map,
        idf_map,
        term_role_map,
        term_source_map,
        parent_anchor_map,
        parent_primary_map,
        stage3_result,
        selected_core_terms,
        selected_support_terms,
        risky_terms,
    )


def _run_stage3_write_compat_term_maps(
    ranked_terms: List[Dict[str, Any]],
    paper_terms: List[Dict[str, Any]],
    score_map: Dict[str, float],
    term_map: Dict[str, str],
    idf_map: Dict[str, float],
    term_role_map: Dict[str, str],
    term_source_map: Dict[str, str],
    parent_anchor_map: Dict[str, str],
    parent_primary_map: Dict[str, str],
    recall: Any,
    query_vector: Any,
) -> None:
    """Fill flat maps for Stage4/Stage5 compat (unchanged behavior vs inline loops in `run_stage3`)."""
    top_survivors = ranked_terms[:STAGE3_TOP_K]
    for rec in top_survivors:
        _write_term_maps(
            score_map,
            term_map,
            idf_map,
            term_role_map,
            term_source_map,
            parent_anchor_map,
            parent_primary_map,
            recall,
            rec,
            query_vector,
        )
    for rec in paper_terms:
        _write_term_maps_if_missing(
            score_map,
            term_map,
            idf_map,
            term_role_map,
            term_source_map,
            parent_anchor_map,
            parent_primary_map,
            recall,
            rec,
            query_vector,
        )


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
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    Stage3 主流程（MAINLINE）：normalize → term-level evidence aggregate → 全局共识 → light guardrail
    → score_term_record + GC/unified rerank → family 角色约束 → risky/bucket（仅标签）→ 输出 ranked term records。

    注意：本函数只负责 Stage3 主链闭环并产出 `stage3_result.ranked_terms`。
    Stage4 prep（paper gate / coverage / lane fill 等）必须由 `run_stage3(...)` 在主链结束后显式调用后置入口完成。
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

    # --- Phase 1: normalize + duplicate merge / term-level aggregate + consensus + family context ---
    merged_candidates = _stage3_mainline_phase1_normalize_merge_aggregate(recall, raw_candidates)

    topic_gate_bypass_primary_like_count = 0
    guardrail_reject_audit_rows: List[Dict[str, Any]] = []
    guardrail_soft_samples: List[Dict[str, Any]] = []
    guardrail_hard_reject_reasons: List[str] = []

    # --- Phase 2–3: per-record bottom-line guardrail + topic gate + continuous evidence (same loop; order preserved) ---
    for rec in merged_candidates:
        rec["family_key"] = rec.get("family_key") or build_family_key(rec)
        if not rec.get("term_role") and rec.get("term_roles"):
            roles = list(rec["term_roles"]) if isinstance(rec["term_roles"], (list, set)) else [rec["term_roles"]]
            rec["term_role"] = "primary" if "primary" in roles else (roles[0] if roles else "")
        rec["term_role"] = rec.get("term_role") or ""
        rec["retrieval_role"] = rec.get("retrieval_role") or get_retrieval_role_from_term_role(rec.get("term_role"))

        # Phase 2 — bottom-line hard/soft guardrail (no rerank verdict here beyond drop/continue)
        rec["risk_flags"] = _stage3_compute_basic_risk_flags(rec)
        hard_reject, gr_reason = _stage3_should_hard_reject_term(rec)
        if hard_reject:
            rec["reject_reason"] = gr_reason
            rec["guardrail_flags"] = [gr_reason]
            dropped_with_reason.append(rec)
            guardrail_hard_reject_reasons.append(gr_reason)
            guardrail_reject_audit_rows.append(
                {
                    "tid": rec.get("tid"),
                    "term": rec.get("term"),
                    "guardrail_reason": gr_reason,
                    "guardrail_flags": [gr_reason],
                    "guardrail_detail": rec.get("stage3_guardrail_reject_detail"),
                }
            )
            if STAGE3_DETAIL_DEBUG or stage3_debug:
                print(f"[Stage3 硬过滤] drop tid={rec.get('tid')} term={rec.get('term')!r} reason={gr_reason}")
            continue

        soft, gr_notes = _stage3_collect_guardrail_soft_flags(rec)
        rec["stage3_guardrail_soft_flags"] = soft
        rec["stage3_guardrail_notes"] = gr_notes
        rec["stage3_admission_mode"] = "guardrail_v2"
        if soft:
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

        # Phase 3 — continuous local / cross-anchor / backbone inputs via existing scorers (formula unchanged)
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
    _print_stage3_guardrail_summary(
        hard_reject_reasons=guardrail_hard_reject_reasons,
        soft_flag_lists=[r.get("stage3_guardrail_soft_flags") or [] for r in survivors],
        survivor_count=len(survivors),
    )
    # (guardrail audit prints complete Phase 2 observability)

    if label_trace and topic_gate_bypass_primary_like_count and not STAGE3_DEBUG_FOCUS_TERMS:
        print(
            f"[Stage3 topic_gate] bypass primary-like count={topic_gate_bypass_primary_like_count} "
            f"(明细仅 STAGE3_DEBUG_FOCUS_TERMS 非空时逐条打印)"
        )

    # --- Phase 4: unified rerank (pre-GC ordering + global coherence blend in stage3_build_score_map) ---
    survivors.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    for i, rec in enumerate(survivors):
        rec["stage3_rank"] = i

    # 批注：家族 primary/support 微调挪入 stage3_build_score_map；stage3_bucket 仅在 GC 主分定稿后写入。

    survivors = stage3_build_score_map(survivors, recall)
    for rec in survivors:
        rec["risk_reasons"] = _collect_risky_reasons(rec)

    _stage3_try_promote_robot_motion_backbone_core(survivors)
    _stage3_try_rescue_motion_control_from_risky(survivors)

    # 窄表审计：final adjust → cross-anchor → support/risky（与 STAGE3_DEBUG_FOCUS_TERMS 配合可只看定点词）
    if STAGE3_AUDIT_DEBUG:
        _print_stage3_final_adjust_audit(survivors, top_k=12)
        _print_stage3_cross_anchor_audit(survivors, top_k=40)
        _print_stage3_support_subtype_audit(survivors, top_k=24)
        _print_stage3_support_risky_concise(survivors, top_k=20)
        _print_stage3_core_miss_audit(survivors, limit=24)

    # --- Phase 5: post-hoc bucket split (tags only; after final_score / GC) ---
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
    # --- Phase 6: finalize stage3_result (mainline end = ranked_terms) + Stage3 summary logs ---
    stage3_result: Dict[str, Any] = {
        "ranked_terms": survivors,
        "stage3_survivors": len(survivors),
        "dropped_terms": dropped_with_reason,
        "stage3_dropped": len(dropped_with_reason),
        "boundary_source": "stage3_ranked_terms",
        "guardrail_summary": {
            "hard_reject_total": len(guardrail_hard_reject_reasons),
            "hard_reject_by_reason": dict(Counter(guardrail_hard_reject_reasons)),
            "soft_flagged_survivors": len(guardrail_soft_samples),
        },
    }

    rerank_delta_rows = []
    for rec in survivors[:25]:
        rerank_delta_rows.append({
            "term": rec.get("term") or "",
            "stage2_rank": rec.get("stage2_rank", 0),
            "stage3_rank": rec.get("stage3_rank", 0),
            "delta": (rec.get("stage2_rank") or 0) - (rec.get("stage3_rank") or 0),
        })
    # Stage3 主链日志：只打印 Stage3 自身输出；Stage4 prep 相关输出必须在后置层打印。
    top_survivors = survivors[:STAGE3_TOP_K]
    _debug_print_stage3_output(
        raw_candidates=merged_candidates,
        survivors=survivors,
        core_terms=core_terms_list,
        support_terms=support_terms_list,
        risky_terms=risky_terms_list,
        paper_terms=[],
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
        stage3_result,
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
        print(f"[Stage3] 去重后={len(raw_candidates)} 幸存={len(survivors)} top_k={len(top_survivors)}")
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
    # Stage4 prep / paper bridge 输出必须后置（仅当 paper_terms 非空时打印）
    if STAGE3_OUTPUT_BREAKDOWN_DEBUG and (STAGE3_DETAIL_DEBUG or stage3_debug) and paper_terms:
        print(
            f"[Stage4 prep paper_term_selection] selected={len(paper_terms)} "
            "detail: family_key | term | term_role | retrieval_role | final_score"
        )
        for r in paper_terms[:30]:
            print(
                f"  {r.get('family_key','')} | {r.get('term','')!r} | {r.get('term_role','')} | "
                f"{r.get('retrieval_role','')} | {r.get('final_score',0):.4f}"
            )
        if len(paper_terms) > 30:
            print(f"  ... 共 {len(paper_terms)} 条")
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
    """Legacy path: no Stage3 buckets; approximate core rows as top survivors by score_map order."""
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
    Stage3 — Global Coherence Rerank（职责冻结版）。

    正式身份：Stage3 = **Global Coherence Rerank**。正式职责为：
    - term-level aggregation
    - cross-anchor coherence
    - global rerank
    - light guardrail

    消费对象：Stage2 已产出的 **local evidence candidates**（`stage2_output` 四键契约）。
    Stage2 完成 local landing 与 local evidence construction；Stage3 **不承担**大规模补做
    local candidate generation。

    主链（仅此六步；概念顺序，与具体函数名不必一一对应）：
    1. duplicate merge
    2. local fit estimation
    3. cross-anchor coherence estimation
    4. backbone alignment / risk penalty estimation
    5. unified rerank
    6. post-hoc bucket split（core / support / risky）

    主链自然终点：`stage3_result["ranked_terms"]`。

    Stage4 prep：`prepare_stage4_terms_from_stage3(...)`（`stage4_prep_bridge`）仅为 **POST-LAYER**，须在主链结束之后
    调用，且 **只允许** 消费 `ranked_terms`；不得将 coverage / quota / contamination 等 prep
    决策混回 Stage3 主链的 merge、排序或主链 gate。

    下列事项 **不属于** Stage3 **主路径**（可与同模块其它实现共存，边界上视为后置桥接 /
    Stage4 prep / 论文侧策略，而非主链编排）：
    - paper lane orchestration
    - dynamic floor
    - tail expand
    - swap
    - quota / budget
    - support contamination admit
    - risky compete
    - term→paper selection policy

    入参 `stage2_output` 须含：all_candidates, anchor_to_candidates, candidate_graph, stage2_report。

    返回：除 `stage3_result`、`stage4_prep_result` 外，仍保留 selected_*、paper_terms、score_map
    等兼容字段供调用方与回归使用。
    """
    # === Phase 0: Stage2 contract validate + attach context for Stage3 mainline ===
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

    # Phase 0 — contract-size log (before mainline normalize/merge)
    # --- Stage3 visible entry log (verification mouth) ---
    # Keep this lightweight and contract-only: no scoring logic, no structural mutation.
    try:
        edge_counts = {k: len(v or []) for k, v in (candidate_graph or {}).items()}
    except Exception:
        edge_counts = {"_error": "edge_counts_failed"}
    print(
        "\n[stage3_input]",
        {
            "stage2_output_keys": sorted(list(stage2_output.keys())),
            "all_candidates": len(all_candidates),
            "anchors": len(anchor_to_candidates),
            "candidate_graph_edge_counts": edge_counts,
            "stage2_report_summary": {
                "anchor_count": stage2_report.get("anchor_count"),
                "candidate_count": stage2_report.get("candidate_count"),
                "graph_edge_counts": stage2_report.get("graph_edge_counts"),
            },
            "graph_non_empty": graph_non_empty,
            "anchor_map_non_empty": anchor_map_non_empty,
        },
    )

    # Phase 0 — recall-side Stage2 pass-through (unchanged attachment point)
    # 向下透传（供后续 `_aggregate_stage3_term_evidence` / global rerank 使用；本版不在此消费图结构）
    _pass_through = {
        "anchor_to_candidates": anchor_to_candidates,
        "candidate_graph": candidate_graph,
        "stage2_report": stage2_report,
    }
    setattr(recall, "_stage3_stage2_context", _pass_through)
    if getattr(recall, "debug_info", None) is not None:
        recall.debug_info.stage2_contract_pass_through = _pass_through  # type: ignore[attr-defined]

    # 观测用 notes（非 docstring）：主链仍走双闸门或 legacy 分支，与 Step0 职责冻结不冲突。
    notes: List[str] = [
        "stage3_mainline: dual_gate when term_role/identity_score present else legacy final_weights",
    ]
    candidate_count_in = len(all_candidates)

    if not all_candidates:
        # Phase 0 — empty pool: skip Phases 1–5, still run POST-LAYER prep on empty ranked_terms
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
            "stage3_result": {
                "ranked_terms": [],
                "dropped_terms": [],
                "stage3_survivors": 0,
                "stage3_dropped": 0,
                "boundary_source": "stage3_ranked_terms",
                "guardrail_summary": {
                    "hard_reject_total": 0,
                    "hard_reject_by_reason": {},
                    "soft_flagged_survivors": 0,
                },
            },
            "stage4_prep_result": {
                "paper_terms": [],
                "prep_debug": None,
                "stage4_prep_selected": 0,
                "stage4_prep_rejected_pre_coverage": 0,
                "boundary_source": "stage3_ranked_terms",
            },
            "global_coherence_report": empty_report,
            "score_map": {},
            "term_map": {},
            "idf_map": {},
            "term_role_map": {},
            "term_source_map": {},
            "parent_anchor_map": {},
            "parent_primary_map": {},
        }

    # === Phases 1–5: Stage3 rerank mainline (orchestrated in `_run_stage3_execute_mainline`) ===
    raw_candidates = all_candidates
    selected_core_terms: List[Dict[str, Any]] = []
    selected_support_terms: List[Dict[str, Any]] = []
    risky_terms: List[Dict[str, Any]] = []
    paper_terms: List[Dict[str, Any]] = []
    stage3_result: Dict[str, Any] = {}
    stage4_prep_result: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Stage3 主链边界（冻结口径，与 Step0 文档一致）
    #
    # - 主链自然终点：`stage3_result["ranked_terms"]`（之前的 merge / fit / coherence /
    #   backbone|risk / unified rerank / bucket 均服务于该产物）。
    # - `prepare_stage4_terms_from_stage3(...)` 仅为 POST-LAYER：输入应是 ranked_terms，
    #   输出 paper_terms / prep_debug；不得将 Stage4 prep 的 coverage / quota / contamination
    #   等决策嵌回主链排序或主链 gate。
    # - paper lane / dynamic floor / tail expand / swap / quota、budget、support contamination admit、
    #   risky compete、term→paper selection policy 等同理，非主路径。
    # -------------------------------------------------------------------------
    (
        score_map,
        term_map,
        idf_map,
        term_role_map,
        term_source_map,
        parent_anchor_map,
        parent_primary_map,
        stage3_result,
        selected_core_terms,
        selected_support_terms,
        risky_terms,
    ) = _run_stage3_execute_mainline(
        recall, raw_candidates, query_vector, anchor_vids, notes
    )

    # --- Mainline closure: ranked_terms must exist before any Stage4 prep ---
    ranked_terms = list(stage3_result.get("ranked_terms") or [])
    dropped_terms = list(stage3_result.get("dropped_terms") or [])

    if STAGE3_AUDIT_DEBUG or LABEL_PATH_TRACE or getattr(term_scoring, "STAGE3_DEBUG", False):
        print(
            f"\n[Stage3 mainline closure] ranked_terms={len(ranked_terms)} "
            f"dropped={len(dropped_terms)} survivors_reported={stage3_result.get('stage3_survivors')} "
            f"-> entering Stage4 prep POST-LAYER\n"
        )

    # === Phase 6: Stage4 prep POST-LAYER (consume ranked_terms only) ===
    paper_terms, prep_dbg = prepare_stage4_terms_from_stage3(
        ranked_terms,
        recall=recall,
        stage3_dropped_terms=dropped_terms,
    )
    stage4_prep_result = {
        "paper_terms": paper_terms,
        "prep_debug": prep_dbg,
        "stage4_prep_selected": int((prep_dbg or {}).get("stage4_prep_selected") or 0),
        "stage4_prep_rejected_pre_coverage": int((prep_dbg or {}).get("stage4_prep_rejected_pre_coverage") or 0),
        "boundary_source": str((prep_dbg or {}).get("boundary_source") or "stage3_ranked_terms"),
    }

    # Phase 6 — handoff summary (includes Stage4 prep counts; after POST-LAYER)
    if STAGE3_AUDIT_DEBUG or LABEL_PATH_TRACE or getattr(term_scoring, "STAGE3_DEBUG", False):
        print("\n" + "-" * 80)
        print("[Stage3/Stage4 boundary summary]")
        print("-" * 80)
        print(f"stage3_ranked_terms={len(ranked_terms)}")
        print(f"stage3_survivors={int(stage3_result.get('stage3_survivors') or 0)}")
        print(f"stage4_prep_selected={stage4_prep_result.get('stage4_prep_selected')}")
        print(f"stage4_prep_rejected_pre_coverage={stage4_prep_result.get('stage4_prep_rejected_pre_coverage')}")
        print(f"boundary_source={stage4_prep_result.get('boundary_source')!r}")
        _pd = stage4_prep_result.get("prep_debug") or {}
        if isinstance(_pd, dict):
            print(f"stage4_prep_lane_counter={_pd.get('stage4_prep_lane_counter')!r}")
            print(f"stage4_prep_reason_counter={_pd.get('stage4_prep_reason_counter')!r}")
        print(
            "note: Stage4 prep uses continuous bridge (final/gc_pre/lf/bb/risk) + paper_ready; "
            "stage3_bucket is weak prior (readiness mult / risky quota), not a duplicate hard sort."
        )
        print("-" * 80 + "\n")

    # Phase 6 — Stage4→Stage5 flat map fill (compat; unchanged ordering)
    # Downstream still relies on these flat maps to build final_term_ids_for_paper and author aggregation payloads.
    _run_stage3_write_compat_term_maps(
        ranked_terms,
        paper_terms,
        score_map,
        term_map,
        idf_map,
        term_role_map,
        term_source_map,
        parent_anchor_map,
        parent_primary_map,
        recall,
        query_vector,
    )

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
        "stage3_result": stage3_result,
        "stage4_prep_result": stage4_prep_result,
        "global_coherence_report": global_coherence_report,
        "score_map": score_map,
        "term_map": term_map,
        "idf_map": idf_map,
        "term_role_map": term_role_map,
        "term_source_map": term_source_map,
        "parent_anchor_map": parent_anchor_map,
        "parent_primary_map": parent_primary_map,
    }

