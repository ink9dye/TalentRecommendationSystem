from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

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
# paper_readiness · 锚点先验（Stage3）：以 Step2 锚点分为「主证据」，名次只做轻平滑（避免 rank 主导把 robot control 等主线词打出 dynamic_floor）
PAPER_READINESS_ANCHOR_SCORE_NORM = 1.20  # anchor_score_norm = clip(score / 此值, 0, 1)
PAPER_READINESS_ANCHOR_PRIOR_W_SCORE = 0.88  # 先验里分值权重（主）
PAPER_READINESS_ANCHOR_PRIOR_W_RANK = 0.12  # 先验里名次平滑权重（辅）
PAPER_READINESS_ANCHOR_RANK_SMOOTH_LAMBDA = 0.04  # rank_smooth = 1/(1+λ·max(rank-1,0))
PAPER_READINESS_ANCHOR_READINESS_BLEND_LO = 0.82  # readiness *= LO + HI * anchor_prior
PAPER_READINESS_ANCHOR_READINESS_BLEND_HI = 0.18
# core + 已命中主线 + 可扩：paper_readiness 保底（结构信号，非词表）；防「主线可扩却被 floor 误杀」
PAPER_READINESS_CORE_MAINLINE_EXPAND_MIN = 0.78
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
# Paper：risky + 无主线命中 + 不可扩 + side/expansion 角色 — 仅靠 unified 分过线仍会混入（如 motion controller）
STAGE3_PAPER_RISKY_SIDE_BLOCK_ENABLED = True
STAGE3_PAPER_RISKY_SIDE_TERM_ROLES = frozenset(
    {"primary_side", "dense_expansion", "cluster_expansion", "cooc_expansion"}
)
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
# 与 select_terms_for_paper_recall 对齐的入稿闸门说明（逐项 term）
STAGE3_PAPER_GATE_DEBUG = False  # 默认降噪；需要逐项 gate 时再开
STAGE3_PAPER_CUTOFF_AUDIT = True  # [Stage3 paper cutoff] 排名与截断原因
STAGE3_SUPPORT_CONTAMINATION_AUDIT = True  # Stage3→4 之间：可疑 support 摘要
STAGE3_UNIFIED_SCORE_DEBUG = True  # 连续分特征拆解（替代原「规则乘子」视角）
# Stage3 窄表审计：final adjust / cross-anchor / support·risky / paper bridge（与 bucket reason 互补）
STAGE3_AUDIT_DEBUG = True
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


def _paper_recall_dynamic_floor(
    candidates: List[Dict[str, Any]],
    score_key: str = "final_score",
) -> float:
    """Paper 动态下限：Top1(score_key)×REL 与 ABS 取 max。paper 选词可与 final_score 同尺度或改用 paper_select_score。"""
    if not candidates:
        return PAPER_RECALL_DYNAMIC_FLOOR_ABS
    top = max(float(r.get(score_key) or 0.0) for r in candidates)
    return max(PAPER_RECALL_DYNAMIC_FLOOR_ABS, top * PAPER_RECALL_DYNAMIC_FLOOR_REL)


def _print_stage3_unified_breakdown(survivors: List[Dict[str, Any]], top_k: int = 16) -> None:
    """打印连续分特征拆解，便于回答「谁被哪一项抬高/压低」。"""
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
    仅 eligible 内、顺序与真实选词一致（paper_select_score 降序）。
    """
    if not STAGE3_PAPER_CUTOFF_AUDIT or not ordered:
        return
    print("\n" + "-" * 80)
    print("[Stage3 paper selection audit] eligible · structural ranking (no blocklist)")
    print("-" * 80)
    hdr = "term · fs · ready · p_sel · bucket · role · ml · anch · … · panch · a_sc · rk · outcome"
    print(hdr)
    print("-" * 90)
    for rec in ordered:
        term = (rec.get("term") or "")[:20]
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
        panch = (rec.get("parent_anchor") or "")[:12]
        if not panch and rec.get("parent_anchors"):
            pa_list = rec.get("parent_anchors") or []
            panch = str(pa_list[0])[:12] if pa_list else ""
        print(
            f"{term:20} | {fs:4.2f} | {ready:5.2f} | {pss:5.2f} | {b:6} | {role:10} | {ml:^2} | {anch:^4} | "
            f"{str(exp)[:1]:^3} | {str(co)[:1]:^5} | {pg:4.2f} | {str(sel)[:1]:^3} | "
            f"panch={panch:12} a_sc={pas:4.2f} r={par_i:^3} | {outcome[:20]}"
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
    print("term                 | final | ready | p_sel  | Δ末席  | a_sc | rk | reject/cutoff")
    print("-" * 80)
    for pss, rec, reason, delta in rows[:24]:
        t = (rec.get("term") or "")[:20]
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
        print(
            f"{t:20} | {fs:5.3f} | {rd:5.3f} | {pss:6.3f} | {delta:+6.3f} | {pas:4.2f} | {pr_i:2d} | {reason[:36]}"
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


def _print_stage3_duplicate_merge_audit(merged: List[Dict[str, Any]]) -> None:
    """多锚 tid：逐条列出各来源 raw 字段与合并后口径，定位 merge 是否盖掉强主线证据。"""
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
            print("[Stage3 duplicate merge audit] anchor_count>=2：raw 来源 vs 合并后 term 级语义")
            print("-" * 80)
        lines_printed += 1
        recs = obj.get("records") or []
        raw_pb = [((r.get("primary_bucket") or "")) for r in recs]
        raw_roles = [((r.get("role_in_anchor") or "")) for r in recs]
        raw_ce = [bool(r.get("can_expand") or r.get("can_expand_from_2a")) for r in recs]
        print(f"term={term!r} tid={obj.get('tid')} records={len(recs)}")
        print(f"  parent_anchors={obj.get('parent_anchors')!r}")
        print(f"  primary_buckets_raw={raw_pb!r}")
        print(f"  role_in_anchor_raw={raw_roles!r}")
        print(f"  can_expand_raw={raw_ce!r}")
        print(
            f"  merged_primary_bucket={obj.get('primary_bucket')!r} "
            f"merged_can_expand={bool(obj.get('can_expand'))} "
            f"merged_role_in_anchor={obj.get('role_in_anchor')!r} "
            f"merged_term_role={obj.get('term_role')!r}"
        )
        print(
            f"  merge_source={obj.get('stage3_merge_source')!r} "
            f"fallback_primary={bool(obj.get('fallback_primary'))}"
        )
    if lines_printed:
        print("-" * 80 + "\n")


def _merge_stage3_duplicates(raw_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按 tid 聚合同一词的多个候选，保留多来源结构信息，供 Stage3 分层与准入使用。
    合并后保留：anchor_count, evidence_count, family_keys, source_types, parent_anchors, parent_primaries,
    mainline_hits, side_hits、best_* 标量等。

    **语义字段（term 级）**：`primary_bucket` / `primary_reason` 按 **STAGE3_PRIMARY_BUCKET_MERGE_PRIORITY** 取最强一条；
    `mainline_candidate`、`can_expand_local`、`can_expand`、`can_expand_from_2a` 为来源间 **OR**；
    `fallback_primary` 仅当 **每条来源** 均为 fallback 时为 True；
    `role_in_anchor` **任一为 mainline 则 mainline**；`term_role` 按 **STAGE3_TERM_ROLE_MERGE_PRIORITY** 取最强。
    其余字段仍来自 **seed 分最高** 的记录（同分则 **primary_bucket 优先级更高** 者优先），避免再出现
    「anchor_count=2 但 primary_bucket 只剩 risky_keep」的继承撕裂。
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
                "anchor_count": 0,
                "evidence_count": 0,
                "mainline_hits": 0,
                "side_hits": 0,
                "has_primary_role": False,
                "has_support_role": False,
                "role_in_anchor": "",
                "can_expand": False,
                "can_expand_from_2a": False,
                "primary_bucket": "",
                "primary_reason": "",
                "mainline_candidate": False,
                "can_expand_local": False,
                "fallback_primary_all": True,
                "term_role": "",
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
        if role_anchor == "mainline":
            obj["mainline_hits"] = obj.get("mainline_hits", 0) + 1
        elif role_anchor:
            obj["side_hits"] = obj.get("side_hits", 0) + 1
        if tr == "primary" or st == "similar_to":
            obj["has_primary_role"] = True
        else:
            obj["has_support_role"] = True
        if rec.get("can_expand"):
            obj["can_expand"] = True
        if rec.get("can_expand_from_2a"):
            obj["can_expand_from_2a"] = True
            obj["can_expand"] = True
        rm = (rec.get("retain_mode") or "normal").strip().lower()
        if rm == "normal":
            obj["retain_mode"] = "normal"

        # ---- 结构语义：多源聚合，禁止被「单条 best seed」覆盖冲淡 ----
        cur_pb = (rec.get("primary_bucket") or "").strip().lower()
        best_pb = (obj.get("primary_bucket") or "").strip().lower()
        if STAGE3_PRIMARY_BUCKET_MERGE_PRIORITY.get(cur_pb, 0) > STAGE3_PRIMARY_BUCKET_MERGE_PRIORITY.get(
            best_pb, 0
        ):
            obj["primary_bucket"] = rec.get("primary_bucket") or ""
            obj["primary_reason"] = rec.get("primary_reason") or obj.get("primary_reason") or ""

        obj["mainline_candidate"] = bool(
            obj.get("mainline_candidate", False) or rec.get("mainline_candidate", False)
        )
        obj["can_expand_local"] = bool(
            obj.get("can_expand_local", False) or rec.get("can_expand_local", False)
        )
        obj["fallback_primary_all"] = bool(obj.get("fallback_primary_all", True)) and bool(
            rec.get("fallback_primary", False)
        )

        cur_tr = (rec.get("term_role") or "").strip().lower()
        prev_tr = (obj.get("term_role") or "").strip().lower()
        if STAGE3_TERM_ROLE_MERGE_PRIORITY.get(cur_tr, 0) > STAGE3_TERM_ROLE_MERGE_PRIORITY.get(
            prev_tr, 0
        ):
            obj["term_role"] = rec.get("term_role") or ""

        if role_anchor == "mainline":
            obj["role_in_anchor"] = "mainline"
        elif role_anchor and (obj.get("role_in_anchor") or "").strip().lower() != "mainline":
            obj["role_in_anchor"] = rec.get("role_in_anchor") or role_anchor

        prev_sc = float(obj.get("_merge_best_seed_sc", -1.0))
        if seed_sc > prev_sc:
            obj["_merge_best_seed_sc"] = seed_sc
            obj["_merge_best_seed_rec"] = rec
        elif seed_sc == prev_sc and seed_sc >= 0.0:
            old = obj.get("_merge_best_seed_rec") or {}
            old_pb = (old.get("primary_bucket") or "").strip().lower()
            if STAGE3_PRIMARY_BUCKET_MERGE_PRIORITY.get(cur_pb, 0) > STAGE3_PRIMARY_BUCKET_MERGE_PRIORITY.get(
                old_pb, 0
            ):
                obj["_merge_best_seed_rec"] = rec

    merged: List[Dict[str, Any]] = []
    for tid, obj in bucket.items():
        best_rec = obj.pop("_merge_best_seed_rec", None)
        obj.pop("_merge_best_seed_sc", None)
        if best_rec:
            for k, v in best_rec.items():
                if k in _STAGE3_MERGE_SKIP_KEYS or k in _STAGE3_MERGE_STRUCTURE_FIELDS:
                    continue
                obj[k] = v

        obj["fallback_primary"] = bool(obj.get("fallback_primary_all", False))
        obj.pop("fallback_primary_all", None)
        obj["stage3_merge_source"] = "strongest_bucket_aggregate"

        obj["parent_anchors"] = sorted(obj["parent_anchors"])
        obj["parent_primaries"] = sorted(obj["parent_primaries"])
        obj["source_types"] = sorted(obj["source_types"])
        obj["term_roles"] = sorted(obj["term_roles"])
        obj["family_keys"] = sorted(obj.get("family_keys") or [])
        obj["anchor_count"] = len(obj["parent_anchors"])
        obj["evidence_count"] = len(obj["records"])
        obj["best_stage2_score"] = obj.get("best_stage2_score") or obj.get("best_seed_score") or 0.0
        if not obj.get("role_in_anchor") and obj.get("mainline_hits", 0) > 0:
            obj["role_in_anchor"] = "mainline"
        if not obj.get("role_in_anchor"):
            obj["role_in_anchor"] = "side" if obj.get("side_hits", 0) > 0 else ""
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


def _check_stage3_admission(
    term: Dict[str, Any],
    jd_profile: Any,
    active_domains: Any,
) -> Dict[str, Any]:
    """
    分层审查：trusted_primary 默认不 hard_drop；secondary 可降分不轻易杀；support_expansion 才严格。
    返回 hard_drop, reason, risk_flags, admission_strength。
    """
    group = term.get("stage3_entry_group") or "support_expansion"
    risk_flags: List[str] = []
    poly = float(term.get("polysemy_risk") or 0.0)
    obj = float(term.get("object_like_risk") or 0.0)
    generic = float(term.get("generic_risk") or 0.0)
    anchor_id = float(term.get("best_anchor_identity") or term.get("best_identity_score") or 0.0)
    jd_align = float(term.get("best_jd_align") or 0.0)
    ctx = float(term.get("best_context_continuity") or 0.0)
    if poly > 0.75:
        risk_flags.append("high_polysemy")
    if obj > 0.55:
        risk_flags.append("object_like")
    if generic > 0.55:
        risk_flags.append("too_generic")
    term["risk_flags"] = risk_flags

    if group == "trusted_primary":
        return {
            "hard_drop": False,
            "reason": "",
            "risk_flags": risk_flags,
            "admission_strength": "trusted",
        }
    if group == "secondary_primary":
        if anchor_id < 0.16 and jd_align < 0.72 and ctx < 0.48:
            return {
                "hard_drop": True,
                "reason": "secondary_too_weak",
                "risk_flags": risk_flags,
                "admission_strength": "weak",
            }
        return {
            "hard_drop": False,
            "reason": "",
            "risk_flags": risk_flags,
            "admission_strength": "secondary",
        }
    if group == "support_expansion":
        if anchor_id < 0.14 and ctx < 0.50:
            return {
                "hard_drop": True,
                "reason": "weak_evidence_noise",
                "risk_flags": risk_flags,
                "admission_strength": "support",
            }
        if obj > 0.70 and generic > 0.60:
            return {
                "hard_drop": True,
                "reason": "object_generic_noise",
                "risk_flags": risk_flags,
                "admission_strength": "support",
            }
        return {
            "hard_drop": False,
            "reason": "",
            "risk_flags": risk_flags,
            "admission_strength": "support",
        }
    return {
        "hard_drop": False,
        "reason": "",
        "risk_flags": risk_flags,
        "admission_strength": "support",
    }


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


def _apply_paper_readiness_for_recall(rec: Dict[str, Any]) -> None:
    """
    Paper 选词序专用乘子（不改 final_score）：stage3 结构字段 + **Step2 锚点主线强度**（无词表）。
    - `best_parent_anchor_final_score`：tid 合并时对各来源 `parent_anchor_final_score` 取 max（对应 JD 里更强的工业锚）。
    - 无分则 `best_parent_anchor_step2_rank`（Step2 列表名次，1 最强）→ 递减式 rank prior。
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

    # 8) 锚点主线先验：同结构下区分「背后锚是 JD 主轴」vs「弱锚/加分锚」（解决 ready 同化）
    best_fs = float(
        rec.get("best_parent_anchor_final_score")
        or rec.get("parent_anchor_final_score")
        or 0.0
    )
    if best_fs > 1e-6:
        apn = max(
            0.0,
            min(1.0, best_fs / PAPER_READINESS_ANCHOR_SCORE_NORM),
        )
        anchor_prior_mult = PAPER_READINESS_ANCHOR_PRIOR_LO + PAPER_READINESS_ANCHOR_PRIOR_HI * apn
    else:
        ar = int(
            rec.get("best_parent_anchor_step2_rank")
            or rec.get("parent_anchor_step2_rank")
            or 999
        )
        anchor_prior_mult = 1.0 / (
            1.0 + PAPER_READINESS_ANCHOR_RANK_LAMBDA * max(0, ar - 1)
        )
    paper_readiness *= anchor_prior_mult
    rec["paper_anchor_prior_mult"] = float(anchor_prior_mult)

    paper_select_score = float(rec.get("final_score") or 0.0) * paper_readiness
    rec["paper_readiness"] = float(paper_readiness)
    rec["paper_select_score"] = float(paper_select_score)


def select_terms_for_paper_recall(
    records: List[Dict[str, Any]],
    max_terms: int = 12,
) -> List[Dict[str, Any]]:
    """
    Paper 召回选词（重写）：
    - 仍阻断：conditioned_only + 单锚、fallback_primary（与 Stage2A 漂移锚对齐，纯安全网）。
    - **弱 support 结构门**（见 STAGE3_PAPER_SUPPORT_*）：在进 eligible 前拦截「support + primary_support_*」
      且锚/主线/grounding/可扩/conditioned 结构薄弱 的词；统一分只管排序，本门管「该不该上 paper bridge」。
    - **risky_side_block**（STAGE3_PAPER_RISKY_SIDE_*）：`stage3_bucket=risky` 且不可扩、无 mainline 命中、
      **`term_role` 为 side/expansion** 的词不进 paper，避免 motion controller 类尾词靠分+floor 混入。
    - **eligible 内主排序键**：`paper_select_score = final_score * paper_readiness`（readiness = 结构连乘 × **Step2 锚点分/名次先验**，无词表）；
      仍 **family_key 去重** + **dynamic_floor（与 paper_select_score 同尺度）** + **topN**。
    - `final_score` 仍为 Stage3 全局统一分；paper 侧分离「背后锚是否 JD 主轴」主要靠在 readiness 中显式吃进 **`best_parent_anchor_*`**。
    """
    if not records:
        return []

    used_family: Set[str] = set()
    selected: List[Dict[str, Any]] = []

    eligible: List[Dict[str, Any]] = []
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

        if conditioned_only and anchor_count <= 1:
            rec["paper_reject_reason"] = "conditioned_only_single_anchor_block"
            rec["paper_cutoff_reason"] = "conditioned_only_single_anchor_block"
            continue

        if is_fallback_primary:
            rec["paper_reject_reason"] = "fallback_primary_block"
            rec["paper_cutoff_reason"] = "fallback_primary_block"
            continue

        # ---- weak support contamination：入 eligible 前的结构 gate（复用 _paper_grounding_score）----
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
            rec["paper_support_gate"] = "weak_support_contamination_block"
            rec["paper_reject_reason"] = "weak_support_contamination_block"
            rec["paper_cutoff_reason"] = "weak_support_contamination_block"
            continue

        # ---- risky side / expansion 尾词：非主线、无 ml、不可扩，仅靠分数+floor 会混进 paper_support ----
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
            continue

        rec["paper_support_gate"] = "pass"

        fk = rec.get("family_key") or build_family_key(rec)
        rec["family_key"] = fk
        eligible.append(rec)

    # eligible 内：结构 readiness × final_score → paper 赛道排序（与全局 final_score 解耦）
    for rec in eligible:
        _apply_paper_readiness_for_recall(rec)

    floor = _paper_recall_dynamic_floor(eligible, score_key="paper_select_score")
    ordered = sorted(
        eligible,
        key=lambda r: float(r.get("paper_select_score") or 0.0),
        reverse=True,
    )

    for rec in ordered:
        pss = float(rec.get("paper_select_score") or 0.0)
        fk = str(rec.get("family_key") or "")
        bucket = (rec.get("stage3_bucket") or "").strip().lower()

        if pss < floor:
            rec["paper_reject_reason"] = "below_dynamic_floor"
            rec["paper_cutoff_reason"] = "below_dynamic_floor"
            continue

        if fk in used_family:
            rec["paper_reject_reason"] = "family_duplicate_block"
            rec["paper_cutoff_reason"] = "family_dup_or_below_cut"
            continue

        rec["selected_for_paper"] = True
        rec["paper_cutoff_reason"] = "top_n_paper_select_score"
        rec["select_reason"] = "paper_select_score_rank"
        rec["retrieval_role"] = "paper_primary" if bucket == "core" else "paper_support"
        used_family.add(fk)
        selected.append(rec)
        if len(selected) >= max_terms:
            break

    # 提前填满 max_terms 后，后续名次仅差一轮标注（便于 cutoff 表读因）
    for rec in ordered:
        if rec.get("selected_for_paper"):
            continue
        if not rec.get("paper_cutoff_reason"):
            rec["paper_cutoff_reason"] = "past_paper_recall_max_terms"

    if STAGE3_AUDIT_DEBUG:
        _print_stage3_to_paper_bridge(records, top_k=20)
    # cutoff / narrow 表与真实选词序一致（eligible · paper_select_score）
    _print_stage3_paper_cutoff_audit(ordered, floor, max_terms)
    _print_stage3_paper_selection_narrow_audit(ordered)
    _print_stage3_paper_near_miss_audit(ordered, selected, floor)
    _print_paper_term_selection_audit(
        sorted(
            records,
            key=lambda x: float(x.get("paper_select_score") or x.get("final_score") or 0.0),
            reverse=True,
        )
    )
    _print_stage3_support_contamination_summary(records)
    return selected


def stage3_build_score_map(survivors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stage3 第二段（重写）：**统一连续分**，取代原 bucket / primary_support_* / conditioned_only 链式乘子。

    - **已移除**：score_mult、按桶 cross_bonus、risky 额外乘子、support 子类分档等一切「走分支再乘」的逻辑。
    - **保留观测**：`stage3_bucket` 仍由 `_assign_stage3_bucket` 写入，仅用于日志与下游 `paper_primary`/`paper_support` 软区分，**不**再决定能不能进 paper。
    - **家族约束**：在统一分算完后调用 `_apply_family_role_constraints`，使 primary  Visibility 与连续分尺度一致。

    `stage3_explain.mainline_risk_penalty` / `cross_anchor_score_bonus` 固定为 1.0，兼容依赖字段的旧表。
    """
    for rec in survivors:
        rec["_stage3_pre_adjust_score"] = float(rec.get("final_score") or 0.0)
        unified, breakdown = _compute_stage3_unified_continuous_score(rec)
        rec["final_score"] = unified
        rec["stage3_unified_breakdown"] = breakdown
        ex = rec.get("stage3_explain") or {}
        ex["mainline_risk_penalty"] = 1.0
        ex["cross_anchor_score_bonus"] = 1.0
        ex["unified_continuous_score"] = unified
        ex["conditioned_only"] = _stage3_is_conditioned_only(rec)
        rec["stage3_explain"] = ex

    survivors.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    survivors = _apply_family_role_constraints(survivors)
    survivors.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)

    for i, rec in enumerate(survivors):
        rec["stage3_rank"] = i
        rec["bucket_reason_flags"] = _collect_stage3_bucket_reason_flags(rec)

    if STAGE3_UNIFIED_SCORE_DEBUG:
        _print_stage3_unified_breakdown(survivors, top_k=16)

    if DEBUG_LABEL_PATH:
        print("\n" + "-" * 80)
        print("[Stage3 rerank summary] unified_continuous + family role constraint")
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
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], List[Dict[str, Any]]]:
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
            rec["reject_reason"] = gate.get("reason") or "admission"
            dropped_with_reason.append(rec)
            if label_trace:
                print(f"[Stage3 准入拒绝] tid={rec.get('tid')} term={rec.get('term')!r} reason={rec['reject_reason']}")
            continue

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
        rec["stage3_bucket"] = _assign_stage3_bucket(rec)
        survivors.append(rec)

    if label_trace and topic_gate_bypass_primary_like_count and not STAGE3_DEBUG_FOCUS_TERMS:
        print(
            f"[Stage3 topic_gate] bypass primary-like count={topic_gate_bypass_primary_like_count} "
            f"(明细仅 STAGE3_DEBUG_FOCUS_TERMS 非空时逐条打印)"
        )

    survivors.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    for i, rec in enumerate(survivors):
        rec["stage3_rank"] = i

    # 批注：家族 primary/support 微调挪入 stage3_build_score_map，与统一连续分同尺度叠代，避免「先乘规则链再压家族」双套逻辑

    for rec in survivors:
        rec["risk_reasons"] = _collect_risky_reasons(rec)
        rec["bucket_reason_flags"] = _collect_stage3_bucket_reason_flags(rec)

    survivors = stage3_build_score_map(survivors)

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
    paper_terms = select_terms_for_paper_recall(survivors, PAPER_RECALL_MAX_TERMS)

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
    recall.debug_info.dropped_with_reason = dropped_with_reason
    return score_map, term_map, idf_map, term_role_map, term_source_map, parent_anchor_map, parent_primary_map, paper_terms


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


def run_stage3(
    recall,
    raw_candidates: List[Dict[str, Any]],
    query_vector,
    anchor_vids=None,
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float], Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str], List[Dict[str, Any]]]:
    """
    阶段 3：词权重。
    若候选含 term_role/identity_score（Stage2 新格式），走双闸门路径并返回 paper_terms（family 保送式选词）；否则走原有 _calculate_final_weights。
    返回 score_map, term_map, idf_map, term_role_map, term_source_map, parent_anchor_map, parent_primary_map, paper_terms。
    """
    if not raw_candidates:
        return {}, {}, {}, {}, {}, {}, {}, []

    use_dual_gate = (
        raw_candidates
        and ("term_role" in raw_candidates[0] or "identity_score" in raw_candidates[0])
    )
    paper_terms: List[Dict[str, Any]] = []
    if use_dual_gate:
        score_map, term_map, idf_map, term_role_map, term_source_map, parent_anchor_map, parent_primary_map, paper_terms = _run_stage3_dual_gate(
            recall, raw_candidates, query_vector, anchor_vids=anchor_vids
        )
    else:
        score_map, term_map, idf_map = recall._calculate_final_weights(
            raw_candidates, query_vector, anchor_vids=anchor_vids
        )
        term_role_map = {}
        term_source_map = {}
        parent_anchor_map = {}
        parent_primary_map = {}

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

    return score_map, term_map, idf_map, term_role_map, term_source_map, parent_anchor_map, parent_primary_map, paper_terms

