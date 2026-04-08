import collections
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np

from config import (
    DB_PATH,
    VOCAB_P95_PAPER_COUNT,
    SIMILAR_TO_TOP_K,
    SIMILAR_TO_MIN_SCORE,
    TOPIC_ALIGN_SUBFIELD,
    TOPIC_ALIGN_FIELD,
    TOPIC_ALIGN_NONE,
    TRUSTED_SOURCE_TYPES_FOR_DIFFUSION,
)
from src.core.recall.label_means.hierarchy_guard import (
    compute_hierarchical_fit,
    compute_purity,
)
from src.utils.domain_utils import DomainProcessor
from src.core.recall.label_means.label_debug import debug_print


def _cjk_len(s: str) -> int:
    """CJK 字符计数（仅用于局部文本收口/长度判断）。"""
    if not s:
        return 0
    try:
        return sum(1 for ch in str(s) if "\u4e00" <= ch <= "\u9fff")
    except Exception:
        return 0

# ---------- Stage2/3 保守常量：单一决策链，无冗余阈值（详见 README） ----------
LABEL_EXPANSION_DEBUG = True  # 调试时打印 Stage2A/2B 流程（观测用途）
STAGE2_VERBOSE_DEBUG = True   # True 时输出 Stage2 详细工整表格，便于调试
# 关键内部决策点（Stage2 内部用途）：Stage2A 选主因子 / Stage2B dense·cooc 漏斗（可与 STAGE2_VERBOSE_DEBUG 独立开关）
# 注意：Stage2A 的细桶体系（primary_bucket 等）用于“锚内选主/Stage2B 扩展资格/调试摘要”，
# 不是跨阶段正式语义。跨阶段推荐以 Stage2 输出的 local_role + 局部证据字段为主。
STAGE2_RULING_DEBUG = False
# 噪声较大的逐候选/逐阶段打印（SIMILAR_TO 明细、dual evidence、fallback cand、seed factors、dense/cooc funnel 等）
# 默认关闭；仅在深度诊断时临时打开。
STAGE2_NOISY_DEBUG = False

# Stage2A 定点调试：仅命中下列锚点/学术词时打印 Focus 块（当前瓶颈在 Stage3 时仍以本集合为「关注词」主线）
DEBUG_STAGE2A_FOCUS_TERMS = frozenset(
    {
        "motion control",
        "robot control",
        "digital control",
        "route planning",
        "path planning",
        "reinforcement learning",
        "robotic arm",
    }
)
DEBUG_STAGE2A_FOCUS_ANCHORS = frozenset({"运动控制", "机器人运动控制", "传统控制", "Robot control"})
_DEBUG_STAGE2A_FOCUS_ANCHORS_LOWER = frozenset(x.lower() for x in DEBUG_STAGE2A_FOCUS_ANCHORS)
# [Stage2A merge evidence detail]：默认只打 support_seed/support_keep/近似 conditioned_only；非空时按 term 字面补行
STAGE2A_MERGE_EVIDENCE_DETAIL_FOCUS_TERMS: Set[str] = set()


def _stage2a_candidate_conditioned_only_sources(cand: Any) -> bool:
    """合并来源上近似 Stage3 conditioned_only（仅 conditioned_vec，无 similar_to / family_landing）。"""
    ss = getattr(cand, "source_set", None)
    if ss:
        stl = {str(x).strip().lower() for x in ss if x is not None}
    else:
        stl = {(getattr(cand, "source", "") or "").strip().lower()} - {""}
    if not stl:
        return False
    return "conditioned_vec" in stl and "similar_to" not in stl and "family_landing" not in stl


def _should_emit_stage2a_merge_evidence_detail(cand: Any, group_name: str) -> bool:
    if group_name in ("primary_support_seed", "primary_support_keep"):
        return True
    if _stage2a_candidate_conditioned_only_sources(cand):
        return True
    focus = STAGE2A_MERGE_EVIDENCE_DETAIL_FOCUS_TERMS
    return bool(focus and (cand.term or "").strip() in focus)


def _is_stage2a_focus_case(anchor_term: str, term: str) -> bool:
    a = (anchor_term or "").strip().lower()
    t = (term or "").strip().lower()
    return t in DEBUG_STAGE2A_FOCUS_TERMS or a in _DEBUG_STAGE2A_FOCUS_ANCHORS_LOWER


def _debug_stage2a_focus(anchor_term: str, term: str, bucket: str, reason: str, snap: Dict[str, Any]) -> None:
    # 大块 Focus：仅在开启 NOISY 且命中焦点词时打出；**调用方**若需「仅 reconcile 跨桶」，由 select_primary_per_anchor 在 pre≠final 时再调。
    if (
        not LABEL_EXPANSION_DEBUG
        or not STAGE2_NOISY_DEBUG
        or not _is_stage2a_focus_case(anchor_term, term)
    ):
        return
    print("\n" + "=" * 72)
    print("[Stage2A Focus Debug]")
    print("=" * 72)
    print(f"anchor={anchor_term!r} term={term!r}")
    print(f"bucket={bucket!r} reason={reason!r}")
    print(
        f"final_expandable_strong={bucket == 'primary_expandable'} "
        f"final_weak_seed={bucket == 'primary_support_seed'}"
    )

    print("\n[基础相似度]")
    print(f"  source_type={snap.get('source_type')!r}")
    print(f"  static_sim={float(snap.get('static_sim', 0.0) or 0.0):.3f}")
    print(f"  ctx_sim={float(snap.get('ctx_sim', 0.0) or 0.0):.3f}")
    print(f"  ctx_present={bool(snap.get('ctx_present', False))}")
    print(f"  ctx_drop={float(snap.get('ctx_drop', 0.0) or 0.0):.3f}")
    print(f"  best_sim={float(snap.get('best_sim', 0.0) or 0.0):.3f}")

    print("\n[Primary 判定]")
    print(f"  semantic_ok={bool(snap.get('semantic_ok', False))}")
    print(f"  context_path_ok={bool(snap.get('context_path_ok', False))}")
    print(f"  structural_dual={bool(snap.get('structural_dual', False))}")
    print(f"  structural_multi={bool(snap.get('structural_multi', False))}")
    print(f"  dual_support={bool(snap.get('dual_support', False))}")
    print(f"  multi_anchor_ok={bool(snap.get('multi_anchor_ok', False))}")
    print(f"  family_match={float(snap.get('family_match', 0.0) or 0.0):.3f}")
    print(f"  primary_ok={bool(snap.get('primary_ok', False))}")

    print("\n[Expand 判定]")
    print(f"  drop_ok_primary={bool(snap.get('drop_ok_primary', False))}")
    print(f"  drop_ok_exp={snap.get('drop_ok_exp')!r}")
    print(f"  branch_blocked={snap.get('branch_blocked')!r}")
    print(f"  object_like_risk={float(snap.get('object_like_risk', 0.0) or 0.0):.3f}")
    print(f"  generic_risk={float(snap.get('generic_risk', 0.0) or 0.0):.3f}")
    print(f"  polysemy_risk={float(snap.get('polysemy_risk', 0.0) or 0.0):.3f}")
    print(f"  base_ctx_ok={bool(snap.get('base_ctx_ok', False))}")
    print(f"  dual_expand_ok={bool(snap.get('dual_expand_ok', False))}")
    print(f"  multi_expand_ok={bool(snap.get('multi_expand_ok', False))}")
    print(f"  solo_ctx_expand={bool(snap.get('solo_ctx_expand', False))}")
    print(f"  strong_mainline_direct={bool(snap.get('strong_mainline_direct', False))}")
    print(f"  snap.can_expand_from_2a={bool(snap.get('can_expand_from_2a', False))}")
    print(f"  expand_block={snap.get('expand_block')!r}")
    print("=" * 72)


def _debug_stage2a_commit(anchor_term: str, c: Any) -> None:
    # 与 Focus / reconcile 重复信息多；默认关，仅 STAGE2_NOISY_DEBUG 且命中 focus 时打出。
    if not LABEL_EXPANSION_DEBUG or not STAGE2_NOISY_DEBUG:
        return
    term = (getattr(c, "term", "") or "").strip()
    if not _is_stage2a_focus_case(anchor_term, term):
        return
    print(
        f"[Stage2A Commit Debug] "
        f"anchor={anchor_term!r} term={term!r} "
        f"primary_bucket={getattr(c, 'primary_bucket', None)!r} "
        f"survive_primary={bool(getattr(c, 'survive_primary', False))} "
        f"can_expand={bool(getattr(c, 'can_expand', False))} "
        f"can_expand_from_2a={bool(getattr(c, 'can_expand_from_2a', False))} "
        f"admission_reason={getattr(c, 'admission_reason', None)!r}"
    )


def _stage2_header(title: str, char: str = "=") -> None:
    """Stage2 调试：打印一节标题，工整分隔。"""
    if not (LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG):
        return
    line = char * 72
    print(f"\n{line}\n  [Stage2] {title}\n{line}")


def _stage2_table(rows: List[List[str]], header: List[str], col_widths: Optional[List[int]] = None) -> None:
    """Stage2 调试：打印一张对齐的表格。"""
    if not (LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG) or not header:
        return
    if not col_widths:
        col_widths = [max(len(str(h)), 4) for h in header]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(f"  {fmt.format(*header)}")
    print("  " + "-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows:
        row_str = [str(x)[:w] for x, w in zip(row, col_widths)]
        if len(row_str) < len(col_widths):
            row_str.extend([""] * (len(col_widths) - len(row_str)))
        print(f"  {fmt.format(*row_str[:len(col_widths)])}")


def _print_stage2a_global_bucket_summary(all_anchor_results: List[Dict[str, Any]], max_rows: int = 120) -> None:
    """
    [Stage2A global bucket summary]
    跨锚一行一词：与五层桶、stage2b_seed_tier、来源集合对齐，便于对照「弱 seed / support_keep」为何分流。
    """
    if not (LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG) or not all_anchor_results:
        return
    rows_out: List[Tuple[str, str, str, str, str, str]] = []
    for block in all_anchor_results:
        anchor = block.get("anchor")
        anc = getattr(anchor, "anchor", "") or ""
        for group_name in (
            "primary_expandable",
            "primary_support_seed",
            "primary_support_keep",
            "risky_keep",
        ):
            for cand in block.get(group_name, []):
                term = (getattr(cand, "term", "") or "")[:40]
                pb = str(getattr(cand, "primary_bucket", None) or group_name)
                tier = str(getattr(cand, "stage2b_seed_tier", None) or "none")
                src = getattr(cand, "source_set", None)
                if isinstance(src, (set, frozenset)):
                    src_types = ",".join(sorted(str(x) for x in src))
                elif src is None:
                    st = getattr(cand, "source_type", None) or getattr(cand, "source", "")
                    src_types = str(st or "")
                else:
                    src_types = str(src)
                if len(src_types) > 26:
                    src_types = src_types[:24] + ".."
                reason = (
                    getattr(cand, "mainline_block_reason", None)
                    or getattr(cand, "bucket_reason", None)
                    or getattr(cand, "admission_reason", "")
                    or ""
                )
                reason_str = str(reason)[:44]
                rows_out.append((term, pb, tier, src_types, anc, reason_str))
    if not rows_out:
        return
    print("\n" + "-" * 80)
    print("[Stage2A global bucket summary]")
    print("-" * 80)
    print("term                         | primary_bucket       | tier     | source_types         | parent_anchor  | reason")
    for r in rows_out[:max_rows]:
        term, pb, tier, src_types, anc, reason_str = r
        print(
            f"{term[:28]:<28} | {pb[:20]:<20} | {tier[:8]:<8} | {src_types[:22]:<22} | {str(anc)[:14]:<14} | {reason_str}"
        )
    if len(rows_out) > max_rows:
        print(f"  ... ({len(rows_out) - max_rows} more rows truncated)")


def _print_stage2b_seed_tier_audit(anchor_text: str, seed_rows: List[Tuple[Any, bool, float, Optional[str]]]) -> None:
    """
    [Stage2B seed tier audit]
    逐项对照 stage2b_seed_tier（strong/weak/none）与 check_seed_eligibility 结果，避免「2A 标了弱 seed 但 2B 又挡掉」时无从查因。
    """
    if not (LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG) or not seed_rows:
        return
    print("\n" + "-" * 80)
    print(f"[Stage2B seed tier audit] anchor={anchor_text!r}")
    print("-" * 80)
    for p, eligible, score, reason in seed_rows:
        tier = (getattr(p, "stage2b_seed_tier", None) or "none").strip().lower()
        term = getattr(p, "term", "") or ""
        detail = "ok" if eligible else (str(reason) if reason else "blocked")
        print(
            f"  term={term!r} tier={tier!r} eligible={eligible} score={score:.4f} detail={detail!r}"
        )


def _split_stage2b_carryover_and_seed_candidates(primary_landings: Optional[List[Any]]) -> Tuple[List[Any], List[Any]]:
    """
    Stage2B 输入拆口径（仅主流程命名，不改 2A 判桶）：
    - carryover_terms：2A 非 reject 的全部保留 landing，供 merge 回本锚输出并进入 Stage3；
    - seed_candidates：Stage2B 的“扩展起点（support expansion seed）”候选，入口契约如下：
      - 首选：primary_support_seed（弱 seed）与 primary_expandable（强 seed）；
      - primary_support_keep / primary_keep_no_expand / risky_keep：默认仅 carryover，不作为 seed；
      - 即便出现异常标记（tier=strong/weak），也会被入口侧二次约束挡在 seed 外（只做扩展资格收口，不改 2A 分桶公式）。

    备注：
    - 返回类型保持不变：Tuple[carryover_terms, seed_candidates]；
    - “谁被挡在 seed 外、原因是什么”由 Stage2B 日志解释（见 [Stage2B input audit] / [Stage2B no-seed reason]）。
    """
    carryover_terms = list(primary_landings or [])
    seed_candidates: List[Any] = []
    blocked: List[Tuple[str, str]] = []  # (reason, primary_bucket)
    for p in carryover_terms:
        tier = (getattr(p, "stage2b_seed_tier", None) or "none").strip().lower()
        pb = (getattr(p, "primary_bucket", None) or "").strip()
        # 收紧入口语义：seed 只接受 2A 的“可扩/弱 seed”桶；其余桶仅 carryover
        if pb not in ("primary_expandable", "primary_support_seed"):
            if tier in ("strong", "weak"):
                blocked.append(("bucket_not_seed", pb or "(empty)"))
            continue
        if tier in ("strong", "weak"):
            seed_candidates.append(p)
        else:
            blocked.append(("tier_none_after_2a_finalize", pb or "(empty)"))

    # 入口审计：仅在 debug 下打印，避免刷屏；帮助解释“谁进入 seed、谁被挡在 seed 外”
    if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG and carryover_terms:
        blocked_counts: Dict[str, int] = {}
        for r, _pb in blocked:
            blocked_counts[r] = blocked_counts.get(r, 0) + 1
        seed_pb_counts: Dict[str, int] = {}
        for p in seed_candidates:
            _pb = (getattr(p, "primary_bucket", None) or "").strip() or "(empty)"
            seed_pb_counts[_pb] = seed_pb_counts.get(_pb, 0) + 1
        print(
            f"[Stage2B seed entry contract] carryover={len(carryover_terms)} "
            f"seed_candidates={len(seed_candidates)} seed_bucket_counts={seed_pb_counts} "
            f"blocked_counts={blocked_counts}"
        )
    return carryover_terms, seed_candidates


def _print_stage2b_input_audit(anchor_text: str, carryover_terms: List[Any], seed_candidates: List[Any]) -> None:
    """carryover vs seed_candidates：与下面的 diffusion 分母对齐，避免「input=3 却只扩 1」的误读。"""
    if not LABEL_EXPANSION_DEBUG:
        return

    def _pb(p: Any) -> str:
        return (getattr(p, "primary_bucket", "") or "").strip()

    exp = sum(1 for p in carryover_terms if _pb(p) == "primary_expandable")
    sd = sum(1 for p in carryover_terms if _pb(p) == "primary_support_seed")
    sk = sum(
        1 for p in carryover_terms
        if _pb(p) in ("primary_support_keep", "primary_keep_no_expand", "primary_fallback_keep_no_expand")
    )
    rk = sum(1 for p in carryover_terms if _pb(p) == "risky_keep")
    print(
        f"[Stage2] 锚点 anchor={anchor_text!r} carryover_terms={len(carryover_terms)} "
        f"(exp={exp} sd={sd} sk={sk} rk={rk})"
    )
    if not STAGE2_VERBOSE_DEBUG:
        return
    seed_names = [getattr(p, "term", "") for p in seed_candidates[:5]]
    print(
        f"[Stage2B input audit] anchor={anchor_text!r} "
        f"seed_candidates={len(seed_candidates)} sample={seed_names!r}"
    )


def _print_stage2b_no_seed_reason(anchor_text: str, carryover_terms: List[Any]) -> None:
    """
    seed_candidates=0 时一眼可读：是「全是 sk/rk」还是「桶名像 seed 但 tier 已在 2A 收口成 none」。
    不替代 [Stage2A anchor bucket summary]，只为 2B 长日志省掉回头翻 2A 的成本。
    """
    if not LABEL_EXPANSION_DEBUG or not carryover_terms:
        return

    def _pb(p: Any) -> str:
        return (getattr(p, "primary_bucket", "") or "").strip()

    tag_list: List[str] = []
    for p in carryover_terms:
        pb = _pb(p)
        if pb == "primary_expandable":
            tag_list.append("exp")
        elif pb == "primary_support_seed":
            tag_list.append("sd")
        elif pb in ("primary_support_keep", "primary_keep_no_expand", "primary_fallback_keep_no_expand"):
            tag_list.append("sk")
        elif pb == "risky_keep":
            tag_list.append("rk")
        else:
            tag_list.append("ot")
    uniq = sorted(set(tag_list))
    buckets = ",".join(uniq)
    has_seedish_bucket = any(t in ("exp", "sd") for t in tag_list)
    if not has_seedish_bucket:
        reason = "all_carryover_are_nonseed_buckets"
    else:
        reason = "seedish_bucket_but_tier_none_after_2a_finalize"
    print(
        f"[Stage2B no-seed reason] anchor={anchor_text!r} reason={reason!r} buckets={buckets!r} "
        f"(seed_pref=primary_support_seed; keep/risky carryover_only)"
    )


PRIMARY_MIN_IDENTITY = 0.62       # Stage2A 准入：identity 下限（与 PRIMARY_MIN_PATH_MATCH 共同构成准入）
PRIMARY_MAX_PER_ANCHOR = 2        # 每锚点最多 primary 数
PRIMARY_TOP_M_PER_ANCHOR = 8      # 每锚先保留 top-m 候选，后面严判不早剪枝
CONDITIONED_VEC_TOP_K = 12        # 每锚点 conditioned_vec 检索学术词 top-k，与 SIMILAR_TO 合并
# Stage2A 采集阶段放宽 top_k，主线词可能排第 4～8
STAGE2A_COLLECT_BASE_TOP_K = 8   # base 视角取 6～8，严判在后
STAGE2A_COLLECT_CONDITIONED_TOP_K = 8
STAGE2A_COLLECT_CONDITIONED_TOP_K_RESCUE = 12  # similar_to 空池或极弱时 conditioned_vec 多拿几条
# 综述式收缩：conditioned 以「局部分数 + 极弱池少量补召回」为主，不做第二主召回器
CONDITIONED_SUPPLEMENT_MAX = 3
STAGE2A_SIMILAR_TO_WEAK_MAX_N = 2
STAGE2A_SIMILAR_TO_WEAK_SIM = 0.79
STAGE2A_FAMILY_LANDING_TOP_K = 6  # canonical_academic_like 锚点 family 先手补池条数
# 动力学/运动学/仿真等锚点 → 轻量 family 查询词（补池用，不保送 mainline）
CANONICAL_ACADEMIC_ANCHOR_FAMILY_QUERIES: Dict[str, List[str]] = {
    "动力学": ["dynamics", "robot dynamics", "rigid body dynamics", "inverse dynamics", "multibody dynamics"],
    "运动学": ["kinematics", "robot kinematics"],
    "仿真": ["simulation", "simulations"],
    "振动抑制": ["vibration", "vibration suppression", "vibration control"],
    "动力学参数辨识": ["dynamics", "parameter identification"],
    "路径规划": ["path planning", "motion planning", "trajectory planning"],
    "抓取": ["grasping", "grasp"],
    "端到端": ["end-to-end", "end to end"],
}
SEED_MIN_IDENTITY = 0.65         # Stage2B seed 准入：唯一常量，不与其他阈值叠加
SEED_CTX_MIN = 0.62              # Stage2B seed：identity 路径下的最小上下文阈值
SEED_SCORE_STRONG = 0.68         # Stage2B seed：dual_support 路径下的强 seed 分数线
DENSE_MAX_PER_PRIMARY = 4
DENSE_PARENT_CAP = 0.85   # Stage2B：dense 扩展词 keep_score 不超过 parent primary 的此比例，避免 Motion controller > motion control
CLUSTER_MAX_PER_PRIMARY = 3
COOC_SUPPORT_MIN_FREQ = 2
COOC_MAX_PER_PRIMARY = 2

# ---------- Stage2A 终稿：候选分型 + 组内选主（不硬编码、不训练集） ----------
CTX_SUPPORT_MIN = 0.70           # 上下文支持阈值，用于 family 分型与 scene_shift 判断
CTX_GAP_SHIFT_MIN = 0.15          # 语义-上下文差超过此视为场景漂移
HIER_WEAK_MIN = 0.25              # 层级证据弱于此为 weak
JD_ALIGN_WEAK_MIN = 0.45          # JD 对齐弱于此为 weak
GENERIC_RISK_MIN = 0.45           # 泛词风险高于此为 generic_like
POLY_RISK_MIN = 0.55              # 多义风险高于此为 generic_like
MAINLINE_IDENTITY_MIN = 0.50      # 主线准入 identity 下限
RETAIN_MIN = 0.40                 # 仅保留（不扩）准入 retain_score 下限
# Stage2A 终稿：三档分桶阈值（主线回 primary_expandable，歧义/弱词压住）
STAGE2A_MAINLINE_PREF_MIN = 0.50   # 主线偏好下限 → primary_expandable
STAGE2A_EXPANDABLE_IDENTITY_MIN = 0.52  # 锚点一致性下限 → 可扩散
STAGE2A_WEAK_KEEP_MIN = 0.20       # 文档/门槛与 PRIMARY_KEEP 对齐；不再用于 select_primary_per_anchor 前置 reject
# Stage2A pre-primary 分桶：明显坏分支 hard_reject，弱相关技术词 primary_keep_no_expand
STAGE2A_REJECT_IDENTITY_FLOOR = 0.33   # low_identity_no_context 阈值
STAGE2A_REJECT_MAINLINE_FLOOR = 0.40   # low_mainline_no_context 阈值
STAGE2A_POLY_HARD_RISK = 0.55         # 高歧义 + 无上下文 + identity<0.40 → hard_reject
# 明显坏分支（对象/多义 或 抽象漂移）：dynamism/propulsion/surgical robot/control flow 等
STAGE2A_BAD_OBJECT_OR_POLY_ID_CAP = 0.28
STAGE2A_BAD_OBJECT_OR_POLY_CTX_CAP = 0.50
STAGE2A_BAD_OBJECT_MIN = 0.18
STAGE2A_BAD_POLY_MIN = 0.45
STAGE2A_BAD_OBJECT_OR_POLY_HIER_CAP = 0.30
STAGE2A_DRIFT_ID_CAP = 0.32
STAGE2A_DRIFT_CTX_CAP = 0.50
STAGE2A_DRIFT_MAINLINE_CAP = 0.43
STAGE2A_DRIFT_JD_CAP = 0.79
STAGE2A_DRIFT_HIER_CAP = 0.55
# 弱相关但技术主干词保留：simulation/feedback control/vibration/mechanics 救回 keep_no_expand
STAGE2A_WEAK_TECH_ID_CAP = 0.45
STAGE2A_WEAK_TECH_CTX_CAP = 0.55
STAGE2A_WEAK_TECH_JD_MIN = 0.74
STAGE2A_WEAK_TECH_POLY_CAP = 0.40
STAGE2A_WEAK_TECH_OBJECT_CAP = 0.18
# Stage2A 可扩散新规则（不再强依赖 context_gain>0）：双路一致/静态强匹配/多锚一致 均可扩
STAGE2A_ID_MAIN = 0.52            # 双路一致主线 anchor_identity 下限（= EXPANDABLE_IDENTITY_MIN）
STAGE2A_ID_STRONG = 0.58         # 静态强匹配主线 anchor_identity 下限
STAGE2A_ID_MULTI = 0.48          # 多锚一致主线 anchor_identity 下限
STAGE2A_ID_WEAK = 0.45           # mainline_candidate 弱线（identity_ok 或 jd 分支）
STAGE2A_JD_OK = 0.74             # 与 STAGE2A_WEAK_TECH_JD_MIN 一致，主线 jd_align 可用线
STAGE2A_CTX_FLOOR = 0.42         # 双路/静态时 context_sim 最低可用线
STAGE2A_CTX_DROP_TOL = 0.08      # 静态强匹配允许 context_sim 略低于 semantic_score 的容差
STAGE2A_PRIMARY_KEEP_MIN = STAGE2A_WEAK_KEEP_MIN  # 保守留存门槛（judge/弱保留链）；非 select 前置 reject
SEED_SCORE_MIN = 0.50              # Stage2B：check_seed 通过后按 seed_score≥此裁剪（仅 can_expand_from_2a 可过 check）
SEED_SCORE_MIN_WEAK = 0.54         # Stage2B：主线近邻弱 seed（primary_support_seed）略高的 seed_score 下限
# 与 check_seed_eligibility 中 strong 的 axis_consistency_seed 下限一致；select 仅当过此线才维持/升格 primary_expandable
SEED_AXIS_CONSISTENCY_STRONG_MIN = 0.45
# keep/seed 救回 expandable：须为组内 axis_consistency_seed 第 1，且与第 2 名间隔≥此，避免 supervised/hand 与真主线并列升格
STAGE2A_PROMOTE_MIN_AXIS_GAP = 0.03
DENSE_MAX_PER_PRIMARY_WEAK = 2     # Stage2B：弱 seed 的 dense 每 primary 上限（强 seed 见 DENSE_MAX_PER_PRIMARY）

# ---------- Stage2A 五层落点（内部细桶；降级为“内部用途”，不作为跨阶段正式语义） ----------
# primary_expandable / primary_support_seed / primary_support_keep / risky_keep / reject
# 用途限定：
# - anchor 内部选主（组内相对排序 + 配额）
# - Stage2B 扩展资格判断（strong/weak seed 的来源）
# - debug / summary（观测与验收）
# 对外（跨阶段）语义：以 local_role（core/support/risky）+ 局部证据字段为主。
# 设计说明见 README「Stage2A 五层落点体系」；以下为与伪代码对齐的数值门（与 expand 的 branch_blocked 阈值解耦）。
STAGE2A_WEAK_SEED_GENERIC_CAP = 0.65   # 弱 seed：generic 须低于此（高于 expand 的 0.46，避免 robot control 被误打成 risky）
STAGE2A_WEAK_SEED_POLY_CAP = 0.35
STAGE2A_WEAK_SEED_OBJ_CAP = 0.35
STAGE2A_RISKY_KEEP_HIGH_GENERIC = 0.65  # 泛词风险「高」→ risky_keep（与弱 seed 上界衔接）
STAGE2A_RISKY_KEEP_HIGH_POLY = 0.55
STAGE2A_RISKY_KEEP_MIN_BEST_SIM = 0.36  # 未达 primary_ok 时，best_sim 高于此可落 risky_keep 而非直接 reject

# ---------- Stage2 最小收尾：primary（宽）与 expand（严）解耦 + 空锚 fallback ----------
# primary：禁止「仅 static 高相似 + 无 conditioned_sim」冒充主线（ctx_drop=0 不算支持）
STAGE2A_CONDITIONED_SIM_EPS = 1e-6  # conditioned_sim 存在且 > 此视为「真实上下文纠偏分」
STAGE2A_PRIMARY_MIN_SIM = 0.38
STAGE2A_PRIMARY_MAX_CTX_DROP = 0.14
STAGE2A_PRIMARY_MIN_CTX_SIM = 0.36  # 真实 conditioned 路径下 ctx 下限
STAGE2A_PRIMARY_DUAL_MIN_CTX = 0.34  # dual_support 且已有 conditioned 时略宽
STAGE2A_PRIMARY_MULTI_MIN_FAMILY = 0.46  # 多锚救援：仅 similar_to 无 ctx 时，须更高 family 一致性
# expand：无 conditioned_sim → 默认不可扩；且挡 object/generic/poly 旁枝
STAGE2A_EXPAND_MIN_CTX_SIM = 0.46
STAGE2A_EXPAND_DUAL_MIN_CTX = 0.46
STAGE2A_EXPAND_MULTI_MIN_CTX = 0.47
STAGE2A_EXPAND_MAX_CTX_DROP = 0.10
STAGE2A_EXPAND_MAX_OBJECT_LIKE = 0.38
STAGE2A_EXPAND_MAX_GENERIC_RISK = 0.46
STAGE2A_EXPAND_MAX_POLY_RISK = 0.54
# 单锚无 multi、dual 标志异常时：仅当 conditioned 极强 + family 够稳才给 expandable
STAGE2A_EXPAND_SOLO_STRONG_CTX = 0.52
STAGE2A_EXPAND_SOLO_MIN_FAMILY = 0.50
# 仅当本锚点 0-primary 时，对高质量 canonical 锚点做一条「保线」fallback（不扩、交 Stage3）
STAGE2A_FALLBACK_ANCHOR_MIN_SCORE = 0.35
STAGE2A_FALLBACK_MIN_LOCAL_HITS = 1
STAGE2A_FALLBACK_MIN_CO_HITS = 1
STAGE2A_FALLBACK_MIN_SIM = 0.32

# ---------- Stage2A 准入：仅两个门槛 + source 折扣，不再按 source 分多套阈值 ----------
PRIMARY_MIN_HIERARCHY_MATCH = 0.30   # hierarchy_match = 0.4*topic + 0.4*path + 0.2*subfield，effective = hierarchy_match * source_factor
PRIMARY_MIN_PATH_MATCH = 0.35
CONDVEC_SOURCE_FACTOR = 0.85         # conditioned_vec 来源时 effective_hierarchy = hierarchy_match * 0.85；similar_to = 1.0
PRIMARY_RESCUE_CROSS_ANCHOR_MIN = 2  # rescue：多锚支持 + path/jd/semantic 足够时可准入
PRIMARY_RESCUE_PATH_MATCH_MIN = 0.45
PRIMARY_RESCUE_JD_ALIGN_MIN = 0.80
PRIMARY_RESCUE_SEMANTIC_MIN = 0.78
HIERARCHY_LEVEL_TOPIC_EXACT = "topic_exact"
HIERARCHY_LEVEL_SUBFIELD_MATCH = "subfield_match"
HIERARCHY_LEVEL_FIELD_ONLY = "field_only"
HIERARCHY_LEVEL_OFF_PATH = "off_path"
TOPIC_SPAN_PENALTY_FACTOR = 0.2   # 泛化软惩罚：topic_span_penalty = 1/(1 + factor * max(0, span-1))
DOMAIN_SPAN_EXTREME = 24          # support 扩散仅在极端异常时硬拒绝（>24）；泛化主要由 topic_span_penalty 表达
SUPPORT_MIN_DOMAIN_FIT = 0.20     # Stage2B support 准入：domain_fit 低于此不进词池（唯一 support 门槛）
# ---------- Dense 最小修复补丁：support 锚点语义复核四道门（不靠硬编码） ----------
DENSE_SUPPORT_PRIMARY_CONSISTENCY_MIN = 0.72   # 候选与 parent primary 向量一致性
DENSE_SUPPORT_ANCHOR_CONSISTENCY_MIN = 0.70   # 候选与 anchor conditioned_vec 一致性
DENSE_SUPPORT_CONTEXT_STABILITY_MIN = 0.72    # 候选在 anchor context 邻域中的稳定性
DENSE_SUPPORT_FAMILY_SUPPORT_MIN = 0.68      # 候选对同锚 surviving primary family 的支撑
# 强冲突领域 ID：term 主领域在此集且与激活领域无交时返回 domain_conflict_strong（医学/社科/管理等），空集则所有 domain_no_match 仅做 soft retain
STRONG_CONFLICT_DOMAIN_IDS: Set[str] = set()
# ---------- Primary 排序：只保留一套权重（PRIMARY_SCORE_W_*），不再使用 PRIMARY_W_* ----------
PRIMARY_SCORE_W_SEMANTIC = 0.22
PRIMARY_SCORE_W_IDENTITY = 0.18
PRIMARY_SCORE_W_JD_ALIGN = 0.18
PRIMARY_SCORE_W_CROSS_ANCHOR = 0.10
PRIMARY_SCORE_W_NEIGHBOR = 0.08
PRIMARY_SCORE_W_FIELD = 0.10
PRIMARY_SCORE_W_SUBFIELD = 0.18
PRIMARY_SCORE_W_TOPIC = 0.24
PRIMARY_SCORE_W_PATH = 0.18
PRIMARY_SCORE_W_SPECIFICITY = 0.10

# ---------- Stage2A 四分桶：同源双视角落点稳定性裁判（详见 README） ----------
# 全局底线阈值（相对阈值在 calibrate_anchor_thresholds 中按锚点内分布计算）
STAGE2A_GLOBAL_FLOOR = SimpleNamespace(
    identity_low=0.18,
    identity_primary=0.35,
    identity_keep=0.40,
    identity_expand=0.55,
    view_stability_low=0.25,
    view_stability_primary=0.45,
    view_stability_keep=0.50,
    view_stability_expand=0.60,
    hierarchy_low=0.20,
    hierarchy_mid=0.35,
    hierarchy_expand=0.50,
    shift_quality_low=0.20,
    shift_quality_mid=0.40,
    shift_quality_expand=0.55,
    primary_keep_line=0.35,
    primary_expand_line=0.50,
    primary_keep=0.35,
    primary_expand=0.50,
    canonical_expand=0.45,
    ambiguity_high=0.70,
    ambiguity_mid=0.50,
    ambiguity_low=0.35,
    generic_mid=0.45,
    generic_low=0.35,
    branch_drift_high=0.65,
    branch_drift_mid=0.45,
    branch_drift_low=0.35,
    jd_align_mid=0.55,
    base_expand_line=0.55,
    max_reasonable_shift=0.25,
    max_reasonable_rank_gap=8,
    max_useful_gain=0.20,
    max_tolerable_drop=0.15,
    mainline_low=0.25,
    mainline_keep=0.40,
    mainline_expand=0.55,
    object_like_low=0.35,
)
# primary_score 权重（同源双视角统一打分）
STAGE2A_WEIGHTS = SimpleNamespace(
    base=0.28,
    identity=0.22,
    hierarchy=0.18,
    jd_align=0.14,
    view_stability=0.10,
    shift_quality=0.08,
    ambiguity_penalty=0.35,
    generic_penalty=0.30,
    branch_penalty=0.35,
)


@dataclass
class PreparedAnchor:
    """Stage1 输出，Stage2 输入。无缩写扩写表时 expanded_forms 仅 [anchor]。conditioned_vec 为 JD 上下文条件化表示。"""
    anchor: str
    vid: int
    anchor_type: str = "unknown"
    expanded_forms: List[str] = field(default_factory=list)
    conditioned_vec: Optional[np.ndarray] = None  # 条件化锚点向量，用于 Stage2A 落点打分
    source_type: str = "skill_direct"    # skill_direct | jd_vector_supplement
    source_weight: float = 1.0          # 补充锚点用较低权重，primary 打分时乘此值
    # 上下文条件化检索：支撑 context_gain 与双路证据
    local_phrases: List[str] = field(default_factory=list)
    co_anchor_terms: List[str] = field(default_factory=list)
    jd_snippet: str = ""
    surface_vec: Optional[np.ndarray] = None   # 裸词向量（vocab 或 encode(anchor)）
    conditioned_text: str = ""                 # mention+局部上下文，用于编码 conditioned_vec（可观测）
    conditioning_mode: str = "unknown"         # real_context_encoded | precomputed_stage1 | fallback_surface
    surface_text: str = ""                     # 与词面对齐的 surface 字符串（通常等于 anchor）
    surface_conditioned_cosine: Optional[float] = None  # 观测用：surface_vec 与 conditioned_vec 余弦
    light_conditioned_text: str = ""  # 轻量条件化句，用于 retrieval backoff 二次编码（非 surface 拷贝）


@dataclass
class ConditionedTextBundle:
    """强/轻两路局部文本 + 实际参与拼接的片段（可观测、可调试）。"""
    strong_text: str
    light_text: str
    selected_local_phrases: List[str]
    selected_co_anchor_terms: List[str]
    selected_jd_window: str
    selected_hints: List[str]


@dataclass
class LandingCandidate:
    """Stage2A 落点候选。similar_to 初始近邻 + conditioned_vec 上下文纠偏。"""
    vid: int
    term: str
    source: str  # similar_to | conditioned_vec
    semantic_score: float
    anchor_vid: int = 0
    anchor_term: str = ""

    # ===== Stage2A 上下文纠偏 / 准入 / 打分 =====
    context_sim: float = 0.0                 # conditioned_vec 下的相似度；若无则 0
    context_supported: bool = False          # 是否得到上下文支持
    context_gap: float = 1.0                 # raw_sim - context_sim 的差，越大越可疑
    source_role: str = "seed_candidate"     # seed_candidate | context_fallback
    primary_eligible: bool = False
    primary_eligibility_reasons: List[str] = field(default_factory=list)

    # ===== 已显式化（原 setattr 动态塞入） =====
    anchor_identity_score: float = 0.5
    jd_candidate_alignment: float = 0.5
    neighborhood_consistency: float = 0.5
    local_neighborhood_consistency: float = 0.5
    cross_anchor_support_count: int = 1
    hierarchy_evidence: Dict[str, Any] = field(default_factory=dict)
    primary_score: float = 0.0
    retain_mode: str = "normal"
    suppress_seed: bool = False
    retain_reason: Optional[str] = None
    topic_source: str = "missing"
    identity_score: float = 0.0
    identity_gate: float = 1.0
    domain_fit: float = 1.0
    domain_reason: str = ""                  # domain_conflict_strong 等，供 check_primary_eligibility

    # ===== 双路证据 + context_gain（merge 后统一算） =====
    surface_sim: Optional[float] = None      # 静态 similar_to 相似度
    conditioned_sim: Optional[float] = None # 动态 conditioned_vec 相似度
    context_gain: Optional[float] = None     # conditioned_sim - surface_proxy，上下文增益
    source_set: Optional[Set[str]] = None      # merge 后 similar_to | conditioned_vec | family_landing

    # ===== Stage2A 定性字段（collect_landing_candidates 补齐，供 admission/mainline 用） =====
    context_continuity: float = 0.0          # 连续分，非布尔
    context_local_support: float = 0.0
    context_co_anchor_support: float = 0.0
    context_jd_support: float = 0.0
    hierarchy_consistency: float = 0.0       # 与 JD 层级一致性（辅助约束）
    field_fit: float = 0.0
    subfield_fit: float = 0.0
    topic_fit: float = 0.0
    hierarchy_note: str = ""
    polysemy_risk: float = 0.0
    polysemy_note: str = ""
    object_like_risk: float = 0.0
    object_like_note: str = ""
    generic_risk: float = 0.0
    generic_note: str = ""
    source_type: str = "similar_to"
    source_rank: int = 0
    source_score: float = 0.0
    # ===== Stage2A 终稿：候选分型 + 禁扩散 =====
    family_type: str = ""                    # exact_like | near_synonym | generic | shifted
    scene_shifted: bool = False
    generic_like: bool = False
    expand_block_reason: Optional[str] = None  # None | generic | scene_shift | low_identity | weak_context
    source_trust: float = 1.0
    ctx_supported: bool = False             # 与 context_supported 同步，供分型用


@dataclass
class Stage2ACandidate:
    """Stage2A 统一候选对象：只做证据与组内相对排序，不做固定阈值淘汰。"""
    tid: int
    term: str
    source: str  # similar_to | conditioned_vec | alias_or_exact ...

    # 原始证据（仅用于组内相对比较）
    semantic_score: float = 0.0
    context_sim: float = 0.0
    surface_sim: Optional[float] = None
    conditioned_sim: Optional[float] = None
    context_gain: float = 0.0
    source_set: Optional[Set[str]] = None  # 合并后的来源集合 similar_to | conditioned_vec | family_landing
    jd_align: float = 0.0
    mainline_sim: Optional[float] = None
    cross_anchor_support: float = 0.0
    cross_anchor_support_count: int = 0
    cross_anchor_support_weight: float = 0.0
    cross_anchor_anchor_list: List[int] = field(default_factory=list)
    family_match: float = 0.0
    source_type: str = "similar_to"
    source_rank: int = 0
    source_score: float = 0.0
    hierarchy_consistency: float = 0.0
    field_fit: float = 0.0
    subfield_fit: float = 0.0
    topic_fit: float = 0.0
    hierarchy_note: str = ""
    polysemy_risk: float = 0.0
    isolation_risk: float = 0.0
    object_like_risk: float = 0.0
    generic_risk: float = 0.0
    context_continuity: float = 0.0

    # 主线偏好（标量，仅用于 2A 内部排序；原 mainline_preference 字典保留兼容）
    mainline_preference: Optional[Any] = None  # float 或 Dict

    # 组内相对排序用（percentile 0~1，由 assign_relative_scores_within_anchor 填充）
    relative_scores: Dict[str, float] = field(default_factory=dict)
    composite_rank_score: float = 0.0

    # 主线优先排序后的组内名次与快照（由 enrich 填充）
    anchor_internal_rank: Optional[int] = None
    mainline_rank: Optional[float] = None
    sort_key_snapshot: Optional[Tuple[Any, ...]] = None

    # 准入与角色（由 enrich 设 retain_mode/suppress_seed/role_in_anchor_candidate_pool，由 select 设 role/can_expand）
    retain_mode: str = "normal"       # normal | weak_retain | reject
    suppress_seed: bool = False
    admission_reasons: List[str] = field(default_factory=list)
    primary_eligibility_reasons: List[str] = field(default_factory=list)  # 与 admission_reasons 一致，checklist 字段名
    role_in_anchor_candidate_pool: str = ""  # mainline_candidate | secondary_candidate | reject_candidate

    # 组内落点标签（由 select_primary_per_anchor 设置；Stage2 内部使用，非跨阶段正式语义）
    survive_primary: bool = False
    can_expand: bool = False  # primary_expandable 与 primary_support_seed（弱 seed）为 True
    can_expand_from_2a: bool = False  # 强/弱 seed 均为 True；仅支线/高风险保留为 False
    fallback_primary: bool = False  # anchor_core_fallback：保线不交 2B 扩
    reject_reason: Optional[str] = None
    role: Optional[str] = None  # mainline | side | off | unknown
    role_in_anchor: Optional[str] = None  # mainline | side | dropped（与 role 同步）
    # 五层：primary_expandable | primary_support_seed | primary_support_keep | risky_keep | reject
    # 兼容旧日志：primary_keep_no_expand | primary_fallback_keep_no_expand
    primary_bucket: str = ""
    anchor_identity_score: float = 0.0  # 与 family_match 同步，供 admission/mainline 用
    # Stage2A 终稿：分型 + 主线/保留分
    family_type: str = ""
    scene_shifted: bool = False
    generic_like: bool = False
    expand_block_reason: Optional[str] = None
    source_trust: float = 1.0
    ctx_supported: bool = False
    mainline_pref_score: float = 0.0
    retain_score: float = 0.0
    mainline_admissible: bool = False
    admission_reason: str = ""
    # Stage2B seed tier：与 primary_bucket 配套的扩展资格标签（Stage2 内部用途）
    # strong=primary_expandable，weak=primary_support_seed，none=不可作 seed
    stage2b_seed_tier: str = "none"

    def is_stage2b_seed(self) -> bool:
        return self.stage2b_seed_tier in ("strong", "weak")


@dataclass
class PrimaryLanding:
    """Stage2A 选出的主落点。"""
    vid: int
    term: str
    identity_score: float
    source: str
    anchor_vid: int
    anchor_term: str
    domain_fit: float = 1.0


@dataclass
class ExpandedTermCandidate:
    """Stage2 输出 / Stage3 输入，含 term_role 与三层领域 topic_align。"""
    vid: int
    term: str
    term_role: str  # primary | dense_expansion | cluster_expansion | cooc_expansion
    identity_score: float
    source: str
    anchor_vid: int
    anchor_term: str
    semantic_score: float = 0.0
    degree_w: int = 0
    domain_span: int = 0
    target_degree_w: int = 0
    degree_w_expanded: int = 0
    cov_j: float = 0.0
    src_vids: List[int] = field(default_factory=list)
    hit_count: int = 0
    topic_align: float = 1.0
    topic_level: str = "missing"
    topic_confidence: float = 1.0
    domain_fit: float = 1.0
    parent_primary: str = ""  # 产生该词的 primary term（primary 自身为 term）
    # Stage2A 排序与调试（透传至 Stage3 _debug）
    mainline_preference: Optional[Dict[str, Any]] = None
    mainline_rank: Optional[float] = None
    anchor_internal_rank: Optional[int] = None
    survive_primary: Optional[bool] = None
    can_expand: Optional[bool] = None
    sort_key_snapshot: Optional[Tuple[Any, ...]] = None
    role_in_anchor: Optional[str] = None  # mainline | side | dropped
    cross_anchor_support: Optional[float] = None


def expand_semantic_map(
    label,
    core_vids: List[int],
    anchor_skills: Dict[str, Any],
    domain_regex: Optional[str] = None,
    query_vector=None,
    query_text: Optional[str] = None,
    return_raw: bool = False,
):
    """
    从 LabelRecallPath._expand_semantic_map 迁移而来。
    label 需提供：_query_expansion_with_topology/_query_expansion_by_context_vector/_expand_with_clusters/
              _calculate_academic_resonance/_calculate_anchor_resonance/_get_cooccurrence_domain_metrics/
              _calculate_final_weights，以及 debug_info。
    """
    regex = domain_regex if domain_regex else ".*"

    raw_edge = query_expansion_with_topology(label, core_vids, regex)
    raw_ctx = (
        query_expansion_by_context_vector(label, anchor_skills, query_text, regex, topk_per_anchor=3)
        if query_text
        else []
    )

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
        base = label.EDGE_WEIGHT * sim_edge + label.CTX_EDGE_WEIGHT * sim_ctx
        degree_w = int((rec_e or rec_c).get("degree_w", 0) or 0)
        if degree_w >= label.HIT_BONUS_DEGREE_GATE:
            bonus = 1.0
        else:
            hit_eff = min(hit, label.HIT_BONUS_HIT_CAP) if hit >= 1 else 1
            bonus = min(label.HIT_BONUS_CAP, 1.0 + label.HIT_BONUS_BETA * math.log(hit_eff))
        sim_merged = base * bonus
        rec = dict(rec_e or rec_c)
        rec["sim_score"] = sim_merged
        rec["src_vids"] = sorted(src_vids)
        rec["hit_count"] = hit
        # 来源标记：供 Stage3 做来源可信度加权（无硬编码词表）
        if rec_e and rec_c:
            rec["source"] = "edge_and_ctx"
        elif rec_e:
            rec["source"] = "edge_only"
        else:
            rec["source"] = "ctx_only"
        raw_merged.append(rec)

    # Stage2 三路调试：raw_edge / raw_ctx / raw_merged 各 top20，统一含 tid, term, sim_score, source/origin, degree_w, domain_span
    _top = lambda lst, key="sim_score": sorted(lst, key=lambda r: float(r.get(key, 0.0) or 0.0), reverse=True)[:20]
    label.debug_info.stage2_raw_edge_top20 = [
        {
            "tid": r.get("tid"),
            "term": (r.get("term") or "")[:40],
            "sim_score": round(float(r.get("sim_score", 0) or 0), 4),
            "origin": "edge",
            "degree_w": int(r.get("degree_w", 0) or 0),
            "domain_span": int(r.get("domain_span", 0) or 0),
        }
        for r in _top(raw_edge)
    ]
    label.debug_info.stage2_raw_ctx_top20 = [
        {
            "tid": r.get("tid"),
            "term": (r.get("term") or "")[:40],
            "sim_score": round(float(r.get("sim_score", 0) or 0), 4),
            "origin": "ctx",
            "degree_w": int(r.get("degree_w", 0) or 0),
            "domain_span": int(r.get("domain_span", 0) or 0),
        }
        for r in _top(raw_ctx)
    ]
    label.debug_info.stage2_raw_merged_top20 = [
        {
            "tid": r.get("tid"),
            "term": (r.get("term") or "")[:40],
            "sim_score": round(float(r.get("sim_score", 0) or 0), 4),
            "source": r.get("source", ""),
            "degree_w": int(r.get("degree_w", 0) or 0),
            "domain_span": int(r.get("domain_span", 0) or 0),
        }
        for r in _top(raw_merged)
    ]

    raw_results = raw_merged
    if not raw_results:
        return [] if return_raw else ({}, {}, {})

    raw_results = expand_with_clusters(label, raw_results, regex, topk_per_seed=7, weight_decay=0.2)

    tids = [r["tid"] for r in raw_results]
    resonance_map = calculate_academic_resonance(label, tids)
    for rec in raw_results:
        rec["resonance"] = resonance_map.get(rec["tid"], 0.0)

    first_layer_core = [r["tid"] for r in raw_results if r.get("hit_count", 0) >= 2]
    if not first_layer_core:
        first_layer_core = tids
    anchor_resonance_map = calculate_anchor_resonance(label, tids, first_layer_core)
    for rec in raw_results:
        rec["anchor_resonance"] = anchor_resonance_map.get(rec["tid"], 0.0)

    active_domain_ids = set(re.findall(r"\d+", regex)) if regex and regex != ".*" else set()
    cooc_metrics = get_cooccurrence_domain_metrics(label, raw_results, active_domain_ids)
    for rec in raw_results:
        tid_key = str(rec["tid"])
        rec["cooc_span"] = cooc_metrics.get(tid_key, {}).get("cooc_span", 0.0)
        rec["cooc_purity"] = cooc_metrics.get(tid_key, {}).get("cooc_purity", 0.0)

    label.debug_info.expansion_raw_results = raw_results
    if return_raw:
        return raw_results
    return label._calculate_final_weights(raw_results, query_vector)


def expand_with_clusters(label, raw_results, domain_regex, topk_per_seed=5, weight_decay=0.2):
    # 直接复用原实现：依赖 label.cluster_members/label.voc_to_clusters/label.vocab_to_idx/label.all_vocab_vectors 等
    if not getattr(label, "cluster_members", None) or not getattr(label, "voc_to_clusters", None):
        return raw_results

    active_domain_ids = set(re.findall(r"\d+", domain_regex)) if domain_regex and domain_regex != ".*" else set()

    seed_vids = [int(rec["tid"]) for rec in raw_results]
    seed_vids_set = set(seed_vids)

    seed_to_cluster = {}
    for vid in seed_vids:
        clusters = label.voc_to_clusters.get(int(vid))
        if not clusters:
            continue
        cid, cscore = max(clusters, key=lambda x: x[1])
        seed_to_cluster[int(vid)] = (cid, cscore)

    if not seed_to_cluster:
        return raw_results

    seed_sim_map = {}
    for rec in raw_results:
        try:
            vid = int(rec["tid"])
        except Exception:
            continue
        seed_sim_map[vid] = float(rec.get("sim_score", 1.0))

    CLUSTER_EXPAND_TOP_SEEDS = 15
    sorted_by_sim = sorted(raw_results, key=lambda r: float(r.get("sim_score", 0.0)), reverse=True)
    allowed_seed_vids = {int(r["tid"]) for r in sorted_by_sim[:CLUSTER_EXPAND_TOP_SEEDS]}

    expansion_log = []
    cluster_expanded = {}

    for rec in raw_results:
        try:
            vid = int(rec["tid"])
        except Exception:
            continue
        if vid not in seed_to_cluster or vid not in allowed_seed_vids:
            continue

        cid, _ = seed_to_cluster[vid]
        members = label.cluster_members.get(int(cid)) or []
        if not members:
            continue

        candidates = [m for m in members if m not in seed_vids_set]
        if not candidates:
            continue

        seed_idx = label.vocab_to_idx.get(str(vid))
        if seed_idx is None:
            continue
        seed_vec = label.all_vocab_vectors[seed_idx]

        sims = []
        for m in candidates:
            midx = label.vocab_to_idx.get(str(m))
            if midx is None:
                continue
            mvec = label.all_vocab_vectors[midx]
            sim_in_cluster = float(np.dot(seed_vec, mvec))
            sims.append((m, sim_in_cluster))

        CLUSTER_MIN_SIM = 0.6
        sims = [(m, s) for m, s in sims if s >= CLUSTER_MIN_SIM]
        if not sims:
            continue

        sims.sort(key=lambda x: x[1], reverse=True)
        top = sims[:topk_per_seed]
        seed_sim = seed_sim_map.get(vid, 1.0)

        seed_term = rec.get("term") or (
            label._vocab_meta.get(vid, ("", ""))[0] if getattr(label, "_vocab_meta", None) else ""
        )
        for m, sim_in_cluster in top:
            contrib = weight_decay * seed_sim * sim_in_cluster
            if contrib <= 0:
                continue
            entry = cluster_expanded.setdefault(int(m), {"sim_score": 0.0, "support": 0, "seed_vids": set()})
            entry["sim_score"] = max(entry["sim_score"], contrib)
            entry["support"] += 1
            entry["seed_vids"].add(int(vid))
            expansion_log.append(
                {
                    "term_tid": int(m),
                    "seed_vid": vid,
                    "seed_term": seed_term or str(vid),
                    "sim_in_cluster": round(sim_in_cluster, 4),
                    "seed_sim": round(seed_sim, 4),
                    "contrib": round(contrib, 6),
                }
            )

    if not cluster_expanded:
        return raw_results

    # 种子 vid -> source，用于传播到 cluster 扩展出的新词
    source_rank = {"edge_and_ctx": 2, "edge_only": 1, "ctx_only": 0}
    seed_to_source = {}
    for rec in raw_results:
        try:
            vid = int(rec["tid"])
            seed_to_source[vid] = rec.get("source") or "edge_only"
        except Exception:
            continue

    new_vids = [vid for vid in cluster_expanded.keys() if vid not in seed_vids_set]
    if not new_vids:
        return raw_results

    term_map = {}
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        ph = ",".join("?" * len(new_vids))
        rows = conn.execute(
            f"SELECT voc_id, term, entity_type FROM vocabulary WHERE voc_id IN ({ph})", new_vids
        ).fetchall()
        for r in rows:
            if (r["entity_type"] or "").lower() != "concept":
                continue
            term_map[int(r["voc_id"])] = r["term"]

    stats_map = {}
    ph = ",".join("?" * len(new_vids))
    rows = label.stats_conn.execute(
        f"SELECT voc_id, work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id IN ({ph})",
        new_vids,
    ).fetchall()
    for r in rows:
        stats_map[int(r[0])] = (int(r[1]), int(r[2]), r[3])

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
        expanded = expand_domain_dist(label, dist)
        degree_w_expanded = sum(expanded.values())
        if active_domains:
            target_degree_w = sum(expanded.get(str(d), 0) for d in active_domains)
        else:
            target_degree_w = degree_w_expanded

        domain_ratio = target_degree_w / degree_w_expanded if degree_w_expanded else 0.0
        T = float(VOCAB_P95_PAPER_COUNT)
        if degree_w > T:
            x = min(float(degree_w) / T, 4.0)
            size_penalty = (1.0 / x) ** 2
        else:
            size_penalty = 1.0
        eps = 0.05
        r = max(domain_ratio, eps)
        domain_penalty = r ** 2

        # 继承种子中最好的 source，供 Stage3 来源可信度加权
        seed_vids_list = list(agg.get("seed_vids") or [])
        best_source = "cluster"
        for sid in seed_vids_list:
            s = seed_to_source.get(int(sid))
            if s and source_rank.get(s, -1) > source_rank.get(best_source, -1):
                best_source = s
        raw_results.append(
            {
                "tid": vid,
                "term": term_map[vid],
                "sim_score": agg["sim_score"] * size_penalty * domain_penalty,
                "hit_count": agg["support"],
                "seed_vids": sorted(seed_vids_list),
                "degree_w": degree_w,
                "degree_w_expanded": degree_w_expanded,
                "target_degree_w": target_degree_w,
                "domain_span": domain_span,
                "cov_j": 0.0,
                "origin": "cluster",
                "source": best_source,
            }
        )

    label.debug_info.cluster_expansion_log = expansion_log
    return raw_results


def _voc_id_to_term_lower(conn, voc_ids: List[int]) -> Dict[int, str]:
    """从主库 vocabulary 取 voc_id -> term_lower，与 build_vocab_stats 的清洗一致（strip+lower）。"""
    if not voc_ids:
        return {}
    ph = ",".join("?" * len(voc_ids))
    rows = conn.execute(
        f"SELECT voc_id, term FROM vocabulary WHERE voc_id IN ({ph})",
        voc_ids,
    ).fetchall()
    out = {}
    for (vid, term) in rows:
        t = (term or "").strip().lower()
        if t:
            out[int(vid)] = t
    return out


def calculate_academic_resonance(label, tids: List[int]) -> Dict[int, float]:
    """
    从 vocab_stats.db 的 vocabulary_cooccurrence 计算候选词集内部共现权重和（学术共鸣）。
    与 build_vocab_stats_index 的共现数据一致，不再依赖 Neo4j CO_OCCURRED_WITH。
    """
    out = {tid: 0.0 for tid in tids}
    if len(tids) < 2 or not getattr(label, "stats_conn", None):
        return out
    try:
        with sqlite3.connect(DB_PATH) as main_conn:
            tid_to_term = _voc_id_to_term_lower(main_conn, tids)
        if len(tid_to_term) < 2:
            return out
        term_to_tid = {t: vid for vid, t in tid_to_term.items()}
        terms = list(term_to_tid.keys())
        ph = ",".join("?" * len(terms))
        sql = (
            f"SELECT term_a, term_b, freq FROM vocabulary_cooccurrence "
            f"WHERE term_a IN ({ph}) AND term_b IN ({ph})"
        )
        rows = label.stats_conn.execute(sql, terms + terms).fetchall()
        for (ta, tb, freq) in rows:
            f = int(freq or 0)
            if ta in term_to_tid and tb in term_to_tid:
                vid_a, vid_b = term_to_tid[ta], term_to_tid[tb]
                if vid_a in out:
                    out[vid_a] += f
                if vid_b in out:
                    out[vid_b] += f
    except Exception:
        pass
    return out


def calculate_anchor_resonance(label, tids: List[int], first_layer_tids: List[int]) -> Dict[int, float]:
    """
    从 vocab_stats.db 的 vocabulary_cooccurrence 计算候选词与第一层学术词的共现权重和（锚点共鸣）。
    与 build_vocab_stats_index 的共现数据一致，不再依赖 Neo4j CO_OCCURRED_WITH。
    """
    out = {tid: 0.0 for tid in tids}
    if not first_layer_tids or not getattr(label, "stats_conn", None):
        return out
    try:
        with sqlite3.connect(DB_PATH) as main_conn:
            tid_to_term_cand = _voc_id_to_term_lower(main_conn, tids)
            tid_to_term_first = _voc_id_to_term_lower(main_conn, first_layer_tids)
        if not tid_to_term_cand or not tid_to_term_first:
            return out
        terms_cand = set(tid_to_term_cand.values())
        terms_first = set(tid_to_term_first.values())
        term_to_tid_cand = {t: vid for vid, t in tid_to_term_cand.items()}
        ph_c = ",".join("?" * len(terms_cand))
        ph_f = ",".join("?" * len(terms_first))
        sql = (
            f"SELECT term_a, term_b, freq FROM vocabulary_cooccurrence "
            f"WHERE (term_a IN ({ph_c}) AND term_b IN ({ph_f})) OR (term_a IN ({ph_f}) AND term_b IN ({ph_c}))"
        )
        params = list(terms_cand) + list(terms_first) + list(terms_first) + list(terms_cand)
        rows = label.stats_conn.execute(sql, params).fetchall()
        for (ta, tb, freq) in rows:
            f = int(freq or 0)
            if ta in terms_cand and tb in terms_first and ta in term_to_tid_cand:
                out[term_to_tid_cand[ta]] = out.get(term_to_tid_cand[ta], 0) + f
            elif ta in terms_first and tb in terms_cand and tb in term_to_tid_cand:
                out[term_to_tid_cand[tb]] = out.get(term_to_tid_cand[tb], 0) + f
    except Exception:
        pass
    return out


def get_cooccurrence_domain_metrics(label, raw_results, active_domain_ids):
    # 直接复用原实现：依赖 label.stats_conn 与 DB_PATH 主库
    if not raw_results or not active_domain_ids:
        return {str(rec["tid"]): {"cooc_span": 0.0, "cooc_purity": 0.0} for rec in raw_results}

    cooc_purity_from_table = {}
    try:
        tids = [rec["tid"] for rec in raw_results]
        domain_list = [str(d) for d in active_domain_ids]
        if tids and domain_list:
            ph_t = ",".join("?" * len(tids))
            ph_d = ",".join("?" * len(domain_list))
            rows = label.stats_conn.execute(
                f"SELECT voc_id, SUM(ratio) AS cooc_purity FROM vocabulary_cooc_domain_ratio WHERE voc_id IN ({ph_t}) AND domain_id IN ({ph_d}) GROUP BY voc_id",
                tids + domain_list,
            ).fetchall()
            cooc_purity_from_table = {str(r[0]): float(r[1]) for r in rows}
    except Exception:
        pass

    try:
        terms = list({rec["term"] for rec in raw_results})
        terms_set = set(terms)
        placeholders = ",".join("?" * len(terms))
        sql_cooc = (
            f"SELECT term_a, term_b, freq FROM vocabulary_cooccurrence "
            f"WHERE term_a IN ({placeholders}) OR term_b IN ({placeholders})"
        )
        rows = label.stats_conn.execute(sql_cooc, terms + terms).fetchall()

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

        default_out = {
            str(rec["tid"]): {"cooc_span": 0.0, "cooc_purity": cooc_purity_from_table.get(str(rec["tid"]), 0.0)}
            for rec in raw_results
        }
        if not partner_terms:
            return default_out

        partner_list = list(partner_terms)
        ph = ",".join("?" * len(partner_list))
        with sqlite3.connect(DB_PATH) as main_conn:
            main_conn.row_factory = sqlite3.Row
            main_rows = main_conn.execute(
                f"SELECT voc_id, term FROM vocabulary WHERE term IN ({ph})", partner_list
            ).fetchall()
        partner_term_to_vocid = {row["term"]: row["voc_id"] for row in main_rows}

        partner_voc_ids = list(partner_term_to_vocid.values())
        if not partner_voc_ids:
            return default_out

        ph2 = ",".join("?" * len(partner_voc_ids))
        stats_rows = label.stats_conn.execute(
            f"SELECT voc_id, work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id IN ({ph2})",
            partner_voc_ids,
        ).fetchall()
        vocid_to_stats = {r[0]: (r[1], r[2], r[3]) for r in stats_rows}

        out = {}
        for rec in raw_results:
            tid, term = rec["tid"], rec["term"]
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
                expanded = expand_domain_dist(label, dist)
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
                out[str(tid)] = {"cooc_span": 0.0, "cooc_purity": cooc_purity_from_table.get(str(tid), 0.0)}
        return out
    except Exception:
        return {
            str(rec["tid"]): {"cooc_span": 0.0, "cooc_purity": cooc_purity_from_table.get(str(rec["tid"]), 0.0)}
            for rec in raw_results
        }


def expand_domain_dist(label, dist):
    if not dist:
        return {}
    out = {}
    for key, count in dist.items():
        if not key or not count:
            continue
        for d in DomainProcessor.to_set(key):
            out[d] = out.get(d, 0) + count
    return out


def load_vocab_meta(label) -> None:
    if getattr(label, "_vocab_meta", None) is not None:
        return
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT voc_id, term, entity_type FROM vocabulary").fetchall()
            label._vocab_meta = {int(r[0]): (r[1] or "", r[2] or "") for r in rows}
    except Exception:
        label._vocab_meta = {}


def query_expansion_by_context_vector(label, anchor_skills, query_text, regex, topk_per_anchor=5):
    if not query_text or not anchor_skills:
        return []
    load_vocab_meta(label)
    encoder = label._query_encoder
    jd_snippet = (query_text or "").strip()[:500]
    if getattr(label, "verbose", False):
        print(f"[Bridge Debug] query_expansion_by_context_vector 收到 query_text 片段: {jd_snippet[:120]}")
    active_domains = set(re.findall(r"\d+", regex)) if regex and regex != ".*" else set()

    # 同步主召回的文本预处理/桥接逻辑：优先走 QueryEncoder.encode，再做 L2 归一化，
    # 避免 ctx 扩展与主 query 的表示不一致。
    v_jd, _ = encoder.encode(jd_snippet)
    if v_jd is None:
        return []
    v_jd = np.asarray(v_jd, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(v_jd)

    if not hasattr(label, "_term_vec_cache") or label._term_vec_cache is None:
        label._term_vec_cache = {}

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

    # Context-Aware：用 JD 片段作为上下文拼接，使 embedding 向目标领域偏移，减少多义词误匹配
    context_snippet = (query_text or "").strip()[:100].replace("(", " ").replace(")", " ")
    to_encode = []
    to_encode_keys = []
    for tkey, term in zip(terms_lower, terms_raw):
        if tkey not in label._term_vec_cache:
            term_with_ctx = f"{term} ({context_snippet})" if context_snippet else term
            to_encode.append(term_with_ctx)
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
            label._term_vec_cache[k] = vec

    v_terms = np.stack([label._term_vec_cache[tkey] for tkey in terms_lower], axis=0).astype(np.float32)
    if v_terms.ndim == 1:
        v_terms = v_terms.reshape(1, -1)

    lam = float(label.CTX_MIX_LAMBDA)
    embs = lam * v_jd + (1.0 - lam) * v_terms
    faiss.normalize_L2(embs)

    k = min(topk_per_anchor * 3, 30)
    scores, labels_arr = label.vocab_index.search(embs, k)

    by_tid = {}
    for row_i, src_vid in enumerate(ctx_src_vids):
        for score, tid in zip(scores[row_i], labels_arr[row_i]):
            tid = int(tid)
            if tid <= 0 or tid == int(src_vid):
                continue
            meta = label._vocab_meta.get(tid, ("", ""))
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
        row = label.stats_conn.execute(
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
        expanded = expand_domain_dist(label, dist)
        degree_w_expanded = sum(expanded.values())
        target_degree_w = sum(expanded.get(str(d), 0) for d in active_domains)
        domain_ratio = target_degree_w / degree_w_expanded if degree_w_expanded else 0.0

        rec = by_tid[tid]
        ctx_sim = float(rec.get("ctx_sim", 0.0) or 0.0)
        if ctx_sim < float(label.SEMANTIC_MIN):
            continue

        if degree_w <= 40:
            purity_min = 0.4
        else:
            purity_min = float(label.DOMAIN_PURITY_MIN)
        if active_domains and domain_ratio < purity_min:
            continue

        T = float(VOCAB_P95_PAPER_COUNT)
        if degree_w > T:
            x = min(float(degree_w) / T, 4.0)
            size_penalty = (1.0 / x) ** 2
        else:
            size_penalty = 1.0

        src_vids = sorted(rec["src_vids"])
        results.append(
            {
                "tid": tid,
                "term": rec["term"] or label._vocab_meta.get(tid, ("", None))[0],
                "sim_score": ctx_sim * size_penalty,
                "src_vids": src_vids,
                "hit_count": len(src_vids),
                "degree_w": degree_w,
                "degree_w_expanded": degree_w_expanded,
                "target_degree_w": target_degree_w,
                "domain_span": domain_span,
                "cov_j": 0.0,
                "origin": "context_vector",
            }
        )
    return results


def query_expansion_with_topology(label, v_ids, regex):
    if not v_ids:
        return []
    active_domains = set(re.findall(r"\d+", regex))
    params = {"v_ids": list(v_ids), "min_score": SIMILAR_TO_MIN_SCORE, "top_k": SIMILAR_TO_TOP_K}
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
    rows = label.graph.run(cypher, **params).data()
    if not rows:
        label.debug_info.similar_to_raw_rows = []
        label.debug_info.similar_to_agg = []
        label.debug_info.similar_to_pass = []
        return []

    label.debug_info.similar_to_raw_rows = [
        {"src_vid": r.get("src_vid"), "tid": r.get("tid"), "term": r.get("term"), "sim_score": float(r.get("sim_score", 0.0) or 0.0)}
        for r in rows
    ]

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

    by_tid = {}
    for r in rows:
        tid = r["tid"]
        term = r["term"] or ""
        sim = float(r["sim_score"])
        if tid not in by_tid:
            by_tid[tid] = {"tid": tid, "term": term, "sim_score": sim, "src_vids": set(), "hit_count": 0, "origin": "similar_to"}
        by_tid[tid]["sim_score"] = max(by_tid[tid]["sim_score"], sim)
        src_vid = r.get("src_vid")
        if src_vid is not None:
            try:
                by_tid[tid]["src_vids"].add(int(src_vid))
            except Exception:
                pass

    for tid, rec in by_tid.items():
        rec["hit_count"] = len(rec.get("src_vids") or [])
        rec["src_vids"] = sorted(list(rec.get("src_vids") or []))

    tids = list(by_tid.keys())
    pipeline["n_unique_tids"] = len(tids)

    label.debug_info.similar_to_agg = [
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
        row = label.stats_conn.execute(
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
        expanded = expand_domain_dist(label, dist)
        degree_w_expanded = sum(expanded.values())
        target_degree_w = sum(expanded.get(str(d), 0) for d in active_domains)
        domain_ratio = target_degree_w / degree_w_expanded if degree_w_expanded else 0.0
        if degree_w <= 40:
            purity_min = 0.4
        else:
            purity_min = float(label.DOMAIN_PURITY_MIN)
        if domain_ratio < purity_min:
            if degree_w_expanded == 0:
                pipeline["n_fail_degree_w_expanded_zero"] += 1
            pipeline["n_fail_domain_ratio"] += 1
            if len(pipeline["sample_fail_ratio"]) < 5:
                pipeline["sample_fail_ratio"].append(tid)
            continue

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
        rec["sim_score"] = float(rec.get("sim_score", 0.0) or 0.0) * size_penalty
        results.append(rec)

    label.debug_info.expansion_pipeline_stats = pipeline
    label.debug_info.similar_to_pass = [
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


# ---------- Stage2A：学术落点（跨类型 SIMILAR_TO 为主，可选 JD 向量） ----------


# 仅用于 debug 打印标签 [高歧义]，不参与准入或打分
HIGH_AMBIGUITY_ANCHOR_TYPES = frozenset({"acronym", "generic_task_term"})


def _print_stage2b_seed_factors(
    anchor_text: str,
    primary_term: str,
    can_expand: bool,
    source_type_str: str,
    dual_support: bool,
    static_sim: float,
    ctx_sim_val: Optional[float],
    ctx_drop_str: str,
    strong_static_seed: bool,
    final_eligible: bool,
    reason: Optional[str],
) -> None:
    """Stage2B seed 资格分解表：便于验证 2B 改完后是哪类 seed 被救活。"""
    # 与 seed tier audit 同档降噪：默认关，深度诊断时再开 STAGE2_NOISY_DEBUG
    if not (LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG):
        return
    ctx_sim_str = f"{ctx_sim_val:.3f}" if ctx_sim_val is not None else "N/A"
    print(
        f"[Stage2B seed factors] anchor={anchor_text!r} term={primary_term!r}\n"
        f"  can_expand_from_2a={can_expand}\n"
        f"  source_type={source_type_str!r}\n"
        f"  dual_support={dual_support}\n"
        f"  static_sim={static_sim:.3f}\n"
        f"  ctx_sim={ctx_sim_str}\n"
        f"  ctx_drop={ctx_drop_str}\n"
        f"  strong_static_seed={strong_static_seed}\n"
        f"  final_eligible={final_eligible}\n"
        f"  reason={reason!r}"
    )


def _print_stage2b_blocked_weak_seed_audit(
    anchor_text: str,
    seed_detail: List[Tuple[Any, bool, float, Optional[str]]],
) -> None:
    """弱 seed 被 2B 门挡掉时的窄表（与 tier audit 互补：`LABEL_EXPANSION_DEBUG` 即可）。"""
    if not LABEL_EXPANSION_DEBUG:
        return
    rows: List[Tuple[str, str, str, str, str, str]] = []
    for p, eligible, seed_score, block_reason in seed_detail:
        tier = (getattr(p, "stage2b_seed_tier", "") or "").strip().lower()
        if tier != "weak" or eligible:
            continue
        pl_src = _primary_landing_source_type_set(p)
        cond_only = (
            "conditioned_vec" in pl_src
            and "similar_to" not in pl_src
            and "family_landing" not in pl_src
        )
        identity = float(getattr(p, "identity_score", 0.0) or 0.0)
        anchor_identity = float(getattr(p, "anchor_identity_score", identity) or identity)
        family_match_seed = float(getattr(p, "family_match", 0.0) or 0.0)
        if family_match_seed <= 0.0:
            family_match_seed = anchor_identity
        jd_align = float(getattr(p, "jd_align", 0.5) or 0.5)
        mlp = float(getattr(p, "mainline_pref_score", 0.0) or 0.0)
        axis_cs = (
            0.35 * anchor_identity
            + 0.30 * family_match_seed
            + 0.20 * jd_align
            + 0.15 * max(0.0, mlp)
        )
        gen = float(getattr(p, "generic_risk", 0.0) or 0.0)
        poly = float(getattr(p, "polysemy_risk", 0.0) or 0.0)
        obj = float(getattr(p, "object_like_risk", 0.0) or 0.0)
        rows.append(
            (
                (getattr(p, "term", "") or "")[:44],
                f"{axis_cs:.3f}",
                f"{seed_score:.3f}",
                str(cond_only),
                f"g={gen:.2f} p={poly:.2f} o={obj:.2f}",
                (block_reason or "")[:56],
            )
        )
    if not rows:
        return
    print(f"[Stage2B blocked weak seed audit] anchor={anchor_text!r} n_blocked_weak={len(rows)}")
    for t, ax, ss, co, rsk, br in rows:
        print(
            f"  term={t!r} axis_consistency_seed={ax} seed_score={ss} "
            f"conditioned_only={co} gen_poly_obj={rsk!r} blocked_reason={br!r}"
        )


def check_seed_eligibility(
    label,
    p: "PrimaryLanding",
    jd_profile: Optional[Dict[str, Any]] = None,
    *,
    emit_seed_factors: bool = True,
) -> Tuple[bool, float, Optional[str]]:
    """
    Stage2B 的 seed 资格判定（职责收口：support expansion，不是“准终判层”）。

    本函数是 Stage2B 的唯一 seed 决策入口，用于决定哪些 Stage2A 产生的“强局部落点（local landing）”
    可以作为 Stage2B 的扩展起点（seed），从而补齐 support candidates（dense/cluster/cooc 等）。

    口径说明：
    - seed 判定只服务于 Stage2B 的“支持项扩展资格”与扩展排序裁剪；
      不应被理解为最终保留/淘汰或跨锚全局裁决（这些属于 Stage3）。

    判定流程（不改算法，仅澄清）：
    - 先挡 fallback / tier=none / semantic_mismatch；再挡 **`blocked_by_2a`**（can_expand_from_2a=False）
    - **strong / weak** 按 **seed_score** 与 **axis_consistency_seed**（identity+family+jd+mainline_pref）分流；
      weak 另拦 **conditioned_only** 与 **generic/poly/object** 上限
    - seed_score 仍参与扩展侧的排序裁剪（SEED_SCORE_MIN / SEED_SCORE_MIN_WEAK）
    - emit_seed_factors=False：dense 内复核时不重复打印 [Stage2B seed factors]
    """
    def _emit_sf(*args: Any, **kwargs: Any) -> None:
        if emit_seed_factors:
            _print_stage2b_seed_factors(*args, **kwargs)

    identity = float(getattr(p, "identity_score", 0.0) or 0.0)
    anchor_identity = float(getattr(p, "anchor_identity_score", identity) or identity)
    primary_score = float(getattr(p, "primary_score", identity) or identity)
    jd_align = float(getattr(p, "jd_align", 0.5) or 0.5)
    context_gain = float(getattr(p, "context_gain", 0.0) or 0.0)

    src = (getattr(p, "source", "") or getattr(p, "source_type", "") or "").strip().lower()
    can_expand = bool(getattr(p, "can_expand", getattr(p, "expandable", False)))
    _cec_raw = getattr(p, "can_expand_from_2a", None)
    if _cec_raw is None:
        can_expand_from_2a = can_expand
    else:
        can_expand_from_2a = bool(_cec_raw)

    anchor_text = getattr(p, "anchor_term", "") or ""
    primary_term = getattr(p, "term", "") or ""
    dual_support = bool(getattr(p, "dual_support", False))
    seed_tier = (getattr(p, "stage2b_seed_tier", None) or "").strip().lower()

    static_sim = float(getattr(p, "semantic_score", None) or primary_score or 0.0)
    ctx_sim_val = getattr(p, "conditioned_sim", None)
    if ctx_sim_val is not None:
        ctx_sim_val = float(ctx_sim_val)
    ctx_drop = max(0.0, static_sim - float(ctx_sim_val or 0.0)) if ctx_sim_val is not None else 0.0
    ctx_drop_str = f"{static_sim - ctx_sim_val:.3f}" if (ctx_sim_val is not None and static_sim is not None) else "N/A"
    source_type_str = getattr(p, "source_type", None) or src or ""

    # --------------------------------------------------
    # 0. fallback / 非 seed（tier=none 在 2A 已收口，此处不再用「primary 数」混称 seed）
    # --------------------------------------------------
    if getattr(p, "fallback_primary", False):
        block_reason = "fallback_primary_no_expand"
        _emit_sf(
            anchor_text, primary_term, can_expand_from_2a, source_type_str, dual_support,
            static_sim, ctx_sim_val, ctx_drop_str, False, False, block_reason,
        )
        setattr(p, "seed_eligible", False)
        setattr(p, "seed_block_reason", block_reason)
        setattr(p, "seed_grounded", False)
        return False, 0.0, block_reason

    if seed_tier == "none":
        block_reason = "stage2a_not_seed"
        _emit_sf(
            anchor_text, primary_term, can_expand_from_2a, source_type_str, dual_support,
            static_sim, ctx_sim_val, ctx_drop_str, False, False, block_reason,
        )
        setattr(p, "seed_eligible", False)
        setattr(p, "seed_block_reason", block_reason)
        setattr(p, "seed_grounded", False)
        return False, 0.0, block_reason

    if is_semantic_mismatch_seed(anchor_text, primary_term):
        block_reason = "semantic_mismatch_seed"
        _emit_sf(
            anchor_text, primary_term, can_expand_from_2a, source_type_str, dual_support,
            static_sim, ctx_sim_val, ctx_drop_str, False, False, block_reason,
        )
        setattr(p, "seed_eligible", False)
        setattr(p, "seed_block_reason", block_reason)
        setattr(p, "seed_grounded", False)
        return False, 0.0, block_reason

    # --------------------------------------------------
    # 0.5～1：先算公共量，再按 tier=strong | weak 分流（弱 seed 不再混入强门）
    # --------------------------------------------------
    generic_risk = float(getattr(p, "generic_risk", 0.0) or 0.0)
    polysemy_risk = float(getattr(p, "polysemy_risk", 0.0) or 0.0)
    object_like_risk = float(getattr(p, "object_like_risk", 0.0) or 0.0)
    cross_anchor_support = int(getattr(p, "cross_anchor_support_count", 1) or 1)
    has_family_evidence = bool(getattr(p, "has_family_evidence", False))
    role_in_anchor = (getattr(p, "role_in_anchor", "") or "").strip().lower()

    trusted_set = {s.strip().lower() for s in (TRUSTED_SOURCE_TYPES_FOR_DIFFUSION or []) if s}
    trusted_source = (source_type_str or "").strip().lower() in trusted_set if trusted_set else False

    ctx_present = ctx_sim_val is not None and float(ctx_sim_val or 0.0) > 0.0
    ctx_sim_val_f = float(ctx_sim_val or 0.0)

    if context_gain >= 0.03:
        ctx_bonus = 0.05
    elif context_gain >= 0.00:
        ctx_bonus = 0.02
    elif context_gain >= -0.05:
        ctx_bonus = -0.01
    elif context_gain >= -0.10:
        ctx_bonus = -0.03
    else:
        ctx_bonus = -0.06

    seed_score = (
        0.45 * primary_score
        + 0.25 * anchor_identity
        + 0.20 * jd_align
        + 0.10 * max(0.0, min(1.0, static_sim))
        + ctx_bonus
    )
    seed_score = max(0.0, min(1.0, seed_score))

    seed_grounded = (
        dual_support
        or (ctx_present and ctx_sim_val_f >= 0.78)
        or (float(static_sim or 0.0) >= 0.88 and trusted_source)
    )
    setattr(p, "seed_grounded", bool(seed_grounded))

    source_type_norm = (source_type_str or "").strip().lower()
    ctx_val = float(ctx_sim_val_f if ctx_present else 0.0)

    def _fail(br: str) -> Tuple[bool, float, str]:
        _emit_sf(
            anchor_text, primary_term, can_expand_from_2a, source_type_str, dual_support,
            static_sim, ctx_sim_val, ctx_drop_str, False, False, br,
        )
        setattr(p, "seed_eligible", False)
        setattr(p, "seed_block_reason", br)
        setattr(p, "seed_grounded", False)
        return False, seed_score, br

    def _ok() -> Tuple[bool, float, None]:
        _emit_sf(
            anchor_text, primary_term, can_expand_from_2a, source_type_str, dual_support,
            static_sim, ctx_sim_val, ctx_drop_str, False, True, None,
        )
        setattr(p, "seed_eligible", True)
        setattr(p, "seed_block_reason", None)
        return True, seed_score, None

    # 2A 已收口 can_expand_from_2a；此处硬挡，避免支线桶误扩。
    if not can_expand_from_2a:
        return _fail("blocked_by_2a")

    pl_src = _primary_landing_source_type_set(p)
    conditioned_only_seed = (
        "conditioned_vec" in pl_src
        and "similar_to" not in pl_src
        and "family_landing" not in pl_src
    )
    family_match_seed = float(getattr(p, "family_match", 0.0) or 0.0)
    if family_match_seed <= 0.0:
        family_match_seed = anchor_identity
    # axis：与 Stage2A 主线一致性同族，专用于 seed 门（不靠词表）
    axis_consistency_seed = (
        0.35 * anchor_identity
        + 0.30 * family_match_seed
        + 0.20 * jd_align
        + 0.15 * max(0.0, float(getattr(p, "mainline_pref_score", 0.0) or 0.0))
    )

    def _weak_seed_audit(ok: bool, br: Optional[str]) -> None:
        if seed_tier != "weak" or not (LABEL_EXPANSION_DEBUG and STAGE2_RULING_DEBUG):
            return
        print(
            f"[Stage2B weak-seed audit] term={primary_term!r} "
            f"tier={seed_tier!r} score={seed_score:.3f} "
            f"axis={axis_consistency_seed:.3f} "
            f"conditioned_only={conditioned_only_seed} "
            f"eligible={ok} reason={br!r}"
        )

    # ---------- strong：轴一致 + seed_score（替换原多路 method_like 结构，减少误放行）----------
    if seed_tier == "strong":
        if seed_score < SEED_SCORE_MIN:
            return _fail("strong_seed_score_low")
        if axis_consistency_seed < SEED_AXIS_CONSISTENCY_STRONG_MIN:
            return _fail("strong_seed_axis_low")
        return _ok()

    # ---------- weak：轴更严 + 风险上限；conditioned_only 改为软惩罚，不交前置一票否决 ----------
    if seed_tier == "weak":
        if conditioned_only_seed:
            seed_score = max(0.0, seed_score - 0.12)
            setattr(p, "seed_evidence_soft", True)
            setattr(p, "seed_soft_penalty", "conditioned_only_soft")
        if seed_score < SEED_SCORE_MIN_WEAK:
            _weak_seed_audit(False, "weak_seed_score_low")
            return _fail("weak_seed_score_low")
        if axis_consistency_seed < 0.50:
            _weak_seed_audit(False, "weak_seed_axis_low")
            return _fail("weak_seed_axis_low")
        if generic_risk >= 0.55 or polysemy_risk >= 0.35 or object_like_risk >= 0.25:
            _weak_seed_audit(False, "weak_seed_risk_high")
            return _fail("weak_seed_risk_high")
        _weak_seed_audit(True, None)
        return _ok()

    _weak_seed_audit(False, "tier_none")
    return _fail("tier_none")


def normalize_anchor_context_tokens(phrases: Optional[List[str]], max_items: int = 8) -> List[str]:
    """去重、截断 mention 邻域短语，保持顺序。"""
    seen: Set[str] = set()
    out: List[str] = []
    for p in phrases or []:
        t = (p or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_items:
            break
    return out


def _is_short_cn_style_anchor(anchor_term: str) -> bool:
    """短、偏任务/领域口语的中文锚点：更依赖 local/co/jd 上下文（Learning to Link：mention 局部窗口）。"""
    s = (anchor_term or "").strip()
    if not s or len(s) > 12:
        return False
    n_cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    return n_cjk >= max(1, len(s) // 2)


# 强条件化：局部窗上限（Learning to Link：mention 邻域，非全文主题）
_STRONG_CONDITIONED_MAX_CHARS = 260
_LIGHT_CONDITIONED_MAX_CHARS = 120
_MENTION_SPAN_EACH = 44


def _jd_snippet_around_mention(jd_snippet: str, anchor_term: str, max_chars: int = 88) -> str:
    """JD 中围绕锚词的短窗；默认明显短于旧版，避免「大段摘要」向量。"""
    s = (jd_snippet or "").strip().replace("\n", " ")
    if not s:
        return ""
    if anchor_term and anchor_term in s:
        i = s.index(anchor_term)
        half = max(24, max_chars // 2)
        lo = max(0, i - half // 2)
        hi = min(len(s), i + len(anchor_term) + half)
        frag = s[lo:hi].strip()
        return frag[:max_chars]
    return s[: min(max_chars, 72)]


def _query_text_window(query_text: Optional[str], anchor_term: str, max_chars: int = 96) -> str:
    """query 中围绕锚词的短窗；不再默认返回整段前缀长窗。"""
    if not query_text or not (query_text or "").strip():
        return ""
    q = query_text.strip().replace("\n", " ")
    if anchor_term and anchor_term in q:
        i = q.index(anchor_term)
        lo = max(0, i - _MENTION_SPAN_EACH)
        hi = min(len(q), i + len(anchor_term) + _MENTION_SPAN_EACH)
        return q[lo:hi].strip()[:max_chars]
    return ""


def _best_mention_window(jd_snippet: str, query_text: Optional[str], anchor_term: str, max_total: int = 96) -> str:
    """
    当 local_phrases 为空时，仅从 jd_snippet / query_text 中取锚词邻近短窗。
    找不到锚词出现位置时不拼长段全局描述（避免锚间同质化）。
    """
    at = (anchor_term or "").strip()
    if not at:
        return ""
    for src in ((jd_snippet or "").strip().replace("\n", " "), (query_text or "").strip().replace("\n", " ")):
        if not src or at not in src:
            continue
        i = src.index(at)
        lo = max(0, i - _MENTION_SPAN_EACH)
        hi = min(len(src), i + len(at) + _MENTION_SPAN_EACH)
        return src[lo:hi][:max_total]
    return ""


def _truncate_conditioned_text(text: str, max_chars: int, anchor_term: str) -> str:
    """超长时截断：优先保留锚词行与前几行局部证据。"""
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    lines = [ln for ln in t.split("\n") if ln.strip()]
    if not lines:
        return (anchor_term or "")[:max_chars]
    out_lines: List[str] = []
    n = 0
    for ln in lines:
        if n + len(ln) + 1 > max_chars:
            break
        out_lines.append(ln)
        n += len(ln) + 1
    joined = "\n".join(out_lines)
    if len(joined) <= max_chars:
        return joined
    return joined[:max_chars]


def _extract_latin_alnum_hints(*chunks: str, max_hints: int = 14) -> List[str]:
    """从轻量字符型规则抽取英文/数字术语，无人工词典（feature-rich 局部证据）。"""
    bag: List[str] = []
    seen: Set[str] = set()
    pat = re.compile(r"[A-Za-z][A-Za-z0-9]+(?:[\s\-_/][A-Za-z0-9]+){0,5}")
    for ch in chunks:
        if not ch:
            continue
        for m in pat.finditer(ch):
            w = m.group(0).strip()
            if len(w) < 3:
                continue
            key = w.lower()
            if key in seen:
                continue
            seen.add(key)
            bag.append(w)
            if len(bag) >= max_hints:
                return bag
    return bag


def _conditioned_text_substantial_vs_anchor(anchor_term: str, full_text: str) -> bool:
    """是否比裸锚点多出可编码的局部语境（避免无意义重复 encode）。"""
    a = (anchor_term or "").strip()
    f = (full_text or "").strip()
    if len(f) <= len(a) + 4:
        return False
    if f.startswith(a):
        rest = f[len(a) :].strip()
        return len(rest) > 4
    return len(f) > len(a) + 4


def build_light_conditioned_anchor_text(
    anchor_term: str,
    selected_local: List[str],
    selected_hints: List[str],
    mention_window: str,
) -> str:
    """
    轻量条件化句：仅锚词 + 至多一条局部/mention 窗 + 一条英文 hint，用于 FAISS backoff。
    非 surface 拷贝；与强向量区分，贴近索引邻域。
    """
    at = (anchor_term or "").strip()
    if not at:
        return ""
    parts: List[str] = [at]
    if selected_local:
        parts.append(selected_local[0])
    else:
        mw = (mention_window or "").strip()
        if len(mw) > 6:
            parts.append(mw[:72])
    if selected_hints:
        parts.append(selected_hints[0])
    s = " ".join(p for p in parts if p).strip()
    return s[:_LIGHT_CONDITIONED_MAX_CHARS]


def build_conditioned_anchor_text_variants(
    anchor_term: str,
    local_phrases: Optional[List[str]] = None,
    co_anchor_terms: Optional[List[str]] = None,
    jd_snippet: str = "",
    query_text: Optional[str] = None,
    stage2_process_mode: Optional[str] = None,
    anchor_type: Optional[str] = None,
    anchor_term_to_process_mode: Optional[Dict[str, str]] = None,
    all_anchor_terms: Optional[List[str]] = None,
) -> ConditionedTextBundle:
    """
    强/轻两路 + 实际选用片段（Stage2A conditioning 入口）。

    收口目标（变干净→可用）：
    - 显式利用 Stage2-Entry 的 stage2_process_mode 拉开 main/aux 的 conditioning 强度；
    - hints 仅按锚点类型受控注入（规划 hints 不再全局泛注入）；
    - local phrase 优先“锚点局部专属”，宁可缺省也不塞泛主线高频短语；
    - co-anchor 更克制，避免 surviving aux/英文补锚互喂小团。
    """
    anchor_term = (anchor_term or "").strip()
    if not anchor_term:
        return ConditionedTextBundle(
            strong_text="",
            light_text="",
            selected_local_phrases=[],
            selected_co_anchor_terms=[],
            selected_jd_window="",
            selected_hints=[],
        )
    mode = (stage2_process_mode or "").strip() or "aux_weak_process"
    profile = "main_strong" if mode == "main_strong_process" else "aux_weak"

    # profile：显式拉开强度（不引入新模式体系）
    n_loc = 2 if profile == "main_strong" else 1
    # aux 默认更保守：co-anchor 可为 0（宁缺毋滥）
    n_co = 2 if profile == "main_strong" else 1

    loc_all = normalize_anchor_context_tokens(local_phrases, 8)
    co_all = normalize_anchor_context_tokens(co_anchor_terms, 8)

    # ---------- local phrase：优先“锚点专属”，降低主段高频短语支配 ----------
    anchor_l = anchor_term.strip().lower()
    all_anchor_l = {str(x).strip().lower() for x in (all_anchor_terms or []) if x and str(x).strip()}

    def _local_score(ph: str, jd_window: str) -> float:
        pl = (ph or "").strip().lower()
        if not pl:
            return -1e9
        s = 0.0
        # 字面贴近优先
        if anchor_l and anchor_l in pl:
            s += 3.0
        # 共享锚词惩罚：避免“路径规划”被“运动控制/机器人运动学”这类全局锚词劫持
        if pl in all_anchor_l and pl != anchor_l:
            s -= 2.2
        # 与当前紧窗口的贴合（局部近）
        jw = (jd_window or "").strip().lower()
        if jw and pl in jw:
            s += 1.2
        # 极短泛片段（更易成为主线高频词）：保守下压
        if _cjk_len(pl) > 0 and _cjk_len(pl) <= 4 and anchor_l not in pl:
            s -= 0.6
        return s

    # 先取紧窗口，再用窗口来排序 local phrase
    jd_tight = _jd_snippet_around_mention(jd_snippet, anchor_term, max_chars=88 if profile == "main_strong" else 64)
    q_tight = _query_text_window(query_text, anchor_term, max_chars=96 if profile == "main_strong" else 72)
    mention_win = _best_mention_window(jd_snippet, query_text, anchor_term, max_total=96 if profile == "main_strong" else 72)
    selected_jd_window = (jd_tight or mention_win or q_tight or "").strip()
    if selected_jd_window:
        selected_jd_window = selected_jd_window[: (120 if profile == "main_strong" else 84)]

    loc_ranked = sorted(loc_all, key=lambda ph: _local_score(ph, selected_jd_window), reverse=True)
    loc_pick = []
    for ph in loc_ranked:
        if len(loc_pick) >= n_loc:
            break
        # aux_weak：若只有“主线共享锚词”可选，宁可不选
        pl = (ph or "").strip().lower()
        if profile == "aux_weak" and pl in all_anchor_l and pl != anchor_l:
            continue
        loc_pick.append(ph)

    # ---------- co-anchor：避免弱互喂；优先 main→main ----------
    term_mode = {str(k).strip().lower(): str(v) for k, v in (anchor_term_to_process_mode or {}).items() if k}

    def _co_score(t: str) -> float:
        tl = (t or "").strip().lower()
        if not tl or tl == anchor_l:
            return -1e9
        s = 0.0
        m = term_mode.get(tl, "")
        if profile == "main_strong":
            if m == "main_strong_process":
                s += 2.0
            elif m == "aux_weak_process":
                s += 0.6
            else:
                s += 0.0
        else:
            # aux：默认只吃 main 的少量 support
            if m == "main_strong_process":
                s += 1.5
            else:
                s -= 0.8
        # 英文补锚互喂限流：英文锚默认不堆英文 co-anchor
        is_en_anchor = anchor_l.isascii() and _cjk_len(anchor_l) == 0
        if is_en_anchor and tl.isascii() and _cjk_len(tl) == 0:
            s -= 1.2
        # 紧窗口共现加分（局部近）
        jw = selected_jd_window.lower() if selected_jd_window else ""
        if jw and tl and tl in jw:
            s += 0.8
        return s

    co_ranked = sorted(co_all, key=_co_score, reverse=True)
    co_pick = []
    for t in co_ranked:
        if len(co_pick) >= n_co:
            break
        if profile == "aux_weak":
            # aux 默认更严：仅保留“来自 main”的少量 co-anchor
            if term_mode.get(str(t).strip().lower(), "") != "main_strong_process":
                continue
        co_pick.append(t)
    if profile == "aux_weak" and not loc_pick:
        # aux 若已有 local 专属证据不足，则更保守：co-anchor 也不硬塞
        co_pick = co_pick[:0]

    latin_src = " ".join(
        [
            selected_jd_window,
            " ".join(loc_pick),
            " ".join(co_pick),
            (jd_snippet or "")[:120],
            (query_text or "")[:120],
        ]
    )
    hints = _extract_latin_alnum_hints(latin_src, max_hints=6)

    # ---------- hints：按锚点类型注入（规划 hints 不再全局泛注入） ----------
    # 只对“规划/轨迹/运动规划/最优控制”类锚点开放 planning hints；其它锚点宁可不加。
    def _looks_like_planning_anchor(a: str) -> bool:
        al = (a or "").strip().lower()
        if not al:
            return False
        if "path" in al or "trajectory" in al or "motion planning" in al:
            return True
        if "规划" in al or "路径" in al or "轨迹" in al or "运动规划" in al or "轨迹规划" in al:
            return True
        # anchor_type=task_chain / unknown 时不额外强推；只按表面词形
        return False

    planning_ok = _looks_like_planning_anchor(anchor_term)
    planning_tokens = ("rrt", "prm", "chomp", "mpc", "ilqr")

    filtered_hints: List[str] = []
    for h in hints:
        hl = (h or "").strip().lower()
        if not hl:
            continue
        is_planning_hint = any(tok in hl for tok in planning_tokens)
        if is_planning_hint and not planning_ok:
            continue
        # aux_weak：默认更严，除非是 planning anchor 才给 hints
        if profile == "aux_weak" and not planning_ok:
            continue
        filtered_hints.append(h)

    selected_hints = filtered_hints[: (2 if profile == "main_strong" else (1 if planning_ok else 0))]

    lines: List[str] = [anchor_term]
    if loc_pick:
        lines.append(" ".join(loc_pick))
    if co_pick:
        lines.append(" ".join(co_pick))
    if selected_jd_window:
        lines.append(selected_jd_window[: (120 if profile == "main_strong" else 84)])
    if selected_hints:
        lines.append(" ".join(selected_hints))

    strong = _truncate_conditioned_text("\n".join(x for x in lines if x), _STRONG_CONDITIONED_MAX_CHARS, anchor_term)
    light = build_light_conditioned_anchor_text(anchor_term, loc_pick, selected_hints, mention_win or selected_jd_window)
    return ConditionedTextBundle(
        strong_text=strong,
        light_text=light,
        selected_local_phrases=list(loc_pick),
        selected_co_anchor_terms=list(co_pick),
        selected_jd_window=(selected_jd_window or "")[: (160 if profile == "main_strong" else 96)],
        selected_hints=list(selected_hints),
    )


def build_conditioned_anchor_text(
    anchor_term: str,
    local_phrases: Optional[List[str]] = None,
    co_anchor_terms: Optional[List[str]] = None,
    jd_snippet: str = "",
    query_text: Optional[str] = None,
) -> str:
    """兼容入口：返回强条件化短文本。"""
    return build_conditioned_anchor_text_variants(
        anchor_term,
        local_phrases=local_phrases,
        co_anchor_terms=co_anchor_terms,
        jd_snippet=jd_snippet,
        query_text=query_text,
    ).strong_text


def _vec_cosine_debug(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        x = np.asarray(a, dtype=np.float64).flatten()
        y = np.asarray(b, dtype=np.float64).flatten()
        if x.size != y.size or x.size == 0:
            return None
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        if nx < 1e-12 or ny < 1e-12:
            return None
        return float(np.dot(x, y) / (nx * ny))
    except Exception:
        return None


def build_conditioned_anchor_vec(
    label: Any,
    anchor_term: str,
    conditioned_text: str,
    surface_vec: Optional[np.ndarray],
    stage1_conditioned_vec: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], str, Optional[float]]:
    """
    生成 conditioned_vec：优先对 conditioned_text 用 QueryEncoder（与词表索引同空间）；
    失败或无语境时依次尝试 Stage1 预计算向量、最后才 surface 回退。
    返回 (vec, conditioning_mode, surface_conditioned_cosine)。
    """
    enc = getattr(label, "_query_encoder", None)
    out_vec: Optional[np.ndarray] = None
    mode = "fallback_surface"
    cos_sc: Optional[float] = None

    if enc is not None and conditioned_text and _conditioned_text_substantial_vs_anchor(anchor_term, conditioned_text):
        try:
            raw, _dur = enc.encode(conditioned_text)
            if raw is not None:
                out_vec = np.asarray(raw, dtype=np.float32).flatten()
                mode = "real_context_encoded"
        except Exception:
            out_vec = None

    if out_vec is None and stage1_conditioned_vec is not None:
        try:
            out_vec = np.asarray(stage1_conditioned_vec, dtype=np.float32).flatten()
            mode = "precomputed_stage1"
        except Exception:
            out_vec = None

    if out_vec is None and surface_vec is not None:
        try:
            out_vec = np.asarray(surface_vec, dtype=np.float32).flatten().copy()
            mode = "fallback_surface"
        except Exception:
            out_vec = None

    cos_sc = _vec_cosine_debug(surface_vec, out_vec)
    return out_vec, mode, cos_sc


def _anchor_skills_to_prepared_anchors(
    label,
    anchor_skills: Dict[str, Any],
    query_text: Optional[str] = None,
) -> List[PreparedAnchor]:
    """将现有 anchor_skills (vid -> {term, anchor_type, conditioned_vec?, _anchor_ctx?, ...}) 转为 List[PreparedAnchor]。
    conditioned_vec：优先由 mention+局部上下文文本经 QueryEncoder 得到（真正条件化）；仅缺语境或编码失败时用 Stage1 向量或 surface 回退并打标。
    """
    load_vocab_meta(label)
    out = []
    # Stage2-Entry 模式映射：用于 conditioning 收口（不新增 Stage1 字段）
    all_terms: List[str] = []
    term_to_mode: Dict[str, str] = {}
    for _vid0, _info0 in (anchor_skills or {}).items():
        t0 = str((_info0 or {}).get("term") or "").strip()
        if t0:
            all_terms.append(t0)
            tl0 = t0.lower()
            if tl0 and tl0 not in term_to_mode:
                term_to_mode[tl0] = str((_info0 or {}).get("stage2_process_mode") or "")
    for vid_str, info in (anchor_skills or {}).items():
        try:
            vid = int(vid_str)
        except (TypeError, ValueError):
            continue
        term = (info.get("term") or "").strip() or (label._vocab_meta.get(vid, ("", ""))[0])
        if not term:
            continue
        anchor_type = (info.get("anchor_type") or "unknown").strip().lower()
        stage1_conditioned = info.get("conditioned_vec")
        if stage1_conditioned is not None and hasattr(stage1_conditioned, "__len__"):
            stage1_conditioned = np.asarray(stage1_conditioned, dtype=np.float32).flatten()
        else:
            stage1_conditioned = None
        source_type = (info.get("anchor_source") or "skill_direct").strip()
        source_weight = float(info.get("anchor_source_weight", 1.0))
        ctx = info.get("_anchor_ctx") or {}
        local_phrases = list(ctx.get("local_phrases") or info.get("local_phrases") or [])
        co_anchor_terms = list(ctx.get("co_anchor_terms") or info.get("co_anchor_terms") or [])
        jd_snippet = str(info.get("jd_snippet") or ctx.get("jd_snippet") or "")
        stage2_process_mode = str(info.get("stage2_process_mode") or "")
        surface_vec = _get_candidate_vec_for_mainline(label, vid)
        bundle = build_conditioned_anchor_text_variants(
            term,
            local_phrases=local_phrases,
            co_anchor_terms=co_anchor_terms,
            jd_snippet=jd_snippet,
            query_text=query_text,
            stage2_process_mode=stage2_process_mode,
            anchor_type=anchor_type,
            anchor_term_to_process_mode=term_to_mode,
            all_anchor_terms=all_terms,
        )
        conditioned_text = bundle.strong_text
        conditioned_vec, conditioning_mode, cos_sc = build_conditioned_anchor_vec(
            label,
            term,
            conditioned_text,
            surface_vec,
            stage1_conditioned,
        )
        pa = PreparedAnchor(
            anchor=term,
            vid=vid,
            anchor_type=anchor_type,
            expanded_forms=[term],
            conditioned_vec=conditioned_vec,
            source_type=source_type,
            source_weight=source_weight,
            local_phrases=local_phrases,
            co_anchor_terms=co_anchor_terms,
            jd_snippet=jd_snippet,
            surface_vec=surface_vec,
            conditioned_text=conditioned_text,
            conditioning_mode=conditioning_mode,
            surface_text=term,
            surface_conditioned_cosine=cos_sc,
            light_conditioned_text=bundle.light_text,
        )
        setattr(pa, "stage2_process_mode", stage2_process_mode or "")
        setattr(pa, "conditioning_profile", "main_strong" if stage2_process_mode == "main_strong_process" else "aux_weak")
        setattr(pa, "conditioned_is_fallback", conditioning_mode == "fallback_surface")
        setattr(pa, "conditioned_text_selected_local", bundle.selected_local_phrases)
        setattr(pa, "conditioned_text_selected_co", bundle.selected_co_anchor_terms)
        setattr(pa, "conditioned_text_selected_jd_window", bundle.selected_jd_window)
        setattr(pa, "conditioned_text_selected_hints", bundle.selected_hints)
        if info.get("final_anchor_score") is not None:
            try:
                setattr(pa, "final_anchor_score", float(info.get("final_anchor_score")))
            except (TypeError, ValueError):
                setattr(pa, "final_anchor_score", 0.0)
        if LABEL_EXPANSION_DEBUG:
            print(
                f"[Stage2A conditioning] anchor={term!r} stage2_process_mode={stage2_process_mode!r} "
                f"conditioning_profile={getattr(pa, 'conditioning_profile', '')!r} mode={conditioning_mode} "
                f"conditioned_is_fallback={getattr(pa, 'conditioned_is_fallback', False)}\n"
                f"  surface_text={term!r}\n"
                f"  conditioned_text={conditioned_text[:260]!r}{'...' if len(conditioned_text) > 260 else ''}\n"
                f"  light_conditioned_text={bundle.light_text!r}\n"
                f"  selected_local_phrases={bundle.selected_local_phrases!r}\n"
                f"  selected_co_anchor_terms={bundle.selected_co_anchor_terms!r}\n"
                f"  selected_jd_window={bundle.selected_jd_window!r}\n"
                f"  selected_hints={bundle.selected_hints!r}\n"
                f"  surface_conditioned_cosine={cos_sc}"
            )
        out.append(pa)
    return out


def _top_terms_by_vector(label, vec, k: int = 5) -> List[str]:
    """按余弦相似度返回与 vec 最接近的 k 个词（term 字符串），用于 Neighbor Compare。"""
    if vec is None or getattr(label, "vocab_to_idx", None) is None or getattr(label, "all_vocab_vectors", None) is None:
        return []
    try:
        v = np.asarray(vec, dtype=np.float32).flatten()
        mat = np.asarray(label.all_vocab_vectors, dtype=np.float32)
        if mat.ndim == 1:
            return []
        dots = np.dot(mat, v)
        norms = np.linalg.norm(mat, axis=1)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-9:
            return []
        sims = dots / (norms * norm_v + 1e-9)
        top_idx = np.argsort(sims)[::-1][:k]
        load_vocab_meta(label)
        idx_to_vid = {idx: vid_str for vid_str, idx in label.vocab_to_idx.items()}
        out = []
        for idx in top_idx:
            vid_str = idx_to_vid.get(int(idx))
            if vid_str is None:
                continue
            try:
                vid = int(vid_str)
            except (TypeError, ValueError):
                continue
            term = (label._vocab_meta.get(vid, ("", ""))[0] or "").strip() or vid_str
            out.append(term)
        return out
    except Exception:
        return []


def _compute_jd_candidate_alignment(label, vid: int, query_vector) -> float:
    """数据驱动：候选词与 JD 整体向量的余弦相似度，用于 primary 打分，无词表。"""
    if query_vector is None or getattr(label, "vocab_to_idx", None) is None or getattr(label, "all_vocab_vectors", None) is None:
        return 0.5
    idx = label.vocab_to_idx.get(str(vid))
    if idx is None:
        return 0.5
    try:
        term_vec = np.asarray(label.all_vocab_vectors[idx], dtype=np.float32).flatten()
        q = np.asarray(query_vector, dtype=np.float32).flatten()
        if term_vec.size != q.size or term_vec.size == 0:
            return 0.5
        cos_sim = float(np.dot(term_vec, q))
        cos_sim = max(-1.0, min(1.0, cos_sim))
        return 0.5 + 0.5 * max(0.0, cos_sim)
    except Exception:
        return 0.5


def retrieve_academic_term_by_similar_to(
    label,
    anchor: PreparedAnchor,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
    top_k: Optional[int] = None,
) -> List[LandingCandidate]:
    """Stage2A 落点：从锚点（industry）查跨类型 SIMILAR_TO → 学术词；仅保留与激活领域（及可选三级领域）一致的词。"""
    load_vocab_meta(label)
    if not getattr(label, "graph", None):
        if LABEL_EXPANSION_DEBUG and STAGE2_NOISY_DEBUG:
            print(f"[Stage2A] SIMILAR_TO 跳过 anchor={anchor.anchor!r} vid={anchor.vid}（无 graph）")
        return []
    use_top_k = int(top_k) if top_k is not None else SIMILAR_TO_TOP_K
    params = {
        "anchor_vid": anchor.vid,
        "min_score": SIMILAR_TO_MIN_SCORE,
        "top_k": use_top_k,
    }
    cypher = """
    MATCH (v:Vocabulary {id: $anchor_vid})-[r:SIMILAR_TO]->(v_rel:Vocabulary)
    WHERE r.score >= $min_score
      AND coalesce(v_rel.type, 'concept') IN ['concept', 'keyword']
    RETURN v_rel.id AS tid, v_rel.term AS term, r.score AS sim_score
    ORDER BY sim_score DESC
    LIMIT $top_k
    """
    try:
        rows = label.graph.run(cypher, **params).data()
    except Exception as e:
        if LABEL_EXPANSION_DEBUG and STAGE2_NOISY_DEBUG:
            print(f"[Stage2A] SIMILAR_TO 查询异常 anchor_vid={anchor.vid} anchor={anchor.anchor!r}: {e}")
        return []
    # 诊断：将本锚点命中的 similar_to 原始行写入 debug_info.similar_to_raw_rows（跨锚点累积）
    if rows and getattr(label, "debug_info", None) is not None:
        raw_list = getattr(label.debug_info, "similar_to_raw_rows", None)
        if raw_list is None:
            raw_list = []
            label.debug_info.similar_to_raw_rows = raw_list
        for r in rows:
            raw_list.append({
                "src_vid": anchor.vid,
                "tid": r.get("tid"),
                "term": r.get("term"),
                "sim_score": float(r.get("sim_score", 0.0) or 0.0),
            })
    if LABEL_EXPANSION_DEBUG and STAGE2_NOISY_DEBUG:
        print(f"[Stage2A] SIMILAR_TO anchor_vid={anchor.vid} anchor={anchor.anchor!r} min_score={SIMILAR_TO_MIN_SCORE} top_k={use_top_k} -> 命中 {len(rows)} 条")
    out = []
    for r in rows:
        tid = r.get("tid")
        if tid is None:
            continue
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            continue
        term = (r.get("term") or "").strip() or (label._vocab_meta.get(tid, ("", ""))[0])
        sim = max(0.0, min(1.0, float(r.get("sim_score", 0.0) or 0.0)))
        out.append(
            LandingCandidate(
                vid=tid,
                term=term or str(tid),
                source="similar_to",
                semantic_score=sim,
                anchor_vid=anchor.vid,
                anchor_term=anchor.anchor,
                surface_sim=sim,
                conditioned_sim=None,
                context_gain=None,
            )
        )
    if active_domain_set is not None or jd_field_ids or jd_subfield_ids or jd_topic_ids:
        n_before = len(out)
        kept = []
        dropped_with_reason = []
        for c in out:
            ok, reason = _term_in_active_domains_with_reason(
                label, c.vid,
                active_domain_set=active_domain_set,
                jd_field_ids=jd_field_ids,
                jd_subfield_ids=jd_subfield_ids,
                jd_topic_ids=jd_topic_ids,
            )
            if ok:
                kept.append(c)
                if reason and LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG:
                    print(f"[Stage2A] 保留（层级未命中）vid={c.vid} term={c.term!r} sim={c.semantic_score:.3f} reason={reason}")
            elif reason == "domain_conflict_strong":
                dropped_with_reason.append((c, reason))
            else:
                # domain_no_match：上位通用主词（mechanics/simulation/route planning）保留但降权，进后续 admission
                setattr(c, "soft_domain_retain", True)
                setattr(c, "domain_fit", 0.85)
                kept.append(c)
                if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG:
                    print(f"[Stage2A] 软保留（domain_no_match）vid={c.vid} term={c.term!r} sim={c.semantic_score:.3f}")
        out = kept
        if LABEL_EXPANSION_DEBUG and STAGE2_NOISY_DEBUG and n_before != len(out):
            print(f"[Stage2A] 领域过滤后 SIMILAR_TO 落点: {len(out)} 个（过滤前 {n_before}）")
    if LABEL_EXPANSION_DEBUG and STAGE2_NOISY_DEBUG and out:
        sample = [f"{c.term!r}(sim={c.semantic_score:.2f})" for c in out[:3]]
        print(f"[Stage2A] SIMILAR_TO 落点 共 {len(out)} 个 前3: {sample}")
    return out


def _faiss_conditioned_neighbors_once(
    label: Any,
    anchor: PreparedAnchor,
    vec_flat: np.ndarray,
    similar_to_candidates: List[LandingCandidate],
    conditioned_top_k: Optional[int],
    active_domain_set: Optional[Set[int]],
    jd_field_ids: Optional[Set[str]],
    jd_subfield_ids: Optional[Set[str]],
    jd_topic_ids: Optional[Set[str]],
    retrieval_tier: str,
) -> Tuple[List[LandingCandidate], Dict[int, float], int, int]:
    """
    单次 FAISS + 与本轮既有相同的过滤链。
    返回 (neighbors, score_map, raw_prefilter_hits, postfilter_hits)。
    raw_prefilter_hits：通过 sim 门槛与词表类型、尚未做 domain 过滤的条数。
    postfilter_hits：最终入池条数。
    """
    context_neighbors: List[LandingCandidate] = []
    context_score_map: Dict[int, float] = {}
    similar_to_vids = {int(c.vid) for c in similar_to_candidates if getattr(c, "vid", None) is not None}
    vec = np.asarray(vec_flat, dtype=np.float32).flatten()
    if vec.size == 0:
        return [], {}, 0, 0
    vec = vec.reshape(1, -1)
    faiss.normalize_L2(vec)
    use_k = int(conditioned_top_k) if conditioned_top_k is not None else STAGE2A_COLLECT_CONDITIONED_TOP_K
    use_k = max(use_k, 6)
    k = min(use_k, getattr(label.vocab_index, "ntotal", 100))
    if k <= 0:
        return [], {}, 0, 0
    scores, ids = label.vocab_index.search(vec, k)
    conditioned_min_sim = max(SIMILAR_TO_MIN_SCORE, 0.78)
    raw_prefilter = 0
    for score, tid in zip(scores[0], ids[0]):
        try:
            tid = int(tid)
        except (TypeError, ValueError):
            continue
        if tid <= 0 or tid == getattr(anchor, "vid", -1):
            continue
        meta = label._vocab_meta.get(tid, ("", ""))
        term = (meta[0] or "").strip() or str(tid)
        vocab_type = meta[1] or ""
        if vocab_type not in ("concept", "keyword") and vocab_type:
            continue
        sim = max(0.0, min(1.0, float(score)))
        if sim < conditioned_min_sim:
            continue
        raw_prefilter += 1
        ok, _ = _term_in_active_domains_with_reason(
            label, tid,
            active_domain_set=active_domain_set,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
        )
        if not ok:
            continue
        has_similar_to_support = tid in similar_to_vids
        if not has_similar_to_support and sim < 0.82:
            continue
        cand = LandingCandidate(
            vid=tid,
            term=term,
            source="conditioned_vec",
            semantic_score=sim,
            anchor_vid=getattr(anchor, "vid", 0),
            anchor_term=getattr(anchor, "anchor", ""),
            conditioned_sim=sim,
            surface_sim=None,
            context_gain=None,
        )
        cand.context_sim = sim
        cand.context_supported = sim >= 0.80
        cand.context_gap = 0.0
        cand.source_role = "seed_candidate"
        setattr(cand, "conditioned_only", not has_similar_to_support)
        setattr(cand, "has_similar_to_support", has_similar_to_support)
        setattr(cand, "conditioned_retrieval_tier", retrieval_tier)
        context_neighbors.append(cand)
        context_score_map[tid] = sim
    return context_neighbors, context_score_map, raw_prefilter, len(context_neighbors)


def _retrieve_academic_terms_by_conditioned_vec(
    label,
    anchor: PreparedAnchor,
    similar_to_candidates: Optional[List[LandingCandidate]] = None,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
    conditioned_top_k: Optional[int] = None,
) -> Tuple[List[LandingCandidate], Dict[int, Dict[str, float]]]:
    """
    Legacy / 调试：独立 FAISS conditioned 召回 + backoff。主流程已改为 collect 内 **点积局部分数 + 限量 supplement**。
    """
    similar_to_candidates = similar_to_candidates or []
    if getattr(anchor, "conditioned_vec", None) is None:
        return [], {}
    if not getattr(label, "vocab_index", None) or not getattr(label, "_vocab_meta", None):
        return [], {}
    load_vocab_meta(label)

    strong_vec = np.asarray(anchor.conditioned_vec, dtype=np.float32).flatten()
    strong_raw_pre = 0
    strong_post = 0
    light_raw_pre = 0
    light_post = 0
    retrieval_mode = "fallback_none"

    try:
        nbr, smap, sr_pre, sr_post = _faiss_conditioned_neighbors_once(
            label,
            anchor,
            strong_vec,
            similar_to_candidates,
            conditioned_top_k,
            active_domain_set,
            jd_field_ids,
            jd_subfield_ids,
            jd_topic_ids,
            "strong",
        )
        strong_raw_pre, strong_post = sr_pre, sr_post
        context_neighbors = nbr
        context_score_map = smap
    except Exception:
        context_neighbors = []
        context_score_map = {}

    if context_neighbors:
        retrieval_mode = "strong_conditioned"
    else:
        lt = (getattr(anchor, "light_conditioned_text", None) or "").strip()
        enc = getattr(label, "_query_encoder", None)
        at = (getattr(anchor, "anchor", "") or "").strip()
        if (
            enc is not None
            and lt
            and at
            and _conditioned_text_substantial_vs_anchor(at, lt)
        ):
            try:
                raw_lv, _ = enc.encode(lt)
                if raw_lv is not None:
                    lv = np.asarray(raw_lv, dtype=np.float32).flatten()
                    nbr2, smap2, lr_pre, lr_post = _faiss_conditioned_neighbors_once(
                        label,
                        anchor,
                        lv,
                        similar_to_candidates,
                        conditioned_top_k,
                        active_domain_set,
                        jd_field_ids,
                        jd_subfield_ids,
                        jd_topic_ids,
                        "light_backoff",
                    )
                    light_raw_pre, light_post = lr_pre, lr_post
                    if nbr2:
                        context_neighbors = nbr2
                        context_score_map = smap2
                        retrieval_mode = "light_conditioned_backoff"
            except Exception:
                pass

    setattr(anchor, "_cond_strong_raw_hits", strong_raw_pre)
    setattr(anchor, "_cond_strong_postfilter_hits", strong_post)
    setattr(anchor, "_cond_light_raw_hits", light_raw_pre)
    setattr(anchor, "_cond_light_postfilter_hits", light_post)
    setattr(anchor, "conditioned_retrieval_mode", retrieval_mode)

    rerank_signals: Dict[int, Dict[str, float]] = {}
    for cand in similar_to_candidates:
        ctx_sim = context_score_map.get(cand.vid, 0.0)
        rerank_signals[cand.vid] = {
            "context_sim": ctx_sim,
            "context_supported": 1.0 if ctx_sim >= 0.80 else 0.0,
            "context_gap": max(0.0, float(getattr(cand, "semantic_score", 0.0) or 0.0) - ctx_sim),
        }

    if LABEL_EXPANSION_DEBUG:
        print(
            f"[Stage2A conditioned_retrieval] anchor={getattr(anchor, 'anchor', '')!r} "
            f"conditioned_retrieval_mode={retrieval_mode!r}\n"
            f"  strong_conditioned_raw_hits={strong_raw_pre} strong_conditioned_postfilter_hits={strong_post}\n"
            f"  light_conditioned_raw_hits={light_raw_pre} light_conditioned_postfilter_hits={light_post}"
        )
    if LABEL_EXPANSION_DEBUG and STAGE2_NOISY_DEBUG and context_neighbors:
        print(
            f"[Stage2A] conditioned_vec 命中 anchor={getattr(anchor, 'anchor', '')!r} -> "
            f"{len(context_neighbors)} 个 前3: {[c.term for c in context_neighbors[:3]]}"
        )
    return context_neighbors, rerank_signals


# ---------- Stage2A 极简版：3 个判断 + 1 个轻量排序，不堆参数 ----------

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ---------- Stage2A 极简重构：组内相对排序（无固定阈值） ----------


def _assign_rank_percentile(candidates: List["Stage2ACandidate"], key: str, out_key: str) -> None:
    """同锚点内按 key 排序，将百分位 [0,1] 写入 relative_scores[out_key]。"""
    if not candidates:
        return
    vals = [getattr(c, key, 0.0) for c in candidates]
    n = len(vals)
    order = sorted(range(n), key=lambda i: (vals[i], -i), reverse=True)
    rank_by_idx = {}
    for r, i in enumerate(order):
        rank_by_idx[i] = 1.0 - (r / (n - 1)) if n > 1 else 1.0
    for i, c in enumerate(candidates):
        c.relative_scores[out_key] = rank_by_idx.get(i, 0.0)


def _assign_rank_percentile_nullable(candidates: List["Stage2ACandidate"], key: str, out_key: str) -> None:
    """同锚点内按 key 排序，None 不参与排序，有值的给百分位；无值写 0.5。"""
    if not candidates:
        return
    valid = [(i, c) for i, c in enumerate(candidates) if getattr(c, key, None) is not None]
    if not valid:
        for c in candidates:
            c.relative_scores[out_key] = 0.5
        return
    n = len(valid)
    order = sorted(range(n), key=lambda j: (getattr(valid[j][1], key), -valid[j][0]), reverse=True)
    rank_by_orig = {valid[order[r]][0]: (1.0 - (r / (n - 1)) if n > 1 else 1.0) for r in range(n)}
    for i, c in enumerate(candidates):
        c.relative_scores[out_key] = rank_by_orig.get(i, 0.5)


def _assign_reverse_rank_percentile(candidates: List["Stage2ACandidate"], key: str, out_key: str) -> None:
    """风险项：值越大越差，按 key 升序排，最小的给最高分。"""
    if not candidates:
        return
    vals = [getattr(c, key, 0.0) for c in candidates]
    n = len(vals)
    order = sorted(range(n), key=lambda i: (vals[i], i))  # 升序，order[0]=最小
    rank_by_idx = {order[r]: (1.0 - (r / (n - 1)) if n > 1 else 1.0) for r in range(n)}
    for i, c in enumerate(candidates):
        c.relative_scores[out_key] = rank_by_idx.get(i, 0.0)


def _assign_mainline_rank_from_preference(candidates: List["Stage2ACandidate"]) -> None:
    """按复合主线偏好 mainline_preference 排序，赋 mainline_rank 百分位（0~1）。"""
    if not candidates:
        return
    n = len(candidates)
    order = sorted(
        range(n),
        key=lambda i: mainline_preference_sort_key(getattr(candidates[i], "mainline_preference", None)),
        reverse=True,
    )
    rank_by_idx = {order[r]: (1.0 - (r / (n - 1))) if n > 1 else 1.0 for r in range(n)}
    for i, c in enumerate(candidates):
        c.relative_scores["mainline_rank"] = rank_by_idx.get(i, 0.0)


def assign_relative_scores_within_anchor(candidates: List["Stage2ACandidate"]) -> None:
    """把绝对值转为锚点内部相对位置，不依赖固定阈值；mainline_rank 由复合主线偏好决定。"""
    if not candidates:
        return
    _assign_rank_percentile(candidates, "semantic_score", "semantic_rank")
    _assign_rank_percentile(candidates, "context_sim", "context_rank")
    _assign_rank_percentile(candidates, "jd_align", "jd_rank")
    _assign_mainline_rank_from_preference(candidates)
    _assign_rank_percentile(candidates, "cross_anchor_support", "cross_anchor_rank")
    _assign_rank_percentile(candidates, "family_match", "family_rank")
    _assign_rank_percentile(candidates, "hierarchy_consistency", "hier_rank")
    _assign_reverse_rank_percentile(candidates, "polysemy_risk", "polysemy_rank")
    _assign_reverse_rank_percentile(candidates, "isolation_risk", "isolation_rank")


def compose_anchor_internal_rank_score(cand: "Stage2ACandidate") -> float:
    """组内综合排序：证据均值，不加新参数。"""
    keys = [
        "semantic_rank", "context_rank", "jd_rank", "mainline_rank",
        "cross_anchor_rank", "family_rank", "hier_rank",
        "polysemy_rank", "isolation_rank",
    ]
    vals = [cand.relative_scores.get(k) for k in keys if cand.relative_scores.get(k) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def is_competitive_runner_up(
    best: "Stage2ACandidate",
    second: "Stage2ACandidate",
    candidates: List["Stage2ACandidate"],
) -> bool:
    """不用固定阈值：第二名是否与第一名形成「前二集团」而值得一起保留。极简：组内 mainline_rank 不差于最后一名。"""
    if len(candidates) <= 2:
        return True
    mr_second = (getattr(second, "relative_scores", None) or {}).get("mainline_rank", 0.0)
    mr_last = (getattr(candidates[-1], "relative_scores", None) or {}).get("mainline_rank", 0.0)
    return (mr_second or 0.0) >= (mr_last or 0.0)


def judge_expandability_relative(anchor: PreparedAnchor, cand: "Stage2ACandidate", candidates: List["Stage2ACandidate"]) -> bool:
    """是否可扩散：仅对 surviving primary 判断，用相对证据（强维度数 >= 弱维度数）。"""
    if not cand.survive_primary:
        return False
    dims = ["semantic_rank", "context_rank", "jd_rank", "mainline_rank", "cross_anchor_rank"]
    strong = sum(1 for k in dims if (cand.relative_scores.get(k) or 0) >= 0.5)
    weak = len(dims) - strong
    return strong >= weak


def build_stage2a_mainline_profile(
    recall: Any,
    anchors: List[PreparedAnchor],
    query_vector: Optional[np.ndarray] = None,
    query_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Stage2A 主线画像：质心 + 锚点 term/vid，供主线偏好排序用，不用于硬 reject。"""
    anchor_terms: List[str] = []
    anchor_vids: List[int] = []
    vecs: List[np.ndarray] = []
    for a in anchors:
        anchor_terms.append(getattr(a, "anchor", "") or "")
        anchor_vids.append(int(getattr(a, "vid", 0) or 0))
        cv = getattr(a, "conditioned_vec", None)
        if cv is not None:
            try:
                v = np.asarray(cv, dtype=np.float32).flatten()
                if v.size > 0:
                    vecs.append(v)
            except Exception:
                pass
            continue
        anchor_vec = _get_candidate_vec_for_mainline(recall, anchor_vids[-1])
        if anchor_vec is not None:
            vecs.append(anchor_vec)
    centroid = None
    if vecs:
        mat = np.array(vecs, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        centroid = np.mean(mat, axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid = centroid / norm
    return {
        "centroid": centroid,
        "anchor_terms": anchor_terms,
        "anchor_vids": anchor_vids,
        "query_vector": query_vector,
    }


def mainline_preference_sort_key(pref: Optional[Any]) -> Tuple[float, ...]:
    """把主线偏好（标量或旧版字典）转为排序元组。"""
    if pref is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    if isinstance(pref, (int, float)):
        return (float(pref), 0.0, 0.0, 0.0, 0.0)
    if not isinstance(pref, dict):
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        float(pref.get("mainline_sim") if pref.get("mainline_sim") is not None else 0.0),
        float(pref.get("anchor_continuity") or 0.0),
        float(pref.get("jd_continuity") or 0.0),
        float(pref.get("cross_support") or 0.0),
        float(pref.get("anti_polysemy") or 0.0),
    )


def build_stage2a_sort_key(cand: "Stage2ACandidate") -> Tuple[Any, ...]:
    """主线第一优先级排序键：mainline_rank 首位，其余为组内相对秩。"""
    rs = getattr(cand, "relative_scores", None) or {}
    return (
        rs.get("mainline_rank", 0.0),
        rs.get("semantic_rank", 0.0),
        rs.get("context_rank", 0.0),
        rs.get("jd_rank", 0.0),
        rs.get("cross_anchor_rank", 0.0),
        rs.get("family_rank", 0.0),
        rs.get("hier_rank", 0.0),
        rs.get("polysemy_rank", 0.0),
        rs.get("isolation_rank", 0.0),
    )


def build_mainline_preference(cand: Any, return_breakdown: bool = False) -> Any:
    """
    只负责「像不像该锚点主干」的局部判断，仅用于 2A 内部排序。
    正向：anchor_identity_score, context_continuity, jd_candidate_alignment, hierarchy_consistency；
    负向：polysemy_risk, object_like_risk, generic_risk（惩罚项，不过重）。
    """
    anchor_identity = float(
        getattr(cand, "anchor_identity_score", None)
        or getattr(cand, "anchor_identity", None)
        or getattr(cand, "family_match", 0)
        or 0
    )
    context_continuity = float(getattr(cand, "context_continuity", 0) or 0)
    jd_align = float(
        getattr(cand, "jd_align", 0) or getattr(cand, "jd_candidate_alignment", 0) or 0
    )
    hierarchy_consistency = float(getattr(cand, "hierarchy_consistency", 0) or 0)
    context_gain = float(getattr(cand, "context_gain", 0) or 0)
    context_gain_score = _clip01((context_gain + 0.10) / 0.20)
    dual_support_bonus = 0.08 if getattr(cand, "dual_support", False) else 0.0
    polysemy_risk = float(getattr(cand, "polysemy_risk", 0) or 0)
    object_like_risk = float(getattr(cand, "object_like_risk", 0) or 0)
    generic_risk = float(getattr(cand, "generic_risk", 0) or 0)

    positive = (
        0.30 * anchor_identity
        + 0.20 * jd_align
        + 0.20 * context_continuity
        + 0.15 * context_gain_score
        + 0.10 * hierarchy_consistency
        + dual_support_bonus
    )
    negative = (
        0.10 * polysemy_risk
        + 0.08 * object_like_risk
        + 0.06 * generic_risk
    )
    pref = float(positive - negative)
    if return_breakdown:
        return pref, {
            "positive": positive,
            "negative": negative,
            "anchor_identity": anchor_identity,
            "context_continuity": context_continuity,
            "jd_align": jd_align,
            "context_gain_score": context_gain_score,
            "dual_support_bonus": dual_support_bonus,
            "hierarchy_consistency": hierarchy_consistency,
            "polysemy_risk": polysemy_risk,
            "object_like_risk": object_like_risk,
            "generic_risk": generic_risk,
        }
    return pref


def compute_mainline_similarity(
    label: Any,
    cand: "Stage2ACandidate",
    mainline_profile: Dict[str, Any],
) -> Optional[float]:
    """候选与主线质心的相似度；无向量时返回 None（表示未知，不判死）。"""
    cand_vec = _get_candidate_vec_for_mainline(label, cand.tid)
    centroid = mainline_profile.get("centroid")
    if cand_vec is None or centroid is None:
        return None
    return float(_cos_sim_mainline(cand_vec, centroid))


def build_cross_anchor_index(
    per_anchor_candidates: Dict[int, List["Stage2ACandidate"]],
) -> Dict[int, List[int]]:
    """tid -> 支持该候选的 anchor vid 列表，供 cross_anchor_support 证据用。"""
    index: Dict[int, List[int]] = {}
    for anchor_vid, cands in per_anchor_candidates.items():
        for c in cands:
            tid = c.tid
            if tid not in index:
                index[tid] = []
            index[tid].append(anchor_vid)
    return index


def _compute_cross_anchor_support_score(tid: int, cross_anchor_index: Dict[int, List[int]]) -> float:
    """0~1：该 tid 被多少锚点支持，归一化到组内。"""
    support_list = cross_anchor_index.get(tid, [])
    if not support_list:
        return 0.0
    max_sup = max(len(v) for v in cross_anchor_index.values()) if cross_anchor_index else 1
    return len(support_list) / max(max_sup, 1)


def _compute_hierarchy_consistency_for_candidate(
    label: Any,
    tid: int,
    jd_field_ids: Optional[Set[str]],
    jd_subfield_ids: Optional[Set[str]],
    jd_topic_ids: Optional[Set[str]],
) -> float:
    """单值层级一致性，供组内相对排序用。"""
    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))
    if not jd_f and not jd_s and not jd_t:
        return 1.0
    ev = compute_hierarchy_evidence(label, tid, jd_f, jd_s, jd_t)
    path = float(ev.get("effective_path_match", 0) or 0)
    topic = float(ev.get("effective_topic_overlap", 0) or 0)
    sub = float(ev.get("effective_subfield_overlap", 0) or 0)
    return (0.4 * path + 0.35 * topic + 0.25 * sub)


def _compute_isolation_risk_for_candidates(
    label: Any,
    candidates: List["Stage2ACandidate"],
) -> None:
    """原地写入每个候选的 isolation_risk（与同组其他候选的语义平均相似度越低越高）。"""
    vocab_idx = getattr(label, "vocab_to_idx", None)
    all_vecs = getattr(label, "all_vocab_vectors", None)
    if not candidates or vocab_idx is None or all_vecs is None:
        for c in candidates:
            c.isolation_risk = 0.5
        return
    vecs: Dict[int, Optional[np.ndarray]] = {}
    for c in candidates:
        if c.tid in vecs:
            continue
        idx = label.vocab_to_idx.get(str(c.tid))
        if idx is None:
            vecs[c.tid] = None
            continue
        try:
            v = np.asarray(label.all_vocab_vectors[idx], dtype=np.float32).flatten()
            vecs[c.tid] = v
        except Exception:
            vecs[c.tid] = None
    for c in candidates:
        v = vecs.get(c.tid)
        if v is None or v.size == 0:
            c.isolation_risk = 0.5
            continue
        sims = []
        for c2 in candidates:
            if c2.tid == c.tid:
                continue
            v2 = vecs.get(c2.tid)
            if v2 is None or v2.size != v.size:
                continue
            try:
                s = float(np.dot(v, v2))
                s = max(-1.0, min(1.0, s))
                sims.append(s)
            except Exception:
                pass
        if not sims:
            c.isolation_risk = 0.5
            continue
        mean_sim = max(0.0, float(np.mean(sims)))
        c.isolation_risk = 1.0 - mean_sim


def enrich_stage2a_candidates(
    label: Any,
    anchor: PreparedAnchor,
    candidates: List["Stage2ACandidate"],
    anchors: List[PreparedAnchor],
    mainline_profile: Dict[str, Any],
    cross_anchor_index: Dict[int, List[int]],
    query_vector: Optional[np.ndarray] = None,
    query_text: Optional[str] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> List["Stage2ACandidate"]:
    """对每个候选补全证据；统一补 admission 结果、mainline_preference 与 role_in_anchor_candidate_pool。"""
    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))

    max_sup = max(len(v) for v in cross_anchor_index.values()) if cross_anchor_index else 1
    for c in candidates:
        c.context_gain = float(getattr(c, "context_gain", 0) or 0.0)
        c.has_dynamic_support = getattr(c, "conditioned_sim", None) is not None
        c.has_static_support = getattr(c, "surface_sim", None) is not None
        c.dual_support = c.has_dynamic_support and c.has_static_support
        c.jd_align = _compute_jd_candidate_alignment(label, c.tid, query_vector) if query_vector is not None else 0.5
        anchor_list = list(cross_anchor_index.get(c.tid, []))
        c.cross_anchor_support_count = len(anchor_list)
        c.cross_anchor_support_weight = len(anchor_list) / max(max_sup, 1)
        c.cross_anchor_anchor_list = anchor_list
        c.cross_anchor_support = _compute_cross_anchor_support_score(c.tid, cross_anchor_index)
        c.family_match = compute_anchor_identity_score(
            getattr(anchor, "anchor", "") or "",
            c.term or "",
            getattr(anchor, "anchor_type", None),
            edge_strength=float(getattr(c, "semantic_score", 0) or 0),
            context_sim=float(getattr(c, "context_sim", 0) or 0),
        )
        c.anchor_identity_score = c.family_match
        c.hierarchy_consistency = _compute_hierarchy_consistency_for_candidate(label, c.tid, jd_f, jd_s, jd_t)
        ev = compute_hierarchy_evidence(label, c.tid, jd_f, jd_s, jd_t)
        c.field_fit = float(ev.get("field_overlap", 0) or 0)
        c.subfield_fit = float(ev.get("effective_subfield_overlap", 0) or 0)
        c.topic_fit = float(ev.get("effective_topic_overlap", 0) or 0)
        c.polysemy_risk, _ = compute_polysemy_risk(c.term or "", c.family_match)

        admission = check_primary_admission(c)
        c.retain_mode = admission["retain_mode"]
        c.suppress_seed = admission["suppress_seed"]
        c.admission_reasons = list(admission.get("reasons", []))
        c.primary_eligibility_reasons = list(admission.get("primary_eligibility_reasons", admission.get("reasons", [])))

        c.mainline_preference = build_mainline_preference(c)

        if c.retain_mode == "reject":
            c.role_in_anchor_candidate_pool = "reject_candidate"
        elif c.retain_mode == "normal":
            c.role_in_anchor_candidate_pool = "mainline_candidate"
        else:
            c.role_in_anchor_candidate_pool = "secondary_candidate"

        score_stage2a_candidate(c)

    _compute_isolation_risk_for_candidates(label, candidates)
    assign_relative_scores_within_anchor(candidates)
    for c in candidates:
        c.composite_rank_score = compose_anchor_internal_rank_score(c)
    ranked = sorted(candidates, key=lambda x: (getattr(x, "mainline_pref_score", 0) or 0), reverse=True)
    for idx, c in enumerate(ranked, start=1):
        c.anchor_internal_rank = idx
        c.mainline_rank = c.relative_scores.get("mainline_rank")
        c.sort_key_snapshot = build_stage2a_sort_key(c)
    return ranked


def _get_stage2a_feat(cand: Any) -> Any:
    """从候选取出 2A 判定用特征（避免重复 getattr）。"""
    ctx_cont = float(getattr(cand, "context_continuity", 0) or getattr(cand, "context_sim", 0) or 0)
    context_supported = bool(getattr(cand, "ctx_supported", getattr(cand, "context_supported", False)))
    if ctx_cont <= 0 and context_supported:
        ctx_cont = 0.55
    return SimpleNamespace(
        anchor_id=float(getattr(cand, "anchor_identity_score", 0) or getattr(cand, "family_match", 0) or 0),
        mainline_pref=float(getattr(cand, "mainline_pref_score", 0) or 0),
        jd_align=float(getattr(cand, "jd_align", 0.5) or 0.5),
        ctx_cont=ctx_cont,
        hier=float(getattr(cand, "hierarchy_consistency", 0) or 0),
        poly=float(getattr(cand, "polysemy_risk", 0) or 0),
        object=float(getattr(cand, "object_like_risk", 0) or 0),
        context_supported=context_supported,
    )


def _stage2a_rule_flags(cand: Any, feat: Any) -> Dict[str, Any]:
    """判定链路调试：返回各规则布尔结果，供 [Stage2A rule flags] 打印。"""
    identity_ok = feat.anchor_id >= STAGE2A_EXPANDABLE_IDENTITY_MIN
    ctx_ok = feat.ctx_cont >= 0.50
    hier_ok = feat.hier >= STAGE2A_BAD_OBJECT_OR_POLY_HIER_CAP
    poly_bad = feat.poly >= STAGE2A_POLY_HARD_RISK
    object_bad = feat.object >= STAGE2A_BAD_OBJECT_MIN
    drift_bad = _is_semantic_drift_branch_stage2a(cand, feat)
    obvious_bad = _is_obviously_bad_branch_stage2a(cand, feat)
    tech_keep_ok = _is_weak_technical_keep_candidate(cand)
    mainline_pref_ok = feat.mainline_pref >= STAGE2A_MAINLINE_PREF_MIN
    jd_ok = feat.jd_align >= STAGE2A_WEAK_TECH_JD_MIN
    context_gain_ok = float(getattr(cand, "context_gain", 0) or 0) >= 0.03
    dynamic_support_ok = bool(getattr(cand, "has_dynamic_support", False))
    dual_support_ok = bool(getattr(cand, "dual_support", False))
    return {
        "identity_ok": identity_ok,
        "ctx_ok": ctx_ok,
        "hier_ok": hier_ok,
        "poly_bad": poly_bad,
        "object_bad": object_bad,
        "drift_bad": drift_bad,
        "obvious_bad": obvious_bad,
        "tech_keep_ok": tech_keep_ok,
        "mainline_pref_ok": mainline_pref_ok,
        "jd_ok": jd_ok,
        "context_gain_ok": context_gain_ok,
        "dynamic_support_ok": dynamic_support_ok,
        "dual_support_ok": dual_support_ok,
    }


def _is_object_or_poly_bad_branch_stage2a(cand: Any, feat: Any) -> bool:
    """明显对象/多义坏分支：surgical robot、robotic surgery、dyskinesia、sports science、control (management)、control flow。"""
    if feat.anchor_id >= STAGE2A_BAD_OBJECT_OR_POLY_ID_CAP:
        return False
    if feat.ctx_cont >= STAGE2A_BAD_OBJECT_OR_POLY_CTX_CAP:
        return False
    if feat.object < STAGE2A_BAD_OBJECT_MIN and feat.poly < STAGE2A_BAD_POLY_MIN:
        return False
    if feat.hier >= STAGE2A_BAD_OBJECT_OR_POLY_HIER_CAP:
        return False
    return True


def _is_weak_technical_keep_candidate(cand: Any) -> bool:
    """
    弱技术词保活（二次修正）：simulation、vibration、feedback control 等可进 primary_keep_no_expand，
    不被 hard_reject；在 semantic_drift / low_mainline_no_context 前生效。
    """
    st = (getattr(cand, "source_type", None) or getattr(cand, "source", "") or "").strip().lower()
    if st not in ("similar_to", "conditioned_vec", "conditioned_vec_fallback", ""):
        if st:
            return False
    anchor_id = float(getattr(cand, "anchor_identity_score", 0) or getattr(cand, "family_match", 0) or 0)
    if anchor_id < 0.20:
        return False
    jd_align = float(getattr(cand, "jd_align", 0.5) or 0.5)
    if jd_align < 0.74:
        return False
    ctx_cont = float(getattr(cand, "context_continuity", 0) or getattr(cand, "context_sim", 0) or 0)
    if ctx_cont < 0.44:
        return False
    poly_risk = float(getattr(cand, "polysemy_risk", 0) or 0)
    if poly_risk >= 0.55:
        return False
    object_like = float(getattr(cand, "object_like_risk", 0) or 0)
    if object_like >= 0.35:
        return False
    return True


def _is_semantic_drift_branch_stage2a(cand: Any, feat: Any) -> bool:
    """
    抽象漂移分支：动力学→dynamism/propulsion；二次修正：弱技术词豁免，vibration/simulation 不误杀。
    收紧：anchor_id>=0.24 不判漂移；(anchor_id<0.24 且 jd_align<0.76) 或 (anchor_id<0.24 且 ctx_cont<0.46) 才判漂移。
    """
    if _is_weak_technical_keep_candidate(cand):
        return False
    if feat.anchor_id >= 0.24:
        return False
    if feat.jd_align < 0.76:
        return True
    if feat.ctx_cont < 0.46:
        return True
    return False


def _is_obviously_bad_branch_stage2a(cand: Any, feat: Any) -> bool:
    """明显坏分支：对象/多义坏 或 抽象漂移；dynamism/propulsion/surgical robot 等直接砍。"""
    term = (getattr(cand, "term", None) or cand.get("term") or "").strip().lower()
    if "(management)" in term:
        return True
    if _is_object_or_poly_bad_branch_stage2a(cand, feat):
        return True
    if _is_semantic_drift_branch_stage2a(cand, feat):
        return True
    return False


def _is_weak_but_technical_keep_stage2a(cand: Any, feat: Any) -> bool:
    """弱相关但技术主干词：simulation、feedback control、vibration、mechanics 保留进 primary_keep_no_expand。"""
    if feat.anchor_id >= STAGE2A_WEAK_TECH_ID_CAP:
        return False
    if feat.ctx_cont >= STAGE2A_WEAK_TECH_CTX_CAP:
        return False
    if feat.jd_align < STAGE2A_WEAK_TECH_JD_MIN:
        return False
    if feat.poly >= STAGE2A_WEAK_TECH_POLY_CAP:
        return False
    if feat.object >= STAGE2A_WEAK_TECH_OBJECT_CAP:
        return False
    return True


def _should_hard_reject_stage2a_candidate(cand: Any) -> Tuple[bool, str]:
    """
    Stage2A 分桶判定顺序：① 明显坏分支 → hard_reject；② 弱相关技术词 → primary_keep_no_expand；
    ③ low_identity/low_mainline_no_context → hard_reject；④ 高歧义无上下文 → hard_reject。
    只改桶边界，不改 score 公式。
    """
    feat = _get_stage2a_feat(cand)
    anchor_identity = feat.anchor_id
    mainline_pref = feat.mainline_pref
    poly_risk = feat.poly
    context_supported = feat.context_supported

    low_identity_no_ctx = (not context_supported) and anchor_identity < STAGE2A_REJECT_IDENTITY_FLOOR
    low_mainline_no_ctx = (not context_supported) and mainline_pref < STAGE2A_REJECT_MAINLINE_FLOOR
    obviously_bad = _is_obviously_bad_branch_stage2a(cand, feat)
    weak_but_technical = _is_weak_but_technical_keep_stage2a(cand, feat)
    weak_tech_keep_candidate = _is_weak_technical_keep_candidate(cand)

    if obviously_bad:
        setattr(cand, "stage2a_reject_cls", "hard_reject")
        if _is_semantic_drift_branch_stage2a(cand, feat):
            return True, "semantic_drift_branch"
        if _is_object_or_poly_bad_branch_stage2a(cand, feat):
            return True, "object_or_poly_bad_branch"
        return True, "obviously_bad_branch"
    if low_mainline_no_ctx:
        if weak_tech_keep_candidate:
            setattr(cand, "stage2a_reject_cls", "weak_tech_keep")
            setattr(cand, "precheck_hint", "weak_tech_keep")
            return False, ""
        setattr(cand, "stage2a_reject_cls", "hard_reject")
        return True, "low_mainline_no_context"
    if weak_but_technical or weak_tech_keep_candidate:
        setattr(cand, "stage2a_reject_cls", "weak_tech_keep")
        setattr(cand, "precheck_hint", "weak_tech_keep")
        return False, ""
    if low_identity_no_ctx:
        setattr(cand, "stage2a_reject_cls", "hard_reject")
        return True, "low_identity_no_context"
    if low_mainline_no_ctx:
        setattr(cand, "stage2a_reject_cls", "hard_reject")
        return True, "low_mainline_no_context"
    if poly_risk >= STAGE2A_POLY_HARD_RISK and (not context_supported) and anchor_identity < 0.40:
        setattr(cand, "stage2a_reject_cls", "hard_reject")
        return True, "polysemous_no_context"
    setattr(cand, "stage2a_reject_cls", "pass")
    return False, ""


def _stage2a_static_ctx_for_primary_expand_split(c: "Stage2ACandidate") -> Tuple[float, float, float, float]:
    """双路相似度：static=semantic_score，ctx 优先 conditioned_sim，否则 context_sim / context_continuity。"""
    static_sim = float(getattr(c, "semantic_score", 0.0) or 0.0)
    cs = getattr(c, "conditioned_sim", None)
    if cs is not None:
        ctx_sim = float(cs)
    else:
        ctx_raw = getattr(c, "context_sim", None)
        if ctx_raw is None:
            ctx_raw = getattr(c, "context_continuity", 0)
        ctx_sim = float(ctx_raw) if ctx_raw is not None else 0.0
    if static_sim > 0 and ctx_sim > 0:
        ctx_drop = max(0.0, static_sim - ctx_sim)
    else:
        ctx_drop = 0.0
    best_sim = max(static_sim, ctx_sim)
    return static_sim, ctx_sim, ctx_drop, best_sim


def _stage2a_has_real_conditioned_sim(c: "Stage2ACandidate") -> bool:
    """仅图/合并产出的 conditioned_sim 算「真实上下文纠偏」；无则不得靠 ctx_drop=0 当支持。"""
    cs = getattr(c, "conditioned_sim", None)
    if cs is None:
        return False
    try:
        return float(cs) > STAGE2A_CONDITIONED_SIM_EPS
    except (TypeError, ValueError):
        return False


def _classify_keepish_primary_bucket(
    cand: "Stage2ACandidate",
    snap: Dict[str, Any],
    *,
    expand_block: str,
    prefer_seed: bool,
    no_real_ctx: bool = False,
) -> Tuple[str, str]:
    """
    五层体系中，在已通过 primary_ok 但未能评为 primary_expandable 时的分流（主线近邻 seed / 支线保留 / 高风险保留）。
    规则优先级与 README「二、五层判定规则」一致：先挡明显高风险泛词/歧义，再尝试弱 seed，再支线 keep。
    """
    gen = float(getattr(cand, "generic_risk", 0) or 0)
    poly = float(getattr(cand, "polysemy_risk", 0) or 0)
    obj = float(getattr(cand, "object_like_risk", 0) or 0)
    mainline_cand = bool(getattr(cand, "mainline_admissible", False))
    pool = (getattr(cand, "role_in_anchor_candidate_pool", "") or "").strip().lower()
    if pool == "mainline_candidate":
        mainline_cand = True
    ctx_path_ok = bool(snap.get("context_path_ok"))
    dual_support = bool(snap.get("dual_support"))
    snap["can_expand_from_2a"] = False

    if no_real_ctx:
        if gen >= STAGE2A_RISKY_KEEP_HIGH_GENERIC * 0.85 or poly >= STAGE2A_RISKY_KEEP_HIGH_POLY * 0.85:
            return "risky_keep", "risky_no_ctx_high_generic_poly"
        if mainline_cand:
            return "primary_support_keep", "usable_mainline_no_ctx_keep"
        return "risky_keep", "risky_no_ctx_weak_mainline"

    # 对象/部件型偏高：保留进支线，不做弱 seed（易把池带偏）
    if obj >= STAGE2A_EXPAND_MAX_OBJECT_LIKE and mainline_cand and ctx_path_ok:
        return "primary_support_keep", "object_branch_keep"

    # 明显「高风险保留」档（与弱 seed 的 gen<0.65 衔接）
    if gen >= STAGE2A_RISKY_KEEP_HIGH_GENERIC or poly >= STAGE2A_RISKY_KEEP_HIGH_POLY:
        return "risky_keep", "risky_high_generic_or_poly"
    weak_mainline = (not dual_support) and (not mainline_cand)
    if weak_mainline and gen >= 0.42:
        return "risky_keep", "risky_weak_mainline_support"

    # 主线近邻弱 seed：给 can_expand_from_2a，由 stage2b_seed_tier=weak 收紧 Stage2B
    if (
        prefer_seed
        and mainline_cand
        and ctx_path_ok
        and gen < STAGE2A_WEAK_SEED_GENERIC_CAP
        and poly < STAGE2A_WEAK_SEED_POLY_CAP
        and obj < STAGE2A_WEAK_SEED_OBJ_CAP
        and expand_block
        in (
            "ctx_drop_or_branch_risk",
            "good_mainline_but_risky_or_weak_identity",
            "ctx_thresholds_not_met",
        )
    ):
        snap["can_expand_from_2a"] = True
        return "primary_support_seed", "mainline_neighbor_weak_seed"

    if mainline_cand and ctx_path_ok:
        return "primary_support_keep", "usable_mainline_branch_keep"

    return "risky_keep", "risky_weak_evidence"


def judge_primary_and_expandability(
    c: "Stage2ACandidate",
    anchor_term: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    单候选：primary 资格与 expand 资格解耦；**五层落点**（见 README「Stage2A 五层落点体系」）：
    primary_expandable / primary_support_seed / primary_support_keep / risky_keep / reject。
    primary：须（真实 conditioned 上下文合格）或（多锚 + 高 family）或（双路 + conditioned 合格）；
    禁止仅凭 static 相似 + 无 conditioned 进主线。
    **expand 与 seed 身份**：在 primary_ok 之后，用连续特征构造 **axis_consistency / effective_mainline**
   （主线一致性）二次收口，避免「弱相关但近邻」误判为 expandable 或弱 seed（详见 README）。
    返回 (bucket, reason, snapshot) 供 [Stage2A primary factors] 与 expand 调试打印。
    anchor_term：仅用于定点 Focus 调试打印，可省略。
    """
    _anc = (anchor_term or "").strip()
    _term_dbg = (getattr(c, "term", None) or "").strip()

    rr = getattr(c, "reject_reason", None) or getattr(c, "stage2a_hard_reject_reason", None) or ""
    if rr in ("obviously_bad_branch", "object_or_poly_bad_branch", "semantic_drift_branch"):
        snap_bad: Dict[str, Any] = {
            "prior_reject_reason": rr,
            "bad_branch": True,
            "can_expand_from_2a": False,
        }
        _stage2a_enrich_light_judge_snapshot(snap_bad, "reject", c)
        return "reject", "bad_branch", snap_bad

    static_sim = float(getattr(c, "semantic_score", 0.0) or 0.0)
    has_real_ctx = _stage2a_has_real_conditioned_sim(c)
    ctx_sim = float(getattr(c, "conditioned_sim", 0.0) or 0.0) if has_real_ctx else 0.0
    if static_sim > 0 and ctx_sim > 0:
        ctx_drop = max(0.0, static_sim - ctx_sim)
    else:
        ctx_drop = 0.0
    best_sim = max(static_sim, ctx_sim) if has_real_ctx else static_sim

    dual_support = bool(getattr(c, "dual_support", False))
    support_count = int(getattr(c, "cross_anchor_support_count", 1) or 1)
    multi_anchor_ok = support_count >= 2
    family_match = float(getattr(c, "family_match", 0) or getattr(c, "anchor_identity_score", 0) or 0.0)

    semantic_ok = best_sim >= STAGE2A_PRIMARY_MIN_SIM
    drop_ok_primary = (static_sim <= STAGE2A_CONDITIONED_SIM_EPS) or (ctx_drop <= STAGE2A_PRIMARY_MAX_CTX_DROP)

    ctx_path_ok = (
        has_real_ctx
        and ctx_sim >= STAGE2A_PRIMARY_MIN_CTX_SIM
        and drop_ok_primary
    )
    structural_multi = (
        multi_anchor_ok
        and (not has_real_ctx)
        and family_match >= STAGE2A_PRIMARY_MULTI_MIN_FAMILY
        and static_sim >= STAGE2A_PRIMARY_MIN_SIM
    )
    structural_dual = (
        dual_support
        and has_real_ctx
        and ctx_sim >= STAGE2A_PRIMARY_DUAL_MIN_CTX
        and drop_ok_primary
    )

    primary_ok = semantic_ok and (ctx_path_ok or structural_multi or structural_dual)
    snap0: Dict[str, Any] = {
        "static_sim": static_sim,
        "ctx_sim": ctx_sim,
        "ctx_drop": ctx_drop,
        "ctx_present": has_real_ctx,
        "best_sim": best_sim,
        "dual_support": dual_support,
        "multi_anchor_ok": multi_anchor_ok,
        "family_match": family_match,
        "semantic_ok": semantic_ok,
        "context_path_ok": ctx_path_ok,
        "structural_multi": structural_multi,
        "structural_dual": structural_dual,
        "primary_ok": primary_ok,
        "drop_ok_primary": drop_ok_primary,
    }
    if not primary_ok:
        snap0["can_expand_from_2a"] = False
        snap0["source_type"] = (getattr(c, "source_type", None) or getattr(c, "source", "") or "").strip().lower()
        # 无完整 primary 证据但仍有静态/弱相关：高风险保留，供 Stage3 降权（非直接 reject）
        if float(best_sim) >= STAGE2A_RISKY_KEEP_MIN_BEST_SIM:
            _stage2a_enrich_light_judge_snapshot(snap0, "risky_keep", c)
            return "risky_keep", "weak_line_not_primary", snap0
        _stage2a_enrich_light_judge_snapshot(snap0, "reject", c)
        return "reject", "not_primary_enough", snap0

    # ---------- primary_ok 之后：主线一致性（axis）分桶，替代旧 expand/weak_seed 分叉 ----------
    static_sim, ctx_sim, ctx_drop, best_sim = _stage2a_static_ctx_for_primary_expand_split(c)
    jd_align = float(getattr(c, "jd_align", 0.0) or 0.0)
    anchor_identity = float(getattr(c, "anchor_identity_score", 0.0) or 0.0)
    fm = float(getattr(c, "family_match", 0.0) or 0.0)
    if fm <= 0.0:
        fm = anchor_identity
    obj_risk = float(getattr(c, "object_like_risk", 0.0) or 0.0)
    gen_risk = float(getattr(c, "generic_risk", 0.0) or 0.0)
    poly_risk = float(getattr(c, "polysemy_risk", 0.0) or 0.0)
    conditioned_only = _stage2a_is_conditioned_only_for_seed(c)
    ctx_tail = max(0.0, ctx_sim - min(ctx_drop, 0.25))
    mainline_pref = float(getattr(c, "mainline_pref_score", 0.0) or 0.0)
    # axis_consistency：identity + 家族 + JD 对齐 + 上下文稳态 + 主线偏好（无词表）
    axis_consistency = (
        0.30 * anchor_identity
        + 0.25 * fm
        + 0.20 * jd_align
        + 0.15 * ctx_tail
        + 0.10 * max(0.0, mainline_pref)
    )
    risk_penalty = 0.45 * obj_risk + 0.35 * gen_risk + 0.20 * poly_risk
    effective_mainline = axis_consistency - risk_penalty

    source_type = (getattr(c, "source_type", None) or getattr(c, "source", "") or "").strip().lower()
    snap_axis: Dict[str, Any] = {
        "static_sim": static_sim,
        "ctx_sim": ctx_sim,
        "ctx_drop": ctx_drop,
        "best_sim": best_sim,
        "ctx_present": has_real_ctx,
        "dual_support": dual_support,
        "multi_anchor_ok": multi_anchor_ok,
        "family_match": fm,
        "semantic_ok": semantic_ok,
        "context_path_ok": ctx_path_ok,
        "structural_multi": structural_multi,
        "structural_dual": structural_dual,
        "primary_ok": True,
        "drop_ok_primary": drop_ok_primary,
        "axis_consistency": axis_consistency,
        "effective_mainline": effective_mainline,
        "conditioned_only": conditioned_only,
        "object_like_risk": obj_risk,
        "generic_risk": gen_risk,
        "polysemy_risk": poly_risk,
        "anchor_identity_score": anchor_identity,
        "jd_align": jd_align,
        "source_type": source_type,
        "strong_static_ok": static_sim >= STAGE2A_PRIMARY_MIN_SIM,
        "strong_ctx_ok": has_real_ctx and ctx_sim >= STAGE2A_EXPAND_MIN_CTX_SIM,
        "strong_mainline_direct": False,
    }

    bucket: str
    reason: str
    can_expand_flag: bool
    expand_block: Optional[str] = None

    if obj_risk >= 0.55 or poly_risk >= 0.60:
        bucket, reason = "reject", "object_or_poly_bad"
        can_expand_flag = False
        expand_block = "object_or_poly_bad"
    elif (
        effective_mainline >= 0.52
        and anchor_identity >= 0.52
        and fm >= 0.48
        and ctx_sim >= 0.42
        and ctx_drop <= 0.10
        and gen_risk <= 0.52
        and not conditioned_only
    ):
        bucket, reason = "primary_expandable", "axis_strong_mainline"
        can_expand_flag = True
    elif (
        effective_mainline >= 0.42
        and anchor_identity >= 0.42
        and fm >= 0.38
        and jd_align >= 0.50
        and gen_risk <= 0.58
        and poly_risk <= 0.35
        and obj_risk <= 0.25
    ):
        can_expand_flag = (not conditioned_only) and (effective_mainline >= 0.46)
        if can_expand_flag:
            bucket, reason = "primary_support_seed", "axis_neighbor_weak_seed"
        else:
            bucket, reason = "primary_support_keep", "axis_weak_or_conditioned_only"
        expand_block = None if can_expand_flag else "axis_weak_or_conditioned_only"
    elif effective_mainline >= 0.28 and jd_align >= 0.45:
        bucket, reason = "primary_support_keep", "axis_keep_no_expand"
        can_expand_flag = False
        expand_block = "demote_to_keep"
    elif float(best_sim) >= 0.36:
        bucket, reason = "risky_keep", "weak_line_not_primary"
        can_expand_flag = False
        expand_block = "risky_keep_only"
    else:
        bucket, reason = "reject", "low_mainline"
        can_expand_flag = False
        expand_block = "low_mainline"

    snap_axis["can_expand_from_2a"] = bool(
        can_expand_flag and bucket in ("primary_expandable", "primary_support_seed")
    )
    snap_axis["expand_block"] = expand_block

    # 无真实 conditioned 或 family_fallback：只允许保留类落点，禁止 2B 扩散（结构型 primary_ok 仍可能为 True）
    if getattr(c, "family_fallback_only", False) or not has_real_ctx:
        snap_axis["expand_block"] = snap_axis.get("expand_block") or "no_real_conditioned_or_family_fallback"
        if bucket == "primary_expandable":
            bucket, reason = "primary_support_keep", "no_real_ctx_demote_expandable"
        elif bucket == "primary_support_seed":
            bucket, reason = "primary_support_keep", "no_real_ctx_demote_seed"
        snap_axis["can_expand_from_2a"] = False

    _stage2a_enrich_light_judge_snapshot(snap_axis, bucket, c)
    return bucket, reason, snap_axis


def _print_stage2a_primary_factors(
    anchor: PreparedAnchor,
    c: "Stage2ACandidate",
    bucket: str,
    reason: str,
    snap: Dict[str, Any],
) -> None:
    """[Stage2A primary factors]：锚点下该候选为何能/不能当 primary（与 judge_primary_and_expandability 同源）。"""
    if not (LABEL_EXPANSION_DEBUG and STAGE2_RULING_DEBUG):
        return
    anchor_term = getattr(anchor, "anchor", "") or ""
    term = (getattr(c, "term", None) or "")[:48]
    print(f"\n{'-' * 80}\n[Stage2A] Primary factors\n{'-' * 80}")
    print(f"[Stage2A primary factors] anchor={anchor_term!r} term={term!r}")
    if snap.get("bad_branch"):
        print(f"  pre_gate_reject={snap.get('prior_reject_reason')!r}")
        print(f"  primary_ok=False primary_reason={reason!r}")
        return
    src = getattr(c, "source_type", "") or ""
    static_sim = snap.get("static_sim")
    ctx_present = bool(snap.get("ctx_present"))
    ctx_sim = snap.get("ctx_sim")
    ctx_drop = snap.get("ctx_drop")
    _ss = f"{float(static_sim):.3f}" if static_sim is not None else "N/A"
    _cs = f"{float(ctx_sim):.3f}" if ctx_present and ctx_sim is not None else "N/A"
    _cd = f"{float(ctx_drop):.3f}" if ctx_drop is not None else "N/A"
    ctx_path_ok = bool(snap.get("context_path_ok"))
    primary_survived = bucket != "reject"
    print(f"  source_type={src!r}")
    print(f"  static_sim={_ss}")
    print(f"  ctx_sim={_cs}")
    print(f"  ctx_present={ctx_present}")
    print(f"  ctx_drop={_cd}")
    print(f"  semantic_ok={snap.get('semantic_ok')}")
    print(f"  context_ok={ctx_path_ok}")
    print(f"  multi_anchor_ok={snap.get('multi_anchor_ok')}")
    print(f"  local_context_hit={getattr(c, 'local_context_hit', 0)}")
    print(f"  family_centrality={float(getattr(c, 'family_centrality', 0.0) or 0.0):.3f}")
    print(f"  mainline_support_ok={bool(getattr(c, 'mainline_admissible', False))}")
    print(f"  primary_ok={primary_survived}")
    print(f"  primary_reason={reason!r}")
    # expand 侧：final_* 与最终 bucket 一致；snap_can_expand_from_2a 应与 judge 返回值对齐
    print(f"[Stage2A expand factors] anchor={anchor_term!r} term={term!r}")
    print(f"  final_bucket={bucket!r}")
    print(
        f"  final_expandable={bucket == 'primary_expandable'} "
        f"final_weak_seed={bucket == 'primary_support_seed'}"
    )
    print(f"  snap_can_expand_from_2a={snap.get('can_expand_from_2a')!r}")
    print(f"  dual_support={snap.get('dual_support')}")
    print(f"  multi_anchor_ok={snap.get('multi_anchor_ok')}")
    print(f"  static_sim={_ss}")
    print(f"  ctx_sim={_cs}")
    print(f"  ctx_drop={_cd}")
    print(f"  strong_static_ok={snap.get('strong_static_ok', False)}")
    print(f"  strong_ctx_ok={snap.get('strong_ctx_ok', False)}")
    print(f"  strong_mainline_direct={snap.get('strong_mainline_direct', False)}")
    eb = snap.get("expand_block")
    print(f"  expand_reason={reason!r}" + (f" | expand_block={eb!r}" if eb else ""))


def anchor_allows_fallback_primary(anchor: PreparedAnchor) -> bool:
    """仅对 canonical 学术锚点开放；缩写/高歧义型不允许 fallback。"""
    at = (anchor.anchor_type or "").strip().lower()
    if at == "acronym":
        return False
    if at in HIGH_AMBIGUITY_ANCHOR_TYPES:
        return False
    if not _is_canonical_academic_like_anchor(anchor):
        return False
    fa = float(getattr(anchor, "final_anchor_score", 0.0) or 0.0)
    n_local = len(getattr(anchor, "local_phrases", None) or [])
    n_co = len(getattr(anchor, "co_anchor_terms", None) or [])
    strong_anchor = fa >= STAGE2A_FALLBACK_ANCHOR_MIN_SCORE
    strong_context = (n_local >= STAGE2A_FALLBACK_MIN_LOCAL_HITS) or (n_co >= STAGE2A_FALLBACK_MIN_CO_HITS)
    return strong_anchor or strong_context


def _stage2a_source_type_set(c: "Stage2ACandidate") -> Set[str]:
    """合并来源集合（similar_to | conditioned_vec | …），用于 weak seed 结构门。"""
    ss = getattr(c, "source_set", None)
    if ss:
        return {str(x).strip().lower() for x in ss if x}
    st = (getattr(c, "source_type", None) or getattr(c, "source", "") or "").strip().lower()
    return {st} if st else set()


def _primary_landing_source_type_set(p: Any) -> Set[str]:
    """PrimaryLanding 上的 source_set / source_type，与 _stage2a_source_type_set 对齐。"""
    ss = getattr(p, "source_set", None)
    if ss:
        return {str(x).strip().lower() for x in ss if x}
    st = (getattr(p, "source_type", None) or getattr(p, "source", "") or "").strip().lower()
    return {st} if st else set()


def _stage2a_is_conditioned_only_for_seed(c: "Stage2ACandidate") -> bool:
    """仅 conditioned_vec、无 similar_to / family_landing：不得做 weak seed（2A 直接降级为 support_keep）。"""
    ss = _stage2a_source_type_set(c)
    if not ss:
        return False
    has_sim = "similar_to" in ss
    has_fam = "family_landing" in ss
    has_cond = "conditioned_vec" in ss
    return bool(has_cond and (not has_sim) and (not has_fam))


def _finalize_stage2b_seed_tiers(
    primary_expandable: List["Stage2ACandidate"],
    primary_support_seed: List["Stage2ACandidate"],
    primary_support_keep: List["Stage2ACandidate"],
    risky_keep: List["Stage2ACandidate"],
) -> Tuple[List["Stage2ACandidate"], List["Stage2ACandidate"], List["Stage2ACandidate"], List["Stage2ACandidate"]]:
    """
    2A 只负责分层；此处把 **Stage2B seed 资格**收紧为：
    - primary_expandable → tier=strong（可多 strong，2B 对每 seed 分别扩散）
    - primary_support_seed 且 **含 similar_to** 且非 conditioned-only → tier=weak；否则降级为 primary_support_keep，不给扩。
    - 其余 tier=none，不可扩。最终主线由 Stage3 裁决。
    """
    for c in primary_expandable:
        c.stage2b_seed_tier = "strong"
        c.can_expand_from_2a = True
        c.can_expand = True

    new_seed: List["Stage2ACandidate"] = []
    for c in primary_support_seed:
        ss = _stage2a_source_type_set(c)
        # 综述收缩：conditioned-only 不再硬降级为 keep，保留弱 seed + 软标记，全局一致性交 Stage3
        if _stage2a_is_conditioned_only_for_seed(c):
            c.stage2b_seed_tier = "weak"
            c.can_expand_from_2a = True
            c.can_expand = True
            setattr(c, "seed_evidence_soft", True)
            setattr(c, "legacy_seed_note", "compatibility: conditioned-only weak seed — TODO Stage3 collective filter")
            new_seed.append(c)
            continue
        if "similar_to" not in ss:
            c.primary_bucket = "primary_support_keep"
            c.stage2b_seed_tier = "none"
            c.can_expand = False
            c.can_expand_from_2a = False
            c.role_in_anchor = "side"
            c.role = "side"
            setattr(c, "seed_downgrade_reason", "weak_seed_requires_similar_to")
            primary_support_keep.append(c)
        else:
            c.stage2b_seed_tier = "weak"
            c.can_expand_from_2a = True
            c.can_expand = True
            new_seed.append(c)
    primary_support_seed = new_seed

    for c in primary_support_keep:
        c.stage2b_seed_tier = "none"
        c.can_expand = False
        c.can_expand_from_2a = False
    for c in risky_keep:
        c.stage2b_seed_tier = "none"
        c.can_expand = False
        c.can_expand_from_2a = False

    return primary_expandable, primary_support_seed, primary_support_keep, risky_keep


def pick_fallback_primary_for_anchor(
    anchor: PreparedAnchor,
    candidates: List["Stage2ACandidate"],
) -> Optional["Stage2ACandidate"]:
    """
    本锚点在正常选主后 0-primary 时调用：从组内挑一条「最不像坏分支」的保线词，不扩、交 Stage3。
    须与锚点本体词形/家族足够贴近，压低 mechanics/kinesiology/simulation 等泛词保线。
    """
    if not anchor_allows_fallback_primary(anchor):
        return None
    anchor_term = (anchor.anchor or "").strip()
    fallback_pool: List["Stage2ACandidate"] = []
    for cand in candidates:
        feat = _get_stage2a_feat(cand)
        if _is_obviously_bad_branch_stage2a(cand, feat):
            continue
        if _is_object_or_poly_bad_branch_stage2a(cand, feat):
            continue
        if _is_semantic_drift_branch_stage2a(cand, feat):
            continue
        static_sim, ctx_sim, ctx_drop, best_sim = _stage2a_static_ctx_for_primary_expand_split(cand)
        if best_sim < STAGE2A_FALLBACK_MIN_SIM:
            continue
        cand_term = (getattr(cand, "term", "") or "").strip()
        lexical_core = lexical_shape_match(anchor_term, cand_term)
        anchor_identity = float(getattr(cand, "anchor_identity_score", 0.0) or 0.0)
        family_match = float(getattr(cand, "family_match", 0.0) or anchor_identity)
        if lexical_core < 0.35 and family_match < 0.58:
            continue
        obj = float(getattr(cand, "object_like_risk", 0.0) or 0.0)
        gen = float(getattr(cand, "generic_risk", 0.0) or 0.0)
        poly = float(getattr(cand, "polysemy_risk", 0.0) or 0.0)
        if obj >= 0.55 or gen >= 0.55 or poly >= 0.60:
            continue
        fallback_score = (
            0.38 * best_sim
            + 0.18 * ctx_sim
            + 0.16 * (1.0 - min(ctx_drop, 1.0))
            + 0.14 * lexical_core
            + 0.14 * family_match
        )
        setattr(cand, "fallback_score", fallback_score)
        fallback_pool.append(cand)
    if not fallback_pool:
        return None
    fallback_pool.sort(key=lambda x: float(getattr(x, "fallback_score", 0.0)), reverse=True)
    best = fallback_pool[0]
    best.survive_primary = True
    best.can_expand = False
    best.can_expand_from_2a = False
    best.fallback_primary = True
    best.primary_bucket = "primary_fallback_keep_no_expand"
    best.role_in_anchor = "side"
    best.role = "side"
    best.admission_reason = "anchor_core_fallback"
    best.reject_reason = None
    setattr(best, "bucket_reason", "anchor_core_fallback")
    setattr(best, "mainline_block_reason", "anchor_core_fallback")
    return best


def _stage2a_promote_strong_allowed(
    w: Dict[str, Any],
    *,
    axis_gap_ok: bool,
    has_judge_expandable_stable: bool,
) -> Tuple[bool, str]:
    """
    仅影响 **升格到 primary_expandable** 的最后一跳（局部角色/优先级），不决定候选是否从 Stage2 输出中消失。
    keep/seed → primary_expandable：须组内 axis_consistency_seed 排名第 1、与第 2 名间隔够大，
    且 judge 未已给出「稳定强 expandable」；否则留在 seed/sk，交 Stage3 做全局裁决。
    """
    if w["group_rank"] != 1:
        return False, "not_group_top1"
    if not axis_gap_ok:
        return False, "group_axis_gap_too_small"
    if has_judge_expandable_stable and w["pre_bucket"] in ("primary_support_keep", "primary_support_seed"):
        return False, "judge_expandable_stable_exists"
    return True, ""


def _stage2a_phrase_naturalness_score(term: str) -> float:
    """
    Stage2A 组内“术语自然性/规范性”弱偏好（不依赖具体词表）：
    - 更像稳定 academic phrase（长度适中、少噪声符号）→ 更高
    - 过长、噪声符号/数字/下划线等“派生味重” → 降权
    仅用于 **组内相对排序**，不作为硬拒绝。
    """
    if not term:
        return 0.0
    t = str(term).strip()
    if not _lexical_term_sanity(t, None):
        return 0.0
    low = t.lower()
    toks = [x for x in low.replace("-", " ").split() if x]
    n_tok = len(toks)

    # token 数：2-4 更像稳定术语；1 或 >=6 更像泛词/派生描述
    if n_tok <= 0:
        tok_score = 0.0
    elif n_tok == 1:
        tok_score = 0.55
    elif 2 <= n_tok <= 4:
        tok_score = 1.0
    elif n_tok == 5:
        tok_score = 0.72
    else:
        tok_score = 0.45

    # 词面噪声：数字/下划线/多符号会降低“术语感”
    noise = 0.0
    if any(ch.isdigit() for ch in t):
        noise += 0.25
    if "_" in t:
        noise += 0.25
    if any(ch in t for ch in ("(", ")", "[", "]", "{", "}", "/", "\\", "|", ";", ":", "@", "#")):
        noise += 0.15
    # 过长短语偏惩罚（避免把描述句当术语）
    if len(t) >= 56:
        noise += 0.20
    elif len(t) >= 42:
        noise += 0.10

    return _clip01(0.70 * tok_score + 0.30 * (1.0 - _clip01(noise)))


def _stage2a_local_primary_preference_snapshot(
    *,
    anchor_term: str,
    cand: Any,
    axis_consistency_seed: float,
    mainline_pref_score: float,
    jd_align: float,
    family_match: float,
    ctx_sim: float,
    ctx_drop: float,
    context_continuity: float,
    conditioned_only: bool,
    generic_risk: float,
    polysemy_risk: float,
    object_like_risk: float,
) -> Dict[str, Any]:
    """
    Stage2A 的 local candidate ranking / local disambiguation 核心：组内“主位偏好”快照。
    只负责 **相对排序偏好**（谁更可能成为该 anchor 的自然学术落点），不决定全局一致性。
    """
    # 主线/锚一致性：轴一致性 + mainline_pref + family_match
    mainline = (
        0.44 * _clip01(axis_consistency_seed)
        + 0.34 * _clip01(mainline_pref_score)
        + 0.22 * _clip01(family_match)
    )

    # 语境连续性：偏好“真实 ctx + 低 drop + 与 local phrases 连贯”
    ctx_quality = _clip01(
        0.50 * _clip01(ctx_sim)
        + 0.25 * _clip01(context_continuity)
        + 0.25 * (1.0 - _clip01(ctx_drop / 0.10))  # drop<=0.10 近似可信上限
    )

    # 风险项：高风险不能靠偶然高分抢主位（但不一刀切 reject）
    risk = _clip01(
        0.45 * _clip01(generic_risk)
        + 0.30 * _clip01(polysemy_risk)
        + 0.25 * _clip01(object_like_risk)
    )

    # conditioned_only 惩罚：只在“抢主位”时显式降权（保持可保留、但不应轻易 top1）
    cond_pen = 0.18 if conditioned_only else 0.0

    # 短语自然性：不靠词表，轻量偏好规范术语形态
    natural = _stage2a_phrase_naturalness_score(getattr(cand, "term", "") or "")

    # 最终 local preference：主线优先 + 语境连续 + JD 主轴一致 + 风险/cond_only/不自然词形惩罚
    # 注意：这是“组内相对排序”信号，不用于 hard gate，不改变 Stage2B 的强弱阈值体系。
    score = _clip01(
        0.46 * mainline
        + 0.22 * _clip01(jd_align)
        + 0.20 * ctx_quality
        + 0.12 * natural
        - 0.22 * risk
        - cond_pen
    )

    return {
        "local_pref": score,
        "mainline": mainline,
        "ctx_quality": ctx_quality,
        "jd": float(jd_align),
        "family": float(family_match),
        "axis_seed": float(axis_consistency_seed),
        "mainline_pref": float(mainline_pref_score),
        "ctx_sim": float(ctx_sim),
        "ctx_drop": float(ctx_drop),
        "context_cont": float(context_continuity),
        "natural": float(natural),
        "risk": float(risk),
        "cond_only": bool(conditioned_only),
        "risk_parts": {
            "generic": float(generic_risk),
            "poly": float(polysemy_risk),
            "object": float(object_like_risk),
        },
    }


def select_primary_per_anchor(
    anchor: PreparedAnchor,
    candidates: List["Stage2ACandidate"],
) -> Dict[str, List["Stage2ACandidate"]]:
    """
    Stage2A 组内选主：**新三层** candidate_core / candidate_support / candidate_noise（见返回键）+
    **legacy 五层桶**（primary_expandable 等，Stage3 迁移前保留）。
    综述式收缩：仍以 feature 与相对排序为主，细粒度 promote/reconcile 降噪（见 STAGE2_NOISY_DEBUG）。
    """
    empty_out = {
        "primary_expandable": [],
        "primary_support_seed": [],
        "primary_support_keep": [],
        "risky_keep": [],
        "primary_keep_no_expand": [],
        "rejected": [],
        "candidate_core": [],
        "candidate_support": [],
        "candidate_noise": [],
    }
    if not candidates:
        # landing_state 闭环（空候选也必须可观测）：避免 fallback-only / empty anchor 没有 landing summary
        anchor_term_sel = getattr(anchor, "anchor", "") or ""
        stage2_mode = (getattr(anchor, "stage2_process_mode", "") or "").strip()
        setattr(anchor, "landing_state", "empty_landing")
        setattr(anchor, "landing_reason", "no_candidates_generated")
        setattr(anchor, "candidate_generated_count", 0)
        setattr(anchor, "landing_kept_count", 0)
        if LABEL_EXPANSION_DEBUG:
            print(
                f"[Stage2A landing summary] anchor={anchor_term_sel!r} stage2_process_mode={stage2_mode!r} "
                f"candidate_generated_count=0 landing_kept_count=0 landing_state='empty_landing' "
                f"landing_reason='no_candidates_generated'"
            )
        return empty_out

    primary_expandable: List["Stage2ACandidate"] = []
    primary_support_seed: List["Stage2ACandidate"] = []
    primary_support_keep: List["Stage2ACandidate"] = []
    risky_keep: List["Stage2ACandidate"] = []
    rejected: List["Stage2ACandidate"] = []

    anchor_term_sel = getattr(anchor, "anchor", "") or ""

    def _rank_key(x: "Stage2ACandidate") -> float:
        return float(getattr(x, "composite_rank_score", 0) or getattr(x, "primary_score", 0) or 0.0)

    work_rows: List[Dict[str, Any]] = []

    for c in candidates:
        primary_score = float(getattr(c, "primary_score", 0.0) or getattr(c, "composite_rank_score", 0) or 0.0)
        retain_mode = getattr(c, "retain_mode", "normal") or "normal"

        feat = _get_stage2a_feat(c)
        flags = _stage2a_rule_flags(c, feat)
        obviously_bad_branch = _is_obviously_bad_branch_stage2a(c, feat)
        drift_bad = flags["drift_bad"]
        poly_bad = flags["poly_bad"]
        object_bad = flags["object_bad"]
        ctx_ok = flags["ctx_ok"]

        # --------------------------------------------------
        # 0. 先处理真正的硬拒绝（仅 retain_mode == "reject"）
        #    低 primary_score 不再在此 reject，交给 judge_primary_and_expandability()
        #    判 primary_keep_no_expand / reject（避免弱主线技术词过早进 rejected）。
        #    STAGE2A_WEAK_KEEP_MIN 仅作文档阈值对齐，见模块常量注释。
        # --------------------------------------------------
        if retain_mode == "reject":
            c.primary_bucket = "reject"
            c.survive_primary = False
            c.can_expand = False
            c.can_expand_from_2a = False
            c.fallback_primary = False
            c.reject_reason = "retain_mode_reject"
            setattr(c, "bucket_reason", "retain_mode_reject")
            setattr(c, "mainline_block_reason", "retain_mode_reject")
            rejected.append(c)
            continue

        if obviously_bad_branch or drift_bad:
            c.primary_bucket = "reject"
            c.can_expand = False
            c.can_expand_from_2a = False
            c.fallback_primary = False
            c.reject_reason = "obviously_bad_branch" if obviously_bad_branch else "semantic_drift_branch"
            setattr(c, "bucket_reason", c.reject_reason)
            setattr(c, "mainline_block_reason", c.reject_reason)
            rejected.append(c)
            continue

        if poly_bad and not ctx_ok:
            c.primary_bucket = "reject"
            c.can_expand = False
            c.can_expand_from_2a = False
            c.fallback_primary = False
            c.reject_reason = "polysemous_no_context"
            setattr(c, "bucket_reason", "polysemous_no_context")
            setattr(c, "mainline_block_reason", "polysemous_no_context")
            rejected.append(c)
            continue

        # --------------------------------------------------
        # 1. primary / expand 解耦判定
        # --------------------------------------------------
        bucket, reason, p_snap = judge_primary_and_expandability(c, anchor_term=anchor_term_sel)

        # ------------------------------------------------------------------
        # final bucket reconcile：仅压制「conditioned-only / 高风险 / 上下文虚」，
        # 对 **similar_to + primary_ok + 真实 ctx** 的强/次主线放行或从 sk 救回，避免 Stage2B 入口被 sk 掐死。
        # judge 已定 axis 初桶；此处用结构化证据二次校正，与 README「 reconcile 层」一致。
        # ------------------------------------------------------------------
        _ps_static = float(p_snap.get("static_sim", 0.0) or 0.0)
        _ps_ctx = float(p_snap.get("ctx_sim", 0.0) or 0.0)
        _ps_drop = float(p_snap.get("ctx_drop", 1.0) or 1.0)
        _fam = float(p_snap.get("family_match", 0.0) or 0.0)
        if _fam <= 0.0:
            _fam = float(getattr(c, "family_match", 0.0) or getattr(c, "anchor_identity_score", 0.0) or 0.0)
        _aid = float(p_snap.get("anchor_identity_score", 0.0) or 0.0)
        if _aid <= 0.0:
            _aid = float(getattr(c, "anchor_identity_score", 0.0) or 0.0)
        _jd = float(p_snap.get("jd_align", 0.0) or getattr(c, "jd_align", 0.0) or 0.0)
        _gen = float(p_snap.get("generic_risk", getattr(c, "generic_risk", 0.0)) or 0.0)
        _poly = float(p_snap.get("polysemy_risk", getattr(c, "polysemy_risk", 0.0)) or 0.0)
        _obj = float(p_snap.get("object_like_risk", getattr(c, "object_like_risk", 0.0)) or 0.0)
        _dual = bool(p_snap.get("dual_support", False))
        _multi = bool(p_snap.get("multi_anchor_ok", False))
        _ctx_path = bool(p_snap.get("context_path_ok", False))
        _sem_ok = bool(p_snap.get("semantic_ok", False))
        _primary_ok = bool(p_snap.get("primary_ok", False))
        _src = _stage2a_source_type_set(c)
        _has_sim = "similar_to" in _src
        _has_real_ctx = bool(_ps_ctx > 0.0)
        _cond_only = _stage2a_is_conditioned_only_for_seed(c)

        strong_mainline_path_a = (
            _has_sim
            and _sem_ok
            and _ctx_path
            and _dual
            and _has_real_ctx
            and _ps_drop <= 0.065
            and _gen <= 0.50
            and _poly <= 0.26
            and _obj <= 0.20
        )
        strong_mainline_path_b = (
            _has_sim
            and _sem_ok
            and _primary_ok
            and _aid >= 0.55
            and _jd >= 0.50
            and _fam >= 0.40
            and _ps_drop <= 0.065
            and _gen <= 0.50
            and _poly <= 0.26
            and _obj <= 0.20
        )
        strong_mainline_path_c = (
            _has_sim
            and _sem_ok
            and _primary_ok
            and _multi
            and _aid >= 0.50
            and _jd >= 0.50
            and _gen <= 0.45
            and _poly <= 0.24
            and _obj <= 0.20
        )
        strong_expandable_ok = bool(
            strong_mainline_path_a or strong_mainline_path_b or strong_mainline_path_c
        )
        weak_seed_ok = (
            _has_sim
            and _sem_ok
            and _primary_ok
            and _has_real_ctx
            and _ps_drop <= 0.08
            and _fam >= 0.36
            and _gen <= 0.58
            and _poly <= 0.30
            and _obj <= 0.25
        )
        keep_no_expand_only = (
            _cond_only
            or _gen > 0.58
            or _poly > 0.30
            or _obj > 0.25
            or not _has_real_ctx
        )
        # Stage2B strong 同源（check_seed_eligibility）；维持/升格 primary_expandable 须同阈，避免 strong_seed_axis_low
        _mlp = float(getattr(c, "mainline_pref_score", 0.0) or 0.0)
        axis_consistency_seed = (
            0.35 * _aid
            + 0.30 * _fam
            + 0.20 * _jd
            + 0.15 * max(0.0, _mlp)
        )
        p_snap["axis_consistency_seed"] = axis_consistency_seed
        _strong_axis_ok = bool(axis_consistency_seed >= SEED_AXIS_CONSISTENCY_STRONG_MIN)

        # --------------------------------------------------
        # Stage2A local ranking（组内主位偏好）：
        # - 只做单 anchor 内的相对排序/消歧（local disambiguation）
        # - 不引入 Stage3 的 collective/global 裁决
        # - 目标：让更自然、更贴 JD 主轴、更稳的学术术语更容易成为 top1/top2
        # --------------------------------------------------
        _ctx_cont = float(getattr(c, "context_continuity", 0.0) or 0.0)
        _pref_snap = _stage2a_local_primary_preference_snapshot(
            anchor_term=anchor_term_sel,
            cand=c,
            axis_consistency_seed=axis_consistency_seed,
            mainline_pref_score=_mlp,
            jd_align=_jd,
            family_match=_fam,
            ctx_sim=_ps_ctx,
            ctx_drop=_ps_drop,
            context_continuity=_ctx_cont,
            conditioned_only=_cond_only,
            generic_risk=_gen,
            polysemy_risk=_poly,
            object_like_risk=_obj,
        )
        local_pref = float(_pref_snap.get("local_pref", 0.0) or 0.0)
        p_snap["local_primary_pref"] = local_pref

        work_rows.append({
            "c": c,
            "pre_bucket": bucket,
            "pre_reason": reason,
            "p_snap": p_snap,
            "axis_consistency_seed": axis_consistency_seed,
            "local_primary_pref": local_pref,
            "_local_pref_snap": _pref_snap,
            "strong_expandable_ok": strong_expandable_ok,
            "weak_seed_ok": weak_seed_ok,
            "keep_no_expand_only": keep_no_expand_only,
            "_strong_axis_ok": _strong_axis_ok,
            "_primary_ok": _primary_ok,
            "_has_sim": _has_sim,
            "_cond_only": _cond_only,
            "_ps_drop": _ps_drop,
            "_fam": _fam,
            "_gen": _gen,
            "_poly": _poly,
            "_obj": _obj,
            "_mlp": _mlp,
            "_jd": _jd,
            "_ps_ctx": _ps_ctx,
            "_ctx_cont": _ctx_cont,
        })

    sorted_w = sorted(
        work_rows,
        # local preference first；其后用 axis_seed/mainline/jd/ctx_cont 稳定打平
        key=lambda ww: (
            ww.get("local_primary_pref", 0.0),
            ww["axis_consistency_seed"],
            ww["_mlp"],
            ww["_jd"],
            ww.get("_ctx_cont", 0.0),
            ww["_ps_ctx"],
        ),
        reverse=True,
    )
    nw = len(sorted_w)
    for i, ww in enumerate(sorted_w):
        ww["group_rank"] = i + 1
    top0_axis = sorted_w[0]["axis_consistency_seed"] if sorted_w else 0.0
    top1_axis_val = sorted_w[1]["axis_consistency_seed"] if len(sorted_w) > 1 else None
    axis_gap_ok = top1_axis_val is None or (top0_axis - top1_axis_val >= STAGE2A_PROMOTE_MIN_AXIS_GAP)
    has_judge_expandable_stable = any(
        ww["pre_bucket"] == "primary_expandable" and ww["strong_expandable_ok"] and ww["_strong_axis_ok"]
        for ww in work_rows
    )

    if LABEL_EXPANSION_DEBUG and sorted_w:
        print(
            f"\n[Stage2A group rank] anchor={anchor_term_sel!r} n={nw} "
            f"axis_gap_ok={axis_gap_ok} gap_min={STAGE2A_PROMOTE_MIN_AXIS_GAP:.3f} "
            f"judge_exp_stable={has_judge_expandable_stable}"
        )
        for ww in sorted_w:
            cc = ww["c"]
            print(
                f"  rank={ww['group_rank']}/{nw} term={(getattr(cc, 'term', '') or '')!r} "
                f"pre_bucket={ww['pre_bucket']!r} axis_seed={ww['axis_consistency_seed']:.3f} "
                f"local_pref={ww.get('local_primary_pref', 0.0):.3f} "
                f"mainline_pref={ww['_mlp']:.3f} jd={ww['_jd']:.3f}"
            )

        # 轻量“组内主位偏好审计”：仅 noisy debug 打印 topK，便于验证 local ranking 是否按论文式信号在起作用
        if STAGE2_NOISY_DEBUG:
            topk = min(5, len(sorted_w))
            print(f"[Stage2A local primary pref audit] anchor={anchor_term_sel!r} topK={topk}")
            for i in range(topk):
                ww = sorted_w[i]
                cc = ww["c"]
                snap = ww.get("_local_pref_snap", {}) or {}
                rp = snap.get("risk_parts", {}) or {}
                print(
                    f"  top{i+1} term={(getattr(cc, 'term', '') or '')!r} "
                    f"local_pref={float(snap.get('local_pref', 0.0) or 0.0):.3f} "
                    f"mainline={float(snap.get('mainline', 0.0) or 0.0):.3f} "
                    f"ctxQ={float(snap.get('ctx_quality', 0.0) or 0.0):.3f} "
                    f"jd={float(snap.get('jd', 0.0) or 0.0):.3f} "
                    f"family={float(snap.get('family', 0.0) or 0.0):.3f} "
                    f"axis_seed={float(snap.get('axis_seed', 0.0) or 0.0):.3f} "
                    f"cond_only={bool(snap.get('cond_only', False))} "
                    f"natural={float(snap.get('natural', 0.0) or 0.0):.3f} "
                    f"risk={float(snap.get('risk', 0.0) or 0.0):.3f} "
                    f"(g={float(rp.get('generic', 0.0) or 0.0):.2f},"
                    f"p={float(rp.get('poly', 0.0) or 0.0):.2f},"
                    f"o={float(rp.get('object', 0.0) or 0.0):.2f}) "
                    f"why=mainline+ctx+jd+natural - risk - cond_only"
                )

    for w in work_rows:
        c = w["c"]
        p_snap = w["p_snap"]
        axis_consistency_seed = w["axis_consistency_seed"]
        strong_expandable_ok = w["strong_expandable_ok"]
        weak_seed_ok = w["weak_seed_ok"]
        keep_no_expand_only = w["keep_no_expand_only"]
        _strong_axis_ok = w["_strong_axis_ok"]
        _primary_ok = w["_primary_ok"]
        _has_sim = w["_has_sim"]
        _cond_only = w["_cond_only"]
        _ps_drop = w["_ps_drop"]
        _fam = w["_fam"]
        _gen = w["_gen"]
        _poly = w["_poly"]
        _obj = w["_obj"]

        bucket = w["pre_bucket"]
        reason = w["pre_reason"]
        pre_bucket = bucket
        pre_reason = reason

        if bucket not in ("reject", "risky_keep"):
            if bucket == "primary_expandable":
                if strong_expandable_ok and _strong_axis_ok:
                    bucket, reason = "primary_expandable", "axis_strong_mainline"
                    p_snap["can_expand_from_2a"] = True
                elif weak_seed_ok:
                    bucket, reason = "primary_support_seed", "axis_neighbor_weak_seed"
                    p_snap["can_expand_from_2a"] = True
                elif strong_expandable_ok and not _strong_axis_ok:
                    bucket, reason = "primary_support_keep", "expandable_demote_strong_seed_axis_low"
                    p_snap["can_expand_from_2a"] = False
                else:
                    bucket, reason = "primary_support_keep", "expandable_demote_axis_not_hard_enough"
                    p_snap["can_expand_from_2a"] = False
            elif bucket == "primary_support_seed":
                if strong_expandable_ok and _strong_axis_ok:
                    _p_ok, _p_blk = _stage2a_promote_strong_allowed(
                        w, axis_gap_ok=axis_gap_ok, has_judge_expandable_stable=has_judge_expandable_stable
                    )
                    if _p_ok:
                        bucket, reason = "primary_expandable", "seed_promote_to_expandable"
                        p_snap["can_expand_from_2a"] = True
                    elif weak_seed_ok:
                        bucket, reason = "primary_support_seed", "axis_neighbor_weak_seed"
                        p_snap["can_expand_from_2a"] = True
                    else:
                        _rs = f"seed_promote_blocked_{_p_blk}" if _p_blk else "seed_promote_blocked_group_gate"
                        bucket, reason = "primary_support_keep", _rs
                        p_snap["can_expand_from_2a"] = False
                elif weak_seed_ok:
                    bucket, reason = "primary_support_seed", "axis_neighbor_weak_seed"
                    p_snap["can_expand_from_2a"] = True
                elif strong_expandable_ok and not _strong_axis_ok:
                    bucket, reason = "primary_support_keep", "seed_demote_strong_seed_axis_low"
                    p_snap["can_expand_from_2a"] = False
                else:
                    bucket, reason = "primary_support_keep", "weak_seed_demote_axis_weak"
                    p_snap["can_expand_from_2a"] = False
            elif bucket == "primary_support_keep":
                if strong_expandable_ok and not keep_no_expand_only and _strong_axis_ok:
                    _p_ok, _p_blk = _stage2a_promote_strong_allowed(
                        w, axis_gap_ok=axis_gap_ok, has_judge_expandable_stable=has_judge_expandable_stable
                    )
                    if _p_ok:
                        bucket, reason = "primary_expandable", "keep_promote_strong_mainline"
                        p_snap["can_expand_from_2a"] = True
                    elif weak_seed_ok and not keep_no_expand_only:
                        bucket, reason = "primary_support_seed", "keep_promote_weak_seed"
                        p_snap["can_expand_from_2a"] = True
                    else:
                        _rs = f"keep_promote_blocked_{_p_blk}" if _p_blk else "keep_promote_blocked_group_gate"
                        bucket, reason = "primary_support_keep", _rs
                        p_snap["can_expand_from_2a"] = False
                elif weak_seed_ok and not keep_no_expand_only:
                    bucket, reason = "primary_support_seed", "keep_promote_weak_seed"
                    p_snap["can_expand_from_2a"] = True
                else:
                    p_snap["can_expand_from_2a"] = False

        if LABEL_EXPANSION_DEBUG and STAGE2_NOISY_DEBUG and w["pre_bucket"] in ("primary_support_keep", "primary_support_seed"):
            if strong_expandable_ok and _strong_axis_ok:
                _p_ok, _p_blk = _stage2a_promote_strong_allowed(
                    w, axis_gap_ok=axis_gap_ok, has_judge_expandable_stable=has_judge_expandable_stable
                )
                _agap = (top0_axis - top1_axis_val) if top1_axis_val is not None else 1.0
                print(
                    f"[Stage2A promote audit] anchor={anchor_term_sel!r} term={(getattr(c, 'term', '') or '')!r} "
                    f"pre_bucket={w['pre_bucket']!r} axis_seed={axis_consistency_seed:.3f} "
                    f"group_rank={w['group_rank']}/{len(work_rows)} "
                    f"top1_term={(getattr(sorted_w[0]['c'], 'term', '') or '')!r} "
                    f"axis_gap={_agap:.3f} promote_allowed={_p_ok} promote_block={_p_blk!r} "
                    f"final_bucket={bucket!r} reason={reason!r}"
                )

        if pre_bucket != bucket and LABEL_EXPANSION_DEBUG:
            print(
                f"[Stage2A final bucket reconcile] "
                f"anchor={anchor_term_sel!r} term={(getattr(c, 'term', '') or '')!r} "
                f"pre_bucket={pre_bucket!r} pre_reason={pre_reason!r} "
                f"final_bucket={bucket!r} reason={reason!r} "
                f"axis_consistency_seed={axis_consistency_seed:.3f} "
                f"group_rank={w['group_rank']}/{len(work_rows)} "
                f"strong_axis_min={SEED_AXIS_CONSISTENCY_STRONG_MIN:.2f}"
            )
            if STAGE2_NOISY_DEBUG and _is_stage2a_focus_case(anchor_term_sel, getattr(c, "term", "")):
                _debug_stage2a_focus(anchor_term_sel, getattr(c, "term", ""), bucket, reason, p_snap)

        static_sim, ctx_sim, ctx_drop, _best = _stage2a_static_ctx_for_primary_expand_split(c)

        if bucket == "primary_support_keep" and _primary_ok and LABEL_EXPANSION_DEBUG:
            print(
                f"[Stage2A expand deny summary] "
                f"anchor={anchor_term_sel!r} term={(getattr(c, 'term', '') or '')!r} "
                f"deny_main={reason!r} "
                f"axis_consistency_seed={axis_consistency_seed:.3f} (min_strong={SEED_AXIS_CONSISTENCY_STRONG_MIN:.2f}) "
                f"group_rank={w['group_rank']}/{len(work_rows)} "
                f"has_similar_to={_has_sim} conditioned_only={_cond_only} "
                f"ctx_drop={_ps_drop:.3f} family={_fam:.3f} "
                f"generic={_gen:.3f} poly={_poly:.3f} obj={_obj:.3f}"
            )
        setattr(c, "is_good_mainline", bucket not in ("reject",))
        setattr(c, "_stage2a_primary_snap", p_snap)

        if LABEL_EXPANSION_DEBUG and STAGE2_RULING_DEBUG:
            _axis_c = float(p_snap.get("axis_consistency", 0.0) or 0.0)
            _eff = float(p_snap.get("effective_mainline", 0.0) or 0.0)
            _cond_only = bool(p_snap.get("conditioned_only", False))
            term_axis = (getattr(c, "term", None) or "")[:40]
            _acs = float(p_snap.get("axis_consistency_seed", 0.0) or 0.0)
            print(
                f"[Stage2A axis audit] term={term_axis!r} "
                f"axis={_axis_c:.3f} effective={_eff:.3f} "
                f"axis_consistency_seed={_acs:.3f} "
                f"conditioned_only={_cond_only} "
                f"final_bucket={bucket!r} reason={reason!r}"
            )
            _print_stage2a_primary_factors(anchor, c, bucket, reason, p_snap)

        # RULING 已打 axis audit + primary factors 时不再重复本行，降噪
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and not STAGE2_RULING_DEBUG:
            anchor_term = getattr(anchor, "anchor", "") or ""
            term_short = (getattr(c, "term", None) or "")[:28]
            print(
                f"[Stage2A primary/expand split] anchor={anchor_term!r} term={term_short!r} "
                f"bucket={bucket!r} reason={reason!r} static={static_sim:.3f} ctx={ctx_sim:.3f} ctx_drop={ctx_drop:.3f}"
            )

        if bucket == "reject":
            c.primary_bucket = "reject"
            c.survive_primary = False
            c.can_expand = False
            c.can_expand_from_2a = False
            c.fallback_primary = False
            c.reject_reason = reason
            setattr(c, "bucket_reason", reason)
            setattr(c, "mainline_block_reason", reason)
            rejected.append(c)
            _debug_stage2a_commit(anchor_term_sel, c)
            continue

        if bucket == "primary_expandable":
            c.survive_primary = True
            c.can_expand = True
            c.can_expand_from_2a = True
            c.fallback_primary = False
            c.primary_bucket = "primary_expandable"
            c.stage2b_seed_tier = "strong"
            c.role_in_anchor = "mainline"
            c.role = "mainline"
            c.admission_reason = reason
            setattr(c, "bucket_reason", reason)
            primary_expandable.append(c)
            _debug_stage2a_commit(anchor_term_sel, c)
            continue

        if bucket == "primary_support_seed":
            c.survive_primary = True
            c.can_expand = True
            c.can_expand_from_2a = bool(p_snap.get("can_expand_from_2a", True))
            c.fallback_primary = False
            c.primary_bucket = "primary_support_seed"
            # stage2b_seed_tier 在 _finalize_stage2b_seed_tiers 中定为 weak 或降级为 support_keep
            c.role_in_anchor = "mainline"
            c.role = "mainline"
            c.admission_reason = reason
            setattr(c, "bucket_reason", reason)
            primary_support_seed.append(c)
            _debug_stage2a_commit(anchor_term_sel, c)
            continue

        if bucket == "primary_support_keep":
            c.survive_primary = True
            c.can_expand = False
            c.can_expand_from_2a = False
            c.fallback_primary = False
            c.primary_bucket = "primary_support_keep"
            c.stage2b_seed_tier = "none"
            c.role_in_anchor = "side"
            c.role = "side"
            c.admission_reason = reason
            setattr(c, "bucket_reason", reason)
            primary_support_keep.append(c)
            _debug_stage2a_commit(anchor_term_sel, c)
            continue

        if bucket == "risky_keep":
            c.survive_primary = True
            c.can_expand = False
            c.can_expand_from_2a = False
            c.fallback_primary = False
            c.primary_bucket = "risky_keep"
            c.stage2b_seed_tier = "none"
            c.retain_mode = "weak_retain"
            c.role_in_anchor = "side"
            c.role = "side"
            c.admission_reason = reason
            setattr(c, "bucket_reason", reason)
            risky_keep.append(c)
            _debug_stage2a_commit(anchor_term_sel, c)
            continue

        # 未知桶：保守落入高风险保留
        c.survive_primary = True
        c.can_expand = False
        c.can_expand_from_2a = False
        c.primary_bucket = "risky_keep"
        c.stage2b_seed_tier = "none"
        c.retain_mode = "weak_retain"
        c.role_in_anchor = "side"
        c.role = "side"
        c.admission_reason = reason
        setattr(c, "bucket_reason", reason)
        risky_keep.append(c)
        _debug_stage2a_commit(anchor_term_sel, c)

    primary_expandable.sort(key=_rank_key, reverse=True)
    primary_support_seed.sort(key=_rank_key, reverse=True)
    primary_support_keep.sort(key=_rank_key, reverse=True)
    risky_keep.sort(key=_rank_key, reverse=True)

    # 无 primary_expandable：保留「可重审」极小家族，而不是 sd+sk+rk 合并后只留同一 tid 的一条（等于 2A 越权终审）。
    if not primary_expandable and (primary_support_seed or primary_support_keep or risky_keep):
        n_sd_b = len(primary_support_seed)
        n_sk_b = len(primary_support_keep)
        n_rk_b = len(risky_keep)
        top_sd_b = [getattr(x, "term", "") for x in primary_support_seed[:2]]
        top_sk_b = [getattr(x, "term", "") for x in primary_support_keep[:2]]
        top_rk_b = [getattr(x, "term", "") for x in risky_keep[:2]]

        primary_support_seed.sort(key=_rank_key, reverse=True)
        primary_support_keep.sort(key=_rank_key, reverse=True)
        risky_keep.sort(key=_rank_key, reverse=True)

        # 无 expandable：按槽限额保留多候选（非单胜者）；rk 在 sd/sk 存在时仍可留至多 1 条弱证据
        _MAX_SD = 2
        _MAX_SK = 2
        _MAX_RK = 1
        kept_seed = primary_support_seed[:_MAX_SD]
        kept_keep = primary_support_keep[:_MAX_SK]
        if kept_seed or kept_keep:
            risky_sorted = sorted(risky_keep, key=_rank_key, reverse=True)
            kept_risky = risky_sorted[:_MAX_RK]
            policy = f"quota_seed<={_MAX_SD}_sk<={_MAX_SK}_rk<={_MAX_RK}_when_sd_or_sk"
            why_risky = "retain_top_risky_as_evidence" if kept_risky else ""
        else:
            kept_risky = risky_keep[:1]
            policy = "risky_only_fallback_1"
            why_risky = "no_support_seed_and_no_support_keep" if kept_risky else ""

        primary_support_seed = kept_seed
        primary_support_keep = kept_keep
        risky_keep = kept_risky

        if LABEL_EXPANSION_DEBUG:
            top_sd_a = [getattr(x, "term", "") for x in primary_support_seed[:3]]
            top_sk_a = [getattr(x, "term", "") for x in primary_support_keep[:3]]
            top_rk_a = [getattr(x, "term", "") for x in risky_keep[:2]]
            print(
                f"[Stage2A no-expandable shrink audit] anchor={anchor_term_sel!r}\n"
                f"  before: exp=0 sd={n_sd_b} {top_sd_b!r} | sk={n_sk_b} {top_sk_b!r} | rk={n_rk_b} {top_rk_b!r}\n"
                f"  policy: {policy}\n"
                f"  why_keep_risky_fallback: {why_risky!r}\n"
                f"  after : sd={len(primary_support_seed)} {top_sd_a!r} | sk={len(primary_support_keep)} {top_sk_a!r} "
                f"| rk={len(risky_keep)} {top_rk_a!r}"
            )

    # --------------------------------------------------
    # 2. 高质量锚点空组保线：只保 keep_no_expand，不允许扩散
    #    目的：当一个强锚点本组（primary_expandable / primary_keep_no_expand）为空，
    #    从“非硬坏分支”里按组内相对信号救回 1 条 primary_keep_no_expand，
    #    但不会拿到 Stage2B seed（can_expand_from_2a=False，fallback_primary=True）。
    # --------------------------------------------------
    if not primary_expandable and not primary_support_seed and not primary_support_keep and not risky_keep:
        anchor_strength = float(getattr(anchor, "source_weight", 1.0) or 1.0)

        fallback_pool: List[Tuple[float, Stage2ACandidate]] = []
        for c in candidates:
            feat = _get_stage2a_feat(c)
            flags = _stage2a_rule_flags(c, feat)
            obviously_bad_branch = _is_obviously_bad_branch_stage2a(c, feat)
            drift_bad = flags["drift_bad"]
            poly_bad = flags["poly_bad"]
            ctx_ok = flags["ctx_ok"]

            # 不能从真正 hard reject / obviously bad 里救
            retain_mode = (getattr(c, "retain_mode", "") or "").strip().lower()
            if retain_mode == "reject":
                continue
            if obviously_bad_branch:
                continue
            if drift_bad:
                continue
            if poly_bad and not ctx_ok:
                continue

            # 只允许“弱但还像主线”的技术词：用结构/证据信号救线，而不是词表
            family_match = float(getattr(c, "family_match", 0.0) or 0.0)
            jd_align = float(getattr(c, "jd_align", 0.0) or 0.0)
            context_cont = float(getattr(c, "context_continuity", 0.0) or 0.0)
            hier = float(getattr(c, "hierarchy_consistency", 0.0) or 0.0)
            generic_risk = float(getattr(c, "generic_risk", 0.0) or 0.0)
            object_risk = float(getattr(c, "object_like_risk", 0.0) or 0.0)
            poly_risk = float(getattr(c, "polysemy_risk", 0.0) or 0.0)

            semantic_score = float(getattr(c, "semantic_score", 0.0) or 0.0)
            rank_score = float(
                getattr(c, "composite_rank_score", 0.0)
                or getattr(c, "primary_score", 0.0)
                or 0.0
            )

            # “结构保底”弱保线条件：
            # - 不是硬坏分支（已在上游过滤）
            # - 与锚点/JD/上下文/层级结构仍有一定一致性
            # - 风险不过高（generic/object/poly）
            weak_keep_ok = (
                family_match >= 0.26
                and jd_align >= 0.58
                and context_cont >= 0.30
                and hier >= 0.20
                and generic_risk <= 0.72
                and object_risk <= 0.40
                and poly_risk <= 0.62
            )

            # --- fallback debug: 每个候选为何通过/不通过（仅在 verbose debug 打开时）---
            if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG:
                print(
                    f"[Stage2A fallback cand] anchor={anchor_term_sel!r} term={getattr(c, 'term', '')!r} "
                    f"family={family_match:.3f} jd={jd_align:.3f} context={context_cont:.3f} hier={hier:.3f} "
                    f"gen={generic_risk:.3f} obj={object_risk:.3f} poly={poly_risk:.3f} "
                    f"weak_ok={weak_keep_ok}"
                )
            if not weak_keep_ok:
                continue

            # 组内相对排序：更像主线、更少风险、更贴 JD 的优先
            rescue_score = (
                0.34 * family_match
                + 0.24 * jd_align
                + 0.18 * context_cont
                + 0.10 * hier
                + 0.08 * semantic_score
                + 0.06 * rank_score
                - 0.08 * generic_risk
                - 0.06 * object_risk
                - 0.06 * poly_risk
            )
            fallback_pool.append((rescue_score, c))

        # 只对“强锚点”启用（防止弱锚点用救线把噪声抬上来）
        trigger = anchor_strength >= 0.90 and bool(fallback_pool)
        # --- fallback debug: 是否触发落地 append ---
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG:
            print(
                f"[Stage2A fallback debug] anchor={getattr(anchor, 'anchor', '')!r} "
                f"source_weight={anchor_strength:.3f} pool_size={len(fallback_pool)} "
                f"trigger={trigger}"
            )
        if trigger:
            fallback_pool.sort(key=lambda x: x[0], reverse=True)
            best = fallback_pool[0][1]

            best.primary_bucket = "primary_support_keep"
            best.stage2b_seed_tier = "none"
            best.survive_primary = True
            best.can_expand = False
            best.can_expand_from_2a = False
            best.fallback_primary = True
            best.admission_reason = "anchor_core_fallback"
            # 让 Stage3 统计到 mainline_hits：fallback 词在“身份上像主线救线”，但仍禁止扩散
            best.role_in_anchor = "mainline"
            best.role = "mainline"
            setattr(best, "bucket_reason", "anchor_core_fallback")
            setattr(best, "mainline_block_reason", "anchor_core_fallback")
            # --- Stage3 / 审计对齐字段：不依赖词表，只用于传递“弱主线救线身份” ---
            setattr(best, "is_grounded_fallback", True)
            setattr(best, "weak_mainline_support", True)
            setattr(best, "mainline_candidate", True)
            setattr(best, "mainline_support_hits", max(1, int(getattr(best, "mainline_support_hits", 0) or 0)))
            # fallback 可保留，但永不作为可扩散 seed
            setattr(best, "can_expand_from_2a", False)

            primary_support_keep.append(best)

    # --------------------------------------------------
    # 3. conditioned_vec 来源配额：每锚最多 1 expandable + 1 keep
    #    仅做来源裁剪，不做词面黑白名单。
    # --------------------------------------------------
    def _is_conditioned_source(cand: "Stage2ACandidate") -> bool:
        st = (getattr(cand, "source_type", None) or getattr(cand, "source", "") or "").strip().lower()
        return st == "conditioned_vec"

    def _conditioned_priority_key(cand: "Stage2ACandidate") -> Tuple[float, ...]:
        # dual_support/mainline_support/family 更优，风险更低更优；最后用 rank_score 打破平局
        dual_support = 1.0 if bool(getattr(cand, "dual_support", False)) else 0.0
        mainline_support_hits = float(getattr(cand, "mainline_support_hits", 0) or 0.0)
        family_match = float(getattr(cand, "family_match", 0.0) or 0.0)
        generic_risk = float(getattr(cand, "generic_risk", 0.0) or 0.0)
        object_risk = float(getattr(cand, "object_like_risk", 0.0) or 0.0)
        poly_risk = float(getattr(cand, "polysemy_risk", 0.0) or 0.0)
        rank_score = float(
            getattr(cand, "composite_rank_score", 0.0)
            or getattr(cand, "primary_score", 0.0)
            or 0.0
        )
        return (
            dual_support,
            mainline_support_hits,
            family_match,
            -generic_risk,
            -object_risk,
            -poly_risk,
            rank_score,
        )

    def _apply_conditioned_quota(cands: List["Stage2ACandidate"], cap: int) -> List["Stage2ACandidate"]:
        if not cands:
            return cands
        conditioned = [x for x in cands if _is_conditioned_source(x)]
        if len(conditioned) <= cap:
            return cands
        conditioned_sorted = sorted(conditioned, key=_conditioned_priority_key, reverse=True)
        keep_ids = {id(x) for x in conditioned_sorted[:cap]}
        out: List["Stage2ACandidate"] = []
        cond_used = 0
        for x in cands:
            if _is_conditioned_source(x):
                if id(x) in keep_ids:
                    out.append(x)
                    cond_used += 1
                continue
            out.append(x)
        # 兜底：理论不应触发；若顺序过滤导致少于 cap，则按优先级补齐
        if cond_used < cap:
            for x in conditioned_sorted:
                if id(x) in keep_ids and x not in out:
                    out.append(x)
                    cond_used += 1
                if cond_used >= cap:
                    break
        return out

    # 配额与 finalize 前快照：与 judge 后写在 c.primary_bucket 上的「预落地桶」对比，解释 Focus=support_seed 但最终 select=support_keep
    _pre_reconcile_bucket_by_id: Dict[int, str] = {}
    for c in primary_expandable + primary_support_seed + primary_support_keep + risky_keep:
        _pre_reconcile_bucket_by_id[id(c)] = (getattr(c, "primary_bucket", None) or "").strip()

    # conditioned_vec 来源：提高 cap，支持「小规模多 landing」共存；rk 仍最严（最终主线交 Stage3）
    primary_expandable = _apply_conditioned_quota(primary_expandable, cap=2)
    primary_support_seed = _apply_conditioned_quota(primary_support_seed, cap=2)
    primary_support_keep = _apply_conditioned_quota(primary_support_keep, cap=2)
    risky_keep = _apply_conditioned_quota(risky_keep, cap=1)

    # --------------------------------------------------
    # 1.5 受控的 weak-seed fallback（为 Stage2B 保留极小扩展出口；不改主排序公式）
    # 条件（极保守）：
    # - 本锚无 primary_expandable 且 support_seed=0（否则不需要 fallback）
    # - 仅从 primary_support_keep 提升最多 1 条为 primary_support_seed
    # - 必须 rank 靠前（优先 top1），且非 conditioned-only，且风险不过高，且具备较稳定局部证据（prefer similar_to）
    # - 永不从 risky_keep 提升
    #
    # 目的：让少量合理 anchor 出现 seed_candidates>0，便于 Stage2B support expansion 启动；
    # 仍保持整体门控保守，避免大放水。
    # --------------------------------------------------
    if (not primary_expandable) and (not primary_support_seed) and primary_support_keep:
        cand0 = primary_support_keep[0]
        pb0 = (getattr(cand0, "primary_bucket", "") or "").strip()
        src0 = _stage2a_source_type_set(cand0)
        has_sim0 = "similar_to" in src0 or "family_landing" in src0
        cond_only0 = _stage2a_is_conditioned_only_for_seed(cand0)
        gen0 = float(getattr(cand0, "generic_risk", 0.0) or 0.0)
        poly0 = float(getattr(cand0, "polysemy_risk", 0.0) or 0.0)
        obj0 = float(getattr(cand0, "object_like_risk", 0.0) or 0.0)
        fam0 = float(getattr(cand0, "family_match", 0.0) or getattr(cand0, "anchor_identity_score", 0.0) or 0.0)
        jd0 = float(getattr(cand0, "jd_align", 0.0) or 0.0)
        ctx0 = float(getattr(cand0, "context_continuity", 0.0) or 0.0)
        rk0 = getattr(cand0, "anchor_internal_rank", None)
        rk_ok = (rk0 is None) or (isinstance(rk0, int) and rk0 <= 1)

        promote_ok = bool(
            rk_ok
            and has_sim0
            and (not cond_only0)
            and gen0 <= 0.55
            and poly0 <= 0.28
            and obj0 <= 0.22
            and fam0 >= 0.34
            and jd0 >= 0.58
            and ctx0 >= 0.30
        )

        if promote_ok:
            # promote: support_keep -> support_seed（仅改变桶标签/扩展资格，不改排序公式）
            primary_support_keep = primary_support_keep[1:]
            primary_support_seed = [cand0] + primary_support_seed
            cand0.primary_bucket = "primary_support_seed"
            cand0.can_expand = True
            cand0.can_expand_from_2a = True
            cand0.suppress_seed = False
            setattr(cand0, "seed_fallback_promoted", True)
            setattr(cand0, "seed_fallback_from_bucket", pb0)
            setattr(cand0, "seed_fallback_reason", "no_expandable_and_no_seed_promote_keep_top1")
            if LABEL_EXPANSION_DEBUG:
                print(
                    f"[Stage2A weak-seed fallback] anchor={anchor_term_sel!r} triggered=True "
                    f"term={(getattr(cand0, 'term', '') or '')!r} "
                    f"from_bucket={pb0!r} -> to_bucket='primary_support_seed' "
                    f"evidence={{'has_sim':{has_sim0},'cond_only':{cond_only0},'rk_ok':{rk_ok},"
                    f"'gen':{gen0:.2f},'poly':{poly0:.2f},'obj':{obj0:.2f},"
                    f"'fam':{fam0:.2f},'jd':{jd0:.2f},'ctx':{ctx0:.2f}}}"
                )
        else:
            if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
                # 仅在“确实缺 expandable 且缺 seed”时打印一行未触发原因，避免刷屏
                reason_bits: List[str] = []
                if not rk_ok:
                    reason_bits.append("rank_not_top1")
                if not has_sim0:
                    reason_bits.append("no_similar_to_or_family_landing")
                if cond_only0:
                    reason_bits.append("conditioned_only")
                if gen0 > 0.55:
                    reason_bits.append("generic_risk_high")
                if poly0 > 0.28:
                    reason_bits.append("polysemy_risk_high")
                if obj0 > 0.22:
                    reason_bits.append("object_like_risk_high")
                if fam0 < 0.34:
                    reason_bits.append("family_match_low")
                if jd0 < 0.58:
                    reason_bits.append("jd_align_low")
                if ctx0 < 0.30:
                    reason_bits.append("context_continuity_low")
                why = ",".join(reason_bits) if reason_bits else "unknown"
                print(
                    f"[Stage2A weak-seed fallback] anchor={anchor_term_sel!r} triggered=False "
                    f"top_keep={(getattr(cand0, 'term', '') or '')!r} bucket={pb0!r} reason={why!r}"
                )

    primary_expandable, primary_support_seed, primary_support_keep, risky_keep = _finalize_stage2b_seed_tiers(
        primary_expandable, primary_support_seed, primary_support_keep, risky_keep
    )

    primary_keep_no_expand: List["Stage2ACandidate"] = (
        primary_support_seed + primary_support_keep + risky_keep
    )

    # --------------------------------------------------
    # 2. Stage2A landing 闭环：显式区分“有候选”与“已落地”，并把状态用于 seed 资格收口
    # 目的：减少“看起来有候选但不可落地”的锚点继续甩给后续阶段。
    # 注意：不改 Stage2A 主排序公式；仅做 anchor 级状态语义与极小的 post-closure 调整。
    # --------------------------------------------------
    kept_all: List["Stage2ACandidate"] = (
        list(primary_expandable) + list(primary_support_seed) + list(primary_support_keep) + list(risky_keep)
    )
    candidate_generated_count = len(candidates or [])
    landing_kept_count = len(kept_all)

    def _cand_src_set(c: "Stage2ACandidate") -> Set[str]:
        try:
            return _stage2a_source_type_set(c)
        except Exception:
            return set()

    kept_src_union: Set[str] = set()
    for _c0 in kept_all:
        kept_src_union |= _cand_src_set(_c0)
    has_normal_retrieval = ("similar_to" in kept_src_union) or ("conditioned_vec" in kept_src_union)
    family_only = (landing_kept_count > 0) and (not has_normal_retrieval) and ("family_landing" in kept_src_union)

    stage2_mode = (getattr(anchor, "stage2_process_mode", "") or "").strip()
    landing_state = "empty_landing"
    landing_reason = "no_kept_candidates"

    # main_strong：尽量形成“弱但正常”的落地（不等于抬主，只是不要长期停在全是 risky_keep 的模糊态）
    if landing_kept_count <= 0:
        landing_state = "empty_landing"
        landing_reason = "no_kept_candidates_after_select"
    elif family_only:
        landing_state = "fallback_only_landing"
        landing_reason = "family_fallback_only_no_normal_retrieval"
    else:
        if stage2_mode == "main_strong_process":
            if primary_expandable or primary_support_seed:
                landing_state = "good_landing"
                landing_reason = "has_seed_or_expandable"
            elif primary_support_keep:
                landing_state = "weak_landing"
                landing_reason = "support_keep_only"
            elif risky_keep:
                landing_state = "weak_landing"
                landing_reason = "risky_only_keep"
            else:
                landing_state = "empty_landing"
                landing_reason = "no_kept_candidates_after_select"
        else:
            # aux：有保留但不强认落点；无则 empty
            landing_state = "weak_landing" if landing_kept_count > 0 else "empty_landing"
            landing_reason = "aux_kept_some_evidence" if landing_kept_count > 0 else "aux_no_evidence"

    # main_strong 放行：当 normal retrieval 的 top1 足够贴锚且风险不过高、对齐充分时，赋予 good_landing 身份
    # 仅改变 landing_state/landing_reason（可解释），不改 Stage2A 主排序公式与候选生成逻辑。
    if stage2_mode == "main_strong_process" and (not family_only) and kept_all:
        def _top1_sim(c: "Stage2ACandidate") -> float:
            # 优先使用静态相似（similar_to），缺失则用 semantic_score/conditioned_sim 的可用值
            v = getattr(c, "semantic_score", None)
            if v is None:
                v = getattr(c, "surface_sim", None)
            if v is None:
                v = getattr(c, "conditioned_sim", None)
            try:
                return float(v or 0.0)
            except (TypeError, ValueError):
                return 0.0

        top1 = max(kept_all, key=_top1_sim)
        sim1 = _top1_sim(top1)
        gen1 = float(getattr(top1, "generic_risk", 0.0) or 0.0)
        poly1 = float(getattr(top1, "polysemy_risk", 0.0) or 0.0)
        obj1 = float(getattr(top1, "object_like_risk", 0.0) or 0.0)
        jd1 = float(getattr(top1, "jd_align", 0.0) or 0.0)
        fam1 = float(getattr(top1, "family_match", 0.0) or getattr(top1, "anchor_identity_score", 0.0) or 0.0)

        top1_ok = bool(
            sim1 >= 0.85
            and gen1 <= 0.55
            and poly1 <= 0.30
            and obj1 <= 0.25
            and (jd1 >= 0.58 or fam1 >= 0.34)
        )
        if top1_ok:
            landing_state = "good_landing"
            landing_reason = "main_strong_normal_retrieval_top1_ok"

    # main_strong 的最小闭环补丁：若只有 risky_keep 但 top1 来自 similar_to 且风险不高，则提升为 support_keep（非 seed）
    # 这不改变排序公式，只是让“正常落地”与“纯 risky 挂着”在状态语义上分开，便于后续闭环。
    if (
        stage2_mode == "main_strong_process"
        and (not primary_expandable)
        and (not primary_support_seed)
        and (not primary_support_keep)
        and len(risky_keep) == 1
        and (not family_only)
    ):
        rk0 = risky_keep[0]
        src0 = _cand_src_set(rk0)
        if "similar_to" in src0:
            gen0 = float(getattr(rk0, "generic_risk", 0.0) or 0.0)
            poly0 = float(getattr(rk0, "polysemy_risk", 0.0) or 0.0)
            obj0 = float(getattr(rk0, "object_like_risk", 0.0) or 0.0)
            jd0 = float(getattr(rk0, "jd_align", 0.0) or 0.0)
            fam0 = float(getattr(rk0, "family_match", 0.0) or getattr(rk0, "anchor_identity_score", 0.0) or 0.0)
            promote_keep_ok = bool(gen0 <= 0.50 and poly0 <= 0.26 and obj0 <= 0.20 and jd0 >= 0.62 and fam0 >= 0.30)
            if promote_keep_ok:
                risky_keep = []
                primary_support_keep = [rk0]
                rk0.primary_bucket = "primary_support_keep"
                rk0.can_expand = False
                rk0.can_expand_from_2a = False
                rk0.stage2b_seed_tier = "none"
                setattr(rk0, "landing_keep_promoted_from_risky", True)
                setattr(rk0, "landing_keep_promote_reason", "main_strong_similar_to_top1_promote_to_support_keep")
                kept_all = list(primary_support_keep)
                landing_kept_count = len(kept_all)
                landing_state = "weak_landing"
                landing_reason = "promoted_risky_top1_to_support_keep"

    # seed 资格收口：landing_state 不能只是 debug 标签，需影响 Stage2B
    if landing_state in ("fallback_only_landing", "empty_landing"):
        # 不允许 seed；保留项作为弱 evidence（仍可进 Stage3），但不能扩散
        for _c in primary_expandable + primary_support_seed:
            _c.can_expand = False
            _c.can_expand_from_2a = False
            _c.stage2b_seed_tier = "none"
            _c.suppress_seed = True
            setattr(_c, "seed_block_reason", f"landing_state={landing_state}")
        primary_expandable = []
        primary_support_seed = []
    elif landing_state == "weak_landing":
        # weak_landing：默认不当 seed（避免从弱落地扩散）；但不强行改其 support/keep 身份
        for _c in primary_expandable + primary_support_seed:
            _c.can_expand = False
            _c.can_expand_from_2a = False
            _c.stage2b_seed_tier = "none"
            _c.suppress_seed = True
            setattr(_c, "seed_block_reason", "weak_landing_no_seed_by_default")
        primary_expandable = []
        primary_support_seed = []

    # 写回 anchor 级可读状态（Stage2B/日志/下游可读）
    setattr(anchor, "landing_state", landing_state)
    setattr(anchor, "landing_reason", landing_reason)
    setattr(anchor, "candidate_generated_count", int(candidate_generated_count))
    setattr(anchor, "landing_kept_count", int(landing_kept_count))

    if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
        # 中间快照 vs 配额/_finalize 后 final_bucket：对齐 Focus Debug 与 [Stage2A select] 的差异
        _seen_reconcile: Set[int] = set()
        for c in primary_expandable + primary_support_seed + primary_support_keep + risky_keep:
            cid = id(c)
            if cid in _seen_reconcile:
                continue
            _seen_reconcile.add(cid)
            pre_b = _pre_reconcile_bucket_by_id.get(cid, "")
            final_b = (getattr(c, "primary_bucket", None) or "").strip()
            if pre_b and final_b and pre_b != final_b:
                rreason = getattr(c, "seed_downgrade_reason", None) or "conditioned_quota_or_finalize_rebalance"
                print(
                    f"[Stage2A final bucket reconcile] anchor={anchor_term_sel!r} "
                    f"term={(getattr(c, 'term', '') or '')!r} "
                    f"pre_bucket={pre_b!r} final_bucket={final_b!r} reason={rreason!r}"
                )

        anchor_term = getattr(anchor, "anchor", "") or ""
        print(
            f"[Stage2A select] anchor={anchor_term!r} "
            f"expandable={[x.term for x in primary_expandable]} "
            f"support_seed={[x.term for x in primary_support_seed]} "
            f"support_keep={[x.term for x in primary_support_keep]} "
            f"risky_keep={[x.term for x in risky_keep]}"
        )
        _seeds_2b = [x.term for x in primary_expandable] + [x.term for x in primary_support_seed]
        print(
            f"[Stage2A anchor bucket summary] anchor={anchor_term!r} "
            f"expandable_count={len(primary_expandable)} "
            f"support_seed_count={len(primary_support_seed)} "
            f"support_keep_count={len(primary_support_keep)} "
            f"risky_keep_count={len(risky_keep)} "
            f"rejected_count={len(rejected)} "
            f"seed_candidates_for_stage2b={_seeds_2b!r}"
        )

    if LABEL_EXPANSION_DEBUG:
        _nland = (
            len(primary_expandable)
            + len(primary_support_seed)
            + len(primary_support_keep)
            + len(risky_keep)
        )
        print(
            f"[Stage2A retain summary] anchor={anchor_term_sel!r} "
            f"landing={_nland} "
            f"expandable={len(primary_expandable)} support_seed={len(primary_support_seed)} "
            f"support_keep={len(primary_support_keep)} risky_keep={len(risky_keep)}"
        )
        # landing 闭环摘要：区分 candidate_generated vs landing_kept，并输出落地状态（可解释）
        _anchor_text = getattr(anchor, "anchor", "") or ""
        _m = (getattr(anchor, "stage2_process_mode", "") or "").strip()
        print(
            f"[Stage2A landing summary] anchor={_anchor_text!r} stage2_process_mode={_m!r} "
            f"candidate_generated_count={candidate_generated_count} landing_kept_count={landing_kept_count} "
            f"landing_state={landing_state!r} landing_reason={landing_reason!r}"
        )

    for c in primary_expandable:
        setattr(c, "candidate_layer", "candidate_core")
        setattr(c, "candidate_layer_note", "legacy: primary_bucket kept for Stage3 compat; prefer candidate_layer")
    for c in primary_support_seed + primary_support_keep + risky_keep:
        setattr(c, "candidate_layer", "candidate_support")
    for c in rejected:
        setattr(c, "candidate_layer", "candidate_noise")

    if LABEL_EXPANSION_DEBUG:
        print(
            f"[Stage2A candidate_layer_summary] anchor={anchor_term_sel!r} "
            f"core_count={len(primary_expandable)} "
            f"support_count={len(primary_support_seed) + len(primary_support_keep) + len(risky_keep)} "
            f"noise_count={len(rejected)}"
        )

    return {
        "primary_expandable": primary_expandable,
        "primary_support_seed": primary_support_seed,
        "primary_support_keep": primary_support_keep,
        "risky_keep": risky_keep,
        "primary_keep_no_expand": primary_keep_no_expand,
        "rejected": rejected,
        "candidate_core": primary_expandable,
        "candidate_support": primary_support_seed + primary_support_keep + risky_keep,
        "candidate_noise": rejected,
    }


def _update_merged_best_scores(merged_item: Dict[str, Any], cand: Any) -> None:
    """合并时更新 best_* 为各锚点下该 tid 的最优分。"""
    mp = float(cand.mainline_preference) if isinstance(cand.mainline_preference, (int, float)) else 0.0
    ai = float(getattr(cand, "anchor_identity_score", 0) or getattr(cand, "family_match", 0) or 0)
    jd = float(getattr(cand, "jd_align", 0) or 0)
    cc = float(getattr(cand, "context_continuity", 0) or 0)
    merged_item["best_mainline_preference"] = max(merged_item.get("best_mainline_preference", 0), mp)
    merged_item["best_anchor_identity"] = max(merged_item.get("best_anchor_identity", 0), ai)
    merged_item["best_jd_align"] = max(merged_item.get("best_jd_align", 0), jd)
    merged_item["best_context_continuity"] = max(merged_item.get("best_context_continuity", 0), cc)


def _family_type_rank(ft: str) -> int:
    """越大越差：exact_like=0, near_synonym=1, generic=2, shifted=3。"""
    if ft == "shifted":
        return 3
    if ft == "generic":
        return 2
    if ft == "near_synonym":
        return 1
    return 0


def _init_merged_primary(cand: "Stage2ACandidate", anchor_term: str) -> Dict[str, Any]:
    """merge_stage2a_primary 时每个 tid 的初始合并项；含 support_roles、retain_modes、best_*、family_type。"""
    role = getattr(cand, "role_in_anchor", None) or "side"
    retain = getattr(cand, "retain_mode", "normal")
    ft = getattr(cand, "family_type", "") or ""
    return {
        "tid": cand.tid,
        "term": cand.term or str(cand.tid),
        "anchors": [anchor_term],
        "support_count": 1,
        "best_rank": cand.anchor_internal_rank if getattr(cand, "anchor_internal_rank", None) is not None else 10**9,
        "mainline_hits": 1 if role == "mainline" else 0,
        "can_expand": bool(getattr(cand, "can_expand_from_2a", getattr(cand, "can_expand", False))),
        "support_roles": [role],
        "retain_modes": [retain],
        "family_type": ft,
        "best_mainline_preference": float(cand.mainline_preference) if isinstance(cand.mainline_preference, (int, float)) else 0.0,
        "best_anchor_identity": float(getattr(cand, "anchor_identity_score", 0) or getattr(cand, "family_match", 0) or 0),
        "best_jd_align": float(getattr(cand, "jd_align", 0) or 0),
        "best_context_continuity": float(getattr(cand, "context_continuity", 0) or 0),
    }


def merge_stage2a_primary(all_anchor_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    跨锚汇总：同一 tid 合并，保留 anchors、support_count、mainline_hits、best_rank、can_expand、
    support_roles、retain_modes。
    """
    merged: Dict[int, Dict[str, Any]] = {}
    for block in all_anchor_results:
        anchor = block.get("anchor")
        anchor_term = getattr(anchor, "anchor", "") or str(getattr(anchor, "vid", "")) if anchor else ""
        # primary_keep_no_expand 为 seed/support_keep/risky 的并集，仅用于旧接口；合并时只读四类避免重复计数
        for group_name in (
            "primary_expandable",
            "primary_support_seed",
            "primary_support_keep",
            "risky_keep",
        ):
            for cand in block.get(group_name, []):
                tid = cand.tid
                role = getattr(cand, "role_in_anchor", None) or "side"
                retain = getattr(cand, "retain_mode", "normal")
                if tid not in merged:
                    merged[tid] = _init_merged_primary(cand, anchor_term)
                else:
                    merged[tid]["anchors"].append(anchor_term)
                    merged[tid]["support_count"] += 1
                    rank = getattr(cand, "anchor_internal_rank", None)
                    if rank is not None:
                        merged[tid]["best_rank"] = min(merged[tid]["best_rank"], rank)
                    merged[tid]["mainline_hits"] += 1 if role == "mainline" else 0
                    merged[tid]["can_expand"] = merged[tid]["can_expand"] or bool(
                        getattr(cand, "can_expand_from_2a", getattr(cand, "can_expand", False))
                    )
                    merged[tid]["support_roles"].append(role)
                    merged[tid]["retain_modes"].append(retain)
                    _update_merged_best_scores(merged[tid], cand)
                    ft_cand = getattr(cand, "family_type", "") or ""
                    if _family_type_rank(ft_cand) > _family_type_rank(merged[tid].get("family_type", "")):
                        merged[tid]["family_type"] = ft_cand
        # 兼容旧结构：若仍有 block["primary"] 列表也参与合并
        for cand in block.get("primary", []):
            tid = cand.tid
            role = getattr(cand, "role_in_anchor", None) or "side"
            retain = getattr(cand, "retain_mode", "normal")
            if tid not in merged:
                merged[tid] = _init_merged_primary(cand, anchor_term)
            else:
                merged[tid]["anchors"].append(anchor_term)
                merged[tid]["support_count"] += 1
                rank = getattr(cand, "anchor_internal_rank", None)
                if rank is not None:
                    merged[tid]["best_rank"] = min(merged[tid]["best_rank"], rank)
                merged[tid]["mainline_hits"] += 1 if role == "mainline" else 0
                merged[tid]["can_expand"] = merged[tid]["can_expand"] or bool(
                    getattr(cand, "can_expand_from_2a", getattr(cand, "can_expand", False))
                )
                merged[tid]["support_roles"].append(role)
                merged[tid]["retain_modes"].append(retain)
                _update_merged_best_scores(merged[tid], cand)
                ft_cand = getattr(cand, "family_type", "") or ""
                if _family_type_rank(ft_cand) > _family_type_rank(merged[tid].get("family_type", "")):
                    merged[tid]["family_type"] = ft_cand
    # Stage2A 终稿：跨锚仅弱加分，不洗白 generic/shifted
    for term in merged.values():
        raw_cnt = len(term["anchors"])
        raw_bonus = min(0.10, 0.03 * max(0, raw_cnt - 1))
        ft = term.get("family_type", "") or ""
        if ft in ("generic", "shifted"):
            effective_bonus = raw_bonus * 0.3
        else:
            effective_bonus = raw_bonus
        term["cross_anchor_support_raw"] = raw_cnt
        term["cross_anchor_support_effective"] = effective_bonus
        term["cross_anchor_bonus"] = effective_bonus
    return sorted(
        merged.values(),
        key=lambda x: (x["mainline_hits"], x["support_count"], -x["best_rank"], x["can_expand"]),
        reverse=True,
    )


def landing_candidates_to_stage2a(landing_list: List[LandingCandidate]) -> List[Stage2ACandidate]:
    """将 collect_landing_candidates 的 LandingCandidate 列表转为 Stage2ACandidate，并复制定性字段与分型。"""
    out: List[Stage2ACandidate] = []
    for c in landing_list:
        src_set = getattr(c, "source_set", None) or {getattr(c, "source", "") or c.source}
        if not isinstance(src_set, set):
            src_set = set(src_set) if src_set else {getattr(c, "source", "") or c.source}
        out.append(Stage2ACandidate(
            tid=c.vid,
            term=(c.term or "").strip() or str(c.vid),
            source=(getattr(c, "source", "") or "similar_to").strip(),
            semantic_score=float(getattr(c, "semantic_score", 0) or 0),
            context_sim=float(getattr(c, "context_sim", 0) or 0),
            surface_sim=getattr(c, "surface_sim", None),
            conditioned_sim=getattr(c, "conditioned_sim", None),
            context_gain=float(getattr(c, "context_gain", 0) or 0),
            source_set=src_set,
            jd_align=float(getattr(c, "jd_candidate_alignment", 0.5) or 0.5),
            family_match=float(getattr(c, "anchor_identity_score", 0.5) or 0.5),
            anchor_identity_score=float(getattr(c, "anchor_identity_score", 0.5) or 0.5),
            hierarchy_consistency=float(getattr(c, "hierarchy_consistency", 0) or 0),
            polysemy_risk=float(getattr(c, "polysemy_risk", 0) or 0),
            object_like_risk=float(getattr(c, "object_like_risk", 0) or 0),
            generic_risk=float(getattr(c, "generic_risk", 0) or 0),
            context_continuity=float(getattr(c, "context_continuity", 0) or 0),
            source_type=getattr(c, "source_type", "similar_to") or "similar_to",
            source_rank=int(getattr(c, "source_rank", 0) or 0),
            source_score=float(getattr(c, "source_score", 0) or 0),
            family_type=getattr(c, "family_type", "") or "",
            scene_shifted=bool(getattr(c, "scene_shifted", False)),
            generic_like=bool(getattr(c, "generic_like", False)),
            expand_block_reason=getattr(c, "expand_block_reason", None),
            source_trust=float(getattr(c, "source_trust", 1.0) or 1.0),
            ctx_supported=bool(getattr(c, "ctx_supported", getattr(c, "context_supported", False))),
        ))
        setattr(out[-1], "has_family_evidence", getattr(c, "has_family_evidence", False))
        setattr(out[-1], "family_fallback_only", getattr(c, "family_fallback_only", False))
    return out


def _score_anchor_family_match(cand: Dict[str, Any], anchor: PreparedAnchor, _context: Dict[str, Any]) -> float:
    """是否仍在 anchor 语义族里：alias / 近义 canonical family，非黑名单。"""
    anchor_term = getattr(anchor, "anchor", "") or ""
    candidate_term = (cand.get("term") or "").strip()
    if not candidate_term:
        return 0.0
    return compute_anchor_identity_score(anchor_term, candidate_term, getattr(anchor, "anchor_type", None))


def _get_candidate_vec_for_mainline(label: Any, tid: int) -> Optional[np.ndarray]:
    """取候选词向量供 mainline/bonus alignment 用；无 label 或无向量时返回 None。"""
    idx = getattr(label, "vocab_to_idx", None)
    if idx is None:
        return None
    i = idx.get(str(tid)) if isinstance(idx, dict) else idx.get(tid)
    if i is None:
        return None
    all_vecs = getattr(label, "all_vocab_vectors", None)
    if all_vecs is None:
        return None
    try:
        vec = np.asarray(all_vecs[i], dtype=np.float32)
        return vec.flatten() if vec.ndim != 1 else vec
    except Exception:
        return None


def _cos_sim_mainline(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度，限制到 [0, 1]。"""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    if a.size != b.size or a.size == 0:
        return 0.0
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n <= 1e-9:
        return 0.0
    return float(max(0.0, min(1.0, np.dot(a, b) / n)))


def compute_view_stability(feat: Dict[str, Any], T: Optional[SimpleNamespace] = None) -> float:
    """同源双视角稳定性；降权为辅助项，不再主导。"""
    base = float(feat.get("base_score", 0) or 0)
    ctx = float(feat.get("ctx_score", 0) or 0)
    base_hit = bool(feat.get("base_hit"))
    ctx_hit = bool(feat.get("ctx_hit"))
    if not base_hit and not ctx_hit:
        return 0.0
    overlap = 1.0 if (base_hit and ctx_hit) else 0.6
    score_gap = abs(base - ctx)
    rank_gap = 0.0
    br, cr = feat.get("base_rank"), feat.get("ctx_rank")
    if br is not None and cr is not None:
        rank_gap = abs(int(br) - int(cr))
    return _clip01(
        overlap
        * (1.0 - min(1.0, score_gap / 0.20) * 0.45)
        * (1.0 - min(1.0, rank_gap / 5.0) * 0.25)
    )


def _score_family_alias_match(anchor: PreparedAnchor, feat: Dict[str, Any], _jd_profile: Optional[Dict[str, Any]]) -> float:
    anchor_term = getattr(anchor, "anchor", "") or ""
    candidate_term = (feat.get("term") or "").strip()
    if not candidate_term:
        return 0.0
    return compute_anchor_identity_score(anchor_term, candidate_term, getattr(anchor, "anchor_type", None))


def _score_co_anchor_identity_support(
    feat: Dict[str, Any],
    all_anchors: List[PreparedAnchor],
    current_anchor: PreparedAnchor,
    label: Any,
) -> float:
    """候选与其它主线锚点的协同度（向量相似度平均）。"""
    tid = feat.get("tid")
    if tid is None:
        return 0.0
    cand_vec = _get_candidate_vec_for_mainline(label, int(tid))
    if cand_vec is None:
        return 0.0
    sims = []
    for a in all_anchors:
        if getattr(a, "vid", None) == getattr(current_anchor, "vid", None):
            continue
        cv = getattr(a, "conditioned_vec", None)
        if cv is None:
            continue
        try:
            v = np.asarray(cv, dtype=np.float32).flatten()
            if v.size == cand_vec.size:
                sims.append(_cos_sim_mainline(cand_vec, v))
        except Exception:
            pass
    return float(np.mean(sims)) if sims else 0.0


def compute_anchor_candidate_identity(
    anchor: PreparedAnchor,
    feat: Dict[str, Any],
    jd_profile: Optional[Dict[str, Any]],
    all_anchors: Optional[List[PreparedAnchor]] = None,
    label: Any = None,
) -> float:
    """候选是否仍保持锚点身份；显式加入与其它主线 anchor 的协同度。"""
    base_anchor_match = float(feat.get("base_score", 0) or 0)
    stable_overlap = min(float(feat.get("base_score", 0) or 0), float(feat.get("ctx_score", 0) or 0))
    alias_family_match = _score_family_alias_match(anchor, feat, jd_profile)
    co_anchor_identity = 0.0
    if all_anchors and label:
        co_anchor_identity = _score_co_anchor_identity_support(feat, all_anchors, anchor, label)
    return _clip01(
        0.35 * base_anchor_match
        + 0.20 * stable_overlap
        + 0.20 * alias_family_match
        + 0.25 * co_anchor_identity
    )


def _compute_topic_path_proximity(feat: Dict[str, Any], _jd_profile: Optional[Dict[str, Any]]) -> float:
    return _clip01(0.5 * (float(feat.get("subfield_fit", 0) or 0) + float(feat.get("topic_fit", 0) or 0)))


def compute_hierarchy_consistency(feat: Dict[str, Any], jd_profile: Optional[Dict[str, Any]]) -> float:
    """
    候选与 JD 四层层级一致性：field / subfield / topic / path_match 四段，
    高 topic 但低 path_match 时施加惩罚（单一 topic 命中、路径不落地视为可疑）。
    """
    field_fit = float(feat.get("field_fit", 0) or 0)
    subfield_fit = float(feat.get("subfield_fit", 0) or 0)
    topic_fit = float(feat.get("topic_fit", 0) or 0)
    path_match = float(feat.get("path_match", 0) or 0)
    if path_match <= 0 and (subfield_fit > 0 or topic_fit > 0):
        path_match = _compute_topic_path_proximity(feat, jd_profile)
    raw = (
        0.20 * field_fit
        + 0.25 * subfield_fit
        + 0.25 * topic_fit
        + 0.30 * path_match
    )
    mismatch_penalty = 0.0
    if topic_fit >= 0.70 and path_match < 0.25:
        mismatch_penalty = 0.25
    return _clip01(raw * (1.0 - mismatch_penalty))


def compute_context_shift_quality(
    feat: Dict[str, Any],
    mainline_alignment: float,
) -> float:
    """上下文偏移只有当让候选更贴主线时才算正收益。"""
    gain = float(feat.get("shift_gain", 0) or 0)
    drop = float(feat.get("shift_drop", 0) or 0)
    jd_align = float(feat.get("jd_align", 0) or 0)
    gain_term = min(1.0, gain / 0.08) if 0.08 > 0 else 0.0
    drop_term = min(1.0, drop / 0.08) if 0.08 > 0 else 0.0
    return _clip01(
        0.50 * gain_term
        + 0.35 * mainline_alignment
        + 0.15 * jd_align
        - 0.45 * drop_term
    )


def compute_ambiguity_risk(feat: Dict[str, Any], jd_profile: Optional[Dict[str, Any]]) -> float:
    view_stability = float(feat.get("view_stability", 0.5) or 0.5)
    hierarchy = float(feat.get("hierarchy", 0.5) or 0.5)
    return _clip01((1.0 - view_stability) * 0.5 + (1.0 - hierarchy) * 0.5)


# Stage2A 泛词风险：过泛词（压），与 polysemy 分开
GENERIC_RISK_TERMS = ("control", "automatic control", "machine control", "general robotics", "general robotics family")


def compute_generic_risk(feat: Dict[str, Any], jd_profile: Optional[Dict[str, Any]]) -> float:
    """单独衡量「词是不是太泛」，用于压 control / automatic control / machine control / general robotics family。与 polysemy 分开。"""
    risk, _ = compute_generic_risk_with_note(feat, jd_profile)
    return risk


def compute_generic_risk_with_note(
    feat: Dict[str, Any], jd_profile: Optional[Dict[str, Any]]
) -> Tuple[float, str]:
    """返回 (generic_risk, generic_note)。"""
    term = (feat.get("term") or "").strip().lower()
    hierarchy = float(feat.get("hierarchy", 0.5) or 0.5)
    jd_align = float(feat.get("jd_align", 0.5) or 0.5)
    base = _clip01((1.0 - hierarchy) * 0.5 + (1.0 - jd_align) * 0.2)
    note = ""
    for g in GENERIC_RISK_TERMS:
        if g in term or term == g:
            base = max(base, 0.5)
            note = "generic_term"
            break
    return base, note


def compute_branch_drift_risk(
    mainline_alignment: float,
    bonus_alignment: float,
    generic_risk: float,
    object_like_risk: float,
) -> float:
    """
    支线漂移风险：主线一致性越低越高；bonus/支线一致性越高越高；泛词/对象词风险越高越高。
    用于在 Stage2A 压住 medical robotics、reinforcement learning 等支线。
    """
    risk = 0.0
    risk += max(0.0, 0.60 - mainline_alignment) * 0.45
    risk += bonus_alignment * 0.30
    risk += generic_risk * 0.15
    risk += object_like_risk * 0.10
    return _clip01(risk)


def _lexical_object_like_score(term: str) -> float:
    """词形上更像具体对象/器官/部件而非 canonical 学术概念。"""
    if not term:
        return 0.0
    t = term.lower()
    object_like = ("hand", "arm", "finger", "leg", "joint", "organ", "surgical robot", "manipulator")
    if any(x in t for x in object_like):
        return 0.5
    if len(t.split()) >= 4:
        return 0.2
    return 0.0


def _neighborhood_specificity_score(feat: Dict[str, Any]) -> float:
    """邻域细粒度：占位，无 label 时返回 0。"""
    return 0.0


def _hierarchy_tail_specificity(feat: Dict[str, Any]) -> float:
    """层级末端细粒度：topic 过细、更像具体对象层级。占位返回 0。"""
    return 0.0


def compute_object_like_risk(feat: Dict[str, Any]) -> float:
    """
    比 anchor 更细粒度、更像具体对象/器官/部件而非 canonical academic term 则加风险。
    综合：lexical + neighborhood + hierarchy_tail。用于压 motion controller/robotic hand 等，不压死 robotic arm。
    """
    term = (feat.get("term") or "").strip()
    if not term:
        return 0.0
    lexical = _lexical_object_like_score(term)
    neighborhood = _neighborhood_specificity_score(feat)
    hierarchy_tail = _hierarchy_tail_specificity(feat)
    return _clip01(0.40 * lexical + 0.35 * neighborhood + 0.25 * hierarchy_tail)


def compute_object_like_risk_with_note(feat: Dict[str, Any]) -> Tuple[float, str]:
    """返回 (object_like_risk, object_like_note)。"""
    risk = compute_object_like_risk(feat)
    note = "object_like" if risk >= 0.4 else ""
    return risk, note


def compute_polysemy_risk(
    term: str,
    anchor_identity: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Stage2A 多义风险：连续分，用于压 control flow / dyskinesia / sports science / simula，
    不把 motion control / reinforcement learning / robotic arm 压死。不按来源一刀切。
    """
    t = (term or "").strip().lower()
    risk = 0.25
    note = ""
    for s in POLYSEMY_SAFE_SUBSTRINGS:
        if s in t:
            risk = min(risk, 0.22)
            note = "safe_substring"
            break
    for s in POLYSEMY_HIGH_RISK_SUBSTRINGS:
        if s in t:
            risk = max(risk, 0.58)
            note = "high_polysemy"
            break
    if anchor_identity is not None and anchor_identity >= 0.5 and not note:
        risk = risk * 0.7
    return (max(0.0, min(1.0, risk)), note)


def compute_candidate_risks(feat: Dict[str, Any], jd_profile: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    """返回 (ambiguity_risk, generic_risk)；branch_drift_risk 由 compute_branch_drift_risk(mainline, bonus, generic, object_like) 单独算。"""
    return (
        compute_ambiguity_risk(feat, jd_profile),
        compute_generic_risk(feat, jd_profile),
    )


def compute_canonicalness(
    feat: Dict[str, Any],
    anchor: PreparedAnchor,
    alias_match_fn=None,
) -> float:
    """
    Canonical 学术表达偏好：alias 家族匹配 + cross_anchor 支持 + path_match。
    用于抬升 motion control、压制 movement control 等非标准表述。
    """
    alias_match = _score_family_alias_match(anchor, feat, None) if alias_match_fn is None else alias_match_fn(feat)
    cross_anchor = float(feat.get("cross_anchor_support", feat.get("identity", 0) or 0))
    if cross_anchor <= 0 and feat.get("identity") is not None:
        cross_anchor = float(feat.get("identity", 0) or 0) * 0.5
    path_match = float(feat.get("path_match", 0) or 0)
    if path_match <= 0:
        path_match = 0.5 * (float(feat.get("subfield_fit", 0) or 0) + float(feat.get("topic_fit", 0) or 0))
    return _clip01(0.40 * alias_match + 0.30 * cross_anchor + 0.30 * path_match)


def score_stage2a_primary(feat: Dict[str, Any], W: Optional[SimpleNamespace] = None) -> float:
    """
    主线优先 + canonical 落点：mainline_alignment / identity / canonicalness 提权，
    view_stability 降为辅助；三项风险惩罚。
    """
    base = (
        0.22 * float(feat.get("mainline_alignment", 0) or 0)
        + 0.18 * float(feat.get("identity", 0) or 0)
        + 0.14 * float(feat.get("canonicalness", 0) or 0)
        + 0.14 * float(feat.get("base_score", 0) or 0)
        + 0.12 * float(feat.get("hierarchy", 0) or 0)
        + 0.08 * float(feat.get("jd_align", 0) or 0)
        + 0.06 * float(feat.get("context_shift_quality", 0) or 0)
        + 0.06 * float(feat.get("view_stability", 0) or 0)
    )
    penalty = (
        (1.0 - 0.30 * float(feat.get("branch_drift_risk", 0) or 0))
        * (1.0 - 0.20 * float(feat.get("object_like_risk", 0) or 0))
        * (1.0 - 0.20 * float(feat.get("ambiguity_risk", 0) or 0))
    )
    return _clip01(base * max(0.0, penalty))


def decide_stage2a_bucket(feat: Dict[str, Any], T: SimpleNamespace) -> str:
    """
    四分桶收紧：reject 明显错义/支线漂移；primary_expandable 需主线强+canonical+风险低；
    primary_keep_no_expand 主线相关且可解释；其余 observe_only。
    """
    mainline = float(feat.get("mainline_alignment", 0) or 0)
    identity = float(feat.get("identity", 0) or 0)
    branch_drift = float(feat.get("branch_drift_risk", 0) or 0)
    primary_score = float(feat.get("primary_score", 0) or 0)
    canonicalness = float(feat.get("canonicalness", 0) or 0)
    object_like = float(feat.get("object_like_risk", 0) or 0)

    if (
        mainline < getattr(T, "mainline_low", 0.25)
        and identity < getattr(T, "identity_low", 0.18)
        and branch_drift > getattr(T, "branch_drift_high", 0.65)
    ):
        return "reject"

    if (
        mainline >= getattr(T, "mainline_expand", 0.55)
        and identity >= getattr(T, "identity_expand", 0.55)
        and canonicalness >= getattr(T, "canonical_expand", 0.45)
        and branch_drift < getattr(T, "branch_drift_low", 0.35)
        and object_like < getattr(T, "object_like_low", 0.35)
        and primary_score >= getattr(T, "primary_expand", 0.50)
    ):
        return "primary_expandable"

    if (
        mainline >= getattr(T, "mainline_keep", 0.40)
        and primary_score >= getattr(T, "primary_keep", 0.35)
    ):
        return "primary_keep_no_expand"

    return "observe_only"


def calibrate_anchor_thresholds(evaluated_candidates: List[Dict[str, Any]], global_floor: SimpleNamespace) -> SimpleNamespace:
    """按锚点内分布得到相对阈值，与全局底线取 max；含 mainline_*、object_like_low。"""
    if not evaluated_candidates:
        return global_floor
    identity_vals = sorted([float(c.get("identity", 0) or 0) for c in evaluated_candidates])
    view_vals = sorted([float(c.get("view_stability", 0) or 0) for c in evaluated_candidates])
    primary_vals = sorted([float(c.get("primary_score", 0) or 0) for c in evaluated_candidates])
    mainline_vals = sorted([float(c.get("mainline_alignment", 0) or 0) for c in evaluated_candidates])
    n = len(identity_vals)
    p25 = max(0, n * 25 // 100)
    p60 = max(0, min(n - 1, n * 60 // 100))
    p75 = max(0, min(n - 1, n * 75 // 100))
    p80 = max(0, min(n - 1, n * 80 // 100))
    T = SimpleNamespace()
    T.identity_low = max(getattr(global_floor, "identity_low", 0.18), identity_vals[p25] if identity_vals else 0)
    T.identity_primary = max(getattr(global_floor, "identity_primary", 0.35), identity_vals[p60] if identity_vals else 0)
    T.identity_keep = max(getattr(global_floor, "identity_keep", 0.40), identity_vals[p60] if identity_vals else 0)
    T.identity_expand = max(getattr(global_floor, "identity_expand", 0.55), identity_vals[p75] if identity_vals else 0)
    T.view_stability_low = max(getattr(global_floor, "view_stability_low", 0.25), view_vals[p25] if view_vals else 0)
    T.view_stability_primary = max(getattr(global_floor, "view_stability_primary", 0.45), view_vals[p60] if view_vals else 0)
    T.view_stability_keep = max(getattr(global_floor, "view_stability_keep", 0.50), view_vals[p60] if view_vals else 0)
    T.view_stability_expand = max(getattr(global_floor, "view_stability_expand", 0.60), view_vals[p75] if view_vals else 0)
    T.primary_keep_line = max(getattr(global_floor, "primary_keep_line", 0.35), primary_vals[p60] if primary_vals else 0)
    T.primary_expand_line = max(getattr(global_floor, "primary_expand_line", 0.50), primary_vals[p80] if primary_vals else 0)
    T.primary_keep = max(getattr(global_floor, "primary_keep", 0.35), primary_vals[p60] if primary_vals else 0)
    T.primary_expand = max(getattr(global_floor, "primary_expand", 0.50), primary_vals[p80] if primary_vals else 0)
    canonical_vals = sorted([float(c.get("canonicalness", 0) or 0) for c in evaluated_candidates])
    T.canonical_expand = max(getattr(global_floor, "canonical_expand", 0.45), canonical_vals[p75] if canonical_vals else 0)
    T.mainline_low = max(getattr(global_floor, "mainline_low", 0.25), mainline_vals[p25] if mainline_vals else 0)
    T.mainline_keep = max(getattr(global_floor, "mainline_keep", 0.40), mainline_vals[p60] if mainline_vals else 0)
    T.mainline_expand = max(getattr(global_floor, "mainline_expand", 0.55), mainline_vals[p75] if mainline_vals else 0)
    T.base_expand_line = getattr(global_floor, "base_expand_line", 0.55)
    T.hierarchy_low = getattr(global_floor, "hierarchy_low", 0.20)
    T.hierarchy_mid = getattr(global_floor, "hierarchy_mid", 0.35)
    T.hierarchy_expand = getattr(global_floor, "hierarchy_expand", 0.50)
    T.shift_quality_low = getattr(global_floor, "shift_quality_low", 0.20)
    T.shift_quality_mid = getattr(global_floor, "shift_quality_mid", 0.40)
    T.shift_quality_expand = getattr(global_floor, "shift_quality_expand", 0.55)
    T.ambiguity_high = getattr(global_floor, "ambiguity_high", 0.70)
    T.ambiguity_mid = getattr(global_floor, "ambiguity_mid", 0.50)
    T.ambiguity_low = getattr(global_floor, "ambiguity_low", 0.35)
    T.generic_mid = getattr(global_floor, "generic_mid", 0.45)
    T.generic_low = getattr(global_floor, "generic_low", 0.35)
    T.branch_drift_high = getattr(global_floor, "branch_drift_high", 0.65)
    T.branch_drift_mid = getattr(global_floor, "branch_drift_mid", 0.45)
    T.branch_drift_low = getattr(global_floor, "branch_drift_low", 0.35)
    T.object_like_low = getattr(global_floor, "object_like_low", 0.35)
    T.jd_align_mid = getattr(global_floor, "jd_align_mid", 0.55)
    T.max_reasonable_shift = getattr(global_floor, "max_reasonable_shift", 0.25)
    T.max_reasonable_rank_gap = getattr(global_floor, "max_reasonable_rank_gap", 8)
    T.max_useful_gain = getattr(global_floor, "max_useful_gain", 0.20)
    T.max_tolerable_drop = getattr(global_floor, "max_tolerable_drop", 0.15)
    return T


# ---------- Identity Gate：候选与锚点"本义"一致性，用于压制错义（propulsion/kinesics/simula 等） ----------候选与锚点“本义”一致性，用于压制错义（propulsion/kinesics/simula 等） ----------
# 软闸门：identity_score -> gate 乘数，不硬删
IDENTITY_GATE_THRESHOLDS = [(0.75, 1.00), (0.55, 0.90), (0.35, 0.72)]  # (min_score, gate); else 0.45
# 错义/泛词惩罚：candidate 为这些词时 identity 压低（与锚点无稳定 lexical family 时不得过高）
IDENTITY_AMBIGUITY_TERMS = frozenset({
    "control", "robot", "robotics", "machine", "learning", "retrieval", "data", "crawling",
    "point", "point-to-point", "principle", "flow", "management", "digital", "automatic",
    "personal", "robot", "palo", "simula", "kinesics", "propulsion", "dynamism", "mechanics",
})
# Stage2A 多义风险：高风险词（压）、安全子串（不压）
POLYSEMY_HIGH_RISK_SUBSTRINGS = (
    "control flow", "dyskinesia", "kinesics", "sports science", "simula",
    "data retrieval", "crawling", "point-to-point", "end-to-end principle",
)
POLYSEMY_SAFE_SUBSTRINGS = (
    "motion control", "reinforcement learning", "robotic arm", "robot arm",
    "path planning", "route planning", "motion planning", "trajectory planning",
)
# 常见“锚点本义”英文对应（最小白名单，用于 boost；中文锚点可逐步加）
ANCHOR_IDENTITY_ALIASES: Dict[str, Set[str]] = {
    "动力学": {"dynamics", "dynamic", "mechanics", "kinetics"},
    "运动学": {"kinematics", "kinesiology"},
    "仿真": {"simulation", "simulate", "simulator"},
    "路径规划": {"route planning", "path planning", "motion planning", "trajectory planning"},
    "控制": {"control", "controller", "control engineering"},
    "运动控制": {"motion control", "movement control", "motion controller"},
    "机械臂": {"robotic arm", "robot arm", "manipulator"},
    "强化学习": {"reinforcement learning", "reinforcement", "rl"},
    "抓取": {"grasping", "grasp", "manipulation", "gripper"},
    "端到端": {"end-to-end", "end to end"},
}


def normalize_identity_surface(term: str) -> Dict[str, Any]:
    """
    轻量 identity 归一化，供 lexical identity 计算用。
    返回: raw, norm, tokens, token_set, head
    """
    if not isinstance(term, str):
        term = str(term or "")
    raw = term.strip()
    s = raw.lower().strip()
    # 去括号说明：control (management) -> control
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    # 连字符统一成空格：end-to-end -> end to end
    s = re.sub(r"-", " ", s)
    # 多空格压缩
    s = re.sub(r"\s+", " ", s).strip()
    norm = s
    tokens = [t for t in s.split() if t]
    token_set = set(tokens)
    head = tokens[0] if tokens else ""
    return {"raw": raw, "norm": norm, "tokens": tokens, "token_set": token_set, "head": head}


def lexical_shape_match(anchor_term: str, candidate_term: str) -> float:
    """
    词形结构匹配度 0~1，用于候选分型：真近义 vs 泛词/场景漂移。
    基于 token 重叠与规范形式，无硬编码词表。
    """
    a = normalize_identity_surface(anchor_term or "")
    c = normalize_identity_surface(candidate_term or "")
    atok = a["token_set"]
    ctok = c["token_set"]
    if not atok and not ctok:
        return 1.0 if (anchor_term or "").strip() == (candidate_term or "").strip() else 0.0
    if not atok or not ctok:
        return 0.0
    inter = len(atok & ctok)
    union = len(atok | ctok)
    jaccard = inter / union if union else 0.0
    if a["norm"] == c["norm"]:
        return 1.0
    if a["norm"] in c["norm"] or c["norm"] in a["norm"]:
        return max(jaccard, 0.85)
    return jaccard


def compute_anchor_identity_score(
    anchor_term: str,
    candidate_term: str,
    anchor_type: Optional[str] = None,
    edge_strength: Optional[float] = None,
    context_sim: Optional[float] = None,
) -> float:
    """
    锚点连续性分数（0~1）：综合词面/主干语义、similar_to 边强度、conditioned 对齐，
    用于软区分本义/偏义；不单凭字符串相似或边权。
    """
    a = normalize_identity_surface(anchor_term)
    c = normalize_identity_surface(candidate_term)
    atok = a["token_set"]
    ctok = c["token_set"]
    a_norm = a["norm"]
    c_norm = c["norm"]

    # 信号 1：完全词面/子串一致
    exact_or_substring = 0.0
    if a_norm and c_norm:
        if a_norm == c_norm:
            exact_or_substring = 1.0
        elif a_norm in c_norm or c_norm in a_norm:
            exact_or_substring = 0.85
        elif atok == ctok:
            exact_or_substring = 0.9

    # 信号 2：token overlap + 锚点英文别名字族
    inter = atok & ctok
    union = atok | ctok
    token_overlap_score = len(inter) / len(union) if union else 0.0
    alias_hit = False
    for _anchor, aliases in ANCHOR_IDENTITY_ALIASES.items():
        an_norm = normalize_identity_surface(_anchor)["norm"]
        if an_norm and a_norm and (an_norm == a_norm or normalize_identity_surface(_anchor)["token_set"] == atok):
            for al in aliases:
                al_norm = normalize_identity_surface(al)["norm"]
                if al_norm and (al_norm == c_norm or al_norm in c_norm or c_norm in al_norm):
                    token_overlap_score = max(token_overlap_score, 0.72)
                    alias_hit = True
                    break
            break
    if exact_or_substring >= 0.85:
        token_overlap_score = max(token_overlap_score, 0.9)

    # 信号 3：head 一致性
    head_consistency_score = 0.0
    if a["head"] and a["head"] in ctok:
        head_consistency_score = 0.8
    elif c["head"] and c["head"] in atok:
        head_consistency_score = 0.6
    if exact_or_substring >= 0.85:
        head_consistency_score = max(head_consistency_score, 0.85)

    # 中英映射 / 锚点别名命中时，给一个较高的基础分，避免全体被压成 low identity
    base_floor = 0.0
    if alias_hit:
        base_floor = max(base_floor, 0.34)
    if a_norm and c_norm and any(t in c_norm for t in a_norm.split()):
        base_floor = max(base_floor, 0.26)
    if token_overlap_score >= 0.45 and head_consistency_score >= 0.6:
        base_floor = max(base_floor, 0.30)

    score = max(
        base_floor,
        0.40 * exact_or_substring
        + 0.35 * token_overlap_score
        + 0.25 * head_consistency_score
    )

    # 泛词/歧义惩罚：保留，但不再把正常中英映射一刀砍死
    generic_penalty = 1.0
    if ctok and len(ctok) <= 2:
        low_tokens = {t.lower() for t in c["tokens"]}
        if low_tokens & IDENTITY_AMBIGUITY_TERMS and not (atok & ctok):
            generic_penalty = 0.72
    ambiguity_penalty = 1.0
    c_head_lower = c["head"].lower() if c["head"] else ""
    if c_head_lower in IDENTITY_AMBIGUITY_TERMS and c_head_lower not in atok:
        ambiguity_penalty = 0.78
    if (c_norm in ("control (management)", "control flow", "data retrieval", "crawling",
                   "point-to-point", "end-to-end principle", "simula", "kinesics", "propulsion") and
            not (atok & ctok)):
        ambiguity_penalty = min(ambiguity_penalty, 0.45)

    score *= generic_penalty * ambiguity_penalty
    if edge_strength is not None or context_sim is not None:
        edge = max(0.0, min(1.0, float(edge_strength or 0)))
        ctx = max(0.0, min(1.0, float(context_sim or 0)))
        score = 0.50 * score + 0.25 * edge + 0.25 * ctx
    return max(0.0, min(1.0, score))


def _identity_gate_from_score(anchor_identity_score: float) -> float:
    """软闸门：0.75+ -> 1.0, 0.55+ -> 0.9, 0.35+ -> 0.72, else 0.45"""
    for thresh, gate in IDENTITY_GATE_THRESHOLDS:
        if anchor_identity_score >= thresh:
            return gate
    return 0.45


# ---------- Stage2A 终稿：候选分型（不硬编码岗位词） ----------


def classify_candidate_family(
    anchor_term: str,
    candidate_term: str,
    semantic_score: float,
    context_sim: float,
) -> str:
    """
    轻量语义分型：候选与锚点是 真近义 / 邻域近词 / 泛词 / 场景漂移。
    返回: exact_like | near_synonym | generic | shifted
    """
    lexical = lexical_shape_match(anchor_term or "", candidate_term or "")
    ctx_ok = context_sim >= CTX_SUPPORT_MIN
    sem = float(semantic_score)
    if sem >= 0.82 and lexical >= 0.70 and ctx_ok:
        return "exact_like"
    if sem >= 0.80 and lexical >= 0.45:
        return "near_synonym"
    if sem >= 0.78 and lexical < 0.35 and not ctx_ok:
        return "shifted"
    return "generic"


def is_candidate_generic_like(cand: Any) -> bool:
    """泛词风险：显式布尔，用于主线准入门。"""
    family = getattr(cand, "family_type", None) or cand.get("family_type", "")
    if family in {"generic", "shifted"}:
        return True
    gr = float(getattr(cand, "generic_risk", 0) or cand.get("generic_risk", 0) or 0)
    if gr >= GENERIC_RISK_MIN:
        return True
    pr = float(getattr(cand, "polysemy_risk", 0) or cand.get("polysemy_risk", 0) or 0)
    if pr >= POLY_RISK_MIN:
        return True
    return False


def is_candidate_scene_shifted(cand: Any, jd_ctx: Optional[Dict[str, Any]] = None) -> bool:
    """语义相近但应用场景跑偏（如 kinesiology / medical robotics / propulsion）。"""
    if getattr(cand, "family_type", None) == "shifted" or (isinstance(cand, dict) and cand.get("family_type") == "shifted"):
        return True
    ctx_supported = getattr(cand, "ctx_supported", None)
    if ctx_supported is None:
        ctx_supported = getattr(cand, "context_supported", False)
    if isinstance(cand, dict):
        ctx_supported = cand.get("ctx_supported", cand.get("context_supported", False))
    ctx_gap_val = getattr(cand, "ctx_gap", 0) or getattr(cand, "context_gap", 0)
    if isinstance(cand, dict):
        ctx_gap_val = ctx_gap_val or cand.get("ctx_gap", cand.get("context_gap", 0))
    ctx_gap = float(ctx_gap_val or 0)
    if ctx_supported is False and ctx_gap >= CTX_GAP_SHIFT_MIN:
        return True
    hier = getattr(cand, "hierarchy_score", 0) or getattr(cand, "hierarchy_consistency", 0)
    if isinstance(cand, dict):
        hier = hier or cand.get("hier_score", cand.get("hierarchy_consistency", 0))
    hier = float(hier or 0)
    jd_align = getattr(cand, "jd_align", 0) or getattr(cand, "jd_candidate_alignment", 0.5)
    if isinstance(cand, dict):
        jd_align = jd_align or cand.get("jd_align", cand.get("jd_candidate_alignment", 0.5))
    jd_align = float(jd_align or 0.5)
    if hier < HIER_WEAK_MIN and jd_align < JD_ALIGN_WEAK_MIN:
        return True
    return False


def infer_expand_block_reason(cand: Any) -> Optional[str]:
    """禁止扩散原因：None 表示可参与扩散竞争；否则为 generic | scene_shift | low_identity | weak_context。"""
    if is_candidate_generic_like(cand):
        return "generic"
    if is_candidate_scene_shifted(cand, None):
        return "scene_shift"
    identity = float(getattr(cand, "anchor_identity_score", 0) or getattr(cand, "identity_score", 0) or (cand.get("identity_score", 0) if isinstance(cand, dict) else 0))
    if identity < 0.35:
        return "low_identity"
    ctx_ok = getattr(cand, "ctx_supported", getattr(cand, "context_supported", True))
    if isinstance(cand, dict):
        ctx_ok = cand.get("ctx_supported", cand.get("context_supported", True))
    if ctx_ok is False:
        return "weak_context"
    return None


def get_source_trust(source_type: str) -> float:
    """similar_to / conditioned_vec 基础可信度，用于分型与合并。"""
    s = (source_type or "").strip().lower()
    if s == "similar_to":
        return 1.0
    if s in ("conditioned_vec", "conditioned"):
        return 0.85
    return 0.85


def estimate_family_centeredness(cand: Any) -> float:
    """家族中心性 0~1：exact_like 最高，shifted 最低。"""
    ft = getattr(cand, "family_type", None) or (cand.get("family_type") if isinstance(cand, dict) else None) or "generic"
    if ft == "exact_like":
        return 1.0
    if ft == "near_synonym":
        return 0.85
    if ft == "generic":
        return 0.5
    return 0.3


def is_mainline_admissible(candidate: Any) -> bool:
    """Stage2A 主线准入门：scene_shifted/generic_like 不能扩散，identity 与 family_type 需达标。"""
    if getattr(candidate, "scene_shifted", False) or (candidate.get("scene_shifted") if isinstance(candidate, dict) else False):
        return False
    if getattr(candidate, "generic_like", False) or (candidate.get("generic_like") if isinstance(candidate, dict) else False):
        return False
    identity = float(getattr(candidate, "identity_score", 0) or getattr(candidate, "anchor_identity_score", 0) or (candidate.get("anchor_identity_score", 0) if isinstance(candidate, dict) else 0))
    if identity < MAINLINE_IDENTITY_MIN:
        return False
    ft = getattr(candidate, "family_type", "") or (candidate.get("family_type", "") if isinstance(candidate, dict) else "")
    if ft not in ("exact_like", "near_synonym"):
        return False
    return True


def score_stage2a_candidate(candidate: Any) -> None:
    """
    主线资格分 mainline_pref_score 与保留资格分 retain_score 分开；
    并设 mainline_admissible。原地写入 candidate。
    """
    identity = float(getattr(candidate, "anchor_identity_score", 0) or getattr(candidate, "identity_score", 0) or 0)
    jd_align = float(getattr(candidate, "jd_align", 0) or getattr(candidate, "jd_candidate_alignment", 0.5) or 0.5)
    ctx = float(getattr(candidate, "context_sim", 0) or getattr(candidate, "context_continuity", 0) or 0)
    hier = float(getattr(candidate, "hierarchy_consistency", 0) or getattr(candidate, "hierarchy_score", 0) or 0)
    generic_pen = float(getattr(candidate, "generic_risk", 0) or 0)
    poly_risk = float(getattr(candidate, "polysemy_risk", 0) or 0)
    mainline_pref = (
        0.35 * identity
        + 0.25 * jd_align
        + 0.20 * ctx
        + 0.10 * hier
        - 0.05 * generic_pen
        - 0.05 * poly_risk
    )
    retain_score = (
        0.40 * identity
        + 0.30 * jd_align
        + 0.10 * ctx
        + 0.10 * hier
        - 0.05 * generic_pen
        - 0.05 * poly_risk
    )
    mainline_pref = max(0.0, min(1.0, mainline_pref))
    retain_score = max(0.0, min(1.0, retain_score))
    if hasattr(candidate, "mainline_pref_score"):
        candidate.mainline_pref_score = mainline_pref
    if hasattr(candidate, "retain_score"):
        candidate.retain_score = retain_score
    setattr(candidate, "mainline_pref_score", mainline_pref)
    setattr(candidate, "retain_score", retain_score)
    adm = is_mainline_admissible(candidate)
    if hasattr(candidate, "mainline_admissible"):
        candidate.mainline_admissible = adm
    setattr(candidate, "mainline_admissible", adm)
    # 占位，组选主时在 select_primary_per_anchor 中按 good_mainline 填
    setattr(candidate, "is_good_mainline", False)


def _is_canonical_academic_like_anchor(anchor: PreparedAnchor) -> bool:
    """动力学/运动学/仿真等核心学术锚点，需先走 family landing 补池。"""
    anchor_type = (getattr(anchor, "anchor_type", "") or "").strip().lower()
    if anchor_type == "canonical_academic_like":
        return True
    anchor_text = (getattr(anchor, "anchor", "") or "").strip()
    return anchor_text in CANONICAL_ACADEMIC_ANCHOR_FAMILY_QUERIES


def retrieve_family_landing_candidates(
    label,
    anchor: PreparedAnchor,
    top_k: int = None,
) -> List[LandingCandidate]:
    """
    面向 canonical_academic_like 锚点的轻量 family 补池：先 exact/alias/family 命中，再交给 Stage2A 常规打分。
    只负责补池，不保送 mainline。
    """
    # 收严：family fallback 只做“保线桥接”，不造新路线；默认最多补 1~2 条即可
    top_k = int(top_k) if top_k is not None else STAGE2A_FAMILY_LANDING_TOP_K
    top_k = max(0, min(2, top_k))
    anchor_text = (getattr(anchor, "anchor", "") or "").strip()
    if not anchor_text:
        return []
    family_queries = CANONICAL_ACADEMIC_ANCHOR_FAMILY_QUERIES.get(anchor_text)
    if not family_queries:
        return []
    load_vocab_meta(label)
    meta = getattr(label, "_vocab_meta", None)
    if not meta:
        return []

    # family 候选保线 gate：比 normal retrieval 更严（不靠大词表，主要靠结构与轻量不匹配信号）
    _METHOD_TAIL_RE = re.compile(
        r"\b(analysis|microscopy|spectroscopy|tomography|imaging|assay|measurement|metrology)\b",
        re.IGNORECASE,
    )
    _METHOD_PHRASE_RE = re.compile(
        r"\b(scanning probe|electron microscopy|atomic force|mass spectrometry)\b",
        re.IGNORECASE,
    )

    def _family_keep_gate(anchor_txt: str, term_txt: str, best_score: float) -> Tuple[bool, str]:
        tl = (term_txt or "").strip().lower()
        if not tl:
            return False, "empty_term"
        # 非强贴近（best=0.85）时更保守：过长学术句更易越界
        if best_score < 1.0 and len(tl) >= 28:
            return False, "too_long_for_family_bridge"
        # 研究方法/仪器场景长尾：对“任务型锚点”（抑制/控制/规划等）默认拒绝
        a = (anchor_txt or "").strip()
        task_like = bool(a and any(k in a for k in ("抑制", "控制", "规划", "抓取", "端到端")))
        if task_like and (" with " in tl or "," in tl):
            if _METHOD_TAIL_RE.search(tl) or _METHOD_PHRASE_RE.search(tl):
                return False, "academic_method_tail"
        # 单词过泛：best<1.0 时不收单 token 泛词（避免“vibration”之类泛词靠子串进池）
        if best_score < 1.0 and " " not in tl and len(tl) <= 10:
            return False, "single_token_too_generic"
        return True, "ok"

    scored: List[Tuple[float, int, str, str]] = []
    raw_n = 0
    blocked_reason_counts: Dict[str, int] = {}
    blocked_samples: Dict[str, List[str]] = {}
    for vid, tup in meta.items():
        try:
            vid_int = int(vid)
        except (TypeError, ValueError):
            continue
        term = (tup[0] or "").strip() if isinstance(tup, (list, tuple)) else ""
        vocab_type = (tup[1] or "").strip() if isinstance(tup, (list, tuple)) and len(tup) > 1 else ""
        if vocab_type not in ("concept", "keyword", "") and vocab_type:
            continue
        term_lower = term.lower()
        best = 0.0
        for q in family_queries:
            if q == term_lower:
                best = max(best, 1.0)
            elif q in term_lower or term_lower in q:
                best = max(best, 0.85)
        if best <= 0:
            continue
        raw_n += 1
        keep, why = _family_keep_gate(anchor_text, term, best)
        if not keep:
            blocked_reason_counts[why] = blocked_reason_counts.get(why, 0) + 1
            if len(blocked_samples.get(why, [])) < 3:
                blocked_samples.setdefault(why, []).append(term[:72])
            continue
        scored.append((best, vid_int, term, vocab_type))
    scored.sort(key=lambda x: (-x[0], -len(x[2])))
    out: List[LandingCandidate] = []
    for rank, (score, vid_int, term, _) in enumerate(scored[:top_k], start=1):
        out.append(LandingCandidate(
            vid=vid_int,
            term=term,
            source="family_landing",
            semantic_score=score,
            anchor_vid=getattr(anchor, "vid", 0),
            anchor_term=anchor_text,
        ))
        out[-1].source_type = "family_landing"
        out[-1].source_rank = rank
        out[-1].source_score = score
        setattr(out[-1], "has_family_evidence", True)
    if LABEL_EXPANSION_DEBUG:
        kept_n = len(out)
        blocked_n = max(0, raw_n - kept_n)
        sample = [getattr(c, "term", str(c)) for c in out[:3]]
        print(
            f"[Stage2A] family_landing anchor={anchor_text!r} raw={raw_n} kept={kept_n} blocked={blocked_n} "
            f"top_k={top_k} kept_sample={sample}"
        )
        if blocked_reason_counts:
            top_r = sorted(blocked_reason_counts.items(), key=lambda x: -x[1])[:4]
            rs = ",".join(f"{k}:{v}" for k, v in top_r)
            print(f"[Stage2A] family_landing blocked_reasons_top=[{rs}] blocked_samples={blocked_samples}")
    return out


def compute_candidate_context_gain(cand: Any) -> float:
    """context_gain = conditioned_sim - surface_proxy；surface_proxy 优先 surface_sim，否则 semantic_score，否则 0。
    若 conditioned_sim 为 None，则 context_gain = 0（不套用 0 当作条件化分数）。"""
    cs = getattr(cand, "conditioned_sim", None)
    if cs is None:
        return 0.0
    ss = getattr(cand, "surface_sim", None)
    if ss is not None:
        surf = float(ss)
    else:
        sem = getattr(cand, "semantic_score", None)
        surf = float(sem) if sem is not None else 0.0
    return float(cs) - surf


def merge_landing_candidates_by_tid(candidates: List[LandingCandidate]) -> List[LandingCandidate]:
    """同 tid 合并：保留双路证据 surface_sim / conditioned_sim / source_set，family_landing 只留痕迹不主导主分。"""
    merged: Dict[int, LandingCandidate] = {}
    for c in candidates:
        tid = int(c.vid)
        if tid not in merged:
            merged[tid] = c
            source_set = {getattr(c, "source", "") or c.source}
            setattr(merged[tid], "source_set", source_set)
            setattr(merged[tid], "all_sources", source_set)
            setattr(merged[tid], "has_family_evidence", getattr(c, "source", "") == "family_landing" or getattr(c, "has_family_evidence", False))
            if getattr(c, "source", "") == "family_landing":
                setattr(merged[tid], "has_family_fallback", True)
            continue
        m = merged[tid]
        m.source_set = getattr(m, "source_set", set()) or set()
        m.source_set.add(getattr(c, "source", "") or c.source)
        setattr(m, "all_sources", m.source_set)
        if getattr(c, "source", "") == "similar_to":
            m.surface_sim = max(getattr(m, "surface_sim", None) or 0.0, getattr(c, "surface_sim", None) or c.semantic_score or 0.0)
        if getattr(c, "source", "") == "conditioned_vec":
            m.conditioned_sim = max(getattr(m, "conditioned_sim", None) or 0.0, getattr(c, "conditioned_sim", None) or c.semantic_score or 0.0)
        if getattr(c, "source", "") == "family_landing":
            setattr(m, "has_family_fallback", True)
            setattr(m, "has_family_evidence", True)
        m.semantic_score = max(m.semantic_score, c.semantic_score)
        m.context_sim = max(getattr(m, "context_sim", 0) or 0, getattr(c, "context_sim", 0) or 0)
        m.source_score = max(getattr(m, "source_score", 0) or 0, getattr(c, "source_score", 0) or 0)
    out = list(merged.values())
    for c in out:
        c.context_gain = compute_candidate_context_gain(c)
    return out


def _conditioned_rerank_signals_dot_product(
    label: Any,
    anchor: PreparedAnchor,
    similar_to_candidates: List[LandingCandidate],
) -> Dict[int, Dict[str, float]]:
    """
    用 conditioned_vec 与词表向量的点积为 similar_to 候选补局部证据（非独立 FAISS 主召回）。
    对应 Learning to Link：mention 条件用于候选区分，而非第二套召回主链。
    """
    out: Dict[int, Dict[str, float]] = {}
    cv = getattr(anchor, "conditioned_vec", None)
    if cv is None or not similar_to_candidates:
        for cand in similar_to_candidates:
            out[cand.vid] = {"context_sim": 0.0, "context_supported": 0.0, "context_gap": 1.0}
        return out
    cv = np.asarray(cv, dtype=np.float32).flatten()
    nv = float(np.linalg.norm(cv))
    if nv < 1e-9:
        for cand in similar_to_candidates:
            out[cand.vid] = {"context_sim": 0.0, "context_supported": 0.0, "context_gap": 1.0}
        return out
    cv = cv / nv
    for cand in similar_to_candidates:
        ev = _get_candidate_vec_for_mainline(label, cand.vid)
        static_sim = float(getattr(cand, "semantic_score", 0) or 0)
        if ev is None:
            out[cand.vid] = {"context_sim": 0.0, "context_supported": 0.0, "context_gap": max(0.0, static_sim)}
            continue
        ev = np.asarray(ev, dtype=np.float32).flatten()
        ne = float(np.linalg.norm(ev))
        cos = float(np.dot(cv, ev / ne)) if ne > 1e-9 else 0.0
        cos = max(0.0, min(1.0, cos))
        ctx_gap = max(0.0, static_sim - cos)
        out[cand.vid] = {
            "context_sim": cos,
            "context_supported": 1.0 if cos >= 0.80 else 0.0,
            "context_gap": ctx_gap,
        }
    return out


def _similar_to_pool_needs_conditioned_supplement(cands: List[LandingCandidate]) -> bool:
    """空池或极弱（≤2 条且最高分低于阈值）时允许 conditioned 少量补召回。"""
    if not cands:
        return True
    if len(cands) > STAGE2A_SIMILAR_TO_WEAK_MAX_N:
        return False
    best = max(float(getattr(c, "semantic_score", 0) or 0) for c in cands)
    return best < STAGE2A_SIMILAR_TO_WEAK_SIM


def _retrieve_conditioned_supplement_capped(
    label: Any,
    anchor: PreparedAnchor,
    similar_to_candidates: List[LandingCandidate],
    active_domain_set: Optional[Set[int]],
    jd_field_ids: Optional[Set[str]],
    jd_subfield_ids: Optional[Set[str]],
    jd_topic_ids: Optional[Set[str]],
    conditioned_top_k: Optional[int],
    cap: int = CONDITIONED_SUPPLEMENT_MAX,
) -> Tuple[List[LandingCandidate], Dict[int, float], str]:
    """
    仅在弱/空 similar_to 时调用；最多 cap 条；强向量先试，失败再轻句编码（与 2.1 backoff 同 spirit）。
    """
    if getattr(anchor, "conditioned_vec", None) is None:
        setattr(anchor, "conditioned_retrieval_mode", "supplement_skipped_no_vec")
        return [], {}, "supplement_skipped_no_vec"
    if not getattr(label, "vocab_index", None):
        return [], {}, "supplement_skipped_no_index"
    load_vocab_meta(label)
    strong_vec = np.asarray(anchor.conditioned_vec, dtype=np.float32).flatten()
    nbr, smap, sr_pre, sr_post = _faiss_conditioned_neighbors_once(
        label, anchor, strong_vec, similar_to_candidates,
        conditioned_top_k, active_domain_set, jd_field_ids, jd_subfield_ids, jd_topic_ids,
        "supplement_strong",
    )
    setattr(anchor, "_cond_strong_raw_hits", sr_pre)
    setattr(anchor, "_cond_strong_postfilter_hits", sr_post)
    setattr(anchor, "_cond_light_raw_hits", 0)
    setattr(anchor, "_cond_light_postfilter_hits", 0)
    nbr = nbr[:cap]
    mode = "supplement_strong" if nbr else "supplement_strong_empty"
    if not nbr:
        enc = getattr(label, "_query_encoder", None)
        lt = (getattr(anchor, "light_conditioned_text", None) or "").strip()
        at = (getattr(anchor, "anchor", "") or "").strip()
        if enc and lt and at and _conditioned_text_substantial_vs_anchor(at, lt):
            try:
                raw_lv, _ = enc.encode(lt)
                if raw_lv is not None:
                    lv = np.asarray(raw_lv, dtype=np.float32).flatten()
                    nbr2, smap2, lr_pre, lr_post = _faiss_conditioned_neighbors_once(
                        label, anchor, lv, similar_to_candidates,
                        conditioned_top_k, active_domain_set, jd_field_ids, jd_subfield_ids, jd_topic_ids,
                        "supplement_light",
                    )
                    setattr(anchor, "_cond_light_raw_hits", lr_pre)
                    setattr(anchor, "_cond_light_postfilter_hits", lr_post)
                    nbr = nbr2[:cap]
                    smap = smap2
                    mode = "supplement_light_backoff" if nbr else "supplement_light_empty"
            except Exception:
                pass
    if not nbr:
        setattr(anchor, "conditioned_retrieval_mode", mode)
        return [], {}, mode
    setattr(anchor, "conditioned_retrieval_mode", mode)
    return nbr, smap, mode


def _stage2a_enrich_light_judge_snapshot(snap: Dict[str, Any], bucket: str, c: Any) -> None:
    """轻量 meta（Stage3 / 全局一致性前置）；非终审桶。legacy 五层桶仍由上层写入。"""
    gr = float(getattr(c, "generic_risk", 0) or 0)
    pr = float(getattr(c, "polysemy_risk", 0) or 0)
    orisk = float(getattr(c, "object_like_risk", 0) or 0)
    if gr > 0.52 or pr > 0.55 or orisk > 0.45:
        snap["risk_level"] = "high"
    elif gr > 0.32 or pr > 0.30:
        snap["risk_level"] = "medium"
    else:
        snap["risk_level"] = "low"
    snap["is_core_like"] = bucket == "primary_expandable"
    snap["is_expand_suggested"] = bucket in ("primary_expandable", "primary_support_seed")


def collect_landing_candidates(
    label,
    anchor: PreparedAnchor,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
    jd_profile: Optional[Dict[str, Any]] = None,
    query_vector=None,
) -> List[LandingCandidate]:
    """
    Stage2A：主召回 similar_to；conditioned_vec 以点积为 similar_to 补局部证据；仅在空/弱池时少量 FAISS 补召回；
    family 仅空池弱保底（非主来源）。综述：候选生成优先，强裁决后移 Stage3。
    """
    # 1) 主召回 + conditioned 局部分数（非第二套主 FAISS）
    similar_to_candidates = retrieve_academic_term_by_similar_to(
        label, anchor,
        active_domain_set=active_domain_set,
        jd_field_ids=jd_field_ids,
        jd_subfield_ids=jd_subfield_ids,
        jd_topic_ids=jd_topic_ids,
        top_k=STAGE2A_COLLECT_BASE_TOP_K,
    )
    setattr(anchor, "conditioned_used_for_scoring", bool(getattr(anchor, "conditioned_vec", None) is not None))
    setattr(anchor, "conditioned_used_for_supplement", False)
    rerank_signals = _conditioned_rerank_signals_dot_product(label, anchor, similar_to_candidates)

    context_neighbors: List[LandingCandidate] = []
    if not _similar_to_pool_needs_conditioned_supplement(similar_to_candidates):
        setattr(anchor, "conditioned_retrieval_mode", "scoring_only_dot_product")
        setattr(anchor, "_cond_strong_raw_hits", 0)
        setattr(anchor, "_cond_strong_postfilter_hits", 0)
        setattr(anchor, "_cond_light_raw_hits", 0)
        setattr(anchor, "_cond_light_postfilter_hits", 0)
    if _similar_to_pool_needs_conditioned_supplement(similar_to_candidates):
        ctx_top_k = max(6, STAGE2A_COLLECT_CONDITIONED_TOP_K)
        context_neighbors, _, _sup_mode = _retrieve_conditioned_supplement_capped(
            label, anchor, similar_to_candidates,
            active_domain_set=active_domain_set,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
            conditioned_top_k=ctx_top_k,
            cap=CONDITIONED_SUPPLEMENT_MAX,
        )
        setattr(anchor, "conditioned_used_for_supplement", len(context_neighbors) > 0)
    # 2) family：仅「仍无任何候选」时的弱保底（弱边 / 证据后移，不当主召回）
    family_cands: List[LandingCandidate] = []
    if len(similar_to_candidates) + len(context_neighbors) == 0 and _is_canonical_academic_like_anchor(anchor):
        # family fallback 收严：只保线，不造新路线（最多 1~2 条；更严 gate 在 retrieve_family_landing_candidates 内）
        family_cands = retrieve_family_landing_candidates(
            label, anchor, top_k=min(2, STAGE2A_FAMILY_LANDING_TOP_K)
        )
        for c in family_cands:
            setattr(c, "family_fallback_only", True)
            setattr(c, "default_expand_block_reason", "family_fallback_no_expand")

    setattr(anchor, "_context_neighbors", context_neighbors)
    setattr(anchor, "_context_score_map", {c.vid: getattr(c, "context_sim", getattr(c, "semantic_score", 0.0)) for c in context_neighbors})
    for cand in similar_to_candidates:
        sig = rerank_signals.get(cand.vid, {})
        cand.context_sim = float(sig.get("context_sim", 0.0) or 0.0)
        cand.context_supported = bool(sig.get("context_supported", 0.0) >= 1.0)
        cand.context_gap = float(sig.get("context_gap", 1.0) or 1.0)
        cand.source_role = "seed_candidate"
    # 3) 按 tid merge，保留 surface_sim / conditioned_sim / source_set，merge 内已算 context_gain
    merged_list = merge_landing_candidates_by_tid(similar_to_candidates + context_neighbors + family_cands)
    n_merged = len(merged_list)
    dual_after_merge = sum(
        1 for c in merged_list
        if {"similar_to", "conditioned_vec"}.issubset(getattr(c, "source_set", None) or set())
    )
    for c in merged_list:
        if c.vid in rerank_signals:
            sig = rerank_signals[c.vid]
            c.context_sim = max(c.context_sim, float(sig.get("context_sim", 0) or 0))
            c.context_supported = c.context_supported or bool(sig.get("context_supported", 0.0) >= 1.0)
            c.context_gap = min(getattr(c, "context_gap", 1.0), float(sig.get("context_gap", 1.0) or 1.0))
    cands = merged_list
    for c in cands:
        c.domain_fit = _compute_domain_fit(
            label, c.vid,
            active_domain_set=active_domain_set,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
        )
        if getattr(c, "soft_domain_retain", False):
            c.domain_fit = (getattr(c, "domain_fit", 1.0) or 1.0) * 0.85
    # 在 candidate 上保留层级状态，供 primary 打分惩罚（topic=1.0, subfield=0.65, field=0.35, none=0.10）
    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))
    for c in cands:
        if not jd_f and not jd_s and not jd_t:
            setattr(c, "hierarchy_score", 1.0)
            setattr(c, "hierarchy_level", "missing")
            setattr(c, "hierarchy_reason", "")
        else:
            topic_row = _load_vocabulary_topic_stats(label, c.vid)
            if not topic_row:
                setattr(c, "hierarchy_score", 1.0)
                setattr(c, "hierarchy_level", "missing")
                setattr(c, "hierarchy_reason", "")
            else:
                score, level = _compute_hierarchy_match_score(topic_row, jd_f, jd_s, jd_t)
                setattr(c, "hierarchy_score", score)
                setattr(c, "hierarchy_level", level)
                setattr(c, "hierarchy_reason", "topic_hierarchy_no_match" if score <= 0 else "")
    if jd_profile:
        jd_profile_for_fit = {k: v for k, v in jd_profile.items() if k != "active_domains"}
        jd_profile_for_fit["active_subfields"] = set(jd_profile.get("active_subfields") or [])
        jd_profile_for_fit["active_topics"] = set(jd_profile.get("active_topics") or [])
        filtered = []
        for c in cands:
            snap = get_vocab_hierarchy_snapshot(label, c.vid)
            term_info = {
                "field_dist": snap.get("field_dist") or {},
                "subfield_dist": snap.get("subfield_dist") or {},
                "topic_dist": snap.get("topic_dist") or {},
                "domain_dist": snap.get("domain_dist") or {},
            }
            fit_info = compute_hierarchical_fit(term_info, jd_profile_for_fit)
            setattr(c, "fit_info", fit_info)
            setattr(c, "work_count", snap.get("work_count") or 0)
            setattr(c, "domain_span", snap.get("domain_span") or 0)
            setattr(c, "subfield_fit", fit_info.get("subfield_fit", 0))
            setattr(c, "topic_fit", fit_info.get("topic_fit", 0))
            setattr(c, "field_fit", float(fit_info.get("field_overlap", fit_info.get("domain_fit", 0)) or 0))
            setattr(c, "outside_subfield_mass", fit_info.get("outside_subfield_mass", 0))
            setattr(c, "outside_topic_mass", fit_info.get("outside_topic_mass", 0))
            setattr(c, "topic_entropy", 0.0)
            fit_info["subfield_dist"] = snap.get("subfield_dist") or {}
            fit_info["topic_dist"] = snap.get("topic_dist") or {}
            # 内联 landing 排序分：仅用于 top-m 排序，不依赖 hierarchy_guard 旧体系
            df = fit_info.get("domain_fit") or 0.5
            span = int(snap.get("domain_span") or 0)
            genericity = 1.0 / (1.0 + TOPIC_SPAN_PENALTY_FACTOR * max(0, span - 1))
            land_score = (c.semantic_score or 0) * (0.5 + 0.5 * df) * genericity
            setattr(c, "landing_score", land_score)
            # 零硬编码：不再用 subfield_fit/topic_fit/outside_subfield_mass 硬门槛一票否决；
            # 层级信息仅参与 landing_score，最终由 primary_score（含 jd_align、neighborhood、isolation）统一排序
            filtered.append(c)
        cands = filtered
    # 数据驱动：候选与 JD 整体语义对齐（无词表）
    if query_vector is not None:
        for c in cands:
            jd_align = _compute_jd_candidate_alignment(label, c.vid, query_vector)
            setattr(c, "jd_candidate_alignment", jd_align)
    else:
        for c in cands:
            setattr(c, "jd_candidate_alignment", 0.5)
    # 定性字段：收集 + 分型（family_type / generic_like / scene_shifted），不在此做选主
    anchor_term = getattr(anchor, "anchor", "") or ""
    anchor_type_opt = getattr(anchor, "anchor_type", None)
    jd_profile_for_risk = jd_profile or {}
    for rank, c in enumerate(cands, start=1):
        c.anchor_term = anchor_term
        c.anchor_vid = getattr(anchor, "vid", 0)
        c.source_type = (getattr(c, "source", "") or "similar_to").strip()
        c.source_rank = rank
        c.source_score = float(getattr(c, "semantic_score", 0) or 0)
        aid = compute_anchor_identity_score(
            anchor_term, c.term or "", anchor_type_opt,
            edge_strength=float(getattr(c, "semantic_score", 0) or 0),
            context_sim=float(getattr(c, "context_sim", 0) or 0),
        )
        setattr(c, "identity_gate", _identity_gate_from_score(aid))
        ctx_cont, ctx_local, ctx_co, ctx_jd = compute_context_continuity(
            c, jd_align=float(getattr(c, "jd_candidate_alignment", 0.5) or 0.5), co_anchor_support=0.0
        )
        c.context_continuity = ctx_cont
        c.context_local_support = ctx_local
        c.context_co_anchor_support = ctx_co
        c.context_jd_support = ctx_jd
        c.hierarchy_consistency = float(getattr(c, "hierarchy_score", 0.5) or 0.5)
        c.polysemy_risk, c.polysemy_note = compute_polysemy_risk(c.term or "", aid)
        c.object_like_risk, c.object_like_note = compute_object_like_risk_with_note({"term": c.term or ""})
        gr, gn = compute_generic_risk_with_note(
            {
                "term": c.term or "",
                "hierarchy": c.hierarchy_consistency,
                "jd_align": float(getattr(c, "jd_candidate_alignment", 0.5) or 0.5),
            },
            jd_profile_for_risk,
        )
        c.generic_risk = gr
        c.generic_note = gn
        # Stage2A 终稿：候选分型 + 四段式 identity
        c.ctx_supported = c.context_supported
        c.ctx_gap = max(0.0, float(getattr(c, "semantic_score", 0) or 0) - float(getattr(c, "context_sim", 0) or 0))
        c.family_type = classify_candidate_family(
            anchor_term, c.term or "", c.semantic_score, float(getattr(c, "context_sim", 0) or 0)
        )
        c.generic_like = is_candidate_generic_like(c)
        c.scene_shifted = is_candidate_scene_shifted(c, jd_profile_for_risk)
        c.expand_block_reason = infer_expand_block_reason(c)
        c.source_trust = get_source_trust(c.source_type)
        score_academic_identity(c)
        if getattr(c, "has_family_evidence", False):
            prev = float(getattr(c, "anchor_identity_score", 0) or 0)
            setattr(c, "anchor_identity_score", max(prev, prev + 0.08))
            reasons = getattr(c, "primary_eligibility_reasons", None) or []
            if not isinstance(reasons, list):
                reasons = []
            reasons.append("family_landing_support")
            setattr(c, "primary_eligibility_reasons", reasons)
    if LABEL_EXPANSION_DEBUG:
        n_raw_sim = len(similar_to_candidates)
        n_raw_ctx = len(context_neighbors)
        dual_final = sum(
            1 for c in cands
            if {"similar_to", "conditioned_vec"}.issubset(getattr(c, "source_set", None) or set())
        )
        print(
            f"[Stage2A candidate_generation_summary] anchor={anchor.anchor!r} "
            f"similar_to_count={n_raw_sim} conditioned_supplement_count={n_raw_ctx} "
            f"family_fallback_count={len(family_cands)} final_candidate_count={len(cands)} "
            f"conditioned_used_for_scoring={getattr(anchor, 'conditioned_used_for_scoring', False)} "
            f"conditioned_used_for_supplement={getattr(anchor, 'conditioned_used_for_supplement', False)}"
        )
        print(
            f"[Stage2A landing source_distribution] anchor={anchor.anchor!r} "
            f"similar_to={n_raw_sim} conditioned_vec={n_raw_ctx} merged_after_tid={n_merged} "
            f"dual_source_after_merge={dual_after_merge} final_pool={len(cands)} dual_in_final={dual_final}"
        )
        print(
            f"[Stage2A landing anchor_ctx] conditioning_mode={getattr(anchor, 'conditioning_mode', '')!r} "
            f"conditioned_is_fallback={getattr(anchor, 'conditioned_is_fallback', False)} "
            f"surface_conditioned_cosine={getattr(anchor, 'surface_conditioned_cosine', None)} "
            f"conditioned_retrieval_mode={getattr(anchor, 'conditioned_retrieval_mode', '')!r} "
            f"strong_raw/post={getattr(anchor, '_cond_strong_raw_hits', None)}/{getattr(anchor, '_cond_strong_postfilter_hits', None)} "
            f"light_raw/post={getattr(anchor, '_cond_light_raw_hits', None)}/{getattr(anchor, '_cond_light_postfilter_hits', None)}"
        )
        print(
            f"[Stage2A landing merge_counts] raw_similar_to={n_raw_sim} raw_conditioned_vec={n_raw_ctx} "
            f"merged_candidate_count={n_merged}"
        )
        for i, c in enumerate(cands[:3], 1):
            _sset = getattr(c, "source_set", None) or {getattr(c, "source", "") or ""} - {""}
            print(
                f"  [top{i}] term={c.term!r} source_set={_sset or getattr(c, 'source', '')} "
                f"surface_sim={getattr(c, 'surface_sim', None)} "
                f"conditioned_sim={getattr(c, 'conditioned_sim', None)} "
                f"context_gain={getattr(c, 'context_gain', None)} "
                f"conditioned_is_fallback={getattr(anchor, 'conditioned_is_fallback', False)} "
                f"conditioned_retrieval_tier={getattr(c, 'conditioned_retrieval_tier', None)}"
            )
    return cands


def _compute_neighborhood_and_isolation(label, flat_pool: List[Tuple[Any, LandingCandidate]]) -> None:
    """数据驱动：为每个候选计算与其它候选的邻域一致性及语义离群惩罚，无词表。原地写入 c.neighborhood_consistency、c.semantic_isolation_penalty。"""
    if not flat_pool or getattr(label, "vocab_to_idx", None) is None or getattr(label, "all_vocab_vectors", None) is None:
        for _, c in flat_pool:
            setattr(c, "neighborhood_consistency", 0.5)
            setattr(c, "semantic_isolation_penalty", 0.0)
        return
    vecs = {}
    for _, c in flat_pool:
        vid = c.vid
        if vid in vecs:
            continue
        idx = label.vocab_to_idx.get(str(vid))
        if idx is None:
            vecs[vid] = None
            continue
        try:
            v = np.asarray(label.all_vocab_vectors[idx], dtype=np.float32).flatten()
            vecs[vid] = v
        except Exception:
            vecs[vid] = None
    for _, c in flat_pool:
        v = vecs.get(c.vid)
        if v is None or v.size == 0:
            setattr(c, "neighborhood_consistency", 0.5)
            setattr(c, "semantic_isolation_penalty", 0.0)
            continue
        sims = []
        for (_, c2) in flat_pool:
            if c2.vid == c.vid:
                continue
            v2 = vecs.get(c2.vid)
            if v2 is None or v2.size != v.size:
                continue
            try:
                s = float(np.dot(v, v2))
                s = max(-1.0, min(1.0, s))
                sims.append(s)
            except Exception:
                pass
        if not sims:
            setattr(c, "neighborhood_consistency", 0.5)
            setattr(c, "semantic_isolation_penalty", 0.0)
            continue
        mean_sim = float(np.mean(sims))
        mean_sim = max(-1.0, min(1.0, mean_sim))
        setattr(c, "neighborhood_consistency", max(0.0, mean_sim))
        setattr(c, "semantic_isolation_penalty", max(0.0, 1.0 - mean_sim))


def _compute_conditioned_anchor_align_and_multi_anchor_support(
    label,
    flat_pool: List[Tuple[Any, LandingCandidate]],
    prepared_anchors: List[PreparedAnchor],
) -> None:
    """
    为每个候选设置 conditioned_anchor_align（与当前锚点条件化向量相似度）与 multi_anchor_support
    （与其它锚点条件化向量平均相似度）。无词表，纯向量与图结构。
    """
    if getattr(label, "vocab_to_idx", None) is None or getattr(label, "all_vocab_vectors", None) is None:
        for _, c in flat_pool:
            setattr(c, "conditioned_anchor_align", None)
            setattr(c, "multi_anchor_support", 0.5)
        return
    # 候选向量缓存
    vecs: Dict[int, Optional[np.ndarray]] = {}
    for _, c in flat_pool:
        if c.vid in vecs:
            continue
        idx = label.vocab_to_idx.get(str(c.vid))
        if idx is None:
            vecs[c.vid] = None
            continue
        try:
            v = np.asarray(label.all_vocab_vectors[idx], dtype=np.float32).flatten()
            vecs[c.vid] = v
        except Exception:
            vecs[c.vid] = None
    # 其它锚点条件化向量（按 vid）
    other_conditioned: Dict[int, np.ndarray] = {}
    for a in prepared_anchors:
        if a.conditioned_vec is None:
            continue
        try:
            v = np.asarray(a.conditioned_vec, dtype=np.float32).flatten()
            if v.size > 0:
                other_conditioned[a.vid] = v
        except Exception:
            pass
    for anchor, c in flat_pool:
        c_vec = vecs.get(c.vid)
        if c_vec is None or c_vec.size == 0:
            setattr(c, "conditioned_anchor_align", None)
            setattr(c, "multi_anchor_support", 0.5)
            continue
        # 当前锚点条件化对齐
        cond_align = None
        if getattr(anchor, "conditioned_vec", None) is not None:
            try:
                cv = np.asarray(anchor.conditioned_vec, dtype=np.float32).flatten()
                if cv.size == c_vec.size:
                    sim = float(np.dot(c_vec, cv))
                    sim = max(-1.0, min(1.0, sim))
                    cond_align = 0.5 + 0.5 * max(0.0, sim)
            except Exception:
                pass
        setattr(c, "conditioned_anchor_align", cond_align)
        # 多锚支持：与其它锚点条件化向量的平均相似度
        support_sims = []
        for a2_vid, a2_vec in other_conditioned.items():
            if a2_vid == anchor.vid:
                continue
            if a2_vec.size != c_vec.size:
                continue
            try:
                s = float(np.dot(c_vec, a2_vec))
                s = max(-1.0, min(1.0, s))
                support_sims.append(0.5 + 0.5 * max(0.0, s))
            except Exception:
                pass
        multi_support = float(np.mean(support_sims)) if support_sims else 0.5
        setattr(c, "multi_anchor_support", max(0.0, min(1.0, multi_support)))


def score_academic_identity(c: Any) -> float:
    """
    Stage2A 终稿：四段式 identity（边语义 + 词形 + 上下文支持 + 家族中心性），
    不再由 embedding 一项主导；写入 identity_breakdown 与 anchor_identity_score。
    """
    edge_sem = float(getattr(c, "semantic_score", 0) or 0)
    anchor_term = getattr(c, "anchor_term", "") or ""
    candidate_term = (getattr(c, "term", "") or "").strip()
    lexical = lexical_shape_match(anchor_term, candidate_term)
    ctx_sim = float(getattr(c, "context_sim", 0) or 0)
    ctx_ok = getattr(c, "ctx_supported", getattr(c, "context_supported", False))
    context_support = ctx_sim if ctx_ok else 0.0
    family_centeredness = estimate_family_centeredness(c)
    identity = (
        0.45 * edge_sem
        + 0.25 * lexical
        + 0.20 * context_support
        + 0.10 * family_centeredness
    )
    identity = max(0.0, min(1.0, identity))
    if hasattr(c, "anchor_identity_score"):
        c.anchor_identity_score = identity
    if hasattr(c, "identity_score"):
        c.identity_score = identity
    setattr(c, "identity_breakdown", {
        "edge_semantic": edge_sem,
        "lexical_shape": lexical,
        "context_support": context_support,
        "family_centeredness": family_centeredness,
    })
    return identity


# ---------- Stage2B：学术侧补充（dense / 簇 / 共现，不再用 SIMILAR_TO 学术→学术） ----------

def _stage2b_top_blocked_reasons(blocked_counts: Dict[str, int], top_k: int = 6) -> List[Tuple[str, int]]:
    """Stage2B 审计：取 blocked reasons 的 top-k（仅用于日志/观测）。"""
    if not blocked_counts:
        return []
    return sorted(blocked_counts.items(), key=lambda x: (-int(x[1] or 0), x[0]))[:top_k]


def _stage2b_write_expansion_audit(label, chain: str, audit: Dict[str, Any]) -> None:
    """
    将 Stage2B 扩展链路审计信息挂到 label 上，供主流程汇总打印。
    不改变扩展链路返回类型；仅用于解释 kept=0 的原因。
    """
    try:
        store = getattr(label, "_stage2b_last_expansion_audit", None)
        if not isinstance(store, dict):
            store = {}
            setattr(label, "_stage2b_last_expansion_audit", store)
        store[chain] = audit
    except Exception:
        return



def expand_from_vocab_dense_neighbors(
    label,
    primary_landings: List[PrimaryLanding],
    top_k_per_primary: int = None,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
    jd_profile: Optional[Dict[str, Any]] = None,
) -> List[ExpandedTermCandidate]:
    """从词汇向量索引取 primary 的学术近邻；须过 support_expandable_for_anchor + **按 seed tier 分档**的 dense 闸：
    strong：略松检索 cap、min_dense_sim=0.80、anchor/family/context_stability 阈值见实现；
    weak：max_keep=1、min_dense_sim=0.84、一致性阈值更严，且不用 weak_support_release（防弱 seed 带脏邻）。"""
    top_k_per_primary = top_k_per_primary or DENSE_MAX_PER_PRIMARY
    if not primary_landings or not getattr(label, "vocab_index", None) or not getattr(label, "vocab_to_idx", None):
        return []
    if getattr(label, "all_vocab_vectors", None) is None:
        return []
    load_vocab_meta(label)
    seen = set(p.vid for p in primary_landings)
    out = []
    support_domain_min = max(SUPPORT_MIN_DOMAIN_FIT, 0.72)

    anchor_to_primaries: Dict[int, List[Any]] = {}
    # --- Stage2B expansion audit (aggregated for this call) ---
    _audit_raw = 0
    _audit_post = 0
    _audit_kept = 0
    _audit_blocked: Dict[str, int] = {}
    _audit_blocked_samples: Dict[str, List[str]] = {}
    _audit_no_domain_weak_detail: List[str] = []
    def _blk(reason: str) -> None:
        _audit_blocked[reason] = _audit_blocked.get(reason, 0) + 1
    def _blk_sample(reason: str, sample: str, cap: int = 3) -> None:
        """每类 reason 仅保留少量样本，避免刷屏。"""
        lst = _audit_blocked_samples.get(reason)
        if lst is None:
            lst = []
            _audit_blocked_samples[reason] = lst
        if len(lst) >= cap:
            return
        if sample not in lst:
            lst.append(sample)
    def _cap_append(lst: List[str], item: str, cap: int = 3) -> None:
        if len(lst) >= cap:
            return
        if item not in lst:
            lst.append(item)

    for p in primary_landings:
        a_vid = getattr(p, "anchor_vid", 0)
        anchor_to_primaries.setdefault(a_vid, []).append(p)

    for p in primary_landings:
        # 资格已在 stage2_generate_academic_terms 主循环判过；此处只复核 ok，关闭 seed factors 避免重复打印
        ok, seed_score, block_reason = check_seed_eligibility(
            label, p, jd_profile, emit_seed_factors=False,
        )
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG:
            anchor_term = getattr(p, "anchor_term", "") or ""
            term = getattr(p, "term", "") or ""
            identity = float(getattr(p, "identity_score", 0) or getattr(p, "anchor_identity_score", 0) or 0)
            src = (getattr(p, "source", "") or "").strip().lower()
            trusted_set = {s.strip().lower() for s in (TRUSTED_SOURCE_TYPES_FOR_DIFFUSION or []) if s}
            trusted_source = (src in trusted_set) if trusted_set else True
            identity_ok = identity >= SEED_MIN_IDENTITY
            print(
                f"[Stage2B seed gate] anchor={anchor_term!r} term={term!r} | can_expand={getattr(p, 'can_expand', False)} "
                f"identity={identity:.3f} >= {SEED_MIN_IDENTITY}? {identity_ok} trusted_source={trusted_source} "
                f"seed_ok={ok} seed_block_reason={block_reason!r}"
            )
        if not ok:
            _blk(f"seed_ineligible:{block_reason or 'blocked'}")
            continue
        sem_p = float(getattr(p, "semantic_score", 0.0) or 0.0)
        cond_p = float(getattr(p, "conditioned_sim", 0.0) or 0.0)
        tier_dn = (getattr(p, "stage2b_seed_tier", None) or "").strip().lower()
        if tier_dn == "weak":
            top_k_per_primary = min(top_k_per_primary, DENSE_MAX_PER_PRIMARY_WEAK)
        strong_seed = (tier_dn != "weak") and (seed_score >= 0.68 or (sem_p >= 0.83 and cond_p >= 0.80))
        if tier_dn == "weak":
            max_keep = 1
            min_dense_sim = 0.84
            min_anchor_consistency = 0.78
            min_family_support = 0.74
        else:
            max_keep = 3 if strong_seed else 2
            min_dense_sim = 0.80
            min_anchor_consistency = 0.72
            min_family_support = 0.68
        idx = label.vocab_to_idx.get(str(p.vid))
        if idx is None:
            _blk("seed_vid_not_in_vocab_index")
            continue
        vec = np.asarray(label.all_vocab_vectors[idx], dtype=np.float32).reshape(1, -1)
        k = min(top_k_per_primary + 8, 24)
        scores, ids = label.vocab_index.search(vec, k)
        kept = 0
        anchor_term = getattr(p, "anchor_term", "") or ""

        anchor_stub = SimpleNamespace(
            anchor_term=anchor_term,
            anchor_vid=getattr(p, "anchor_vid", 0),
            conditioned_vec=getattr(p, "anchor_conditioned_vec", None),
            _context_score_map=getattr(p, "_context_score_map", None) or {},
            _context_neighbors=getattr(p, "_context_neighbors", None) or [],
            _stage2b_anchor_primaries=anchor_to_primaries.get(getattr(p, "anchor_vid", 0), []),
        )

        def _dense_neighbor_support_scores(tid_local: int) -> Optional[Dict[str, Any]]:
            """与 support_expandable_for_anchor 同源中间量，仅供强 seed 弱放行（不提高函数出口数量）。"""
            candidate_vec = _get_vocab_vec(label, tid_local)
            if candidate_vec is None:
                return None
            primary_vid = getattr(p, "vid", None)
            if primary_vid is None:
                return None
            primary_vec = _get_vocab_vec(label, int(primary_vid))
            if primary_vec is None:
                return None
            primary_consistency = _cos_sim(primary_vec, candidate_vec)
            conditioned = getattr(anchor_stub, "conditioned_vec", None)
            anchor_consistency = 0.0
            has_anchor_vec = conditioned is not None
            if has_anchor_vec:
                try:
                    cv = np.asarray(conditioned, dtype=np.float32).flatten()
                    if cv.size > 0:
                        anchor_consistency = _cos_sim(cv, candidate_vec)
                except Exception:
                    pass
            else:
                anchor_consistency = 1.0
            context_stability = _estimate_context_support(
                label=label,
                anchor=anchor_stub,
                candidate_vid=tid_local,
                candidate_vec=candidate_vec,
            )
            ctx_map = getattr(anchor_stub, "_context_score_map", None) or {}
            ctx_neighbors = getattr(anchor_stub, "_context_neighbors", None) or []
            has_context_data = bool(ctx_neighbors) or (tid_local in ctx_map)
            if not has_context_data:
                context_stability = 1.0
            surviving_primaries = getattr(anchor_stub, "_stage2b_anchor_primaries", None) or []
            family_support = _estimate_anchor_family_support(
                label=label,
                candidate_vec=candidate_vec,
                surviving_primaries=surviving_primaries,
            )
            if not surviving_primaries:
                family_support = 1.0
            return {
                "primary_consistency": primary_consistency,
                "anchor_consistency": anchor_consistency,
                "context_stability": context_stability,
                "family_support": family_support,
                "has_anchor_vec": has_anchor_vec,
            }

        n_raw = n_src = n_sim = n_dom = n_dom_fit = n_supp = n_post_mainline = 0
        top_raw: List[str] = []
        top_kept: List[str] = []

        for score, tid in zip(scores[0], ids[0]):
            tid = int(tid)
            if tid <= 0:
                _blk("retrieval_empty_or_invalid_tid")
                continue
            n_raw += 1
            if tid in seen:
                _blk("duplicate_or_in_carryover")
                continue
            meta = label._vocab_meta.get(tid, ("", ""))
            term = (meta[0] or "").strip() or str(tid)
            vocab_type = meta[1] or ""
            if len(top_raw) < 5:
                top_raw.append(term)
            if vocab_type not in ("concept", "keyword", "") and vocab_type:
                _blk("vocab_type_filtered")
                _blk_sample("vocab_type_filtered", f"{term[:36]!r}:{str(vocab_type)[:16]}")
                continue
            n_src += 1
            sim = max(0.0, min(1.0, float(score)))
            if sim < min_dense_sim:
                _blk("sim_below_min_dense_sim")
                _blk_sample("sim_below_min_dense_sim", f"{term[:36]!r}:sim={sim:.3f}<min={min_dense_sim:.2f}")
                continue
            n_sim += 1
            if active_domain_set is not None or jd_field_ids or jd_subfield_ids or jd_topic_ids:
                if not _term_in_active_domains(
                    label, tid,
                    active_domain_set=active_domain_set,
                    jd_field_ids=jd_field_ids,
                    jd_subfield_ids=jd_subfield_ids,
                    jd_topic_ids=jd_topic_ids,
                ):
                    _blk("no_domain_or_topic_fit")
                    _blk_sample("no_domain_or_topic_fit", f"{term[:44]!r}")
                    # weak-seed domain/topic gate audit (keep lightweight; do NOT change gate behavior)
                    if tier_dn == "weak":
                        try:
                            df = _compute_domain_fit(
                                label, tid,
                                active_domain_set=active_domain_set,
                                jd_field_ids=jd_field_ids,
                                jd_subfield_ids=jd_subfield_ids,
                                jd_topic_ids=jd_topic_ids,
                            )
                            ta, tl, tc = _attach_topic_align(
                                label,
                                int(tid),
                                set(str(x) for x in (jd_field_ids or [])),
                                set(str(x) for x in (jd_subfield_ids or [])),
                                set(str(x) for x in (jd_topic_ids or [])),
                            )
                            sample = (
                                f"term={term[:38]!r} sim={sim:.3f} vocab_type={str(vocab_type)[:12]!r} "
                                f"domain_fit={float(df or 0.0):.3f} topic_align={float(ta or 0.0):.3f} "
                                f"topic_level={str(tl)[:10]!r} topic_conf={float(tc or 0.0):.2f}"
                            )
                        except Exception:
                            sample = (
                                f"term={term[:38]!r} sim={sim:.3f} vocab_type={str(vocab_type)[:12]!r} "
                                f"domain_fit=? topic_align=?"
                            )
                        _cap_append(_audit_no_domain_weak_detail, sample, cap=3)
                    continue
            n_dom += 1
            domain_fit = _compute_domain_fit(
                label, tid,
                active_domain_set=active_domain_set,
                jd_field_ids=jd_field_ids,
                jd_subfield_ids=jd_subfield_ids,
                jd_topic_ids=jd_topic_ids,
            )
            domain_fit_floor = 0.58 if strong_seed else support_domain_min
            if domain_fit < domain_fit_floor:
                _blk("domain_fit_too_low")
                continue
            n_dom_fit += 1

            keep, keep_meta = support_expandable_for_anchor(
                label=label,
                anchor=anchor_stub,
                parent_primary=p,
                candidate_vid=tid,
                candidate_term=term,
            )
            # 弱 seed 禁止 weak_support_release，仅靠四门 + 下方 tier 一致性闸。
            if not keep and strong_seed and tier_dn != "weak":
                wm = _dense_neighbor_support_scores(tid)
                ac_w = float(wm.get("anchor_consistency", 0.0) or 0.0) if wm else 0.0
                fs_w = float(wm.get("family_support", 0.0) or 0.0) if wm else 0.0
                weak_support_ok = (
                    wm is not None
                    and bool(wm.get("has_anchor_vec"))
                    and sim >= 0.72
                    and domain_fit >= 0.58
                    and ac_w >= 0.52
                    and fs_w >= 0.45
                    and not is_device_or_object_term(term)
                    and not _is_bad_support_for_anchor(anchor_term, term)
                )
                if weak_support_ok:
                    keep = True
                    keep_meta = dict(keep_meta)
                    parent_ps = float(getattr(p, "primary_score", 1.0) or 1.0)
                    keep_meta["keep_score"] = min(
                        0.92 * parent_ps,
                        0.55 * sim + 0.25 * domain_fit + 0.20 * ac_w,
                    )
                    keep_meta["reason"] = "weak_support_release_for_strong_seed"
            if keep:
                wm_gate = _dense_neighbor_support_scores(tid)
                if not wm_gate:
                    keep = False
                    keep_meta = {"reason": "dense_tier_gate_no_support_scores"}
                else:
                    ac_g = float(wm_gate.get("anchor_consistency", 0.0) or 0.0)
                    fs_g = float(wm_gate.get("family_support", 0.0) or 0.0)
                    cs_g = float(wm_gate.get("context_stability", 0.0) or 0.0)
                    if bool(wm_gate.get("has_anchor_vec")) and ac_g < min_anchor_consistency:
                        keep = False
                        keep_meta = {"reason": "dense_tier_anchor_consistency_low"}
                    elif fs_g < min_family_support:
                        keep = False
                        keep_meta = {"reason": "dense_tier_family_support_low"}
                    elif cs_g < 0.72:
                        keep = False
                        keep_meta = {"reason": "dense_tier_context_stability_low"}
            if not keep:
                # 细因见 keep_meta['reason']（若有）；这里先归为稳定粗类，避免大量原因键碎片化
                _blk(str(keep_meta.get("reason") or "support_gate_blocked"))
                if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
                    ac = keep_meta.get("anchor_consistency")
                    fs = keep_meta.get("family_support")
                    ac_val = float(ac) if ac is not None else 0.0
                    fs_val = float(fs) if fs is not None else 0.0
                    family_ok = fs is not None and float(fs) >= 0.68
                    print(
                        f"[Stage2B dense reject] anchor={anchor_stub.anchor_term!r} parent={getattr(p, 'term', '')!r} cand={term!r} | "
                        f"cand_sim={sim:.3f} anchor_consistency={ac_val:.3f} family_support={fs_val:.3f} family_ok={family_ok} reason={keep_meta.get('reason')!r}"
                    )
                continue
            n_supp += 1

            if is_device_or_object_term(term) and not anchor_allows_device_expansion(anchor_term):
                continue
            domain_span = 0
            if getattr(label, "stats_conn", None):
                row = label.stats_conn.execute(
                    "SELECT domain_span FROM vocabulary_domain_stats WHERE voc_id=?",
                    (tid,),
                ).fetchone()
                if row:
                    domain_span = int(row[0] or 0)
            if domain_span > DOMAIN_SPAN_EXTREME:
                _blk("domain_span_extreme")
                continue
            n_post_mainline += 1

            if kept >= max_keep:
                _blk("max_keep_reached")
                continue
            seen.add(tid)
            keep_score = float(keep_meta.get("keep_score", sim))
            parent_primary_score = float(getattr(p, "primary_score", 1.0) or 1.0)
            keep_score = min(keep_score, parent_primary_score * DENSE_PARENT_CAP)
            _dense_c = ExpandedTermCandidate(
                vid=tid,
                term=term,
                term_role="dense_expansion",
                identity_score=keep_score,
                source="dense",
                anchor_vid=p.anchor_vid,
                anchor_term=p.anchor_term,
                semantic_score=sim,
                src_vids=[p.vid],
                hit_count=1,
                parent_primary=p.term,
            )
            setattr(
                _dense_c,
                "parent_anchor_final_score",
                float(getattr(p, "anchor_final_score", 0.0) or 0.0),
            )
            setattr(
                _dense_c,
                "parent_anchor_step2_rank",
                int(getattr(p, "anchor_step2_rank", 999) or 999),
            )
            out.append(_dense_c)
            kept += 1
            if len(top_kept) < 5:
                top_kept.append(term)
            if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
                print(
                    f"[stage2b_expanded] tid={tid} term={term!r} source_type=dense "
                    f"parent_anchor={p.anchor_term!r} parent_primary={p.term!r} "
                    f"score={sim:.3f} keep_score={keep_score:.3f}"
                )

        # aggregate per-seed funnel into call-level audit
        _audit_raw += int(n_raw)
        _audit_post += int(n_post_mainline)
        _audit_kept += int(kept)

        if LABEL_EXPANSION_DEBUG and STAGE2_RULING_DEBUG:
            seed_term = getattr(p, "term", "") or ""
            print(f"\n{'-' * 80}\n[Stage2B] Dense expansion funnel\n{'-' * 80}")
            print(f"[Stage2B dense debug] seed={seed_term!r}")
            print(f"  raw_neighbors={n_raw}")
            print(f"  after_source_filter={n_src}")
            print(f"  after_similarity_filter={n_sim}")
            print(f"  after_domain_filter={n_dom}")
            print(f"  after_domain_fit={n_dom_fit}")
            print(f"  after_mainline_filter={n_supp}")
            print(f"  after_device_domain_span={n_post_mainline}")
            print(f"  final_kept={kept}")
            print(f"  top_raw={top_raw}")
            print(f"  top_kept={top_kept}")
    if LABEL_EXPANSION_DEBUG and STAGE2_NOISY_DEBUG:
        sample = [c.term for c in out[:3]]
        print(f"[Stage2B] expand_from_vocab_dense_neighbors primary数={len(primary_landings)} -> dense_expansion {len(out)} 个 前3: {sample}")

    _stage2b_write_expansion_audit(
        label,
        "dense",
        {
            "raw": int(_audit_raw),
            "post": int(_audit_post),
            "kept": int(_audit_kept),
            "blocked_top": _stage2b_top_blocked_reasons(_audit_blocked),
            "blocked_samples": dict(_audit_blocked_samples),
            "no_domain_weak_detail": list(_audit_no_domain_weak_detail),
        },
    )
    return out


def expand_from_cluster_members(
    label,
    primary_landings: List[PrimaryLanding],
    max_per_primary: int = None,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> List[ExpandedTermCandidate]:
    """Cluster 扩散当前默认关闭（全关），避免脏簇成员混入。term_role=cluster_expansion。

    Stage2B 审计说明：当 eligible seeds > 0 但 cluster 仍为 0 时，首先检查是否因为此处关闭。
    """
    _stage2b_write_expansion_audit(
        label,
        "cluster",
        {
            "raw": 0,
            "post": 0,
            "kept": 0,
            "blocked_top": [("disabled_by_default", 1)],
        },
    )
    return []


def expand_from_cooccurrence_support(
    label,
    primary_landings: List[PrimaryLanding],
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
    jd_profile: Optional[Dict[str, Any]] = None,
) -> List[ExpandedTermCandidate]:
    """共现支持词；仅允许强 normal + 多锚 + 高 jd_align 的 seed，support 须过 support_expandable_for_anchor，每 seed 最多 2 条。"""
    # --- Stage2B expansion audit (aggregated for this call) ---
    _audit_raw = 0
    _audit_post = 0
    _audit_kept = 0
    _audit_blocked: Dict[str, int] = {}
    def _blk(reason: str) -> None:
        _audit_blocked[reason] = _audit_blocked.get(reason, 0) + 1

    if not primary_landings or not getattr(label, "stats_conn", None):
        _blk("stats_conn_missing_or_no_primary_landings")
        _stage2b_write_expansion_audit(
            label,
            "cooc",
            {
                "raw": 0,
                "post": 0,
                "kept": 0,
                "blocked_top": _stage2b_top_blocked_reasons(_audit_blocked),
            },
        )
        return []
    load_vocab_meta(label)
    term_to_vid = {}
    if getattr(label, "_vocab_meta", None):
        for v, (t, _) in label._vocab_meta.items():
            if t and t not in term_to_vid:
                term_to_vid[t.strip()] = v
    strong_primaries = [
        p for p in primary_landings
        if getattr(p, "retain_mode", "normal") == "normal"
        and float(getattr(p, "primary_score", 0) or 0) >= 0.70
        and float(getattr(p, "jd_align", 0) or 0) >= 0.82
        and int(getattr(p, "cross_anchor_support_count", 1) or 1) >= 2
    ]
    anchor_to_primaries: Dict[int, List[Any]] = {}
    for p in primary_landings:
        a_vid = getattr(p, "anchor_vid", 0)
        anchor_to_primaries.setdefault(a_vid, []).append(p)
    out = []
    seen = set(p.vid for p in primary_landings)
    cooc_min_freq = max(COOC_SUPPORT_MIN_FREQ, 2)
    for p in strong_primaries:
        term = getattr(p, "term", "") or ""
        if not term:
            continue
        try:
            rows = label.stats_conn.execute(
                "SELECT term_a, term_b, freq FROM vocabulary_cooccurrence WHERE (term_a = ? OR term_b = ?) AND freq >= ?",
                (term, term, cooc_min_freq),
            ).fetchall()
        except Exception:
            _blk("stats_query_failed")
            continue
        anchor_stub = SimpleNamespace(
            anchor_term=getattr(p, "anchor_term", "") or "",
            anchor_vid=getattr(p, "anchor_vid", 0),
            conditioned_vec=getattr(p, "anchor_conditioned_vec", None),
            _context_score_map=getattr(p, "_context_score_map", None) or {},
            _context_neighbors=getattr(p, "_context_neighbors", None) or [],
            _stage2b_anchor_primaries=anchor_to_primaries.get(getattr(p, "anchor_vid", 0), []),
        )
        kept = 0
        raw_cooc = len(rows)
        n_freq3 = n_dom = n_dom_fit = n_supp = n_span_ok = 0
        top_raw_co: List[str] = []
        top_kept_co: List[str] = []
        for row in rows:
            ta, tb, freq = row[0], row[1], row[2]
            other = (tb if ta == term else ta) or ""
            if other == term or not other:
                _blk("retrieval_empty_or_self")
                continue
            if len(top_raw_co) < 5:
                top_raw_co.append((other.strip() or "")[:48])
            vid_other = term_to_vid.get(other.strip())
            if vid_other is None:
                _blk("term_to_vid_missing")
                continue
            if vid_other in seen:
                _blk("duplicate_or_in_carryover")
                continue
            if (freq or 0) < 3:
                _blk("freq_below_min")
                continue
            n_freq3 += 1
            cooc_strength = min(1.0, float(freq or 0) / 5.0)
            if active_domain_set is not None or jd_field_ids or jd_subfield_ids or jd_topic_ids:
                if not _term_in_active_domains(
                    label, vid_other,
                    active_domain_set=active_domain_set,
                    jd_field_ids=jd_field_ids,
                    jd_subfield_ids=jd_subfield_ids,
                    jd_topic_ids=jd_topic_ids,
                ):
                    _blk("no_domain_or_topic_fit")
                    continue
            n_dom += 1
            domain_fit = _compute_domain_fit(
                label, vid_other,
                active_domain_set=active_domain_set,
                jd_field_ids=jd_field_ids,
                jd_subfield_ids=jd_subfield_ids,
                jd_topic_ids=jd_topic_ids,
            )
            if domain_fit < 0.75:
                _blk("domain_fit_too_low")
                continue
            n_dom_fit += 1
            keep, _ = support_expandable_for_anchor(
                label=label,
                anchor=anchor_stub,
                parent_primary=p,
                candidate_vid=int(vid_other),
                candidate_term=other,
            )
            if not keep:
                _blk("support_gate_blocked")
                continue
            n_supp += 1
            domain_span = 0
            if getattr(label, "stats_conn", None):
                r = label.stats_conn.execute(
                    "SELECT domain_span FROM vocabulary_domain_stats WHERE voc_id=?",
                    (vid_other,),
                ).fetchone()
                if r:
                    domain_span = int(r[0] or 0)
            if domain_span > DOMAIN_SPAN_EXTREME:
                _blk("domain_span_extreme")
                continue
            n_span_ok += 1
            if kept >= 2:
                _blk("max_keep_reached")
                continue
            seen.add(vid_other)
            meta = label._vocab_meta.get(vid_other, ("", ""))
            kt = (meta[0] or other or "")[:48]
            _cooc_c = ExpandedTermCandidate(
                vid=vid_other,
                term=meta[0] or other,
                term_role="cooc_expansion",
                identity_score=cooc_strength,
                source="cooc",
                anchor_vid=getattr(p, "anchor_vid", 0),
                anchor_term=getattr(p, "anchor_term", "") or "",
                semantic_score=cooc_strength,
                src_vids=[getattr(p, "vid", 0)],
                hit_count=int(freq),
                parent_primary=term,
            )
            setattr(
                _cooc_c,
                "parent_anchor_final_score",
                float(getattr(p, "anchor_final_score", 0.0) or 0.0),
            )
            setattr(
                _cooc_c,
                "parent_anchor_step2_rank",
                int(getattr(p, "anchor_step2_rank", 999) or 999),
            )
            out.append(_cooc_c)
            kept += 1
            if len(top_kept_co) < 5:
                top_kept_co.append(kt)

        if LABEL_EXPANSION_DEBUG and STAGE2_RULING_DEBUG:
            print(f"\n{'-' * 80}\n[Stage2B] Co-occurrence funnel\n{'-' * 80}")
            print(f"[Stage2B cooc debug] seed={term!r}")
            print(f"  raw_cooc={raw_cooc}")
            print(f"  after_purity_filter={n_freq3}")
            print(f"  after_domain_filter={n_dom}")
            print(f"  after_score_filter={n_dom_fit}")
            print(f"  after_support_gate={n_supp}")
            print(f"  after_domain_span={n_span_ok}")
            print(f"  final_kept={kept}")
            print(f"  top_raw={top_raw_co}")
            print(f"  top_kept={top_kept_co}")

        _audit_raw += int(raw_cooc)
        _audit_post += int(n_span_ok)
        _audit_kept += int(kept)
    if LABEL_EXPANSION_DEBUG and STAGE2_NOISY_DEBUG:
        sample = [c.term for c in out[:3]] if out else []
        print(f"[Stage2B] expand_from_cooccurrence_support primary数={len(primary_landings)} -> cooc_expansion {len(out)} 个 前3: {sample}")

    _stage2b_write_expansion_audit(
        label,
        "cooc",
        {
            "raw": int(_audit_raw),
            "post": int(_audit_post),
            "kept": int(_audit_kept),
            "blocked_top": _stage2b_top_blocked_reasons(_audit_blocked),
        },
    )
    return out


# ---------- 领域/三级领域过滤：Stage2A/2B 仅保留与当前查询领域一致的词 ----------


def _term_in_active_domains(
    label,
    vid: int,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> bool:
    """
    判断词汇 vid 是否落在当前激活领域（及可选的三级领域）内，供 Stage2A/2B 过滤用。
    - 无 active_domain_set 时不做领域过滤，返回 True。
    - 有 active_domain_set 时查 vocabulary_domain_stats，要求 term 的 domain_dist 与激活领域有交集。
    - 若提供了 jd_field_ids/jd_subfield_ids/jd_topic_ids，则同时要求 vocabulary_topic_stats 至少 field 级命中。
    """
    active = set(int(x) for x in (active_domain_set or [])) if active_domain_set else set()
    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))

    if not active:
        domain_ok = True
    else:
        if not getattr(label, "stats_conn", None):
            domain_ok = True
        else:
            row = None
            try:
                row = label.stats_conn.execute(
                    "SELECT domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                    (vid,),
                ).fetchone()
            except Exception:
                domain_ok = True
            if not row or not row[0]:
                domain_ok = True
            else:
                try:
                    dist = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                except Exception:
                    dist = {}
                expanded = expand_domain_dist(label, dist or {})
                active_str = set(str(d) for d in active)
                term_domains = set(expanded.keys()) if expanded else set()
                domain_ok = bool(active_str & term_domains)
                # 主领域守卫：主领域（权重最大）必须在 active_domains 内，否则一票否决
                if domain_ok and expanded:
                    main_domain = max(expanded, key=expanded.get)
                    if main_domain not in active_str:
                        domain_ok = False

    if not domain_ok:
        return False
    if not jd_f and not jd_s and not jd_t:
        return True
    topic_row = _load_vocabulary_topic_stats(label, vid)
    if not topic_row:
        return True
    score, _ = _compute_hierarchy_match_score(topic_row, jd_f, jd_s, jd_t)
    # 与 _term_in_active_domains_with_reason 一致：层级不对齐不再一票否决，交由 primary 惩罚
    return True


def _term_in_active_domains_with_reason(
    label,
    vid: int,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    """
    同 _term_in_active_domains，但返回 (是否通过, 原因描述)。
    通过时 reason 为空或 "topic_hierarchy_no_match"（层级未命中仍保留进池，供日志与 primary 惩罚）；
    未通过时仅 "domain_no_match"。
    """
    active = set(int(x) for x in (active_domain_set or [])) if active_domain_set else set()
    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))

    if not active:
        domain_ok = True
    else:
        if not getattr(label, "stats_conn", None):
            domain_ok = True
        else:
            row = None
            try:
                row = label.stats_conn.execute(
                    "SELECT domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                    (vid,),
                ).fetchone()
            except Exception:
                domain_ok = True
            if not row or not row[0]:
                domain_ok = True
            else:
                try:
                    dist = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                except Exception:
                    dist = {}
                expanded = expand_domain_dist(label, dist or {})
                active_str = set(str(d) for d in active)
                term_domains = set(expanded.keys()) if expanded else set()
                domain_ok = bool(active_str & term_domains)
                if domain_ok and expanded:
                    main_domain = max(expanded, key=expanded.get)
                    if main_domain not in active_str:
                        domain_ok = False
                # 强冲突：主领域在医学/社科/管理等且与激活领域无交时硬拒（见 README Stage2A 漏点修复）
                if not domain_ok and expanded and STRONG_CONFLICT_DOMAIN_IDS:
                    main_domain = max(expanded, key=expanded.get)
                    if str(main_domain) in STRONG_CONFLICT_DOMAIN_IDS:
                        return (False, "domain_conflict_strong")

    if not domain_ok:
        return (False, "domain_no_match")
    if not jd_f and not jd_s and not jd_t:
        return (True, "")
    topic_row = _load_vocabulary_topic_stats(label, vid)
    if not topic_row:
        return (True, "")
    score, level = _compute_hierarchy_match_score(topic_row, jd_f, jd_s, jd_t)
    # 不再在此处一票否决：topic_hierarchy_no_match 的候选保留进池，仅打上 reason 供日志与后续 primary 惩罚
    if score <= 0:
        return (True, "topic_hierarchy_no_match")
    return (True, "")


def _compute_domain_fit(
    label,
    vid: int,
    active_domain_set: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
) -> float:
    """
    为候选 term 计算领域拟合分，用于 Stage2A primary 门控与 Stage3 domain_gate。
    domain_fit = 0.4*domain_overlap + 0.3*field_overlap + 0.2*subfield_overlap + 0.1*topic_overlap。
    无表/无 JD 层级时对应 overlap 视为 1.0。
    """
    w_d, w_f, w_s, w_t = (0.4, 0.3, 0.2, 0.1)
    active = set(int(x) for x in (active_domain_set or [])) if active_domain_set else set()
    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))

    domain_overlap = 1.0
    if active and getattr(label, "stats_conn", None):
        try:
            row = label.stats_conn.execute(
                "SELECT domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (vid,),
            ).fetchone()
            if row and row[0]:
                dist = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                expanded = expand_domain_dist(label, dist or {})
                total = sum(expanded.values())
                if total > 0:
                    in_active = sum(expanded.get(str(d), 0) for d in active)
                    domain_overlap = in_active / total
                else:
                    domain_overlap = 1.0 if expanded and set(expanded.keys()) & {str(d) for d in active} else 0.0
        except Exception:
            pass

    field_overlap = 1.0
    subfield_overlap = 1.0
    topic_overlap = 1.0
    topic_row = _load_vocabulary_topic_stats(label, vid)
    if topic_row and (jd_f or jd_s or jd_t):
        term_f = _extract_ids_from_row(topic_row, "field_id", "field_dist")
        term_s = _extract_ids_from_row(topic_row, "subfield_id", "subfield_dist")
        term_t = _extract_ids_from_row(topic_row, "topic_id", "topic_dist")

        def _overlap(a: Set[str], b: Set[str]) -> float:
            if not b:
                return 1.0
            return 1.0 if (a & b) else 0.0

        field_overlap = _overlap(term_f, jd_f)
        subfield_overlap = _overlap(term_s, jd_s)
        topic_overlap = _overlap(term_t, jd_t)

    return float(
        w_d * domain_overlap
        + w_f * field_overlap
        + w_s * subfield_overlap
        + w_t * topic_overlap
    )


# ---------- 层级守卫：vocabulary 快照与层级 fit（仅标签路） ----------

_hierarchy_snapshot_cache: Dict[int, Dict[str, Any]] = {}


def get_vocab_hierarchy_snapshot(label, voc_id: int) -> Dict[str, Any]:
    """
    统一读取 vocabulary_topic_stats + vocabulary_domain_stats 为层级快照。
    返回 field_id, subfield_id, topic_id, field_dist, subfield_dist, topic_dist, domain_dist, domain_span, work_count（已解析）。
    """
    global _hierarchy_snapshot_cache
    if voc_id in _hierarchy_snapshot_cache:
        return _hierarchy_snapshot_cache[voc_id]
    out = {
        "voc_id": voc_id,
        "field_id": None,
        "subfield_id": None,
        "topic_id": None,
        "field_dist": {},
        "subfield_dist": {},
        "topic_dist": {},
        "domain_dist": {},
        "domain_span": 0,
        "work_count": 0,
    }
    if not getattr(label, "stats_conn", None):
        return out
    try:
        row_t = label.stats_conn.execute(
            "SELECT field_id, subfield_id, topic_id, field_dist, subfield_dist, topic_dist FROM vocabulary_topic_stats WHERE voc_id=?",
            (voc_id,),
        ).fetchone()
        if row_t:
            out["field_id"] = row_t[0]
            out["subfield_id"] = row_t[1]
            out["topic_id"] = row_t[2]
            out["field_dist"] = _parse_dist(row_t[3])
            out["subfield_dist"] = _parse_dist(row_t[4])
            out["topic_dist"] = _parse_dist(row_t[5])
        row_d = label.stats_conn.execute(
            "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
            (voc_id,),
        ).fetchone()
        if row_d:
            out["work_count"] = int(row_d[0] or 0)
            out["domain_span"] = int(row_d[1] or 0)
            out["domain_dist"] = _parse_dist(row_d[2])
    except Exception:
        pass
    _hierarchy_snapshot_cache[voc_id] = out
    return out


# ---------- 三层领域：vocabulary_topic_stats 查表与 topic_align 计算 ----------


def _load_vocabulary_topic_stats(label, voc_id: int) -> Optional[Dict[str, Any]]:
    """按 voc_id 查询 vocabulary_topic_stats，无记录或表不存在返回 None。"""
    if not getattr(label, "stats_conn", None):
        return None
    try:
        row = label.stats_conn.execute(
            "SELECT field_id, subfield_id, topic_id, field_dist, subfield_dist, topic_dist, source FROM vocabulary_topic_stats WHERE voc_id=?",
            (voc_id,),
        ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    return {
        "field_id": row[0],
        "subfield_id": row[1],
        "topic_id": row[2],
        "field_dist": row[3],
        "subfield_dist": row[4],
        "topic_dist": row[5],
        "source": row[6] if len(row) > 6 else None,
    }


def _parse_dist(dist_raw: Any) -> Dict[str, float]:
    """解析 JSON 分布，返回 {id_str: prob}。"""
    if not dist_raw:
        return {}
    try:
        d = json.loads(dist_raw) if isinstance(dist_raw, str) else dist_raw
        return {str(k): float(v) for k, v in (d or {}).items()}
    except Exception:
        return {}


def _extract_ids_from_row(topic_row: Dict[str, Any], key_id: str, key_dist: str) -> Set[str]:
    """从主值或分布提取 ID 集合。"""
    out = set()
    main = topic_row.get(key_id)
    if main is not None and str(main).strip():
        for p in re.split(r"[|,\s]+", str(main).strip()):
            if p:
                out.add(p.strip())
    if not out:
        dist = _parse_dist(topic_row.get(key_dist))
        for k, v in dist.items():
            if v and float(v) > 0:
                out.add(k)
    return out


def _compute_hierarchy_match_score(
    topic_row: Dict[str, Any],
    jd_field_ids: Set[str],
    jd_subfield_ids: Set[str],
    jd_topic_ids: Set[str],
) -> Tuple[float, str]:
    """层级命中分与档位（仅用于 debug/explain）。准入以 compute_hierarchy_evidence 的连续 effective_* 为准，不做硬档位拒绝。"""
    if not jd_topic_ids and not jd_subfield_ids and not jd_field_ids:
        return 1.0, "missing"
    term_topic = _extract_ids_from_row(topic_row, "topic_id", "topic_dist")
    term_subfield = _extract_ids_from_row(topic_row, "subfield_id", "subfield_dist")
    term_field = _extract_ids_from_row(topic_row, "field_id", "field_dist")

    def has_overlap(a: Set[str], b: Set[str]) -> bool:
        return bool(a & b) if a and b else False

    if jd_topic_ids and has_overlap(term_topic, jd_topic_ids):
        return 1.0, "topic"
    if jd_subfield_ids and has_overlap(term_subfield, jd_subfield_ids):
        return TOPIC_ALIGN_SUBFIELD, "subfield"
    if jd_field_ids and has_overlap(term_field, jd_field_ids):
        return TOPIC_ALIGN_FIELD, "field"
    return 0.0, "none"


def _overlap_ratio(cand_set: Set[str], active_set: Set[str]) -> float:
    """JD 激活集合被候选覆盖的比例：|cand & active| / |active|，active 空时返回 0。"""
    if not active_set:
        return 0.0
    inter = len(cand_set & active_set)
    return inter / len(active_set) if inter else 0.0


def _compute_path_match(
    cand_fields: Set[str],
    cand_subfields: Set[str],
    cand_topics: Set[str],
    active_fields: Set[str],
    active_subfields: Set[str],
    active_topics: Set[str],
) -> float:
    """
    三层路径一致性：加权平均，避免几何平均强连坐导致 topic 不完整时 path 塌成 0.03。
    权重 0.2*field + 0.3*subfield + 0.5*topic；某层 active 为空则跳过并重新归一化。
    """
    w_field, w_sub, w_topic = 0.2, 0.3, 0.5
    total_w = 0.0
    score = 0.0
    if active_fields:
        score += w_field * (_overlap_ratio(cand_fields, active_fields) or 0.0)
        total_w += w_field
    if active_subfields:
        score += w_sub * (_overlap_ratio(cand_subfields, active_subfields) or 0.0)
        total_w += w_sub
    if active_topics:
        score += w_topic * (_overlap_ratio(cand_topics, active_topics) or 0.0)
        total_w += w_topic
    if total_w <= 0:
        return 0.0
    return score / total_w


def compute_hierarchy_evidence(
    label,
    voc_id: int,
    active_fields: Set[str],
    active_subfields: Set[str],
    active_topics: Set[str],
) -> Dict[str, Any]:
    """
    三层领域特征 + 来源可信度。准入/裁判使用 effective_*（overlap * hierarchy_reliability），
    cooc 补出的三级领域不再与 direct 同权。
    返回含 topic_source, hierarchy_reliability, effective_topic_overlap, effective_subfield_overlap, effective_path_match。
    """
    snap = get_vocab_hierarchy_snapshot(label, voc_id)
    topic_row = _load_vocabulary_topic_stats(label, voc_id) or {}
    topic_source = (topic_row.get("source") or "missing").strip().lower()
    if topic_source == "direct":
        hierarchy_reliability = 1.0
    elif topic_source == "direct+cooc":
        hierarchy_reliability = 0.8
    elif topic_source == "cooc":
        hierarchy_reliability = 0.5
    else:
        hierarchy_reliability = 0.4

    cand_fields = _extract_ids_from_row(
        {"field_id": snap.get("field_id"), "field_dist": snap.get("field_dist") or {}},
        "field_id",
        "field_dist",
    )
    cand_subfields = _extract_ids_from_row(
        {"subfield_id": snap.get("subfield_id"), "subfield_dist": snap.get("subfield_dist") or {}},
        "subfield_id",
        "subfield_dist",
    )
    cand_topics = _extract_ids_from_row(
        {"topic_id": snap.get("topic_id"), "topic_dist": snap.get("topic_dist") or {}},
        "topic_id",
        "topic_dist",
    )
    field_overlap = _overlap_ratio(cand_fields, active_fields) if active_fields else 0.0
    subfield_overlap = _overlap_ratio(cand_subfields, active_subfields) if active_subfields else 0.0
    topic_overlap = _overlap_ratio(cand_topics, active_topics) if active_topics else 0.0
    path_match = _compute_path_match(
        cand_fields, cand_subfields, cand_topics,
        active_fields, active_subfields, active_topics,
    )
    effective_topic = topic_overlap * hierarchy_reliability
    effective_subfield = subfield_overlap * hierarchy_reliability
    effective_path = path_match * hierarchy_reliability

    topic_dist = _parse_dist(snap.get("topic_dist")) or _parse_dist(topic_row.get("topic_dist"))
    topic_specificity = compute_purity(topic_dist) if topic_dist else 0.0
    domain_span = int(snap.get("domain_span") or 0)
    topic_span_penalty = 1.0 / (1.0 + TOPIC_SPAN_PENALTY_FACTOR * max(0, domain_span - 1))
    if topic_overlap >= 0.5 and active_topics:
        hierarchy_level = HIERARCHY_LEVEL_TOPIC_EXACT
    elif subfield_overlap >= 0.35 and active_subfields:
        hierarchy_level = HIERARCHY_LEVEL_SUBFIELD_MATCH
    elif field_overlap >= 0.2 and active_fields:
        hierarchy_level = HIERARCHY_LEVEL_FIELD_ONLY
    else:
        hierarchy_level = HIERARCHY_LEVEL_OFF_PATH
    return {
        "field_overlap": field_overlap,
        "subfield_overlap": subfield_overlap,
        "topic_overlap": topic_overlap,
        "path_match": path_match,
        "topic_specificity": topic_specificity,
        "topic_span_penalty": topic_span_penalty,
        "hierarchy_level": hierarchy_level,
        "topic_source": topic_source,
        "hierarchy_reliability": hierarchy_reliability,
        "effective_topic_overlap": effective_topic,
        "effective_subfield_overlap": effective_subfield,
        "effective_path_match": effective_path,
    }


def _lexical_term_sanity(term: str, meta: Any) -> bool:
    """术语合法性：非空、非纯数字、长度合理。"""
    if not term or not str(term).strip():
        return False
    t = str(term).strip()
    if t.isdigit():
        return False
    if len(t) > 120:
        return False
    return True


# ---------- 边界保留与禁扩（正确但 hierarchy/path 弱 → 保留降权；错误/窄义 → 禁扩或剔除） ----------

# 工程主干锚点/术语关键词，用于 should_retain_borderline_candidate 的 engineering_core 判断
_ENGINEERING_CORE_ANCHORS = frozenset({
    "控制", "运动控制", "传统控制", "路径规划", "强化学习", "仿真", "机器人", "机器人运动控制",
    "医疗机器人", "动力学", "动力学参数辨识", "抓取", "端到端", "机械臂", "robotics", "robot control",
    "motion control", "reinforcement learning", "simulation", "pathfinding", "automatic control",
})
_ENGINEERING_CORE_TERM_SUBSTR = frozenset({
    "control", "motion", "robot", "learning", "simulation", "path", "planning", "identification",
    "dynamics", "manipulator", "robotics", "reinforcement", "digital control", "feedback control",
})


def _is_engineering_core_anchor(anchor_text: str) -> bool:
    """锚点是否为控制/规划/学习/仿真/机器人等工程主干。"""
    if not anchor_text:
        return False
    a = (anchor_text or "").strip().lower()
    return any(k in a or a in k for k in _ENGINEERING_CORE_ANCHORS)


def _is_engineering_core_term(term_text: str) -> bool:
    """候选是否为明显工程术语（关键词子串）。"""
    if not term_text:
        return False
    t = (term_text or "").strip().lower()
    return any(k in t for k in _ENGINEERING_CORE_TERM_SUBSTR)


def should_retain_borderline_candidate(
    anchor_text: str,
    candidate_text: str,
    semantic_score: float,
    jd_align: float,
    anchor_identity: float,
    hierarchy_level: str,
    topic_overlap: float,
    subfield_overlap: float,
    path_match: float,
    source_type: str,
) -> Tuple[bool, Optional[str]]:
    """
    当 check_primary_admission 原本要 reject 时，判断是否属于“虽然 hierarchy/path 弱，但语义上该保留”的边界词。
    返回 (retain, retain_reason)；retain_reason 为 "strong_*" 时 retain_mode=strong、不 suppress_seed，否则 weak、suppress_seed=True。
    """
    cand = (candidate_text or "").strip()
    anchor = (anchor_text or "").strip().lower()
    src = (source_type or "").strip().lower()

    # 1. 主干语义强：semantic + jd_align 双高 → strong retain
    if semantic_score >= 0.82 and jd_align >= 0.80:
        return True, "strong_semantic_retain"

    # 2. 工程主干锚点 + 明显工程术语，path 弱也先保留
    if _is_engineering_core_anchor(anchor_text) and _is_engineering_core_term(cand):
        if semantic_score >= 0.78 and jd_align >= 0.76:
            return True, "engineering_core_retain"

    # 3. similar_to 时，非明显错义则允许弱保留
    if src == "similar_to":
        if semantic_score >= 0.80 and jd_align >= 0.76:
            return True, "similar_to_weak_retain"

    # 4. hierarchy 只是弱（非 off_path 硬冲突）时也可弱保留
    if hierarchy_level != HIERARCHY_LEVEL_OFF_PATH:
        if semantic_score >= 0.78 and jd_align >= 0.78:
            return True, "borderline_hierarchy_retain"

    # 5. 即使 off_path，双高也弱保留（避免 simulation / reinforcement learning 等误杀）
    if semantic_score >= 0.80 and jd_align >= 0.78:
        return True, "weak_off_path_retain"

    return False, None


def _is_bad_support_for_anchor(anchor_text: str, support_term: str) -> bool:
    """扩散支撑词是否明显偏义（与 is_semantic_mismatch_seed 同族逻辑，避免 抓取→retrieval、动力学→propulsion 等）。"""
    return is_semantic_mismatch_seed(anchor_text, support_term)


def _get_vocab_vec(label: Any, vid: int) -> Optional[np.ndarray]:
    """取词汇向量，供 dense support gate 使用。"""
    idx = getattr(label, "vocab_to_idx", None)
    if idx is None:
        return None
    i = idx.get(str(vid)) if isinstance(idx, dict) else None
    if i is None:
        i = idx.get(vid)
    if i is None:
        return None
    all_vecs = getattr(label, "all_vocab_vectors", None)
    if all_vecs is None:
        return None
    try:
        vec = np.asarray(all_vecs[i], dtype=np.float32)
    except Exception:
        return None
    if vec.ndim != 1:
        vec = vec.flatten()
    return vec


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度，限制到 [0, 1]。"""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    if a.size != b.size or a.size == 0:
        return 0.0
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n <= 1e-9:
        return 0.0
    return float(max(0.0, min(1.0, np.dot(a, b) / n)))


def _estimate_context_support(
    label: Any,
    anchor: Any,
    candidate_vid: int,
    candidate_vec: Optional[np.ndarray] = None,
) -> float:
    """
    候选是否也出现在 anchor 的 conditioned top-k 附近。
    优先查缓存的 _context_score_map；没有则和 _context_neighbors 比局部相似度。
    """
    context_score_map = getattr(anchor, "_context_score_map", None) or {}
    if candidate_vid in context_score_map:
        try:
            return float(context_score_map[candidate_vid])
        except Exception:
            pass
    context_neighbors = getattr(anchor, "_context_neighbors", None) or []
    if not context_neighbors:
        return 0.0
    if candidate_vec is None:
        candidate_vec = _get_vocab_vec(label, candidate_vid)
    if candidate_vec is None:
        return 0.0
    best = 0.0
    for n in context_neighbors:
        n_vid = getattr(n, "vid", n) if not isinstance(n, int) else n
        n_vec = _get_vocab_vec(label, int(n_vid))
        if n_vec is None:
            continue
        best = max(best, _cos_sim(candidate_vec, n_vec))
    return best


def _estimate_anchor_family_support(
    label: Any,
    candidate_vec: np.ndarray,
    surviving_primaries: List[Any],
) -> float:
    """
    候选是否得到同锚 surviving primary family 的共同支持。
    防止只贴某一个旁支 primary（drift_penalty）。
    """
    if not surviving_primaries:
        return 0.0
    sims: List[float] = []
    for p in surviving_primaries:
        p_vid = getattr(p, "vid", None)
        if p_vid is None:
            continue
        p_vec = _get_vocab_vec(label, int(p_vid))
        if p_vec is None:
            continue
        sims.append(_cos_sim(candidate_vec, p_vec))
    if not sims:
        return 0.0
    mean_sim = sum(sims) / len(sims)
    max_sim = max(sims)
    min_sim = min(sims)
    drift_penalty = max(0.0, max_sim - min_sim)
    score = mean_sim - 0.25 * drift_penalty
    return max(0.0, min(1.0, score))


def is_semantic_mismatch_seed(anchor_text: str, primary_term: str) -> bool:
    """
    临时护栏：明显错义/偏义 seed 禁止扩散。
    动力学→propulsion、抓取→retrieval/crawling、端到端→principle、强化学习→仅 q-learning 等。
    """
    anchor = (anchor_text or "").strip().lower()
    term = (primary_term or "").strip().lower()
    if not term:
        return False

    if "动力学" in anchor or "dynamics" in anchor:
        if "propulsion" in term or "propulsor" in term:
            return True
    if "抓取" in anchor or "grasping" in anchor or "manipulation" in anchor:
        if any(x in term for x in ["retrieval", "crawler", "crawling"]):
            return True
    if "端到端" in anchor or "end-to-end" in anchor:
        if any(x in term for x in ["principle", "delay", "point-to-point", "end point"]):
            return True
    if "强化学习" in anchor or "reinforcement learning" in anchor:
        if term == "q-learning":
            return True
    return False


def _is_over_specific_subterm(anchor_text: str, primary_term: str) -> bool:
    """是否仅为锚点的过窄子项（如 q-learning 对 强化学习、robot hand 对 机械臂），不宜单独当 seed。"""
    anchor = (anchor_text or "").strip().lower()
    term = (primary_term or "").strip().lower()
    # 子项黑名单：锚点 -> 仅该词时禁扩
    over_specific = (
        ("机械臂", "robot hand"),
        ("机械臂", "robotic hand"),
        ("强化学习", "q-learning"),
        ("robot control", "machine control"),
        ("robot control", "servo control"),
        ("robotics", "telerobotics"),
    )
    for a_key, t_key in over_specific:
        if a_key in anchor or anchor in a_key:
            if t_key in term or term in t_key:
                return True
    return False


def should_block_seed_expansion(
    anchor_text: str,
    primary_term: str,
    primary_score: float,
    anchor_identity: float,
    jd_align: float,
    source_type: str,
    support_count: int,
    retain_mode: str,
    suppress_seed: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    surviving primary 是否应禁止扩散。
    返回 (block, block_reason)；block=True 时不应作为 seed。
    weak_retain 且 suppress_seed=True 时禁扩；suppress_seed=False 时可扩。
    """
    if retain_mode in ("weak", "weak_primary"):
        return True, "weak_primary_no_expand"
    if retain_mode == "weak_retain" and suppress_seed:
        return True, "weak_primary_no_expand"
    if anchor_identity < 0.20 and support_count < 2:
        return True, "low_identity_single_support"
    if is_semantic_mismatch_seed(anchor_text, primary_term):
        return True, "semantic_mismatch_seed"
    if _is_over_specific_subterm(anchor_text, primary_term):
        return True, "over_specific_without_head_term"
    src = (source_type or "").strip().lower()
    if src == "conditioned_vec" and anchor_identity < 0.35 and support_count < 2:
        return True, "weak_condvec_seed"
    return False, None


# ---------- Stage2B 双层门：seed gate（谁能扩）+ support gate（扩出来的词谁能留） ----------

# 禁止作为扩散 seed 的窄方法/器件支线词（做 primary 可保留，做 seed 会带偏 dense/cluster）
NARROW_METHOD_OR_BRANCH_TERMS = frozenset({
    "q-learning", "digital control", "automatic control", "route planning",
    "pathfinding", "servo control", "machine control", "instrument control",
    "radio control", "electronic control unit", "automatic train control",
    "automatic frequency control", "digitally controlled oscillator",
})

# 设备/对象/组件/应用支线词：dense 默认不进入 support，除非锚点本身允许
DEVICE_OBJECT_TERM_PATTERNS = (
    "oscillator", "radio control", "train control", "unit", "ecu",
    "instrument control", "comparator", "discharge machining", "frequency",
    "decimal", "design strategy", "two-sided market", "protocol", "resolution",
    "social computing", "social software", "electrical discharge",
    "frenet", "serret", "center frequency",
)


def is_narrow_method_term(primary_term: str) -> bool:
    """是否属于窄方法/支线词，不宜作为扩散 seed（seed_expand_factor 惩罚）。"""
    if not primary_term:
        return False
    t = (primary_term or "").strip().lower()
    if t in NARROW_METHOD_OR_BRANCH_TERMS:
        return True
    for k in NARROW_METHOD_OR_BRANCH_TERMS:
        if k in t or t in k:
            return True
    return False


def is_device_or_object_term(term: str) -> bool:
    """是否像设备/对象/组件/应用支线词，dense 默认不进入 support。"""
    if not term:
        return False
    t = (term or "").strip().lower()
    for pat in DEVICE_OBJECT_TERM_PATTERNS:
        if pat in t:
            return True
    return False


def anchor_allows_device_expansion(anchor_text: str) -> bool:
    """锚点是否允许设备类扩展（如「医疗机器人」可带出部分设备词）。当前保守：一律不允许。"""
    if not anchor_text:
        return False
    a = (anchor_text or "").strip().lower()
    # 若以后要对「医疗机器人」「手术机器人」等开放，在此加白名单
    return False


def is_primary_expandable(
    anchor_text: str,
    primary_term: str,
    primary_score: float,
    anchor_identity: float,
    jd_align: float,
    support_count: int,
    source_type: str,
) -> bool:
    """
    判断一个 primary 是否允许进入 Stage2B 扩散。
    原则：主线强匹配/双证据/多锚一致允许扩；conditioned-only/窄支线/明显对象词禁止扩。
    """
    anchor = (anchor_text or "").strip().lower()
    term = (primary_term or "").strip().lower()
    src = (source_type or "").strip().lower()
    if not term:
        return False

    # 0. 窄方法/明显支线，一律不扩
    if is_narrow_method_term(primary_term):
        return False

    # 1. 明显设备/对象词默认不扩（除非锚点白名单放行）
    if is_device_or_object_term(primary_term) and not anchor_allows_device_expansion(anchor_text):
        return False

    # 2. conditioned_only 候选默认更保守
    conditioned_only = src == "conditioned_vec"

    # 3. 三条放行主规则
    strong_identity_ok = (
        anchor_identity >= 0.55
        and primary_score >= 0.50
        and jd_align >= 0.74
    )
    dual_support_ok = (
        anchor_identity >= 0.48
        and primary_score >= 0.46
        and jd_align >= 0.72
        and not conditioned_only
    )
    multi_anchor_ok = (
        support_count >= 2
        and anchor_identity >= 0.42
        and primary_score >= 0.42
        and jd_align >= 0.72
    )
    allow = strong_identity_ok or dual_support_ok or multi_anchor_ok
    if not allow:
        return False

    # 4. 对 conditioned_only 再补一道保守门
    if conditioned_only:
        if not (
            anchor_identity >= 0.62
            and primary_score >= 0.55
            and jd_align >= 0.76
            and support_count >= 2
        ):
            return False
    return True


def support_expandable_for_anchor(
    label: Any,
    anchor: Any,
    parent_primary: Any,
    candidate_vid: int,
    candidate_term: str,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Dense support 最小修复 gate：不靠硬编码词表，只靠语义一致性 + 上下文稳定性 + family 支撑。
    四道门：primary_consistency、anchor_consistency、context_stability、family_support。
    返回 (keep, meta)；meta 含 reason、keep_score、各分量。
    """
    candidate_vec = _get_vocab_vec(label, candidate_vid)
    if candidate_vec is None:
        return False, {"reason": "missing_candidate_vec"}

    primary_vid = getattr(parent_primary, "vid", None)
    if primary_vid is None:
        return False, {"reason": "missing_primary_vid"}
    primary_vec = _get_vocab_vec(label, int(primary_vid))
    if primary_vec is None:
        return False, {"reason": "missing_primary_vec"}

    primary_consistency = _cos_sim(primary_vec, candidate_vec)

    anchor_consistency = 0.0
    conditioned = getattr(anchor, "conditioned_vec", None)
    if conditioned is not None:
        try:
            cv = np.asarray(conditioned, dtype=np.float32).flatten()
            if cv.size > 0:
                anchor_consistency = _cos_sim(cv, candidate_vec)
        except Exception:
            pass
    has_anchor_vec = conditioned is not None
    if not has_anchor_vec:
        anchor_consistency = 1.0  # 无 conditioned_vec 时跳过此项门控

    context_stability = _estimate_context_support(
        label=label,
        anchor=anchor,
        candidate_vid=candidate_vid,
        candidate_vec=candidate_vec,
    )
    ctx_map = getattr(anchor, "_context_score_map", None) or {}
    ctx_neighbors = getattr(anchor, "_context_neighbors", None) or []
    has_context_data = bool(ctx_neighbors) or (candidate_vid in ctx_map)
    if not has_context_data:
        context_stability = 1.0  # 无 context 数据时跳过此项门控

    surviving_primaries = getattr(anchor, "_stage2b_anchor_primaries", None) or []
    family_support = _estimate_anchor_family_support(
        label=label,
        candidate_vec=candidate_vec,
        surviving_primaries=surviving_primaries,
    )
    if not surviving_primaries:
        family_support = 1.0  # 无同锚 primary 时跳过此项门控

    if primary_consistency < DENSE_SUPPORT_PRIMARY_CONSISTENCY_MIN:
        return False, {
            "reason": "low_primary_consistency",
            "primary_consistency": primary_consistency,
        }
    if has_anchor_vec and anchor_consistency < DENSE_SUPPORT_ANCHOR_CONSISTENCY_MIN:
        return False, {
            "reason": "low_anchor_consistency",
            "anchor_consistency": anchor_consistency,
        }
    if has_context_data and context_stability < DENSE_SUPPORT_CONTEXT_STABILITY_MIN:
        return False, {
            "reason": "low_context_stability",
            "context_stability": context_stability,
        }
    if surviving_primaries and family_support < DENSE_SUPPORT_FAMILY_SUPPORT_MIN:
        return False, {
            "reason": "low_family_support",
            "family_support": family_support,
        }

    # 新增：dense support 对锚点的 identity 约束，抑制 robotic arm -> robotic hand 等 family 漂移
    anchor_term = getattr(anchor, "anchor_term", "") or getattr(anchor, "anchor", "") or ""
    anchor_identity_for_support = compute_anchor_identity_score(anchor_term, candidate_term)
    if anchor_identity_for_support < 0.16:
        return False, {"reason": "low_anchor_identity_support", "anchor_identity_for_support": anchor_identity_for_support}

    # 新增：family 漂移惩罚（候选只极贴某一 primary 而对整锚支撑不均衡时降分）
    drift_penalty = max(0.0, primary_consistency - family_support)

    keep_score = (
        0.26 * primary_consistency
        + 0.26 * anchor_consistency
        + 0.22 * context_stability
        + 0.16 * family_support
        + 0.10 * anchor_identity_for_support
    )
    keep_score = keep_score - 0.18 * drift_penalty

    if keep_score < 0.76:
        return False, {
            "reason": "keep_score_too_low",
            "keep_score": keep_score,
            "primary_consistency": primary_consistency,
            "anchor_consistency": anchor_consistency,
            "context_stability": context_stability,
            "family_support": family_support,
            "anchor_identity_for_support": anchor_identity_for_support,
            "drift_penalty": drift_penalty,
        }

    return True, {
        "reason": "ok",
        "keep_score": keep_score,
        "primary_consistency": primary_consistency,
        "anchor_consistency": anchor_consistency,
        "context_stability": context_stability,
        "family_support": family_support,
        "anchor_identity_for_support": anchor_identity_for_support,
        "drift_penalty": drift_penalty,
    }


def compute_context_consistency(c: LandingCandidate) -> float:
    """
    raw semantic_score 与 conditioned_vec context_sim 的一致性。
    越一致，说明这个词在「无上下文近邻」和「有上下文近邻」里都站得住。
    """
    raw_sim = max(0.0, min(1.0, float(getattr(c, "semantic_score", 0.0) or 0.0)))
    ctx_sim = max(0.0, min(1.0, float(getattr(c, "context_sim", 0.0) or 0.0)))
    if ctx_sim <= 0.0:
        return raw_sim * 0.65
    consistency = 1.0 - abs(raw_sim - ctx_sim)
    score = 0.45 * raw_sim + 0.45 * ctx_sim + 0.10 * consistency
    return max(0.0, min(1.0, score))


def compute_context_continuity(
    c: Any,
    jd_align: Optional[float] = None,
    co_anchor_support: float = 0.0,
) -> Tuple[float, float, float, float]:
    """
    Stage2A 上下文连续性：连续分，非布尔。
    同时看：anchor local（raw vs context_sim）、co_anchor、jd_profile/conditioned。
    返回 (context_continuity, context_local_support, context_co_anchor_support, context_jd_support)。
    """
    raw_sim = max(0.0, min(1.0, float(getattr(c, "semantic_score", 0.0) or 0.0)))
    ctx_sim = max(0.0, min(1.0, float(getattr(c, "context_sim", 0.0) or 0.0)))
    # local: raw 与 context_sim 一致性
    if ctx_sim <= 0.0:
        context_local_support = raw_sim * 0.65
    else:
        consistency = 1.0 - abs(raw_sim - ctx_sim)
        context_local_support = 0.45 * raw_sim + 0.45 * ctx_sim + 0.10 * consistency
    context_local_support = max(0.0, min(1.0, context_local_support))
    context_co_anchor_support = max(0.0, min(1.0, float(co_anchor_support)))
    jd_a = jd_align if jd_align is not None else float(getattr(c, "jd_candidate_alignment", 0.5) or 0.5)
    context_jd_support = max(0.0, min(1.0, jd_a))
    context_continuity = (
        0.50 * context_local_support
        + 0.25 * context_co_anchor_support
        + 0.25 * context_jd_support
    )
    return (
        max(0.0, min(1.0, context_continuity)),
        context_local_support,
        context_co_anchor_support,
        context_jd_support,
    )


def check_primary_eligibility(
    anchor: PreparedAnchor,
    c: LandingCandidate,
    hier_ev: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """资格赛：只拦特别危险的，边界情况留给 admission 做 weak_retain。返回 (eligible, reasons)。"""
    reasons: List[str] = []
    aid = float(getattr(c, "anchor_identity_score", 0.0) or 0.0)
    ctx_sim = float(getattr(c, "context_sim", 0.0) or 0.0)
    source_role = (getattr(c, "source_role", "") or "").strip()

    # identity 不再作为普遍硬门；只有非常低且上下文也不支持时才拦
    if aid < 0.20 and ctx_sim < 0.80:
        reasons.append("identity_too_low")
    # similar_to 候选：上下文明显反对时再拦；边界情况留给 admission 做 weak_retain
    if c.source == "similar_to" and ctx_sim > 0.0 and ctx_sim < 0.70:
        reasons.append("context_not_supporting_similar_to")
    # context fallback：仍保守，但降低 identity 门槛
    if source_role == "context_fallback":
        if ctx_sim < 0.82:
            reasons.append("context_fallback_not_strong_enough")
        if aid < 0.32 and ctx_sim < 0.88:
            reasons.append("context_fallback_identity_weak")
    domain_reason = getattr(c, "domain_reason", "") or ""
    if domain_reason == "domain_conflict_strong":
        reasons.append("domain_conflict_strong")
    return len(reasons) == 0, reasons


def check_primary_admission(candidate: Any) -> Dict[str, Any]:
    """
    Stage2A 主落点准入：三档判定（normal / weak_retain / reject）。
    只回答「候选是否有资格做主词、能否当扩散 seed」。宁可多留，明显错位才 reject。
    返回：retain_mode, suppress_seed, primary_eligibility_reasons（或 reasons）。
    """
    term = (getattr(candidate, "term", None) or "").strip()
    if not _lexical_term_sanity(term, None):
        return {
            "admit": False,
            "retain_mode": "reject",
            "suppress_seed": True,
            "reasons": ["lexical_not_term"],
            "primary_eligibility_reasons": ["lexical_not_term"],
        }

    semantic_score = float(getattr(candidate, "semantic_score", 0) or 0)
    jd_align = float(getattr(candidate, "jd_align", 0) or getattr(candidate, "jd_candidate_alignment", 0) or 0)
    context_continuity = float(getattr(candidate, "context_continuity", 0) or 0)
    anchor_identity = float(
        getattr(candidate, "anchor_identity_score", None)
        or getattr(candidate, "anchor_identity", None)
        or getattr(candidate, "family_match", 0)
        or 0
    )
    object_like_risk = float(getattr(candidate, "object_like_risk", 0) or 0)
    polysemy_risk = float(getattr(candidate, "polysemy_risk", 0) or 0)
    reasons: List[str] = []

    # 1. normal：可信主词，可扩
    if (
        semantic_score >= 0.68
        and jd_align >= 0.65
        and context_continuity >= 0.55
        and anchor_identity >= 0.18
        and object_like_risk <= 0.70
        and polysemy_risk <= 0.72
    ):
        return {
            "admit": True,
            "retain_mode": "normal",
            "suppress_seed": False,
            "reasons": ["normal_primary"],
            "primary_eligibility_reasons": ["normal_primary"],
        }

    # 2. weak_retain：保留继续竞争，但不扩
    if (
        semantic_score >= 0.55
        and jd_align >= 0.52
        and (context_continuity >= 0.45 or anchor_identity >= 0.12)
    ):
        reasons.append("weak_primary_retain")
        return {
            "admit": True,
            "retain_mode": "weak_retain",
            "suppress_seed": True,
            "reasons": reasons,
            "primary_eligibility_reasons": reasons,
        }

    # 3. reject：明显错位才拒（高歧义/对象化/明显偏题）
    reasons.append("primary_admission_failed")
    return {
        "admit": False,
        "retain_mode": "reject",
        "suppress_seed": True,
        "reasons": reasons,
        "primary_eligibility_reasons": reasons,
    }


def _piecewise_identity_factor(anchor_identity: float) -> float:
    """极低 identity 时大幅下沉。"""
    if anchor_identity >= 0.5:
        return 1.0
    if anchor_identity >= 0.30:
        return 0.85
    if anchor_identity >= 0.15:
        return 0.60
    return 0.40


def compute_primary_score(
    candidate: LandingCandidate,
    semantic_score: float,
    anchor_identity: float,
    jd_align: float,
    cross_anchor_support_count: int,
    local_neighborhood_consistency: float,
    hierarchy_evidence: Dict[str, Any],
    source_type: str,
) -> float:
    """
    本义 + 双空间一致性做主干，hierarchy 只微调；不再让 path/topic/span 单独翻盘。
    """
    raw_sim = max(0.0, min(1.0, float(semantic_score or 0.0)))
    identity = max(0.0, min(1.0, float(anchor_identity or 0.0)))
    jd = max(0.0, min(1.0, float(jd_align or 0.0)))
    cross = max(0.0, min(1.0, min(int(cross_anchor_support_count or 0), 2) / 2.0))
    neigh = max(0.0, min(1.0, float(local_neighborhood_consistency or 0.0)))
    context_consistency = compute_context_consistency(candidate)

    base = (
        0.30 * identity
        + 0.26 * context_consistency
        + 0.24 * jd
        + 0.10 * raw_sim
        + 0.06 * cross
        + 0.04 * neigh
    )
    field_fit = max(0.0, min(1.0, float(hierarchy_evidence.get("field_overlap", 0.0) or 0.0)))
    subfield_fit = max(0.0, min(1.0, float(hierarchy_evidence.get("subfield_overlap", 0.0) or 0.0)))
    topic_fit = max(0.0, min(1.0, float(hierarchy_evidence.get("topic_overlap", 0.0) or 0.0)))
    path_match = max(0.0, min(1.0, float(hierarchy_evidence.get("path_match", 0.0) or 0.0)))
    topic_span_penalty = max(0.0, min(1.0, float(hierarchy_evidence.get("topic_span_penalty", 1.0) or 1.0)))
    topic_specificity = max(0.0, min(1.0, float(hierarchy_evidence.get("topic_specificity", 0.0) or 0.0)))
    hierarchy_bonus = (
        0.04 * field_fit
        + 0.05 * subfield_fit
        + 0.05 * topic_fit
        + 0.03 * path_match
        + 0.02 * topic_specificity
    )
    span_factor = 0.95 + 0.05 * topic_span_penalty
    final = (base + hierarchy_bonus) * span_factor
    if getattr(candidate, "retain_mode", "normal") == "weak_retain":
        final *= 0.92
    if getattr(candidate, "source", "") == "similar_to" and not getattr(candidate, "context_supported", False):
        final *= 0.90
    return max(0.0, min(1.0, final))


# 主词优先：锚点->主干表达给小幅加分，避免 robot hand 长期压过 robotic arm
_HEAD_TERM_BONUS_MAP = (
    ("机械臂", ("robotic arm", "robot arm")),
    ("运动控制", ("motion control",)),
    ("机器人运动控制", ("robot control", "motion control")),
    ("路径规划", ("path planning", "pathfinding")),
    ("强化学习", ("reinforcement learning",)),
    ("仿真", ("simulation",)),
    ("医疗机器人", ("medical robotics",)),
    ("传统控制", ("automatic control",)),
)


def _head_term_bonus(anchor_text: str, term: str) -> float:
    """锚点主干词匹配时加小 bonus，便于 robotic arm 压过 robot hand。"""
    if not anchor_text or not term:
        return 0.0
    anchor = (anchor_text or "").strip().lower()
    t = (term or "").strip().lower()
    for a_key, heads in _HEAD_TERM_BONUS_MAP:
        if a_key not in anchor and anchor not in a_key:
            continue
        for h in heads:
            if h in t or t in h:
                return 0.08
    return 0.0


def choose_better_term_with_hierarchy(
    a: LandingCandidate,
    b: LandingCandidate,
    anchor: PreparedAnchor,
) -> LandingCandidate:
    """锚点内冲突裁判：三层领域 + 主词优先 bonus，避免子项压主项。"""
    ev_a = getattr(a, "hierarchy_evidence", {}) or {}
    ev_b = getattr(b, "hierarchy_evidence", {}) or {}
    anchor_text = getattr(anchor, "anchor", "") or ""
    score_a = (
        0.30 * (getattr(a, "primary_score", 0) or 0)
        + 0.20 * (ev_a.get("effective_topic_overlap", ev_a.get("topic_overlap", 0)) or 0)
        + 0.20 * (ev_a.get("effective_path_match", ev_a.get("path_match", 0)) or 0)
        + 0.10 * ev_a.get("topic_specificity", 0)
        + 0.10 * max(0, getattr(a, "jd_candidate_alignment", 0.5) or 0.5)
        + 0.10 * getattr(a, "anchor_identity_score", 0.5)
    ) * ev_a.get("topic_span_penalty", 1.0)
    score_a += _head_term_bonus(anchor_text, getattr(a, "term", "") or "")
    score_b = (
        0.30 * (getattr(b, "primary_score", 0) or 0)
        + 0.20 * (ev_b.get("effective_topic_overlap", ev_b.get("topic_overlap", 0)) or 0)
        + 0.20 * (ev_b.get("effective_path_match", ev_b.get("path_match", 0)) or 0)
        + 0.10 * ev_b.get("topic_specificity", 0)
        + 0.10 * max(0, getattr(b, "jd_candidate_alignment", 0.5) or 0.5)
        + 0.10 * getattr(b, "anchor_identity_score", 0.5)
    ) * ev_b.get("topic_span_penalty", 1.0)
    score_b += _head_term_bonus(anchor_text, getattr(b, "term", "") or "")
    return a if score_a >= score_b else b


def _are_semantically_overlapping(a: LandingCandidate, b: LandingCandidate) -> bool:
    """是否语义重叠（同 vid 或 term 高度相似）。"""
    if a.vid == b.vid:
        return True
    ta = (getattr(a, "term", "") or "").strip().lower()
    tb = (getattr(b, "term", "") or "").strip().lower()
    if ta == tb:
        return True
    return False


def resolve_anchor_local_conflicts(
    anchor: PreparedAnchor,
    candidates: List[LandingCandidate],
    primary_top_k: int = PRIMARY_MAX_PER_ANCHOR,
) -> List[LandingCandidate]:
    """
    同一锚点下泛词 vs 具体词、错义词 vs 对义词的裁决；控制下 control flow 不压过 motion control，抓取下 Data retrieval 不压过 grasping 类。
    """
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda x: getattr(x, "primary_score", 0.0) or 0.0, reverse=True)
    kept: List[LandingCandidate] = []
    for cand in ranked:
        conflict = False
        for i, existing in enumerate(kept):
            if not _are_semantically_overlapping(cand, existing):
                continue
            winner = choose_better_term_with_hierarchy(cand, existing, anchor)
            if winner is existing:
                conflict = True
                if LABEL_EXPANSION_DEBUG:
                    debug_print(2, f"[Stage2A Conflict Drop] anchor={anchor.anchor!r} drop={cand.term!r} kept={existing.term!r}", None)
                break
            else:
                kept.pop(i)
                if LABEL_EXPANSION_DEBUG:
                    debug_print(2, f"[Stage2A Conflict Replace] anchor={anchor.anchor!r} old={existing.term!r} new={cand.term!r}", None)
                break
        if not conflict:
            kept.append(cand)
        if len(kept) >= primary_top_k:
            break
    return kept[:primary_top_k]


def log_primary_reject(
    anchor: Any,
    cand: LandingCandidate,
    hierarchy_evidence: Dict[str, Any],
    reasons: List[str],
) -> None:
    """Stage2A 准入拒绝日志，便于排查为何错词被挡。"""
    if not LABEL_EXPANSION_DEBUG:
        return
    topic = hierarchy_evidence.get("topic_overlap", 0)
    sub = hierarchy_evidence.get("subfield_overlap", 0)
    path = hierarchy_evidence.get("path_match", 0)
    anchor_text = getattr(anchor, "anchor", "") or ""
    term = getattr(cand, "term", "") or ""
    print(
        f"[Stage2A Reject] anchor={anchor_text!r} cand={term!r} "
        f"topic={topic:.3f} sub={sub:.3f} path={path:.3f} reasons={reasons}"
    )


def _compute_topic_confidence(topic_row: Dict[str, Any]) -> float:
    """按 source 映射主题可信度。"""
    source = (topic_row.get("source") or "").strip().lower()
    if source == "direct":
        return 1.0
    if source == "direct+cooc":
        return 0.9
    if source == "cooc":
        return 0.7
    return 1.0


def _attach_topic_align(
    label,
    voc_id: int,
    jd_field_ids: Set[str],
    jd_subfield_ids: Set[str],
    jd_topic_ids: Set[str],
) -> Tuple[float, str, float]:
    """为单个 voc_id 计算 topic_align, topic_level, topic_confidence。缺表/无记录时返回 (1.0, 'missing', 1.0)。"""
    topic_row = _load_vocabulary_topic_stats(label, voc_id)
    if not topic_row:
        return 1.0, "missing", 1.0
    hierarchy_score, topic_level = _compute_hierarchy_match_score(
        topic_row, jd_field_ids, jd_subfield_ids, jd_topic_ids
    )
    confidence = _compute_topic_confidence(topic_row)
    topic_align = hierarchy_score * confidence
    return topic_align, topic_level, confidence


def merge_primary_and_support_terms(
    primary_landings: List[PrimaryLanding],
    dense_list: List[ExpandedTermCandidate],
    cluster_list: List[ExpandedTermCandidate],
    cooc_list: List[ExpandedTermCandidate],
    label,
    active_domains: Optional[Set[int]] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
    *,
    emit_merge_debug: bool = True,
) -> List[ExpandedTermCandidate]:
    """
    Stage2B merge：carryover landing + merge support expansions（职责收口：support expansion，不是“准终判层”）。

    本函数负责把本锚点的 Stage2A carryover landings（局部落点证据）与 Stage2B 扩展出来的支持项
    （dense/cluster/cooc expansions）合并成一条候选流，供 Stage2 出口打包并进入 Stage3。

    口径说明：
    - 这里的合并目标是“补齐候选与局部证据”，并补充 provenance/cluster/family/source 等可解释痕迹；
      不做跨锚的全局裁决，也不应被描述为最终保留/最终主词的决定层（全局裁决属于 Stage3）。
    - `primary_bucket` / `support_seed` / `support_keep` / `weak_primary` 等属于 Stage2 内部细分与兼容字段，
      会继续保留以维持旧链路，但跨阶段推荐语义应以 `local_role` + 局部证据字段为主。

    实现要点（不改算法，仅澄清）：
    - carryover landings 逐条转为 ExpandedTermCandidate（维持原字段与分数口径）
    - 合并 dense/cluster/cooc 扩展项并补齐统计/领域特征
    - 兼容层：仍根据 Stage2A 的局部标签填充少量历史字段（供旧 Stage3 逻辑读取；后续迁移可逐步降级）
    """
    load_vocab_meta(label)
    active = active_domains or set()
    jd_f = set(str(x) for x in (jd_field_ids or []))
    jd_s = set(str(x) for x in (jd_subfield_ids or []))
    jd_t = set(str(x) for x in (jd_topic_ids or []))
    out = []
    for p in primary_landings:
        row = None
        if getattr(label, "stats_conn", None):
            row = label.stats_conn.execute(
                "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (p.vid,),
            ).fetchone()
        degree_w = int(row[0]) if row else 0
        domain_span = int(row[1]) if row and len(row) > 1 else 0
        dist = {}
        if row and len(row) > 2 and row[2]:
            try:
                dist = json.loads(row[2]) if isinstance(row[2], str) else row[2]
            except Exception:
                pass
        expanded = expand_domain_dist(label, dist)
        degree_w_expanded = sum(expanded.values())
        target_degree_w = sum(expanded.get(str(d), 0) for d in active)
        topic_align, topic_level, topic_conf = _attach_topic_align(label, p.vid, jd_f, jd_s, jd_t)

        can_expand = bool(getattr(p, "can_expand", False))
        role_in_anchor = getattr(p, "role_in_anchor", None) or "side"
        primary_bucket = getattr(p, "primary_bucket", None) or ""

        # 按 2A 分桶决定 term_role，供 Stage3 直接区分
        if can_expand and role_in_anchor == "mainline":
            merged_term_role = "primary"
            weak_primary_flag = False
        else:
            merged_term_role = "primary_side"
            weak_primary_flag = True

        e = ExpandedTermCandidate(
            vid=p.vid,
            term=p.term,
            term_role=merged_term_role,
            identity_score=p.identity_score,
            source=p.source,
            anchor_vid=p.anchor_vid,
            anchor_term=p.anchor_term,
            semantic_score=1.0,
            degree_w=degree_w,
            domain_span=domain_span,
            degree_w_expanded=degree_w_expanded,
            target_degree_w=target_degree_w,
            src_vids=[],
            hit_count=1,
            topic_align=topic_align,
            topic_level=topic_level,
            topic_confidence=topic_conf,
            domain_fit=getattr(p, "domain_fit", 1.0),
            parent_primary=p.term,
        )
        if getattr(p, "subfield_fit", None) is not None:
            setattr(e, "subfield_fit", p.subfield_fit)
        if getattr(p, "topic_fit", None) is not None:
            setattr(e, "topic_fit", p.topic_fit)
        setattr(e, "field_fit", getattr(p, "field_fit", 0))
        setattr(e, "path_match", getattr(p, "path_match", 0))
        setattr(e, "genericity_penalty", getattr(p, "topic_span_penalty", 1.0))
        setattr(e, "retain_mode", getattr(p, "retain_mode", "normal"))
        setattr(e, "topic_source", getattr(p, "topic_source", "missing"))
        setattr(e, "seed_blocked", getattr(p, "seed_blocked", False))
        setattr(e, "seed_block_reason", getattr(p, "seed_block_reason", None))
        setattr(e, "has_family_evidence", getattr(p, "has_family_evidence", False))
        if getattr(p, "outside_subfield_mass", None) is not None:
            setattr(e, "outside_subfield_mass", p.outside_subfield_mass)
        if getattr(p, "outside_topic_mass", None) is not None:
            setattr(e, "outside_topic_mass", p.outside_topic_mass)
        if getattr(p, "topic_entropy", None) is not None:
            setattr(e, "topic_entropy", p.topic_entropy)
        if getattr(p, "landing_score", None) is not None:
            setattr(e, "landing_score", p.landing_score)
        if getattr(p, "fit_info", None) and p.fit_info.get("main_subfield_match") is not None:
            setattr(e, "main_subfield_match", p.fit_info.get("main_subfield_match"))
        if getattr(label, "voc_to_clusters", None):
            clusters = label.voc_to_clusters.get(int(p.vid)) or []
            if clusters:
                cid, _ = max(clusters, key=lambda x: x[1])
                setattr(e, "cluster_id", cid)
        setattr(e, "mainline_preference", getattr(p, "mainline_preference", None))
        setattr(e, "mainline_rank", getattr(p, "mainline_rank", None))
        setattr(e, "anchor_internal_rank", getattr(p, "anchor_internal_rank", None))
        setattr(e, "can_expand", can_expand)
        setattr(e, "sort_key_snapshot", getattr(p, "sort_key_snapshot", None))
        setattr(e, "role_in_anchor", role_in_anchor)
        setattr(e, "cross_anchor_support", getattr(p, "cross_anchor_support", None))
        setattr(e, "is_weak_primary", weak_primary_flag)
        setattr(e, "primary_bucket", primary_bucket)
        setattr(e, "mainline_support_factor", 1.0 if not weak_primary_flag else 0.65)
        # Stage2A 决策字段 → Stage3 / paper：`run_stage2._expanded_to_raw_candidates` 顶层透传依赖此处
        setattr(e, "can_expand_from_2a", bool(getattr(p, "can_expand_from_2a", False)))
        setattr(e, "fallback_primary", bool(getattr(p, "fallback_primary", False)))
        setattr(e, "admission_reason", str(getattr(p, "admission_reason", "") or ""))
        _rj = getattr(p, "reject_reason", None)
        setattr(e, "reject_reason", str(_rj) if _rj is not None else "")
        setattr(e, "survive_primary", bool(getattr(p, "survive_primary", False)))
        setattr(e, "stage2b_seed_tier", str(getattr(p, "stage2b_seed_tier", None) or "none"))
        setattr(e, "mainline_candidate", bool(getattr(p, "mainline_candidate", False)))
        setattr(e, "primary_reason", str(getattr(p, "primary_reason", "") or ""))
        setattr(e, "surface_sim", getattr(p, "surface_sim", None))
        setattr(e, "conditioned_sim", getattr(p, "conditioned_sim", None))
        setattr(e, "context_gain", float(getattr(p, "context_gain", 0) or 0))
        setattr(e, "source_set", getattr(p, "source_set", None))
        setattr(e, "has_dynamic_support", bool(getattr(p, "has_dynamic_support", False)))
        setattr(e, "has_static_support", bool(getattr(p, "has_static_support", False)))
        setattr(e, "dual_support", bool(getattr(p, "dual_support", False)))
        setattr(
            e,
            "parent_anchor_final_score",
            float(getattr(p, "anchor_final_score", 0.0) or 0.0),
        )
        setattr(
            e,
            "parent_anchor_step2_rank",
            int(getattr(p, "anchor_step2_rank", 999) or 999),
        )
        out.append(e)
    for c in dense_list:
        row = None
        if getattr(label, "stats_conn", None):
            row = label.stats_conn.execute(
                "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (c.vid,),
            ).fetchone()
        if row:
            c.degree_w = int(row[0])
            c.domain_span = int(row[1]) if len(row) > 1 else 0
            try:
                dist = json.loads(row[2]) if isinstance(row[2], str) else row[2] if len(row) > 2 else {}
            except Exception:
                dist = {}
            expanded = expand_domain_dist(label, dist)
            c.degree_w_expanded = sum(expanded.values())
            c.target_degree_w = sum(expanded.get(str(d), 0) for d in active)
        c.topic_align, c.topic_level, c.topic_confidence = _attach_topic_align(label, c.vid, jd_f, jd_s, jd_t)
        c.domain_fit = _compute_domain_fit(label, c.vid, active_domain_set=active_domains, jd_field_ids=jd_field_ids, jd_subfield_ids=jd_subfield_ids, jd_topic_ids=jd_topic_ids)
        if c.domain_fit < SUPPORT_MIN_DOMAIN_FIT or (getattr(c, "domain_span", 0) or 0) > DOMAIN_SPAN_EXTREME:
            continue
        if getattr(label, "voc_to_clusters", None):
            clusters = label.voc_to_clusters.get(int(c.vid)) or []
            if clusters:
                cid, _ = max(clusters, key=lambda x: x[1])
                setattr(c, "cluster_id", cid)
        out.append(c)
    for c in cluster_list:
        row = None
        if getattr(label, "stats_conn", None):
            row = label.stats_conn.execute(
                "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (c.vid,),
            ).fetchone()
        if row:
            c.degree_w = int(row[0])
            c.domain_span = int(row[1]) if len(row) > 1 else 0
            try:
                dist = json.loads(row[2]) if isinstance(row[2], str) else row[2] if len(row) > 2 else {}
            except Exception:
                dist = {}
            expanded = expand_domain_dist(label, dist)
            c.degree_w_expanded = sum(expanded.values())
            c.target_degree_w = sum(expanded.get(str(d), 0) for d in active)
        c.topic_align, c.topic_level, c.topic_confidence = _attach_topic_align(label, c.vid, jd_f, jd_s, jd_t)
        c.domain_fit = _compute_domain_fit(label, c.vid, active_domain_set=active_domains, jd_field_ids=jd_field_ids, jd_subfield_ids=jd_subfield_ids, jd_topic_ids=jd_topic_ids)
        if c.domain_fit < SUPPORT_MIN_DOMAIN_FIT or (getattr(c, "domain_span", 0) or 0) > DOMAIN_SPAN_EXTREME:
            continue
        if getattr(label, "voc_to_clusters", None):
            clusters = label.voc_to_clusters.get(int(c.vid)) or []
            if clusters:
                cid, _ = max(clusters, key=lambda x: x[1])
                setattr(c, "cluster_id", cid)
        out.append(c)
    for c in cooc_list:
        row = None
        if getattr(label, "stats_conn", None):
            row = label.stats_conn.execute(
                "SELECT work_count, domain_span, domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (c.vid,),
            ).fetchone()
        if row:
            c.degree_w = int(row[0])
            c.domain_span = int(row[1]) if len(row) > 1 else 0
            try:
                dist = json.loads(row[2]) if isinstance(row[2], str) else row[2] if len(row) > 2 else {}
            except Exception:
                dist = {}
            expanded = expand_domain_dist(label, dist)
            c.degree_w_expanded = sum(expanded.values())
            c.target_degree_w = sum(expanded.get(str(d), 0) for d in active)
        c.topic_align, c.topic_level, c.topic_confidence = _attach_topic_align(label, c.vid, jd_f, jd_s, jd_t)
        c.domain_fit = _compute_domain_fit(label, c.vid, active_domain_set=active_domains, jd_field_ids=jd_field_ids, jd_subfield_ids=jd_subfield_ids, jd_topic_ids=jd_topic_ids)
        if c.domain_fit < SUPPORT_MIN_DOMAIN_FIT or (getattr(c, "domain_span", 0) or 0) > DOMAIN_SPAN_EXTREME:
            continue
        if getattr(label, "voc_to_clusters", None):
            clusters = label.voc_to_clusters.get(int(c.vid)) or []
            if clusters:
                cid, _ = max(clusters, key=lambda x: x[1])
                setattr(c, "cluster_id", cid)
        out.append(c)
    if LABEL_EXPANSION_DEBUG and emit_merge_debug:
        # 参数名仍为 primary_landings，语义上即本锚 carryover（2A 非 reject 全量）
        n_carryover = len(primary_landings)
        n_dense, n_cluster, n_cooc = len(dense_list), len(cluster_list), len(cooc_list)
        sample = [c.term for c in out[:3]]
        print(
            f"[Stage2B] merge_primary_and_support_terms carryover={n_carryover} "
            f"dense={n_dense} cluster={n_cluster} cooc={n_cooc} -> 合计 {len(out)} 项 前3: {sample}"
        )
    return out


def stage2_generate_academic_terms(
    label,
    prepared_anchors: List[PreparedAnchor],
    active_domain_set: Optional[Set[int]] = None,
    domain_regex: Optional[str] = None,
    query_vector=None,
    query_text: Optional[str] = None,
    jd_field_ids: Optional[Set[str]] = None,
    jd_subfield_ids: Optional[Set[str]] = None,
    jd_topic_ids: Optional[Set[str]] = None,
    jd_profile: Optional[Dict[str, Any]] = None,
) -> List[ExpandedTermCandidate]:
    """
    Stage2 总入口（职责收口口径，不改算法）。

    - **Stage2A = local landing / local ranking**
      围绕每个 anchor 收集落点候选（landing candidates），做局部证据增强与组内相对排序，
      产出按桶分型的“本锚局部落点集合”（仍然是局部视角，不做跨锚全局裁决）。

    - **Stage2B = support expansion around strong local landings**
      仅围绕 Stage2A 产生的强局部落点（可扩/seed）做支持项扩展（dense/cluster/cooc 等），
      目标是补齐 recall 侧的局部证据，而不是决定最终保留项。

    - **Stage2 输出的是候选与局部证据，不是最终全局裁决**
      本函数返回 ExpandedTermCandidate 列表，供 Stage2 出口打包为 raw candidates；
      跨锚一致性、全局重排、最终准入/淘汰等全局决策属于 Stage3。

    说明：
    - 无有效局部落点则不做扩展；
    - 可选传入 jd_field_ids/jd_subfield_ids/jd_topic_ids/jd_profile，用于局部证据字段（如 topic_align/domain_fit 等）的计算与观测。
    """
    active_domains = set(int(x) for x in (active_domain_set or [])) if active_domain_set else set()
    if domain_regex and not active_domains:
        try:
            active_domains = set(int(x) for x in re.findall(r"\d+", domain_regex))
        except (ValueError, TypeError):
            pass
    # 诊断：Stage2 流水线统一在此初始化 similar_to 相关 debug（观测用途），供后续阶段/面板使用
    if getattr(label, "debug_info", None) is not None:
        label.debug_info.similar_to_raw_rows = []
        label.debug_info.similar_to_agg = []
        label.debug_info.similar_to_pass = []
    all_terms = []
    debug_print(1, "\n" + "-" * 80 + "\n[Stage2A] Primary Landing\n" + "-" * 80, label)
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2] stage2_generate_academic_terms 开始 锚点数={len(prepared_anchors)} active_domains={len(active_domains)}")
    # ---------- Stage2 总览（详细调试） ----------
    _stage2_header("总览")
    if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
        has_jd = jd_profile is not None
        jd_fields = len(jd_field_ids) if jd_field_ids else 0
        jd_sub = len(jd_subfield_ids) if jd_subfield_ids else 0
        jd_top = len(jd_topic_ids) if jd_topic_ids else 0
        print(f"  锚点数: {len(prepared_anchors)}  |  active_domains: {len(active_domains)}  |  有 JD: {has_jd}")
        print(f"  JD 维度: field={jd_fields}  subfield={jd_sub}  topic={jd_top}  |  Stage2A=组内相对选主（无固定阈值）")
        print(f"  锚点列表: {[getattr(a, 'anchor', a) for a in prepared_anchors]}")
    # ---------- Stage2A：主线优先组内选主 ----------
    for _ri, _a in enumerate(prepared_anchors, start=1):
        setattr(_a, "step2_anchor_rank", _ri)
    mainline_profile = build_stage2a_mainline_profile(
        label, prepared_anchors,
        query_vector=query_vector,
        query_text=query_text,
    )
    per_anchor_candidates: Dict[int, List[Stage2ACandidate]] = {}
    for anchor in prepared_anchors:
        landing_list = collect_landing_candidates(
            label, anchor,
            active_domain_set=active_domains,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
            jd_profile=jd_profile,
            query_vector=query_vector,
        )
        stage2a_list = landing_candidates_to_stage2a(landing_list)
        per_anchor_candidates[anchor.vid] = stage2a_list
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG and landing_list:
            sample = [c.term for c in landing_list[:3]]
            print(f"[Stage2A dual evidence] anchor={getattr(anchor, 'anchor', '')!r} 共 {len(landing_list)} 条 前3: {sample}")

    cross_anchor_index = build_cross_anchor_index(per_anchor_candidates)
    all_anchor_results: List[Dict[str, Any]] = []
    primary_landings_by_anchor: Dict[int, List[PrimaryLanding]] = {}
    evidence_table: List[Dict[str, Any]] = []

    _stage2_header("Stage2A 极简：主线优先组内选主（无固定阈值）", "-")
    for anchor in prepared_anchors:
        candidates = per_anchor_candidates.get(anchor.vid, [])
        if not candidates:
            primary_landings_by_anchor[anchor.vid] = []
            all_anchor_results.append({
                "anchor": anchor,
                "candidates": [],
                "primary_expandable": [],
                "primary_support_seed": [],
                "primary_support_keep": [],
                "risky_keep": [],
                "primary_keep_no_expand": [],
                "rejected": [],
                "stage2a_hard_rejected": [],
            })
            continue
        enriched = enrich_stage2a_candidates(
            label, anchor, candidates, prepared_anchors,
            mainline_profile=mainline_profile,
            cross_anchor_index=cross_anchor_index,
            query_vector=query_vector,
            query_text=query_text,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
        )
        # Stage2A pre-primary 明显错落点拦截：只拦大错词，不拦次优词
        stage2a_hard_rejected: List[Stage2ACandidate] = []
        kept: List[Stage2ACandidate] = []
        for c in enriched:
            hard_reject, reject_reason = _should_hard_reject_stage2a_candidate(c)
            setattr(c, "stage2a_hard_reject", hard_reject)
            setattr(c, "stage2a_hard_reject_reason", reject_reason or "")
            if hard_reject:
                stage2a_hard_rejected.append(c)
            else:
                kept.append(c)

        # --- fallback debug: keep 是否为空（用于定位是否触发 select_primary_per_anchor）---
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG:
            anchor_term = getattr(anchor, "anchor", "") or ""
            print(
                f"[Stage2A kept debug] anchor={anchor_term!r} "
                f"enriched={len(enriched)} kept={len(kept)}"
            )
        if not kept:
            primary_landings_by_anchor[anchor.vid] = []
            all_anchor_results.append({
                "anchor": anchor,
                "candidates": enriched,
                "primary_expandable": [],
                "primary_support_seed": [],
                "primary_support_keep": [],
                "risky_keep": [],
                "primary_keep_no_expand": [],
                "rejected": [],
                "stage2a_hard_rejected": stage2a_hard_rejected,
            })
            if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG and stage2a_hard_rejected:
                anchor_term = getattr(anchor, "anchor", "") or ""
                sample = [f"{c.term!r}={getattr(c, 'stage2a_hard_reject_reason', '')}" for c in stage2a_hard_rejected[:3]]
                print(f"[Stage2A hard reject] anchor={anchor_term!r} 共 {len(stage2a_hard_rejected)} 条 前3: {sample}")
            continue
        selected = select_primary_per_anchor(anchor, kept)
        anchor_term = getattr(anchor, "anchor", "") or ""
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG and stage2a_hard_rejected:
            sample = [f"{c.term!r}={getattr(c, 'stage2a_hard_reject_reason', '')}" for c in stage2a_hard_rejected[:3]]
            print(f"[Stage2A hard reject] anchor={anchor_term!r} 共 {len(stage2a_hard_rejected)} 条 前3: {sample}")

        # 本锚点 0-primary：canonical 核心锚点尝试 fallback 保线（不扩）
        if not selected["primary_expandable"] and not selected["primary_keep_no_expand"]:
            fb = pick_fallback_primary_for_anchor(anchor, kept)
            if fb is not None:
                selected["primary_support_keep"] = [fb]
                selected["primary_keep_no_expand"] = (
                    selected["primary_support_seed"] + selected["primary_support_keep"] + selected["risky_keep"]
                )
                if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG:
                    print(
                        f"[Stage2A fallback] anchor={anchor_term!r} "
                        f"selected={fb.term!r} reason=anchor_core_fallback"
                    )

        # 为选中的候选挂 parent_anchor，供 merge / Stage2B 使用
        anchor_term = getattr(anchor, "anchor", "") or ""
        for cand in selected["primary_expandable"] + selected["primary_keep_no_expand"]:
            setattr(cand, "parent_anchor", anchor_term)
            setattr(cand, "parent_anchor_obj", anchor)

        primary_landings_list = []
        for cand in selected["primary_expandable"] + selected["primary_keep_no_expand"]:
            p = PrimaryLanding(
                vid=cand.tid,
                term=cand.term or str(cand.tid),
                identity_score=cand.family_match,
                source=cand.source,
                anchor_vid=anchor.vid,
                anchor_term=anchor.anchor,
                domain_fit=1.0,
            )
            setattr(p, "primary_score", cand.composite_rank_score)
            setattr(p, "anchor_identity_score", cand.family_match)
            _cec = bool(getattr(cand, "can_expand_from_2a", cand.can_expand))
            setattr(p, "can_expand_from_2a", _cec)
            setattr(p, "fallback_primary", bool(getattr(cand, "fallback_primary", False)))
            _rm = getattr(cand, "retain_mode", None)
            if not _rm or (not isinstance(_rm, str)):
                _rm = "normal" if _cec else "weak_retain"
            setattr(p, "retain_mode", _rm)
            setattr(p, "suppress_seed", not _cec)
            setattr(p, "stage2b_seed_tier", (getattr(cand, "stage2b_seed_tier", None) or "none"))
            setattr(p, "generic_risk", float(getattr(cand, "generic_risk", 0) or 0))
            setattr(p, "polysemy_risk", float(getattr(cand, "polysemy_risk", 0) or 0))
            setattr(p, "object_like_risk", float(getattr(cand, "object_like_risk", 0) or 0))
            setattr(p, "topic_source", "missing")
            setattr(p, "jd_align", cand.jd_align)
            _pb = getattr(cand, "primary_bucket", None) or ("primary_expandable" if _cec else "primary_keep_no_expand")
            setattr(p, "bucket", _pb)
            setattr(p, "primary_bucket", _pb)
            setattr(p, "expandable", _cec)
            setattr(p, "field_fit", 0.0)
            setattr(p, "path_match", 0.0)
            setattr(p, "topic_span_penalty", 1.0)
            setattr(p, "subfield_fit", 0.0)
            setattr(p, "topic_fit", 0.0)
            setattr(p, "cross_anchor_support_count", max(1, len(cross_anchor_index.get(cand.tid, []))))
            setattr(p, "has_family_evidence", getattr(cand, "has_family_evidence", False))
            setattr(p, "mainline_preference", getattr(cand, "mainline_preference", None))
            setattr(p, "mainline_rank", getattr(cand, "mainline_rank", None))
            setattr(p, "anchor_internal_rank", getattr(cand, "anchor_internal_rank", None))
            setattr(p, "survive_primary", getattr(cand, "survive_primary", None))
            setattr(p, "can_expand", _cec)
            setattr(p, "sort_key_snapshot", getattr(cand, "sort_key_snapshot", None))
            setattr(p, "role_in_anchor", getattr(cand, "role_in_anchor", None))
            setattr(p, "cross_anchor_support", getattr(cand, "cross_anchor_support", None))
            setattr(p, "context_gain", float(getattr(cand, "context_gain", 0) or 0))
            setattr(p, "semantic_score", float(getattr(cand, "semantic_score", 0) or 0))
            _cs = getattr(cand, "conditioned_sim", None)
            setattr(p, "conditioned_sim", float(_cs) if _cs is not None else None)
            _surf = getattr(cand, "surface_sim", None)
            setattr(p, "surface_sim", float(_surf) if _surf is not None else None)
            setattr(p, "source_set", getattr(cand, "source_set", None))
            setattr(p, "has_dynamic_support", bool(getattr(cand, "has_dynamic_support", False)))
            setattr(p, "has_static_support", bool(getattr(cand, "has_static_support", False)))
            setattr(p, "dual_support", bool(getattr(cand, "dual_support", False)))
            setattr(p, "source_type", getattr(cand, "source_type", None) or getattr(cand, "source", "") or "")
        # Stage2A 审计字段 → PrimaryLanding → merge → raw_candidates 顶层
        # 说明：以下字段属于 Stage2A 细桶体系/审计痕迹，长期口径应降级为 debug/兼容镜像，
        # 不作为跨阶段正式语义；跨阶段推荐使用 local_role + 局部证据字段（见 Stage2 出口打包层）。
            setattr(p, "admission_reason", str(getattr(cand, "admission_reason", "") or ""))
            _crj = getattr(cand, "reject_reason", None)
            setattr(p, "reject_reason", str(_crj) if _crj is not None else "")
            setattr(p, "mainline_candidate", bool(getattr(cand, "mainline_candidate", False)))
            setattr(p, "primary_reason", str(getattr(cand, "primary_reason", "") or ""))
            setattr(
                p,
                "anchor_final_score",
                float(getattr(anchor, "final_anchor_score", 0.0) or 0.0),
            )
            setattr(
                p,
                "anchor_step2_rank",
                int(getattr(anchor, "step2_anchor_rank", 999) or 999),
            )
            primary_landings_list.append(p)
            _mainline_num = 0.7 if cand.role == "mainline" else (0.4 if cand.role == "side" else 0.0)
            evidence_table.append({
                "anchor": anchor.anchor,
                "anchor_vid": anchor.vid,
                "candidate": cand.term,
                "tid": cand.tid,
                "bucket": _pb,
                "primary_score": cand.composite_rank_score,
                "mainline_alignment": _mainline_num,
                "expandable": _cec,
                "source": cand.source,
                "semantic_score": cand.semantic_score,
                "jd_align": cand.jd_align,
                "anchor_identity_score": cand.family_match,
                "identity_gate": 1.0,
                "base_primary_score": cand.composite_rank_score,
                "edge_affinity": cand.family_match,
                "conditioned_anchor_align": None,
                "multi_anchor_support": cand.cross_anchor_support,
                "hierarchy_consistency": cand.hierarchy_consistency,
                "neighborhood_consistency": 0.5,
                "isolation_penalty": cand.isolation_risk,
                "polysemy_risk": cand.polysemy_risk,
                "specificity_prior": 0.5,
            })
        primary_landings_by_anchor[anchor.vid] = primary_landings_list
        all_anchor_results.append({
            "anchor": anchor,
            "candidates": enriched,
            "primary_expandable": selected["primary_expandable"],
            "primary_support_seed": selected["primary_support_seed"],
            "primary_support_keep": selected["primary_support_keep"],
            "risky_keep": selected["risky_keep"],
            "primary_keep_no_expand": selected["primary_keep_no_expand"],
            "rejected": selected["rejected"],
            "stage2a_hard_rejected": stage2a_hard_rejected,
        })

    final_primary_merged = merge_stage2a_primary(all_anchor_results)

    # 五层 + tier 全局窄表：下一轮日志先看这张再盯单锚 Focus
    _print_stage2a_global_bucket_summary(all_anchor_results)

    if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
        for block in all_anchor_results:
            anc = getattr(block.get("anchor"), "anchor", "") or ""
            for group_name in (
                "primary_expandable",
                "primary_support_seed",
                "primary_support_keep",
                "risky_keep",
            ):
                for cand in block.get(group_name, []):
                    if not _should_emit_stage2a_merge_evidence_detail(cand, group_name):
                        continue
                    local_bucket = getattr(cand, "primary_bucket", "") or group_name
                    mainline_cand = getattr(cand, "is_good_mainline", False)
                    can_expand_local = bool(getattr(cand, "can_expand_from_2a", getattr(cand, "can_expand", False)))
                    local_reason = getattr(cand, "mainline_block_reason", "") or getattr(cand, "bucket_reason", "")
                    print(
                        f"[Stage2A merge evidence detail] term={(cand.term or '')[:28]!r} | anchor={anc!r} | "
                        f"local_bucket={local_bucket!r} mainline_candidate={mainline_cand} can_expand_local={can_expand_local} reason={local_reason!r}"
                    )
    # Stage2A 选主/合并汇总：条数 + 前3，不打印全表
    if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
        merged_sample = [m.get("term") for m in final_primary_merged[:3]]
        print(f"[Stage2A] Merged Primary 共 {len(final_primary_merged)} 条 前3: {merged_sample}")
        for block in all_anchor_results:
            anc = getattr(block.get("anchor"), "anchor", "") or ""
            pe = block.get("primary_expandable", [])
            ps = block.get("primary_support_seed", [])
            pkp = block.get("primary_support_keep", [])
            rk = block.get("risky_keep", [])
            pe_terms = [c.term for c in pe[:3]] if pe else []
            ps_terms = [c.term for c in ps[:3]] if ps else []
            pkp_terms = [c.term for c in pkp[:3]] if pkp else []
            rk_terms = [c.term for c in rk[:3]] if rk else []
            print(
                f"  [Stage2A 选主] anchor={anc!r} expandable={pe_terms} "
                f"support_seed={ps_terms} support_keep={pkp_terms} risky_keep={rk_terms}"
            )

    _stage2_header("Stage2A 每锚 primary（组内相对选主）", "-")
    if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
        vid_to_block = {getattr(b.get("anchor"), "vid", None): b for b in all_anchor_results}
        rows_summary = []
        for a in prepared_anchors:
            plist = primary_landings_by_anchor.get(a.vid, [])
            n_cand = len(per_anchor_candidates.get(a.vid, []))
            blk = vid_to_block.get(a.vid)
            pe = len(blk.get("primary_expandable", [])) if blk else 0
            ps = len(blk.get("primary_support_seed", [])) if blk else 0
            pkp = len(blk.get("primary_support_keep", [])) if blk else 0
            rk = len(blk.get("risky_keep", [])) if blk else 0
            rows_summary.append([
                getattr(a, "anchor", str(a))[:16],
                str(n_cand),
                str(len(plist)),
                str(pe),
                str(ps),
                str(pkp),
                str(rk),
            ])
        if rows_summary:
            _stage2_table(
                rows_summary,
                ["锚点", "候选数", "landings", "exp", "sd", "sk", "rk"],
                col_widths=[18, 8, 10, 5, 5, 5, 5],
            )
        for a in prepared_anchors:
            plist = primary_landings_by_anchor.get(a.vid, [])
            if not plist:
                continue
            sample = [p.term for p in plist[:3]]
            blk = vid_to_block.get(a.vid)
            pe = len(blk.get("primary_expandable", [])) if blk else 0
            ps = len(blk.get("primary_support_seed", [])) if blk else 0
            pkp = len(blk.get("primary_support_keep", [])) if blk else 0
            rk = len(blk.get("risky_keep", [])) if blk else 0
            print(
                f"  锚点 {getattr(a, 'anchor', a)!r} carryover_terms={len(plist)} "
                f"(exp={pe} sd={ps} sk={pkp} rk={rk}) 前3: {sample}"
            )

    if getattr(label, "debug_info", None) is not None:
        label.debug_info.stage2_anchor_evidence_table = evidence_table
        # 按 term 聚合：term | sources | similar_to_score | conditioned_score | primary_score_max（便于区分双路来源）
        by_term: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for row in evidence_table:
            tid, term = row["tid"], (row.get("candidate") or "").strip() or str(row["tid"])
            key = (tid, term)
            if key not in by_term:
                by_term[key] = {
                    "tid": tid,
                    "term": term,
                    "sources": set(),
                    "similar_to_score": None,
                    "conditioned_score": None,
                    "final_primary_score": 0.0,
                }
            s = (row.get("source") or "").strip().lower()
            sem = float(row.get("semantic_score") or 0)
            ps = float(row.get("primary_score") or 0)
            by_term[key]["sources"].add(s if s else "similar_to")
            if s == "similar_to":
                if by_term[key]["similar_to_score"] is None or sem > (by_term[key]["similar_to_score"] or 0):
                    by_term[key]["similar_to_score"] = sem
            elif s == "conditioned_vec":
                if by_term[key]["conditioned_score"] is None or sem > (by_term[key]["conditioned_score"] or 0):
                    by_term[key]["conditioned_score"] = sem
            if ps > (by_term[key]["final_primary_score"] or 0):
                by_term[key]["final_primary_score"] = ps
        term_breakdown = []
        for (tid, term), v in by_term.items():
            term_breakdown.append({
                "tid": tid,
                "term": v["term"],
                "sources": sorted(v["sources"]) if v["sources"] else [],
                "similar_to_score": v["similar_to_score"],
                "conditioned_score": v["conditioned_score"],
                "final_primary_score": v["final_primary_score"],
            })
        term_breakdown.sort(key=lambda x: -(x["final_primary_score"] or 0))
        label.debug_info.stage2a_term_source_breakdown = term_breakdown
    for anchor in prepared_anchors:
        # carryover：2A 保留全集；seed_candidates：仅 strong/weak tier，避免把 support_keep/risky 误当成「被 2B 门挡掉的 seed」
        carryover_terms = primary_landings_by_anchor.get(anchor.vid) or []
        if not carryover_terms:
            if LABEL_EXPANSION_DEBUG:
                anchor_type_lower = (getattr(anchor, "anchor_type", "") or "").strip().lower()
                amb_tag = " [高歧义]" if anchor_type_lower in HIGH_AMBIGUITY_ANCHOR_TYPES else ""
                n_cand = len(per_anchor_candidates.get(anchor.vid, []))
                print(
                    f"[Stage2] 锚点 anchor={anchor.anchor!r} vid={anchor.vid}{amb_tag} 无 primary，跳过 | "
                    f"原因: 本锚点组内相对选主后无 primary（候选数={n_cand}）"
                )
            continue
        carryover_terms, seed_candidates = _split_stage2b_carryover_and_seed_candidates(carryover_terms)

        # 非扩散候选：不调用 check_seed_eligibility（减少 tier=none 噪音），下传字段与「门入口处即 stage2a_not_seed」一致
        _seed_cand_ids = {id(p) for p in seed_candidates}
        for p in carryover_terms:
            if id(p) not in _seed_cand_ids:
                setattr(p, "seed_score", 0.0)
                setattr(p, "seed_blocked", True)
                setattr(p, "seed_block_reason", "stage2a_not_seed")
                setattr(p, "seed_eligible", False)

        # seed_candidates=0 且无 NOISY：折叠为一行，避免动力学/仿真/无 seed 锚点刷屏（开 STAGE2_NOISY_DEBUG 仍走完整审计）
        if len(seed_candidates) == 0 and not STAGE2_NOISY_DEBUG:
            merged = merge_primary_and_support_terms(
                carryover_terms,
                [],
                [],
                [],
                label,
                active_domains=active_domains,
                jd_field_ids=jd_field_ids,
                jd_subfield_ids=jd_subfield_ids,
                jd_topic_ids=jd_topic_ids,
                emit_merge_debug=False,
            )
            if LABEL_EXPANSION_DEBUG:
                _print_stage2b_no_seed_reason(anchor.anchor, carryover_terms)
                print(
                    f"[Stage2B] anchor={anchor.anchor!r} carryover={len(carryover_terms)} "
                    f"seed_candidates=0 eligible_seeds=0 diffusion=0 merged={len(merged)}"
                )
                print(
                    f"[Stage2B retain summary] anchor={getattr(anchor, 'anchor', '')!r} "
                    f"seed_candidates_for_stage2b=0 eligible_seeds=0 expansion_evidence_added=0 "
                    f"carryover={len(carryover_terms)} merged={len(merged)}"
                )
            all_terms.extend(merged)
            continue

        _print_stage2b_input_audit(anchor.anchor, carryover_terms, seed_candidates)
        if len(seed_candidates) == 0:
            _print_stage2b_no_seed_reason(anchor.anchor, carryover_terms)

        # Stage2B：check_seed_eligibility 仅遍历 seed_candidates（强/弱 seed）
        diffusion_scored: List[Tuple[Any, float]] = []
        _seed_detail: List[Tuple[Any, bool, float, Optional[str]]] = []
        for p in seed_candidates:
            eligible, seed_score, block_reason = check_seed_eligibility(label, p, jd_profile)
            setattr(p, "seed_score", seed_score)
            setattr(p, "seed_blocked", not eligible)
            setattr(p, "seed_block_reason", block_reason)
            if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG:
                print(
                    f"[Stage2B seed gate] term={getattr(p, 'term', '')!r} "
                    f"eligible={eligible} block_reason={block_reason!r}"
                )
            _seed_detail.append((p, eligible, seed_score, block_reason))
            if eligible:
                diffusion_scored.append((p, seed_score))
        _anch_txt = getattr(anchor, "anchor", "") or ""
        _print_stage2b_seed_tier_audit(_anch_txt, _seed_detail)
        _print_stage2b_blocked_weak_seed_audit(_anch_txt, _seed_detail)
        diffusion_scored.sort(key=lambda x: x[1], reverse=True)
        seed_primaries = [p for p, _ in diffusion_scored]
        strong_seed_count = sum(
            1 for p in seed_primaries if (getattr(p, "stage2b_seed_tier", "") or "").strip().lower() == "strong"
        )
        weak_seed_count = sum(
            1 for p in seed_primaries if (getattr(p, "stage2b_seed_tier", "") or "").strip().lower() == "weak"
        )
        eligible_seed_count = len(seed_primaries)
        _stage2_header(f"Stage2B seed 明细 [锚点 {getattr(anchor, 'anchor', anchor)!r}]", "-")
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG and _seed_detail:
            sample = [p.term for (p, el, _, _) in _seed_detail[:3]]
            print(
                f"  diffusion 数={eligible_seed_count}/{len(seed_candidates)} 前3: {sample} "
                f"diffusion 前3: {[p.term for p in seed_primaries[:3]]}"
            )
        # 只保留 stdout 一条汇总；diffusion/seed_terms 的 debug_print 与 seed tier audit 信息重复，默认关闭
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
            print(
                f"  [Stage2B] anchor={anchor.anchor!r} "
                f"seed_candidates={len(seed_candidates)} eligible_seeds={eligible_seed_count} "
                f"strong={strong_seed_count} weak={weak_seed_count}"
            )
        if STAGE2_NOISY_DEBUG:
            debug_print(
                1,
                f"[Stage2B] diffusion={eligible_seed_count}/{len(seed_candidates)}（仅对 seed_candidates 统计）",
                label,
            )
            if seed_primaries:
                debug_print(2, f"[Stage2B] seed_terms={[p.term for p in seed_primaries[:10]]}", label)
        if LABEL_EXPANSION_DEBUG and STAGE2_NOISY_DEBUG and (eligible_seed_count != len(seed_candidates) or not seed_primaries):
            print(
                f"[Stage2B] eligible_seeds={eligible_seed_count}/{len(seed_candidates)} "
                f"（check_seed_eligibility 仅扫 seed_candidates；carryover={len(carryover_terms)} 仍全量 merge）"
            )
        # 为 dense/cooc 的 support_expandable_for_anchor 提供锚点 context
        for p in seed_primaries:
            setattr(p, "anchor_conditioned_vec", getattr(anchor, "conditioned_vec", None))
            setattr(p, "_context_neighbors", getattr(anchor, "_context_neighbors", None) or [])
            setattr(p, "_context_score_map", getattr(anchor, "_context_score_map", None) or {})
        dense_list = expand_from_vocab_dense_neighbors(
            label, seed_primaries,
            active_domain_set=active_domains,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
            jd_profile=jd_profile,
        )
        cluster_list = expand_from_cluster_members(
            label, seed_primaries,
            active_domain_set=active_domains,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
        )
        cooc_list = expand_from_cooccurrence_support(
            label, seed_primaries,
            active_domain_set=active_domains,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
            jd_profile=jd_profile,
        )
        _stage2_header("Stage2B 扩展汇总（dense / cluster / cooc）", "-")
        # Stage2B 扩展产出审计增强：解释 eligible seed 存在但 kept=0 的原因（不改阈值/公式）
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
            audits = getattr(label, "_stage2b_last_expansion_audit", None)
            audits = audits if isinstance(audits, dict) else {}
            da = audits.get("dense") or {}
            ca = audits.get("cluster") or {}
            oa = audits.get("cooc") or {}
            ds = da.get("blocked_samples") or {}
            dense_samples = {
                k: ds.get(k, [])
                for k in ("vocab_type_filtered", "no_domain_or_topic_fit", "sim_below_min_dense_sim")
                if ds.get(k)
            }
            weak_no_dom = da.get("no_domain_weak_detail") or []
            print(
                f"[Stage2B expansion audit] anchor={getattr(anchor, 'anchor', '')!r} "
                f"dense_raw/post/kept={da.get('raw', 0)}/{da.get('post', 0)}/{da.get('kept', 0)} "
                f"cluster_raw/post/kept={ca.get('raw', 0)}/{ca.get('post', 0)}/{ca.get('kept', 0)} "
                f"cooc_raw/post/kept={oa.get('raw', 0)}/{oa.get('post', 0)}/{oa.get('kept', 0)} "
                f"dense_blocked_top={da.get('blocked_top', [])} "
                f"cluster_blocked_top={ca.get('blocked_top', [])} "
                f"cooc_blocked_top={oa.get('blocked_top', [])} "
                f"dense_blocked_samples={dense_samples} "
                f"dense_no_domain_weak_samples={weak_no_dom}"
            )
        if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG and STAGE2_NOISY_DEBUG:
            d3 = [getattr(c, "term", c) for c in dense_list[:3]]
            c3 = [getattr(c, "term", c) for c in cluster_list[:3]]
            o3 = [getattr(c, "term", c) for c in cooc_list[:3]]
            print(
                f"  dense={len(dense_list)} cluster={len(cluster_list)} cooc={len(cooc_list)} "
                f"carryover={len(carryover_terms)} 前3: dense={d3} cooc={o3}"
            )
        debug_print(2, (
            f"[Stage2B Expansion Summary] dense_kept={len(dense_list)} | "
            f"cluster_kept={len(cluster_list)} | cooc_kept={len(cooc_list)} | "
            f"carryover={len(carryover_terms)} -> merged 本锚"
        ), label)
        merged = merge_primary_and_support_terms(
            carryover_terms,
            dense_list,
            cluster_list,
            cooc_list,
            label,
            active_domains=active_domains,
            jd_field_ids=jd_field_ids,
            jd_subfield_ids=jd_subfield_ids,
            jd_topic_ids=jd_topic_ids,
        )
        all_terms.extend(merged)
        _stage2_header("Stage2B 本锚合并", "-")
        if LABEL_EXPANSION_DEBUG:
            print(f"[Stage2] 锚点 anchor={anchor.anchor!r} 本锚合并 +{len(merged)} 项 累计 {len(all_terms)} 项")
            _exp_n = len(dense_list) + len(cluster_list) + len(cooc_list)
            print(
                f"[Stage2B retain summary] anchor={getattr(anchor, 'anchor', '')!r} "
                f"seed_candidates_for_stage2b={len(seed_candidates)} eligible_seeds={eligible_seed_count} "
                f"expansion_evidence_added={_exp_n} carryover={len(carryover_terms)} merged={len(merged)}"
            )
    # 诊断：从 similar_to_raw_rows 聚合出 similar_to_agg；从最终 all_terms 中筛出 similar_to 来源的项写入 similar_to_pass
    if getattr(label, "debug_info", None) is not None:
        raw_rows = getattr(label.debug_info, "similar_to_raw_rows", None) or []
        by_tid_agg = {}
        for r in raw_rows:
            tid = r.get("tid")
            if tid is None:
                continue
            tid_key = int(tid) if isinstance(tid, (int, float)) or (isinstance(tid, str) and tid.isdigit()) else tid
            if tid_key not in by_tid_agg:
                by_tid_agg[tid_key] = {"tid": tid, "term": r.get("term", ""), "sim_score": 0.0, "src_vids": []}
            by_tid_agg[tid_key]["sim_score"] = max(by_tid_agg[tid_key]["sim_score"], float(r.get("sim_score", 0) or 0))
            src_vid = r.get("src_vid")
            if src_vid is not None and src_vid not in by_tid_agg[tid_key]["src_vids"]:
                by_tid_agg[tid_key]["src_vids"].append(src_vid)
        label.debug_info.similar_to_agg = [
            {"tid": v["tid"], "term": v["term"], "sim_score": v["sim_score"], "hit_count": len(v["src_vids"]), "src_vids": sorted(v["src_vids"])}
            for v in by_tid_agg.values()
        ]
        similar_to_vids = {c.vid for c in all_terms if (getattr(c, "source", "") or "").strip().lower() == "similar_to"}
        pass_list = []
        for tid_key, agg in by_tid_agg.items():
            if tid_key not in similar_to_vids:
                continue
            pass_list.append({
                "tid": agg["tid"],
                "term": agg["term"],
                "sim_score": float(agg["sim_score"]),
                "hit_count": len(agg["src_vids"]),
                "src_vids": agg["src_vids"],
                "degree_w": 0,
                "degree_w_expanded": 0,
                "target_degree_w": 0,
                "domain_span": 0,
            })
        label.debug_info.similar_to_pass = pass_list
    _stage2_header("Stage2 结束汇总", "=")
    if LABEL_EXPANSION_DEBUG and STAGE2_VERBOSE_DEBUG:
        by_role: Dict[str, int] = collections.Counter(getattr(t, "term_role", "?") for t in all_terms)
        print(f"  总学术词数: {len(all_terms)}")
        print(f"  按 term_role: {dict(by_role)}")
    if LABEL_EXPANSION_DEBUG:
        print(f"[Stage2] stage2_generate_academic_terms 结束 总学术词数={len(all_terms)}")
    return all_terms

