import faiss

import json

import re

import sqlite3

import time

from collections import defaultdict

from typing import Any, Dict, List, Optional, Tuple



import numpy as np



try:
    from config import ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH, DB_PATH, JOB_INDEX_PATH, JOB_MAP_PATH
except Exception:
    # 兼容某些环境下 `config` 被第三方同名包遮蔽：将项目根目录提前到 sys.path
    import os
    import sys

    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    if "config" in sys.modules:
        del sys.modules["config"]
    from config import ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH, DB_PATH, JOB_INDEX_PATH, JOB_MAP_PATH

from src.utils.domain_utils import DomainProcessor

from src.utils.tools import apply_text_decay, get_decay_rate_for_domains

from src.utils.time_features import compute_paper_recency, compute_author_time_features

from src.core.recall.works_to_authors import accumulate_author_scores

from src.core.recall.vector_query_builder import build_vector_query_bundle, format_query_bundle_summary



# SQLite 单条语句绑定参数上限（常见编译为 999）；IN 列表过长会触发 OperationalError: too many SQL variables

_SQLITE_MAX_VARS_PER_QUERY = 900



# Step2：多路 dense 融合权重（raw 主路；其余为补充）。分支未启用时不计入分母，避免单路时整体尺度被压低。

_MULTI_QUERY_FUSION_WEIGHTS: Dict[str, float] = {

    "raw_query": 0.55,

    "compressed_query": 0.20,

    "task_focused_query": 0.15,

    "method_focused_query": 0.10,

}



# 与 query_bundle 字段一致；Step4 对 clause_queries 做浅层补充检索

_MULTI_QUERY_BRANCH_KEYS: Tuple[str, ...] = (

    "raw_query",

    "compressed_query",

    "task_focused_query",

    "method_focused_query",

)



# paper record 中各分支得分字段名（便于 debug 与后续扩展）

_SCORE_FIELD_BY_BRANCH: Dict[str, str] = {

    "raw_query": "score_raw_query",

    "compressed_query": "score_compressed_query",

    "task_focused_query": "score_task_query",

    "method_focused_query": "score_method_query",

}


# --- Step3：paper 层轻量 hybrid（dense 主导；surface 为在 fused_dense 之上的有界加性修正，避免稀释 Step2 尺度）---

_HYBRID_W_LEX = 0.045

_HYBRID_W_ACR = 0.04

_HYBRID_W_TOOL = 0.035

_HYBRID_W_CONS = 0.02

_HYBRID_BUMP_CAP = 0.10


# --- Step4：clause 局部补充检索（浅层 topK；不替代 raw/compressed/task/method 主路）---

_CLAUSE_MAX_SELECTED = 6

_CLAUSE_MIN_LEN = 14

_CLAUSE_SEARCH_K = 110

_CLAUSE_BONUS_CAP = 0.048


# Step5：每位作者随结果输出的 evidence 条数上限（与 author_top_works 的 top_k 对齐，通常 ≤3）

_EVIDENCE_PAPERS_PER_AUTHOR = 4


_MIN_EN_STOP = frozenset(

    {

        "the",

        "and",

        "for",

        "with",

        "this",

        "that",

        "from",

        "have",

        "has",

        "are",

        "was",

        "were",

        "been",

        "being",

        "not",

        "but",

        "you",

        "all",

        "can",

        "her",

        "she",

        "may",

        "use",

        "any",

        "our",

        "out",

    }

)


def _normalize_surface_text(text: str) -> str:

    """英文小写 + 空白折叠，便于子串匹配；中文保持原样由调用方处理。"""

    if not text:

        return ""

    t = text.lower()

    t = re.sub(r"\s+", " ", t)

    return t.strip()


def _extract_query_surface_terms(query_bundle: Dict[str, Any]) -> Dict[str, Any]:

    """

    从 query bundle 动态抽 surface terms，不依赖大规模静态词表。

    - lexical：raw/compressed/task 的通用词面（中英）

    - acronym：全大写、混合大小写技术缩写、字母数字混合短 token

    - method_tool：以 method_focused 为主，并并入 compressed 中的英文技术词

    """

    raw = (query_bundle.get("raw_query") or "").strip()

    comp = (query_bundle.get("compressed_query") or "").strip()

    task = (query_bundle.get("task_focused_query") or "").strip()

    method = (query_bundle.get("method_focused_query") or "").strip()

    blob = " ".join([x for x in (raw, comp, task, method) if x])

    lexical: List[str] = []

    acronym: List[str] = []

    method_tool: List[str] = []



    def _add_unique(bucket: List[str], term: str, cap: int = 96) -> None:

        t = term.strip()

        if len(t) < 2 or len(bucket) >= cap:

            return

        if t not in bucket:

            bucket.append(t)



    # 英文长 token（潜在一般词面）

    for m in re.finditer(r"[A-Za-z][A-Za-z0-9/+.\-]{2,}", blob):

        w = m.group(0)

        wl = w.lower()

        if wl in _MIN_EN_STOP:

            continue

        if len(w) >= 3:

            _add_unique(lexical, w)



    # 中文片段（2~8 字）

    for m in re.finditer(r"[\u4e00-\u9fff]{2,8}", blob):

        _add_unique(lexical, m.group(0))



    # 缩写 / 方法名片段：全大写、驼峰、iLQR 类

    for m in re.finditer(r"\b[A-Z]{2,}\b", blob):

        _add_unique(acronym, m.group(0))

    for m in re.finditer(r"\b[A-Za-z]{1,3}\d+[A-Za-z0-9]*\b|\b[A-Za-z]*[A-Z][a-z0-9]+[A-Za-z0-9]*\b", blob):

        _add_unique(acronym, m.group(0))



    # 斜杠列举：RRT/PRM/MPC

    for part in re.split(r"[/，,;\s]+", blob):

        p = part.strip()

        if 2 <= len(p) <= 16 and re.match(r"^[A-Za-z0-9+\-]+$", p):

            if any(c.isupper() for c in p):

                _add_unique(acronym, p)



    method_blob = " ".join([method, comp])

    for m in re.finditer(r"[A-Za-z][A-Za-z0-9/+.\-]{2,}", method_blob):

        _add_unique(method_tool, m.group(0))

    for m in re.finditer(r"[\u4e00-\u9fff]{2,8}", method_blob):

        _add_unique(method_tool, m.group(0))



    # 从 lexical 去掉与 acronym 完全重复的短项，降低重复计数

    acr_set = {a.lower() for a in acronym}

    lexical = [x for x in lexical if x.lower() not in acr_set or len(x) > 4]



    has_any = bool(lexical or acronym or method_tool)

    return {

        "lexical_terms": lexical[:96],

        "acronym_terms": acronym[:64],

        "method_tool_terms": method_tool[:64],

        "has_query_terms": has_any,

    }



def _term_hit_in_title(title: str, term: str) -> bool:

    """轻量命中：中文子串；英文用词界或子串（短词用词界减噪）。"""

    if not title or not term:

        return False

    t = term.strip()

    if len(t) < 2:

        return False

    if re.search(r"[\u4e00-\u9fff]", t):

        return t in title

    tl = title.lower()

    tlw = t.lower()

    if len(tlw) <= 5 and tlw.encode("utf-8").isalpha():

        return re.search(r"\b" + re.escape(tlw) + r"\b", tl) is not None

    return tlw in tl



def _coverage_ratio(terms: List[str], title: str) -> float:

    if not terms:

        return 0.0

    hits = sum(1 for x in terms if _term_hit_in_title(title, x))

    return float(min(1.0, hits / float(len(terms))))



def _multi_query_consistency_surface(rec: Dict[str, Any]) -> float:

    """

    与 Step2 的 multi-hit dense bonus 正交：奖励「任务视角 + 方法视角」同时命中等结构，

    数值仅作 0~1 特征，再乘极小权重，避免重复放大 Step2。

    """

    hits = set(rec.get("hit_query_types") or [])

    v = 0.0

    if "task_focused_query" in hits and "method_focused_query" in hits:

        v = 1.0

    elif "task_focused_query" in hits or "method_focused_query" in hits:

        v = 0.5

    if "compressed_query" in hits and "raw_query" in hits:

        v = min(1.0, v + 0.25)

    return float(min(1.0, v))



def _compute_paper_surface_match_features(

    paper_title: str,

    query_terms: Optional[Dict[str, Any]],

    paper_record: Dict[str, Any],

) -> Dict[str, Any]:

    """单篇 paper 的 surface 特征，全部压到 [0,1]。"""

    qt = query_terms or {}

    title = paper_title or ""

    lex = qt.get("lexical_terms") or []

    acr = qt.get("acronym_terms") or []

    mt = qt.get("method_tool_terms") or []



    lexical_coverage = _coverage_ratio(lex, title)

    acronym_bonus = _coverage_ratio(acr, title)

    method_tool_bonus = _coverage_ratio(mt, title)

    consistency_bonus = _multi_query_consistency_surface(paper_record)



    return {

        "lexical_coverage": float(lexical_coverage),

        "acronym_bonus": float(acronym_bonus),

        "method_tool_bonus": float(method_tool_bonus),

        "consistency_bonus": float(consistency_bonus),

    }



def _compute_paper_hybrid_score(

    fused_dense_score: float,

    surface_features: Dict[str, Any],

    has_query_terms: bool,

) -> float:

    """

    在 Step2 的 fused_dense_score 之上叠加有界 surface bump（dense 仍为主分；surface 全 0 时与 Step2 完全一致）。

    bump = Σ w_i * feat_i，并 cap 在 _HYBRID_BUMP_CAP，避免单靠术语把小 dense 论文顶到最前。

    """

    fd = float(min(1.0, max(0.0, fused_dense_score)))

    if not has_query_terms:

        return fd

    bump = (

        float(surface_features.get("lexical_coverage", 0.0)) * _HYBRID_W_LEX

        + float(surface_features.get("acronym_bonus", 0.0)) * _HYBRID_W_ACR

        + float(surface_features.get("method_tool_bonus", 0.0)) * _HYBRID_W_TOOL

        + float(surface_features.get("consistency_bonus", 0.0)) * _HYBRID_W_CONS

    )

    bump = min(_HYBRID_BUMP_CAP, bump)

    return float(min(1.0, fd + bump))



def _norm_clause_key(text: str) -> str:

    return re.sub(r"\s+", "", (text or "").strip().lower())



def _compute_clause_bonus(rec: Dict[str, Any]) -> float:

    """Step4：与 Step2/3 信号解耦的轻量局部 clause bonus；上界 _CLAUSE_BONUS_CAP。"""

    hc = int(rec.get("hit_clause_count") or 0)

    if hc <= 0:

        return 0.0

    best = float(rec.get("clause_best_score") or 0.0)

    cov = float(rec.get("clause_coverage_ratio") or 0.0)

    bonus = 0.007 * min(hc, 6) + 0.022 * best * (0.45 + 0.55 * cov)

    return float(min(_CLAUSE_BONUS_CAP, bonus))



def _select_effective_clause_queries(

    clause_queries: Optional[List[str]],

    query_bundle: Dict[str, Any],

) -> List[str]:

    """从 Step1 的 clause_queries 中选少量可检索子句：过短/与全局视角重复/前缀重复的跳过。"""

    gnorm: set = set()

    for k in ("raw_query", "compressed_query", "task_focused_query", "method_focused_query"):

        t = (query_bundle.get(k) or "").strip()

        if t:

            gnorm.add(_norm_clause_key(t))

    out: List[str] = []

    seen_norm: set = set()

    for c in clause_queries or []:

        t = (c or "").strip()

        if len(t) < _CLAUSE_MIN_LEN:

            continue

        if len(t) > 520:

            continue

        nk = _norm_clause_key(t)

        if len(nk) < 12:

            continue

        if nk in gnorm or nk in seen_norm:

            continue

        dup_prefix = False

        for ex in out:

            if min(len(t), len(ex)) >= 36 and t[:36] == ex[:36]:

                dup_prefix = True

                break

        if dup_prefix:

            continue

        out.append(t)

        seen_norm.add(nk)

        if len(out) >= _CLAUSE_MAX_SELECTED:

            break

    return out



def _apply_clause_hits_to_merged_records(

    merged_records: Dict[str, Dict[str, Any]],

    selected_clauses: List[str],

    per_clause_maps: Dict[str, Dict[str, float]],

) -> None:

    """将各 clause 的命中并入 paper 记录；仅 clause 命中的 wid 以较低 synthetic fused_dense 进入候选池（补覆盖）。"""

    selected_n = len(selected_clauses)

    if selected_n == 0:

        return

    wid_detail: Dict[str, Dict[str, float]] = defaultdict(dict)

    for i, _txt in enumerate(selected_clauses):

        cid = f"c{i}"

        for wid, sim in (per_clause_maps.get(cid) or {}).items():

            cur = wid_detail[wid]

            cur[cid] = max(float(sim), float(cur.get(cid, 0.0)))

    for wid, detail in wid_detail.items():

        best = max(detail.values()) if detail else 0.0

        cids = sorted(detail.keys())

        if wid in merged_records:

            rec = merged_records[wid]

            rec["hit_clause_ids"] = cids

            rec["hit_clause_count"] = len(cids)

            rec["clause_best_score"] = best

            rec["clause_scores_detail"] = dict(detail)

            rec["clause_selected_n"] = selected_n

            rec["clause_coverage_ratio"] = float(len(cids)) / float(max(1, selected_n))

            merged_records[wid] = rec

        else:

            fd = float(min(0.40, 0.09 + 0.58 * best))

            merged_records[wid] = {

                "wid": wid,

                "score_raw_query": None,

                "score_compressed_query": None,

                "score_task_query": None,

                "score_method_query": None,

                "hit_query_types": [],

                "hit_clause_ids": cids,

                "hit_clause_count": len(cids),

                "clause_best_score": best,

                "clause_scores_detail": dict(detail),

                "clause_selected_n": selected_n,

                "clause_coverage_ratio": float(len(cids)) / float(max(1, selected_n)),

                "fused_dense_score": fd,

                "best_dense_score": best * 0.55,

                "query_hit_count": 0,

                "clause_only_entry": True,

            }

    for wid, rec in merged_records.items():

        if wid in wid_detail:

            continue

        rec.setdefault("hit_clause_ids", [])

        rec.setdefault("hit_clause_count", 0)

        rec.setdefault("clause_best_score", 0.0)

        rec.setdefault("clause_scores_detail", {})

        rec.setdefault("clause_selected_n", selected_n)

        rec.setdefault("clause_coverage_ratio", 0.0)



def _build_paper_evidence_item(

    wid: str,

    merged_records: Dict[str, Dict[str, Any]],

    meta_dict: Dict[str, Any],

) -> Dict[str, Any]:

    """单篇论文的可序列化 evidence 片段（来自 Step2~4 已有字段，非 _last_debug 全量）。"""

    rec = merged_records.get(wid) or {}

    meta = meta_dict.get(wid) or {}

    sf = rec.get("surface_features") or {}

    aware = float(rec.get("paper_clause_aware_score") or rec.get("paper_hybrid_score") or 0.0)

    return {

        "wid": wid,

        "title": str(meta.get("title") or "")[:500],

        "year": meta.get("year"),

        "score_dense": float(rec.get("fused_dense_score") or 0.0),

        "score_hybrid": float(rec.get("paper_hybrid_score") or 0.0),

        "score_clause_aware": aware,

        "hit_query_types": list(rec.get("hit_query_types") or []),

        "query_hit_count": int(rec.get("query_hit_count") or 0),

        "hit_clause_ids": list(rec.get("hit_clause_ids") or []),

        "hit_clause_count": int(rec.get("hit_clause_count") or 0),

        "clause_best_score": float(rec.get("clause_best_score") or 0.0),

        "lexical_coverage": float(sf.get("lexical_coverage") or 0.0),

        "acronym_bonus": float(sf.get("acronym_bonus") or 0.0),

        "method_tool_bonus": float(sf.get("method_tool_bonus") or 0.0),

    }



def _select_author_evidence_paper_items(

    author_works: List[Tuple[str, float]],

    merged_records: Dict[str, Dict[str, Any]],

    meta_dict: Dict[str, Any],

    top_k: int,

) -> List[Dict[str, Any]]:

    """在作者 Top 贡献论文中，按最终 paper 分（clause_aware 优先）取少量 evidence。"""

    if not author_works:

        return []

    wids = [w for w, _ in author_works]

    def _final_score(w: str) -> float:

        r = merged_records.get(w) or {}

        return float(

            r.get("paper_clause_aware_score")

            or r.get("paper_hybrid_score")

            or r.get("fused_dense_score")

            or 0.0

        )

    wids_sorted = sorted(wids, key=_final_score, reverse=True)[:top_k]

    return [_build_paper_evidence_item(w, merged_records, meta_dict) for w in wids_sorted]



def _summarize_author_vector_evidence(paper_items: List[Dict[str, Any]]) -> Dict[str, Any]:

    """作者级轻量摘要，供候选池 / 解释层消费。"""

    if not paper_items:

        return {

            "top_evidence_count": 0,

            "best_paper_score": 0.0,

            "max_query_hit_count": 0,

            "max_clause_hit_count": 0,

            "query_type_coverage": [],

            "evidence_sources_summary": {"query_branches": [], "clause_signal_present": False},

        }

    qt_all: set = set()

    for p in paper_items:

        for t in p.get("hit_query_types") or []:

            qt_all.add(t)

    best = max(float(p.get("score_clause_aware") or 0.0) for p in paper_items)

    max_qh = max(int(p.get("query_hit_count") or 0) for p in paper_items)

    max_ch = max(int(p.get("hit_clause_count") or 0) for p in paper_items)

    clause_sig = any(int(p.get("hit_clause_count") or 0) > 0 for p in paper_items)

    return {

        "top_evidence_count": len(paper_items),

        "best_paper_score": float(best),

        "max_query_hit_count": max_qh,

        "max_clause_hit_count": max_ch,

        "query_type_coverage": sorted(qt_all),

        "evidence_sources_summary": {

            "query_branches": sorted(qt_all),

            "clause_signal_present": bool(clause_sig),

        },

    }


def _safe_join_text_parts(parts: List[Any]) -> str:

    out: List[str] = []

    for p in parts:

        if not p:

            continue

        if isinstance(p, str):

            s = p.strip()

            if s:

                out.append(s)

            continue

        if isinstance(p, (list, tuple)):

            for x in p:

                if isinstance(x, str):

                    sx = x.strip()

                    if sx:

                        out.append(sx)

                elif x is not None:

                    sx = str(x).strip()

                    if sx:

                        out.append(sx)

            continue

        if isinstance(p, dict):

            for k in ("name", "display_name", "keyword", "term", "text", "label", "value"):

                v = p.get(k)

                if isinstance(v, str):

                    sv = v.strip()

                    if sv:

                        out.append(sv)

            continue

        s = str(p).strip()

        if s:

            out.append(s)

    return " ".join(out)


def _build_vector_job_axis_diagnostics(evidence_papers: List[Dict[str, Any]]) -> Dict[str, Any]:

    """

    只读诊断：从 evidence_papers 的已有字段中拼文本，做轻量 job-axis 命中与风险标记。

    注意：不参与任何打分/排序/过滤。

    """

    papers = evidence_papers or []

    titles: List[str] = []

    text_parts: List[Any] = []



    for p in papers:

        if not isinstance(p, dict):

            continue

        t = (p.get("title") or "").strip()

        if t:

            titles.append(t)

        for k in ("title", "abstract", "summary", "concepts", "keywords", "terms", "tags", "fields"):

            if k in p and p.get(k) is not None:

                text_parts.append(p.get(k))



    blob = _safe_join_text_parts(text_parts).lower()



    axis_rules: List[Tuple[str, List[str]]] = [

        (

            "robotics",

            [

                "robot",

                "robotic",

                "robotics",

                "manipulator",

                "mobile robot",

                "robot arm",

                "humanoid",

                "legged robot",

                "autonomous robot",

            ],

        ),

        (

            "control",

            [

                "control",

                "controller",

                "control law",

                "motion control",

                "real-time control",

                "feedback control",

                "optimal control",

            ],

        ),

        (

            "dynamics_kinematics",

            [

                "kinematic",

                "kinematics",

                "dynamic",

                "dynamics",

                "inverse kinematics",

                "forward kinematics",

                "rigid body",

                "multibody",

            ],

        ),

        (

            "planning_trajectory",

            [

                "planning",

                "path planning",

                "motion planning",

                "trajectory",

                "trajectory optimization",

                "collision avoidance",

                "rrt",

                "prm",

                "chomp",

            ],

        ),

        (

            "optimization_optimal_control",

            [

                "optimization",

                "optimal control",

                "mpc",

                "ilqr",

                "ddp",

                "model predictive control",

            ],

        ),

        (

            "simulation_sim2real",

            [

                "simulation",

                "simulator",

                "sim-to-real",

                "sim2real",

                "gazebo",

                "mujoco",

                "isaac sim",

                "digital twin",

            ],

        ),

        (

            "manipulation_locomotion",

            [

                "manipulation",

                "grasping",

                "dexterous",

                "locomotion",

                "walking",

                "rolling locomotion",

                "bimanual",

            ],

        ),

        (

            "reinforcement_learning",

            [

                "reinforcement learning",

                "deep reinforcement learning",

                " rl ",

                "policy learning",

                "visuomotor",

            ],

        ),

        (

            "estimation_state",

            [

                "state estimation",

                "observer",

                "kalman",

                "localization",

                "estimation",

            ],

        ),

        (

            "engineering_implementation",

            [

                "ros",

                "ros2",

                "moveit",

                "pinocchio",

                "drake",

                "ocs2",

                "c++",

                "python",

                "real-time system",

            ],

        ),

    ]



    axis_hits: List[str] = []

    padded_blob = f" {blob} "

    for axis, keys in axis_rules:

        hit = False

        for kw in keys:

            if kw == " rl ":

                if " rl " in padded_blob:

                    hit = True

                    break

                continue

            if kw in blob:

                hit = True

                break

        if hit:

            axis_hits.append(axis)



    scores: List[float] = []

    for p in papers:

        if not isinstance(p, dict):

            continue

        s = p.get("score_clause_aware")

        if s is None:

            s = p.get("score_hybrid")

        if s is None:

            s = p.get("score_dense")

        try:

            scores.append(float(s or 0.0))

        except Exception:

            scores.append(0.0)



    denom = float(sum(scores))

    dominance = float((max(scores) / denom) if denom > 1e-12 and scores else 0.0)



    axis_count = int(len(axis_hits))

    risk_flags: List[str] = []

    if axis_count == 0:

        risk_flags.append("weak_job_axis_coverage")

    elif axis_count == 1:

        risk_flags.append("thin_job_axis_coverage")



    if dominance >= 0.80:

        risk_flags.append("single_paper_dominated")



    top1_title = (titles[0] if titles else "")

    top1_l = top1_title.lower()

    off_context_phrases = [

        "solar habitat",

        "space habitat",

        "renewable energy habitat",

        "genome-wide association",

        "bayesian mixed model",

        "association power",

        "augmented reality",

        "medical robotics",

        "human-machine interface",

        "engineering optimization theory",

        "applied numerical methods",

    ]

    strong_axes = {"robotics", "control", "planning_trajectory", "dynamics_kinematics", "manipulation_locomotion"}

    has_strong_axis = any(a in strong_axes for a in axis_hits)

    if axis_count <= 1 and top1_l:

        for ph in off_context_phrases:

            if ph in top1_l and not has_strong_axis:

                risk_flags.append("possible_off_context_dense_match")

                break



    generic_book_patterns = [

        r"\\bintroduction to\\b",

        r"\\btextbook\\b",

        r"\\bhandbook\\b",

        r"\\btheory and practice\\b",

        r"\\bapplied numerical methods\\b",

        r"\\bengineering optimization\\b",

    ]

    if top1_l:

        for pat in generic_book_patterns:

            if re.search(pat, top1_l):

                risk_flags.append("generic_method_or_book")

                break



    top_titles = [t for t in (tt.strip() for tt in titles[:3]) if t]



    return {

        "vector_job_axis_hits": axis_hits,

        "vector_job_axis_count": axis_count,

        "vector_single_paper_dominance": float(dominance),

        "vector_risk_flags": risk_flags,

        "vector_top_evidence_titles": top_titles,

        "vector_source_work_count": int(len(papers)),

    }



def _faiss_dist_to_similarity(index: Any, x: float) -> float:

    """与旧版 vector_path 一致：IP 越大越好；L2 越小越好 → 转成 (0,1] 相似度。"""

    mt = getattr(index, "metric_type", None)

    if mt == faiss.METRIC_L2:

        return 1.0 / (1.0 + max(0.0, x))

    return float(x)





class VectorPath:

    """

    向量路召回：基于 SBERT；Step2 multi-query dense；Step3 paper 层轻量 hybrid（dense 仍主导）。

    """



    def __init__(self, recall_limit=300):

        self.search_k = 500  # 检索深度

        self.recall_limit = recall_limit



        # 1. 加载论文摘要索引

        self.index = faiss.read_index(ABSTRACT_INDEX_PATH)

        with open(ABSTRACT_MAP_PATH, "r", encoding="utf-8") as f:

            self.id_map = json.load(f)



        # 2. 加载岗位描述索引

        self.job_index = faiss.read_index(JOB_INDEX_PATH)

        with open(JOB_MAP_PATH, "r", encoding="utf-8") as f:

            self.job_id_map = json.load(f)



        # 惰性加载：仅当需要编码 compressed/task/method 时初始化（total_recall 仅传向量时可零编码）

        self._query_encoder = None



    def _get_query_encoder(self):

        if self._query_encoder is None:

            from src.core.recall.input_to_vector import QueryEncoder



            self._query_encoder = QueryEncoder()

        return self._query_encoder



    def _build_query_bundle_for_recall(self, query_text: Optional[str]) -> Dict[str, Any]:

        """

        调用 Step1 builder；query_text 缺失时退化为空串 bundle，保证不抛错。

        multi-query 缓解长 JD 语义平均化：任务/方法视角独立检索，仍由向量模型吸收语义，而非标签式概念判决。

        """

        try:

            return build_vector_query_bundle(query_text or "")

        except Exception:

            return build_vector_query_bundle("")



    def _encode_query_bundle(

        self,

        query_bundle: Dict[str, Any],

        base_query_vector: np.ndarray,

        query_text: Optional[str],

    ) -> Tuple[Dict[str, np.ndarray], List[str]]:

        """

        为各分支准备向量：raw 复用调用方传入的 base_query_vector（与上游 encode(JD) 对齐，避免重复编码）。

        compressed / task / method 在文本非空且与 raw 文本不等价时再 encode。

        返回 (branch_key -> (1,dim) float32 L2 归一化向量, 实际参与编码的分支名列表)。

        """

        v0 = np.asarray(base_query_vector, dtype=np.float32)

        if v0.ndim == 1:

            v0 = v0.reshape(1, -1)

        faiss.normalize_L2(v0)



        vec_map: Dict[str, np.ndarray] = {"raw_query": v0.copy()}

        raw_txt = (query_bundle.get("raw_query") or "").strip()



        texts_to_encode: List[Tuple[str, str]] = []

        for key in ("compressed_query", "task_focused_query", "method_focused_query"):

            t = (query_bundle.get(key) or "").strip()

            if not t:

                continue

            if raw_txt and t == raw_txt:

                vec_map[key] = v0.copy()

                continue

            # 与已缓存分支同文复用（避免 task/compressed 相同再算一次）

            reused = False

            for uk, uvec in vec_map.items():

                if uk == key:

                    continue

                ut = (query_bundle.get(uk) if uk in query_bundle else "") or ""

                if isinstance(ut, str) and ut.strip() == t:

                    vec_map[key] = uvec.copy()

                    reused = True

                    break

            if reused:

                continue

            texts_to_encode.append((key, t))



        if texts_to_encode:

            enc = self._get_query_encoder()

            uniq: List[str] = []

            seen = set()

            for _, tx in texts_to_encode:

                if tx not in seen:

                    seen.add(tx)

                    uniq.append(tx)

            if uniq:

                batch = enc.encode_batch(uniq)

                text_to_row = {t: batch[i : i + 1].copy() for i, t in enumerate(uniq)}

                for key, tx in texts_to_encode:

                    row = text_to_row[tx]

                    faiss.normalize_L2(row)

                    vec_map[key] = row

        # 稳定顺序：按分支定义序输出 used_types

        ordered = [k for k in _MULTI_QUERY_BRANCH_KEYS if k in vec_map]

        return vec_map, ordered



    def _search_papers_for_each_query(

        self, query_vec_map: Dict[str, np.ndarray]

    ) -> Dict[str, Dict[str, float]]:

        """

        每路 query 单独 Faiss search_k；返回 query_type -> {wid: similarity}（同 wid 取该路最优位次得分，即首次出现）。

        若两分支共享同一向量（例如 compressed 与 raw 同文复用），只 search 一次再复制结果，避免重复计算。

        """

        out: Dict[str, Dict[str, float]] = {}

        # 按向量内容去重：浮点逐元素比较（L2 归一化后已稳定）
        unique_groups: List[Tuple[np.ndarray, List[str]]] = []

        for qtype, vec in query_vec_map.items():

            merged = False

            for v0, keys in unique_groups:

                if v0.shape == vec.shape and np.allclose(v0, vec, rtol=0.0, atol=1e-6):

                    keys.append(qtype)

                    merged = True

                    break

            if not merged:

                unique_groups.append((vec, [qtype]))

        for vec, qtypes in unique_groups:

            scores, indices = self.index.search(vec, self.search_k)

            wid_to_sim: Dict[str, float] = {}

            for i, idx in enumerate(indices[0]):

                if 0 <= idx < len(self.id_map):

                    wid = self.id_map[idx]

                    sim = _faiss_dist_to_similarity(self.index, float(scores[0][i]))

                    if wid not in wid_to_sim:

                        wid_to_sim[wid] = sim

            for qt in qtypes:

                out[qt] = wid_to_sim

        return out



    def _search_papers_for_selected_clauses(

        self, selected_clauses: List[str]

    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:

        """

        Step4：每个子句独立浅层检索（深度 < 主 multi-query），clause_id 形如 c0,c1,...

        """

        if not selected_clauses:

            return {}, {}

        enc = self._get_query_encoder()

        batch = enc.encode_batch(selected_clauses)

        per_maps: Dict[str, Dict[str, float]] = {}

        per_counts: Dict[str, int] = {}

        for i, _txt in enumerate(selected_clauses):

            row = batch[i : i + 1].copy()

            faiss.normalize_L2(row)

            scores, indices = self.index.search(row, _CLAUSE_SEARCH_K)

            cid = f"c{i}"

            m: Dict[str, float] = {}

            for j, idx in enumerate(indices[0]):

                if 0 <= idx < len(self.id_map):

                    wid = self.id_map[idx]

                    sim = _faiss_dist_to_similarity(self.index, float(scores[0][j]))

                    if wid not in m:

                        m[wid] = sim

            per_maps[cid] = m

            per_counts[cid] = len(m)

        return per_maps, per_counts



    def _merge_multi_query_paper_hits(

        self, per_query_hits: Dict[str, Dict[str, float]], active_branches: List[str]

    ) -> Dict[str, Dict[str, Any]]:

        """

        按 wid 合并多路命中；每篇 paper 保留各分支得分、命中分支列表，供融合与审计。

        """

        records: Dict[str, Dict[str, Any]] = {}

        for qtype, wid_map in per_query_hits.items():

            sf = _SCORE_FIELD_BY_BRANCH.get(qtype)

            if not sf:

                continue

            for wid, sim in wid_map.items():

                if wid not in records:

                    records[wid] = {

                        "wid": wid,

                        "score_raw_query": None,

                        "score_compressed_query": None,

                        "score_task_query": None,

                        "score_method_query": None,

                        "hit_query_types": [],

                    }

                rec = records[wid]

                rec[sf] = float(sim)

                if qtype not in rec["hit_query_types"]:

                    rec["hit_query_types"].append(qtype)

        for rec in records.values():

            rec["hit_query_types"] = sorted(

                rec["hit_query_types"],

                key=lambda t: _MULTI_QUERY_BRANCH_KEYS.index(t) if t in _MULTI_QUERY_BRANCH_KEYS else 99,

            )

        for wid, rec in records.items():

            vals = []

            for q in active_branches:

                sf = _SCORE_FIELD_BY_BRANCH[q]

                v = rec.get(sf)

                if v is not None:

                    vals.append(v)

            rec["best_dense_score"] = max(vals) if vals else 0.0

            rec["query_hit_count"] = len(rec["hit_query_types"])

            rec["fused_dense_score"] = self._score_multi_query_paper_record(rec, active_branches)

        return records



    def _score_multi_query_paper_record(self, rec: Dict[str, Any], active_branches: List[str]) -> float:

        """

        保守融合：对**已启用**分支做加权平均（分母为启用分支权重和），避免未启用分支稀释 raw。

        多路同时命中给予温和 bonus，上限 0.05，防止 method 单分支压过 raw 强相关。

        """

        wsum = sum(_MULTI_QUERY_FUSION_WEIGHTS[q] for q in active_branches if q in _MULTI_QUERY_FUSION_WEIGHTS)

        if wsum <= 1e-9:

            return 0.0

        num = 0.0

        for q in active_branches:

            sf = _SCORE_FIELD_BY_BRANCH[q]

            s = rec.get(sf)

            if s is None:

                s = 0.0

            num += _MULTI_QUERY_FUSION_WEIGHTS[q] * float(s)

        base = num / wsum

        hc = int(rec.get("query_hit_count") or 0)

        bonus = min(0.05, 0.012 * max(0, hc - 1))

        return float(min(1.0, base + bonus))



    def _attach_vector_evidence_to_meta_list(

        self,

        meta_list: List[Dict[str, Any]],

        agg_result: Any,

        merged_records: Dict[str, Dict[str, Any]],

        meta_dict: Dict[str, Any],

    ) -> Dict[str, Any]:

        """

        Step5：将 Step2~4 已算好的 paper 级信息收敛为每位作者的 vector_evidence（JSON 友好，非 debug 全量）。

        """

        author_evidence_preview: List[Dict[str, Any]] = []

        total_evidence_papers = 0

        atw = agg_result.author_top_works

        for item in meta_list:

            aid = str(item["author_id"])

            works = atw.get(aid, [])

            if not works:

                for k, v in atw.items():

                    if str(k) == aid:

                        works = v

                        break

            papers = _select_author_evidence_paper_items(

                works,

                merged_records,

                meta_dict,

                _EVIDENCE_PAPERS_PER_AUTHOR,

            )

            summary = _summarize_author_vector_evidence(papers)

            item["vector_evidence"] = {"top_papers": papers, "summary": summary}
            item.update(_build_vector_job_axis_diagnostics(papers))

            total_evidence_papers += len(papers)

            if len(author_evidence_preview) < 8:

                author_evidence_preview.append(

                    {

                        "author_id": aid,

                        "top_evidence_count": summary.get("top_evidence_count"),

                        "best_paper_score": summary.get("best_paper_score"),

                        "query_type_coverage": summary.get("query_type_coverage"),

                    }

                )

        n_auth = max(1, len(meta_list))

        author_evidence_stats = {

            "authors_with_evidence": len(meta_list),

            "avg_evidence_papers_per_author": float(total_evidence_papers) / float(n_auth),

            "total_evidence_papers": total_evidence_papers,

        }

        vec_top: Dict[str, Any] = {}

        if meta_list:

            ev0 = meta_list[0].get("vector_evidence") or {}

            tp0 = ev0.get("top_papers") or []

            vec_top = {

                "author_id": meta_list[0].get("author_id"),

                "summary": ev0.get("summary"),

                "first_paper_wid": (tp0[0].get("wid") if tp0 else None),

                "first_paper_score_clause_aware": (tp0[0].get("score_clause_aware") if tp0 else None),

            }

        return {

            "author_evidence_preview": author_evidence_preview,

            "author_evidence_stats": author_evidence_stats,

            "vector_evidence_top_author_preview": vec_top,

        }



    def recall(self, query_vector, target_domains=None, verbose=False, query_text=None):

        """

        向量路召回：实现基于领域 ID 的论文级硬过滤

        :param query_vector: 输入向量（与上游对 JD 的编码一致时，作为 raw_query 分支，避免重复 encode）

        :param target_domains: 领域 ID 列表或字符串，例如 ['1', '4'] 或 '1|4'

        :param verbose: 是否打印中间调试信息

        :param query_text: 原始 JD 文本（可选）；用于 Step1 bundle 与 Step2 多路编码；不传则仅 raw 分支有向量

        """

        start_t = time.time()

        conn = sqlite3.connect(DB_PATH)

        meta_list: List[Dict[str, Any]] = []



        query_bundle = self._build_query_bundle_for_recall(query_text)

        qb_summary = format_query_bundle_summary(query_bundle)

        clause_reserved = query_bundle.get("clause_queries") or []

        self._last_debug = {

            "query_bundle": query_bundle,

            "query_bundle_summary": qb_summary,

            "clause_queries_reserved_count": len(clause_reserved),

        }

        if verbose:

            print(f"[VectorPath] query_bundle: {qb_summary}")
            try:
                clauses = query_bundle.get("clause_queries") or []
                kept = query_bundle.get("clause_queries_kept") or clauses
                removed = query_bundle.get("clause_queries_removed") or []
                comp = (query_bundle.get("compressed_query") or "")
                task = (query_bundle.get("task_focused_query") or "")
                method = (query_bundle.get("method_focused_query") or "")
                print(
                    "[VectorPath] query_bundle_detail: "
                    f"clauses={len(clauses)} kept={len(kept)} removed={len(removed)} "
                    f"compressed_len={len(comp)} task_len={len(task)} method_len={len(method)} "
                    f"branches_used={query_vector_types_used if 'query_vector_types_used' in locals() else None}"
                )
            except Exception:
                pass



        target_set = DomainProcessor.to_set(target_domains) if target_domains else None

        decay_rate = get_decay_rate_for_domains(target_set or [])

        purity_min = 1.0

        try:

            # --- Step2：multi-query retrieval → paper 层融合 → 得到 wid 序列与 dense 主分 ---

            query_vec_map, query_vector_types_used = self._encode_query_bundle(

                query_bundle, query_vector, query_text

            )

            active_branches = [k for k in _MULTI_QUERY_BRANCH_KEYS if k in query_vec_map]



            per_query_hits = self._search_papers_for_each_query(query_vec_map)

            per_query_hit_counts = {k: len(per_query_hits.get(k, {})) for k in query_vec_map.keys()}



            merged_records = self._merge_multi_query_paper_hits(per_query_hits, active_branches)

            merged_paper_candidate_count = len(merged_records)

            # --- Step4：clause 局部多视角补充检索（浅层），并入 merged_records ---

            selected_clause_queries = _select_effective_clause_queries(clause_reserved, query_bundle)

            per_clause_hit_counts: Dict[str, int] = {}

            clause_maps: Dict[str, Dict[str, float]] = {}

            if selected_clause_queries:

                clause_maps, per_clause_hit_counts = self._search_papers_for_selected_clauses(selected_clause_queries)

                _apply_clause_hits_to_merged_records(merged_records, selected_clause_queries, clause_maps)

            else:

                for _w, rec in merged_records.items():

                    rec.setdefault("hit_clause_ids", [])

                    rec.setdefault("hit_clause_count", 0)

                    rec.setdefault("clause_best_score", 0.0)

                    rec.setdefault("clause_coverage_ratio", 0.0)

                    rec.setdefault("clause_selected_n", 0)

                    rec.setdefault("clause_scores_detail", {})

            merged_paper_candidate_count = len(merged_records)

            clause_merged_paper_count = merged_paper_candidate_count

            self._last_debug.update(

                {

                    "selected_clause_queries": list(selected_clause_queries),

                    "per_clause_hit_counts": per_clause_hit_counts,

                    "clause_merged_paper_count": clause_merged_paper_count,

                }

            )

            if verbose and selected_clause_queries:

                print(

                    f"[VectorPath] clause retrieval: selected={len(selected_clause_queries)} "

                    f"hits={per_clause_hit_counts} merged_papers={clause_merged_paper_count}"

                )

            # 按融合分排序，取前 search_k 进入原主链（与旧版「单路 topK」带宽对齐）

            sorted_wids = sorted(

                merged_records.keys(),

                key=lambda w: merged_records[w].get("fused_dense_score", 0.0),

                reverse=True,

            )[: self.search_k]



            raw_work_ids = sorted_wids

            merged_paper_top_preview: List[Dict[str, Any]] = []

            for w in raw_work_ids[:15]:

                r = merged_records.get(w) or {}

                merged_paper_top_preview.append(

                    {

                        "work_id": w,

                        "fused_dense_score": r.get("fused_dense_score"),

                        "hit_query_types": r.get("hit_query_types"),

                        "best_dense_score": r.get("best_dense_score"),

                    }

                )

            query_surface_terms = _extract_query_surface_terms(query_bundle)

            query_surface_terms_preview = {

                "lexical_terms": (query_surface_terms.get("lexical_terms") or [])[:28],

                "acronym_terms": (query_surface_terms.get("acronym_terms") or [])[:28],

                "method_tool_terms": (query_surface_terms.get("method_tool_terms") or [])[:28],

                "has_query_terms": query_surface_terms.get("has_query_terms", False),

            }

            self._last_debug.update(

                {

                    "per_query_hit_counts": per_query_hit_counts,

                    "merged_paper_candidate_count": merged_paper_candidate_count,

                    "merged_paper_top_preview": merged_paper_top_preview,

                    "query_vector_types_used": query_vector_types_used,

                    "multi_query_active_branches": active_branches,

                    "query_surface_terms_preview": query_surface_terms_preview,

                }

            )

            if verbose:

                print(

                    f"[VectorPath] multi-query: hits_per_branch={per_query_hit_counts} | "

                    f"merged_unique_papers={merged_paper_candidate_count} | "

                    f"branches_used={query_vector_types_used}"

                )
                print(
                    f"[VectorPath] clause_summary: selected={len(selected_clause_queries)} "
                    f"clause_merged_papers={clause_merged_paper_count}"
                )

                prev_titles = []

                for item in merged_paper_top_preview[:5]:

                    prev_titles.append(f"{item['work_id']}{item.get('hit_query_types')}")

                print(f"[VectorPath] merged_top5_wid+hit_types: {prev_titles}")

                print(

                    f"[VectorPath] hybrid terms: lexical={len(query_surface_terms.get('lexical_terms') or [])} "

                    f"acronym={len(query_surface_terms.get('acronym_terms') or [])} "

                    f"method_tool={len(query_surface_terms.get('method_tool_terms') or [])}"

                )

            if not raw_work_ids:

                self._last_debug.update(

                    {

                        "paper_hybrid_feature_preview": [],

                        "paper_hybrid_top_preview": [],

                        "hybrid_feature_stats": {

                            "mean_lexical_coverage": 0.0,

                            "mean_acronym_bonus": 0.0,

                            "mean_method_tool_bonus": 0.0,

                            "pct_any_surface_signal": 0.0,

                            "papers_count": 0,

                        },

                    }

                )

                return [], (time.time() - start_t) * 1000

            # --- 步骤 2: 获取论文的领域标签与元信息用于过滤与调试 ---

            placeholders = ",".join(["?"] * len(raw_work_ids))

            sql = f"SELECT work_id, domain_ids, title, year FROM works WHERE work_id IN ({placeholders})"

            work_data = conn.execute(sql, raw_work_ids).fetchall()

            domain_dict = {row[0]: row[1] for row in work_data}

            meta_dict = {row[0]: {"title": row[2], "year": row[3]} for row in work_data}

            # --- Step3：paper 层轻量 hybrid（title 匹配；dense 主干不变）---

            has_qt = bool(query_surface_terms.get("has_query_terms"))

            lex_vals: List[float] = []

            acr_vals: List[float] = []

            tool_vals: List[float] = []

            nonzero_hybrid = 0

            for wid in raw_work_ids:

                rec = merged_records.get(wid) or {}

                title = (meta_dict.get(wid, {}).get("title") or "")

                feats = _compute_paper_surface_match_features(title, query_surface_terms, rec)

                for k, arr in (

                    ("lexical_coverage", lex_vals),

                    ("acronym_bonus", acr_vals),

                    ("method_tool_bonus", tool_vals),

                ):

                    arr.append(float(feats.get(k, 0.0)))

                phs = _compute_paper_hybrid_score(float(rec.get("fused_dense_score", 0.0)), feats, has_qt)

                rec["surface_features"] = feats

                rec["paper_hybrid_score"] = phs

                cb = _compute_clause_bonus(rec)

                rec["clause_bonus"] = cb

                rec["paper_clause_aware_score"] = float(min(1.0, phs + cb))

                merged_records[wid] = rec

                if (feats.get("lexical_coverage", 0) + feats.get("acronym_bonus", 0) + feats.get("method_tool_bonus", 0)) > 0.02:

                    nonzero_hybrid += 1

            n_p = max(1, len(raw_work_ids))

            hybrid_feature_stats = {

                "mean_lexical_coverage": float(sum(lex_vals) / n_p),

                "mean_acronym_bonus": float(sum(acr_vals) / n_p),

                "mean_method_tool_bonus": float(sum(tool_vals) / n_p),

                "pct_any_surface_signal": float(nonzero_hybrid / float(n_p)),

                "papers_count": len(raw_work_ids),

            }

            raw_work_ids = sorted(

                raw_work_ids,

                key=lambda w: float((merged_records.get(w) or {}).get("paper_clause_aware_score", 0.0)),

                reverse=True,

            )

            faiss_score_map: Dict[str, float] = {

                w: float((merged_records.get(w) or {}).get("paper_clause_aware_score", 0.0)) for w in raw_work_ids

            }

            paper_hybrid_feature_preview: List[Dict[str, Any]] = []

            for w in raw_work_ids[:14]:

                r = merged_records.get(w) or {}

                sf = r.get("surface_features") or {}

                paper_hybrid_feature_preview.append(

                    {

                        "work_id": w,

                        "lexical_coverage": sf.get("lexical_coverage"),

                        "acronym_bonus": sf.get("acronym_bonus"),

                        "method_tool_bonus": sf.get("method_tool_bonus"),

                        "consistency_bonus": sf.get("consistency_bonus"),

                        "paper_hybrid_score": r.get("paper_hybrid_score"),

                        "clause_bonus": r.get("clause_bonus"),

                        "paper_clause_aware_score": r.get("paper_clause_aware_score"),

                        "fused_dense_score": r.get("fused_dense_score"),

                    }

                )

            paper_hybrid_top_preview: List[Dict[str, Any]] = []

            for w in raw_work_ids[:18]:

                r = merged_records.get(w) or {}

                t = (meta_dict.get(w, {}).get("title") or "")

                if len(t) > 120:

                    t = t[:117] + "..."

                sf = r.get("surface_features") or {}

                paper_hybrid_top_preview.append(

                    {

                        "work_id": w,

                        "title": t,

                        "fused_dense_score": r.get("fused_dense_score"),

                        "lexical_coverage": sf.get("lexical_coverage"),

                        "acronym_bonus": sf.get("acronym_bonus"),

                        "method_tool_bonus": sf.get("method_tool_bonus"),

                        "consistency_bonus": sf.get("consistency_bonus"),

                        "query_hit_count": r.get("query_hit_count"),

                        "paper_hybrid_score": r.get("paper_hybrid_score"),

                        "clause_bonus": r.get("clause_bonus"),

                        "paper_clause_aware_score": r.get("paper_clause_aware_score"),

                    }

                )

            clause_top_preview: List[Dict[str, Any]] = []

            cb_vals = [

                float((merged_records.get(w) or {}).get("clause_bonus") or 0.0) for w in raw_work_ids

            ]

            hit_any_clause = sum(

                1

                for w in raw_work_ids

                if int((merged_records.get(w) or {}).get("hit_clause_count") or 0) > 0

            )

            clause_feature_stats = {

                "mean_clause_bonus": float(sum(cb_vals) / max(1, len(cb_vals))),

                "pct_papers_with_clause_hit": float(hit_any_clause / float(max(1, len(raw_work_ids)))),

                "selected_clause_n": len(selected_clause_queries) if selected_clause_queries else 0,

            }

            for w in raw_work_ids[:14]:

                r = merged_records.get(w) or {}

                t = (meta_dict.get(w, {}).get("title") or "")

                if len(t) > 100:

                    t = t[:97] + "..."

                clause_top_preview.append(

                    {

                        "work_id": w,

                        "title": t,

                        "hit_clause_ids": r.get("hit_clause_ids"),

                        "hit_clause_count": r.get("hit_clause_count"),

                        "clause_best_score": r.get("clause_best_score"),

                        "clause_bonus": r.get("clause_bonus"),

                        "paper_hybrid_score": r.get("paper_hybrid_score"),

                        "paper_clause_aware_score": r.get("paper_clause_aware_score"),

                    }

                )

            self._last_debug.update(

                {

                    "paper_hybrid_feature_preview": paper_hybrid_feature_preview,

                    "paper_hybrid_top_preview": paper_hybrid_top_preview,

                    "hybrid_feature_stats": hybrid_feature_stats,

                    "clause_top_preview": clause_top_preview,

                    "clause_feature_stats": clause_feature_stats,

                }

            )

            if verbose:

                h1 = paper_hybrid_top_preview[0] if paper_hybrid_top_preview else {}

                print(

                    f"[VectorPath] hybrid top1: wid={h1.get('work_id')} "

                    f"dense={h1.get('fused_dense_score')} hybrid={h1.get('paper_hybrid_score')} "

                    f"lex={h1.get('lexical_coverage')} acr={h1.get('acronym_bonus')} tool={h1.get('method_tool_bonus')}"

                )

                c1 = clause_top_preview[0] if clause_top_preview else {}

                print(

                    f"[VectorPath] clause top1: wid={c1.get('work_id')} "

                    f"clause_hit={c1.get('hit_clause_count')} cb={c1.get('clause_bonus')} "

                    f"aware={c1.get('paper_clause_aware_score')}"

                )

            # --- 步骤 3: 领域硬过滤（只有对应领域的论文才能发挥作用） ---

            filtered_work_ids = []

            work_score_map = {}

            work_debug_map = {}

            for wid in raw_work_ids:

                if wid not in domain_dict:

                    continue



                if target_set:

                    work_domains_raw = domain_dict[wid]



                    if not DomainProcessor.has_intersect(work_domains_raw, target_set):

                        continue



                    paper_set = DomainProcessor.to_set(work_domains_raw)

                    purity = len(paper_set & target_set) / max(1, len(paper_set))

                    if purity < purity_min:

                        continue

                    domain_coeff = purity**4

                else:

                    domain_coeff = 1.0

                    purity = None



                filtered_work_ids.append(wid)

                base_sim = faiss_score_map.get(wid, 0.0)

                title = (meta_dict.get(wid, {}).get("title") or "")

                year_val = meta_dict.get(wid, {}).get("year")



                type_decay = apply_text_decay(title)

                time_decay = compute_paper_recency(year_val, target_set or [])



                work_score = base_sim * domain_coeff * type_decay * time_decay

                work_score_map[wid] = work_score



                if verbose:

                    mq = merged_records.get(wid, {})

                    sf = mq.get("surface_features") or {}

                    work_debug_map[wid] = {

                        "faiss_sim": float(base_sim),

                        "fused_dense_score": float(mq.get("fused_dense_score", base_sim)),

                        "paper_hybrid_score": float(mq.get("paper_hybrid_score", base_sim)),

                        "clause_bonus": float(mq.get("clause_bonus") or 0.0),

                        "paper_clause_aware_score": float(mq.get("paper_clause_aware_score", base_sim)),

                        "hit_clause_ids": mq.get("hit_clause_ids"),

                        "hit_clause_count": mq.get("hit_clause_count"),

                        "lexical_coverage": sf.get("lexical_coverage"),

                        "acronym_bonus": sf.get("acronym_bonus"),

                        "method_tool_bonus": sf.get("method_tool_bonus"),

                        "consistency_bonus": sf.get("consistency_bonus"),

                        "hit_query_types": mq.get("hit_query_types"),

                        "purity": None if purity is None else float(purity),

                        "domain_coeff": float(domain_coeff),

                        "work_score": float(work_score),

                        "domain_ids": domain_dict.get(wid),

                        "title": title,

                        "year": year_val,

                        "type_decay": float(type_decay),

                        "time_decay": float(time_decay),

                    }



            if not filtered_work_ids:

                return [], (time.time() - start_t) * 1000



            # --- 步骤 4: 统一走“论文 → 作者”分摊聚合逻辑（不重写） ---

            work_placeholders = ",".join(["?"] * len(filtered_work_ids))

            pairs_query = f"""

                SELECT author_id, work_id

                FROM authorships

                WHERE work_id IN ({work_placeholders})

            """

            rows = conn.execute(pairs_query, filtered_work_ids).fetchall()



            papers_by_wid = {

                wid: {"wid": wid, "score": float(work_score_map.get(wid, 0.0)), "authors": []}

                for wid in filtered_work_ids

                if wid in work_score_map

            }



            for aid, wid in rows:

                p = papers_by_wid.get(wid)

                if p is None:

                    continue

                p["authors"].append(

                    {

                        "aid": str(aid),

                        "pos_weight": 1.0,

                    }

                )



            papers = [p for p in papers_by_wid.values() if p["authors"]]



            agg_result = accumulate_author_scores(papers, top_k_per_author=3)

            author_scores = agg_result.author_scores



            author_ids = agg_result.sorted_authors()

            if author_ids:

                year_rows = []

                for off in range(0, len(author_ids), _SQLITE_MAX_VARS_PER_QUERY):

                    batch = author_ids[off : off + _SQLITE_MAX_VARS_PER_QUERY]

                    ph = ",".join(["?"] * len(batch))

                    year_rows.extend(

                        conn.execute(

                            f"""

                            SELECT a.author_id, w.year

                            FROM authorships a

                            JOIN works w ON a.work_id = w.work_id

                            WHERE a.author_id IN ({ph})

                            """,

                            batch,

                        ).fetchall()

                    )



                years_by_author = defaultdict(list)

                for aid, year in year_rows:

                    years_by_author[str(aid)].append(year)



                for aid in author_ids:

                    base_score = float(author_scores.get(aid, 0.0))

                    years = years_by_author.get(str(aid), [])

                    _, _, time_weight = compute_author_time_features(years)

                    author_scores[aid] = base_score * float(time_weight)



            author_ids = [aid for aid, _ in sorted(author_scores.items(), key=lambda x: x[1], reverse=True)]

            meta_list = [

                {

                    "author_id": str(aid),

                    "vector_score_raw": float(author_scores.get(aid, 0.0)),

                    "vector_rank": i + 1,

                    "vector_evidence": None,

                }

                for i, aid in enumerate(author_ids[: self.recall_limit])

            ]

            if verbose:
                before_diag_top20_author_ids = [str(x.get("author_id")) for x in meta_list[:20]]

            ev_dbg = self._attach_vector_evidence_to_meta_list(meta_list, agg_result, merged_records, meta_dict)

            self._last_debug.update(ev_dbg)

            if verbose:
                after_diag_top20_author_ids = [str(x.get("author_id")) for x in meta_list[:20]]
                same = before_diag_top20_author_ids == after_diag_top20_author_ids
                print(f"[VectorPath] diag_order_check: top20_same={same}")
                print(f"[VectorPath] before_diag_top20_author_ids={before_diag_top20_author_ids}")
                print(f"[VectorPath] after_diag_top20_author_ids={after_diag_top20_author_ids}")

                st = ev_dbg.get("author_evidence_stats") or {}

                print(

                    f"[VectorPath] evidence attached: authors={st.get('authors_with_evidence', 0)} "

                    f"avg_papers={st.get('avg_evidence_papers_per_author', 0.0):.2f}"

                )

                tp = ev_dbg.get("vector_evidence_top_author_preview") or {}

                print(

                    f"[VectorPath] evidence top1 author preview: aid={tp.get('author_id')} "

                    f"best={(tp.get('summary') or {}).get('best_paper_score')} "

                    f"first_wid={tp.get('first_paper_wid')}"

                )

                top20_items = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)[:20]



                top_work_ids = []

                author_top3 = {}

                for aid, _ in top20_items:

                    works = agg_result.author_top_works.get(aid, [])[:3]

                    author_top3[aid] = works

                    for wid, _ in works:

                        top_work_ids.append(wid)

                top_work_ids = list(dict.fromkeys(top_work_ids))



                work_meta = {}

                if top_work_ids:

                    ph = ",".join(["?"] * len(top_work_ids))

                    rows = conn.execute(

                        f"SELECT work_id, title, year FROM works WHERE work_id IN ({ph})",

                        top_work_ids,

                    ).fetchall()

                    work_meta = {r[0]: {"title": r[1], "year": r[2]} for r in rows}

                meta_by_aid = {str(x.get("author_id")): x for x in meta_list if x.get("author_id") is not None}

                # Step 1.5: 非 Top20 指定标题诊断检查（只在当前 recall 集合内查找）
                try:
                    wanted_titles = [
                        "A Mathematical Introduction to Robotic Manipulation",
                        "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware",
                        "Learning to Walk Via Deep Reinforcement Learning",
                        "Rolling Locomotion of Cable-Driven Soft Spherical Tensegrity Robots",
                    ]
                    current_wids = list(dict.fromkeys((raw_work_ids or []) + (filtered_work_ids or [])))
                    wid_to_title: Dict[str, str] = {}
                    for wid, meta in (meta_dict or {}).items():
                        wid_to_title[str(wid)] = str((meta or {}).get("title") or "")
                    missing = [w for w in current_wids if str(w) not in wid_to_title]
                    if missing:
                        for off in range(0, len(missing), _SQLITE_MAX_VARS_PER_QUERY):
                            batch = missing[off : off + _SQLITE_MAX_VARS_PER_QUERY]
                            ph = ",".join(["?"] * len(batch))
                            rows = conn.execute(
                                f"SELECT work_id, title FROM works WHERE work_id IN ({ph})",
                                batch,
                            ).fetchall()
                            for wid, t in rows:
                                wid_to_title[str(wid)] = str(t or "")

                    print("[VectorPath] title_axis_diagnostics_check:")
                    for wt in wanted_titles:
                        found_wid = None
                        for wid, t in wid_to_title.items():
                            if t and t.strip().lower() == wt.strip().lower():
                                found_wid = wid
                                break
                        if not found_wid:
                            print(f"    - {wt}: NOT_FOUND_IN_CURRENT_RECALL")
                            continue
                        diag = _build_vector_job_axis_diagnostics([{"title": wid_to_title.get(found_wid, "")}])
                        print(
                            f"    - {wt}: FOUND wid={found_wid} axis_hits={diag.get('vector_job_axis_hits')} "
                            f"axis_count={diag.get('vector_job_axis_count')} risk_flags={diag.get('vector_risk_flags')}"
                        )
                except Exception:
                    pass



                self._last_debug.update(

                    {

                        "query_bundle": query_bundle,

                        "query_bundle_summary": qb_summary,

                        "target_domains": target_domains,

                        "target_set": sorted(list(target_set)) if target_set else [],

                        "purity_min": purity_min,

                        "decay_rate": decay_rate,

                        "top20": [

                            {

                                "author_id": aid,

                                "author_score": float(a_score),
                                "vector_job_axis_hits": (meta_by_aid.get(str(aid), {}) or {}).get("vector_job_axis_hits", []),
                                "vector_job_axis_count": int((meta_by_aid.get(str(aid), {}) or {}).get("vector_job_axis_count", 0) or 0),
                                "vector_single_paper_dominance": float(
                                    (meta_by_aid.get(str(aid), {}) or {}).get("vector_single_paper_dominance", 0.0) or 0.0
                                ),
                                "vector_risk_flags": (meta_by_aid.get(str(aid), {}) or {}).get("vector_risk_flags", []),
                                "vector_top_evidence_titles": (meta_by_aid.get(str(aid), {}) or {}).get("vector_top_evidence_titles", []),
                                "vector_source_work_count": int((meta_by_aid.get(str(aid), {}) or {}).get("vector_source_work_count", 0) or 0),

                                "top3_works": [

                                    {

                                        "work_id": wid,

                                        **(work_meta.get(wid, {})),

                                        **(work_debug_map.get(wid, {})),

                                    }

                                    for wid, _ in author_top3.get(aid, [])

                                ],

                            }

                            for aid, a_score in top20_items

                        ],

                    }

                )



        finally:

            conn.close()



        duration = (time.time() - start_t) * 1000

        if not meta_list:

            return [], duration

        return meta_list, duration





if __name__ == "__main__":

    from src.core.recall.input_to_vector import QueryEncoder  # 确保路径正确

    from src.core.recall.label_path import LabelRecallPath

    from src.utils.domain_detector import DomainDetector



    v_path = VectorPath(recall_limit=300)

    encoder = QueryEncoder()

    l_path = LabelRecallPath(recall_limit=150)

    d_detector = DomainDetector(l_path)



    fields = {

        "1": "计算机科学",

        "2": "医学",

        "3": "政治学",

        "4": "工程学",

        "5": "物理学",

        "6": "材料科学",

        "7": "生物学",

        "8": "地理学",

        "9": "化学",

        "10": "商学",

        "11": "社会学",

        "12": "哲学",

        "13": "环境科学",

        "14": "数学",

        "15": "心理学",

        "16": "地质学",

        "17": "经济学",

    }



    def get_work_title(author_id):

        conn = sqlite3.connect(DB_PATH)

        res = conn.execute(

            """

                           SELECT w.title

                           FROM works w

                                    JOIN authorships a ON w.work_id = a.work_id

                           WHERE a.author_id = ? LIMIT 1

                           """,

            (author_id,),

        ).fetchone()

        conn.close()

        return res[0] if res else "无论文数据"



    print("\n" + "=" * 115)

    print("向量路 (Vector Path) 独立语义召回测试")

    print("-" * 115)

    f_list = list(fields.items())

    for i in range(0, len(f_list), 6):

        print(" | ".join([f"{k}:{v}" for k, v in f_list[i : i + 6]]))

    print("=" * 115)



    try:

        domain_choice = input("\n请选择领域编号 (1-17, 0跳过): ").strip() or "0"

        current_field = fields.get(domain_choice, "全领域")



        while True:

            user_input = input(f"\n[{current_field}] 请输入岗位需求 (q退出): ").strip()

            if not user_input or user_input.lower() == "q":

                break

            # 容错：部分 Windows 控制台/管道输入可能引入不可编码代理字符，导致 tokenizer 报类型错误
            raw_len = len(user_input)
            raw_head = user_input[:80]
            raw_tail = user_input[-80:] if len(user_input) > 80 else user_input
            try:
                user_input = user_input.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            except Exception:
                user_input = str(user_input)
            cleaned_len = len(user_input)
            cleaned_head = user_input[:80]
            cleaned_tail = user_input[-80:] if len(user_input) > 80 else user_input

            print(f"[Main Debug] domain_choice={domain_choice}")
            print(f"[Main Debug] jd_len_raw={raw_len} jd_len_cleaned={cleaned_len}")
            print(f"[Main Debug] jd_head_raw={raw_head}")
            print(f"[Main Debug] jd_tail_raw={raw_tail}")
            print(f"[Main Debug] jd_head_cleaned={cleaned_head}")
            print(f"[Main Debug] jd_tail_cleaned={cleaned_tail}")

            try:
                import os as _os
                from config import SBERT_DIR as _SBERT_DIR

                print(f"[Main Debug] cwd={_os.getcwd()}")
                print(f"[Main Debug] vector_path_file={__file__}")
                print(f"[Main Debug] sqlite_db_path={DB_PATH}")
                print(f"[Main Debug] faiss_abstract_index={ABSTRACT_INDEX_PATH}")
                print(f"[Main Debug] faiss_job_index={JOB_INDEX_PATH}")
                print(f"[Main Debug] sbert_dir={_SBERT_DIR}")
            except Exception:
                pass



            query_vec, _ = encoder.encode(user_input)

            faiss.normalize_L2(query_vec)



            if domain_choice != "0":

                user_domain = domain_choice

            else:

                user_domain = None



            active_domains, applied_domains_str, _ = d_detector.detect(

                query_vec, query_text=user_input, user_domain=user_domain

            )

            target_domains = "|".join(sorted(active_domains)) if active_domains else None



            v_meta, duration = v_path.recall(

                query_vec, target_domains=target_domains, verbose=True, query_text=user_input

            )



            print(f"\n[召回报告] 耗时: {duration:.2f}ms | 命中人数: {len(v_meta)} | 应用领域: {applied_domains_str}")

            print("-" * 115)

            print(f"{'排名':<6} | {'作者 ID':<12} | {'检索路径':<15} | {'代表作标题 (数据源: SQLite)'}")

            print("-" * 115)



            for rank, item in enumerate(v_meta[:20], 1):

                aid = item["author_id"]

                title = get_work_title(aid)

                if len(title) > 70:

                    title = title[:67] + "..."

                print(f"#{rank:<5} | {aid:<12} | {'Vector (V)':<15} | {title}")



            dbg = getattr(v_path, "_last_debug", None)

            if dbg and dbg.get("top20"):

                print("\n[Top20 Debug] author_score 与 Top3 work 贡献明细")

                print("-" * 115)

                for i, item in enumerate(dbg["top20"], 1):

                    aid = item["author_id"]

                    a_score = item["author_score"]

                    print(f"#{i:<3} {aid} | author_score={a_score:.6f}")

                    axis_hits = item.get("vector_job_axis_hits") or []
                    axis_count = int(item.get("vector_job_axis_count") or 0)
                    dominance = float(item.get("vector_single_paper_dominance") or 0.0)
                    risk_flags = item.get("vector_risk_flags") or []
                    top_titles = item.get("vector_top_evidence_titles") or []

                    print(f"    axis_hits={axis_hits}")
                    print(f"    axis_count={axis_count} | single_paper_dominance={dominance:.2f} | risk_flags={risk_flags}")
                    print("    top_evidence_titles=[")
                    for tt in top_titles[:3]:
                        tts = (tt or "").strip()
                        if len(tts) > 120:
                            tts = tts[:117] + "..."
                        print(f'        "{tts}",')
                    print("    ]")

                    for w in item.get("top3_works", []):

                        w_title = (w.get("title") or "N/A").strip()

                        if len(w_title) > 90:

                            w_title = w_title[:87] + "..."

                        print(

                            f"    - {w.get('work_id')} ({w.get('year')})"

                            f" | work_score={w.get('work_score', 0.0):.6f}"

                            f" | faiss={w.get('faiss_sim', 0.0):.6f}"

                            f" | purity={w.get('purity')}"

                            f" | domains={w.get('domain_ids')}"

                            f" | {w_title}"

                        )



            print("-" * 115)



    except KeyboardInterrupt:

        print("\n[!] 测试结束")

