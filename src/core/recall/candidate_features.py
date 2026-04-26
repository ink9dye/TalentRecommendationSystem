# -*- coding: utf-8 -*-
"""
候选池特征计算与 KGAT-AX 侧车构建。

提供：标签证据解析、6 个 query-author/作者特征小函数、分桶截断、KGAT-AX 特征行构建。
供 total_recall 与 RankScorer / 训练数据生成复用。
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from config import DB_PATH


def extract_terms_from_label_evidence(label_evidence: Any) -> List[Dict[str, Any]]:
    """
    从 label_evidence 中提取词项列表，每项含 bucket（core/support/risky）与 score。
    允许可选字段 term（用于解释展示）。
    若证据为 None 或格式不符，返回 []。
    """
    if label_evidence is None:
        return []
    if isinstance(label_evidence, list):
        return [
            {
                "term": x.get("term"),
                "bucket": (x.get("bucket") or "support"),
                "score": float(x.get("score") or 0.0),
            }
            for x in label_evidence
            if isinstance(x, dict)
        ]
    if isinstance(label_evidence, dict):
        terms = label_evidence.get("terms") or label_evidence.get("term_list")
        if isinstance(terms, list):
            return [
                {
                    "term": x.get("term"),
                    "bucket": (x.get("bucket") or "support"),
                    "score": float(x.get("score") or 0.0),
                }
                for x in terms
                if isinstance(x, dict)
            ]
    return []


# Step7/8：向量路 evidence 弱 bonus 的全局上界（不得上调；Step8 仅允许更小 cap / 门控 / 关闭）
_VECTOR_EVIDENCE_BONUS_CAP = 0.048


@dataclass(frozen=True)
class VectorEvidenceBonusConfig:
    """
    Step8：极小配置面，便于本地消融（关 bonus / 关 gate / 紧 cap），不做复杂实验框架。
    默认 gate_vector_origin=True：仅「确有向量路来源」的候选吃 bonus，比 Step7 更干净、可解释。
    """

    enabled: bool = True
    gate_vector_origin: bool = True
    cap: Optional[float] = None  # None 表示使用全局上界 _VECTOR_EVIDENCE_BONUS_CAP；显式值仅允许 ≤ 该上界
    scale: float = 1.0

    def effective_cap(self) -> float:
        if self.cap is None:
            return float(_VECTOR_EVIDENCE_BONUS_CAP)
        return min(float(_VECTOR_EVIDENCE_BONUS_CAP), float(self.cap))


DEFAULT_VECTOR_EVIDENCE_BONUS_CONFIG = VectorEvidenceBonusConfig()


def candidate_has_vector_origin(rec: Any) -> bool:
    """
    来源门控：是否经向量路进入合并池（依据路径/来源字段，不用 evidence 摘要非零反推）。
    无法解析或异常时返回 False（保守不加 bonus）。
    """
    try:
        if bool(getattr(rec, "from_vector", False)):
            return True
        if getattr(rec, "vector_rank", None) is not None:
            return True
        if getattr(rec, "vector_score_raw", None) is not None:
            return True
    except Exception:
        return False
    return False


def extract_vector_evidence_summary(vector_evidence: Any) -> Dict[str, Any]:
    """
    从 vector_path 输出的 vector_evidence 抽取少量稳定标量，供 KGAT-AX sidecar / 调试。
    Step7/8：同一套摘要由 compute_vector_evidence_bonus_from_summary 以极小权重可并入 candidate_pool_score；
    Step8 对 bonus 做来源门控与配置化。主分仍以 RRF / 路径融合为主。缺失时摘要与 bonus 均安全为 0。
    """
    out = {
        "vector_evidence_top_paper_count": 0,
        "vector_evidence_best_paper_score": 0.0,
        "vector_evidence_max_query_hit_count": 0,
        "vector_evidence_max_clause_hit_count": 0,
        "vector_evidence_query_type_coverage_count": 0,
        "vector_evidence_has_clause_signal": 0,
        "vector_evidence_has_method_hit_signal": 0,
        "vector_evidence_avg_top_paper_score": 0.0,
    }
    if not vector_evidence or not isinstance(vector_evidence, dict):
        return out
    summ = vector_evidence.get("summary") or {}
    top_papers = vector_evidence.get("top_papers") or []
    n_top = int(summ.get("top_evidence_count") or len(top_papers) or 0)
    best = float(summ.get("best_paper_score") or 0.0)
    max_qh = int(summ.get("max_query_hit_count") or 0)
    max_ch = int(summ.get("max_clause_hit_count") or 0)
    cov = summ.get("query_type_coverage") or []
    cov_n = len(cov) if isinstance(cov, list) else 0
    es = summ.get("evidence_sources_summary") or {}
    clause_sig = bool(es.get("clause_signal_present"))
    method_hit = False
    for p in top_papers:
        if not isinstance(p, dict):
            continue
        h = p.get("hit_query_types") or []
        if isinstance(h, (list, tuple)) and "method_focused_query" in h:
            method_hit = True
            break
    scores = [
        float(p.get("score_clause_aware") or p.get("score_hybrid") or 0.0)
        for p in top_papers
        if isinstance(p, dict)
    ]
    avg_top = sum(scores) / len(scores) if scores else 0.0
    out["vector_evidence_top_paper_count"] = n_top
    out["vector_evidence_best_paper_score"] = best
    out["vector_evidence_max_query_hit_count"] = max_qh
    out["vector_evidence_max_clause_hit_count"] = max_ch
    out["vector_evidence_query_type_coverage_count"] = cov_n
    out["vector_evidence_has_clause_signal"] = 1 if clause_sig else 0
    out["vector_evidence_has_method_hit_signal"] = 1 if method_hit else 0
    out["vector_evidence_avg_top_paper_score"] = float(avg_top)
    return out


def compute_vector_evidence_bonus_from_summary(
    summary: Optional[Dict[str, Any]],
    *,
    cap: Optional[float] = None,
    scale: float = 1.0,
) -> float:
    """
    从 vector_evidence_summary（与 extract_vector_evidence_summary 输出键一致）计算非负、有上界的弱 bonus。

    设计原则（Step7 低风险；Step8 仅参数化 cap/scale，不放大上界）：
    - 先对各分量截断/饱和，再加权求和，再乘 scale，最后 min(cap_eff)；不把 clause 当强信号。
    - 不替代 RRF；量级控制在主分尾部；enabled=0 或来源门控在调用侧处理。
    - 比 HyDE 等生成式改写更低风险：仅用已有结构化标量，无额外推理链。

    cap_eff = min(_VECTOR_EVIDENCE_BONUS_CAP, cap)（若传入 cap）；不得高于全局上界。
    """
    if not summary or not isinstance(summary, dict):
        return 0.0
    cap_eff = min(_VECTOR_EVIDENCE_BONUS_CAP, float(cap)) if cap is not None else float(_VECTOR_EVIDENCE_BONUS_CAP)
    try:
        sc = float(scale)
    except (TypeError, ValueError):
        sc = 1.0
    sc = max(0.0, sc)
    try:
        best = min(1.0, max(0.0, float(summary.get("vector_evidence_best_paper_score") or 0.0)))
        qhit = max(0, int(summary.get("vector_evidence_max_query_hit_count") or 0))
        cov_n = max(0, int(summary.get("vector_evidence_query_type_coverage_count") or 0))
        chit = max(0, int(summary.get("vector_evidence_max_clause_hit_count") or 0))
        clause_sig = 1 if int(summary.get("vector_evidence_has_clause_signal") or 0) else 0
        n_top = max(0, int(summary.get("vector_evidence_top_paper_count") or 0))
        avg = min(1.0, max(0.0, float(summary.get("vector_evidence_avg_top_paper_score") or 0.0)))
    except (TypeError, ValueError):
        return 0.0

    b = 0.0
    b += 0.018 * best
    b += 0.008 * min(qhit / 4.0, 1.0)
    b += 0.006 * min(cov_n / 4.0, 1.0)
    b += 0.004 * min(chit / 3.0, 1.0)
    b += 0.003 * float(clause_sig)
    b += 0.003 * min(n_top / 3.0, 1.0)
    b += 0.004 * avg
    return float(min(cap_eff, max(0.0, b * sc)))


def apply_vector_evidence_bonus_for_candidate(
    rec: Any,
    summary: Dict[str, Any],
    cfg: VectorEvidenceBonusConfig,
) -> float:
    """按配置计算最终并入 candidate_pool_score 的 bonus（含 enabled 与来源门控）。"""
    if not cfg.enabled:
        return 0.0
    raw = compute_vector_evidence_bonus_from_summary(
        summary,
        cap=cfg.effective_cap(),
        scale=cfg.scale,
    )
    if cfg.gate_vector_origin and not candidate_has_vector_origin(rec):
        return 0.0
    return raw


def infer_dominant_path(rec: Any) -> str:
    """根据 from_vector/from_label/from_collab 推断主导召回路径。"""
    path_count = getattr(rec, "path_count", 0) or 0
    if path_count >= 3:
        return "multi"
    if path_count == 2:
        if getattr(rec, "from_label", False) and getattr(rec, "from_vector", False):
            return "label+vector"
        if getattr(rec, "from_label", False) and getattr(rec, "from_collab", False):
            return "label+collab"
        if getattr(rec, "from_vector", False) and getattr(rec, "from_collab", False):
            return "vector+collab"
        return "multi"
    if getattr(rec, "from_label", False):
        return "label"
    if getattr(rec, "from_vector", False):
        return "vector"
    if getattr(rec, "from_collab", False):
        return "collab"
    return "unknown"


def calc_query_author_topic_similarity(
    query_vec: Any,
    top_works: List[Dict[str, Any]],
) -> Optional[float]:
    """
    Query 与作者代表作的主题相似度（标量）。
    若无可用的向量或 top_works 为空，返回 None。
    当前实现：若无预计算 work 向量则返回 0.5 占位；后续可接 SBERT(work_title) 与 query_vec 点积。
    """
    if not top_works or query_vec is None:
        return None
    # 占位：真实实现需对 top_works 的 title 编码后与 query_vec 求相似度并聚合
    return 0.5


def calc_domain_consistency(
    active_domains: Optional[Set[str]],
    top_works: List[Dict[str, Any]],
) -> Optional[float]:
    """
    作者代表作领域与当前 query 活跃领域的一致性 [0, 1]。
    若无 active_domains 或 top_works 为空，返回 None。
    """
    if not active_domains or not top_works:
        return None
    work_domains: Set[str] = set()
    for w in top_works:
        d = w.get("domain_ids") or w.get("domains")
        if d is not None:
            if isinstance(d, str):
                work_domains.update(x.strip() for x in d.replace("|", ",").split(",") if x.strip())
            elif isinstance(d, (list, tuple)):
                work_domains.update(str(x).strip() for x in d)
    if not work_domains:
        return 0.0
    inter = len(work_domains & active_domains)
    return inter / len(work_domains) if work_domains else 0.0


def calc_recent_activity_match(
    author_recent_stats: Dict[str, Any],
    top_works: List[Dict[str, Any]],
) -> Optional[float]:
    """
    近年活跃度与代表作的匹配度 [0, 1]。
    若有 recent_works_count / recent_citations，按归一化粗算；否则按 top_works 近年占比。
    """
    if not author_recent_stats and not top_works:
        return None
    recent_count = author_recent_stats.get("recent_works_count") or 0
    total = author_recent_stats.get("works_count") or 1
    if total and total > 0:
        return min(1.0, (recent_count or 0) / total)
    # 用 top_works 中近年论文占比
    if not top_works:
        return 0.0
    try:
        from datetime import datetime
        current_year = datetime.now().year
        recent = sum(1 for w in top_works if (w.get("year") or 0) >= current_year - 5)
        return recent / len(top_works) if top_works else 0.0
    except Exception:
        return 0.0


def calc_skill_coverage_ratio(
    label_term_count: int,
    label_core_term_count: int,
    label_support_term_count: int,
) -> Optional[float]:
    """
    标签路技能覆盖比：core 权重大，support 次之，总词数归一。
    返回 [0, 1] 或 None（无词时为 None）。
    """
    if label_term_count <= 0:
        return None
    # 简单加权：core=1.0, support=0.5
    weighted = label_core_term_count * 1.0 + label_support_term_count * 0.5
    max_weighted = label_term_count * 1.0
    return min(1.0, weighted / max_weighted) if max_weighted else None


def calc_paper_hit_strength(
    label_evidence: Any,
    vector_evidence: Any,
    top_works: List[Dict[str, Any]],
) -> Optional[float]:
    """
    论文命中强度：标签/向量证据与代表作的匹配程度 [0, 1]。
    若无可用的 evidence 或 top_works，返回 None。
    """
    if not top_works:
        return None
    has_label = label_evidence is not None and (isinstance(label_evidence, (list, dict)) and (len(label_evidence) if isinstance(label_evidence, (list, dict)) else 0) > 0)
    has_vector = vector_evidence is not None
    if has_label and has_vector:
        return 0.9
    if has_label or has_vector:
        return 0.5
    return 0.0


def calc_top_work_quality(top_works: List[Dict[str, Any]]) -> Optional[float]:
    """代表作质量分：基于引用或权重的简单聚合，归一到 [0, 1] 量级。"""
    if not top_works:
        return None
    scores = []
    for w in top_works:
        c = w.get("citation_count") or w.get("citations") or w.get("weight") or 0
        scores.append(float(c))
    if not scores:
        return None
    import math
    return min(1.0, math.log1p(sum(scores)) / 10.0)


# ---------------------------------------------------------------------------
# 批量加载（SQLite）
# ---------------------------------------------------------------------------

def batch_load_author_stats_from_sqlite(author_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """批量从 authors 表加载基础信息。"""
    if not author_ids:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        ph = ",".join("?" * len(author_ids))
        rows = conn.execute(
            f"SELECT author_id, name, h_index, works_count, cited_by_count FROM authors WHERE author_id IN ({ph})",
            author_ids,
        ).fetchall()
        for row in rows:
            aid = str(row["author_id"])
            out[aid] = {
                "name": row["name"],
                "h_index": row["h_index"],
                "works_count": row["works_count"],
                "cited_by_count": row["cited_by_count"],
            }
        conn.close()
    except Exception:
        pass
    return out


def batch_load_recent_author_stats(author_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    批量查近年 works 数、近年引用、机构层级分。
    若表无近年/机构字段则用 works+authorships 近似。
    """
    if not author_ids:
        return {}
    out: Dict[str, Dict[str, Any]] = {aid: {"recent_works_count": 0, "recent_citations": 0, "institution_level_score": 0.0} for aid in author_ids}
    try:
        conn = sqlite3.connect(DB_PATH)
        # 近年 5 年论文数与引用和
        try:
            conn.execute("SELECT 1 FROM works LIMIT 1")
        except Exception:
            conn.close()
            return out
        placeholders = ",".join("?" * len(author_ids))
        # 使用 authorships + works，year >= current_year - 5
        sql = f"""
        SELECT a.author_id, COUNT(DISTINCT a.work_id) AS recent_works, COALESCE(SUM(w.citation_count), 0) AS recent_cites
        FROM authorships a
        JOIN works w ON a.work_id = w.work_id
        WHERE a.author_id IN ({placeholders}) AND w.year >= CAST(strftime('%Y', 'now') AS INT) - 5
        GROUP BY a.author_id
        """
        cur = conn.execute(sql, author_ids)
        for row in cur.fetchall():
            aid = str(row[0])
            if aid in out:
                out[aid]["recent_works_count"] = row[1] or 0
                out[aid]["recent_citations"] = int(row[2] or 0)
        conn.close()
    except Exception:
        pass
    return out


def batch_load_top_works(
    author_ids: List[str],
    top_n_per_author: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """批量加载作者代表作（按引用或 pos 权重取 top）。"""
    if not author_ids:
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {aid: [] for aid in author_ids}
    try:
        conn = sqlite3.connect(DB_PATH)
        ph = ",".join("?" * len(author_ids))
        sql = f"""
        SELECT a.author_id, w.work_id, w.title, w.year, w.citation_count, w.domain_ids
        FROM authorships a
        JOIN works w ON a.work_id = w.work_id
        WHERE a.author_id IN ({ph})
        ORDER BY a.author_id, COALESCE(w.citation_count, -1) DESC
        """
        cur = conn.execute(sql, author_ids)
        by_author: Dict[str, List[Dict]] = {}
        for row in cur.fetchall():
            aid = str(row[0])
            if aid not in by_author:
                by_author[aid] = []
            if len(by_author[aid]) >= top_n_per_author:
                continue
            by_author[aid].append({
                "work_id": row[1],
                "title": row[2],
                "year": row[3],
                "citation_count": row[4],
                "citations": row[4],
                "domain_ids": row[5],
                "domains": row[5],
            })
        for aid, works in by_author.items():
            if aid in out:
                out[aid] = works
        conn.close()
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# 分桶截断与 KGAT-AX 侧车
# ---------------------------------------------------------------------------

def bucket_quota_truncate(
    records: List[Any],
    quotas: Dict[str, int],
    top_n: int,
) -> List[Any]:
    """
    按桶配额截断后再按 candidate_pool_score 取前 top_n。
    配额不足的桶取满为止，超出部分按总分补足到 top_n。
    """
    by_bucket: Dict[str, List[Any]] = {}
    for r in records:
        bt = (getattr(r, "bucket_type") or "Z").strip().upper()
        if bt not in by_bucket:
            by_bucket[bt] = []
        by_bucket[bt].append(r)
    result: List[Any] = []
    for bucket_type, cap in quotas.items():
        lst = by_bucket.get(bucket_type, [])
        lst_sorted = sorted(lst, key=lambda x: -(getattr(x, "candidate_pool_score") or 0.0))
        result.extend(lst_sorted[:cap])
    # 按总分再排序并取前 top_n
    result.sort(key=lambda x: -(getattr(x, "candidate_pool_score") or 0.0))
    return result[:top_n]


def _safe_rank(v: Any, default: int = 999) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_num(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _encode_bucket(bucket_type: Optional[str]) -> float:
    """桶类型编码为标量，供模型使用。"""
    m = {"A": 0.25, "B": 0.35, "C": 0.5, "D": 0.6, "E": 0.75, "F": 0.9, "Z": 1.0}
    if not bucket_type:
        return 0.5
    return m.get((bucket_type or "").strip().upper(), 0.5)


def build_kgatax_feature_row(rec: Any) -> Dict[str, Any]:
    """
    从一条 CandidateRecord 构建 KGAT-AX 四分支侧车的一行。
    返回结构：recall_features (dict/list), author_aux (dict/list), interaction_features (dict/list)，
    以及 author_id，便于推理时按 author_id 索引。
    """
    recall_features = {
        "from_vector": int(getattr(rec, "from_vector", False)),
        "from_label": int(getattr(rec, "from_label", False)),
        "from_collab": int(getattr(rec, "from_collab", False)),
        "path_count": getattr(rec, "path_count", 0) or 0,
        "bucket_type": (getattr(rec, "bucket_type") or "").strip() or "Z",
        "vector_rank": _safe_rank(getattr(rec, "vector_rank", None)),
        "label_rank": _safe_rank(getattr(rec, "label_rank", None)),
        "collab_rank": _safe_rank(getattr(rec, "collab_rank", None)),
        "candidate_pool_score": _safe_num(getattr(rec, "candidate_pool_score", None)),
        "vector_evidence_bonus": _safe_num(getattr(rec, "vector_evidence_bonus", None)),
    }
    author_aux = {
        "h_index": _safe_num(getattr(rec, "h_index", None)),
        "works_count": _safe_num(getattr(rec, "works_count", None)),
        "cited_by_count": _safe_num(getattr(rec, "cited_by_count", None)),
        "recent_works_count": _safe_num(getattr(rec, "recent_works_count", None)),
        "recent_citations": _safe_num(getattr(rec, "recent_citations", None)),
        "institution_level": _safe_num(getattr(rec, "institution_level", None)),
        "top_work_quality": _safe_num(getattr(rec, "top_work_quality", None)),
    }
    interaction_features = {
        "topic_similarity": _safe_num(getattr(rec, "topic_similarity", None)),
        "skill_coverage_ratio": _safe_num(getattr(rec, "skill_coverage_ratio", None)),
        "domain_consistency": _safe_num(getattr(rec, "domain_consistency", None)),
        "paper_hit_strength": _safe_num(getattr(rec, "paper_hit_strength", None)),
        "recent_activity_match": _safe_num(getattr(rec, "recent_activity_match", None)),
        "label_term_count": getattr(rec, "label_term_count", 0) or 0,
        "label_core_term_count": getattr(rec, "label_core_term_count", 0) or 0,
        "label_support_term_count": getattr(rec, "label_support_term_count", 0) or 0,
        "label_risky_term_count": getattr(rec, "label_risky_term_count", 0) or 0,
        "label_best_term_score": _safe_num(getattr(rec, "label_best_term_score", None)),
    }
    # Step6/7：摘要进 sidecar；Step7 弱 bonus 已并入 candidate_pool_score，此处同步导出便于审计
    vec_summ = extract_vector_evidence_summary(getattr(rec, "vector_evidence", None))
    return {
        "author_id": getattr(rec, "author_id", ""),
        "recall_features": recall_features,
        "author_aux": author_aux,
        "interaction_features": interaction_features,
        "vector_evidence_summary": vec_summ,
        # 训练侧车诊断字段（不参与模型张量维度，仅 JSON/日志）
        "candidate_trust_tier": str(getattr(rec, "candidate_trust_tier", "") or ""),
        "risk_flags": list(getattr(rec, "risk_flags", None) or []),
        "sampleability_flags": list(getattr(rec, "sampleability_flags", None) or []),
    }
