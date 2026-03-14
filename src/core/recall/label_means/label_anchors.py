import json
import math
import sqlite3
from typing import Dict, Any, Set, List, Tuple

import faiss
import numpy as np

from config import DB_PATH, VOCAB_P95_PAPER_COUNT
from src.utils.tools import extract_skills

# backbone_score 权重：in_jd_context / is_task_like 权重大于 job_freq，避免图热词再次主导
BACKBONE_W_JOB_FREQ = 0.15
BACKBONE_W_COV_J = 0.2
BACKBONE_W_SPAN = 0.1
BACKBONE_W_IN_JD = 0.35
BACKBONE_W_TASK_LIKE = 0.35
BACKBONE_W_SIM = 0.2  # query 向量相似度（可选）
BACKBONE_FLOOR_FOR_BAODI = 0.5  # 层0 保底词的最低得分，确保参与排序后有机会进 TopN


def clean_job_skills(skills_text: str) -> Set[str]:
    """
    复用全局 JD 技能抽取逻辑，将岗位技能字段统一走 src.utils.tools.extract_skills，
    返回小写去重集合，便于与图谱 term 对齐。
    """
    if not skills_text or not isinstance(skills_text, str):
        return set()
    return set(extract_skills(str(skills_text)))


def _in_jd_context(term_text: str, cleaned_terms: Set[str] | None) -> bool:
    """
    判断图谱中的短 term 是否与当前 JD 清洗得到的技能短语存在“语境重合”：
      - 先看精确命中；
      - 再看 term 是否作为子串出现在任一 cleaned term 中（例如 '路径规划' 命中
        '轨迹规划与全身控制算法开发'）。
    仅使用简单的子串规则，避免过早引入复杂相似度。
    """
    if not term_text or not cleaned_terms:
        return False
    t = term_text.lower().strip()
    if not t:
        return False
    if t in cleaned_terms:
        return True
    # 过短的 term（单字）不做子串匹配，避免噪声放大
    if len(t) <= 1:
        return False
    for jt in cleaned_terms:
        if not jt:
            continue
        if t in jt:
            return True
    return False


def _is_task_like(term_text: str) -> bool:
    """
    轻量判断 term 是否更像“任务骨干”而非泛 soft-skill：
      - 命中运动学/动力学/轨迹/规划/仿真/识别/估计/控制/优化/参数等关键词之一。
    """
    if not term_text:
        return False
    t = term_text.lower()
    task_kws = [
        "运动学",
        "动力学",
        "trajectory",
        "轨迹",
        "planning",
        "规划",
        "仿真",
        "simulation",
        "识别",
        "estimation",
        "估计",
        "control",
        "控制",
        "优化",
        "optimization",
        "参数",
    ]
    return any(k in t for k in task_kws)


def extract_anchor_skills(label, target_job_ids, query_vector=None, total_j=None) -> Dict[str, Any]:
    """
    复用 LabelRecallPath._extract_anchor_skills 的逻辑（移动到独立模块，便于后续拆分）。
    label 需提供：graph, total_job_count, vocab_to_idx, all_vocab_vectors, debug_info, 以及相关阈值常量。
    """
    total_j = float(total_j or 0)
    if total_j <= 0:
        total_j = label.total_job_count

    # 优先使用基于当前 JD 文本抽取的技能短语集合（由 stage1_domain_anchors 预先挂载），
    # 保证锚点过滤与本次查询语境强绑定；若不存在则回退到岗位 skills 字段的清洗结果。
    cleaned_terms = getattr(label, "_jd_cleaned_terms", None)
    if cleaned_terms:
        cleaned_terms = {str(t).lower() for t in cleaned_terms}
    else:
        cleaned_terms = set()
        try:
            cursor = label.graph.run(
                "MATCH (j:Job) WHERE j.id IN $j_ids RETURN j.skills AS skills",
                j_ids=target_job_ids[: label.ANCHOR_JOBS_TOP_K],
            )
            for row in cursor:
                if row.get("skills"):
                    cleaned_terms |= clean_job_skills(str(row["skills"]))
        except Exception:
            cleaned_terms = set()
        if not cleaned_terms:
            cleaned_terms = None

    # 打印 1：JD 清洗后的短语样本（若有），便于观察与本次查询语境相关的技能集合
    sample_cleaned = list(cleaned_terms)[:50] if cleaned_terms else []
    if getattr(label, "verbose", False):
        print(f"[Step2 Debug] JD 清洗后技能短语样本({len(sample_cleaned)}): {sample_cleaned}")

    cypher1 = """
    MATCH (j:Job) WHERE j.id IN $j_ids
    MATCH (j)-[:REQUIRE_SKILL]->(v:Vocabulary)
    WITH v.id AS vid, v.term AS term, count(DISTINCT j.id) AS job_freq
    RETURN vid, term, job_freq
    ORDER BY job_freq DESC
    """
    rows = []
    try:
        for r in label.graph.run(cypher1, j_ids=target_job_ids[: label.ANCHOR_JOBS_TOP_K]):
            if r.get("term") and len(str(r.get("term") or "")) > 1:
                rows.append(dict(r))
    except Exception:
        rows = []

    if not rows:
        stats = {
            "before_melt": 0,
            "after_melt": 0,
            "after_top30": 0,
            "melted_sample": [],
            "jd_cleaned_terms_sample": list(cleaned_terms)[:50] if cleaned_terms else [],
        }
        label.debug_info.anchor_melt_stats = stats
        label._last_anchor_melt_stats = stats
        return {}

    v_ids = list({int(r["vid"]) for r in rows})
    global_count = {}
    try:
        cypher2 = """
        UNWIND $v_ids AS vid
        MATCH (v:Vocabulary {id: vid})<-[:REQUIRE_SKILL]-(j:Job)
        RETURN vid, count(j) AS cnt
        """
        for r in label.graph.run(cypher2, v_ids=v_ids):
            global_count[int(r["vid"])] = int(r.get("cnt") or 0)
    except Exception:
        pass

    # 额外加载锚点候选的领域跨度（domain_span），用于对“长尾高纯度技术词”做统计型保活
    stats_span: dict[int, int] = {}
    try:
        if v_ids:
            ph = ",".join("?" * len(v_ids))
            rows_stats = label.stats_conn.execute(
                f"SELECT voc_id, domain_span FROM vocabulary_domain_stats WHERE voc_id IN ({ph})",
                v_ids,
            ).fetchall()
            for s in rows_stats:
                stats_span[int(s[0])] = int(s[1] or 0)
    except Exception:
        stats_span = {}

    melt_threshold = float(label.ANCHOR_MELT_COV_J)
    terms_before_melt = [r.get("term") or "" for r in rows]
    melted_terms: List[Tuple[str, float]] = []
    dropped_terms: List[Tuple[str, str, Any]] = []

    # ---------- 层1：噪声硬过滤 ----------
    # 只丢弃：过短、极端全行业泛词(cov_j >= melt_threshold)
    n_original_rows = len(rows)
    candidates_for_score: List[Tuple[Dict, float, int | None, bool, bool]] = []
    for r in rows:
        vid = int(r.get("vid"))
        g = global_count.get(vid, 0)
        cov_j = (g / total_j) if total_j else 0
        term_text = (r.get("term") or "").strip()
        if len(term_text) <= 1:
            dropped_terms.append((term_text, "length", len(term_text)))
            continue
        if cov_j >= melt_threshold:
            melted_terms.append((term_text, round(cov_j, 4)))
            dropped_terms.append((term_text, "cov_j", round(cov_j, 4)))
            continue
        job_freq = int(r.get("job_freq") or 0)
        span = stats_span.get(vid)
        in_jd = cleaned_terms is not None and _in_jd_context(term_text, cleaned_terms)
        task_like = _is_task_like(term_text)
        candidates_for_score.append((r, cov_j, span, in_jd, task_like))

    # ---------- 层0：保底标记 ----------
    # 满足 in_jd + is_task_like + 非超级泛词(已过层1) 的 term 记为保底，后续 backbone_score 给下限
    def _is_baodi(in_jd: bool, task_like: bool) -> bool:
        return bool(in_jd and task_like)

    # ---------- 层2：统一 backbone_score，一次排序 + 一次 TopN ----------
    q = np.asarray(query_vector, dtype=np.float32).flatten() if query_vector is not None else None
    scored: List[Tuple[float, Dict]] = []
    for r, cov_j, span, in_jd, task_like in candidates_for_score:
        job_freq = int(r.get("job_freq") or 0)
        span_val = span if span is not None else 10
        sim = 0.0
        if q is not None and q.size > 0:
            vid_str = str(r.get("vid"))
            idx = label.vocab_to_idx.get(vid_str)
            if idx is not None:
                try:
                    sim = float(np.dot(label.all_vocab_vectors[idx], q))
                except Exception:
                    pass
        score = (
            BACKBONE_W_JOB_FREQ * math.log(1 + job_freq)
            + BACKBONE_W_COV_J * (1.0 - min(1.0, cov_j))
            + BACKBONE_W_SPAN * (1.0 / (1 + span_val))
            + (BACKBONE_W_IN_JD if in_jd else 0.0)
            + (BACKBONE_W_TASK_LIKE if task_like else 0.0)
            + BACKBONE_W_SIM * max(0.0, sim)
        )
        if _is_baodi(in_jd, task_like):
            score = max(score, BACKBONE_FLOOR_FOR_BAODI)
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_n = int(getattr(label, "ANCHOR_FINAL_TOP_K", 20))
    rows = [r for _, r in scored[:top_n]]

    terms_after_melt = [r.get("term") or "" for r, *_ in candidates_for_score]
    stats = {
        "before_melt": n_original_rows,
        "after_layer1": len(candidates_for_score),
        "melted_sample": melted_terms[:25],
        "melt_threshold": melt_threshold,
        "terms_before_melt": terms_before_melt,
        "terms_after_melt": terms_after_melt,
        "cleaned_terms_sample": list(cleaned_terms)[:50] if cleaned_terms else [],
        "dropped_terms": dropped_terms[:50],
        "after_topN": len(rows),
        "terms_after_topN": [r.get("term") or "" for r in rows],
    }
    label.debug_info.anchor_melt_stats = stats
    label._last_anchor_melt_stats = stats
    if getattr(label, "verbose", False):
        n_cov = sum(1 for _, reason, _ in dropped_terms if reason == "cov_j")
        n_len = sum(1 for _, reason, _ in dropped_terms if reason == "length")
        print(
            f"[Step2 Debug] REQUIRE_SKILL 原始 rows={n_original_rows}，"
            f"层1 噪声过滤后候选={len(candidates_for_score)}；cov_j 砍掉 {n_cov} 个，length 砍掉 {n_len} 个"
        )
        for term, reason, value in dropped_terms[:30]:
            print(f"[Step2 Debug] REQUIRE_SKILL 丢弃: term={term} reason={reason} value={value}")
        print(f"[Step2 Debug] backbone_score 排序后 TopN={top_n} 样本: {[r.get('term') or '' for r in rows[:20]]}")

    if not rows:
        if getattr(label, "verbose", False):
            print("[Step2 Debug] 层2 排序+TopN 后无锚点可用。")
        return {}

    anchors = {str(r.get("vid")): {"term": r.get("term")} for r in rows}

    if getattr(label, "verbose", False):
        print(f"[Step2 Debug] 最终 industrial anchors 数量: {len(anchors)}")
        print(
            "[Step2 Debug] 最终 anchors 词样本:",
            [v["term"] for _, v in list(anchors.items())[:20]],
        )
    return anchors


def supplement_anchors_from_jd_vector(label, query_text, anchor_skills, total_j=None, top_k=None, active_domain_ids=None) -> None:
    """
    复用 LabelRecallPath._supplement_anchors_from_jd_vector 的逻辑（移动到独立模块）。
    label 需提供：vocab_index/stats_conn/graph/_query_encoder/_load_vocab_meta/_vocab_meta 以及阈值常量。
    """
    if not query_text or not getattr(label, "vocab_index", None):
        label.debug_info.supplement_anchors = []
        label.debug_info.supplement_anchors_report = []
        label._last_supplement_anchors = label.debug_info.supplement_anchors
        label._last_supplement_anchors_report = label.debug_info.supplement_anchors_report
        return

    total_j = float(total_j or 0) or label.total_job_count
    label._load_vocab_meta()
    encoder = label._query_encoder
    jd_snippet = (query_text or "").strip()[:500]
    if getattr(label, "verbose", False):
        print(f"[Bridge Debug] semantic_query_text 片段: {jd_snippet[:120]}")
    if not jd_snippet:
        label.debug_info.supplement_anchors = []
        label.debug_info.supplement_anchors_report = []
        label._last_supplement_anchors = label.debug_info.supplement_anchors
        label._last_supplement_anchors_report = label.debug_info.supplement_anchors_report
        return

    # 通过 QueryEncoder.encode 复用与主召回一致的文本预处理/桥接逻辑，再手动做 L2 归一化，
    # 避免 supplement 链路绕开 input_to_vector 的增强。
    v_jd, _ = encoder.encode(jd_snippet)
    if v_jd is None:
        label.debug_info.supplement_anchors = []
        label.debug_info.supplement_anchors_report = []
        label._last_supplement_anchors = label.debug_info.supplement_anchors
        label._last_supplement_anchors_report = label.debug_info.supplement_anchors_report
        return
    v_jd = np.asarray(v_jd, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(v_jd)
    k = min(int(top_k or label.SUPPLEMENT_TOP_K), 30)
    scores, labels = label.vocab_index.search(v_jd, k)
    added = []
    label.debug_info.supplement_anchors_report = []
    melt_threshold = float(label.ANCHOR_MELT_COV_J)
    v_ids_to_check = []
    candidates = []
    for score, tid in zip(scores[0], labels[0]):
        tid = int(tid)
        if tid <= 0:
            continue
        if str(tid) in anchor_skills:
            continue
        term, etype = label._vocab_meta.get(tid, ("", ""))
        etype = (etype or "").lower()
        if not term or len(term) < 2:
            continue
        if etype and etype not in label.SUPPLEMENT_ALLOW_ENTITY_TYPES:
            label.debug_info.supplement_anchors_report.append(
                {"tid": tid, "term": term, "etype": etype, "score": float(score), "reason": "etype_not_allowed"}
            )
            continue
        candidates.append((tid, term, etype, float(score)))
        v_ids_to_check.append(tid)

    if not v_ids_to_check:
        label.debug_info.supplement_anchors = []
        label.debug_info.supplement_anchors_report = []
        label._last_supplement_anchors = label.debug_info.supplement_anchors
        label._last_supplement_anchors_report = label.debug_info.supplement_anchors_report
        return

    global_count = {}
    stats_map = {}
    try:
        cypher = """
        UNWIND $v_ids AS vid
        MATCH (v:Vocabulary {id: vid})<-[:REQUIRE_SKILL]-(j:Job)
        RETURN vid, count(j) AS cnt
        """
        for r in label.graph.run(cypher, v_ids=v_ids_to_check):
            global_count[int(r["vid"])] = int(r.get("cnt") or 0)
    except Exception:
        global_count = {vid: 0 for vid in v_ids_to_check}

    try:
        ph = ",".join("?" * len(v_ids_to_check))
        rows = label.stats_conn.execute(
            f"SELECT voc_id, work_count, domain_dist FROM vocabulary_domain_stats WHERE voc_id IN ({ph})",
            v_ids_to_check,
        ).fetchall()
        for r in rows:
            stats_map[int(r[0])] = (int(r[1] or 0), r[2])
    except Exception:
        stats_map = {}

    active_domains = set(str(d) for d in (active_domain_ids or []))
    ranked = []
    for tid, term, etype, score in candidates:
        g = global_count.get(tid, 0)
        cov_j = (g / total_j) if total_j else 0.0
        reason = None
        if cov_j >= melt_threshold:
            reason = "cov_j_melt"
        row = stats_map.get(tid)
        degree_w = 0
        domain_ratio = 0.0
        if row:
            degree_w, dist_json = row
            try:
                dist = json.loads(dist_json) if isinstance(dist_json, str) else (dist_json or {})
            except (TypeError, ValueError):
                dist = {}
            expanded = label._expand_domain_dist(dist)
            degree_w_expanded = sum(expanded.values())
            if active_domains:
                target_degree_w = sum(expanded.get(str(d), 0) for d in active_domains)
            else:
                target_degree_w = degree_w_expanded
            domain_ratio = (target_degree_w / degree_w_expanded) if degree_w_expanded else 0.0
            if active_domains and domain_ratio < float(label.SUPPLEMENT_DOMAIN_RATIO_MIN):
                reason = "low_domain_ratio"
        else:
            reason = reason or "no_stats"

        report_row = {
            "tid": tid,
            "term": term,
            "etype": etype,
            "score": float(score),
            "cov_j": float(cov_j),
            "domain_ratio": float(domain_ratio),
            "reason": reason or "candidate",
        }
        label.debug_info.supplement_anchors_report.append(report_row)
        if reason is None:
            ranked.append(report_row)

    ranked.sort(key=lambda x: (x["domain_ratio"], x["score"]), reverse=True)
    for item in ranked[: label.SUPPLEMENT_MAX_ADD]:
        tid = item["tid"]
        term = item["term"]
        cov_j = item["cov_j"]
        anchor_skills[str(tid)] = {"term": term}
        added.append((tid, term, round(item["score"], 4), round(cov_j, 4), round(item["domain_ratio"], 4)))

    label.debug_info.supplement_anchors = added
    label._last_supplement_anchors = label.debug_info.supplement_anchors
    label._last_supplement_anchors_report = label.debug_info.supplement_anchors_report
    label._cached_v_jd = v_jd

