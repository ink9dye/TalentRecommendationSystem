import json
import sqlite3
from typing import Dict, Any, Set

import faiss
import numpy as np

from config import DB_PATH, VOCAB_P95_PAPER_COUNT
from src.utils.tools import extract_skills


def clean_job_skills(skills_text: str) -> Set[str]:
    """
    复用全局 JD 技能抽取逻辑，将岗位技能字段统一走 src.utils.tools.extract_skills，
    返回小写去重集合，便于与图谱 term 对齐。
    """
    if not skills_text or not isinstance(skills_text, str):
        return set()
    return set(extract_skills(str(skills_text)))


def extract_anchor_skills(label, target_job_ids, query_vector=None, total_j=None) -> Dict[str, Any]:
    """
    复用 LabelRecallPath._extract_anchor_skills 的逻辑（移动到独立模块，便于后续拆分）。
    label 需提供：graph, total_job_count, vocab_to_idx, all_vocab_vectors, debug_info, 以及相关阈值常量。
    """
    total_j = float(total_j or 0)
    if total_j <= 0:
        total_j = label.total_job_count

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
        pass
    if not cleaned_terms:
        cleaned_terms = None

    # 打印 1：JD 清洗后的短语样本，便于观察岗位技能清洗效果
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
    rows_with_cov = []
    melted_terms = []
    dropped_terms = []
    # 结合 JD 语境做轻量保活：在清洗后的 JD 短语中出现、且领域跨度较小的“长尾技术词”不应轻易被熔断。
    # 这里通过 cleaned_terms + 领域跨度(domain_span) 共同决定是否放宽 job_freq 门槛，而不依赖具体词面。

    for r in rows:
        vid = int(r.get("vid"))
        g = global_count.get(vid, 0)
        cov_j = (g / total_j) if total_j else 0
        if cov_j >= melt_threshold:
            melted_terms.append((r.get("term") or "", round(cov_j, 4)))
            dropped_terms.append((r.get("term") or "", "cov_j", round(cov_j, 4)))
            continue

        job_freq = int(r.get("job_freq") or 0)
        # 对“长尾但领域跨度较小、且在当前 JD 语境中出现的技术词”做统计型保活：
        # 仅当 term 确实出现在 JD cleaned_terms 中，且 domain_span 不大时，才放宽 job_freq 门槛。
        span = stats_span.get(vid)
        term_text = (r.get("term") or "").strip()
        lowered = term_text.lower()
        in_jd_context = cleaned_terms is not None and lowered in cleaned_terms

        if job_freq < int(label.ANCHOR_MIN_JOB_FREQ):
            if span is not None and span <= 3 and in_jd_context:
                rows_with_cov.append((r, cov_j))
            else:
                dropped_terms.append((term_text, "job_freq", job_freq))
            continue

        rows_with_cov.append((r, cov_j))

    terms_after_melt = [x[0].get("term") or "" for x in rows_with_cov]
    stats = {
        "before_melt": len(rows),
        "after_melt": len(rows_with_cov),
        "melted_sample": melted_terms[:25],
        "melt_threshold": melt_threshold,
        "terms_before_melt": terms_before_melt,
        "terms_after_melt": terms_after_melt,
        "cleaned_terms_sample": list(cleaned_terms)[:50] if cleaned_terms else [],
        "dropped_terms": dropped_terms[:50],
    }
    label.debug_info.anchor_melt_stats = stats
    label._last_anchor_melt_stats = stats
    if getattr(label, "verbose", False):
        n_cov = sum(1 for _, reason, _ in dropped_terms if reason == "cov_j")
        n_freq = sum(1 for _, reason, _ in dropped_terms if reason == "job_freq")
        print(
            f"[Step2 Debug] REQUIRE_SKILL 原始 rows={len(rows)}，"
            f"熔断/共识过滤后保留={len(rows_with_cov)}；cov_j 砍掉 {n_cov} 个，job_freq 砍掉 {n_freq} 个"
        )
        # 打印 2：每个 term 在熔断/频次过滤中的去留原因样本
        for term, reason, value in dropped_terms[:30]:
            print(f"[Step2 Debug] REQUIRE_SKILL 丢弃: term={term} reason={reason} value={value}")
        for term in terms_after_melt[:30]:
            print(f"[Step2 Debug] REQUIRE_SKILL 熔断后保留样本: term={term}")

    rows_with_cov.sort(key=lambda x: (x[0].get("job_freq") or 0), reverse=True)
    rows = [x[0] for x in rows_with_cov[: label.ANCHOR_FREQ_TOP_K]]
    label.debug_info.anchor_melt_stats["after_top30"] = len(rows)
    label.debug_info.anchor_melt_stats["terms_after_top30"] = [r.get("term") or "" for r in rows]

    if cleaned_terms is not None:
        rows = [r for r in rows if (r.get("term") or "").lower() in cleaned_terms]
    label.debug_info.anchor_melt_stats["terms_after_cleaned"] = [r.get("term") or "" for r in rows]

    if not rows:
        if getattr(label, "verbose", False):
            print("[Step2 Debug] 熔断+Top30+清洗后无锚点可用。")
        return {}

    if query_vector is not None:
        q = np.asarray(query_vector, dtype=np.float32).flatten()
        scored = []
        for r in rows:
            vid = str(r.get("vid"))
            idx = label.vocab_to_idx.get(vid)
            sim = -1.0
            if idx is not None and q.size > 0:
                try:
                    sim = float(np.dot(label.all_vocab_vectors[idx], q))
                except Exception:
                    sim = -1.0
            scored.append((sim, int(r.get("job_freq") or 0), r))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        sim_min = float(label.ANCHOR_SIM_MIN)
        scored_keep = [x for x in scored if (x[0] is not None and float(x[0]) >= sim_min)]
        label.debug_info.anchor_melt_stats["sim_min"] = sim_min
        label.debug_info.anchor_melt_stats["sim_kept"] = len(scored_keep)
        label.debug_info.anchor_melt_stats["sim_dropped"] = max(0, len(scored) - len(scored_keep))
        rows = [x[2] for x in scored_keep[: label.ANCHOR_FINAL_TOP_K]]
    else:
        rows = rows[: label.ANCHOR_FINAL_TOP_K]
    label.debug_info.anchor_melt_stats["terms_after_sim"] = [r.get("term") or "" for r in rows]

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

