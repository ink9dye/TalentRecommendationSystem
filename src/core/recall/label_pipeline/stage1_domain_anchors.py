from typing import Tuple, Set, Dict, Any

from src.core.recall.label_means.base import LabelContext  # 保留潜在使用
from src.core.recall import label_anchors
from src.utils.domain_utils import DomainProcessor


def run_stage1(recall, query_vector, query_text=None, domain_id=None) -> Tuple[Set[int], str, Dict[str, Any], Dict[str, Any]]:
    """
    阶段 1：领域与锚点。
    直接搬运 LabelRecallPath._stage1_domain_and_anchors 的逻辑，
    但作为模块级函数，接收 LabelRecallPath 实例 recall。
    """
    job_ids = []
    inferred_domains = []
    dominance = 0.0
    job_previews = []
    anchor_debug = {}

    if getattr(recall, "domain_detector", None) is not None:
        active_set, _, debug = recall.domain_detector.detect(
            query_vector,
            query_text=query_text,
            user_domain=None,
        )
        sd = debug.get("stage1_debug", {}) if isinstance(debug, dict) else {}
        job_ids = sd.get("job_ids", []) or []
        inferred_domains = sd.get("candidate_domains", []) or list(active_set or [])
        dominance = sd.get("dominance", 0.0) or 0.0
        job_previews = sd.get("job_previews", []) or []
        anchor_debug = sd.get("anchor_debug", {}) or {}
    else:
        job_ids, inferred_domains, dominance = recall._detect_domain_context(query_vector)
        job_previews = recall._get_job_previews(job_ids)
        anchor_debug = recall._get_anchor_debug_stats(job_ids[:20], recall.total_job_count) if job_ids else {}

    anchor_skills = label_anchors.extract_anchor_skills(
        recall, job_ids, query_vector=query_vector, total_j=recall.total_job_count
    )
    if query_text and anchor_skills is not None:
        label_anchors.supplement_anchors_from_jd_vector(
            recall, query_text, anchor_skills, total_j=recall.total_job_count, top_k=recall.JD_VOCAB_TOP_K
        )
    if not anchor_skills:
        return set(), "", {}, {
            "job_ids": job_ids,
            "job_previews": job_previews,
            "anchor_debug": anchor_debug,
            "dominance": dominance,
        }

    industrial_kws = [v["term"] for v in anchor_skills.values()]
    if domain_id and str(domain_id) != "0":
        active_domain_set = DomainProcessor.to_set(domain_id)
        if len(active_domain_set) > recall.ACTIVE_DOMAINS_TOP_K:
            active_domain_set = set(list(sorted(active_domain_set))[: recall.ACTIVE_DOMAINS_TOP_K])
    else:
        candidate_5 = inferred_domains
        if recall.domain_vectors and len(candidate_5) > recall.ACTIVE_DOMAINS_TOP_K:
            import numpy as np

            q = np.asarray(query_vector, dtype=np.float32).flatten()
            if q.size > 0:
                scores = []
                for d in candidate_5:
                    dv = recall.domain_vectors.get(str(d))
                    if dv is not None and dv.size == q.size:
                        sc = float(np.dot(q, dv))
                        scores.append((d, sc))
                scores.sort(key=lambda x: x[1], reverse=True)
                active_domain_set = set(x[0] for x in scores[: recall.ACTIVE_DOMAINS_TOP_K])
            else:
                active_domain_set = set(list(sorted(candidate_5))[: recall.ACTIVE_DOMAINS_TOP_K])
        else:
            active_domain_set = set(list(sorted(candidate_5))[: recall.ACTIVE_DOMAINS_TOP_K])

    regex_str = DomainProcessor.build_neo4j_regex(active_domain_set)
    debug_1 = {
        "job_ids": job_ids,
        "job_previews": job_previews,
        "anchor_debug": anchor_debug,
        "dominance": dominance,
        "industrial_kws": industrial_kws,
        "anchor_skills": anchor_skills,
    }

    # 保持对 recall._last_stage1_result 的写入，兼容现有调试逻辑
    from src.core.recall.label_path import Stage1Result  # 避免循环导入时延迟使用

    recall._last_stage1_result = Stage1Result(
        active_domains=set(active_domain_set),
        domain_regex=regex_str,
        anchor_skills=dict(anchor_skills or {}),
        job_ids=list(job_ids),
        job_previews=list(job_previews),
        dominance=float(dominance),
        anchor_debug=dict(anchor_debug or {}),
    )

    return active_domain_set, regex_str, anchor_skills, debug_1

