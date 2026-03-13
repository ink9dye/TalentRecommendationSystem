from typing import Tuple, Set, Dict, Any

import numpy as np

from src.core.recall.label_means import label_anchors
from src.utils.domain_utils import DomainProcessor
from src.core.recall.label_path import Stage1Result


def run_stage1(
    recall,
    query_vector,
    query_text: str | None = None,
    domain_id: str | None = None,
) -> Tuple[Set[int], str, Dict[str, Any], Dict[str, Any]]:
    """
    阶段 1：领域与锚点。

    逻辑与原 LabelRecallPath._stage1_domain_and_anchors 等价：
      1) 用 DomainDetector 或回退逻辑做岗位 / 领域探测；
      2) 用岗位 skills 抽取锚点（+ JD 语义补充）；
      3) 决定 active_domain_set 与 regex_str；
      4) 写入 recall._last_stage1_result 供后续调试；
      5) 返回 (active_domain_set, regex_str, anchor_skills, debug_1)。
    """
    job_ids: list[int] = []
    inferred_domains: list[int] = []
    dominance: float = 0.0
    job_previews: list[dict[str, Any]] = []
    anchor_debug: Dict[str, Any] = {}

    # 1) 领域与岗位：优先使用 DomainDetector，缺失时回退到旧实现
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

    # 2) 工业锚点：岗位技能提取 + JD 语义补充
    anchor_skills = label_anchors.extract_anchor_skills(
        recall,
        job_ids,
        query_vector=query_vector,
        total_j=recall.total_job_count,
    )
    if query_text and anchor_skills is not None:
        label_anchors.supplement_anchors_from_jd_vector(
            recall,
            query_text,
            anchor_skills,
            total_j=recall.total_job_count,
            top_k=recall.JD_VOCAB_TOP_K,
        )
    if not anchor_skills:
        # 无锚点时，后续阶段直接短路
        return set(), "", {}, {
            "job_ids": job_ids,
            "job_previews": job_previews,
            "anchor_debug": anchor_debug,
            "dominance": dominance,
        }

    industrial_kws = [v["term"] for v in anchor_skills.values()]

    # 3) 领域集合：如果用户指定 domain_id，则优先；否则使用推断领域 + 向量语义排序
    if domain_id and str(domain_id) != "0":
        active_domain_set: Set[int] = DomainProcessor.to_set(domain_id)
        if len(active_domain_set) > recall.ACTIVE_DOMAINS_TOP_K:
            active_domain_set = set(sorted(active_domain_set)[: recall.ACTIVE_DOMAINS_TOP_K])
    else:
        candidate_5 = inferred_domains
        if recall.domain_vectors and len(candidate_5) > recall.ACTIVE_DOMAINS_TOP_K:
            q = np.asarray(query_vector, dtype=np.float32).flatten()
            if q.size > 0:
                scores: list[tuple[int, float]] = []
                for d in candidate_5:
                    dv = recall.domain_vectors.get(str(d))
                    if dv is not None and dv.size == q.size:
                        sc = float(np.dot(q, dv))
                        scores.append((d, sc))
                scores.sort(key=lambda x: x[1], reverse=True)
                active_domain_set = set(d for d, _ in scores[: recall.ACTIVE_DOMAINS_TOP_K])
            else:
                active_domain_set = set(sorted(candidate_5)[: recall.ACTIVE_DOMAINS_TOP_K])
        else:
            active_domain_set = set(sorted(candidate_5)[: recall.ACTIVE_DOMAINS_TOP_K])

    regex_str = DomainProcessor.build_neo4j_regex(active_domain_set)

    debug_1: Dict[str, Any] = {
        "job_ids": job_ids,
        "job_previews": job_previews,
        "anchor_debug": anchor_debug,
        "dominance": dominance,
        "industrial_kws": industrial_kws,
        "anchor_skills": anchor_skills,
    }

    # 4) 回填 Stage1Result（供调试使用）
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

