from typing import Any, Dict, List, Set


def run_stage2(
    recall,
    anchor_skills: Dict[str, Any],
    active_domain_set: Set[int],
    regex_str: str,
    query_vector,
    query_text: str = None,
) -> List[Dict[str, Any]]:
    """
    阶段 2：学术词扩展。
    搬运 LabelRecallPath._stage2_expand_academic_terms 的逻辑，保持签名与行为不变。
    """
    recall._compute_cluster_task_factors(query_vector)
    return recall._expand_semantic_map(
        [int(k) for k in anchor_skills.keys()],
        anchor_skills,
        domain_regex=regex_str,
        query_vector=query_vector,
        query_text=query_text,
        return_raw=True,
    )

