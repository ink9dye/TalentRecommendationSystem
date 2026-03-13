from typing import Any, Dict, List


def run_stage4(recall, vocab_ids: List[int], regex_str: str) -> List[Dict[str, Any]]:
    """
    阶段 4：图检索。
    搬运 LabelRecallPath._stage4_graph_search 的 Cypher 逻辑。
    """
    params = {"v_ids": vocab_ids, "total_w": recall.total_work_count}
    domain_clause = ""
    if regex_str:
        domain_clause = "AND w.domain_ids =~ $regex"
        params["regex"] = regex_str
    final_cypher = f"""
    MATCH (v:Vocabulary) WHERE v.id IN $v_ids
    WITH v, COUNT {{ (v)<-[:HAS_TOPIC]-() }} AS degree_w
    WHERE (degree_w * 1.0 / $total_w) < 0.03 
    WITH v, log10($total_w / (degree_w + 1)) AS idf_weight
    MATCH (v)<-[:HAS_TOPIC]-(w:Work) 
    WHERE 1=1 {domain_clause} 
    WITH w, collect({{vid: v.id, idf: idf_weight}}) AS hit_info LIMIT 2000
    MATCH (w)<-[auth_r:AUTHORED]-(a:Author)
    RETURN a.id AS aid, collect({{wid: w.id, hits: hit_info, weight: auth_r.pos_weight, 
                                 title: w.title, year: w.year, domains: w.domain_ids}}) AS papers
    """
    cursor = recall.graph.run(final_cypher, **params)
    return list(cursor)

