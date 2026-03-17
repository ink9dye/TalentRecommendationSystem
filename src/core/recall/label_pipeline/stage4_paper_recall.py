from typing import Any, Dict, List


def run_stage4(recall, vocab_ids: List[int], regex_str: str) -> List[Dict[str, Any]]:
    """
    阶段 4：论文召回。用带权学术词在 Neo4j 上沿 HAS_TOPIC 拉论文，按作者聚合。
    输入 vocab_ids 为 Stage3 选出的高质量学术词 ID（int），3% 熔断在 Cypher 内完成。
    返回: list of { 'aid': str, 'papers': [ { wid, hits, weight, title, year, domains }, ... ] }，供 Stage5 消费。
    """
    if not vocab_ids or not getattr(recall, "graph", None):
        return []
    v_ids = [int(x) for x in vocab_ids if x is not None]
    if not v_ids:
        return []
    total_w = float(getattr(recall, "total_work_count", 1e6) or 1e6)
    params = {"v_ids": v_ids, "total_w": total_w}
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
    WITH w, collect({{vid: v.id, idf: idf_weight}}) AS hit_info
    LIMIT 2000
    MATCH (w)<-[auth_r:AUTHORED]-(a:Author)
    RETURN a.id AS aid, collect({{wid: w.id, hits: hit_info, weight: auth_r.pos_weight,
                                 title: w.title, year: w.year, domains: w.domain_ids}}) AS papers
    """
    try:
        cursor = recall.graph.run(final_cypher, **params)
    except Exception:
        return []
    out = []
    for record in cursor:
        try:
            out.append({
                "aid": str(record["aid"]) if record.get("aid") is not None else "",
                "papers": list(record["papers"]) if record.get("papers") else [],
            })
        except Exception:
            continue
    return out

