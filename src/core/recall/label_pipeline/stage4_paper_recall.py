from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.utils.time_features import compute_paper_recency

# 层级守卫：单 term 最多贡献论文数，避免泛词占满 paper 池
TERM_MAX_PAPERS = 50
# 单 term 对某作者总贡献占比上限（在 Stage5 / paper_scoring 中实现）
TERM_MAX_AUTHOR_SHARE = 0.25

# 词侧熔断：degree_w/total_w 超过此比例的泛词在 Cypher 内过滤
MELT_RATIO = 0.05
# 领域软奖励：论文 domain_ids 匹配目标领域时的乘数（小幅加成，不主导）
DOMAIN_BONUS_MATCH = 1.2
DOMAIN_BONUS_NO_MATCH = 1.0

# 全局 paper 池上限
GLOBAL_PAPER_LIMIT = 2000


def get_term_role_weight(term_retrieval_roles: Optional[Dict[int, str]], vid: int) -> float:
    """按 retrieval_role 给权重：paper_primary=1.0，paper_support=0.7，blocked/其他=0.4。不看领域词。"""
    if not term_retrieval_roles:
        return 1.0
    role = (term_retrieval_roles.get(vid) or term_retrieval_roles.get(str(vid)) or "").strip().lower()
    if role == "paper_primary":
        return 1.0
    if role == "paper_support":
        return 0.7
    return 0.4


def run_stage4(
    recall,
    vocab_ids: List[int],
    regex_str: str,
    term_scores: Optional[Dict[int, float]] = None,
    term_retrieval_roles: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """
    阶段 4：二层论文召回。用带权学术词沿 HAS_TOPIC 拉论文，按 paper_score 全局排序后截断。

    - 取消论文层 domain 硬过滤，改为软奖励（匹配则乘 DOMAIN_BONUS_MATCH，否则 1.0）。
    - paper_score = Σ (term_final_score × idf_weight × domain_bonus × recency_factor)，含 Stage3 词质量与 per-term 限流。
    - 词侧熔断放宽为 MELT_RATIO（默认 5%），避免合理词被误杀。

    输入:
      - vocab_ids: 参与检索的词汇 ID（即 final_term_ids_for_paper）。
      - regex_str: 领域正则，用于计算 domain_bonus；为空则不奖励。
      - term_scores: vid -> Stage3 的 final_score；若为 None 则按 1.0 处理。

    返回: list of { 'aid': str, 'papers': [ { wid, hits, weight, title, year, domains }, ... ] }，供 Stage5 消费。
    """
    if not vocab_ids or not getattr(recall, "graph", None):
        return []
    v_ids = [int(x) for x in vocab_ids if x is not None]
    if not v_ids:
        return []
    total_w = float(getattr(recall, "total_work_count", 1e6) or 1e6)
    term_scores = term_scores or {}
    # 统一用 int key 查找
    def _term_score(vid: int) -> float:
        return float(term_scores.get(vid) or term_scores.get(str(vid)) or 1.0)

    # ---------- 第一层：按 term 拉 (vid, wid, idf_weight, domain_bonus, year)，无论文层硬过滤 ----------
    params: Dict[str, Any] = {"v_ids": v_ids, "total_w": total_w}
    if regex_str and regex_str.strip():
        params["regex"] = regex_str.strip()
        domain_bonus_expr = (
            "CASE WHEN $regex IS NOT NULL AND size($regex) > 0 AND w.domain_ids =~ $regex "
            f"THEN {DOMAIN_BONUS_MATCH} ELSE {DOMAIN_BONUS_NO_MATCH} END"
        )
    else:
        domain_bonus_expr = str(DOMAIN_BONUS_NO_MATCH)

    cypher_layer1 = f"""
    MATCH (v:Vocabulary) WHERE v.id IN $v_ids
    WITH v, count {{ (v)<-[:HAS_TOPIC]-() }} AS degree_w
    WHERE (degree_w * 1.0 / $total_w) < $melt_ratio
    WITH v, log10($total_w / (degree_w + 1)) AS idf_weight
    MATCH (v)<-[:HAS_TOPIC]-(w:Work)
    WITH v, w, idf_weight, {domain_bonus_expr} AS domain_bonus, w.year AS year
    RETURN v.id AS vid, w.id AS wid, idf_weight, domain_bonus, year
    """
    params["melt_ratio"] = MELT_RATIO

    try:
        cursor = recall.graph.run(cypher_layer1, **params)
        rows = list(cursor)
    except Exception:
        return []

    if not rows:
        return []

    # ---------- Python：recency、role_weight、term_contrib，per-term 限流，再按 paper 聚合 ----------
    # wid 与图库一致：保持字符串（如 'W2756749562'），全链路不再转 int
    term_retrieval_roles = term_retrieval_roles or {}
    by_term: Dict[int, List[tuple]] = defaultdict(list)
    for r in rows:
        vid = int(r["vid"])
        raw_wid = r["wid"]
        wid = str(raw_wid) if raw_wid is not None else None
        if wid is None:
            continue
        idf_weight = float(r.get("idf_weight") or 0.0)
        domain_bonus = float(r.get("domain_bonus") or 1.0)
        year = r.get("year")
        recency = compute_paper_recency(year, None)
        term_final = _term_score(vid)
        role_weight = get_term_role_weight(term_retrieval_roles, vid)
        term_contrib = term_final * role_weight * idf_weight * domain_bonus * recency
        by_term[vid].append((wid, term_contrib, idf_weight))

    # 每个 term 最多保留 TERM_MAX_PAPERS 篇（按 term_contrib 降序）
    limited: List[tuple] = []
    for vid, triples in by_term.items():
        triples.sort(key=lambda x: -x[1])
        for (wid, term_contrib, idf_weight) in triples[:TERM_MAX_PAPERS]:
            limited.append((wid, vid, term_contrib, idf_weight))

    # 按 wid 聚合：paper_score = Σ term_contrib，hits = [ {vid, idf}, ... ]
    by_wid: Dict[str, tuple] = {}
    for (wid, vid, term_contrib, idf_weight) in limited:
        if wid not in by_wid:
            by_wid[wid] = (0.0, [])
        score, hits = by_wid[wid]
        by_wid[wid] = (score + term_contrib, hits + [{"vid": vid, "idf": idf_weight}])

    # 全局按 paper_score 排序，取前 GLOBAL_PAPER_LIMIT
    sorted_wids = sorted(
        by_wid.keys(),
        key=lambda w: -by_wid[w][0],
    )[:GLOBAL_PAPER_LIMIT]
    if not sorted_wids:
        return []

    # ---------- 第二层：按 wid 查作者与论文元数据，按 aid 聚合为 author_papers_list ----------
    params2 = {"wids": sorted_wids}
    cypher_layer2 = """
    MATCH (w:Work) WHERE w.id IN $wids
    MATCH (w)<-[r:AUTHORED]-(a:Author)
    WITH a.id AS aid, w.id AS wid, r.pos_weight AS weight, w.title AS title, w.year AS year, w.domain_ids AS domains
    WITH aid, collect({wid: wid, weight: weight, title: title, year: year, domains: domains}) AS papers
    RETURN aid, papers
    """
    try:
        cursor2 = recall.graph.run(cypher_layer2, **params2)
        author_rows = list(cursor2)
    except Exception:
        return []

    # 为每篇 paper 挂上 Stage4 算好的 hits 与 score（供 Stage5 / debug 使用）
    wid_to_hits_and_score = {wid: (hits, score) for wid, (score, hits) in by_wid.items()}

    out: List[Dict[str, Any]] = []
    for rec in author_rows:
        aid = rec.get("aid")
        papers_raw = rec.get("papers") or []
        papers = []
        for p in papers_raw:
            wid = p.get("wid")
            if wid is None:
                continue
            hits, score = wid_to_hits_and_score.get(wid, ([], 0.0))
            papers.append({
                "wid": wid,
                "hits": hits,
                "weight": p.get("weight"),
                "title": p.get("title"),
                "year": p.get("year"),
                "domains": p.get("domains"),
                "score": score,
            })
        if aid is not None and papers:
            out.append({
                "aid": str(aid),
                "papers": papers,
            })
    return out
