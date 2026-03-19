import time
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
    term_meta: Optional[Dict[int, Dict[str, Any]]] = None,
    jd_text: Optional[str] = None,
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
      - term_meta: 可选，vid -> {term,parent_anchor,parent_primary,source_type,retrieval_role,...}，用于 Stage4 的 paper grounding 二次门控。
      - jd_text: 可选，整段 JD 文本；用于 grounding 计算时的辅助关键词命中。

    返回: list of { 'aid': str, 'papers': [ { wid, hits, weight, title, year, domains }, ... ] }，供 Stage5 消费。
    """
    di = getattr(recall, "debug_info", None)

    def _save_sub(ms: Dict[str, float]) -> None:
        if di is not None:
            di.stage4_sub_ms = ms

    if not vocab_ids or not getattr(recall, "graph", None):
        _save_sub({})
        return []
    v_ids = [int(x) for x in vocab_ids if x is not None]
    if not v_ids:
        _save_sub({})
        return []
    total_w = float(getattr(recall, "total_work_count", 1e6) or 1e6)
    term_scores = term_scores or {}
    # 统一用 int key 查找
    def _term_score(vid: int) -> float:
        return float(term_scores.get(vid) or term_scores.get(str(vid)) or 1.0)

    def _get_term_meta(vid: int) -> Dict[str, Any]:
        if not term_meta:
            return {}
        return term_meta.get(vid) or term_meta.get(str(vid)) or {}

    def _compute_grounding_score(
        vid: int,
        title: str,
        domains: str,
    ) -> Dict[str, float]:
        """
        Stage4 的 paper grounding：用 paper 的 title/domains 对齐岗位主轴与 term 证据。
        返回：
          - grounding: 0~1（主轴/词面落地强度）
          - off_topic_penalty: 额外偏题惩罚（用于抑制泛命中论文池）
        """
        meta = _get_term_meta(vid)

        term_text = str(meta.get("term") or "").lower()
        parent_anchor = str(meta.get("parent_anchor") or "").lower()
        parent_primary = str(meta.get("parent_primary") or "").lower()
        retrieval_role = str(meta.get("retrieval_role") or "").lower()

        t = (title or "").lower()
        d = (domains or "").lower()
        _jd = (jd_text or "").lower()

        # 1) 词面命中
        lexical_hit = 1.0 if term_text and term_text in t else 0.0

        # 2) 父锚 / 父主词命中（主轴证据）
        anchor_axis_hit = 0.0
        if parent_anchor and parent_anchor in t:
            anchor_axis_hit += 0.5
        if parent_primary and parent_primary in t:
            anchor_axis_hit += 0.5
        anchor_axis_hit = min(anchor_axis_hit, 1.0)

        # 3) 机器人/控制主轴关键词命中（title/domains）
        axis_keywords = [
            "robot",
            "robotic",
            "manipulator",
            "motion control",
            "robot control",
            "dynamics",
            "kinematics",
            "trajectory",
            "planning",
            "optimal control",
            "state estimation",
            "simulation",
        ]
        axis_hit_cnt = sum(1 for kw in axis_keywords if (kw in t) or (kw in d))
        jd_axis_match = min(axis_hit_cnt / 4.0, 1.0)

        # 4) 偏题惩罚（交通/调度/泛 AI 等）
        off_topic_penalty = 1.0
        off_keywords = [
            "charging station",
            "vehicle routing",
            "traffic",
            "transportation",
            "logistics",
            "supply chain",
            "crystallization",
            "v2x",
            "cybersecurity",
        ]
        if any(kw in t for kw in off_keywords):
            # Stage4 需要“压分”而不是“灭门”：收敛力道先降一档
            off_topic_penalty *= 0.75

        # RL 相关额外约束：如不是机器人/控制体系，则压
        if ("reinforcement learning" in term_text) or ("q-learning" in term_text):
            if not any(
                kw in t for kw in ["robot", "robotic", "control", "manipulator", "motion"]
            ):
                off_topic_penalty *= 0.75

        # route planning 相关额外约束：更像交通规划则压
        if "route planning" in term_text:
            if any(kw in t for kw in ["charging station", "traffic", "transportation", "vehicle"]):
                off_topic_penalty *= 0.75

        # robotic arm 相关额外约束：纯器件/结构而非控制也压一点
        if "robotic arm" in term_text:
            if not any(
                kw in t
                for kw in [
                    "control",
                    "trajectory",
                    "motion",
                    "dynamics",
                    "planning",
                ]
            ):
                off_topic_penalty *= 0.88

        grounding = (0.35 * lexical_hit + 0.25 * anchor_axis_hit + 0.40 * jd_axis_match)
        if retrieval_role == "paper_support":
            grounding *= 0.92

        grounding = max(0.0, min(1.0, float(grounding)))
        return {"grounding": grounding, "off_topic_penalty": float(off_topic_penalty)}

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
    WITH v, w, idf_weight, {domain_bonus_expr} AS domain_bonus, w.year AS year,
         coalesce(w.title, '') AS title, coalesce(w.domain_ids, '') AS domains
    RETURN v.id AS vid, w.id AS wid, idf_weight, domain_bonus, year, title, domains
    """
    params["melt_ratio"] = MELT_RATIO

    sub_ms: Dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        cursor = recall.graph.run(cypher_layer1, **params)
        rows = list(cursor)
    except Exception:
        sub_ms["cypher1"] = (time.perf_counter() - t0) * 1000.0
        sub_ms["total"] = sub_ms["cypher1"]
        _save_sub(sub_ms)
        return []

    t1 = time.perf_counter()
    sub_ms["cypher1"] = (t1 - t0) * 1000.0

    if not rows:
        sub_ms["total"] = (time.perf_counter() - t0) * 1000.0
        _save_sub(sub_ms)
        return []

    # ---------- Python：recency、role_weight、grounding/off_topic、term_contrib，per-term 限流，再按 paper 聚合 ----------
    term_retrieval_roles = term_retrieval_roles or {}
    by_term: Dict[int, List[tuple]] = defaultdict(list)

    # -------------------------
    # Stage4 诊断：过滤漏斗/死因/样本
    # 只做计数与打印，不改变筛选与聚合逻辑
    # -------------------------
    GROUNDING_MIN = 0.12

    def _new_funnel() -> Dict[str, int]:
        return {
            "cypher_raw": 0,
            "after_year_filter": 0,
            "after_basic_meta": 0,
            "after_grounding_gate": 0,
            "after_offtopic_penalty_sort": 0,
            "after_local_cap": 0,
            "final_unique": 0,
        }

    def _new_reject_reason() -> Dict[str, int]:
        return {
            "low_grounding": 0,
            "off_topic_penalty_too_low": 0,
            "duplicate_dropped": 0,
            "local_cap_cut": 0,
            "global_cap_cut": 0,
        }

    term_funnel_counts: Dict[int, Dict[str, int]] = defaultdict(_new_funnel)
    term_reject_reason_counts: Dict[int, Dict[str, int]] = defaultdict(_new_reject_reason)
    term_low_grounding_samples: Dict[int, List[Dict[str, Any]]] = defaultdict(list)  # 每 term 最多 3
    term_local_cap_cut_samples: Dict[int, List[Dict[str, Any]]] = defaultdict(list)  # 每 term 最多 3

    term_capped_unique_wids: Dict[int, set] = defaultdict(set)
    wid_to_paper_meta: Dict[str, Dict[str, str]] = {}  # wid -> {title, domains}
    for r in rows:
        vid = int(r["vid"])
        raw_wid = r["wid"]
        wid = str(raw_wid) if raw_wid is not None else None
        if wid is None:
            continue

        # -------- funnel: cypher_raw --------
        term_funnel_counts[vid]["cypher_raw"] += 1
        term_funnel_counts[vid]["after_year_filter"] += 1 if r.get("year") is not None else 0
        term_funnel_counts[vid]["after_basic_meta"] += 1

        idf_weight = float(r.get("idf_weight") or 0.0)
        domain_bonus = float(r.get("domain_bonus") or 1.0)
        year = r.get("year")
        title = str(r.get("title") or "")
        domains = str(r.get("domains") or "")
        wid_to_paper_meta[wid] = {"title": title, "domains": domains}

        recency = compute_paper_recency(year, None)
        term_final = _term_score(vid)
        role_weight = get_term_role_weight(term_retrieval_roles, vid)

        base_term_contrib = term_final * role_weight * idf_weight * domain_bonus * recency
        ground = _compute_grounding_score(vid, title, domains)
        grounding = float(ground["grounding"])
        off_topic_penalty = float(ground["off_topic_penalty"])

        # 主轴落地乘子 + 偏题惩罚：避免 reinforcement learning / route planning / robotic arm 等泛命中池偏移
        term_contrib = base_term_contrib * (0.45 + 0.55 * grounding) * off_topic_penalty

        # 低落地主轴论文直接不进池
        # 放松硬截断：避免把真正的主轴论文也直接砍光
        if grounding < GROUNDING_MIN:
            term_reject_reason_counts[vid]["low_grounding"] += 1
            if len(term_low_grounding_samples[vid]) < 3:
                term_low_grounding_samples[vid].append(
                    {
                        "wid": wid,
                        "title": title,
                        "domains": domains,
                        "reason": "low_grounding",
                        "grounding": grounding,
                        "penalty": off_topic_penalty,
                    }
                )
            continue

        term_funnel_counts[vid]["after_grounding_gate"] += 1
        by_term[vid].append((wid, term_contrib, idf_weight))

    # 每个 term 最多保留 TERM_MAX_PAPERS 篇（按 term_contrib 降序）
    limited: List[tuple] = []
    for vid, triples in by_term.items():
        triples.sort(key=lambda x: -x[1])
        role = (term_retrieval_roles.get(vid) or term_retrieval_roles.get(str(vid)) or "").strip().lower()
        local_cap = TERM_MAX_PAPERS
        if role == "paper_support":
                # support 词本来就更少；先把上限从 15 放到 30
                local_cap = min(TERM_MAX_PAPERS, 30)

        term_funnel_counts[vid]["after_offtopic_penalty_sort"] = term_funnel_counts[vid][
            "after_grounding_gate"
        ]

        cut_cnt = max(0, len(triples) - local_cap)
        term_reject_reason_counts[vid]["local_cap_cut"] += cut_cnt

        # local cap samples（用于打印，不影响 limited）
        if cut_cnt > 0 and len(term_local_cap_cut_samples[vid]) < 3:
            cut_triples = triples[local_cap : local_cap + 3]
            for (cut_wid, _, _) in cut_triples:
                meta = wid_to_paper_meta.get(cut_wid) or {"title": "", "domains": ""}
                if len(term_local_cap_cut_samples[vid]) < 3:
                    term_local_cap_cut_samples[vid].append(
                        {
                            "wid": cut_wid,
                            "title": meta.get("title") or "",
                            "domains": meta.get("domains") or "",
                            "reason": "local_cap_cut",
                        }
                    )

        kept_triples = triples[:local_cap]
        term_funnel_counts[vid]["after_local_cap"] = len(kept_triples)
        for (kept_wid, _, _) in kept_triples:
            term_capped_unique_wids[vid].add(kept_wid)

        for (wid, term_contrib, idf_weight) in triples[:local_cap]:
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
    selected_wids_set = set(sorted_wids)

    # -------- final_unique / global_cap_cut 统计（按 term 级唯一 wid）--------
    for vid, capped_wids in term_capped_unique_wids.items():
        final_unique = len(capped_wids.intersection(selected_wids_set))
        term_funnel_counts[vid]["final_unique"] = final_unique
        term_reject_reason_counts[vid]["global_cap_cut"] = len(capped_wids) - final_unique
    t2 = time.perf_counter()
    sub_ms["python_agg"] = (t2 - t1) * 1000.0
    # -------------------------
    # Stage4 诊断打印（只打印“折叠明显”的 term）
    # -------------------------
    # retain_ratio = final_unique / cypher_raw，按最糟的少量 term 输出
    retain_list: List[tuple] = []
    for vid, f in term_funnel_counts.items():
        cypher_raw = f.get("cypher_raw", 0) or 0
        if cypher_raw <= 0:
            continue
        final_unique = f.get("final_unique", 0) or 0
        retain_ratio = final_unique / float(cypher_raw) if cypher_raw > 0 else 0.0
        retain_list.append((retain_ratio, -cypher_raw, vid))

    retain_list.sort()
    focus_vids = [vid for (_, __, vid) in retain_list[:6] if term_funnel_counts[vid].get("cypher_raw", 0) >= 10]

    if focus_vids:
        for vid in focus_vids:
            meta = _get_term_meta(vid)
            term_name = str(meta.get("term") or meta.get("anchor") or meta.get("parent_primary") or vid)
            role = (term_retrieval_roles.get(vid) or term_retrieval_roles.get(str(vid)) or "").strip().lower()

            f = term_funnel_counts[vid]
            r = term_reject_reason_counts[vid]

            print(
                f"[Stage4 term funnel] term='{term_name}' role='{role}' "
                f"cypher_raw={f['cypher_raw']} "
                f"after_year_filter={f['after_year_filter']} "
                f"after_basic_meta={f['after_basic_meta']} "
                f"after_grounding_gate={f['after_grounding_gate']} "
                f"after_offtopic_penalty_sort={f['after_offtopic_penalty_sort']} "
                f"after_local_cap={f['after_local_cap']} "
                f"final_unique={f['final_unique']}"
            )

            print(
                f"[Stage4 reject reason summary] term='{term_name}' "
                f"low_grounding={r['low_grounding']} "
                f"off_topic_penalty_too_low={r['off_topic_penalty_too_low']} "
                f"duplicate_dropped={r['duplicate_dropped']} "
                f"local_cap_cut={r['local_cap_cut']} "
                f"global_cap_cut={r['global_cap_cut']}"
            )

            # reject samples：low_grounding -> local_cap_cut -> global_cap_cut 依次补齐到 3
            reject_samples: List[Dict[str, Any]] = []
            reject_samples.extend(term_low_grounding_samples.get(vid, [])[:3])
            if len(reject_samples) < 3:
                reject_samples.extend(term_local_cap_cut_samples.get(vid, [])[: 3 - len(reject_samples)])

            if len(reject_samples) < 3:
                capped_wids = term_capped_unique_wids.get(vid, set()) or set()
                diff_wids = list(capped_wids - selected_wids_set)
                diff_wids = diff_wids[: 3 - len(reject_samples)]
                for w in diff_wids:
                    meta2 = wid_to_paper_meta.get(w) or {"title": "", "domains": ""}
                    g = _compute_grounding_score(vid, meta2.get("title") or "", meta2.get("domains") or "")
                    reject_samples.append(
                        {
                            "wid": w,
                            "title": meta2.get("title") or "",
                            "domains": meta2.get("domains") or "",
                            "reason": "global_cap_cut",
                            "grounding": g.get("grounding", 0.0),
                            "penalty": g.get("off_topic_penalty", 1.0),
                        }
                    )

            if reject_samples:
                for s in reject_samples[:3]:
                    wid = s.get("wid")
                    title = s.get("title") or ""
                    reason = s.get("reason") or ""
                    grounding_val = s.get("grounding", None)
                    penalty_val = s.get("penalty", None)
                    if grounding_val is None or penalty_val is None:
                        g = _compute_grounding_score(vid, title, s.get("domains") or "")
                        grounding_val = g.get("grounding", 0.0)
                        penalty_val = g.get("off_topic_penalty", 1.0)
                    grounding = float(grounding_val or 0.0)
                    penalty = float(penalty_val or 1.0)
                    print(
                        f"[Stage4 reject samples] term='{term_name}' "
                        f"pid='{wid}' title={title[:80]!r} reason='{reason}' "
                        f"grounding={grounding:.3f} penalty={penalty:.3f}"
                    )

            # kept papers：最终入选（selected_wids_set）里，取对该 term 贡献的 top3（按 paper_score）
            capped_wids = term_capped_unique_wids.get(vid, set()) or set()
            kept_wids = list(capped_wids.intersection(selected_wids_set))
            kept_wids.sort(key=lambda w: -by_wid.get(w, (0.0, []))[0])
            kept_wids = kept_wids[:3]
            for i, w in enumerate(kept_wids, start=1):
                meta3 = wid_to_paper_meta.get(w) or {"title": "", "domains": ""}
                g = _compute_grounding_score(vid, meta3.get("title") or "", meta3.get("domains") or "")
                final_score = float(by_wid.get(w, (0.0, []))[0] or 0.0)
                print(
                    f"[Stage4 kept papers] term='{term_name}' rank={i} pid='{w}' "
                    f"grounding={float(g.get('grounding') or 0.0):.3f} "
                    f"penalty={float(g.get('off_topic_penalty') or 1.0):.3f} "
                    f"final_paper_score={final_score:.3f} title={meta3.get('title')[:80]!r}"
                )

            # rejected papers sample：再取一些未入选且可用的样本（最多 3 条）
            rejected_wids: List[str] = []
            rejected_wids.extend([s.get("wid") for s in reject_samples if s.get("wid")][:3])
            if len(rejected_wids) < 3:
                extra = list((capped_wids - selected_wids_set) or [])[: 3 - len(rejected_wids)]
                rejected_wids.extend(extra)
            rejected_wids = [w for w in rejected_wids if w is not None][:3]

            for w in rejected_wids:
                meta3 = wid_to_paper_meta.get(w) or {"title": "", "domains": ""}
                g = _compute_grounding_score(vid, meta3.get("title") or "", meta3.get("domains") or "")
                final_score = float(by_wid.get(w, (0.0, []))[0] or 0.0)
                # 是否 low_grounding：看 grounding 是否低于阈值
                reason = "low_grounding" if float(g.get("grounding") or 0.0) < GROUNDING_MIN else "pruned_after_terms"
                print(
                    f"[Stage4 rejected papers] term='{term_name}' pid='{w}' reason='{reason}' "
                    f"grounding={float(g.get('grounding') or 0.0):.3f} "
                    f"penalty={float(g.get('off_topic_penalty') or 1.0):.3f} "
                    f"final_paper_score={final_score:.3f} title={meta3.get('title')[:80]!r}"
                )

    if not sorted_wids:
        sub_ms["total"] = (time.perf_counter() - t0) * 1000.0
        _save_sub(sub_ms)
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
    t3 = time.perf_counter()
    try:
        cursor2 = recall.graph.run(cypher_layer2, **params2)
        author_rows = list(cursor2)
    except Exception:
        sub_ms["cypher2"] = (time.perf_counter() - t3) * 1000.0
        sub_ms["total"] = (time.perf_counter() - t0) * 1000.0
        _save_sub(sub_ms)
        return []

    t4 = time.perf_counter()
    sub_ms["cypher2"] = (t4 - t3) * 1000.0

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
    sub_ms["build_list"] = (time.perf_counter() - t4) * 1000.0
    sub_ms["total"] = (time.perf_counter() - t0) * 1000.0
    _save_sub(sub_ms)
    return out
