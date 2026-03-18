import re
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from src.core.recall.label_means import label_anchors
from src.core.recall.label_means.hierarchy_guard import parse_json_dist
from src.core.recall.label_means.label_debug import debug_print
from src.utils.domain_utils import DomainProcessor
from src.utils.tools import extract_skills
from src.core.recall.label_path import Stage1Result


def attach_anchor_contexts(
    anchor_skills: Dict[str, Any],
    query_text: str,
    window: int = 10,
) -> None:
    """
    为每个锚点增加 local_context（前后窗口字符）与 phrase_context（含该词的短语）。
    原地修改 anchor_skills 中每项的 local_context / phrase_context。
    """
    if not query_text or not anchor_skills:
        return
    # 按空格/标点切分为片段，便于取 phrase
    segments = re.split(r"[\s,，。；;!?、]+", query_text)
    text_lower = query_text.lower()
    for _vid, info in anchor_skills.items():
        term = (info.get("term") or "").strip()
        if not term:
            info["local_context"] = ""
            info["phrase_context"] = ""
            continue
        term_lower = term.lower()
        pos = text_lower.find(term_lower)
        if pos >= 0:
            start = max(0, pos - window)
            end = min(len(query_text), pos + len(term) + window)
            info["local_context"] = query_text[start:end].strip()
        else:
            info["local_context"] = query_text[: 80 + window].strip() if len(query_text) > 80 else query_text.strip()
        phrase = ""
        for seg in segments:
            if term in seg or term_lower in seg.lower():
                phrase = seg.strip()
                break
        info["phrase_context"] = phrase or info.get("local_context", "")[: 30]


def build_jd_hierarchy_profile(
    anchor_skills: Dict[str, Any],
    query_text: str,
    recall,
) -> Dict[str, Any]:
    """
    利用锚点对应的学术落点候选（SIMILAR_TO）与 vocabulary_topic_stats / vocabulary_domain_stats
    聚合成 JD 的四层领域画像，供 Stage2 层级守卫使用。
    返回 jd_profile: domain_weights, field_weights, subfield_weights, topic_weights,
    active_domains, active_fields, active_subfields, active_topics, main_*_id。
    """
    jd_profile: Dict[str, Any] = {
        "domain_weights": {},
        "field_weights": {},
        "subfield_weights": {},
        "topic_weights": {},
        "active_domains": [],
        "active_fields": [],
        "active_subfields": [],
        "active_topics": [],
        "main_domain_id": None,
        "main_field_id": None,
        "main_subfield_id": None,
        "main_topic_id": None,
    }
    if not getattr(recall, "graph", None) or not getattr(recall, "stats_conn", None):
        return jd_profile
    # 每个锚点 vid 查 SIMILAR_TO 取 top 学术词，用其 topic/domain 分布加权聚合
    SIMILAR_TOP_K = 5
    all_domain: Dict[str, float] = {}
    all_field: Dict[str, float] = {}
    all_subfield: Dict[str, float] = {}
    all_topic: Dict[str, float] = {}
    total_weight = 0.0
    for vid_str, info in (anchor_skills or {}).items():
        try:
            anchor_vid = int(vid_str)
        except (TypeError, ValueError):
            continue
        try:
            rows = recall.graph.run(
                """
                MATCH (v:Vocabulary {id: $vid})-[r:SIMILAR_TO]->(v2:Vocabulary)
                WHERE r.score >= 0.5 AND coalesce(v2.type, 'concept') IN ['concept', 'keyword']
                RETURN v2.id AS tid, r.score AS s
                ORDER BY r.score DESC
                LIMIT $k
                """,
                vid=anchor_vid,
                k=SIMILAR_TOP_K,
            ).data()
        except Exception:
            continue
        for r in rows:
            tid = r.get("tid")
            if tid is None:
                continue
            try:
                tid = int(tid)
            except (TypeError, ValueError):
                continue
            w = float(r.get("s") or 0.5)
            # 读 topic_stats
            row_t = recall.stats_conn.execute(
                "SELECT field_id, subfield_id, topic_id, field_dist, subfield_dist, topic_dist FROM vocabulary_topic_stats WHERE voc_id=?",
                (tid,),
            ).fetchone()
            row_d = recall.stats_conn.execute(
                "SELECT domain_dist FROM vocabulary_domain_stats WHERE voc_id=?",
                (tid,),
            ).fetchone()
            if row_t:
                fd = parse_json_dist(row_t[3])
                sd = parse_json_dist(row_t[4])
                td = parse_json_dist(row_t[5])
                for k, v in fd.items():
                    all_field[k] = all_field.get(k, 0.0) + v * w
                for k, v in sd.items():
                    all_subfield[k] = all_subfield.get(k, 0.0) + v * w
                for k, v in td.items():
                    all_topic[k] = all_topic.get(k, 0.0) + v * w
            if row_d and row_d[0]:
                dd = parse_json_dist(row_d[0])
                for k, v in dd.items():
                    all_domain[k] = all_domain.get(k, 0.0) + v * w
            total_weight += w
    if total_weight <= 0:
        return jd_profile
    def _norm(d: Dict[str, float]) -> Dict[str, float]:
        s = sum(d.values())
        return {k: v / s for k, v in d.items()} if s else d
    jd_profile["domain_weights"] = _norm(all_domain)
    jd_profile["field_weights"] = _norm(all_field)
    jd_profile["subfield_weights"] = _norm(all_subfield)
    jd_profile["topic_weights"] = _norm(all_topic)
    jd_profile["active_domains"] = list(jd_profile.get("active_domains") or []) or list(all_domain.keys())[:10]
    jd_profile["active_fields"] = list(all_field.keys())[:15]
    jd_profile["active_subfields"] = list(all_subfield.keys())[:20]
    jd_profile["active_topics"] = list(all_topic.keys())[:25]
    if all_domain:
        jd_profile["main_domain_id"] = max(all_domain, key=all_domain.get)
    if all_field:
        jd_profile["main_field_id"] = max(all_field, key=all_field.get)
    if all_subfield:
        jd_profile["main_subfield_id"] = max(all_subfield, key=all_subfield.get)
    if all_topic:
        jd_profile["main_topic_id"] = max(all_topic, key=all_topic.get)
    return jd_profile


def run_stage1(
    recall,
    query_vector,
    query_text: str | None = None,
    semantic_query_text: str | None = None,
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
    anchor_skills 中每项含 anchor_type，供 Stage2/3 做门槛与扩散权限控制（不切换候选源）。
    """
    job_ids: list[int] = []
    inferred_domains: list[int] = []
    dominance: float = 0.0
    job_previews: list[dict[str, Any]] = []
    anchor_debug: Dict[str, Any] = {}

    # 0) 预清洗当前 JD 的技能短语，供 Step2 稀疏锚点保活使用
    # 使用 extract_skills 保持与 JD 技能抽取链路一致，只在本次查询作用域内生效。
    text_for_skills = semantic_query_text or query_text
    jd_terms_cleaned = None
    if text_for_skills:
        try:
            jd_terms_cleaned = {s.lower() for s in extract_skills(text_for_skills) if s}
        except Exception:
            jd_terms_cleaned = None
    # 每次调用都覆盖上一轮缓存，避免跨查询串扰
    setattr(recall, "_jd_cleaned_terms", jd_terms_cleaned)
    setattr(recall, "_jd_raw_text", text_for_skills or "")

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
    if (semantic_query_text or query_text) and anchor_skills is not None:
        label_anchors.supplement_anchors_from_jd_vector(
            recall,
            semantic_query_text or query_text,
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

    # 方式1：扩展现有 anchor_skills，为每个锚点打上 anchor_type，供 Stage2/Stage3 按类型分策略
    for info in anchor_skills.values():
        info["anchor_type"] = label_anchors.classify_anchor_type(info.get("term") or "")

    # 调试：打印锚点及类型，便于排查
    if getattr(recall, "verbose", False):
        print("\n【Stage1 锚点类型】tid | term | anchor_type")
        for tid, info in anchor_skills.items():
            term = (info.get("term") or "")[:40]
            atype = info.get("anchor_type", "")
            print(f"  {tid} | {term} | {atype}")

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

    # 4) 锚点上下文化 + JD 四层领域画像（供 Stage2 层级守卫）
    query_for_ctx = (semantic_query_text or query_text) or ""
    attach_anchor_contexts(anchor_skills, query_for_ctx, window=12)
    # 条件化锚点表示：泛锚点带 JD 上下文，供 Stage2A 用 conditioned_vec 做落点打分
    encoder = getattr(recall, "_query_encoder", None)
    if encoder and query_for_ctx and anchor_skills:
        anchor_ctx_count = 0
        for _vid, info in anchor_skills.items():
            term = (info.get("term") or "").strip()
            if not term:
                continue
            try:
                ctx = label_anchors.build_conditioned_anchor_representation(
                    term, info, anchor_skills, query_for_ctx, encoder
                )
                if ctx.get("conditioned_vec") is not None:
                    info["conditioned_vec"] = ctx["conditioned_vec"]
                    info["anchor_vec"] = ctx.get("anchor_vec")
                    info["local_phrase_vec"] = ctx.get("local_phrase_vec")
                    info["co_anchor_vec"] = ctx.get("co_anchor_vec")
                    info["jd_vec"] = ctx.get("jd_vec")
                    info["_anchor_ctx"] = ctx
                if anchor_ctx_count < 10:
                    debug_print(2, f"[Anchor Context] anchor={term!r}", recall)
                    debug_print(2, f"  local_phrases={ctx.get('local_phrases', [])[:8]}", recall)
                    debug_print(2, f"  co_anchor_terms={ctx.get('co_anchor_terms', [])[:8]}", recall)
                    debug_print(2, (
                        "  weights="
                        f"{{anchor:{ctx.get('w_anchor', 0):.2f}, local:{ctx.get('w_local', 0):.2f}, "
                        f"co:{ctx.get('w_co', 0):.2f}, jd:{ctx.get('w_jd', 0):.2f}}}"
                    ), recall)
                    anchor_ctx_count += 1
            except Exception:
                pass
    jd_profile = build_jd_hierarchy_profile(anchor_skills, query_for_ctx, recall)
    jd_profile["active_domains"] = list(active_domain_set)

    debug_1: Dict[str, Any] = {
        "job_ids": job_ids,
        "job_previews": job_previews,
        "anchor_debug": anchor_debug,
        "dominance": dominance,
        "industrial_kws": industrial_kws,
        "anchor_skills": anchor_skills,
        "jd_profile": jd_profile,
    }

    # 5) 回填 Stage1Result（含 jd_profile 供 Stage2 使用）
    recall._last_stage1_result = Stage1Result(
        active_domains=set(active_domain_set),
        domain_regex=regex_str,
        anchor_skills=dict(anchor_skills or {}),
        job_ids=list(job_ids),
        job_previews=list(job_previews),
        dominance=float(dominance),
        anchor_debug=dict(anchor_debug or {}),
        jd_profile=jd_profile,
    )

    return active_domain_set, regex_str, anchor_skills, debug_1

