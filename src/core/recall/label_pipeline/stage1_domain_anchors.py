import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# SQLite 变量占位上限保守值，避免 IN (...) 过长
_SQLITE_IN_CHUNK = 900

from src.core.recall.label_means import label_anchors
from src.core.recall.label_means.hierarchy_guard import parse_json_dist
from src.core.recall.label_means.label_debug import debug_print
from src.utils.domain_utils import DomainProcessor
from src.utils.tools import extract_skills


@dataclass
class Stage1Result:
    """
    阶段 1 结构化结果壳，用于逐步解耦领域与锚点阶段的中间状态。
    含 jd_profile（四层领域画像）供 Stage2 层级守卫使用。
    """

    active_domains: Set[int]
    domain_regex: str
    anchor_skills: Dict[Any, Any]
    job_ids: List[int]
    job_previews: List[Dict[str, Any]]
    dominance: float
    anchor_debug: Dict[str, Any]
    jd_profile: Optional[Dict[str, Any]] = None  # domain/field/subfield/topic_weights, active_*, main_*


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


def _batch_load_vocabulary_stats_for_tids(
    recall,
    tids: Set[int],
) -> Tuple[Dict[int, Any], Dict[int, Any]]:
    """
    批量加载 vocabulary_topic_stats / vocabulary_domain_stats，与逐条 WHERE voc_id=? 结果一致。
    返回 (topic_row_by_voc_id, domain_dist_by_voc_id)；topic 行为整行 tuple，与 fetchone 相同列序。
    """
    topic_map: Dict[int, Any] = {}
    domain_dist_map: Dict[int, Any] = {}
    if not tids or not getattr(recall, "stats_conn", None):
        return topic_map, domain_dist_map
    conn = recall.stats_conn
    tid_list = sorted(int(t) for t in tids)
    for i in range(0, len(tid_list), _SQLITE_IN_CHUNK):
        chunk = tid_list[i : i + _SQLITE_IN_CHUNK]
        if not chunk:
            continue
        ph = ",".join("?" * len(chunk))
        try:
            rows = conn.execute(
                f"SELECT voc_id, field_id, subfield_id, topic_id, field_dist, subfield_dist, topic_dist, source "
                f"FROM vocabulary_topic_stats WHERE voc_id IN ({ph})",
                chunk,
            ).fetchall()
            for row in rows:
                topic_map[int(row[0])] = row
        except Exception:
            pass
        try:
            rows = conn.execute(
                f"SELECT voc_id, domain_dist FROM vocabulary_domain_stats WHERE voc_id IN ({ph})",
                chunk,
            ).fetchall()
            for row in rows:
                domain_dist_map[int(row[0])] = row[1]
        except Exception:
            pass
    return topic_map, domain_dist_map


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
    # (tid, weight) 保留锚点×相似词的多重贡献
    tid_weight_pairs: List[Tuple[int, float]] = []

    anchor_vids: List[int] = []
    for vid_str in (anchor_skills or {}):
        try:
            anchor_vids.append(int(vid_str))
        except (TypeError, ValueError):
            continue
    # -------------------------------------------------------------------------
    # Neo4j：由「每锚点 1 次 MATCH + LIMIT k」改为「全锚点 1 次 IN + Python 侧每 src 取 top-k」
    # 等价性（指标/语义/数据）：
    #   - 过滤：与原 Cypher 相同 — v.id 锚定、r.score>=0.5、v2.type in concept/keyword。
    #   - 每锚点的 top-k：按 r.score DESC 取前 k 条；与 per-anchor ORDER BY score DESC LIMIT k
    #     一致，除非多条边 score  bitwise 完全相等（Neo4j 未保证并列次序），对加权聚合影响可忽略。
    #   - 返回的 (tid, w) 多重集合与原 N 次查询的并集一致（含同一 tid 被多锚点命中）。
    # -------------------------------------------------------------------------
    if anchor_vids:
        try:
            from itertools import groupby

            sim_rows = recall.graph.run(
                """
                MATCH (v:Vocabulary)-[r:SIMILAR_TO]->(v2:Vocabulary)
                WHERE v.id IN $vids AND r.score >= 0.5
                  AND coalesce(v2.type, 'concept') IN ['concept', 'keyword']
                RETURN v.id AS src_vid, v2.id AS tid, r.score AS s
                ORDER BY src_vid, s DESC
                """,
                vids=anchor_vids,
            ).data()
            for _src_key, group in groupby(
                sim_rows or [],
                key=lambda x: x.get("src_vid"),
            ):
                for i, r in enumerate(group):
                    if i >= SIMILAR_TOP_K:
                        break
                    tid = r.get("tid")
                    if tid is None:
                        continue
                    try:
                        tid_i = int(tid)
                    except (TypeError, ValueError):
                        continue
                    w = float(r.get("s") or 0.5)
                    tid_weight_pairs.append((tid_i, w))
        except Exception:
            tid_weight_pairs = []

    unique_tids = {t for t, _ in tid_weight_pairs}
    topic_map, domain_dist_map = _batch_load_vocabulary_stats_for_tids(recall, unique_tids)

    for tid, w in tid_weight_pairs:
        row_t = topic_map.get(tid)
        dist_json = domain_dist_map.get(tid)
        row_d = (dist_json,) if dist_json is not None else None
        if row_t:
            # 批注：SELECT 列为 voc_id, field_id, subfield_id, topic_id, field_dist, subfield_dist, topic_dist, source
            # 必须用 [4][5][6] 取三层 JSON；误用 [3] 会把 topic_id 当成 field_dist，导致 JD 画像与词侧 key 全对不上、Stage4 overlap 恒为 0。
            fd = parse_json_dist(row_t[4])
            sd = parse_json_dist(row_t[5])
            td = parse_json_dist(row_t[6])
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
    stage1_sub_ms: Dict[str, float] = {}
    _t_stage1 = time.perf_counter()

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
    _t0 = time.perf_counter()
    stage1_sub_ms["prep"] = (_t0 - _t_stage1) * 1000.0

    # JD 与 anchor_ctx / supplement 共用同一段文本切片 + 单次编码（共振键写入 jd_encode_cache）
    query_for_ctx = (semantic_query_text or query_text) or ""
    jd_canonical = label_anchors.canonical_jd_text_for_encode(query_for_ctx)
    encoder = getattr(recall, "_query_encoder", None)
    jd_encode_cache: Dict[str, np.ndarray] = {}
    setattr(recall, "_jd_query_vec_1d", None)
    if encoder and jd_canonical:
        try:
            encoder.lookup_or_encode(jd_canonical, jd_encode_cache)
            enh = encoder._apply_dynamic_resonance(jd_canonical)
            row = jd_encode_cache.get(enh)
            if row is not None:
                recall._jd_query_vec_1d = np.asarray(row, dtype=np.float32).flatten().copy()
        except Exception:
            pass

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
    _t1 = time.perf_counter()
    stage1_sub_ms["domain_detect"] = (_t1 - _t0) * 1000.0

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
            jd_encode_cache=jd_encode_cache,
        )
    _t2 = time.perf_counter()
    stage1_sub_ms["anchors"] = (_t2 - _t1) * 1000.0
    if not anchor_skills:
        # 无锚点时，后续阶段直接短路
        stage1_sub_ms["total"] = (_t2 - _t_stage1) * 1000.0
        return set(), "", {}, {
            "job_ids": job_ids,
            "job_previews": job_previews,
            "anchor_debug": anchor_debug,
            "dominance": dominance,
            "stage1_sub_ms": stage1_sub_ms,
        }

    # 方式1：扩展现有 anchor_skills，为每个锚点打上 anchor_type，供 Stage2/Stage3 按类型分策略
    for info in anchor_skills.values():
        info["anchor_type"] = label_anchors.classify_anchor_type(info.get("term") or "")

    # 调试：打印锚点及类型，便于排查
    if getattr(recall, "verbose", False) and not getattr(recall, "silent", False):
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
    _t3 = time.perf_counter()
    stage1_sub_ms["domain_regex"] = (_t3 - _t2) * 1000.0

    # 4) 锚点上下文化 + JD 四层领域画像（供 Stage2 层级守卫）
    attach_anchor_contexts(anchor_skills, query_for_ctx, window=12)
    # 条件化锚点表示：泛锚点带 JD 上下文，供 Stage2A 用 conditioned_vec 做落点打分
    if encoder and query_for_ctx and anchor_skills:
        # 复用 supplement 前写入的 JD 共振缓存，prefill 内跳过 JD 的 encode_batch 重复项
        encode_cache = jd_encode_cache
        label_anchors.prefill_encode_cache_for_anchor_ctx(
            encoder, anchor_skills, query_for_ctx, encode_cache
        )
        jd_vec_once = None
        try:
            jd_arr = label_anchors.encode_text_with_optional_cache(
                encoder, jd_canonical, encode_cache
            )
            if jd_arr is not None:
                jd_vec_once = np.asarray(jd_arr, dtype=np.float32).flatten().copy()
        except Exception:
            jd_vec_once = None
        anchor_ctx_count = 0
        for _vid, info in anchor_skills.items():
            term = (info.get("term") or "").strip()
            if not term:
                continue
            try:
                ctx = label_anchors.build_conditioned_anchor_representation(
                    term,
                    info,
                    anchor_skills,
                    query_for_ctx,
                    encoder,
                    jd_vec_precomputed=jd_vec_once,
                    encode_cache=encode_cache,
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
    _t4 = time.perf_counter()
    stage1_sub_ms["anchor_ctx"] = (_t4 - _t3) * 1000.0
    jd_profile = build_jd_hierarchy_profile(anchor_skills, query_for_ctx, recall)
    _t5 = time.perf_counter()
    stage1_sub_ms["jd_profile"] = (_t5 - _t4) * 1000.0
    jd_profile["active_domains"] = list(active_domain_set)

    debug_1: Dict[str, Any] = {
        "job_ids": job_ids,
        "job_previews": job_previews,
        "anchor_debug": anchor_debug,
        "dominance": dominance,
        "industrial_kws": industrial_kws,
        "anchor_skills": anchor_skills,
        "jd_profile": jd_profile,
        "stage1_sub_ms": stage1_sub_ms,
    }
    stage1_sub_ms["finalize"] = (time.perf_counter() - _t5) * 1000.0
    stage1_sub_ms["total"] = (time.perf_counter() - _t_stage1) * 1000.0

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

