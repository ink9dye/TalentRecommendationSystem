import time
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.time_features import compute_paper_recency
from config import ABSTRACT_MAP_PATH, INDEX_DIR

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

    def _merge_hits(old_hits: List[Dict[str, Any]], new_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        按 canonical vid 合并 hit，避免同 wid 下多次覆盖/重复。
        规则：同 vid 保留 idf 更高的一条（视为更强命中证据）。
        """
        by_vid: Dict[str, Dict[str, Any]] = {}
        for h in (old_hits or []) + (new_hits or []):
            if not isinstance(h, dict):
                continue
            vid_s = str(h.get("vid") or "")
            if not vid_s:
                continue
            prev = by_vid.get(vid_s)
            if prev is None:
                by_vid[vid_s] = dict(h)
                continue
            if float(h.get("idf") or 0.0) > float(prev.get("idf") or 0.0):
                by_vid[vid_s] = dict(h)
        return list(by_vid.values())

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
            # 加重泛交通/物流/工业流程偏题惩罚，优先清理“看似相关但主轴不对”的高分论文
            off_topic_penalty *= 0.55

        # RL 相关额外约束：如不是机器人/控制体系，则压
        if ("reinforcement learning" in term_text) or ("q-learning" in term_text):
            if not any(
                kw in t
                for kw in ["robot", "robotic", "control", "manipulator", "motion", "locomotion", "planning"]
            ):
                # RL 词必须更贴机器人/控制主轴，否则强压
                off_topic_penalty *= 0.45

        # route planning 相关额外约束：更像交通规划则压
        if "route planning" in term_text:
            if any(
                kw in t
                for kw in ["charging station", "traffic", "transportation", "vehicle", "bus", "rescue route", "ship"]
            ):
                off_topic_penalty *= 0.40

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
                    "kinematics",
                    "manipulation",
                    "grasp",
                ]
            ):
                off_topic_penalty *= 0.60

        # 提高词面命中权重，降低“只沾主轴泛词”论文的 grounding 分
        grounding = (0.55 * lexical_hit + 0.20 * anchor_axis_hit + 0.25 * jd_axis_match)
        if retrieval_role == "paper_support":
            grounding *= 0.92

        grounding = max(0.0, min(1.0, float(grounding)))
        return {"grounding": grounding, "off_topic_penalty": float(off_topic_penalty)}

    def _ensure_stage4_resources() -> None:
        """
        Stage4 资源懒加载（最小补丁）：
        1) 论文摘要向量矩阵
        2) paper_id -> row_idx 映射
        3) 缓存到 recall 对象，避免重复加载
        """
        if bool(getattr(recall, "_stage4_vec_ready", False)):
            return

        vec_path = os.path.join(INDEX_DIR, "abstract_vectors.npy")
        map_path = ABSTRACT_MAP_PATH

        paper_vecs = None
        paper_id_to_row: Dict[str, int] = {}
        paper_norms = None

        try:
            if os.path.exists(vec_path):
                paper_vecs = np.load(vec_path)
        except Exception:
            paper_vecs = None

        try:
            if os.path.exists(map_path):
                with open(map_path, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)
                if isinstance(raw_map, list):
                    paper_id_to_row = {str(pid): i for i, pid in enumerate(raw_map)}
                elif isinstance(raw_map, dict):
                    paper_id_to_row = {str(pid): int(idx) for pid, idx in raw_map.items()}
        except Exception:
            paper_id_to_row = {}

        if isinstance(paper_vecs, np.ndarray) and paper_vecs.size > 0:
            try:
                paper_norms = np.linalg.norm(paper_vecs, axis=1)
                paper_norms = np.where(paper_norms <= 1e-12, 1e-12, paper_norms)
            except Exception:
                paper_norms = None

        setattr(recall, "_paper_abstract_vecs", paper_vecs)
        setattr(recall, "_paper_id_to_row", paper_id_to_row)
        setattr(recall, "_paper_abstract_norms", paper_norms)
        setattr(recall, "_stage4_vec_ready", True)

        dim = int(paper_vecs.shape[1]) if isinstance(paper_vecs, np.ndarray) and paper_vecs.ndim == 2 else 0
        print(f"[Stage4 vec ready] papers={len(paper_id_to_row)} dim={dim}")

    def _score_paper_for_term(
        vid: int,
        wid: str,
        title: str,
        domains: str,
        year: Any,
        idf_weight: float,
        domain_bonus: float,
        term_final: float,
        role_weight: float,
        query_vec_1d: Optional[np.ndarray],
        query_norm: float,
    ) -> Dict[str, float]:
        """
        Stage4 单篇打分（最小侵入版）：
        在原有 term grounding + penalty 基础上，新增 JD↔摘要向量 jd_align 软融合。
        """
        tt = term_type_cache.get(vid)
        if tt is None:
            tt = _compute_term_type_factors(vid)
            term_type_cache[vid] = tt

        ground = _compute_grounding_score(vid, title, domains)
        term_grounding = float(ground["grounding"]) * float(tt["grounding_factor"])
        term_grounding = max(0.0, min(1.0, term_grounding))
        off_topic_penalty = float(ground["off_topic_penalty"])
        term_paper_role_factor = float(tt["paper_factor"])

        # 默认中性值：向量缺失时不一票否决
        jd_align = 0.50
        paper_vecs = getattr(recall, "_paper_abstract_vecs", None)
        paper_id_to_row = getattr(recall, "_paper_id_to_row", {}) or {}
        paper_norms = getattr(recall, "_paper_abstract_norms", None)
        row_idx = paper_id_to_row.get(str(wid))
        if (
            row_idx is not None
            and isinstance(paper_vecs, np.ndarray)
            and isinstance(paper_norms, np.ndarray)
            and query_vec_1d is not None
            and query_norm > 1e-12
            and 0 <= int(row_idx) < len(paper_vecs)
            and int(row_idx) < len(paper_norms)
        ):
            try:
                pv = np.asarray(paper_vecs[int(row_idx)], dtype=np.float32).flatten()
                if pv.size == query_vec_1d.size:
                    cos = float(np.dot(query_vec_1d, pv) / (query_norm * float(paper_norms[int(row_idx)])))
                    jd_align = 0.5 * (cos + 1.0)
                    jd_align = max(0.0, min(1.0, jd_align))
            except Exception:
                jd_align = 0.50

        # 关键融合（最小修正）：
        # 1) term_grounding 继续当主轴
        # 2) jd_align 只做弱辅助，不能“救活”term 不够像的论文
        # 3) term_grounding 很低时，JD 相似度影响进一步缩小
        jd_boost = 0.85 + 0.15 * jd_align
        if term_grounding < 0.20:
            jd_boost = 0.92 + 0.08 * jd_align

        hybrid_grounding = term_grounding * jd_boost
        hybrid_grounding = max(0.0, min(1.0, float(hybrid_grounding)))

        recency = compute_paper_recency(year, None)
        base_term_contrib = term_final * role_weight * idf_weight * domain_bonus * recency
        final_paper_score = (
            base_term_contrib
            * (0.10 + 0.90 * hybrid_grounding)
            * off_topic_penalty
            * term_paper_role_factor
        )

        return {
            "term_grounding": term_grounding,
            "jd_align": jd_align,
            "hybrid_grounding": hybrid_grounding,
            "offtopic_penalty": off_topic_penalty,
            "paper_factor": term_paper_role_factor,
            "year_factor": recency,
            "domain_bonus": domain_bonus,
            "idf_weight": idf_weight,
            "final_paper_score": final_paper_score,
        }

    def _compute_term_type_factors(vid: int) -> Dict[str, float]:
        """
        Stage4 最小 term-type 因子（无词表硬编码）：
        - 方法骨架词：grounding 略放宽、local cap 略放宽
        - 对象型词：grounding / paper 贡献 / local cap 略收紧
        """
        meta = _get_term_meta(vid)
        retrieval_role = str(meta.get("retrieval_role") or "").strip().lower()
        stage3_bucket = str(meta.get("stage3_bucket") or "").strip().lower()
        can_expand = bool(meta.get("can_expand", False))
        term_role = str(meta.get("term_role") or "").strip().lower()

        object_like_penalty = float(meta.get("object_like_penalty", 1.0) or 1.0)
        bonus_term_penalty = float(meta.get("bonus_term_penalty", 1.0) or 1.0)
        generic_penalty = float(meta.get("generic_penalty", 1.0) or 1.0)
        object_like_risk = 1.0 - max(0.0, min(1.0, object_like_penalty))
        generic_risk = 1.0 - max(0.0, min(1.0, generic_penalty))
        bonus_risk = 1.0 - max(0.0, min(1.0, bonus_term_penalty))

        method_like = (
            can_expand
            or stage3_bucket == "core"
            or retrieval_role == "paper_primary"
            or term_role == "primary"
        )
        object_like = object_like_risk >= 0.35 or object_like_penalty <= 0.82

        grounding_factor = 1.00
        paper_factor = 1.00
        local_cap = TERM_MAX_PAPERS

        if method_like and not object_like:
            grounding_factor = 1.14
            local_cap = min(TERM_MAX_PAPERS, 15)
        elif object_like:
            # 对象型 term（继续收紧版）：保留召回，但进一步压制作者榜“对象词霸榜”
            # - grounding_factor 下调：降低仅靠主轴泛命中的通过强度
            # - paper_factor 下调：削弱进入作者聚合前的单篇贡献
            # - local_cap 收紧到 3：减少同一对象词向 Stage5 喂入的高分论文数量
            grounding_factor = 0.72
            paper_factor = 0.60 if retrieval_role == "paper_primary" else 0.70
            local_cap = min(TERM_MAX_PAPERS, 3)
        else:
            # 普通词：轻微收敛，防止泛词累积分过快
            if retrieval_role == "paper_primary":
                paper_factor = 0.93
            local_cap = min(TERM_MAX_PAPERS, 12)

        # 泛词/bonus 风险高时，再做轻微收敛（不做硬门）
        if generic_risk > 0.40:
            grounding_factor *= 0.96
        if bonus_risk > 0.35:
            paper_factor *= 0.96

        return {
            # 下限放宽到 0.70，允许对象词 grounding_factor=0.72 生效
            "grounding_factor": max(0.70, min(1.20, float(grounding_factor))),
            # 下限放宽到 0.55，允许对象词 primary 使用 0.60 真正生效
            "paper_factor": max(0.55, min(1.00, float(paper_factor))),
            # 下限放到 3，确保对象型 term 的收紧策略可真实生效
            "local_cap": int(max(3, min(TERM_MAX_PAPERS, local_cap))),
        }

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
    # 批注：将单一硬门改为“双阈值门”，保留主命中严格性，同时允许高 JD 对齐的弱辅助命中进入。
    PRIMARY_GROUNDING_MIN = 0.12
    SECONDARY_GROUNDING_MIN = 0.06
    SECONDARY_JD_ALIGN_MIN = 0.78

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

    # Stage4 摘要向量资源懒加载（只做一次）
    _ensure_stage4_resources()
    query_vec_1d: Optional[np.ndarray] = None
    query_norm: float = 0.0
    try:
        enc = getattr(recall, "_query_encoder", None)
        if enc is not None and jd_text:
            qv, _ = enc.encode(jd_text)
            if qv is not None:
                query_vec_1d = np.asarray(qv, dtype=np.float32).flatten()
                query_norm = float(np.linalg.norm(query_vec_1d))
    except Exception:
        query_vec_1d = None
        query_norm = 0.0

    term_capped_unique_wids: Dict[int, set] = defaultdict(set)
    wid_to_paper_meta: Dict[str, Dict[str, Any]] = {}  # wid -> {title, domains, year}
    term_type_cache: Dict[int, Dict[str, float]] = {}
    # 调试断点：cap 前/后按 term 的论文行，用于 overlap 与生存性分析
    term_rows_before_cap: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    term_rows_after_cap: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    # 审计用：按 term 收集进入 Stage4 评分链路的论文分解，便于定位 Stage4->Stage5 放大点
    term_kept_paper_audit: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
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
        wid_to_paper_meta[wid] = {"title": title, "domains": domains, "year": year}

        term_final = _term_score(vid)
        role_weight = get_term_role_weight(term_retrieval_roles, vid)
        score_detail = _score_paper_for_term(
            vid=vid,
            wid=wid,
            title=title,
            domains=domains,
            year=year,
            idf_weight=idf_weight,
            domain_bonus=domain_bonus,
            term_final=term_final,
            role_weight=role_weight,
            query_vec_1d=query_vec_1d,
            query_norm=query_norm,
        )
        term_contrib = float(score_detail["final_paper_score"])
        grounding = float(score_detail["term_grounding"])  # 硬门：先保证论文真的像这个 term
        hybrid_grounding = float(score_detail["hybrid_grounding"])  # 软排：JD 只参与排序
        jd_align = float(score_detail["jd_align"])
        off_topic_penalty = float(score_detail["offtopic_penalty"])
        # 记录“论文贡献前的因子分解”：只用于 debug 打印，不参与排序逻辑
        term_kept_paper_audit[vid].append(
            {
                "paper_id": wid,
                "title": title,
                "term_grounding": float(score_detail["term_grounding"]),
                "jd_align": float(score_detail["jd_align"]),
                "hybrid_grounding": float(score_detail["hybrid_grounding"]),
                "grounding": grounding,
                "gating_grounding": grounding,
                "hybrid_grounding": hybrid_grounding,
                "offtopic_penalty": off_topic_penalty,
                "paper_factor": float(score_detail["paper_factor"]),
                "year_factor": float(score_detail["year_factor"]),
                "domain_bonus": domain_bonus,
                "idf_weight": idf_weight,
                "final_paper_score": term_contrib,
            }
        )

        # 双阈值门：
        # - primary：grounding 必须 >= 0.12
        # - secondary：允许 grounding 较低但 jd_align 足够高的辅助命中进入
        allow_hit = False
        hit_level = "primary"
        if grounding >= PRIMARY_GROUNDING_MIN:
            allow_hit = True
            hit_level = "primary"
        elif grounding >= SECONDARY_GROUNDING_MIN and jd_align >= SECONDARY_JD_ALIGN_MIN:
            allow_hit = True
            hit_level = "secondary"

        if not allow_hit:
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
                        "jd_align": jd_align,
                    }
                )
            continue

        term_funnel_counts[vid]["after_grounding_gate"] += 1
        # 批注：secondary 命中保留但做轻降权，防止弱命中反客为主。
        effective_term_contrib = term_contrib * (0.65 if hit_level == "secondary" else 1.0)
        by_term[vid].append(
            (
                wid,
                effective_term_contrib,
                idf_weight,
                hit_level,
                grounding,
                jd_align,
            )
        )
        term_rows_before_cap[vid].append(
            {
                "pid": wid,
                "final_paper_score": float(effective_term_contrib),
            }
        )

    # 每个 term 最多保留 TERM_MAX_PAPERS 篇（按 term_contrib 降序）
    limited: List[tuple] = []
    for vid, triples in by_term.items():
        triples.sort(key=lambda x: -x[1])
        tt = term_type_cache.get(vid)
        if tt is None:
            tt = _compute_term_type_factors(vid)
            term_type_cache[vid] = tt
        local_cap = int(tt["local_cap"])

        term_funnel_counts[vid]["after_offtopic_penalty_sort"] = term_funnel_counts[vid][
            "after_grounding_gate"
        ]

        cut_cnt = max(0, len(triples) - local_cap)
        term_reject_reason_counts[vid]["local_cap_cut"] += cut_cnt

        # local cap samples（用于打印，不影响 limited）
        if cut_cnt > 0 and len(term_local_cap_cut_samples[vid]) < 3:
            cut_triples = triples[local_cap : local_cap + 3]
            for (cut_wid, *_rest) in cut_triples:
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
        for (kept_wid, *_rest) in kept_triples:
            term_capped_unique_wids[vid].add(kept_wid)
            term_rows_after_cap[vid].append({"pid": kept_wid})

        for (wid, term_contrib, idf_weight, hit_level, grounding, jd_align) in triples[:local_cap]:
            limited.append((wid, vid, term_contrib, idf_weight, hit_level, grounding, jd_align))

        # 批注：输出每个 term 在关键层级的数量，优先判断是 grounding 还是 cap 在砍样本。
        meta = _get_term_meta(vid)
        term_name = str(meta.get("term") or meta.get("anchor") or meta.get("parent_primary") or vid)
        print(
            f"[Stage4 term stats] term='{term_name}' "
            f"raw={term_funnel_counts[vid].get('cypher_raw', 0)} "
            f"after_grounding={term_funnel_counts[vid].get('after_grounding_gate', 0)} "
            f"before_cap={len(term_rows_before_cap.get(vid, []))} "
            f"after_cap={len(term_rows_after_cap.get(vid, []))}"
        )
        rows_for_paper_map = kept_triples
        term = term_name
        print(f"[Stage4 paper_map source] term='{term}' source_stage='after_cap' rows={len(rows_for_paper_map)}")

    # 断点 1：local cap 前的 cross-term overlap
    print("\n[Stage4 overlap before local-cap]")
    term_pid_map_before_local_cap: Dict[int, set] = {
        vid: {str(r.get("pid")) for r in rows if r.get("pid") is not None}
        for vid, rows in term_rows_before_cap.items()
    }
    overlap_pairs: List[tuple] = []
    v_ids_sorted = sorted(term_pid_map_before_local_cap.keys())
    for i, vid_a in enumerate(v_ids_sorted):
        for vid_b in v_ids_sorted[i + 1 :]:
            inter = sorted(term_pid_map_before_local_cap[vid_a].intersection(term_pid_map_before_local_cap[vid_b]))
            if not inter:
                continue
            name_a = str((_get_term_meta(vid_a) or {}).get("term") or vid_a)
            name_b = str((_get_term_meta(vid_b) or {}).get("term") or vid_b)
            print(f"  {name_a} x {name_b} -> overlap={len(inter)} sample={inter[:10]}")
            overlap_pairs.append((vid_a, vid_b, inter))

    # 统计型打印：全局 multi-hit 潜力（cap 前）
    pid_term_count: Dict[str, set] = defaultdict(set)
    for vid, rows in term_rows_before_cap.items():
        term_name = str((_get_term_meta(vid) or {}).get("term") or vid)
        for r in rows:
            pid = str(r.get("pid") or "")
            if pid:
                pid_term_count[pid].add(term_name)
    multi_candidates = [(pid, sorted(list(ts))) for pid, ts in pid_term_count.items() if len(ts) >= 2]
    print(f"\n[Stage4 multi-hit potential before cap] count={len(multi_candidates)}")
    for pid, terms in multi_candidates[:20]:
        print(f"  pid='{pid}' terms={terms}")

    # 断点 2：overlap 论文在各 term 的 cap 前后生存情况
    cross_term_overlap_pids = {pid for (_, _, inter) in overlap_pairs for pid in inter}
    print("\n[Stage4 overlap survival audit]")
    for vid, rows in term_rows_before_cap.items():
        term_name = str((_get_term_meta(vid) or {}).get("term") or vid)
        rows_sorted = sorted(rows, key=lambda x: float(x.get("final_paper_score") or 0.0), reverse=True)
        top_after_cap = {str(r.get("pid")) for r in term_rows_after_cap.get(vid, []) if r.get("pid") is not None}
        for rank, r in enumerate(rows_sorted, 1):
            pid = str(r.get("pid") or "")
            if not pid or pid not in cross_term_overlap_pids:
                continue
            print(
                f"term='{term_name}' pid='{pid}' "
                f"rank_before_cap={rank} "
                f"score={float(r.get('final_paper_score') or 0.0):.3f} "
                f"survive_after_cap={pid in top_after_cap}"
            )

    # 按 wid 聚合：paper_score = Σ term_contrib，hits 合并为 canonical term 证据
    by_wid: Dict[str, Dict[str, Any]] = {}
    for (wid, vid, term_contrib, idf_weight, hit_level, grounding, jd_align) in limited:
        tt = term_type_cache.get(vid)
        if tt is None:
            tt = _compute_term_type_factors(vid)
            term_type_cache[vid] = tt
        meta = _get_term_meta(vid)
        retrieval_role = (
            term_retrieval_roles.get(vid)
            or term_retrieval_roles.get(str(vid))
            or meta.get("retrieval_role")
            or "paper_primary"
        )
        hit = {
            "vid": str(vid),
            # 批注：Stage4 不再二次清洗 term，直接使用 Stage3/canonical term。
            "term": str(meta.get("term") or ""),
            "idf": float(idf_weight),
            "role": str(retrieval_role),
            "term_score": float(_term_score(vid)),
            "paper_factor": float(tt.get("paper_factor") or 1.0),
            "hit_level": str(hit_level),
            "grounding": float(grounding),
            "jd_align": float(jd_align),
        }
        if wid not in by_wid:
            wid_meta = wid_to_paper_meta.get(wid) or {"title": "", "domains": ""}
            by_wid[wid] = {
                "wid": wid,
                "title": wid_meta.get("title") or "",
                "year": wid_meta.get("year"),
                "domains": wid_meta.get("domains") or "",
                "paper_score": 0.0,
                "hits": [],
            }
        old_hits = list(by_wid[wid].get("hits") or [])
        old_terms = [str(h.get("term") or h.get("vid") or "") for h in old_hits if isinstance(h, dict)]
        by_wid[wid]["paper_score"] = float(by_wid[wid]["paper_score"]) + float(term_contrib)
        by_wid[wid]["hits"] = _merge_hits(by_wid[wid].get("hits") or [], [hit])
        new_hits = list(by_wid[wid].get("hits") or [])
        new_terms = [str(h.get("term") or h.get("vid") or "") for h in new_hits if isinstance(h, dict)]
        # 断点 3：paper_map 写入过程，确认是“追加合并”还是“覆盖丢失”
        print(
            f"[Stage4 paper_map write] wid='{wid}' "
            f"incoming_term='{str(hit.get('term') or hit.get('vid') or '')}' "
            f"old_hit_count={len(old_hits)} new_hit_count={len(new_hits)} "
            f"old_terms={old_terms} new_terms={new_terms}"
        )

    # 全局按 paper_score 排序，取前 GLOBAL_PAPER_LIMIT
    sorted_wids = sorted(
        by_wid.keys(),
        key=lambda w: -float((by_wid[w] or {}).get("paper_score") or 0.0),
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
            kept_wids.sort(key=lambda w: -float((by_wid.get(w) or {}).get("paper_score") or 0.0))
            kept_wids = kept_wids[:3]
            for i, w in enumerate(kept_wids, start=1):
                meta3 = wid_to_paper_meta.get(w) or {"title": "", "domains": ""}
                g = _compute_grounding_score(vid, meta3.get("title") or "", meta3.get("domains") or "")
                final_score = float((by_wid.get(w) or {}).get("paper_score") or 0.0)
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
                final_score = float((by_wid.get(w) or {}).get("paper_score") or 0.0)
                # 是否 low_grounding：看 grounding 是否低于阈值
                gg = float(g.get("grounding") or 0.0)
                reason = "low_grounding" if gg < PRIMARY_GROUNDING_MIN else "pruned_after_terms"
                print(
                    f"[Stage4 rejected papers] term='{term_name}' pid='{w}' reason='{reason}' "
                    f"grounding={float(g.get('grounding') or 0.0):.3f} "
                    f"penalty={float(g.get('off_topic_penalty') or 1.0):.3f} "
                    f"final_paper_score={final_score:.3f} title={meta3.get('title')[:80]!r}"
                )

    # 精准审计块 1/3：
    # [Stage4 kept paper score audit] 聚焦“论文分 -> 作者分”前的 paper_factor 分解，
    # 用于判断是 Stage4 因子过松，还是 Stage5 聚合过度放大。
    print("\n[Stage4 kept paper score audit]")
    for vid in v_ids:
        meta = _get_term_meta(vid)
        term_name = str(meta.get("term") or meta.get("anchor") or meta.get("parent_primary") or vid)
        rows_audit = term_kept_paper_audit.get(vid, [])
        rows_audit.sort(key=lambda x: float(x.get("final_paper_score") or 0.0), reverse=True)
        print(f"term='{term_name}' kept={len(rows_audit)}")
        for i, p in enumerate(rows_audit[:10], 1):
            print(
                f"[Stage4 jd audit] term='{term_name}' pid='{p.get('paper_id')}' "
                f"tg={float(p.get('term_grounding') or 0.0):.3f} "
                f"jd={float(p.get('jd_align') or 0.5):.3f} "
                f"hybrid={float(p.get('hybrid_grounding') or p.get('grounding') or 0.0):.3f} "
                f"penalty={float(p.get('offtopic_penalty') or 1.0):.3f} "
                f"final={float(p.get('final_paper_score') or 0.0):.3f} "
                f"title={str(p.get('title') or '')[:80]!r}"
            )
            print(
                f"  #{i} pid={p.get('paper_id')} "
                f"tg={float(p.get('term_grounding') or 0.0):.3f} "
                f"jd={float(p.get('jd_align') or 0.5):.3f} "
                f"hybrid={float(p.get('hybrid_grounding') or p.get('grounding') or 0.0):.3f} "
                f"grounding={float(p.get('grounding') or 0.0):.3f} "
                f"penalty={float(p.get('offtopic_penalty') or 1.0):.3f} "
                f"paper_factor={float(p.get('paper_factor') or 1.0):.3f} "
                f"year_factor={float(p.get('year_factor') or 1.0):.3f} "
                f"domain_bonus={float(p.get('domain_bonus') or 1.0):.3f} "
                f"idf_weight={float(p.get('idf_weight') or 0.0):.3f} "
                f"final_paper_score={float(p.get('final_paper_score') or 0.0):.3f} "
                f"title={str(p.get('title') or '')[:70]!r}"
            )

    if not sorted_wids:
        sub_ms["total"] = (time.perf_counter() - t0) * 1000.0
        _save_sub(sub_ms)
        return []

    # 审计 1：确认 Stage4 内部 wid 聚合后确实存在 multi-hit 论文
    print("\n[Stage4 merged wid multi-hit audit]")
    stage4_multi_hit_rows: List[tuple] = []
    for wid, rec in by_wid.items():
        hits = rec.get("hits") or []
        if len(hits) >= 2:
            terms = [str(h.get("term") or h.get("vid") or "") for h in hits if isinstance(h, dict)]
            stage4_multi_hit_rows.append((wid, rec.get("title") or "", terms))
    print(f"multi_hit_papers={len(stage4_multi_hit_rows)}")
    for wid, title, terms in stage4_multi_hit_rows[:20]:
        print(f"  wid={wid} hit_terms={terms} title='{(title or '')[:100]}'")
    # 断点 4A：merged paper_map 细节（每条多 hit 明细）
    print("\n[Stage4 merged wid multi-hit detail]")
    for wid, rec in by_wid.items():
        hits = rec.get("hits") or []
        if len(hits) < 2:
            continue
        terms = [str(h.get("term") or h.get("vid") or "") for h in hits if isinstance(h, dict)]
        print(
            f"wid='{wid}' hit_count={len(hits)} terms={terms} "
            f"title='{str(rec.get('title') or '')[:100]}'"
        )

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
    wid_to_hits_and_score = {
        str(wid): (
            list((rec or {}).get("hits") or []),
            float((rec or {}).get("paper_score") or 0.0),
            (rec or {}).get("title"),
            (rec or {}).get("domains"),
            (rec or {}).get("year"),
        )
        for wid, rec in by_wid.items()
    }

    out: List[Dict[str, Any]] = []
    for rec in author_rows:
        aid = rec.get("aid")
        papers_raw = rec.get("papers") or []
        papers = []
        for p in papers_raw:
            wid = p.get("wid")
            if wid is None:
                continue
            wid_s = str(wid)
            hits, score, title_s4, domains_s4, year_s4 = wid_to_hits_and_score.get(wid_s, ([], 0.0, None, None, None))
            papers.append({
                "wid": wid_s,
                "hits": hits,
                "weight": p.get("weight"),
                "title": title_s4 if title_s4 is not None else p.get("title"),
                "year": year_s4 if year_s4 is not None else p.get("year"),
                "domains": domains_s4 if domains_s4 is not None else p.get("domains"),
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

    # 审计 2：确认下发到 author payload 的论文仍携带 multi-hit
    print("\n[Stage4 author payload audit]")
    payload_multi_cnt = 0
    for rec in out[:20]:
        aid = rec.get("aid")
        for p in (rec.get("papers") or [])[:5]:
            hits = p.get("hits") or []
            if len(hits) >= 2:
                payload_multi_cnt += 1
                terms = [str(h.get("term") or h.get("vid") or "") for h in hits if isinstance(h, dict)]
                print(f"aid={aid} wid={p.get('wid')} hit_terms={terms}")
    print(f"author_payload_multi_hit_papers={payload_multi_cnt}")
    # 断点 4B：author payload 多 hit 明细
    print("\n[Stage4 author payload multi-hit detail]")
    for rec in out:
        aid = rec.get("aid")
        for p in rec.get("papers") or []:
            hits = p.get("hits") or []
            if len(hits) < 2:
                continue
            terms = [str(h.get("term") or h.get("vid") or "") for h in hits if isinstance(h, dict)]
            print(f"aid='{aid}' wid='{p.get('wid')}' hit_count={len(hits)} terms={terms}")
    return out
