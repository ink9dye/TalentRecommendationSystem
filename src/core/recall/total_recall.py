# -*- coding: utf-8 -*-
"""
多路召回总控：统一编码、三路并行召回、候选池构建、打分、特征补全、硬过滤、分桶、导出。

Step1：主链在分桶后拆出 base（候选中间层）、training pool（骨架 top100）、display pool（分桶截断 top50）；
后续 Step 仍会调整 hard filter / 分数 / 分桶等，本文件内注释与命名与之对齐。
"""
import sys
from pathlib import Path
import time
import faiss
import os
import copy
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

# 允许以“python path/to/total_recall.py”直接运行（Windows 下常见），不改变作为包导入时的行为
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.recall.input_to_vector import QueryEncoder
from src.core.recall.vector_path import VectorPath
from src.core.recall.label_path import LabelRecallPath
from src.core.recall.collaboration_path import CollaborativeRecallPath
from src.core.recall.candidate_pool import (
    CandidateRecord,
    CandidatePool,
    PoolDebugSummary,
)
from src.core.recall.candidate_features import (
    extract_terms_from_label_evidence,
    infer_dominant_path,
    batch_load_author_stats_from_sqlite,
    batch_load_recent_author_stats,
    batch_load_top_works,
    calc_query_author_topic_similarity,
    calc_domain_consistency,
    calc_recent_activity_match,
    calc_skill_coverage_ratio,
    calc_paper_hit_strength,
    calc_top_work_quality,
    bucket_quota_truncate,
    build_kgatax_feature_row,
    extract_vector_evidence_summary,
    compute_vector_evidence_bonus_from_summary,
    VectorEvidenceBonusConfig,
    DEFAULT_VECTOR_EVIDENCE_BONUS_CONFIG,
    candidate_has_vector_origin,
    apply_vector_evidence_bonus_for_candidate,
)
from src.utils.domain_detector import DomainDetector
from src.utils.domain_utils import DomainProcessor

# 限制底层模型并行，防止多线程冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# RRF 与配额默认值
RRF_K = 60
PATH_WEIGHTS = {"v": 2.0, "l": 3.0, "c": 1.0}
# Step4：多路命中“轻微鼓励”系数（粗排 prior 的 tie-break 级别，不承担预精排裁决）
ALPHA_MULTI_PATH = 0.02
# 历史主链截断上限；Step1 后主链已改用 DISPLAY_POOL_TOP_N 参与分桶截断，本常量仅保留以免外部误删引用时报错
FINAL_POOL_TOP_N = 200
# Step1：双池骨架上限（训练池 / 展示池）；后续 Step5 可再分叉策略，不在本步改分桶实现本身
TRAINING_POOL_TOP_N = 100
DISPLAY_POOL_TOP_N = 50
# 分桶配额（多路一致优先）
BUCKET_QUOTAS = {"A": 80, "B": 30, "C": 60, "D": 20, "E": 20, "F": 10}


def _ensure_meta_list(result: Any, path_tag: str) -> List[Dict[str, Any]]:
    """
    将三路返回值统一为 meta 列表。若为 (id_list, duration) 则取 id_list；若为 id_list 则直接用。
    若元素已是 dict 则原样返回，否则按 author_id 与 rank 转为 meta。
    """
    if isinstance(result, (list, tuple)) and len(result) == 2 and not isinstance(result[0], dict):
        id_list, _ = result
    else:
        id_list = result
    if not id_list:
        return []
    if isinstance(id_list[0], dict):
        return id_list
    return [
        {
            "author_id": str(aid),
            f"{path_tag}_rank": i + 1,
            f"{path_tag}_score_raw": None,
            f"{path_tag}_evidence": None,
        }
        for i, aid in enumerate(id_list)
    ]


class TotalRecallSystem:
    DOMAIN_PROMPTS = {
        "1": "CS, IT, Software", "2": "Medicine, Biology", "3": "Politics, Law",
        "4": "Engineering, Manufacturing", "5": "Physics, Energy", "6": "Material Science",
        "7": "Biology, Life", "8": "Geography, Earth", "9": "Chemistry",
        "10": "Business, Management", "11": "Sociology", "12": "Philosophy",
        "13": "Environment", "14": "Mathematics", "15": "Psychology",
        "16": "Geology", "17": "Economics"
    }

    def __init__(
        self,
        k_vector: int = 150,
        k_label: int = 150,
        k_collab: int = 80,
        vector_evidence_bonus_config: Optional[VectorEvidenceBonusConfig] = None,
    ):
        print("[*] 正在初始化全量召回系统 (Training-Safe Mode)...", flush=True)
        self.encoder = QueryEncoder()
        self.v_path = VectorPath(recall_limit=k_vector)
        self.l_path = LabelRecallPath(recall_limit=k_label, silent=True)
        self.c_path = CollaborativeRecallPath(recall_limit=k_collab)
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.domain_detector = DomainDetector(self.l_path)
        # Step8：默认开启来源门控，比 Step7 更保守、可解释；单次 execute 可覆盖
        self.vector_evidence_bonus_config = (
            vector_evidence_bonus_config
            if vector_evidence_bonus_config is not None
            else DEFAULT_VECTOR_EVIDENCE_BONUS_CONFIG
        )

    def _get_author_works(self, author_id, top_n=2):
        """利用知识图谱获取作者贡献度最高的代表作"""
        try:
            graph = self.l_path.graph
            cypher = """
            MATCH (a:Author {id: $aid})-[r:AUTHORED]->(w:Work)
            RETURN w.title AS title, w.year AS year, r.pos_weight AS weight
            ORDER BY r.pos_weight DESC LIMIT $limit
            """
            return graph.run(cypher, aid=author_id, limit=top_n).data()
        except Exception:
            return []

    def build_candidate_records(
        self,
        v_meta: List[Dict],
        l_meta: List[Dict],
        c_meta: List[Dict],
        summary: PoolDebugSummary,
    ) -> List[CandidateRecord]:
        """去重合并三路，生成 CandidateRecord，填 rank/path/score_raw/evidence。"""
        summary.v_after_quota = len(v_meta)
        summary.l_after_quota = len(l_meta)
        summary.c_after_quota = len(c_meta)
        all_aids = set()
        for m in v_meta:
            all_aids.add(str(m.get("author_id", m.get("author_id"))))
        for m in l_meta:
            all_aids.add(str(m.get("author_id", m.get("author_id"))))
        for m in c_meta:
            all_aids.add(str(m.get("author_id", m.get("author_id"))))
        summary.before_dedup_count = len(all_aids)

        # 按 author_id 合并
        by_id: Dict[str, CandidateRecord] = {}
        for rank, m in enumerate(v_meta):
            aid = str(m.get("author_id"))
            if aid not in by_id:
                by_id[aid] = CandidateRecord(author_id=aid)
            by_id[aid].from_vector = True
            by_id[aid].vector_rank = m.get("vector_rank") or rank + 1
            by_id[aid].vector_score_raw = m.get("vector_score_raw")
            by_id[aid].vector_evidence = m.get("vector_evidence")
        for rank, m in enumerate(l_meta):
            aid = str(m.get("author_id"))
            if aid not in by_id:
                by_id[aid] = CandidateRecord(author_id=aid)
            rec = by_id[aid]
            rec.from_label = True
            rec.label_rank = m.get("label_rank") or rank + 1
            rec.label_score_raw = m.get("label_score_raw")
            rec.label_evidence = m.get("label_evidence")
            terms = extract_terms_from_label_evidence(rec.label_evidence)
            rec.label_term_count = len(terms)
            rec.label_core_term_count = sum(1 for t in terms if (t.get("bucket") or "").lower() == "core")
            rec.label_support_term_count = sum(1 for t in terms if (t.get("bucket") or "").lower() == "support")
            rec.label_risky_term_count = sum(1 for t in terms if (t.get("bucket") or "").lower() == "risky")
            rec.label_best_term_score = max((t.get("score") or 0.0) for t in terms) if terms else 0.0
        for rank, m in enumerate(c_meta):
            aid = str(m.get("author_id"))
            if aid not in by_id:
                by_id[aid] = CandidateRecord(author_id=aid)
            by_id[aid].from_collab = True
            by_id[aid].collab_rank = m.get("collab_rank") or rank + 1
            by_id[aid].collab_score_raw = m.get("collab_score_raw")
            by_id[aid].collab_evidence = m.get("collab_evidence")

        for r in by_id.values():
            r.path_count = sum([r.from_vector, r.from_label, r.from_collab])
            r.is_multi_path_hit = r.path_count >= 2
        summary.after_dedup_count = len(by_id)
        return list(by_id.values())

    def score_candidate_pool(
        self,
        records: List[CandidateRecord],
        bonus_cfg: Optional[VectorEvidenceBonusConfig] = None,
    ) -> List[CandidateRecord]:
        """
        Step4：候选中间层粗排 prior（light fusion / coarse score）。

        目标：为 base candidate layer 提供一个轻量、可解释、不过度裁决的合成分，用于“粗排序”与 tie-break。
        - 保留 RRF 作为主体（来源共识/多路命中倾向）；
        - 仅保留弱 multi-path / pair bonus 与轻量 label_hint / vector_evidence_bonus 做校准；
        - 该分数不是最终精排分、不是训练标签、也不替 KGAT-AX 做排序裁决。
        """
        cfg = bonus_cfg if bonus_cfg is not None else self.vector_evidence_bonus_config
        for r in records:
            rrf = 0.0
            if r.vector_rank is not None:
                rrf += PATH_WEIGHTS["v"] * (1.0 / (RRF_K + r.vector_rank))
            if r.label_rank is not None:
                rrf += PATH_WEIGHTS["l"] * (1.0 / (RRF_K + r.label_rank))
            if r.collab_rank is not None:
                rrf += PATH_WEIGHTS["c"] * (1.0 / (RRF_K + r.collab_rank))
            r.rrf_score = rrf
            r.multi_path_bonus = ALPHA_MULTI_PATH * max(0, r.path_count - 1)
            pair_bonus = 0.0
            if r.from_vector and r.from_label:
                # Step4：弱 bonus / tie-break（不代表训练标签，不应主导排序）
                pair_bonus += 0.04
            elif r.from_vector and r.from_collab:
                pair_bonus += 0.02
            elif r.from_label and r.from_collab:
                pair_bonus += 0.015
            r.pair_path_bonus = pair_bonus
            label_hint = 0.0
            if r.from_label:
                # Step4：轻量校准（非弱监督标签）
                label_hint += min(0.05, 0.01 * r.label_core_term_count)
                label_hint -= min(0.05, 0.01 * r.label_risky_term_count)
            prior_score = r.rrf_score + r.multi_path_bonus + r.pair_path_bonus + label_hint
            # Step7/8：主分形成后再加极小 vector evidence bonus；Step8 用配置 + 来源门控收紧生效范围
            summ = extract_vector_evidence_summary(getattr(r, "vector_evidence", None))
            ev_bonus = apply_vector_evidence_bonus_for_candidate(r, summ, cfg)
            r.vector_evidence_bonus = ev_bonus
            r.candidate_pool_score = prior_score + ev_bonus
            r.dominant_recall_path = infer_dominant_path(r)
        records.sort(key=lambda x: x.candidate_pool_score, reverse=True)
        return records

    def _build_training_pool_records(
        self,
        base_records: List[CandidateRecord],
        top_n: int = TRAINING_POOL_TOP_N,
    ) -> List[CandidateRecord]:
        """
        Step5：训练池生成策略（与展示池显式分叉）。

        - training pool：服务训练样本构造，优先保留来源多样性与可分层候选，允许一定边界样本进入；
          这是“轻量补位”，不是展示池那种分桶配额系统，也不是重规则/teacher 分。
        - display pool：仍由分桶 + 截断生成（在 execute 中保持原逻辑不动）。
        """
        if not base_records:
            return []

        # base_records 当前已按 coarse prior（candidate_pool_score）降序排列；本函数在该主序上做轻量补位
        def _group_key(r: CandidateRecord) -> str:
            if getattr(r, "path_count", 0) >= 2:
                return "multi_path"
            if getattr(r, "from_label", False):
                return "label_backed_single"
            if getattr(r, "from_vector", False):
                return "vector_only"
            if getattr(r, "from_collab", False):
                return "collab_only"
            return "other"

        # 1) 主干：先取较小前段作为 backbone（刻意小于 top_n，给后续补位留空间）
        backbone_n = min(70, top_n, len(base_records))
        selected: List[CandidateRecord] = []
        selected_ids = set()
        for r in base_records[:backbone_n]:
            if r.author_id not in selected_ids:
                selected.append(r)
                selected_ids.add(r.author_id)

        # 2) 从较大的前段窗口里补位（避免从尾部深挖噪声）
        window_n = min(max(top_n * 6, 300), len(base_records))
        window = base_records[:window_n]

        by_group: Dict[str, List[CandidateRecord]] = {
            "multi_path": [],
            "label_backed_single": [],
            "vector_only": [],
            "collab_only": [],
            "other": [],
        }
        for r in window:
            by_group[_group_key(r)].append(r)

        def _add_from_group(g: str, need: int) -> None:
            if need <= 0:
                return
            for r in by_group.get(g, []):
                if len(selected) >= top_n:
                    return
                if r.author_id in selected_ids:
                    continue
                selected.append(r)
                selected_ids.add(r.author_id)
                need -= 1
                if need <= 0:
                    return

        # 3) 轻量“来源层次”补位：不做复杂比例控制，只设几个保守下限（不足则跳过）
        def _cnt(g: str) -> int:
            return sum(1 for r in selected if _group_key(r) == g)

        # multi-path 仍应占优，但不把训练池压成展示池
        _add_from_group("multi_path", max(0, min(30, top_n // 2) - _cnt("multi_path")))
        # label-backed 单路：保留结构化证据入口
        _add_from_group("label_backed_single", max(0, 15 - _cnt("label_backed_single")))
        # vector-only：保留语义相似但证据较薄的对比样本
        _add_from_group("vector_only", max(0, 15 - _cnt("vector_only")))
        # 少量 collab-only：用于边界关系对比（若存在）
        _add_from_group("collab_only", max(0, 5 - _cnt("collab_only")))

        # 4) 软标记边界样本：给 risk_flags 非空候选留少量机会（不强保全量）
        risk_pool = [r for r in window if getattr(r, "risk_flags", None)]
        risk_in_selected = sum(1 for r in selected if getattr(r, "risk_flags", None))
        risk_target = min(10, max(0, len(risk_pool)))
        if risk_in_selected < risk_target:
            need = risk_target - risk_in_selected
            for r in risk_pool:
                if len(selected) >= top_n:
                    break
                if r.author_id in selected_ids:
                    continue
                selected.append(r)
                selected_ids.add(r.author_id)
                need -= 1
                if need <= 0:
                    break

        # 5) 最后按主序补满
        if len(selected) < top_n:
            for r in window:
                if len(selected) >= top_n:
                    break
                if r.author_id in selected_ids:
                    continue
                selected.append(r)
                selected_ids.add(r.author_id)

        # 保持与 base 主序一致（按 base_records 中的顺序输出）
        idx = {r.author_id: i for i, r in enumerate(base_records)}
        selected.sort(key=lambda r: idx.get(r.author_id, 10**9))
        selected = selected[:top_n]

        # 若仍退化为“机械前缀”，则做一次最小扰动：从 top_n 之后补入 1 个可解释的候选（优先 soft-flag / label-backed）
        # 目的：保证 training pool 与 base[:top_n] 在策略上“显式分叉”，但不引入复杂配额或重排序体系。
        if len(base_records) > top_n:
            prefix_ids = [r.author_id for r in base_records[:top_n]]
            selected_ids_list = [r.author_id for r in selected]
            if selected_ids_list == prefix_ids:
                tail_window = base_records[top_n:window_n]
                swap_in = None
                for r in tail_window:
                    if r.author_id in selected_ids:
                        continue
                    if getattr(r, "risk_flags", None):
                        swap_in = r
                        break
                if swap_in is None:
                    for r in tail_window:
                        if r.author_id in selected_ids:
                            continue
                        if getattr(r, "from_label", False):
                            swap_in = r
                            break
                if swap_in is None:
                    for r in tail_window:
                        if r.author_id in selected_ids:
                            continue
                        swap_in = r
                        break
                if swap_in is not None and selected:
                    # 只替换最后一位，保证对粗排 prior 的尊重（前段排序不动）
                    drop = selected.pop(-1)
                    selected_ids.discard(drop.author_id)
                    selected.append(swap_in)
                    selected_ids.add(swap_in.author_id)
                    selected.sort(key=lambda r: idx.get(r.author_id, 10**9))

        return selected

    def _enrich_candidate_features(
        self,
        records: List[CandidateRecord],
        active_domains: Optional[Any],
        query_text: str,
    ) -> None:
        """补全作者静态指标与 query-author 交叉特征，供精排与硬过滤使用。"""
        if not records:
            return
        aids = [r.author_id for r in records]
        author_stats = batch_load_author_stats_from_sqlite(aids)
        author_recent_stats = batch_load_recent_author_stats(aids)
        author_top_works = batch_load_top_works(aids)
        query_vec, _ = self.encoder.encode(query_text)
        active_domain_set = None
        if active_domains:
            active_domain_set = set(str(d) for d in active_domains) if not isinstance(active_domains, set) else active_domains

        for r in records:
            st = author_stats.get(r.author_id) or {}
            rt = author_recent_stats.get(r.author_id) or {}
            tw = author_top_works.get(r.author_id) or []
            r.author_name = st.get("name")
            r.h_index = float(st["h_index"]) if st.get("h_index") is not None else None
            r.works_count = int(st["works_count"]) if st.get("works_count") is not None else None
            r.cited_by_count = int(st["cited_by_count"]) if st.get("cited_by_count") is not None else None
            r.recent_works_count = rt.get("recent_works_count")
            r.recent_citations = rt.get("recent_citations")
            r.institution_level = rt.get("institution_level_score")
            r.top_work_quality = calc_top_work_quality(tw)
            r.topic_similarity = calc_query_author_topic_similarity(query_vec, tw)
            r.domain_consistency = calc_domain_consistency(active_domain_set, tw)
            r.recent_activity_match = calc_recent_activity_match(rt, tw)
            r.skill_coverage_ratio = calc_skill_coverage_ratio(
                r.label_term_count, r.label_core_term_count, r.label_support_term_count,
            )
            r.paper_hit_strength = calc_paper_hit_strength(r.label_evidence, r.vector_evidence, tw)

    def _apply_hard_filters(
        self,
        records: List[CandidateRecord],
        summary: PoolDebugSummary,
    ) -> List[CandidateRecord]:
        """
        Step3 过滤口径：
        - 真硬过滤（hard delete）：只删除明显垃圾/空壳候选，删除是安全的；
        - 软标记（soft flag only）：边界/偏弱/高风险但仍可能有训练对比价值的候选，不在这里删除，
          仅写入 risk_flags / sampleability_flags，保留进候选中间层。
        """
        out = []
        for r in records:
            hard_reasons = []
            # --- A. 真硬过滤：明显垃圾/空壳 ---
            # 1) 纯协同来源且缺少任何主题/证据支撑：更像噪声扩散，不适合作为候选中间层成员
            if r.from_collab and r.path_count == 1 and not r.from_label and not r.from_vector:
                hard_reasons.append("collab_only_no_topic")
            # 2) 无论文/无指标，且也不是 label/vector 支撑：极端数据残缺
            if (r.works_count is None or r.works_count == 0) and not r.from_label and not r.from_vector:
                hard_reasons.append("no_paper_no_metrics")

            if hard_reasons:
                r.passed_hard_filter = False
                r.hard_filter_reasons = hard_reasons
                summary.hard_filtered_count += 1
                continue

            # --- B. 软标记：边界候选保留，仅写 risk_flags / sampleability_flags ---
            # 仅对 label_only 场景保守打标（沿用旧 reason 的原词，便于日志对照）
            if r.from_label and r.path_count == 1:
                if r.domain_consistency is not None and r.domain_consistency < 0.25:
                    if "label_only_low_domain_consistency" not in r.risk_flags:
                        r.risk_flags.append("label_only_low_domain_consistency")
                if r.paper_hit_strength is not None and r.paper_hit_strength < 0.15:
                    if "label_only_weak_paper_hit" not in r.risk_flags:
                        r.risk_flags.append("label_only_weak_paper_hit")
                if r.recent_activity_match is not None and r.recent_activity_match < 0.20:
                    if "label_only_stale_activity" not in r.risk_flags:
                        r.risk_flags.append("label_only_stale_activity")
                if r.label_risky_term_count >= 2 and r.label_core_term_count == 0:
                    if "label_only_risky_terms_high" not in r.risk_flags:
                        r.risk_flags.append("label_only_risky_terms_high")

                # 若存在任一风险标记，作为“边界候选”预留给后续样本构造（不等同训练标签）
                if r.risk_flags and "borderline_candidate" not in r.sampleability_flags:
                    r.sampleability_flags.append("borderline_candidate")

            r.passed_hard_filter = True
            out.append(r)
        return out

    def _assign_buckets(self, records: List[CandidateRecord], summary: PoolDebugSummary) -> None:
        """分桶 A～F：便于截断与精排；A=vector+label, B=vector+collab, C=vector_only, D=label+collab, E=label_only, F=collab_only。"""
        for r in records:
            if r.from_vector and r.from_label:
                r.bucket_type = "A"
                r.bucket_reasons = "vector+label"
                summary.bucket_a_count += 1
            elif r.from_vector and r.from_collab:
                r.bucket_type = "B"
                r.bucket_reasons = "vector+collab"
                summary.bucket_b_count += 1
            elif r.from_vector:
                r.bucket_type = "C"
                r.bucket_reasons = "vector_only"
                summary.bucket_c_count += 1
            elif r.from_label and r.from_collab:
                r.bucket_type = "D"
                r.bucket_reasons = "label+collab"
                summary.bucket_d_count += 1
            elif r.from_label:
                r.bucket_type = "E"
                r.bucket_reasons = "label_only"
                summary.bucket_e_count += 1
            elif r.from_collab:
                r.bucket_type = "F"
                r.bucket_reasons = "collab_only"
                summary.bucket_f_count += 1
            else:
                r.bucket_type = "Z"
                r.bucket_reasons = "unknown"
        summary.final_pool_size = len(records)

    def execute(
        self,
        query_text,
        domain_id=None,
        is_training=False,
        vector_evidence_bonus_config: Optional[VectorEvidenceBonusConfig] = None,
    ):
        """
        执行多路召回：编码 → 三路召回 → build_candidate_records → score_candidate_pool
        → _enrich_candidate_features → _apply_hard_filters → _assign_buckets
        → base / training / display 双池骨架 → CandidatePool。

        返回除 candidate_pool、kgatax_sidecar_rows 等外，含 training_pool_top_100、display_pool_top_50
        及对应 count 字段；final_top_200 / final_top_500 为 deprecated，仅兼容旧调用，语义同展示池列表。
        """
        start_time = time.time()
        raw_vec, _ = self.encoder.encode(query_text)
        faiss.normalize_L2(raw_vec)

        user_domain = None
        if domain_id and str(domain_id).strip() not in ["0", ""]:
            user_domain = str(domain_id).strip()

        active_domains, applied_domain_str, domain_debug = self.domain_detector.detect(
            raw_vec,
            query_text=query_text,
            user_domain=user_domain,
        )

        vector_domains = None
        if active_domains:
            vector_domains = "|".join(sorted(active_domains))

        final_query = query_text
        if vector_domains and not is_training:
            domain_set = DomainProcessor.to_set(vector_domains)
            if domain_set:
                primary_domain = sorted(domain_set)[0]
                if primary_domain in self.DOMAIN_PROMPTS:
                    bias = self.DOMAIN_PROMPTS[primary_domain]
                    final_query = f"{query_text} | Area: {bias}"

        # 训练模式跳过领域 prompt，且多数情况下 final_query==query_text：复用已归一化的 raw_vec，避免二次 encode
        if final_query == query_text:
            query_vec = raw_vec
        else:
            query_vec, _ = self.encoder.encode(final_query)
            faiss.normalize_L2(query_vec)

        # 传入 query_text 以便向量路构建完整 vector_evidence（Step5）；不改变三路职责与主排序
        future_v = self.executor.submit(
            self.v_path.recall,
            query_vec,
            target_domains=vector_domains,
            verbose=False,
            query_text=query_text,
        )
        future_l = self.executor.submit(
            self.l_path.recall,
            query_vec,
            domain_id=user_domain,
            query_text=query_text,
            semantic_query_text=query_text,
        )
        v_result, v_cost = future_v.result()
        l_result, l_cost = future_l.result()

        v_meta = _ensure_meta_list(v_result, "vector")
        l_meta = _ensure_meta_list(l_result, "label")
        summary = PoolDebugSummary()
        summary.v_raw_count = len(v_meta)
        summary.l_raw_count = len(l_meta)

        seeds = list(set([m.get("author_id") for m in v_meta[:100]] + [m.get("author_id") for m in l_meta[:100]]))
        c_result, c_cost = self.c_path.recall(seeds)
        c_meta = _ensure_meta_list(c_result, "collab")
        summary.c_raw_count = len(c_meta)

        records = self.build_candidate_records(v_meta, l_meta, c_meta, summary)
        bonus_cfg = (
            vector_evidence_bonus_config
            if vector_evidence_bonus_config is not None
            else self.vector_evidence_bonus_config
        )
        records = self.score_candidate_pool(records, bonus_cfg=bonus_cfg)
        self._enrich_candidate_features(records, active_domains, query_text)
        records = self._apply_hard_filters(records, summary)
        self._assign_buckets(records, summary)
        # 候选中间层：三路合并 + 打分 + enrich + 硬过滤 + 分桶之后，尚未做分桶配额截断
        base_records = records
        for r in base_records:
            # Step2：训练中间层友好字段；非真值/非标签，仅标记该记录当前“被 base 视角引用”
            r.pool_role = "base"
        # Step3：软标记统计（仅统计保留下来的候选中间层成员）
        summary.soft_flagged_count = sum(1 for r in base_records if getattr(r, "risk_flags", None))
        # Step5：训练池正式分叉（轻量补位策略，服务训练样本构造；不追求榜单观感）
        training_records = self._build_training_pool_records(base_records, top_n=TRAINING_POOL_TOP_N)
        # 展示池：沿用既有 bucket_quota_truncate 实现，仅将截断上限改为 DISPLAY_POOL_TOP_N
        display_records = bucket_quota_truncate(base_records, BUCKET_QUOTAS, DISPLAY_POOL_TOP_N)
        # Step2：同一作者可能同时出现在 training 与 display；为避免 pool_role 单字段被覆盖，这里为不同池视角构造轻量副本
        training_records_view = [copy.copy(r) for r in training_records]
        for r in training_records_view:
            r.pool_role = "training"
        display_records_view = [copy.copy(r) for r in display_records]
        for r in display_records_view:
            r.pool_role = "display"

        training_pool_top_100 = [r.author_id for r in training_records]
        display_pool_top_50 = [r.author_id for r in display_records]
        training_candidate_records_count = len(training_records)
        display_candidate_records_count = len(display_records)
        # Step2：池级 summary 的三层规模（soft_flagged_count 留给 Step3）
        summary.base_pool_size = len(base_records)
        summary.training_pool_size = len(training_records)
        summary.display_pool_size = len(display_records)

        # Step6/8：池内审计；范围对齐「训练池 ∪ 展示池」涉及记录，避免仅扫 display 导致统计过窄
        audit_records = list(training_records)
        seen_audit = {r.author_id for r in audit_records}
        for r in display_records:
            if r.author_id not in seen_audit:
                seen_audit.add(r.author_id)
                audit_records.append(r)

        # Step6/8：最终池内审计（bonus 已在 score_candidate_pool 写入；raw 统计用于观察门控剥掉的量）
        ve_attached = sum(1 for r in audit_records if getattr(r, "vector_evidence", None))
        ve_summary_nz = 0
        bonus_vals = [float(getattr(r, "vector_evidence_bonus", 0.0) or 0.0) for r in audit_records]
        ve_bonus_nz = sum(1 for x in bonus_vals if x > 1e-9)
        ve_bonus_avg = float(sum(bonus_vals)) / max(1, len(bonus_vals))
        ve_bonus_max = max(bonus_vals) if bonus_vals else 0.0
        vo_cnt = 0
        raw_nz = 0
        for r in audit_records:
            srow = extract_vector_evidence_summary(getattr(r, "vector_evidence", None))
            if srow.get("vector_evidence_best_paper_score", 0.0) > 0.0:
                ve_summary_nz += 1
            if candidate_has_vector_origin(r):
                vo_cnt += 1
            if bonus_cfg.enabled:
                rw = compute_vector_evidence_bonus_from_summary(
                    srow,
                    cap=bonus_cfg.effective_cap(),
                    scale=bonus_cfg.scale,
                )
                if rw > 1e-9:
                    raw_nz += 1
        summary.vector_evidence_attached_count = ve_attached
        summary.vector_evidence_summary_nonzero_count = ve_summary_nz
        summary.vector_evidence_bonus_nonzero_count = ve_bonus_nz
        summary.vector_evidence_bonus_avg = ve_bonus_avg
        summary.vector_evidence_bonus_max = ve_bonus_max
        summary.vector_origin_candidate_count = vo_cnt
        summary.vector_evidence_bonus_raw_nonzero_count = raw_nz
        print(
            f"[TotalRecall] vector_evidence: attached={ve_attached}/{len(audit_records)} "
            f"summary_nonzero={ve_summary_nz}",
            flush=True,
        )
        print(
            f"[TotalRecall] vector bonus gate: origin={vo_cnt} raw_nz={raw_nz} applied_nz={ve_bonus_nz} "
            f"enabled={bonus_cfg.enabled} gate={bonus_cfg.gate_vector_origin} "
            f"cap={bonus_cfg.effective_cap():.4f} scale={bonus_cfg.scale} "
            f"avg={ve_bonus_avg:.4f} max={ve_bonus_max:.4f}",
            flush=True,
        )

        kgatax_rows = [build_kgatax_feature_row(r) for r in base_records]
        evidence_rows = []
        for r in base_records:
            if r.vector_evidence:
                evidence_rows.append({"author_id": r.author_id, "path": "vector", "evidence": r.vector_evidence})
            if r.label_evidence:
                evidence_rows.append({"author_id": r.author_id, "path": "label", "evidence": r.label_evidence})
            if r.collab_evidence:
                evidence_rows.append({"author_id": r.author_id, "path": "collab", "evidence": r.collab_evidence})

        pool = CandidatePool(
            query_text=query_text,
            applied_domains=applied_domain_str,
            candidate_records=base_records,
            training_pool_records=training_records_view,
            display_pool_records=display_records_view,
            candidate_evidence_rows=evidence_rows,
            pool_debug_summary=summary,
            path_costs={"v_cost": v_cost, "l_cost": l_cost, "cost_c": c_cost},
            domain_debug=domain_debug,
        )

        # deprecated / compatibility only：与 display_pool_top_50 同源，不代表真实 200/500 规模
        compat_display_ids = display_pool_top_50
        rank_map = {}
        for r in base_records:
            rank_map[r.author_id] = {
                "v": r.vector_rank if r.vector_rank is not None else "-",
                "l": r.label_rank if r.label_rank is not None else "-",
                "c": r.collab_rank if r.collab_rank is not None else "-",
            }

        return {
            "candidate_pool": pool,
            "kgatax_sidecar_rows": kgatax_rows,
            "training_pool_top_100": training_pool_top_100,
            "display_pool_top_50": display_pool_top_50,
            "training_candidate_records_count": training_candidate_records_count,
            "display_candidate_records_count": display_candidate_records_count,
            # deprecated / compatibility only：与 display_pool_top_50 一致，勿理解为 200/500 人规模
            "final_top_200": compat_display_ids,
            "final_top_500": compat_display_ids,
            "rank_map": rank_map,
            "total_ms": (time.time() - start_time) * 1000,
            "applied_domains": applied_domain_str,
            "details": {
                "v_cost": v_cost,
                "l_cost": l_cost,
                "cost_c": c_cost,
                "domain_debug": domain_debug,
                "vector_evidence_attached_count": ve_attached,
                "vector_evidence_summary_nonzero_count": ve_summary_nz,
                "vector_evidence_bonus_nonzero_count": ve_bonus_nz,
                "vector_evidence_bonus_avg": ve_bonus_avg,
                "vector_evidence_bonus_max": ve_bonus_max,
                "vector_origin_candidate_count": vo_cnt,
                "vector_evidence_bonus_raw_nonzero_count": raw_nz,
                "vector_bonus_enabled": bonus_cfg.enabled,
                "vector_bonus_gate_vector_origin": bonus_cfg.gate_vector_origin,
                "vector_bonus_cap_effective": bonus_cfg.effective_cap(),
                "vector_bonus_scale": bonus_cfg.scale,
            },
        }


def run_step8_vector_bonus_ablation(
    system: TotalRecallSystem,
    query_text: str,
    domain_id: Any = None,
    is_training: bool = False,
) -> Dict[str, Any]:
    """
    Step8 最小消融：同一 system 下多次 execute，对比 bonus 关 / gate off / gate on / 更紧 cap。
    仅用于本地验证，非通用实验平台；完整流水线会重复运行，耗时与单次总召回成正比。
    """
    configs: List[Tuple[str, VectorEvidenceBonusConfig]] = [
        ("no_bonus", VectorEvidenceBonusConfig(enabled=False, gate_vector_origin=False)),
        ("bonus_gate_off", VectorEvidenceBonusConfig(enabled=True, gate_vector_origin=False)),
        ("bonus_gate_on", VectorEvidenceBonusConfig(enabled=True, gate_vector_origin=True)),
        (
            "bonus_gate_on_half_scale",
            # 实测 bonus 常低于 0.036，单独紧 cap 不易与 gate_on 区分；用 scale=0.5 做保守校准消融更可见
            VectorEvidenceBonusConfig(enabled=True, gate_vector_origin=True, cap=0.048, scale=0.5),
        ),
    ]
    runs: List[Dict[str, Any]] = []
    print("[TotalRecall] Step8 ablation: start", flush=True)
    for name, cfg in configs:
        out = system.execute(
            query_text,
            domain_id,
            is_training=is_training,
            vector_evidence_bonus_config=cfg,
        )
        recs = out["candidate_pool"].candidate_records
        top5 = [r.author_id for r in recs[:5]]
        mix = {
            "vector_only": sum(1 for r in recs[:20] if r.from_vector and not r.from_label and not r.from_collab),
            "label_only": sum(1 for r in recs[:20] if r.from_label and not r.from_vector and not r.from_collab),
            "multi": sum(1 for r in recs[:20] if (r.from_vector + r.from_label + r.from_collab) >= 2),
        }
        nz_bonus_in_top20 = sum(1 for r in recs[:20] if (r.vector_evidence_bonus or 0) > 1e-9)
        d = out["details"]
        runs.append(
            {
                "name": name,
                "top5": top5,
                "top20_mix": mix,
                "top20_nonzero_bonus": nz_bonus_in_top20,
                "vector_origin_candidate_count": d.get("vector_origin_candidate_count"),
                "vector_evidence_bonus_raw_nonzero_count": d.get("vector_evidence_bonus_raw_nonzero_count"),
                "vector_evidence_bonus_nonzero_count": d.get("vector_evidence_bonus_nonzero_count"),
                "vector_evidence_bonus_avg": round(float(d.get("vector_evidence_bonus_avg", 0)), 6),
                "vector_evidence_bonus_max": round(float(d.get("vector_evidence_bonus_max", 0)), 6),
            }
        )
        print(
            f"[TotalRecall] Step8 ablation row: {name} raw_nz={d.get('vector_evidence_bonus_raw_nonzero_count')} "
            f"applied_nz={d.get('vector_evidence_bonus_nonzero_count')} "
            f"avg={d.get('vector_evidence_bonus_avg')} max={d.get('vector_evidence_bonus_max')} top5={top5}",
            flush=True,
        )
    print("[TotalRecall] Step8 ablation: done", flush=True)
    return {"query_text": query_text[:120], "runs": runs}


if __name__ == "__main__":
    system = TotalRecallSystem()
    fields = {
        "1": "计算机科学", "2": "医学", "3": "政治学", "4": "工程学", "5": "物理学",
        "6": "材料科学", "7": "生物学", "8": "地理学", "9": "化学", "10": "商学",
        "11": "社会学", "12": "哲学", "13": "环境科学", "14": "数学", "15": "心理学",
        "16": "地质学", "17": "经济学"
    }

    print("\n" + "=" * 115)
    # Windows 控制台常为 GBK 编码，避免 emoji 触发 UnicodeEncodeError
    print("[TotalRecall] 人才推荐系统 - 全量召回集成版")
    print("-" * 115)
    f_list = list(fields.items())
    for i in range(0, len(f_list), 6):
        print(" | ".join([f"{k}:{v}" for k, v in f_list[i:i + 6]]))
    print("=" * 115)

    try:
        domain_choice = input("\n请选择领域编号 (1-17, 0或回车跳过): ").strip()
        current_field = fields.get(domain_choice, "全领域")
        print(f"[*] 当前系统环境：{current_field} {'(硬过滤已激活)' if domain_choice in fields else '(全领域广度搜索)'}")

        while True:
            user_input = input(f"\n[{current_field}] 请输入岗位需求或技术关键词 (q退出): ").strip()
            if not user_input or user_input.lower() == 'q':
                break

            results = system.execute(user_input, domain_id=domain_choice)
            candidates = results['final_top_200']
            rank_map = results['rank_map']
            pool = results.get("candidate_pool")
            if pool and pool.pool_debug_summary:
                s = pool.pool_debug_summary
                print(f"\n[候选池统计] 去重后={s.after_dedup_count} 硬过滤掉={s.hard_filtered_count} 最终={s.final_pool_size} A={s.bucket_a_count} B={s.bucket_b_count} C={s.bucket_c_count} D={s.bucket_d_count}")

            print(f"\n[召回报告] 耗时: {results['total_ms']:.2f}ms | V={results['details']['v_cost']:.1f}ms L={results['details']['l_cost']:.1f}ms C={results['details']['cost_c']:.1f}ms")
            print("-" * 115)
            print(f"{'综合排名':<6} | {'作者 ID':<10} | {'各路名次 (V/L/C)':<15} | {'知识图谱核心作 (权重)'}")
            print("-" * 115)

            for rank, aid in enumerate(candidates[:50], 1):
                rm = rank_map.get(aid, {'v': '-', 'l': '-', 'c': '-'})
                v_rank = str(rm['v']) if rm.get('v') != '-' else "-"
                l_rank = str(rm['l']) if rm.get('l') != '-' else "-"
                c_rank = str(rm['c']) if rm.get('c') != '-' else "-"
                path_ranks = f"V:{v_rank:>3} L:{l_rank:>3} C:{c_rank:>3}"
                works = system._get_author_works(aid, top_n=1)
                if works:
                    work_title = works[0]['title']
                    if len(work_title) > 60:
                        work_title = work_title[:57] + "..."
                    info = f"《{work_title}》({works[0]['weight']:.3f})"
                else:
                    info = "暂无图谱论文数据"
                print(f"#{rank:<5} | {aid:<10} | {path_ranks:<15} | {info}")

            print("-" * 115)
            print(f"[*] 已召回 {len(candidates)} 名候选人，上方显示前 50 名综合最优解。")

    except KeyboardInterrupt:
        print("\n[!] 系统安全退出。")
