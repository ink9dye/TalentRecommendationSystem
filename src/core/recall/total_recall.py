# -*- coding: utf-8 -*-
"""
多路召回总控：统一编码、三路并行召回、候选池构建、打分、特征补全、硬过滤、分桶、导出。
"""
import time
import faiss
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

from src.core.recall.input_to_vector import QueryEncoder
from src.core.recall.vector_path import VectorPath
from src.core.recall.label_path import LabelRecallPath
from src.core.recall.collaboration_path import CollaborativeRecallPath
from src.core.recall.candidate_pool import (
    CandidateRecord,
    CandidatePool,
    PoolDebugSummary,
)
from src.utils.domain_detector import DomainDetector
from src.utils.domain_utils import DomainProcessor
from config import DB_PATH

# 限制底层模型并行，防止多线程冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# RRF 与配额默认值
RRF_K = 60
PATH_WEIGHTS = {"v": 2.0, "l": 3.0, "c": 1.0}
ALPHA_MULTI_PATH = 0.05
FINAL_POOL_TOP_N = 200


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

    def __init__(self, k_vector: int = 150, k_label: int = 150, k_collab: int = 80):
        print("[*] 正在初始化全量召回系统 (Training-Safe Mode)...", flush=True)
        self.encoder = QueryEncoder()
        self.v_path = VectorPath(recall_limit=k_vector)
        self.l_path = LabelRecallPath(recall_limit=k_label)
        self.c_path = CollaborativeRecallPath(recall_limit=k_collab)
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.domain_detector = DomainDetector(self.l_path)

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
            by_id[aid].from_label = True
            by_id[aid].label_rank = m.get("label_rank") or rank + 1
            by_id[aid].label_score_raw = m.get("label_score_raw")
            by_id[aid].label_evidence = m.get("label_evidence")
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

    def score_candidate_pool(self, records: List[CandidateRecord]) -> List[CandidateRecord]:
        """RRF、multi_path_bonus、candidate_pool_score，排序。"""
        scores = {}
        for r in records:
            s = 0.0
            if r.vector_rank is not None:
                s += PATH_WEIGHTS["v"] * (1.0 / (RRF_K + r.vector_rank))
            if r.label_rank is not None:
                s += PATH_WEIGHTS["l"] * (1.0 / (RRF_K + r.label_rank))
            if r.collab_rank is not None:
                s += PATH_WEIGHTS["c"] * (1.0 / (RRF_K + r.collab_rank))
            r.rrf_score = s
            bonus = ALPHA_MULTI_PATH * (r.path_count - 1) if r.path_count >= 1 else 0.0
            r.multi_path_bonus = bonus
            r.candidate_pool_score = r.rrf_score + bonus
            scores[r.author_id] = r.candidate_pool_score

        # dominant_recall_path
        for r in records:
            if r.path_count >= 3:
                r.dominant_recall_path = "multi"
            elif r.path_count == 2:
                if r.from_label and r.from_vector:
                    r.dominant_recall_path = "label+vector"
                elif r.from_label and r.from_collab:
                    r.dominant_recall_path = "label+collab"
                elif r.from_vector and r.from_collab:
                    r.dominant_recall_path = "vector+collab"
                else:
                    r.dominant_recall_path = "multi"
            else:
                if r.from_label:
                    r.dominant_recall_path = "label"
                elif r.from_vector:
                    r.dominant_recall_path = "vector"
                else:
                    r.dominant_recall_path = "collab"

        records.sort(key=lambda x: x.candidate_pool_score, reverse=True)
        return records

    def _enrich_candidate_features(
        self,
        records: List[CandidateRecord],
        active_domains: Optional[Any],
        query_text: str,
    ) -> None:
        """补全作者静态指标与 query-author 交叉特征（第一版粗算，缺的填 None）。"""
        if not records:
            return
        aids = [r.author_id for r in records]
        author_stats: Dict[str, Dict[str, Any]] = {}
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            ph = ",".join("?" * len(aids))
            rows = conn.execute(
                f"SELECT author_id, name, h_index, works_count, cited_by_count FROM authors WHERE author_id IN ({ph})",
                aids,
            ).fetchall()
            for row in rows:
                author_stats[str(row["author_id"])] = {
                    "name": row["name"],
                    "h_index": row["h_index"],
                    "works_count": row["works_count"],
                    "cited_by_count": row["cited_by_count"],
                }
            conn.close()
        except Exception:
            pass

        for r in records:
            st = author_stats.get(r.author_id) or {}
            r.author_name = st.get("name")
            r.h_index = float(st["h_index"]) if st.get("h_index") is not None else None
            r.works_count = int(st["works_count"]) if st.get("works_count") is not None else None
            r.cited_by_count = int(st["cited_by_count"]) if st.get("cited_by_count") is not None else None
            r.recent_works_count = None
            r.recent_citations = None
            r.institution_level = None
            r.top_work_quality = None
            r.topic_similarity = None
            r.skill_coverage_ratio = None
            r.domain_consistency = None
            r.paper_hit_strength = None
            r.recent_activity_match = None

    def _apply_hard_filters(
        self,
        records: List[CandidateRecord],
        summary: PoolDebugSummary,
    ) -> List[CandidateRecord]:
        """第一层轻硬过滤：仅协同无主题、无论文无指标。"""
        out = []
        for r in records:
            reasons = []
            if r.from_collab and r.path_count == 1 and not r.from_label and not r.from_vector:
                reasons.append("collab_only_no_topic")
            if (r.works_count is None or r.works_count == 0) and not r.from_label and not r.from_vector:
                reasons.append("no_paper_no_metrics")
            if reasons:
                r.passed_hard_filter = False
                r.hard_filter_reasons = reasons
                summary.hard_filtered_count += 1
                continue
            r.passed_hard_filter = True
            out.append(r)
        return out

    def _assign_buckets(self, records: List[CandidateRecord], summary: PoolDebugSummary) -> None:
        """第一版分桶：A/B/C/D 仅按 from_label、from_vector、from_collab、path_count。"""
        for r in records:
            if r.from_label and (r.path_count >= 2 or r.from_vector):
                r.bucket_type = "A"
                r.bucket_reasons = "label_multi_path_or_label_vector"
                summary.bucket_a_count += 1
            elif r.from_label:
                r.bucket_type = "B"
                r.bucket_reasons = "label_only"
                summary.bucket_b_count += 1
            elif r.from_vector:
                r.bucket_type = "C"
                r.bucket_reasons = "vector_only"
                summary.bucket_c_count += 1
            elif r.from_collab:
                r.bucket_type = "D"
                r.bucket_reasons = "collab_only"
                summary.bucket_d_count += 1
            else:
                r.bucket_type = "D"
                r.bucket_reasons = "unknown"
                summary.bucket_d_count += 1
        summary.final_pool_size = len(records)

    def execute(self, query_text, domain_id=None, is_training=False):
        """
        执行多路召回：编码 → 三路召回 → build_candidate_records → score_candidate_pool
        → _enrich_candidate_features → _apply_hard_filters → _assign_buckets → CandidatePool。
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
            user_domain=user_domain if not is_training else user_domain,
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

        query_vec, _ = self.encoder.encode(final_query)
        faiss.normalize_L2(query_vec)

        future_v = self.executor.submit(self.v_path.recall, query_vec, target_domains=vector_domains)
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
        records = self.score_candidate_pool(records)
        self._enrich_candidate_features(records, active_domains, query_text)
        records = self._apply_hard_filters(records, summary)
        self._assign_buckets(records, summary)

        evidence_rows = []
        for r in records:
            if r.vector_evidence:
                evidence_rows.append({"author_id": r.author_id, "path": "vector", "evidence": r.vector_evidence})
            if r.label_evidence:
                evidence_rows.append({"author_id": r.author_id, "path": "label", "evidence": r.label_evidence})
            if r.collab_evidence:
                evidence_rows.append({"author_id": r.author_id, "path": "collab", "evidence": r.collab_evidence})

        pool = CandidatePool(
            query_text=query_text,
            applied_domains=applied_domain_str,
            candidate_records=records,
            candidate_evidence_rows=evidence_rows,
            pool_debug_summary=summary,
            path_costs={"v_cost": v_cost, "l_cost": l_cost, "cost_c": c_cost},
            domain_debug=domain_debug,
        )

        final_list = [r.author_id for r in records[:FINAL_POOL_TOP_N]]
        rank_map = {}
        for r in records:
            rank_map[r.author_id] = {
                "v": r.vector_rank if r.vector_rank is not None else "-",
                "l": r.label_rank if r.label_rank is not None else "-",
                "c": r.collab_rank if r.collab_rank is not None else "-",
            }

        return {
            "candidate_pool": pool,
            "final_top_200": final_list,
            "final_top_500": final_list,
            "rank_map": rank_map,
            "total_ms": (time.time() - start_time) * 1000,
            "applied_domains": applied_domain_str,
            "details": {
                "v_cost": v_cost,
                "l_cost": l_cost,
                "cost_c": c_cost,
                "domain_debug": domain_debug,
            },
        }


if __name__ == "__main__":
    system = TotalRecallSystem()
    fields = {
        "1": "计算机科学", "2": "医学", "3": "政治学", "4": "工程学", "5": "物理学",
        "6": "材料科学", "7": "生物学", "8": "地理学", "9": "化学", "10": "商学",
        "11": "社会学", "12": "哲学", "13": "环境科学", "14": "数学", "15": "心理学",
        "16": "地质学", "17": "经济学"
    }

    print("\n" + "=" * 115)
    print("🚀 人才推荐系统 - 生产级全量召回集成版")
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
