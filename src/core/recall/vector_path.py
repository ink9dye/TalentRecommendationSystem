import faiss

import json

import sqlite3

import time

from collections import defaultdict

from typing import Any, Dict, List, Optional, Tuple



import numpy as np



from config import ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH, DB_PATH, JOB_INDEX_PATH, JOB_MAP_PATH

from src.utils.domain_utils import DomainProcessor

from src.utils.tools import apply_text_decay, get_decay_rate_for_domains

from src.utils.time_features import compute_paper_recency, compute_author_time_features

from src.core.recall.works_to_authors import accumulate_author_scores

from src.core.recall.vector_query_builder import build_vector_query_bundle, format_query_bundle_summary



# SQLite 单条语句绑定参数上限（常见编译为 999）；IN 列表过长会触发 OperationalError: too many SQL variables

_SQLITE_MAX_VARS_PER_QUERY = 900



# Step2：多路 dense 融合权重（raw 主路；其余为补充）。分支未启用时不计入分母，避免单路时整体尺度被压低。

_MULTI_QUERY_FUSION_WEIGHTS: Dict[str, float] = {

    "raw_query": 0.55,

    "compressed_query": 0.20,

    "task_focused_query": 0.15,

    "method_focused_query": 0.10,

}



# 与 query_bundle 字段一致；clause 仅预留，本轮不检索

_MULTI_QUERY_BRANCH_KEYS: Tuple[str, ...] = (

    "raw_query",

    "compressed_query",

    "task_focused_query",

    "method_focused_query",

)



# paper record 中各分支得分字段名（便于 debug 与后续扩展）

_SCORE_FIELD_BY_BRANCH: Dict[str, str] = {

    "raw_query": "score_raw_query",

    "compressed_query": "score_compressed_query",

    "task_focused_query": "score_task_query",

    "method_focused_query": "score_method_query",

}





def _faiss_dist_to_similarity(index: Any, x: float) -> float:

    """与旧版 vector_path 一致：IP 越大越好；L2 越小越好 → 转成 (0,1] 相似度。"""

    mt = getattr(index, "metric_type", None)

    if mt == faiss.METRIC_L2:

        return 1.0 / (1.0 + max(0.0, x))

    return float(x)





class VectorPath:

    """

    向量路召回：实现基于 SBERT 的语义召回（Step2：内部 multi-query retrieval + paper 层保守融合）

    """



    def __init__(self, recall_limit=300):

        self.search_k = 500  # 检索深度

        self.recall_limit = recall_limit



        # 1. 加载论文摘要索引

        self.index = faiss.read_index(ABSTRACT_INDEX_PATH)

        with open(ABSTRACT_MAP_PATH, "r", encoding="utf-8") as f:

            self.id_map = json.load(f)



        # 2. 加载岗位描述索引

        self.job_index = faiss.read_index(JOB_INDEX_PATH)

        with open(JOB_MAP_PATH, "r", encoding="utf-8") as f:

            self.job_id_map = json.load(f)



        # 惰性加载：仅当需要编码 compressed/task/method 时初始化（total_recall 仅传向量时可零编码）

        self._query_encoder = None



    def _get_query_encoder(self):

        if self._query_encoder is None:

            from src.core.recall.input_to_vector import QueryEncoder



            self._query_encoder = QueryEncoder()

        return self._query_encoder



    def _build_query_bundle_for_recall(self, query_text: Optional[str]) -> Dict[str, Any]:

        """

        调用 Step1 builder；query_text 缺失时退化为空串 bundle，保证不抛错。

        multi-query 缓解长 JD 语义平均化：任务/方法视角独立检索，仍由向量模型吸收语义，而非标签式概念判决。

        """

        try:

            return build_vector_query_bundle(query_text or "")

        except Exception:

            return build_vector_query_bundle("")



    def _encode_query_bundle(

        self,

        query_bundle: Dict[str, Any],

        base_query_vector: np.ndarray,

        query_text: Optional[str],

    ) -> Tuple[Dict[str, np.ndarray], List[str]]:

        """

        为各分支准备向量：raw 复用调用方传入的 base_query_vector（与上游 encode(JD) 对齐，避免重复编码）。

        compressed / task / method 在文本非空且与 raw 文本不等价时再 encode。

        返回 (branch_key -> (1,dim) float32 L2 归一化向量, 实际参与编码的分支名列表)。

        """

        v0 = np.asarray(base_query_vector, dtype=np.float32)

        if v0.ndim == 1:

            v0 = v0.reshape(1, -1)

        faiss.normalize_L2(v0)



        vec_map: Dict[str, np.ndarray] = {"raw_query": v0.copy()}

        raw_txt = (query_bundle.get("raw_query") or "").strip()



        texts_to_encode: List[Tuple[str, str]] = []

        for key in ("compressed_query", "task_focused_query", "method_focused_query"):

            t = (query_bundle.get(key) or "").strip()

            if not t:

                continue

            if raw_txt and t == raw_txt:

                vec_map[key] = v0.copy()

                continue

            # 与已缓存分支同文复用（避免 task/compressed 相同再算一次）

            reused = False

            for uk, uvec in vec_map.items():

                if uk == key:

                    continue

                ut = (query_bundle.get(uk) if uk in query_bundle else "") or ""

                if isinstance(ut, str) and ut.strip() == t:

                    vec_map[key] = uvec.copy()

                    reused = True

                    break

            if reused:

                continue

            texts_to_encode.append((key, t))



        if texts_to_encode:

            enc = self._get_query_encoder()

            uniq: List[str] = []

            seen = set()

            for _, tx in texts_to_encode:

                if tx not in seen:

                    seen.add(tx)

                    uniq.append(tx)

            if uniq:

                batch = enc.encode_batch(uniq)

                text_to_row = {t: batch[i : i + 1].copy() for i, t in enumerate(uniq)}

                for key, tx in texts_to_encode:

                    row = text_to_row[tx]

                    faiss.normalize_L2(row)

                    vec_map[key] = row

        # 稳定顺序：按分支定义序输出 used_types

        ordered = [k for k in _MULTI_QUERY_BRANCH_KEYS if k in vec_map]

        return vec_map, ordered



    def _search_papers_for_each_query(

        self, query_vec_map: Dict[str, np.ndarray]

    ) -> Dict[str, Dict[str, float]]:

        """

        每路 query 单独 Faiss search_k；返回 query_type -> {wid: similarity}（同 wid 取该路最优位次得分，即首次出现）。

        若两分支共享同一向量（例如 compressed 与 raw 同文复用），只 search 一次再复制结果，避免重复计算。

        """

        out: Dict[str, Dict[str, float]] = {}

        # 按向量内容去重：浮点逐元素比较（L2 归一化后已稳定）
        unique_groups: List[Tuple[np.ndarray, List[str]]] = []

        for qtype, vec in query_vec_map.items():

            merged = False

            for v0, keys in unique_groups:

                if v0.shape == vec.shape and np.allclose(v0, vec, rtol=0.0, atol=1e-6):

                    keys.append(qtype)

                    merged = True

                    break

            if not merged:

                unique_groups.append((vec, [qtype]))

        for vec, qtypes in unique_groups:

            scores, indices = self.index.search(vec, self.search_k)

            wid_to_sim: Dict[str, float] = {}

            for i, idx in enumerate(indices[0]):

                if 0 <= idx < len(self.id_map):

                    wid = self.id_map[idx]

                    sim = _faiss_dist_to_similarity(self.index, float(scores[0][i]))

                    if wid not in wid_to_sim:

                        wid_to_sim[wid] = sim

            for qt in qtypes:

                out[qt] = wid_to_sim

        return out



    def _merge_multi_query_paper_hits(

        self, per_query_hits: Dict[str, Dict[str, float]], active_branches: List[str]

    ) -> Dict[str, Dict[str, Any]]:

        """

        按 wid 合并多路命中；每篇 paper 保留各分支得分、命中分支列表，供融合与审计。

        """

        records: Dict[str, Dict[str, Any]] = {}

        for qtype, wid_map in per_query_hits.items():

            sf = _SCORE_FIELD_BY_BRANCH.get(qtype)

            if not sf:

                continue

            for wid, sim in wid_map.items():

                if wid not in records:

                    records[wid] = {

                        "wid": wid,

                        "score_raw_query": None,

                        "score_compressed_query": None,

                        "score_task_query": None,

                        "score_method_query": None,

                        "hit_query_types": [],

                    }

                rec = records[wid]

                rec[sf] = float(sim)

                if qtype not in rec["hit_query_types"]:

                    rec["hit_query_types"].append(qtype)

        for rec in records.values():

            rec["hit_query_types"] = sorted(

                rec["hit_query_types"],

                key=lambda t: _MULTI_QUERY_BRANCH_KEYS.index(t) if t in _MULTI_QUERY_BRANCH_KEYS else 99,

            )

        for wid, rec in records.items():

            vals = []

            for q in active_branches:

                sf = _SCORE_FIELD_BY_BRANCH[q]

                v = rec.get(sf)

                if v is not None:

                    vals.append(v)

            rec["best_dense_score"] = max(vals) if vals else 0.0

            rec["query_hit_count"] = len(rec["hit_query_types"])

            rec["fused_dense_score"] = self._score_multi_query_paper_record(rec, active_branches)

        return records



    def _score_multi_query_paper_record(self, rec: Dict[str, Any], active_branches: List[str]) -> float:

        """

        保守融合：对**已启用**分支做加权平均（分母为启用分支权重和），避免未启用分支稀释 raw。

        多路同时命中给予温和 bonus，上限 0.05，防止 method 单分支压过 raw 强相关。

        """

        wsum = sum(_MULTI_QUERY_FUSION_WEIGHTS[q] for q in active_branches if q in _MULTI_QUERY_FUSION_WEIGHTS)

        if wsum <= 1e-9:

            return 0.0

        num = 0.0

        for q in active_branches:

            sf = _SCORE_FIELD_BY_BRANCH[q]

            s = rec.get(sf)

            if s is None:

                s = 0.0

            num += _MULTI_QUERY_FUSION_WEIGHTS[q] * float(s)

        base = num / wsum

        hc = int(rec.get("query_hit_count") or 0)

        bonus = min(0.05, 0.012 * max(0, hc - 1))

        return float(min(1.0, base + bonus))



    def recall(self, query_vector, target_domains=None, verbose=False, query_text=None):

        """

        向量路召回：实现基于领域 ID 的论文级硬过滤

        :param query_vector: 输入向量（与上游对 JD 的编码一致时，作为 raw_query 分支，避免重复 encode）

        :param target_domains: 领域 ID 列表或字符串，例如 ['1', '4'] 或 '1|4'

        :param verbose: 是否打印中间调试信息

        :param query_text: 原始 JD 文本（可选）；用于 Step1 bundle 与 Step2 多路编码；不传则仅 raw 分支有向量

        """

        start_t = time.time()

        conn = sqlite3.connect(DB_PATH)

        meta_list: List[Dict[str, Any]] = []



        query_bundle = self._build_query_bundle_for_recall(query_text)

        qb_summary = format_query_bundle_summary(query_bundle)

        clause_reserved = query_bundle.get("clause_queries") or []

        self._last_debug = {

            "query_bundle": query_bundle,

            "query_bundle_summary": qb_summary,

            "clause_queries_reserved_count": len(clause_reserved),

        }

        if verbose:

            print(f"[VectorPath] query_bundle: {qb_summary}")



        target_set = DomainProcessor.to_set(target_domains) if target_domains else None

        decay_rate = get_decay_rate_for_domains(target_set or [])

        purity_min = 1.0

        try:

            # --- Step2：multi-query retrieval → paper 层融合 → 得到 wid 序列与 dense 主分 ---

            query_vec_map, query_vector_types_used = self._encode_query_bundle(

                query_bundle, query_vector, query_text

            )

            active_branches = [k for k in _MULTI_QUERY_BRANCH_KEYS if k in query_vec_map]



            per_query_hits = self._search_papers_for_each_query(query_vec_map)

            per_query_hit_counts = {k: len(per_query_hits.get(k, {})) for k in query_vec_map.keys()}



            merged_records = self._merge_multi_query_paper_hits(per_query_hits, active_branches)

            merged_paper_candidate_count = len(merged_records)



            # 按融合分排序，取前 search_k 进入原主链（与旧版「单路 topK」带宽对齐）

            sorted_wids = sorted(

                merged_records.keys(),

                key=lambda w: merged_records[w].get("fused_dense_score", 0.0),

                reverse=True,

            )[: self.search_k]



            raw_work_ids = sorted_wids

            faiss_score_map: Dict[str, float] = {

                w: float(merged_records[w]["fused_dense_score"]) for w in raw_work_ids

            }



            merged_paper_top_preview: List[Dict[str, Any]] = []

            for w in raw_work_ids[:15]:

                r = merged_records.get(w) or {}

                merged_paper_top_preview.append(

                    {

                        "work_id": w,

                        "fused_dense_score": r.get("fused_dense_score"),

                        "hit_query_types": r.get("hit_query_types"),

                        "best_dense_score": r.get("best_dense_score"),

                    }

                )



            self._last_debug.update(

                {

                    "per_query_hit_counts": per_query_hit_counts,

                    "merged_paper_candidate_count": merged_paper_candidate_count,

                    "merged_paper_top_preview": merged_paper_top_preview,

                    "query_vector_types_used": query_vector_types_used,

                    "multi_query_active_branches": active_branches,

                }

            )



            if verbose:

                print(

                    f"[VectorPath] multi-query: hits_per_branch={per_query_hit_counts} | "

                    f"merged_unique_papers={merged_paper_candidate_count} | "

                    f"branches_used={query_vector_types_used}"

                )

                prev_titles = []

                for item in merged_paper_top_preview[:5]:

                    prev_titles.append(f"{item['work_id']}{item.get('hit_query_types')}")

                print(f"[VectorPath] merged_top5_wid+hit_types: {prev_titles}")



            if not raw_work_ids:

                return [], (time.time() - start_t) * 1000



            # --- 步骤 2: 获取论文的领域标签与元信息用于过滤与调试 ---

            placeholders = ",".join(["?"] * len(raw_work_ids))

            sql = f"SELECT work_id, domain_ids, title, year FROM works WHERE work_id IN ({placeholders})"

            work_data = conn.execute(sql, raw_work_ids).fetchall()

            domain_dict = {row[0]: row[1] for row in work_data}

            meta_dict = {row[0]: {"title": row[2], "year": row[3]} for row in work_data}



            # --- 步骤 3: 领域硬过滤（只有对应领域的论文才能发挥作用） ---

            filtered_work_ids = []

            work_score_map = {}

            work_debug_map = {}

            for wid in raw_work_ids:

                if wid not in domain_dict:

                    continue



                if target_set:

                    work_domains_raw = domain_dict[wid]



                    if not DomainProcessor.has_intersect(work_domains_raw, target_set):

                        continue



                    paper_set = DomainProcessor.to_set(work_domains_raw)

                    purity = len(paper_set & target_set) / max(1, len(paper_set))

                    if purity < purity_min:

                        continue

                    domain_coeff = purity**4

                else:

                    domain_coeff = 1.0

                    purity = None



                filtered_work_ids.append(wid)

                base_sim = faiss_score_map.get(wid, 0.0)

                title = (meta_dict.get(wid, {}).get("title") or "")

                year_val = meta_dict.get(wid, {}).get("year")



                type_decay = apply_text_decay(title)

                time_decay = compute_paper_recency(year_val, target_set or [])



                work_score = base_sim * domain_coeff * type_decay * time_decay

                work_score_map[wid] = work_score



                if verbose:

                    mq = merged_records.get(wid, {})

                    work_debug_map[wid] = {

                        "faiss_sim": float(base_sim),

                        "multi_query_fused_dense": float(mq.get("fused_dense_score", base_sim)),

                        "hit_query_types": mq.get("hit_query_types"),

                        "purity": None if purity is None else float(purity),

                        "domain_coeff": float(domain_coeff),

                        "work_score": float(work_score),

                        "domain_ids": domain_dict.get(wid),

                        "title": title,

                        "year": year_val,

                        "type_decay": float(type_decay),

                        "time_decay": float(time_decay),

                    }



            if not filtered_work_ids:

                return [], (time.time() - start_t) * 1000



            # --- 步骤 4: 统一走“论文 → 作者”分摊聚合逻辑（不重写） ---

            work_placeholders = ",".join(["?"] * len(filtered_work_ids))

            pairs_query = f"""

                SELECT author_id, work_id

                FROM authorships

                WHERE work_id IN ({work_placeholders})

            """

            rows = conn.execute(pairs_query, filtered_work_ids).fetchall()



            papers_by_wid = {

                wid: {"wid": wid, "score": float(work_score_map.get(wid, 0.0)), "authors": []}

                for wid in filtered_work_ids

                if wid in work_score_map

            }



            for aid, wid in rows:

                p = papers_by_wid.get(wid)

                if p is None:

                    continue

                p["authors"].append(

                    {

                        "aid": str(aid),

                        "pos_weight": 1.0,

                    }

                )



            papers = [p for p in papers_by_wid.values() if p["authors"]]



            agg_result = accumulate_author_scores(papers, top_k_per_author=3)

            author_scores = agg_result.author_scores



            author_ids = agg_result.sorted_authors()

            if author_ids:

                year_rows = []

                for off in range(0, len(author_ids), _SQLITE_MAX_VARS_PER_QUERY):

                    batch = author_ids[off : off + _SQLITE_MAX_VARS_PER_QUERY]

                    ph = ",".join(["?"] * len(batch))

                    year_rows.extend(

                        conn.execute(

                            f"""

                            SELECT a.author_id, w.year

                            FROM authorships a

                            JOIN works w ON a.work_id = w.work_id

                            WHERE a.author_id IN ({ph})

                            """,

                            batch,

                        ).fetchall()

                    )



                years_by_author = defaultdict(list)

                for aid, year in year_rows:

                    years_by_author[str(aid)].append(year)



                for aid in author_ids:

                    base_score = float(author_scores.get(aid, 0.0))

                    years = years_by_author.get(str(aid), [])

                    _, _, time_weight = compute_author_time_features(years)

                    author_scores[aid] = base_score * float(time_weight)



            author_ids = [aid for aid, _ in sorted(author_scores.items(), key=lambda x: x[1], reverse=True)]

            meta_list = [

                {

                    "author_id": str(aid),

                    "vector_score_raw": float(author_scores.get(aid, 0.0)),

                    "vector_rank": i + 1,

                    "vector_evidence": None,

                }

                for i, aid in enumerate(author_ids[: self.recall_limit])

            ]



            if verbose:

                top20_items = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)[:20]



                top_work_ids = []

                author_top3 = {}

                for aid, _ in top20_items:

                    works = agg_result.author_top_works.get(aid, [])[:3]

                    author_top3[aid] = works

                    for wid, _ in works:

                        top_work_ids.append(wid)

                top_work_ids = list(dict.fromkeys(top_work_ids))



                work_meta = {}

                if top_work_ids:

                    ph = ",".join(["?"] * len(top_work_ids))

                    rows = conn.execute(

                        f"SELECT work_id, title, year FROM works WHERE work_id IN ({ph})",

                        top_work_ids,

                    ).fetchall()

                    work_meta = {r[0]: {"title": r[1], "year": r[2]} for r in rows}



                self._last_debug.update(

                    {

                        "query_bundle": query_bundle,

                        "query_bundle_summary": qb_summary,

                        "target_domains": target_domains,

                        "target_set": sorted(list(target_set)) if target_set else [],

                        "purity_min": purity_min,

                        "decay_rate": decay_rate,

                        "top20": [

                            {

                                "author_id": aid,

                                "author_score": float(a_score),

                                "top3_works": [

                                    {

                                        "work_id": wid,

                                        **(work_meta.get(wid, {})),

                                        **(work_debug_map.get(wid, {})),

                                    }

                                    for wid, _ in author_top3.get(aid, [])

                                ],

                            }

                            for aid, a_score in top20_items

                        ],

                    }

                )



        finally:

            conn.close()



        duration = (time.time() - start_t) * 1000

        if not meta_list:

            return [], duration

        return meta_list, duration





if __name__ == "__main__":

    from src.core.recall.input_to_vector import QueryEncoder  # 确保路径正确

    from src.core.recall.label_path import LabelRecallPath

    from src.utils.domain_detector import DomainDetector



    v_path = VectorPath(recall_limit=300)

    encoder = QueryEncoder()

    l_path = LabelRecallPath(recall_limit=150)

    d_detector = DomainDetector(l_path)



    fields = {

        "1": "计算机科学",

        "2": "医学",

        "3": "政治学",

        "4": "工程学",

        "5": "物理学",

        "6": "材料科学",

        "7": "生物学",

        "8": "地理学",

        "9": "化学",

        "10": "商学",

        "11": "社会学",

        "12": "哲学",

        "13": "环境科学",

        "14": "数学",

        "15": "心理学",

        "16": "地质学",

        "17": "经济学",

    }



    def get_work_title(author_id):

        conn = sqlite3.connect(DB_PATH)

        res = conn.execute(

            """

                           SELECT w.title

                           FROM works w

                                    JOIN authorships a ON w.work_id = a.work_id

                           WHERE a.author_id = ? LIMIT 1

                           """,

            (author_id,),

        ).fetchone()

        conn.close()

        return res[0] if res else "无论文数据"



    print("\n" + "=" * 115)

    print("🚀 向量路 (Vector Path) 独立语义召回测试")

    print("-" * 115)

    f_list = list(fields.items())

    for i in range(0, len(f_list), 6):

        print(" | ".join([f"{k}:{v}" for k, v in f_list[i : i + 6]]))

    print("=" * 115)



    try:

        domain_choice = input("\n请选择领域编号 (1-17, 0跳过): ").strip() or "0"

        current_field = fields.get(domain_choice, "全领域")



        while True:

            user_input = input(f"\n[{current_field}] 请输入岗位需求 (q退出): ").strip()

            if not user_input or user_input.lower() == "q":

                break



            query_vec, _ = encoder.encode(user_input)

            faiss.normalize_L2(query_vec)



            if domain_choice != "0":

                user_domain = domain_choice

            else:

                user_domain = None



            active_domains, applied_domains_str, _ = d_detector.detect(

                query_vec, query_text=user_input, user_domain=user_domain

            )

            target_domains = "|".join(sorted(active_domains)) if active_domains else None



            v_meta, duration = v_path.recall(

                query_vec, target_domains=target_domains, verbose=True, query_text=user_input

            )



            print(f"\n[召回报告] 耗时: {duration:.2f}ms | 命中人数: {len(v_meta)} | 应用领域: {applied_domains_str}")

            print("-" * 115)

            print(f"{'排名':<6} | {'作者 ID':<12} | {'检索路径':<15} | {'代表作标题 (数据源: SQLite)'}")

            print("-" * 115)



            for rank, item in enumerate(v_meta[:20], 1):

                aid = item["author_id"]

                title = get_work_title(aid)

                if len(title) > 70:

                    title = title[:67] + "..."

                print(f"#{rank:<5} | {aid:<12} | {'Vector (V)':<15} | {title}")



            dbg = getattr(v_path, "_last_debug", None)

            if dbg and dbg.get("top20"):

                print("\n[Top20 Debug] author_score 与 Top3 work 贡献明细")

                print("-" * 115)

                for i, item in enumerate(dbg["top20"], 1):

                    aid = item["author_id"]

                    a_score = item["author_score"]

                    print(f"#{i:<3} {aid} | author_score={a_score:.6f}")

                    for w in item.get("top3_works", []):

                        w_title = (w.get("title") or "N/A").strip()

                        if len(w_title) > 90:

                            w_title = w_title[:87] + "..."

                        print(

                            f"    - {w.get('work_id')} ({w.get('year')})"

                            f" | work_score={w.get('work_score', 0.0):.6f}"

                            f" | faiss={w.get('faiss_sim', 0.0):.6f}"

                            f" | purity={w.get('purity')}"

                            f" | domains={w.get('domain_ids')}"

                            f" | {w_title}"

                        )



            print("-" * 115)



    except KeyboardInterrupt:

        print("\n[!] 测试结束")

