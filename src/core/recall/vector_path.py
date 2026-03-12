import faiss
import json
import sqlite3
import time
import re
from datetime import datetime
from collections import defaultdict
from config import ABSTRACT_INDEX_PATH, ABSTRACT_MAP_PATH, DB_PATH, JOB_INDEX_PATH, JOB_MAP_PATH
from src.utils.domain_utils import DomainProcessor
from src.utils.tools import apply_text_decay, get_decay_rate_for_domains
from src.utils.time_features import compute_paper_recency, compute_author_time_features
from src.core.recall.works_to_authors import accumulate_author_scores

class VectorPath:
    """
    向量路召回：实现基于 SBERT 的语义召回
    """

    def __init__(self, recall_limit=300):
        self.search_k = 500  # 检索深度
        self.recall_limit = recall_limit

        # 1. 加载论文摘要索引
        self.index = faiss.read_index(ABSTRACT_INDEX_PATH)
        with open(ABSTRACT_MAP_PATH, 'r', encoding='utf-8') as f:
            self.id_map = json.load(f)

        # 2. 加载岗位描述索引
        self.job_index = faiss.read_index(JOB_INDEX_PATH)
        with open(JOB_MAP_PATH, 'r', encoding='utf-8') as f:
            self.job_id_map = json.load(f)

    def recall(self, query_vector, target_domains=None, verbose=False):
        """
        向量路召回：实现基于领域 ID 的论文级硬过滤
        :param query_vector: 输入向量
        :param target_domains: 领域 ID 列表或字符串，例如 ['1', '4'] 或 '1|4'
        :param verbose: 是否打印中间调试信息
        """
        start_t = time.time()
        conn = sqlite3.connect(DB_PATH)
        # 供 CLI 调试：当 verbose=True 时写入
        self._last_debug = None

        # --- 【修改 1】：统一使用 DomainProcessor 处理 target_domains 格式 ---
        # 无论是 '1|4' 还是 ['1', '4']，统一转化为 set({'1', '4'})
        target_set = DomainProcessor.to_set(target_domains) if target_domains else None
        # 统一时间衰减策略：根据目标领域集合选取 decay_rate
        decay_rate = get_decay_rate_for_domains(target_set or [])
        purity_min = 1.0  # 领域纯度门槛
        current_year = datetime.now().year

        try:
            # --- 步骤 1: 语义检索相似论文 (Faiss 获取最相关的论文 ID 序列) ---
            scores, indices = self.index.search(query_vector, self.search_k)

            # 兼容不同 Faiss 度量：Inner Product(越大越相似) / L2(越小越相似)
            def _to_similarity(x: float) -> float:
                mt = getattr(self.index, "metric_type", None)
                if mt == faiss.METRIC_L2:
                    return 1.0 / (1.0 + max(0.0, x))
                return float(x)

            raw_work_ids = []
            faiss_score_map = {}
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.id_map):
                    wid = self.id_map[idx]
                    raw_work_ids.append(wid)
                    # 保留最靠前的那次分数
                    if wid not in faiss_score_map:
                        faiss_score_map[wid] = _to_similarity(float(scores[0][i]))

            if not raw_work_ids:
                return [], (time.time() - start_t) * 1000

            # --- 步骤 2: 获取论文的领域标签与元信息用于过滤与调试 ---
            placeholders = ','.join(['?'] * len(raw_work_ids))
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

                    # 1) 交集校验（无标签在 has_intersect 中会被拦截）
                    if not DomainProcessor.has_intersect(work_domains_raw, target_set):
                        continue

                    # 2) 纯度过滤：|intersect| / |paper_domains|
                    paper_set = DomainProcessor.to_set(work_domains_raw)
                    purity = len(paper_set & target_set) / max(1, len(paper_set))
                    if purity < purity_min:
                        continue
                    domain_coeff = purity ** 4  # 标签路同款“断崖式专精惩罚”
                else:
                    domain_coeff = 1.0
                    purity = None

                filtered_work_ids.append(wid)
                # work_score：语义相似度 × 领域专精 × 文献类型衰减 × 时间衰减
                base_sim = faiss_score_map.get(wid, 0.0)
                title = (meta_dict.get(wid, {}).get("title") or "")
                year_val = meta_dict.get(wid, {}).get("year")

                # 1) 文本类型衰减：survey/overview/review/handbook + data from:/dataset:/supplementary data
                type_decay = apply_text_decay(title)

                # 2) 时间衰减：统一通过 time_features.compute_paper_recency（阶梯式 bucket）
                time_decay = compute_paper_recency(year_val, target_set or [])

                work_score = base_sim * domain_coeff * type_decay * time_decay
                work_score_map[wid] = work_score

                if verbose:
                    work_debug_map[wid] = {
                        "faiss_sim": float(base_sim),
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

            # --- 步骤 4: 统一走“论文 → 作者”分摊聚合逻辑 ---
            # 构造每篇论文的作者列表（此处 SQLite authorships 若无 pos_weight 字段，则等价于 1/n 均分）
            work_placeholders = ",".join(["?"] * len(filtered_work_ids))
            pairs_query = f"""
                SELECT author_id, work_id
                FROM authorships
                WHERE work_id IN ({work_placeholders})
            """
            rows = conn.execute(pairs_query, filtered_work_ids).fetchall()

            # 1) 为每篇论文建立骨架
            papers_by_wid = {
                wid: {"wid": wid, "score": float(work_score_map.get(wid, 0.0)), "authors": []}
                for wid in filtered_work_ids
                if wid in work_score_map
            }

            # 2) 填充作者；SQLite 若无作者级权重，则默认 pos_weight=1.0，后续在同篇内做 1/n 均分
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

            # 3) 聚合为作者分数：每个作者取自己 Top3 论文贡献度之和（与原逻辑保持一致语义）
            agg_result = accumulate_author_scores(papers, top_k_per_author=3)
            author_scores = agg_result.author_scores

            # ---- 作者层时间特征：活跃度 + 动量（统一 time_features）----
            author_ids = agg_result.sorted_authors()
            if author_ids:
                # 为每个作者收集其所有论文年份，用于计算 author-level activity & momentum
                placeholders = ",".join(["?"] * len(author_ids))
                year_rows = conn.execute(
                    f"""
                    SELECT a.author_id, w.year
                    FROM authorships a
                    JOIN works w ON a.work_id = w.work_id
                    WHERE a.author_id IN ({placeholders})
                    """,
                    author_ids,
                ).fetchall()

                years_by_author = defaultdict(list)
                for aid, year in year_rows:
                    years_by_author[str(aid)].append(year)

                for aid in author_ids:
                    base_score = float(author_scores.get(aid, 0.0))
                    years = years_by_author.get(str(aid), [])
                    _, _, time_weight = compute_author_time_features(years)
                    author_scores[aid] = base_score * float(time_weight)

            # 重新按时间加权后的得分排序
            author_ids = [aid for aid, _ in sorted(author_scores.items(), key=lambda x: x[1], reverse=True)]

            if verbose:
                # 构造 Top20 调试信息：对齐原有 _last_debug 结构
                top20_items = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)[:20]

                # 收集 Top20 作者的 Top3 论文 ID
                top_work_ids = []
                author_top3 = {}
                for aid, _ in top20_items:
                    works = agg_result.author_top_works.get(aid, [])[:3]
                    author_top3[aid] = works
                    for wid, _ in works:
                        top_work_ids.append(wid)
                top_work_ids = list(dict.fromkeys(top_work_ids))  # 去重且保序

                work_meta = {}
                if top_work_ids:
                    ph = ",".join(["?"] * len(top_work_ids))
                    rows = conn.execute(
                        f"SELECT work_id, title, year FROM works WHERE work_id IN ({ph})",
                        top_work_ids,
                    ).fetchall()
                    work_meta = {r[0]: {"title": r[1], "year": r[2]} for r in rows}

                self._last_debug = {
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

        finally:
            conn.close()

        duration = (time.time() - start_t) * 1000
        return author_ids[:self.recall_limit], duration


# 修改 vector_path.py 最后的 if __name__ == "__main__": 部分
if __name__ == "__main__":
    from src.core.recall.input_to_vector import QueryEncoder  # 确保路径正确
    from src.core.recall.label_path import LabelRecallPath
    from src.utils.domain_detector import DomainDetector

    v_path = VectorPath(recall_limit=300)
    encoder = QueryEncoder()
    # 标签路 + 领域探测器：用于在 domain_choice=0 时自动推断目标领域
    l_path = LabelRecallPath(recall_limit=150)
    d_detector = DomainDetector(l_path)

    fields = {
        "1": "计算机科学", "2": "医学", "3": "政治学", "4": "工程学", "5": "物理学",
        "6": "材料科学", "7": "生物学", "8": "地理学", "9": "化学", "10": "商学",
        "11": "社会学", "12": "哲学", "13": "环境科学", "14": "数学", "15": "心理学",
        "16": "地质学", "17": "经济学"
    }


    def get_work_title(author_id):
        """简单从 SQLite 获取该作者的一篇论文标题用于展示"""
        conn = sqlite3.connect(DB_PATH)
        res = conn.execute("""
                           SELECT w.title
                           FROM works w
                                    JOIN authorships a ON w.work_id = a.work_id
                           WHERE a.author_id = ? LIMIT 1
                           """, (author_id,)).fetchone()
        conn.close()
        return res[0] if res else "无论文数据"


    print("\n" + "=" * 115)
    print("🚀 向量路 (Vector Path) 独立语义召回测试")
    print("-" * 115)
    f_list = list(fields.items())
    for i in range(0, len(f_list), 6):
        print(" | ".join([f"{k}:{v}" for k, v in f_list[i:i + 6]]))
    print("=" * 115)

    try:
        domain_choice = input("\n请选择领域编号 (1-17, 0跳过): ").strip() or "0"
        current_field = fields.get(domain_choice, "全领域")

        while True:
            user_input = input(f"\n[{current_field}] 请输入岗位需求 (q退出): ").strip()
            if not user_input or user_input.lower() == 'q':
                break

            # 1. 文本转向量（基础向量，用于向量检索与自动领域探测）
            query_vec, _ = encoder.encode(user_input)
            faiss.normalize_L2(query_vec)

            # 2. 领域口径：若用户选择 1-17 则使用用户领域，否则启用自动领域探测
            if domain_choice != "0":
                user_domain = domain_choice
            else:
                user_domain = None

            active_domains, applied_domains_str, _ = d_detector.detect(
                query_vec, query_text=user_input, user_domain=user_domain
            )
            target_domains = "|".join(sorted(active_domains)) if active_domains else None

            # 3. 执行召回（根据 target_domains 做领域硬过滤）
            author_ids, duration = v_path.recall(query_vec, target_domains=target_domains, verbose=True)

            # 3. 打印报告
            print(f"\n[召回报告] 耗时: {duration:.2f}ms | 命中人数: {len(author_ids)} | 应用领域: {applied_domains_str}")
            print("-" * 115)
            print(f"{'排名':<6} | {'作者 ID':<12} | {'检索路径':<15} | {'代表作标题 (数据源: SQLite)'}")
            print("-" * 115)

            for rank, aid in enumerate(author_ids[:20], 1):
                title = get_work_title(aid)
                if len(title) > 70: title = title[:67] + "..."
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