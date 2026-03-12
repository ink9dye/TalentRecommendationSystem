import time
import faiss
import os
from concurrent.futures import ThreadPoolExecutor
from src.core.recall.input_to_vector import QueryEncoder
from src.core.recall.vector_path import VectorPath
from src.core.recall.label_path import LabelRecallPath
from src.core.recall.collaboration_path import CollaborativeRecallPath
from src.utils.domain_detector import DomainDetector
from src.utils.domain_utils import DomainProcessor

# 限制底层模型并行，防止多线程冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


class TotalRecallSystem:
    # --- 全 17 领域精简语义锚点 ---
    DOMAIN_PROMPTS = {
        "1": "CS, IT, Software", "2": "Medicine, Biology", "3": "Politics, Law",
        "4": "Engineering, Manufacturing", "5": "Physics, Energy", "6": "Material Science",
        "7": "Biology, Life", "8": "Geography, Earth", "9": "Chemistry",
        "10": "Business, Management", "11": "Sociology", "12": "Philosophy",
        "13": "Environment", "14": "Mathematics", "15": "Psychology",
        "16": "Geology", "17": "Economics"
    }

    def __init__(self):
        print("[*] 正在初始化全量召回系统 (Training-Safe Mode)...", flush=True)
        self.encoder = QueryEncoder()
        # 各路召回上限：每路最多保留 150 名候选人，后续在多路融合阶段统一精排
        self.v_path = VectorPath(recall_limit=150)
        self.l_path = LabelRecallPath(recall_limit=150)
        self.c_path = CollaborativeRecallPath(recall_limit=150)
        self.executor = ThreadPoolExecutor(max_workers=3)
        # 统一的领域探测服务：供向量路 / 标签路调用
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

    def execute(self, query_text, domain_id=None, is_training=False):
        """
        执行多路召回
        :param domain_id: 输入 1-17 的字符串，若为 "0" 或 None 则视为全领域搜索
        """
        start_time = time.time()

        # --- 1. 基础向量编码（原始 JD 文本）---
        raw_vec, _ = self.encoder.encode(query_text)
        faiss.normalize_L2(raw_vec)

        # --- 2. 统一领域探测逻辑 ---
        # user_domain：用户显式指定的领域；为 None 表示“未指定/全领域”
        user_domain = None
        if domain_id and str(domain_id).strip() not in ["0", ""]:
            user_domain = str(domain_id).strip()

        active_domains, applied_domain_str, domain_debug = self.domain_detector.detect(
            raw_vec,
            query_text=query_text,
            user_domain=user_domain if not is_training else user_domain,
        )

        # --- 4. 为向量路构造领域过滤 ID（字符串形式，如 "1|4|14"）---
        vector_domains = None
        if active_domains:
            vector_domains = "|".join(sorted(active_domains))

        # --- 5. 为向量路构造最终 Query（自动/手动领域 bias）---
        final_query = query_text
        if vector_domains and not is_training:
            domain_set = DomainProcessor.to_set(vector_domains)
            if domain_set:
                primary_domain = sorted(domain_set)[0]
                if primary_domain in self.DOMAIN_PROMPTS:
                    bias = self.DOMAIN_PROMPTS[primary_domain]
                    final_query = f"{query_text} | Area: {bias}"

        # --- 6. 向量转换（供向量路与标签路共用同一编码空间）---
        query_vec, _ = self.encoder.encode(final_query)
        faiss.normalize_L2(query_vec)

        # --- 3. 并行召回分发 ---
        # 向量路：使用自动/手动推断的 vector_domains 进行领域过滤
        # 标签路：若用户指定了 domain，则使用用户口径；否则内部按自动探测结果执行
        future_v = self.executor.submit(self.v_path.recall, query_vec, target_domains=vector_domains)
        future_l = self.executor.submit(
            self.l_path.recall,
            query_vec,
            domain_id=user_domain,
            query_text=query_text,
        )

        v_list, v_cost = future_v.result()
        l_list, l_cost = future_l.result()

        # --- 4. 协同路扩散 (基于向量路和标签路的最优种子) ---
        seeds = list(set(v_list[:100] + l_list[:100]))
        c_list, c_cost = self.c_path.recall(seeds)

        # --- 5. RRF 结果融合（多路各 150 上限，融合后整体再截断为约 200 人候选池） ---
        final_list, rank_map = self._fuse_results(v_list, l_list, c_list)

        return {
            # 三路融合后的全局候选池（约 200 人，用于后续精排）
            "final_top_200": final_list,
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

    def _fuse_results(self, v_res, l_res, c_res):
        """
        RRF (Reciprocal Rank Fusion) 多路融合。
        输入：各路最多 150 名候选人；输出：按融合分数排序后的前 200 名作者 ID。
        """
        rrf_k = 60
        scores = {}
        rank_map = {}

        # 赋予不同召回路径不同的基础权重（V:L:C = 2:3:1）
        paths = [("v", v_res, 2.0), ("l", l_res, 3.0), ("c", c_res, 1.0)]

        for p_tag, res_list, weight in paths:
            for rank, aid in enumerate(res_list):
                score = weight * (1.0 / (rrf_k + rank + 1))
                scores[aid] = scores.get(aid, 0) + score

                if aid not in rank_map:
                    rank_map[aid] = {'v': '-', 'l': '-', 'c': '-'}
                rank_map[aid][p_tag] = rank + 1

        # --- multi-route bonus：命中多条路径的作者额外加分，强化 V∩L∩C 交集 ---
        # 1. 统计每个作者被几条路径命中（1/2/3）
        path_count = {}
        for aid, ranks in rank_map.items():
            cnt = 0
            if ranks["v"] != "-":
                cnt += 1
            if ranks["l"] != "-":
                cnt += 1
            if ranks["c"] != "-":
                cnt += 1
            path_count[aid] = cnt

        # 2. 在 RRF 分基础上增加一个与 path_count 成正比的小 bonus（柔性多路交集加分）
        alpha = 0.05  # 可按效果微调：增大则更偏好多路交集，减小则更接近原始 RRF
        final_score = {}
        for aid, base in scores.items():
            pc = path_count.get(aid, 1)
            bonus = alpha * (pc - 1)  # 1 路: +0, 2 路: +alpha, 3 路: +2alpha
            final_score[aid] = base + bonus

        sorted_candidates = sorted(final_score.items(), key=lambda x: x[1], reverse=True)

        # 3. 柔性多路交集优先：不强制过滤单路，但命中多路的作者在排序上天然占优
        # 若你想进一步收紧，也可以在这里加入硬过滤逻辑（例如只保留至少命中 L 或 path_count>=2 的作者）

        # 最终多路融合候选池：全局最多保留 200 名作者供下游精排
        return [item[0] for item in sorted_candidates[:200]], rank_map


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
        # 1. 领域选择（一次锁定，多次查询）
        domain_choice = input("\n请选择领域编号 (1-17, 0或回车跳过): ").strip()
        current_field = fields.get(domain_choice, "全领域")
        print(f"[*] 当前系统环境：{current_field} {'(硬过滤已激活)' if domain_choice in fields else '(全领域广度搜索)'}")

        while True:
            # 2. 需求输入
            user_input = input(f"\n[{current_field}] 请输入岗位需求或技术关键词 (q退出): ").strip()
            if not user_input or user_input.lower() == 'q':
                break

            # 3. 执行系统召回
            results = system.execute(user_input, domain_id=domain_choice)
            # 多路融合后的全局候选池（约 200 名作者），此处交由终端 Demo 打印前 50 名
            candidates = results['final_top_200']
            rank_map = results['rank_map']

            # 4. 打印报告
            print(f"\n[召回报告] 耗时: {results['total_ms']:.2f}ms | 路径耗时: V={results['details']['v_cost']:.1f}ms, L={results['details']['l_cost']:.1f}ms, C={results['details']['cost_c']:.1f}ms")
            print("-" * 115)
            print(f"{'综合排名':<6} | {'作者 ID':<10} | {'各路名次 (V/L/C)':<15} | {'知识图谱核心作 (权重)'}")
            print("-" * 115)

            for rank, aid in enumerate(candidates[:50], 1):
                rm = rank_map[aid]
                # 格式化各路名次显示
                v_rank = f"{rm['v']}" if rm['v'] != '-' else "-"
                l_rank = f"{rm['l']}" if rm['l'] != '-' else "-"
                c_rank = f"{rm['c']}" if rm['c'] != '-' else "-"
                path_ranks = f"V:{v_rank:>3} L:{l_rank:>3} C:{c_rank:>3}"

                # 获取代表作信息
                works = system._get_author_works(aid, top_n=1)
                if works:
                    work_title = works[0]['title']
                    if len(work_title) > 60: work_title = work_title[:57] + "..."
                    info = f"《{work_title}》({works[0]['weight']:.3f})"
                else:
                    info = "暂无图谱论文数据"

                print(f"#{rank:<5} | {aid:<10} | {path_ranks:<15} | {info}")

            print("-" * 115)
            print(f"[*] 已召回 {len(candidates)} 名候选人，上方显示前 50 名综合最优解。")

    except KeyboardInterrupt:
        print("\n[!] 系统安全退出。")