import sqlite3
import time
import json
from collections import defaultdict
from config import COLLAB_DB_PATH, DB_PATH
from src.utils.time_features import compute_author_time_features


class CollaborativeRecallPath:
    """
    协同路召回优化版：
    1. 移除 Pandas，改用原生 sqlite3 游标以消除对象初始化开销。
    2. 使用 set 替代 list 进行成员检测，将过滤复杂度从 O(N) 降至 O(1)。
    3. 预配置数据库缓存，最大化利用 build_collaborative_index.py 生成的覆盖索引。
    """

    def __init__(self, recall_limit=200):
        self.collab_db_path = COLLAB_DB_PATH
        self.recall_limit = recall_limit

    def recall(self, seed_author_ids, timeout=5.0):
        """
        根据种子作者 ID 极速召回协作关系最紧密的候选人
        """
        if not seed_author_ids:
            return [], 0

        start_time = time.time()
        deadline = start_time + timeout
        aggregated_results = {}

        # 将种子转为 set，确保在循环内过滤时的极速响应
        seed_set = set(seed_author_ids)

        # 建立协作索引连接并启用性能预调优
        conn = sqlite3.connect(self.collab_db_path)
        # 增加内存缓存页，确保索引常驻内存
        conn.execute("PRAGMA cache_size = -100000")
        cursor = conn.cursor()

        try:
            # 维持 50-100 的分批大小，平衡 SQL 解析开销与占位符限制
            chunk_size = 100
            for i in range(0, len(seed_author_ids), chunk_size):
                if time.time() > deadline:
                    break

                chunk = seed_author_ids[i:i + chunk_size]
                placeholders = ','.join(['?'] * len(chunk))

                # 利用 build_collaborative_index.py 构建的双向覆盖索引进行 Index-Only Scan
                query = f"""
                    SELECT aid2 as target_id, score FROM scholar_collaboration 
                    WHERE aid1 IN ({placeholders})
                    UNION ALL
                    SELECT aid1 as target_id, score FROM scholar_collaboration 
                    WHERE aid2 IN ({placeholders})
                """

                # 执行原生查询，不转换为 DataFrame
                cursor.execute(query, chunk + chunk)
                rows = cursor.fetchall()

                # 原生元组迭代：比 pd.iterrows() 快数十倍
                for tid, s in rows:
                    if tid in seed_set:
                        continue
                    # 字典累加
                    aggregated_results[tid] = aggregated_results.get(tid, 0) + s

        finally:
            conn.close()

        if not aggregated_results:
            duration = (time.time() - start_time) * 1000
            return [], duration

        # ---- 作者层时间特征：对协作得分进行活跃度 + 动量加权 ----
        candidate_ids = list(aggregated_results.keys())

        # 从主学术库中拉取作者的论文年份
        main_conn = sqlite3.connect(DB_PATH)
        try:
            placeholders = ",".join(["?"] * len(candidate_ids))
            year_rows = main_conn.execute(
                f"""
                SELECT a.author_id, w.year
                FROM authorships a
                JOIN works w ON a.work_id = w.work_id
                WHERE a.author_id IN ({placeholders})
                """,
                candidate_ids,
            ).fetchall()
        finally:
            main_conn.close()

        years_by_author = defaultdict(list)
        for aid, year in year_rows:
            years_by_author[aid].append(year)

        for aid, base_score in list(aggregated_results.items()):
            years = years_by_author.get(aid, [])
            _, _, time_weight = compute_author_time_features(years)
            aggregated_results[aid] = float(base_score) * float(time_weight)

        # 排序并返回得分最高的候选人
        sorted_res = sorted(aggregated_results.items(), key=lambda x: x[1], reverse=True)
        duration = (time.time() - start_time) * 1000

        return [r[0] for r in sorted_res[:self.recall_limit]], duration


if __name__ == "__main__":
    # 测试代码保持不变
    c_path = CollaborativeRecallPath(recall_limit=200)

    print("\n" + "=" * 60)
    print("🤝 协同路 (Collaborative Network) 极速版独立测试")
    print("=" * 60)

    try:
        while True:
            raw_v = input("\n[1/2] 请粘贴『向量路』的作者 ID 列表 (JSON格式):\n>> ").strip()
            if raw_v.lower() == 'q': break
            raw_l = input("\n[2/2] 请粘贴『标签路』的作者 ID 列表 (JSON格式):\n>> ").strip()
            if raw_l.lower() == 'q': break

            try:
                v_seeds = json.loads(raw_v.replace("'", '"'))
                l_seeds = json.loads(raw_l.replace("'", '"'))
                combined_seeds = list(set(v_seeds + l_seeds))

                print(f"[*] 正在从协作索引中挖掘潜在合作伙伴 (种子数: {len(combined_seeds)})...")
                collab_ids, cost = c_path.recall(combined_seeds)

                print(f"\n[召回报告]")
                print(f"- 耗时: {cost:.2f} ms (预期应比原版下降一个数量级)")
                print(f"- 关联人才数: {len(collab_ids)}")
                print("-" * 30)
                print(collab_ids[:10])  # 仅展示前10名预览

            except Exception as e:
                print(f"[Error] 解析失败: {e}")
    except KeyboardInterrupt:
        print("\n已安全退出")