import sqlite3
import pandas as pd
import time
import json
import numpy as np
# 确保从你的配置中导入 COLLAB_DB_PATH
from config import COLLAB_DB_PATH


class CollaborativeRecallPath:
    """
    协同路召回：基于学术协作网络的二跳扩展
    职责：从种子作者出发，发现协作关系最紧密的合作伙伴
    """

    def __init__(self, recall_limit=100):
        self.collab_db_path = COLLAB_DB_PATH
        self.recall_limit = recall_limit

    def recall(self, seed_author_ids, timeout=5.0):
        """
        根据种子作者 ID 召回协作关系最紧密的候选人
        """
        if not seed_author_ids:
            return [], 0

        start_time = time.time()
        deadline = start_time + timeout
        aggregated_results = {}

        # 建立连接
        conn = sqlite3.connect(self.collab_db_path)

        try:
            # 分批处理种子作者，防止 SQL 占位符过多导致崩溃
            chunk_size = 50
            for i in range(0, len(seed_author_ids), chunk_size):
                if time.time() > deadline:
                    break

                chunk = seed_author_ids[i:i + chunk_size]
                placeholders = ','.join(['?'] * len(chunk))

                # 核心逻辑：双向查询索引 (aid1 < aid2 存储策略)
                # score 字段已包含 builder.py 计算的署名权重与时序衰减
                query = f"""
                    SELECT aid2 as target_id, score FROM scholar_collaboration 
                    WHERE aid1 IN ({placeholders})
                    UNION ALL
                    SELECT aid1 as target_id, score FROM scholar_collaboration 
                    WHERE aid2 IN ({placeholders})
                """

                # 使用 params 传参防止 SQL 注入并提高效率
                df = pd.read_sql_query(query, conn, params=chunk + chunk)

                for _, row in df.iterrows():
                    tid, s = row['target_id'], row['score']
                    # 过滤掉种子作者自身，防止形成闭合回路
                    if tid in seed_author_ids:
                        continue
                    # 累加协作得分：一个候选人如果与越多“种子”有合作，其学术圈关联度评分越高
                    aggregated_results[tid] = aggregated_results.get(tid, 0) + s

        finally:
            conn.close()

        # 排序并返回得分最高的候选人
        sorted_res = sorted(aggregated_results.items(), key=lambda x: x[1], reverse=True)
        duration = (time.time() - start_time) * 1000

        # 返回前 N 名合作伙伴 ID 列表
        return [r[0] for r in sorted_res[:self.recall_limit]], duration


if __name__ == "__main__":
    c_path = CollaborativeRecallPath(recall_limit=100)

    print("\n" + "=" * 60)
    print("🤝 协同路 (Collaborative Network) 双段输入独立测试")
    print("说明：请先后粘贴『向量路』和『标签路』的召回 ID 列表")
    print("=" * 60)

    try:
        while True:
            # 第一阶段输入
            raw_v = input("\n[1/2] 请粘贴『向量路』的作者 ID 列表 (JSON格式):\n>> ").strip()
            if raw_v.lower() == 'q': break

            # 第二阶段输入
            raw_l = input("\n[2/2] 请粘贴『标签路』的作者 ID 列表 (JSON格式):\n>> ").strip()
            if raw_l.lower() == 'q': break

            try:
                # 解析 JSON 并合并去重
                v_seeds = json.loads(raw_v.replace("'", '"'))
                l_seeds = json.loads(raw_l.replace("'", '"'))

                # 模拟全量召回中的逻辑：合并两路种子作为协同扩充的起点
                combined_seeds = list(set(v_seeds + l_seeds))
                print(f"[*] 种子合并完成，去重后共计 {len(combined_seeds)} 个核心作者")

                print(f"[*] 正在从协作索引中挖掘潜在合作伙伴...")
                collab_ids, cost = c_path.recall(combined_seeds)

                print(f"\n[召回报告]")
                print(f"- 协作索引查询耗时: {cost:.2f} ms")
                print(f"- 成功挖掘到的关联人才数: {len(collab_ids)}")
                print("-" * 30)
                print("【协同路最终召回 100 人顺位列表】:")
                print(collab_ids)
                print("-" * 30)

            except Exception as e:
                print(f"[Error] 解析失败，请检查输入格式是否为正确的列表格式: {e}")

    except KeyboardInterrupt:
        print("\n已安全退出")