from py2neo import Graph
import pandas as pd
import time
import logging
import json
# 导入全局配置
from config import CONFIG_DICT


class LabelRecallPath:
    """
    标签路召回：基于知识图谱的路径扩展
    职责：根据 Job ID 执行多步图路径推理，返回 Top 100 作者 ID
    """

    def __init__(self, recall_limit=100):
        self.recall_limit = recall_limit
        self.graph = None
        try:
            self.graph = Graph(
                CONFIG_DICT["NEO4J_URI"],
                auth=(CONFIG_DICT["NEO4J_USER"], CONFIG_DICT["NEO4J_PASSWORD"]),
                name=CONFIG_DICT["NEO4J_DATABASE"]
            )
            # 心跳检测
            self.graph.run("RETURN 1").evaluate()
            print("[OK] Neo4j 连接成功 (LabelRecallPath)")
        except Exception as e:
            logging.error(f"Neo4j 连接失败: {e}")

    def recall(self, job_id, timeout=1.0):
        """
        核心召回逻辑：直接匹配 + 语义扩展
        """
        if self.graph is None: return []

        start_time = time.time()
        deadline = start_time + timeout
        all_authors = {}  # {aid: score}

        # 阶段 1：直接路径匹配 (权重 1.0)
        # 路径：Job -> Skill -> Work -> Author
        cypher_direct = """
        MATCH (j:Job {id: $job_id})-[:REQUIRE_SKILL]->(v:Vocabulary)
        MATCH (v)<-[:HAS_TOPIC]-(w:Work)<-[r:AUTHORED]-(a:Author)
        RETURN a.id AS aid, sum(r.pos_weight) AS score
        ORDER BY score DESC LIMIT $limit
        """

        try:
            res1 = self.graph.run(cypher_direct, job_id=job_id, limit=self.recall_limit).to_data_frame()
            if not res1.empty:
                all_authors = dict(zip(res1['aid'], res1['score']))

            # 阶段 2：语义桥接扩展 (权重 0.8)
            # 时间允许且为了增强多样性时执行
            if time.time() < deadline:
                cypher_expanded = """
                MATCH (j:Job {id: $job_id})-[:REQUIRE_SKILL]->(v:Vocabulary)-[:SIMILAR_TO]-(v_rel:Vocabulary)
                MATCH (v_rel)<-[:HAS_TOPIC]-(w:Work)<-[r:AUTHORED]-(a:Author)
                RETURN a.id AS aid, sum(r.pos_weight) AS score
                ORDER BY score DESC LIMIT $limit
                """
                res2 = self.graph.run(cypher_expanded, job_id=job_id, limit=self.recall_limit).to_data_frame()
                if not res2.empty:
                    for _, row in res2.iterrows():
                        # 应用衰减系数
                        current_score = row['score'] * 0.8
                        all_authors[row['aid']] = all_authors.get(row['aid'], 0) + current_score

        except Exception as e:
            print(f"[Error] Cypher 执行失败: {e}")
            return []

        # 按融合分数排序并提取前 100 个 ID
        sorted_authors = sorted(all_authors.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_authors[:self.recall_limit]]


if __name__ == "__main__":
    # 实例化：设定召回 100 人
    l_path = LabelRecallPath(recall_limit=100)

    print("\n" + "=" * 60)
    print("🕸️ 标签路 (Knowledge Graph) 独立测试模块")
    print("=" * 60)

    try:
        while True:
            # 这里的输入应为数据库中存在的 Job ID (例如 securityId)
            job_id = input("\n请输入岗位 ID (Job ID) 进行召回测试 (或输入 'q' 退出):\n>> ").strip()

            if job_id.lower() in ['q', 'exit']: break
            if not job_id: continue

            print(f"[*] 正在 Neo4j 中执行路径扩展推理...")
            start_t = time.time()
            id_list = l_path.recall(job_id)
            duration = (time.time() - start_t) * 1000

            print(f"\n[召回报告]")
            print(f"- 检索总耗时: {duration:.2f} ms")
            print(f"- 最终召回作者数: {len(id_list)}")

            if id_list:
                print("-" * 30)
                print("100 个 Author ID 列表如下:")
                print(id_list)
                print("-" * 30)
            else:
                print("⚠️ 未找到匹配作者。请检查：")
                print("1. Job ID 是否真实存在于 Neo4j 中")
                print("2. 岗位是否关联了 Vocabulary 节点")
                print("3. 知识图谱中是否存在相关的作者关系链")

    except KeyboardInterrupt:
        print("\n已安全退出")