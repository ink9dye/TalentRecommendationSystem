from py2neo import Graph
import pandas as pd
import time
import logging
# 导入全局配置
from config import CONFIG_DICT


class LabelRecallPath:
    """
    标签路召回：基于知识图谱的路径扩展
    职责：通过 Job 节点关联的技能/词汇，在图谱中通过多步关系发现相关作者
    """

    def __init__(self, top_k=300):
        self.top_k = top_k
        self.graph = None
        try:
            # 初始化 Neo4j 连接
            self.graph = Graph(
                CONFIG_DICT["NEO4J_URI"],
                auth=(CONFIG_DICT["NEO4J_USER"], CONFIG_DICT["NEO4J_PASSWORD"]),
                name=CONFIG_DICT["NEO4J_DATABASE"]
            )
            # 测试连接
            self.graph.run("RETURN 1").evaluate()
            print("[OK] Neo4j 连接成功 (LabelRecallPath)")
        except Exception as e:
            logging.error(f"Neo4j 连接失败: {e}")
            self.graph = None

    def recall(self, job_id, timeout=0.8):
        """
        根据岗位 ID 召回人才
        """
        if self.graph is None:
            print("[Warning] Neo4j 未连接，标签路召回跳过")
            return []

        start_time = time.time()
        deadline = start_time + timeout
        all_authors = {}

        # 优化点 1：修正关系名称
        # 根据 config.py: Job->Vocab 是 REQUIRE_SKILL; Work->Vocab 是 HAS_TOPIC
        # 阶段 1：直接匹配 (Job -> Skill -> Work -> Author)
        cypher_direct = """
        MATCH (j:Job {id: $job_id})-[:REQUIRE_SKILL]->(v:Vocabulary)
        MATCH (v)<-[:HAS_TOPIC]-(w:Work)<-[r:AUTHORED]-(a:Author)
        RETURN a.id AS aid, sum(r.pos_weight) AS score
        ORDER BY score DESC LIMIT $limit
        """

        try:
            res1 = self.graph.run(cypher_direct, job_id=job_id, limit=self.top_k).to_data_frame()
            if not res1.empty:
                all_authors = dict(zip(res1['aid'], res1['score']))

            # 阶段 2：语义桥接扩展 (SIMILAR_TO)
            # 只有在时间允许且第一路结果不足时，或为了增加多样性时执行
            if time.time() < deadline:
                cypher_expanded = """
                MATCH (j:Job {id: $job_id})-[:REQUIRE_SKILL]->(v:Vocabulary)-[:SIMILAR_TO]-(v_rel:Vocabulary)
                MATCH (v_rel)<-[:HAS_TOPIC]-(w:Work)<-[r:AUTHORED]-(a:Author)
                RETURN a.id AS aid, sum(r.pos_weight) AS score
                ORDER BY score DESC LIMIT $limit
                """
                res2 = self.graph.run(cypher_expanded, job_id=job_id, limit=self.top_k).to_data_frame()
                if not res2.empty:
                    for _, row in res2.iterrows():
                        # 语义扩展的分数通常赋予一个衰减系数（如 0.8）
                        all_authors[row['aid']] = all_authors.get(row['aid'], 0) + (row['score'] * 0.8)

        except Exception as e:
            print(f"[Error] Cypher 查询出错: {e}")
            return []

        # 按总分排序返回
        sorted_authors = sorted(all_authors.items(), key=lambda x: x[1], reverse=True)
        return [a[0] for a in sorted_authors[:self.top_k]]