from py2neo import Graph
from config import CONFIG_DICT, CYPHER_TEMPLATES
import time

class LabelRecallPath:
    """
    标签路召回：基于知识图谱的路径扩展
    职责：通过 Job 节点关联的词汇，在图谱中多步推理发现人才
    """
    def __init__(self, top_k=300):
        # 初始化 Neo4j 连接
        self.graph = Graph(
            CONFIG_DICT["NEO4J_URI"],
            auth=(CONFIG_DICT["NEO4J_USER"], CONFIG_DICT["NEO4J_PASSWORD"]),
            name=CONFIG_DICT["NEO4J_DATABASE"]
        )
        self.top_k = top_k

    def recall(self, job_id, timeout=0.5):
        start_time = time.time()
        deadline = start_time + timeout
        all_authors = {}

        # 阶段 1：直接匹配 (Job -> Vocabulary -> Work -> Author)
        cypher_direct = """
        MATCH (j:Job {id: $job_id})-[:REQUIRES]->(v:Vocabulary)<-[:TAGGED]-(w:Work)<-[r:AUTHORED]-(a:Author)
        RETURN a.id AS aid, sum(r.pos_weight) AS score
        ORDER BY score DESC LIMIT $limit
        """
        res1 = self.graph.run(cypher_direct, job_id=job_id, limit=self.top_k).to_data_frame()
        if not res1.empty:
            all_authors = dict(zip(res1['aid'], res1['score']))

        # 阶段 2：如果时间充裕，执行语义桥接扩展 [cite: 3]
        if time.time() < deadline:
            cypher_expanded = """
            MATCH (j:Job {id: $job_id})-[:REQUIRES]->(v:Vocabulary)-[:SIMILAR_TO]-(v_rel:Vocabulary)
            MATCH (v_rel)<-[:TAGGED]-(w:Work)<-[r:AUTHORED]-(a:Author)
            RETURN a.id AS aid, sum(r.pos_weight) AS score
            ORDER BY score DESC LIMIT $limit
            """
            res2 = self.graph.run(cypher_expanded, job_id=job_id, limit=self.top_k).to_data_frame()
            if not res2.empty:
                # 合并分数
                for _, row in res2.iterrows():
                    all_authors[row['aid']] = all_authors.get(row['aid'], 0) + row['score']

        # 按总分排序返回
        sorted_authors = sorted(all_authors.items(), key=lambda x: x[1], reverse=True)
        return [a[0] for a in sorted_authors[:self.top_k]]