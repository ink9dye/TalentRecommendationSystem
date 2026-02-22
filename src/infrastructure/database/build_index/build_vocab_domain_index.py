import sqlite3
import json
import os
import collections
from tqdm import tqdm
from datetime import datetime
from py2neo import Graph
from config import CONFIG_DICT, VOCAB_STATS_DB_PATH


class VocabDomainIndexer:
    """
    学术词汇领域分布索引构建器。
    逻辑：扫描图谱中 (v:Vocabulary)<-[:HAS_TOPIC]-(w:Work) 的拓扑结构，
          统计每个词汇关联论文的领域分布（Domain Distribution），
          预计算领域跨度（Domain Span）以支持实时降噪。
    """

    def __init__(self):
        # 数据库路径：建议直接集成在主 DB 或是独立的词汇统计库
        self.db_path = VOCAB_STATS_DB_PATH
        self._init_graph()
        self._prepare_db()

    def _init_graph(self):
        """初始化 Neo4j 连接"""
        self.graph = Graph(
            CONFIG_DICT["NEO4J_URI"],
            auth=(CONFIG_DICT["NEO4J_USER"], CONFIG_DICT["NEO4J_PASSWORD"]),
            name=CONFIG_DICT["NEO4J_DATABASE"]
        )

    def _prepare_db(self):
        """初始化统计索引表结构"""
        conn = sqlite3.connect(self.db_path)
        # 优化：WAL 模式提升并发读写性能
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=OFF;

            DROP TABLE IF EXISTS vocabulary_domain_stats;
            CREATE TABLE vocabulary_domain_stats (
                voc_id INTEGER PRIMARY KEY,
                work_count INTEGER,    -- 该词关联的论文总数 (degree_w)
                domain_span INTEGER,   -- 涉及的唯一领域总数 (span)
                domain_dist TEXT,      -- 领域统计分布的 JSON: {"1": 100, "4": 50}
                updated_at TIMESTAMP
            );
            CREATE INDEX idx_vds_span ON vocabulary_domain_stats(domain_span);
        """)
        conn.close()

    def build_index(self):
        """核心构建流程"""
        print(f"--- 开始构建词汇领域统计索引 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---")

        # 1. 获取所有有效的词汇 ID
        print("-> 正在获取词汇列表...")
        voc_query = "MATCH (v:Vocabulary) RETURN v.id AS vid"
        voc_ids = [r['vid'] for r in self.graph.run(voc_query)]

        batch_size = 500  # 分批处理，防止 Neo4j 内存溢出
        conn = sqlite3.connect(self.db_path)
        pbar = tqdm(total=len(voc_ids), desc="Indexing Vocab Domain Distribution")

        batch_results = []
        for i in range(0, len(voc_ids), batch_size):
            v_batch = voc_ids[i:i + batch_size]

            # 2. 批量聚合 Cypher：一次性拉取该批词汇关联的所有论文领域 ID
            cypher = """
            MATCH (v:Vocabulary)<-[:HAS_TOPIC]-(w:Work)
            WHERE v.id IN $vids
            RETURN v.id AS vid, collect(w.domain_ids) AS domains_list
            """
            cursor = self.graph.run(cypher, vids=v_batch)

            for record in cursor:
                vid = record['vid']
                all_domains = record['domains_list']  # 格式如 ["1,4", "1", "4,14"]

                # 3. 统计分布
                dist = collections.Counter()
                for d_str in all_domains:
                    if d_str:
                        for d_id in d_str.split(','):
                            dist[d_id.strip()] += 1

                if not dist:
                    continue

                work_count = sum(dist.values())
                domain_span = len(dist)

                batch_results.append((
                    vid,
                    work_count,
                    domain_span,
                    json.dumps(dist),
                    datetime.now().isoformat()
                ))

            # 4. 批量写入 SQLite
            if len(batch_results) >= 1000:
                conn.executemany("""
                                 INSERT INTO vocabulary_domain_stats
                                 VALUES (?, ?, ?, ?, ?)
                                 """, batch_results)
                conn.commit()
                batch_results = []

            pbar.update(len(v_batch))

        # 处理剩余数据
        if batch_results:
            conn.executemany("INSERT INTO vocabulary_domain_stats VALUES (?, ?, ?, ?, ?)", batch_results)
            conn.commit()

        pbar.close()
        conn.close()
        print("--- 词汇领域统计索引构建完成 ---")


if __name__ == "__main__":
    indexer = VocabDomainIndexer()
    try:
        start_time = datetime.now()
        indexer.build_index()
        print(f"--- 任务总耗时: {datetime.now() - start_time} ---")
    except Exception as e:
        print(f"[Fatal Error] 构建失败: {e}")