# -*- coding: utf-8 -*-
"""
词汇统计索引构建模块（领域分布 + 共现矩阵）

本模块负责构建并写入 vocab_stats.db，包含两类与 Vocabulary 相关的统计型索引：
  1. 领域分布索引 (vocabulary_domain_stats)：每个词汇关联论文的领域分布与跨度，供标签路召回降噪/排序。
  2. 共现索引 (vocabulary_cooccurrence)：词对在同一篇 Work 下的共现频次，与知识图谱 CO_OCCURRED_WITH 一致，
     可供标签路/解释或离线分析使用，无需依赖 Neo4j。

数据来源：
  - 领域分布：从 Neo4j 图谱 (Vocabulary)<-[:HAS_TOPIC]-(Work) 聚合得到。
  - 共现统计：从主库 academic_dataset_v5.db 的 works 表 (concepts_text/keywords_text) 计算得到，与 build_kg 逻辑一致。

运行方式：在项目根目录执行
  python -m src.infrastructure.database.build_index.build_vocab_stats_index
  或进入本目录后：python build_vocab_stats_index.py
"""

import re
import sqlite3
import json
import os
import collections
from tqdm import tqdm
from datetime import datetime
from py2neo import Graph

# 词汇统计库路径、Neo4j 配置、主库路径及共现用 SQL 均从 config 统一读取
from config import (
    CONFIG_DICT,
    VOCAB_STATS_DB_PATH,
    DB_PATH,
    SQL_QUERIES,
)


class VocabStatsIndexer:
    """
    学术词汇统计索引构建器（领域分布 + 共现）。

    职责：
      - 领域分布：扫描 Neo4j 中 (v:Vocabulary)<-[:HAS_TOPIC]-(w:Work)，统计每个词汇的
        work_count、domain_span、domain_dist（JSON），写入 vocabulary_domain_stats。
      - 共现统计：在主库中构建 work_terms_temp，执行 GET_VOCAB_CO_OCCURRENCE 得到 (term_a, term_b, freq)，
        写入 vocabulary_cooccurrence，与 build_kg 的 CO_OCCURRED_WITH 数据源一致。
    """

    def __init__(self):
        # 词汇统计库路径，与 config.VOCAB_STATS_DB_PATH 一致（通常为 data/build_index/vocab_stats.db）
        self.db_path = VOCAB_STATS_DB_PATH
        self._init_graph()
        self._prepare_db()

    def _init_graph(self):
        """初始化 Neo4j 连接，用于拉取 Vocabulary–Work 拓扑以计算领域分布。"""
        self.graph = Graph(
            CONFIG_DICT["NEO4J_URI"],
            auth=(CONFIG_DICT["NEO4J_USER"], CONFIG_DICT["NEO4J_PASSWORD"]),
            name=CONFIG_DICT["NEO4J_DATABASE"],
        )

    def _prepare_db(self):
        """
        初始化 vocab_stats.db 的表结构及索引。

        创建两张表：
          1. vocabulary_domain_stats：词汇维度的领域统计（voc_id, work_count, domain_span, domain_dist）。
          2. vocabulary_cooccurrence：词对共现频次（term_a, term_b, freq），约定 term_a < term_b 与 config 中 SQL 一致。
        启用 WAL 与 synchronous=OFF 以提升批量写入性能。
        """
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=OFF;

            -- 表1：词汇领域分布（按词统计）
            DROP TABLE IF EXISTS vocabulary_domain_stats;
            CREATE TABLE vocabulary_domain_stats (
                voc_id INTEGER PRIMARY KEY,
                work_count INTEGER,    -- 该词关联的论文总数 (degree_w)
                domain_span INTEGER,   -- 涉及的唯一领域总数 (span)
                domain_dist TEXT,      -- 领域统计分布的 JSON: {"1": 100, "4": 50}
                updated_at TIMESTAMP
            );
            CREATE INDEX idx_vds_span ON vocabulary_domain_stats(domain_span);

            -- 表2：词汇共现频次（按词对统计，与 KG CO_OCCURRED_WITH 对应）
            DROP TABLE IF EXISTS vocabulary_cooccurrence;
            CREATE TABLE vocabulary_cooccurrence (
                term_a TEXT,
                term_b TEXT,
                freq INTEGER,
                PRIMARY KEY (term_a, term_b)
            );
            CREATE INDEX idx_cooc_term_a ON vocabulary_cooccurrence(term_a);
            CREATE INDEX idx_cooc_term_b ON vocabulary_cooccurrence(term_b);
        """)
        conn.close()

    def _build_domain_stats(self):
        """
        构建词汇领域分布索引并写入 vocabulary_domain_stats。

        流程：
          1. 从 Neo4j 拉取所有 Vocabulary 的 id。
          2. 按批（500 个）执行 Cypher：对每批词汇，聚合其 HAS_TOPIC 指向的 Work 的 domain_ids。
          3. 在 Python 中统计每个词汇的 work_count、domain_span、domain_dist（Counter 转 JSON）。
          4. 批量 INSERT 进 vocabulary_domain_stats（每满 1000 条提交一次）。
        """
        print("-> 正在构建词汇领域分布索引 (vocabulary_domain_stats)...")
        voc_query = "MATCH (v:Vocabulary) RETURN v.id AS vid"
        voc_ids = [r["vid"] for r in self.graph.run(voc_query)]

        batch_size = 500
        conn = sqlite3.connect(self.db_path)
        pbar = tqdm(total=len(voc_ids), desc="Indexing Vocab Domain Distribution")

        batch_results = []
        for i in range(0, len(voc_ids), batch_size):
            v_batch = voc_ids[i : i + batch_size]
            cypher = """
            MATCH (v:Vocabulary)<-[:HAS_TOPIC]-(w:Work)
            WHERE v.id IN $vids
            RETURN v.id AS vid, collect(w.domain_ids) AS domains_list
            """
            cursor = self.graph.run(cypher, vids=v_batch)

            for record in cursor:
                vid = record["vid"]
                # domain_ids 在图中可能为 "1,4"、"1"、"4,14" 等逗号分隔字符串
                all_domains = record["domains_list"]

                dist = collections.Counter()
                for d_str in all_domains:
                    if d_str:
                        for d_id in d_str.split(","):
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
                    datetime.now().isoformat(),
                ))

            if len(batch_results) >= 1000:
                conn.executemany(
                    "INSERT INTO vocabulary_domain_stats VALUES (?, ?, ?, ?, ?)",
                    batch_results,
                )
                conn.commit()
                batch_results = []

            pbar.update(len(v_batch))

        if batch_results:
            conn.executemany(
                "INSERT INTO vocabulary_domain_stats VALUES (?, ?, ?, ?, ?)",
                batch_results,
            )
            conn.commit()

        pbar.close()
        conn.close()
        print("  -> 词汇领域分布索引构建完成。")

    def _build_cooccurrence_index(self):
        """
        构建词汇共现表并写入 vocabulary_cooccurrence。
        重构方案：放弃 SQL Join，改用 Python 内存计数 + 批量写入 + 精准进度条。
        """
        import itertools
        from collections import Counter

        print("-> 正在构建词汇共现索引 (Python 内存计数版)...")
        main_db_path = DB_PATH
        if not os.path.exists(main_db_path):
            print(f"  [Skip] 主库不存在: {main_db_path}，跳过共现索引。")
            return

        # 1. 初始化内存计数器
        cooc_counts = Counter()

        # 2. 阶段一：从主库读取数据并进行内存聚合计数
        with sqlite3.connect(main_db_path) as main_conn:
            main_conn.row_factory = sqlite3.Row

            # 获取总记录数用于初始化进度条
            count_cursor = main_conn.execute(
                "SELECT COUNT(*) as total FROM works WHERE concepts_text IS NOT NULL OR keywords_text IS NOT NULL"
            )
            total_works = count_cursor.fetchone()["total"]

            cursor = main_conn.execute(
                "SELECT concepts_text, keywords_text FROM works "
                "WHERE concepts_text IS NOT NULL OR keywords_text IS NOT NULL"
            )

            # 遍历论文提取词对
            for row in tqdm(cursor, total=total_works, desc="Step 1/2: Counting pairs in memory"):
                # 合并概念和关键词
                raw_meta = f"{row['concepts_text'] or ''}|{row['keywords_text'] or ''}"
                # 清洗、去重并排序（排序保证了 term_a < term_b，符合数据库索引约定）
                terms = sorted(list(set([
                    t.strip().lower()
                    for t in re.split(r"[|;,]", raw_meta)
                    if t.strip()
                ])))

                # 只有词数 >= 2 时才生成共现组合
                if len(terms) >= 2:
                    # itertools.combinations 生成所有唯一对 (n! / (2! * (n-2)!))
                    for pair in itertools.combinations(terms, 2):
                        cooc_counts[pair] += 1

        # 3. 统计有效共现数 (freq > 1)，以便展示精准的写入进度条
        print(f"-> 内存处理完成，总词对数: {len(cooc_counts)}")
        valid_pairs_count = sum(1 for f in cooc_counts.values() if f > 1)
        print(f"-> 有效共现对 (freq > 1): {valid_pairs_count}")

        # 4. 阶段二：批量写入目标统计库
        stats_conn = sqlite3.connect(self.db_path)
        # 启用性能优化 PRAGMA
        stats_conn.execute("PRAGMA journal_mode=WAL;")
        stats_conn.execute("PRAGMA synchronous=OFF;")

        write_batch = []
        batch_size = 10000

        # 写入阶段进度条
        pbar = tqdm(total=valid_pairs_count, desc="Step 2/2: Writing to Database")

        for (term_a, term_b), freq in cooc_counts.items():
            if freq > 1:
                write_batch.append((term_a, term_b, freq))

                if len(write_batch) >= batch_size:
                    stats_conn.executemany(
                        "INSERT OR REPLACE INTO vocabulary_cooccurrence (term_a, term_b, freq) VALUES (?, ?, ?)",
                        write_batch,
                    )
                    stats_conn.commit()
                    pbar.update(len(write_batch))
                    write_batch = []

        # 写入最后剩余的批次
        if write_batch:
            stats_conn.executemany(
                "INSERT OR REPLACE INTO vocabulary_cooccurrence (term_a, term_b, freq) VALUES (?, ?, ?)",
                write_batch,
            )
            stats_conn.commit()
            pbar.update(len(write_batch))

        pbar.close()
        stats_conn.close()

        # 清空计数器释放内存
        cooc_counts.clear()
        print("  -> 词汇共现索引构建完成。")

    def build_index(self):
        """
        全量构建词汇统计索引：先领域分布，再共现。

        执行顺序：
          1. _build_domain_stats()：依赖 Neo4j，写入 vocabulary_domain_stats。
          2. _build_cooccurrence_index()：依赖主库 works，写入 vocabulary_cooccurrence。
        """
        print(f"--- 开始构建词汇统计索引 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---")

        self._build_domain_stats()
        self._build_cooccurrence_index()

        print("--- 词汇统计索引构建完成 ---")


if __name__ == "__main__":
    indexer = VocabStatsIndexer()
    try:
        start_time = datetime.now()
        indexer.build_index()
        print(f"--- 任务总耗时: {datetime.now() - start_time} ---")
    except Exception as e:
        print(f"[Fatal Error] 构建失败: {e}")
        raise