# -*- coding: utf-8 -*-
"""
词汇统计索引构建模块（领域分布 + 共现矩阵）

本模块负责构建并写入 vocab_stats.db，包含四类与 Vocabulary 相关的统计型索引：
  1. 领域分布索引 (vocabulary_domain_stats)：每个词汇关联论文的领域分布与跨度，供标签路召回降噪/排序。
  2. 领域占比索引 (vocabulary_domain_ratio)：按 (voc_id, domain_id) 存 ratio=该领域论文数/work_count，供查询时快速筛「单词领域占比≥阈值」。
  3. 共现索引 (vocabulary_cooccurrence)：词对在同一篇 Work 下的共现频次，与知识图谱 CO_OCCURRED_WITH 一致。
  4. 共现领域占比索引 (vocabulary_cooc_domain_ratio)：按 (voc_id, domain_id) 存「共现伙伴的领域占比」的 freq 加权均值，供标签路直接查 cooc_purity。

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
import gc
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
      - 领域占比：从 vocabulary_domain_stats 的 domain_dist 展开为 (voc_id, domain_id, ratio)，写入 vocabulary_domain_ratio。
      - 共现统计：在主库中构建 work_terms_temp，执行 GET_VOCAB_CO_OCCURRENCE 得到 (term_a, term_b, freq)，
        写入 vocabulary_cooccurrence，与 build_kg 的 CO_OCCURRED_WITH 数据源一致。
      - 共现领域占比：从 vocabulary_cooccurrence + vocabulary_domain_ratio + 主库 vocabulary 计算每个词在各领域的「共现伙伴占比」freq 加权均值，写入 vocabulary_cooc_domain_ratio。
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

        创建四张表：
          1. vocabulary_domain_stats：词汇维度的领域统计（voc_id, work_count, domain_span, domain_dist）。
          2. vocabulary_domain_ratio：按 (voc_id, domain_id) 存 ratio，供查询时一条 SQL 筛出领域占比≥阈值的词。
          3. vocabulary_cooccurrence：词对共现频次（term_a, term_b, freq）。
          4. vocabulary_cooc_domain_ratio：按 (voc_id, domain_id) 存共现伙伴的领域占比之 freq 加权均值，供标签路查 cooc_purity。
        """
        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.executescript("""
        PRAGMA journal_mode=TRUNCATE;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-200000;
        PRAGMA busy_timeout=60000;
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

            -- 表2：词汇领域占比（按词+领域展开，便于查询时 SUM(ratio) 得目标领域占比）
            DROP TABLE IF EXISTS vocabulary_domain_ratio;
            CREATE TABLE vocabulary_domain_ratio (
                voc_id INTEGER,
                domain_id TEXT,
                ratio REAL,            -- 该词在该领域的论文数 / work_count
                PRIMARY KEY (voc_id, domain_id)
            );
            CREATE INDEX idx_vdr_domain ON vocabulary_domain_ratio(domain_id, voc_id);

            -- 表3：词汇共现频次（按词对统计，与 KG CO_OCCURRED_WITH 对应）
            DROP TABLE IF EXISTS vocabulary_cooccurrence;
            CREATE TABLE vocabulary_cooccurrence (
                term_a TEXT,
                term_b TEXT,
                freq INTEGER,
                PRIMARY KEY (term_a, term_b)
            );
            CREATE INDEX idx_cooc_term_a ON vocabulary_cooccurrence(term_a);
            CREATE INDEX idx_cooc_term_b ON vocabulary_cooccurrence(term_b);
            CREATE INDEX idx_cooc_pair
            ON vocabulary_cooccurrence(term_a,term_b);

            -- 表4：词汇共现领域占比（按词+领域，伙伴的 vocabulary_domain_ratio 按 freq 加权平均）
            DROP TABLE IF EXISTS vocabulary_cooc_domain_ratio;
            CREATE TABLE vocabulary_cooc_domain_ratio (
                voc_id INTEGER,
                domain_id TEXT,
                ratio REAL,            -- 该词的共现伙伴在该领域的占比之 freq 加权均值
                PRIMARY KEY (voc_id, domain_id)
            );
            CREATE INDEX idx_vcodr_domain ON vocabulary_cooc_domain_ratio(domain_id, voc_id);
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
        conn = sqlite3.connect(self.db_path, timeout=60)
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

    def _build_domain_ratio(self):
        """
        从 vocabulary_domain_stats 的 domain_dist JSON 展开为 (voc_id, domain_id, ratio)，
        写入 vocabulary_domain_ratio。ratio = 该领域论文数 / work_count。
        使用 Python 内存展开 + 批处理写入 + 进度条。
        """
        print("-> 正在构建词汇领域占比索引 (vocabulary_domain_ratio)...")
        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT voc_id, work_count, domain_dist FROM vocabulary_domain_stats WHERE work_count > 0 AND domain_dist IS NOT NULL AND domain_dist != ''"
        )
        rows = cursor.fetchall()
        cursor.close()

        # 阶段一：内存展开，带进度条
        batch = []
        for row in tqdm(rows, desc="Step 1/2: 展开 domain_dist (内存计数)"):
            voc_id = row["voc_id"]
            work_count = int(row["work_count"])
            if work_count <= 0:
                continue
            try:
                dist = json.loads(row["domain_dist"])
            except (TypeError, json.JSONDecodeError):
                continue
            for domain_id, count in dist.items():
                if not domain_id or count <= 0:
                    continue
                ratio = float(count) / work_count
                batch.append((voc_id, str(domain_id).strip(), ratio))

        # 阶段二：批处理写入
        batch_size = 10000
        for i in tqdm(range(0, len(batch), batch_size), desc="Step 2/2: 批量写入 DB"):
            chunk = batch[i : i + batch_size]
            conn.executemany(
                "INSERT OR REPLACE INTO vocabulary_domain_ratio (voc_id, domain_id, ratio) VALUES (?, ?, ?)",
                chunk,
            )
            conn.commit()
        conn.close()
        print(f"  -> 词汇领域占比索引构建完成，共 {len(batch)} 条。")

    def _build_cooccurrence_index(self):
        """
        优化版：
        - 不使用巨型 Counter
        - 分批写入 SQLite
        - 使用 INSERT ... ON CONFLICT 累加 freq
        - 内存常数级
        """
        import itertools

        print("-> 正在构建词汇共现索引 (流式优化版)...")

        if not os.path.exists(DB_PATH):
            print(f"  [Skip] 主库不存在: {DB_PATH}")
            return

        stats_conn = sqlite3.connect(self.db_path, timeout=60)


        # 清空旧数据
        stats_conn.execute("DELETE FROM vocabulary_cooccurrence")
        stats_conn.commit()

        batch = []
        batch_size = 20000

        with sqlite3.connect(DB_PATH) as main_conn:
            main_conn.row_factory = sqlite3.Row

            cursor = main_conn.execute("""
                                       SELECT concepts_text, keywords_text
                                       FROM works
                                       WHERE concepts_text IS NOT NULL
                                          OR keywords_text IS NOT NULL
                                       """)

            for row in tqdm(cursor, desc="Counting & Writing Cooccurrence"):
                raw_meta = f"{row['concepts_text'] or ''}|{row['keywords_text'] or ''}"

                terms = sorted(set(
                    t.strip().lower()
                    for t in re.split(r"[|;,]", raw_meta)
                    if t.strip()
                ))

                if len(terms) < 2:
                    continue

                for term_a, term_b in itertools.combinations(terms, 2):
                    batch.append((term_a, term_b, 1))

                    if len(batch) >= batch_size:
                        stats_conn.executemany("""
                                               INSERT INTO vocabulary_cooccurrence (term_a, term_b, freq)
                                               VALUES (?, ?, ?) ON CONFLICT(term_a, term_b)
                            DO
                                               UPDATE SET freq = freq + 1
                                               """, batch)
                        stats_conn.commit()
                        batch = []

        if batch:
            stats_conn.executemany("""
                                   INSERT INTO vocabulary_cooccurrence (term_a, term_b, freq)
                                   VALUES (?, ?, ?) ON CONFLICT(term_a, term_b)
                DO
                                   UPDATE SET freq = freq + 1
                                   """, batch)
            stats_conn.commit()

        stats_conn.close()

        print("  -> 共现索引构建完成（内存安全版）")

    def _build_cooc_domain_ratio(self):

        print("-> 正在构建词汇共现领域占比索引 (chunk版)...")

        if not os.path.exists(DB_PATH):
            print(f"[Skip] 主库不存在: {DB_PATH}")
            return

        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.execute("ATTACH DATABASE ? AS main_db", (DB_PATH,))
        conn.row_factory = sqlite3.Row

        try:
            conn.execute("DROP TABLE IF EXISTS temp.term_voc_map")

            conn.execute("""
                         CREATE
                         TEMP TABLE term_voc_map AS
                         SELECT LOWER(term) term, voc_id
                         FROM main_db.vocabulary
                         WHERE term IS NOT NULL
                           AND term!=''
                         """)

            conn.execute("CREATE INDEX idx_temp_term ON term_voc_map(term)")
            conn.execute("CREATE INDEX idx_temp_voc ON term_voc_map(voc_id)")

            conn.execute("DELETE FROM vocabulary_cooc_domain_ratio")

            # 总数用于进度条
            total = conn.execute(
                "SELECT COUNT(*) FROM vocabulary_cooccurrence"
            ).fetchone()[0]

            chunk = 20000

            for offset in tqdm(range(0, total, chunk), desc="Computing cooc_domain_ratio"):
                rows = conn.execute(
                    f"""
                    SELECT term_a, term_b, freq
                    FROM vocabulary_cooccurrence
                    LIMIT {chunk} OFFSET {offset}
                    """
                ).fetchall()

                conn.execute("BEGIN IMMEDIATE")

                conn.executemany("""
                                 INSERT INTO vocabulary_cooc_domain_ratio
                                     (voc_id, domain_id, ratio)

                                 SELECT target_voc_id,
                                        domain_id,
                                        SUM(weighted) * 1.0 / SUM(freq)

                                 FROM (SELECT t.voc_id    target_voc_id,
                                              r.domain_id,
                                              ?           freq,
                                              ? * r.ratio weighted
                                       FROM term_voc_map t
                                                JOIN vocabulary_domain_ratio r ON r.voc_id = t.voc_id
                                       WHERE t.term = ?

                                       UNION ALL

                                       SELECT t.voc_id    target_voc_id,
                                              r.domain_id,
                                              ?           freq,
                                              ? * r.ratio weighted
                                       FROM term_voc_map t
                                                JOIN vocabulary_domain_ratio r ON r.voc_id = t.voc_id
                                       WHERE t.term = ?)
                                 GROUP BY target_voc_id, domain_id
                                 """, [
                                     (row["freq"], row["freq"], row["term_b"],
                                      row["freq"], row["freq"], row["term_a"])
                                     for row in rows
                                 ])

                conn.commit()

            print("-> 共现领域占比索引构建完成")

        finally:
            conn.close()

    def build_index(self):
        """
        全量构建词汇统计索引：先领域分布，再共现。

        执行顺序：
          1. _build_domain_stats()：依赖 Neo4j，写入 vocabulary_domain_stats。
          2. _build_domain_ratio()：从 vocabulary_domain_stats 展开，写入 vocabulary_domain_ratio。
          3. _build_cooccurrence_index()：依赖主库 works，写入 vocabulary_cooccurrence。
          4. _build_cooc_domain_ratio()：依赖 vocabulary_cooccurrence + vocabulary_domain_ratio + 主库 vocabulary，写入 vocabulary_cooc_domain_ratio。
        """
        print(f"--- 开始构建词汇统计索引 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---")

        # self._build_domain_stats()
        # self._build_domain_ratio()
        # self._build_cooccurrence_index()
        self._build_cooc_domain_ratio()

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