# -*- coding: utf-8 -*-
"""
词汇统计索引构建模块（领域分布 + 共现矩阵 + 概念簇）

本模块负责构建并写入 vocab_stats.db，包含五类与 Vocabulary 相关的统计型索引：
  1. 领域分布索引 (vocabulary_domain_stats)：每个词汇关联论文的领域分布与跨度，供标签路召回降噪/排序。
  2. 领域占比索引 (vocabulary_domain_ratio)：按 (voc_id, domain_id) 存 ratio=该领域论文数/work_count，供查询时快速筛「单词领域占比≥阈值」。
  3. 共现索引 (vocabulary_cooccurrence)：词对在同一篇 Work 下的共现频次，与知识图谱 CO_OCCURRED_WITH 一致。
  4. 共现领域占比索引 (vocabulary_cooc_domain_ratio)：按 (voc_id, domain_id) 存「共现伙伴的领域占比」的 freq 加权均值，供标签路直接查 cooc_purity。
  5. 概念簇索引 (vocabulary_cluster, cluster_members)：词→簇、簇→学术词成员，供标签路按簇扩展；并产出 cluster_centroids.npy。

数据来源：
  - 领域分布：从 Neo4j 图谱 (Vocabulary)<-[:HAS_TOPIC]-(Work) 聚合得到。
  - 共现统计：从主库 academic_dataset_v5.db 的 works 表 (concepts_text/keywords_text) 计算得到，与 build_kg 逻辑一致。
  - 概念簇：依赖 build_vector_index 产出的 vocabulary 向量与主库 vocabulary.entity_type。

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
import numpy as np
from tqdm import tqdm
from datetime import datetime
from py2neo import Graph
from sklearn.cluster import KMeans

# 词汇统计库路径、Neo4j 配置、主库路径、索引路径及共现用 SQL 均从 config 统一读取
from config import (
    CONFIG_DICT,
    VOCAB_STATS_DB_PATH,
    DB_PATH,
    SQL_QUERIES,
    VOCAB_INDEX_PATH,
    VOCAB_MAP_PATH,
    INDEX_DIR,
)

# 概念簇参数：学术词聚类数；工业词最多归属簇数；工业词归属最低相似度
CONCEPT_CLUSTER_K = 700
INDUSTRY_TOP_CLUSTERS = 2
INDUSTRY_CLUSTER_MIN_SCORE = 0.3


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

        创建多张表 + 进度表：
          1. vocabulary_domain_stats：词汇维度的领域统计（voc_id, work_count, domain_span, domain_dist）。
          2. vocabulary_domain_ratio：按 (voc_id, domain_id) 存 ratio，供查询时一条 SQL 筛出领域占比≥阈值的词。
          3. vocabulary_cooccurrence：词对共现频次（term_a, term_b, freq）。
          4. vocabulary_cooc_domain_ratio：按 (voc_id, domain_id) 存共现伙伴的领域占比之 freq 加权均值。
          5. vocabulary_cooc_domain_accum：共现领域占比的分子分母累加表，支持分块/断点计算。
          6. build_progress：存各步骤断点信息，支持断点续传。
          7. vocabulary_cluster：概念簇表一，词→簇（voc_id, cluster_id, score）。
          8. cluster_members：概念簇表二，簇→学术词成员（cluster_id, voc_id）。
        """
        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-200000;
        PRAGMA busy_timeout=60000;

        -- 表1：词汇领域分布（按词统计）
        CREATE TABLE IF NOT EXISTS vocabulary_domain_stats (
            voc_id INTEGER PRIMARY KEY,
            work_count INTEGER,
            domain_span INTEGER,
            domain_dist TEXT,
            updated_at TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_vds_span ON vocabulary_domain_stats(domain_span);

        -- 表2：词汇领域占比（按词+领域展开，便于查询时 SUM(ratio) 得目标领域占比）
        CREATE TABLE IF NOT EXISTS vocabulary_domain_ratio (
            voc_id INTEGER,
            domain_id TEXT,
            ratio REAL,
            PRIMARY KEY (voc_id, domain_id)
        );
        CREATE INDEX IF NOT EXISTS idx_vdr_domain ON vocabulary_domain_ratio(domain_id, voc_id);
        CREATE INDEX IF NOT EXISTS idx_vdr_voc ON vocabulary_domain_ratio(voc_id);

        -- 表3：词汇共现频次（按词对统计，与 KG CO_OCCURRED_WITH 对应）
        CREATE TABLE IF NOT EXISTS vocabulary_cooccurrence (
            term_a TEXT,
            term_b TEXT,
            freq INTEGER,
            PRIMARY KEY (term_a, term_b)
        );
        CREATE INDEX IF NOT EXISTS idx_cooc_term_a ON vocabulary_cooccurrence(term_a);
        CREATE INDEX IF NOT EXISTS idx_cooc_term_b ON vocabulary_cooccurrence(term_b);
        CREATE INDEX IF NOT EXISTS idx_cooc_pair ON vocabulary_cooccurrence(term_a,term_b);

        -- 表4：词汇共现领域占比（按词+领域，伙伴的 vocabulary_domain_ratio 按 freq 加权平均）
        CREATE TABLE IF NOT EXISTS vocabulary_cooc_domain_ratio (
            voc_id INTEGER,
            domain_id TEXT,
            ratio REAL,
            PRIMARY KEY (voc_id, domain_id)
        );
        CREATE INDEX IF NOT EXISTS idx_vcodr_domain ON vocabulary_cooc_domain_ratio(domain_id, voc_id);

        -- 表4-中间累加表：存共现领域占比的分子分母，避免一次性巨型 SQL
        CREATE TABLE IF NOT EXISTS vocabulary_cooc_domain_accum (
            voc_id INTEGER,
            domain_id TEXT,
            sum_freq REAL,
            sum_weight REAL,
            PRIMARY KEY (voc_id, domain_id)
        );

        -- 构建进度表：支持断点续传
        CREATE TABLE IF NOT EXISTS build_progress (
            step TEXT PRIMARY KEY,
            checkpoint INTEGER,
            done INTEGER DEFAULT 0,
            updated_at TIMESTAMP
        );

        -- 概念簇表1：词 -> 簇（学术词 + 工业词）
        CREATE TABLE IF NOT EXISTS vocabulary_cluster (
            voc_id INTEGER NOT NULL,
            cluster_id INTEGER NOT NULL,
            score REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY (voc_id, cluster_id)
        );
        CREATE INDEX IF NOT EXISTS idx_vc_voc ON vocabulary_cluster(voc_id);
        CREATE INDEX IF NOT EXISTS idx_vc_cluster ON vocabulary_cluster(cluster_id);

        -- 概念簇表2：簇 -> 成员（仅学术词）
        CREATE TABLE IF NOT EXISTS cluster_members (
            cluster_id INTEGER NOT NULL,
            voc_id INTEGER NOT NULL,
            PRIMARY KEY (cluster_id, voc_id)
        );
        CREATE INDEX IF NOT EXISTS idx_cm_cluster ON cluster_members(cluster_id);
        """)
        conn.close()

    def _get_progress(self, step: str):
        """
        从 build_progress 表读取某一步骤的断点信息。
        返回: (checkpoint, done)
        """
        conn = sqlite3.connect(self.db_path, timeout=60)
        cur = conn.execute(
            "SELECT checkpoint, done FROM build_progress WHERE step = ?",
            (step,),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return 0, False
        checkpoint, done = row
        return int(checkpoint or 0), bool(done)

    def _set_progress(self, step: str, checkpoint: int, done: bool = False):
        """
        更新某一步骤的断点信息。
        """
        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.execute(
            """
            INSERT INTO build_progress(step, checkpoint, done, updated_at)
            VALUES(?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(step) DO UPDATE SET
                checkpoint = excluded.checkpoint,
                done = excluded.done,
                updated_at = excluded.updated_at
            """,
            (step, int(checkpoint), int(done)),
        )
        conn.commit()
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

        if not voc_ids:
            print("  -> Neo4j 中无 Vocabulary 节点，跳过。")
            return

        total = len(voc_ids)
        batch_size = 500

        # 断点续传：checkpoint 存的是已处理到的下标 i
        start_idx, done = self._get_progress("domain_stats")
        if done:
            print("  -> 已完成，跳过。")
            return

        start_idx = max(0, min(start_idx, total))

        conn = sqlite3.connect(self.db_path, timeout=60)
        pbar = tqdm(
            total=total - start_idx,
            desc="Indexing Vocab Domain Distribution",
        )

        batch_results = []
        for i in range(start_idx, total, batch_size):
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

                batch_results.append(
                    (
                        vid,
                        work_count,
                        domain_span,
                        json.dumps(dist),
                        datetime.now().isoformat(),
                    )
                )

            if len(batch_results) >= 1000:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO vocabulary_domain_stats
                    (voc_id, work_count, domain_span, domain_dist, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    batch_results,
                )
                conn.commit()
                batch_results = []
                # 已成功写入，以 i+batch_size 作为新的断点
                self._set_progress("domain_stats", i + batch_size, done=False)

            pbar.update(len(v_batch))

        if batch_results:
            conn.executemany(
                """
                INSERT OR REPLACE INTO vocabulary_domain_stats
                (voc_id, work_count, domain_span, domain_dist, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                batch_results,
            )
            conn.commit()

        pbar.close()
        conn.close()
        # 全部完成
        self._set_progress("domain_stats", total, done=True)
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

        # 断点续传：checkpoint 存的是已处理到的 voc_id
        last_voc_id, done = self._get_progress("domain_ratio")
        if done:
            print("  -> 已完成，跳过。")
            conn.close()
            return

        # 预估总量用于进度条（非必需，仅用于显示）
        total_rows = conn.execute(
            """
            SELECT COUNT(*)
            FROM vocabulary_domain_stats
            WHERE work_count > 0
              AND domain_dist IS NOT NULL
              AND domain_dist != ''
              AND voc_id > ?
            """,
            (last_voc_id,),
        ).fetchone()[0]

        pbar = tqdm(
            total=total_rows,
            desc="Building vocabulary_domain_ratio (stream)",
        )

        page_size = 1000
        processed_rows = 0

        while True:
            rows = conn.execute(
                """
                SELECT voc_id, work_count, domain_dist
                FROM vocabulary_domain_stats
                WHERE work_count > 0
                  AND domain_dist IS NOT NULL
                  AND domain_dist != ''
                  AND voc_id > ?
                ORDER BY voc_id
                LIMIT ?
                """,
                (last_voc_id, page_size),
            ).fetchall()

            if not rows:
                break

            batch = []
            for row in rows:
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
                last_voc_id = voc_id
                processed_rows += 1

            if batch:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO vocabulary_domain_ratio
                    (voc_id, domain_id, ratio)
                    VALUES (?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                self._set_progress("domain_ratio", last_voc_id, done=False)

            pbar.update(len(rows))

        pbar.close()
        # 全部完成
        self._set_progress("domain_ratio", last_voc_id, done=True)
        conn.close()
        print("  -> 词汇领域占比索引构建完成。")

    def _build_cooccurrence_index(self):
        """
        优化版：
        - 不使用巨型 Counter
        - 分批写入 SQLite
        - 使用 INSERT ... ON CONFLICT 累加 freq
        - 内存常数级
        """
        import itertools

        print("-> 正在构建词汇共现索引 (流式 + 断点版)...")

        if not os.path.exists(DB_PATH):
            print(f"  [Skip] 主库不存在: {DB_PATH}")
            return

        stats_conn = sqlite3.connect(self.db_path, timeout=60)

        # 断点续传：checkpoint 存的是主库 works.rowid
        last_rowid, done = self._get_progress("cooccurrence")
        if done:
            print("  -> 已完成，跳过。")
            stats_conn.close()
            return

        batch = []
        batch_size = 100000

        with sqlite3.connect(DB_PATH) as main_conn:
            main_conn.row_factory = sqlite3.Row

            # 预估总量用于进度条
            total_rows = main_conn.execute(
                """
                SELECT COUNT(*)
                FROM works
                WHERE (concepts_text IS NOT NULL OR keywords_text IS NOT NULL)
                  AND rowid > ?
                """,
                (last_rowid,),
            ).fetchone()[0]

            pbar = tqdm(
                total=total_rows,
                desc="Counting & Writing Cooccurrence (stream)",
            )

            while True:
                rows = main_conn.execute(
                    """
                    SELECT rowid, concepts_text, keywords_text
                    FROM works
                    WHERE (concepts_text IS NOT NULL OR keywords_text IS NOT NULL)
                      AND rowid > ?
                    ORDER BY rowid
                    LIMIT 10000
                    """,
                    (last_rowid,),
                ).fetchall()

                if not rows:
                    break

                for row in rows:
                    raw_meta = f"{row['concepts_text'] or ''}|{row['keywords_text'] or ''}"

                    terms = sorted(
                        set(
                            t.strip().lower()
                            for t in re.split(r"[|;,]", raw_meta)
                            if t.strip()
                        )
                    )

                    if len(terms) < 2:
                        last_rowid = row["rowid"]
                        continue

                    for term_a, term_b in itertools.combinations(terms, 2):
                        batch.append((term_a, term_b, 1))

                        if len(batch) >= batch_size:
                            stats_conn.executemany(
                                """
                                INSERT INTO vocabulary_cooccurrence (term_a, term_b, freq)
                                VALUES (?, ?, ?)
                                ON CONFLICT(term_a, term_b) DO UPDATE SET
                                    freq = freq + excluded.freq
                                """,
                                batch,
                            )
                            stats_conn.commit()
                            batch = []
                            self._set_progress(
                                "cooccurrence", last_rowid, done=False
                            )

                    last_rowid = row["rowid"]
                    pbar.update(1)

            if batch:
                stats_conn.executemany(
                    """
                    INSERT INTO vocabulary_cooccurrence (term_a, term_b, freq)
                    VALUES (?, ?, ?)
                    ON CONFLICT(term_a, term_b) DO UPDATE SET
                        freq = freq + excluded.freq
                    """,
                    batch,
                )
                stats_conn.commit()

            pbar.close()

        # 全部完成
        self._set_progress("cooccurrence", last_rowid, done=True)
        stats_conn.close()

        print("  -> 共现索引构建完成（流式 + 断点版）")

    def _build_cooc_domain_ratio(self):

        print("-> 正在构建词汇共现领域占比索引 (分块 + 高速版)...")

        if not os.path.exists(DB_PATH):
            print(f"[Skip] 主库不存在: {DB_PATH}")
            return

        conn = sqlite3.connect(self.db_path, timeout=60, isolation_level=None)
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-200000")
        conn.execute("ATTACH DATABASE ? AS main_db", (DB_PATH,))
        conn.row_factory = sqlite3.Row

        try:
            # term -> voc_id 映射
            conn.execute("DROP TABLE IF EXISTS temp.term_voc_map")

            conn.execute("""
                         CREATE
                         TEMP TABLE term_voc_map AS
                         SELECT LOWER(term) AS term, voc_id
                         FROM main_db.vocabulary
                         WHERE term IS NOT NULL
                           AND term != ''
                         """)

            conn.execute("CREATE INDEX idx_temp_term ON term_voc_map(term)")

            # chunk 临时表
            conn.execute("""
                         CREATE
                         TEMP TABLE IF NOT EXISTS temp_chunk_cooc(
                term_a TEXT,
                term_b TEXT,
                freq INTEGER
            )
                         """)

            last_rowid, done = self._get_progress("cooc_domain_ratio")

            if done:
                print("  -> 已完成，跳过。")
                return

            if last_rowid == 0:
                conn.execute("DELETE FROM vocabulary_cooc_domain_accum")
                conn.commit()

            total = conn.execute("""
                                 SELECT COUNT(*)
                                 FROM vocabulary_cooccurrence
                                 WHERE rowid > ?
                                 """, (last_rowid,)).fetchone()[0]

            chunk = 50000

            pbar = tqdm(
                total=total,
                desc="Computing cooc_domain_ratio (fast chunk)",
            )

            while True:

                rows = conn.execute("""
                                    SELECT rowid, term_a, term_b, freq
                                    FROM vocabulary_cooccurrence
                                    WHERE rowid > ?
                                    ORDER BY rowid LIMIT ?
                                    """, (last_rowid, chunk)).fetchall()

                if not rows:
                    break

                conn.execute("BEGIN IMMEDIATE")

                conn.execute("DELETE FROM temp_chunk_cooc")

                conn.executemany("""
                                 INSERT INTO temp_chunk_cooc(term_a, term_b, freq)
                                 VALUES (?, ?, ?)
                                 """, ((r["term_a"], r["term_b"], r["freq"]) for r in rows))

                conn.execute("""
                             INSERT INTO vocabulary_cooc_domain_accum
                             SELECT target_voc_id,
                                    domain_id,
                                    SUM(freq),
                                    SUM(weighted)
                             FROM (SELECT v.voc_id         AS target_voc_id,
                                          r.domain_id,
                                          c.freq,
                                          c.freq * r.ratio AS weighted
                                   FROM temp_chunk_cooc c
                                            JOIN term_voc_map v ON v.term = c.term_b
                                            JOIN vocabulary_domain_ratio r ON r.voc_id = v.voc_id

                                   UNION ALL

                                   SELECT v.voc_id,
                                          r.domain_id,
                                          c.freq,
                                          c.freq * r.ratio
                                   FROM temp_chunk_cooc c
                                            JOIN term_voc_map v ON v.term = c.term_a
                                            JOIN vocabulary_domain_ratio r ON r.voc_id = v.voc_id)
                             GROUP BY target_voc_id, domain_id ON CONFLICT(voc_id, domain_id) DO
                             UPDATE SET
                                 sum_freq = sum_freq + excluded.sum_freq,
                                 sum_weight = sum_weight + excluded.sum_weight
                             """)

                conn.commit()

                last_rowid = rows[-1]["rowid"]

                self._set_progress("cooc_domain_ratio", last_rowid, done=False)

                pbar.update(len(rows))

            pbar.close()

            conn.execute("DELETE FROM vocabulary_cooc_domain_ratio")

            conn.execute("""
                         INSERT INTO vocabulary_cooc_domain_ratio (voc_id, domain_id, ratio)
                         SELECT voc_id,
                                domain_id,
                                CASE
                                    WHEN sum_freq > 0 THEN sum_weight * 1.0 / sum_freq
                                    ELSE 0
                                    END
                         FROM vocabulary_cooc_domain_accum
                         """)

            conn.commit()

            self._set_progress("cooc_domain_ratio", last_rowid, done=True)

            print("-> 共现领域占比索引构建完成（高速版）")

        finally:
            conn.close()

    def _build_concept_clusters(self):
        """
        构建概念簇索引：对学术词做 K-means 聚类，工业词按与簇中心相似度归属到 top-K 簇。
        写入 vocabulary_cluster（词→簇）、cluster_members（簇→学术词），并保存 cluster_centroids.npy。
        依赖 build_vector_index 已产出的 vocabulary 向量与主库 vocabulary.entity_type。
        """
        print("-> 正在构建概念簇索引 (vocabulary_cluster, cluster_members)...")

        _, done = self._get_progress("concept_clusters")
        if done:
            print("  -> 已完成，跳过。")
            return

        vec_path = VOCAB_INDEX_PATH.replace(".faiss", "_vectors.npy")
        if not os.path.exists(vec_path):
            print(f"  [Skip] 未找到 vocabulary 向量文件: {vec_path}，请先运行 build_vector_index 构建词汇表索引。")
            return

        # 1. 读入向量与 voc_id 顺序
        vectors = np.load(vec_path).astype(np.float32)
        if vectors.ndim != 2:
            print("  [Skip] 向量维度异常，跳过概念簇。")
            return

        with open(VOCAB_MAP_PATH, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        if isinstance(raw_map, list):
            voc_ids_ordered = [str(x) for x in raw_map]
        else:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT voc_id FROM vocabulary WHERE term IS NOT NULL AND term != '' ORDER BY voc_id ASC"
                ).fetchall()
            voc_ids_ordered = [str(r["voc_id"]) for r in rows]
        if len(voc_ids_ordered) != len(vectors):
            print(f"  [Skip] 向量行数 {len(vectors)} 与 voc_id 数 {len(voc_ids_ordered)} 不一致，跳过概念簇。")
            return

        # 2. 主库取 entity_type
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT voc_id, entity_type FROM vocabulary").fetchall()
        voc_id_to_entity = {}
        for r in rows:
            vid = r["voc_id"]
            voc_id_to_entity[str(vid)] = (r["entity_type"] or "").strip().lower()

        # 3. 拆出学术词与工业词
        academic_indices = []
        industry_indices = []
        for i, vid in enumerate(voc_ids_ordered):
            et = voc_id_to_entity.get(vid) or ""
            if et in ("concept", "keyword"):
                academic_indices.append(i)
            elif et == "industry":
                industry_indices.append(i)

        X_academic = vectors[academic_indices]
        voc_ids_academic = [voc_ids_ordered[i] for i in academic_indices]
        n_academic = len(voc_ids_academic)

        K = min(CONCEPT_CLUSTER_K, max(2, n_academic - 1))
        if n_academic < K:
            print(f"  [Skip] 学术词数 {n_academic} 小于簇数 K={CONCEPT_CLUSTER_K}，跳过概念簇。")
            return

        # 4. L2 归一化后对学术词做 K-means
        norms = np.linalg.norm(X_academic, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_academic_norm = (X_academic / norms).astype(np.float32)

        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        km.fit(X_academic_norm)
        labels_academic = km.labels_
        centroids = km.cluster_centers_.astype(np.float32)
        cent_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        cent_norms[cent_norms == 0] = 1.0
        centroids = (centroids / cent_norms).astype(np.float32)

        # 5. 写表一、表二（学术词）
        stats_conn = sqlite3.connect(self.db_path, timeout=60)
        stats_conn.execute("DELETE FROM vocabulary_cluster")
        stats_conn.execute("DELETE FROM cluster_members")
        stats_conn.commit()

        batch_vc = []
        batch_cm = []
        batch_size = 3000
        for j in range(n_academic):
            vid = int(voc_ids_academic[j]) if voc_ids_academic[j].isdigit() else voc_ids_academic[j]
            cid = int(labels_academic[j])
            batch_vc.append((vid, cid, 1.0))
            batch_cm.append((cid, vid))
            if len(batch_vc) >= batch_size:
                stats_conn.executemany(
                    "INSERT OR REPLACE INTO vocabulary_cluster (voc_id, cluster_id, score) VALUES (?, ?, ?)",
                    batch_vc,
                )
                stats_conn.executemany(
                    "INSERT OR REPLACE INTO cluster_members (cluster_id, voc_id) VALUES (?, ?)",
                    batch_cm,
                )
                stats_conn.commit()
                batch_vc, batch_cm = [], []
        if batch_vc:
            stats_conn.executemany(
                "INSERT OR REPLACE INTO vocabulary_cluster (voc_id, cluster_id, score) VALUES (?, ?, ?)",
                batch_vc,
            )
            stats_conn.executemany(
                "INSERT OR REPLACE INTO cluster_members (cluster_id, voc_id) VALUES (?, ?)",
                batch_cm,
            )
            stats_conn.commit()

        # 6. 工业词归属到簇（只写表一）
        if industry_indices:
            vectors_industry = vectors[industry_indices].astype(np.float32)
            in_norms = np.linalg.norm(vectors_industry, axis=1, keepdims=True)
            in_norms[in_norms == 0] = 1.0
            vectors_industry = (vectors_industry / in_norms).astype(np.float32)
            scores = np.dot(vectors_industry, centroids.T)

            batch_industry = []
            for idx, i in enumerate(industry_indices):
                vid = voc_ids_ordered[i]
                vid_int = int(vid) if vid.isdigit() else vid
                row = scores[idx]
                top_k = min(INDUSTRY_TOP_CLUSTERS, len(row))
                top_indices = np.argsort(row)[::-1][:top_k]
                for cid in top_indices:
                    sc = float(row[cid])
                    if sc >= INDUSTRY_CLUSTER_MIN_SCORE:
                        batch_industry.append((vid_int, int(cid), sc))
                if len(batch_industry) >= batch_size:
                    stats_conn.executemany(
                        "INSERT OR REPLACE INTO vocabulary_cluster (voc_id, cluster_id, score) VALUES (?, ?, ?)",
                        batch_industry,
                    )
                    stats_conn.commit()
                    batch_industry = []
            if batch_industry:
                stats_conn.executemany(
                    "INSERT OR REPLACE INTO vocabulary_cluster (voc_id, cluster_id, score) VALUES (?, ?, ?)",
                    batch_industry,
                )
                stats_conn.commit()

        # 7. 保存簇中心
        centroids_path = os.path.join(INDEX_DIR, "cluster_centroids.npy")
        np.save(centroids_path, centroids)
        self._set_progress("concept_clusters", 1, done=True)
        stats_conn.close()
        print("  -> 概念簇索引构建完成。")

    def build_index(self):
        """
        全量构建词汇统计索引：先领域分布，再共现，再概念簇。

        执行顺序：
          1. _build_domain_stats()：依赖 Neo4j，写入 vocabulary_domain_stats。
          2. _build_domain_ratio()：从 vocabulary_domain_stats 展开，写入 vocabulary_domain_ratio。
          3. _build_cooccurrence_index()：依赖主库 works，写入 vocabulary_cooccurrence。
          4. _build_cooc_domain_ratio()：依赖 vocabulary_cooccurrence + vocabulary_domain_ratio + 主库 vocabulary，写入 vocabulary_cooc_domain_ratio。
          5. _build_concept_clusters()：依赖 vocabulary 向量与 entity_type，写入 vocabulary_cluster、cluster_members，并保存 cluster_centroids.npy。
        """
        print(f"--- 开始构建词汇统计索引 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---")

        # 各步骤均支持断点续传；多次执行不会自动清空表，只会按需增量/覆盖。
        self._build_domain_stats()
        self._build_domain_ratio()
        self._build_cooccurrence_index()
        self._build_cooc_domain_ratio()
        self._build_concept_clusters()

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