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

    def _build_domain_ratio(self):
        """
        从 vocabulary_domain_stats 的 domain_dist JSON 展开为 (voc_id, domain_id, ratio)，
        写入 vocabulary_domain_ratio。ratio = 该领域论文数 / work_count。
        使用 Python 内存展开 + 批处理写入 + 进度条。
        """
        print("-> 正在构建词汇领域占比索引 (vocabulary_domain_ratio)...")
        conn = sqlite3.connect(self.db_path)
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

    def _build_cooc_domain_ratio(self):
        """
        构建词汇共现领域占比表 vocabulary_cooc_domain_ratio。
        对每个 (voc_id T, domain_id d)：ratio = Σ freq(T,P)*ratio(P,d) / Σ freq(T,P)，
        其中 ratio(P,d) 来自 vocabulary_domain_ratio（伙伴 P 的领域占比），freq 来自 vocabulary_cooccurrence。
        依赖：vocabulary_cooccurrence、vocabulary_domain_ratio 已就绪；主库 vocabulary 表提供 term->voc_id。
        """
        print("-> 正在构建词汇共现领域占比索引 (vocabulary_cooc_domain_ratio)...")
        if not os.path.exists(DB_PATH):
            print(f"  [Skip] 主库不存在: {DB_PATH}，跳过共现领域占比。")
            return
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # 1. 主库：term -> voc_id（共现表为小写 term，用小写做 key），带进度条
        with sqlite3.connect(DB_PATH) as main_conn:
            main_conn.row_factory = sqlite3.Row
            rows = main_conn.execute("SELECT voc_id, term FROM vocabulary WHERE term IS NOT NULL AND term != ''").fetchall()
        term_to_vocid = {}
        for r in tqdm(rows, desc="Step 1/4: 加载 term->voc_id"):
            t = (r["term"] or "").strip().lower()
            if t:
                term_to_vocid[t] = r["voc_id"]

        # 2. vocabulary_domain_ratio：voc_id -> [(domain_id, ratio), ...]，内存加载
        ratio_rows = conn.execute(
            "SELECT voc_id, domain_id, ratio FROM vocabulary_domain_ratio"
        ).fetchall()
        vocid_to_ratios = collections.defaultdict(list)
        for r in tqdm(ratio_rows, desc="Step 2/4: 加载领域占比 (内存)"):
            vocid_to_ratios[r["voc_id"]].append((str(r["domain_id"]), float(r["ratio"])))

        # 3. 共现表：遍历 (term_a, term_b, freq)，Python 内存计数 + 进度条
        cooc_rows = conn.execute("SELECT term_a, term_b, freq FROM vocabulary_cooccurrence").fetchall()
        num = collections.defaultdict(float)   # (voc_id_T, domain_id) -> sum(freq * ratio_P_d)
        den = collections.defaultdict(float)   # (voc_id_T, domain_id) -> sum(freq)

        for row in tqdm(cooc_rows, desc="Step 3/4: 共现领域聚合 (内存计数)"):
            term_a, term_b, freq = row["term_a"], row["term_b"], int(row["freq"])
            if freq <= 0:
                continue
            va = term_to_vocid.get((term_a or "").strip().lower())
            vb = term_to_vocid.get((term_b or "").strip().lower())
            if va is None or vb is None:
                continue
            for (d, r) in vocid_to_ratios.get(vb, []):
                num[(va, d)] += freq * r
                den[(va, d)] += freq
            for (d, r) in vocid_to_ratios.get(va, []):
                num[(vb, d)] += freq * r
                den[(vb, d)] += freq

        # 4. 批处理写入 vocabulary_cooc_domain_ratio + 进度条
        batch = []
        for (vid, d), total_freq in den.items():
            if total_freq <= 0:
                continue
            batch.append((vid, d, num[(vid, d)] / total_freq))
        batch_size = 10000
        for i in tqdm(range(0, len(batch), batch_size), desc="Step 4/4: 批量写入 DB"):
            chunk = batch[i : i + batch_size]
            conn.executemany(
                "INSERT OR REPLACE INTO vocabulary_cooc_domain_ratio (voc_id, domain_id, ratio) VALUES (?, ?, ?)",
                chunk,
            )
            conn.commit()
        conn.close()
        print(f"  -> 词汇共现领域占比索引构建完成，共 {len(batch)} 条。")

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

        self._build_domain_stats()
        self._build_domain_ratio()
        self._build_cooccurrence_index()
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