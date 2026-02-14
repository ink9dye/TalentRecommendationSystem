import sqlite3
import math
import os
import gc
from tqdm import tqdm
from datetime import datetime
from config import DB_PATH, COLLAB_DB_PATH


class WeightStrategy:
    """
    人才推荐系统权重计算策略类。
    W = (Base + Bonus) * e^(-0.1 * Δt)
    """

    @staticmethod
    def calculate(pos_index: int, is_corr: int, is_alpha: int, pub_year: int) -> float:
        current_year = datetime.now().year
        delta_t = max(0, current_year - (pub_year or 2000))
        # 指数时间衰减
        time_weight = math.exp(-0.1 * delta_t)
        # 贡献度基数计算：考虑作者排名与通讯作者加成
        contribution = 1.0 + (0.2 if is_alpha == 0 and pos_index == 1 else 0) + (0.2 if is_corr == 1 else 0)
        return round(contribution * time_weight, 4)


class LocalSimilarityIndexer:
    def __init__(self, db_path, collab_db_path):
        self.db_path = db_path
        self.collab_db_path = collab_db_path
        self._prepare_db()

    def _prepare_db(self):
        """初始化协作索引数据库结构并配置磁盘存储策略"""
        db_dir = os.path.dirname(os.path.abspath(self.collab_db_path))
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

        self.conn = sqlite3.connect(self.collab_db_path)

        # 核心优化参数：增加缓存并启用 WAL
        self.conn.executescript(f"""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=OFF;
            PRAGMA cache_size=200000; -- 约 200MB 缓存
            PRAGMA temp_store = MEMORY; 

            -- 阶段1结果：单人贡献权重缓存
            DROP TABLE IF EXISTS weighted_authorships;
            CREATE TABLE weighted_authorships (
                work_id TEXT, 
                author_id TEXT, 
                weight REAL, 
                h_index INTEGER
            );

            -- 阶段2结果：直接协作分 (S_direct)
            DROP TABLE IF EXISTS direct_scores;
            CREATE TABLE direct_scores (
                aid1 TEXT, 
                aid2 TEXT, 
                s_val REAL, 
                h1 INTEGER, 
                h2 INTEGER,
                PRIMARY KEY (aid1, aid2)
            ) WITHOUT ROWID;

            -- 阶段3结果：最终合成索引 (S_total)
            DROP TABLE IF EXISTS scholar_collaboration;
            CREATE TABLE scholar_collaboration (
                aid1 TEXT, 
                aid2 TEXT, 
                score REAL,
                PRIMARY KEY (aid1, aid2)
            ) WITHOUT ROWID;
        """)

    def step1_precompute_weights(self):
        """阶段 1: 计算权重。将引用因子提前融合进权重值。"""
        print("--- Step 1/3: 计算单体贡献权重 (预处理引用因子) ---")
        src_conn = sqlite3.connect(self.db_path)
        sql = """
              SELECT a.work_id,
                     a.author_id,
                     a.pos_index,
                     a.is_corresponding,
                     a.is_alphabetical,
                     w.year,
                     au.h_index,
                     w.citation_count
              FROM authorships a
                       JOIN works w ON a.work_id = w.work_id
                       JOIN authors au ON a.author_id = au.author_id
              """
        cursor = src_conn.execute(sql)

        batch = []
        for row in tqdm(cursor, desc="Processing Individual Weights"):
            base_w = WeightStrategy.calculate(row[2], row[3], row[4], row[5])
            # 将引用因子融合进权重：sqrt(log(cite + e))
            cite_factor = math.sqrt(math.log((row[7] or 0) + 2.71828))
            composite_weight = base_w * cite_factor

            batch.append((row[0], row[1], composite_weight, row[6]))
            if len(batch) >= 50000:
                self.conn.executemany("INSERT INTO weighted_authorships VALUES (?,?,?,?)", batch)
                batch = []

        if batch:
            self.conn.executemany("INSERT INTO weighted_authorships VALUES (?,?,?,?)", batch)
        self.conn.commit()

        print("-> 正在建立索引...")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_wa_work ON weighted_authorships(work_id)")
        src_conn.close()

    def step2_compute_direct_scores(self):
        """阶段 2: 计算协作分。使用内存迭代规避笛卡尔积开销。"""
        print("--- Step 2/3: 分批计算直接协作分 (内存保护模式) ---")

        print("-> 正在分析作品作者分布...")
        work_stats = self.conn.execute("""
                                       SELECT work_id, COUNT(author_id) as cnt
                                       FROM weighted_authorships
                                       GROUP BY work_id
                                       HAVING cnt > 1
                                          AND cnt < 100
                                       """).fetchall()

        all_works = [r[0] for r in work_stats]
        pbar = tqdm(total=len(all_works), desc="Computing Direct Scores")

        batch_data = []
        for i, wid in enumerate(all_works):
            authors = self.conn.execute(
                "SELECT author_id, weight, h_index FROM weighted_authorships WHERE work_id = ?",
                (wid,)
            ).fetchall()

            num_authors = len(authors)
            for idx1 in range(num_authors):
                for idx2 in range(idx1 + 1, num_authors):
                    a1, w1, h1 = authors[idx1]
                    a2, w2, h2 = authors[idx2]

                    if a1 > a2: a1, a2, h1, h2, w1, w2 = a2, a1, h2, h1, w2, w1

                    score = w1 * w2
                    if score > 0.05:
                        batch_data.append((a1, a2, score, h1, h2))

            if len(batch_data) >= 15000:
                self.conn.executemany("""
                                      INSERT INTO direct_scores (aid1, aid2, s_val, h1, h2)
                                      VALUES (?, ?, ?, ?, ?) ON CONFLICT(aid1, aid2) DO
                                      UPDATE SET s_val = s_val + excluded.s_val
                                      """, batch_data)
                self.conn.commit()
                batch_data = []

            if i % 10000 == 0:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                gc.collect()

            pbar.update(1)

        if batch_data:
            self.conn.executemany(
                "INSERT INTO direct_scores VALUES (?,?,?,?,?) ON CONFLICT(aid1, aid2) DO UPDATE SET s_val = s_val + excluded.s_val",
                batch_data)
            self.conn.commit()

        pbar.close()
        self.conn.execute("VACUUM;")

    def step3_compute_final_scores(self):
        """阶段 3: 合成最终索引并注入双向覆盖索引"""
        print("--- Step 3/3: 合成最终索引 (S_total) ---")

        print("-> 3a: 写入直接协作分基数...")
        self.conn.execute("INSERT INTO scholar_collaboration SELECT aid1, aid2, s_val FROM direct_scores")
        self.conn.commit()

        print("-> 3b: 累加间接协作分 (Bridge)...")
        all_aids = [r[0] for r in self.conn.execute("SELECT DISTINCT aid1 FROM direct_scores").fetchall()]
        batch_size = 800

        pbar = tqdm(total=len(all_aids), desc="Synthesizing Indirect")
        for i in range(0, len(all_aids), batch_size):
            batch_aids = all_aids[i:i + batch_size]
            placeholders = ','.join(['?'] * len(batch_aids))

            sql = f"""
                INSERT INTO scholar_collaboration (aid1, aid2, score)
                SELECT d1.aid1, d2.aid2,
                       0.3 * SUM((d1.s_val * d2.s_val) / ((d1.h2 + 1) * SQRT(d1.h1 + 1) * SQRT(d2.h2 + 1)))
                FROM direct_scores d1
                JOIN direct_scores d2 ON d1.aid2 = d2.aid1
                WHERE d1.aid1 IN ({placeholders}) AND d1.aid1 != d2.aid2
                GROUP BY d1.aid1, d2.aid2
                HAVING SUM((d1.s_val * d2.s_val)) > 0.01
                ON CONFLICT(aid1, aid2) DO UPDATE SET score = score + excluded.score
            """
            self.conn.execute(sql, batch_aids)
            self.conn.commit()
            pbar.update(len(batch_aids))
        pbar.close()

        print("-> 3c: 执行 Top-100 裁剪与全链路覆盖索引优化...")
        self.conn.executescript("""
                                CREATE TABLE scholar_collaboration_tmp AS
                                SELECT aid1, aid2, score
                                FROM (SELECT aid1,
                                             aid2,
                                             score,
                                             ROW_NUMBER() OVER (PARTITION BY aid1 ORDER BY score DESC) as rank
                                      FROM scholar_collaboration)
                                WHERE rank <= 100;

                                DROP TABLE scholar_collaboration;
                                ALTER TABLE scholar_collaboration_tmp RENAME TO scholar_collaboration;

                                -- 核心优化：建立双向覆盖索引，解决 UNION ALL 查询全表扫描问题
                                -- 直接包含 score，查询时无需读取数据行 (Index Only Scan)
                                CREATE UNIQUE INDEX IF NOT EXISTS idx_collab_covering_1 ON scholar_collaboration (aid1, aid2, score);
                                CREATE INDEX IF NOT EXISTS idx_collab_covering_2 ON scholar_collaboration (aid2, aid1, score);

                                PRAGMA
                                wal_checkpoint(TRUNCATE);
                                VACUUM;
                                ANALYZE; -- 更新统计信息，使规划器正确命中覆盖索引
                                """)
        print("--- 协作索引构建完成 ---")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    indexer = LocalSimilarityIndexer(DB_PATH, COLLAB_DB_PATH)
    try:
        start_time = datetime.now()
        indexer.step1_precompute_weights()
        indexer.step2_compute_direct_scores()
        indexer.step3_compute_final_scores()
        print(f"--- 耗时合计: {datetime.now() - start_time} ---")
    finally:
        indexer.close()