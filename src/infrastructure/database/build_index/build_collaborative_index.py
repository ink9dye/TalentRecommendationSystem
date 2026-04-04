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
    def calculate(pos_index: int, is_corr: int, is_alpha: int, pub_year: int, base_year: int) -> float:
        """
        计算单篇论文的贡献权重。
        :param base_year: 统一的计算基准年，防止权重随运行时间漂移
        """
        # 容错处理：若无年份则默认 2000 年
        effective_pub_year = pub_year if pub_year else 2000
        delta_t = max(0, base_year - effective_pub_year)

        # 指数时间衰减
        time_weight = math.exp(-0.1 * delta_t)

        # 贡献度基数：考虑署名顺位与通讯作者加成 [cite: 2, 3]
        # 第一作者(非字母排序)加 0.5，通讯作者加 0.5 [cite: 2]
        contribution = 1.0 + (0.5 if is_alpha == 0 and pos_index == 1 else 0) + (0.5 if is_corr == 1 else 0)

        return round(contribution * time_weight, 4)


class LocalSimilarityIndexer:
    def __init__(self, db_path, collab_db_path, base_year=None):
        self.db_path = db_path
        self.collab_db_path = collab_db_path
        # 若不指定基准年，则取当前年份，并在整个运行期间固定
        self.base_year = base_year if base_year else datetime.now().year
        self._prepare_db()

    def _prepare_db(self):
        """初始化协作索引数据库结构并配置磁盘存储策略"""
        db_dir = os.path.dirname(os.path.abspath(self.collab_db_path))
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

        self.conn = sqlite3.connect(self.collab_db_path)

        # 核心优化：启用 WAL 模式与内存缓存
        self.conn.executescript(f"""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=OFF;
            PRAGMA cache_size=200000; 
            PRAGMA temp_store = MEMORY; 

            -- 阶段1结果：缓存单人贡献
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

            -- 阶段3结果：合成 Top-100 最终索引 
            DROP TABLE IF EXISTS scholar_collaboration;
            CREATE TABLE scholar_collaboration (
                aid1 TEXT, 
                aid2 TEXT, 
                score REAL,
                PRIMARY KEY (aid1, aid2)
            ) WITHOUT ROWID;
        """)

    def step1_precompute_weights(self):
        """阶段 1: 计算权重。融合引用因子 (ln 转换)。"""
        print(f"--- Step 1/3: 计算权重 (基准年: {self.base_year}) ---")
        src_conn = sqlite3.connect(self.db_path)
        sql = """
              SELECT a.work_id, \
                     a.author_id, \
                     a.pos_index, \
                     a.is_corresponding,
                     a.is_alphabetical, \
                     w.year, \
                     au.h_index, \
                     w.citation_count
              FROM authorships a
                       JOIN works w ON a.work_id = w.work_id
                       JOIN authors au ON a.author_id = au.author_id
              """
        cursor = src_conn.execute(sql)

        batch = []
        for row in tqdm(cursor, desc="Processing Individual Weights"):
            # 使用统一基准年计算衰减
            base_w = WeightStrategy.calculate(row[2], row[3], row[4], row[5], self.base_year)

            # 融合引用因子：sqrt(ln(cite + e)) 确保量级稳定
            cite_val = row[7] if row[7] else 0
            cite_factor = math.sqrt(math.log(cite_val + 2.71828))
            composite_weight = base_w * cite_factor

            batch.append((row[0], row[1], composite_weight, row[6]))
            if len(batch) >= 50000:
                self.conn.executemany("INSERT INTO weighted_authorships VALUES (?,?,?,?)", batch)
                batch = []

        if batch:
            self.conn.executemany("INSERT INTO weighted_authorships VALUES (?,?,?,?)", batch)
        self.conn.commit()

        print("-> 建立中间查询索引...")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_wa_work ON weighted_authorships(work_id)")
        src_conn.close()

    def step2_compute_direct_scores(self):
        """阶段 2: 分批计算直接协作分 (S_direct) """
        print("--- Step 2/3: 计算直接协作分 ---")

        # 排除孤立作品和超大规模合作作品（降噪）
        work_stats = self.conn.execute("""
                                       SELECT work_id
                                       FROM weighted_authorships
                                       GROUP BY work_id
                                       HAVING COUNT(author_id) BETWEEN 2 AND 99
                                       """).fetchall()

        all_works = [r[0] for r in work_stats]
        pbar = tqdm(total=len(all_works), desc="Computing Direct Scores")

        batch_data = []
        for i, wid in enumerate(all_works):
            authors = self.conn.execute(
                "SELECT author_id, weight, h_index FROM weighted_authorships WHERE work_id = ?",
                (wid,)
            ).fetchall()

            n = len(authors)
            for idx1 in range(n):
                for idx2 in range(idx1 + 1, n):
                    a1, w1, h1 = authors[idx1]
                    a2, w2, h2 = authors[idx2]
                    # 排序 ID 确保 (a1, a2) 唯一性
                    if a1 > a2: a1, a2, h1, h2, w1, w2 = a2, a1, h2, h1, w2, w1

                    score = w1 * w2
                    if score > 0.01:
                        batch_data.append((a1, a2, score, h1, h2))

            if len(batch_data) >= 20000:
                self.conn.executemany("""
                                      INSERT INTO direct_scores (aid1, aid2, s_val, h1, h2)
                                      VALUES (?, ?, ?, ?, ?) ON CONFLICT(aid1, aid2) DO
                                      UPDATE SET s_val = s_val + excluded.s_val
                                      """, batch_data)
                self.conn.commit()
                batch_data = []

            if i % 10000 == 0:
                gc.collect()
            pbar.update(1)

        if batch_data:
            self.conn.executemany("""
                                  INSERT INTO direct_scores
                                  VALUES (?, ?, ?, ?, ?) ON CONFLICT(aid1, aid2) DO
                                  UPDATE SET s_val = s_val + excluded.s_val
                                  """, batch_data)
            self.conn.commit()
        pbar.close()

    def step3_compute_final_scores(self):
        """阶段 3: 合成 S_total (直接 + 0.3 * 间接) 并进行 Top-100 裁剪 """
        print("--- Step 3/3: 合成最终索引 (S_total) ---")

        # 3a. 基础分注入
        self.conn.execute("INSERT INTO scholar_collaboration SELECT aid1, aid2, s_val FROM direct_scores")
        self.conn.commit()

        # 3b. 间接协作分 (Bridge 算法)
        all_aids = [r[0] for r in self.conn.execute("SELECT DISTINCT aid1 FROM direct_scores").fetchall()]
        pbar = tqdm(total=len(all_aids), desc="Adding Indirect Links")

        for i in range(0, len(all_aids), 500):
            batch = all_aids[i:i + 500]
            placeholders = ','.join(['?'] * len(batch))

            # 引入 alpha=0.3 的间接影响因子，并使用 H-index 进行正则化惩罚
            sql = f"""
                INSERT INTO scholar_collaboration (aid1, aid2, score)
                SELECT d1.aid1, d2.aid2,
                       0.3 * SUM((d1.s_val * d2.s_val) / ((d1.h2 + 1) * SQRT(d1.h1 + 1) * SQRT(d2.h2 + 1)))
                FROM direct_scores d1
                JOIN direct_scores d2 ON d1.aid2 = d2.aid1
                WHERE d1.aid1 IN ({placeholders}) AND d1.aid1 != d2.aid2
                GROUP BY d1.aid1, d2.aid2
                ON CONFLICT(aid1, aid2) DO UPDATE SET score = score + excluded.score
            """
            self.conn.execute(sql, batch)
            self.conn.commit()
            pbar.update(len(batch))
        pbar.close()

        # 3c. 裁剪与覆盖索引优化
        print("-> 执行 Top-100 裁剪与覆盖索引构建...")
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

                                -- 构建双向覆盖索引以支持高效召回
                                CREATE UNIQUE INDEX idx_collab_covering_1 ON scholar_collaboration (aid1, aid2, score);
                                CREATE INDEX idx_collab_covering_2 ON scholar_collaboration (aid2, aid1, score);

                                VACUUM;
                                ANALYZE;
                                """)
        print("--- 协作索引构建完成 ---")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    indexer = LocalSimilarityIndexer(DB_PATH, COLLAB_DB_PATH)
    try:
        start_t = datetime.now()
        indexer.step1_precompute_weights()
        indexer.step2_compute_direct_scores()
        indexer.step3_compute_final_scores()
        print(f"--- 耗时合计: {datetime.now() - start_t} ---")
    finally:
        indexer.close()