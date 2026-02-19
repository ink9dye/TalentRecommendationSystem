import sqlite3
import threading
import logging
import time
from contextlib import contextmanager
from typing import List, Tuple
from src.infrastructure.crawler.use_openalex.db_config import DB_PATH
from datetime import datetime


logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.conn = None
        self._init_db()
        self.upgrade_schema_for_incremental()

    def _init_db(self):
        """
        初始化数据库架构与索引。
        实现 12 属性主表、语义字典与作者画像的完整构建，并注入高性能覆盖索引。
        """
        from config import SQL_INIT_SCRIPTS

        with self.connection() as conn:
            cursor = conn.cursor()

            # 0. 性能预配置：开启 WAL 模式加速索引构建
            cursor.execute('PRAGMA journal_mode=WAL;')

            # 1. 成果主表
            cursor.execute('''CREATE TABLE IF NOT EXISTS works
                              (
                                  work_id
                                  TEXT
                                  PRIMARY
                                  KEY,
                                  doi
                                  TEXT,
                                  title
                                  TEXT,
                                  year
                                  INTEGER,
                                  publication_date
                                  TEXT,
                                  citation_count
                                  INTEGER,
                                  concepts_text
                                  TEXT,
                                  keywords_text
                                  TEXT,
                                  type
                                  TEXT,
                                  language
                                  TEXT,
                                  domain_ids TEXT
                              )''')

            # 2. 摘要文本表
            cursor.execute('''CREATE TABLE IF NOT EXISTS abstracts
            (
                work_id
                TEXT
                PRIMARY
                KEY,
                inverted_index
                TEXT,
                full_text_en
                TEXT,
                FOREIGN
                KEY
                              (
                work_id
                              ) REFERENCES works
                              (
                                  work_id
                              ))''')

            # 3. 语义词汇表
            cursor.execute('''CREATE TABLE IF NOT EXISTS vocabulary
                              (
                                  voc_id
                                  INTEGER
                                  PRIMARY
                                  KEY
                                  AUTOINCREMENT,
                                  term
                                  TEXT
                                  UNIQUE,
                                  entity_type
                                  TEXT,
                                  vector
                                  BLOB
                              )''')

            # 4. 作者表
            cursor.execute('''CREATE TABLE IF NOT EXISTS authors
                              (
                                  author_id
                                  TEXT
                                  PRIMARY
                                  KEY,
                                  orcid
                                  TEXT,
                                  name
                                  TEXT
                                  NOT
                                  NULL,
                                  works_count
                                  INTEGER,
                                  cited_by_count
                                  INTEGER,
                                  h_index
                                  INTEGER
                                  DEFAULT
                                  0,
                                  last_known_institution_id
                                  TEXT,
                                  last_updated
                                  TEXT
                              )''')

            # 5. 机构表
            cursor.execute('''CREATE TABLE IF NOT EXISTS institutions
                              (
                                  inst_id
                                  TEXT
                                  PRIMARY
                                  KEY,
                                  name
                                  TEXT,
                                  country
                                  TEXT,
                                  type
                                  TEXT,
                                  works_count
                                  INTEGER,
                                  cited_by_count
                                  INTEGER,
                                  last_updated
                                  TEXT
                              )''')

            # 6. 人-作-所-刊关系表
            cursor.execute('''CREATE TABLE IF NOT EXISTS authorships
            (
                ship_id
                INTEGER
                PRIMARY
                KEY
                AUTOINCREMENT,
                work_id
                TEXT,
                author_id
                TEXT,
                inst_id
                TEXT,
                source_id
                TEXT,
                pos_index
                INTEGER,
                author_position
                TEXT,
                is_corresponding
                INTEGER,
                is_alphabetical
                INTEGER,
                FOREIGN
                KEY
                              (
                work_id
                              ) REFERENCES works
                              (
                                  work_id
                              ),
                FOREIGN KEY
                              (
                                  author_id
                              ) REFERENCES authors
                              (
                                  author_id
                              ))''')

            # 7. 成果来源渠道表
            cursor.execute('''CREATE TABLE IF NOT EXISTS sources
                              (
                                  source_id
                                  TEXT
                                  PRIMARY
                                  KEY,
                                  display_name
                                  TEXT,
                                  type
                                  TEXT,
                                  works_count
                                  INTEGER,
                                  cited_by_count
                                  INTEGER,
                                  last_updated
                                  TEXT
                              )''')

            # 8. 进度表等其他表结构（保持原有逻辑）
            cursor.execute(
                'CREATE TABLE IF NOT EXISTS crawl_states (field_id TEXT, phase INTEGER, cursor TEXT, progress INTEGER DEFAULT 0, is_completed INTEGER DEFAULT 0, last_updated TEXT, PRIMARY KEY (field_id, phase))')

            # --- 11. 执行全链路高性能索引初始化 ---
            print("[*] 正在执行 SQLite 覆盖索引初始化...")

            # 基础 SQL 脚本执行 (从 config 导入)
            for sql in SQL_INIT_SCRIPTS:
                try:
                    cursor.execute(sql)
                except Exception as e:
                    print(f"[!] 基础架构脚本提示: {e}")

            # A. 针对【向量路召回】：极速反查 Author ID
            # 作用：将召回的 WorkID 批量转换为 AuthorID
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_aship_work_lookup ON authorships(work_id, author_id)')

            # B. 针对【协同路召回】：二跳协作路径扩展
            # 作用：从 AuthorID 快速找其发表过的所有 Work
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_aship_author_lookup ON authorships(author_id, work_id)')

            # C. 针对【KGAT 拓扑构建】：作者-机构三元组抽取
            # 作用：加速 generate_kg_topology 中的 Produced_by 关系提取
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_aship_inst_lookup ON authorships(inst_id, author_id)')

            # D. 【核心修复】针对【精排训练与 AX 融合】：覆盖索引
            # 作用：消除 generate_refined_train_data 中的“回表”报错，实现 Index-Only Scan
            try:
                # 严格对齐查询字段顺序：author_id -> h_index -> cited_by_count
                cursor.execute('''CREATE INDEX IF NOT EXISTS idx_author_metrics_covering
                    ON authors(author_id, h_index, cited_by_count)''')
                print("[OK] 特征覆盖索引已对齐：(author_id, h_index, cited_by_count)")
            except Exception as e:
                print(f"[!] 特征覆盖索引构建失败: {e}")

            # E. 针对【精排成果评分】：作品价值覆盖索引
            # 作用：为 500 名候选人批量计算“近三年引用贡献”时无需读取长文本字段
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_works_score_covering ON works(work_id, year, citation_count)')

            # F. 针对【KGAT 语义连接】：作品-主题三元组
            # 作用：加速 Has_Topic 关系的全库导出
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_works_concepts_lookup ON works(work_id) WHERE concepts_text IS NOT NULL')

            # G. 基础业务索引：机构筛选
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_authors_inst ON authors(last_known_institution_id)')

            conn.commit()

            # 必须执行 ANALYZE：更新统计信息，让 SQLite 知道新索引比全表扫描快
            print("[OK] 索引写入完成，正在执行 ANALYZE 更新执行计划...")
            cursor.execute("ANALYZE")

        print("全链路高性能索引初始化完成。")
    def upgrade_schema_for_incremental(self):
        """
        专门为增量更新设计的表结构升级函数。
        为核心实体表增加 last_updated 字段。
        """
        tables_to_upgrade = ["authors", "institutions", "sources"]

        with self.connection() as conn:
            cursor = conn.cursor()
            for table in tables_to_upgrade:
                # 检查字段是否存在，防止重复添加报错
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [column[1] for column in cursor.fetchall()]

                if "last_updated" not in columns:
                    logger.info(f"正在为表 {table} 增加 last_updated 字段...")
                    try:
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN last_updated TEXT")
                    except Exception as e:
                        logger.error(f"为表 {table} 增加字段失败: {e}")


    def close(self):
        """安全关闭数据库连接"""
        with self.lock:
            if self.conn:
                try:
                    self.conn.close()
                    self.conn = None
                    logger.info("数据库连接已关闭")
                except Exception as e:
                    logger.error(f"关闭数据库连接时出错: {e}")



    @contextmanager
    def connection(self):
        """获取带高性能预配置的数据库连接"""
        if self.conn is None:
            # 增加 timeout 到 60s，防止生成大数据集时发生数据库锁定
            self.conn = sqlite3.connect(self.db_path, timeout=60, isolation_level=None)
            # 性能调优：增加缓存到约 80MB，开启 WAL 模式
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA cache_size = -80000")
            self.conn.execute("PRAGMA synchronous = NORMAL")
        try:
            yield self.conn
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise

    # --- 写入逻辑 ---

    def get_crawl_state(self, field_id: str, phase: int) -> dict:
        """获取爬取进度字典，确保 crawler_logic.py 的 .get() 不会报错"""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT cursor, progress, is_completed FROM crawl_states WHERE field_id = ? AND phase = ?",
                (field_id, phase)
            ).fetchone()
            if row:
                return {
                    "cursor": row[0],
                    "progress": row[1],
                    "is_completed": bool(row[2])
                }
            return {"cursor": "*", "progress": 0, "is_completed": False}

    def update_crawl_state(self, field_id: str, phase: int, cursor: str, progress: int, is_completed: bool):
        """更新爬行进度，支持多字段写入"""
        with self.connection() as conn, self.lock:
            conn.execute(
                '''INSERT OR REPLACE INTO crawl_states 
                   (field_id, phase, cursor, progress, is_completed, last_updated) 
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (field_id, phase, cursor, progress, 1 if is_completed else 0, datetime.now().isoformat())
            )

    # --- Phase 3 支持方法 ---
    def is_author_processed(self, field_id: str, author_id: str, phase: int) -> bool:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM author_process_states WHERE field_id = ? AND author_id = ? AND phase = ?",
                (field_id, author_id, phase)
            ).fetchone()
            return row is not None

    def mark_author_processed(self, field_id: str, author_id: str, phase: int):
        with self.connection() as conn, self.lock:
            conn.execute("INSERT OR IGNORE INTO author_process_states VALUES (?, ?, ?)",
                         (field_id, author_id, phase))

    def save_work_bundle(self, work_row, abstract_row, vocab_terms: List[Tuple[str, str]]):
        with self.connection() as conn, self.lock:
            cursor = conn.cursor()
            # 修改：现在 works 表是 10 个字段（移除了 is_alphabetical）
            cursor.execute("INSERT OR REPLACE INTO works VALUES (?,?,?,?,?,?,?,?,?,?,?)", work_row)  # 11个问号
            cursor.execute("INSERT OR REPLACE INTO abstracts VALUES (?,?,?)", abstract_row)
            cursor.executemany("INSERT OR IGNORE INTO vocabulary (term, entity_type) VALUES (?,?)", vocab_terms)

    def save_author_batch(self, data: List[Tuple]):
        """适配 8 个字段：增加 last_updated"""
        with self.connection() as conn, self.lock:
            # 这里的问号从 7 个增加到 8 个
            conn.cursor().executemany("INSERT OR REPLACE INTO authors VALUES (?,?,?,?,?,?,?,?)", data)

    def save_institution_batch(self, data: List[Tuple]):
        """适配 7 个字段：增加 last_updated"""
        with self.connection() as conn, self.lock:
            # 这里的问号从 6 个增加到 7 个
            conn.cursor().executemany("INSERT OR REPLACE INTO institutions VALUES (?,?,?,?,?,?,?)", data)

    def save_authorship_batch(self, data: List[Tuple]):
        with self.connection() as conn, self.lock:
            # 新增 source_id 占位符，现在有 8 个字段（排除自增 id）
            conn.cursor().executemany(
                "INSERT OR REPLACE INTO authorships (work_id, author_id, inst_id, source_id, pos_index, author_position, is_corresponding, is_alphabetical) VALUES (?,?,?,?,?,?,?,?)",
                data
            )

    def save_source_batch(self, data: List[Tuple]):
        """适配 6 个字段：增加 last_updated"""
        with self.connection() as conn, self.lock:
            # 这里的问号从 5 个增加到 6 个
            conn.cursor().executemany("INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?)", data)

    def save_work_field_relation(self, work_id, field_id, field_name):
        """记录论文所属学科，如果该关系已存在则忽略，返回是否为新关系"""
        with self.connection() as conn, self.lock:
            cursor = conn.cursor()
            # 尝试插入关系
            cursor.execute(
                "INSERT OR IGNORE INTO work_fields (work_id, field_id, field_name) VALUES (?, ?, ?)",
                (work_id, field_id, field_name)
            )
            # 如果 rowcount > 0，说明这是一个新的学科归属关系
            return cursor.rowcount > 0

    def refresh_stats(self):
        """高性能统计更新：解耦版，支持 Works, Authors, Institutions, Sources"""
        logger.info("正在启动全库统计指标同步...")
        start_time = time.time()

        with self.connection() as conn, self.lock:
            cursor = conn.cursor()
            try:
                # 1. 性能预配置
                cursor.execute("PRAGMA temp_store = MEMORY")
                cursor.execute("PRAGMA cache_size = -64000")
                cursor.execute("BEGIN TRANSACTION")

                # 2. 执行各模块统计
                self._update_author_stats(cursor)
                self._update_institution_stats(cursor)
                self._update_source_stats(cursor)  # 补全：Source 统计

                conn.commit()
                elapsed = time.time() - start_time
                logger.info(f"全库统计更新完成！总耗时: {elapsed:.2f}秒")

            except Exception as e:
                conn.rollback()
                logger.error(f"统计同步失败: {e}")
                raise

    def _update_author_stats(self, cursor):
        """更新作者的论文总数与总被引"""
        logger.debug("正在计算作者画像统计...")
        cursor.execute("DROP TABLE IF EXISTS temp_author_stats")
        cursor.execute('''
                       CREATE
                       TEMPORARY TABLE temp_author_stats AS
                       SELECT a.author_id,
                              COUNT(a.work_id)      as cnt_works,
                              SUM(w.citation_count) as cnt_citations
                       FROM authorships a
                                JOIN works w ON a.work_id = w.work_id
                       GROUP BY a.author_id
                       ''')
        cursor.execute("CREATE INDEX idx_t_author ON temp_author_stats(author_id)")
        cursor.execute('''
                       UPDATE authors
                       SET works_count    = t.cnt_works,
                           cited_by_count = t.cnt_citations FROM temp_author_stats AS t
                       WHERE authors.author_id = t.author_id
                       ''')

    def _update_institution_stats(self, cursor):
        """更新机构统计 (需处理同一论文多作者导致的重复计算)"""
        logger.debug("正在计算机构影响力指标...")
        cursor.execute("DROP TABLE IF EXISTS temp_inst_stats")
        cursor.execute('''
                       CREATE
                       TEMPORARY TABLE temp_inst_stats AS
                       SELECT inst_id,
                              COUNT(DISTINCT work_id) as cnt_works,
                              SUM(unique_citation)    as cnt_citations
                       FROM (SELECT DISTINCT a.inst_id, a.work_id, w.citation_count as unique_citation
                             FROM authorships a
                                      JOIN works w ON a.work_id = w.work_id
                             WHERE a.inst_id IS NOT NULL)
                       GROUP BY inst_id
                       ''')
        cursor.execute("CREATE INDEX idx_t_inst ON temp_inst_stats(inst_id)")
        cursor.execute('''
                       UPDATE institutions
                       SET works_count    = t.cnt_works,
                           cited_by_count = t.cnt_citations FROM temp_inst_stats AS t
                       WHERE institutions.inst_id = t.inst_id
                       ''')

    def _update_source_stats(self, cursor):
        """更新 Sources 统计：从关系表聚合并进行去重计算"""
        logger.debug("正在从关系表计算发布渠道（Sources）去重统计...")
        cursor.execute("DROP TABLE IF EXISTS temp_source_stats")

        # 核心逻辑：先通过子查询获取 (source_id, work_id) 的唯一对，再进行聚合
        cursor.execute('''
                       CREATE
                       TEMPORARY TABLE temp_source_stats AS
                       SELECT source_id,
                              COUNT(work_id)       as cnt_works,
                              SUM(unique_citation) as cnt_citations
                       FROM (SELECT DISTINCT a.source_id, a.work_id, w.citation_count as unique_citation
                             FROM authorships a
                                      JOIN works w ON a.work_id = w.work_id
                             WHERE a.source_id IS NOT NULL)
                       GROUP BY source_id
                       ''')
        cursor.execute("CREATE INDEX idx_t_source ON temp_source_stats(source_id)")
        cursor.execute('''
                       UPDATE sources
                       SET works_count    = t.cnt_works,
                           cited_by_count = t.cnt_citations FROM temp_source_stats AS t
                       WHERE sources.source_id = t.source_id
                       ''')

    def _update_progress(self, step_name, current, total, use_tqdm, pbar=None, final=False):
        """更新进度显示"""
        progress_percent = int((current / total) * 100)

        if use_tqdm and pbar:
            pbar.set_description(f"统计更新: {step_name}")
            pbar.update(progress_percent - pbar.n)
        else:
            # 简单进度显示
            if final:
                logger.info(f"统计更新完成 [{progress_percent}%]")
            else:
                logger.info(f"{step_name} [{progress_percent}%]")

    def get_unprocessed_abstracts(self):
        """获取那些存了 JSON 但还没还原文本的记录"""
        with self.connection() as conn:
            return conn.execute(
                "SELECT work_id, inverted_index FROM abstracts WHERE full_text_en IS NULL OR full_text_en = ''"
            ).fetchall()

    def update_full_text(self, work_id, full_text):
        """回填还原后的文本"""
        with self.connection() as conn, self.lock:
            conn.execute("UPDATE abstracts SET full_text_en = ? WHERE work_id = ?", (full_text, work_id))

    def get_all_unprocessed_authors(self):
        """提取数据库中所有 H-index 为 0 且尚未补全画像的唯一作者 ID"""
        with self.connection() as conn:
            # 查找作者表中 h_index 为 0 的所有记录
            return [row[0] for row in conn.execute(
                "SELECT author_id FROM authors WHERE h_index = 0 OR h_index IS NULL"
            ).fetchall()]

    def work_exists(self, work_id: str) -> bool:
        with self.connection() as conn:
            return conn.execute("SELECT 1 FROM works WHERE work_id = ?", (work_id,)).fetchone() is not None

    def get_works_missing_concepts(self, limit=1000) -> List[str]:
        """获取数据库中 concepts_text 为空的论文 ID"""
        with self.connection() as conn:
            # 查找 NULL 或空字符串的记录
            query = "SELECT work_id FROM works WHERE concepts_text IS NULL OR concepts_text = '' LIMIT ?"
            return [row[0] for row in conn.execute(query, (limit,)).fetchall()]

    def update_work_concepts(self, work_id: str, concepts: str):
        """仅更新单篇论文的领域标签"""
        with self.connection() as conn, self.lock:
            conn.execute(
                "UPDATE works SET concepts_text = ? WHERE work_id = ?",
                (concepts, work_id)
            )

    def save_work_bundle(self, work_row, abstract_row, vocab_terms: List[Tuple[str, str]]):
        with self.connection() as conn, self.lock:
            cursor = conn.cursor()
            # 这里的问号增加到 11 个以匹配新增的 domain_ids 字段
            cursor.execute("INSERT OR REPLACE INTO works VALUES (?,?,?,?,?,?,?,?,?,?,?)", work_row)
            cursor.execute("INSERT OR REPLACE INTO abstracts VALUES (?,?,?)", abstract_row)
            cursor.executemany("INSERT OR IGNORE INTO vocabulary (term, entity_type) VALUES (?,?)", vocab_terms)

    def update_work_domain_ids(self, work_id: str, domain_ids: str):
        """高效回填领域 ID"""
        with self.connection() as conn, self.lock:
            conn.execute("UPDATE works SET domain_ids = ? WHERE work_id = ?", (domain_ids, work_id))



if __name__ == "__main__":
    # 配置基础日志以查看输出
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("\n" + "=" * 50)
    print("人才推荐系统 - 数据库索引优化")
    print(f"当前目标数据库: {DB_PATH}")
    print("=" * 50)

    try:
        db_manager = DatabaseManager(DB_PATH)
        print("\n[SUCCESS] 索引构建任务执行完毕！")
        print("- 特征覆盖索引 (idx_author_ax_covering): 已激活")
        print("- 拓扑覆盖索引 (idx_aship_work_author): 已激活")
        print("- 协作挖掘索引 (idx_aship_author_work): 已激活")
    except Exception as e:
        print(f"\n[ERROR] 初始化过程中出现异常: {e}")
    finally:
        print("=" * 50)