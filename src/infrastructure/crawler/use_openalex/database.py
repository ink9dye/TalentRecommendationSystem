import sqlite3
import threading
import logging
import time
from contextlib import contextmanager
from typing import List, Tuple
from src.infrastructure.crawler.use_openalex.config import DB_PATH
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
        """初始化表格：实现12属性主表、语义字典与作者画像的完整构建"""
        with self.connection() as conn:
            cursor = conn.cursor()

            # 1. 成果主表 (12个核心属性，含 concepts/keywords 文本)
            cursor.execute('''CREATE TABLE IF NOT EXISTS works
                        (
                            work_id TEXT PRIMARY KEY,
                            doi TEXT,
                            title TEXT,
                            year INTEGER,
                            publication_date TEXT,
                            citation_count INTEGER,
                            concepts_text TEXT,
                            keywords_text TEXT,
                            -- 移除 is_alphabetical INTEGER,
                            type TEXT,
                            language TEXT
                        )''')

            # 2. 摘要文本表 (1:1 剥离存储)
            cursor.execute('''CREATE TABLE IF NOT EXISTS abstracts
                              (work_id TEXT PRIMARY KEY, inverted_index TEXT, 
                               full_text_en TEXT,
                               FOREIGN KEY(work_id) REFERENCES works(work_id))''')

            # 3. 语义词汇表 (Vocabulary: 未来向量计算的核心)
            cursor.execute('''CREATE TABLE IF NOT EXISTS vocabulary
                              (
                                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                                  term TEXT UNIQUE,
                                  entity_type TEXT, -- "concept", "keyword" 或 "industry"
                                  vector BLOB
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
                                  0, -- 新增字段
                                  last_known_institution_id
                                  TEXT,
                                  last_updated TEXT
                              )''')

            # 5. 机构表
            cursor.execute('''CREATE TABLE IF NOT EXISTS institutions
                              (inst_id TEXT PRIMARY KEY, name TEXT, country TEXT, 
                               type TEXT, works_count INTEGER, cited_by_count INTEGER,last_updated TEXT)''')

            # 6. 人-作-所关系表
            # 6. 人-作-所-刊关系表：新增 source_id 字段
            cursor.execute('''CREATE TABLE IF NOT EXISTS authorships
            (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                work_id TEXT,
                author_id TEXT,
                inst_id TEXT,
                source_id TEXT, 
                pos_index INTEGER,
                author_position TEXT,
                is_corresponding INTEGER,
                is_alphabetical INTEGER,
                FOREIGN KEY(work_id) REFERENCES works(work_id),
                FOREIGN KEY(source_id) REFERENCES sources(source_id),
                FOREIGN KEY(author_id) REFERENCES authors(author_id))''')

            # 7. 成果来源渠道表 (Sources)
            cursor.execute('''CREATE TABLE IF NOT EXISTS sources
                              (
                                  source_id TEXT PRIMARY KEY,
                                  display_name TEXT,
                                  type TEXT,
                                  works_count INTEGER,
                                  cited_by_count INTEGER,
                                  last_updated TEXT
                              )''')
            # 8. 爬虫任务进度表 (增加 progress 和 is_completed 字段)
            cursor.execute('''CREATE TABLE IF NOT EXISTS crawl_states
            (
                field_id
                TEXT,
                phase
                INTEGER,
                cursor
                TEXT,
                progress
                INTEGER
                DEFAULT
                0,
                is_completed
                INTEGER
                DEFAULT
                0,
                last_updated
                TEXT,
                PRIMARY
                KEY
                              (
                field_id,
                phase
                              ))''')

            # 9. 精英作者画像进度表 (Phase 3 专用)
            cursor.execute('''CREATE TABLE IF NOT EXISTS author_process_states
            (
                field_id
                TEXT,
                author_id
                TEXT,
                phase
                INTEGER,
                PRIMARY
                KEY
                              (
                field_id,
                author_id,
                phase
                              ))''')
            #10,学科论文对应表，辅助爬取的
            cursor.execute('''CREATE TABLE IF NOT EXISTS work_fields
            (
                work_id
                TEXT,
                field_id
                TEXT,
                field_name
                TEXT,
                PRIMARY
                KEY
                              (
                work_id,
                field_id
                              ),
                FOREIGN KEY
                              (
                                  work_id
                              ) REFERENCES works
                              (
                                  work_id
                              )
                )''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_authorships_source ON authorships(source_id)')
            cursor.execute('''CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_rel
                ON authorships (work_id, author_id, inst_id, pos_index)''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_authorships_author ON authorships(author_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_authorships_inst ON authorships(inst_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_authorships_work ON authorships(work_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_authors_inst ON authors(last_known_institution_id)')

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
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
            self.conn.execute("PRAGMA journal_mode=WAL")
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
            cursor.execute("INSERT OR REPLACE INTO works VALUES (?,?,?,?,?,?,?,?,?,?)", work_row)  # 10个问号
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