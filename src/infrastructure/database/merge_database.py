import sqlite3
import pandas as pd
import os
import logging
from database import DatabaseManager  # 引用你提供的 database.py 类

logger = logging.getLogger(__name__)


class UnifiedTalentDB(DatabaseManager):
    def __init__(self, db_path=None):
        # 1. 执行原有的 10 张表初始化逻辑（含 authors, works, vocabulary 等）
        super().__init__(db_path)
        self._add_job_table()

    def _add_job_table(self):
        """核心合并：在 SQLite 中追加行业岗位表"""
        with self.connection() as conn:
            cursor = conn.cursor()
            # 增加 jobs 表，字段完全对应你的爬虫采集数据
            cursor.execute('''CREATE TABLE IF NOT EXISTS jobs
                              (
                                  securityId
                                  TEXT
                                  PRIMARY
                                  KEY,
                                  job_name
                                  TEXT,
                                  salary
                                  TEXT,
                                  skills
                                  TEXT,
                                  description
                                  TEXT,
                                  company
                                  TEXT,
                                  city
                                  TEXT,
                                  keyword
                                  TEXT,
                                  crawl_time
                                  TEXT
                              )''')
            # 为标签路召回的核心字段建立索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_keyword ON jobs(keyword)')
            logger.info("工业侧 jobs 表及索引已合并至数据库。")

    def _extract_and_sync_skills(self, skills_str: str):
        """私有方法：将岗位技能提取并打上 'industry' 标签存入 vocabulary"""
        if not skills_str:
            return

        # 统一处理中英文逗号，拆分技能词
        skills = [s.strip() for s in skills_str.replace('，', ',').split(',') if s.strip()]
        vocab_data = [(s, 'industry') for s in skills]

        with self.connection() as conn, self.lock:
            # 使用 INSERT OR IGNORE 确保 industry 类型词汇与已有的 concept/keyword 不冲突
            conn.executemany(
                "INSERT OR IGNORE INTO vocabulary (term, entity_type) VALUES (?, ?)",
                vocab_data
            )

    def merge_csv_to_db(self, csv_path='jobs.csv'):
        """Merge 逻辑：将历史 CSV 数据迁移进 SQLite，并同步构建词库"""
        if not os.path.exists(csv_path):
            logger.warning(f"未发现 {csv_path}，跳过历史数据合并。")
            return

        try:
            df = pd.read_csv(csv_path)

            # 1. 迁移岗位数据
            with self.connection() as conn, self.lock:
                df.to_sql('jobs', conn, if_exists='append', index=False, method='multi')

            # 2. 批量提取行业技能词（Industry）
            # 使用 Pandas explode 快速处理所有职位的 skills 字段
            all_skills = df['skills'].dropna().str.replace('，', ',').str.split(',').explode().str.strip().unique()
            vocab_data = [(str(s), 'industry') for s in all_skills if s]

            with self.connection() as conn, self.lock:
                conn.executemany(
                    "INSERT OR IGNORE INTO vocabulary (term, entity_type) VALUES (?, ?)",
                    vocab_data
                )

            logger.info(f"成功合并 {len(df)} 条岗位及 {len(vocab_data)} 个行业技能词。")
        except Exception as e:
            logger.error(f"合并 CSV 时出错: {e}")

    def save_single_job(self, record: dict):
        """同步更新：在保存职位的瞬间，将技能词注入 vocabulary"""
        with self.connection() as conn, self.lock:
            conn.execute('''INSERT OR REPLACE INTO jobs 
                           (securityId, job_name, salary, skills, description, company, city, keyword, crawl_time) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                         (record['securityId'], record['job_name'], record['salary'],
                          record['skills'], record['description'], record['company'],
                          record['city'], record['keyword'], record['crawl_time']))

        # 提取并同步词汇
        self._extract_and_sync_skills(record.get('skills', ''))