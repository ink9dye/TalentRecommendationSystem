import pandas as pd
import os
import logging

from src.infrastructure.crawler.use_openalex.database import DatabaseManager
from src.utils.tools import extract_skills

logger = logging.getLogger(__name__)


class UnifiedTalentDB(DatabaseManager):

    def __init__(self, db_name='academic_dataset_v5.db'):

        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )
        )

        self.data_dir = os.path.join(base_dir, 'data')
        db_path = os.path.join(self.data_dir, db_name)

        super().__init__(db_path)

        self._add_job_table()

        # 新增：确保 vocabulary.term 唯一
        self._ensure_vocab_index()

        logger.info(f"已成功连接并初始化数据库: {db_path}")


    def _ensure_vocab_index(self):
        """确保 vocabulary.term 唯一，避免重复 skill"""

        with self.connection() as conn:

            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_vocab_term
                ON vocabulary(term)
            """)

            conn.commit()

        logger.info("已确保 vocabulary.term 唯一索引存在")


    def _add_job_table(self):
        """在 SQLite 中追加岗位表"""

        with self.connection() as conn:

            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS jobs
                (
                    securityId TEXT PRIMARY KEY,
                    job_name TEXT,
                    salary TEXT,
                    skills TEXT,
                    description TEXT,
                    company TEXT,
                    city TEXT,
                    qualification TEXT,
                    keyword TEXT,
                    crawl_time TEXT
                )
            ''')

            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_jobs_keyword ON jobs(keyword)'
            )

            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_jobs_qual ON jobs(qualification)'
            )

            conn.commit()


    def merge_csv_to_db(self, csv_filename='jobs.csv'):

        csv_path = os.path.join(self.data_dir, csv_filename)

        if not os.path.exists(csv_path):
            logger.error(f"未发现数据源文件: {csv_path}")
            return

        try:

            # 1. 读取 CSV
            df = pd.read_csv(csv_path)

            # 2. 学历过滤
            target_qualifications = ['硕士', '博士']
            df = df[df['qualification'].isin(target_qualifications)]

            # 3. CSV 内部去重
            df = df.drop_duplicates(subset=['securityId'])


            # 4. 找出新增岗位
            with self.connection() as conn:

                try:
                    existing_ids = pd.read_sql(
                        "SELECT securityId FROM jobs",
                        conn
                    )['securityId'].astype(str).tolist()

                except Exception:
                    existing_ids = []

                new_df = df[~df['securityId'].astype(str).isin(existing_ids)]

                if not new_df.empty:

                    new_df.to_sql(
                        'jobs',
                        conn,
                        if_exists='append',
                        index=False
                    )

                    logger.info(
                        f"成功存入 {len(new_df)} 条新岗位数据（目标学历：硕/博）。"
                    )

                else:
                    logger.info("没有新的岗位需要更新。")


            # 5. skill 同步 vocabulary
            if 'skills' in df.columns:

                skill_set = set()

                for text in new_df['skills'].dropna():

                    skills = extract_skills(text)

                    skill_set.update(skills)

                vocab_data = [(s, 'industry') for s in skill_set]

                if vocab_data:

                    with self.connection() as conn:

                        conn.executemany(
                            "INSERT OR IGNORE INTO vocabulary (term, entity_type) VALUES (?, ?)",
                            vocab_data
                        )

                        conn.commit()

                logger.info(
                    f"已清洗并同步 {len(skill_set)} 个行业技能标签至 vocabulary 表。"
                )

        except Exception as e:

            logger.error(f"合并 CSV 时出错: {e}")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    talent_db = UnifiedTalentDB()

    talent_db.merge_csv_to_db()