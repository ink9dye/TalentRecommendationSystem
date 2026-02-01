import sqlite3
import pandas as pd
import os
import logging
# 保持原有的导入路径
from src.infrastructure.database.use_openalex.database import DatabaseManager

logger = logging.getLogger(__name__)


class UnifiedTalentDB(DatabaseManager):
    def __init__(self, db_name='academic_dataset_v5.db'):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.data_dir = os.path.join(base_dir, 'data')
        db_path = os.path.join(self.data_dir, db_name)

        super().__init__(db_path)
        self._add_job_table()
        logger.info(f"已成功连接并初始化数据库: {db_path}")

    def _add_job_table(self):
        """核心合并：在 SQLite 中追加行业岗位表（同步新增 qualification 字段）"""
        with self.connection() as conn:
            cursor = conn.cursor()
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
                                  qualification
                                  TEXT, -- 新增字段：学历要求
                                  keyword
                                  TEXT,
                                  crawl_time
                                  TEXT
                              )''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_keyword ON jobs(keyword)')
            # 新增学历索引，方便后续进行人才分层筛选
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_jobs_qual ON jobs(qualification)')
            conn.commit()

    def merge_csv_to_db(self, csv_filename='jobs.csv'):
        """
        核心逻辑：将 data 目录下的 jobs.csv 迁移进 SQLite 数据库
        包含：学历二次校验、数据去重、增量入库、技能词库同步
        """
        csv_path = os.path.join(self.data_dir, csv_filename)

        if not os.path.exists(csv_path):
            logger.error(f"未发现数据源文件: {csv_path}")
            return

        try:
            # 1. 读取 CSV 数据
            df = pd.read_csv(csv_path)

            # --- 【关键修改：学历强制过滤】 ---
            # 确保即使 CSV 中存在其他学历数据（如本科），也不会进入数据库
            target_qualifications = ['硕士', '博士']
            df = df[df['qualification'].isin(target_qualifications)]
            # -------------------------------

            # 2. DataFrame 内部去重
            # 这里的 drop_duplicates 确保如果爬虫多次运行产生了重复行，入库前先清理
            df = df.drop_duplicates(subset=['securityId'])

            # 3. 增量迁移岗位数据（防止主键冲突）
            with self.connection() as conn:
                # 获取数据库中已有的 ID 以实现增量更新
                existing_ids_query = "SELECT securityId FROM jobs"
                try:
                    existing_ids = pd.read_sql(existing_ids_query, conn)['securityId'].astype(str).tolist()
                except Exception:
                    # 如果表不存在或为空，则 existing_ids 为空列表
                    existing_ids = []

                # 仅保留数据库中不存在的新数据
                new_df = df[~df['securityId'].astype(str).isin(existing_ids)]

                if not new_df.empty:
                    # 使用 append 模式，因为已经通过上面的逻辑过滤掉了重复 ID
                    new_df.to_sql('jobs', conn, if_exists='append', index=False)
                    logger.info(f"成功存入 {len(new_df)} 条新岗位数据（目标学历：硕/博）。")
                else:
                    logger.info("没有新的目标学历岗位数据需要更新。")

            # 4. 提取行业技能词同步到 vocabulary 表
            # 这里的逻辑是将 jobs.csv 里的技能提取出来，打上 'industry' 标签，实现数据融合
            if 'skills' in df.columns:
                # 兼容中英文逗号，并处理爆炸拆分后的空白字符
                all_skills = (df['skills'].dropna()
                              .str.replace('，', ',')
                              .str.split(',')
                              .explode()
                              .str.strip()
                              .unique())
                vocab_data = [(str(s), 'industry') for s in all_skills if s]

                with self.connection() as conn:
                    # INSERT OR IGNORE 保证了 term 的唯一性
                    conn.executemany(
                        "INSERT OR IGNORE INTO vocabulary (term, entity_type) VALUES (?, ?)",
                        vocab_data
                    )
                    conn.commit()
                logger.info(f"已同步 {len(vocab_data)} 个行业技能标签至 vocabulary 表。")

        except Exception as e:
            logger.error(f"合并 CSV 时出错: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    talent_db = UnifiedTalentDB()
    talent_db.merge_csv_to_db()