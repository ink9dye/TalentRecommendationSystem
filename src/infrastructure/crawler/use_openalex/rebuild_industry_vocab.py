import logging
import pandas as pd

from config import DB_PATH
from src.infrastructure.crawler.use_openalex.database import DatabaseManager
from src.utils.tools import extract_skills

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rebuild_industry_vocab():

    db = DatabaseManager(DB_PATH)

    logger.info("开始重建 industry vocabulary")

    with db.connection() as conn:

        # 保证唯一索引
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_vocab_term
            ON vocabulary(term)
        """)

        # 删除旧 industry skill
        logger.info("删除旧 industry skill")

        conn.execute("""
            DELETE FROM vocabulary
            WHERE entity_type='industry'
        """)

        conn.commit()

        # 读取 jobs
        logger.info("读取 jobs 表")

        df = pd.read_sql(
            "SELECT skills FROM jobs WHERE skills IS NOT NULL",
            conn
        )

    logger.info(f"读取岗位数: {len(df)}")

    skill_set = set()

    # 解析 skill
    for text in df["skills"].dropna():

        skills = extract_skills(text)

        for s in skills:
            skill_set.add(s.lower())

    logger.info(f"解析得到 skill 数: {len(skill_set)}")

    vocab_data = [(s, "industry") for s in skill_set]

    # 写入 vocabulary
    with db.connection() as conn:

        conn.executemany(
            "INSERT OR IGNORE INTO vocabulary (term, entity_type) VALUES (?, ?)",
            vocab_data
        )

        conn.commit()

    logger.info("industry vocabulary 重建完成")


if __name__ == "__main__":

    rebuild_industry_vocab()