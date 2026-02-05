import sqlite3
import logging
import time
import argparse
from api_client import APIClient
from config import DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IDMigrator:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self.api = APIClient()
        self._setup_db()

    def _setup_db(self):
        """配置高性能参数"""
        self.cursor.execute("PRAGMA synchronous = OFF")
        self.cursor.execute("PRAGMA journal_mode = WAL")
        self.cursor.execute("PRAGMA cache_size = -1000000")  # 1GB 缓存

    def fast_sql_cleanup(self):
        """步骤 1: 使用 SQL substr 快速清理 'oa:' 前缀 (处理速度最快)"""
        logger.info("开始执行 SQL 级联清理 'oa:' 前缀...")
        tables = ['works', 'abstracts', 'authorships', 'work_fields']
        total_affected = 0

        for table in tables:
            # substr(work_id, 4) 截掉 'oa:' 后的内容
            self.cursor.execute(f"""
                UPDATE {table} 
                SET work_id = substr(work_id, 4) 
                WHERE work_id LIKE 'oa:%'
            """)
            total_affected += self.cursor.rowcount

        self.conn.commit()
        logger.info(f"本地 SQL 清理完成，共影响 {total_affected} 行。")

    def api_doi_repair(self, batch_size=50):
        """步骤 2: 针对无法本地转换的 doi: 记录，通过 API 批量打捞"""
        logger.info("正在检索需要 API 修复的 DOI 记录...")
        self.cursor.execute("""
                            SELECT work_id, doi
                            FROM works
                            WHERE work_id NOT LIKE 'W%'
                              AND doi IS NOT NULL
                            """)
        to_fix = self.cursor.fetchall()

        total = len(to_fix)
        if total == 0:
            logger.info("未发现需要 API 修复的记录。")
            return

        logger.info(f"发现 {total} 条旧 ID 记录，开始分批 (size={batch_size}) 请求 OpenAlex...")

        for i in range(0, total, batch_size):
            chunk = to_fix[i: i + batch_size]
            dois = [row[1] for row in chunk if row[1]]
            if not dois: continue

            try:
                # 批量请求 OpenAlex 获取正式 ID
                doi_filter = "|".join(dois)
                data = self.api.make_request("https://api.openalex.org/works", {"filter": f"doi:{doi_filter}"})

                if not data or "results" not in data:
                    continue

                # 建立映射: DOI -> 正式 ID (W...)
                mapping = {res.get('doi'): res.get('id', '').split('/')[-1] for res in data["results"]}

                # 批量更新数据库
                self.cursor.execute("BEGIN TRANSACTION")
                success = 0
                for old_id, original_doi in chunk:
                    new_id = mapping.get(original_doi)
                    if new_id and new_id != old_id:
                        for table in ['works', 'abstracts', 'authorships', 'work_fields']:
                            self.cursor.execute(f"UPDATE {table} SET work_id = ? WHERE work_id = ?", (new_id, old_id))
                        success += 1
                self.conn.commit()

                logger.info(f"进度: {min(i + batch_size, total)}/{total} | 本批成功修复: {success}")
                time.sleep(0.1)  # 礼貌延迟

            except Exception as e:
                self.conn.rollback()
                logger.error(f"批次 {i} 修复失败: {e}")

    def run(self):
        start = time.time()
        # 1. 先跑本地 SQL (处理 90% 的数据)
        self.fast_sql_cleanup()
        # 2. 再跑 API 修复 (处理剩下的 doi: 顽固分子)
        self.api_doi_repair()

        logger.info(f"全库百万级转换完成！总耗时: {time.time() - start:.2f} 秒")
        self.conn.close()


if __name__ == "__main__":
    import shutil
    import os

    # 自动备份
    if os.path.exists(DB_PATH):
        backup = DB_PATH + ".final_migrate.bak"
        shutil.copy2(DB_PATH, backup)
        logger.info(f"备份已创建: {backup}")

    migrator = IDMigrator()
    migrator.run()