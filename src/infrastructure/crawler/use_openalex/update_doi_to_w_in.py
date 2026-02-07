import time
import logging
import sqlite3
from tqdm import tqdm
from database import DatabaseManager

# 设置日志
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("Local_Deduplicator")


def perform_local_deduplication(db: DatabaseManager):
    """
    带进度条和断点续传的本地合并逻辑
    """
    BATCH_SIZE = 1000  # 每一批处理的记录数

    with db.connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout = 30000")

        # 1. 建立持久化映射表（断点续传的基础）
        # 只要这个表存在且有数据，就说明还有没合并完的任务
        conn.execute("""
                     CREATE TABLE IF NOT EXISTS _internal_dedup_mapping
                     (
                         old_id
                         TEXT
                         PRIMARY
                         KEY,
                         new_id
                         TEXT,
                         is_processed
                         INTEGER
                         DEFAULT
                         0
                     )
                     """)

        # 2. 检查任务是否已初始化
        check_init = conn.execute("SELECT COUNT(*) FROM _internal_dedup_mapping WHERE is_processed = 0").fetchone()[0]

        if check_init == 0:
            print("🔍 正在扫描数据库寻找冗余记录 (这可能需要 1-3 分钟)...")
            # 基于 title, year, date, type, language 匹配
            # 这里的逻辑是：找出 DOI 行，并为其找到一个内容完全一致的 W 行
            conn.execute("""
                         INSERT
                         OR IGNORE INTO _internal_dedup_mapping (old_id, new_id)
                         SELECT old_w.work_id AS old_id,
                                new_w.work_id AS new_id
                         FROM works old_w
                                  JOIN works new_w ON
                             old_w.title = new_w.title AND
                             old_w.year = new_w.year AND
                             old_w.type = new_w.type AND
                             old_w.language = new_w.language
                         WHERE old_w.work_id LIKE 'doi:%'
                           AND new_w.work_id LIKE 'W%'
                           AND old_w.work_id != new_w.work_id
                         """)

            total_tasks = conn.execute("SELECT COUNT(*) FROM _internal_dedup_mapping").fetchone()[0]
            if total_tasks == 0:
                print("✨ 未发现可合并的冗余记录。")
                # 清理临时表
                conn.execute("DROP TABLE IF EXISTS _internal_dedup_mapping")
                return
            print(f"📦 发现 {total_tasks} 组冗余数据，准备合并...")
        else:
            print(f"🔄 检测到未完成的合并任务，继续处理剩余的 {check_init} 条记录...")

        # 3. 分批执行合并
        total_to_do = conn.execute("SELECT COUNT(*) FROM _internal_dedup_mapping WHERE is_processed = 0").fetchone()[0]
        pbar = tqdm(total=total_to_do, desc="🧹 合并进度")

        while True:
            # 获取一盘批次
            cursor = conn.cursor()
            cursor.execute("""
                           SELECT old_id, new_id
                           FROM _internal_dedup_mapping
                           WHERE is_processed = 0 LIMIT ?
                           """, (BATCH_SIZE,))
            batch = cursor.fetchall()

            if not batch:
                break

            try:
                # 开启事务执行资产转移
                conn.execute("BEGIN TRANSACTION")

                for old_id, new_id in batch:
                    # 更新关联表
                    conn.execute("UPDATE OR IGNORE authorships SET work_id = ? WHERE work_id = ?", (new_id, old_id))
                    conn.execute("UPDATE OR IGNORE work_fields SET work_id = ? WHERE work_id = ?", (new_id, old_id))
                    conn.execute("UPDATE OR IGNORE abstracts SET work_id = ? WHERE work_id = ?", (new_id, old_id))

                    # 删除主表冗余行
                    conn.execute("DELETE FROM works WHERE work_id = ?", (old_id,))

                    # 标记映射表为已处理
                    conn.execute("UPDATE _internal_dedup_mapping SET is_processed = 1 WHERE old_id = ?", (old_id,))

                conn.execute("COMMIT")
                pbar.update(len(batch))

            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"批次合并失败: {e}")
                time.sleep(1)

        pbar.close()

        # 4. 任务彻底完成后，清理映射表
        print("🎉 本地合并任务全部完成！正在清理临时数据...")
        conn.execute("DROP TABLE IF EXISTS _internal_dedup_mapping")


if __name__ == "__main__":
    db_mgr = DatabaseManager()
    perform_local_deduplication(db_mgr)
    db_mgr.close()