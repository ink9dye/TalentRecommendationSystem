import logging
import sys
import os
import re
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 直接从你的 config 导入
from src.infrastructure.crawler.use_openalex.db_config import EMAIL, DB_PATH, MAX_WORKERS
from api_client import APIClient
from database import DatabaseManager
from utils import clean_id

logging.basicConfig(level=logging.ERROR)


class QuotaManager:
    def __init__(self, limit=95000):
        self.count = 0
        self.limit = limit
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.count += 1
            if self.count >= self.limit:
                print(f"\n🛑 达到今日配额上限 ({self.limit})，安全退出。")
                os._exit(0)


quota_mgr = QuotaManager()


def process_single_doi(old_id, actual_doi, api_client):
    """
    返回: (old_id, status, work_data)
    """
    # 提取并清洗 DOI
    target = actual_doi if actual_doi and len(str(actual_doi)) > 5 else (
        old_id[4:] if old_id.startswith("doi:") else None)
    if not target:
        return old_id, "SKIPPED", None

    doi_match = re.search(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', str(target), re.I)
    if not doi_match:
        return old_id, "SKIPPED", None

    clean_doi = doi_match.group(1)

    # 消耗额度
    quota_mgr.increment()
    try:
        params = {"filter": f"doi:{clean_doi}"}
        res = api_client.make_request("https://api.openalex.org/works", params)

        if res:
            results = res.get('results', [])
            if results:
                return old_id, "SUCCESS", results[0]
            else:
                return old_id, "NOT_FOUND", None
        return old_id, "API_ERROR", None
    except Exception as e:
        return old_id, "API_ERROR", None


def init_mapping_table(db_mgr):
    """初始化中间表"""
    with db_mgr.connection() as conn:
        conn.execute("""
                     CREATE TABLE IF NOT EXISTS doi_mapping
                     (
                         old_work_id
                         TEXT
                         PRIMARY
                         KEY,
                         status
                         TEXT,
                         last_check
                         DATE
                     )
                     """)


def migrate_doi_to_openalex():
    db_mgr = DatabaseManager(db_path=DB_PATH)
    api_client = APIClient()
    init_mapping_table(db_mgr)

    current_workers = MAX_WORKERS if MAX_WORKERS else 5
    print(f"🚀 启动 23w DOI 精准转换项目 (中间表模式)...")

    global_stats = {"success": 0, "merged": 0, "not_found": 0, "failed": 0, "skipped": 0}

    while True:
        # 使用 LEFT JOIN 排除掉已经在 mapping 表里处理过的 DOI
        # 这样即使重启，也只会处理从未尝试过的记录
        with db_mgr.connection() as conn:
            cursor = conn.cursor()
            query = """
                    SELECT w.work_id, w.doi
                    FROM works w
                             LEFT JOIN doi_mapping m ON w.work_id = m.old_work_id
                    WHERE w.work_id LIKE 'doi:%'
                      AND m.old_work_id IS NULL LIMIT 1000 \
                    """
            cursor.execute(query)
            rows = cursor.fetchall()

        if not rows:
            print("🎉 任务全部完成或今日额度内无可处理数据！")
            break

        with ThreadPoolExecutor(max_workers=current_workers) as executor:
            pbar = tqdm(total=len(rows), desc="⚡ 处理中", unit="doi")
            future_to_oid = {
                executor.submit(process_single_doi, row[0], row[1], api_client): row[0]
                for row in rows
            }

            # buffer 存储需要更新主表和映射表的数据
            result_buffer = []

            for future in as_completed(future_to_oid):
                old_id, status, work_data = future.result()
                result_buffer.append((old_id, status, work_data))

                if status == "NOT_FOUND":
                    global_stats["not_found"] += 1
                elif status == "API_ERROR":
                    global_stats["failed"] += 1
                elif status == "SKIPPED":
                    global_stats["skipped"] += 1

                if len(result_buffer) >= 100:
                    flush_all_results(db_mgr, result_buffer, global_stats)
                    result_buffer = []

                pbar.update(1)
                pbar.set_postfix({
                    "OK": global_stats["success"],
                    "NF": global_stats["not_found"],
                    "Err": global_stats["failed"],
                    "API": quota_mgr.count
                })

            if result_buffer:
                flush_all_results(db_mgr, result_buffer, global_stats)
            pbar.close()

    db_mgr.close()


def flush_all_results(db, buffer, stats):
    """同时更新主表关联和中间表状态"""
    today = datetime.now().strftime('%Y-%m-%d')
    with db.connection() as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            for old_id, status, work_data in buffer:
                # 1. 无论结果如何，记录到 mapping 表，占住位置不再重刷
                conn.execute(
                    "INSERT OR REPLACE INTO doi_mapping (old_work_id, status, last_check) VALUES (?, ?, ?)",
                    (old_id, status, today)
                )

                # 2. 只有成功时才更新业务表
                if status == "SUCCESS" and work_data:
                    perform_update(conn, old_id, work_data, stats)

            conn.execute("COMMIT")
        except Exception as e:
            conn.execute("ROLLBACK")
            print(f"写入磁盘失败: {e}")


def perform_update(conn, old_id, work_data, stats):
    new_id = clean_id(work_data['id'])
    if not new_id or new_id == old_id: return

    cur = conn.cursor()
    cur.execute("SELECT 1 FROM works WHERE work_id = ?", (new_id,))
    exists = cur.fetchone()

    for table in ['authorships', 'work_fields', 'abstracts']:
        conn.execute(f"UPDATE OR IGNORE {table} SET work_id = ? WHERE work_id = ?", (new_id, old_id))

    if exists:
        conn.execute("DELETE FROM works WHERE work_id = ?", (old_id,))
        stats["merged"] += 1
    else:
        conn.execute("UPDATE works SET work_id = ? WHERE work_id = ?", (new_id, old_id))
        stats["success"] += 1


if __name__ == "__main__":
    migrate_doi_to_openalex()