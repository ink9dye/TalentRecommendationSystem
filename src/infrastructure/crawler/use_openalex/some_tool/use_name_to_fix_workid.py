import logging
import os
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 从你的项目结构导入
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


def process_by_metadata(old_id, title, year, keywords, work_type, api_client):
    """
    通过 标题 + 关键词 进行搜索，并严格过滤 年份 和 类型
    """
    if not title:
        return old_id, "SKIPPED", None

    # 1. 构造搜索词：标题 + 前5个关键词 (提高相关度权重)
    search_query = title.strip()
    if keywords:
        kw_list = [k.strip() for k in keywords.split('|') if k.strip()]
        search_query += " " + " ".join(kw_list[:5])

    quota_mgr.increment()
    try:
        # 2. 构造 API 参数
        params = {
            "search": search_query,
        }

        # 3. 构造过滤器 (多个条件用逗号分隔，表示 AND 关系)
        filters = []
        if year:
            filters.append(f"publication_year:{year}")
        if work_type:
            # 确保类型为小写，以符合 OpenAlex API 标准 (如 article, book-chapter)
            filters.append(f"type:{work_type.lower()}")

        if filters:
            params["filter"] = ",".join(filters)

        res = api_client.make_request("https://api.openalex.org/works", params)

        if res and res.get('results'):
            # 返回搜索结果列表中最匹配的一个
            return old_id, "SUCCESS", res['results'][0]

        return old_id, "NOT_FOUND", None
    except Exception:
        return old_id, "API_ERROR", None

def flush_updates(db, buffer, stats):
    """更新主表及关联表"""
    with db.connection() as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            for old_id, status, work_data in buffer:
                if status == "SUCCESS" and work_data:
                    perform_update(conn, old_id, work_data, stats)
            conn.execute("COMMIT")
        except Exception as e:
            conn.execute("ROLLBACK")
            print(f"数据库写入失败: {e}")


def perform_update(conn, old_id, work_data, stats):
    new_id = clean_id(work_data['id'])
    if not new_id or new_id == old_id: return

    cur = conn.cursor()
    # 检查转换后的新 ID 是否已经在数据库中存在
    cur.execute("SELECT 1 FROM works WHERE work_id = ?", (new_id,))
    exists = cur.fetchone()

    # 更新所有关联表
    for table in ['authorships', 'work_fields', 'abstracts']:
        conn.execute(f"UPDATE OR IGNORE {table} SET work_id = ? WHERE work_id = ?", (new_id, old_id))

    if exists:
        # 如果新 ID 已存在，删除旧的 doi 记录（合并操作）
        conn.execute("DELETE FROM works WHERE work_id = ?", (old_id,))
        stats["merged"] += 1
    else:
        # 如果新 ID 不存在，直接更新主表的主键
        conn.execute("UPDATE works SET work_id = ? WHERE work_id = ?", (new_id, old_id))
        stats["success"] += 1


def migrate_metadata_to_openalex():
    db_mgr = DatabaseManager(db_path=DB_PATH)
    api_client = APIClient()

    current_workers = MAX_WORKERS if MAX_WORKERS else 5
    print(f"🚀 启动增强版转换项目 (Title + Keywords + Year + Type)...")

    global_stats = {"success": 0, "merged": 0, "not_found": 0, "failed": 0, "skipped": 0}

    while True:
        # --- 修改点：SQL 增加 type 字段 ---
        with db_mgr.connection() as conn:
            cursor = conn.cursor()
            query = """
                    SELECT work_id, title, year, keywords_text, type
                    FROM works
                    WHERE work_id LIKE 'doi:%'
                        LIMIT 1000 \
                    """
            cursor.execute(query)
            rows = cursor.fetchall()

        if not rows:
            print("🎉 所有符合条件的记录已处理完毕！")
            break

        with ThreadPoolExecutor(max_workers=current_workers) as executor:
            pbar = tqdm(total=len(rows), desc="⚡ 转换中", unit="doi")

            # --- 修改点：向函数传递 r[4] 即 type 字段 ---
            future_to_oid = {
                executor.submit(process_by_metadata, r[0], r[1], r[2], r[3], r[4], api_client): r[0]
                for r in rows
            }

            result_buffer = []

            for future in as_completed(future_to_oid):
                old_id, status, work_data = future.result()

                if status == "SUCCESS":
                    result_buffer.append((old_id, status, work_data))
                elif status == "NOT_FOUND":
                    global_stats["not_found"] += 1
                elif status == "API_ERROR":
                    global_stats["failed"] += 1
                elif status == "SKIPPED":
                    global_stats["skipped"] += 1

                if len(result_buffer) >= 50:
                    flush_updates(db_mgr, result_buffer, global_stats)
                    result_buffer = []

                pbar.update(1)
                pbar.set_postfix({
                    "成功": global_stats["success"],
                    "未找到": global_stats["not_found"],
                    "配额": quota_mgr.count
                })

            if result_buffer:
                flush_updates(db_mgr, result_buffer, global_stats)
            pbar.close()

    db_mgr.close()

if __name__ == "__main__":
    migrate_metadata_to_openalex()