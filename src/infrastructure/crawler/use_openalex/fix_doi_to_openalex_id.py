import logging
import sys
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 直接从你的 config 导入
from src.infrastructure.crawler.use_openalex.config import EMAIL, DB_PATH, MAX_WORKERS
from api_client import APIClient
from database import DatabaseManager
from utils import clean_id

# 降低日志级别以减少控制台冗余
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
    quota_mgr.increment()

    # 1. 提取 DOI (优先使用 doi 属性，没用的话从 work_id 里挖)
    target_doi = None
    if actual_doi and str(actual_doi).strip().lower() not in ['none', '', 'null']:
        target_doi = str(actual_doi).strip()
    elif old_id.startswith("doi:"):
        target_doi = old_id[4:].strip()

    if not target_doi:
        return old_id, "SKIPPED", None  # 根本没 DOI 属性，直接跳过

    try:
        # 清洗 DOI 格式 (只保留 10.xxxx...)
        doi_match = re.search(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', target_doi, re.I)
        if not doi_match:
            return old_id, "SKIPPED", None

        clean_doi = doi_match.group(1)
        params = {"filter": f"doi:{clean_doi}"}

        # 发起请求
        res = api_client.make_request("https://api.openalex.org/works", params)

        if res:
            results = res.get('results', [])
            if results:
                return old_id, "SUCCESS", results[0]  # 成功匹配
            else:
                return old_id, "NOT_FOUND", None  # API 通了，但库里没这号人

        return old_id, "API_ERROR", None  # API 返回为空或失败

    except Exception as e:
        if quota_mgr.count % 50 == 0:
            print(f"\n[System Error] {e}")
        return old_id, "API_ERROR", None


def migrate_doi_to_openalex():
    db_mgr = DatabaseManager(db_path=DB_PATH)
    api_client = APIClient()

    current_workers = MAX_WORKERS if MAX_WORKERS else 5
    print(f"🚀 启动 23w DOI 精准转换项目...")

    # 细化统计维度
    global_stats = {
        "success": 0,  # 成功找到并更新
        "merged": 0,  # 成功找到并合并旧数据
        "not_found": 0,  # OpenAlex 确实没收录 (NF)
        "failed": 0,  # 真正的程序/网络报错 (Err)
        "skipped": 0  # 数据格式不对直接跳过的
    }

    while True:
        with db_mgr.connection() as conn:
            cursor = conn.cursor()
            # 💡 增加一个过滤：只选还没被标记为“已检查但没找到”的记录（可选）
            cursor.execute("SELECT work_id, doi FROM works WHERE work_id LIKE 'doi:%' LIMIT 1000")
            rows = cursor.fetchall()

        if not rows:
            print("🎉 任务全部完成！")
            break

        with ThreadPoolExecutor(max_workers=current_workers) as executor:
            pbar = tqdm(total=len(rows), desc="⚡ 处理中", unit="doi")

            future_to_oid = {
                executor.submit(process_single_doi, row[0], row[1], api_client): row[0]
                for row in rows
            }

            buffer = []
            for future in as_completed(future_to_oid):
                old_id, status, work_data = future.result()

                if status == "SUCCESS":
                    buffer.append((old_id, work_data))
                elif status == "NOT_FOUND":
                    global_stats["not_found"] += 1
                    # 💡 可选：在这里可以将该 old_id 记录到某个临时表，防止下次循环又查一遍
                elif status == "API_ERROR":
                    global_stats["failed"] += 1
                else:
                    global_stats["skipped"] += 1

                # 批量提交
                if len(buffer) >= 100:
                    flush_buffer(db_mgr, buffer, global_stats)
                    buffer = []

                pbar.update(1)
                # 重点：在进度条展示 NF (Not Found)
                pbar.set_postfix({
                    "OK": global_stats["success"],
                    "NF": global_stats["not_found"],
                    "Err": global_stats["failed"],
                    "API": quota_mgr.count
                })

            if buffer:
                flush_buffer(db_mgr, buffer, global_stats)
            pbar.close()

    db_mgr.close()


def flush_buffer(db, buffer, stats):
    """批量事务更新"""
    with db.connection() as conn:
        conn.execute("BEGIN TRANSACTION")
        try:
            for old_id, work_data in buffer:
                perform_update(conn, old_id, work_data, stats)
            conn.execute("COMMIT")
        except Exception as e:
            conn.execute("ROLLBACK")
            print(f"写入磁盘失败: {e}")


def perform_update(conn, old_id, work_data, stats):
    """
    数据库重映射逻辑
    """
    new_id = clean_id(work_data['id'])
    if not new_id or new_id == old_id: return

    cur = conn.cursor()
    cur.execute("SELECT 1 FROM works WHERE work_id = ?", (new_id,))
    exists = cur.fetchone()

    # 更新关联表
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