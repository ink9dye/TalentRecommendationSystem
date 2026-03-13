import logging
import time
import json
from tqdm import tqdm
from api_client import APIClient
from processor import DataProcessor
from database import DatabaseManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def recrawl_missing_abstracts():
    db = DatabaseManager()
    api_client = APIClient()
    processor = DataProcessor(db)

    # 判定逻辑：继续沿用对 "null" 字符串的兼容
    MISSING_FILTER = """
        work_id LIKE 'W%' 
        AND (
            inverted_index IS NULL 
            OR inverted_index = '' 
            OR LOWER(inverted_index) = 'null'
            OR LOWER(inverted_index) = 'none'
        )
        AND inverted_index NOT IN ('N/A', 'NOT_FOUND')
    """

    with db.connection() as conn:
        stats = conn.execute(f"SELECT COUNT(*) FROM abstracts WHERE {MISSING_FILTER}").fetchone()[0]

    if stats == 0:
        logger.info("✅ 提示：所有记录已具备索引，或已标记为不可修复。")
        return

    print("\n" + "=" * 60)
    print(f"📊 摘要定向补全 (增强型 ID 匹配逻辑)")
    print(f"  - 待处理总量: {stats:,}")
    print(f"  - 目标: 解决 Reply/Peer-review 记录被误判为 NOT_FOUND 的问题")
    print("=" * 60 + "\n")

    batch_size = 50
    success_count = 0
    pbar = tqdm(total=stats, desc="API 补全进度", unit="paper")

    while True:
        with db.connection() as conn:
            rows = conn.execute(f"SELECT work_id FROM abstracts WHERE {MISSING_FILTER} LIMIT ?",
                                (batch_size,)).fetchall()

        if not rows:
            break

        chunk_ids = [row[0] for row in rows]
        ids_filter = "|".join(chunk_ids)

        try:
            # 增加 ids 字段，用于处理重定向和多 ID 关联
            params = {
                "filter": f"openalex_id:{ids_filter}",
                "select": "id,abstract_inverted_index,ids,display_name"
            }
            data = api_client.make_request("https://api.openalex.org/works", params)

            found_in_this_batch = set()

            if data and "results" in data:
                for work in data["results"]:
                    # --- 改进的 ID 提取逻辑 ---
                    possible_ids = set()

                    # 1. 提取 canonical ID (例如 https://openalex.org/W123 -> W123)
                    if work.get('id'):
                        possible_ids.add(work.get('id').split('/')[-1].upper())

                    # 2. 提取 ids 字典中所有的 openalex 关联 ID
                    ids_dict = work.get('ids', {})
                    if ids_dict:
                        for key, val in ids_dict.items():
                            if 'openalex' in key:
                                possible_ids.add(val.split('/')[-1].upper())

                    # 3. 找出这个 Result 对应了我们请求中的哪些 ID
                    # (使用 .upper() 忽略大小写差异)
                    matched_ids = [rid for rid in chunk_ids if rid.upper() in possible_ids]

                    for mid in matched_ids:
                        found_in_this_batch.add(mid)

                    raw_index = work.get('abstract_inverted_index')

                    with db.connection() as conn:
                        for mid in matched_ids:
                            if raw_index:
                                full_text = processor.restore_abstract(raw_index)
                                conn.execute("""
                                             UPDATE abstracts
                                             SET inverted_index = ?,
                                                 full_text_en   = ?
                                             WHERE work_id = ?
                                             """, (json.dumps(raw_index), full_text, mid))
                                success_count += 1
                            else:
                                # 像你提到的 Reply/Peer-review 记录会落入这里
                                # 标记为 N/A，因为它在 API 里确实没摘要
                                conn.execute("""
                                             UPDATE abstracts
                                             SET inverted_index = 'N/A',
                                                 full_text_en   = 'N/A'
                                             WHERE work_id = ?
                                             """, (mid,))

            # --- 关键修改：处理没出现在 results 中的 ID ---
            missing_ids = set(chunk_ids) - found_in_this_batch
            if missing_ids:
                with db.connection() as conn:
                    for m_id in missing_ids:
                        # 只有在 batch 接口完全不返回该 ID 时才标记
                        # 此时大概率是该 ID 确实在 OpenAlex 系统中由于某种原因无法被检索
                        conn.execute(
                            "UPDATE abstracts SET inverted_index = 'NOT_FOUND', full_text_en = 'N/A' WHERE work_id = ?",
                            (m_id,))

            pbar.update(len(chunk_ids))
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"批量请求异常: {e}")
            time.sleep(2)
            continue

    pbar.close()
    logger.info(f"🎊 补全结束。成功救回: {success_count:,} 条文本。")


if __name__ == "__main__":
    recrawl_missing_abstracts()