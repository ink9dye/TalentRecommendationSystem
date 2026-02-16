import logging
import time
import json
from tqdm import tqdm
from api_client import APIClient
from processor import DataProcessor
from database import DatabaseManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_not_found_records():
    db = DatabaseManager()
    api_client = APIClient()
    processor = DataProcessor(db)

    # 【判定逻辑】只针对之前批量请求失败、被标记为 'NOT_FOUND' 的记录
    TARGET_FILTER = "work_id LIKE 'W%' AND inverted_index = 'NOT_FOUND'"

    # 1. 统计需要二次处理的疑难杂症总量
    with db.connection() as conn:
        stats = conn.execute(f"SELECT COUNT(*) FROM abstracts WHERE {TARGET_FILTER}").fetchone()[0]

    if stats == 0:
        logger.info("✅ 提示：没有发现被标记为 'NOT_FOUND' 的记录，无需执行补爬。")
        return

    print("\n" + "=" * 60)
    print(f"🎯 定向单点补全 (针对 NOT_FOUND 记录)")
    print(f"  - 待修复总量: {stats:,}")
    print(f"  - 策略: 放弃批量 filter，直接请求 /works/{{id}} 接口")
    print("=" * 60 + "\n")

    success_count = 0
    na_count = 0
    error_404_count = 0

    # 2. 逐条取出进行单点请求
    # 注意：单点请求速度较慢（受 API 配额限制），建议分批运行
    with db.connection() as conn:
        rows = conn.execute(f"SELECT work_id FROM abstracts WHERE {TARGET_FILTER}").fetchall()

    pbar = tqdm(total=len(rows), desc="单点修复进度", unit="paper")

    for row in rows:
        work_id = row[0]

        try:
            # 直接通过 ID 路径访问单条记录
            # 这种请求方式最精准，能穿透 dataset/peer-review 的过滤限制
            url = f"https://api.openalex.org/works/{work_id}"

            # 可以在这里根据你的 API 配额调整 select 字段
            params = {"select": "id,abstract_inverted_index"}
            data = api_client.make_request(url, params)

            with db.connection() as conn:
                if data:
                    # 情况 A: 单点请求成功找到记录
                    raw_index = data.get('abstract_inverted_index')
                    if raw_index:
                        # 终于救回了摘要
                        full_text = processor.restore_abstract(raw_index)
                        conn.execute("""
                                     UPDATE abstracts
                                     SET inverted_index = ?,
                                         full_text_en   = ?
                                     WHERE work_id = ?
                                     """, (json.dumps(raw_index), full_text, work_id))
                        success_count += 1
                    else:
                        # 记录存在但确实没摘要（例如你提到的 dataset）
                        conn.execute("""
                                     UPDATE abstracts
                                     SET inverted_index = 'N/A',
                                         full_text_en   = 'N/A'
                                     WHERE work_id = ?
                                     """, (work_id,))
                        na_count += 1
                else:
                    # 情况 B: 单点请求也返回空（真正的 404）
                    # 此时我们可以选择保持 NOT_FOUND 或标记为 DELETED
                    # 这里选择保持原样或更新一下时间戳/日志
                    error_404_count += 1

            # 单点请求建议稍微增加延迟，避免触发 API 限制
            # 如果你有邮件订阅密钥（Polite Pool），可以设为 0.05
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"处理 ID {work_id} 时发生异常: {e}")
            time.sleep(1)

        pbar.update(1)

    pbar.close()

    print("\n" + "=" * 60)
    print(f"🎊 修复任务完成！")
    print(f"  - 救回摘要: {success_count:,} 条")
    print(f"  - 确认无摘要 (转为 N/A): {na_count:,} 条")
    print(f"  - 依然无法找到 (404): {error_404_count:,} 条")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    fix_not_found_records()