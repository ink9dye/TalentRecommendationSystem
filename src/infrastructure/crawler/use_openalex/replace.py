import logging
from api_client import APIClient
from processor import DataProcessor
from database import DatabaseManager

# 注意：如果 DataProcessor 或 APIClient 需要 EMAIL 等配置，确保它们在内部已处理

logger = logging.getLogger(__name__)


def compensate_missing_details():
    """
    专门为已经转换成功（W开头）但缺少详细信息的 Work 补全数据
    """
    # 假设你的 DatabaseManager 默认就会读取正确的 DB 路径
    db_mgr = DatabaseManager()
    api_client = APIClient()
    processor = DataProcessor(db_mgr)

    # 1. 查找“空壳”论文
    # 逻辑：work_id 是 W 开头，但在作者表或摘要表中查不到记录的论文
    # 使用 EXISTS 通常比 LEFT JOIN 在大数据量下更快
    with db_mgr.connection() as conn:
        query = """
                SELECT work_id \
                FROM works w
                WHERE work_id LIKE 'W%'
                  AND NOT EXISTS (SELECT 1 FROM abstracts a WHERE a.work_id = w.work_id) LIMIT 5000
                """
        work_ids = [row[0] for row in conn.execute(query).fetchall()]

    if not work_ids:
        print("✅ 所有已转换论文详情均已补齐，无需操作。")
        return

    print(f"🔍 发现 {len(work_ids)} 篇‘空壳’论文，开始从 OpenAlex 批量拉取详情...")

    # 2. 批量请求（OpenAlex API 支持管道符号 | 批量查询）
    batch_size = 50
    success_count = 0

    for i in range(0, len(work_ids), batch_size):
        batch = work_ids[i:i + batch_size]
        id_filter = "|".join(batch)

        try:
            # 这里的过滤条件：openalex_id 支持批量 W ID
            params = {"filter": f"openalex_id:{id_filter}"}
            res = api_client.make_request("https://api.openalex.org/works", params)

            if res and res.get('results'):
                for work_data in res['results']:
                    # 调用你已有的处理器，它会解析 authorships, concepts, abstracts 等
                    # field_id 传 None 是因为这是存量补全，不是按学科抓取
                    processor.process_work(work_data, field_id=None, field_name=None)
                    success_count += 1

            print(f"⚡ 进度: {min(i + batch_size, len(work_ids))}/{len(work_ids)} (已成功补全 {success_count} 篇)")

        except Exception as e:
            print(f"❌ 批次 {i // batch_size + 1} 处理失败: {e}")

    print(f"🏁 补全任务完成！成功修复 {success_count} 篇论文的数据结构。")


if __name__ == "__main__":
    compensate_missing_details()