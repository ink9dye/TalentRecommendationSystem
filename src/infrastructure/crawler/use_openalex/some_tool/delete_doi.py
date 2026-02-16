import logging
from src.infrastructure.crawler.use_openalex.db_config import DB_PATH
from database import DatabaseManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def cleanup_unresolved_dois():
    db_mgr = DatabaseManager(db_path=DB_PATH)
    detail_tables = ['authorships', 'work_fields', 'abstracts']
    main_table = 'works'

    print(f"🧹 开始扫描待清理的残留 DOI 数据...")

    with db_mgr.connection() as conn:
        cursor = conn.cursor()

        # 1. 详细统计各表数据量
        stats = {}
        cursor.execute(f"SELECT COUNT(*) FROM {main_table} WHERE work_id LIKE 'doi:%'")
        stats[main_table] = cursor.fetchone()[0]

        for table in detail_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE work_id LIKE 'doi:%'")
            stats[table] = cursor.fetchone()[0]

        total_rows = sum(stats.values())

        if stats[main_table] == 0:
            print("✅ 没有发现以 'doi:' 开头的残留记录，无需清理。")
            return

        print("\n--- 待删除数据统计 ---")
        print(f"📄 主表 [{main_table}]: {stats[main_table]} 条记录")
        print(f"👥 作者表 [authorships]: {stats['authorships']} 条记录")
        print(f"🏷️ 领域表 [work_fields]: {stats['work_fields']} 条记录")
        print(f"📝 摘要表 [abstracts]: {stats['abstracts']} 条记录")
        print(f"📊 总计受影响行数: {total_rows}")
        print("----------------------\n")

        confirm = input("⚠️ 注意：执行后这些数据将从数据库彻底抹除！确认删除请按 'y': ")
        if confirm.lower() != 'y':
            print("❌ 操作取消，数据已保留。")
            return

        try:
            conn.execute("BEGIN TRANSACTION")

            # 2. 执行级联删除
            for table in detail_tables:
                cursor.execute(f"DELETE FROM {table} WHERE work_id LIKE 'doi:%'")

            cursor.execute(f"DELETE FROM {main_table} WHERE work_id LIKE 'doi:%'")

            conn.execute("COMMIT")
            print(f"\n🎉 清理成功！已从数据库移除所有无效的 DOI 关联记录。")

            # 3. 整理磁盘
            print("⏳ 正在执行 VACUUM 释放磁盘碎片...")
            conn.execute("VACUUM")
            print("✨ 数据库已焕然一新。")

        except Exception as e:
            conn.execute("ROLLBACK")
            print(f"🚨 清理过程中出错，已安全回滚: {e}")


if __name__ == "__main__":
    cleanup_unresolved_dois()