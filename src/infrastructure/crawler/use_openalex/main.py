import argparse
import logging
import time
import os
from src.infrastructure.crawler.use_openalex.config import FIELDS, DB_PATH, DATA_DIR # 确保导入了 DATA_DIR
from database import DatabaseManager
from crawler_logic import fetch_phase4,fetch_phase5


def show_stats(db):
    """显示数据库当前的详细统计"""
    with db.connection() as conn:
        print(f"\n" + "=" * 30)
        print(f"数据库数据总览:")
        print(f"  总论文数: {conn.execute('SELECT COUNT(*) FROM works').fetchone()[0]:,}")
        print(f"  总作者数: {conn.execute('SELECT COUNT(*) FROM authors').fetchone()[0]:,}")
        print(f"  总机构数: {conn.execute('SELECT COUNT(*) FROM institutions').fetchone()[0]:,}")
        print(f"  关联总数: {conn.execute('SELECT COUNT(*) FROM authorships').fetchone()[0]:,}")

        print(f"\n各领域分布:")
        rows = conn.execute(
            "SELECT field_name, COUNT(*) FROM work_fields GROUP BY field_name ORDER BY COUNT(*) DESC").fetchall()
        for name, count in rows:
            print(f"  {name:20s}: {count:,} 篇")
        print("=" * 30 + "\n")


def run_task(db, field_id, force=False):
    """执行单个领域的爬取任务"""
    f_name, c_id = FIELDS[field_id]

    # 【新增逻辑】如果使用了 --force 参数，先重置该领域的进度
    if force:
        print(f"\n[!] 正在强制重置领域进度: {f_name}...")
        with db.connection() as conn:
            # 清除爬虫进度
            conn.execute("DELETE FROM crawl_states WHERE field_id = ?", (field_id,))
            # 清除作者处理记录（如果需要重新补全精英作者画像）
            conn.execute("DELETE FROM author_process_states WHERE field_id = ?", (field_id,))
            # 注意：不删除 work_fields，因为它只作为辅助统计
        print(f"[!] {f_name} 的本地进度已清空，准备重新爬取。")

    print(f"\n" + "*" * 50)
    print(f"正在启动任务: {f_name} (ID: {c_id})")
    print("*" * 50)

    # 保持原有调用逻辑不变
    # fetch_phase1(db, field_id, f_name, c_id, target_count=10000)
    # elite_authors = fetch_phase2(db, field_id, f_name, c_id,target_count=2000)
    # fetch_phase3(db, field_id, f_name, c_id, elite_authors)

    print(f"\n[成功] 领域 {f_name} 爬取任务已结束。")
    print("\n[重点] 正在执行全量作者 H-index 同步，请耐心等待...")
    fetch_phase4(db)
    print("\n[重点] 正在执行机构、渠道信息同步，请耐心等待...")
    fetch_phase5(db)
    print("所有阶段抓取完成，开始同步本地统计指标...")
    db.refresh_stats()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(DATA_DIR, "crawler.log")),
            logging.StreamHandler()  # 这行负责把日志显示在控制台
        ]
    )
    parser = argparse.ArgumentParser(description="OpenAlex 学术数据爬取工具 v5.1")
    parser.add_argument("--field", help="领域编号 (1-17)")
    parser.add_argument("--force", action="store_true", help="强制重新爬取")
    parser.add_argument("--stats", action="store_true", help="查看统计")
    args = parser.parse_args()

    db = DatabaseManager(DB_PATH)

    if args.stats:
        show_stats(db)
        return

    if args.field:
        if args.field in FIELDS:
            run_task(db, args.field, args.force)
        else:
            print(f"错误: 领域编号 {args.field} 不存在。")
        return

    # 交互模式
    print("=== OpenAlex 学术数据爬取工具 v5.1 (模块化版) ===")
    print("可用领域:")
    for k, v in FIELDS.items():
        print(f"  {k:2s}. {v[0]}")

    choice = input("\n请选择领域编号 (输入 'all' / 'stats' / 或回车退出): ").strip().lower()

    if choice == 'stats':
        show_stats(db)
    elif choice == 'all':
        print(f"确定要爬取所有领域吗？(y/n): ", end='', flush=True)
        if input().lower() == 'y':
            for f_id in FIELDS.keys():
                run_task(db, f_id, args.force)
                time.sleep(2)
            show_stats(db)
    elif choice in FIELDS:
        run_task(db, choice, args.force)
    else:
        print("程序已退出。")

    db.close()


if __name__ == "__main__":
    main()