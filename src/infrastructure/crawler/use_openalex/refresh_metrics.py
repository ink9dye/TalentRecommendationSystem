import logging
from datetime import datetime
from database import DatabaseManager

# 配置日志，设为 INFO 以便观察刷新进度
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def refresh_all_metrics():
    """
    仅执行指标刷新逻辑：
    1. 重新计算作者、机构、渠道的统计数据
    2. 更新对应的 last_updated 时间戳
    """
    db = DatabaseManager()

    try:
        logger.info(" 正在启动本地科研影响力指标重新校准...")

        # 1. 调用内置的高性能统计刷新方法
        # 该方法会自动处理：
        # - _update_author_stats: 更新作者论文数和总被引
        # - _update_institution_stats: 更新机构指标（含去重逻辑）
        # - _update_source_stats: 更新发布渠道/期刊统计
        db.refresh_stats()

        # 2. 强制同步更新时间戳，以便 Phase 4/5 逻辑识别数据已最新
        current_time = datetime.now().isoformat()
        with db.connection() as conn:
            logger.info("🕒 正在同步各表 last_updated 时间戳...")
            conn.execute("UPDATE authors SET last_updated = ?", (current_time,))
            conn.execute("UPDATE institutions SET last_updated = ?", (current_time,))
            conn.execute("UPDATE sources SET last_updated = ?", (current_time,))

        logger.info(f"✅ 指标刷新完成！当前统计基准时间: {current_time}")

    except Exception as e:
        logger.error(f"❌ 指标刷新失败: {e}")


if __name__ == "__main__":
    refresh_all_metrics()