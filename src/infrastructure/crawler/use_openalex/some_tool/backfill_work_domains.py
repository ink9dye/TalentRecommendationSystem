import sqlite3
import logging
from src.infrastructure.crawler.use_openalex.db_config import DB_PATH, FIELDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DomainBackfiller")


def run_local_backfill():
    # 构建映射表：{"Computer science": "1", "Medicine": "2", ...}
    # 注意：FIELDS 里的名称带有下划线，需处理为空格以对齐 OpenAlex display_name
    name_to_id = {v[0].replace("_", " ").lower(): k for k, v in FIELDS.items()}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 获取所有已有 concepts 的论文
    works = cursor.execute("SELECT work_id, concepts_text FROM works WHERE concepts_text IS NOT NULL").fetchall()
    logger.info(f"[*] 准备处理 {len(works)} 篇论文...")

    updates = []
    for wid, c_text in works:
        if not c_text: continue

        # 寻找交集：检查 concepts_text 中是否包含 17 个大类中的名称
        matched = []
        c_list = [c.lower() for c in c_text.split('|')]
        for name, d_id in name_to_id.items():
            if any(name in c for c in c_list):
                matched.append(d_id)

        if matched:
            domain_str = "|".join(sorted(list(set(matched))))
            updates.append((domain_str, wid))

    if updates:
        cursor.executemany("UPDATE works SET domain_ids = ? WHERE work_id = ?", updates)
        conn.commit()
        logger.info(f"✅ 成功补全 {len(updates)} 篇论文的领域属性 (1-17 ID 体系)。")

    conn.close()


if __name__ == "__main__":
    run_local_backfill()