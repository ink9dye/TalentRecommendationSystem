import logging
from api_client import APIClient
from processor import DataProcessor
from utils import safe_get, clean_id
from config import EMAIL, BATCH_SIZE_AUTHORS, MAX_WORKERS
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


def fetch_phase1(db, field_id, field_name, concept_id, target_count=10000):
    """阶段1：获取最新论文（直到补满 target_count 篇新论文为止）"""
    logger.info(f"[Phase 1] 开始获取 {field_name} 的新论文，目标新增: {target_count:,} 篇")

    processor = DataProcessor(db)
    api_client = APIClient()

    # 获取状态：progress 记录的是已入库的新论文数
    state = db.get_crawl_state(field_id, 1)
    cursor = safe_get(state, ['cursor'], "*")
    new_papers = safe_get(state, ['progress'], 0)

    total_processed = 0
    batch_count = 0
    batch_size = 200

    while new_papers < target_count:
        params = {
            "filter": f"concepts.id:{concept_id}",
            "sort": "publication_date:desc",
            "per-page": batch_size,
            "cursor": cursor
        }

        data = api_client.make_request("https://api.openalex.org/works", params)
        if not data:
            logger.warning("Phase 1 API请求失败，中断当前翻页")
            break

        works = data.get("results", [])
        if not works:
            logger.info("Phase 1 已翻完所有可用数据，无法凑齐目标数")
            break

        batch_new = 0
        for work in works:
            if new_papers >= target_count:
                break

            result = processor.process_work(work, field_id, field_name)
            total_processed += 1
            # --- 修改这里 ---
            if result.get("is_new"):  # 改为判断学科新发现
                new_papers += 1
                batch_new += 1

        batch_count += 1
        cursor = data.get('meta', {}).get('next_cursor')

        db.update_crawl_state(field_id, 1, cursor, new_papers, False)

        if batch_count % 5 == 0 or new_papers >= target_count:
            logger.info(
                f"Phase 1 进度: 已新增 {new_papers:,}/{target_count:,} (本页处理 {len(works)} 篇，新增 {batch_new} 篇)")

        if not cursor:
            logger.info("Phase 1 已经到达 OpenAlex 数据终点")
            break

    db.update_crawl_state(field_id, 1, None, new_papers, new_papers >= target_count)
    logger.info(f"[Phase 1] 完成! 最终新增: {new_papers:,} 篇 (共扫描了 {total_processed:,} 篇)")
    return {"total": total_processed, "new": new_papers}


def fetch_phase2(db, field_id, field_name, concept_id, target_count=2000):
    """阶段2：获取高引论文，并收集精英作者"""
    logger.info(f"[Phase 2] 开始获取 {field_name} 的高引论文，目标新增: {target_count:,} 篇")

    processor = DataProcessor(db)
    api_client = APIClient()

    state = db.get_crawl_state(field_id, 2)
    cursor = state.get('cursor') if state else "*"
    new_papers = state.get('progress', 0) if state else 0

    elite_authors = set()
    total_processed = 0
    batch_count = 0
    batch_size = 200

    while new_papers < target_count:
        params = {
            "filter": f"concepts.id:{concept_id}",
            "sort": "cited_by_count:desc",
            "per-page": batch_size,
            "cursor": cursor
        }

        data = api_client.make_request("https://api.openalex.org/works", params)
        if not data:
            logger.warning("Phase 2 API请求失败")
            break

        works = data.get("results", [])
        if not works:
            logger.info("Phase 2 已翻完所有高引数据")
            break

        batch_new = 0
        for work in works:
            if new_papers >= target_count:
                break

            # 收集精英作者
            for authorship in work.get('authorships', []) or []:
                pos = authorship.get('author_position')
                is_corr = authorship.get('is_corresponding')
                if pos == 'first' or is_corr:
                    aid = clean_id(safe_get(authorship, ['author', 'id']))
                    if aid: elite_authors.add(aid)

            result = processor.process_work(work, field_id, field_name)
            total_processed += 1
            # --- 修改这里 ---
            if result.get("is_new"):  # 改为判断学科新发现
                new_papers += 1
                batch_new += 1

        batch_count += 1
        cursor = data.get('meta', {}).get('next_cursor')

        db.update_crawl_state(field_id, 2, cursor, new_papers, False)

        if batch_count % 5 == 0 or new_papers >= target_count:
            logger.info(
                f"Phase 2 进度: 已新增 {new_papers:,}/{target_count:,} (扫描 {total_processed:,} 篇，发现 {len(elite_authors):,} 位精英作者)")

        if not cursor: break

    db.update_crawl_state(field_id, 2, None, new_papers, new_papers >= target_count)
    logger.info(f"[Phase 2] 完成! 最终新增 {new_papers:,} 篇，精英作者种子库规模: {len(elite_authors):,}")
    return elite_authors


def fetch_phase3(db, field_id, field_name, concept_id, elite_authors):
    """
    阶段3：精英作者科研画像补足
    新增逻辑：获取作者详情中的 H-index 并更新画像
    """
    if not elite_authors:
        return {"total": 0, "new": 0}

    PROFILE_DEPTH = 20
    processor = DataProcessor(db)
    api_client = APIClient()

    authors_to_process = [aid for aid in elite_authors if not db.is_author_processed(field_id, aid, 3)]
    logger.info(f"[Phase 3] 开始为 {len(authors_to_process)} 位作者补全 H-index 与代表作...")

    stats = {"authors_processed": 0, "papers_new": 0}

    for idx, aid in enumerate(authors_to_process, 1):
        try:
            # --- 新增逻辑：获取并保存作者的 H-index ---
            author_entity = api_client.fetch_author_entity(aid)  # 需要在 api_client 中添加此方法
            if author_entity:
                # 从 OpenAlex 预计算好的统计数据中直接提取 H-index
                h_index = safe_get(author_entity, ['summary_stats', 'h_index'], 0)

                # 更新作者画像表（包含新增的 h_index 字段）
                db.save_author_batch([(
                    aid,
                    clean_id(author_entity.get('orcid')),
                    author_entity.get('display_name') or "Unknown",
                    author_entity.get('works_count', 0),
                    author_entity.get('cited_by_count', 0),
                    h_index,  # 存储到数据库
                    clean_id(safe_get(author_entity, ['last_known_institution', 'id'])),
                    datetime.now().isoformat()  # 建议在此记录同步时间
                )])

            # --- 原有逻辑：获取作者高引代表作 ---
            top_works = api_client.fetch_author_top_works(aid, limit=PROFILE_DEPTH)
            if top_works:
                for work in top_works:
                    result = processor.process_work(work, field_id, field_name)
                    if result.get("is_new"):
                        stats["papers_new"] += 1

            stats["authors_processed"] += 1
            db.mark_author_processed(field_id, aid, 3)

            if idx % 20 == 0:
                logger.info(f"Phase 3 进度: {idx}/{len(authors_to_process)} | 已补全 H-index 并入库代表作")

        except Exception as e:
            logger.error(f"处理作者 {aid} 失败: {e}")
            continue

    return stats


def fetch_phase4(db, refresh_days=30):
    """
    阶段 4：作者全球学术指标同步
    目标：利用 API 获取作者真实的 H-index、全球发文数及总被引
    """
    api_client = APIClient()

    with db.connection() as conn:
        # 筛选：H-index缺失 或 数据已过有效期
        query = """
                SELECT author_id \
                FROM authors
                WHERE h_index = 0 \
                   OR h_index IS NULL
                   OR last_updated IS NULL
                   OR julianday('now') - julianday(last_updated) > ? \
                """
        target_ids = [row[0] for row in conn.execute(query, (refresh_days,)).fetchall()]

    total = len(target_ids)
    if total == 0:
        logger.info("所有作者 H-index 均在有效期内。")
        return

    logger.info(f"[Phase 4] 正在同步 {total:,} 名作者的全球学术指标...")

    batch_size = 50
    processed_count = 0  # 新增：累计处理计数器

    for i in range(0, total, batch_size):
        chunk = target_ids[i: i + batch_size]
        logger.info(f"[Phase 4] 正在处理第 {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} 批次，共 {len(chunk)} 位作者...")

        try:
            data = api_client.fetch_authors_batch(chunk)
            if not data or "results" not in data:
                logger.warning(f"[Phase 4] 第 {i // batch_size + 1} 批次返回空数据，跳过。")
                continue

            batch_data = []
            current_time = datetime.now().isoformat()
            for item in data["results"]:
                batch_data.append((
                    clean_id(item.get('id')),
                    clean_id(item.get('orcid')),
                    item.get('display_name') or "Unknown",
                    item.get('works_count', 0),
                    item.get('cited_by_count', 0),
                    safe_get(item, ['summary_stats', 'h_index'], 0),
                    None,
                    current_time
                ))

            if batch_data:
                db.save_author_batch(batch_data)
                processed_count += len(batch_data)  # 更新累计处理数

            logger.info(f"[Phase 4] 第 {i // batch_size + 1} 批次完成，累计处理 {processed_count}/{total} 位作者。")

        except Exception as e:
            logger.error(f"作者同步批次 {i // batch_size + 1} 失败: {e}")

    logger.info(f"[Phase 4] 完成！总共处理 {processed_count} 位作者。")



def fetch_phase5(db, refresh_days=30):
    """
    阶段 5：本地科研影响力动态校准
    目标：基于本地 authorships 关系表，强制刷新机构与渠道的领域活跃指标
    """
    logger.info(f"[Phase 5] 正在启动本地科研影响力动态校准（刷新周期：{refresh_days} 天）...")

    try:
        # 查询最近一次刷新时间
        with db.connection() as conn:
            last_refreshed = conn.execute(
                "SELECT MAX(last_updated) FROM institutions"
            ).fetchone()[0]

        # 如果未超过刷新周期，则跳过执行
        if last_refreshed:
            days_since_last_refresh = (datetime.now() - datetime.fromisoformat(last_refreshed)).days
            if days_since_last_refresh < refresh_days:
                logger.info(f"[Phase 5] 上次刷新距今 {days_since_last_refresh} 天，未达到刷新周期（{refresh_days} 天），跳过执行。")
                return

        # 执行刷新逻辑
        logger.info("[Phase 5] 正在刷新本地统计信息...")
        db.refresh_stats()
        logger.info("[Phase 5] 本地统计信息刷新完成。")

        # 更新时间戳
        current_time = datetime.now().isoformat()
        logger.info("[Phase 5] 正在更新各表的时间戳...")

        with db.connection() as conn:
            conn.execute("UPDATE institutions SET last_updated = ?", (current_time,))
            conn.execute("UPDATE sources SET last_updated = ?", (current_time,))
            conn.execute("UPDATE authors SET last_updated = ?", (current_time,))

        logger.info("[Phase 5] 时间戳更新完成。")
        logger.info("[Phase 5] 本地校准完成，机构、渠道和作者指标已根据当前库容更新。")

    except Exception as e:
        logger.error(f"Phase 5 校准失败: {e}")


