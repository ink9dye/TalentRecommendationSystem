# generate_kg.py 补全版
import logging
from kg_utils import GraphEngine, SyncStateManager, Monitor
from builder import KGBuilder


def run_pipeline(config):
    monitor = Monitor()
    engine = GraphEngine(config)
    state = SyncStateManager(config['DB_PATH'])
    builder = KGBuilder(config, engine, state)

    try:
        # --- 1. 初始化 Schema (赋予约束以保证性能) ---
        with monitor.track("Init Schema"):
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (w:Work) REQUIRE w.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Vocabulary) REQUIRE v.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE"
            ]
            for cmd in constraints:
                engine.send_batch(cmd, [])  # 发送空数据执行命令

        # --- 2. 增量节点同步 (搬运实体) ---
        with monitor.track("Incremental Nodes"):
            # 定义 任务名, SQL键, Cypher键, 时间戳字段名
            node_tasks = [
                ("author_sync", "SYNC_AUTHORS", "MERGE_AUTHOR", "last_updated"),
                ("work_sync", "SYNC_WORKS", "MERGE_WORK", "year"),  # Works 可能没时间戳，用 year 或 ID
                ("inst_sync", "SYNC_INSTITUTIONS", "MERGE_INSTITUTION", "last_updated"),
                ("source_sync", "SYNC_SOURCES", "MERGE_SOURCE", "last_updated"),
                ("job_sync", "SYNC_JOBS", "MERGE_JOB", "crawl_time")
            ]

            # 首先同步词汇表（全量同步，不设 Marker）
            builder.engine.send_batch("MATCH (n:Vocabulary) DETACH DELETE n", [])  # 重置词汇
            builder.sync_nodes_task("vocab_sync", "GET_ALL_VOCAB", "MERGE_VOCAB", "id")

            # 增量同步其他业务实体
            for task, sql, cypher, t_field in node_tasks:
                builder.sync_nodes_task(task, sql, cypher, t_field)

        # --- 3. 核心：带权重的增量拓扑构建 ---
        with monitor.track("Incremental Topology"):
            builder.build_topology_incremental()

        # --- 4. 语义对齐：跨领域织网 ---
        with monitor.track("Semantic Alignment"):
            builder.build_semantic_bridge()

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
    finally:
        engine.close()
        logging.info("Building KG process finished.")


if __name__ == "__main__":
    # 请确保你的 config.py 存在并定义了 CONFIG_DICT
    # 包含 DB_PATH, NEO4J_URI, INDEX_DIR 等
    from config import CONFIG_DICT

    run_pipeline(CONFIG_DICT)