# generate_kg.py
import logging
from kg_utils import GraphEngine, SyncStateManager, Monitor
from builder import KGBuilder
# 确保从根目录 config 导入
from config import CONFIG_DICT, SQL_QUERIES, CYPHER_TEMPLATES


def run_pipeline(config):
    monitor = Monitor()
    engine = GraphEngine(config)

    # 修正：匹配你 kg_utils.py 中的 __init__(self, engine, config)
    state = SyncStateManager(engine, config)
    builder = KGBuilder(config, state)

    try:
        # 1. 节点同步任务
        node_tasks = [
            ("author_sync", "SYNC_AUTHORS", "MERGE_AUTHOR", "last_updated"),
            ("work_sync", "SYNC_WORKS", "MERGE_WORK", "year"),
            ("inst_sync", "SYNC_INSTITUTIONS", "MERGE_INSTITUTION", "last_updated"),
            ("source_sync", "SYNC_SOURCES", "MERGE_SOURCE", "last_updated"),
            ("job_sync", "SYNC_JOBS", "MERGE_JOB", "crawl_time")
        ]

        with monitor.track("Syncing All Entities"):
            # 改进：如果执行了 DELETE，必须重置 marker
            engine.send_batch("MATCH (n:Vocabulary) DETACH DELETE n", [])
            state.reset_marker("vocab_sync")  # 确保从头开始同步词汇

            builder.sync_nodes_task("vocab_sync", "GET_ALL_VOCAB", "MERGE_VOCAB", "id")

            for task, sql, cypher, t_field in node_tasks:
                builder.sync_nodes_task(task, sql, cypher, t_field)

        # 2. 核心增量拓扑
        with monitor.track("Building Topology"):
            builder.build_topology_incremental()

        # 3. 语义对齐
        with monitor.track("Semantic Bridge"):
            builder.build_semantic_bridge()

    finally:
        engine.close()
        logging.info("Building KG process finished.")


if __name__ == "__main__":
    run_pipeline(CONFIG_DICT)