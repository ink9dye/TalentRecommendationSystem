import logging
from kg_utils import GraphEngine, SyncStateManager, Monitor
from builder import KGBuilder
from config import CONFIG_DICT

def run_pipeline(config):
    monitor = Monitor()
    engine = GraphEngine(config)
    state = SyncStateManager(engine, config)
    builder = KGBuilder(config, state)

    try:
        # --- 0. 自动建立索引 (关键步骤) ---
        with monitor.track("Ensuring Constraints and Indexes"):

            # 为 Vocabulary 的 term 建立索引，支撑 build_work_semantic_links
            engine.send_batch("CREATE INDEX vocab_term_idx IF NOT EXISTS FOR (v:Vocabulary) ON (v.term)", [])
            # 为 Work 和 Job 的 ID 建立索引，支撑关系建立
            engine.send_batch("CREATE CONSTRAINT work_id_unique IF NOT EXISTS FOR (w:Work) REQUIRE w.id IS UNIQUE", [])
            engine.send_batch("CREATE CONSTRAINT job_id_unique IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE", [])

        # --- 1. 节点同步任务 ---
        node_tasks = [
            ("vocab_sync", "GET_ALL_VOCAB", "MERGE_VOCAB", "id"),
            ("author_sync", "SYNC_AUTHORS", "MERGE_AUTHOR", "last_updated"),
            ("work_sync", "SYNC_WORKS", "MERGE_WORK", "year"),
            ("inst_sync", "SYNC_INSTITUTIONS", "MERGE_INSTITUTION", "last_updated"),
            ("source_sync", "SYNC_SOURCES", "MERGE_SOURCE", "last_updated"),
            ("job_sync", "SYNC_JOBS", "MERGE_JOB", "crawl_time")
        ]

        with monitor.track("Syncing All Entities"):
            # 警告：只有在需要彻底重构词库时才取消下面的注释
            # state.reset_marker("vocab_sync")
            # state.reset_marker("author_sync")
            # state.reset_marker("work_sync")

            for task, sql, cypher, t_field in node_tasks:
                builder.sync_nodes_task(task, sql, cypher, t_field)

        # --- 2. 核心拓扑连接 (Author-Work-Inst) ---
        with monitor.track("Building Topology (Author-Work-Inst)"):
            builder.build_topology_incremental()

        # --- 3. 语义知识连接 (Work-Vocab & Job-Vocab) ---
        # 这是你新写的逻辑，必须在这里显式调用
        with monitor.track("Linking Semantics (Work/Job to Vocab)"):
            builder.build_work_semantic_links()
            builder.build_job_skill_links()

        # --- 4. 向量空间桥接 (Vocab-Vocab) ---
        with monitor.track("Semantic Bridge (Vocab-Vocab via Faiss)"):
            builder.build_semantic_bridge()

    finally:
        engine.close()
        logging.info("Building KG process finished.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline(CONFIG_DICT)