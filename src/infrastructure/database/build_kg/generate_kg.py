import logging
from kg_utils import GraphEngine, SyncStateManager, Monitor
from builder import KGBuilder
from config import CONFIG_DICT
import sqlite3

def run_pipeline(config):
    monitor = Monitor()
    engine = GraphEngine(config)
    state = SyncStateManager(engine, config)
    builder = KGBuilder(config, state)

    try:
        # --- 0. 自动建立索引 (在这里改) ---
        with monitor.track("Ensuring Constraints and Indexes"):
            # A. 执行 Neo4j 地基
            from config import CYPHER_TEMPLATES, SQL_INIT_SCRIPTS
            for cypher in CYPHER_TEMPLATES["INIT_SCHEMA"]:
                # 改成这个方法，它不要求带参数，也不会被 if not data 拦截
                engine.execute_query(cypher)

            # B. 执行 SQLite 地基 (通过标准 sqlite3 库)
            with sqlite3.connect(config['DB_PATH']) as conn:
                for sql in SQL_INIT_SCRIPTS:
                    conn.execute(sql)
                conn.commit()
            logging.info("SQLite indexes verified.")
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
            # --- 基础实体同步标记 ---
            state.reset_marker("vocab_sync")
            state.reset_marker("author_sync")
            state.reset_marker("work_sync")
            state.reset_marker("inst_sync")  # 机构同步
            state.reset_marker("source_sync")  # 出版源同步
            state.reset_marker("job_sync")  # 岗位同步

            # --- 关系与语义标记 ---
            state.reset_marker("topology_sync")  # 专家-论文-机构拓扑关系
            state.reset_marker("job_skill_sync")  # 岗位-技能关联
            state.reset_marker("semantic_bridge_sync")  # 词库间的 Faiss 相似度关联

            for task, sql, cypher, t_field in node_tasks:
                builder.sync_nodes_task(task, sql, cypher, t_field)

        # --- 2. 核心拓扑连接 (Author-Work-Inst) ---
        with monitor.track("Building Topology (Author-Work-Inst/Source)"):
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