import logging
from kg_utils import GraphEngine, SyncStateManager, Monitor
from builder import KGBuilder
from config import CONFIG_DICT
import sqlite3

def run_pipeline(config):
    """
    知识图谱构建全链路流水线（增强版）
    任务排期逻辑：
    1. 实体同步：建立 Author, Work, Vocabulary 等孤立节点。
    2. 外部拓扑：建立 Authorship 专家署名权重边。
    3. 语义打标：执行 Aho-Corasick 标题回溯扫描，补全 HAS_TOPIC 关系。
    4. 共现拓扑：利用 HAS_TOPIC 边，计算并构建 CO_OCCURRED_WITH 权重网。
    5. 向量桥接：计算词汇间 SBERT 语义相似度边。
    """
    monitor = Monitor()
    engine = GraphEngine(config)
    state = SyncStateManager(engine, config)
    builder = KGBuilder(config, state)

    try:
        # --- 0. 环境地基建设 ---
        with monitor.track("Ensuring Constraints and Indexes"):
            from config import CYPHER_TEMPLATES, SQL_INIT_SCRIPTS
            for cypher in CYPHER_TEMPLATES["INIT_SCHEMA"]:
                engine.execute_query(cypher)

            with sqlite3.connect(config['DB_PATH']) as conn:
                for sql in SQL_INIT_SCRIPTS:
                    conn.execute(sql)
                conn.commit()
            logging.info("SQLite/Neo4j 地基验证完成。")

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
            # 只有在需要彻底重构语义关联时，才重置向量桥接的标记
            # state.reset_marker("vocab_sync")
            # state.reset_marker("author_sync")
            # state.reset_marker("work_sync")
            # state.reset_marker("inst_sync")  # 机构同步
            # state.reset_marker("source_sync")  # 出版源同步
            # state.reset_marker("job_sync")  # 岗位同步
            #
            # # --- 关系与语义标记 ---
            # state.reset_marker("topology_sync")  # 专家-论文-机构拓扑关系
            # state.reset_marker("job_skill_sync")  # 岗位-技能关联
            # state.reset_marker("semantic_bridge_sync")  # 词库间的 Faiss 相似度关联
            for task, sql, cypher, t_field in node_tasks:
                builder.sync_nodes_task(task, sql, cypher, t_field)

        # --- 2. 核心拓扑连接 (Author-Work-Inst) ---
        with monitor.track("Building Topology (Authorship & Affiliation)"):
            builder.build_topology_incremental()

        # --- 3. 语义打标 (含 Aho-Corasick 标题扫描) ---
        # 必须先执行此步，为后续共现频率计算提供基础数据通路
        with monitor.track("Linking Semantics (Work/Job with Title Scan)"):
            # build_work_semantic_links 内部已集成 pyahocorasick 极速扫描逻辑
            builder.build_work_semantic_links()
            builder.build_job_skill_links()

        # --- 4. 【核心新增】共现权重拓扑构建 ---
        # 统计词汇对在 55 万篇论文中的实际共现频率，用于过滤虚假相似度
        with monitor.track("Building Co-occurrence Knowledge Network"):
            builder.build_cooccurrence_links()

        # --- 5. 向量空间桥接 (Vocab-Vocab) ---
        # 利用 SBERT 构建词汇间的跨领域语义跳转边
        with monitor.track("Semantic Bridge (Vocab-Vocab via Model Similarity)"):
            builder.build_semantic_bridge()

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise e
    finally:
        engine.close()
        logging.info("KG Pipeline Process Finished Successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline(CONFIG_DICT)