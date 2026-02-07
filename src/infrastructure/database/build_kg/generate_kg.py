# build_kg/main.py
import logging
from kg_utils import GraphEngine, SyncStateManager, Monitor
from builder import KGBuilder

def run_pipeline(config):
    monitor = Monitor()
    engine = GraphEngine(config)
    state = SyncStateManager(config['DB_PATH'])
    builder = KGBuilder(config, engine, state)

    try:
        with monitor.track("Init Schema"):
            # 执行索引与约束创建...
            pass

        with monitor.track("Incremental Nodes"):
            # 调用 builder 同步 Author, Job 等节点...
            pass

        with monitor.track("Incremental Topology"):
            builder.build_topology_incremental()

        with monitor.track("Semantic Alignment"):
            builder.build_semantic_bridge()

    finally:
        engine.close()
        logging.info("Building KG process finished.")

if __name__ == "__main__":
    from config import CONFIG_DICT
    run_pipeline(CONFIG_DICT)