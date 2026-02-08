# build_kg/kg_utils.py
import sqlite3
import time
import psutil
import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase
from contextlib import contextmanager

class GraphEngine:
    def __init__(self, config):
        self.driver = GraphDatabase.driver(config['NEO4J_URI'], auth=(config['NEO4J_USER'], config['NEO4J_PASSWORD']))
        self.db_name = config['NEO4J_DATABASE']

    def send_batch(self, query: str, data: List[Dict]):
        with self.driver.session(database=self.db_name) as session:
            session.run(query, {"data": data})

    def close(self):
        self.driver.close()

class SyncStateManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS sync_metadata (task_name TEXT PRIMARY KEY, last_marker TEXT)")

    def get_marker(self, task: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT last_marker FROM sync_metadata WHERE task_name=?", (task,)).fetchone()
            return row[0] if row else ("0" if "topology" in task else "1970-01-01 00:00:00")

    def update_marker(self, task: str, marker: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO sync_metadata VALUES (?, ?)", (task, str(marker)))

    def run_incremental_sync(self, task_name, sync_logic_callback):
        """
        [新增] 增量同步管理逻辑：
        1. 自动获取旧断点 (Marker)
        2. 执行传入的业务代码 (Callback)
        3. 如果执行成功且有进度更新，自动保存新断点
        """
        # 1. 自动读取断点
        last_marker = self.get_marker(task_name)

        # 2. 执行业务逻辑，业务逻辑需返回最新的 marker 值
        # 这里把 last_marker 传给 builder 里的具体函数
        new_marker = sync_logic_callback(last_marker)

        # 3. 如果任务执行完毕且返回了有效的新断点，则持久化保存
        if new_marker is not None and str(new_marker) > str(last_marker):
            self.update_marker(task_name, new_marker)
            logging.info(f"Task [{task_name}] sync completed. Marker updated: {last_marker} -> {new_marker}")

class Monitor:
    def __init__(self):
        self.stats = {}
        self.process = psutil.Process()

    @contextmanager
    def track(self, name: str):
        t0 = time.time()
        yield
        self.stats[name] = time.time() - t0
        logging.info(f"Task {name} completed in {self.stats[name]:.2f}s")