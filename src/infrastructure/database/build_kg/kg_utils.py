import sqlite3
import time
import psutil
import logging
from tqdm import tqdm
from typing import List, Dict, Callable
from neo4j import GraphDatabase
from contextlib import contextmanager


class GraphEngine:
    def __init__(self, config):
        self.driver = GraphDatabase.driver(config['NEO4J_URI'], auth=(config['NEO4J_USER'], config['NEO4J_PASSWORD']))
        self.db_name = config['NEO4J_DATABASE']

    def execute_query(self, query: str):
        """专门用于执行不带参数的 Schema 指令（如创建索引）"""
        with self.driver.session(database=self.db_name) as session:
            session.run(query)

    def send_batch(self, query: str, data: List[Dict]):
        if not data: return
        with self.driver.session(database=self.db_name) as session:
            session.run(query, {"data": data})

    def close(self):
        self.driver.close()


class SyncStateManager:
    def __init__(self, engine: GraphEngine, config: Dict):
        self.db_path = config['DB_PATH']
        self.engine = engine
        self.batch_size = config.get('BATCH_SIZE', 2000)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS sync_metadata (task_name TEXT PRIMARY KEY, last_marker TEXT)")

    def get_marker(self, task: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT last_marker FROM sync_metadata WHERE task_name=?", (task,)).fetchone()
            # 增加对 "vocab" 和 "work" 的判定，确保它们默认起始值为 "0" (整数模式)
            return row[0] if row else (
                "0" if any(x in task.lower() for x in ["topology", "sync_id", "id", "vocab", "work"])
                else "1970-01-01 00:00:00")

    def update_marker(self, task: str, marker: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO sync_metadata VALUES (?, ?)", (task, str(marker)))

    def reset_marker(self, task: str):
        """重置特定任务的同步位点"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sync_metadata WHERE task_name=?", (task,))
            logging.info(f"Marker for task [{task}] has been reset.")

    def sync_engine(self, task_name: str, sql: str, cypher: str, time_field: str, row_processor: Callable = None):
        marker = self.get_marker(task_name)
        new_marker = marker

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 重要：在执行前提醒开发者 SQL 必须包含 ORDER BY {time_field} ASC
            # 否则增量逻辑在中断恢复时会丢失数据
            cursor.execute(sql, (marker,))

            batch = []
            # 使用迭代器而非 fetchall()，保护内存
            for r in tqdm(cursor, desc=f"Syncing {task_name}"):
                row_data = dict(r)
                curr = str(row_data[time_field])

                # 只有当当前行处理成功后，才考虑更新 marker
                if row_processor:
                    try:
                        row_data = row_processor(row_data)
                    except Exception as e:
                        logging.error(f"Error processing row: {e}")
                        continue

                batch.append(row_data)

                if len(batch) >= self.batch_size:
                    self.engine.send_batch(cypher, batch)
                    # 只有在批次成功发送后，才记录该批次最后一条数据的时间戳
                    new_marker = curr
                    self.update_marker(task_name, new_marker)
                    batch = []

            if batch:
                self.engine.send_batch(cypher, batch)
                # 记录最后一条数据的标记
                last_row_marker = str(dict(r)[time_field])
                self.update_marker(task_name, last_row_marker)

class Monitor:
    """
    性能监控器：用于追踪任务运行耗时和资源状态
    """
    def __init__(self):
        self.stats = {}
        # 尝试获取当前进程，用于后续可能的内存监控扩展
        self.process = psutil.Process()

    @contextmanager
    def track(self, name: str):
        """
        上下文管理器：使用 with monitor.track("任务名") 记录耗时
        """
        start_time = time.time()
        logging.info(f"--- Starting Task: [{name}] ---")
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.stats[name] = duration
            logging.info(f"--- Completed Task: [{name}] | Time Spent: {duration:.2f}s ---")