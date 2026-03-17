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
        """
        初始化 Neo4j 驱动。
        修正：移除了无效的 socket_timeout，使用官方支持的超时参数。
        """
        self.driver = GraphDatabase.driver(
            config['NEO4J_URI'],
            auth=(config['NEO4J_USER'], config['NEO4J_PASSWORD']),
            # --- 核心修复：使用 5.x 驱动支持的参数 ---
            connection_timeout=60.0,      # 建立连接的超时时间 (秒)
            max_transaction_retry_time=60.0, # 允许事务在网络波动或繁忙时自动重试的总时间
            max_connection_lifetime=3600, # 限制连接寿命，防止陈旧连接导致假死
            keep_alive=True               # 保持 TCP 长连接，防止因静默被防火墙切断
        )
        self.db_name = config['NEO4J_DATABASE']

    def execute_query(self, query: str):
        """专门用于执行不带参数的 Schema 指令（如创建索引）"""
        with self.driver.session(database=self.db_name) as session:
            session.run(query)

    def send_batch(self, query: str, data: List[Dict]):
        """执行带参数的批量写入"""
        if not data: return
        with self.driver.session(database=self.db_name) as session:
            session.run(query, {"data": data})

    def close(self):
        """关闭驱动连接"""
        self.driver.close()

class SyncStateManager:
    def __init__(self, engine: GraphEngine, config: Dict):
        self.db_path = config['DB_PATH']
        self.engine = engine
        self.batch_size = config.get('BATCH_SIZE', 2000)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS sync_metadata (task_name TEXT PRIMARY KEY, last_marker TEXT)")

    def get_marker(self, task: str) -> str:
        """
        获取同步位点标记。
        针对不同的任务类型，如果位点不存在，则返回对应的初始默认值。
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT last_marker FROM sync_metadata WHERE task_name=?", (task,)).fetchone()

            if row:
                return row[0]

            # 核心逻辑：判定该任务是否基于“整数 ID”进行排序
            # 增加了 "voc_id" 和 "ship_id" 以适配新的数据库架构
            is_numeric_id = any(x in task.lower() for x in [
                "topology", "sync_id", "id", "vocab", "work", "voc_id", "ship_id"
            ])

            # 如果是基于 ID 的任务，初始位点为 "0"；如果是基于时间的任务，初始位点为 1970年
            return "0" if is_numeric_id else "1970-01-01 00:00:00"

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
                    # 返回 None 表示跳过该行（如词汇清洗未通过）
                    if row_data is None:
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