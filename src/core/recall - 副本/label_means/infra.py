import collections
import json
import os
import sqlite3

import faiss
import numpy as np
from py2neo import Graph

from config import (
    CONFIG_DICT,
    JOB_INDEX_PATH,
    JOB_MAP_PATH,
    VOCAB_INDEX_PATH,
    DB_PATH,
    VOCAB_STATS_DB_PATH,
    INDEX_DIR,
)


class LabelMeansInfra:
    """
    标签路召回所需的底层资源初始化与访问封装。

    主要职责：
      - 管理 Neo4j 图连接 Graph；
      - 加载 Faiss 索引（Job 与 Vocabulary）与向量矩阵；
      - 建立 vocabulary 的 voc_id -> 向量下标映射；
      - 加载簇索引（cluster_members / voc_to_clusters / cluster_centroids）；
      - 提供简单的节点计数接口，用于 IDF 计算。
    """

    def __init__(self) -> None:
        self.graph = None
        self.job_index = None
        self.job_id_map = []
        self.vocab_index = None
        self.all_vocab_vectors = None
        self.vocab_to_idx = {}
        self.stats_conn = None
        self.cluster_members = collections.defaultdict(list)
        self.voc_to_clusters = collections.defaultdict(list)
        self.cluster_centroids = None

    def init_resources(self) -> None:
        """
        【资源初始化】解决 Faiss ID 与 向量矩阵的同步问题
        1. Faiss 索引：仅用于快速 Top-K 检索。
        2. .npy 矩阵：存储原始归一化向量，用于计算词汇间的语义紧密度（Proximity）。
        3. SQLite 映射：确保矩阵行号(Index)与数据库 voc_id 严格对齐。
        """
        try:
            # A. 初始化图数据库连接
            self.graph = Graph(
                CONFIG_DICT["NEO4J_URI"],
                auth=(CONFIG_DICT["NEO4J_USER"], CONFIG_DICT["NEO4J_PASSWORD"]),
                name=CONFIG_DICT["NEO4J_DATABASE"],
            )

            # B. 加载岗位描述索引（用于第一阶段：领域探测）
            self.job_index = faiss.read_index(JOB_INDEX_PATH)
            with open(JOB_MAP_PATH, "r", encoding="utf-8") as f:
                self.job_id_map = json.load(f)

            # C. 加载词汇索引与向量快照
            self.vocab_index = faiss.read_index(VOCAB_INDEX_PATH)
            vec_path = VOCAB_INDEX_PATH.replace(".faiss", "_vectors.npy")
            if not os.path.exists(vec_path):
                raise FileNotFoundError(f"未发现向量快照: {vec_path}，请先运行 build_vector_index.py。")

            # 直接加载原始向量矩阵，避开 IndexIDMap 不支持 reconstruct 的局限
            self.all_vocab_vectors = np.load(vec_path).astype("float32")

            # D. 建立 { 'voc_id': 矩阵行下标 } 映射
            # 必须 ORDER BY voc_id 以匹配向量编码时的顺序
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("SELECT voc_id FROM vocabulary ORDER BY voc_id ASC").fetchall()
                self.vocab_to_idx = {str(r[0]): i for i, r in enumerate(rows)}
            self.stats_conn = sqlite3.connect(VOCAB_STATS_DB_PATH, check_same_thread=False)

            # E. 概念簇缓存：cluster_id -> [voc_id]，voc_id -> [(cluster_id, score)]
            self.cluster_members = collections.defaultdict(list)
            self.voc_to_clusters = collections.defaultdict(list)
            try:
                cur = self.stats_conn.execute("SELECT cluster_id, voc_id FROM cluster_members")
                for cid, vid in cur:
                    self.cluster_members[int(cid)].append(int(vid))
                cur = self.stats_conn.execute("SELECT voc_id, cluster_id, score FROM vocabulary_cluster")
                for vid, cid, sc in cur:
                    self.voc_to_clusters[int(vid)].append((int(cid), float(sc)))
            except Exception:
                # 概念簇索引缺失时退化为无簇模式
                self.cluster_members = collections.defaultdict(list)
                self.voc_to_clusters = collections.defaultdict(list)

            # F. 概念簇中心向量：供 JD 语义 gating 使用
            try:
                centroids_path = os.path.join(INDEX_DIR, "cluster_centroids.npy")
                if os.path.exists(centroids_path):
                    centroids = np.load(centroids_path).astype(np.float32)
                    if centroids.ndim == 2 and centroids.size > 0:
                        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
                        norms[norms == 0] = 1.0
                        self.cluster_centroids = centroids / norms
                    else:
                        self.cluster_centroids = None
                else:
                    self.cluster_centroids = None
            except Exception:
                self.cluster_centroids = None

            print("[OK] 标签路资源初始化完成")
        except Exception as e:  # pragma: no cover - 初始化失败时仅打印日志
            print(f"[Error] 资源加载失败: {e}")
            self.graph = None

    def get_node_count(self, label: str) -> float:
        """统计图谱节点总数，作为计算 IDF 的分母。"""
        try:
            if not self.graph:
                return 1000000.0
            res = self.graph.run(f"MATCH (n:{label}) RETURN count(n) AS c").data()
            return float(res[0]["c"]) if res else 1000000.0
        except Exception:
            return 1000000.0

