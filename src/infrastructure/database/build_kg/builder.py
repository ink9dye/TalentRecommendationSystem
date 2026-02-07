import math
import os
import json
import faiss
import sqlite3
import logging
from datetime import datetime
from tqdm import tqdm
from contextlib import contextmanager
from kg_config import SQL_QUERIES, CYPHER_TEMPLATES


# ==========================================
# 算法策略单元：负责量化实体间的关联强度
# ==========================================
class WeightStrategy:
    """
    [策略类] 计算学术贡献权重：实现从平面关系到加权图谱的转化
    """

    @staticmethod
    def calculate(pos_index: int, is_corr: int, is_alpha: int, pub_year: int) -> float:
        """
        严格对齐论文 中的权重建模公式：
        1. 时序衰减: W_time = e^(-0.1 * Δt) —— 保证推荐结果向近期活跃学者倾斜
        2. 署名权重: 基于顺位与通讯标识进行贡献度量化
        """
        # --- 步骤 1: 计算时间衰减系数 ---
        # Δt 为当前年份与发表年份之差，pub_year 缺失则默认 2000 年
        current_year = datetime.now().year
        delta_t = max(0, current_year - (pub_year or 2000))
        # 指数衰减公式：W_time = e^(-0.1 * Δt)
        time_weight = math.exp(-0.1 * delta_t)

        # --- 步骤 2: 计算署名贡献权重 ---
        contribution_weight = 1.0

        # 逻辑：只有当署名不是按字母排序时 (is_alpha == 0)，顺位才具有贡献度参考价值
        if is_alpha == 0:
            # 第一作者 (pos_index == 1) 额外赋予 0.2 的权重偏置
            if pos_index == 1:
                contribution_weight += 0.2

        # 通讯作者 (is_corr == 1) 无论排序规则，均额外加权 0.2，代表其核心科研贡献
        if is_corr == 1:
            contribution_weight += 0.2

        # 最终边权重 = 贡献度 * 时间系数，保留4位小数 [cite: 3]
        return round(contribution_weight * time_weight, 4)


# ==========================================
# 任务执行单元：负责图谱构建的具体流水线任务
# ==========================================
class KGBuilder:
    def __init__(self, config, engine, state_manager):
        """
        [初始化] 建立资源连接
        :param config: 包含路径、BATCH_SIZE 等静态配置
        :param engine: GraphEngine 实例，负责 Neo4j 写入
        :param state_manager: SyncStateManager 实例，负责增量状态维护
        """
        self.config = config
        self.engine = engine
        self.state = state_manager
        self.batch_size = config.get('BATCH_SIZE', 2000)

    @contextmanager
    def _sqlite_cursor(self):
        """
        [工具层] 上下文管理器：确保 SQLite 连接与游标在任务结束或异常时能正确释放
        """
        conn = sqlite3.connect(self.config['DB_PATH'])
        conn.row_factory = sqlite3.Row  # 允许通过字段名访问结果，如 r['aid']
        try:
            yield conn.cursor()
        finally:
            conn.close()

    # ------------------------------------------
    # 任务 1: 物理拓扑关系构建 (AUTHORED)
    # ------------------------------------------
    def build_topology_incremental(self):
        """
        [拓扑层] 增量构建作者-作品拓扑关系：
        通过 ID 偏移量实现断点续传，并在边上注入时序与署名权重
        """
        # 获取上次成功同步的 ID 位置（Marker）
        marker = int(self.state.get_marker("topology_sync"))
        new_marker = marker

        with self._sqlite_cursor() as cursor:
            # 执行 SYNC_AUTHORED_TOPOLOGY (SELECT ... WHERE id > ?)
            cursor.execute(SQL_QUERIES["SYNC_AUTHORED_TOPOLOGY"], (marker,))
            batch = []

            for r in tqdm(cursor, desc="Incremental Topology"):
                # 记录这批数据中的最大同步 ID (sync_id)，用于更新断点
                new_marker = max(new_marker, r['sync_id'])

                # 调用策略类计算当前关系的动态权重属性
                weight = WeightStrategy.calculate(
                    r['pos_index'],
                    r['is_corresponding'],
                    r['is_alphabetical'],
                    r['year']
                )

                # 构造字典对象，SET 语句会将其注入 AUTHORED 边的 pos_weight 属性 [cite: 3]
                batch.append({
                    "aid": r['aid'],
                    "wid": r['wid'],
                    "iid": r['iid'],
                    "sid": r['sid'],
                    "pos_w": weight,
                    "year": r['year'] or 2000
                })

                # 满批提交，优化数据库事务开销
                if len(batch) >= self.batch_size:
                    self.engine.send_batch(CYPHER_TEMPLATES["LINK_AUTHORED_COMPLEX"], batch)
                    batch = []

            # 提交最后一批不满 batch_size 的尾数据
            if batch:
                self.engine.send_batch(CYPHER_TEMPLATES["LINK_AUTHORED_COMPLEX"], batch)

        # 任务全量执行成功后，更新本地 sync_metadata 表的状态
        self.state.update_marker("topology_sync", str(new_marker))

    # ------------------------------------------
    # 任务 2: 跨领域语义对齐 (SIMILAR_TO)
    # ------------------------------------------
    def build_semantic_bridge(self, threshold=0.75):
        """
        跨领域语义对齐 :
        实现“每个词汇节点至少有三个与不同类型词汇节点的关联边”。
        不论是学术侧（Concept/Keyword）还是应用侧（Industry），均执行该对齐逻辑。
        """
        # --- 准备阶段: 加载向量检索索引 ---
        idx_p = os.path.join(self.config['INDEX_DIR'], "vocabulary.faiss")
        map_p = os.path.join(self.config['INDEX_DIR'], "vocabulary_mapping.json")

        if not (os.path.exists(idx_p) and os.path.exists(map_p)):
            logging.warning("FAISS 资源缺失，跳过语义对齐。")
            return

        index = faiss.read_index(idx_p)
        with open(map_p, 'r', encoding='utf-8') as f:
            all_ids = json.load(f)

        with self._sqlite_cursor() as cursor:
            # 加载全量词汇元数据 (ID -> 种类)
            all_vocabs = cursor.execute(SQL_QUERIES["GET_ALL_VOCAB"]).fetchall()
            vocab_meta = {str(r['id']): r['entity_type'] for r in all_vocabs}

            batch = []

            # --- 执行阶段: 为每个词汇建立跨领域连接 ---
            # 无论学术侧还是应用侧，均作为搜索起点
            for v in tqdm(all_vocabs, desc="Full-Spectrum Semantic Alignment"):
                source_id = str(v['id'])
                source_type = v['entity_type']

                if source_id not in all_ids: continue

                # 获取向量并检索候选池（Top 100 确保异类节点存在）
                vec = index.reconstruct(all_ids.index(source_id)).reshape(1, -1)
                scores, indices = index.search(vec, 100)

                links_found = []

                # 按分数降序遍历候选池
                for score, n_idx in zip(scores[0], indices[0]):
                    neighbor_id = all_ids[n_idx]
                    neighbor_type = vocab_meta.get(neighbor_id)

                    # 核心约束：排除自身且类型不同
                    if neighbor_id != source_id and neighbor_type != source_type:
                        # 直接按顺序（最高分）选取
                        links_found.append({"f": int(source_id), "t": int(neighbor_id), "s": float(score)})

                        # 只要攒够 3 个异类节点，立刻停止当前词汇的搜索
                        if len(links_found) >= 3:
                            break

                batch.extend(links_found)

                if len(batch) >= self.batch_size:
                    self.engine.send_batch(CYPHER_TEMPLATES["LINK_SIMILAR"], batch)
                    batch = []

            if batch:
                self.engine.send_batch(CYPHER_TEMPLATES["LINK_SIMILAR"], batch)