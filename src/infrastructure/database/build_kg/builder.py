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


# ========================================================================
# 模块一：算法策略单元 (Algorithm Strategy)
# 功能：负责将原始的“论文-作者”二元关系，量化为具备学术影响力的权值。
# ========================================================================
class WeightStrategy:
    """
    [策略类] 计算学术贡献权重：实现从“平面拓扑关系”到“动态加权图谱”的转化。
    该类使用静态方法，方便在流式处理中快速调用，不依赖实例状态。
    """

    @staticmethod
    def calculate(pos_index: int, is_corr: int, is_alpha: int, pub_year: int) -> float:
        """
        核心权重建模公式：权重 = 署名权重*时间系数

        参数说明:
        :param pos_index: 作者署名顺位 (1代表第一作者)
        :param is_corr: 是否为通讯作者 (1: 是, 0: 否)
        :param is_alpha: 是否按姓氏字母排序 (1: 是, 0: 否)。若是，则顺位不代表贡献度。
        :param pub_year: 论文发表年份。
        """

        # --- 步骤 1: 计算时间衰减系数 ---
        # 业务逻辑：科研领域“时效性”很重要，近期的研究对刻画学者当前画像贡献更大。
        current_year = datetime.now().year
        delta_t = max(0, current_year - (pub_year or 2000))
        # 指数衰减公式：W_time = e^(-0.1 * Δt)
        # 示例：今年发表权重为1.0，10年前发表的权重衰减至 e^-1 ≈ 0.36
        time_weight = math.exp(-0.1 * delta_t)

        # --- 步骤 2: 计算署名贡献权重 (Contribution Weight) ---
        contribution_weight = 1.0

        # 逻辑 A：非字母排序下的顺位加权
        # 如果不是字母排序，说明署名顺序经过人为设计，通常一作贡献最大。
        if is_alpha == 0:
            if pos_index == 1:
                contribution_weight += 0.2  # 第一作者额外奖励

        # 逻辑 B：通讯作者加权
        # 无论排序规则如何，通讯作者通常是课题负责人（PI），赋予高权重。
        if is_corr == 1:
            contribution_weight += 0.2

        # 最终边权重 = 贡献度偏置 * 时间衰减，保留4位小数以平衡精度与存储空间
        return round(contribution_weight * time_weight, 4)


# ========================================================================
# 模块二：任务执行单元 (Knowledge Graph Builder)
# 功能：负责从底层数据源（SQLite/FAISS）到图数据库（Neo4j）的 ETL 过程。
# ========================================================================
class KGBuilder:
    def __init__(self, config, engine, state_manager):
        """
        [初始化] 建立资源连接
        :param config: 静态配置，包含 DB 路径、批处理大小等。
        :param engine: GraphEngine 实例，封装了 Neo4j 的驱动与写入接口。
        :param state_manager: SyncStateManager 实例，管理增量同步的“水位线”（Markers）。
        """
        self.config = config
        self.engine = engine
        self.state = state_manager
        self.batch_size = config.get('BATCH_SIZE', 2000)  # 默认每 2000 条记录提交一次事务

    @contextmanager
    def _sqlite_cursor(self):
        """
        [工具层] 上下文管理器：利用 yield 模式确保资源释放。
        即使在同步过程中发生异常，也能保证 SQLite 连接被正确关闭。
        """
        conn = sqlite3.connect(self.config['DB_PATH'])
        conn.row_factory = sqlite3.Row  # 关键设置：允许通过列名访问数据（r['aid']）而非索引（r[0]）
        try:
            yield conn.cursor()
        finally:
            conn.close()

    # --------------------------------------------------------------------
    # 任务 1: 物理拓扑关系构建 (Incremental Topology Building)
    # --------------------------------------------------------------------
    def build_topology_incremental(self):
        """
        [重构后] 建立 Author -> Work 的 AUTHORED 关系
        职责：只定义“如何抓取数据”和“如何计算权重”，把断点交给 state 托管。
        """

        # 1. 定义具体的同步逻辑体
        def _topology_logic(marker):
            new_marker = marker  # 初始化本次任务的终点

            with self._sqlite_cursor() as cursor:
                # 使用传入的 marker 进行查询
                cursor.execute(SQL_QUERIES["SYNC_AUTHORED_TOPOLOGY"], (marker,))
                batch = []

                for r in tqdm(cursor, desc="Processing Topology"):
                    # 跟踪当前处理到的最大 ID
                    if r['sync_id'] > new_marker:
                        new_marker = r['sync_id']

                    # 权重计算逻辑 (保持不变)
                    weight = WeightStrategy.calculate(
                        r['pos_index'], r['is_corresponding'],
                        r['is_alphabetical'], r['year']
                    )

                    batch.append({
                        "aid": r['aid'], "wid": r['wid'], "iid": r['iid'],
                        "sid": r['sid'], "pos_w": weight, "year": r['year'] or 2000
                    })

                    if len(batch) >= self.batch_size:
                        self.engine.send_batch(CYPHER_TEMPLATES["LINK_AUTHORED_COMPLEX"], batch)
                        batch = []

                if batch:
                    self.engine.send_batch(CYPHER_TEMPLATES["LINK_AUTHORED_COMPLEX"], batch)

            return new_marker  # 将最新的断点返回给“管家”

        # 2. 调用托管函数执行任务
        self.state.run_incremental_sync("topology_sync", _topology_logic)
    # --------------------------------------------------------------------
    # 任务 2: 跨领域语义对齐 (Cross-Domain Semantic Alignment)
    # --------------------------------------------------------------------
    def build_semantic_bridge(self, threshold=0.75):
        """
        [语义织网] 利用向量检索技术建立 Vocabulary 间的 SIMILAR_TO 关系。
        目的：打破学术术语（Concept）与行业需求（Industry）的壁垒。
        逻辑约束：每个节点必须寻找至少 3 个“非同类”的近邻。
        """
        # --- 准备阶段: 加载 FAISS 向量索引 ---
        idx_p = os.path.join(self.config['INDEX_DIR'], "vocabulary.faiss")
        map_p = os.path.join(self.config['INDEX_DIR'], "vocabulary_mapping.json")

        if not (os.path.exists(idx_p) and os.path.exists(map_p)):
            logging.warning("FAISS 资源缺失，跳过语义对齐。")
            return

        index = faiss.read_index(idx_p)  # 向量空间索引
        with open(map_p, 'r', encoding='utf-8') as f:
            all_ids = json.load(f)  # 向量 ID 到 业务 ID 的映射

        with self._sqlite_cursor() as cursor:
            # 加载词汇元数据，用于在检索时识别“异类”节点（如 Concept vs Industry）
            all_vocabs = cursor.execute(SQL_QUERIES["GET_ALL_VOCAB"]).fetchall()
            vocab_meta = {str(r['id']): r['entity_type'] for r in all_vocabs}

            batch = []

            # --- 执行阶段: 向量检索与跨类匹配 ---
            for v in tqdm(all_vocabs, desc="Full-Spectrum Semantic Alignment"):
                source_id = str(v['id'])
                source_type = v['entity_type']

                if source_id not in all_ids: continue

                # 从 FAISS 中提取该节点的向量，并搜索 Top 100 候选者
                # 取 100 是为了确保在过滤掉同类节点后，仍有足够的异类节点可选
                vec = index.reconstruct(all_ids.index(source_id)).reshape(1, -1)
                scores, indices = index.search(vec, 100)

                links_found = []

                # 遍历候选池，筛选出不同类型的词汇
                for score, n_idx in zip(scores[0], indices[0]):
                    neighbor_id = all_ids[n_idx]
                    neighbor_type = vocab_meta.get(neighbor_id)

                    # 核心约束条件：
                    # 1. 不能是自己
                    # 2. 必须是不同领域（例如：Python技能 -> 软件开发行业）
                    if neighbor_id != source_id and neighbor_type != source_type:
                        links_found.append({
                            "f": int(source_id),
                            "t": int(neighbor_id),
                            "s": float(score)
                        })

                        # 策略：每个词汇仅建立 3 条最相关的异类连接，保持图的稀疏性
                        if len(links_found) >= 3:
                            break

                batch.extend(links_found)

                if len(batch) >= self.batch_size:
                    self.engine.send_batch(CYPHER_TEMPLATES["LINK_SIMILAR"], batch)
                    batch = []

            if batch:
                self.engine.send_batch(CYPHER_TEMPLATES["LINK_SIMILAR"], batch)

    # --------------------------------------------------------------------
    # 通用任务: 节点同步 (Generic Node Sync)
    # --------------------------------------------------------------------
    def sync_nodes_task(self, task_name: str, sql_key: str, cypher_key: str, time_field: str):
        """
        [重构后] 通用的节点同步逻辑
        """

        def _node_sync_logic(marker):
            current_max = marker
            with self._sqlite_cursor() as cursor:
                cursor.execute(SQL_QUERIES[sql_key], (marker,))
                batch = []
                for r in tqdm(cursor, desc=f"Syncing {task_name}"):
                    val = str(r[time_field])
                    if val > current_max:
                        current_max = val

                    batch.append(dict(r))
                    if len(batch) >= self.batch_size:
                        self.engine.send_batch(CYPHER_TEMPLATES[cypher_key], batch)
                        batch = []
                if batch:
                    self.engine.send_batch(CYPHER_TEMPLATES[cypher_key], batch)
            return current_max

        # 托管执行
        self.state.run_incremental_sync(task_name, _node_sync_logic)