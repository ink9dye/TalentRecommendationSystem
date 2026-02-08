import math
import os
import json
import faiss
import sqlite3
from tqdm import tqdm
from datetime import datetime
from config import SQL_QUERIES, CYPHER_TEMPLATES


class WeightStrategy:
    """
    人才推荐系统权重计算策略类。
    用于量化专家（Author）在某项研究（Work）中的贡献度和时效性。
    """

    @staticmethod
    def calculate(pos_index: int, is_corr: int, is_alpha: int, pub_year: int) -> float:
        """
        核心权重建模公式：W = (Base + Bonus) * e^(-0.1 * Δt)

        Args:
            pos_index: 作者排名，1 表示第一作者。
            is_corr: 是否为通讯作者 (1: 是, 0: 否)。
            is_alpha: 是否按姓氏字母排序 (1: 是, 0: 否)。如果为 1，则首位作者不设额外加成。
            pub_year: 论文发表年份。

        Returns:
            round(float, 4): 计算后的贡献权重，保留四位小数。
        """
        current_year = datetime.now().year
        # 计算发表间隔，最小值为 0 防止年份数据异常
        delta_t = max(0, current_year - (pub_year or 2000))

        # 指数时间衰减：每早 10 年，权重约变为原来的 1/e (~36%)
        time_weight = math.exp(-0.1 * delta_t)

        # 贡献度基数计算：
        # 1. 初始权重为 1.0
        # 2. 如果非字母排序且为第一作者，加成 0.2 (权重 1.2)
        # 3. 如果是通讯作者，额外加成 0.2
        contribution = 1.0 + (0.2 if is_alpha == 0 and pos_index == 1 else 0) + (0.2 if is_corr == 1 else 0)

        return round(contribution * time_weight, 4)


class KGBuilder:
    """
    知识图谱构建执行器。
    负责将本地 SQLite 中的结构化数据通过增量或分批的方式推送至 Neo4j 图数据库。
    """

    def __init__(self, config, state_manager):
        """
        初始化构建器。

        Args:
            config: 全局配置字典 (包含路径、Neo4j 连接等)。
            state_manager: SyncStateManager 实例，处理增量标记(Marker)和 Neo4j 批处理。
        """
        self.config = config
        self.state = state_manager

    def sync_nodes_task(self, task_name, sql_key, cypher_key, time_field):
        """
        通用节点同步任务。
        将简单的实体表（如 Author, Institution）同步至 Neo4j。
        """
        self.state.sync_engine(task_name, SQL_QUERIES[sql_key], CYPHER_TEMPLATES[cypher_key], time_field)

    def build_topology_incremental(self):
        """
        增量构建复杂拓扑关系（Authorship, Affiliation, Publication）。
        由于涉及关系权重计算，需在数据传输前进行 row_processor 处理。
        """

        def _proc(row):
            # 将多维属性转化为图谱中的边权重 pos_w
            row['pos_w'] = WeightStrategy.calculate(
                row['pos_index'], row['is_corresponding'], row['is_alphabetical'], row['year']
            )
            # 容错处理：年份缺失时设为 2000
            row['year'] = row['year'] or 2000
            return row

        # 这里的 sync_id 通常是 authorships 表的自增 ID
        self.state.sync_engine("topology_sync", SQL_QUERIES["SYNC_AUTHORED_TOPOLOGY"],
                               CYPHER_TEMPLATES["LINK_AUTHORED_COMPLEX"], "sync_id", _proc)

    def build_semantic_bridge(self):
        """
        构建语义桥接（跨领域词汇关联）。
        利用 Faiss 向量索引在不同类型的实体（如领域标签、技能）之间建立 SIMILAR_TO 关系。
        这是系统实现“隐式召回”的关键步骤。
        """
        idx_path = self.config['VOCAB_INDEX_PATH']
        map_path = self.config['VOCAB_MAP_PATH']

        # 检查向量索引文件是否存在，不存在则跳过语义对齐
        if not (os.path.exists(idx_path) and os.path.exists(map_path)):
            return

        # 加载向量索引及 ID 映射
        index = faiss.read_index(idx_path)
        with open(map_path, 'r', encoding='utf-8') as f:
            all_ids = json.load(f)

        with sqlite3.connect(self.config['DB_PATH']) as conn:
            conn.row_factory = sqlite3.Row

            # 【性能与内存优化】
            # 1. 预先加载词汇实体的类型映射 (entity_type)，用于在建立相似关系时过滤掉相同类型的实体
            # 例如：只在“技能”和“学科”之间建连，不在“技能”和“技能”之间建连，以保持图谱稀疏度。
            vocab_meta = {str(r[0]): r[1] for r in
                          conn.execute("SELECT id, entity_type FROM vocabulary_table").fetchall()}

            # 2. 使用迭代器分批读取 SQLite 数据，避免大规模词库撑爆内存
            cursor = conn.execute(SQL_QUERIES["GET_ALL_VOCAB"])

            batch = []
            for v in tqdm(cursor, desc="Building Semantic Bridge"):
                source_id, source_type = str(v['id']), v['entity_type']
                if source_id not in all_ids: continue

                # 获取词汇对应的向量索引位置并检索
                try:
                    # 提示：如果 all_ids 很大，此处建议先转字典，否则 .index() 是 O(N) 复杂度
                    vec_idx = all_ids.index(source_id)
                    # 重构向量并进行近邻搜索 (Top 50)
                    vec = index.reconstruct(vec_idx).reshape(1, -1)
                    scores, indices = index.search(vec, 50)
                except:
                    continue

                links = 0
                for score, n_idx in zip(scores[0], indices[0]):
                    if n_idx == -1: continue  # Faiss 空返回处理
                    neighbor_id = all_ids[n_idx]

                    # 语义过滤策略：
                    # 1. 排除自身
                    # 2. 类型不同（如：将 Work 关键词关联到 Job 技能标签上）
                    if neighbor_id != source_id and vocab_meta.get(neighbor_id) != source_type:
                        batch.append({"f": int(source_id), "t": int(neighbor_id), "s": float(score)})
                        links += 1
                        # 限制每个实体最多建立 3 条语义桥接，防止关系爆炸，保证核心召回准确性
                        if links >= 3: break

                # 达到批次大小后写入 Neo4j
                if len(batch) >= 500:
                    self.state.engine.send_batch(CYPHER_TEMPLATES["LINK_SIMILAR"], batch)
                    batch = []

            # 处理余下的批次
            if batch:
                self.state.engine.send_batch(CYPHER_TEMPLATES["LINK_SIMILAR"], batch)