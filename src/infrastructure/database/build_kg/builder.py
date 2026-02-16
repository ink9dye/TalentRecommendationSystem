import math
import os
import re
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
        方案 A：自适应 K 值搜索，确保跨类型关联的达成率。
        已修复：增加了增量位点 (Marker) 绑定，解决 sqlite3.ProgrammingError。
        """
        idx_path = self.config['VOCAB_INDEX_PATH']
        map_path = self.config['VOCAB_MAP_PATH']

        if not (os.path.exists(idx_path) and os.path.exists(map_path)):
            return

        # 1. 获取增量标记 (Marker)
        task_name = "semantic_bridge_sync"
        marker = self.state.get_marker(task_name)

        index = faiss.read_index(idx_path)
        with open(map_path, 'r', encoding='utf-8') as f:
            all_ids = json.load(f)

        with sqlite3.connect(self.config['DB_PATH']) as conn:
            conn.row_factory = sqlite3.Row
            # 预加载类型映射
            vocab_meta = {str(r[0]): r[1] for r in
                          conn.execute("SELECT voc_id, entity_type FROM vocabulary").fetchall()}

            # 2. 核心修复点：传入 (marker,) 参数
            cursor = conn.execute(SQL_QUERIES["GET_ALL_VOCAB"], (marker,))
            batch = []

            for v in tqdm(cursor, desc="Building Semantic Bridge (Adaptive)"):
                source_id, source_type = str(v['voc_id']), v['entity_type']
                if source_id not in all_ids: continue

                try:
                    vec_idx = all_ids.index(source_id)
                    vec = index.reconstruct(vec_idx).reshape(1, -1)

                    # --- 方案 A 核心：阶梯式扩大搜索范围 ---
                    for k_step in [1000, 3000, 5000]:
                        scores, indices = index.search(vec, k_step)
                        current_links = []
                        for score, n_idx in zip(scores[0], indices[0]):
                            if n_idx == -1: continue
                            neighbor_id = all_ids[n_idx]

                            # 过滤规则：非自身且类型不同
                            if neighbor_id != source_id and vocab_meta.get(neighbor_id) != source_type:
                                current_links.append({"f": int(source_id), "t": int(neighbor_id), "s": float(score)})
                                if len(current_links) >= 3:
                                    break

                        if len(current_links) >= 3 or k_step == 5000:
                            batch.extend(current_links)
                            break

                except Exception:
                    continue

                # 3. 达到批次大小后写入 Neo4j 并更新位点
                if len(batch) >= 500:
                    self.state.engine.send_batch(CYPHER_TEMPLATES["LINK_SIMILAR"], batch)
                    self.state.update_marker(task_name, str(v['id']))
                    batch = []

            if batch:
                self.state.engine.send_batch(CYPHER_TEMPLATES["LINK_SIMILAR"], batch)
                self.state.update_marker(task_name, str(v['id']))

    def build_work_semantic_links(self):
        """
        建立论文与词汇节点的关联边 (Work)-[:HAS_TOPIC]->(Vocabulary)
        """
        sql = "SELECT work_id as wid, concepts_text, keywords_text FROM works"

        with sqlite3.connect(self.config['DB_PATH']) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)

            batch = []
            for row in tqdm(cursor, desc="Linking Works to Knowledge Nodes"):
                # 1. 使用正则支持 |,; 和换行符拆分
                raw_text = f"{row['concepts_text'] or ''}|{row['keywords_text'] or ''}"
                # 拆分并过滤掉空字符、转小写去重
                terms = set([t.strip().lower() for t in re.split(r'[|;,]', raw_text) if t.strip()])

                for term in terms:
                    batch.append({
                        "wid": row['wid'],
                        "term": term
                    })

                    if len(batch) >= 1000:
                        # 建议在执行前确保 Neo4j 索引已建
                        self.state.engine.send_batch(CYPHER_TEMPLATES["LINK_WORK_VOCAB"], batch)
                        batch = []

            if batch:
                self.state.engine.send_batch(CYPHER_TEMPLATES["LINK_WORK_VOCAB"], batch)

    def build_job_skill_links(self):
        """
        建立岗位与技能词汇的关联边 (Job)-[:REQUIRE_SKILL]->(Vocabulary)
        增加了增量标记 (Marker) 处理，修复 Incorrect number of bindings 错误。
        """
        sql = SQL_QUERIES["SYNC_JOB_SKILLS"]

        # 1. 获取当前任务的增量起始点
        task_name = "job_skill_sync"
        marker = self.state.get_marker(task_name)

        with sqlite3.connect(self.config['DB_PATH']) as conn:
            conn.row_factory = sqlite3.Row
            # 2. 传入 marker 参数，修复 bindings 缺失问题
            cursor = conn.execute(sql, (marker,))

            batch = []
            last_row_time = marker

            for row in tqdm(cursor, desc="Linking Jobs to Skills"):
                raw_skills = row['skills']
                if not raw_skills: continue

                # 拆分并清洗：支持中英文逗号、分号及斜杠
                skills = set([s.strip().lower() for s in re.split(r'[,，;；/]', raw_skills) if s.strip()])

                for skill in skills:
                    batch.append({
                        "jid": row['jid'],
                        "term": skill
                    })

                    if len(batch) >= 1000:
                        self.state.engine.send_batch(CYPHER_TEMPLATES["LINK_JOB_VOCAB"], batch)
                        # 3. 批次成功后更新标记
                        last_row_time = str(row['crawl_time'])
                        self.state.update_marker(task_name, last_row_time)
                        batch = []

            if batch:
                self.state.engine.send_batch(CYPHER_TEMPLATES["LINK_JOB_VOCAB"], batch)
                # 更新最后一条记录的标记
                self.state.update_marker(task_name, str(row['crawl_time']))