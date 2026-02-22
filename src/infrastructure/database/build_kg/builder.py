import math
import os
import re
import json
import logging
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
        语义桥接终极版：通过向量空间计算词汇间的相似度，并在图谱中建立连接。

        【解决的核心痛点】：
        1. 修复 KeyError: 直接从 config 模块导入无法在 CONFIG_DICT 找到的模板。
        2. 绕过 reconstruct 限制: 既然索引不能直接反推向量，我们就重新编码。
        3. 增量更新: 依靠 SQLite 记录进度，防止重复劳动。
        """
        import numpy as np
        from sentence_transformers import SentenceTransformer
        # 核心：直接引用模板字典，解决 self.config 缺少该键的问题
        from config import CYPHER_TEMPLATES

        # --- 第一步：参数准备（全部对齐 config.py） ---
        idx_path = self.config['VOCAB_INDEX_PATH']  # Faiss 索引文件路径
        db_path = self.config['DB_PATH']  # SQLite 数据库路径
        sbert_dir = self.config['SBERT_DIR']  # 模型本地缓存目录
        # 引用全局设定的批次大小（通常为 5000），保证写入 Neo4j 时的效率
        batch_size_neo4j = self.config.get('BATCH_SIZE', 5000)

        # 安全检查：如果没有索引文件，后续检索无法进行
        if not os.path.exists(idx_path):
            logging.warning(f"Vocab index path [{idx_path}] not found, skipping.")
            return

        # --- 第二步：断点续传状态初始化 ---
        task_name = "semantic_bridge_sync"
        # 从 SQLite 中读取上一次成功同步到的词汇 ID (Marker)
        marker_str = self.state.get_marker(task_name)
        marker = int(marker_str) if marker_str.isdigit() else 0

        # 加载 HNSW 索引（用于极速寻找最近邻）
        index = faiss.read_index(idx_path)
        # 修复加载逻辑：使用‘模型名 + 路径’，解决 OSError 找不到权重文件的问题
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder=sbert_dir)

        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # 使返回结果可以通过字段名访问

            # --- 第三步：预加载元数据（性能优化关键） ---
            logging.info("[*] 正在加载词汇元数据以执行语义桥接...")
            # 我们需要 entity_type 来判断两个词是不是“同一类”
            all_rows = conn.execute("SELECT voc_id, term, entity_type FROM vocabulary").fetchall()
            # 将所有词汇类型存入字典，后续在循环中 O(1) 速度查询，避免频繁查库
            vocab_meta = {int(r['voc_id']): r['entity_type'] for r in all_rows}

            # 增量过滤：只处理 ID 大于上一次标记（Marker）的词汇
            pending_rows = [r for r in all_rows if int(r['voc_id']) > marker]
            if not pending_rows:
                logging.info("[*] 语义桥接任务已完成，无需操作。")
                return

            # --- 第四步：核心批量处理循环 ---
            # 为了提速，我们每 1024 个词汇作为一个块（Chunk）进行处理
            encode_chunk_size = 1024
            total_pending = len(pending_rows)
            neo4j_batch = []  # 准备发往 Neo4j 的数据池

            # 进度条显示：让 6.6 万条数据的同步过程可视化
            for start_idx in tqdm(range(0, total_pending, encode_chunk_size), desc="Building Semantic Bridge"):
                # 获取当前批次的待处理词汇
                current_chunk = pending_rows[start_idx: start_idx + encode_chunk_size]

                # 【重算向量】：批量生成向量，避开索引无法 reconstruct 的死穴
                chunk_texts = [r['term'] for r in current_chunk]
                chunk_embeddings = model.encode(chunk_texts, batch_size=128, show_progress_bar=False)
                # 必须归一化：将向量长度变为 1，从而让 Dot Product 等价于余弦相似度
                faiss.normalize_L2(chunk_embeddings)

                # 【批量检索】：一次性寻找当前块内所有词的前 100 个邻居
                all_scores, all_labels = index.search(chunk_embeddings.astype('float32'), 100)

                # 遍历当前块中的每个词，筛选最合适的 3 个邻居
                for idx, row in enumerate(current_chunk):
                    source_id = int(row['voc_id'])
                    source_type = row['entity_type']

                    scores = all_scores[idx]
                    labels = all_labels[idx]

                    current_links = []
                    # 遍历检索出的邻居（按相似度由高到低排列）
                    for score, neighbor_id in zip(scores, labels):
                        neighbor_id = int(neighbor_id)

                        # 过滤无效项（-1 代表没搜到）以及自身
                        if neighbor_id == -1 or neighbor_id == source_id:
                            continue

                        # 【核心桥接逻辑】：只连接“不同类型”的词
                        # 例如：让“Java(技能)”去连接“后端开发(岗位)”，而不是连接“Python(技能)”
                        if vocab_meta.get(neighbor_id) != source_type:
                            current_links.append({
                                "f": source_id,  # From (源词 ID)
                                "t": neighbor_id,  # To (目标词 ID)
                                "s": float(score)  # Similarity Score (相似度权重)
                            })
                            # 限制数量：每个词最多建立 3 个最强跨类关联，防止图谱爆炸
                            if len(current_links) >= 3:
                                break

                    # 将本次筛选出的 3 个关联加入待推送大池子
                    neo4j_batch.extend(current_links)

                    # --- 第五步：写入数据库并记录进度 ---
                    # 当池子里的关系累积到 5000 条时，统一写入一次 Neo4j
                    if len(neo4j_batch) >= batch_size_neo4j:
                        self.state.engine.send_batch(CYPHER_TEMPLATES["LINK_SIMILAR"], neo4j_batch)
                        # 重要：每成功写入一批，就更新一次 SQLite 里的 Marker
                        self.state.update_marker(task_name, str(source_id))
                        neo4j_batch = []

            # --- 第六步：清理收尾 ---
            # 处理循环结束后，池子里可能还剩下没凑够 5000 条的“尾数”
            if neo4j_batch:
                self.state.engine.send_batch(CYPHER_TEMPLATES["LINK_SIMILAR"], neo4j_batch)
                # 将最后一个处理的 ID 作为最终 Marker 保存
                self.state.update_marker(task_name, str(pending_rows[-1]['voc_id']))
    def build_work_semantic_links(self):
        """
        建立论文与词汇节点的关联边 (Work)-[:HAS_TOPIC]->(Vocabulary)
        """
        sql = "SELECT work_id as id, concepts_text, keywords_text FROM works"

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
                        "id": row['id'],
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
                        "id": row['id'],
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