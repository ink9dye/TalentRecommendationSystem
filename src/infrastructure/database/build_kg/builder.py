import math
import os
import re
import json
import logging
import faiss
import sqlite3
from tqdm import tqdm
from datetime import datetime
from config import SQL_QUERIES, CYPHER_TEMPLATES, SBERT_MODEL_NAME
from src.utils.tools import extract_skills, normalize_skill, is_bad_skill


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
        # 2. 如果非字母排序且为第一作者，加成 0.5 (权重 1.5)
        # 3. 如果是通讯作者，额外加成 0.5
        contribution = 1.0 + (0.5 if is_alpha == 0 and pos_index == 1 else 0) + (0.5 if is_corr == 1 else 0)

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

    def sync_vocab_filtered(self):
        """
        词汇节点同步（带清洗）：仅将通过 tools 校验的 term 同步到图。
        表不动，仅在写入图前过滤，未通过 is_bad_skill(normalize_skill(term)) 的词汇不会入图。
        """
        def _vocab_row_filter(row):
            term = row.get("term") or row.get("name") or ""
            if not term or not str(term).strip():
                return None
            normalized = normalize_skill(str(term).strip())
            if not normalized or is_bad_skill(normalized):
                return None
            return row

        self.state.sync_engine(
            "vocab_sync",
            SQL_QUERIES["GET_ALL_VOCAB"],
            CYPHER_TEMPLATES["MERGE_VOCAB"],
            "id",
            row_processor=_vocab_row_filter,
        )

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
        model = SentenceTransformer(
            self.config['SBERT_MODEL_NAME'],
            cache_folder=sbert_dir,
            trust_remote_code=True,
            device="cpu"
        )

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

            # 语义桥接阈值：工业词与学术词统一规则 ≥0.85 全建，不足 3 条时用 [0.8, 0.85) 补满 3 条（仍不足也完成）
            SIMILAR_HIGH = 0.85   # ≥此分数不设上限
            SIMILAR_FILL = 0.80   # 补边时最低分数，不建 < 0.8 的边
            SIMILAR_TARGET = 3    # 不足时从 [0.8, 0.85) 补到此数

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
                    industry_candidates = []  # 工业词：收集 ≥0.8 的候选，内层循环后按两档规则建边
                    others_candidates = []     # 学术词：同上，仿照工业词规则
                    # 遍历检索出的邻居（按相似度由高到低排列）
                    for score, neighbor_id in zip(scores, labels):
                        neighbor_id = int(neighbor_id)

                        # 过滤无效项（-1 代表没搜到）以及自身
                        if neighbor_id == -1 or neighbor_id == source_id:
                            continue

                        neigh_type = vocab_meta.get(neighbor_id)

                        if source_type == "industry":
                            if neigh_type not in ("concept", "keyword"):
                                continue
                            if score < SIMILAR_FILL:
                                continue
                            industry_candidates.append((float(score), neighbor_id))
                            continue
                        else:
                            # 学术词：只连“不同类型”，避免同类自环
                            if source_type in ("concept", "keyword") and neigh_type in ("concept", "keyword"):
                                continue
                            if neigh_type == source_type:
                                continue
                            if score < SIMILAR_FILL:
                                continue
                            others_candidates.append((float(score), neighbor_id))
                            continue

                    # 工业词：≥0.85 全建，不足 3 条时用 [0.8, 0.85) 按分数从高到低补满 3 条（仍不足也完成）
                    if source_type == "industry" and industry_candidates:
                        industry_candidates.sort(key=lambda x: -x[0])
                        for s, nid in industry_candidates:
                            if s >= SIMILAR_HIGH:
                                current_links.append({"f": source_id, "t": nid, "s": s})
                        if len(current_links) < SIMILAR_TARGET:
                            for s, nid in industry_candidates:
                                if SIMILAR_FILL <= s < SIMILAR_HIGH:
                                    current_links.append({"f": source_id, "t": nid, "s": s})
                                    if len(current_links) >= SIMILAR_TARGET:
                                        break
                    # 学术词：与工业词同一规则（≥0.85 全建，不足 3 条用 [0.8, 0.85) 补满 3 条）
                    elif others_candidates:
                        others_candidates.sort(key=lambda x: -x[0])
                        for s, nid in others_candidates:
                            if s >= SIMILAR_HIGH:
                                current_links.append({"f": source_id, "t": nid, "s": s})
                        if len(current_links) < SIMILAR_TARGET:
                            for s, nid in others_candidates:
                                if SIMILAR_FILL <= s < SIMILAR_HIGH:
                                    current_links.append({"f": source_id, "t": nid, "s": s})
                                    if len(current_links) >= SIMILAR_TARGET:
                                        break

                    # 将本次筛选出的关联加入待推送大池子
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
        【工业级增强版】建立 (Work)-[:HAS_TOPIC]->(Vocabulary) 关联
        升级点：
        1. 引入 Aho-Corasick 自动机：实现 $O(N)$ 复杂度的极速多模式匹配。
        2. 自动化标题回溯：不再依赖原始标签，自动扫描标题补全 HAS_TOPIC 关系。
        3. 单词边界保护：手动实现 is_word_boundary 校验，确信匹配的是独立单词。
        """
        import ahocorasick
        from config import SQL_QUERIES, CYPHER_TEMPLATES

        # --- 1. 构建 Aho-Corasick 自动机 (模式树) ---
        logging.info("Initializing Aho-Corasick Automaton for title scanning...")
        A = ahocorasick.Automaton()
        with sqlite3.connect(self.config['DB_PATH']) as conn:
            # 加载词库中所有已有的 term
            all_vocab = [r[0] for r in conn.execute("SELECT term FROM vocabulary").fetchall() if r[0]]

        for term in all_vocab:
            # 将 term 作为 Key，其本身作为 Value 存入自动机
            term_lower = term.lower()
            A.add_word(term_lower, term_lower)
        A.make_automaton()

        # --- 2. 定义单词边界校验逻辑 (模拟正则 \b) ---
        def is_word_match(text, start_pos, end_pos):
            """
            校验匹配到的片段是否为独立单词。
            逻辑：检查片段前后字符是否为字母数字。
            """
            # 检查左边界：如果左侧有字符且是字母数字，则不是独立单词
            if start_pos > 0 and text[start_pos - 1].isalnum():
                return False
            # 检查右边界：如果右侧有字符且是字母数字，则不是独立单词
            if end_pos < len(text) and text[end_pos].isalnum():
                return False
            return True

        # --- 3. 扫描 Works 表并执行打标 ---
        # 使用你新增加的 GET_WORK_METADATA_FOR_TAGGING SQL
        sql = SQL_QUERIES.get("GET_WORK_METADATA_FOR_TAGGING",
                              "SELECT work_id as id, title, concepts_text, keywords_text FROM works")

        with sqlite3.connect(self.config['DB_PATH']) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)
            batch = []

            for row in tqdm(cursor, total=550000, desc="Semantic Indexing (AC-Auto)"):
                title_clean = (row['title'] or "").lower()
                terms = set()

                # 语义来源 A：原始元数据提取 (Concepts & Keywords)
                raw_meta = f"{row['concepts_text'] or ''}|{row['keywords_text'] or ''}"
                meta_terms = [t.strip().lower() for t in re.split(r'[|;,]', raw_meta) if t.strip()]
                terms.update(meta_terms)

                # 语义来源 B：自动化标题扫描 (AC 自动机)
                # 一次扫描标题，找出所有匹配的词库术语
                for end_index, original_term in A.iter(title_clean):
                    # A.iter 返回的是结束位置，计算起始位置
                    start_index = end_index - len(original_term) + 1

                    # 关键：执行单词边界验证，排除子字符串干扰
                    if is_word_match(title_clean, start_index, end_index + 1):
                        terms.add(original_term)

                # 4. 组装批次并推送到 Neo4j
                for t in terms:
                    batch.append({"id": row['id'], "term": t})
                    if len(batch) >= 2000:
                        self.state.engine.send_batch(CYPHER_TEMPLATES["LINK_WORK_VOCAB"], batch)
                        batch = []

            if batch:
                self.state.engine.send_batch(CYPHER_TEMPLATES["LINK_WORK_VOCAB"], batch)

        logging.info("基于标题的标记与语义链接已完成。")

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
                skills = set(extract_skills(raw_skills))

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

    def build_cooccurrence_links(self):
        """
        【新增核心】计算词汇共现权重：(Vocab)-[:CO_OCCURRED_WITH]-(Vocab)
        逻辑：利用 SQLite 预处理 55 万篇论文的词汇对，统计频率后同步至 Neo4j。
        """
        db_path = self.config['DB_PATH']
        from config import SQL_QUERIES, CYPHER_TEMPLATES

        with sqlite3.connect(db_path) as conn:
            # 1. 构建临时表以优化聚合性能 (针对 55 万量级数据)
            logging.info("[*] 正在准备 SQLite 临时映射表以计算共现频率...")
            conn.executescript("""
                               DROP TABLE IF EXISTS work_terms_temp;
                               CREATE TABLE work_terms_temp
                               (
                                   work_id TEXT,
                                   term    TEXT
                               );
                               CREATE INDEX idx_wt_id ON work_terms_temp (work_id);
                               CREATE INDEX idx_wt_term ON work_terms_temp (term);
                               """)

            # 2. 填充数据：从 works 的 concepts_text/keywords_text 按分隔符拆成 (work_id, term) 写入临时表
            # concepts_text 实际存储为竖线分隔 "a|b|c"（见 processor.py），非 JSON，故用 Python 拆分
            logging.info("[*] 正在平铺 Work-Term 数据...")
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT work_id, concepts_text, keywords_text FROM works WHERE concepts_text IS NOT NULL OR keywords_text IS NOT NULL"
            )
            batch = []
            for row in cursor:
                raw_meta = f"{row['concepts_text'] or ''}|{row['keywords_text'] or ''}"
                terms = [t.strip().lower() for t in re.split(r'[|;,]', raw_meta) if t.strip()]
                for term in terms:
                    batch.append((row['work_id'], term))
                if len(batch) >= 50000:
                    conn.executemany("INSERT INTO work_terms_temp (work_id, term) VALUES (?, ?)", batch)
                    batch = []
            if batch:
                conn.executemany("INSERT INTO work_terms_temp (work_id, term) VALUES (?, ?)", batch)
            conn.commit()

            # 3. 执行共现聚合查询
            logging.info("[*] 正在计算共现频次 (Co-occurrence Matrix)...")
            cursor = conn.execute(SQL_QUERIES["GET_VOCAB_CO_OCCURRENCE"])

            batch = []
            for row in tqdm(cursor, desc="Syncing Co-occurrence Weights"):
                batch.append({
                    "term_a": row['term_a'],
                    "term_b": row['term_b'],
                    "freq": row['freq']
                })
                if len(batch) >= 1000:
                    self.state.engine.send_batch(CYPHER_TEMPLATES["MERGE_CO_OCCURRENCE"], batch)
                    batch = []

            if batch:
                self.state.engine.send_batch(CYPHER_TEMPLATES["MERGE_CO_OCCURRENCE"], batch)

        logging.info("共现网络拓扑构建完成。")