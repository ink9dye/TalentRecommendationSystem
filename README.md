## 项目简介

`TalentRecommendationSystem` 是一个面向 **高端科技人才智能推荐** 的完整工程项目，采用「**多路召回（Vector / Label / Collaboration）+ 深度精排（KGAT‑AX）+ 知识图谱解释**」的两阶段架构。  
项目从 OpenAlex 学术数据与 BOSS 直聘岗位数据出发，构建本地 SQLite 数据库与 Neo4j 知识图谱，离线训练 KGAT‑AX 排序模型，线上通过 Streamlit / Vue 前端提供交互式专家推荐与可视化解释能力。

---

## 目录总览

- **[目录树](#目录树)**
- **[整体架构与数据流](#整体架构与数据流)**
- **[知识图谱构建详解](#知识图谱构建详解)**
- **[三路召回与文本转向量详解](#三路召回与文本转向量详解)**
- **[索引构建详解](#索引构建详解)**
- **[KGAT-AX 模型详解](#kgat-ax-模型详解)**
- **[精排与解释详解](#精排与解释详解)**
- **[代码文件详细说明](#代码文件详细说明)**
  - [根目录](#根目录)
  - [核心推荐引擎 `src/core`](#核心推荐引擎-srccore)
  - [召回模块 `src/core/recall`](#召回模块-srccorerecall)
  - [精排与解释模块 `src/core/ranking`](#精排与解释模块-srccoreranking)
  - [接口层 `src/interface`](#接口层-srcinterface)
  - [通用工具 `src/utils`](#通用工具-srcutils)
  - [基础设施：数据库与图谱 `src/infrastructure/database`](#基础设施数据库与图谱-srcinfrastructuredatabase)
  - [基础设施：数据抓取与合并 `src/infrastructure/crawler`](#基础设施数据抓取与合并-srcinfrastructurecrawler)
  - [基础设施：SBERT 模型工具 `src/infrastructure/database/models/sbert_model`](#基础设施sbert-模型工具-srcinfrastructuredatabasemodelssbert_model)
  - [辅助数据脚本与配置](#辅助数据脚本与配置)
- **[技术栈综述](#技术栈综述)**
- **[环境准备与安装](#环境准备与安装)**
- **[运行指南](#运行指南)**
  - [1. 快速体验（已有数据与索引）](#1-快速体验已有数据与索引)
  - [2. 从零构建完整数据与模型管线](#2-从零构建完整数据与模型管线)
- **[开发与扩展建议](#开发与扩展建议)**

---

## 目录树

> 仅保留与业务 / 算法相关的核心文件，IDE 配置、缓存与大模型权重目录已省略。

```bash
TalentRecommendationSystem-master/
├── app.py                          # 简化版 Streamlit Demo（使用 mock 数据）
├── main.py                         # 启动入口：以 CLI 方式拉起 Streamlit 界面
├── config.py                       # 全局配置：SQLite / Neo4j / 索引路径 / 域映射
├── index.html                      # Vue 前端入口 HTML
├── vite.config.ts                  # Vite + Vue3 + Element Plus 构建配置
├── tsconfig*.json                  # TypeScript 编译配置
├── test.js                         # Node 环境性能测试脚本（非业务逻辑）
├── README.md                       # 原始英文 README（较老的结构说明）
├── .gitignore                      # Git 忽略规则（数据 / 模型 / IDE 等）
├── data/
│   └── attribute.py                # 示意性数据脚本（可扩展领域属性）
└── src/
    ├── core/
    │   ├── total_core.py           # 「召回 + 精排」核心编排入口
    │   ├── recall/
    │   │   ├── input_to_vector.py  # OpenVINO 加速的 SBERT 文本编码器
    │   │   ├── total_recall.py     # 多路召回总控（向量 / 标签 / 协同）
    │   │   ├── vector_path.py      # 向量路：Faiss + 向量索引召回作者
    │   │   ├── label_path.py       # 标签路：基于 Job-Skill 与 Vocabulary 的图谱召回
    │   │   └── collaboration_path.py # 协同路：基于本地协作索引的协同召回
    │   └── ranking/
    │       ├── ranking_engine.py   # 精排引擎：融合召回与 KGAT 打分
    │       ├── rank_scorer.py      # KGAT 子空间打分逻辑
    │       └── rank_explainer.py   # 推荐解释生成（Neo4j + 注意力权重）
    ├── interface/
    │   └── app.py                  # 主 Streamlit 前端（JD 输入 / 结果展示）
    ├── utils/
    │   └── domain_utils.py         # 领域 ID 解析 / 正则构造 / 交集判定工具
    ├── infrastructure/
    │   ├── mock_data.py            # Streamlit Demo 用到的模拟人才与图谱数据
    │   ├── database/
    │   │   ├── build_kg/
    │   │   │   ├── builder.py      # KG 构建器：节点同步 / 语义打标 / 共现网络
    │   │   │   ├── kg_utils.py     # Neo4j & SQLite 管理封装（GraphEngine 等）
    │   │   │   └── generate_kg.py  # 一键运行完整 KG 构建流水线
    │   │   ├── build_index/
    │   │   │   ├── build_vector_index.py      # 构建词汇 / 摘要 / Job 向量索引
    │   │   │   ├── build_collaborative_index.py # 基于 authorships 构建协作索引
    │   │   │   ├── build_vocab_stats_index.py  # 词汇统计索引（领域分布 + 共现）
    │   │   │   ├── build_feature_index.py       # 作者 / 机构特征 JSON 索引
    │   │   │   └── download_model.py            # 下载并缓存 SBERT 模型权重
    │   │   ├── kgat_ax/
    │   │   │   ├── model.py           # KGAT‑AX 主模型（图注意力 + AX 学术特征）
    │   │   │   ├── data_loader.py     # 加权子图采样 & 拉普拉斯矩阵构建
    │   │   │   ├── trainer.py         # 训练主循环（CF + KG 两阶段）
    │   │   │   ├── generate_training_data.py # 调用召回系统生成分层训练样本
    │   │   │   ├── build_kg_index.py  # 将 kg_final.txt 导入 SQLite 并建索引
    │   │   │   ├── verify_recall.py   # 训练后离线验证与策略对比
    │   │   │   ├── pipeline.py        # 一键执行「生成数据 → 建索引 → 训练」流水线
    │   │   │   ├── kgat_utils/        # 评估 / 日志 / 早停等工具
    │   │   │   └── kgat_parser/       # KGAT / CKE / BPRMF 参数解析
    │   │   ├── models/
    │   │   │   └── sbert_model/
    │   │   │       ├── README.md      # HuggingFace SBERT 模型说明（英文）
    │   │   │       ├── download_sbert.py  # 将 SBERT 模型下载到本地路径
    │   │   │       └── test_similarity.py # 本地 SBERT 编码与相似度测试
    │   └── crawler/
    │       ├── new_getdata_likelogin0104.py  # DrissionPage 抓取 BOSS 直聘岗位
    │       ├── merge_database.py             # 将 jobs.csv 合并进统一 SQLite
    │       └── use_openalex/
    │           ├── main.py           # OpenAlex 抓取 CLI（分阶段任务入口）
    │           ├── crawler_logic.py  # 各阶段抓取逻辑：works / authors / stats
    │           ├── api_client.py     # 带重试与配额检测的 OpenAlex API 客户端
    │           ├── database.py       # 学术数据库管理（统一索引与统计）
    │           ├── models.py         # Work / AuthorProfile 数据类定义
    │           ├── processor.py      # 单篇论文解析与写入 SQLite
    │           ├── utils.py          # ID 生成 / 字段清洗等通用工具
    │           └── db_config.py      # OpenAlex 抓取配置（EMAIL / FIELDS / 路径）
    └── infrastructure/
        └── crawler/
            └── use_openalex/
                └── some_tool/        # 若干一键修复 / 维护脚本（补 DOI、刷新指标等）
                    ├── delete_doi.py
                    ├── migrate_ids.py
                    ├── fix_doi_to_openalex_id.py
                    ├── refresh_metrics.py
                    ├── replace.py
                    ├── single_recrawl_not_found.py
                    └── use_name_to_fix_workid.py
```

---

## 整体架构与数据流

从数据到推荐结果的完整链路如下：

1. **岗位数据采集（Industry Side）**
   - 使用 `src/infrastructure/crawler/new_getdata_likelogin0104.py` 基于 DrissionPage 自动化登录 BOSS 直聘，按城市和岗位关键词抓取高学历（硕士 / 博士）岗位，生成 `jobs.csv`。
   - `src/infrastructure/crawler/use_openalex/merge_database.py` 中的 `UnifiedTalentDB` 将 `jobs.csv` 合并进 SQLite 数据库，写入 `jobs` 表，并将岗位技能同步到 `vocabulary` 表（类型标记为 `industry`）。

2. **学术数据采集 & 画像构建（Academic Side）**
   - `src/infrastructure/crawler/use_openalex/main.py` 通过 `DatabaseManager` 初始化数据库结构（`works / authors / institutions / sources / authorships / vocabulary` 等表）。
   - `crawler_logic.py` + `api_client.py` 分阶段从 OpenAlex 抓取论文、作者、机构与摘要信息：
     - Phase 1：按领域概念 ID 抓取最新论文；
     - Phase 2：抓取高被引论文并收集领域精英作者；
     - Phase 3：补全精英作者画像和代表作；
     - Phase 4：批量同步作者 H‑index / 总引用 / 论文数；
     - Phase 5：更新机构与渠道统计。
   - `processor.py` 负责将单篇 `work` 解析写入 SQLite，并根据概念名称自动映射到 17 个业务领域，填充 `works.domain_ids`。

3. **知识图谱构建与语义增强（Neo4j + SBERT）**
   - `src/infrastructure/database/build_kg/generate_kg.py` 调用 `run_pipeline(CONFIG_DICT)`：
     - 通过 `builder.KGBuilder` & `kg_utils.GraphEngine` 将 SQLite 数据增量同步至 Neo4j，建立节点与关系：
       - 节点：`Author / Work / Vocabulary / Institution / Source / Job`
       - 关系：`AUTHORED / PRODUCED_BY / PUBLISHED_IN / HAS_TOPIC / REQUIRE_SKILL / SIMILAR_TO / CO_OCCURRED_WITH`
     - 使用 `build_work_semantic_links()` + Aho‑Corasick 自动机扫描论文标题与关键词，补齐 `(Work)-[:HAS_TOPIC]->(Vocabulary)`。
     - `build_cooccurrence_links()` 利用 SQLite 计算词汇在论文中的共现频次，写入 `CO_OCCURRED_WITH`。
     - `build_semantic_bridge()` 利用 SBERT 计算跨类型词汇相似度（如岗位技能 ↔ 学术词汇），写入 `SIMILAR_TO`。
   - `build_index/` 下的一系列脚本继续构建：
     - 词汇 / 论文 / 岗位的 Faiss 向量索引；
     - 作者和机构的结构化特征索引；
     - 词汇跨领域统计索引；
     - 作者协作关系索引。

4. **KGAT‑AX 训练数据与图索引构建**
   - `kgat_ax/generate_training_data.py` 使用 `TotalRecallSystem` 在线召回，构造「岗位 → 专家」的四级梯度训练样本（Pos / Fair / Neutral / EasyNeg），并将全图导出为加权三元组 `kg_final.txt`。
   - `kgat_ax/build_kg_index.py` 将 `kg_final.txt` 导入 SQLite，生成高性能的 `kg_triplets` 覆盖索引。

5. **图神经网络训练（KGAT‑AX）**
   - `kgat_ax/trainer.py` 使用 `DataLoaderKGAT` 加载训练数据与加权图，训练 `model.KGAT`：
     - CF 阶段学习岗位与专家的排序规律；
     - KG 阶段学习多关系图中的语义结构；
     - 通过 AX 学术特征（H‑index / 总引用 / 论文数）在向量空间中做小幅增益。
   - 最终权重以 `best_model_epoch_*.pth` 形式保存在 `data/kgatax_train_data/weights` 下。

6. **线上召回与精排**
   - `core/recall/` 模块加载 Faiss 索引和 Neo4j 图谱，实现：
     - 向量路：岗位 JD → SBERT 向量 → 论文 / 岗位相似度召回作者；
     - 标签路：Job-Skill → Vocabulary → Work → Author 的图谱级召回；
     - 协同路：基于本地协作索引挖掘高频合作者。
   - `core/total_core.py` 加载训练好的 KGAT 模型，调用 `TotalRecallSystem` 完成召回，之后交给 `RankingEngine` 进行召回排序 + KGAT 分数归一化与融合，并生成解释信息。

7. **前端展示与交互**
   - `src/interface/app.py`：基于 Streamlit 提供 Web 表单，支持：
     - JD 文本输入；
     - 多选业务领域过滤；
     - Top‑N 专家列表展示（分数拆解、代表作、解释文案等）。
   - `index.html + vite.config.ts + tsconfig*.json`：为 Vue3 + Element Plus 前端预留的 SPA 壳，后续可将推荐 API 集成到独立的 Vue 控制台中。

---

## 知识图谱构建详解

本节从**构建顺序、节点/边属性、边权重、构建逻辑**等角度说明 Neo4j 知识图谱的构建方式，对应代码：`config.py`、`src/infrastructure/database/build_kg/builder.py`、`generate_kg.py`、`kg_utils.py`。

### 1. 整体构建顺序（generate_kg.py）

流水线在 `run_pipeline(config)` 中按固定顺序执行：

| 步骤 | 内容 |
|------|------|
| **Step 0** | Neo4j 约束与索引、SQLite 索引（地基） |
| **Step 1** | 六类节点同步：Vocabulary → Author → Work → Institution → Source → Job |
| **Step 2** | 作者–论文–机构–渠道拓扑（含 **AUTHORED 边权重**） |
| **Step 3** | 语义打标 → `(Work)-[:HAS_TOPIC]->(Vocabulary)`、`(Job)-[:REQUIRE_SKILL]->(Vocabulary)` |
| **Step 4** | 词汇共现 → `(Vocabulary)-[:CO_OCCURRED_WITH]-(Vocabulary)`，边属性 **weight** |
| **Step 5** | 语义桥接 → `(Vocabulary)-[:SIMILAR_TO]->(Vocabulary)`，边属性 **score** |

数据从 SQLite（`config['DB_PATH']`）读出，经 `SyncStateManager` 做增量（marker），由 `GraphEngine.send_batch` 批量写入 Neo4j。

### 2. 节点类型与点属性

所有节点由 `config.py` 的 `SQL_QUERIES` 从 SQLite 查询，再通过 `CYPHER_TEMPLATES` 中的 `MERGE_*` 写入 Neo4j。

| 节点标签 | 主键 | 点属性 | 来源 |
|----------|------|--------|------|
| **Author** | `id` (author_id) | `name`, `h_index`, `works_count`, `citations` | authors |
| **Work** | `id` (work_id) | `title`, `name`, `year`, `citations`, `domain_ids` | works |
| **Vocabulary** | `id` (voc_id) | `term`(小写), `type`(即 entity_type) | vocabulary |
| **Institution** | `id` (inst_id) | `name`, `works_count`, `citations` | institutions |
| **Source** | `id` (source_id) | `name`, `type`, `works_count`, `citations` | sources |
| **Job** | `id` (securityId) | `name`, `skills`, `description`, `domain_ids` | jobs |

- **Author**：学术画像（h_index、论文数、被引数），用于精排与展示。  
- **Work**：论文实体；`domain_ids` 为业务领域 ID（如 `"1|4"`），用于领域过滤与共现统计。  
- **Vocabulary**：统一词表（概念、关键词、岗位技能等），`type` 区分 concept / keyword / industry 等，语义桥接时只做**跨类型**连接。  
- **Job**：岗位，与 Vocabulary 通过 REQUIRE_SKILL 相连，供标签路召回。

Neo4j 中还会建立索引：`Work.domain_ids`、`Job.domain_ids`、`Vocabulary.term`、`Author.h_index`，以及 `CO_OCCURRED_WITH.weight`。

### 3. 边类型、边属性与边权重

#### 3.1 AUTHORED（Author → Work）

- **含义**：作者署名某篇论文。  
- **边属性**：`pos_index`（作者顺序）、`pub_year`（发表年份）、`is_corresponding`（是否通讯作者）、**`pos_weight`**（边权重）。

**权重公式**（`builder.py` 中 `WeightStrategy.calculate`）：

- 时间衰减：`time_weight = exp(-0.1 * (current_year - pub_year))`（约每 10 年衰减到 1/e）。  
- 基础贡献：`contribution = 1.0 + 0.2`（若非字母序且第一作者）`+ 0.2`（若通讯作者）。  
- **pos_weight = contribution × time_weight**，保留 4 位小数。

即：第一作者/通讯作者、近期论文权重大；字母序共同一作不加成。

#### 3.2 PRODUCED_BY（Work → Institution）、PUBLISHED_IN（Work → Source）

- 无额外边属性，仅表示「该 Work 由该 Institution 产出」「该 Work 发表于该 Source」。  
- 与 AUTHORED 在同一条 Cypher `LINK_AUTHORED_COMPLEX` 中一并创建（根据 `row.iid`/`row.sid` 是否存在）。

#### 3.3 HAS_TOPIC（Work → Vocabulary）

- **含义**：论文涉及某术语/概念/关键词。  
- **边属性**：无。

来源有两部分（`build_work_semantic_links`）：  
- **元数据**：`works.concepts_text`、`keywords_text` 按 `|;,\)` 拆成 term，与 vocabulary 的 term 匹配（小写）。  
- **标题扫描**：用 Aho-Corasick 自动机在 `title` 上对词库做多模式匹配，并做**单词边界**校验（避免 "machine" 命中 "in"），再对匹配到的 term 建 `(Work)-[:HAS_TOPIC]->(Vocabulary)`。

#### 3.4 REQUIRE_SKILL（Job → Vocabulary）

- **含义**：岗位需要某技能/词。  
- **边属性**：无。  
- 数据来自 `jobs.skills`，按 `,，;；/` 拆成多个 skill，转小写后与 `Vocabulary.term` 匹配建边；增量以 `crawl_time` 为 marker。

#### 3.5 CO_OCCURRED_WITH（Vocabulary – Vocabulary，无向）

- **含义**：两词在多篇论文中一起出现，共现次数越多关系越强。  
- **边属性**：**`weight`**（整数），表示**共现频次**（多少篇 Work 同时包含这两个 term）。

**计算方式**（`build_cooccurrence_links`）：  
- 在 SQLite 建临时表 `work_terms_temp(work_id, term)`。  
- 当前实现用 `json_each(concepts_text)` 将 works 的 concepts 展开为 (work_id, term) 再插入（依赖 `concepts_text` 为合法 JSON 数组；若为 `|` 分隔需先 ETL 或改 SQL）。  
- 执行 `GET_VOCAB_CO_OCCURRENCE`：`a.work_id = b.work_id AND a.term < b.term` 自连接，按 (term_a, term_b) 聚合 `COUNT(*)` 为 `freq`，且 `HAVING freq > 1`。  
- 在 Neo4j 中 `MERGE (v1)-[r:CO_OCCURRED_WITH]-(v2) SET r.weight = row.freq`。

即：**边权重 = 同时包含这两词的论文数**，且至少为 2 才建边。

#### 3.6 SIMILAR_TO（Vocabulary → Vocabulary，有向）

- **含义**：两词在向量空间中语义相近，且为**跨类型**连接（如技能↔岗位描述词）。  
- **边属性**：**`score`**（浮点），即 SBERT 归一化后的**相似度**（点积 ≈ 余弦相似度）。

**计算方式**（`build_semantic_bridge`）：  
- 用 Faiss 词汇索引 + SentenceTransformer 对 vocabulary 的 `term` 向量化（L2 归一化）。  
- 对每个源词在 Faiss 中取 Top-100 最近邻。  
- **只保留与源词 `entity_type` 不同的邻居**（如 concept 连到 skill/industry），每个源词最多保留 3 条边，取相似度最高的 3 个邻居。  
- 写入 Neo4j：`MERGE (v1)-[r:SIMILAR_TO]->(v2) SET r.score = row.s`。

**边属性与权重小结**：

| 边类型 | 方向 | 边属性/权重 | 含义 |
|--------|------|--------------|------|
| AUTHORED | Author→Work | pos_index, pub_year, is_corresponding, **pos_weight** | 署名顺序、是否通讯、**时间衰减+署名加成** |
| PRODUCED_BY | Work→Institution | 无 | 机构产出论文 |
| PUBLISHED_IN | Work→Source | 无 | 发表渠道 |
| HAS_TOPIC | Work→Vocabulary | 无 | 论文–主题/关键词 |
| REQUIRE_SKILL | Job→Vocabulary | 无 | 岗位–技能 |
| CO_OCCURRED_WITH | Vocabulary–Vocabulary | **weight** (int) | **共现论文数** |
| SIMILAR_TO | Vocabulary→Vocabulary | **score** (float) | **SBERT 语义相似度（仅跨类型）** |

### 4. 构建逻辑要点

- **Step 0**：Neo4j 执行 `INIT_SCHEMA`（各节点 `id` 唯一约束 + 上述索引）；SQLite 执行 `SQL_INIT_SCRIPTS`，保证按 `last_updated`/`year`/`crawl_time`/`ship_id` 等增量查表高效。  
- **Step 1 节点同步**：对每类实体调用 `sync_engine(task_name, sql, cypher, time_field)`，从 SQLite 按 `time_field` 增量拉取（如 `last_updated > marker`），按 `BATCH_SIZE` 批处理，每批用对应 `MERGE_*` 写入 Neo4j，并在 `sync_metadata` 表更新 marker，实现断点续跑。  
- **Step 2 拓扑**：从 `authorships` 联表 `works` 取每条署名的 author_id、work_id、inst_id、source_id、pos_index、is_corresponding、is_alphabetical、year；对每行先用 `WeightStrategy.calculate` 算 `pos_w`，再调用 `LINK_AUTHORED_COMPLEX` 一次创建 AUTHORED（含 pos_weight）及可选的 PRODUCED_BY、PUBLISHED_IN。  
- **Step 3 语义打标**：先 `build_work_semantic_links()`（AC 自动机 + concepts/keywords），再 `build_job_skill_links()`；这样 **HAS_TOPIC 先于 CO_OCCURRED_WITH**，共现统计依赖的「论文–词」关系已存在（若共现改用 work_terms_temp，需与 HAS_TOPIC 的数据源一致，见下）。  
- **Step 4 共现**：依赖 Step 3 的 HAS_TOPIC（或与 HAS_TOPIC 同源的 work–term 表）；当前代码用 `json_each(concepts_text)` 填 `work_terms_temp`，若你的 `concepts_text` 是 `"a|b|c"` 这种格式，需在别处先转成 JSON 再灌，或改 SQL 为按 `|` 拆分的逻辑，否则共现会为空或报错。  
- **Step 5 语义桥接**：只加「跨类型」的 SIMILAR_TO，避免同类型词之间重复连接；增量由 `semantic_bridge_sync` 的 marker（voc_id）控制，只处理 `voc_id > marker` 的词。

### 5. 实现细节与注意点

1. **共现数据源**：`build_cooccurrence_links` 里用 `json_each(concepts_text)` 填充 `work_terms_temp`，要求 `works.concepts_text` 为 JSON 数组。若实际是 `"a|b|c"` 这种格式，需要先在一处统一成 (work_id, term) 再写入临时表（或改 SQL），否则 Step 4 会没有数据或报错。  
2. **HAS_TOPIC 与 Vocabulary 同步**：HAS_TOPIC 和 REQUIRE_SKILL 都通过 `Vocabulary.term` 匹配，因此必须先有 Step 1 的 vocab_sync；且 vocabulary 表里要有从 works（concepts/keywords）和 jobs（skills）来的 term。  
3. **语义桥接的 type**：SIMILAR_TO 只连 `entity_type` 不同的词对，所以 vocabulary 的 `entity_type`（如 concept / keyword / industry）必须正确填写，否则可能几乎没有桥接边。  
4. **增量与顺序**：每次运行 pipeline 会重置 `semantic_bridge_sync` 的 marker，但其他任务（如 topology、job_skill）用各自 marker 增量；若中途改过 builder 逻辑或 config，需要视情况清空 Neo4j 或重置对应 marker 再跑。

---

## 三路召回与文本转向量详解

本节从**核心目的**与**为达成目的所使用的方法**两方面，说明「文本转向量」以及「向量路 / 标签路 / 协同路」三路召回的详细过程。对应代码：`src/core/recall/input_to_vector.py`、`vector_path.py`、`label_path.py`、`collaboration_path.py`、`total_recall.py`。

### 1. 文本转向量（input_to_vector.py — QueryEncoder）

**核心目的**  
将岗位描述（JD）转成与离线索引**同一向量空间**的固定维向量，供向量路在摘要/Job 索引上做 Faiss 检索，以及标签路做「JD 与学术词」的语义守门（余弦相似度）。同时通过**动态自共振**，让 JD 里与业务强相关的词在向量里占更大权重，提升检索相关性。

**使用的方法**

1. **动态词库 `_build_dynamic_lexicon()`**  
   从 SQLite 的 `jobs.skills` 与 `vocabulary.term` 统计词频；只保留「在 vocabulary 里存在、且出现次数在 [3, total_docs×0.03]」的词（既非生僻也非“沟通、办公”这类超高频泛词），得到集合 `hardcore_lexicon`。

2. **动态自共振增强 `_apply_dynamic_resonance(text)`**  
   对输入 JD 分词，找出落在 `hardcore_lexicon` 里的词；对每个命中词重复三次拼成一段字符串再拼到原文后面（如：`原文 ... 词A 词A 词A 词B 词B 词B`），使这些词在 attention 里占更大权重，从而拉高与「含这些技能的论文/岗位」的向量相似度。

3. **编码与池化**  
   用 AutoTokenizer（与 config 中 SBERT 模型一致、本地）对增强后的文本 tokenize（max_length=1024）；用 OpenVINO 的 OVModelForFeatureExtraction（export=False，直接读已导出的模型）前向得到 last_hidden_state；**Mean pooling**（按 attention_mask 对 token 维度加权平均，与离线建索引时一致）；**L2 归一化**（与 Faiss 内积检索等价余弦一致）；输出 `(vector, duration)`，vector 为 float32 的 1×d 数组。

4. **与离线一致性**  
   池化方式、归一化、模型与离线建摘要/词汇/Job 索引时一致，保证同一语义空间，避免「线上向量和索引向量不匹配」导致检索偏。

---

### 2. 向量路召回（vector_path.py — VectorPath）

**核心目的**  
用**语义相似度**找「与当前 JD 最像的论文」，再把这些论文的**作者**当作候选人；即「JD ≈ 论文内容 → 写这些论文的人」。强调论文级语义 + 领域约束，避免跨领域论文被误召回。

**使用的方法**

1. **Faiss 检索相似论文**  
   使用论文摘要的 Faiss 索引（ABSTRACT_INDEX_PATH）与 id_map；用 query_vector 在索引上 search(..., search_k=500)，得到 500 个最相似的 work_id，顺序按相似度从高到低。

2. **领域硬过滤**  
   从 SQLite 的 works 表查出这批 work_id 的 domain_ids；若调用方传了 target_domains（如 "1|4"），用 DomainProcessor.to_set / has_intersect 判断论文是否属于目标领域之一；只保留有交集的论文。

3. **论文 → 作者并保持顺序**  
   对过滤后的 work_id 列表，在 authorships 里查每个 work 的所有 author_id；用 `ORDER BY MIN(instr(ordered_work_str, work_id))`：作者在「排序后的论文列表里第一次出现的位置」越靠前，排名越前；即作者排名由「其最相关的那篇论文在语义排序中的位置」决定。

4. **去重与截断**  
   同一作者可能在多篇论文中出现，只保留第一次出现的位置（GROUP BY author_id + 上述 ORDER BY）；返回前 recall_limit（如 500）个 author_id 及耗时。

---

### 3. 标签路召回（label_path.py — LabelRecallPath）

**核心目的**  
走「**岗位技能 → 学术词 → 论文 → 作者**」的图谱路径，用知识图谱 + 多维度打分，找出与 JD 技能要求在语义和共现上都对齐的学术词，再只保留领域垂直、论文质量高的论文与作者；强调硬技能/概念对齐，并抑制泛词、万金油论文和弱相关论文。

**使用的方法（按阶段）**

1. **领域探测 `_detect_domain_context(query_vector)`**  
   用 Job 的 Faiss 索引对 query_vector 做 Top-10 检索，得到最相似的 10 个岗位 ID；在 Neo4j 里查这些 Job 的 domain_ids，统计各领域出现次数；取 Top-3 领域作为 inferred_domains，并算主导领域占比 dominance，供后续领域纯度和权重用。

2. **锚点技能提取 `_extract_anchor_skills(target_job_ids)`**  
   在 Neo4j 中对上述 Top 岗位（如前 5 个）做 (Job)-[:REQUIRE_SKILL]->(Vocabulary)，统计每个词被多少 Job 引用，算 cov_j = count/total_job；**1% 熔断**：只保留 cov_j < 0.01 的技能词，过滤“沟通、办公”等泛词；按 cov_j 升序取前 50 个词作为工业侧锚点（高含金量、偏稀缺技能）。

3. **语义扩展 `_expand_semantic_map()`**  
   - **图扩展** `_query_expansion_with_topology`：从锚点词出发沿 (Vocabulary)-[:SIMILAR_TO]-(Vocabulary) 找邻居学术词；对每个邻居统计「被多少个不同锚点命中」hit_count、在 Job 侧的覆盖率 cov_j；在 SQLite 的 vocabulary_domain_stats 里查 work_count、domain_span、domain_dist，只保留在目标领域里有产出的词。  
   - **学术共鸣** `_calculate_academic_resonance`：对当前候选词集合在 Neo4j 里查 CO_OCCURRED_WITH，对每个词汇总共现边的 weight 之和作为 resonance_score（与其它候选词在论文里经常一起出现的词加分，即「单词协作」）。  
   - **共现领域指标** `_get_cooccurrence_domain_metrics`：从 vocab_stats.db 的 vocabulary_cooccurrence 与 vocabulary_domain_stats（及主库 vocabulary）为每个候选词算 **cooc_span**（共现伙伴的平均领域跨度）与 **cooc_purity**（共现伙伴在目标领域的论文占比）。与各种领域的词都共现 → 万金油 → 用 cooc_span 降权；只跟特定领域的词共现 → 专精 → 用 cooc_purity 加权。  
   - **词权重** `_apply_word_quality_penalty`：IDF、岗位惩罚（cov_j 过高压分）、领域纯度与跨度、共鸣熔断/加成、收敛奖励（多锚点命中 + 共现强）；**SBERT 语义守门员**：semantic_factor = max(0, cos_sim)^6；**共现领域**：span_penalty = 1/(1+log1p(cooc_span))（万金油降权）、purity_bonus = 1+log1p(cooc_purity)（领域专精加权）；最终公式：Weight = (IDF/job_penalty) × purity² × (convergence_bonus/span) × semantic_factor × span_penalty × purity_bonus。

4. **图谱反查论文与作者**  
   Cypher 从带权重的 Vocabulary 集合出发沿 (Vocabulary)<-[:HAS_TOPIC]-(Work) 找论文（可选 w.domain_ids =~ $regex）；再 (Work)<-[auth_r:AUTHORED]-(Author)；对论文做 1% 熔断（degree_w/total_w < 0.01）；为每篇论文收集命中的 (vid, idf_weight)、auth_r.pos_weight、title、year、domains 等，用于下一步作者级打分。

5. **论文贡献度 `_compute_contribution(paper, context)`**  
   撤稿拦截；**领域纯度**（论文领域与目标无交集则 0，有交集则 (目标领域数/论文涉及领域数)^4）；标签得分累加（score_map[vid] × hit.idf）；综述降权（survey/overview/review × 0.1，多标签 1/n²）；**紧密度**：命中词在向量空间两两余弦相似度均值，(1+proximity)^n 加成；时序与署名：time_decay = decay_rate^year_diff，再乘 auth_r.pos_weight；单篇论文得分 = 上述各项相乘；作者得分 = 其所有论文得分之和，按作者总分排序取前 recall_limit 个 author_id。

---

### 4. 协同路召回（collaboration_path.py — CollaborativeRecallPath）

**核心目的**  
不依赖 JD 语义，而是给定一批「种子作者」（通常为向量路 + 标签路 Top 100 的并集），从预建好的作者协作表里找出与这些种子**合作最紧密**的作者，作为补充候选人；利用「谁和谁经常一起发论文」扩展候选池，发现同方向、同圈子的学者。

**使用的方法**

1. **种子来源**  
   在 total_recall 里由调用方传入：seeds = list(set(v_list[:100] + l_list[:100]))。

2. **协作表查询**  
   使用 build_collaborative_index 生成的 SQLite 表 scholar_collaboration(aid1, aid2, score)；对种子分批（如 100 个一批），用「WHERE aid1 IN (...) 取 aid2, score」与「WHERE aid2 IN (...) 取 aid1, score」两段 SQL，UNION ALL 覆盖「A 在 aid1 或 aid2 一侧」的所有协作边。

3. **聚合与过滤**  
   aggregated_results[target_id] += score；若 target_id 已在种子集合中则跳过；设 timeout（如 5s）避免过久。

4. **排序与截断**  
   按聚合得分降序排序，取前 recall_limit 个 author_id 及耗时；不依赖 Faiss、不依赖 JD 向量，只依赖预计算的协作索引与 set 查表。

---

### 5. 总控与融合（total_recall.py — TotalRecallSystem）

**核心目的**  
对 JD **只编码一次**得到统一 query_vec；**并行**跑向量路与标签路，再用两路 Top 100 做种子跑协同路；用 **RRF（Reciprocal Rank Fusion）** 把三路作者列表融合成一条最终排序列表（如 Top 500），并记录每个作者在各路的排名，供精排和展示用。

**使用的方法**

1. **领域与 Query 预处理**  
   若传入 domain_id，则 processed_domain 非空，否则为全领域；**Query 扩展**：若 processed_domain 有效且非训练模式，在 JD 后追加 `" | Area: {DOMAIN_PROMPTS[domain_id]}"`，加强向量路和标签路的领域偏向。

2. **统一编码**  
   query_vec, _ = self.encoder.encode(final_query)，再 faiss.normalize_L2(query_vec)；向量路、标签路共用这一 query_vec。

3. **并行召回**  
   ThreadPoolExecutor：future_v = v_path.recall(query_vec, target_domains)，future_l = l_path.recall(query_vec, domain_ids)；v_list、l_list 取回后，seeds = list(set(v_list[:100] + l_list[:100]))，再 c_list, c_cost = c_path.recall(seeds)。

4. **RRF 融合 `_fuse_results(v_res, l_res, c_res)`**  
   对三路赋予路径权重：向量路 1.2、标签路 1.0、协同路 0.6；对每个作者在其出现的路径上按排名算 RRF 分：score += weight × 1/(rrf_k + rank)，rrf_k=60；同一作者多路出现则分数累加；按总分降序取前 500 个 author_id，并记录每个作者在 v/l/c 三路的排名（rank_map）。

---

### 6. 小结表

| 模块 | 核心目的 | 主要方法 |
|------|----------|----------|
| **文本转向量** | JD → 与索引一致的语义向量，并强化核心技能词 | 动态词库、自共振增强、OpenVINO SBERT、mean pooling、L2 归一化 |
| **向量路** | 按「与 JD 最像的论文」找作者 | 摘要 Faiss 检索、领域硬过滤、论文序→作者序映射 |
| **标签路** | 按「岗位技能→学术词→论文→作者」+ 多维度打分 | 领域探测、锚点 1% 熔断、SIMILAR_TO 扩展、共现共鸣、词权重公式、论文贡献度（纯度/紧密度/时序/署名）、作者聚合 |
| **协同路** | 由种子作者扩展合作者 | 种子=V+L Top100、协作表双向查询、score 聚合、按总分排序 |
| **总控** | 单次编码、三路并行、统一排序 | Query 领域扩展、RRF 融合、路径权重 1.2/1.0/0.6 |

整体上：**向量路**偏语义相似度，**标签路**偏技能/概念与图谱结构，**协同路**偏合作网络；三路在总控里用 RRF 合成一份作者列表，再交给精排与解释模块使用。

---

## 索引构建详解

本节说明四类离线索引的**目的**与**实现方法**，对应代码：`src/infrastructure/database/build_index/` 下的 `build_collaborative_index.py`、`build_feature_index.py`、`build_vector_index.py`、`build_vocab_stats_index.py`；路径与开关由 `config.py` 统一配置。

### 1. 协作索引（build_collaborative_index.py）

**目的**  
预先算出「作者–作者」协作强度，供**协同路召回**使用：给定种子作者（如向量路+标签路 Top100），能快速查出「和谁合作最紧密」的作者并排序，而不在线上现算。

**依赖配置（config.py）**  
- `DB_PATH`：主库（authorships、works、authors）  
- `COLLAB_DB_PATH`：协作库路径，如 `data/build_index/scholar_collaboration.db`

**实现方式**

- **Step 1：单人贡献权重**  
  从 `authorships` + `works` + `authors` 取：`work_id, author_id, pos_index, is_corresponding, is_alphabetical, year, h_index, citation_count`。单篇贡献：`WeightStrategy.calculate(pos_index, is_corr, is_alpha, pub_year, base_year)` → 基数（第一作者 +0.2、通讯 +0.2）× 时间衰减 `e^(-0.1*Δt)`，再乘引用因子 `sqrt(ln(cite + e))`，得到 `composite_weight`。写入协作库表 `weighted_authorships(work_id, author_id, weight, h_index)`，并建 `work_id` 索引。

- **Step 2：直接协作分 S_direct**  
  只保留 2–99 人的论文（排除单人、排除超大合作，降噪）。对每篇论文内作者两两配对 `(a1,a2)`，分数 `score = w1 * w2`（两人在该篇的权相乘）；同一对 `(a1,a2)` 在多篇论文中累加（`ON CONFLICT DO UPDATE s_val += excluded.s_val`）。结果存 `direct_scores(aid1, aid2, s_val, h1, h2)`，主键 `(aid1, aid2)`，`aid1 < aid2` 保证唯一。

- **Step 3：合成最终索引 S_total**  
  先把 `direct_scores` 的 `(aid1, aid2, s_val)` 拷到 `scholar_collaboration`。**间接协作（Bridge）**：对每个 aid1，找「d1.aid2 = d2.aid1」的 d2，即「A–B、B–C」推出「A–C」；间接分 = `0.3 * Σ(d1.s_val * d2.s_val) / ((h2+1)*sqrt(h1+1)*sqrt(h2+1))`，用 h_index 做正则化。按 `aid1` 分区，只保留每个作者 Top-100 合作者（`ROW_NUMBER() ... rank <= 100`）。建覆盖索引：`(aid1, aid2, score)` 与 `(aid2, aid1, score)`，方便协同路双向查「以 aid1 为种子找 aid2」和「以 aid2 为种子找 aid1」。

**产出**  
表 `scholar_collaboration(aid1, aid2, score)` + 双向覆盖索引，供 `collaboration_path.py` 按种子作者批量查协作伙伴并聚合得分。

---

### 2. 特征索引（build_feature_index.py）

**目的**  
为 **KGAT-AX 精排**提供作者、机构的**数值特征**，作为「全息嵌入层」的输入；需要预先做**尺度统一**（对数 + 0–1 归一化），避免 h_index、引用量等长尾特征压制模型对中层学者的区分度。

**依赖配置**  
- `DB_PATH`：主库  
- `FEATURE_INDEX_PATH`：如 `data/build_index/feature_index.json`

**实现方式**

- **作者特征**：SQL 取 `authors` 的 `author_id, h_index, works_count, cited_by_count`。对 `h_index, works_count, cited_by_count` 做 **log1p** 再 **Min-Max** 到 [0,1]（`_normalize`）。存成 `author_features[id] = {h_index, works_count, cited_by_count}`。

- **机构特征**：SQL 取 `institutions` 的 `inst_id, works_count, cited_by_count`。同样 log1p + Min-Max，存成 `inst_features[id] = {...}`。

- **持久化**：JSON 格式 `{"author": author_features, "institution": inst_features, "metadata": {version, scaling_method: "log1p_minmax", timestamp}}`，写入 `FEATURE_INDEX_PATH`。

**产出**  
单个 JSON 文件，精排加载后按 `author_id` / `inst_id` 查表，输入 KGAT-AX 的特征层。

---

### 3. 向量索引（build_vector_index.py）

**目的**  
为**语义检索**提供 Faiss 索引，使线上只需一次向量计算即可在三个对象上做近似最近邻：**摘要索引**供向量路用「JD 向量」搜「最像的论文」再转作者；**岗位索引**供标签路用「JD 向量」搜相似岗位以推断领域、锚点技能；**词汇索引**供标签路用「学术词向量」与 query 做语义守门（cos 相似度）。与 `input_to_vector.py` 使用同一 SBERT 模型与归一化，保证向量空间一致。

**依赖配置（config.py）**  
- `DB_PATH`, `INDEX_DIR`, `SBERT_DIR`, `SBERT_MODEL_NAME`  
- `VOCAB_INDEX_PATH/MAP`, `ABSTRACT_INDEX_PATH/MAP`, `JOB_INDEX_PATH/MAP`

**实现方式**

- **模型与预处理**：使用 `SentenceTransformer(MODEL_NAME, cache_folder=SBERT_DIR)`，`max_seq_length=1024`，CPU + MKL-DNN。长文本 `_smart_trim`：超 4000 字符取前 2000 + 后 2000，避免截断丢失尾部。

- **索引结构**：维度 d 来自模型；`faiss.IndexHNSWFlat(d, 32, METRIC_INNER_PRODUCT)`，`efConstruction=200`；向量 **L2 归一化**，因此内积等价余弦。词汇/岗位：id 为字符串（voc_id / securityId），用 `IndexIDMap` 包一层再 `add_with_ids`，id 写进 map 文件。摘要：id 为 work_id 字符串，写入 `abstract_mapping.json`。

- **三个子任务**：  
  1. **Vocabulary**：`vocabulary(voc_id, term)`，对 `term` 编码 → `vocabulary.faiss` + `vocabulary_mapping.json`。  
  2. **Job**：`jobs(securityId, job_name, description)`，拼接 `job_name + description` 再 trim → `job_description.faiss` + `job_description_mapping.json`。  
  3. **Abstract**：`abstracts(work_id, full_text_en)`，**分片**：每 1 万条一个 shard（`SHARD_SIZE=10000`），先全量 `fetchall()` 到内存避免 OFFSET 慢查，按片 encode、保存 `shard_i.npy` 与 `shard_i_ids.json`，最后 `_merge_abstract_shards` 合并成一份 `abstract.faiss` + `abstract_mapping.json`。

**产出**  
`vocabulary.faiss` + mapping、`job_description.faiss` + mapping、`abstract.faiss` + mapping（及可选 `*_vectors.npy`）。向量路、标签路、输入编码都依赖这些索引与同一 SBERT 空间。

---

### 4. 词汇统计索引（build_vocab_stats_index.py，领域分布 + 共现）

**目的**  
为**标签路召回**提供「每个学术词」的领域统计：**work_count (degree_w)**：该词关联多少篇论文；**domain_span**：涉及多少个不同领域；**domain_dist**：各领域论文数分布（如 `{"1":100,"4":50}`）。用于：领域纯度、领域跨度惩罚（跨领域多的词降权）、以及和 Neo4j 的 HAS_TOPIC 一起做 1% 熔断、IDF/稀缺度等计算。

**依赖配置**  
- `CONFIG_DICT`（Neo4j 连接）  
- `VOCAB_STATS_DB_PATH`：如 `data/build_index/vocab_stats.db`

**实现方式**

- **数据来源**：Neo4j 查询 `(v:Vocabulary)<-[:HAS_TOPIC]-(w:Work)`，对每批词汇 ID 执行 `RETURN v.id, collect(w.domain_ids)`；`domain_ids` 为字符串（如 `"1,4"`），按逗号拆开再统计。

- **统计逻辑**：对每个 vocab：把所有 work 的 domain_ids 拆成单领域 id，用 `Counter` 得到 `domain_dist`；`work_count = sum(dist.values())`，`domain_span = len(dist)`；写入 SQLite：`vocabulary_domain_stats(voc_id, work_count, domain_span, domain_dist, updated_at)`，并建 `domain_span` 索引便于「按跨度过滤/排序」。

- **建库**：若库不存在会创建；`PRAGMA journal_mode=WAL; synchronous=OFF` 提升写入性能；先 `DROP/CREATE vocabulary_domain_stats` 再批量 `executemany`（batch 约 500 词/批，每 1000 条提交）。

- **共现表**：同一脚本还会从主库 `works` 的 concepts_text/keywords_text 构建临时表 `work_terms_temp`，执行 `GET_VOCAB_CO_OCCURRENCE` 得到 (term_a, term_b, freq)，写入 `vocabulary_cooccurrence`（term_a < term_b），与 build_kg 的 CO_OCCURRED_WITH 数据源一致，供索引侧查询词对共现频次而无需依赖 Neo4j。

**产出**  
SQLite 库 `vocab_stats.db` 内两张表：`vocabulary_domain_stats`（标签路在 `_expand_semantic_map`、`_apply_word_quality_penalty` 等处读取，用于领域纯度、跨度惩罚、共振与收敛奖励等）；`vocabulary_cooccurrence`（词对共现频次，标签路通过 `_get_cooccurrence_domain_metrics` 计算 cooc_span/cooc_purity，实现「与多领域词共现→万金油降权」与「只与目标领域词共现→专精加权」，亦可被解释模块按需查询）。

---

### 5. 与 config 的对应关系小结

| 索引 | 目的简述 | 主要方法 | config 中的路径/配置 |
|------|----------|----------|------------------------|
| **协作索引** | 预计算作者协作分，支撑协同路 | 权重(署名+时间+引用)→直接协作→间接 Bridge→Top100+双向覆盖索引 | `DB_PATH`, `COLLAB_DB_PATH` |
| **特征索引** | 作者/机构特征供 KGAT-AX 精排 | log1p + Min-Max，JSON 存 author/inst 特征 | `DB_PATH`, `FEATURE_INDEX_PATH` |
| **向量索引** | 摘要/岗位/词汇的语义检索 | 同一 SBERT、L2 归一化、HNSW 内积；摘要分片合并 | `DB_PATH`, `INDEX_DIR`, `SBERT_*`, `*_INDEX_PATH`, `*_MAP_PATH` |
| **词汇统计索引** | 领域分布 + 词对共现 | Neo4j HAS_TOPIC → vocabulary_domain_stats；主库 work_terms_temp → vocabulary_cooccurrence | `CONFIG_DICT`(Neo4j), `VOCAB_STATS_DB_PATH`, `DB_PATH`, `SQL_QUERIES` |

整体上：**协作索引**面向「谁和谁合作」；**特征索引**面向精排输入；**向量索引**面向三路召回里的语义与领域探测；**词汇领域索引**面向标签路里学术词的质量与领域约束。

---

## KGAT-AX 模型详解

本节说明精排模块 `src/infrastructure/database/kgat_ax` 的**模型结构**、**训练数据从何而来**、**训练与评测流程**，力求在保留技术细节的前提下相对易懂。

### 1. 整体流程与角色

KGAT-AX 是「**知识图谱注意力网络 + 学术指标增强（AX）**」的排序模型，作用是对召回得到的约 500 名候选人做**精排**，产出最终 Top 100 及推荐理由。整条链路可以概括为：

1. **离线准备**：用真实岗位做查询，跑多路召回 → 得到「岗位–候选人」排序与层级 → 写成**四级梯度训练样本**（见下）；同时从 Neo4j 收割**带权图谱三元组**，并建成 SQLite 索引供训练时采样。
2. **训练**：在「User = 岗位、Item = 作者」的协同过滤（CF）信号上，叠加图谱（KG）上的关系约束；用 **AX 特征**（H-index、引用量等）对嵌入做**门控微调**，使排序在「专业对口」为主的前提下，用学术影响力做小幅调优。
3. **评测**：在**测试集**上构造 500 人候选池，用模型打分排序，算 Recall@K、NDCG@K，并用**早停**防止过拟合。

涉及的主要文件：`generate_training_data.py`（样本生成）、`build_kg_index.py`（图谱索引）、`data_loader.py`（数据加载与图构建）、`model.py`（KGAT-AX 结构）、`trainer.py`（训练与评估）、`kgat_utils/metrics.py`（指标）、`pipeline.py`（一键串联）。

---

### 2. 训练数据从何而来

**2.1 四级梯度精排样本（train.txt / test.txt）**

- **目的**：为每个「岗位」准备一批「正例作者」和分层负例，让模型学习「谁更该排前面」的排序关系，而不是简单二分类。
- **谁算正例**：对每个岗位跑一次完整召回（三路 + RRF），得到约 500 人；再按「召回序 × 0.6 + 学术质量序 × 0.4」做融合排序，**前 100 名**视为该岗位的正例（Pos）。
- **四级负例**（由易到难）：
  - **Fair**：融合排序 100–400 中随机抽的一批（和正例较接近，难区分）；
  - **Neutral**：400 名之后的候选人；
  - **Easy**：全库作者中随机抽的（明显不相关，易区分）。
- **格式**：每行一条样本，形如 `user_id;pos1,pos2,...;fair1,...;neutral1,...;easy1,...`。训练时按一定概率从 Fair / Neutral / Easy 里抽负例，与正例组成 (user, pos, neg) 三元组，用于 BPR 类损失。
- **划分**：默认 3000 条岗位→样本进 `train.txt`，300 条进 `test.txt`；且采用 **75% / 25%** 的领域策略：75% 概率用岗位的 `domain_ids` 做领域过滤召回，25% 全库召回，以增强泛化。

**2.2 加权图谱拓扑（kg_final.txt）**

- **目的**：把 Neo4j 里和「岗位–作者–论文–机构–来源–技能词」相关的边导出为带权三元组，供模型学习**关系结构**（谁写了哪篇论文、论文属于哪些领域词等），并在邻接矩阵里使用**权重**（如 AUTHORED 的 pos_weight、SIMILAR_TO 的 score）。
- **关系类型**：AUTHORED(1)、PRODUCED_BY(2)、PUBLISHED_IN(3)、HAS_TOPIC(4)、REQUIRE_SKILL(5)、SIMILAR_TO(6)；每条边存为 `h r t w`（头实体、关系 ID、尾实体、权重）。实体 ID 与训练样本中的 user/entity 共用同一套 `id_map.json`（见下）。
- **ID 与偏移**：所有 **Job 节点**先登记到 `user_to_int`，占 0 ~ ENTITY_OFFSET-1；Author、Work、Vocabulary 等占 ENTITY_OFFSET 往后。这样模型里「User = 岗位、Entity = 岗位+作者+论文+…」统一在一张图上。

**2.3 图谱索引（kg_index.db）**

- **目的**：将 `kg_final.txt` 导入 SQLite，并建立 **(h, r, t, w)** 与 **(t, r, h, w)** 的覆盖索引，方便 DataLoader 按头节点或尾节点**秒级**拉取子图（1-hop + 2-hop 语义桥），而不用每次扫全图。

---

### 3. 数据加载与图结构（DataLoaderKGAT）

- **ID 映射**：读 `id_map.json`，得到 `entity_to_int`、`user_to_int`、`ENTITY_OFFSET`、`n_users_entities`、`n_users`；用于后续所有张量下标。
- **CF 数据**：解析 `train.txt` / `test.txt`，得到 `train_user_dict` / `test_user_dict`（每个 user 对应的正例列表），以及 **tiered_cf_dict**（每个 user 的 fair/neutral/easy 负例列表），供**阶梯负采样**使用。
- **AX 特征**：从 `feature_index.json` 读入作者（和可选机构）的归一化特征（如 h_index、cited_by_count、works_count），按 `entity_to_int` 对齐到 `aux_info_all`，形状为 `(n_users_entities, n_aux_features)`；若文件不存在则全 0。
- **KG 子图采样**：从 `kg_index.db` 中，以「训练集里出现过的 user + entity」为种子，拉取 1-hop 边；再对涉及到的 Vocabulary 节点拉取 **SIMILAR_TO (r=6)** 的 2-hop 边，形成带权子图。边存为 `train_kg_dict`（按头节点）和 `train_relation_dict`（按关系类型），每条边带权重 `w`。
- **加权拉普拉斯矩阵**：用上述边的权重 `w` 构建稀疏邻接矩阵，按行归一化得到 **归一化拉普拉斯**（或随机游走形式），并转为 PyTorch 稀疏张量 `A_in`，供图卷积做**消息传递**；可缓存为 `weighted_adj_cache.npz` 以加速后续运行。

---

### 4. 模型结构（model.py）

**4.1 核心思想**

- 把「岗位」和「作者」等都看成图上的**节点**，共享同一套 **entity embedding**；用 **AX 特征**对嵌入做**乘法门控**，使学术影响力只在「语义相近」的候选人之间起微调作用（约 15% 以内），避免把不相关领域的大佬排到前面。
- 通过多层 **图卷积聚合**（GCN / GraphSAGE / Bi-Interaction 等）在 `A_in` 上做消息传递，得到多跳邻居信息，最终用**拼接各层嵌入**作为该节点的表示，用于 CF 打分和 KG 约束。

**4.2 主要组件**

- **AX 映射层**：`aux_embed_layer` 把 `n_aux_features` 维（如 3 维：H-index、引用、著作数）线性映射到 `embed_dim`；输入会先做 **log1p** 平滑，再通过 sigmoid 得到门控系数，缩放区间为 [1.0, 1.15]（训练时）或 [0.8, 1.0]（精排子集推理时），实现「学术加分但有上限」。
- **类型偏置**：`type_embed` 为「Job」与「Author（及其他实体）」各学一个偏置向量，用 `ENTITY_OFFSET` 区分：id < OFFSET 为 Job，否则为 Entity，使模型区分「查询方」与「被推荐方」。
- **全息融合（holographic_fusion）**：对基础 entity embedding 做「乘以 AX 门控 → L2 归一化 → 加上类型偏置」，得到最终用于 CF 的表示；保证**语义相关性主导、学术影响力微调**。
- **关系与 TransR 相关参数**：`relation_embed`、`trans_M`（每个关系一个变换矩阵）、`W_a` 用于 KG 损失和注意力；在 KG 阶段约束「h + r ≈ t」的 TransR 式距离。
- **聚合器（Aggregator）**：每层用邻接矩阵做一次「邻居聚合 + 激活 + Dropout」；支持 **gcn**（ego+side 再线性）、**graphsage**（concat(ego, side) 再线性）、**bi-interaction**（和与逐元素积两条支路再相加）。多层堆叠后，将**各层输出按维拼接**，作为该节点的多跳表示。

**4.3 前向与损失**

- **CF 损失（calc_cf_loss）**：对 (user, pos, neg) 三元组，取三者的多跳嵌入（已含 AX 融合），用**点积**作为得分，BPR 损失：`softplus(neg_score - pos_score)`，并加 L2 正则。
- **KG 损失（calc_kg_loss）**：对 (h, r, pos_t, neg_t) 及对应的 AX 特征，用 TransR 式「h + r ≈ pos_t、远离 neg_t」的约束，同样用 softplus 形式加 L2；使嵌入在关系空间里符合图谱结构。
- **精排打分（calc_cf_embeddings_subset）**：线上只对「当前岗位 + 候选作者」等子集节点算嵌入（带 AX 门控与类型偏置），不做全图卷积；用岗位嵌入与候选人嵌入的点积作为精排分数。

---

### 5. 训练过程（trainer.py）

- **设备与数据**：默认 CPU；`DataLoaderKGAT` 一次性加载 CF、KG、AX、拉普拉斯矩阵与 tiered_cf_dict。
- **每轮两个阶段**：
  - **CF 阶段**：按 batch 做**阶梯负采样**（从 fair/neutral/easy 中按概率选负例），计算 `calc_cf_loss`，反向传播时对 loss 乘 2 以增强梯度；多轮迭代直到本 epoch 的 CF 步数用完。
  - **KG 阶段**：从 `train_kg_dict` 中按头节点采样 (h, r, pos_t)，随机采 neg_t，计算 `calc_kg_loss` 并更新。
- **周期性地更新图注意力**：如每 1 个 epoch 或每 10 个 epoch 调用 `update_attention_batch`，用当前嵌入和关系参数更新注意力相关权重（用于解释或后续扩展）。
- **评估与早停**：每隔 `evaluate_every` 个 epoch：
  - 在**训练集**上算一次 Recall/NDCG（拟合度，仅作参考）；
  - 在**测试集**上算 Recall/NDCG（**泛化指标**）：对每个测试岗位，用其 tiered 候选池（约 500 人）作为候选集，用模型重排后与 test_user_dict 中的正例比较，调用 `calc_metrics_at_k` 得到 Recall@K、NDCG@K。
  - 若测试集 Recall 连续 `stopping_steps` 次未提升，则**早停**；仅当测试集表现优于历史最佳时，保存为 best model。
- **断点续训**：会扫描权重目录中的最新 checkpoint，自动加载 `model_state_dict`、`optimizer_state_dict`、epoch 与 best_recall，从下一轮继续训练。

---

### 6. 评测指标与方式（metrics.py）

- **Recall@K**：对每个 user，将其正例在排序中的前 K 位命中数除以该 user 的正例总数，再对 user 求平均。
- **NDCG@K**：考虑排序位置的折扣增益，归一化后对 user 求平均。
- **实现要点**：评测前会把「训练集正例」从候选得分中 mask 掉（设为 -inf），只根据「测试集正例」计算命中；这样评估的是**泛化**到未见过的岗位–作者对上的排序能力。拟合度评估时则用「不 mask 训练集」的配置，仅观察模型是否能在训练集上复现排序。

---

### 7. 小结

| 环节 | 目的 | 关键方法 |
|------|------|----------|
| **训练数据** | 得到岗位→候选人层级与图谱边 | 多路召回 + 融合排序 → 四级梯度；Neo4j 加权拓扑 → kg_final.txt；SQLite 索引 |
| **数据加载** | 为模型提供 CF 对、KG 三元组、AX 特征、归一化图 | id_map、tiered 负采样、1/2-hop 加权子图、拉普拉斯缓存 |
| **模型** | 在图上学习岗位–作者匹配，并用学术指标微调排序 | 实体嵌入 + AX 门控 + 类型偏置；多层图聚合；CF BPR + KG TransR 损失 |
| **训练** | 稳定优化并避免过拟合 | CF/KG 交替；测试集 Recall 驱动早停与 best 保存；断点续训 |
| **评测** | 衡量精排泛化能力 | 500 人候选池重排；Recall@K、NDCG@K；mask 训练集正例 |

整体上，KGAT-AX 把「岗位–作者」的协同信号与「作者–论文–技能词–岗位技能」的图谱信号放在同一张图上，用**加权图卷积 + AX 门控**学出联合表示，从而在召回结果上做**语义与影响力平衡**的精排，并为后续解释（如注意力、路径）提供接口。

---

## 精排与解释详解

本节说明**精排阶段**的细节、流程与设计考虑，对应代码：`src/core/ranking/ranking_engine.py`、`rank_scorer.py`、`rank_explainer.py`；精排在 `TotalCore.suggest()` 中位于「语义导航 → 多路召回」之后，负责对约 500 名候选人做重排与可解释输出。

### 1. 精排在整体链路中的位置

精排由 **TotalCore.suggest()** 触发，顺序为：

1. **语义导航**：用 JD 向量在岗位 Faiss 索引上搜 Top-3 岗位，得到 `real_job_ids`（精排用的「岗位锚点」）。
2. **多路召回**：对同一 query 做向量路 + 标签路 + 协同路，RRF 融合得到约 500 人 `candidates`。
3. **精排过滤策略**：若用户指定了领域则用该领域；否则用三个锚点岗位的 `domain_ids` 并集得到 `filter_pattern`。
4. **精排引擎**：`RankingEngine.execute_rank(real_job_ids, candidates, filter_domain=filter_pattern)`，得到最终 Top 100 及解释。

即：**精排输入** = 若干岗位锚点 + 约 500 名召回候选人 + 可选领域过滤；**输出** = 排序后的 100 人及每人一条推荐理由、代表作等结构化信息。

---

### 2. RankingEngine.execute_rank：主流程与设计考虑

**2.1 领域并集预过滤（可选）**

- **目的**：在算 KGAT 分之前，按「作者是否在目标领域有论文」做硬过滤，避免明显跨领域的人进入精排并减少计算量。
- **实现**：若 `filter_domain` 非空且非 `"0"`，调用 `_filter_by_domain_union`：用 `DomainProcessor.to_set(pattern)` 得到目标领域集合；在 SQLite 查 `authorships` + `works` 得到每位候选人的 `domain_ids`；用 `DomainProcessor.has_intersect` 保留至少有一篇论文落在目标领域的作者，并**维持原有召回顺序**。
- **考虑**：过滤后列表可能变短；若无人通过则直接返回空，避免无意义的模型推理。

**2.2 批量计算 KGAT 精排分（RankScorer）**

- **目的**：在「岗位锚点 + 当前候选人子集」上只算这部分节点的嵌入与得分，不做全图前向。
- **实现**：`kgat_scores = self.scorer.compute_scores(real_job_ids, active_candidates)`，得到与 `active_candidates` 等长的一维分数张量；具体打分逻辑见下节。

**2.3 召回顺序分（recall_scores）**

- **目的**：保留「召回阶段已经排好的序」的信息，避免精排模型完全重写顺序导致不稳定或过度依赖训练分布。
- **实现**：对当前 `active_candidates` 按在列表中的下标给线性递减分：`recall_scores = torch.linspace(1.0, 0.0, steps=recall_len)`，即第一名 1.0，最后一名 0.0。
- **考虑**：召回顺序是 RRF 融合结果，用线性分等价于给「召回名次」一个单调权重。

**2.4 KGAT 分的归一化与 40/60 融合**

- **目的**：把「模型分」和「召回序分」放在同一量纲再融合，并通过权重控制「模型改序」的力度。
- **实现**：对 `kgat_scores` 做 Min-Max 归一化到 [0,1]；融合：`final_fusion_scores = 0.4 * kgat_norm + 0.6 * recall_scores`。
- **考虑**：**60% 召回 + 40% 模型** 表示更信任多路召回的排序，用 KGAT 做「微调」而不是彻底重排，既利用模型学到的岗位–作者匹配与 AX，又降低模型偏差或冷启动风险。

**2.5 取 Top 100 并组装结果**

- 对 `final_fusion_scores` 做 `topk(100)`；对每个 Top 100 内的候选人：用 `active_candidates[original_idx]` 得到原始作者 ID，从 SQLite 拉基础学术指标（`_fetch_sqlite_stats`）；调用 **RankExplainer.explain(raw_aid, real_job_ids)** 得到代表作、推荐理由、合作者、匹配类型等；拼成一条结果（rank、author_id、name、score、representative_work、recommendation_reason、metrics、collaboration、details 等）。
- **考虑**：最终展示的 `score` 是融合分；details 里保留 `kgat_score` 和 `recall_score` 便于分析与调试。

---

### 3. RankScorer.compute_scores：打分逻辑与考虑

**3.1 ID 对齐**

- `real_job_ids` 用 `dataloader.user_to_int` 转成训练时的 user 下标 `job_int_ids`；`candidate_raw_ids` 用 `entity_to_int` 的 key `a_{author_id}` 转成 entity 下标 `item_int_ids`。

**3.2 局部子集嵌入与 AX**

- 构造 `all_active_ids = [job_int_ids; item_int_ids]`，只包含本请求的岗位 + 候选人；从 `dataloader.aux_info_all` 取这些下标的 AX 特征，得到 `aux_info_subset`；调用 `model.calc_cf_embeddings_subset(all_active_ids, aux_info_subset)`，得到**带 AX 门控和类型偏置的嵌入**（不跑全图卷积）。

**3.3 「理想人选」向量与得分**

- 把嵌入按岗位数切开：`job_embeds`、`item_embeds`；**理想人选向量**：`u_e_avg = mean(job_embeds, dim=0)`，即对多个岗位锚点取平均；**综合得分**：`combined_scores = u_e_avg @ item_embeds.T`，即每个候选人与「平均岗位向量」的内积。
- **考虑**：多锚点取均是为了用「多个相似岗位」的整体需求向量打分，减轻单岗位噪声；内积在 L2 归一化下等价于余弦相似度，体现「语义 + AX 微调」后的匹配度；「语义主导、学术择优」由 `calc_cf_embeddings_subset` 内部的 AX 门控（如 0.8~1.0 的缩放）保证。

**3.4 诊断输出**

- 打印锚点数、候选人数、分数最小值/最大值和极差，便于观察分数是否塌缩或区分度是否足够。

---

### 4. RankExplainer.explain：解释链与设计考虑

**4.1 多跳路径查询（主路径）**

- **目的**：找到「岗位要求的技能词 →（可选）语义相似词 → 该作者写过且命中这些词的论文」的路径，作为可解释证据。
- **实现**：从当前请求的 Job 出发，沿 `REQUIRE_SKILL` 到 `v1:Vocabulary`，并过滤过于泛化的词；`OPTIONAL MATCH (v1)-[r:SIMILAR_TO]-(v2)` 且 `r.score > 0.7`，得到与岗位技能强相似的学术词 `target_v`，没有则用 `v1`；再 `(target_v)<-HAS_TOPIC-(w:Work)<-AUTHORED-(a:Author)`，限制到当前被解释的作者；顺带拉取发表来源与合作者（最多 2 人）；对每条路径标记 `match_type`：若 `v1 = target_v` 为 `exact`，否则为 `semantic`；按 `match_type` 排序并 LIMIT 15。
- **考虑**：**score > 0.7** 控制只走强相似边，避免语义漂移；**过滤泛词** 避免用「计算机科学」等大而化之的标签当证据；**exact / semantic** 区分岗位词直接命中与通过相似词间接命中。

**4.2 语义保底（Fuzzy Match）**

- **目的**：若主路径一条都没命中，用岗位 skills 的字符串包含关系去匹配 Vocabulary。
- **实现**：用 `split(job_skills, ',')` 与 `v.term CONTAINS s` 之类条件在 Author→Work→Vocabulary 上再查一次，最多 5 条，标记为 `match_type='fuzzy'`。

**4.3 终极 Fallback**

- 若主路径和 Fuzzy 都没有结果，返回 `_generate_fallback_response()`：通用文案（如「全息领域匹配」「多篇核心领域产出」及概括性总结），避免前端没有理由可展示。

**4.4 用 KGAT 注意力选「证据论文」**

- **目的**：同一作者可能有多条路径（多篇论文），选「模型最认可」的一篇作为代表作。
- **实现**：对当前作者和每条路径上的 work 节点，用 `model.update_attention_batch` 算 AUTHORED 关系上的注意力权重，赋给每条路径的 `att_weight`；按 `(match_type == 'fallback', -att_weight)` 排序，优先非 fallback，再按注意力从高到低取第一条作为 `best_path`。
- **考虑**：把「图结构证据」和「模型置信度」结合：路径存在且注意力高，说明既有图谱依据又符合模型学到的关系强度。

**4.5 动态总结文案**

- **目的**：把 `best_path` 里的技能词、论文标题、来源、合作者转成一句可读的推荐理由，并降低重复感。
- **实现**：`_build_dynamic_summary(best_path)`：从 path 取 `req_skill`、`match_skill`、`title`、`source_name`、`co_authors`；多套模板（精准对齐 / 语义关联与社交背书 / 能力验证），若 `match_type == 'exact'` 用模板 A，否则在 B/C 中随机选；若有合作者则追加协作关系后缀。
- **考虑**：用 **exact → 固定模板**、**semantic/fuzzy → 随机模板** 在保证精确匹配时说法一致的前提下，增加语义/模糊匹配时的表述多样性。

**4.6 返回结构**

- 返回字典包含：matched_skill、key_evidence_work、work_id、work_url、source、collaborators、match_type、model_confidence（注意力）、summary；fallback 时部分字段为通用占位。ranking_engine 再把这些填进每条结果的 `representative_work`、`recommendation_reason`、`collaboration`、`details.match_type` 等字段。

---

### 5. 设计考虑小结

| 环节 | 考虑 |
|------|------|
| **领域预过滤** | 用 DomainProcessor 做作者–领域交集，只精排「有该领域论文」的人，兼顾可解释性与效率。 |
| **多锚点平均** | 用 3 个语义最近岗位的嵌入平均作为「理想人选」，平滑单岗位噪声。 |
| **局部嵌入 + AX** | 只对锚点+候选人算嵌入并注入 AX，保证「语义为主、学术微调」，且推理不扫全图。 |
| **40/60 融合** | 精排分与召回序分归一化后 4:6 融合，既用模型改序又不过度颠覆召回结果。 |
| **解释三层** | 主路径(精确/语义) → Fuzzy 匹配 → Fallback 通用句，保证总有理由可展示。 |
| **SIMILAR_TO > 0.7** | 解释时只用强相似边，避免证据链语义漂移。 |
| **注意力选论文** | 多篇命中时用 KGAT 注意力选一篇作为代表作，兼顾图谱证据与模型置信度。 |

整体上，精排阶段在「多路召回 + 领域过滤」之后，用 **RankScorer** 做一次「岗位–候选人」的 KGAT+AX 打分，用 **召回序分** 做稳定性约束，再用 **RankExplainer** 从图谱和注意力生成可解释的代表作与推荐理由，最终输出 Top 100 及每条的结构化详情。

---

## 代码文件详细说明

### 根目录

- **`app.py`**  
  简化版 Streamlit 应用入口，从 `src.interface.dashboard`（历史版本）与 `src.infrastructure.mock_data` 读取模拟人才与图谱数据，主要用于快速 Demo。

- **`main.py`**  
  使用 `streamlit.web.cli` 以编程方式执行：
  ```bash
  streamlit run src/interface/app.py --server.port=8501 --server.address=127.0.0.1
  ```
  并在控制台打印启动横幅，是推荐的主启动脚本。

- **`config.py`**  
  全局配置中心：
  - 定义 SQLite DB 路径、索引文件路径、SBERT 模型目录；
  - 定义 Neo4j 连接参数与 Cypher 模板；
  - 提供 `SQL_QUERIES` 与 `SQL_INIT_SCRIPTS` 供 KG 构建与索引初始化使用；
  - 定义 17 个业务领域 ID 与名称的映射（`DOMAIN_MAP` / `NAME_TO_DOMAIN_ID`）。

- **`index.html`**  
  Vite + Vue 单页应用入口 HTML，挂载 `#app` 容器并加载 `/src/main.ts`，当前工程中主要用于前端实验，不影响 Python 推荐链路。

- **`vite.config.ts`**  
  Vite 配置文件：
  - 集成 Vue3 插件与 Element Plus 组件按需自动导入；
  - 配置 `@` 别名指向 `src`；
  - 预优化常用依赖（vue / element-plus / echarts 等）；
  - 配置开发服务器 `hmr.overlay = false` 减少前端调试噪音。

- **`tsconfig.json` / `tsconfig.node.json`**  
  TypeScript 编译配置，指定模块系统、库文件、路径别名以及对 Vite 配置文件的引用。

- **`test.js`**  
  ES Module 风格的 Node.js 性能测试脚本，用于简单测试 CPU 与文件 IO 能力，与推荐业务无直接关系。

- **`README.md`**  
  原始英文 README，描述了较早版本的项目结构和用法，本文件 `README_NEW.md` 为最新、完整的中文说明。

- **`.gitignore`**  
  定义前端构建产物、Python 虚拟环境、数据库文件、向量索引、模型权重和 IDE 配置等需要忽略的文件夹。

### `data` 目录

- **`data/attribute.py`**  
  演示 / 占位脚本，可用于扩展与岗位或作者相关的领域属性配置，目前未在主链路中被直接引用。

---

### 核心推荐引擎 `src/core`

- **`src/core/total_core.py`**  
  - 定义 `TotalCore` 类，是 **线上推荐核心入口**。  
  - 主要职责：
    - 解析 KGAT‑AX 训练参数，初始化数据加载器与 KGAT 模型；
    - 构建 `TotalRecallSystem`（召回子系统）与 `RankingEngine`（精排子系统）；
    - 提供 `suggest(text_query, manual_domain_id)` 方法：完成文本编码、召回、领域处理与精排，并返回结构化的专家推荐列表。
  - 在命令行直接运行该文件可进入交互式专家推荐控制台。

---

### 召回模块 `src/core/recall`

- **`src/core/recall/input_to_vector.py`**  
  - 实现 `QueryEncoder` 类，利用 **OpenVINO 加速的 SBERT 模型** 将 JD 文本编码为归一化向量；
  - 自动构建「技能词动态词典」，对 JD 中命中的关键术语进行「自共振增强」；
  - 输出 `(向量, 耗时)`，供向量召回与标签路使用。

- **`src/core/recall/total_recall.py`**  
  - 定义 `TotalRecallSystem`：
    - 管理 Query 编码器与三条召回路径（向量路 / 标签路 / 协同路）；
    - 将 JD 文本扩展为带领域 bias 的查询；
    - 并行调用三条路径，使用 Reciprocal Rank Fusion (RRF) 融合为最终候选作者列表；
    - 返回前 500 名候选作者以及各路召回排名信息。
  - 同时提供命令行模式，用于单独测试召回效果。

- **`src/core/recall/vector_path.py`**  
  - `VectorPath` 类，负责向量语义召回：
    - 加载论文摘要与岗位描述的 Faiss 索引与 ID 映射；
    - 通过 `recall(query_vector, target_domains)` 执行：
      1. Faiss 在摘要索引上检索相似论文；
      2. 用 SQLite 查询这些论文的 `domain_ids`，并通过 `DomainProcessor` 做硬过滤；
      3. 将论文映射为作者，并按论文语义相关性排序；
      4. 返回作者 ID 列表与耗时。
  - 文件末尾提供独立测试脚本，可交互式体验向量路召回。

- **`src/core/recall/label_path.py`**  
  - `LabelRecallPath` 类，是 **最复杂的标签路召回引擎**，关键步骤：
    1. 通过 Job 向量索引检测 Query 所属领域；
    2. 从 Job 节点中提取高含金量技能锚点；
    3. 在 Neo4j 中沿 `SIMILAR_TO`、`HAS_TOPIC` 等关系扩展学术词集合；
    4. 结合 SQLite 的 `vocabulary_domain_stats` 与 SBERT 向量对候选学术词多维打分；并利用 `vocabulary_cooccurrence` 通过 `_get_cooccurrence_domain_metrics` 计算 **cooc_span**（共现伙伴领域跨度）与 **cooc_purity**（共现伙伴目标领域纯度），实现万金油降权与领域专精加权；同时保留学术共鸣（与本次搜索词共现）加成；
    5. 在 Neo4j 中反查 `(Vocabulary)<-[:HAS_TOPIC]-(Work)<-[:AUTHORED]-(Author)`，基于论文贡献度与时序衰减聚合成作者得分。
  - 输入输出格式不变：`recall(query_vector, domain_ids)` 仍返回 `(author_id_list, elapsed_ms)`；语义扩展仍返回 `(score_map, term_map, idf_map)`。
  - `last_debug_info` 字段用于输出完整诊断链路（领域探测、锚点、学术词质量、论文/作者规模等），便于调参与可解释性分析。

- **`src/core/recall/collaboration_path.py`**  
  - `CollaborativeRecallPath` 类，实现协同路召回：
    - 从 `COLLAB_DB_PATH`（协作索引 SQLite）读取 `scholar_collaboration` 表；
    - 基于种子作者 ID 集合，查询「协作关系最紧密的 Top‑K 伙伴」；
    - 利用高性能覆盖索引与批量 SQL，避免 pandas 带来的额外开销；
    - 返回扩展候选人列表与耗时。

---

### 精排与解释模块 `src/core/ranking`

- **`src/core/ranking/ranking_engine.py`**  
  - 封装精排主流程：
    - 初始化 Neo4j Graph、ID 映射表和 `RankScorer` / `RankExplainer`；
    - 在 `execute_rank(real_job_ids, candidate_raw_ids, filter_domain)` 中：
      1. 可选执行领域并集过滤（通过 `DomainProcessor.has_intersect` 对作者论文领域做交集判断）；
      2. 使用 `RankScorer` 计算 KGAT 分数；
      3. 生成「召回顺序分」并进行归一化，与 KGAT 分数按 0.4 / 0.6 权重融合；
      4. 选取 Top 100，查询作者基础学术指标，并调用 `RankExplainer` 生成推荐理由与代表作；
      5. 组装为前端友好的结构化结果。

- **`src/core/ranking/rank_scorer.py`**  
  - `RankScorer` 主要负责：
    - 将岗位锚点与候选作者原始 ID 映射到 KGAT 内部 ID 空间；
    - 构造活跃节点子集，调用 `model.calc_cf_embeddings_subset` 获取融合 AX 特征的嵌入；
    - 对岗位嵌入取均值得到「理想人选向量」，与候选人嵌入做点积得到综合得分；
    - 打印诊断信息（分数区间 / 极差等）。

- **`src/core/ranking/rank_explainer.py`**  
  - `RankExplainer` 将 Neo4j 图谱信息与 KGAT 注意力结合，生成可读的推荐解释：
    - 优先沿 `Job-REQUIRE_SKILL-Vocabulary-SIMILAR_TO-Vocabulary-HAS_TOPIC-Work-AUTHORED-Author` 路径寻找证据链；
    - 为每条路径计算注意力权重，选出最有说服力的论文与技能匹配；
    - 拼接多模板中文解释文案，包含技能对齐、代表作、发表平台、合作伙伴等信息；
    - 提供 Fallback 逻辑，确保在图谱欠完备时也能返回合理解释。

---

### 接口层 `src/interface`

- **`src/interface/app.py`**  
  - Streamlit Web 前端主程序：
    - 调用 `TotalCore` 懒加载初始化（`@st.cache_resource`）；
    - 左侧侧边栏允许多选业务领域；如果不选，则走自动岗位并集模式；
    - 主界面提供 JD 文本输入与「开始深度匹配」按钮；
    - 结果部分以卡片形式展示：
      - 综合得分 & 召回 / 精排拆分；
      - 人才名字与 Author ID；
      - 推荐理由（由 `RankExplainer` 生成）；
      - 代表作标题、OpenAlex 链接及发布平台；
      - H‑Index / 总论文数 / 总引用数等学术指标。

---

### 通用工具 `src/utils`

- **`src/utils/domain_utils.py`**  
  - `DomainProcessor`：对领域 ID 相关操作进行统一封装：
    - `to_set()`：将 `1|4|14`、`"1,4"`、列表等各种输入统一解析为集合；
    - `build_neo4j_regex()`：构造用于 Neo4j 的正则过滤表达式；
    - `build_python_regex()`：构造用于 Python `re` 过滤的正则对象；
    - `has_intersect()`：判断论文 `domain_ids` 与目标领域集合是否有交集，是召回与精排阶段过滤的核心工具。

---

### 基础设施：数据库与图谱 `src/infrastructure/database`

#### `build_kg` 子模块

- **`src/infrastructure/database/build_kg/builder.py`**  
  - `KGBuilder`：封装 Neo4j 知识图谱构建的各阶段任务：
    - `sync_nodes_task()`：通用节点同步（Author / Work / Vocabulary / Institution / Source / Job）；
    - `build_topology_incremental()`：基于 authorships 建立加权 `AUTHORED`、`PRODUCED_BY`、`PUBLISHED_IN` 等复杂拓扑；
    - `build_semantic_bridge()`：基于 SBERT + Faiss 计算词汇间相似度，并写入 `SIMILAR_TO`；
    - `build_work_semantic_links()`：基于 Aho‑Corasick 自动机扫描论文标题和元数据，补齐 `HAS_TOPIC`；
    - `build_job_skill_links()`：基于 `jobs.skills` 将岗位与技能词汇建立 `REQUIRE_SKILL` 关系；
    - `build_cooccurrence_links()`：统计词汇在同一论文中的共现频率，构建 `CO_OCCURRED_WITH` 共现网络。

- **`src/infrastructure/database/build_kg/kg_utils.py`**  
  - 定义：
    - `GraphEngine`：Neo4j 驱动封装，负责批量写入、执行 schema 变更等；
    - `SyncStateManager`：在 SQLite 中维护各同步任务的断点标记（marker），支持增量构建；
    - `Monitor`：监控各阶段耗时与资源占用。

- **`src/infrastructure/database/build_kg/generate_kg.py`**  
  - `run_pipeline(CONFIG_DICT)`：一键执行完整 KG 构建流程：
    1. 确保 Neo4j 约束与索引存在、SQLite 建立必要索引；
    2. 执行节点同步与拓扑构建；
    3. 执行语义打标 / 共现网络构建 / 语义桥接；
    4. 在脚本末尾提供 `__main__` 入口，便于命令行直接启动流水线。

#### `build_index` 子模块

- **`build_vector_index.py`**  
  - `StableVectorGenerator` 使用 SentenceTransformer 将 Vocabulary / Job / Abstracts 编码为向量：
    - 构建词汇向量索引（Vocabulary）；
    - 构建岗位描述向量索引（Job Description）；
    - 构建论文摘要大型向量索引（Abstract），支持分片处理与合并；
  - 输出 HNSW 型 Faiss 索引及配套 `_vectors.npy` 快照与 ID 映射 JSON。

- **`build_collaborative_index.py`**  
  - `LocalSimilarityIndexer`：基于 authorships 与 works 表构建本地协作相似度索引：
    - 阶段 1：为每条作者‑论文关系计算时间衰减与引用强度加权后的贡献度；
    - 阶段 2：统计作者两两合作的直接协作分数；
    - 阶段 3：引入间接协作（共同合作者）与 H‑index 正则化，生成最终 `scholar_collaboration` 表，并裁剪为 Top‑100 伙伴。

- **`build_vocab_stats_index.py`**  
  - `VocabStatsIndexer` 构建 vocab_stats.db 中两张表：（1）`vocabulary_domain_stats`：从 Neo4j 收集 `(Vocabulary)<-[:HAS_TOPIC]-(Work)`，为每个词计算关联论文数、领域跨度、各领域论文分布（JSON），供标签路评分；（2）`vocabulary_cooccurrence`：从主库 works 计算词对共现频次 (term_a, term_b, freq)，与 KG CO_OCCURRED_WITH 一致，供索引侧查询共现。

- **`build_feature_index.py`**  
  - `FeatureIndexBuilder` 从 `authors / institutions` 表抽取学术指标（H‑index、论文数、引用数），做 log1p + Min‑Max 归一化，生成：
    - 作者特征字典；
    - 机构特征字典；
  - 将结果写入 JSON，作为 KGAT‑AX `aux_info_all` 输入。

- **`download_model.py`**  
  - 使用 `huggingface_hub.snapshot_download` 断点续传方式下载 `Alibaba-NLP/gte-multilingual-base` 到本地 `SBERT_DIR`，解决国内网络环境下模型下载不稳定问题。

#### `kgat_ax` 子模块

- **`model.py`**  
  - 定义 KGAT‑AX 主模型：
    - 包含多层聚合器（GCN / GraphSAGE / Bi‑Interaction）；
    - 将结构化 AX 特征通过线性层与 sigmoid 门控注入实体向量；
    - 提供 CF / KG 损失计算、注意力更新与局部子集嵌入计算接口。

- **`data_loader.py`**  
  - `DataLoaderKGAT`：加载训练数据与加权图：
    - 从 `train.txt / test.txt / kg_final.txt` 中解析用户‑正样本‑负样本四级梯度结构；
    - 读取 `kg_index.db`，构建包含权重的三元组与拉普拉斯矩阵；
    - 加载 AX 特征 JSON 并映射到全局节点空间；
    - 提供 CF / KG 批次采样与数据质量检查工具。

- **`trainer.py`**  
  - 训练主循环：
    - 支持断点续训与最佳模型保存；
    - 在 CF 阶段采用四级难度负采样策略；
    - 在 KG 阶段执行 TransR 风格图谱训练；
    - 定期更新图注意力权重；
    - 通过 `evaluate()` 在训练集与测试集上分别评估 Recall / NDCG，并使用早停策略。

- **`pipeline.py`**  
  - 封装三步流水线：
    1. `generate_training_data.py`：生成四级梯度训练样本；
    2. `build_kg_index.py`：构建加权 KG 索引；
    3. `trainer.py`：执行 KGAT‑AX 训练；
  - 用于一键启动从样本生成到模型训练的全流程。

- **`generate_training_data.py`**  
  - `KGATAXTrainingGenerator`：
    - 通过 `TotalRecallSystem` 对抽样岗位执行召回，将候选专家按召回排名与学术质量融合排序；
    - 生成 `train.txt / test.txt`，每行包含 `user_id;pos;fair;neutral;easyNeg` 四级样本；
    - 从 Neo4j 导出加权关系并映射到统一 ID 空间，生成 `kg_final.txt` 与 `id_map.json`。

- **`build_kg_index.py`**  
  - `KGIndexBuilder`：将 `kg_final.txt` 导入到 `kg_index.db` 中的 `kg_triplets` 表，并建立双向覆盖索引，提升子图采样效率。

- **`verify_recall.py`**  
  - 离线验证脚本：
    - 加载最新或最佳模型权重；
    - 构造「领域硬过滤」与「全库搜索」两种召回策略，比较 Recall；
    - 使用标准 `evaluate()` 在 test 集上评估 Recall@K 与 NDCG@K。

- **`kgat_utils/metrics.py` / `log_helper.py` / `model_helper.py` / `early_stopping` 等**  
  - 提供通用评估指标计算、日志配置、模型存取与早停判断工具。

- **`kgat_parser/parser_kgat.py` / `parser_ecfkg.py` / `parser_bprmf.py` / `parser_cke.py`**  
  - 为 KGAT / ECFKG / BPRMF / CKE 等不同模型提供命令行参数解析，当前主线使用 `parse_kgat_args()`。

#### `models/sbert_model` 子模块

- **`README.md`**  
  HuggingFace 官方 SBERT 模型说明（多语言句向量模型）。

- **`download_sbert.py`**  
  简单的脚本式下载器，在本地 `sbert_model` 目录不存在核心文件时，从 HuggingFace 拉取 `paraphrase-multilingual-MiniLM-L12-v2`。

- **`test_similarity.py`**  
  验证本地 SBERT 模型是否能正确加载与编码，并测试几组中英文短语的语义相似度。

---

### 基础设施：数据抓取与合并 `src/infrastructure/crawler`

- **`new_getdata_likelogin0104.py`**  
  - 基于 DrissionPage 的 BOSS 直聘高学历岗位爬虫：
    - 支持城市与岗位关键词的断点续爬；
    - 内置多级休眠与页面稳定性监测，尽量模拟真实用户行为；
    - 仅保留学历为「硕士」或「博士」的岗位数据，写入 `jobs.csv`。

- **`merge_database.py`**  
  - `UnifiedTalentDB` 继承自 `use_openalex.database.DatabaseManager`：
    - 创建并管理统一的 `academic_dataset_v5.db` 数据库；
    - 新增 `jobs` 表与学历索引；
    - 将 `jobs.csv` 增量合并到数据库，并将岗位技能插入 `vocabulary` 作为 `industry` 词汇。

#### `use_openalex` 子模块（学术数据抓取）

- **`db_config.py`**  
  - 定义：
    - `EMAIL`：OpenAlex polite pool 身份信息；
    - `DB_PATH` / `DATA_DIR`：SQLite 数据库路径与导出目录；
    - `FIELDS`：17 个学科概念及其 OpenAlex concept ID；
    - 抓取延迟与批大小等性能参数。

- **`database.py`**  
  - `DatabaseManager`：学术数据库统一管理类：
    - 初始化并升级 SQLite 架构，创建 works / authors / institutions / abstracts / authorships / sources / vocabulary 等表及大量性能索引；
    - 提供保存论文、作者、机构、关系、词汇等的一系列批量方法；
    - 维护 `crawl_states` / `author_process_states` 等表，实现增量爬取；
    - `refresh_stats()` 用于更新作者 / 机构 / 渠道的统计指标。

- **`models.py`**  
  - `Work` / `AuthorProfile` 数据类：抽象了论文与作者的结构化画像，用于后续特征工程与分析。

- **`utils.py`**  
  - 提供：
    - `safe_get()`：安全访问嵌套字典；
    - `clean_id()`：从 OpenAlex / DOI URL 中提取纯 ID；
    - `generate_work_id()`：统一生成论文主键（优先使用 OpenAlex ID）。

- **`api_client.py`**  
  - `APIClient`：对 OpenAlex API 的封装：
    - 内建 polite pool 规范与速率限制调度；
    - 使用 `requests.Session` + `Retry` 实现自动重试；
    - 针对 401/402/403/429 等错误编码进行专门处理，当额度耗尽时友好退出。

- **`crawler_logic.py`**  
  - 各 Phase 具体实现：详见「整体架构」部分的 Phase 1–5 描述。

- **`processor.py`**  
  - `DataProcessor` 负责单篇 `work` 的完整落库：
    - 还原摘要文本；
    - 推断领域 ID 并写入 `works.domain_ids`；
    - 拆解 authorships 并写入 authors / institutions / sources / authorships 表。

- **`main.py`**  
  - CLI 入口：提供 `--field`、`--force`、`--stats` 等参数，支持：
    - 查看当前数据库规模与领域分布；
    - 单领域或全领域的增量 / 强制重爬；
    - 执行 Phase 4 / Phase 5 的同步任务。

#### `use_openalex/some_tool` 辅助脚本

- **`delete_doi.py` / `migrate_ids.py` / `fix_doi_to_openalex_id.py` / `refresh_metrics.py` / `replace.py` / `single_recrawl_not_found.py` / `use_name_to_fix_workid.py`**  
  一组用于维护与修复数据库状态的小工具脚本，例如：
  - 使用名称或 DOI 重新匹配 OpenAlex ID；
  - 刷新统计指标；
  - 重新爬取缺失论文等。

---

### 基础设施：SBERT 模型工具 `src/infrastructure/database/models/sbert_model`

（已在上文 `models/sbert_model` 小节说明，这里不再赘述。）

---

### 辅助数据脚本与配置

- **`src/infrastructure/mock_data.py`**  
  - 定义少量模拟人才 (`mock_talents`) 与简化图结构 (`mock_graph`)，用于无数据库情况下快速展示前端界面效果。

---

## 技术栈综述

- **语言与运行环境**
  - Python 3.x
  - Node.js + Vite + Vue3（前端壳）

- **后端框架与可视化**
  - **Streamlit**：推荐系统可视化界面与快速 Demo。

- **检索与向量化**
  - **SentenceTransformers / Transformers**：SBERT 文本向量编码；
  - **Optimum Intel (OpenVINO)**：推理加速；
  - **Faiss**：大规模向量相似度检索（HNSW）。

- **图数据库与图算法**
  - **Neo4j**：存储人才‑技能‑岗位知识图谱；
  - **py2neo / neo4j 驱动**：图谱读写；
  - 自研 **KGAT‑AX** 模型：知识图谱注意力网络 + 学术特征增强。

- **数据存储与预处理**
  - **SQLite**：统一存储 OpenAlex 抓取数据与岗位数据；
  - **pandas / NumPy / SciPy**：特征处理与稀疏矩阵构建；
  - **tqdm / logging**：进度与日志管理。

- **数据抓取**
  - **DrissionPage**：BOSS 直聘 Web 自动化；
  - **requests + urllib3**：OpenAlex REST API 调用。

- **其他**
  - **huggingface_hub**：模型下载；
  - **ahocorasick**：高性能多模式匹配，用于标题打标。

---

## 环境准备与安装

### 1. Python 环境

```bash
# 建议使用 Python 3.9+ 虚拟环境
cd TalentRecommendationSystem-master
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
```

### 2. 安装核心依赖（示例）

若仓库中已有 `requirements.txt`，优先使用：

```bash
pip install -r requirements.txt
```

如无，可按下列最小依赖自行安装（根据需要增减）：

```bash
pip install streamlit torch faiss-cpu sentence-transformers \
            optimum[openvino] py2neo neo4j \
            pandas numpy scipy scikit-learn tqdm requests \
            drissionpage huggingface_hub ahocorasick
```

> **建议**：对于 GPU / CPU 与操作系统的具体版本，请根据本地环境查阅各库官方安装文档。

### 3. 准备后端服务

- **Neo4j**  
  - 安装 Neo4j Community / Desktop；
  - 创建名为 `talent-graph` 的数据库；
  - 保持与 `config.py` 中一致的连接配置：
    - `bolt://127.0.0.1:7687`，用户 `neo4j`，密码 `Aa1278389701`（生产环境请修改并同步到 `config.py`）。

- **SQLite 数据文件路径**
  - 确保 `config.py` 与 `use_openalex/db_config.py` 中的 `DB_PATH` 指向同一 SQLite 文件；
  - 若用默认配置，则目标路径为项目 `data/academic_dataset_v5.db`。

---

## 运行指南

### 1. 快速体验（已有数据与索引）

前提：你已经准备好以下内容：

- `data/academic_dataset_v5.db` 已包含 OpenAlex 与岗位数据；
- 已在 Neo4j 中构建好知识图谱（运行过 `generate_kg.py`）；
- 已构建向量 / 特征 / 协作索引与词汇领域索引；
- 已完成一次 KGAT‑AX 训练，`data/kgatax_train_data/weights` 下存在 `best_model_epoch_*.pth`。

此时仅需：

```bash
# 激活虚拟环境
cd TalentRecommendationSystem-master
source .venv/Scripts/activate  # 或对应的激活命令

# 启动 Streamlit 前端
python main.py
```

访问浏览器中的 `http://127.0.0.1:8501`，输入岗位 JD 与可选领域多选，即可获得专家推荐列表与解释信息。

### 2. 从零构建完整数据与模型管线

以下步骤为推荐的最小流水线顺序（可根据已有数据适当跳过某些阶段）：

#### 2.1 抓取岗位数据（BOSS 直聘）

```bash
cd src/infrastructure/crawler
python new_getdata_likelogin0104.py
```

- 按提示在浏览器中登录 BOSS 直聘；
-.tool_config 中已预置多个城市与岗位关键词，脚本会持续抓取高学历岗位并写入 `jobs.csv`。

#### 2.2 合并岗位数据进 SQLite

```bash
cd src/infrastructure/crawler/use_openalex
python merge_database.py
```

- 会在 `data/` 下创建或更新 `academic_dataset_v5.db`；
- 将 `jobs.csv` 内容写入 `jobs` 表，并同步岗位技能到 `vocabulary`。

#### 2.3 抓取 OpenAlex 学术数据

```bash
cd src/infrastructure/crawler/use_openalex

# 查看当前统计（可选）
python main.py --stats

# 按领域抓取，例如计算机科学领域（ID=1）
python main.py --field 1
```

- 根据网络与配额，抓取过程可能耗时较长；  
- 建议分多天按字段分批抓取；  
- 可通过 `--force` 重新爬取某个领域。

#### 2.4 构建知识图谱（Neo4j）

```bash
cd src/infrastructure/database/build_kg
python generate_kg.py
```

- 该脚本会自动：
  - 在 Neo4j 中创建索引与约束；
  - 将 SQLite 中的作者 / 论文 / 机构 / 渠道 / 词汇 / 岗位同步到图数据库；
  - 建立 Authorship 与 Job‑Skill 等关系；
  - 通过标题扫描与统计构建 HAS_TOPIC / CO_OCCURRED_WITH / SIMILAR_TO 等语义边。

#### 2.5 构建向量索引与特征索引

```bash
cd src/infrastructure/database/build_index

# 下载 SBERT 模型（如尚未下载）
python download_model.py

# 构建词汇 / Job / 摘要向量索引（可能耗时较长）
python build_vector_index.py

# 构建作者 / 机构特征索引
python build_feature_index.py

# 构建词汇领域分布索引
python build_vocab_stats_index.py

# 构建作者协作相似度索引
python build_collaborative_index.py
```

#### 2.6 生成 KGAT‑AX 训练数据与图索引

```bash
cd src/infrastructure/database/kgat_ax

# 1) 生成训练样本与加权 KG 文本
python generate_training_data.py

# 2) 将 kg_final.txt 导入 SQLite 并建立索引
python build_kg_index.py
```

#### 2.7 训练 KGAT‑AX 模型

```bash
cd src/infrastructure/database/kgat_ax

# 方式一：手动启动
python trainer.py --n_epoch 50 --evaluate_every 5

# 方式二：使用流水线脚本（相当于顺序调用 generate_training_data / build_kg_index / trainer）
python pipeline.py
```

训练完成后，在 `data/kgatax_train_data/weights` 下会生成多个 `.pth` 权重文件，`TotalCore` 会自动选择最新 / 最优权重进行加载。

#### 2.8 启动在线推荐服务

完成以上步骤后，即可回到项目根目录启动前端：

```bash
cd ../../../../..
python main.py
```

---

## 开发与扩展建议

- **扩展召回路径**  
  - 在 `src/core/recall/` 下新增自定义召回路径（例如基于兴趣标签或简历相似度），再在 `TotalRecallSystem` 中注册该路径并参与 RRF 融合。

- **优化精排逻辑**  
  - 可根据业务场景调整 `RankingEngine` 中 KGAT 分数与召回分数的权重（目前为 0.4 / 0.6），或在 `RankScorer` 中加入更多特征（如机构声望）。

- **前端重构**  
  - 将当前 Streamlit 流程抽象为 REST API（例如使用 FastAPI / Flask），再通过已有的 Vue3 + Element Plus 前端壳（`index.html + vite.config.ts`）构建完整运营后台。

- **生产化部署**  
  - 建议将 Neo4j / SQLite / 模型服务解耦至独立容器或服务进程，并引入日志采集、监控与告警体系，保障推荐服务的可观测性与可扩展性。

