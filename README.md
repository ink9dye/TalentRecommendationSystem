## 项目简介

**本节回答：**
- 项目是做什么的？整体架构是什么（多路召回 + 精排 + 解释）？
- 数据从哪来、结果怎么呈现？

`TalentRecommendationSystem` 是一个面向 **高端科技人才智能推荐** 的完整工程项目，采用「**多路召回（Vector / Label / Collaboration）+ 深度精排（KGAT‑AX）+ 知识图谱解释**」的两阶段架构。  
项目从 OpenAlex 学术数据与 BOSS 直聘岗位数据出发，构建本地 SQLite 数据库与 Neo4j 知识图谱，离线训练 KGAT‑AX 排序模型，线上通过 Streamlit / Vue 前端提供交互式专家推荐与可视化解释能力。

---

## 目录总览

- **[目录树](#目录树)**
- **[整体架构与数据流](#整体架构与数据流)**
- **[知识图谱构建详解](#知识图谱构建详解)**
  - [6. 数据表与图模型速查](#6-数据表与图模型速查)
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
  - [0. 配置项速查（config.py）](#0-配置项速查configpy)
- **[运行指南](#运行指南)**
  - [1. 快速体验（已有数据与索引）](#1-快速体验已有数据与索引)
  - [2. 从零构建完整数据与模型管线](#2-从零构建完整数据与模型管线)
- **[开发与扩展建议](#开发与扩展建议)**
- **[后续规划](#后续规划)**

---

## 目录树

**本节回答：**
- 核心业务代码分布在哪些目录？某个脚本/模块在树里大概哪一层？
- 想改召回、精排、图谱、索引时，应去哪个路径找对应文件？

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
    │   │   ├── label_path.py       # 标签路入口：编排五阶段流水线（label_pipeline + label_means）
    │   │   ├── label_path_pre.py   # 标签路旧版/备用实现（内部或历史参考）
    │   │   ├── works_to_authors.py # 论文级得分聚合为作者级，供标签路 Stage5 使用
    │   │   ├── diagnose_embedding_neighbors.py # 嵌入邻居诊断工具（开发与调试用）
    │   │   ├── collaboration_path.py # 协同路：基于本地协作索引的协同召回
    │   │   ├── label_means/        # 标签路子模块：基础设施、锚点、扩展、词/论文打分与调试
    │   │   │   ├── infra.py        # 资源管理：Neo4j / Faiss / vocab_stats.db / 簇中心等
    │   │   │   ├── base.py
    │   │   │   ├── label_anchors.py
    │   │   │   ├── label_expansion.py
    │   │   │   ├── term_scoring.py
    │   │   │   ├── paper_scoring.py
    │   │   │   ├── simple_factors.py
    │   │   │   ├── advanced_metrics.py
    │   │   │   └── label_debug_cli.py
    │   │   └── label_pipeline/      # 标签路五阶段流水线实现
    │   │       ├── stage1_domain_anchors.py   # 领域探测与工业侧锚点提取
    │   │       ├── stage2_expansion.py       # 学术落点与学术侧补充：先 anchor→academic landing（跨类型 SIMILAR_TO，可选 JD 向量），再 dense/簇/共现补充
    │   │       ├── stage3_term_filtering.py  # 词过滤与权重（IDF、纯度、语义守门等）
    │   │       ├── stage4_paper_recall.py    # 论文召回与贡献度
    │   │       └── stage5_author_rank.py     # 作者排序与截断
    │   └── ranking/
    │       ├── ranking_engine.py   # 精排引擎：融合召回与 KGAT 打分
    │       ├── rank_scorer.py      # KGAT 子空间打分逻辑
    │       └── rank_explainer.py   # 推荐解释生成（Neo4j + 注意力权重）
    ├── interface/
    │   └── app.py                  # 主 Streamlit 前端（JD 输入 / 结果展示）
    ├── utils/
    │   ├── domain_config.py       # 领域常量唯一数据源（17 领域、DOMAIN_MAP、decay_rate 等）
    │   ├── domain_detector.py     # 领域探测器：Query → Job 索引 + Neo4j 推断领域集合
    │   ├── domain_utils.py        # 领域 ID 解析 / 正则构造 / 交集判定（DomainProcessor）
    │   ├── time_features.py       # 论文与作者时序权重（recency、time_decay 等）
    │   └── tools.py               # 技能清洗、JD 句式/停用词、get_decay_rate_for_domains 等
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
    │   │   │   ├── build_vocab_stats_index.py  # 词汇统计索引（领域分布 + 共现 + 三级领域 vocabulary_topic_stats）
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
    │           ├── alex_utils.py          # ID 生成 / 字段清洗等通用工具
    │           └── db_config.py      # OpenAlex 抓取配置（EMAIL / FIELDS / 路径）
    └── infrastructure/
        └── crawler/
            └── use_openalex/
                └── some_tool/        # 若干一键修复 / 维护脚本（补 DOI、刷新指标、三级领域导出等）
                    ├── backfill_vocabulary_topic.py   # 从 topic 快照回填主库 vocabulary_topic
                    ├── export_vocabulary_topic_index.py  # 方案 B：导出 vocabulary_topic_index.json 供 vocab_stats 三级领域用
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

**本节回答：**
- 从原始数据到最终推荐，整条链路分哪几步？
- 岗位数据、学术数据、图谱、索引、训练、召回、精排各在什么阶段？
- 新人上手应按什么顺序读文档或跑脚本？

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

**本节回答：**
- 知识图谱里有哪些节点和边？边权重怎么来的？
- 构建顺序是什么？先同步节点还是先建语义边？
- 查库、写 Cypher、扩展 ETL 时去哪张表、哪个脚本找定义？

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

### 6. 数据表与图模型速查

以下为查库、写查询与扩展 ETL 时的快速参考；完整建表与索引见 `src/infrastructure/crawler/use_openalex/database.py`、`merge_database.py`、`build_vocab_stats_index.py`。表结构可与项目内 `data/attribute.py` 等脚本输出对照。

#### 6.1 知识图谱结构总览（Neo4j）

**节点**

| 节点标签 | 主键 | 主要属性 | 来源表 |
|----------|------|----------|--------|
| Author | id (author_id) | name, h_index, works_count, citations | authors |
| Work | id (work_id) | title, name, year, citations, domain_ids | works |
| Vocabulary | id (voc_id) | term, type（即 entity_type） | vocabulary |
| Institution | id (inst_id) | name, works_count, citations | institutions |
| Source | id (source_id) | name, type, works_count, citations | sources |
| Job | id (securityId) | name, skills, description, domain_ids | jobs |

**边**

| 边类型 | 方向 | 边属性/权重 | 含义 |
|--------|------|-------------|------|
| AUTHORED | Author→Work | pos_index, pub_year, is_corresponding, **pos_weight** | 署名顺序、是否通讯、时间衰减+署名加成 |
| PRODUCED_BY | Work→Institution | 无 | 机构产出论文 |
| PUBLISHED_IN | Work→Source | 无 | 发表渠道 |
| HAS_TOPIC | Work→Vocabulary | 无 | 论文–主题/关键词 |
| REQUIRE_SKILL | Job→Vocabulary | 无 | 岗位–技能 |
| CO_OCCURRED_WITH | Vocabulary–Vocabulary | **weight** (int) | 共现论文数 |
| SIMILAR_TO | Vocabulary→Vocabulary | **score** (float) | SBERT 语义相似度（仅跨类型） |

（边权重与建边逻辑见上文「3. 边类型、边属性与边权重」。）

#### 6.2 SQLite 主库（DB_PATH）表结构（完整字段）

以下字段与类型以主库实际建表为准，可与 `data/attribute.py` 输出对照。

**works**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| work_id | TEXT | 是 |
| doi | TEXT |  |
| title | TEXT |  |
| year | INTEGER |  |
| publication_date | TEXT |  |
| citation_count | INTEGER |  |
| concepts_text | TEXT |  |
| keywords_text | TEXT |  |
| type | TEXT |  |
| language | TEXT |  |
| domain_ids | TEXT |  |

**abstracts**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| work_id | TEXT | 是 |
| inverted_index | TEXT |  |
| full_text_en | TEXT |  |

**authors**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| author_id | TEXT | 是 |
| orcid | TEXT |  |
| name | TEXT |  |
| works_count | INTEGER |  |
| cited_by_count | INTEGER |  |
| h_index | INTEGER |  |
| last_known_institution_id | TEXT |  |
| last_updated | TEXT |  |

**institutions**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| inst_id | TEXT | 是 |
| name | TEXT |  |
| country | TEXT |  |
| type | TEXT |  |
| works_count | INTEGER |  |
| cited_by_count | INTEGER |  |
| last_updated | TEXT |  |

**authorships**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| ship_id | INTEGER | 是 |
| work_id | TEXT |  |
| author_id | TEXT |  |
| inst_id | TEXT |  |
| source_id | TEXT |  |
| pos_index | INTEGER |  |
| author_position | TEXT |  |
| is_corresponding | INTEGER |  |
| is_alphabetical | INTEGER |  |

**sources**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| source_id | TEXT | 是 |
| display_name | TEXT |  |
| type | TEXT |  |
| works_count | INTEGER |  |
| cited_by_count | INTEGER |  |
| last_updated | TEXT |  |

**jobs**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| securityId | TEXT | 是 |
| job_name | TEXT |  |
| salary | TEXT |  |
| skills | TEXT |  |
| description | TEXT |  |
| company | TEXT |  |
| city | TEXT |  |
| qualification | TEXT |  |
| keyword | TEXT |  |
| crawl_time | TEXT |  |
| domain_ids | TEXT |  |

**vocabulary**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| voc_id | INTEGER | 是 |
| term | TEXT |  |
| entity_type | TEXT |  |
| vector | BLOB |  |
| old_voc_id | TEXT |  |

**vocabulary_topic**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| voc_id | INTEGER | 是 |
| topic_id | TEXT |  |
| topic_display_name | TEXT |  |
| domain_id | TEXT |  |
| domain_name | TEXT |  |
| field_id | TEXT |  |
| field_name | TEXT |  |
| subfield_id | TEXT |  |
| subfield_name | TEXT |  |
| openalex_topic_url | TEXT |  |
| updated_at | TEXT |  |
| hierarchy_path | TEXT |  |

**crawl_states**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| field_id | TEXT | 是 |
| phase | INTEGER | 是 |
| cursor | TEXT |  |
| progress | INTEGER |  |
| is_completed | INTEGER |  |
| last_updated | TEXT |  |

**author_process_states**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| field_id | TEXT | 是 |
| author_id | TEXT | 是 |
| phase | INTEGER | 是 |

**work_fields**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| work_id | TEXT | 是 |
| field_id | TEXT | 是 |
| field_name | TEXT |  |

**doi_mapping**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| old_work_id | TEXT | 是 |
| status | TEXT |  |
| last_check | DATE |  |

**sync_metadata**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| task_name | TEXT | 是 |
| last_marker | TEXT |  |

**work_terms_temp**（构建共现等时的临时表，work_id + term）

| 字段名 | 类型 | 主键 |
|--------|------|------|
| work_id | TEXT |  |
| term | TEXT |  |

#### 6.3 SQLite 主库与词汇统计库速查（用途与主要字段）

**SQLite 主库（config 中 `DB_PATH`）**

| 表名 | 主要字段 | 用途 |
|------|----------|------|
| **works** | work_id, title, year, citation_count, concepts_text, keywords_text, domain_ids | 论文主表；领域过滤、共现、HAS_TOPIC 数据源 |
| **abstracts** | work_id, full_text_en | 摘要文本；向量路摘要索引 |
| **vocabulary** | voc_id, term, entity_type | 统一词表（概念/关键词/岗位技能）；REQUIRE_SKILL / HAS_TOPIC / SIMILAR_TO |
| **authors** | author_id, name, h_index, works_count, cited_by_count | 作者画像；精排 AX、展示 |
| **authorships** | ship_id, work_id, author_id, inst_id, source_id, pos_index, is_corresponding, is_alphabetical | 署名关系；拓扑、AUTHORED 权重、协作索引 |
| **institutions** | inst_id, name, works_count, cited_by_count | 机构；PRODUCED_BY、特征索引 |
| **sources** | source_id, display_name, type, works_count, cited_by_count | 发表渠道；PUBLISHED_IN |
| **jobs** | securityId, job_name, skills, description, crawl_time, domain_ids | 岗位；标签路锚点、Job 向量索引；domain_ids 可由后续脚本回填 |
| **vocabulary_topic** | voc_id, topic_id, field_id, subfield_id, hierarchy_path 等 | 词汇与 OpenAlex 三级领域对应；供导出 vocabulary_topic_index.json |
| **sync_metadata** | task_name, last_marker | 增量同步断点 |

**SQLite 词汇统计库（`VOCAB_STATS_DB_PATH`，如 vocab_stats.db）**

| 表名 | 主要字段 | 用途 |
|------|----------|------|
| **vocabulary_domain_stats** | voc_id, work_count, domain_span, domain_dist, updated_at | 词关联论文数、领域跨度、各领域分布；标签路领域纯度/熔断 |
| **vocabulary_cooccurrence** | term_a, term_b, freq | 词对共现频次；与 KG CO_OCCURRED_WITH 一致；标签路 cooc_span/cooc_purity |
| **vocabulary_topic_stats** | voc_id, field_id, field_name, subfield_id, subfield_name, topic_id, topic_display_name, field_dist, subfield_dist, topic_dist, source, updated_at | 三级领域：有标签词由 JSON 直接填，无标签词由共现算 field/subfield/topic 占比补全；source 为 direct / cooc / direct+cooc |

**vocabulary_domain_stats 完整字段**

| 字段名 | 类型 | 主键 |
|--------|------|------|
| voc_id | INTEGER | 是 |
| work_count | INTEGER |  |
| domain_span | INTEGER |  |
| domain_dist | TEXT |  |
| updated_at | TIMESTAMP |  |

#### 6.4 Neo4j 节点与边速查（与 6.1 一致，便于本节约尾引用）

| 节点标签 | 主键 | 主要属性 |
|----------|------|----------|
| Author | id | name, h_index, works_count, citations |
| Work | id | title, name, year, citations, domain_ids |
| Vocabulary | id | term, type |
| Institution | id | name, works_count, citations |
| Source | id | name, type, works_count, citations |
| Job | id | name, skills, description, domain_ids |

| 边类型 | 方向 | 边主要属性 |
|--------|------|------------|
| AUTHORED | Author→Work | pos_index, pub_year, is_corresponding, **pos_weight** |
| PRODUCED_BY | Work→Institution | 无 |
| PUBLISHED_IN | Work→Source | 无 |
| HAS_TOPIC | Work→Vocabulary | 无 |
| REQUIRE_SKILL | Job→Vocabulary | 无 |
| CO_OCCURRED_WITH | Vocabulary–Vocabulary | **weight** |
| SIMILAR_TO | Vocabulary→Vocabulary | **score** |

（边权重含义见上文「3. 边类型、边属性与边权重」。）

---

## 三路召回与文本转向量详解

**本节回答：**
- 系统如何从 JD 召回候选专家？
- 向量路、标签路、协同路分别负责什么？三路结果如何合并？
- 哪些问题留给精排解决？文本转向量在召回里扮演什么角色？

本节从**核心目的**与**为达成目的所使用的方法**两方面，说明「文本转向量」以及「向量路 / 标签路 / 协同路」三路召回的详细过程。对应代码：`src/core/recall/input_to_vector.py`、`vector_path.py`、`label_path.py`、`collaboration_path.py`、`total_recall.py`。

### 1. 文本转向量（input_to_vector.py — QueryEncoder）

**核心目的**  
将岗位描述（JD）转成与离线索引**同一向量空间**的固定维向量，供向量路在摘要/Job 索引上做 Faiss 检索，以及标签路做「JD 与学术词」的语义守门（余弦相似度）。同时通过**动态自共振**，让 JD 里与业务强相关的词在向量里占更大权重，提升检索相关性。

| 维度 | 说明 |
|------|------|
| **输入** | JD 文本、可选领域（用于追加 Area 提示）、动态词库来源（jobs.skills、vocabulary.term） |
| **依赖** | SBERT 模型（config）、OpenVINO 导出模型、vocabulary / jobs 表（建动态词库） |
| **输出** | 固定维向量（L2 归一化）、编码耗时 |
| **核心中间产物** | hardcore_lexicon、自共振增强后的文本、tokenized input、mean pooling 向量 |
| **主要问题** | 未做缩写展开时 JD 与索引 term 不一致；长文本截断；动态词库与离线索引词集需一致 |

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

| 维度 | 说明 |
|------|------|
| **输入** | query_vector、target_domains（可选）、recall_limit |
| **依赖** | 摘要 Faiss 索引（ABSTRACT_INDEX_PATH）、id_map、主库 works（domain_ids）、authorships |
| **输出** | 作者 ID 列表（按「最相关论文」顺序）、耗时 |
| **核心中间产物** | 500 篇相似 work_id、领域过滤后的 work 列表、work→author 映射与排序 |
| **主要问题** | 纯语义可能召回泛领域论文；依赖摘要质量；无技能级对齐 |

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
走「**岗位技能 → 学术词 → 论文 → 作者**」的图谱路径，用知识图谱 + 多维度打分，找出与 JD 技能要求在语义和共现上都对齐的学术词，再只保留领域垂直、论文质量高的论文与作者；强调硬技能/概念对齐，并抑制泛词、万金油论文和弱相关论文。实现上采用**五阶段流水线**（`label_pipeline`）与**标签子模块**（`label_means`），基础设施与索引（Neo4j、Faiss、vocab_stats.db、簇中心等）由 `label_means.infra` 统一加载与管理。

| 维度 | 说明 |
|------|------|
| **输入** | JD 文本、query_vector、领域约束（domain_ids）、skills 清洗结果（来自 Job / 总控） |
| **依赖** | **基础设施**：Neo4j、vocab_stats.db（vocabulary_domain_stats、vocabulary_cooccurrence、vocabulary_cooc_domain_ratio 等）、Faiss（Job、Vocabulary）+ 词汇向量 .npy、簇索引（cluster_members、voc_to_clusters、cluster_centroids）。**配置**：config（DB_PATH、VOCAB_P95_PAPER_COUNT、SIMILAR_TO_TOP_K/MIN_SCORE 等）、domain_config（DOMAIN_MAP、decay）、LabelRecallPath 类常量。**工具**：DomainDetector、extract_skills、DomainProcessor、time_features、get_decay_rate_for_domains。 |
| **输出** | 候选作者 ID 列表、RecallDebugInfo（领域探测、锚点、扩展、词权重、论文/作者规模等） |
| **核心中间产物** | **Stage1**：Stage1Result、anchor_skills、_jd_cleaned_terms、job_ids、regex_str。**Stage2A**：primary_landings（含 domain_fit）。**Stage2B**：raw_candidates（term_role、domain_fit、parent_anchor、parent_primary）。**Stage3**：score_map、term_map、idf_map、term_role_map、term_source_map、parent_anchor_map、parent_primary_map、tag_purity_debug；label_path 据此构建 term_confidence_map。**Stage4**：author_papers_list（按作者聚合的论文及 hits/weight/title/year/domains）。**Stage5**：paper_map、每篇 contribution（含 term_confidence）、作者聚合分、最终 author_id 列表。 |
| **主要问题** | 缩写歧义、任务词泛化、领域漂移、万金油词与弱相关论文需靠熔断与权重抑制 |

#### 3.1 标签路数据流与阶段总览

| 阶段 | 输入 | 输出 | 依赖的 label_means / 外部 |
|------|------|------|---------------------------|
| **Stage1** | query_vector, query_text, domain_id | active_domains, domain_regex, anchor_skills, Stage1Result | DomainDetector；label_anchors.extract_anchor_skills、supplement_anchors_from_jd_vector；tools.extract_skills |
| **Stage2A** | prepared_anchors, active_domain_set, query_vector, 可选 jd_field/subfield/topic_ids | primary_landings（含 domain_fit） | retrieve_academic_term_by_similar_to；_compute_domain_fit；collect_landing_candidates；select_primary_academic_landings（domain_fit≥DOMAIN_FIT_MIN_PRIMARY 才可做 primary） |
| **Stage2B** | primary_landings（仅高可信参与扩散）, active_domain_set, jd_*_ids | raw_candidates（含 term_role、domain_fit、parent_primary） | expand_from_vocab_dense_neighbors/cluster_members/cooccurrence_support（domain_fit/domain_span 门控）；merge_primary_and_support_terms；_expanded_to_raw_candidates |
| **Stage3** | raw_candidates（含 term_role、identity_score、domain_fit、parent_anchor、parent_primary）, query_vector, anchor_vids | score_map, term_map, idf_map, term_role_map, term_source_map, parent_anchor_map, parent_primary_map | passes_identity_gate；passes_topic_consistency；score_term_expansion_quality；compose_term_final_score（六因子：base×source_weight×domain_gate×task_consistency×role_penalty×expansion_penalty）；term_scoring |
| **Stage4** | vocab_ids, regex_str | author_papers_list（按作者聚合的论文及 hits） | Neo4j Cypher（HAS_TOPIC + 3% 熔断）；paper_scoring 在 Stage5 用 |
| **Stage5** | author_papers_list, score_map, term_map, debug_1（含 term_role_map、term_confidence_map） | author_id 列表、last_debug_info（含 author_evidence_by_term_role） | paper_scoring.compute_contribution（paper_term_contrib=term_weight×term_confidence×paper_match_strength；护栏 5）；accumulate_author_scores；aggregate_author_evidence_by_term_role；AUTHOR_BEST_PAPER_MIN_RATIO |

#### 3.2 各阶段目的、关键步骤与关键数据（概要）

- **Stage1 领域与锚点**：得到 active_domains、domain_regex、工业侧锚点（+ 可选 JD 向量补充）；无锚点时后续短路。
- **Stage2A 学术落点**：锚点 → 跨类型 SIMILAR_TO + 可选 JD 向量 → primary_landings（每锚点 1～3 个主落点）。
- **Stage2B 学术侧补充**：围绕 primary 做 dense 近邻、簇内、共现扩展，输出 List[ExpandedTermCandidate]（term_role：primary / dense_expansion / cluster_expansion / cooc_expansion）。
- **Stage3 词过滤与权重**：身份闸门 → topic 闸门 → quality 分 → compose 最终分 → 质量闸门；输出 score_map、term_map、idf_map、tag_purity_debug。
- **Stage4 论文召回**：Neo4j HAS_TOPIC + 3% 熔断 + 可选 domain 正则 → author_papers_list（按作者聚合，单次上限 2000 篇）。
- **Stage5 作者排序**：paper_scoring.compute_contribution（含 term_role 权重与护栏 5）→ accumulate_author_scores → AUTHOR_BEST_PAPER_MIN_RATIO 过滤 → 排序截断。

#### 3.2.1 各阶段详细说明（作用、思路、逻辑、输入输出、公式、表与索引）

以下按阶段给出：**作用**、**思路**、**逻辑流程**、**输入/输出参数（名字与含义）**、**主要公式**、**调用的表或知识图谱**。实现文件：`label_pipeline/stage1_domain_anchors.py`、`stage2_expansion.py`、`stage3_term_filtering.py`、`stage4_paper_recall.py`、`stage5_author_rank.py`，以及 `label_means` 下 `label_anchors`、`label_expansion`、`term_scoring`、`paper_scoring` 等。

---

**Stage1：领域与锚点**

| 项 | 说明 |
|----|------|
| **作用** | 确定本次召回的**活跃领域集合**（active_domain_set）与 **Neo4j 领域正则**（regex_str），并产出**工业侧锚点词**（anchor_skills：岗位技能 + 可选 JD 向量补充），供 Stage2 做「技能→学术词」落点。无锚点时整条标签路短路。 |
| **思路** | 先由岗位/向量推断「本 query 属于哪些领域」，再在这些领域对应的岗位上抽取「高频且非泛词」的技能作为锚点；可选地用 JD 文本在词汇向量索引上做 Top-K 检索，把与 JD 语义贴近的学术词补进锚点；为每个锚点打 anchor_type，供 Stage2/3 **做门槛与扩散权限控制**。 |
| **逻辑流程** | ① 预清洗 JD 技能（`extract_skills`）→ 写入 `recall._jd_cleaned_terms`；② 领域与岗位探测：若有 `DomainDetector` 则 `detect_from_jobs`（Job Faiss + Neo4j 统计 domain_ids）得 job_ids、inferred_domains、dominance，否则回退 `_detect_domain_context`；③ 锚点提取：`label_anchors.extract_anchor_skills`（Neo4j REQUIRE_SKILL + **3% 熔断** cov_j ≥ ANCHOR_MELT_COV_J + backbone_score 排序 + TopN）；④ 可选 `supplement_anchors_from_jd_vector`（JD 编码后在 Vocabulary Faiss 上 Top-K）；⑤ `label_anchors.classify_anchor_type` 打 anchor_type；⑥ 若指定 domain_id 则优先用，否则由 inferred_domains + 向量语义排序取 ACTIVE_DOMAINS_TOP_K；⑦ `DomainProcessor.build_neo4j_regex(active_domain_set)` 得到 regex_str；⑧ 写入 `recall._last_stage1_result`（Stage1Result）。 |
| **输入参数（名字与含义）** | **query_vector**：JD 编码向量，用于领域语义排序与锚点 backbone 中的 sim；**query_text** / **semantic_query_text**：JD 文本，供技能清洗与 JD 向量补充；**domain_id**：可选，指定则优先作 active_domains。 |
| **输出参数（名字与含义）** | **active_domain_set**：Set[int]，参与召回的领域 ID；**regex_str**：str，Neo4j 领域正则，供 Stage4 过滤 Work.domain_ids；**anchor_skills**：Dict[tid_str, { term, anchor_type }]，工业锚点，key 为 Vocabulary id；**debug_1**：dict，含 job_ids、job_previews、anchor_debug、dominance、industrial_kws、anchor_skills 等，供 Stage5 与调试。 |
| **主要公式** | 锚点熔断：丢弃 **cov_j ≥ ANCHOR_MELT_COV_J**（如 0.03）的词。**backbone_score**（排序用）：`BACKBONE_W_JOB_FREQ*log(1+job_freq) + BACKBONE_W_COV_J*(1-cov_j) + BACKBONE_W_SPAN*(1/(1+span)) + BACKBONE_W_IN_JD*in_jd + BACKBONE_W_TASK_LIKE*task_like + BACKBONE_W_SIM*sim`；保底词（in_jd 且 task_like）得分不低于 BACKBONE_FLOOR_FOR_BAODI。 |
| **调用的表/知识图谱** | **Neo4j**：**Job** 节点（id, skills, domain_ids）、**Vocabulary**、边 **REQUIRE_SKILL**（Job→Vocabulary）；**Faiss**：Job 索引（领域探测）、Vocabulary 索引（JD 向量补充）；**SQLite**：`vocabulary_domain_stats`（voc_id, domain_span，用于锚点候选 span）；**外部**：`industrial_abbr_expansion.json`（缩写 key，anchor_type=acronym）。 |

**锚点分型后的行为差异（Stage2A / Stage2B 约束）**

`classify_anchor_type` 打出类型后，不同类型**不改变 Stage2A 的候选来源**，当前版本统一仍由**跨类型 SIMILAR_TO** 负责 primary landing；不同类型主要影响：

- **min_identity**
- **min_domain_fit**
- **top_k**
- **是否允许进入 Stage2B 扩散**
- **扩散时是否采用更保守门槛**

当前建议按现有代码类型体系约束：

| **anchor_type** | **约束策略** |
|-----------------|--------------|
| **canonical_academic_like** | 使用默认 identity/domain_fit 门槛，可正常参与 primary 与后续扩散 |
| **acronym** | 提高 min_identity；若形成 primary，优先只保留少量 high-confidence primary；扩散更保守 |
| **generic_task_term** | 提高 min_identity 与 min_domain_fit；无 primary 时仅跳过该锚点，不短路整路；扩散最保守 |
| **application_term** | 使用默认或略保守门槛；可参与 primary，但 Stage2B 扩散需满足更高 high-confidence 条件 |
| **unknown** | 按保守策略处理；若 primary 质量不足，则不进入扩散 |

*说明*：当前版本不引入额外候选源分流，不使用词面黑/白名单；锚点类型的职责是**控制落点保守程度与扩散权限**，而不是切换「exact / alias / bridge_dict / dense」这类候选来源。

---

**Stage2A：学术落点（Academic Landing）**

| 项 | 说明 |
|----|------|
| **作用** | 将工业侧锚点**对齐到学术主落点**（primary），不做大范围扩展；对每个候选算 **domain_fit**，仅 **domain_fit ≥ DOMAIN_FIT_MIN_PRIMARY** 的候选才允许做 primary；产出每锚点 1～2 个 primary_landings，供 Stage2B 围绕其做 dense / 簇 / 共现扩展。**高歧义锚点**（如 acronym、generic_task_term）采用更高 identity 与 domain_fit 门槛；**本锚点无 primary 时仅跳过该锚点，不短路整路**。 |
| **候选来源（当前版本）** | 当前版本**仅一条**候选来源：**跨类型 SIMILAR_TO**。即在 Neo4j 中通过 `(锚点 Vocabulary)-[SIMILAR_TO]->(v_rel:Vocabulary)` 获取候选，约束 `v_rel.type IN ['concept','keyword']`，并按 top_k、min_score 过滤。候选逐条结合 active_domain_set 与可选 jd_field/subfield/topic_ids 做 `_term_in_active_domains` 过滤，并计算 domain_fit。*说明*：Stage1 的 `supplement_anchors_from_jd_vector` 仅用于「补锚点」，**不参与 Stage2A 候选合并**；若未来扩展 Stage2A 候选源，再以独立 source 并轨，不影响当前主流程。 |
| **思路** | 图中 SIMILAR_TO 为**跨类型**（如 industry→concept / keyword），且边本身基于**清洗后的文本、缩写扩写与向量相似**构建，因此当前版本以「锚点 → SIMILAR_TO → 学术词」作为**唯一** primary landing 通道。随后结合激活领域与三层领域计算 domain_fit，并按 anchor_type 调整 primary 门槛，以降低高歧义锚点的误落点风险。 |
| **逻辑流程** | ① anchor_skills 转为 **PreparedAnchor** 列表（含 **anchor_type**）；② 对每个 anchor 调用 `collect_landing_candidates`（内部仅 `retrieve_academic_term_by_similar_to`，再做领域过滤）；③ 对每个候选计算 `_compute_domain_fit`；④ `score_academic_identity`；⑤ **根据 anchor_type 计算本锚点的 min_identity 与 min_domain_fit**；⑥ `select_primary_academic_landings`（domain_fit 门控 + identity 门控）→ **primary_landings**；⑦ 若本锚点 primary 数为 0，则 continue（不短路整路）。 |
| **输入参数（名字与含义）** | **prepared_anchors**：Stage1 的 anchor_skills 转成的 PreparedAnchor 列表（含 **anchor_type**）；**active_domain_set**：Set[int]；**query_vector**、**query_text**；**jd_field_ids** / **jd_subfield_ids** / **jd_topic_ids**：可选，三层领域 ID 集合，用于 domain_fit 与过滤。 |
| **输出参数（名字与含义）** | 本阶段内部产出 **primary_landings**：List[PrimaryLanding]，每项含 vid、term、identity_score、source、anchor_vid、anchor_term、**domain_fit**；Stage2 整体对外输出为 **raw_candidates**（见 Stage2B），其中 primary 来源于此。 |
| **主要公式** | **domain_fit** = 0.4×domain_overlap + 0.3×field_overlap + 0.2×subfield_overlap + 0.1×topic_overlap（无表时对应 overlap 视为 1.0）。**primary 门控**：`domain_fit ≥ min_domain_fit` 且 `identity_score ≥ min_identity`。其中普通锚点默认 min_identity = PRIMARY_MIN_IDENTITY，高歧义锚点（如 acronym、generic_task_term）使用更高 PRIMARY_MIN_IDENTITY_HIGH_AMBIGUITY；min_domain_fit 也可按 anchor_type 做更保守设置。**identity_score 在当前版本中等于 SIMILAR_TO 边权**。 |
| **高歧义锚点规则** | **定义**：当前主要指 anchor_type ∈ { acronym, generic_task_term }。**规则**：① 做 primary 时采用更高 identity 门槛；② 可配更高 domain_fit 门槛；③ 本锚点若 primary 数为 0，仅跳过该锚点，不视为整路失败；④ 日志打标 `[高歧义]` 便于排查。 |
| **调用的表/知识图谱** | **Neo4j**：**Vocabulary** 节点、**(Vocabulary)-[SIMILAR_TO]->(Vocabulary)**（跨类型，边权 sim_score）；**SQLite**：`vocabulary_domain_stats`（domain_dist）、`vocabulary_topic_stats`（field_id/subfield_id/topic_id 及 *_dist）；**Faiss**：当前 Stage2A 不直接使用。 |

---

**Stage2B：学术侧补充（限制跨域扩展）**

| 项 | 说明 |
|----|------|
| **作用** | 仅围绕**高可信 primary** 做 dense / cluster / cooc 扩展；对扩展词做**领域门控**：domain_fit < DOMAIN_FIT_MIN_EXPANSION 或 domain_span > DOMAIN_SPAN_MAX_EXPANSION 的**不进最终词池**；合并 primary + 扩展为 List[ExpandedTermCandidate]，带 term_role、domain_fit、parent_primary，再转为 raw_candidates 供 Stage3。 |
| **思路** | 不再用 SIMILAR_TO 做学术→学术。仅用：① 词汇向量索引上 primary 的**学术近邻**（dense）；② 簇成员（cluster）；③ 共现表高支持度词（cooc）。扩展前/合并前均做 **domain_fit**、**domain_span**、**topic_align** 检查，跨域或主题发散过大的词直接丢弃；合并时为每词补全 degree_w、topic_align、domain_fit、parent_primary。 |
| **高可信 primary 定义** | **primary_high_confidence** 由**结构化条件共同决定，而非词面规则**：通常要求 identity_score ≥ PRIMARY_HIGH_CONFIDENCE_THRESHOLD、domain_fit ≥ DOMAIN_FIT_HIGH_CONFIDENCE、source ∈ TRUSTED_SOURCE_TYPES_FOR_DIFFUSION，并且在**可用时**满足较保守的 domain_span / topic_align 条件。其目标是避免「高相似但跨域或主题发散」的 primary 进入扩散。 |
| **逻辑流程** | ① 通过 **_is_high_confidence_primary** 筛出 **diffusion_primaries**：通常同时检查 identity、domain_fit、source，并在**有对应字段时**检查 domain_span、topic_align；② `expand_from_vocab_dense_neighbors`（每 primary 最多 DENSE_MAX_PER_PRIMARY，过滤 _term_in_active_domains + domain_fit/domain_span 门控，设 parent_primary）；③ `expand_from_cluster_members`（同上门控）；④ `expand_from_cooccurrence_support`（同上门控）；⑤ `merge_primary_and_support_terms`（为 primary 与各扩展补 degree_w、topic_align、domain_fit，再次过滤 domain_fit/domain_span）；⑥ `_expanded_to_raw_candidates` 转为 **raw_candidates**（含 tid、term、term_role、identity_score、source、domain_fit、parent_anchor、parent_primary 等）。 |
| **输入参数（名字与含义）** | 同 Stage2 总入口：**prepared_anchors**、**active_domain_set**、**query_vector**、**query_text**、**jd_field_ids** / **jd_subfield_ids** / **jd_topic_ids**；内部使用 **primary_landings**（及高可信子集 diffusion_primaries）。 |
| **输出参数（名字与含义）** | **raw_candidates**：List[Dict]，每项含 **tid**、**term**、**term_role**（primary / dense_expansion / cluster_expansion / cooc_expansion）、**identity_score**、**source**、**degree_w**、**domain_span**、**target_degree_w**、**degree_w_expanded**、**topic_align**、**topic_level**、**topic_confidence**、**domain_fit**、**parent_anchor**、**parent_primary** 等，供 Stage3 消费。 |
| **主要公式** | **领域门控**：扩展词仅当 `domain_fit ≥ DOMAIN_FIT_MIN_EXPANSION` 且 `domain_span ≤ DOMAIN_SPAN_MAX_EXPANSION` 才进入词池。topic_align 来自 `_attach_topic_align`（vocabulary_topic_stats 层级命中 × topic_confidence）。 |
| **调用的表/知识图谱** | **Neo4j**：**Vocabulary**、**CO_OCCURRED_WITH**（共现）；**Faiss**：Vocabulary 向量索引（dense 近邻）；**SQLite**：`vocabulary_domain_stats`（work_count、domain_span、domain_dist）、`vocabulary_cooccurrence`（term_a、term_b、freq）、`vocabulary_topic_stats`（field_id、subfield_id、topic_id、*_dist、source）；**簇数据**：cluster_members、voc_to_clusters、cluster_centroids（label_means.infra）。 |

---

**Stage3：词过滤与权重（双闸门 + 六因子最终分）**

| 项 | 说明 |
|----|------|
| **作用** | 对 raw_candidates 做**身份闸门**、**topic 闸门**、**质量分**与**六因子最终分**，产出 score_map、term_map、idf_map、term_role_map、term_source_map、parent_anchor_map、parent_primary_map，并写入 tag_purity_debug；供 Stage4 选词（final_term_ids_for_paper）、Stage5 的 term_confidence_map 与护栏 5 使用。 |
| **思路** | 先过滤再打分：通过身份闸门与 topic 闸门的词才参与 quality 与 **compose_term_final_score**；最终分采用 **final_score = base_score × source_weight × domain_gate × task_consistency × role_penalty × expansion_penalty**，把来源、三层领域、任务一致性、角色与 cluster/cooc 惩罚全部乘回排序。**整个 Stage3 以结构化门控为主，不依赖词面黑/白名单**。 |
| **逻辑流程** | ① **passes_identity_gate**：按 term_role 设阈值（primary≥0.62, dense/cluster≥0.45, cooc≥0.35）；② **passes_topic_consistency**；③ **score_term_expansion_quality**：idf 骨架 + 纯度 + 语义；④ **compose_term_final_score**：base_score（按 role 的 identity/quality 加权）× source_weight × domain_gate × task_consistency × role_penalty × expansion_penalty；⑤ final_score < FINAL_MIN_TERM_SCORE 则丢弃；⑥ 写入 score_map、term_map、idf_map、**term_role_map**、**term_source_map**、**parent_anchor_map**、**parent_primary_map**，并 append 到 tag_purity_debug（含 domain_fit、parent_anchor、parent_primary 等）。 |
| **输入参数（名字与含义）** | **raw_candidates**：Stage2 输出的字典列表，含 tid、term、term_role、identity_score、source、degree_w、domain_span、topic_align、domain_fit、parent_anchor、parent_primary 等；**query_vector**：JD 向量；**anchor_vids**：可选，用于锚点距离等门控。 |
| **输出参数（名字与含义）** | **score_map**：Dict[tid_str, float]，词最终权重；**term_map**：Dict[tid_str, term 文本]；**idf_map**：Dict[tid_str, float]，平滑 IDF；**term_role_map**：Dict[tid_str, term_role]；**term_source_map**：Dict[tid_str, source]；**parent_anchor_map** / **parent_primary_map**：Dict[tid_str, str]；**tag_purity_debug**：列表，每项含 tid、term、term_role、identity_score、quality_score、final_score、domain_fit、parent_anchor、parent_primary 等。 |
| **主要公式** | **身份闸门**：identity_score ≥ threshold(role)，primary=0.62, dense/cluster=0.45, cooc=0.35。**quality**：`idf_val * (0.5+0.5*purity) * (0.5+0.5*min(1, sim))`。**final_score** = **base_score** × **source_weight** × **domain_gate** × **task_consistency** × **role_penalty** × **expansion_penalty**（六因子见下表）。**idf**：平滑 IDF(degree_w, degree_w_expanded)。 |
| **调用的表/知识图谱** | **内存**：recall.total_work_count、raw_candidates 中各字段、all_vocab_vectors/vocab_to_idx（get_term_debug_metrics 等）；**无额外 DB 表**，仅用 Stage2 传入字段与 config 常量。 |

**Stage3 六因子：来源与默认值策略**

| 因子 | 含义 | 来源 | 默认值/策略 |
|------|------|------|-------------|
| **base_score** | 按 term_role 的 identity 与 quality 加权 | rec.term_role + rec.identity_score + rec.quality_score（quality 由 score_term_expansion_quality 计算） | primary: 0.7×identity+0.3×quality；dense/cluster: 0.4×identity+0.6×quality；cooc: 0.3×identity+0.7×quality；缺省: 0.5×identity+0.5×quality |
| **source_weight** | 按来源可信度加权 | rec.source / rec.origin + **config** | SOURCE_WEIGHT_SIMILAR_TO=1.0、SOURCE_WEIGHT_DENSE=0.85、SOURCE_WEIGHT_CLUSTER=0.75、SOURCE_WEIGHT_COOC=0.70；若未来引入其他 source（如 jd_vector），再单独配置；未匹配时取 1.0 |
| **domain_gate** | 三层领域匹配乘回 | rec.domain_fit + **config** | domain_gate = DOMAIN_GATE_MIN + (1−DOMAIN_GATE_MIN)×domain_fit；如 DOMAIN_GATE_MIN=0.5。**若 domain_fit 缺失，不取 1.0**，退化为保守默认值（如 DOMAIN_GATE_MIN 或中性值 0.75） |
| **task_consistency** | JD 语义 + 强锚点共振（**当前主要通过 topic_align 落地**） | rec.topic_align、rec.topic_level + **config** TOPIC_WEIGHT_* / TOPIC_MIN_ALIGN / TOPIC_LOW_ALIGN_PENALTY | task_consistency = 1 - topic_weight + topic_weight×topic_align；若为 expansion 且 topic_align < TOPIC_MIN_ALIGN(0.20)，再乘 TOPIC_LOW_ALIGN_PENALTY(0.50)；topic_weight 按 role 取 TOPIC_WEIGHT_PRIMARY/DENSE/CLUSTER/COOC |
| **role_penalty** | 对不同 term_role 施加保守角色权重 | rec.term_role | term_scoring._get_role_penalty：primary=1.0、dense_expansion=0.95、cluster_expansion=0.90、cooc_expansion=0.85；缺省 1.0 |
| **expansion_penalty** | 对 cluster/cooc 类扩展额外惩罚 | rec.term_role + **config** | CLUSTER_EXPANSION_PENALTY=0.75、COOC_EXPANSION_PENALTY=0.65；仅 cluster_expansion/cooc_expansion 生效，其余为 1.0 |

---

**Stage4：论文召回**

| 项 | 说明 |
|----|------|
| **作用** | 用 **final_term_ids_for_paper**（经 _select_terms_for_paper 从 score_map/term_map 精检后的至多 20 个词 ID）在 Neo4j 上沿 **HAS_TOPIC** 拉论文，词侧 **3% 熔断**，并按作者聚合，产出 author_papers_list，供 Stage5 算论文贡献度与作者排序。 |
| **思路** | 词侧 (degree_w/total_w) ≥ 0.03 的泛词在 Cypher 内过滤；论文侧可选按 domain_ids 正则过滤；单次 LIMIT 2000 篇 Work；每篇 Work 收集 hit_info（vid, idf_weight=log10(total_w/(degree_w+1))），供 Stage5 作 paper_match_strength。 |
| **逻辑流程** | ① 输入 **vocab_ids**（List[int]，即 final_term_ids_for_paper）、**regex_str**；② Cypher：MATCH (v:Vocabulary) WHERE v.id IN $v_ids，WITH 计算 degree_w，WHERE (degree_w*1.0/$total_w) < 0.03，WITH log10($total_w/(degree_w+1)) AS idf_weight；③ MATCH (v)<-[:HAS_TOPIC]-(w:Work) [可选 AND w.domain_ids =~ $regex]；④ WITH w, collect({vid, idf}) AS hit_info，LIMIT 2000；⑤ MATCH (w)<-[auth_r:AUTHORED]-(a:Author)，RETURN a.id, collect({wid, hits, weight: pos_weight, title, year, domains})；⑥ 按 aid 聚合为 **author_papers_list**。 |
| **输入参数（名字与含义）** | **vocab_ids**：List[int]，用于论文检索的词汇 ID（来自 _select_terms_for_paper）；**regex_str**：str，Neo4j 领域正则，可为空。 |
| **输出参数（名字与含义）** | **author_papers_list**：List[Dict]，每项为 { **"aid"**: str, **"papers"**: [ { **"wid"**, **"hits"**: [{ vid, idf }], **"weight"**, **"title"**, **"year"**, **"domains"** }, ... ] }，供 Stage5 展开为 paper_map 并算 contribution。 |
| **主要公式** | 熔断：`(degree_w * 1.0 / total_w) < 0.03`；命中权重 **idf_weight** = `log10(total_w / (degree_w + 1))`（即 Stage5 中的 paper_match_strength）。 |
| **调用的表/知识图谱** | **Neo4j**：**Vocabulary**（id）、**Work**（id, title, year, domain_ids）、**Author**（id）；边 **HAS_TOPIC**（Work→Vocabulary）、**AUTHORED**（Author→Work，pos_weight）。 |

---

**Stage5：作者排序**

| 项 | 说明 |
|----|------|
| **作用** | 将 author_papers_list 展开为 **paper_map**，对每篇论文算**贡献度**（含 **term_confidence**、护栏 5），再按作者聚合（accumulate_author_scores）、时序与活跃度加权、最佳论文比过滤，得到最终作者排序列表与 **last_debug_info**（含 term_role_map、term_confidence_map、author_evidence_by_term_role 等）。 |
| **思路** | 论文贡献度中，**单 term 贡献** = **term_weight × term_confidence × paper_match_strength**（即 score_map[vid] × term_confidence_map[vid] × idf_hit），再经领域纯度、综述/覆盖/紧密度/时序/簇奖励、JD 语义门得到 paper score；**护栏 5**：primary_count=0 且 supporting_count<2 的论文 score *= 0.05；作者分 = 论文贡献按署名权重拆分后取 top_k_per_author=3 累加，再乘 time_weight、recency，按 AUTHOR_BEST_PAPER_MIN_RATIO 过滤后归一化排序。 |
| **逻辑流程** | ① 从 debug_1 读取 **term_role_map**、**term_confidence_map**（由 label_path 从 term_role_map/term_source_map 构建：primary 高、similar_to 中、cluster/cooc 低），放入 context；② 展开 author_papers_list 为 paper_map（wid→{ hits, title, year, domains, authors }）；③ 对每篇 paper 调用 **paper_scoring.compute_contribution**，得到 (score, hit_terms, rank_score, term_weights, primary_count, supporting_count)；④ 护栏 5：primary_count==0 且 supporting_count<2 则 score *= 0.05；⑤ 论文得分经 tanh 压缩（95 分位 tau）；⑥ **accumulate_author_scores**（top_k_per_author=3）→ author_scores、author_top_works；⑦ 按作者时间特征 time_weight、recency 加权；⑧ AUTHOR_BEST_PAPER_MIN_RATIO 过滤；⑨ 归一化排序，构建 scored_authors；⑩ **aggregate_author_evidence_by_term_role** 写入 last_debug_info。 |
| **输入参数（名字与含义）** | **author_papers_list**：Stage4 输出；**score_map**、**term_map**：Stage3 输出；**active_domain_set**、**dominance**；**debug_1**：含 **term_role_map**、**term_confidence_map**、industrial_kws、anchor_skills、query_vector、filter_closed_loop 等。 |
| **输出参数（名字与含义）** | **author_id 列表**：按得分降序，截断 recall_limit；**last_debug_info**：active_domains、dominance、score_map、term_map、term_role_map、term_confidence_map、filter_closed_loop、top_terms_final_contrib、work_count、author_count、**author_evidence_by_term_role** 等。 |
| **主要公式** | **单 term 对论文的贡献**：**paper_term_contrib** = **term_weight** × **term_confidence** × **paper_match_strength** = score_map[vid] × term_confidence_map[vid] × idf_hit；**rank_score** = Σ paper_term_contrib。**论文得分** score = rank_score × coverage_norm × cluster_bonus × proximity_bonus × domain_coeff × time_decay × survey_decay × jd_semantic_gate（撤稿拦截、领域纯度≤0 则直接 0）。**term_confidence**：primary 0.95、dense 0.75、cluster/cooc 0.6（见 label_path._build_term_confidence_map）。**护栏 5**：primary_count>=1 或 (primary_count+supporting_count)>=2 否则 score *= 0.05。**作者拆分**：按 AUTHORED.pos_weight 分配论文贡献，取每作者 top_k_per_author=3 篇累加，再乘 time_weight、recency。 |
| **调用的表/知识图谱** | **内存**：score_map、term_map、term_role_map、term_confidence_map、paper_map、voc_to_clusters、all_vocab_vectors、vocab_to_idx（proximity、jd_semantic_gate）；**无直接 DB**。**works_to_authors**：accumulate_author_scores 用 papers 中的 wid、score、authors（含 pos_weight）。 |

#### 3.2.2 标签路必查日志（从哪一步开始跑偏）

**目的**：每次召回结束**必打**一行汇总日志，并写入 `debug_1["pipeline_checkpoints"]` / `last_debug_info["pipeline_checkpoints"]`，用于快速定位**从哪一步开始跑偏**（首个 ok=False 或计数为 0 的阶段即为异常起点）。

**实现**：`label_path._emit_label_pipeline_checkpoints(checkpoints, debug_1)`；在 `recall()` 内每阶段结束后 push 一条 checkpoint，最后统一打印并写入 debug。

**必查日志格式与含义**

| 阶段 | checkpoint 字段 | 含义 | 跑偏判断 |
|------|------------------|------|----------|
| **S1** | anchors, active_domains, ok | 锚点数量、激活领域数、本阶段是否成功 | anchors=0 或 ok=False → 阶段 1 跑偏（无锚点或领域异常） |
| **S2** | raw_candidates, ok | Stage2 合并后候选词数 | raw_candidates=0 或 ok=False → 阶段 2 跑偏（无学术词扩展结果） |
| **S3** | score_map_terms, ok | 通过双闸门后进入 score_map 的词数 | score_map_terms=0 或 ok=False → 阶段 3 跑偏（词权重阶段全滤掉） |
| **S3_select** | final_term_ids, ok | 精检后参与论文检索的词数 | final_term_ids=0 或 ok=False → 选词阶段跑偏（无词进图检索） |
| **S4** | authors, papers, ok | 命中作者数、命中论文数 | authors=0 或 papers=0 → 阶段 4 跑偏（图检索无结果） |
| **S5** | ranked_authors, ok | 最终排序后的作者数 | 通常 ok=True；若前面某步为 0，此处 ranked 也会为 0 |

**控制台输出示例**：`[Label必查] S1 anchors=20 domains=3 | S2 raw_candidates=42 | S3 score_map_terms=38 | S3_select final_term_ids=20 | S4 authors=800 papers=1200 | S5 ranked=150`  
若某步异常，会追加一行：`[Label必查] 首次异常阶段: S2（此处开始跑偏，请优先排查）`。

**使用方式**：跑完一次标签路召回后，先看该行；若某一阶段计数骤降或为 0，从该阶段对应的代码与输入（如 S2 看 anchor_skills、SIMILAR_TO 与领域过滤，S3 看 identity/topic 闸门与 final_score 阈值）排查。

**典型排查顺序**：

- 看 Stage1 的 cleaned_skills 与 anchor_skills  
- 看 Stage2A 的 primary_landings  
- 看 Stage2B 的 raw_candidates  
- 看 Stage3 的 score_map_terms  
- 看 final_term_ids_for_paper  
- 看 Stage4 的 papers / authors  
- 看 Stage5 的 author_evidence_by_term_role  

#### 3.3 label_means 子模块职责（在标签路中的角色）

| 模块 | 职责 | 主要使用阶段 |
|------|------|--------------|
| **infra** | Neo4j、Job/Vocab Faiss、vocab 向量与 voc_id↔idx、vocab_stats.db、簇成员与簇中心 | 全阶段 |
| **label_anchors** | 岗位技能清洗（clean_job_skills）、锚点提取（extract_anchor_skills）、JD 向量补充锚点（supplement_anchors_from_jd_vector） | Stage1 |
| **label_expansion** | 学术落点（跨类型 SIMILAR_TO）与学术侧补充（dense 近邻、簇、共现）的封装 | Stage2A / Stage2B |
| **term_scoring** | 词级最终权重（calculate_final_weights）、IDF/纯度/语义守门/source_credibility 等 | Stage3 |
| **paper_scoring** | 论文贡献度（compute_contribution）：撤稿、领域纯度、标签累加（按 term_role 加权）、综述降权、紧密度、时序与署名；compute_primary_term_coverage（primary/supporting 计数供护栏 5） | Stage5 |
| **simple_factors** | survey_decay_factor、coverage_norm_factor、paper_cluster_bonus、paper_jd_semantic_gate_factor 等论文侧因子 | paper_scoring 与词/论文打分 |
| **advanced_metrics** | 共现/共鸣相关（term_resonance、cooc_span_penalty、cooc_purity_bonus 等，部分当前为占位 1.0） | Stage2/Stage3 词权重 |
| **base** | 公共基类或工具 | 按需 |
| **label_debug_cli** | 调试 CLI，与 RecallDebugInfo 配合 | 开发与排查 |

#### 3.4 调参与排查提示

- **锚点太少/太多**：看 Stage1 的 ANCHOR_*、JD_VOCAB_TOP_K、ANCHOR_MELT_COV_J；以及 DomainDetector 的 Job Top-K、active_domains 数量。  
- **学术词偏泛或偏窄**：看 Stage2 的 SIMILAR_TO_TOP_K/MIN_SCORE、vocabulary_domain_stats 的领域过滤；Stage3 的 SEMANTIC_POWER、ANCHOR_TERM_SIM_MIN、ANCHOR_BASE/ANCHOR_GAIN。  
- **论文/作者分数异常**：看 Stage4 的 3% 熔断、domain_regex；Stage5 的 paper_scoring 各因子（综述降权、时序、署名）、AUTHOR_BEST_PAPER_MIN_RATIO；RecallDebugInfo 里各阶段统计与 last_debug_info。  
- **Stage2A 落点偏移**：优先看 anchor_type、identity_score、domain_fit、本锚点 primary 数量；不要先盯作者榜。  
- **Stage2B 噪声扩散**：优先看 primary_high_confidence 的筛选结果、domain_span、topic_align、各扩展来源数量（dense / cluster / cooc）。  
- **Stage3 全部滤空或几乎全过**：优先看 passes_identity_gate、passes_topic_consistency、FINAL_MIN_TERM_SCORE 以及 domain_fit/topic_align 是否真实透传。

**标签路可调类常量（LabelRecallPath）**

以下为 `label_path.py` 中 `LabelRecallPath` 的类常量，调参与复现时可参考。完整定义见 `src/core/recall/label_path.py`。

| 常量 | 典型值 | 含义 |
|------|--------|------|
| **DETECT_JOBS_TOP_K** | 20 | 领域探测时在 Job 索引检索的岗位数 |
| **CANDIDATE_DOMAINS_TOP_K** | 5 | 候选领域数量（按 Job 统计） |
| **ACTIVE_DOMAINS_TOP_K** | 3 | 最终参与召回的领域数 |
| **ANCHOR_JOBS_TOP_K** | 20 | 参与锚点提取的 Top 岗位数 |
| **ANCHOR_FREQ_TOP_K** | 30 | 熔断后按频次取的词数 |
| **ANCHOR_FINAL_TOP_K** | 20 | 语义重排后保留的锚点数 |
| **ANCHOR_MELT_COV_J** | 0.03 | 锚点熔断：cov_j ≥ 3% 的技能不参与 |
| **JD_VOCAB_TOP_K** | 20 | JD 向量在 vocabulary 索引上补充锚点的 Top-K |
| **ANCHOR_SIM_MIN** | 0.4 | 锚点语义重排的最小相似度，低于则丢弃 |
| **ANCHOR_MIN_JOB_FREQ** | 2 | 技能至少出现在几个命中岗位中才保留 |
| **ANCHOR_TERM_SIM_MIN** | 0.45 | 学术词与任意锚点最大余弦相似度下限，低于则降权 |
| **AUTHOR_BEST_PAPER_MIN_RATIO** | 0.05 | 作者最佳论文贡献低于全局最大此比例则不出现在排序 |
| **SEMANTIC_POWER** | 3 | 词权重中语义因子 cos_sim 的次方 |
| **ANCHOR_BASE** / **ANCHOR_GAIN** | 0.35, 0.65 | 词权重中锚点相关系数 |
| **SPAN_PENALTY_EXPONENT** | 0.35 | 领域跨度惩罚 (1+domain_span)^exp 的指数 |

---

### 4. 协同路召回（collaboration_path.py — CollaborativeRecallPath）

**核心目的**  
不依赖 JD 语义，而是给定一批「种子作者」（通常为向量路 + 标签路 Top 100 的并集），从预建好的作者协作表里找出与这些种子**合作最紧密**的作者，作为补充候选人；利用「谁和谁经常一起发论文」扩展候选池，发现同方向、同圈子的学者。

| 维度 | 说明 |
|------|------|
| **输入** | 种子作者 ID 列表（seeds）、recall_limit、timeout |
| **依赖** | scholar_collaboration 表（build_collaborative_index 产出） |
| **输出** | 作者 ID 列表（按协作分排序）、耗时 |
| **核心中间产物** | 种子对应的 (aid2, score) / (aid1, score)、聚合后的 target_id → total_score |
| **主要问题** | 种子若过偏则扩展到的圈子也偏；不利用 JD 语义，纯结构扩展 |

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

| 维度 | 说明 |
|------|------|
| **输入** | JD 文本、可选 domain_id、recall_limit、训练模式开关 |
| **依赖** | QueryEncoder、VectorPath、LabelRecallPath、CollaborativeRecallPath、config 路径与 RRF 参数 |
| **输出** | 融合后的作者 ID 列表（如 Top 500）、各作者在 v/l/c 三路的排名（rank_map）、总耗时 |
| **核心中间产物** | query_vec、v_list / l_list / c_list、seeds、RRF 融合分、最终排序列表 |
| **主要问题** | 路径权重（1.2/1.0/0.6）与 rrf_k 需与业务平衡；单次编码若失败则整次召回失败 |

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
| **标签路** | 按「岗位技能→学术词→论文→作者」+ 多维度打分 | Stage1 领域与锚点（3% 熔断、JD 补充）→ Stage2A 学术落点（跨类型 SIMILAR_TO、可选 JD 向量）→ Stage2B 学术侧补充（dense/簇/共现）→ Stage3 双闸门与两分制（身份/topic 闸门、quality、final_score）→ Stage4 论文召回 → Stage5 作者排序；依赖 `label_means` 与 `label_pipeline` 子模块；基础设施由 `label_means.infra` 管理 |
| **协同路** | 由种子作者扩展合作者 | 种子=V+L Top100、协作表双向查询、score 聚合、按总分排序 |
| **总控** | 单次编码、三路并行、统一排序 | Query 领域扩展、RRF 融合、路径权重 1.2/1.0/0.6 |

整体上：**向量路**偏语义相似度，**标签路**偏技能/概念与图谱结构，**协同路**偏合作网络；三路在总控里用 RRF 合成一份作者列表，再交给精排与解释模块使用。

---

## 索引构建详解

**本节回答：**
- 离线要建哪几类索引？各自给谁用（召回 / 精排 / 标签路）？
- 每类索引的数据从哪来、写到哪、config 里对应哪些路径？
- 想改索引结构或重建顺序时，应看哪一段？

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
  1. **Vocabulary**：从 `vocabulary(voc_id, term, entity_type)` 读取。**用于建向量的文本**：  
     - **工业词（entity_type='industry'）**：先用 `tools.normalize_skill(term)` 清洗；若清洗后为空则**不进入索引**；否则用清洗后的词做 key 查 `industrial_abbr_expansion.json`，若有缩写则拼成 `清洗名 | abbr1 | abbr2`，否则用清洗名，对该文本做 SBERT 编码。  
     - **非工业词**：不清洗，用原始 `term` 查缩写表，有则 `term | abbr1 | abbr2`，否则 `term`，对该文本编码。  
     索引中只保留 **id**（voc_id），Faiss 使用 `IndexIDMap`；产出 `vocabulary.faiss` + `vocabulary_mapping.json`。  
  2. **Job**：`jobs(securityId, job_name, description)`，拼接 `job_name + description` 再 trim → `job_description.faiss` + `job_description_mapping.json`。  
  3. **Abstract**：`abstracts(work_id, full_text_en)`，**分片**：每 1 万条一个 shard（`SHARD_SIZE=10000`），先全量 `fetchall()` 到内存避免 OFFSET 慢查，按片 encode、保存 `shard_i.npy` 与 `shard_i_ids.json`，最后 `_merge_abstract_shards` 合并成一份 `abstract.faiss` + `abstract_mapping.json`。

**产出**  
`vocabulary.faiss` + mapping、`job_description.faiss` + mapping、`abstract.faiss` + mapping（及可选 `*_vectors.npy`）。向量路、标签路、输入编码都依赖这些索引与同一 SBERT 空间。

---

### 4. 词汇统计索引（build_vocab_stats_index.py，领域分布 + 共现 + 三级领域）

**目的**  
为**标签路召回**提供「每个学术词」的领域统计：**work_count (degree_w)**、**domain_span**、**domain_dist**（用于领域纯度、3% 熔断、IDF 等）；**共现**词对频次；以及**三级领域**（field / subfield / topic）：有标签词直接填，无标签词用共现伙伴的占比补全，供后续按三级领域过滤或加权。

**依赖配置**  
- `DB_PATH`、`DATA_DIR`、`VOCAB_STATS_DB_PATH`（如 `data/build_index/vocab_stats.db`）。**不依赖 Neo4j**。

**实现方式**

- **领域分布**：从主库 `works` 的 `concepts_text`、`keywords_text` 解析 term（与共现逻辑同源），与 `vocabulary` 映射后按 work 的 `domain_ids` 聚合，用 `Counter` 得到 `domain_dist`；`work_count`、`domain_span` 写入 `vocabulary_domain_stats`。不依赖知识图谱。

- **共现表**：从主库 works 的 concepts_text/keywords_text 构建共现，写入 `vocabulary_cooccurrence`（term_a, term_b, freq），与 build_kg 的 CO_OCCURRED_WITH 一致。

- **三级领域（vocabulary_topic_stats）**：先读 `DATA_DIR/vocabulary_topic_index.json`（由方案 B 脚本 `export_vocabulary_topic_index.py` 从主库 `vocabulary_topic` 表导出）；对 JSON 中至少有一级（field_id 或 field_name）的 voc_id 直接写入 field/subfield/topic 标量，`source='direct'`。再对**未在 JSON 中出现**的 voc_id，用 `vocabulary_cooccurrence` 中有标签的共现伙伴按 freq 加权聚合，得到 field_dist、subfield_dist、topic_dist（JSON 占比），`source='cooc'`。对有标签但缺层级的词，用共现补全 *_dist 并设 `source='direct+cooc'`。

**前置**：需先运行 `backfill_vocabulary_topic` 回填主库 `vocabulary_topic`，再运行 `export_vocabulary_topic_index` 生成 `vocabulary_topic_index.json`，否则三级领域步骤会跳过。

**产出**  
SQLite 库 `vocab_stats.db` 内三张表：`vocabulary_domain_stats`（领域分布）；`vocabulary_cooccurrence`（词对共现频次）；`vocabulary_topic_stats`（三级领域：直接填 + 共现占比补全）。另含 vocabulary_domain_ratio、vocabulary_cooc_domain_ratio、vocabulary_cluster、cluster_members 等，详见脚本。

---

### 5. 与 config 的对应关系小结

| 索引 | 目的简述 | 主要方法 | config 中的路径/配置 |
|------|----------|----------|------------------------|
| **协作索引** | 预计算作者协作分，支撑协同路 | 权重(署名+时间+引用)→直接协作→间接 Bridge→Top100+双向覆盖索引 | `DB_PATH`, `COLLAB_DB_PATH` |
| **特征索引** | 作者/机构特征供 KGAT-AX 精排 | log1p + Min-Max，JSON 存 author/inst 特征 | `DB_PATH`, `FEATURE_INDEX_PATH` |
| **向量索引** | 摘要/岗位/词汇的语义检索 | 同一 SBERT、L2 归一化、HNSW 内积；摘要分片合并 | `DB_PATH`, `INDEX_DIR`, `SBERT_*`, `*_INDEX_PATH`, `*_MAP_PATH` |
| **词汇统计索引** | 领域分布 + 共现 + 三级领域 | 主库 works(concepts_text/keywords_text) → vocabulary_domain_stats；主库 → vocabulary_cooccurrence；JSON(vocabulary_topic_index) 直接填 + 共现补全 → vocabulary_topic_stats（不依赖 Neo4j） | `VOCAB_STATS_DB_PATH`, `DB_PATH`, `DATA_DIR`, `SQL_QUERIES` |

整体上：**协作索引**面向「谁和谁合作」；**特征索引**面向精排输入；**向量索引**面向三路召回里的语义与领域探测；**词汇领域索引**面向标签路里学术词的质量与领域约束。

---

## KGAT-AX 模型详解

**本节回答：**
- KGAT-AX 在系统里处于什么位置？是「主排序」还是「召回后重排」？
- 它吃什么输入、输出什么结果？训练数据从哪来？
- 当前它对指标信息（AX）的利用程度如何？评测用什么指标？

本节说明精排模块 `src/infrastructure/database/kgat_ax` 的**模型结构**、**训练数据从何而来**、**训练与评测流程**，力求在保留技术细节的前提下相对易懂。

### 1. 整体流程与角色

KGAT-AX 是「**知识图谱注意力网络 + 学术指标增强（AX）**」的排序模型，作用是对召回得到的约 500 名候选人做**精排**，产出最终 Top 100 及推荐理由。整条链路可以概括为：

1. **离线准备**：用真实岗位做查询，跑多路召回 → 得到「岗位–候选人」排序与层级 → 写成**四级梯度训练样本**（见下）；同时从 Neo4j 收割**带权图谱三元组**，并建成 SQLite 索引供训练时采样。
2. **训练**：在「User = 岗位、Item = 作者」的协同过滤（CF）信号上，叠加图谱（KG）上的关系约束；用 **AX 特征**（H-index、引用量等）对嵌入做**门控微调**，使排序在「专业对口」为主的前提下，用学术影响力做小幅调优。
3. **评测**：在**测试集**上构造 500 人候选池，用模型打分排序，算 Recall@K、NDCG@K，并用**早停**防止过拟合。

涉及的主要文件：`generate_training_data.py`（样本生成）、`build_kg_index.py`（图谱索引）、`data_loader.py`（数据加载与图构建）、`model.py`（KGAT-AX 结构）、`trainer.py`（训练与评估）、`kgat_utils/metrics.py`（指标）、`pipeline.py`（一键串联）。

| 维度 | 说明 |
|------|------|
| **输入** | 召回候选、训练样本（train.txt / test.txt）、kg_final.txt、feature_index.json、id_map.json |
| **依赖** | SQLite 图索引（kg_index.db）、训练权重、AX 特征、召回系统（生成训练样本时） |
| **输出** | 精排阶段：候选作者重排分数；训练阶段：best_model_epoch_*.pth、评测指标 |
| **核心中间产物** | CF loss、KG loss、attention aggregation、融合分数；拉普拉斯矩阵、entity/relation 嵌入 |
| **主要问题** | 指标特征（AX）利用偏弱、样本质量受召回限制、难以单独主导排序，故与召回序 40/60 融合 |

#### 1.1 KGAT-AX 模型结构与输入输出

**模型类**：`KGAT`（`model.py`），构造函数 `KGAT(args, n_users, n_total_nodes, n_relations, A_in=None)`。

**结构组件与维度（默认来自 `parser_kgat.py`）**

| 组件 | 类型 | 维度/含义 |
|------|------|-----------|
| **entity_user_embed** | nn.Embedding | (n_total_nodes, embed_dim)，全局节点嵌入；embed_dim 默认 64 |
| **relation_embed** | nn.Embedding | (n_relations, relation_dim)，关系嵌入；relation_dim 默认 64 |
| **type_embed** | nn.Embedding | (2, embed_dim)，节点类型偏置：0=Job，1=Entity（Author 等）；用 ENTITY_OFFSET 区分 |
| **aux_embed_layer** | nn.Linear | (n_aux_features, embed_dim)，学术特征→门控；n_aux_features 默认 3（h_index, cited_by_count, works_count） |
| **trans_M** | Parameter | (n_relations, embed_dim, relation_dim)，TransR 关系变换矩阵 |
| **W_a** | Parameter | (relation_dim, 1)，注意力/得分用 |
| **aggregator_layers** | ModuleList[Aggregator] | 多层图卷积；conv_dim_list 默认 [64, 32]，即 2 层；aggregation_type 默认 bi-interaction |
| **A_in** | buffer (sparse) | (n_total_nodes, n_total_nodes)，归一化加权邻接矩阵（拉普拉斯），由 DataLoader 构建 |

**关键方法**

| 方法 | 作用 |
|------|------|
| **holographic_fusion**(entity_embed, aux_info, node_ids) | 基础嵌入 × AX 门控（log1p(aux)→sigmoid×0.15+1.0）+ L2 归一化 + type_embed；训练时门控区间 [1.0, 1.15] |
| **calc_cf_embeddings**(aux_info_all) | 全图：holographic_fusion 后经多层 Aggregator(A_in) 聚合，拼接各层输出并返回 |
| **calc_cf_loss**(user_ids, pos_ids, neg_ids, aux_info_all) | BPR 损失：点积 pos_score/neg_score + L2 正则 |
| **calc_kg_loss**(h, r, pos_t, neg_t, h_aux, …) | TransR 式 h+r≈pos_t、远离 neg_t；softplus + L2 |
| **calc_cf_embeddings_subset**(node_ids, aux_info_subset) | 精排用：只对 node_ids 算嵌入（无全图卷积），门控区间 [0.8, 1.0]，加 type_embed |
| **calc_score**(user_ids, item_ids, aux_info_all) | 全图嵌入后 user 与 item 的点积得分 |
| **update_attention_batch**(h_list, t_list, r_list) | 计算边注意力，供解释模块用 |

**训练时输入**

| 来源 | 内容 |
|------|------|
| **数据目录** | `args.data_dir` / `args.data_name`（默认 `data/kgatax_train_data`） |
| **文件** | `train.txt`（四级梯度 CF）、`test.txt`、`kg_final.txt`（h r t w）、`id_map.json`（entity_to_int, total_nodes, user_count, offset）、`feature_index.json`（author/inst 特征） |
| **DataLoader 输出** | `entity_to_int` / `user_to_int`、`n_users_entities`、`n_users`、`ENTITY_OFFSET`、`n_relations`、`aux_info_all` (n_users_entities × n_aux_features)、`A_in`（稀疏）、`train_user_dict` / `test_user_dict`、`tiered_cf_dict`（fair/neutral/easy 负例）、`train_kg_dict` |
| **CF 阶段** | `user_ids`, `item_pos_ids`, `item_neg_ids`（LongTensor batch）、`aux_info_all` |
| **KG 阶段** | `h`, `r`, `pos_t`, `neg_t`（LongTensor）、`h_aux`, `pos_t_aux`, `neg_t_aux`（对应节点的 AX 特征行） |

**精排时输入**

| 输入 | 含义 |
|------|------|
| **real_job_ids** | 岗位原始 ID（如 securityId），由 RankScorer 转为 user_to_int 下标 |
| **candidate_raw_ids** | 候选人作者原始 ID，转为 entity_to_int 中 `a_{author_id}` 下标 |
| **node_ids** | 精排时子集 = [job_int_ids; item_int_ids]，即当前请求的岗位 + 候选人共 n 个节点 |
| **aux_info_subset** | (n, n_aux_features)，上述节点在 `aux_info_all` 中对应的行 |
| **输出** | `calc_cf_embeddings_subset` 得到 (n, embed_dim) 嵌入；岗位嵌入取均后与候选人嵌入点积得精排分 |

**ID 约定**

- `id_map.json`：`entity` 为原始 ID→整数；Job 占 0～ENTITY_OFFSET-1，Author/Work/Vocabulary 等占 ENTITY_OFFSET 及以上；`total_nodes`、`user_count`、`offset` 与模型 `global_max_id`、`n_users`、`ENTITY_OFFSET` 一致。

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

**本节回答：**
- 精排的输入是什么（岗位锚点、候选人、领域）？输出是什么？
- KGAT 分和召回序分如何融合？为什么是 40/60？
- 推荐理由（代表作、匹配类型）是怎么生成的？主路径、Fuzzy、Fallback 分别何时触发？

本节说明**精排阶段**的细节、流程与设计考虑，对应代码：`src/core/ranking/ranking_engine.py`、`rank_scorer.py`、`rank_explainer.py`；精排在 `TotalCore.suggest()` 中位于「语义导航 → 多路召回」之后，负责对约 500 名候选人做重排与可解释输出。

### 1. 精排在整体链路中的位置

精排由 **TotalCore.suggest()** 触发，顺序为：

1. **语义导航**：用 JD 向量在岗位 Faiss 索引上搜 Top-3 岗位，得到 `real_job_ids`（精排用的「岗位锚点」）。
2. **多路召回**：对同一 query 做向量路 + 标签路 + 协同路，RRF 融合得到约 500 人 `candidates`。
3. **精排过滤策略**：若用户指定了领域则用该领域；否则用三个锚点岗位的 `domain_ids` 并集得到 `filter_pattern`。
4. **精排引擎**：`RankingEngine.execute_rank(real_job_ids, candidates, filter_domain=filter_pattern)`，得到最终 Top 100 及解释。

即：**精排输入** = 若干岗位锚点 + 约 500 名召回候选人 + 可选领域过滤；**输出** = 排序后的 100 人及每人一条推荐理由、代表作等结构化信息。

| 维度 | 说明 |
|------|------|
| **输入** | real_job_ids（语义 Top-3 岗位）、candidates（约 500 名召回作者）、filter_domain（可选） |
| **依赖** | Neo4j、KGAT 模型与权重、DataLoader（id_map、entity_to_int、aux_info_all）、SQLite（作者画像） |
| **输出** | Top 100 作者及每人：rank、score、representative_work、recommendation_reason、metrics、collaboration、details |
| **核心中间产物** | 领域过滤后候选人、kgat_scores、recall_scores、归一化与 40/60 融合分、best_path、动态总结文案 |
| **主要问题** | 多锚点平均可能平滑掉单岗位强需求；解释依赖图谱完备性与 SIMILAR_TO 质量；Fallback 时理由较泛 |

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

**本节回答：**
- 某个功能或某段逻辑在哪个文件、哪个类/函数里？
- 根目录、core、interface、utils、infrastructure 各放什么？
- 想改召回/精排/图谱/索引时，应先打开哪几个文件？

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
  - `LabelRecallPath` 类，是 **标签路召回入口与编排**：统一调用 **label_pipeline 五阶段**（stage1_domain_anchors → stage2_expansion → stage3_term_filtering → stage4_paper_recall → stage5_author_rank）与 **label_means** 中的锚点、扩展、词/论文打分逻辑。定义 `Stage1Result`、`RecallContext`、`RecallDebugInfo` 等结构体用于阶段间状态与调试。输入输出格式不变：`recall(query_vector, domain_ids)` 仍返回 `(author_id_list, elapsed_ms)`；完整诊断链路通过 `debug_info` / `last_debug_info` 输出，便于调参与可解释性分析。
  - **`label_means/`** 子模块：`infra` 统一管理 Neo4j、Faiss、vocab_stats.db、簇中心等资源；`label_anchors`、`label_expansion` 负责锚点提取与语义扩展；`term_scoring`、`paper_scoring`、`simple_factors`、`advanced_metrics` 负责词级与论文级打分与指标；`base`、`label_debug_cli` 提供基类与调试 CLI。
  - **`label_pipeline/`** 子模块：五阶段实现分别对应 `stage1_domain_anchors.py`（领域探测与工业侧锚点）、`stage2_expansion.py`（学术落点与学术侧补充：先 anchor→academic landing（跨类型 SIMILAR_TO、可选 JD 向量），再 dense/簇/共现补充）、`stage3_term_filtering.py`（双闸门与两分制）、`stage4_paper_recall.py`（论文召回与贡献度）、`stage5_author_rank.py`（作者排序与截断）。

- **`src/core/recall/works_to_authors.py`**  
  - 将论文级得分聚合为作者级得分，供标签路 Stage5 及总召回使用；被 `label_path` 与流水线阶段调用。

- **`src/core/recall/diagnose_embedding_neighbors.py`**  
  - 嵌入邻居诊断工具，用于分析向量/标签路检索结果，便于开发与调试（如检查 Faiss 或词汇向量的近邻分布）。

- **`src/core/recall/label_path_pre.py`**  
  - 标签路旧版或备用实现，当前主入口为 `label_path.py`；保留供内部或历史参考。

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

- **`src/utils/domain_config.py`**  
  - 领域常量的**唯一数据源**：定义 17 个业务领域的 `DOMAIN_TABLE`（含 name、name_en、openalex_concept_id、decay_rate）；导出 `DOMAIN_MAP`、`NAME_TO_DOMAIN_ID`、`DOMAIN_DECAY_RATES`、`OPENALEX_FIELDS`、`NAME_EN_TO_DOMAIN_ID` 等，供 config、召回、爬虫、标签路时序衰减等使用，避免在多处重复定义。

- **`src/utils/domain_detector.py`**  
  - `DomainDetector`：对外提供「给定 Query 向量 → 推断领域 ID 集合」的接口。默认基于 Job Faiss 索引 + Neo4j 统计岗位 domain_ids 完成探测；若无图/索引则可选回退到 Label 路的 `_stage1_domain_and_anchors`。标签路 Stage1 领域探测使用本组件（或等价逻辑）。

- **`src/utils/domain_utils.py`**
  - `DomainProcessor`：对领域 ID 相关操作进行统一封装：
    - `to_set()`：将 `1|4|14`、`"1,4"`、列表等各种输入统一解析为集合；
    - `build_neo4j_regex()`：构造用于 Neo4j 的正则过滤表达式；
    - `build_python_regex()`：构造用于 Python `re` 过滤的正则对象；
    - `has_intersect()`：判断论文 `domain_ids` 与目标领域集合是否有交集，是召回与精排阶段过滤的核心工具。

- **`src/utils/time_features.py`**  
  - 论文与作者层**时序权重**：`compute_paper_recency(year)` 按论文年龄分段给出 recency（如 0–3 年 1.0，10+ 年 0.01）；`compute_author_time_features`、`compute_author_recency_by_latest` 等提供作者侧时间特征，供标签路论文贡献度、向量路等使用。

- **`src/utils/tools.py`**  
  - 技能与 JD 文本清洗：技能分隔符、JD 句式前缀/尾缀、垃圾词与停用词（`FORBIDDEN`、`JD_NOISE_PATTERN` 等）；`get_decay_rate_for_domains` 按领域 ID 返回衰减率（依赖 `domain_config.DOMAIN_DECAY_RATES`），供标签路时序衰减等使用。

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
  - `VocabStatsIndexer` 构建 vocab_stats.db 中多张表，**仅依赖主库 SQLite，不依赖 Neo4j**：（1）`vocabulary_domain_stats`：从主库 `works` 的 `concepts_text`、`keywords_text` 解析 term 并聚合 `domain_ids`，为每个词计算 work_count、领域跨度、domain_dist；（2）`vocabulary_cooccurrence`：从主库 works 计算词对共现频次 (term_a, term_b, freq)，与 KG CO_OCCURRED_WITH 一致；（3）`vocabulary_topic_stats`：**三级领域**，先用 `data/vocabulary_topic_index.json`（方案 B 导出）直接填有标签词的 field/subfield/topic，再用共现对无标签或缺层级词补全 field_dist、subfield_dist、topic_dist（百分比）。构建前需先运行 `backfill_vocabulary_topic` 与 `export_vocabulary_topic_index`（仅三级领域步骤依赖该 JSON）。

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

- **`backfill_vocabulary_topic.py`**  
  从本地 OpenAlex topic 快照（`data/topic`，JSONL）匹配 vocabulary.term，将 topic 层级（domain/field/subfield/topic）回填到主库表 **vocabulary_topic**，供后续导出与词汇索引使用。

- **`export_vocabulary_topic_index.py`**（方案 B）  
  从主库 **vocabulary_topic** 与 **vocabulary** 导出 `data/vocabulary_topic_index.json`，键为 voc_id，值为 field/subfield/topic、our_domain_id、hierarchy_path 等；供 **build_vocab_stats_index** 的 `vocabulary_topic_stats` 步骤「先 JSON 直接填、再共现补全」使用。路径使用 config 的 `DATA_DIR`、`DB_PATH`。

- **`delete_doi.py` / `migrate_ids.py` / `fix_doi_to_openalex_id.py` / `refresh_metrics.py` / `replace.py` / `single_recrawl_not_found.py` / `use_name_to_fix_workid.py`**  
  一组用于维护与修复数据库状态的小工具脚本，例如：使用名称或 DOI 重新匹配 OpenAlex ID；刷新统计指标；重新爬取缺失论文等。

---

### 基础设施：SBERT 模型工具 `src/infrastructure/database/models/sbert_model`

（已在上文 `models/sbert_model` 小节说明，这里不再赘述。）

---

### 辅助数据脚本与配置

- **`src/infrastructure/mock_data.py`**  
  - 定义少量模拟人才 (`mock_talents`) 与简化图结构 (`mock_graph`)，用于无数据库情况下快速展示前端界面效果。

---

## 技术栈综述

**本节回答：**
- 项目用了哪些语言、框架、数据库和第三方库？
- 检索、向量化、图谱、训练各依赖什么技术？便于排查环境与依赖问题。

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

**本节回答：**
- 要跑通项目需要装什么、配哪些路径？
- config.py 里和索引/标签路/数据库相关的关键项有哪些？改完配置要注意什么？

### 0. 配置项速查（config.py）

以下为与运行、索引及标签路强相关的主要配置，完整定义见项目根目录 `config.py`。

| 类别 | 配置项 | 含义 / 典型值 |
|------|--------|----------------|
| **数据库与图** | `DB_PATH` | 主库 SQLite 路径，如 `data/academic_dataset_v5.db` |
| | `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` / `NEO4J_DATABASE` | Neo4j 连接与库名（如 `talent-graph`） |
| | `VOCAB_STATS_DB_PATH` | 词汇领域统计库，如 `data/build_index/vocab_stats.db` |
| | `COLLAB_DB_PATH` | 协作索引库，如 `data/build_index/scholar_collaboration.db` |
| **向量与索引** | `INDEX_DIR` | 索引根目录，如 `data/build_index` |
| | `VOCAB_INDEX_PATH` / `VOCAB_MAP_PATH` | 词汇 Faiss 索引及 id 映射 |
| | `ABSTRACT_INDEX_PATH` / `ABSTRACT_MAP_PATH` | 摘要 Faiss 索引及 id 映射 |
| | `JOB_INDEX_PATH` / `JOB_MAP_PATH` | 岗位 Faiss 索引及 id 映射 |
| | `FEATURE_INDEX_PATH` | 作者/机构特征 JSON，如 `feature_index.json` |
| **SBERT** | `SBERT_DIR` / `SBERT_MODEL_NAME` | 模型本地目录与 HuggingFace 模型名（如 `Alibaba-NLP/gte-multilingual-base`） |
| **标签路** | `VOCAB_P95_PAPER_COUNT` | 学术词 paper_count 上限，过滤泛词（如 800） |
| | `SIMILAR_TO_TOP_K` | 每锚点沿 SIMILAR_TO 最多扩展词数（如 3） |
| | `SIMILAR_TO_MIN_SCORE` | SIMILAR_TO 边权重下限（如 0.65） |

修改上述配置后需确保：`DB_PATH` 与爬虫/合并脚本一致；Neo4j 与 KG 构建、召回使用的连接一致；索引路径与构建脚本输出一致。

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

**本节回答：**
- 已有数据和索引时，怎样最快启动并看到推荐结果？
- 从零构建时，推荐的最小流水线顺序是什么？各步在哪个目录、跑哪个脚本？

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

**本节回答：**
- 想加新召回路径、改精排权重、做前端重构或生产部署时，应从哪入手？
- 哪些扩展点与现有架构兼容、风险较小？

- **扩展召回路径**  
  - 在 `src/core/recall/` 下新增自定义召回路径（例如基于兴趣标签或简历相似度），再在 `TotalRecallSystem` 中注册该路径并参与 RRF 融合。

- **优化精排逻辑**  
  - 可根据业务场景调整 `RankingEngine` 中 KGAT 分数与召回分数的权重（目前为 0.4 / 0.6），或在 `RankScorer` 中加入更多特征（如机构声望）。

- **前端重构**  
  - 将当前 Streamlit 流程抽象为 REST API（例如使用 FastAPI / Flask），再通过已有的 Vue3 + Element Plus 前端壳（`index.html + vite.config.ts`）构建完整运营后台。

- **生产化部署**  
  - 建议将 Neo4j / SQLite / 模型服务解耦至独立容器或服务进程，并引入日志采集、监控与告警体系，保障推荐服务的可观测性与可扩展性。

---

## 后续规划

**本节回答：**
- 近期计划做哪些功能或改造（缩写检测、三级领域编码等）？
- 与当前实现的衔接点在哪？

- **文本转向量前的缩写检测**  
  在现有文本转向量流程前增加缩写检测与展开步骤：先对查询/岗位文本做缩写识别与扩展，再用扩展后的文本进行向量编码；仅需改底层编码入口逻辑即可接入。

- **学术词带三级领域再编码**
  在知识图谱构建与正式标签路中统一使用「三级领域」表示学术词：每个学术词在参与文本转向量（以及写入图谱或参与标签路计算）时，不再仅使用裸词，而是附加其对应的一级、二级、三级领域名称，格式为「词 (一级领域) (二级领域) (三级领域)」，领域数据来自 `vocabulary_topic_stats` / `vocabulary_topic_index.json`。同一词在不同领域下将具有不同的向量表示，便于区分与排序；KG 构建与标签路将共用同一套「带括号领域」的生成与编码逻辑，保证语义一致。

- **缩写扩写与三级领域的使用策略**
  本系统在语义召回与知识图谱扩展中同时使用 **缩写扩写（Abbreviation Expansion）** 与 **三级领域结构（Field → Subfield → Topic）**，两者职责不同，采用 **Index-side 与 Query-side 分离** 策略：**向量索引保持语义纯净，查询阶段进行语义增强，图谱阶段利用领域信息进行结构化排序与扩展。**
  - **1 缩写扩写**
    - **缩写词典**：系统维护一份缩写扩写词典（如 `rrt`→Rapidly-exploring Random Tree、`mpc`→Model Predictive Control、`slam`→Simultaneous Localization and Mapping、`cnn`→Convolutional Neural Network、`llm`→Large Language Model 等），用于解决技术岗位 JD 中常见缩写。
    - **向量索引中的使用方式（Alias Embedding）**：构建 Vocabulary 向量索引时，缩写**不替换原词**，而是作为 alias 加入 embedding 文本，格式为 `term | abbreviation`。例如 `rapidly exploring random tree | rrt`、`model predictive control | mpc`。这样 embedding 同时包含缩写与全称，用户输入缩写或完整术语都能召回，且不修改原始词汇结构。**工业词**（entity_type='industry'）会先经 `tools.normalize_skill` 清洗，清洗后为空的词不进入索引；用于编码的文本为「清洗后的词」或「清洗后的词 | 缩写…」，再与缩写扩写合并后建向量；索引内只保留 voc_id。
    - **Query 阶段**：在 Query 预处理中先做缩写检测并扩写（如 `RRT planning algorithm` → `rapidly exploring random tree planning algorithm`），再进行向量编码。流程：JD → Skill Clean → Abbreviation Expansion → Embedding。
  - **2 三级领域结构**
    词汇统计阶段构建 Field → Subfield → Topic 层级（如 Engineering → Robotics → Motion Planning）。词汇在不同领域中可有不同概率分布（例：`rapidly exploring random tree` 在 robotics.motion_planning / control.theory / optimization 下的分布），存储在 `vocabulary_topic_stats`。
  - **3 为什么领域信息不进入向量索引**
    领域信息**不**加入 embedding（不做「词 (field.subfield.topic)」形式的向量编码）。原因：embedding 模型会把括号内内容当作语义，导致 robotics、motion planning、robot control 等向量过度聚集，产生**语义空间塌缩（Semantic Collapse）**。因此策略为：**领域信息只用于图谱与标签路排序/扩展，不进入向量索引。**
  - **4 三级领域在标签路中的作用**  
    三级领域**不是**独立图节点系统，而是 **vocabulary_topic_stats 里的词级附属统计**：有直接标签时存 `field_id`/`subfield_id`/`topic_id`，无标签或缺层级时存 `field_dist`/`subfield_dist`/`topic_dist`（占比 JSON，来自共现伙伴按 freq 加权补全）。因此 README 中的正确定位是：**Stage2 把三级领域当作单词级过滤与扩展护栏**，**Stage3 把三级领域当作单词级一致性分与加权项**；并需明确使用**百分比阈值**（如 topic_overlap > 0.20、subfield_overlap > 0.30、field_overlap > 0.40），因为表里存的就是 `topic_dist`/`subfield_dist`/`field_dist` 这类占比。具体接入方式见后文「三级领域在 Stage2 / Stage3 中的接入方式（基于单词级统计索引）」。
    - **领域加权排序**：候选词评分中 `domain_ratio` / topic_fit 来自 `vocabulary_topic_stats` 的占比分布，作为加权项而非主语义来源。
    - **扩展词过滤**：SIMILAR_TO 扩展阶段要求候选在主领域上的占比超过阈值（如 `topic_overlap > 0.20`）才允许扩展。
    - **Cluster 扩展控制**：`cluster_weight = cluster_factor × topic_similarity`。
  - **5 完整召回流程**
    JD → Skill Clean → Abbreviation Expansion → Embedding → **Vocabulary Projection**（top20 vocab）→ **Domain Detection + Domain Anchored Expansion**（topic 聚合 + topic_overlap 过滤，取 top5 作 new query）→ 再 Embedding → Vector Recall (final) → **Anchor Rerank**（semantic × idf）→ **Cluster Diversification**（按簇去重，anchor ≈20）→ LabelPath Expansion → Academic Term → Paper → Author → KGAT-AX Ranking。（参见「Vector 与 LabelPath 衔接的两个隐藏问题」与「数据驱动的 Query 语义标准化」）
  - **6 设计总结**
    | 模块 | 作用 |
    |------|------|
    | 缩写扩写 | 提升技术缩写召回能力 |
    | 向量索引 alias | 支持缩写与全称双匹配 |
    | 三级领域统计 | 提供领域约束 |
    | 标签路 | 进行结构化扩展 |
    | KGAT-AX | 最终排序 |
    整体为 **语义召回 + 领域统计 + 图谱扩展** 的混合架构：Vector Recall 解决语义匹配，Domain Statistics 解决领域偏差，Knowledge Graph Expansion 解决结构化推理。

- **Vector 与 LabelPath 衔接的两个隐藏问题（召回结构 + 评分一致）**
  VectorPath 与 LabelPath 衔接处有 **两个容易被忽略但影响很大的结构问题**：语义簇坍缩 与 召回–排序目标不一致。Graph reasoning 完全依赖 anchor 质量；若 anchor 不多样或语义偏移，后续 SIMILAR_TO / cluster / topic 都会被带偏。系统已具备 `vocabulary_cluster`、`vocabulary_topic_stats`、`paper_count`、cooccurrence 等，修复只需在 Vector Recall 与 LabelPath 之间插一层 **anchor rerank + cluster diversify**，几十行代码即可。  
  - **隐藏问题 ①：语义簇坍缩（semantic cluster collapse）**  
    向量检索 topK 时，embedding 呈簇结构，同一簇（如 trajectory planning / trajectory optimization / trajectory tracking …）可能占满名额，LabelPath 的 anchor 集中在单一方向，扩展只往该簇走，其它方向（motion planning、RRT、PRM、robot motion control 等）被淹没。**解决**：在 Vector Recall 之后做 **cluster diversified recall**——按簇去重，每簇最多保留 `cluster_max_per` 条，得到约 `anchor_final` 个多簇 anchor。推荐参数：`vector_topK = 50`，`cluster_max_per = 3`，`anchor_final = 20`。  
  - **隐藏问题 ②：Vector 与 LabelPath 评分体系不一致（recall–ranking objective mismatch）**  
    Vector Recall 只依赖 **cosine(query_vec, term_vec)**；LabelPath 依赖 **semantic × idf × topic_ratio × cooccurrence × cluster**。因此“Vector 排名靠前”≠“LabelPath 最优 anchor”。例如 query「robot motion planning」时，trajectory learning / trajectory prediction 等 embedding 更近、易进 topK，而 motion planning、path planning、RRT、PRM 等 semantic 略远但 domain_ratio/cooccurrence 更高，在 Vector 排名靠后，进不了 anchor，LabelPath 无法扩展它们。**解决**：在 **Vector Recall → Anchor** 之间插一层 **anchor rerank**，用与 LabelPath 一致的信号做一次重排：`anchor_score = semantic_score × idf(term)`，idf 可用 `1 / log(1 + paper_count)`，用现有 `paper_count` 零成本实现；同时 idf 能压制常见泛词（model、learning、method、framework、system 等 hub nodes），避免其占满 anchor。  
  - **最终推荐召回 pipeline**  
    `JD → abbrev expand → embedding → vector recall (top50) → **anchor rerank**（semantic × idf）→ **cluster diversification** → anchor terms (≈20) → label_path expansion → academic term → paper → author → KGAT-AX ranking`。  
  - **实现要点**  
    - **anchor rerank**：对 vector top50 用 `semantic_score × idf(term)` 重排后再做 cluster 去重。  
    - **cluster diversify**：用 `vocabulary_cluster` 遍历重排结果，每簇最多取 `cluster_max_per` 条，直至 anchor 数达 `anchor_final`。  
  - **小结**  
    两个问题都指向 **anchor 质量与结构**；补上 **anchor rerank + cluster diversified recall** 后，再接入现有 LabelPath 与 KGAT-AX，整体效果会更稳定。

- **数据驱动的 Query 语义标准化（Vocabulary Projection + Domain Anchored Expansion）**
  不做规则/词典/if-else 的 **硬编码 normalization**，而做 **数据驱动的 semantic grounding**：目标是把 JD 拉到「更接近学术词汇空间」（工业词 → 学术词邻域），而不是「工业词 → 唯一 canonical term」的确定性映射。利用现有约 6.5w **vocabulary embedding** 做 **Vocabulary Projection**：query 先 embedding，再对 vocabulary 做 vector search 取 top3～top5 词，用这些词组成 **new query** 再继续召回，即 **embedding-driven query expansion / semantic grounding**（IR 里常称 query expansion）；无需额外模型，与现有「JD → vector recall vocabulary → label_path」一致，projection 可视为 vector recall 的副产物。取 top3～top5 而非 top1，使 query 对应一个语义子空间而非单词。  
  - **Semantic Drift（Query Drift）及必要性**  
    若直接用 top5 vocabulary 做 expansion，会出现 **semantic drift**：embedding 中某些 hub 词（如 trajectory、model、learning、optimization）与很多 query 都较近，第二轮 embedding 后 query 被带偏（例如全变成 trajectory*），真正需要的 motion planning、RRT、PRM、robot motion control 被挤出。文献结论：**bad expansion > no expansion**，必须对 expansion 做约束。  
  - **Domain Anchored Expansion（防 drift）**  
    思路：**只允许与 query 主领域一致的 vocabulary 参与 expansion**。步骤：(1) 对 query 做 **vector search vocabulary top20**；(2) **确定 query 主领域**：对候选词的 `vocabulary_topic_stats`（topic_dist/subfield_dist/field_dist）做聚合，取占比最高的 topic 作为 main_domain；(3) **过滤候选**：只保留 `topic_overlap > threshold`（如 0.2）的 term，即该 term 在 main_domain 上的概率超过阈值；(4) 用过滤后的 term 取 top5 做 **query expansion**，得到 `grounded_query = " ".join(filtered_terms[:5])`，再 encode 做后续 vector recall。这样 trajectory prediction / trajectory dataset 等与主领域 robotics.motion_planning 不一致的会被滤掉，保留 motion planning、path planning、sampling based planning、trajectory planning 等，抑制 semantic drift。  
  - **实现要点**  
    - 依赖现有 **vocabulary_topic_stats**（topic_dist 等），无需新数据。  
    - 逻辑示例：`candidates = vector_search(query, topk=20)` → 聚合各 term 的 topic_dist 得 `domain_scores` → `main_domain = argmax(domain_scores)` → `filtered_terms = [t for t in candidates if topic_dist[t].get(main_domain, 0) > 0.2]` → `query = " ".join(filtered_terms[:5])`，再 encode。  
  - **完整 Query 预处理与召回 pipeline**  
    `JD → skill clean → abbrev expand → embedding → vector recall vocabulary (top20) → domain detection（topic 聚合）→ domain filtered expansion（topic_overlap > 0.2，取 top5）→ new query → 再 embedding → vector recall (final) → anchor rerank (semantic × idf) → cluster diversification → anchor terms (≈20) → label_path expansion → academic term → paper → author → KGAT-AX ranking`。  
  - **小结**  
    语义标准化采用 **embedding-driven query grounding**，不维护规则；用 **domain anchored expansion** 防止 semantic drift，充分利用现有 **semantic embedding + vocabulary_topic_stats + knowledge graph** 的 Hybrid Retrieval 架构。

- **三层领域在 Stage2B / Stage3 中的接入方案（修订版）**  
  三层领域（field / subfield / topic）**不参与**“工业词是否能够成功落到学术词”的判定，仅用于**学术词扩散阶段的主题纠偏**与**最终词打分阶段的层级一致性校正**。Stage2A 保持只用一层领域；三层领域明确限定在 Stage2B 与 Stage3。

  - **1. 定位与原则**  
    - **Stage2A（工业词 → 学术词）**：保持现状，只使用一层领域与现有桥接信号完成 academic landing（exact/alias/lexical landing、dense landing、跨类型 SIMILAR_TO、primary 候选合并与 identity 评分）。本阶段**不引入**三层领域约束。  
    - **Stage2B（学术词扩散）**：在 dense / cluster / cooc 扩散候选基础上，为每个学术词补充三层主题信息并计算 `topic_align`，作为扩散质量控制信号。  
    - **Stage3（最终词打分）**：将三层领域作为第二层主题一致性信号接入最终词分，抑制 dense/cluster/cooc 扩散带来的语义漂移，对主题一致的扩散词做有限加成，**不替代**现有 identity / quality / role-based base_score。

  - **2. JD 的三层主题输入**  
    统一记为 `jd_field_ids`、`jd_subfield_ids`、`jd_topic_ids`。若召回入口已提供 field/subfield/topic 则直接传入；若只有 `domain_ids`，需在进入 Stage2B 前完成映射 `domain_id -> {field_ids, subfield_ids, topic_ids}`（配置或静态表）。工业词→学术词的 landing 仍只依赖一层领域，三层集合仅供 Stage2B/Stage3 使用。

  - **3. Stage2B：为每个学术词补三层对齐信息**  
    在 `merge_primary_and_support_terms`（或生成 raw_candidates 的同层）中，对每个候选按 `voc_id` 查 **vocabulary_topic_stats**，读取：`field_id`、`subfield_id`、`topic_id`、`field_dist`、`subfield_dist`、`topic_dist`、`source`。用主值或分布与 JD 三层集合计算 `topic_align`，并可选保留 `topic_level`、`topic_confidence` 便于调试。

  - **4. topic_align：层级命中 × 可信度**  
    - **公式**：`topic_align = hierarchy_match_score * topic_confidence`。  
    - **hierarchy_match_score**：topic 一致 → 1.0；仅 subfield 一致 → `TOPIC_ALIGN_SUBFIELD`（默认 0.65）；仅 field 一致 → `TOPIC_ALIGN_FIELD`（默认 0.35）；均未命中 → 0.0。  
    - **topic_confidence**：按 `source` 映射，例如 `direct`→1.0、`direct+cooc`→0.9、`cooc`→0.7；暂未稳定时可退化为 1.0。  
    - **仅存在 dist 无主值**：用 JD 集合在 `*_dist` 上的概率和做自上而下优先的档位映射（topic_overlap > 0 → topic 档；否则 subfield → field → none）。  
    - 每个候选补充字段：`topic_align`、`topic_level`（可选）、`topic_confidence`（建议保留）。

  - **5. Stage3：乘性融合与按 term_role 的 topic 权重**  
    - **乘性融合**：`topic_factor = 1 - topic_weight + topic_weight * topic_align`，`final_score = base_score * topic_factor`。  
    - **按 term_role 分配 topic_weight**：`primary` 建议 0.10；`dense_expansion` 0.25；`cluster_expansion` 0.35；`cooc_expansion` 0.25。  
    - **低对齐惩罚仅作用于 expansion**：若 `term_role ∈ {dense_expansion, cluster_expansion, cooc_expansion}` 且 `topic_align < TOPIC_MIN_ALIGN`（如 0.20），则 `final_score *= TOPIC_LOW_ALIGN_PENALTY`（如 0.50）。**不对 primary 施加**该惩罚。

  - **6. 参数汇总（建议进入 config）**  

    | 参数 | 含义 | 默认建议 |
    |------|------|----------|
    | TOPIC_ALIGN_SUBFIELD | 仅 subfield 对齐时的层级分 | 0.65 |
    | TOPIC_ALIGN_FIELD | 仅 field 对齐时的层级分 | 0.35 |
    | TOPIC_WEIGHT_PRIMARY | primary 落点的三层领域权重 | 0.10 |
    | TOPIC_WEIGHT_DENSE | dense 扩散词的三层领域权重 | 0.25 |
    | TOPIC_WEIGHT_CLUSTER | cluster 扩散词的三层领域权重 | 0.35 |
    | TOPIC_WEIGHT_COOC | cooc 扩散词的三层领域权重 | 0.25 |
    | TOPIC_MIN_ALIGN | expansion 低对齐门限 | 0.20 |
    | TOPIC_LOW_ALIGN_PENALTY | 低对齐 expansion 惩罚系数 | 0.50 |

  - **7. 实现要点（方案级）**  
    - **JD 主题准备**：Stage2B 入口前准备 `jd_field_ids`、`jd_subfield_ids`、`jd_topic_ids`；若仅有 `domain_ids` 则先做映射。  
    - **Stage2B**：在 `merge_primary_and_support_terms` 中对每个 voc_id 查 vocabulary_topic_stats，算 `hierarchy_match_score`、`topic_confidence`、`topic_align`，写回 `rec["topic_align"]`、`rec["topic_level"]`、`rec["topic_confidence"]`。  
    - **Stage3**：在 `compose_term_final_score` 中先得 base_score，按 term_role 取 topic_weight，计算 `topic_factor` 与 `final_score = base_score * topic_factor`；若为 expansion 且 `topic_align < TOPIC_MIN_ALIGN` 再乘低对齐惩罚。  
    - **缺层/缺表退化**：无记录或信息不全时推荐 `topic_align = 1.0`（不参与惩罚，退化为现有一层逻辑）；仅在覆盖率足够时再考虑 0.0。

  - **8. Stage2B 小节成稿（README 可落地）**  
    Stage2A 完成后进入学术词扩散。三层领域在此的职责是：不参与工业词能否落地；只参与学术词扩散候选的主题一致性评估；为 Stage3 提供层级主题信号。Stage2B 输入为 Stage2A 的 primary academic terms 与 JD 的三层主题集合（`jd_field_ids` / `jd_subfield_ids` / `jd_topic_ids`）。候选来源仍为 primary_landing、dense_expansion、cluster_expansion、cooc_expansion。在候选合并阶段对每个 voc_id 查 vocabulary_topic_stats，计算 `topic_align = hierarchy_match_score * topic_confidence`，输出字段含 `topic_align`、`topic_level`、`topic_confidence`。

  - **9. Stage3 小节成稿（README 可落地）**  
    Stage3 在已有候选上做过滤与最终词得分。三层领域作为“扩散纠偏器”：先算 base_score，再按 `topic_factor = 1 - topic_weight + topic_weight * topic_align` 做乘性修正，`final_score = base_score * topic_factor`。按 term_role 取不同 topic_weight；对 expansion 且 `topic_align < TOPIC_MIN_ALIGN` 再乘 `TOPIC_LOW_ALIGN_PENALTY`，且该惩罚**不作用于 primary**。缺表/缺层时 `topic_align = 1.0` 退化。

  - **10. 伪代码骨架**  

```python
# 伪代码 1：Stage2B 候选补充三层主题
def merge_primary_and_support_terms(primary_terms, dense_terms, cluster_terms, cooc_terms, jd_field_ids, jd_subfield_ids, jd_topic_ids):
    raw_candidates = merge_all_sources(primary_terms, dense_terms, cluster_terms, cooc_terms)
    for rec in raw_candidates:
        voc_id = rec["voc_id"]
        topic_row = load_vocabulary_topic_stats(voc_id)
        if not topic_row:
            rec["topic_align"] = 1.0
            rec["topic_level"] = "missing"
            rec["topic_confidence"] = 1.0
            continue
        hierarchy_match_score, topic_level = compute_hierarchy_match_score(topic_row, jd_field_ids, jd_subfield_ids, jd_topic_ids)
        topic_confidence = compute_topic_confidence(topic_row)
        rec["topic_align"] = hierarchy_match_score * topic_confidence
        rec["topic_level"] = topic_level
        rec["topic_confidence"] = topic_confidence
    return raw_candidates

# 伪代码 2：层级命中
def compute_hierarchy_match_score(topic_row, jd_field_ids, jd_subfield_ids, jd_topic_ids):
    topic_ids = extract_topic_ids(topic_row)
    subfield_ids = extract_subfield_ids(topic_row)
    field_ids = extract_field_ids(topic_row)
    if has_overlap(topic_ids, jd_topic_ids): return 1.0, "topic"
    if has_overlap(subfield_ids, jd_subfield_ids): return TOPIC_ALIGN_SUBFIELD, "subfield"
    if has_overlap(field_ids, jd_field_ids): return TOPIC_ALIGN_FIELD, "field"
    return 0.0, "none"

# 伪代码 3：主题可信度
def compute_topic_confidence(topic_row):
    source = topic_row.get("source")
    if source == "direct": return 1.0
    if source == "direct+cooc": return 0.9
    if source == "cooc": return 0.7
    return 1.0

# 伪代码 4：Stage3 最终词打分
def compose_term_final_score(rec):
    base_score = compute_base_score(rec)
    topic_align = rec.get("topic_align", 1.0)
    topic_weight = get_topic_weight_by_role(rec["term_role"])
    topic_factor = 1 - topic_weight + topic_weight * topic_align
    final_score = base_score * topic_factor
    if rec["term_role"] in {"dense_expansion", "cluster_expansion", "cooc_expansion"} and topic_align < TOPIC_MIN_ALIGN:
        final_score *= TOPIC_LOW_ALIGN_PENALTY
    return final_score

# 伪代码 5：按 role 取 topic_weight
def get_topic_weight_by_role(term_role):
    if term_role == "primary": return TOPIC_WEIGHT_PRIMARY
    if term_role == "dense_expansion": return TOPIC_WEIGHT_DENSE
    if term_role == "cluster_expansion": return TOPIC_WEIGHT_CLUSTER
    if term_role == "cooc_expansion": return TOPIC_WEIGHT_COOC
    return 0.0
```

  - **11. 总结**  
    三层领域不参与工业词到学术词的 primary landing 判定，仅在 academic term expansion 与 term final scoring 中作为二级主题一致性信号接入。核心目标是抑制 dense/cluster/cooc 扩散带来的语义漂移，对 support term 做层级主题纠偏，同时不替代现有 identity/quality backbone，保持标签路“先落地、再扩散、后校正”的整体结构稳定。

- **流行度偏置（Popularity Bias / Citation Bias）与论文/作者评分**
  在「词 → 论文 → 作者」链路上存在 **Popularity Bias / Citation Bias**：若不处理，召回看起来合理，但作者几乎全是「老牌高引大牛」而非 **当前岗位最适合的活跃研究者**。问题出在 **vocabulary → paper** 这一层：OpenAlex/Semantic Scholar 呈 **citation power-law**，极少数论文拥有极大引用量，paper 聚合容易变成 **citation ranking** 而非 **relevance ranking**，导致 1990s 开创性论文（如 RRT/PRM 原论文）及其作者长期霸榜，而近 5 年在该方向持续产出的工程研究者被压低。系统目标已是「最适合岗位的活跃研究者」，因此必须在论文评分与作者聚合阶段显式抑制流行度偏置。  
  - **论文评分公式（时间衰减 + 引用归一）**  
    推荐：`paper_score = term_score × time_decay × citation_norm`。其中 **term_score** 来自 label_path（semantic × idf × topic_ratio）。**time_decay**：`decay_rate ** year_diff`，例如 `decay_rate = 0.92`，则 2023 年论文 ≈ 0.92^1、2015 年 ≈ 0.92^9≈0.48、2005 年 ≈ 0.92^19≈0.19，老论文自然下降；可按领域配置 `get_decay_rate_for_domains(domain_ids)`。**citation_norm**：不用原始 citation，推荐 `log(1 + citation)` 或 `citation / log(1 + age)`，避免高引论文压死一切。  
  - **作者评分：top-K 聚合 + 活跃度因子**  
    不要用 `author_score = sum(paper_score)`（会导致「发很多篇的人」统治排名）。推荐：`author_score = sum(top_k paper_score) × activity_factor`，例如 **top5** 只取作者最相关的 5 篇论文；**activity_factor** 衡量近期活跃度，如 `recent_5y_papers / total_papers`（近 5 年论文比例）。  
  - **最终 ranking pipeline（含论文/作者层）**  
    `JD → … → label_path expansion → academic terms → **paper retrieval** → **paper scoring**（semantic × topic × time_decay × citation_norm）→ **author aggregation**（top5 papers）→ **activity weighting** → KGAT-AX ranking`。  
  - **与前述三个问题的关系**  
    当前设计同时应对：① **semantic drift**（domain anchored expansion）；② **candidate collapse**（cluster diversification + anchor rerank）；③ **popularity bias**（time decay + citation_norm + topK 论文聚合 + activity_factor）。  
  - **实现成本**  
    已有 `paper_year`、`citation_count`、`domain` 等字段，改动集中在 paper scoring 与 author aggregation 两处，几十行代码即可接入。

- **多路召回融合策略（Multi-Source Candidate Fusion）**
  本系统采用三路召回：**Vector Recall**（语义相似）、**Label Path Recall**（知识图谱路径）、**Collaboration Recall**（合作网络），分别对应 JD↔论文/词汇语义、term→paper→author 图谱结构、co-author/community 网络邻近性。三路输出为不同分布的候选（semantic / graph / network relevance），若直接并集后送 KGAT-AX，会出现 **distribution mismatch**（三路“相关性”定义不同）、**source bias**（模型学到“来自某一路→高分”的 shortcut）、**negative sampling instability**（负样本混杂三类，梯度与难负例比例不可控）。因此需在进入 KGAT-AX 前引入 **候选对齐层（Candidate Calibration）**：多路召回 → 去重与合并 → 分数归一化 → 融合预排序 → 统一候选池 → KGAT-AX。  
  - **融合流程**  
    (1) 多路召回：`vector_topK=150`、`label_topK=150`、`collab_topK=150`。(2) 按 author_id 去重，保留 **source_flags**（from_vector / from_label / from_collab）。(3) 每个候选保留三路原始分数 vector_score、label_score、collab_score。(4) **分数归一化**：对三路分数分别做 rank-based 或 min-max 归一化，得到 vector_score_norm、label_score_norm、collab_score_norm ∈ [0,1]。(5) **融合预排序**：`fusion_score = w1*vector_score_norm + w2*label_score_norm + w3*collab_score_norm`，推荐初始权重 w1=0.4、w2=0.4、w3=0.2（vector 与 label 为主信号，collab 为补充）。(6) 按 fusion_score 截断为 **top 150～200** 作为 KGAT-AX 输入候选池。  
  - **统一候选结构**  
    每个候选作者包含：author_id、vector_score / label_score / collab_score、*_score_norm、fusion_score、source_flags；该结构作为 KGAT-AX 输入特征的一部分，便于模型学习多路证据融合而非依赖单路。  
  - **训练阶段：分层负采样**  
    为稳定训练，采用 **分层负样本**：每个正样本对应约 2×vector negatives、2×label negatives、1×collab negatives，使 hard negative 覆盖不同语义、防止模型偏向某一路、提升判别能力。  
  - **最终召回与排序流程**  
    `JD → Query preprocessing → Vector Recall + Label Path Recall + Collaboration Recall → Candidate Calibration（去重、归一化、融合排序）→ Top-N Candidates (≈150)→ KGAT-AX Ranking → Top-K Authors`。  
  - **小结**  
    通过候选对齐与多路融合，实现 **Semantic + Graph + Network → Unified Ranking**，避免 distribution mismatch、source bias、training instability，使推荐结果更稳定、可解释（可分析某作者因 semantic/graph/collaboration 哪路被推荐）。

- **标签路 Stage1～Stage3 改造（学术词落点 + 双闸门）**
  当前主问题集中在「学术词不对」。目标是把链路从「锚点 → 直接扩学术词 → 再过滤」改成「**锚点分类 → 学术落点 → 受限扩展 → 身份闸门 → 质量闸门**」。不新增大模型重写器，而是补齐三层函数能力：**学术落点层**（先把工业锚点对到可信学术词）、**受限扩展层**（围绕落点少量扩展，不围绕原始锚点大扩）、**双闸门层**（先判「是不是那个词」再判「值不值得保留」）。  
  - **Stage1**（`stage1_domain_anchors.py`、`label_anchors.py`、`tools.py`、`domain_detector.py`）：新增锚点类型标注 `classify_anchor_type`（acronym / canonical_academic_like / application_term / generic_task_term / unknown）、缩写扩写 `expand_anchor_acronyms`、规范化 `normalize_anchor_term`、预处理总控 `prepare_anchor_candidates`（输出含 anchor_type），让 Stage2 不再直接吃裸 anchor。  
  - **Stage2 拆为 2A + 2B**（`stage2_expansion.py`、`label_expansion.py`、`infra.py`）：**Stage2A 学术落点**：锚点为工业侧 skill，**直接使用跨类型 SIMILAR_TO**（图内已是带缩词扩写的向量相似度）得到学术词候选；可选 JD 文本→词汇向量补充；`retrieve_academic_term_by_similar_to`、`collect_landing_candidates`、`score_academic_identity`、`select_primary_academic_landings`（每 anchor 1～3 个主落点，min_identity_score 约 0.62）。**不做** exact/alias（锚点非学术词）、**不做** anchor→dense（与 SIMILAR_TO 同源冗余）。**Stage2B 学术侧补充**：仅围绕 primary 做 `expand_from_vocab_dense_neighbors`（词汇向量索引取学术近邻）、`expand_from_cluster_members`（簇内支持词）、`expand_from_cooccurrence_support`（共现高支持度）、`merge_primary_and_support_terms`（标注 term_role：primary / dense_expansion / cluster_expansion / cooc_expansion）。**不再**通过 SIMILAR_TO 做学术词→学术词扩展。  
  - **Stage3**（`stage3_term_filtering.py`、`term_scoring.py`、`advanced_metrics.py`、`simple_factors.py`）：从「一个总词分」改为「两分 + 双闸门」。新增身份闸门 `passes_identity_gate`（按 term_role：primary / dense_expansion / cluster_expansion / cooc_expansion 设不同阈值）、质量分 `score_term_expansion_quality`（idf、domain_purity、cooc_purity、resonance、span_penalty 等）、最终分 `compose_term_final_score`（identity + quality 按 term_role 加权）、泛任务惩罚 `apply_generic_task_penalty`、topic 一致性 `passes_topic_consistency`、调试 `diagnose_term_error_type`、`build_term_debug_record`。改造 `compute_anchor_landing_alignment`：区分原始 anchor、扩写词、别名词的对齐。  
  - **Stage4/5 轻量改造**：Stage4 的 term→paper 贡献区分 term_role（primary 权重大、expansion 适中/更低）；新增 `compute_primary_term_coverage`（论文被多少 primary term 支撑）。Stage5 的 `aggregate_author_evidence_by_term_role` 区分 primary-supported 与 expansion-supported evidence，便于可解释性。  
  - **统一 term 结构**：建议采用 `TermCandidate` 等结构，字段含 anchor、anchor_type、term_id、term、term_role、source、identity_score、quality_score、final_score、dense_sim、alias_hit、acronym_hit、domain_fit、topic_fit、cooc_purity、is_primary_landing、debug_error_type 等，避免「一个 record 只存一个总分」难以扩展。
  - **改造顺序建议**：**第一批**（学术落点）：`retrieve_academic_term_by_similar_to`、`collect_landing_candidates`、`score_academic_identity`、`select_primary_academic_landings`。**第二批**（Stage2/3 成型）：`passes_identity_gate`、`score_term_expansion_quality`、`compose_term_final_score`、`expand_from_vocab_dense_neighbors`、`expand_from_cluster_members`、`expand_from_cooccurrence_support`、`merge_primary_and_support_terms`。**第三批**（可解释与调试）：可选 JD 向量补充、`diagnose_term_error_type`、`build_term_debug_record`、`compute_primary_term_coverage`、`aggregate_author_evidence_by_term_role`。
  - **Stage2/Stage3 主流程骨架**：Stage2 总入口 `stage2_generate_academic_terms(prepared_anchors, active_domains)` → 对每个 anchor 调用 `collect_landing_candidates`（**内部仅**：跨类型 SIMILAR_TO + 可选 JD 向量）→ 对候选算 `score_academic_identity` → `select_primary_academic_landings` → Stage2B 仅对 primary 做 `expand_from_vocab_dense_neighbors`、`expand_from_cluster_members`、`expand_from_cooccurrence_support` → `merge_primary_and_support_terms`。Stage3 总入口 `stage3_filter_and_score_terms(term_candidates, active_domains)` → 对每个 term 先 `passes_identity_gate`、再 `passes_topic_consistency` → 算 `score_term_expansion_quality` → `compose_term_final_score` → `apply_generic_task_penalty_if_needed` → 质量闸门 → `diagnose_term_error_type`、`build_term_debug_record`。
  - **标签路改造目标伪代码骨架**（计划落地后的形态）：

```python
# 统一 term 候选结构（目标）
@dataclass
class TermCandidate:
    anchor: str
    anchor_type: str  # acronym | canonical_academic_like | application_term | generic_task_term | unknown
    term_id: int
    term: str
    term_role: str    # primary | dense_expansion | cluster_expansion | cooc_expansion
    source: str
    identity_score: float
    quality_score: float
    final_score: float
    dense_sim: float
    alias_hit: bool
    acronym_hit: bool
    domain_fit: float
    topic_fit: float
    cooc_purity: float
    is_primary_landing: bool
    debug_error_type: str  # acronym_error | alias_mapping_error | generic_drift | domain_mismatch | weak_identity | ...

# Stage2 目标主流程
def stage2_generate_academic_terms(prepared_anchors, active_domains):
    term_candidates = []
    for anchor in prepared_anchors:
        landing_candidates = collect_landing_candidates(anchor)  # 内部：跨类型 SIMILAR_TO + 可选 JD 向量
        for c in landing_candidates:
            c["identity_score"] = score_academic_identity(c, anchor)
        primary_landings = select_primary_academic_landings(landing_candidates, min_identity_score=0.62)
        dense_list = expand_from_vocab_dense_neighbors(primary_landings)
        cluster_list = expand_from_cluster_members(primary_landings)
        cooc_list = expand_from_cooccurrence_support(primary_landings)
        term_candidates.extend(merge_primary_and_support_terms(primary_landings, dense_list, cluster_list, cooc_list))
    return term_candidates

# Stage3 目标主流程
def stage3_filter_and_score_terms(term_candidates, active_domains):
    out = []
    for tc in term_candidates:
        if not passes_identity_gate(tc): continue
        if not passes_topic_consistency(tc, active_domains): continue
        tc.quality_score = score_term_expansion_quality(tc)
        tc.final_score = compose_term_final_score(tc)
        apply_generic_task_penalty_if_needed(tc)
        tc.debug_error_type = diagnose_term_error_type(tc)
        build_term_debug_record(tc)
        if passes_quality_gate(tc):
            out.append(tc)
    return out
```

  - **最小可运行版（MVP）**：先落地 `retrieve_academic_term_by_similar_to`、`collect_landing_candidates`、`score_academic_identity`、`select_primary_academic_landings`、`expand_from_vocab_dense_neighbors` / `expand_from_cluster_members` / `expand_from_cooccurrence_support`、`merge_primary_and_support_terms`、`passes_identity_gate`、`score_term_expansion_quality`、`compose_term_final_score`、`build_term_debug_record`，即可从「锚点直接扩词」升级为「锚点经跨类型 SIMILAR_TO 找主学术落点，再围绕主落点用 dense/簇/共现做少量补充」。  
  - **核心原则**：Stage2A 回答「我到底在说哪个学术词」（以跨类型 SIMILAR_TO 为主），Stage2B 回答「围绕这个词可以补哪些学术支持词」（dense/簇/共现，不再用 SIMILAR_TO 做学术→学术），Stage3 闸门1 回答「这个词身份对不对」，闸门2 回答「这个词质量高不高」；职责切开后调试与迭代会更清晰。

  - **标签路目标（一句话）**：标签路的目标**不是扩大召回面**，而是提高「学术词命中正确率」和「高价值论文证据纯度」；本质上是**高精度学术语义对齐模块**（应用词→学术词落点、缩写消歧、对口论文证据提纯），而不只是普通召回路。

  - **工程护栏与落地约束（六条）**：为稳定落地，必须在计划中显式加入以下约束与规则。  
    - **护栏 1：无主落点禁止扩展**。只要某个 anchor 没有通过 `select_primary_academic_landings` 选出 primary landing，就**禁止进入 Stage2B 扩展**：不许 dense/cluster/cooc 扩展；该 anchor 直接记为 `unresolved_anchor`。避免“SIMILAR_TO 无命中但仍拿模糊近邻继续扩”，把泛任务词带回来。  
    - **护栏 2：primary 需唯一性优势**。除 `min_identity_score≈0.62` 外，增加**唯一性判断**：仅当 `top1_identity - top2_identity >= identity_margin`（建议 `identity_margin >= 0.08`）时才判定为稳定主落点；否则标为 `ambiguous_primary`，只保留 primary、不做邻域扩展，或降级为极保守模式。避免多义词/缩写映到多个学术主题时“两个都像”导致乱扩。  
    - **护栏 3：Stage2A 以跨类型 SIMILAR_TO 为主**。落点来源：**跨类型 SIMILAR_TO**（图内已是带缩词扩写的向量相似度）为主通道；可选 JD 文本→词汇向量作补充。**不做** anchor 的 exact/alias（锚点是 skill）、**不做** anchor→dense（与 SIMILAR_TO 同源冗余）。topic prior 仅作辅助，不单独决定 primary。  
    - **护栏 4：Stage3 按 term_role 设数量与权重配额**。除双闸门外，增加**角色配额**：每 anchor 最多保留 primary 3～5 个、dense_expansion / cluster_expansion / cooc_expansion 合计 5～8 个；最终候选中 **primary 贡献至少占总 term 权重的 50% 以上**。避免 expansion 数量过多在 Stage4 淹没主落点。  
    - **护栏 5：Stage4 primary coverage 硬门槛**。论文进入高优先级候选，至少满足其一：命中 ≥1 个高权重 primary term，或命中 ≥2 个 primary/supporting terms；**纯 expansion 支撑的论文不允许排到 very top**。即高分论文必须有「主学术词证据」，不能只靠扩展词堆出来。`compute_primary_term_coverage` 需与该规则一致。  
    - **护栏 6：调试输出做阶段级失败类型统计**。除单条 `diagnose_term_error_type` / `build_term_debug_record` 外，**每次 query 输出阶段级统计**，例如：`unresolved_anchor_count`、`ambiguous_primary_count`、`acronym_error_count`、`alias_mapping_error_count`、`generic_drift_count`、`domain_mismatch_count`、`weak_identity_count`。便于定位是落点、扩展还是质量闸门出问题，提升调参效率。

  - **Topic/Domain 约束原则**：domain/topic 只做**约束与校验**，不主导 primary 选择。落点以 academic identity 为核心；topic/domain 用于排除明显错域；不因 topic 看起来像就把 identity 不够强的 term 拉成 primary。

- **Stage2 / Stage3 实现契约与默认参数表**  
  本节补齐「可直接照着施工」所需的：**数据结构定义**、**默认阈值表**、**文件级改动顺序与函数挂载点**。补完后可按此逐函数实现，无需临场决定字段与阈值。

  - **一、数据结构表（中间对象与字段契约）**  
    以下 dataclass / 类型约定为 Stage2/Stage3 全链路统一使用，避免 dict 字段越传越乱。必填字段以 **粗体** 标出。

    | 类型名 | 用途 | 必填字段 | 可选/衍生字段 |
    |--------|------|----------|----------------|
    | **PreparedAnchor** | Stage1 输出，Stage2 输入锚点（见「四、实现细节契约」状态传递） | **anchor: str**, **anchor_type: str** (acronym \| canonical \| application \| task \| unknown), **expanded_forms: List[str]**, **weight: float**, **domain_hints: Set[str]** | normalized_term |
    | **LandingCandidate** | Stage2A 落点检索单条候选 | **vid: int**, **term: str**, **source: str** (similar_to \| jd_vector), **semantic_score: float** (边权或 cos_sim) | topic_prior_score |
    | **PrimaryLanding** | Stage2A 选出的主落点 | **vid: int**, **term: str**, **identity_score: float**, **source: str**, **anchor: str** | 同 LandingCandidate 可选字段 |
    | **ExpandedTermCandidate** | Stage2B 扩展后的单条 term（含 primary） | **vid: int**, **term: str**, **term_role: str** (primary \| dense_expansion \| cluster_expansion \| cooc_expansion), **identity_score: float**, **source: str**, **anchor: str** | quality_score, topic_fit, cooc_purity, resonance, span_penalty, from_primary_vid；**topic_align, topic_level, topic_confidence**（三层领域修订版） |
    | **TermCandidate** | Stage3 输入/输出统一 term 结构（与上文伪代码一致） | **anchor, anchor_type, term_id/vid, term, term_role, source, identity_score, quality_score, final_score** | domain_fit, topic_fit, cooc_purity, is_primary_landing, debug_error_type；**topic_align, topic_level, topic_confidence**（供 compose_term_final_score 乘性融合） |
    | **TermDebugRecord** | 调试与阶段统计用 | **term_id, term, term_role, error_type** (acronym_error \| alias_mapping_error \| generic_drift \| domain_mismatch \| weak_identity \| pass), **stage** (stage2a \| stage2b \| stage3) | identity_score, quality_score, gate_fail_reason |

    **函数级入参/出参契约（关键接口）**：

    | 函数 | 入参 | 出参 | 说明 |
    |------|------|------|------|
    | `retrieve_academic_term_by_similar_to(anchor: PreparedAnchor)` | 单锚点 | **List[LandingCandidate]** | 从 anchor（industry）查跨类型 SIMILAR_TO→学术词，图内为带扩写向量相似度 |
    | `collect_landing_candidates(anchor: PreparedAnchor)` | PreparedAnchor | **List[LandingCandidate]** | 内部调 similar_to + 可选 JD 向量，合并去重 |
    | `score_academic_identity(c: LandingCandidate, anchor: PreparedAnchor)` | 单条候选 + 锚点 | **float** [0,1] | 按 source（similar_to / jd_vector）与 semantic_score 等算身份分 |
    | `select_primary_academic_landings(candidates: List[LandingCandidate], min_identity_score, identity_margin)` | 候选列表 + 阈值 | **List[PrimaryLanding]** | 过滤 ≥ min_identity_score，且满足 top1−top2 ≥ identity_margin，每 anchor 最多 PRIMARY_MAX_PER_ANCHOR 个 |
    | `expand_from_vocab_dense_neighbors(primary_landings)` | 主落点列表 | **List[ExpandedTermCandidate]**（term_role=dense_expansion） | 词汇向量索引取 primary 的学术近邻，每 primary 最多 DENSE_MAX_PER_PRIMARY 个 |
    | `expand_from_cluster_members(primary_landings)` | 主落点列表 | **List[ExpandedTermCandidate]**（term_role=cluster_expansion） | 簇内成员，每 primary 最多 CLUSTER_MAX_PER_PRIMARY 个 |
    | `expand_from_cooccurrence_support(primary_landings)` | 主落点列表 | **List[ExpandedTermCandidate]**（term_role=cooc_expansion） | 共现频 ≥ COOC_SUPPORT_MIN_FREQ，每 primary 最多 COOC_MAX_PER_PRIMARY 个 |
    | `merge_primary_and_support_terms(primary_landings, dense_list, cluster_list, cooc_list)` | 主落点 + 三路扩展列表 | **List[ExpandedTermCandidate]** | 合并并标注 term_role；**并**对每条候选查 vocabulary_topic_stats 补充 topic_align、topic_level、topic_confidence（见上文三层领域修订版方案） |
    | `passes_identity_gate(tc: TermCandidate)` | 单条 TermCandidate | **bool** | 按 term_role 查表：primary ≥ PRIMARY_MIN_IDENTITY_GATE，dense_expansion/cluster_expansion ≥ DENSE_CLUSTER_MIN_IDENTITY_GATE，cooc ≥ COOC_MIN_IDENTITY_GATE |
    | `passes_topic_consistency(tc, active_domains)` | term + 当前领域 | **bool** | 三级领域占比：topic_overlap ≥ TOPIC_OVERLAP_MIN 或 subfield ≥ SUBFIELD_OVERLAP_MIN 或 field ≥ FIELD_OVERLAP_MIN |
    | `score_term_expansion_quality(tc)` | TermCandidate | **float** | 仅衡量「作为召回 term 好不好用」：idf、domain_purity、cooc_purity、resonance、span_penalty 等 |
    | `compose_term_final_score(tc)` | TermCandidate（含 identity_score、quality_score、topic_align） | **float** | 先得 base_score（identity+quality 按 term_role 加权），再乘 topic_factor = 1 - topic_weight + topic_weight * topic_align；expansion 且 topic_align < TOPIC_MIN_ALIGN 时再乘 TOPIC_LOW_ALIGN_PENALTY（不作用于 primary） |
    | `build_term_debug_record(tc)` | TermCandidate | **TermDebugRecord** | 填 error_type、stage、gate_fail_reason 等 |

  - **二、默认阈值表（实现时可直接照抄的常量）**  
    以下为推荐默认值，可在 config 或 label_means 常量中集中定义，便于调参。

    | 常量名 | 默认值 | 说明 |
    |--------|--------|------|
    | **Stage2A 主落点** | | |
    | PRIMARY_MIN_IDENTITY | 0.62 | 主落点身份分下限 |
    | IDENTITY_MARGIN | 0.08 | top1 与 top2 身份分差下限，否则标 ambiguous_primary |
    | PRIMARY_MAX_PER_ANCHOR | 3 | 每锚点最多主落点个数（可配置 3～5） |
    | **Stage2A SIMILAR_TO** | | |
    | SIMILAR_TO_MIN_SCORE | 0.65 | 跨类型 SIMILAR_TO 边权重下限（建图时已用带扩写向量） |
    | SIMILAR_TO_TOP_K | 3 | 每锚点沿 SIMILAR_TO 取学术词数 |
    | **Stage2B 扩展** | | |
    | DENSE_MAX_PER_PRIMARY | 4 | 每个主落点最多 dense 近邻数 |
    | CLUSTER_MAX_PER_PRIMARY | 3 | 每个主落点最多簇内支持词数 |
    | COOC_SUPPORT_MIN_FREQ | 2 | 共现最少出现频次 |
    | COOC_MAX_PER_PRIMARY | 2 | 每个主落点最多 cooc 扩展数（2～3 可调） |
    | **Stage3 三级领域（占比阈值，可选）** | | |
    | TOPIC_OVERLAP_MIN | 0.20 | topic 一致性闸门：候选词在主 topic 占比下限 |
    | SUBFIELD_OVERLAP_MIN | 0.30 | subfield 占比下限 |
    | FIELD_OVERLAP_MIN | 0.40 | field 占比下限 |
    | **Stage2B/Stage3 三层领域（topic_align，见上文修订版方案）** | | |
    | TOPIC_ALIGN_SUBFIELD | 0.65 | 仅 subfield 对齐时的层级分 |
    | TOPIC_ALIGN_FIELD | 0.35 | 仅 field 对齐时的层级分 |
    | TOPIC_WEIGHT_PRIMARY | 0.10 | primary 的 topic 权重 |
    | TOPIC_WEIGHT_DENSE | 0.25 | dense_expansion 的 topic 权重 |
    | TOPIC_WEIGHT_CLUSTER | 0.35 | cluster_expansion 的 topic 权重 |
    | TOPIC_WEIGHT_COOC | 0.25 | cooc_expansion 的 topic 权重 |
    | TOPIC_MIN_ALIGN | 0.20 | expansion 低对齐软门限（仅 expansion 生效） |
    | TOPIC_LOW_ALIGN_PENALTY | 0.50 | 低对齐 expansion 额外惩罚系数 |
    | **Stage3 身份闸门（按 term_role）** | | |
    | PRIMARY_MIN_IDENTITY_GATE | 0.62 | primary 通过身份闸门下限（可与 Stage2A 同值） |
    | DENSE_CLUSTER_MIN_IDENTITY_GATE | 0.45 | dense_expansion / cluster_expansion 身份分下限 |
    | COOC_MIN_IDENTITY_GATE | 0.35 | cooc_expansion 身份分下限 |
    | **Stage3 质量与最终** | | |
    | FINAL_MIN_TERM_SCORE | 0.15 | 最终 term 得分下限，低于则丢弃 |
    | PRIMARY_WEIGHT_MIN_RATIO | 0.50 | 最终候选中 primary 贡献占总 term 权重的比例下限 |
    | **已有/沿用** | | |
    | SIMILAR_TO_TOP_K | 3 | 每锚点 Stage2A 跨类型 SIMILAR_TO 取学术词数 |
    | SIMILAR_TO_MIN_SCORE | 0.65 | 跨类型 SIMILAR_TO 边权重下限 |

  - **三、文件级改动顺序表（替换/保留/谁调用谁）**  
    按「先改谁、再改谁」列出每个文件的改动类型与挂载点，实现时按表施工即可。

    | 文件 | 改动类型 | 具体说明 |
    |------|----------|----------|
    | **stage2_expansion.py** | 新增 + 改写 | **新增** `stage2_generate_academic_terms(prepared_anchors, active_domains)` 为 Stage2 总入口；入口由 label_path 先调 Stage2A（SIMILAR_TO 落点）再调 Stage2B（dense/簇/共现补充）；不再用「从裸 anchor 直接 _expand_semantic_map」出 primary。 |
    | **label_expansion.py** | 新增 + 改写 | **新增** `retrieve_academic_term_by_similar_to(anchor)`（跨类型 SIMILAR_TO）；**新增** `collect_landing_candidates`（内部 similar_to + 可选 JD 向量）；**新增** `expand_from_vocab_dense_neighbors`、`expand_from_cluster_members`、`expand_from_cooccurrence_support`、`merge_primary_and_support_terms`。**不再**用 SIMILAR_TO 做学术词→学术词扩展。 |
    | **label_anchors.py / stage1_domain_anchors.py** | 新增 | **新增** `classify_anchor_type`、`expand_anchor_acronyms`、`normalize_anchor_term`、`prepare_anchor_candidates`；Stage1 输出由「裸 anchor_skills」改为 **List[PreparedAnchor]**，供 stage2_generate_academic_terms 消费。 |
    | **stage3_term_filtering.py** | 替换 + 新增 | **原** 单一总分入口改为：先 `passes_identity_gate` → 再 `passes_topic_consistency` → `score_term_expansion_quality` → `compose_term_final_score` → `apply_generic_task_penalty_if_needed` → 质量闸门；**新增** `diagnose_term_error_type`、`build_term_debug_record`；输入为 List[ExpandedTermCandidate]/TermCandidate，输出为 List[TermCandidate]（含 debug 字段）。 |
    | **term_scoring.py** | 拆分 + 新增 | **原** `calculate_final_weights` 拆成两块：**identity 相关**（供 Stage2A `score_academic_identity` 及 Stage3 身份闸门）、**quality 相关**（供 `score_term_expansion_quality`）；**新增** `score_term_expansion_quality`、`compose_term_final_score`；identity 权重占主导由 compose 中按 term_role 加权体现。 |
    | **advanced_metrics.py** | 职责明确 | 负责 **topic_fit**、**cooc_purity**、**resonance**、**span_penalty** 等，供 `score_term_expansion_quality` 调用；不再主导「身份判断」，仅提供质量子项。 |
    | **simple_factors.py** | 保留 + 限定 | 只保留轻量 **penalty / bonus**（如泛任务惩罚、覆盖归一等），不再主导身份判断；`apply_generic_task_penalty` 由 Stage3 在 compose 之后调用。 |

    **调用关系小结**：  
    `stage2_generate_academic_terms` → 对每个 anchor：`collect_landing_candidates`（内部 `retrieve_academic_term_by_similar_to` + 可选 JD 向量）→ `score_academic_identity` → `select_primary_academic_landings` → Stage2B：`expand_from_vocab_dense_neighbors` + `expand_from_cluster_members` + `expand_from_cooccurrence_support` → `merge_primary_and_support_terms` → 输出 List[ExpandedTermCandidate]（term_role：primary / dense_expansion / cluster_expansion / cooc_expansion）。  
    `stage3_filter_and_score_terms` → 对每条 term 依次：`passes_identity_gate` → `passes_topic_consistency` → `score_term_expansion_quality`（调 advanced_metrics）→ `compose_term_final_score`（调 term_scoring 的 identity/quality 融合）→ `apply_generic_task_penalty_if_needed`（调 simple_factors）→ 质量闸门 → `diagnose_term_error_type`、`build_term_debug_record`。

  - **四、实现细节契约（Stage 1→2 状态传递、三级领域公式、物理查询定位、主落点熔断）**  
    以下四条契约确保 Stage 2/3 与底层数据库及上层总控逻辑的物理对齐，编码阶段须严格遵循以实现「零误差」衔接。

    **1. 状态传递契约（Stage 1 → Stage 2）**  
    Stage 1 输出须从简单字符串列表升级为对象列表。`stage1_domain_anchors.py` 返回的 `Stage1Result` 必须包含如下精确结构的 `PreparedAnchor`（定义于 `src/core/recall/label_path.py` 或专用 types 文件）：

    ```python
    @dataclass
    class PreparedAnchor:
        anchor: str                 # 原始清洗后的词，如 "rrt"
        anchor_type: str            # acronym | canonical | application | task | unknown
        expanded_forms: List[str]   # 扩写结果，如 ["rapidly exploring random tree"]
        weight: float               # Stage 1 计算的原始权重
        domain_hints: Set[str]      # 该锚点命中的领域 ID 集合
    ```

    **2. 三级领域分布的数学对齐**  
    在 Stage 2 判定主领域（`aggregate_main_domain`）和 Stage 3 计算 `topic_fit` 时，`vocabulary_topic_stats` 表中的分布以 JSON 字符串存储的百分比形式存在，计算须遵循：

    $$TopicScore(z) = \sum_{t \in Candidates} (IdentityScore(t) \times P(Topic=z|t))$$

    其中 $P(Topic=z|t)$ 从数据库字段 `topic_dist` 中解析出对应 Topic ID 的数值。  
    **实现提示**：若某词在 `topic_dist` 中不存在，代码必须显式按优先级回退到 `subfield_dist` 或 `field_dist` 进行补全计算。

    **3. 物理数据库查询定位**  
    为避免在 `label_expansion.py` 中写错路径，检索方式的物理落点约定如下：

    | 检索方式 | 依赖索引 / 表 | 检索范围约束 |
    | :--- | :--- | :--- |
    | **跨类型 SIMILAR_TO** | Neo4j `(Vocabulary)-[SIMILAR_TO]->(Vocabulary)` | anchor 为 industry，目标为 concept/keyword；边权为带扩写向量相似度 |
    | **可选 JD 向量** | `VOCAB_INDEX_PATH` (Faiss) + JD 文本编码 | 用 JD 文本在词汇索引取 top-K 学术词作补充 |

    **4. Stage 2 主落点熔断逻辑（硬护栏）**  
    在实现 `select_primary_academic_landings` 时，必须在代码中加入以下**逻辑契约**：  
    若某锚点经跨类型 SIMILAR_TO（及可选 JD 向量）检索后，其最高分的 `identity_score` 仍低于 `PRIMARY_MIN_IDENTITY (0.62)`，则该锚点**必须**被标记为 `unresolved`。**禁止**为其生成任何 `dense_expansion`、`cluster_expansion` 或 `cooc_expansion`。此条为防止「语义漂移」的最有效关卡。

以下为整合后的 KGAT-AX v2 修改方案（已合并原 README 中多处重复与差异表述，保留全部细节）：
## KGAT-AX v2 修改方案


### 1. 改造背景
当前系统整体采用“两阶段推荐”架构：
多路召回阶段
包括：
向量语义召回（Vector Path）
标签路径召回（Label Path）
协作网络召回（Collaboration Path）
深度精排阶段
使用 KGAT-AX 对召回后的候选专家进行排序。
现有 KGAT-AX 版本已经具备以下能力：
利用知识图谱中的高阶关系进行图表示学习；
融合部分作者学术指标特征；
对候选作者进行二次排序。
但在当前阶段，仍存在以下问题：
#### 1.1 缺乏真实监督数据
目前缺少大规模、标准化的真实标签数据，例如：
岗位 -> 最终录用专家
岗位 -> 面试通过专家
岗位 -> 专家人工排序结果
这意味着模型无法直接学习真实招聘偏好。
1.2 学术硬指标利用不足
现有版本对以下信号利用仍偏弱：
H-index
作者总被引数
近年学术活跃度
机构权威性
期刊 / 会议 / 渠道权威性
代表作影响力
这会导致模型在排序时更偏向“语义相似”，而不足以体现“高端科技人才”的真实筛选逻辑。
1.3 图结构优势未被充分释放
若仅用简单内积或浅层特征融合，KGAT 的高阶关系传播优势难以充分发挥，容易退化为“特征回归模型”。

2. 改造目标
KGAT-AX v2 的核心目标是：
在保留 KGAT 图传播主干的前提下，将“图结构关系”“作者学术指标”“岗位-作者匹配特征”“弱监督 teacher 信号”统一纳入精排模型，使其能够在无真实标签条件下稳定训练，并具备更强的指标感与解释性。
具体目标如下：
#### 2.1 保留 KGAT 主干
不推翻现有 KGAT / GNN 主体，而是在其基础上增强输入与排序头。
#### 2.2 引入显式作者指标分支
将以下作者特征显式接入模型：
H-index
总被引数
总论文数
近五年论文数
近五年被引数
代表作被引数
机构权威性
渠道权威性
#### 2.3 引入岗位-作者匹配特征分支
显式建模 (Query, Author) 对应关系，包括：
召回来源信息
召回分数
岗位与作者的语义相似度
领域重合度
Topic 重合度
#### 2.4 在无真实标签条件下进行弱监督训练
通过规则教师（teacher score）与分层伪标签构造训练信号，使模型能够在没有真实招聘标签时仍然稳定学习。
#### 2.5 保持工程可部署性
训练时与推理时使用同构的在线可复算特征，避免线上线下特征偏移问题。

### 3. 总体设计思路

KGAT-AX v2 不再是简单的“图嵌入内积排序模型”，而是升级为：
图表示分支 + 作者指标分支 + Query-Author 匹配分支 + 弱监督训练目标
总体流程如下：
Query / JD
↓
三路召回（Vector / Label / Collaboration）
↓
合并候选 Author
↓
构造 Query-Author Pair 样本
↓
生成：
- Query 子图关系
- Author 辅助特征（author_aux）
- Query-Author 在线特征（pair_aux_online）
- Teacher-only 特征（pair_aux_teacher_only）
↓
KGAT 图传播得到 Query / Author 图嵌入
↓
图嵌入 + author_aux + pair_aux_online 融合打分
↓
使用 teacher_score / label_level / negative_type 训练
↓
输出最终精排分数


### 4. Query-aware 图输入改造

为了让 KGAT-AX 更适合岗位推荐任务，v2 版本中将岗位 JD 显式作为 Query 节点接入图谱。

#### 4.1 新增 Query 节点
每个岗位 JD 视为一个独立 Query 节点。
#### 4.2 Query 节点的关系设计
建议至少新增以下关系：
QUERY_HAS_TOPIC -> Topic
QUERY_REQUIRES_SKILL -> Skill
QUERY_RECALLS_AUTHOR -> Author
其中：
QUERY_HAS_TOPIC
表示岗位文本通过 Topic 映射后对应的研究主题。
QUERY_REQUIRES_SKILL
表示岗位中抽取出的关键技能、方法、研究方向术语。
QUERY_RECALLS_AUTHOR
表示该作者是通过召回阶段进入候选池的。
可进一步根据来源细分为：
QUERY_RECALLS_AUTHOR_VECTOR
QUERY_RECALLS_AUTHOR_LABEL
QUERY_RECALLS_AUTHOR_COLLAB
也可以保留单一关系并在 pair 特征中记录来源。
#### 4.3 Query-aware 子图的意义
这样设计的作用是：
让模型显式理解“岗位需求”；
让排序过程具备上下文条件；
让作者排序不再是“全局静态强者排序”，而是“针对当前岗位的条件排序”。

### 5. 输入特征改造

KGAT-AX v2 的输入特征分为三类：Query 侧特征、Author 侧特征、Query-Author Pair 特征。

#### 5.1 Query 侧特征
每个 Query 至少包含：
query_id
query_text
domain_ids
core_terms
topic_ids
query_embedding
用途包括：
生成 Query 节点的图关系；
计算 Query-Author 匹配特征；
训练与推理时统一使用。

#### 5.2 Author 侧特征（author_aux）
这部分是 KGAT-AX v2 的重要增强分支。
建议固定列顺序如下：
AUTHOR_AUX_COLUMNS = [
"author_h_index_log",
"author_citations_log",
"author_works_log",
"author_recent_5y_works_log",
"author_recent_5y_citations_log",
"best_paper_citations_log",
"top3_paper_citations_mean_log",
"paper_recency_score",
"best_institution_authority",
"avg_institution_authority",
"best_source_authority",
"avg_source_authority"
]

字段说明
author_h_index_log：作者 H-index 取 log(1+x) 后的值
author_citations_log：作者总被引数取 log(1+x)
author_works_log：作者论文总量取 log(1+x)
author_recent_5y_works_log：作者近五年论文数取 log(1+x)
author_recent_5y_citations_log：作者近五年被引数取 log(1+x)
best_paper_citations_log：代表作被引数取 log(1+x)
top3_paper_citations_mean_log：Top3 代表作平均被引取 log(1+x)
paper_recency_score：近年代表作时新性评分
best_institution_authority：作者所关联最强机构的权威性分数
avg_institution_authority：作者机构整体平均权威性
best_source_authority：作者所发最强渠道的权威性分数
avg_source_authority：作者渠道整体平均权威性
设计原则
计数类特征全部先做 log(1+x) 变换
权威性特征尽量归一化到 [0,1]
author_aux 属于静态/半静态特征，可离线预生成

#### 5.3 Query-Author Pair 特征
Pair 特征必须分成两类：
A. Online Pair Features（线上可复算）
这些特征允许进入模型推理阶段。
建议包括：
ONLINE_PAIR_FEATURES = [
"from_vector",
"from_label",
"from_collab",
"vector_rank_norm",
"label_rank_norm",
"collab_rank_norm",
"vector_score",
"label_score_raw",
"collab_score",
"recall_count",
"jd_author_semantic_sim",
"domain_overlap_score",
"topic_overlap_score",
]

B. Teacher-only Features（仅 teacher 使用）
这些特征仅参与规则教师打分，默认不直接输入线上模型：
TEACHER_ONLY_FEATURES = [
"jd_best_paper_sim",
"skill_hit_count",
"skill_hit_ratio",
]

为什么要拆成两类
这是为了避免两个问题：
线上线下特征不一致
某些重计算特征训练时算得出来，但线上无法实时复现。
Teacher Collapse
如果 teacher 使用的全部特征都原样喂给模型，模型极易退化成“teacher 公式拟合器”。

### 6. 弱监督训练数据设计

由于缺乏真实标签，KGAT-AX v2 使用规则教师构造弱监督训练目标。

#### 6.1 Teacher Score
对每个 (Query, Author)，计算一个规则教师分数 teacher_score。
建议总体形式为：
[
teacher_score
0.46 \cdot relevance
0.24 \cdot impact
0.18 \cdot authority
0.12 \cdot recency
]

#### 6.2 relevance（匹配度）
反映岗位与作者的语义与主题匹配情况。
可由以下信号构成：
vector_score
label_score_raw
collab_score
jd_author_semantic_sim
domain_overlap_score
topic_overlap_score
jd_best_paper_sim
skill_hit_ratio
建议形式如下：
relevance_score = (
0.22 * vector_score +
0.16 * label_score_raw +
0.06 * collab_score +
0.12 * jd_author_semantic_sim +
0.08 * domain_overlap_score +
0.06 * topic_overlap_score +
0.15 * jd_best_paper_sim +
0.15 * skill_hit_ratio
)


#### 6.3 impact（学术影响力）
反映作者学术产出与影响力。
建议由以下信号构成：
author_h_index_log
author_citations_log
best_paper_citations_log
top3_paper_citations_mean_log
author_recent_5y_citations_log

#### 6.4 authority（权威性）
反映作者所属机构与发表渠道的可信度与顶尖程度。
建议由以下信号构成：
best_institution_authority
best_source_authority

#### 6.5 recency（时新性）
反映作者近年是否仍持续产出高质量成果。
建议由以下信号构成：
paper_recency_score
author_recent_5y_works_log

### 7. 伪标签设计

#### 7.1 分层标签 label_level
根据 teacher_score 将样本划分为 4 层：
3：strong positive
2：positive
1：neutral
0：negative
建议阈值：
if teacher_score >= 0.82:
label_level = 3
elif teacher_score >= 0.68:
label_level = 2
elif teacher_score >= 0.52:
label_level = 1
else:
label_level = 0


#### 7.2 负样本类型 negative_type
为了提升排序能力，将负样本进一步细分为：
easy
semi_hard
hard
confusing
easy
随机弱相关样本，通常既不匹配主题，也不具备强学术影响力。
semi_hard
进入过召回候选池，但整体得分偏低。
hard
学术指标较强（如 H-index 高），但岗位主题不匹配。
confusing
机构或渠道权威性较高，但语义匹配较弱，容易干扰排序。

#### 7.3 样本权重 sample_weight
权重设计要保守，避免过早对 hard negative 施加强压制。
建议初始方案：
strong positive：1.4
positive：1.2
其他：1.0
不要一开始给 hard negative 过大权重。
hard negative 的主要价值应体现在采样概率上，而不是 loss 放大倍数上。

---

### 排序原则补充：Relevance-first 与 Active Researcher 优先

本系统的最终目标，**不是**输出“全局学术影响力最高的人”，也**不是**输出“指标最华丽的人”，而是：

> **在与当前岗位高度相关的候选中，优先找出最适合当前岗位、且近期仍然活跃的研究者。**

因此，KGAT-AX v2 的排序目标必须遵循 **Relevance-first** 原则，并在此基础上兼顾学术实力与近期活跃度。

#### 1. 排序目标的正式定义

对于岗位推荐场景，最终排序逻辑定义为：

1. **先保证岗位相关性足够高**  
   候选人必须在研究主题、代表作方向、关键词路径、标签图谱路径等方面，与当前岗位需求形成稳定对应关系。

2. **再在“已相关”的候选中比较综合价值**  
   当多个候选都已满足岗位相关性要求后，再利用：学术影响力（impact）、权威性（authority）、近期活跃度（recency / recent productivity）来拉开排序差距。

3. **优先保留“当前仍在做这件事”的研究者**  
   对于与岗位相关但长期未持续产出的作者，应低于同等相关、但近几年持续发表和持续被引用的作者。

因此，本系统默认追求的是：

> **Most relevant active researchers for the current job**  
> 而不是  
> **Most globally prestigious scholars regardless of job fit**

#### 2. Relevance 是硬前置，而不是普通特征

在本项目中，`relevance` 不是一个可以被其他高指标替代的普通加分项，而是**硬前置约束**。

* 不能因为某作者 `h-index` 很高、总引用很多，就弥补其与岗位主题不匹配的问题；
* 不能因为某作者机构很强、论文很多，就压过一个研究方向高度对口且近期持续活跃的候选人；
* 模型必须首先回答“这个人是否真的适合这个岗位要做的研究”。

排序逻辑原则：

* **相关性不过线，最终总分必须被压制**
* **高 impact / high authority 只能在 relevance 达标后发挥作用**
* **任何与岗位主题明显错位的候选，都不应进入 Top ranks**

形式化表达：

```text
FinalScore = ValueScore × RelevanceGate
```

其中：`ValueScore` 为作者的综合人才价值分数，`RelevanceGate` 为由岗位相关性决定的门控因子。若 `relevance` 明显不足，则即使 `ValueScore` 很高，也不能进入最终前列。

#### 3. teacher_score 的目标含义

在 KGAT-AX v2 中，`teacher_score` 不再表示“作者是否学术上很强”，而表示：

> **该作者作为当前岗位候选人的综合优先级**

它应综合体现四类信号：**relevance**（与岗位需求、研究主题、技能标签、语义表达的匹配程度）、**impact**（学术影响力与代表作质量）、**authority**（机构、期刊/会议、学术地位等）、**recency**（近几年是否仍持续活跃、持续产出、持续被关注）。默认优先级为：

```text
relevance  >  recency  ≈  impact  >  authority
```

* `relevance` 决定“是不是对的人”；`recency` 决定“是不是现在还在做这件事的人”；`impact` 决定“是不是做得足够强的人”；`authority` 决定“其背景与外部认可度是否进一步增强可信度”。**authority 只能增强排序可信度，不能主导排序本身**。

#### 4. 为什么要强调 Active Researcher 优先

本项目面向科技岗位推荐。对于岗位而言，更需要的是：当前仍在该方向工作的人、近几年持续有成果输出的人、近期仍在该研究主题上有论文/项目/引用反馈的人。因此：

**应优先推荐**：方向高度对口；近 3–5 年仍持续发表相关工作；近期仍有高质量成果或持续引用增长；代表作与岗位任务直接相关。

**不应仅因“资历老”而排前**：早年做过相关工作但近年已明显转向或停滞；总指标很高但当前研究重心与岗位不一致；权威性很强但实际岗位匹配度一般。

本系统更偏好 **“现在还在这个方向上持续做事的人”**，而不是 **“曾经在这个方向上很强的人”**。

#### 5. 对 teacher_score 的结构性约束

* **5.1 relevance-first**：relevance 低于阈值时总分必须受限；relevance 不足时 impact 不得强行补分，authority 不得替代 topic fit。
* **5.2 active over inactive**：在相关性相近时，近期持续活跃的作者应优于长期沉寂的作者（可用 recent_3y/5y_works、recent_3y/5y_citations、recent/topic-aligned papers ratio 等）。
* **5.3 strong over merely famous**：在相关性和活跃度接近时，再用 h-index、total citations、top paper citations、venue/institution prestige 等拉开差距。
* **5.4 authority is supplementary**：机构/来源权威性只能作为增强项，不得成为压倒性主项。

#### 6. 正样本定义也要服从该原则

**Strong Positive**：relevance 明显达标；至少来自 2 路召回的一致支持或在单路中具有极强证据；代表作与岗位主题显著对齐；不属于明显 topic mismatch；最近几年仍有该方向的持续产出。

**Positive**：满足基本相关性，但在活跃度、影响力或证据强度上略弱于 strong positive。

目的是让模型学习 **“什么样的人是适合当前岗位的活跃研究者”**，而不是仅学习 **“什么样的人在 teacher 公式里得分高”**。

#### 7. 线上推理阶段的排序解释口径

推荐理由应围绕：**为什么与岗位相关**（命中的技能、对齐的主题/标签路径、对口的代表论文、图谱路径证据）；**为什么说明他现在还活跃**（近 3–5 年持续发表、最近相关方向有稳定产出）；**为什么说明他值得排前**（代表作质量高、近年活跃且方向稳定、学术影响力与岗位需求匹配）。避免只展示“总引用高”“h-index 高”“机构强”，而应优先展示“对口”“活跃”“有代表作”“当前仍值得联系/引入”。

#### 8. 离线评估也应围绕该目标设计

* **Relevance-first 检查**：Top-K 中是否仍出现大量主题明显不对口的作者。
* **Active Researcher 检查**：在 relevance 接近的候选中，是否优先排出了近几年持续产出的作者。
* **Famous-but-off-target 抑制检查**：是否成功抑制了“总体指标很强但岗位主题并不对口”的作者进入前列。
* **Young / Rising Researcher 保留检查**：是否能在不牺牲相关性的前提下，保留近年快速成长、持续活跃的高潜候选。

#### 9. 本项目的默认排序哲学

> **先找真正对口的人，再在对口的人里优先找当前仍活跃、且综合实力更强的人。**

核心不是“谁最有名/谁总指标最高/谁资历最深”，而是：**谁最适合当前岗位**、**谁现在还在持续做这个方向**、**谁最值得被优先推荐**。

---

### teacher_score 建议形式

为贯彻 **Relevance-first + Active Researcher 优先**，KGAT-AX v2 中的 `teacher_score` 建议采用**两段式结构**：先做 relevance 门控，再做综合价值排序。

#### 1. 总体形式

```text
teacher_score = relevance_gate(relevance) × value_score
```

* `relevance`：岗位与作者之间的主题/技能/语义/图谱匹配程度  
* `relevance_gate(.)`：相关性门控函数  
* `value_score`：在“已经对口”的前提下，对作者综合价值的打分  

relevance 不够时，后面的高 impact / 高 authority 都不能强行补上来；relevance 达标后，再用活跃度、影响力、权威性去拉开层次。

#### 2. value_score 的建议结构

```text
value_score = w_r·relevance + w_a·activity + w_i·impact + w_u·authority
```

默认权重关系：`w_r > w_a >= w_i > w_u`。推荐的默认起点：`w_r=0.46`，`w_a=0.24`，`w_i=0.20`，`w_u=0.10`。对应解释：relevance 决定“是否对口”（权重最大）；activity 决定“是否仍在当前方向活跃”；impact 决定“做得是否足够强”；authority 只作增强项，不允许主导排序。

#### 3. relevance_gate 的建议形式

**3.1 分段门控（推荐）**

```text
if relevance < 0.35:  gate = 0.20
elif relevance < 0.45: gate = 0.45
elif relevance < 0.55: gate = 0.70
elif relevance < 0.65: gate = 0.88
else:                 gate = 1.00
teacher_score = gate × value_score
```

含义：relevance < 0.35 明显不对口直接压死；0.35~0.45 可疑候选只保留少量分数；0.45~0.55 弱相关允许存在但难进前排；0.55~0.65 较强相关开始正常竞争；≥0.65 强相关完全放开。

**3.2 平滑 Sigmoid 门控**：`gate = sigmoid(alpha × (relevance - tau))`，如 `sigmoid(10×(relevance-0.52))`，更平滑可微，可解释性略差，初版建议先用分段门控。

**3.3 上限封顶式门控**：低相关时对 `teacher_score` 设上限（如 relevance<0.40 时 `teacher_score = min(raw_value_score, 0.30)`），可作为 debug/ablation 版本。

#### 4. 各子分数的建议定义

* **relevance**（归一到 [0,1]）：回答“这个作者是不是在做当前岗位要做的方向”。建议融合 semantic_match、topic_match、representative_work_match、graph_alignment，例如 `0.40*semantic_match + 0.25*topic_match + 0.20*representative_work_match + 0.15*graph_alignment`。  
* **activity**：回答“这个人是不是现在还在做这件事”。建议 `0.40*recent_topic_productivity + 0.30*recent_topic_citation_signal + 0.20*continuity + 0.10*growth_trend`。关键不是“最近发了多少篇总论文”，而是最近是否仍在当前岗位对应方向上持续产出。  
* **impact**：回答“这个人做得强不强”。建议 `0.35*normalized_h_index + 0.25*normalized_total_citations + 0.25*top_paper_strength + 0.15*recent_impact_quality`，做 log1p 与分位数归一。  
* **authority**：回答“外部背书是否增加可信度”。建议 `0.45*institution_strength + 0.35*venue_strength + 0.20*collaboration_prestige`。仅作增强项，不允许机构/venue prestige 压过 relevance。

#### 5. 建议的最终公式

```text
value_score = 0.46*relevance + 0.24*activity + 0.20*impact + 0.10*authority

relevance_gate(relevance) =
  0.20  if relevance < 0.35
  0.45  if 0.35 <= relevance < 0.45
  0.70  if 0.45 <= relevance < 0.55
  0.88  if 0.55 <= relevance < 0.65
  1.00  if relevance >= 0.65

teacher_score = relevance_gate(relevance) * value_score
```

业务解释：relevance 决定能不能上桌；activity 决定是不是“当前还在做”；impact 决定是不是“做得强”；authority 只负责锦上添花。

#### 6. pair label 的建议划分规则

```text
if relevance < 0.35:                pair_label = 0   # clear negative
elif teacher_score >= 0.78 and activity >= 0.55: pair_label = 3   # strong positive
elif teacher_score >= 0.62 and relevance >= 0.55: pair_label = 2   # positive
elif relevance >= 0.45:             pair_label = 1   # weak positive / borderline
else:                                pair_label = 0
```

Strong positive 必须既相关又活跃；不能让“历史上很强但现在不活跃”的作者轻易进入最高正样本层。

#### 7. negative_type 的建议补充

* **easy_negative**：语义远、图谱远、topic 不相关。  
* **hard_negative**：语义接近，但 topic / representative works 不对口。  
* **famous_but_off_target**：impact 很高、authority 很强，但 relevance 不足。  
* **inactive_but_relevant**：topic 曾经相关，但 recent activity 很弱。  
* **confusing_same_cluster**：相邻 topic/同簇词，但岗位任务并不一致。  

用于显式教模型区分“强但不对口”“对口但不活跃”“相似但不是这个岗位要的人”。

#### 8. 伪代码骨架

**8.1 teacher_score 构造**

```python
def build_teacher_score(job, author, pair_features, author_features):
    # 1. relevance
    relevance = (
        0.40 * pair_features["semantic_match"]
        + 0.25 * pair_features["topic_match"]
        + 0.20 * pair_features["representative_work_match"]
        + 0.15 * pair_features["graph_alignment"]
    )
    # 2. activity
    activity = (
        0.40 * author_features["recent_topic_productivity"]
        + 0.30 * author_features["recent_topic_citation_signal"]
        + 0.20 * author_features["continuity"]
        + 0.10 * author_features["growth_trend"]
    )
    # 3. impact
    impact = (
        0.35 * author_features["normalized_h_index"]
        + 0.25 * author_features["normalized_total_citations"]
        + 0.25 * author_features["top_paper_strength"]
        + 0.15 * author_features["recent_impact_quality"]
    )
    # 4. authority
    authority = (
        0.45 * author_features["institution_strength"]
        + 0.35 * author_features["venue_strength"]
        + 0.20 * author_features["collaboration_prestige"]
    )
    # 5. value_score
    value_score = 0.46*relevance + 0.24*activity + 0.20*impact + 0.10*authority
    # 6. relevance gate（分段门控）
    if relevance < 0.35:   gate = 0.20
    elif relevance < 0.45: gate = 0.45
    elif relevance < 0.55: gate = 0.70
    elif relevance < 0.65: gate = 0.88
    else:                  gate = 1.00
    teacher_score = gate * value_score
    return {"teacher_score": teacher_score, "relevance": relevance, "activity": activity, "impact": impact, "authority": authority, "gate": gate}
```

**8.2 label 生成**

```python
def build_pair_label(score_dict):
    relevance = score_dict["relevance"]
    activity = score_dict["activity"]
    teacher_score = score_dict["teacher_score"]
    if relevance < 0.35: return 0
    if teacher_score >= 0.78 and activity >= 0.55: return 3
    if teacher_score >= 0.62 and relevance >= 0.55: return 2
    if relevance >= 0.45: return 1
    return 0
```

**8.3 negative_type 生成**

```python
def build_negative_type(score_dict, pair_features, author_features):
    relevance = score_dict["relevance"]
    activity = score_dict["activity"]
    impact = score_dict["impact"]
    semantic_match = pair_features["semantic_match"]
    topic_match = pair_features["topic_match"]
    if relevance < 0.25 and semantic_match < 0.30: return "easy_negative"
    if relevance < 0.45 and impact > 0.70: return "famous_but_off_target"
    if relevance >= 0.50 and activity < 0.30: return "inactive_but_relevant"
    if semantic_match >= 0.60 and topic_match < 0.40: return "hard_negative"
    return "confusing_same_cluster"
```

#### 9. 训练时的配合方式

* **用作 soft target**：让 student 学习连续排序信号，而不是只学二元标签。  
* **用作 label 分层依据**：生成 strong positive / positive / weak positive / negative。  
* **用作 sample weighting 依据**：strong positive 权重更高；famous_but_off_target 等 hard negative 权重更高；easy_negative 权重适中。  
* **不直接作为线上推理公式照抄**：线上最终分数由模型输出主导；teacher 的作用是给模型一个符合业务目标的学习方向，而不是在线上永久替代模型。

#### 10. README 总结

> KGAT-AX v2 中的 teacher_score 不是“谁学术最强”的分数，而是“谁在当前岗位语境下更值得优先推荐”的分数；其核心原则是：**先保证对口，再在对口候选中优先选择当前仍活跃、且综合实力更强的研究者。**

---

### Loss 设计：Ranking + Classification + Distillation 的联合优化

KGAT-AX v2 的训练目标应同时回答三个问题：**谁应该排在更前面**（岗位候选之间的相对排序）；**这个岗位–作者 pair 属于哪一层匹配强度**（clear negative / weak positive / positive / strong positive）；**student 是否学到了符合业务目标的连续价值信号**（“最适合当前岗位的活跃研究者”的排序哲学）。因此 loss 采用三部分联合优化：

```text
L_total = λ_rank * L_rank + λ_cls * L_cls + λ_distill * L_distill
```

其中：**L_rank** 排序损失，负责学“谁应该更靠前”；**L_cls** 分层分类损失，负责学“这个 pair 属于哪一档”；**L_distill** 蒸馏损失，负责让 student 对齐 teacher 的连续价值判断。默认排序是主任务，分类和蒸馏是辅助稳定器。

#### 1. 总体设计原则

* **1.1 排序优先**：最终输出是 Top-K 推荐名单，训练时必须显式优化排序，而不是只做 pair classification。  
* **1.2 分类用于稳边界**：分类头帮助模型区分强正样本、一般正样本、边界样本、明确负样本。  
* **1.3 蒸馏用于传递业务偏好**：把 relevance-first、active researcher 优先、famous-but-off-target 抑制等传给 student。  
* **1.4 防止 student 退化成 teacher 公式复读机**：蒸馏为辅损失，配合 online/teacher-only 特征拆分、warmup、graph-only head、aux dropout/random mask。

#### 2. 排序损失 L_rank

主损失，解决“同岗位下为什么作者 A 应排在作者 B 前面”。

**2.1 推荐形式：Pairwise Margin Ranking Loss**

```text
L_rank = max(0, margin - (s_pos - s_neg))
```

s_pos/s_neg 为正/负样本分数，margin 为安全间隔。正样本已明显高于负样本则不惩罚；否则继续优化。推荐 **margin = 0.20**（可试 0.15～0.25）。

**2.2 排序对的构造原则**（同 job 内优先）：A. strong positive vs negative；B. positive vs negative；C. strong positive vs positive；D. positive vs weak positive；E. relevant but inactive vs active relevant；F. famous-but-off-target vs truly relevant。E、F 对业务目标特别重要，采样阶段显式保留。

**2.3 排序损失加权**（pair type 不同权重）：strong_positive vs famous_but_off_target → 1.50；strong_positive vs inactive_but_relevant → 1.40；strong_positive vs hard_negative → 1.25；positive vs hard_negative → 1.10；positive vs easy_negative → 1.00；weak_positive vs easy_negative → 0.70。

#### 3. 分类损失 L_cls

4 类：0=negative，1=weak positive/borderline，2=positive，3=strong positive。`L_cls = CrossEntropy(logits_cls, pair_label)`。分类头帮助区分明确错配、弱相关、强匹配、强正样本。**类别权重建议**：weight(strong_positive) > weight(positive) > weight(weak_positive) > weight(negative)，例如 `{0:1.00, 1:1.15, 2:1.35, 3:1.60}`。**negative_type 对分类的额外加权**：easy_negative 1.00、hard_negative 1.20、famous_but_off_target 1.35、inactive_but_relevant 1.25、confusing_same_cluster 1.15。最终 `L_cls = sample_weight * CrossEntropy(logits_cls, pair_label)`。

#### 4. 蒸馏损失 L_distill

让 student 对齐 teacher 的连续判断（relevance-first、active 优先、强但 off-target 压制等）。**推荐形式**：`L_distill = SmoothL1(score_pred, teacher_score)`（首版推荐 SmoothL1，比 MSE 更稳）。蒸馏不是主损失，只占辅助地位。**Distill Warmup**：前期适度依赖蒸馏稳边界，后期逐步降低权重。例如 `λ_distill(epoch)`：epoch 1–2 → 0.20，3–5 → 0.12，6+ → 0.06。

#### 5. 总损失权重与 graph-only 配合

**默认**：`λ_rank > λ_cls >= λ_distill`。首版建议：`λ_rank = 1.00`，`λ_cls = 0.50`，`λ_distill = 0.15`。保留 **graph_only_head**，定义 `L_graph = ranking_loss(graph_only_score, ranking_pairs)`，总损失扩展为 `L_total = λ_rank*L_rank + λ_cls*L_cls + λ_distill*L_distill + λ_graph*L_graph`，建议 **λ_graph = 0.12**（0.10～0.20），迫使图分支保留表达能力。

#### 6. sample_weight 总原则

```text
sample_weight = label_weight × negative_type_weight × confidence_weight × optional_path_consistency_weight
```

label_weight 强调 strong/positive；negative_type_weight 强调 famous_but_off_target、inactive_but_relevant、hard_negative；confidence_weight 按 teacher 证据充分度；path_consistency_weight 可对多路召回支持的样本适度提高。

#### 7. 推荐的最小可运行版 Loss

```text
L_total = 1.00*L_rank + 0.50*L_cls + 0.15*L_distill
```

L_rank：pairwise margin ranking；L_cls：4 类交叉熵；L_distill：SmoothL1(score_pred, teacher_score)。

#### 8. 伪代码骨架

**8.1 主损失计算**

```python
def compute_total_loss(batch, model_outputs, config):
    score_pred = model_outputs["score_pred"]
    logits_cls = model_outputs["logits_cls"]
    teacher_score = batch["teacher_score"]
    pair_label = batch["pair_label"]
    sample_weight = batch["sample_weight"]
    ranking_pairs = batch["ranking_pairs"]
    margin = config.rank_margin

    rank_losses = []
    for i, j, pair_weight in ranking_pairs:
        loss_ij = max(0.0, margin - (score_pred[i] - score_pred[j]))
        rank_losses.append(pair_weight * loss_ij)
    L_rank = mean(rank_losses) if rank_losses else 0.0

    cls_loss_each = cross_entropy_per_sample(logits=logits_cls, targets=pair_label, class_weights=config.class_weights)
    L_cls = mean(sample_weight * cls_loss_each)

    distill_each = smooth_l1_per_sample(score_pred, teacher_score)
    L_distill = mean(sample_weight * distill_each)

    L_graph = 0.0
    if "graph_only_score" in model_outputs and config.use_graph_aux_loss:
        graph_score = model_outputs["graph_only_score"]
        graph_losses = [max(0.0, margin - (graph_score[i] - graph_score[j])) for i, j, pw in ranking_pairs]
        L_graph = mean(graph_losses) if graph_losses else 0.0

    total_loss = config.lambda_rank*L_rank + config.lambda_cls*L_cls + config.lambda_distill*L_distill + config.lambda_graph*L_graph
    return {"loss": total_loss, "L_rank": L_rank, "L_cls": L_cls, "L_distill": L_distill, "L_graph": L_graph}
```

**8.2 ranking_pairs 构造**（同一 job 下）：仅当 a 明显优于 b 时构造；按 pair_label 与 negative_type 赋予 pair_weight（strong_positive vs famous_but_off_target → 1.50，vs inactive_but_relevant → 1.40 等）；同 label 内若 teacher_score 差 > 0.08 可构造细排序对，权重约 0.60。

**8.3 蒸馏权重调度**：`get_lambda_distill(epoch)`：epoch≤2 → 0.20，3–5 → 0.12，6+ → 0.06。

#### 9. 训练监控指标建议

排序：pairwise ranking accuracy、top-k hit、strong positive ahead rate、famous-but-off-target / inactive-but-relevant suppression rate。分类：4-class accuracy、strong positive recall、negative precision、confusion matrix。蒸馏：teacher-student spearman、MSE/SmoothL1、high-score region alignment。重点看：强正样本是否被推前、famous-but-off-target 是否被压下、active relevant 是否优于 inactive relevant。

#### 10. 本小节总结

> KGAT-AX v2 的 loss 设计以排序为主、分类为辅、蒸馏为稳定器；其目标不是单纯拟合标签，而是让模型学会：在同一岗位下，优先把真正对口且当前仍活跃的研究者排在前面。

---

### 训练样本构造与采样策略

KGAT-AX v2 的效果还取决于：训练样本是否体现业务目标、正负边界是否清晰、难例是否充分、同岗位内是否形成有效排序信号。目标不是“尽可能多造 pair”，而是让模型看到：同岗位下谁是真正对口且仍活跃的研究者，谁只是看起来强但并不适合。样本构造围绕三类监督组织：**连续监督** teacher_score、**分层监督** pair_label、**排序监督** 同岗位 ranking_pairs。

#### 1. 样本构造的总体原则

* **1.1 以 job 为中心**：围绕单个岗位下的一组候选作者组织，而不是脱离岗位做全局作者分类。  
* **1.2 先召回，再精标**：训练样本来源来自多路召回结果，而非全库盲抽。  
* **1.3 既要有强正样本，也要有业务难负样本**：充分包含 strong but off-target、relevant but inactive、same-cluster confusing。  
* **1.4 分布贴近线上但适度重采样**：让关键边界更清晰。

#### 2. 训练样本的基本组织单位

一个 **job → 一组 candidate authors → 每个 candidate 附带 teacher/label/features**。每个 job-batch 至少包含：job_id、job_text/embedding、candidate_authors（每人含 teacher_score、pair_label、negative_type、pair_aux_online、author_aux、图输入）、以及由该池派生的 **ranking_pairs**。天然适配排序任务与“一个岗位 → 一个推荐名单”的线上形态。

#### 3. 候选作者池的来源

合并三路召回：**Vector Path**（语义相近）、**Label Path**（标签/topic/技能路径）、**Collaboration Path**（合作网络）。`candidate_pool(job) = union(vector_candidates, label_candidates, collaboration_candidates)`，保留每人的 source_flags（from_vector、from_label、from_collab、multi_source_count）。

#### 4. 正样本定义策略

**Strong Positive**：高相关 + 当前活跃 + 结构证据强 + 排名价值高；至少多数条件满足（relevance 达标、activity 达标、至少 2 路支持或一路极强、代表作对齐、非 topic mismatch）。**Positive**：明确相关、有一定活跃度或影响力，证据弱于 strong positive。**Weak Positive/Borderline**：有一定相关性但证据不稳或活跃度偏弱，用于教模型认识“边界相关”和构造 positive vs weak positive 排序对。

#### 5. 负样本定义策略

**Easy Negative**：语义远、topic 不对、图谱路径弱。**Hard Negative**：语义/技能局部接近但 topic/representative works/graph alignment 不成立。**Famous-but-off-target**：impact/authority 高但 relevance 不足，最关键的一类负样本之一。**Inactive-but-relevant**：topic 曾相关、历史成果不错但近几年活跃度弱。**Confusing Same-Cluster**：相邻 topic/词面像但任务语义不一致。

#### 6. 候选池大小与采样比例建议

每 job 保留 **40～120** 个 candidate authors，推荐 **target_pool_size = 80**。单岗位内比例建议：strong_positive : positive : weak_positive : negative = 1 : 2 : 2 : 4。negative 内部分层：easy : hard : famous_off_target : inactive_relevant : confusing_cluster ≈ 2 : 2 : 1 : 1 : 1。

#### 7. ranking_pairs 的构造优先级

一级：Strong Positive vs Famous-but-off-target；二级：Strong Positive vs Inactive-but-relevant；三级：Strong Positive vs Hard Negative；四级：Positive vs Hard Negative；五级：Strong Positive vs Positive；六级：Positive vs Weak Positive；七级：同层内 teacher_score 差 > delta 的细排序对（delta 约 0.08～0.12）。**配对数量控制**：每 strong positive 配 2～4 个 famous_off、2～4 个 inactive_relevant、2～4 个 hard_negative、1～2 个 positive；每 positive 配 2～3 个 hard_negative、1～2 个 weak_positive。**target_pairs_per_job ≈ 30～120**，默认可取 64。

#### 8. sample_weight 构造

`sample_weight = label_weight × type_weight × confidence_weight × multi_source_weight`。**label_weight**：strong_positive 1.50、positive 1.20、weak_positive/negative 1.00。**type_weight**：famous_but_off_target 1.35、inactive_but_relevant 1.25、hard_negative 1.15、confusing 1.10、easy 1.00。**confidence_weight**：teacher 证据充分则略提、边界模糊则略降。**multi_source_weight**：1 source 1.00、2 sources 1.08、3 sources 1.15。

#### 9. 训练/验证切分与长尾平衡

**按 job 切分**：train/val/test 按 job_id 划分，同一 job 的全部候选与 ranking_pairs 只属于一个集合，避免信息泄漏。**长尾平衡**：job-level uniform sampling 或 domain-aware rebalance，对长尾领域适度上采样、超高频领域适度下采样。

#### 10. 最小可运行版采样策略

单岗位目标 80：6 strong positive、14 positive、14 weak positive、46 negative（16 easy、12 hard、8 famous_off、6 inactive_relevant、4 confusing）。排序对：每 strong positive 配 2 famous_off、2 inactive_relevant、2 hard、1 positive；每 positive 配 2 hard、1 weak_positive。约每 job 几十到一百个有效 pair。

#### 11. 伪代码骨架

**11.1 单岗位候选池构造**：`build_job_candidate_pool(job_id, recall_outputs, feature_store, cfg)` — merge 三路召回去重，对每个 author 算 pair_features、author_features、build_teacher_score、build_pair_label、build_negative_type、source_flags，返回 candidates 列表（含 teacher_score、relevance、activity、pair_label、negative_type 等）。

**11.2 分层采样**：`stratified_sample_candidates(candidates, cfg)` — 按 pair_label 与 negative_type 分层，sample_up_to 各层到配置数量（num_strong_pos、num_pos、num_easy_neg 等），返回 sampled。

**11.3 ranking_pairs 构造**：对 strong_pos 每人配 2 个 famous_off（权重 1.50）、2 个 inactive_rel（1.40）、2 个 hard_neg（1.25）、1 个 pos（0.90）；对 pos 每人配 2 个 hard_neg（1.10）、1 个 weak_pos（0.75）。

**11.4 sample_weight 构造**：`build_sample_weight(sample)` — label_weight_map、type_weight_map、multi_source_weight（1/2/3 sources → 1.00/1.08/1.15）、confidence_weight（按 relevance/activity 微调），返回乘积。

#### 12. 本小节总结

> KGAT-AX v2 的训练样本构造，不追求“样本越多越好”，而追求“边界越清楚越好”；其核心是围绕单个岗位构造候选池，并通过 strong positive、famous-but-off-target、inactive-but-relevant 等关键样本类型，让模型真正学会：谁才是最适合当前岗位的活跃研究者。

---

### 训练流程与阶段化训练策略

KGAT-AX v2 不建议直接“所有模块一次性联合训练到底”，而是采用 **由简到繁、由稳到强、由粗到细** 的阶段化训练策略：先让模型学会基本岗位–作者相关性结构，再逐步引入综合价值信号、业务难例与细粒度排序约束。

#### 1. 总体训练流程概览

推荐拆为四阶段：**Stage 0：数据准备与 teacher 构造**（多路召回候选池、pair/author 特征、teacher_score、pair_label、negative_type、sample_weight、ranking_pairs、按 job_id 切 train/val/test）；**Stage 1：图表示预热（Graph Warmup）**（图分支先站稳）；**Stage 2：主体联合训练（Ranking + Classification + Distillation）**；**Stage 3：难例强化与排序校准（Hard-negative & Calibration）**。保守版可合并为：Stage0 数据与 teacher，Stage1 Warmup，Stage2 Main Training，Stage3 Fine-tune/Calibration。

#### 2. Stage 0：数据准备与 teacher 构造

不训练模型，只生成监督缓存：**job candidate cache**（每 job 候选池 + 来源标记）、**pair feature cache**（online pair 特征）、**author feature cache**、**teacher supervision cache**（teacher_score、relevance/activity/impact/authority、pair_label、negative_type、sample_weight）、**ranking pair cache**（每 job 的 ranking_pairs）。原则：**训练与推理同构**（pair_aux_online 必须和线上一致，teacher-only 只用于 teacher，不进 student 输入）；**监督噪声可控**（证据不足时宁可降权/不造强正/不造强排序对）。

#### 3. Stage 1：图表示预热（Graph Warmup）

目标：在不依赖复杂辅助特征的情况下，让图分支先学会基础岗位–作者相关性，避免后续被辅助特征淹没。只用图结构输入、基础节点 embedding、job/author 表示、少量最稳定的 online pair 特征；**开启**：graph encoder、job/author embedding、graph message passing、graph_only_head（可选少量 semantic_match）；**暂缓/弱化**：大部分 author_aux、全量 pair_aux_online、蒸馏损失、复杂 hard negative。损失：`L_stage1 = λ_graph_rank * L_graph_rank + λ_light_cls * L_light_cls`，推荐 λ_graph_rank=1.0，λ_light_cls=0.2，甚至只用 L_graph_rank。样本以 strong positive / positive / easy negative / 少量 hard negative 为主（比例约 1:2:3:1），暂缓大量 famous-but-off-target / inactive-but-relevant / confusing same-cluster。退出条件：graph-only ranking accuracy 明显高于随机、easy negative 区分稳定、strong positive vs easy negative margin 拉开、loss 不再大幅震荡。

#### 4. Stage 2：主体联合训练（Main Joint Training）

核心阶段，学习 relevance-first、active researcher 优先、排序主任务 + 分类边界 + teacher 连续监督。**开启完整主模型输入**：graph encoder、主 scoring head、classification head、graph_only_head、pair_aux_online、author_aux、ranking_pairs、teacher_score、pair_label、sample_weight；teacher-only 特征仍不进 student；aux dropout/random mask 开启；graph_only_head 保留。损失：`L_total = λ_rank*L_rank + λ_cls*L_cls + λ_distill*L_distill + λ_graph*L_graph`，推荐起点 λ_rank=1.0，λ_cls=0.5，λ_distill=0.15，λ_graph=0.12。样本包含完整关键边界类型，特别保留 strong positive vs famous-but-off-target 和 strong positive vs inactive-but-relevant。蒸馏采用 **先中等、后衰减**：Stage2 epoch1–2 λ_distill=0.20，3–5 为 0.12，6+ 为 0.06。通过 aux dropout、random feature mask、持续训练 graph_only_head 防止 student 退化为“teacher 公式 + 特征 MLP”。

#### 5. Stage 3：难例强化与排序校准（Hard-negative & Calibration）

不重学全部，只针对关键排序边界做“最后校正”，让 Top-K 更符合业务口径。样本中降低 easy negative 占比，提高 famous-but-off-target、inactive-but-relevant、confusing same-cluster、strong positive vs positive、positive vs weak positive、同层细排序对的比例。损失仍为联合损失，但权重大致调整为：`λ_rank=1.0`、`λ_cls≈0.40`、`λ_distill≈0.06`、`λ_graph≈0.10`。关注指标：Top-K 中 famous-but-off-target 占比是否下降；active relevant 是否稳定压过 inactive relevant；strong positive 是否更稳定在前列；同层候选顺序是否更贴近 teacher/人工判断；推荐解释是否更聚焦“对口+活跃+代表作”。

#### 6. 参数冻结、学习率与日程建议

参数冻结：Stage1 训练 graph encoder + graph_only_head + 基础 scoring head，冻结/弱更新大部分 aux 融合层和 cls/distill 相关头；Stage2 全量训练 backbone+heads；Stage3 可选全量小 lr fine-tune 或仅调 head，推荐优先全量小 lr fine-tune。学习率：推荐 AdamW，backbone lr 小于 head lr，例如 Stage2：graph encoder/backbone 1e-4，scoring/cls heads 3e-4，Stage3 整体再降 0.3～0.5 倍。日程示例：Stage1 3 epoch、Stage2 8 epoch、Stage3 3 epoch，结合验证集早停。

#### 7. 阶段切换与监控

Stage1→Stage2：graph-only ranking accuracy 显著优于随机，easy negative 稳定区分，strong positive vs easy negative gap 稳定，graph loss 收敛。Stage2→Stage3：主排序指标进入平台期，teacher-student 对齐基本稳定，hard negative 区分改善，Top-K 已可用但关键难例仍有错误。整体早停：结合 validation ranking metric、strong positive ahead rate、famous-but-off-target suppression rate、active-over-inactive rate、top-k business consistency，不只盯 val loss。分阶段监控指标：Stage1 看 graph-only；Stage2 看 ranking+classification+distill 组合；Stage3 看 Top-K 业务一致性。

#### 8. 阶段化训练伪代码骨架

```python
def train_kgatax_v2(model, train_jobs, val_jobs, config):
    # Stage 0: build caches
    train_cache = build_supervision_cache(train_jobs, config)
    val_cache = build_supervision_cache(val_jobs, config)

    # Stage 1: graph warmup
    stage1_cfg = config.stage1
    enable_stage1_mode(model)
    for epoch in range(stage1_cfg.epochs):
        train_one_epoch_stage1(model, train_cache, stage1_cfg)
        metrics = evaluate_stage1(model, val_cache, stage1_cfg)
        if should_switch_to_stage2(metrics, stage1_cfg):
            break

    # Stage 2: main joint training
    stage2_cfg = config.stage2
    enable_stage2_mode(model)
    for epoch in range(stage2_cfg.epochs):
        stage2_cfg.lambda_distill = get_lambda_distill_stage2(epoch)
        train_one_epoch_stage2(model, train_cache, stage2_cfg)
        metrics = evaluate_stage2(model, val_cache, stage2_cfg)
        if should_switch_to_stage3(metrics, stage2_cfg):
            break

    # Stage 3: hard-negative & calibration
    stage3_cfg = config.stage3
    enable_stage3_mode(model)
    for epoch in range(stage3_cfg.epochs):
        stage3_cfg.lambda_distill = get_lambda_distill_stage3(epoch)
        train_one_epoch_stage3(model, train_cache, stage3_cfg)
        metrics = evaluate_stage3(model, val_cache, stage3_cfg)
        if should_early_stop(metrics, stage3_cfg):
            break
    return model
```

模式控制示例：

```python
def enable_stage1_mode(model):
    model.enable_graph_only_head(True)
    model.enable_main_head(True)
    model.enable_cls_head(False)
    model.enable_distill(False)
    model.set_aux_dropout(0.0)
    model.freeze_aux_fusion_layers(True)

def enable_stage2_mode(model):
    model.enable_graph_only_head(True)
    model.enable_main_head(True)
    model.enable_cls_head(True)
    model.enable_distill(True)
    model.set_aux_dropout(0.15)
    model.freeze_aux_fusion_layers(False)

def enable_stage3_mode(model):
    model.enable_graph_only_head(True)
    model.enable_main_head(True)
    model.enable_cls_head(True)
    model.enable_distill(True)
    model.set_aux_dropout(0.10)
    model.freeze_aux_fusion_layers(False)
```

#### 9. 本小节总结

> KGAT-AX v2 采用阶段化训练策略，不追求“一步到位联合拟合”，而是先让图分支学会基本相关性结构，再逐步引入综合价值信号、业务难例与细粒度排序约束，从而稳定地逼近本项目的目标：在当前岗位语境下，优先识别并排出真正对口且仍然活跃的研究者。

---

### 推理流程与最终排序输出

KGAT-AX v2 的训练目标最终必须服务于线上推理：不仅“训练时学得对”，还要“推理时算得出、排得稳、解释得清”。推理原则：严格对齐训练期输入语义，在同构特征上完成候选精排，并输出符合岗位语境的推荐结果。

#### 1. 推理阶段总体流程

推荐拆为五步：**Step1 多路召回生成候选池**；**Step2 构建 online pair features 与 author features**；**Step3 KGAT-AX v2 对候选作者打分**；**Step4 排序后处理与结果校准**；**Step5 输出 Top-K 结果与推荐解释**。训练期是“学如何排”，推理期是在同构输入上的稳定执行。

#### 2. Step1：多路召回生成候选池

KGAT-AX v2 只负责精排，候选来自三路召回并集：**Vector Path**（语义/长文本 JD 对齐）、**Label Path**（技能/topic/field 结构化对齐与 Job–Skill–Topic–Paper–Author 路径）、**Collaboration Path**（合作网络扩展）。定义 `candidate_pool(job) = union(vector_candidates, label_candidates, collaboration_candidates)`，并保留来源标记 from_vector/from_label/from_collab/multi_source_count。候选池大小控制：`raw_candidate_pool_size ≈ 100～400`，`rerank_pool_size ≈ 80～200`，如 vector_topk=80、label_topk=80、collab_topk=60 合并后约 100～180。

#### 3. Step2：构建 online pair features 与 author features

推理严格遵循：**只用 online-available 特征，不用 teacher-only 特征**。允许输入：图结构（job/author 节点表示、邻域、GNN 结果）、pair-level online 特征（semantic_match、topic_match、representative_work_match、graph_alignment、multi_source_count 等）、author-level 特征（recent_topic_productivity/citation_signal、continuity、growth_trend、normalized_h_index/total_citations、top_paper_strength、recent_impact_quality、institution_strength、venue_strength、collaboration_prestige）。禁止输入：仅离线可见的 future-style 标注、依赖训练集统计的泄漏特征、teacher-only 规则结果。缺失处理：缺失即回退，用默认值/标志位/保守归一值，不中断推理；核心特征优先保证，非核心增强项可缺省。

#### 4. Step3：KGAT-AX v2 对候选作者打分

对每个 (job, author) 输出至少：`score_pred`（最终连续排序分数）、`cls_logits`（4 档分类输出）、可选 `graph_only_score`（调试用）。最终排序主分数是 `final_model_score = score_pred`，cls_logits 用于解释/边界分析/监控。teacher_score 不直接用作线上打分，原因：它是规则组合而非学习输出；student 应能超越 teacher；部分 teacher 构造过程不适合线上实时使用。`score_pred` 业务含义：在当前岗位语境下，该作者作为“优先推荐对象”的综合分数（对口 + 活跃 + 实力），而非全局学术地位或单一指标。

#### 5. Step4：排序后处理与结果校准

基础排序：按 score_pred 降序。可选轻量分类辅助：对 P(negative) 高的候选轻度降权，对 P(strong_positive) 高的候选轻度增益，形成 `post_score = score_pred * cls_adjustment`，其中 cls_adjustment 接近 1。推荐加 **relevance-safe 保护**：若 online relevance proxy 明显过低，对 post_score 乘以小于 1 的系数（如 0.85），作为安全带而非主逻辑。去重/实体合并：按 author_id 主键去重或按 author merge 规则保留最高分实例。最后做 Top-K 截断，如 `default_top_k = 50`，上层可区分 Top-10 强推荐与 Top-50 候选池。

#### 6. Step5：输出 Top-K 结果与推荐解释

最小输出字段建议包含：author_id、author_name、final_score、rank、pred_label、relevance_signal、activity_signal、impact_signal、top_support_paths、representative_works、matched_topics/matched_skills；精简版也应包含基本信息 + 分数 + 推荐理由摘要。推荐解释围绕三类问题组织：**为什么与岗位对口**（技能/主题/代表作/图谱路径）、**为什么说明当前仍活跃**（近年持续产出与关注）、**为什么说明值得排前**（代表作质量、活跃且方向稳定、实力与需求匹配）。避免解释只写“总引用高/机构强”，优先展示“对口/活跃/有代表作/当前值得联系”。

#### 7. 推理稳定性与监控

建议监控：候选池覆盖（各路召回数量、合并后大小、multi-source 比例）、排序输出结构（Top-K 中 strong positive 概率占比、negative 概率高样本占比、famous-but-off-target 与 inactive 风险样本占比）、特征健康度（在线特征缺失率、relevance/activity/impact 分布漂移、分数分布漂移）。异常兜底：候选池过小/特征大量缺失/分布塌陷时，可退回多路召回融合排序作为 fallback，或临时弱化复杂增强项，仅依赖图与核心 pair 特征打分。

#### 8. 推理伪代码骨架

```python
def infer_for_job(job, recall_system, feature_store, model, cfg):
    # 1. multi-path recall
    recall_outputs = recall_system.recall(job)
    candidate_pool = merge_and_deduplicate(
        recall_outputs[\"vector\"],
        recall_outputs[\"label\"],
        recall_outputs[\"collab\"],
    )
    candidate_pool = truncate_candidates(candidate_pool, cfg.rerank_pool_size)

    # 2. build online features
    samples = []
    for author_id in candidate_pool:
        pair_features = feature_store.build_pair_features_online(job[\"job_id\"], author_id)
        author_features = feature_store.build_author_features(author_id)
        source_flags = {
            \"from_vector\": author_id in recall_outputs[\"vector\"],
            \"from_label\": author_id in recall_outputs[\"label\"],
            \"from_collab\": author_id in recall_outputs[\"collab\"],
        }
        samples.append({
            \"job\": job,
            \"author_id\": author_id,
            \"pair_features\": pair_features,
            \"author_features\": author_features,
            \"source_flags\": source_flags,
        })

    # 3. model scoring
    batch = build_inference_batch(samples, cfg)
    outputs = model(batch)  # score_pred, cls_logits
    for i, sample in enumerate(samples):
        sample[\"score_pred\"] = float(outputs[\"score_pred\"][i])
        sample[\"cls_logits\"] = outputs[\"cls_logits\"][i]

    # 4. post process
    ranked = []
    for sample in samples:
        post_score = apply_post_ranking_adjustment(sample, cfg)
        sample[\"final_score\"] = post_score
        ranked.append(sample)
    ranked.sort(key=lambda x: x[\"final_score\"], reverse=True)
    ranked = deduplicate_authors(ranked)[: cfg.top_k]

    # 5. build explanations
    results = []
    for rank, sample in enumerate(ranked, start=1):
        results.append(build_ranked_result(sample, rank, cfg))
    return results
```

`apply_post_ranking_adjustment` 中可根据 cls_logits 轻量调节分数，并用 semantic_match/topic_match 拼出 relevance_proxy 做安全带；`build_ranked_result` 里组织 why_relevant / why_active / why_ranked_high 三类解释字段。

#### 9. 本小节总结

> KGAT-AX v2 的推理流程遵循“多路召回 → 在线特征构造 → student 模型精排 → 轻量后处理 → 可解释输出”的闭环设计；其核心不是返回“全局最强学者”，而是围绕当前岗位语境，稳定输出最对口、仍活跃、且最值得优先联系的研究者名单。

---

### 评估指标与实验验证设计

KGAT-AX v2 的评估目标不是单纯让训练 loss 更漂亮，也不是只提升某个离线分类指标，而是验证：**排序结果是否更符合当前岗位语境**、**是否更偏向“最对口且仍活跃的研究者”**、**是否成功抑制强但不对口或相关但已不活跃的作者**，以及 **相较原版 KGAT-AX 是否在业务目标上形成稳定提升**。

#### 1. 评估设计的总体原则

- **排序指标优先于分类指标**：输出是候选名单排序，而不是单条 pair 的二元判别。  
- **业务指标优先于纯训练指标**：loss/accuracy/AUC 只能反映拟合程度，不能直接说明是否符合岗位语境。  
- **难例指标必须单独看**：特别是 famous-but-off-target、inactive-but-relevant、confusing same-cluster。  
- **按 job 维度评估**：围绕“单岗位 Top-K 结果 + 多岗位整体平均表现”，而不是全局 pair 混合。

#### 2. 数据集划分与验证协议

- **按 job_id 切分**：train/val/test 必须 `split by job_id`，避免同一岗位 pair 被打散造成信息泄漏。  
- **推荐比例**：如 `train:val:test = 8:1:1` 或 `7:1.5:1.5`；测试集岗位不参与 teacher 调参，验证集只用于超参与早停，最终结果仅在 test 汇报。  
- **领域分布检查**：切分后检查各领域岗位数、候选规模、strong positive / hard negative 占比，避免 test 某些领域极端稀少。

#### 3. 核心离线排序指标

- **NDCG@K**：主指标之一，`NDCG@10/20/50`，gain 取 `3/2/1/0` 对应 strong/positive/weak/negative，反映 Top ranks 是否符合业务预期。  
- **Recall@K**：`Recall@10/20/50`，并单独汇报 Strong Positive Recall@K 与 (Positive+Strong) Recall@K。  
- **MRR/MAP**：可作补充。

#### 4. 分类与分层判别指标

- **4-class Accuracy**：对 {0,1,2,3} 四档总体准确率，仅作参考。  
- **Macro-F1 / Weighted-F1**：更好反映少数关键类（strong positive）。  
- **Strong Positive Recall**：强烈建议单独汇报。  
- **混淆矩阵**：关注 strong positive 是否被压成较低档，famous-but-off-target / inactive-but-relevant 是否被误判为高档。

#### 5. 业务关键指标（强烈建议）

- **OffTargetIntrusion@K**：Top-K 中“明显 topic 不对口”的作者占比，越低越好。  
- **ActiveOverInactiveRate**：在 relevant active vs relevant inactive 对比中，active 排在前面的比例。  
- **FamousOffTargetSuppressionRate**：在存在更对口 strong positive 时，压制 high-impact but off-target 的成功率。  
- **StrongPositiveAheadRate**：strong positive 是否稳定排在 positive / weak positive / famous-but-off-target / inactive-but-relevant 前。  
- **Top-K Business Consistency**：人工抽样岗位，评估 Top-K 是否对口、是否有明显离谱/不活跃且高排位候选、解释是否与排序一致。

#### 6. 难例专项评估

- **Famous-but-Off-Target 子集**：仅含 strong positive + famous-but-off-target，评估 pairwise ranking accuracy、NDCG、suppression rate。  
- **Inactive-but-Relevant 子集**：active relevant vs inactive relevant，评估 ActiveOverInactiveRate、Top-K active retention。  
- **Same-Cluster Confusion 子集**：邻近簇/高词面相似但任务不一致，评估 confusion suppression、ranking correctness、Top-K intrusion rate。

#### 7. 消融实验设计（Ablation）

- **模型结构消融**：Base KGAT、KGAT+Author Aux、KGAT+Pair Aux、KGAT-AX v1、KGAT-AX v2(full)。  
- **Teacher/Loss 消融**：无 distillation、无 classification、无 ranking、full loss。  
- **Relevance Gate 消融**：无 gate、分段 gate、sigmoid gate，对比 OffTargetIntrusion、NDCG、TopKBusinessConsistency。  
- **Activity 信号消融**：无 activity / 低权重 / 完整 activity，对比 ActiveOverInactiveRate。  
- **Hard Negative 策略消融**：只 easy、easy+hard、全 negative 策略，对比 hard 子集排序与 intrusion 减少。

#### 8. 训练过程监控

- **Teacher-Student Alignment**：Spearman、high-score 区域对齐，仅作训练中间指标。  
- **Graph-only vs Main Head**：若 main 很强但 graph-only 完全塌陷，说明过度依赖 aux 特征。  
- **按 negative_type 错误率**：单独看 famous-but-off-target、inactive-but-relevant、hard-negative 的错误情况。

#### 9. 定性案例分析（Case Study）

建议在 README 中展示若干真实岗位案例，每个包含：岗位描述摘要、原始召回概览、Baseline/KGAT 原版 Top-K、KGAT-AX v2 Top-K、关键对比解释（强但不对口被压下、active relevant 提上、same-cluster 混淆被纠正）。至少覆盖三类：**强但不对口**、**相关但不活跃 vs 相关且活跃**、**近义 topic 混淆**。

#### 10. Baseline 设计建议

- **Recall-level Baselines**：仅 Vector Path、仅 Label Path、三路召回简单融合。  
- **Ranker-level Baselines**：启发式打分、MLP on handcrafted features、原版 KGAT、原版 KGAT-AX。  
- **可选强基线**：LightGBM/XGBoost on pair+author features。

#### 11. 最小可运行版实验方案

必做：NDCG@10/20、Strong Positive Recall@10、Famous-Off-Target Suppression Rate、ActiveOverInactiveRate、原版 KGAT vs KGAT-AX v2 对比、2～3 个岗位案例分析。强烈建议：activity 消融、relevance gate 消融、hard negative 策略消融。可后补：更完整 ablation、领域分组、score calibration 实验。

#### 12. 评估伪代码骨架

```python
def evaluate_job_ranking(job_results):
    # job_results: 排好序的候选列表，每项含 final_score, pair_label, negative_type, relevance_flag, active_flag
    ndcg_10 = compute_ndcg(job_results, k=10)
    ndcg_20 = compute_ndcg(job_results, k=20)
    recall_10 = compute_recall(job_results, {2, 3}, k=10)
    strong_recall_10 = compute_recall(job_results, {3}, k=10)
    off_target_intrusion_10 = compute_off_target_intrusion(job_results, k=10)
    active_over_inactive = compute_active_over_inactive(job_results)
    famous_off_target_supp = compute_famous_off_target_suppression(job_results)
    return {
        "ndcg@10": ndcg_10,
        "ndcg@20": ndcg_20,
        "recall@10": recall_10,
        "strong_recall@10": strong_recall_10,
        "off_target_intrusion@10": off_target_intrusion_10,
        "active_over_inactive": active_over_inactive,
        "famous_off_target_suppression": famous_off_target_supp,
    }
```

```python
def evaluate_test_jobs(all_job_results):
    metrics = [evaluate_job_ranking(r) for r in all_job_results]
    return aggregate_metrics(metrics)

def run_ablation_suite(configs, train_data, val_data, test_data):
    all_results = {}
    for name, cfg in configs.items():
        model = train_kgatax_v2(cfg, train_data, val_data)
        test_results = infer_all_jobs(model, test_data, cfg)
        metrics = evaluate_test_jobs(test_results)
        all_results[name] = metrics
    return all_results
```

#### 13. 本小节总结

> KGAT-AX v2 的评估体系以排序质量和业务一致性为核心，不仅关注模型是否“学会了区分”，更关注它是否真正学会了：在当前岗位语境下，把最对口且仍活跃的研究者稳定排在前面。

---

### 工程改造清单与代码落地顺序

本节不再解释“为什么这样设计”，而是回答：**如果现在开始动代码，应该先改什么、后改什么、每一步改到什么程度才算完成**，并给出从方案到代码的施工蓝图。

#### 1. 工程改造总原则

- **1.1 保留主干，逐步替换**：先保留现有可运行的 KGAT/KGAT-AX 训练主干，在其上新增 v2 所需字段、特征、loss 与 head；v2 跑通并验证后，再按需清理旧接口，避免“大爆炸式重写”带来全崩和难定位 bug。  \n- **1.2 先打通数据链路，再改模型主体**：优先确保训练数据真实携带 v2 所需监督信号（teacher、pair/author 特征、ranking_pairs）进入 batch，而不是先魔改 GNN。  \n- **1.3 先做 MVP，再做完整增强版**：先实现“teacher_score 可算、pair_label 可产、ranking loss 可跑、模型能输出连续分数、inference 能对同一 job 候选排序”，再逐步加 negative_type 权重、classification head、distill、graph_only_head、Stage3 calibration。  \n- **1.4 训练/推理同构**：任何新增特征先问“线上能算吗”；不能算的只能进 teacher-only，不进 student 输入、不进 inference pipeline。

#### 2. 需要新增或修改的核心模块

- **数据准备层（Data Preparation）**：  
  - candidate pool builder：三路召回合并/去重/标记 source flags、截断到 rerank pool size。  
  - pair feature builder：构造 online pair 特征（semantic_match/topic_match/representative_work_match/graph_alignment/matched_topics/matched_skills/representative_works）。  
  - author feature builder：构造作者特征（recent_topic_productivity/citation_signal、continuity、growth_trend、normalized_h_index/total_citations、top_paper_strength、recent_impact_quality、institution_strength、venue_strength、collaboration_prestige）。  
  - teacher builder：根据 pair+author 特征构造 relevance/activity/impact/authority/gate/teacher_score。  
  - label builder：根据 teacher_score/relevance/activity 生成 pair_label、negative_type。  
  - ranking pair builder：同一 job 内构造强正 vs hard/famous/off-target/inactive/weak 等排序对。  \n- **数据缓存层（Cache/Dataset）**：  
  - supervision cache：存 pair_features、author_features、teacher_score、pair_label、negative_type、sample_weight、source_flags。  
  - job unit cache：每 job 的 sampled candidates、ranking_pairs、job-level metadata。  
  - split manifest：train/val/test job_id 列表。  \n- **Dataset/DataLoader 层**：  
  - 从单 pair 输入升级为 “job + candidate pool + ranking_pairs”；batch 同时包含连续目标、class label、ranking pairs、graph/pair/author aux tensors；支持 stage1/stage2/stage3 模式。  \n- **Model 层**：  
  - 保留 graph encoder 主干；新增 aux_fusion_layer、main scoring head（score_pred）、classification head（cls_logits）、graph_only_head（graph_only_score）、aux dropout/feature mask。  \n- **Loss/Trainer 层**：  
  - ranking loss module、classification loss module、distill loss module、graph aux loss、sample weighting（label_weight × negative_type_weight × confidence_weight × multi_source_weight）、stage-aware trainer（不同阶段的 loss 组合/学习率/冻结/采样/蒸馏权重）。  \n- **Inference 层**：  
  - recall→rerank pipeline、inference feature builder（与训练 online feature 同构）、model scoring（score_pred+cls_logits）、post-process（cls 调整、relevance-safe、去重、Top-K 截断）、explanation builder（why_relevant/why_active/why_ranked_high 等）。  \n- **Evaluation 层**：  
  - ranking metrics、business metrics、ablation runner、case study exporter。

#### 3. 推荐的代码落地顺序

1）**先完成 teacher 与样本监督链路**：实现 `build_pair_features_online()`、`build_author_features()`、`build_teacher_score()`、`build_pair_label()`、`build_negative_type()`、`build_sample_weight()`；能在某个 job 下打印出合理的监督表（author_id + relevance/activity/impact/authority/teacher_score/pair_label/negative_type/source flags）。  \n2）**完成 job-level 训练样本缓存**：固定每 job candidate pool，缓存监督字段与 ranking_pairs，按 job_id 切 train/val/test；能从 job_id 读出完整训练单元。  \n3）**改造 Dataset/Collator**：dataloader 输出 job-level batch，包含 pair_aux_online/author_aux/teacher_score/pair_label/sample_weight/ranking_pairs + graph tensors，支持 stage mode。  \n4）**实现最小可运行版模型输出**：在现有模型上新增 main scoring head，吃 graph output+pair_aux+author_aux，输出 score_pred；先不加 cls/head/graph_only/head 等。  \n5）**先只接 ranking loss，跑通 MVP**：仅用 L_rank 训练，验证集 NDCG/Recall/StrongPositiveRecall 可用即可。  \n6）**加入 classification head 与 L_cls**：输出 cls_logits，接 4-class CE + class weights + negative_type weighting。  \n7）**加入 distill loss 与 teacher 对齐**：接入 teacher_score 与 L_distill，做蒸馏权重调度，保证 teacher-only 特征不进 student。  \n8）**加入 graph_only_head 与 Stage1 warmup**：实现 L_graph 与 enable_stage1_mode()，支持 graph warmup + Stage2 联合训练。  \n9）**加入 aux dropout / random mask / harder negatives**：在 Stage3 提升 famous-but-off-target 与 inactive-but-relevant 的抑制能力。  \n10）**打通 inference pipeline 与 explanation 输出**：接 recall→rerank、同构 online feature、最终输出 Top-K+推荐理由摘要。

#### 4. 开发阶段与 MVP

- **Phase A：监督链路搭建**：teacher/label/negative_type/ranking_pairs 离线生成；可检查 candidate supervision 表与 job cache。  
- **Phase B：MVP 排序器**：model+dataloader+L_rank 跑通，NDCG/Recall 有区分；strong positive 能基本排到前面。  
- **Phase C：完整联合训练**：ranking+classification+distill+graph aux + Stage1/2；可做主实验对比。  
- **Phase D：业务增强与推理闭环**：hard-negative focus、inference pipeline、explanation 输出、完整业务指标。  \n**MVP 必须包含**：candidate pool merge、pair_features_online、author_features、teacher_score、pair_label、ranking_pairs、job-level dataset、score_pred、L_rank、基础排序指标。其余（classification/distill/graph_only/stage3/post-process/解释美化）可后续增强。

#### 5. 调试顺序与验收标准

- **调试顺序**：先人工检查监督表 → 再查 dataloader tensor 结构与 ranking_pairs 正确性 → 再跑 MVP ranking（只看 loss 降与 NDCG/Recall）→ 再逐步打开 classification/distill/graph aux → 最后打磨 inference 与 explanation。  \n- **阶段验收**：  
  - 阶段1：监督链路合理、缓存可读写。  
  - 阶段2：MVP 训练可跑，排序指标可用。  
  - 阶段3：完整训练优于旧版，业务关键指标改进。  
  - 阶段4：给定 job 能输出 Top-K + 合理推荐理由。  \n- **一句话路线**：先打通 teacher 与样本监督，再打通 job-level dataloader，再用 ranking loss 跑通 MVP，随后逐步接入 classification、distillation、graph-only warmup 与难例强化，最后完成 inference rerank 与解释输出。

#### 6. 本节收尾总结

> KGAT-AX v2 的工程落地不应追求“一次性重写完成”，而应采用“监督链路优先、MVP 先跑通、再逐步增强”的策略；只有这样，才能把前文的排序原则、训练设计与推理闭环真正稳定地落到代码中。

---


**本节说明**：上文各 ### 小节（排序原则、teacher_score、Loss、训练样本、阶段化训练、推理、评估、工程改造清单）已完整覆盖模型结构、Loss、防退化、数据文件、文件改动、修改顺序、训练策略、MVP、预期收益及改造原则；无需再以「8.～17.」重复罗列，按上文实施即可。

整体改造遵循：**先改数据，再改 loader，再改模型，再改 trainer，最后补推理**，以降低工程风险并保证每一步都有明确产出与可验证结果。


