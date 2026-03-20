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
  - [标签路设计定位](#标签路设计定位以任务主线为中心的学术概念落地与证据聚合)
  - [标签路核心设计原则](#标签路核心设计原则)
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
    │   │   ├── candidate_pool.py   # 统一候选池结构：CandidateRecord、CandidatePool、PoolDebugSummary
    │   │   ├── candidate_features.py # 候选池特征：extract_terms_from_label_evidence、6 个 calc_*、bucket_quota_truncate、build_kgatax_feature_row
    │   │   ├── total_recall.py     # 多路召回总控（向量 / 标签 / 协同）+ 候选池构建 / 打分 / enrich / 硬过滤 / 分桶 / 分桶截断 / kgatax_sidecar
    │   │   ├── vector_path.py      # 向量路：Faiss + 向量索引召回作者
    │   │   ├── label_path.py       # 标签路入口：编排五阶段流水线（label_pipeline + label_means）；从 stage1_domain_anchors 再导出 Stage1Result
    │   │   ├── label_path_pre.py   # 标签路旧版/备用实现（内部或历史参考）
    │   │   ├── works_to_authors.py # 论文级得分聚合为作者级，供标签路 Stage5 使用
    │   │   ├── diagnose_embedding_neighbors.py # 嵌入邻居诊断工具（开发与调试用）
    │   │   ├── collaboration_path.py # 协同路：基于本地协作索引的协同召回
    │   │   ├── label_means/        # 标签路子模块：基础设施、锚点、扩展、层级守卫、词/论文打分与调试
    │   │   │   ├── infra.py        # 资源管理：Neo4j / Faiss / vocab_stats.db / 簇中心等
    │   │   │   ├── base.py
    │   │   │   ├── label_anchors.py
    │   │   │   ├── hierarchy_guard.py  # 层级守卫：分布/纯度/熵、hierarchical_fit、泛词惩罚、landing/expansion 打分、should_drop_term（仅对非 primary-like）；score_term_record（base 0.36/0.18/0.20/0.14/0.12；final=base×gate×cross×backbone_boost×object_like_penalty×bonus_term_penalty）、_compute_backbone_boost/_compute_object_like_penalty/_compute_bonus_term_penalty、allow_primary_to_expand（放宽）
    │   │   │   ├── label_expansion.py  # 学术落点+扩展；get_vocab_hierarchy_snapshot、collect_landing_candidates(jd_profile)、allow_primary_to_expand
    │   │   │   ├── term_scoring.py
    │   │   │   ├── paper_scoring.py
    │   │   │   ├── simple_factors.py
    │   │   │   ├── advanced_metrics.py
    │   │   │   └── label_debug_cli.py
    │   │   └── label_pipeline/      # 标签路五阶段流水线实现
    │   │       ├── stage1_domain_anchors.py   # 领域与锚点；Stage1Result、attach_anchor_contexts、build_jd_hierarchy_profile，产出 jd_profile 与锚点 local_context/phrase_context
    │   │       ├── stage2_expansion.py       # 学术落点+扩展；接收 jd_profile，raw_candidates 带 subfield_fit/topic_fit/landing_score/cluster_id 等
    │   │       ├── stage3_term_filtering.py  # … → select_terms_for_paper_recall（eligible 前 JD 主轴门：仅 core，父锚 rk 与 a_sc **OR** 达线；gate summary；paper_select_score=…）→ main-axis / centrality / cutoff / …
    │   │       ├── stage4_paper_recall.py    # 论文二层召回；_compute_grounding_score 按 term（robot control/route/RL 等）偏题惩罚；paper_map 写入日志可降噪
    │   │       └── stage5_author_rank.py     # 作者排序与截断；预留 CoverageBonus/HierarchyConsistency/FamilyBalancePenalty
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
    │   └── tools.py               # 技能清洗（含泛用 JD 前后缀过滤 GENERIC_JD_PREFIXES/SUFFIXES）、JD 句式/停用词、get_decay_rate_for_domains 等
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
       - 关系：`AUTHORED / PRODUCED_BY / PUBLISHED_IN / HAS_TOPIC / REQUIRE_SKILL / SIMILAR_TO`（共现不写入 Neo4j，见下）
     - 使用 `build_work_semantic_links()` + Aho‑Corasick 自动机扫描论文标题与关键词，补齐 `(Work)-[:HAS_TOPIC]->(Vocabulary)`。
     - 共现数据由 `build_index/build_vocab_stats_index.py` 流式写入 `vocab_stats.db` 的 `vocabulary_cooccurrence`，标签路召回（学术共鸣、锚点共鸣、cooc_span/cooc_purity）均从该库读取，KG 流水线不再构建 `CO_OCCURRED_WITH`，避免大表自连接导致磁盘占满。
     - `build_semantic_bridge()` 利用 SBERT 计算跨类型词汇相似度（如岗位技能 ↔ 学术词汇），写入 `SIMILAR_TO`。
   - `build_index/` 下的一系列脚本继续构建：
     - 词汇 / 论文 / 岗位的 Faiss 向量索引；
     - 作者和机构的结构化特征索引；
     - 词汇跨领域统计索引；
     - 作者协作关系索引。

4. **KGAT‑AX 训练数据与图索引构建**
   - `kgat_ax/generate_training_data.py` 优先基于 `candidate_pool.candidate_records` 构造分层正负样本（Strong/Weak Positive、EasyNeg/FieldNeg/HardNeg/CollabNeg），产出 `train.txt` / `test.txt` 与可选四分支侧车 `train_four_branch.json` / `test_four_branch.json`，并将全图导出为加权三元组 `kg_final.txt`。
   - `kgat_ax/build_kg_index.py` 将 `kg_final.txt` 导入 SQLite，生成高性能的 `kg_triplets` 覆盖索引。

5. **图神经网络训练（KGAT‑AX）**
   - `kgat_ax/trainer.py` 使用 `DataLoaderKGAT` 加载训练数据与加权图，训练 `model.KGAT`：
     - CF 阶段学习岗位与专家的排序规律；
     - KG 阶段学习多关系图中的语义结构；
     - 通过 AX 学术特征（H‑index / 总引用 / 论文数）在向量空间中做小幅增益。
   - 最终权重以 `best_model_epoch_*.pth` 形式保存在 `data/kgatax_train_data/weights` 下。

6. **线上召回与精排**
   - **总召回**（`core/recall/`）：构建**候选池**。加载 Faiss 索引和 Neo4j 图谱，实现向量路、标签路、协同路三路召回，合并去重后得到统一候选池（约 500 人），供下游精排使用。**总召回不负责最终排序，只负责“谁进入候选池”。**
   - **精排**（`core/total_core.py` → `RankingEngine`）：**仅作用于候选池内部**。对总召回输出的候选人做深度重排（KGAT-AX 打分 + 召回序融合），产出最终 Top 100 及解释。即：**总召回输出候选池 → KGAT-AX 在候选池内做深度重排**，二者是**上下游关系**，不是并列的两个评分器。精排阶段可理解为：**Re-ranker（候选池内排序器），不负责全库检索（not retriever）**。

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
- **每条关联边的建立逻辑是什么？**（数据来源、匹配规则、建边条件、代码入口、批处理与增量）见 **[3.7 知识图谱关联边建立逻辑](#37-知识图谱关联边建立逻辑按边类型)**。
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
| **Step 4** | 共现已迁移至 `build_vocab_stats_index`（`vocab_stats.db` 的 `vocabulary_cooccurrence`），KG 流水线不再构建 Neo4j 共现边 |
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

Neo4j 中还会建立索引：`Work.domain_ids`、`Job.domain_ids`、`Vocabulary.term`、`Author.h_index`。共现数据由 `vocab_stats.db` 的 `vocabulary_cooccurrence` 提供，不在 Neo4j 中建 `CO_OCCURRED_WITH`。

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

#### 3.5 共现数据（Vocabulary – Vocabulary，不写入 Neo4j）

- **含义**：两词在多篇论文中一起出现，共现次数越多关系越强；用于标签路的学术共鸣、锚点共鸣及 cooc_span/cooc_purity 等指标。  
- **数据来源**：由 `build_index/build_vocab_stats_index.py` 从主库 `works` 的 `concepts_text`/`keywords_text` 流式计算词对共现频次，写入 **vocab_stats.db** 的 **vocabulary_cooccurrence**（term_a, term_b, freq），与 KG 原逻辑一致（strip+lower 清洗、同一 Work 内词对、freq 至少为 2）。  
- **使用方式**：标签路召回中的 `calculate_academic_resonance`、`calculate_anchor_resonance` 及 `get_cooccurrence_domain_metrics` 均从 `vocabulary_cooccurrence` 读取，**不再依赖 Neo4j**。KG 流水线不再执行 `build_cooccurrence_links()`，避免在主库上做大表自连接导致「database or disk is full」。

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
| （共现） | 由 vocab_stats.db 的 vocabulary_cooccurrence 提供，不写入 Neo4j | term_a, term_b, **freq** | **共现论文数**；标签路共鸣与 cooc_span/cooc_purity 由此表读取 |
| SIMILAR_TO | Vocabulary→Vocabulary | **score** (float) | **SBERT 语义相似度（仅跨类型）** |

### 3.7 知识图谱关联边建立逻辑（按边类型）

以下按每条边的**数据来源 → 匹配/计算规则 → 建边条件 → 代码入口 → 批处理与增量**逐条说明，便于查库、改 ETL 或扩展新边类型时对照。

| 边类型 | 数据来源 | 匹配/计算规则 | 建边条件 | 代码入口 | 批处理与增量 |
|--------|----------|----------------|----------|----------|----------------|
| **AUTHORED** | `authorships` 联表 `works`（author_id, work_id, inst_id, source_id, pos_index, is_corresponding, is_alphabetical, year） | 每行用 `WeightStrategy.calculate(pos_index, is_corr, is_alpha, pub_year)` 算 `pos_weight`；year 缺失按 2000 处理 | 每条 authorship 一行即建一条 AUTHORED；同一 Cypher 内若存在 inst_id/source_id 则顺带建 PRODUCED_BY、PUBLISHED_IN | `build_topology_incremental()` → `LINK_AUTHORED_COMPLEX` | `sync_engine("topology_sync", ...)`，按 `sync_id`（authorships 自增 ID）增量，marker 存于 `sync_metadata` |
| **PRODUCED_BY** / **PUBLISHED_IN** | 同上，来自 `authorships` 的 `inst_id`、`source_id` | 无额外计算；仅判断 `row.iid`/`row.sid` 非空 | 若该条署名记录带机构/渠道则建 Work→Institution / Work→Source | 同上，与 AUTHORED 同批 | 同上 |
| **HAS_TOPIC** | `works`：`work_id, title, concepts_text, keywords_text`；词库来自 `vocabulary.term` | **元数据**：`concepts_text`、`keywords_text` 按 `\|;,\)` 拆成 term，转小写后与 vocabulary 的 term 匹配。**标题**：Aho-Corasick 自动机扫描 `title`，匹配时做**单词边界**校验（前后非字母数字），避免子串误命中 | 匹配到的 (work_id, term) 对建一条 (Work)-[:HAS_TOPIC]->(Vocabulary)；term 以小写形式与词库匹配 | `build_work_semantic_links()` → `LINK_WORK_VOCAB` | 无增量；全表扫描 works，每 2000 条 (id, term) 一批 `send_batch` |
| **REQUIRE_SKILL** | `jobs`：`securityId, skills, crawl_time` | `extract_skills(skills)` 按 `,，;；/` 拆成 skill，转小写后与 `Vocabulary.term` 匹配 | 每个 (job_id, term) 建一条 (Job)-[:REQUIRE_SKILL]->(Vocabulary) | `build_job_skill_links()` → `LINK_JOB_VOCAB` | 按 `crawl_time > marker` 增量，每 1000 条一批，批后更新 marker |
| **共现（不写 Neo4j）** | 主库 `works` 的 `concepts_text`、`keywords_text` | 同 HAS_TOPIC 的 term 解析（strip+lower）；同一 work 内词对计数，freq≥2 才写入 | 由 **build_vocab_stats_index** 写入 `vocab_stats.db` 的 `vocabulary_cooccurrence(term_a, term_b, freq)`；标签路从该表读，不建 Neo4j 边 | `build_vocab_stats_index.py`（非 KG 流水线） | 流式/分块计算，不占 KG 的 sync_metadata |
| **SIMILAR_TO** | `vocabulary`：`voc_id, term, entity_type`；向量来自当前批次重算（不用索引 reconstruct） | SBERT 对 term 编码 → L2 归一化 → Faiss 检索 Top-100 最近邻；**仅保留与源词 entity_type 不同的邻居**（industry↔concept/keyword；concept/keyword 不与同 type 连）。阈值：≥0.85 全建，不足 3 条时用 [0.80, 0.85) 按分数从高到低补满 3 条 | 每个源词最多 3 条出边，边属性 `score` 为相似度浮点值 | `build_semantic_bridge()` → `LINK_SIMILAR` | 按 `voc_id > marker` 增量；每 1024 词一批重算向量并检索；Neo4j 每 5000 条边一批写入并更新 marker |

**建立逻辑要点小结**：

- **拓扑类边（AUTHORED / PRODUCED_BY / PUBLISHED_IN）**：一条 SQL 拉取 authorships+works，每行先算权重再一条 Cypher 写出 1～3 条边，强依赖 `sync_metadata` 的 topology_sync marker。  
- **语义打标边（HAS_TOPIC / REQUIRE_SKILL）**：均依赖 **Vocabulary 已同步**（Step 1 的 vocab_sync），通过 **term 字符串匹配**建边；HAS_TOPIC 全量扫描 + AC 自动机，REQUIRE_SKILL 按 crawl_time 增量。  
- **共现**：不建 Neo4j 边，由独立脚本写 vocab_stats.db，与 KG 顺序解耦。  
- **SIMILAR_TO**：依赖词汇向量与 Faiss 索引；**仅跨类型**建边，且每词最多 3 条，避免图过密；增量由 semantic_bridge_sync 的 voc_id marker 控制。

### 4. 构建逻辑要点

- **Step 0**：Neo4j 执行 `INIT_SCHEMA`（各节点 `id` 唯一约束 + 上述索引）；SQLite 执行 `SQL_INIT_SCRIPTS`，保证按 `last_updated`/`year`/`crawl_time`/`ship_id` 等增量查表高效。  
- **Step 1 节点同步**：对每类实体调用 `sync_engine(task_name, sql, cypher, time_field)`，从 SQLite 按 `time_field` 增量拉取（如 `last_updated > marker`），按 `BATCH_SIZE` 批处理，每批用对应 `MERGE_*` 写入 Neo4j，并在 `sync_metadata` 表更新 marker，实现断点续跑。  
- **Step 2 拓扑**：从 `authorships` 联表 `works` 取每条署名的 author_id、work_id、inst_id、source_id、pos_index、is_corresponding、is_alphabetical、year；对每行先用 `WeightStrategy.calculate` 算 `pos_w`，再调用 `LINK_AUTHORED_COMPLEX` 一次创建 AUTHORED（含 pos_weight）及可选的 PRODUCED_BY、PUBLISHED_IN。  
- **Step 3 语义打标**：先 `build_work_semantic_links()`（AC 自动机 + concepts/keywords），再 `build_job_skill_links()`。  
- **Step 4 共现**：共现数据由 **build_vocab_stats_index** 流式写入 `vocab_stats.db` 的 `vocabulary_cooccurrence`，KG 流水线不再构建；标签路共鸣与共现指标均从该表读取。  
- **Step 5 语义桥接**：只加「跨类型」的 SIMILAR_TO，避免同类型词之间重复连接；增量由 `semantic_bridge_sync` 的 marker（voc_id）控制，只处理 `voc_id > marker` 的词。

### 5. 实现细节与注意点

1. **共现数据源**：共现由 **build_vocab_stats_index** 从主库 `works.concepts_text`/`keywords_text` 流式计算并写入 `vocab_stats.db` 的 `vocabulary_cooccurrence`，与 HAS_TOPIC 同源（strip+lower 清洗）；KG 流水线不再执行 `build_cooccurrence_links`。  
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
| （共现） | 不写入 Neo4j | 由 vocab_stats.db 的 vocabulary_cooccurrence 提供 | 共现论文数；标签路共鸣等由此表读取 |
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

**work_terms_temp**（已废弃：KG 共现步骤已迁移至 build_vocab_stats_index，共现由 vocab_stats.db 的 vocabulary_cooccurrence 提供；主库中不再建此临时表。）

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
| **vocabulary_cooccurrence** | term_a, term_b, freq | 词对共现频次；由 build_vocab_stats_index 写入；标签路学术共鸣、锚点共鸣及 cooc_span/cooc_purity 均由此表读取（KG 不再构建 Neo4j CO_OCCURRED_WITH） |
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
| （共现） | 由 vocab_stats 提供，不建边 | vocabulary_cooccurrence(term_a, term_b, freq) |
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

#### 标签路设计定位：以任务主线为中心的学术概念落地与证据聚合

标签路（Label Path）的目标，并不是为岗位描述中的每个技能短语寻找「字面上最相近」的学术词，也不是单纯依赖向量近邻做术语翻译；其核心任务是：在完整 JD 的任务语境下，将工业侧技能表达稳定地落到一组能够代表岗位主线需求的学术概念上，并进一步通过论文与作者证据完成闭环验证。因此，标签路本质上是一个面向科技人才发现的、由任务主线驱动的多阶段概念落地与证据聚合系统。

从系统定位上看，标签路结合了多种成熟信息检索与知识图谱系统的设计思想：一方面，它继承了 Expert Finding（专家发现）系统中「需求解析 → 概念映射 → 证据聚合 → 专家排序」的基本链路；另一方面，它也借鉴了 Candidate Generation + Rerank（二段式候选生成与重排）的经典架构，将「高召回的候选生成」与「强调主线一致性的全局重审」分开处理。此外，Stage2A 的学术落点过程与 Concept Normalization / Entity Linking（概念标准化 / 实体消歧）高度相似，但又进一步引入了多锚点、多候选共同竞争的 collective disambiguation（集体消歧）思想：一个候选学术词是否正确，不仅取决于它与单个锚点的局部相似度，更取决于它能否与其它锚点、上下文和后续论文证据共同构成一条稳定的任务主线。

基于这一定位，标签路坚持如下核心原则：局部语义负责候选准入，主线一致性负责主词裁决。也就是说，SIMILAR_TO、conditioned_vec、alias/exact matching 等信号的作用，是为每个工业锚点提供一组「可能的学术落点」；但最终谁能够成为 primary、谁只应作为 keep_no_expand、谁应被视为 risky term，并不由单点相似度直接决定，而是由候选词是否与当前 JD 的任务主线一致、是否与其它锚点形成共振、是否能在论文层与作者层形成稳定证据链共同决定。这一设计可以有效降低短词、泛词、跨学科同形词带来的语义漂移风险，避免系统被「局部很像、整体不对」的近邻词误导。

从工程结构看，标签路采用五阶段流水线：Stage1 负责从 JD 中抽取工业锚点与上下文；Stage2A 负责**候选学术主线的生成与组内第一轮组织**（避免明显落错、保留可供 Stage3 重审的结构标签，**不是全局定案**）；Stage2B **仅围绕 Stage2A 已认可的强/弱 seed** 做小半径保守扩族，**不重新翻译锚点、不推翻 2A 主词身份**；Stage3 对 Stage2A/2B 的全部候选做**跨锚点、跨来源、跨家族的统一重审**，完成去歧义、support/risky 下压与 **paper-term 终审**；Stage4 负责论文召回；Stage5 再将论文证据聚合为作者级排序。这样的分层设计使系统具备清晰边界：Stage2A 关注「锚点是否落在合理学术落点、是否给真主线留出后续成长空间」，Stage2B 关注「是否仅对 seed 稳扩、不抢选主」，Stage3 关注「全局比较与 paper recall 用词的最终裁决」，Stage4/5 则负责「论文与作者侧证据闭环」。

因此，标签路的关键不在于「为每个词找到最近邻」，而在于：为整份 JD 找到一组最能解释该岗位核心能力结构的学术概念族群。一个学术词是否真正优质，不应只看它与锚点的局部相似度，还应看它能否召回主题集中、方向正确的论文，能否在作者层形成稳定、可解释的专家排序结果。换言之，标签路中的「学术词」不是最终目标，而是连接 JD、论文与作者的桥梁；其设计目标是最大限度地提升这座桥梁在任务语境中的稳定性、可解释性与检索效能。

#### 标签路核心设计原则

为保证岗位需求能够稳定、准确地映射到学术概念空间，标签路在设计上遵循以下原则。

1. **主线一致性优先于局部近邻相似**  
   标签路不将「与某个锚点最相似」直接等同于「最适合作为该岗位的学术主词」。对于岗位检索任务而言，真正重要的不是某个候选词与单个锚点的局部相似度，而是它能否在完整 JD 语境下，与其它锚点共同构成一条稳定的任务主线。因此，局部语义相似度只承担「候选准入」的作用，而「能否成为 primary、能否进入论文召回主词集合」则主要由主线一致性决定。这样可以有效抑制跨学科同形词、泛义词和局部近邻误召回带来的偏移。

2. **候选生成与全局裁决分层进行**  
   标签路采用「先生成、后裁决」的两段式设计。Stage2A/Stage2B 负责为锚点提供尽可能完整但仍受控的学术候选集合（其中 **Stage2A 是候选主线生成与组织层，不在本层做全局定案**；**Stage2B 仅对 seed 保守扩族，不重新翻译锚点**），Stage3 再对这些候选进行跨锚点、跨来源、跨家族的**统一重审与 paper-term 终审**。这样做的好处在于：前段不需要过早承担最终决策责任，后段则可以利用更完整的上下文证据进行全局判断，避免在候选尚不充分时过早剪枝，也避免仅凭局部证据直接确定最终学术词。

3. **强主词少量扩展，弱主词保守保留**  
   扩展并不是越多越好。标签路坚持「高置信主词少量扩展、低置信主词只保留不扩散」的策略：只有在 Stage2A 中被判为主线明确、上下文稳定、来源可靠的 primary term，才允许进入 Stage2B 做 dense / cluster / co-occurrence 扩展；而那些虽可作为主线代理、但主线证据不足的候选，只作为 keep_no_expand 进入后续 Stage3 重审，不直接触发扩散。这样可以在不丢失主线信息的前提下，尽量控制语义漂移。

4. **允许保底，不允许失控**  
   对于某些岗位核心锚点，候选集合可能暂时没有足够强的主线词，但这并不意味着整条任务支线应被直接丢弃。为此，标签路允许对「高质量核心锚点」设置保底主词（fallback primary）：当组内不存在足够强的 expandable primary 时，可从候选中选择最像主线的一项暂时保留，供 Stage3 继续判断。但这类保底词默认不参与扩展，且会带有明确的 fallback 标记。该原则的核心是：宁可保守保留一条潜在主线，也不让关键锚点在早期阶段整体失活；但保底仅用于保线，不用于放大噪声。
   此外，当 `select_primary_per_anchor` 在组内仍然找不到任何 primary 时，会启用“高质量锚点空组保线”：从非硬坏分支里按结构证据救回 1 条 `primary_keep_no_expand`。该 fallback 候选不做词面黑白名单，只用结构信号门控（如 `family_match / jd_align / context_continuity / hierarchy_consistency` 与 `generic/object/poly` 风险阈值），并同样标记 `fallback_primary`，同时将 `role_in_anchor` 置为 `mainline` 并标记 `is_grounded_fallback/weak_mainline_support`；禁止 Stage2B seed（不扩散且 `can_expand_from_2a=False`）且在 paper 阶段不抢配额。

5. **软约束优先于硬过滤**  
   标签路尽量避免依赖黑白名单、领域硬裁剪或固定规则表来决定候选去留，而是采用「软奖励 + 软惩罚 + 分桶重排」的方式处理歧义。领域路径一致性、上下文支持、多锚点共振、家族中心性等因素主要用于排序和分层，而不是直接作为一票否决条件。这样设计的原因在于：科技岗位往往天然存在跨学科特征，知识图谱中的领域标注也可能不完整。如果采用过强硬过滤，很容易误杀真实相关但标注不足的概念；而软约束更适合在保证召回的同时稳步提升排序质量。

6. **概念质量最终由论文与作者证据闭环验证**  
   标签路中的学术词不是终点，而是连接 JD、论文与作者的中间桥梁。因此，一个学术词是否真正「好」，不能只看它在 Stage2/Stage3 中的分数，还必须看它在 Stage4/Stage5 中是否能够召回主题集中、方向正确的论文，并进一步聚合出符合岗位要求的作者候选。若某个词虽然在局部语义上看似合理，但在论文层与作者层持续带来噪声，那么它就不应被视为高质量主词。换言之，标签路采用的是「概念层生成 + 对象层验证」的闭环设计。

7. **可解释性与工程可控性并重**  
   标签路不仅追求效果，也强调每一步的可解释性。锚点如何产生、候选从何而来、为何成为 primary、为何只能 keep_no_expand、为何在 Stage3 被打入 risky bucket、最终哪些词进入论文召回，这些决策应尽量能够通过日志与特征面板回溯。这样的设计既有利于调试和迭代，也有利于在毕业设计、项目答辩和后续工程优化中清晰说明系统行为，避免形成「只能看结果、无法解释过程」的黑盒链路。

---

**核心目的**  
走「**岗位技能 → 学术词 → 论文 → 作者**」的图谱路径，用知识图谱 + 多维度打分，找出与 JD 技能要求在语义和共现上都对齐的学术词，再只保留领域垂直、论文质量高的论文与作者；强调硬技能/概念对齐，并抑制泛词、万金油论文和弱相关论文。实现上采用**五阶段流水线**（`label_pipeline`）与**标签子模块**（`label_means`），基础设施与索引（Neo4j、Faiss、vocab_stats.db、簇中心等）由 `label_means.infra` 统一加载与管理。

| 维度 | 说明 |
|------|------|
| **输入** | JD 文本、query_vector、领域约束（domain_ids）、skills 清洗结果（来自 Job / 总控） |
| **依赖** | **基础设施**：Neo4j、vocab_stats.db（vocabulary_domain_stats、vocabulary_cooccurrence、vocabulary_cooc_domain_ratio 等）、Faiss（Job、Vocabulary）+ 词汇向量 .npy、簇索引（cluster_members、voc_to_clusters、cluster_centroids）。**配置**：config（DB_PATH、VOCAB_P95_PAPER_COUNT、SIMILAR_TO_TOP_K/MIN_SCORE 等）、domain_config（DOMAIN_MAP、decay）、LabelRecallPath 类常量。**工具**：DomainDetector、extract_skills、DomainProcessor、time_features、get_decay_rate_for_domains。 |
| **输出** | 候选作者 ID 列表、RecallDebugInfo（领域探测、锚点、扩展、词权重、论文/作者规模等） |
| **核心中间产物** | **Stage1**：Stage1Result（**含 jd_profile**）、anchor_skills（含 local_context、phrase_context、**anchor_source/anchor_source_weight**）、_jd_cleaned_terms、job_ids、regex_str。**Stage2A**：primary_landings（含 domain_fit、**landing_score**、**subfield_fit**、**topic_fit** 等；有 jd_profile 时做层级 fit 与硬门槛）。**Stage2B**：raw_candidates（term_role、domain_fit、parent_anchor、parent_primary、**subfield_fit**、**topic_fit**、**landing_score**、**cluster_id**、**main_subfield_match**；**`run_stage2`→`_expanded_to_raw_candidates` 顶层透传** **`primary_bucket`、`can_expand_from_2a`、`fallback_primary`、`admission_reason`、`reject_reason`、`survive_primary`、`stage2b_seed_tier`、`mainline_candidate`、`primary_reason`、`parent_anchor_final_score`、`parent_anchor_step2_rank`**，供 Stage3 聚合与统一/paper 门消费）。**Stage3**：**按 tid 去重聚合**、**classify_stage3_entry_groups**、**check_stage3_admission**、**第一段** score_term_record×**identity_factor**×**risk_penalty**×**role_factor**、**assign_stage3_bucket**（观测）、**`stage3_build_score_map`（统一连续分）**、**select_terms_for_paper_recall**（**JD 主轴门** `STAGE3_PAPER_MAIN_AXIS_*` 仅 core + `paper_select_score`+family+floor）、_collect_risky_reasons/_bucket_stage3_terms；score_map、term_map、idf_map、term_role_map、term_source_map、parent_anchor_map、parent_primary_map、tag_purity_debug（含 **stage3_entry_group**、**identity_factor**、**risk_penalty**、**stage3_bucket**、anchor_count、evidence_count、family_centrality、path_topic_consistency、generic_penalty、cross_anchor_factor、retrieval_role 等）；label_path 据此构建 term_confidence_map、term_uniqueness_map。**Stage4**：author_papers_list。**Stage5**：paper_map、contribution、作者聚合分、最终 author_id 列表。 |
| **主要问题** | 缩写歧义、任务词泛化、领域漂移、万金油词与弱相关论文需靠熔断与权重抑制 |

#### 3.1 标签路数据流与阶段总览

| 阶段 | 输入 | 输出 | 依赖的 label_means / 外部 |
|------|------|------|---------------------------|
| **Stage1** | query_vector, query_text, domain_id | active_domains, domain_regex, anchor_skills, Stage1Result | DomainDetector；label_anchors.extract_anchor_skills、supplement_anchors_from_jd_vector；tools.extract_skills |
| **Stage2A** | prepared_anchors, active_domain_set, query_vector, query_text, 可选 jd_*_ids | primary_landings（含 **can_expand_from_2a**、**fallback_primary**、**can_expand**（与前者同步）、**bucket**（含 primary_fallback_keep_no_expand）、primary_score 等；_debug 透传至 Stage3） | **最小收尾四点**：① `judge_primary_and_expandability` 将 **primary 资格（宽）** 与 **expand 资格（严）** 解耦，并对 **branch_blocked 误伤的控制主线词** 设 **窄逃生口**（见 README「三函数最小收尾」表）；→ `select_primary_per_anchor`；**无 `primary_expandable` 时**对 **sd/sk/rk 做「极小家族」收口**（**sd≤1、sk≤1**；**rk 仅当 sd 与 sk 皆空时兜底 1 条**），不把三桶合并压成单 tid，日志 **`[Stage2A no-expandable shrink audit]`**；② `select_primary_per_anchor` 在“高质量锚点空组”场景会救回 1 条结构型 `primary_keep_no_expand`（标记 `fallback_primary`、并将 role_in_anchor 置为 mainline；不扩散、禁止 Stage2B seed且不参与 paper）；③ 仍 0-primary 且锚点允许时 **`pick_fallback_primary_for_anchor`** 保线（不扩、交 Stage3）；④ `merge_stage2a_primary` |
| **Stage2B** | **carryover_terms**（2A 全量保留 landing）+ **seed_candidates**（`stage2b_seed_tier∈{strong,weak}`）, active_domain_set, jd_*_ids | raw_candidates（retain_mode、topic_source、seed_blocked；**另见 `stage2_expansion._expanded_to_raw_candidates`：`primary_bucket` 等 2A/2B 决策字段顶层键**） | **`check_seed_eligibility` 仅遍历 seed_candidates**（tier=none 的 carryover 不下门、只预置 `seed_blocked=True`/`stage2a_not_seed` 与 merge 一致）；**eligible_seeds** 再跑 dense/cluster/cooc；**`merge_primary_and_support_terms(carryover_terms, …)`** 必须把 support_keep/risky_keep 一并带进 Stage3，并从 **PrimaryLanding** 拷贝 **`admission_reason`/`reject_reason`/`mainline_candidate`/`primary_reason`** 到 **ExpandedTermCandidate**；**`seed_candidates=0` 紧凑路径** 对 **`merge_primary_and_support_terms`** 设 **`emit_merge_debug=False`**，避免与单行 `[Stage2B] anchor=…` 重复刷屏；**DENSE_PARENT_CAP=0.85**；expand_from_cluster_members；expand_from_cooccurrence_support |
| **Stage3** | raw_candidates（含 term_role、role_in_anchor、source_types、retain_mode、**primary_bucket / fallback_primary / mainline_candidate**、**parent_anchor_final_score / parent_anchor_step2_rank** 等）, query_vector, anchor_vids | score_map, term_map, idf_map, **paper_terms** | 去重聚合 → classify_stage3_entry_groups → check_stage3_admission；**第一段**：score_term_record × identity × risk × **role_factor**；**`_assign_stage3_bucket`**；**`stage3_build_score_map`** → **`_apply_family_role_constraints`**；**`select_terms_for_paper_recall`**：weak support（**可 `SUPPORT_TO_PAPER_*` 软放行进 `support_pool`**）/ **`risky_side_block`** → **JD 主轴门**（core 过线 → **`paper_recall_quota_lane=primary`**；**双弱 core** → **`core_axis_near_miss_soft_admit`** 进池）；**其余未硬挡 support/risky/其它桶先入 `support_pool`** → **`eligible.extend(support_pool)`** → **`[Stage3 paper gate summary]`** 等 → **`paper_select_score`**（**公式不变**；**扫描全局序**：**强主轴 core→bonus_core→其余**，**`PAPER_SELECT_STRONG_MAIN_AXIS_*`**）+ **family** + **`dynamic_floor`** + **support 槽配额**（**`paper_recall_quota_lane=support`**，**`support_quota_full`**）；**`[Stage3 paper gate reject audit]`** / soft-admit / pool / centrality / quota；**主 cutoff 后、tail 前** **`[Stage3 paper swap audit]`**（**core near-miss** 与最弱 **support soft-admit** 在 **`PAPER_RECALL_SUPPORT_SWAP_SCORE_MARGIN`** 内可一对一换位；近失手窗口 **`PAPER_RECALL_CORE_NEAR_MISS_SWAP_MAX_BELOW_FLOOR`**）→ **`PAPER_RECALL_TAIL_EXPAND_*`**（**`len(selected)≤3`** 或 **support 槽计数为 0** 时 **near-miss** 再补 1~2 词；**分层**：**`tail_expand_core_near_miss`** 先于 **`tail_expand_support_soft_admit`**，见 **`[Stage3 tail expand audit]`**）；**unified breakdown** 默认 Top **`STAGE3_UNIFIED_SCORE_DEBUG_TOP_K`（3）**；仍硬挡 **单锚 conditioned_only**、**fallback_primary** |
| **Stage4** | vocab_ids, regex_str, term_scores, term_retrieval_roles, term_meta, jd_text | author_papers_list（按作者聚合的论文及 hits） | 二层召回；**role_weight**（paper_primary=1.0、paper_support=0.7）；MELT_RATIO、domain 软奖励；**term-type 动态 grounding / local cap**；**wid 级 hierarchy_consensus_bonus**：默认 **`STAGE4_HIERARCHY_BONUS_POSITIVE_MAIN_AXIS_ONLY`** 下仅当 hit 加权里 **strong_main_axis_core** 占比 ≥ **`STAGE4_HIERARCHY_STRONG_AXIS_WEIGHT_FRAC`** 才允许 **bonus>1**，否则 **cap 1.0**（抑制 bonus_core 如 RL 的组级放大）；**term_meta** 透传 **`paper_select_lane_tier`**；**`[Stage4 hierarchy bonus by term-group audit]`**；**overlap survival** 默认 **`STAGE4_OVERLAP_SURVIVAL_MAX_LINES=0`**；**hierarchy consensus audit** 等 |
| **Stage5** | author_papers_list, score_map, term_map, debug_1（含 term_role_map、term_confidence_map、**term_family_keys**、**term_paper_meta**、**term_retrieval_roles**） | author_id 列表、last_debug_info（含 stage5 审计字段） | **paper_scoring**；护栏 5；**95 分位 `tanh` 压尾后**、**`accumulate_author_scores` 前** **`PAPER_AUTHOR_FANOUT_*`**：按署名 **作者数**（封顶 **`PAPER_AUTHOR_FANOUT_MAX_COUNT`**）给每篇论文 **`fanout_factor`** 并缩 **`p[\"score\"]`**（削弱单篇跨多作者线性刷榜；与作者内多篇递减 / term cap / 结构乘子叠加）；**`[Stage5 paper fanout audit]`**；同词递减 → **`TERM_MAX_AUTHOR_SHARE`**（仅缩放，**不改变** dominant_share 形状）→ **time_weight** → **`structure_mult_total`** = **`structure_factor`** × **term_strength_mult** × **paper_evidence_mult** × **multi_hit_mult**（全程 **`_compute_author_structure_shape`**）：**`structure_factor`** 按 **st/pc/mtp** 分段（st≤1∧pc≤1∧mtp=0→0.42；st≤1∧pc≤2∧mtp=0→0.58；st≤1∧mtp=0→0.72；mtp≥1∨st≥2→1+0.06·min(mtp,3)；else 0.88）；平滑项 **0.90+0.10·min(st,3)/3**、**0.88+0.12·min(pc,4)/4**、**1+0.08·min(mtp,3)**。**其后**（新作比过滤与 **max 归一化** 之前）**`STAGE5_SUPPORT_ONLY_AUTHOR_PENALTY_*`**：仅 **`sup_only_papers≥1`**、**`sup_with_pri_papers=0`**、**`top_pri_c≤ε`** 且 **`sup_share`** 达线 → **`author_score×0.62/0.78`**（纯 support、无 **paper_primary 同框** 论文托底；**不改 Stage3/4 词分**）；**`[Stage5 support-only author penalty audit]`**。AUTHOR_BEST_PAPER_MIN_RATIO；诊断：**`[Stage5 term-cap & structure audit]`**、**`[Stage5 author structure audit]`**（含 base 与分项 mult）、support dominance、**term→author top（前 4 paper terms × 5 作者）**、**top-author term mix**；CLI **合一作者榜** |

#### 3.1.1 正确分层图（各层定义）

以下给出标签路各层的**职责**、**输出**与**不负责**边界，便于理解分层设计。

---

**Step2：工业锚点抽取**

| 维度 | 说明 |
|------|------|
| **职责** | 从 JD 中抽出可桥接的工业锚点 |
| **输出** | `anchor_terms`、`anchor_score`、`anchor_context`、`anchor_type` |
| **不负责** | 学术翻译；学术主词选举；论文导向排序 |

---

**Stage2A：锚点落地（候选主线生成层）**

| 维度 | 说明 |
|------|------|
| **职责** | 将每个工业锚点映射到一组**候选学术主词**，并在锚点组内完成**第一轮主线化组织**。目标不是「给出全局正确答案」，而是**避免明显落错**、保留可供后续重审的主线候选，并尽量产出可扩 **seed**。**设计原则**：**候选主线生成优先，最终定案后置**（本层不替代 Stage3 的终审）。 |
| **输入** | `anchor_terms`、`anchor_score`、`anchor_context`、JD 全局语义、领域约束；候选来源含 SIMILAR_TO、conditioned_vec、alias/exact、family_landing 等（与实现一致）。 |
| **输出** | `per-anchor candidate pool`；五分桶：`primary_expandable`、`primary_support_seed`、`primary_support_keep`、`risky_keep`、`reject`；以及 `can_expand_from_2a`、`mainline_candidate`、`fallback_primary` 等**结构标签**（供 2B/3 消费，而非终局结论）。 |
| **负责的判断** | ① 候选是否仍属该锚点的**合理学术落点**；② 是可扩主词、弱 seed、仅保留主线，还是高风险旁枝；③ 为 Stage2B / Stage3 提供**可解释的桶与标记**，而不是只输出「好/坏」一刀裁断。 |
| **不负责** | 大规模扩散；跨锚点全局胜负；最终 paper-term 资格判定；作者层证据闭环。 |

**说明（五分桶语义）**：`primary_expandable` 为高置信主线、Stage2B **强 seed**；`primary_support_seed` 主线感成立但证据略弱，可作**弱 seed**；`primary_support_keep` 主线可保留、**默认不扩散**，交 Stage3 再审；`risky_keep` 局部相关但高风险，**不作为主线输入**；`reject` 为明显偏题或坏分支。**「保留」≠「定案」**，**「不扩散」≠「无效」**；Stage2A 主要职责是**组织候选**，而非提前替 Stage3 做完最终判决。兼容表述：`primary_keep_no_expand` 常对应 support_keep / 弱 seed 侧并集等（详见下节「五层落点」与代码字段）。

---

**Stage2B：局部家族扩展（保守增殖层）**

| 维度 | 说明 |
|------|------|
| **职责** | **仅**围绕 Stage2A 已认可的 seed（`primary_expandable` / `primary_support_seed` 且通过 `check_seed_eligibility`）做**小半径、保守式**家族扩展，补入与主线紧邻、可增召回覆盖的近邻。**设计原则**：**围绕主线增殖，不重新选主**。 |
| **输入** | Stage2A 输出的强/弱 **seed** 及 `can_expand_from_2a`、`parent_primary`、family 语义等；**非 seed** 的 carryover（如 `support_keep` / `risky_keep`）**默认 carryover，不自动参与扩散**。 |
| **输出** | `dense_expansion`、`cooc_expansion`、`cluster_expansion`（实现侧 cluster 可关）、与 carryover 合并后的 anchor-family 术语流（raw_candidates）。 |
| **负责的判断** | ① 哪些 **seed** 允许扩散；② 扩展近邻是否仍属该**主线家族**；③ 扩展词作为 support / 侧证是否保留。 |
| **不负责** | 重新翻译工业锚点；推翻 Stage2A 的主词身份；跨锚点全局 term 胜负；最终 paper recall 选词。 |

**说明**：2B 的关键是「扩得稳」而非「扩得多」。文档口径：**仅围绕 seed 的保守扩族**；非 seed 桶不全线进扩散链路，与「主词扩族」相比，更强调**谁有资格被扩**。

---

**Stage3：统一重审 + 最终 paper-term 定案**

| 维度 | 说明 |
|------|------|
| **职责** | 对 Stage2A/2B 产生的**全部**候选做**跨锚点、跨来源、跨家族**的统一重审：去歧义、降噪、support/risky 下压，以及**最终 paper-term / paper recall 用词选择**。**关键词**：**重审，不是继承 Stage2A 判决**；前层 `primary_bucket`、`fallback_primary`、`mainline_candidate` 等是**参考输入**，**不是不可推翻的结论**（例如：`primary_expandable` 不保证最终进 paper，`support_keep` 也不等于一定无用，`fallback_primary` 也不等价于普通 primary）。 |
| **输入** | 全部 Stage2 carryover 与 expansion；`primary_bucket`、`mainline_candidate`、`can_expand_from_2a`、`fallback_primary`、锚点计数、主线命中等聚合特征。 |
| **输出** | **统一连续分**（`stage3_build_score_map` 等）、观测用 core/support/risky 分桶、**selected_for_paper** 逻辑对应的 **`final_term_ids_for_paper`** / paper_terms 等（与 Stage4 衔接）。 |
| **负责的判断** | ① 跨锚点重复与合并；② 真假主线的**全局**比较；③ support / risky 的最终降级；④ term 是否进入 paper recall。 |
| **不负责** | 回到工业锚点层改写 JD；大规模词扩散；论文与作者的最终打分（Stage4/5）。 |

---

#### Stage2A / Stage2B / Stage3 的关系说明

为避免将三层职责混淆，本系统采用「**候选生成 → 保守扩展 → 全局定案**」分层：

1. **Stage2A** 把工业锚点落到**合理学术候选主线**上，并输出桶与结构标签；**目标不是最终定案**，而是高质量候选与明显错误的早期拦截。  
2. **Stage2B** **只**对少量已认可 seed 做局部扩族；**目标不是重新选主**，而是**围绕已承认的主线**补近邻。  
3. **Stage3** 才对所有候选做跨锚点统一比较；**目标不是机械继承前层桶**，而是完成**真正的全局裁决**与 paper-term 选择。

---

**Stage4：论文召回**

| 维度 | 说明 |
|------|------|
| **职责** | 用 final academic terms 检论文 |
| **输出** | 论文级召回结果（供 Stage5 作者聚合使用） |
| **不负责** | 学术词生成与筛选（由 Stage3 负责） |

---

**Stage5：作者聚合排序**

| 维度 | 说明 |
|------|------|
| **职责** | 由论文贡献汇总到作者 |
| **输出** | 作者排序列表及可选 debug 信息 |
| **不负责** | 论文召回与 term 权重（由 Stage4 / Stage3 负责） |

---

#### 3.2 各阶段目的、关键步骤与关键数据（概要）

- **Stage1 领域与锚点**：得到 active_domains、domain_regex、工业侧锚点（+ 可选 JD 向量补充）；无锚点时后续短路。
- **Stage2A 学术落点**：锚点 → 跨类型 SIMILAR_TO + conditioned_vec + alias/exact → **组内选主**：`judge_primary_and_expandability` 在 **`primary_ok`（宽）** 成立之后，用 **主线一致性连续分** `axis_consistency` / `effective_mainline`（`anchor_identity`、`family_match`、`jd_align`、上下文项、`mainline_pref_score` 与 object/generic/poly **风险罚项**）将候选落入 **五层**（见下节）；**`select_primary_per_anchor`**：**Pass1** 按锚点组收集 judge 结果与 **`axis_consistency_seed` 等特征；**Pass2** 在组内按 **(axis_seed, mainline_pref, jd, ctx)** 排序后做 **final reconcile**。**`keep_promote_strong_mainline` / `seed_promote_to_expandable`** 须同时满足：**`similar_to` 强证据链**、**`axis_consistency_seed≥SEED_AXIS_CONSISTENCY_STRONG_MIN`**、**组内排名=1**、**与第 2 名 axis 差 ≥ `STAGE2A_PROMOTE_MIN_AXIS_GAP`**、且 **`judge_expandable_stable` 未成立**（即 judge 未已给出可维持的强 `primary_expandable`，否则不再把 keep/sd 旁枝抬成 strong）。轴不足或组门失败则 **sd/sk** 或 **`keep_promote_blocked_*`**。调试：**`[Stage2A group rank]`**、**`[Stage2A promote audit]`**、**`[Stage2A final bucket reconcile]`**（含 **group_rank**）。**兼容字段** `primary_keep_no_expand` = 后三者与弱 seed 的并集（旧代码路径）。**无 `primary_expandable` 时**：**`primary_support_seed` / `primary_support_keep` 各按排序至多保留 1 条**；**`risky_keep` 默认清空**，**仅当 sd 与 sk 均为空时保留 1 条 rk 兜底**（避免旧版把三桶合并成「全局一名」导致 Stage3 无对局空间）；调 **`[Stage2A no-expandable shrink audit]`**（`LABEL_EXPANSION_DEBUG`）。仍 **全空** 时对合格锚点走 **空组保线 / `pick_fallback_primary_for_anchor`** 等路径。**`stage2b_seed_tier`**：`strong` | `weak` | `none`。（`_classify_keepish_primary_bucket` 仍为模块内辅助函数，**不再作为 judge 主路径**。）
- **Stage2B 学术侧补充**：**`can_expand_from_2a=True`** 的候选经 `check_seed_eligibility`：**先 `blocked_by_2a` 硬挡**；**strong / weak** 分别要求 **`seed_score` ≥ `SEED_SCORE_MIN` / `SEED_SCORE_MIN_WEAK`** 且 **`axis_consistency_seed`**（identity+family+jd+mainline_pref）达线；**weak** 另拦 **`conditioned_only_weak_seed`** 与 **generic/poly/object** 上限。`expand_from_vocab_dense_neighbors` 按 tier **分档**：**strong** `min_dense_sim=0.80`、`min_anchor_consistency=0.72`、`min_family_support=0.68`、**weak** `0.84` / `0.78` / `0.74`，邻居还须 **`context_stability≥0.72`**；**weak seed 禁用** `weak_support_release_for_strong_seed`。**fallback / 非 seed 桶不扩散**。**cluster 全关**；**DENSE_PARENT_CAP=0.85**；输出 List[ExpandedTermCandidate]（primary / dense_expansion / cooc_expansion）。
- **Stage3 全局复审层（非全杀门）**：去重聚合 → classify_stage3_entry_groups → check_stage3_admission → **第一段分** = score_term_record × identity_factor × risk_penalty × **role_factor**（同左 Role 因子）；**assign_stage3_bucket** 仍为 core/support/risky，**仅观测**；**第二段** **`stage3_build_score_map`**：改为 **统一连续分**（seed / anchor_identity / jd_align / family_centrality / cross_anchor / mainline / expand − drift / generic / poly / object），再 **`_apply_family_role_constraints`**；**select_terms_for_paper_recall**：**入 eligible 前**依次 **weak support 结构门**（`STAGE3_PAPER_SUPPORT_*`）：对 **`weak_support_contamination`**，若 **`SUPPORT_TO_PAPER_ENABLED`** 且（**`primary_support_seed`** → **`support_seed_soft_admit`×`PAPER_SUPPORT_SEED_FACTOR`**）或（**`primary_support_keep`** ∧ (**`mainline_candidate`∨`mainline_hits≥1`∨可扩**）→ **`support_keep_soft_admit`×`PAPER_SUPPORT_KEEP_FACTOR`**），写入 **`support_pool`**（**`paper_readiness`** 在 `_apply_paper_readiness_for_recall` 内乘因子）；否则仍 **`weak_support_contamination_block`**。随后 **`risky_side_block`**（`STAGE3_PAPER_RISKY_SIDE_*`：`stage3_bucket=risky` ∧ ¬`can_expand` ∧ `mainline_hits≤0` ∧ **`term_role∈{primary_side,dense_expansion,cluster_expansion,cooc_expansion}`**）→ **JD 全局主轴门**（`STAGE3_PAPER_MAIN_AXIS_*`，**默认开**）：**仅** **`stage3_bucket=core`** 参评；**`rk≤MAX` ∨ `a_sc≥MIN`（OR）** 过线者 **`paper_recall_quota_lane=primary`** 进排序主干；**双弱** 不再整词丢弃，改为 **`core_axis_near_miss_soft_admit`**（×**`PAPER_CORE_AXIS_NEAR_MISS_FACTOR`**，`paper_recall_quota_lane=support`）进 **`support_pool`**。凡未触上述硬挡的 **support / 未挡 risky / 其它桶** **统一先入 `support_pool`**（如 **`support_lane`**、**`risky_support_lane`** ×`PAPER_RISKY_SUPPORT_LANE_FACTOR`），**`loop` 结束 `eligible.extend(support_pool)`**——与 Stage2「可保留但不扩」对齐：**不扩散 ≠ 禁止参与 paper 竞争**。区分 **Stage2「相对父锚的局部主线」** 与 paper 侧 **父锚全局强弱**；**非 core** 跳过主轴门。**入选阶段**：**`paper_recall_quota_lane=support`** 计数 ≤ **`min(SUPPORT_TO_PAPER_MAX, ⌊max_terms/3⌋)`**，超出 **`support_quota_full`**；**`retrieval_role`**：**primary lane** → `paper_primary`，否则 `paper_support`。审计 **`[Stage3 paper gate reject audit]`**、**`[Stage3 support soft-admit audit]`**、**`[Stage3 paper quota audit]`**（**`selected_primary_lane` / `selected_support_lane`**）。注意 **仅靠 rk/a_sc 无法区分「rk=8」与「rk=11」且分都低于 MIN 的两类 core**——要同时压 RL 又保 route planning 类词需再加结构信号或接受放宽 **MAX_RANK** 后由 **`paper_select_score`/floor** 二次分流。未过前述前置硬挡：**`weak_support_contamination_block`（或未达软放行条件）/ `risky_side_block`**；合并后候选计算 **`paper_select_score`**：`final_score×paper_readiness` + Stage2 结构 **bonus** − **对数型候选池规模惩罚**（**`papers_before_filter`** 或回落 **`degree_w`**，`PAPER_SELECT_POOL_PENALTY_*`，缓释少词表下超大池词垄断排序与 **dynamic_floor**），见 **`[Stage3 paper pool penalty audit]`**；然后 **family_key** 去重 + **与 `paper_select_score` 对齐的** **`PAPER_RECALL_DYNAMIC_FLOOR_*`**；其余 **risky** 仍可入 paper；上限 **PAPER_RECALL_MAX_TERMS**（默认 12）。
- **Stage4 论文召回**：Neo4j HAS_TOPIC + 3% 熔断 + 可选 domain 正则 → author_papers_list（按作者聚合，单次上限 2000 篇）。**结构约束**：exact「robot control」在 Stage4 内视为 **paper_support** 并 **paper_factor≤0.68**，避免 umbrella 泛词以 Primary 权重灌榜；与 Stage5 单 term 作者占比上限互补，而非依赖领域黑名单。
- **Stage5 作者排序**：paper_scoring（护栏 5）→ **作者×词** → 同词递减 → **`TERM_MAX_AUTHOR_SHARE`** → **时间权重** → **`_compute_author_structure_shape`**：**`structure_factor`** 分段拉开 **单词单篇（mtp=0）** vs **multi-term 论文 / 多强词**，再乘 **较轻的** term/paper/mhit 平滑三项，避免叠乘把 singleton 抬回前排；**最佳作比** → 归一化。调试：**`[Stage5 author structure audit]`**（base、st、pc、mtp、struct_f、t/p/m mult、struct_tot、after）、**term→author contribution top** 仅 **4 terms×5 authors**。`[Stage4 hierarchy bonus distribution]` 默认摘要。`STAGE3_DEBUG` 下 **final_term_ids** 行用 **top_unique_papers(wid:hit_n)**。

#### 3.2.1 各阶段详细说明（作用、思路、逻辑、输入输出、公式、表与索引）

以下按阶段给出：**作用**、**思路**、**逻辑流程**、**输入/输出参数（名字与含义）**、**主要公式**、**调用的表或知识图谱**。实现文件：`label_pipeline/stage1_domain_anchors.py`、`stage2_expansion.py`、`stage3_term_filtering.py`、`stage4_paper_recall.py`、`stage5_author_rank.py`，以及 `label_means` 下 `label_anchors`、`label_expansion`、**`hierarchy_guard`**（分布/纯度/熵、层级 fit、泛词惩罚、landing 与 expansion 打分）、`term_scoring`、`paper_scoring` 等。

---

**Stage1：领域与锚点**

| 项 | 说明 |
|----|------|
| **作用** | 确定本次召回的**活跃领域集合**（active_domain_set）与 **Neo4j 领域正则**（regex_str），并产出**工业侧锚点词**（anchor_skills：岗位技能 + 可选 JD 向量补充），以及**JD 四层领域画像**（jd_profile）与锚点**局部上下文**（local_context、phrase_context），供 Stage2 做「技能→学术词」落点与层级守卫。无锚点时整条标签路短路。 |
| **思路** | 先由岗位/向量推断「本 query 属于哪些领域」，再在这些领域对应的岗位上抽取「高频且非泛词」的技能作为锚点；可选地用 JD 文本在词汇向量索引上做 Top-K 检索，把与 JD 语义贴近的学术词补进锚点；为每个锚点打 anchor_type；**新增** `attach_anchor_contexts` 为锚点附加 local_context / phrase_context，**新增** `build_jd_hierarchy_profile` 利用锚点 SIMILAR_TO 学术词与 vocabulary_topic_stats/domain_stats 聚合成 jd_profile（domain/field/subfield/topic_weights、active_*、main_*_id），供 Stage2 层级契合与泛词抑制。 |
| **逻辑流程** | ① 预清洗 JD 技能（`extract_skills`，含**泛用 JD 前后缀过滤** GENERIC_JD_PREFIXES/SUFFIXES，见下）→ 写入 `recall._jd_cleaned_terms`、**`recall._jd_raw_text`**；② 领域与岗位探测；③ 锚点提取：`label_anchors.extract_anchor_skills`（REQUIRE_SKILL + 熔断 + **短语中心性**：算 specificity/context_richness/taskness/local_cluster_support 后先做 **support_mean 凸组合**，再 **`final_anchor_score = backbone_score × (0.72 + 0.28 × support_mean)`**（**非四因子连乘**，避免主线锚被压到 0.0x；权重见 `ANCHOR_SUPPORT_W_*` / `ANCHOR_FINAL_*`），排序 + TopN）；④ 可选 `supplement_anchors_from_jd_vector`；⑤ `classify_anchor_type` 打 anchor_type；⑥ 领域 regex；⑦ **attach_anchor_contexts**；⑧ **build_conditioned_anchor_representation** 为每个锚点算 conditioned_vec 并写入 anchor_skills；⑨ **build_jd_hierarchy_profile** 产出 jd_profile；⑩ 写入 `recall._last_stage1_result`（含 jd_profile）。 |
| **输入参数（名字与含义）** | **query_vector**：JD 编码向量；**query_text** / **semantic_query_text**：JD 文本，供技能清洗、JD 向量补充与锚点上下文化；**domain_id**：可选，指定则优先作 active_domains。 |
| **输出参数（名字与含义）** | **active_domain_set**、**regex_str**、**anchor_skills**（每项含 term、anchor_type、**local_context**、**phrase_context**）；**debug_1** 含 job_ids、anchor_debug、**jd_profile** 等；**Stage1Result.jd_profile** 供 Stage2 传入 `run_stage2(..., jd_profile=...)`。 |
| **主要公式** | 锚点熔断：丢弃 **cov_j ≥ ANCHOR_MELT_COV_J**（如 0.03）的词。**backbone_score**：`BACKBONE_W_JOB_FREQ*log(1+job_freq) + … + BACKBONE_W_SIM*sim`；保底词（in_jd 且 task_like）不低于 **BACKBONE_FLOOR_FOR_BAODI**。**短语支撑**：`support_mean = 0.30·specificity + 0.25·context_richness + 0.25·taskness + 0.20·local_cluster_support`（均在 [0,1]）；**`final_anchor_score = backbone_score × (0.72 + 0.28 × support_mean)`**，**`collapse_ratio = final / max(backbone, ε)`** 用于日志 **「anchor collapse audit」**。无词表。 |
| **调用的表/知识图谱** | **Neo4j**：**Job** 节点、**Vocabulary**、边 **REQUIRE_SKILL**、**SIMILAR_TO**（build_jd_hierarchy_profile 用锚点→学术词）；**Faiss**：Job 索引、Vocabulary 索引（JD 向量补充）；**SQLite**：`vocabulary_domain_stats`（锚点候选 span、**build_jd_hierarchy_profile 聚合 domain_dist**）、**vocabulary_topic_stats**（**build_jd_hierarchy_profile 聚合 field/subfield/topic 分布**；批量 SELECT 行含 **voc_id 排第 0 列**，三层 JSON 分布在 **第 4–6 列**，须与 **row[4]/[5]/[6]** 对齐读取，**勿用 row[3]（topic_id）误当 field_dist**）；**外部**：`industrial_abbr_expansion.json`（缩写 key，anchor_type=acronym）。 |

**泛用 JD 短语过滤（tools.py）：前后缀 + 残片结构规则**

技能清洗 `extract_skills` 在 `is_bad_skill` 中采用**前后缀 + 结构规则**，避免只堆具体短语、减少误杀：

- **前缀（可直接过滤）**：`GENERIC_JD_PREFIXES` — `对系统`、`建立高`、`技术追踪`。不裸杀单字前缀（如「对」），以免误伤正常短语。
- **后缀（可直接过滤）**：`GENERIC_JD_SUFFIXES` — `技术文档`、`有高要求`。**不**用「高性能」裸后缀，以免误杀「高性能计算/高性能存储」等。
- **「动词+高性能」残片**：仅拦 `建立高性能`、`构建高性能`、`实现高性能` 等前缀（`GENERIC_JD_VERB_HIGH_PERF_PREFIXES`）。
- **残片结构规则** `is_generic_jd_fragment(term)`：判断是否为 JD 叙述残片（动作/要求/目标/软描述），而非可独立映射学术词的概念。规则包括：以「的」结尾且较长；「的」过重或长句带「的」；**动作链前缀**（`推动`、`提升`、`形成`、`开展`、`确保`）仅在与长度/评价性/「的」联动时判脏；**评价性表达**（高要求、优秀、扎实、良好）与长度/「的」联动；**泛描述后缀**（鲁棒性、可执行性、系统化思维、沟通协作能力）与长度/「的」联动。不单独裸杀「优化/调优/控制/系统/仿真」等易误伤术语。
- **句式剥离**：`normalize_skill` 中对「熟练使用 X」做剥离，保留 X（如 `熟练使用 c` → `c`，`熟练使用 python` → `python`），不把「熟练使用 c」当长期前缀黑名单扩展。

扩展时：前后缀可增删元组项；结构性规则在 `is_generic_jd_fragment()` 中按「是否可独立映射学术词」标准调整。

**锚点分型后的行为差异（Stage2A / Stage2B 约束）**

`classify_anchor_type` 打出类型后，不同类型**不改变 Stage2A 的候选来源**，当前版本统一由**跨类型 SIMILAR_TO + 召回时 conditioned_vec 检索**两路合并负责 primary landing；不同类型主要影响：

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

#### Stage2A 五层落点体系（设计与实现对应）

**目标**：先为每个候选学术词分配 **主线身份层级**，再决定 **是否进入 Stage3**、**是否作为 Stage2B seed**（强/弱），避免「主线近邻词被压成普通 keep」「泛词与近邻混桶」「扩散仅依赖极少数强 seed」。

| 层级 | 含义 | Stage2B | Stage3 倾向 |
|------|------|-----------|----------------|
| **primary_expandable** | 强主线，可强扩散 | 强 seed（`stage2b_seed_tier=strong`，`DENSE_MAX_PER_PRIMARY`） | core / primary-like |
| **primary_support_seed** | 主线近邻，可弱扩散 | 弱 seed（`tier=weak`，`SEED_SCORE_MIN_WEAK`，dense **max_keep=1**、**top_k** 收紧） | support（可信） |
| **primary_support_keep** | 支线/对象/场景向，保留不扩 | 不参与 | support |
| **risky_keep** | 高风险保留，弱相关或泛/歧义 | 不参与 | **默认 `risky`**（`_assign_stage3_bucket` 显式消费 `primary_bucket==risky_keep`） |
| **reject** | 淘汰 | 不参与 | 不进入 |

**判定顺序（实现，axis 版）**：① 坏分支 → `reject`；② `primary_ok` 不成立 → `best_sim≥STAGE2A_RISKY_KEEP_MIN_BEST_SIM` 则 **`risky_keep`**，否则 **`reject`**；③ `primary_ok` 成立后计算 **`axis_consistency` / `effective_mainline`** 与 **`conditioned_only`**，按阈链落桶：`object_or_poly_bad` → **`reject`**；**强主线**（高 effective + identity/family/ctx + 非 conditioned_only）→ **`primary_expandable`**（reason `axis_strong_mainline`）；**主线近邻** → **`primary_support_seed`** 或（偏弱 / 仅 conditioned）**`primary_support_keep`**；**可留不扩** → **`primary_support_keep`**（`axis_keep_no_expand`）；**弱证据** → **`risky_keep`** 或 **`low_mainline` → `reject`**；④ **无真实 `conditioned_sim` 或 `family_fallback_only`**：仍在 **`judge_primary_and_expandability`** 内 **关扩散** 并入 keep；⑤ **`select_primary_per_anchor`**：**两阶段 reconcile**——**组内头名 + axis 间隔**（`STAGE2A_PROMOTE_MIN_AXIS_GAP`）+ **`judge_expandable_stable`** 门控 **keep/sd→exp** 的最后一跳；**维持** judge **`primary_expandable`** 仍须 **强轴**；否则 **sd/sk**；⑥ **reconcile 之后、若仍无 `primary_expandable`**：**极小家族收缩**（sd/sk 各≤1，rk 仅兜底）— 见 **`[Stage2A no-expandable shrink audit]`**。reason 含 **`keep_promote_blocked_*` / `seed_promote_blocked_*`**。常量：`SEED_AXIS_CONSISTENCY_STRONG_MIN`、`STAGE2A_PROMOTE_MIN_AXIS_GAP`、`SEED_SCORE_MIN_WEAK`、`DENSE_MAX_PER_PRIMARY_WEAK` 等。

**`select_primary_per_anchor`** 输出显式四列表 + **`primary_keep_no_expand`（并集）**；**`merge_stage2a_primary`** 仅遍历四列表，避免并集重复计数。

**审计与口径（2A→2B→3 贯通，避免「primary 数」混称 seed / keep）**

| 打印块 | 位置 | 作用 |
|--------|------|------|
| `[Stage2A anchor bucket summary]` | `select_primary_per_anchor` 末尾 | 每锚四段计数 + `winner_for_stage2b`（强/弱 seed 候选列表），不把 risky 与 expandable 混为一列「primary」 |
| `[Stage2A global bucket summary]` | `merge_stage2a_primary` 之后 | 跨锚窄表：`term \| primary_bucket \| stage2b_seed_tier \| source_types \| parent_anchor \| reason`，总览弱 seed 与 support_keep 分流 |
| 每锚汇总表 | `stage2_generate_academic_terms` | 表头：`候选数 \| landings \| exp \| sd \| sk \| rk`（与 2A 块一致）；**landings** = **carryover_terms** 条数（2A 非 reject 保留项总数，`merge_primary_and_support_terms` 始终传入全集） |
| `[Stage2] … carryover_terms=` / **`[Stage2B input audit]`** | 每锚进入 2B 时 | **carryover_terms**（含 exp/sd/sk/rk 分解）与 **seed_candidates**（仅 `stage2b_seed_tier∈{strong,weak}`）；与「diffusion 分母」对齐 |
| `[Stage2B seed tier audit]` | 每锚对 **seed_candidates** 跑完 `check_seed_eligibility` 后 | 仅 **强/弱 seed 候选** 行：`tier` + `eligible` + `seed_score` + `detail`（不再有大量 `tier=none` 的 `stage2a_not_seed` 刷屏） |
| **`[Stage2A group rank]`** | **`select_primary_per_anchor` Pass2 前** | 每锚 **pre_bucket / axis_seed / mainline_pref / jd** 排序表；**axis_gap_ok**、**judge_exp_stable** |
| **`[Stage2A promote audit]`** | **keep/sd 且 strong+axis** | **group_rank**、**top1_term**、**axis_gap**、**promote_allowed**、**promote_block**、final reason |
| **`[Stage2A axis audit]`** | **`select_primary_per_anchor`** 内（`LABEL_EXPANSION_DEBUG` ∧ **`STAGE2_RULING_DEBUG`**） | **axis / effective_mainline / `axis_consistency_seed`** / conditioned_only / final_bucket / reason |
| **`[Stage2A final bucket reconcile]`** | **`select_primary_per_anchor` reconcile 后** | `pre_bucket`→`final_bucket` + **`axis_consistency_seed`** + **`group_rank`** + **`strong_axis_min`** |
| **`[Stage2A expand deny summary]`** | 同上，**最终为 `primary_support_keep` 且 `primary_ok`** | **deny_main** + **axis_seed (min_strong)** + **group_rank** + risk 分解 |
| **`[Stage2B weak-seed audit]`** | **`check_seed_eligibility`**、`tier=weak` | **axis_consistency_seed**、**conditioned_only**、**eligible** 与 **block_reason**（降噪：同左 `STAGE2_RULING_DEBUG`） |
| **`[Stage2B blocked weak seed audit]`** | **`stage2_generate_academic_terms`** 内，对 **seed_candidates** 跑完 `check_seed_eligibility` 之后 | **仅 tier=weak 且 eligible=False**：**term / axis_consistency_seed / seed_score / conditioned_only / g·p·o / blocked_reason**（**`LABEL_EXPANSION_DEBUG`** 即打，与 tier 总表互补） |
| **`[Stage2A no-expandable shrink audit]`** | **`select_primary_per_anchor`**、`primary_expandable` 为空且存在 sd/sk/rk 时 | **收缩前后**各桶 **count + 桶内前 2 term**、**policy**、**why_keep_risky_fallback**（**`LABEL_EXPANSION_DEBUG`**） |
| `[Stage2A final bucket reconcile]` | `select_primary_per_anchor` 内，**conditioned 配额** + **`_finalize_stage2b_seed_tiers` 之后** | 仅当 **pre_bucket ≠ final_bucket**（与配额前快照比）：解释 Focus/Debug 里看似 `support_seed`、最终 `[Stage2A select]` 却进 `support_keep` 的原因（常见 `reason=weak_seed_requires_similar_to_not_conditioned_only`） |
| **`[Stage2A merge evidence detail]`** | `stage2_generate_academic_terms`（`label_expansion`） | 默认只打印 **support_seed / support_keep** 与 **来源近似 conditioned_only**；**`STAGE2A_MERGE_EVIDENCE_DETAIL_FOCUS_TERMS`** 非空时再按 term 补行 |
| `[Stage2B] merge_primary_and_support_terms carryover=…` | `merge_primary_and_support_terms` 调试出口（**`emit_merge_debug=True` 默认**；紧凑无 seed 路径为 **False**） | 首字段为 **carryover** 条数，与 `primary=` 旧口径切断，避免与「仅 seed」混淆 |
| **`[Stage2->3 field audit]`** | **`stage2_expansion._expanded_to_raw_candidates` 末尾** | 开 **`LABEL_EXPANSION_DEBUG`** 时仅打 **前 10 条**：`term`、`primary_bucket`、`can_expand_from_2a`、`fallback_primary`；用于确认 Stage3 聚合前 **2A 桶是否已写入 raw dict**（若此处为空，问题在 Stage2 链路；若有而 Stage3 审计仍空，问题在去重合并） |
| `[Stage2B] anchor=… carryover=… seed_candidates=0 … merged=…` | `seed_candidates=0` 且 **未**开 `STAGE2_NOISY_DEBUG` | 无扩散锚点 **单行折叠**；深度诊断仍开 `STAGE2_NOISY_DEBUG` 走完整 2B 审计 |
| **`[Stage2B no-seed reason]`** | **仅** `seed_candidates=0`（紧凑与 NOISY 全量路径） | **`reason`**：`all_carryover_are_nonseed_buckets`（carryover 无 exp/sd 语义桶）或 **`seedish_bucket_but_tier_none_after_2a_finalize`**；**`buckets`** 为 `exp`/`sd`/`sk`/`rk`/`ot` 逗号缩写 |
| `check_seed_eligibility(..., emit_seed_factors=False)` | `expand_from_vocab_dense_neighbors` 内 | dense 内二次复核 seed 时不重复打印 `[Stage2B seed factors]`（主流程已打印过） |
| `[Stage3 support subtype audit]` | `stage3_build_score_map` 之后 | 仅 **stage3_bucket=support**：拆 **primary_support_seed / primary_support_keep** 与 **conditioned_only**，对照最终序 |
| **`[Stage3 duplicate merge audit]`** | **`_merge_stage3_duplicates` 返回前**（`STAGE3_DUPLICATE_MERGE_AUDIT`） | 仅 **`anchor_count≥2`**：各来源 **`primary_buckets_raw` / `role_in_anchor_raw` / `can_expand_raw`** 与合并后 **`merged_primary_bucket`、`merged_can_expand`、`stage3_merge_source`**（**`strongest_bucket_aggregate`**）对照 |
| **`[Stage3 core-miss audit]`** | **`run_stage3_dual_gate` 内**（`STAGE3_CORE_MISS_AUDIT` ∧ **`STAGE3_AUDIT_DEBUG`**） | **`primary_expandable` ∧ `stage3_bucket≠core`**：`identity_factor`、`jd_align`、**`core_miss_or_cap`**（**`stage3_core_miss_reason`** 或 **`stage3_core_cap_reason`**） |
| **`[Stage3 rerank summary]`** | **`stage3_build_score_map` 末尾**（`DEBUG_LABEL_PATH`） | 增加 **`primary_bucket`** 列，与 **bucket / conditioned_only** 同屏对照 |
| **`[Stage3 topic_gate]`** | `_run_stage3_dual_gate` 内 | **`STAGE3_DEBUG_FOCUS_TERMS` 非空**：仅对命中词逐条 **`bypass primary-like`**；**为空**：只打一行 **`bypass primary-like count=N`** |
| **`[paper_term_selection gate]`** | **`_print_paper_term_selection_audit`**（`STAGE3_PAPER_GATE_DEBUG`，**默认 False 降噪**） | 逐项：同上，并补 **`paper_anchor_*`** 规范化字段；**主裁决顺序**：**weak support** → **`risky_side_block`** → **JD 主轴门（core：`rk∨a_sc`；否 → near-miss 进池）** → **`support_pool` 合并** → **`[Stage3 paper gate summary]`** → **`paper_select_score` + family + `dynamic_floor`** |
| **`[Stage3 unified score breakdown]`** | **`stage3_build_score_map`** 后（`STAGE3_UNIFIED_SCORE_DEBUG`） | 默认仅 **Top `STAGE3_UNIFIED_SCORE_DEBUG_TOP_K`（默认 3）**；**`STAGE3_DEBUG_FOCUS_TERMS` 非空**时先只打焦点词再截断。连续分项 → **final_score**；**paper 入选/排序优先看 support 软放行 / pool penalty / centrality / gate summary** |
| **`[Stage3 paper gate summary]`** | **`select_terms_for_paper_recall`**（`support_pool` 并入后；main-axis 明细表前/后依序） | 一行：`core_at_main_axis_gate`、**`axis_direct_core`、`support_pool_n`、`merged_eligible_n`**、**`support_soft_admit(weak_sup)`**、**`core_axis_near_miss_soft_admit`**、**`blocked_main_axis_hard=0`**、`blocked_support`、`blocked_risky` |
| **`[Stage3 support soft-admit audit]`** | **`select_terms_for_paper_recall`**（`weak_support_contamination` 路径逐条） | **term / primary_bucket / ml / exp / psf / papers_bef / admitted / reason**（**`STAGE3_PAPER_CUTOFF_AUDIT`**） |
| **`[Stage3 paper gate reject audit]`** | **`select_terms_for_paper_recall`**（合并池前） | 前置硬挡按原因聚类：**`conditioned_only_single_anchor_block`**、**`fallback_primary_block`**、**`weak_support_contamination_block`**、**`risky_side_block`** 等 |
| **`[Stage3 paper quota audit]`** | **选词结束后** | **`max_terms`、`support_cap`、`selected_primary_lane`、`selected_support_lane`、`support_quota_full_count`** |
| **`[Stage3 paper cutoff]`** | **`select_terms_for_paper_recall` 末尾**（`STAGE3_PAPER_CUTOFF_AUDIT`） | **合并后 eligible**：**lane-tier 全局序**下做 `dynamic_floor`、`family_key`、截断（非「全表单纯按 p_sel 一条序」）；`cutoff_reason`（含 **`support_quota_full`** / **`below_dynamic_floor`** / **`family_duplicate_block`** / **`swapped_out_by_core_near_miss`** / **`core_near_miss_replace_support_soft_admit`** / **`tail_expand_*`** 等）；列含 **final / paper_readiness / paper_select_score** |
| **`[Stage3 paper selection audit]`** | **同上** | **窄表**（与 cutoff 重叠）：**仅 `STAGE3_DEBUG_FOCUS_TERMS` 非空** 时打印 **命中词** 行；**`STAGE3_ELIGIBLE_CORE_CLOSE_CALL_AUDIT`**（默认 False）控制 **`[Stage3 eligible core close-call audit]`** |
| **`[Stage3 paper lane tier audit]`** | **`select_terms_for_paper_recall`**（**floor 计算之后**、**centrality 之前**） | **全 eligible** 按 **`paper_select_lane_tier`** 与**真实扫描序**输出：**term / tier / p_sel / a_sc / rk / ml / exp / lane / parent_anchor**（阈值见 **`PAPER_SELECT_STRONG_MAIN_AXIS_*`**） |
| **`[Stage3 paper selected composition]`** | **同上**（**swap + tail 之后**、**cutoff 表之前**） | **`selected_total=`** 及 **`strong_main_axis_core=`**、**`bonus_core=`**、**`support_lane=`**、**`tail_expand_*=`** 等 **`paper_select_lane_tier`** 计数 |
| **`[Stage3 paper floor block audit]`** | **同上** | **仅** `selected=False` ∧ **`paper_cutoff_reason/reject_reason=below_dynamic_floor`** ∧ **`stage3_bucket=core`**：**term、final、paper_readiness、paper_select_score、dynamic_floor、Δfloor（floor−p_sel）、mainline_hits、can_expand、parent_anchor、parent_anchor_score**；专抓「core + 主线 + 可惜扩却被 floor 误杀」（如 *robot control*） |
| **`[Stage3 paper near-miss audit]`** | **同上** | **eligible 但未入选**：`paper_select_score` 贴近 **dynamic_floor** 或 **末席/top5 入选分** 的词；列 **bucket、`primary_bucket`、Δ末席、`parent_anchor`、`paper_support_reason`、a_sc、rk、reject/cutoff** |
| **`[Stage3 paper swap audit]`** | **`select_terms_for_paper_recall`**（**主 cutoff 之后**、**`PAPER_RECALL_TAIL_EXPAND_*` 之前**；`STAGE3_PAPER_CUTOFF_AUDIT`） | 单行：**`swapped`**、**out/in term**、**out_p_sel / in_p_sel**、**delta**、**floor**；**与 tail 触发无关**（**`selected>3`** 且 **已有 support 入选** 时仍可能对 **core near-miss** 与 **support soft-admit** 做一对一双替换）。常量：**`PAPER_RECALL_CORE_NEAR_MISS_SWAP_MAX_BELOW_FLOOR`**、**`PAPER_RECALL_SUPPORT_SWAP_SCORE_MARGIN`** |
| **`[Stage3 tail expand audit]`** | **`select_terms_for_paper_recall`**（**主** floor+family+support_cap 截断之后） | **触发**：**`len(selected)≤3`** 或 **主选阶段 `selected_support==0`**；**补位**：未入选、**`p_sel≥floor−Δ`**、禁 risky/fallback/conditioned_only；**分层优先级**：**core ∧ 主截断原因 `below_dynamic_floor` ∧ ml≥1 ∧ 可扩** → **`paper_recall_quota_lane=support`** 且 **`support_seed/support_keep_soft_admit`**（**`paper_support_reason`**，gate 常为 **`support_soft_admit`**）→ **其余**（若开 **`PAPER_RECALL_TAIL_EXPAND_REQUIRE_CORE_OR_SEED`** 则须 **core 或 `primary_support_seed`**）；至多 **`PAPER_RECALL_TAIL_EXPAND_MAX_EXTRA`**；总长 **`max_terms+MAX_EXTRA`**；仍 **family** 去重、**support_cap**；**`p_order`** 日志：**core_near_miss(1)>support_soft_admit(2)>other(3)** |
| **`[Stage3 paper main-axis gate audit]`** | **`select_terms_for_paper_recall`**（合并池前构造完成、readiness 前；`STAGE3_PAPER_CUTOFF_AUDIT`） | **全量 input 顺序**：**term、bucket、parent_anchor、a_sc、rk、axis_pass、reason**；core **`(rk≤MAX)∨(a_sc≥MIN)`** 过线为 **`True`**；**双弱** 为 **`False`** 且 **reason=`core_axis_near_miss_soft_admit`**（已入 **`support_pool`**，非硬拒）；更靠前被拒为 **`n/a`** |
| **`[Stage3 paper centrality audit]`** | **`select_terms_for_paper_recall`**（`paper_select_score` 排好序后、family+floor 选词前；`STAGE3_PAPER_CUTOFF_AUDIT`） | **按全局 `p_sel` 取 Top12**（**与 lane-tier 扫描序不同**，便于对照「若只看分会谁在前」）；增列 **`p_bef`**、**`paper_select_score_base`、`+bonus`、`p_sel`**（**−pool_penalty**） |
| **`[Stage3 paper pool penalty audit]`** | **同上**（紧跟 centrality） | **Top16 eligible**：**papers_bef、base、+bonus、−pool_pen、final `p_sel`**；对照 **`PAPER_SELECT_POOL_PENALTY_*`** 开关与系数 |
| **`[Stage3 support contamination]`** | **`select_terms_for_paper_recall` 末尾**（`STAGE3_SUPPORT_CONTAMINATION_AUDIT`） | 已入 paper 且 **stage3_bucket=support**、结构仍可疑的摘要；行内带 **`paper_support_gate`**（结构门全挡住时此处应接近空，仅作 Stage4 前残漏报警） |
| **`[Stage5 paper fanout audit]`** | **`stage5_author_rank`**（`papers_for_agg` 已 built、**percentile_compress** 之后 **`accumulate_author_scores` 之前**） | Top 稿：**`wid`、`score_before_fanout`、作者数、`fanout_factor`、`score`（乘后）、hit_terms**（**`STAGE5_FANOUT_AUDIT`**） |
| **`[Stage5 support-only author penalty audit]`** | **`stage5_author_rank`**（**`structure_mult_total` 之后**、**`AUTHOR_BEST_PAPER_MIN_RATIO` 之前**） | **`author_id | raw_score | sup_share | top_pri_c | sup_only_papers | sup_with_pri_papers | penalty | after`**（**`STAGE5_SUPPORT_ONLY_PENALTY_AUDIT`**；开关 **`STAGE5_SUPPORT_ONLY_AUTHOR_PENALTY_ENABLED`**） |
| **`[Stage5 support dominance audit]`** | **`stage5_author_rank`** | Top 作者的 primary vs support 分解；**默认关闭**，设 **`STAGE5_SUPPORT_DOMINANCE_AUDIT=1`** 开启（**paper_primary** 主导 JD 时多为噪音） |

**Stage3 `stage3_build_score_map` 与 paper 准入（2025-03 修订：统一连续分）**

- **已移除**：按 **core/support/risky**、**primary_support_seed/keep**、**conditioned_only** 链式 **`score_mult`** 与 **按桶 `cross_bonus`**；**不再**用 **`_support_grounded_enough`** 决定 paper。
- **`stage3_build_score_map`**：对每条候选计算 **统一连续分**（权重见 `stage3_term_filtering.py` 顶部 **`STAGE3_UNIFIED_W_*`**），clamp 到 **[0,1]**，写入 **`final_score`** 与 **`stage3_unified_breakdown`**；**`stage3_explain.mainline_risk_penalty` / `cross_anchor_score_bonus` 固定 1.0**（兼容旧审计列）。随后 **`_apply_family_role_constraints`**（与原顺序解耦：**先统一分、再家族微调**）。
- **`select_terms_for_paper_recall`**：**合并前**硬挡 **单锚 conditioned_only**、**fallback_primary**、**`weak_support_contamination`（未达软放行）**、**`risky_side_block`**。**`JD 全局主轴门`（仅 core）**：**`rk≤MAX ∨ a_sc≥MIN`** 过线 → **`paper_recall_quota_lane=primary`**；**双弱** → **`core_axis_near_miss_soft_admit`**（×**`PAPER_CORE_AXIS_NEAR_MISS_FACTOR`**）进 **`support_pool`**。**凡未硬挡的 support / risky / 其它桶先入 `support_pool`**（含 **`support_lane` / `risky_support_lane`**），再 **`eligible.extend(support_pool)`**。**`[Stage3 paper gate reject audit]`**、**`[Stage3 paper gate summary]`**（**`axis_direct_core`、`support_pool_n`、`merged_eligible_n`** 等）。**`paper_select_score` 公式不变**；合并池内对每条写入 **`paper_select_lane_tier`**：**`strong_main_axis_core`**（**`stage3_bucket=core` ∧ `paper_recall_quota_lane=primary` ∧（`a_sc≥PAPER_SELECT_STRONG_MAIN_AXIS_SCORE_MIN` ∨ `rk≤PAPER_SELECT_STRONG_MAIN_AXIS_MAX_RANK` ∨（`ml≥1`∧可扩∧`a_sc≥PAPER_SELECT_STRONG_MAIN_AXIS_ML_EXP_SCORE_MIN`∧`rk≤PAPER_SELECT_STRONG_MAIN_AXIS_ML_EXP_MAX_RANK`）））、**`bonus_core`**（其余 **primary lane** 的 **core**）、**`support_lane`** / **`other_eligible`**。**全局遍历序** = 上述 tier 拼接，**各段内**仍按 **`paper_select_score` 降序**；其后 **`dynamic_floor` + `family_key` + `support_cap`** 扫描。**入选**时 **`paper_recall_quota_lane=support`** 计 **support_cap**（**`support_quota_full`**）。候选先 **`_apply_paper_readiness_for_recall`**（含 **`paper_support_factor`**），再 **`paper_select_score`**：
  - **底分**：`base = final_score × paper_readiness`。其中 **`paper_readiness`** 仍由 **bucket / mainline / 锚数 / 可扩 / conditioned_only / term_role / `_paper_grounding_score`** 连乘，再乘 **锚点主线先验（弱化 rank 主导）**：
    - **`anchor_score_norm`** = $\mathrm{clip}(\texttt{best\_parent\_anchor\_final\_score}/\texttt{PAPER\_READINESS\_ANCHOR\_SCORE\_NORM},\,0,\,1)$（与 Step2 **`PreparedAnchor.final_anchor_score`** 透传一致）。
    - **`rank_smooth`** = $1/(1+\lambda\cdot\max(\texttt{rank}-1,0))$，**λ=`PAPER_READINESS_ANCHOR_RANK_SMOOTH_LAMBDA`（默认 0.04）**。
    - **`anchor_prior`** → **`paper_readiness *= (blend_lo + blend_hi * anchor_prior)`**（系数见 **`PAPER_READINESS_ANCHOR_*`**）。
    - **结构保底**：**`stage3_bucket=core` ∧ `mainline_hits≥1` ∧ 可扩** → **`paper_readiness ≥ PAPER_READINESS_CORE_MAINLINE_EXPAND_MIN`**。
    - **support seed 结构保底**：**`primary_bucket=primary_support_seed` ∧ ml≥1 ∧ 可扩 ∧ ¬conditioned_only** → **`≥ PAPER_READINESS_SUPPORT_SEED_MAINLINE_EXPAND_MIN`**（后再允许 × **`PAPER_SUPPORT_SEED_FACTOR`**）。
  - **加性注入（仅 paper 排序）**：在 **`base`** 上叠加 **`bucket_bonus`**（`primary_expandable` / `primary_support_seed` / `primary_support_keep` / `risky_keep` 等）、**`mainline_bonus`**、**`expand_bonus`**（`can_expand_from_2a` 或 `can_expand`）、**父锚名次档**（rk≤3/6/10）、**`anchor_score_bonus = PAPER_SELECT_ANCHOR_SCORE_BONUS_COEF × norm(父锚分)`**、**`role_bonus`**（`primary` / `primary_side` / `dense_expansion`）；系数见 **`PAPER_SELECT_*`**。若缺 **`parent_anchor_*`** 则 **`best_parent_anchor_*`** 回填。写入 **`paper_select_score_base`**、**`paper_select_score_bonus_total`**。
  - **候选池规模惩罚（连续、非词表）**：**`pool_size`** = **`papers_before_filter`** 或回落 **`degree_w`**（至少 1）；**`pool_penalty = min(PAPER_SELECT_POOL_PENALTY_MAX, PAPER_SELECT_POOL_PENALTY_LOG_COEF × log10(pool_size))`**（**`PAPER_SELECT_POOL_PENALTY_ENABLED`** 关则 0）。**`paper_select_score = base + bonus_total − pool_penalty`**（clamp≥0）。写入 **`paper_select_pool_size`**、**`paper_select_pool_penalty`**。少词表场景下缓释 **RL / 大池泛词** 凭规模抬高 **`paper_select_score` 与 `dynamic_floor`**。
  - **readiness·support seed**：**`primary_support_seed` ∧ ml≥1 ∧ 可扩 ∧ ¬conditioned_only** 时 **`paper_readiness ≥ PAPER_READINESS_SUPPORT_SEED_MAINLINE_EXPAND_MIN`**；**`support_seed_soft_admit`** 再 × **`PAPER_SUPPORT_SEED_FACTOR`（默认 0.82）**。
  - **主 cutoff 后、tail 前（换位，`PAPER_RECALL_*_SWAP_*`）**：在 **不改 floor 公式** 的前提下，若存在 **未入选**、**`stage3_bucket=core`**、主截断/拒绝原因含 **`below_dynamic_floor`**、**ml≥1**、**可扩**、且 **`p_sel≥floor−PAPER_RECALL_CORE_NEAR_MISS_SWAP_MAX_BELOW_FLOOR`** 的 **core near-miss**，同时 **已入选** 中最弱的一条为 **`stage3_bucket=support`** 且 **`paper_recall_quota_lane=support`** 的 **weak support soft-admit**（**`paper_support_reason`** 为 **`support_seed_soft_admit`/`support_keep_soft_admit`** 或 **`paper_support_gate=support_soft_admit`**），且 **`in_psel ≥ out_psel − PAPER_RECALL_SUPPORT_SWAP_SCORE_MARGIN`**，则 **一对一替换**：落选侧 **`swapped_out_by_core_near_miss`**，入选侧 **`core_near_miss_replace_support_soft_admit`**，并 **`paper_recall_quota_lane=primary`** / **`retrieval_role=paper_primary`**；更新 **`used_family`** 与 **`selected_support`**。与 **`PAPER_RECALL_TAIL_EXPAND_*` 是否触发无关**（解决「词表够宽、tail 不触发，但 core 仍被 support 压过」）。
  - **尾部补位（`PAPER_RECALL_TAIL_EXPAND_*`）**：主 **`dynamic_floor` + family + support_cap** 得到 **`selected`**（及上述换位）后，若 **`len(selected)≤3`** 或 **`selected_support==0`**，再从 **near-miss**（**`p_sel≥floor−Δ`**）补 **≤`MAX_EXTRA`**，总长至 **`max_terms + MAX_EXTRA`**。**分层优先级**（专治「support 软词抢在 core near-miss 前入栏」）：① **core** 且主截断已标 **`below_dynamic_floor`**、**ml≥1**、**可扩** → 先补，记 **`tail_expand_core_near_miss`**；② **`support_seed_soft_admit` / `support_keep_soft_admit`**（weak support 软放行，`paper_support_gate` 多为 **`support_soft_admit`**）→ 后补，记 **`tail_expand_support_soft_admit`**；③ **其余** 候选（若 **`PAPER_RECALL_TAIL_EXPAND_REQUIRE_CORE_OR_SEED`** 则须 **core 或 `primary_support_seed`**）记 **`tail_expand_other`**。仍 **family** 去重、**support_cap**；禁 **risky / fallback_primary / conditioned_only** 借尾窗混入。不修改 **gate** 与 **`paper_select_score`** 定义。
  - 审计：**`[Stage3 paper gate reject audit]`**、**`[Stage3 support soft-admit audit]`**、**`[Stage3 paper lane tier audit]`**、**`[Stage3 paper centrality audit]`**、**`[Stage3 paper pool penalty audit]`**（**eligible core close-call** 由 **`STAGE3_ELIGIBLE_CORE_CLOSE_CALL_AUDIT`** 控制，默认关）、**`[Stage3 paper swap audit]`**、**`[Stage3 tail expand audit]`**、**`[Stage3 paper selected composition]`**、**`[Stage3 paper quota audit]`**（**primary/support lane**）、 **`[Stage3 paper main-axis gate audit]`**、**`[Stage3 paper gate summary]`**、**`paper_anchor_*`** 等。**`[Stage3 paper selection audit]`**：**仅 `STAGE3_DEBUG_FOCUS_TERMS` 非空** 时打印焦点词；增列 **`psf` / `sup_reason`**。**主轴门**：**OR**；双弱 **降级进池**，非整词黑名单。**`_paper_recall_dynamic_floor(..., score_key="paper_select_score")`**；**`family_key` 去重**；**`PAPER_RECALL_MAX_TERMS`**；总长可至 **`max_terms + PAPER_RECALL_TAIL_EXPAND_MAX_EXTRA`**。**`retrieval_role`**：**`paper_recall_quota_lane=primary`** → `paper_primary`，否则 **`paper_support`**（**含 bucket 仍为 core 但 near-miss 降级者**）。

---

**Stage2 最小收尾（三点闭环，`label_expansion.py`）**

| 优先级 | 改动 | 说明 |
|--------|------|------|
| **1** | **primary 资格 ≠ expand / seed 资格** | `judge_primary_and_expandability`：axis 初桶 + **无真实 conditioned** 时关扩。`select_primary_per_anchor`：**similar_to reconcile** + **`axis_consistency_seed` 与 Stage2B strong 同阈**（防 promote 过宽）+ **无 `primary_expandable` 时极小家族收口**（sd/sk 各≤1，rk 仅兜底，见 **`[Stage2A no-expandable shrink audit]`**）。 |
| **2** | **空主线锚点 fallback** | 仅当 **本锚点 0-primary**：`anchor_allows_fallback_primary`（**非 acronym / 非高歧义类型**、**canonical_academic_like**、且 `final_anchor_score≥STAGE2A_FALLBACK_ANCHOR_MIN_SCORE` **或** local_phrases/co_anchor 命中数达标）→ `pick_fallback_primary_for_anchor` 从 `kept` 池按 `fallback_score` 选 1 条，标记 **`primary_fallback_keep_no_expand` / `fallback_primary`**，**不进入 Stage2B**。 |
| **3** | **Stage2B：tier + axis 分流** | `check_seed_eligibility`：**`blocked_by_2a`**；**`stage2b_seed_tier=none`** → `tier_none`；**strong**：`seed_score` + **`axis_consistency_seed≥0.45`**；**weak**：更高分与轴阈 + **generic/poly/object** 上限 + **`conditioned_only_weak_seed`**；`fallback_primary` 不扩；**semantic_mismatch_seed** 硬挡。扩散父节点列表 = **`eligible=True`** 的项。 |
| **4** | **Stage3：单锚可扩旁枝压 support** | `stage3_term_filtering._assign_stage3_bucket`：当 **`anchor_count==1` ∧ `mainline_hits≥1` ∧ `can_expand`** 且 **family_centrality / path_topic_consistency 双弱**，或 **object/drift/generic/poly** 超结构阈（`STAGE3_SUPPORT_DEMOTE_*`），原应进 **support** 的词条 **降级 `risky`**，并打 `single_anchor_expand_branch_demote`，减轻 RL/医疗机器人/抓手等 **support 污染 paper terms**（非词表硬编码）。 |
| **5** | **Stage3：Stage2A 语义透传 + 分桶/paper 对齐** | 同文件 **`_assign_stage3_bucket`** / **`select_terms_for_paper_recall`** / **`_collect_stage3_bucket_reason_flags`**：用 **`primary_bucket` / `primary_reason` / `fallback_primary` / `mainline_candidate` / `can_expand_local`** 区分 **fallback primary**（默认 **risky**、paper **`fallback_primary_block`**）与 **locked mainline**（**support**、paper 侧 **has_mainline_support**），避免 **primary_keep_no_expand** / **usable_mainline_no_expand** 因 mainline_hits=0∧¬can_expand 误标 **risky** 或被 paper 门误拒；日志 flag 增补 **`locked_mainline_no_expand`**、**`fallback_primary`**。 |

**建议日志**：`[Stage2A select]`、**`[Stage2A group rank]`**、**`[Stage2A promote audit]`**、**`[Stage2A final bucket reconcile]`**、`[Stage2A anchor bucket summary]`、`[Stage2A global bucket summary]`、`[Stage2A fallback]`、**`[Stage2] carryover_terms` / `[Stage2B input audit]`**（有 seed 时）、**无 seed：`[Stage2B no-seed reason]` + 单行 `[Stage2B] anchor=… merged=…`**、`[Stage2B seed gate]`（NOISY）/ **`[Stage2B seed tier audit]`** 与 **`[Stage2B seed factors]`**（均需 **`STAGE2_NOISY_DEBUG=True`**，瓶颈在 Stage3 时默认降噪）、**有 seed 时 `[Stage2B] anchor=… eligible…` 仅打一条**（已去掉与 `debug_print(2)` 的重复汇总）、**`merge_primary_and_support_terms carryover=…`**、`[Stage2B] diffusion=…`（`debug_print(1)` 保留）、`[Stage3 rerank summary]`、**`[Stage3 support subtype audit]`**、`[Stage3 bucket reason]` 中的 `bucket_reason_flags`。**`[Stage2A Focus Debug]`**：**不再在 `judge_primary_and_expandability` 出口即打**；仅在 **`[Stage2A final bucket reconcile]` 已发生跨桶**（pre_bucket≠final_bucket）、**`STAGE2_NOISY_DEBUG=True`** 且命中 **`DEBUG_STAGE2A_FOCUS_*`** 时随 reconcile 行后输出大块 Focus，避免「命中焦点就刷屏」。**`[Stage2A Commit Debug]`** 仍仅 **`STAGE2_NOISY_DEBUG=True`** 且 focus 命中时输出。**弱 seed 被 2B 挡**：**`[Stage2B blocked weak seed audit]`**（`LABEL_EXPANSION_DEBUG`，窄表）。Stage3 主视角：**`[Stage3 paper centrality audit]`**、**`[Stage3 paper selection audit]`**、**`[Stage3 paper near-miss audit]`**。

**标签路降噪**：`label_path` 在 verbose 下打印 **`[标签路-similar_to 原始候选]`** 时，**仅列出前 10 条**明细，避免图谱近邻刷屏（全量仍在 **`debug_info.similar_to_raw_rows`**）。

**相关常量**：`STAGE2A_CONDITIONED_SIM_EPS`、…、`STAGE3_SUPPORT_DEMOTE_*`；**paper 最后一跳**：**`PAPER_SELECT_*`** 加性 bonus（与 **`PAPER_READINESS_*`** 分工：后者只进 **`paper_readiness`**）+ **`PAPER_SELECT_POOL_PENALTY_*`**（**`paper_select_score`** 减项）+ **`PAPER_SELECT_STRONG_MAIN_AXIS_*`**（**强主轴 core / bonus_core 分层**，与 **`STAGE3_ELIGIBLE_CORE_CLOSE_CALL_AUDIT`**、**`STAGE3_DEBUG_FOCUS_TERMS`** 配合降噪）。

**Stage2A/2B 核心收口（`label_expansion.py`，含 `check_seed_eligibility`）**

| 函数 | 目的与要点 |
|------|------------|
| **`judge_primary_and_expandability`** | **`primary_ok` 之后**：用 **`axis_consistency` / `effective_mainline`**（连续特征 + 风险罚项）五层落桶：**`axis_strong_mainline`** → `primary_expandable`；**`axis_neighbor_weak_seed`** → `primary_support_seed`（扩散权受 **effective_mainline** 与 **conditioned_only** 约束）；**keep / risky / reject** 见实现分支；**无真实 conditioned** → 扩散关闭并入 **`primary_support_keep`**。已移除旧 **`strong_mainline_direct` / `control_mainline_escape` / `_classify_keepish_primary_bucket`** 主链。 |
| **`select_primary_per_anchor`** | **`judge_*` 之后**：**组排序 + final reconcile**；**keep/sd→exp** 须 **top1 + axis 间隔 + 强轴 + 无 judge 稳定强 exp**；**无 exp 时** **sd/sk 各≤1、rk 仅兜底**，**`[Stage2A no-expandable shrink audit]`**；调试 **`[Stage2A group rank]`**、**`[Stage2A promote audit]`**。**`STAGE2_RULING_DEBUG` 开时不再重复** **`[Stage2A primary/expand split]`**。 |
| **`check_seed_eligibility`** | **`blocked_by_2a`**；**strong**：`SEED_SCORE_MIN` + **`axis_consistency_seed≥SEED_AXIS_CONSISTENCY_STRONG_MIN`**；**weak**：`SEED_SCORE_MIN_WEAK` + **`axis_consistency_seed≥0.50`** + 风险上限 + **`conditioned_only_weak_seed`**；**`[Stage2B weak-seed audit]`**。 |
| **`pick_fallback_primary_for_anchor`** | 0-primary 保线（未改本轮）：**lexical/family** + 风险过滤 + **fallback_score**。 |
| **`expand_from_vocab_dense_neighbors`** | **分档 `min_dense_sim` / anchor·family·context_stability**（strong vs weak）；**weak 禁用** **`weak_support_release_for_strong_seed`**；保留强 seed 弱放行与 **domain_fit** 逻辑。 |

**验收日志**：**`[Stage2A axis audit]`**；**`[Stage2B weak-seed audit]`**（weak）；`[Stage2A expand factors]` 中 **`axis_consistency` / `effective_mainline`**；`[Stage2B] Dense expansion funnel` 中 **after_mainline_filter / final_kept**；`[Stage2A fallback]` 泛词保线是否仍偏多。

---

**Stage2A：学术落点（主线优先组内选主，无固定阈值）**

Stage2A **不再做「达线考试」**，改为**主线偏好第一优先级 + 组内相对选主**。核心：不因 `mainline_sim < x`、`role == off` 或「不能扩散」而整组全灭；**primary 生存** 与 **扩散权（`can_expand_from_2a`）** 分离，Stage2B **仅**围绕 **`can_expand_from_2a=True`** 的 primary 扩展；**keep / fallback 只保线、不扩**。

| 项 | 说明 |
|----|------|
| **理念** | **主线第一优先级、无固定阈值**：不做 mainline_sim/role 一票否决；证据只参与组内百分位排序；**排序键**以 mainline_rank 为首位（字典序），生死由「锚点内相对竞赛」决定。 |
| **职责** | ① 构建全局 **主线画像** `build_stage2a_mainline_profile`；② 每锚点召回候选（similar_to + conditioned_vec + alias/exact），**不判死**；③ 建 **cross_anchor 证据** `build_cross_anchor_index`；④ 每锚 **enrich** → **mainline_rank / composite_rank_score**；⑤ **选 primary**：`judge_primary_and_expandability` + `select_primary_per_anchor`（expandable vs keep 解耦；**无 exp 时 sd/sk/rk 极小家族保留**见 shrink audit；可选 **fallback**）；⑥ **merge_stage2a_primary**；⑦ **Stage2B 仅对 `can_expand_from_2a=True` 的 primary 做扩展**。 |
| **候选来源** | **similar_to**（主）+ **conditioned_vec**（补位）+ **alias/exact** 对齐；`collect_landing_candidates` 只负责召回与打标记（domain_fit/hierarchy 等），不 hard reject。 |
| **逻辑流程** | ① `build_stage2a_mainline_profile`；② 每锚 `collect_landing_candidates` → `landing_candidates_to_stage2a`；③ `build_cross_anchor_index`；④ 每锚 `enrich_stage2a_candidates`；⑤ **pre-primary 分桶**（hard_reject 等）；⑥ `select_primary_per_anchor`（必要时 **`pick_fallback_primary_for_anchor`**）；⑦ `merge_stage2a_primary`；⑧ Stage2B：`check_seed_eligibility` 仅 **`can_expand_from_2a`**。 |
| **输入/输出** | 输入：prepared_anchors（可带 **final_anchor_score**，由 `_anchor_skills_to_prepared_anchors` 透传）、active_domain_set、query_vector、query_text、jd_*_ids、label。输出：primary_landings（**can_expand_from_2a**、**fallback_primary**、expandable、mainline_* 等）；Stage2B 仅强主线；raw_candidates `_debug` 透传。 |

**已删除的旧逻辑（不再使用）**

- ~~`mainline_role == "off"` → reject~~
- ~~`mainline_sim < 固定阈值` → reject~~
- ~~`candidate_vec is None` 时 sim=0 再 reject~~
- ~~hierarchy_no_match / domain_no_match 一票否决~~

以上均改为「证据不利」参与组内相对排序或禁扩散，不单独判死。

**数据结构：Stage2ACandidate**

- `tid, term, source`；原始证据：`semantic_score, context_sim, jd_align`，`cross_anchor_support, family_match, hierarchy_consistency, polysemy_risk, isolation_risk, object_like_risk, generic_risk, context_continuity`。
- **mainline_preference**（标量 float）：由 build_mainline_preference(cand) 计算，供 2A 内部排序；旧版 dict 形态仍被 mainline_preference_sort_key 兼容。
- 准入与角色（enrich 阶段）：**retain_mode**（normal / weak_retain / reject）、**suppress_seed**、**admission_reasons**、**role_in_anchor_candidate_pool**（mainline_candidate / secondary_candidate / reject_candidate）。
- 相对排序：`relative_scores`（含 **mainline_rank** 由主线偏好排序赋百分位）、`composite_rank_score`；**anchor_internal_rank**、**sort_key_snapshot**（主线优先排序后的名次与快照）。
- 最终标签：`survive_primary, can_expand, can_expand_from_2a, fallback_primary, reject_reason, role`，**role_in_anchor**（mainline / side / dropped）；**primary 与扩散权分离**：**`can_expand_from_2a=True`** 才允许 Stage2B seed；keep / fallback 仅保留。

**关键函数（无固定阈值）**

- `build_stage2a_mainline_profile(recall, anchors, query_vector, query_text)`：主线画像 centroid + anchor_terms/anchor_vids/query_vector，供主线偏好用，不用于硬 reject。
- `build_mainline_preference(cand)`：仅用候选上的定性字段，返回**标量** mainline_preference（positive − negative），供 2A 内部排序。
- `mainline_preference_sort_key(pref)`：复合主线偏好转排序元组，无新增阈值。
- `build_stage2a_sort_key(cand)`：**主线第一优先级**排序键，(mainline_rank, semantic_rank, context_rank, jd_rank, cross_anchor_rank, family_rank, hier_rank, polysemy_rank, isolation_rank)。
- `assign_relative_scores_within_anchor(candidates)`：含 **mainline_rank** 由 `_assign_mainline_rank_from_preference` 按 mainline_preference 复合键赋百分位。
- `is_competitive_runner_up(best, second, candidates)`：极简版为 second.mainline_rank ≥ 组内最后一名即保留为 side。
- **judge_primary_and_expandability**：**primary** 须 **真实 `conditioned_sim`** 路径或 **多锚高 family** 或 **dual+conditioned**；通过后以 **`axis_consistency`−风险 → `effective_mainline`** 决定 **五层桶** 与 **`can_expand_from_2a`**（含 **无 conditioned 时关扩**）；返回桶 + reason + **axis 快照**。
- **select_primary_per_anchor**：返回 **五列表** + **`primary_keep_no_expand`**；**前置硬拒绝仅** `retain_mode=="reject"`；**obviously_bad_branch / drift_bad / polysemous_no_context** 仍前置 reject；**`judge_primary_and_expandability` + similar_to 主线 reconcile**；**无 `primary_expandable` 时** **sd/sk 各至多 1 条、rk 仅当二者皆空时兜底 1 条**（**`[Stage2A no-expandable shrink audit]`**）。
- **pick_fallback_primary_for_anchor** / **anchor_allows_fallback_primary**：0-primary 时的 **保线 fallback**（不扩）；fallback 池须过 **词形/家族贴近** 与 **结构风险** 过滤，**fallback_score** 含 lexical + family 权重。
- **merge_stage2a_primary**：合并时 **can_expand** 字段取 **`can_expand_from_2a`**；Stage2B 仅 **`can_expand_from_2a` ∧ primary_score≥SEED_SCORE_MIN**。

**调试透传（Stage3 _debug）**

`_expanded_to_raw_candidates` 将以下字段写入 `rec["_debug"]`：mainline_preference、mainline_rank、anchor_internal_rank、survive_primary、can_expand、sort_key_snapshot、role_in_anchor、cross_anchor_support。日志可打印：每锚点主线优先排序结果、主线偏好拆解、Primary Selected、Merged Primary。

**一句话总结**

Stage2A 从「候选平均打分器」改为「主线优先的组内选主器」：**primary 生存**、**强主线扩散权（`can_expand_from_2a`）**、**空锚 fallback** 三线分工；Stage2B 只扩强主线。

---

**Stage2A 候选来源与数据结构（实现细节）**

| 项 | 说明 |
|----|------|
| **候选来源（实现）** | **similar_to** 为主池 + **conditioned_vec** 为上下文复核；`_retrieve_academic_terms_by_conditioned_vec` 返回 context_neighbors + rerank_signals；主池≤1 时补最多 2 个 context_fallback（context_sim≥0.82）。领域过滤使用 `_term_in_active_domains_with_reason`，仅强冲突剔除。 |
| **调用的表/知识图谱** | **Neo4j**：Vocabulary、SIMILAR_TO；**SQLite**：vocabulary_domain_stats、vocabulary_topic_stats；**Faiss**：vocab_index（conditioned_vec 检索）。 |

**Stage2A 上下文纠偏与双空间准入（conditioned_vec 作复核器）**

- **LandingCandidate 新增字段**：`context_sim`（conditioned_vec 下相似度）、`context_supported`（是否≥0.78）、`context_gap`（raw_sim − context_sim）、`source_role`（seed_candidate | context_fallback）、`primary_eligible` / `primary_eligibility_reasons`、`domain_reason`（如 domain_conflict_strong）、**soft_domain_retain**（domain_no_match 时软保留进池，domain_fit×0.85）。
- **`_retrieve_academic_terms_by_conditioned_vec`**：入参增加 `similar_to_candidates`；返回 `(context_neighbors, rerank_signals)`，其中 `rerank_signals[vid] = {context_sim, context_supported, context_gap}`，供 similar_to 候选挂载。
- **`collect_landing_candidates`**：① similar_to 得初始候选；② 调 conditioned_vec 得 context_neighbors + rerank_signals；③ 为 similar_to 候选附加 context 信号并设 source_role=seed_candidate；④ 仅当候选数≤1 时从 context_neighbors 补最多 2 个 context_fallback（context_sim≥0.82）。
- **预期日志/现象**：① **propulsion**：semantic 尚可但 context_sim 低或 context_consistency 不足 → admission 被压或 primary_score 下降；② **motion control vs motion controller**：motion control 的 context_sim 更稳、motion controller 的 context_gap 更大 → motion control 优先；③ **robotic arm vs robot hand**：双空间一致性更强的 robotic arm 被抬升；④ **pathfinding**：raw_sim 高但 context_consistency 不足时不再轻松排第一。

**Stage2A 上下文条件化 + context_gain（最小修改，不新增 Step2.5）**

原则：不推倒重做、不加黑白名单；把 **SIMILAR_TO** 与 **向量索引（conditioned_vec）** 当作同源证据，只补「上下文条件化」那半边。

- **总思路**：保留 2A 原框架（retrieve_academic_term_by_similar_to、_retrieve_academic_terms_by_conditioned_vec、collect_landing_candidates、landing_candidates_to_stage2a、enrich_stage2a_candidates、select_primary_per_anchor、check_seed_eligibility），不拆阶段、不重命名主流程。只做四件事：① 把 conditioned_vec 从「弱补位」升格为「正式证据源」；② 给每个 candidate 显式计算 **context_gain**；③ 组选主时把 context_gain / conditioned 证据纳入主线判定；④ seed 门控再加一层「上下文不增益则不扩散」。
- **PreparedAnchor**：增加 `local_phrases`、`co_anchor_terms`、`jd_snippet`、`surface_vec`、`conditioned_vec`（后者原有）；`_anchor_skills_to_prepared_anchors` 从 anchor_skills / _anchor_ctx 与 label 词向量补全上述字段，供后续双路检索与 context_gain 用。
- **LandingCandidate / Stage2ACandidate**：增加 `surface_sim`、`conditioned_sim`、`context_gain`、`source_set`；merge 时保留双路信息（同一 tid 可同时来自 similar_to 与 conditioned_vec）。
- **compute_candidate_context_gain(cand)**：`context_gain = conditioned_sim - surface_proxy`（surface_proxy 优先 surface_sim，否则 semantic_score，否则 0）。
- **merge_landing_candidates_by_tid**：按 tid 合并时维护 `surface_sim`、`conditioned_sim`、`source_set`、`has_family_fallback`，合并后对每个候选统一算 `context_gain`。
- **collect_landing_candidates**：① 优先两路正式证据：similar_to + conditioned_vec（conditioned_top_k ≥ max(6, STAGE2A_COLLECT_CONDITIONED_TOP_K)）；② family_landing 仅当 `len(sim_cands) + len(ctx_cands) <= 2` 时补池，且标 `family_fallback_only`、`default_expand_block_reason="family_fallback_no_expand"`，不参与主脑。
- **_retrieve_academic_terms_by_conditioned_vec**：升格为正式分支，`use_k = max(use_k, 6)`；返回的 LandingCandidate 带 `source="conditioned_vec"`、`conditioned_sim`、`surface_sim=None`，source_role 设为 seed_candidate。
- **build_mainline_preference**：纳入 `context_gain_score = clip01((context_gain + 0.10) / 0.20)`、`dual_support_bonus = 0.08`（当 dual_support 时）；权重改为 0.30×anchor_identity + 0.20×jd_align + 0.20×context_continuity + 0.15×context_gain_score + 0.10×hierarchy_consistency + dual_support_bonus − risk_penalties。
- **enrich_stage2a_candidates**：为候选补 `context_gain`、`has_dynamic_support`、`has_static_support`、`dual_support`。
- **_stage2a_rule_flags**：新增 `context_gain_ok`（context_gain ≥ 0.03）、`dynamic_support_ok`、`dual_support_ok`。
- **select_primary_per_anchor**（最小收尾）：① **前置仅** `retain_mode=="reject"`；**obviously_bad_branch、drift_bad、polysemous_no_context** 仍硬拒绝；**低 primary_score 不前置 reject**，交给 **`judge_primary_and_expandability`**；② 空锚 fallback 改为**结构型保底**（`family/jd/context/hierarchy` + `generic/object/poly` 风险门），不依赖词面名单；③ fallback 一律 `primary_keep_no_expand` 且 `can_expand_from_2a=False`；④ 新增 **conditioned_vec 来源配额**：每锚最多 `1 expandable + 1 keep`（按 `dual_support/mainline_support/family` 与风险排序）。
- **judge_primary_and_expandability**：`primary_ok`（宽）与 `can_expand`（严）解耦保持不变；但 `good_mainline` 不再自动给 `primary_expandable`，需额外通过风险/identity 过滤（`family_match`、`generic_risk`、`semantic_drift_risk`、`bonus_term_penalty`）；并对 `conditioned_vec` 来源追加更严 expandable 门（需 dual+multi_anchor+更高 family/ctx 且低风险），否则降级 `primary_keep_no_expand`。
- **check_seed_eligibility**：**仅 `can_expand_from_2a=True`** 可 seed；`fallback_primary` 挡扩；保留 semantic_mismatch 硬挡；seed gate 改为 **按 source_type 分流**：`similar_to` 主线 seed 走“放宽门”（dual_support+ctx+低风险，避免 route planning 误杀）；`conditioned_vec` 走“收紧门”（identity/ctx/seed_score 与风险更严格）；其余来源维持硬门。未通过统一标记 `weak_expandable_seed`。
- **调试打印**：`[Stage2A primary/expand split]`、`[Stage2A select]`、`[Stage2A fallback]`、`[Stage2B seed gate]`、`[Stage2B final seed gate]`。
- **预期现象**：动力学/运动学/仿真/路径规划等锚点下，conditioned_sim > static_sim、context_gain > 0 的候选有机会 primary_expandable；kinesiology 等 static 尚可但 context_gain 不佳则降级；family_landing 仅在两路都很弱时补池且不扩散。

**Stage2A 最小修复补丁（防全灭）**

在「上下文纠偏 + 双空间准入」基础上，为避免 Stage2A 因 identity/eligibility/admission 过严导致 **raw_candidates=0 全灭**，对以下 3 个函数做了最小放宽，目标是把系统从「全灭」拉回「能出正常 primary」：

| 函数 | 改动要点 |
|------|----------|
| **compute_anchor_identity_score** | identity 从「硬词面门」改为**软本义分**：别名命中给 base_floor≥0.34，避免 motion control / reinforcement learning / robotic arm 等正常中英映射被压成低 identity；保留对 propulsion、simula、control flow、data retrieval 等明显错义词的惩罚；generic/ambiguity 惩罚放宽（0.72/0.78），错义黑名单仍 0.45。 |
| **check_primary_eligibility** | 从「先斩后奏」改为**只拦特别危险的**：aid<0.20 且 ctx_sim<0.80 才拦 identity_too_low；similar_to 仅当 ctx_sim>0 且 ctx_sim<0.70（上下文明显反对）才拦；context_fallback 的 identity 门槛从 0.50 降到 0.32（且与 ctx_sim<0.88 组合）。 |
| **check_primary_admission** | 主裁判从 identity 改为 **context_consistency + jd_align + semantic_score**：主通道不再要求 anchor_identity≥0.35，改为 context_consistency≥0.76、jd_align≥0.78、semantic≥0.80，且（anchor_identity≥0.22 或 ctx_sim≥0.82）；identity 低但上下文强时可 weak_retain；新增 **hierarchy 边界**通道：topic_overlap≥PRIMARY_MIN_HIERARCHY_MATCH、subfield≥0.25、context_consistency≥0.70 ⇒ weak_retain，不再直接全砍。 |

**预期改善**：motion control、reinforcement learning/q-learning、robotic arm、route planning、medical robotics 等不再因 identity_too_low 或 dual_space_not_stable 被直接全灭；下一轮日志应能看到正常 primary 数量恢复，再根据剩余偏差做下一刀微调。

**Stage2A 漏点修复（动力学→mechanics / 仿真→simulation / 路径规划→route planning）**

针对「正确上位词没被放进来」的三类漏点，只改领域过滤、admission 边界放行与 identity 对通用主词的容忍，不放大改排序：

| 漏点 | 问题 | 修改位置与要点 |
|------|------|----------------|
| **动力学→mechanics** | mechanics 被 domain_no_match 过滤掉，锚点为空 | **retrieve_academic_term_by_similar_to**：对 SIMILAR_TO 候选，不再把所有 domain_no_match 当硬拒。区分：① **强冲突**（主领域在 STRONG_CONFLICT_DOMAIN_IDS 且与激活领域无交）→ 硬拒；② 其余 domain_no_match → **soft retain**（设 soft_domain_retain=True、domain_fit×0.85），保留进 flat_pool 进后续 admission。**collect_landing_candidates**：对 soft_domain_retain 候选在算完 domain_fit 后再乘 0.85。**\_term_in_active_domains_with_reason**：当 domain_ok=False 且主领域在 STRONG_CONFLICT_DOMAIN_IDS 时返回 reason="domain_conflict_strong"（空集则全为 domain_no_match，仅做 soft retain）。 |
| **仿真→simulation** | simulation 被 dual_space_not_stable 拒，admission 对通用主词过严 | **check_primary_admission**：在最终 reject 前增加 **generic_main_term weak pass**：若 semantic_score≥0.80、jd_align≥0.78、context_consistency≥0.68、anchor_identity≥0.16，则 weak_retain + suppress_seed=True、retain_reason="generic_main_term_supported"，不因 hierarchy 弱而全砍。 |
| **路径规划→route planning** | route planning / path planning 等被拒，identity 与 admission 无保留通道 | **ANCHOR_IDENTITY_ALIASES**：增加「路径规划」→ {"route planning", "path planning", "motion planning", "trajectory planning"}，提升 planning 族中英映射；**check_primary_admission** 同上 generic_main_term weak pass 即可。 |

**Stage2A 终稿版（候选分型 + 组内选主，不硬编码岗位词）**

目标：把「候选收集」改成「候选分型 + 组内选主」，让主线词能进，偏词只能降级不能扩散。

| 函数/能力 | 删什么 | 改什么 | 新增字段/返回值 |
|-----------|--------|--------|-----------------|
| **collect_landing_candidates** | 平权收集、只存薄候选 | 收集 + 初筛标签化：对每个候选补 family_type、generic_like、scene_shifted、expand_block_reason、source_trust、ctx_supported、ctx_gap | `family_type`(exact_like/near_synonym/generic/shifted)、`scene_shifted`、`generic_like`、`expand_block_reason`、`source_trust`、`ctx_supported`、`ctx_gap`、`anchor_term`/`anchor_vid` |
| **classify_candidate_family**（新增） | — | 轻量语义分型：词形 + 语义 + 上下文 → exact_like / near_synonym / generic / shifted | 返回枚举字符串 |
| **is_candidate_generic_like**（新增） | — | 泛词风险显式布尔 | 返回 bool |
| **is_candidate_scene_shifted**（新增） | — | 识别语义相近但场景跑偏（如 kinesiology / medical robotics / propulsion） | 返回 bool |
| **infer_expand_block_reason**（新增） | — | 禁止扩散原因：generic / scene_shift / low_identity / weak_context | 返回 str 或 None |
| **score_academic_identity** | 仅由 edge/semantic 一项决定 | 四段式：edge_semantic(0.45) + lexical_shape(0.25) + context_support(0.20) + family_centeredness(0.10) | `identity_breakdown`、`anchor_identity_score` |
| **score_stage2a_candidate**（新增） | — | 主线资格分 mainline_pref_score 与保留资格分 retain_score 分开；不在此决定 bucket | `mainline_pref_score`、`retain_score`、`mainline_admissible`、`is_good_mainline`（占位，组选主时填） |
| **is_mainline_admissible**（新增） | — | 主线准入门：非 scene_shifted、非 generic_like、identity≥MAINLINE_IDENTITY_MIN、family_type∈{exact_like,near_synonym} | 返回 bool |
| **select_primary_per_anchor** | mainline_admissible/suppress_seed 一票否决主线词 | **三档分桶（预检与分桶解耦后）**：**前置硬拒绝仅** `retain_mode=="reject"`（**不再**因 `primary_score < STAGE2A_WEAK_KEEP_MIN` 在此打掉）；低分词进入 **`judge_primary_and_expandability`** 再判 `primary_keep_no_expand` / reject。其后仍直接 reject：**obviously_bad_branch**、**semantic_drift_branch**、**polysemous_no_context**（poly_bad∧¬ctx_ok）。good_mainline（retain_mode≠reject 且 mainline_pref≥MAINLINE_PREF_MIN 且 anchor_identity≥EXPANDABLE_IDENTITY_MIN 且非明显坏分支且 source≠conditioned_vec）→ primary_expandable；否则 → primary_keep_no_expand。**额外收尾**：若本组在优先分桶后仍空，且锚点本身质量高，则救回 1 条 `primary_keep_no_expand`（标记 `fallback_primary`，禁止扩散与 paper 抢配额）。precheck 不再改写 retain_mode，仅打 stage2a_reject_cls/precheck_hint。 | `bucket`、`can_expand`、`bucket_reason`、`is_good_mainline`、`role_in_anchor`、`admission_reason` |
| **merge_stage2a_primary** | support_count 高就更可信、多锚洗白 generic/shifted | 跨锚支持仅弱加分；family_type 为 generic/shifted 时 effective_bonus = raw_bonus×0.3；can_expand 直接沿用候选已算好的值 | `cross_anchor_support_raw`、`cross_anchor_support_effective`、`cross_anchor_bonus`、`family_type` |

常量：`CTX_SUPPORT_MIN`、`CTX_GAP_SHIFT_MIN`、`HIER_WEAK_MIN`、`JD_ALIGN_WEAK_MIN`、`GENERIC_RISK_MIN`、`POLY_RISK_MIN`、`MAINLINE_IDENTITY_MIN`、`RETAIN_MIN`。**Stage2A 终稿分桶**：`STAGE2A_MAINLINE_PREF_MIN=0.50`、`STAGE2A_EXPANDABLE_IDENTITY_MIN=0.52`、`STAGE2A_WEAK_KEEP_MIN=0.20`（与 `STAGE2A_PRIMARY_KEEP_MIN` 同源，**文档/judge 弱保留对齐**；**`select_primary_per_anchor` 前置硬拒绝不再使用该阈值**）；**Stage2B seed**：`SEED_SCORE_MIN=0.50`（仅消费 can_expand + primary_score≥此，不做第二轮主线审批）。  
**lexical_shape_match(anchor_term, candidate_term)**：词形匹配 0~1，供分型用，无硬编码词表。  
预期效果：dynamism、kinesiology、simulation、digital control、Machine control、Mechatronics、Robotic paradigms、medical robotics 等**不能再 expandable**（只能 retain 或 reject）。

**Stage2A 预检与分桶解耦（二次最小修正）**

**问题定位**：日志显示大量本应竞争主线的词（motion control、reinforcement learning、robotic arm、robot control、digital control、Robot manipulator、medical robotics）被统一打成 `mainline_block_reason='retain_mode_not_normal'`，原因是 **precheck 阶段**（`_should_hard_reject_stage2a_candidate`）在命中 weak_but_technical / weak_tech_keep_candidate 时**直接改写**了 `retain_mode='weak_retain'`、`suppress_seed=True`、`role_in_anchor_candidate_pool='secondary_candidate'`，导致进入 `select_primary_per_anchor` 时 `retain_mode != "normal"` 一票否决，无法再进 primary_expandable。即：问题不是「分数公式」或「阈值」，而是「主线资格在进入分桶前就被预检批量改成了 weak_retain」。

**修正思路**：① **预检不再写 retain_mode**：precheck 只做 hard_reject 与打标签；弱技术保留只设 `stage2a_reject_cls='weak_tech_keep'`、`precheck_hint='weak_tech_keep'`，**不**改 `retain_mode`/`suppress_seed`/`role_in_anchor_candidate_pool`。② **retain_mode 只允许在 check_primary_admission() 里定一次**：enrich 阶段 admission 赋值后，后续流程只读、不覆盖。③ **最终分桶由「门控」决定主线**：`select_primary_per_anchor` 中 good_mainline 仍为：`retain_mode != "reject"` 且 `mainline_pref >= STAGE2A_MAINLINE_PREF_MIN` 且 `anchor_identity >= STAGE2A_EXPANDABLE_IDENTITY_MIN` 且**非**明显坏分支（`_is_obviously_bad_branch_stage2a`）且 `source_type != "conditioned_vec"`。**前置 obvious_bad 收紧为仅 `retain_mode == "reject"`**；`primary_score < STAGE2A_WEAK_KEEP_MIN` **改由 `judge_primary_and_expandability` 决定** keep_no_expand / reject，避免弱主线技术词（simulation、mechanics、feedback control、vibration 等）未到 judge 就被打掉。weak_retain / 带 precheck_hint 的候选仍可由最终分桶升到 primary_expandable；conditioned_vec 仍仅 primary_keep_no_expand。

**代码改动**：

| 位置 | 改动 |
|------|------|
| **\_should_hard_reject_stage2a_candidate** | 当 `low_mainline_no_ctx` 且 weak_tech_keep_candidate，或当 `weak_but_technical or weak_tech_keep_candidate` 时：只 `setattr(cand, "stage2a_reject_cls", "weak_tech_keep")`、`setattr(cand, "precheck_hint", "weak_tech_keep")`，**删除**对 `retain_mode`、`suppress_seed`、`role_in_anchor_candidate_pool` 的写入；return False（不 hard reject）。 |
| **select_primary_per_anchor** | **前置**：仅 `retain_mode == "reject"` → reject（`reject_reason`/`bucket_reason`/`mainline_block_reason`=`retain_mode_reject`，`survive_primary=False`）。**不再**在此处因 `primary_score < STAGE2A_WEAK_KEEP_MIN` reject。其后 **obviously_bad_branch / drift_bad**、**polysemous_no_context** 仍直接 reject。再进入 **`judge_primary_and_expandability`** 得 expandable / keep_no_expand / reject。`good_mainline`（可扩侧）条件同上表。**`primary_score_below_weak_keep`** 不再作为本函数前置 reject 的 block_reason（若 judge 仍 reject 低分，由 judge 的 reason 表达）。 |

**`select_primary_per_anchor` 前置硬拒绝收窄（弱主线进 judge）**（`label_expansion.py`）：**不动** `run_stage2`、`stage2_generate_academic_terms` 与 **`pick_fallback_primary_for_anchor`** 接线；只改组内选主循环第 0 段。**原则**：`retain_mode=="reject"` 仍前置打掉；**`primary_score < STAGE2A_WEAK_KEEP_MIN` 不再前置打掉**，交给 **`judge_primary_and_expandability`** 判 **`primary_keep_no_expand`** 或 reject。**验收日志**：① `[Stage2A select]` 某锚 `keep_no_expand` 含 mechanics / simulation 等弱主线；② 每锚 primary 汇总里原 0-primary 锚点是否出现 1 条 keep；③ **`pick_fallback_primary_for_anchor`** 触发次数应减少（fallback 仅兜底，非常态）。

**预期**：motion control、reinforcement learning、robotic arm、robot control、digital control、Robot manipulator、medical robotics 等由**最终分桶**按门控决定是否 primary_expandable，不再被 precheck 提前压死；dynamism、propulsion、mechanics、vibration、kinesiology、movement control、motion controller、robotic hand 等仍可落在 primary_keep_no_expand（由门控或 admission 的 weak_retain 决定）。

**Stage2A select_primary_per_anchor 可扩散规则修正（让 2A 真正产出 primary_expandable）**

- **目标**：不再因 `context_gain <= 0` 把明显正确的主线词全打成 `primary_keep_no_expand`，让 2A 能产出少量 `primary_expandable`。
- **问题**：候选已找到、`mainline_candidate=True` 也成立，但原逻辑要求 `context_gain >= 0.03` 且 `has_dynamic_support` 才 `can_expand`，导致可扩词被全灭。
- **改动要点**（只改 `select_primary_per_anchor`）：
  1. **保留硬拒绝**：`obviously_bad_branch`、`semantic_drift_branch`、`object_or_poly_bad_branch`、`polysemous_no_context` 仍直接 reject，不动。
  2. **主线候选放宽**：`mainline_candidate` = `identity_ok` 或（`anchor_identity >= ID_WEAK` 且 `jd_align >= JD_OK`）或（`dual_support` 且 `ctx_ok`），不再强依赖 `context_gain_ok`。
  3. **可扩散新规则**（满足任一即可 `can_expand`，不再要求 `context_gain > 0`）：
     - **双路一致主线**：`dual_support`、`anchor_identity >= ID_MAIN`、`context_supported`、`context_sim >= CTX_FLOOR`；
     - **静态强匹配主线**：`source_has_similar_to`、`anchor_identity >= ID_STRONG`、`jd_align >= JD_OK`、`context_sim >= max(CTX_FLOOR, semantic_score - CTX_DROP_TOL)`；
     - **多锚点一致主线**：`support_count >= 2`、`anchor_identity >= ID_MULTI`、`jd_align >= JD_OK`，且非 poly_bad/object_bad。
  4. **分桶**：`can_expand` → primary_expandable；否则 mainline_candidate → primary_keep_no_expand；否则 primary_score ≥ PRIMARY_KEEP_MIN → primary_keep_no_expand，再否则 reject。
- **新增常量**：`STAGE2A_ID_MAIN`(0.52)、`STAGE2A_ID_STRONG`(0.58)、`STAGE2A_ID_MULTI`(0.48)、`STAGE2A_ID_WEAK`(0.45)、`STAGE2A_JD_OK`(0.74)、`STAGE2A_CTX_FLOOR`(0.42)、`STAGE2A_CTX_DROP_TOL`(0.08)、`STAGE2A_PRIMARY_KEEP_MIN`(0.20)。
- **预期现象**：motion control、reinforcement learning、robotic arm、robot control、route planning、digital control 等有机会从 keep_no_expand 变为 primary_expandable；Nonlinear system identification、simulation 谨慎放行；dynamism、propulsion、kinesiology、Leukocytopenia、Educational robotics、Telerobotics、End-to-end principle 等仍不放行。验收时看日志：primary_expandable 非空，约 4～8 个词可扩。

**三函数补丁：check_seed_eligibility + is_primary_expandable + Stage3 paper 选词（救活 2B + 压掉伪相关高排）**

- **目标**：① 2B 不再被 context_gain≤0 全灭，主线词能重新成为可扩散 seed；② 可扩散 primary 定义收稳，只放行强 identity/双证据/多锚一致；③ Stage3 与 paper 选词压掉 conditioned-only / no_mainline_support 的伪相关高排词。
- **1）check_seed_eligibility**  
  - 删除 **context_gain ≤ 0 硬拦截**，改为只作 seed_score 软加减分（ctx_bonus：≥0.03 加 0.05，0～0.03 加 0.02，-0.05～0 减 0.01，-0.10～-0.05 减 0.03，<-0.10 减 0.06）。  
  - 新增 0.5：泛方法词 seed 收紧：若 `generic_risk` / `polysemy_risk` 偏高或 `role_in_anchor=side`，但缺少 grounding_ok（`dual_support` 或 `cross_anchor_support_count>=2` 或 `has_family_evidence` 或 `anchor_identity>=0.60` 或 `jd_align>=0.76`），则不 eligible（`block_reason=method_like_without_grounding`）。  
  - 真正 seed 判定改为三条满足其一即 eligible：**strong_static_seed**（anchor_identity≥0.52、primary_score≥0.50、jd_align≥0.74）、**dual_support_seed**（dual_support 且 identity≥0.48、primary≥0.45、jd≥0.72）、**cross_anchor_seed**（support_count≥2 且 identity≥0.42、primary≥0.42、jd≥0.72）。  
  - seed_score 公式：`0.42*primary_score + 0.25*anchor_identity + 0.18*jd_align + 0.10*min(1, support_count/2) + 0.05*dual_support + ctx_bonus`。  
  - 不再依赖 `is_primary_expandable` 做 2B 准入（2B 只认 2A 的 can_expand + 上述三条）。
- **2）is_primary_expandable**  
  - 收稳「可扩散 primary」定义：只放行 **强 identity 主线**（identity≥0.55、primary≥0.50、jd≥0.74）、**双证据主线**（identity≥0.48、primary≥0.46、jd≥0.72 且非 conditioned_only）、**多锚一致主线**（support_count≥2、identity≥0.42、primary≥0.42、jd≥0.72）。  
  - 窄方法词、设备/对象词（且未在锚点白名单）一律不扩；conditioned_only 再补一道门（identity≥0.62、primary≥0.55、jd≥0.76、support_count≥2 才放行）。
- **3）Stage3 第二段 + paper 选词（现行）**
  - **`stage3_build_score_map`**：**统一连续分** 覆盖 **`final_score`**（模块常量 **`STAGE3_UNIFIED_W_*`**），**已移除** 按桶/子类的 **`score_mult`** 链。
  - **`select_terms_for_paper_recall`**：**weak support** + **`risky_side_block`** + **`JD 全局主轴门`（core：`rk≤MAX ∨ a_sc≥MIN`；双弱 → **`core_axis_near_miss_soft_admit` 进 `support_pool`**）** + **统一下池合并** + **`[Stage3 paper gate summary]`** → **`paper_select_score`** + family + floor …
  - **数量截断**：总上限 **PAPER_RECALL_MAX_TERMS**（默认 12）。
- **预期现象**：  
  - 2B 被救活：motion control、reinforcement learning、robotic arm、route planning 等重新变为可扩散 seed；dense/cluster/cooc 不再全 0。  
  - 仍保留但不扩散：q-learning、robotic hand、medical robotics、Control engineering、simulation。  
  - Stage3 压下去：Educational robotics、Telerobotics、End-to-end principle、Servo control、Machine control、Robot manipulator 等不再冲前排；motion control、robot control、robotic arm、route planning、reinforcement learning、simulation 更靠前。  
  - **验收时重点看的 4 处日志**：`[Stage2B final seed gate]` 期待 motion control / reinforcement learning / robotic arm / route planning → eligible=True；`[Stage2B Expansion Summary]` 不再 dense_kept=0 cluster_kept=0 cooc_kept=0；`[Stage3 Buckets]` 中 Educational robotics / Telerobotics 不再稳居 support 前排；最终 paper 词表更接近 motion control、robot control、robotic arm、route planning、reinforcement learning、simulation，而非 Educational robotics、Telerobotics、End-to-end principle、Servo control 等。

**四函数补丁（Stage2A/2B 收尾：弱保留收紧 + conditioned 辅助 + 2B 放行 + term_role 区分）**

在「可扩散规则修正」与「三函数补丁」基础上，对以下 4 个函数做了最小收尾，目标：① 坏组不被迫选主，弱词（kinesiology / GRASP / End-to-end principle / Telerobotics 等）不再仅因 primary_score≥0.20 混入 primary；② conditioned_vec 作为辅助证据源不霸榜；③ 2A 已认定 can_expand=True 的主线词在 2B 更顺畅放行；④ Stage3 能区分真主线与弱保留（term_role=primary vs primary_side）。

| 函数 | 修改要点 |
|------|----------|
| **select_primary_per_anchor** | ① **前置硬拒绝**：仅 `retain_mode=="reject"`；**obviously_bad_branch、drift_bad、polysemous_no_context** 不变；**已取消**「`primary_score < STAGE2A_WEAK_KEEP_MIN` 与 reject 绑在一起」的前置 obvious_bad。② mainline_candidate 与 can_expand 判定保持 **`judge_primary_and_expandability`** 主干；③ **弱保留收紧**（若仍存在于 judge/后续分支）：仅当 `primary_score >= STAGE2A_PRIMARY_KEEP_MIN` **且** `weak_but_technical_keep` … 时才进 primary_keep_no_expand 等（以代码为准）。④ 不再对候选预排序，按遍历顺序分桶。 |
| **_retrieve_academic_terms_by_conditioned_vec** | ① conditioned_vec 定位为**辅助证据源**（source_role=auxiliary_evidence）；② use_k 保守为 `max(use_k, 4)`，不再无脑升到 6；③ 独立门槛 `conditioned_min_sim = max(SIMILAR_TO_MIN_SCORE, 0.78)`，conditioned-only 词需 `sim >= 0.82` 才入池；④ 为每条候选设 `conditioned_only`、`has_similar_to_support`；⑤ context_supported 以 0.80 为界。 |
| **check_seed_eligibility** | ① 先尊重 2A（can_expand、retain_mode、suppress_seed、semantic_mismatch_seed）不变；② **seed 通路**：direct_seed_ok（can_expand + primary_score≥SEED_SCORE_MIN + anchor_identity≥0.50 + jd_align≥0.72）、dual_support_seed、strong_static_seed（source_has_similar_to + can_expand + identity≥0.52 + primary≥0.48 + jd≥0.72）、cross_anchor_seed（support_count≥2 + can_expand + identity≥0.45 + primary≥0.44 + jd≥0.70）四路满足其一即 eligible；③ context_gain 只做软修正（ctx_bonus）；④ seed_score = 0.45×primary + 0.25×anchor_identity + 0.20×jd_align + 0.10×static_sim + ctx_bonus，clip 到 [0,1]。 |
| **merge_primary_and_support_terms** | ① 对每条 primary：`can_expand and role_in_anchor=="mainline"` → `term_role="primary"`、`is_weak_primary=False`、`mainline_support_factor=1.0`；否则 → `term_role="primary_side"`、`is_weak_primary=True`、`mainline_support_factor=0.65`；② 透传 `primary_bucket`、`is_weak_primary`、`mainline_support_factor` 到 ExpandedTermCandidate，供 Stage3 直接降权或选词。dense/cluster/cooc 三段逻辑不变。 |

**预期现象**：① Stage2A primary 数下降，运动学/抓取/端到端/Robotics 等锚点更容易出现空组；② Stage3 Top20 更干净，kinesiology、GRASP、End-to-end principle、Telerobotics、Mechatronics、Robotic paradigms 等更容易下去；③ conditioned_vec 霸榜减轻，Control engineering / Servo control / Telerobotics 不易高位；④ Stage2B 更像「从好 primary 出发扩散」，robot control / q-learning / digital control 等不该全死。

**改完后重点看的 5 类日志**：`[Stage2A 选主]`、`[Stage2A merge evidence detail]`、`[Stage2B seed factors]`、`[Stage3 Buckets]`、`final_term_ids_for_paper`。理想情况：primary_keep_no_expand 总数下降、空组数量略升、conditioned_vec only 词减少、paper_terms 不再被 robotic arm + reinforcement learning 旁支过度挤占。

**Stage2 最后两刀：seed 契约唯一化 + 动力学候选补池**

**刀 1：2A→2B seed 判定唯一出口**  
目标：Stage2B 只认 `check_seed_eligibility()` 的最终结果，不再出现 `can_expand=True` 或 `trusted_source=True` 旁路放行。

- **check_seed_eligibility**：① 先判 `can_expand`（2A 明确不可扩散则返回 `stage2a_not_expandable`）；② 仅 `retain_mode=="normal"` 且非 `suppress_seed` 才可能进 seed；③ source 必须在 `TRUSTED_SOURCE_TYPES_FOR_DIFFUSION`；④ 语义错义 seed 禁扩；⑤ 最终准入由 `is_primary_expandable(...)` 决定；`seed_score` 只参与排序/裁剪，不翻案。⑥ 调试时打印 `[Stage2B final seed gate]`（anchor、term、can_expand、retain_mode、suppress_seed、identity、primary、jd、eligible、block_reason）。
- **stage2_generate_academic_terms**：seed 收集**只**通过 `check_seed_eligibility(label, p, jd_profile)`；`diffusion_primaries` 仅来自 `eligible and seed_score >= SEED_SCORE_MIN`，按 `seed_score` 排序；删除任何直接依据 `p.can_expand` / `p.primary_score >= SEED_SCORE_MIN` / `p.source in TRUSTED_*` 的放行逻辑。

**刀 2：动力学等核心锚点候选补池**  
目标：`动力学` 等锚点不再只拿到 dynamism/propulsion/mechanics，而要能进池 dynamics、robot dynamics、rigid body dynamics、inverse dynamics 等。

- **retrieve_family_landing_candidates(label, anchor, top_k)**：新增；面向 `canonical_academic_like` 或锚点文本在 `CANONICAL_ACADEMIC_ANCHOR_FAMILY_QUERIES` 的锚点，用轻量 family 查询词（动力学→dynamics/robot dynamics/…）在 vocabulary 中做词面/含词匹配，返回 `source="family_landing"` 的 `LandingCandidate` 列表，仅补池不保送 mainline。
- **merge_landing_candidates_by_tid(candidates)**：新增；同 tid 保留证据最强者，合并 `all_sources`、`has_family_evidence`，供后续统一打分。
- **collect_landing_candidates**：① 对 canonical_academic_like 锚点先调用 `retrieve_family_landing_candidates`；② 再 similar_to；③ 再 conditioned_vec，且当 `len(similar_to_candidates)==0` 时使用 `conditioned_top_k=STAGE2A_COLLECT_CONDITIONED_TOP_K_RESCUE` 多拿几条；④ 用 `merge_landing_candidates_by_tid(family_cands + similar_to_candidates + context_neighbors)` 合并后统一做 domain_fit / hierarchy / 定性 / score_academic_identity；⑤ 对 `has_family_evidence` 的候选做轻量 boost（anchor_identity_score += 0.08，primary_eligibility_reasons 追加 `family_landing_support`），不保送 primary_expandable。
- **select_primary_per_anchor**：排序 key 中加入 `has_family_evidence`（在 mainline_pref 之后、anchor_identity 之前），优先 family candidate 但不断主线门控。
- **_retrieve_academic_terms_by_conditioned_vec**：增加可选参数 `conditioned_top_k`，用于空池救援时加大检索条数。
- **PrimaryLanding / ExpandedTermCandidate / _expanded_to_raw_candidates**：透传 `has_family_evidence`、`seed_block_reason` 到 Stage3 的 `_debug`，便于日志排查。

**常量**：`STAGE2A_COLLECT_CONDITIONED_TOP_K_RESCUE`、`STAGE2A_FAMILY_LANDING_TOP_K`、`CANONICAL_ACADEMIC_ANCHOR_FAMILY_QUERIES`（动力学/运动学/仿真/振动抑制/路径规划/抓取/端到端等→family 查询词列表）。

**预期**：刀 1 后日志中不再出现「identity&lt;SEED_MIN 但 seed_ok=True」的旁路；刀 2 后「动力学」等锚点候选池中应出现 dynamics、robot dynamics、rigid body dynamics、inverse dynamics 等，由组内选主与门控决定是否 surviving。

---

**Stage2A 函数级修改（第一批：定性 + 三档 + 选主结构）**

本批仅改 Stage2A，目标：**先定性、再选主**，明确「候选 / 主词 / 仅保留 / 拒绝」与「可扩 / 不可扩」；不引入硬编码词表、不依赖训练集。

| 函数 | 修改要点 |
|------|----------|
| **collect_landing_candidates** | ① 召回仍为 similar_to + conditioned_vec，主池≤1 时补 context_fallback；② 为每个候选**补定性字段**：`context_continuity`（= compute_context_consistency）、`hierarchy_consistency`、`polysemy_risk`、`object_like_risk`、`generic_risk`（复用现有 compute_*_risk）；`anchor_identity` 沿用 anchor_identity_score；不做最终选主。 |
| **check_primary_admission** | 改为**三档判定**，入参仅 `candidate`（含上述定性字段），返回 `{admit, retain_mode, suppress_seed, reasons}`。**normal**：semantic≥0.80、jd_align≥0.78、context_continuity≥0.74、anchor_identity≥0.22、object_like_risk≤0.55、polysemy_risk≤0.60 → 可进 primary 且可当 seed；**weak_retain**：语义边界达标但 identity 或风险略逊 → 保留进 2A/3，**suppress_seed=True**；**reject**：其余。 |
| **build_mainline_preference** | 只回答「像不像该锚点学术主干」：入参单候选，只用 anchor_identity、context_continuity、jd_align、hierarchy_consistency、polysemy_risk、object_like_risk、generic_risk；返回**标量** mainline_preference = positive − negative（权重 0.32/0.26/0.22/0.12 与 0.16/0.12/0.10），仅用于 2A 内部排序。 |
| **enrich_stage2a_candidates** | 对每个候选调用 **check_primary_admission**，写入 `retain_mode`、`suppress_seed`、`admission_reasons`；算 cross_anchor_support、**mainline_preference**（标量）；设 **role_in_anchor_candidate_pool**：reject→reject_candidate，normal→mainline_candidate，weak_retain→secondary_candidate。 |
| **select_primary_per_anchor** | 返回 **{ primary_expandable, primary_keep_no_expand, rejected }**。kept = 非 reject 候选，按 (retain_mode==normal, mainline_preference, cross_anchor_support, jd_align) 降序；**primary_expandable**：仅从 normal 且非 suppress_seed 中取 top1；**primary_keep_no_expand**：runner-up 一个（保留不扩）；其余入 rejected。 |
| **merge_stage2a_primary** | 入参为每锚的 `primary_expandable` / `primary_keep_no_expand` 列表；合并后每条含 **tid, term, anchors, support_count, mainline_hits, best_rank, can_expand, support_roles, retain_modes**（support_anchors 更名为 anchors 与 support_roles/retain_modes 一并保留）。 |

**Stage2A 逐函数 checklist 落地（最新实现）**

以下为当前代码已落地的 Stage2A 行为与字段，与「宁可多收、不要早杀」「三档准入」「辅助约束」一致。

| 函数 | 删什么 | 改什么 | 新增字段 |
|------|--------|--------|----------|
| **collect_landing_candidates** | 提前一刀切 reject、primary_cap=secondary_only、在此决定 normal/weak_retain/reject 或 can_expand | 仅负责「候选召回 + 候选补字段」；从 similar_to / conditioned_vec / fallback 收候选、去重、补字段后返回完整候选池 | anchor_identity_score, context_continuity, context_local_support, context_co_anchor_support, context_jd_support, jd_candidate_alignment, hierarchy_consistency, field_fit, subfield_fit, topic_fit, polysemy_risk, polysemy_note, object_like_risk, object_like_note, generic_risk, generic_note, source_type, source_rank, source_score |
| **compute_anchor_identity_score** | identity 直接等于 semantic_score、单通道实现 | 锚点连续性分：综合 similar_to 边强度、conditioned 对齐、主干语义、漂向对象/跨域（可选参数 edge_strength, context_sim） | anchor_identity_score；可选 anchor_identity_reason |
| **compute_context_continuity** | 布尔 context_supported 主导、只看 JD 全文 | 连续分；同时看 anchor local、co_anchor、jd_profile/conditioned | context_continuity, context_local_support, context_co_anchor_support, context_jd_support |
| **compute_hierarchy_consistency** | 层级不中即硬杀、hierarchy 当主裁判 | 辅助约束：好→加分，一般→不杀，明显反常→才降级 | hierarchy_consistency, field_fit, subfield_fit, topic_fit, hierarchy_note |
| **compute_polysemy_risk** | 极端 0/1、按来源一刀切 | 连续风险分；压 control flow/dyskinesia/simula，不压 motion control/reinforcement learning/robotic arm | polysemy_risk, polysemy_note |
| **compute_object_like_risk** | 完全不用、仅词尾硬编码 | 对象/器件倾向惩罚项；压 motion controller/robotic hand，不压死 robotic arm | object_like_risk, object_like_note |
| **compute_generic_risk** | 固定常数、与 polysemy 合并 | 单独衡量「词是否过泛」；压 control/automatic control/machine control/general robotics | generic_risk, generic_note |
| **check_primary_admission** | 过硬 normal 条件、过大 reject 区间、「非 perfect 即 reject」 | 明确三档：normal（可信主词可扩）、weak_retain（保留不扩）、reject（明显错位才拒）；门槛放宽 | retain_mode, suppress_seed, primary_eligibility_reasons |
| **build_mainline_preference** | 过重负惩罚、final_score 当 mainline、全局终局项 | 仅局部「像不像锚点主干」；正向 identity/context/jd_align/hierarchy，负向 poly/object_like/generic（惩罚项不过重） | mainline_preference；可选 mainline_preference_breakdown |
| **build_cross_anchor_index** | support 当主裁判、support_count 低即不能做主词 | 仅提供辅助证据：多锚支撑更稳，不一票否决 | 回填：cross_anchor_support_count, cross_anchor_support_weight, cross_anchor_anchor_list |
| **enrich_stage2a_candidates** | enrich 完即定稿、enrich 内直接选主 | 只负责：调 check_primary_admission、build_mainline_preference、回填 cross_anchor、标角色 | retain_mode, suppress_seed, primary_eligibility_reasons, mainline_preference, cross_anchor_* , role_in_anchor_candidate_pool（mainline_candidate/secondary_candidate/reject_candidate） |
| **select_primary_per_anchor** | mainline_admissible/suppress_seed 一票否决、整锚归零 | **前置**仅 `retain_mode=="reject"`→reject；**obviously_bad_branch/drift/poly** 仍 reject；低分进 **judge** 再分桶；good_mainline→primary_expandable；否则 primary_keep_no_expand | role_in_anchor, can_expand, anchor_internal_rank, primary_bucket, bucket_reason, is_good_mainline |
| **merge_stage2a_primary** | merge 后丢结构、只留最终分 | 合并时保留跨锚结构 | support_count, mainline_hits, anchors, support_roles, retain_modes, can_expand, best_mainline_preference, best_anchor_identity, best_jd_align, best_context_continuity |

**Stage2A 字段总表（查漏补缺用）**

- **Candidate 级**：semantic_score, jd_candidate_alignment, context_continuity, anchor_identity_score, hierarchy_consistency, polysemy_risk, object_like_risk, generic_risk；retain_mode, suppress_seed, primary_eligibility_reasons, mainline_preference；cross_anchor_support_count, cross_anchor_support_weight；role_in_anchor_candidate_pool。
- **Selected primary 级**：role_in_anchor, can_expand, anchor_internal_rank, primary_bucket。
- **Merged primary 级**：support_count, mainline_hits, anchors, support_roles, retain_modes, can_expand, best_mainline_preference, best_anchor_identity, best_jd_align, best_context_continuity。

**主流程**：`collect_landing_candidates` → `landing_candidates_to_stage2a` → `build_cross_anchor_index` → `enrich_stage2a_candidates` → **pre-primary 分桶**（明显坏分支→hard_reject；弱相关技术词→keep_no_expand；low_identity/low_mainline_no_ctx→hard_reject；高歧义无上下文→hard_reject）→ `select_primary_per_anchor` → `merge_stage2a_primary`；Stage2B 仅 **can_expand=True 且 primary_score≥SEED_SCORE_MIN** 扩散。

**Stage2A pre-primary 分桶（只改桶边界、不改 score 公式）**：在 `enrich_stage2a_candidates` 之后、`select_primary_per_anchor` 之前按**判定顺序**分桶：① **明显坏分支** → hard_reject（对象/多义坏、抽象漂移且非弱技术词、含 "(management)"）；② **low_mainline_no_context**：若 `_is_weak_technical_keep_candidate` 则 primary_keep_no_expand，否则 hard_reject；③ **弱相关技术词**（`_is_weak_but_technical_keep_stage2a` 或 `_is_weak_technical_keep_candidate`）→ primary_keep_no_expand；④ low_identity_no_context / 高歧义无上下文 → hard_reject。**二次最小修正**：① **弱技术词保活** `_is_weak_technical_keep_candidate`（source similar_to/conditioned_vec、anchor_id≥0.20、jd_align≥0.74、ctx_cont≥0.44、poly&lt;0.55、object&lt;0.35）在 hard_reject 前生效，救回 simulation、vibration、feedback control；② **semantic_drift** 收紧且弱技术词豁免（anchor_id≥0.24 不判漂移；vibration/simulation 不误杀）；③ **conditioned_vec 默认不扩散**：`select_primary_per_anchor` 中 source_type==conditioned_vec 的候选一律 primary_keep_no_expand（bucket_reason=conditioned_vec_no_expand），压住 Servo control。**预期**：simulation、vibration、feedback control、mechanics → keep_no_expand；dynamism、propulsion、surgical robot、control (management)、control flow、simula → hard_reject；Servo control → keep_no_expand。日志含 `stage2a_reject_cls`。

**Stage2A 验收期望**（看日志对照）：动力学→mechanics 为 keep_no_expand、dynamism/propulsion 为 hard_reject；控制→feedback control 为 keep_no_expand、control (management)/control flow 为 hard_reject；仿真→**simulation** 为 keep_no_expand、simula 为 hard_reject；抖动→**vibration** 为 keep_no_expand、dyskinesia 为 hard_reject；医疗机器人→medical robotics 可为 keep_no_expand、surgical robot/robotic surgery 为 hard_reject；Robot control 锚下 **Servo control** 为 primary_keep_no_expand（conditioned_vec 不扩散）。**二次修正**：simulation/vibration 不再被错杀；Servo control 不再 primary_expandable。

**调试日志**（`LABEL_EXPANSION_DEBUG` + `STAGE2_VERBOSE_DEBUG` 时）：  
① **Stage2A 候选定性表**：anchor \| term \| anchor_id \| jd_align \| ctx_cont \| hier \| poly \| object \| hard_reject_reason \| **stage2a_reject_cls**（验收用）；  
② **Stage2A 选主结果表**：anchor \| primary_expandable \| primary_keep_no_expand \| rejected_top_reasons \| **hard_reject(term=reason)**；  
③ **Stage2A Merged Primary**：term \| support_count \| mainline_hits \| best_rank \| can_expand \| anchors \| support_roles \| retain_modes \| best_mainline_preference \| best_anchor_identity \| best_jd_align \| best_context_continuity。

**Stage2A primary/expand 调试一致性**（`LABEL_EXPANSION_DEBUG=True` 时，`label_means/label_expansion.py`）：

- **`judge_primary_and_expandability` 返回的 `snap["can_expand_from_2a"]`** 与最终 **bucket** 一致：凡 **`reject`** / **`primary_keep_no_expand`** 路径均在返回前显式置 **`can_expand_from_2a=False`**；仅 **`primary_expandable`** 为 **`True`**。
- **`[Stage2A expand factors]`** 中的 **`final_bucket` / `final_expandable`** 按 **最终落桶** 打印（`final_expandable ≡ bucket == "primary_expandable"`），不再把中间态 `can_expand` 误标为「最终可扩」。**`snap_can_expand_from_2a`** 为快照字段，应与上式一致。
- **定点诊断**（控制类对照）：模块常量 **`DEBUG_STAGE2A_FOCUS_TERMS`**（`motion control`、`robot control`、`digital control`）与 **`DEBUG_STAGE2A_FOCUS_ANCHORS`**（`运动控制`、`机器人运动控制`、`传统控制`、`Robot control`）命中时，额外打印 **`[Stage2A Focus Debug]`**（`ctx_drop`、`drop_ok_exp`、`branch_blocked`、object/generic/poly 风险、`dual_expand_ok` / `multi_expand_ok` / `solo_ctx_expand`、`expand_block` 等），用于区分 **keep 而不进 expandable** 究竟是 **`ctx_drop` 超过 `STAGE2A_EXPAND_MAX_CTX_DROP`（`drop_ok_exp=False`）** 还是 **`branch_blocked=True`（object/generic/poly 超阈）**；选主写回对象后打印 **`[Stage2A Commit Debug]`**（`primary_bucket`、`can_expand_from_2a`、`admission_reason`）。

**Stage2/Stage3 收缩设计与原则（无冗余参数、无硬编码）**

- **Stage2A 主线优先组内选主（分桶顺序 + 二次修正）**：主流程为 `build_stage2a_mainline_profile → … → enrich_stage2a_candidates` → **pre-primary 分桶**（① 明显坏分支 → hard_reject；② low_mainline_no_ctx 且弱技术词豁免 → keep_no_expand；③ 弱相关技术词 → keep_no_expand；④ low_identity/高歧义无上下文 → hard_reject）→ `select_primary_per_anchor`（**conditioned_vec 一律 primary_keep_no_expand**）→ `merge_stage2a_primary`。**只改桶、不改 score**。**救回**：simulation、vibration、feedback control、mechanics。**压住**：Servo control（conditioned_vec 不扩散）。**继续 hard_reject**：dynamism、propulsion、surgical robot、control (management)、control flow、sports science、simula。
- **Stage2B 固定成两层门**：**Seed 门**：仅 can_expand=True 且 primary_score≥SEED_SCORE_MIN(0.50) 的 primary 进 diffusion_primaries；**Support 门**：dense/cooc 须过 **support_expandable_for_anchor**。**Dense parent cap**：dense 候选 keep_score ≤ parent primary_score × **DENSE_PARENT_CAP(0.85)**，避免扩展词压过 parent（如 Motion controller 不超过 motion control）。**Cluster 当前全关**（`expand_from_cluster_members` 直接 return []）。
- **准入与排序**：Stage2A 准入仅 **PRIMARY_MIN_HIERARCHY_MATCH**、**PRIMARY_MIN_PATH_MATCH** + **CONDVEC_SOURCE_FACTOR**；primary 排序只一套 **PRIMARY_SCORE_W_***。不再新增参数。
- **Stage3 全局复审层**：classify_stage3_entry_groups → check_stage3_admission；**第一段** = score_term_record × identity_factor × risk_penalty × **role_factor**；**`stage3_build_score_map`** 用 **统一连续分** 覆盖 **`final_score`**，再 **`_apply_family_role_constraints`**；**select_terms_for_paper_recall**：**weak support** + **`risky_side_block`** + **JD 主轴门（core：`rk∨a_sc`）** + **`[Stage3 paper gate summary]`** → **`paper_select_score`** + **family** + **dynamic_floor**。**[Stage3 unified score breakdown]** 默认降噪（Top `STAGE3_UNIFIED_SCORE_DEBUG_TOP_K`）。其余日志含 **paper main-axis / centrality / cutoff / near-miss** 等。
- **泛化只软惩罚**：seed 不做 domain_span 硬杀；support 仅在 **DOMAIN_SPAN_EXTREME** 以上硬拒绝，其余由 **topic_span_penalty** 表达。
- **禁止**：不要多余参数；不要硬编码（避免 `min(常量, 0.65)` 等双重定义）。

**Stage2 逐函数修改（主干词保留、错误 seed 禁扩）**

目标：**不再把正确主干词直接 reject**，**不再让错误 surviving primary 去扩散**。以下 8 个关键函数已按此原则修改：

| 函数 | 修改要点 |
|------|----------|
| **Stage2A** | |
| `_compute_path_match` | 几何平均改为**加权平均**（0.2×field + 0.3×subfield + 0.5×topic），某层 active 为空则跳过并重新归一化，避免 topic 不完整时 path 塌成 0.03。 |
| `compute_hierarchy_evidence` | 新增 **topic_source**（direct / direct+cooc / cooc / missing）、**hierarchy_reliability**（1.0 / 0.8 / 0.5 / 0.4）、**effective_topic_overlap / effective_subfield_overlap / effective_path_match**（overlap × reliability）；cooc 补出的三级领域不再与 direct 同权。 |
| `check_primary_eligibility` | **资格赛**（只拦特别危险的）：① identity<0.20 **且** ctx_sim<0.80 才拦（identity_too_low）；② similar_to 仅当**上下文明显反对**（ctx_sim>0 且 ctx_sim<0.70）拦；③ context_fallback 仍保守：ctx_sim<0.82 或（aid<0.32 且 ctx_sim<0.88）拦；④ domain_conflict_strong 照常拦。边界情况留给 admission 做 weak_retain。 |
| `check_primary_admission` | **三档准入、宁可多留**（当前实现）：① **normal**：semantic≥0.68、jd_align≥0.65、context_continuity≥0.55、anchor_identity≥0.18、object_like_risk≤0.70、polysemy_risk≤0.72 → 可进 primary 且可当 seed；② **weak_retain**：semantic≥0.55、jd_align≥0.52、且（context_continuity≥0.45 或 anchor_identity≥0.12）→ 保留进 2A/3，suppress_seed=True；③ **reject**：仅明显错位（含 lexical_not_term）。返回 retain_mode、suppress_seed、primary_eligibility_reasons。 |
| `compute_primary_score` | **主干**：0.30×identity + 0.26×**context_consistency** + 0.24×jd_align + 0.10×raw_sim + 0.06×cross + 0.04×neighbor；**hierarchy 只微调**（0.04×field + 0.05×subfield + 0.05×topic + 0.03×path + 0.02×specificity），span_factor=0.95+0.05×topic_span_penalty；weak_retain ×0.92；**similar_to 且无 context_supported ×0.90**。 |
| `choose_better_term_with_hierarchy` | 新增 **head_term_bonus**：对明显主干表达（如 机械臂→robotic arm、运动控制→motion control）加小 bonus，避免 robot hand / robotic hand 长期压过 robotic arm。 |
| **Stage2B** | |
| `check_seed_eligibility` | 返回 (eligible, seed_score, block_reason)。**无 fallback**：仅 eligible 的 primary 进 diffusion_primaries，无可扩则 []。① **weak_retain 或 suppress_seed** 一律不扩；② source∈可信；③ **is_semantic_mismatch_seed** 直接 block；④ **is_primary_expandable**（本体主词/多锚骨干可扩，q-learning/digital control/automatic control/route planning 等禁止扩散）；⑤ **seed_expand_factor**：is_narrow_method_term×0.75、is_device_or_object_term×0.65、support_count==1×0.85。准入后打 **seed_blocked / seed_block_reason**。 |
| `expand_from_vocab_dense_neighbors` | 入口对每个 primary 再跑 **check_seed_eligibility**；默认须过 **support_expandable_for_anchor**（四道门 + keep_score≥0.76）。**强 seed**（见上文「三函数最小收尾」）可 **domain_fit 下限 0.58**、**max_keep=3**、检索 k 放宽，并在四门失败时用 **weak_support_release_for_strong_seed** 弱放行（须锚点有 **conditioned_vec**）。**Dense 不压过 parent**：**keep_score = min(…, parent.primary_score × DENSE_PARENT_CAP(0.85))**。sim≥0.55；弱 seed 仍 domain_fit≥0.72。 |
| `expand_from_cluster_members` | **当前全关**：直接 `return []`。若将来重开，须三重门：强 seed（primary_score≥0.70、cross_anchor_support_count≥2）、cluster purity 高（cluster_confidence≥0.75）、member 过 **support_expandable_for_anchor**；不再使用固定 identity_score=0.5。 |
| `expand_from_cooccurrence_support` | 仅 **强 normal seed** 触发：retain_mode==normal、primary_score≥0.70、jd_align≥0.82、cross_anchor_support_count≥2。共现词须过 **support_expandable_for_anchor**、domain_fit≥0.75、freq≥3；每 seed 最多 **2 条**。 |
| `merge_primary_and_support_terms` | 将 **retain_mode、topic_source、seed_blocked、seed_block_reason** 写入 ExpandedTermCandidate，供 Stage3 与日志使用。 |

`_compute_hierarchy_match_score` 保留，但**仅用于 debug/explain**；准入与排序以 **compute_hierarchy_evidence** 的连续 **effective_*** 为准，不做硬档位拒绝。

**数据驱动 Primary 选主（零硬编码）**

Stage2A 不再使用「subfield_fit / topic_fit / outside_subfield_mass 硬门槛一票否决」或任何词级黑/白名单，改为**纯数据驱动**的 primary 打分与单一决策链：

1. **证据量（无词表）**  
   - **edge_affinity**：SIMILAR_TO 边权（identity/semantic_score）。  
   - **anchor_candidate_alignment** / **conditioned_anchor_align**：候选与**当前锚点**的语义相似度；若有条件化表示则用候选与 **conditioned_vec** 的相似度，否则用边权。  
   - **jd_candidate_alignment**：候选向量与 **JD 整体向量** 的余弦相似度。  
   - **multi_anchor_support**：候选与**其它锚点**条件化向量的平均相似度（多锚共识）。  
   - **hierarchical_consistency**：候选的 domain/subfield/topic 与 jd_profile 的契合度。  
   - **local_neighborhood_consistency**：候选与当前批次其它候选的向量平均相似度。  
   - **semantic_isolation_penalty**：与候选群整体语义的离群程度。

2. **单一 primary 链（无两套并存）**  
   - 对每个锚点调用 `collect_landing_candidates`，保留 **top-m** 候选；组成 **flat_pool** 后统一算邻域与 conditioned_anchor_align。  
   - **准入**：`check_primary_admission`（PRIMARY_MIN_HIERARCHY_MATCH、PRIMARY_MIN_PATH_MATCH，source 用 CONDVEC_SOURCE_FACTOR 折扣）；**打分**：`compute_primary_score`（仅 **PRIMARY_SCORE_W_*** 一套权重：semantic、identity、jd_align、field/subfield/topic/path/specificity 等）。  
   - **冲突消解**：`resolve_anchor_local_conflicts` 后每锚取前 **PRIMARY_MAX_PER_ANCHOR** 个为 primary，进入 Stage2B。不再使用 select_primary_academic_landings 或 _primary_score_data_driven/PRIMARY_W_*。

3. **Identity Gate（锚点本义守门，软本义分）**  
   - 目的：压制「JD 场景很像、但概念不对」的错义（propulsion、simula、control flow、data retrieval 等），**不再把正常中英映射一刀砍死**（motion control、reinforcement learning、robotic arm、medical robotics 等应能进 primary）。  
   - **anchor_identity_score**：由 `compute_anchor_identity_score` 计算，**软本义分**：① 别名命中（**ANCHOR_IDENTITY_ALIASES**：含 动力学/运动学/仿真/**路径规划**（route planning、path planning、motion planning、trajectory planning）等）给 **base_floor≥0.34**，中英 token 重叠/head 一致给 0.26～0.30 基础分；② score = max(base_floor, 0.40×exact_or_substring + 0.35×token_overlap + 0.25×head_consistency）；③ 泛词/歧义惩罚保留但放宽（generic_penalty 0.72、ambiguity_penalty 0.78），明显错义词（propulsion、simula、control flow、data retrieval 等）仍 0.45。  
   - **identity_gate**：软闸门，不硬删；低 identity 候选分数被乘 0.45～0.72，高 identity 保持 1.0。  
   - **collect 不提前判死**：collect_landing_candidates 内不再设 primary_cap=secondary_only；准入与可扩由 check_primary_admission / select_primary_per_anchor 统一处理。

4. **调试用证据表**  
   - **stage2_anchor_evidence_table**：每行对应一个 (锚点, 候选)，字段包括 anchor、candidate、tid、edge_affinity、anchor_align、**conditioned_anchor_align**、**multi_anchor_support**、jd_align、hierarchy_consistency、neighborhood_consistency、isolation_penalty、**anchor_identity_score**、**identity_gate**、**base_primary_score**、primary_score。  
   - 写入 `debug_info.stage2_anchor_evidence_table` 与 `debug_1["stage2_anchor_evidence_table"]`，诊断时看「为何某候选胜出/某候选被压」无需任何词表，只看上述结构量。  
   - 控制台在 `LABEL_EXPANSION_DEBUG=True` 时先打印 **【Stage2A Identity Gate】**（anchor | candidate | source | identity | gate | base | final）前 30 条，再打印证据表前 40 条；`label_debug_cli` 深度诊断中也会打印该表（前 35 条）。

**最小施工单（八项，零硬编码）**

在「数据驱动 Primary 选主」基础上，以下八项已落地，**不依赖任何词黑/白名单**，仅靠上下文、图结构、语义一致性与多锚共识：

| 序号 | 内容 | 位置 |
|------|------|------|
| 1 | **Step2 锚点重排**：四维短语特征 → **support_mean 凸组合** → **`final = backbone × (0.72+0.28·support_mean)`**（替代旧版四因子近连乘塌陷）；抑制「上下文漂亮但整体乘塌」的排序扭曲，利于主线工业锚先于加分锚 | `label_anchors.extract_anchor_skills`；常量 **`ANCHOR_SUPPORT_W_*`、`ANCHOR_FINAL_BACKBONE_FLOOR_FRAC`、`ANCHOR_FINAL_SUPPORT_SPREAD_FRAC`**；**`[Stage1-Step2 anchor collapse audit]`**（`ANCHOR_COLLAPSE_AUDIT`） |
| 2 | **条件化锚点表示**：`build_conditioned_anchor_representation` 为每个锚点构造 local_phrase_vec、co_anchor_vec、jd_vec，与 anchor_vec 按 specificity 加权得到 conditioned_vec；泛锚点更多依赖上下文 | `label_anchors.build_conditioned_anchor_representation`；Stage1 中挂到 `anchor_skills[v_id]["conditioned_vec"]` |
| 3 | **Stage2A 使用 conditioned_vec 作上下文复核**：similar_to 主池 + conditioned_vec 仅产出 **rerank_signals**（context_sim/context_supported/context_gap）；主池≤1 时补 1～2 个 context_fallback；准入与打分纳入 **compute_context_consistency**、**check_primary_eligibility**、双空间一致/rescue/context_fallback 通道 | `label_expansion._retrieve_academic_terms_by_conditioned_vec`（返回 context_neighbors + rerank_signals）；`collect_landing_candidates`；`compute_context_consistency`；`check_primary_eligibility`；`check_primary_admission`；`compute_primary_score` |
| 4 | **jd_global_align 入 primary 打分**：候选与 JD 整体向量的相似度参与 primary_score | `collect_landing_candidates` 中 `jd_candidate_alignment`；`compute_primary_score` 中 jd 权重 0.24 |
| 5 | **multi_anchor_support 入 primary 打分**：候选与其它锚点条件化向量的平均相似度 | `_compute_conditioned_anchor_align_and_multi_anchor_support`；`compute_primary_score` 中 PRIMARY_SCORE_W_CROSS_ANCHOR / NEIGHBOR |
| 6 | **Stage2B 两层门**：Seed 门 `check_seed_eligibility`（weak_retain/suppress_seed 不扩、is_primary_expandable、**无 fallback**）；Support 门 dense/cooc 过 **support_expandable_for_anchor**；**cluster 全关** | `label_expansion.check_seed_eligibility`、`is_primary_expandable`、`support_expandable_for_anchor`、`expand_from_*` |
| 7 | **Stage3 唯一主分**：score_term_record 用 base_score、gate、cross_anchor_factor、backbone_boost、object_like_penalty、bonus_term_penalty；_compute_stage3_global_consensus 只写 cross_anchor_evidence、semantic_drift_risk（仅 debug），无 cluster | `hierarchy_guard.score_term_record`、`_compute_backbone_boost`、`_compute_object_like_penalty`、`_compute_bonus_term_penalty`；`_compute_stage3_global_consensus` |
| 8 | **README 与调试**：上述逻辑与证据表字段（含 conditioned_anchor_align、multi_anchor_support）已写入本文档与证据表打印 | 本节 + stage2_anchor_evidence_table 表头 |
| 9 | **Identity Gate（锚点本义守门，软本义分）**：`compute_anchor_identity_score` 采用**软本义分**（别名命中 base_floor、错义词惩罚保留）；`check_primary_eligibility` 只拦特别危险；`check_primary_admission` 上下文优先、identity 软约束、hierarchy 边界弱保留；**compute_primary_score** 中 identity 与 context_consistency 参与；**check_seed_eligibility** 使用 SEED_MIN_IDENTITY；压制 propulsion/simula 等错义，不误杀 motion control/reinforcement learning/robotic arm 等正常映射 | `label_expansion` 中 compute_anchor_identity_score、check_primary_eligibility、check_primary_admission、compute_primary_score、check_seed_eligibility；`[Stage2A Identity Gate]` 打印 |

**领域过滤增强（主领域守卫、层级不否决、门槛与 Context-Aware / Term 纯度）**

为减轻「多义词/跨领域词」导致的领域错配（如“机器人 JD”召回网安/航模作者），标签路在以下多处做了增强：

| 增强项 | 位置 | 说明 |
|--------|------|------|
| **主领域守卫（Hard Domain Guard）** | `label_expansion._term_in_active_domains` / `_term_in_active_domains_with_reason` | 在「有交集」通过后增加**硬约束**：候选词的 **主领域**（domain_dist 中权重最大的领域）必须在 `active_domains` 内，否则直接 discard；避免如 vibration（土木主领域）、reinforcement learning（博弈/安全主领域）误入机器人召回。 |
| **domain_no_match 区分强冲突与 soft retain（Stage2A 漏点）** | `_term_in_active_domains_with_reason`、`retrieve_academic_term_by_similar_to` | **SIMILAR_TO 候选**：domain_ok=False 时，若主领域在 **STRONG_CONFLICT_DOMAIN_IDS**（如医学/社科/管理，空集则未启用）→ 返回 "domain_conflict_strong" 硬拒；否则为 "domain_no_match" → 在 **retrieve_academic_term_by_similar_to** 内做 **soft retain**（soft_domain_retain=True、domain_fit×0.85），保留进池，mechanics/simulation/route planning 等上位词可进后续 admission。**collect_landing_candidates** 中对 soft_domain_retain 候选在 _compute_domain_fit 后再乘 0.85。 |
| **topic_hierarchy_no_match 不再一票否决** | `_term_in_active_domains_with_reason`、`_term_in_active_domains` | 当三级层级均未命中时**不再剔除**候选，仅保留 reason 供日志；候选进池后由 **check_primary_admission**（PRIMARY_MIN_HIERARCHY_MATCH、PRIMARY_MIN_PATH_MATCH）与 **compute_primary_score**（PRIMARY_SCORE_W_* 含 hierarchy 项）统一处理。无 _hierarchy_norm/PRIMARY_W_* 旧链。 |
| **补充锚点来源与权重** | `label_anchors.extract_anchor_skills`、`supplement_anchors_from_jd_vector`、`_anchor_skills_to_prepared_anchors`、Stage2A primary 循环 | **extract_anchor_skills** 写入的锚点带 `anchor_source="skill_direct"`、`anchor_source_weight=1.0`；**supplement_anchors_from_jd_vector** 写入的带 `anchor_source="jd_vector_supplement"`、`anchor_source_weight=ANCHOR_SOURCE_WEIGHT_JD_SUPPLEMENT`（默认 0.65）。**PreparedAnchor** 含 **source_type**、**source_weight**；Stage2A 在算完 base×identity_gate 后 **primary 再乘 anchor.source_weight**，使 Robotics/Robot control 等补充锚点下的候选（如 Telerobotics、Medical robotics）不会被顶到全局最前，同时仍参与召回。 |
| **domain_fit 门槛提高** | `config.py` | **DOMAIN_FIT_MIN_PRIMARY**：0.25 → **0.45**（做 primary 须近半“生命力”在目标领域）；**DOMAIN_FIT_HIGH_CONFIDENCE**：0.35 → **0.55**（仅主场明显的词才可参与 Stage2B 扩散）。 |
| **Context-Aware Query** | `label_expansion.query_expansion_by_context_vector` | 构造编码输入时不再仅用裸锚点词（如“动力学”），而是拼接 JD 片段：`term + " (" + context_snippet + ")"`（context_snippet 为 query_text 前 100 字），使 embedding 向目标领域偏移，减少“运动学→体育”“控制→管理”等误匹配。 |
| **Term 领域纯度（Stage5）** | `label_path._build_term_uniqueness_map` + `paper_scoring.compute_contribution` | 对每个 term 用 `vocabulary_domain_stats` 算**目标领域占比**（target_degree_w/degree_w_expanded）作为 **term_uniqueness**；Stage5 论文贡献度中 **paper_term_contrib** = term_weight × term_confidence × paper_match_strength × **term_uniqueness**，领域专属性强的词（如 robotic arm）抬权，通用词（如 RL）降权。 |

---

**Stage2B：学术侧补充（两层门：Seed 门 + Support 门）**

| 项 | 说明 |
|----|------|
| **作用** | **两层门**：① **Seed 门**（谁能扩）：仅 `check_seed_eligibility` 通过且 **is_primary_expandable** 的 primary 进入 diffusion_primaries，**无 fallback 兜底**；② **Support 门**（扩出来的词谁能留）：dense/cooc 候选须过 **support_expandable_for_anchor**（锚点语义复核）、domain_fit/sim 门槛、设备/对象词默认拒。仅围绕 diffusion_primaries 做 dense/cooc 扩展（**cluster 当前全关**）；合并后转为 raw_candidates 供 Stage3。 |
| **思路** | 不再「只要 primary 就能往外扩」，改为「只有适合扩散的 primary 才能扩；扩出来的 support 再过锚点语义复核」。扩展来源：① 词汇向量索引上 seed 的学术近邻（**dense**：sim≥0.55、domain_fit≥0.72、每 seed 最多 2/3 条）；② **cluster 当前全关**（return []）；③ 共现表（**cooc**：仅强 normal seed、freq≥3、每 seed 最多 2 条）。 |
| **Seed 定义** | `check_seed_eligibility(label, primary, jd_profile)` 返回 (eligible, seed_score, block_reason)。**必过**：retain_mode==normal、非 suppress_seed；source∈可信；非 is_semantic_mismatch_seed；**is_primary_expandable**（本体主词/多锚骨干可扩，q-learning/digital control/automatic control 等禁止扩散）。**seed_expand_factor** 惩罚：窄方法词×0.75、设备/对象词×0.65、单支撑×0.85。**无 fallback**：无可扩 seed 则 diffusion_primaries=[]。 |
| **逻辑流程** | ① **check_seed_eligibility** 筛出 **diffusion_primaries**（无兜底）；② **expand_from_vocab_dense_neighbors**（入口审 seed、support 过 support_expandable_for_anchor）；③ **expand_from_cluster_members** 当前 return []；④ **expand_from_cooccurrence_support**（仅强 seed、support 过 support_expandable_for_anchor）；⑤ merge_primary_and_support_terms；⑥ _expanded_to_raw_candidates → raw_candidates。 |
| **输入参数（名字与含义）** | 同 Stage2 总入口，含 **jd_profile**（可选）；内部使用 **primary_landings** 及 **diffusion_primaries**（仅 check_seed_eligibility 通过且 is_primary_expandable，无 fallback）。 |
| **输出参数（名字与含义）** | **raw_candidates**：每项含 tid、term、term_role、identity_score、source、degree_w、domain_span、domain_fit、parent_anchor、parent_primary；**Stage3 正式层级字段**：field_fit、subfield_fit、topic_fit、path_match、genericity_penalty（及 outside_subfield_mass 供 should_drop_term）；**rec["_debug"]** 含 topic_align、topic_confidence、outside_topic_mass、topic_entropy、main_subfield_match、landing_score 等仅供排查。 |
| **主要公式** | **support 准入**：domain_fit ≥ max(SUPPORT_MIN_DOMAIN_FIT, **0.72**)、domain_span ≤ DOMAIN_SPAN_EXTREME；**dense**：sim≥**0.55**、过 **support_expandable_for_anchor 四道门**（primary/anchor/context/family 一致性，无硬编码词表）、非设备/对象词（或 anchor 允许）；**cooc**：freq≥3、domain_fit≥0.75、过 support_expandable_for_anchor。topic_align 来自 _attach_topic_align，仅写 _debug。 |
| **Dense 最小修复补丁（再收一刀）** | **目标**：拦住 robotic arm→robotic hand/robot hand、motion control→motion controller 等脏扩散。**support_expandable_for_anchor** 返回 (keep, meta)。① 四道门：primary_consistency≥0.72、anchor_consistency≥0.70、context_stability≥0.72、family_support≥0.68；② **anchor_identity_for_support** = compute_anchor_identity_score(anchor_term, candidate_term)，<0.16 直接拒；③ **drift_penalty** = max(0, primary_consistency − family_support)；keep_score = 0.26×primary + 0.26×anchor + 0.22×context + 0.16×family + 0.10×anchor_identity − 0.18×drift_penalty，≥0.76 保留。meta 含 keep_score、anchor_identity_for_support、drift_penalty。 |
| **调用的表/知识图谱** | **Neo4j**：**Vocabulary**；**Faiss**：Vocabulary 向量索引（dense 近邻）；**SQLite**：`vocabulary_domain_stats`、`vocabulary_cooccurrence`（term_a、term_b、freq）、`vocabulary_topic_stats`；**簇数据**：当前未用于扩散（cluster 全关）。 |

---

**Stage3：全局复审层（去重聚合 + 分层准入 + 软 identity + 分桶 + top_k）**

| 项 | 说明 |
|----|------|
| **作用** | 对 raw_candidates **按 tid 去重聚合**（**`_merge_stage3_duplicates`**：统计类 max/OR，**`primary_bucket`** 按 **`STAGE3_PRIMARY_BUCKET_MERGE_PRIORITY` 取最强来源**，`mainline_candidate`/`can_expand*` **OR**，`fallback_primary` **仅当每来源均为 fallback**，`role_in_anchor` **任一为 mainline 则 mainline**，`term_role` 按 **`STAGE3_TERM_ROLE_MERGE_PRIORITY`**；其余字段仍来自 **seed 分最高**记录，同分取 **primary_bucket 更优先**者）后：① classify_stage3_entry_groups … ⑨ 写 map、tag_purity_debug。**目标**：消除「多锚统计已聚合、语义字段却被弱来源整条覆盖」的撕裂（如 motion control）。 |
| **思路** | 分两段：**结构性 pipeline 分**保留证据链；**全局排序主序**改为连续模型，**bucket 不当生死裁判**。 |
| **逻辑流程** | ① _merge_stage3_duplicates；② _compute_stage3_global_consensus；③ _build_family_buckets、_compute_family_centrality；④ **_classify_stage3_entry_groups**；⑤ should_drop_term → **_check_stage3_admission**；⑥ score_term_record × identity × risk × **role_factor**；⑦ **`_assign_stage3_bucket`**；⑧ _collect_risky_reasons、**_collect_stage3_bucket_reason_flags**（基于 pipeline 分，供 debug）；⑨ **`stage3_build_score_map`**：写入 **统一 `final_score`**、重排、再 **`_apply_family_role_constraints`**、重刷 **`bucket_reason_flags`**；⑩ _bucket_stage3_terms；⑪ **`select_terms_for_paper_recall`**；⑫ 写 map、**_debug_print_stage3_tables**。 |
| **输入参数（名字与含义）** | **raw_candidates**：Stage2 输出，含 term_role、role_in_anchor、can_expand、retain_mode、polysemy_risk、object_like_risk、generic_risk、context_continuity、jd_candidate_alignment 等；**query_vector**；**anchor_vids**：可选。 |
| **输出参数（名字与含义）** | **score_map**、**term_map**、**idf_map**、**term_role_map**、**term_source_map**、**parent_anchor_map**、**parent_primary_map**、**paper_terms**；**tag_purity_debug** 含 stage3_entry_group、identity_factor、risk_penalty、stage3_bucket、retrieval_role、stage3_explain 等。 |
| **主要公式** | **第一段**（pipeline）：**score_term_record** → × **identity_factor** × **_compute_stage3_risk_penalty** × **role_factor**（写入 **`_stage3_pre_adjust_score`** 供审计）。**第二段**（**`stage3_build_score_map`**）：**统一连续分** `final_score = clamp01( Σ w⁺·特征⁺ − Σ w⁻·风险⁻ )`，权重为模块常量 **`STAGE3_UNIFIED_W_*`**；再 **`_apply_family_role_constraints`**。**分桶公式**（`_assign_stage3_bucket`）不变，但 **仅影响观测标签与 `retrieval_role` 软分档**，**不**再乘入第二段。**Paper**：`sort(eligible, final_score desc)` + **family 去重** + **`dynamic_floor`** + **topN**；硬挡 **单锚 conditioned_only**、**fallback_primary**。**`_collect_stage3_bucket_reason_flags`** 语义同前（flags 与统一分并存，供日志）。 |

**Stage3 主分**：**compose_term_final_score** 已退出主链（deprecated）。当前对外 **`final_score`** = **统一连续分**（第二段）；**`stage3_explain`** 仍保留第一段因子（identity、risk_penalty、role_factor、base_score 等）与 **`unified_continuous_score`**。

**Stage3 两函数硬门（封死 conditioned_only 单锚 → core / paper，实现：`label_pipeline/stage3_term_filtering.py`）**

| 项 | 说明 |
|----|------|
| **目标** | **`_assign_stage3_bucket`**、**`select_terms_for_paper_recall`** 消费 Stage2A 透传，区分 **fallback primary** 与 **locked mainline**（`primary_keep_no_expand` / `usable_mainline_no_expand` 等），避免「mainline_hits=0 且不可扩」误伤真主线保留词。仍关死 **conditioned_only 且单锚** 进 **core** 与 **final_term_ids_for_paper**。**conditioned_only** 定义不变：含 **`conditioned_vec`** 且无 **`similar_to`**、无 **`family_landing`**。 |
| **函数 1：分桶** | **is_fallback_primary** → **risky**。**单锚** conditioned_only：**support** 若 has_mainline_support（强主线 ∨ locked mainline），否则 **risky**；**永不为 core**。**多锚** conditioned_only：**support**。**is_locked_mainline**（非 fallback）→ **support**。**`primary_expandable`→`core`**：`mainline_hits≥1` ∧ `can_expand` ∧（**`identity_factor≥STAGE3_EXPANDABLE_CORE_IDENTITY_HARD(0.95)`** ∨（**¬conditioned_only** ∧ **`jd_align≥STAGE3_EXPANDABLE_CORE_JD_ALIGN_MIN(0.50)`**）），再经 **`_cap_core_if_conditioned_single_anchor`**；未升舱则 **`stage3_core_miss_reason`** 解释。其余回退路径：**core** 仍要求 `mainline_hits≥1` ∧ `can_expand` ∧ **`identity_factor≥0.95`**（**core_ok**）。**未达 core** 时：**support** 若 has_mainline_support 否则 **risky**；**support** 仍可能被 **`_stage3_single_anchor_expand_branch_demote`** → **risky**。 |
| **函数 2：paper** | **conditioned_only** ∧ **单锚** → **`conditioned_only_single_anchor_block`**。**is_fallback_primary** → **`fallback_primary_block`**。**weak_support_contamination**：**软放行**（**`SUPPORT_TO_PAPER_*` / `PAPER_SUPPORT_*_FACTOR`**）→ **`support_pool`**；否则 **`weak_support_contamination_block`**。**`risky_side_block`**：`risky` ∧ ¬扩 ∧ `mainline_hits≤0` ∧ **`term_role` 为 side/expansion`**。**JD 主轴门**：**仅 core**，**`(rk≤MAX) ∨ (父锚分≥MIN)`** → **`paper_recall_quota_lane=primary`**；**否则** **`core_axis_near_miss_soft_admit`**（×**`PAPER_CORE_AXIS_NEAR_MISS_FACTOR`**，lane=support）进 **`support_pool`**。**未硬挡** 的 **support / risky / 其它** 先入 **`support_pool`**，再 **`eligible.extend(support_pool)`**。**`paper_select_score` 排序**，**`family_key` 去重**，**`dynamic_floor`**，**`paper_recall_quota_lane=support`** 计 **support 配额**（**`support_quota_full`**），**topN**；**`retrieval_role`**：**primary lane**→`paper_primary`，否则 **`paper_support`**。 |
| **函数 3（日志，建议同步）** | **`_collect_stage3_bucket_reason_flags`**：`no_mainline_support` / `only_weak_keep_sources` 与上述 **has_mainline_support**、**is_fallback_primary** 对齐；增补 **`locked_mainline_no_expand`**、**`fallback_primary`**、**`conditioned_only_core_cap`**（单锚 conditioned 从 core 压降时）。 |
| **全局分：`stage3_build_score_map`** | **`stage3_term_filtering.stage3_build_score_map`**：在 bucket 已赋值、flags 已收集之后，**用连续特征重算 `final_score`**（见 **`_compute_stage3_unified_continuous_score`**），**不再读取 bucket 做乘子**；然后 **`_apply_family_role_constraints`**，重排并重刷 **`stage3_rank` / `bucket_reason_flags`**。 |
| **调试审计（可选）** | **`STAGE3_AUDIT_DEBUG`**：**`[Stage3 final adjust audit]`** … **`[Stage3->Paper bridge]`**。**`STAGE3_UNIFIED_SCORE_DEBUG`** + **`STAGE3_UNIFIED_SCORE_DEBUG_TOP_K`**：**`[Stage3 unified score breakdown]`**（默认短表；焦点词集非空时先筛焦点）。**`STAGE3_DUPLICATE_MERGE_AUDIT`**、**`STAGE3_CORE_MISS_AUDIT`** 同前。**`STAGE3_PAPER_CUTOFF_AUDIT`**：**`[Stage3 paper gate summary]`**、**main-axis**、**cutoff** 等。**`DEBUG_LABEL_PATH`**：**`[Stage3 rerank summary]`**。 |

**`stage3_build_score_map` 词序验收参考（典型机器人/控制类 JD）**：跑一条召回后重点看 **`Motion control`**、**`medical robotics`**、**`q-learning`**、**`Educational robotics`**、**`Telerobotics`** 的相对顺序；期望 **Educational robotics / Telerobotics / Robot manipulator** 等弱证据侧翼被压到 **medical robotics / q-learning / robotic hand** 等强主线相关词之后，或至少不再单靠多锚/跨锚把侧翼顶到前排。

**Stage3 六因子：来源与默认值策略**（仅用于 compose_term_final_score 初分，已 deprecated）

| 因子 | 含义 | 来源 | 默认值/策略 |
|------|------|------|-------------|
| **base_score** | 按 term_role 的 identity 与 quality 加权 | rec.term_role + rec.identity_score + rec.quality_score（quality 由 score_term_expansion_quality 计算） | primary: 0.7×identity+0.3×quality；dense/cluster: 0.4×identity+0.6×quality；cooc: 0.3×identity+0.7×quality；缺省: 0.5×identity+0.5×quality |
| **source_weight** | 按来源可信度加权 | rec.source / rec.origin + **config** | SOURCE_WEIGHT_SIMILAR_TO=1.0、SOURCE_WEIGHT_JD_VECTOR/conditioned_vec=0.95、SOURCE_WEIGHT_DENSE=0.85、SOURCE_WEIGHT_CLUSTER=0.75、SOURCE_WEIGHT_COOC=0.70；TRUSTED_SOURCE_TYPES_FOR_DIFFUSION 含 similar_to、jd_vector、**conditioned_vec**；未匹配时取 1.0 |
| **domain_gate** | 三层领域匹配乘回 | rec.domain_fit + **config** | domain_gate = DOMAIN_GATE_MIN + (1−DOMAIN_GATE_MIN)×domain_fit；如 DOMAIN_GATE_MIN=0.5。**若 domain_fit 缺失，不取 1.0**，退化为保守默认值（如 DOMAIN_GATE_MIN 或中性值 0.75） |
| **task_consistency** | JD 语义 + 强锚点共振（**当前主要通过 topic_align 落地**） | rec.topic_align、rec.topic_level + **config** TOPIC_WEIGHT_* / TOPIC_MIN_ALIGN / TOPIC_LOW_ALIGN_PENALTY | task_consistency = 1 - topic_weight + topic_weight×topic_align；若为 expansion 且 topic_align < TOPIC_MIN_ALIGN(0.20)，再乘 TOPIC_LOW_ALIGN_PENALTY(0.50)；topic_weight 按 role 取 TOPIC_WEIGHT_PRIMARY/DENSE/CLUSTER/COOC |
| **role_penalty** | 对不同 term_role 施加保守角色权重 | rec.term_role | term_scoring._get_role_penalty：primary=1.0、dense_expansion=0.95、cluster_expansion=0.90、cooc_expansion=0.85；缺省 1.0 |
| **expansion_penalty** | 对 cluster/cooc 类扩展额外惩罚 | rec.term_role + **config** | CLUSTER_EXPANSION_PENALTY=0.75、COOC_EXPANSION_PENALTY=0.65；仅 cluster_expansion/cooc_expansion 生效，其余为 1.0 |

---

**Stage4：论文召回（二层召回 + 领域软奖励）**

| 项 | 说明 |
|----|------|
| **作用** | 用 **final_term_ids_for_paper** 在 Neo4j 上沿 **HAS_TOPIC** 做**二层召回**：先按 term 拉 Work、做 **per-term 限流** 与 **paper_score** 聚合，再在 Stage4 内部先按 **wid 合并多 term hits**（同 vid 去重保留强证据），之后再**全局排序**取前 2000 篇并查作者聚合。**不再对论文做 domain 硬过滤**，改为**领域软奖励**（匹配则乘 DOMAIN_BONUS_MATCH，否则 1.0）；**paper_score** 纳入 **Stage3 的 term_final_score**，避免歪词/泛词单靠 idf 占坑。 |
| **思路** | **词侧**：熔断放宽为 **MELT_RATIO**（默认 5%），(degree_w/total_w) ≥ 0.05 的泛词在 Cypher 内过滤。**论文侧**：不做 `WHERE w.domain_ids =~ $regex` 剔除，仅用 **domain_bonus**（匹配 1.2、不匹配 1.0）参与打分。在原有 term-grounding 基础上，新增 **JD↔论文摘要向量一致性（jd_align）**，但只做弱辅助：`jd_boost = 0.85 + 0.15*jd_align`（当 `term_grounding < 0.20` 时改为 `0.92 + 0.08*jd_align`），`hybrid_grounding = term_grounding * jd_boost`。这保证“先像 term，再由 JD 精排”，避免 JD 相似度把 term 不够像的论文抬上来。再叠加 **term-type 因子**（来自 `term_meta`：`can_expand`、`stage3_bucket`、`retrieval_role`、`object_like_penalty`、`bonus_term_penalty`、`generic_penalty`）：方法骨架词略放宽、对象词进一步收紧；最后 `term_contrib = base_term_contrib × (0.10 + 0.90*hybrid_grounding) × off_topic_penalty × term_paper_role_factor`。**`retrieval_role==paper_support`** 时再乘 **0.85**，并按 **`term_grounding`** 分档：**&lt;0.10 → ×0.25**，**&lt;0.18 → ×0.45**，**&lt;0.28 → ×0.70**（软压、非硬杀），避免 support 词与 core 在 **wid 聚合 / 作者榜** 上同台抢分。入池门控仍为双阈值：`primary_gate: term_grounding>=0.12`；`secondary_gate: term_grounding>=0.06 且 jd_align>=0.78`（secondary 命中按 0.65 降权后入池）。每个 term 再做 **动态 local cap**；先 **paper_score** = Σ term_contrib 并 **按 wid 合并 hits**；再对每篇 wid 乘 **hierarchy_consensus_bonus**（见下：**三级领域只属于词**，用命中词群在 field/subfield/topic 上的加权投票与 Stage1 **jd_profile** 的分布 overlap 得软乘子，**不**依赖论文自身三级领域）；最后全局 ORDER BY paper_score DESC、LIMIT 2000。 |
| **逻辑流程** | ① 输入 **vocab_ids**、**term_map**、**score_map**、**term_retrieval_roles/term_meta**（均来自 Stage3），Stage4 **不再对 term 做 extract_skills 或二次清洗**；② **第一层 Cypher**：MATCH (v:Vocabulary) WHERE v.id IN $v_ids，WITH degree_w，WHERE (degree_w/total_w) < MELT_RATIO，WITH idf_weight；MATCH (v)<-[:HAS_TOPIC]-(w:Work)，计算 domain_bonus（CASE WHEN regex 匹配 THEN 1.2 ELSE 1.0），并 RETURN `vid, wid, idf_weight, domain_bonus, year, title, domains`；③ **Python**：先懒加载摘要向量资源（`abstract_vectors.npy` + `abstract_mapping.json`），构建 `paper_id -> row_idx`；对每个 (vid,wid) 取 `term_final_score` 与 `role_weight` 得 `base_term_contrib`，算 `term_grounding/off_topic`，再算 `jd_align`（query_vec 与 abstract_vec 余弦归一到 [0,1]），得到 `jd_boost` 与 `hybrid_grounding`，再乘 `term_grounding_factor` 与 `term_paper_role_factor` 得 `term_contrib`；入池门控采用 **primary+secondary 双阈值**（`primary: grounding>=0.12`，`secondary: grounding>=0.06 且 jd_align>=0.78`），secondary 命中按 0.65 降权后进入 per-term 候选；每 term 按 term_contrib 降序，使用 **动态 local cap** 截断；随后在 Stage4 内按 wid 聚合 `paper_score = Σ term_contrib`，并执行 `hits = merge_hits_by_vid(...)`（同 vid 去重，保留较强 idf）；**④ wid 级三级共识**：从 **`recall._last_stage1_result.jd_profile`** 取 `field_weights/subfield_weights/topic_weights`，用 **`vocabulary_topic_stats`**（与 `stage1_domain_anchors._batch_load_vocabulary_stats_for_tids` 同路径）批量拉取命中词的 `field_id/subfield_id/topic_id` 或 dist；对每篇 wid 上全部 hit 用权重 **`term_score × idf`** 混合三层 dist，与 JD 分布算 overlap，得 **`hierarchy_consensus`**；再对**本批全部 wid** 取 consensus **中位数** **`m`**，**`hierarchy_consensus_bonus = clip(1 + β·(consensus−m), lo, hi)`**（默认 β=12、**[lo,hi]=[0.82,1.15]**），**`paper_score *= bonus`**（无 jd 画像或 `STAGE4_HIERARCHY_CONSENSUS_ENABLED=0` 时跳过，乘子 1.0）；**⑤** 按 paper_score 降序取前 GLOBAL_PAPER_LIMIT(2000)；**⑥ 第二层 Cypher**：对上述 wids 查 (w:Work)-[:AUTHORED]-(a:Author) 并按 aid 聚合；⑦ 下发到 author_papers_list 时直接携带 Stage4 已合并好的 multi-hit（非单 term 副本）。 |
| **输入参数（名字与含义）** | **vocab_ids**：List[int]，即 final_term_ids_for_paper；**regex_str**：str，领域正则，用于 domain_bonus，可为空；**term_scores**：Dict[int, float]，vid→Stage3 的 final_score；**term_retrieval_roles**：Dict[int, str]，vid→paper_primary|paper_support|blocked，用于 role_weight（无则默认 1.0）。另外：**term_meta**（可选）：vid→{term,parent_anchor,parent_primary,retrieval_role,...}，用于 Stage4 grounding；**jd_text**（可选）：整段 JD 文本辅助 grounding。 |
| **输出参数（名字与含义）** | **author_papers_list**：List[Dict]，每项为 { **"aid"**, **"papers"**: [ { **"wid"**, **"hits"**, **"weight"**, **"title"**, **"year"**, **"domains"**, **"score"**（Stage4 聚合分，已含 **hierarchy_consensus_bonus**；可选供 debug） }, ... ] }；其中 `hits` 为 canonical term 证据列表（至少含 `vid/idf`，并保留 `term/role/term_score/paper_factor` 等调试字段）。 |
| **主要公式** | 熔断：`(degree_w * 1.0 / total_w) < MELT_RATIO`（默认 0.05）；**idf_weight** = `log10(total_w / (degree_w + 1))`；**role_weight**：paper_primary=1.0、paper_support=0.7、其他=0.4（**不看领域词，只看 retrieval_role**）；**base_term_contrib** = term_final_score × role_weight × idf_weight × domain_bonus × recency(year)；**term_grounding_raw**（0~1）来自 paper 的 `title/domains` 与 `term_meta`：`0.55*lexical_hit + 0.20*anchor_axis_hit + 0.25*jd_axis_match`，再乘 `term_grounding_factor`（方法骨架词 1.10~1.20、对象词当前收紧到约 **0.72**）；**jd_align**：`query_vec` 与摘要向量余弦，映射到 [0,1]；**jd_boost**：`0.85 + 0.15*jd_align`（若 `term_grounding < 0.20`，改为 `0.92 + 0.08*jd_align`）；**hybrid_grounding**：`term_grounding * jd_boost`（仅用于排序，不用于硬门）；**off_topic_penalty**（`_compute_grounding_score`）：泛交通/物流约 **0.55**；RL 非机器人/控制主轴约 **0.45**；**route planning** 交通/船舶/公交路线等约 **0.32**；**robotic arm** 非控制语境约 **0.60**；**robot control** 专项：标题含 chat/safety/LLM 等 **×0.25**；无运动/操作/机构学主轴证据（title/domains）**×0.55**；**supervised learning** 无机器人/控制/模仿等主轴 **×0.50**；**final_paper_score** = base_term_contrib × **(0.10 + 0.90×hybrid_grounding)** × off_topic_penalty × `term_paper_role_factor`；若 **`retrieval_role==paper_support`**（`term_meta` 或 `term_retrieval_roles`）：再 **×0.85**，且 **term_grounding &lt;0.10 → ×0.25**，**&lt;0.18 → ×0.45**，**&lt;0.28 → ×0.70**。入池硬门仍只看 gating 用 **term_grounding**（primary **≥0.12** / secondary **≥0.06 且 jd_align≥0.78**）；secondary 入 **`by_term`** 后再 **×0.65**；per-term **动态 local cap**。**wid 聚合后**：每条命中词从 **vocabulary_topic_stats** 构三层 **归一化 dist**（JSON/dict，key 一律 str；**有 dist 优先**，否则用 **field_id/subfield_id/topic_id** 补 one-hot）；按 **`term_score×idf`** 对三层 dist **加权混合**再归一化；**field_cons/subfield_cons/topic_cons** = 混合分布与 **jd_profile** 同层权重的 **Σ p·q**（Stage1 **`build_jd_hierarchy_profile`** 必须使用批量行 **`row[4]/[5]/[6]`** 作为 field/subfield/topic_dist，与词侧同源，否则 JD 与词 key 错位、overlap 恒为 0）；**hierarchy_consensus** = **0.20·field + 0.35·subfield + 0.45·topic**；**方案 B（相对中位数）**：**`median_c = median({consensus_w | wid∈本批候选})`**，**`Δ = hierarchy_consensus − median_c`**，**`hierarchy_consensus_bonus = clip(1 + β·Δ, lo, hi)`**（默认 **β=`STAGE4_HIERARCHY_BONUS_BETA`=12**，**`lo/hi`=`STAGE4_HIERARCHY_BONUS_CLIP_LOW/HIGH`=0.82/1.15**），**以 1.0 为锚**拉开「高于池内典型一致」与「低于典型」的论文，避免 raw consensus 整体偏小时映射成**近乎统一 ~0.82 缩放**；**paper_score_wid** = **(Σ term_contrib) × bonus**。详见 **`hierarchy_consensus_detail`**（**`median_consensus`**、**`delta_vs_median`**、**`beta`**）。全局 **GLOBAL_PAPER_LIMIT** = 2000。**审计**：**`[Stage4 jd-topic-profile audit]`**；**`[Stage4 topic-meta coverage audit]`**；**`[Stage4 hierarchy bonus distribution]`**（**`hierarchy_consensus` / `hierarchy_bonus`** 的 min·p25·p50·p75·max，及 **top boosted / top penalized**，**`STAGE4_HIERARCHY_DISTRIBUTION_TOP_N`** 默认 10）；**`[Stage4 hierarchy consensus audit]`** 分 **`top_by_paper_score_before`** / **`top_by_abs_score_delta`**（**`STAGE4_HIERARCHY_AUDIT_TOP_BY_SCORE`**、**`STAGE4_HIERARCHY_AUDIT_TOP_BY_DELTA`**）；**`[Stage4 support grounding audit]`**。**主题开关**：**`STAGE4_TOPIC_META_COVERAGE_AUDIT`**。**调试降噪**：**`STAGE4_PAPER_MAP_WRITE_VERBOSE`**；**`STAGE4_JD_AUDIT_TOP_K`**；**`STAGE4_JD_AUDIT_PRIMARY_DETAIL_K`**（默认 **3**，环境变量可调）与 **`STAGE4_JD_AUDIT_FULL`**：仅对 **Stage3 分最高的前 K 个 `paper_primary`** 打印 **`[Stage4 jd audit]`** 明细，**support** 等只保留一行 **`jd_audit_detail=false`**；**`STAGE4_OVERLAP_SURVIVAL_MAX_LINES`**（默认 **30**）限制 **overlap survival** 行数（优先 **`survive_after_cap=true`**）；**`STAGE4_HIERARCHY_CONSENSUS_ENABLED`**（默认开）、**`STAGE4_AUTHOR_PAYLOAD_AUDIT_VERBOSE`**。**`[Stage4 pre-gate paper score audit]`** 的 **`pre_gate_scored_rows`** = grounding 门前的评分行数（勿与 **after_cap** 混淆）。 |
| **调用的表/知识图谱** | **Neo4j**：**Vocabulary**（id）、**Work**（id, title, year, domain_ids）、**Author**（id）；边 **HAS_TOPIC**、**AUTHORED**（pos_weight）。**time_features.compute_paper_recency** 用于 recency_factor。 |

---

**Stage5：作者排序**

| 项 | 说明 |
|----|------|
| **作用** | 展开 **paper_map**、论文贡献度→作者；**`TERM_MAX_AUTHOR_SHARE`** 后由 **`_compute_author_structure_shape`** 用 **分段 `structure_factor`** 压 **单词单篇、无 mtp** 的作者、抬 **mtp≥1 或 st≥2**，**无词表特判**；×时间权重；再对 **纯 support 线、无 primary 论文托底** 的作者施加 **`support_only` 乘子**（**不改词分**）；× **AUTHOR_BEST_PAPER_MIN_RATIO** 与归一化，输出排序与 **last_debug_info**。 |
| **思路** | **compute_contribution** 与 **`STAGE5_SUPPORT_*`** 论文侧同前。**作者侧**：**TERM_CAP** 不改 dominant_share 形状；**`structure_factor`** 拉开「大池单词单篇」与「多词共现」；**`support_only` 乘子** 专压 **sup_share 高却无 primary 论文** 的冲顶作者，与 **support dominance**、**`[Stage5 support-only author penalty audit]`** 对照读。 |
| **逻辑流程** | ①～⑪ 读 debug、paper_map、贡献与护栏、递减与 **TERM_MAX_AUTHOR_SHARE** 同前；⑫ **×时间权重**；⑬ **`_compute_author_structure_shape`** → **`structure_mult_total`**；⑬′ **support-only 作者乘子**（**`SUPPORT_ONLY_*`**，见 **`[Stage5 support-only author penalty audit]`**）；⑭ **AUTHOR_BEST_PAPER_MIN_RATIO**、归一化；⑮ **term-cap**、**author structure**（base + 分项 mult）、support dominance、**term→author top（前 4 paper terms×5）**、**top-author term mix**、provenance；⑯ **aggregate_author_evidence_by_term_role**。 |
| **输入参数（名字与含义）** | **author_papers_list**：Stage4 输出（hits 含 **`role`**）；**score_map**、**term_map**：Stage3 输出；**active_domain_set**、**dominance**；**debug_1**：含 **term_role_map**、**term_confidence_map**、**term_uniqueness_map**、**term_family_keys**、**term_paper_meta**（vid→**mainline_hits**、retrieval_role 等）、**term_retrieval_roles**、industrial_kws、anchor_skills、query_vector、filter_closed_loop 等。 |
| **输出参数（名字与含义）** | **author_id 列表**：按得分降序，截断 recall_limit；**last_debug_info**：active_domains、dominance、score_map、term_map、term_role_map、term_confidence_map、filter_closed_loop、top_terms_final_contrib、work_count、author_count、**author_evidence_by_term_role** 等。 |
| **主要公式** | 论文侧同 **`paper_scoring`** · **`STAGE5_SUPPORT_*`**。**`structure_mult_total` = `structure_factor` × `term_strength_mult` × `paper_evidence_mult` × `multi_hit_mult`**。**st** = 贡献 ≥ 0.35·max(atc) 的强词数；**pc** = 保留论文唯一 wid 数；**mtp** = 其中 **wid_n_terms≥2** 的篇数。**structure_factor**：st≤1∧pc≤1∧mtp=0→**0.42**；st≤1∧pc≤2∧mtp=0→**0.58**；st≤1∧mtp=0→**0.72**；mtp≥1∨st≥2→**1+0.06·min(mtp,3)**；else **0.88**。**term_strength_mult**=0.90+0.10·min(st,3)/3；**paper_evidence_mult**=0.88+0.12·min(pc,4)/4；**multi_hit_mult**=1+0.08·min(mtp,3)。**`STAGE5_AUTHOR_STRUCTURE_AUDIT`** 控制 **`[Stage5 author structure audit]`** 宽表。 |
| **调用的表/知识图谱** | **内存**：score_map、term_map、term_role_map、term_confidence_map、term_uniqueness_map、paper_map、voc_to_clusters、all_vocab_vectors、vocab_to_idx（proximity、jd_semantic_gate）；**无直接 DB**。**works_to_authors**：accumulate_author_scores 用 papers 中的 wid、score、authors（含 pos_weight）。 |

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
- 看 **Stage2 锚点-候选证据表**（stage2_anchor_evidence_table）：edge_affinity、**conditioned_anchor_align**、**multi_anchor_support**、jd_align、hierarchy_consistency、neighborhood_consistency、isolation_penalty、primary_score，判断为何某候选被选/被压（无词表，纯结构量）  
- 看 Stage2A 的 primary_landings（数据驱动选主后）  
- 看 Stage2B 的 raw_candidates  
- 看 Stage3 的 score_map_terms 与 **paper_term_selection**（family 保送式）  
- 看 final_term_ids_for_paper（含 term_role / retrieval_role）  
- 看 Stage4 的 papers / authors  
- 看 Stage5 的 author_evidence_by_term_role  

**方便调试的打印汇总**（`LABEL_EXPANSION_DEBUG=True` 或 `verbose=True` 时）：  

| 阶段 | 打印内容 |
|------|----------|
| Stage2A | 每锚 `collect_landing_candidates` 的候选明细：tid \| term \| source \| semantic_score \| domain_fit \| **anchor_identity** \| **identity_gate** \| landing_score \| jd_align（含 primary_cap=secondary_only 标记）；候选上挂 **hierarchy_reason** / **hierarchy_score** / **hierarchy_level** |
| Stage2A | **【Stage2A 保留（层级未命中）】**：当 reason=topic_hierarchy_no_match 时仍保留进池，会打印 vid \| term \| sim \| reason，便于与「领域过滤剔除」区分 |
| Stage2A | **【Stage2A Identity Gate】**：anchor \| candidate \| source \| identity \| gate \| base \| final（前 30 条） |
| Stage2 | **【Stage2 锚点-候选证据表】**：anchor \| candidate(tid) \| edge_affinity \| cond_align \| jd_align \| hierarchy_cons \| neighborhood_cons \| isolation_penalty \| primary_score（前 40 条） |
| Stage2 | 每锚「数据驱动 primary 数」及无 primary 时跳过原因 |
| label_debug_cli | 深度诊断中 **【Stage2 锚点-候选证据表】** 再次打印（前 35 条），便于与最终 primary 对照 |  

**Stage4 多 term 合并断点日志（定位“多 term 为何未并到同一 wid”）**

- **断点 1（cap 前是否存在交集）**：`[Stage4 overlap before local-cap]`  
  两两输出 term 交集 `overlap` 与样本 pid；若这里已有交集，说明“多 term 潜力”在 local cap 前真实存在。
- **断点 2（交集生存性）**：`[Stage4 overlap survival audit]`（**默认关闭**；需时将 **`STAGE4_OVERLAP_SURVIVAL_MAX_LINES`** 设为正整数）  
  对交集 pid 输出 `rank_before_cap/score/survive_after_cap`；用于判断是否被 per-term local cap 提前砍掉。默认总行长上限 **`STAGE4_OVERLAP_SURVIVAL_MAX_LINES`**（优先 **`survive_after_cap=true`** 的行）。
- **断点 3（wid 写入行为）**：`[Stage4 paper_map write]`  
  **默认关闭**（`STAGE4_PAPER_MAP_WRITE_VERBOSE=False`）；逐 wid 排查时置 **`True`**（`stage4_paper_recall.py`）。字段：`incoming_term/old_hit_count/new_hit_count/old_terms/new_terms`。
- **断点 4（paper->author 传递）**：`[Stage4 merged wid multi-hit detail]` 与 **`[Stage4 author payload multi-hit detail]`**（由 **`STAGE4_AUTHOR_PAYLOAD_AUDIT_VERBOSE`** 控制是否逐条打印；**默认 False**，仅汇总 **`author_payload_multi_hit_papers=`**）  
  前者看 merged paper_map 的多 hit，后者看下发给 author payload 后多 hit 是否仍保留。
- **统计辅助 1**：`[Stage4 term stats]`  
  每个 term 打印 `raw/after_grounding/before_cap/after_cap`，快速判断是 grounding 过严还是 cap 过严。
- **`[Stage4 support grounding audit]`**：仅 **`paper_support`**，在 **local cap 后** 打 **`kept_total`**、**`tg_lt_0.10/0.18/0.28`**（累计： grounding 低于阈值的篇数）、**`secondary_hits`**，判断 support 是否靠弱 tg+高 JD 混池。
- **审计（gating 前评分行）**：`[Stage4 pre-gate paper score audit]` 中 **`pre_gate_scored_rows`** = 进入双阈值门控前的 scored 行数，**不是** cap 后 **kept**。
- **审计（JD 三级画像是否为空）**：`[Stage4 jd-topic-profile audit]`  
  打印各层 **key 数量**与 **top keys**，用于确认 Stage1 jd_profile 是否接入、是否与词表 id 同键空间。
- **审计（词表三级元数据是否命中）**：`[Stage4 topic-meta coverage audit]`  
  每参与 term：**topic_meta_found**、**source**（direct/cooc 等）、**has_*_id / has_*_dist**。
- **审计（hierarchy bonus 分布）**：`[Stage4 hierarchy bonus distribution]`  
  **candidate 篇数**；**hierarchy_consensus** 与 **hierarchy_bonus** 的 **min / p25 / p50 / p75 / max**（**默认**即到此，避免与 **`[Stage4 hierarchy consensus audit]`** 重复刷屏）。**top boosted / top penalized** 逐行表需设环境变量 **`STAGE4_HIERARCHY_BONUS_DISTRIBUTION_DETAIL=1`**。
- **审计（按 term 组的 hierarchy 解释）**：`[Stage4 hierarchy bonus by term-group audit]`（**`STAGE4_HIERARCHY_TERM_GROUP_AUDIT=0`** 可关）  
  每 **vid**：**paper_select_lane_tier**、wid 池内 **papers** 数、**mean_consensus**、**mean_bonus** vs **mean_bonus_raw**（**raw>mean** 表示 wid 级正向被 **strong_main_axis_core 质量门** 截断）、**parent_anchor / pa_sc / rk**。**环境变量**：**`STAGE4_HIERARCHY_BONUS_POSITIVE_MAIN_AXIS_ONLY`**（默认 1）、**`STAGE4_HIERARCHY_STRONG_AXIS_WEIGHT_FRAC`**（默认 **0.20**，即 hit 的 **Σ(term_score×idf)** 中 **strong_main_axis_core** 至少占此比例才允许 **bonus>1**）。
- **审计（wid 三级共识）**：`[Stage4 hierarchy consensus audit]`  
  两节：**按乘 bonus 前 paper_score 前 10**、**按绝对分差 abs(score_after−score_before) 前 10**；字段含 **field_cons/subfield_cons/topic_cons**、**median_consensus**、**delta_vs_median**、**hierarchy_bonus**（及 detail 内 **hierarchy_bonus_raw**、**hierarchy_tier_boost_gate**、**hierarchy_strong_axis_mass_frac** 等）、**paper_score_before/after**。
- **统计辅助 2**：`[Stage4 multi-hit potential before cap]`  
  汇总 cap 前 `pid -> terms` 的多 term 潜力样本，便于和 Stage5 的 multi-hit 审计对照。

**判读规则（最短链路）**

- cap 前有 overlap，但 `paper_map write` 第二次写入不存在：优先排查 local cap/截断顺序。  
- 第二次写入存在，但 `new_hit_count` 不增长：优先排查 `merge_hits` 或 hit 结构。  
- merged wid 有 multi-hit，但 author payload 无 multi-hit：排查 paper->author 下发覆盖问题。

**标签路分区 Debug 打印（6 组）**

由 `label_debug.py` 的 **DEBUG_LABEL_PIPELINE**、**DEBUG_LABEL_PIPELINE_LEVEL** 与 **debug_print(level, msg, label_or_recall)** 控制，便于一眼判断故障落在哪层。**LEVEL**：0=不打印；1=只打印汇总；2=汇总+top 明细；3=汇总+top+rejected/borderline/risky。`verbose=True` 时视为 level≥1 均打印。

| 组 | 内容 | 主要打印块 |
|----|------|------------|
| **1. Step2 锚点** | 为何入选/落选 | `[Step2]` 分区头；**`[Stage1-Step2 anchor collapse audit]`**：anchor、bb、四特征、**support_mean**、final、**collapse_ratio**、rank（解释「主线锚被压穿」 vs 「RL 仍高」）；`[Step2 Anchor Score Breakdown]`（含 smean/cratio）；`[Step2 Borderline Rejected]`；`[Step2 Final Anchors]` |
| **2. 条件化锚点** | 上下文与权重 | `[Anchor Context]` anchor、local_phrases、co_anchor_terms、weights（anchor/local/co/jd） |
| **3. Stage2A** | 候选保留/淘汰、primary 分解、多锚共识 | `[Stage2A Neighbor Compare]` raw_top / conditioned_top；`[Stage2A Primary Score Breakdown]` edge \| cond_align \| jd \| hier \| multi_anchor \| neigh \| poly_risk \| isolation \| final；`[Stage2A Cross-Anchor Evidence]` term \| support_count \| support_weight_sum \| anchors |
| **4. Stage2B** | 为何没启动、扩散产出 | `[Stage2B Seed Eligibility]` term \| identity \| domain_fit \| domain_span \| eligible；`[Stage2B]` 高可信 seed 数、seed_terms；`[Stage2B Expansion Summary]` dense_kept \| cluster_kept \| cooc_kept |
| **5. Stage3** | 全局复审与风险分桶 | `[Stage3]` 分区头；**`[Stage3 Entry Group]`** term \| stage3_entry_group \| term_role \| role_in_anchor \| can_expand \| source_type；**`[Stage3 Admission]`** term \| group \| hard_drop \| reason \| risk_flags；**`[Stage3 Scoring]`** term \| final_score \| identity_factor \| family_centrality \| path_topic_consistency \| generic_penalty \| object_like_penalty \| bucket；`[Stage3 Buckets]` / **`[Stage3 Bucket Details]`**；`[Stage3 Risky Term Reasons]`（LEVEL≥3） |

**结构信号与 family 保送（无领域词硬编码）**

标签路选词与打分**不依赖具体学科词**（如 control / planning / robotics 等词面），只依赖：**来源稳定性**（multi_source_support）、**上下文与层级一致性**（domain/subfield/topic fit）、**family 结构**（family_key、parent_primary、cluster_id）、**检索角色**（retrieval_role：paper_primary / paper_support / blocked）。  

- **hierarchy_guard**：`build_family_key(rec)`、`get_retrieval_role_from_term_role`、`compute_multi_source_support(rec)`、`score_term_record`（primary-like 的 path_topic 保底 0.45）、`should_drop_term`（仅对非 primary-like）；**term_scoring**：`passes_topic_consistency` 只拦 support-like，primary-like 默认通过；`allow_primary_to_expand`（只看 multi_source_support、subfield_fit、topic_fit、outside_subfield_mass）。  
- **Stage3**：幸存词打 **stage3_bucket**（**观测**）与 **retrieval_role**；**select_terms_for_paper_recall**：**paper 结构门**（含 **`risky_side_block`**、**JD 主轴门：core 父锚 `rk∨a_sc`**、**`[Stage3 paper gate summary]`**）→ **`paper_select_score`** + **family** + **dynamic_floor**，上限 **PAPER_RECALL_MAX_TERMS**；**未命中 side/expansion risky 拦截的 risky** 仍可入选。  
- **Stage4**：**get_term_role_weight**：paper_primary=1.0、paper_support=0.7、其他=0.4；在 `base_term_contrib` 之上叠加 **grounding/off_topic**，并新增 **term-type 因子**（`term_grounding_factor`、`term_paper_role_factor`）与 **动态 local cap**：方法骨架词放宽、对象词收紧，缓解 `robotic arm` 作者榜偏科并降低 `route planning` 全灭概率。  
- **Stage5**：**CoverageBonus** = 1 + 0.10×min(family_count, 5)；**FamilyBalancePenalty** = 1/(1 + 0.6×max_family_share)；作者分再乘二者（**term_family_keys** 来自 debug_1）。  

#### 3.3 label_means 子模块职责（在标签路中的角色）

| 模块 | 职责 | 主要使用阶段 |
|------|------|--------------|
| **infra** | Neo4j、Job/Vocab Faiss、vocab 向量与 voc_id↔idx、vocab_stats.db、簇成员与簇中心 | 全阶段 |
| **label_anchors** | 岗位技能清洗（clean_job_skills）、锚点提取（**extract_anchor_skills** 写 **anchor_source=skill_direct、anchor_source_weight=1.0**）、**JD 向量补充锚点**（**supplement_anchors_from_jd_vector** 写 **anchor_source=jd_vector_supplement、anchor_source_weight=ANCHOR_SOURCE_WEIGHT_JD_SUPPLEMENT**） | Stage1 |
| **hierarchy_guard** | 分布/纯度/熵、层级 fit、泛词惩罚；**结构信号**：build_family_key、get_retrieval_role_from_term_role；Stage3 **should_drop_term** 仅对非 primary-like；**score_term_record** 为 raw 主分（再乘 identity_factor、risk_penalty），含 backbone_boost、object_like_penalty、bonus_term_penalty，path_topic 保底 0.45。allow_primary_to_expand 不再被 Stage2B seed 决策调用。 | Stage2A / Stage2B / Stage3 |
| **label_expansion** | 学术落点与学术侧补充；**_compute_path_match** 加权平均；**compute_hierarchy_evidence** 含 topic_source、effective_*；**check_primary_admission** 三档 normal/weak_retain/reject；**compute_primary_score** base+hierarchy 加分、weak_retain×0.85；**choose_better_term_with_hierarchy** 含 head_term_bonus；**check_seed_eligibility** 无 fallback、**is_primary_expandable**、seed_expand_factor；**expand_from_vocab_dense_neighbors** 过 support_expandable_for_anchor、sim≥0.55、每 seed 2/3 条；**expand_from_cluster_members** 当前全关；**expand_from_cooccurrence_support** 仅强 seed、过 support_expandable_for_anchor、每 seed 最多 2 条；merge 时写 **retain_mode、topic_source、seed_blocked、seed_block_reason**。 | Stage2A / Stage2B |
| **term_scoring** | 词级最终权重（calculate_final_weights）、IDF/纯度/语义守门/source_credibility；**compute_identity_factor**（Stage3 软因子，不硬杀）；与 hierarchy_guard.score_term_record 配合 | Stage3 |
| **paper_scoring** | 论文贡献度（compute_contribution）：撤稿、领域纯度、标签累加（按 term_role 加权）、综述降权、紧密度、时序与署名；compute_primary_term_coverage（primary/supporting 计数供护栏 5） | Stage5 |
| **simple_factors** | survey_decay_factor、coverage_norm_factor、paper_cluster_bonus、paper_jd_semantic_gate_factor 等论文侧因子 | paper_scoring 与词/论文打分 |
| **advanced_metrics** | 共现/共鸣相关（term_resonance、cooc_span_penalty、cooc_purity_bonus 等，部分当前为占位 1.0） | Stage2/Stage3 词权重 |
| **base** | 公共基类或工具 | 按需 |
| **label_debug_cli** | 调试 CLI，与 RecallDebugInfo 配合；**【Top term 最终贡献表】**按 **`final_term_ids_for_paper`** 拆成 **「实际进论文检索」** 与 **「高分但未进论文检索」** 两张，避免 robotic hand / motion controller 等看起来像有效下游贡献 | 开发与排查 |

#### 3.4 调参与排查提示

- **锚点太少/太多**：看 Stage1 的 ANCHOR_*、JD_VOCAB_TOP_K、ANCHOR_MELT_COV_J；以及 DomainDetector 的 Job Top-K、active_domains 数量。  
- **学术词偏泛或偏窄**：看 Stage2 的 SIMILAR_TO_TOP_K/MIN_SCORE、vocabulary_domain_stats 的领域过滤；Stage3 的 SEMANTIC_POWER、ANCHOR_TERM_SIM_MIN、ANCHOR_BASE/ANCHOR_GAIN。  
- **论文/作者分数异常**：看 Stage4 的 MELT_RATIO、domain 软奖励、term_scores、TERM_MAX_PAPERS；Stage5 的 paper_scoring 各因子（综述降权、时序、署名）、AUTHOR_BEST_PAPER_MIN_RATIO；RecallDebugInfo 里各阶段统计与 last_debug_info。  
- **Stage2A 落点偏移**：优先看 anchor_type、identity_score、domain_fit、本锚点 primary 数量；不要先盯作者榜。  
- **Stage2B 噪声扩散**：优先看 primary_high_confidence 的筛选结果、domain_span、topic_align、各扩展来源数量（dense / cluster / cooc）。  
- **Stage3 全部滤空或几乎全过**：已改为**全局复审层**，trusted_primary **不再被 identity 硬杀**。看 **classify_stage3_entry_groups**（trusted_primary/secondary_primary/support_expansion）与 **check_stage3_admission**（仅 secondary/support 在弱证据或高 object+generic 时 hard_drop）；**identity 为软因子**（compute_identity_factor）参与 final_score。可开 STAGE3_DETAIL_DEBUG 与三张表（Entry Group / Admission / Scoring）看 entry_group、hard_drop、reason、risk_flags、identity_factor、risk_penalty、bucket；**Stage3 Buckets** 重点看 core 是否含 motion control、robot control、reinforcement learning 等 2A 主词。

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
| **PAPER_RECALL_MAX_TERMS**（stage3_term_filtering） | 12 | 参与论文检索的最大词数（**weak support 门**通过后：**统一分**排序 + family 去重 + dynamic floor 截断） |
| **PAPER_RECALL_DYNAMIC_FLOOR_REL / ABS**（stage3_term_filtering） | 0.62 / 0.12 | paper 入稿相对 Top1 与绝对下限：`max(ABS, top×REL)` |
| **STAGE3_PAPER_SUPPORT_BLOCK_ENABLED** 等（stage3_term_filtering） | 默认 True / 1 / 1 / 0.62 / 0.30 / True | **Paper bridge 弱 support 结构门**：`MAX_ANCHOR`/`MAX_MAINLINE`、**MIN_GROUNDING**、**MIN_FINAL_SCORE**、**BLOCK_COND_ONLY** |
| **SUPPORT_TO_PAPER_*** / **PAPER_SUPPORT_*_FACTOR** | 默认 True / 3 / ⅓ / 0.72 / 0.58 | **contamination 路径软放行**：**seed** 与 **keep**（需 **mainline_candidate∨ml∨扩**）进 **`support_pool`**，**readiness×因子**；最终 **support** ≤ **min(3, ⌊max_terms/3⌋)** |
| **STAGE3_PRIMARY_BUCKET_MERGE_PRIORITY** / **STAGE3_TERM_ROLE_MERGE_PRIORITY** / **STAGE3_DUPLICATE_MERGE_AUDIT**（stage3_term_filtering） | 见源码 dict / 默认 True | **tid 合并**：多来源时 `primary_bucket` / `term_role` 的优先级表；合并窄表开关（`[Stage3 duplicate merge audit]`） |
| **STAGE3_PAPER_RISKY_SIDE_BLOCK_ENABLED** / **STAGE3_PAPER_RISKY_SIDE_TERM_ROLES**（stage3_term_filtering） | 默认 True / 见 frozenset | **Paper risky_side_block**：`risky` ∧ ¬扩 ∧ `mainline_hits≤0` ∧ `term_role` 为 side 或 expansion 时不进 `final_term_ids_for_paper` |
| **STAGE3_EXPANDABLE_CORE_IDENTITY_HARD** / **STAGE3_EXPANDABLE_CORE_JD_ALIGN_MIN**（stage3_term_filtering） | 0.95 / 0.50 | **`primary_expandable`→`core`**：identity 硬杠与 **JD 结构化升舱**（非 conditioned_only 时） |
| **STAGE3_CORE_MISS_AUDIT**（stage3_term_filtering） | 默认 True | **`[Stage3 core-miss audit]`** 开关 |
| **STAGE5_AUTHOR_STRUCTURE_AUDIT**（stage5_author_rank） | 默认 True | **`[Stage5 author structure audit]`** 宽表：base_score、st/pc/mtp、struct_f 与 t/p/m mult、struct_tot、after |
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
对 JD **只编码一次**得到统一 query_vec；**并行**跑向量路与标签路，再用两路 Top 100 做种子跑协同路；将三路结果**构建为统一候选池**（CandidateRecord + CandidatePool），经**合并 → 打分 → 特征补全（enrich）→ 硬过滤 → 分桶**后导出，供**精排与训练**直接使用。  

**与精排的关系**：**总召回只负责构建候选池**，不负责最终排序；**KGAT-AX 精排只作用于该候选池内部**，对池内作者做深度重排。二者是**上下游关系**（总召回 → 候选池 → KGAT-AX 精排），不是并列的两个评分器。同时保留 RRF 融合与 rank_map 以兼容现有调用。

| 维度 | 说明 |
|------|------|
| **输入** | JD 文本、可选 domain_id、训练模式开关；三路配额 K_vector / K_label / K_collab 可配置 |
| **依赖** | QueryEncoder、VectorPath、LabelRecallPath、CollaborativeRecallPath、candidate_pool（CandidateRecord/CandidatePool/PoolDebugSummary） |
| **输出** | **candidate_pool**（候选主表 + 证据明细 + pool_debug_summary）、兼容的 final_top_200、rank_map、总耗时 |
| **核心中间产物** | query_vec、三路 meta 列表、CandidateRecord 列表、RRF/candidate_pool_score、pool_debug_summary |

#### 5.1 统一候选池结构（candidate_pool.py）

**CandidateRecord**（一人一条，供训练与精排直接吃）

- **基础与来源**：author_id、author_name（可选）；from_vector、from_label、from_collab、path_count；vector_rank、label_rank、collab_rank；vector_score_raw、label_score_raw、collab_score_raw；rrf_score、multi_path_bonus、candidate_pool_score、is_multi_path_hit。
- **作者静态指标**（KGAT-AX / 训练用，第一版可 None）：h_index、works_count、cited_by_count、recent_works_count、recent_citations、institution_level、top_work_quality。
- **query-author 交叉**（精排用，第一版可粗算）：topic_similarity、skill_coverage_ratio、domain_consistency、paper_hit_strength、recent_activity_match。
- **候选池辅助标记**：bucket_type（A/B/C/D）、passed_hard_filter、dominant_recall_path、hard_filter_reasons、bucket_reasons；vector_evidence、label_evidence、collab_evidence。

**CandidatePool**（显式三块）

- **candidate_records**：候选主表，一人一条 CandidateRecord。
- **candidate_evidence_rows**：候选证据明细表，一人多条（召回路径、skill/term/paper/collab 等）。
- **pool_debug_summary**：三路原始召回数、配额截断后数量、去重前/去重后数量、被硬过滤人数、各桶人数、最终送入精排人数；调参与排查用。

#### 5.2 总召回流程（八阶段）

1. **统一编码与领域探测**：raw_vec、query_vec、active_domains、applied_domain_str、vector_domains（沿用现有）。
2. **三路独立召回**：Vector/Label/Collab 返回 (author_meta_list, duration)；meta 含 author_id、*_score_raw、*_rank、*_evidence（可为 None）。
3. **build_candidate_records(v_meta, l_meta, c_meta)**：去重、合并三路、生成 CandidateRecord，填 rank/path/raw_score/evidence；更新 pool_debug_summary 去重前后数量。
4. **score_candidate_pool(records)**：算 RRF、multi_path_bonus、candidate_pool_score，排序。
5. **_enrich_candidate_features(records, active_domains, query_text)**：填作者静态指标、query-author 交叉特征、dominant_recall_path 等；训练/精排/解释共用，避免多处再查。
6. **_apply_hard_filters(records, ...)**：**第一层（轻）**：仅协同路且 path_count==1 且无 label/vector → 过滤；无论文无指标 → 过滤。**第二层（强）**：领域不交、活跃度过低等，待作者统计接好后再开。写 hard_filter_reasons、pool_debug_summary.hard_filtered_count。
7. **_assign_buckets(records)**：第一版规则——A：from_label 且 (path_count>=2 或 from_vector)；B：from_label 且非 A；C：from_vector 且非 A/B；D：from_collab 且非 A/B/C。写 bucket_type、bucket_reasons、pool_debug_summary 各桶人数。
8. **组装 CandidatePool 并返回**：candidate_records、candidate_evidence_rows、pool_debug_summary；兼容 final_top_200、rank_map。

#### 5.3 融合拆分为两函数

- **build_candidate_records(v_meta, l_meta, c_meta)**：只做去重、合并、生成 Record 并填 rank/path/score/evidence；不计算 RRF。
- **score_candidate_pool(records)**：只做 RRF、multi_path_bonus、candidate_pool_score 与排序。后续硬过滤、分桶、特征补全、导出训练样本均基于 records 列表，逻辑清晰。

**使用的方法（与现有一致的部分）**

1. **领域与 Query 预处理**  
   若传入 domain_id，则 processed_domain 非空；Query 扩展：若 processed_domain 有效且非训练模式，在 JD 后追加 `" | Area: {DOMAIN_PROMPTS[domain_id]}"`。

2. **统一编码**  
   query_vec = self.encoder.encode(final_query)，faiss.normalize_L2(query_vec)；向量路、标签路共用。

3. **并行召回**  
   ThreadPoolExecutor 调三路；seeds = 向量路与标签路 Top100 并集，再调协同路。三路建议返回 (author_meta_list, duration)，便于 build_candidate_records 直接填 score_raw 与 evidence；若仍返回 (id_list, duration)，总控内转换为 meta 再进 build_candidate_records。

---

### 6. 小结表

| 模块 | 核心目的 | 主要方法 |
|------|----------|----------|
| **文本转向量** | JD → 与索引一致的语义向量，并强化核心技能词 | 动态词库、自共振增强、OpenVINO SBERT、mean pooling、L2 归一化 |
| **向量路** | 按「与 JD 最像的论文」找作者 | 摘要 Faiss 检索、领域硬过滤、论文序→作者序映射 |
| **标签路** | 按「岗位技能→学术词→论文→作者」+ 多维度打分与层级守卫 | Stage1 领域与锚点（3% 熔断、JD 补充、**jd_profile** 与锚点 local_context/phrase_context）→ Stage2A 学术落点（SIMILAR_TO，有 jd_profile 时层级 fit 与 landing 打分、硬门槛；**Stage2A 候选明细调试打印**）→ Stage2B 学术侧补充（**allow_primary_to_expand 放宽**：identity≥0.70、domain_fit≥0.50、source∈可信，raw_candidates 带 subfield_fit/topic_fit/cluster_id/main_subfield_match 等）→ Stage3 **按 tid 去重聚合** + **classify_stage3_entry_groups** + **check_stage3_admission** + **第一段分×role_factor** + **assign_stage3_bucket** + **`stage3_build_score_map`（统一连续分）** + **`select_terms_for_paper_recall`（core 父锚 rk∨a_sc、`[Stage3 paper gate summary]`、paper_select_score+family+floor）** + 调试表 → Stage4 论文二层召回（领域软奖励、per-term 限流、MELT_RATIO）→ Stage5 作者排序（CoverageBonus 等预留）；依赖 `label_means`（含 **hierarchy_guard**）、`label_pipeline`；基础设施由 `label_means.infra` 管理 |
| **协同路** | 由种子作者扩展合作者 | 种子=V+L Top100、协作表双向查询、score 聚合、按总分排序 |
| **总控** | 单次编码、三路并行、统一候选池 | Query 领域扩展、build_candidate_records → score_candidate_pool → enrich → 硬过滤 → 分桶；RRF + multi_path_bonus；路径权重 2.0/3.0/1.0 |

整体上：**向量路**偏语义相似度，**标签路**偏技能/概念与图谱结构，**协同路**偏合作网络；三路在总控里用 RRF 合成一份作者列表，**形成统一候选池**，再交给 **KGAT-AX 精排**（仅对候选池内作者重排）与解释模块使用。

#### 5.2.1 标签路冻结后总召回与 KGAT-AX 最小改造（已落地）

在「标签路逻辑冻结、不默认信标签路」的前提下，对总召回与 KGAT-AX 做最小改造，使候选池多路一致优先、精排能利用召回来源与作者/交叉特征，训练样本基于多特征一致性而非单纯标签路命中。

**思路概要**

- **总召回**：合并三路后从 `label_evidence` 提摘要（core/support/risky 词数、best_term_score），RRF + **pair_path_bonus**（vector+label 最高、vector+collab/label+collab 次之）+ **label_hint**（core 微加、risky 微减）；**分桶 A～F**（A=vector+label, B=vector+collab, C=vector_only, D=label+collab, E=label_only, F=collab_only）；**分桶配额截断**再取 top N；**硬过滤**增加 label_only 弱命中规则（domain_consistency/paper_hit_strength/recent_activity_match/risky_terms）；**enrich** 补全作者静态与 query-author 交叉特征并产出 **kgatax_sidecar_rows** 供精排与训练使用。
- **KGAT-AX 推理**：`RankScorer` 从 `CandidateRecord` 构建 recall/author_aux/interaction 三路特征，调用 `model.calc_score_v2` 做四塔融合；提供 `rerank_from_candidate_pool(real_job_ids, candidate_records)` 返回按精排分降序的 `(author_id, score, rec)` 列表。
- **KGAT-AX 训练**：`KGATAXTrainingGenerator` 的 `_classify_record` 按「多特征一致性」分配训练标签（vector+label 高质量→strong_pos，label only 且 topic/domain/activity 弱→hard_neg，vector only 但整体强→weak_pos）；`_export_four_branch_row` 导出 13 维 recall、12 维 author_aux、8 维 interaction（含标签路摘要）；`trainer` 在 CF 阶段若有四分支侧车则用 `calc_score_v2` + BPR 损失。

**CandidateRecord 新增/扩展字段**

- 路径与融合：`pair_path_bonus`、`dominant_recall_path`（由 `infer_dominant_path` 推断）。
- 标签路摘要：`label_term_count`、`label_core_term_count`、`label_support_term_count`、`label_risky_term_count`、`label_best_term_score`（由 `extract_terms_from_label_evidence` 从 `label_evidence` 解析）。
- 分桶：`bucket_type` 扩展为 A/B/C/D/E/F/Z，`PoolDebugSummary` 增加 `bucket_e_count`、`bucket_f_count`。

**6 个小函数（`src/core/recall/candidate_features.py`）**

供总召回 enrich、硬过滤与 KGAT-AX 侧车/精排复用：

| 函数 | 用途 |
|------|------|
| `extract_terms_from_label_evidence(label_evidence)` | 从标签证据解析词列表（含 bucket、score），用于填充 label_*_count、label_best_term_score |
| `calc_query_author_topic_similarity(query_vec, top_works)` | Query 与作者代表作主题相似度（可接 SBERT 扩展） |
| `calc_domain_consistency(active_domains, top_works)` | 作者代表作领域与 query 活跃领域一致性 |
| `calc_recent_activity_match(author_recent_stats, top_works)` | 近年活跃度与代表作的匹配 |
| `calc_skill_coverage_ratio(label_term_count, label_core_term_count, label_support_term_count)` | 标签路技能覆盖比 |
| `calc_paper_hit_strength(label_evidence, vector_evidence, top_works)` | 论文命中强度 |
| `build_kgatax_feature_row(candidate_record)` | 从一条 CandidateRecord 构建 KGAT-AX 四分支侧车一行（recall_features / author_aux / interaction_features） |
| `bucket_quota_truncate(records, quotas, top_n)` | 按桶配额截断后再按 candidate_pool_score 取前 top_n |
| `infer_dominant_path(rec)` | 根据 from_vector/from_label/from_collab 推断主导召回路径 |

**总召回流程改动要点**

- `build_candidate_records`：合并三路后对来自 label 的 record 用 `extract_terms_from_label_evidence` 填 label_* 摘要字段。
- `score_candidate_pool`：RRF + multi_path_bonus + **pair_path_bonus** + **label_hint**，`dominant_recall_path = infer_dominant_path(rec)`。
- `_enrich_candidate_features`：批量加载 `batch_load_author_stats_from_sqlite`、`batch_load_recent_author_stats`、`batch_load_top_works`，再对每条 record 调用上述 5 个 calc_* 与 `calc_top_work_quality`。
- `_apply_hard_filters`：保留原有 collab_only_no_topic、no_paper_no_metrics；新增 label_only 弱命中规则（domain_consistency&lt;0.25、paper_hit_strength&lt;0.15、recent_activity_match&lt;0.20、label_risky_term_count≥2 且 label_core_term_count==0）。
- `_assign_buckets`：分桶 A～F/Z 及对应 `bucket_reasons`，统计 `bucket_e_count`、`bucket_f_count`。
- `execute`：在 enrich、硬过滤、分桶后调用 `bucket_quota_truncate(records, BUCKET_QUOTAS, FINAL_POOL_TOP_N)`；构造 `kgatax_sidecar_rows = [build_kgatax_feature_row(r) for r in records]`；返回值增加 `kgatax_sidecar_rows`。

**KGAT-AX 推理与训练**

- **RankScorer**：`_build_recall_features_from_candidate_record`、`_build_author_aux_from_candidate_record`、`_build_interaction_features_from_candidate_record` 从 CandidateRecord 构建与模型维度一致的向量；`rerank_from_candidate_pool(real_job_ids, candidate_records)` 调用 `calc_score_v2` 返回按精排分降序的 `(author_id, score, rec)`。
- **KGATAXTrainingGenerator**：`_classify_record` 按多特征一致性分配 strong_pos / weak_pos / hard_neg / collab_neg / field_neg；`_export_four_branch_row` 对齐 13/12/8 维与 E/F 桶编码。
- **trainer**：CF 阶段若 `get_four_branch_for_batch` 返回四分支张量，则用 `calc_score_v2` 计算正负样本得分并做 BPR 损失，否则沿用 `calc_cf_loss`。

**最短落地顺序**

1. **第一步**：改 `total_recall.py`（build_candidate_records → score_candidate_pool → _enrich_candidate_features → _apply_hard_filters → _assign_buckets → execute，含分桶截断与 kgatax_sidecar_rows）。
2. **第二步**：改 KGAT-AX 推理（RankScorer 的 _build_* 与 rerank_from_candidate_pool；model.calc_score_v2 已支持四分支）。
3. **第三步**：改训练（KGATAXTrainingGenerator 的 _classify_record / _export_four_branch_row；DataLoader 已有 get_four_branch_for_batch；trainer 在 CF 阶段接入四分支 BPR）。

---

## 层级化领域守卫与自动负向领域屏蔽改造方案（仅修改标签路）

当前标签路的主要问题不是“候选完全不在目标大领域内”，而是：

1. **领域守卫颗粒度过粗**：只看一级 `domain`，导致“机器人控制”“网络安全强化学习”“管理控制”都可能在一级层面被误判为可接受。
2. **多义词学术落点错误**：如“运动学”“动力学”“控制”等工业语境中的核心词，容易被落到非目标学术方向。
3. **泛词扩散过强**：如 `reinforcement learning`、`optimization`、`control` 这类大词，覆盖大量子方向，容易靠论文规模与共现关系霸榜。
4. **错 primary 会继续扩散**：一旦 Stage2A 的 primary 落错，Stage2B 的 dense / cluster / cooc 扩展会把整片候选带偏。

本改造方案的目标是：**不改索引构建，只利用现有统计表，在标签路内部完成四层领域画像、自动负向领域屏蔽、多义词消歧、泛词抑制与扩展门控。**

---

### 1. 现有索引可直接利用的统计信息

在不修改索引构建代码的前提下，标签路可直接使用以下表：

#### 1.1 `vocabulary_topic_stats`

用于提供候选学术词的二三级领域画像：

* `field_id`
* `subfield_id`
* `topic_id`
* `field_dist`
* `subfield_dist`
* `topic_dist`

用途：候选词的主 field / subfield / topic；候选词与 JD 在 field / subfield / topic 层的分布重合；自动负向领域屏蔽；在线计算 purity / entropy。

#### 1.2 `vocabulary_domain_stats`

用于补充一级领域与泛词强度信息：`domain_dist`、`domain_span`、`work_count`。用途：一级领域兜底守卫；自动抑制跨域过宽的大词；自动抑制论文量极大的泛词。

#### 1.3 `vocabulary_cooc_domain_ratio`

用于辅助共现侧的领域纯度判断：某词的共现伙伴主要活在哪些领域。用途：当 topic/subfield 信息不完整时作为弱证据；在 cooc 扩展中辅助判断“这个词虽然和 seed 共现，但是否仍然主要活在 JD 外领域”。

#### 1.4 `vocabulary_cluster / cluster_members`

用于扩展阶段的簇控制：词所属概念簇、簇成员列表。用途：Stage2B 的 cluster 扩展；term family / group 去重；避免某一簇的泛词刷满前排。

---

### 2. 总体改造思路

本次改造将标签路升级为：

> **Stage1：构建 JD 四层领域画像与上下文锚点** → **Stage2A：带上下文的 academic landing（解决多义词落点）** → **Stage2B：带领域引力的扩展（放宽 allow_primary_to_expand，使靠谱 primary 能带出 support term）** → **Stage3：全局复审层**（去重聚合 + **classify_stage3_entry_groups** + **check_stage3_admission** + 第一段分×identity×risk×role + **assign_stage3_bucket** + **统一连续分** + **paper 选词按分排序**）→ **Stage4：论文层二次守卫 + 单词贡献限流** → **Stage5：作者层覆盖度与一致性排序**

核心原则：同时使用 domain / field / subfield / topic 四层；使用“分布重合 + 主层级匹配 + 外部领域质量”；只有高可信 primary 才能继续扩散；引入 purity / entropy / domain_span / family coverage 抑制泛词。

---

### 3. 改动文件与职责划分

| 阶段 | 文件 | 职责 |
|------|------|------|
| Stage1 | `stage1_domain_anchors.py` | 生成工业锚点、提取锚点局部上下文、构建 JD 四层领域画像 |
| Stage2 | `stage2_expansion.py`、`label_expansion.py` | Stage2A academic landing、Stage2B expansion、层级契合度与泛词惩罚、primary 扩散门控 |
| Stage3 | `stage3_term_filtering.py`、`term_scoring.py` | term 级过滤与重打分、family/cluster 去重、泛词抑制 |
| Stage4 | `stage4_paper_recall.py`、`paper_scoring.py` | 二层论文召回、领域软奖励、term_final_score 入 paper_score、per-term 限流、MELT_RATIO |
| Stage5 | `stage5_author_rank.py` | 作者层汇总、多锚点覆盖奖励、family 失衡惩罚、领域一致性排序 |

---

### 4. 推荐新增的辅助函数与公式（概要）

- **分布与统计**：`parse_json_dist`、`compute_purity`、`compute_entropy`、`compute_dist_overlap`、`compute_outside_mass`
- **层级 fit**：`compute_hierarchical_fit`（domain/field/subfield/topic fit、main_*_match、outside_*_mass）
- **泛词抑制**：`compute_generic_penalty(work_count, domain_span)`、`compute_external_penalty(fit_info)`、purity_bonus、entropy_penalty
- **Landing**：`score_landing_candidate`（BaseSemantic × ContextScore × HierarchyGate × PurityBonus × EntropyPenalty × ExternalPenalty × GenericPenalty）
- **扩展门控**：`allow_primary_to_expand(primary_record)`；`score_expansion_candidate` 含 FieldGravity
- **Stage3**：按 tid 去重聚合；`should_drop_term`（仅对非 primary-like）；**topic 闸门只对 support-like 严格**，primary-like 跳过；`score_term_record`（唯一主分，含 backbone_boost/object_like_penalty/bonus_term_penalty）；_collect_risky_reasons（weak_family_centrality、high_drift_risk、**weak_topic_fit_tail**）；_bucket_stage3_terms（**core 两条路**：强主干 final≥0.66 或 结构型 final≥0.62 且 ptc≥0.55 且 cross≥0.94；support final≥0.56 且无 high_drift_risk）；**_debug_print_stage3_bucket_details** 打印 bucket 判定依据；family 角色约束、按 final_score 排序 top_k；缺失 topic/subfield 时退化为 field/domain
- **Stage4**：`TERM_MAX_PAPERS`、单 term 对作者贡献上限、`score_paper_record` 含 PaperHierarchyFit
- **Stage5**：CoverageBonus、HierarchyConsistency、FamilyBalancePenalty

详细公式与工程落地顺序见仓库内标签路代码注释及 `label_means/hierarchy_guard.py` 实现。

**当前已落地（防断流与轻惩罚）**：① **Stage3** 改为「按 tid 去重聚合 → 轻硬过滤 → identity/topic gate（primary-like 跳过 topic）→ 唯一主分 score_term_record（含 **backbone_boost**、**object_like_penalty**、**bonus_term_penalty**）→ family 角色约束 → _collect_risky_reasons/_bucket_stage3_terms（core/support/risky）→ 按 final_score 排序 top_k（STAGE3_TOP_K=20）」；不再用 FINAL_MIN_TERM_SCORE 阈值淘汰。② **score_term_record**：base 权重 0.36/0.18/0.20/0.14/0.12；final=base×gate×cross×backbone_boost×object_like_penalty×bonus_term_penalty；backbone 轻推主轴、object 轻压 arm/manipulator/hand、bonus 轻压 RL/q-learning/medical robotics。③ **risky 理由**（更严，避免主干词误标）：weak_family_centrality、high_drift_risk（drift>0.75 且 ptc<0.30）、**weak_topic_fit_tail**（ptc<0.50 且 final<0.58）。④ **bucket**：**core 两条路**——路 A 强主干直通（paper_primary 且 final≥0.66），路 B 结构型主干（paper_primary 且 final≥0.62、ptc≥0.55、cross≥0.94）；support 为 final≥0.56 且无 high_drift_risk；其余 risky。⑤ **日志**：**[Stage3 Bucket Details]** 打印 term | bucket | final | ptc | cross | reasons。⑥ **调试**：Stage3 明细含 backbone_boost、object_like_penalty、bonus_term_penalty（STAGE3_DETAIL_DEBUG）。

---

## 索引构建详解

**本节回答：**
- 离线要建哪几类索引？各自给谁用（召回 / 精排 / 标签路）？
- 每类索引的数据从哪来、写到哪、config 里对应哪些路径？
- 索引的建立顺序与依赖关系是什么？设计上为何采用当前结构（如 HNSW、分片、log1p 等）？
- 想改索引结构或重建顺序时，应看哪一段？

本节说明四类离线索引的**建立方式**、**在系统中的作用**与**设计考量**，对应代码：`src/infrastructure/database/build_index/` 下的 `build_collaborative_index.py`、`build_feature_index.py`、`build_vector_index.py`、`build_vocab_stats_index.py`；路径与开关由 `config.py` 统一配置。

### 索引设计总览与建立顺序

**四类索引及其定位**：

| 索引类型 | 主要消费者 | 建立时机建议 | 数据依赖 |
|----------|------------|----------------|----------|
| 向量索引 | 向量路、标签路（领域/锚点/语义守门） | 需主库 works/abstracts/jobs/vocabulary 就绪；可与 KG 并行 | 主库 SQLite |
| 词汇统计索引 | 标签路（领域分布、共现、三级领域、概念簇） | 依赖主库 works + 可选 vocabulary_topic_index.json；**共现与 KG 的 HAS_TOPIC 同源，但不依赖 Neo4j** | 主库 + 可选 DATA_DIR 下 JSON |
| 协作索引 | 协同路召回 | 依赖 authorships + works + authors | 主库 SQLite |
| 特征索引 | KGAT-AX 精排 | 依赖 authors、institutions | 主库 SQLite |

**建议建立顺序**：  
1）主库数据就绪后，可先建**向量索引**（词汇/摘要/岗位），供标签路领域探测与向量路使用；2）**词汇统计索引**（含共现、领域分布、三级领域、概念簇）可独立于 Neo4j 运行，若需三级领域则先跑 `backfill_vocabulary_topic` 与 `export_vocabulary_topic_index`；3）**协作索引**与**特征索引**仅依赖主库表，无交叉依赖，顺序任意。KG 构建与上述索引可并行，仅语义桥接（SIMILAR_TO）会用到词汇向量索引（读 Faiss 做检索，不写）。

**索引在系统中的作用总览**：

- **向量索引**：向量路用摘要索引「JD 向量 → 相似论文 → 作者」；标签路用岗位索引「JD 向量 → 相似岗位」做领域/锚点，用词汇索引做学术词语义守门（cos 相似度）；与 `input_to_vector.py` 共用同一 SBERT 与 L2 归一化，保证空间一致。  
- **词汇统计索引**：标签路用 `vocabulary_domain_stats` 做领域纯度、熔断、IDF；用 `vocabulary_cooccurrence` 做学术共鸣、锚点共鸣、cooc_span/cooc_purity；用 `vocabulary_topic_stats` 做三级领域对齐与 topic_align；用概念簇做簇扩展。  
- **协作索引**：协同路给定种子作者 ID 列表，查 `scholar_collaboration` 得 Top 合作者并聚合得分，与向量/标签路分数融合。  
- **特征索引**：精排加载 JSON 后按 author_id/inst_id 查表，作为 KGAT-AX 全息嵌入层输入，与图结构一起参与精排打分。

---

### 1. 协作索引（build_collaborative_index.py）

**作用**  
预先算出「作者–作者」协作强度，供**协同路召回**使用：给定种子作者（如向量路+标签路 Top100），能快速查出「和谁合作最紧密」的作者并排序，而不在线上现算。线上 `collaboration_path.py` 按种子作者 ID 批量查 `scholar_collaboration` 并聚合得分。

**建立方式**  
- **触发**：单独运行 `build_collaborative_index.py`（或脚本内调用 `LocalSimilarityIndexer` 三步骤）。  
- **输入**：主库 `authorships`、`works`、`authors`（work_id, author_id, pos_index, is_corresponding, is_alphabetical, year, h_index, citation_count）。  
- **输出**：`COLLAB_DB_PATH`（如 `data/build_index/scholar_collaboration.db`）内表 `scholar_collaboration(aid1, aid2, score)` 及双向覆盖索引。

**依赖配置（config.py）**  
- `DB_PATH`：主库路径  
- `COLLAB_DB_PATH`：协作库路径

**实现与设计要点**

- **Step 1：单人贡献权重**  
  从 `authorships` + `works` + `authors` 取上述字段。单篇贡献：`WeightStrategy.calculate(pos_index, is_corr, is_alpha, pub_year, base_year)` → 基数（第一作者 +0.2、通讯 +0.2）× 时间衰减 `e^(-0.1*Δt)`，再乘引用因子 `sqrt(ln(cite + e))`，得到 `composite_weight`。写入协作库表 `weighted_authorships(work_id, author_id, weight, h_index)`，并建 `work_id` 索引。**设计用意**：与 KG 的 AUTHORED 权重一致（署名+时间），另加引用因子使高影响力论文的合作关系权更大。

- **Step 2：直接协作分 S_direct**  
  只保留 2–99 人的论文（排除单人、排除超大合作，降噪）。对每篇论文内作者两两配对 `(a1,a2)`，分数 `score = w1 * w2`（两人在该篇的权相乘）；同一对 `(a1,a2)` 在多篇论文中累加（`ON CONFLICT DO UPDATE s_val += excluded.s_val`）。结果存 `direct_scores(aid1, aid2, s_val, h1, h2)`，主键 `(aid1, aid2)`，`aid1 < aid2` 保证唯一。

- **Step 3：合成最终索引 S_total**  
  先把 `direct_scores` 的 `(aid1, aid2, s_val)` 拷到 `scholar_collaboration`。**间接协作（Bridge）**：对每个 aid1，找「d1.aid2 = d2.aid1」的 d2，即「A–B、B–C」推出「A–C」；间接分 = `0.3 * Σ(d1.s_val * d2.s_val) / ((h2+1)*sqrt(h1+1)*sqrt(h2+1))`，用 h_index 做正则化。按 `aid1` 分区，只保留每个作者 **Top-100** 合作者（`ROW_NUMBER() ... rank <= 100`）。**设计用意**：控制每作者出度，避免图过密；双向覆盖索引 `(aid1, aid2, score)` 与 `(aid2, aid1, score)` 支持「以 aid1 为种子找 aid2」和「以 aid2 为种子找 aid1」的对称查询。

**产出**  
表 `scholar_collaboration(aid1, aid2, score)` + 双向覆盖索引，供 `collaboration_path.py` 按种子作者批量查协作伙伴并聚合得分。

---

### 2. 特征索引（build_feature_index.py）

**作用**  
为 **KGAT-AX 精排**提供作者、机构的**数值特征**，作为「全息嵌入层」的输入；精排加载后按 `author_id` / `inst_id` 查表，与图结构一起参与打分，避免 h_index、引用量等长尾特征在原始尺度下压制模型对中层学者的区分度。

**建立方式**  
- **触发**：单独运行 `build_feature_index.py`（或脚本内调用 `FeatureIndexBuilder.build()`）。  
- **输入**：主库 `authors`（author_id, h_index, works_count, cited_by_count）、`institutions`（inst_id, works_count, cited_by_count）。  
- **输出**：单文件 `FEATURE_INDEX_PATH`（如 `data/build_index/feature_index.json`）。

**依赖配置**  
- `DB_PATH`：主库  
- `FEATURE_INDEX_PATH`：特征索引输出路径

**实现与设计要点**

- **作者特征**：SQL 取 `authors` 的 `author_id, h_index, works_count, cited_by_count`。对 `h_index, works_count, cited_by_count` 做 **log1p** 再 **Min-Max** 到 [0,1]（`_normalize`）。存成 `author_features[id] = {h_index, works_count, cited_by_count}`。

- **机构特征**：SQL 取 `institutions` 的 `inst_id, works_count, cited_by_count`。同样 log1p + Min-Max，存成 `inst_features[id] = {...}`。

- **持久化**：JSON 格式 `{"author": author_features, "institution": inst_features, "metadata": {version, scaling_method: "log1p_minmax", timestamp}}`，写入 `FEATURE_INDEX_PATH`。

**设计用意**：  
- **log1p**：缓解 h_index、cited_by_count 的长尾分布，使顶级学者与中层学者的数值差距缩小，模型更能学到「中等影响力但匹配岗位」的信号。  
- **Min-Max 到 [0,1]**：与图嵌入等其它输入尺度统一，便于 KGAT-AX 全息层融合。  
- **JSON 单文件**：精排启动时一次加载，按 ID 查表，无额外数据库依赖。

**产出**  
单个 JSON 文件，精排加载后按 `author_id` / `inst_id` 查表，输入 KGAT-AX 的特征层。

---

### 3. 向量索引（build_vector_index.py）

**作用**  
为**语义检索**提供 Faiss 索引，使线上只需一次向量计算即可在三个对象上做近似最近邻：  
- **摘要索引**：向量路用「JD 向量 → 最像的论文 → 作者」召回。  
- **岗位索引**：标签路用「JD 向量 → 相似岗位」推断领域、锚点技能。  
- **词汇索引**：标签路用「学术词向量」与 query 做语义守门（cos 相似度）；KG 语义桥接（SIMILAR_TO）也会用词汇向量做 Faiss 检索（建边时重算向量，不依赖本索引的 reconstruct）。  
与 `input_to_vector.py` 使用同一 SBERT 模型与 L2 归一化，保证向量空间一致。

**建立方式**  
- **触发**：单独运行 `build_vector_index.py`（`StableVectorGenerator.run_all()`，可按需注释某子任务）。  
- **输入**：主库 `vocabulary`、`jobs`、`abstracts`；可选 `data/industrial_abbr_expansion.json`（词汇缩写扩写）。  
- **输出**：`INDEX_DIR` 下 `vocabulary.faiss` + `vocabulary_mapping.json`、`job_description.faiss` + `job_description_mapping.json`、`abstract.faiss` + `abstract_mapping.json`；摘要另有 `shards/shard_*.npy` 与 `shard_*_ids.json` 中间文件。

**依赖配置（config.py）**  
- `DB_PATH`, `INDEX_DIR`, `SBERT_DIR`, `SBERT_MODEL_NAME`  
- `VOCAB_INDEX_PATH/MAP`, `ABSTRACT_INDEX_PATH/MAP`, `JOB_INDEX_PATH/MAP`

**实现与设计要点**

- **模型与预处理**：使用 `SentenceTransformer(MODEL_NAME, cache_folder=SBERT_DIR)`，`max_seq_length=1024`，CPU + MKL-DNN。长文本 `_smart_trim`：超 4000 字符取前 2000 + 后 2000，避免截断丢失尾部。**设计用意**：与线上 `input_to_vector.py` 同一模型与长度策略，保证 JD/摘要/词汇在同一向量空间可比。

- **索引结构**：维度 d 来自模型；`faiss.IndexHNSWFlat(d, 32, METRIC_INNER_PRODUCT)`，`efConstruction=200`；向量 **L2 归一化**，因此内积等价余弦。词汇/岗位：id 为整数或字符串（voc_id / securityId），用 `IndexIDMap` 包一层再 `add_with_ids`，id 写进 map 文件，便于线上由 Faiss 返回的内部 id 反查业务 id。摘要：id 为 work_id 字符串，写入 `abstract_mapping.json`。**设计用意**：HNSW 在召回率与速度之间折中；32 为 M 参数，200 为构建时 efConstruction，适合十万级向量。

- **三个子任务**：  
  1. **Vocabulary**：从 `vocabulary(voc_id, term, entity_type)` 读取。**用于建向量的文本**：  
     - **工业词（entity_type='industry'）**：先用 `tools.normalize_skill(term)` 清洗；若清洗后为空则**不进入索引**；否则用清洗后的词做 key 查 `industrial_abbr_expansion.json`，若有缩写则拼成 `清洗名 | abbr1 | abbr2`，否则用清洗名，对该文本做 SBERT 编码。  
     - **非工业词**：不清洗，用原始 `term` 查缩写表，有则 `term | abbr1 | abbr2`，否则 `term`，对该文本编码。  
     索引中只保留 **id**（voc_id），Faiss 使用 `IndexIDMap`；产出 `vocabulary.faiss` + `vocabulary_mapping.json`。  
  2. **Job**：`jobs(securityId, job_name, description)`，拼接 `job_name + description` 再 trim → `job_description.faiss` + `job_description_mapping.json`。  
  3. **Abstract**：`abstracts(work_id, full_text_en)`，**分片**：每 1 万条一个 shard（`SHARD_SIZE=10000`），先全量 `fetchall()` 到内存避免 OFFSET 慢查，按片 encode、保存 `shard_i.npy` 与 `shard_i_ids.json`，最后 `_merge_abstract_shards` 合并成一份 `abstract.faiss` + `abstract_mapping.json`。**设计用意**：摘要量大，分片可断点续跑、控制内存；合并后单索引便于线上一次加载。

**产出**  
`vocabulary.faiss` + mapping、`job_description.faiss` + mapping、`abstract.faiss` + mapping（及可选 `*_vectors.npy`）。向量路、标签路、输入编码都依赖这些索引与同一 SBERT 空间。

---

### 3.5 论文标题向量库（build_work_title_embeddings.py，标签路 Stage5 加速）

**作用**  
离线把主库 `works.title` 用与线上一致的 **`QueryEncoder.encode_batch`（共振 + L2）** 编成向量，写入独立 SQLite：`config.WORK_TITLE_EMB_DB_PATH`（默认 `data/build_index/work_title_embeddings.db`）。标签路 **Stage5** 对 JD 门控按 **`work_id` 查表**，缺失再在线 `encode`，显著减少 CPU 上对大标题集合的重复编码。

**延迟与 5s 目标**：单靠本脚本无法把整链压到 5s；S1 `anchor_ctx` / S5 逐篇打分仍是 CPU 大户。完整清单与改法优先级见 **`docs/LABEL_RECALL_LATENCY.md`**。

**建立方式**  
- 在项目根执行：  
  `python src/infrastructure/database/build_index/build_work_title_embeddings.py`  
  可选：`--limit N` 试跑前 N 条；`--out 路径` 覆盖输出库；**`--resume` 断点续跑**（跳过输出库中已有 `work_id`，`dim` 须与当前 SBERT 一致）。  
- **换 SBERT 目录或模型后须全量重建**。

**常驻服务（避免冷启动）**  
- `python src/core/recall/label_recall_service.py` 或  
  `uvicorn src.core.recall.label_recall_service:app --host 127.0.0.1 --port 8765`  
- `POST /recall` JSON：`query_text`、`domain_id`（默认 `"0"`）、可选 `verbose`。  
- 细粒度累计 `paper_scoring` 耗时：进程环境变量 `LABEL_PROFILE_STAGE5=1`。
- 试验关闭「标题↔JD」向量门控：`LABEL_NO_JD_TITLE_GATE=1`（或 `true`/`yes`/`on`）；`LabelRecallPath` 初始化时会打印一行确认；Stage5 亦不再查标题向量库或 batch 编码标题（省 CPU）。
- **交互式 CLI**（`python src/core/recall/label_path.py`）：启动时会询问  
  `标题↔JD 语义门控 (0=开启默认  1=关闭以加速) [0]:`，选 `1` 等价于设置上述环境变量（在创建 `LabelRecallPath` 之前生效）。

**共振词表 / 领域向量快照（减少冷启动耗时）**  
- **共振词表**：`config.HARDCORE_LEXICON_SNAPSHOT_PATH`（默认 `data/build_index/query_encoder_hardcore_lexicon.json`）。首次从主库 `jobs`+`vocabulary` 统计后写入；**主库文件 mtime 变化**自动失效并重建。  
- **领域向量**：`LABEL_DOMAIN_VECTORS_NPZ_PATH` + `LABEL_DOMAIN_VECTORS_META_PATH`（默认 `data/build_index/label_domain_vectors.npz` 与同目录 `_meta.json`）。首次对 17 个领域中文名 `encode` 后写入；**`DOMAIN_MAP` 或 SBERT 目录下 `config.json` 变更**会失效并重建。  
- **耗时预期**：快照命中时，冷启动一般可少掉 **约 0.5～3s（词表）+ 约 1～5s（17 条领域 encode，视 CPU 而定）**；**单次召回总耗时**仍主要由 S1 `anchor_ctx`、S5 `paper_contribution`、Neo4j 等决定，快照不直接缩短这些阶段。选 `1` 关门控可再省 Stage5 标题向量相关 CPU（与是否已建 `work_title_embeddings.db` 有关）。
- **领域向量 npz（Windows）**：经 **`open(..., "wb")` + `np.savez_compressed(fp, …)`** 写入；数组键为 **`dom_1`…`dom_17`**（旧文件仍为 `"1"`…`"17"`，加载兼容）。先写 `.tmp` 再 `os.replace`，失败则 **直写最终 npz** 并 `shutil.copy2` 兜底；写入后校验文件存在且非空。
- **S1 `anchor_ctx`**：对锚点/JD/local/co 字符串 **去重后 `encode_batch` 预填缓存**，再 `build_conditioned_anchor_representation`（与逐条 encode 等价，显著减少前向次数）。

---

### 4. 词汇统计索引（build_vocab_stats_index.py，领域分布 + 共现 + 三级领域 + 概念簇）

**作用**  
为**标签路召回**提供「每个学术词」的统计与结构信息：  
- **领域分布**：`work_count`、`domain_span`、`domain_dist`，用于领域纯度、熔断、IDF、domain_fit 等。  
- **共现**：词对在同一 Work 下的共现频次，供学术共鸣、锚点共鸣、cooc_span/cooc_purity；与 KG 的 HAS_TOPIC 同源，但**不写入 Neo4j**，避免大表自连接。  
- **三级领域**：field / subfield / topic 的直接标注或共现占比补全，供 topic_align、hierarchy_norm 等。  
- **概念簇**：词→簇、簇→成员，供标签路按簇扩展（cluster_expansion）；并产出 `cluster_centroids.npy` 等。  
**不依赖 Neo4j**，仅依赖主库 SQLite 与可选 `vocabulary_topic_index.json`。

**建立方式**  
- **触发**：单独运行 `build_vocab_stats_index.py`（或脚本内按步骤调用 `VocabStatsIndexer`）。  
- **输入**：主库 `works`（concepts_text, keywords_text, domain_ids）、`vocabulary`；可选 `DATA_DIR/vocabulary_topic_index.json`（由 `export_vocabulary_topic_index.py` 从主库 `vocabulary_topic` 导出）；词汇向量来自 `build_vector_index` 产出的 vocabulary 向量（概念簇步骤）。  
- **输出**：SQLite 库 `VOCAB_STATS_DB_PATH`（如 `data/build_index/vocab_stats.db`），内多张表（见下）；另可选 `cluster_centroids.npy` 等。

**依赖配置**  
- `DB_PATH`、`DATA_DIR`、`VOCAB_STATS_DB_PATH`；部分步骤用 `VOCAB_INDEX_PATH`、`VOCAB_MAP_PATH`、`INDEX_DIR`。**不依赖 Neo4j**。

**实现与设计要点**

- **领域分布（vocabulary_domain_stats）**：从主库 `works` 的 `concepts_text`、`keywords_text` 解析 term（与共现逻辑同源：strip+lower，按 `|;,\)` 拆分），与 `vocabulary` 映射后按 work 的 `domain_ids` 聚合，用 `Counter` 得到 `domain_dist`；`work_count`、`domain_span` 写入 `vocabulary_domain_stats`。**设计用意**：与 HAS_TOPIC 同源，保证「出现在多少篇、跨多少领域」与图谱语义一致；不依赖 Neo4j，可独立于 KG 构建。

- **领域占比（vocabulary_domain_ratio）**：由 `vocabulary_domain_stats.domain_dist` 展开为 (voc_id, domain_id, ratio)，便于查询时按「某领域占比≥阈值」筛词。

- **共现表（vocabulary_cooccurrence）**：从主库 works 的 concepts_text/keywords_text 流式构建词对共现频次，写入 (term_a, term_b, freq)；freq 至少为 2。标签路学术共鸣、锚点共鸣及 cooc_span/cooc_purity 均由此表提供；KG 流水线不再在 Neo4j 中构建 CO_OCCURRED_WITH，避免磁盘打满。**设计用意**：与 KG 原共现逻辑一致，但存于独立库，读写分离、可断点续传。

- **共现领域占比（vocabulary_cooc_domain_ratio）**：按 (voc_id, domain_id) 存「共现伙伴在该领域的占比」的 freq 加权均值，供标签路直接查 cooc_purity 等。

- **三级领域（vocabulary_topic_stats）**：先读 `DATA_DIR/vocabulary_topic_index.json`；对 JSON 中至少有一级（field_id 或 field_name）的 voc_id 直接写入 field/subfield/topic 标量，`source='direct'`。对**未在 JSON 中出现**的 voc_id，用 `vocabulary_cooccurrence` 中有标签的共现伙伴按 freq 加权聚合，得到 field_dist、subfield_dist、topic_dist，`source='cooc'`。对有标签但缺层级的词，用共现补全 *_dist 并设 `source='direct+cooc'`。**设计用意**：有 OpenAlex 层级则直接用，无则用共现伙伴的分布推断，保证标签路 topic_align/hierarchy 有据可查。

- **概念簇（vocabulary_cluster、cluster_members）**：依赖 build_vector_index 产出的词汇向量，对学术词做 K-Means（如 K=700），工业词按与簇中心相似度归属 Top-K 簇；产出 vocabulary_cluster、cluster_members 及 cluster_centroids.npy，供标签路 cluster_expansion。

**前置**：若需三级领域，须先运行 `backfill_vocabulary_topic` 回填主库 `vocabulary_topic`，再运行 `export_vocabulary_topic_index` 生成 `vocabulary_topic_index.json`，否则三级领域步骤会跳过。

**产出**  
SQLite 库 `vocab_stats.db` 内多张表：`vocabulary_domain_stats`（领域分布）；`vocabulary_domain_ratio`（领域占比）；`vocabulary_cooccurrence`（词对共现频次）；`vocabulary_cooc_domain_ratio`（共现领域占比）；`vocabulary_topic_stats`（三级领域）；`vocabulary_cluster`、`cluster_members`（概念簇）；以及 build_progress 等，详见脚本。

---

### 5. 与 config 的对应关系小结

| 索引 | 作用简述 | 建立方式概要 | config 中的路径/配置 |
|------|----------|----------------|------------------------|
| **协作索引** | 协同路：种子作者 → Top 合作者及得分 | 权重(署名+时间+引用)→直接协作→间接 Bridge→每作者 Top100→双向覆盖索引 | `DB_PATH`, `COLLAB_DB_PATH` |
| **特征索引** | KGAT-AX 精排：作者/机构数值特征输入 | log1p + Min-Max → JSON（author/inst） | `DB_PATH`, `FEATURE_INDEX_PATH` |
| **向量索引** | 向量路摘要检索；标签路岗位/词汇语义与领域探测 | 同一 SBERT、L2 归一化、HNSW 内积；词汇/岗位 IDMap；摘要分片再合并 | `DB_PATH`, `INDEX_DIR`, `SBERT_*`, `*_INDEX_PATH`, `*_MAP_PATH` |
| **词汇统计索引** | 标签路：领域分布、共现、三级领域、概念簇 | 主库 works→domain_stats/cooccurrence；JSON(topic_index)+共现→topic_stats；向量→概念簇（不依赖 Neo4j） | `VOCAB_STATS_DB_PATH`, `DB_PATH`, `DATA_DIR`, `VOCAB_INDEX_PATH`, `INDEX_DIR` |

**建立顺序建议**：向量索引、词汇统计索引可先建（仅依赖主库与可选 JSON）；协作索引与特征索引无交叉依赖，顺序任意。详见上文「索引设计总览与建立顺序」。

整体上：**协作索引**面向「谁和谁合作」；**特征索引**面向精排输入；**向量索引**面向三路召回里的语义与领域探测；**词汇统计索引**面向标签路里学术词的质量、领域约束与共现/簇扩展。

---

## KGAT-AX 模型详解（精简版）

**本节回答：** KGAT-AX 在系统里处于什么位置？输入输出是什么？与总召回如何衔接？当前版本边界在哪？

本节说明精排模块 `src/infrastructure/database/kgat_ax` 的**定位**、**输入输出**、**核心思路**与**与总召回的衔接**，写成当前可落地版本，不展开 v2/四分支/teacher 等超前规划。

---

### 1. 模型定位

KGAT-AX 在本系统中定位为**第二阶段候选池精排器**，而不是全库召回模型。系统先通过多路召回（Vector Recall、Label Path Recall、Collaboration Recall）构建候选作者池，再由 KGAT-AX 对候选作者进行深度重排序，输出最终推荐结果。

因此，KGAT-AX 的职责不是“从全体作者中直接找人”，而是：

* 接收总召回产生的候选作者集合；
* 利用知识图谱结构信息进一步区分候选作者；
* 结合作者学术特征对候选结果做精排；
* 为最终排序与解释模块提供模型分数。

可概括为：

`JD -> Total Recall -> Candidate Pool -> KGAT-AX Re-rank -> Final Ranking`

---

### 2. 输入与输出

#### 2.1 输入

KGAT-AX 的输入主要包括两部分：

**（1）图结构输入**  
来自岗位、作者、论文、学术词、技能词等节点及其关系构成的知识图谱，用于建模 Job 与 Author 之间的多跳关联。

**（2）候选作者输入**  
来自总召回阶段的候选作者列表。每个候选作者至少包含：

* `author_id`
* 候选来源信息（如是否来自 vector / label / collab）
* 基础召回分数或排序信息
* 可选的作者统计特征（如 works_count、cited_by_count、h_index）

#### 2.2 输出

KGAT-AX 输出每位候选作者的匹配分数 `kgat_score`，该分数用于候选池内部重排序，并在最终排序阶段与召回分数、规则稳定项共同融合。

---

### 3. 模型核心思路

KGAT-AX 的核心思想是：在候选池已经初步筛出“看起来相关”的作者后，再通过知识图谱传播机制建模更深层的结构关系，从而提升最终排序质量。

**其作用重点不是“找全”，而是“排准”。**

对于科技人才推荐场景，单纯依赖语义相似度容易引入以下问题：

* 只看文本相似，容易召回“语义接近但研究方向不完全对口”的作者；
* 只看作者指标，容易把高产高引但岗位不贴题的人排得过前；
* 只看协作网络，容易把同圈层但非目标方向的作者带进来。

KGAT-AX 的价值在于：将岗位需求、学术词、论文、作者等实体放入同一图结构中，通过图传播学习岗位与作者之间的多跳关联，再在候选池内部完成更细粒度的区分。

---

### 4. AX 的含义与使用方式

本项目中的 AX 表示对作者辅助特征的增强注入，主要包括作者的基础学术统计信息，例如：

* 论文数量
* 被引次数
* h-index
* 近期活跃度

这些特征的作用是：在候选作者都具备一定主题相关性的前提下，帮助模型进一步区分“谁更成熟、谁更活跃、谁更有持续产出能力”。

需要强调的是：

* AX 特征在本项目中是**辅助信号**；
* 它用于对图结构得分做**轻量增强**；
* 它不能替代岗位相关性，也不能压过主题匹配本身。

因此，KGAT-AX 仍应以岗位–作者的主题相关性为核心，学术指标只作为精排阶段的补充信息。

---

### 5. 训练目标

KGAT-AX 的训练仍保持较基础的双目标思路：

**5.1 协同过滤目标（CF）**  
学习 Job 与 Author 之间的匹配偏好，使模型能够区分更相关与更不相关的候选作者。

**5.2 知识图谱目标（KG）**  
学习图谱中实体与关系的结构信息，使模型在多跳关联传播时获得更好的表示能力。

在当前版本中，训练目标以“可训练、可收敛、可用于候选池精排”为首要目标，不额外引入复杂的 teacher、蒸馏、多头监督或阶段化训练机制。

---

### 6. 与总召回的衔接方式

KGAT-AX 不单独工作，而是放在总召回之后：

1. **总召回阶段**  
   通过 Vector Recall、Label Path Recall、Collaboration Recall 构建候选池。

2. **候选池预处理**  
   对候选作者做去重、融合、特征补全，并形成统一候选记录。

3. **KGAT-AX 精排阶段**  
   对候选作者进行图模型打分，得到 `kgat_score`。

4. **最终排序阶段**  
   将 KGAT-AX 分数与召回融合分、规则稳定项共同整合，输出最终 Top-K 作者。

这种设计的优点是：

* 让召回模块负责“找全”；
* 让 KGAT-AX 负责“排准”；
* 降低图模型对全库检索的压力；
* 让线上系统更稳定，也更容易解释。

---

### 7. 当前版本的边界

当前版本的 KGAT-AX 重点是：

* 能接入候选池；
* 能完成候选池内部精排；
* 能与现有排序引擎融合；
* 能在不大改现有训练框架的前提下稳定运行。

因此，以下增强能力**暂不作为当前版本主线**：

* teacher score
* distillation
* 四分支复杂融合
* 多头分项输出
* 复杂负样本分层体系
* 分阶段训练策略

这些内容可以作为后续优化方向，但不纳入当前 README 主体描述，以避免系统设计与当前代码实现脱节。

---

### 8. 小结

在本项目中，KGAT-AX 的最佳定位不是“端到端全库专家搜索模型”，而是：

> **面向统一候选池的知识图谱精排模型。**

它的核心作用是：在多路召回已经找到一批“可能合适”的作者后，进一步利用图谱结构与作者辅助特征，对这些候选作者做更细粒度、更稳健的排序。

**实现与文件入口**：`src/infrastructure/database/kgat_ax/`（`model.py`、`data_loader.py`、`trainer.py`、`generate_training_data.py`、`build_kg_index.py`、`pipeline.py` 等）。训练数据优先从总召回输出的候选池（`candidate_pool` 或 `final_top_XXX`）生成；精排时输入为岗位锚点 + 候选作者列表，输出为 `kgat_score`，与召回序融合后得到最终排序。

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
**可直接开工的规格**（精排三步骤职责、候选池预排序、最终稳定融合与 rule_stability 组成、解释与 CandidatePool 对齐、四段式证据链）见本文 **[后续规划 · 两阶段架构升级与落地计划](#两阶段架构升级与落地计划成稿)** 中「精排与解释的重新定义与证据链升级」6.1～6.7。

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

- **`src/core/recall/candidate_pool.py`**  
  - 定义 **CandidateRecord**（一人一条）：基础与来源（author_id、from_vector/from_label/from_collab、path_count、各路 rank/score_raw/evidence）、**作者静态指标**（h_index、works_count、cited_by_count、recent_works_count、recent_citations、institution_level、top_work_quality）、**query-author 交叉**（topic_similarity、skill_coverage_ratio、domain_consistency、paper_hit_strength、recent_activity_match）、候选池辅助标记（bucket_type、passed_hard_filter、dominant_recall_path、hard_filter_reasons、bucket_reasons）。
  - 定义 **CandidatePool**：**candidate_records**（候选主表）、**candidate_evidence_rows**（证据明细表）、**pool_debug_summary**（三路召回数、去重前后、硬过滤人数、各桶人数、最终精排人数等统计）。
  - 定义 **PoolDebugSummary**：各阶段计数，供调参与排查。

- **`src/core/recall/total_recall.py`**  
  - 定义 `TotalRecallSystem`：
    - 管理 Query 编码器与三条召回路径（向量路 / 标签路 / 协同路），三路配额 K_vector / K_label / K_collab 可配置；
    - 将 JD 文本扩展为带领域 bias 的查询，并行调用三路；
    - **build_candidate_records**：去重合并三路，生成 CandidateRecord，填 rank/path/score_raw/evidence；
    - **score_candidate_pool**：RRF、multi_path_bonus、candidate_pool_score，排序；
    - **_enrich_candidate_features**：补全作者静态指标与 query-author 交叉特征，供训练与精排直接使用；
    - **_apply_hard_filters**：两段式（第一层：仅协同无主题、无论文无指标；第二层待接好作者统计后再开）；
    - **_assign_buckets**：A/B/C/D 四桶（第一版按 from_label/from_vector/from_collab、path_count 规则）；
    - 返回 **CandidatePool**（candidate_records + candidate_evidence_rows + pool_debug_summary），并兼容 **final_top_200**、**rank_map**。
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
  - `LabelRecallPath` 类，是 **标签路召回入口与编排**：统一调用 **label_pipeline 五阶段** 与 **label_means** 中的锚点、扩展、**层级守卫**（hierarchy_guard）、词/论文打分逻辑。**Stage1Result** 含 **jd_profile**（四层领域画像），recall 内将 jd_profile 传入 Stage2；定义 `RecallContext`、`RecallDebugInfo` 等用于阶段间状态与调试。输入输出格式不变：`recall(query_vector, domain_ids)` 仍返回 `(author_id_list, elapsed_ms)`；诊断通过 `debug_info` / `last_debug_info` 输出。
  - **`label_means/`** 子模块：`infra` 统一管理 Neo4j、Faiss、vocab_stats.db、簇中心等；`label_anchors`、`label_expansion` 负责锚点提取与语义扩展（含 **get_vocab_hierarchy_snapshot**、**collect_landing_candidates(jd_profile)**、**allow_primary_to_expand**（放宽条件）；**Stage2A 候选明细/primary 胜出明细调试打印**）；**`hierarchy_guard`** 提供分布/纯度/熵、层级 fit、泛词/负向惩罚、landing/expansion 打分、**should_drop_term（仅对非 primary-like）**、**score_term_record（primary-like 的 path_topic_consistency 保底 0.45）**；**`term_scoring`** 中 **passes_topic_consistency 只拦 support-like，primary-like 默认通过**；`paper_scoring`、`simple_factors`、`advanced_metrics` 负责论文级打分；`base`、`label_debug_cli` 提供基类与调试 CLI。
  - **`label_pipeline/`** 子模块：**stage1_domain_anchors.py**（领域与锚点；**attach_anchor_contexts**、**build_jd_hierarchy_profile**，产出 jd_profile 与锚点 local_context/phrase_context）；**stage2_expansion.py**（学术落点与扩展；接收 jd_profile，raw_candidates 带 subfield_fit/topic_fit/landing_score/cluster_id/**main_subfield_match** 等）；**stage3_term_filtering.py**（**按 tid 去重聚合**、轻硬过滤、topic gate 仅对 support-like 严格、唯一主分 score_term_record 含 **backbone_boost/object_like_penalty/bonus_term_penalty**、_collect_risky_reasons/_bucket_stage3_terms（core/support/risky）、family 角色约束、按 final_score 排序 top_k）；**stage4_paper_recall.py**（二层论文召回、领域软奖励、term_scores 入 paper_score、TERM_MAX_PAPERS/per-term 限流、MELT_RATIO）；**stage5_author_rank.py**（作者排序与截断；预留 CoverageBonus/HierarchyConsistency/FamilyBalancePenalty）。

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
    - **有 candidate_records 且模型启用四分支时**：从候选记录构建 recall/author_aux/interaction 特征，调用 `model.calc_score_v2` 得到四塔融合的 `final_score`；
    - **否则**：构造活跃节点子集，调用 `model.calc_cf_embeddings_subset` 获取融合 AX 的嵌入，对岗位嵌入取均值与候选人嵌入点积得到综合得分；
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
    - 共现数据由 `build_index/build_vocab_stats_index.py` 写入 `vocab_stats.db`，KG 不再调用 `build_cooccurrence_links()`。

- **`src/infrastructure/database/build_kg/kg_utils.py`**  
  - 定义：
    - `GraphEngine`：Neo4j 驱动封装，负责批量写入、执行 schema 变更等；
    - `SyncStateManager`：在 SQLite 中维护各同步任务的断点标记（marker），支持增量构建；
    - `Monitor`：监控各阶段耗时与资源占用。

- **`src/infrastructure/database/build_kg/generate_kg.py`**  
  - `run_pipeline(CONFIG_DICT)`：一键执行完整 KG 构建流程：
    1. 确保 Neo4j 约束与索引存在、SQLite 建立必要索引；
    2. 执行节点同步与拓扑构建；
    3. 执行语义打标与语义桥接（共现由 build_vocab_stats_index 提供，不在此流水线构建）；
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
  - `VocabStatsIndexer` 构建 vocab_stats.db 中多张表，**仅依赖主库 SQLite，不依赖 Neo4j**：（1）`vocabulary_domain_stats`：从主库 `works` 的 `concepts_text`、`keywords_text` 解析 term 并聚合 `domain_ids`，为每个词计算 work_count、领域跨度、domain_dist；（2）`vocabulary_cooccurrence`：从主库 works 流式计算词对共现频次 (term_a, term_b, freq)，标签路共鸣与共现指标均由此表提供（KG 不再构建 Neo4j CO_OCCURRED_WITH）；（3）`vocabulary_topic_stats`：**三级领域**，先用 `data/vocabulary_topic_index.json`（方案 B 导出）直接填有标签词的 field/subfield/topic，再用共现对无标签或缺层级词补全 field_dist、subfield_dist、topic_dist（百分比）。构建前需先运行 `backfill_vocabulary_topic` 与 `export_vocabulary_topic_index`（仅三级领域步骤依赖该 JSON）。

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
    - **四分支侧车（已实现）**：当模型启用四分支时，自动加载 `train_four_branch.json` / `test_four_branch.json`（若有），提供 `get_four_branch_for_batch` 供训练/评估使用；
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
    1. `generate_training_data.py`：生成训练样本（候选池入口 + 分层正负样本 + 可选四分支侧车）；
    2. `build_kg_index.py`：构建加权 KG 索引；
    3. `trainer.py`：执行 KGAT‑AX 训练；
  - 用于一键启动从样本生成到模型训练的全流程。

- **`generate_training_data.py`**（已对齐 README 5.5 与四分支导出）  
  - `KGATAXTrainingGenerator`：
    - **训练样本入口**：优先基于 `candidate_pool.candidate_records` 构造样本；不足时回退到 `final_top_500`。
    - **分层正负样本**：Strong/Weak Positive、EasyNeg/FieldNeg/HardNeg/CollabNeg（README 5.5）；产出 `train.txt / test.txt`，每行 `user_id;pos;fair;neutral;easyNeg`。
    - **四分支侧车**：可选导出 `train_four_branch.json` / `test_four_branch.json`（recall/author_aux/interaction 特征），供 DataLoader 与四分支模型使用。
    - 从 Neo4j 导出加权关系并映射到统一 ID 空间，生成 `kg_final.txt` 与 `id_map.json`。
    - **训练加速**：`generate_refined_train_data(..., recall_workers=N)`（`N≥2`）时多进程并行 `TotalRecallSystem.execute(is_training=True)`，主进程再顺序做 `get_user_id/get_ent_id` 与行组装；环境变量 `KGATAX_RECALL_WORKERS` 可在直接运行脚本时指定。`TotalRecallSystem.execute` 在 `is_training=True` 时跳过领域 prompt 二次编码，与线上默认 `is_training=False` 行为区分。

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
| | `TOPIC_ALIGN_NONE` | 层级无命中时的层级分（topic_align 用；primary 侧用 HIERARCHY_NORM_*） |
| | `HIERARCHY_NORM_TOPIC` / `SUBFIELD` / `FIELD` / `NONE` | primary 打分中 hierarchy_norm 保守映射：0.75 / 0.45 / 0.20 / 0.05 |
| | `HIERARCHY_IDENTITY_THRESHOLD` / `HIERARCHY_IDENTITY_DISCOUNT` | identity 联动上限：anchor_identity_score < 阈值时 hierarchy 乘折扣（默认 0.50 / 0.50） |
| | `ANCHOR_SOURCE_WEIGHT_JD_SUPPLEMENT` | JD 向量补充锚点的 source_weight（默认 **0.65**），primary 打分时乘此值，避免 Robotics 等把 Telerobotics 顶到最前 |
| | `DOMAIN_FIT_MIN_PRIMARY` | Stage2A primary 门控：domain_fit 下限（当前 **0.45**，提高以压缩跨领域词） |
| | `DOMAIN_FIT_HIGH_CONFIDENCE` | Stage2B 高可信 primary 的 domain_fit 下限，仅达标者参与扩散（当前 **0.55**） |

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
  - 通过标题扫描构建 HAS_TOPIC、通过 SBERT 构建 SIMILAR_TO 等语义边；共现数据由 `build_vocab_stats_index` 写入 `vocab_stats.db`，不在此脚本中构建。

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

# 1) 生成训练样本与加权 KG 文本（可选多进程召回，例如 4 个 worker）
# set KGATAX_RECALL_WORKERS=4   # Windows CMD: set KGATAX_RECALL_WORKERS=4
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
    - **hierarchy_match_score**：topic 一致 → 1.0；仅 subfield 一致 → `TOPIC_ALIGN_SUBFIELD`（默认 0.65）；仅 field 一致 → `TOPIC_ALIGN_FIELD`（默认 0.35）；**均未命中 → `TOPIC_ALIGN_NONE`（默认 0.10）**。  
    - **topic_hierarchy_no_match 不再一票否决**：Stage2A 领域过滤中，当层级均未命中时（原 `topic_hierarchy_no_match`）**不再剔除**候选，仅保留 reason 供日志；候选仍进池，在 **primary 打分**中通过 `_hierarchy_norm` 使用 `TOPIC_ALIGN_NONE`（0.10）惩罚，若语义/identity/jd_align 足够强仍可留下。  
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
    | TOPIC_ALIGN_NONE | 层级无命中时的层级分（primary 惩罚用，不在此一票否决） | 0.10 |
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
    return TOPIC_ALIGN_NONE, "none"  # 实际实现中 none 对应 TOPIC_ALIGN_NONE（0.10），候选保留进池、在 primary 中惩罚

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

  - **第一轮改造（已落地）：Stage2A 三层领域准入与锚点内冲突消解**  
    目标：把「三层领域」从打分项升级为**主落点准入门槛**和**锚点内冲突裁判**，减少错词进 primary（如 动力学→propulsion、仿真→simula、运动学→kinesics、抓取→Data retrieval）。  
    - **开关与条件**：`label_expansion.USE_HIERARCHY_PRIMARY_ADMISSION = True` 且存在 JD 三级领域（`jd_field_ids` / `jd_subfield_ids` / `jd_topic_ids` 至少一组非空）时启用新逻辑；否则走原有 hierarchy_norm + 按 primary_score/identity/domain_fit 选 primary。  
    - **新增函数（均在 `label_expansion.py`）**：  
      - `compute_hierarchy_evidence(label, voc_id, active_fields, active_subfields, active_topics)`：返回 `field_overlap`、`subfield_overlap`、`topic_overlap`、`path_match`、`topic_specificity`、`topic_span_penalty`、`hierarchy_level`（topic_exact / subfield_match / field_only / off_path），供准入与打分使用。  
      - `check_primary_admission(anchor_text, anchor_meta, candidate, hierarchy_evidence, semantic_score, anchor_identity, jd_align, source_type, cross_anchor_support_count)`：先判「能不能进 primary」再谈打分。similar_to 要求 topic_overlap≥0.20 或 subfield_overlap≥0.35 或 path_match≥0.40；conditioned_vec 更严（topic≥0.30 或 path≥0.50）；identity 很低时需 topic/jd_align/semantic 强一致才可救；多锚共证可救 borderline。返回 `(admitted, reasons, rescued)`。  
      - `compute_primary_score(candidate, semantic_score, anchor_identity, jd_align, cross_anchor_support_count, local_neighborhood_consistency, hierarchy_evidence, source_type)`：对已准入候选排序，三层领域权重提升（field/subfield/topic/path/specificity 占约 0.8），topic_span 仅作软惩罚，identity 极低时 piecewise 降权。  
      - `resolve_anchor_local_conflicts(anchor, candidates, primary_top_k)`：同锚点内语义重叠候选（同 vid 或同 term）用 `choose_better_term_with_hierarchy(a, b, anchor)` 裁决，泛词/错义词下沉，保留 primary_top_k 个。  
    - **主流程变更**：`stage2_generate_academic_terms` 在 `flat_pool` 构建并算完 neighborhood / multi_anchor 后，若启用准入则：按 (vid, term) 统计 `by_term_cross` → 对每条候选算 `compute_hierarchy_evidence`（按 vid 缓存）→ `check_primary_admission`，未准入且未 rescued 的标记并 `log_primary_reject` → 准入的写 `hierarchy_evidence` 与 `compute_primary_score` → 过滤掉被拒候选得到新 `flat_pool` → 按锚点 `resolve_anchor_local_conflicts` 得到每锚 primary 列表 → 再写 `evidence_table` 供后续 debug。  
    - **日志**：`[Stage2A Reject] anchor=... cand=... topic= sub= path= reasons=[...]`；冲突时 `[Stage2A Conflict Drop]` / `[Stage2A Conflict Replace]`（需 `LABEL_EXPANSION_DEBUG`）。  
    - **可调常量**：`PRIMARY_MIN_TOPIC_OVERLAP_SIMILAR`、`PRIMARY_MIN_PATH_MATCH_SIMILAR`、`PRIMARY_MIN_TOPIC_OVERLAP_CONDVEC`、`PRIMARY_MIN_PATH_MATCH_CONDVEC`、`PRIMARY_LOW_IDENTITY_RESCUE_*`、`PRIMARY_RESCUE_CROSS_ANCHOR_MIN`、`TOPIC_SPAN_PENALTY_FACTOR`、`PRIMARY_SCORE_W_*` 等均在 `label_expansion.py` 顶部集中定义。

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
    | **PreparedAnchor** | Stage1 输出，Stage2 输入锚点（见「四、实现细节契约」状态传递） | **anchor: str**, **vid: int**, **anchor_type: str** (acronym \| canonical \| application \| task \| unknown), **expanded_forms: List[str]**, **conditioned_vec** (可选), **source_type** (skill_direct \| jd_vector_supplement), **source_weight: float** (primary 乘此值) | normalized_term |
    | **LandingCandidate** | Stage2A 落点检索单条候选（collect 内部） | **vid: int**, **term: str**, **source: str** (similar_to \| jd_vector), **semantic_score: float** (边权或 cos_sim) | topic_prior_score |
    | **Stage2ACandidate** | Stage2A 极简重构统一候选对象 | **tid, term, source**；证据字段 semantic_score、context_sim、jd_align、mainline_sim、cross_anchor_support、family_match、hierarchy_consistency、polysemy_risk、isolation_risk；**relative_scores、composite_rank_score**；**survive_primary、can_expand、role** | 组内相对排序用，无固定阈值判死 |
    | **PrimaryLanding** | Stage2A 选出的主落点 | **vid: int**, **term: str**, **identity_score: float**, **source: str**, **anchor: str** | 同 LandingCandidate 可选字段；expandable 由 judge_expandability_relative 决定 |
    | **ExpandedTermCandidate** | Stage2B 扩展后的单条 term（含 primary） | **vid: int**, **term: str**, **term_role: str** (primary \| dense_expansion \| cluster_expansion \| cooc_expansion), **identity_score: float**, **source: str**, **anchor: str** | quality_score, topic_fit, cooc_purity, resonance, span_penalty, from_primary_vid；**topic_align, topic_level, topic_confidence**（三层领域修订版） |
    | **TermCandidate** | Stage3 输入/输出统一 term 结构（与上文伪代码一致） | **anchor, anchor_type, term_id/vid, term, term_role, source, identity_score, quality_score, final_score** | domain_fit, topic_fit, cooc_purity, is_primary_landing, debug_error_type；**topic_align, topic_level, topic_confidence**（供 compose_term_final_score 乘性融合） |
    | **TermDebugRecord** | 调试与阶段统计用 | **term_id, term, term_role, error_type** (acronym_error \| alias_mapping_error \| generic_drift \| domain_mismatch \| weak_identity \| pass), **stage** (stage2a \| stage2b \| stage3) | identity_score, quality_score, gate_fail_reason |

    **函数级入参/出参契约（关键接口）**：

    | 函数 | 入参 | 出参 | 说明 |
    |------|------|------|------|
    | `retrieve_academic_term_by_similar_to(anchor: PreparedAnchor)` | 单锚点 | **List[LandingCandidate]** | 从 anchor（industry）查跨类型 SIMILAR_TO→学术词，图内为带扩写向量相似度 |
    | `collect_landing_candidates(anchor: PreparedAnchor)` | PreparedAnchor | **List[LandingCandidate]** | 只负责召回；再经 `landing_candidates_to_stage2a` 转为 Stage2ACandidate 进入组内相对选主流程 |
    | `score_academic_identity(c: LandingCandidate, anchor: PreparedAnchor)` | 单条候选 + 锚点 | **float** [0,1] | 按 source（similar_to / jd_vector）与 semantic_score 等算身份分 |
    | `select_primary_academic_landings(candidates: List[LandingCandidate], min_identity_score, identity_margin)` | 候选列表 + 阈值 | **List[PrimaryLanding]** | 过滤 ≥ min_identity_score，且满足 top1−top2 ≥ identity_margin，每 anchor 最多 PRIMARY_MAX_PER_ANCHOR 个 |
    | `expand_from_vocab_dense_neighbors(primary_landings, ..., jd_profile)` | 主落点列表 + 领域/jd_profile | **List[ExpandedTermCandidate]**（term_role=dense_expansion） | 仅对 check_seed_eligibility 通过的 seed 扩展；候选过 **support_expandable_for_anchor 四道门**（primary/anchor/context/family 一致性），identity_score 用 keep_score；sim≥0.55、domain_fit≥0.72；设备/对象词默认拒；每 seed 最多 2 条（强 seed 3 条）。见「Dense 最小修复补丁」。 |
    | `expand_from_cluster_members(...)` | 主落点列表 | **List[ExpandedTermCandidate]** | **当前全关**：直接 return []；若重开须三重门（强 seed + 高 purity + support_anchor_fit） |
    | `expand_from_cooccurrence_support(primary_landings, ..., jd_profile)` | 主落点列表 + 领域/jd_profile | **List[ExpandedTermCandidate]**（term_role=cooc_expansion） | 仅强 normal seed（primary_score≥0.70、jd_align≥0.82、cross_anchor≥2）；共现词过 support_expandable_for_anchor、freq≥3；每 seed 最多 2 条 |
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
    | TOPIC_ALIGN_NONE | 0.10 | 层级无命中时的层级分（topic_align 用） |
    | **Stage2A hierarchy 保守映射与 identity 联动** | | |
    | HIERARCHY_NORM_TOPIC / SUBFIELD / FIELD / NONE | 0.75 / 0.45 / 0.20 / 0.05 | primary 打分中 _hierarchy_norm 的保守映射 |
    | HIERARCHY_IDENTITY_THRESHOLD / HIERARCHY_IDENTITY_DISCOUNT | 0.50 / 0.50 | identity 联动上限：identity&lt;阈值时 hierarchy 乘折扣 |
    | **Stage2A 补充锚点来源权重** | | |
    | ANCHOR_SOURCE_WEIGHT_JD_SUPPLEMENT | 0.65 | JD 向量补充锚点的 source_weight，primary 最后乘此值 |
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
    Stage 1 输出须从简单字符串列表升级为对象列表。`stage1_domain_anchors.py` 返回的 `Stage1Result` 必须包含如下精确结构的 `PreparedAnchor`（`Stage1Result` 定义于同文件 `stage1_domain_anchors.py`）：

    ```python
    @dataclass
    class PreparedAnchor:
        anchor: str                 # 原始清洗后的词，如 "rrt"
        vid: int                    # Vocabulary id
        anchor_type: str            # acronym | canonical | application | task | unknown
        expanded_forms: List[str]   # 扩写结果，如 ["rapidly exploring random tree"]
        conditioned_vec: Optional[np.ndarray]  # JD 上下文条件化向量
        source_type: str = "skill_direct"     # skill_direct | jd_vector_supplement
        source_weight: float = 1.0             # 补充锚点 0.65，primary 打分时乘此值
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

---

### 两阶段架构升级与落地计划（成稿）

以下为可直接落地的修改计划成稿：总召回、训练数据、KGAT-AX 结构、精排与证据链的升级方向与实施顺序，可整体作为后续规划正文引用或拆入对应章节。

#### 整体架构与数据流

##### 2.1 两阶段推荐范式：从“多路名单融合”升级为“统一候选池 + 深度精排”

当前系统已经具备：

- 多路召回（Vector / Label / Collaboration）
- 图模型精排（KGAT-AX）
- Neo4j 证据解释

但从工程职责上看，后续系统将进一步明确为更标准的两阶段推荐架构：

> **总召回（构建候选池）→ 轻量预排序/去噪 → KGAT-AX（候选池内深度重排）→ 最终融合与证据链解释**

这意味着后续不再把总召回理解为“三条路径分别给名单，再做一次融合”，而是把它升级为一个真正的**候选池构建器（Candidate Pool Builder）**。
候选池的目标不是直接给最终答案，而是为下游精排提供一个：

- 高覆盖
- 低噪声
- 来源清晰
- 特征完整
- 与训练分布一致

的统一候选集合。

因此，整条链路的职责将被重新定义为：

1. **召回层**
- 负责尽量不漏掉相关专家；
- 强调“覆盖率”和“多样性”，不追求最终精确排序。

2. **候选池层**
- 对三路召回结果进行配额控制、去重合并、来源特征保留、基础过滤与分桶；
- 这是线上推理和线下训练共用的中间层。

3. **预排序层**
- 在进入 KGAT-AX 之前，对候选池做一次轻量级压缩；
- 目的是降低精排成本，而不是代替精排。

4. **精排层**
- 使用 KGAT-AX 对统一候选池内部作者做重排序；
- 核心目标是从“都看起来相关”的作者中找出“最适合当前岗位的活跃研究者”。

5. **解释层**
- 将召回来源、图路径证据、论文证据、作者指标与模型排序依据统一输出为可追溯证据链。

##### 2.2 升级后的完整线上流程

后续线上推荐流程建议固定为如下顺序：

1. **输入 JD / 查询文本**
- 用户输入岗位职责、技术栈、研究方向、业务领域等。
- 若用户手动指定领域，则优先以用户领域为硬约束；
- 否则由领域探测器自动判定 active domains。

2. **统一编码与领域探测**
- 对原始 JD 做基础编码；
- 使用 `DomainDetector` 输出 `active_domains`；
- 同时为向量路构造带有领域 bias 的查询向量。

3. **三路独立召回**
- Vector 路输出高覆盖候选；
- Label 路输出高精度候选；
- Collaboration 路基于前两路种子做结构补充。

4. **候选池构建**
- 三路结果做配额控制；
- 按 `author_id` 去重合并；
- 保留各路来源与分数；
- 执行硬过滤；
- 执行分桶；
- 产出统一候选池。

5. **候选池预排序**
- 使用轻量规则分数或基础特征分数对候选池做一次压缩；
- 将候选池规模控制到适合 KGAT-AX 处理的范围。

6. **KGAT-AX 深度精排**
- 结合图谱结构、高阶关系、作者显式指标、召回来源和 query-author 交叉特征，得到最终模型分数。

7. **稳定融合**
- 将候选池基础分、KGAT-AX 模型分、规则稳定项做最终融合；
- 输出 TopN 专家。

8. **证据链解释**
- 输出“召回来源 → 技能词/学术词 → 论文 → 作者 → 合作者/机构/期刊”的完整解释链。

#### 总召回模块的重新定位与升级（total_recall）

##### 3.1 总召回模块的重新定位

`src/core/recall/total_recall.py` 当前已经完成了三项关键能力：

- 统一 Query 编码；
- 并行调起向量路与标签路；
- 使用前两路种子驱动协同路；
- 使用 RRF 做多路融合；
- 对多路命中作者追加柔性 bonus。

后续不推翻这一设计，而是在其基础上进一步升级为**统一候选池构建器**。

也就是说，`TotalRecallSystem` 不再只返回：

- `final_top_200`
- `rank_map`

而是应逐步升级为返回一个结构化候选池对象，包含：

- 候选作者主表；
- 候选来源特征；
- 候选证据明细；
- 候选分桶信息；
- 可直接供训练脚本与线上精排复用的中间表示。

##### 3.2 总召回的目标与原则

- **3.2.1 高覆盖但不放任脏候选**  
  总召回不是最终排序器，因此必须允许一定程度的“相关但未必最优”的候选进入池中；但它也不能过脏，否则精排模型会浪费容量去清理明显错误的人。

- **3.2.2 各路径职责必须清晰**  
  三条路径不再是完全平级的“投票器”，而是职责明确的召回源：**Vector 路**：高覆盖，负责语义泛化；**Label 路**：高精度，负责岗位技能到学术词再到论文/作者的精确链路；**Collaboration 路**：结构补充，负责把核心作者周围的高价值合作者补进来。

- **3.2.3 候选池必须能复用到训练**  
  总召回输出不能只服务线上展示。后续训练数据生成也应尽量复用这套候选池逻辑，保证线上看到什么样的候选分布，训练就学会在什么样的候选分布上做排序。

- **3.2.4 来源信息本身就是排序特征**  
  作者是从哪条路被召回进来的，本身就是极强的排序信号。因此，总召回必须保留路径来源，而不是只输出一个融合后的 author_id 列表。

##### 3.3 总召回的升级后流程（七阶段）

- **阶段一：统一编码与领域处理**  
  输入：`query_text`、`domain_id`、`is_training`。输出：`raw_vec`、`query_vec`、`active_domains`、`applied_domain_str`、`domain_debug`、`vector_domains`。此阶段继续沿用当前逻辑。

- **阶段二：三路独立召回**  
  **Vector 路输出**：`author_id`、`vector_score_raw`、`vector_rank`、`vector_evidence`。**Label 路输出**：`author_id`、`label_score_raw`、`label_rank`、`label_evidence`。**Collaboration 路输出**：`author_id`、`collab_score_raw`、`collab_rank`、`collab_evidence`。即使当前代码里三路很多地方还只返回作者列表，README 里也应先把目标输出接口定义清楚。

- **阶段三：配额控制**  
  明确独立可调的三路配额：`K_vector`、`K_label`、`K_collab`。推荐原则：`K_vector` 较大负责广覆盖；`K_label` 较大或略高负责高精度；`K_collab` 显著低于前两路，仅作补充。协同路不再与向量路、标签路争夺主导地位，而是作为一种“结构补全路径”存在，抑制“熟人圈放大”问题。

- **阶段四：去重合并**  
  为每个作者构建统一候选记录，推荐字段：`author_id`、`recall_from_vector`、`recall_from_label`、`recall_from_collab`、`recall_path_count`、`vector_rank`、`label_rank`、`collab_rank`、`vector_score_raw`、`label_score_raw`、`collab_score_raw`、`rrf_score`、`multi_path_bonus`、`candidate_pool_score`、`is_multi_path_hit`。

- **阶段五：硬过滤**  
  领域硬过滤、活跃度硬过滤、论文质量硬过滤、协同弱命中过滤、极弱单路命中过滤。目标不是把池子缩得非常小，而是清掉明显不值得进入精排的候选。

- **阶段六：分桶**  
  推荐分桶：**A 桶**：主题强相关 + 近期活跃；**B 桶**：主题强相关 + 指标一般；**C 桶**：指标较强 + 主题中等相关；**D 桶**：协同补充型候选。防止候选池被同一种风格的作者占满，给 KGAT-AX 提供更有层次的比较对象。

- **阶段七：统一导出**  
  (1) 候选作者主表：面向精排与训练，每个作者一行，包含所有统一特征。(2) 候选证据明细表：面向解释，记录作者被召回的路径、命中的 skill / term / paper / collaborator / source 等证据。

##### 3.4 总召回建议新增的中间结构

- **CandidateRecord**  
  用于统一表示某位作者在候选池中的信息。建议包含：基础标识（`author_id`、`author_name`）；路径来源（`from_vector`、`from_label`、`from_collab`、`path_count`）；各路表现（`vector_rank`、`label_rank`、`collab_rank`、`vector_score_raw`、`label_score_raw`、`collab_score_raw`）；融合分（`rrf_score`、`multi_path_bonus`、`candidate_pool_score`）；候选质量特征（`bucket_type`、`is_multi_path_hit`、`passed_hard_filter`）；证据（`vector_evidence`、`label_evidence`、`collab_evidence`）。

- **CandidatePool**  
  用于统一表示一轮查询的候选池结果。建议包含：`query_text`、`applied_domains`、`candidate_records`、`candidate_evidence_rows`、`stats_summary`、`path_costs`、`domain_debug`。后续训练脚本、精排引擎、解释模块都从这个对象读取。

##### 3.5 总召回的预期收益

提升线上候选池质量；降低 KGAT-AX 的清噪负担；保留路径来源，便于精排学习；统一线上与训练的数据分布；为解释模块提供更完整的中间证据。

#### KGAT-AX v2：项目增强版结构升级（规划，非论文原生）

**说明**：以下 5.1～5.7 所述 **teacher、distill、四分支、pair_label、negative_type、阶段化训练** 等，均**不属于论文 KGAT-AX 原生内容**，而是针对本项目「岗位→作者」专家推荐任务提出的 **KGAT-AX v2 增强方案（规划中）**。论文原生 KGAT-AX 仅包含 Embedding / Propagation / Prediction / Fusion 四层及 BPR + KG 三元组训练目标。

##### 5.1 KGAT-AX 的重新定位：从“图排序模型”升级为“第二阶段候选池精排器”

后续 KGAT-AX 不再被描述为“对全量作者直接做端到端检索”的模型，而明确定位为：

> **第二阶段深度精排器（Re-ranker）**

其职责不是在全库高速找人，而是对总召回输出的统一候选池进行精细排序，判断：

- 哪些作者虽然相关，但不值得排到前面；
- 哪些作者既相关、又活跃、又有较强学术实力；
- 哪些作者具有更强的多路共识与图谱支撑，适合进入最终 Top-N 结果。

因此，KGAT-AX 的输入不再只是一对简单的 `(job_id, author_id)`，而应是：**图结构信息**、**作者显式指标**（当前已用）、以及可选的召回来源与 query-author 交叉特征（后续扩展）。

**关于四分支与 v2 规划**：四分支输入字段定义、`s_graph`/`s_author`/`s_recall`/`s_interaction` 分项输出、teacher、distill、pair_label/negative_type、Stage0～Stage3 阶段化训练、复杂 loss 设计等，**不作为当前版本主线**，已从本 README 主体中移除；若需扩展可参考独立规划文档或后续版本。当前版本以候选池精排与总召回衔接、CF+KG 双目标训练为主。

##### 5.2 KGAT-AX 训练数据入口与样本定义（已实现）

**已实现**：`generate_training_data.py` 优先基于 `candidate_pool.candidate_records` 构造训练样本；当候选池不足或不可用时回退到 `final_top_500`，训练数据主入口与线上候选池保持一致。

- **训练样本来源**：使用 `results["candidate_pool"].candidate_records`、`results["candidate_pool"].candidate_evidence_rows` 作为主要输入来源；前者用于生成 (job, author) 排序样本，后者用于构造证据强度、主题匹配和论文命中等辅助特征。
- **正样本定义（分层）**：**Strong Positive**：满足之一或组合——`passed_hard_filter == True`、`bucket_type == 'A'`、`from_label == True`、`path_count >= 2`、具备较强主题命中与论文证据支撑。**Weak Positive**：`passed_hard_filter == True`、`bucket_type in {'A', 'B'}`、有明确主题支撑，但多路命中或作者指标不如 Strong Positive 稳定。强正样本用于学习“谁应该明显排前”，弱正样本用于保留排序边界的柔性。
- **负样本定义（四类）**：**EasyNeg**：明显不相关、候选池外或候选池尾部弱相关作者，用于学习基础边界。**FieldNeg**：领域相近但主题偏移，用于学习同领域内的细粒度区分。**HardNeg**：与正样本处于同一 job 的同一候选池中，`from_label == True` 或 `path_count >= 2`、`passed_hard_filter == True`，排名接近正样本、指标也不差但不是最佳人选；优先从同桶或相邻桶中采样。**CollabNeg**：`from_collab == True` 但缺乏足够主题支撑，用于抑制合作关系带来的误抬升。
- **训练样本导出字段**：除基本图边外，建议额外导出：（1）召回来源特征：`from_vector`、`from_label`、`from_collab`、`path_count`、`vector_rank`、`label_rank`、`collab_rank`、`candidate_pool_score`、`bucket_type`；（2）作者显式指标：`h_index`、`works_count`、`cited_by_count`、`recent_works_count`、`recent_citations`、`institution_level`、`top_work_quality`；（3）query-author 交叉特征：`topic_similarity`、`skill_coverage_ratio`、`domain_consistency`、`paper_hit_strength`、`recent_activity_match`、`academic_term_hit_strength`。这样四分支输入与训练数据导出字段一一对应。

##### 5.3 KGAT-AX 训练与评估建议

训练目标仍可保留 pairwise / BPR 风格主目标，但评估阶段应明显偏向 **top-heavy 指标**。建议重点关注：Top10 命中质量、Top20 排序稳定性、Strong Positive 的前排保持率、多路命中候选的前排占比、证据链一致性。即后续优化目标不再只是“整体平均排序误差更小”，而是：**让真正值得推荐的人，稳定地出现在前排。**

##### 5.4 当前优势与为什么仍保留 KGAT-AX

当前 KGAT-AX 已有离线训练脚本、图索引构建工具、训练数据导出链路、线上调用与排序融合能力。收益最大的做法不是“重写精排器”，而是：把总召回做成高质量候选池；把训练数据对齐线上候选池；让现有 KGAT-AX 稳定作为候选池精排器运行。后续可再考虑扩展更多特征或分支，但不作为当前版本主线。

#### 精排与解释的重新定义与证据链升级（可直接开工规格）

##### 6.1 精排阶段的完整职责划分

后续精排不再被描述为“召回后再做一次融合打分”，而明确分为三个步骤：**候选池预排序（Pre-Rank）** → **KGAT-AX 深度精排（Re-Rank）** → **最终稳定融合（Stable Fusion）**。这样拆分的原因是：总召回负责构建高质量候选池；预排序负责压缩规模、去掉明显弱候选；KGAT-AX 负责在“都像相关的人”里做细粒度排序；稳定融合负责控制线上结果的稳定性与可解释性。

##### 6.2 候选池预排序（ranking_engine 职责）

候选池预排序建议由 **ranking_engine.py** 明确承担，而不是只停留在规划层。

- **输入**：`candidate_pool.candidate_records`。
- **预排序特征**：可优先使用轻量级特征：`candidate_pool_score`、`from_label`、`from_vector`、`from_collab`、`path_count`、`bucket_type`、`domain_consistency`、`paper_hit_strength`、`recent_activity_match`。
- **输出**：压缩后的精排候选集，例如从 300～500 缩到 150～200；保留候选来源、分桶和证据明细。

预排序的目标不是给最终排序，而是：降低 KGAT-AX 推理负担；提升精排输入质量；保留精排真正需要区分的作者。

##### 6.3 KGAT-AX 深度精排

在一批都看起来相关的作者里，判断谁更贴题、谁更活跃、谁更有实力、谁更值得排到前面。更重视 Top10/Top20 的前排质量、不同类型候选之间的稳定比较、排序结果与证据链的一致性。

##### 6.4 最终稳定融合

后续线上最终排序不建议仅依据 `kgatax_score`，而建议使用：

\[
final\_score =
\lambda_1 \cdot candidate\_pool\_score +
\lambda_2 \cdot kgatax\_score +
\lambda_3 \cdot rule\_stability
\]

其中：`candidate_pool_score` 为候选池阶段基础质量分；`kgatax_score` 为 KGAT-AX 深度精排得分；`rule_stability` 为稳定项，用于避免线上结果出现明显异常。

##### 6.5 rule_stability 的组成建议

后续 `rule_stability` 不应只作为抽象概念出现，而建议明确由以下项组成：**multi_path_bonus**（多路共识加成）、**label_support_bonus**（标签路支撑加成）、**collab_only_penalty**（仅协同命中惩罚）、**low_activity_penalty**（低活跃度惩罚）、**weak_paper_evidence_penalty**（弱论文证据惩罚）。即：多条召回路径共同支持的候选更可信；有明确技能→学术词→论文路径支撑的作者更应排前；仅通过合作网络进入候选池、缺乏主题支撑的作者应更谨慎处理；长期不活跃作者不应因历史指标过高而被过度抬升；若缺乏强论文命中证据，则降低最终排序稳定性。

##### 6.6 解释模块与 CandidatePool 的对齐

当前解释模块已具备图路径和注意力基础，后续建议进一步接入候选池来源信息。解释模块后续输入不应只有 `job_raw_ids`、`author_id`，还应增加：`candidate_record`、`candidate_evidence_rows`。这样解释模块才能同时回答：这个人为什么被召回？为什么排得比别人更前？关键证据链是什么？

##### 6.7 四段式证据链输出格式

后续解释模块建议统一采用四段式结构。

1. **召回来源摘要**：说明该作者来自哪条召回路径；是否命中多条路径；主导召回来源是什么。
2. **主题匹配摘要**：说明岗位核心技能或主题词；对应的 academic term；命中的关键论文；论文与作者的连接路径。
3. **学术实力摘要**：说明近年活跃程度；H-index、引用、论文数等指标；机构或发表渠道质量。
4. **模型置信摘要**：说明 KGAT-AX 更偏向哪一类支撑（图路径 / 指标 / 召回来源 / 交叉匹配）；关键论文的注意力或贡献度；排序结果与候选池证据是否一致。

这样最终解释就不仅是“找到一条图路径”，而是形成 **召回原因 + 排序原因 + 关键证据 + 学术实力支撑** 的完整证据链。

#### 后续规划落地顺序与目标（可直接开工）

##### 10.1 总召回升级计划

优先对 `src/core/recall/total_recall.py` 做结构升级：三路独立配额控制、去重合并、路径来源特征化、硬过滤、分桶、候选池统一导出给线上与训练。同时改善线上候选池质量、训练样本质量、精排输入质量、证据链完整度。

##### 10.2 KGAT-AX 增补计划

为进一步提升精排阶段的效果与可解释性，后续 KGAT-AX 改造重点不在于更换主模型，而在于：**当前版本**以候选池精排、训练数据与 CandidatePool 对齐、CF+KG 双目标为主；四分支、分项输出、teacher/distill 等不作为主线。

- **训练样本入口与 CandidatePool 对齐（已实现）**：`generate_training_data.py` 优先基于 `candidate_pool.candidate_records` 构造样本；`final_top_500` 仅作兼容回退。
- **正负样本定义显式化（已实现）**：Strong Positive / Weak Positive；EasyNeg / FieldNeg / HardNeg / CollabNeg（详见 5.2）。
- **预排序职责归属明确**：由 `ranking_engine.py` 承担轻量预排序逻辑；不让“预排序”停留在规划文字层面。
- **稳定项显式实现**：`rule_stability` 由 bonus / penalty 组成（见 6.5）；保证线上排序更稳定。
- **解释模块对齐 CandidatePool**：让解释模块同时接入候选来源和图路径证据；统一“召回-排序-解释”的中间信息口径。

##### 10.3 建议新增的验收与消融实验

为验证后续改造是否真正带来收益，建议在训练与离线验证中加入以下对比实验：

1. **分支消融实验**：仅 Graph Tower → Graph + Author → Graph + Author + Recall → 全量四分支。
2. **候选池策略实验**：有/无硬过滤、有/无分桶、有/无预排序。
3. **稳定融合实验**：仅 `kgatax_score` → `candidate_pool_score + kgatax_score` → 全量 `candidate_pool_score + kgatax_score + rule_stability`。
4. **解释一致性分析**：前排作者是否具备多路来源支撑；排序前列作者是否有关键论文证据；模型高分与证据链是否一致。

这样可以避免只从单一离线指标判断模型是否“变好”，而是从**排序质量、稳定性、可解释性、工程可控性**四个维度共同评估。

##### 10.4 精排与解释升级计划

精排阶段固定为：候选池预排序 → KGAT-AX 深度精排 → 最终稳定融合 → 四段式证据链输出。证据链统一覆盖：召回来源、主题匹配、学术实力、模型置信；解释模块接入 `candidate_record` 与 `candidate_evidence_rows`，输出四段式证据链（见 6.7）。

##### 10.5 当前阶段的改造优先级

为兼顾工期与收益，建议按以下顺序推进：

- **第一阶段**：先完成总召回升级——配额控制、CandidatePool、去重合并、硬过滤、分桶、统一导出给线上与训练。
- **第二阶段**：再补齐 KGAT-AX 输入与训练数据——CandidatePool 样本入口、四分支字段导出、分层正负样本、dataloader 与模型 forward 对齐。
- **第三阶段**：最后补强最终精排与解释——`ranking_engine` 预排序、`rule_stability`、四段式证据链、排序与解释对齐。

该顺序的核心原则是：**先把候选池做成稳定的中间层，再让 KGAT-AX 学会在这个中间层上高质量排序，最后把排序原因与证据链统一表达。**

##### 10.6 升级后的总体目标

完成上述改造后，系统将从“多路召回 + 模型融合”的原型架构，升级为一个更接近工业推荐系统的完整两阶段专家推荐框架：总召回负责高质量候选池；KGAT-AX 负责候选池内部深度排序；训练样本与线上候选池分布一致；证据链能够完整说明召回原因、排序原因与关键论文依据。这也是当前工期下最稳、最现实、收益最高的演进方向。

---

**KGAT-AX v2 详细方案说明**  
teacher、distill、四分支、pair_label/negative_type、Stage0～Stage3 阶段化训练、复杂 loss 设计、防 teacher 坍缩等详细内容已从 README 主体中移除，**当前版本以候选池精排与总召回衔接、CF+KG 双目标为主**。若需扩展为 v2 增强版，可参考历史设计文档或后续单独成稿。以下为原 v2 方案的结构摘要，仅作占位与索引，具体实现不纳入当前主线。

<details>
<summary>KGAT-AX v2 方案结构摘要（折叠，非当前版本主线）</summary>

- 改造背景：缺乏真实监督、学术指标利用不足、图结构优势未充分释放。
- 改造目标：保留 KGAT 主干，引入作者指标分支、Query-Author 匹配分支、弱监督 teacher 训练。
- 总体设计：图表示 + 作者指标 + 匹配特征 + teacher 信号；teacher_score、pair_label、negative_type；Loss = L_rank + L_cls + L_distill + L_graph；Stage0 数据与 teacher → Stage1 图预热 → Stage2 联合训练 → Stage3 难例校准。
- 详细字段、公式与实现清单见历史版本或独立规划文档。

</details>

---
（以下为原 v2 长文档占位，内容已精简，保留小节标题便于检索）

### 原 KGAT-AX v2 修改方案（仅保留标题索引，正文已移除）

#### 1. 改造背景（略）
#### 2. 改造目标（略）
#### 3. 总体设计思路（略）

**（以上为 KGAT-AX v2 方案结构索引；详细正文如 teacher、distill、四分支、阶段化训练、loss 设计等已移除，当前版本以候选池精排为主。）**

---



