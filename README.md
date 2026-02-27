# TalentRecommendationSystem (智能人才推荐系统)

## 📖 项目概述

`TalentRecommendationSystem` 是一个面向工业级场景的智能人才推荐引擎。不同于传统的关键词匹配，本项目采用了 **“多路召回 + 深度精排”** 的双阶段架构。

核心亮点在于集成了 **KGAT-AX (Knowledge Graph Attention Network - AX)** 排序模型，能够深度挖掘“人才-技能-岗位”知识图谱中的隐式关联，并针对非标准 DOI（如 `doi:egusphere-...`）进行了健壮的文本解析处理。

---

## 📂 详细项目结构

该项目遵循高度模块化的设计原则，将算法逻辑、基础设施和交互界面清晰解耦：

```bash
TalentRecommendationSystem/
├── data/                       # 数据仓库
│   ├── raw/                    # 原始简历、JD 文本及三元组数据
│   ├── processed/              # 预处理后的特征矩阵及图谱索引
│   └── index/                  # Faiss 向量索引与协同过滤缓存
├── models/                     # 模型存储
│   ├── sbert_backbone/         # 语义向量化模型
│   └── kgat_ax_weights/        # 训练好的 KGAT-AX 权重文件
├── src/                        # 源代码核心
│   ├── core/                   # 推荐引擎核心逻辑
│   │   ├── recall/             # 第一阶段：多路召回模块
│   │   │   ├── vector_path.py  # 基于 SBERT + Faiss 的向量语义召回
│   │   │   ├── label_path.py   # 基于技能标签（Tags）的精确召回
│   │   │   └── collaboration_path.py # 基于历史投递行为的协同召回
│   │   ├── total_recall.py     # 召回汇总器：多路结果去重与分值对齐
│   │   ├── ranking_engine.py   # 第二阶段：精排引擎（调用 KGAT-AX）
│   │   ├── talent_app.py       # 业务逻辑封装（Top-N 筛选、过滤规则）
│   │   └── total_core.py       # 系统总控入口：串联召回与排序流
│   ├── infrastructure/         # 基础设施与数据底座
│   │   ├── database/           # 数据访问层
│   │   │   └── kgat_ax/        # KGAT-AX 排序模型核心实现
│   │   │       ├── model.py    # PyTorch 神经网络架构
│   │   │       ├── data_loader.py # 图数据采样与 Batch 序列化
│   │   │       └── kgat_parser/ 
│   │   │           └── parser_kgat.py # 超参数定义与命令行解析
│   │   ├── build_vector_index.py        # 构建 Faiss 向量检索索引
│   │   ├── build_feature_index.py       # 构建结构化特征倒排索引
│   │   └── build_collaborative_index.py # 构建用户行为协同矩阵
│   ├── interface/              # 交互界面
│   │   └── app.py              # 基于 Streamlit 的可视化演示端
│   ├── preprocess.py           # 全局数据清洗与 Work_ID (DOI) 标准化
│   ├── recommend.py            # API 级接口封装
│   └── evaluate.py             # 模型离线指标评估 (Hit Rate, NDCG, MRR)
├── requirements.txt            # 项目依赖清单
└── README.md                   # 本文件

```

---

## 🛠️ 技术深度解析

### 1. 多路召回 (Multi-path Recall)

为了平衡系统的召回率与响应速度，`total_recall.py` 调度了三条路径：

* **语义路径 (`vector_path.py`)**：通过 SBERT 将岗位描述映射为 768 维向量，捕获“懂 Python 的数据分析师”与“算法工程师”之间的深层语义联系。
* **结构化路径 (`label_path.py`)**：强制匹配核心硬技能（如学历、证书、编程语言）。
* **图关联路径 (`collaboration_path.py`)**：利用知识图谱的一阶和二阶邻居，发现潜在的交叉学科人才。

### 2. KGAT-AX 精排模型

精排层位于 `src/infrastructure/database/kgat_ax/`，其核心思想是：

* **图注意力机制**：在“人才-技能-岗位”构成的异质图上进行消息传递。
* **AX 增强优化**：针对人才推荐场景中极其稀疏的交互数据进行了正则化优化，提升了长尾人才的推荐精度。

### 3. 数据处理特色

在 `preprocess.py` 中，项目对人才简历中的学术/项目标识符（Work ID）进行了特殊处理，支持如下非标准 DOI 格式：

* `doi:egusphere-2025-4093-ac2`
* `doi:v2`
* 传统的 `10.xxx` 格式

---

## 🚀 快速上手

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/ink9dye/TalentRecommendationSystem.git
cd TalentRecommendationSystem

# 安装依赖
pip install -r requirements.txt

```

### 初始化索引

在首次运行推荐前，需要预计算索引文件：

```bash
python src/infrastructure/build_vector_index.py
python src/infrastructure/build_feature_index.py

```

### 运行可视化 Demo

```bash
streamlit run src/interface/app.py

```

---

## 📊 评估指标

项目通过 `evaluate.py` 提供标准的推荐系统评估：
| 指标 | 描述 |
| :--- | :--- |
| **Recall@100** | 多路召回阶段对真实投递人才的覆盖率 |
| **NDCG@10** | 精排阶段前 10 名推荐结果的排序质量 |
| **Inference Time** | 从输入 JD 到输出 Top 100 结果的平均耗时 |

---

## 🤝 参与贡献

1. Fork 本项目。
2. 在 `src/core/recall/` 下新增你的召回路径文件。
3. 在 `total_recall.py` 中注册新路径。
4. 提交 Pull Request。

---

## 📧 联系方式

* **项目维护者**: ink9dye
* **GitHub**: [https://github.com/ink9dye](https://www.google.com/search?q=https://github.com/ink9dye)

---

**Would you like me to generate a specific `config.yaml` or a sample `requirements.txt` based on the tech stack mentioned above?**