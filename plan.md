# Talent Rec System - 项目目录结构

## 核心目录说明

###  `app/` - FastAPI 后端核心代码
- **`api/`** - 接口路由层
  - `v1/` - API 版本控制，处理推荐请求、详情查询等接口
- **`core/`** - 核心配置模块
  - 配置管理、常量定义、安全认证等
- **`models/`** - 数据模型层
  - Pydantic 请求/响应模型（Schema）
- **`services/`** - 业务逻辑层
  - 召回引擎、精排调度、证据链生成等核心业务逻辑

###  `crawler/` - 数据采集模块
- **`openalex/`** - OpenAlex 学术数据爬虫
  - 爬取学术论文、作者信息、机构数据
- **`boss/`** - BOSS 直聘职位爬虫
  - 爬取招聘职位、公司信息、技能要求

###  `data/` - 数据存储（被 `.gitignore` 忽略）
- **`raw/`** - 原始数据
  - JSON、CSV 等原始格式数据
- **`processed/`** - 处理后的数据
  - 清洗、标准化后的数据文件
- **`sqlite/`** - SQLite 数据库文件
  - 本地关系型数据库存储

###  `database/` - 数据库操作层
- **`sqlite_client.py`** - SQLite 客户端
  - 处理 8 张基础表的 CRUD 操作
- **`neo4j_client.py`** - Neo4j 图数据库客户端
  - Cypher 查询语句封装与图操作逻辑

###  `models/` - 算法模型层
- **`kgat_ax/`** - KGAT-AX 图神经网络模型
  - `layers.py` - 全息嵌入层、Hadamard 积等模型层实现
  - `model.py` - 网络拓扑结构定义（PyTorch 实现）
- **`sbert/`** - 语义向量化模块
  - Sentence-BERT 封装（使用 `paraphrase-multilingual` 模型）

###  `scripts/` - 离线任务与预计算脚本
- **`build_index.py`** - 向量索引构建
  - 构建 Faiss 向量索引以支持快速相似度检索
- **`build_graph.py`** - 图数据库同步脚本
  - 将 SQLite 数据同步到 Neo4j 图数据库
- **`train.py`** - 模型训练脚本
  - KGAT-AX 模型的训练和评估

###  `web/` - 前端界面（Vue/React）
- **`src/`** - 前端源代码
  - `components/` - 可复用组件
    - 可视化证据链组件（使用 ECharts/D3）
  - `views/` - 页面组件
    - 搜索页面、结果展示页面
- **`package.json`** - 前端依赖管理

###  `tests/` - 测试目录
- 单元测试与集成测试文件

###  配置文件
- **`.env`** - 环境变量配置
  - 数据库密码、API Key 等敏感信息
- **`requirements.txt`** - Python 依赖包列表
- **`main.py`** - 项目主入口文件