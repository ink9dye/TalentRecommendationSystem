from dataclasses import dataclass
from typing import Optional


@dataclass
class Work:
    work_id: str
    doi: Optional[str]
    title: str
    year: Optional[int]
    citation_count: int
    # source_id 已移除：现已迁移至关系表以支持更高效的“人才-刊物偏好”履历挖掘
    keywords: str  # 存储前 5 个关键词，用 | 分隔
    is_alphabetical: bool

    # --- 以下为推荐系统核心扩展字段 ---
    publication_date: Optional[str] = None
    abstract: Optional[str] = None  # 核心：用于 TF-IDF 提取研究方向
    concepts: Optional[str] = None  # 核心：知识图谱中的“概念/领域”实体节点
    type: Optional[str] = None  # 区分 Article, Book, Dataset，用于加权影响力
    language: Optional[str] = None  # 用于 NLP 预处理时的语言过滤
    concept_id: Optional[str] = None  # 该论文主领域的 ID，用于快速聚类


@dataclass
class AuthorProfile:
    author_id: str
    name: str

    # 推荐系统特征向量 (Feature Vector)
    influence_score: float = 0.0  # 综合影响力得分
    h_index: int = 0  # 学术阶位
    recent_activity: float = 0.0  # 基于历时性特征的时间权重分
    research_topics: str = ""  # 基于提取的研究标签 (|分隔)

    # 协同过滤与图谱特征
    main_field: str = ""  # 主攻领域
    collaboration_count: int = 0  # 合作者数量

    # --- 履历特征扩展 (可选) ---
    # 由于 source_id 已进入关系表，您可以在计算画像时增加如下特征：
    top_venue_ratio: float = 0.0  # 顶刊/顶会发文比例 (基于 authorships 统计)

    last_updated: Optional[str] = None