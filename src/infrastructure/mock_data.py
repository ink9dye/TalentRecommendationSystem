# src/infrastructure/mock_data.py

# 模拟人才列表数据，对应原本的 mockTalents
mock_talents = [
    {
        "id": "A001",
        "name": "张三",
        "institution": "某重点人工智能实验室",
        "h_index": 24,
        "citations": 1200,
        "match_score": 0.95,  # KGAT-AX 计算的精排分值
        "tags": ["图计算", "知识图谱", "推荐系统"]
    },
    {
        "id": "A002",
        "name": "李华",
        "institution": "科技大学计算机学院",
        "h_index": 18,
        "citations": 850,
        "match_score": 0.88,
        "tags": ["自然语言处理", "SBERT", "向量检索"]
    }
]

# 模拟异质网络图谱数据，对应原本的 mockGraph
# 包含“岗、词、作、人”等节点类型
mock_graph = {
    "nodes": [
        {"id": "job_1", "name": "高级算法工程师", "symbolSize": 50, "category": 0},  # 岗
        {"id": "word_1", "name": "协同过滤", "symbolSize": 30, "category": 1},     # 词
        {"id": "work_1", "name": "DeepICF 研究论文", "symbolSize": 40, "category": 2}, # 作
        {"id": "author_1", "name": "张三", "symbolSize": 40, "category": 3}       # 人
    ],
    "links": [
        {"source": "job_1", "target": "word_1", "label": "技能需求"},
        {"source": "word_1", "target": "work_1", "label": "语义关联"},
        {"source": "work_1", "target": "author_1", "label": "署名创作"}
    ],
    "categories": [
        {"name": "岗位需求"},
        {"name": "核心技能/词汇"},
        {"name": "学术作品/论文"},
        {"name": "候选人才"}
    ]
}