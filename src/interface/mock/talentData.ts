// src/mock/talentData.ts
export const mockTalents = [
  {
    id: "A001",
    name: "张三",
    institution: "某重点实验室",
    h_index: 24, // 对应特征索引中的指标
    citations: 1200,
    match_score: 0.95, // KGAT-AX 计算的精排分值 [cite: 2]
    tags: ["图计算", "知识图谱", "推荐系统"]
  }
];

export const mockGraph = {
  // 模拟“人—作—所—源—词—岗”异质网络 [cite: 1, 3]
  nodes: [
    { id: "job_1", name: "算法工程师", category: 0 }, // 岗
    { id: "word_1", name: "协同过滤", category: 1 },  // 词
    { id: "work_1", name: "DeepICF 研究论文", category: 2 }, // 作 [cite: 2]
    { id: "author_1", name: "张三", category: 3 }    // 人
  ],
  links: [
    { source: "job_1", target: "word_1", label: "技能需求" },
    { source: "word_1", target: "work_1", label: "语义关联" },
    { source: "work_1", target: "author_1", label: "署名" }
  ]
};