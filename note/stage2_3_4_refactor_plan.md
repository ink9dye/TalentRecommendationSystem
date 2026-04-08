# Stage2 / Stage3 / Stage4 重构：修改纲领与详细计划

本文档说明本仓库中段标签链路（`label_pipeline`）的重构**纲领**与**分阶段实施计划**，与《Stage2 / Stage3 / Stage4 逐函数 Patch 级重构清单》一致，并映射到当前代码位置。**后续代码改造以本文档为权威参考**；字段命名、分层与验收以本文 **第二节「Stage2 数据契约（新版）」** 为约束，避免与外层 patch 清单出现口径漂移时以本文为准并回写对齐。

---

## 零、实施状态快照（与当前代码对齐，便于对照计划勾选）

> 下列条目随仓库演进更新；**Stage2 以第二节验收 2.7 为口径**。

### 0.1 Stage1（与重构纲领相邻、主链已稳定的部分）

| 项目 | 状态 | 现状说明 |
|------|------|----------|
| JD 早期粗分型 | **已完成** | `build_jd_coarse_roles_map` → `recall._jd_coarse_roles`，在 `extract_anchor_skills` 之前写入。 |
| 锚点分层修正 | **已完成** | `refine_stage1_anchor_layers`：主/辅/上下文、`final_anchor_score`（向量补充封顶、context 帽）、`anchor_type` 流程化修正。 |
| context_only 剥离 | **已完成** | `_strip_context_only_from_anchor_skills` → `recall._stage1_context_only_anchors`。 |
| `jd_profile` 查询 | **已完成** | `build_jd_hierarchy_profile`：多锚点单次 `IN` + Python 侧每锚 top-k，等价原 per-anchor Cypher。 |
| `Stage1Result` | **已完成** | `recall._last_stage1_result` 含 `jd_profile` 等，供调试与后续解耦。 |

### 0.2 Stage2（第二节数据契约）

| 第二节 / P0.1 要点 | 状态 | 现状说明 |
|-------------------|------|----------|
| **`run_stage2` 返回 `dict`** | **已完成** | 键：`all_candidates`、`anchor_to_candidates`、`candidate_graph`、`stage2_report`。 |
| **主链消费方式** | **已完成（契约层）** | `LabelRecallPath.recall` 将 **`stage2_output` 整份** 传给 **`run_stage3`**；Stage3 内部仍以 **`all_candidates`** 作为与旧逻辑对齐的扁平行输入；**`anchor_to_candidates` / `candidate_graph` / `stage2_report`** 经 **`run_stage3` 校验并透传**到 **`recall._stage3_stage2_context`**（及 debug），**尚未**用于 merge/rerank 中的图推理（见 0.3）。 |
| **2.3 字段分层** | **大部分完成** | `_expanded_to_raw_candidates`：provisional 写入 **`stage2_local_meta`**；顶层为证据类字段 + **`risk_flags`**、`landing_confidence` / `expansion_confidence`。 |
| **2.3 顶层禁止强 provisional** | **部分完成** | **仍保留 legacy 顶层镜像**（`primary_bucket`、`term_role` 等）以便 **`run_stage3` / `_merge_stage3_duplicates` 不改即跑**；与代码注释 TODO 一致，**待 Stage3 迁移后删除**。 |
| **2.4 `anchor_to_candidates`** | **部分完成** | **`_organize_anchor_to_candidates`** 产出带 `landing_candidates` / `expansion_candidates` / `summary` 的节点视图；**分组键**优先 `anchor_id`，缺省回退 **`anchor_term` / `parent_anchor`**（与第二节「key 必须为 anchor_id」的严格版相比仍为**过渡实现**）。**Landing/expansion 槽位**由 **`_rec_goes_to_expansion_slot`** 近似归类，非 2A/2B 显式字段直通。 |
| **2.5 `candidate_graph` MUV** | **已完成（首版）** | **`build_stage2_candidate_graph`**：四类边键齐全；门控与权重为保守近似，可非空（视数据而定）。 |
| **2.6 `stage2_report`** | **最小版已完成** | **`_build_stage2_report_min`**：`anchor_count`、`candidate_count`、`landing_candidate_count`、`expansion_candidate_count`、`source_distribution`、`per_anchor_stats`、`graph_edge_counts`。**尚未包含** 第二节表格中的 **`risk_flag_distribution`**（可并入后续迭代）。 |
| **2.7 验收 1–4** | **见上** | dict 与四键、分层与 meta 语义已落地；**验收 3 的「不得再作为顶层强逻辑」因兼容镜像未算完全净场**。 |
| **2.7 验收 5–6** | **部分对齐** | **未**单独实现 **`topk_landing_per_anchor` 参数化截断**；槽位为近似 split。 |
| **2.7 验收 8–9** | **已完成** | Stage2 不对外主契约输出 core/support/risky 分桶或终局 `paper_terms`。 |
| **`convert_stage2_candidates_to_records` 重命名**（P1） | **未做** | 仍以 **`_expanded_to_raw_candidates`** 为名；旧名 wrapper 未加。 |

### 0.3 Stage3 / Stage4（本文第四节 P0.2–P0.3 及以后）

| 项目 | 状态 | 说明 |
|------|------|------|
| **`run_stage3` 入参为完整 `stage2_output` dict** | **已完成** | 实现方式：`run_stage3` 要求四键齐全并做类型检查；从 **`all_candidates`** 得到与旧版等价的 **`raw_candidates`** 再走双闸门或 `_calculate_final_weights`。 |
| **`run_stage3` 透传 `candidate_graph` / `anchor_to_candidates`** | **已完成（透传层）** | 实现方式：写入 **`recall._stage3_stage2_context`** 与 **`debug_info.stage2_contract_pass_through`**；**全局 coherence 报告**中带 **`used_candidate_graph` / `used_anchor_to_candidates`**（表示结构非空，**非**「已在打分路径使用图」）。 |
| **`run_stage3` 在 merge/rerank 中消费边表做图重排** | **未做** | docstring 明示壳层迁移；**`_merge_stage3_duplicates` 等仍以扁平行证据为主**。 |
| **`run_stage3` 返回结构化 `dict`** | **已完成（壳层）** | 实现方式：稳定键 **`selected_core_terms`、`selected_support_terms`、`risky_terms`、`paper_terms`、`global_coherence_report`**，并保留 **`score_map` / `term_map` / …** 供 `label_path.recall` 与 Stage4 薄适配；`recall` 将 **`global_coherence_report`** 写入 **`debug_1["stage3_global_coherence_report"]`**。 |
| **`run_stage4` 按 P0.3 显式接收 Stage3 分桶输出** | **未做** | 仍为 **`run_stage4(self, vocab_ids, ...)`** 与既有论文检索形态；未接 **`selected_core_terms` / `global_coherence_report`** 等独立参数。 |
| P1–P3（merge 重命名、审计迁出等） | **未做** | 按计划排队。 |

**一句话**：**Stage2 四键 dict 与 Stage3 入参/出参壳层契约已对齐主链**；**图结构已挂接 recall 供后续迭代，尚未驱动 Stage3 打分**；**Stage4 仍为旧签名**，P0.3 待办。

---

## 一、修改纲领

### 1.1 职责边界（三层模型）

| 阶段 | 角色 | 一句话 |
|------|------|--------|
| **Stage2** | Candidate Generation | 按 anchor 生成小而干净的学术概念候选集，输出局部证据、风险与候选图结构；**不**将内部 provisional 判决当作下游事实（详见第二节） |
| **Stage3** | Global Coherence Rerank | 全局一致重排与 core/support/risky 裁决，以 Stage2 证据与图为输入，弱化 Stage2 局部 meta 的支配 |
| **Stage4** | Paper-level Evidence Validation | 论文检索 + 证据验证 + veto，而非仅「按 term 拉 paper + bonus」 |

### 1.2 本轮优先不做的事

- 不优先继续调一批 Stage3 常量或 support/risky 阈值。
- 不优先围绕 `paper_select_score` 打补丁。
- 不把 Stage2 实现成「更聪明的 primary 预裁决器」或在 Stage2 顶层堆叠终局式 primary 逻辑。
- 审计类 `_print_*` 不阻塞主链，主链稳定后再迁移。

**顺序**：先改结构与数据契约 → 再调评分公式 → 最后微调常量。

### 1.3 数据契约原则

本节与 **第二节 Stage2 新版协议** 配套阅读；Stage2 细则以下第二节为准。

1. **`run_stage2` 必须返回结构化 `dict`**（见第二节 2.2），不得仅以平铺 `list` 作为唯一对外契约；平铺全集仅作为 `all_candidates` 字段存在。
2. **Stage2 字段分层固定**：顶层仅保留标识、local evidence、risk、confidence 等证据类字段；一切强 provisional 语义统一下沉至 `stage2_local_meta`（见第二节 2.3），Stage3 **不得**将其当作 final fact 直接消费。
3. **`anchor_to_candidates` 与 `candidate_graph` 为 Stage2 正式产出**：前者按 `anchor_id` 分组并区分 landing / expansion 语义；后者提供集体裁决所需的最小边集（见第二节 2.4、2.5）。
4. **`stage2_report` 为结构化改造组成部分**：用于观测、截断与分布验收，非可有可无的杂项日志（见第二节 2.6）。
5. **Stage3 合并多锚证据**：`_merge_stage3_duplicates` 以多值证据聚合为主，避免单条 winning row 定义整行语义。
6. **Stage3 评分三段化**：`local_fit`、`cross_anchor_coherence`、`backbone_alignment`、`risk_penalty` 可分解、可观测（如 `stage3_score_breakdown`）。
7. **Stage4 显式验证**：`run_stage4` 接收 Stage3 的 term 集合与 `global_coherence_report`，输出 `paper_validation_report` 与结构化 summary。

### 1.4 兼容与风险控制

- 主入口短期内可保留旧名；重命名通过 **wrapper / 别名** 过渡。
- `run_stage2` 改为返回 `dict` 时，所有调用方需按第二节契约梳理；可先返回新结构、内部仍复用现有子流程，再逐步填满 `candidate_graph` / `stage2_report`。
- `stage3_term_filtering.py` 体积大，改动按「函数边界」切 PR，避免单次巨型 diff。

---

## 二、Stage2 数据契约（新版）

本节定义 **Stage2 唯一职责**、**`run_stage2` 返回结构**、**单条候选记录字段分层**、**分组与图**、**报告定位** 及 **验收标准**，作为实现与 Code Review 的硬约束。

### 2.1 Stage2 的唯一职责

Stage2 的唯一职责是：

> 为每个 anchor 生成一个小而干净的 **academic concept candidate set**，并输出局部证据、风险标记和候选关系结构；**不把**内部 provisional 判决当作下游事实。

#### Stage2 应该做

- **Landing candidates 生成**：在每个 anchor 下收敛一批高置信的「落点」候选（规模由 top-k 等参数约束）。
- **Expansion evidence candidates 生成**：在 landing 集合基础上扩展证据型候选，语义上与 landing 区分，**不得**与「唯一 landing 胜者」混为一谈。
- **Local evidence 计算**：在 anchor 局部上下文中可解释、可复核的相似度、fit、对齐等分数（见 2.3 顶层证据字段）。
- **Risk flags 输出**：可结构化消费的 risk / flags，供 Stage3 与审计使用。
- **Anchor 分组输出**：以 `anchor_id` 为 key 的 `anchor_to_candidates`（见 2.4）。
- **Candidate graph 输出**：为跨候选、跨 anchor 的集体裁决提供最小边集（见 2.5）。

#### Stage2 不应该做

- 最终 **primary** 拍板（全局主线由 Stage3 在一致性与 backbone 上下文中决定）。
- 最终 **reject / fallback** 裁决。
- 最终 **core / support / risky** 判定。
- 最终 **paper term** 编排或「给 Stage4 的终局列表」。
- 将 `primary_bucket` 等 **内部标签** 作为下游可直接采信的 **事实字段** 置于记录顶层。

---

### 2.2 `run_stage2` 的新版返回结构

**约定**：`run_stage2` **必须**返回 `dict`，**不得**仅以平铺 `list` 作为对外契约。

```python
{
    "all_candidates": List[Stage2CandidateRecord],
    "anchor_to_candidates": Dict[str, List[Stage2CandidateRecord]],
    "candidate_graph": Stage2CandidateGraph,
    "stage2_report": Dict[str, Any],
}
```

**语义说明**：

| 字段 | 语义 |
|------|------|
| `all_candidates` | 全量候选的**平铺列表**，便于 Stage3 做 merge、去重与全局遍历；元素类型为 `Stage2CandidateRecord`，与分组视图一致、可冗余。 |
| `anchor_to_candidates` | 以 **`anchor_id` 为 key** 的分组视图；每个 anchor 下包含该 anchor 的 landing 与 expansion 候选（二者在记录元数据中可区分，见 2.4），是 Stage2「按锚组织」的主结构。 |
| `candidate_graph` | 候选级关系图（见 2.5），供 Stage3 做跨锚一致性与支撑关系推理；类型为 `Stage2CandidateGraph`（实现可为 `TypedDict` 或与图构建函数一致的 dict 结构）。 |
| `stage2_report` | Stage2 **观测与验收**用结构化摘要：规模、分布、按 anchor 统计等（见 2.6）；与业务日志分离，属于协议的一部分。 |

---

### 2.3 `Stage2CandidateRecord` 的字段分层原则

Stage2 输出字段**必须分层**，避免把 Stage2 内部机造痕迹伪装成全局真值。

#### 顶层：证据字段（允许出现在 `Stage2CandidateRecord` 顶层）

仅保留下列类别：

- **标识**：如 `tid` / `term` / `anchor_term` / `candidate_source` 等与候选身份、来源相关的字段。
- **Local evidence**：如 `identity_score`、`sim_score`、`field_fit`、`subfield_fit`、`topic_fit`、与 JD/路径相关的局部对齐度量等（具体命名实现可与现有代码对齐，但**语义**须为「可复核的局部证据」）。
- **Risk**：如 `generic_risk`、`polysemy_risk`、`object_like_risk`，以及聚合后的 `risk_flags`（若采用列表或结构化 flags）。
- **Confidence**：如 `landing_confidence`、`expansion_confidence` 等（实现阶段可逐步落地）。

#### `stage2_local_meta`：强 provisional 语义统一下沉

以下字段 **不得**再作为顶层强逻辑字段；**必须**放入 `stage2_local_meta`（或实现上等价的嵌套命名空间）：

- `primary_bucket`
- `fallback_primary`
- `admission_reason`
- `reject_reason`
- `survive_primary`
- `stage2b_seed_tier`
- `mainline_candidate`
- `primary_reason`
- `parent_primary`
- `parent_anchor_final_score`
- `parent_anchor_step2_rank`
- `anchor_internal_rank`
- `can_expand_local`
- `role_in_anchor`
- `seed_block_reason`
- `has_family_evidence`

**原则（须写入实现与 Review 检查项）**：

> `stage2_local_meta` 只代表 Stage2 内部运行痕迹、provisional 判断与 debug 证据，**不能**被 Stage3 当作 final fact 直接消费；Stage3 若使用其中信息，仅可作为 **特征** 或 **弱先验**，且须经过 merge 与全局 rerank 的统一处理。

---

### 2.4 `anchor_to_candidates` 的定义

**Key 约定**：

- `anchor_to_candidates` 的 key **必须为** `anchor_id`（字符串或项目内统一 ID 类型，与 Stage1/PreparedAnchor 一致）。

**Landing 与 expansion**：

- 每个 anchor **默认**保留 **top-k landing candidates**（k 由 `topk_landing_per_anchor` 或等价配置定义）。
- **Expansion candidates 独立存在**：在数据模型或 `stage2_local_meta`/来源字段中可与 landing 区分；**不得**默认「单一 landing winner 驱动全部扩展」的世界观——扩展可依赖多 landing 或显式 provenance（见 2.5 `provenance_edges`）。

**排序语义**：

- **Local rank**（如 anchor 内排序）**不等于** Stage3 的 **final rank**；Stage2 若输出 rank 类信息，应落在 `stage2_local_meta` 或明确命名的局部字段中，避免与全局排序混淆。

---

### 2.5 `candidate_graph` 的最小可用版（MUV）

Stage2 **新增** `candidate_graph`，使 Stage3 的集体裁决有对象可依，而非仅依赖扁平行。

**最小可用版**须预先定义以下 **四类边**（实现上为四类列表或边表；首版允许某类边为空列表，但 **键/结构须齐备**，便于后续填满）：

| 键 | 作用 |
|----|------|
| `same_anchor_edges` | 同一 `anchor_id` 下候选之间的**竞争**关系（可替代或补充纯分数比较，使「同锚多强候选」显式化）。 |
| `cross_anchor_support_edges` | 不同 anchor 的候选之间的**支持**关系（跨锚一致性、互证）。 |
| `family_edges` | **Family / hierarchy** 上的相邻或从属关系，支撑主题族一致性推理。 |
| `provenance_edges` | **Expansion** 候选由哪一个或哪一类 **landing** 候选引出（扩展溯源，落实「expansion 不与 landing winner 混淆」）。 |

**说明**：MUV 的目标是 **协议与可观测性**：先有四类边与语义，再迭代边的密度与权重；**不得**以「暂时不用图」为由省略结构键。

---

### 2.6 `stage2_report` 的定位

`stage2_report` 是 Stage2 **观测与验收**的辅助结构，属于结构化改造的一部分，**不是**可有可无的临时 print 或杂项 debug。

**至少**应能支撑以下信息（键名实现可微调，语义须覆盖）：

| 内容 | 说明 |
|------|------|
| `anchor_count` | 参与本阶段处理的 anchor 数量。 |
| `candidate_count` | `all_candidates` 总规模（或等价全局计数）。 |
| `landing_candidate_count` | landing 侧候选总数或分层计数。 |
| `expansion_candidate_count` | expansion 侧候选总数或分层计数。 |
| `source_distribution` | 按来源（如 2a/2b/其它）的分布。 |
| `risk_flag_distribution` | 风险标记的分布概况。 |
| `per_anchor_stats` | 每 anchor 的 landing/expansion 计数、截断情况等。 |

用于：截断诊断、回归对比、与 Stage3 输入规模对账。

---

### 2.7 Stage2 验收标准

以下条件为 **Stage2 新版协议** 落地的**正式验收标准**（可与 CI/手工 checklist 对齐）：

1. **`run_stage2` 对外返回值为 `dict`，不得仅返回平铺 `list` 作为唯一契约。**
2. 返回值中 **必须** 包含：`all_candidates`、`anchor_to_candidates`、`candidate_graph`、`stage2_report`。
3. **`primary_bucket`、`fallback_primary`、`reject_reason` 不得再作为顶层强逻辑字段**；须已收拢至 `stage2_local_meta`（或协议约定的等价嵌套）。
4. **`stage2_local_meta` 的语义**符合 2.3 节原则：Stage3 不将其当作 final fact 直接消费。
5. **每个 anchor 默认保留 top-k landing candidates**（k 与配置一致且文档化）。
6. **Expansion candidates 与 landing 区分存在**，不与「单一 landing winner」混写为唯一扩展源。
7. **`candidate_graph` 结构**须包含四类边键：`same_anchor_edges`、`cross_anchor_support_edges`、`family_edges`、`provenance_edges`；**至少** `same_anchor_edges`、`cross_anchor_support_edges`、`provenance_edges` 在首版验收中可非空或可验证构建逻辑已接入（`family_edges` 允许首版为空列表但须占位）。
8. Stage2 **不直接输出** core/support/risky 分桶结果作为对外主契约字段。
9. Stage2 **不直接输出** 最终 `paper_terms` 或 Stage4 终局编排结果。

---

## 三、代码映射（本仓库）

| 模块 | 路径 | 与本文档的关联 |
|------|------|----------------|
| Stage2 | `src/core/recall/label_pipeline/stage2_expansion.py` | `run_stage2`、`_expanded_to_raw_candidates`（及重命名后的 convert）须符合 **第二节** |
| Stage3 | `src/core/recall/label_pipeline/stage3_term_filtering.py` | 入参为 **`stage2_output` dict**（四键）；**透传** `candidate_graph` / `anchor_to_candidates`；**merge/rerank 仍以 `all_candidates` 为主**（图尚未驱动打分）；返回结构化 dict（见 **0.3**） |
| Stage4 | `src/core/recall/label_pipeline/stage4_paper_recall.py` | 消费 Stage3 输出；不依赖 Stage2 顶层 provisional 字段 |

调用链与类型定义需全局检索：`run_stage2`、`run_stage3`、`run_stage4`、`_expanded_to_raw_candidates`。

---

## 四、详细修改计划

### P0：主入口与返回结构（最高优先级）

**目标**：三个 `run_stage*` 的入参/返回值与文档契约对齐；**Stage2 以第二节为唯一口径**。逻辑可先「薄封装」旧实现，再逐步内聚。

#### P0.1 `run_stage2`（`stage2_expansion.py`）

- **契约**：实现 **第二节 2.2–2.7** 的返回 dict 与验收标准；参数侧保留/新增：`topk_landing_per_anchor`、`topk_expansion_per_anchor`、`build_candidate_graph`、`return_stage2_report`（默认值与现有行为对齐）。
- **单条记录**：构造 `Stage2CandidateRecord` 时严格执行 **2.3** 顶层与 `stage2_local_meta` 分层。
- **实现路径**：landing_map / expansion_map → `all_candidates` + `anchor_to_candidates`；`build_stage2_candidate_graph` 填充 **2.5** MUV；`build_stage2_report` 填充 **2.6**。

#### P0.2 `run_stage3`（`stage3_term_filtering.py`）

- **目标契约（原文）**：`candidate_graph` 等进入 Stage3；**返回 dict**；内部阶段化且弱化对 Stage2 顶层 provisional 的硬依赖。
- **当前已落地（实现方式，与 0.3 一致）**：**单一入参 `stage2_output`**（内含四键，等价于「从 Stage2 带入图」）；**校验 + 透传**至 **`recall._stage3_stage2_context`**；**返回**含 **`selected_*`、`paper_terms`、`global_coherence_report`** 及 **`score_map` 等**；合并与双闸门仍以 **`all_candidates`** 为输入。**尚未**：在 **`_merge_stage3_duplicates` / 全局 rerank 中消费边表**；**尚未**：完全停止读取 legacy 顶层镜像（与 Stage2 遗留字段仍并存）。
- **仍待**：独立形参 `paper_term_limit` 等若需与 `run_stage2` 对齐再暴露；**迁出/弱化** paper 细编排与 Stage2 补锅逻辑。

#### P0.3 `run_stage4`（`stage4_paper_recall.py`）

- **显式入参**：`selected_core_terms`、`selected_support_terms`、`paper_terms`、`global_coherence_report`（及 recall 等）。
- **返回**：`paper_records`、`paper_validation_report`、`stage4_summary`。

**P0 验收**：全链路可运行；Stage2 满足 **2.7**；**Stage2→Stage3** 约定 dict 键已贯通；**Stage4 仍接 `label_path` 薄适配（vocab_ids、term_meta 等），P0.3 完整 dict 契约待续**。

---

### P1：关键中间函数与拆分

| 函数 | 动作概要（须与第二节一致） |
|------|---------------------------|
| `_expanded_to_raw_candidates` | 重命名为 `convert_stage2_candidates_to_records`（旧名保留 wrapper）。**输出即 `Stage2CandidateRecord` 语义**：顶层仅 2.3 证据字段 + `risk_flags`/confidence；**2.3 所列** provisional 字段一律入 `stage2_local_meta`；参数 `include_stage2_local_meta`、`emit_risk_flags`、`collapse_stage2_bucket_labels` 等。 |
| `_merge_stage3_duplicates` | `merge_local_meta_as_features`、`preserve_multi_anchor_evidence`；`stage2_local_meta_list` 多锚聚合；不把任一行的 `primary_bucket` 当作合并后唯一真值。 |
| `_check_stage3_admission` | 重命名为 `stage3_guardrail_filter`；不做 Stage2 bucket 强拒。 |
| `stage3_build_score_map` | 演进为 `stage3_global_rerank`；三段分数 + `stage3_score_breakdown`。 |
| `_assign_stage3_bucket` | 演进为 `assign_bucket_from_stage3_scores`；Stage3 分数主导，Stage2 local meta 仅特征。 |
| `_run_stage3_dual_gate` | 瘦身；非主排序、非 Stage4 总闸。 |
| `select_terms_for_paper_recall` | 拆为 `select_term_sets_for_output` 与 `prepare_terms_for_stage4`。 |

**P1 验收**：merge/convert 的固定样例符合 2.3；Stage3 单测不依赖已禁止的 Stage2 顶层字段。

---

### P2：新增函数与升格

**Stage2**

- `build_stage2_candidate_graph(anchor_to_candidates, jd_profile, active_domains)`：产出 **2.5** MUV。
- `group_candidates_by_anchor`、`build_stage2_report`（对应 **2.4、2.6**）。
- `collect_stage2_risk_flags`、`compute_landing_confidence`、`compute_expansion_confidence`

**Stage3**

- `stage3_local_guardrails`、`stage3_global_rerank`、`compute_local_fit`、`compute_cross_anchor_coherence`、`compute_backbone_alignment`、`build_global_coherence_report`、`select_term_sets_for_output`、`prepare_terms_for_stage4`

**Stage4**

- `validate_paper_against_stage3_output` 及 coverage/topic/risk 辅助函数；`_batch_jd_align_for_wids` 语义澄清；`compute_hierarchy_consensus_for_paper` 扩展 detail；`_clip_hierarchy_bonus_by_main_axis_mass` 兼容策略

**P2 验收**：`global_coherence_report` 与 `paper_validation_report` 字段稳定、可被审计模块消费。

---

### P3：审计与 Debug 迁移

- `_print_*` 迁至 `stage3_audit.py` / `stage3_debug.py`。
- 新 audit 优先消费 `global_coherence_report`、`stage3_score_breakdown`、`paper_validation_report` 及 Stage2 的 `stage2_report`。

**P3 验收**：主文件行数下降；调试可开关。

---

## 五、建议实施顺序（汇总）

1. **P0**：按 **第二节** 完成 `run_stage2` 契约 → `run_stage3` → `run_stage4` 与调用方。
2. **P1**：`convert_stage2_candidates_to_records`（原 `_expanded_to_raw_candidates`）与其它 Stage3 中间函数。
3. **P2**：图构建、coherence、Stage4 validation。
4. **P3**：审计模块迁移。

---

## 六、一句话总结

> **Stage2 按协议造候选与图（不判终局），Stage3 做全局一致裁决与分桶，Stage4 做论文级证据验证。**  
> 结构摆正后再调阈值与公式，避免在错误结构上堆补丁。

---

*文档版本：与仓库 `label_pipeline` 当前布局同步；**Stage2 以第二节为权威契约**；实施时函数签名与类型别名以代码为准，但语义不得弱化本节验收标准。*
