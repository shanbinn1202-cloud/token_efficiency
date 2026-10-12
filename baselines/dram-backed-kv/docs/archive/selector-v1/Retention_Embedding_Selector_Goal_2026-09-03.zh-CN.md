# Goal 模式下一阶段：Retention 与 Embedding Selector

> 状态：已归档的 Selector V1 离线实验，不是当前 Online Selector V2。

日期：2026-09-03

## 执行决策（2026-09-03）

- 任务 A 只交付可工作的初版，不继续搜索“完美” retention policy；当前冻结为
  `grace_turns=2, score_margin=0, slack_pages=1`。
- 后续分析以任务 B 为主：在相同 Goal、预算和上述 retention 下回答 embedding 与 lexical
  哪个更可靠。
- 不因 retention 的小实现调整重复申请 GPU；仅保留一次已冻结配置的 H200 baseline/retention
  数值复验。
- Selector 结论扩展到全部 20 条 recorded trajectory。旧 10 条因结果已经可见，只作为
  development；新增 10 条在 Oracle 标注前按 resolution 分层和固定 SHA 排序冻结为 5 条
  validation、5 条 test。
- pure embedding 初版未稳定胜过 lexical，因此加入一个小范围、可审核的 hybrid RRF；只在
  development/validation 选择，新增 test 仅在配置冻结后读取一次。

## 总目标

在真实 recorded SWE-agent trajectory 上，使用同一份冻结的在线 Goal 流，依次完成：

1. 固定 Goal 与 lexical Selector，加入 retention slack/hysteresis，减少 Page active-set 抖动、
   reactivation 和 H2D churn；
2. 固定 Goal、retention 和预算，只替换 Selector，公平比较 lexical、embedding 与 hybrid retrieval。

本阶段参考 2.3.1 的“轻量语义向量做页级检索”思想，但实现服从本仓库现有接口和安全边界，
不直接复制其离线全上下文假设，也不使用 KV tensor 作为检索向量。

## 固定项：Goal mode

两项实验必须复用同一份 GoalRecord 序列，避免 retention 或 Selector 变化反过来改变 Goal：

- Goal constructor：当前默认 `MaintainedGoalConstructor`；
- conditional rewriter：Qwen3-0.6B，greedy constrained JSON decoding；
- Qwen3 输出失败：严格 validator 拒绝并使用 deterministic fallback；
- embedding drift：关闭；
- root Goal：不可变；
- 在线输入：仅最新有界 action/observation、当前 retrieval Goal 和紧凑状态；
- Oracle：只在选择完成后计算离线指标，不能进入 Goal、Qwen3 或 Selector。

先顺序生成并保存一份带 hash 的 Goal trace，随后所有 retention/Selector 分支只读这份冻结 trace。
这样比较的是“同一个 Goal 下怎样管理和选择 Pages”，而不是把 Goal 变化混入结果。

## GPU 资源硬约束

- 每次申请 GPU/H200 作业时，walltime 上限设为 **30 分钟**；不得提交超过 30 分钟的单个作业；
- 原始累计上限为 **2 GPU-hours**。2026-09-03 用户另行明确批准一次 20 分钟 H200 embedding
  作业 `21796759`，因此修订后的唯一上限为 **2 GPU-hours 20 分钟**；CPU Shadow 不计入；
- 先用 CPU Shadow 完成正确性检查和候选筛选，仅将少量冻结候选提交到 H200；
- 每次 GPU 运行必须记录 job ID、GPU 型号/数量、申请 walltime、实际运行时间和累计 GPU-hours；
- 在提交新作业前检查累计用量；剩余额度不足以安全完成时停止提交，不能通过增加作业数量绕过总上限；
- 作业失败、超时或重跑仍计入累计实际 GPU 使用量，并在实验报告中单独说明。

## 任务 A：Retention slack/hysteresis

### 要解决的问题

当前 Selector 每轮独立选 top Pages。预算边界附近的 Page 分数发生小幅交换时，Page 可能被
移出后很快重新选中，形成反复 release/restore。它既增加 reactivation，也可能让 H2D 流量高于
预期。

### 实现目标

在 lexical 分数和基础预算不变的情况下增加独立 retention policy：

- **active-set hysteresis**：旧 active Page 不因很小的分差立即退出；新 Page 必须超过明确
  score margin，或者旧 Page 连续若干 turn 未入选，才执行替换；
- **grace turns**：记录 Page 连续离开候选集的次数，只保存小型计数状态，不保存 trajectory；
- **hard-capped slack**：允许在 target budget 外短暂保留少量 Page，但设置不可突破的 hard cap；
- **HBM standby 与 attention active 分账**：如果实现 residency slack，暂存 Page 不进入当前
  attention context，只占有限 HBM buffer；恢复命中时才重新进入 SelectionPlan。

先比较少量配置，不做大网格：

1. baseline：无 retention；
2. 1-turn grace；
3. 2-turn grace；
4. score-margin hysteresis；
5. 最佳 hysteresis + 小幅 hard-capped slack。

### 评测

CPU semantic Shadow 报告：

- Recall@Budget（不能为了稳定性明显下降）；
- added/removed/retained/reactivated Page 数；
- turn-to-turn active-set Jaccard churn；
- 平均/峰值 active 和 standby Page 数；
- budget/hard-cap violation 必须为 0。

CPU Shadow 没有真实 KV bytes，只能筛选 churn/recall Pareto 候选。之后在少量 H200 cumulative
replay 上验证真实 H2D bytes/count、HBM residency、delta-NLL 和 top-1 agreement。

### 架构边界

Retention 位于 Selector 与 runtime 之间，输入 proposed ranking/SelectionPlan 和有限的上一轮
resident state，输出最终 SelectionPlan/residency plan。它不修改 Goal，不读取 Oracle，不把
retention 逻辑放进 `ShadowBackend` 或 Qwen runtime。

## 任务 B：Lexical 与 Embedding Selector

### 公平比较条件

任务 A 完成后冻结一个 retention 配置。两个 Selector 使用：

- 完全相同的冻结 Goal trace；
- 相同 mandatory Pages 和 recent safety quota；
- 相同 Page/token target budget、hard cap 和 retention policy；
- 相同历史 PageCatalog 与 stable logical-order output；
- 相同 development/validation/test 实例和 Oracle evaluator。

唯一变化是 Goal-relevant 历史 Page 的打分方式。

### Lexical Selector

沿用当前 `LexicalMetadataSelector`：

- files/tests/symbols/errors 做大小写归一化后的 typed exact match；
- Goal description/evidence 与 Page summary/metadata 做低权重 lexical overlap；
- mandatory、recent、budget 和稳定 tie-break 保持不变。

这是可审计、无模型依赖的 baseline。

### Embedding Selector

Embedding Selector 不扫描 KV，也不读取完整 trajectory。Page 被 seal 后，在 host/后台侧为它建立
一个 canonical retrieval document：

```text
role + bounded summary + files + symbols + tests + errors
```

然后用固定轻量 embedding backend 计算并缓存 Page vector，索引键为：

```text
page_id -> document_hash -> embedding_model_hash -> vector/embedding_ref
```

每个 turn 从冻结 `RetrievalGoal` 构造 bounded query document：

```text
description + files + symbols + tests + errors + evidence_needed
```

计算一次 query vector，与所有 eligible sealed historical Page vector 做 cosine similarity，按得分
降序选取。mandatory/recent 仍由公共 safety policy 处理；最后恢复为原始逻辑顺序，不能按相似度
顺序拼接 KV。

为保持可复现，必须记录 embedding checkpoint/revision、tokenizer、pooling、normalization、
dtype、document schema version 和输入 hash。Page vector 在 seal 时计算一次，不能每轮重复编码；
query vector 可按 Goal retrieval hash 缓存，KEEP 时直接复用。

本轮 Qwen3-0.6B metadata embedding 是离线质量对照，不代表正式 runtime 必须常驻独立 embedding
模型。正式方案可优先验证在主模型首次 prefill 时顺手 pooling Page hidden-state sketch，并在
mandatory/current-turn prefill 后生成 query vector；该复用-prefill 路径需单独实现和评测。

pure embedding 初轮已经完成但对 split 敏感。后续只增加两项受控变化：字段优先、长度有界的
v2 Page/Goal 文档，以及 lexical/dense 全候选排名的 weighted reciprocal-rank fusion。RRF 不把
lexical raw score 与 cosine 当作同一量纲，避免额外做分数校准；不引入可训练 reranker。

### Selector 指标

- primary：Oracle older-evidence micro/macro Recall@Budget；
- 每实例 Recall 和最差实例，避免总体平均掩盖失败；
- mandatory/recent invariant 和 budget violation；
- selection overlap、reactivation 和 churn；
- Page index build time、query encode time、similarity/search time；
- embedding index CPU/DRAM bytes；
- H200 候选再测 H2D、HBM、delta-NLL、top-1 agreement。

### 判定方式

Embedding 只有在固定预算下稳定提高 validation/test Recall，且延迟、索引内存和 churn 代价可控，
才算“效果得到验证”。只完成 vector 生成、cosine 排序和 Shadow 接线，仍只能称为“链路跑通”。

## 执行顺序与产物

1. 用 Qwen3-0.6B 生成并冻结 10-task Goal trace；
2. 实现独立 retention policy 与统计字段；
3. 在 CPU Shadow 上比较少量 retention 配置，冻结 Pareto 候选；
4. 实现带缓存和 hash provenance 的 Embedding Selector；
5. 在完全相同 Goal/预算/retention 下跑 lexical vs embedding vs hybrid；
6. 将 Oracle 扩展到全部 20 条，并按冻结的 `10/5/5` v2 split 做一次性确认；
7. 选择少量配置上 H200 cumulative replay；
8. 汇总并核验 GPU 作业账本，保证单次不超过 30 分钟、累计不超过修订后的 2 GPU-hours 20 分钟；
9. 输出原始 JSON/JSONL、配置/hash manifest 和中文版报告。

预期产物目录：

```text
results/raw/goal_mode_qwen3_10task_*/
results/raw/retention_shadow_10task_*/
results/raw/selector20_devval_v2_*/
results/raw/selector20_test_frozen_*/
results/raw/retention_selector_h200_*/
```

整个阶段继续保持：`GoalConstructor`、`Selector`、retention/`SelectionPlan`、`ShadowBackend`、
DRAM/Qwen runtime 解耦。
