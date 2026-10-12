# 基于 Prefill 复用的 16-token Page Selector

> 状态：当前 Selector V2 的基础设计；实际代码与运行参数以同目录的代码架构文档为准。

日期：2026-09-04
状态：V2 正式实现基线；检索、预算、传输和 retention 均保持 16-token Page 粒度。

## 目标

不额外常驻 embedding 模型，不给物理 Page 构造长 metadata document。每个 16-token KV Page 在
主模型首次正常 prefill/decode 时顺手产生一个 compact vector；Selector 直接给每个物理 Page
打分，并输出 physical Page IDs。

## Page 数据结构

```text
PhysicalKVPage
- page_id
- 16-token logical range
- DRAM KV address
- compact_vector / embedding_ref
- 少量 page-local handles（可选）
- residency / retention state
```

不存在必需的中间 `SemanticSpan`，也不保存 per-Page summary/document。若后续为了 ANN 或分层索引
增加 coarse group，它只能是加速结构，最终分数、预算和恢复仍以 physical Page 为单位。

## Page vector 如何免费产生

设一个物理 Page 的 16 个 token 在主模型某个固定层产生 hidden states
`h_1 ... h_16`：

```text
v_raw  = Pool(h_1 ... h_16)
v_proj = Project(v_raw)              # 固定投影到 128/256 维
v_page = Quantize(L2Normalize(v_proj))
```

- tool observation 在 prefill 时按 16 tokens 分页并收集 hidden states；
- assistant action 在逐 token decode 时累计，满 16 tokens 后 seal Page 并生成 vector；
- 不增加 Transformer forward，只增加 pooling、投影、归一化和量化；
- 不直接把分层、分 head、带位置处理的 K/V 当 cosine embedding；
- 不使用 input embedding，因为它缺少经过模型后的上下文语义。

初版建议比较固定层的 masked mean 与 last-token pooling，但只在 development/validation 选择一次。

## 最新 observation 如何成为 query

工具返回最新 observation `O_t` 后，先使用上一轮 active KV 对它做正常 prefill，同时获得它所有
Page 的 hidden states。除逐 Page vector 外，再对 `O_t` 的有效 hidden states做一次相同
projection/normalization，得到当前 query `q_t`：

```text
q_t = Normalize(Project(Pool(hidden(O_t))))
```

query 可以覆盖整个有界 observation 或固定长度 tail；候选仍然是独立的 16-token Pages，因此 query
较长不会改变 Page 级选择粒度。

随后 GoalConstructor 读取同一个 `O_t` 完成 KEEP/PATCH/REWRITE。Goal 不需要再做 embedding：

- semantic rank 使用 `q_t`；
- Goal 的 files/symbols/tests/errors/evidence_needed 继续产生 lexical/metadata rank；
- KEEP/PATCH/REWRITE 都不触发额外 embedding forward。

## 每轮时序

```text
1. 工具返回 observation O_t
2. 用上一轮 active KV prefill O_t
3. 每 16 tokens seal KV Page，同时保存 compact page vector
4. pooling O_t hidden states，得到 q_t
5. GoalConstructor 读取 O_t，更新结构化 Goal
6. Selector 对所有 eligible historical physical Pages 直接打分
7. 在 physical Page budget 下选 Page，应用 retention/standby
8. 从 DRAM 恢复所选 Page，按 logical order 组装
9. 生成下一条 assistant action
```

本轮新恢复的历史 Page 不会反向改变已经计算好的 `O_t` hidden/KV，但下一步 assistant tokens 可以
同时 attention 到 `O_t` KV 和新恢复的旧 KV。这是需要用 delta-NLL/top-1 agreement 验证的近似。

## Page 级打分

语义分数：

```text
score_sem(page) = cosine(q_t, v_page)
```

Lexical 不挂长文档。实现可以从该 Page 自己的 16-token slice 提取 page-local token/handle，或维护
`file/symbol/test/error handle → page_ids` 的紧凑倒排表。不能把整个 turn 的 summary 和所有实体复制
到该 turn 的每个 Page。

Hybrid 延续 rank fusion：

```text
score_hybrid(page)
  = w_lex / (k + rank_lex(page))
  + w_sem / (k + rank_sem(page))
```

启动配置可沿用 `w_lex:w_sem=3:1, k=60`，但此前的 20-task 结果是 turn-level semantic Shadow，只能
作为初值，不能视为 16-token Page 参数已经验证。

## 安全与预算顺序

1. 只考虑当前 turn 之前已 seal 的 Pages；
2. mandatory Pages 先占预算，超预算则 fail closed；
3. recent-safety quota 仍以 physical Page 数计；
4. 其余 Page 按 lexical、semantic 或 hybrid score 排序填满预算；
5. 同分使用 rank、logical position、page ID 稳定决胜；
6. 最终按原始 logical position 输出；
7. retention 的 active/standby 都以 physical Page 计数，standby 不进入 attention。

## 索引成本

逐 Page 语义选择必然至少需要一个 compact vector/Page，但不需要文档和第二个模型。例如 100 万
历史 tokens 对应 62,500 个 16-token Pages：

| Vector 格式 | 每 Page | 约总向量空间 |
| --- | ---: | ---: |
| 128-d int8 | 128 B | 7.6 MiB |
| 256-d int8 | 256 B | 15.3 MiB |
| 128-d fp16 | 256 B | 15.3 MiB |

还需计入 Page ID、logical range 和索引结构；这些必须与 DRAM KV 本体分账报告。

## 与现有实验的边界

现有 20-task CPU Shadow 明确采用“一条 completed turn = 一个 semantic Page”，Qwen3-0.6B 编码
metadata document。它验证了 lexical/semantic/hybrid 控制链路和 turn-level evidence Recall，未验证：

- 16-token Page hidden-state vector 的检索质量；
- Page-local lexical index；
- 逐 Page H2D、HBM、delta-NLL 和 top-1 agreement；
- selection-conditioned hidden representation drift。

因此正式实现不能直接复用当前 1,187 个 turn-level document/vector 作为物理 Page 索引。

## 版本边界与迁移

已有实验实现冻结为 **Selector V1**：它以 completed turn/metadata document 为语义候选，调用独立
Qwen embedding，并用于复现已经完成的 20-task Shadow 对比。V1 不删除、不原地改写，也不能把其
向量缓存或指标表述为 16-token Page 结果。此前讨论的 `SemanticSpan` 是未进入正式 runtime 的候选
设计，归档文件仅保留设计演进记录。

本文件定义 **Selector V2**：

- `PrefillPageEmbeddingSelectorV2` 和 `PrefillPageHybridSelectorV2` 直接排序 physical Page；
- `PageVectorPublisher` 在引擎侧发布 `page_id -> compact vector`；
- `PageVectorIndex` 保存向量、模型/层/pooling/projection provenance；
- 公共边界仍是 `select(context, goal, catalog) -> SelectionPlan`；
- 向量缺失、provenance 不匹配或引擎不支持时 fail closed 到确定性 lexical Selector。

实验 manifest 和 `SelectionPlan` provenance 至少记录 selector 名称/版本、16-token page size、vector
source、hidden layer、pooling、projection hash、selector config hash 与 frozen Goal trace hash。V1 和
V2 的 cache、结果目录及指标不得混用。

### GoalConstructor 冻结边界

V2 开发不修改 `GoalConstructor` 的行为或 `GoalRecord` 既有语义：Goal 不读取 hidden state、Page
vector、SemanticSpan、完整 trajectory 或 Oracle；Selector 结果也不反向影响 Goal 更新。开发与公平
比较优先只读同一份带 hash 的 frozen Qwen3 Goal trace。V2 的引擎依赖通过 Selector 构造时注入，
`Pipeline`、`ShadowBackend` 和 DRAM runtime 仍只交换已验证的 `GoalRecord`/`SelectionPlan`。

## vLLM 集成边界

vLLM 正常 generation forward 内部已经产生 hidden states，但公开 generate API 通常不返回整段
prefill hidden states。正式路径应在 model runner 内、forward 之后做 GPU-side Page pooling、固定低维
投影、归一化与量化，只异步发布 compact vector；禁止把完整 token hidden states 同步搬到 CPU。

实现需覆盖：chunked prefill 跨 step 的 Page `sum/count` 累积、prefix-cache 首次建块时随 KV 保存
vector、仅在 speculative decoding token 被 target model 接受后提交，以及 CUDA graph 下的预分配输出
buffer。Observation prefill 完成后需要一个受控调度点，先发布 query/Goal/SelectionPlan 并恢复历史
KV，再继续下一轮 decode。若无法修改引擎，后备顺序是 lexical/hashing、异步小 encoder；独立 vLLM
pooling request 仅用于验证，因为它会额外执行一次模型 forward。

## Git 与可复现性策略

1. 当前 Goal、Oracle、SLM、retention 和 turn-level Selector 成果形成只读 **V1 baseline commit**；
2. 给该提交打 annotated tag `selector-text-embedding-v1-shadow-20260904`；
3. V2 在 `feature/prefill-page-selector-v2` 分支实现，不直接覆盖 V1 类、cache 或结果；
4. 合并前必须通过 GoalConstructor 的既有回归测试，并增加 frozen Goal trace byte/hash 不变测试；
5. 只提交代码、配置、Oracle、可审核的小型 manifest/汇总；大向量、逐 turn trace、scheduler 日志等生成物
   由 `.gitignore` 排除，并通过 manifest 记录生成命令、输入 hash 和输出 hash；
6. V2 经 Shadow 和 H200 验证后再单独 tag，明确区分“链路跑通”与“效果得到验证”。

## 下一实现步骤

1. 从 V1 baseline tag 创建 `feature/prefill-page-selector-v2`；
2. 为 `PageRecord` 增加 compact vector provenance/ref，不增加长文本字段；
3. 在 Qwen prefill/decode bridge 中按 16-token logical Page 收集固定层 hidden states；
4. 实现 Page-local lexical/handle 倒排索引和 observation query pooling；
5. 让 Embedding/Hybrid Selector 直接输出 physical Page IDs；
6. 在 recorded H200 cumulative replay 上比较 lexical、prefill-vector、hybrid；
7. 链路和数值稳定后，再运行真实 10-instance closed-loop。
