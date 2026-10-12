# Embedding 与 Hybrid Selector 构造说明

> 状态：已归档的 Selector V1 离线实验，不是当前 Online Selector V2。

日期：2026-09-04
状态：供设计审阅；描述当前已实现并完成 20-task CPU Shadow 的版本。

> **重要口径：**当前 20-task CPU Shadow 中的一个 “Page” 是一个完整的 completed SWE-agent
> turn，而不是正式 runtime 的 16-token 物理 KV Page。`build_semantic_shadow_payload()` 明确采用
> `one completed turn = one semantic Page`。本文第 2～5 节描述的是这个 semantic-Page 检索实验；
> 不能把结果直接表述成 16-token 物理 Page selector 已得到验证。

## 1. 模块位置与共同约束

两个 Selector 都位于 `GoalConstructor` 与 retention/runtime 之间：

```text
latest bounded action/observation
        → GoalConstructor（heuristic gate + 可选 SLM rewrite）
        → GoalRecord / RetrievalGoal
        → EmbeddingSelector 或 HybridSelector
        → RetentionSelector
        → SelectionPlan
        → ShadowBackend 或 DRAM/Qwen runtime
```

共同约束：

- 只读取当前 `GoalRecord` 和已经 seal 的 `PageRecord`，不读取完整 trajectory 或未来消息；
- 不读取 KV tensor；Oracle 只在 SelectionPlan 产生后计算 Recall；
- 先保护 mandatory Pages，再保留有限 recent-safety Pages，最后才用检索分数填剩余预算；
- 当前实验固定 `page_budget=8`、`recent_turns=2`、`recent_quota=2`；
- 选中集合最终按原始 logical position 排序后交给 runtime，不能按相似度顺序拼接 KV；
- 稳定 tie-break 使用 rank、logical position 和 page ID，保证相同输入可复现。

## 2. Embedding Selector

### 2.1 Page 文档

当前 recorded Shadow 为每个 turn-level semantic Page 构造一个有界、可 hash 的 v2
field-first 文档：

```text
document_type: SWE-agent historical evidence page
files: ...
symbols: ...
tests: ...
errors: ...
summary: ...                 # 最多 1,600 字符
role: assistant/tool/...
```

把 files/symbols/tests/errors 放在 summary 之前，是为了在 tokenizer 达到 `max_length` 时优先保留
高信号实体，避免长 shell 输出稀释或截掉结构化字段。文档不包含原始 KV，也不包含 Oracle 标签。
这个 canonical document 是本次实验为调用文本 encoder 引入的中间输入，不是用户原始设计要求的
物理 Page 附属对象，也不会被拼入模型上下文。

### 2.2 Goal query 文档

当前 `RetrievalGoal` 转成同一检索空间中的 query：

```text
query_type: Retrieve a relevant SWE-agent historical evidence page
errors: ...
tests: ...
files: ...
symbols: ...
evidence_needed: ...
task_context: ...            # description 最多 800 字符
```

`root_goal` 不会被 SLM 改写；query 读取的是 Goal 中面向检索的有界字段，而不是把 root Goal 变成
执行计划。

### 2.3 当前向量生成

当前离线实验使用 Qwen3-0.6B 的最后一层 hidden state。设 tokenizer 后有效 token hidden state 为
`h_1 ... h_n`：

```text
mean pooling:  v_raw = (h_1 + ... + h_n) / n
last pooling:  v_raw = h_n
L2 normalize:  v = v_raw / ||v_raw||_2
```

向量维度为 1,024。20-task 候选阶段同时比较了 mean 和 last；按照预先冻结的
development/validation 规则，最终 pure embedding 与 hybrid 都选择 last pooling。

这不是专门训练的 embedding checkpoint。它是在固定成本下验证“通用小 Qwen hidden state 是否
能提供 Page 语义排序”的实现，不能外推为所有 embedding 模型的结论。

### 2.4 Embedding 分数

Page vector 和 query vector 都已 L2 normalize，因此分数就是点积，也等于 cosine similarity：

```text
score_emb(page, goal) = clamp(q · p, -1, 1)
```

分数越大，Page 排名越靠前。分数只用于非 mandatory、非 recent-safety 的候选 Page。

排序键为：

```text
(-score_emb, logical_position_start, page_id)
```

在 Page budget 下从高到低填满，之后把已选 Page 恢复成 logical position 顺序。SelectionPlan 中会
记录 cosine、document schema、embedding model hash、document hash 和 cache entry 数。

### 2.5 增量缓存

正式语义是增量编码，不是每轮重算所有 Page：

```text
Page cache key  = (session_id, page_id, page_document_sha256)
Query cache key = (session_id, retrieval_goal_hash)
```

- semantic Page seal 后首次参与选择时编码一次；文档不变则一直复用；
- Goal gate 为 KEEP、`retrieval_goal_hash` 不变时直接复用 query vector；
- PATCH/REWRITE 只有在检索字段实际改变、hash 改变时才产生新 query vector；
- 离线实验使用 `PrecomputedTextEmbedder`，按文档 SHA-256 严格查冻结向量；artifact 另外绑定模型、
  revision、tokenizer、pooling、dtype、schema 和 corpus hash。

20 条 recorded trajectory 的 1,867 个唯一文档包括 1,187 个 **turn-level semantic Page** 文档和
680 个去重 Goal query，不是 1,187 个 16-token 物理 Page，也不是单条在线任务每轮需要处理
1,867 个 Page。

## 3. Lexical 分数（Hybrid 的一个输入）

Hybrid 复用现有 lexical 完整排名。设 Goal policy hint 中：

```text
s = structure_weight，默认 0.5
d = diversity_weight，默认 0.5
```

typed exact-match 分数为：

```text
typed = (12 × file_hits
       + 11 × test_hits
       + 10 × symbol_hits
       +  9 × error_hits) × (1 + s)
```

匹配经过 strip/case-fold，但要求完整实体字符串相等。自由文本部分从 Goal description、不可变
root Goal、evidence_needed 和所有 typed entities 抽 token，与 Page summary/metadata 求交：

```text
lexical = min(sum(matched_token_weight) × (0.75 + 0.25d), 8)
score_lex = typed + lexical
```

自由文本项被 cap 到 8，保证泛化词重叠不会压过一个 typed exact match。

## 4. Hybrid Selector

### 4.1 为什么不直接把两个 raw score 相加

`score_lex` 可能达到几十或更高，cosine 在 `[-1,1]`；直接线性相加需要额外校准，且容易对 20 条
小数据过拟合。因此当前实现只融合两个排序名次，使用 weighted reciprocal-rank fusion（RRF）。

对每个非 mandatory、非 recent Page，分别计算：

```text
r_lex(page) = Page 在 lexical 完整候选排名中的名次，从 1 开始
r_emb(page) = Page 在 embedding 完整候选排名中的名次，从 1 开始

score_hybrid(page)
  = w_lex / (k + r_lex(page))
  + w_emb / (k + r_emb(page))
```

当前冻结参数：

```text
w_lex = 3
w_emb = 1
k = 60
pooling = last_non_padding_hidden_state
```

例如 Page A 的 lexical/embedding rank 为 1/5：

```text
score(A) = 3/61 + 1/65 ≈ 0.06456
```

Page B 的 rank 为 3/1：

```text
score(B) = 3/63 + 1/61 ≈ 0.06401
```

因此 A 略高：当前配置让 lexical 主导，同时允许 embedding 把语义相关但没有精确实体命中的 Page
向前拉。RRF 同分时依次比较 lexical rank、embedding rank、logical position、page ID。

### 4.2 完整选择顺序

```text
1. 取当前 turn 之前所有 sealed historical Pages
2. 强制加入 mandatory Pages；若已超预算则 fail closed
3. 从最近 2 turns 中按新到旧加入最多 2 个 recent-safety Pages
4. 对其余 Page 分别形成 lexical 和 embedding 完整排名
5. 计算 weighted RRF，按融合分数填满剩余 Page budget
6. 将集合恢复为原始 logical order
7. 交给独立 RetentionSelector 施加 grace/standby slack
8. 输出并校验 SelectionPlan
```

Retention 当前固定为 `grace_turns=2, score_margin=0, slack_pages=1`。standby Page 可以短期驻留，
但不进入 attention active context。Hybrid 不读取 retention 内部状态，retention 也不重算 hybrid
分数，二者保持解耦。

## 5. 当前实验结果应如何理解

固定 test（5 条、85 条 evidence links）：

| Selector | Recall@Budget | active reactivation | 模拟 H2D Pages |
| --- | ---: | ---: | ---: |
| lexical | 31/85 = 0.3647 | 45 | 350 |
| pure embedding | 24/85 = 0.2824 | 162 | 457 |
| hybrid 3:1 | 33/85 = 0.3882 | 127 | 432 |

20 条冻结后汇总：lexical=`99/318=0.3113`，pure embedding=`81/318=0.2547`，hybrid
3:1=`109/318=0.3428`。

因此：

- pure embedding 当前不适合替代 lexical；
- hybrid 多找到了一些 evidence，但排序更易变化，reactivation/H2D churn 明显更高；
- lexical 是低成本、可解释默认项；hybrid 是 Recall 优先候选；
- 5-task test 很小，hybrid 的 test 净收益只有 2 条 evidence，不能宣称稳定大幅胜出；
- CPU Shadow 的 Page 次数不是实际 H2D bytes；真实 HBM/H2D 和生成质量还需 cumulative replay。

## 6. 正式 16-token 物理 Page 不应挂长文档

把本实验的完整 metadata document 复制到每个 16-token Page 是不正确的：

- 一个 turn 跨越许多物理 Page，会重复 files/symbols/tests/errors/summary；
- 16 tokens 本身语义过短，多个 Page 会得到相同或高度相似的 metadata；
- 控制面内存会随物理 Page 数线性增加；
- 当前 Shadow Recall 是按完整 turn evidence 标注的，无法监督一个 turn 内究竟是哪几个 16-token
  Page 真正必要。

正式主方案保持逐 16-token physical Page 打分：

```text
PhysicalKVPage
  ├─ page_id / logical positions
  ├─ DRAM KV address
  └─ prefill-pooled compact vector
```

每页直接 pooling 其 16 个 tokens 首次 prefill/decode 的 hidden states，再投影、归一化和量化；
Selector、预算、恢复和 retention 始终保持 Page 粒度。此前提出的 SemanticSpan 会把实际检索粒度
放大到 128–256 tokens，现已从正式主方案移除，只可在将来作为不改变最终 Page 打分的 ANN/coarse
index 加速结构。

### 6.1 复用 prefill

当前 Qwen3-0.6B metadata encoder 是 turn-level 离线质量对照，不代表正式系统必须再常驻一个
embedding 模型或在每个 16-token Page 下保存 metadata document。
可优先验证：

```text
Page 首次 main-model prefill
  → 对固定层 hidden states 做一次 pooling
  → 缓存 Page semantic sketch

当前 mandatory action/observation prefill
  → 得到 query sketch
  → Selector 排名，同时准备历史 Page 的 H2D
```

不能直接把带 layer/head/RoPE 结构的 K/V 当 cosine embedding；更合理的是保存一个固定层、位置处理
明确的 pooled hidden-state sketch。该路径尚未实现，必须与当前 turn-level metadata embedding
分开报告。

## 7. 审阅时建议重点确认

1. Page v2 文档是否还应进一步缩短 summary，或增加 tool/action 类型标签；
2. query 是否应继续包含 task context，还是只保留当前检索实体；
3. 正式默认选 lexical，还是接受更高 churn 采用 hybrid 3:1；
4. 真实 runtime 应优先尝试 main-model prefill sketch，还是常驻小 embedding encoder；
5. 下一批独立任务是否继续固定 `k=60, 3:1`，避免看过当前 test 后再次调参。

## 8. 对应代码与结果

- Embedding：`src/dram_kv/selectors/embedding.py`
- Hybrid：`src/dram_kv/selectors/hybrid.py`
- Lexical：`src/dram_kv/selectors/lexical.py`
- Retention：`src/dram_kv/selectors/retention.py`
- 文档语料：`scripts/prepare_embedding_corpus.py`
- 向量生成：`scripts/encode_embedding_corpus.py`
- 对比脚本：`scripts/run_selector_shadow_comparison.py`
- 候选冻结：`scripts/select_selector_candidates.py`
- 完整结果：`results/retention_embedding_selector_2026-09-03.zh-CN.md`
