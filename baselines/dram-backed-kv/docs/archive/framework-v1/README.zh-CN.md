# DRAM 支撑的 Goal 条件化 KV 工作集

> 已归档的框架/recorded-replay 长说明。当前 Online Selector V2 入口见仓库根目录 README。

本仓库是一个净室实现的 Stage 0 实验基础，用于研究这样一种方案：完整且已封存的
KV 历史由 CPU DRAM 支撑，GPU HBM 中只驻留由 Goal 条件化选出的工作集。

当前实验状态、冻结决策、原始结果目录与下一步统一记录在
`results/EXPERIMENT_PROGRESS_2026-09-03.zh-CN.md`。
当前 Embedding/Hybrid 的文档构造、打分公式、缓存和选择顺序见
`Embedding_Hybrid_Selector_Design_2026-09-04.zh-CN.md`。
正式 16-token KV Page 与 prefill hidden-state 复用方案见
`Selector_Prefill_Reused_Page_Design_2026-09-04.zh-CN.md`。

本框架刻意分离了三个关注点：

```text
TurnContext -> GoalConstructor -> GoalRecord
                                   |
PageCatalog -----------------> Selector -> SelectionPlan
                                              |
                                       RuntimeBackend
```

Goal Constructor 和 Selector 从不操作 GPU tensor。Runtime Backend 消费保存下来的
`SelectionPlan`，且不得内嵌检索策略。这样，同一份 plan 就可以在 Shadow、语义 replay
以及未来的 DRAM/HBM runtime 中复现。

当前 `feature/prefill-page-selector-v2` 的初步实现完成了 **CPU/Shadow 控制链路**：
`PageVectorPublisher`/`PageVectorIndex` 保存主模型 hidden-state 路径应发布的 compact vector，
`PrefillPageEmbeddingSelectorV2` 与 `PrefillPageHybridSelectorV2` 直接排序 16-token physical Page，
向量缺失、provenance/ref 不匹配或 Page 尺寸不兼容时整轮确定性退回 lexical Selector。CPU
模拟 publisher 只接收已经产生的 hidden states，用于验证 pooling、投影、归一化、int8 量化、
chunked accumulation 和 accepted-token 提交，不执行额外 Transformer forward。

`scripts/extract_prefill_page_vectors.py` 另提供离线真实模型 trajectory 提取器：单次加载主模型，
通过固定 decoder-layer hook 复用 teacher-forced prefill/decode hidden states，在 GPU 上直接汇聚并
投影为 16-token Page vector，只把 compact int8 vector 写给 CPU replay。它用于链路和 proxy
实验，不是 production model-runner publisher，也不是 selection-conditioned 执行。

这不表示 fresh closed-loop 效果已经验证：真实 model-runner publisher、精确 16-token Oracle 检索质量
以及在线 observation-query 的 selection-conditioned drift 均仍待后续实验。

`scripts/run_cumulative_oracle_sweep.py --selection-policy prefill_v2` 已进一步接通 recorded H200
enforcement：`SelectionPlan` 实际控制 DRAM restore/bind 和 selection-conditioned Qwen forward，并保留
跨 turn 未满 16 tokens 的 live tail。它可测 delta-NLL、top-1 agreement 与 managed-HBM/H2D；当前 query
仍来自带 hash 的离线主模型向量 artifact，在线 model-runner publisher 和 fresh closed-loop agent 仍待接入。

## 按模块理解设计

这一节从“每个模块负责什么、接收什么、产出什么”解释整条链路，不要求读者了解具体代码。
最重要的边界是：Goal 只负责表达“为了检索历史，现在需要什么证据”，Selector 决定选哪些
Page，Runtime 只执行已经确定的选择；三者不能互相隐藏策略。

### 1. Schema 与 session Goal 状态

`src/dram_kv/schema.py` 定义模块之间共同使用的数据契约。输入侧的 `TurnContext` 描述当前
session、任务、turn、最新 action/observation 和已经封存的历史 Page；输出侧的
`GoalRecord` 同时保存不可变的 `root_goal`、可维护的 `RetrievalGoal`、策略提示、gate
decision、trigger 和审计信息。`RetrievalGoal` 包括检索描述、阶段以及 files、symbols、
tests、errors、evidence 等实体，它不是任务执行计划。

每个 session 的可变状态由 `src/dram_kv/goals/maintained.py` 独立保存。第一次 boundary 会
冻结 root；之后如果同一个 session ID 换了 instance、root objective 或出现倒序 turn，系统
会直接拒绝，而不是悄悄继承其他 session 的状态。所有可能影响 Selector 的 Goal 字段都进入
稳定的 canonical hash；decision、latency、counter 等纯审计字段不会改变 selector hash。

### 2. 确定性特征抽取

`src/dram_kv/goals/features.py` 的输入只有最新 action 和最新 tool observation，不读取完整
trajectory。它通过有界、无第三方依赖的规则，分别抽取文件、符号、测试、错误、stack frame、
return code、action 类型和测试结果。action 与 observation 的 provenance 始终分开，避免把
工具打印的文本误认为 agent 行为；普通 observation 也不能添加硬约束。输出是尺寸固定上限、
可以放进 JSONL trace 重放检查的 `FeatureSnapshot`。

### 3. Phase、stuck 统计与滞回

`src/dram_kv/goals/state.py` 把特征转换为 exploration、hypothesis、implementation、
verification 或 stuck。这里没有分类模型，只有按优先级执行的规则：

| 原始 phase | 判定信号（从上到下，先命中者优先） |
|---|---|
| `implementation` | action 被识别为编辑，例如 `apply_patch`、`sed -i`、编辑工具；编辑优先级最高，即使 patch 文本中出现测试命令也仍算 implementation |
| `verification` | action 是 pytest/unittest/cargo test 等测试命令，或 observation 已包含测试通过/失败证据 |
| `hypothesis` | action 是 gdb/lldb/strace/debug/git diff 等诊断动作，或抽到了确认过的 error/stack top frame |
| `exploration` | action 是 cat/rg/find/ls/sed -n 等只读检查 |
| 沿用当前 phase | 没有命中上述信号；若刚从 `stuck` 恢复则以 exploration 为候选 |

工作 phase 的顺序是 `exploration < hypothesis < implementation < verification`。向前推进立即
接受；回退或从 `stuck` 恢复需要同一个候选 phase 连续出现默认 2 个 boundary，才通过
hysteresis。这里的 **boundary** 就是 Goal constructor 获得一次新 turn 输入并更新状态的时点，
通常位于“上一条 assistant action 已执行、对应 tool observation 已返回、下一次模型调用将开始”
的位置，不是 Page 边界。

stuck 也是确定性规则。默认先至少观察 10 个 boundary（`stuck_floor=10`），然后任一计数器
达到阈值即触发：连续非零 return code 5 次；同一文件自上次计数器重置以来累计编辑 6 次；
自最近一次 edit 后累计 15 个 inspect（中间的其他 action 不增加也不清零）；或归一化 action
连续相同 8 次。action 归一化会
转成小写、合并空白，并把引号字符串和数字替换成占位符，所以只改变测试编号或字符串参数的
重复命令仍可能被视为同一 action。计数器在阈值处饱和；文件表最多保存 32 项。触发 stuck
时无需等待滞回，立即进入 `stuck`，下一 boundary 前清空 stuck 计数器，并进入 8-boundary
cooldown，期间不重复报警。

稳定 phase 会映射成 `GoalPolicyHint`。例如 exploration/stuck 给更大的建议 token budget 和
更高 diversity，implementation 给较小 budget、低 diversity、高 structure，verification
居中。该 hint 会参与 selector hash；但目前 `LexicalMetadataSelector` 只实际使用其中的
`diversity_weight` 与 `structure_weight` 调分，`active_token_budget`、`slack` 和 `priorities`
仍是给后续 runtime policy 的建议值，不会自动覆盖命令行传入的 Page/token budget。

### 4. KEEP/PATCH/REWRITE gate

`src/dram_kv/goals/maintained.py` 汇总确定性实体变化、phase/stuck 状态、错误签名、显式用户
请求以及可选 drift 信号。它不是打分分类器，而是下列短路优先级规则；只采用第一个命中的
分支：

1. 收到可信 user/system 的 `explicit_requirement`：`REWRITE(user_request)`。
2. phase tracker 报 stuck：`PATCH(stuck_recovery)`，确定性加入 recent successful state、
   failed attempts、verification records 和 root task anchor，不调用 SLM。
3. 可选 embedding drift 报警：`REWRITE(semantic_drift)`。
4. 出现新的 error type，或 stack top frame 相比上次改变：
   `REWRITE(hypothesis_invalidated)`。
5. 最新非空实体与 Goal 的已有实体完全不相交，连续达到默认 2 个 boundary：
   `REWRITE(goal_invalid)`。
6. 测试通过：`PATCH(evidence_satisfied)`；blocked 后 return code 恢复为 0：
   `PATCH(blocker_resolved)`；非零 return code 首次需要失败诊断：
   `PATCH(execution_failed)`。
7. Goal 还没有实体而本轮发现实体：`PATCH(entities_discovered)`；新实体与已有实体既有交集又有
   扩展：`PATCH(entities_extended)`。
8. 其他情况：`KEEP(no_boundary)`。

`KEEP` 不改 `RetrievalGoal`，但 phase 导致的 policy hint 仍可能变化；`PATCH` 只做有界字段更新：
files/symbols 最多 32 项、tests 24 项、errors 16 项，并按 trigger 修改 status、evidence 或经
认证的 constraint，不自由改写 description。`REWRITE` 才请求生成模型重新表述 selector 的
检索焦点。注意 stuck 在当前实现中故意走模板化 PATCH，而不是生成式 REWRITE。

这里还存在一个不要与 phase 混淆的 `RetrievalGoal.status`：初始化是 `active`；
`stuck_recovery`、`hypothesis_invalidated` 或 `execution_failed` 的确定性 fallback 会置为
`blocked`；测试通过置为 `done`；blocked 后 return code 为 0 再回到 `active`。schema 还允许
`verifying`，但当前确定性 PATCH 没有主动进入它，只可能由通过校验的 rewriter candidate 给出。
稳定 phase 描述“agent 正在探索/实现/验证什么”，status 描述“当前检索 Goal 是否受阻/完成”，
两者是正交字段。

trace 同时保存 `gate_decision` 和 `applied_decision`。例如 gate 请求 REWRITE，但没有配置
rewriter、子进程失败或输出未通过校验时，会按同一 trigger 做 deterministic PATCH；如果 PATCH
也没有产生字段变化，则实际结果是 KEEP。因此“请求过重写”和“模型重写真的生效”不会混为一谈。

### 5. 可插拔 rewriter 与失败保护

`src/dram_kv/goals/rewriter.py` 定义统一的 `GoalRewriter` 边界。模型只会收到不可变 root、
当前 retrieval Goal、最新且截断到上限的 action/observation、特征、trigger 和紧凑统计，
不会看到完整 trajectory。输出必须通过严格 schema 校验；返回 root、plan、steps、commands
等受保护字段，返回未知字段、重复/越界实体或非法约束，都会被拒绝。失败时 gate 使用
deterministic PATCH/KEEP fallback，root 不会被模型改写。仓库提供 deterministic fake 供
测试，以及 command adapter 供真实模型接入，核心包本身不依赖模型库。

当前 `scripts/qwen_goal_rewriter.py` 发送给模型的 system prompt 原文如下：

```text
You rewrite a selector-side retrieval Goal. Return exactly one JSON object. It is not an execution plan. Never return root_goal, commands, steps, reasoning, or unknown fields. Allowed fields: description, phase_hint, files, symbols, tests, errors, evidence_needed, status, constraints. Preserve constraints unless trigger is user_request and allow_constraint_updates is true. Prefer concise existing entities.
```

user prompt 是一个紧凑 JSON object，字段为 `root_goal`、`current_goal`、`latest_action`、
`latest_observation`、`features`、`trigger`、`stats` 和 `allow_constraint_updates`。核心边界允许的
action/observation 上限分别为 4,000/8,000 字符；这个 Qwen wrapper 为照顾小模型又缩到
1,000/2,000 字符。它不接收完整 trajectory、执行计划或仓库文件。

可选 backend/模型如下：

| 选项 | 用途 |
|---|---|
| `DeterministicFakeRewriter` | CPU 单元测试和可重复 replay；不是真模型 |
| `Qwen2.5-0.5B` | 已跑过的 baseline；现有 wrapper 可直接加载，但完整轨迹只有 6/11 次输出通过校验 |
| [`Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) | 最适合先测的近似 drop-in 候选；仍使用 `AutoModelForCausalLM`，但应在 chat template 中显式设置 `enable_thinking=False` |
| [`Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 新一代候选，默认 non-thinking；官方 checkpoint 使用多模态 processor/model 接口，需要改 wrapper 或通过常驻 API 接入 |
| [`Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 若将“1.5B 以下”按不超过 1.5B 计算，可作为同模型族的容量上界；现有 wrapper 可直接加载 |
| [`Gemma 3 1B IT`](https://huggingface.co/google/gemma-3-1b-it) / [`LFM2-1.2B-Tool`](https://huggingface.co/LiquidAI/LFM2-1.2B-Tool) | 跨模型族对照；前者需接受 Gemma license，后者偏向简洁 tool call，均应先验证本项目 JSON schema 的有效率 |
| 自定义 command | 只要实现相同 JSON stdin/stdout 协议，也可以封装常驻服务或其他模型；核心包无需依赖其 SDK |

validator 不评价文本“聪不聪明”，只检查安全与结构：candidate 只能含 RetrievalGoal 的 9 个
字段；禁止 root、plan、steps、commands、reasoning 等字段；description 必须非空且不超过
1,200 字符；数组每项必须是非空字符串、每字段最多 64 项、单项最多 512 字符且最终不能重复；
phase/status 必须属于 schema 枚举；constraint 只有 `user_request` 且显式授权时才能改变。
candidate 缺失的合法字段继承当前 Goal。超时、非零退出、非法 JSON、未知字段或 schema 错误
都会 fail closed，再由 gate 执行确定性 PATCH/KEEP fallback。

### 6. Optional embedding 与 EWMA/CUSUM 累计漂移

`src/dram_kv/goals/drift.py` 是可选旁路。`TextEmbedder` 可以接无依赖的 signed-hashing
实现，也可以通过 JSON command adapter 接真实 embedding 模型。每个 boundary 的当前焦点
向量会与更新前的 EWMA centroid 比较：单次 cosine distance 捕捉突变，CUSUM 累加多个不够
大的连续偏移；min-samples 和 cooldown 抑制早期误报与连续触发。它只向 gate 提供可审计的
drift snapshot，不直接修改 Goal。首轮 astropy 单轨迹上，hashing 默认参数新增 7 次
`semantic_drift` rewrite 且没有提高 Recall，所以该模块默认关闭，不能把这次结果理解成
embedding 已带来收益。

### 7. Lexical、Embedding 与 Hybrid Page Selector

`src/dram_kv/selectors/lexical.py` 真正消费 RetrievalGoal 中的 files、symbols、tests、errors、
evidence 和 Page metadata。它先保护 mandatory task anchor 与 recent safety quota，再在预算内
按稳定的 lexical/metadata 分数选择历史 Page，最后恢复为原始逻辑顺序，产出
`SelectionPlan`。因此 Goal entity 的变化会实际改变被选 Page，而不只是改变日志文字；同时
相同输入的选择结果可重复、可审计。

Selector 不读取 K/V tensor 本身。同一个 `page_id` 连接两张职责不同的表：控制面的
`PageCatalog` 保存 `page_id -> PageRecord`，其中只有 token/逻辑位置、turn、role、summary 和
files/symbols/tests/errors 等检索元数据；runtime 则另存
`page_id -> CPU K/V payload`。前一张表在 recorded 消息完成 tokenizer/chat-template 渲染并按
`page_size` 划分逻辑 Page 时建立，后一张表要等该 turn 的真实 Qwen forward 完成、K/V slice 被
`offload_sealed()` 后才逐页建立；纯 Shadow 只有前一张表。selector 先给 `PageRecord` 打分得到
Page ID，runtime 再用这些 ID restore 对应 K/V。

`src/dram_kv/selectors/embedding.py` 已实现按 document/model hash 缓存的 Page/Goal vector 与
cosine ranking；`hybrid.py` 用 weighted reciprocal-rank fusion 合并 lexical/dense 排名。这里的
Page retrieval embedding 与第 6 节默认关闭的 Goal drift 是两条独立链路。20-task 固定 test 上，
pure embedding Recall=`0.2824`，低于 lexical=`0.3647`；冻结的 hybrid 3:1 为 `0.3882`，但
reactivation/H2D Page 次数更高。完整条件、split 和限制见
`results/retention_embedding_selector_2026-09-03.zh-CN.md`。

设 `s=policy_hint.structure_weight`、`d=policy_hint.diversity_weight`，某页的分数为：

```text
typed = (12 * file_exact_hits
       + 11 * test_exact_hits
       + 10 * symbol_exact_hits
       +  9 * error_exact_hits) * (1 + s)

lexical = min(sum(matched_query_token_weight) * (0.75 + 0.25 * d), 8)
score = typed + lexical
```

typed match 是大小写折叠后的**完整实体字符串精确匹配**。lexical token 用正则
`[A-Za-z0-9_./:$-]+` 从 Page 的 summary 与 typed metadata 中抽取，再与 query token 求交集；
query 中 description 权重 1.0、root 0.5、evidence 1.25、各 typed entity 1.1，同一个 token
若来自多个字段只取最大权重。lexical 总分封顶 8，刻意低于一个 typed exact hit，让文件名、
测试名等结构化 handle 主导泛化文本重合。

选择顺序是：mandatory 全保留；在剩余容量内从最近 `recent_turns` 中最多保护
`recent_quota` 页；其余页按 score 从高到低装入 Page/token budget。零分页也可能作为剩余
容量的 filler 被选中。并列时按原始逻辑位置、Page ID 稳定打破；选完后再恢复成原始逻辑顺序，
因为 attention 上下文不能按相关性重排。

### 8. Recorded SWE-agent Shadow replay 与评测

`src/dram_kv/replay/swe_agent.py` 是 **recorded trajectory ingestion adapter**：它读取并校验一条
已经完成的 SWE-agent trajectory/Oracle bundle，把相邻的真实 assistant action、tool
observation、return code 和历史 Page 按顺序恢复成内部 `TurnContext`。它不重新启动
SWE-agent、不执行命令，也不产生新的环境 observation。完整离线链路是：

```text
完整 SWE-agent trajectory
        ↓ recorded trajectory adapter
按顺序恢复 TurnContext
  - previous action
  - current observation
  - return code
  - historical Pages
        ↓
Goal constructor + selector
        ↓
Shadow trace + Recall@Budget
```

`ShadowBackend` 会真正运行 Goal constructor 和 selector，但不 restore KV、调用任务模型或改变
dense context；它只把选中 Page 与 Oracle evidence label 比较。Oracle 的 older-turn selection
只用于离线 Recall 标签，绝不进入 Goal/selector 输入，因此这里测的是“若当时使用这套控制面会
选择哪些旧历史”，不是新的 agent rollout。trace/summary 记录 decision、trigger、Goal update、
SLM 调用成本、root invariance 和 Evidence Recall@Budget；它适合低成本调规则和阈值，但不能
证明真实 KV 数值质量、HBM 节省或 serving latency。

### 9. Recorded token span 与逻辑 KV Page 构造

`src/dram_kv/replay/recorded_loop.py` 负责把 recorded 消息稳定地接到 token-level KV replay。
主 Qwen 的 tokenizer/chat template 先把消息渲染成固定 token IDs；每个 `RenderedSpan` 保存一段
连续 token、来源 message ID、role，以及从该消息抽取的 summary/files/symbols/tests/errors。
随后 `build_page_catalog()` 按 `page_size` 把 span 切成逻辑 `PageRecord` 并传播这些元数据；真实
Qwen forward 后，代码再从 cache 中切出相同逻辑范围的 K/V，按 Page ID 封存到 DRAM。

这确实是因为当前还没有接入 vLLM 式 block-native PagedAttention：仓库手动实现了 Page 粒度的
标识、选择、K/V 切片和 DRAM/HBM 搬运，但 restore 后仍把 selected Pages 拼成连续 K/V tensor
交给标准 attention。CPU Shadow 使用的 semantic Page 则只是“一条 completed turn = 一页”的
抽象元数据单元，没有真实 K/V；真实 token Page 通常是 128-token 切片。两者共用 selector API，
但只有后者能测 KV 数值、搬运和显存。

### 10. Qwen Goal Rewriter command backend/wrapper（可选）

这一模块只负责把**外部小模型接成 Goal rewriter**，和第 8 节读取 trajectory 的 recorded adapter
无关。核心侧保留一次性 `CommandGoalRewriter`，并新增复用同一子进程的
`PersistentCommandGoalRewriter` NDJSON 协议；`scripts/qwen_goal_rewriter.py` 是具体的
Transformers/Qwen wrapper，使用 schema-constrained generation，最终仍由核心 validator 验收。

```text
REWRITE trigger -> persistent command -> constrained Qwen -> candidate JSON -> validator
```

它是可选实验模块，不是 selector、Shadow 或 DRAM runtime 的前置条件：未配置或输出失败时，
gate 会降级为 deterministic PATCH/KEEP。它只在可选初始化和 hard REWRITE trigger 上调用，
不是每轮生成 Goal，也不是执行 SWE-agent action 的主模型。现有 Qwen2.5-0.5B/H200 实验 11 次
调用仅 6 次通过校验；one-shot wrapper 每次冷启动，p50 19.8 秒。常驻/约束解码代码已经接入，
但这组历史数字不能用于证明新路径有效；仍需在固定请求集上分别实跑两个小模型。

### 11. Qwen/DRAM cumulative replay

`scripts/run_cumulative_oracle_sweep.py` 把以上控制面接到已有 Qwen/DRAM token-level runner。
每个 turn 的顺序固定为：构造 Goal、选择并形成 plan、DRAM restore/bind、执行模型、封存新 Page。
`goal` branch 使用在线 Goal + lexical/metadata selector；`oracle` branch 保留为会读取未来标注的
离线上界，两者不会混用。每个 branch 有独立 runtime 状态，便于与 full-active 做相同 recorded
token 的数值对照。

首轮 H200/Qwen3.6-27B cumulative 实验已经跑通全部 48 turns，但它目前只是集成与诊断
基线：40% Goal branch 的平均 top-1 agreement 为 0.9370、平均 delta-NLL 为 +0.1574，并产生
125 次 reactivation。也就是说，链路和 HBM 驻留控制已经得到验证，selector 质量与 retention
churn 仍需优化，不能把这些结果表述为质量收益。

这个 replay **会用 GPU 跑真实主模型**。checked-in 结果使用 H200 上的 Qwen3.6-27B；CPU
DRAM 保存封存的 full-attention K/V，选中的 Page 被 restore 到 GPU，模型 forward 后新 K/V
再封存回 DRAM。tokenization 一般在 CPU 完成，但主 forward/KV 计算在 GPU。

不过它是 **teacher-forced recorded replay，不是新的 agent rollout**，具体过程是：

1. 用同一个 Qwen tokenizer/chat template 渲染整条 recorded messages，并验证相邻前缀的
   token IDs 完全一致；由此切出每轮的 current-observation prefill token 和 recorded
   assistant-action token。
2. Goal constructor 在 turn `t` 看到 recorded 的上一 action 与当前 observation；selector 只在
   已封存的 `t` 之前历史中选 Page。
3. runtime restore/bind 选中历史 K/V。随后代码不是调用自由采样的 `model.generate()`，而是把
   “本轮 prefill token + recorded action token”一次 teacher-force 给 Qwen forward。
4. Qwen 的 logits 与同一 recorded action token 作 NLL，并与独立的 full-active branch 比较
   top-1 agreement/logit delta；模型自己最想生成的 action 不会被采用。
5. 从这次真实 forward 的 cache 中截取本轮 token 对应的新 K/V Page，封存后进入该 branch
   下一轮。下一 observation 仍直接取 recorded trajectory，不会执行模型输出去查询环境。

默认只有 canonical prefill 超过 `--switch-threshold`（当前默认 8,192 token）后才开始稀疏
选择；此前 selected branch 也走 full-active，用来建立共同且稳定的早期 cache。

所以“把 replay 的 observation 和 action 喂给 agent”更准确的说法是：把它们的 canonical
tokens 喂给任务模型做 teacher-forced forward，以测选择性历史造成的 logits/KV 差异；并没有
让模型自主决定 action，也没有真实运行 shell/environment。每个 ratio/full-active branch 都有
独立、连续的 runtime/KV 状态；Goal 分支后续 Page 的 KV 是在该分支选中历史条件下重新算出的
`selection_conditioned` KV，不是从 dense branch 偷来的。

最后还要区分两个 Qwen：`scripts/qwen_goal_rewriter.py` 的 0.5B/0.6B 小 Qwen 负责可选 Goal
重写；cumulative replay 的 27B Qwen 是被评测的任务模型。cumulative `goal` branch 现在可通过
`--slm-command ... --slm-persistent` 注入同一个常驻 rewriter；省略该参数时仍安全降级为
deterministic PATCH/KEEP，便于把纯 heuristic 基线与 SLM 配置分开。

### 端到端数据流

```text
真实 action/observation
        -> 有界确定性特征
        -> phase + stuck + hysteresis
        -> KEEP / PATCH / conditional REWRITE
        -> 结构化 GoalRecord（root 不可变）
        -> lexical/metadata Selector + PageCatalog
        -> SelectionPlan
        +-> ShadowBackend -> JSONL + Recall@Budget
        `-> DRAM runtime -> restore/bind -> Qwen -> seal new Pages
```

当前阈值只在一条 astropy 轨迹上扫描，不能泛化到其他任务；要称为“标定完成”，还需要更多
recorded tasks 和人工 gate-boundary label。完整的首轮中文实验报告见
[`goal_constructor_astropy14995_2026-09-02.zh-CN.md`](../../../results/archive/framework-v1/goal_constructor_astropy14995_2026-09-02.zh-CN.md)。

## 已实现

- 带版本的 `TurnContext`、`GoalRecord`、`PageCatalog` 和 `SelectionPlan`。
- 稳定的逻辑 Page 构造和原始位置校验。
- 跨 turn 的新增、保留、移除及重新激活 Page 统计。
- raw-objective 和透明 heuristic Goal baseline。
- session 作用域的 heuristic Goal 维护：包含不可变 root anchor、确定性特征、phase
  滞回、有界 stuck 统计，以及可审计的 KEEP/PATCH/REWRITE gate。
- 通过确定性 fake rewriter 或无依赖 command adapter 对 retrieval Goal 进行有条件、
  经 schema 校验的重写；包含 `root_goal` 或形似执行计划的输出会被拒绝，失败时采用
  确定性 fallback。
- Goal-aware 的 lexical/Page-metadata selection；当 Goal entity 改变时，排序会随之改变。
- Recent-only 和离线 Oracle Selector。
- 类型化 Shadow JSONL trace，以及 decision/update/generation/cost/root-invariance 指标和
  macro/micro Evidence Recall@Budget。
- JSONL artifact 和一个小型 CLI smoke runner。
- 防止选择当前/未来 Page 以及 Page 重排序的 invariant。
- 一个兼容 adapter：在调用此前已验证的 Qwen Dense/Reset/Cumulative 数值 replay engine
  之前，先校验新框架 record。
- 一个与框架无关的 DRAM KV runtime：支持事务化 sealed-Page offload、完整/选择性 restore、
  驻留复用、重新激活，以及按 session 统计传输量和驻留量。
- 一个采用 duck typing 的 Qwen hybrid-cache adapter：只对 full-attention K/V 分页，
  保留精确的逻辑位置，并且不触碰 recurrent/conv state。
- 一个经过 capability 检查的 Qwen pre-prefill bridge，分别处理物理 mask/cache position
  和逻辑 RoPE position。
- 一个零依赖的 recorded turn loop：具有稳定的 rendered-token boundary、Page/message
  provenance、Oracle budget plan 和 generation 后 sealing。

尚未实现：production serving loop、避免 contiguous-cache assembly 的 block-native
attention，以及在线语义检索。实验用 cumulative runner 已支持彼此独立且连续的 runtime
branch，但这不等于 production serving 集成；其余能力仍是显式的集成阶段，而不是核心
runtime 的隐藏依赖。

## 运行测试

无需任何第三方依赖：

```bash
PYTHONPATH=src python3.12 -m unittest discover -s tests -v
```

运行 Shadow smoke 示例：

```bash
PYTHONPATH=src python3.12 -m dram_kv.cli \
  --input examples/shadow_input.json \
  --output /tmp/dram-kv-shadow.jsonl \
  --goal heuristic \
  --page-size 128 \
  --bytes-per-token 0
```

## Goal Constructor Shadow 实验

第一版离线闭环刻意与 DRAM/Qwen 执行分离：

```text
latest action + latest observation
       -> deterministic bounded features
       -> phase/stuck detector -> advisory working-set policy
       -> KEEP / deterministic PATCH / conditional REWRITE
       -> lexical + Page-metadata Selector
       -> SelectionPlan -> ShadowBackend -> Recall@Budget
```

每个 session 的 `root_goal` 在第一个 boundary 冻结。之后如果 task objective、instance
identity 或 turn ordering 发生变化，系统会 fail closed。SLM 只能看到不可变 root、当前
retrieval Goal、最新且有界的 action 和 observation、抽取出的特征、trigger 及紧凑计数器。
它不能输出 `root_goal`、执行计划、命令、步骤或任意其他字段。

checked-in 的第一版 Goal Shadow 仍包含 **13-turn synthetic replay**（`turn_id` 6–18），
用于快速验证控制逻辑；现在另外已经接入 48-turn 真实 recorded SWE-agent trajectory，
并完成一次真实 Qwen2.5-0.5B/H200 conditional rewrite 验证。两者的实验层级和口径见下文。

在 checked-in synthetic replay 上运行全部四种配置：

```bash
PYTHONPATH=src python3.12 -m dram_kv.cli \
  --input examples/goal_shadow_replay.json \
  --output /tmp/dram-kv-goal-shadow \
  --shadow-config all
```

该命令会写出：

```text
/tmp/dram-kv-goal-shadow/
  raw_objective.jsonl
  stateless_heuristic.jsonl
  heuristic_only.jsonl
  heuristic_slm.jsonl
  comparison.json
```

四种配置共享完全相同的 Page catalog、label、Page budget 和 selector：

| configuration | Goal 行为 | model call |
|---|---|---|
| `raw_objective` | 每个 boundary 都使用原始 task objective | 无 |
| `stateless_heuristic` | 现有的逐 turn regex 重构 | 无 |
| `heuristic_only` | 冻结 root + 有状态 heuristic 维护 | 无；请求的 rewrite 使用经审计的确定性 PATCH/KEEP fallback |
| `heuristic_slm` | 相同 gate + conditional rewrite | 默认使用 deterministic fake；仅用于可选的一次性初始化和 hard-trigger rewrite |

每个 JSONL 文件在每个 boundary 包含一个自包含的 `shadow_turn` envelope，并以最终的
`run_summary` 结尾。turn envelope 包含有界 context hash 和 excerpt、完整的
Goal/feature/phase/gate/rewriter 状态、全部 selector hash、完整的 `SelectionPlan`，以及
Shadow evidence 指标。summary 固定使用以下分母：

- `maintenance_boundaries`：每个 session 除第一个 turn 外的所有 turn；
- `effective_retrieval_goal_update_rate`：retrieval-goal hash 变化次数除以 maintenance
  boundary 数；仅 phase-policy 发生变化时单独报告；
- `zero_retrieval_goal_change_rate` 和 `keep_invariance`：互补的 fast-path rate，以及
  KEEP 从未改变 retrieval state 的显式检查；
- `slm.maintenance.generation_rate`：实际 rewrite attempt 数除以 maintenance boundary
  数；初始化 call 单独统计；
- Recall@Budget macro/micro：仅统计 evidence label 非空的 boundary；
- root invariance：精确的 per-session root hash 加 constructor diagnostic。

`comparison.json` 中值得关注的字段有：

```text
summaries.<config>.decision_counts
summaries.<config>.trigger_counts
summaries.<config>.effective_retrieval_goal_update_rate
summaries.<config>.slm.{initialization,maintenance}
summaries.<config>.latency_ms
summaries.<config>.token_cost_totals
summaries.<config>.root_invariance
summaries.<config>.shadow_evidence_recall_at_budget
```

fake rewriter 是确定性的，因此测试/replay 无需网络。真实的本地 `<1B` SLM 推荐走常驻
NDJSON 模式；进程只加载一次模型，启动握手和逐请求 timeout 分离。Qwen adapter 使用
`lm-format-enforcer` 在每个 token 上执行 JSON Schema 约束，生成后仍须经过核心侧的严格
validator；任一环节失败都沿用 deterministic fallback。`torch`、`transformers` 和
`lm-format-enforcer` 仍只是脚本侧可选依赖：

```bash
PYTHONPATH=src python3.12 -m dram_kv.cli \
  --input examples/goal_shadow_replay.json \
  --output /tmp/qwen-goal-shadow.jsonl \
  --shadow-config heuristic_slm \
  --slm-command 'python3 -u scripts/qwen_goal_rewriter.py --model /path/to/Qwen3-0.6B --local-files-only --serve' \
  --slm-persistent --slm-startup-timeout-seconds 600
```

Qwen3 的 chat template 会显式设置 `enable_thinking=False`；同一 adapter 也可将 model 换成
Qwen2.5-0.5B。非 `--slm-persistent` 模式保持原有一次性协议兼容。一次性 command 可以返回
retrieval-Goal JSON object，也可以返回
`{"candidate": {...}, "input_tokens": N, "output_tokens": N,
"model_cost": X}`。它不得返回 `root_goal` 或执行计划字段。使用 `--no-slm-init` 可以把
maintenance rewrite cost 与可选的一次性 initializer call 隔离开来。

要在完全相同的离线 RewriteRequest 集上比较两个 checkpoint，可分别运行：

```bash
PYTHONPATH=src python scripts/benchmark_goal_rewriter.py \
  --requests examples/qwen_rewrite_request.json \
  --command 'python3 -u scripts/qwen_goal_rewriter.py --model /path/to/Qwen2.5-0.5B-Instruct --local-files-only --serve' \
  --output /tmp/qwen25.jsonl
PYTHONPATH=src python scripts/benchmark_goal_rewriter.py \
  --requests examples/qwen_rewrite_request.json \
  --command 'python3 -u scripts/qwen_goal_rewriter.py --model /path/to/Qwen3-0.6B --local-files-only --serve' \
  --output /tmp/qwen3.jsonl
```

### 真实 recorded SWE-agent replay（astropy-14995）

CLI 可以直接读取提供的 trajectory/Oracle zip。在线路径只读取 48 个真实
action/observation boundary；Oracle 只生成离线 older-evidence label，不会进入 Goal
constructor 或 selector。首轮 CPU 实验按 annotation 的语义单元设置“一条 completed
turn = 一页”，因此它是真实消息控制面实验，但不是 token-level KV 性能实验。

```bash
PYTHONPATH=src python3.12 -m dram_kv.cli \
  --input /path/to/astropy__astropy-14995_trajectory_oracle_selection_2026-09-02.zip \
  --output /tmp/astropy-14995-goal-shadow \
  --shadow-config all \
  --calibrate-thresholds
```

`calibration.json` 扫描 12 组可审计配置。仅在这条轨迹上，推荐项是
`entity_drift_steps=3`、two-step hysteresis 和默认 stuck threshold：micro
Recall@Budget 为 `0.459`，Goal update rate 为 `0.787`，rewrite rate 为 `0.170`。
这只是单实例标定；Oracle 没有提供 KEEP/PATCH/REWRITE 的 boundary 真值。

`--embedding-drift hashing` 可在无模型条件下运行 optional embedding + EWMA/CUSUM
状态机。首轮未调参运行新增了 7 次 semantic-drift trigger 且没有提高 Recall，所以默认
关闭。真实 embedding 可实现 `TextEmbedder`，或使用 JSON `CommandTextEmbedder`。

`scripts/qwen_goal_rewriter.py` 是 optional Transformers adapter。H200 上本地
Qwen2.5-0.5B 已成功加载；最终 48-turn replay 的 47 个 maintenance boundary 中，gate
调用 11 次（23.4%），严格校验接受 6 次（54.5%），其余 5 次安全 fallback（4 次未生成
JSON，1 次返回重复 entity）；SLM latency 为 p50 19.8 秒、p95 35.3 秒，因为 command
adapter 每次都会冷启动。root invariance 保持不变。这是旧 one-shot 路径的兼容性基线；
新常驻/约束解码路径已完成代码与 CPU 协议测试，但尚未获得 H200 效果数字。

### 10-task hash-bound Oracle cohort（2026-09-03）

第二阶段已从 20 条 recorded trajectory 中冻结 10 条长、resolved 实例，并在标注前固定
development/validation/test=6/2/2。最终 v2 Oracle 覆盖 649 个可回放 boundary、140 个
retrieval-positive boundary 和 166 条 older-evidence association，同时提供 gate 与 phase gold。
Gate 双重交叉复核后的 KEEP/PATCH/REWRITE 分布为 492/146/1；正常 phase transition 不再误作
REWRITE，`stuck_recovery` 固定为 PATCH。

产物与复现说明见
`results/recorded_oracle_10task_and_slm_2026-09-03.zh-CN.md`。每条 Oracle 绑定 exact trajectory
SHA-256 和 canonical message-map SHA-256；同 instance 的另一条轨迹会 fail closed。该数据已
通过 semantic Shadow payload 接线校验，并已用于下述冻结默认参数的跨任务 baseline。

随后按冻结默认参数完成了全部 10 条 CPU Shadow replay。结果显示链路可复现，但默认配置效果
较差：Gate macro-F1=0.239、Phase macro-F1=0.369、Recall@Budget=0.205、Goal update
rate=77.5%。因此它是 baseline，不是已调优配置。相同 replay 还冻结了 20 条真实 bounded
RewriteRequest；H200 上 Qwen2.5-0.5B 与 Qwen3-0.6B 的严格有效率分别为 30% 和 70%，成功
请求 p50 latency 分别为 23.05 秒和 6.06 秒。Qwen3 的实体覆盖与保留也更高，后续优先使用
Qwen3-0.6B；但尚未验证接回 closed-loop 后的 Recall 收益。完整口径见
`results/recorded_shadow_default_and_qwen_small_2026-09-03.zh-CN.md`。

recorded tokenizer span 现在会把 `files`、`symbols`、`tests`、`errors` 传播到 Page
metadata。cumulative Qwen/DRAM runner 保留 Oracle 上界默认值，并新增不读取 Oracle
selection 的在线 Goal 分支：

```bash
python scripts/run_cumulative_oracle_sweep.py \
  --model /path/to/Qwen3.6-27B \
  --bundle /path/to/trajectory_oracle.zip \
  --selection-policy goal --ratios 0.40 0.60 \
  --output /tmp/cumulative-goal.json
```

在 H200 cumulative Goal branch 中启用相同常驻 rewriter 时追加：

```text
--slm-command 'python3 -u scripts/qwen_goal_rewriter.py --model /path/to/Qwen3-0.6B --local-files-only --serve' \
--slm-persistent --slm-startup-timeout-seconds 600
```

首轮 H200 token-level 实跑完成 full-active 和一个 Goal 40% branch 的全部 48 turns，
共 96 个顺序正确的 artifact。Goal branch 平均 active token ratio 为 39.98%，最终 HBM
resident 从 2.487 GB 降到 0.995 GB；代价是 125 次 reactivation、相对 full-active 的
平均 top-1 agreement 0.9370、平均 delta-NLL +0.1574。因此它是“链路闭环 + 诊断基线”，
还不能称为 selector 质量收益。

### Schema 迁移说明

`SCHEMA_VERSION` 现在是 `1.2.0`。`GoalRecord` 保留其历史 positional field，并将它们
镜像到 canonical nested `RetrievalGoal` 中，因此现有 runtime/replay 代码仍保持源码兼容。
新 record 还会携带 root anchor、请求/实际应用的 gate decision、trigger、changed field、
working-set policy、SLM/fallback telemetry 和有界 diagnostic。selector signal hash 是对
selector 可能使用的每个字段（root、完整 retrieval Goal 和 policy hint）计算出的 canonical
JSON。Gate decision、latency、counter 和 cost 不会扰动该 hash。没有 nested field 的
version-1.0 Goal artifact 仍然可读。

### 尚待完成的实验

1. 在主模型首次 prefill 时提取并缓存 Page hidden-state sketch，验证是否可替代单独 metadata
   embedding forward。
2. 在 H200 和权威 GH200 上运行冻结候选的 cumulative Goal branch，对比 Oracle/full-active 的
   数值质量、真实 H2D bytes、HBM、delta-NLL、top-1 agreement 和 latency。
3. 用新的独立 trajectory 检查 hybrid Recall 收益，并在不复用当前 test 的前提下研究其较高
   reactivation/churn。

checked-in 的 synthetic 与 semantic-Page Shadow 闭环都是 CPU-only；只有 optional
真实 SLM 与 token-level cumulative 验证需要 GPU。

## 后续阶段

1. 在相同的 message/Page evidence label 上以 Shadow mode 比较 Goal 和 Selector 实现。
2. 在多条 trajectory 上累计运行最强的 online selector。
3. 集成 block-native attention，消除 contiguous-cache assembly copy。
4. 增加 retention slack/hysteresis，降低 H2D reactivation churn。
5. 将 allocator validation 从单 session 扩展到 concurrent serving。

## Legacy drift regression

`scripts/run_legacy_regression.py` 通过新 contract 迁移 recorded trajectory 和 Oracle
selection。在模型执行之前应用 budget 和受控 reactivation；随后用所得 `SelectionPlan`
中的 Page ID 重构数值 engine 消费的 plan。

在 DeltaAI 上，用以下命令提交完整的 13579 compatibility regression：

```bash
sbatch scripts/run_13579_regression_gh200.sbatch
```

launcher 申请一张 GH200、运行 30 分钟，并执行：

- nine-turn 100% active invariant；
- 完整的 40% active-budget replay；
- 完整的 Oracle replay，其中省略 turn-30 被重新激活的 Page。

`scripts/run_13579_restored_gh200.sbatch` 为 turn-30 causal comparison 提供成对的
restored-Oracle baseline。

该 backend 证明了与此前 semantic experiment 的兼容性。它**不能**证明 CPU DRAM
transfer、物理 HBM block release、resident reuse 或 serving throughput；这些需要原生
DRAM/HBM backend。

## 原生 DRAM runtime 基础

`DramKVRuntime` 持有不可变的 sealed CPU copy 和每个 session 的 active GPU working set。
`DramRuntimeBackend.execute()` 会在返回前同步 restore `SelectionPlan`，因此模型集成遵循
以下顺序：

```python
result = pipeline.run_turn(context, catalog)  # Goal -> select -> restore
active_kv = dram_backend.prepared_payloads(
    context.session_id,
    context.turn_id,
    preparation_id=result.backend_metrics["preparation_id"],
)
model.prefill(live_suffix, past_key_values=active_kv)
# after generation and Page sealing:
dram_backend.seal_after_generation(session_id, page_id, new_gpu_kv)
```

runtime 当前提供：

- 通过 tensor adapter 实现的可选 pinned CPU backing；
- 按原始逻辑 Page 顺序进行 selected 和 full restore；
- active-set delta reuse、inactive release 以及之后的 reactivation；
- session isolation，以及 transfer failure 时的事务化 state update；
- DRAM byte、active HBM byte、H2D/D2H byte、count 和 latency 指标。

`release_count` 表示 runtime 已丢弃其 GPU reference。调用方也必须释放自己的 reference；
model-specific GPU test 必须验证 allocator 的 `memory_allocated`/`memory_reserved` 行为。
generic runtime 自身不管理 recurrent state，也不能证明物理 HBM 节省。

## Qwen hybrid-cache 集成

`QwenKVPageAdapter` 将一个 Page 存储为每个已初始化 full-attention layer 的一对 K/V。
`extract_page()` 必须在 source span 仍有连续 physical-to-logical mapping 时运行。
selected restore 后，`assemble_pages()` 保留所选 Page 的原始逻辑位置，
`write_active_cache()` 则返回这些位置。调用方必须在后续 prefill 期间将它们传给模型的
RoPE/cache-position 路径。

linear-attention recurrent state 和 convolution state 被刻意排除在 Page payload 之外，
adapter 从不覆盖它们。当前 Transformers bridge 会把 restored Page 拼接为 contiguous
tensor；这可能暂时与 Page tensor 共存并形成 assembly peak。要消除这次 copy，需要
block-native attention backend。

可使用以下命令提交 GH200 allocator validation：

```bash
sbatch scripts/run_qwen_dram_runtime_gh200.sbatch
```

它会实例化模型配置中真实的 hybrid `DynamicCache` 和精确 K/V shape，但不加载模型权重，
随后用 CUDA allocator counter 测量 dense residency、DRAM-only backing、full restore、
selected restore、reactivation、contiguous assembly peak 和 final release。它是对上述
weight-loaded logit regression 的补充，而非替代。

已完成的 GH200 run 和 allocator table 记录在
`results/qwen_dram_runtime_gh200_2026-08-31.md` 中。

## Pre-prefill model bridge

`QwenForwardBridge` 把 restore 和 cache installation 设为正式 model prefill 之前经过
capability 检查的边界。它对 causal mask 和 cache update 使用紧凑物理位置，同时只把
suffix 的原始逻辑位置发送给 Qwen full-attention RoPE hook。superseded preparation、错误
turn、重排序 Page、不正确的逻辑 span 或 suffix/history boundary 都会在模型执行前失败。

使用以下命令提交 weight-loaded Dense/full/selected validation：

```bash
sbatch scripts/run_qwen_forward_bridge_gh200.sbatch
```

已完成的 Qwen 27B GH200 comparison 记录在
`results/qwen_forward_bridge_gh200_2026-09-01.md` 中。100% restored branch 在 logits、
full-attention K/V 和全部 recurrent/conv state tensor 上与 Dense 精确一致。这建立了第一条
原生端到端 pre-prefill bridge；它尚未提供 serving loop，也未消除 contiguous assembly
copy。

## Recorded turn loop 与 Oracle budget pilot

`RecordedTurnLoop` 在固定 rendered token 上强制执行
`plan -> restore/bind -> model -> seal`。Oracle evidence 通过显式 source message ID 进行
映射，因此 annotation 的 completed semantic turn 同时包含其 action 和之后的 observation，
同时不会把当前 live observation 泄漏进 sealed history。mandatory overflow 是 hard error；
每个 plan 都会保存被省略的 recent/evidence Page、capacity filler 和 unused capacity。

使用以下命令运行 Qwen 27B pilot：

```bash
# 历史一次性 Slurm wrapper 已删除；底层 Python 工具仍保留。
```

已完成的 48-turn 原生 DRAM replay 和 six-turn 25%/40%/60% Oracle sweep 记录在
`results/oracle_budget_sweep_gh200_2026-09-01.md` 中。

## 累计独立 runtime 实验

`scripts/run_cumulative_oracle_sweep.py` 让 full-active、40% 和 60% branch 分别沿各自独立的
DRAM/HBM lifecycle 和连续 recurrent state 推进。新 Page 从每个 branch 的 sparse physical
cache 中逻辑连续的 suffix 封存，并在切换后标记为 selection-conditioned。

已完成的 GH200 实验记录在
`results/cumulative_oracle_sweep_gh200_2026-09-02.md` 中。实验确认了预期的 HBM 降低，
同时也表明：与 full-active 相比，朴素 selected-set churn 使 H2D traffic 在 40% 配置下
增加 26.4%，在 60% 配置下增加 16.1%。
