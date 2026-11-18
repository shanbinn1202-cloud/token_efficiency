# 选页规则与证据

[English](design.md) | 简体中文

第 5 节和第 8 节，integrate-1107 之前在 README 中，编号不变。各文档页面的节号互不重复；[文档地图](../README.zh-CN.md#文档地图)列出了每一节所在的页面。

## 5. query 策略的选页规则

本节是选页策略（`--policy`）的目录：每个策略的规则、它服务的协议、它的选项，以及哪些不能组合；下面各段给出每条规则的完整说明。哪些策略在 GPU 上跑过、跑得怎样，见 [docs/status.zh-CN.md](status.zh-CN.md)；它们的限制见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md)。Gemma 4 profile 服务哪些，见 4.14 节。

| `--policy` | 类型 | 规则 | 协议 | 选项 | 详见 |
| --- | --- | --- | --- | --- | --- |
| `full` | full | 每个可用页（参照组） | 全部 | | |
| `random50` | random | ⌊n/2⌋ 个可用页，由种子 42 和前缀最后一个完整 KV block 的哈希抽取 | 全部 | | 下面的差异 |
| `goal_query` | query | Selector V2：必保留页、recent 页，然后是与维护的 Goal 的 query 向量最接近的页 | 全部 | `--goal-rewriter-*`、`--goal-overlap`、`--token-gather` | 本节 |
| `observation_query` | query | Selector V2，以最新 observation 末尾的 hidden state 为 query | 全部 | `--token-gather` | 本节 |
| `random_fill` | random | Selector V2 的轮次流程和容量，其余按带种子的随机顺序 | 全部 | `--fill-seed` | 下面的随机填充对照 |
| `recent_fill` | window | 容量相同，其余从新到旧 | 全部 | | 下面的最近窗口策略 |
| `recent_tools` | window | 容量相同：工具调用页，然后工具返回，最后 reasoning，各自从新到旧 | 全部；与 `--thinking-history strip` 一起时被拒绝 | | 下面的最近窗口策略 |
| `akv_rule` | rule | 所有回复和最近 12 个工具返回；没有预算 | 全部；与 `--thinking-history strip` 一起时被拒绝 | | 下面的 AKV 规则 |
| `ledger_query` | query 插件（observation 向量） | 填充先给最近一次编辑和每个文件最近一次读取留出空间；去重；不填旧的 reasoning | `mswea-xml`，Qwen | `--ledger-cap-percent` | [policy-registry 第 8 节](policy-registry.zh-CN.md#8-query-钩子与-ledger_queryledger-policy-1112) |
| `goal_query_markers` | query 插件（Goal 向量） | 填充先取含协议标记的页，然后是 `goal_query` 的填充 | `mswea-xml`，Qwen | `--marker-set`、`--marker-cap-percent`、Goal 的选项 | [policy-registry 第 9 节](policy-registry.zh-CN.md#9-标记页goal_query_markers-与-observation_query_markersmarker-pages-1114) |
| `observation_query_markers` | query 插件（observation 向量） | 同样的标记页，然后是 `observation_query` 的填充 | `mswea-xml`，Qwen | `--marker-set`、`--marker-cap-percent` | policy-registry 第 9 节 |

`--token-gather markers` 或 `separators` 是 `goal_query` 和 `observation_query` 的选项，不是策略：在同一预算内于选中页之外保留单个历史 token，恢复时把它们的 K/V 聚集到整 16-token 块中（`mswea-xml`、Qwen、16 token 的页；与插件和 `--release-idle-pages` 一起时被拒绝；[policy-registry 第 6 节](policy-registry.zh-CN.md#6-token-级选择方案-a标记页方案-btoken-聚集)、[docs/token-gather/DESIGN.zh-CN.md](token-gather/DESIGN.zh-CN.md)）。

`goal_query` 和 `observation_query` 执行 dram-backed-kv PoC（`scripts/serve_online_selector_v2.py`，上游 commit `a70701e`）的 Online Selector V2 合同。服务在启动时把它固定为自己的**选页合同**：`startup.json` 和 `ready.json` 中的一个 JSON 对象（`selection_contract`；`runtime.jsonl` 的行在 `query.selection.contract_sha256` 中给出它的摘要），它确定 P、切换阈值、预算、recent 配额、投影以及每个策略自己的固定项。页大小 P 是这个合同的参数。默认 P = 784 时，vLLM 以 784 token 的 Page 恢复全注意力 KV，所以 PoC 按 16 token 计的预算要换算成整页，每个 Page 用它最后一个 16-token 子页来代表。P = 16 时（`--page-tokens 16`，4.6 节），Page、预算和向量都是 PoC 的 16 token 版本。

下文用到的术语：
- **session：** 一个 agent 在一个任务上的多轮对话，有它自己的 KV 历史（4.4 节）。使用 XML 协议时，服务启动时带着它的冻结任务，每个任务一个 session（4.1 节）。
- **可用（候选）页：** 最新 observation 之前历史中已封存的 Page。
- **Gated-DeltaNet（GDN）：** Qwen3.6-27B 的线性注意力层，占其 64 层中的 48 层。它们的递归状态（153,944,064 字节）承载整个历史；Page 及其选择属于 16 个全注意力层。vLLM 把这些状态放在它的 Mamba 池里（第 8 节）。
- **live tail：** session 的最后一页，还没写满，所以尚未封存；每一轮都关注它。
- **覆盖：** 一个 Page 只要含有某类 token 中的任一个，就算覆盖了这一类。
- **rebuild：** 不延续其 session 的请求会让 session 从整个请求重新开始：从文本重新渲染，释放之前的 Page，重置 selector 和 Goal（6.1 节；`runtime.jsonl` 中的 `stage_b.rebuild`）。唯一的例外是丢弃回复之后 `goal_query`（以及插件 `goal_query_markers`）的 Goal（上一个请求加一条 user 消息，即格式错误重试；goal-rollback-1116）：它从被丢弃回复所在的轮次继续（running.zh-CN.md 6.1 节）。`local_turn` 统计 session 自开始或上次 rebuild 以来的请求数；`turn` 统计全部请求。

会话第一个请求之后的每个请求，每个策略都按 PoC 的两阶段轮次（PoC 称为 V0 时序）执行：

1. **阶段 1（observation）。** 新的 observation 在上一轮之后仍常驻的 Page（上一轮的激活页，加上服务上一轮时封存的页）和 live tail 页上 prefill，并接续上一轮的 Gated-DeltaNet 状态。这一步得到它所封存的 Page 的向量，以及 observation query：observation 最后一个 token 的 hidden state。
2. **Goal。** 对 `goal_query`，由上一轮回复和最新 observation 构造维护中的 Goal。其 query 文本单独做一次冷 forward，并与上游一样按 retrieval-goal hash 缓存。
3. **选页**：在 observation 之前的 Page 里挑选（见下）。
4. **阶段 2（生成）。** 生成头和 decode 在选中的 Page 加上阶段 1 留下的 observation KV 上运行。

每个 token 只计算一次；本轮封存的 Page 使用这唯一一次计算得到的向量。会话的第一个请求是一次普通 prefill。

选页规则：

1. 第 0 轮，以及已封存历史不超过 **8192 token** 时，所有 Page 都保持激活（`reason` 为 `cold_start` / `below_switch_threshold`）。
2. 否则容量为 `min(n, max(1, 必保留页数, floor(0.40 × 历史 token) // P))` 页，其中 n 是候选页数（observation 之前已封存的 Page），历史为 n × P 个 token。`--active-token-budget N` 用绝对预算代替比例；`--active-ratio R` 修改比例。
3. 始终保留**必保留**页：初始 system + 任务 prompt 触及的每一页。
4. 加入 **recent** 页：前两轮里的非必保留页，从新到旧，最多 ⌈1,024 / P⌉ 页（对应 PoC 的 64 × 16 = 1,024 token）：P = 784 时 2 页，P = 16 时 64 页。
5. 剩余容量用与 query 余弦最高的页填满，使用上游的 `PrefillPageEmbeddingSelectorV2`。

向量：取模型最终 norm 之前的 last-token 隐藏状态（即 PoC 的最后一层 decoder 输出），用 PoC 的带种子 128 维 Rademacher 矩阵投影（seed 20260904，hidden size 5120 时 SHA-256 为 `20c2e1cf…87aa`），再 L2 归一化并逐向量 int8 量化。Goal query 是维护中 Goal 的 PoC `<retrieval_query>` JSON 文本；`--goal-query-variant candidate_entity_cap8` 在这段文本里只保留最后 8 个 files、symbols 和 tests。维护中的 Goal 使用 entity-drift 3；指定 `--goal-rewriter-model` 时使用 PoC 的常驻 Qwen3-0.6B rewriter（默认引擎）。`--goal-rewriter-engine vllm` 改由一个 vLLM 子进程提供同一个模型（4.10 节）；在 README 的稳定配置命令之外，它属于实验性功能（[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 3.2）。某一轮 rewrite 失败时与上游一样回退到确定性字段，并记录下来；`goals.jsonl` 保存每个 Goal、Goal query 和 rewriter 交互的文本（第 8 节）。

`observation_query` 不使用 Goal rewriter，因为它的排序只取决于 observation 向量：服务从不为它启动 rewriter，它的 Goal 只用确定性字段构建（仅用于证据），钉住 rewriter 的启动配置会被拒绝，给出的 `--goal-rewriter-model` 会在 `startup.json` 和 dry run 中记为未使用（`goal_rewriter`）。

**随机填充对照。** `--policy random_fill` 就是只改了第 5 步的 `observation_query`：阈值、容量、必保留页和 recent 页都相同，阶段 1 和确定性 Goal（仅用于证据）也相同；不使用 Goal rewriter（给出的 `--goal-rewriter-model` 记为未使用，钉住 rewriter 的配置会被拒绝），也不做 Goal forward。去掉必保留页和 recent 页之后剩下的候选页被随机排序，取前面的页填满容量；不用 query 向量、投影或分数。排序用 SplitMix64 加 Fisher-Yates（不做任何哈希），由启动时钉住的种子（`--fill-seed`，默认 42，即选页合同的 `random_fill` 键）、session 序号和该 session 的请求序号决定。每个选页生效的轮次，`query.selection.fill` 记录种子、session 序号、请求序号（`turn`）以及可选页数和抽中页数，`retrieved_pages` 按抽取顺序列出抽中的页。如果 `goal_query` 和 `observation_query` 不比这个对照好，它们的排序就没有带来有用的信息。与 `random50` 不同，它保留 query 策略的必保留页、recent 页和预算。

**AKV 规则（akv-rule）。** `--policy akv_rule` 复现 AKV（Fu 等，"Cache Engineering for Agents"，arXiv 2610.28577）的工具返回滚动丢弃策略：论文保留全部 reasoning 和最近 12 个工具返回的缓存（3.2 节和 5.1 节），工具返回超过这 12 个之后丢弃更早的工具返回缓存（附录 A.4.2），没有 token 阈值。akv_rule 关注必保留页（初始的 system + task 提示，论文从不丢弃它们）、含有任一 agent 回复 token 的所有 Page（即 60c5d21 的 `recent_tools` 的回复片段：生成头、reasoning、`</think>`、content 和收尾边界；覆盖规则），以及最近 12 个工具返回的 Page，其中当前轮的 observation 算作最新的一个（它是 live tail，总是被关注），所以最近 11 个已封存的 observation 片段被整段保留；除此之外什么都不选。只有当一个 Page 的每个 token 都属于更早的工具返回时，它才被丢弃；rebuild 后首请求带来的历史无法按角色切分，保留（论文从不丢弃 reasoning）。没有预算：容量就是规则选中的内容，所以合同的 `active_ratio` 和 `active_token_budget` 都是 null（serve.py 对 akv_rule 拒绝 `--active-token-budget` 和 `--active-ratio`），也没有 token 阈值（`switch_threshold_tokens` 为 0：规则从 local turn 1 开始生效，在第 13 个工具返回到来之前保留所有 Page，那时最早的一个已封存工具返回被丢弃）。填充顺序是保留的 Page 从新到旧，然后是被丢弃的 Page；对 akv_rule，原生 admission gate 把每个 Page 都算作可能的恢复（`capacity.selected_max`）。只有它的合同带 `akv_rule` 键（`{rule: replies_and_latest_tool_responses_v1, latest_tool_responses: 12}`）；其余所有合同的字节和摘要不变。每个选页生效的轮次，`query.selection.window` 除上面的窗口字段外还记录 `selected_page_count` 和 `selected_tokens`、`reply_pages`、`kept_tool_responses`（min(11, 已封存的工具返回数)；当前的那个是第 12 个）和 `dropped_tool_responses`、`dropped_pages`，以及保留的最早一个已封存工具返回的 Page 和 token 位置（`first_tool_page`、`first_tool_position`）。与 AKV 的已知差异：(1) 位置：保留的 Page 使用原始位置（AKV 按逻辑 token 顺序重编码位置的 `repos` 没有实现）；(2) 被丢弃的 Page 留在 DRAM 中，规则不会再选中它们（AKV 优先淘汰被丢弃的缓存）；(3) Qwen3.6 的 GDN 层保留整个历史（包括被丢弃的工具返回）的状态，所以丢弃只作用于它的 full-attention 层（论文的模型 MiniMax-M2.7、GPT-OSS-120B 和 MiroThinker-1.7-Mini 是 MoE transformer，论文没有描述这类状态）；(4) 规则由服务端自己执行，而 AKV 由 agent harness 在每个请求中发出缓存操作（消息下标）；(5) 粒度：按覆盖规则以 P token 的 Page 为单位（`--page-tokens 16` 时为 16，默认时为 784），所以被丢弃工具返回中与保留 token 共享 Page 的 token 仍被关注（AKV 丢弃整条消息）；(6) "最近 12 个"按包含最新工具返回来理解，论文没有明确说明。测试：`test_akv_rule.py`。

**最近窗口策略（recent-fill）。** `--policy recent_fill` 和 `--policy recent_tools` 就是把随机顺序换成固定填充顺序的 `random_fill`：阈值、容量、必保留页和 recent 页都相同，阶段 1（worker 仍然计算并发送 Page 向量）和确定性 Goal（仅用于证据）也相同；不使用 Goal rewriter，也不做 Goal forward；不用 query 向量、投影或分数，也不对每个 Page 做哈希。`recent_fill` 把候选页（既不是必保留页也不是 recent 页的已封存 Page）按位置从新到旧填入。recent 页本来就是最新的页，所以它选中的是必保留页，加上从窗口第一页到最新已封存 Page 的连续窗口：最近的若干工具返回和其间的回复，预算能放下多少个完整轮次就是多少个。`recent_tools` 按覆盖规则（含有某类任一 token 的 Page，与必保留页相同）给每个候选页分类，优先级依次为：含任一工具调用 token 的是 CALL 页，否则含任一工具返回 token（observation 片段）的是 TOOL 页，其余是 REASONING 页；先从新到旧填 CALL 页，再从新到旧填 TOOL 页，最后从新到旧填 REASONING 页。一次回复从某轮 observation 片段的结尾（第一轮：其 stable end）到下一轮的 stable end：生成头、生成的 token 和收尾边界。其工具调用是生成头之后第一个 `</think>` 之后的全部内容（content，直到收尾边界）；其 reasoning 是生成头和 `<think>` ... `</think>`。thinking 关闭时，或回复中没有 `</think>` 时，生成头之后的整个回复都是工具调用。这一划分在 transcript 自己的 token 流上计算：开启 `--enable-thinking` 时 XML 协议在历史中保留 reasoning（6.8 节）；`--thinking-history strip`（chat-text）在启动时拒绝 `recent_tools` 和 `akv_rule`，因为去掉 reasoning 后的历史把每条过去的回复放在下一个 observation 片段里，其中的工具调用和 reasoning 无法与工具返回区分（在 review-fixes-1110 之前这个组合会被接受，此时 `recent_tools` 的顺序与 `recent_fill` 完全相同，`akv_rule` 会丢弃旧回复的 Page）；kv-session transcript 没有 encoder，所以其回复全部算作工具调用。一个首请求如果带有初始消息之外的历史（rebuild），这部分历史不属于任何 observation 或回复片段，与工具返回一起参与窗口（TOOL）。每个选页合同带自己的键（`recent_fill` 或 `recent_tools`：`{order}`，无选项）；其余所有合同的字节和摘要不变。每个选页生效的轮次，`query.selection.window` 记录必保留页之后第一个被选中的 Page 及其 token 位置（`first_page`、`first_position`），已封存历史中有多少个完整轮次（一个工具返回及其后的回复）和多少个完整工具返回被关注（`covered_turns`/`history_turns`、`covered_tool_responses`/`history_tool_responses`，即 N；当前 observation 总是额外被关注），对 `recent_tools` 还记录保留的 CALL 页（`call_pages` 中的 `calls_kept`、`calls_truncated`）、保留的 TOOL 页（`tool_pages` 中的 `tools_kept`）、保留的 REASONING 页（`reasoning_pages` 中的 `reasoning_kept`），以及第一个被选中的 TOOL 页及其位置（`first_tool_page`、`first_tool_position`）。`retrieved_pages` 按填充顺序列出填入的页。测试：`test_recent_fill.py`。

**插件策略与 token 聚集。** `ledger_query`、`goal_query_markers` 和 `observation_query_markers` 是策略注册表的 query 插件（`source-overlay/page_policies/`）：每个选页生效的轮次中，Selector V2 完全按其基础策略计算划分和余弦顺序，插件的钩子在余弦填充取走其余预算之前保护、排除或成组候选页（[docs/policy-registry.zh-CN.md](policy-registry.zh-CN.md) 第 8 节和第 9 节：规则、固定值和记录）。`--token-gather` 在选中页之外保留历史中的单个 token；页的选择不变（[docs/token-gather/DESIGN.zh-CN.md](token-gather/DESIGN.zh-CN.md)）。token 聚集不能与插件组合。

与 PoC 的已知差异，以及两种布局之间的差异：
- 默认 784 token 的 Page 下，预算都是整页数，live tail 是一个未写满的 Page。16 token 的 Page 下，预算、recent 配额和 live tail（不足 16 token）都与 PoC 相同。
- 必保留页覆盖 system + 任务 prompt 碰到的每一页。PoC 只保留完全落在其中的页，在 784-token 的页上可能丢掉任务的结尾；在 16 token 的页上两者最多差一页。
- 解耦布局下，worker 在 GPU 上一批地计算 int8 Page 向量；默认布局下在 CPU 上逐个计算。两条路径的结果可能不同：在 600 个合成的 5,120 维行上（x86 CPU torch），约 0.35 % 的向量（600 个中约 2 个）有一个元素相差 1，可能改变近乎并列的排序。测试（`tests/test_decouple_wp3_vectors.py`）把这个比例限制在 2 % 以下，并要求每处差值为 1（`docs/exp-decouple-pages/DESIGN.md`，Integration 第 4 项）。
- `random50` 保持固定的 floor(n/2) 随机抽取。16 token 的 Page 下它的种子来自 16 token block 的哈希，所以它的抽取结果不能与 784 token 的 Page 下比较。

## 8. 每个服务的证据

每个 `--output` 目录包含：

| 文件 | 内容 |
| --- | --- |
| `startup.json`、`launch.json`、`chat_template.jinja` | 冻结的任务/配置哈希、确切的服务命令、固定模板 |
| `ready.json`、`worker-attached.json` | READY 身份（READY 见 4.1 节；含选页合同、norm 前向量空间和 `epoch`，即本次服务运行的身份标识，native 文件、`goals.jsonl` 和清理回执都会重复它）与 worker 接入回执 |
| `runtime.jsonl` | 每轮一行：响应 `id`、内部 request ID、token/顺序哈希、`candidate_page_count`、`selected_logical_pages`、`prepare`/`restore` 的 H2D 字节数、joined purge/drain、延迟。`stage_b` 记录第 5 节两阶段轮次（Stage B）的几何：已计算起点 C（本请求之前该 session 已计算的 token 数）、候选终点、observation 区间、阶段 1 区间、阶段 2 起点、`retain`/`restore`/`release`、已计算终点 C′ = prompt + 生成 − 1、封存终点和 `rebuild`。Selector V2 策略（除 `full` 和 `random50` 之外的每个策略）的行另有 `query.selection`（`reason`、`capacity_pages`、`mandatory_pages`、`recent_pages`、`retrieved_pages`、合同与投影哈希；`random_fill` 另有 `fill`，`recent_fill`、`recent_tools` 和 `akv_rule` 另有 `window`，插件另有 `hook`）和 `query.goal`（变体、rewriter 是否尝试/成功/回退、缓存命中、冷 forward）；`goal_query` 的行另有 `query.goal_text`（见下文） |
| `goals.jsonl` | 仅 `goal_query`：每个请求的 Goal 哈希背后的文本，以及 Goal rewriter 的输入和输出（见下文） |
| `native-<sha256(request id)>.json` | native plan（`selected`、`retain`/`restore`/`release`；`run_pages`，即阶段 1 交给阶段 2 的 Page；tail 标记 `tail_candidate`（tail 页在阶段 1 中写满）和 `tail_live`（tail 页延续到阶段 2）；`phase1`，即阶段 1 的 block lease 的 id 和摘要）、worker frames（`prepare`、`phase1`、`phase1_drain`、`reserve`、`allocation_update`、`complete`）和 drain 回执（`drain`：所有拷贝 join 之后的 `drained`、carry 的已计算终点和位于 DRAM 的 Page） |
| `failures.jsonl` | 约定/运行时失败，带异常 `chain` 和 `traceback`（成功时不存在：只在有请求失败时才写） |
| `engine-cleanup.json`、`parent-cleanup.json` | 关停回执 |

解耦布局会增加一些字段；默认布局不写其中任何一个（`docs/exp-decouple-pages/WP4.md` 第 4 节）：
- `ready.json`：`kv_layout`（P = `page_tokens`，注意力 Page；M = `mamba_tokens`，一个 Mamba block 的 token 数，Gated-DeltaNet 状态每过这么多 token 跨到新 block；K = `mamba_slots`，Mamba 池中的状态槽位数）、`num_blocks_by_pool` 和 `free_blocks_by_pool`（注意力池在前，Mamba 池在后）；`worker-attached.json`：同样的内容，记为 `kv_pools`。
- 每个 `runtime.jsonl` 行和 native 文件：该请求 drain 之后的 `kv_pools.free_blocks_by_pool`。
- `engine-cleanup.json`：所有 session 关闭后的 `kv_pools`。运行干净时两个池都回到 READY 时的基线（除每个池的 null block 外全部空闲），已验证的运行检查过这一点。
- 封存 Page 的记录用 `token_span` [pP, (p+1)P] 而不是 `token_sha256` 来指明其 token，Page 向量使用 `compact_int8_v1` 形式 `[[128 int8], scale]`。

session 池服务（4.4 节）还会写 `sessions.jsonl`，每次 session 打开和关闭各一行（`schema` 为 `remote-ra-session-event-v1`，含 session id、序号和任务 id、`open_sessions`、各池的空闲 block，关闭时还有其回执）；`ready.json` 增加 `session_pool`（`max_running_sessions`、`registered`）；每个 `runtime.jsonl` 行增加 `open_wait_s`；`engine-cleanup.json` 的 `sessions` 列出关停时仍打开的 session，并增加 `session_pool`（哪些由 harness 关闭、哪些未打开就关闭、哪些在关停时仍打开、哪些从未打开）。不带这个选项的服务不写其中任何一项。

vLLM 的 KV connector 有一个锁页的 CPU 池，本 runtime 从不向它存取数据（4.9 节）。开了 `--connector-pool-gib` 的服务只在 `startup.json` 中记录 `connector_pool_bytes`。默认速度选项把它设为 1 GiB（4.15 节），所以不加 `--no-speed-defaults` 或 `--no-connector-pool-gib` 启动的服务都会记录它（1,073,741,824）。

开了 admission 参数的服务（4.10 节）在 `startup.json`、`worker-attached.json` 和 `ready.json` 中记录固定的 `admission` 记录；开 `--gdn-device-handoff` 时，每个 `runtime.jsonl` 行增加 `restore.d2d_bytes`。开 `--goal-overlap` 时，每个 native 文件增加 `goal_overlap_ms`；运行了 Goal forward 的轮次的 native 文件还增加 `goal_forward_ms` 和 worker 帧 `goal_forward`（其中包含这次 forward 的 drain），其 `runtime.jsonl` 行不变。不开这些参数的服务不写其中任何一项。开 `--goal-rewriter-batch N` 的服务在 `startup.json` 和 `ready.json` 中记录 `goal_rewriter_batch` 及其 token 上限 `goal_rewriter_batch_tokens`，并在每个 `goals.jsonl` 交互中增加 `batch_id` 和 `batch_size`。

开了 `--language-model-only` 的服务（4.12 节）在 `startup.json` 及其 `effective_server_config` 中记录 `language_model_only`，在 `worker-attached.json` 中记录 worker 的纯文本记录，`ready.json` 两者都记录。不开这个参数的服务不写其中任何一项。

开了 `--deferred-completion-io` 的服务（4.11 节）在 `startup.json` 和 `ready.json` 中记录 `deferred_completion_io`；它的 native 文件和 `goals.jsonl` 与不开时引擎线程写出的逐字节相同，只是写得更晚（在前端读取之前）。

kv-session 服务（6.6 节）在每个 `runtime.jsonl` 行增加 `kv_session`。chat-text 服务（6.7 节）在每个 `runtime.jsonl` 行增加 `chat_text`（`client_session`、`kind` session/main/aside、`key_sha256`、`session_id`（X-Session-ID 或 null）、`slot`、`slot_turn`、`client_turn`、`binding` bind/rebind、`rebind_reason`、`rebuild`、请求的和实际的 `max_tokens`、`prompt_tokens`、`completion_tokens`、`finish_reason`、`busy_wait_s`），并写 `chat_sessions.jsonl`（`schema` `remote-ra-chat-text-session-v1`：每次绑定、换绑和关闭一行，含槽位、原因、空闲时间和关闭回执里的轮数）。chat-text 的 `goal_query` 服务（暂定的 Goal 输入，6.7 节）在 `startup.json` 中记录 `chat_text_goal`；它的 `chat_text` 行增加 `goal_inputs`（`definition`、`request`：`main`、`summary_call`、`answers_call`、`questions_call` 或 `short_summary_call`，`requirement` 和 `observation` 的规则名，`rewriter`：`allowed` 或 `skipped`），它的 `goals.jsonl` 行增加 `inputs`（同样的内容，加上每个输入在请求消息中的 `span`，即消息序号以及在其 content 中的起止位置，它的字符长度，`goal_source`：`maintained`、`session_last` 或 `requirement_only`，以及摘要交接时的 `goal_restarted`）。开 `--release-idle-pages` 的服务（4.13 节）写 `capacity.jsonl`，在每个 native 文件和 `runtime.jsonl` 行增加 `idle_release`，在 `engine-cleanup.json` 增加总计，并在 `startup.json`、`worker-attached.json` 和 `ready.json` 中记录 `release_idle_pages`。其他服务不写其中任何一项。

开启 thinking（`--enable-thinking`，6.8 节）时，每个 `runtime.jsonl` 行增加 `thinking`：reasoning 和 content 的 token 数、`closed`，以及 `history`，即请求与 session 的关系（`fresh`、`append`、自 keep-rebuild-1116 起 XML 协议上的 `rollback`、`rebuild_reasoning_dropped`、`rebuild_reasoning_differs`、`rebuild`）；`rollback` 或 XML 重建的行另有 `kept_tokens` 和 `reencoded_replies`。session 池中关闭一个 session 会返回关闭回执（`POST /v1/sessions/close` 的响应及其 `sessions.jsonl` 行），其 `released` 记录逐项列出释放的 block：`released.blocks` = `resident_pages` + `tail` + `gather_blocks`（不开 `--token-gather` 时 `gather_blocks` 为 0；gather-receipt-1115）。

**Goal 文本（仅 `goal_query`）。** `goals.jsonl`（`schema` 为 `remote-ra-goal-text-v1`）对每个到达 native 终态的请求（完成或取消）写一行，保存 `runtime.jsonl` 中 Goal 哈希背后的文本：
- `request_id`、`session_id`、`session_index`、`turn`、`local_turn`、`epoch`、`terminal_state`：与该请求的 native 文件相同（`epoch` 是 `ready.json` 中本次服务运行的标识）。
- `goal`：`text` 是 selector 实际使用的 Goal，即其 selector payload（schema 版本、root goal、检索 Goal、policy hint）的规范 JSON；`sha256` 等于该行的 `query.goal_hash`。
- `query`：`text` 是 Goal query 的 `<retrieval_query>` 文本（`variant`，第 5 节）；`sha256` 等于 `query.goal.payload_sha256`。
- `decision`：gate 决策和实际采用的决策、触发条件、变化的字段、rewriter 是否尝试和成功、`fallback_reason`。
- `carry`（只出现在丢弃了上一条回复的重建上，goal-rollback-1116）：Goal 是否从被丢弃回复所在的轮次继续；与该行 `query.goal.carry` 相同（running.zh-CN.md 6.1 节）。
- `rewrites`：构造 Goal 时每次调用 Qwen3-0.6B rewriter 一项：`request`（发送的确切 JSON payload）、`response`（rewriter 进程返回的原始一行；没有返回时为 `null`）和 `error`（失败调用的 `Type: message`，即异常类型和消息，等于 `fallback_reason`）。成功时这一行包含 rewriter 从其生成结果中解析出的候选。rewriter 无法解析自己的生成结果时，这一行只有错误信息和 token 计数：随仓库提供的 rewriter 脚本不返回生成的文本。调用之前就被拒绝的请求（超出大小上限）没有对应项。每一项还记录该次调用的时间（用服务进程的 `time.perf_counter` 测量）和子进程自己报告的计数：
  - `wait_ms`：从发起调用到 rewriter 接手。rewriter 在锁内一次只服务一个调用，所以这是排在其他 session 的调用后面所花的时间。
  - `start_ms`：调用内部启动 rewriter 进程的时间；只有进程已退出或之前某次调用超时（rewriter 随后会关闭进程）时才不为 0。
  - `call_ms`：从发送到收到回答的那一行；没有回答时到失败为止（超时一直算到 120 秒上限）。
  - `input_tokens`、`output_tokens`、`generation_latency_ms`：从回答行中复制（子进程的 prompt token 数、生成 token 数和它的 `generate` 用时）；回答行中没有时为 `null`，例如超时之后。
  - `batch_id`、`batch_size`（只在开 `--goal-rewriter-batch N` 时，4.10 节）：服务内的批次序号和一起发送的调用数。这种服务上，`wait_ms` 是排队到批次形成的时间，`start_ms` 是子进程启动和批次行，`call_ms` 是批次从发送到回答，`generation_latency_ms` 是批次的 `generate` 用时；token 计数是该调用自己的。`generate` 失败的调用（作业 3404230 中 4.10 节所述的显存不足和 cuDNN 错误）的 token 计数和 `generation_latency_ms` 为 `null`。

该请求的 native 文件和 `runtime.jsonl` 行分别用 `goal_text` 和 `query.goal_text`（`{"file": "goals.jsonl", "sha256": …}`，即该行不含换行符的字节的 SHA-256）指向这一行。验证方对 `runtime.jsonl` 中每个 `goal_query` 行检查：`goals.jsonl` 中有一行的 SHA-256 等于该值，且 `request_id` 相同；`goal.text`（UTF-8）的 SHA-256 等于 `query.goal_hash`；`query.text` 的 SHA-256 等于 `query.goal.payload_sha256`。在 GPU 上该文件每轮约 11.1–11.3 KB，主要是 `goal.text` 中每轮重复的 root goal：作业 3461703 中 2,109 行共 23,443,474 字节，作业 3428631 中 2,303 行共 26,002,489 字节（SWE-bench 的 `goal_query` 服务）。16 个 session × 70 轮时，每个服务约 12.4–12.6 MB。记录不改变任何 Goal、选页或 rewriter 调用，也不做逐请求检查；服务启动时检查 Goal 文本的哈希等于 GoalRecord 的 version hash。其他策略既不写该文件，也不写引用。

**Goal 用时（`goal_query`，多个 session 一起解码时）。** Goal 在 Goal 线程上构造时（4.10 节），native 文件记录 `goal_async_ms`（从挂起该 admission 到恢复它）和 `goal_construct_ms`（构造的实际用时）。开 `--goal-overlap` 时，`goal_async_ms` 是阶段 1 之后在 Goal 处挂起的时间（那时 Goal 已构造好则为 0），`goal_overlap_ms` 是从开始构造 Goal 到这一时刻的时间。如果构造调用了 rewriter，native 文件再增加 `goal_rewrite_wait_ms` 和 `goal_rewrite_call_ms`，即其各次调用 `wait_ms` 与 `call_ms` 之和；`goal_construct_ms` 包含这两部分。没有调用的构造两者都不写。其他策略的服务，无论 Qwen 还是 Gemma 4，都不写这些字段，也不写上面每次调用的用时（`test_gemma_goal_rewriter.py`）；带 rewriter 的 Gemma 4 `goal_query` 服务（4.14 节）通过与 Qwen 相同的代码写入它们，这一点没有单独测试。

**单个 session（带 rewriter 的 `goal_query`）。** 单 session 服务没有可供挂起的 Goal 线程：Goal 在同步的 admission 过程中构造，服务不写上面这些用时字段。自 review-fixes-1110 起，构造在单独的线程上运行，引擎线程等待它并应答管理进程的心跳（`goal_async.InlineGoals`），因此一次跑满 120 秒超时的 rewriter 调用会以上面的回退（`fallback_reason`）结束。此前由引擎线程自己构造 Goal，无法应答 90 秒的心跳，服务会先被杀掉。Goal、选择结果和证据都不变。仅在 CPU 上测试（`test_inline_goal_heartbeat.py`）。

干净关停需要同时满足（`examples/slurm/server_checks.py cleanup` 会检查这些回执）：
- `parent.cleanup_sha256 == SHA256(engine-cleanup.json 原始字节)`；
- 两份回执的 `joined == true`；
- `epoch`/`session_id` 一致；
- `engine.runtime_manifest == ready.runtime_manifest`，`engine.pending_operations == 0`；
- `parent.child_exitcode == 0`，`parent.natural_exit == true`；
- APIServer 进程实际以 0 退出。

如何理解传输数字：
- `restore.h2d_bytes` 统计的是这一轮模型执行期间所有 host 到 device 的拷贝，不只是注意力 Page，还包括混合模型的递归（Gated-DeltaNet）状态。对 Qwen3.6-27B，两种布局下一份递归状态都是 153,944,064 字节，一个 784-token 的注意力 Page 是 51,380,224 字节，一个 16-token 的注意力 Page 是 1,048,576 字节。
- 历史短、全部放得进 GPU KV 时，没有任何注意力 Page 需要从 CPU 恢复，所以所有策略的恢复字节数相同。
- `prepare.h2d_bytes` 是阶段 1 的开销（恢复从上一轮带过来的递归状态）；`restore.h2d_bytes` 统计本轮其余部分。
- 开 `--gdn-device-handoff`（4.10 节）时，递归状态在 device 上移动：延续轮次的 `restore.h2d_bytes` 只统计恢复的注意力 Page，`restore.d2d_bytes` 统计状态拷贝（一次，加上每次跨 Mamba block 一次）；`prepare.h2d_bytes` 仍是 carry。
- 只有在历史确实需要把 Page 换出到 CPU 时，比较各策略才有意义；并且要把以上几部分分开报告。

这些文件可能包含 prompt 和工具输出，不要放进公开的源码发布包。Joined transfers 只证明机制行为，不证明任务质量或加速。回执里 `NOT_SYSTEM_VALID`、`cuda_qualified=false` 这类历史标签是有意保守的能力描述；评判一次运行，要看它的 READY、逐轮 native 证据、清理回执和任务评测。
