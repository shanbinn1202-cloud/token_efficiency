# Goal overlap：在构造 Goal 的同时运行阶段 1（设计，2026-10-30）

[English](DESIGN.md) | 简体中文

2026-10-30 的设计（第 1-8 节对应 `cf2b28d`，第 9 节对应 `5c8bc24`）；实现和 Gemma 的后续工作都已在 `main` 上，`--goal-overlap` 需显式开启（默认关闭）。它是稳定配置的一部分（与 vLLM Goal 引擎一起，作业 3576565）。它的 GPU 运行及其判定见 [docs/status.zh-CN.md](../status.zh-CN.md) 和[历史](../history.zh-CN.md#历史)，Goal 路径的限制见 [docs/known-limits.zh-CN.md](../known-limits.zh-CN.md) 第 3 节；第 7 节的 GPU 检查测得的结果见第 8 节。

分支 `goal-overlap`，从 `cf2b28d` 切出。SO = `source-overlay/`。关于当前代码的每一条陈述都在 `cf2b28d` 的源码中核对过；行号是该 commit 的行号。

符号（`remote_ra_native_growing/contract.py` 中 `turn_geometry` 的轮次几何）：P 是页大小；C 是 session 的已计算终点，阶段 1 从这里开始；o 是 observation 的起点（stable end）；e 是 observation 的终点，所以阶段 1 计算 `T[C:e)`；N 是 prompt 长度；Q = floor(C/P)·P 是封存终点；R 是常驻 Page；n = floor(o/P) 是候选页数，L = n·P。G 是 Goal 的 token 数。A 和 M 是注意力 group 和 Mamba group 的个数（`capacity.py`；这个 M 不是 docs/exp-decouple-pages 中的 Mamba block）。K 是 `--admission-overlap` 的值。

## 0. 概要

**问题。** 一次 `goal_query` admission 先构造它的 Goal（在调用 rewriter 的轮次上由 Qwen3-0.6B rewriter 完成：要几秒），然后才运行阶段 1（该轮新 token 的 prefill 以及 Page 和 observation 向量，在 10-26 的 Terminal-Bench 运行中每轮约 150-250 ms）。Goal query 向量在阶段 1 的 forward pass 内部计算。能否在构造 Goal 的同时运行阶段 1，之后再计算 Goal 向量，并得到相同的向量？

**结论：可以，而且完全相同。** Goal forward 本来就是一次单独的冷 forward。它与阶段 1 在同一个 worker RPC 中运行，但在其之后，在它自己的 block 上，从位置 0 开始，不读历史 KV、该轮的新 token 或 GDN carry。它不读取阶段 1 计算或改变的任何东西（第 2 节）。之后在它自己的 probe-only pass 中运行时，它是同一个 `execute_model` 调用：相同的 token、相同的位置、相同的空上下文、相同的单请求 eager batch 以及相同的行宽。只有 block id 不同，而它们改变的是 KV 存放的位置，不是数值。因此 Goal 向量逐位相同，只差 GPU kernel 在不同运行之间的不确定性，而这种不确定性同样影响现在的运行（作业 3349078 重跑了一个服务，全部 2,866 个 Page 的阶段 1 向量都相同）。于是 Page 分数（与该向量的余弦）、排序和选页也都相同。

**设计。** 放在一个新的、需显式开启的参数 `--goal-overlap`（默认关闭）之后，在会挂起 `goal_query` admission 的服务上：
1. admission 在它的 Goal 线程上开始构造 Goal，但不挂起；
2. 它运行不带 Goal forward 的阶段 1（就地运行，或开 `--phase1-in-step` 时放进一个 decode 步里）；
3. 然后它 join Goal（只有 Goal 还没构造好时才挂起）；
4. 当该轮需要 Goal forward 时（选页生效的轮次上缓存未命中），它把 Goal forward 作为单独的一次 probe-only pass 运行，使用自己的租约，其回复带有它的 drain（除现在已有的之外不增加逐请求哈希：3.11）；
5. 接着是选页（`plan_turn`，不变）、adoption 和预留。

Goal 的构造（它的输入、构造器状态、rewriter 调用）不变：TurnContext 在与现在相同的位置构建，构造也从同一个位置开始。

**收益。** 每个 Goal 需要时间的轮次：admission 延迟减少约 min(Goal 用时, 阶段 1 用时)。在 10-26 的运行中，每个 rewriter 轮次约为 0.15-0.25 s（这些轮次 admission 的百分之几）；一旦更快的 rewriter 让 Goal 与阶段 1 一样快，最多约为 Goal 加阶段 1 关键路径的一半（`rewriter-vllm` 的 vLLM rewriter 没有那么快：第 4 节）。开 `--phase1-in-step` 时，需要 Goal forward 的轮次的阶段 1 也可以加入 decode 步（现在它仍单独运行一次），所以只有短的 Goal forward 仍会让 decode 停顿。

**实现与 GPU 检查。** 第 8 节。Gemma 4 profile（窗口布局）：第 9 节。

## 1. 现状（已核对）

### 1.1 一次 admission 的顺序

`SplitPrivateEndpoint.reserve_steps`（SO/remote_ra_live_v2/private_endpoint.py:221-283）：
1. `geo = turn_geometry(...)`；
2. `query = yield from selection.begin_turn_steps(binding, geo, conversation)`（237-240）：构建 TurnContext，在 Goal 可能调用 rewriter 时 yield 一个 `GoalWork`（query_selection.py:906-909），然后构建 Goal query 文本，查找 Goal 缓存（以 `retrieval_goal_hash` 为键），并在该轮需要 Goal forward 时设置 `goal_payload`（912-921）；
3. 对 local turn > 0：`--release-idle-pages` 的重新载入，`--admission-overlap` 的 `lease_turn`，然后以 `goal_payload=query.goal_payload` 运行阶段 1（242-259），就地运行（`run`），或开 `--phase1-in-step` 时作为一个 `Phase1Work`（`run_steps`）；
4. `selection.plan_turn(binding, geo, generation, query, phase1)`（261），`phase1.adopt`，`adapter.reserve`。

失败：一旦阶段 1 运行过，该 session 就失效，尚未被 adopt 的阶段 1 会被中止（270-280）。

### 1.2 Goal forward 在哪里运行

引擎，`Phase1Coordinator.run`（phase1_coordinator.py:135-289）：
- 用 `PLAIN_GOAL_SCHEMA` 对 `goal_payload` 做 tokenize（原样，不加特殊 token；167-176），上限为一次 native forward；
- 在阶段 1 的 block 旁边分配 Goal 自己的 block：每个注意力 group `ceil(G/P)` 个 FA block，每个 Mamba group 一个 Mamba block（202-203）；
- 把它们作为 `lease['goal']` 放进阶段 1 的租约：rows（`width` 个 block 的 FA 行，用 null 填充的 Mamba 行）、owned、zero（FA 部分）、incarnations、`payload_sha256`（224-233）；
- 把 `query_tokens`（仅用于证据）设为 Goal 的 token（279-280）。

容量门控把 Goal block 计入阶段 1 的峰值：`A·(fresh + goal + evicted)` 个 FA block 和 `2·M` 个 GDN block，即每个 Mamba group 两个各占一个 block 的 GDN slot（capacity.py:211-222）。gate 无法知道 G，所以 `goal` 取 Goal 的上限 `pages(phase1.max_tokens, P)`（一次 native forward；窗口布局下为 `goal_tokens`），在 `goal_query` 轮次上选页可能生效时计入。

Worker，`NativeQueryForward.phase1`（native_probe.py:144-193）：
1. 在 `[R][tail][fresh]` 上运行阶段 1 的 forward（一次或多次）；
2. `_cleanup_request(rid)`：阶段 1 的请求离开 runner，它的 KV 保留；
3. 运行 Goal forward：`self._execute(goal.lease_id, goal.tokens, 0, goal.rows, goal.zero)`，`phase='goal'`（179-186）。

`model_queries.phase1` 对 Goal 的最后一行做归一化（final norm 之前，`pre_final_norm_last_token_l2`；126-139），并把它随阶段 1 回执一起签名（`goal_vector`、`goal_token_count`、`goal_tokens_sha256`、`goal_payload_sha256`）。`plan_turn`（query_selection.py:1072-1082）用 `phase1.goal_vector` 作为 query（命中时用缓存的向量），并把它存入缓存。

### 1.3 Goal 的行关注和读取什么

| 输入 | 阶段 1 | Goal forward |
| --- | --- | --- |
| token | `T[C:e)`（新 token） | Goal query 文本，单独 tokenize |
| 位置 | 逻辑位置 `C … e-1`（`SparsePlan` 把 FA slot 映射到 `x - (Q - |R|P)`，RoPE 保持逻辑位置） | `0 … G-1`（`start=0`；`text_positions` 用同一段代码） |
| 读取的 FA KV | 常驻 Page R、live tail、它自己的 fresh block | **只有它自己的 fresh block**，经 `new_block_ids_to_zero` 清零并由这次 forward 写入。没有 `SparsePlan`（阶段 1 的 plan 已被 `_cleanup_request` 弹出），所以没有历史 KV |
| GDN 状态 | 恢复到 fresh slot 中的已提交 carry（`before_forward`，native_probe.py:279-288，只用于 `phase == 'phase1'`） | 一个 fresh slot。vLLM 让没有已计算 token 的请求从零状态开始（`has_initial_state = context_lens > 0`，SO/vllm/v1/attention/backends/gdn_attn.py:390），所以从不读取该 slot 的内容 |
| 写入的状态 | `e` 处的 GDN 状态（被捕获）、tail 的捕获、为阶段 2 保留的 fresh FA KV | 只写它自己的 block，在 joined drain 时释放 |
| 输出 | Page 向量、observation 向量（第 `e-1` 行） | 最后一行，final norm 之前，L2 归一化 |
| batch | 一个请求，probe-only，eager（`force_eager`），或开 `--phase1-in-step` 时作为 decode 步的一行 | 一个请求，probe-only，eager；带 Goal 的租约从不进入 step（phase1_step.py:133-139） |

除了模型权重，Goal forward 不读取阶段 1 写入的任何东西，也不读取阶段 1 读取的任何东西。

### 1.4 现在的挂起

`goal_async.GoalAdmissions` 驱动 admission 的 generator。一个 `GoalWork` 会挂起：它被提交给 Goal 线程，开 `--admission-overlap K` 时提交给该 session 自己的 Goal 线程。引擎继续执行 step，admission 每个引擎 pass 恢复一个，最早完成的先恢复。然后阶段 1 在引擎线程上运行。证据：`goal_async_ms`、`goal_construct_ms`、rewriter 的总计、`phase1_park_ms`、`lease_wait_ms`；挂起的时间从 `admission_engine_ms` 中扣除（registered_route.py:65-70）。

## 2. 单独的 Goal forward 的等价性

**相同的输入。**
- token：同一个 `goal_payload`，由同一次 renderer 调用 tokenize。
- 位置：相同，0 到 G-1。
- 上下文：相同，都没有（没有历史 KV、没有新 token、没有 carry）。
- 行：宽度相同（`ceil(G/P)` 个 FA block，一个 Mamba slot）。
- 清零：相同的 zero 列表（FA block）。
- 表示：相同，final norm 之前的最后一行，L2 归一化。

**相同的计算。** worker 通过同一个 `_execute` 调用运行它，同样作为一次 probe-only pass。runner 先从它的 persistent batch 中去掉其他请求（vLLM 的 `_update_states` 会去掉未调度的请求），与现在对阶段 1 的做法相同，所以 batch 同样恰好是一个 G token、`num_computed_tokens = 0` 的请求。无论 CUDA graph 模式如何，probe-only pass 都以 eager 方式运行，所以 kernel、它们的 launch shape 以及依赖 shape 的 split 启发式都与现在的 Goal forward 相同。block id 不同（不同的分配），但 paged attention 通过 block table 读取数值，所以 id 改变的是 KV 存放的位置，不是计算的内容。

**完全相同，而不只是“数值误差范围内相同”。** 在同一个服务进程和 GPU 上，单独的 forward 以与现在 RPC 内的 Goal forward 相同的顺序计算相同的浮点运算。可能让两次运行不同的是 GPU 自身在不同运行之间的不确定性（例如每个进程各自的 Triton autotuner 选择）。它同样影响现在的运行，而且没有观察到（作业 3349078，重跑的服务上阶段 1 向量相同）。在同一 GPU 上同时运行 rewriter 不会改变确定性 kernel 的结果，只改变它们的速度。

**选页。** Page 分数是每个 Page 向量与 query 向量在经过相同的投影和 int8 压缩之后的余弦（`query_vector`、`plan`）。Page 向量来自阶段 1，它不变：同一次 forward，只是现在同一个 RPC 中它之后不再有 Goal forward，而它与那次 Goal forward 本来就没有共享任何东西。query 向量相同时，每个分数、排序、选中的页数和签名的回执都相同。Goal 缓存存的也是同一个向量。

**仍可能不同的地方，与现在一样。**
- Goal 本身只在挂起的 Goal 现在就可能不同的地方才会不同（goal_async.py:24-33）：一次因共用 GPU 而变慢的 rewriter 调用（现在还会与阶段 1 的 prefill 共用）可能达到它 120 s 的超时并回退（`fallback_reason`）。
- 开 `--goal-rewriter-batch` 时，哪些调用合成一批取决于时序，而批量 greedy 解码并非逐位相同（[docs/reference.zh-CN.md 4.10 节](../reference.zh-CN.md#410-admission-优化需显式开启)）。任何时序变化都是如此。
- 阶段 1 的数值只在 `--phase1-in-step` 本来就会改变它们的地方改变：目前因为带 Goal forward 而单独运行的阶段 1，现在可能放进 step 里运行，就像开这个参数时其他每一轮的阶段 1 已经做的那样。放进 step 的阶段 1 会改变阶段 1 向量：在作业 3349078 中，s-all 与 s-off（`mixed_share` 0.9375：大多数阶段 1 加入了 step）在比较的 51 轮中 51 轮不同（5,752 个 Page 的余弦低于 1，中位数 0.9986）；单 session 的几对服务中没有阶段 1 加入 step，它们完全相同（2,866 个 Page 中 2,866 个）。

## 3. 设计（`--goal-overlap`）

### 3.1 开该参数时的顺序

`goal_query` session 上开了 `endpoint.goal_overlap` 时的 `reserve_steps`：
1. `turn = yield from selection.begin_turn_overlap(binding, geo, conversation)` 运行 `begin_turn_steps` 的前半部分：它的检查、TurnContext、摘要子调用的 Goal（`subcall_goal`，不用 rewriter）、hand-off 的释放。可能调用 rewriter 的 Goal 以一个 `GoalStart(work)` yield 出来，不挂起。driver 把它提交给 Goal 线程，admission 立即继续。
2. 重新载入、`lease_turn` 和阶段 1，`goal_payload=None`：就地运行，或开 `--phase1-in-step` 时作为一个会挂起的 `Phase1Work`。
3. `query = yield from turn.finish()`：一个 `GoalJoin` 返回 GoalRecord。它只在 Goal 还没构造好时挂起。然后是 `begin_turn_steps` 的后半部分，不变：Goal 文本、缓存查找、`goal_payload`、`pending`。
4. 如果阶段 1 运行过且设置了 `query.goal_payload`：`phase1 = yield from self.phase1.goal_forward_steps(...)`（3.3）。
5. `plan_turn`、`adopt`、`reserve` 与现在相同。

`begin_turn_steps` 拆成 `_begin` + `_finish`，语句相同、顺序相同，所以不开该参数的路径只是重构，不是改动。

### 3.2 Goal 的开始和 join（goal_async）

- `GoalStart(work)`（kind `goal_start`）。
  - `GoalAdmissions._advance` 提交它（`_start`）：提交给 Goal 线程，开 `--admission-overlap` 时提交给该 session 的线程，计时器 `_build` 和唤醒回调与挂起的 Goal 相同。然后它不发回任何东西，继续执行。
  - `run_inline`（没有挂起表）就地构造它，这就是现在的顺序。
- `GoalJoin(start)`（kind `goal`）。
  - 只在 Goal 的 future 尚未完成时挂起（`_parks`）。挂起时，它的 future 就是 Goal 的 future：`ready`/`resume` 不变。
  - join 时已完成：它就地运行，返回记录或抛出构造时的错误。
  - 挂起时被中止：future 被取消，在 join 处抛出 `GoalAborted`。
- 已开始但从未被 join 的 Goal（阶段 1 失败，或在步内的 `Phase1Work` 或 `LeaseWait` 处挂起时被中止）在 `reserve_steps` 的失败路径中被丢弃（`future.cancel()`；正在运行的构造在它的线程上完成，结果被丢掉），与现在被中止的挂起 Goal 完全一样。无论如何该 session 都会因这次失败而失效，而 session 关闭时会在它的 Goal 线程上、排在该线程上仍在运行的 Goal 之后释放它的 Goal 状态（`after_goals`）。

### 3.3 不带 Goal 的阶段 1；单独的 Goal forward

阶段 1 就是现在的租约加 `goal=None`：现在每个不带 Goal forward 的轮次走的就是这条代码路径。

`Phase1Coordinator.goal_forward_steps(binding, geo, phase1, goal_payload, allocation_generation)`：
1. 开 `--admission-overlap` 时：`lease_turn`（3.4）。
2. 像 `run` 一样做 tokenize，并像 `run` 一样分配 Goal 的 block（每个注意力 group FA `ceil(G/P)` 个，每个 Mamba group 一个 Mamba block）。
3. 构建一个 Goal-forward 租约：现在的 Goal 租约（`GOAL_LEASE_KEYS`），加上该轮的身份（request、binding、session、namespace、allocation generation、epoch、source、local turn、页大小）以及该轮已 drain 的阶段 1 租约的 id。它把这个租约记入 `goal_records`。该租约**不签名，也不做摘要**（3.11）。
4. 调用 worker 新的 `goal_forward` 操作（`query_tokens` = Goal 的 token，与现在阶段 1 的调用一样）。它的回复带有 worker 对该租约的 joined drain。引擎检查它，释放 Goal block，并返回 `dataclasses.replace(phase1, goal_vector=…, goal_payload_sha256=…, goal_token_count=…)`。

如果调用失败（没有回复），引擎在释放任何东西之前，先以 Goal 请求通过 `phase1_drain` 请求 drain（幂等）：与阶段 1 一样，它从不根据错误推断已经释放。drain 失败时 Goal block 保持被引用，由 `abort` 重试。`plan_turn` 和预留不变。

Worker：
- `RunnerAllocationOwner._goal_forward`：只在固定了该参数时（`owner.goal_overlap`）接受，在本轮阶段 1 之后、`reserve` 之前，只接受一次。
- `model_queries.goal_forward`：阶段 1 的身份检查，以及对租约（`contract.parse_goal_lease`，只检查结构）对照已 prepare 的轮次及其阶段 1 租约（其 epoch 已由阶段 1 检查过）的逐字段检查。
  - 它驱动 `NativeQueryForward.goal_forward`，后者运行现在的 Goal 分支：`_execute(lease_id, tokens, 0, rows, zero)`，`phase='goal'`。
  - 这次 forward 在它的 `finally` 中 drain（`drain_goal`）。
  - 回复不签名，与 `prepare` 和 `reserve` 的回复一样。它包含现在的 Goal 字段（`goal_vector`、`goal_token_count`、`goal_tokens_sha256`、`goal_payload_sha256`）、位置证据、`goal_ns`、`drained` 和 drain 记录。
- driver 把 Goal 租约登记在它自己的租约表中，并像阶段 1 租约一样对照 claim、被保留的 slot 和未 drain 的租约做别名检查。所以 `reserve`、`complete` 和 `cancel` 所要求的 `request_drained` 覆盖它，`close_session` 的租约清理和 `evict` 的 drain 检查也覆盖它。对 Goal-forward 租约做 `phase1_drain` 会返回那份 drain 记录（从未开始的租约被记为 tombstone），不签名。

### 3.4 租约：同一时间只有一个未 drain 的租约

现在引擎同一时间只允许一条阶段 1 租约记录，直到 adoption 或中止（`require(not self.records)`，phase1_coordinator.py:153，以及 `LeaseWait`）。拆分之后，admission 在等待 Goal（几秒）时持有它已 drain 的阶段 1 记录。按旧规则，其他每个 admission 的阶段 1 都要等这个 Goal，这会抵消 `--admission-overlap` 带来的收益。worker 只要求之前的每个租约都已 drain（native_probe.py:96-98）。

所以开该参数时，引擎的规则变成**没有未 drain 的租约**（`Phase1Coordinator.busy()`）：
- 已 drain 的记录持有 fresh 的 FA run block（开 hand-off 时还有 probe slot），其他任何分配都不能与它们别名：它们被引用，且不在空闲队列中。
- B 的阶段 1 属于另一个 session（每个 session 一个 admission）。
- A 的 Goal-forward 租约也是租约：B 的步内阶段 1 尚未 drain 时（从它的租约到它恢复），它等待（`LeaseWait`）；A 的 Goal forward 运行时，B 等待。那次 forward 在一次引擎线程调用内运行，并在那里 drain。

不开该参数时：`busy()` 就是 `bool(self.records)`，即现在的规则。

### 3.5 `--gdn-device-handoff`

probe slot 从阶段 1 到阶段 2 的第一步一直保存 `e` 处的 GDN 状态（`HeldSlots`）。开该参数时，admission 等待 Goal 期间它也被保留。还是同一个 slot、同样的 claim（`held_claims` 拒绝任何别名，包括 Goal 租约的 block）和同样的释放规则；只是保留得更久。Goal forward 自己的 Mamba slot 是 fresh 的，从不被保留。

### 3.6 `--phase1-in-step`

开该参数时阶段 1 的租约从不带 Goal forward，所以 `Phase1Steps.accepts` 像其他任何轮次的阶段 1 一样把它放进 step。Goal forward 是 admission 恢复时的一次短 probe-only pass：在有 Goal forward 的轮次上多一次单独的 pass，但只针对 Goal 的 token。不把它放进 step 就能保持完全相同（2）。步内的 Goal forward 会改变它的 batch shape，所以只能在数值误差范围内相同；它不在范围内（5）。

### 3.7 `--admission-overlap K` 与容量

- Block：该轮的峰值不增加。现在阶段 1 同时持有 fresh FA run、probe slot、Goal 的 FA block 和 Goal slot。开该参数时，Goal block 在阶段 1 的 drain 之后才分配：run 仍被持有，但 probe slot 已释放（开 hand-off 时除外）。所以现有的阶段 1 数字（它计入了 Goal block，capacity.py:211-222）已经覆盖这种情况。
- Decode：admission 在阶段 1 之后而不是之前等待 Goal 时，运行中的请求照常 decode。Goal 挂起时，gate 已经在阶段 1 数字之外计入它们的 headroom（`later = headroom`）。
- 账本：已 drain 的记录所持有的从该 admission 的预留中扣除（`held`），`held` 也计入 Goal 记录的 block。
- 死锁：没有任何东西在持有 block 的同时等待 block。挂起的 join 只需要它的 Goal 线程，Goal forward 只需要它已经持有的预留。

### 3.8 取消、过期、关停

- 在 join 处挂起：`GoalAdmissions.abort` 取消 future，并在那里抛出 `GoalAborted`。因为阶段 1 已经运行，`reserve_steps` 的失败路径使该 session 失效，并中止阶段 1（释放 run，移走 hand-off slot）和任何 Goal 记录。随后 lifecycle 像对任何未完成的 admission 一样向 worker 发送 `cancel`。这种请求的 native 文件中有阶段 1 的帧，这与现在在 Goal 处取消不同。
- Goal 已开始、在步内阶段 1 或 `LeaseWait` 处挂起：同上，另加 3.2 的丢弃。
- 关停（`close`）中止挂起的东西，然后用 `cancel_futures` 关闭 Goal 线程，与现在相同。

### 3.9 Gemma 4（窗口布局）

Phase B 在 Gemma 4 profile 上拒绝 `--goal-overlap`，因为窗口账本依赖于“admission 只在阶段 1 之前在它的 Goal 处挂起，之后不在任何地方挂起”（`window_capacity.ledger_ahead`）。后续工作（分支 `gemma-overlap`）在那里启用了它；分析见第 9 节。这里此前的说法，即等待 Goal 的已 drain 阶段 1 按账本自身的算法已被多预留，是错的：不改变窗口布局的话，这样的等待会让 pool 1 的 block 耗尽（9.3）。

### 3.10 证据

只在开该参数时：
- 有 Goal forward 的轮次上，阶段 1 的帧之后的 worker 帧 `goal_forward`。它的 drain 记录在其中（`drain`），所以没有单独的 `phase1_drain` 帧；只有 Goal forward 失败后才会出现一个；
- native 文件：`goal_overlap_ms`（从 Goal 开始到它的 join：期间引擎线程上完成的工作，包括阶段 1）；`goal_forward_ms`（运行了单独的 Goal forward 时，它在引擎线程上的 `goal_forward` RPC；worker 自己的用时是回复中的 `goal_ns`）；`goal_async_ms`，含义不变（在 Goal 处挂起的时间，join 时 Goal 已构造好则为 0；与以前一样从 `admission_engine_ms` 中扣除）；`goal_construct_ms` 和 rewriter 的总计与现在相同。

`runtime.jsonl` 的行与不开该参数时相同：有 Goal-forward 帧时，`stage_b.phase1.goal_forward` 和 `query.query_token_sha256` 取自该帧，prepare 的 H2D 总和也计入该帧（0 字节：这次 forward 是冷的）。`goals.jsonl` 不变。

不开该参数时什么都不变：没有配置键，没有 worker 或 READY 键，没有帧，没有字段。

### 3.11 参数的接线

- `serve.py --goal-overlap`：只用于 Qwen 上带 `--goal-rewriter-model` 的 `goal_query`。
- 它固定 `admission.goal_overlap = true`（`admission_flags.FLAGS`），worker 回显它，READY 记录它；它在启动时检查一次，不增加任何 vLLM 参数。
- 引擎在 `goal_async` 之后接上它（`goal_overlap.attach_engine`）。在没有 Goal 挂起表的服务上（一个 session，`max_num_seqs` 为 1），它不起作用，与 `--phase1-in-step` 一样：每个 admission 都保持现在的顺序。
- worker 只在固定了该参数时接受 `goal_forward`。
- **除了不开该参数的轮次已有的哈希，不增加逐请求哈希**（规则：完整性检查只在启动时做）。
  - Goal-forward 租约没有 MAC，也没有租约摘要或请求摘要。worker 对照已 prepare 的轮次及其阶段 1 租约逐项比较相等来检查它，而后两者保留现在的 MAC 和摘要。epoch 与那个租约的 epoch 比较，而不是再哈希一次。
  - `goal_forward` 的回复及其 drain 记录不是签名的回执。
  - 在 CPU harness 中，runtime 源码的每次 SHA-256/HMAC 调用，无论总数还是按函数统计，都与不开该参数的服务相同，只是每次 Goal forward 有两处例外，各移动一次调用。Goal payload 的摘要从阶段 1 的租约（`run`）移到 Goal forward 的租约（`_goal_lease`）。Goal token 的摘要从阶段 1 的回执和 `_result` 移到 Goal forward 的回复和 `_goal_result`，两者都在 `contract.digest` 内部，所以那里的计数不变。`test_goal_overlap.py` 固定了这一点。
  - 第一版实现（`230f9a9`）按 CPU harness 的统计，每次 Goal forward 多做 20 次 SHA-256/HMAC 调用（commit `5c8bc24`）。

## 4. 预期的收益和代价

**收益**：对于阶段 1 本该开始时 Goal 还没构造好的每一轮，admission 大约提前 min(T_goal, T_phase1') 完成，其中 T_phase1' 是该 admission 的阶段 1 工作（重新载入、租约等待、阶段 1）。
- 在 10-26 的 Terminal-Bench 运行中（阶段 1 150-250 ms，rewriter 要几秒），这就是每个 rewriter 轮次的阶段 1 用时：这些轮次 admission 延迟的百分之几。它不是引擎时间：阶段 1 和 Goal forward 仍在引擎线程上运行。
- 开 `--admission-overlap 16` 时，其他 admission 在构造 Goal 期间本来就能继续，所以对吞吐量的影响很小。节省的是 rewriter 轮次自己的延迟。
- 一旦 rewriter 变快，节省的时间接近 min(T_goal, T_phase1)，两者相等时最多为 Goal 加阶段 1 路径的一半。vLLM rewriter 没有那么快：在 rw3（作业 3499550）中，一次调用单独运行时的墙钟时间中位数约为 975 ms，16 个调用同时进行时为 16.4 s。
- 开 `--phase1-in-step` 时，有 Goal forward 的轮次也让 decode 停顿得更少（只有 Goal forward 仍单独运行一次）。

**代价。**
- 有 Goal forward 的轮次多一次 probe-only pass 和一对 RPC：运行中的 decode 要多被去掉并重新加入一次。forward 本身的工作量相同。
- 等待 Goal 期间，阶段 1 记录以及（开 hand-off 时）probe slot 被保留。它们都在该轮已有的预留之内。
- rewriter 不只与 decode 步共用 GPU，还与阶段 1 的 prefill 共用。它的调用可能更慢；要测量 `call_ms` 和回退。

## 5. 不在范围内：改变 Goal 行为的 overlap

这些属于 Goal 负责人。每一项都会改变一轮使用哪个 Goal，或它在何时构造：
- **更早启动 rewriter。** 例如请求到达前端时（在 admission gate 或 `CAPACITY_WAIT` 之前），或在上一轮 decode 期间从部分 observation 开始。两者都会改变 TurnContext，或读取构造器状态的时间。
- **用上一轮的 Goal 做本轮的选页**，并为下一轮异步更新 Goal：按设计就是一个过期的 query。
- **跳过 rewriter 或对它做投机**（例如先用确定性 Goal 选页，rewrite 结果不同时再重新选页）。
- **在 decode 步内做 Goal forward。** 它不改变 Goal 的语义，但只在数值误差范围内相同（batch shape 不同），所以需要它自己的参数和一次数值比较。

## 6. 风险

- rewriter 调用被同时进行的阶段 1 prefill 拖慢（超时会回退并被记录）。
- 开 `--admission-overlap` 时阶段 1 记录被保留得更久：放宽后的租约规则（3.4）是新的。worker 本来就只要求“全部已 drain”；CPU 测试覆盖了两个 admission 交错的情况，在 GPU 上作业 3500565 运行了最多 16 个同时进行的 admission（第 8 节）。
- Goal join 处的失败现在发生在阶段 1 运行之后：无论哪种情况 session 都会失效，但失败时的帧与现在不同。
- Goal forward 的租约和回复没有 MAC 或摘要（3.11）。它们的完整性依靠进程内的引擎–worker 通道、worker 对照该轮签名的阶段 1 租约所做的相等检查，以及别名检查。这与不签名的 `prepare` 和 `reserve` 回复处于同样的地位。
- Gemma 4：第 9.7 节。

## 7. GPU 检查应当测量什么

在一张 GH200 上，Qwen3.6-27B，`goal_query` 加 Qwen3-0.6B rewriter：开与不开 `--goal-overlap` 时运行相同的脚本和 session（再加一次同时开 `--phase1-in-step`）。
1. **相等。** 每一轮：Goal 向量的字节（`goal_forward` 帧对照不开该参数时的阶段 1 帧）、Page 向量、选页和输出 token。在 Goal 相同的轮次上，它们应当逐轮相等。rewriter 调用不要批量运行，否则就要接受 batch 组成带来的噪声。
2. **收益。** 每个 rewriter 轮次：`goal_overlap_ms`、`goal_async_ms`、`goal_construct_ms` 和 admission 延迟（ACK 等待）。预期节省约 min(`goal_construct_ms`, `goal_overlap_ms`)。
3. **代价。** 有 Goal forward 的轮次上的 `goal_forward_ms` 和 decode 停顿（`admission_engine_ms`、token 间延迟）；开与不开该参数时 rewriter 的 `call_ms` 和回退。
4. **容量。** 开 `--admission-overlap 16` 时：`CAPACITY_WAIT` 的次数和 `lease_wait_ms`，没有抢占，运行后两个池都回到基线。

## 8. 实现与 GPU 检查

Phase A：本设计（commit `d8774fd`）。Phase B：在本分支上实现，放在 `--goal-overlap` 之后（默认关闭）。`230f9a9` 加入了实现。其后的后续 commit 去掉了 Goal forward 额外的哈希（3.11）：它的 drain 并入它的回复，它的租约和回复不再签名或做摘要。Gemma 4 起初拒绝该参数（3.9）；分支 `gemma-overlap` 在那里启用了它（第 9 节）。GPU 运行及其判定：[docs/status.zh-CN.md](../status.zh-CN.md)。

代码：
- 引擎：`goal_overlap.py`（attach、`GoalForward`），`goal_async.py`（`GoalStart`、`GoalJoin`），`query_selection.py`（`_begin` / `_finish`、`begin_turn_overlap`、`StartedTurn`），`private_endpoint.py`，`phase1_coordinator.py`（`busy`、`goal_forward_steps`、`recover_goal`），`composition.py`；
- worker：`allocation_owner.py`、`model_queries.goal_forward`、`native_probe.goal_forward`、`gdn_handoff.attach_worker`；
- 共用：`contract.parse_goal_lease`；
- 证据和 launcher：`remote_ra_http/evidence.py`、`remote_ra_http/config.py`、`serve.py`。

不开该参数时，每个新增的代码分支都被跳过或为空（`goal_records` 保持为空，`busy()` 就是 `bool(records)`），现有的测试集照常运行。

`test_goal_overlap.py`，23 个测试：
- launcher、配置、attach 和 worker 的拒绝；
- 各 driver；
- 租约规则；
- Goal-forward 租约；
- 在成批的 HTTP 假实现上端到端：与每个 session 单独在不开该参数的服务上运行时等价（Goal、Goal 向量、排序、选页、token、`runtime.jsonl`、plan、`goals.jsonl`）；rewrite 被扣住时运行阶段 1；在 join 处断开连接；失败的 Goal forward 经 `phase1_drain` drain；`--admission-overlap 2` 下有一个已 drain 的租约在等待；
- 在解耦布局的 16-token 假实现上开 `--admission-overlap 3` 端到端运行，单独开、加 `--gdn-device-handoff` 以及加 `--phase1-in-step`：每个决策都等于全部关闭的服务的决策，有 Goal forward 的轮次把它们的阶段 1 混合放进 step 中运行；
- 哈希：每次 SHA-256/HMAC 调用都与不开该参数的服务相同，只有 Goal payload 的摘要从 `run` 移到 `_goal_lease`。

第 7 节的 GPU 检查测得的结果（cmg1，runtime `c07d42a`；取自各作业的离线核验，未审计）：
- **Qwen，作业 3500604**，两对服务都开启 thinking。strip 一对 qref / qovl（`--thinking-history strip`）在 30 个 lockstep 轮次上为 EXACT：30 轮中 30 轮的 Goal 相同，比较了 18 个有 Goal forward 的轮次。keep 一对 kref / kovl（`--protocol mswea-xml`，只开 `--enable-thinking`）在 18 轮上为 EXACT，其中 8 轮有 Goal forward；这一对比较的是开与不开该参数，所以它并不检查 keep 模式本身。
- **Qwen 加 `--admission-overlap 16`。** 在 SWE-bench 作业 3500565 中（服务 goalvllm，使用 [vLLM Goal 引擎](../rewriter-vllm/DESIGN.zh-CN.md)），`--goal-overlap` 在最多 16 个 admission 同时进行时运行：2,093 次 admission，1,149 次 Goal forward，没有抢占。
- **Gemma 4（第 9 节），作业 3538523。** gref、govl 和 gvll 各自回答了 49 个请求中的 49 个；gref / govl 一对在 25 个 lockstep 轮次上为 EXACT（25 轮 Goal 相同，比较了 15 个有 Goal forward 的轮次），govl / gvll 一对（gvll 使用 vLLM Goal 引擎）在 23 轮上为 EXACT（22 轮 Goal 相同：有一轮 Goal 不同，该 session 的 lockstep 在那里结束；比较了 12 个有 Goal forward 的轮次）。

## 9. Gemma 4 31B IT（窗口布局；分支 `gemma-overlap`，2026-10-30）

在 `5c8bc24` 的源码中核对。数字都是 Gemma 的：W = 1,024（滑动窗口），P = 16，N = 8,192（一次 native forward），G = 2,048（Goal 瞬态，`model_profile.GEMMA4_GOAL_TOKENS`），五个滑窗组；每个滑窗组一个窗口 slot 是 ⌈W/P⌉ + 1 = 65 个 block，一次 forward 的瞬态 512 个，Goal 瞬态 128 个。

**结论：完全相同，而且改动是局部的。** Goal forward 单独运行时，Gemma 的 Goal 向量逐位相同（9.2）。阶段 1 之后的挂起需要对窗口布局做两处改动，都放在该参数之后：阶段 1 的 drain 释放 adoption 本会释放的 session 已钉住的窗口 block（`WindowPins.trim`），滑窗账本把等待中的 admission 仍持有的部分从它的 base 中扣除（`window_capacity.ledger_ahead_split`）。没有它们就不安全（9.3）。pool 1 的大小、gate 的数字以及每条不开该参数的路径都不变。

### 9.1 现在的窗口布局

- **池。** pool 0 以 P token 的 block 存放全局（全注意力）层：即 Page。pool 1 存放滑窗层，在启动时预留：一个 null block，加上每个滑窗组的 S 个窗口 slot、一次 forward 的瞬态，以及（`goal_query` 服务上）Goal 瞬态（`kv_layout.window_pool_blocks`）。
- **窗口钉住（W-GPU）。** 两轮之间，session 在每个滑窗组中钉住它下一次 query 要读的 block，即 `kept_blocks(C)`（≤ 65；`WindowPins`）。
- **阶段 1。** 它借用这个窗口（null head、已钉住的 block，只读），并为 [C, e) 追加 fresh 的 pool-1 block（`sliding_window.phase1_row`）。长于一次 forward 的 run 在一个租约上按 4,096 的分块运行，已离开窗口的 token 的 fresh block 被重用（`phase1_ring`）。
  - drain 时，阶段 1 结束时所在窗口（`kept_blocks(e)`）之前的 fresh block 被释放。其余的（`stay`）等待 adoption。
  - adoption 把阶段 1 结束时所在的窗口（从 `kept_blocks(e)` 的第一个下标起的已钉住 block，加上 `stay`）交给阶段 2 的预留，并释放更早的已钉住 block（`WindowPins.handover`）。
- **Goal forward（现在）。** 它在阶段 1 的 RPC 内、阶段 1 的 forward 之后作为冷 forward 运行。它的行是它自己的 block：⌈G/P⌉ 个 FA block（清零）和每个滑窗组 ⌈G/P⌉ 个滑窗 block，没有 null head，token 数最多为 G（`WindowPhase1Coordinator.run`）。它的滑窗 block 取自 Goal 瞬态；pool 1 预留它，是因为 Goal forward 运行时阶段 1 的 fresh block 仍被持有。
- **Gate 和账本。** gate 的全局数字按 Qwen 的规则计算，滑窗数字取自 `window_capacity`。开 `--admission-overlap K` 时，每个进行中的 admission 预留它这一轮的滑窗 *base*（被接受后在它 session 旧的钉住之外还保留的部分：max(adopted 窗口, 阶段 2 峰值) − 旧窗口），再加上所有 admission 共用一次的最大 *transient*（peak − base：fresh ring 和 Goal forward）（`ledger_ahead`）。这是可靠的，因为现在 admission 只在它的 Goal 处、在阶段 1 之前挂起，什么都不持有。从它的阶段 1 租约到它被接受，它都在一次引擎线程调用中运行。这种布局拒绝 `--phase1-in-step`。

### 9.2 等价性：Gemma 的 Goal 向量

单独的 Goal forward 与现在 RPC 内的那次是同一个 `_execute` 调用：
- 相同的 Goal token（`<bos>` 和 `<retrieval_query>` 文本，同一次 renderer 调用）；
- 位置 0 … G−1；
- 宽度相同的行（每个 group 中 ⌈G/P⌉ 个 block，滑窗行没有 null head）；
- 相同的 zero 列表（pool-0 的 id）；
- 一个单请求、probe-only、eager 的 batch。

新的代码路径是 `WindowPhase1Coordinator._goal_lease` 和 worker 的 `NativeQueryForward.goal_forward`，后者没有窗口专用的代码。

它读取的内容：
- **全局层：** 只有它自己的 FA block，已清零并由这次 forward 写入。它的 request id 没有 SparsePlan，也没有历史 KV。
- **滑窗层：** 只有它自己的 pool-1 block。
  - 位置 p 上的 query 读取 [max(0, p − W + 1), p]，它们都在同一次 forward 中更早写入。G > W 时窗口在 forward 内部移动，与现在一样。
  - 这些 block 不清零（KV zeroer 只写 FA 张量），与现在相同，也不会读取这次 forward 自己的位置之外的位置。
  - 它完全不读窗口状态：不读 session 已钉住的窗口（只有阶段 1 借用它），不读阶段 1 的 fresh block 或 chunk-ring block，也不读被 trim 的 block（9.3）。

只有 block id 不同。所以向量逐位相同，只差 GPU 在不同运行之间的不确定性，与 Qwen 一样（第 2 节）。向量是 final norm 之前的最后一行（`ProbeRows.last`，fp32），L2 归一化，与现在的回执相同。

Page 向量来自阶段 1，现在同一个 RPC 中它之后不再有 Goal forward。那次 Goal forward 与阶段 1 不共享任何东西，所以 Page 向量不变。分块时，Goal forward 原来在最后一个分块之后运行，现在仍然如此。CPU 测试逐轮比较 Goal 向量的字节：worker 的 `goal_forward` 回复对照不开该参数时的阶段 1 回执。

### 9.3 阶段 1 之后的挂起需要什么

**只去掉拒绝是不安全的。** drain 之后等待 Goal 的一轮持有该 session 的整个旧窗口（每组 ≤ 64 个 block）以及新窗口的 `stay` block（最多 65 个）。以前，两者都只存在于一次引擎线程调用之内。账本把这些持有算作该 admission 的租约（`held`），并从它的 *transient* 中扣除，而 transient 只预留一次，取最大的那个。于是它掩盖了其他 admission 的需求：下一个 admission 的阶段 1 或一个等待中的 admission 的 Goal forward 可能找不到足够的空闲 block。在 GPU 上，那个阶段 1 的 admission 会失败（`insufficient native blocks`，session 失效），而找不到 block 的 decode 就是一次抢占。
- 在 Gemma 的几何上对 pool 1 的模拟（`test_gemma_goal_overlap.simulate`）显示了这一点：16 个 session，K = 16，随机的 run（1 到 12,000 token），70 % 的轮次有 Goal，有 decode 和完成，每个 admission 都按 `capacity.ledger_gate` 的方式做 gate。
- 只是去掉拒绝时，40 次随机运行中 40 次耗尽了 pool 1（`phase 1 needs 422 of 392 free`、`Goal forward needs 128 of 123 free`）。
- 按现在的顺序，40 次中 0 次耗尽。

**改动 1：在 drain 时释放旧窗口（`WindowPins.trim`）。** 开该参数时，一旦阶段 1 的 drain 得到证实，session 在 `kept_blocks(e)` 第一个下标之前的已钉住 block 就被释放（`WindowPhase1Coordinator._run_split`）。这些就是现在 adoption 释放的 block。
- 为什么释放它们是安全的：之后没有 forward 读取它们。session 的下一次 query 读取 [e − W + 1, e]，阶段 2 以 `kept_blocks(e)` 安装，Goal forward 是冷的。阶段 1 已离开 runner（它的 drain）。之后 `handover` 不丢弃任何东西。
- worker 看不到任何变化。它已提交的窗口 carry 仍指向旧 block，但只有该 session 的下一次阶段 1 会读它，而那时一次 commit 已经替换了它。失败的 admission 与现在一样使 session 失效，session 关闭时丢弃 carry。`WindowPins.close` 释放仍被钉住的部分。`idle_release` 从不选中带有阶段 1 记录的 session（`phase1.pending`）。
- trim 之后，等待中的 admission 在它 session 旧的钉住之外只持有 `stay − trimmed` = 每组的 adopted 窗口 − 旧窗口 block 数。这正是它这一轮被接受后保留的部分，从不超过它的 base：
  - session 后期 1 个 block（4,300-token 的 observation：stay 65，trimmed 64；base 2）；
  - session 早期、窗口还在增长时更多（C = 300，e = 1,400：46，base 47）；
  - e 落在最后一个已钉住的 block 中时，可能为负。

**改动 2：等待中的 admission 持有的部分从它的 base 中扣除（`ledger_ahead_split`）。** 对已 drain 的记录，`WindowPhase1Coordinator.held` 报告这个净数字（`_held_split`）；设置了 `phase1.split` 时，`capacity.reserved` 使用拆分规则：kept = base − held，transient = peak − base。不变式是 free ≥ Σ kept + 最大的 transient + 运行中请求的 headroom。gate 建立它，每一步都保持它：
- **阶段 1**（一次调用）：它需要它的 ring ≤ peak = 它的 kept + 它的 transient。
- **drain 和 trim：** free 减少净数字，Σ kept 也减少同样多；transient 不变。
- **Goal forward**（一次调用，在它的 drain 时释放）：它需要 ⌈G/P⌉ ≤ peak − held。这成立，因为 peak ≥ ring + ⌈G/P⌉，且 held ≤ stay ≤ ring。
- **接受：** admission 离开账本，它的增长变成 headroom。kept = base − net ≥ 阶段 2 峰值 − adopted 窗口。
- **Decode：** 它们取用 headroom 计入的部分。

每个 held 数字都 ≥ 0 时，现在的规则预留的至少与拆分规则一样多，所以是安全的，但更保守。有负数时，它在一个更大的 transient 旁边少预留了那么多 block。拆分规则在两种情况下都是精确的。

**租约。** Gemma 的每个租约，阶段 1 和 Goal forward，仍在一次引擎线程调用内运行并 drain（没有 `--phase1-in-step`）。所以在 gate 时没有未 drain 的租约，各个 transient 仍然从不同时分配。
- `busy()`（只看未 drain 的租约，3.4）让下一个 admission 的阶段 1 可以在已 drain、正在等待的记录旁边运行，与 Qwen 一样。
- 开该参数时，窗口布局的 `run` 检查 `busy()`。
- 窗口布局上如果阶段 1 在步内运行，引擎拒绝该参数（`goal_overlap.attach_engine`）。

**不开 `--admission-overlap` 时**（K = 1，Goal 挂起）：一个 admission 挂起期间，gate 拒绝其他所有加入，它的阶段 1 数字计入运行中请求的 headroom。该轮的滑窗峰值变成 max(ring, net + ⌈G/P⌉)，而不是 ring + ⌈G/P⌉，不超过 gate 检查过的数字。阶段 2 不变。

**未采用：等待期间保持旧窗口钉住**，每个 admission 预留 max(base, stay)。这不需要 trim，但会减少同时进行的 admission。在 Terminal-Bench 的 observation 大小下（2,600 到 4,300 token：stay 65，base 2，transient 226 或更多），16 个 session 时 pool 1 只能容纳约 6 个同时进行的 admission，而不是 16 个。有 trim 时，预留与现在相同。在 CPU 的几何上，同时等待的轮次与不开该参数时同时在 Goal 处等待的一样多；在模拟中，最多有 16 个 admission 同时在它们的阶段 1 之后等待，40 次运行中都没有出现不足。

### 9.4 窗口布局的其他部分

- **Goal 瞬态（G = 2,048）。** 不变。单独的 Goal forward 在每组取它的 ⌈G/P⌉ 个滑窗 block，在它的 drain 时释放，而它这一轮只持有它的净窗口。更长的 Goal 在分配任何东西之前就以 `PHASE1_TOO_LONG` 失败，与现在相同。Goal-forward 租约按阶段 1 租约中的 Goal 来检查（`contract.parse_goal_lease`：⌈G/P⌉ 个 block 的 FA 行和滑窗行，token 数 ≤ 布局的 `goal_tokens`，zero 列表 = pool-0 的 id）。
- **分块的阶段 1。** 它仍是一个租约，它的分块在一次 RPC 中运行，drain 一次。`stay` 是它最后一次 forward 结束时所在的窗口，trim 和 Goal forward 在 drain 之后。chunk ring 中没有任何东西在 drain 之后还存在。
- **`--release-idle-pages`。** 它只作用于全局 Page，窗口从不离开 pool 1。重新载入与现在一样发生在阶段 1 之前。等待中的 session 从不被选为释放对象（`phase1.pending`），它重新载入的 Page 留在它的 FA 预留中（多预留了，与 Qwen 一样）。
- **`--decode-fastpath3`。** trim 像 `handover` 一样经 pool 1 释放，所以窗口池的 journal 记录它（`window_rows.Journal`）。在 Goal forward 的 RPC 之前，proxy 发送排队的分配更新，与阶段 1 之前一样（`decode_fastpath2`）。probe-only pass 像阶段 1 一样，把运行中的请求再去掉并重新加入一次。
- **Worker。** 无论有没有 `admission_overlap`，`window_worker` 都接受 admission 记录 `{goal_overlap: true}`，`gdn_handoff.attach_worker` 设置 `owner.goal_overlap`。worker 路径的其余部分（`model_queries.goal_forward`、`NativeQueryForward.goal_forward`、`drain_goal`）就是与 profile 无关的那条路径。
- **Launcher 和配置。** `serve.py` 在 Gemma profile 上接受带 `goal_query` 和 `--goal-rewriter-model` 的 `--goal-overlap`，只固定 `admission.goal_overlap`。`remote_ra_http.config` 接受它与 `--admission-overlap` 并用。`--phase1-in-step` 和 `--gdn-device-handoff` 仍被拒绝。
- **哈希。** 与 Qwen 一样（3.11），Goal payload 的摘要从 `WindowPhase1Coordinator.run` 移到 `_goal_lease`，其他都不变。

### 9.5 收益

与 Qwen 相同（第 4 节）：每个 rewriter 轮次约 min(T_goal, T_phase1)。本设计没有测量 Gemma 的阶段 1。3.6 的第二项好处不存在，因为 Gemma 拒绝 `--phase1-in-step`。代价是每个有 Goal forward 的轮次多一次 probe-only pass，与 Qwen 相同。

### 9.6 代码和测试

- 引擎：
  - `window_phase1.py`：`run`（开该参数时 `busy()`）、`_run_split`、`held` / `_held_split`、`_goal_lease`，以及 `release`（先释放 Goal-forward 租约）；
  - `remote_ra_native_growing/window_pins.py`：`trim`；
  - `window_capacity.ledger_ahead_split` 和 `capacity.reserved`；
  - `goal_overlap.attach_engine`。
- 共用：`contract.parse_goal_lease`。
- Worker：`window_worker.py`。
- Launcher 和配置：`serve.py prepare_gemma4`、`remote_ra_http/config.py`。
- 不开该参数时：`split` 为 False，所以 `run` 和 `held` 走现在的代码，`reserved` 使用 `ledger_ahead`，不 trim 任何东西。

`test_gemma_goal_overlap.py`，19 个测试：
- launcher、配置和 worker 的记录与拒绝；引擎的接入；
- 窗口布局的 Goal-forward 租约；
- `trim`；`ledger_ahead_split`；模拟（9.3）；
- 在 test_gemma_release 的引擎和 worker 假实现上，用 test_gemma_overlap 的 rewriter 替身端到端运行。每一轮的 token、回复、选页、关注的 Page 和 Goal 向量都等于不开该参数、K = 1 的服务的结果：
  - 开 `--admission-overlap 8`；
  - K = 1 且 Goal 挂起；
  - 有空闲释放并开三条解码快速路径；
  - 阶段 1 按 256 分块；
- 扣住的 Goal：每个被接纳的轮次都运行了它的阶段 1，并只带着它的净窗口在 join 处等待。pool 1 恰好持有这些 block，账本预留 Σ (base − held) 加上最大的 transient，同时等待的轮次与不开该参数时一样多；
- 在 join 处取消：session 失效，它的阶段 1 被清理，被 trim 的窗口在关闭时释放；
- 哈希：只有 Goal payload 的摘要移动。

`test_goal_overlap.py` 现在检查 Gemma 上对不带 rewriter 的策略的拒绝，而不再检查对该 profile 的拒绝。

### 9.7 风险和 GPU 检查

- **trim** 是新的：它比现在更早释放 pool-1 block，在阶段 1 的 drain 之后而不是在 adoption 时。CPU 假实现检查每一次滑窗读取以及两个池的基线；在 GPU 上，开该参数的 Gemma 服务在作业 3538523 中运行过（第 8 节）。
- **拆分账本**是新的算法。证明（9.3）和模拟覆盖了它，但 gate 的数字假设的是 gate 预测的轮次几何（从 transcript 得出的 C、e），与现在相同。
- **join 处的失败**发生在阶段 1 和 trim 之后。session 失效，它的窗口在关闭时释放，与阶段 1 之后的任何失败一样。
- **GPU 检查。** 在 Gemma 的启动配置（`goal_query`、`--admission-overlap 16`、`--release-idle-pages`、Qwen3-0.6B rewriter）上，开与不开该参数，做第 7 节的检查：
  - 逐轮相等的 Goal 向量字节（`goal_forward` 帧对照不开该参数时的阶段 1 帧）和相等的选页；
  - 没有 `insufficient native blocks`，没有抢占，两个池都回到基线；
  - `CAPACITY_WAIT` 的次数和同时进行的 admission 数，与不开该参数的服务对比。
