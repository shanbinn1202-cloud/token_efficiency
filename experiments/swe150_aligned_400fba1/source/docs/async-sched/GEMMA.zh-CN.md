# Gemma 4 31B IT profile 上的 --async-scheduling（方案 C，窗口布局）

[English](GEMMA.md) | 简体中文

日期：2026-11-03。分支 `async-gemma`，基于 `integrate-1103`（`ea18256`：async-c、async 与 `--goal-overlap`
的组合、速度默认值、带 `--goal-overlap` 的 Gemma profile）。路径相对于 `source-overlay/`，
以顶层文件开头的除外。行号以 `ea18256` 为准，本页注明已修改的文件除外。
它的 GPU 运行及其判定见 [docs/status.zh-CN.md](../status.zh-CN.md) 和[历史](../history.zh-CN.md#历史)；仍然成立的限制
（部分通过）见 [docs/known-limits.zh-CN.md](../known-limits.zh-CN.md) 5.3；GPU 检查测得的结果见第 8 节和第 9 节。
请结合 [DESIGN.zh-CN.md](DESIGN.zh-CN.md)（隐患 D1-D5、H1-H7 和方案 C）和
[IMPLEMENTATION.zh-CN.md](IMPLEMENTATION.zh-CN.md)（async-c 为 Qwen 实现了什么；它的第 11 节，即 async-maxtok-1112
`13ee618`，是第 1 节第 4 项所说的共用 resolve，窗口布局也走它）阅读。

## 0. 概要

1. **Gemma 为什么曾被拒绝。** async-c 为 Qwen 实现了方案 C，并在三处拒绝 Gemma 4 profile
   （`serve.py` `prepare`、`remote_ra_live_v2/async_sched.check_config`，以及隐含地在窗口 worker 中：它从不
   读取该记录），理由是“窗口布局的 owner、行和快速路径 3 的 journal 没有适配”
   （IMPLEMENTATION.md 第 1 节）。只有一处代码确实只适用于 Qwen：作废写入器要求 Gated-DeltaNet
   状态下标（`async_sched.void_rows`：`require(states > 0 ...)`），而窗口布局没有它，所以 Gemma 的每次
   停止都会直接失败；在各组的 slot mapping 中，它只列出了 full-attention 组。
2. **Gemma 特有的部分**（第 3 节）：没有递归状态，所以唯一的设备隐患是 C' 处的 KV（D3/D5），分布在六个
   KV 组中（pool 0 中的全局组，pool 1 中的五个滑窗组）；一个已停止请求的推测分配
   会移动它的*窗口*（提前一步释放一个最前面的块，每个滑窗组一个新块），这些块
   必须留在提交所 pin 住的窗口之外；engine 增量地跟踪每个窗口（快速路径 2 和 3）；
   `--goal-overlap` 的挂起在多个 step 之间持有一个净窗口；4,096 或 8,192 token 的分块前向。
3. **结论：可以安全开启，但在各 gate 之外需要两处 Gemma 特有的 runtime 改动。** 每一条 Gemma 特有的路径
   要么是 pass 边界（与 Qwen 一样先排空），要么可以证明不受多出的那一行影响（第 3 节），前提是：
   (a) 窗口布局上的作废在全局组和每个滑窗组中把该 token 的 slot 写成 -1，并且
   在那里拒绝 GDN 状态下标；(b) 被作废的请求持有的 worker claim 转给 vLLM 在同一次调度中
   把该块交给的那个请求（推测步释放的窗口最前面的块会进入
   vLLM 空闲队列的队首；3.1 节）。没有 (b) 时，这样的 step 会直接失败（`allocation owned by another
   generation`）：每 16 次停止中有一次释放最前面的块；16 个 session 同时解码时，这样的停止中约有三分之一会遇到
   一个排在它之后、并在该 step 中取得新滑窗块的请求（每个请求 16 次中有一次，排在它之后的约有 7.5 个），
   有 prefill 时更多。
4. **保留的拒绝**（第 5 节）：不带 `--decode-fastpath2`；带 `--gdn-device-handoff` 或 `--phase1-in-step`
   （两者本来都不适用于 Gemma：被 profile 和 async 各拒绝一次）；带 `--thinking-history strip`
   （gemma-thinking：`--enable-thinking` 的保留模式可以组合）。不是速度默认值（integrate-1103）。
5. **一致性合同：沿用 Qwen 的**（第 4 节）：b = 1 时，异步服务给出同步服务的 token
   和证据（按宽松合同）；在 Gemma 上，窗口 carry 的 block id、incarnation 和摘要也属于
   被改名的部分。b > 1 时：结构检查加一项测量，与 perf-async-g1 的判定规则相同。
6. **一个共享的 bug，已为两个 profile 修复**（第 6 节；IMPLEMENTATION.md 第 9 节）：一个 pass 边界
   排空了在途 step 却没有发出 worker RPC（CAPACITY_WAIT 拒绝、utility 请求、唤醒）之后，下一次
   resolve 会再次判定那个 step 的停止，把一个 token 计了两次，并因丢失一个 step 而停止 engine。
   Gemma 压力测试发现了它；现在 resolve 跳过 engine 已经应用的 token（`EngineLink.applied`）。

## 1. 异步调度改变了 vLLM 0.24 循环中的什么

`max_concurrent_batches == 2` 时的 `EngineCore.step_with_batch_queue`（`vllm/v1/engine/core.py:536-652`）：

1. 前向 N 在 GPU 上运行时，`schedule(N+1)` 就开始运行。对一个在 step N 中采样的请求，scheduler 还
   不知道它的 token：它记一个**输出占位符**（`num_output_placeholders`），并且已经在
   `_update_after_schedule(N)` 时推进了 `num_computed_tokens`。所以 step N+1 会按 C' + 1 个 token 为一个请求分配，
   而这个请求可能已在 N 停止。
2. `execute_model(N+1)`（UniProc：内联、非阻塞）准备输入、metadata 和 slot mapping；解码请求的输入 id
   来自设备（`prev_sampled_token_ids`）。我们的 `before_forward` hook 在这里运行
   （`gpu_model_runner.py:4605-4610`），在 vLLM 的 builder 填好该 step 的 metadata（以及 FULL graph 下的
   持久缓冲区）之后。
3. 这之后才取出 step N 的输出并运行 `update_from_output(N)`（判定停止、完成、释放）。
4. 停止确定时，vLLM 自己会跳过多出的那一步：`max_tokens`（`vllm/v1/core/sched/scheduler.py:489-503`）。
   EOS、停止 id、XML action 标记或上下文上限都只有在 N 采样之后才知道。
   `schedule(N+1)` 漏掉的请求（它的第 `max_tokens` 个 token、达到 `max_model_len` 的 token，`scheduler.py:527-534`
   和 `562-578`；或者 token 预算、抢占）在 `_update_states(N+1)` 中离开 runner 的批次，
   `InputBatch.remove_request` 把它从 `prev_req_id_to_index` 中删除：N+1 中没有它在 N 的 token
   （async-maxtok-1112，IMPLEMENTATION.zh-CN.md 第 11 节）。
5. 已完成请求在 N+1 中的 token 由 vLLM 丢弃（`scheduler.py:1723-1732`）。
6. **滑窗。** `KVCacheManager.allocate_slots` 在*调度时*释放滑窗组不再读取的块，
   依据是 `num_computed_tokens`（`vllm/v1/core/kv_cache_manager.py:495-499`、
   `single_type_kv_cache_manager.py:479-511`；`SlidingWindowManager.get_num_skipped_tokens(n) = n - W + 1`，第 770 行）。
   在异步下，这个计数已经包含在途的 step：窗口的前端比 GPU 早一步移动，
   每个请求都是如此（stock vLLM 的行为；stream 保证在途 step 的读取先于下一步的写入）。

方案 C（DESIGN.md 4.3 节，由 async-c 实现）：我们的 `before_forward(N+1)` 先 resolve N（收尾 N 的 forward 之后的工作，
只读取一次 N 中被 N+1 调度的请求的采样 id，用 vLLM 的 `check_stop`（作用于一个影子请求）加 `MarkerStop` 判定它们的停止，在设备上作废
N+1 中已停止的行，去掉已停止请求的条目后发送 N+1 暂存的分配更新，让继续的请求
进入），并且每个 pass 边界都先排空在途批次。

## 2. 我们那些假定同步 step 的 hook（两个 profile）

| Hook | 假定 | async-c（Qwen） | 窗口布局（Gemma） |
| --- | --- | --- | --- |
| Gate：`bootstrap_contract`、`policy_runtime/bootstrap`、`native_worker_registration`、`startup_runtime`、`positions.runtime_mode` | `max_concurrent_batches == 1`，没有 batch queue，`use_async_scheduling` 为 false | 当且仅当已固定时为异步（`async_sched.allowed`） | 同一份代码；`positions.runtime_mode` 已处理窗口布局（没有 Mamba align 模式） |
| 启动器、配置加载器（`serve.py`、`remote_ra_http/config.py`） | `--no-async-scheduling` | 固定的配置项 + `effective_server_config.async_scheduling` | 原先被拒绝（`serve.py:497`、`check_config:79`）；现在是同样的固定（`prepare_gemma4`） |
| Engine 分配（`engine_boundary.dispatch_mixed_request`、`allocation_guard`、`chunk_protocol`） | 解码的新 token 在 host 上 | RUNNING 的解码上有一个占位符 | 同一条代码路径（窗口布局的分配也走这里） |
| 暂存的更新（`EngineRpcBoundary.flush_updates`，快速路径 2） | 在 `schedule` 返回时发送 | 暂存到 resolve；已停止请求的条目被丢弃，其 `native_allocation` / `native_tables` 被恢复（H2） | 相同；engine 为该请求维护的窗口跟踪器（`window_rows.grow`）已前进，并在它完成时丢弃（3.3 节） |
| Worker step（`WorkerBridge.before_forward` / `after_forward`、`finish_step`） | step 在它自己的 `execute_model` 内 join | 发出后在下一次 resolve 时收尾 | `WindowLifecycle` 继承 `issue_execute` / `leave_issued`；window owner 继承 `scheduled_rows`，它跳过作废的行 |
| Generation guard（`generation_guard.guarded_execute`） | step 之前做一次设备排空 | 先做 connector 检查，再 `async_sched.guarded` | 同一个对象（`window_worker` 安装同一个 guard） |
| 作废（`async_sched.void_rows`） | 不适用 | GDN 状态下标 `NULL_BLOCK_ID` + FA slot -1 | **已修改**：没有 GDN；在全局组和每个滑窗组中写 slot -1（3.2 节） |
| 完成（`allocation_owner._complete` 的 cursor 检查、`WorkerBridge.terminal`） | worker 的 cursor 为 C' | 未作废的停止直接失败 | 相同；`WindowWorkerBackend.drain` 根据 worker 在 step N 的行提交 `final_window` |
| Pass 边界（admission、Goal 恢复、客户端请求、关闭、关停） | 没有在途 step | 每个之前 `_remote_ra_async_drain`（`core.py:664-690`、`1430`、`1459`、`1479`、`1515`）；worker 在任何非稳态 RPC 之前收尾 | 相同；Gemma 的阶段 1、Goal forward、trim、gate 和 ledger、空闲释放和窗口关闭都在这里运行 |
| 旧代码（`remote_ra_growing_worker/plan.py`、`remote_ra_qwen_semantics/hooks.py`）、`private_authority.Topology` | 只支持同步 | 不变（不在服务路径上） | 不变 |
| Step profiler（`REMOTE_RA_STEP_PROFILE=1`） | 一次一个 step | 未适配，未与 async 一起测试 | 相同 |

## 3. Gemma 特有的部分

几何（[docs/reference.zh-CN.md 4.14 节](../reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）：10 个全局层（一个 KV 组，pool 0，P token 的 Page，P = 16），50 个滑窗层（W = 1,024）
分为五个 KV 组，共用 pool 1。pool 1 在启动时预留：每个滑窗组 S 个窗口槽位，每个槽位
`ceil(W / P) + 1` = 65 个块；一次前向的瞬态块；对 `goal_query` 另有 Goal 的瞬态块
（`kv_layout.WindowLayout.window_blocks` / `transient_blocks`、`window_pool_blocks`）。两轮之间，session 为每个滑窗组 pin 住
`kept_blocks(C')` = `[(C' - W + 1) // P, ceil(C' / P))`（`sliding_window.py:56`、
`remote_ra_native_growing/window_pins.py:118`）。

### 3.1 已停止请求 R 的推测步（在 N 停止，已计算终点 C'）

`schedule(N+1)` 为 R 分配时，就像为位置 C' 的一次解码分配一样（num_computed_tokens = C'）。R 是 N+1 仍会调度的
停止（EOS、停止 id、输出停止）；`max_tokens` 或 `max_model_len` 的停止在那里不分配任何东西（第 1 节第 4 项）：

| 事件 | 何时 | 对窗口的影响 | 为什么无害 |
| --- | --- | --- | --- |
| 前端释放 | (C' - W + 1) % P == 0（W = 1,024、P = 16 时：(C' + 1) % 16 == 0），C' >= W | 每个滑窗组提前一步释放块 `(C' - W + 1) // P - 1` | 它位于 `kept_blocks(C').first` 之前：提交从不 pin 住它；同步服务在 R 完成时释放它 |
| 新块 | C' % P == 0 | 为位置 C' 分配一个新的全局块（pool 0），每个滑窗组一个新块（pool 1） | 下标 `C' // P = ceil(C' / P)` 在 `kept_blocks(C').end` 之后：从不被 pin 住，随 R 释放；作废不会向它们写入任何内容 |
| 窗口跟踪器（快速路径 2/3，engine） | 每一步 | R 的 `record['incremental']['windows']` 前进 | 暂存的条目被丢弃；`complete_and_drain` 在任何读取之前弹出 `incremental`（`engine_boundary.py:442`） |
| 池 journal（快速路径 3） | 每次分配 / 释放 | R 的推测事件被记入 journal | journal 只会让其他请求的窗口检查*更*彻底（`window_rows._unchanged`：被点名的块会被完整检查） |
| Worker claim | 一次前端释放，且同一次 `schedule(N+1)` 中有另一个请求在 R 之后分配 | vLLM 0.24 把释放的未哈希块放在空闲队列的**队首**（`block_pool.free_blocks`：`prepend_n`），所以那个请求立刻取得 R 释放的块，而 R（已作废，这一步不被观测）在本轮迭代稍后完成之前仍持有 worker 对它的 claim | **已修改**：当 claim 持有者在这一步的行被作废时，claim 转移（`RunnerAllocationOwner._voided_holder`）；此前这一步会失败（`allocation owned by another generation`）。R 不会再碰这个块（3.2 节），R 的完成只读取 vLLM 仍为它保留的块 |

- **提交与同步服务相同。** engine 根据 R 的行 pin 住 `kept_blocks(C')`，时机是
  `update_from_output(N)` 期间（`WindowPins.pin_after`）：`[first, end)` 中的每个下标仍是 R 的 live 块，只有一个
  引用，正是同步服务 pin 住的那些块。worker 的 carry（`WindowAllocationOwner.final_window`）
  由 worker 的 native 行构建，这些行是 step N 的行，因为 R 在 N+1 的更新从未发送
  （`EngineLink.flush(voided)`）；下标相同，块也相同。
- **容量。** R 的推测步是位置 C' 处的一次解码，也就是同步服务在回复
  长一个 token 时会运行的那一步：在 pool 1 中它每组最多占 `ceil(W / P) + 1` 个块（R 自己的窗口槽位；vLLM 先释放
  前端再分配），在 pool 0 中位置 C' < prompt + max_tokens（在 `max_tokens` 停止时 vLLM 跳过这一步）
  且 < `max_model_len`。两者都在 gate 和滑窗 ledger 已经把 R 作为运行中请求计入的范围之内
  （`capacity.headroom_by_pool`、`window_capacity`），并且都在同一次 engine 迭代中释放
  （`update_from_output(N)`），早于下一个 pass 边界。每个 gate 决策都在排空之后做出，所以它看到的空闲计数
  与同步服务相同（块的身份除外）。
- **块身份（H1）。** 提前释放和新块改变了 pool 0 和 pool 1 空闲队列的顺序，所以
  之后的 block id 和 incarnation 计数与同步服务不同，b = 1 时也是如此（上面两种情况各占
  每 16 次停止中的一次）。宽松合同对它们改名（第 4 节）。
- **Prefix cache。** 每次生成分配都经过 `dispatch_mixed_request`，它设置 `skip_cache=True`
  （`engine_boundary.py:213`），所以 `allocate_slots` 从不缓存块（`kv_cache_manager.py:561`）：一个最后位置
  C' 被算作已计算却被作废的块，永远不会作为 prefix 命中提供（perf-async-g1 关于
  (C' + 1) % 16 == 0 停止的未决风险在两个 profile 上都不成立）。

### 3.2 窗口布局上的作废

- **没有递归状态。** 多出的那一行在设备上只写 C' 处的 KV：全局组的 tail Page 和每个
  滑窗组的块 `C' // P`（最后一个保留块，或 C' % P == 0 时的新块）。不作废时这些
  写入无害（DESIGN.md D3、D5：下一轮的阶段 1 从 C = C' 开始，并在任何 query 读取 C' 之前
  写入它）。作废之后 KV 字节与同步服务相同，这正是一致性合同所要求的。
- **它写什么。** 在窗口布局上，`void_rows`（已修改）拒绝 GDN 状态下标，并在作废的 token 处写 slot -1：
  每层 metadata 的 `slot_mapping`、forward context 中每个逐层 mapping，以及
  六个 KV 组各自的 mapping（`owner.attention_groups` + `owner.async_sliding`，即 window owner 的滑窗
  组，由 `attach_worker` 设置）；每个组的 mapping 都必须存在。vLLM 的 mapping 是 block
  table 的持久 `slot_mapping.gpu` 缓冲区的视图（`gpu_model_runner._get_slot_mappings`、`4085-4157`），所以这次写入
  能到达 FULL graph 的 replay；TRITON_ATTN 的 metadata 和 KV-cache 更新（窗口布局上是 splitting op，
  `cuda_graphs.WINDOW_SPLITTING_OPS`）读取的是同样的张量。
- **TRITON_ATTN 上的 slot -1。** 在 FULL graph 下，vLLM 对每种后端都用 -1 填充未使用的 token
  （`gpu_model_runner.py:4131`："Needed for reshape_and_cache in full cuda graph mode"），所以 Triton 的 KV 更新 kernel
  与 FA3 的一样跳过 -1。这不在 overlay 中：由 GPU 检查确认（第 8 节）。
- 被作废的行仍然会读取（在 R 的块上做 attention，包括未写入的 slot C'）；它的输出从不被使用：
  `scheduled_rows` 跳过它（没有 lease，没有 Page 向量，也没有它自己的有限性判定），vLLM 丢弃它的 token。

### 3.3 Engine 和 worker 的记录（H2、H3、H4）

- Engine：与 Qwen 一样，R 暂存的条目被丢弃，其 `native_allocation` / `native_tables` 被恢复。R 的
  `incremental` 记录中的窗口跟踪器和全局行的 `fa`/`ids` 在 `window_rows.grow` 中前进，且从不
  恢复：在 `complete_and_drain` 弹出它们之前没有任何东西读取它们。`WindowPins` 的状态只在提交时和
  pass 边界处改变。
- Worker：R 的行、claim、观测（快速路径 1-3：`_WindowObserved`、稳态的 `native3` 行）、lease 和
  `window_ids` 都保持 step N 时的状态。继续的请求与同步服务上完全一样地前进：它们的增量
  相对于 engine 的跟踪器计算，并应用到 worker 最后接受的行上，两者都是每个已调度的 step
  前进一步。
- `WindowLifecycle`（一个 `QwenAlignLifecycle`）原样沿用 `issue_execute` / `leave_issued`；`before_forward3`
  （快速路径 3）只看到已进入的请求。

### 3.4 阶段 1 挂起（`--goal-overlap`，effa85e）

一个 `goal_query` admission 在它自己的 pass 中运行阶段 1，在阶段 1 的排空时 trim 掉 session 的旧窗口
（`WindowPins.trim`），在它的 Goal 线程上挂起，同时持有它在 pool 1 中的净窗口和新块，然后在
Goal 恢复的 pass 中继续（Goal join、作为 Goal 瞬态块上独立冷 lease 的 Goal forward、adoption、
预留）。

- 两个 pass 都是 pass 边界：engine 在登记请求的 add 之前和 Goal 恢复之前都会排空在途批次
  （`core.py:1430`、`1515`），worker 在阶段 1 和 Goal-forward RPC 之前收尾已发出的 step
  （`worker_dispatch`）。阶段 1、trim 和 Goal forward 从不遇到在途 step。
- 它挂起期间，其他 session 的解码 step 异步运行。它们的推测分配留在
  它们自己的窗口槽位和回复预留之内（3.1 节），所以滑窗 ledger 的不变式（空闲 >= 保留块之和 + 最大的
  瞬态块 + 运行中请求的余量；goal-overlap 的 DESIGN.md 9.3 节）在 pass 之间成立，与同步时
  一样。挂起的 admission 所持有的内容（`WindowPhase1Coordinator.held`）不会被解码 step 改动。
- `--phase1-in-step` 是唯一会把 admission 工作放进 step 内的选项，它在 Gemma 上不可用，并且与
  `--async-scheduling` 同用时仍被拒绝。

### 3.5 分块前向（`--max-num-batched-tokens` 8,192 或 4,096）

- 分块的 **阶段 1**（`window_phase1`、`phase1_chunks`，每块最多 N - W - P 个 token）在其 admission 的
  pass 内运行：已排空，一个 lease，没有在途 step。
- **阶段 2 prefill 分块**（冷轮次、重建）：chunk k 在途时 vLLM 就调度 chunk k+1，中间
  不做采样。滑窗前端在调度时移动（第 1 节第 6 项）；stream 保证 chunk k 的读取
  先于 chunk k+1 的写入。worker 只为在已发出的 step 中 prompt 已完整、且被下一个 step 调度的请求读取 token
  （`RequestState.prompt_complete`，vLLM 的 `prev_req_id_to_index` 不包含未完成的 prefill；async-maxtok-1112）。engine 的
  chunk k 的 `scheduled_cursor` 由 step k 的 resolve 推进，而这次 resolve 在 `schedule(k+1)` 检查 chunk
  k+1 的区间之前运行。
- **b > 1 时的 token 预算。** 被作废的行占用 step N+1 的 `max_num_batched_tokens` 中的一个 token，所以并发的
  prefill 分块可能比同步服务上早一个 token 结束：batch 组成不同，这就是
  b > 1 只做结构检查的原因（第 4 节）。它从不超出前向的瞬态块：被作废的行的块就是 R 自己的槽位。

### 3.6 Gemma 上的解码快速路径

`--async-scheduling` 需要 `--decode-fastpath2`（暂存的 `allocation_batch`）；使用速度默认值时三条都
开启。快速路径 1 的增量窗口观测、快速路径 2 的增量和紧凑行，以及快速路径 3 的稳态
观测、journal 和 `before_forward3` 都只在已进入（未作废）的请求上运行，这些请求的观测与同步服务上
完全相同（3.3 节）。

### 3.7 哪些不需要修改，以及原因

`window_owner`、`window_lifecycle`、`window_rows`、`window_pins`、`window_phase1`、`window_capacity`、`capacity`、
`idle_release`、`goal_overlap`、`deferred_completion`：它们的每步工作在已进入的行上运行，其余每次调用
都在 pass 边界处，或在 `update_from_output(N)` 内的完成时运行，在 N+1 的 resolve 之后（3.1、3.3 节）。
`--deferred-completion-io`：窗口布局的完成工作（carry、vLLM 释放之前的 pin、释放之后的
检查）与以前一样留在 engine 线程上、`update_from_output(N)` 之内；在那里，被 pin 住的块的引用计数
与同步服务相同（R 的推测块不在 `kept_blocks(C')` 中）。

## 4. 一致性合同（沿用 Qwen 的）

perf-async-g1（GPU harness；它和它的 README 在维护者的工作区中，不在
本仓库里，所以在此重述该规则）的判定规则，不变：

- **b = 1，完全一致。** 每个请求都等于同步服务的：prompt id、finish reason、completion 数、content、
  生成的 id、worker frame 的操作序列，以及 native 文件和 runtime
  行中按宽松合同比较的每个内容字段：token、位置、已封存 Page、选页、Page 向量、Goal 向量；block id、
  incarnation 和身份标识在一致改名的意义下相同；空闲计数和用时不比较（H6 没有实现）。
  **Gemma 另加：** 提交的窗口属于内容（相同的组，相同的块下标
  `kept_blocks(C')`，每个槽位中相同的位置）；它的 block id 和 incarnation 被改名，基于它们的摘要
  与其他基于改名值的摘要一样不比较：窗口 carry 的摘要（`window_lifecycle.window_digest`；
  `drain.carry` 和 frame 的 `carry_restored` 的 `digest`），它作为 session `boundary` 的回显，以及阶段 1 的
  窗口 `state_digest`（`WindowWorkerBackend.capture_phase1_state`）。释放的槽位按 (block,
  incarnation) 对改名：一个离开某请求窗口的滑窗块，可能以新的 incarnation 回到同一个请求，
  取决于空闲队列的顺序（H1），所以每组不同块的数量不是不变量，
  不同槽位的数量才是。CPU 测试还要求不在已完成请求的位置 >= C' 处写入 KV
  （[C, C') 只计算一次）。
- **b > 1，结构检查加一项测量。** batch 组成不同（下一步中有一个被作废的行，以及
  3.5 节的 token 预算），所以在 GPU 上不要求 token 一致。要求：每个请求都得到回答；`length` 恰好在
  上限处，`stop` 在上限之内且停在停止 id 或标记上；行和 carry 中 C' = N + g - 1；提交的窗口在
  每个滑窗组中都是 `kept_blocks(C')`，engine 的 pin 等于 worker 的 carry；frame 从 `prepare` 到
  `complete`；每个继续的轮次都在其 prompt 后接上回复的 id。测量：一致的比例，以 stock
  服务对的 async 与 sync 差异为参照。（在没有 batch 差异的 CPU fake 上，b > 1 也检查为相等。）

## 5. 保留的拒绝

| 组合 | 位置 | 消息 |
| --- | --- | --- |
| `--async-scheduling` 不带 `--decode-fastpath2` | `serve.prepare_gemma4`、`async_sched.check_config`、`window_worker`、`async_sched.attach_worker` | `serve.prepare_gemma4`：`--async-scheduling requires --decode-fastpath2`；`check_config` 和 `attach_worker`：`--async-scheduling needs --decode-fastpath2`；`window_worker`：`async_scheduling needs decode_fastpath2`（使用速度默认值时它已开启；用 `--no-speed-defaults` 时与 Qwen 一样给出 `--decode-fastpath --decode-fastpath2`） |
| 带 `--gdn-device-handoff` | `serve.prepare`（在 profile 分支之前）、`check_config`、`attach_worker`；Gemma profile 本身也拒绝它 | `--async-scheduling refuses --gdn-device-handoff and --phase1-in-step (v1)` |
| 带 `--phase1-in-step` | 同上 | 同上 |
| 带 `--thinking-history strip`（gemma-thinking；单独的 `--enable-thinking` 可以组合） | `serve.prepare_gemma4`、`thinking_cut.check_config` | `--thinking-history strip on the Gemma 4 profile refuses --async-scheduling (v1)` |
| 作为速度默认值 | `serve.SPEED_SWITCHES` | 在 GPU 检查通过之前，在两个 profile 上都不是默认值 |

## 6. 实现（async-gemma）

- `remote_ra_live_v2/async_sched.py`：`check_config` 不再拒绝 `model_profile`；`attach_worker` 记录
  window owner 的滑窗组（`owner.async_sliding`，Qwen 的布局上为 None），此前先检查 owner、lifecycle
  和 backend 在布局上一致；`void_rows(..., sliding_groups)` 如 3.2 节所述；`resolve` 传入它们。
- `remote_ra_live_v2/window_worker.py`：读取并检查 worker 记录（需要快速路径 2），在快速路径之后、
  generation guard 之前接入 `async_sched.attach_worker`（与 `attach_loaded_worker` 的做法相同），
  在证据中回显 `async_scheduling`（compose_engine 会比较它）。
- `serve.py`：`prepare` 把 `async_scheduling` 传给 `prepare_gemma4`，后者与 Qwen 一样固定它（配置项、
  `effective_server_config.async_scheduling`、用 `--async-scheduling` 代替 `--no-async-scheduling`；没有别的）；
  帮助文本和 docstring。`remote_ra_http/config.py`：只改 docstring。
- `remote_ra_live_v2/allocation_owner.py`：`_claim_full` 允许 claim 从一个这一步的行已被作废的请求转出
  （`_voided_holder`，读取 resolve 的 `runner._remote_ra_live_v2_voided`），并像以前一样拒绝其他所有外来
  claim（3.1 节）。共享：在 Qwen 上，已停止请求的推测步释放的 Mamba block（align
  模式）也适用同一规则，但需要两个请求在同一步中都跨越 Mamba block（很少见）。
- **一个共享的 bug，由 Gemma 压力测试发现，已为两个 profile 修复**（`async_sched.EngineLink.applied`、
  `resolve`）：一个没有发出 worker RPC 的 pass 边界之后，N+1 的 resolve 会再次判定 N 的停止
  （第 0 节第 6 项）；现在它只判定 engine 尚未应用的 token。机制和测试见
  IMPLEMENTATION.md 第 9 节（“Drains without a worker RPC”）。
- release pin 重新生成（333 个源文件，242 个 payload）。

## 7. CPU 测试（`test_async_gemma.py`，`4a498dc` 上 23 个测试；main 上 24 个）

main 上的第 24 个测试（`tests/test_async_gemma.py`）随 async-fix2（`4ed1013`）加入：
`test_without_the_unbatched_report_a_phase1_pass_loses_the_token_ids`。

- `PinTests`、`LauncherTests`：Gemma 的配置检查；在 Gemma release 上带与不带该开关时的 `prepare`，覆盖
  mswea-xml、chat-text 和 kv-session，以及 `--goal-overlap`（和 `--admission-overlap`）、CUDA graph、快速路径 3、
  `--deferred-completion-io`、`--release-idle-pages` 和 4,096 token 的前向：一个配置项，一个 effective 配置项，
  用 `--async-scheduling` 代替 `--no-async-scheduling`，而 `async_scheduling=False` 就是原先的启动方式；
  该配置上的加载器、前端的已解析参数检查、worker 参数和启动 gate；各种拒绝。
- `WindowWorkerAttachTests`：窗口 worker 的记录检查，`attach_worker` 在快速路径 2 和 3 之后、
  guard 之前，回显，各种拒绝；在真实的 window owner、lifecycle 和 backend 上的 `attach_worker`（滑窗组
  被记录；与 window owner 搭配的 Qwen 形状的 lifecycle 被拒绝）。
- `WindowVoidTests`：窗口 metadata 上的作废（eager 和持久缓冲区）：六个组，每个 mapping 和
  forward context 中的副本，别的都不写；async-c 的作废拒绝该布局；各种拒绝。
- `AsyncWindowEndToEndTests`（mswea-xml，test_gemma_release 的 scheduler，按 vLLM 的异步顺序，对照
  使用相同快速路径的同步服务）：一个 session 跑六轮（快速路径 1-2 和 1-3；每次停止都作废；
  两种推测窗口步都发生了，并在行上检查过），三个 session（`full`、`random_fill`、
  `goal_query`），在选择性池上带空闲释放的四个 session（CAPACITY_WAIT、驱逐和重新 admission），
  以及确定性地复现的 claim 转移：一次前端释放的停止，同时另一个 session 的首批 prefill 分块在同一次调度中取得
  被释放的块（`reused`），结果与同步服务相同，没有转移时被拒绝；
  变异：缺少作废、作废漏掉一个滑窗组、漏判的停止、错判的停止、去掉 applied 检查。
- `AppliedTests`：`EngineLink.applied` 对各种计数的处理。
- `AsyncGoalOverlapTests`：`--goal-overlap` 配 `--admission-overlap` 4（快速路径 1-3）和 1：轮次、Goal 向量、
  frame 操作和 native 文件都相同；Goal 在其他 session 解码时挂起，Goal 恢复时遇到在途批次。
- `AsyncKvSessionTests`：1 个、4 个（空闲释放）和 3 个（`goal_query`）session 的 MemAgent lane。
- `AsyncChatTextTests`：带 Terminus 客户端的 chat-text API 服务，其 engine 循环是 test_ms2_e2e 的 stub，配上
  `core.py` 自己的输入循环（其中的排空就是随附发布的那些），按 vLLM 的异步顺序：一个客户端（以及一轮在其上限处被截断，
  vLLM 不会调度超过上限的部分）和四个一起解码的客户端。
- 其他地方：`test_speed_defaults.async_problems` 在两个 profile 上运行它的检查（单独一个 `--async-scheduling`
  在 Gemma 上配默认值，等于给出每个开关的启动；用 `--no-speed-defaults` 时它需要
  `--decode-fastpath --decode-fastpath2`；各种拒绝）；`test_async_sched`（pin 测试，以及
  针对 Qwen 上共享修复的 `test_a_drain_with_no_worker_rpc_before_the_next_step`）。
- Scheduler 的构造（async-fix，已合并）：GPU 作业 3499729 表明，上面的 fake 从不构建 vLLM 的
  真实 Scheduler，而每个异步 Qwen 服务都在那里失败了（`defer_block_free` 在解耦布局上被拒绝，
  包括窗口布局的两个池）。`test_deferred_free_pools.GemmaServerKindTests` 运行 overlay 的
  `Scheduler.__init__`，所用的是 planner 为每种 Gemma 服务给出的 KV cache 配置（chat-text eager 和开 FULL
  graph、mswea-xml、kv-session、4,096 token 的前向；每种都带与不带 `--async-scheduling`），并在其池上做一次
  延迟释放。

## 8. GPU 检查必须显示什么（g1g 计划）

按 perf-async-g1 形式的 Gemma G1（在一张 GH200 上先 sync 后 async 的服务对，相同的判定规则），在 Qwen 的 G1
核验之后进行（作业 3532746 即 g1 r3 已核验；作业 3499729 即 g1 r1 在构造 Scheduler 时失败）：

- b = 1 下的作废 probe：停在 C' % 16 == 0 且 C' > W 处（为 C' 新分配一个全局块和滑窗块），以及停在
  (C' + 1) % 16 == 0 且 C' >= W 处（提前的前端释放），每个都带一个继续的轮次，读取提交的窗口
  和全局 tail；
- TRITON_ATTN 的 KV 更新在全局层和滑窗层中跳过 slot -1，eager 和 FULL graph 下都要
  （六个持久 slot mapping）；
- XML 标记停止（Gemma 的语法）和 chat-text 停止 id（0、1、50、106）；
- b = 4 和 16 的结构检查，以 stock Gemma 的 async 与 sync 差异作为读数；b = 16 时前端释放的停止会遇到
  其他 session 在同一次调度中的新滑窗块（3.1 节的 claim 转移），这是任何 Qwen 单元都覆盖不到的；
- 由于本分支修改了共享代码（`EngineLink.applied`、claim 转移），在同一 commit 上跑一个 Qwen 服务对。

perf-async-g1 的 g1g 中计划的服务对（先 sync，后 async，同一节点）：`ge` chat-text eager，`gg` chat-text
`--cuda-graphs full`，`gx` mswea-xml eager，然后是 g1 的 Qwen 服务对 `fe`（FA3 eager：同一 commit 上的
共享代码）（必需）；`gst` 使用 TRITON_ATTN 的 stock vLLM 0.24.0 Gemma（可选，最后跑）。`gk`（kv-session eager）不在
计划中：harness 没有 kv-session 客户端。共用参数：`--kv-cache-gib 25 --page-tokens 16 --max-model-len
65536 --language-model-only --connector-pool-gib 1`，以及三条快速路径加 `--deferred-completion-io`。
计划的时间安排：一张 GH200 上一个 02:00:00 的 ghx4 batch 作业（`--mem 113560M`），必需部分最坏 7,093 s（剩余 107 s），
预计约 3,606 s；最多 2.0 GPU 小时，预计约 1.0。perf-async-g1 于
2026-11-03 加入了 g1g 所需的内容：Gemma 的 pin 和计划长度，C' = 1104 和 1167 处的窗口 probe，窗口布局的宽松
合同（其摘要不比较，释放的槽位改名），READY 检查以及 stock Gemma 服务对。

GPU 运行测得的结果（取自各作业的离线核验，未审计）：
- g1g r2（作业 3549350，runtime `4ed1013`）：每个服务都到达 READY，b = 1 时 ge、gg 和 gx 三对中每个比较过的请求都 token 完全一致；
  每个 ge / gg 服务 97 个请求中有 12 个在 harness 的客户端超时。
- g1g r3（作业 3560784，runtime `4ed1013`）：运行了的每个服务都回答了全部请求，gg 和 gx 在 b = 1 时 token 完全一致，ge 对的
  同步服务因作业的时间预算被跳过。
- Gemma 的 MemoryAgentBench 冒烟 smga（作业 3601174，`4ebff97`：`goal_query`、`observation_query`、`random_fill`；
  kv-session，thinking 关）以 `--async-scheduling` 运行；不知道它是否遇到过第 1 节第 4 项的情形（一个回复达到
  `max_tokens`、同时另一个请求还在解码）。在 Qwen 上，作业 3604146 遇到它 16 次，没有出错
  （IMPLEMENTATION.zh-CN.md 第 11 节）。

## 9. 未解决的风险

- 只能在 GPU 上验证的：TRITON_ATTN 对 slot -1 的跳过、持久缓冲区中的作废、影子 `check_stop` 与 vLLM
  请求对象的对照、可重入的暂存更新 RPC（除滑窗组的 mapping 外都与 Qwen 共享）。在 Qwen 上，
  g1 r3（作业 3532746，`4ed1013`）在 b = 1 下检验了 FULL graph 持久缓冲区中的作废、影子 `check_stop` 和
  暂存更新的 RPC（b = 1 的每一对都逐 token 一致）。b = 1 的一致性无法体现两种后端上 KV 更新对 slot -1
  的跳过：下一轮的阶段 1 会在任何 query 读取 C' 之前重写它。
- Gemma 特有、在 2026-11-08（g1g r2 和 r3 之前）尚未解决的：六个 KV 组中的作废，以及 b = 16 时的 claim 转移。
  g1g r2 和 r3 向每个 Gemma chat-text 服务各发了 48 个 b = 16 的请求（见它们的离线核验记录，未审计），但第 8 节
  只记录了 b = 1 时的 token 一致性，所以本说明不把这两项算作已检查。
- 异步 step 之间的 probe pass（阶段 1 和 Goal forward 经由 `NativeQueryForward`，在一次
  排空之后）：跨越只有 probe 的前向时，runner 的 `prev_sampled_token_ids` 记账由 vLLM 负责；Qwen 的 G1 先
  覆盖它（共享）。确实如此：在 g1 r2（作业 3519531，`4a498dc`）中，vLLM 的 GPU runner 在一次阶段 1 pass 之后
  抛出 KeyError（`gpu_model_runner.py:1426`）；async-fix2（`4ed1013`）修复了它（IMPLEMENTATION.md 第 10 节）。
- H6 没有实现（空闲计数不比较），step profiler 没有适配，EngineLink 计数器没有写入
  证据（共享）。
- CPU fake 模拟 vLLM 的 scheduler、runner 和 kernel；它们并不运行这些。
