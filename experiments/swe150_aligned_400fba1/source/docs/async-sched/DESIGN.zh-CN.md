# Selective-prefix runtime 的异步调度：设计更新（方案 C 此后已实现）

[English](DESIGN.md) | 简体中文

日期：2026-10-30。更新 2026-10-16 的说明（基线 `integrate-1015` @ `6c5c9ab`；那份说明不在本
仓库中，本说明复述了其中用到的内容）。本文所读的基线：分支 `async-sched` @ `cf2b28d`（143
个 commit，在 `6c5c9ab` 之后）。路径相对于 `source-overlay/`，以顶层文件
（`serve.py`、`README.md`）或盘符开头的除外。行号是 `cf2b28d` 的行号。写作当日尚未实现任何内容。
本说明不做任何决定；第 6 节列出维护者当时需要决定的事项。

按实际实现：方案 C（4.3 节）已作为需显式开启的 `--async-scheduling` 进入 main（默认关闭，不属于速度
默认值）：async-c `f162120`、async-fix `a0798a8`、async-gemma `4a498dc`（Gemma 4 profile）和 async-fix2
`4ed1013`；实现了哪些内容、与本说明有哪些出入，见 [IMPLEMENTATION.zh-CN.md](IMPLEMENTATION.zh-CN.md) 和
[GEMMA.zh-CN.md](GEMMA.zh-CN.md)。它的 GPU 检查（Qwen 和 Gemma 4 上的 G1）及其判定见
[docs/status.zh-CN.md](../status.zh-CN.md) 和[历史](../history.zh-CN.md#历史)；Gemma 4 上仍然成立的限制见
[docs/known-limits.zh-CN.md](../known-limits.zh-CN.md) 5.3。第 0、4 和 6 节中的收益是本说明的估计值。实际实现定下了
第 6 节的第 1 和 2 项（方案 C）、第 3 项（宽松合同：H1 的回滚和 H6 都没有实现）和第 4 项（两个 profile
都做；异步时仍拒绝 `--phase1-in-step` 和 `--gdn-device-handoff`），并取代了第 5 节的“G1 和 G2 通过之后才合并到
main”（方案 C 已作为需显式开启的开关进入 main）。按实际实现，lifecycle sweep 不做排空（H7 列出了它）：见
IMPLEMENTATION.zh-CN.md 第 2 节和第 4 节。本页其余部分是 2026-10-30 的设计记录。

## 0. 摘要

1. **异步能带来什么：在 stock vLLM 0.24.0 上的测量**（一张 GH200，[docs/reference.zh-CN.md 4.11 节](../reference.zh-CN.md#411-解码快速路径默认开启)的 decode 重放）：
   每个 decode 步快 1.4-2.0 ms。用 FlashAttention 3 时为 4.8-9.2 %（作业 3429022），用
   TRITON_ATTN 时为 5.1-10.2 %（作业 3461668）。有完整测量窗口的轮次给出 7.4-8.3 %（FA3，b = 1）和
   5.1-10.2 %（Triton，b = 1 和 4）。表见 4.1 节。
2. **每轮值多少。** 10-26 的 Qwen Terminal-Bench dry run（作业 3461661）中，包括停顿在内的 decode
   在每个请求的服务端时间中占 80.9 %（`goal_query`）和 98.3 %（`observation_query`）。
   因此 decode 步缩短 5-8 % 最多使每轮服务端时间缩短 4.0-6.5 %（17.7 s 中的 0.72-1.15 s）和
   4.9-7.9 %（12.7 s 中的 0.62-1.00 s）。
3. **自 10-16 以来，风险的性质没有变。** 异步在读取第 N 步的 token 之前就启动第 N+1 步。
   因此每个在第 N 步停止的请求都会在 N+1 中多跑一行。
   - Qwen：这一行就地推进该请求的 Gated-DeltaNet slot，而 carry 在完成时从
     这个 slot 捕获。于是 carry 含有停止 token，而且什么都不会发现。这是唯一
     影响正确性的 device 端风险。
   - 停止位置的 KV：多出的一行还会写位置 C′ 的 KV（FA，以及 Gemma 的滑窗层
     和全局层）。下一轮的阶段 1 会在任何读取之前重写 C′。
   - Host 状态：多出的一步会分配同步服务从不分配的 block（每 16 次停止有 1 次，
     P = 16 时），并把我们的记录推进一个 token。
4. **10-16 以来的变化让精确的设计更可行，也更有用。**
   - `--decode-fastpath2` 已经把每步 forward 之后的工作延后到一个 `finish_step` 里。
   - `--decode-fastpath3` 和 `--deferred-completion-io` 削减了异步本来要隐藏的 host 工作。
   - Gemma 4 没有递归状态。
   - 我们的同步服务现在每个 token 的 decode 比 stock *异步*慢 2.3-5.5 ms，后者是组里的
     baseline。异步是 decode 上剩下的最大一项。
5. **只去掉 gate 会怎样（从我们的代码读出）。** 我们没有读合作者的 fork；
   下面是只去掉 `require`、`allocation_guard` 和 `chunk_protocol` 这几个 gate 时*我们的*代码会怎么做。
   - 每个以 EOS 或标记结束的完成都会在 worker 的 cursor 检查处直接失败
     （`remote_ra_live_v2/allocation_owner.py:488-491`）。
   - 如果这个检查也放宽，carry 会悄悄地包含停止 token
     （`worker_backend.py:261-269`）。
   - 无论哪种情况，每次 forward 仍在它自己的 `execute_model` 内部 join
     （`generation_guard.py:35-38`、`worker_composition.py:172-173`）。所以它重叠的只有采样等待和
     `update_from_output`，按我们的逐步计时约 0.5-1 ms。
6. **建议。** 如果要实现异步，就实现方案 C（4.3 节）：在 forward N 期间准备第 N+1 步，
   等 N 的停止判定之后才启动它，并在 device 上把真正停止的行作废。不要实现
   只去 gate 的形式。方案 F 精确但收益小。方案 S 最接近 stock，风险也最高。

## 1. 10-16 以来的变化

| 变化 | 位置（cf2b28d） | 对异步设计的影响 |
| --- | --- | --- |
| `--decode-fastpath2`（每步一次 `allocation_batch` RPC；forward 之后的工作延后到 `finish_step`） | `remote_ra_live_v2/decode_fastpath2.py:1-48`；`engine_boundary.py:287-319`（`flush_updates`）；`worker_composition.py:129-148`（`_defer`），`155-186`（`finish_step`，第 173 行的 `event.synchronize()`）；`generation_guard.py:40-42` | 每步的 worker 工作已经是一个延后的单元，而 batch 是一条消息，可以暂存到停止判定之后（方案 C）。但 `finish_step` 仍在 `execute_model(N)` 内部 join forward N，所以在它移到第 N+1 步的启动前时点之前，什么都不会重叠。 |
| `--decode-fastpath3`（稳态步复用 lease、key 元组和 claim 差异；不带行的 batch 条目；一次拷贝读取 position） | `decode_fastpath3.py:1-49`；`allocation_owner.py:1149`（`materialize_steady`）；`qwen.py:237-305`（`before_forward3`）；`positions.py:47-57`（`step_positions`：对 device tensor 调用 `.tolist()`） | 这些缓存以“worker 上次接受的行”为键。如果已停止请求的条目从不到达 worker（C），就没有什么需要回滚。如果到达了（S），已停止请求的缓存会超前一步，完成时必须把它们截回。从 device 读取 position 是一次 stream 同步：在 C 中无害（它在 N 确定之后运行），在 S 中是阻碍。 |
| `--deferred-completion-io`（在 `post_step` 之后由一个写线程编码并写出文件） | `deferred_completion.py:1-80`，`682-708`（`attach_engine` 包装 `post_step`）；[docs/reference.zh-CN.md 4.11 节](../reference.zh-CN.md#411-解码快速路径默认开启) | 它按 busy loop 的每次迭代工作，所以异步的 step 函数保留它。native 记录仍在引擎线程上、在 `update_from_output(N)` 内构建。异步时这发生在第 N+1 步的分配已经生效时，所以记录中的空闲 block 数会不同（H6）。worker 的 `complete` RPC（517 个新 Page 时约 40 ms，[docs/reference.zh-CN.md 4.11 节](../reference.zh-CN.md#411-解码快速路径默认开启)）仍在引擎线程上，所以完成在异步下与在同步下一样是一个气泡。 |
| P = 16 的解耦布局（FA 池为 16 token 的 block；Mamba 池有 K = 2S + 1 个 slot，M = 784） | `kv_layout.py:184-185`；[docs/reference.zh-CN.md 4.6 节](../reference.zh-CN.md#46-解耦-kv-布局与-16-token-的-page)；`pool_incarnations.py:8-19`（每个池的 incarnation） | C′ 落在 Page 起点（C′ % 16 == 0）的停止会让第 N+1 步分配一个同步服务从来没有的 FA block。这是每 16 次停止中有 1 次，是 784 token Page 时的 49 倍。跨 Mamba block（C′ % 784 == 0）时会分配一个 Mamba block。 |
| `--cuda-graphs full`（FULL decode graph；物理序列长度绑定；持久的 metadata 缓冲区） | `remote_ra_live_v2/cuda_graphs.py:165-196`；`vllm/v1/attention/backends/gdn_attn.py:464-476`（state indices 拷进持久缓冲区，填充部分为 `NULL_BLOCK_ID`） | 作废必须在 builder 的拷贝之后、重放之前写持久缓冲区。我们的 `before_forward` hook 恰好位于这里（`vllm/v1/worker/gpu_model_runner.py:4605-4606`）。graph 的 hidden state 输出是一个静态缓冲区：N 的 Page 压缩和 GDN 捕获必须在 N+1 的重放覆盖它之前入队。 |
| Gemma 4 31B IT profile（窗口布局：池 1 中有 50 个滑窗层，W = 1,024，另有 10 个全局层；TRITON_ATTN；拒绝 `--phase1-in-step` 和 `--gdn-device-handoff`） | `serve.py:951`（还有 `--no-async-scheduling`）；`sliding_window.py:56-76`；`remote_ra_native_growing/window_pins.py:102-122`；`window_rows.py:78-108`（fastpath3 的池 journal）；`window_owner.py:48`（继承 `_complete`）；[docs/reference.zh-CN.md 4.14 节](../reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)（`refused` 行，第 841 行） | 没有递归状态：多出的一行在 device 上只写 C′ 处的 KV，而阶段 1 会重写它（`sliding_window.phase1_row` 追加到最后一个保留的 block 中）。host 端的影响在池 1：C′ % 16 == 0 时多一个新 block，窗口之前的一个 block 提前一步释放，二者都记入 incarnation 计数和 fastpath3 的 journal。完成时的 cursor 检查与 Qwen 共用。 |
| Qwen 的 `--triton-attn`（ef62aa6） | [docs/reference.zh-CN.md 4.6 节](../reference.zh-CN.md#46-解耦-kv-布局与-16-token-的-page) | 两种后端的 KV 更新 op 都必须跳过作废用的 slot −1。这些 kernel 不在 overlay 中：要在 GPU 上验证。 |
| `--phase1-in-step`（10-16 以来未变） | `vllm/v1/engine/core.py:495-516`（probe hook 只在 `step` 中）；`phase1_step.py:158-230` | `step_with_batch_queue`（`core.py:536-649`）没有 probe hook。第 1 版中异步时拒绝这个开关；它不在 benchmark 配置中（README 第 41 行）。 |
| `--admission-overlap K`、挂起的 Goal、Gemma 的滑窗账本、gate 之前的释放 | `admission_overlap.py:1-30`；`goal_async.py:285`（`resume`）；`core.py:1366-1434`，`1454-1468`；`capacity.py:312-418`，`550`（`ledger_gate`）；`idle_release.py:206/283/310`；`window_capacity.py:262/302` | 这些都是引擎线程上的 pass 边界操作。每一项都需要先排空流水线（H7）。 |
| 测量 | 作业 3429022（FA3）和 3461668（Triton）验证目录中的 decode 表；作业 3429022 的逐步计时 `step-profile.csv`；作业 3461661 的逐轮 profile（本仓库之外的一份分析；4.1 节引用了其中的数字） | 4.1 节。 |

下面每一处仍然要求同步调度器：
- `serve.py:614`，`951`；
- `native_worker_registration.py:57-63`，`195-196`；
- `bootstrap_contract.py:21-22`，`29-30`（`max_concurrent_batches == 1`）；
- `startup_runtime.py:73`（"batch queue unsupported"）；
- `policy_runtime/bootstrap.py:39-45`；
- `private_authority/protocol.py:90`，`99-100`；
- `remote_ra_live_v2/positions.py:18`，每次生成 forward 都会运行（`hooks.py:20`）；
- `remote_ra_growing_worker/plan.py:106`；
- `remote_ra_qwen_semantics/hooks.py:32`；
- `allocation_guard.py:35-38`（输出占位）；
- `chunk_protocol.py:20-21`（`num_tokens == start + 1`）和 `38-44`（`worker_interval`）；
- 逐步 profiler：`remote_ra_step_profile.py:899-902`，`973-976`。

vLLM 自身的机制（10-16 说明的第 1 节）是：`core.py:536-649`，位于
`vllm/v1/core/sched/scheduler.py:490-522` 的占位，在 `1723-1732` 丢弃已完成的输出，位于
`gpu_model_runner.py:4952-5009` 的异步 runner 输出，以及在 `1748-1855` 从 `prev_sampled_token_ids` 取 input id。

## 2. 多出的 forward，逐步分解（当前代码）

请求 R 在第 N 步停止，已计算终点 C′ = prompt + 输出 − 1。在 stock 异步下：

1. `schedule(N+1)` 仍然包含 R。占位跳过只覆盖确定会发生的 `max_tokens` 停止
   （`scheduler.py:490-504`）。
2. `EngineRpcBoundary.dispatch_mixed_request` 推进 R 的记录（`engine_boundary.py:112-155`，`273-283`）。
   当 C′ % 16 == 0 时，vLLM 为第 C′/16 页分配一个 block。
3. `execute_model(N+1)` 内联运行（UniProc）。我们的 `before_forward` 为 R 打开 [C′, C′+1) 的 lease。
   forward 写 C′ 处的 KV，并推进 R 的 GDN slot。`finish_step` 关闭 lease，于是 `state.cursor`
   变为 C′+1（`qwen.py:419`），它还可能为第 C′/16 页压缩一个向量（`qwen.py:399-417`）。
4. 直到这时，`update_from_output(N)` 才发现停止并调用 `lifecycle.before_success`
   （`scheduler.py:1848-1866`）。这会走到 adapter 的 `complete`（`remote_ra_native_growing/adapter.py:409-414`，
   C′ 取自 vLLM 的输出）和 worker 的 `_complete`，后者要求
   `state.cursor == len(tokens) + len(output) − 1`（`allocation_owner.py:488-491`）。
   - 代码不变时：这个检查失败，请求以 `FINISHED_ERROR` 结束。
   - 放宽检查时：`qwen.finish` 从当前 slot 捕获 carry（`qwen.py:349-352`，
     `worker_backend.py:261-269`）。这个 slot 已经是 C′ 之后的状态，摘要也是
     从这些字节算出的。

## 3. 各所有者的风险，以及各自的处理方式

复杂度：S 少于约 100 行，M 为 100-300 行，L 超过 300 行（runtime 加 overlay，不含测试）。

### 3.1 Device 端

| # | 所有者 | 多出的一行做了什么 | 作废 / 回滚 | 避免这一行 | 复杂度 |
| --- | --- | --- | --- | --- | --- |
| D1 | Qwen GDN 运行 slot（48 层的 conv 和 SSM 状态，3 个 Mamba group，153.9 MB） | 被停止 token 就地推进。于是 carry（`worker_backend.py:261-269`）等于 C′ 之后的状态。下一轮的阶段 1 恢复它并再次计算 C′：**静默的错误 carry**。 | (a) **作废**：在 forward 之前，把每个 GDN group 的 `non_spec_state_indices_tensor`（FULL graph 下的持久缓冲区）中 R 的条目设为 `NULL_BLOCK_ID` = 0。kernel 随后跳过这一行（`fused_recurrent.py:295-299`，`fused_sigmoid_gating.py:113/162`，`causal_conv1d.py:814-817`）。host 必须在启动之前就知道真正的停止（方案 C）。(b) **快照**（方案 S）：在 forward N+1 之前，一个基于 `prev_sampled_token_ids` 的 device 端谓词（停止 id，加上在 XML 协议下解码结果含 '>' 的 token）把每个被标记行的 slot 拷到预留区域，carry 从该区域取。每个快照行需要 154 MB GPU 显存，还需要溢出时的后备（没有快照的停止会强制该 session 重建）。 | 只适用于部分停止（3.3）。 | (a) S，加上对每条 GDN kernel 路径的 GPU 检查。(b) M-L。 |
| D2 | Qwen 跨 Mamba block（C′ % 784 == 0，约每 784 次停止有 1 次） | 第 N+1 步取一个新的 Mamba block，vLLM 的 preprocess 把 R 的状态拷进去。旧 slot 仍然存在：align 模式释放的是两步之前的 block（`vllm/v1/core/single_type_kv_cache_manager.py:1086-1103`）。 | 有 D1(a) 时这次拷贝白做但无害；host 上的 block 属于 H1。在 S 下，carry 是 M 对齐终点处最近一次捕获（`qwen.py:337-347`），它在第 N 步就已取得。 | 不适用 | S |
| D3 | C′ 处的 FA KV（session 的 live tail Page；Gemma：全局 tail 和 50 个滑窗层最后保留的 block） | 写入停止 token 在 C′ 处的 KV。 | 每个 group 的 slot mapping（FULL graph 下是持久的）中的 slot −1 会跳过它。需验证 FA3 和 TRITON_ATTN 的 KV 更新 kernel 会跳过 −1。不作废时，只要不变式成立，这次写入就是**无害的**：下一轮的阶段 1 从 C = C′ 开始，并在任何 query 读取之前重写 C′（adapter C2；`sliding_window.py:69-76`），而且 live tail 从不被释放（[docs/reference.zh-CN.md 4.13 节](../reference.zh-CN.md#413-释放空闲-session-的-page需显式开启)）。没有任何东西对 tail 的字节做哈希。 | 不适用 | S |
| D4 | Page 向量和静态的 hidden state 缓冲区 | 当 (C′+1) % 16 == 0 时（每 16 次有 1 次），多出的一行写满第 C′/16 页，并压缩出一个同步服务从不生成的向量。FULL graph 下，N+1 的重放会覆盖 N 的 hidden 行。 | C：R 在 N+1 中没有 lease，所以不会生成向量，而 `finish_step(N)`（压缩、M 对齐的捕获）在 N+1 启动之前运行。S：在启动之前把 N 的 GPU 部分入队，稍后验证，在完成截断时丢掉 R 的向量。 | 不适用 | S（C）/ M（S） |
| D5 | Gemma 的滑动窗口 | 除了 pin 住的窗口，没有状态跨轮次；该窗口由它的 block 及其 incarnation 标识（`window_lifecycle.py:348-373`）。`WindowPins.pin_after` 从 R 的行读取 [C′ − W + 1, C′)，这一区间保持 live。 | 只有 D3；其余属于 host（H1）。 | 不适用 | S |

### 3.2 Host 端

| # | 所有者 | 哪里出现分歧 | 作废 / 回滚 | 复杂度 |
| --- | --- | --- | --- | --- |
| H1 | vLLM 的 block 池和我们的 incarnation 计数器（FA 池、Mamba 池、Gemma 的池 1；fastpath3 的 journal） | R 的推测性 block：C′ % 16 == 0 时的 FA block；跨块时的 Mamba block；Gemma 上每个 group 一个新的滑窗 block，以及一个提前一步释放的窗口之前的 block。R 释放时把它们还回空闲队列：`BlockPool.free_blocks` 把没有哈希的 block 放到队首（`prepend_n`），有哈希的放到队尾（`vllm/v1/core/block_pool.py:614-635`；队列有 `popleft`、`popleft_n`、`remove`、`append`、`prepend_n` 和 `append_n`，`vllm/v1/core/kv_cache_utils.py:232-366`）。生成的 block 从不做哈希（`skip_cache`，`engine_boundary.py:198`），所以 R 的 block 回到队首，但其中有一个同步服务从未取过的 block。于是队列的顺序与同步服务不同，之后的 block id 可能不同，incarnation 计数（`pool_incarnations.py:8-19`）和 journal（`window_rows.py:78-108`）也一样。token 不受影响。 | 精确：停止判定之后，把 R 的推测性 block 按原来的顺序放回队首（用队列已有的 `prepend_n`；不加新方法），把提前释放的 block 取回（`remove`）放进 R 的行，并减少 incarnation 计数。这只在 b = 1 时精确。b > 1 时，同一个 `schedule(N+1)` 中在 R 之后分配的请求已经拿到了移位后的 id，它们的 block table 也已在途。宽松：比较证据时允许 block id 和 incarnation 有一致的重命名。这是离线验证程序的改动，没有 runtime 回滚，也覆盖 b > 1（由维护者决定，第 6 节）。 | M（精确）/ 无（宽松） |
| H2 | 引擎记录（`EngineRpcBoundary`：`scheduled_cursor`、区间、`native_tables`、`accepted_rows`、已见 id、Mamba 快照；Gemma 的 `window_rows` 记录） | 被 `schedule(N+1)` 为 R 推进（`engine_boundary.py:273-319`）。 | R 已到终态，所以从 `schedule(N+1)` 之前取的副本恢复它的记录（或者证明完成时从不读取它）。 | S |
| H3 | worker 的 allocation owner（claim、`native_rows`、`last_tables`、fastpath3 的稳态缓存） | 被 R 在第 N+1 步的 `allocation_batch` 条目推进。 | C：暂存该步的 batch，去掉已停止请求的条目后再发送，所以不需要回滚。S：在完成时截断。 | S（C）/ M（S） |
| H4 | Qwen 和窗口的 lease 状态（`state.active`、cursor、`latest_hybrid`、`page_vectors`、`forwards`/`sequence`、执行状态） | 被推进一个 token。还有流水线本身：`enter_execute` 要求 `idle`/`joined`（`qwen.py:104-113`），`leave_execute` 要求所有 lease 都已关闭（`115-142`），`worker_interval` 要求 `active is None`（`chunk_protocol.py:38-44`）。 | C：R 的行不分配 lease。在 `execute_model(N)` 和 `before_forward(N+1)` 之间新增一个 `issued` 状态，让 `finish_step(N)` 在 N+1 的启动前时点运行，guard 在执行前的排空（`generation_guard.py:35-38` → `remote_ra_growing_worker/tensors.py:474-480`，`torch.cuda.synchronize`）变成一个 connector 检查。S：lease N 只是 issued 时 lease N+1 就打开，完成时接受一个超前一个 token 的 worker 并把它截回（cursor、向量、carry、frame、`worker_backend.py:350-354`）。 | M（C）/ L（S） |
| H5 | Transcript 和 commit（`after_drain` → `commit_transcript`，`composition.py:207-212`） | 没有。它提交的是 vLLM 的输出 id，而 vLLM 会丢掉已完成请求在 N+1 的 token（`scheduler.py:1723-1732`）。 | 不需要。 | 无 |
| H6 | native 证据和 `runtime.jsonl`（frame、排空之后的 `kv_pools.free_blocks_by_pool`；开 `--deferred-completion-io` 时也一样） | (a) 仅 S：R 多一个 `allocation_update` frame。(b) 空闲计数在 `update_from_output(N)` 中读取，此时 `schedule(N+1)` 已经为*所有*运行中的请求做过分配。即使有 H1 的回滚，b > 1 时它们也不同。 | (a) C：条目被暂存，所以没有 frame。(b) 按步边界报告计数：把 `schedule(N+1)` 在每个池取走的加回去，在 `pool_incarnations` 的包装中计数。 | S |
| H7 | 容量账本和每个 pass 边界操作：admission、Goal 恢复、阶段 1 收尾、cancel 和 abort、lifecycle tick、空闲释放和重新 admission、session 关闭、关停 | batch 在途时，空闲计数包含第 N+1 步，却缺少 N 的释放。lease 处于打开状态。一次单独经过 runner 的阶段 1 forward 会扰乱异步 runner 的 input batch。 | **Pass 排空**：在其中任何一项之前，先完成在途的 batch（读取其输出、`finish_step`、`update_from_output`），不再调度新的，然后完全按同步引擎的方式继续。vLLM 的 abort 队列也以同样方式排空。这时 gate 的决定就是晚一步的同步服务的决定。每次排空损失一步的重叠。16 个 session、回复约 500 token 时，每约 500 步有约 16-32 次排空（每个 session 的每一轮一次 admission，`goal_query` 另加一次 Goal 恢复），所以收益减少约 3-6 %。 | M |

### 3.3 避免多出的一行：停止预测

| 停止 | 在 `schedule(N+1)` 之前能知道吗？ | 用法 |
| --- | --- | --- |
| `max_tokens`（回复上限） | 能。stock 已经跳过这一步（`scheduler.py:490-504`）。 | 保持不变。 |
| 上下文上限（`max_model_len`） | 能（`num_tokens + 1 ≥ max_model_len`）。 | 在 overlay 中加进同一条规则。S。 |
| 动作标记 `</mswea_bash_command>`（仅 XML 协议，`composition.py:430-450`，`mswea_protocol.py:117-147`） | 能精确知道，依据截至 N−1 的文本。在 N 停止要求这段文本以标记的一个非空真前缀结尾，或者有一个词表 token 包含整个标记（启动时检查一次）。 | 把被标记的行留在第 N+1 步之外，或者把它们标记出来做作废或快照。这会减少 D1 适用的次数，但不能取代它。 |
| EOS、pad 和停止 id（所有 chat-text 和 kv-session 停止，部分 XML 轮次） | 不能：它们只有在 N 采样之后才在 device 上得知。 | 没有 host 端的避免办法。任何在读取 N 的 token 之前启动 N+1 的设计，都需要带真正停止的 D1(a)（C）或 D1(b)（S）。 |

完全避免推测性分配（不在会触发分配的步中调度该请求）会让每个请求
每 16 步中延后 1 步。其代价与异步的收益相当：16 步中多一步，就是每个请求的步数多 6.25 %，
而 stock 每步的收益是 5-8 %（4.1 节）。

## 4. 方案

### 4.1 收益：测量值与估计值

稳态 decode ITL，stock vLLM 0.24.0 的同步调度对异步调度（ms；窗口 `ok` = 至少
约 150 步，`short`/`missing` = 23-119 步；由各作业的 `itl-by-batch.csv` 重新计算）：

| (L, b) | FA3，作业 3429022：同步 → 异步 | Triton，作业 3461668：同步 → 异步 |
| --- | --- | --- |
| 2K b=1 | 18.59 → 17.05：−1.53 ms，−8.25 %（ok） | 18.61 → 17.14：−1.47 ms，−7.92 %（ok） |
| 20K b=1 | 18.95 → 17.40：−1.55，−8.20 %（ok） | 20.29 → 18.80：−1.49，−7.35 %（ok） |
| 117K b=1 | 20.70 → 19.16：−1.54，−7.44 %（ok） | 28.99 → 27.51：−1.48，−5.10 %（ok） |
| 2K b=4 | 19.26 → 17.55：−1.71，−8.88 %（missing） | 19.69 → 17.68：−2.01，−10.19 %（ok） |
| 20K b=4 | 20.53 → 18.87：−1.66，−8.08 %（short） | 21.55 → 19.96：−1.60，−7.40 %（ok） |
| 2K b=16 | 21.36 → 19.40：−1.96，−9.17 %（missing） | 21.51 → 19.55：−1.96，−9.10 %（missing） |
| 20K b=16 | 26.76 → 25.06：−1.70，−6.33 %（missing） | 27.11 → 25.40：−1.71，−6.32 %（missing） |
| W2（类 SWE）b=16 | 29.48 → 28.06：−1.42，−4.80 %（missing/short） | 30.94 → 28.99：−1.94，−6.28 %（missing） |

**本 runtime（FA3，作业 3429022）。**
- 开三条快速路径、不开 profiler 时的稳态 ITL：19.40 ms（2K b=1），24.85（2K b=16），31.88（W2 b=16）。
  相对 stock 异步分别为 +2.35、+5.45 和 +3.82 ms（Triton 上为 +2.32 到 +5.22 ms，作业 3461668）。
- fast3 服务的逐步计时（开着 profiler）给出的每步数据：
  - b=1：GPU forward 16.2 ms，host 3.4 ms。host 工作为 schedule 0.18、vLLM 的准备 1.79
    （`update_states`、`prepare_inputs`、attention metadata）、`worker.before` 0.18、forward 之后的工作 0.08、
    采样 0.67 和 update 0.05。
  - b=16，2K：forward 18.6 ms，host 9.0 ms。host 工作为 schedule 1.20、准备 2.61、
    `worker.before` 1.14、forward 之后的工作 1.94、采样 0.90 和 update 0.19。
  - 其他轮次在 b=16 时的 host 时间为 6.3-6.4 ms（20K 为 6.37 ms，W2 为 6.27 ms）。

**每轮（作业 3461661，Qwen Terminal-Bench dry run，FA3）。** 包括停顿在内，decode 时间占
`goal_query` 每个请求 17.73 s 的 80.9 %，占 `observation_query` 每个请求 12.67 s 的 98.3 %；每个请求的 ITL 为
29.7 和 29.0 ms。步收益 g 最多让每轮服务端时间缩短 g × 该占比，因为残差中
排在其他 session 的 admission 后面的停顿不会缩短：

| 步收益 | `goal_query` 每轮 | `observation_query` 每轮 |
| --- | --- | --- |
| 5-8 %（stock 测得的收益） | 4.0-6.5 %（0.72-1.15 s） | 4.9-7.9 %（0.62-1.00 s） |
| F：1-4 % | 0.8-3.2 % | 1.0-3.9 % |
| C：5-11 % | 4.0-8.9 % | 4.9-10.8 % |
| S：最多 12-22 % | 最多 9.7-17.8 % | 最多 11.8-21.6 % |

F、C 和 S 这几行是根据 profile 得出的估计，不是测量值（4.2-4.4 节）。

### 4.2 方案 F：前瞻调度（精确，收益小）

**做法。**
- `schedule(N+1)` 与 forward N 同时运行。执行 N+1 要等 `update_from_output(N)` 之后才开始。
- 在执行之前，把已停止的请求从 N+1 的 scheduler 输出中去掉。它们得到的分配
  会被回滚（H1、H2）。
- GPU op 的顺序与同步服务相同。

**范围。**
- overlay 的 `core.py`：一个 step 函数和对 scheduler 输出的修改，约 150 行。
- 池的回滚（`kv_cache_utils.py` / `block_pool.py`，用队列已有的 `prepend_n` 和 `remove`），加上
  incarnation 和 journal 的回滚，约 150-250 行。
- 第 1 节的 gate 和开关，约 150 行。
- 共约 500-700 行 runtime 代码和约 500 行测试。

**风险。** 低。没有 device 端改动；调度的 host 状态是唯一新增的所有者。

**测试。**
- CPU：在 MultiHarness fake 上，把去掉停止请求的结果与“从未调度”对比；把池回滚的结果与
  “从未分配”对比，用两个池和 Gemma 池上的随机序列。
- GPU：与同步服务相比 token 和证据都精确一致，b = 1 时逐位相同，b = 16 时
  只要各步的组成相同就完全相同。

**收益。** 只有 `step.schedule`：b=1 时 0.18 ms，b=16 时 1.2-1.26 ms（profile 测得）。即 b=1 时不到 1 %，
b=16 时约 4 %，约占一轮的 1-4 %。

### 4.3 方案 C：提前准备，停止判定之后再启动（精确；推荐的形式）

**一步的流程。**
1. 开启 vLLM 异步：`schedule(N+1)`、`execute_model(N+1)` 和 vLLM 自己的准备工作与 forward N
   同时运行。
2. 我们的 `before_forward(N+1)` hook（`gpu_model_runner.py:4605`）先确定 N 的结果：
   1. 等待 N 的采样 id 拷贝事件。
   2. 用 `update_from_output` 之后会用的同一个谓词判定真正的停止（先 `check_stop`，再
      `MarkerStop`；纯函数，不修改任何状态）。
   3. 运行延后的 `finish_step(N)`，仍在 N+1 的重放之前（D4）。
   4. 在持久缓冲区中作废已停止的行（D1(a)、D3）。
   5. 应用暂存的 `allocation_batch`，去掉它们的条目（H3）。
   6. 打开继续运行的行的 lease，然后启动。
3. 然后 `update_from_output(N)` 在 forward N+1 期间运行，所以完成时看到的 slot 和 tail 都未被改动。

**回滚。** 只有 vLLM 为已停止请求做的推测性分配（H1，
精确或宽松）和我们的引擎记录（H2）需要 host 回滚。每个 pass 边界操作都先排空（H7）。

**第 1 版中拒绝的。** `--phase1-in-step` 和 `--gdn-device-handoff`，与 Gemma 已有的做法相同。

**范围（大致行数）。**

| 部分 | 文件 | 行数 |
| --- | --- | --- |
| 开关、pin、把第 1 节中只允许同步的 gate 改为有条件的、launcher | `serve.py`；`native_worker_registration.py`；`bootstrap_contract.py`；`startup_runtime.py`；`policy_runtime/bootstrap.py`；`private_authority/protocol.py`；`positions.py`；`plan.py`；`qwen_semantics/hooks.py`；`remote_ra_step_profile.py` | 200-300 |
| 带 pass 排空的异步 step 函数（registered add、resume、settle、tick、abort、evict、close、shutdown） | `core.py` overlay，新的 `remote_ra_live_v2/async_steps.py` | 250-350 |
| 感知占位的分配；batch 暂存到停止判定之后 | `allocation_guard.py`、`chunk_protocol.py`、`engine_boundary.py`、`decode_fastpath2.py`、`window_rows.py` | 150-250 |
| 延后的 `finish_step`、`issued` 状态、去掉排空的 guard | `generation_guard.py`、`worker_composition.py`、`qwen.py`、`window_lifecycle.py` | 250-350 |
| Resolve hook：等待 token、停止谓词、作废（3 个 GDN group 和 1 个 FA group；Gemma 的 6 个 group；eager 和 FULL） | 新的 `remote_ra_live_v2/async_void.py`、`hooks.py`、`mswea_protocol.py` | 350-500 |
| H1 回滚（精确）和 H2 | overlay 的 `kv_cache_utils.py` / `block_pool.py`、`pool_incarnations.py`、`window_rows.py`、`engine_boundary.py` | 250-400 |
| H6 按步边界计数；deferred IO 的检查 | `remote_ra_http/evidence.py`、`deferred_completion.py` | 50-100 |
| **Runtime 合计** | | **约 1,500-2,250** |
| 测试 | 见下 | 约 1,200-1,800 |

**风险。** 中等。决定正确性的部分只能在 GPU 上运行：
- FULL graph 下持久缓冲区内的作废，覆盖两种注意力后端和每条 GDN
  kernel 路径；
- stream 上延后的 join；
- 做两次的停止判定，两次必须一致。

CPU fake 模仿 vLLM 的 scheduler 和 runner，并不运行它们。

**测试。**
- CPU，单元级：
  - 在随机输出、EOS 和标记的情形下，停止谓词与 `update_from_output` 的判定相同，
    使用固定的 tokenizer；
  - 在 eager 和 FULL metadata fake 上测试作废写入器；
  - 在随机的分配、释放和 touch 序列上，池回滚等于“从未分配”（FA、Mamba、
    Gemma 池 1、journal）。
- CPU，一个异步 MultiHarness：一个带 device 端 token 和拷贝事件的 fake runner，以及带 pass 排空的
  batch queue 顺序。
  - 设置：Qwen 和 Gemma fake，快速路径 1-3，`--deferred-completion-io`，`--admission-overlap 16`，
    `--release-idle-pages`。
  - 停止情形：C′ % 16 == 0、(C′+1) % 16 == 0、C′ % 784 == 0、标记、EOS、`max_tokens`、在途的 cancel 和
    abort。
  - 通过条件：b = 1 时，回复、plan、被关注的 Page、carry 摘要和证据与同步 fake 相同，
    b > 1 时只要组成相同也相同。
  - Identity probe：开关关闭时，启动命令、启动配置和证据都不变。
- GPU，ghx4 batch，thinking 关闭：
  - G1：严格检查，与 g1c 类似。stage-0 XML append 会话中的一个 session，11 轮，eager 和
    `--cuda-graphs full`，两个 profile，同一节点上异步对同步。token、选页、Page
    向量、carry 摘要和 native 文件必须逐字节相同（宽松合同：除 block id 的
    重命名之外相同）。
  - G2：spd3 decode 重放（2K、20K、117K、W2；b = 1-16；FA3 和 Triton），我们的异步和同步，旁边是
    stock 的同步和异步。报告 ITL，以及在 stock 自身异步对同步的波动范围内的回复一致性（作业 3429022 中
    243 个回复里有 202 个相同）。
  - G3：一次 TB16 或 SWE16 dry run，检查 admission、排空、空闲释放和计数器（作废的行、
    回滚、排空）。

**收益（估计，未测量）。** 如果 C 重叠的正是 stock 隐藏的部分，至少有 stock 的 1.5-2 ms。按
profile（作业 3429022 的 `step-profile.csv`，开着 profiler；逐步 profiler 本身每步约耗
0.2-0.5 ms），C 重叠的是 schedule、准备和 update：b=1 时约 2.0 ms（约占该步的 10 %），b=16 时
3.5-4.0 ms（W2 3.52，20K 3.61，2K 4.01 ms：分别占 profile 步长 31.91、30.19 和 27.59 ms 的 11-15 %）。仍然
串行的是 forward 之后的工作、`worker.before`、采样等待和作废：b=1 时约 1 ms，b=16 时 2.0-4.0 ms
（W2 1.98，20K 2.01，2K 3.98 ms，不含作废）。重叠部分是收益的上界。本说明
2026-10-30 的估计按记录保留，未经测量（G2 尚未运行）：每步约 5-11 %，每轮
4-11 %。按这个估计，我们在 b=1 时接近 stock 异步，在 b=16 时接近 stock 同步。

### 4.4 方案 S：按 stock 顺序的流水线（收益最大，风险最高）

**做法。**
- 与 stock 一样，在读取 N 的 token 之前启动 N+1。
- Qwen：D1(b)，对可能停止的行的 slot 做快照。Gemma 不需要 device 端改动（D3、D5）。
- lease 模型完全流水线化（H4 的大改动）。
- host 对 device tensor 的读取移出 `before_forward`：`positions.py:47-57`、压缩的
  device 到 host 拷贝。
- 完成时截断超前一个 token 的 worker。
- H1、H2 和 H6 与 C 相同，另加 R 多出的 frame。

**范围。** 约 3,000-4,000 行 runtime 代码和约 2,000 行测试。每个快照行需要 154 MB GPU 显存，
与 25-32 GiB 的 KV 和 rewriter 放在一起很紧张。

**风险。** 高。误报只多一次拷贝，但快照区域溢出需要一个会重建该 session 的
后备方案。forward 之后的每项检查都会推迟。

**测试。** 与 C 相同，另加快照溢出和从快照取得的 carry。

**收益。** 最多是我们与 stock 异步的差距：b=1 时 2.3-2.4 ms（约 12 %），b=16 时 3.8-5.5 ms（12-22 %），
再减去快照拷贝和排空的开销。

### 4.5 只去 gate（合作者 fork 所描述的形式）：不是可选方案

第 0 节第 5 项和第 2 节描述了我们的代码会怎么做。结果要么是每次
EOS 或标记停止时完成失败，要么是静默的错误 carry，取决于完成时的检查。join 仍留在原处
（H4），所以收益约 0.5-1 ms。他们的运行不能作为本 runtime 上异步的证据。

## 5. 建议

1. 只以方案 C 的形式实现异步，需显式开启（`--async-scheduling`；默认关闭，并用 identity probe 确认
   不开时一切不变），两个 profile 都做。异步时拒绝 `--phase1-in-step` 和
   `--gdn-device-handoff`。
2. 安排工作顺序，让每个阶段都能先在 CPU 上检查，再上 GPU：
   1. gate 和开关；
   2. 延后的 `finish_step` 和不做排空的 guard，此时异步仍关闭（证据必须保持
      逐字节相同）；
   3. pass 排空；
   4. 感知占位的分配和暂存的 batch；
   5. resolve hook 和作废；
   6. H1 和 H2 的回滚；
   7. 异步 MultiHarness；
   8. GPU G1（b = 1，两个 profile），然后 G2 和 G3。
3. G1 和 G2 通过并获批准之后才合并到 main（与每个 runtime 改动一样）。
4. 只有当 C 在 G2 中测得的收益远低于预期、并且要求与 stock 异步持平时，才考虑 S。
5. 不要合并只去 gate 的形式，也不要用它跑 benchmark。把第 2 节的结论告诉合作者。

## 6. 维护者需要决定的事项

1. **是否实现。** 每个 decode 步约 5-11 %（估计；stock 测得 5-10 %），每轮服务端时间约 4-11 %。
   代价是 C 约 1,500-2,250 行 runtime 代码加 1,200-1,800 行测试，
   以及三个 ghx4 batch 作业（G1-G3）。
2. **选哪个方案：** C（推荐），F（精确，约 1-4 %），或 S（最多 12-22 %，风险高）。
3. **证据合同。**
   - 精确：H1 回滚，b = 1 时 native 文件与同步服务逐字节相同。b > 1 时 block
     id 无论如何都会移位（H1）。
   - 宽松：token、选页、Page 向量和 carry 精确一致；block id 和 incarnation 在一致的重命名
     之下相同；空闲计数按步边界报告（H6）。这样能省下约 250-400
     行 runtime 代码，并把比较移到离线验证程序里。
4. **Profile 和拒绝项：** 第 1 版同时支持两个 profile（推荐），或者先做 Qwen。确认异步时
   仍拒绝 `--phase1-in-step` 和 `--gdn-device-handoff`。
5. **Benchmark 和论文。** G1-G3 通过后，Table 1 的服务是否改用异步。stock 对照
   已经在跑 stock 异步。论文中的延迟仍是“优化后的 runtime 对 stock”，只取自经过 GPU 验证的
   commit。
6. **合作者的 fork：** 是否把本说明（第 2 节）发给他们，以及他们的运行是否
   要搁置。
