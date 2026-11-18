# --async-scheduling 方案 C：实现了什么（Qwen 已在 GPU 上以 b = 1 检查）

[English](IMPLEMENTATION.md) | 简体中文

日期：2026-10-31（第 8-10 节：2026-11-03 到 2026-11-05；第 11 节：2026-11-12）。分支 `async-c`，基于 thinking-strip `8ca8511`（设计说明
`DESIGN.md` 从 `async-sched` `3e5715e` cherry-pick 而来）。截至当日，这里的内容都还没有在 GPU 上运行过。
在 device 上决定正确性的部分（持久缓冲区中的作废、kernel 跳过作废的
行、停止判定与 vLLM 的一致、从 `execute_model` 内部发出暂存的更新）要由 G1 来证明
（G1 在 Qwen 上测得了什么，见第 6 节末尾）。代码：`source-overlay/remote_ra_live_v2/async_sched.py`；测试：
`tests/test_async_sched.py`。

按实际实现：async-c `f162120`、async-fix `a0798a8`（第 8 节）、async-gemma `4a498dc`（第 9 节）和 async-fix2
`4ed1013`（第 10 节）已在 main 中；async-maxtok-1112 `13ee618`（第 11 节）随 integrate-1112 `4ebff97` 进入，
后者自 2026-11-15 起是 `main` 的 runtime。各部分的 GPU 检查（Qwen 和 Gemma 4 上的 G1、第 11 节的 MemoryAgentBench
作业）及其判定见 [docs/status.zh-CN.md](../status.zh-CN.md) 和[历史](../history.zh-CN.md#历史)；Gemma 4 上仍然成立的
限制见 [docs/known-limits.zh-CN.md](../known-limits.zh-CN.md) 5.3。本页其余部分是各节所注日期时的实现记录。

## 1. 开关及其拒绝条件

- `serve.py --async-scheduling`（默认关闭）。它需要 `--decode-fastpath2`。
- 与 `--gdn-device-handoff` 或 `--phase1-in-step` 同用时被拒绝。async-c 拒绝 Gemma 4 profile；
  async-gemma 接受它（第 9 节，[GEMMA.zh-CN.md](GEMMA.zh-CN.md)）。
- 开启时：
  - 配置固定 `async_scheduling: true` 和 `effective_server_config.async_scheduling: true`；
  - vLLM 得到 `--async-scheduling`，而不是 `--no-async-scheduling`；
  - 前端比较 vLLM 解析出的开关（关闭时：vLLM 一定不能把它解析为开启）；
  - 引擎向 worker 发送记录 `async_scheduling: true`，并比较 worker 的回显。
- 原本只允许同步的门控现在要求 vLLM 的 scheduler 当且仅当配置固定异步时才是异步的：
  - 固定时 `max_concurrent_batches` 为 2，否则为 1；
  - 组装时 batch queue 存在且为空；
  - worker 门控和每次 forward 的 `runtime_mode` 检查遵循同一规则。
  - 这些门控位于 `bootstrap_contract`、`policy_runtime/bootstrap`、`native_worker_registration`、
    `startup_runtime` 和 `positions.runtime_mode`。
- 不变的部分：
  - `private_authority.Topology`，它声明 authority 自己的进程拓扑；
  - 旧的 `remote_ra_growing_worker/plan.py` 和 `remote_ra_qwen_semantics/hooks.py`，它们仍只支持同步。
- Gemma：async-c 中拒绝（窗口布局的 owner、行和快速路径 3 的 journal 都没有检查过，而且作废
  需要 Gated-DeltaNet 状态索引）；自 async-gemma 起接受，见第 9 节。

## 2. 实际实现的 step（DESIGN 4.3 节）

1. **vLLM。** 两个批次时，`step_with_batch_queue` 运行 `schedule(N+1)` 和 `execute_model(N+1)`（它原生的
   准备工作），与此同时 forward N 在运行。
2. **暂存的更新。** `async_sched.attach_engine` 解开快速路径 2 带 flush 的 `schedule` 包装，所以该 step 的
   分配更新留在队列中（`EngineRpcBoundary.async_hold`）。
3. **占位符。** 分配检查和区间检查接受 RUNNING 状态的 decode 请求上有一个输出占位符
   （`allocation_guard` 和 `chunk_protocol` 中的 `placeholders=True`）。同步时没有任何变化。
4. **Resolve。** guard（`async_sched.guarded`）不先排空 device（只做 connector 检查），也
   不进入 step。runner 的 `before_forward` hook 调用 `resolve`：
   1. `settle`：已发出的 step N 被延后的、forward 之后的工作（`finish_step`）关闭它的 lease，然后
      `leave_issued`。
   2. 从 vLLM 的 `prev_sampled_token_ids` / `prev_req_id_to_index` 读取一次 N 的采样 id。这是一次
      device 读取；DESIGN 写的是“拷贝事件”，见第 4 节。只读取 N 中被 N+1 调度的请求（async-maxtok-1112，
      第 11 节）：如果某请求在 N 的 token 是它的第 `max_tokens` 个，或达到 `max_model_len`，vLLM 的
      `schedule(N+1)` 会跳过它（token 预算或抢占也可能漏掉别的请求），runner 随后把它移出批次和
      `prev_req_id_to_index`。它的停止由 `update_from_output(N)` 判定，与同步服务相同；它在 N+1 中也没有
      需要作废的行。
   3. 停止由 `EngineLink.stops` 判定（async-gemma：只对引擎尚未应用的 token 判定，
      `EngineLink.applied`，第 9 节）：
      - vLLM 自己的 `check_stop`，作用于 `_Shadow(request, token)`：计数和输出 id 前进一位，写入留在
        影子对象上；
      - 然后是 `marker_stops`：MarkerStop 在追加之后的答案，不写它的逐请求表。
      - 不修改任何状态。
   4. N+1 中请求已在 N 停止的每一行都被作废（`void_rows`），作用于 runner 在 builder 之后暂存的
      metadata（`gpu_model_runner`：`_remote_ra_step_metadata`，仅在 `use_async_scheduling` 时）：
      - 每个 GDN metadata 的 `non_spec_state_indices_tensor[row] = NULL_BLOCK_ID`；
      - 该 token 的 slot 在每个 FA 层的 metadata、它的逐层 slot mapping 以及每个 FA
        组的 mapping 中都是 -1；
      - Mamba 组的 slot mapping 从不写入。
   5. `EngineLink.flush(voided)`：
      - 发出暂存的更新，去掉已停止请求的条目；
      - 恢复它们的 `native_allocation` 和 `native_tables`（H2）。
   6. 继续的请求进入 step（`enter_execute`）。`scheduled_rows` 跳过作废的行，就像它
      跳过 probe 行一样，所以不为它们生成 lease、向量或更新（H3、H4、D4）。
5. **execute 之后。** guard 把已进入的请求标为 `issued`（`qwen.issue_execute`）；它们 forward 之后的
   工作等待下一次 resolve。
6. **更新与检查。** 随后 `update_from_output(N)` 在 forward N+1 期间运行。之后引擎要求
   每个预测的停止都已结束（`step_with_batch_queue` 和排空中的 `EngineLink.check`）。
   预测停止却没有停止，就是丢了一个 step，会直接失败。
7. **未预测的停止。** 没有预测到的停止在 N+1 中带着一个打开的 lease 运行了。它的完成会直接
   失败：worker 的 cursor 是 C'+1，而已计算终点是 C'，或者它的 lease 仍是打开的。

**Pass 边界。**

- 引擎先排空（`EngineCore._remote_ra_async_drain`）：
  - **何时：** 每个客户端请求（已注册的 add、abort、utility 调用）、Goal 恢复和关停之前；
  - **做什么：** 取出在途批次并更新，不再调度新的 step。
- 生命周期清扫不排空。它只让注册和非活动记录过期，它发出的任何 worker RPC
  都会收尾已发出的 step。
- 在 worker 上，`worker_dispatch` 在以下时刻收尾已发出的 step：
  - 在 `allocation_batch` / `allocation_update` 以外的任何 RPC 之前；
  - 以及在该 step 中某个请求的终态操作（`complete`、`cancel`）之前。

## 3. 证据合同（放宽）

- **完全一致：** token、选页、Page 向量、carry 和 cursor。
- **可改名：** block id 和 incarnation，以及各种身份（request id、generation、binding、nonce、lease id）
  可以相差一个一致的改名。
- **不比较：** 基于改名值的摘要和 MAC（`lease_digest`、`receipt_digest`、`request_digest`、`mac`、
  `plan`）不比较，也不检查它们的一致性。
- **比较器**（`test_async_sched.relaxed_differences`），在一个 native 文件之内：
  - 每个 frame、字段和 KV 组的 block 改名都是一一对应的，身份改名也是；
  - 释放的 slot 按相同的 (group, kind) 多重集比较，每个组的不同 block 数
    也相同；
  - 空闲计数和其他取决于交错顺序的字段（`kv_pools`、`http_measurement`、
    `admission_engine_ms`、`free_at_terminal`）不比较。
- **GPU harness 的比较器**（perf-async-g1 的 `ag1checks.relaxed_differences`，不在本仓库中）是
  把这个比较器扩展到两个服务进程：更多的身份键（其中有 `server_run_id`、`process_start_id` 以及
  carry 和 Page 的 `digest` / `content` 标识），更多的摘要键，计时字段不比较。
  - 在 Qwen 布局上，它没有列出 worker frames 的 `state_digest`（每个进程各自的身份标识），所以
    按值比较它。内容字段以外的差异就是部分通过的发现 NATIVE_RELAXED_DIFF。
  - 只是这一点就让作业 3532746 成为 PARTIAL：`worker_frames[].state_digest` 和
    `worker_frames[].worker_receipt.body.state_digest` 在 fe、fg、te 和 tg 的 37 个 b = 1 请求中有 20 个不同，
    在 se 和 xe 的 4 个中有 3 个不同。
  - 这些差异每一个都是一一对应的改名（每一对中 0 处冲突）。在本地重新比较，
    像其他身份标识一样给 `state_digest` 改名，结果为 ASG1_PASS。
  - 这是 harness 的缺口，不是 runtime 的。修正留给 perf-async-g1 的下一次重新固定：在 Qwen 布局的
    身份键中加入 `state_digest`，并加一个测试。（本页的比较器也没有列出
    `state_digest`。）
- **未实现：H6（“以 step 边界为准的空闲计数”）。** native 文件和行中的空闲计数按读取时的值记录，
  即下一个 step 的 schedule 取走它的 block 之后的值。G3 的验证程序必须修正它们，或者
  不比较它们。
- **H1（推测 block 的精确回滚）：** 按放宽的合同，没有实现。

## 4. 与 DESIGN.md 的偏差

1. **token 读取。** 采样 id 的读取是 resolve 中对 `prev_sampled_token_ids` 的一次 device 读取，而不是等待
   vLLM 的异步拷贝事件。
2. **停止判定。** 它是作用于影子请求的 vLLM `check_stop`，加上引擎一侧对每个
   预测停止都已结束的检查。如果判定读取了影子请求不推进的属性，结果就会不一致；两个
   方向都会直接失败。
3. **暂存更新的 flush。** worker 的 resolve 在同一线程上（UniProc）调用引擎边界，引擎边界再通过 RPC
   回到 `execute_model` 内部的 worker。需在 GPU 上检查。
4. **生命周期清扫之前不排空**（第 2 节）。
5. **H6 未实现**（第 3 节）。
6. **Commit。** 第 1-7 阶段是一个 runtime commit：它们相互依赖，并且一起测试过。
7. **Profiler。** step profiler（`REMOTE_RA_STEP_PROFILE=1`）没有做适配；在异步下未经测试。
8. **结构化输出。** vLLM 延后的结构化输出采样路径没有被显式拒绝；本 runtime
   不向主引擎发送结构化输出请求。

## 5. 异步下的 --thinking-history strip（组合规则）

- **来源。** strip 带位置的 carry 是：
  - `phase1`：阶段 1 在 e 处的状态，与运行 slot 分开保存；或者
  - `split`：由在 X 处结束的 local turn 0 prefill forward 捕获。
- **为什么多出的那一行碰不到它。** 两者都在请求的第一个 decode step 之前取得，从不取自
  运行 slot，所以已停止请求多出的那一行碰不到 carry。不生成 X 之后的 Page。
- **两个选项可以组合。** 作废照样生效：它使 slot 和 C' 处的 KV 保持不变，并让
  这一行不产生任何东西。
- **快照来源。** 第三种来源 `snapshot`（方案 S）没有实现，仍被拒绝。它的谱系规则是
  `async_sched.SNAPSHOT_RULE`：
  - 位置是提交终点；
  - 它由 lease 在那里结束的那次 forward 从请求自己的运行 slot 中捕获，并且在该请求任何更晚的
    forward 启动之前；
  - 其间该 slot 没有被作废、迁移或释放；
  - 分配的 generation 和身份标识相同；
  - 否则 session 执行重建。

## 6. G1-G3 必须测量什么

**G1**：严格检查，一块 GH200。关闭 thinking，另加一次 strip 运行。在同一节点上对比异步与同步。eager 和
`--cuda-graphs full` 都跑，FA3 和 `--triton-attn` 也都跑。Qwen 使用 11 轮的 XML session（类似 g1c）和一个
chat-text session。先 b = 1，再 b = 4 和 16。

1. **每一轮，对比同步：**
   - token、选页、Page 向量和 carry 摘要逐字节相同；
   - native 文件在 `relaxed_differences` 下相等。
2. **计数器：**
   - `link.check` 失败：0；
   - 未预测的停止：0；
   - `voids` 等于 vLLM 在其后继续调度的停止数（EOS 和 marker；vLLM 跳过 `max_tokens` 停止）。
3. **device 上的作废：**
   - 作废的行不改变它的 GDN slot（carry 摘要等于同步的）；
   - C' 处的 KV 不被改动，即 FA3 和 TRITON_ATTN 的 KV 更新以及每条 GDN kernel 路径都跳过 slot -1 和
     NULL_BLOCK_ID，在 FULL graph 的持久缓冲区中也是如此；
   - 覆盖跨 Mamba block 的停止（C' % 784 == 0）和 Page 起点的停止（C' % 16 == 0）。
4. **暂存的更新：** 从 `execute_model` 内部的 flush（重入 RPC）能正常工作，并带有它的 frame 和测得的
   字节数。

**G2**：spd3 的 decode 重放（2K、20K、117K、W2；b = 1-16；FA3 和 Triton）：

- 我们的异步对比我们的同步的 ITL，以及对比 stock 同步和异步；
- 回复一致率落在 stock 自己异步对同步的离散范围内；
- 每个 step 的 resolve 等待（token 读取）和收尾用时；
- 每 1,000 个 step 的排空次数。

**G3**：一次 TB16 或 SWE16 dry run（benchmark 开关加上 `--async-scheduling`）：

- 排空下的 admission、Goal 恢复、abort 和 session 关闭；
- `--release-idle-pages` 和 `--admission-overlap 16`；
- 计数器（作废、resolve、排空）；
- 每轮的服务时间对比同步；
- 空闲计数证据（H6）被正确读取。

**G1 在 Qwen 上测得了什么**（perf-async-g1 g1 r3，作业 3532746，`4ed1013`；它的判定见
[docs/status.zh-CN.md](../status.zh-CN.md)；数字取自该作业的离线核验，未审计）：

- 每个服务都回答了每个请求（fea、fga、tea 和 tga 为 97 个中 97 个，xea 为 12 个中 12 个，sea 为 4 个中 4 个），
  b = 1 的每一对都逐 token 一致（fe、fg、te 和 tg 为 37 个中 37 个；xe 和 se 为 4 个中 4 个）。它的 PARTIAL
  只来自 harness 比较器的 `state_digest` 缺口（第 3 节）；把它改名后在本地重新比较，结果为 ASG1_PASS。
- 在 b = 1 下，这证明了第 1 项、第 3 项的 GDN 作废（eager 和 FULL graph，FA3 和 TRITON_ATTN）、停止判定，以及
  第 4 项从 `execute_model` 内部发出暂存的更新。检查作废的请求停在一个 Page 起点（C' = 608）和一个跨 Mamba
  block 处（C' = 1568），每个 chat 服务上都是如此。
- b > 1 是测量：在每个有 stock 参照的单元上读数 B1 都达到了，只有 te 的 b = 4 例外（12 个中 7 个
  相同，stock 为 12 个中 12 个）。
- 没有测量：第 2 项的计数器。EngineLink 的计数器（作废、resolve、排空）没有写进任何证据，所以 G1 依靠的是
  每次停止之后那一轮在 b = 1 下的一致性。没有观察到：第 3 项中 KV 更新跳过 slot -1，因为下一轮的阶段 1
  会在任何 query 读取 C' 之前重写它。

## 7. 未解决的风险

- **只能在 GPU 上确认的正确性：** 作废的写入、kernel 的跳过语义、影子 `check_stop` 与
  真实 vLLM 0.24.0 请求对象的对照，以及重入的暂存更新 RPC。
- **收益小于估计：** 每次 admission 都排空一个 step 的重叠，而且 resolve 要等 N 的
  采样完成，N+1 的 forward 才能启动（这是 C 固有的）。DESIGN 4.1 节估计每个 step 5-11 %；在当时还没有任何
  测量。之后 SWE-bench 的一对作业 3611917 / 3612333（`4ebff97`）测量了它：按 batch 大小比较，稳定的 decode step
  是同步时的 0.892（[docs/status.zh-CN.md](../status.zh-CN.md)，未审计）。
- **只有 fake：** vLLM 异步 scheduler 的内部行为（占位符、`max_tokens` 跳过、丢弃已结束请求的
  在途 token）只在 fake 中建模。在 async-maxtok-1112 之前，fake 在 N+1 中调度每个运行中的请求，并把
  已移出的请求留在 `prev_req_id_to_index` 中，所以没有抓到作业 3568311 的故障（第 11 节）。
- **H6 未实现。**

## 7a. 与 --goal-overlap 同用（async-int）

- **为什么可以组合。** `--goal-overlap` 改变 goal_query admission 的工作在哪里运行，但从不把它放进 decode step：
  - 阶段 1 在该 admission 的 pass（一个已注册的 add）中运行；Goal 在它的 Goal 线程上构造；
  - 后续部分（Goal join、作为单独 lease 的 Goal forward、阶段 1 的 joined drain、`plan_turn`、
    采用和预留）在 `EngineCoreProc._remote_ra_resume` 的 pass 中运行。
  - 两个 pass 都是 pass 边界：引擎先排空在途批次（第 2 节）。worker 也会在
    分配更新以外的任何 RPC 之前收尾已发出的 step，所以 Goal forward 的 RPC 和阶段 1
    从不会遇到在途 step。
  - `--phase1-in-step` 是唯一把 admission 工作放进 step 的选项，与 `--async-scheduling` 同用时仍被拒绝。
- **测试**（`test_async_goal_overlap.py`）：test_adm_overlap 的三个 goal_query session（16 token 的 Page、
  `--admission-overlap 3`、替身 rewriter、Goal 停在各自的线程上），开 `--goal-overlap`，
  分别以同步方式（快速路径 1 和 2）和按 vLLM 的异步顺序服务。每一轮的输出、选页、carry、Page
  向量、Goal、Goal 哈希和 worker 操作都相等；每个 native 文件在放宽的合同下相等
  （时钟字段和 goals.jsonl 引用不比较：它们包含时间和 request id）。停止被作废，Goal
  恢复遇到在途批次，Goal forward 作为单独的 lease 运行。
- **变异。** 跳过 Goal forward 会被拒绝（`Goal forward missing or foreign`）；漏掉作废会被发现。
  如果引擎在 Goal 恢复之前不排空，运行会被拒绝（`complete computed/sealed end mismatch`），或者在
  某些交错顺序下仍然相等：排空是必需的，缺少它会直接失败。
- **Launcher。** 在 mswea-xml 和 chat-text 上，`--async-scheduling` 加 `--goal-overlap`（以及 `--admission-overlap`），
  thinking 关闭、keep 和 strip，就是异步服务的启动加上 admission 记录；loader 和
  `compose_engine` 接受它。vLLM Goal rewriter 的子进程保留它自己的引擎参数：它的记录不随
  `--async-scheduling` 改变。
- **没有覆盖。** 没有按异步顺序端到端运行 chat-text 服务（strip 或 keep）或 HTTP mswea-xml 服务：
  它们的引擎循环没有异步 fake（与单独开 `--async-scheduling` 时一样）。keep 模式
  只改变 transcript 层；strip carry 的来源就是第 5 节的那些。

## 7b. H6（以 step 边界为准的空闲计数）：只有证据需要它

- **不是正确性的输入。** 每个读取空闲计数的决策（容量门控、`--release-idle-pages`、
  admission 账本）都在 pass 边界上、排空之后运行，这时没有已调度的 step 占用 block：计数就是
  同步服务的计数。端到端测试在不比较空闲计数的情况下比较每个决策，结果都相等。
- **哪里不同。** native 文件的 `kv_pools`、`free_at_terminal` 和 HTTP measurement 的空闲计数在
  完成时读取，即在 `update_from_output(N)` 内部、`schedule(N+1)` 取走它的 block 之后。它们把 N+1 的新 block
  算作已用。
- **设计（未实现）。** 在 resolve 时，按池记录 N+1 的暂存更新新分配的 block（即
  `EngineLink.flush` 发出的条目，减去作废请求的条目）；`update_from_output(N)` 期间的完成把它们
  加回它写出的空闲计数，并把这个修正记为 `free_correction`，以便验证程序撤销它。规模：
  `async_sched` 和 `deferred_completion` / `evidence` 中约 60-100 行，再加一个测试：异步运行修正后的计数
  等于同步运行的计数。
- **在此之前**，验证程序不比较空闲计数（`perf-async-g1` 就是这样做的）。

## 8. async-fix：解耦布局上延后的 block 释放（GPU 作业 3499729）

- **什么失败了。** 在 perf-async-g1（作业 3499729，runtime `f162120`）中，每个开 `--async-scheduling` 的 runtime 服务
  （fea、fga、tea、xea、sea、tga）都在 EngineCore 构造 Scheduler 时退出；同步服务和 stock
  vLLM 的异步服务都运行了。`vllm/v1/core/sched/scheduler.py` 抛出 `decoupled KV layout does not support deferred
  block frees (overlapping batches with a consumer KV connector)`。
- **原因。** vLLM 设置 `defer_block_free` 的条件是批次重叠（`max_concurrent_batches > 1`：异步调度）且
  KV connector 是 consumer（`is_kv_consumer`：我们的 OffloadingConnector 是 `kv_both`），所以已释放请求的 block
  要等可能仍会写入它们的 step 处理完之后才回到池中（`processed_step_seq >= fence`）。
  解耦布局的工作（exp-decouple-pages，DESIGN.md 第 4 项）拒绝这种组合，因为延后释放的
  条目把所有组的 block 放在一个列表里，而 `_drain_deferred_frees` 把它们全部还给池 0
  （`kv_cache_manager.block_pool`），而且当时本 runtime 中它是关闭的。async-c 把它打开了；它的 CPU 测试使用的 fake
  引擎从不构造真实的 Scheduler（而且 test_kv_pools 断言了这个拒绝）。
- **修复。** 这个拒绝去掉了。在解耦布局上（`kv_cache_config.group_pool`，即 Qwen 的 FA / Mamba 池和
  Gemma 窗口布局的全局 / 滑动窗口池），延后释放的条目保存 (block pool, block) 对，顺序为
  `pop_blocks_for_free` 的顺序（`KVCacheManager.pop_blocks_for_free_by_pool`、coordinator），drain 在同一个
  fence 之后把每个 block 还给它自己的池，顺序与共享池相同：条目倒序，每个池一次 `free_blocks`
  调用。共享池的路径不变。runtime 的 adapter 拥有的请求（每个生成请求）
  仍然立即释放（`_free_request_blocks`）：它们的 fence 是经过认证的终态 drain，而且本 runtime 中 connector
  是空闲的（没有加载会与写入竞争）。
- **测试**（`test_deferred_free_pools.py`）：从源码执行真实的 overlay `Scheduler.__init__`（KV
  cache 模块用 overlay 自己的，vLLM 的其余部分是 stub）：开 `defer_block_free` 时在 Qwen
  解耦布局、Gemma 窗口布局和共享池上的构造；perf-async-g1 计划中每个服务的
  构造路径（serve.prepare 的启动、overlay planner 的 KV cache 配置、异步服务的两个批次、
  connector 的角色）；ea18256 的 scheduler 拒绝它（取自 Git）；延后的释放在 fence 之前
  不归还任何 block，每个 block 恰好一次地还给它自己的池且顺序与共享池相同，FIFO fence，与立即释放
  相同的池，adapter 拥有的请求直接绕过；共享池与 ea18256 冻结的方法逐字节相同。
  test_kv_pools 和 test_gemma_kv_pools 随之更新。
- **异步路径上构造时的其他检查**（读过代码，没有在 GPU 上运行）：vLLM 自己的配置检查（UniProc
  支持异步调度；在 GPU 上通过了）、两个批次的 batch queue（`EngineCore`、`startup_runtime`
  `StartupTicket.quiescent`、`async_sched.attach_engine`）、要求当且仅当固定时才异步的启动门控
  （`bootstrap_contract`、`policy_runtime/bootstrap`、`native_worker_registration`、`positions.runtime_mode`）、
  worker 的 `async_sched.attach_worker`（`runner.use_async_scheduling`），以及快速路径 2 的 schedule 包装。这些
  都不拒绝解耦布局。还有一个直接拒绝异步：step profiler（`REMOTE_RA_STEP_PROFILE=1`、
  `remote_ra_step_profile.py`："needs the plain EngineCoreProc loop"），所以 GPU 检查不能设置它。

## 9. async-gemma：Gemma 4 profile，以及不带 worker RPC 的排空

**Gemma 4 profile**（设计见 [GEMMA.zh-CN.md](GEMMA.zh-CN.md)）。`serve.py prepare_gemma4` 和
`async_sched.check_config` 接受 `--async-scheduling`，固定项和拒绝条件与 Qwen 相同（需要 `--decode-fastpath2`；
拒绝 `--gdn-device-handoff` 和 `--phase1-in-step`，该 profile 本来就拒绝它们）；`window_worker` 检查一次
worker 记录，在快速路径之后、generation guard 之前挂上 `async_sched.attach_worker`，并
回显该记录。窗口布局没有 Gated-DeltaNet 状态：`void_rows(..., sliding_groups)` 在这里拒绝 GDN 状态
索引，并在每层的 metadata 和 forward-context mapping 中，以及在
全局组和每个滑动窗口组的 mapping 中（`owner.async_sliding`，即窗口 owner 的滑动窗口组，
由 `attach_worker` 记录）把作废 token 的 slot 写成 -1。窗口布局的其他部分都不变：已停止请求在
step N+1 的推测分配（C' % P == 0 时为 C' 新分配一个 block，
(C' - W + 1) % P == 0 时提前一步释放窗口最前面的 block）位于 `kept_blocks(C')` 之外，即提交固定、worker 的 carry 所指明的窗口（取自
step N 的行，因为它暂存的更新从不发出），并位于该请求的窗口 slot 和回复预留之内。

**作废请求持有的 claim（两个 profile 上都已修复）。** vLLM 0.24 的 `BlockPool.free_blocks` 把释放的
未哈希 block 放到空闲队列的队首（`prepend_n`；本 runtime 的分配从不缓存，所以没有 block
被哈希）。当已停止请求在 step N+1 的推测分配释放了它的一个 block 时（在 Gemma 上是滑动
窗口最前面的 block，每 16 次停止有一次；在 Qwen 上是 align 模式下的 Mamba block，很少见），同一次 schedule 中
下一个被分配的请求会立刻拿到它。已停止的请求在 N+1 中被作废，所以 worker 观察不到它，它的
claim 一直保留到它在该迭代稍后完成为止：另一个请求的 claim 遇到了一个外来的 claim，step
失败（`allocation owned by another generation`）。现在 `RunnerAllocationOwner._claim_full` 会转移这样的 claim：其
持有者在本 step 中的行已作废（`_voided_holder`）；其他所有外来 claim 仍像以前一样拒绝。作废的
请求再也不会碰这个 block（它的行什么也不写；它的完成只读取 vLLM 仍为它保存的内容）。
测试：`test_async_gemma.test_a_freed_window_block_taken_in_the_same_schedule`（确定性的；没有这个转移时
被拒绝）；之前一次带 Goal 线程的 CPU 运行随机遇到过它。

**不带 worker RPC 的排空（两个 profile 上都已修复）。** pass 边界会排空在途批次：引擎
更新 step N（它的停止结束，它的 token 被追加）。worker 已发出的 step N 由下一次 worker
RPC（分配更新除外）收尾，admission 或完成总会发出这样的 RPC；而 CAPACITY_WAIT 拒绝、
utility 请求、引擎循环的唤醒或另一个请求的取消都不发出。这时 step N+1 的 resolve
遇到仍处于已发出状态的 N，在已经持有 N 的 token 的请求上再次判定停止：这个 token 被计了两次，
一个只剩一个 token 的继续请求被预测为停止，它在 N+1 中的行被作废，`EngineLink.check`
让该 step 失败（`a predicted stop did not stop`），即引擎停止了：直接失败，而不是悄悄出错。在 GPU
紧张时，被拒绝的 admission 每次重试都会排空，开 `--goal-overlap` 时每次 Goal 线程的唤醒也会。
现在 `EngineLink.applied(rid, cursor)` 区分这两种情况：N 更新之前，vLLM 的请求持有的 token
直到 worker 的 cursor（`num_tokens == cursor`），更新之后多一个（`cursor + 1`）；已应用的 token 不再
判定（它们已停止的请求在 N+1 被调度之前就结束了），其他任何计数都直接失败。测试：Gemma
在 selective 池上带空闲释放的压力运行（`test_async_gemma`，正是它发现了这个问题）及其变异；在 Qwen 上
是 `test_async_sched.test_a_drain_with_no_worker_rpc_before_the_next_step`（每个 pass 都排空：每一轮都与
同步服务相同，不需要作废；没有这个检查时运行会直接失败）。

**测试**（`test_async_gemma.py`，CPU）：mswea-xml、chat-text 和 kv-session 上的固定项和 launcher（带
`--goal-overlap`、CUDA graph、4,096 token 的 forward），loader，前端对解析出的开关的检查和门控，
拒绝条件，窗口 worker 的记录、挂接顺序和回显；窗口 metadata 上的作废（eager 和持久
缓冲区）及其拒绝条件；`EngineLink.applied`；在窗口引擎和 worker 的 fake 上按 vLLM 的
异步顺序端到端运行，对比开同样快速路径的同步服务：b = 1 的 mswea-xml（两种推测
窗口步都覆盖到）和 b > 1、压力下的空闲释放、`--goal-overlap` 加 `--admission-overlap` 4 和 1、
kv-session（MemAgent lane、Gemma 的停止 id），以及经过 API server 的 chat-text，它的引擎循环运行
`core.py` 自己的输入循环和排空（包括一个 vLLM 不越过调度的长度停止）。变异：漏掉
作废、作废时漏掉一个滑动窗口组、漏掉一个停止、错判一个停止、去掉 applied 检查。在窗口布局上，
放宽的比较器还对基于 block id 的摘要改名（窗口 carry 的 `digest`、它回显的 `boundary`、阶段
1 的 `state_digest`），并用 (block, incarnation) 对来命名释放的 slot：一个已离开某请求窗口的滑动窗口 block
可能以新的 incarnation 回到该请求，取决于空闲队列的顺序（H1）。

## 10. async-fix2：阶段 1 的 pass 移出 runner 批次的请求（GPU 作业 3519531）

- **什么失败了。** 在 perf-async-g1 g1 r2（作业 3519531，runtime `4a498dc`）中，每个同步服务都通过；异步
  服务 fea、fga、tea 和 tga 在 97 个请求中的 41 个之后、b4 单元的第二轮开始时退出，xea 在 12 个中的 8 个
  之后、x4 的第二轮开始时退出；sea（只有 b = 1）和 stock vLLM 的异步服务通过。每个 b = 1 请求都与同步相同（fe、fg、te 和 tg
  为 37 个中 37 个；se 和 xe 为 4 个中 4 个）；在比较过的 b = 4 请求中，fe 和 fg 有 4 个中 3 个相同，te、
  tg 和 xe 的 x4 为 4 个中 4 个（b > 1 是测量，不是一致性检查）。EngineCore 的 step 抛出 KeyError，位置在
  vLLM 的 GPU runner（`_update_states`，`gpu_model_runner.py:1426`），针对该 step 的 cached 请求，它有一个
  输出 token，且不在持久批次中。
- **原因。** admission 的阶段 1 作为单独的 pass 运行（`native_probe.NativeQueryForward._execute`）：runner
  执行一个只调度该 probe 的 SchedulerOutput，`_update_states` 把它没有
  调度的每个请求都移出持久批次（`gpu_model_runner.py:1211-1217`）；probe 的清理（`_cleanup_request`，一个
  空输出）也一样。vLLM 的 scheduler 仍把这些请求算作在上一个 step 中已调度
  （`prev_step_scheduled_req_ids`，`scheduler.py:1202-1204`），所以它的下一个 step 不为它们发送 `all_token_ids`
  （`scheduler.py:1439-1440`）。同步时，runner 用它自己缓存的输出 id（即
  真实的 token）把这样的请求加回来。在异步调度下这些 id 是 -1 占位符（`_bookkeeping_sync`），所以一个有输出
  token 的请求只能从 `all_token_ids` 重建（`gpu_model_runner.py:1423-1427`）：KeyError。在 b = 1 下，admission
  期间什么都没有在运行，而且 session 的第一轮没有阶段 1；在 b4 的第二轮，第一个继续的轮次被
  调度并采样，而下一个 admission 的阶段 1（在排空之后）把它移出了批次。
- **修复。** 开 `--async-scheduling` 时，probe driver 运行它的 forward 和清理时经过
  `NativeQueryForward._runner_pass`，它把该 pass 移出批次的请求报告给 engine link；
  `EngineLink.unbatched` 把它们从 `prev_step_scheduled_req_ids` 中去掉（就像 vLLM 的 `reset_prefix_cache` 在 flush
  批次时清空这个集合），于是下一个 step 会发送它们的 token id，即 scheduler 自己的。它要求没有批次
  在途（`core.batch_queue` 为空：pass 边界的排空已更新每个已调度的 step，所以没有留下输出占位符）；
  否则这些 id 会少一个 token，所以直接失败。不开该开关时没有任何变化（不报告，
  不检查）。批次不跨 pass 保留：vLLM 用持久批次的每一行构建一个 step 的输入，
  probe 必须单独运行（`probe mixed with generation batch`），而移出再加回是 vLLM 自己的路径（
  某个 step 中没有调度的请求、preemption、`reset_prefix_cache`），也就是同步服务逐 token 一致地跑过的那条路径。
  它覆盖两个 profile（Gemma 窗口 worker 使用同一个 driver）、阶段 1、它的分块 forward、Goal
  forward（`--goal-overlap`）以及它们的 drain。
- **测试**（`test_async_probe_batch.py`，CPU，不需要 torch）：真实的 overlay `_update_states` 和 `CachedRequestState`、
  真实的 Scheduler（两个批次，Qwen 解耦布局）及其 `_make_cached_request_data`、真实的
  `NativeQueryForward` 和 `EngineLink`，按 `step_with_batch_queue` 的顺序跑 GPU 上的那个序列（step N 调度 A
  并采样；B 的 admission 排空并运行它的阶段 1；step N+1 把 B 作为 new、A 作为 cached 调度）：A 的 token id
  被发送，runner 输入的每个 id 都等于采样的 token；4a498dc 的 native_probe（取自 Git）和去掉报告的版本
  都抛出 GPU 上的那个 KeyError（第 1426 行）；有在途批次时拒绝；同步调度不变。
  异步端到端 fake（`test_async_sched`、`test_async_gemma`、`test_async_goal_overlap`）现在模拟
  `all_token_ids`（`test_async_sched.cached_token_ids`、`require_token_ids`）：它们阶段 1 的 pass 会把运行中的
  请求移出，没有报告时 Qwen 和窗口 harness 都抛出该 KeyError。异步窗口
  harness 的完成把 runner 中已结束的行留给下一个 step 的 `finished_req_ids`，与 vLLM 的 scheduler
  发送它的方式相同：同步 fake 的捷径（一个空的带外更新）还会把每个运行中的请求移出
  批次，除了阶段 1 的 pass 之外，没有任何生产路径会这样做。
- **GPU 检查：** perf-async-g1 g1 r3（Qwen）作为作业 3532746 运行：ASG1_PARTIAL_PASS_VERIFIED，每个服务都通过，
  b = 1 的每一对都逐 token 一致；PARTIAL 只来自 harness 比较器的 `state_digest` 缺口（第
  3 节），在本地重新比较的结果为 ASG1_PASS（第 6 节）。Gemma 4 的检查（作业 3549350 和 3560784）及其判定见
  [docs/status.zh-CN.md](../status.zh-CN.md)。

## 11. async-maxtok-1112：步 N+1 不调度的请求（GPU 作业 3568311）

分支 async-maxtok-1112 `13ee618`，基于 integrate-1111 `4bf2c40`，已合并进 integrate-1112。截至
2026-11-12 仅 CPU 测试；此后它针对的情况已在 GPU 上运行（作业 3604146，见下面最后一项）。

- **出了什么问题。** perf-mab smqt（作业 3568311，runtime `ba4d1ba`：Qwen3.6-27B，thinking 开启，
  `--async-scheduling`，MemoryAgentBench kv-session；smqot、smqrt 和 smqgt 三个服务依次运行）的三个服务，都在第一次
  有回复被 `max_tokens`（4,096）截断、同时另一个请求还在解码时退出。`before_forward(N+1)` -> `resolve` 抛出
  `LiveV2Error: async resolve: a request sampled without its token`（`async_sched.py:525`，`contract.py:23`）。
  thinking 开启时，推理循环会让回复一直跑到上限。
- **原因。** resolve 收集步 N 中每个 `prompt_complete` 的已发出请求，并要求每个都有采样到的 token。vLLM 0.24.0
  的异步 scheduler 在某运行中请求于 N 的 token 是它的第 `max_tokens` 个（`scheduler.py:490-504`）或达到
  `max_model_len`（527-534、562-578）时，不在 N+1 中调度它；token 预算或抢占也会漏掉请求。随后
  `_update_states(N+1)` 把它移出持久批次（`gpu_model_runner.py:1197-1217`），`InputBatch.remove_request` 把它从
  `prev_req_id_to_index` 中删除（`gpu_input_batch.py:556-557`），所以 `sampled_tokens` 只找到其他请求。单独一个请求
  到达 `max_tokens` 是安全的：N+1 什么都不调度，guard 在任何 resolve 之前就返回。
- **修复。** resolve 的 `sampled` 只保留被 N+1 调度的已发出请求（`scheduler_output.num_scheduled_tokens`）。N 中不在
  N+1 的请求没有需要作废的行；它的停止由 `update_from_output(N)` 判定，与同步服务相同。
  - `settle()` 仍然关闭 N 的每个 lease。
  - `schedule(N+1)` 没有为它分配任何东西，所以暂存的更新（`flush`）中没有它的条目。
  - 它不进入该 step，所以它的完成不会遇到 step lease（`WorkerBridge.terminal`）。
  - 预测的停止仍是 N 中已调度请求的子集（`EngineLink.check`）。
  - 抢占会中止一个带标记的请求（`scheduler._preempt_request`），vLLM 丢弃它的 token：从这个 token 预测停止本来就是错的。
  - 对 N+1 调度的请求，检查照旧有效：其中任何一个缺少 token 仍按 fail-closed 失败。
  - docstring：模块的第 3.2 步、`sampled_tokens`、窗口布局的推测分配。重新生成了发布固定值（只有
    `async_sched.py` 改动）。
- **测试**（CPU）：
  - `test_async_sched.ResolveTests`：B 在 N+1 被跳过（`max_tokens`、`max_model_len` 或没有停止）并从
    `prev_req_id_to_index` 中删除：不报错，没有作废或预测，只有 A 进入，N 中两者的 lease 都关闭；跳过的 B 旁边
    A 的 EOS 仍被作废；N+1 调度的请求缺少 token 时仍按 fail-closed 失败。
  - 异步 harness 现在按 vLLM 的行为运行：`test_async_sched.async_harness` 和 `test_async_gemma.async_harness`
    应用 `max_tokens` 跳过，没有任何调度时按 vLLM 提前返回（空 step）；`test_stageb_worker.FakeRunner._remove`
    和 `test_async_gemma.AsyncRunner._remove` 像 `remove_request` 一样从 `prev_req_id_to_index` 删除。各轮交替以
    EOS（再被调度一次，然后作废）和 `max_tokens` 停止，所以作废数仍大于 0；多 session 的运行断言有一个
    `max_tokens` 跳过发生在一个已调度请求旁边。
  - 在 `4bf2c40` 的 `async_sched.py` 上，这些测试以 GPU 上的同一个 `LiveV2Error` 失败（`test_async_sched`、
    `test_async_gemma` 和 `test_async_goal_overlap` 中 14 个错误、1 个失败）；修复后全部通过。
- **两个 profile。** resolve 是共用的：Gemma 4 窗口布局走同一条路径（[GEMMA.zh-CN.md](GEMMA.zh-CN.md) 第 1 节）。
- **GPU 检查：** 直接的检查，即在 integrate-1112 `4ebff97` 上重跑 perf-mab smqt（作业 3584105，
  `PMAB_PASS_VERIFIED`：`observation_query`、`random_fill` 和 `goal_query`），通过了，但没有进入这个状态：每个服务
  唯一一个被 `max_tokens` 截断的回复结束时，没有别的请求在跑。此后（2026-11-15）：
  perf-mab 的计划 smqc（作业 3604146，`4ebff97`，`PMAB_PASS_VERIFIED`）专门制造这个状态。它唯一的服务是 `observation_query`，
  用 smqt 的设置跑四个 MemoryAgentBench case，同时还运行一个截断单元：一个载体请求解码一个长回复，
  同时 8 个探测请求逐个发送，每个两轮，都被它们的 `max_tokens` 截断（24 到 136 个 token，然后是 32）。
  它的 async-cut 检查（16 个中至少 12 个）在全部 16 个探测轮次中的 16 个上发现了这个状态，这是根据请求
  记录推断的（runtime 没有为此计数）；作业中的每个请求都得到了回答，没有引擎错误。其他带
  `--async-scheduling` 的 release 冒烟（perf-swe32 smoket、perf-tb16-strip smokeqt、perf-gemma-swe32 smokea 和
  perf-mab smga）运行同一个 resolve，只要有回复在另一个请求解码时达到 `max_tokens`，就会进入
  这个状态。其中 smoket、smokeqt 和 smokea 已于 2026-11-16 取消，没有运行；smga 运行了（作业 3601174，
  [docs/status.zh-CN.md](../status.zh-CN.md)）。
