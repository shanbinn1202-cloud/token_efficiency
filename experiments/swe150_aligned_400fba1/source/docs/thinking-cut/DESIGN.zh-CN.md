# 开启 thinking，方案 (a)：截掉最后一条回复并重新 prefill（设计，2026-10-30；形式 E 即 `--thinking-history strip`）

[English](DESIGN.md) | 简体中文

2026-10-30 的设计（分支 `thinking-cut`，基于 `cf2b28d`）；形式 E 已在 `main` 上（`8ca8511`，含第 1 阶段 `94888e1`；
合并为 `34f274b`），作为需显式开启的 `--thinking-history strip`。它的 GPU 运行及其判定见
[docs/status.zh-CN.md](../status.zh-CN.md) 和[历史](../history.zh-CN.md#历史)，开启 thinking 的限制见
[docs/known-limits.zh-CN.md](../known-limits.zh-CN.md)（第 1 节和 4.3 节）；GPU 检查 G1 和 G2 测得的结果见
第 6 节。下面先是形式 E 的实现情况（2026-10-30），然后是设计。

## 形式 E 的实现（2026-10-30）

形式 E（“在截断点提交”）已在分支 `thinking-strip`（基于第 1 阶段，`94888e1`）上实现，即
`serve.py --thinking-history strip`（配合 `--enable-thinking`；[docs/running.zh-CN.md 6.9 节](../running.zh-CN.md#69-开启-thinking-并丢弃-reasoning--thinking-history-strip)；规则和挂接点在
`remote_ra_live_v2/thinking_cut.py`）。GPU 检查 G1 和 G2（第 6 节）是之后运行的（结果见第 6 节）。
封存的 verifier（`perf-tb16/`）没有改动：4.2 节的规则 V1-V5 被加进了它的一份副本
`perf-tb16-strip`（不在本仓库中），由它离线核验 strip 运行。
维护者对第 8 节的决定：(1) 形式 E，而不是 L；(2) 每个服务固定 `--thinking-history strip`，
仅限 Qwen profile 上的 chat-text，在 Gemma 4、kv-session 和 mswea-xml 上拒绝（不加它时为第 1 阶段的 keep
模式）；(3) 下一轮的稳定终点为 X + 3，CANDIDATE_RANGE 不变；(4) 被丢弃的 token id 记入
证据（`truncate.dropped`）；(5) v1 中 `--gdn-device-handoff` 和 `--phase1-in-step` 与 strip 同用时拒绝；
(6) 超过一次原生 forward 的重新 prefill 使该轮失败（`PHASE1_TOO_LONG`，不变）；与不带该选项的一轮相比，
不增加任何逐请求哈希。

实现情况，对照下面各节：
- 3.1：transcript（`chat_text_protocol.ChatTextTranscript(strip=True)`）丢弃客户端发送的每个 reasoning
  字段（参照也丢弃 R），所以 `extends` 只比较 role 和 content。稳定终点是
  X + K，K = |encode('<|im_start|>assistant\n')|，即生成头的公共开头：3（固定的 Qwen3.6
  tokenizer，在 `test_thinking_cut` 中检查），字符级 tokenizer 的 fake 中为 11。开头不一致时拒绝
  该请求（HTTP 500 `STRIP_REPLY_HEAD`，不做任何改动）；固定的 BPE 从未出现过不一致（试过的每种
  content，包括最后一个 `</think>` 之后带 `\r` 的情况）。
- 3.3：没有新的 RenderAssignment 字段。plan 带有 `commit_end`（以及封存规则 `floor(X/P)`），在
  `adapter.reserve` 时设置；worker 在 reserve 时检查它等于 assignment 的 observation 终点 e。
  prepare 的 payload 不变。
- 2.3：`worker_backend.PositionedCarry`，含 `hold_carry`、`hold_phase1_carry`（在 finish 时持有，即
  阶段 1 状态的 payload 和身份 token）和 `capture_split_carry`，由
  `drain(..., positioned=True)` 提交；strip backend 拒绝其他任何完成方式。切分处的捕获也
  记为 X 处的最新状态，所以从 X 开始跨过一个 M block 的下一个 chunk 会迁移它（只有
  生成头比从 X 到下一个 M 倍数的距离更长时才可能：Qwen 的 5-token
  生成头在 1 <= N mod M <= 4 时，如 G1 中 N mod 784 = 1、3 和 4 的冷 probe；fake 的 19-token
  生成头也会如此）。
- 3.4：`EngineBridge.scheduler_split`（仅 local turn 0）在 vLLM 的对齐切分之后，与设计相同。阅读
  `vllm/v1/core/sched/scheduler.py` 的 `_mamba_block_aligned_split` 和 `vllm/v1/worker/mamba_utils.py` 的
  `preprocess_mamba` / `utils.mamba_get_block_table_tensor` 可知：结束于 X 的 chunk 使用
  X - 1 所在的对齐 block，下一个 chunk（X 在最后一个 M 倍数之前时，按 vLLM 自己的规则在该倍数处切开）
  继续使用同一个 block，与目前最后一个不完整 block 的各个 chunk 一样；CPU 侧不需要任何近似，
  状态在 GPU 上是否精确由 G1 检查。
- 4.1：`truncate.dropped.think_end` 是单 token 的 `</think>` id 在生成 id 中的下标（不存在
  或 `</think>` 不是单个 token 时为 null），不做 decode；chat-text runtime 行的 `cut` 和
  `reprefill` 在其 `thinking` 记录中；该行增加 `truncate` 计数和 `stage_b.phase2_end`。
- 测试（`test_thinking_cut.py`）：T1（模板，固定 tokenizer），T2（插入 reasoning 的 178 个 ATIF trial：
  1,684 个请求，每个 prompt 都是 stock 渲染），T3（引擎和 worker fake：在 X 处提交，tail
  none/live/candidate，单 Page、vLLM 对齐和整个 prompt 三种 chunk 下 local turn 0 的切分，阶段 2 中
  取消，截断之后重建；每个选页输入的稠密重算），T4（decode 快速
  路径、`--deferred-completion-io`、`--admission-overlap 4`、`--release-idle-pages`（压力下，配合
  goal_query）；关闭该选项时的 identity probe），变异测试。T5（verifier）没有做：verifier
  是封存的。

日期：2026-10-30。分支 `thinking-cut` @ `cf2b28d`。基于第 1 阶段（分支 `thinking-on`：服务端的 reasoning
切分、`</think>` 之后的 MarkerStop、客户端回传 R 时追加），它当时还不在代码树中（之后与形式 E 一起合并：
`94888e1`）；3.0 节列出本设计从中取用的内容。请先读：
[docs/history/thinking-on-analysis.zh-CN.md](../history/thinking-on-analysis.zh-CN.md)（10-30 的附录，第 2-3 节）。

路径相对于 `source-overlay/`，除非以顶层文件（`serve.py`、`README.md`）或
`perf-tb16/`（封存的离线 verifier，即项目工作区的 `development/exp-decouple-pages/perf-tb16/`，不在本仓库中）开头。
行号是 `cf2b28d` 的行号。简称：不带路径的模块名在 `remote_ra_live_v2/` 下，但
`adapter.py`、`ownership.py` 和 `contract.py` 在 `remote_ra_native_growing/` 下（live-v2 的
`contract.py` 写出完整路径）；`tensors.py` 是 `remote_ra_growing_worker/tensors.py`，`evidence.py` 和
`chat_text_route.py` 在 `remote_ra_http/` 下，`scheduler.py` 是 `vllm/v1/core/sched/scheduler.py`。
撰写本说明时尚未实现任何内容，本说明也没有作出任何决定；第 8 节列出这些决定（已经作出：见上面的“形式 E 的实现”）。

## 0. 概要

1. **截断点是 observation 终点 e_t，在 chat-text 上就是生成头起点 h_t。** chat-text transcript
   渲染 `tokens = stream + observed + header`，区间为 `(stable, stable + len(observed))`
   （`remote_ra_live_v2/chat_text_protocol.py:582-586`），所以在每个 local turn 上都恰好有 e_t = N_t - |H|（local
   turn 0 也是：o = e = stable_end，`remote_ra_native_growing/contract.py:248-249`）。token 在 3 个 token 之后才分歧
   （h_t + 3，在 `<|im_start|>assistant\n` 之后），但 runtime 只在 e_t 处有 GDN 状态（阶段 1 在那里结束），而且
   GDN 不做一次同时写 FA KV 的完整 forward 就无法重算。所以截断点在 e_t，生成头的 3
   个 token 重新计算。
2. **截断点处的状态已经存在**：e 处的阶段 1 GDN 状态（`Phase1State`，`remote_ra_live_v2/worker_backend.py:49-62`，
   在 `remote_ra_live_v2/native_probe.py:300-307` 捕获）。目前它在 complete drain 时被丢弃
   （`worker_backend.py:365`），carry 改在 C' 处捕获（`qwen.py:349-352`、`worker_backend.py:261-269`）。
3. **推荐形式 E，“在截断点提交”**：在 stock 默认模板下（`enable_thinking` 为 true，不加
   `preserve_thinking`；视为合作者在 Terminal-Bench 上的 Vanilla，没有对照合作者的
   仓库重新核实），最后一条 user 消息之前的每条 assistant 消息
   都渲染为 `<|im_start|>assistant\n` + C，不论客户端回传什么（模板的 assistant 分支：
   只有开 `preserve_thinking` 或在最后一个 user query 之后才渲染 `<think>` 块）。所以在固定为
   这种历史的服务上，*每个*延续请求都在 e_t 处分歧。因此完成时可以直接在 e_t 处提交
   session：阶段 1 被提交，阶段 2（生成头、R、C、EOS）是本轮内的临时数据。截断点之后
   什么都不封存，所以不需要回滚任何东西：不解封，不释放 Page，不丢弃 arena，不截断向量或索引，没有
   与空闲释放的交互，没有第二份 carry（额外锁页字节为 0，而不是每个 session 约 154 MB），admission
   几何（`turn_geometry`）不变。
4. **Local turn 0**（新槽位、重建、换绑）没有阶段 1：它的 prefill 必须让一次 forward 结束于 e_0（现有的
   `scheduler_split` hook，`remote_ra_live_v2/composition.py:249-252`），并在那里捕获 GDN 状态。
5. **重新 prefill**：下一轮的阶段 1 计算 [e_t, e_{t+1}) = `<|im_start|>assistant\n` + trim(C) + `<|im_end|>\n`
   + user 消息，在常驻集合和 live tail（第 e_t//P 页，valid_end e_t）上，从已提交的 carry
   （e_t 处的状态）开始。目前的阶段 1 是 2 + |U| 个 token（EOS、`\n`、user 消息）；它变为 |C| + 5 + |U|。
6. **备选 L，“在下一个请求时回滚”**（ANALYSIS 勾勒的设计）保留目前的完成方式，
   在请求分歧时撤销它。只有一个服务必须在同一个 session 中服务两种历史（strip 和 keep）时
   才需要它。它的代价是每个 session 第二份 carry（约 154 MB 锁页内存）、第 1 节中逐个属主的回滚，以及大约
   两倍的代码。
7. **工作量（E）**：约 14 个文件中 550-900 行 runtime 代码，800-1,200 行测试，封存的 harness 中 150-250 行
   verifier 代码；约 1-1.5 周，外加两个 ghx4 batch 作业（G1：对照开启 thinking 的 stock eager 做 token 对齐，G2：TB16
   dry run）。L：约多 700-1,200 行 runtime 代码，2.5-3.5 周。主要风险：local turn 0 在 vLLM 的 Mamba 对齐模式下
   不在 M 对齐位置的 prefill 切分（GPU 检查），以及 verifier 的返工。

## 1. 截断点以及各属主在其后持有的内容

### 1.1 Token 几何（Qwen3.6，开启 thinking，chat-text，P = 16）

生成头 H = `<|im_start|>`、`assistant`、`\n`、`<think>`、`\n`（5 个 token）。第 t 轮：prompt T_t 有 N_t 个 token，
e_t = N_t - 5，阶段 1 为 [C_t, e_t)，阶段 2 为 [e_t, C'_t)，其中 C'_t = N_t + g_t - 1。模型写出
`R \n</think>\n\n C <|im_end|>`。

| 量 | 值 |
| --- | --- |
| 截断点 X_t（第 t 轮的提交终点） | e_t = N_t - 5（生成头起点） |
| 与 T_{t+1} 的 token 分歧点 | X_t + 3（`<think>`） |
| 被丢弃的分支 | [X_t, C'_t + 1)：`<|im_start|>assistant\n<think>\n` + g_t 个生成的 id |
| 截断后的封存终点 | Q_X = floor(X_t/P)·P |
| 截断后的 tail | 第 X_t//P 页，起点 Q_X，valid_end X_t（X_t % P == 0 时没有 tail） |
| 下一轮 | C_{t+1} = X_t，next_token = T_t[X_t] = `<|im_start|>`，o_{t+1} = X_t + 3（3.2 节） |

X_t//P 之前的每个 Page 只含已提交的 token；它要么在第 t 轮的阶段 1 中关闭（其向量是
阶段 1 向量，`phase1_coordinator.py`/`plan_turn`，见 `query_selection.py:1054-1059`），要么更早关闭。从 X_t//P 起
的每个 Page 都由第 t 轮的阶段 2 产生。**截断恰好就是“撤销第 t 轮的阶段 2”**（在 local turn 0 上则是撤销
prefill 中 e_0 之后的部分）。

### 1.2 逐个属主：第 t 轮完成后目前留下什么，形式 E，形式 L

| 属主 | 目前第 t 轮之后（文件:行） | E：完成时改为提交什么 | L：第 t+1 轮 admission 时的回滚必须撤销什么 |
| --- | --- | --- | --- |
| **GDN carry**（device 状态，主机副本） | 在 C' 处从运行中的槽位捕获（`qwen.py:349-352` → `worker_backend.py:261-269`；在 `worker_backend.py:346-363` 提交）；e 处的阶段 1 状态被丢弃（`:365`） | e 处的状态成为 carry（第 2 节）；不在 C' 处捕获 | 把阶段 1 状态保留为第二份 carry（每个 session 多 154 MB 锁页内存）；截断时把它设为 `backend.carry[session]`，使 `prepare`（`allocation_owner.py:233-244`）和 `verify_carry`（`checkpoint.py:27-34`）与回滚后的镜像一致 |
| **跨截断点的 tail 页**（第 X//P 页） | C' ≥ (X//P + 1)P 时在完成时封存：arena 捕获（`allocation_owner.py:501-507`），作为常驻 Page 钉住（`allocation_owner.py:520-522`、`ownership.py:263-291`），阶段 2 向量（`qwen.py:413-417` → `adapter.py:457` → `query_selection.py:1121-1132`） | 作为 tail 提交：`owner.commit(tail=(X//P, Q_X, X))`（`ownership.py:265-269` 已接受任何 ≥ L/P 的页）；从不封存，从不捕获，没有向量 | 解封：把它的 block 从 `owner.sessions[ns]` 移到 `owner.tails[ns]`，作为 `TailPin(X//P, Q_X, X)`（`ownership.py:22-27`），设置 worker 的 tail（`allocation_owner.py:518-519` 的形式），删除它的目录项（引擎 `adapter.py:465`，worker `allocation_owner.py:542`；arena 槽位经 `PagePayload` 的 finalizer 归还，`remote_ra_growing_worker/tensors.py:321-332`），删除它的向量，重建 `page_index[sid]`（索引拒绝同一 page id 的向量发生变化，`dram_kv/page_vectors.py:245-261`），截断 `page_columns`（`fast_selection.py:90-128`）；如果该页已被空闲释放换出（`idle_release.py:283-308`），先从它的 DRAM 副本恢复到一个新 block |
| **tail block 的 KV 槽位 [X, (X//P + 1)P)** | 生成头/R 的 KV | valid_end 之后的脏槽位，允许（`ownership.py:26`）；阶段 1 从 X 开始重写 | 相同 |
| **截断后封存的 Page**（[X//P + 1, Q'/P)） | GPU pin（`resident2`）、arena 副本、向量、引擎和 worker 的目录（`adapter.py:465-467`、`allocation_owner.py:542-545`），可能还有 `s.evicted`（`adapter.py:85`） | 从不封存：它们的行 block 在 native free 时随请求释放（`ownership.py:298-318` 检查计数） | 释放 session 的 pin（`pool.free_blocks`、弹出 `block_namespaces`，同 `ownership.evict` 347-360），删除目录项 / arena 槽位 / 向量 / 索引 / 列，过滤 `s.evicted`（重新接纳时拒绝 ≥ tail 页的页，`ownership.py:372`） |
| **运行中的 GDN 槽位、阶段 2 的 Mamba block** | 请求释放时由 vLLM 释放 | 相同 | 相同 |
| **Block 池、incarnation 计数器**（`pool_incarnations.py:8-19`） | 每个 block id 单调计数 | 无：释放的 block 回到空闲队列；下一次分配计一个新的 incarnation | 无需回退（计数器是身份标识，不是状态）；空闲队列的顺序与从未封存 R 的服务不同，与任何请求之后一样 |
| **引擎的 session 镜像**（`adapter.py:66-93`，在 `:459-469` 设置） | 已封存/已计算的 token 到 Q'/C'，next_token = full[C']，tail 在 C'，resident，boundary | `computed = full[:X]`，`sealed = full[:Q_X]`，`next_token = full[X]`，tail 在 X，resident = selected ∪ [L/P, Q_X/P)，boundary = X 处的 carry | 在截断时重写这些字段（在 `reset_session` 旁新增一个 adapter 方法，`adapter.py:229-243`） |
| **Worker 的 session 镜像**（`allocation_owner.py:173-176`，在 `:542-545` 设置） | 目录 [0, Q'/P)、resident、tail、已计算/已封存终点、carry 摘要 | 在 `_complete` 中设为 X | 新增一个 worker RPC `cut` 来重写它（且必须在 `prepare` 之前运行，后者会把它与 assignment 比较） |
| **选页表**（`query_selection.py:360-390`） | `vector_catalog` = [0, Q'/P) 和 `catalog_version`（`:1121-1132`）；`page_index`、`page_columns`、`compact_cache`；`turn_ends`（`:1116`）；`previous_active` / `ever_selected`（诊断用，所有页 < n_t） | `publish_completion` 得到阶段 1 向量和 `closed_pages(S2, X)` 的阶段 2 向量（local turn > 0 时为空），封存终点 Q_X；其他不变 | 删除 ≥ X//P 的目录键，递增 `catalog_version`，重建 `page_index[sid]`，截断 `page_columns`；`turn_ends` 和诊断信息无需改动（它们的条目都 < e_t） |
| **Transcript**（`chat_text_protocol.py:617-634`） | `stream = prompt + generated + boundary` | `stream = prompt[:X]`；轮次记录保留 C、R 和被丢弃的 id（3.1 节） | 保留 stream；在 `plan` 中于最后一轮的生成头起点截断它 |
| **Goal 记录**（`dram_kv/goals/maintained.py:134-148`，由 `query_selection.py:861-929` 调用） | 下一次 admission 时由 `previous_action()` 构建（`chat_text_protocol.py:636-638`，在 `query_selection.py:951, 971, 990, 1040` 读取） | 无：渲染（以及随之而来的截断决定）先于 `begin_turn`（`private_endpoint.py:163-179` 在 `:237-240` 之前）；有第 1 阶段时上一个 action 是 C | 相同：只要截断在渲染时决定，就没有要撤销的 |
| **证据** | native 文件和 runtime 行（第 4 节） | 新字段；不回滚 | 新字段；不回滚 |

## 2. 截断点处的状态从哪里来

### 2.1 来源

| 轮次 | X = e 处 GDN 状态的来源 | 目前 | E |
| --- | --- | --- | --- |
| Local turn > 0，默认 | `Phase1State`：来自 `io.capture` 的锁页主机 payload（`worker_backend.py:183-187`、`tensors.py:513-534`），在阶段 2 的第一步恢复到阶段 2 槽位（`worker_backend.py:208-214`） | 在 drain 时丢弃 | 提升为已提交的 `CarryState`（同一个 payload 对象和身份 token，不拷贝）；跳过 C' 处的捕获（`qwen.finish` → `_final_state`） |
| Local turn > 0，`--gdn-device-handoff` | `HeldPhase1State`：没有主机字节，是 probe 槽位，直到阶段 2 在 device 上拷贝它并释放它（`worker_backend.py:173-182`、`195-207`） | 拷贝后释放 | 在捕获时还要把 probe 槽位 D2H 到该 session 空闲的 carry buffer（`_capture_carry` 在 C' 处会用的那个 buffer，`worker_backend.py:278-293`），在 `release_probe` 之前发出；complete drain 会 join 它（`:337-339`）。字节与目前 C' 处的捕获相同，只是更早 |
| Local turn 0 | 无（阶段 2 从零状态运行 [0, N)） | 不适用 | 在 e 处切分 prefill（3.4 节），并在结束于该处的 forward 末尾捕获状态（非 handoff：`io.capture`；handoff：发出到一个 carry buffer） |

local-turn > 0 的状态就是每个延续轮次的阶段 2 本来就从它开始的状态，所以它的正确性正是
目前的 runtime 已经依赖的；E 只是提交它。

### 2.2 各方案的代价

| 方案 | 锁页主机内存 | GPU | 额外计算 | 说明 |
| --- | --- | --- | --- | --- |
| **E：提交 e 处的状态**（推荐） | 0（它取代 C' 处的 carry） | 0 | 0；每轮省下 C' 处 154 MB 的捕获（开 hand-off 时同样的 D2H 移到阶段 1 结束时） | 需要每个服务固定 strip 历史 |
| L：在 C' 处的 carry 旁再保留一份回滚 carry | 每个 session 多 154 MB（`GDN_CARRY_BYTES`，`serve.py:95`；预算 `pinned_host_bytes` 按 `sessions*carries` 计算，`serve.py:156-167`，`:505`：carries + 1）。20 个 session：在 128 GiB 预算中多 3.1 GB | 0 | 0 | 另加 1.2 节的整个 L 列 |
| 重算：改为保留 C_t 处的 carry，并重跑 [C_t, e_t) | 使用 handoff 的第二个 buffer 时为 0（提交之后它仍持有前一份 carry），否则多 154 MB | 0 | 每轮多 \|obs_t\| 个 token | 截断点移到 C_t：[C_t//P, e_t//P) 的 Page（阶段 1 Page，有效）必须回滚或原地重写（于是它们的向量和 DRAM 副本可能与重算的 KV 不同）；否决 |
| 把状态保存在一个 device Mamba 槽位中 | 0 | 每个 session 多 154 MB（Mamba 池中每个 session 第三个槽位，`kv_layout.py`，K = 2S + 1） | 0 | 16-20 个 session 时 2.5-3 GB，而 KV 为 25-32 GiB；否决 |
| 在最后一条回复的生成头起点保存 carry | = E：在 chat-text 上生成头起点就是 e | | | |

### 2.3 带位置的 carry（可复用的机制）

目前，已提交的 carry 按构造就是请求游标处的运行槽位（`drain` 要求
`latest[0] == computed_end`，`worker_backend.py:347-355`）。E 引入一个**带位置的 carry 候选**：位于指定终点
X ≤ 游标、有指定来源的 GDN 状态，完成时提交它而不是槽位。

- Worker（`worker_backend.py`）：`hold_candidate(identity, end, source, payload | issued buffer)`；来源 `phase1`
  （probe 状态，end == lease 终点 == assignment 的 e），`split`（在 local
  turn 0 时于 forward 终点 == X 处的捕获）。`drain(complete, computed_end=X)` 提交具有该终点、谱系已证明
  （`phase1_restored` / `zero_verified`，`:355-356`）的候选，而不是 `latest`。
- 引擎（`adapter.complete`，`adapter.py:395-480`）：plan 指定 `commit_end` X；完成时由 X 而不是 C' 推导封存、
  tail、resident 和向量；回执同时带有 `computed_end`（C'，实际算到的位置）
  和 carry 的 `computed_end`（X）。

异步调度（[docs/async-sched/DESIGN.zh-CN.md](../async-sched/DESIGN.zh-CN.md)）可以复用这两部分：
- 方案 S 的 D1(b)（在第 N+1 步之前对某行槽位做的快照）是第三种来源 `snapshot`，终点为 C'。它的
  “完成时截断领先一个 token 的 worker”（H4）就是同一条“在 X < 游标处提交”的路径。
- 方案 C 可以把同一条路径用作 launch 之后才决定的停止的回退（而不是让请求失败）。
  按已实现的形式，它没有这样做：async C 没有预测到的停止会直接失败（`async_sched.py`）。
- 在 strip 服务上，异步完全不需要 GDN void：已提交的 carry 和每个已提交的 Page 都在回复之前，
  所以 C' 处多出的那一行只写阶段 2 的临时数据（D1、D3 和 D4 碰不到已提交的状态；H1、H2、H6 和 H7
  仍然存在）。按已实现的形式，async C 在 strip 服务上也保留 void；带位置的 carry 不在它的作用范围内
  （`async_sched.py`）。

## 3. 重新 prefill

### 3.0 第 1 阶段提供什么（假定的接口）

- chat-text 上开启 thinking 的生成头 `<|im_start|>assistant\n<think>\n`（5 个 token），以及适用于 thinking 的回复上限
  （第 1 阶段的 `MAX_TOKENS_CHOICES` 决定）。
- 把生成的 id 切分为 R 和 C（第一个 `</think>` 处），响应的 `content` = C、`reasoning_content`
  = R（目前两个 reasoning 字段都被置空，`remote_ra_http/chat_text_route.py:178-180`）。
- 一条 transcript 轮次记录，其 `text` 为 C（所以 `previous_action()` 和 `extends` 只看到 C），并保留 R。
- 第 1 阶段的 keep 路径（客户端回传 R 时追加原样的 id，`preserve_thinking`）保持不变；本
  设计增加第二种固定的历史模式。

### 3.1 strip 模式下的 transcript（`chat_text_protocol.py`，前端和引擎）

启动时固定的模式 `history = 'strip'`，参照 kwargs 为 `{enable_thinking: true}`（stock 默认：不加
`preserve_thinking`），在 Gemma 语法上拒绝。

- `commit(prompt, generated)`（`:617-634`）：`X = len(prompt) - len(header)`；要求 `prompt[X:] == header`；
  `stream = prompt[:X]`；追加 `dict(observation, text=C, reasoning=R, ids=prompt[X:] + generated, cut=X)`。不
  追加 boundary：模板的 `<|im_end|>\n` 属于下一个片段。
- `extends(messages)`（`:550-555`）：规则不变（历史相同，然后是 assistant，然后是一条 user 消息；
  `content.strip() == text.strip()`，其中 text = C）。
- `plan(messages)`，延续轮次（`:583-587`）：`A = encode('<|im_start|>assistant\n' + strip_content(c) +
  '<|im_end|>\n')`，其中 `c` 是客户端回传的 content，`strip_content` 是模板自身的变换（`|trim`，
  然后取最后一个 `</think>` 之后的部分，去掉开头的 `\n`）；对上一轮的 ids 要求 `A[:3] == ids[:3]`（一次
  3-token 的比较，不做哈希）；`tokens = stream + A + U(obs) + header`；`stable = len(stream) + 3`；
  `span = (stable, len(stream) + len(A) + len(U))`。每个片段都以特殊 token 开始和结束，所以单独编码
  的结果与在整段渲染中完全相同。
- Local turn 0 和重建：用 strip kwargs 调用 `render_chat`；过往回复只渲染为 C，与
  stream 一致。
- `check_template`（`:225-245`）增加 strip 变体：一段 probe 对话，其回复 `R</think>\n\n{"a": 1}`
  必须渲染为 `user_piece + A('{"a": 1}') + user_piece + header`（仅在启动时）。
- 重新 prefill 使用客户端的回传，即恰好是 stock 的输入；回传不同时 `extends` 不成立，session 重建，
  与目前一样。

### 3.2 Admission：几何不变

`turn_geometry`（`contract.py:235-274`）原样成立：T[:X] == computed，T[X] == next_token（`<|im_start|>`），
C = X < o = X + 3 < e < N，n = floor((X + 3)/P) ∈ [Q_X/P, Q_X/P + 1]（P ≥ 3），tail 标记由
X % P 决定：为 0 时没有 tail，1..P-4 时 tail live，P-3..P-1 时 tail candidate（由生成头的 3 个 token 写满）。阶段 1（`phase1_coordinator.py:135-289`）在常驻
Page 和 live tail 上运行 [X, e_{t+1})，已提交的 carry 恢复到 probe 槽位（`native_probe.py:279-288`），与
任何一轮相同。它的向量覆盖它关闭的 Page，包括写满时的第 X//P 页。

**为什么是 o = X + 3 而不是 C 之后。** 把稳定终点放在重新编码的回复之后，会使 n = floor(o/P) 在
每当 |C| + 5 > P 时越过 Q/P + 1，而 `CANDIDATE_RANGE` 拒绝这种情况（`contract.py:264`、`208-210`）：C 的 Page
会成为恰好由这次阶段 1 计算的候选页，在释放之前没有 CPU 副本（目前只有唯一的
tail candidate 在阶段 1 结束时被捕获，`native_probe.py:308-312`、`allocation_owner.py:508-513`）。o = X + 3 时，
重新编码的 C 位于 observation 区间内：它在第 t+1 轮的阶段 2 中被稠密地关注（run Page），
并在第 t+2 轮成为候选。另一种做法（多个阶段 1 候选）是决定 D3。

**长度。** 阶段 1 是一次原生 forward（`PHASE1_TOO_LONG`，`phase1_coordinator.py:154`；Qwen 上为 16,384 个 token）。
它每轮增长 |C| + 3 个 token。[X, e) 超出这次 forward 的一轮，要么像目前过长的
observation 那样失败，要么改为计划一次重建（决定 D6）。

### 3.3 strip 模式下的完成

引擎（`adapter.complete`，`adapter.py:395-480`）和 worker（`allocation_owner._complete`，`:478-557`），其中
X = `plan["commit_end"]` = e（在 `adapter.reserve` 时设置，`adapter.py:300-319`，并写入 worker 的 assignment，
`remote_ra_live_v2/contract.py:216-235`，一个新字段），Q_X = floor(X/P)·P：

- 封存 [L/P, Q_X/P)，以及与目前一样的阶段 1 tail candidate；目录变为 [0, Q_X/P)；
- tail = 第 X//P 页，valid_end X，它的 block 取自该行（`block(p)`，`allocation_owner.py:495-500`）；
- resident = selected ∪ [L/P, Q_X/P)；`owner.commit(keep_pages=resident, tail=(X//P, Q_X, X))`；
- 向量：阶段 2 发布 `closed_pages(S2, X)`（local turn > 0 时为空；local turn 0 时为 [0, Q_X/P)），
  lifecycle 跳过 ≥ X//P 的 Page 的压缩（`qwen.py:399-417`；开 `--decode-fastpath2` 时在
  `finish_step` 内），因此 `publish_pages`（`qwen.py:436-445`）检查的是这个集合；
- carry = X 处带位置的候选（2.3 节）；镜像同 1.2 节；
- 回执报告 `computed_end` C' 和 `carry.computed_end` X；引擎像目前一样检查
  `C' == N + g - 1`，并对照 X 检查封存/tail/向量集合。

### 3.4 Local turn 0

- `EngineBridge.scheduler_split`（`composition.py:249-252`，在 vLLM 的 Mamba 对齐切分之后调用，
  `vllm/v1/core/sched/scheduler.py:556-560` 和 `:974`）返回 `X - start`，条件是 `start < X < start + count`，且请求为
  local turn 0 的 strip 模式请求（X = o = e = N - 5）。
- `QwenAlignLifecycle._after_state`（`qwen.py:337-347`）在 `lease.end == X` 时捕获候选。
- 完成时按 3.3 节提交，L = 0，S2 = 0。
- vLLM 的对齐模式把最后一个 M 倍数之前的 prefill chunk 切成 M 的倍数（`scheduler.py:383-401`）；
  结束于 X 的 chunk 不结束于 M 的倍数。decode 步本来就结束在 block 中间，所以 runner 应该能处理，但
  这是主要的 GPU 检查（G1）。生成头跨过一个 M 边界时（1 <= N mod M <= 4），vLLM 随后在该边界处切分 [X, N)，
  依据的是它自己的规则（`scheduler.py:402-408`），这是 runtime 已经能处理的 M 对齐终点。

### 3.5 下一轮的选页看到什么

- 候选 [0, n_{t+1}) = 已提交的历史，其中每条过往回复都是 C（重新编码），恰好是 Vanilla
  prompt 的历史，切成 Page；每个 Page 的向量来自它唯一一次已提交的计算（阶段 1，或
  local-turn-0 的 prefill）。
- 轮次分桶（`turn_ends`，`query_selection.py:1116`；归属见 `selection_contract.py:274-281`）：桶 k+1 =
  [X_{k-1} + 3, X_k + 3) = C_{k-1} 的片段 + obs_k + 3 个生成头 token，所以一条回复与*后面的*
  observation 归为一组（关闭 thinking 时：与前面的一个）。recent 集合（最后两个桶，P = 16 时 64 页，
  `selection_contract.py:82-83`）因此覆盖 C_{t-1}、obs_t、obs_{t-1}，而 C_t 被稠密关注。
- 必保留页（ceil(initial_length/P)，`selection_contract.py:268-271`）、冷启动（local turn 0，≤ 8,192 token）和
  Goal 输入都不变。`random_fill` 以请求序号为键（不变）；`random50` 以 prompt 的
  block 哈希为键（取决于内容，`private_endpoint.py:158-159`）。

## 4. 证据与 verifier

### 4.1 记录

- native 文件（`remote_ra_http/evidence.py:198-235`，在引擎线程上构建，开
  `--deferred-completion-io` 时也是如此，`remote_ra_live_v2/deferred_completion.py:465-496`）：
  - `plan.commit_end` = X，`plan.seal_at_completion.rule` = `floor(X/P)`；
  - `truncate = {cut: X, dropped: {start: X, end: C' + 1, header_ids: T[X:N], generated_ids: G, think_end: k | null}}`。
    生成的 id 必须记录在这里：目前 verifier 从下一个 prompt 中读取它们
    （`perf-tb16/tb_checks.py:1167-1175`），而下一个 prompt 不再包含它们；
  - 对延续轮次，`reprefill = {start: C, piece_end: C + |A|}`（id 在 `prompt_token_ids` 中）。
- runtime 行（`evidence.py:456-464`）：`stage_b.computed_end` 仍取自 complete frame 的 carry（现在是 X，
  即提交的终点），另加 `phase2_end` = 该 frame 的 `computed_end`（C'）和 `truncate` 计数。
- chat-text 行（`chat_text_route.py:346-351`）：`cut`，以及 reasoning 和 content 的 token 数。

### 4.2 Verifier 规则（`perf-tb16/`）

规则“每个 token 恰好计算一次”（`perf-tb16/smoke_checks.py:808-815`、`README.md:997`）变为
**“每个流版本只计一次”**：第 t 轮之后提交的流是 v_t = T_t[:X_t]；v_t 的每个位置
在其谱系中恰好计算一次，每一轮的阶段 2 是恰好计算一次的被丢弃分支。

- V1，谱系：`T_{t+1}[:X_t] == T_t[:X_t]`；`C_{t+1} == X_t`，`Q_{t+1} == Q_{X_t}`（`check_stage_geometry`，
  `perf-tb16/tb_checks.py:1219-1226`，仍读取 `computed_end` / `sealed_end`，现在它们是提交的终点）。
  V2-V4 以及下面 local turn 0 规则中只写 `:NNNN` 的引用都在同一个文件中。
- V2，已提交的计算：阶段 1 == [X_t, e_{t+1})，且 `e_{t+1} == N_{t+1} - 5 == X_{t+1}`；local turn 0 时
  prefill 先是 [0, X_0)，然后是 [X_0, C'_0)（“一次普通 prefill”规则，`:1214-1217`，加上这次切分）。
- V3，被丢弃的分支：`truncate.dropped == [X_t, C'_t + 1)`，其 id == `T_t[X_t:N_t] + G_t`，
  `|G_t| == completion_tokens`，且 `phase2_end == N + g - 1`（取代 `:1230-1234`）；EOS / length 规则与目前相同
  （`:1172-1176`）。
- V4，重新 prefill（取代 `check_chat_stream` 的追加规则，`:1165-1185`）：`stable == X_t + 3`；
  `T_{t+1}[X_t:X_t+3] == T_t[X_t:X_t+3]`；`T_{t+1}[X_t:X_t+|A|] == encode(A(content(G_t)))`（用固定的
  tokenizer）；然后是一条 user 消息和生成头。
- V5，计数：每一轮计算的 token 是阶段 1 中的 (|v_t| - |v_{t-1}|)（或 X_0 之前的 prefill）加上
  阶段 2 中的 (|dropped_t| - 1)（vLLM 从不计算最后一个采样的 token）。
- Local turn 0 的渲染：过往回复不带任何 thought 块（strip），而不是“带空的 thought 块”
  （`:1162-1163`）。
- `join_tb16.py` 比较 token 数；completion token 不变，prompt token 减少。
- `test_chat_text_replay.py:84-119`：参照是 strip 渲染。

## 5. 与其他功能的交互

| 功能 | E 的影响 | 改动 |
| --- | --- | --- |
| `--decode-fastpath` 1-3 | decode 步不变；逐请求的稳态缓存随请求一起消失（`allocation_owner.py:547`） | 跳过阶段 2 Page 的压缩（3.3）；别无其他 |
| `--cuda-graphs full` | decode 中没有新的 device op；local-turn-0 的捕获跟在一次 eager prefill forward 之后；handoff 的捕获在 eager probe 中运行 | 无 |
| `--admission-overlap K`、挂起的 Goal | Admission 不变；阶段 1 长 \|C\| + 3 个 token，所以唯一的阶段 1 lease 被持有得稍久一些 | 无 |
| `--deferred-completion-io` | `truncate` 和被丢弃的 id 在引擎线程上于 `terminal_value` 中构建；写入线程及其 join 点（`deferred_completion.py:575-623`、`746-767`）不变 | 每个 native 文件约多 16-40 KB JSON |
| `--release-idle-pages` | 截断点之后的 Page 从不被钉住；tail 在换出时始终钉住（`ownership.py:347-360`） | 无（L 需要过滤 `s.evicted`，并恢复被换出的截断页） |
| 容量门控（`capacity.py:186-222`） | 读取已提交（已截断）的 session；阶段 1 约需多 (\|C\| + 3)/P 个新 Page，`fresh = ceil(N/P) - base` 已经计入；`:198` 处的重建判断就是 transcript 不变的 `extends` | 除第 1 阶段的回复上限外无改动；锁页预算不变 |
| `--gdn-device-handoff` | 需要在阶段 1 结束时发出的 D2H（2.1） | 建议 v1 中与 strip 模式同用时拒绝（不在 benchmark 配置中） |
| `--phase1-in-step` | 经 arm/collect 进行同样的捕获调用（`native_probe.py:195-224`） | 建议 v1 中拒绝（与 async C 一样） |
| `--goal-overlap`（分支 goal-overlap） | TurnContext 在同一处构建，读取 `previous_action()` = C；更长的阶段 1 让 overlap 有更多时间可以掩盖 | 无 |
| vLLM rewriter（分支 rewriter-vllm） | 独立的引擎进程，它的 prompt 为 `enable_thinking=False` | 无 |
| Async C（分支 async-sched） | 2.3 节 | 复用带位置的 carry（计划如此；按已实现的形式它没有复用，`async_sched.py`） |
| 重建、换绑、`close_session`（`private_endpoint.py:188-206`、`session_pool.py:112`） | 不变；下一个 local turn 0 在 e 处切分 | 无 |
| 阶段 1 或 2 中的取消 / 中止 | 不变：已提交的 carry（在 X_{t-1} 处）和镜像保留；session 被标记为 poisoned（`allocation_owner.py:559-567`） | 无 |
| Gemma 4 | Window 布局：跨轮次的状态是 C' 之后钉住的窗口（`adapter.py:461-464`）；开启 thinking 会改变必保留前缀 | 拒绝 strip 模式；Gemma 保持关闭 thinking（2026-11-09 起不再适用：[docs/thinking-gemma/DESIGN.zh-CN.md](../thinking-gemma/DESIGN.zh-CN.md)，保留切点处的窗口） |
| kv-session（MemoryAgentBench）、XML 协议（SWE-bench） | 它们的 baseline 保留 R（ANALYSIS 的附录） | 拒绝 strip 模式；使用第 1 阶段的追加 |

## 6. 测试

CPU（fake，固定 tokenizer）：
- T1 模板：`A(c)` 对随机 content 都等于模板的历史渲染（空白、Unicode、C 中的 `</think>`、
  空的 C、在 `max_tokens` 处被截断且没有 `</think>` 的回复）；`A[:3]` 等于生成头的前 3 个 id；
  strip 的 `check_template`。
- T2 transcript 回放：`test_chat_text_replay` 的 178 个 TB ATIF trial，在每条回复中插入合成的 reasoning：
  服务过的每个 prompt 都等于 stock 的 strip 渲染；每次提交之后 stream 等于 `prompt[:X]`；
  摘要交接从文本重建。
- T3 引擎和 worker fake（stage-B / chat-text 端到端 harness）：P = 16 和旧的 784 下在 X 处提交，X % P == 0，
  tail candidate（X % P ≥ P - 3），tail live；目录、resident、tail、pin、block namespace、释放的行 block，worker
  镜像 == 引擎镜像；carry 的身份 token == 阶段 1 状态的身份 token；不发出 C' 处的捕获；local-turn-0 的
  切分和捕获；阶段 2 中取消；截断后重建；容量门控的计数。
- T4 选项矩阵：快速路径 1-3、`--deferred-completion-io`、`--admission-overlap 16`、`--release-idle-pages`：
  plan、token 和证据都等于不开这些选项的 fake；identity probe：关闭 strip 时，启动记录和每个
  证据文件都与 `cf2b28d` 逐字节相同。
- T5 在合成证据上的 verifier 单元测试：正例运行，以及反例（跳过一次截断，一个位置在同一版本中
  计算了两次，错误的重新 prefill 片段，被丢弃的 id 与 completion token 不一致）。

GPU（ghx4 batch，作业邮件 BEGIN,END,FAIL；在 CPU 测试套件和 release-check 回执之后）：
- G1 token 对齐（一个 session，b = 1，`full`，eager 和 `--cuda-graphs full`，Qwen）：一个脚本化的、形如 Terminus 的
  客户端，observation 固定；向使用默认模板且 `enable_thinking` 为 true 的 stock vLLM 0.24.0 eager 发送
  相同的消息。每个 prompt 的 id 必须相同；回复逐 token 比较（报告第一处
  分歧和相同的比例，与 stock 自身的波动对照）。还在我们的服务上：把截断后的一轮与
  同一 prompt 的强制重建比较（检查 X 处的 carry 和 tail）。包括
  N mod 784 ≥ 5 和 < 5 的冷 prompt，以及一条在上限处被截断的回复。
- G2 TB16 dry run，开启 thinking，strip 模式，每个策略一次（`full`、`observation_query`、`random_fill`，以及 `goal_query`
  配合 `--admission-overlap 16` 和 rewriter batch），使用 benchmark 选项，另加开启 thinking 的 stock 对照；
  用新规则离线核验。

它们测得的结果（每个 runtime 服务都以 `--enable-thinking --thinking-history strip` 在 runtime `8ca8511` 上运行；取自各作业的离线核验，未审计）：
- **G1，作业 3493649。** 每个服务上的 prompt id 都与 stock 相同（20 个中 20 个）。截断之后的一轮与它的强制重建
  做了比较（每个服务 12 对）。读数 A1 和 A2 未达到（这是读数，不是结论）。冷 probe 包括
  N mod 784 = 1、3 和 4，此时生成头跨过一个 M 的倍数。
- **G1b，作业 3500646**（决策 GRAPH_NUMERICS）：G1 的两处早期分歧（读数 A1）是 CUDA graph 下的近似平局
  （差值 0.125 和 0）。
- **G2，作业 3499759**（计划 strip25，即那次 Terminal-Bench dry run）：stripobs262 解决了 16 个任务中的 8 个，stripgoal262
  解决了 16 个中的 6 个，开启 thinking 的 stock 对照 stockthink262 解决了 16 个中的 8 个；用 `perf-tb16-strip` 中的规则 V1-V5 核验。

## 7. 工作量与风险

| 部分（E） | 文件 | 行数 |
| --- | --- | --- |
| Strip transcript：commit、plan、片段、启动检查 | `chat_text_protocol.py`、`runtime_renderer.py` | 130-200 |
| 选项、pin、拒绝、启动记录 | `serve.py`、`remote_ra_http/config.py`、`chat_text_route.py` | 80-130 |
| plan 和 assignment 中的 `commit_end`；在 X 处完成 | `adapter.py`、`contract.py`（live v2）、`allocation_owner.py` | 120-200 |
| 带位置的 carry、drain 谱系、跳过 C' 处的捕获、跳过压缩、local-turn-0 的捕获 | `worker_backend.py`、`checkpoint.py`、`qwen.py` | 120-200 |
| local-turn-0 的切分 | `composition.py` | 15-30 |
| 选页发布、容量规则 | `query_selection.py`、`capacity.py` | 10-30 |
| 证据字段 | `evidence.py`、`chat_text_route.py` | 50-100 |
| **Runtime 合计** | 约 14 个文件 | **550-900** |
| 测试（T1-T5） | 新的 `test_thinking_cut*.py`、证据测试 | 800-1,200 |
| Verifier（封存的 harness） | `perf-tb16/tb_checks.py`、`smoke_checks.py`、`join_tb16.py` | 150-250 |

约 1-1.5 周的 CPU 工作，外加 G1 和 G2。形式 L 约多出 700-1,200 行 runtime 代码和 500-800 行测试（即
1.2 节的 L 列、一个 `cut` RPC、第二份 carry 及其锁页预算）：2.5-3.5 周，与 ANALYSIS 估计的
1.5-3 周一致。

风险：
1. vLLM 的 Mamba 对齐模式下，local-turn-0 在非 M 对齐位置的切分（中等；G1）。如果失败，local
   turn 0 需要另一个捕获点（例如 e 放得进一次 forward 时，对 [0, e) 做一次 probe prefill）。
2. 封存的 harness 中 verifier 的返工（工作量中等，技术风险低）。
3. 重新编码片段的模板精确性：trim、C 中的 `</think>`、因长度被截断的回复（有 T1/T2 时风险低；
   每轮的 3-token 检查失败即拒绝）。
4. 选页语义的变化（3.5）：没有正确性风险，但 strip 运行只能与 strip 运行比较。
5. 阶段 1 长 |C| + 3 个 token：更容易遇到 `PHASE1_TOO_LONG`，每轮多几十 ms；被每轮省下的
   C' carry D2H（154 MB）和 R 的 Page 的 arena D2H（约 |R|/16 MiB）抵消。
6. 第 1 阶段尚未合并；它的轮次记录和生成头接口可能会变（之后与形式 E 一起合并，`94888e1`）。

## 8. 请维护者决定的事项

1. **E（在截断点提交，推荐）还是 L（在下一个请求时回滚）。** 只有要在同一个 session 中服务两种
   历史时才需要 L；它的代价是每个 session 约 154 MB 锁页内存和大约两倍的代码。
2. **每个服务固定历史**：`--thinking-history strip`（Terminal-Bench，chat-text，仅 Qwen；参照 = stock
   默认模板），与第 1 阶段的 keep 模式（SWE-bench、MemoryAgentBench）并存。确认 strip 在
   Gemma、kv-session 和 XML 协议上被拒绝。
3. **稳定终点在 X + 3**（推荐：几何不变；C 在下一轮被稠密关注，并与
   后面的 observation 归为一组）**还是多个阶段 1 候选**（C 作为写出它的那一轮的候选，如
   ANALYSIS 所勾勒：放宽 CANDIDATE_RANGE，阶段 1 中捕获每个候选 Page，多 300-500 行，GPU 风险更大）。
4. **证据**：在 native 文件中记录被丢弃的生成 id（推荐），还是只记摘要和计数（这样
   verifier 无法重新推出 C）；由谁修改封存的 verifier，以及 strip 运行的判定名称。
5. **v1 的拒绝**：`--gdn-device-handoff` 和 `--phase1-in-step` 与 strip 模式同用时拒绝（推荐；两者都不在
   benchmark 配置中）。
6. **过长的重新 prefill**（[X, e) 超过一次原生 forward）：像目前一样使该轮失败，还是重建。
7. **顺序**：在 vLLM 中的 rewriter 之后、async C 之前（按 10-29 的计划）；之后 async C 复用带位置的
   carry，并且在 strip 服务上不需要 GDN void。（按已实现的形式，async C 两者都没有做：`async_sched.py`。）
