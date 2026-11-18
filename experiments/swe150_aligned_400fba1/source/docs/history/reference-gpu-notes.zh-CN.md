# 服务选项：GPU 运行与移出的说明（冻结）

[English](reference-gpu-notes.md) | 简体中文

> **2026-11-16 冻结**（在 `4dcae3f`）：[docs/reference.zh-CN.md](../reference.zh-CN.md) 第 4.6 到 4.15 节中关于 GPU 运行的段落、从选项正文中删去的 GPU 运行说明，以及原来对 Gemma 4 CPU 测试的描述，由 docs-reference-1116 移到这里，除相对链接外不做改动；不再更新。下文的节号以及“见上”“见下”“本版本”等说法都指 `4dcae3f` 时的 docs/reference.zh-CN.md。当前的作业结果与判定见 [docs/status.zh-CN.md](../status.zh-CN.md)；限制见 [docs/known-limits.zh-CN.md](../known-limits.zh-CN.md)。

## 出自 4.6 节：解耦 KV 布局与 16 token 的 Page

移自 docs/reference.zh-CN.md 的 [4.6 节](../reference.zh-CN.md#46-解耦-kv-布局与-16-token-的-page)。

**`--triton-attn`。** GPU 上（作业 3461668，plan tritonc，runtime `cf2b28d`，一张 GH200；离线核验为 `SPD3_PARTIAL_PASS_VERIFIED`：每个服务都通过，只是部分轮次缺少测量窗口，即 stock 服务的窗口，以及 20K b=16 轮次中 FlashAttention 3 上 `observation_query` 的窗口，所以是部分通过）：一次与作业 3429022 相同类型的解码检查（4.11 节）让本 runtime（32 GiB、131,072 token、P = 16、CUDA graph `full`、三条解码快速路径）开 `--triton-attn` 分别运行 `full` 和 `observation_query`（8,192 token 预算），旁边是 FlashAttention 3 上同样的两个服务，以及两种后端上的 stock vLLM；两个开了该开关的服务，其 worker 都加载了 TRITON_ATTN（`vllm.v1.attention.backends.triton_attn`）。本 runtime 的稳态 token 间延迟，TRITON_ATTN 对 FlashAttention 3（轮次按 prompt 长度和同时运行的请求数 b 命名）：`full` 在 2K b=1 为 19.46 对 19.38 ms，2K b=16 为 24.78 对 24.63，20K b=16 为 30.54 对 29.82，117K b=1 为 29.85 对 21.62（stock vLLM：28.99 对 20.80）；`observation_query` 在 117K b=1 为 20.02 对 19.52 ms。在 prompt 逐字节相同的请求中，它的回复与 TRITON_ATTN 上 stock vLLM 的回复相同的有 73 个中的 27 个（FlashAttention 3 上：77 个中的 31 个）。

## 出自 4.7 节：16 token Page 的验证

移自 docs/reference.zh-CN.md 的 [4.7 节](../reference.zh-CN.md#47-16-token-page-的验证)。

下面所有运行都使用作业 3270721 的干净环境（复用）和一张 GH200；每项只跑了一次。

**GPU 冒烟测试**（runtime `8f1af6a`；合成的 Mini-SWE-Agent XML 会话，每轮 observation 6,500 token；按作业自身的结果，每个服务和每项计划检查都通过；未复核：这些作业的证据从未取回；表下的作业 3321538 跑了冒烟测试的 session 规模和池大小，但没有 3308015 的 784 token 解耦布局，也没有 `random50`）：

| 作业 | 服务 | 结果 |
| --- | --- | --- |
| 3308015 | `full`，1 个 session × 10 轮，prompt 最长约 60K token，8 GiB，三种布局：默认、784 解耦、16 解耦 | 三种布局下每一轮的 prompt token、回复和生成 token 数都相同；每轮的 prepare 和 restore H2D 字节数也相同（未复核；作业 3321538 在默认布局与 16 token Page 之间验证了同样的结果） |
| 3308414 | 16 token Page 下的 `full`、`random50`、`goal_query`、`observation_query`，1 个 session × 10 轮，8 GiB | 历史超过 8,192 token 后，query 策略选了约 40 % 的 16 token Page，选择性策略都恢复了 Page（未复核；作业 3321538 验证了同样的结果） |
| 3308482 | 4 个 session × 6 轮，24 GiB：默认布局的 `full`；16 token Page 的 `full` 和 `goal_query` | 4 个 session 成批 decode，无抢占，无容量等待；关停时每个解耦服务的两个池都回到 READY 时的基线（未复核；作业 3321538 验证了无抢占、无容量等待的成批 decode） |

**16 token Page 下已验证的结果**（作业 3321538，perf-p16，runtime `b37044f`，eager，两种布局都预热过；已离线复核为 `SPF1_PERF_PASS_VERIFIED`）：session 规模和池大小与冒烟测试相同，但没有 784 token 解耦布局，也没有 `random50`。1 个 session × 10 轮、8 GiB、prompt 最长 60,440 token：`full` 在默认布局和 16 token Page 下全部 10 轮的 token 和 restore H2D 字节数都相同；16 token Page 下 `observation_query` 和 `goal_query` 从第四轮起选了约 40 % 的 Page（该轮为 844 个中的 337 个；最后一轮为 3,346 个中的 1,338 个和 3,332 个中的 1,332 个），并在 6 轮中恢复了 Page。4 个 session × 6 轮、24 GiB（默认布局的 `full`；16 token Page 的 `full`、`observation_query`、`goal_query` 和 `random_fill`）：每个服务都让 4 个 session 成批 decode，无抢占，无容量等待。

`780fe54` 在 `8f1af6a` 之上只加了 `random_fill`；其余四个策略在两者上发送和写出的字节完全相同（`test_random_fill_legacy.py`）。

**16 token Page 的 SWE-bench pilot**（runtime `780fe54`；Mini-SWE-Agent 2.4.2 和第 7 节的设置，与作业 3300434 相同的 10 个 Astropy instance，分成两组、每组 5 个 session；24 GiB；每个服务都加 `--page-tokens 16 --kv-layout decoupled`；每个策略一个作业、只跑一次；agent 每组上限 3,000 秒）。每个服务都通过了服务端检查。轮数即 agent 的请求数；resolved 来自 swebench 5.0.2：

| 策略 | 作业 | 轮数 | Resolved |
| --- | --- | --- | --- |
| `full` | 3313802 | 517 | 7/10 |
| `observation_query` | 3314010 | 538 | 8/10 |
| `goal_query` | 3314319 | 643 | 7/10 |
| `random_fill` | 3314623 | 608 | 8/10 |

同样的 instance 在 784 token 的 Page 下（runtime `b1e6988`，作业 3300434）：`full` 490 轮、9/10 resolved，`goal_query` 716 轮、6/10，`observation_query` 715 轮、7/10。

这是 10 个 instance 上每个策略一次运行。在作业 3321538（见上）中，`full` 的 token 与页大小无关，但它在两次 pilot 之间有 2 个 instance 的结果变了，所以这种量级的差异单靠两次运行之间的波动就可能出现。**这不是 benchmark 证据**，也不能据此给策略排序。

已离线按 token 逐一核对的服务端证据归档：作业 3313802、3314010、3314319 和 3314623（四个策略全部）。

**性能说明**（每项只跑了一次）：
- 作业 3321538（eager；两种布局都预热过），`full` 的稳态 token 间延迟：单个请求 decode 时默认布局 67.5 ms、16 token Page 71.4 ms，四个请求时 70.5 ms 对 73.1 ms。它在 4 个 session 下的每步开销检查显示 16 token Page 比默认布局每个请求每步多 0.3 到 3.9 ms（1 个 session 时为 −0.3 到 1.5 ms），超过了设计目标“每个请求每步最多多 0.1 ms”（DESIGN R5）。冒烟测试 3308015 和 3308482（未复核）各只预热了一种布局，曾显示差距来自预热；两种布局都预热后差距仍在。作业 3366128 的 decode 检查（4.11 节；已离线复核）在 CUDA graph `full`、16 个请求在 SWE 类历史上 decode 时测了它：16 token Page 每 token 53.8 ms，784 token Page 40.8 ms，两者的前向时间相同，所以每个请求每步多出约 0.8 ms 的主机工作（(53.8 − 40.8) / 16）；加 `--decode-fastpath`（现已默认开启）后 16 token Page 为 36.1 ms。
- 作业 3321538，4 个 session：`full` 在 16 token Page 下每分钟完成 14.94 轮，默认布局下 14.68 轮。

## 出自 4.8 节：CUDA graph

移自 docs/reference.zh-CN.md 的 [4.8 节](../reference.zh-CN.md#48-cuda-graph)。

- **GPU 上的运行**（合成会话，使用作业 3270721 的干净环境（复用）和一张 GH200，每项只跑一次；测的是机制和开销，不是 benchmark 结果）：
  - 作业 3338893（runtime `f19f464`；策略 `full`，1 个 session × 3 轮，prompt 最长约 13K token，784 token 和 16 token 的 Page 都跑了）：`piecewise`、`full-decode` 和 `full` 每一轮的 token 都与 eager 相同；同一模式下的 stock vLLM 0.24.0 重放这些轮次，生成的 token 也相同。这个作业里没有任何一轮选页。该作业的判定是 `SCG2_FAIL_VERIFIED`，原因是它的 `observation_query` 服务没有通过“有一轮选了页”的检查（`query-selection-exercised`）；其余服务都通过了。
  - 作业 3339448（同一 runtime；16 token Page 下的 `observation_query`，1 个和 8 个 session × 4 轮，prompt 最长约 37K token，每种模式）：graph 模式的输出与 eager 不同。1 个 session 时，第一处差异出现在第 1 轮的回复里，早于任何一轮选页；8 个 session 时，选中的 Page 也不同（在选页生效的轮次上，与 eager 的 Jaccard 为 0.81–0.93）。8 个请求同时 decode 时的稳态 decode 步：eager 82.7 ms，`piecewise` 29.2 ms，`full-decode` 34.0 ms，`full` 28.9 ms；这是开销测量，不是端到端加速。
  - 作业 3343069（runtime `613a7d5`）：用 `full` 的 16 个 session 与 eager 下的同一负载相比，输出和选页都不同；session 池用 `full-decode` 运行（4.4 节）。

从选项正文中删去的说明（原文照录）：

- 作业 3343069 在 16 个 session 时记录的正是这些尺寸，K = 8 时为 1、2、4、8、16）
- GPU 上作业 3410750（`GEMR_PASS_VERIFIED`）跑了 `full-decode` 和 `full`，作业 3429673、3461663 以及 Gemma 的 benchmark 运行跑了 `full`（4.14 节）

## 出自 4.9 节：上下文上限与回复上限

移自 docs/reference.zh-CN.md 的 [4.9 节](../reference.zh-CN.md#49-上下文上限与回复上限)。

三个 launcher 选项让服务能承载更长的 agent 运行，三个都已在作业 3348705 中在 GPU 上跑过（见下）。不使用它们时，vLLM 命令、`startup.json`、其中的固定值、证据文件和选页结果都与 `8c5f4d0` 逐字节相同（CPU 测试）。`--max-model-len 262144` 是后来加入的第三个上下文选项，已在作业 3360375 中在 GPU 上跑过（见下）。加入它之后，默认配置和 131072 的配置都保持不变（CPU 测试，`test_ctx_256k.py`）。

- **GPU 上的运行（作业 3348705，runtime `710bfc6`，一张 GH200；已离线复核，token 完全一致）。** 所有服务都使用 `--max-model-len 131072 --max-tokens 4096` 和 16 token 的解耦布局。单个 session（`full` 和 `observation_query`，eager，16 GiB KV）：prompt 增长到 120,864 token；128,871 和 128,873 token 的请求在准入前被前端以 HTTP 400 拒绝，之后服务继续正常工作。16 个 session（`observation_query` 加 `--active-token-budget 8192`，`--cuda-graphs full`，32 GiB KV，`--dram-arena-gib 88`）：160 轮，其中 114 轮启用了选页，各 session 达到 60K 到 111K token，没有抢占；有 6 轮正好生成了 4,096 个 token（prompt 为 33.6K 到 76.2K），其后的轮次正常继续；16 个同时运行时 ITL 为 40.3 ms；进程 RSS 最多 117.9 GiB，cgroup 峰值 134 GiB（申请了 200 GiB）。没有覆盖到：接近上下文上限时的 4,096 token 回复（计划在约 121K token 处的长回复轮次分别只生成了 69 和 215 个 token 就结束了）。锁页内存不会出现在进程的 VmPin 中，所以 128 GiB 的锁页主机内存预算依据的是 launcher 的计算，而不是实测。
- **GPU 上的运行（作业 3360375，runtime `e3efc22`，一张 GH200；已离线复核，token 完全一致，33 项检查无一失败）。** `e3efc22` 是本版本的 262144 token 各 commit 在 `175a183` 上的版本，即 rebase 到 4.11 节的 step profile 和解码快速路径之前的版本；它的 runtime 源码等于本版本去掉这两项之后的源码，而这里的服务都没有用到这两项（262144 与解码快速路径同时使用的情形后来在作业 3461661 和 3461703 中在 GPU 上跑过，见下；CPU：`test_ctx_256k.FastPathTests`）。所有服务都使用 `--max-model-len 262144 --max-tokens 4096` 和 16 token 的解耦布局。
  - 单个 session，`full`，eager，24 GiB KV：21 轮。prompt 增长到 239,527 token，对这个 prompt 的回复生成了完整的 4,096 个 token（合计 243,623）。一个 259,561 token 的请求（加上 4,096 token 的回复后超过 262,144）在准入前被前端以 HTTP 400 拒绝，之后健康检查返回 200。1 个同时运行时 ITL 为 77.4 ms。同样的运行换成 `observation_query` 时达到 239,525 token，离线选页重放在全部 19 个选页轮次上都与记录的选页相同。
  - 16 个 session，`observation_query` 加 `--active-token-budget 8192`，`--cuda-graphs full`，32 GiB KV，`--dram-arena-gib 88`：273 个请求，prompt 最多 221,416 token，峰值时 16 个同时运行，16 个同时运行时 ITL 为 41.3 ms，没有抢占，也没有容量等待，GPU 显存最多 93,349 MiB。选页重放在全部 199 个选页轮次上都与记录的选页相同。一个 268,473 token 的请求被前端以 HTTP 400 拒绝，服务继续正常工作。有 5 个回复达到了 4,096 token 的上限（prompt 为 147,672 到 221,416 token）；其中有下一轮的 3 个，下一轮都 token 完全一致地继续。
  - arena 上限（2 个 session，`full`，24 GiB KV，`--dram-arena-gib 17`）：第二个 session 在第 7 轮的捕获放不进 arena，于是该请求以 HTTP 500 失败，服务日志中记录 `DRAM_ARENA_EXHAUSTED`，符合设计；第一个 session 的下一轮（197,276 token）正常完成。
  - 主机内存：16 个 session 的服务 RSS 最多 111.8 GiB，launcher 估算的最坏情况为 122.3 GiB；作业的 cgroup 峰值为 134.6 GiB（申请了 200 GiB）。唯一的发现：锁页内存不会出现在 VmPin 中（读数为 0），所以 128 GiB 的锁页主机内存预算依据的仍是 launcher 的计算。
- **25 GiB 下 GPU 上的运行（作业 3461661 和 3461703，runtime `cf2b28d`，一张 GH200；已离线复核，token 完全一致）。** Qwen 使用 `--kv-cache-gib 25 --max-model-len 262144 --max-tokens 4096`、16 token 的 Page、`--dram-arena-gib 70`、`--cuda-graphs full`、三条解码快速路径、`--deferred-completion-io`、8,192 token 的预算，`goal_query` 另加 `--admission-overlap 16 --goal-rewriter-batch 4`。
  - Terminal-Bench 2.0 dry run（作业 3461661，`TB16_PASS_VERIFIED`，每个服务的合并 16 个全部通过；chat-text，K = 20，`--release-idle-pages`，16 个任务）：`goal_query` 362 个请求，prompt 最多 92,354 token，16 个中解决 8 个；`observation_query` 390 个请求，prompt 最多 102,611 token，解决 9 个。两者都没有容量等待、没有重建、没有释放空闲 Page、没有上下文拒绝，也没有抢占。
  - SWE-bench Verified，前 32 个 instance、16 个 session（作业 3461703，`SWE32_PASS_VERIFIED`，Goal 文本已核验，合并 32 个全部通过）：`goal_query` 2,109 个请求，prompt 最多 78,990 token，279 次容量等待，2 次重建，解决 19 个；`observation_query` 2,183 个请求，prompt 最多 70,114 token，没有容量等待，4 次重建，解决 20 个。两者都没有上下文拒绝，也没有抢占。

从选项正文中删去的说明（原文照录）：

- 自 6.7 节的 Terminal-Bench dry run 起每个 benchmark 服务都用它）
- 在 SWE-bench pilot 作业 3313802、3314010、3314319 和 3314623（4.7 节；四个策略都有，使用 32 GiB 的池）的服务日志中，vLLM 的 `kv_offload_cpu_cache_usage_perc` 在全部 1,166 行里都是 0.0
- 262144 token（已在作业 3360375 中经 GPU 验证，见上）
- （Qwen 的 25 GiB：作业 3461661 和 3461703，见上）
- （24 GiB：18、12、7；一个服务最多运行 16 个）。Qwen 已在 GPU 上以 25 GiB 和 262,144 token 运行过（作业 3461661 和 3461703，见上）
- （按同一规则推导；作业 3360375 在 32 GiB 下用 8192 token 的预算运行，没有容量等待）
- （最多 16；Qwen 的 25 GiB：作业 3461661 和 3461703，见上）

## 出自 4.10 节：admission 优化

移自 docs/reference.zh-CN.md 的 [4.10 节](../reference.zh-CN.md#410-admission-优化需显式开启)。

  - **收益与代价。** Goal 比阶段 1 慢的轮次，大约早一个阶段 1 的时间被接受。在作业 3500604（见下）中，重叠窗口 `goal_overlap_ms`（从 Goal 开始到 join）的中位数在 chat-text 上为 170 ms，在 mswea-xml 上为 121 ms；节省的时间最多是 `goal_construct_ms` 和 `goal_overlap_ms` 中较小的那个，该作业没有单独测量它（开与不开该选项时请求时间的中位数：chat-text 上 7.24 s 对 7.23 s，mswea-xml 上 4.37 s 对 4.32 s）。有 Goal forward 的轮次要多一次 probe-only pass。

  - **GPU 上（作业 3500604，runtime `c07d42a`，一张 GH200；已离线复核为 `CMG1_PASS_VERIFIED`）。** 两对 Qwen `goal_query` 服务（不开与开该选项）以 lockstep 运行相同的会话，每个都带 `--admission-overlap 4`、`--release-idle-pages`、CUDA graph `full` 和三个解码快速路径：chat-text 上的 strip 历史（`--thinking-history strip`，30 轮），以及 mswea-xml 上的 keep 模式（只开 `--enable-thinking`，18 轮）。两对都是 `EXACT`：每轮的 Goal 都相同（30 轮中 30 轮，18 轮中 18 轮），其中 18 轮和 8 轮有 Goal forward。该选项还在作业 3500565 失败的 SWE-bench 运行中与 vLLM rewriter 一起跑过（`--goal-rewriter-engine vllm`，见下）。

- **GPU 上的运行（作业 3349504，runtime `4eae441`，即合并进本版本的 admission 工作；一张 GH200；已离线复核，token 完全一致）。** `goal_query` 带 Qwen3-0.6B rewriter，5 个和 16 个 session，`--cuda-graphs full`，24 GiB KV。离线选页重放不适用：它只覆盖 `observation_query` 和 `random_fill`。16 个 session 时，不开参数，请求平均等待 admission 48.3 秒（占请求时间的 86 %），服务每分钟完成 13.8 轮。开 `--admission-overlap 16` 时，admission 等待为 0，服务每分钟完成 17.3 轮（只跑了一次）；等待转移到了 rewriter 上，它是一个一次只服务一个请求的进程（Goal 构造平均用时：不开参数时 3.8 秒，开时 33.9 秒）。三个参数都开、K = 16 时，每分钟完成 15.7 轮，有 726 次 `CAPACITY_WAIT` 拒绝。5 个 session 时，三个服务每分钟分别完成 11.1、12.4 和 12.4 轮（各跑一次）。开了参数的服务，其输出和选页与不开参数的服务不同，成批运行本来就是如此（4.4 节）。
- **GPU 上的运行（作业 3349078，runtime `4eae441`，一张 GH200；已离线复核，token 完全一致，选页重放与记录一致）。** `observation_query`，16 token 解耦布局，24 GiB KV。单个 session、eager、8 轮 6,500 token 的 observation：分别开 `--gdn-device-handoff`、`--phase1-in-step` 以及两者都开时，每一轮的 token、选页和 Page 向量都与不开的服务相同；不开的服务重跑一次也完全相同（全部 2,866 个 Page 的阶段 1 向量一致）。16 个 session、`--cuda-graphs full`、每个 session 6 轮 2,600 token 的 observation：两个开关都开时，引擎线程上每轮的 admission 时间从 342 ms 降到 55 ms，解码中请求的停顿从运行时间的 36.8 % 降到 6.3 %（94 % 的后续轮次在 step 内完成阶段 1），16 个同时运行时的 ITL 从 32.0 ms 升到 32.9 ms，96 个请求早 1.8 % 跑完（73.8 s 对 75.2 s；各只运行了一次，而该作业中唯一一次相同服务的重跑，即单个 session、eager、不开这两个开关的服务，两次分别用时 87.6 s 和 93.0 s，相差 5.9 %，开开关的单 session 服务用时 88.2 到 89.6 s，落在两者之间；2026-11-10 根据该作业的证据重新计算）。把变慢的混合 step 算进去，停顿是运行时间的 32.1–33.2 %，不开这两个开关时是 36.8 %。harness 报告的是 13.0 %，偏低：它把服务在同样多请求运行时的 ITL 当作混合 step 对应的普通 step，而在 4 到 14 个请求运行时，这个 ITL 主要就是这些混合 step 本身（328–352 ms；一个混合 step 用时 272–302 ms）。这个范围是根据该作业的证据重新计算的（2026-11-10），普通 step 取 21.9 ms（8 个请求在 2K token 的上下文上 decode，作业 3561459，runtime `51ecc30`）到 32.9 ms（本服务 16 个同时运行时）之间。5 个 session、150 token 的 observation：停顿从 13.5 % 降到 4.1 %（算上 step 为 12.2 %），早 3.8 % 跑完（61.5 s 对 64.0 s；各只运行了一次，在上述 5.9 % 之内）。

  - **在 GPU 上（作业 3404230，runtime `5b15b2f`，N = 4；一块 GH200；已离线核验）。** SWE-bench，第 7 节的 32 个任务、16 个 session，`goal_query` 加 Qwen3-0.6B rewriter、`--admission-overlap 16`、CUDA graphs `full`、两个 decode 快速路径，对照作业 3404696（同一 runtime、同样配置但不开这个参数；各运行一次）。362 次 rewriter 调用组成 177 个批次：101 个批次只有一个调用，16 个有两个，11 个有三个，49 个有四个（196 次调用）。每次调用（中位数，`goals.jsonl`）：排队 15.6 秒，对照 98.2 秒；调用本身 16.8 秒，对照 8.6 秒（四个调用的批次 26.5 秒，单独的调用 2.2 秒）；有 rewriter 调用的 Goal 用时 41.2 秒，对照 108.8 秒；服务每分钟完成 44.9 个 turn，对照 34.2 个（+31 %）。其中一部分来自只有少数 agent 运行的时间：不批量的运行在 3,706 秒的跨度里有 736 秒只有 3 个或更少的 agent 在运行，批量的运行在 3,000 秒里只有 241 秒；按每个 agent 分钟计，提升为 +24 %（3.43 对 2.76 个 turn）。之后开这个参数的运行每分钟完成 41.19 个 turn（作业 3428631，见下；按每个 agent 分钟计比作业 3404696 高 22 %）和 40.15 个（作业 3461703，runtime `cf2b28d`，25 GiB、262,144 token，见 [docs/SWEBENCH_BENCHMARK.zh-CN.md](../SWEBENCH_BENCHMARK.zh-CN.md)）；这些按 agent 计的数字是 2026-11-10 根据这些作业的证据重新计算的。59 次调用回退，对照 294 次中的 35 次：重复的 symbols 32 次对照 18 次（占调用的 8.8 % 对照 6.1 %，对各运行一次而言在偶然范围内：Fisher 精确检验 p = 0.24；单独到来的调用为 6.9 %），无法解析的输出 17 次对照 16 次，其他字段重复 1 次对照 1 次，以及 9 次新的回退，全部发生在 GPU 已满时（从 12:43 起的 18 分钟内使用 97,871 MiB 中的 96,312–97,358 MiB；不开这个参数时最多 94,138 MiB）：4 次显存不足，同一个四个调用的批次，它的两半以及随后单独运行的每个请求也都显存不足（全部用时 0.9 秒；每次失败时子进程已分配 5.2 GiB 张量、剩余 512 KiB，而其中任一请求所需的 KV cache 都远小于 1 GiB）；5 次 cuDNN attention 错误（`mha_graph.execute`）：一个三个调用的批次，其请求单独运行时以同样方式失败，以及两个单独到来的调用（上游的 `generate`）。修复（rewriter-batch-fix；已在作业 3428631 中在 GPU 上运行，见下一项）在重试之前释放失败的批次，在每次尝试之前和每次批量 `generate` 之后清空 CUDA 缓存，并且 attention kernel 仍由 PyTorch 选择：cuDNN 错误也出现在没有填充的单独调用上，而且只在 GPU 已满时出现。
  - **修复后在 GPU 上（作业 3428631，runtime `120bc3c`，N = 4，默认 T = 16,384；一块 GH200；已离线核验）。** SWE-bench，与作业 3404230 相同的 32 个任务和 16 个 session，另外开了 `--decode-fastpath3` 和 `--deferred-completion-io`：320 次 rewriter 调用；7 个批次（5 个三个调用的、2 个两个调用的）显存不足，按 1 + 2 或 1 + 1 运行后恢复，所以没有 Goal 因显存而回退，也没有 cuDNN 错误；322 次 Goal 重写中回退 62 次（19.3 %）：320 次调用中 60 次校验或生成失败（"retrieval_goal.symbols must be unique" 37 次，占调用的 11.6 %，不批处理的作业 3404696 中为 294 次中 18 次，6.1 %，Fisher 精确检验 p = 0.023；rewriter 输出的 JSON 未闭合 20 次，不批处理 16 次；"files must be unique" 3 次），另有 2 次在调用之前就超出了 compact rewrite-request 上限；全部回退 19.3 % 对 11.9 %（p = 0.015）。几次运行的 runtime 版本和 agent 轨迹不同，所以这只是相关，不是受控对比；批处理路径里没有找到会产生重复的机制。32 个任务全部完成，解决 19 个（[docs/SWEBENCH_BENCHMARK.zh-CN.md](../SWEBENCH_BENCHMARK.zh-CN.md)）。

  - **GPU 上（实验性，未通过验证）。** Qwen：作业 3488202 的回放（perf-rwreplay2，`RWR2_PASS_VERIFIED`）发现每次 lm-format-enforcer 调用都失败（见上；xgrammar 回答了每次调用）；修复之后，作业 3499550 没有通过一致性检查（见上）；作业 3500565 的 SWE-bench 运行（runtime `c07d42a`，带 `--goal-overlap`）的结果是 `SWE32_FAIL_VERIFIED`（未通过的检查为 `capture-sizes`，其中 PIECEWISE 只 capture 了 7 个尺寸中的 5 个，以及 `context-limits`；发现项 `CUDA_GRAPH_CONFIG_DIFFERS` 和 `GOAL_FALLBACK`）。Gemma 4：在作业 3508345（`CMG1_FAIL_VERIFIED`）中，它带 vLLM 子进程和 `--goal-overlap` 的服务始终没有 READY（权重没能在启动期限之前加载完，原因尚未查明）；重试作业 3538523 在 `c07d42a` 上（`CMG1_PASS_VERIFIED`；4.14 节）运行了它：该服务 gvll 回答了 49 个请求中的 49 个，与使用 Hugging Face rewriter 的同样服务 govl 相比，这一对为 EXACT，23 个 lockstep 轮次中 22 个的 Goal 相同。

- **推荐**，也是我们 SWE-bench 运行使用的配置：`goal_query` 加 `--admission-overlap 16`（16 个 session 时）和 `--goal-rewriter-batch 4`，其他 admission 参数关闭，rewriter 引擎用默认的 `hf`；其他所有策略的 admission 参数全部关闭。

从选项正文中删去的说明（原文照录）：

- （Qwen：已在作业 3500604 中在 GPU 上运行；Gemma 4：已在作业 3538523 中在 GPU 上运行，此前作业 3508345 的服务始终没有 READY）
- （未经 GPU 验证：作业 3569412 在 `b6f6b9f` 上运行过它，部分通过；见下面它的条目）
- （作业 3349504 中，5 个 session 时有 2 或 3 个在进行（多数为 3 个）、16 个 session 时有 8 到 12 个在进行（多数为 10 或 11 个）时，新的 admission 被拒绝）
- （Qwen：已在作业 3500604 中在 GPU 上检查，见下；Gemma 4 profile 也接受它，已在作业 3538523 中在 GPU 上运行，见 4.14 节）
- （integrate-1031 和 keep-overlap；已在作业 3500604 中在 GPU 上运行，见下一条）
- 见 README 的 [`main` 包含什么](../../README.zh-CN.md#main-包含什么runtime-4ebff97)，其发现项见[历史](../history.zh-CN.md#integrate-11122026-11-12)）
- 已在作业 3404230 中以 N = 4 在 GPU 上运行（见下文），其中从 GPU 显存故障中恢复的路径失败了，修复和 token 上限之后又在作业 3428631 中运行（见下文），其中显存不足的 7 个批次都通过拆分恢复）
- 0：不设上限；已在作业 3428631 中在 GPU 上运行，见下文）
- 在作业 3428631（见下文）中仍有 7 个两到三个调用的批次显存不足；每个都通过分开运行恢复（1 + 2 或 1 + 1，见上文），所以没有 Goal 因显存而回退
- 实验性：已在 GPU 上运行，但未通过验证，见下
- GPU 上（perf-rwreplay3，作业 3499550，runtime `c07d42a`，`RWR3_FAIL_VERIFIED`）启动检查通过，也没有调用在没有 token 的情况下失败，但单独调用时该后端的回答只在 160 次调用中的 99 次与 Hugging Face 子进程相同，低于回放的 0.7 阈值
- （integrate-1030；在 Qwen 上经过 CPU 测试；在 GPU 上只出现在下面失败的 Qwen SWE-bench 运行中）

## 出自 4.11 节：解码快速路径

移自 docs/reference.zh-CN.md 的 [4.11 节](../reference.zh-CN.md#411-解码快速路径默认开启)。

**GPU 上的运行（作业 3366128，runtime `43fd9c3`，一张 GH200；已离线复核）。**

- **负载。** 解码微基准：16、8、4、1 个请求同时运行，上下文约 2K 和 20K token；另有 16 段类 SWE 历史，8K 到 48K token（平均 prompt 21.6K）。
- **服务。** 本 runtime 使用 `--cuda-graphs full` 和同步调度，对照为开启和关闭异步调度的 stock vLLM 0.24.0。

16 个请求同时运行时，在类 SWE 历史上的 token 间延迟：

| 服务 | 每个 token 的 ms |
| --- | ---: |
| 本 runtime，16 token 的 Page | 53.8 |
| 本 runtime，16 token 的 Page，`--decode-fastpath` | 36.1 |
| 本 runtime，784 token 的 Page | 40.8 |
| 本 runtime，`observation_query`，`--decode-fastpath` | 31.0 |
| stock vLLM，同步调度 | 29.2 |
| stock vLLM，异步调度 | 27.6 |

- 16、64、784 token 的 Page 下，前向时间相同。按服务所有 16 个请求同时运行的 step 计（全部轮次：step profiler 的均值，不只是表中的 W2 轮），每步约 21 ms（21.09、21.02 和 21.27 ms）；在表中的 W2 轮里约 26 ms（25.87、25.89 和 26.33 ms；各一个 profiler 窗口，16 个请求运行的 step 为 39 到 44 个）。
- 打开开关后，所有 16 个请求同时运行的 step 的 host 时间减半，从 18.5 ms 降到 9.2 ms；在表中的 W2 轮里从 28.1 ms 降到 10.2 ms（各一个 profiler 窗口，39 和 27 个 step）。
- 只有一个请求时，开关影响很小：2K token 时开和关都是 20.3 ms。
- KV cache 布局（HND 或 NHD）在两种服务上都没有可测的差别（在约 1 ms 以内）。
- 结论为部分通过：每轮的测量窗口在第一个请求结束时截止，91 个窗口中有 84 个只有 28 到 62 个解码步（7 个单请求轮次的窗口为 73 到 229 步），而不是计划的 150 步；所有服务都跑完了全部轮次。（W2 的逐步计时数字和这些步数是 2026-11-10 根据该作业的证据重新计算的。）

**推荐所有策略都打开。** [docs/SWEBENCH_BENCHMARK.zh-CN.md](../SWEBENCH_BENCHMARK.zh-CN.md) 中 runtime `5b15b2f` 上的 SWE-bench 运行打开了这个开关和 `--decode-fastpath2`（作业 3404696 和 3404230），`120bc3c` 上的运行打开了全部三条快速路径和 `--deferred-completion-io`（作业 3428631 和 3429500），Terminal-Bench 和 Gemma 的 benchmark 配置也是如此；早先的 SWE-bench 运行使用 `e877051`，没有打开它。

**Gemma 的差距在哪里。** 在作业 3410750 中（一个 session，`full`，CUDA graph `full`，开两个快速路径），本 runtime 每个 token 比 stock vLLM 慢 3.8 ms（29.30 对 25.49 ms）。其中约 1.7 ms（1.3 到 2.1）是每轮的完成工作，vLLM 把它算进最后一个 token 的间隔（699 个间隔中有 10 个在 75 到 300 ms 之间；11 轮的回复平均 64.5 个 token）；稳定的解码步约慢 2.1 ms（1.7 到 2.5）。如果 Gemma 与 Qwen 在单个请求时相同（作业 3379206；Gemma 上没有测量），其中 stock 的异步调度约占 1.4 到 1.6 ms，其余 0.5 到 0.7 ms 是本 runtime 的同步 host 工作（CPU 模拟环境中本 runtime 自己的部分在 `--decode-fastpath2` 下约为每步 0.50 ms）。16 个请求时（作业 3410948，`observation_query`，2K token）差距为每个 token 6.0 ms（33.36 对 27.34 ms）。

**GPU 上的运行。** Qwen：它在 SWE-bench 作业 3428631 和 3429500（runtime `120bc3c`）中运行过。解码检查（perf-decode3，作业 3429022，runtime `120bc3c`；离线核验为 `SPD3_PARTIAL_PASS_VERIFIED`：每个 runtime 服务都通过，只是部分轮次缺少 stock 服务的测量窗口，所以是部分通过）：16 个并发 session 时，任何配置在两次运行之间都不是逐字节相同的，因为 batch 的组成会改变 greedy 解码中接近平局的选择（4.4 节）。在 prompt 逐字节相同的请求中，`--decode-fastpath3` 对照 `--decode-fastpath2` 时 76–83 % 的回复相同，同一配置只切换 step profiler 时为 81–82 %（stock vLLM 异步调度对照同步调度：90 %）。差异集中在 8 个及以上并发 session 的轮次中；在 1 到 4 个 session 的轮次中，比较的同 prompt 回复除一个 2 个 session 的轮次中的一条外都相同（不开 profiler：3 条中 2 条相同），1 个 session 的轮次中的回复全部相同。所以第三条快速路径没有带来超出运行间波动的差异；Gemma 的 g1c（下面）是严格的单 session 检查。不开 step profiler 时，按表中打印的窗口均值，它使 token 间延迟在 full attention 下变化 −0.51 到 +0.10 ms，在 `observation_query` 下变化 −0.06 到 −0.90 ms（最多快 3.3 %）；按每个窗口内 0.25 秒区间的中位数，则分别为 −0.42 到 +0.19 ms 和 −0.05 到 −0.43 ms（最多快 1.7 %）。这些都只运行了一次，且与同样开关的运行间波动大小相当：在两边窗口都完整的格子里，作业 3461668（runtime `cf2b28d`）和作业 3561459（runtime `51ecc30`）中与本作业开关相同的服务，16 个请求时与本作业相差最多 1.12 ms（按中位数 0.52 ms），1 到 8 个请求时最多 0.56 ms（0.26 ms）（2026-11-10 根据这些作业的证据重新计算）。Gemma：在作业 3429673（runtime `9ef0d6d`，stage-0 的 XML append 会话，一个 session）中，开三条快速路径的服务与开两条的服务在 eager 和 `--cuda-graphs full` 下 11 轮的 token 和证据都完全相同；16 个 session 的检查 g1b（其 `observation_query` 和 `goal_query` 服务开了三条快速路径）也已通过（作业 3429672，`GEMR_PASS_VERIFIED`）。

**GPU 上的运行。** Qwen：SWE-bench 作业 3428631 和 3429500（runtime `120bc3c`）开着它运行。在作业 3428631 中，一项 harness 检查（goal-async）还不接受一直保留到该请求的文件写完才发出的 Cancel 确认（见上文“谁等待这些文件”）；该请求被正确服务，修正后的检查在该作业的整个日志上通过。在作业 3429022 的解码检查（上文）中，prompt 逐字节相同的请求里开和不开它时 76–80 % 的回复相同，在 16 个并发 session 的运行间波动之内（1 到 4 个 session 的轮次中除一条外都相同：`observation_query` 下一个 2 个 session 的轮次中的一条，5 条中 4 条相同）。按表中打印的窗口均值，它使 token 间延迟变化 −0.32 到 +1.12 ms：full attention 下 2K token、16 个请求时 +1.12 ms（+4.5 %），20K、16 个请求时 +0.58 ms（+2.0 %），其余在约 1 % 以内。这些都只运行了一次，而且这两个均值来自突发：两个 2K 窗口里，前 5 秒的大部分时间每个 token 约 23.7 ms，之后有几段慢到 34 ms（不开该开关）和 38 ms（开），所以均值比窗口内 0.25 秒区间的中位数高 0.90 和 1.92 ms。按这些中位数，该开关在这两格的变化是 +0.11 ms（+0.5 %）和 +0.01 ms，在两边窗口都完整的所有格子里是 −0.21 到 +0.11 ms；同样的开关在作业 3561459（runtime `51ecc30`）中，2K token、16 个请求时按窗口均值每个 token 24.86 ms，比本作业开该开关的服务低 1.12 ms，与不开的服务相同（24.85 ms）。每个请求固定开销的中位数（端到端时间减去按该轮稳定 token 间延迟计的生成时间）在两边窗口都完整的轮次中每轮变化 −375 到 +39 ms：full attention 下 20K token、8 个请求时 −375 ms，20K、16 个请求时 −317 ms，`observation_query` 下 20K token、8 个请求时 +39 ms。W2 轮的 −442 ms（16 个请求，full attention）不计入：开该开关的服务在该轮的窗口只有 2.3 秒，到它的一个请求结束为止，之后该轮以 15 个请求跑了 12.75 秒，而该公式把每个 token 都按 16 个请求时的延迟计（2026-11-10 根据这些作业的证据重新计算）。Gemma：在作业 3429673 中，开它的服务与只开三条快速路径的服务 11 轮的 token 和证据都完全相同；MemoryAgentBench 作业 3429503 和 3429507 也开着它运行。尚未测量：只开该开关时 16 session agent 的吞吐、写入线程在 GH200 的 CPU 上对解码步的影响，以及作业所用文件系统的 fsync 时间。worker 的 `complete` RPC（作业 3410181 中 517 个新 Page 时约 40 ms）仍在路径上。

从选项正文中删去的说明（原文照录）：

- （GPU 上：在作业 3410750（runtime `316a1b4`，stage-0 append 会话，一个 session）中，只开 `--decode-fastpath` 和开两条快速路径的服务都在全部 11 轮中给出了 eager 服务的 token 和证据；它们也在作业 3429503、3429507 和 3429673 中运行过）
- GPU 上的运行：** 它在 SWE-bench 作业 3404696 和 3404230（runtime `5b15b2f`）中运行过，但还没有测量它对 token 间延迟的影响

## 出自 4.12 节：纯文本

移自 docs/reference.zh-CN.md 的 [4.12 节](../reference.zh-CN.md#412-纯文本--language-model-only默认开启)。

**GPU 上的运行：** `full` 的 Terminal-Bench dry run（6.7 节）是第一次带这个参数的运行；它的日志有 vLLM 的纯文本日志行，没有多模态预热。作业 3404696 和 3404230 的 SWE-bench 运行（`goal_query`）也带了它；之后的每次 benchmark 运行也都带了它，两个模型 profile 都是如此（Gemma profile 要求这个参数）。

## 出自 4.13 节：释放空闲 session 的 Page

移自 docs/reference.zh-CN.md 的 [4.13 节](../reference.zh-CN.md#413-释放空闲-session-的-page需显式开启)。

**GPU 上的运行：** MemoryAgentBench 作业 3397842（Qwen3.6，runtime `0d37346`，16 个 session）开着这个选项在两种池大小下运行：16 GiB 的池释放 139 次、重新载入 124 次，24 GiB 释放 13 次、重新载入 13 次。两种池下全部 32 个 case 完全相同，每次重新载入都恰好恢复了一个已完成的轮次提交过的 Page。该作业唯一失败的检查（route-b）是 harness 检查本身的错误：它把引擎完成这一轮之后 HTTP 尾部中的释放也算了进去；用改正后的检查在该作业的证据上重新判定，route-b 通过（perf-mab harness，2026-10-18）。该作业自身的结果仍是 `PMAB_FAIL`（route-b）；它的验证目录显示 `NOT_VERIFIED`：那是 2026-10-18 的第一次核验，用的是当时的作业 harness，因 pin 和交付代码不符而未通过。`full` 的 Terminal-Bench dry run（6.7 节）释放过空闲 session 的 Page，但没有重新载入。

## 出自 4.14 节：Gemma 4 31B IT

移自 docs/reference.zh-CN.md 的 [4.14 节](../reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)。

本版本包含第二个模型 profile 的第一阶段：[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)，revision `842da3794eaa0b77d5f08bae87a17459d91ff475`。它可以安装；其 engine 和 worker（包括滑窗层的生命周期）已在 CPU 上做过端到端测试，启动检查另有单独测试；`serve.py --start` 可以启动它。**GPU 上的运行**（作业 3397695，runtime `e0df9d5`，一张 GH200，离线核验）：它的两个池的布局在关闭选择性 runtime 的情况下提供服务（经本构建的普通服务）：16 token 的 Page 启动成功（两个池分别为 15,867 和 2,886 个 block，KV 26.41 GiB），32 token 的 Page 也启动成功。与作业 3364469 的 stock vLLM 回放（一个 10 轮的 session）相比，32 token 的 Page 在 10 轮中 10 轮 token 级一致；16 token 的 Page 无论开不开 CUDA graph（eager）都从第 3 轮起出现分歧（第一个差异在第 3 到 5 个 token；三个服务在前 3 轮一致），这是 TRITON_ATTN 窗口布局的数值或分块差异。开启选择性 runtime 时，每个 Gemma runtime 服务都在启动时、还没有提供任何服务之前就停止了（作业 3398198，runtime `0d37346`；作业 3398614，runtime `4eb9f9d`；九个服务都走 XML 协议，它们的 stock vLLM 服务都通过了）：`engine renderer and dynamic provider required`，即引擎组装只接受 Qwen 的 renderer。本版本已修复这一问题（gemma-assembly-fix，见[历史](../history.zh-CN.md#历史)）。修复之后，runtime 服务于 2026-10-21 在 runtime `316a1b4` 上第一次在 GPU 上启动：单 session 检查 g1a（作业 3410750，`GEMR_PASS_VERIFIED`：eager、前两条解码快速路径和 CUDA graph；4.11 节）和 16 个 session 的检查 g1b（作业 3410948，`GEMR_PARTIAL_PASS_VERIFIED`：其 `goal_query` 服务在 32 GiB 下与 rewriter 一起运行时显存不足，见下）。**在 runtime `9ef0d6d` 上（一张 GH200，离线核验）** 它们跑完了 MemoryAgentBench 在 kv-session 协议上的端到端运行，`observation_query`（作业 3429503）和带批量 rewriter 的 `goal_query`（作业 3429507），8 个 case、8 个 session、25 GiB，开 CUDA graph `full`、三条解码快速路径、`--deferred-completion-io` 和 `--release-idle-pages`（见下文“MemoryAgentBench”）；以及 XML 协议上的单 session 检查 g1c（作业 3429673）：`--decode-fastpath3` 对照 `--decode-fastpath2`（eager 和 `--cuda-graphs full`），`--deferred-completion-io` 对照只开三条快速路径，11 轮的 token 和证据全部相同，GPU 显存至少剩 5,356 MiB。16 个 session 的检查 g1b 也已通过（作业 3429672，`GEMR_PASS_VERIFIED`）：32 GiB 的 `full` 放得下，但很紧（峰值时剩 746 MiB），它的轮次 222 次等待空间（`CAPACITY_WAIT`，没有抢占）。benchmark 运行：SWE-bench（XML 协议，16 个 session，25 GiB）的 `goal_query` 加 `--goal-rewriter-batch-tokens 15360`（作业 3429808：32 个任务全部完成，解决 15 个，32 个任务的合并全部 token 级一致；rewriter 调用 179 次（180 次 Goal 重写，其中一次在调用之前就回退），Goal fallback 32 次，容量等待 21,121 次；按 runner 的时间跨度为 58.5 tokens/s）和 `observation_query`（作业 3429830：pool 的 2,760 秒上限截掉了 32 个任务中的 7 个；完成 25 个，解决 17 个，token 级一致；容量等待 12,178 次；按服务端统计的全部生成 token 为 73.9 tokens/s，只算完成的 25 个任务的 token、除以同一跨度为 46.7 tokens/s；两者都不能与作业 3429808 的 58.5 直接比较，见 [docs/SWEBENCH_BENCHMARK.zh-CN.md](../SWEBENCH_BENCHMARK.zh-CN.md)）；以及 Terminal-Bench（chat-text 协议）的 dry run：`observation_query`（作业 3433706，runtime `9ef0d6d`：16 个 trial 中解决 4 个，7 个撞到 dry run 的 agent 超时）和 `goal_query`（作业 3434544，runtime `53e3b7a`：解决 5 个，6 个撞到 agent 超时，Goal fallback 3 次），都与各自的轨迹合并通过。作业 3433706 和 3434544 通过、作业 3429830 核验为部分通过（没有未通过的检查；32 个任务中 7 个被时间上限截掉），都是在用修正后的、针对客户端在 admission 期间取消请求的测试工具检查离线重新核验之后得出的（6.7 节）；都没有重跑。依赖已加载模型的启动检查（attention 后端的 cache 形状、final norm、视觉塔的参数名）在不符时直接拒绝启动。Qwen release 不受影响：其 vLLM 命令、启动配置、pin 和证据与上一版本逐字节相同（identity probe 共 555 + 219 项，0 处差异；未复核：这些 probe 的输出从未存档）。该 profile 也以 Gemma 的轮次格式服务 MemoryAgentBench 的 kv-session 协议（见下文“MemoryAgentBench”和 6.6 节）。

- GPU 上（runtime `cf2b28d`，一张 GH200；已离线核验）。单 session 检查 g1d（作业 3461663，`GEMR_PASS_VERIFIED`：stage 0 的 XML 追加 session，28 GiB，P = 16，`--cuda-graphs full`、三条解码快速路径和 `--deferred-completion-io`）：使用 4,096 token 前向的服务通过了每项 runtime 检查，pool 为 19,725 个全局 Page 和 1,606 个滑窗 block id（8,192 时为 17,165 和 2,886），全部 10 个后续轮次的 phase 1 都分成 3,056 token 的块运行（30 次前向，run 最长 7,386 token），峰值 90,037 MiB，剩 7,834 MiB（8,192 时剩 5,356）。它的轮次在 11 轮中有 3 轮与 8,192 token 的服务相同（第一处差异在第 3 轮回复的第 39 个 token）；与作业 3364469 的 stock 重放相比，两者的第一处差异都在第 3 轮的第 5 个 token，而开 `--max-num-batched-tokens 4096` 的 stock vLLM 本身从第 2 轮第 14 个 token 起就与该重放不同。SWE-bench 运行（作业 3462014，`GSWE32_PASS_VERIFIED`，合并 32 个全部通过，token 完全一致）：A.14 的 `observation_query` 服务加 `--max-num-batched-tokens 4096`（[docs/SWEBENCH_BENCHMARK.zh-CN.md](../SWEBENCH_BENCHMARK.zh-CN.md)），前 32 个 instance、16 个 session、25 GiB，pool 为 7,518 个全局 Page 和 6,481 个滑窗 block id（见上表）；32 个任务全部完成，解决 18 个；956 个请求，prompt 最多 72,058 token，两个超过 4,096 token 的 phase 1 分块运行，6,307 次容量等待，空闲 session 的 Page 释放 516 次、重新载入 494 次，1 次重建，没有上下文拒绝，也没有抢占。

**仍需在 GPU 上检查的内容**（服务配置本身已在每次 Gemma benchmark 运行中使用：作业 3429503、3429507、3429808、3429830、3433706、3434544 和 3462014）：131,072 token 下 `full` 和 `observation_query` 各一个 session，服务配置与 eager 服务对比（只开快速路径时必须与 eager 的 token 一致；graph 可能改变 token，见 4.8 节）；对 `--decode-fastpath3`：在 16 个请求时与开两条快速路径的服务比较 inter-token latency，并开启 step profile（`sched.allocate.grow`、`worker.before`、`rpc.allocation_batch`）。已完成：window 布局在 `TRITON_ATTN` 上的启动和 capture（作业 3410750：FULL decode graph 用 1 秒 capture 完，占 0.04–0.05 GiB，KV 为 28 GiB）；在 stage-0 append 会话上，`full-decode` 在全部 11 轮中给出了 eager 服务的 token 和证据（作业 3410750，runtime `316a1b4`）；同一会话上 `full` 对照 `full` 模式的 stock vLLM：stock 服务在全部 10 轮中重现了作业 3364469 的回放，runtime 服务从第 3 轮的第 5 个 token 起与它不同（作业 3410750）；在 1 个和 16 个运行中请求时与 Triton 上的 stock vLLM 对比的 inter-token latency 已在 g1b 中测量（作业 3410948 和 3429672）；`--decode-fastpath3` 给出了开两条快速路径的服务的 token 和证据（作业 3429673，即 g1c：eager 与 `ra-f2` 并列，`--cuda-graphs full` 与 `ra-gf` 并列，`--deferred-completion-io` 与只开三条快速路径的服务并列；11 轮全部相同），并且在 1 个请求、`--cuda-graphs full` 时每 token 29.05 ms，对照 29.26 ms（作业 3429673：单个 session 在 699 个 token 上的整段 token 间延迟；eager 时开三条快速路径的服务为 62.55 ms，对照 61.77 ms；预计最多少零点几 ms：1 个请求时的差距大部分是 stock 的异步调度和每轮的完成工作，见 4.11 节）。

- **GPU 上的运行**（作业 3429503 `observation_query` 和 3429507 `goal_query`，runtime `9ef0d6d`，一张 GH200，离线核验为 `PMAB_PASS_VERIFIED`）：32 个 case 中的 8 个（每项能力的前两个），8 个 session，262,144 token 下 25 GiB，一份内存份额上 `--dram-arena-gib 60`，CUDA graph `full`、三条解码快速路径、`--deferred-completion-io` 和 `--release-idle-pages`；`goal_query` 加 `--admission-overlap 8` 和 `--goal-rewriter-batch 4`。每个服务处理了 155 个请求；8 个 case 中 7 个给出回答，1 个回答停在回复上限；批量 rewriter 共 40 次调用，没有回退，也没有显存不足。32,768 token 预算下 16 个 session 放不进 25 GiB（需要 5,699 个全局 Page，可用 4,957 个）；28 GiB 能容纳（见上）。设置见 [docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](../MEMORYAGENTBENCH_BENCHMARK.zh-CN.md) 的 A.15。

**CPU 测试**（`test_gemma_*.py`）。若 `REMOTE_RA_GEMMA4_METADATA` 指向包含五个固定元数据文件的目录，渲染测试会使用真实 tokenizer（需要 Transformers 5.13 和 Tokenizers 0.22.2，与 2.1 节相同）：transcript 在全部 10 轮（最长 62,919 token）上逐 token 重现 stock vLLM 0.24.0 Gemma 会话的 prompt id。不设置该变量时这些测试会被跳过。`test_gemma_e2e` 用 vLLM 自己的 KV manager（一个全局组、五个滑窗组、64 token 的窗口）对每个策略跑三个 session 的 engine 和 worker：每次前向在每个滑窗组中读到的恰好是最近 63 个 token，并且 gate 的数值是每轮从滑窗池实际取走的块数的上界。`test_gemma_release` 让 16 个 session 一起运行，全局池取选择性规则的最小值：每一轮读到的内容与一个从不释放的池完全相同，被释放的只有空闲 session（最久未用的优先），重新载入恰好恢复被释放的 Page 及其 KV，释放不触碰窗口和滑窗池，而一个永远放不下的重建会以 `CAPACITY_EXCEEDED` 被拒绝。`test_gemma_goal` 覆盖 `goal_query`：Goal 瞬态的布局、pin 和启动器数值、gate，以及 16 个带释放的 `goal_query` session（每一轮与从不释放的池完全相同，每次 Goal 前向都在它的瞬态块内，其 query 向量是 final norm 之前的最后一行、位于 Page 向量的投影空间；没有该瞬态时同样的池会失败）。`test_gemma_chunked` 覆盖分块的 phase 1：算术和 gate 的数值、lease 检查，以及三个 session 上 `full`、`observation_query` 和 `goal_query` 的最长约 2,600 token 的观测，按每次 256 token 前向运行（一次 native 前向 1,024），每一轮都与一次前向的结果相同，每次滑窗读取都精确。`test_gemma_full` 覆盖释放规则下 16 个 session 的 `full`（每一轮与从不释放的池完全相同；重新载入恰好带回每个已封存的 Page 及其 KV；不加该选项时同样的池会失败）。`test_gemma_overlap` 覆盖 `--admission-overlap`：启动记录、滑窗 ledger，以及 16 个 `goal_query` session（rewriter 用替身，Goal 停在各 session 自己的线程上），有无释放时的决策都与不加该选项的服务完全相同。使用真实 tokenizer 时，`test_gemma_renderer` 检查 Goal 从 Gemma transcript 得到的输入及其 Goal probe 的 token。`test_gemma_graphs` 覆盖 CUDA graph 的 pin 和检查、overlay runner 在 Gemma 六个组上的缓冲区绑定，以及挂上 graph 的端到端测试（只有 probe 前向走 eager）；`test_gemma_fastpath` 在关闭、开启一个和开启两个快速路径时运行 window 端到端测试（把每个 batch 条目展开后 RPC 完全相同，证据文件、frame、关注的 Page、回复、滑窗读取、每步的 lease 和 claim 也完全相同）；`test_gemma_window_rows` 在带有篡改的随机步骤序列上比较增量观测与完整观测，以及增量行；`test_gemma_fastpath3` 覆盖 `--decode-fastpath3`（4.11 节）。`test_gemma_flags` 把这些选项放在一起运行：320 次启动（`full`、带与不带 rewriter 的 `goal_query`、`observation_query` 和 `random_fill`，各自配合 `--cuda-graphs`、第一条、前两条或全部三条解码快速路径、`--release-idle-pages` 和 `--admission-overlap 4` 的每一种组合，16 个 session、131,072 token）每次都只多出自己的启动键和 vLLM 参数，启动检查和 window worker 全部接受；显式给出 T 的 `--goal-rewriter-batch 4`（这些启动用的 32 GiB 没有默认值）在带 rewriter 的 `goal_query` 与 `--admission-overlap 4` 的 24 种组合中每次都只多出它的两个键（worker 启动记录不变），在那里不给 T、配合其他策略或不开 overlap 时被拒绝；单独的 `--decode-fastpath2`、没有 `--decode-fastpath2` 的 `--decode-fastpath3`、`--phase1-in-step`、`--gdn-device-handoff`、开 thinking 的 kv-session 协议和带 `--task` 列表的 chat-text 协议无论其他选项如何都被拒绝。在 CPU 替身上，`--cuda-graphs full` 加两条或全部三条快速路径的 `goal_query`（三个 session 一起解码；只有 phase 1 和 Goal 前向走 eager）、分别带两条和三条快速路径的分块 phase 1、在 8 个 `goal_query` session 上带三条快速路径的 `--admission-overlap 8 --release-idle-pages`，以及带三条快速路径的 `full` 和 `observation_query` 释放规则压力运行，每一轮都与不开这些选项的运行相同，释放和重新载入也相同。常规测试在 6 或 8 个 session 上运行这些端到端测试和 `test_gemma_release` 的测试（`goal_query`、`full`、`observation_query`、`random_fill`、overlap），并对 `goal_query` 运行分块的 phase 1；设 `REMOTE_RA_SLOW_TESTS=1` 时还运行 16 个 session、每个 5 轮、与从不释放的池对比的版本（上文 16 个第 1 轮 admission 全部停住的结果），`full` 和 `observation_query` 的分块 phase 1，以及每个策略和模式的 graph 与快速路径运行，约多 6 分钟。`test_gemma_goal_rewriter` 覆盖 `--goal-rewriter-batch`（上文），并检查未固定 rewriter 的服务不写 Goal rewriter 的任何用时字段。`test_gemma_kv_session` 覆盖 kv-session 的 Gemma 格式：重新实现的客户端、transcript 的几何与拒绝、启动记录和生成头检查、启动器（262,144 token 的 16 个 session、各种拒绝、`--host-memory-shares`），以及带 `request_pages` 的 vLLM 启动检查；使用固定 tokenizer 时，MemAgent 的每种轮次（读取、一次读两个 chunk、无效调用、格式错误、问题、补回的采样）都与模板逐 token 一致。`test_gemma_kv_session_e2e` 让该客户端的 MemAgent session 经窗口布局的 engine 和 worker 运行，开 `--release-idle-pages`、全局池取选择性规则的最小值（常规测试中 `observation_query` 和 `goal_query` 各 6 个 session；设 `REMOTE_RA_SLOW_TESTS=1` 时 16 个，并且每个回复都与从不释放的池相同，约 1.5 分钟）：每个 prompt 都是客户端的构造，Gemma 的几何成立，释放和重新载入不动窗口；另有三个 session 经打包的路由和带 `Gemma4SourceRenderer` 的前端 capture 运行。`test_gemma_chat_text` 覆盖 chat-text 的轮次格式（gemma-tb）：启动记录和模板检查；Terminus transcript 的每个 prompt 都等于参照渲染，去掉 thought channel 后等于 stock vLLM 的 prompt（用固定 tokenizer 和模板，也用替身）；chat-text transcript 复现第 0 阶段的 stock session；以及 Terminal-Bench 设置下的启动器和启动检查。`test_gemma_chat_text_e2e` 让 Gemma chat-text session 经打包的 router 和 chat-text 前端，在窗口布局的引擎和 worker 替身上运行，并开 `--release-idle-pages`：K = 20 时 16 个 session 一起解码，16 个 session 在选择性规则最小的全局池上运行（释放并重新载入空闲 session），摘要的 aside 和交接重建，回复上限和上下文拒绝，取消及其换绑，以及把窗口还给 pool 1 的关闭和驱逐；`REMOTE_RA_SLOW_TESTS=1` 再加 16 个 session × 8 个 episode 在释放压力下的摘要运行、压力下的 `random_fill` 和 `full`，以及与从不释放的池的比较。`test_gemma_chat_text_goal` 覆盖 chat-text 上的 `goal_query`（gemma-tb-goal）：带 Goal rewriter 和 `--admission-overlap 16` 的 Terminal-Bench 启动经过 dry run、启动器和启动检查（`chat_text_goal` 记录、Goal 瞬态块、`request_pages`、仍然保留的拒绝），28 GiB 下 131,072 token 的 16 个 session 和 65,536 token 的 K = 20 的内存规则及 vLLM 自己的启动估算，以及在窗口替身上以 K = 20、`--admission-overlap 16`、池压力下的 `--release-idle-pages` 和 `--decode-fastpath` 运行的 Terminus 2 `goal_query` 客户端，含摘要、aside、交接重建和一次被取消请求的重试：每个 prompt 精确，每轮的 Goal 输入和 Goal 来源与客户端所发的一致，每次 Goal 前向都在其瞬态块内，每个客户端 session 都与不带这些开关的服务相同（常规测试 4 个客户端，`REMOTE_RA_SLOW_TESTS=1` 时 16 个）；以及（integrate-1021）同一启动在 25 GiB、K = 17、`--goal-rewriter-batch 4` 和 token 上限 25 GiB 的默认值（25,600）下配合每条解码快速路径，不开和开 `--deferred-completion-io`，以及它的客户端在窗口替身上配合批量的进程内 rewriter 子进程、`--release-idle-pages`、`--decode-fastpath3` 和 `--deferred-completion-io`（第一条、前两条快速路径、不开该开关的三条快速路径和 16 个客户端需 `REMOTE_RA_SLOW_TESTS=1`）：每个 session 都与不带这些开关的服务相同，rewriter 调用按批执行，完成写入线程写的每个文件都与同步路径的相同；用固定 tokenizer 时，一个 Terminus 2 session 的 Goal probe 经过 `Gemma4SourceRenderer`。`test_gemma_chat_text_replay`（需显式开启，`REMOTE_RA_TB_ATIF_DIR`）用 Gemma 的轮次格式回放 8 月的 Terminus 2 轨迹（6.7 节）。`test_gemma_deferred_completion` 在 window 布局上运行 `--deferred-completion-io`（4.11 节）：启动器和启动检查，引擎和 worker 模拟上开关打开和关闭时逐字节相同的文件（一个、三个以及池压力下六个 session；`REMOTE_RA_SLOW_TESTS=1` 时 16 个），文件仍在写入时取消、admission 和关闭服务的顺序，以及基于 window 引擎、确认被延后的 chat-text、kv-session 和 mswea-xml API 服务。`test_gemma_forward4k` 覆盖 `--max-num-batched-tokens 4096`（见上文“4,096 token 的前向”）。

从选项正文中删去的说明（原文照录）：

- 已在一次 16 个任务的 GPU dry run 中运行，即作业 3560498，`TB16_FAIL_VERIFIED`；在它之前的 dry run 用的是 `--max-running-sessions 17`、65,536 token、8,192 token 的前向和 `--dram-arena-gib 64`
- 已在作业 3461663 和 3462014 中在 GPU 上运行
- （async-gemma；已在 GPU 上于 b = 1 部分检查：作业 3560784，`ASG1_PARTIAL_PASS_VERIFIED`，gg 和 gx 两对 token 完全一致，ge 对没有比较；此前的尝试失败了：作业 3519554 因基础设施停滞，作业 3549350 因客户端超时）
- GPU 上只检查了它的 token 对齐：GG1（作业 3583846，integrate-1112 `4ebff97`，`GG1_PASS_VERIFIED`）在 `chat-text` 上以 `full` 运行 strip 服务（两个 hold slot）和 keep 服务（`--enable-thinking`），eager 和 `--cuda-graphs full` 都有，与 stock vLLM 0.24.0 对比：比较过的每个 prompt 的 token id 都与 stock 相同（每个 strip 服务 45 个中 45 个，每个 keep 服务 6 个中 6 个）；它关于回复的读数 A1 和 A2 未达到。`mswea-xml`、`--kv-session-thinking on` 和各 benchmark 尚未在 GPU 上运行
- （Terminal-Bench 2.0 与 Terminus 2；GPU 上的 dry run：作业 3433706 和 3434544）
- 作业 3560498 之前的 dry run（作业 3433706 和 3434544）用的是 K = 17、65,536 token、25 GiB、8,192 token 的前向和 64 GiB 的 arena
- （gemma-tb-goal；它的 Terminal-Bench dry run 已在作业 3434544 中通过）
- 它已在 MemoryAgentBench 作业 3429507 中在 GPU 上与 Gemma 一起运行（见下）
- （262,144 token 时 K = 20；到目前为止的 dry run：K = 17）
- 因此在本版本之前，尽管启动器的选择性规则接受，它也会在启动时拒绝 32 GiB 以下、131,072 token 的 16 个 session（此后已有低于 32 GiB 的 16 session Gemma 服务在 GPU 上运行：作业 3429672、3429808、3429830 和 3462014），并在任何大小下拒绝 262,144 token 的 16 个 session。现在在选择性规则下，窗口布局固定
- （可选，gemma-forward4k；已在作业 3461663 和 3462014 中在 GPU 上运行，见下）
- 它也在作业 3429507、3429808 和 3434544 中与 Gemma 一起运行过
- （4.10 节；已在 `c07d42a` 上于作业 3538523 中在 GPU 上运行，`CMG1_PASS_VERIFIED`，gref / govl 和 govl / gvll 均为 EXACT；此前作业 3508345（`CMG1_FAIL_VERIFIED`）中它的服务始终没有 READY，权重没能在启动期限之前加载完，原因尚未查明）
- （4.10 节；gemma-rewriter-batch；已在 MemoryAgentBench 作业 3429507 中在 GPU 上运行：N = 4，25 GiB 时的默认 T = 25,600，40 次 rewriter 调用，没有回退，没有显存不足）
- 这是估计：25 GiB 的默认值在作业 3429507 中以 8 个 session 运行过，16 个 session 的 SWE-bench 和 K = 17 的 Terminal-Bench 运行显式给出至少留出 2 GiB 的 T：15,360 和 20,480）
- （GPU 上：作业 3410750、3429503、3429507 和 3429673）
- （Gemma 上 `--decode-fastpath3` 在作业 3429673 中与 `--decode-fastpath2` 的 token 和证据完全相同；benchmark 运行三条都开）
- GPU 上在作业 3429673 中与 `--decode-fastpath2` 的 token 和证据完全相同（4.11 节）
- （4.11 节；GPU 上在作业 3429673 中 token 和证据完全相同）
- （gemma-mab；已在作业 3429503 和 3429507 中在 GPU 上端到端运行）

## 出自 4.15 节：默认开启的速度选项

移自 docs/reference.zh-CN.md 的 [4.15 节](../reference.zh-CN.md#415-默认开启的速度选项)。

从选项正文中删去的说明（原文照录）：

- （4.13 节；Qwen 的 SWE-bench 服务不带它）
- 在 Qwen 上它已在 GPU 上于 b = 1 检查过：token 完全一致（作业 3532746，runtime `4ed1013`，`ASG1_PARTIAL_PASS_VERIFIED`：b = 1 的每一对都完全一致，部分通过只是因为比较器按值比较了一个进程级的身份 token `state_digest`）。在 Gemma 4 profile 上它已在 GPU 上于 b = 1 部分检查（runtime `4ed1013`）：作业 3560784（`ASG1_PARTIAL_PASS_VERIFIED`）中 gg 和 gx 两对 token 完全一致，ge 对没有比较（其同步服务因作业的时间预算被跳过）。第一次尝试因基础设施停滞而失败（作业 3519554）；第二次（作业 3549350，`ASG1_FAIL_VERIFIED`）在 b = 1 时每个比较过的请求都 token 完全一致，但每个 ge / gg 服务 97 个请求中有 12 个在 harness 的客户端超时。
