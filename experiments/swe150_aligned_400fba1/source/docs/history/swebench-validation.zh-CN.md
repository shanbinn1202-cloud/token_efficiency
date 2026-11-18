# 截至 2026-11-16 的 SWE-bench Verified 验证状态（冻结）

[English](swebench-validation.md) | 简体中文

> **2026-11-16 冻结**（在 `e6d0c88`）：原来 [docs/SWEBENCH_BENCHMARK.zh-CN.md](../SWEBENCH_BENCHMARK.zh-CN.md#验证状态) 的“验证状态”一节，由 docs-reorg-1116 原样移到这里，只改了相对链接；不再更新。“A.x”和“B.x”指该页的各节；“Part A”是该页正式记录的配置（thinking 关闭、32 GiB、131,072 token）。下面有些句子在冻结时已经过时（例如 `recent_tools`“仅在 CPU 上测试”）。每次运行的当前状态见 [docs/status.zh-CN.md](../status.zh-CN.md)。

本配置已在一张 GH200 上用 `goal_query` 运行过：runtime `5b15b2f`，前 32 个 instance，16 个 session，2026-10-18（美国中部时间）：

- 作业 3404696：`goal_query`，rewriter 不做批处理。验证结果 `SWE32_PASS_VERIFIED`，合并结果 `JOIN_PASS`；32 个 instance 全部完成，解决 19 个。
- 作业 3404230：`goal_query` 加 `--goal-rewriter-batch 4`（A.3）。验证结果 `SWE32_PASS_VERIFIED`，合并结果 `JOIN_PASS`；32 个 instance 全部完成，解决 20 个。

解决数来自服务端记录与评测结果经过验证的合并（join）。`full`、`observation_query`、`random_fill` 和 `stock` 没有在 `5b15b2f` 上运行。

在 runtime `120bc3c`（integrate-1019-tbgoal；其 Qwen 代码路径与 `9ef0d6d` 相同，`53e3b7a`（2026-10-22）在 `9ef0d6d` 之上只加了 A.2 所述的 Goal 特征修复）上，以 A.3 的参数（三个解码快速路径和 `--deferred-completion-io`）、前 32 个 instance、16 个 session 运行过：

- 作业 3428631：`goal_query` 加 `--goal-rewriter-batch 4`，每批 token 上限为默认值（16,384）。32 个 instance 全部完成，解决 19 个；rewriter 调用 320 次；7 个批次显存不足，都通过拆分恢复（没有 Goal 因显存而 fallback）；322 次 Goal 重写中 Goal fallback 62 次（60 次调用失败，另有 2 次在调用之前就超出请求上限）。作业自身的判定 `SWE32_FAIL` 来自测试工具的一项检查（goal-async）：它当时还不接受 `--deferred-completion-io` 在请求文件写完之前暂缓发出的 Cancel 确认（[docs/reference.zh-CN.md 4.11 节](../reference.zh-CN.md#411-解码快速路径默认开启)）；那个请求本身服务正确。用修正后的检查，整份日志都通过；用作业自己的代码离线验证，每项检查都通过（`SWE32_FAIL_VERIFIED`，合并结果 `JOIN_PASS`）。
- 作业 3429500：`observation_query`。验证结果 `SWE32_PASS_VERIFIED`；32 个 instance 全部完成，解决 21 个。
- 作业 3429193：A.4 的 stock 对照（不运行本 runtime 的任何代码）。验证结果 `SWE32_PASS_VERIFIED`；解决 20 个。

输出吞吐（A.10；定义仍是暂定的）：`goal_query` 140.9、`observation_query` 226.1、stock 对照 243.7 tokens/s，分别是 stock 的 58 % 和 93 %；两个策略之间的差距主要是 session 等待 Goal rewriter 的时间。`full` 和 `random_fill` 还没有在 `120bc3c` 上运行。

Qwen 上的解码检查（作业 3429022，runtime `120bc3c`；离线核验为 `SPD3_PARTIAL_PASS_VERIFIED`：每个 runtime 服务都通过，只是部分轮次缺少 stock 服务的测量窗口，所以是部分通过）：16 个并发 session 时，任何配置在两次运行之间都不是逐字节相同的（A.13），所以这两个开关是在 prompt 逐字节相同的请求上比较的。`--decode-fastpath3` 对照 `--decode-fastpath2` 时 76–83 % 的回复相同，`--deferred-completion-io` 开对照关时 76–80 % 相同，而同一配置只切换 step profiler 时为 81–82 %；差异集中在 8 个及以上并发 session 的轮次中，在 1 到 4 个 session 的轮次中，比较的同 prompt 回复只在两组中各有一条不同，都在一个 2 个 session 的轮次中（不开 profiler 时 `--decode-fastpath3` 对照 `--decode-fastpath2`；`observation_query` 下的 `--deferred-completion-io`）。这两个开关没有带来超出这一运行间波动的差异（[docs/reference.zh-CN.md 4.11 节](../reference.zh-CN.md#411-解码快速路径默认开启)）。

Gemma 4 31B IT（A.14），runtime `9ef0d6d`，前 32 个 instance，25 GiB 下 16 个 session：

- 作业 3429808：`goal_query` 加 `--goal-rewriter-batch 4 --goal-rewriter-batch-tokens 15360`。验证结果 `GSWE32_PASS_VERIFIED`；32 个 instance 全部完成，解决 15 个；合并结果 `JOIN_PASS`，32 个全部 token 级一致（合并时的重新渲染改用 Gemma 4 renderer 之后）。958 次调用，rewriter 调用 179 次（180 次 Goal 重写，其中一次在调用之前就 fallback），Goal fallback 32 次，容量等待 21,121 次；输出吞吐 58.5 tokens/s（按 runner 的时间跨度）。
- 作业 3429830：`observation_query`。经下文的重新核验后为 `GSWE32_PARTIAL_VERIFIED`：没有未通过的检查；只因为 pool 的 2,760 秒时间上限截掉了 32 个 instance 中的 7 个才是部分通过；完成 25 个，解决 17 个；合并结果 `JOIN_PASS`，25 个 token 级一致；容量等待 12,178 次；finding `CANCELLED_ADMISSION`、`CANCELLED_IN_BACKOFF`（2 个请求）、`IDLE_RELEASE_ROWS_DIFFER` 和 `WINDOW_CLOSE_DELTA`。按服务端统计的全部生成 token（203,602 个，其中 74,989 个属于被截掉的 7 个 instance，它们的 runner 没有留下轨迹）除以 runner 的时间跨度 2,755 秒，输出吞吐为 73.9 tokens/s。合并结果里的 46.7 tokens/s 只把完成的 25 个 instance 的 token（128,613 个）除以这个跨度，而这个跨度也包含被截掉的 instance 的时间（2026-11-10 根据该作业的证据重新计算）。两个数都不能与作业 3429808 的 58.5 直接比较：本次运行在时间上限处结束时还有 7 个 session 在跑，没有只剩少数 session 的尾段（平均 14.0 个 agent 同时运行，对照 10.2 个）；按 agent 秒计，两个服务分别生成 5.26 和 5.73 个 token（各运行一次）。

作业 3429830 自身的判定是 FAIL，来自测试工具的几项检查：它们当时还不接受客户端在 admission 期间取消的请求（这里是在 pool 的时间上限时取消；服务端结束了该 admission 并做了补偿）。这些检查已经修正：只有在有正面证据时（补偿成功、没有 runtime 行、引擎的取消回执或容量拒绝、匹配的断开 failure 行），这样的请求才算干净，并报告为 finding `CANCELLED_ADMISSION`；归档的证据已离线重新核验，没有在 GPU 上重跑。

后来的运行，前 32 个 instance、16 个 session；Qwen 服务使用发布仓库的设置 `--kv-cache-gib 25` 和 `--max-model-len 262144`（[benchmarks/swe-bench.zh-CN.md](../benchmarks/swe-bench.zh-CN.md)）：

- 作业 3461703（runtime `cf2b28d`，integrate-1026）：`goal_query` 加 `--goal-rewriter-batch 4`，以及 `observation_query`。验证结果 `SWE32_PASS_VERIFIED`，合并结果 `JOIN_PASS`；每个服务的 32 个 instance 全部完成，分别解决 19 个和 20 个。
- 作业 3507294：25 GiB、262,144 token 下的 stock 对照。验证结果 `SWE32_PASS_VERIFIED`；解决 21 个。作业 3526362 与之相同，另加 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`：`SWE32_PASS_VERIFIED`，解决 21 个。另外两次 stock 运行不完整，为 `SWE32_PARTIAL_VERIFIED`：作业 3506513，以及作业 3525442（16 个 instance，使用 vLLM 的 CPU KV offload）。
- 作业 3500565（runtime `c07d42a`）：使用 vLLM Goal 引擎（`--goal-rewriter-engine vllm`，实验性）的 `goal_query`。验证结果 `SWE32_FAIL_VERIFIED`（未通过的检查为 capture-sizes 和 context-limits；finding `CUDA_GRAPH_CONFIG_DIFFERS` 和 `GOAL_FALLBACK`）；解决 20 个；未通过验证。
- Gemma：作业 3462014（runtime `cf2b28d`）以 4,096 token 的前向运行了 `observation_query`（A.14）：`GSWE32_PASS_VERIFIED`，32 个完成，解决 18 个。

自 `cf2b28d` 起，Qwen 的 SWE-bench 运行都使用这个 25 GiB、262,144 token 的设置。Part A 是 `5b15b2f` 和 `120bc3c` 上那些运行所用的 32 GiB、131,072 token 配置。

在 `5b15b2f` 上的运行之前，本配置的以下部分已在一张 GH200 上运行过（README）：

- 一个服务上 16 个 session：作业 3339019 和 3343069（合成 session；[docs/running.zh-CN.md 4.4 节](../running.zh-CN.md#44-一个服务上的多个-session)）。
- session 池（`--max-running-sessions`）：作业 3343069（24 个任务、K = 8，合成 session；[docs/running.zh-CN.md 4.4 节](../running.zh-CN.md#44-一个服务上的多个-session)）。
- CUDA graph：作业 3338893（`SCG2_FAIL_VERIFIED`：其 plan 的步骤没有完成，finding `STEPS_INCOMPLETE`）、3339448（`SCG2_PASS_VERIFIED`）和 3343069（`SINT1_PASS_VERIFIED`），其中作业 3343069 以 16 个 session 运行 `--cuda-graphs full`（[docs/reference.zh-CN.md 4.8 节](../reference.zh-CN.md#48-cuda-graph)）。
- 131072 token 的上下文、4096 token 的回复和 `--dram-arena-gib 88`，配合 16 个 `observation_query` session、`--active-token-budget 8192`、`--cuda-graphs full` 和 32 GiB GPU KV：作业 3348705（[docs/reference.zh-CN.md 4.9 节](../reference.zh-CN.md#49-上下文上限与回复上限)）。
- `--admission-overlap 16` 配合 `goal_query`、Qwen3-0.6B Goal rewriter 和 16 个 session：作业 3349504（[docs/reference.zh-CN.md 4.10 节](../reference.zh-CN.md#410-admission-优化需显式开启)）。
- Mini-SWE-Agent 2.4.2 对接本 runtime：10 个 Astropy instance 的 pilot，每个策略一个作业，每个服务 5 个 session，步数上限 90，回复 1536 token（作业 3313802、3314010、3314319、3314623；[docs/reference.zh-CN.md 4.7 节](../reference.zh-CN.md#47-16-token-page-的验证)）。
- runtime `e877051` 上的 `full` 和 `random_fill`：32 个 instance、16 个 session、32 GiB、131,072 token、4096 token 的回复、步数上限 120，以及早先的 88 GiB arena（A.3“主机内存”）：作业 3357655（`full`，解决 22 个）和 3360287（`random_fill`，解决 19 个），都是 `SWE16_PASS_VERIFIED`。

尚未运行：`120bc3c` 或更新 commit 上 thinking 关闭的 `full` 和 `random_fill`（开启 thinking 时 `full` 在 `ba4d1ba` 上运行过：作业 3579273，`SWE32_FAIL_VERIFIED`，以及带 `--release-idle-pages` 的作业 3583081，`SWE32_PASS_VERIFIED`）；thinking 关闭的 `recent_fill` 和 `akv_rule`（开启 thinking 时：作业 3539242，`60c5d21` 上的 `recent_fill`，`SWE32_PASS_VERIFIED`；作业 3577833，`ba4d1ba` 上的 `akv_rule`，`SWE32_PARTIAL_VERIFIED`）；SWE-bench 上的 `recent_tools`（仅在 CPU 上测试；它开启 thinking 的运行，作业 3540110，在开始之前就被取消了）；带 `MAX_RUNNING`、`STEP_LIMIT`、`MAX_TOKENS` 或 `STOCK=1` 的 `run_agents.sh`，以及 `evaluate.sh`（[docs/benchmarks/swe-bench-mini-swe-agent.zh-CN.md 7.4 节](../benchmarks/swe-bench-mini-swe-agent.zh-CN.md#74-运行一个组)；上面各次运行用的是其自身测试工具的 agent 池和评测脚本）。B.5 的 Slurm 脚本、B.3 的 gold patch 检查和 chat template 检查、B.7 的停止命令和 B.9 的吞吐脚本都未经测试。除上面这些运行外，本文档不报告任何结果。
