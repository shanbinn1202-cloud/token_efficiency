# 状态

[English](status.md) | 简体中文

唯一随 GPU 作业变化的页面：`main` 是哪个提交、哪个提交是稳定的、当前配置，以及按模型、benchmark 和策略列出的每个 GPU 作业，带它的判定、数字是否经过审计和离线核验的日期。其他页面都指向这里。限制见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md)，一个数字必须满足的规则见 [docs/release-gates.zh-CN.md](release-gates.zh-CN.md)。截至 runtime `cf2b28d` 的验证记录（含请 Goal 负责人审阅的说明）冻结在 [docs/history/validation-cf2b28d.zh-CN.md](history/validation-cf2b28d.zh-CN.md)；其他页面到 2026-11-16 为止记载的作业叙述也冻结在 [docs/history/](history/) 下（README 的[文档表](../README.zh-CN.md#文档)列出了它们）。docs/ 下任一页面或 README 提到的 33xxxxx 及之后的每个 DeltaAI 作业在这里都有一行，没有核验记录的作业也在内（并写明这一点）；更早的作业（干净安装验收 3270721 及其之前的运行）只在提到它们的页面上描述，[历史](history.zh-CN.md#历史)是其中之一。更新于 2026-11-17。

## `main` 与稳定提交

- **`main`** 是 2026-11-17 与之前的 `main` `7892db3` 合并的 integrate-1117：runtime `2fb7f62`（integrate-1116 `7a8c36e` 的 runtime 加上 session 池的任务上限 256，`06c1e90`，以及 `examples/swebench/verified-150.txt`）；`2fb7f62` 的 release check：2,287 个测试，跳过 141 个。其中有些部分还没有在 GPU 上跑过（[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 5.7）。之前的 `main` `7892db3`（runtime `4ebff97`）是 tag `main-integrate-1112`。
- **稳定 tag：** 暂无。还没有提交通过全部六个模型 × benchmark 格；最接近的是 runtime `cf2b28d`（它自己通过了三格：作业 3461703、3461661、3462014）。
- **稳定配置**（Qwen 开启 thinking 跑 SWE-bench Verified）：runtime `ba4d1ba`，作业 3576565（README 中的命令）。

## 当前配置

即将进行的 64 个任务的运行所用的配置（2026-11-16 决定）。三个 benchmark 页面的正式配置（A 部分，都是 thinking 关：SWE-bench 为 32 GiB 和 131,072 token，MemoryAgentBench 为 32 GiB 和 262,144 token，Terminal-Bench 为 25 GiB 和 262,144 token）是历史记录。

| 设置 | 值 |
| --- | --- |
| 模型 | Qwen3.6-27B，只用文本部分 |
| Thinking | 开，keep 模式（`--enable-thinking`） |
| 调度 | 异步（`--async-scheduling`） |
| KV cache | 25 GiB（`--kv-cache-gib 25`），16 token 的 Page，解耦布局 |
| 上下文、回复 | 262,144 token，4,096 token 的回复 |
| Session | 每个服务 16 个；并发 24 是单独计划的扫描（XML 和 kv-session 协议上超过 16 个还没有在 GPU 上运行过；Terminal-Bench 的 dry run 在更早的 runtime 上跑过 20 个 chat-text session：[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 2.1） |
| 预算 | 每个 benchmark 都是 8,192 token（`--active-token-budget 8192`）；MemoryAgentBench 目前不打分 |
| 速度开关 | 默认开启的速度选项（CUDA graph `full`、三条解码快速路径、`--deferred-completion-io`、`--language-model-only`、1 GiB 的 connector 池） |

## 按模型、benchmark 和策略的状态

每个作业和策略一行。**判定**：作业 harness 的判定，除非该行另有说明，都已离线核验（`..._VERIFIED`）。**审计**：该行的数字已从原始证据独立重算时写 `是（日期）`；只经过作业 harness 的离线核验器检查时写 `仅核验器（未审计）`（历史中 `7b70e14` 的文档审计没有点名任何一行，所以没有一行按它标为已审计）。**日期**：作业离线核验的日期。[docs/release-gates.zh-CN.md](release-gates.zh-CN.md#3-一个数字什么时候可以用) 第 3 节的三项参照检查把关 integrate-1116 及之后的 runtime 上作业的数字（第二项，full 与 stock 的比较，目前推迟）；除了 integrate-1116 的三个功能性检查（作业 3613280、3613408 和 3613537，不报告 benchmark 数字）之外，下面每一行都是更早的作业，没有经过这些检查（keep-rebuild-1116 之前 thinking 开的运行，在 session 第一次格式错误之后通不过 keep 渲染检查，见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 1.4）。

### Qwen3.6-27B，SWE-bench Verified（前 32 个实例，16 个 session）

| 策略 | 作业 | 提交 | Thinking | 判定 | 解决数 | 审计 | 日期 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `observation_query` | 3461703 | `cf2b28d` | 关 | `SWE32_PASS_VERIFIED` | 20 / 32 | 仅核验器（未审计） | 2026-10-30 |
| `goal_query` | 3461703 | `cf2b28d` | 关 | `SWE32_PASS_VERIFIED` | 19 / 32 | 仅核验器（未审计） | 2026-10-30 |
| `full`（`--release-idle-pages`） | 3583081 | `ba4d1ba` | 开 | `SWE32_PASS_VERIFIED` | 21 / 32 | 是（2026-11-16） | 2026-11-12 |
| `observation_query` | 3539260 | `60c5d21` | 开 | `SWE32_PASS_VERIFIED` | 21 / 32 | 是（2026-11-16） | 2026-11-10 |
| `recent_fill` | 3539242 | `60c5d21` | 开 | `SWE32_PASS_VERIFIED` | 22 / 32 | 是（2026-11-16） | 2026-11-08 |
| `goal_query`（稳定配置） | 3576565 | `ba4d1ba` | 开 | `SWE32_PASS_VERIFIED` | 19 / 32 | 是（2026-11-16） | 2026-11-11 |
| `akv_rule` | 3577833 | `ba4d1ba` | 开 | `SWE32_PARTIAL_VERIFIED`（一个实例在时间上限处被截断） | 20 / 32 | 是（2026-11-16） | 2026-11-12 |
| `goal_query_markers` | 3602728 | `325cdc5`（integrate-1114） | 开 | `SWE32_PASS_VERIFIED` | 20 / 32 | 是（2026-11-16） | 2026-11-14 |
| `goal_query` 加 `--token-gather markers` | 3603443 | `2b2a752`（integrate-1114） | 开 | `SWE32_PASS_VERIFIED` | 21 / 32 | 是（2026-11-16） | 2026-11-14 |
| `ledger_query` | 3611601 | `51b4085`（ledger-int-1112） | 开 | `SWE32_PASS_VERIFIED` | 21 / 32 | 仅核验器（未审计） | 2026-11-16 |
| `recent_fill`，`--async-scheduling`（rfillasync32ti） | 3611917 | `4ebff97` | 开 | `SWE32_PASS_VERIFIED` | 22 / 32 | 仅核验器（未审计） | 2026-11-16 |
| `recent_fill`，同步对照（rfillsync32ti） | 3612333 | `4ebff97` | 开 | `SWE32_PASS_VERIFIED` | 20 / 32 | 仅核验器（未审计） | 2026-11-16 |
| `goal_query` 加 `--goal-rewriter-engine vllm` 和 `--goal-overlap`（vllm25，25 GiB） | 3500565 | `c07d42a` | 关 | `SWE32_FAIL_VERIFIED`（`CUDA_GRAPH_CONFIG_DIFFERS`、`GOAL_FALLBACK`）；失败的检查 `capture-sizes` 和 `context-limits` 把 vLLM Goal 子引擎的日志行当成了主引擎的：用修正后的检查重新评估，只剩 `GOAL_FALLBACK`（harness 快照 perf-swe32，仅供参考） | 20 / 32 | 仅核验器（未审计） | 2026-11-02 |
| stock vLLM 0.24.0（`reasoning_dropped`） | 3583822 | — | 开 | `SWE32_PASS_VERIFIED` | 21 / 32 | 是（2026-11-16） | 2026-11-12 |
| `full`（不带 `--release-idle-pages`） | 3579273 | `ba4d1ba` | 开 | `SWE32_FAIL_VERIFIED`（服务在准入门控处停滞了 55.6 分钟，[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 2.4） | 21 / 32 | 仅核验器（未审计） | 2026-11-11 |
| stock vLLM 0.24.0（25 GiB，262,144 token） | 3507294 | — | 关 | `SWE32_PASS_VERIFIED` | 21 / 32 | 仅核验器（未审计） | 2026-11-03 |
| stock vLLM 0.24.0 加 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0`（25 GiB，262,144 token） | 3526362 | — | 关 | `SWE32_PASS_VERIFIED` | 21 / 32 | 仅核验器（未审计） | 2026-11-06 |
| stock vLLM 0.24.0（25 GiB，262,144 token） | 3506513 | — | 关 | `SWE32_PARTIAL_VERIFIED` | 14 / 32 | 仅核验器（未审计） | 2026-11-02 |
| stock vLLM 0.24.0 加 vLLM 的 CPU KV offload（16 个实例） | 3525442 | — | 关 | `SWE32_PARTIAL_VERIFIED` | 8 / 16 | 仅核验器（未审计） | 2026-11-04 |
| `goal_query`（A 部分：32 GiB，131,072 token） | 3404696 | `5b15b2f` | 关 | `SWE32_PASS_VERIFIED` | 19 / 32 | 仅核验器（未审计） | 2026-10-19 |
| `goal_query` 加 `--goal-rewriter-batch 4`（A 部分） | 3404230 | `5b15b2f` | 关 | `SWE32_PASS_VERIFIED` | 20 / 32 | 仅核验器（未审计） | 2026-10-19 |
| `goal_query` 加 `--goal-rewriter-batch 4`（A 部分，开三条解码快速路径和 `--deferred-completion-io`） | 3428631 | `120bc3c` | 关 | `SWE32_FAIL_VERIFIED`（harness 的一项检查 goal-async 当时还不接受 `--deferred-completion-io` 在请求的文件写完之前一直保留的 Cancel 确认；该请求的服务是正确的） | 19 / 32 | 仅核验器（未审计） | 2026-10-21 |
| `observation_query`（A 部分） | 3429500 | `120bc3c` | 关 | `SWE32_PASS_VERIFIED` | 21 / 32 | 仅核验器（未审计） | 2026-10-22 |
| stock vLLM 0.24.0（A 部分） | 3429193 | — | 关 | `SWE32_PASS_VERIFIED` | 20 / 32 | 仅核验器（未审计） | 2026-10-22 |
| `full`（32 GiB，131,072 token，4,096 token 的回复，步数上限 120，88 GiB 的 arena） | 3357655 | `e877051` | 关 | `SWE16_PASS_VERIFIED` | 22 / 32 | 仅核验器（未审计） | 2026-10-11 |
| `random_fill`（同样的设置） | 3360287 | `e877051` | 关 | `SWE16_PASS_VERIFIED` | 19 / 32 | 仅核验器（未审计） | 2026-10-12 |
| `goal_query`（同一个作业 harness 和提交） | 3363871 | `e877051` | 关 | `SWE16_PASS_VERIFIED` | 19 / 32 | 仅核验器（未审计） | 2026-10-12 |
| `random_fill`、`recent_tools` | — | | 开 | 未运行（`recent_tools`：作业 3540110 在开始之前就被取消；没有核验记录） | | | |

keep-rebuild-1116 之前 thinking 开的 runtime 运行，在每个 session 第一次格式错误之后都丢失了之前的 reasoning（[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 1.4）。这些运行之后，integrate-1116 改变了两个策略的选择，都还没有在 GPU 上运行：`ledger_query` v2（作业 3611601 跑的是 v1；[docs/policy-registry.zh-CN.md](policy-registry.zh-CN.md#8-query-钩子与-ledger_queryledger-policy-1112) 第 8 节），以及 `goal_query` 和 `goal_query_markers` 从 session 第一次格式错误重试起的 Goal（已知限制 1.3）。`recent_tools` 还没有在 thinking 开时运行过；自 ledger-fixes-1116 起，它对 thinking 开、在 `</think>` 之前被截断的回复的排序也变了（这类回复的页是 REASONING 而不是 CALL；[docs/policy-registry.zh-CN.md](policy-registry.zh-CN.md#8-query-钩子与-ledger_queryledger-policy-1112) 第 8 节）。异步对照（同一提交、同一 payload，只有异步开关不同）：输出 267.9 对 225.7 token/s（+18.7 %），每分钟轮次 +21 %，每个批大小的稳态解码步是同步的 0.892 倍（约快 11 %）；两次的轨迹已经分叉（生成 541K 对 630K token）；未审计。稳定配置的运行，即作业 3576565：从第一个实例开始到最后一个实例结束共 46.3 分钟，在这段时间内输出 238.6 token/s（来自它的核验记录中的表格；未审计）。`4ebff97` 的 Qwen 冒烟（smoket，作业 3586847）已于 2026-11-16 取消：未在 `4ebff97` 上运行（推迟）。thinking 开的小规模 `akv_rule` 运行（作业 3549065：32 个任务中的 8 个，4 个 session）在开始之前就被取消（没有核验记录）。stock 各行不运行本 runtime 的任何代码（没有提交）；A 部分各行是 [docs/SWEBENCH_BENCHMARK.zh-CN.md](SWEBENCH_BENCHMARK.zh-CN.md) 的正式配置，其叙述见 [docs/history/swebench-validation.zh-CN.md](history/swebench-validation.zh-CN.md)。

### Qwen3.6-27B，Terminal-Bench 2.0（16 个任务的 dry run）

| 策略 | 作业 | 提交 | Thinking | 判定 | 解决数 | 审计 | 日期 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `full`（32 GiB） | 3398129 | `0d37346` | 关 | 归档的核验为 `NOT_VERIFIED`（`pre-claim-gates` 失败）；引擎崩溃：一次解码拿走了最后一个空闲块，vLLM 抢占了它（`68d19b4` 的容量门控修复了这一点） | 16 个中 9 个（未核验） | 未审计 | 2026-10-17 |
| `full`（K = 20，32 GiB，65,536 token，`--release-idle-pages`） | 3399075 | `68d19b4` | 关 | `TB16_FAIL`（两项 harness 检查，之后已修正）；它归档的离线核验为 `NOT_VERIFIED` | 16 个中 9 个（未核验） | 未审计 | 2026-10-18 |
| `observation_query`（K = 20，32 GiB，65,536 token） | 3399437 | `68d19b4` | 关 | `TB16_PASS_VERIFIED` | 16 个中 8 个 | 仅核验器（未审计） | 2026-10-18 |
| `goal_query`（K = 20，32 GiB，65,536 token） | 3429670 | `120bc3c` | 关 | `TB16_FAIL_VERIFIED`（服务卡在 Goal 特征中；`53e3b7a` 已修复） | 16 个中 8 个 | 仅核验器（未审计） | 2026-10-22 |
| `goal_query`（K = 20，32 GiB，65,536 token） | 3434533 | `53e3b7a` | 关 | `TB16_PASS_VERIFIED` | 16 个中 8 个 | 仅核验器（未审计） | 2026-10-24 |
| `observation_query`、`goal_query`（K = 20，262,144 token） | 3461661 | `cf2b28d` | 关 | `TB16_PASS_VERIFIED` | 16 个中 9 个和 8 个 | 仅核验器（未审计） | 2026-10-26 |
| `observation_query`、`goal_query`、stock（`--thinking-history strip`） | 3499759 | `8ca8511` | 开 | `TB16_PASS_VERIFIED`（`stripgoal262` 上有 finding `GOAL_FALLBACK`） | 16 个中 8、6、8 个 | 仅核验器（未审计） | 2026-11-02 |

`4ebff97` 上 thinking 开加 `--async-scheduling` 的冒烟（smokeqt，作业 3586720）已于 2026-11-16 取消：未在 `4ebff97` 上运行（推迟）。65,536 token 的 dry run 的吞吐，来自它们的核验记录（未审计）：169.1、181.03 和 177.63 token/s（作业 3399075、3399437、3434533）。

### Qwen3.6-27B，MemoryAgentBench（kv-session；不打分）

| 策略 | 作业 | 提交 | Thinking | 判定 | 审计 | 日期 |
| --- | --- | --- | --- | --- | --- | --- |
| `observation_query`（G4，32 个 case） | 3405844 | `f9322a6` | 关 | `PMAB_PARTIAL_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-19 |
| `goal_query`（G4） | 3405845 | `f9322a6` | 关 | `PMAB_FAIL_VERIFIED`；用修正后的 route-B 检查为部分通过 | 仅核验器（未审计） | 2026-10-19 |
| `observation_query`、`random_fill`、`goal_query`（`--async-scheduling`） | 3568311 | `ba4d1ba` | 开 | `PMAB_FAIL_VERIFIED`：崩溃：一个回复在另一个请求解码时被 `max_tokens` 截断，异步 resolve 失败（`4ebff97` 已修复） | 仅核验器（未审计） | 2026-11-12 |
| `observation_query`、`random_fill`、`goal_query`（`--async-scheduling`） | 3584105 | `4ebff97` | 开 | `PMAB_PASS_VERIFIED` | 仅核验器（未审计） | 2026-11-14 |
| `observation_query`（`--async-scheduling`，16 个探测轮次中 16 个跑到 async-cut 情况） | 3604146 | `4ebff97` | 开 | `PMAB_PASS_VERIFIED` | 仅核验器（未审计） | 2026-11-15 |

更早的运行，来自[验证记录](history/validation-cf2b28d.zh-CN.md)：G1 作业 3397844 为 `PMAB_PASS`，G2 作业 3397842 因 harness 的一项检查为 `PMAB_FAIL`，重新判定为通过（两者的归档核验用的都是更晚的 harness，结果都为 `NOT_VERIFIED`）；G4 作业 3405855 为 `PMAB_FAIL_VERIFIED`，离线重新判定为 `PMAB_PASS`。

### Gemma 4 31B IT

| Benchmark | 策略 | 作业 | 提交 | 判定 | 结果 | 审计 | 日期 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SWE-bench Verified | `observation_query`（4,096 token 的前向，`--max-num-batched-tokens 4096`；服务 obs4k） | 3462014 | `cf2b28d` | `GSWE32_PASS_VERIFIED` | 18 / 32 | 仅核验器（未审计） | 2026-10-30 |
| SWE-bench Verified | `goal_query` | 3429808 | `9ef0d6d` | `GSWE32_PASS_VERIFIED` | 15 / 32 | 仅核验器（未审计） | 2026-10-24 |
| SWE-bench Verified | `observation_query` | 3429830 | `9ef0d6d` | `GSWE32_PARTIAL_VERIFIED`（用修正后的“准入期间被取消的请求”检查重新核验；作业自己的判定是 FAIL；部分通过是因为池的 2,760 秒上限截断了 32 个实例中的 7 个） | 17 / 32（完成 25 个） | 仅核验器（未审计） | 2026-10-24 |
| Terminal-Bench 2.0，65,536 token | `observation_query` | 3433706 | `9ef0d6d` | `TB16_PASS_VERIFIED`（重新核验） | 16 个中 4 个 | 仅核验器（未审计） | 2026-10-24 |
| Terminal-Bench 2.0，65,536 token | `goal_query` | 3434544 | `53e3b7a` | `TB16_PASS_VERIFIED`（重新核验） | 16 个中 5 个 | 仅核验器（未审计） | 2026-10-24 |
| Terminal-Bench 2.0，262,144 token | `observation_query`、`goal_query`、`random_fill`、stock | 3560498 | `f5a5c36` | `TB16_FAIL_VERIFIED`（`goal_query` 为 `CAPACITY_EXCEEDED`） | 16 个中 4、4、3、3 个 | 仅核验器（未审计） | 2026-11-15 |
| MemoryAgentBench（8 个 case，8 个 session） | `observation_query` | 3429503 | `9ef0d6d` | `PMAB_PASS_VERIFIED` | 不打分 | 仅核验器（未审计） | 2026-10-22 |
| MemoryAgentBench | `goal_query` | 3429507 | `9ef0d6d` | `PMAB_PASS_VERIFIED` | 不打分 | 仅核验器（未审计） | 2026-10-22 |
| MemoryAgentBench（`--async-scheduling`，thinking 关） | `goal_query`、`observation_query`、`random_fill` | 3601174 | `4ebff97` | `PMAB_PASS_VERIFIED` | 不打分 | 仅核验器（未审计） | 2026-11-15 |

`4ebff97` 的 Gemma SWE-bench 冒烟（smokea，作业 3586987）已于 2026-11-16 取消：未在 `4ebff97` 上运行（推迟）。作业 3560498 中，`goal_query` 服务的 Goal 还在它 219 个轮次中的 2 个上回退（finding `GOAL_FALLBACK`），对账还标出一个在 agent 超时处被截断的 `random_fill` trial：服务回答了当时在途的请求，而客户端从未收到（服务端比客户端调用多一行）；仅核验器（未审计）。

### 尚未运行

还没有任何 GPU 作业在任何提交上跑过的内容，按 benchmark 列出（截至 2026-11-16；上面已取消的 `4ebff97` 冒烟除外）：

- **SWE-bench Verified。** Qwen 开启 thinking：`random_fill` 和 `recent_tools`（见上面的一行）。Gemma 4：`full`、`random_fill` 和 stock 对照（只跑过 `observation_query` 和 `goal_query`）。
- **Terminal-Bench 2.0。** 正式测试的整体（全部 89 个任务；只跑过 16 个任务的 dry run）。Qwen：262,144 token 下的 `full`、`random_fill`，以及 262,144 token 下 thinking 关的 stock 对照（在那里它只在 thinking 开时跑过，作业 3499759）。Gemma 4：GG2（在 Terminal-Bench 上用 `--thinking-history strip`）没有运行；它的作业 harness 计划还没有构建。
- **MemoryAgentBench。** Gemma 4：32 个 case 的配置（它的运行用的是 8 个 case、8 个 session）和 stock（Vanilla）对照。没有一次运行打过分（[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 5.1）。

### 功能性 GPU 检查（不是 benchmark 运行）

| 检查 | 作业 | 提交 | 判定 | 审计 | 日期 |
| --- | --- | --- | --- | --- | --- |
| 开空闲释放的 24 个 session（sess24ri r2）：`goal_query`（vLLM Goal 引擎，`--goal-overlap`），24 个 session，`--release-idle-pages`，`--admission-overlap 24`，thinking 开，脚本化的边界情况客户端 | 3616345 | `7a8c36e`（integrate-1116） | `SWE32_FAIL_VERIFIED`（Slurm `FAILED`），原因是作业 harness 的空闲释放检查有缺陷，不是 runtime：agent 池跑完了（24 个任务中 24 个，没有停住；181 次释放、113 次重新载入，与引擎的总数相等），唯一失败的检查按到达顺序而不是准入顺序排列请求，把一个 session 的第二次释放算成了中间没有重建。用修正后的规则离线重查（作业 harness 的 payload `269a00e6`）：服务端的每项检查都通过（53 次第二次释放都发生在重建之后），`JOIN_PASS`，Goal 文本 `VERIFIED`，finding `GOAL_FALLBACK`；判定仍为 FAIL，因为它必须与作业自己的记录一致 | 仅核验器（未审计） | 2026-11-17 |
| 24 个 session（sess24i）：`goal_query`（vLLM Goal 引擎，`--goal-overlap`），24 个 session，不释放空闲 session，thinking 开，脚本化的边界情况客户端 | 3613537 | `7a8c36e`（integrate-1116） | `SWE32_FAIL_VERIFIED`（在 tbench 上核验，Slurm `FAILED`；Goal 文本 `VERIFIED`，join 为 `JOIN_FAIL`，停住的运行本应如此）：格式错误的回滚之后 `CAPACITY_STALL`（[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 2.6） | 仅核验器（未审计） | 2026-11-17 |
| 带 Goal 的脚本化边界情况（keepedgegoalpi）：`goal_query`（vLLM Goal 引擎），`--goal-overlap` 关和开，16 个 session，thinking 开 | 3613408 | `7a8c36e`（integrate-1116） | `SWE32_PASS_VERIFIED`，两个服务上的 Goal 文本各在 192 次调用中 192 次 `VERIFIED`，`JOIN_PASS`（关 / 开这一对不是逐 token 相同，192 个请求中 60 个相同：按作业 harness 的分析，最早的差异出现在相同 prompt 的生成中） | 仅核验器（未审计） | 2026-11-17 |
| 脚本化边界情况（keepedgei r2）：`full`，16 个 session，thinking 开，格式错误和回滚 | 3613280 | `7a8c36e`（integrate-1116） | `SWE32_PASS_VERIFIED`，release gate 1（keep 渲染）在 192 次调用中 192 次通过；join 修复之后 `JOIN_PASS`（第一次运行为 `JOIN_FAIL`，是 join 的缺陷，不是 runtime 的） | 仅核验器（未审计） | 2026-11-17 |
| GG1：Gemma 4 开启 thinking，与 stock 的 token 对齐（`--thinking-history strip`、`--enable-thinking`） | 3583846 | `4ebff97` | `GG1_PASS_VERIFIED`（读数 A1、A2 未达到） | 仅核验器（未审计） | 2026-11-14 |
| ASG1：Gemma 4 `--async-scheduling` | 3560784 | `4ed1013` | `ASG1_PARTIAL_PASS_VERIFIED` | 仅核验器（未审计） | 2026-11-12 |
| rw2：在 Qwen 上重放 vLLM Goal 引擎 | 3488202 | `cd42995` | `RWR2_PASS_VERIFIED`（每次 lm-format-enforcer 调用都失败，没有产生 token） | 仅核验器（未审计） | 2026-11-01 |
| rw3：lm-format-enforcer 修复之后在 Qwen 上重放 vLLM Goal 引擎 | 3499550 | `c07d42a` | `RWR3_FAIL_VERIFIED`（单独的 lm-format-enforcer 在 160 次调用中有 99 次与 Hugging Face rewriter 相同，低于 0.7；4 次调用超时） | 仅核验器（未审计） | 2026-11-01 |
| THG1（G1）：Qwen 在 `chat-text` 上用 `--thinking-history strip` | 3493649 | `8ca8511` | `THG1_PASS_VERIFIED` | 仅核验器（未审计） | 2026-11-01 |
| THG1B（G1b）：Qwen 在 `chat-text` 上用 `--thinking-history strip` | 3500646 | `8ca8511` | `THG1B_PASS_VERIFIED`（与 stock 的 token 差异是 CUDA graph 的近似并列） | 仅核验器（未审计） | 2026-11-02 |
| CMG1：Qwen 开启 thinking 用 `--goal-overlap`（一对 `--thinking-history strip`、一对 keep 模式） | 3500604 | `c07d42a` | `CMG1_PASS_VERIFIED`（两对都 EXACT） | 仅核验器（未审计） | 2026-11-02 |
| CMG1：Gemma 4 `--goal-overlap`，其中一个服务用 `--goal-rewriter-engine vllm` | 3508345 | `c07d42a` | `CMG1_FAIL_VERIFIED`（带该开关的两个服务始终没有 READY） | 仅核验器（未审计） | 2026-11-05 |
| CMG1：Gemma 4 `--goal-overlap`，其中一个服务用 `--goal-rewriter-engine vllm` | 3538523 | `c07d42a` | `CMG1_PASS_VERIFIED`（两对都 EXACT） | 仅核验器（未审计） | 2026-11-08 |
| ASG1：Qwen `--async-scheduling` | 3519531 | `4a498dc` | `ASG1_FAIL_VERIFIED`（第二轮开始时出现 KeyError；async-fix2 已修复） | 仅核验器（未审计） | 2026-11-05 |
| ASG1：Qwen `--async-scheduling` | 3532746 | `4ed1013` | `ASG1_PARTIAL_PASS_VERIFIED`（每对 b = 1 都 token 完全一致；部分通过来自比较器的 `state_digest` 检查） | 仅核验器（未审计） | 2026-11-07 |
| ASG1：Gemma 4 `--async-scheduling` | 3519554 | `4a498dc` | `ASG1_FAIL_VERIFIED`（没有运行任何异步服务：权重加载卡住时第一个服务没有 READY） | 仅核验器（未审计） | 2026-11-06 |
| ASG1：Gemma 4 `--async-scheduling` | 3549350 | `4ed1013` | `ASG1_FAIL_VERIFIED`（b = 1 时 token 完全一致；每个服务 97 个请求中有 12 个在 harness 的客户端中超时） | 仅核验器（未审计） | 2026-11-09 |
| `--phase1-graph-tokens` | 3569412 | `b6f6b9f` | `SPD3_PARTIAL_PASS_VERIFIED` | 仅核验器（未审计） | 2026-11-14 |
| gbench：聚集探针和 `tools/token_gather_bench.py` | 3611486 | `bdc1ca8`（integrate-1114） | `SWE32_PASS_VERIFIED`、`GATHER_PROBE_PASS`、`TOKEN_GATHER_BENCH_PASS`（114 行中 114 行；决定 `DEVICE_DIRECT_ADVISED`） | 仅核验器（未审计） | 2026-11-16 |
| token 聚集，一个 SWE-bench 任务：`observation_query` 带和不带 `--token-gather markers`，thinking 开 | 3603240 | `2b2a752`（integrate-1114） | `SWE32_FAIL_VERIFIED`（harness 的一项检查在 session 关闭之前没有计入它持有的聚集块，问题不在 runtime；聚集探针为 `GATHER_PROBE_PASS`，决定 `UVA_ADVISED`） | 仅核验器（未审计） | 2026-11-14 |
| token 聚集的第一个 GPU 作业：聚集探针，然后一个聚集服务 | 3602601 | | 没有核验记录：聚集探针通过，第一个聚集服务在启动时因 vLLM 0.24.0 的 FlashAttention 视图布局被拒绝（由 `74c1d32`、`eec08cf` 修复） | 未离线核验 | |
| ASG1（G1 r1）：Qwen `--async-scheduling` | 3499729 | `f162120` | `ASG1_FAIL_VERIFIED`（每个 runtime 异步服务都在构造 Scheduler 时崩溃；由 async-fix 修复） | 仅核验器（未审计） | 2026-11-03 |

### 更早的 GPU 检查、pilot 与测量

docs/ 中各页面为单个功能、一次测量或一次 pilot 提到的作业，直到 runtime `cf2b28d` 和发布仓库的验收，从旧到新。判定来自每个作业的核验记录（作业被再次核验时取最后一次）；“审计”同上。

| 检查 | 作业 | 提交 | 判定 | 审计 | 日期 |
| --- | --- | --- | --- | --- | --- |
| 在作业 3270721 的干净环境上的 4 session 合成运行 | 3300301 | `6e52a94` | `MS2A_MEASURE_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-03 |
| SWE-bench pilot：10 个 Astropy 实例 × `full`、`goal_query`、`observation_query`，每个服务 5 个 session，784 token 的 Page | 3300434 | `b1e6988` | `V12_PILOT_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-04 |
| 解耦布局的 smoke-dp1：三种布局下的 `full`；16 token Page 下的四个策略；24 GiB 下 4 个 session | 3308015、3308414、3308482 | `8f1af6a` | 没有核验记录（没有重新核验） | 未离线核验 | |
| 16 token Page 的 SWE-bench pilot，10 个实例：`full` | 3313802 | `780fe54` | `V14_PILOT_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-05 |
| 同一 pilot：`observation_query` | 3314010 | `780fe54` | `V14_PILOT_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-06 |
| 同一 pilot：`goal_query` | 3314319 | `780fe54` | `V14_PILOT_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-06 |
| 同一 pilot：`random_fill` | 3314623 | `780fe54` | `V14_PILOT_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-06 |
| P = 16 的解耦布局：性能，两种布局下 `full` 的 token 相同 | 3321538 | `b37044f` | `SPF1_PERF_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-06 |
| CUDA graph 的各种模式 | 3338893 | `f19f464` | `SCG2_FAIL_VERIFIED`（计划的步骤不完整，finding `STEPS_INCOMPLETE`） | 仅核验器（未审计） | 2026-10-08 |
| 一个服务上 16 个 session（合成 session） | 3339019 | `afffd2b` | `SMS1_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-08 |
| CUDA graph | 3339448 | `f19f464` | `SCG2_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-08 |
| 开 `--cuda-graphs full` 的 16 个 session；用 `full-decode` 的 session 池（24 个任务，K = 8） | 3343069 | `613a7d5` | `SINT1_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-09 |
| 131,072 token 的上下文、4,096 token 的回复和 arena 上限，16 个 `observation_query` session | 3348705 | `710bfc6` | `SCTX1_PARTIAL_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-10 |
| admission 优化：一个 session 和 16 个 session 的 `observation_query` | 3349078 | `4eae441` | `SADM1_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-10 |
| admission 优化：5 个和 16 个 session 的 `goal_query` | 3349504 | `4eae441` | `SADM1_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-10 |
| Transformers 5.13 与固定的 `tokenizer.json` 的逐请求比较（对这类请求返回 HTTP 500） | 3354504 | | 没有核验记录 | 未离线核验 | |
| Gemma 4 第 0 阶段：stock vLLM，一个 10 轮的 session | 3364469 | — | `GEM0_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-12 |
| Gemma 4 第 0 阶段：stock vLLM，agent 运行 | 3364911 | — | `GEM0_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-12 |
| 解码快速路径（64 token 的 Page 运行过一次，解码测量中的 `full`） | 3366128 | `43fd9c3` | `SPD1_PARTIAL_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-12 |
| 262,144 token 的上下文 | 3360375 | `e3efc22` | `SCTX2_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-13 |
| 解码测量（SPD2） | 3379206 | `227dedb` | `SPD2_PARTIAL_PASS_VERIFIED_WITH_VERIFIER_FIX`（第一次核验为 `SPD2_FAIL_VERIFIED`） | 仅核验器（未审计） | 2026-10-16 |
| Gemma 4：两个池的布局，选择性 runtime 关闭 | 3397695 | `e0df9d5` | `GEM1_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-18 |
| Gemma 4 runtime 服务（g1a）：每个 runtime 服务都在启动时停止 | 3398198 | `0d37346` | `GEMR_PARTIAL_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-19 |
| Gemma 4 runtime 服务（g1b）：每个 runtime 服务都在启动时停止 | 3398614 | `4eb9f9d` | `GEMR_PARTIAL_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-19 |
| 每次完成的主机工作：一次一个请求，2K 到 257K token 的历史 | 3410181 | `f9322a6` | `SHBM_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-19 |
| 装配修复之后的 Gemma 4 runtime 服务（g1a） | 3410750 | `316a1b4` | `GEMR_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-21 |
| 装配修复之后的 Gemma 4 runtime 服务（g1b） | 3410948 | `316a1b4` | `GEMR_PARTIAL_PASS_VERIFIED`（它的 `goal_query` 服务失败） | 仅核验器（未审计） | 2026-10-21 |
| Gemma 4 runtime 服务（g1c） | 3429673 | `9ef0d6d` | `GEMR_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-22 |
| Gemma 4 runtime 服务（g1b），16 个 session | 3429672 | `9ef0d6d` | `GEMR_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-24 |
| Qwen 的解码检查：`--decode-fastpath3` 和 `--deferred-completion-io` 对照运行间的波动 | 3429022 | `120bc3c` | `SPD3_PARTIAL_PASS_VERIFIED`（有些轮次缺少 stock 服务的测量窗口） | 仅核验器（未审计） | 2026-10-24 |
| 重放记录下的 Goal rewriter 调用，逐个和按记录分批 | 3443061 | `53e3b7a` | `RWRP_PARTIAL_PASS_VERIFIED`（计时部分被截断） | 仅核验器（未审计） | 2026-10-24 |
| Gemma 4 runtime 服务（g1d） | 3461663 | `cf2b28d` | `GEMR_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-26 |
| 用 `TRITON_ATTN` 的解码检查 | 3461668 | `cf2b28d` | `SPD3_PARTIAL_PASS_VERIFIED` | 仅核验器（未审计） | 2026-10-28 |
| 发布仓库的 GPU 验收（它的快速测试） | 3504770 | — | `RLAC_PASS_VERIFIED` | 仅核验器（未审计） | 2026-11-02 |
| 解码步测量（8 个请求在 2K token 的上下文上解码） | 3561459 | `51ecc30` | `SPD3_PARTIAL_PASS_VERIFIED` | 仅核验器（未审计） | 2026-11-09 |

## 分支

| 分支 | 提交 | 内容 | 检查 |
| --- | --- | --- | --- |
| `main` | integrate-1117 与 `7892db3` 的合并（2026-11-17） | runtime `2fb7f62`（integrate-1117） | `2fb7f62` 的 release check：2,287 个测试，跳过 141 个；`tools/policy_parity.py` 与 `7a8c36e` 对比：只有 `serve.py --help` 不同；GPU：见上面各表，还没有跑过的部分见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 5.7 |
| `main-integrate-1112`（tag） | `7892db3` | runtime `4ebff97`（integrate-1112），2026-11-15 至 2026-11-17 为 `main` | release check 2,059 个测试；每个模型的 GPU 冒烟（见上面各表） |
| integrate-1114、integrate-1116、integrate-1117 | `a6d3ef7`、`21f69e4`，以及 integrate-1117 在合并前的最后一个提交 | 2026-11-17 合并进 `main`，并已从 GitHub 删除；它们的条目见[历史](history.zh-CN.md#版本记录) | release check：`bdc1ca8` 2,175 个测试，`b88a6a7` 2,283 个，`2fb7f62` 2,287 个 |
| `data-swe32`（在 GitHub 上） | `947e9fb` | 八次 SWE-bench 运行（3583822、3583081、3539242、3539260、3576565、3577833、3461703、3611601）的运行归档和逐次表格，以及两张图；不含代码 | |

计划中的 GPU 工作：按 README 中的[24 个 session 的配置](../README.zh-CN.md#24-个-session-的配置组里的-150-个实例的运行2026-11-17尚未经过-gpu-验证)，在 `main` 的 runtime `2fb7f62` 上跑 150 个实例，2026-11-17 已排队：作业 3616690（`full`），然后 3616705（stock 对照），然后 3616729（`ledger_query`）；之后是并发扫描。
