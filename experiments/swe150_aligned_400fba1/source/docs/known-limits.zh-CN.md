# 已知限制

[English](known-limits.md) | 简体中文

写给运行或测量本 runtime 的人：哪些还不能用，哪些只能小心地用。每项限制写明现象、影响谁、规避办法、出处和何时解除。这里只列在本分支上仍然成立的限制；解除后移到[历史](history.zh-CN.md#版本记录)的一个条目里。作业结果见 [docs/status.zh-CN.md](status.zh-CN.md)；标为“未审计”的数字还没有被独立重算过（[docs/release-gates.zh-CN.md](release-gates.zh-CN.md#3-一个数字什么时候可以用)）。截至 2026-11-17（integrate-1117）。

## 1. 历史、回滚与 reasoning（thinking 开）

### 1.1 格式错误的回滚会重新 prefill 整个 session

- **现象。** Mini-SWE-Agent 拒绝一条回复（格式错误，最常见的是一条在 4,096 token 上限处被截断、仍在 reasoning 的回复），并用上一个请求加一条用户消息重试时，服务渲染出与之前完全相同的历史，之前的每段 reasoning 和每个生成的 token id 都保留（回滚，历史标签 `rollback`，[6.8 节](running.zh-CN.md#68-开启-thinking第一阶段--enable-thinking)）。但在引擎一侧它仍是一次在本地第 0 轮的重建：整个 session 重新 prefill，该 session 的 selector、ledger、标记页和 gather 的状态都从头开始。`goal_query` 和 `goal_query_markers` 的 Goal 延续下去（1.3）。
- **影响。** thinking 开、`--protocol mswea-xml`，所有策略；代价是每次格式错误一次完整的 prefill（经审计的五次运行中，格式错误占调用的 1.70 % 到 2.65 %，见 1.4）。
- **规避。** runtime 里没有；在上限处被截断的回复越少，回滚越少。
- **出处。** keep-rebuild-1116（`eb43675`、`51af079`）；[6.1 节](running.zh-CN.md#61-mini-swe-agent-xml-协议)和 6.8 节；`tests/test_keep_rebuild.py`。
- **解除条件：** 回滚时保留 KV：这需要在每次完成之后保留被拒回复的生成头起点处的 Gated-DeltaNet carry（以及 Gemma 在该点的窗口），是 worker 和引擎的改动，需要它自己的 GPU 测试。

### 1.2 Gemma 4 上，不是回滚的重建会丢掉之前的 reasoning

- **现象。** 1.1 的回滚之外的重建会重新渲染历史：Qwen 用模板的 `preserve_thinking`，所以 reasoning 保留；Gemma 4 按其模板原样渲染，会丢掉之前回复的 reasoning。
- **影响。** Gemma 4 以 keep 模式开启 thinking（`--enable-thinking`），任何不是回滚的重建之后。
- **规避。** 没有；回滚（常见情况）在两个 profile 上都保留完全相同的历史。
- **出处。** keep-rebuild-1116；[6.8 节](running.zh-CN.md#68-开启-thinking第一阶段--enable-thinking)。
- **解除条件：** Gemma 的重建渲染出保留的 reasoning。

### 1.3 回滚时保留 Goal 使 `goal_query` 的结果从 session 第一次重试起发生变化

- **现象。** goal-rollback-1116 之前，每次重建（包括格式错误的重试）都让维护的 Goal 从任务重新开始。从它起，重复上一个请求再加一条用户消息的重试会保留 `goal_query` 和 `goal_query_markers` 的 Goal：重试轮沿用被丢弃回复所在轮的 Goal，记为一个由 runtime 写出的 `KEEP`，既不运行 Goal 门控也不调用 rewriter（trigger `rollback`，`query.goal.carry`），之后的轮次像被丢弃的回复从未发出过那样更新它（[6.1 节](running.zh-CN.md#61-mini-swe-agent-xml-协议)）。在 session 第一次重试之前什么都不变；从那时起 Goal 与之前的 runtime 不同：对作业 3576565 的请求做离线重放（未审计），40 次回滚全部保留了 Goal，被保留的回滚之后的请求在全部 38 个情形中得到不同的 Goal（之前的 runtime 为 `INITIALIZE` 5、`PATCH` 31、`REWRITE` 2；现在为 `KEEP` 13、`PATCH` 23、`REWRITE` 2）。保留 Goal 的轮次不调用 rewriter，也不构建 Goal：它不在 Goal 处停车，它的 native 记录没有 Goal 计时字段。Goal 负责人尚未确认这一改动。未在 GPU 上运行（[docs/status.zh-CN.md](status.zh-CN.md)）。
- **影响。** XML 协议上的 `goal_query` 和 `goal_query_markers`（Qwen 和 Gemma 4，thinking 开或关）；跨这一改动比较 `goal_query` 的结果；决策直方图（保留下来的 `KEEP` 不是门控给出的）；要求停车服务上每个 `goal_query` 准入都在 Goal 处停车的作业 harness（Gemma 见 5.5）。
- **规避。** 只比较同一 runtime 版本的 `goal_query` 结果；把 trigger 为 `rollback`（或 `query.goal.carry.carried`）的行与门控给出的 `KEEP` 分开统计；harness 检查把保留 Goal 的行当作没有构建 Goal 的轮次。
- **出处。** goal-rollback-1116（`f9ada25`、`a39dd08`、`43ff7b0`；合并为 `7a8c36e`）；6.1 节；`tests/test_goal_rollback.py`。
- **解除条件：** Goal 负责人确认这一改动，且在本 runtime 上用接受保留 Goal 行的 harness 跑的一次 `goal_query` GPU 运行通过；跨这一改动的结果仍不可比。

### 1.4 keep-rebuild 修复之前的运行在格式错误之后丢失了之前的 reasoning

- **现象。** keep-rebuild-1116 之前，一条被拒的回复会让 XML 协议从 content 重新编码之前的每条消息：从一个 session 的第一次格式错误起，之前每条回复的 reasoning 在该 session 余下的部分都不在 KV 历史里了。这些运行只在每个 session 第一次格式错误之前是 keep 模式的 baseline。
- **影响。** runtime `60c5d21`、`ba4d1ba` 和 `4ebff97` 以及包含它们的构建上每一次 thinking 开的 SWE-bench 运行。各作业中位于 session 第一次重建处或之后的调用所占比例（已审计，2026-11-16）：

| 作业 | 策略 | runtime | 调用 | 格式错误 | 受影响的 session | 第一次错误之后的调用 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 3583081 | `full` | `ba4d1ba` | 1,262 | 26 | 12 / 32 | 351（27.8 %） |
| 3539242 | `recent_fill` | `60c5d21` | 1,613 | 32 | 12 / 32 | 683（42.3 %） |
| 3539260 | `observation_query` | `60c5d21` | 1,548 | 41 | 16 / 32 | 640（41.3 %） |
| 3576565 | `goal_query` | `ba4d1ba` | 1,843 | 40 | 14 / 32 | 869（47.2 %） |
| 3577833 | `akv_rule` | `ba4d1ba` | 1,411 | 24 | 11 / 32 | 429（30.4 %） |

  `ledger_query` 的运行 3611601（`51b4085`）情况相同（35 次回滚，1,548 次调用中 519 次在第一次错误之后；离线重放，未审计）；异步对照 3611917 / 3612333 在 `4ebff97` 上运行，没有这个修复（未测量）。作业 3583081 中，26 条被拒回复里有 25 条是在 4,096 token 上限处被截断的 reasoning。
- **规避。** 这些运行之间互相比较，不要与保留 reasoning 的运行比较；stock 参照（作业 3583822）按设计丢掉之前所有的 reasoning（`reasoning_dropped`）。
- **出处。** 2026-11-16 的审计（每次运行的数字来自 runtime 行、轨迹以及固定的 tokenizer 和模板）；用 `tools/keep_render_check.py` 的重放。
- **解除：** 对在含 keep-rebuild-1116 的 runtime（integrate-1116 及之后）上的运行已解除。

## 2. 容量、session 与池

<a id="21-超过-16-个-session-和---gdn-inplace-crossing-都只经过-cpu-测试"></a>

### 2.1 超过 16 个 session（XML 和 kv-session）和 `--gdn-inplace-crossing` 都只经过 CPU 测试

- **现象。** 一个服务最多接受 32 个 session（`--max-running-sessions`、`--admission-overlap`；chat-text 池仍为 24），`--gdn-inplace-crossing` 让每个运行中的请求只占一个 Gated-DeltaNet 槽位。XML 和 kv-session 协议上超过 16 个 session（sessions-32-1116）和 `--gdn-inplace-crossing` 都没有在 GPU 上运行过。chat-text 服务的上限此前就是 24，它们在更早的 runtime 上确实在 GPU 上跑过超过 16 个 session：Terminal-Bench 的 dry run 用的是 K = 20（例如作业 3399075、3399437、3461661 和 3560498，见 [docs/status.zh-CN.md](status.zh-CN.md)）。
- **影响。** 超过 16 的并发扫描；在 25 GiB、262,144 token、4,096 token 回复时，不开该开关的 launcher 接受 K ≤ 25（[4.4 节](running.zh-CN.md#44-一个服务上的多个-session)）。作业 harness 在重新固定之前仍然固定为 16（它们的打开上限、任务构建器和捕获尺寸检查）。
- **规避。** 出结果时保持 16 个 session；并发扫描作为单独的计划运行。
- **出处。** sessions-32-1116（`7d62df4`、`67b601e`）、gdn-inplace-1116（`ab5d8be`、`dcd4c45`）；`tests/test_ms32.py`、`tests/test_gdn_inplace.py`、`tests/test_inplace_32.py`；[4.6 节](reference.zh-CN.md#46-解耦-kv-布局与-16-token-的-page)。
- **解除条件：** 每种设置一次 GPU 冒烟通过（发布门槛中需显式开启改动的规则）。

### 2.2 Gated-DeltaNet 池在注意力池之前预留 2S + 1 个槽位

- **现象。** 不开 `--gdn-inplace-crossing` 时，解耦布局为 S 个 session 预留 K = 2S + 1 个 Mamba 槽位（1 + 3K 个 51,314,688 字节的 block），从 `--kv-cache-gib` 中先扣，注意力池拿剩下的：25 GiB、16 token 的 Page 时，16 / 20 / 24 / 32 个 session 分别为 100 / 124 / 148 / 196 个 block（4.78 / 5.93 / 7.07 / 9.37 GiB），注意力 block 剩 20,706 / 19,531 / 18,357 / 16,008 个。stock vLLM 共用一个池。一次对 16 个 session 的六次运行的调查（未审计）在请求结束时最多看到 17 个槽位在用（一次异步运行中出现过一次 18 个）；预留覆盖的是容量门控假定的最坏情况。20 和 24 个 session 时，CPU 替身各出现过一次 Mamba 的 `CAPACITY_WAIT`（保守的等待，没有抢占）。
- **影响。** 解耦布局上每个 Qwen 服务的注意力池，超过 16 个 session 时更明显。
- **规避。** `--gdn-inplace-crossing`（K = S + 1：16 / 24 / 32 个 session 时注意力 block 多 2,349 / 3,523 / 4,698 个），只经过 CPU 测试（2.1）。
- **出处。** [4.6 节](reference.zh-CN.md#46-解耦-kv-布局与-16-token-的-page)和 [4.4 节](running.zh-CN.md#44-一个服务上的多个-session)；[docs/exp-decouple-pages/DESIGN.zh-CN.md](exp-decouple-pages/DESIGN.zh-CN.md) 的 D4。
- **解除条件：** `--gdn-inplace-crossing` 通过 GPU 验证并成为默认。

### 2.3 K = 20 时，超过 16 行的解码步退回 piecewise graph

- **现象。** vLLM 0.24 的分派器只为不超过 `max_num_seqs` 的捕获尺寸捕获 FULL 解码 graph。K = 20 时，17 到 20 个请求的一步被填充到 24，在 `--cuda-graphs full` 下回放 piecewise graph（K = 24 和 32 时 K 本身就是一个尺寸）。
- **影响。** K = 20 的服务（A.3 和 A.14 的 Terminal-Bench 服务），以及其他不是捕获尺寸的 K（例如 12）。
- **规避。** 配置允许时用 K = 16、24 或 32。
- **出处。** [4.8 节](reference.zh-CN.md#48-cuda-graph)；`tests/test_ms32.py`。
- **解除条件：** 捕获尺寸包含 K（改 vLLM 配置，然后在 GPU 上做一次解码检查）。

### 2.4 25 GiB 中 16 个 session 的 `full` 需要 `--release-idle-pages`

- **现象。** 不开它时，作业 3579273 的 `full` 服务在准入检查处卡了 55.6 分钟：没有轮次在运行，空闲 Page 放不下任何等待中的轮次。开了之后 `full` 通过（作业 3583081，两次释放共 1,726 个 Page）。
- **影响。** `full`；`akv_rule`（没有预算）也应该开，尽管作业 3577833 不开也没有卡住；`akv_rule` 加 `--release-idle-pages` 还没有在 GPU 上跑过。
- **规避。** 这些服务用 `--release-idle-pages` 启动。
- **出处。** [4.13 节](reference.zh-CN.md#413-释放空闲-session-的-page需显式开启)；[docs/status.zh-CN.md](status.zh-CN.md)。
- **解除条件：** 不解除：这是容量规则，不是缺陷。

### 2.5 Gemma 4 kv-session：32,768 token 预算下 25 GiB 放不下 16 个 session

- **现象。** 选择性规则需要 5,699 个全局 Page，而可用的只有 4,957 个，所以 Gemma 的 MemoryAgentBench 服务跑 8 个 session。
- **影响。** Gemma 4 按其正式配置的 32,768 token 预算跑 MemoryAgentBench。
- **规避。** 8 个 session，或更小的预算（当前配置的 8,192 token，[docs/status.zh-CN.md](status.zh-CN.md#当前配置)）。
- **出处。** [docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](MEMORYAGENTBENCH_BENCHMARK.zh-CN.md#a15-gemma-4-31b-it) 的 A.15。
- **解除条件：** 不解除：这是容量规则。

### 2.6 超过 16 个 session 而不开 `--release-idle-pages` 时，格式错误的回滚之后可能停住

- **现象。** 格式错误的回滚会在 phase 1 重新 prefill 该 session 的全部历史（1.1）。当许多空闲 session 占着它们的常驻 Page 时，这一轮需要的注意力块可能多于空闲的块，而又没有能释放块的请求在运行：服务永久停住（`CAPACITY_STALL`）。作业 3613537（sess24i，2026-11-16，runtime `7a8c36e`：24 个 session，`goal_query`，不释放空闲 session，脚本客户端）就这样停住：每个 session 都回滚之后，下一轮在 phase 1 需要约 1,075 个注意力块，空闲的只有 211 个，也没有任何请求在运行。
- **影响。** XML 协议、thinking 开（会出现格式错误）、超过 16 个 session 的 session 池，所有策略；25 GiB 中 16 个 session 的 `full` 本来就需要这个选项（2.4）。
- **规避。** 这样的服务用 `--release-idle-pages` 启动（[4.13 节](reference.zh-CN.md#413-释放空闲-session-的-page需显式开启)）：容量闸门正是为这一轮释放空闲 session 的常驻 Page（它们在 DRAM 中的副本保留）。README 的[24 个 session 的配置](../README.zh-CN.md#24-个-session-的配置组里的-150-个实例的运行2026-11-17尚未经过-gpu-验证)在每个服务上都用它。
- **出处。** 作业 3613537（`SWE32_FAIL_VERIFIED`，Slurm `FAILED`，[docs/status.zh-CN.md](status.zh-CN.md)）；[4.4 节](running.zh-CN.md#44-一个服务上的多个-session)。
- **解除条件：** 不解除：这是容量规则，不是缺陷（`--release-idle-pages` 计划改为 runtime 的默认值）。

## 3. Goal

### 3.1 vLLM Goal 引擎子进程没有预留的 GPU 显存

- **现象。** 用 `--goal-rewriter-engine vllm` 时，Qwen3-0.6B 子进程使用 27B 服务旁边剩下的 GPU 显存（权重、3 GiB 的 KV、激活、它的 graph 池和一个 CUDA 上下文：约 5–6 GiB；重放作业 3499550 中，四个引擎分组的峰值为 5,550 到 5,786 MiB，两个显存适配分组 mem25 和 mem32 为 5,786 和 5,718 MiB；仅核验器，未审计）；vLLM 的启动检查要求 GPU 有 5 % 空闲。launcher 不为它计入预算。
- **影响。** 用 vLLM 引擎的 `goal_query`；Gemma 4 上留下的显存更少：作业 3538523 中，用 vLLM 引擎的 Gemma 服务（gvll）峰值为 GPU 的 97,871 MiB 中的 92,500 MiB，用 Hugging Face 子进程的服务（govl）为 89,874 MiB（仅核验器，未审计）。
- **规避。** 保持稳定配置命令中的大小（KV 25 GiB，`--goal-rewriter-vllm-kv-gib 3`）。
- **出处。** [4.10 节](reference.zh-CN.md#410-admission-优化需显式开启)；[docs/rewriter-vllm/DESIGN.zh-CN.md](rewriter-vllm/DESIGN.zh-CN.md)。
- **解除条件：** launcher 为子进程预留显存。

### 3.2 vLLM Goal rewriter 只在稳定配置的命令中验证过

- **现象。** `--goal-rewriter-engine vllm` 在稳定配置中通过（作业 3576565）；单独使用时，一次重放中它的 Goal 在 160 次调用中只有 99 次与 Hugging Face 子进程相同（作业 3499550，低于 0.7 的门槛）。
- **影响。** 其他所有组合（其他 benchmark、Gemma 4、不开 `--goal-overlap`）。
- **规避。** 用默认的 Hugging Face rewriter，或原样使用稳定配置的命令。
- **出处。** 4.10 节。
- **解除条件：** 该组合的重放或运行通过。

### 3.3 Goal 特征的正则表达式经由 runtime 的替换运行

- **现象。** 上游的引号字面量正则在未闭合的引号上耗时呈指数增长（作业 3429670 就卡在这里）；runtime 在启动时用一个线性时间、结果相同的对象替换它，遇到别的正则则拒绝。同一文件中的三个 observation 正则在空白多的文本上是平方复杂度（8 KB 时各约 1 s）。
- **影响。** 所有协议上的 `goal_query`。
- **规避。** 这个替换（自 `53e3b7a` 起默认开启）。
- **出处。** [验证记录](history/validation-cf2b28d.zh-CN.md)中请 Goal 负责人审阅的说明；历史中的 fix-engine-stall-1022。
- **解除条件：** Goal 负责人在上游修复这些正则。

### 3.4 Terminal-Bench 的 Goal 输入是暂定的

- **现象。** chat-text 协议上的 `goal_query` 使用团队在 2026-10-18 为 Terminus 2 选定的输入，Goal 负责人尚未确认。
- **影响。** Terminal-Bench 上的 `goal_query`，两个 profile。
- **出处。** [6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)；[验证记录](history/validation-cf2b28d.zh-CN.md)。
- **解除条件：** Goal 负责人确认或修改它们。

## 4. 策略与协议

### 4.1 `ledger_query` 的命令解析只覆盖常见的 shell 形式

- **现象。** ledger 把回复中唯一的 `<mswea_bash_command>` 块解析为读取、编辑和 diff；其他命令两者都不是（不形成 ledger 项，也不覆盖读取）。`ledger_query` v2（ledger-fixes-1116，order `ledger_dedupe_reasoning_light_v2`）修复了对该策略评审中的发现：编辑前后的重复命令、没有以写模式打开的 `.write()`、被截断的读取、按完整路径匹配文件、注释、子 shell 与 `pushd` / `popd`、`git -C` / `rm` / `mv` 以及更多选项形式。解析器仍然看不到的：
  - 读取是 cat、nl、less、more、bat、head、tail、不带 `-i` 的 sed、awk 行范围以及带文件操作数的 grep / rg（下游只能用 head、tail、sed、grep、awk、nl 或 cat 过滤），每个操作数都必须是带文件后缀、不含通配符或变量的路径：`cat Makefile`、`cat $f`、目录搜索（`grep -rn foo src/`）、find、ls 和 `cat a.py | wc -l` 都不算读取。
  - 不知道写了哪些文件的编辑会清空所有文件的读取历史：编辑器、patch、没有文件操作数的 `git apply` / `am` / `stash` / `reset --hard` 等、循环中或经由 xargs 的 `sed -i`、循环中的 `perl -i`、重定向到含变量的路径、没有带引号文件路径的写文件 Python 代码（`Path(p).write_text(...)`）。
  - Python 代码（`-c` 或 heredoc）按其文本判断：写入是带写模式的 `open()` / `Path.open()`、`write_text` / `write_bytes`、shutil 的复制或移动、os 的重命名或删除（其他写法看不到），写入的文件是其中带引号的路径（/tmp、/dev 和输出文件除外）。只有本 session 之前用这样的代码写出的脚本，运行它才算编辑；仓库自带的脚本或用其他方式写出的脚本，对 ledger 来说不编辑任何文件。
  - `cd $VAR`（目录未知）之后，相对路径与所有以它结尾的文件路径匹配。
  - 无法切分成词的命令（引号不配对）既不是读取也不是编辑。
- **影响。** `ledger_query`（Qwen，只有 XML 协议；kv-session、chat-text 和 Gemma 4 被拒绝）。
- **出处。** [docs/policy-registry.zh-CN.md](policy-registry.zh-CN.md#8-query-钩子与-ledger_queryledger-policy-1112) 第 8 节（ledger-fixes-1116 `dc49e7a`、`8b81f35`；合并为 `a2f9e36`）；`source-overlay/page_policies/ledger_query.py`；`tests/test_ledger_query.py`。
- **解除条件：** 不会整体解除：这是解析器的范围。

### 4.2 标记页和 token 聚集只服务 Qwen 的 XML 路径，且不能组合

- **现象。** `goal_query_markers`、`observation_query_markers` 和 `--token-gather` 服务 `--protocol mswea-xml`、16 token Page 上的 Qwen3.6-27B；Gemma 4 和其他协议被拒绝。`--token-gather` 只服务内置的 `goal_query` 和 `observation_query`，所以与标记策略和 `ledger_query` 一起给出时 launcher 拒绝它。runtime 不记录聚集的耗时。
- **出处。** [docs/policy-registry.zh-CN.md](policy-registry.zh-CN.md#9-标记页goal_query_markers-与-observation_query_markersmarker-pages-1114) 第 6 节和第 9 节；[docs/token-gather/DESIGN.zh-CN.md](token-gather/DESIGN.zh-CN.md)。
- **解除条件：** 有进一步的设计覆盖它们。

### 4.3 thinking 开的结果与 thinking 关的 baseline 不可比

- **现象。** 每个 benchmark 的正式配置（A 部分的页面）都是 thinking 关；当前的运行是 thinking 开。thinking 开的结果只与 thinking 开的运行和 thinking 开的 stock 参照比较。
- **出处。** [docs/status.zh-CN.md](status.zh-CN.md#当前配置)。

## 5. Benchmark 与测量

### 5.1 MemoryAgentBench 没有打分

- **现象。** MemoryAgentBench 的运行只报告判定，不报告分数。长程理解（InfBench 摘要）的 case 需要一个没有运行的 LLM 评判，所以保持 `judge_pending`。打分时，综合分数不含它们，取其余三项能力的平均（`composite_score_without_judge`）；有评判 case 待定时，四项能力的 `composite_score` 保持为 null（[docs/benchmarks/memoryagentbench.zh-CN.md](benchmarks/memoryagentbench.zh-CN.md#步骤) 第 5 步）。打分推迟。
- **出处。** [docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](MEMORYAGENTBENCH_BENCHMARK.zh-CN.md#a10-评测) 的 A.10 和 B 部分。
- **解除条件：** 安排带评判的打分。

### 5.2 Gemma 4 在 262,144 token 下跑 Terminal-Bench，`goal_query` 失败

- **现象。** 作业 3560498（`TB16_FAIL_VERIFIED`）中，`goal_query` 的一个轮次需要 1,581 个全局 Page，而其他 session 钉住了池中 3,637 个 Page 里的 2,175 个，任何释放都腾不出被钉住的 Page（`CAPACITY_EXCEEDED`）。
- **影响。** Gemma 4 的 `goal_query` 在客户端默认上下文下跑 Terminal-Bench。
- **规避。** 已通过的 65,536 token 运行（`make_job.py --context 65536`）。
- **出处。** [docs/status.zh-CN.md](status.zh-CN.md)；[docs/TERMINALBENCH_BENCHMARK.zh-CN.md](TERMINALBENCH_BENCHMARK.zh-CN.md#a14-gemma-4-31b-it) 的 A.14。

### 5.3 Gemma 4 开 `--async-scheduling` 是部分通过

- **现象。** 作业 3549350 为 `ASG1_FAIL_VERIFIED`（b = 1 时 token 完全一致，但每个服务 97 个请求中有 12 个在 harness 的客户端超时）；作业 3560784 为 `ASG1_PARTIAL_PASS_VERIFIED`；开它的 Gemma MemoryAgentBench 冒烟通过（作业 3601174）。
- **出处。** [docs/async-sched/GEMMA.zh-CN.md](async-sched/GEMMA.zh-CN.md)。

### 5.4 各 benchmark 页面 B 部分的作业脚本未经测试

- **现象。** benchmark 页面中的示例作业脚本（B 部分：Slurm 脚本，以及 Terminal-Bench 的 `render_job.py` 和 `tb_agents.sh`）没有按原样运行过；GPU 运行用的是作业 harness。
- **出处。** [docs/TERMINALBENCH_BENCHMARK.zh-CN.md](TERMINALBENCH_BENCHMARK.zh-CN.md#part-b-流程)、[docs/SWEBENCH_BENCHMARK.zh-CN.md](SWEBENCH_BENCHMARK.zh-CN.md#part-b-流程)、[docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](MEMORYAGENTBENCH_BENCHMARK.zh-CN.md#part-b-流程)。

### 5.5 Gemma 的作业 harness 还没有针对 integrate-1116 的 runtime 更新

- **现象。** Gemma 的 SWE-bench 作业 harness（perf-gemma-swe32）不认识 integrate-1116 的 runtime 写出的证据（`rollback` 历史行、回滚时保留的 Goal 的行、32 个 session 的上限），所以在 integrate-1116 上跑的 Gemma 作业会在核验时失败。
- **影响。** integrate-1116 及之后的 Gemma 4 GPU 作业；当前的实验用 Qwen（[docs/status.zh-CN.md](status.zh-CN.md#当前配置)）。
- **规避。** 在 integrate-1116 上跑 Gemma 作业之前先更新该 harness。
- **出处。** [harnesses/README.zh-CN.md](../harnesses/README.zh-CN.md)；2026-11-16 的决定。
- **解除条件：** Gemma harness 更新并重新固定到 integrate-1116 的 runtime。

### 5.6 SWE-bench：agent 的 prompt 是为 Astropy 写的，agent 的命令在仓库的环境之外运行

- **现象。** 每次不带 `REPO_ENV=testbed` 的 SWE-bench 运行（2026-11-17 的 150 个 instance 的运行之前的每次运行）都原样使用 PoC 的 agent prompt（`examples/swebench/upstream/swebench_prompt_override.yaml`）。它的测试一节是为 Astropy 写的（"The testbed is an Astropy repository"，`python -m pytest`），Django 的 instance 也用它（前 32 个中有 10 个，前 150 个中有 128 个）。agent 的命令在 `bash -c` 下运行（Mini-SWE-Agent 2.4.2 的 SWE-bench 配置），不会读取镜像的 `~/.bashrc`，所以 `python` 是 base conda 的解释器，在 agent 自己激活 `testbed` 环境之前，仓库无法导入。2026-11-17 在 agent 主机上检查：django-11099 中 `./tests/runtests.py` 报 "No module named 'django'"，astropy-12907 中 `import astropy` 报 "No module named 'erfa'"；`source /opt/miniconda3/bin/activate testbed` 之后两者都能运行。prompt 中针对无法导入的仓库的规则（写一个独立的复现，而不是修复环境）就是 PoC 绕开这一点的办法。运行中遇到它的频率没有统计。
- **影响。** 每次不带 `REPO_ENV=testbed` 的 SWE-bench 运行的绝对解决率，以及每个 instance 的步数和 token：它们不能与在仓库环境激活时得到的 SWE-bench 结果比较。配置之间的比较（stock 对照、`full`、选择性策略）不受影响：它们使用同一个 prompt 和环境。
- **规避。** 在 `examples/swebench/run_agents.sh` 的环境中设 `REPO_ENV=testbed`（自 2026-11-17 起；组里的 150 个 instance 的运行使用它），与 `MAX_RUNNING`、`MAX_TOKENS`、`STEP_LIMIT` 和 `STOCK` 并列（[benchmarks/swe-bench.zh-CN.md](benchmarks/swe-bench.zh-CN.md#agent)）；`examples/slurm/serve_gh200.sh` 会把它传下去。不设时一切照旧。它做两件事：
  1. agent 的命令在仓库的环境激活后运行：model override 的 interpreter 变为 `["bash", "-lc"]`（Mini-SWE-Agent 自己的默认值；它的 SWE-bench 配置设为 `["bash", "-c"]`），镜像的 login shell 会激活 `testbed`。2026-11-17 在 agent 主机上、在按 Mini-SWE-Agent 的方式启动的容器中检查过（django-10097、django-11099、django-14315、astropy-7166、astropy-12907 和 astropy-14995）：`python` 是 `/opt/miniconda3/envs/testbed/bin/python`，工作目录仍是 `/testbed`，命令的输出与 `bash -c` 下相同（没有 banner；只有 `CONDA_*` 变量和 `PATH` 不同），每条命令多用约 0.09 s，`./tests/runtests.py` 和 `python -m pytest` 都能运行。
  2. prompt 是 `examples/swebench/swebench_prompt_testbed.yaml`：PoC 的 prompt，其测试一节按镜像名选择（对 `image` 的 Jinja 条件判断，Mini-SWE-Agent 2.4.2 会把它传给 prompt 模板）。Django 镜像得到 `cd /testbed && ./tests/runtests.py --settings=test_sqlite --parallel 1 <test label>`（标签是 `tests/` 下的模块路径，例如 `utils_tests.test_text`），并以 "Django" 代替 Astropy 的依赖列表；其他每个镜像得到逐字节不变的 PoC 的 Astropy 文字（2026-11-17 用 Mini-SWE-Agent 2.4.2 自己的代码为前 150 个 instance 渲染：22 个 Astropy 的任务消息与 PoC prompt 的相同，128 个 Django 的只有测试一节不同）。

  对一次比较中的每个配置（stock 对照、每个策略、baseline）都使用它：带它的结果不能与不带它的运行比较。
- **出处。** [docs/SWEBENCH_BENCHMARK.zh-CN.md](SWEBENCH_BENCHMARK.zh-CN.md)（PoC 文件的性质）；上面 2026-11-17 的检查。
- **解除条件：** 运行设置 `REPO_ENV=testbed`（一个新配置，不能与不带它的运行比较）；不带它的运行仍有这一限制。

### 5.7 `main` 中还没有 GPU 作业跑过的部分（自 2026-11-17 起）

- **现象。** `main` 在 2026-11-17 合并了 integrate-1114、integrate-1116 和 integrate-1117，当时并非每个部分都已在 GPU 上跑过。还没有在 GPU 上跑过的：session 池的 256 个任务上限和 150 个任务的池；`ledger_query` v2；作为通过核验的运行的 16 个以上的 session（24 个 session 的 smoke，作业 3616345，跑完了全部 24 个任务，没有停住；它的 FAIL 判定来自作业 harness 的检查缺陷，[docs/status.zh-CN.md](status.zh-CN.md)）；`--gdn-inplace-crossing`（默认关闭，仅 CPU 测试，2.1）；integrate-1116 及之后的 Gemma 4 路径（5.5）。
- **影响。** 从 `main` 运行这些部分的人。
- **规避。** tag `main-integrate-1112`（`7892db3`，runtime `4ebff97`）是之前的 `main`，它的 runtime 每个模型都跑过一次 GPU 冒烟。各部分有哪些作业见 [docs/status.zh-CN.md](status.zh-CN.md)。
- **出处。** 2026-11-17 在这些运行之前合并的决定（目前只用 Qwen；`--gdn-inplace-crossing` 保持需显式开启、仅 CPU 测试）。
- **解除条件：** `full` 和 `ledger_query` 的 150 个实例的运行（作业 3616690、3616729）通过，覆盖任务上限、`ledger_query` v2 和 24 个 session；`--gdn-inplace-crossing` 的 GPU 成对测试；更新后的 harness 上的一次 Gemma 冒烟。

## 6. 记录

### 6.1 准入期间被取消的请求的 failure 行没有 request id，退避期间被取消的请求少计一次拒绝

- **现象。** 在准入期间被取消的请求，其 failure 行没有 `request_id`；在容量退避期间被取消的请求少计最后一次拒绝（各路由只在退避返回之后才计入一次拒绝）：作业 3429830 的日志中有 12,408 次拒绝，计数器为 12,406（来自 harness 快照，未审计）。只涉及记账；请求本身都被正确地服务或取消。
- **出处。** 历史中的 runtime `cf2b28d`（“已知问题”）；[harnesses/README.zh-CN.md](../harnesses/README.zh-CN.md)（Terminal-Bench harness 的状态）。
- **解除条件：** 各路由在退避之前计入拒绝，failure 行带上 id。

### 6.2 runtime 不计数的事件

- **现象。** runtime 没有计数器记录 `--async-scheduling` 下一个回复在另一个请求解码时被 `max_tokens` 截断的情况（作业 3604146 的 async-cut 情况是从请求记录推断的），也不记录 token 聚集的耗时。
- **解除条件：** 证据中加入计数器（[docs/design.zh-CN.md](design.zh-CN.md#8-每个服务的证据) 第 8 节）。
