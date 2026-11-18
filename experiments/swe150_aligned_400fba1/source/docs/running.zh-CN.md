# 启动策略与 harness 接口

[English](running.md) | 简体中文

第 4.1 到 4.5 节和第 6 节，integrate-1107 之前在 README 中，编号不变（第 4 节中的服务选项，即 4.6 到 4.15，在 [reference.zh-CN.md](reference.zh-CN.md)）。各文档页面的节号互不重复；[文档地图](../README.zh-CN.md#文档地图)列出了每一节所在的页面。本页说明如何启动服务和各 harness 协议；哪些运行在 GPU 上通过见 [docs/status.zh-CN.md](status.zh-CN.md)，限制见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md)。

下面的命令沿用 README 快速上手里的占位变量（[从零到一个冒烟请求](../README.zh-CN.md#从零到一个冒烟请求)），在仓库目录下、按 [install.zh-CN.md 第 3 节](install.zh-CN.md#3-运行环境)的运行环境执行。`RELEASE` 是 `install.py --apply` 创建的 release（[install.zh-CN.md 2.4 节](install.zh-CN.md#24-安装-release)）；`RUNS` 必须已存在，因为 `serve.py` 要求 `--output` 的父目录存在：

```sh
REPO=/abs/path/to/vllm-0.24.0-selective-prefix-policies
VENV=/abs/path/to/vllm-env
MODELS=/abs/path/to/models
RELEASE=/abs/path/to/releases/selective-prefix
RUNS=/abs/path/to/runs
cd "$REPO"
mkdir -p "$RUNS"
```

## 4. 启动各个策略

### 4.1 单个服务

每个服务固定**一个策略及其冻结任务**。任务文件只包含 `instance_id` 和 `problem_statement`；每个请求的任务消息（第一条 user 消息）必须原样包含这段 problem statement。输出目录必须尚不存在（其父目录要存在）。先看 dry run；只有 `--start` 才会加载模型。

```sh
python -B serve.py --release-dir "$RELEASE" \
  --task examples/task.json --policy full --output "$RUNS/full" --port 8000
python -B serve.py --release-dir "$RELEASE" \
  --task examples/task.json --policy full --output "$RUNS/full" --port 8000 --start
```

`goal_query` 必须明确选择 Goal rewriter：要么给一个本地 Qwen3-0.6B checkpoint 目录（PoC 的设置），要么用 `--no-goal-rewriter` 只用确定性的 Goal 字段。`observation_query` 和 `random_fill` 不需要选择；它们接受 `--goal-rewriter-model`，但从不使用（见第 5 节）。其余 selector 选项可选，见第 5 节。`serve.py --help` 列出所有参数。对 query 策略和 `random_fill`，我们推荐 16 token 的 Page（4.6 节）：

```sh
python -B serve.py --release-dir "$RELEASE" \
  --task examples/task.json --policy goal_query --output "$RUNS/goal" --port 8000 \
  --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --page-tokens 16 --kv-layout decoupled --start
```

`--start` 会把启动器进程替换成 APIServer，所以启动器的 PID 就是服务 PID。服务只监听 `127.0.0.1:<port>`（从其他主机访问见 6.4 节）。启动需要几分钟：加载模型、分配 KV，以及首次 Triton 编译。已验证的作业里，每个作业的第一个服务在启动后 198–231 秒 READY，之后的服务为 103–146 秒；这些服务都以 eager 方式运行（现在对应 `--no-speed-defaults`，4.15 节），使用默认速度选项的 CUDA graph 时，启动时还要捕获 graph（4.8 节）。

**READY 的判定：** `<output>/ready.json` 存在且能完整解析，并且满足：
- `native_http_ready == true`；
- `evidence_kind == "NATIVE_HTTP_RUNTIME"`；
- `policy` 与请求一致；
- `runtime_manifest == SHA256(<release>/runtime-manifest.json 原始字节)`；
- task/config 哈希与 `startup.json` 一致。

这个文件是先创建、再写入内容的，解析失败时应重试。`/v1/models` 能访问不等于 READY。`examples/slurm/server_checks.py ready <output> <policy> <release>` 正是检查这些条件（退出码 0 表示 READY，1 表示尚未就绪，2 表示不匹配）。

**停止：** 只对该 PID 发送 `SIGTERM`，不要发给整个进程组。等进程以 0 退出后，再核对清理回执（第 8 节；`examples/slurm/server_checks.py cleanup <output>`）。被强杀或超时的关停不算干净关停。

### 4.2 一张 GPU 上依次跑已验收运行的四个策略

一张 GH200 同一时间只能放一个服务。每个策略（以及每个任务）都要用全新的服务和全新的输出目录。下面沿用已验收运行（作业 3270721）的流程，任务是快速上手的任务。6.2 节的冒烟客户端（发送一个首轮请求）代替 harness：把它那一行 `examples/http_client.py` 换成你的客户端。这些服务使用现在的默认速度选项（CUDA graph `full`、解码快速路径、1 GiB 的 connector 池，以及 4.15 节的其他开关）；作业 3270721 以 eager 方式运行，使用 32 GiB 的 connector 池，runtime 为 `509eeae`。要与它的设置一致，给 `serve.py` 那一行加上 `--no-speed-defaults`。

```sh
TASK="$REPO/examples/task.json"
OUT="$RUNS/example-array-fix"
REWRITER="$MODELS/Qwen3-0.6B-c1899de"
PORT=8000
mkdir -p "$OUT"
for POLICY in full random50 goal_query observation_query; do
  SELECTOR=""
  case "$POLICY" in goal_query) SELECTOR="--goal-rewriter-model $REWRITER";; esac
  python -B serve.py --release-dir "$RELEASE" --task "$TASK" --policy "$POLICY" $SELECTOR \
    --output "$OUT/$POLICY" --port "$PORT" --start > "$OUT/$POLICY.server.log" 2>&1 &
  SERVER=$!
  until python -B examples/slurm/server_checks.py ready "$OUT/$POLICY" "$POLICY" "$RELEASE"; do
    [ $? -eq 1 ] || { echo "READY for something else: $POLICY"; kill -TERM "$SERVER"; wait "$SERVER"; exit 1; }
    kill -0 "$SERVER" || { echo "server exited before READY: $POLICY"; exit 1; }
    sleep 5
  done
  python -B examples/http_client.py --base-url "http://127.0.0.1:$PORT/v1" \
    --messages examples/messages.json > "$OUT/$POLICY.reply.json"
  kill -TERM "$SERVER"
  wait "$SERVER" || { echo "server exit not zero: $POLICY"; exit 1; }
  python -B examples/slurm/server_checks.py cleanup "$OUT/$POLICY" || exit 1
done
```

下一个服务必须在 GPU 空闲、没有残留进程时启动。已验收的运行里，依次跑四个服务、每个六轮对话（prompt 最长约 33K token），共用时约 18 分钟。

已验收的运行里每个策略都用 784 token 的 Page。按 4.6 节的推荐，给 `goal_query` 和 `observation_query` 的服务加上 `--page-tokens 16 --kv-layout decoupled`（如果在循环里加入 `random_fill` 对照，也给它加上）。在作业 3321538 中，`full` 在两种布局下产生的 token 完全相同（1 个 session × 10 轮，8 GiB，prompt 最长 60,440 token：每一轮的 prompt token、回复和 completion 数都相同）；不过 16 token Page 的 SWE-bench pilot 仍然给每个服务（包括 `full`）都加了这两个参数。

### 4.3 多张 GPU

每张 GPU 并行跑一个策略（每个服务分别设置 `CUDA_VISIBLE_DEVICES=<n>`、不同的 `--port` 和不同的 `--output`），符合“一卡一服务”的设计，但**不在**已验证的运行范围内。永远不要让两个跑同一个任务的 harness 连同一个服务。

### 4.4 一个服务上的多个 session

```sh
python -B serve.py --release-dir "$RELEASE" \
  --task /abs/path/to/tasks-g0.json --policy observation_query --kv-cache-gib 24 \
  --page-tokens 16 --kv-layout decoupled --output "$RUNS/obs-g0" --port 8000 --start
```

`--task` 接受 1 到 32 个 `{instance_id, problem_statement}` 对象组成的 JSON 列表，每个对象一个 session（`max_num_seqs` = session 数；各上限及其决定的大小见下面的**最多 32 个 session**）。instance id 必须互不相同，任何一个 problem statement 都不能包含另一个。每个 harness 跑其中一个任务，连同一个端点；服务把每个请求路由到唯一一个 problem statement 出现在该请求任务消息（`messages[1]`）中的 session，每个 session 的行为和单任务服务一样（见第 5、6 节）。匹配不到任何任务或匹配到多个任务的请求会被拒绝。admission 按到达顺序一次处理一个请求（使用 `--admission-overlap K` 时除外，4.10 节），之后 decode 在各 session 之间成批进行。各 session 共用一个 GPU KV 池，不会发生抢占：最坏情况峰值放不下的一轮会等待（`CAPACITY_WAIT`），直到正在运行的请求释放 block。

`--kv-cache-gib` 的大小要保证在预期的最长上下文下，所有 session 的常驻 Page 加上一轮仍能放下。经过验证的设置如下（“可用”= 注意力 block 数 − 1）。784 token 的 Page 下，一页 51,380,224 字节，注意力 Page 和 Gated-DeltaNet 状态共用一个池；16 token Page 的解耦布局下，一页 1,048,576 字节，Gated-DeltaNet 状态有自己的池（4.6 节）：

| `--kv-cache-gib` | 每个服务的 session 数 | Page | 运行 | 证据 |
| --- | --- | --- | --- | --- |
| 8 | 1 | 784 | 干净安装验收：4 个策略 × 6 轮，prompt 最长约 33K token | 作业 3270721；vLLM：GPU KV cache size 121,605 token |
| 24 | 4 | 784 | 合成运行：4 个 session × 6 轮、每轮 6,500 token observation，`full`/`observation_query`/`goal_query` | 作业 3299809（以及 3292739、3295203、3300301）；池 501 个 block，可用 500 |
| 24 | 3 | 784 | SWE-bench pilot，3 个 instance × 3 个策略，一组 | 作业 3299922；`kv_usable_blocks` 500，session 上限 5 |
| 24 | 5 | 784 | SWE-bench pilot，10 个 instance × 3 个策略，两组各 5 个 | 作业 3300434；`kv_usable_blocks` 500，session 上限 5；vLLM：GPU KV cache size 364,817 token |
| 8 | 1 | 16，解耦 | 合成运行：`full`/`observation_query`/`goal_query`，10 轮，prompt 最长约 60K token | 作业 3321538（runtime `b37044f`）；7,702 个注意力 block + 10 个 Mamba block |
| 24 | 4 | 16，解耦 | 合成运行：4 个 session × 6 轮、每轮 6,500 token observation，`full`/`observation_query`/`goal_query`/`random_fill` | 作业 3321538；23,205 个注意力 block + 28 个 Mamba block |
| 24 | 5 | 16，解耦 | SWE-bench pilot，10 个 instance × 4 个策略，两组各 5 个 | 作业 3313802、3314010、3314319、3314623；22,912 个注意力 block + 34 个 Mamba block（11 个 Gated-DeltaNet slot），可用 22,911，session 上限 5；vLLM：GPU KV cache size 366,592 token |
| 24 | 16 | 16，解耦 | 合成运行：16 个 session × 6 轮、每轮 2,600 token observation，`full`/`observation_query`/`random_fill`，eager | 作业 3339019；19,682 个注意力 block + 100 个 Mamba block（33 个 Gated-DeltaNet slot），可用 19,681；vLLM：GPU KV cache size 314,912 token |
| 24 | 16 | 16，解耦 | 合成运行：同样的负载，`observation_query`，eager 以及 `--cuda-graphs full` | 作业 3343069；两个池与作业 3339019 相同 |
| 24 | 24 个任务，最多 8 个同时运行 | 16，解耦 | 合成 session 池：24 个任务 × 6 轮、每轮 3,000 token observation，`--max-running-sessions 8`，`observation_query`，`--cuda-graphs full-decode` | 作业 3343069；22,031 个注意力 block + 52 个 Mamba block（17 个 Gated-DeltaNet slot），可用 22,030；vLLM：GPU KV cache size 352,496 token |
| 25（integrate-1026 起 Qwen 的一个选项，组里的运行用它） | 32 个任务，最多 16 个同时运行 | 16，解耦 | SWE-bench Verified，前 32 个 instance × 2 个策略（`goal_query`、`observation_query`），262,144 token（4.9 节），`--cuda-graphs full`（现在是默认速度选项，4.15 节） | 作业 3461703（runtime `cf2b28d`）；20,706 个注意力 block + 100 个 Mamba block，可用 20,705，与 CPU 上 vLLM 规划器的计算值相同（`test_qwen_kv25.py`；24 GiB：19,682） |
| 25 | 最多 20 个打开（K = 20），16 个 trial | 16，解耦 | Terminal-Bench 2.0 dry run，16 个任务 × 2 个策略（`goal_query`、`observation_query`），chat-text 协议，262,144 token（4.9 节） | 作业 3461661（runtime `cf2b28d`）；19,531 个注意力 block + 124 个 Mamba block（41 个 Gated-DeltaNet slot），可用 19,530 |

pilot 对 session 数设了上限，保证即使每个 session 都停在 65,536 token 的上限，池也不会用尽：N − 1 个空闲 session 各占 I = 85 个 block，加上一轮的峰值 P = 91 个 block，再加 3 个 block 余量，必须放进可用 block 数 U，所以 N = 1 + ⌊(U − 91 − 3) / 85⌋（这是我们容量分析中 `full` 在 64K 时的占用；因为 session 重建会重新 prefill 每一页，所以对所有策略都按它算）。由此 8 GiB（U = 166）为 1 个 session，24 GiB（U = 500）为 5 个。同一规则给出 16 GiB 为 3 个、32 GiB 为 7 个；这两种大小**不在已验证的运行范围内**。

16 token Page 的 pilot 对注意力池用了同样的规则：停在上限的空闲 session 占 I = ⌈65,536 / 16⌉ + 1 = 4,097 个 block，一轮的峰值是 I + ⌈16,384 / 16⌉ = 5,121，余量是一次 1,536 token 的回复（97 个 block）加上 vLLM 的 1 % watermark（229）。24 GiB 下五个 session 需要 4 × 4,097 + 5,121 + 326 = 21,835 个 block，可用的有 22,911 个；六个放不下。Mamba 池每个 session 两个 slot，另加一个备用（4.6 节；开 `--gdn-inplace-crossing` 时每个 session 一个，另加一个备用，见下）。这条规则同样给出 8 GiB 为 1 个 session、16 GiB 为 3 个、32 GiB 为 7 个（**不在已验证的运行范围内**）。

**最多 32 个 session。** session 上限以及由此决定的池大小只在本节给出；其他页面都指向这里。`remote_ra_live_v2/sessions.py` 中的 `MAX_SESSIONS`（分支 sessions-32-1116 起为 32；此前是 16，更早是 8）限制固定任务列表的大小和 session 池的 K（见下），`remote_ra_live_v2/admission_flags.py` 中的 `MAX_PERMITS` 把 `--admission-overlap` 同样限制在 32；16 个以内的每次启动与 16 上限时逐字节相同。chat-text 池的 K 改为读取 `MAX_CHAT_TEXT_RUNNING`（24）（6.7 节）。session 池最多注册 `MAX_TASKS`（integrate-1117 起为 256；此前为 64）个任务，chat-text 池最多注册 `MAX_CHAT_TEXT_SLOTS`（1,024）个槽位。当前配置每个服务 16 个 session（[docs/status.zh-CN.md](status.zh-CN.md#当前配置)）；XML 和 kv-session 协议上超过 16 个 session，以及 `--gdn-inplace-crossing`，都只经过 CPU 测试（chat-text 池在 GPU 上跑过 K = 20；[docs/known-limits.zh-CN.md](known-limits.zh-CN.md#21-超过-16-个-session-和---gdn-inplace-crossing-都只经过-cpu-测试) 2.1；CPU 测试 `test_ms32.py`、`test_gdn_inplace.py`、`test_inplace_32.py`）。随 session 数 S 增长的有：

- 解耦布局的 Mamba 池，1 + 3(2S + 1) 个 block：S = 16 时为 100 个 block（4.78 GiB），在注意力池之前预留（stock vLLM 只有一个共用的池：[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 2.2）。开 `--gdn-inplace-crossing` 时（需显式开启；Qwen，解耦布局；[reference.zh-CN.md 4.6 节](reference.zh-CN.md#46-解耦-kv-布局与-16-token-的-page)），跨 Mamba block 的请求保留它唯一的 slot，所以池只有 S + 1 个 slot，即 1 + 3(S + 1) 个 block：S = 16、24、32 时为 52、76、100 个 block，P = 16 时注意力池因此多出 2,349、3,523、4,698 个 block（+2.29、+3.44、+4.59 GiB）。如果注意力池连一个 session 在 65,536 token 上限处的一轮都放不下（⌈65,536 / P⌉ + 1 + ⌈16,384 / P⌉ + ⌈1,536 / P⌉ + 1 个 block，加上 1 % watermark），launcher 会拒绝这个预算。8 GiB 下，16、32、64 token Page 最多允许 9 个 session，解耦的 784 token 模式最多 8 个。1 到 8 个 session 的每种配置都能通过；共享的 784 token 池没有 Mamba 池，不受影响。
- 一个服务的终身请求上限（引擎 lifecycle 条目、请求 tombstone 和 render binding，都不会被回收）：每个 session 256 个，即单个 session 的请求上限，至少 1,024（16 个时为 4,096；4.9 节）。
- 超过 16 个 session 时（P = 16，K = 20、24、32；放不下的配置由 `serve.py` 拒绝）：25 GiB 时 Mamba 池占 124、148、196 个 block（5.93、7.07、9.37 GiB），注意力池剩 19,531、18,357、16,008 个 block（所有 session 合计 312K、294K、256K token，16 个时为 331K）。在 262,144 token、4,096 token 回复下，一个 session 的一轮约需 17,850 个 block，所以 25 GiB 允许 K ≤ 25，24 GiB 允许 K ≤ 20，32 GiB 允许 32 以内的所有 K（开 `--gdn-inplace-crossing` 时 24 和 25 GiB 都允许 32 以内的所有 K）；在 65,536 和 131,072 token 时 25 GiB 允许所有 K。使用 1 GiB connector 池时的锁页主机内存：65,536 token 下 24 个 session 需要 100.7 GiB，32 个需要 arena（最多 122 GiB）；在 131,072 和 262,144 token 下，`--dram-arena-gib` 在 K = 24 时最多 123 GiB、K = 32 时最多 122 GiB，70 GiB 的 arena 在 K = 24 时平均每个 session 约 48K token（16 个时为 72K）。vLLM 的 capture size 是 1、2、4 和 8 的倍数，它只为不超过 `max_num_seqs` 的 size 捕获 FULL decode graph：K = 24 和 32 时，`--cuda-graphs full` 下每个解码步都重放 FULL graph；K = 20 时，17 到 20 个运行中请求的解码步被补齐到 24，重放 piecewise graph（[reference.zh-CN.md 4.8 节](reference.zh-CN.md#48-cuda-graph)；[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 2.3），所以在配置允许时 K 宜取 16、24 或 32 而不是 20。`--admission-overlap K` 最大可到 32，固定的 admission 等待时间为 300 + 120 (K − 1) 秒（24 时 3,060 秒，32 时 4,020 秒）；客户端超时必须比它长。
- 主机内存：固定（pinned）Page arena（4.6 节）在 P = 16、65,536 token 上限时最多 S × 4,097 个 1 MiB 的 Page（S = 16 时 64 GiB；超过 16 个 session 和更长的上下文见上一项），另有 connector 的 CPU 池（默认速度选项下 1 GiB，`--no-speed-defaults` 时 32 GiB；4.15 节），以及每个 session 一份 154 MB 的 Gated-DeltaNet carry（开 `--gdn-device-handoff` 时两份，4.10 节）。

成批 decode 时回复会随 batch 而变：同一个 16 session 负载的两次 eager 运行（作业 3339019 和 3343069，见上表；prompt 最长约 14K token，远低于 65,536 token 的上限，所以没有触及下面说的那些限制）在 96 轮中有 72 轮 token 相同，因为回复可能随 batch 的组成而变，而 batch 的组成取决于时序。

在一块 GH200 上，16 个 `full` session 在 64K 时无法满足防死锁规则：它们需要约 67,350 个 1 MiB 的注意力 block 加上 Mamba 池，约 70.6 GiB KV，而 GPU 启动时只有 94.53 GiB 空闲（作业 3313802 的 vLLM 日志），权重占 51.1 GiB。选择性策略只在两次重建之间需要更少的 block，因为重建会回到全部驻留。16 token Page 下使用固定的 `--active-token-budget 8192` 时，15 个空闲 session 各自保留预算和上一轮的 Page，加上一轮的峰值：每轮约 7,000 token 时放得进 24 GiB，达到 16,384 token 的 observation 上限时需要 34 GiB；同样两种假设下，40 % 预算分别需要 41 和 51 GiB。请据此确定 `--kv-cache-gib` 和负载。只要正在运行的请求还能释放 block，`CAPACITY_WAIT` 就会让一轮等待。没有其他请求运行时，如果空闲 session 钉住的 Page 没给这一轮留下空间，它同样等待，直到有 session 关闭，或在 `--release-idle-pages`（4.13 节）下释放空闲 session 的 Page；所有 session 都在等待时，服务停滞。（在 commit `68d19b4` 之前，门控会放行这样的一轮，它在 GPU 上可能被抢占：见[历史](history.zh-CN.md#历史)。）没有其他 session 钉住 block 时，这一轮放行（如同单任务服务）。`--kv-cache-gib` 的最大值仍为 32：在使用 Goal rewriter 时，估计 GPU 上约有 36 GiB 可留给 KV（未测量）。

**Session 池：注册多个任务，最多 K 个同时运行（`--max-running-sessions K`）。** 使用这个选项时，`--task` 注册 1 到 256 个任务（integrate-1117 之前为 64 个；与任何任务列表一样固定在 `startup.json` 中，另加 `max_running_sessions` = K），任一时刻最多打开 K 个 session（1 到 32，且不超过任务数）。这样一个服务就能例如用 16 个并发 agent 跑 32 个 SWE-bench 任务，agent 结束后补入下一个任务；或者用 24 个跑 SWE-bench Verified 的前 150 个实例（README 中的[24 个 session 的配置](../README.zh-CN.md#24-个-session-的配置组里的-150-个实例的运行2026-11-17尚未经过-gpu-验证)，尚未经过 GPU 验证）。按任务消息路由、session 之间的隔离、轮次调度、容量门控和不抢占都不变。改由 K 决定的有：`max_num_seqs` = K、Mamba 池（2K + 1 个 slot，开 `--gdn-inplace-crossing` 时 K + 1 个）和 launcher 的注意力池检查、上面的防死锁规则（N = K）以及固定的 Page arena。由注册任务数 T 决定的是终身请求上限 max(1,024, 256 × T)（T = 32 时为 8,192，T = 150 时为 38,400，T = 256 时为 65,536；4.9 节），它是对已发出请求数的上限，不是预留。一个注册的任务只保存它的记录（任务本身、session id、一个空的 transcript），所以注册 150 或 256 个任务时启动的 KV 池、arena 和 pinned host 内存与注册 32 个时相同；一个 session 的 KV Page、Mamba slot 和 arena Page 只在它打开期间存在，关闭时丢弃它的请求在 host 上的记录（只留下精简的 tombstone；CPU 测试 `test_ms_pool.py`、`test_tasks256.py`：K = 24、150 个任务的 session 池）。

```sh
python -B serve.py --release-dir "$RELEASE" --task /abs/path/to/tasks-pool.json --policy full \
  --max-running-sessions 16 --kv-cache-gib 24 --page-tokens 16 --kv-layout decoupled \
  --output "$RUNS/full-pool" --port 8000 --start
```

- **打开。** 一个 session 在其任务的第一个请求（turn 0）时打开。已有 K 个 session 打开时，这个请求在 admission 之前按到达顺序等待，直到有 session 关闭；等待时间记为它在 `runtime.jsonl` 中那一行的 `open_wait_s`。等待期间离开的请求不会打开它的任务。
- **关闭。** agent 结束后，harness 关闭其任务的 session：`POST /v1/sessions/close`，请求体 `{"instance_id": "<任务 id>"}`（6.5 节）。引擎释放这个 session 持有的一切：GPU 上的驻留 Page 和 live tail，主机上的 Gated-DeltaNet carry 和固定 arena 中的 Page，它的选页表、Goal 状态和 transcript，以及它保留的逐请求记录。两个池恰好收回该 session 固定的 block，下一个等待的任务随即打开。关闭的任务不会重新打开；之后对它的请求立即失败（`session ... closed by the harness`）。
- **失败。** 失败的请求照旧关闭它的 session，但该 session 会保留它的打开名额和引擎状态，直到 harness 关闭它；因此 harness 要关闭它启动的每个 session，不论成败。
- **关停**时会关闭仍然打开的 session，与任何服务一样。

K 决定 Mamba 池的大小，而不是注册的任务数：24 个任务、K = 8 时它有 17 个 slot（2K + 1；作业 3343069，见上表）。

### 4.5 GPU 作业（Slurm）

`examples/slurm/serve_gh200.sh` 在一个 Slurm 作业里运行一个服务：作业本地的 `TMPDIR` 和缓存、第 3 节的环境、dry run、`serve.py --start`、等待 READY、运行 harness、对服务 PID 发送 `SIGTERM`、检查退出码和清理回执。它参照已验证作业的 `job.sh` 和作业步骤。那些作业在命令行上传给 `sbatch` 的选项如下：

| 选项 | 已验证的值 |
| --- | --- |
| `--partition` | 一个 GH200 分区：上限 2 小时的交互分区 `ghx4-interactive`（作业 3270721、3299809、3300301，以及 16 token Page 的 pilot 3313802、3314010、3314319、3314623），784 token Page 的 pilot（作业 3299922、3300434）和 benchmark 作业（例如作业 3461703）用批处理分区 `ghx4` |
| `--gpus` / `--nodes` / `--ntasks` | `1` / `1` / `1`（其他集群上通常对应 `--gres=gpu:1`；不在已验证的运行范围内） |
| `--cpus-per-task` / `--mem` | 上面这些作业为 `16` / `200g`。自 2026-10-14 起，benchmark 作业和下面的命令申请 `113560M`，即一张 GH200 的内存份额；MemoryAgentBench 的 Qwen 服务申请两个，即 `227120M` |
| `--time` | `00:30:00`（4 个单 session 服务，作业 3270721），`00:45:00`（4 个四 session 服务：先是 2 轮的 `full` 预热，然后是 `full`、`observation_query` 和 `goal_query`；作业 3299809 和 3300301），`02:00:00`（2 个五 session 服务，作业 3313802、3314010、3314319、3314623），`03:45:00`（3 个服务，作业 3299922），`10:45:00`（6 个服务，作业 3300434） |
| `--mail-type` | `BEGIN,END,FAIL`（作业 3270721 用的是 `END,FAIL`；批处理队列可能要等很久） |
| `--export` | `ALL`（脚本随后自己重设 `PATH`、`LD_LIBRARY_PATH`、`HOME`、缓存和 `TMPDIR`） |

脚本从提交作业的 shell 读取设置（脚本开头列出了全部变量）。`ACCOUNT` 是你所在 allocation 的 Slurm 账户，`PARTITION=ghx4` 选择 benchmark 作业用过的 GH200 批处理分区（其他集群上用该集群的 GH200 分区），`MAIL` 是接收作业邮件的地址。`RUN_ROOT` 必须在 `sbatch` 之前就存在：Slurm 把作业的输出文件写在那里，脚本也在其中创建作业目录。用快速上手的任务做一次冒烟运行：

```sh
ACCOUNT=your-account
PARTITION=ghx4
MAIL=you@example.org
export REPO VENV RELEASE
export TASKS="$REPO/examples/task.json" POLICY=full RUN_ROOT=/abs/short/jobs MODE=smoke
mkdir -p "$RUN_ROOT"
sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus=1 \
  --cpus-per-task=16 --mem=113560M --time=00:30:00 --job-name=selective-prefix \
  --output="$RUN_ROOT/slurm-%j.out" --export=ALL --mail-type=BEGIN,END,FAIL --mail-user="$MAIL" \
  "$REPO/examples/slurm/serve_gh200.sh"
```

`$RUN_ROOT/slurm-<作业号>.out` 以 `server exit 0, harness exit 0` 和 `CLEAN_SHUTDOWN` 这两行结束（4.1 节的清理检查），作业才算成功。作业的文件在 `$RUN_ROOT/<作业号>/` 中：dry run 的 `dry-run.json`、服务日志 `server.log`、回复 `smoke.json`，以及服务的输出目录 `full/`。

冒烟测试的服务最多锁页约 5.2 GiB 主机内存（`--no-speed-defaults` 时为 36.2 GiB，即 vLLM 32 GiB 的 connector 池），远在一个份额之内。一个份额能容纳的服务，其锁页最坏情况加上进程约 32 GiB、再加 5 GiB 余量不超过 113,560 MiB，即锁页最坏情况最多约 73.9 GiB；超过时请申请两个份额 `--mem=227120M`（[MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](MEMORYAGENTBENCH_BENCHMARK.zh-CN.md#a4-本-runtime服务参数与-vllm-参数) 的 A.4 主机内存，其中 16 个 session 的 Qwen 服务锁页 127.29 GiB）。Qwen 启动器本身只拒绝锁页最坏情况超过其 128 GiB 预算的配置（4.9 节，按原来 200 GiB 的作业确定）；Gemma 启动器自己检查一个份额，即其锁页内存加 30 GiB（4.14 节）。在 `ghx4` 上，一个内存份额按一张 GPU 计费：申请两个份额的作业按两张 GPU 计费。

`RUN_ROOT` 必须短：作业把 `$RUN_ROOT/<作业号>/tmp` 用作 `TMPDIR`，超出 AF_UNIX 预算的路径会被拒绝。`MODE=swebench` 会打开 6.4 节的反向隧道，并通过 SSH 在 agent 主机上运行 `examples/swebench/run_agents.sh`（7.4 节）。`PAGE_TOKENS=16 KV_LAYOUT=decoupled` 会把 `--page-tokens 16 --kv-layout decoupled` 传给 `serve.py`（4.6 节）；两者都不设时，服务使用默认的 784 token 布局。已验证的作业由我们的私有控制器驱动，还有额外的检查（环境快照、GPU 空闲检查、逐轮证据检查）；这个脚本只保留服务端的步骤，**它本身不在已验证的运行范围内**。

## 6. Harness 接口

启动器的 `--protocol` 接受 `mswea-xml`（默认值）、`kv-session` 和 `chat-text`。`mswea-xml` 逐 token 复现 dram-backed-kv PoC 服务端，是 SWE-bench 实验用的协议（6.1 到 6.5 节）。`kv-session` 是 MemoryAgentBench harness 的 token 级 session 协议（6.6 节）。`chat-text` 是普通的 OpenAI chat 文本，面向 Terminal-Bench 2.0 上 Terminus 2 这样的 agent（6.7 节）。`--enable-thinking` 在 Qwen profile 上为 `mswea-xml` 和 `chat-text` 开启 thinking（需显式开启，6.8 节）。在此基础上，`--thinking-history strip` 为 `chat-text` 提供默认模板丢弃 reasoning 后的历史（需显式开启，6.9 节）。当前配置以 6.8 节的保留模式开启 thinking；哪些运行用了这两个选项见 [docs/status.zh-CN.md](status.zh-CN.md)。

### 6.1 Mini-SWE-Agent XML 协议

这是 PoC 服务端的约定，Mini-SWE-Agent 2.4.2 以单个 `<mswea_bash_command>` 动作使用：

- **请求。** `POST /v1/chat/completions`，只看 `messages`：完整历史，每条是 `{"role", "content"}` 文本。依次是 system 消息、任务消息（必须包含冻结的 issue），然后是 agent 的 assistant 和 user 消息，最后一条必须是 user 消息（带 `<returncode>` 的 observation、格式错误提示等）。带 `tools` 的请求会被拒绝，消息里的其他字段会被忽略。
- **采样。** 服务端和 PoC 一样忽略客户端的采样字段：贪心解码，最多生成服务端回复上限那么多个 token（1536，即 PoC 的值；`--max-tokens`，4.9 节），解码文本里出现 `</mswea_bash_command>` 或遇到 EOS/pad 就停。动作结束标记在引擎内部检测。
- **渲染。**
  - system 和任务消息走固定的 chat template，`enable_thinking=false`。
  - 之后每条消息都是 `<|im_start|>{role}\n{content}<|im_end|>\n`。
  - 生成头是 `<|im_start|>assistant\n<think>\n\n</think>\n\n`。
  - 历史里的回复会换回生成时的原始 token ID，按 `strip()` 之后的文本匹配，所以客户端裁掉首尾空白也没关系。
- **为什么关闭 thinking。** 本协议默认关闭 thinking（开启 thinking 需显式指定，6.8 节；当前配置开启它：[docs/status.zh-CN.md](status.zh-CN.md#当前配置)），与 PoC 对齐：PoC 的服务端（dram-backed-kv 的 `scripts/serve_online_selector_v2.py`）以 `enable_thinking=False` 渲染，因为 LiteLLM / Mini-SWE-Agent 在发送下一个请求之前会从 assistant 内容中删去 thinking 块。开启 thinking 时，下一个请求不再延续服务端持有的 KV 前缀（前缀不稳定，KV 无法复用）；关闭 thinking 时，空的 think 块属于服务端渲染的 prompt（即上面的生成头），历史保持只追加。因此开启 thinking 的 baseline 运行与关闭 thinking 的运行不可比（[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 4.3）。
- **响应。**
  - `choices[0].message.content` 是全部生成 token 解码后的文本，包括结束标记，以及最后一个 token 越过标记的字符。
  - `finish_reason` 恒为 `stop`，没有 tool call。
  - 边界 `<|im_end|>\n`（EOS 之后是 `\n`）进入服务端的历史，但不返回。
- **Goal-query 的输入。** Goal 使用任务消息里的 PR 描述、上一轮回复、最新的 user 消息及其中的 `<returncode>`，与 PoC 相同。
- **会话。** 请求若是上一个请求，加上一轮回复，再加一条新的 user 消息，就延续当前会话（第 5 节）。
  - 其他请求和 PoC 服务端一样开一个冷启动新会话。例如丢掉被拒回复的格式错误重试（PoC cohort 的 5 个实例中有 1 个出现），以及上下文压缩。
  - 新会话从文本重新渲染整个请求，释放旧会话的 Page，并重置 selector 和 Goal 状态。它的 runtime 行记录 `stage_b.rebuild == true`。
  - **丢弃回复之后的 Goal（goal-rollback-1116；查询向量来自 Goal 的策略：`goal_query` 和 marker-pages 插件的 `goal_query_markers`；Qwen 与 Gemma 4，开启或关闭 thinking）。** 若重建的请求是上一个请求加一条 user 消息（客户端丢弃了上一条回复：Mini-SWE-Agent 的格式错误重试；开启 thinking 时就是下面的回滚，关闭时是同样的消息形状，其 prompt 仍从文本重新编码），session 的 Goal 保留：constructor 的状态（阶段跟踪、漂移计数、retrieval Goal）和 Goal 缓存都不变，这一重建轮复用被丢弃回复所在轮次的 Goal，记为不调用 rewriter 的 `KEEP`（`reused_previous`，触发条件 `rollback`，`diagnostics` 为 `{"kept_from_turn": …}`）。这个 `KEEP` 由运行时给出，不是 Goal gate 的决策（该轮 gate 不运行），所以决策直方图应把触发条件为 `rollback`（或 `query.goal.carry.carried` 为真）的行与 gate 的 `KEEP` 分开统计。延续的轮不构造 Goal：不调用 rewriter，不在 Goal 处停车，其 native 文件没有 `goal_async_ms`、`goal_construct_ms` 和 `goal_overlap_ms`。之后的轮次用最新的 observation 和格式错误之后保留下来的回复更新它，就像被丢弃的回复从未发出；它们的 Goal 轮次编号从保留轮次的编号继续。此前每次重建都让 Goal 从任务重新开始（一次 `INITIALIZE`，observation 是格式错误消息，又因是 user 消息而被忽略：作业 3576565 的 40 次重建全是这种重试，其 Goal 中 40 次没有 errors、30 次没有 files）。重建的 runtime 行增加 `query.goal.carry`（`goals.jsonl` 中为 `carry`）：`carried`、`from_turn`（构建所保留 Goal 的请求轮次）、`goal_turn`（它的 Goal 轮次编号）和 `reason`（未能延续的原因：没有更早轮次的 Goal、请求没有重复保留轮次的请求，或 Goal 状态此后已改变；这时 Goal 和以前一样重新开始）。其他重建（重发、改动过的历史、上下文压缩）、其他策略以及其他行都不变。重建轮的选择不变（local turn 0：没有候选 Page）。测试：`test_goal_rollback.py`。
  - 开启 thinking 时（6.8 节；分支 keep-rebuild-1116），格式错误重试改为回滚（rollback）：新会话的 prompt 是旧会话截至被拒回复生成头之前的原始 token（之前所有 reasoning 和生成的 token id 都保留），再接格式错误消息和生成头。它和任何新会话一样重新 prefill（`stage_b.rebuild == true`，`thinking.history == "rollback"`），它的 Goal 继续（见上）。keep-rebuild-1116 之前，它从消息的 content 重新渲染，因此该会话此后再也没有之前的 reasoning（6.8 节）。

### 6.2 最小客户端

`examples/http_client.py` 发送一个首轮请求（`python -B examples/http_client.py --base-url http://127.0.0.1:8000/v1 --messages examples/messages.json`）。它只是冒烟客户端，不是 agent。`examples/messages.json` 与 `examples/task.json` 对应，所以只能对以该任务启动的服务使用。

本版本拒绝 OpenAI tool-call 协议（`tools`、`tool_calls`、`tool` 消息）。上一个验收版本（commit `4d90616`）提供该协议，并在其 README 里有说明。`examples/http_client.py` 仍保留 `--tools` 选项，因为 `package.py` 会打包这个文件，而且有测试加载它；不要对本 runtime 使用它，本 runtime 会拒绝任何带 `tools` 的请求。

### 6.3 错误处理

出现任何非 2xx 状态、超时、断连或服务退出，都视为这个服务的这次运行已经结束：
1. 保存 HTTP 错误 body、`failures.jsonl` 和服务日志。
2. 用 `SIGTERM` 停止服务。
3. 如需重试，启动一个全新的服务。

不要静默重试 admission 状态不明的请求，不要推进轮次，也不要复用这个 session。违反 6.1 约定的请求会被 HTTP 错误拒绝。

引擎拒绝请求时，HTTP 500 的 body 是 vLLM 针对 `EngineGenerateError` 的 JSON 错误响应，其消息为空：它不给出原因。原因记录在两处：
- `failures.jsonl` 里该请求那一行的 `chain` 和 `traceback` 字段；
- 服务日志里的 `REMOTE_RA_REGISTERED_ADD {"phase": "engine_rejected", ...}` 一行。

`CAPACITY_WAIT` 拒绝（4.4 节）也会写出同样的日志行，带有 traceback，但它不是错误：前端会重试这一轮，客户端只会看到延迟。

服务只为自己协议路由（6.1、6.6 或 6.7）的请求生成。其他 vLLM 路由上的生成请求（`POST /v1/completions`、`/v1/responses`、pooling 等）会在 API server 内、到达引擎之前被拒绝，返回 vLLM 针对该路由的错误响应（消息为 "this server generates only for requests of its own protocol route; generate requests on other vLLM routes are refused"）；服务及其 session 继续运行。在 review-fixes-1110 之前，一个这样的请求就会终止引擎和所有 session。不生成的路由（`/health`、`/metrics`、`/v1/models`）照常应答。仅在 CPU 上测试：vLLM 给这个拒绝的 HTTP 状态码未在运行中的服务上核对。

外部 prefix 借用、speculative decoding、async loading、preemption、历史改动和未知 drain 都会直接失败，不会静默回退到 Full。

### 6.4 从其他主机访问服务

服务只监听 `127.0.0.1`，没有任何认证；永远不要把它暴露到 loopback 之外。SWE-bench 需要 Docker，所以 pilot 里 agent 运行在另一台 agent 主机上，通过反向 SSH 隧道访问 GPU 作业（作业 3299922 和 3300434）。每个服务按下面的顺序进行：

1. READY 之后，GPU 作业打开一条反向隧道，远端是 agent 主机上的一个 **Unix socket**：

   ```sh
   ssh -N -R "$AGENT_SOCKET:127.0.0.1:$PORT" -o ExitOnForwardFailure=yes \
     -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
     -i "$TUNNEL_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes \
     -o UserKnownHostsFile="$KNOWN_HOSTS" -o GlobalKnownHostsFile=/dev/null -o ConnectTimeout=20 \
     "$AGENT_HOST"
   ```

   这是 `examples/slurm/serve_gh200.sh` 的写法；pilot 的隧道还传了 `-o CheckHostIP=no -o UpdateHostKeys=no`。`PORT` 是服务的端口（默认 8000），`AGENT_HOST` 为 `user@agent-host`，`TUNNEL_KEY` 是 agent 主机接受的密钥，`KNOWN_HOSTS` 是固定其主机密钥的文件，`AGENT_SOCKET` 是该主机上一个 `0700` 目录里的新路径（每个服务一个，最长 100 字节）。
2. 作业从 agent 主机检查隧道：`curl -sf --unix-socket "$AGENT_SOCKET" http://localhost/v1/models` 必须列出 `selective-prefix`。
3. 在 agent 主机上，`examples/swebench/relay.py "$AGENT_SOCKET" <port>` 只监听 `127.0.0.1:<port>`，并把每个连接转发到该 socket；`run_agents.sh` 会启动它，并用 `ss` 检查该端口上只有 `127.0.0.1` 在监听。agent 使用 `http://127.0.0.1:<port>/v1`。
4. agent 运行结束后，作业先关闭隧道，再用 `SIGTERM` 停止服务。

为什么用 Unix socket：如果 agent 主机的 sshd 设置了 `GatewayPorts yes`，每个 TCP 远程转发都会绑定到所有网卡，不管客户端要求的是哪个地址，这会把没有认证的服务暴露出去。Unix socket 转发没有 TCP 监听，relay 又显式绑定 loopback，所以无论 sshd 怎么设置，这种形式都只停留在 loopback 上。计算节点需要能向外 SSH 到 agent 主机；pilot 在每个作业的第一个服务之前检查过一次。

一种更简单的 TCP 形式，**不是已验证的形式**：`ssh -N -R 127.0.0.1:<p>:127.0.0.1:8000 <agent-host>`。只有当 agent 主机的 sshd 设置为 `GatewayPorts no`（OpenSSH 的默认值）时，它才只停留在 loopback 上。任何 agent 运行之前，先在 agent 主机上用 `ss -ltn` 检查；`run_agents.sh -`（不用 socket）会做这项检查，并拒绝非 loopback 的监听。

### 6.5 关闭 session（session 池）

在用 `--max-running-sessions` 启动的服务上（4.4 节），`POST /v1/sessions/close`，JSON 请求体 `{"instance_id": "<启动任务 id>"}`，会永久关闭该任务的 session。它只接受回环地址的对端（服务本来就只绑定回环地址；经过 6.4 节的隧道时对端同样是回环地址），与 chat 路由一样没有 HTTP 认证；前端把它作为私有通道上一个经过认证的帧转发给引擎。响应是引擎的关闭回执（`"object": "session.close"`、`opened`、`turns`、释放的 Page 和 block、两个池关闭前后的空闲 block、`open_sessions`）；对已关闭的任务再次关闭，会返回同一份回执并带 `"repeated": true`。拒绝的情况：403（不是回环地址）、404（不是启动任务）、409（`SESSION_BUSY`：该 session 有排队或进行中的请求；请在它结束后重试）、400（没有 session 池，或请求体不对）。从未打开过的任务关闭时仍经过引擎，引擎记录一份 `"opened": false` 的回执；不释放任何东西，也不触及 worker 和 KV 池。

### 6.6 MemoryAgentBench kv-session 协议

`--protocol kv-session` 服务 MemoryAgentBench harness 中 KV-session 客户端（方法 `dram_backed_kv`）使用的 token 级 session 协议。客户端自己渲染 chat template、自己解析回复；服务端从不渲染模板。这样的服务需要 session 池（`--max-running-sessions K`，4.4 节），不接受 `--task`：它注册的任务是槽位，每个客户端 session 在第一个请求时绑定到下一个未用的槽位（`--kv-session-slots N`，1 到 64，默认 64，所以一个服务在其生命周期内最多服务 N 个客户端 session）。`--kv-session-thinking on|off`（默认 `on`）固定客户端发送的生成头。自 gemma-thinking 起 Gemma 4 profile 的默认值也是 `on`（此前启动器会拒绝不带 `off` 的 Gemma kv-session 服务），所以关闭 thinking 的客户端（例如 MemoryAgentBench 客户端的 Gemma 尾部）必须传 `--kv-session-thinking off`：否则它的请求不以固定的生成头结尾，每个请求都会被拒绝（`HEADER_MISMATCH`）。这样的服务拒绝 `/v1/chat/completions` 和 `/v1/sessions/close`。

```sh
python -B serve.py --release-dir "$RELEASE" --protocol kv-session --kv-session-thinking off \
  --policy observation_query --page-tokens 16 --kv-layout decoupled --active-token-budget 32768 \
  --max-running-sessions 16 --kv-cache-gib 32 --max-tokens 4096 --max-model-len 262144 \
  --dram-arena-gib 124 --release-idle-pages --output "$RUNS/mab-obs" --port 8000 --start
```

这是 MemoryAgentBench `observation_query` 服务的设置（作业 3405844：关闭 thinking，124 GiB 的 arena）；它带快速路径开关的完整命令见 [MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](MEMORYAGENTBENCH_BENCHMARK.zh-CN.md#a4-本-runtime服务参数与-vllm-参数)（A.4）。它最多锁页 127.29 GiB 主机内存，所以它的 Slurm 作业申请两个内存份额 `--mem=227120M`（4.5 节）。这是配置记录（关闭 thinking，32,768 token 预算，32 GiB KV），现已是历史配置；当前配置见 [docs/status.zh-CN.md](status.zh-CN.md#当前配置)。

- **`GET /v1/kv-session/info`** 返回 `{"method": "dram_backed_kv", "budget": <token 数或 null>, "options": {...}}`：服务的注意力预算（即 `--active-token-budget`，否则为 `null`）和它实际运行的设置（策略、页大小、预算方式和选页合同字段、回复上限、上下文上限、thinking、session 数）。
- **`POST /v1/kv-session/complete`** 只接受以下字段（已缩写）：

```json
{"method": "dram_backed_kv", "session_id": "<32 lowercase hex>", "input_ids": [248046, 198, 248045, "..."],
 "header_tokens": 7, "prefix_tokens": 423, "boundary_tokens": 2, "budget": 32768, "max_tokens": 4096,
 "temperature": 0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "seed": 42, "messages": ["..."]}
```

  - session 的第一个请求发送完整的渲染后 prompt：`boundary_tokens` 为 0，`prefix_tokens` + `header_tokens` 等于 `input_ids` 的个数。
  - 之后每个请求只发送新的尾部：上一条回复的结束分隔符（上一条回复的最后一个 token 不是 EOS 时，把它放在最前面）、新消息和下一个生成头。`input_ids[0]` 必须重复上一条回复的最后一个输出 id，`prefix_tokens` 不能变，尾部必须以固定的生成头结束。服务端的 prompt 是该 session 的 token 流加上 `input_ids[1:]`；阶段 1 从上一条回复的最后一个 token 算到生成头之前（第 5 节）。
  - 采样：只支持贪心（`temperature` 0、`top_p` 1、`top_k` −1 或 0、`repetition_penalty` 1）；`max_tokens` 为 1 到服务的回复上限（`--max-tokens`），并截到剩余上下文以内。`budget` 和 `method` 必须与服务端一致。
  - `messages`（客户端的完整历史）从不生成 token。只使用上一个请求之后新增的消息，供 `goal_query` 的 Goal 使用：其中 tool 和 user 消息的文本作为 observation；user 消息（MemoryAgentBench 随最后一个历史块发送的问题，或格式错误提示）成为 Goal 的显式要求。
- **响应：** `text`（全部生成 token 的解码，含特殊 token）、`output_ids`（含最后一个 token）、`prompt_tokens`（prompt 长度）、`completion_tokens`、`cached_tokens`（`prompt_tokens` 减去 `input_ids` 的个数）、`finish_reason`（`stop` 或 `length`），另有单独命名的 `kv_slot`、`kv_turn`、`effective_max_tokens`、各项等待时间和本轮的页数统计。
- **`POST /v1/kv-session/close`**，请求体 `{"session_id": ...}`，返回 200 `{"closed": true, ...}`；服务从未见过或已关闭的 id 也返回 200（已关闭时返回第一次的结果并带 `"repeated": true`）；该 session 有排队或进行中的请求时返回 409。关闭过的 id 不会再被服务。
- **错误。** complete 请求的任何失败都是 400 `{"error": ..., "type": ...}`。prompt 没有为回复留出空间时，在任何模型计算之前拒绝，`error` 恰好是 `"Logical RoPE context limit reached"`（`type` 为 `CONTEXT_LIMIT`）。接纳之前发现的拒绝（请求体、采样字段、生成头、接续 token、prefix、boundary、上下文）不影响该 session；引擎中的失败会使该 session 失效，客户端随后关闭它。`boundary_tokens` 超出轮次几何允许的范围也属于这类拒绝（`type` 为 `BOUNDARY_RANGE`）：boundary 是上一条回复的结束分隔符（2-3 个 token），新消息最晚必须在该回复最后一个 token 所在 Page 的下一个 Page 内开始。在 review-fixes-1110 之前，这样的请求在引擎中被拒绝，而这个拒绝会在请求的 360 秒租约到期时终止引擎。
- **证据。** 每个 `runtime.jsonl` 行增加 `kv_session`：客户端 session id 和槽位、token 几何（`header_tokens`、`prefix_tokens`、`boundary_tokens`、`input_tokens`）、客户端给的和实际生效的 `max_tokens`、`finish_reason`，以及 Goal 的消息来源和显式要求。
- **Gemma 4（4.14 节）。** 在 Gemma release 上，该协议使用关闭 thinking 的 Gemma 轮次格式：user 轮之后为 7 个 token 的生成头，tool response 之后 `header_tokens` 为 0（tail 以 `<tool_response|>` 结尾，由 phase 2 计算），新消息从 `<|turn>` 或 `<|tool_response>` 之后开始，停止 id 为 0、1、50 和 106，启动记录带 `"grammar": "gemma4"`。
- **测试。** CPU 测试（`test_kv_session_protocol.py`，装了 Transformers 时使用固定的 Qwen3.6 tokenizer：thinking 开启时生成头为 5 个 token，服务端的原始 token 流与模板在 `preserve_thinking` 下的渲染逐 token 相同；`test_kv_session_server.py`；`test_kv_session_e2e.py`，MemAgent session 经打包的路由和引擎假实现运行）。
- **GPU 运行。** 本协议在 Qwen 和 Gemma 4 上的 GPU 运行见 [docs/status.zh-CN.md](status.zh-CN.md)（MemoryAgentBench 不打分：[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 5.1）；本节原来的状态文字冻结在 [docs/history/running-gpu-notes.zh-CN.md](history/running-gpu-notes.zh-CN.md)。

MemoryAgentBench 正式测试的配置记录（32 个 case，每项能力 8 个，最多 16 个同时运行，关闭 thinking，`observation_query` 和 `goal_query` 两个策略，以及一个经由 harness 的 Vanilla transport 访问的 stock vLLM 0.24.0 对照；这是历史配置，当前配置见 [docs/status.zh-CN.md](status.zh-CN.md#当前配置)）、Gemma 4 31B IT 端到端运行的设置和 baseline 与之对齐的检查表见 [docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](MEMORYAGENTBENCH_BENCHMARK.zh-CN.md)。

### 6.7 Chat text 协议（Terminal-Bench 2.0 与 Terminus 2）

`--protocol chat-text` 在 `/v1/chat/completions` 上服务普通的 OpenAI chat 文本，token 由服务端自己的 transcript 给出，面向每轮发送完整文本历史、自己解析回复的 agent。目标是原样的 Harbor 0.21.0 Terminus 2 跑 Terminal-Bench 2.0：JSON 动作，没有 system 消息，thinking 关闭。这样的服务需要 session 池（`--max-running-sessions K`，4.4 节；chat-text 服务为 1 到 24），不接受 `--task`：它注册的任务是槽位，惰性绑定到客户端 session（`--chat-text-slots N`，1 到 1024，默认 512）。它服务 [design.zh-CN.md 第 5 节](design.zh-CN.md#5-query-策略的选页规则)的八个内置策略（三个插件策略 `ledger_query`、`goal_query_markers` 和 `observation_query_markers` 只服务 XML 协议）；`goal_query` 只服务 Harbor 0.21.0 的 Terminus 2，其 Goal 输入是**暂定的**（见下）。这样的服务拒绝 `/v1/sessions/close` 和 kv-session 的路由。Gemma 4 profile 的 release 以 Gemma 的轮次格式服务该协议（4.14 节）：除渲染、停止 id 和容量数字外，下文对两个模型都成立，包括 `goal_query` 的 Goal 输入（同样的暂定规则和记录；gemma-tb-goal）。

```sh
python -B serve.py --release-dir "$RELEASE" --protocol chat-text --policy observation_query \
  --page-tokens 16 --kv-layout decoupled --active-token-budget 8192 \
  --max-running-sessions 20 --kv-cache-gib 25 --max-model-len 262144 --max-tokens 4096 \
  --connector-pool-gib 1 --dram-arena-gib 70 --release-idle-pages --output "$RUNS/tb-obs" --port 8000 --start
```

这是 benchmark 的 Qwen 服务所用的大小（[TERMINALBENCH_BENCHMARK.zh-CN.md](TERMINALBENCH_BENCHMARK.zh-CN.md#a3-本-runtime服务参数与-vllm-参数) A.3），即作业 3461661 的设置：25 GiB KV、262,144 token 的上下文和 70 GiB 的 arena（`--kv-cache-gib 25 --max-model-len 262144 --dram-arena-gib 70`）。最早几次 Qwen dry run（作业 3399075、3399437、3429670 和 3434533）使用 32 GiB KV、默认的 65,536 token 上下文和 64 GiB 的 arena（`--kv-cache-gib 32 --dram-arena-gib 64`）。这些设置是配置记录（关闭 thinking）；当前配置见 [docs/status.zh-CN.md](status.zh-CN.md#当前配置)。

- **请求。** 非流式，一个 choice，`model` 为 `selective-prefix`。`messages`：可选的 system 消息，然后是 user 和 assistant 文本；system 之后的第一条是 user 消息，最后一条也是。content 必须是字符串；`tools`、`tool_calls` 和 `tool` 消息返回 400，消息的其他字段被忽略。
- **采样。** 贪心解码，忽略客户端的采样字段。`max_tokens`（或 `max_completion_tokens`）从 1 到回复上限（`--max-tokens`）；没给时取该上限，并按剩余上下文截断。回复只在 EOS/pad 处停止（没有 marker）。
- **渲染。** 参照是固定模板在 `enable_thinking=false`、`preserve_thinking=true` 下的渲染。新的或重建的 session 把整段对话交给模板渲染，再接生成头 `<|im_start|>assistant\n<think>\n\n</think>\n\n`。续接的请求追加上一条回复生成时的原始 token id 和 `<|im_end|>\n`（EOS/pad 之后只追加 `\n`），再追加模板对新 user 消息的渲染 `<|im_start|>user\n{content.strip()}<|im_end|>\n`；启动时用模板检查一次这些片段。一个请求如果等于上一个请求加上一条回复（`strip()` 后比较）再加一条新 user 消息，就续接 session；其他请求（例如 Terminus 的摘要交接、回退后的历史、重发的请求）都会重建 session（`stage_b.rebuild`）。只要每条回复重新编码后仍是它自己的 id，且首尾不带空白，每个 prompt 都等于模板对该请求消息的渲染。
- **响应。** `choices[0].message.content` 是解码后的回复（跳过特殊 token），`finish_reason` 在 EOS/pad 处为 `stop`、达到上限时为 `length`，`usage` 给出 prompt、completion 和总 token 数。
- **上下文上限。** prompt 加上请求的回复长度超过上下文上限（没有 `max_tokens` 时：连一个 token 都放不下）的请求返回 400，正文是 vLLM 自己的文字 `This model's maximum context length is …`。litellm 和 Harbor 会把它映射为上下文长度错误。服务端什么也不变；该 session 的下一个请求照常继续。
- **Session。** 请求的键是它的初始段（如有 system 消息则包括它，再加第一条 user 消息），客户端发送 `X-Session-ID` header 时还包括该 header（litellm 也会把它作为 `session_id` 放进请求体）。Terminus 2 默认不发 header，它的第一个 prompt 含有容器的主机名，所以每个 trial 都是自己的 session。在同一个 header id 下，初始段不同的请求（Terminus 摘要里不带历史的 "questions" 调用）是单独的 session（类型 `aside`），不会重置主 session。一个键在第一个请求时绑定到下一个未用的槽位。如果一个键的槽位上已有排队或进行中的请求，新请求最多等它 30 s，之后返回 409 `SESSION_BUSY`。当槽位因请求失败（客户端断开或取消、引擎失败并返回 500）、被服务端关闭或已服务 256 个请求时，该键换到一个新槽位（rebind：新槽位上的第一个请求从文本渲染，做一次完整 prefill）。客户端在请求仍处于 admission 阶段时断开，同样会使槽位失败（review-fixes-1110；此前该槽位仍然有效，而它对应的引擎 session 已被取消操作作废，客户端的重试会先收到一次 500 `ENGINE_FAILURE`，然后该键才被 rebind）。槽位从不复用；全部用完后，新 session 返回 400 `SESSION_LIMIT`。
- **关闭。** 默认路径上不会有关闭调用。失败的、已换走的、被客户端关闭的以及达到请求上限的槽位立即关闭。只有在没有空余的打开 session（即已有 K 个打开）时才会关闭正在使用的 session：有请求在等打开的 session 时，关闭最久未用、且空闲至少 `--chat-text-evict-after` 秒（默认 30）的 session，优先关 aside，每个等待的请求关一个；没有请求在等时，关闭最久未用、且空闲至少 `--chat-text-idle-timeout` 秒（默认 600）的一个 session，让下一个新 session 有空位。只要还有空位，无论空闲多久都不会关闭正在使用的 session。发送 `X-Session-ID` 的客户端可以自己关闭：`POST /v1/chat-text/close`，请求体 `{"session_id": ...}`（或 `{"key_sha256": ...}`，即初始段规范 JSON 的 SHA-256）关闭该 id 下的所有 session。它返回 200，未知的或已关闭的 id 也返回 200（已关闭时返回第一次的结果并带 `"repeated": true`），其中任一 session 有进行中的请求时返回 409；关闭后的 id 不再被服务。`GET /v1/chat-text/info` 返回选项、计数和各客户端 session。
- **容量。** 最多 K 个 session 同时打开。对 Harbor 的 `-n 16`，设置为 K = 20：四个空余 session 容纳结束的 trial（不会有关闭调用）和 Terminus 的摘要子调用，因此执行长命令的运行中 trial 不会被驱逐。K 决定 `max_num_seqs`、Mamba 池和锁页 Page arena 的大小（4.4 节）。在 P = 16 下，K = 20 有 41 个 Gated-DeltaNet 槽位，Mamba 池 124 个块（5.93 GiB，比 K = 16 多 24 个块即 1.15 GiB）。这样在 benchmark 的 25 GiB KV 下剩下 19,531 个注意力块（可用 19,530），在最早几次 dry run 的 32 GiB 下剩下 26,699 个（比 K = 16 少 1,175 个）。按 4.4 节的规则、4096 token 的回复，20 个 session 在 262,144 token 时：8192 token 预算需要 33.0 GiB（K = 16 时 29.8），加上典型的最后一轮需要 41.2 GiB（36.3），加上最坏的最后一轮需要 57.0 GiB（48.7）；在 131,072 token 时分别为 24.9（21.7）、33.1（28.2）和 48.9 GiB（40.7）；在 65,536 token 时分别为 20.8、29.1 和 44.8 GiB。262,144 token 时，一个停在上限的 session 占 16,385 个块，所以 25 GiB 放不下两个这样的 session：benchmark 依靠容量门控和 `--release-idle-pages`（作业 3461661 两者都没用上：没有释放，也没有容量等待）。262,144 token 时的锁页主机内存：不设 arena 为 324.1 GiB，超过 launcher 的 128 GiB，所以 benchmark 的 `--dram-arena-gib 70` 把它保持在 73.9 GiB（70 GiB 的 1 MiB Page 副本，例如 20 个 57K token 的 session）；131,072 token 时不设 arena 为 164.1 GiB；65,536 token 时（最早几次 dry run）不设 arena 为 84.1 GiB（K = 16 时 67.5），它们的 `--dram-arena-gib 64` 把它保持在 67.9 GiB（例如 20 个 52K token 的 session）。一个槽位服务 256 个请求；一个服务最多绑定 N 个槽位（客户端 session、aside 和 rebind 都算）；生命周期内的请求上限是 max(1,024, 256 × N)。空闲 session 在两轮之间把 Page 留在 GPU 上，又不会有关闭调用，所以打开的 session 的 Page 可能占满池（在 K = 20、32 GiB、65,536 token 下，`full` 的一次 Terminal-Bench dry run 中就是这样，作业 3398129：见 [docs/status.zh-CN.md](status.zh-CN.md) 和[历史](history.zh-CN.md#历史)）：Terminal-Bench 的服务要开 `--release-idle-pages`（4.13 节）。
- **`goal_query` 的输入（暂定：团队的提案，待 Goal 负责人确认）。** Goal 的构造与 SWE-bench 路径相同（第 5 节：维护的 Goal、Qwen3-0.6B 改写器及其 prompt、schema 和 gate、回退）；只有输入不同。这些输入只为 Harbor 0.21.0 的 Terminus 2（版本 2.0.0，JSON 解析器）定义，不适用于其他 agent；每个选择都是同一个模块 `remote_ra_live_v2/terminus2_goal.py` 里的一条具名规则，每一轮都记录它用到的规则。
  - *需求*（Goal 的根目标）：任务指令，从 session 的第一条消息中切出；第一条消息必须是 Terminus 2 的 prompt 模板 `terminus-json-plain.txt`（SHA-256 `89a3dc3a…b54a34`，其字节在该模块中）填入一条指令和一个终端画面后的结果。第一条消息既不是该模板、也不是 Terminus 摘要子调用的 prompt 之一，或者带有 system 消息的请求，在绑定 session 或到达模型之前就返回 400 `GOAL_INPUTS`。
  - *观测*：只取最新一条消息里的终端输出，去掉 Terminus 固定的包装文字（`New Terminal Output:` 或 `Current Terminal Screen:` 标签、解析器警告和完成确认的包装、超时模板中终端状态外的部分）；第一轮以及解析错误或回复超长的提示之后为空，因为它们不含终端输出。不跨轮累积。上一条回复与 SWE 路径一样传入；没有工具名和返回码。
  - *摘要*：摘要之后，Terminus 以 `[第一条消息, questions prompt, questions, 交接]` 继续，第一条消息不变，所以服务端重建同一个客户端 session。Goal 在这里重新开始：需求仍是（第一条消息中的）指令，观测是摘要文本（questions prompt 里的 "Summary from Previous Agent"）；短摘要交接（取其摘要）和不调用 LLM 的交接（取其画面）也一样。摘要子调用（summary、questions、answers 以及短摘要）从不调用改写器。summary 和 answers 调用在主 session 的历史上运行，原样沿用该 session 最近的 Goal（记为 `KEEP`；若该 session 还没有 Goal，例如 rebind 后的新槽位，则用只含需求的 Goal）；questions 和短摘要调用以各自独立的客户端 session 到达，使用只含需求的确定性 Goal。
  - 固定项（也以 `chat_text_goal` 写入 `startup.json` 和 dry run，并以 `goal_inputs` 出现在 `GET /v1/chat-text/info`）：Harbor 0.21.0 的 `terminus-json-plain.txt`、`timeout.txt`、`terminus_2.py` 和 `tmux_session.py` 的 SHA-256，以及各规则名。启动方式是上面的命令加 `--policy goal_query` 和 `--goal-rewriter-model PATH`（或 `--no-goal-rewriter`）；`--admission-overlap K` 和 `--goal-rewriter-batch N` 与 XML 时相同（4.10 节）。
  - **请 Goal 负责人审阅：** 上面这些选择列在请 Goal 负责人审阅的说明里，冻结在 [docs/history/validation-cf2b28d.zh-CN.md](history/validation-cf2b28d.zh-CN.md)；[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 3.4 把它们列为暂定。
- **证据。** 每个 `runtime.jsonl` 行增加 `chat_text`，`chat_sessions.jsonl` 记录每次绑定、换绑和关闭（第 8 节）。`goal_query` 时，`chat_text.goal_inputs` 和每个 `goals.jsonl` 行的 `inputs` 给出该轮用到的规则，`inputs.goal_source` 给出该轮 Goal 的来源（`maintained`、`session_last` 或 `requirement_only`）。
- **测试。** `test_terminus2_goal.py`（在合成的 Terminus 2 请求上检查 `goal_query` 的输入规则；设置 `REMOTE_RA_HARBOR_TERMINUS2_DIR` 时还用 Harbor 自己的文件核对固定项和字面量）、`test_chat_text_protocol.py`（装了 Transformers 时用固定的 Qwen3.6 tokenizer：带转义的 JSON 回复、非 ASCII 指令、有和没有 system 消息，都与 Transformers 的模板渲染相等）、`test_chat_text_server.py`、`test_chat_text_e2e.py`（16 个 session 经打包的 router 和引擎替身一起解码；取消、换绑、空闲关闭和驱逐、上下文拒绝；`goal_query` 的 Terminus 2 session 带摘要，在池压力下配合 `--release-idle-pages` 和替身改写器运行，逐轮检查 Goal 的来源，以及 `GOAL_INPUTS` 拒绝；`REMOTE_RA_SLOW_TESTS=1` 再加 K = 20 下 16 个 session × 8 个 episode，含摘要和 aside，16 个 `random_fill` session × 4 个 episode，以及 16 个 `goal_query` session）和 `test_chat_text_replay.py`；`test_integrate_flags.py` 在三条解码快速路径下运行 chat-text session，并在池压力下配合 `--release-idle-pages` 运行；它还在池压力下把 Terminal-Bench 的 `goal_query` 服务所用的开关放在一起，运行带摘要的 Terminus 2 `goal_query` session（`--admission-overlap 16`、`--goal-rewriter-batch 4`、`--release-idle-pages`、`--decode-fastpath`、16 token 的 Page，以及一个只有释放空闲 Page 后才放得下的单独一轮），每一轮都与不开这些开关的服务相同；该服务连同 CUDA graph `full` 和其他 Terminal-Bench 开关的启动只在启动和启动检查中检查。回放测试需要显式开启（`REMOTE_RA_TB_ATIF_DIR`；轨迹不在本仓库中）：8 月两次 Terminus 2 全量运行的 178 个 trial，每个用固定 tokenizer 回放两次（客户端 token 估计准确、以及偏低 20 %），共 3,363 个请求，其中 83 次上下文拒绝、33 次重建、7 个 aside，每个被服务的 prompt 都等于模板对其消息的渲染。Gemma（4.14 节）：`test_gemma_chat_text.py`、`test_gemma_chat_text_e2e.py`、`test_gemma_chat_text_goal.py`（`goal_query`）和 `test_gemma_chat_text_replay.py`；后者用固定的 Gemma tokenizer 和模板、以 Gemma 的轮次格式把同样 178 个 trial 各回放两次：3,359 个请求，126 次上下文拒绝，43 次重建，7 个 aside，每个被服务的 prompt 都等于参照渲染，去掉 thought channel 后等于 stock vLLM 的 prompt（0 处差异）。
- **stock 对照。** stock vLLM 0.24.0 对照沿用 vLLM 默认的 chat template 参数（与 SWE-bench 的运行相同）：它重新渲染每条较早的回复时不带空的思考块，因此每条历史回复比本服务的 token 流少 4 个 token（`<think>`、`\n\n`、`</think>`、`\n\n`），并且它重新编码回复文本，而本服务保留生成时的 id。stock Gemma 对照的 prompt 同样每条过去的回复少 4 个 token（`<|channel>thought\n<channel|>`）。
- **GPU 运行。** 本协议在 Qwen 和 Gemma 4 上的 Terminal-Bench dry run 见 [docs/status.zh-CN.md](status.zh-CN.md)；本节原来的状态文字冻结在 [docs/history/running-gpu-notes.zh-CN.md](history/running-gpu-notes.zh-CN.md)。

Terminal-Bench 2.0 正式测试的配置记录（全部 89 个任务，Harbor 的 Terminus 2 以 `-n 16` 运行，`full`、`observation_query`、`random_fill` 和 `goal_query` 四个策略，以及一个 stock vLLM 0.24.0 对照，Qwen3.6-27B 和 Gemma 4 31B IT 各一套，Gemma 没有 `full`，两者都是 262,144 token；这是历史配置，当前配置见 [docs/status.zh-CN.md](status.zh-CN.md#当前配置)）和 baseline 与之对齐的检查表见 [docs/TERMINALBENCH_BENCHMARK.zh-CN.md](TERMINALBENCH_BENCHMARK.zh-CN.md)。

<a id="68-开启-thinking第一阶段--enable-thinking未经验证"></a>

### 6.8 开启 thinking（第一阶段，`--enable-thinking`）

`--enable-thinking`（需显式开启；Qwen3.6-27B，`--protocol mswea-xml` 或 `chat-text`）开启 thinking。当前配置使用它（即本节的保留模式），用过它的运行见 [docs/status.zh-CN.md](status.zh-CN.md)，本节早先的功能性检查见 [docs/history/running-gpu-notes.zh-CN.md](history/running-gpu-notes.zh-CN.md)。它的回复上限默认为 4,096 token（含思考内容），与 baseline 一致（4.15 节；`--max-tokens 1536` 保留以前的上限）：容量门控和 launcher 每轮预留 ceil(4096 / P) 个 Page。不加该选项时，所有启动、startup config、证据字节和响应都不变。在 Gemma 4 profile 上（gemma-thinking）：使用 Gemma 的 thinking-on 生成头 `<|turn>model\n`，强制前缀中带有模板的 `<|think|>` system 轮，回复按 vLLM 0.24.0 的 `gemma4` reasoning parser 的规则在其 channel token id 上切分（[docs/thinking-gemma/DESIGN.zh-CN.md](thinking-gemma/DESIGN.zh-CN.md)）。kv-session 协议沿用自己的 `--kv-session-thinking`（6.6 节；在 Gemma 上 `on` 固定两个 thinking-on 生成头）。背后的分析（一份未经审阅、已冻结的报告）见 [docs/history/thinking-on-analysis.zh-CN.md](history/thinking-on-analysis.zh-CN.md)。

- **Pin。** `startup.json` 和 `ready.json` 增加 `thinking`（`remote_ra_live_v2/thinking.py`：生成头 `<|im_start|>assistant\n<think>\n`、模板参数、在 `</think>` 处切分、响应字段、追加规则；dry run 会打印它）。chat-text 的 `chat_text` 记录中 `header_text` 和 `template_kwargs` 换成开启 thinking 的值（`enable_thinking=True, preserve_thinking=True`）；XML 协议以 `enable_thinking=True` 渲染 system 和 task。不改动任何 vLLM 参数。固定模板的 thinking 生成头（chat-text 还有其历史渲染）在启动时检查一次。
- **响应。** 解码出的回复 `R\n</think>\n\nC` 在第一个 `</think>` 处切开：`content` 为 C，`reasoning_content` 和 `reasoning` 都是 R，均保持生成时的原样（不 strip）。没有 `</think>` 的回复（在回复上限处被截断，或在其之前结束）全部算作 reasoning，`content` 为 `""`。vLLM 的 Qwen3 reasoning parser（其源码不在本仓库中）被认为以同样方式处理开启 thinking 且没有结束 token 的回复（content 为 null）。
- **动作标记。** XML 协议在 `</mswea_bash_command>` 处的停止只计 `</think>` 之后的文本：R 中的草稿动作不会让回复停止。服务端没有其他地方扫描回复中的协议标记。
- **历史。** 路由从请求体读取每条 assistant 消息的 reasoning（`reasoning_content`，否则 `reasoning`；vLLM 的请求模型只保留 `reasoning`）。请求在以下情况下延续 session（流中保留 R 的原始 token id）：完全重复上一个请求的消息（包括其中的 reasoning），然后是上一条回复，其 content 等于返回的 C、其 reasoning 等于返回的 R（都在 `strip()` 之后比较；缺少 reasoning 视为空），最后是一条新的 user 消息。经 litellm 的 Mini-SWE-Agent 2.4.2 和开启 `interleaved_thinking` 的 Terminus 2 都这样回传 R。其他请求走协议的重建：chat-text 用开启 thinking 的参数经模板渲染整段对话（客户端回传了 R 的地方带上 R）。丢弃 R 的客户端（Terminus 2 的默认设置）因此每轮都重建，除非服务端使用 `--thinking-history strip`（6.9 节）：它改为截掉最后一条回复再重新 prefill。
- **格式错误之后的历史（XML 协议，keep-rebuild-1116）。** Mini-SWE-Agent 会丢弃不是恰好一个 action 的回复（几乎都是在 reasoning 内被 4,096 token 上限截断、content 为 `""` 的回复），然后发送上一个请求加一条格式错误消息。这样的请求（完全重复上一个请求，再加一条 user 消息）是一次**回滚（rollback）**：它的 prompt 是 session 的 token 流截至被拒回复的生成头之前的部分，逐 token 保留（之前所有 R 和生成的 token id 都在），再接新消息和生成头。引擎把它当作 local turn 0 的重建重新 prefill（`stage_b.rebuild == true`；不保留 session 的 KV：那需要生成头起点处的 Gated-DeltaNet 状态，而一次 completion 之后不再持有它），之后的请求在它之上追加。连续多次格式错误，每次都同样回滚。`goal_query` 的 Goal 跨回滚继续（goal-rollback-1116，6.1 节）：回滚轮复用被丢弃回复所在轮次的 Goal，下一轮用最新的 observation 和保留回复的 content 更新它；把 goal_query 3576565 的请求流离线重放到该分支上，40 次回滚全部延续了 Goal（其中 5 次是连续两次中的第二次）。XML 协议的其他重建以模板渲染 system 和 task，之后的每条消息按模板在 `preserve_thinking=True` 下的渲染（客户端回传了 R 的地方带上 R，文本 trim；Gemma：模板自身的渲染，其中没有 R）。这也改变了普通的追加：自 keep-rebuild-1116 起，开启 thinking 的追加（Qwen，XML 协议）中新的 user 消息按模板的方式 trim 后编码；此前不 trim，因此首尾带空白的 observation 会使该次追加的 prompt 与模板渲染不同（例如末尾换行多出一个 token），observation 区间也相应变长。在重放的五次运行中，session 第一次格式错误之前没有任何调用因此改变（Mini-SWE-Agent 的 observation 首尾没有空白）；关闭 thinking 和 Gemma 的 token 不变（Gemma 的 user 片段本来就 trim）；它们的 `goal_query` Goal 同样跨这种重试延续（6.1 节）。回滚或重建的渲染若在引擎打开新 session 之前失败（被放弃的渲染），会把 session 恢复到渲染之前的状态，重试时仍基于引擎持有的 session 渲染。因此开启 thinking 的 `mswea-xml` Qwen 服务端，每个 prompt 都等于固定模板以 `enable_thinking=True, preserve_thinking=True` 对其消息的渲染（前提是每条回复重新编码后仍是它自己的 id）；`tools/keep_render_check.py` 按这一渲染逐个检查一次运行的调用。**keep-rebuild-1116 之前的运行在 session 第一次格式错误时丢失之前所有的 R**：重建从 content（C）重新编码之后的每条消息，之后的请求在此之上追加。2026-11-16 审计的数字（session 第一次格式错误时及之后的调用 / 全部调用）：full 3583081（`ba4d1ba`）351/1,262（27.8 %，32 个 session 中 12 个，26 次格式错误），recent_fill 3539242（`60c5d21`）683/1,613（42.3 %），observation_query 3539260（`60c5d21`）640/1,548（41.3 %），goal_query 3576565（`ba4d1ba`）869/1,843（47.2 %），akv_rule 3577833（`ba4d1ba`）429/1,411（30.4 %），以及 keep-rebuild-1116 离线重放得到的 ledger_query 3611601（`51b4085`）519/1,548（33.5 %，32 个 session 中 13 个，35 次格式错误）；在 3583081 中，276 条回复的 36,524 个 reasoning token（保留回复 reasoning 的 29.5 %）离开了 session，其 prompt 比保留渲染短 4.2 %。每个 session 第一次格式错误之前，服务端的 prompt 都等于保留渲染（3583081 中 911/911 次调用）。这些运行只在每个 session 第一次格式错误之前是保留模式的运行。
- **回复上限。** `--max-tokens`（1536 或 4096）把 reasoning 计在内：对容量规则和上下文检查来说，每个生成的 token 都是回复 token，下一个 prompt 也包含 R 的 token。
- **Goal 输入（供 Goal 负责人审阅）。** 开启该选项时，Goal 的上一条回复（`previous_action`）只用 content C，不是 R 加 C。
- **证据。** 每行 `runtime.jsonl` 增加 `thinking`：`reasoning_tokens`（完成第一个 `</think>` 的那个 token 之前的 token 数）、`think_end_tokens`（1 或 0）、`content_tokens`、`closed`、reasoning 和 content 的字符数，以及 `history`（`fresh`、`append`、`rollback`（XML 协议，keep-rebuild-1116）、`rebuild_reasoning_dropped`、`rebuild_reasoning_differs`、`rebuild`）。`rollback` 行和 XML 重建行另加 `kept_tokens`（取自 session 自身 token 流的 prompt token 数：回滚时是被拒回复之前的全部前缀，重建时为 0）和 `reencoded_replies`（从文本编码、未保留原始生成 id 的 assistant 消息数）；`fresh` 和 `append` 行的字段不变。对保留了 R 且 R 已闭合的轮次，在每条回复重新编码后仍是它自己的 id 时，固定模板以 `enable_thinking=True, preserve_thinking=True`、R 作为 `reasoning_content` 的渲染与服务端的 prompt 逐 token 相同（XML 协议自 keep-rebuild-1116 起 observation 首尾有空白时也相同：user 消息按模板的方式 trim 后编码；此前只在都没有空白时相同）。
- **测试。** `test_thinking_on.py`（pin 与拒绝、切分、标记停止、两种协议上的追加与重建、在安装了 Transformers 时用固定 Qwen3.6 tokenizer 检查模板等价，以及经打包 router 和引擎 fake 的 chat-text 与 XML 端到端 session，其中包括一次格式错误回滚）；`test_keep_rebuild.py`（单条消息的保留渲染与模板对照，在 reasoning 内截断的回复、已闭合但没有 action 的回复、连续多次格式错误和第一条回复被丢弃之后的回滚，其他重建，关闭 thinking 时不变，记录字段，容量等待的 checkpoint，Gemma，回滚之后 query hook 看到的历史，以及 `tools/keep_render_check.py`）；`test_goal_rollback.py`（`goal_query` 的 Goal 跨回滚：等于没有被丢弃回复的 session，开启和关闭 thinking，使用替身 rewriter、Hugging Face 与 vLLM rewriter 子进程的 fake，以及 `--goal-overlap` 关和开）。

<a id="69-开启-thinking-并丢弃-reasoning--thinking-history-strip已做-gpu-检查尚无-benchmark"></a>

### 6.9 开启 thinking 并丢弃 reasoning（`--thinking-history strip`）

`--thinking-history strip`（需显式开启，并配合 `--enable-thinking`；仅限 Qwen3.6-27B 的 `--protocol chat-text`）提供默认模板渲染的历史：`enable_thinking=True`、不加 `preserve_thinking`，不论客户端回传什么 reasoning，每条过往回复都只渲染其 content（即 Terminal-Bench Vanilla 基线的输入）；它的 GPU 检查见 [docs/status.zh-CN.md](status.zh-CN.md)，本节早先的功能性检查见 [docs/history/running-gpu-notes.zh-CN.md](history/running-gpu-notes.zh-CN.md)。不加该选项时沿用第一阶段的保留模式（6.8 节），所有启动、startup config、证据字节和响应都不变。在 `mswea-xml` 和 `kv-session` 上，（本首个版本中）与 `--gdn-device-handoff`、`--phase1-in-step` 同时使用时，以及与 `--policy recent_tools`、`akv_rule` 同时使用时被拒绝（这两个策略的回复分类无法从去掉 reasoning 的历史中读出，见第 5 节）。在 Gemma 4 profile 上（gemma-thinking），transcript 一侧已经实现（每条过往回复为 `<|turn>model\n` + 其 content + `<turn|>\n`，即 stock 的 thinking-on 渲染），并拒绝与 `--async-scheduling` 同时使用；它的引擎一侧在 GPU 上保留切点处的滑动窗口（[docs/thinking-gemma/DESIGN.zh-CN.md](thinking-gemma/DESIGN.zh-CN.md) 第 4 节）：`--window-hold-slots H`（1 到同时运行的 session 数；不给时取放得下的最大值，连一个都放不下时拒绝启动）预留 H 个 hold slot，每个为一个窗口（16-token Page 下 812.5 MiB）。一个 strip 轮次从准入到提交都占一个 hold slot（phase 1、停在 Goal 处的 goal_query 准入和冷 prefill 都算在内），因此同时在途的 strip 轮次最多 H 个，其余在容量门处等待：使用 `--admission-overlap K` 时同时在途的准入最多 min(K, H) 个，一个 Goal 批次最多 H 个调用（H = 1 时服务一次只运行一个轮次）。默认值让池 0 只剩选择性规则的最小值，所以较小的 H 能同时服务更多长上下文（设计说明 4.3 节）；H 固定为 `kv_layout.hold_slots`，dry run 会打印（`window_hold`）。设计与决策见 [docs/thinking-cut/DESIGN.zh-CN.md](thinking-cut/DESIGN.zh-CN.md)（方案 (a) 的形式 E）；规则在 `remote_ra_live_v2/thinking_cut.py`。

- **在截断点提交。** 每次完成时，session 在 X 处提交：X 是本轮生成头的起点（observation 区间的终点，N - 5）。floor(X/P) 之前的 Page 被封存，第 X/P 页成为 tail，其有效终点为 X；Gated-DeltaNet carry 是 X 处的状态（延续轮次取 phase 1 的状态；session 的首轮在 vLLM 按 Mamba 对齐切分之后让 prefill 的一次 forward 结束于 X，并在那里捕获状态）。生成头和回复（reasoning 与 content）只属于本轮：截断点之后的 Page 不封存、不拷贝、不参与排序，也不捕获回复末尾的 carry。
- **重新 prefill。** 下一个请求的 prompt 是已提交的流、客户端回传的回复按模板渲染的结果（`<|im_start|>assistant\n` + trim 后的 content，若含 `</think>` 则取最后一个之后的部分，+ `<|im_end|>\n`）、新的 user 消息和生成头。其稳定终点为 X + 3（生成头的前三个 token，每轮比较一次，不一致则拒绝该请求），因此轮次几何不变，重新编码的回复归入下一个 observation：phase 1 将它与 observation 一起计算，作为一次原生 forward（16,384 个 token；更长时该轮像以前一样失败）。于是每个 prompt 都等于默认模板对其消息的渲染，在回复上限处被截断的回复（其 content 为空）之后也是如此。
- **延续。** 请求在以下情况下延续 session：重复上一个请求（role 与 content），然后是上一条回复，其 content 等于返回的 C（`strip()` 之后比较），最后是一条 user 消息；其他请求从文本重建。响应和 Goal 的上一条回复与 6.8 节相同。
- **证据。** `startup.json` 和 `ready.json` 固定带 `history: strip`、模板参数为 `enable_thinking=True` 的 `thinking` 记录；worker 回显 `thinking_history`。native 文件的 plan 记录 `commit_end`（X）和封存规则 `floor(X/P)`；文件增加 `truncate`（截断点，以及被丢弃的分支：起止位置、生成头 id、生成的 id、`</think>` 的位置）和延续轮次的 `reprefill`（重新 prefill 的回复的起止）。`runtime.jsonl` 中 `stage_b.computed_end` 和 `sealed_end` 为提交的终点，`stage_b.phase2_end` 为回复的终点（N + g - 1），并增加 `truncate` 以及 `thinking.cut` / `thinking.reprefill`。离线 Terminal-Bench verifier 逐个请求检查这些字段：它核验了作业 3499759 的 dry run（其 `observation_query` 服务上有 294 个被截断的轮次）。
- **测试。** `test_thinking_cut.py`（在安装了 Transformers 时用固定 Qwen3.6 tokenizer 检查过往回复的模板渲染、transcript、pin 与拒绝、引擎 fake 上的 chat-text session，检查每次完成的截断点、carry、证据以及每个决策与稠密重算的一致性，其他 chat-text 场景、decode fast path、`--deferred-completion-io`、`--admission-overlap`、`--release-idle-pages`、变异测试；设置 `REMOTE_RA_TB_ATIF_DIR` 时，回放加入 reasoning 的 178 个 Terminal-Bench trial）。
