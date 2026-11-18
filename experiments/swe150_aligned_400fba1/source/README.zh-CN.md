# vLLM 0.24.0 的 Selective-prefix 策略

[English](README.md) | 简体中文

这是面向多轮工具调用 agent 的服务端 prefix Page 选择机制，基于 vLLM 0.24.0：每个服务运行十一种[策略](#策略)之一，由策略决定下一轮从 CPU 内存恢复 session KV 历史中哪些已封存的 Page。

> **现状在哪里看。** `main` 是 2026-11-17 合并进来的 integrate-1117：runtime `2fb7f62`（integrate-1116 加上 session 池的任务上限 256 和 150 个实例的列表）。其中有些部分还没有在 GPU 上跑过（[docs/known-limits.zh-CN.md](docs/known-limits.zh-CN.md) 5.7）。之前的 `main`（`7892db3`，runtime `4ebff97`）是 tag `main-integrate-1112`；目前还没有稳定 tag。按模型、benchmark 和策略列出的 GPU 运行（每个作业的判定、数字是否经过审计）以及当前配置见 [docs/status.zh-CN.md](docs/status.zh-CN.md)。哪些还不能用、哪些要小心用，见 [docs/known-limits.zh-CN.md](docs/known-limits.zh-CN.md)。改动怎样进入 `main`、一个数字什么时候可以用，见 [docs/release-gates.zh-CN.md](docs/release-gates.zh-CN.md)。

## 各模型 × benchmark 的状态

原来在这里的状态表自 2026-11-16（docs-reorg-1116）起移到 [docs/status.zh-CN.md](docs/status.zh-CN.md)，并扩展为每个策略一行；本页不再记录作业结果。

## 稳定配置：Qwen 跑 SWE-bench Verified

用 Qwen3.6-27B 跑 SWE-bench Verified 时，用下面这套配置。它在 runtime `ba4d1ba`（2026-11-15 之前 `main` 的 runtime）上开启 thinking 跑了 SWE-bench Verified 的前 32 个实例：作业 3576565，`SWE32_PASS_VERIFIED`，解决 32 个中的 19 个（[docs/status.zh-CN.md](docs/status.zh-CN.md)）。64 个任务的运行在它之上再加 `--async-scheduling`（见 [docs/status.zh-CN.md](docs/status.zh-CN.md) 中的当前配置）。

- **硬件：** 一张 GH200（Slurm `--gpus=1 --cpus-per-task=16 --mem=113560M`）。
- **模型：** Qwen3.6-27B，只用文本部分。Goal 由 Qwen3-0.6B 在 vLLM 子引擎里构造（KV 3 GiB），和主模型的 phase 1 重叠执行。
- **Agent：** Mini-SWE-Agent 2.4.2，XML 协议，同时 16 个实例，最多 120 步，温度 0.0，不重试，请求超时 3,600 秒。
- **服务**（`$TASKS` 是实例的任务文件，见第 [4.1](docs/running.zh-CN.md#41-单个服务) 节；`$GOAL_MODEL` 是 Qwen3-0.6B 的快照目录）：

```sh
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy goal_query --protocol mswea-xml \
  --output "$RUNS/goal" --port 8000 \
  --kv-cache-gib 25 --page-tokens 16 --kv-layout decoupled --cuda-graphs full \
  --max-running-sessions 16 --max-tokens 4096 --max-model-len 262144 \
  --dram-arena-gib 70 --connector-pool-gib 1 \
  --decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io --language-model-only \
  --enable-thinking --active-token-budget 8192 --admission-overlap 16 \
  --goal-rewriter-model "$GOAL_MODEL" --goal-rewriter-engine vllm \
  --goal-rewriter-guided-backend lm-format-enforcer --goal-rewriter-vllm-kv-gib 3 \
  --goal-rewriter-batch-invariant --goal-overlap --start
```

**当前配置**（64 任务的运行，[docs/status.zh-CN.md](docs/status.zh-CN.md#当前配置)）：上面的命令加上 `--async-scheduling`。在 SWE-bench Verified 上带它跑过的只有 `recent_fill`（作业 3611917）；`goal_query` 还没有。

- **其他策略：** 同一条命令换一个 `--policy`，并去掉 Goal 的选项（各 `--goal-rewriter-*` 选项和 `--goal-overlap`：对没有 Goal 的策略，launcher 拒绝 `--goal-rewriter-engine vllm` 和 `--goal-overlap`）；`full` 在 25 GiB 里跑 16 个 session 时需要 `--release-idle-pages`，`akv_rule` 也应该开（[docs/known-limits.zh-CN.md](docs/known-limits.zh-CN.md) 2.4）。当前配置下 64 个任务的运行再加 `--async-scheduling`（[docs/status.zh-CN.md](docs/status.zh-CN.md#当前配置)）。

## 24 个 session 的配置：组里的 150 个实例的运行（2026-11-17，尚未经过 GPU 验证）

组里在 SWE-bench Verified 前 150 个实例上、同时 24 个 session 的运行所用的配置，自 2026-11-17 起为 `main`，即 integrate-1117：在 integrate-1116 之上把 session 池的任务上限从 64 提高到 256（第 [4.4](docs/running.zh-CN.md#44-一个服务上的多个-session) 节）。**尚未经过 GPU 验证：** XML 协议上超过 16 个 session 还没有在 GPU 上跑过（[docs/known-limits.zh-CN.md](docs/known-limits.zh-CN.md) 2.1）。只有 24 个 session 的 smoke 和 150 个实例的运行都通过之后，它才成为稳定配置；在此之前，结果来自上面的[稳定配置](#稳定配置qwen-跑-swe-bench-verified)。分工：stock 对照、`full` 和 `ledger_query` 在这个 runtime 上运行；两位成员在各自合并了 integrate-1117 的分支上运行自己的策略。

- **硬件：** 一张 GH200（Slurm `--gpus=1 --cpus-per-task=16 --mem=113560M`）。
- **模型：** Qwen3.6-27B，只用文本部分，thinking 开。
- **Agent：** Mini-SWE-Agent 2.4.2，XML 协议，SWE-bench Verified 按数据集顺序的前 150 个实例（`examples/swebench/verified-150.txt`：它们的 id 和镜像 digest，作为 `IMAGE_PINS` 交给 `run_agents.sh` 和 `evaluate.sh`），同时 24 个，最多 120 步，温度 0.0，不重试，请求超时 4,400 秒（作业 harness 在超过 16 个 session 时的规则：K = 24 时的准入等待 300 + 120 × 23 = 3,060 秒，再加 1,340 秒；stock 对照没有准入闸门，保持 3,600 秒）。
- **`full` 的服务**（`$TASKS` 是 150 个实例的任务文件：`examples/swebench/build_tasks.py --pool`，见第 [4.1](docs/running.zh-CN.md#41-单个服务) 节）：

```sh
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy full --protocol mswea-xml \
  --output "$RUNS/full" --port 8000 \
  --kv-cache-gib 25 --page-tokens 16 --kv-layout decoupled \
  --max-running-sessions 24 --max-model-len 262144 --dram-arena-gib 68 \
  --enable-thinking --async-scheduling --release-idle-pages --start
```

- **`ledger_query` 的服务**（按观察排序的插件策略；只用于 Qwen 的 XML 协议）：同一条命令，改为 `--policy ledger_query --active-token-budget 8192`（并用它自己的 `--output`）。它的 ledger 保持默认上限，即填充预算的 50 %，所以不给 `--ledger-cap-percent`。不加 Goal 选项，也不加 `--admission-overlap`：所有按观察排序或填充类的策略都用这种形式。
- **Goal 策略**（成员的 Goal 策略所用的形式；24 个 session 的 smoke 就这样跑 `goal_query`）：`full` 的命令再加 `--active-token-budget 8192 --admission-overlap 24` 和稳定命令中的 Goal 选项（`--goal-rewriter-model "$GOAL_MODEL" --goal-rewriter-engine vllm --goal-rewriter-guided-backend lm-format-enforcer --goal-rewriter-vllm-kv-gib 3 --goal-rewriter-batch-invariant --goal-overlap`）。
- **这些命令依赖的默认值**（默认开启的速度选项，[docs/reference.zh-CN.md](docs/reference.zh-CN.md#415-默认开启的速度选项) 4.15 节）：CUDA graph `full`、三个解码快速路径、`--deferred-completion-io`、`--language-model-only`、1 GiB 的 connector 池，以及开 `--enable-thinking` 时 4,096 token 的回复上限。稳定命令把它们都写了出来；这里只写非默认的选项，两种写法得到的启动配置相同。
- **`--dram-arena-gib 68` 而不是 70：** 作业 harness 在 `--mem=113560M` 的 110.9 GiB 中，除了服务的 pinned host 内存（arena、1 GiB 的 connector 池、每个 session 一个 154 MB 的 Gated-DeltaNet carry）和其余进程的 32 GiB 之外，还要留出 5 GiB；K = 24 时 70 GiB 的 arena 只剩 4.46 GiB，68 GiB 剩 6.46 GiB。
- **stock 对照：** 未修改的 vLLM 0.24.0，用 [docs/benchmarks/swe-bench.zh-CN.md](docs/benchmarks/swe-bench.zh-CN.md) 中 262,144 token 的 `STOCK_CMD`（25 GiB KV），改为 `--max-num-seqs 24`，thinking 开，并保留之前的 reasoning：作业 harness 中保留 reasoning 的 stock 计划（`stockkeep64t`，24 个序列时为 `stockkeep64k24`）给服务加上 `--reasoning-parser qwen3` 和 `--default-chat-template-kwargs '{"preserve_thinking": true}'`，它的 agent 发送 `chat_template_kwargs: {"enable_thinking": true}`、`max_tokens` 4096、不带 stop 字符串，并把之前每条回复的 reasoning 也放进 `reasoning` 字段（vLLM 0.24.0 只从这个字段读取之前回复的 reasoning；Mini-SWE-Agent 2.4.2 把它作为 `reasoning_content` 发送）。丢弃 reasoning 的 stock 参照（作业 3583822，[docs/status.zh-CN.md](docs/status.zh-CN.md)）恰好在这两点上不同：没有 `preserve_thinking` 和 `reasoning` 字段时，之前的每条回复渲染时都不带 reasoning。
- **为什么要 `--release-idle-pages`：** 作业 3613537（sess24i，2026-11-16，runtime `7a8c36e`：24 个 session，`goal_query`，不释放空闲 session）永久停住了。一次格式错误的回滚之后，下一轮在 phase 1 需要它的全部历史，约 1,075 个注意力块，而空闲的只有 211 个，也没有正在运行的请求（`CAPACITY_STALL`）。开 `--release-idle-pages` 时，容量闸门正是在这种情况下释放空闲 session 的常驻 Page（[docs/known-limits.zh-CN.md](docs/known-limits.zh-CN.md) 2.6）。
- **agent 的 prompt 和环境：** agent 的命令在仓库的 `testbed` 环境激活后运行，prompt 给出每个仓库的测试命令（PoC prompt 的测试一节是为 Astropy 写的，而 150 个实例中有 128 个是 Django）：`examples/swebench/run_agents.sh` 的 `REPO_ENV=testbed`（[docs/known-limits.zh-CN.md](docs/known-limits.zh-CN.md) 5.6）。每个配置（stock 对照、`full`、`ledger_query`、成员的策略）都这样运行；其结果不能与不带它的运行比较。
- **本 runtime 的服务的 agent**（在 agent 主机上，在带下面客户端行的 `examples/swebench` 副本中运行，与 [docs/benchmarks/swe-bench.zh-CN.md](docs/benchmarks/swe-bench.zh-CN.md) 相同；`IDS` 是按 `verified-150.txt` 顺序的 150 个实例 id，`HF_DATASETS_CACHE` 由 `build_tasks.py` 为它们填好）：

```sh
MAX_RUNNING=24 MAX_TOKENS=4096 STEP_LIMIT=120 REPO_ENV=testbed \
  IMAGE_PINS=/abs/path/on/agent/swebench-client/verified-150.txt \
  MINI_PYTHON=/abs/path/on/agent/mini-env/bin/python MSWEA_CONFIG_DIR=/abs/path/on/agent/mswea-config \
  HF_DATASETS_CACHE=/abs/path/on/agent/swebench-datasets \
  MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1 REQUEST_TIMEOUT=4400 \
  /abs/path/on/agent/swebench-client/run_agents.sh "$AGENT_SOCKET" 20000 "$AGENT_OUT" $IDS
```

副本的 `model_override.yaml` 在 `max_tokens` 之后有这三行（4,400 秒的请求超时，不重试）：

```yaml
    timeout: 4400
    max_retries: 0
    num_retries: 0
```

`run_agents.sh` 的默认值是 pilot 的值（90 步、1,536 token 的回复、所有实例一个 runner、32 个固定镜像、`bash -c` 和 PoC prompt），会得到另一个配置：上面的每个变量都要设置。作业 harness 的计划 `full150k24e` / `ledger150k24e` 渲染出同样的 model override（`max_tokens` 4096、`step_limit` 120、`timeout` 4400、`max_retries` 和 `num_retries` 为 0、`interpreter: ["bash", "-lc"]`、同一个 prompt 文件），另外还发送 `chat_template_kwargs: {"enable_thinking": true}`，它对本 runtime 的服务没有作用（thinking 由 `--enable-thinking` 设置）。stock 对照的 agent 是作业 harness 的（见上；超时 3,600 秒）：`run_agents.sh` 的 `STOCK=1` 是 thinking 关闭的 stock 约定，这里不适用。
- **状态：** 尚未经过 GPU 验证。24 个 session 的 smoke（作业 3616345，在 `7a8c36e` 上：`goal_query`，开空闲释放和 `--admission-overlap 24`）跑完了全部 24 个任务，没有停住；它的判定为 FAIL 只是因为作业 harness 的检查有缺陷，该缺陷已修复（[docs/status.zh-CN.md](docs/status.zh-CN.md)）。下一步：stock 对照、`full` 和 `ledger_query` 的 150 个实例的运行，以及成员在各自分支上的策略的运行。

## 该用什么

- **第一次运行：** `cf2b28d` 的五个策略（`full`、`random50`、使用默认 Hugging Face Goal rewriter 的 `goal_query`、`observation_query`、`random_fill`），按[快速上手](#快速上手)的方式启动；这时开启的速度默认值就是 `cf2b28d` 上经过 GPU 验证的 benchmark 服务所用的开关。
- **Qwen 开启 thinking 跑 SWE-bench Verified：** 用[稳定配置](#稳定配置qwen-跑-swe-bench-verified)。
- **稳定配置的命令之外属于实验性的：** `--goal-overlap`、`--goal-rewriter-engine vllm` 和 `--enable-thinking`（稳定配置的命令在 Qwen 上用了这三个）；`--async-scheduling`、`--thinking-history strip`（两个 profile）、`--window-hold-slots`、`--phase1-graph-tokens`、`--gdn-inplace-crossing`、超过 16 个 session、Gemma 4 开启 thinking、策略 `recent_fill`、`recent_tools` 和 `akv_rule`、插件策略以及 `--token-gather`。其中哪些在 GPU 上跑过、跑得怎样，见 [docs/status.zh-CN.md](docs/status.zh-CN.md)。64 个任务的运行所用的当前配置是稳定配置的命令再加 `--async-scheduling`（[docs/status.zh-CN.md](docs/status.zh-CN.md#当前配置)）。

<a id="main-包含什么runtime-4ebff97"></a>

## `main` 包含什么（runtime `2fb7f62`，自 2026-11-17 起）

runtime `2fb7f62` 就是 integrate-1117：`4ebff97` 加上 integrate-1114（`ledger_query`、标记页、token 聚集）、integrate-1116（格式错误回滚时完全相同的历史、`--gdn-inplace-crossing`、最多 32 个 session、`ledger_query` v2、回滚时保留 Goal、`serve.py` 的帮助文字和文档的重组）以及 integrate-1117（session 池的任务上限 256、`examples/swebench/verified-150.txt`）。`4ebff97`（tag `main-integrate-1112`）是 `ba4d1ba` 加上七个分支（策略注册表、2026-11-10 代码审查的修复和主机工作精简，始终开启；针对在 `max_tokens` 处被截断的回复的 `--async-scheduling` 修复、阶段 1 的 CUDA graph 和 Gemma 4 开启 thinking，需显式开启；Terminal-Bench 客户端给 Gemma 的 262,144 token 上下文）。每个分支的条目和测试见[历史](docs/history.zh-CN.md#版本记录)；各部分的 GPU 作业见 [docs/status.zh-CN.md](docs/status.zh-CN.md)。自 `4ebff97` 以来还没有在 GPU 上跑过的部分见 [docs/known-limits.zh-CN.md](docs/known-limits.zh-CN.md) 5.7。

## `main` 的规则与稳定 tag

2026-11-15 的规则（需显式开启的改动在它自己的 GPU 测试和一致性证明之后进入，共用路径上的改动在每个模型一次 GPU 冒烟之后进入，每次推送都需要维护者批准）和稳定 tag 见 [docs/release-gates.zh-CN.md](docs/release-gates.zh-CN.md)；`main` 写明的稳定提交见 [docs/status.zh-CN.md](docs/status.zh-CN.md)。

## 快速上手

### 需要准备什么

本仓库加一个干净的 vLLM 0.24.0 环境，就足以完成安装和启动服务：

1. **本仓库**（Git checkout）：`git clone https://github.com/CFXM/vllm-0.24.0-selective-prefix-policies.git "$REPO"`，`$REPO` 见下文。本仓库是私有的，访问权限由维护者授予。
2. **一个 Python 3.12 环境，装有未经修改的官方 `vllm==0.24.0` wheel，以及恰好 `transformers==5.13.0` 和 `tokenizers==0.22.2`。** 其他版本会被 `install.py` 拒绝。直接 `pip install vllm==0.24.0` 可能会解析出别的 Transformers 版本，所以两者都要固定版本（[2.1 节](docs/install.zh-CN.md#21-基础环境)给出经过验证的做法）。不支持从源码或 editable 方式安装的 vLLM：安装器的 wheel RECORD 检查会失败。
3. **Qwen3.6-27B 权重**，revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`；只有使用 Goal rewriter 的 `goal_query` 才需要 **Qwen3-0.6B**（也可以用 `--no-goal-rewriter` 启动 `goal_query`）（[2.2 节](docs/install.zh-CN.md#22-模型)）。
4. **Linux，带 `gcc` 或 `clang`，以及该解释器的 Python 头文件**（Triton 在运行时编译 kernel），外加一个**较短的 `TMPDIR`**（[第 3 节](docs/install.zh-CN.md#3-运行环境)）。只在一张 NVIDIA GH200（Linux aarch64）上验证过。安装器不检查架构和 GPU，但其他平台都未测试。

安装器从不修改 vLLM 环境。它另建一个 release 目录（已安装 `vllm/` 包的副本加上本 overlay），所以原环境仍可以当普通 vLLM 使用。启动服务不需要联网：`serve.py` 会设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。

运行 SWE-bench harness 还需要 Mini-SWE-Agent 2.4.2 和一台装有 SWE-bench 镜像的 Docker 主机，这超出了安装和启动服务的范围（[第 7 节](docs/benchmarks/swe-bench-mini-swe-agent.zh-CN.md#7-用-mini-swe-agent-242-跑-swe-bench-verified)）。

### 从零到一个冒烟请求

下面的命令使用这些占位变量，后面各节沿用它们。请使用绝对路径。

```sh
REPO=/abs/path/to/vllm-0.24.0-selective-prefix-policies
VENV=/abs/path/to/vllm-env
MODELS=/abs/path/to/models
RELEASE=/abs/path/to/releases/selective-prefix
RUNS=/abs/path/to/runs
cd "$REPO"
```

1. 建好基础环境 `$VENV`（[2.1 节](docs/install.zh-CN.md#21-基础环境)），并把模型放到 `$MODELS`（[2.2 节](docs/install.zh-CN.md#22-模型)）。
2. 运行快速测试（`run_tests.py`；完整测试 `run_tests.py --all` 和 `check_git_release.py --ref HEAD` 留到 push 之前），检查环境，然后创建 release（[2.3](docs/install.zh-CN.md#23-测试) 和 [2.4 节](docs/install.zh-CN.md#24-安装-release)）。运行测试前先导出[第 3 节](docs/install.zh-CN.md#3-运行环境)的 `PATH` 和 `LD_LIBRARY_PATH`（下面前两行），发布仓库的 GPU 验收（作业 3504770，`RLAC_PASS_VERIFIED`）就是这样跑它的快速测试的。快速默认集只需要 CPU（CI 在 GitHub 的 ubuntu-24.04 runner 上跑它）；完整测试（`--all`）请在计算节点上跑，不要在共享的登录节点上跑：

```sh
export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
"$VENV/bin/python" -B run_tests.py
mkdir -p "$(dirname "$RELEASE")" "$RUNS"
"$VENV/bin/python" -B install.py --site-packages "$VENV/lib/python3.12/site-packages" \
  --model-dir "$MODELS/qwen3.6-27b" --release-dir "$RELEASE"
"$VENV/bin/python" -B install.py --site-packages "$VENV/lib/python3.12/site-packages" \
  --model-dir "$MODELS/qwen3.6-27b" --release-dir "$RELEASE" --apply
```

3. 在 GH200 节点上、Slurm 作业的 shell 里（`cf2b28d` 的五项 GPU 检查都是 GH200 批处理分区 `ghx4` 上的 batch 作业，每个一张 GH200；[4.5 节](docs/running.zh-CN.md#45-gpu-作业slurm)把这一步作为 batch 作业运行）：在这个 shell 里重新运行上面的占位变量代码块，设置运行环境（[第 3 节](docs/install.zh-CN.md#3-运行环境)），创建较短的 `TMPDIR`（私有，`len(TMPDIR) + 37` 不超过 107 字节），然后先 dry run，再启动，等待 READY，发送一个请求，然后用 `SIGTERM` 停止（[4.1 节](docs/running.zh-CN.md#41-单个服务)和[第 8 节](docs/design.zh-CN.md#8-每个服务的证据)）：

```sh
export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
export TMPDIR=/abs/short/tmp
mkdir -p -m 700 "$TMPDIR"
export VLLM_USE_FLASHINFER_SAMPLER=0
unset CC VLLM_RPC_BASE_PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
python -B serve.py --release-dir "$RELEASE" --task examples/task.json --policy full \
  --output "$RUNS/full" --port 8000
python -B serve.py --release-dir "$RELEASE" --task examples/task.json --policy full \
  --output "$RUNS/full" --port 8000 --start > "$RUNS/full.server.log" 2>&1 &
SERVER=$!
until python -B examples/slurm/server_checks.py ready "$RUNS/full" full "$RELEASE"; do
  [ $? -eq 1 ] || { echo "READY for something else"; kill -TERM "$SERVER"; wait "$SERVER"; exit 1; }
  kill -0 "$SERVER" || { echo "server exited before READY"; exit 1; }
  sleep 5
done
python -B examples/http_client.py --base-url http://127.0.0.1:8000/v1 --messages examples/messages.json
kill -TERM "$SERVER"
wait "$SERVER"
python -B examples/slurm/server_checks.py cleanup "$RUNS/full"
```

客户端以 JSON 输出一个 chat completion，回复在 `choices[0].message` 里，最多 16 个 token（它发送 `max_tokens` 16 和 `temperature` 0）。`wait` 必须返回 0，最后一条命令必须输出 `CLEAN_SHUTDOWN`。如果没有到达 READY，循环会停止服务并退出 shell（`exit 1`），与 [4.2 节](docs/running.zh-CN.md#42-一张-gpu-上依次跑已验收运行的四个策略)的循环相同。`--output` 必须是尚不存在的目录（其父目录必须存在）：每次启动都用一个新目录（dry run 不会创建它）。在 Slurm 作业里，`examples/slurm/serve_gh200.sh` 会替你完成第 3 步（[4.5 节](docs/running.zh-CN.md#45-gpu-作业slurm)）。冒烟测试的服务使用默认开启的速度选项（CUDA graph `full` 和解码快速路径，[4.15 节](docs/reference.zh-CN.md#415-默认开启的速度选项)），所以启动时还要捕获 CUDA graph；[4.1 节](docs/running.zh-CN.md#41-单个服务)给出的 READY 时间是 eager 服务（`--no-speed-defaults`）的时间。

对 `goal_query`、`observation_query` 和 `random_fill`，在两条 `serve.py` 命令里都加上 `--page-tokens 16 --kv-layout decoupled`（[4.6 节](docs/reference.zh-CN.md#46-解耦-kv-布局与-16-token-的-page)）。单独的 `--page-tokens 16` 就已选用解耦 KV 布局（16、32 和 64 token 的 Page 总是使用它；`--kv-layout decoupled` 只在 784 时有作用，是一种验证模式），benchmark 服务两个参数都传。开启 thinking 的 `recent_fill` 和 `akv_rule` SWE-bench 运行（作业 3539242 和 3577833）也用了同样的 16 token Page。这时 dry run 显示的 vLLM 命令中有 `--block-size 16 --mamba-block-size 784 --additional-config ...`：

```sh
python -B serve.py --release-dir "$RELEASE" --task examples/task.json --policy observation_query \
  --page-tokens 16 --kv-layout decoupled --output "$RUNS/obs" --port 8000
```

## 策略

每个服务运行一个策略（`--policy`）；[docs/design.zh-CN.md](docs/design.zh-CN.md#5-query-策略的选页规则) 第 5 节是策略目录：每个策略的规则、它的选项、它服务的协议和 profile，以及哪些不能组合。术语（session、可用页和必保留页、预算、轮次流程）也在那里定义。

| `--policy` | 选择方式 |
| --- | --- |
| `full` | 恢复全部可用历史 Page（参照组）。 |
| `random50` | 按种子随机选一半可用 Page。 |
| `goal_query` | Online Selector V2：必保留页、recent 页，以及与维护的检索 Goal 的 query 向量最接近的页填满预算。 |
| `observation_query` | 同一个 selector；query 是模型在最新 observation 末尾的 hidden state。 |
| `random_fill` | 两个 query 策略的对照：轮次流程和容量相同，其余按带种子的随机顺序填满。 |
| `recent_fill` | 容量相同，其余从新到旧填满：一个以最新已封存 Page 结尾的连续窗口。 |
| `recent_tools` | 与 `recent_fill` 相同，但先取工具调用的页，再取工具返回，最后取 reasoning，各自从新到旧。 |
| `akv_rule` | AKV 的规则：所有回复和最近 12 个工具返回，没有预算。 |
| `ledger_query` | `observation_query` 的填充先给最近一次编辑和每个文件最近一次读取留出空间（插件；Qwen profile 的 XML 协议）。 |
| `goal_query_markers` | `goal_query` 的填充先取含协议标记的页（插件；Qwen profile 的 XML 协议）。 |
| `observation_query_markers` | 同样的标记页，放在 `observation_query` 的填充之上。 |

`--token-gather markers` 或 `separators`（`goal_query` 和 `observation_query` 的选项）在同一预算内于选中页之外保留单个历史 token（[docs/policy-registry.zh-CN.md](docs/policy-registry.zh-CN.md#6-token-级选择方案-a标记页方案-btoken-聚集) 第 6 节）。

## 协议与仓库内容

使用默认协议（`--protocol mswea-xml`，Mini-SWE-Agent 的 XML action）时，Harness 通过普通的 OpenAI Chat Completions 端点通信，每轮发送完整的、不断增长的消息历史。policy、session、Goal、向量和 Page 选择都不会出现在 HTTP 里。其他协议和 session 池会带上 session 标识：kv-session 协议有自己的端点和 session id，chat-text 协议接受 `X-Session-ID` 请求头，session 池用 `POST /v1/sessions/close` 关闭一个 session（[第 6 节](docs/running.zh-CN.md#6-harness-接口)）。

本仓库包含核心 runtime、安装器和启动器、CPU 测试（`tests/`，由 `run_tests.py` 运行；[docs/install.zh-CN.md](docs/install.zh-CN.md#23-测试)），另有示例：一个冒烟客户端、一个 Slurm 作业脚本、我们的 SWE-bench pilot 围绕 Mini-SWE-Agent 使用的衔接脚本（[第 7 节](docs/benchmarks/swe-bench-mini-swe-agent.zh-CN.md#7-用-mini-swe-agent-242-跑-swe-bench-verified)），以及 MemoryAgentBench 和 Terminal-Bench 2.0 的客户端（`examples/memoryagentbench/`、`examples/terminalbench/`；[docs/benchmarks/](docs/benchmarks/)）。仓库里还带有 pilot 使用的 PoC（本 runtime 在 vLLM 上重新实现的概念验证：DRAM-KV 原型 `dram-backed-kv`，含其 Selector V2 和 Goal）的 SWE-bench runner 和 prompt（`examples/swebench/upstream/`）。

另有两个目录不属于 runtime。`harnesses/` 是跑过发布冒烟测试的四个 GPU 作业 harness（`perf-swe32`、`perf-gemma-swe32`、`perf-tb16-strip`、`perf-mab`）的快照；发布导出时会去掉它（[harnesses/README.zh-CN.md](harnesses/README.zh-CN.md)）。`tools/` 放开发工具：`policy_parity.py`（策略的跨源码树探针）、`step_profile.py`（从服务日志生成每步计时表）、`keep_render_check.py`（thinking 开运行的 keep 渲染检查）、`token_gather_bench.py`、`docs_set.py`（测试读取的文档集合）和 `docs_check.py`（文档的链接和中英一致性检查）。

仓库里没有结果文件或 trajectory；它带的任务列表（在客户端和 `harnesses/` 中）列出的都是公开 benchmark 的任务。运行归档在分支 `data-swe32` 上。内部的 `remote_ra_*` 模块名属于实现细节，不是公开接口。

发布仓库是 GitHub 上的 `CFXM/agentscope`，和本仓库一样是私有的：它导出本 runtime 的某一个 commit，不带运行历史，内部的 `remote_ra_*` 名称改成了 `spp_*`。

MemoryAgentBench 客户端按本项目的运行方式收录了 Shanbin Sun 的 read_next harness（agent 循环及其提示词，以及 MemAgent32 的 case 选择和切块）（[docs/benchmarks/memoryagentbench.zh-CN.md](docs/benchmarks/memoryagentbench.zh-CN.md) 的 Harness 一节）。它取自 MemoryAgentBench 和 mini-swe-agent 的字符串的 MIT 声明见 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)。

## 如何添加一个选页策略

每个选页策略只在 `source-overlay/policy_registry.py` 中声明一次：一个 `PagePolicy` 类加一次注册；启动器、启动配置、selection contract、receipt 和 `runtime.jsonl` 都从这里读取它的名字和属性。新的填充策略就是 `source-overlay/page_policies/` 中一个给 Selector V2 留下的候选页排序的类，加上 `PLUGINS` 中的一项；然后重新生成发布固定值并重新安装。query 插件再加一个钩子，在余弦填充之前保护、排除或成组页（如 `ledger_query` 和两个标记策略）。完整的示例是 `tests/policy_example.py`；策略能看到的 context 见 [docs/policy-registry.zh-CN.md](docs/policy-registry.zh-CN.md#23-填充路径插件) 第 2.3 节，注册 API 和测试辅助 `policy_registry.registered(...)` 见第 2.4 节，query 钩子见第 8 节。

## 文档

阅读顺序：先跑起来，再看策略怎样插进来，最后看内部实现。编号章节沿用 integrate-1107 之前在 README 中的编号；每个编号只对应一个页面中的一节，代码中写作 `docs/reference.md 4.9`。

| 页面 | 内容 |
| --- | --- |
| [docs/status.zh-CN.md](docs/status.zh-CN.md) | 唯一的状态页：`main` 及其稳定提交、当前配置、按模型、benchmark 和策略列出的 GPU 作业（判定、是否审计、日期；文档提到的 33xxxxx 及之后的每个作业）和尚未运行的内容、各分支 |
| [docs/known-limits.zh-CN.md](docs/known-limits.zh-CN.md) | 哪些还不能用、哪些要小心用：每项限制、影响谁、规避办法、出处和何时解除 |
| [docs/install.zh-CN.md](docs/install.zh-CN.md) | 1 支持边界；2 安装与卸载（2.1 到 2.5：基础环境、模型、测试、安装、卸载）；3 运行环境；9 发布完整性与声明 |
| [docs/running.zh-CN.md](docs/running.zh-CN.md) | 4.1 到 4.5：单个服务、已验收运行的四个策略、多张 GPU、多个 session、Slurm 作业；6：harness 接口（XML、kv-session 和 chat-text 协议、错误处理、从其他主机访问、关闭 session、开启 thinking） |
| [docs/SWEBENCH_BENCHMARK.zh-CN.md](docs/SWEBENCH_BENCHMARK.zh-CN.md)、[docs/TERMINALBENCH_BENCHMARK.zh-CN.md](docs/TERMINALBENCH_BENCHMARK.zh-CN.md)、[docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md) | 各 benchmark 的正式配置（A 部分，带日期的记录；当前配置见 docs/status.zh-CN.md）和流程（B 部分） |
| [docs/benchmarks/](docs/benchmarks/) | 各客户端的页面，与发布仓库相同：[SWE-bench Verified](docs/benchmarks/swe-bench.zh-CN.md)、[Terminal-Bench 2.0](docs/benchmarks/terminal-bench.zh-CN.md)（含本仓库对 Gemma 262,144 token 的修改）、[MemoryAgentBench](docs/benchmarks/memoryagentbench.zh-CN.md)；第 7 节：[SWE-bench pilot](docs/benchmarks/swe-bench-mini-swe-agent.zh-CN.md) |
| [docs/design.zh-CN.md](docs/design.zh-CN.md) | 第 5 节：策略目录与选页；第 8 节：每个服务写出的证据 |
| [docs/policy-registry.zh-CN.md](docs/policy-registry.zh-CN.md) | 写一个策略：注册表、属性、填充路径、等价性证据、token 级选择（第 6 节）、query 钩子与 `ledger_query`（第 8 节）、标记页（第 9 节） |
| [docs/reference.zh-CN.md](docs/reference.zh-CN.md) | 4.6 到 4.15，服务选项：解耦 KV 布局、CUDA graph、上下文上限、admission 优化、解码快速路径、纯文本、释放空闲 Page、Gemma 4 31B IT profile、默认开启的速度选项；`python -B serve.py --help` 列出所有参数 |
| [docs/release-gates.zh-CN.md](docs/release-gates.zh-CN.md) | 分支与 `main`、一个改动需要的 GPU 测试、一个数字必须通过的检查、边界冒烟、提交前的检查、各项指标的定义 |
| [docs/history.zh-CN.md](docs/history.zh-CN.md) | 每个版本，由新到旧，只追加；[docs/history/](docs/history/) 放冻结的记录，不再更新：[截至 `cf2b28d` 的验证记录](docs/history/validation-cf2b28d.zh-CN.md)、从 [running.md](docs/history/running-gpu-notes.zh-CN.md) 和 [reference.md](docs/history/reference-gpu-notes.zh-CN.md) 移出的作业说明、[SWEBENCH_BENCHMARK](docs/history/swebench-validation.zh-CN.md)、[TERMINALBENCH_BENCHMARK](docs/history/terminalbench-validation.zh-CN.md)（其中各次 dry run 的记述）和 [MEMORYAGENTBENCH_BENCHMARK](docs/history/memoryagentbench-validation.zh-CN.md) 原来的验证状态一节、[SWE-bench pilot 的 agent 主机、任务与评测](docs/history/swe-bench-mini-pilots.zh-CN.md)（原来的 7.2、7.3 和 7.5 节），以及[一份未审阅的 thinking 开分析](docs/history/thinking-on-analysis.zh-CN.md) |
| `docs/<topic>/` | 设计说明，各自带日期：[async-sched](docs/async-sched/)、[exp-decouple-pages](docs/exp-decouple-pages/)、[goal-overlap](docs/goal-overlap/)、[rewriter-vllm](docs/rewriter-vllm/)、[thinking-cut](docs/thinking-cut/)、[thinking-gemma](docs/thinking-gemma/)、[token-gather](docs/token-gather/) |
| [harnesses/README.zh-CN.md](harnesses/README.zh-CN.md) | 四个 GPU 作业 harness 的快照（它们自己的页面是副本，不在这里修改） |

### 文档地图

每个页面都有中文对照（`.zh-CN.md`）；`tools/docs_check.py` 检查两者是否一致、每个链接是否有效。章节引用写明该节所在的页面，见上表。
