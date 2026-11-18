# 用 Mini-SWE-Agent 2.4.2 跑 SWE-bench Verified（pilot）

[English](swe-bench-mini-swe-agent.md) | 简体中文

第 7 节，integrate-1107 之前在 README 中，编号不变。正式测试的配置见 [SWEBENCH_BENCHMARK.zh-CN.md](../SWEBENCH_BENCHMARK.zh-CN.md)，发布仓库的页面见 [swe-bench.zh-CN.md](swe-bench.zh-CN.md)。各文档页面的节号互不重复；[文档地图](../../README.zh-CN.md#文档地图)列出了每一节所在的页面。

## 7. 用 Mini-SWE-Agent 2.4.2 跑 SWE-bench Verified

SWE-bench Verified 正式测试的配置（前 32 个 instance，16 个 agent 并发、按列表顺序补入，`full`、`observation_query`、`goal_query` 和 `random_fill` 四个策略，以及一个 stock vLLM 0.24.0 对照）和 baseline 与之对齐的检查表见 [docs/SWEBENCH_BENCHMARK.zh-CN.md](../SWEBENCH_BENCHMARK.zh-CN.md)。

本节介绍多 session SWE-bench pilot 的设置：3 个 Astropy instance × `full`/`goal_query`/`observation_query`、每个服务 3 个 session（作业 3299922），以及 10 个 instance × 同样 3 个策略、每个服务 5 个 session（作业 3300434），runtime 为 `b1e6988`，784 token 的 Page。之后同样 10 个 instance 又以 16 token 的 Page × `full`/`observation_query`/`goal_query`/`random_fill` 跑了一遍，每个策略一个作业（作业 3313802、3314010、3314319、3314623，runtime `780fe54`；结果见 4.7 节），agent 主机不变。每个服务都通过了服务端检查。每个 instance 只跑一次，不构成 benchmark 证据。这两次 pilot 没有跑 `random50`。

### 7.1 设置

| 设置 | 值 |
| --- | --- |
| Agent | Mini-SWE-Agent 2.4.2（litellm 1.103.1 和 datasets 5.0.1：未重新核验，作业的证据只记录了 Mini-SWE-Agent 的版本） |
| Runner | PoC 的 `swebench_runner.py`，参数 `--subset verified --split test --workers <session 数> --filter '^(<id>\|<id>…)$' --output <dir>` |
| 配置叠加 | `-c swebench_xml.yaml`（Mini-SWE-Agent 内置）`-c swebench_prompt_override.yaml`（PoC 的 prompt）`-c model_override.yaml` |
| 模型 | `examples/swebench/model_override.yaml`：`litellm_textbased`、`openai/selective-prefix`、`api_base` `http://127.0.0.1:<port>/v1`、占位 key、temperature 0.0、`max_tokens` 1536（`run_agents.sh` 的 `MAX_TOKENS`；4.9 节）、`cost_tracking: ignore_errors` |
| 环境 | `docker`、`cwd` `/testbed`、超时 120 秒、`bash -c`、`PAGER`/`MANPAGER` 为 `cat`、`LESS` 为 `-R`、`PIP_PROGRESS_BAR` off、`TQDM_DISABLE` 1 |
| 限制 | `step_limit` 90（`run_agents.sh` 的 `STEP_LIMIT`；4.9 节）、`cost_limit` 0 |
| 进程环境变量 | 不设置 `MSWEA_CTX_PATCH`、`AGENTDIET_MODE=off`、`HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`、`MSWEA_GLOBAL_CONFIG_DIR` 指向一个空目录 |
| 服务 | 每个策略、每个组一个全新的服务，`--task tasks-g<k>.json`、`--kv-cache-gib 24`，`goal_query` 带 0.6B rewriter；worker 数 = session 数 = 组的大小。16 token Page 的 pilot 给每个服务（包括 `full`）都加了 `--page-tokens 16 --kv-layout decoupled`，`random_fill` 不带 rewriter 选项 |
| 评测 | swebench 5.0.2、`SWE-bench/SWE-bench_Verified`、split `test`、`--timeout 1800`、`--max_workers 2`（pilot 脚本的默认值） |

填入端口后的 `model_override.yaml` 与 10-instance pilot 使用的六个文件逐字节相同。服务端本来就忽略客户端的采样字段（6.1 节）。Mini-SWE-Agent 会写出 `<output>/<instance>/<instance>.traj.json` 和 `<output>/preds.json`。

**PoC 文件。** `examples/swebench/upstream/` 里是 dram-backed-kv PoC 仓库的 `swebench_runner.py` 和 `swebench_prompt_override.yaml`（commit `a70701e` 的 `scripts/`，LF 的 Git blob 字节；prompt override 与 cohort-2 分支 `0ab0078` 上的副本相同），与 pilot 实际运行的副本逐字节一致（见 `upstream/SOURCE.txt`、`upstream/SHA256SUMS`）。`run_agents.sh` 默认使用它们，SHA-256 与下列固定值不同的文件会被拒绝：

| 文件 | SHA-256 |
| --- | --- |
| `swebench_runner.py` | `34ca6d9288c72830e61bd00ad8b15790cab4c29515db24b8df30775cfc0880a7` |
| `swebench_prompt_override.yaml` | `1b6a48666ad528775bd11225fa4d5c002721c25c0cdbc05b99cc15b2de31336a` |

这两个文件有两点会影响结果，和其他设置比较时要注意：
- runner 替换了 Mini-SWE-Agent 的 agent 类。达到步数上限时，它会提交 agent 最后打印过的、至少 3 行的 `git diff`（退出状态 `LimitsExceeded_InformalSubmission`）；连续 10 次相同命令时终止该实例（`LoopDetected`，不论是否已有这样的补丁；已有补丁时仍会提交它，状态为 `LoopDetected_InformalSubmission`）。它还会尝试导入一个可选的 `agentdiet_adapter`，本仓库不包含；`AGENTDIET_MODE=off`（pilot 的设置）时不会用到它。
- prompt 是为 Astropy 写的（"The testbed is an Astropy repository"，并带有 Astropy 专用的测试建议）。pilot 只跑了 Astropy 实例；其他仓库需要改写 prompt，这**不属于已验证的运行**。

### 7.2 Agent 主机

现在的 agent 主机（装有按固定 digest 拉取的 instance 镜像的 Docker、所需工具、agent 环境以及固定 commit 的评测程序）见 swe-bench 页面的 [Agent 主机的准备](swe-bench.zh-CN.md#agent-主机的准备)。pilot 在 2026-09/10 的 agent 主机（未固定的镜像、PyPI 上的评测程序、没有记录的命令）现在的脚本会部分拒绝，它冻结在 [docs/history/swe-bench-mini-pilots.zh-CN.md](../history/swe-bench-mini-pilots.zh-CN.md#出自-72-节agent-主机)。

### 7.3 任务

每个服务的 `--task` 列表必须恰好包含其 agent 要跑的 instance，`{instance_id, problem_statement}` 原样取自 SWE-bench Verified：请求按任务消息里的 problem statement 路由（4.4 和 6.3 节）。`examples/swebench/build_tasks.py` 从固定 revision 的数据集生成这些文件：session 池服务用 `--pool`（swe-bench 页面的 [任务](swe-bench.zh-CN.md#任务)），分组用 `--max-sessions`（[用 Slurm 脚本跑一个小组](swe-bench.zh-CN.md#用-slurm-脚本跑一个小组)）。pilot 的任务文件冻结在 [docs/history/swe-bench-mini-pilots.zh-CN.md](../history/swe-bench-mini-pilots.zh-CN.md#出自-73-节任务)。

### 7.4 运行一个组

每个策略、每个组：用该组列表启动一个全新的服务，打开隧道，然后在 agent 主机上运行一个 runner（`run_agents.sh <socket> <port> <新的输出目录> <该组的 instance id>`）。使用 Slurm 脚本时（这是 pilot 提交命令的当前写法：pilot 申请的是 `--mem=200g`，它们的 `run_agents.sh` 也不需要 `HF_DATASETS_CACHE`；该缓存由 swe-bench 页面的 [任务](swe-bench.zh-CN.md#任务) 一节准备）：`ACCOUNT`、`PARTITION` 和 `MAIL` 与 [Slurm 作业](../running.zh-CN.md#45-gpu-作业slurm)相同，`REPO`、`VENV`、`RELEASE` 和 `MODELS` 是 README 快速上手的占位变量（[从零到一个冒烟请求](../../README.zh-CN.md#从零到一个冒烟请求)）。`RUN_ROOT` 必须在 `sbatch` 之前就存在：

```sh
export REPO VENV RELEASE
export TASKS=/abs/path/to/tasks/tasks-g0.json POLICY=goal_query KV_GIB=24 REWRITER="$MODELS/Qwen3-0.6B-c1899de"
export PAGE_TOKENS=16 KV_LAYOUT=decoupled
export RUN_ROOT=/abs/short/jobs MODE=swebench AGENT_SECONDS=5400
mkdir -p "$RUN_ROOT"
export AGENT_HOST=user@agent-host TUNNEL_KEY=/abs/path/to/tunnel_key KNOWN_HOSTS=/abs/path/to/agent_known_hosts
export AGENT_SOCKET=/abs/path/on/agent/sock/g0-goal_query.sock AGENT_PORT=20000
export AGENT_RUN=/abs/path/on/agent/examples/swebench/run_agents.sh AGENT_OUT=/abs/path/on/agent/runs/goal_query-g0
export AGENT_ENV="MINI_PYTHON=/abs/path/on/agent/mini-env/bin/python MSWEA_CONFIG_DIR=/abs/path/on/agent/mswea-config
  HF_DATASETS_CACHE=/abs/path/on/agent/swebench-datasets"
sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus=1 \
  --cpus-per-task=16 --mem=113560M --time=03:45:00 --job-name=selective-prefix \
  --output="$RUN_ROOT/slurm-%j.out" --export=ALL --mail-type=BEGIN,END,FAIL --mail-user="$MAIL" \
  "$REPO/examples/slurm/serve_gh200.sh"
```

说明：
- 每个策略一个服务，每个组都用全新的服务、全新的输出目录和 socket。pilot 让每个策略跑同样的分组，在一张 GPU 上一个服务接一个服务地运行。
- `PAGE_TOKENS=16 KV_LAYOUT=decoupled` 是 `goal_query`、`observation_query` 和 `random_fill` 的推荐设置（4.6 节）；两者都不设就是默认的 784 token 布局。跑 `random_fill` 时设 `POLICY=random_fill`；脚本只给 `goal_query` 传 rewriter。
- 时间：作业 3300434 中，一组 5 个 agent 用时 1,255–2,120 秒，它的 6 个服务（3 个策略 × 2 组）中最后一个在作业开始后 10,851 秒结束，在 10:45:00 的申请时间之内。pilot 对每次 agent 运行设了时间上限：3,300 秒（作业 3299922）或 5,400 秒（作业 3300434）。16 token Page 的 pilot（每个作业一个策略、2 组，agent 上限 3,000 秒）中，一组用时 1,215–2,045 秒，每个作业用了 02:00:00 申请时间中的 49–68 分钟。
- SSH 断开或超时不会停止远端的 runner。pilot 在这种情况下会结束该 runner，并只删除它启动的容器（日志里 `minisweagent-<8 位十六进制>` 的名字）。
- 作业 3299922、3300434 和 3314319 中有些 `goal_query` 轮次回退到了确定性的 Goal 字段（finding `GOAL_FALLBACK`；第 5 节）；服务仍然通过。
- Session 池（4.4 节）：设 `TASKS=.../tasks-pool.json` 和 `MAX_RUNNING=<K>`。Slurm 脚本会给 `serve.py` 传 `--max-running-sessions K`，给 `run_agents.sh` 传 `MAX_RUNNING`；后者随即运行一个 K 个 agent 的池，按列表顺序每个 instance 一个 runner，每个 runner 结束时关闭该 instance 的 session（其最后一个请求仍在清理时会重试），再启动下一个。它写出 `runs/<instance>/`、`logs/<instance>.log`、`closes/<instance>.json`、`timeline.jsonl`（每个 instance 的开始和结束时间、runner 退出码和关闭状态：即从第一个开始到最后一个结束的墙钟时间）以及供 `evaluate.sh` 使用的合并后的 `preds.json`。`run_agents.sh` 本身还没有在已验证的作业中以这种模式运行过；同样的池形式（K 个 agent 按列表顺序补入，每个 instance 关闭一次）经 benchmark harness 在 SWE-bench Verified 的 benchmark 作业中运行过，例如作业 3461703（`SWE32_PASS_VERIFIED`，32 个任务，最多 16 个同时运行）。
- 步数上限和回复上限（4.9 节）：`STEP_LIMIT=<1-256>` 和 `MAX_TOKENS=<1536|4096>` 设定 `run_agents.sh` 所写 model override（`render_override.py`）中 Mini-SWE-Agent 的 `step_limit` 和 `max_tokens`；Slurm 脚本把两者传给 agent 主机，并把 `MAX_TOKENS`（以及 `MAX_MODEL_LEN`、`DRAM_ARENA_GIB`）传给 `serve.py`。不设置时，override 与 pilot 用的文件逐字节相同，`serve.py` 也不会多出任何参数。步数上限 120 加 4096 token 的回复上限在 SWE-bench Verified 的 benchmark 作业中运行过（例如作业 3461703），由 benchmark harness 设定，而不是 `run_agents.sh`。

### 7.5 评测

agent 结束后，在 agent 主机上按策略评测：`examples/swebench/evaluate.sh`，把该策略的所有组输出作为最后几个参数，做法同 swe-bench 页面的 [评测](swe-bench.zh-CN.md#评测)（固定版本 test split 的本地 Parquet 文件，commit `02e7a74` 的评测程序）。pilot 的评测（PyPI 上的 `swebench==5.0.2`，评测 `SWE-bench/SWE-bench_Verified`）冻结在 [docs/history/swe-bench-mini-pilots.zh-CN.md](../history/swe-bench-mini-pilots.zh-CN.md#出自-75-节评测)。trajectory、predictions 和日志都包含 prompt 和工具输出，不要放进公开发布。
