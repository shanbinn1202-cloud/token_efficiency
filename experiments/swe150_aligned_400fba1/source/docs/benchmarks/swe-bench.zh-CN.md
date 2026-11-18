# 用 Mini-SWE-Agent 2.4.2 跑 SWE-bench Verified

[English](swe-bench.md) | 简体中文

> 在 integrate-1107 从发布仓库同步（release-repo `5b47394` 的 benchmark 页面）：`examples/swebench/` 中的客户端和本页都来自发布仓库，只在发布仓库中有效的链接改指本仓库的 README 和 benchmark 文档。各次运行及其作业 id 见 [docs/status.zh-CN.md](../status.zh-CN.md)。Qwen 有两套配置。下面的“服务命令”是发布仓库的设置 `--kv-cache-gib 25 --max-model-len 262144`（作业 3461661 和 3461703，[docs/reference.zh-CN.md 4.9 节](../reference.zh-CN.md#49-上下文上限与回复上限)）：自 runtime `cf2b28d` 起，Qwen 的 SWE-bench 运行都用这个设置（作业 3461703：32 个中 `goal_query` 解决 19 个、`observation_query` 解决 20 个；同一设置下的 stock 对照，作业 3507294：21 个）。SWEBENCH_BENCHMARK.zh-CN.md 的 Part A 是此前作为正式记录的配置，32 GiB、131,072 token（`5b15b2f` 和 `120bc3c` 上的运行），有它自己的服务命令。这里的 Gemma 服务带 `--max-num-batched-tokens 4096`（作业 3462014，[docs/history.zh-CN.md](../history.zh-CN.md#历史)，integrate-1026）。2026-11-08 的文档修正改动了本页中不清楚或没有出处的地方：“GPU 上的运行”一段、客户端的超时和重试设置（[Agent](#agent)）、stock 生命周期，以及运行证据中没有记录的版本。

## 范围

- SWE-bench Verified（`princeton-nlp/SWE-Bench_Verified`，`test` split，revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`）：`examples/swebench/verified-32.txt` 中的 32 个 instance，即该 split 按数据集顺序的前 32 个，按这个顺序运行（[任务](#任务)）；注册 32 个任务，最多 16 个 agent 同时运行；一个 agent 结束后，按列表顺序启动下一个 instance。
- benchmark 的输入都是固定的：有序的 instance 列表、数据集的 revision（任务用 `build_tasks.py` 中的，评测用 `evaluate.sh` 中的）、32 个 problem statement（SHA-256，由 `build_tasks.py` 检查），以及按 digest 固定的每个 instance 的镜像（`verified-32.txt`，由 `run_agents.sh` 和 `evaluate.sh` 在任何步骤开始之前检查）。运行期间没有任何一步从 registry 解析数据集的 HEAD 或镜像的 `:latest`。
- 每个策略一个全新的服务（`full`、`observation_query`、`goal_query`、`random_fill`），另有一个 stock vLLM 0.24.0 对照，各自一个 GPU 作业；一个服务把所有 instance 各跑一次，中间不中断。
- Agent 软件：Mini-SWE-Agent `2.4.2`，使用 XML 协议（`--protocol mswea-xml`，默认值；[docs/running.zh-CN.md 6.1 节](../running.zh-CN.md#61-mini-swe-agent-xml-协议)），加 litellm `1.103.1` 和 datasets；评测程序：commit `02e7a74ffd0b707aab73d203fe87bdc7c76afc8e` 的 SWE-bench（版本 5.0.2），加 datasets。运行的证据没有记录 datasets 的版本；下面的环境安装 `5.0.1`，即发布仓库固定的版本。agent 和评测在一台 x86_64 Docker 主机（agent 主机）上运行：instance 镜像是 x86_64 的（`sweb.eval.x86_64.*`）。服务运行在 GPU 节点上（测试用的是一台 aarch64 GH200），agent 通过[运行一个策略](#运行一个策略)中的反向 SSH 隧道访问它。不支持在一台 aarch64 机器上同时运行两者：本发布版本没有供 agent 和评测程序使用的 ARM64 镜像或模拟环境设置。

**GPU 运行：** 这些客户端的运行，连同各自的 runtime 提交、模型、策略、thinking 设置、KV 大小、上下文和判定，见 [docs/status.zh-CN.md](../status.zh-CN.md)，其中有 33xxxxx 及之后的每个作业；第一次多 session pilot，即作业 3299922，见[历史](../history.zh-CN.md#历史)。发布仓库的 `5b47394` 从 `5466fc2` 导出，即 `cf2b28d` 加上 serve.py 的速度默认值和两项测试修复。下面的命令关闭 thinking，是正式记录的配置；当前的运行（thinking 开、`--async-scheduling`）也在那里说明。

## 服务参数

四个 Qwen 策略共用。Gemma 服务（[Gemma 4 31B IT](#gemma-4-31b-it)）的不同之处：`--max-model-len 131072`、`--dram-arena-gib 72`、`--release-idle-pages`、`--max-num-batched-tokens 4096`，不加 `--kv-layout`，`full` 用 `--kv-cache-gib 32`、其他策略用 25，`goal_query` 另加 `--goal-rewriter-batch-tokens 15360`：

| 参数 | 取值 |
| --- | --- |
| `--task` | 32 个 instance 的池文件，由 `examples/swebench/build_tasks.py --pool` 生成 |
| `--max-running-sessions` | `16` |
| `--kv-cache-gib` | `25` |
| `--page-tokens` / `--kv-layout` | `16` / `decoupled` |
| `--max-model-len` / `--max-tokens` | `262144` / `4096` |
| `--dram-arena-gib` | `70` |
| `--connector-pool-gib` | `1`（速度默认值） |
| `--cuda-graphs` | `full`（速度默认值） |
| `--decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io` | 开启（速度默认值） |
| `--language-model-only` | 开启（速度默认值） |

各策略：

| 策略 | 选页参数 | Admission 参数 |
| --- | --- | --- |
| `full` | 无 | 无 |
| `observation_query` | `--active-token-budget 8192` | 无 |
| `random_fill` | `--active-token-budget 8192`（默认的 `--fill-seed`） | 无 |
| `goal_query` | `--active-token-budget 8192 --goal-rewriter-model <Qwen3-0.6B> --goal-rewriter-batch 4` | `--admission-overlap 16` |

stock 对照就是从同一个环境启动的普通 `vllm.entrypoints.openai.api_server`，不带本 runtime 的任何代码。它的生命周期（stock 生命周期）：在后台启动 `"${STOCK_CMD[@]}"`；`GET /health` 返回 200 且 `GET /v1/models` 列出 `selective-prefix` 时即为 READY（[SWEBENCH_BENCHMARK.zh-CN.md 的 A.4](../SWEBENCH_BENCHMARK.zh-CN.md#a4-stock-vllm-对照)）；向它的 PID 发送 `SIGTERM` 停止它，并等待它退出（[SWEBENCH_BENCHMARK.zh-CN.md 的 B.5](../SWEBENCH_BENCHMARK.zh-CN.md#b5-slurm-作业脚本) 在一个作业脚本里给出了整个流程）。`serve.py` 的 dry run、`--start`、READY 文件和清理回执的检查只适用于 selective release。跑过的 stock 服务都在[运行环境](../install.zh-CN.md#3-运行环境)中启动，另加 `serve.py` 为本 runtime 设置的两个变量和 `VLLM_NO_USAGE_STATS=1`；`STOCK_CMD` 就是生命周期所启动的命令。与更早构建的服务对照时（32 GiB、131,072 token）：

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
STOCK_CMD=("$VENV/bin/python" -I -B -m vllm.entrypoints.openai.api_server
           --model "$MODELS/qwen3.6-27b" --served-model-name selective-prefix --host 127.0.0.1 --port 8000
           --distributed-executor-backend uni --tensor-parallel-size 1
           --max-num-seqs 16 --max-model-len 131072 --max-num-batched-tokens 16384
           --kv-cache-memory-bytes 34359738368 --seed 42
           --enable-prefix-caching --prefix-caching-hash-algo sha256 --language-model-only
           --mamba-cache-mode align --gdn-prefill-backend triton)
```

此前在本发布版本的 Qwen 设置（262,144 token、25 GiB）下的 GPU 运行中，用的是同一条命令，改为 `--max-model-len 262144` 和 `--kv-cache-memory-bytes 26843545600`（vLLM 把这些值记录在它的非默认参数中；`--tensor-parallel-size 1` 和 `--prefix-caching-hash-algo sha256` 是它的默认值）：

```sh
STOCK_CMD=("$VENV/bin/python" -I -B -m vllm.entrypoints.openai.api_server
           --model "$MODELS/qwen3.6-27b" --served-model-name selective-prefix --host 127.0.0.1 --port 8000
           --distributed-executor-backend uni --tensor-parallel-size 1
           --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 16384
           --kv-cache-memory-bytes 26843545600 --seed 42
           --enable-prefix-caching --prefix-caching-hash-algo sha256 --language-model-only
           --mamba-cache-mode align --gdn-prefill-backend triton)
```

本页不包含 Gemma 4 stock 对照服务的配方，也不涵盖这条路径。

stock vLLM 从请求中读取采样设置。stock 运行的 agent 发送了 `temperature` 0.0、`max_tokens` 4096、`stop` `["</mswea_bash_command>"]` 加 `include_stop_str_in_output: true`，以及 `chat_template_kwargs: {"enable_thinking": false}`；`run_agents.sh` 在 `STOCK=1` 时发送后两项（[Agent](#agent)）。stock vLLM 没有 `/v1/sessions/close` 路由：agent 池的关闭请求会得到 HTTP 404，`run_agents.sh` 只在 `STOCK=1` 时把它当作一个已结束 agent 的关闭（对本 runtime 的服务，关闭必须返回 200）。

## Agent

`examples/swebench/` 下的衔接脚本：

- `build_tasks.py`：从数据集生成任务文件（session 池用 `--pool`）。
- `model_override.yaml`、`render_override.py`：Mini-SWE-Agent 的模型配置（端口、回复上限、步数上限）。
- `run_agents.sh`：在 agent 主机上运行 agent，带 loopback 中继 `relay.py`。设了 `MAX_RUNNING`（benchmark 的 session 池）时，由这么多个 agent 组成的池按列表顺序处理各个 instance，每个 instance 一个 runner；不设时，只有一个 runner，每个 instance 一个 worker 线程。它在自己的进程组中运行，并把进程组 id 写到输出目录中的 `agents.pgid`（[停止 agent](#停止-agent)）。
- `evaluate.sh`：在固定的评测数据集上对 predictions 做 SWE-bench 评测（[评测](#评测)）。
- `verified-32.txt` 和 `images.py`：按运行顺序列出的 32 个 instance，每个的镜像按 digest 固定（`<instance_id> <image>@sha256:<digest>`），以及 `run_agents.sh` 和 `evaluate.sh` 最先运行的检查：每个 instance 的本地 `<image>:latest` 必须是它固定的镜像，否则脚本停止，并打印放置该镜像的 `docker pull` 和 `docker tag` 命令。`IMAGE_PINS` 指定另一个固定值文件，用于其他 instance：`verified-150.txt` 以同样方式固定按数据集顺序的前 150 个 instance（README 的 24 个 session 配置）；它的前 32 行就是 `verified-32.txt`，其余 digest 是 agent 主机上 2026-11-17 的镜像。

`run_agents.sh` 默认使用 DRAM-KV 原型的 SWE-bench runner 和 prompt（`examples/swebench/upstream/`，已包含在仓库中；其中的 `SHA256SUMS` 固定它们）。它的设置：`MAX_RUNNING`（session 池）、`MAX_TOKENS`（1536 或 4096，即服务的 `--max-tokens`）、`STEP_LIMIT`（1-256）和 `STOCK`：服务是 stock vLLM 对照时为 `1`，这时 model override 还会发送停止标记（保留在输出中）并关闭 thinking，即 stock 对照运行时所用的那两行；本 runtime 的服务用 `0`（默认值），这时 model override 不变。其他任何值都会在任何步骤开始之前被拒绝。池中的一个 agent 成功，是指它的 runner 以 0 退出且它的 session 关闭返回 HTTP 200，或在 `STOCK=1` 时返回 HTTP 404（stock vLLM 没有关闭路由）。`HF_DATASETS_CACHE`（必需）是[任务](#任务)一节中 agent 的 datasets 缓存。`REPO_ENV=testbed`（可选；默认不设）让 agent 的命令在仓库的环境激活后运行（model override 的 interpreter 为 `["bash", "-lc"]`），并使用脚本旁的 `swebench_prompt_testbed.yaml`，即对 Django 镜像给出 Django 测试命令的 PoC prompt（在 `run_agents.sh` 中固定；[docs/known-limits.zh-CN.md](../known-limits.zh-CN.md) 5.6）；使用 `serve_gh200.sh` 时在作业的环境中设置它。带它的结果不能与不带它的运行比较。

**客户端超时与重试。** 模板中没有客户端超时或重试的行，`run_agents.sh` 也不添加。经过验证的运行（例如作业 3428631、3429193、3461703 和 3507294）在 model override 中有 `timeout: 3600`、`max_retries: 0` 和 `num_retries: 0`，在 agent 的环境中有 `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1` 和 `REQUEST_TIMEOUT=3600`：在本 runtime 上，客户端重试一个仍在处理中的请求、超时或非 2xx 状态都会结束该服务的这次运行（[SWEBENCH_BENCHMARK.zh-CN.md 的 A.7](../SWEBENCH_BENCHMARK.zh-CN.md#a7-限制)）。要像它们那样运行，把这三行加到 `examples/swebench` 的一份副本中（即下文的 `/abs/path/on/agent/swebench-client`：[SWEBENCH_BENCHMARK.zh-CN.md 的 B.2](../SWEBENCH_BENCHMARK.zh-CN.md#b2-任务列表与-agent-文件)），使用这份副本的 `run_agents.sh`，并传入这两个变量，本页的命令就是这样做的。这时带 `STOCK=1` 渲染出的文件包含 stock 对照的五行。

stock 对照在 agent 主机上运行，`AGENT_SOCKET` 和 `AGENT_OUT` 与[运行一个策略](#运行一个策略)中相同，`IDS` 与[任务](#任务)中相同：

```sh
STOCK=1 MAX_RUNNING=16 MAX_TOKENS=4096 STEP_LIMIT=120 MINI_PYTHON=/abs/path/on/agent/mini-env/bin/python \
  MSWEA_CONFIG_DIR=/abs/path/on/agent/mswea-config HF_DATASETS_CACHE=/abs/path/on/agent/swebench-datasets \
  MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1 REQUEST_TIMEOUT=3600 \
  /abs/path/on/agent/swebench-client/run_agents.sh "$AGENT_SOCKET" 20000 "$AGENT_OUT" $IDS
```

### 仓库的环境（REPO_ENV=testbed）

`REPO_ENV=testbed`（[docs/known-limits.zh-CN.md](../known-limits.zh-CN.md) 5.6）让 agent 的命令在仓库的 `testbed` conda 环境激活后运行，并在 prompt 中给出每个仓库的测试命令。它适用于任何按 [Agent 主机的准备](#agent-主机的准备)准备好的 agent 主机（Docker、`MINI_PYTHON` 中的 Mini-SWE-Agent 2.4.2、datasets 缓存、按 digest 固定的本次运行的 instance 镜像：前 150 个 instance 用 `IMAGE_PINS=.../verified-150.txt`）。在 `examples/swebench` 的一份已有副本（例如带本节客户端行的那份）中使用它：

1. 把本仓库 `examples/swebench/` 中的三个文件复制到副本中，替换副本自己的：`run_agents.sh`、`render_override.py` 和新的 `swebench_prompt_testbed.yaml`。保留副本的 `model_override.yaml`（它必须仍然恰好含有一行 `  interpreter: ["bash", "-c"]`，否则渲染器会拒绝它）、`upstream/` 和镜像固定文件。
2. 在 `run_agents.sh` 的环境中加上 `REPO_ENV=testbed`，与 `MAX_RUNNING`、`MAX_TOKENS`、`STEP_LIMIT` 和 `STOCK` 并列（使用 `serve_gh200.sh` 或作业脚本时：放在作业的环境中或 `AGENT_ENV` 中）。其他都不变：不设它时这三个文件的运行与之前完全相同。
3. 运行中的变化：`AGENT_OUT` 中渲染出的 `model_override.yaml` 为 `interpreter: ["bash", "-lc"]`，每个 runner 得到 `-c <副本>/swebench_prompt_testbed.yaml` 以代替 PoC 的 prompt。prompt 的测试一节取决于镜像：Django 镜像得到 Django 一节（`./tests/runtests.py --settings=test_sqlite --parallel 1 <label>`），其他每个镜像得到逐字节不变的 PoC 的 Astropy 文字。`run_agents.sh` 在任何步骤开始之前检查 prompt 的 SHA-256（`1e4ec193b34a92a3b56e33d05ebb9fc70591a194f8d888d6a7392aae4eefaf32`），并拒绝 `REPO_ENV` 的其他任何值。

真正运行之前在 agent 主机上的检查。文件：上面的 prompt SHA-256，以及渲染出的 interpreter `  interpreter: ["bash", "-lc"]`：

```sh
cd /abs/path/on/agent/swebench-client
sha256sum swebench_prompt_testbed.yaml
python3 -I -B render_override.py model_override.yaml 20000 4096 120 0 testbed | grep interpreter
```

Mini-SWE-Agent 为一个 Django 镜像渲染出的 prompt（用镜像名执行 Jinja2；打印 Django 一节 `cd /testbed && ./tests/runtests.py ...`；换成 Astropy 镜像则是 PoC 的 Astropy 一节）：

```sh
"$MINI_PYTHON" -c 'import sys, yaml, jinja2
t = yaml.safe_load(open(sys.argv[1]))["agent"]["instance_template"]
print(jinja2.Template(t, undefined=jinja2.StrictUndefined).render(task="T", step_limit=120, image=sys.argv[2]))' \
  swebench_prompt_testbed.yaml docker.io/swebench/sweb.eval.x86_64.django_1776_django-11099:latest \
  | grep -A 8 '## Running Tests'
```

每个仓库一个容器，按 Mini-SWE-Agent 的方式启动（`docker run -d -w /testbed`，再 `docker exec -w /testbed ... bash -lc`）：用 `bash -lc` 时，`which python` 打印 `/opt/miniconda3/envs/testbed/bin/python`，`pwd` 打印 `/testbed`，Django 测试打印 `OK`，Astropy 测试打印 `passed`；用 `bash -c` 时，`which python` 打印 `/opt/miniconda3/bin/python`（base 解释器，仓库在其中无法导入）：

```sh
IMG=docker.io/swebench/sweb.eval.x86_64.django_1776_django-11099:latest
C=$(docker run -d --rm -w /testbed "$IMG" sleep 600)
docker exec -w /testbed "$C" bash -lc 'which python; pwd; ./tests/runtests.py --settings=test_sqlite --parallel 1 utils_tests.test_text 2>&1 | tail -3'
docker exec -w /testbed "$C" bash -c 'which python'
docker rm -f "$C"
IMG=docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest
C=$(docker run -d --rm -w /testbed "$IMG" sleep 600)
docker exec -w /testbed "$C" bash -lc 'which python; python -m pytest astropy/utils/tests/test_misc.py -q -p no:cacheprovider 2>&1 | tail -1'
docker rm -f "$C"
```

运行之后（或其最初几个 instance 之后）：

```sh
grep interpreter "$AGENT_OUT/model_override.yaml"
grep -l 'Running Tests (Django-specific)' "$AGENT_OUT"/runs/django__django-11099/*/*.traj.json
```

第一行打印 `  interpreter: ["bash", "-lc"]`，第二行打印该 Django instance 的轨迹。带 `REPO_ENV=testbed` 的结果不能与不带它的运行比较：对一次比较中的每个配置都使用它。

## Agent 主机的准备

agent 和评测在一台装有 Docker 的 x86_64 Linux 主机（agent 主机）上运行（要求见 [agent 主机](../SWEBENCH_BENCHMARK.zh-CN.md#b1-agent-主机)）。服务运行在 GPU 节点上，agent 通过 GPU 节点向 agent 主机打开的反向 SSH 隧道访问它（[docs/running.zh-CN.md 6.4 节](../running.zh-CN.md#64-从其他主机访问服务)）。agent 主机需要：

- Docker：正在运行，运行脚本的用户可以使用它（不需要交互式的 `sudo`），并有按固定 digest 拉取的 32 个 instance 镜像（见下）。
- 数据集，放在 agent 主机上一个专用的 datasets 缓存中（`HF_DATASETS_CACHE`，一个只留给固定 revision 的新目录：[任务](#任务)）：`run_agents.sh` 要求设置它，并用它离线运行 Mini-SWE-Agent（`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`）；Mini-SWE-Agent 的 `--subset verified --split test` 不指定 revision 加载 `princeton-nlp/SWE-Bench_Verified`，离线读取该缓存里的内容，而[任务](#任务)一节中的 `build_tasks.py` 按固定 revision 填充这个缓存。服务会拒绝不含其冻结任务 problem statement 的任务消息，所以 agent 不可能在另一份 statement 上运行。`evaluate.sh` 使用它自己固定的 revision 的 `SWE-bench/SWE-bench_Verified`（[评测](#评测)）。
- 两个 Python 环境：`$MINI`（Mini-SWE-Agent 2.4.2、litellm、datasets；`run_agents.sh` 会检查版本）和 `$EVAL`（commit `02e7a74` 的 SWE-bench 评测程序，版本 5.0.2，以及 datasets），用 `uv` 创建（见下）。
- agent 主机上的一份本仓库副本：`run_agents.sh` 使用它所在目录中的 `render_override.py`、`model_override.yaml`、`relay.py`、`images.py`、`verified-32.txt` 和 `upstream/`，`evaluate.sh` 使用其中的两个镜像文件。`AGENT_RUN`（[运行一个策略](#运行一个策略)）是加了客户端设置行的那份 `examples/swebench` 副本（[Agent](#agent)）中 `run_agents.sh` 的路径。
- `PATH` 上有 `bash`、`python3`、`curl`、`ss`、`sha256sum`、`setsid`、`docker` 和 `git`（uv 从评测程序的 git URL 安装它）。
- 一个给 `MSWEA_CONFIG_DIR` 用的空目录（Mini-SWE-Agent 的全局配置目录）。
- 隧道：GPU 节点能用一把 key（`TUNNEL_KEY`）和一个固定 agent 主机 host key 的 known-hosts 文件（`KNOWN_HOSTS`）向 agent 主机发起出站 SSH 连接；`AGENT_SOCKET` 是 agent 主机上一个 0700 目录中的新 socket 路径，`AGENT_OUT` 是一个父目录已存在的新目录。

在 Ubuntu 24.04（测试用的 agent 主机）上：从 Ubuntu 软件源安装 Docker、这些工具和 Docker Compose v2（Terminal-Bench 的 Harbor 用 `docker compose` 运行每个任务；SWE-bench 不需要它），让你的用户可以使用 Docker（之后重新登录），并用 uv 官方的安装脚本安装 uv。下面固定的版本是发布仓库的；运行的证据没有记录 agent 主机的 Docker、Compose 或 uv 版本。软件源中已没有某个版本时，去掉 `=<version>`：

```sh
sudo apt-get update
sudo apt-get install -y docker.io=29.1.3-0ubuntu3~24.04.2 curl iproute2 python3 coreutils util-linux git
sudo apt-get install -y docker-compose-v2=2.40.3+ds1-0ubuntu1~24.04.1
sudo usermod -aG docker "$USER"
curl -LsSf https://astral.sh/uv/0.12.5/install.sh | sh
```

重新登录后，不用 `sudo` 运行 `docker version` 必须显示 `Server` 一节，`docker compose version` 也必须在这台主机上打印出 Docker Compose v2 的版本。然后在 agent 主机上、在本仓库目录中，按 digest 拉取 instance 镜像，再按 agent 和评测程序使用的名字打上标签（`docker.io/swebench/sweb.eval.x86_64.<instance>:latest`）：

```sh
while read -r instance image; do
  docker pull "$image"
  docker tag "$image" "${image%@*}:latest"
done < examples/swebench/verified-32.txt
```

两个环境，在 agent 主机上创建（`uv` 是 `~/.local/bin/uv`）：

```sh
MINI=/abs/path/to/mini-env
EVAL=/abs/path/to/eval-env
uv venv --python 3.12 "$MINI"
uv pip install --python "$MINI/bin/python" mini-swe-agent==2.4.2 litellm==1.103.1 datasets==5.0.1
uv venv --python 3.12 "$EVAL"
uv pip install --python "$EVAL/bin/python" \
  "swebench @ git+https://github.com/SWE-bench/SWE-bench@02e7a74ffd0b707aab73d203fe87bdc7c76afc8e" datasets==5.0.1
```

评测程序是 commit `02e7a74ffd0b707aab73d203fe87bdc7c76afc8e` 的 SWE-bench harness（`swebench` 版本 `5.0.2`），也就是评测已验证运行的那一个：那些运行的评测环境以这个 commit 命名。上面的 git URL 安装的是同一个 commit。它的其他依赖由 uv 在安装时解析。不要改为从 PyPI 安装 `swebench==5.0.2`：`evaluate.sh` 以本地 Parquet 文件传入数据集，这个 commit 的 `load_swebench_dataset` 能读取它，而 PyPI 上的版本没有为此检查过（`evaluate.sh` 会拒绝不能读取它的评测程序）。

## 任务

在 agent 主机上、在本仓库目录中运行（`$MINI` 装有 `datasets`；第一次加载需要联网并会填充缓存，所以 `unset` 那一行取消 shell 中可能存在的离线开关）。`HF_DATASETS_CACHE` 是 agent 的 datasets 缓存，一个新目录：`build_tasks.py` 把固定 revision 的数据集加载到其中，`run_agents.sh` 把同一个目录交给 Mini-SWE-Agent，后者离线加载数据集且不指定 revision，所以取的是这个缓存里的内容。这个缓存只留给固定的 SWE-bench revision；在准备和运行之间，不要往里面放其他 revision。把同一个目录传给 `run_agents.sh`（[运行一个策略](#运行一个策略)中 `AGENT_ENV` 里的 `HF_DATASETS_CACHE`）。

```sh
export HF_DATASETS_CACHE=/abs/path/on/agent/swebench-datasets   # new: the "mkdir" line creates it
mkdir "$HF_DATASETS_CACHE"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
IDS=$(cut -d ' ' -f 1 examples/swebench/verified-32.txt)
"$MINI/bin/python" examples/swebench/build_tasks.py --output-dir /abs/path/to/tasks --pool $IDS
```

`IDS` 是按运行顺序排列的 32 个 instance id。`build_tasks.py` 加载固定 revision 的数据集（`REVISION`，`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`），按 SHA-256 固定值检查每个 problem statement（32 个全部有固定值），写出 `tasks-pool.json`（32 个任务，按这个顺序，也就是 agent 的启动顺序）和每个 instance 一个 `task-<instance_id>.json`，从不覆盖文件，并打印一份 JSON 报告。把 `tasks-pool.json` 复制到 GPU 一侧：服务读取它（`TASKS`），agent 主机只拿到 instance id。

## 运行一个策略

每个策略一个全新的服务，各自一个 GPU 作业。下面的脚本就是 `examples/slurm/serve_gh200.sh` 在 `MODE=swebench` 下的流程（服务、READY、隧道、agent、停止），用的是 benchmark 服务的参数：`serve_gh200.sh` 本身不能传 `--active-token-budget`、`--goal-rewriter-batch` 或 `--admission-overlap`。设置好[运行环境](../install.zh-CN.md#3-运行环境)后在 GPU 节点上运行它，并设置 `REPO`、`RELEASE`、`RUNS`、`MODELS`、`TASKS`（复制过来的 `tasks-pool.json`）和 `LABEL`（每次尝试都用新的）。agent 得到 `MAX_RUNNING=16`（服务的 `--max-running-sessions`）、`MAX_TOKENS=4096`（服务的 `--max-tokens`；`run_agents.sh` 默认为 1536）和 `STEP_LIMIT=120`（`run_agents.sh` 默认为 90；最多 256，因为服务每个 session 最多服务 256 个请求，之后的请求会被拒绝，见 [XML 协议](../running.zh-CN.md#61-mini-swe-agent-xml-协议)），`AGENT_ENV` 为加了客户端设置行的副本（[Agent](#agent)）带上 `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1` 和 `REQUEST_TIMEOUT=3600`。

```sh
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
SHARED=(...)                  # replace (...) with the SHARED array of "Server commands" > Qwen3.6-27B
POLICY=observation_query      # for another policy, replace --active-token-budget 8192 in SERVE with its flags
SERVER_RELEASE="$RELEASE"     # the server's release, for serve.py and the READY check; Gemma 4: "$RELEASE-gemma4"
OUT="$RUNS/$POLICY-$LABEL"
SERVE=(python -B serve.py --release-dir "$SERVER_RELEASE" --task "$TASKS" --policy "$POLICY" "${SHARED[@]}"
       --active-token-budget 8192 --output "$OUT" --port 8000)
AGENT_HOST=user@agent-host TUNNEL_KEY=/abs/path/to/tunnel_key KNOWN_HOSTS=/abs/path/to/agent_known_hosts
# AGENT_PORT: the loopback port on the agent host where relay.py listens and the agents connect
AGENT_SOCKET="/abs/path/on/agent/sock/$POLICY-$LABEL.sock" AGENT_PORT=20000
AGENT_RUN=/abs/path/on/agent/swebench-client/run_agents.sh AGENT_OUT="/abs/path/on/agent/runs/$POLICY-$LABEL"
AGENT_ENV="MINI_PYTHON=/abs/path/on/agent/mini-env/bin/python MSWEA_CONFIG_DIR=/abs/path/on/agent/mswea-config"
AGENT_ENV="$AGENT_ENV HF_DATASETS_CACHE=/abs/path/on/agent/swebench-datasets"   # the datasets cache of Tasks
AGENT_ENV="$AGENT_ENV MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1 REQUEST_TIMEOUT=3600"   # client retries and timeout (Agents)
"${SERVE[@]}"                                   # dry run
"${SERVE[@]}" --start > "$OUT.server.log" 2>&1 &
SERVER=$!
deadline=$((SECONDS + 900))
while true; do
  set +e; python -B examples/slurm/server_checks.py ready "$OUT" "$POLICY" "$SERVER_RELEASE"; rc=$?; set -e
  (( rc == 0 )) && break
  (( rc == 1 )) || { kill -TERM "$SERVER"; wait "$SERVER" || true; exit 1; }
  kill -0 "$SERVER" 2>/dev/null || { echo "server exited before READY: read $OUT.server.log" >&2; exit 1; }
  (( SECONDS < deadline )) || { echo "no READY within 900 s" >&2; kill -TERM "$SERVER"; wait "$SERVER" || true; exit 1; }
  sleep 5
done
SSH_OPTS=(-i "$TUNNEL_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes
          -o UserKnownHostsFile="$KNOWN_HOSTS" -o GlobalKnownHostsFile=/dev/null -o ConnectTimeout=20)
ssh -N -R "$AGENT_SOCKET:127.0.0.1:8000" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 "${SSH_OPTS[@]}" "$AGENT_HOST" > "$OUT.tunnel.log" 2>&1 &
TUNNEL=$!
up=0
for _ in $(seq 1 30); do
  kill -0 "$TUNNEL" 2>/dev/null || break
  if ssh -T "${SSH_OPTS[@]}" "$AGENT_HOST" curl -sf --max-time 5 --unix-socket "$AGENT_SOCKET" \
       http://localhost/v1/models 2>/dev/null | grep -q '"selective-prefix"'; then up=1; break; fi
  sleep 2
done
harness_rc=0
if (( up == 1 )); then
  INSTANCES=$(python -B -c 'import json,sys; print(" ".join(t["instance_id"] for t in json.load(open(sys.argv[1]))))' "$TASKS")
  ssh -T "${SSH_OPTS[@]}" "$AGENT_HOST" env $AGENT_ENV MAX_RUNNING=16 MAX_TOKENS=4096 STEP_LIMIT=120 \
    "$AGENT_RUN" "$AGENT_SOCKET" "$AGENT_PORT" "$AGENT_OUT" $INSTANCES > "$OUT.agents.log" 2>&1 || harness_rc=$?
else
  echo "tunnel not up (see $OUT.tunnel.log)" >&2; harness_rc=1
fi
kill -TERM "$TUNNEL" 2>/dev/null || true; wait "$TUNNEL" 2>/dev/null || true
kill -TERM "$SERVER"                            # SIGTERM to the server PID only
set +e; wait "$SERVER"; server_rc=$?; set -e
echo "server exit $server_rc, harness exit $harness_rc"
(( server_rc == 0 )) || exit 1
python -B examples/slurm/server_checks.py cleanup "$OUT"
exit "$harness_rc"
```

`server_checks.py cleanup` 输出 `CLEAN_SHUTDOWN`。`$OUT.agents.log` 的最后一行是 `run_agents.sh` 的汇总（`runner_exit`、`seconds`、`containers_left`、`preds`）。Gemma profile 的 `SHARED` 和策略参数取自“服务命令” > Gemma 4 31B IT，并设 `SERVER_RELEASE="$RELEASE-gemma4"`：这一个变量同时把服务的 release 交给 `serve.py --release-dir` 和 `server_checks.py ready`。stock 对照用[服务参数](#服务参数)一节的 stock 生命周期替换 dry run、`--start` 那一行、READY 循环和 `server_checks.py cleanup`：用该节的一个 `STOCK_CMD` 运行 `"${STOCK_CMD[@]}" > "$OUT.server.log" 2>&1 &`，READY 循环检查 `GET /health` 和 `GET /v1/models`（[SWEBENCH_BENCHMARK.zh-CN.md 的 A.4](../SWEBENCH_BENCHMARK.zh-CN.md#a4-stock-vllm-对照)），最后同样是 `SIGTERM` 和 `wait`，但不做退出状态和清理检查（[SWEBENCH_BENCHMARK.zh-CN.md 的 B.5](../SWEBENCH_BENCHMARK.zh-CN.md#b5-slurm-作业脚本) 给出了这个流程）；agent 得到 `STOCK=1`。如果 SSH 连接断开，`run_agents.sh` 及其 runner 会在 agent 主机上继续运行：在那里停止它们（[停止 agent](#停止-agent)）。如果服务已经在 agent 主机本身的 `http://127.0.0.1:<port>/v1` 上应答（例如服务和 agent 在同一台带 GPU 的 x86_64 机器上），就用 `-` 代替 socket 并跳过隧道：`run_agents.sh - <port> <output dir> <instance id>...`；这种形式不在已验证的运行范围内。

### 停止 agent

`run_agents.sh` 在自己的进程组中运行：其中只有它的中继、它的 runner 和这些 runner 的 `docker` 客户端。它把进程组 id 写到输出目录（`AGENT_OUT`）中的 `agents.pgid`，并把各 instance 固定的镜像写到同一目录的 `images.txt`。在 agent 主机上，把 `RUN_OUT` 设为那次运行的 `AGENT_OUT`，第一行用 `SIGTERM` 停止该进程组；随后的循环列出由那次运行的 instance 镜像启动、仍在运行的容器，Mini-SWE-Agent 会异步停止它们；循环仍列出的容器用 `docker stop <id>` 停止（只停这些 id）。不要用范围宽泛的 `pkill` 或停止无关容器的方式来停止 agent。

```sh
RUN_OUT=/abs/path/on/agent/runs/observation_query-1
kill -TERM -- "-$(cat "$RUN_OUT/agents.pgid")"
while read -r instance image; do docker ps -q --filter "ancestor=$image"; done < "$RUN_OUT/images.txt"
```

## 评测

评测数据集是 revision `78f471bf655a3137b2e8a75af1501690ec009ec3` 的 `SWE-bench/SWE-bench_Verified`（在 `evaluate.sh` 中与其 test split 文件的 SHA-256 一起固定）：评测程序（`02e7a74` 的 SWE-bench，版本 5.0.2）从该数据集读取每个 instance 的镜像名和评测脚本。`evaluate.sh` 获取该 revision 的 split 文件（或从 Hugging Face 缓存中取：第一次评测需要联网），检查它的 SHA-256，再作为本地文件传给评测程序；它也像 `run_agents.sh` 那样检查 instance 的镜像。每次运行之后，在 agent 主机上、在本仓库目录中运行（最后一个参数是那次运行的 `AGENT_OUT`）：

```sh
EVAL_PYTHON="$EVAL/bin/python" examples/swebench/evaluate.sh observation_query-1 observation_query \
  /abs/path/on/agent/eval/observation_query-1 /abs/path/on/agent/runs/observation_query-1
```

参数依次是一个 run id、一个标签、一个新的评测目录（父目录已存在的绝对路径；`evaluate.sh` 在其中写入数据集文件 `dataset.parquet` 和镜像列表 `images.txt`），以及一个或多个 `run_agents.sh` 的输出目录。`evaluate.sh` 为每个 instance 输出一行 JSON（`instance_id`、`resolved`、`empty_patch`、`error`、`patch_chars`），最后一行是 `{"evaluator_exit": ..., "instances": 32, "report": ..., "report_missing": false, "resolved": N, "run_id": ...}`：解决率是 `resolved` / `instances`（没有 prediction 的 instance 也计入，按空 patch 评测）。

## 用 Slurm 脚本跑一个小组

`examples/slurm/serve_gh200.sh` 在 `MODE=swebench` 下于一个 Slurm 作业中运行一个服务和一个小的 agent 组，使用脚本自己的服务参数（它不能传 benchmark 服务的 `--active-token-budget`、`--goal-rewriter-batch` 或 `--admission-overlap`）。一个三个任务的组，在 agent 主机上设好[任务](#任务)一节的 `HF_DATASETS_CACHE` 后生成（一个 `tasks-g0.json`，每组最多五个任务）；把它复制到 GPU 一侧：

```sh
"$MINI/bin/python" examples/swebench/build_tasks.py --output-dir /abs/path/to/tasks --max-sessions 5 \
  astropy__astropy-12907 astropy__astropy-13236 astropy__astropy-14096
```

用 `sbatch` 提交，`ACCOUNT`、`PARTITION` 和 `MAIL` 与 [Slurm 作业](../running.zh-CN.md#45-gpu-作业slurm)相同。`RUN_ROOT` 必须已存在且路径要短（`TMPDIR` 在它下面）；agent 主机上 `AGENT_SOCKET` 所在的 0700 目录和 `AGENT_OUT` 的父目录必须已存在。这里 agent 使用 `run_agents.sh` 的默认值（`MAX_TOKENS` 1536，即服务默认的回复上限，以及 `STEP_LIMIT` 90），以及 checkout 中不带 [Agent](#agent) 一节客户端设置行的模板：即 pilot 的 model override。`--mem=113560M` 是一张 GH200 在节点主机内存中的份额，与已验证的 SWE-bench 作业申请的相同（[SWEBENCH_BENCHMARK.zh-CN.md 的 A.3“主机内存”](../SWEBENCH_BENCHMARK.zh-CN.md#a3-本-runtime服务参数与-vllm-参数)）。

```sh
export REPO VENV RELEASE
export TASKS=/abs/path/to/tasks/tasks-g0.json POLICY=goal_query KV_GIB=24 REWRITER="$MODELS/Qwen3-0.6B-c1899de"
export PAGE_TOKENS=16 KV_LAYOUT=decoupled
export RUN_ROOT=/abs/short/jobs MODE=swebench AGENT_SECONDS=5400
export AGENT_HOST=user@agent-host TUNNEL_KEY=/abs/path/to/tunnel_key KNOWN_HOSTS=/abs/path/to/agent_known_hosts
export AGENT_SOCKET=/abs/path/on/agent/sock/g0-goal_query.sock AGENT_PORT=20000
export AGENT_RUN=/abs/path/on/agent/examples/swebench/run_agents.sh AGENT_OUT=/abs/path/on/agent/runs/goal_query-g0
export AGENT_ENV="MINI_PYTHON=/abs/path/on/agent/mini-env/bin/python MSWEA_CONFIG_DIR=/abs/path/on/agent/mswea-config"
export AGENT_ENV="$AGENT_ENV HF_DATASETS_CACHE=/abs/path/on/agent/swebench-datasets"   # the datasets cache of Tasks
sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus=1 \
  --cpus-per-task=16 --mem=113560M --time=03:45:00 --job-name=selective-prefix \
  --output="$RUN_ROOT/slurm-%j.out" --export=ALL --mail-type=BEGIN,END,FAIL --mail-user="$MAIL" \
  "$REPO/examples/slurm/serve_gh200.sh"
```

然后在 agent 主机上评测这个组（它的 `AGENT_OUT`），方法同[评测](#评测)：

```sh
EVAL_PYTHON="$EVAL/bin/python" examples/swebench/evaluate.sh my-run-goal_query goal_query \
  /abs/path/on/agent/eval/goal_query /abs/path/on/agent/runs/goal_query-g0
```

## 服务命令

本 benchmark 各服务的 dry run，每个策略一条（加 `--start` 即启动服务）。`TASKS` 是任务文件，`LABEL` 是每次尝试新用的标签；发布仓库的 `tests/test_docs_commands.py` 会 dry run 这些命令。

### Qwen3.6-27B

```sh
SHARED=(--max-running-sessions 16 --kv-cache-gib 25 --page-tokens 16 --kv-layout decoupled
        --max-model-len 262144 --max-tokens 4096 --dram-arena-gib 70 --connector-pool-gib 1 --cuda-graphs full
        --decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io --language-model-only)
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy full "${SHARED[@]}" \
  --output "$RUNS/full-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy observation_query "${SHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy random_fill "${SHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/random_fill-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy goal_query "${SHARED[@]}" \
  --active-token-budget 8192 --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --goal-rewriter-batch 4 \
  --admission-overlap 16 --output "$RUNS/goal_query-$LABEL" --port 8000
```

### Gemma 4 31B IT

```sh
SHARED=(--max-running-sessions 16 --page-tokens 16 --max-model-len 131072 --max-tokens 4096
        --dram-arena-gib 72 --connector-pool-gib 1 --cuda-graphs full --decode-fastpath --decode-fastpath2
        --decode-fastpath3 --deferred-completion-io --language-model-only --release-idle-pages
        --max-num-batched-tokens 4096)
python -B serve.py --release-dir "$RELEASE-gemma4" --task "$TASKS" --policy full --kv-cache-gib 32 \
  "${SHARED[@]}" --output "$RUNS/gemma-full-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE-gemma4" --task "$TASKS" --policy observation_query --kv-cache-gib 25 \
  "${SHARED[@]}" --active-token-budget 8192 --output "$RUNS/gemma-observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE-gemma4" --task "$TASKS" --policy random_fill --kv-cache-gib 25 \
  "${SHARED[@]}" --active-token-budget 8192 --output "$RUNS/gemma-random_fill-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE-gemma4" --task "$TASKS" --policy goal_query --kv-cache-gib 25 \
  "${SHARED[@]}" --active-token-budget 8192 --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" \
  --goal-rewriter-batch 4 --goal-rewriter-batch-tokens 15360 --admission-overlap 16 \
  --output "$RUNS/gemma-goal_query-$LABEL" --port 8000
```
