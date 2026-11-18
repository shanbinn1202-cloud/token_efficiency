# 用 Harbor 的 Terminus 2 跑 Terminal-Bench 2.0

[English](terminal-bench.md) | 简体中文

> 在 integrate-1107 从发布仓库同步（release-repo `5b47394` 的 benchmark 页面）：`examples/terminalbench/` 中的客户端和本页都来自发布仓库，只在发布仓库中有效的链接改指本仓库的 README 和 benchmark 文档。正式记录的配置、各次运行及其作业 id 见 [TERMINALBENCH_BENCHMARK.zh-CN.md](../TERMINALBENCH_BENCHMARK.zh-CN.md)。下面的“服务命令”是发布仓库的设置：Qwen 服务用 `--kv-cache-gib 25 --max-model-len 262144` 和 `--dram-arena-gib 70`，即作业 3461661 的配置（[docs/reference.zh-CN.md 4.9 节](../reference.zh-CN.md#49-上下文上限与回复上限)），`make_job.py` 把这个上下文交给 Terminus 2；TERMINALBENCH_BENCHMARK.zh-CN.md 的 A.3 是同样的 Qwen 服务，该页更早的 dry run 用的是 32 GiB 和 65,536 token。自 2026-11-08 起，客户端和本页也给 Gemma 4 31B IT 它的 262,144 token 配置（TERMINALBENCH_BENCHMARK.zh-CN.md 的 A.14：25 GiB、K = 20、4,096 token 的前向、72 GiB 的 arena、没有 `full`），发布仓库中还没有：“服务命令”中的 Gemma 服务是 A.14 的设置，其后是 Gemma 的 stock 命令；`make_job.py --context 65536` 写出更早的 dry run 所用的 65,536 token 的 job；发布仓库的客户端副本（导出的 `release-src`）有同样的修改。

## 范围

- Terminal-Bench 2.0 的全部任务，Harbor `0.21.0` 及其 Terminus 2 agent（不作修改），16 个并发 trial（`-n 16`）。
- 每个策略一个全新的服务，另有一个 stock vLLM 0.24.0 对照：Qwen3.6-27B 用 `full`、`observation_query`、`random_fill` 和 `goal_query`；Gemma 4 profile 用 `observation_query`、`random_fill` 和 `goal_query`（`full` 在 262,144 token 下放不下）。
- 协议：`--protocol chat-text`（[docs/running.zh-CN.md 6.7 节](../running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)），下面的命令关闭 thinking（正式记录的配置；当前的运行见 [docs/status.zh-CN.md](../status.zh-CN.md#当前配置)）。
- 主机：Harbor、任务容器及其 verifier 在一台 x86_64 Linux Docker 主机（agent 主机）上运行（Terminal-Bench 2.0 的任务镜像是 x86_64 的；它的要求见 [agent 主机](../TERMINALBENCH_BENCHMARK.zh-CN.md#b1-agent-主机)）；服务运行在 GPU 节点上（测试用的是一台 aarch64 GH200），通过反向 SSH 隧道和 loopback 中继访问（[docs/running.zh-CN.md 6.4 节](../running.zh-CN.md#64-从其他主机访问服务)）。只验证过这种两台主机的设置。Harbor 也可以和服务一起运行在一台带 GPU 的 x86_64 主机上（[运行一个策略](#运行一个策略)）；这时任务容器的内存和 CPU 要加在服务的之上。不支持在一台 aarch64 机器上同时运行两者。

**GPU 运行：** 在 89 个任务中的 16 个（见下）上以 `-n 16` 做的 trial 运行，连同每个作业的 runtime 提交、模型、上下文和判定，见 [docs/status.zh-CN.md](../status.zh-CN.md)；全部任务上的 benchmark 还没有运行。发布仓库的 `5b47394` 是 runtime `5466fc2` 的导出，即 `cf2b28d` 加上 serve.py 的速度默认值（每个都有关闭开关）和测试修复。

trial 运行的 16 个任务，每个的 agent 超时都是 900 秒，从此前 Terminus 2 运行中 prompt 最大的任务里挑选：

```text
build-cython-ext build-pmars configure-git-webserver crack-7z-hash db-wal-recovery extract-elf
fix-code-vulnerability gcode-to-text git-multibranch make-doom-for-mips password-recovery pytorch-model-cli
raman-fitting sanitize-git-repo tune-mjcf vulnerable-secret
```

`examples/terminalbench/tasks.json` 把它们命名为 `trial16`。`smoke32` 另加 16 个 agent 超时为 900 秒的任务（没有在 GPU 上运行过），`all` 是 89 个任务。这个文件固定数据集，即 Harbor 的包 `terminal-bench/terminal-bench-2`，固定在 `sha256:c6fc2e2382c1dbae99b2d5ecd2f4f4a60c3c01e0d84642d69b4afd92e99d078b`，并为每个任务固定 Harbor 为它记录的包 ref 和任务校验和、它的 agent 和 verifier 超时、它的镜像和资源限制。每个任务包按标签（例如 `alexgshaw/build-pmars:20251031`）而不是按 digest 指定镜像，Harbor 也按这个标签启动它：Docker Compose 只在镜像不存在时才拉取，所以一次新的拉取得到的是该标签在拉取时指向的镜像。对 `smoke32` 的 32 个任务（包括 `trial16`），`tasks.json` 还记录了测试用的 agent 主机上该镜像的 digest（`docker_image_digest`，取自 `docker image inspect`；这些镜像在 trial 运行之前拉取）；其余 57 个任务没有记录 digest。如果 job 的任务中某个已存在的镜像 digest 不同，`run_harbor.sh` 会拒绝运行（[Agent 主机的准备](#agent-主机的准备)）；不存在的镜像由 Harbor 按标签拉取，不做检查。

## 服务

各服务的命令见下面的“服务命令”。主要设置：

- `--max-running-sessions 20`，对应 Harbor 的 `-n 16`，两个 profile 都是如此（65,536 token 下的 Gemma trial 运行用的是 17）：Terminus 2 从不关闭它的 session，所以已结束 trial 的 session 一直打开，直到被驱逐；Terminus 的摘要子调用还会打开它们自己的 session。4 个空余 session 容纳这些 session，所以新 session 不必驱逐正在运行的 trial。
- `--page-tokens 16 --kv-layout decoupled`；query 策略和 `random_fill` 用 `--active-token-budget 8192`；`--max-tokens 4096`；`--connector-pool-gib 1`；锁页 Page arena 用 `--dram-arena-gib`；`--release-idle-pages`。
- 两个 profile 都用 `--kv-cache-gib 25 --max-model-len 262144`；Qwen 另加 `--dram-arena-gib 70`，Gemma 另加 `--dram-arena-gib 72` 和 `--max-num-batched-tokens 4096`（[TERMINALBENCH_BENCHMARK.zh-CN.md 的 A.14](../TERMINALBENCH_BENCHMARK.zh-CN.md#a14-gemma-4-31b-it)）。作业 3560498（按这些设置的一次 16 个任务的 dry run）之前的 Gemma trial 运行用的是 65,536 token。
- `goal_query` 另加 `--goal-rewriter-model`（或 `--no-goal-rewriter`）；它面向 Terminus 2 的 Goal 输入是暂定的（[docs/running.zh-CN.md 6.7 节](../running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)）。

## 客户端

`examples/terminalbench/`：

- `job-qwen3.6-27b.json`、`job-gemma4-31b-it.json`：每个模型 profile 的 Harbor job，使用 trial 运行的设置（[Job 设置](#job-设置)）。
- `tasks.json`：数据集的固定值和各任务集。
- `make_job.py`：从模板写出一个 job 文件；它填入端口、任务集、jobs 目录和 job 名称（加 `--context 65536` 时还有更早 trial 运行所用的 65,536 token 上下文），拒绝其他任何内容，也拒绝每个不是 Harbor 0.21.0 模型字段的键（Harbor 会忽略未知的键，不给警告）。
- `run_harbor.sh`：在 agent 主机上运行一个 job，依次是：按固定值检查 Harbor 的安装、job 文件、镜像检查、中继、服务检查、`harbor run`、关闭 session、容器检查、报告。
- `check_images.py`：镜像检查：job 中每个已存在的任务镜像都必须有 `tasks.json` 记录的 digest（`check`）；`list` 打印记录的 digest，供 `docker pull` 使用。
- `close_sessions.py`：运行之前的 `check`（服务与这个 job 相符，且还没有 session），以及运行之后的 `close`（关闭 Terminus 2 遗留的打开 session）。
- `report.py`：根据 Harbor 保存的 job 结果和 trial 结果，给出一个已结束 job 的任务数、异常和平均 reward。
- `requirements.lock`、`upstream/harbor-wheels.SHA256SUMS` 和 `upstream/terminus-2.SHA256SUMS`：Harbor 安装的固定值（每个包的哈希锁；Harbor、litellm 和 openai wheel 的 SHA-256，锁文件中也重复了它们；Terminus 2 的四个文件）。

这些文件只有 CPU 测试（`tests/test_tb_client.py`）；它们还没有在 GPU 作业中运行过。trial 运行用的是作业 harness 自己的 Harbor 脚本，job 设置和步骤顺序相同，只是没有镜像检查（`check_images.py`），它是在 trial 运行之后才加入的（见 `run_harbor.sh` 的文件头）。

### Job 设置

每次运行的 job，即 `make_job.py` 写出的内容（Harbor 0.21.0 的 `JobConfig`；agent 的 `kwargs` 传给 Terminus 2 的构造函数，再从那里传给 litellm）：

| 设置 | 取值 |
| --- | --- |
| agent | `terminus-2`（Terminus 2 2.0.0，使用其默认设置：JSON 解析器，距上下文上限 8,000 token 时做摘要） |
| `model_name` | `hosted_vllm/selective-prefix`，即服务的模型名 |
| `api_base` | `http://127.0.0.1:<port>/v1`：agent 主机上的中继，或同一主机上的服务 |
| `temperature` | `0.0`（服务端无论如何都贪心解码） |
| `llm_kwargs` | `max_tokens` 4096；`timeout` 3600，使在服务 admission 中等待的请求不会被取消并重发；`max_retries` 0 和 `num_retries` 0（litellm 和 OpenAI 客户端都不重试；Terminus 2 自己的重试仍然保留） |
| `model_info` | `max_input_tokens` 262144，即服务的上下文（用 `make_job.py --context 65536` 时为 65536，即更早 trial 运行的上下文）；`max_output_tokens` 4096；`input_cost_per_token` 和 `output_cost_per_token` 为 0。对 `hosted_vllm` 模型，Harbor 要求这四个字段 |
| `n_concurrent_trials` | 16（`-n 16`） |
| `n_attempts`、`retry` | 一次尝试，`max_retries` 0 |
| `timeout_multiplier` | 1.0：每个任务用自己的超时。`trial16` 和 `smoke32` 的每个任务的 agent 超时都是 900 秒，由 `make_job.py` 检查 |
| `environment` | `docker`，`delete` true（每个 trial 的容器在该 trial 结束时删除），`force_build` false |
| `verifier` | `disable` false：Harbor 在每个任务的 agent 之后运行该任务的测试，agent 超时后也运行 |
| `datasets` | `terminal-bench/terminal-bench-2`，取其固定的 `ref`；`task_names`：任务集中的任务，写作 `terminal-bench/<task>` |
| `quiet` | true |
| stock 对照 | `make_job.py --stock` 加上 `llm_call_kwargs` `{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}`：每次调用都关闭 thinking，因为 stock 服务保留 vLLM 默认的模板开关（本 runtime 的服务自己以 thinking 关闭渲染） |

字段名取自 Harbor 源码 tag `v0.21.0`（`src/harbor/models/job/config.py`、`src/harbor/models/trial/config.py`、`src/harbor/agents/terminus_2/terminus_2.py`；`hosted_vllm` 的字段：`src/harbor/llms/utils.py`）。Qwen trial 运行的 job 文件，打印出来（`--out -`）而不是写入文件：

```sh
python3 -I -B examples/terminalbench/make_job.py --profile qwen3.6-27b --tasks trial16 --port 20000 \
  --jobs-dir /abs/path/on/agent/runs/tb-example/jobs --job-name tb-example --out -
```

## Agent 主机的准备

agent 主机需要：

- 装有 Docker 的 Linux x86_64：Docker 正在运行，运行脚本的用户不需要交互式的 `sudo` 即可使用它（SWE-bench 页面在测试用的系统 Ubuntu 24.04 上安装它：[Agent 主机的准备](swe-bench.zh-CN.md#agent-主机的准备)），以及 Docker Compose v2：Harbor 用 `docker compose` 运行每个任务（测试用的主机：Ubuntu 的 `docker-compose-v2` 包 `2.40.3+ds1-0ubuntu1~24.04.1`，SWE-bench 页面的安装代码块会安装它；`docker compose version` 打印 `Docker Compose version 2.40.3+ds1-0ubuntu1~24.04.1`）。
- CPython 3.12，即 `python3.12`（Harbor 0.21.0 需要 Python 3.12 或更新的版本；trial 运行用的是 3.12.3），以及 `PATH` 上的 `python3`、`bash`、`curl`、`ss`、`sha256sum` 和 `setsid`。
- Harbor 第一次下载任务包、Docker 拉取它们的镜像时需要联网，还需要存放镜像的磁盘空间。
- 容器所需的资源：在 `-n 16` 下，每个 `trial16` 任务限制为 1 个 CPU 和 2,048 MB（`tasks.json` 中的 `memory_mb`），限制合计 16 个 CPU 和 32,768 MB（`smoke32`：每个任务最多 4,096 MB；`all`：最多 4 个 CPU 和 8,192 MB）。测试用的 agent 主机有 16 个 CPU 和 33 GB 内存。
- 本仓库的一份副本：`run_harbor.sh` 使用它所在目录中的文件和 `examples/swebench/relay.py`。`HARBOR_RUN`（[运行一个策略](#运行一个策略)）是这份副本中 `examples/terminalbench/run_harbor.sh` 的路径。
- 隧道：与 SWE-bench 相同（[Agent 主机的准备](swe-bench.zh-CN.md#agent-主机的准备)），GPU 节点能用一把 key（`TUNNEL_KEY`）和一个固定 agent 主机 host key 的 known-hosts 文件（`KNOWN_HOSTS`）向 agent 主机发起出站 SSH 连接；`AGENT_SOCKET` 是 agent 主机上一个 0700 目录中的新 socket 路径，`AGENT_OUT` 是一个父目录已存在的新目录。

Harbor 环境，在 agent 主机上、在本仓库目录中创建。`examples/terminalbench/requirements.lock` 固定 Harbor 0.21.0、litellm 1.97.0 和 openai 2.54.0（即 `upstream/harbor-wheels.SHA256SUMS` 中的 wheel）以及它们需要的 80 个包，版本是 trial 运行时 agent 主机上 Harbor 环境中的版本，每个包都带该环境所装 PyPI wheel 的 SHA-256；有了 `--require-hashes`，pip 要么正好安装这些文件，要么停止。然后按 `upstream/terminus-2.SHA256SUMS`（它们在 Harbor 0.21.0 wheel 的 `RECORD` 中的条目）检查已安装的包中 Terminus 2 的四个文件：

```sh
TB_VENV=/abs/path/on/agent/harbor-env       # new: the "python3.12 -m venv" line creates it
python3.12 -m venv "$TB_VENV"
"$TB_VENV/bin/python" -m pip install --only-binary=:all: --require-hashes -r examples/terminalbench/requirements.lock
SITE=$("$TB_VENV/bin/python" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
(cd "$SITE" && sha256sum --check) < examples/terminalbench/upstream/terminus-2.SHA256SUMS
"$TB_VENV/bin/harbor" --version
```

`sha256sum --check` 对四个文件各打印一个 `OK`，`harbor --version` 打印 `0.21.0`。锁文件记录的是 trial 运行时 agent 主机上的 Harbor 环境，在运行之后从其已安装的包读出；那个环境是在主机的 CPython 3.12.3 上用 uv 创建的，还装有这三个包不需要的 15 个包（datasets 和 mini-swe-agent 及其依赖），锁文件没有收录它们。没有从锁文件做过全新安装。每次运行之前，`run_harbor.sh` 都会再次检查这三个版本和 Terminus 2 的四个文件。

任务镜像：`run_harbor.sh` 在启动中继之前运行 `check_images.py check --tasks <task set>`，并把结果写到 `images.json`：`IMAGES_OK`，附带已检查的任务、镜像不存在的任务（Harbor 按标签拉取）和没有记录 digest 的任务；`IMAGES_REFUSED`（某个已存在的镜像 digest 不同，或 `docker` 出错）会停止运行。要让一个任务集的每个镜像都得到检查，就在运行之前按记录的 digest 拉取每个镜像，再给它打上 Harbor 使用的标签（这不属于 trial 运行，它们的镜像是按标签拉取的）：

```sh
python3 -I -B examples/terminalbench/check_images.py list --tasks trial16 | while read -r pinned tagged; do
  docker pull "$pinned" && docker tag "$pinned" "$tagged"
done
```

## 运行一个策略

每次运行一个全新的服务，各自一个 GPU 作业。下面的脚本还没有在 GPU 作业上端到端运行过（trial 运行用的是作业 harness 的脚本；[客户端](#客户端)）。它启动服务、等待 READY、打开隧道、通过 SSH 在 agent 主机上运行 `run_harbor.sh`，然后停止服务；设置好[运行环境](../install.zh-CN.md#3-运行环境)后在 GPU 节点上运行它，并设置 `REPO`、`RELEASE`、`RUNS`、`MODELS` 和 `LABEL`（每次尝试都用新的），与 SWE-bench 的脚本相同（[运行一个策略](swe-bench.zh-CN.md#运行一个策略)）。在 agent 主机上，`run_harbor.sh` 检查 Harbor 安装，用 `make_job.py` 写出 job 文件，检查任务镜像（`check_images.py`），启动中继，检查服务与这个 job 相符（`close_sessions.py check`），运行 `harbor run -c job.json -y`，关闭 Terminus 2 遗留的打开 session（`close_sessions.py close`，经由隧道：服务看到的是一个 loopback 对端），等到 job 的 trial 不再留有任何容器，再运行 `report.py`。

```sh
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
SHARED=(...)                  # replace (...) with the SHARED array of "Server commands" > Qwen3.6-27B
POLICY=observation_query      # for another policy, replace --active-token-budget 8192 in SERVE with its flags
SERVER_RELEASE="$RELEASE"     # the server's release, for serve.py and the READY check; Gemma 4: "$RELEASE-gemma4"
OUT="$RUNS/tb-$POLICY-$LABEL"
SERVE=(python -B serve.py --release-dir "$SERVER_RELEASE" --policy "$POLICY" "${SHARED[@]}"
       --active-token-budget 8192 --output "$OUT" --port 8000)
AGENT_HOST=user@agent-host TUNNEL_KEY=/abs/path/to/tunnel_key KNOWN_HOSTS=/abs/path/to/agent_known_hosts
# AGENT_PORT: the loopback port on the agent host where relay.py listens and Harbor's agents connect
AGENT_SOCKET="/abs/path/on/agent/sock/tb-$POLICY-$LABEL.sock" AGENT_PORT=20000
HARBOR_RUN=/abs/path/on/agent/examples/terminalbench/run_harbor.sh AGENT_OUT="/abs/path/on/agent/runs/tb-$POLICY-$LABEL"
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
  ssh -T "${SSH_OPTS[@]}" "$AGENT_HOST" env HARBOR_VENV=/abs/path/on/agent/harbor-env \
    "$HARBOR_RUN" "$AGENT_SOCKET" "$AGENT_PORT" "$AGENT_OUT" qwen3.6-27b trial16 > "$OUT.harbor.log" 2>&1 \
    || harness_rc=$?
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

`server_checks.py cleanup` 输出 `CLEAN_SHUTDOWN`。`$OUT.harbor.log` 的最后一行是 `run_harbor.sh` 的汇总：`job`、`harbor_exit`、`close`（`SESSIONS_CLOSED`）、`containers_left`（`docker ps` 失败时为 `null`）、`images`（`IMAGES_OK`）、`report`（`REPORT_OK`）、`planned`、`n_exceptions`、`solved` 和 `mean_reward`；只有当 Harbor 以 0 退出、每个 session 都已关闭、`docker ps` 没有列出 job 的 trial 的任何容器，并且 `report.py` 接受了这个 job 时，`run_harbor.sh` 才以 0 退出。它的文件在 agent 主机上的 `AGENT_OUT` 中：`job.json`、`images.json`、`check.json`、`harbor.log`、Harbor 的 job 目录 `jobs/<the name of AGENT_OUT>/`、`close.json` 和 `report.json`。在 `-n 16` 下，`trial16` 的 16 个 trial 同时运行，每个最多运行到它 900 秒的 agent 超时，再加上准备和测试的时间；GPU 作业的时间上限必须在 READY 之后留出这段时间。

其他运行：

- 另一个策略或 Gemma profile：把“服务命令”中的服务命令放进 `SERVE`（Gemma：它的 `GSHARED` 数组和参数），把它的 release 放进 `SERVER_RELEASE`（Gemma：`SERVER_RELEASE="$RELEASE-gemma4"`；这一个变量同时把服务的 release 交给 `serve.py --release-dir` 和 `server_checks.py ready`），并把 profile `gemma4-31b-it` 作为 `run_harbor.sh` 的第四个参数。profile 必须是服务的 profile：`close_sessions.py check` 会拒绝上下文、回复上限、session 池或模型 profile 与 job 不同的服务。
- 另一个任务集：把 `smoke32` 或 `all` 作为第五个参数。
- 更早 trial 运行的 65,536 token 上下文：在启动 `run_harbor.sh` 的 SSH 调用的 `env` 中、`HARBOR_VENV` 旁边设置 `TB_CONTEXT=65536`（`run_harbor.sh` 把它作为 `make_job.py --context 65536` 传入；在 GPU 节点上设置的变量到不了 agent 主机），对应 `--max-model-len 65536` 的服务（那些运行的 Gemma 设置见 [TERMINALBENCH_BENCHMARK.zh-CN.md 的 A.14](../TERMINALBENCH_BENCHMARK.zh-CN.md#a14-gemma-4-31b-it)）。
- stock vLLM 对照：vLLM 的 OpenAI 服务，在 `127.0.0.1:8000` 上服务 `selective-prefix`，以 job 的上下文作为 `--max-model-len`（对 Qwen，即 [TERMINALBENCH_BENCHMARK.zh-CN.md 的 A.4](../TERMINALBENCH_BENCHMARK.zh-CN.md#a4-stock-vllm-对照) 中的命令，262,144 token、25 GiB，也就是 SWE-bench 页面 stock 命令的大小，见[服务参数](swe-bench.zh-CN.md#服务参数)；对 Gemma，即“服务命令” > Gemma 4 31B IT 下的 stock 命令，262,144 token、25 GiB，加 `--max-num-batched-tokens 4096`，因为 vLLM 自己的启动检查在那里拒绝 8,192），用 [TERMINALBENCH_BENCHMARK.zh-CN.md 的 A.4](../TERMINALBENCH_BENCHMARK.zh-CN.md#a4-stock-vllm-对照) 的 stock 生命周期启动和停止，替换 dry run、`--start` 那一行、READY 循环和 `server_checks.py cleanup`：`serve.py` 的 dry run、`--start`、READY 文件和清理回执的检查只适用于 selective release，stock 对照用 HTTP 就绪检查和进程关闭。这时 `stock` 是 `run_harbor.sh` 的第六个参数；job 在每次调用时都发送 thinking 关闭，也不关闭任何 session（stock vLLM 没有 chat-text 路由）。哪些 stock 对照运行过（作业 3499759：thinking 开，Qwen，262,144 token，25 GiB；作业 3560498：Gemma 4，262,144 token，25 GiB），见 [docs/status.zh-CN.md](../status.zh-CN.md)。
- Harbor 和服务在同一台带 GPU 的 x86_64 主机上（未经验证）：照上面启动服务，然后在同一台主机上运行 `run_harbor.sh`，不用隧道和 SSH 调用，socket 用 `-`，端口用服务的端口；Harbor 的 agent 这时直接使用 `http://127.0.0.1:8000/v1`。容器的内存和 CPU 要加在服务的之上。

```sh
HARBOR_VENV=/abs/path/to/harbor-env examples/terminalbench/run_harbor.sh - 8000 \
  "$RUNS/tb-observation_query-$LABEL-harbor" qwen3.6-27b trial16
```

如果 SSH 连接断开，`run_harbor.sh` 和 Harbor 会在 agent 主机上继续运行。Harbor 在它自己的进程组中运行，`run_harbor.sh` 把进程组 id 写到 `AGENT_OUT/harbor.pgid`；在 agent 主机上运行 `kill -TERM -- -<that id>` 只停止 Harbor，不影响其他任何进程（不在已验证的运行范围内）。job 的 trial 的容器就是 Docker Compose 项目 `<trial directory name, lowercased>__env` 和 `<trial directory name, lowercased>__verifier__...`（`docker ps -a --format '{{.Label "com.docker.compose.project"}}'`）。

## 结果

Harbor 自己给每个 trial 评分：agent 结束（或超时）之后，它在任务的环境中运行该任务的测试，并记录测试的 reward，任务解决时为 1，否则为 0。准确率就是这个逐 trial 的判定。Harbor 在 job 目录中保存 job 的 `result.json`（它的计数和统计）和每个 trial 的 `result.json`（它的任务、reward 和异常，如果有的话）。`report.py` 读取它们；`run_harbor.sh` 在最后运行它，你也可以在 agent 主机上用 job 文件再次运行它：

```sh
python3 -I -B examples/terminalbench/report.py --job /abs/path/on/agent/runs/tb-observation_query-1/job.json
```

它输出一行 JSON。为 `"status": "REPORT_OK"`（退出码 0）时：`planned`（job 的任务数）、`trials`、`finished`、`n_exceptions` 和 `exceptions`（每种异常类型及其任务；agent 超时是 `AgentTimeoutError`，它的 trial 仍有测试的 reward）、`rewarded`、`solved`（reward 为 1）、`unrewarded`（没有 reward 的 trial，按 0 计）、`mean_reward`（计划内任务的平均 reward，即 Harbor 的 mean 指标）和 `harbor_mean`（Harbor 自己从 job 统计中得出的平均值，如果统计中有的话；两者必须相等）。`warnings` 列出 `task_checksum` 与 `tasks.json` 中不同的 trial。如果一个 job 不是每个计划内任务恰好有一个固定包的已结束 trial，它会拒绝该 job（退出码 1，`"status": "REPORT_REFUSED"`，列出每个原因，不给出 `mean_reward`）：没有 job 结果或 job 未结束；计划内的任务没有 trial 结果；trial 没有 `result.json` 或 `finished_at`，或者被取消；不在计划内的任务的 trial，或同一任务的第二个 trial；任务包、agent 或模型与 job 的不同；统计中的计数与 trial 不符。

## 手动关闭 session

`run_harbor.sh` 会自己关闭 session。手动关闭时，在 GPU 节点上（服务只接受来自 loopback 对端的关闭请求）使用 `close_sessions.py`，它关闭每个打开的 session，然后检查没有剩下任何 session：

```sh
python3 -I -B examples/terminalbench/close_sessions.py close --base-url http://127.0.0.1:8000
```

或者用 `curl`，按 `GET /v1/chat-text/info` 给出的每个打开 session 的 `key_sha256` 各发一次关闭请求（Terminus 2 不发送 `X-Session-ID`；[docs/running.zh-CN.md 6.7 节](../running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)）：

```sh
curl -s http://127.0.0.1:8000/v1/chat-text/info
curl -s -X POST -H 'Content-Type: application/json' -d '{"key_sha256": "<key>"}' \
  http://127.0.0.1:8000/v1/chat-text/close
```

## 服务命令

本 benchmark 各服务的 dry run，每个策略一条（加 `--start` 即启动服务）。`LABEL` 是每次尝试新用的标签；发布仓库的 `tests/test_docs_commands.py` 会 dry run 这些命令。末尾的 Gemma stock 命令不是 `serve.py` 的 dry run：它启动 vLLM 自己的服务，发布仓库的导出不带它（导出只附加含 `serve.py` 行的 sh 代码块）。

### Qwen3.6-27B

```sh
SHARED=(--protocol chat-text --chat-text-slots 512 --chat-text-idle-timeout 600 --chat-text-evict-after 30
        --kv-cache-gib 25 --max-model-len 262144 --page-tokens 16 --kv-layout decoupled --cuda-graphs full
        --max-running-sessions 20 --max-tokens 4096 --dram-arena-gib 70 --connector-pool-gib 1
        --decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io
        --language-model-only --release-idle-pages)
python -B serve.py --release-dir "$RELEASE" --policy full "${SHARED[@]}" \
  --output "$RUNS/tb-full-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --policy observation_query "${SHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/tb-observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --policy random_fill "${SHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/tb-random_fill-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --policy goal_query "${SHARED[@]}" \
  --active-token-budget 8192 --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --admission-overlap 16 \
  --goal-rewriter-batch 4 --output "$RUNS/tb-goal_query-$LABEL" --port 8000
```

### Gemma 4 31B IT

```sh
GSHARED=(--protocol chat-text --chat-text-slots 512 --chat-text-idle-timeout 600 --chat-text-evict-after 30
         --kv-cache-gib 25 --max-model-len 262144 --page-tokens 16 --cuda-graphs full --max-running-sessions 20
         --max-tokens 4096 --max-num-batched-tokens 4096 --dram-arena-gib 72 --connector-pool-gib 1
         --decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io --language-model-only
         --release-idle-pages)
python -B serve.py --release-dir "$RELEASE-gemma4" --policy observation_query "${GSHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/tb-gemma-observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE-gemma4" --policy random_fill "${GSHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/tb-gemma-random_fill-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE-gemma4" --policy goal_query "${GSHARED[@]}" \
  --active-token-budget 8192 --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --admission-overlap 16 \
  --goal-rewriter-batch 4 --goal-rewriter-batch-tokens 20480 \
  --output "$RUNS/tb-gemma-goal_query-$LABEL" --port 8000
```

Gemma 的 stock 对照，即 vLLM 自己的 OpenAI 服务，用 [TERMINALBENCH_BENCHMARK.zh-CN.md 的 A.4](../TERMINALBENCH_BENCHMARK.zh-CN.md#a4-stock-vllm-对照) 的 stock 生命周期启动和停止（它的设置见 [A.14](../TERMINALBENCH_BENCHMARK.zh-CN.md#a14-gemma-4-31b-it)）：

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
"$VENV/bin/python" -I -B -m vllm.entrypoints.openai.api_server \
  --model "$MODELS/gemma-4-31b-it" --served-model-name selective-prefix --host 127.0.0.1 --port 8000 \
  --distributed-executor-backend uni --tensor-parallel-size 1 \
  --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 4096 \
  --kv-cache-memory-bytes 26843545600 --seed 42 \
  --enable-prefix-caching --prefix-caching-hash-algo sha256 --attention-backend TRITON_ATTN \
  --language-model-only
```
