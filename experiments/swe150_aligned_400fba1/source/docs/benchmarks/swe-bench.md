# SWE-bench Verified with Mini-SWE-Agent 2.4.2

English | [简体中文](swe-bench.zh-CN.md)

> Synced from the release repository (release-repo `5b47394`, its benchmark page) in integrate-1107: the client in `examples/swebench/` and this page are the release's, with the release-only links pointed at this repository's README and benchmark docs. The runs and their job ids are in [docs/status.md](../status.md). There are two Qwen configurations. The "Server commands" below are the release's, `--kv-cache-gib 25 --max-model-len 262144` (jobs 3461661 and 3461703, [docs/reference.md section 4.9](../reference.md#49-context-limit-and-reply-bound)): the setting of the Qwen SWE-bench runs since runtime `cf2b28d` (job 3461703: `goal_query` 19 and `observation_query` 20 of 32 resolved; the stock control at that setting, job 3507294: 21). SWEBENCH_BENCHMARK.md's Part A is the earlier configuration of record, 32 GiB and 131,072 tokens (the runs at `5b15b2f` and `120bc3c`), with its own server commands. The Gemma servers here have `--max-num-batched-tokens 4096` (job 3462014, [docs/history.md](../history.md#history), integrate-1026). The docs fix of 2026-11-08 changed this page where it was unclear or unsourced: the "On the GPU" paragraph, the client timeout and retry settings ([Agents](#agents)), the stock lifecycle, and versions that the runs' evidence does not record.

## Scope

- SWE-bench Verified (`princeton-nlp/SWE-Bench_Verified`, `test` split, at revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`): the 32 instances of `examples/swebench/verified-32.txt`, the first 32 of that split in dataset order, in that order ([Tasks](#tasks)); 32 registered tasks with at most 16 agents running; when an agent ends, the next instance in list order starts.
- The benchmark's inputs are pinned: the ordered instance list, the dataset revisions (`build_tasks.py` for the tasks, `evaluate.sh` for the evaluation), the 32 problem statements (SHA-256, checked by `build_tasks.py`) and every instance's image by digest (`verified-32.txt`, checked by `run_agents.sh` and `evaluate.sh` before anything starts). No step resolves the dataset's HEAD or an image's `:latest` from the registry during a run.
- One fresh server per policy (`full`, `observation_query`, `goal_query`, `random_fill`) and a stock vLLM 0.24.0 control, each in its own GPU job; one server runs all instances once, without interruption.
- Agent software: Mini-SWE-Agent `2.4.2` with the XML protocol (`--protocol mswea-xml`, the default; [docs/running.md section 6.1](../running.md#61-mini-swe-agent-xml-protocol)), litellm `1.103.1` and datasets; evaluator: SWE-bench at commit `02e7a74ffd0b707aab73d203fe87bdc7c76afc8e` (version 5.0.2), with datasets. The runs' evidence records no datasets version; the environments below install `5.0.1`, the release's pin. The agents and the evaluation run on an x86_64 Docker host, the agent host: the instance images are x86_64 (`sweb.eval.x86_64.*`). The server runs on the GPU node (the tested one is an aarch64 GH200) and the agents reach it through the reverse SSH tunnel of [Run one policy](#run-one-policy). One aarch64 machine for both is not supported: this release has no ARM64 images or emulation setup for the agents and the evaluator.

**GPU runs:** the runs of these clients, with each one's runtime commit, model, policy, thinking setting, KV size, context and verdict, are in [docs/status.md](../status.md), every job from 33xxxxx on; the first multi-session pilot, job 3299922, is in [History](../history.md#history). The release repository's `5b47394` is exported from `5466fc2`: `cf2b28d` with serve.py's speed defaults and two test fixes. The commands below are thinking off, the configuration of record; the current runs (thinking on, `--async-scheduling`) are described there too.

## Server flags

Shared by the four Qwen policies. The Gemma servers ([Gemma 4 31B IT](#gemma-4-31b-it)) differ: `--max-model-len 131072`, `--dram-arena-gib 72`, `--release-idle-pages`, `--max-num-batched-tokens 4096`, no `--kv-layout`, `--kv-cache-gib 32` for `full` and 25 for the others, and `--goal-rewriter-batch-tokens 15360` for `goal_query`:

| Flag | Value |
| --- | --- |
| `--task` | the pool file of the 32 instances, built by `examples/swebench/build_tasks.py --pool` |
| `--max-running-sessions` | `16` |
| `--kv-cache-gib` | `25` |
| `--page-tokens` / `--kv-layout` | `16` / `decoupled` |
| `--max-model-len` / `--max-tokens` | `262144` / `4096` |
| `--dram-arena-gib` | `70` |
| `--connector-pool-gib` | `1` (a speed default) |
| `--cuda-graphs` | `full` (a speed default) |
| `--decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io` | on (speed defaults) |
| `--language-model-only` | on (a speed default) |

Per policy:

| Policy | Selection flags | Admission flags |
| --- | --- | --- |
| `full` | none | none |
| `observation_query` | `--active-token-budget 8192` | none |
| `random_fill` | `--active-token-budget 8192` (default `--fill-seed`) | none |
| `goal_query` | `--active-token-budget 8192 --goal-rewriter-model <Qwen3-0.6B> --goal-rewriter-batch 4` | `--admission-overlap 16` |

The stock control is plain `vllm.entrypoints.openai.api_server` from the same environment, without any code of this runtime. Its lifecycle (the stock lifecycle): start `"${STOCK_CMD[@]}"` in the background; it is READY when `GET /health` returns 200 and `GET /v1/models` lists `selective-prefix` ([SWEBENCH_BENCHMARK.md A.4](../SWEBENCH_BENCHMARK.md#a4-stock-vllm-control)); stop it with `SIGTERM` to its PID and wait for its exit ([SWEBENCH_BENCHMARK.md B.5](../SWEBENCH_BENCHMARK.md#b5-slurm-job-script) has the whole sequence in one job script). The `serve.py` dry run, `--start`, READY file and cleanup-receipt checks apply to selective releases only. The stock servers that ran were started in the [runtime environment](../install.md#3-runtime-environment) plus the two variables that `serve.py` sets for this runtime and `VLLM_NO_USAGE_STATS=1`; `STOCK_CMD` is the command that the lifecycle starts. Next to the earlier builds' servers, at 32 GiB and 131,072 tokens:

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

In earlier GPU runs at this release's Qwen setting, 262,144 tokens and 25 GiB, the same command with `--max-model-len 262144` and `--kv-cache-memory-bytes 26843545600` (vLLM logged these values among its non-default arguments; `--tensor-parallel-size 1` and `--prefix-caching-hash-algo sha256` are its defaults):

```sh
STOCK_CMD=("$VENV/bin/python" -I -B -m vllm.entrypoints.openai.api_server
           --model "$MODELS/qwen3.6-27b" --served-model-name selective-prefix --host 127.0.0.1 --port 8000
           --distributed-executor-backend uni --tensor-parallel-size 1
           --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 16384
           --kv-cache-memory-bytes 26843545600 --seed 42
           --enable-prefix-caching --prefix-caching-hash-algo sha256 --language-model-only
           --mamba-cache-mode align --gdn-prefill-backend triton)
```

A Gemma 4 stock-control server recipe is not included; this page does not cover that path.

Stock vLLM takes sampling from the request. The agents of the stock runs sent `temperature` 0.0, `max_tokens` 4096, `stop` `["</mswea_bash_command>"]` with `include_stop_str_in_output: true`, and `chat_template_kwargs: {"enable_thinking": false}`; `run_agents.sh` sends the last two with `STOCK=1` ([Agents](#agents)). Stock vLLM has no `/v1/sessions/close` route: the agent pool's close requests get HTTP 404, which `run_agents.sh` accepts as the close of a finished agent with `STOCK=1` only (for this runtime's servers the close must answer 200).

## Agents

The glue under `examples/swebench/`:

- `build_tasks.py`: the task file (`--pool` for the session pool) from the dataset.
- `model_override.yaml`, `render_override.py`: the Mini-SWE-Agent model configuration (port, reply bound, step limit).
- `run_agents.sh`: the agents on the agent host, with the loopback relay `relay.py`. With `MAX_RUNNING` (the benchmark's session pool) a pool of that many agents works through the instances in list order, one runner per instance; without it, one runner with one worker thread per instance. It runs in a process group of its own and writes the process group's id to `agents.pgid` in its output directory ([Stopping the agents](#stopping-the-agents)).
- `evaluate.sh`: the SWE-bench evaluation of the predictions, on the pinned evaluation dataset ([Evaluation](#evaluation)).
- `verified-32.txt` and `images.py`: the 32 instances in run order, each with its image pinned by digest (`<instance_id> <image>@sha256:<digest>`), and the check that `run_agents.sh` and `evaluate.sh` run first: each instance's local `<image>:latest` must be its pinned image, or the script stops and prints the `docker pull` and `docker tag` commands that place it. `IMAGE_PINS` names another pins file, for other instances: `verified-150.txt` pins the first 150 instances in dataset order (the README's 24-session configuration) the same way; its first 32 lines are `verified-32.txt`, and the other digests are those of the agent host's images on 2026-11-17.

`run_agents.sh` uses the DRAM-KV prototype's SWE-bench runner and prompt by default (`examples/swebench/upstream/`, included; `SHA256SUMS` there pins them). Its settings: `MAX_RUNNING` (the session pool), `MAX_TOKENS` (1536 or 4096, the server's `--max-tokens`), `STEP_LIMIT` (1-256) and `STOCK`: `1` when the server is the stock vLLM control, so that the model override also sends the stop marker (kept in the output) and thinking off, the two lines the stock control ran with; `0`, the default, for this runtime's servers, whose model override is then unchanged. Any other value is refused before anything starts. An agent of the pool succeeds when its runner exits 0 and its session close answers HTTP 200, or, with `STOCK=1`, HTTP 404 (stock vLLM has no close route). `HF_DATASETS_CACHE` (required) is the agents' datasets cache of [Tasks](#tasks). `REPO_ENV=testbed` (optional; unset by default) runs the agents' commands with the repository's environment active (the model override's interpreter `["bash", "-lc"]`) and uses `swebench_prompt_testbed.yaml` next to the script, the PoC prompt with the Django test command for Django images (pinned in `run_agents.sh`; [docs/known-limits.md](../known-limits.md) 5.6); with `serve_gh200.sh`, set it in the job's environment. Results with it are not comparable with runs without it.

**Client timeout and retries.** The template has no client timeout or retry lines, and `run_agents.sh` adds none. The validated runs (for example jobs 3428631, 3429193, 3461703 and 3507294) had `timeout: 3600`, `max_retries: 0` and `num_retries: 0` in their model override and `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1` and `REQUEST_TIMEOUT=3600` in the agents' environment: on this runtime a client retry of a request still being served, a timeout or a non-2xx status ends the server's run ([SWEBENCH_BENCHMARK.md A.7](../SWEBENCH_BENCHMARK.md#a7-limits)). To run as they did, add the three lines to a copy of `examples/swebench` (`/abs/path/on/agent/swebench-client` below: [SWEBENCH_BENCHMARK.md B.2](../SWEBENCH_BENCHMARK.md#b2-task-list-and-agent-files)), use that copy's `run_agents.sh`, and pass the two variables, as the commands on this page do. With `STOCK=1` the rendered file then has the stock control's five lines.

For the stock control, on the agent host, with `AGENT_SOCKET` and `AGENT_OUT` as in [Run one policy](#run-one-policy) and `IDS` as in [Tasks](#tasks):

```sh
STOCK=1 MAX_RUNNING=16 MAX_TOKENS=4096 STEP_LIMIT=120 MINI_PYTHON=/abs/path/on/agent/mini-env/bin/python \
  MSWEA_CONFIG_DIR=/abs/path/on/agent/mswea-config HF_DATASETS_CACHE=/abs/path/on/agent/swebench-datasets \
  MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1 REQUEST_TIMEOUT=3600 \
  /abs/path/on/agent/swebench-client/run_agents.sh "$AGENT_SOCKET" 20000 "$AGENT_OUT" $IDS
```

### The repository's environment (REPO_ENV=testbed)

`REPO_ENV=testbed` ([docs/known-limits.md](../known-limits.md) 5.6) runs the agents' commands with the repository's `testbed` conda environment active and gives each repository's test command in the prompt. It works on any agent host prepared as in [Agent host setup](#agent-host-setup) (Docker, Mini-SWE-Agent 2.4.2 in `MINI_PYTHON`, the datasets cache, the run's instance images pinned by digest: `IMAGE_PINS=.../verified-150.txt` for the first 150 instances). To use it in an existing copy of `examples/swebench` (for example one with the client lines of this section):

1. Copy three files of this repository's `examples/swebench/` into the copy, replacing the copy's own: `run_agents.sh`, `render_override.py` and the new `swebench_prompt_testbed.yaml`. Keep the copy's `model_override.yaml` (it must still hold the line `  interpreter: ["bash", "-c"]` exactly once; the renderer refuses it otherwise), `upstream/` and the image pins.
2. Add `REPO_ENV=testbed` to `run_agents.sh`'s environment, next to `MAX_RUNNING`, `MAX_TOKENS`, `STEP_LIMIT` and `STOCK` (with `serve_gh200.sh` or a job script: in the job's environment or in `AGENT_ENV`). Nothing else changes: without it the three files run exactly as before.
3. What changes in the run: the rendered `model_override.yaml` in `AGENT_OUT` has `interpreter: ["bash", "-lc"]`, and each runner gets `-c <copy>/swebench_prompt_testbed.yaml` in place of the PoC prompt. The prompt's test section depends on the image: Django images get the Django section (`./tests/runtests.py --settings=test_sqlite --parallel 1 <label>`), every other image the PoC's Astropy text byte for byte. `run_agents.sh` checks the prompt's SHA-256 (`1e4ec193b34a92a3b56e33d05ebb9fc70591a194f8d888d6a7392aae4eefaf32`) before anything starts and refuses any other `REPO_ENV` value.

Checks on the agent host before the real run. The files: the prompt's SHA-256 above, and the rendered interpreter `  interpreter: ["bash", "-lc"]`:

```sh
cd /abs/path/on/agent/swebench-client
sha256sum swebench_prompt_testbed.yaml
python3 -I -B render_override.py model_override.yaml 20000 4096 120 0 testbed | grep interpreter
```

The prompt as Mini-SWE-Agent renders it for a Django image (Jinja2 with the image name; prints the Django section, `cd /testbed && ./tests/runtests.py ...`; with an Astropy image, the PoC's Astropy section):

```sh
"$MINI_PYTHON" -c 'import sys, yaml, jinja2
t = yaml.safe_load(open(sys.argv[1]))["agent"]["instance_template"]
print(jinja2.Template(t, undefined=jinja2.StrictUndefined).render(task="T", step_limit=120, image=sys.argv[2]))' \
  swebench_prompt_testbed.yaml docker.io/swebench/sweb.eval.x86_64.django_1776_django-11099:latest \
  | grep -A 8 '## Running Tests'
```

One container per repository, started as Mini-SWE-Agent starts them (`docker run -d -w /testbed`, then `docker exec -w /testbed ... bash -lc`): with `bash -lc`, `which python` prints `/opt/miniconda3/envs/testbed/bin/python`, `pwd` prints `/testbed`, the Django test prints `OK` and the Astropy one `passed`; with `bash -c`, `which python` prints `/opt/miniconda3/bin/python` (the base interpreter, where the repository does not import):

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

After the run (or its first instances):

```sh
grep interpreter "$AGENT_OUT/model_override.yaml"
grep -l 'Running Tests (Django-specific)' "$AGENT_OUT"/runs/django__django-11099/*/*.traj.json
```

The first line prints `  interpreter: ["bash", "-lc"]`, the second the Django instance's trajectory. Results with `REPO_ENV=testbed` are not comparable with runs without it: use it for every configuration of a comparison.

## Agent host setup

The agents and the evaluation run on an x86_64 Linux host with Docker (the agent host; its requirements: [agent host](../SWEBENCH_BENCHMARK.md#b1-agent-host)). The server runs on the GPU node, and the agents reach it through a reverse SSH tunnel that the GPU node opens to the agent host ([docs/running.md section 6.4](../running.md#64-reaching-the-server-from-another-host)). The agent host needs:

- Docker, running and usable by the user who runs the scripts (no interactive `sudo`), with the 32 instance images at their pinned digests (below).
- The dataset in a datasets cache of its own on the agent host (`HF_DATASETS_CACHE`, a new directory reserved for the pinned revision: [Tasks](#tasks)): `run_agents.sh` requires it and runs Mini-SWE-Agent offline (`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`) with it; Mini-SWE-Agent's `--subset verified --split test` loads `princeton-nlp/SWE-Bench_Verified` without a revision, offline from whatever that cache holds, and `build_tasks.py` in [Tasks](#tasks) fills it at the pinned revision. The server refuses a task message without the problem statement of its frozen task, so an agent cannot run on a different statement. `evaluate.sh` takes `SWE-bench/SWE-bench_Verified` at its own pinned revision ([Evaluation](#evaluation)).
- Two Python environments: `$MINI` (Mini-SWE-Agent 2.4.2, litellm, datasets; `run_agents.sh` checks the version) and `$EVAL` (the SWE-bench evaluator at commit `02e7a74`, version 5.0.2, and datasets), made with `uv` (below).
- A copy of this repository on the agent host: `run_agents.sh` uses `render_override.py`, `model_override.yaml`, `relay.py`, `images.py`, `verified-32.txt` and `upstream/` from its own directory, and `evaluate.sh` the two image files. `AGENT_RUN` ([Run one policy](#run-one-policy)) is the path of `run_agents.sh` in the copy of `examples/swebench` with the client lines ([Agents](#agents)).
- `bash`, `python3`, `curl`, `ss`, `sha256sum`, `setsid`, `docker` and `git` (uv installs the evaluator from its git URL) on `PATH`.
- An empty directory for `MSWEA_CONFIG_DIR` (Mini-SWE-Agent's global config directory).
- For the tunnel: the GPU node can open outbound SSH to the agent host with a key (`TUNNEL_KEY`) and a known-hosts file that pins the agent host's key (`KNOWN_HOSTS`); `AGENT_SOCKET` is a new socket path in a 0700 directory on the agent host, and `AGENT_OUT` a new directory whose parent exists.

On Ubuntu 24.04 (the tested agent host), Docker from Ubuntu's archive, the tools, Docker Compose v2 (Terminal-Bench's Harbor runs every task with `docker compose`; SWE-bench does not need it), Docker for your user (log in again afterwards), and uv from its vendor's installer. The version pins below are the release's; the runs' evidence does not record the agent host's Docker, Compose or uv versions. Drop `=<version>` when the archive no longer has it:

```sh
sudo apt-get update
sudo apt-get install -y docker.io=29.1.3-0ubuntu3~24.04.2 curl iproute2 python3 coreutils util-linux git
sudo apt-get install -y docker-compose-v2=2.40.3+ds1-0ubuntu1~24.04.1
sudo usermod -aG docker "$USER"
curl -LsSf https://astral.sh/uv/0.12.5/install.sh | sh
```

After you log in again, `docker version` must show a `Server` section without `sudo`, and `docker compose version` must print a Docker Compose v2 version there. The instance images, by digest, then tagged as the agents and the evaluator name them (`docker.io/swebench/sweb.eval.x86_64.<instance>:latest`), from this repository on the agent host:

```sh
while read -r instance image; do
  docker pull "$image"
  docker tag "$image" "${image%@*}:latest"
done < examples/swebench/verified-32.txt
```

The two environments, on the agent host (`uv` is `~/.local/bin/uv`):

```sh
MINI=/abs/path/to/mini-env
EVAL=/abs/path/to/eval-env
uv venv --python 3.12 "$MINI"
uv pip install --python "$MINI/bin/python" mini-swe-agent==2.4.2 litellm==1.103.1 datasets==5.0.1
uv venv --python 3.12 "$EVAL"
uv pip install --python "$EVAL/bin/python" \
  "swebench @ git+https://github.com/SWE-bench/SWE-bench@02e7a74ffd0b707aab73d203fe87bdc7c76afc8e" datasets==5.0.1
```

The evaluator is SWE-bench's harness at commit `02e7a74ffd0b707aab73d203fe87bdc7c76afc8e` (`swebench` version `5.0.2`), the one that evaluated the validated runs: their evaluator environment is named after this commit. The git URL above installs the same commit. Its other dependencies are resolved by uv at install time. Do not install `swebench==5.0.2` from PyPI instead: `evaluate.sh` passes the dataset as a local Parquet file, which this commit's `load_swebench_dataset` reads, and the PyPI release was not checked for that (`evaluate.sh` refuses an evaluator without it).

## Tasks

On the agent host, from this repository (`$MINI` has `datasets`; the first load needs network and fills the cache, so the `unset` line undoes any offline switches of the shell). `HF_DATASETS_CACHE` is the agents' datasets cache, a new directory: `build_tasks.py` loads the dataset at its pinned revision into it, and `run_agents.sh` gives the same directory to Mini-SWE-Agent, which loads the dataset offline without a revision and so takes what this cache holds. This cache is reserved for the pinned SWE-bench revision; do not populate it with another revision between preparation and execution. Pass the same directory to `run_agents.sh` (`HF_DATASETS_CACHE`, in `AGENT_ENV` of [Run one policy](#run-one-policy)).

```sh
export HF_DATASETS_CACHE=/abs/path/on/agent/swebench-datasets   # new: the "mkdir" line creates it
mkdir "$HF_DATASETS_CACHE"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
IDS=$(cut -d ' ' -f 1 examples/swebench/verified-32.txt)
"$MINI/bin/python" examples/swebench/build_tasks.py --output-dir /abs/path/to/tasks --pool $IDS
```

`IDS` is the 32 instance ids in run order. `build_tasks.py` loads the dataset at its pinned revision (`REVISION`, `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`), checks each problem statement against its SHA-256 pin (all 32 are pinned), writes `tasks-pool.json` (the 32 tasks in this order, the order in which the agents start) and one `task-<instance_id>.json` per instance, never overwrites a file, and prints a JSON report. Copy `tasks-pool.json` to the GPU side: the server reads it (`TASKS`), the agent host gets only the instance ids.

## Run one policy

One fresh server per policy, in its own GPU job. The script below is the sequence of `examples/slurm/serve_gh200.sh` with `MODE=swebench` (server, READY, tunnel, agents, stop), with the benchmark servers' flags: `serve_gh200.sh` itself cannot pass `--active-token-budget`, `--goal-rewriter-batch` or `--admission-overlap`. Run it on the GPU node after the [runtime environment](../install.md#3-runtime-environment), with `REPO`, `RELEASE`, `RUNS`, `MODELS`, `TASKS` (the copied `tasks-pool.json`) and `LABEL` (new per attempt) set. The agents get `MAX_RUNNING=16` (the server's `--max-running-sessions`), `MAX_TOKENS=4096` (the server's `--max-tokens`; `run_agents.sh` defaults to 1536) and `STEP_LIMIT=120` (`run_agents.sh` defaults to 90; at most 256, because the server serves at most 256 requests per session and refuses a later one, [the XML protocol](../running.md#61-mini-swe-agent-xml-protocol)), and `AGENT_ENV` carries `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1` and `REQUEST_TIMEOUT=3600` for the copy with the client lines ([Agents](#agents)).

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

`server_checks.py cleanup` prints `CLEAN_SHUTDOWN`. The last line of `$OUT.agents.log` is `run_agents.sh`'s summary (`runner_exit`, `seconds`, `containers_left`, `preds`). For the Gemma profile, `SHARED` and the policy's flags are those of "Server commands" > Gemma 4 31B IT and `SERVER_RELEASE="$RELEASE-gemma4"`: the one variable gives both `serve.py --release-dir` and `server_checks.py ready` the server's release. For the stock control, the stock lifecycle of [Server flags](#server-flags) replaces the dry run, the `--start` line, the READY loop and `server_checks.py cleanup`: `"${STOCK_CMD[@]}" > "$OUT.server.log" 2>&1 &` with a `STOCK_CMD` of that section, a READY loop on `GET /health` and `GET /v1/models` ([SWEBENCH_BENCHMARK.md A.4](../SWEBENCH_BENCHMARK.md#a4-stock-vllm-control)), and the same `SIGTERM` and `wait` at the end, without the exit-status and cleanup checks ([SWEBENCH_BENCHMARK.md B.5](../SWEBENCH_BENCHMARK.md#b5-slurm-job-script) has this sequence); the agents get `STOCK=1`. If the SSH connection drops, `run_agents.sh` and its runners keep running on the agent host: stop them there ([Stopping the agents](#stopping-the-agents)). When the server already answers at `http://127.0.0.1:<port>/v1` on the agent host itself (for example server and agents on one x86_64 machine with a GPU), pass `-` instead of the socket and skip the tunnel: `run_agents.sh - <port> <output dir> <instance id>...`; that form was not part of the validated runs.

### Stopping the agents

`run_agents.sh` runs in a process group of its own: its relay, its runners and their `docker` clients, nothing else. It writes the process group's id to `agents.pgid` in its output directory (`AGENT_OUT`) and the instances' pinned images to `images.txt` there. On the agent host, with `RUN_OUT` that run's `AGENT_OUT`, the first line stops that group with `SIGTERM`; the loop then lists the containers still running from that run's instance images, which Mini-SWE-Agent stops asynchronously; stop any it still lists with `docker stop <id>` (only those ids). Never stop the agents with a broad `pkill` or by stopping unrelated containers.

```sh
RUN_OUT=/abs/path/on/agent/runs/observation_query-1
kill -TERM -- "-$(cat "$RUN_OUT/agents.pgid")"
while read -r instance image; do docker ps -q --filter "ancestor=$image"; done < "$RUN_OUT/images.txt"
```

## Evaluation

The evaluation dataset is `SWE-bench/SWE-bench_Verified` at revision `78f471bf655a3137b2e8a75af1501690ec009ec3` (pinned in `evaluate.sh` with the SHA-256 of its test split file): the evaluator (SWE-bench at `02e7a74`, version 5.0.2) reads each instance's image name and evaluation script from that dataset. `evaluate.sh` fetches the split file at that revision (or takes it from the Hugging Face cache: network on the first evaluation), checks its SHA-256, and passes it to the evaluator as a local file; it also checks the instances' images, as `run_agents.sh` does. On the agent host, from this repository, after each run (the last argument is that run's `AGENT_OUT`):

```sh
EVAL_PYTHON="$EVAL/bin/python" examples/swebench/evaluate.sh observation_query-1 observation_query \
  /abs/path/on/agent/eval/observation_query-1 /abs/path/on/agent/runs/observation_query-1
```

The arguments are a run id, a label, a new evaluation directory (an absolute path whose parent exists; `evaluate.sh` writes the dataset file `dataset.parquet` and the image list `images.txt` there) and one or more `run_agents.sh` output directories. `evaluate.sh` prints one JSON line per instance (`instance_id`, `resolved`, `empty_patch`, `error`, `patch_chars`) and a last line `{"evaluator_exit": ..., "instances": 32, "report": ..., "report_missing": false, "resolved": N, "run_id": ...}`: the resolve rate is `resolved` / `instances` (an instance without a prediction counts, evaluated with an empty patch).

## A small group with the Slurm script

`examples/slurm/serve_gh200.sh` with `MODE=swebench` runs one server and one small agent group in a Slurm job, with the script's own server flags (it cannot pass the benchmark servers' `--active-token-budget`, `--goal-rewriter-batch` or `--admission-overlap`). A group of three tasks, built on the agent host with `HF_DATASETS_CACHE` of [Tasks](#tasks) set (one `tasks-g0.json`, at most five tasks per group); copy it to the GPU side:

```sh
"$MINI/bin/python" examples/swebench/build_tasks.py --output-dir /abs/path/to/tasks --max-sessions 5 \
  astropy__astropy-12907 astropy__astropy-13236 astropy__astropy-14096
```

Submitted with `sbatch`, with `ACCOUNT`, `PARTITION` and `MAIL` of [the Slurm job](../running.md#45-gpu-job-slurm). `RUN_ROOT` must exist and be short (`TMPDIR` lives under it); the 0700 directory of `AGENT_SOCKET` and the parent of `AGENT_OUT` must exist on the agent host. The agents use `run_agents.sh`'s defaults here (`MAX_TOKENS` 1536, the server's default reply bound, and `STEP_LIMIT` 90) and the checkout's template without the client lines of [Agents](#agents): the model override of the pilots. `--mem=113560M` is one GH200's memory share of a node, as the validated SWE-bench jobs requested ([SWEBENCH_BENCHMARK.md A.3, Host memory](../SWEBENCH_BENCHMARK.md#a3-this-runtime-server-flags-and-vllm-arguments)).

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

Then, on the agent host, the evaluation of that group (its `AGENT_OUT`), as in [Evaluation](#evaluation):

```sh
EVAL_PYTHON="$EVAL/bin/python" examples/swebench/evaluate.sh my-run-goal_query goal_query \
  /abs/path/on/agent/eval/goal_query /abs/path/on/agent/runs/goal_query-g0
```

## Server commands

The dry runs of this benchmark's servers, one per policy (add `--start` to serve). `TASKS` is the task file, `LABEL` a new label per attempt; the release repository's `tests/test_docs_commands.py` dry-runs these commands.

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
