# Terminal-Bench 2.0 with Harbor's Terminus 2

English | [简体中文](terminal-bench.zh-CN.md)

> Synced from the release repository (release-repo `5b47394`, its benchmark page) in integrate-1107: the client in `examples/terminalbench/` and this page are the release's, with the release-only links pointed at this repository's README and benchmark docs. The configuration of record, the runs and their job ids are in [TERMINALBENCH_BENCHMARK.md](../TERMINALBENCH_BENCHMARK.md). The "Server commands" below are the release's: the Qwen servers at `--kv-cache-gib 25 --max-model-len 262144` and `--dram-arena-gib 70`, the configuration of job 3461661 ([docs/reference.md section 4.9](../reference.md#49-context-limit-and-reply-bound)), which `make_job.py` gives Terminus 2 as its context; A.3 of TERMINALBENCH_BENCHMARK.md has the same Qwen servers, and its earlier dry runs ran at 32 GiB and 65,536 tokens. Since 2026-11-08 the client and this page also give Gemma 4 31B IT its 262,144-token configuration (TERMINALBENCH_BENCHMARK.md A.14: 25 GiB, K = 20, 4,096-token forwards, a 72-GiB arena, no `full`), not yet in the release repository: the Gemma servers under "Server commands" are A.14's, followed by the Gemma stock command, and `make_job.py --context 65536` writes the job of the earlier dry runs at 65,536 tokens; the release's copy of the client (the export's `release-src`) has the same edit.

## Scope

- Terminal-Bench 2.0, all tasks, Harbor `0.21.0` with its Terminus 2 agent unchanged, 16 concurrent trials (`-n 16`).
- One fresh server per policy and a stock vLLM 0.24.0 control: for Qwen3.6-27B `full`, `observation_query`, `random_fill` and `goal_query`; for the Gemma 4 profile `observation_query`, `random_fill` and `goal_query` (`full` does not fit at 262,144 tokens).
- Protocol: `--protocol chat-text` ([docs/running.md section 6.7](../running.md#67-chat-text-protocol-terminal-bench-20-with-terminus-2)), thinking off in the commands below (the configuration of record; the current runs: [docs/status.md](../status.md#current-configuration)).
- Hosts: Harbor, the task containers and their verifiers run on an x86_64 Linux Docker host, the agent host (Terminal-Bench 2.0's task images are x86_64; its requirements: [agent host](../TERMINALBENCH_BENCHMARK.md#b1-agent-host)); the server runs on the GPU node (the tested one is an aarch64 GH200), reached through a reverse SSH tunnel and a loopback relay ([docs/running.md section 6.4](../running.md#64-reaching-the-server-from-another-host)). Only this two-host setup was validated. Harbor can also run beside the server on one x86_64 host with the GPU ([Run one policy](#run-one-policy)); the task containers' memory and CPUs then come on top of the server's. One aarch64 machine for both is not supported.

**GPU runs:** the trial runs on 16 of the 89 tasks (below) at `-n 16`, with each job's runtime commit, model, context and verdict, are in [docs/status.md](../status.md); the benchmark on all tasks has not run. The release repository's `5b47394` is the export of runtime `5466fc2`, which is `cf2b28d` plus serve.py's speed defaults (each with an off switch) and test fixes.

The 16 tasks of the trial runs, each with a 900-s agent timeout, chosen among the tasks with the largest prompts in earlier Terminus 2 runs:

```text
build-cython-ext build-pmars configure-git-webserver crack-7z-hash db-wal-recovery extract-elf
fix-code-vulnerability gcode-to-text git-multibranch make-doom-for-mips password-recovery pytorch-model-cli
raman-fitting sanitize-git-repo tune-mjcf vulnerable-secret
```

`examples/terminalbench/tasks.json` names them `trial16`. `smoke32` adds 16 tasks with a 900-s agent timeout (not run on the GPU), `all` is the 89 tasks. The file pins the dataset, Harbor's package `terminal-bench/terminal-bench-2` at `sha256:c6fc2e2382c1dbae99b2d5ecd2f4f4a60c3c01e0d84642d69b4afd92e99d078b`, and per task the package ref and task checksum that Harbor recorded for it, its agent and verifier timeouts, its image and its resource limits. Each task package names its image by tag (for example `alexgshaw/build-pmars:20251031`), not by digest, and Harbor starts it by that tag: Docker Compose pulls an image only when it is absent, so a new pull resolves the tag at that time. For the 32 tasks of `smoke32` (`trial16` included), `tasks.json` also records the digest that the tested agent host had for the image (`docker_image_digest`, from `docker image inspect`; the images were pulled before the trial runs); the other 57 tasks have no recorded digest. `run_harbor.sh` refuses a run when an image of the job's tasks that is already present has another digest ([Agent host setup](#agent-host-setup)); an absent image is pulled by Harbor by its tag and is not checked.

## Server

The servers' commands are under "Server commands" below. Main settings:

- `--max-running-sessions 20` for Harbor's `-n 16`, for both profiles (the Gemma trial runs at 65,536 tokens ran 17): Terminus 2 never closes its sessions, so a finished trial's session stays open until it is evicted, and Terminus's summarization sub-calls open sessions of their own. The 4 spare sessions hold these, so a new session does not have to evict a running trial.
- `--page-tokens 16 --kv-layout decoupled`, `--active-token-budget 8192` for the query policies and `random_fill`, `--max-tokens 4096`, `--connector-pool-gib 1`, `--dram-arena-gib` for the pinned page arena, `--release-idle-pages`.
- `--kv-cache-gib 25 --max-model-len 262144` for both profiles; Qwen with `--dram-arena-gib 70`, Gemma with `--dram-arena-gib 72` and `--max-num-batched-tokens 4096` ([TERMINALBENCH_BENCHMARK.md A.14](../TERMINALBENCH_BENCHMARK.md#a14-gemma-4-31b-it)). The Gemma trial runs before job 3560498 (a 16-task dry run at these settings) ran at 65,536 tokens.
- `goal_query` adds `--goal-rewriter-model` (or `--no-goal-rewriter`); its inputs for Terminus 2 are provisional ([docs/running.md section 6.7](../running.md#67-chat-text-protocol-terminal-bench-20-with-terminus-2)).

## Client

`examples/terminalbench/`:

- `job-qwen3.6-27b.json`, `job-gemma4-31b-it.json`: the Harbor job of each model profile, with the settings of the trial runs ([Job settings](#job-settings)).
- `tasks.json`: the dataset pin and the task sets.
- `make_job.py`: writes one job file from a template; it fills in the port, the task set, the jobs directory and the job name (and, with `--context 65536`, the 65,536-token context of the earlier trial runs) and refuses anything else, also every key that is not a field of Harbor 0.21.0's models (Harbor ignores unknown keys without a warning).
- `run_harbor.sh`: one job on the agent host, in this order: the Harbor install against its pins, the job file, the image check, the relay, the server check, `harbor run`, the session close, the container check, the report.
- `check_images.py`: the image check: every task image of the job that is already present must have the digest `tasks.json` records (`check`); `list` prints the recorded digests for `docker pull`.
- `close_sessions.py`: `check`, before the run (the server serves this job and has no session yet), and `close`, after it (the sessions Terminus 2 left open).
- `report.py`: the task count, the exceptions and the mean reward of a finished job, from Harbor's saved job and trial results.
- `requirements.lock`, `upstream/harbor-wheels.SHA256SUMS` and `upstream/terminus-2.SHA256SUMS`: the pins of the Harbor install (the hash lock of every package; the SHA-256 of the Harbor, litellm and openai wheels, which the lock repeats; Terminus 2's four files).

These files have CPU tests only (`tests/test_tb_client.py`); they have not run in a GPU job. The trial runs used the job harness's own Harbor script with the same job settings and order, except the image check (`check_images.py`), which was added after the trial runs (`run_harbor.sh`'s header).

### Job settings

The job of each run, as `make_job.py` writes it (Harbor 0.21.0's `JobConfig`; the agent's `kwargs` go to Terminus 2's constructor and from there to litellm):

| Setting | Value |
| --- | --- |
| agent | `terminus-2` (Terminus 2 2.0.0 with its defaults: the JSON parser, summarization 8,000 tokens before the context limit) |
| `model_name` | `hosted_vllm/selective-prefix`, the served model name |
| `api_base` | `http://127.0.0.1:<port>/v1`: the relay on the agent host, or the server on the same host |
| `temperature` | `0.0` (the server decodes greedily in any case) |
| `llm_kwargs` | `max_tokens` 4096; `timeout` 3600, so that a request waiting in the server's admission is not cancelled and sent again; `max_retries` 0 and `num_retries` 0 (no retries by litellm or the OpenAI client; Terminus 2's own retries remain) |
| `model_info` | `max_input_tokens` 262144, the servers' context (65536 with `make_job.py --context 65536`, the earlier trial runs' context); `max_output_tokens` 4096; `input_cost_per_token` and `output_cost_per_token` 0. Harbor requires these four fields for a `hosted_vllm` model |
| `n_concurrent_trials` | 16 (`-n 16`) |
| `n_attempts`, `retry` | one attempt, `max_retries` 0 |
| `timeout_multiplier` | 1.0: each task's own timeouts. Every task of `trial16` and `smoke32` has a 900-s agent timeout, which `make_job.py` checks |
| `environment` | `docker`, `delete` true (each trial's containers are removed when it ends), `force_build` false |
| `verifier` | `disable` false: Harbor runs each task's tests after its agent, also after an agent timeout |
| `datasets` | `terminal-bench/terminal-bench-2` at its pinned `ref`; `task_names`: the set's tasks as `terminal-bench/<task>` |
| `quiet` | true |
| stock control | `make_job.py --stock` adds `llm_call_kwargs` `{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}`: thinking off with every call, because a stock server keeps vLLM's default template switches (this runtime's servers render thinking off themselves) |

The field names are those of Harbor's source at tag `v0.21.0` (`src/harbor/models/job/config.py`, `src/harbor/models/trial/config.py`, `src/harbor/agents/terminus_2/terminus_2.py`; the `hosted_vllm` fields: `src/harbor/llms/utils.py`). The job file of the Qwen trial runs, printed (`--out -`) instead of written:

```sh
python3 -I -B examples/terminalbench/make_job.py --profile qwen3.6-27b --tasks trial16 --port 20000 \
  --jobs-dir /abs/path/on/agent/runs/tb-example/jobs --job-name tb-example --out -
```

## Agent host setup

The agent host needs:

- Linux x86_64 with Docker, running and usable by the user who runs the scripts without an interactive `sudo` (the SWE-bench page installs it on Ubuntu 24.04, the tested system: [agent host setup](swe-bench.md#agent-host-setup)), and Docker Compose v2: Harbor runs every task with `docker compose` (the tested host: Ubuntu's `docker-compose-v2` package `2.40.3+ds1-0ubuntu1~24.04.1`, which the SWE-bench page's install block installs; `docker compose version` prints `Docker Compose version 2.40.3+ds1-0ubuntu1~24.04.1`).
- CPython 3.12 as `python3.12` (Harbor 0.21.0 requires Python 3.12 or later; the trial runs used 3.12.3), and `python3`, `bash`, `curl`, `ss`, `sha256sum` and `setsid` on `PATH`.
- Network access when Harbor first downloads the task packages and Docker pulls their images, and disk space for the images.
- Room for the containers: at `-n 16`, each `trial16` task is limited to 1 CPU and 2,048 MB (`memory_mb` in `tasks.json`), 16 CPUs and 32,768 MB of limits in all (`smoke32`: up to 4,096 MB per task; `all`: up to 4 CPUs and 8,192 MB). The tested agent host has 16 CPUs and 33 GB of memory.
- A copy of this repository: `run_harbor.sh` uses the files next to it and `examples/swebench/relay.py`. `HARBOR_RUN` ([Run one policy](#run-one-policy)) is the path of `examples/terminalbench/run_harbor.sh` in that copy.
- For the tunnel: as for SWE-bench ([agent host setup](swe-bench.md#agent-host-setup)), the GPU node can open outbound SSH to the agent host with a key (`TUNNEL_KEY`) and a known-hosts file that pins the agent host's key (`KNOWN_HOSTS`); `AGENT_SOCKET` is a new socket path in a 0700 directory on the agent host, and `AGENT_OUT` a new directory whose parent exists.

The Harbor environment, on the agent host, from this repository. `examples/terminalbench/requirements.lock` pins Harbor 0.21.0, litellm 1.97.0 and openai 2.54.0 (the wheels of `upstream/harbor-wheels.SHA256SUMS`) and the 80 packages they require, at the versions of the agent host's Harbor environment of the trial runs, each with the SHA-256 of the PyPI wheel that environment had installed; with `--require-hashes` pip installs exactly these files or stops. Then Terminus 2's four files in the installed package are checked against `upstream/terminus-2.SHA256SUMS` (their entries in the Harbor 0.21.0 wheel's `RECORD`):

```sh
TB_VENV=/abs/path/on/agent/harbor-env       # new: the "python3.12 -m venv" line creates it
python3.12 -m venv "$TB_VENV"
"$TB_VENV/bin/python" -m pip install --only-binary=:all: --require-hashes -r examples/terminalbench/requirements.lock
SITE=$("$TB_VENV/bin/python" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
(cd "$SITE" && sha256sum --check) < examples/terminalbench/upstream/terminus-2.SHA256SUMS
"$TB_VENV/bin/harbor" --version
```

`sha256sum --check` prints `OK` for each of the four files, and `harbor --version` prints `0.21.0`. The lock records the agent host's Harbor environment of the trial runs, read from its installed packages after the runs; that environment was made with uv on the host's CPython 3.12.3 and also held 15 packages that these three do not require (datasets and mini-swe-agent with their dependencies), which the lock leaves out. A fresh install from the lock was not run. `run_harbor.sh` checks the three versions and Terminus 2's four files again before every run.

The task images: `run_harbor.sh` runs `check_images.py check --tasks <task set>` before it starts the relay and writes the result to `images.json`: `IMAGES_OK` with the checked tasks, the tasks whose image is absent (Harbor pulls it by its tag) and the tasks without a recorded digest; `IMAGES_REFUSED` (an image that is present with another digest, or a `docker` error) stops the run. To have every image of a task set checked, pull each one by its recorded digest and give it the tag that Harbor uses before the run (not part of the trial runs, whose images were pulled by tag):

```sh
python3 -I -B examples/terminalbench/check_images.py list --tasks trial16 | while read -r pinned tagged; do
  docker pull "$pinned" && docker tag "$pinned" "$tagged"
done
```

## Run one policy

One fresh server per run, in its own GPU job. The script below has not run end to end on a GPU job (the trial runs used the job harness's scripts; [Client](#client)). It starts the server, waits for READY, opens the tunnel, runs `run_harbor.sh` on the agent host over SSH and stops the server; run it on the GPU node after the [runtime environment](../install.md#3-runtime-environment), with `REPO`, `RELEASE`, `RUNS`, `MODELS` and `LABEL` (new per attempt) set, like the SWE-bench script ([Run one policy](swe-bench.md#run-one-policy)). On the agent host, `run_harbor.sh` checks the Harbor install, writes the job file with `make_job.py`, checks the task images (`check_images.py`), starts the relay, checks that the server serves this job (`close_sessions.py check`), runs `harbor run -c job.json -y`, closes the sessions that Terminus 2 left open (`close_sessions.py close`, through the tunnel: the server sees a loopback peer), waits until no container of the job's trials is left and runs `report.py`.

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

`server_checks.py cleanup` prints `CLEAN_SHUTDOWN`. The last line of `$OUT.harbor.log` is `run_harbor.sh`'s summary: `job`, `harbor_exit`, `close` (`SESSIONS_CLOSED`), `containers_left` (`null` when `docker ps` failed), `images` (`IMAGES_OK`), `report` (`REPORT_OK`), `planned`, `n_exceptions`, `solved` and `mean_reward`; `run_harbor.sh` exits 0 only when Harbor exited 0, every session closed, `docker ps` listed no container of the job's trials and `report.py` accepted the job. Its files are in `AGENT_OUT` on the agent host: `job.json`, `images.json`, `check.json`, `harbor.log`, Harbor's job directory `jobs/<the name of AGENT_OUT>/`, `close.json` and `report.json`. At `-n 16` the 16 trials of `trial16` run at once, each up to its 900-s agent timeout plus its setup and tests; the GPU job's time limit must leave room for that after READY.

Other runs:

- Another policy or the Gemma profile: the server command of "Server commands" in `SERVE` (Gemma: its `GSHARED` array and its flags) and its release in `SERVER_RELEASE` (Gemma: `SERVER_RELEASE="$RELEASE-gemma4"`; the one variable gives both `serve.py --release-dir` and `server_checks.py ready` the server's release), and the profile `gemma4-31b-it` as `run_harbor.sh`'s fourth argument. The profile must be the server's: `close_sessions.py check` refuses a server whose context, reply bound, session pool or model profile differs from the job's.
- Another task set: `smoke32` or `all` as the fifth argument.
- The 65,536-token context of the earlier trial runs: `TB_CONTEXT=65536` in the `env` of the SSH call that starts `run_harbor.sh`, next to `HARBOR_VENV` (`run_harbor.sh` passes it as `make_job.py --context 65536`; a variable set on the GPU node does not reach the agent host), for a server at `--max-model-len 65536` ([TERMINALBENCH_BENCHMARK.md A.14](../TERMINALBENCH_BENCHMARK.md#a14-gemma-4-31b-it) has the Gemma settings of those runs).
- A stock vLLM control: vLLM's OpenAI server, serving `selective-prefix` on `127.0.0.1:8000` with the job's context as `--max-model-len` (for Qwen, the command of [TERMINALBENCH_BENCHMARK.md A.4](../TERMINALBENCH_BENCHMARK.md#a4-stock-vllm-control) at 262,144 tokens and 25 GiB, the sizes of the SWE-bench page's stock command, [Server flags](swe-bench.md#server-flags); for Gemma, the stock command under "Server commands" > Gemma 4 31B IT, at 262,144 tokens and 25 GiB with `--max-num-batched-tokens 4096`, since vLLM's own startup check refuses 8,192 there), started and stopped with the stock lifecycle of [TERMINALBENCH_BENCHMARK.md A.4](../TERMINALBENCH_BENCHMARK.md#a4-stock-vllm-control) in place of the dry run, the `--start` line, the READY loop and `server_checks.py cleanup`: the `serve.py` dry run, `--start`, READY file and cleanup-receipt checks apply to selective releases only, and the stock control uses HTTP readiness and process shutdown. `stock` is then `run_harbor.sh`'s sixth argument; the job sends thinking off with every call, and nothing is closed (stock vLLM has no chat-text routes). Which stock controls ran (job 3499759: thinking on, Qwen, 262,144 tokens, 25 GiB; job 3560498: Gemma 4, 262,144 tokens, 25 GiB): [docs/status.md](../status.md).
- Harbor and the server on one x86_64 host with the GPU (not validated): start the server as above and, on the same host instead of the tunnel and the SSH call, run `run_harbor.sh` with `-` for the socket and the server's port; Harbor's agents then use `http://127.0.0.1:8000/v1` directly. The containers' memory and CPUs come on top of the server's.

```sh
HARBOR_VENV=/abs/path/to/harbor-env examples/terminalbench/run_harbor.sh - 8000 \
  "$RUNS/tb-observation_query-$LABEL-harbor" qwen3.6-27b trial16
```

If the SSH connection drops, `run_harbor.sh` and Harbor keep running on the agent host. Harbor runs in its own process group, whose id `run_harbor.sh` writes to `AGENT_OUT/harbor.pgid`; `kill -TERM -- -<that id>` on the agent host stops Harbor and nothing else (not part of the validated runs). The containers of the job's trials are the Docker Compose projects `<trial directory name, lowercased>__env` and `<trial directory name, lowercased>__verifier__...` (`docker ps -a --format '{{.Label "com.docker.compose.project"}}'`).

## Results

Harbor scores each trial itself: after the agent ends (or times out), it runs the task's tests in the task's environment and records their reward, 1 when the task is solved and 0 when not. Accuracy is that per-trial verdict. Harbor saves the job's `result.json` (its counts and statistics) and each trial's `result.json` (its task, reward and exception, if any) in the job directory. `report.py` reads them; `run_harbor.sh` runs it at the end, and you can run it again on the agent host with the job file:

```sh
python3 -I -B examples/terminalbench/report.py --job /abs/path/on/agent/runs/tb-observation_query-1/job.json
```

It prints one JSON line. With `"status": "REPORT_OK"` (exit 0): `planned` (the task count of the job), `trials`, `finished`, `n_exceptions` and `exceptions` (each exception type with its tasks; an agent timeout is an `AgentTimeoutError`, and its trial still has the tests' reward), `rewarded`, `solved` (reward 1), `unrewarded` (trials without a reward, which count 0), `mean_reward` (the mean reward over the planned tasks, Harbor's mean metric) and `harbor_mean` (Harbor's own mean from the job's statistics, when they carry one; it must be equal). `warnings` lists trials whose `task_checksum` differs from the one in `tasks.json`. It refuses (exit 1, `"status": "REPORT_REFUSED"`, with every reason and no `mean_reward`) a job that is not one finished trial of the pinned package for every planned task: no job result or an unfinished job; a planned task without a trial result; a trial without `result.json` or `finished_at`, or cancelled; a trial of a task that was not planned, or a second trial of a task; another task package, agent or model than the job's; statistics that do not count the trials.

## Closing sessions by hand

`run_harbor.sh` closes the sessions itself. By hand, on the GPU node (the server accepts a close from a loopback peer only), with `close_sessions.py`, which closes every open session and then checks that none is left:

```sh
python3 -I -B examples/terminalbench/close_sessions.py close --base-url http://127.0.0.1:8000
```

or with `curl`, one close per open session's `key_sha256` from `GET /v1/chat-text/info` (Terminus 2 sends no `X-Session-ID`; [docs/running.md section 6.7](../running.md#67-chat-text-protocol-terminal-bench-20-with-terminus-2)):

```sh
curl -s http://127.0.0.1:8000/v1/chat-text/info
curl -s -X POST -H 'Content-Type: application/json' -d '{"key_sha256": "<key>"}' \
  http://127.0.0.1:8000/v1/chat-text/close
```

## Server commands

The dry runs of this benchmark's servers, one per policy (add `--start` to serve). `LABEL` is a new label per attempt; the release repository's `tests/test_docs_commands.py` dry-runs these commands. The Gemma stock command at the end is not a `serve.py` dry run: it starts vLLM's own server, and the release export does not carry it (the export appends only the sh blocks with a `serve.py` line).

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

The Gemma stock control, vLLM's own OpenAI server, started and stopped with the stock lifecycle of [TERMINALBENCH_BENCHMARK.md A.4](../TERMINALBENCH_BENCHMARK.md#a4-stock-vllm-control) (its settings: [A.14](../TERMINALBENCH_BENCHMARK.md#a14-gemma-4-31b-it)):

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
