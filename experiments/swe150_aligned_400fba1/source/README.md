# Selective-prefix policies for vLLM 0.24.0

English | [简体中文](README.zh-CN.md)

Server-side prefix-page selection for multi-turn tool-using agents on vLLM 0.24.0: each server runs one of eleven [policies](#policies) that decide which sealed pages of a session's KV history are restored from CPU memory on the next turn.

> **Where things stand.** `main` is integrate-1117, merged on 2026-11-17: runtime `2fb7f62` (integrate-1116 plus the session pool's task cap 256 and the 150-instance list). Parts of it have not run on a GPU yet ([docs/known-limits.md](docs/known-limits.md) 5.7). The previous `main` (`7892db3`, runtime `4ebff97`) is the tag `main-integrate-1112`; there is no stable tag yet. What has run on the GPU, per model, benchmark and policy, with each job's verdict and whether its numbers are audited, and the current configuration: [docs/status.md](docs/status.md). What does not work yet or only with care: [docs/known-limits.md](docs/known-limits.md). How a change reaches `main` and when a number may be used: [docs/release-gates.md](docs/release-gates.md).

## Status by model and benchmark

The table that stood here is [docs/status.md](docs/status.md) since 2026-11-16 (docs-reorg-1116), extended to one row per policy; this page carries no job results.

## Stable configuration: Qwen on SWE-bench Verified

Use this configuration for SWE-bench Verified with Qwen3.6-27B. It ran on runtime `ba4d1ba` (`main`'s runtime until 2026-11-15) the first 32 SWE-bench Verified instances with thinking on: job 3576565, `SWE32_PASS_VERIFIED`, 19 of 32 resolved ([docs/status.md](docs/status.md)). The 64-task runs use it with `--async-scheduling` added (the current configuration in [docs/status.md](docs/status.md)).

- **Hardware:** one GH200 (Slurm `--gpus=1 --cpus-per-task=16 --mem=113560M`).
- **Model:** Qwen3.6-27B, text only. The Goal is built by Qwen3-0.6B in a vLLM child engine (3 GiB of KV), overlapped with the main model's phase 1.
- **Agent:** Mini-SWE-Agent 2.4.2 with the XML protocol, 16 instances at once, at most 120 steps, temperature 0.0, no retries, a 3,600 s request timeout.
- **Server** (`$TASKS` holds the instances, section [4.1](docs/running.md#41-one-server); `$GOAL_MODEL` is the Qwen3-0.6B snapshot):

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

**Current configuration** (the 64-task runs, [docs/status.md](docs/status.md#current-configuration)): the command above with `--async-scheduling` added. On SWE-bench Verified only `recent_fill` has run with it (job 3611917); `goal_query` has not yet.

- **Other policies:** the same command with another `--policy` and without the Goal options (the `--goal-rewriter-*` options and `--goal-overlap`: the launcher refuses `--goal-rewriter-engine vllm` and `--goal-overlap` for a policy without a Goal); `full` needs `--release-idle-pages` at 16 sessions in 25 GiB, and so should `akv_rule` ([docs/known-limits.md](docs/known-limits.md) 2.4). The 64-task runs of the current configuration add `--async-scheduling` ([docs/status.md](docs/status.md#current-configuration)).

## 24-session configuration: the group's 150-instance runs (2026-11-17, not yet GPU-validated)

The configuration of the group's runs on the first 150 SWE-bench Verified instances with 24 sessions at once, on `main` since 2026-11-17, that is integrate-1117: integrate-1116 with the session pool's task cap raised from 64 to 256 (section [4.4](docs/running.md#44-several-sessions-on-one-server)). **Not yet GPU-validated:** more than 16 sessions on the XML protocol has not run on a GPU ([docs/known-limits.md](docs/known-limits.md#21-more-than-16-sessions-and---gdn-inplace-crossing-are-cpu-tested-only) 2.1). It becomes the stable configuration only after the 24-session smoke and the 150-instance runs pass; until then results come from the [stable configuration](#stable-configuration-qwen-on-swe-bench-verified) above. Who runs what: the stock control, `full` and `ledger_query` run on this runtime; the two members run their own policies on their branches with integrate-1117 merged.

- **Hardware:** one GH200 (Slurm `--gpus=1 --cpus-per-task=16 --mem=113560M`).
- **Model:** Qwen3.6-27B, text only, thinking on.
- **Agent:** Mini-SWE-Agent 2.4.2 with the XML protocol, the first 150 SWE-bench Verified instances in dataset order (`examples/swebench/verified-150.txt`: their ids and image digests, given to `run_agents.sh` and `evaluate.sh` as `IMAGE_PINS`), 24 at once, at most 120 steps, temperature 0.0, no retries, a 4,400 s request timeout (the job harness's rule above 16 sessions: the admission wait at K = 24, 300 + 120 × 23 = 3,060 s, plus 1,340 s; the stock control, which has no admission gate, keeps 3,600 s).
- **Server for `full`** (`$TASKS` holds the 150 instances: `examples/swebench/build_tasks.py --pool`, section [4.1](docs/running.md#41-one-server)):

```sh
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy full --protocol mswea-xml \
  --output "$RUNS/full" --port 8000 \
  --kv-cache-gib 25 --page-tokens 16 --kv-layout decoupled \
  --max-running-sessions 24 --max-model-len 262144 --dram-arena-gib 68 \
  --enable-thinking --async-scheduling --release-idle-pages --start
```

- **Server for `ledger_query`** (the observation-ranked plug-in; the Qwen XML protocol only): the same command with `--policy ledger_query --active-token-budget 8192` (and its own `--output`). Its ledger keeps the default cap, 50 % of the fill budget, so `--ledger-cap-percent` is not given. No Goal options and no `--admission-overlap`: every observation-ranked or fill policy takes this form.
- **Goal policies** (the form the members' Goal policies take; the 24-session smoke runs `goal_query` so): the `full` command plus `--active-token-budget 8192 --admission-overlap 24` and the stable command's Goal options (`--goal-rewriter-model "$GOAL_MODEL" --goal-rewriter-engine vllm --goal-rewriter-guided-backend lm-format-enforcer --goal-rewriter-vllm-kv-gib 3 --goal-rewriter-batch-invariant --goal-overlap`).
- **Defaults these commands rely on** (the speed defaults, [docs/reference.md](docs/reference.md#415-speed-defaults) 4.15): CUDA graphs `full`, the three decode fast paths, `--deferred-completion-io`, `--language-model-only`, a 1-GiB connector pool, and with `--enable-thinking` 4,096-token replies. The stable command spells them out; these give only the non-default flags, and the startup config is the same either way.
- **`--dram-arena-gib 68`, not 70:** the job harness keeps 5 GiB of the 110.9 GiB of `--mem=113560M` spare beside a server's pinned host memory (the arena, the 1-GiB connector pool and one 154 MB Gated-DeltaNet carry per session) and 32 GiB for the rest of its processes; at K = 24 a 70-GiB arena leaves 4.46 GiB, 68 GiB leaves 6.46 GiB.
- **Stock control:** unmodified vLLM 0.24.0 with the 262,144-token `STOCK_CMD` of [docs/benchmarks/swe-bench.md](docs/benchmarks/swe-bench.md#server-flags) (25 GiB of KV) at `--max-num-seqs 24`, thinking on and the earlier reasoning kept: the job harness's keep-reasoning stock plan (`stockkeep64t`, at 24 sequences `stockkeep64k24`) adds `--reasoning-parser qwen3` and `--default-chat-template-kwargs '{"preserve_thinking": true}'` to the server, and its agents send `chat_template_kwargs: {"enable_thinking": true}`, `max_tokens` 4096 and no stop string, with each earlier reply's reasoning also in the `reasoning` field (vLLM 0.24.0 reads an earlier reply's reasoning only from there; Mini-SWE-Agent 2.4.2 sends it as `reasoning_content`). The reasoning-dropped stock reference (job 3583822, [docs/status.md](docs/status.md)) differs by exactly those two points: without `preserve_thinking` and the `reasoning` field every earlier reply renders without its reasoning.
- **Why `--release-idle-pages`:** job 3613537 (sess24i, 2026-11-16, runtime `7a8c36e`: 24 sessions, `goal_query`, no idle release) stalled for good. After a format-error rollback the next turn needed its whole history in phase 1, about 1,075 attention blocks, while 211 were free and no request was running (`CAPACITY_STALL`). With `--release-idle-pages` the capacity gate frees idle sessions' resident pages for exactly this case ([docs/known-limits.md](docs/known-limits.md) 2.6).
- **Agent prompt and environment:** the agents' commands run with the repository's `testbed` environment active, and the prompt gives each repository's test command (the PoC prompt's test section is written for Astropy, while 128 of the 150 instances are Django): `REPO_ENV=testbed` of `examples/swebench/run_agents.sh` ([docs/known-limits.md](docs/known-limits.md) 5.6). Every configuration (the stock control, `full`, `ledger_query`, the members' policies) runs so; the results are not comparable with runs without it.
- **Agents for this runtime's servers** (on the agent host, in a copy of `examples/swebench` with the client lines below, as in [docs/benchmarks/swe-bench.md](docs/benchmarks/swe-bench.md); `IDS` = the 150 instance ids in the order of `verified-150.txt`, `HF_DATASETS_CACHE` filled by `build_tasks.py` for them):

```sh
MAX_RUNNING=24 MAX_TOKENS=4096 STEP_LIMIT=120 REPO_ENV=testbed \
  IMAGE_PINS=/abs/path/on/agent/swebench-client/verified-150.txt \
  MINI_PYTHON=/abs/path/on/agent/mini-env/bin/python MSWEA_CONFIG_DIR=/abs/path/on/agent/mswea-config \
  HF_DATASETS_CACHE=/abs/path/on/agent/swebench-datasets \
  MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1 REQUEST_TIMEOUT=4400 \
  /abs/path/on/agent/swebench-client/run_agents.sh "$AGENT_SOCKET" 20000 "$AGENT_OUT" $IDS
```

The copy's `model_override.yaml` holds these three lines after `max_tokens` (the 4,400 s request timeout and no retries):

```yaml
    timeout: 4400
    max_retries: 0
    num_retries: 0
```

`run_agents.sh`'s defaults are the pilots' values (90 steps, 1,536-token replies, one runner for all instances, the 32 pinned images, `bash -c` and the PoC prompt) and give a different configuration: set every variable above. The job harness's plans `full150k24e` / `ledger150k24e` render the same model override (`max_tokens` 4096, `step_limit` 120, `timeout` 4400, `max_retries` and `num_retries` 0, `interpreter: ["bash", "-lc"]`, the same prompt file) and also send `chat_template_kwargs: {"enable_thinking": true}`, which has no effect on this runtime's servers (`--enable-thinking` sets thinking). The stock control's agents are the job harness's (above; a 3,600 s timeout): `STOCK=1` of `run_agents.sh` is the thinking-off stock contract and does not apply here.
- **Status:** not yet GPU-validated. The 24-session smoke (job 3616345, on `7a8c36e`: `goal_query` with idle release and `--admission-overlap 24`) completed all 24 tasks without a stall; its verdict is FAIL only from a defect of the job harness's check, fixed since ([docs/status.md](docs/status.md)). Next: the 150-instance runs of the stock control, `full` and `ledger_query`, and of the members' policies on their branches.

## What to use

- **First runs:** the five policies of `cf2b28d` (`full`, `random50`, `goal_query` with the default Hugging Face Goal rewriter, `observation_query`, `random_fill`), started as in the [Quickstart](#quickstart); the speed defaults they then run with are the switches of the GPU-validated `cf2b28d` benchmark servers.
- **Qwen on SWE-bench Verified with thinking on:** the [stable configuration](#stable-configuration-qwen-on-swe-bench-verified).
- **Experimental outside the stable command:** `--goal-overlap`, `--goal-rewriter-engine vllm` and `--enable-thinking` (the stable command uses these three, on Qwen); `--async-scheduling`, `--thinking-history strip` (both profiles), `--window-hold-slots`, `--phase1-graph-tokens`, `--gdn-inplace-crossing`, more than 16 sessions, Gemma 4's thinking on, the policies `recent_fill`, `recent_tools` and `akv_rule`, the plug-in policies and `--token-gather`. Which of them have run on the GPU, and how: [docs/status.md](docs/status.md). The current configuration of the 64-task runs is the stable command plus `--async-scheduling` ([docs/status.md](docs/status.md#current-configuration)).

<a id="what-main-contains-runtime-4ebff97"></a>

## What `main` contains (runtime `2fb7f62`, since 2026-11-17)

Runtime `2fb7f62` is integrate-1117: `4ebff97` plus integrate-1114 (`ledger_query`, the marker pages, token gather), integrate-1116 (the exact history at a format-error rollback, `--gdn-inplace-crossing`, up to 32 sessions, `ledger_query` v2, the Goal kept across a rollback, `serve.py`'s help strings and the documentation's reorganization) and integrate-1117 (the session pool's task cap 256, `examples/swebench/verified-150.txt`). `4ebff97` (the tag `main-integrate-1112`) is `ba4d1ba` plus seven branches (the policy registry, the code-review fixes of 2026-11-10 and the host-work trims, always on; the `--async-scheduling` fix for replies cut at `max_tokens`, phase-1 CUDA graphs and Gemma 4 thinking on, opt-in; the Terminal-Bench client's 262,144-token context for Gemma). Each branch's entry and tests are in [History](docs/history.md#revisions); the GPU jobs of each part are in [docs/status.md](docs/status.md). What has not run on a GPU since `4ebff97`: [docs/known-limits.md](docs/known-limits.md) 5.7.

## The rule for `main` and the stable tag

The rule of 2026-11-15 (an opt-in change after its own GPU test and the parity proof, a shared-path change after one GPU smoke per model, each push approved by the maintainer) and the stable tag are in [docs/release-gates.md](docs/release-gates.md); the stable commit `main` names is in [docs/status.md](docs/status.md).

## Quickstart

### What you need

This repository plus a clean vLLM 0.24.0 environment is enough to install and serve:

1. **This repository** (a Git checkout): `git clone https://github.com/CFXM/vllm-0.24.0-selective-prefix-policies.git "$REPO"`, with `$REPO` as below. The repository is private; the maintainer grants access.
2. **A Python 3.12 environment with the official `vllm==0.24.0` wheel, unmodified, plus exactly `transformers==5.13.0` and `tokenizers==0.22.2`.** `install.py` refuses other versions. A plain `pip install vllm==0.24.0` may resolve a different Transformers version, so pin both (section [2.1](docs/install.md#21-base-environment) has the validated recipe). Source or editable vLLM installs are unsupported: the installer's wheel-RECORD check fails on them.
3. **The Qwen3.6-27B weights** at revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, and **Qwen3-0.6B** only for `goal_query` with the Goal rewriter (or start `goal_query` with `--no-goal-rewriter`) (section [2.2](docs/install.md#22-models)).
4. **Linux with `gcc` or `clang` and the Python headers of that interpreter** (Triton compiles kernels at run time) and a **short `TMPDIR`** (section [3](docs/install.md#3-runtime-environment)). Validated only on one NVIDIA GH200 (Linux aarch64). The installer does not check the architecture or the GPU, but other platforms are untested.

The installer never modifies the vLLM environment. It builds a separate release directory (a copy of the installed `vllm/` package plus this overlay), so the environment stays usable as plain vLLM. Serving needs no network: `serve.py` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

Running the SWE-bench harness additionally needs Mini-SWE-Agent 2.4.2 and a Docker host with the SWE-bench images. That is beyond install and serve (section [7](docs/benchmarks/swe-bench-mini-swe-agent.md#7-swe-bench-verified-with-mini-swe-agent-242)).

### From nothing to a smoke request

The commands below use these placeholders; later sections reuse them. Use absolute paths.

```sh
REPO=/abs/path/to/vllm-0.24.0-selective-prefix-policies
VENV=/abs/path/to/vllm-env
MODELS=/abs/path/to/models
RELEASE=/abs/path/to/releases/selective-prefix
RUNS=/abs/path/to/runs
cd "$REPO"
```

1. Build the base environment `$VENV` (section [2.1](docs/install.md#21-base-environment)) and get the model(s) into `$MODELS` (section [2.2](docs/install.md#22-models)).
2. Run the fast tests (`run_tests.py`; the full suite, `run_tests.py --all` and `check_git_release.py --ref HEAD`, is for before a push), check the environment, then create the release (sections [2.3](docs/install.md#23-tests) and [2.4](docs/install.md#24-install-the-release)). Run the tests with section [3](docs/install.md#3-runtime-environment)'s `PATH` and `LD_LIBRARY_PATH` exported (the first two lines below), as the release repository's GPU acceptance (job 3504770, `RLAC_PASS_VERIFIED`) ran its fast tests. The fast default needs only a CPU (the CI runs it on GitHub's ubuntu-24.04 runners); run the full suite (`--all`) on a compute node, not on a shared login node:

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

3. On a GH200 node, in a shell of a Slurm job (the five GPU checks of `cf2b28d` ran as batch jobs on the GH200 batch partition `ghx4`, one GH200 each; section [4.5](docs/running.md#45-gpu-job-slurm) runs this step as a batch job): run the placeholder block above again in that shell, set the runtime environment (section [3](docs/install.md#3-runtime-environment)) and create the short `TMPDIR` (private, `len(TMPDIR) + 37` at most 107 bytes), then dry-run, start, wait for READY, send one request and stop with `SIGTERM` (sections [4.1](docs/running.md#41-one-server) and [8](docs/design.md#8-evidence-per-server)):

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

The client prints one chat completion as JSON, its reply in `choices[0].message`, at most 16 tokens long (it sends `max_tokens` 16 and `temperature` 0). `wait` must return 0 and the last command must print `CLEAN_SHUTDOWN`. If READY fails, the loop stops the server and leaves the shell (`exit 1`), as the loop of section [4.2](docs/running.md#42-the-accepted-runs-four-policies-on-one-gpu-sequential) does. `--output` must be a directory that does not exist yet (its parent must): use a new one for every start (the dry run creates none). In a Slurm job, `examples/slurm/serve_gh200.sh` does step 3 for you (section [4.5](docs/running.md#45-gpu-job-slurm)). The smoke server runs with the speed defaults (CUDA graphs `full` and the decode fast paths, section [4.15](docs/reference.md#415-speed-defaults)), so its startup also captures the CUDA graphs; the READY times of section [4.1](docs/running.md#41-one-server) are those of eager servers (`--no-speed-defaults`).

For `goal_query`, `observation_query` and `random_fill`, add `--page-tokens 16 --kv-layout decoupled` to both `serve.py` commands (section [4.6](docs/reference.md#46-decoupled-kv-layout-and-16-token-pages)). `--page-tokens 16` alone already selects the decoupled KV layout (16-, 32- and 64-token pages always use it; `--kv-layout decoupled` matters only at 784, as a validation mode), and the benchmark servers pass both. The SWE-bench runs of `recent_fill` and `akv_rule` with thinking on (jobs 3539242 and 3577833) used the same 16-token pages. The dry run then shows `--block-size 16 --mamba-block-size 784 --additional-config ...` in the vLLM command:

```sh
python -B serve.py --release-dir "$RELEASE" --task examples/task.json --policy observation_query \
  --page-tokens 16 --kv-layout decoupled --output "$RUNS/obs" --port 8000
```

## Policies

Each server runs one policy (`--policy`); [docs/design.md](docs/design.md#5-page-selection-for-the-query-policies) section 5 is the catalogue: each policy's rule, its options, the protocols and profiles it serves and what does not combine. The terms (session, eligible and mandatory pages, budget, turn schedule) are defined there.

| `--policy` | Selection |
| --- | --- |
| `full` | Restore every eligible historical page (the reference). |
| `random50` | A seeded random half of the eligible pages. |
| `goal_query` | Online Selector V2: mandatory, recent and the pages closest to a maintained retrieval Goal's query vector fill the budget. |
| `observation_query` | The same selector; the query is the model's hidden state at the end of the newest observation. |
| `random_fill` | Control for the two query policies: their schedule and capacity, the rest filled in a seeded random order. |
| `recent_fill` | The same capacity, the rest filled newest first: one contiguous window ending at the newest sealed page. |
| `recent_tools` | As `recent_fill`, but tool-call pages first, then tool responses, then reasoning, each newest first. |
| `akv_rule` | AKV's rule: every reply and the latest 12 tool responses, no budget. |
| `ledger_query` | `observation_query` whose fill first reserves room for the latest edit and each file's latest read (a plug-in; the XML protocol of the Qwen profile). |
| `goal_query_markers` | `goal_query` whose fill first takes the pages that hold protocol markers (a plug-in; the XML protocol of the Qwen profile). |
| `observation_query_markers` | The same marker pages on top of `observation_query`'s fill. |

`--token-gather markers` or `separators` (an option of `goal_query` and `observation_query`) keeps single history tokens beside the selected pages inside the same budget ([docs/policy-registry.md](docs/policy-registry.md#6-token-level-selection-option-a-marker-pages-option-b-token-gather) section 6).

## Protocols and repository contents

With the default protocol (`--protocol mswea-xml`, the Mini-SWE-Agent XML actions) the harness talks to an ordinary OpenAI Chat Completions endpoint. It sends the complete, growing message history each turn; the policy, session, Goal, vectors and page choices never appear in HTTP. The other protocols and the session pool add session names: the kv-session protocol has its own endpoints with a session id, the chat-text protocol accepts an `X-Session-ID` header, and a session pool closes a session with `POST /v1/sessions/close` (section [6](docs/running.md#6-harness-interface)).

This repository contains the core runtime, installer and launcher, the CPU tests (`tests/`, run by `run_tests.py`; [docs/install.md](docs/install.md#23-tests)), plus examples: a smoke client, a Slurm job script, the glue our SWE-bench pilots put around Mini-SWE-Agent (section [7](docs/benchmarks/swe-bench-mini-swe-agent.md#7-swe-bench-verified-with-mini-swe-agent-242)) and the MemoryAgentBench and Terminal-Bench 2.0 clients (`examples/memoryagentbench/`, `examples/terminalbench/`; [docs/benchmarks/](docs/benchmarks/)). It also carries the SWE-bench runner and prompt of the PoC (the proof of concept that this runtime reimplements on vLLM: the DRAM-KV prototype `dram-backed-kv`, with its Selector V2 and Goal) that the pilots used (`examples/swebench/upstream/`).

Two more directories are not part of the runtime. `harnesses/` is a snapshot of the four GPU job harnesses that ran the release smokes (`perf-swe32`, `perf-gemma-swe32`, `perf-tb16-strip`, `perf-mab`); the release export drops it ([harnesses/README.md](harnesses/README.md)). `tools/` holds developer tools: `policy_parity.py` (the cross-tree probe of the policies), `step_profile.py` (per-step timing tables from server logs), `keep_render_check.py` (the keep-render check of a thinking-on run), `token_gather_bench.py`, `docs_set.py` (the documentation set that the tests read) and `docs_check.py` (the documentation's links and its English / Chinese parity).

The repository contains no result files or trajectories; the task lists it carries (in the clients and in `harnesses/`) name public benchmark tasks. Run archives are on the branch `data-swe32`. Internal `remote_ra_*` module names are implementation-private.

The release repository is `CFXM/agentscope` on GitHub, private like this one: an export of one commit of this runtime without its run history, with the internal `remote_ra_*` names renamed `spp_*`.

The MemoryAgentBench client holds the read_next harness of Shanbin Sun (the agent loop and its prompts, the MemAgent32 case selection and chunking) as this project runs it ([docs/benchmarks/memoryagentbench.md](docs/benchmarks/memoryagentbench.md), Harness). The MIT notices of the strings it takes from MemoryAgentBench and mini-swe-agent are in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

## How to add a page-selection policy

Every page-selection policy is declared once, in `source-overlay/policy_registry.py`: one `PagePolicy` class and one registration; the launcher, the startup config, the selection contract, the receipts and `runtime.jsonl` read its name and properties from there. A new fill policy is one class in `source-overlay/page_policies/` that orders the candidate pages Selector V2 leaves, plus one `PLUGINS` entry; then regenerate the release pins and reinstall. A query plug-in adds a hook that protects, excludes or groups pages before the cosine fill (as `ledger_query` and the marker policies). A worked example is `tests/policy_example.py`; the context a policy sees is in [docs/policy-registry.md](docs/policy-registry.md#23-the-fill-path-plug-ins) section 2.3, the registration API and the test helper `policy_registry.registered(...)` in section 2.4, the query hook in section 8.

## Documentation

Read in this order: run it, then see how a policy plugs in, then the internals. The numbered sections keep the numbers they had in the README until integrate-1107; each number is one section of one page, and code cites them as `docs/reference.md 4.9`.

| Page | Contents |
| --- | --- |
| [docs/status.md](docs/status.md) | the only status page: `main` and its stable commit, the current configuration, the GPU jobs by model, benchmark and policy (verdict, audited or not, date; every job from 33xxxxx on that the docs name) and what has not run yet, the branches |
| [docs/known-limits.md](docs/known-limits.md) | what does not work yet or only with care: each limit, whom it affects, the workaround, its source and when it is lifted |
| [docs/install.md](docs/install.md) | 1 supported boundary; 2 install and uninstall (2.1 to 2.5: base environment, models, tests, install, uninstall); 3 runtime environment; 9 release integrity and notices |
| [docs/running.md](docs/running.md) | 4.1 to 4.5: one server, the accepted run's four policies, several GPUs, several sessions, the Slurm job; 6: the harness interface (XML, kv-session and chat-text protocols, errors, another host, session close, thinking on) |
| [docs/SWEBENCH_BENCHMARK.md](docs/SWEBENCH_BENCHMARK.md), [docs/TERMINALBENCH_BENCHMARK.md](docs/TERMINALBENCH_BENCHMARK.md), [docs/MEMORYAGENTBENCH_BENCHMARK.md](docs/MEMORYAGENTBENCH_BENCHMARK.md) | each benchmark's configuration of record (Part A, a dated record; the current configuration is in docs/status.md) and its procedure (Part B) |
| [docs/benchmarks/](docs/benchmarks/) | the clients' pages, as in the release repository: [SWE-bench Verified](docs/benchmarks/swe-bench.md), [Terminal-Bench 2.0](docs/benchmarks/terminal-bench.md) (with this repository's Gemma 262,144-token edit), [MemoryAgentBench](docs/benchmarks/memoryagentbench.md); 7: [the SWE-bench pilots](docs/benchmarks/swe-bench-mini-swe-agent.md) |
| [docs/design.md](docs/design.md) | 5: the policy catalogue and page selection; 8: the evidence each server writes |
| [docs/policy-registry.md](docs/policy-registry.md) | writing a policy: the registry, its properties, the fill path, the parity evidence, token-level selection (6), the query hook and `ledger_query` (8), the marker pages (9) |
| [docs/reference.md](docs/reference.md) | 4.6 to 4.15, the server options: the decoupled KV layout, CUDA graphs, the context limit, the admission optimizations, the decode fast paths, text only, idle-page release, the Gemma 4 31B IT profile, the speed defaults; `python -B serve.py --help` lists every flag |
| [docs/release-gates.md](docs/release-gates.md) | branches and `main`, the GPU tests a change needs, the checks a number must pass, the edge-case smokes, the checks before a commit, the metric definitions |
| [docs/history.md](docs/history.md) | every revision, newest first, appended to only; [docs/history/](docs/history/) holds frozen records, not updated: [the validation record up to `cf2b28d`](docs/history/validation-cf2b28d.md), the job notes that left [running.md](docs/history/running-gpu-notes.md) and [reference.md](docs/history/reference-gpu-notes.md), the former validation sections of [SWEBENCH_BENCHMARK](docs/history/swebench-validation.md), [TERMINALBENCH_BENCHMARK](docs/history/terminalbench-validation.md) (its dry-run accounts) and [MEMORYAGENTBENCH_BENCHMARK](docs/history/memoryagentbench-validation.md), [the SWE-bench pilots' agent host, tasks and evaluation](docs/history/swe-bench-mini-pilots.md) (the former sections 7.2, 7.3 and 7.5), and [an unreviewed thinking-on analysis](docs/history/thinking-on-analysis.md) |
| `docs/<topic>/` | design notes, each dated: [async-sched](docs/async-sched/), [exp-decouple-pages](docs/exp-decouple-pages/), [goal-overlap](docs/goal-overlap/), [rewriter-vllm](docs/rewriter-vllm/), [thinking-cut](docs/thinking-cut/), [thinking-gemma](docs/thinking-gemma/), [token-gather](docs/token-gather/) |
| [harnesses/README.md](harnesses/README.md) | the snapshot of the four GPU job harnesses (their own pages are copies, not edited here) |

### Documentation map

Every page has a Chinese twin (`.zh-CN.md`); `tools/docs_check.py` checks that the two agree and that every link resolves. Section pointers name the page that holds the section, as in the table above.
