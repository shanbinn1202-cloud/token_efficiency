# SWE-bench Verified with Mini-SWE-Agent 2.4.2 (the pilots)

English | [简体中文](swe-bench-mini-swe-agent.zh-CN.md)

Section 7 of the README until integrate-1107, with its numbers. The benchmark configuration is [SWEBENCH_BENCHMARK.md](../SWEBENCH_BENCHMARK.md), the release's page [swe-bench.md](swe-bench.md). Section numbers are unique across the docs pages; the [documentation map](../../README.md#documentation-map) lists the page that holds each.

## 7. SWE-bench Verified with Mini-SWE-Agent 2.4.2

The configuration of the SWE-bench Verified benchmark run (the first 32 instances, 16 concurrent agents refilled in list order, the policies `full`, `observation_query`, `goal_query` and `random_fill`, and a stock vLLM 0.24.0 control) and the checklist for aligning baselines with it are in [docs/SWEBENCH_BENCHMARK.md](../SWEBENCH_BENCHMARK.md).

This section describes the setup of the multi-session SWE-bench pilots: 3 Astropy instances × `full`/`goal_query`/`observation_query` with 3 sessions per server (job 3299922), and 10 instances × the same 3 policies with 5 sessions per server (job 3300434), on runtime `b1e6988` with 784-token pages. The same 10 instances then ran with 16-token pages × `full`/`observation_query`/`goal_query`/`random_fill`, one job per policy (jobs 3313802, 3314010, 3314319, 3314623, runtime `780fe54`; outcomes in section 4.7), with the agent host unchanged. Every server passed its server-side checks. One run per instance is not benchmark evidence. `random50` was not part of these pilots.

### 7.1 Settings

| Setting | Value |
| --- | --- |
| Agent | Mini-SWE-Agent 2.4.2 (litellm 1.103.1 and datasets 5.0.1: not re-verified, the jobs' evidence records only Mini-SWE-Agent's version) |
| Runner | the PoC's `swebench_runner.py` with `--subset verified --split test --workers <sessions> --filter '^(<id>\|<id>…)$' --output <dir>` |
| Config stack | `-c swebench_xml.yaml` (built into Mini-SWE-Agent) `-c swebench_prompt_override.yaml` (the PoC's prompt) `-c model_override.yaml` |
| Model | `examples/swebench/model_override.yaml`: `litellm_textbased`, `openai/selective-prefix`, `api_base` `http://127.0.0.1:<port>/v1`, dummy key, temperature 0.0, `max_tokens` 1536 (`MAX_TOKENS` of `run_agents.sh`; section 4.9), `cost_tracking: ignore_errors` |
| Environment | `docker`, `cwd` `/testbed`, timeout 120 s, `bash -c`, `PAGER`/`MANPAGER` `cat`, `LESS` `-R`, `PIP_PROGRESS_BAR` off, `TQDM_DISABLE` 1 |
| Limits | `step_limit` 90 (`STEP_LIMIT` of `run_agents.sh`; section 4.9), `cost_limit` 0 |
| Process environment | `MSWEA_CTX_PATCH` unset, `AGENTDIET_MODE=off`, `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `MSWEA_GLOBAL_CONFIG_DIR` = an empty directory |
| Servers | one fresh server per policy and group, `--task tasks-g<k>.json`, `--kv-cache-gib 24`, `goal_query` with the 0.6B rewriter; workers = sessions = group size. The 16-token-page pilot added `--page-tokens 16 --kv-layout decoupled` to every server, `full` included, and gave `random_fill` no rewriter option |
| Evaluation | swebench 5.0.2, `SWE-bench/SWE-bench_Verified`, split `test`, `--timeout 1800`, `--max_workers 2` (the pilot script's default) |

`model_override.yaml` with the port filled in is byte-identical to the six files the 10-instance pilot used. The server ignores the client's sampling fields anyway (section 6.1). Mini-SWE-Agent writes `<output>/<instance>/<instance>.traj.json` and `<output>/preds.json`.

**PoC files.** `examples/swebench/upstream/` holds `swebench_runner.py` and `swebench_prompt_override.yaml` from the dram-backed-kv PoC repository (`scripts/` at commit `a70701e`, Git blob bytes with LF; the prompt override equals the copy on the cohort-2 branch `0ab0078`), byte-identical to the copies the pilots ran (`upstream/SOURCE.txt`, `upstream/SHA256SUMS`). `run_agents.sh` uses them by default and refuses files whose SHA-256 differs from these pins:

| File | SHA-256 |
| --- | --- |
| `swebench_runner.py` | `34ca6d9288c72830e61bd00ad8b15790cab4c29515db24b8df30775cfc0880a7` |
| `swebench_prompt_override.yaml` | `1b6a48666ad528775bd11225fa4d5c002721c25c0cdbc05b99cc15b2de31336a` |

Two properties of these files change results, so keep them in mind when you compare with other setups:
- The runner replaces Mini-SWE-Agent's agent class. When the step limit is reached it submits the last `git diff` of at least 3 lines that the agent printed (exit status `LimitsExceeded_InformalSubmission`), and it stops an instance after 10 identical commands in a row (`LoopDetected`, with or without that patch: a found patch is still submitted, `LoopDetected_InformalSubmission`). It also imports an optional `agentdiet_adapter` that is not included; `AGENTDIET_MODE=off` (as in the pilots) leaves it unused.
- The prompt is written for Astropy ("The testbed is an Astropy repository", with Astropy-specific test advice). The pilots ran Astropy instances only; other repositories need an adapted prompt, which is **not part of the validated runs**.

### 7.2 Agent host

Today's agent host (Docker with the instance images at their pinned digests, the tools, the agent environment and the evaluator at its pinned commit) is in the swe-bench page's [Agent host setup](swe-bench.md#agent-host-setup). The pilots' agent host of 2026-09/10 (unpinned images, the PyPI evaluator, commands not recorded), which today's scripts partly refuse, is frozen in [docs/history/swe-bench-mini-pilots.md](../history/swe-bench-mini-pilots.md#from-section-72-agent-host).

### 7.3 Tasks

Each server's `--task` list must hold exactly the instances its agents run, with `{instance_id, problem_statement}` taken verbatim from SWE-bench Verified: requests are routed by the problem statement inside the task message (sections 4.4 and 6.3). `examples/swebench/build_tasks.py` builds the files from the dataset at its pinned revision: `--pool` for a session-pool server (the swe-bench page's [Tasks](swe-bench.md#tasks)) and `--max-sessions` for groups ([A small group with the Slurm script](swe-bench.md#a-small-group-with-the-slurm-script)). The pilots' task files are frozen in [docs/history/swe-bench-mini-pilots.md](../history/swe-bench-mini-pilots.md#from-section-73-tasks).

### 7.4 Run one group

Per policy and group: a fresh server with that group's list, the tunnel, then one runner on the agent host (`run_agents.sh <socket> <port> <new output dir> <instance ids of the group>`). With the Slurm script (today's form of the pilots' submission: the pilots requested `--mem=200g`, and their `run_agents.sh` needed no `HF_DATASETS_CACHE`; the swe-bench page's [Tasks](swe-bench.md#tasks) fills that cache): `ACCOUNT`, `PARTITION` and `MAIL` are those of [the Slurm job](../running.md#45-gpu-job-slurm), and `REPO`, `VENV`, `RELEASE` and `MODELS` the placeholders of the README Quickstart ([From nothing to a smoke request](../../README.md#from-nothing-to-a-smoke-request)). `RUN_ROOT` must exist before `sbatch`:

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

Notes:
- One server per policy, a fresh one for every group, a fresh output directory and socket each time. The pilots ran every policy over the same groups, one server after another on one GPU.
- `PAGE_TOKENS=16 KV_LAYOUT=decoupled` is the recommended setting for `goal_query`, `observation_query` and `random_fill` (section 4.6); leave both unset for the default 784-token layout. For `random_fill`, set `POLICY=random_fill`; the script passes a rewriter to `goal_query` only.
- Time: in job 3300434 a group of 5 agents took 1,255–2,120 s, and the last of its 6 servers (3 policies × 2 groups) was done 10,851 s after the job started, inside the 10:45:00 allocation. The pilots capped each agent run at 3,300 s (job 3299922) or 5,400 s (job 3300434). In the 16-token-page pilot (one policy and 2 groups per job, agent cap 3,000 s) a group took 1,215–2,045 s and each job 49–68 minutes of its 02:00:00 allocation.
- An SSH drop or timeout does not stop the remote runner. The pilots then killed that runner and removed only the containers it had started (`minisweagent-<8 hex>` names in its log).
- In jobs 3299922, 3300434 and 3314319 some `goal_query` turns fell back to deterministic Goal fields (finding `GOAL_FALLBACK`; section 5); the servers still passed.
- Session pool (section 4.4): set `TASKS=.../tasks-pool.json` and `MAX_RUNNING=<K>`. The Slurm script passes `--max-running-sessions K` to `serve.py` and `MAX_RUNNING` to `run_agents.sh`, which then runs a pool of K agents, one runner per instance in the list's order, and closes each instance's session when its runner ends (retrying while its last request is still being torn down) before starting the next. It writes `runs/<instance>/`, `logs/<instance>.log`, `closes/<instance>.json`, `timeline.jsonl` (start and end time, runner exit and close status per instance: the wall clock from the first start to the last end) and a merged `preds.json` for `evaluate.sh`. `run_agents.sh` itself has not run this mode on a validated job; the same pool form (K agents refilled in list order, one close per instance) ran in the SWE-bench Verified benchmark jobs through the benchmark harness, for example job 3461703 (`SWE32_PASS_VERIFIED`, 16 running of 32 tasks).
- Step limit and reply bound (section 4.9): `STEP_LIMIT=<1-256>` and `MAX_TOKENS=<1536|4096>` set Mini-SWE-Agent's `step_limit` and `max_tokens` in the model override that `run_agents.sh` writes (`render_override.py`); the Slurm script passes both to the agent host and `MAX_TOKENS` (with `MAX_MODEL_LEN` and `DRAM_ARENA_GIB`) to `serve.py`. Unset, the override is byte for byte the pilots' file and `serve.py` gets no new flag. A step limit of 120 with a 4096-token reply bound ran in the SWE-bench Verified benchmark jobs (for example job 3461703), set by the benchmark harness, not by `run_agents.sh`.

### 7.5 Evaluate

After the agents, per policy, on the agent host: `examples/swebench/evaluate.sh` with every group output of that policy as its last arguments, as in the swe-bench page's [Evaluation](swe-bench.md#evaluation) (the pinned test split as a local Parquet file, the evaluator at commit `02e7a74`). The pilots' evaluation (`swebench==5.0.2` from PyPI on `SWE-bench/SWE-bench_Verified`) is frozen in [docs/history/swe-bench-mini-pilots.md](../history/swe-bench-mini-pilots.md#from-section-75-evaluate). Trajectories, predictions and logs contain prompts and tool output: keep them out of public releases.
