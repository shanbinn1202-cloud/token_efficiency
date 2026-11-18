# SWE-bench Verified benchmark configuration

English | [简体中文](SWEBENCH_BENCHMARK.zh-CN.md)

This document fixes the configuration of the SWE-bench Verified benchmark run of this runtime: the first 32 instances of the `test` split, Mini-SWE-Agent 2.4.2 with at most 16 concurrent agents refilled in list order, one server per policy for `full`, `observation_query`, `goal_query` and `random_fill`, and a stock vLLM 0.24.0 server as the control. **Part A** is the configuration of record. A baseline (the stock vLLM control or any other system) is comparable only if it reproduces every item of the checklist in A.12; the configuration does not depend on where the agents run. **Part B** is a manual procedure with the tools of this repository. It is optional: any x86_64 Docker host that can reach the server over loopback can run the agents.

**Configuration of record.** Part A is the configuration of record of this page's commands, kept as it ran so that a baseline can be aligned with those runs: the Qwen runs at runtime `120bc3c` (jobs 3428631 and 3429500, with the stock control 3429193) and the Gemma runs of A.14, in October 2026, with thinking off, synchronous scheduling, 32 GiB of GPU KV (Gemma: 25 GiB; A.14 sets 32 for `full`) and 131,072 tokens. It is a dated record, not the current configuration. Each configuration the documentation names has one page: this record (Part A); the release's thinking-off commands at 25 GiB and 262,144 tokens, whose 25-GiB, 262,144-token sizes the Qwen runs have used since `cf2b28d` ([benchmarks/swe-bench.md](benchmarks/swe-bench.md)); and the current configuration, which differs in thinking, scheduling, KV size and context, together with the stable configuration (the README's command, job 3576565), in [docs/status.md](status.md#current-configuration).

Section references name the page that holds the section, such as [docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound); the README's [documentation map](../README.md#documentation-map) lists the pages.

> **The client in this repository (integrate-1107).** `examples/swebench/` now has the release's client changes: the pinned instance images (`verified-32.txt`, checked by `images.py` before `run_agents.sh` and `evaluate.sh` start), the pinned dataset revisions (`build_tasks.py` `REVISION` and the statement pins of all 32 instances; `evaluate.sh` evaluates a local copy of the pinned test split), `HF_DATASETS_CACHE`, the run's own process group (`agents.pgid`) the opt-in `STOCK=1` switch for the stock vLLM control (`render_override.py`; test: `test_align_ctx.py`) and, since 2026-11-17, the opt-in `REPO_ENV=testbed` (the repository's environment and test command for the agents: [known-limits.md](known-limits.md) 5.6). The release's page for it is [benchmarks/swe-bench.md](benchmarks/swe-bench.md).

## Validation status

Every GPU run of this benchmark, with its runtime commit, model, policy, thinking setting, verdict, resolved count and whether its numbers are audited, is in [docs/status.md](status.md). The record this section held until 2026-11-16 (the runs of Part A's configuration at `5b15b2f` and `120bc3c`, the Gemma runs of A.14, the later 25-GiB, 262,144-token runs and the earlier checks of parts of this configuration) is frozen, unchanged, in [docs/history/swebench-validation.md](history/swebench-validation.md). The scripts of Part B: [docs/known-limits.md](known-limits.md#54-the-part-b-job-scripts-of-the-benchmark-pages-are-untested).

## Configuration for aligning a baseline

What the runs of Part A use (the configuration of record: thinking off, 32 GiB, 131,072 tokens; the later runs' 25-GiB, 262,144-token setting is in [benchmarks/swe-bench.md](benchmarks/swe-bench.md), the current configuration in [docs/status.md](status.md#current-configuration)), for a baseline to set the same way; every value is that of Part A (A.12 is the full checklist, A.13 lists the differences that remain):

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, on one GH200 120GB with the official vLLM 0.24.0 wheel and `--mem=113560M` (A.2); text only, `--language-model-only` on both servers (A.3, A.4) |
| Thinking | Off: the chat template with `enable_thinking=false`, an empty think block in the generation header; a stock baseline sends `chat_template_kwargs: {"enable_thinking": false}` (A.3, A.4) |
| Decoding | Greedy, enforced by the server; a stock baseline sends `temperature` 0.0. vLLM `--seed 42`. Stop at `</mswea_bash_command>`, kept in the output (A.3, A.4) |
| Context limit | `--max-model-len 131072`, prompt plus reply (A.3) |
| Reply bound | 4096 new tokens per model call (`max_tokens` 4096; A.3, A.7) |
| Per-session KV budget | `--active-token-budget 8192` for `observation_query`, `goal_query` and `random_fill`; `full` keeps every page (A.3) |
| Concurrency | 32 registered tasks, at most 16 agents running, refilled in list order; `--max-num-seqs 16` (A.1, A.8) |
| KV cache of this runtime | 32 GiB of GPU KV: `--kv-cache-gib 32`, 34,359,738,368 bytes; on the host a 70-GiB pinned page arena (`--dram-arena-gib 70`) and the 1-GiB connector pool (`--connector-pool-gib 1`) (A.3) |
| Switches of this runtime | `--cuda-graphs full`, `--decode-fastpath --decode-fastpath2 --decode-fastpath3`, `--deferred-completion-io`, `--language-model-only`; `goal_query` adds `--admission-overlap 16` and `--goal-rewriter-batch 4`. The decode fast paths, the deferred completion files and the admission and rewriter flags add no vLLM argument (A.3) |
| Stock control (vLLM) | `--kv-cache-memory-bytes 34359738368` (32 GiB), `--max-model-len 131072`, `--max-num-batched-tokens 16384`, `--max-num-seqs 16`; CUDA graphs: vLLM's default (`FULL_AND_PIECEWISE`); asynchronous scheduling: vLLM's default (on); `--enable-prefix-caching --prefix-caching-hash-algo sha256`; no `--attention-backend` (vLLM picks FLASH_ATTN with FlashAttention 3); `--language-model-only` (A.4, A.5) |
| Throughput | The client-side completion tokens of every model call of the main agents of the 32 instances / the wall time from the first agent's start to the last agent's end on the agent host's clock. Tool execution, task-container start and stop, waits and the session closes before refills are included; server start-up and shutdown, evaluation and the Goal rewriter's tokens are not (A.10) |
| Tasks | SWE-bench Verified `test`, the first 32 instances in dataset order (22 Astropy, 10 Django); Mini-SWE-Agent 2.4.2 (litellm 1.103.1) with the PoC runner and prompt, step limit 120 (A.1, A.6) |
| Evaluation | SWE-bench at commit `02e7a74` (swebench 5.0.2) on the agent (x86_64 Docker) host, `SWE-bench/SWE-bench_Verified` `test`, `--timeout 1800`, 4 workers; the agents' task containers run the official images `docker.io/swebench/sweb.eval.x86_64.*` (A.6, A.9) |
| Gemma 4 31B IT | Fixed in A.14: the same instances, agent, limits and evaluation on Gemma's XML turns; `--kv-cache-gib 25` for `observation_query`, `random_fill` and `goal_query` and 32 for `full`, `--release-idle-pages`, a 72-GiB arena, `goal_query` with `--goal-rewriter-batch-tokens 15360`; the stock control on `TRITON_ATTN` with stage 0's flags. Its GPU runs: [docs/status.md](status.md) |

---

## Part A. Configuration

### A.1 Scope and instances

| Item | Value |
| --- | --- |
| Model | Qwen3.6-27B on one NVIDIA GH200 120GB with vLLM 0.24.0 (A.2) |
| Benchmark | SWE-bench Verified, `test` split, the **first 32 instances in dataset order**: 22 Astropy instances, then 10 Django instances |
| Concurrency | 32 registered tasks, **at most 16 agents running**. When an agent ends, the next instance in list order starts (refill); the run is not two waves of 16 |
| Servers | One fresh server per policy, in its own GPU job: `full`, `observation_query`, `goal_query`, `random_fill` (this runtime) and `stock` (stock vLLM 0.24.0, the control) |
| Run | One server runs all 32 instances once, without interruption (A.11). One run is one sample (A.13) |

The 32 instance IDs, in order (B.1 saves this list as `ids.txt`):

```text
astropy__astropy-12907
astropy__astropy-13033
astropy__astropy-13236
astropy__astropy-13398
astropy__astropy-13453
astropy__astropy-13579
astropy__astropy-13977
astropy__astropy-14096
astropy__astropy-14182
astropy__astropy-14309
astropy__astropy-14365
astropy__astropy-14369
astropy__astropy-14508
astropy__astropy-14539
astropy__astropy-14598
astropy__astropy-14995
astropy__astropy-7166
astropy__astropy-7336
astropy__astropy-7606
astropy__astropy-7671
astropy__astropy-8707
astropy__astropy-8872
django__django-10097
django__django-10554
django__django-10880
django__django-10914
django__django-10973
django__django-10999
django__django-11066
django__django-11087
django__django-11095
django__django-11099
```

The policies of this runtime that this page names (the README's policy table and [docs/design.md section 5](design.md#5-page-selection-for-the-query-policies)). The configuration of record runs the first four; the last three are not part of it (their GPU runs: [docs/status.md](status.md)):

| `--policy` | History pages used for the next turn |
| --- | --- |
| `full` | Every historical page: the reference, and the configuration closest to stock vLLM |
| `observation_query` | Selector V2: mandatory, recent and top-cosine pages within the budget; the query is the hidden state at the end of the newest observation |
| `goal_query` | The same selector; the query is a maintained retrieval Goal, updated by a Qwen3-0.6B rewriter |
| `random_fill` | Control of the two query policies: the same budget, mandatory and recent pages, the remaining capacity filled in a seeded random order |
| `recent_fill` | Recent-window policy (recent-fill): the same budget, mandatory and recent pages, the remaining capacity filled newest first (one contiguous window) |
| `recent_tools` | Recent-window policy (recent-fill): as `recent_fill`, but every tool-call page first, then the tool-response pages, then the reasoning pages, each newest first |
| `akv_rule` | AKV rule (akv-rule): every agent reply and the latest 12 tool responses (the current one among them), older tool responses dropped; no budget and no token threshold |

### A.2 Model and versions

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, a local directory with regular files; metadata SHA-256 values in [docs/install.md section 2.2](install.md#22-models) |
| Goal rewriter (`goal_query` only) | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`; the same resolved path for every `goal_query` server of one comparison ([docs/install.md section 2.2](install.md#22-models)) |
| vLLM | The official wheel `vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl`, unmodified, SHA-256 `700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5` |
| Python environment | CPython 3.12, `transformers==5.13.0`, `tokenizers==0.22.2`, built with `examples/env/constraints-3270721.txt` ([docs/install.md section 2.1](install.md#21-base-environment)). The stock control runs from the same environment; this runtime is installed into a separate release directory and leaves the environment unchanged ([docs/install.md section 2.4](install.md#24-install-the-release)) |
| Runtime | This repository at commit `120bc3c` (`120bc3c0a6db281eeba21f541957c3ae6941542d`, integrate-1019-tbgoal), the commit of the Qwen runs of record (jobs 3428631 and 3429500; the stock control runs no code of this runtime), and `9ef0d6d` (`9ef0d6d721d6f82a5f724d3f37965869092f8bf4`, integrate-1021) for the Gemma runs (A.14); or a later commit that leaves the XML protocol path and the vLLM arguments of A.3 unchanged; `main` contains them. For `goal_query`, use `53e3b7a` (`53e3b7a2304a443e9294768d213af6f4332e2b8c`) or later: it adds the linear-time quoted-literal pass of the Goal features ([docs/history.md](history.md#history)). Before it, an unclosed quote followed by many backslash escapes in a session's previous reply could stall the engine (Terminal-Bench job 3429670); on SWE-bench, job 3428631 shows the same cost at small scale: its two slowest Goals without a rewriter call took 155.9 ms and 14.7 ms, against a median of 0.62 ms. History: the runs of 2026-10-18 ([the frozen validation record](history/swebench-validation.md)) used `5b15b2f` ("Batch the Goal rewriter calls that queue"); earlier runs of this benchmark used `e877051`, which encodes each message with the loaded tokenizer only, as stock vLLM does; with `3496d60` one Django instance was refused with HTTP 500 |
| GPU | One NVIDIA GH200 120GB, Linux aarch64; one server per GPU at a time |
| Slurm resources | `--nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16 --mem=113560M` (one GH200's share of the node's host memory; A.3 explains why it suffices) |
| Agent software | Mini-SWE-Agent `2.4.2`, litellm `1.103.1`, Python 3.12; datasets: the runs' evidence does not record its version (B.1 installs `5.0.1`) |
| Evaluator | SWE-bench at commit `02e7a74ffd0b707aab73d203fe87bdc7c76afc8e` (swebench `5.0.2`; the runs' evaluator environment is named after this commit); datasets as above |

### A.3 This runtime: server flags and vLLM arguments

**Flags shared by the four policies** (`serve.py`):

| Flag | Value | Effect |
| --- | --- | --- |
| `--task` | `tasks-pool.json`: the 32 instances of A.1 in order, built by `examples/swebench/build_tasks.py --pool` | Registers 32 tasks; each request is routed by the problem statement in its task message ([docs/running.md section 4.4](running.md#44-several-sessions-on-one-server)) |
| `--max-running-sessions` | `16` | Session pool: at most 16 sessions open; a session opens at its task's first request and is closed by the agent pool (A.8). vLLM gets `--max-num-seqs 16` |
| `--kv-cache-gib` | `32` | vLLM gets `--kv-cache-memory-bytes 34359738368` |
| `--page-tokens` / `--kv-layout` | `16` / `decoupled` | 16-token attention pages and a separate Gated-DeltaNet pool of 33 slots ([docs/reference.md section 4.6](reference.md#46-decoupled-kv-layout-and-16-token-pages)). vLLM gets `--block-size 16 --mamba-block-size 784 --additional-config ...` |
| `--max-model-len` | `131072` | Context limit, prompt plus reply. A prompt holds at most 131,072 − 4,096 = 126,976 tokens ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)) |
| `--max-tokens` | `4096` | Reply bound, forced by the server on every request ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)) |
| `--dram-arena-gib` | `70` | One pinned host page arena of 70 GiB for all sessions: 71,680 Pages of 1 MiB. It holds 8 sessions at the 131,072-token limit (8,193 Pages each), not 16; the 16 largest sessions of the earlier 16-session SWE-bench runs need 58.32 GiB at once. A capture beyond it fails that request with `DRAM_ARENA_EXHAUSTED` ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)) |
| `--connector-pool-gib` | `1` | The connector's unused CPU pool at 1 GiB instead of 32 ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)). vLLM gets `cpu_bytes_to_use` 1073741824 |
| `--cuda-graphs` | `full` | vLLM gets `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}` instead of `--enforce-eager` ([docs/reference.md section 4.8](reference.md#48-cuda-graphs)) |
| `--decode-fastpath` / `--decode-fastpath2` / `--decode-fastpath3` | on / on / on | Less host work per decode step, the same values; each stage needs the one before ([docs/reference.md section 4.11](reference.md#411-decode-fast-path-on-by-default)). No vLLM argument |
| `--deferred-completion-io` | on | A completed request's native file and `goals.jsonl` row are written by one writer thread after the step's outputs instead of on the engine thread, with the same bytes ([docs/reference.md section 4.11](reference.md#411-decode-fast-path-on-by-default)). No vLLM argument |
| `--language-model-only` | on | No vision tower and no multimodal warmup ([docs/reference.md section 4.12](reference.md#412-text-only---language-model-only-on-by-default)). vLLM gets `--language-model-only` |
| `--protocol` | `mswea-xml` (the default; not passed) | Mini-SWE-Agent XML protocol ([docs/running.md section 6.1](running.md#61-mini-swe-agent-xml-protocol)) |
| `--port` | `8000` | The server binds `127.0.0.1` only |

**Flags per policy:**

| Policy | Selector flags | Admission flags | Goal rewriter |
| --- | --- | --- | --- |
| `full` | none | none | none |
| `observation_query` | `--active-token-budget 8192` | none | not started ([docs/design.md section 5](design.md#5-page-selection-for-the-query-policies)) |
| `random_fill` | `--active-token-budget 8192` (fill seed: the default 42) | none | not started ([docs/design.md section 5](design.md#5-page-selection-for-the-query-policies)) |
| `recent_fill`, `recent_tools` | `--active-token-budget 8192` (the window order is pinned; no option) | none | not started ([docs/design.md section 5](design.md#5-page-selection-for-the-query-policies)) |
| `akv_rule` | none (no budget: the AKV rule's selection is the capacity; `--active-token-budget` is refused) | none | not started ([docs/design.md section 5](design.md#5-page-selection-for-the-query-policies)) |
| `goal_query` | `--active-token-budget 8192 --goal-rewriter-model <Qwen3-0.6B directory> --goal-rewriter-batch 4` | `--admission-overlap 16` | one Qwen3-0.6B process on the same GPU (about 2.7 GiB: 2,806 MiB measured in job 3410948). `--goal-rewriter-batch 4` is the setting of the configuration of record: the batched run is the one compared. Batched greedy decoding of the rewriter is not bit-identical to one call at a time, so compare policies only between servers with the same setting ([docs/reference.md section 4.10](reference.md#410-admission-optimizations-opt-in)); the per-batch token cap is the default 16,384 (`--goal-rewriter-batch-tokens`, not passed; `startup.json` and `ready.json` pin it). The recovery from batch-only GPU memory failures and this cap are in `120bc3c` and later; the measurements behind them are in [docs/reference.md section 4.10](reference.md#410-admission-optimizations-opt-in) |

No server sets `--gdn-device-handoff`, `--phase1-in-step`, `--active-ratio`, `--goal-query-variant`, `--fill-seed`, `--no-goal-rewriter` or `--release-idle-pages`. The admission flags follow [docs/reference.md section 4.10](reference.md#410-admission-optimizations-opt-in) (`--admission-overlap 16` for `goal_query` only). Selector, rewriter and admission flags, the decode fast paths and `--deferred-completion-io` pin the startup configuration but add no vLLM argument, so every server of the commands below starts vLLM with the same arguments.

**Host memory.** The pinned host memory is the 1-GiB connector pool, the Gated-DeltaNet carries (2.29 GiB) and the 70-GiB arena: 73.29 GiB in the servers' dry runs. With about 32 GiB for the processes this fits one GH200's share, `--mem=113560M` (110.9 GiB); `--language-model-only` keeps vLLM's multimodal warmup, which takes tens of GiB of host memory for under a second, out of the job ([docs/reference.md section 4.12](reference.md#412-text-only---language-model-only-on-by-default)). Earlier versions of this configuration used `--dram-arena-gib 88`, the default 32-GiB connector pool and `--mem=200g`; job 3348705 ([the frozen validation record](history/swebench-validation.md)) ran the 88-GiB arena with 200 GiB requested ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)).

**Commands.** Dry runs of six servers: the four of the configuration of record and `recent_fill` and `recent_tools` (add `--start` to serve; `TASKS` is the path of `tasks-pool.json`, `LABEL` a new label per attempt):

```sh
SHARED=(--max-running-sessions 16 --kv-cache-gib 32 --page-tokens 16 --kv-layout decoupled
        --max-model-len 131072 --max-tokens 4096 --dram-arena-gib 70 --connector-pool-gib 1 --cuda-graphs full
        --decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io --language-model-only)
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy full "${SHARED[@]}" \
  --output "$RUNS/full-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy observation_query "${SHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy random_fill "${SHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/random_fill-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy recent_fill "${SHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/recent_fill-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy recent_tools "${SHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/recent_tools-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy goal_query "${SHARED[@]}" \
  --active-token-budget 8192 --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --goal-rewriter-batch 4 \
  --admission-overlap 16 --output "$RUNS/goal_query-$LABEL" --port 8000
```

**Resulting vLLM arguments** (identical for the six servers; the dry run prints the authoritative list, here wrapped by argument group):

```text
--model <model directory> --served-model-name selective-prefix --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
--host 127.0.0.1 --port 8000 --distributed-executor-backend uni --no-async-scheduling
--tensor-parallel-size 1 --pipeline-parallel-size 1
--max-num-seqs 16 --max-model-len 131072 --max-num-batched-tokens 16384
--kv-cache-memory-bytes 34359738368 --seed 42 --enable-prefix-caching --prefix-caching-hash-algo sha256
--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}
--mamba-cache-mode align --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
--chat-template <output>/chat_template.jinja --chat-template-content-format string
--kv-transfer-config {"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":1073741824}}
--gdn-prefill-backend triton
--block-size 16 --mamba-block-size 784 --additional-config {"remote_ra_kv_layout":{"mamba_slots":33,"mamba_tokens":784,"page_tokens":16}}
--language-model-only
```

**Environment** of the shell that starts `serve.py` ([docs/install.md section 3](install.md#3-runtime-environment)):

| Variable | Value |
| --- | --- |
| `PATH` | `$VENV/bin:/usr/local/bin:/usr/bin:/bin` |
| `LD_LIBRARY_PATH` | `$VENV/lib/python3.12/site-packages/nvidia/cu13/lib` |
| `TMPDIR` | A short path: `len(TMPDIR) + 37` ≤ 107 bytes (AF_UNIX limit) |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` (native top-k/top-p sampler) |
| `CC`, `VLLM_RPC_BASE_PATH` | unset |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `PYTHONDONTWRITEBYTECODE` | `1` |
| Set by `serve.py` | `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `VLLM_USE_V2_MODEL_RUNNER=0` and the runtime's own variables |

**Generation contract, enforced by the server** ([docs/running.md section 6.1](running.md#61-mini-swe-agent-xml-protocol) and [docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)). The server ignores the client's sampling fields:
- greedy decoding, at most 4096 new tokens, a stop once the decoded text contains `</mswea_bash_command>` or on EOS/pad;
- thinking off: the system and task messages go through the pinned chat template with `enable_thinking=false`, and the generation header is `<|im_start|>assistant\n<think>\n\n</think>\n\n`;
- every previous reply in the history is replaced by the exact token IDs it was generated with;
- the response content is the decoded text of every generated token, including the marker and anything the final token carries past it.

### A.4 Stock vLLM control

The control is plain `vllm.entrypoints.openai.api_server` from the same environment (A.2), without any code of this runtime.

| Setting | Value | Relation to this runtime |
| --- | --- | --- |
| `--model` | The same Qwen3.6-27B directory | Same |
| `--served-model-name` | `selective-prefix` | Same name, so the agents use the same model name |
| `--host` / `--port` | `127.0.0.1` / `8000` | Same (loopback only) |
| `--distributed-executor-backend` / `--tensor-parallel-size` | `uni` / `1` | Same |
| `--max-num-seqs` | `16` | Same as K = 16 |
| `--max-model-len` | `131072` | Same |
| `--max-num-batched-tokens` | `16384` | Same (fixed by the launcher) |
| `--kv-cache-memory-bytes` | `34359738368` (32 GiB) | Same bytes |
| `--seed` | `42` | Same |
| `--enable-prefix-caching --prefix-caching-hash-algo sha256` | on | Same |
| `--language-model-only` | on | Same ([docs/reference.md section 4.12](reference.md#412-text-only---language-model-only-on-by-default)) |
| `--mamba-cache-mode` | `align` | Same |
| `--gdn-prefill-backend` | `triton` | Same |
| CUDA graphs | vLLM's default: neither `--enforce-eager` nor `--compilation-config`, which gives `FULL_AND_PIECEWISE` with `VLLM_COMPILE` ([docs/reference.md section 4.8](reference.md#48-cuda-graphs)) | Same mode (`--cuda-graphs full`) |
| Asynchronous scheduling | vLLM's default (enabled); no flag | Differs: the launcher passes `--no-async-scheduling` unless the opt-in `--async-scheduling` is given, which this configuration does not use (A.13) |
| Attention backend | Not set; vLLM selects it (A.5) | Same |
| Thinking | Off, requested by the client (`chat_template_kwargs`, A.6) | Same effect, enforced by the server |

Every value is set explicitly, so that the control does not depend on vLLM's defaults for this hybrid model. The arguments that serve the runtime's own route (`--revision`, the reasoning and tool parsers, `--chat-template`, `--kv-transfer-config`, `--block-size`, `--mamba-block-size`, `--additional-config`) are not passed: the stock server uses the model's own chat template and returns the reply as plain `content`.

**Command**, in a shell with the environment of A.3:

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
"$VENV/bin/python" -I -B -m vllm.entrypoints.openai.api_server \
  --model "$MODELS/qwen3.6-27b" --served-model-name selective-prefix --host 127.0.0.1 --port 8000 \
  --distributed-executor-backend uni --tensor-parallel-size 1 \
  --max-num-seqs 16 --max-model-len 131072 --max-num-batched-tokens 16384 \
  --kv-cache-memory-bytes 34359738368 --seed 42 \
  --enable-prefix-caching --prefix-caching-hash-algo sha256 --language-model-only \
  --mamba-cache-mode align --gdn-prefill-backend triton
```

- **Environment.** That of A.3 (`PATH`, `LD_LIBRARY_PATH`, short `TMPDIR`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `CC` and `VLLM_RPC_BASE_PATH` unset, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`), plus the two variables that `serve.py` sets for this runtime (`VLLM_WORKER_MULTIPROC_METHOD=spawn`, `VLLM_USE_V2_MODEL_RUNNER=0`) and `VLLM_NO_USAGE_STATS=1`.
- **Generation contract, sent by the client.** Stock vLLM takes sampling from the request, so the agents' model configuration carries it (A.6): `temperature` 0.0, `max_tokens` 4096, `stop` `["</mswea_bash_command>"]` with `include_stop_str_in_output: true`, and `chat_template_kwargs: {"enable_thinking": false}`. With `enable_thinking=false` the model's chat template is expected to end the prompt with the generation header of A.3; B.3 checks this.
- **Readiness.** `GET /health` returns 200 and `GET /v1/models` lists `selective-prefix`.
- **Session close.** Stock vLLM has no `/v1/sessions/close` route. The agent pool's close requests receive HTTP 404, which is expected (A.8).
- **Preemption.** Stock vLLM can preempt requests when its KV pool is full; this runtime never preempts (A.13). Record vLLM's counter `vllm:num_preemptions_total` (`GET /metrics`) at the end of the run.

### A.5 Kernel backends

Neither server passes `--attention-backend`; vLLM chooses the attention backend automatically on the GH200. Both servers use the same kernels:

| Component | Setting | Server log line |
| --- | --- | --- |
| Attention | FLASH_ATTN with FlashAttention 3 (vLLM's automatic choice) | `Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']` and `Using FlashAttention version 3` |
| Gated-DeltaNet prefill | `--gdn-prefill-backend triton` | `Using Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128)` |
| Sampler | Native top-k/top-p (`VLLM_USE_FLASHINFER_SAMPLER=0`) | `FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0` |
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes 1, 2, 4, 8, 16, 24 and 32 at 16 sequences ([docs/reference.md section 4.8](reference.md#48-cuda-graphs)) | `'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE` and `'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32]` in the engine configuration line |
| Asynchronous scheduling | This runtime: off. Stock: on | `Asynchronous scheduling is disabled.` (this runtime) or `Asynchronous scheduling is enabled.` (stock) |

Confirm them from each server's log after READY:

```sh
grep -oE "Using FLASH_ATTN attention backend|Using FlashAttention version [0-9]+|Using Triton/FLA GDN prefill kernel \(requested=[a-z]+|FlashInfer top-p/top-k sampling disabled|Asynchronous scheduling is [a-z]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|'cudagraph_capture_sizes': \[[0-9, ]*\]" \
  server.log | sort | uniq -c
```

Every line of the table must appear, and no other attention backend, FlashAttention version, prefill kernel or CUDA-graph mode. A server whose log differs is not part of the comparison.

### A.6 Agent configuration

Identical for every server unless marked.

| Item | Value |
| --- | --- |
| Agent | Mini-SWE-Agent 2.4.2 (litellm 1.103.1) |
| Runner | The PoC's `examples/swebench/upstream/swebench_runner.py`, SHA-256 `34ca6d9288c72830e61bd00ad8b15790cab4c29515db24b8df30775cfc0880a7` |
| Prompt | The PoC's `examples/swebench/upstream/swebench_prompt_override.yaml`, SHA-256 `1b6a48666ad528775bd11225fa4d5c002721c25c0cdbc05b99cc15b2de31336a` (LF bytes) |
| Config stack | `-c swebench_xml.yaml` (built into Mini-SWE-Agent) `-c swebench_prompt_override.yaml -c model_override.yaml` |
| One runner per instance | `swebench_runner.py --subset verified --split test --workers 1 --filter '^<instance_id>$' --output <out>/runs/<instance_id>` with the config stack (`run_agents.sh` with `MAX_RUNNING`) |
| Runner dataset | `princeton-nlp/SWE-Bench_Verified` (`--subset verified`), cached and read offline |
| Model | `litellm_textbased`, `openai/selective-prefix`, `api_base` `http://127.0.0.1:<port>/v1`, dummy key, `temperature` 0.0, `max_tokens` 4096, `cost_tracking: ignore_errors`; client timeout and retries in A.7; the stock control adds the generation contract of A.4 |
| Tool environment | `docker`, image `docker.io/swebench/sweb.eval.x86_64.<instance id with "__" replaced by "_1776_", lower case>:latest`, `cwd` `/testbed`, 120 s per command, `bash -c`, `PAGER=cat`, `MANPAGER=cat`, `LESS=-R`, `PIP_PROGRESS_BAR=off`, `TQDM_DISABLE=1` |
| Process environment | `MSWEA_CTX_PATCH` unset, `AGENTDIET_MODE=off`, `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `MSWEA_GLOBAL_CONFIG_DIR` an empty directory, `PYTHONDONTWRITEBYTECODE=1`, `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1`, `REQUEST_TIMEOUT=3600` |

Properties of the PoC files that affect results ([docs/benchmarks/swe-bench-mini-swe-agent.md section 7.1](benchmarks/swe-bench-mini-swe-agent.md#71-settings)):
- When the step limit is reached, the runner submits the last `git diff` of at least 3 lines that the agent printed (exit status `LimitsExceeded_InformalSubmission`). After 10 identical commands in a row it stops the instance (`LoopDetected`).
- The prompt is written for Astropy ("The testbed is an Astropy repository", with Astropy-specific test advice). It is used unchanged for all 32 instances, the 10 Django instances included. A baseline uses the same file; a different prompt is a different configuration.

**`model_override.yaml` for this runtime**, as `run_agents.sh` renders it at port 20000 with `STEP_LIMIT=120 MAX_TOKENS=4096` from a copy of `examples/swebench/model_override.yaml` with the three client lines `timeout`, `max_retries` and `num_retries` added (B.2):

```yaml
model:
  model_name: "openai/selective-prefix"
  model_class: litellm_textbased
  cost_tracking: ignore_errors
  model_kwargs:
    api_base: "http://127.0.0.1:20000/v1"
    api_key: "dummy"
    temperature: 0.0
    max_tokens: 4096
    timeout: 3600
    max_retries: 0
    num_retries: 0

environment:
  environment_class: docker
  cwd: "/testbed"
  timeout: 120
  interpreter: ["bash", "-c"]
  env:
    PAGER: cat
    MANPAGER: cat
    LESS: -R
    PIP_PROGRESS_BAR: "off"
    TQDM_DISABLE: "1"

agent:
  step_limit: 120
  cost_limit: 0
```

**`model_override.yaml` for the stock control:** the same file with the client-side generation contract of A.4 in two more lines, after `num_retries` in the runs of record (below); `run_agents.sh` with `STOCK=1` puts the same two lines directly after `max_tokens` (B.2):

```yaml
model:
  model_name: "openai/selective-prefix"
  model_class: litellm_textbased
  cost_tracking: ignore_errors
  model_kwargs:
    api_base: "http://127.0.0.1:20000/v1"
    api_key: "dummy"
    temperature: 0.0
    max_tokens: 4096
    timeout: 3600
    max_retries: 0
    num_retries: 0
    extra_body: {"chat_template_kwargs": {"enable_thinking": false}, "include_stop_str_in_output": true}
    stop: ["</mswea_bash_command>"]

environment:
  environment_class: docker
  cwd: "/testbed"
  timeout: 120
  interpreter: ["bash", "-c"]
  env:
    PAGER: cat
    MANPAGER: cat
    LESS: -R
    PIP_PROGRESS_BAR: "off"
    TQDM_DISABLE: "1"

agent:
  step_limit: 120
  cost_limit: 0
```

The client lines ran in the runs of record ([the frozen validation record](history/swebench-validation.md)); the stock control's two contract lines ran in job 3429193.

### A.7 Limits

| Limit | Value |
| --- | --- |
| Step limit / cost limit | `step_limit` 120 / `cost_limit` 0 |
| Reply bound | 4096 new tokens per model call (this runtime: forced by the server; stock: `max_tokens` from the client) |
| Context limit | 131,072 tokens, prompt plus reply; this runtime refuses a prompt above 126,976 tokens with HTTP 400 before admission ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)) |
| Tool command timeout | 120 s per command |
| Client timeout | 3600 s per model call (`timeout: 3600`, `REQUEST_TIMEOUT=3600`) |
| Client retries | None: `max_retries: 0` and `num_retries: 0` (litellm and the OpenAI client), `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1` (Mini-SWE-Agent) |
| Requests per server | 32 instances × 120 steps = at most 3,840 requests, within this runtime's lifetime bound of max(1,024, 256 × 32) = 8,192 ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)) |

**Why no client retries and a 3600-s timeout.** On this runtime a client retry of a request that is still being served is a second request for a busy session, and a timeout or a non-2xx status ends the server's run ([docs/running.md section 6.3](running.md#63-errors)). With `--admission-overlap 16`, `goal_query` admissions can wait for the single Goal rewriter process, and the frontend waits up to 2,100 s for an admission ([docs/reference.md section 4.10](reference.md#410-admission-optimizations-opt-in)) before the reply of up to 4096 tokens starts. The client timeout of 3600 s lies above that wait plus a full reply, so slow admissions complete instead of being abandoned and retried by the client. The stock control uses the same settings so that the two sides differ only in the server.

### A.8 Concurrency, refill and session close

- At most 16 agents run at once, one runner process per instance (`--workers 1`), started in the order of A.1. When a runner ends, the next instance starts (`run_agents.sh` with `MAX_RUNNING=16`).
- On this runtime the ended instance's session is closed before its slot is reused: `POST /v1/sessions/close` with `{"instance_id": "<instance id>"}`, retried on HTTP 409 (`SESSION_BUSY`) up to 30 times at 2-s intervals ([docs/running.md section 6.5](running.md#65-session-close-session-pool)). The close frees the session's GPU and host state, and the next waiting task opens. The close time delays the next start by its own duration.
- On the stock control the same request returns HTTP 404 at once; stock vLLM keeps cached prefixes until it evicts them.
- `run_agents.sh` writes one line per instance to `timeline.jsonl` on the agent host: `instance_id`, `start` (taken just before the runner starts), `end` (taken when the runner process exits, before the close request), `runner_exit` and `close_http`. It also writes `runs/<instance>/`, `logs/<instance>.log`, `closes/<instance>.json` and a merged `preds.json`.

### A.9 Evaluation

| Item | Value |
| --- | --- |
| Evaluator | SWE-bench at commit `02e7a74` (swebench 5.0.2, A.2) on the agent (Docker) host, after every agent of the run has ended |
| Dataset | `SWE-bench/SWE-bench_Verified`, split `test`, at revision `78f471bf655a3137b2e8a75af1501690ec009ec3`, the revision the evaluations of the runs of record used (`evaluate.sh`). `evaluate.sh` fetches that revision's test split file, checks its SHA-256 (`030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25`) and passes it to the evaluator as a local file |
| Command | `python -m swebench.harness.run_evaluation --dataset_name <eval dir>/dataset.parquet --split test --predictions_path <eval dir>/predictions.jsonl --max_workers <WORKERS> --timeout 1800 --run_id <run_id>` (`examples/swebench/evaluate.sh`, [docs/benchmarks/swe-bench-mini-swe-agent.md section 7.5](benchmarks/swe-bench-mini-swe-agent.md#75-evaluate)). The runs of record ran the same evaluator from their own harness, with `--dataset_name SWE-bench/SWE-bench_Verified` and `--max_workers 4` |
| Timeout | `--timeout 1800` per instance |
| Workers | `--max_workers 4` in the runs of record (`WORKERS=4` in B.8; `evaluate.sh`'s default is 2). The worker count sets only how many instances are evaluated in parallel; any count gives the same results |
| Predictions | One row per planned instance (32). An instance without a prediction is evaluated with an empty patch and counts as unresolved |

### A.10 Metrics

**Accuracy** = resolved instances / 32 × 100, from the evaluator's `resolved_ids`. One instance is 3.125 points. At an accuracy of 75 % the binomial standard error of one run is about 7.7 points, and that of a difference between two independent runs about 10.8 points (3.5 instances). Keep the per-instance resolved lists so that runs can be compared instance by instance.

**Throughput** = total completion tokens / wall time:

| Part | Definition |
| --- | --- |
| Numerator | The sum of `usage.completion_tokens` over every model call of the main agents of all 32 instances, as recorded in the trajectories on the client side (`extra.response.usage.completion_tokens`). The calls are the assistant messages and the format-error messages (`extra.interrupt_type == "FormatError"`) |
| Denominator | Wall time on the agent host's clock from the first agent's start to the last agent's end: `max(end) − min(start)` over `timeline.jsonl` (A.8) |
| Included | Tool execution, task-container start and stop, waiting for a free agent slot or a session, the session closes that precede each refill (A.8), admission waits, the Goal rewriter's time (`goal_query`) |
| Excluded | Server start-up and shutdown, evaluation, the session close that follows the last agent's end, and the Goal rewriter's own tokens (an auxiliary model) |

Record also, per run: per-instance wall time, steps and exit status; the number of context-limit refusals; for stock, vLLM's preemption counter (A.4).

### A.11 Run rules

- **One run at a time on the agent host**, and no evaluation while agents run: throughput includes tool time, so CPU contention changes it. Use the same agent host type for every server whose throughput is compared.
- **A fresh server and a new output directory for every attempt.** `serve.py`, `run_agents.sh` and `evaluate.sh` refuse an existing output directory.
- **No resume across interrupted runs.** A run that ends early (time limit, tunnel drop, server failure) is partial: its unfinished instances are not continued in another job, the run is excluded from comparisons, and the whole server reruns from the start under a new label.
- **Failures.** A request refused with HTTP 400 because its prompt exceeds the context limit ends that instance only (unresolved); the server keeps serving ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)). Any other non-2xx status, a timeout, a disconnect or a server exit ends the run ([docs/running.md section 6.3](running.md#63-errors)), which is then partial.
- **Transport.** The agents reach the server through a loopback-only route (B.4). Any loopback-only route gives the same results; the server never listens beyond loopback ([docs/running.md section 6.4](running.md#64-reaching-the-server-from-another-host)).

### A.12 Alignment checklist

| # | Item | This runtime | A baseline uses |
| --- | --- | --- | --- |
| 1 | Model and revision | Qwen3.6-27B `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | The same |
| 2 | Software | vLLM 0.24.0 official wheel, Transformers 5.13.0, Tokenizers 0.22.2, `constraints-3270721.txt` | The same environment |
| 3 | Hardware and job | 1 × GH200 120GB, 16 CPUs, 113,560 MiB | The same |
| 4 | Instances | The 32 IDs of A.1, in that order | The same |
| 5 | Concurrency | At most 16 running of 32, refilled in list order | The same |
| 6 | Context limit | 131072 | `--max-model-len 131072` |
| 7 | Batched tokens per step | 16384 | `--max-num-batched-tokens 16384` |
| 8 | Maximum sequences | 16 | `--max-num-seqs 16` |
| 9 | GPU KV | 32 GiB (34,359,738,368 bytes) | `--kv-cache-memory-bytes 34359738368` |
| 10 | Reply bound | 4096 (server) | `max_tokens` 4096 (client) |
| 11 | Decoding | Greedy (server) | `temperature` 0.0 |
| 12 | Thinking | Off (`enable_thinking=false`, empty think block in the header) | `chat_template_kwargs: {"enable_thinking": false}` |
| 13 | Stop | `</mswea_bash_command>`, kept in the output | `stop` plus `include_stop_str_in_output: true` |
| 14 | CUDA graphs | `FULL_AND_PIECEWISE` (`--cuda-graphs full`) | vLLM's default (`FULL_AND_PIECEWISE`) |
| 15 | Asynchronous scheduling | Off (the default; the opt-in `--async-scheduling` is not used) | vLLM's default (on); a known difference (A.13) |
| 16 | Attention | FLASH_ATTN with FlashAttention 3, chosen by vLLM | The same, no `--attention-backend` |
| 17 | Gated-DeltaNet prefill | Triton | `--gdn-prefill-backend triton` |
| 18 | Sampler | Native (`VLLM_USE_FLASHINFER_SAMPLER=0`) | The same variable |
| 19 | Prefix caching | On, sha256 | `--enable-prefix-caching --prefix-caching-hash-algo sha256` |
| 20 | Mamba cache mode | `align` | `--mamba-cache-mode align` |
| 21 | Seed | 42 | `--seed 42` |
| 22 | Text only | `--language-model-only` | vLLM's `--language-model-only` |
| 23 | Agent | Mini-SWE-Agent 2.4.2, litellm 1.103.1 | The same |
| 24 | Runner and prompt | The PoC files, SHA-256 `34ca6d92…` and `1b6a4866…` (the Astropy prompt for all 32 instances) | The same files |
| 25 | Config stack | `swebench_xml.yaml`, prompt override, model override | The same |
| 26 | Step limit / cost limit | 120 / 0 | The same |
| 27 | Tool environment | Docker, `/testbed`, 120 s, the variables of A.6 | The same |
| 28 | Process environment | `AGENTDIET_MODE=off`, offline caches, an empty `MSWEA_GLOBAL_CONFIG_DIR` | The same |
| 29 | Client timeout and retries | 3600 s, no retries | The same |
| 30 | Agent host | One run at a time, no evaluation during a run | The same rule and the same host type |
| 31 | Evaluation | SWE-bench at `02e7a74` (swebench 5.0.2), `SWE-bench/SWE-bench_Verified`, `test`, `--timeout 1800`, 4 workers | The same |
| 32 | Throughput | The definition of A.10 | The same numerator and clock |
| 33 | Partial runs | Excluded, never resumed | The same |

### A.13 Known differences between this runtime and stock vLLM

These differences remain when every checklist item matches. Report them with any comparison.

1. **Previous replies.** This runtime replays each previous reply as the exact token IDs it was generated with; stock vLLM re-renders the history from text through the chat template ([docs/running.md section 6.1](running.md#61-mini-swe-agent-xml-protocol)).
2. **Asynchronous scheduling.** This runtime runs without it here: the launcher passes `--no-async-scheduling` unless the opt-in `--async-scheduling` is given (not used in this configuration of record; the current configuration uses it, [docs/status.md](status.md#current-configuration)). Stock vLLM runs with its default, enabled.
3. **Sampling control.** This runtime enforces greedy decoding, the reply bound and the stop marker in the engine and ignores the client's fields; stock vLLM takes them from the request. At the marker, this runtime returns everything the final token decodes to, including characters past the marker ([docs/running.md section 6.1](running.md#61-mini-swe-agent-xml-protocol)); stock vLLM cuts the text at the stop string.
4. **Context edge.** This runtime refuses a prompt above 126,976 tokens with HTTP 400 ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)); stock vLLM shortens the reply to the room left in the context instead ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound) describes this vLLM behaviour; not measured for the stock control).
5. **KV management.** Both have 32 GiB of GPU KV, in different layouts. This runtime uses 16-token attention pages, a separate Gated-DeltaNet pool, a 1-GiB CPU pool and a 70-GiB pinned page arena; it never preempts, and a turn that does not fit waits (`CAPACITY_WAIT`, [docs/running.md section 4.4](running.md#44-several-sessions-on-one-server)). At 131,072 tokens, 16 `full` sessions cannot all be held at full context in 32 GiB ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)), so `full` turns can wait. Stock vLLM uses its own hybrid layout ([docs/reference.md section 4.6](reference.md#46-decoupled-kv-layout-and-16-token-pages)) and can preempt when the pool is full.
6. **Admission and scheduling.** This runtime admits one request at a time in arrival order (`goal_query`: up to 16 in flight with `--admission-overlap 16`), runs each admission's phase 1 as its own forward pass, then decodes in batches ([docs/running.md section 4.4](running.md#44-several-sessions-on-one-server), [docs/reference.md section 4.10](reference.md#410-admission-optimizations-opt-in), [docs/design.md section 5](design.md#5-page-selection-for-the-query-policies)). Stock vLLM schedules prefill chunks together with decode steps within the 16,384-token step budget.
7. **Session close.** This runtime frees a session's state when its agent ends; stock vLLM keeps cached prefixes until it evicts them.
8. **Rebuilds.** On this runtime a request that does not extend the previous one (for example a format-error retry that drops the rejected reply) starts a fresh session re-rendered from text ([docs/running.md section 6.1](running.md#61-mini-swe-agent-xml-protocol)); stock vLLM serves it with prefix caching.
9. **Selective policies.** `observation_query`, `goal_query`, `random_fill`, `recent_fill`, `recent_tools` and `akv_rule` attend to a selected subset of history pages by design; `full` keeps every page.
10. **Goal rewriter.** `goal_query` runs an auxiliary Qwen3-0.6B on the same GPU. Its time counts in the throughput, its tokens do not.
11. **Run-to-run variation.** Batched runs are not token-identical from run to run on either server: a reply can change with the batch composition, which depends on timing ([docs/running.md section 4.4](running.md#44-several-sessions-on-one-server)). Between two 10-instance pilots, `full` changed its outcome on 2 of 10 instances ([docs/reference.md section 4.7](reference.md#47-validation-at-16-token-pages)).

### A.14 Gemma 4 31B IT

The same benchmark with `google/gemma-4-31B-it` at revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, on a release installed with `install.py --model-profile gemma4-31b-it` ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)). The instances, concurrency, agent configuration and model overrides, limits, evaluation, metrics and run rules of A.1 and A.6 to A.11 are unchanged. The runtime serves the XML protocol with Gemma's turns (`mswea_xml_gemma4_v1`) on the profile's window layout: the global layers' 16-token Pages (1.25 MiB each) in pool 0, the sliding windows in pool 1, and no Gated-DeltaNet state; `--kv-layout` is not passed ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)). Its GPU runs, by policy: [docs/status.md](status.md).

**Flags shared by the four policies**, where they differ from A.3:

| Flag | Value | Effect |
| --- | --- | --- |
| `--kv-cache-gib` | `25` for `observation_query`, `random_fill` and `goal_query`; `32` for `full` (per policy, below) | 25 GiB is the KV cache of the group's Gemma baselines. `full` sizes its global pool by the release rule, which needs at least 30,738 MiB at 16 sessions and 131,072 tokens ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)) |
| `--page-tokens` | `16` | The Gemma profile's window layout ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)). vLLM gets `--block-size 16`, `--attention-backend TRITON_ATTN`, `--max-num-batched-tokens 8192`, the `gemma4` parsers and the layout's `--additional-config`; the dry run prints the list |
| `--dram-arena-gib` | `72` | One pinned arena of 58,982 Pages of 1.25 MiB; with the 1-GiB connector pool 73.22 GiB pinned, within the Gemma launcher's 80.9-GiB budget for one GH200 share and within `--mem=113560M` ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)) |
| `--release-idle-pages` | on | Sixteen Gemma sessions at 131,072 tokens need it: under GPU pressure the gate releases idle sessions' global Pages and readmits them before their next phase 1; the windows stay on the GPU (docs/reference.md sections [4.13](reference.md#413-releasing-idle-sessions-pages-opt-in), [4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)). No vLLM argument |

The other shared flags are those of A.3: `--task`, `--max-running-sessions 16`, `--max-model-len 131072`, `--max-tokens 4096`, `--connector-pool-gib 1`, `--cuda-graphs full`, the three decode fast paths, `--deferred-completion-io` and `--language-model-only` (required by the profile).

**Flags per policy:**

| Policy | `--kv-cache-gib` | Selector flags | Admission flags | Goal rewriter |
| --- | --- | --- | --- | --- |
| `full` | `32` | none | none | none |
| `observation_query` | `25` | `--active-token-budget 8192` | none | not started |
| `random_fill` | `25` | `--active-token-budget 8192` | none | not started |
| `goal_query` | `25` | `--active-token-budget 8192 --goal-rewriter-model <Qwen3-0.6B directory> --goal-rewriter-batch 4 --goal-rewriter-batch-tokens 15360` | `--admission-overlap 16` | One Qwen3-0.6B process on the same GPU. The per-batch token cap is explicit: by the harness's GPU estimate calibrated on job 3410948 (its rule leaves at least 2 GiB, the 1.3× allocator slack not counted), 15,360 leaves 2,200 MiB of the 97,871 MiB free at the peak (95,671 MiB; 1,390 MiB with the slack), while the profile's default at 25 GiB, 25,600 ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)), would leave 80 MiB (−1,366 MiB with the slack) |

**Commands.** Dry runs of the four servers (add `--start` to serve; `TASKS` and `LABEL` as in A.3):

```sh
SHARED=(--max-running-sessions 16 --page-tokens 16 --max-model-len 131072 --max-tokens 4096
        --dram-arena-gib 72 --connector-pool-gib 1 --cuda-graphs full --decode-fastpath --decode-fastpath2
        --decode-fastpath3 --deferred-completion-io --language-model-only --release-idle-pages)
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

**4,096-token forwards (opt-in).** Adding `--max-num-batched-tokens 4096` to `SHARED` gives every server the group's 4,096-token forward ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d), "4,096-token forwards"): vLLM gets `--max-num-batched-tokens 4096`, the window layout reserves one 4,096-token forward's transient (pool 1 3.125 GiB smaller; 2,560 more global Pages, 7,518 at 25 GiB for `observation_query` and `random_fill`, 6,238 for `goal_query`), and a phase 1 longer than 4,096 tokens runs in chunks of 3,056. Give the stock control `--max-num-batched-tokens 4096` too. The configuration of record uses 8,192; the GPU run of this setting (job 3462014, `observation_query`) is in [docs/status.md](status.md), and the frozen [validation record](history/swebench-validation.md) says it ran 4,096-token forwards.

**Stock vLLM control**, with stage 0's Gemma flags (the stock server of the stage-0 agent run, job 3364911). Where it differs from A.4:

| Setting | Value | Relation to this runtime |
| --- | --- | --- |
| `--model` | The Gemma 4 31B IT directory | Same |
| `--attention-backend` | `TRITON_ATTN` | Same backend (the Gemma profile pins it) |
| `--dtype` | `bfloat16` | The checkpoint's dtype |
| KV cache | vLLM's own memory profile: no `--kv-cache-memory-bytes` (26.41 GiB of KV in job 3364911) | Differs: this runtime's servers have 25 or 32 GiB |
| `--max-num-batched-tokens` | `8192` | Same as the Gemma profile's one native forward |
| Prefix caching | `--enable-prefix-caching --prefix-caching-hash-algo sha256`, with `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` | As A.4 |
| `--enable-prompt-tokens-details` | on | Stage 0's setting |
| Parsers | `--reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice` | Stage 0's setting |
| `--mamba-cache-mode`, `--gdn-prefill-backend` | not passed | Gemma has no Gated-DeltaNet state |

The other settings are those of A.4: `--served-model-name selective-prefix`, loopback on port 8000, `uni` on one GPU, `--max-num-seqs 16`, `--max-model-len 131072`, `--seed 42`, `--language-model-only`, and vLLM's defaults for CUDA graphs and asynchronous scheduling; the client sends A.4's generation contract (the stock model override of A.6).

**Command**, in a shell with the environment of A.3:

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0
"$VENV/bin/python" -I -B -m vllm.entrypoints.openai.api_server \
  --model "$MODELS/gemma-4-31b-it" --served-model-name selective-prefix --host 127.0.0.1 --port 8000 \
  --distributed-executor-backend uni --tensor-parallel-size 1 --dtype bfloat16 --attention-backend TRITON_ATTN \
  --max-num-seqs 16 --max-model-len 131072 --max-num-batched-tokens 8192 --seed 42 \
  --enable-prefix-caching --prefix-caching-hash-algo sha256 --enable-prompt-tokens-details \
  --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice --language-model-only
```

**Kernels.** Both servers run `TRITON_ATTN` (FlashAttention 3 has no 512-wide heads, [docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)); there is no Gated-DeltaNet prefill kernel. The sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`), the CUDA-graph mode (`FULL_AND_PIECEWISE`, capture sizes 1, 2, 4, 8, 16, 24 and 32 at 16 sequences) and the scheduling lines are those of A.5; the server log must name `TRITON_ATTN` as the attention backend.

**Known differences** beyond A.13:
- **Previous replies.** As A.13 item 1, with Gemma's empty thought channel (`<|channel>thought\n<channel|>`, 4 tokens per past reply) in place of the think block ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)).
- **KV size.** The stock control's KV cache is vLLM's own memory profile (26.41 GiB in stage 0), not 25 or 32 GiB.
- **KV management.** This runtime keeps the sliding windows on the GPU and releases idle sessions' global Pages under pressure (`--release-idle-pages`); stock vLLM manages Gemma's KV in its own layout and can preempt.

The other items of A.13 apply as they are.

---

## Part B. Procedure

This part runs the configuration of Part A with the tools of this repository: the model on a GH200 node in a Slurm job, the agents and their task containers on a separate x86_64 Docker host (the agent host), and a reverse SSH tunnel from the GPU job to the agent host ([docs/running.md section 6.4](running.md#64-reaching-the-server-from-another-host)). Every step applies to the four policies and to the stock control unless marked.

### B.1 Agent host

- **Requirements.** Linux x86_64 with Docker (usable without `sudo`), `bash`, `curl`, `ss` (iproute2), `python3`, `sha256sum` (coreutils), `setsid` (util-linux), `git` (uv installs the evaluator from its git URL) and `uv`, an agent environment and an evaluator environment with the versions of A.2, and outbound network access while the caches are filled (the datasets, the evaluation split file, the images). The host runs up to 16 runner processes and 16 task containers at once. Inbound SSH from the GPU nodes is needed for the tunnel (B.4).
- **Image naming.** One image per instance: `docker.io/swebench/sweb.eval.x86_64.<instance id with "__" replaced by "_1776_", lower case>:latest`, for example `docker.io/swebench/sweb.eval.x86_64.django_1776_django-10097:latest`. `examples/swebench/verified-32.txt` pins each instance's image by digest (`<instance_id> <image>@sha256:<digest>`, the 32 instances of A.1 in order); `run_agents.sh` and `evaluate.sh` refuse to start unless each local `:latest` tag is its pinned image (`images.py`).

Environments and directories (the commands are not part of the validated runs; [docs/benchmarks/swe-bench-mini-swe-agent.md section 7.2](benchmarks/swe-bench-mini-swe-agent.md#72-agent-host)), with a checkout of this repository at `$AGENT/vllm-0.24.0-selective-prefix-policies`:

```sh
AGENT=/abs/path/on/agent
REPO_A=$AGENT/vllm-0.24.0-selective-prefix-policies
MINI=$AGENT/mini-env
EVAL=$AGENT/eval-env
uv venv --python 3.12 "$MINI"
uv pip install --python "$MINI/bin/python" mini-swe-agent==2.4.2 litellm==1.103.1 datasets==5.0.1
uv venv --python 3.12 "$EVAL"
uv pip install --python "$EVAL/bin/python" \
  "swebench @ git+https://github.com/SWE-bench/SWE-bench@02e7a74ffd0b707aab73d203fe87bdc7c76afc8e" datasets==5.0.1
mkdir -p "$AGENT/sock" "$AGENT/runs" "$AGENT/eval" "$AGENT/mswea-config"
chmod 700 "$AGENT/sock"
```

The evaluator is SWE-bench at commit `02e7a74` (A.2), not `swebench==5.0.2` from PyPI: `evaluate.sh` passes the dataset as a local Parquet file and refuses an evaluator whose `load_swebench_dataset` reads none.

Save the instance list of A.1 as `$AGENT/ids.txt`, then pull the 32 images by digest and tag each as the agents and the evaluator name it, before any run, so that no pull enters the measured time. The last command checks them as `run_agents.sh` does: exit 0 when every image is the pinned one (its output file must not exist yet); otherwise it prints the `docker pull` and `docker tag` commands for each image that is missing or not the pinned one:

```sh
cut -d ' ' -f 1 "$REPO_A/examples/swebench/verified-32.txt" > "$AGENT/ids.txt"
while read -r instance image; do
  docker pull "$image"
  docker tag "$image" "${image%@*}:latest"
done < "$REPO_A/examples/swebench/verified-32.txt"
python3 -I -B "$REPO_A/examples/swebench/images.py" "$REPO_A/examples/swebench/verified-32.txt" \
  "$AGENT/images-check.txt" $(cat "$AGENT/ids.txt")
```

### B.2 Task list and agent files

Build `tasks-pool.json` from the runner's dataset at its pinned revision, with the 32 statements verbatim and in order. `HF_DATASETS_CACHE` is the agents' datasets cache, a new directory; the first load needs the network, so the `unset` line undoes any offline switches of the shell:

```sh
export HF_DATASETS_CACHE=$AGENT/swebench-datasets   # new: the "mkdir" line creates it
mkdir "$HF_DATASETS_CACHE"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
mapfile -t IDS < "$AGENT/ids.txt"
"$MINI/bin/python" "$REPO_A/examples/swebench/build_tasks.py" --output-dir "$AGENT/tasks" --pool "${IDS[@]}"
```

`build_tasks.py` loads `princeton-nlp/SWE-Bench_Verified` at its pinned revision (`REVISION`, `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`) into that cache, checks each problem statement against its SHA-256 pin (all 32 are pinned), writes `tasks-pool.json` (the 32 tasks in this order, the order in which the agents start) and one `task-<instance_id>.json` per instance, never overwrites a file, and prints a JSON report. Copy `tasks-pool.json` to the GPU side (`TASKS` in B.5). `run_agents.sh` requires the same directory (`HF_DATASETS_CACHE` in `AGENT_ENV`, B.5; an existing directory, by its absolute path): Mini-SWE-Agent loads the dataset offline and without a revision, so it takes what this cache holds. Keep the cache for this revision only.

`run_agents.sh` renders the model override from the `model_override.yaml` next to it, and the template has no client timeout or retry lines. Make one copy of `examples/swebench` for the runs and add the three client lines of A.6 after the template's `max_tokens` line (`render_override.py` keeps added lines and replaces only the port, `max_tokens` and `step_limit` lines). The two renderings check the result, for this runtime's servers (`STOCK` 0) and for the stock control (`STOCK` 1):

```sh
cp -r "$REPO_A/examples/swebench" "$AGENT/swebench-client"
python3 - "$AGENT/swebench-client/model_override.yaml" <<'PY'
import sys
anchor = '    max_tokens: 1536\n'
client = '    timeout: 3600\n    max_retries: 0\n    num_retries: 0\n'
with open(sys.argv[1], newline='') as stream:
    text = stream.read()
assert text.count(anchor) == 1, sys.argv[1]
with open(sys.argv[1], 'w', newline='') as stream:
    stream.write(text.replace(anchor, anchor + client))
PY
python3 -I -B "$AGENT/swebench-client/render_override.py" "$AGENT/swebench-client/model_override.yaml" 20000 4096 120 0
python3 -I -B "$AGENT/swebench-client/render_override.py" "$AGENT/swebench-client/model_override.yaml" 20000 4096 120 1
```

The first rendered file must equal A.6's file for this runtime. The second has the lines of A.6's stock file, with the two contract lines directly after `max_tokens` instead of after `num_retries`: the same keys and values. `run_agents.sh` renders that file when it runs with `STOCK=1` (B.5 passes it for the stock control), and then also counts a session close answered HTTP 404 as closed (B.7). Do not add the two contract lines to the template as well: `STOCK=1` would add them a second time. The copy keeps `upstream/` with the pinned PoC files, which `run_agents.sh` checks before every run, and `images.py` with `verified-32.txt`, which it runs before anything starts.

### B.3 Environment check before GPU time

**Gold patches.** Evaluating the reference patches of the 32 instances checks the images, the evaluator and the agent host without a model; every instance is expected to resolve, and any that does not is investigated before a GPU job. The evaluator writes `logs/` under its working directory, so run it in a new directory. The commands take the evaluation dataset as `evaluate.sh` does (A.9: the pinned revision's test split file, its SHA-256 checked). Untested; confirm the options with `--help` of the evaluator:

```sh
mkdir "$AGENT/eval/gold-check" && cd "$AGENT/eval/gold-check"
"$EVAL/bin/python" -I -B -c 'import shutil, sys; from huggingface_hub import hf_hub_download; shutil.copyfile(hf_hub_download(sys.argv[1], sys.argv[3], repo_type="dataset", revision=sys.argv[2]), sys.argv[4])' \
  SWE-bench/SWE-bench_Verified 78f471bf655a3137b2e8a75af1501690ec009ec3 data/test-00000-of-00001.parquet \
  "$AGENT/eval/gold-check/dataset.parquet"
echo "030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25  dataset.parquet" | sha256sum --check
"$EVAL/bin/python" -m swebench.harness.run_evaluation --dataset_name "$AGENT/eval/gold-check/dataset.parquet" \
  --split test --predictions_path gold --instance_ids $(cat "$AGENT/ids.txt") \
  --max_workers 4 --timeout 1800 --run_id gold-check
```

**Thinking-off header of the stock control.** On the GPU side, with the environment of A.3, the model's chat template with `enable_thinking=false` must end the prompt with the generation header of A.3 (untested):

```sh
"$VENV/bin/python" -I -B -c "
import sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'x'}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
print(repr(prompt[-48:]))" "$MODELS/qwen3.6-27b"
```

The printed text must end with `<|im_start|>assistant\n<think>\n\n</think>\n\n`.

### B.4 Tunnel

The GPU job opens a reverse SSH tunnel after READY whose far end is a Unix socket on the agent host; `run_agents.sh` exposes it at `127.0.0.1:<port>` through `examples/swebench/relay.py` and refuses to run if anything but `127.0.0.1` listens on that port ([docs/running.md section 6.4](running.md#64-reaching-the-server-from-another-host)). The same form serves the stock control. It needs:
- a key that the agent host accepts for the tunnel user, readable by the GPU job, and a known-hosts file that pins the agent host's key (the job never uses an SSH agent);
- outbound SSH from the compute nodes to the agent host (check once from a compute node before the first run);
- a new socket path for every server, within the AF_UNIX limit of 107 bytes (`examples/slurm/serve_gh200.sh`), in a `0700` directory on the agent host (`$AGENT/sock`).

A tunnel drop ends the run (A.11); the job does not reconnect.

### B.5 Slurm job script

The script below runs one server of the benchmark in one Slurm job: the server, READY, the kernel lines, the tunnel, the agent pool on the agent host over SSH, and the shutdown. It is derived from `examples/slurm/serve_gh200.sh`, which does not pass `--cuda-graphs`, `--active-token-budget`, `--connector-pool-gib`, the decode fast paths, `--deferred-completion-io`, `--language-model-only` or the admission flags of A.3 (docs/reference.md sections [4.10](reference.md#410-admission-optimizations-opt-in), [4.12](reference.md#412-text-only---language-model-only-on-by-default)). It is an example and **untested**. Save it as `swebench_job.sh`:

```bash
#!/usr/bin/env bash
# One server of the SWE-bench Verified benchmark (docs/SWEBENCH_BENCHMARK.md) in one Slurm job.
# POLICY: full | observation_query | goal_query | random_fill (this runtime; the configuration)
#         | recent_fill | recent_tools | akv_rule (this runtime; not in the configuration) | stock (stock vLLM 0.24.0).
set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}" || { echo "run this under sbatch" >&2; exit 2; }
: "${REPO:?}" "${VENV:?}" "${TASKS:?}" "${POLICY:?}" "${RUN_ROOT:?}" "${AGENT_SECONDS:?}"
: "${AGENT_HOST:?}" "${TUNNEL_KEY:?}" "${KNOWN_HOSTS:?}" "${AGENT_SOCKET:?}" "${AGENT_PORT:?}"
: "${AGENT_RUN:?}" "${AGENT_OUT:?}" "${AGENT_ENV:?}"
PORT=${PORT:-8000}; READY_SECONDS=${READY_SECONDS:-900}

# Job-local directories and the environment of docs/install.md 3.
STAGE="$RUN_ROOT/$SLURM_JOB_ID"
mkdir "$STAGE" "$STAGE/tmp" "$STAGE/home" "$STAGE/cache"
(( ${#STAGE} + 4 + 37 <= 107 )) || { echo "TMPDIR too long for AF_UNIX: $STAGE/tmp" >&2; exit 2; }
export TMPDIR="$STAGE/tmp" TMP="$STAGE/tmp" TEMP="$STAGE/tmp"
export HOME="$STAGE/home" XDG_CACHE_HOME="$STAGE/cache"
export HF_HOME="$STAGE/cache/hf" TORCH_HOME="$STAGE/cache/torch"
export TRITON_CACHE_DIR="$STAGE/cache/triton" CUDA_CACHE_PATH="$STAGE/cache/cuda"
export TORCH_EXTENSIONS_DIR="$STAGE/cache/torch-extensions"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
unset CC VLLM_RPC_BASE_PATH SSH_AUTH_SOCK SSH_AGENT_PID PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_PRELOAD
export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
PY="$VENV/bin/python"

# The server: Part A.3 (this runtime) or Part A.4 (stock).
SHARED=(--max-running-sessions 16 --kv-cache-gib 32 --page-tokens 16 --kv-layout decoupled
        --max-model-len 131072 --max-tokens 4096 --dram-arena-gib 70 --connector-pool-gib 1 --cuda-graphs full
        --decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io --language-model-only)
AGENT_STOCK=0                                     # run_agents.sh STOCK: 1 for the stock control only (B.2)
case $POLICY in
  full)                          POLICY_FLAGS=() ;;
  observation_query|random_fill|recent_fill|recent_tools) POLICY_FLAGS=(--active-token-budget 8192) ;;
  akv_rule)                      POLICY_FLAGS=() ;;
  goal_query)                    POLICY_FLAGS=(--active-token-budget 8192 --goal-rewriter-model "${REWRITER:?}"
                                               --goal-rewriter-batch 4 --admission-overlap 16) ;;
  stock)                         POLICY_FLAGS=(); AGENT_STOCK=1 ;;
  *) echo "unknown POLICY: $POLICY" >&2; exit 2 ;;
esac
OUTPUT="$STAGE/$POLICY"
if [[ $POLICY == stock ]]; then
  : "${MODEL:?}"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
  "$PY" -I -B -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name selective-prefix --host 127.0.0.1 --port "$PORT" \
    --distributed-executor-backend uni --tensor-parallel-size 1 \
    --max-num-seqs 16 --max-model-len 131072 --max-num-batched-tokens 16384 \
    --kv-cache-memory-bytes 34359738368 --seed 42 \
    --enable-prefix-caching --prefix-caching-hash-algo sha256 --language-model-only \
    --mamba-cache-mode align --gdn-prefill-backend triton > "$STAGE/server.log" 2>&1 &
else
  : "${RELEASE:?}"
  SERVE=("$PY" -B "$REPO/serve.py" --release-dir "$RELEASE" --task "$TASKS" --policy "$POLICY"
         --output "$OUTPUT" --port "$PORT" "${SHARED[@]}" ${POLICY_FLAGS[@]+"${POLICY_FLAGS[@]}"})
  "${SERVE[@]}" > "$STAGE/dry-run.json"          # dry run: checks the release, prints the vLLM command
  "${SERVE[@]}" --start > "$STAGE/server.log" 2>&1 &
fi
SERVER=$!                                         # the APIServer PID (serve.py execs it)
TUNNEL=""
stop_tunnel() { if [[ -n $TUNNEL ]]; then kill -TERM "$TUNNEL" 2>/dev/null || true; wait "$TUNNEL" 2>/dev/null || true; TUNNEL=""; fi; }
trap stop_tunnel EXIT

# READY: ready.json (this runtime, docs/running.md 4.1) or /health and /v1/models (stock).
ready() {
  if [[ $POLICY == stock ]]; then
    local models
    curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" > /dev/null || return 1
    models=$(curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models") || return 1
    [[ $models == *'"selective-prefix"'* ]] || return 1
    return 0
  fi
  "$PY" -B "$REPO/examples/slurm/server_checks.py" ready "$OUTPUT" "$POLICY" "$RELEASE" > "$STAGE/ready-check.txt"
}
deadline=$((SECONDS + READY_SECONDS))
while true; do
  set +e; ready; rc=$?; set -e
  (( rc == 0 )) && break
  (( rc == 1 )) || { cat "$STAGE/ready-check.txt" >&2; kill -TERM "$SERVER"; wait "$SERVER" || true; exit 1; }
  kill -0 "$SERVER" 2>/dev/null || { echo "server exited before READY" >&2; wait "$SERVER" || true; exit 1; }
  (( SECONDS < deadline )) || { echo "no READY within $READY_SECONDS s" >&2; kill -TERM "$SERVER"; wait "$SERVER" || true; exit 1; }
  sleep 5
done
echo "READY after $SECONDS s"
grep -oE "Using FLASH_ATTN attention backend|Using FlashAttention version [0-9]+|Using Triton/FLA GDN prefill kernel \(requested=[a-z]+|FlashInfer top-p/top-k sampling disabled|Asynchronous scheduling is [a-z]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|'cudagraph_capture_sizes': \[[0-9, ]*\]" \
  "$STAGE/server.log" | sort | uniq -c > "$STAGE/kernels.txt" || true

# Reverse tunnel: a Unix socket on the agent host -> this node's 127.0.0.1:$PORT (docs/running.md 6.4).
SSH_OPTS=(-i "$TUNNEL_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes
          -o UserKnownHostsFile="$KNOWN_HOSTS" -o GlobalKnownHostsFile=/dev/null -o ConnectTimeout=20)
ssh -N -R "$AGENT_SOCKET:127.0.0.1:$PORT" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 "${SSH_OPTS[@]}" "$AGENT_HOST" > "$STAGE/tunnel.log" 2>&1 &
TUNNEL=$!
up=0
for _ in $(seq 1 30); do
  kill -0 "$TUNNEL" 2>/dev/null || break
  if ssh -T "${SSH_OPTS[@]}" "$AGENT_HOST" curl -sf --max-time 5 --unix-socket "$AGENT_SOCKET" \
       http://localhost/v1/models 2>/dev/null | grep -q '"selective-prefix"'; then up=1; break; fi
  sleep 2
done

# The agent pool (Part A.6 to A.8): 16 running of the 32 instances, refilled in list order.
agents_rc=0
if (( up == 1 )); then
  INSTANCES=$("$PY" -B -c 'import json,sys; print(" ".join(t["instance_id"] for t in json.load(open(sys.argv[1]))))' "$TASKS")
  timeout "$AGENT_SECONDS" ssh -T "${SSH_OPTS[@]}" "$AGENT_HOST" env $AGENT_ENV STOCK=$AGENT_STOCK MAX_RUNNING=16 \
    STEP_LIMIT=120 MAX_TOKENS=4096 "$AGENT_RUN" "$AGENT_SOCKET" "$AGENT_PORT" "$AGENT_OUT" $INSTANCES \
    > "$STAGE/agents.log" 2>&1 || agents_rc=$?
  if (( agents_rc == 124 )); then echo "agent time limit reached: the run is partial (B.7)" >&2; fi
else
  echo "tunnel not up (see $STAGE/tunnel.log)" >&2; agents_rc=1
fi
stop_tunnel

# Stop: SIGTERM to the server PID only; this runtime must exit 0 with clean receipts (docs/design.md 8).
kill -TERM "$SERVER"
set +e; wait "$SERVER"; server_rc=$?; set -e
echo "server exit $server_rc, agents exit $agents_rc"
if [[ $POLICY != stock ]]; then
  (( server_rc == 0 )) || exit 1
  "$PY" -B "$REPO/examples/slurm/server_checks.py" cleanup "$OUTPUT"
fi
exit "$agents_rc"
```

Submit one job per server, with the Quickstart placeholders of the README and settings in the submitting shell (`--export=ALL`). The example below submits `observation_query`:

```sh
LABEL=a1                                        # a new label for every attempt
POLICY=observation_query
export REPO VENV RELEASE POLICY
export MODEL="$MODELS/qwen3.6-27b" REWRITER="$MODELS/Qwen3-0.6B-c1899de"
export TASKS=/abs/path/to/tasks/tasks-pool.json RUN_ROOT=/abs/short/jobs AGENT_SECONDS=4800
export AGENT_HOST=user@agent-host TUNNEL_KEY=/abs/path/to/tunnel_key KNOWN_HOSTS=/abs/path/to/agent_known_hosts
export AGENT_SOCKET="/abs/path/on/agent/sock/$LABEL-$POLICY.sock" AGENT_PORT=20000
export AGENT_RUN=/abs/path/on/agent/swebench-client/run_agents.sh AGENT_OUT="/abs/path/on/agent/runs/$POLICY-$LABEL"
export AGENT_ENV="MINI_PYTHON=/abs/path/on/agent/mini-env/bin/python MSWEA_CONFIG_DIR=/abs/path/on/agent/mswea-config"
export AGENT_ENV="$AGENT_ENV HF_DATASETS_CACHE=/abs/path/on/agent/swebench-datasets"   # the datasets cache of B.2
export AGENT_ENV="$AGENT_ENV MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1 REQUEST_TIMEOUT=3600"   # A.7
sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus=1 \
  --cpus-per-task=16 --mem=113560M --time=02:00:00 --job-name="swebench-$POLICY" \
  --output="$RUN_ROOT/slurm-%j.out" --export=ALL --mail-type=BEGIN,END,FAIL --mail-user="$MAIL" \
  /abs/path/to/swebench_job.sh
```

- **Stock control:** `POLICY=stock`, with the same `AGENT_RUN` and `AGENT_ENV`: the script then passes `STOCK=1` to `run_agents.sh` (B.2, B.7).
- **Time limits.** `--time` covers server start-up (the Qwen servers of record were READY 230–265 s after their launch, jobs 3404230, 3404696, 3428631 and 3429500; [docs/reference.md section 4.8](reference.md#48-cuda-graphs) adds about a minute of torch.compile on a cold cache), the agent run (`AGENT_SECONDS`) and the shutdown. A run cut by `AGENT_SECONDS` or by the job's time limit is partial (A.11), so choose both with margin. `goal_query` runs longer than the other servers, because one Goal rewriter process serves every session, one call at a time or, with `--goal-rewriter-batch 4`, up to 4 queued calls together ([docs/reference.md section 4.10](reference.md#410-admission-optimizations-opt-in)); give it a longer job, for example `--time=04:30:00` with `AGENT_SECONDS=13500`.
- `RUN_ROOT` must be short: `TMPDIR` lives under it. `AGENT_SOCKET` and `AGENT_OUT` must be new for every job.

### B.6 Readiness check

- **This runtime.** READY is `server_checks.py ready` exit 0 ([docs/running.md section 4.1](running.md#41-one-server)); `/v1/models` answering is not READY. `dry-run.json` holds the vLLM command, which must match A.3. **Never send a test chat request to this runtime's server:** a request whose task message matches no registered task is rejected and ends the server's run, and a matching one opens that task's session.
- **Stock control.** `GET /health` returns 200 and `GET /v1/models` lists `selective-prefix`.
- **Kernel lines.** `kernels.txt` in the job directory must show the lines of A.5.
- **From the agent host,** before any agent starts (the job script performs the first check, `run_agents.sh` the second). `LABEL` and `POLICY` are set only in the submitting shell on the GPU side (B.5): on the agent host set them to the job's values first (the first line, for the job of B.5's example), with `AGENT` as in B.1. B.7 and B.8 use the same three variables:

```sh
LABEL=a1 POLICY=observation_query                                                    # the job's values (B.5)
curl -sf --unix-socket "$AGENT/sock/$LABEL-$POLICY.sock" http://localhost/v1/models   # lists selective-prefix
ss -ltn 'sport = :20000'                                                             # only 127.0.0.1 listens
```

### B.7 Running the pool

- Run the five servers one after another, one job each, and never two runs on the agent host at once (A.11). Evaluate each run after its agents end (B.8), outside any other run.
- `run_agents.sh` checks its arguments, `HF_DATASETS_CACHE`, the model-override template, the Mini-SWE-Agent version (2.4.2) and the PoC files, creates `$AGENT_OUT` (it refuses an existing one) with `instances.txt` and `agents.pgid`, checks the pinned images (`images.txt`), starts the relay, waits for the endpoint, checks that only `127.0.0.1` listens on the port, renders the model override and runs the pool. A failed check stops it with a nonzero exit before any agent starts (3 for an unreachable endpoint or another listener on the port, 2 for most other checks). Progress on the agent host: one line per finished instance in `$AGENT_OUT/timeline.jsonl`, and at most 16 task containers in `docker ps`. Its last output line is a summary (`runner_exit`, `seconds`, `containers_left`, `preds`).
- **Exit status.** An agent succeeds when its runner exits 0 and its session close answers HTTP 200, or, with `STOCK=1` (the stock control), HTTP 404: stock vLLM has no close route. `run_agents.sh` exits 0 when every agent succeeded, else 1. Without `STOCK=1` every close on the stock control gets 404 and the run exits 1; judge such a run by `runner_exit` in `timeline.jsonl`.
- **A cut run.** An SSH drop or the job's `timeout` does not stop the runners on the agent host ([docs/benchmarks/swe-bench-mini-swe-agent.md section 7.4](benchmarks/swe-bench-mini-swe-agent.md#74-run-one-group)). `run_agents.sh` runs in a process group of its own and writes its id to `agents.pgid`: the first line below stops that run (the script, its relay and its runners) and nothing else. The loop then lists the containers still running from the run's pinned images (`images.txt`), which Mini-SWE-Agent stops asynchronously; stop any it still lists with `docker stop <id>`, only those ids (untested). The run is partial; rerun the whole server under a new label (A.11).

```sh
kill -TERM -- "-$(cat "$AGENT/runs/$POLICY-$LABEL/agents.pgid")"
while read -r instance image; do docker ps -q --filter "ancestor=$image"; done < "$AGENT/runs/$POLICY-$LABEL/images.txt"
```

### B.8 Evaluation

On the agent host, after the run's agents have ended ([docs/benchmarks/swe-bench-mini-swe-agent.md section 7.5](benchmarks/swe-bench-mini-swe-agent.md#75-evaluate)):

```sh
cd "$AGENT/vllm-0.24.0-selective-prefix-policies"
EVAL_PYTHON="$EVAL/bin/python" WORKERS=4 examples/swebench/evaluate.sh "swebench-$POLICY-$LABEL" "$POLICY" \
  "$AGENT/eval/$POLICY-$LABEL" "$AGENT/runs/$POLICY-$LABEL"
```

The arguments are a run id, a label, a new evaluation directory (its parent must exist) and the run's `AGENT_OUT`. `evaluate.sh` writes one prediction per planned instance (an empty patch for a missing prediction), checks that the evaluator reads a local Parquet dataset (SWE-bench at `02e7a74`, B.1) and that the instances' images are the pinned ones, fetches the evaluation dataset's test split file at its pinned revision (A.9; network on the first evaluation, or the Hugging Face cache) as `dataset.parquet` and checks its SHA-256, runs the command of A.9 in the new evaluation directory and prints one line per instance and a final line with `resolved`. Accuracy = `resolved` / 32 (A.10). Trajectories, predictions and logs contain prompts and tool output; keep them out of public releases.

### B.9 Throughput

The throughput of A.10 from one pool output directory (untested):

```sh
python3 - "$AGENT/runs/$POLICY-$LABEL" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = [json.loads(line) for line in (out / 'timeline.jsonl').read_text().splitlines() if line.strip()]
span = max(r['end'] for r in rows) - min(r['start'] for r in rows)
paths = sorted(out.glob('runs/*/*/*.traj.json'))
tokens = calls = 0
for path in paths:
    for message in json.loads(path.read_text())['messages']:
        extra = message.get('extra') or {}
        if message.get('role') == 'assistant' or (
                message.get('role') == 'user' and extra.get('interrupt_type') == 'FormatError'):
            calls += 1
            usage = (extra.get('response') or {}).get('usage') or {}
            tokens += usage.get('completion_tokens') or 0
print(json.dumps(dict(instances=len(rows), trajectories=len(paths), calls=calls, completion_tokens=tokens,
                      span_s=round(span, 3), tokens_per_s=round(tokens / span, 3))))
PY
```

A complete run has 32 timeline lines and 32 trajectories; a run with fewer is partial (A.11).
