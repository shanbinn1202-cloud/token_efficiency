# SWE-bench Verified benchmark configuration

English | [简体中文](SWEBENCH_BENCHMARK.zh-CN.md)

This document fixes the configuration of the SWE-bench Verified benchmark run of this runtime: the first 32 instances of the `test` split, Mini-SWE-Agent 2.4.2 with at most 16 concurrent agents refilled in list order, one server per policy for `full`, `observation_query`, `goal_query` and `random_fill`, and a stock vLLM 0.24.0 server as the control. **Part A** is the configuration. A baseline (the stock vLLM control or any other system) is comparable only if it reproduces every item of the checklist in A.12; the configuration does not depend on where the agents run. **Part B** is a manual procedure with the tools of this repository. It is optional: any x86_64 Docker host that can reach the server over loopback can run the agents.

Section numbers such as "README §4.9" refer to [README.md](../README.md).

## Validation status

This configuration has run with `goal_query` on one GH200, runtime `5b15b2f`, the first 32 instances, 16 sessions, on 2026-10-18 (US Central):

- Job 3404696: `goal_query` without rewriter batching. Verified `SWE32_PASS_VERIFIED`, joined `JOIN_PASS`; all 32 instances done, 19 resolved.
- Job 3404230: `goal_query` with `--goal-rewriter-batch 4` (A.3). Verified `SWE32_PASS_VERIFIED`, joined `JOIN_PASS`; all 32 instances done, 20 resolved.

The resolved counts come from the verified joins of the server records with the evaluation. The runs of `full`, `observation_query`, `random_fill` and `stock` at `5b15b2f` are pending.

Before these runs, parts of this configuration had run on one GH200 (README):

- 16 sessions on one server: jobs 3339019 and 3343069 (synthetic sessions; README §4.4).
- The session pool (`--max-running-sessions`): job 3343069 (24 tasks with K = 8, synthetic sessions; README §4.4).
- CUDA graphs: jobs 3338893, 3339448 and 3343069, `--cuda-graphs full` with 16 sessions in job 3343069 (README §4.8).
- The 131072-token context, 4096-token replies and `--dram-arena-gib 88`, with 16 `observation_query` sessions, `--active-token-budget 8192`, `--cuda-graphs full` and 32 GiB of GPU KV: job 3348705 (README §4.9).
- `--admission-overlap 16` with `goal_query`, the Qwen3-0.6B Goal rewriter and 16 sessions: job 3349504 (README §4.10).
- Mini-SWE-Agent 2.4.2 against this runtime: the 10-instance Astropy pilot, one job per policy, 5 sessions per server, step limit 90, 1536-token replies (jobs 3313802, 3314010, 3314319, 3314623; README §4.7).

Not yet run: the four pending servers, among them the stock control with the two client-side contract lines of A.6; `run_agents.sh` with `MAX_RUNNING`, `STEP_LIMIT` and `MAX_TOKENS` (README §7.4; the two runs above used the agent pool of their own harness). The Slurm script of B.5, the gold-patch and chat-template checks of B.3, the cleanup command of B.7 and the throughput script of B.9 are untested. Apart from the two `goal_query` runs above, this document reports no result.

## Configuration for aligning a baseline

What our runs use, for a baseline to set the same way; every value is that of Part A (A.12 is the full checklist, A.13 lists the differences that remain):

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
| Stock control (vLLM) | `--kv-cache-memory-bytes 34359738368` (32 GiB), `--max-model-len 131072`, `--max-num-batched-tokens 16384`, `--max-num-seqs 16`; CUDA graphs: vLLM's default (`FULL_AND_PIECEWISE`); asynchronous scheduling: vLLM's default (on); `--enable-prefix-caching --prefix-caching-hash-algo sha256`; no `--attention-backend` (vLLM picks FLASH_ATTN with FlashAttention 3); `--language-model-only` (A.4, A.5) |
| Throughput | The client-side completion tokens of every model call of the main agents of the 32 instances / the wall time from the first agent's start to the last agent's end on the agent host's clock. Tool execution, task-container start and stop, waits and the session closes before refills are included; server start-up and shutdown, evaluation and the Goal rewriter's tokens are not (A.10) |
| Tasks | SWE-bench Verified `test`, the first 32 instances in dataset order (22 Astropy, 10 Django); Mini-SWE-Agent 2.4.2 (litellm 1.103.1) with the PoC runner and prompt, step limit 120 (A.1, A.6) |
| Evaluation | swebench 5.0.2 on the agent (x86_64 Docker) host, `SWE-bench/SWE-bench_Verified` `test`, `--timeout 1800`; the agents' task containers run the official images `docker.io/swebench/sweb.eval.x86_64.*` (A.6, A.9) |
| Gemma 4 31B IT | The same settings are intended; not fixed in this document. The profile is in development and not GPU-validated (README §4.14), and it refuses `--goal-rewriter-batch` (README §4.10) |

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

The four policies of this runtime (README, policy table, and §5):

| `--policy` | History pages used for the next turn |
| --- | --- |
| `full` | Every historical page: the reference, and the configuration closest to stock vLLM |
| `observation_query` | Selector V2: mandatory, recent and top-cosine pages within the budget; the query is the hidden state at the end of the newest observation |
| `goal_query` | The same selector; the query is a maintained retrieval Goal, updated by a Qwen3-0.6B rewriter |
| `random_fill` | Control of the two query policies: the same budget, mandatory and recent pages, the remaining capacity filled in a seeded random order |

### A.2 Model and versions

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, a local directory with regular files; metadata SHA-256 values in README §2.2 |
| Goal rewriter (`goal_query` only) | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`; the same resolved path for every `goal_query` server of one comparison (README §2.2) |
| vLLM | The official wheel `vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl`, unmodified, SHA-256 `700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5` |
| Python environment | CPython 3.12, `transformers==5.13.0`, `tokenizers==0.22.2`, built with `examples/env/constraints-3270721.txt` (README §2.1). The stock control runs from the same environment; this runtime is installed into a separate release directory and leaves the environment unchanged (README §2.4) |
| Runtime | This repository at commit `5b15b2f` (`5b15b2ff2d7497717dffdb4a77641781a8c1090e`, "Batch the Goal rewriter calls that queue"), the commit of the benchmark's current runs, or a later commit that leaves the XML protocol path and the vLLM arguments of A.3 unchanged; `main` contains it. History: earlier runs of this benchmark used `e877051`, which encodes each message with the loaded tokenizer only, as stock vLLM does; with `3496d60` one Django instance was refused with HTTP 500 |
| GPU | One NVIDIA GH200 120GB, Linux aarch64; one server per GPU at a time |
| Slurm resources | `--nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16 --mem=113560M` (one GH200's share of the node's host memory; A.3 explains why it suffices) |
| Agent software | Mini-SWE-Agent `2.4.2`, litellm `1.103.1`, datasets `5.0.1`, Python 3.12 |
| Evaluator | swebench `5.0.2`, datasets `5.0.1` |

### A.3 This runtime: server flags and vLLM arguments

**Flags shared by the four policies** (`serve.py`):

| Flag | Value | Effect |
| --- | --- | --- |
| `--task` | `tasks-pool.json`: the 32 instances of A.1 in order, built by `examples/swebench/build_tasks.py --pool` | Registers 32 tasks; each request is routed by the problem statement in its task message (README §4.4) |
| `--max-running-sessions` | `16` | Session pool: at most 16 sessions open; a session opens at its task's first request and is closed by the agent pool (A.8). vLLM gets `--max-num-seqs 16` |
| `--kv-cache-gib` | `32` | vLLM gets `--kv-cache-memory-bytes 34359738368` |
| `--page-tokens` / `--kv-layout` | `16` / `decoupled` | 16-token attention pages and a separate Gated-DeltaNet pool of 33 slots (README §4.6). vLLM gets `--block-size 16 --mamba-block-size 784 --additional-config ...` |
| `--max-model-len` | `131072` | Context limit, prompt plus reply. A prompt holds at most 131,072 − 4,096 = 126,976 tokens (README §4.9) |
| `--max-tokens` | `4096` | Reply bound, forced by the server on every request (README §4.9) |
| `--dram-arena-gib` | `70` | One pinned host page arena of 70 GiB for all sessions: 71,680 Pages of 1 MiB. It holds 8 sessions at the 131,072-token limit (8,193 Pages each), not 16; the 16 largest sessions of the earlier 16-session SWE-bench runs need 58.32 GiB at once. A capture beyond it fails that request with `DRAM_ARENA_EXHAUSTED` (README §4.9) |
| `--connector-pool-gib` | `1` | The connector's unused CPU pool at 1 GiB instead of 32 (README §4.9). vLLM gets `cpu_bytes_to_use` 1073741824 |
| `--cuda-graphs` | `full` | vLLM gets `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}` instead of `--enforce-eager` (README §4.8) |
| `--decode-fastpath` / `--decode-fastpath2` | on / on | Less host work per decode step, the same values; the second stage needs the first (README §4.11). No vLLM argument |
| `--language-model-only` | on | No vision tower and no multimodal warmup (README §4.12). vLLM gets `--language-model-only` |
| `--protocol` | `mswea-xml` (the default; not passed) | Mini-SWE-Agent XML protocol (README §6.1) |
| `--port` | `8000` | The server binds `127.0.0.1` only |

**Flags per policy:**

| Policy | Selector flags | Admission flags | Goal rewriter |
| --- | --- | --- | --- |
| `full` | none | none | none |
| `observation_query` | `--active-token-budget 8192` | none | not started (README §5) |
| `random_fill` | `--active-token-budget 8192` (fill seed: the default 42) | none | not started (README §5) |
| `goal_query` | `--active-token-budget 8192 --goal-rewriter-model <Qwen3-0.6B directory> --goal-rewriter-batch 4` | `--admission-overlap 16` | one Qwen3-0.6B process on the same GPU (about 2 GiB). `--goal-rewriter-batch 4` is the setting of the paper's configuration: the batched run is the one compared. Batched greedy decoding of the rewriter is not bit-identical to one call at a time, so compare policies only between servers with the same setting (README §4.10); 9 of the 59 Goal fallbacks of job 3404230 were batch-only failures (4 CUDA out-of-memory, 5 cuDNN attention errors; none in job 3404696), whose recovery is being fixed (not yet in `main`) |

No server sets `--gdn-device-handoff`, `--phase1-in-step`, `--active-ratio`, `--goal-query-variant`, `--fill-seed`, `--no-goal-rewriter`, `--decode-fastpath3` or `--release-idle-pages`. The admission flags follow README §4.10 (`--admission-overlap 16` for `goal_query` only). Selector, rewriter and admission flags and the decode fast paths pin the startup configuration but add no vLLM argument, so all four servers start vLLM with the same arguments.

**Host memory.** The pinned host memory is the 1-GiB connector pool, the Gated-DeltaNet carries (2.29 GiB) and the 70-GiB arena: 73.29 GiB in the servers' dry runs. With about 32 GiB for the processes this fits one GH200's share, `--mem=113560M` (110.9 GiB); `--language-model-only` keeps vLLM's multimodal warmup, which takes tens of GiB of host memory for under a second, out of the job (README §4.12). Earlier versions of this configuration used `--dram-arena-gib 88`, the default 32-GiB connector pool and `--mem=200g`; job 3348705 (Validation status) ran the 88-GiB arena with 200 GiB requested (README §4.9).

**Commands.** Dry runs of the four servers (add `--start` to serve; `TASKS` is the path of `tasks-pool.json`, `LABEL` a new label per attempt):

```sh
SHARED=(--max-running-sessions 16 --kv-cache-gib 32 --page-tokens 16 --kv-layout decoupled
        --max-model-len 131072 --max-tokens 4096 --dram-arena-gib 70 --connector-pool-gib 1 --cuda-graphs full
        --decode-fastpath --decode-fastpath2 --language-model-only)
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

**Resulting vLLM arguments** (identical for the four servers; the dry run prints the authoritative list, here wrapped by argument group):

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

**Environment** of the shell that starts `serve.py` (README §3):

| Variable | Value |
| --- | --- |
| `PATH` | `$VENV/bin:/usr/local/bin:/usr/bin:/bin` |
| `LD_LIBRARY_PATH` | `$VENV/lib/python3.12/site-packages/nvidia/cu13/lib` |
| `TMPDIR` | A short path: `len(TMPDIR) + 37` ≤ 107 bytes (AF_UNIX limit) |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` (native top-k/top-p sampler) |
| `CC`, `VLLM_RPC_BASE_PATH` | unset |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `PYTHONDONTWRITEBYTECODE` | `1` |
| Set by `serve.py` | `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `VLLM_USE_V2_MODEL_RUNNER=0` and the runtime's own variables |

**Generation contract, enforced by the server** (README §6.1 and §4.9). The server ignores the client's sampling fields:
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
| `--language-model-only` | on | Same (README §4.12) |
| `--mamba-cache-mode` | `align` | Same |
| `--gdn-prefill-backend` | `triton` | Same |
| CUDA graphs | vLLM's default: neither `--enforce-eager` nor `--compilation-config`, which gives `FULL_AND_PIECEWISE` with `VLLM_COMPILE` (README §4.8) | Same mode (`--cuda-graphs full`) |
| Asynchronous scheduling | vLLM's default (enabled); no flag | Differs: the launcher always passes `--no-async-scheduling` (A.13) |
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
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes 1, 2, 4, 8, 16, 24 and 32 at 16 sequences (README §4.8) | `'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE` and `'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32]` in the engine configuration line |
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
| Agent | Mini-SWE-Agent 2.4.2 (litellm 1.103.1, datasets 5.0.1) |
| Runner | The PoC's `examples/swebench/upstream/swebench_runner.py`, SHA-256 `34ca6d9288c72830e61bd00ad8b15790cab4c29515db24b8df30775cfc0880a7` |
| Prompt | The PoC's `examples/swebench/upstream/swebench_prompt_override.yaml`, SHA-256 `1b6a48666ad528775bd11225fa4d5c002721c25c0cdbc05b99cc15b2de31336a` (LF bytes) |
| Config stack | `-c swebench_xml.yaml` (built into Mini-SWE-Agent) `-c swebench_prompt_override.yaml -c model_override.yaml` |
| One runner per instance | `swebench_runner.py --subset verified --split test --workers 1 --filter '^<instance_id>$' --output <out>/runs/<instance_id>` with the config stack (`run_agents.sh` with `MAX_RUNNING`) |
| Runner dataset | `princeton-nlp/SWE-Bench_Verified` (`--subset verified`), cached and read offline |
| Model | `litellm_textbased`, `openai/selective-prefix`, `api_base` `http://127.0.0.1:<port>/v1`, dummy key, `temperature` 0.0, `max_tokens` 4096, `cost_tracking: ignore_errors`; client timeout and retries in A.7; the stock control adds the generation contract of A.4 |
| Tool environment | `docker`, image `docker.io/swebench/sweb.eval.x86_64.<instance id with "__" replaced by "_1776_", lower case>:latest`, `cwd` `/testbed`, 120 s per command, `bash -c`, `PAGER=cat`, `MANPAGER=cat`, `LESS=-R`, `PIP_PROGRESS_BAR=off`, `TQDM_DISABLE=1` |
| Process environment | `MSWEA_CTX_PATCH` unset, `AGENTDIET_MODE=off`, `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `MSWEA_GLOBAL_CONFIG_DIR` an empty directory, `PYTHONDONTWRITEBYTECODE=1`, `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1`, `REQUEST_TIMEOUT=3600` |

Properties of the PoC files that affect results (README §7.1):
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

**`model_override.yaml` for the stock control:** the same file with the client-side generation contract of A.4 in two more lines after `num_retries`:

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

The client lines ran in the two `goal_query` runs (Validation status); the two contract lines on stock have not yet run in an agent run.

### A.7 Limits

| Limit | Value |
| --- | --- |
| Step limit / cost limit | `step_limit` 120 / `cost_limit` 0 |
| Reply bound | 4096 new tokens per model call (this runtime: forced by the server; stock: `max_tokens` from the client) |
| Context limit | 131,072 tokens, prompt plus reply; this runtime refuses a prompt above 126,976 tokens with HTTP 400 before admission (README §4.9) |
| Tool command timeout | 120 s per command |
| Client timeout | 3600 s per model call (`timeout: 3600`, `REQUEST_TIMEOUT=3600`) |
| Client retries | None: `max_retries: 0` and `num_retries: 0` (litellm and the OpenAI client), `MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1` (Mini-SWE-Agent) |
| Requests per server | 32 instances × 120 steps = at most 3,840 requests, within this runtime's lifetime bound of max(1,024, 256 × 32) = 8,192 (README §4.9) |

**Why no client retries and a 3600-s timeout.** On this runtime a client retry of a request that is still being served is a second request for a busy session, and a timeout or a non-2xx status ends the server's run (README §6.3). With `--admission-overlap 16`, `goal_query` admissions can wait for the single Goal rewriter process, and the frontend waits up to 2,100 s for an admission (README §4.10) before the reply of up to 4096 tokens starts. The client timeout of 3600 s lies above that wait plus a full reply, so slow admissions complete instead of being abandoned and retried by the client. The stock control uses the same settings so that the two sides differ only in the server.

### A.8 Concurrency, refill and session close

- At most 16 agents run at once, one runner process per instance (`--workers 1`), started in the order of A.1. When a runner ends, the next instance starts (`run_agents.sh` with `MAX_RUNNING=16`).
- On this runtime the ended instance's session is closed before its slot is reused: `POST /v1/sessions/close` with `{"instance_id": "<instance id>"}`, retried on HTTP 409 (`SESSION_BUSY`) up to 30 times at 2-s intervals (README §6.5). The close frees the session's GPU and host state, and the next waiting task opens. The close time delays the next start by its own duration.
- On the stock control the same request returns HTTP 404 at once; stock vLLM keeps cached prefixes until it evicts them.
- `run_agents.sh` writes one line per instance to `timeline.jsonl` on the agent host: `instance_id`, `start` (taken just before the runner starts), `end` (taken when the runner process exits, before the close request), `runner_exit` and `close_http`. It also writes `runs/<instance>/`, `logs/<instance>.log`, `closes/<instance>.json` and a merged `preds.json`.

### A.9 Evaluation

| Item | Value |
| --- | --- |
| Evaluator | swebench 5.0.2 on the agent (Docker) host, after every agent of the run has ended |
| Dataset | `SWE-bench/SWE-bench_Verified`, split `test` |
| Command | `python -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Verified --split test --predictions_path predictions.jsonl --max_workers 2 --timeout 1800 --run_id <run_id>` (`examples/swebench/evaluate.sh`, README §7.5) |
| Timeout | `--timeout 1800` per instance |
| Workers | `--max_workers 2` (the default of `examples/swebench/evaluate.sh`). The worker count sets only how many instances are evaluated in parallel; any count gives the same results |
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
- **Failures.** A request refused with HTTP 400 because its prompt exceeds the context limit ends that instance only (unresolved); the server keeps serving (README §4.9). Any other non-2xx status, a timeout, a disconnect or a server exit ends the run (README §6.3), which is then partial.
- **Transport.** The agents reach the server through a loopback-only route (B.4). Any loopback-only route gives the same results; the server never listens beyond loopback (README §6.4).

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
| 15 | Asynchronous scheduling | Off (always) | vLLM's default (on); a known difference (A.13) |
| 16 | Attention | FLASH_ATTN with FlashAttention 3, chosen by vLLM | The same, no `--attention-backend` |
| 17 | Gated-DeltaNet prefill | Triton | `--gdn-prefill-backend triton` |
| 18 | Sampler | Native (`VLLM_USE_FLASHINFER_SAMPLER=0`) | The same variable |
| 19 | Prefix caching | On, sha256 | `--enable-prefix-caching --prefix-caching-hash-algo sha256` |
| 20 | Mamba cache mode | `align` | `--mamba-cache-mode align` |
| 21 | Seed | 42 | `--seed 42` |
| 22 | Text only | `--language-model-only` | vLLM's `--language-model-only` |
| 23 | Agent | Mini-SWE-Agent 2.4.2, litellm 1.103.1, datasets 5.0.1 | The same |
| 24 | Runner and prompt | The PoC files, SHA-256 `34ca6d92…` and `1b6a4866…` (the Astropy prompt for all 32 instances) | The same files |
| 25 | Config stack | `swebench_xml.yaml`, prompt override, model override | The same |
| 26 | Step limit / cost limit | 120 / 0 | The same |
| 27 | Tool environment | Docker, `/testbed`, 120 s, the variables of A.6 | The same |
| 28 | Process environment | `AGENTDIET_MODE=off`, offline caches, an empty `MSWEA_GLOBAL_CONFIG_DIR` | The same |
| 29 | Client timeout and retries | 3600 s, no retries | The same |
| 30 | Agent host | One run at a time, no evaluation during a run | The same rule and the same host type |
| 31 | Evaluation | swebench 5.0.2, `SWE-bench/SWE-bench_Verified`, `test`, `--timeout 1800`, 2 workers | The same |
| 32 | Throughput | The definition of A.10 | The same numerator and clock |
| 33 | Partial runs | Excluded, never resumed | The same |

### A.13 Known differences between this runtime and stock vLLM

These differences remain when every checklist item matches. Report them with any comparison.

1. **Previous replies.** This runtime replays each previous reply as the exact token IDs it was generated with; stock vLLM re-renders the history from text through the chat template (README §6.1).
2. **Asynchronous scheduling.** This runtime runs without it (the launcher always passes `--no-async-scheduling`); stock vLLM runs with its default, enabled.
3. **Sampling control.** This runtime enforces greedy decoding, the reply bound and the stop marker in the engine and ignores the client's fields; stock vLLM takes them from the request. At the marker, this runtime returns everything the final token decodes to, including characters past the marker (README §6.1); stock vLLM cuts the text at the stop string.
4. **Context edge.** This runtime refuses a prompt above 126,976 tokens with HTTP 400 (README §4.9); stock vLLM shortens the reply to the room left in the context instead (README §4.9 describes this vLLM behaviour; not measured for the stock control).
5. **KV management.** Both have 32 GiB of GPU KV, in different layouts. This runtime uses 16-token attention pages, a separate Gated-DeltaNet pool, a 1-GiB CPU pool and a 70-GiB pinned page arena; it never preempts, and a turn that does not fit waits (`CAPACITY_WAIT`, README §4.4). At 131,072 tokens, 16 `full` sessions cannot all be held at full context in 32 GiB (README §4.9), so `full` turns can wait. Stock vLLM uses its own hybrid layout (README §4.6) and can preempt when the pool is full.
6. **Admission and scheduling.** This runtime admits one request at a time in arrival order (`goal_query`: up to 16 in flight with `--admission-overlap 16`), runs each admission's phase 1 as its own forward pass, then decodes in batches (README §4.4, §4.10, §5). Stock vLLM schedules prefill chunks together with decode steps within the 16,384-token step budget.
7. **Session close.** This runtime frees a session's state when its agent ends; stock vLLM keeps cached prefixes until it evicts them.
8. **Rebuilds.** On this runtime a request that does not extend the previous one (for example a format-error retry that drops the rejected reply) starts a fresh session re-rendered from text (README §6.1); stock vLLM serves it with prefix caching.
9. **Selective policies.** `observation_query`, `goal_query` and `random_fill` attend to a selected subset of history pages by design; `full` keeps every page.
10. **Goal rewriter.** `goal_query` runs an auxiliary Qwen3-0.6B on the same GPU. Its time counts in the throughput, its tokens do not.
11. **Run-to-run variation.** Batched runs are not token-identical from run to run on either server: a reply can change with the batch composition, which depends on timing (README §4.4). Between two 10-instance pilots, `full` changed its outcome on 2 of 10 instances (README §4.7).

---

## Part B. Procedure

This part runs the configuration of Part A with the tools of this repository: the model on a GH200 node in a Slurm job, the agents and their task containers on a separate x86_64 Docker host (the agent host), and a reverse SSH tunnel from the GPU job to the agent host (README §6.4). Every step applies to the four policies and to the stock control unless marked.

### B.1 Agent host

- **Requirements.** Linux x86_64 with Docker, `curl`, `ss`, `python3` and `uv`, an agent environment and an evaluator environment with the versions of A.2, and outbound network access while the caches are filled. The host runs up to 16 runner processes and 16 task containers at once. Inbound SSH from the GPU nodes is needed for the tunnel (B.4).
- **Image naming.** One image per instance: `docker.io/swebench/sweb.eval.x86_64.<instance id with "__" replaced by "_1776_", lower case>:latest`, for example `docker.io/swebench/sweb.eval.x86_64.django_1776_django-10097:latest`.

Environments, dataset caches and directories (the commands are not part of the validated runs; README §7.2), with a checkout of this repository at `$AGENT/vllm-0.24.0-selective-prefix-policies`:

```sh
AGENT=/abs/path/on/agent
MINI=$AGENT/mini-env
EVAL=$AGENT/eval-env
uv venv --python 3.12 "$MINI"
uv pip install --python "$MINI/bin/python" mini-swe-agent==2.4.2 litellm==1.103.1 datasets==5.0.1
uv venv --python 3.12 "$EVAL"
uv pip install --python "$EVAL/bin/python" swebench==5.0.2 datasets==5.0.1
"$MINI/bin/python" -c "from datasets import load_dataset; load_dataset('princeton-nlp/SWE-Bench_Verified', split='test')"
"$EVAL/bin/python" -c "from datasets import load_dataset; load_dataset('SWE-bench/SWE-bench_Verified', split='test')"
mkdir -p "$AGENT/sock" "$AGENT/runs" "$AGENT/eval" "$AGENT/mswea-config"
chmod 700 "$AGENT/sock"
```

Save the list of A.1 as `$AGENT/ids.txt` and pull the 32 images before any run, so that no pull enters the measured time:

```sh
while read -r i; do
  img="docker.io/swebench/sweb.eval.x86_64.${i//__/_1776_}:latest"; img=${img,,}
  if docker pull "$img" > /dev/null; then echo "ok $img"; else echo "FAILED $img"; fi
done < "$AGENT/ids.txt"
```

### B.2 Task list and agent files

Build `tasks-pool.json` from the runner's dataset, with the 32 statements verbatim and in order, and copy it to the GPU side:

```sh
REPO_A=$AGENT/vllm-0.24.0-selective-prefix-policies
mapfile -t IDS < "$AGENT/ids.txt"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "$MINI/bin/python" "$REPO_A/examples/swebench/build_tasks.py" \
  --output-dir "$AGENT/tasks" --pool "${IDS[@]}"
```

`run_agents.sh` renders the model override from the `model_override.yaml` next to it. Make two copies of `examples/swebench`, one per server kind, and add the lines of A.6 after the template's `max_tokens` line (`render_override.py` keeps added lines and replaces only the port, `max_tokens` and `step_limit` lines):

```sh
cp -r "$REPO_A/examples/swebench" "$AGENT/swebench-runtime"
cp -r "$REPO_A/examples/swebench" "$AGENT/swebench-stock"
python3 - "$AGENT/swebench-runtime/model_override.yaml" "$AGENT/swebench-stock/model_override.yaml" <<'PY'
import sys
anchor = '    max_tokens: 1536\n'
client = '    timeout: 3600\n    max_retries: 0\n    num_retries: 0\n'
stock = ('    extra_body: {"chat_template_kwargs": {"enable_thinking": false}, "include_stop_str_in_output": true}\n'
         '    stop: ["</mswea_bash_command>"]\n')
for path, extra in zip(sys.argv[1:], (client, client + stock)):
    with open(path, newline='') as stream:
        text = stream.read()
    assert text.count(anchor) == 1, path
    with open(path, 'w', newline='') as stream:
        stream.write(text.replace(anchor, anchor + extra))
PY
python3 -I -B "$AGENT/swebench-runtime/render_override.py" "$AGENT/swebench-runtime/model_override.yaml" 20000 4096 120
python3 -I -B "$AGENT/swebench-stock/render_override.py" "$AGENT/swebench-stock/model_override.yaml" 20000 4096 120
```

The two rendered files must equal the two files of A.6. Each copy keeps `upstream/` with the pinned PoC files, which `run_agents.sh` checks before every run.

### B.3 Environment check before GPU time

**Gold patches.** Evaluating the reference patches of the 32 instances checks the images, the evaluator and the agent host without a model; every instance is expected to resolve, and any that does not is investigated before a GPU job. The evaluator writes `logs/` under its working directory, so run it in a new directory. Untested; confirm the options with `--help` of swebench 5.0.2:

```sh
mkdir "$AGENT/eval/gold-check" && cd "$AGENT/eval/gold-check"
"$EVAL/bin/python" -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Verified \
  --split test --predictions_path gold --instance_ids $(cat "$AGENT/ids.txt") \
  --max_workers 2 --timeout 1800 --run_id gold-check
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

The GPU job opens a reverse SSH tunnel after READY whose far end is a Unix socket on the agent host; `run_agents.sh` exposes it at `127.0.0.1:<port>` through `examples/swebench/relay.py` and refuses to run if anything but `127.0.0.1` listens on that port (README §6.4). The same form serves the stock control. It needs:
- a key that the agent host accepts for the tunnel user, readable by the GPU job, and a known-hosts file that pins the agent host's key (the job never uses an SSH agent);
- outbound SSH from the compute nodes to the agent host (check once from a compute node before the first run);
- a new socket path for every server, at most 100 bytes, in a `0700` directory on the agent host (`$AGENT/sock`).

A tunnel drop ends the run (A.11); the job does not reconnect.

### B.5 Slurm job script

The script below runs one server of the benchmark in one Slurm job: the server, READY, the kernel lines, the tunnel, the agent pool on the agent host over SSH, and the shutdown. It is derived from `examples/slurm/serve_gh200.sh`, which does not pass `--cuda-graphs`, `--active-token-budget`, `--connector-pool-gib`, the decode fast paths, `--language-model-only` or the admission flags of A.3 (README §4.10, §4.12). It is an example and **untested**. Save it as `swebench_job.sh`:

```bash
#!/usr/bin/env bash
# One server of the SWE-bench Verified benchmark (docs/SWEBENCH_BENCHMARK.md) in one Slurm job.
# POLICY: full | observation_query | goal_query | random_fill (this runtime) | stock (stock vLLM 0.24.0).
set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}" || { echo "run this under sbatch" >&2; exit 2; }
: "${REPO:?}" "${VENV:?}" "${TASKS:?}" "${POLICY:?}" "${RUN_ROOT:?}" "${AGENT_SECONDS:?}"
: "${AGENT_HOST:?}" "${TUNNEL_KEY:?}" "${KNOWN_HOSTS:?}" "${AGENT_SOCKET:?}" "${AGENT_PORT:?}"
: "${AGENT_RUN:?}" "${AGENT_OUT:?}" "${AGENT_ENV:?}"
PORT=${PORT:-8000}; READY_SECONDS=${READY_SECONDS:-900}

# Job-local directories and the environment of README section 3.
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
        --decode-fastpath --decode-fastpath2 --language-model-only)
case $POLICY in
  full)                          POLICY_FLAGS=() ;;
  observation_query|random_fill) POLICY_FLAGS=(--active-token-budget 8192) ;;
  goal_query)                    POLICY_FLAGS=(--active-token-budget 8192 --goal-rewriter-model "${REWRITER:?}"
                                               --goal-rewriter-batch 4 --admission-overlap 16) ;;
  stock)                         POLICY_FLAGS=() ;;
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

# READY: ready.json (this runtime, README section 4.1) or /health and /v1/models (stock).
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

# Reverse tunnel: a Unix socket on the agent host -> this node's 127.0.0.1:$PORT (README section 6.4).
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
  timeout "$AGENT_SECONDS" ssh -T "${SSH_OPTS[@]}" "$AGENT_HOST" env $AGENT_ENV MAX_RUNNING=16 STEP_LIMIT=120 \
    MAX_TOKENS=4096 "$AGENT_RUN" "$AGENT_SOCKET" "$AGENT_PORT" "$AGENT_OUT" $INSTANCES > "$STAGE/agents.log" 2>&1 \
    || agents_rc=$?
  if (( agents_rc == 124 )); then echo "agent time limit reached: the run is partial (B.7)" >&2; fi
else
  echo "tunnel not up (see $STAGE/tunnel.log)" >&2; agents_rc=1
fi
stop_tunnel

# Stop: SIGTERM to the server PID only; this runtime must exit 0 with clean receipts (README section 8).
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
export AGENT_RUN=/abs/path/on/agent/swebench-runtime/run_agents.sh AGENT_OUT="/abs/path/on/agent/runs/$POLICY-$LABEL"
export AGENT_ENV="MINI_PYTHON=/abs/path/on/agent/mini-env/bin/python MSWEA_CONFIG_DIR=/abs/path/on/agent/mswea-config MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1 REQUEST_TIMEOUT=3600"
sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus=1 \
  --cpus-per-task=16 --mem=113560M --time=02:00:00 --job-name="swebench-$POLICY" \
  --output="$RUN_ROOT/slurm-%j.out" --export=ALL --mail-type=BEGIN,END,FAIL --mail-user="$MAIL" \
  /abs/path/to/swebench_job.sh
```

- **Stock control:** `POLICY=stock` and `AGENT_RUN=/abs/path/on/agent/swebench-stock/run_agents.sh` (the copy with the stock model override, B.2).
- **Time limits.** `--time` covers server start-up (READY 204–231 s for the first server of a job, README §4.1, plus about a minute of torch.compile on a cold cache, README §4.8), the agent run (`AGENT_SECONDS`) and the shutdown. A run cut by `AGENT_SECONDS` or by the job's time limit is partial (A.11), so choose both with margin. `goal_query` runs longer than the other servers, because one Goal rewriter process serves every session, one call at a time or, with `--goal-rewriter-batch 4`, up to 4 queued calls together (README §4.10); give it a longer job, for example `--time=04:30:00` with `AGENT_SECONDS=13500`.
- `RUN_ROOT` must be short: `TMPDIR` lives under it. `AGENT_SOCKET` and `AGENT_OUT` must be new for every job.

### B.6 Readiness check

- **This runtime.** READY is `server_checks.py ready` exit 0 (README §4.1); `/v1/models` answering is not READY. `dry-run.json` holds the vLLM command, which must match A.3. **Never send a test chat request to this runtime's server:** a request whose task message matches no registered task is rejected and ends the server's run, and a matching one opens that task's session.
- **Stock control.** `GET /health` returns 200 and `GET /v1/models` lists `selective-prefix`.
- **Kernel lines.** `kernels.txt` in the job directory must show the lines of A.5.
- **From the agent host,** before any agent starts (the job script performs the first check, `run_agents.sh` the second):

```sh
curl -sf --unix-socket "$AGENT/sock/$LABEL-$POLICY.sock" http://localhost/v1/models   # lists selective-prefix
ss -ltn 'sport = :20000'                                                             # only 127.0.0.1 listens
```

### B.7 Running the pool

- Run the five servers one after another, one job each, and never two runs on the agent host at once (A.11). Evaluate each run after its agents end (B.8), outside any other run.
- `run_agents.sh` checks the Mini-SWE-Agent version and the PoC files, starts the relay, waits for the endpoint, checks the listener, renders the model override and runs the pool. Progress on the agent host: one line per finished instance in `$AGENT_OUT/timeline.jsonl`, and at most 16 task containers in `docker ps`.
- **Stock control:** every session close returns HTTP 404, so `run_agents.sh` exits 1. Judge the run by `runner_exit` in `timeline.jsonl`.
- **A cut run.** An SSH drop or the job's `timeout` does not stop the runners on the agent host (README §7.4). Stop that pool's `run_agents.sh`, `swebench_runner.py` and `relay.py` processes, then remove only the containers named in its logs (untested). The run is partial; rerun the whole server under a new label (A.11).

```sh
grep -ohE 'minisweagent-[0-9a-f]{8}' "$AGENT/runs/$POLICY-$LABEL"/logs/*.log | sort -u | xargs -r docker rm -f
```

### B.8 Evaluation

On the agent host, after the run's agents have ended (README §7.5):

```sh
cd "$AGENT/vllm-0.24.0-selective-prefix-policies"
EVAL_PYTHON="$EVAL/bin/python" WORKERS=2 examples/swebench/evaluate.sh "swebench-$POLICY-$LABEL" "$POLICY" \
  "$AGENT/eval/$POLICY-$LABEL" "$AGENT/runs/$POLICY-$LABEL"
```

`evaluate.sh` writes one prediction per planned instance (an empty patch for a missing prediction), runs the command of A.9 in the new evaluation directory and prints one line per instance and a final line with `resolved`. Accuracy = `resolved` / 32 (A.10). Trajectories, predictions and logs contain prompts and tool output; keep them out of public releases.

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
