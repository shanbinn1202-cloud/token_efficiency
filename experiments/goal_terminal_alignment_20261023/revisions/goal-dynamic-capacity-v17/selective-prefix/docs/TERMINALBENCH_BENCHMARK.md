# Terminal-Bench 2.0 benchmark configuration

English | [简体中文](TERMINALBENCH_BENCHMARK.zh-CN.md)

This document fixes the configuration of the Terminal-Bench 2.0 benchmark run of this runtime: all 89 tasks of the `terminal-bench/terminal-bench-2` dataset at a pinned ref, Harbor 0.21.0 with its built-in Terminus 2 agent, unchanged and with thinking off, at most 16 trials at once (`-n 16`), one server per policy for `full`, `observation_query` and `random_fill` on the chat-text protocol (`--protocol chat-text`, README §6.7), and a stock vLLM 0.24.0 server as the control. **Part A** is the configuration. A baseline (the stock vLLM control or any other system) is comparable only if it reproduces every item of the checklist in A.12; the configuration does not depend on where the agents run. **Part B** is a manual procedure with the tools of this repository. It is optional: any x86_64 Docker host that can reach the server over loopback can run Harbor.

Two values are **pending confirmation**: the context limit of 65,536 tokens and the per-session KV budget of 8,192 tokens of the selective policies. Each is one value of the configuration (A.3, A.6), and a change of either is a new configuration. `goal_query` is not configured for Terminal-Bench yet: its Goal setup is pending, and the chat-text server of this revision refuses it. The Gemma 4 31B IT profile (README §4.14) serves the chat-text protocol in this branch, but it is not GPU-validated for it; this document covers Qwen3.6-27B only.

Section numbers such as "README §6.7" refer to [README.md](../README.md).

## Validation status

Before the benchmark, a 16-task dry run on one GH200 checks a policy: one fresh server with the server flags of A.3 and the agent configuration of A.6, on the 16 tasks of the dry subset (A.1; every one with a 900-s agent timeout) at `-n 16`, through the tunnel of B.4. Per server the dry run checks:

- **The Harbor run.** Every planned trial finished, verifier included, within the run's time cap; no capacity stall (A.11).
- **The server.** READY and the startup record: the chat-text protocol `chat_text_v1` with its slots, idle timeout, eviction delay, generation header and template switches, K = 20 running sessions, the KV bytes and the decoupled layout with its pools, `--release-idle-pages`, and for `observation_query` the selection contract with the 8,192-token budget. Then the kernel lines of A.5, the CUDA-graph mode and capture sizes, the decode fast path, the 1-GiB connector pool and the text-only mode (vLLM's text-only line present, no multimodal warmup).
- **The chat-text evidence.** Every `runtime.jsonl` row's `chat_text` record; `chat_sessions.jsonl` (bind, rebind and close rows in order, slots never reused, at most 20 open); the engine's session rows; per request, the completion within its effective bound, `finish_reason` `length` exactly at that bound, and prompt plus reply within 65,536 tokens.
- **Idle-page releases.** Every row's and native file's `idle_release` record, and the release and readmission events of `capacity.jsonl` against the KV pools and the totals of `engine-cleanup.json` (README §4.13).
- **Failures.** Only context-limit refusals (HTTP 400 before admission) and requests that their client cancelled at an agent timeout are allowed; nothing is preempted.
- **Shutdown.** After the job has closed the sessions that the run left open (A.8): no session open at shutdown, the cleanup receipts joined and the natural exit with code 0.
- **Token-exact receipts, per request.** Each prompt's token stream against the chat-text rule (a fresh or rebuilt slot renders the conversation through the pinned template; a continuing request appends the previous reply's exact ids, the boundary, one user message and the header), the turn geometry, the selection (budget and projection), the request's own I/O, the admission log, the KV pools, the Page evidence, the Gated-DeltaNet bytes (restored and readmitted Pages) and the phase-1 mode.
- **The join, after the fetch.** Each trial's ATIF trajectories against the server rows by `key_sha256` (A.8): calls equal rows, and each call's prompt and completion tokens equal the server's (A.10 for requests cut by an agent timeout).

**`full`: GPU-validated.** The dry run of `full` at runtime commit `68d19b4`, with the settings of this document: all 16 trials ended, the server completed 333 requests, every request was token-exact against its native receipt, nothing was preempted, the server never stalled for capacity, and every check passed. Two of the checks were corrected in the harness after the run (the open-session count of a bind row, and an admitted request whose client disconnected at an agent timeout); the runtime was not changed. The capacity gate released idle sessions' Pages during the run; no released session came back for another turn, so readmission did not run.

**`observation_query`: GPU-validated.** The dry run of `observation_query` with the 8,192-token budget at runtime commit `68d19b4`, with the settings of this document: all 16 trials ended, the server completed 384 requests (18 of them rebuilds), the selection was active on 238 of them, every request was token-exact against its native receipt, nothing was preempted, the server never stalled for capacity, every check passed, and the evidence was verified again after the fetch. After a harness fix of the join (it now handles the agent's short-summary calls, which run in a separate session, and replies that were not recorded when an agent timed out), all 16 trials joined with their ATIF trajectories. The pool never ran short, so no idle Pages were released.

A first dry run of `full`, at runtime `0d37346` and without `--release-idle-pages`, ended when the engine failed: the idle sessions' Pages filled the attention pool, a turn admitted while nothing ran found no free block, and vLLM preempted it. Commit `68d19b4` makes such a turn wait, and every server now runs with `--release-idle-pages` (A.3). That run is not part of the validation.

On the GPU, these dry runs are the first test of the chat-text protocol, of 20 running sessions on one server, of `--connector-pool-gib`, of `--language-model-only`, of `--release-idle-pages` and of `--dram-arena-gib 64` at 65,536 tokens. CUDA graphs (README §4.8) and the decode fast path (README §4.11) had run on the GPU before, with the XML protocol.

Not yet run: this configuration as a whole (89 tasks with agent timeouts of 600 to 12,000 s, which is the benchmark that this document configures); `random_fill` on Terminal-Bench; the stock control in this configuration. The dry runs used a separate harness with the server arguments, the Harbor job configuration, the tunnel and the session close of this document; the Slurm script of B.5, the agent-side files of B.2, the checks of B.3 and the scripts of B.7 to B.9 are untested. This document reports no result.

## Configuration for aligning a baseline

What our runs use, for a baseline to set the same way; every value is that of Part A (A.12 is the full checklist, A.13 lists the differences that remain):

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, on one GH200 120GB with the official vLLM 0.24.0 wheel (A.2) |
| Thinking | Off: the pinned chat template with `enable_thinking=false` and `preserve_thinking=true`, an empty think block in the generation header; a stock baseline sends `extra_body: {"chat_template_kwargs": {"enable_thinking": false}}` per call (Terminus 2's `llm_call_kwargs`; A.3, A.4) |
| Decoding | Greedy, enforced by the server; a stock baseline sends `temperature` 0.0. vLLM `--seed 42`. No stop string: a reply ends at EOS/pad or at `max_tokens` (A.3, A.4) |
| Context limit | `--max-model-len 65536` and Terminus's `max_input_tokens` 65536, **pending confirmation** (A.3, A.6) |
| Reply bound | 4096: the client's `max_tokens` 4096, within this runtime's `--max-tokens 4096` (A.3, A.7) |
| Per-session KV budget | `--active-token-budget 8192` for `observation_query` and `random_fill`, **pending confirmation**; `full` keeps every page (A.3) |
| Concurrency | 89 trials, at most 16 running (`-n 16`); this runtime opens up to K = 20 sessions (`--max-num-seqs 20`), the stock control runs `--max-num-seqs 16` (A.1, A.3, A.4) |
| KV cache of this runtime | 32 GiB of GPU KV: `--kv-cache-gib 32`, 34,359,738,368 bytes (A.3) |
| Stock control (vLLM) | `--kv-cache-memory-bytes 34359738368` (32 GiB), `--max-model-len 65536` (pending confirmation), `--max-num-batched-tokens 16384`, `--max-num-seqs 16`; CUDA graphs: vLLM's default (`FULL_AND_PIECEWISE`); asynchronous scheduling: vLLM's default (on); `--enable-prefix-caching --prefix-caching-hash-algo sha256`; no `--attention-backend` (vLLM picks FLASH_ATTN with FlashAttention 3); `--language-model-only` (A.4, A.5) |
| Throughput | The completion tokens of the main ATIF trajectories (with their continuations) of the 89 planned tasks' trials, the summarization sub-calls reported apart / the agents' span on the agent host's clock, from the first `agent_execution.started_at` to the last `agent_execution.finished_at`. Tool execution, waits, summarizations and other trials' setup and verifiers within the span are included; server start-up and shutdown, the setup before the first agent and the verifiers after the last are not (A.10) |
| Tasks | Terminal-Bench 2.0: all 89 tasks of `terminal-bench/terminal-bench-2` at ref `sha256:c6fc2e23…`; Harbor 0.21.0 with its built-in Terminus 2, version 2.0.0, unchanged (litellm 1.97.0, openai 2.54.0) (A.1, A.2, A.6) |
| Evaluation | Harbor's verifier inside each trial, right after its agent; reward 1.0 is solved (A.9) |
| Gemma 4 31B IT | The same settings are intended; not fixed in this document. The profile serves the chat-text protocol in this branch, but it is in development and not GPU-validated for it (README §4.14) |

---

## Part A. Configuration

### A.1 Scope and tasks

| Item | Value |
| --- | --- |
| Model | Qwen3.6-27B on one NVIDIA GH200 120GB with vLLM 0.24.0 (A.2) |
| Benchmark | Terminal-Bench 2.0: **all 89 tasks** of the dataset `terminal-bench/terminal-bench-2` at its pinned ref (A.2), each with its own agent and verifier timeouts |
| Agent | Harbor 0.21.0 with its built-in Terminus 2, unchanged, thinking off (A.6) |
| Concurrency | 89 trials, one per task, **at most 16 running** (`-n 16`). Harbor starts the next trial when one ends, in its own order |
| Servers | One fresh server per configuration, each in its own GPU job: `full`, `observation_query`, `random_fill` (this runtime) and `stock` (stock vLLM 0.24.0, the control) |
| Run | One server runs all 89 trials once, without interruption (A.11). One run is one sample (A.13) |

The 89 task names, in name order (B.1 saves this list as `tasks.txt`; Harbor's filter prefixes each with `terminal-bench/`):

```text
adaptive-rejection-sampler
bn-fit-modify
break-filter-js-from-html
build-cython-ext
build-pmars
build-pov-ray
caffe-cifar-10
cancel-async-tasks
chess-best-move
circuit-fibsqrt
cobol-modernization
code-from-image
compile-compcert
configure-git-webserver
constraints-scheduling
count-dataset-tokens
crack-7z-hash
custom-memory-heap-crash
db-wal-recovery
distribution-search
dna-assembly
dna-insert
extract-elf
extract-moves-from-video
feal-differential-cryptanalysis
feal-linear-cryptanalysis
filter-js-from-html
financial-document-processor
fix-code-vulnerability
fix-git
fix-ocaml-gc
gcode-to-text
git-leak-recovery
git-multibranch
gpt2-codegolf
headless-terminal
hf-model-inference
install-windows-3-11
kv-store-grpc
large-scale-text-editing
largest-eigenval
llm-inference-batching-scheduler
log-summary-date-ranges
mailman
make-doom-for-mips
make-mips-interpreter
mcmc-sampling-stan
merge-diff-arc-agi-task
model-extraction-relu-logits
modernize-scientific-stack
mteb-leaderboard
mteb-retrieve
multi-source-data-merger
nginx-request-logging
openssl-selfsigned-cert
overfull-hbox
password-recovery
path-tracing
path-tracing-reverse
polyglot-c-py
polyglot-rust-c
portfolio-optimization
protein-assembly
prove-plus-comm
pypi-server
pytorch-model-cli
pytorch-model-recovery
qemu-alpine-ssh
qemu-startup
query-optimize
raman-fitting
regex-chess
regex-log
reshard-c4-data
rstan-to-pystan
sam-cell-seg
sanitize-git-repo
schemelike-metacircular-eval
sparql-university
sqlite-db-truncate
sqlite-with-gcov
torch-pipeline-parallelism
torch-tensor-parallelism
train-fasttext
tune-mjcf
video-processing
vulnerable-secret
winning-avg-corewars
write-compressor
```

Agent timeouts (`[agent] timeout_sec` of each task): 600 s × 1, 750 s × 1, 900 s × 49, 1,200 s × 6, 1,800 s × 16, 2,400 s × 2, 3,600 s × 12, 7,200 s × 1 (`sam-cell-seg`) and 12,000 s × 1 (`build-pov-ray`). Verifier timeouts range from 360 to 12,000 s. Every task allows internet access; 84 tasks have 1 CPU, 71 have 2 GB of memory, and none uses a GPU.

**Dry subset** (16 tasks with 900-s agent timeouts, chosen among the tasks with the largest prompts in earlier Terminus 2 runs; a check before the benchmark, not part of it):

```text
build-cython-ext
build-pmars
configure-git-webserver
crack-7z-hash
db-wal-recovery
extract-elf
fix-code-vulnerability
gcode-to-text
git-multibranch
make-doom-for-mips
password-recovery
pytorch-model-cli
raman-fitting
sanitize-git-repo
tune-mjcf
vulnerable-secret
```

The policies of this runtime on Terminal-Bench (README, policy table, and §5):

| `--policy` | History pages used for the next turn |
| --- | --- |
| `full` | Every historical page: the reference, and the configuration closest to stock vLLM |
| `observation_query` | Selector V2: mandatory, recent and top-cosine pages within the budget; the query is the hidden state at the end of the newest observation |
| `random_fill` | Control of `observation_query`: the same budget, mandatory and recent pages, the remaining capacity filled in a seeded random order |
| `goal_query` | Not configured for Terminal-Bench yet (its Goal setup is pending); the chat-text server refuses it |

Selection starts only once a session's sealed history exceeds 8,192 tokens (README §5); a trial whose history stays shorter runs every policy as `full`.

### A.2 Model and versions

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, a local directory with regular files; metadata SHA-256 values in README §2.2 |
| Goal rewriter | Not used: no Terminal-Bench server starts one |
| vLLM | The official wheel `vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl`, unmodified, SHA-256 `700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5` |
| Python environment | CPython 3.12, `transformers==5.13.0`, `tokenizers==0.22.2`, built with `examples/env/constraints-3270721.txt` (README §2.1). The stock control runs from the same environment; this runtime is installed into a separate release directory and leaves the environment unchanged (README §2.4) |
| Runtime | This repository at commit `68d19b4`, the commit of the validated dry run, or a later commit that leaves the chat-text protocol, the capacity gate and the vLLM arguments of A.3 unchanged. `0d37346` and earlier admit a turn unchecked while nothing runs (A.3) and are not used. Its chat-text protocol module, `source-overlay/remote_ra_live_v2/chat_text_protocol.py`, has SHA-256 `873ce935ddf91a4f3247469c177c7506ffabbed5649fdb4af29ecb87fe9bb338` |
| GPU | One NVIDIA GH200 120GB, Linux aarch64; one server per GPU at a time |
| Slurm resources | `--nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16 --mem=113560M` (one GH200's share of the node's host memory; A.3 explains why it suffices) |
| Agent software | Harbor `0.21.0` from its wheel, unmodified (`harbor-0.21.0.dist-info/RECORD` SHA-256 `abfaf4e154a499d7e3c8c83ba85a8ee8861a99f66b11b7d7cb035f10b85d1424`, 414 files), its built-in Terminus 2 version `2.0.0`, litellm `1.97.0`, openai `2.54.0`, Python 3.12 |
| Dataset | `terminal-bench/terminal-bench-2` at ref `sha256:c6fc2e2382c1dbae99b2d5ecd2f4f4a60c3c01e0d84642d69b4afd92e99d078b` |
| Evaluator | Harbor's own verifier, inside each trial (A.9) |

### A.3 This runtime: server flags and vLLM arguments

**Flags shared by the three policies** (`serve.py`):

| Flag | Value | Effect |
| --- | --- | --- |
| `--protocol` | `chat-text`, with `--chat-text-slots 512 --chat-text-idle-timeout 600 --chat-text-evict-after 30` (the defaults, pinned explicitly) | Plain OpenAI chat text for Terminus 2 (`chat_text_v1`, README §6.7). No `--task`: the 512 slots bind lazily to client sessions (89 trials, one session per summarization "questions" call, and rebinds) |
| `--max-running-sessions` | `20` | K: at most 20 sessions open. Harbor runs 16 trials at once; the 4 spare sessions take finished trials (Terminus sends no close) and the summary sub-calls (A.8). vLLM gets `--max-num-seqs 20` |
| `--kv-cache-gib` | `32` | vLLM gets `--kv-cache-memory-bytes 34359738368` |
| `--page-tokens` / `--kv-layout` | `16` / `decoupled` | 16-token attention pages and a separate Gated-DeltaNet pool of 41 slots (2K + 1; README §4.6). vLLM gets `--block-size 16 --mamba-block-size 784 --additional-config ...` |
| `--max-model-len` | Not passed: the default `65536` (**pending confirmation**) | Context limit, prompt plus reply. With 4096-token replies requested, a prompt holds at most 65,536 − 4,096 = 61,440 tokens (README §4.9) |
| `--max-tokens` | `4096` | Reply bound: the client's `max_tokens` from 1 to 4096 (Terminus sends 4096, A.6) |
| `--dram-arena-gib` | `64` | One pinned host page arena of 64 GiB for all sessions: 65,536 Pages of 1 MiB, for example 20 sessions of 52K tokens or 16 at the limit. A capture beyond it fails that request with `DRAM_ARENA_EXHAUSTED` (README §4.9) |
| `--connector-pool-gib` | `1` | The connector's unused CPU pool at 1 GiB instead of 32 (README §4.9). vLLM gets `cpu_bytes_to_use` 1073741824 |
| `--cuda-graphs` | `full` | vLLM gets `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}` instead of `--enforce-eager` (README §4.8) |
| `--decode-fastpath` | on | Less host work per decode step, the same values (README §4.11) |
| `--language-model-only` | on | No vision tower and no multimodal warmup (README §4.12). vLLM gets `--language-model-only` |
| `--release-idle-pages` | on | Under GPU pressure the capacity gate releases idle sessions' resident Pages and readmits them before their next turn (README §4.13; below). No vLLM argument |
| `--port` | `8000` | The server binds `127.0.0.1` only |

**Idle sessions and `--release-idle-pages`.** Between turns every open session keeps its resident Pages and its live tail on the GPU (README §4.13). Terminus sends no close, so up to 20 sessions stay open, most of them idle (finished trials, trials inside a long command, "questions" sessions), and their Pages fill the attention pool: in the first dry run, `full` without the switch, the idle sessions' pins held all but 69 of the 26,698 attention blocks (README §4.4). Without the switch the capacity gate can only wait for a running request to finish, and once every session is idle nothing frees a block. With it, the gate covers a turn's shortfall by releasing whole idle sessions' resident Pages, least recently used first (if the idle sessions together cannot cover it, nothing is released). Their host copies stay in the page arena, so no pinned host memory is added, and a released session's Pages come back from those copies before its next phase 1, which then runs as if they had never left. Since `68d19b4` a turn that still does not fit waits (`CAPACITY_WAIT`), also while nothing else runs; before, such a turn was admitted unchecked because nothing ran, its decode took the last free block, and vLLM preempted it, which ends the engine (README §4.4). If nothing can make room, every turn waits and the server stalls (A.11).

**Host memory.** The pinned host memory is the 1-GiB connector pool, 20 Gated-DeltaNet carries (2.87 GiB) and the 64-GiB arena: 67.9 GiB, against 84.1 GiB without the arena (README §6.7); `--release-idle-pages` adds none. With about 32 GiB for the processes this fits one GH200's share, `--mem=113560M` (110.9 GiB).

**Flags per policy:**

| Policy | Selector flags |
| --- | --- |
| `full` | none |
| `observation_query` | `--active-token-budget 8192` (**pending confirmation**) |
| `random_fill` | `--active-token-budget 8192` (**pending confirmation**; fill seed: the default 42) |

The budget is one value for both selective policies. No server sets `--decode-fastpath2`, `--decode-fastpath3`, `--gdn-device-handoff`, `--phase1-in-step`, `--admission-overlap`, `--goal-rewriter-model`, `--goal-rewriter-batch`, `--active-ratio` or `--fill-seed`. The budget and `--release-idle-pages` pin the startup configuration but add no vLLM argument, so the three servers start vLLM with the same arguments.

**Commands.** Dry runs of the three servers (add `--start` to serve; `LABEL` is a new label per attempt):

```sh
SHARED=(--protocol chat-text --chat-text-slots 512 --chat-text-idle-timeout 600 --chat-text-evict-after 30
        --kv-cache-gib 32 --page-tokens 16 --kv-layout decoupled --cuda-graphs full --max-running-sessions 20
        --max-tokens 4096 --dram-arena-gib 64 --connector-pool-gib 1 --decode-fastpath --language-model-only
        --release-idle-pages)
python -B serve.py --release-dir "$RELEASE" --policy full "${SHARED[@]}" \
  --output "$RUNS/tb-full-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --policy observation_query "${SHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/tb-observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --policy random_fill "${SHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/tb-random_fill-$LABEL" --port 8000
```

**Resulting vLLM arguments** (identical for the three servers; the dry run prints the authoritative list, here wrapped by argument group):

```text
--model <model directory> --served-model-name selective-prefix --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
--host 127.0.0.1 --port 8000 --distributed-executor-backend uni --no-async-scheduling
--tensor-parallel-size 1 --pipeline-parallel-size 1
--max-num-seqs 20 --max-model-len 65536 --max-num-batched-tokens 16384
--kv-cache-memory-bytes 34359738368 --seed 42 --enable-prefix-caching --prefix-caching-hash-algo sha256
--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}
--mamba-cache-mode align --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
--chat-template <output>/chat_template.jinja --chat-template-content-format string
--kv-transfer-config {"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":1073741824}}
--gdn-prefill-backend triton
--block-size 16 --mamba-block-size 784 --additional-config {"remote_ra_kv_layout":{"mamba_slots":41,"mamba_tokens":784,"page_tokens":16}}
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

**Generation contract, enforced by the server** (README §6.7). The server ignores the client's sampling fields:
- greedy decoding, at most the client's `max_tokens` (1 to 4096; without one, 4096 clamped to the context left), a stop on EOS/pad only (no stop string); `finish_reason` is `stop` on EOS/pad and `length` at the bound;
- thinking off: the reference rendering is the pinned chat template with `enable_thinking=false` and `preserve_thinking=true`; the generation header is `<|im_start|>assistant\n<think>\n\n</think>\n\n`. Terminus sends no system message, so the initial segment is its first user message;
- a request that repeats the previous request, the previous reply (compared after `strip()`) and one new user message continues the session: the previous reply keeps its exact generated token IDs, followed by `<|im_end|>\n` (`\n` after EOS/pad) and the new user message. Any other request (a summary hand-off, an unwound history, a resent request) rebuilds the session from text;
- the response content is the decoded reply with special tokens skipped, and `usage` carries the server's own prompt and completion token counts;
- a request whose prompt plus requested reply exceeds 65,536 tokens is refused with HTTP 400 and vLLM's own text, `This model's maximum context length is …`, before admission; nothing changes on the server.

### A.4 Stock vLLM control

The control is plain `vllm.entrypoints.openai.api_server` from the same environment (A.2), without any code of this runtime.

| Setting | Value | Relation to this runtime |
| --- | --- | --- |
| `--model` | The same Qwen3.6-27B directory | Same |
| `--served-model-name` | `selective-prefix` | Same name, so Terminus uses the same model name |
| `--host` / `--port` | `127.0.0.1` / `8000` | Same (loopback only) |
| `--distributed-executor-backend` / `--tensor-parallel-size` | `uni` / `1` | Same |
| `--max-num-seqs` | `16` | Harbor's `-n`. Differs: this runtime runs K = 20 (A.13) |
| `--max-model-len` | `65536` (**pending confirmation**, one value with A.3) | Same |
| `--max-num-batched-tokens` | `16384` | Same (fixed by the launcher) |
| `--kv-cache-memory-bytes` | `34359738368` (32 GiB) | Same bytes |
| `--seed` | `42` | Same |
| `--enable-prefix-caching --prefix-caching-hash-algo sha256` | on | Same |
| `--language-model-only` | on | Same (README §4.12) |
| `--mamba-cache-mode` | `align` | Same |
| `--gdn-prefill-backend` | `triton` | Same |
| CUDA graphs | vLLM's default: neither `--enforce-eager` nor `--compilation-config`, which gives `FULL_AND_PIECEWISE` with `VLLM_COMPILE` (README §4.8) | Same mode; capture sizes up to 32 against 40 (A.5) |
| Asynchronous scheduling | vLLM's default (enabled); no flag | Differs: the launcher always passes `--no-async-scheduling` (A.13) |
| Attention backend | Not set; vLLM selects it (A.5) | Same |
| Chat template | The model's own; vLLM's default chat template arguments on the server | Differs in the history rendering (A.13) |
| Thinking | Off, requested by the client per call (A.6) | Same generation header |

Every value is set explicitly, so that the control does not depend on vLLM's defaults for this hybrid model. The arguments that serve the runtime's own route (`--revision`, the reasoning and tool parsers, `--chat-template`, `--kv-transfer-config`, `--block-size`, `--mamba-block-size`, `--additional-config`) are not passed: the stock server uses the model's own chat template and returns the reply as plain `content`.

**Command**, in a shell with the environment of A.3:

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
"$VENV/bin/python" -I -B -m vllm.entrypoints.openai.api_server \
  --model "$MODELS/qwen3.6-27b" --served-model-name selective-prefix --host 127.0.0.1 --port 8000 \
  --distributed-executor-backend uni --tensor-parallel-size 1 \
  --max-num-seqs 16 --max-model-len 65536 --max-num-batched-tokens 16384 \
  --kv-cache-memory-bytes 34359738368 --seed 42 \
  --enable-prefix-caching --prefix-caching-hash-algo sha256 --language-model-only \
  --mamba-cache-mode align --gdn-prefill-backend triton
```

- **Environment.** That of A.3 (`PATH`, `LD_LIBRARY_PATH`, short `TMPDIR`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `CC` and `VLLM_RPC_BASE_PATH` unset, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`), plus the two variables that `serve.py` sets for this runtime (`VLLM_WORKER_MULTIPROC_METHOD=spawn`, `VLLM_USE_V2_MODEL_RUNNER=0`) and `VLLM_NO_USAGE_STATS=1`.
- **Generation contract, sent by the client.** Stock vLLM takes sampling from the request, so the agent's kwargs carry it (A.6): `temperature` 0.0, `max_tokens` 4096, no stop string (the reply ends at EOS or at `max_tokens`, `finish_reason` `length`), and, per call, `extra_body: {"chat_template_kwargs": {"enable_thinking": false}}` (Terminus 2's `llm_call_kwargs`). With `enable_thinking=false` the model's chat template is expected to end the prompt with the generation header of A.3, to add no text around a user message without a system message, and to render a past reply without the empty think block; B.3 checks this.
- **Readiness.** `GET /health` returns 200 and `GET /v1/models` lists `selective-prefix`.
- **Session close.** Stock vLLM has no close route, and nothing sends one: Terminus sends no close, and the job's close step (A.8) runs on this runtime only.
- **Preemption.** Stock vLLM can preempt requests when its KV pool is full; this runtime never preempts (A.13). Record vLLM's counter `vllm:num_preemptions_total` (`GET /metrics`) during and at the end of the run; the harness's check of the stock server requires it to stay 0.

### A.5 Kernel backends

Neither server passes `--attention-backend`; vLLM chooses the attention backend automatically on the GH200. Both servers use the same kernels:

| Component | Setting | Server log line |
| --- | --- | --- |
| Attention | FLASH_ATTN with FlashAttention 3 (vLLM's automatic choice) | `Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']` and `Using FlashAttention version 3` |
| Gated-DeltaNet prefill | `--gdn-prefill-backend triton` | `Using Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128)` |
| Sampler | Native top-k/top-p (`VLLM_USE_FLASHINFER_SAMPLER=0`) | `FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0` |
| CUDA graphs | `FULL_AND_PIECEWISE`. Capture sizes 1, 2, 4, 8, 16, 24, 32 and 40 at 20 sequences (this runtime); 1, 2, 4, 8, 16, 24 and 32 at 16 sequences (stock) (README §4.8) | `'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE` and `'cudagraph_capture_sizes': [...]` in the engine configuration line |
| Asynchronous scheduling | This runtime: off. Stock: on | `Asynchronous scheduling is disabled.` (this runtime) or `Asynchronous scheduling is enabled.` (stock) |
| Text only | `--language-model-only` on both (README §4.12) | vLLM's `running in text-only mode`; neither `Encoder cache will be initialized` nor `Multi-modal warmup completed` may appear (a `Using ... for MMEncoderAttention` line may) |

Confirm them from each server's log after READY:

```sh
grep -oE "Using FLASH_ATTN attention backend|Using FlashAttention version [0-9]+|Using Triton/FLA GDN prefill kernel \(requested=[a-z]+|FlashInfer top-p/top-k sampling disabled|Asynchronous scheduling is [a-z]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|'cudagraph_capture_sizes': \[[0-9, ]*\]|running in text-only mode|Encoder cache will be initialized|Multi-modal warmup completed" \
  server.log | sort | uniq -c
```

Every line of the table must appear, and no other attention backend, FlashAttention version, prefill kernel or CUDA-graph mode, and no multimodal startup line. A server whose log differs is not part of the comparison.

### A.6 Agent configuration

Identical for every server unless marked.

| Item | Value |
| --- | --- |
| Agent | Harbor 0.21.0's built-in Terminus 2 (`terminus-2`, `harbor.agents.terminus_2:Terminus2`, version 2.0.0), unchanged: its JSON action parser, its prompt template, summarization on with its proactive threshold of 8,000 tokens, its own retry layers. It sends no session header and no close |
| Harbor | 0.21.0 from its wheel, unmodified (every file equal to the wheel's RECORD, A.2), litellm 1.97.0, openai 2.54.0; `HARBOR_TELEMETRY=0`; no local patch of Harbor or its agents |
| Run | `harbor run -c job.json -y`, one Harbor job per server, with the job configuration below |
| Dataset | `terminal-bench/terminal-bench-2` at its pinned ref (A.2), filtered to the 89 names of A.1 (`terminal-bench/<name>`) |
| Model | `hosted_vllm/selective-prefix`, `api_base` `http://127.0.0.1:<port>/v1`, `temperature` 0.0, `model_info` (`max_input_tokens` 65536, **pending confirmation** with A.3; `max_output_tokens` 4096; both costs 0), `llm_kwargs` (`max_tokens` 4096, `timeout` 3600, `max_retries` 0, `num_retries` 0); the stock control adds `llm_call_kwargs` (A.4) |
| Trials | `n_concurrent_trials` 16, `n_attempts` 1, `retry.max_retries` 0 (no Harbor trial retry), `timeout_multiplier` 1.0 (every task's own timeouts), verification on, `quiet` |
| Tool environment | `docker`: each trial runs in containers from the task's image (A.1, B.1), with the task's own CPU, memory and internet settings; deleted after the trial (`delete: true`, `force_build: false`) |

Properties of Terminus 2 that affect results:
- Terminus runs inside the Harbor process on the agent host and drives each task container through tmux; its model calls come from the host, so a loopback endpoint suffices (B.4).
- There is no system message. The first message is a user message: the prompt template with the task instruction and the initial terminal screen, which shows the container's host name, so every trial's first message differs. Actions are JSON (`analysis`, `plan`, `commands` with keystrokes and durations, `task_complete`); completion needs a second confirmation. An observation is limited to 10,000 bytes (head and tail kept). A reply that cannot be parsed stays in the history, and the next prompt reports the parsing error.
- A reply with `finish_reason` `length` raises an output-length error in Harbor's LiteLLM wrapper; Terminus appends the truncated reply with an error prompt and asks again.
- Summarization: before an episode in which fewer than 8,000 tokens of `max_input_tokens` are left (by its own token estimate), and after a context-length error (it first unwinds the history), Terminus summarizes in three sub-calls (a summary on the current history, a "questions" call sent without history, the answers) and continues from a hand-off that replaces the history.
- Each task's agent timeout is wall-clock (`[agent] timeout_sec` × 1.0). At the timeout Harbor cancels the agent, with any model call in flight, and the trial records `AgentTimeoutError`.

**`job.json` for this runtime**, the content that `render_job.py` (B.2) writes at port 20000, formatted more compactly; `job_name` and `jobs_dir` are new for every attempt, and `task_names` holds all 89 names (shortened here):

```json
{
  "agents": [
    {
      "kwargs": {
        "api_base": "http://127.0.0.1:20000/v1",
        "llm_kwargs": {"max_retries": 0, "max_tokens": 4096, "num_retries": 0, "timeout": 3600},
        "model_info": {"input_cost_per_token": 0, "max_input_tokens": 65536, "max_output_tokens": 4096,
                       "output_cost_per_token": 0},
        "temperature": 0.0
      },
      "model_name": "hosted_vllm/selective-prefix",
      "name": "terminus-2"
    }
  ],
  "datasets": [
    {
      "name": "terminal-bench/terminal-bench-2",
      "ref": "sha256:c6fc2e2382c1dbae99b2d5ecd2f4f4a60c3c01e0d84642d69b4afd92e99d078b",
      "task_names": ["terminal-bench/adaptive-rejection-sampler", "terminal-bench/bn-fit-modify",
                     "...", "terminal-bench/write-compressor"]
    }
  ],
  "environment": {"delete": true, "force_build": false, "type": "docker"},
  "job_name": "full-a1",
  "jobs_dir": "/abs/path/on/agent/runs/full-a1/jobs",
  "n_attempts": 1,
  "n_concurrent_trials": 16,
  "quiet": true,
  "retry": {"max_retries": 0},
  "timeout_multiplier": 1.0,
  "verifier": {"disable": false}
}
```

**`job.json` for the stock control:** the same file with one more entry in the agent's `kwargs`, the per-call thinking switch of A.4:

```json
"llm_call_kwargs": {"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}
```

This runtime renders the thinking-off header itself and ignores the client's sampling fields, so its job configuration has no such entry; the two configurations differ only there.

### A.7 Limits

| Limit | Value |
| --- | --- |
| Agent timeout | Each task's `[agent] timeout_sec` × 1.0: 600 to 12,000 s (A.1) |
| Verifier timeout | Each task's own: 360 to 12,000 s |
| Reply bound | 4096 new tokens per model call: the client's `max_tokens` 4096 (this runtime: within the server's bound `--max-tokens 4096`; stock: from the client) |
| Context limit | 65,536 tokens, prompt plus reply (**pending confirmation**). A request whose prompt plus 4,096 exceeds it is refused with HTTP 400 and vLLM's text; Terminus then unwinds and summarizes (A.6). Terminus also summarizes proactively below 8,000 tokens left |
| Client timeout | 3600 s per model call (`llm_kwargs.timeout`) |
| Client retries | litellm and the OpenAI client: none (`max_retries` 0, `num_retries` 0). Terminus 2's own retry layers stay as shipped: up to 3 attempts in its LiteLLM wrapper, inside up to 3 in its query loop. Harbor: no trial retry, one attempt |
| Sessions per server | At most 20 open (K); 512 slots over the server's lifetime (client sessions, summarization "questions" sessions and rebinds); a slot serves 256 requests, then the session rebinds; lifetime request bound max(1,024, 256 × 512) = 131,072 (README §6.7) |
| Pinned page arena | 64 GiB, 65,536 Pages of 1 MiB; a capture beyond it fails that request (`DRAM_ARENA_EXHAUSTED`, a failure under A.11) |

**Why a 3600-s client timeout.** litellm's default request timeout is 600 s. On this runtime a client timeout disconnects the request: the server cancels it and retires its slot, and the next request of that trial rebinds to a fresh slot with a full prefill (README §6.7). A request can wait for an open session (when all 20 are open) or for KV capacity (`CAPACITY_WAIT`, README §4.4) before its reply of up to 4096 tokens starts; the 3600-s timeout keeps such waits from being cut and re-sent. The retries left in Terminus 2 resend after an error: on this runtime the resent request rebinds or rebuilds, and a resend while the first request is still in flight waits up to 30 s for it, then gets 409 `SESSION_BUSY`. The stock control uses the same settings, so that the two sides differ only in the server.

### A.8 Concurrency and session lifecycle

- Harbor runs at most 16 trials at once (`n_concurrent_trials` 16), each with its own Terminus 2 instance and task containers. When a trial ends (verifier included), Harbor starts the next one, in its own order.
- **Session binding on this runtime.** Terminus sends no `X-Session-ID`, so the server keys each client session by its initial segment, the first user message, which differs from trial to trial (A.6): one main session per trial. A summarization's "questions" call (sent without history) has another initial segment and is its own session; the summary and answers calls continue the main session or rebuild it. `key_sha256` = SHA-256 of `json.dumps([{"role": "user", "content": <first user message>}], ensure_ascii=False, sort_keys=True, separators=(",", ":"))` in UTF-8; it joins a trial's ATIF trajectory (step 1 `message`) to the server's rows.
- **Closes on this runtime.** Terminus sends no close. With K = 20 and 16 trials, finished trials and "questions" sessions use the 4 spares. A live session closes only when all 20 are open: while a new session waits, the least recently used session idle for at least 30 s closes (asides first), one per waiting request; with nobody waiting, the least recently used session idle for at least 600 s closes, one, so that the next new session finds a spare. While a spare exists no live session closes, however long it is idle, so a trial inside a long command keeps its session. Failed, rebound and bound-reached slots close at once (README §6.7). Under GPU pressure, `--release-idle-pages` frees idle sessions' resident Pages without closing the sessions (A.3).
- **After the run, on this runtime.** The job closes every client session that is still open, on the GPU node itself: `POST /v1/chat-text/close` with `{"key_sha256": "<key>"}` (or `{"session_id": ...}` for a session under a header) for each session whose last slot has a bind or rebind row in `chat_sessions.jsonl` and no close row, retried on HTTP 409 every 2 s for up to 60 s. Then no session is open at shutdown, which the cleanup receipts record (README §8).
- **Stock control.** No close at any time; stock vLLM keeps cached prefixes until it evicts them.
- **Harbor's record**, on the agent host, under `jobs/<job name>/`: `config.json`, `result.json` (the job) and one directory per trial with `result.json` (`started_at`, `finished_at`, the phase times `environment_setup`, `agent_setup`, `agent_execution` and `verifier`, `verifier_result.rewards.reward`, `exception_info`, the agent's token counts), `agent/trajectory.json` (the main ATIF trajectory, with `trajectory.cont-*.json` continuations), `agent/trajectory.summarization-*.json` (the summarization sub-calls) and `verifier/`.

### A.9 Evaluation

| Item | Value |
| --- | --- |
| Evaluator | Harbor's verifier, run inside each trial right after its agent (`verifier.disable: false`); there is no separate evaluation stage |
| Verdict | `verifier_result.rewards.reward` in the trial's `result.json` (also `verifier/reward.txt`); 1.0 is solved |
| Model | Not needed: the verifiers run without the server, but the server stays up until Harbor's job ends |
| Unfinished trials | A planned task whose trial has no finished result (cancelled, or never started) counts as unsolved |

### A.10 Metrics

**Accuracy** = trials with Harbor verifier reward 1.0 / 89 planned tasks × 100. One task is 1.12 points. At an accuracy of 50 % the binomial standard error of one run is about 5.3 points, and that of a difference between two independent runs about 7.5 points. Keep the per-task rewards so that runs can be compared task by task.

**Throughput** = ATIF main-agent completion tokens / the agents' span:

| Part | Definition |
| --- | --- |
| Numerator | The sum of `metrics.completion_tokens` over every agent step with a completion-token count in each trial's main ATIF trajectory, `agent/trajectory.json` and its continuations `agent/trajectory.cont-*.json`, over the trials of the 89 planned tasks. Terminus records these counts from each response's `usage` |
| Apart | The summarization sub-calls, `agent/trajectory.summarization-*.json` (the summary, "questions" and answers calls): reported beside the numerator, not in it |
| Denominator | The agents' span on the agent host's clock: the last `agent_execution.finished_at` minus the first `agent_execution.started_at` over the trials' `result.json` |
| Included | Everything between those two moments: tool execution, waits for the model and for open sessions, summarizations, rebuilds and rebinds, other trials' environment setup and verifiers, the server's own session closes |
| Excluded | Server start-up and shutdown, the environment setup before the first agent starts, the verifiers after the last agent ends, the job's closes after the run |
| Cross-checks | On this runtime, for a finished run: the client's completion tokens (main plus summarization), plus the replies delivered but not recorded (below), equal the server's `chat_text.completion_tokens` of the joined rows (A.8); vLLM's `vllm:generation_tokens_total` increase over the run is at least the client's total and at least the server rows' total. Harbor's job wall time (setup and verifiers included) is reported beside the span |

**Requests cut by an agent timeout.** When a trial reaches its agent timeout, Harbor cancels the model call in flight. The client never receives that reply, so its tokens are not in the ATIF trajectory and not part of the throughput numerator. On this runtime the cancelled request has a `failures.jsonl` row (`client disconnected`) and no `runtime.jsonl` row, but vLLM's counter includes what it generated; the server-side count reports that generation separately, as not delivered (vLLM's increase minus the server rows' completion tokens). A reply that reached Harbor just before the cancellation, but that Terminus had not yet recorded as a step, is missing from the trajectory as well; the join counts it from Harbor's own call record and reports it apart, as delivered but not recorded, outside the numerator too.

Record also, per run: per-trial exception (`AgentTimeoutError` and others), calls, summarizations and the largest prompt; per server the context-limit refusals, rebuilds, rebinds, `length` finishes and selective turns (this runtime), and the preemption counter (stock, A.4).

### A.11 Run rules

- **One run at a time on the agent host**, and nothing else on it while trials run (no other benchmark run, stress test or evaluation). Agent timeouts are wall-clock, so the agent host's load changes both throughput and accuracy. Use the same agent host type for every server that is compared.
- **A fresh server and a new output directory for every attempt**, and a new Harbor job name and jobs directory. `serve.py` and `tb_agents.sh` (B.2) refuse an existing output directory.
- **No resume across interrupted runs.** A run that ends early (the job's time limit, the agents' time cap, a tunnel drop, a server failure) is partial: its unfinished trials are not continued in another job, the run is excluded from comparisons and gives no throughput, and the whole server reruns from the start under a new label.
- **Failures.** A context-limit refusal (HTTP 400 before admission) is a normal outcome: Terminus summarizes and the server keeps serving. A request that its client cancels at a task's agent timeout is normal too. Any other failure on this runtime's server (an engine failure answered with 500, `DRAM_ARENA_EXHAUSTED`, `SESSION_BUSY`, `SESSION_LIMIT`, a refused request, a preemption) fails the server's checks, and the run is excluded.
- **Capacity stalls.** A turn that does not fit waits (`CAPACITY_WAIT`, A.3); if no turn fits, the server stalls. The harness stops the Harbor run when requests are pending and for 10 minutes no request completes and vLLM computes no token; the server fails, and the run is excluded.
- **Transport.** Terminus reaches the server through a loopback-only route (B.4). Any loopback-only route gives the same results; the server never listens beyond loopback (README §6.4).

### A.12 Alignment checklist

| # | Item | This runtime | A baseline uses |
| --- | --- | --- | --- |
| 1 | Model and revision | Qwen3.6-27B `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | The same |
| 2 | Software | vLLM 0.24.0 official wheel, Transformers 5.13.0, Tokenizers 0.22.2, `constraints-3270721.txt` | The same environment |
| 3 | Hardware and job | 1 × GH200 120GB, 16 CPUs, 113,560 MiB | The same |
| 4 | Tasks | The 89 tasks of A.1, dataset ref `sha256:c6fc2e23…` | The same |
| 5 | Concurrency | `-n 16`, one attempt, Harbor's order | The same |
| 6 | Context limit | 65536 (pending confirmation) | `--max-model-len 65536` and `max_input_tokens` 65536 |
| 7 | Batched tokens per step | 16384 | `--max-num-batched-tokens 16384` |
| 8 | Maximum sequences | 20 (K; at most 16 trials run) | `--max-num-seqs 16`; a known difference (A.13) |
| 9 | GPU KV | 32 GiB (34,359,738,368 bytes) | `--kv-cache-memory-bytes 34359738368` |
| 10 | Reply bound | 4096 (client and server) | `max_tokens` 4096 (client) |
| 11 | Decoding | Greedy (server) | `temperature` 0.0 |
| 12 | Thinking | Off (empty think block in the header) | `llm_call_kwargs` with `chat_template_kwargs: {"enable_thinking": false}` per call |
| 13 | Stop | EOS/pad only | No stop string |
| 14 | CUDA graphs | `FULL_AND_PIECEWISE` (`--cuda-graphs full`) | vLLM's default (`FULL_AND_PIECEWISE`) |
| 15 | Asynchronous scheduling | Off (always) | vLLM's default (on); a known difference (A.13) |
| 16 | Attention | FLASH_ATTN with FlashAttention 3, chosen by vLLM | The same, no `--attention-backend` |
| 17 | Gated-DeltaNet prefill | Triton | `--gdn-prefill-backend triton` |
| 18 | Sampler | Native (`VLLM_USE_FLASHINFER_SAMPLER=0`) | The same variable |
| 19 | Prefix caching | On, sha256 | `--enable-prefix-caching --prefix-caching-hash-algo sha256` |
| 20 | Mamba cache mode | `align` | `--mamba-cache-mode align` |
| 21 | Seed | 42 | `--seed 42` |
| 22 | Text only | `--language-model-only` | vLLM's `--language-model-only` |
| 23 | Agent | Harbor 0.21.0, Terminus 2 2.0.0 unchanged, litellm 1.97.0, openai 2.54.0 | The same |
| 24 | Agent behaviour | JSON parser, summarization at 8,000 tokens left, Terminus's own retries, no session header or close | The same (Terminus unchanged) |
| 25 | Model kwargs | `temperature` 0, `model_info` 65536 / 4096 / 0 / 0, `llm_kwargs` 4096 / 3600 s / 0 / 0 | The same, plus `llm_call_kwargs` (row 12) |
| 26 | Harbor job | `n_attempts` 1, no trial retry, `timeout_multiplier` 1.0, Docker with `delete`, verification on | The same |
| 27 | Agent host | One run at a time, nothing else on the host, task images pulled in advance | The same rule and the same host type |
| 28 | Evaluation | Harbor's verifier in each trial; reward 1.0 = solved | The same |
| 29 | Throughput | The definition of A.10 | The same numerator and clock |
| 30 | Partial runs | Excluded, never resumed | The same |

### A.13 Known differences between this runtime and stock vLLM

These differences remain when every checklist item matches. Report them with any comparison.

1. **Previous replies.** This runtime replays each previous reply as the exact token IDs it was generated with, behind the empty think block that the thinking-off header put in front of it. Stock vLLM re-renders the history from text through the model's template with its default arguments, which drop the empty think block of a past reply: each history reply is 4 tokens shorter there (`<think>`, `\n\n`, `</think>`, `\n\n`), and its text is re-encoded (README §6.7).
2. **Asynchronous scheduling.** This runtime runs without it (the launcher always passes `--no-async-scheduling`); stock vLLM runs with its default, enabled.
3. **Sampling control.** This runtime decodes greedily, ignores the client's sampling fields and stops on EOS/pad; stock vLLM takes the sampling from the request (`temperature` 0 gives greedy decoding).
4. **Context edge.** This runtime refuses a request whose prompt plus `max_tokens` exceeds 65,536 tokens with HTTP 400 and vLLM's text, `This model's maximum context length is …`, before admission; stock vLLM 0.24.0 applies its own check to such a request, with the same text (`vllm/renderers/params.py`). Not measured for the stock control.
5. **Sequences and KV management.** Both have 32 GiB of GPU KV, in different layouts. This runtime opens up to 20 sessions (`max_num_seqs` 20, graphs captured up to 40), with 16-token attention pages, a separate Gated-DeltaNet pool, a 1-GiB CPU pool and a 64-GiB pinned page arena; it never preempts, and a turn that does not fit waits (`CAPACITY_WAIT`, README §4.4), also while nothing else runs. Idle sessions keep their resident Pages on the GPU until `--release-idle-pages` releases them under pressure; a released session's Pages are copied back from host memory before its next turn (A.3). Stock vLLM runs up to 16 sequences in its own hybrid layout (README §4.6), keeps a finished request's blocks only as evictable prefix cache, recomputes what it evicted, and can preempt when the pool is full.
6. **Admission and scheduling.** This runtime admits one request at a time in arrival order, runs each admission's phase 1 as its own forward pass, then decodes in batches (README §4.4, §5). Stock vLLM schedules prefill chunks together with decode steps within the 16,384-token step budget.
7. **Session lifecycle.** This runtime keeps a session's state until its own close rules or the job close it (A.8), its GPU Pages until they are released under pressure (item 5); a session that reaches 256 requests, or whose slot failed, moves to a fresh slot with a full prefill. Stock vLLM keeps cached prefixes until it evicts them.
8. **Rebuilds.** On this runtime a request that does not extend the previous one (a summary hand-off, an unwound history, a resend) rebuilds the session from text with a full prefill (README §6.7); stock vLLM serves it with prefix caching.
9. **Selective policies.** `observation_query` and `random_fill` attend to a selected subset of history pages once a session's sealed history exceeds 8,192 tokens; `full` keeps every page. Shorter trials run as `full`.
10. **Wall-clock timeouts.** Agent timeouts are wall-clock, so accuracy depends on each server's speed and on the agent host's load, as well as on the model's outputs.
11. **Run-to-run variation.** Batched runs are not token-identical from run to run on either server: a reply can change with the batch composition, which depends on timing (README §4.4).

---

## Part B. Procedure

This part runs the configuration of Part A with the tools of this repository: the model on a GH200 node in a Slurm job, Harbor and its task containers on a separate x86_64 Docker host (the agent host), and a reverse SSH tunnel from the GPU job to the agent host (README §6.4). Every step applies to the three policies and to the stock control unless marked.

### B.1 Agent host

- **Requirements.** Linux x86_64 with Docker, `curl`, `ss`, `python3` and `uv`, a Python 3.12 environment with Harbor and its dependencies at the versions of A.2, and a checkout of this repository. The host runs 16 trials and their containers at once (each task's limits: 1 to 4 CPUs, 2 to 8 GB of memory). Harbor resolves the dataset's versions online when a job starts, and every task container needs internet access. Inbound SSH from the GPU nodes is needed for the tunnel (B.4). The harness's agent host has 16 vCPUs and 30 GB of RAM, and its host check requires at least 12 GB of available memory and 50 GB of free disk before a run.
- **Images.** Each task's package names its image: `alexgshaw/<task>:20251031`, except `install-windows-3-11`, whose image is `alexgshaw/install-windows-3.11:20251031`.

Environment and directories (the commands are not part of the validated runs), with a checkout of this repository at `$AGENT/vllm-0.24.0-selective-prefix-policies`:

```sh
AGENT=/abs/path/on/agent
HARBOR=$AGENT/harbor-env
uv venv --python 3.12 "$HARBOR"
uv pip install --python "$HARBOR/bin/python" harbor==0.21.0 litellm==1.97.0 openai==2.54.0
mkdir -p "$AGENT/sock" "$AGENT/runs" "$AGENT/tb"
chmod 700 "$AGENT/sock"
```

Save the list of A.1 as `$AGENT/tasks.txt` (and the dry subset as `$AGENT/tasks-dry.txt`), and pull the 89 images before any run, so that no pull enters the measured time:

```sh
while read -r t; do
  img="alexgshaw/${t/install-windows-3-11/install-windows-3.11}:20251031"
  if docker pull "$img" > /dev/null; then echo "ok $img"; else echo "FAILED $img"; fi
done < "$AGENT/tasks.txt"
```

### B.2 Agent-side files

Three files in `$AGENT/tb`: the loopback relay of the repository, a renderer of `job.json` and the script that runs one server's Harbor job. Copy the relay:

```sh
cp "$AGENT/vllm-0.24.0-selective-prefix-policies/examples/swebench/relay.py" "$AGENT/tb/relay.py"
```

`$AGENT/tb/render_job.py` writes the job configuration of A.6 into a new output directory:

```python
"""Write Harbor's job.json of one Terminal-Bench server run (docs/TERMINALBENCH_BENCHMARK.md, A.6)."""
import json
import sys
from pathlib import Path

kind, port, out, tasks = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
names = [line.strip() for line in tasks.read_text().splitlines() if line.strip()]
assert kind in ('runtime', 'stock') and names and len(names) == len(set(names))
kwargs = dict(api_base='http://127.0.0.1:%d/v1' % port, temperature=0.0,
              model_info=dict(max_input_tokens=65536, max_output_tokens=4096, input_cost_per_token=0,
                              output_cost_per_token=0),
              llm_kwargs=dict(max_tokens=4096, timeout=3600, max_retries=0, num_retries=0))
if kind == 'stock':
    kwargs['llm_call_kwargs'] = {'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
config = dict(job_name=out.name, jobs_dir=str(out / 'jobs'), n_attempts=1, timeout_multiplier=1.0,
              n_concurrent_trials=16, quiet=True, retry=dict(max_retries=0),
              environment=dict(type='docker', delete=True, force_build=False), verifier=dict(disable=False),
              agents=[dict(name='terminus-2', model_name='hosted_vllm/selective-prefix', kwargs=kwargs)],
              datasets=[dict(name='terminal-bench/terminal-bench-2',
                             ref='sha256:c6fc2e2382c1dbae99b2d5ecd2f4f4a60c3c01e0d84642d69b4afd92e99d078b',
                             task_names=['terminal-bench/' + name for name in names])])
(out / 'job.json').write_text(json.dumps(config, sort_keys=True, indent=2) + '\n')
```

`$AGENT/tb/tb_agents.sh` (make it executable) runs one server's Harbor job: it creates the output directory, renders `job.json`, starts the relay, waits for the endpoint, checks the listener and runs Harbor. It is an example and **untested**:

```bash
#!/usr/bin/env bash
# Agent side of one Terminal-Bench server run (docs/TERMINALBENCH_BENCHMARK.md, B.2).
# Usage: tb_agents.sh <socket> <port> <output dir> <runtime|stock> <task file>
set -euo pipefail
: "${AGENT:?}" "${HARBOR:?}"
SOCK=$1; PORT=$2; OUT=$3; KIND=$4; TASKS=$5
mkdir "$OUT"                                          # refuses an existing directory (A.11)
python3 -I -B "$AGENT/tb/render_job.py" "$KIND" "$PORT" "$OUT" "$TASKS"
python3 -I -B "$AGENT/tb/relay.py" "$SOCK" "$PORT" > "$OUT/relay.log" 2>&1 &
RELAY=$!
trap 'kill "$RELAY" 2> /dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" -o "$OUT/models.json" && break
  sleep 5
done
grep -q '"selective-prefix"' "$OUT/models.json"
listeners=$(ss -ltnH "sport = :$PORT" | awk '{print $4}' | sort -u | tr '\n' ' ')
[[ $listeners == "127.0.0.1:$PORT " ]] || { echo "port $PORT listens on: $listeners" >&2; exit 3; }
cd "$OUT"
HARBOR_TELEMETRY=0 PYTHONDONTWRITEBYTECODE=1 "$HARBOR/bin/harbor" run -c "$OUT/job.json" -y \
  > "$OUT/harbor.log" 2>&1 < /dev/null
```

Before the first run, render both configurations once into scratch directories and compare them with A.6.

### B.3 Environment check before GPU time

**Images.** Every image of B.1 must be present locally:

```sh
while read -r t; do
  img="alexgshaw/${t/install-windows-3-11/install-windows-3.11}:20251031"
  docker image inspect "$img" > /dev/null 2>&1 || echo "MISSING $img"
done < "$AGENT/tasks.txt"
```

**Harbor and Terminus 2.** The versions of A.2 and the defaults that A.6 relies on (untested):

```sh
"$HARBOR/bin/python" -I -B - <<'PY'
import inspect, json
from importlib.metadata import version
from harbor.agents.terminus_2.terminus_2 import Terminus2
params = inspect.signature(Terminus2.__init__).parameters
names = ('parser_name', 'enable_summarize', 'proactive_summarization_threshold', 'session_id', 'llm_call_kwargs')
print(json.dumps(dict(versions={n: version(n) for n in ('harbor', 'litellm', 'openai')},
                      defaults={k: repr(params[k].default) for k in names if k in params})))
PY
```

Expected: `harbor` 0.21.0, `litellm` 1.97.0, `openai` 2.54.0; `parser_name` `'json'`, `enable_summarize` `True`, `proactive_summarization_threshold` `8000`, `session_id` `None`, and an `llm_call_kwargs` parameter (the stock control needs it).

**Thinking-off rendering of the stock control.** With the environment of A.3 (no GPU needed), the model's chat template with `enable_thinking=false` (untested):

```sh
"$VENV/bin/python" -I -B - "$MODELS/qwen3.6-27b" <<'PY'
import sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
def render(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
print(repr(render([{'role': 'user', 'content': 'u'}])))
print(repr(render([{'role': 'user', 'content': 'u'}, {'role': 'assistant', 'content': 'a'},
                   {'role': 'user', 'content': 'v'}])))
PY
```

The first line must be `'<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'` (no default system text, the header of A.3); the second `'<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\na<|im_end|>\n<|im_start|>user\nv<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'` (the past reply without a think block, A.13).

### B.4 Tunnel

The GPU job opens a reverse SSH tunnel after READY whose far end is a Unix socket on the agent host; `tb_agents.sh` exposes it at `127.0.0.1:<port>` through `relay.py` and refuses to run if anything but `127.0.0.1` listens on that port (README §6.4). The same form serves the stock control. It needs:
- a key that the agent host accepts for the tunnel user, readable by the GPU job, and a known-hosts file that pins the agent host's key (the job never uses an SSH agent);
- outbound SSH from the compute nodes to the agent host (check once from a compute node before the first run);
- a new socket path for every server, at most 100 bytes, in a `0700` directory on the agent host (`$AGENT/sock`).

A tunnel drop ends the run (A.11); the job does not reconnect.

### B.5 Slurm job script

The script below runs one server of the benchmark in one Slurm job: the server, READY, the kernel lines, the tunnel, the Harbor job on the agent host over SSH, the close of the sessions left open, and the shutdown. It is derived from `examples/slurm/serve_gh200.sh`, which does not pass the chat-text flags, `--cuda-graphs`, `--decode-fastpath`, `--connector-pool-gib`, `--language-model-only` or `--release-idle-pages`. It is an example and **untested**. Save it as `tb_job.sh`:

```bash
#!/usr/bin/env bash
# One server of the Terminal-Bench 2.0 benchmark (docs/TERMINALBENCH_BENCHMARK.md) in one Slurm job.
# POLICY: full | observation_query | random_fill (this runtime) | stock (stock vLLM 0.24.0).
set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}" || { echo "run this under sbatch" >&2; exit 2; }
: "${REPO:?}" "${VENV:?}" "${POLICY:?}" "${RUN_ROOT:?}" "${AGENT_SECONDS:?}"
: "${AGENT_HOST:?}" "${TUNNEL_KEY:?}" "${KNOWN_HOSTS:?}" "${AGENT_SOCKET:?}" "${AGENT_PORT:?}"
: "${AGENT_RUN:?}" "${AGENT_OUT:?}" "${AGENT_ENV:?}" "${AGENT_TASKS:?}"
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
SHARED=(--protocol chat-text --chat-text-slots 512 --chat-text-idle-timeout 600 --chat-text-evict-after 30
        --kv-cache-gib 32 --page-tokens 16 --kv-layout decoupled --cuda-graphs full --max-running-sessions 20
        --max-tokens 4096 --dram-arena-gib 64 --connector-pool-gib 1 --decode-fastpath --language-model-only
        --release-idle-pages)
case $POLICY in
  full)                          POLICY_FLAGS=(); KIND=runtime ;;
  observation_query|random_fill) POLICY_FLAGS=(--active-token-budget 8192); KIND=runtime ;;
  stock)                         POLICY_FLAGS=(); KIND=stock ;;
  *) echo "unknown POLICY: $POLICY" >&2; exit 2 ;;
esac
OUTPUT="$STAGE/$POLICY"
if [[ $KIND == stock ]]; then
  : "${MODEL:?}"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
  "$PY" -I -B -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name selective-prefix --host 127.0.0.1 --port "$PORT" \
    --distributed-executor-backend uni --tensor-parallel-size 1 \
    --max-num-seqs 16 --max-model-len 65536 --max-num-batched-tokens 16384 \
    --kv-cache-memory-bytes 34359738368 --seed 42 \
    --enable-prefix-caching --prefix-caching-hash-algo sha256 --language-model-only \
    --mamba-cache-mode align --gdn-prefill-backend triton > "$STAGE/server.log" 2>&1 &
else
  : "${RELEASE:?}"
  SERVE=("$PY" -B "$REPO/serve.py" --release-dir "$RELEASE" --policy "$POLICY"
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
  if [[ $KIND == stock ]]; then
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
grep -oE "Using FLASH_ATTN attention backend|Using FlashAttention version [0-9]+|Using Triton/FLA GDN prefill kernel \(requested=[a-z]+|FlashInfer top-p/top-k sampling disabled|Asynchronous scheduling is [a-z]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|'cudagraph_capture_sizes': \[[0-9, ]*\]|running in text-only mode|Encoder cache will be initialized|Multi-modal warmup completed" \
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

# The Harbor job (Part A.6 to A.8): the task file's trials at -n 16 on the agent host.
agents_rc=0
if (( up == 1 )); then
  timeout "$AGENT_SECONDS" ssh -T "${SSH_OPTS[@]}" "$AGENT_HOST" env $AGENT_ENV "$AGENT_RUN" "$AGENT_SOCKET" \
    "$AGENT_PORT" "$AGENT_OUT" "$KIND" "$AGENT_TASKS" > "$STAGE/agents.log" 2>&1 || agents_rc=$?
  if (( agents_rc == 124 )); then echo "agent time limit reached: the run is partial (B.7)" >&2; fi
else
  echo "tunnel not up (see $STAGE/tunnel.log)" >&2; agents_rc=1
fi

# Close the client sessions the run left open (this runtime; Terminus 2 sends no close, Part A.8).
if [[ $KIND == runtime ]]; then
  "$PY" -I -B - "$OUTPUT" "$PORT" > "$STAGE/closes.jsonl" <<'PY' || true
import json, sys, time, urllib.error, urllib.request
out, port = sys.argv[1], sys.argv[2]
last = {}
with open(out + '/chat_sessions.jsonl', encoding='utf-8') as stream:
    for line in stream:
        event = json.loads(line) if line.strip() else {}
        name = event.get('client_session')
        if event.get('event') in ('bind', 'rebind'):
            last[name] = dict(event, open=True)
        elif event.get('event') == 'close' and name in last and last[name]['slot'] == event.get('slot'):
            last[name]['open'] = False
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for session in (s for s in last.values() if s['open']):
    body = {'session_id': session['session_id']} if session.get('session_id') else {'key_sha256': session['key_sha256']}
    deadline = time.monotonic() + 60
    while True:
        request = urllib.request.Request('http://127.0.0.1:%s/v1/chat-text/close' % port, method='POST',
                                         data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with opener.open(request, timeout=120) as response:
                status = response.status
        except urllib.error.HTTPError as error:
            status = error.code
        if status != 409 or time.monotonic() >= deadline:
            break
        time.sleep(2)
    print(json.dumps(dict(client_session=session['client_session'], slot=session['slot'], http=status)))
PY
fi
stop_tunnel

# Stop: SIGTERM to the server PID only; this runtime must exit 0 with clean receipts (README section 8).
kill -TERM "$SERVER"
set +e; wait "$SERVER"; server_rc=$?; set -e
echo "server exit $server_rc, agents exit $agents_rc"
if [[ $KIND == runtime ]]; then
  (( server_rc == 0 )) || exit 1
  "$PY" -B "$REPO/examples/slurm/server_checks.py" cleanup "$OUTPUT"
  "$PY" -I -B -c 'import json,sys; v=json.load(open(sys.argv[1])); sys.exit((v.get("session_pool") or {}).get("open_at_shutdown") != [])' \
    "$OUTPUT/engine-cleanup.json" || { echo "sessions open at shutdown" >&2; exit 1; }
fi
exit "$agents_rc"
```

Submit one job per server, with the Quickstart placeholders of the README and settings in the submitting shell (`--export=ALL`). The example below submits `observation_query`:

```sh
LABEL=a1                                        # a new label for every attempt
POLICY=observation_query
export REPO VENV RELEASE POLICY
export MODEL="$MODELS/qwen3.6-27b"
export RUN_ROOT=/abs/short/jobs AGENT_SECONDS=16380
export AGENT_HOST=user@agent-host TUNNEL_KEY=/abs/path/to/tunnel_key KNOWN_HOSTS=/abs/path/to/agent_known_hosts
export AGENT_SOCKET="/abs/path/on/agent/sock/$LABEL-$POLICY.sock" AGENT_PORT=20000
export AGENT_RUN=/abs/path/on/agent/tb/tb_agents.sh AGENT_OUT="/abs/path/on/agent/runs/$POLICY-$LABEL"
export AGENT_TASKS=/abs/path/on/agent/tasks.txt
export AGENT_ENV="AGENT=/abs/path/on/agent HARBOR=/abs/path/on/agent/harbor-env"
sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus=1 \
  --cpus-per-task=16 --mem=113560M --time=05:02:00 --job-name="tb-$POLICY" \
  --output="$RUN_ROOT/slurm-%j.out" --export=ALL --mail-type=BEGIN,END,FAIL --mail-user="$MAIL" \
  /abs/path/to/tb_job.sh
```

- **Stock control:** `POLICY=stock`, `AGENT_SECONDS=16320` and `--time=04:53:00`.
- **Time limits.** `--time` covers server start-up (READY 204–231 s after the start for the first server of a job in the validated jobs, README §4.1, plus about a minute of torch.compile on a cold cache, README §4.8), the Harbor job (`AGENT_SECONDS`), the closes and the shutdown. The Harbor job alone needs more than `build-pov-ray`'s 12,000-s agent timeout plus its setup and verifier. The harness gives each 89-task server a Harbor cap of 16,380 s (stock 16,320 s) inside a time limit of 05:02:00 (stock 04:53:00), about 1.3 times its expected time, and refuses to start a Harbor job with less than 12,600 s left. A run cut by `AGENT_SECONDS` or by the job's time limit is partial (A.11).
- **Dry subset.** `AGENT_TASKS=/abs/path/on/agent/tasks-dry.txt` with `AGENT_SECONDS=1260` runs the 16 tasks of the dry subset (A.1). The harness runs `full` and then `observation_query` in one dry job of 01:26:00, or `observation_query` alone in one of 00:52:00, with a Harbor cap of 1,260 s per server.
- **Stalls.** The script has no stall watchdog. The harness's watchdog stops the Harbor run when requests are pending and for 10 minutes no request completes and vLLM computes no token (A.11); without one, a stalled run lasts until `AGENT_SECONDS` or the job's time limit.
- `RUN_ROOT` must be short: `TMPDIR` lives under it. `AGENT_SOCKET` and `AGENT_OUT` must be new for every job.

### B.6 Readiness check

- **This runtime.** READY is `server_checks.py ready` exit 0 (README §4.1); `/v1/models` answering is not READY. `dry-run.json` holds the vLLM command, which must match A.3. **Do not send test chat requests to this runtime's server:** every new first message binds a slot and opens a client session that matches no trial (A.8), which the join and the checks then count.
- **Stock control.** `GET /health` returns 200 and `GET /v1/models` lists `selective-prefix`.
- **Kernel lines.** `kernels.txt` in the job directory must show the lines of A.5.
- **From the agent host,** before any trial starts (the job script performs the first check, `tb_agents.sh` the second):

```sh
curl -sf --unix-socket "$AGENT/sock/$LABEL-$POLICY.sock" http://localhost/v1/models   # lists selective-prefix
ss -ltn 'sport = :20000'                                                             # only 127.0.0.1 listens
```

### B.7 Running the Harbor job

- Run the four servers one after another, one job each, and never two runs on the agent host at once (A.11).
- Progress on the agent host: one directory per trial under `$AGENT_OUT/jobs/<job name>/` (a trial has finished when its `result.json` has `finished_at`), at most 16 trials' containers in `docker ps`. On this runtime, `curl -s http://127.0.0.1:20000/v1/chat-text/info` on the agent host returns the open sessions, the sessions waiting to open and the slots used so far; on the GPU node, `CAPACITY_WAIT` lines in `server.log` show turns waiting for room, and `capacity.jsonl` in the server's output directory lists the releases and readmissions of idle sessions' Pages.
- **A cut run.** An SSH drop or the job's `timeout` does not stop Harbor on the agent host. Stop that run's `harbor` process with `SIGTERM` (Harbor cancels its trials and removes their containers), stop its `relay.py`, then remove only the containers of that run's trials that are left, if any (untested). The run is partial; rerun the whole server under a new label (A.11).

### B.8 Accuracy

Harbor has already run every verifier (A.9). On the agent host, after the run has ended (untested):

```sh
python3 - "$AGENT/runs/$POLICY-$LABEL" "$AGENT/tasks.txt" <<'PY'
import json, sys
from pathlib import Path
out, tasks = Path(sys.argv[1]), Path(sys.argv[2])
planned = [line.strip() for line in tasks.read_text().splitlines() if line.strip()]
trials = {}
for path in sorted(out.glob('jobs/*/*/result.json')):
    result = json.loads(path.read_text())
    name = ((result.get('task_id') or {}).get('name') or str(result.get('task_name') or '')).split('/')[-1]
    trials.setdefault(name, []).append(result)
finished = solved = 0
for name in planned:
    result = (trials.get(name) or [{}])[-1]
    exception = (result.get('exception_info') or {}).get('exception_type')
    done = bool(result.get('finished_at')) and exception != 'CancelledError'
    reward = ((result.get('verifier_result') or {}).get('rewards') or {}).get('reward')
    finished += done
    solved += done and reward == 1.0
    print(json.dumps(dict(task=name, finished=done, reward=reward, exception=exception)))
print(json.dumps(dict(planned=len(planned), finished=finished, solved=solved,
                      accuracy=round(100 * solved / len(planned), 2))))
PY
```

Accuracy is the last line's `accuracy` (A.10). A complete run has a finished trial for every planned task; a run with fewer is partial (A.11). Harbor's job directory holds prompts, terminal output and trajectories; keep it out of public releases.

### B.9 Throughput

The throughput of A.10 from one run's output directory (untested):

```sh
python3 - "$AGENT/runs/$POLICY-$LABEL" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
def epoch(value):
    if not value:
        return None
    moment = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).timestamp()
def completion_tokens(paths):
    total = 0
    for path in paths:
        for step in json.loads(path.read_text()).get('steps') or ():
            tokens = (step.get('metrics') or {}).get('completion_tokens')
            if step.get('source') == 'agent' and isinstance(tokens, int):
                total += tokens
    return total
out = Path(sys.argv[1])
starts, ends, main, summary, trials = [], [], 0, 0, 0
for path in sorted(out.glob('jobs/*/*/result.json')):
    result, agent = json.loads(path.read_text()), path.parent / 'agent'
    trials += 1
    phase = result.get('agent_execution') or {}
    if epoch(phase.get('started_at')) is not None:
        starts.append(epoch(phase.get('started_at')))
    if epoch(phase.get('finished_at')) is not None:
        ends.append(epoch(phase.get('finished_at')))
    main += completion_tokens([p for p in [agent / 'trajectory.json'] if p.is_file()]
                              + sorted(agent.glob('trajectory.cont-*.json')))
    summary += completion_tokens(sorted(agent.glob('trajectory.summarization-*.json')))
span = max(ends) - min(starts)
print(json.dumps(dict(trials=trials, main_completion_tokens=main, summarization_completion_tokens=summary,
                      span_s=round(span, 3), tokens_per_s=round(main / span, 3))))
PY
```

A complete run has 89 trials; a run with fewer is partial (A.11).
