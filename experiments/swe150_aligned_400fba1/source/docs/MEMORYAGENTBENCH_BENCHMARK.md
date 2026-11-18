# MemoryAgentBench benchmark configuration

English | [简体中文](MEMORYAGENTBENCH_BENCHMARK.zh-CN.md)

This document fixes the configuration of the MemoryAgentBench benchmark run of this runtime: 32 cases of MemoryAgentBench data, 8 for each of its four abilities, each a long history that the agent reads chunk by chunk through a `read_next` tool before it gets one question; Qwen3.6-27B with thinking off; at most 16 cases at a time on one server with the kv-session protocol (`--protocol kv-session`, [docs/running.md section 6.6](running.md#66-memoryagentbench-kv-session-protocol)); one server per policy for `observation_query` and `goal_query`; and a stock vLLM 0.24.0 server, reached through the client's Vanilla transport, as the control. **Part A** is the configuration of record. A baseline (the stock vLLM control or any other system) is comparable only if it reproduces every item of the checklist in A.13. **Part B** is a manual procedure with the tools of this repository, among them the client and the case builder of `examples/memoryagentbench/`. It is optional, and it needs two inputs from outside this repository: your download of the public dataset `cxkjntmngm/MemAgent_Bench`, from which `cases.py build` makes the cases (A.1), and a checkout of the official MemoryAgentBench repository (A.2).

**A dated configuration of record.** Part A is the configuration of the 32-case Qwen runs of 2026-10-18 (G4, runtime `f9322a6`), kept as their record with its commands as written. The coming runs use the current configuration of [docs/status.md](status.md#current-configuration), which differs from Part A in these settings (the context of 262,144 tokens, the 4,096-token reply bound and 16 sessions are the same):

| Setting | Part A (record) | Current configuration |
| --- | --- | --- |
| Thinking | off (`--kv-session-thinking off`) | on |
| Scheduling | synchronous (the launcher's `--no-async-scheduling`) | asynchronous (`--async-scheduling`) |
| GPU KV | 32 GiB | 25 GiB |
| Per-session budget | 32,768 tokens | 8,192 tokens |
| Decode switches | `--decode-fastpath` alone (`--no-decode-fastpath2 --no-deferred-completion-io`) | the speed defaults |

Results with thinking on are not directly comparable with the thinking-off runs of Part A ([docs/known-limits.md](known-limits.md) 4.3).

The Gemma 4 31B IT profile ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)) serves the kv-session protocol with Gemma's turn grammar and thinking off ([docs/running.md section 6.6](running.md#66-memoryagentbench-kv-session-protocol)). A.15 gives the settings of its first MemoryAgentBench runs, `observation_query` and `goal_query` on 8 of the 32 cases (jobs 3429503 and 3429507); its later runs and their settings are in [docs/status.md](status.md#gemma-4-31b-it). Everything else in this document is the Qwen3.6-27B configuration.

Section references name the page that holds the section, such as [docs/running.md section 6.6](running.md#66-memoryagentbench-kv-session-protocol); the README's [documentation map](../README.md#documentation-map) lists the pages. Three harnesses appear on this page. "The job harness" is the project's harness that ran the GPU jobs (their plans, checks, caps, warm-up and records), which is not public. "The read_next harness" is the agent loop with its prompts and the MemAgent32 case selection and chunking, the design of Shanbin Sun, public in `examples/memoryagentbench/` since integrate-1107 (below). "The official harness" is the public MemoryAgentBench repository (A.2).

> **The client in this repository (integrate-1107).** The runner of A.7, its cases and its scorer are now `examples/memoryagentbench/` (`run_cases.py`, `kv_client.py`, `cases.py`, `score.py`), synced from the release repository with its page [benchmarks/memoryagentbench.md](benchmarks/memoryagentbench.md) (setup, procedure, server commands) and its test `test_mab_client.py` (CPU only). The read_next harness (the agent loop and its prompts, and the MemAgent32 case selection and chunking) is the design of Shanbin Sun; `examples/memoryagentbench/` holds it as this project runs it. The MIT notices of the strings it takes from MemoryAgentBench and mini-swe-agent are in [THIRD_PARTY_NOTICES.txt](../THIRD_PARTY_NOTICES.txt).

## Validation status

The GPU jobs of this benchmark, Qwen3.6-27B and Gemma 4 31B IT, each with its runtime commit, verdict, audit mark and date, are in [docs/status.md](status.md) (its MemoryAgentBench rows); the record up to runtime `cf2b28d` is frozen in [docs/history/validation-cf2b28d.md](history/validation-cf2b28d.md), and this section's former narrative of the runs (up to 2026-11-16) in [docs/history/memoryagentbench-validation.md](history/memoryagentbench-validation.md). The runs report verdicts, not scores: scoring is deferred ([docs/known-limits.md](known-limits.md) 5.1). The scripts of Part B are examples, untested as written ([docs/known-limits.md](known-limits.md) 5.4); `run_cases.py`, `cases.py` and `score.py` of `examples/memoryagentbench/` have CPU tests (`tests/test_mab_client.py`).

## Configuration for aligning a baseline

What our runs use, for a baseline to set the same way; every value is that of Part A (A.13 is the full checklist, A.14 lists the differences that remain):

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, on one GH200 120GB with the official vLLM 0.24.0 wheel; chat template SHA-256 `e84f32a2…` (A.2) |
| Thinking | Off: `--kv-session-thinking off`, the 7-token generation header that the template appends with `enable_thinking=False`; the client renders with `enable_thinking=False` for both servers (A.3, A.7) |
| Decoding | Greedy: the client sends `temperature` 0, `top_p` 1, `top_k` −1, `repetition_penalty` 1 and `seed` 42; vLLM `--seed 42`; no stop string (A.5, A.7) |
| Context limit | `--max-model-len 262144`, prompt plus reply (A.4, A.5) |
| Reply bound | 4096: the client's `max_tokens` 4096 (A.4, A.8) |
| Per-session KV budget | `--active-token-budget 32768` for `observation_query` and `goal_query`; none for exact attention (A.4, A.13) |
| Concurrency | 32 cases, at most 16 running, largest first, refilled in order; `--max-num-seqs 16` (A.1, A.9) |
| KV cache of this runtime | 32 GiB of GPU KV: `--kv-cache-gib 32`, 34,359,738,368 bytes (A.4) |
| Vanilla control (vLLM) | `--kv-cache-memory-bytes 34359738368` (32 GiB), `--max-model-len 262144`, `--max-num-batched-tokens 16384`, `--max-num-seqs 16`; CUDA graphs: vLLM's default (`FULL_AND_PIECEWISE`); asynchronous scheduling: vLLM's default (on); `--enable-prefix-caching --prefix-caching-hash-algo sha256` with a `cache_salt` per case; no `--attention-backend` (vLLM picks FLASH_ATTN with FlashAttention 3); `--language-model-only`; the prompt as token ids on `/v1/completions` (A.5, A.6) |
| Throughput | The completion tokens of every request of the 32 cases, read and answer turns alike, as the server reported them / the span on the GPU node's clock from the first case's start to the last case's end (after its session close). Admissions and their waits, capacity waits, releases and readmissions, the Goal rewriter's time, the client's rendering and the session closes are included (the runner answers `read_next` itself, A.7); server start-up and shutdown, the job harness's warm-up request, scoring and the Goal rewriter's tokens are not (A.11) |
| Tasks | 32 MemoryAgentBench cases (MemAgent32), 8 per ability (Accurate Retrieval, Test-Time Learning, Long-Range Understanding, Conflict Resolution), one question each, with the chunk counts, history tokens and context SHA-256 values of A.1 (the GPU runs' manifest SHA-256 `010fd109…`); the read_next loop, 120 turns per case at most (A.1, A.7) |
| Evaluation | The official `post_process` of `utils/eval_other_utils.py` at `538026089d…`, offline after the run; Long-Range Understanding: the official GPT-4o judge (`gpt-4o-2024-05-13`), not run: scoring is deferred (A.10, [docs/known-limits.md](known-limits.md) 5.1) |
| Gemma 4 31B IT | A.15: 8 of the 32 cases (the first two of each source) on 8 sessions, with 25 GiB of GPU KV (`--kv-cache-gib 25`), the same context, budget, reply bound, thinking and decoding, CUDA graphs `full`, the three decode fast paths, `--deferred-completion-io`, `TRITON_ATTN`, and for `goal_query` `--admission-overlap 8 --goal-rewriter-batch 4`. Not the 32-case configuration |

---

## Part A. Configuration

### A.1 Scope and cases

| Item | Value |
| --- | --- |
| Model | Qwen3.6-27B on one NVIDIA GH200 120GB with vLLM 0.24.0 (A.2) |
| Benchmark | MemoryAgentBench (A.2): **32 cases, 8 for each of the four abilities**, Accurate Retrieval, Test-Time Learning, Long-Range Understanding and Conflict Resolution; **one question per case**, 32 questions in all |
| Case | One history of 39,409 to 113,458 tokens, cut into 10 to 28 chunks of at most 4,096 tokens, and one question. The 32 cases have 25 distinct histories (the 8 Conflict Resolution questions share one), 564 chunks and 2,255,888 history tokens |
| Concurrency | 32 cases, **at most 16 running**, started in the order below (largest first); when a case ends, the next one starts |
| Servers | One fresh server per configuration, each in its own GPU job: `observation_query`, `goal_query` (this runtime) and `stock` (stock vLLM 0.24.0, the Vanilla control) |
| Run | One server runs all 32 cases once, without interruption (A.12). One run is one sample (A.14) |

Token counts are those of the pinned Qwen3.6 tokenizer.

The abilities (a case id's prefix names its source subset of MemoryAgentBench; the metric is the one the official scorer applies to it, A.10):

| Ability | Source subset | Cases | Histories | Chunks per case | History tokens per case | Metric |
| --- | --- | ---: | ---: | --- | --- | --- |
| Accurate Retrieval | `ruler_qa2_validation_65536` | 8 | 8 | 13–15 | 53,051–58,405 | `substring_exact_match` |
| Test-Time Learning | `icl_trec_coarse_6600shot_balance` | 8 | 8 | 28 | 113,458 | `exact_match` |
| Long-Range Understanding | `infbench_sum_eng_shots2` | 8 | 8 | 17–20 | 67,635–78,885 | `llm_judge_f1` (the official GPT-4o judge) |
| Conflict Resolution | `factconsolidation_sh_32k` | 8 | 1 | 10 | 39,409 | `substring_exact_match` |

**The data.** The cases are MemAgent32, the read_next harness's 32-case selection: rows of the public Hugging Face dataset `cxkjntmngm/MemAgent_Bench` at revision `a785de0f8d2511a43b9e524b5326e1105a4d218a` (not of `ai-hyz/MemoryAgentBench`), cut into chunks by `examples/memoryagentbench/cases.py` (`cases.py build`, B.1). The dataset declares no license, and no data of it is part of this repository. Each case carries its system prompt, its chunks, its question in MemoryAgentBench's query template, its reference answers, its source subset and its metric. A case's context SHA-256 is that of its chunks, concatenated. The GPU runs used the job harness's copy of the cases, a manifest (SHA-256 `010fd10971905e7611ff639c81230f12bad58acd688143cf7b6eb94e30151edf`) and one context per case, which the job harness checked case by case against the list below at job start. A `cases.py build` writes its own `manifest.json`, which nothing pins, and `cases.py verify` compares only the chunk counts and history tokens with MemAgent32's; compare the context SHA-256 values with the list below yourself (B.1).

The 32 cases in run order (id, chunks, history tokens, context SHA-256; B.1 saves the ids as `cases.txt`):

```text
icl_trec_coarse_6600shot_balance-r311-q0  28 113458  433e4d18b8adb1faa842a5b443551cf7c95091b7c3939f2bdeda48aa73e51d01
icl_trec_coarse_6600shot_balance-r324-q0  28 113458  3f0a8bc54e5fb822888afb51502313d4c539eb72077ca817665ff92615557463
icl_trec_coarse_6600shot_balance-r337-q0  28 113458  eae0fbec7f482a695e24cd8c677776af63be4b769c5b2284768118568237c462
icl_trec_coarse_6600shot_balance-r350-q0  28 113458  95bf987db26a3ad9424beff48d2058c7397f8c1b19f91a268a337261db528674
icl_trec_coarse_6600shot_balance-r373-q0  28 113458  c0d3cf0ba25a9219769c3d98cc1356d23c6fb3ba485641e8a00e821d77485d8f
icl_trec_coarse_6600shot_balance-r375-q0  28 113458  105ff35c6fd91a9d727f95113d34b3f1c8978a14fe4739c8c576889017b12191
icl_trec_coarse_6600shot_balance-r393-q0  28 113458  1dbdfd59f8fa35ffac2507e128d4a13990a4b635a62257abc3d0154397e01479
icl_trec_coarse_6600shot_balance-r396-q0  28 113458  1d8923f5291c22a86f55985c985df51f9732abd55d8dfce8013a1f3b9d423c3f
infbench_sum_eng_shots2-r52-q0            20  78885  cb1104516284ee52e6c76e435af3a22c995f8671fb4a2989eff62402364003e6
infbench_sum_eng_shots2-r84-q0            19  77050  c0f8cf56d00012e9b78bce6891f3f9be16408bd64d509672fe5506e9264e6ea6
infbench_sum_eng_shots2-r56-q0            19  75767  8d802a1d4dfbf5f3768f321cff1f1ae0662e451634ed80939b5f7612da5cbe8f
infbench_sum_eng_shots2-r90-q0            18  73699  4c6babc357ffba2d21763fef626c6ad97594b3685a55778f4ff648357dd5de13
infbench_sum_eng_shots2-r29-q0            18  73658  1ae329374a16eea23a72cacf14d7a18f6434652a3c9211808d0cd635e7a97f81
infbench_sum_eng_shots2-r91-q0            18  70981  7b93e166866edaa603f97e7fc45763a0fc2107993453471c101d37224959a481
infbench_sum_eng_shots2-r77-q0            18  70375  d3fa1ef5e2ffc10becdfc8114abdecef784a6f354ea62d987b39065d7bb36b50
infbench_sum_eng_shots2-r58-q0            17  67635  b5ce958a6db779794a0fedfc9e5f2087593af5465654938c65b37df1a5a70eb6
ruler_qa2_validation_65536-r365-q0        15  58405  c399a4cf4ec8846816d30d2cf060c91c8bc5a28d5b7964334bc6c2488fb223fb
ruler_qa2_validation_65536-r350-q0        15  57636  ce6cc2295980d4bb7a89de1e0ea0787ff80ef23e18d551148126fb8458b2b5f7
ruler_qa2_validation_65536-r391-q0        14  56571  b0110fa82a27bb154a0242405ee62c171233d3a4c2231a4046696e6ee7a4920c
ruler_qa2_validation_65536-r370-q0        14  55876  a2386cbb9b123d6a50b71f3bd7a9656c2c4f6e4264d5bf306346c9b87bd4be26
ruler_qa2_validation_65536-r341-q0        14  55215  8d313b9a484c93a318c2be73521313eff6fb419a4a665a36eff2153dde6faf8b
ruler_qa2_validation_65536-r342-q0        14  54852  5d2a586b16062185746bac3cbaa5c34f02b5890aa31d9836258c7bc52e407b06
ruler_qa2_validation_65536-r309-q0        14  53296  7a07b7c28a4a12dec2db350f3f3fdc5cb99dfa7d30d1b3de3ccc355a74ce9e1a
ruler_qa2_validation_65536-r301-q0        13  53051  9c3e2d2798496d3f33c0cb977b6ed66a4ae332ddf4859bacbb74daa0fd8fb563
factconsolidation_sh_32k-r5-q22           10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
factconsolidation_sh_32k-r5-q37           10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
factconsolidation_sh_32k-r5-q4            10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
factconsolidation_sh_32k-r5-q5            10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
factconsolidation_sh_32k-r5-q53           10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
factconsolidation_sh_32k-r5-q60           10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
factconsolidation_sh_32k-r5-q70           10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
factconsolidation_sh_32k-r5-q93           10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
```

The policies of this runtime on MemoryAgentBench (the README's policy table and [docs/design.md section 5](design.md#5-page-selection-for-the-query-policies)):

| `--policy` | History pages used for the next turn |
| --- | --- |
| `observation_query` | Selector V2: mandatory, recent and top-cosine pages within the budget; the query is the hidden state at the end of the newest observation (the new tool message with its chunk, on the last read followed by the question) |
| `goal_query` | The same selector; the query is a maintained retrieval Goal, updated by a Qwen3-0.6B rewriter. On this protocol the Goal's observation is the new tool and user text, and its explicit requirement is the question ([docs/running.md section 6.6](running.md#66-memoryagentbench-kv-session-protocol)) |
| `full` | Every historical page. Not part of the G4 runs: the stock Vanilla control is the exact-attention reference. It ran on this protocol in G1 (job 3397844, [docs/status.md](status.md)); A.4 lists its flags |
| `random_fill` | The control of the two query policies: Selector V2's mandatory and recent pages, the rest of the budget filled in a seeded random order, no query ([docs/design.md section 5](design.md#5-page-selection-for-the-query-policies)). Its servers are on the benchmark page ([benchmarks/memoryagentbench.md](benchmarks/memoryagentbench.md#server-commands)); not part of the G4 runs |

With `--active-token-budget 32768` the capacity is 2,048 pages of 16 tokens. Selection starts once a session's sealed history exceeds 8,192 tokens ([docs/design.md section 5](design.md#5-page-selection-for-the-query-policies)), but up to the budget every page fits, so a session attends to all its pages until its sealed history exceeds 32,768 tokens and to a selected subset after that. Every case exceeds the budget in its later chunks.

### A.2 Model and versions

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, a local directory with regular files; metadata SHA-256 values in [docs/install.md section 2.2](install.md#22-models) |
| Goal rewriter (`goal_query` only) | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca` ([docs/install.md section 2.2](install.md#22-models)) |
| vLLM | The official wheel `vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl`, unmodified, SHA-256 `700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5` |
| Python environment | CPython 3.12, `transformers==5.13.0`, `tokenizers==0.22.2`, built with `examples/env/constraints-3270721.txt` ([docs/install.md section 2.1](install.md#21-base-environment)). The stock control and the client run from the same environment; this runtime is installed into a separate release directory and leaves the environment unchanged ([docs/install.md section 2.4](install.md#24-install-the-release)) |
| Chat template | The runtime's pinned Qwen3.6 template, SHA-256 `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` (`TEMPLATE_SHA256` in `source-overlay/qwen_observation_span.py`; the digest of the template text without one trailing newline). The model directory's template must equal it |
| Runtime | This repository at commit `f9322a6` (`f9322a65477deb7a6660c08a316f2d4c26644c26`), the commit of the G4 runs, or a later commit that leaves the kv-session protocol, the capacity gate and the vLLM arguments of A.4 unchanged. The G1 and G2 validation jobs ran `0d37346` ([docs/history/memoryagentbench-validation.md](history/memoryagentbench-validation.md)). The first Gemma runs (A.15) used `9ef0d6d` (integrate-1021); the later ones are in [docs/status.md](status.md#gemma-4-31b-it). A new `goal_query` run uses `53e3b7a` or later, whose Goal features match quoted literals in linear time with the same output ([docs/history.md](history.md#history)); before it, a reply cut inside a quoted string could stall the engine in the next Goal construction |
| GPU | One NVIDIA GH200 120GB, Linux aarch64; one server per GPU at a time |
| Slurm resources | `--nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16`; `--mem=227120M` (two GH200 shares of the node's host memory, A.4) for this runtime's servers, `--mem=113560M` (one share) for the stock control; one share, `--mem=113560M`, for the Gemma runs of A.15 |
| Client | `examples/memoryagentbench/run_cases.py` with `kv_client.py` (A.7): Python 3.12 with the standard library and Transformers from the environment above, on the GPU node in the same job. The GPU runs used the job harness's runner, which sends the same requests (A.7) |
| Data | The 32 cases of A.1, built with `cases.py build` (B.1); the GPU runs used the job harness's copy (manifest SHA-256 `010fd109…`) |
| Official harness and scorer | `HUST-AI-HYZ/MemoryAgentBench` at commit `538026089d1a8a8eff05121d0db89b388f360eba` (MIT): `utils/eval_other_utils.py`, SHA-256 `d23a5ba9f2750229315cd308463ae99aef9ff36ed0859b69d4851f804976b815` (`post_process`), and `llm_based_eval/summarization_evaluate.py`, SHA-256 `0139db690c4537c8f6ca5fe766ab60caf16050d172412f18ad0535577e708801` (the judge of the summaries); its data are the Hugging Face dataset `ai-hyz/MemoryAgentBench` |

### A.3 The kv-session protocol

The read_next harness's agent loop renders the chat template itself and parses the replies itself; its KV-session client sends token ids, and this runtime serves that client's protocol ([docs/running.md section 6.6](running.md#66-memoryagentbench-kv-session-protocol)). The parts that matter for this configuration:

- **Who renders.** The client renders the conversation (its system prompt, its one tool, the thinking switch) and parses each reply's tool calls; the server never renders a template and needs no tool-call parser. Both servers of a comparison therefore see the client's tokenization: this runtime as one appended token stream per case, stock vLLM as the full re-render of every turn (A.5).
- **One stream per case.** A case is one session whose token stream only grows. Turn 0 sends the whole prompt; every later request sends only the new tail: the previous reply's closing delimiter (with the reply's last token in front when it was not EOS), the new messages and the next generation header. Each reply stays in the stream as the exact ids it was generated with. The server maps a request onto phase 1 (the boundary and the new messages) and phase 2 (the header and the decode) without rendering anything ([docs/design.md section 5](design.md#5-page-selection-for-the-query-policies), [docs/running.md section 6.6](running.md#66-memoryagentbench-kv-session-protocol)).
- **The Goal source.** A request also carries the client's message list, which never makes tokens; `goal_query` takes its observation and its explicit requirement from the messages new since the previous request ([docs/running.md section 6.6](running.md#66-memoryagentbench-kv-session-protocol)).
- **Sessions.** A kv-session server takes no `--task`: its registered tasks are slots, and a client session id binds to the next unused slot at its first request (`--kv-session-slots`, default 64). The client closes a case's session when the case ends (`POST /v1/kv-session/close`).
- **Thinking.** `--kv-session-thinking off` pins the 7-token generation header `<|im_start|>assistant\n<think>\n\n</think>\n\n`, which the template appends with `enable_thinking=False`; the server checks every request's header against it.
- **Generation, enforced by the server.** Greedy only: a request whose sampling fields are not `temperature` 0, `top_p` 1, `top_k` −1 or 0 and `repetition_penalty` 1 is refused. `max_tokens` runs from 1 to the reply bound, clamped to the context left; `finish_reason` is `stop` or `length` (at `max_tokens`); the response carries the text of every generated token, special tokens included, and its ids.
- **Refusals.** Every failure of a complete request is HTTP 400 with `{"error", "type"}`; a close while the session has a request queued or in flight is 409 `SESSION_BUSY`, which the client retries (A.7). A prompt that leaves no room for a reply is refused before any model work with exactly `Logical RoPE context limit reached` (`CONTEXT_LIMIT`). A refusal found before admission leaves the session usable; a failure in the engine fails the session.

### A.4 This runtime: server flags and vLLM arguments

**Flags shared by the policies** (`serve.py`):

| Flag | Value | Effect |
| --- | --- | --- |
| `--protocol` | `kv-session`, with `--kv-session-thinking off`; `--kv-session-slots` not passed (the default 64) | The protocol of A.3 with thinking off. 64 client sessions over the server's lifetime; the 32 cases use 32 (33 in the GPU runs, whose job-harness runner also sent a warm-up request in its own session, A.7) |
| `--max-running-sessions` | `16` | K: at most 16 sessions open. vLLM gets `--max-num-seqs 16` |
| `--kv-cache-gib` | `32` | vLLM gets `--kv-cache-memory-bytes 34359738368` |
| `--page-tokens` / `--kv-layout` | `16` / `decoupled` | 16-token attention pages and a separate Gated-DeltaNet pool of 33 slots (2K + 1): 100 Mamba blocks and 27,873 usable attention blocks at 32 GiB ([docs/running.md section 4.4](running.md#44-several-sessions-on-one-server), [docs/reference.md section 4.6](reference.md#46-decoupled-kv-layout-and-16-token-pages)). vLLM gets `--block-size 16 --mamba-block-size 784 --additional-config ...` |
| `--max-model-len` | `262144` | The model's native context, prompt plus reply ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)). The largest case ends near 128K tokens with the expected replies and near 152K with replies three times as long; 131,072 would refuse its last turns. One turn at the limit needs 17,944 of the 27,873 usable blocks |
| `--max-tokens` | `4096` | Reply bound: the client's `max_tokens` from 1 to 4096 (the client sends 4096) |
| `--dram-arena-gib` | `124` | One pinned host page arena of 124 GiB for all sessions. It holds every sealed page of every open session until the session closes; a capture beyond it fails that request (`DRAM_ARENA_EXHAUSTED`, [docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)). The 16 largest cases at their last turn at once, with replies three times the expected lengths, need 123.35 GiB |
| `--connector-pool-gib` | `1` | The connector's unused CPU pool at 1 GiB instead of vLLM's 32 GiB ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)); on a runtime with the speed defaults ([docs/reference.md section 4.15](reference.md#415-speed-defaults)) 1 GiB is the default, and 32 GiB needs `--no-connector-pool-gib` or `--no-speed-defaults`. vLLM gets `cpu_bytes_to_use` 1073741824 |
| `--cuda-graphs` | `full` | vLLM gets `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}` instead of `--enforce-eager` ([docs/reference.md section 4.8](reference.md#48-cuda-graphs)) |
| `--decode-fastpath` | on | Less host work per decode step, the same values ([docs/reference.md section 4.11](reference.md#411-decode-fast-path-on-by-default)) |
| `--language-model-only` | on | No vision tower and no multimodal warmup ([docs/reference.md section 4.12](reference.md#412-text-only---language-model-only-on-by-default)). vLLM gets `--language-model-only` |
| `--release-idle-pages` | on | Under GPU pressure the capacity gate releases idle sessions' resident pages and readmits them before their next turn ([docs/reference.md section 4.13](reference.md#413-releasing-idle-sessions-pages-opt-in); below). No vLLM argument |
| `--port` | `8000` | The server binds `127.0.0.1` only |

The expected reply lengths are the job harness's planning values, 400 tokens per read turn and 1,000 per answer.

**Idle sessions and `--release-idle-pages`.** Between turns every open session keeps its resident pages on the GPU. With 16 sessions at a 32,768-token budget an idle session pins about 2,300 pages and the 16 together about 37,000, more than the pool holds ([docs/reference.md section 4.13](reference.md#413-releasing-idle-sessions-pages-opt-in)). With the switch the capacity gate covers a turn's shortfall by releasing whole idle sessions' resident pages, least recently used first; their host copies stay in the arena, and a released session's pages come back before its next phase 1. A turn that still does not fit waits (`CAPACITY_WAIT`, [docs/running.md section 4.4](running.md#44-several-sessions-on-one-server)). In this configuration releases are allowed and checked, not required.

**Host memory.** The pinned host memory is the 1-GiB connector pool, 16 Gated-DeltaNet carries (2.29 GiB) and the 124-GiB arena: 127.29 GiB, within the launcher's 128-GiB pinned-host budget ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)); with vLLM's 32-GiB connector pool (`--no-connector-pool-gib`) it would be 158.3 GiB, and the launcher would refuse it. With about 32 GiB for the processes and a 5-GiB margin the job needs 168,237 MiB, more than one GH200's share of the node's host memory (113,560 MiB), so these servers request two shares, `--mem=227120M`.

**Flags per policy:**

| Policy | Selector flags | Admission flags | Goal rewriter |
| --- | --- | --- | --- |
| `observation_query` | `--active-token-budget 32768` | none | not started ([docs/design.md section 5](design.md#5-page-selection-for-the-query-policies)) |
| `goal_query` | `--active-token-budget 32768 --goal-rewriter-model <Qwen3-0.6B directory>` | `--admission-overlap 16` | one Qwen3-0.6B process on the same GPU |
| `full` (not in the G4 runs) | none | none | none |

The budget is one value for both query policies. No server sets `--gdn-device-handoff`, `--phase1-in-step`, `--active-ratio`, `--goal-query-variant`, `--goal-rewriter-batch`, `--fill-seed`, `--decode-fastpath2` or `--decode-fastpath3`. The admission flag follows [docs/reference.md section 4.10](reference.md#410-admission-optimizations-opt-in) (`--admission-overlap 16` for `goal_query` at 16 sessions). Selector, rewriter and admission flags pin the startup configuration but add no vLLM argument, so the servers start vLLM with the same arguments.

These are the flags of the Qwen G4 runs of record, without the optimized switches of the SWE-bench and Terminal-Bench runs (`--decode-fastpath2`, `--decode-fastpath3`, `--deferred-completion-io`, `--goal-rewriter-batch 4`), which the Gemma runs of A.15 use. On a runtime with the speed defaults ([docs/reference.md section 4.15](reference.md#415-speed-defaults)) the decode fast paths and `--deferred-completion-io` are on unless turned off, so the commands add `--no-decode-fastpath2` (the third stage goes with it) and `--no-deferred-completion-io`; `--goal-rewriter-batch` is no speed default and stays off unless passed.

**Commands.** Dry runs of the two servers (add `--start` to serve; `LABEL` is a new label per attempt):

```sh
SHARED=(--protocol kv-session --kv-session-thinking off --max-running-sessions 16 --kv-cache-gib 32
        --page-tokens 16 --kv-layout decoupled --max-model-len 262144 --max-tokens 4096 --dram-arena-gib 124
        --connector-pool-gib 1 --cuda-graphs full --decode-fastpath --no-decode-fastpath2
        --no-deferred-completion-io --language-model-only --release-idle-pages)
python -B serve.py --release-dir "$RELEASE" --policy observation_query "${SHARED[@]}" \
  --active-token-budget 32768 --output "$RUNS/mab-observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --policy goal_query "${SHARED[@]}" \
  --active-token-budget 32768 --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --admission-overlap 16 \
  --output "$RUNS/mab-goal_query-$LABEL" --port 8000
```

**Resulting vLLM arguments** (identical for the servers; the dry run prints the authoritative list, here wrapped by argument group; the dry runs of the G2 servers in job 3397842 printed this list with their own context limit and KV bytes):

```text
--model <model directory> --served-model-name selective-prefix --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
--host 127.0.0.1 --port 8000 --distributed-executor-backend uni --no-async-scheduling
--tensor-parallel-size 1 --pipeline-parallel-size 1
--max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 16384
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

### A.5 Stock vLLM control (Vanilla)

The control is plain `vllm.entrypoints.openai.api_server` from the same environment (A.2), without any code of this runtime, reached through the client's Vanilla transport (`run_cases.py --mode vanilla`, A.7).

| Setting | Value | Relation to this runtime |
| --- | --- | --- |
| `--model` | The same Qwen3.6-27B directory | Same |
| `--served-model-name` | `selective-prefix` | Same name |
| `--host` / `--port` | `127.0.0.1` / `8000` | Same (loopback only) |
| `--distributed-executor-backend` / `--tensor-parallel-size` | `uni` / `1` | Same |
| `--max-num-seqs` | `16` | Same as K = 16 |
| `--max-model-len` | `262144` | Same |
| `--max-num-batched-tokens` | `16384` | Same (fixed by the launcher) |
| `--kv-cache-memory-bytes` | `34359738368` (32 GiB) | Same bytes |
| `--seed` | `42` | Same |
| `--enable-prefix-caching --prefix-caching-hash-algo sha256` | on | Same; the client's `cache_salt` keeps cases apart (A.9) |
| `--language-model-only` | on | Same ([docs/reference.md section 4.12](reference.md#412-text-only---language-model-only-on-by-default)) |
| `--mamba-cache-mode` | `align` | Same |
| `--gdn-prefill-backend` | `triton` | Same |
| CUDA graphs | vLLM's default: neither `--enforce-eager` nor `--compilation-config`, which gives `FULL_AND_PIECEWISE` with `VLLM_COMPILE` ([docs/reference.md section 4.8](reference.md#48-cuda-graphs)) | Same mode |
| Asynchronous scheduling | vLLM's default (enabled); no flag | Differs: the launcher passes `--no-async-scheduling` by default; its opt-in `--async-scheduling` is not used here (A.14) |
| Attention backend | Not set; vLLM selects it (A.6) | Same |
| Route | `POST /v1/completions` with the prompt as token ids | Differs: this runtime serves `/v1/kv-session/complete` (A.3) |
| Thinking | Off, in the client's render (A.7) | Same generation header |

Every other value is set explicitly, so that the control does not depend on vLLM's defaults for this hybrid model; CUDA graphs and asynchronous scheduling stay at vLLM's defaults (above). The arguments that serve the runtime's own route (`--revision`, the reasoning and tool parsers, `--chat-template`, `--kv-transfer-config`, `--block-size`, `--mamba-block-size`, `--additional-config`) are not passed: the client sends token ids and parses the replies itself.

**Command**, in a shell with the environment of A.4:

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
"$VENV/bin/python" -I -B -m vllm.entrypoints.openai.api_server \
  --model "$MODELS/qwen3.6-27b" --served-model-name selective-prefix --host 127.0.0.1 --port 8000 \
  --distributed-executor-backend uni --tensor-parallel-size 1 \
  --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 16384 \
  --kv-cache-memory-bytes 34359738368 --seed 42 \
  --enable-prefix-caching --prefix-caching-hash-algo sha256 \
  --mamba-cache-mode align --gdn-prefill-backend triton --language-model-only
```

- **Environment.** That of A.4 (`PATH`, `LD_LIBRARY_PATH`, short `TMPDIR`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `CC` and `VLLM_RPC_BASE_PATH` unset, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`), plus the two variables that `serve.py` sets for this runtime (`VLLM_WORKER_MULTIPROC_METHOD=spawn`, `VLLM_USE_V2_MODEL_RUNNER=0`) and `VLLM_NO_USAGE_STATS=1`.
- **Generation contract, sent by the client** (the Vanilla transport, A.7). Every turn the whole conversation, rendered with `enable_thinking=False`, goes to `POST /v1/completions` as `prompt` token ids with `max_tokens` 4096, `temperature` 0, `top_p` 1, `top_k` −1, `repetition_penalty` 1, `seed` 42, `skip_special_tokens` false, `return_token_ids` true and a `cache_salt` that is new for each case; there is no stop string. Before sending, the client ends the case itself when the prompt plus 4096 exceeds 262,144 tokens.
- **Readiness.** `GET /health` returns 200 and `GET /v1/models` lists `selective-prefix`.
- **Session close.** Stock vLLM has no session; the client sends no close.
- **Preemption.** Stock vLLM can preempt requests when its KV pool is full; this runtime never preempts (A.14). Record vLLM's counter `vllm:num_preemptions_total` (`GET /metrics`) during and at the end of the run.

### A.6 Kernel backends

Neither server passes `--attention-backend`; vLLM chooses the attention backend automatically on the GH200. Both servers use the same kernels:

| Component | Setting | Server log line |
| --- | --- | --- |
| Attention | FLASH_ATTN with FlashAttention 3 (vLLM's automatic choice) | `Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']` and `Using FlashAttention version 3` |
| Gated-DeltaNet prefill | `--gdn-prefill-backend triton` | `Using Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128)` |
| Sampler | Native top-k/top-p (`VLLM_USE_FLASHINFER_SAMPLER=0`) | `FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0` |
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes 1, 2, 4, 8, 16, 24 and 32 at 16 sequences on both ([docs/reference.md section 4.8](reference.md#48-cuda-graphs)) | `'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE` and `'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32]` in the engine configuration line |
| Asynchronous scheduling | This runtime: off. Stock: on | `Asynchronous scheduling is disabled.` (this runtime) or `Asynchronous scheduling is enabled.` (stock) |
| Text only | `--language-model-only` on both ([docs/reference.md section 4.12](reference.md#412-text-only---language-model-only-on-by-default)) | vLLM's `running in text-only mode`; neither `Encoder cache will be initialized` nor `Multi-modal warmup completed` may appear (a `Using ... for MMEncoderAttention` line may) |

Confirm them from each server's log after READY:

```sh
grep -oE "Using FLASH_ATTN attention backend|Using FlashAttention version [0-9]+|Using Triton/FLA GDN prefill kernel \(requested=[a-z]+|FlashInfer top-p/top-k sampling disabled|Asynchronous scheduling is [a-z]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|'cudagraph_capture_sizes': \[[0-9, ]*\]|running in text-only mode|Encoder cache will be initialized|Multi-modal warmup completed" \
  server.log | sort | uniq -c
```

Every line of the table must appear, and no other attention backend, FlashAttention version, prefill kernel or CUDA-graph mode, and no multimodal startup line. A server whose log differs is not part of the comparison.

### A.7 The runner

The runner is `examples/memoryagentbench/run_cases.py` with the clients of `kv_client.py`: the read_next harness (the design of Shanbin Sun) as this project runs it, with its two transports, the KV-session client (this runtime) and the Vanilla transport (`--mode vanilla`, stock vLLM) ([benchmarks/memoryagentbench.md](benchmarks/memoryagentbench.md), Harness). The loop's instruction texts are constants of `run_cases.py`; this document does not repeat them. The GPU runs ([docs/status.md](status.md)) used the job harness's own runner, whose loop texts and request construction are those of `examples/memoryagentbench/`; the rows Tokenizer and template, kv mode, Warm-up, Timeout and retries, and Records below say where `run_cases.py` differs from it. A baseline uses the read_next harness itself or a client that sends the same requests. The runner runs on the GPU node, in its own process, against `127.0.0.1`.

| Item | Value |
| --- | --- |
| Tokenizer and template | The model directory's tokenizer through Transformers' `apply_chat_template` with `enable_thinking=False` and one tool, `bash`, with one string parameter `command` (mini-swe-agent's tool); tokens by `encode(text, add_special_tokens=False)`. At start the runner checks the generation header (A.3) and records the template's SHA-256 in `run.json` (`tokenizer.template_sha256`); it does not compare it with A.2's pin, which the job harness's runner did, so check the template with B.2 first |
| Workers and order | `--workers 16`; the cases largest first by history tokens, then by id (`--order largest-first`, the default): the order of A.1 |
| Messages | A system message (the case's system prompt followed by the `read_next` instruction, `READ_INSTRUCTION`) and the fixed first user message (`FIRST_USER`); then the replies and the tool and user messages below |
| Chunks | A reply with a valid `bash` call whose command is `read_next` gets the next chunk as a tool message with return code 0 and the chunk's index (`[History chunk k/N]`); several such calls in one reply get several chunks, in order. The last chunk carries the end note (`END_NOTE`), and the case's question follows as a user message in the same request |
| Format errors | A reply without a tool call before the last chunk gets the reminder (`NUDGE`) as a user message; an invalid call (another command, or `read_next` after the last chunk) gets a tool message with return code 1. Two consecutive turns with such an error end the case (`format_error`) |
| Answer | The first reply without a tool call after the last chunk; its content is the prediction (`answered`). An answer that stops at `max_tokens` is `unfinished_thinking` with an empty prediction when the reply has no reasoning part (always with thinking off), else `answer_length_limit` with its text (the read_next harness's rule) |
| Turns | At most 120 requests per case (`turn_limit`). A case of N chunks takes N + 1 requests when every reply reads one chunk |
| Sampling fields | `temperature` 0, `top_p` 1, `top_k` −1, `repetition_penalty` 1, `seed` 42, `max_tokens` 4096 |
| kv mode (this runtime) | `GET /v1/kv-session/info` first: the method must be `dram_backed_kv`, and the server's thinking setting and turn grammar must be the client's (`--thinking`, `--grammar`); the client takes the budget from the server and sends it with every request (it does not check it against 32768; the budget is A.4's `--active-token-budget 32768`, which `GET /v1/kv-session/info` reports, B.4). A new 32-hex session id per case. Turn 0: the whole render (`prefix_tokens` + `header_tokens`, `boundary_tokens` 0). Later turns: the tail after the newest reply, found by rendering the conversation with a marker in place of that reply, with and without the generation prompt: the reply's closing delimiter, the new messages and the header (`header_tokens` 7); `boundary_tokens` counts the tokens before the tail's first `<|im_start|>`, plus 1 when the previous reply's last output id is restored in front (when it is not `<|im_end|>`, after a pad or length stop). The body also carries the normalized message list, the method `dram_backed_kv` and the server's budget. At the case's end `POST /v1/kv-session/close`, retried on 409 every 2 s for up to 120 s |
| vanilla mode (stock) | Every turn the whole conversation re-rendered and sent to `/v1/completions` as token ids (A.5); a new `cache_salt` per case; no close. A prompt whose length plus `max_tokens` exceeds `--max-window` (262144) ends the case before it is sent |
| Warm-up | None. The job harness's runner of the GPU runs sent one request of 2 tokens in its own session before the first case and closed it at once; it was not part of any case or metric |
| Timeout and retries | `--request-timeout`, 3600 s per request by default, the same for every request (the job harness's runner shortened it when the agent time left plus one reply's drain was shorter, to at least 30 s). No retry: a refused or failed request ends its case with the status `error` and the cause as its `runner_status` (`context_limit`, `server_error`, `transport_error`). With `--agent-seconds S` no request starts after S seconds; requests in flight finish, every case closes its session, and the cases cut this way are `error` with `runner_status` `stopped_cap` |
| Records | In `--out`: `run.json` (the settings, the tokenizer's identity, the server's info, the outcome and the count of each case status); `requests.jsonl`, one row per request (case, turn, kind, status, token counts, `finish_reason`, the parsed tool calls, `t_send_ns` and `t_recv_ns` on the node's monotonic clock); per case `cases/<id>/result.json` (`status`, `runner_status`, `prediction`, `chunks_read`, `agent_output_tokens` = the completion tokens of its requests, a per-request summary, and `started` and `finished` in wall-clock seconds); `summary.json`, the input of `score.py`; with `--trace` also `trajectories.jsonl`. The job harness's runner wrote `cases.jsonl` and `runner.json` instead, with `started_ns` and `finished_ns` on the monotonic clock |

### A.8 Limits

| Limit | Value |
| --- | --- |
| Turns per case | 120 requests (`turn_limit`) |
| Format errors | Two consecutive turns end the case (`format_error`) |
| Reply bound | 4096 new tokens per request: the client's `max_tokens` (this runtime: within `--max-tokens 4096`, clamped to the context left; stock: from the client) |
| Context limit | 262,144 tokens, prompt plus reply. This runtime refuses a prompt that leaves no room for a reply (`CONTEXT_LIMIT`, A.3); the Vanilla client ends a case whose prompt plus 4096 exceeds the limit (A.5) |
| Client timeout | 3600 s per request |
| Client retries | None |
| Sessions per server | At most 16 open (K); 64 slots over the server's lifetime, 32 used (33 in the GPU runs, with the warm-up); lifetime request bound max(1,024, 256 × 64) = 16,384 ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)), against at most 32 × 120 = 3,840 case requests |
| Pinned page arena | 124 GiB (this runtime); a capture beyond it fails that request (`DRAM_ARENA_EXHAUSTED`, a failure under A.12) |
| Agent time | The job harness's caps of the GPU runs: at most 3,000 s (`observation_query`), 4,200 s (`goal_query`) and 6,000 s (stock), less what the job needs for start-up and stop. After the cap no request starts, requests in flight may finish one reply, and every open session is closed; cases cut this way end `stopped_cap`, and the run is partial (A.12). `run_cases.py --agent-seconds` applies the same rule (A.7) |

**Why a 3600-s client timeout and no retries.** A request can wait for its admission before its reply of up to 4096 tokens starts: with `--admission-overlap 16` the frontend waits up to 2,100 s for an admission ([docs/reference.md section 4.10](reference.md#410-admission-optimizations-opt-in)), and a `goal_query` admission can wait for the single Goal rewriter process; a turn that does not fit waits for capacity (`CAPACITY_WAIT`, [docs/running.md section 4.4](running.md#44-several-sessions-on-one-server)). The 3600-s timeout keeps such waits from being cut. A retry of a request still in flight would be a second request for a busy session, which the protocol refuses. The stock control uses the same settings, so that the two sides differ only in the server.

### A.9 Concurrency and session lifecycle

- 16 client workers take the 32 cases in the order of A.1, one case each; when a case ends (its session closed), its worker takes the next case. The largest cases go first, so that the 16 workers pack the cases well (list scheduling).
- **Sessions on this runtime.** A case's session id binds to the next unused slot at its turn-0 request ([docs/running.md section 6.6](running.md#66-memoryagentbench-kv-session-protocol)); at most 16 are open. Between turns a session keeps its resident pages on the GPU, its carry and its sealed pages on the host; `--release-idle-pages` can release an idle session's resident pages and readmits them before its next turn (A.4). At the case's end the client closes the session, which frees its GPU and host state ([docs/running.md section 4.4](running.md#44-several-sessions-on-one-server), session pool) before the next case opens.
- **Stock control.** No session and no close; the per-case `cache_salt` lets a case reuse its own prefix blocks and never another case's, and stock vLLM keeps them as an evictable prefix cache.
- **No sharing between cases.** The 8 Conflict Resolution cases share one history, but each reads it in its own session; no server shares KV between cases.

### A.10 Evaluation

| Item | Value |
| --- | --- |
| Evaluator | The official `post_process` of `utils/eval_other_utils.py` (A.2), applied offline after the run, never on the GPU node |
| Input | Per case its final prediction (empty for a case without an answer), its reference answers and its source subset |
| Call | `metrics, parsed = post_process({"output": prediction}, answers, {"sub_dataset": source})`; the case's score is `metrics[metric]` with the metric of A.1 |
| Metrics | Test-Time Learning (ICL): `exact_match` on the parsed answer. Accurate Retrieval (RULER QA) and Conflict Resolution (FactConsolidation): `substring_exact_match`, the larger of the raw and the parsed prediction. Long-Range Understanding (InfBench summaries): `llm_judge_f1` |
| Long-Range Understanding | The official judge `llm_based_eval/summarization_evaluate.py` with `gpt-4o-2024-05-13` (fluency times the F1 of key-point recall and sentence precision); it needs an OpenAI API key. It is not run: scoring is deferred ([docs/known-limits.md](known-limits.md) 5.1), and these 8 cases and the Score stay pending until it runs |
| Dependencies | The scorer file imports numpy, nltk, tiktoken, rouge_score (with absl-py) and editdistance. tiktoken and editdistance are not reached for these four subsets, and rouge_score only adds `rouge*` fields that no metric here uses |
| Cases without an answer | A case that ended `format_error`, `turn_limit`, `unfinished_thinking` or `answer_length_limit` is scored on the prediction it left (an empty one scores 0) |

### A.11 Metrics

**Score.** A case's score is in [0, 1] (A.10). An ability's score is the mean over its 8 cases × 100; the **Score** is the equally weighted mean of the four ability scores, on a 0–100 scale. One case is 12.5 points of its ability and 3.125 points of the Score. For binary metrics at 50 %, the binomial standard error of one ability in one run is about 17.7 points, that of the Score about 8.8 points, and that of a difference of two independent Scores about 12.5 points. Keep the per-case scores so that runs can be compared case by case.

**Throughput** = output tokens / the run's span:

| Part | Definition |
| --- | --- |
| Numerator | The completion tokens of every request of the 32 cases, read and answer turns alike (`agent_output_tokens` of each case), as the server reported them (this runtime: the response's `completion_tokens`; stock: `usage.completion_tokens`) |
| Denominator | The span on the GPU node's clock from the first case's start to the last case's end (after its session close): `max(finished) − min(started)` over the cases' `result.json` (`run_cases.py`, wall-clock seconds); the job harness's runner of the GPU runs recorded `max(finished_ns) − min(started_ns)` over `cases.jsonl` on the monotonic clock |
| Included | Everything between those two moments: admissions and their waits, capacity waits, releases and readmissions, the Goal rewriter's time (`goal_query`), the client's rendering, the session closes |
| Excluded | Server start-up and shutdown, the job harness's warm-up request (GPU runs), scoring, and the Goal rewriter's own tokens (an auxiliary model) |

Record also, per run: per case status, chunks read, turns and wall time; per server the requests, the context refusals, the releases and readmissions (`capacity.jsonl`), the `CAPACITY_WAIT` count (this runtime), and the preemption counter (stock, A.5).

### A.12 Run rules

- **One server per job,** run to the end; the client runs in the same job on the GPU node and shares its 16 CPUs with the server. Use the same node type and job size for every server that is compared.
- **A fresh server and a new output directory for every attempt.** `serve.py` refuses an existing output directory.
- **No resume across interrupted runs.** A run that ends early (the job's time limit, the agent cap, a server failure) is partial: its unfinished cases are not continued in another job, the run is excluded from comparisons and gives no Score and no throughput, and the whole server reruns from the start under a new label.
- **Failures.** A case that ends `error` (`run_cases.py`'s `runner_status`: `server_error`, `transport_error`, `context_limit`, `stopped_cap` or another error) makes the run partial. On this runtime a server also fails its checks when a turn is not token-exact (the server's prompt equals the client's construction, token for token), a request has a `failures.jsonl` row, anything is preempted, a release is not of an idle session or a readmission does not restore its release, or the shutdown is not clean ([docs/design.md section 8](design.md#8-evidence-per-server)).
- **Capacity waits** (`CAPACITY_WAIT`) are normal; a stall that lasts until the agent cap makes the run partial.
- **Transport.** The client reaches the server on the node's loopback; the server never listens beyond loopback ([docs/running.md section 6.4](running.md#64-reaching-the-server-from-another-host)).
- **Placement.** On a two-share job the pinned arena can span two Grace sockets; the job harness records the job's NUMA placement as a finding, not a failure.

### A.13 Alignment checklist

| # | Item | This runtime | A baseline uses |
| --- | --- | --- | --- |
| 1 | Model and revision | Qwen3.6-27B `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | The same |
| 2 | Software | vLLM 0.24.0 official wheel, Transformers 5.13.0, Tokenizers 0.22.2, `constraints-3270721.txt` | The same environment |
| 3 | Hardware and job | 1 × GH200 120GB, 16 CPUs, 227,120 MiB | 1 × GH200 120GB, 16 CPUs, the host memory its own pinned memory needs (stock: 113,560 MiB) |
| 4 | Cases | The 32 cases of A.1, their context SHA-256 values, in that order | The same |
| 5 | Concurrency | At most 16 running, refilled in order | The same |
| 6 | Context limit | 262144 | `--max-model-len 262144` |
| 7 | Batched tokens per step | 16384 | `--max-num-batched-tokens 16384` |
| 8 | Maximum sequences | 16 | `--max-num-seqs 16` |
| 9 | GPU KV | 32 GiB (34,359,738,368 bytes) | `--kv-cache-memory-bytes 34359738368` |
| 10 | Reply bound | 4096 | `max_tokens` 4096 |
| 11 | Decoding | Greedy (server), seed 42 | `temperature` 0, `top_p` 1, `top_k` −1, `repetition_penalty` 1, `seed` 42 |
| 12 | Thinking | Off (`--kv-session-thinking off`, the 7-token header) | The render with `enable_thinking=False` |
| 13 | Stop | EOS/pad only | No stop string |
| 14 | CUDA graphs | `FULL_AND_PIECEWISE` (`--cuda-graphs full`) | vLLM's default (`FULL_AND_PIECEWISE`) |
| 15 | Asynchronous scheduling | Off (the launcher's default; not used here) | vLLM's default (on); a known difference (A.14) |
| 16 | Attention | FLASH_ATTN with FlashAttention 3, chosen by vLLM | The same, no `--attention-backend` |
| 17 | Gated-DeltaNet prefill | Triton | `--gdn-prefill-backend triton` |
| 18 | Sampler | Native (`VLLM_USE_FLASHINFER_SAMPLER=0`) | The same variable |
| 19 | Prefix caching | On, sha256 | `--enable-prefix-caching --prefix-caching-hash-algo sha256`, a `cache_salt` per case |
| 20 | Mamba cache mode | `align` | `--mamba-cache-mode align` |
| 21 | Text only | `--language-model-only` | vLLM's `--language-model-only` |
| 22 | Client | The read_next loop of `run_cases.py` (A.7), rendered with Transformers 5.13.0 and the template `e84f32a2…` | The same loop and template |
| 23 | Loop limits | 120 turns, 2 consecutive format errors, 4096 per reply | The same |
| 24 | Client timeout and retries | 3600 s, no retries | The same |
| 25 | Per-session budget | 32,768 tokens (`observation_query`, `goal_query`) | The same budget for every budget-dependent method; none for exact attention |
| 26 | Evaluation | The official `post_process` at `538026089d…`, the metrics of A.10; the official GPT-4o judge for the summaries | The same |
| 27 | Score | The equally weighted mean of the four ability scores, 0–100 | The same |
| 28 | Throughput | The definition of A.11 | The same numerator and clock |
| 29 | Partial runs | Excluded, never resumed | The same |

### A.14 Known differences

These differences remain when every checklist item matches. Report them with any comparison.

**From the official MemoryAgentBench harness:**

1. **The agent loop.** The official harness feeds the chunks of a history to its memory agent and then asks every question of that history ("inject once, query multiple times"). The read_next harness gives every question its own session, which reads the whole history through `read_next` tool calls and then gets the question. The 8 Conflict Resolution questions share one history, which is read 8 times.
2. **Chunks.** `cases.py` cuts each history into consecutive windows of 4,096 Qwen3.6 tokens (A.1). The official harness cuts histories with its own sentence chunker at `chunk_size` 4096, counted with tiktoken (`configs/data_conf`, `chunk_text_into_sentences` in `utils/eval_other_utils.py`), so its chunk boundaries can differ.
3. **Reply length.** Every turn here may generate 4096 tokens; the official data configurations give each subset its own `generation_max_length` (for example 10 in `Factconsolidation_sh_32k.yaml`, 20 in `ICL_trec_coarse.yaml`, 1,200 in `InfBench_sum.yaml`).
4. **Thinking.** Off here. The read_next harness's own configuration runs with thinking on, as does the current configuration ([docs/status.md](status.md#current-configuration)); results with thinking on are a different configuration and are not directly comparable ([docs/known-limits.md](known-limits.md) 4.3).
5. **Size.** 8 questions per ability and one run. The official configurations ask more questions per subset.
6. **Scoring.** The official `post_process` runs offline on each case's final answer; the summaries need the official GPT-4o judge (A.10).

**Between this runtime and stock vLLM:**

1. **History.** This runtime keeps every reply as the exact ids it was generated with, behind the empty think block of the thinking-off header. The Vanilla control re-renders the whole conversation from text every turn: the template drops the empty think block of every earlier reply (4 tokens each, as [docs/running.md section 6.7](running.md#67-chat-text-protocol-terminal-bench-20-with-terminus-2) describes for the chat-text control) and re-encodes the reply text, so the two prompts first differ at the first earlier reply.
2. **Asynchronous scheduling.** This runtime runs without it (the launcher passes `--no-async-scheduling` by default; its opt-in `--async-scheduling` is not used here); stock vLLM runs with its default, enabled.
3. **Sampling control.** This runtime refuses non-greedy sampling fields and decodes greedily; stock vLLM takes the sampling from the request (`temperature` 0 gives greedy decoding).
4. **Context edge.** This runtime refuses a prompt only when no room for a reply is left and clamps `max_tokens` to the room left (A.3); the Vanilla client ends a case itself when the prompt plus 4096 exceeds 262,144 tokens. At the expected reply lengths no case reaches either edge (A.4).
5. **KV management.** Both have 32 GiB of GPU KV, in different layouts. This runtime uses 16-token attention pages, a separate Gated-DeltaNet pool, a 1-GiB CPU pool and a 124-GiB pinned page arena; it never preempts, a turn that does not fit waits (`CAPACITY_WAIT`), and idle sessions' pages are released and readmitted under pressure (A.4). Stock vLLM uses its own hybrid layout ([docs/reference.md section 4.6](reference.md#46-decoupled-kv-layout-and-16-token-pages)), keeps a finished request's blocks only as an evictable prefix cache, prefills again what it evicted, and can preempt when the pool is full. With 16 long sessions in 32 GiB a session's prefix can be evicted before its next turn.
6. **Admission and scheduling.** This runtime admits one request at a time in arrival order (`goal_query`: up to 16 in flight with `--admission-overlap 16`), runs each admission's phase 1 as its own forward pass, then decodes in batches ([docs/running.md section 4.4](running.md#44-several-sessions-on-one-server), [docs/reference.md section 4.10](reference.md#410-admission-optimizations-opt-in), [docs/design.md section 5](design.md#5-page-selection-for-the-query-policies)). Stock vLLM schedules prefill chunks together with decode steps within the 16,384-token step budget.
7. **Session lifecycle.** This runtime frees a case's state when the client closes its session; stock vLLM keeps cached prefixes until it evicts them.
8. **Selective policies.** `observation_query` and `goal_query` attend to a selected subset of history pages once a session's sealed history exceeds the 32,768-token budget (A.1).
9. **Goal rewriter.** `goal_query` runs an auxiliary Qwen3-0.6B on the same GPU. Its time counts in the throughput, its tokens do not.
10. **Route.** This runtime serves `/v1/kv-session/complete` with the appended tail; stock vLLM serves `/v1/completions` with the whole prompt.
11. **Run-to-run variation.** Batched runs are not token-identical from run to run on either server: a reply can change with the batch composition, which depends on timing ([docs/running.md section 4.4](running.md#44-several-sessions-on-one-server)). The token identity of G1 (job 3397844, [docs/status.md](status.md)) holds for one eager, synchronous session replaying the same prompts, not for the batched runs of this configuration.

### A.15 Gemma 4 31B IT

The Gemma 4 31B IT path of this benchmark: the settings of its first runs (jobs 3429503 and 3429507; the later runs, among them job 3601174 with `--async-scheduling`, `random_fill` and a `goal_query` server on the vLLM Goal engine with `--goal-overlap`, and their settings: [docs/status.md](status.md#gemma-4-31b-it)). It is not the 32-case configuration of A.1 to A.14: it runs 8 cases on 8 sessions. Everything this section does not name is as for Qwen: thinking off, greedy decoding with the sampling fields of A.7, the 4096-token reply bound, the 262,144-token context, the 32,768-token budget, the runner of A.7 (with Gemma's template), the limits of A.8 and the evaluation of A.10.

| Item | Value |
| --- | --- |
| Model | `google/gemma-4-31B-it`, revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, on a release installed with `install.py --model-profile gemma4-31b-it` (`$RELEASE-gemma4`, [docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)) |
| Runtime | `9ef0d6d` (integrate-1021) for its first two runs (jobs 3429503 and 3429507, [docs/status.md](status.md)); a new `goal_query` run uses `53e3b7a` or later (A.2) |
| Cases | 8 of the 32 cases of A.1, the first two of each source in A.1's order (below). A.1's token counts are Qwen's; Gemma's tokenizer gives 0.99–1.01 of them (the job harness's pinned Gemma counts) |
| Concurrency | The 8 cases on 8 sessions at once; one run per server |
| Protocol | kv-session with Gemma's turn grammar and thinking off: the 7-token generation header of the template with thinking off after a user turn, `header_tokens` 0 after tool responses, the stop ids 0, 1, 50 and 106, `"grammar": "gemma4"` in the startup record ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d), [docs/running.md section 6.6](running.md#66-memoryagentbench-kv-session-protocol)) |
| Slurm resources | As A.2, with one share: `--mem=113560M` (host memory below) |

The 8 cases, in A.1's order:

```text
icl_trec_coarse_6600shot_balance-r311-q0
icl_trec_coarse_6600shot_balance-r324-q0
infbench_sum_eng_shots2-r52-q0
infbench_sum_eng_shots2-r84-q0
ruler_qa2_validation_65536-r365-q0
ruler_qa2_validation_65536-r350-q0
factconsolidation_sh_32k-r5-q22
factconsolidation_sh_32k-r5-q37
```

**Flags shared by the policies** (`serve.py`):

| Flag | Value | Effect |
| --- | --- | --- |
| `--protocol` | `kv-session`, with `--kv-session-thinking off` | A.3's protocol with Gemma's grammar (above); this client's Gemma tail renders thinking off, so `--kv-session-thinking on` (gemma-thinking) is not used here. Pass `off` explicitly: on the Gemma profile the launcher's default is now `on` (before gemma-thinking it refused a Gemma kv-session server without `off`), and a server left at `on` refuses every request of this client (`HEADER_MISMATCH`) |
| `--max-running-sessions` | `8` | K: 8 sessions. 16 do not fit 25 GiB at the 32,768-token budget: the selective rule would need 5,699 global Pages, of 4,957 usable. vLLM gets `--max-num-seqs 8` |
| `--kv-cache-gib` | `25` | The KV cache of the group's Gemma baselines ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)). vLLM gets `--kv-cache-memory-bytes 26843545600`. With 8 sessions pool 0 holds 10,158 global Pages against the 5,743 that the selective rule needs (`goal_query`, whose Goal forward has its own transient in pool 1: 8,878 against 5,730) |
| `--page-tokens` / `--kv-layout` | `16` / `decoupled` | The Gemma profile's window layout: pool 0 holds the global layers' 16-token Pages (1.25 MiB each), pool 1 the sessions' sliding windows ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)). vLLM gets `--block-size 16` and the window layout's `--additional-config`, with the selective rule's `request_pages` |
| `--max-model-len` | `262144` | As A.4; the selective rule allows it on Gemma ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)) |
| `--max-tokens` | `4096` | As A.4 |
| `--dram-arena-gib` | `60` | One pinned arena of 60 GiB for the 8 sessions' Gemma Pages; a capture beyond it fails that request (`DRAM_ARENA_EXHAUSTED`) |
| `--connector-pool-gib` | `1` | As A.4 |
| `--cuda-graphs` | `full` | As A.4 (docs/reference.md sections [4.8](reference.md#48-cuda-graphs), [4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)) |
| `--decode-fastpath`, `--decode-fastpath2`, `--decode-fastpath3` | on | The three decode fast paths: less host work per decode step, the same values (docs/reference.md sections [4.11](reference.md#411-decode-fast-path-on-by-default), [4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)). No vLLM argument |
| `--deferred-completion-io` | on | The completion files written on a writer thread after the step, with the same bytes ([docs/reference.md section 4.11](reference.md#411-decode-fast-path-on-by-default)). No vLLM argument |
| `--language-model-only` | on | Required by the Gemma profile ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)). vLLM gets `--language-model-only` |
| `--release-idle-pages` | on | As A.4: idle sessions' global Pages are released under pressure and readmitted before their next phase 1; the sliding windows stay on the GPU ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)) |
| `--port` | `8000` | The server binds `127.0.0.1` only |

**Host memory.** The 60-GiB arena and the 1-GiB connector pool pin 61.01 GiB, within the 80.9 GiB that the Gemma launcher allows for one memory share ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)); Gemma has no Gated-DeltaNet carries. The job therefore requests one share, `--mem=113560M`.

**Flags per policy:**

| Policy | Selector flags | Admission flags | Goal rewriter |
| --- | --- | --- | --- |
| `observation_query` | `--active-token-budget 32768` | none | not started ([docs/design.md section 5](design.md#5-page-selection-for-the-query-policies)) |
| `goal_query` | `--active-token-budget 32768 --goal-rewriter-model <Qwen3-0.6B directory> --goal-rewriter-batch 4` | `--admission-overlap 8` (K = 8) | one Qwen3-0.6B process on the same GPU; the calls that queue run in batches of up to 4 within the Gemma profile's default token cap at 25 GiB, 25,600 tokens (`--goal-rewriter-batch-tokens` not passed; `startup.json` and `ready.json` pin it; docs/reference.md sections [4.10](reference.md#410-admission-optimizations-opt-in), [4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)). Batched greedy decoding is not bit-identical to one call at a time ([docs/reference.md section 4.10](reference.md#410-admission-optimizations-opt-in)) |
| `random_fill` | `--active-token-budget 32768` | none | not started |

No server sets `--gdn-device-handoff` or `--phase1-in-step` (the Gemma profile refuses both, [docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)), `--active-ratio`, `--goal-query-variant`, `--fill-seed`, `--host-memory-shares` or `--goal-rewriter-batch-tokens`.

**Commands.** Dry runs of the two servers (add `--start` to serve; `LABEL` is a new label per attempt):

```sh
SHARED=(--protocol kv-session --kv-session-thinking off --max-running-sessions 8 --kv-cache-gib 25
        --page-tokens 16 --kv-layout decoupled --max-model-len 262144 --max-tokens 4096 --dram-arena-gib 60
        --connector-pool-gib 1 --cuda-graphs full --decode-fastpath --decode-fastpath2 --decode-fastpath3
        --deferred-completion-io --language-model-only --release-idle-pages)
python -B serve.py --release-dir "$RELEASE-gemma4" --policy observation_query "${SHARED[@]}" \
  --active-token-budget 32768 --output "$RUNS/mab-gemma-observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE-gemma4" --policy goal_query "${SHARED[@]}" \
  --active-token-budget 32768 --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --admission-overlap 8 \
  --goal-rewriter-batch 4 --output "$RUNS/mab-gemma-goal_query-$LABEL" --port 8000
```

**vLLM arguments.** The dry run prints the authoritative list. Against A.4's: `--max-num-seqs 8`, `--kv-cache-memory-bytes 26843545600`, `--max-num-batched-tokens 8192` (one native forward of 8,192 tokens, [docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)), `--attention-backend TRITON_ATTN`, the `gemma4` reasoning and tool-call parsers with `--chat-template-content-format openai`, the window layout's `--additional-config`, and no Mamba or Gated-DeltaNet argument (`--mamba-block-size`, `--mamba-cache-mode`, `--gdn-prefill-backend`).

**4,096-token forwards (opt-in, [docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d); GPU runs: [docs/status.md](status.md)).** `--max-num-batched-tokens 4096` on every server gives the group's 4,096-token forward ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d), "4,096-token forwards"): vLLM gets `--max-num-batched-tokens 4096`, the window layout reserves one 4,096-token forward's transient (pool 1 3.125 GiB smaller, 2,560 more global Pages at P = 16), and a phase 1 longer than 4,096 tokens runs in chunks of 3,056 (a read of two chunks is about 8,300 tokens, [docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d), so a read of one chunk may already run in two forwards). Give the stock control `--max-num-batched-tokens 4096` too. The runs of this section used 8,192.

**Kernels.** The runtime's Gemma servers and the stock control both pass `--attention-backend TRITON_ATTN`: FlashAttention 3 has no 512-wide heads, and FA4, vLLM's default for this model, is not validated with this runtime ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)). There is no Gated-DeltaNet prefill. The sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`), CUDA graphs `full` and the text-only lines are those of A.6.

**Stock control (Vanilla), as the job harness plans it.** The job harness's plan for Gemma is plain `vllm.entrypoints.openai.api_server` with `--attention-backend TRITON_ATTN`, `--max-num-batched-tokens 8192`, `--kv-cache-memory-bytes 30064771072` (28 GiB), `--max-num-seqs 16`, `--max-model-len 262144`, `--enable-prefix-caching --prefix-caching-hash-algo sha256` (a `cache_salt` per case), `--seed 42`, `--language-model-only` and `--distributed-executor-backend uni`, served as `selective-prefix`; no Mamba or Gated-DeltaNet flag and no parser (the client sends token ids to `/v1/completions`). The Vanilla transport re-renders the conversation through Gemma's template every turn, with 16 sessions over the 32 cases. It differs from the runtime servers above in KV size (28 against 25 GiB) and in sessions (16 against 8), and because the template drops the empty thought channel of every past model turn when it re-renders, its prompts first differ from the runtime's right after the first `<|turn>model\n` ([docs/reference.md section 4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)).

**Status.** The Gemma 4 31B IT table of [docs/status.md](status.md).

---

## Part B. Procedure

This part runs the configuration of Part A with the tools of this repository: one server and the client in one Slurm job on a GH200 node, the client on the node's loopback. Every step applies to the two policies and to the stock control unless marked. The client, the case builder and the scorer are those of `examples/memoryagentbench/`; the dataset download and the MemoryAgentBench checkout come from outside this repository (B.1).

### B.1 Inputs

- **Model, environment and release.** docs/install.md sections [2.1](install.md#21-base-environment) to [2.4](install.md#24-install-the-release) for the release of the runtime commit (A.2); Qwen3-0.6B for `goal_query` ([docs/install.md section 2.2](install.md#22-models)).
- **Cases.** The 32 cases of A.1, built once with `cases.py build` from your download of `cxkjntmngm/MemAgent_Bench` at `a785de0f8d2511a43b9e524b5326e1105a4d218a` and a MemoryAgentBench checkout, in the helper environment, then checked with `cases.py verify` (`counts_match` 32): [benchmarks/memoryagentbench.md](benchmarks/memoryagentbench.md#procedure), step 1. `$CASES` is that cases directory. `verify` compares the chunk counts and history tokens with A.1's; also compare every case's `context_sha256` in `$CASES/manifest.json` with A.1's list. A case that differs is not part of this configuration.
- **Client.** `examples/memoryagentbench/run_cases.py` (A.7) in the runtime environment, run by the script of B.3 on the GPU node: `--mode kv` for this runtime, `--mode vanilla` for the stock control. It writes the records of A.7 into a new run directory (B.6 and B.7 read them).
- **Scorer.** A checkout of `HUST-AI-HYZ/MemoryAgentBench` at commit `538026089d1a8a8eff05121d0db89b388f360eba` with the two file hashes of A.2, and the helper environment of the benchmark page (its `requirements.lock`, step 1) for `score.py` (B.6). The reference answers come from `$CASES/manifest.json`.

### B.2 Checks before GPU time

**Template and header** (with the environment of A.4; no GPU needed; run on the pinned tokenizer files with Transformers 5.13.0, outside a GPU job):

```sh
"$VENV/bin/python" -I -B - "$MODELS/qwen3.6-27b" <<'PY'
import hashlib, sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
template = tokenizer.chat_template
template = template[:-1] if template.endswith('\n') else template
print(hashlib.sha256(template.encode('utf-8')).hexdigest())
probe = [{'role': 'user', 'content': 'x'}]
without = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=False, enable_thinking=False)
header = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(repr(header[len(without):]), len(tokenizer.encode(header[len(without):], add_special_tokens=False)))
PY
```

The first line must be `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`, the second `'<|im_start|>assistant\n<think>\n\n</think>\n\n' 7`.

**Dry runs.** The dry runs of A.4 (without `--start`) must print the vLLM arguments of A.4; `serve.py` refuses a configuration whose pinned host memory or attention pool does not fit ([docs/reference.md section 4.9](reference.md#49-context-limit-and-reply-bound)).

**Scorer.** `sha256sum utils/eval_other_utils.py llm_based_eval/summarization_evaluate.py` in the checkout must print the two hashes of A.2.

### B.3 Slurm job script

The script below runs one server of the benchmark in one Slurm job: the server, READY, the kernel lines, the client on the node, and the shutdown. It is derived from `examples/slurm/serve_gh200.sh`, which does not pass the kv-session flags, `--cuda-graphs`, `--active-token-budget` or the switches of A.4. It is an example, **not itself part of the GPU runs, and untested** ([docs/known-limits.md](known-limits.md) 5.4). Save it as `mab_job.sh`:

```bash
#!/usr/bin/env bash
# One server of the MemoryAgentBench benchmark (docs/MEMORYAGENTBENCH_BENCHMARK.md) in one Slurm job.
# POLICY: observation_query | goal_query (this runtime) | stock (stock vLLM 0.24.0, the Vanilla control).
set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}" || { echo "run this under sbatch" >&2; exit 2; }
: "${REPO:?}" "${VENV:?}" "${POLICY:?}" "${RUN_ROOT:?}" "${AGENT_SECONDS:?}" "${MODEL:?}" "${CASES:?}"
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

# The server: Part A.4 (this runtime) or Part A.5 (stock).
SHARED=(--protocol kv-session --kv-session-thinking off --max-running-sessions 16 --kv-cache-gib 32
        --page-tokens 16 --kv-layout decoupled --max-model-len 262144 --max-tokens 4096 --dram-arena-gib 124
        --connector-pool-gib 1 --cuda-graphs full --decode-fastpath --no-decode-fastpath2
        --no-deferred-completion-io --language-model-only --release-idle-pages)
case $POLICY in
  observation_query) POLICY_FLAGS=(--active-token-budget 32768); MODE=kv ;;
  goal_query)        POLICY_FLAGS=(--active-token-budget 32768 --goal-rewriter-model "${REWRITER:?}"
                                   --admission-overlap 16); MODE=kv ;;
  stock)             POLICY_FLAGS=(); MODE=vanilla ;;
  *) echo "unknown POLICY: $POLICY" >&2; exit 2 ;;
esac
OUTPUT="$STAGE/$POLICY"
if [[ $POLICY == stock ]]; then
  export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
  "$PY" -I -B -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name selective-prefix --host 127.0.0.1 --port "$PORT" \
    --distributed-executor-backend uni --tensor-parallel-size 1 \
    --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 16384 \
    --kv-cache-memory-bytes 34359738368 --seed 42 \
    --enable-prefix-caching --prefix-caching-hash-algo sha256 \
    --mamba-cache-mode align --gdn-prefill-backend triton --language-model-only > "$STAGE/server.log" 2>&1 &
else
  : "${RELEASE:?}"
  SERVE=("$PY" -B "$REPO/serve.py" --release-dir "$RELEASE" --policy "$POLICY"
         --output "$OUTPUT" --port "$PORT" "${SHARED[@]}" ${POLICY_FLAGS[@]+"${POLICY_FLAGS[@]}"})
  "${SERVE[@]}" > "$STAGE/dry-run.json"          # dry run: checks the release, prints the vLLM command
  "${SERVE[@]}" --start > "$STAGE/server.log" 2>&1 &
fi
SERVER=$!                                         # the APIServer PID (serve.py execs it)

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
grep -oE "Using FLASH_ATTN attention backend|Using FlashAttention version [0-9]+|Using Triton/FLA GDN prefill kernel \(requested=[a-z]+|FlashInfer top-p/top-k sampling disabled|Asynchronous scheduling is [a-z]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|'cudagraph_capture_sizes': \[[0-9, ]*\]|running in text-only mode|Encoder cache will be initialized|Multi-modal warmup completed" \
  "$STAGE/server.log" | sort | uniq -c > "$STAGE/kernels.txt" || true

# The client (Part A.7 to A.9): the 32 cases, 16 at a time, largest first, on this node's loopback.
client_rc=0
timeout "$AGENT_SECONDS" "$PY" -I -B "$REPO/examples/memoryagentbench/run_cases.py" \
  --base-url "http://127.0.0.1:$PORT" --mode "$MODE" --cases "$CASES" --tokenizer "$MODEL" \
  --workers 16 --out "$STAGE/client" ${CLIENT_SECONDS:+--agent-seconds "$CLIENT_SECONDS"} \
  > "$STAGE/client.log" 2>&1 || client_rc=$?
if (( client_rc == 124 )); then echo "agent time limit reached: the run is partial (B.5)" >&2; fi

# Stop: SIGTERM to the server PID only; this runtime must exit 0 with clean receipts (docs/design.md 8).
kill -TERM "$SERVER"
set +e; wait "$SERVER"; server_rc=$?; set -e
echo "server exit $server_rc, client exit $client_rc"
if [[ $POLICY != stock ]]; then
  (( server_rc == 0 )) || exit 1
  "$PY" -B "$REPO/examples/slurm/server_checks.py" cleanup "$OUTPUT"
  "$PY" -I -B -c 'import json,sys; v=json.load(open(sys.argv[1])); sys.exit((v.get("session_pool") or {}).get("open_at_shutdown") != [])' \
    "$OUTPUT/engine-cleanup.json" || { echo "sessions open at shutdown" >&2; exit 1; }
fi
exit "$client_rc"
```

Submit one job per server, with the Quickstart placeholders of the README (`REPO`, `VENV`, `MODELS`, `RELEASE`, `RUNS`), the Slurm settings `ACCOUNT`, `PARTITION` and `MAIL` of [docs/running.md section 4.5](running.md#45-gpu-job-slurm) and settings in the submitting shell (`--export=ALL`). The example below submits `observation_query`:

```sh
POLICY=observation_query
export REPO VENV RELEASE POLICY
export MODEL="$MODELS/qwen3.6-27b" REWRITER="$MODELS/Qwen3-0.6B-c1899de"
export RUN_ROOT=/abs/short/jobs AGENT_SECONDS=1800
export CASES="$RUNS/mab-cases"
sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus=1 \
  --cpus-per-task=16 --mem=227120M --time=00:47:00 --job-name="mab-$POLICY" \
  --output="$RUN_ROOT/slurm-%j.out" --export=ALL --mail-type=BEGIN,END,FAIL --mail-user="$MAIL" \
  /abs/path/to/mab_job.sh
```

- **`goal_query`:** `POLICY=goal_query`, `--time=01:01:00`, for example `AGENT_SECONDS=2400`.
- **Stock control:** `POLICY=stock`, `--mem=113560M` (one share: it pins no arena), `--time=01:27:00`, for example `AGENT_SECONDS=3900`.
- **Time limits.** `--time` covers server start-up (READY 208–263 s after the start for the first server of a job in the MemoryAgentBench jobs with CUDA graphs: 238 s and 263 s for this runtime's G4 servers, 208 s for the stock control, 243 s in G2, 237 s and 242 s in the Gemma jobs; [docs/running.md section 4.1](running.md#41-one-server), [docs/reference.md section 4.8](reference.md#48-cuda-graphs)), the client (`AGENT_SECONDS`) and the shutdown. The job harness's G4 jobs use these limits, about 1.3 times their expected wall times of 2,164 s (`observation_query`), 2,804 s (`goal_query`) and 4,024 s (stock); the `AGENT_SECONDS` values above are examples within them. The expected times rest on the job harness's planning reply lengths (A.4). A run cut by `AGENT_SECONDS` or by the job's time limit is partial (A.12).
- **The client's own cap.** `timeout` is the hard stop. With `CLIENT_SECONDS` exported in the submitting shell (somewhat below `AGENT_SECONDS`, leaving room for the requests in flight), the script passes `run_cases.py --agent-seconds "$CLIENT_SECONDS"`: the client stops starting requests itself, lets the requests in flight finish and closes its sessions (A.7, A.8), so that the shutdown's checks see no open session.
- `RUN_ROOT` must be short: `TMPDIR` lives under it.

### B.4 Readiness check

- **This runtime.** READY is `server_checks.py ready` exit 0 ([docs/running.md section 4.1](running.md#41-one-server)); `/v1/models` answering is not READY. `dry-run.json` holds the vLLM command, which must match A.4. `GET /v1/kv-session/info` returns the method, the budget and the options (policy, page size, budget mode and selection contract, reply bound, context limit, thinking, sessions); the client checks the method, the thinking setting and the turn grammar before its first case (A.7). **Never send a complete request outside the run:** its session id binds one of the server's 64 slots and opens a session that belongs to no case, which the checks then count.
- **Stock control.** `GET /health` returns 200 and `GET /v1/models` lists `selective-prefix`.
- **Kernel lines.** `kernels.txt` in the job directory must show the lines of A.6.

### B.5 Running

- Run the three servers one after another, one job each.
- Progress: one `cases/<id>/result.json` per finished case under `$STAGE/client/`, and one `requests.jsonl` row per request. On this runtime, `capacity.jsonl` in the server's output directory lists the releases and readmissions of idle sessions' pages, and `CAPACITY_WAIT` lines in `server.log` show turns waiting for room; `GET /v1/kv-session/info` may be read at any time.
- **A cut run.** The script has no stall watchdog: a run that stalls lasts until `AGENT_SECONDS` or the job's time limit. A run cut either way is partial; rerun the whole server in a new job (A.12).
- The client's records and the server's output directory hold prompts and history text from the case data; keep them out of public releases.

### B.6 Evaluation

Offline, after the run, with `examples/memoryagentbench/score.py` in the helper environment of B.1 (`MAB_VENV` and the MemoryAgentBench checkout `MAB` of the benchmark page, step 1), with `STAGE=$RUN_ROOT/<job id>`, the job directory of B.3:

```sh
"$MAB_VENV/bin/python" -I -B "$REPO/examples/memoryagentbench/score.py" --run "$STAGE/client" \
  --cases "$CASES" --mab "$MAB" --stub-unused
```

It runs the official `post_process` of the checkout (the file's SHA-256 checked against `upstream/SHA256SUMS`) on every case's prediction with the case's reference answers from `$CASES`, writes each case's metrics and score into its `result.json` and the per-ability means into `accuracy.json`. The 8 Long-Range Understanding cases (`llm_judge_f1`) need the official judge (A.10), which `score.py` does not run: they stay `judge_pending`, and `accuracy.json`'s `composite_score`, the Score of A.11 on a 0–1 scale (× 100 for A.11's 0–100 scale), stays null; `composite_score_without_judge`, on the same 0–1 scale, is the mean over the other three abilities. The judge needs an OpenAI API key and runs with the official `llm_based_eval/summarization_evaluate.py`; without it the Long-Range Understanding score and the Score stay pending.

### B.7 Throughput

The throughput of A.11 from one `run_cases.py` run directory, its cases' `result.json` (`STAGE` as in B.6; untested):

```sh
python3 - "$STAGE/client" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(Path(sys.argv[1]).glob('cases/*/result.json'))]
tokens = sum(row['agent_output_tokens'] for row in rows)
span = max(row['finished'] for row in rows) - min(row['started'] for row in rows)
print(json.dumps(dict(cases=len(rows), output_tokens=tokens, span_s=round(span, 3), tokens_per_s=round(tokens / span, 3))))
PY
```

A complete run has 32 rows and no case with status `error`; otherwise it is partial (A.12).
