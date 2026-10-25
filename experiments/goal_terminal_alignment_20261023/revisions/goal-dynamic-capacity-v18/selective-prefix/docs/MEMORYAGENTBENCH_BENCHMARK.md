# MemoryAgentBench benchmark configuration

English | [简体中文](MEMORYAGENTBENCH_BENCHMARK.zh-CN.md)

This document fixes the configuration of the MemoryAgentBench benchmark run of this runtime: 32 cases of MemoryAgentBench data, 8 for each of its four abilities, each a long history that the agent reads chunk by chunk through a `read_next` tool before it gets one question; Qwen3.6-27B with thinking off; at most 16 cases at a time on one server with the kv-session protocol (`--protocol kv-session`, README §6.6); one server per policy for `observation_query` and `goal_query`; and a stock vLLM 0.24.0 server, reached through the harness's Vanilla transport, as the control. **Part A** is the configuration. A baseline (the stock vLLM control or any other system) is comparable only if it reproduces every item of the checklist in A.13. **Part B** is a manual procedure with the tools of this repository. It is optional, and it needs two inputs that this repository does not contain: the case data (A.1) and a client that sends the requests of A.7.

The Gemma 4 31B IT profile (README §4.14) serves the kv-session protocol in this revision with thinking off, but it is in development and not GPU-validated: its runtime servers do not start on the GPU yet, and a startup fix is in progress; this document covers Qwen3.6-27B only.

Section numbers such as "README §6.6" refer to [README.md](../README.md). "The harness" is the project's MemoryAgentBench harness for agent sessions, which is not public; "the official harness" is the public MemoryAgentBench repository (A.2).

## Validation status

GPU-validated on one GH200: Qwen3.6-27B on the kv-session protocol with thinking off. Both jobs ran on 2026-10-17 at runtime commit `0d37346` and were verified offline on 2026-10-18:

- **Job 3397844 (G1): `PMAB_PASS_VERIFIED`.** One synthetic case of 12 chunks (about 48K tokens of history) and its question, one session per server, the 262,144-token context, 24 GiB of GPU KV, eager execution: this runtime with `full` (the reference) and with `observation_query` at a 32,768-token budget, then stock vLLM 0.24.0, eager and with synchronous scheduling, replaying every prompt of the `full` server. Every turn of both runtime servers was token-exact against the client's construction; the stock replay generated the same tokens as `full` on all 13 turns, and `observation_query` gave the same replies as `full` on all 13. The protocol probes passed: a repeated and an unknown close (200), a turn-0 request for a closed id (400), a request for a busy session (400), a 262,144-token prompt refused with `CONTEXT_LIMIT` before any model work and the session usable afterwards, and `/health` (200). An optional stock server with vLLM's defaults, through the Vanilla transport of A.5, passed too.
- **Job 3397842 (G2): the runtime was correct; the job's verdict `PMAB_FAIL` came from a bug in one harness check, which was corrected and re-verified offline as PASS.** 16 sessions of `observation_query` at a 32,768-token budget with `--cuda-graphs full --decode-fastpath`, a 131,072-token context, `--dram-arena-gib 39` and `--release-idle-pages` (README §4.13), once with a 16-GiB and once with a 24-GiB pool: 32 synthetic cases of 5 to 7 chunks, 16 at a time, every case answered. 139 releases and 124 readmissions at 16 GiB, 13 and 13 at 24 GiB; nothing was preempted; all 32 cases were identical between the two pools, and every readmission restored exactly the Pages that a completed turn had committed. The failed check (route-b) counted releases that came after the engine had finished a turn, in the request's HTTP tail; with the corrected check the server passes. An optional third server without `--release-idle-pages` failed a turn for lack of blocks, as predicted; at the benchmark's runtime such a turn waits instead (README §4.4).

**Run, scores pending:** the 32-case runs of this configuration (G4), all at runtime commit `f9322a6`, ran on 2026-10-18: job 3405844 (`observation_query`), job 3405845 (`goal_query` with the Goal rewriter) and job 3405855 (the stock Vanilla control). Each server served all 32 cases, 16 at a time (614 to 618 requests), without a runtime error. Job 3405844: `PMAB_PARTIAL_PASS_VERIFIED`; every runtime check passed, and one case ended at the agent's own limit of consecutive format errors. Job 3405845: the job's verdict `PMAB_FAIL` came from the harness's route-B check alone, which counted a request as running from its arrival at the server although it was still waiting for admission (such a session is idle, README §4.13); with the corrected check, re-verified offline, the job is `PMAB_PARTIAL_PASS` (one case at the format-error limit). Job 3405855: the job's verdict `PMAB_FAIL` came from the harness's metrics check, which also required the stock server to preempt nothing; stock vLLM preempted 9 requests, which the corrected check records as a finding, and the job is `PMAB_PASS`. Route B released idle sessions' pages 385 times in job 3405844 and 460 times in job 3405845, and every readmission restored exactly the released pages; our runtime preempted nothing. On a Qwen release `f9322a6` gives the same launcher, startup, worker and engine outputs as `0d37346`, byte for byte (README, History: identity probes, 0 differences).

Not yet done: the official scoring of these runs (A.10), including the official judge of the Long-Range Understanding cases. The Slurm script of B.3, the dry-run and scorer checks of B.2 and the scripts of B.6 and B.7 are untested. This document reports no score.

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
| Throughput | The completion tokens of every request of the 32 cases, read and answer turns alike, as the server reported them / the span on the GPU node's monotonic clock from the first case's start to the last case's end (after its session close). Admissions and their waits, capacity waits, releases and readmissions, the Goal rewriter's time, the client's rendering and the session closes are included (the runner answers `read_next` itself, A.7); server start-up and shutdown, the warm-up request, scoring and the Goal rewriter's tokens are not (A.11) |
| Tasks | 32 MemoryAgentBench cases, 8 per ability (Accurate Retrieval, Test-Time Learning, Long-Range Understanding, Conflict Resolution), one question each, manifest SHA-256 `010fd109…`; the harness's MemAgent loop, 120 turns per case at most (A.1, A.7) |
| Evaluation | The official `post_process` of `utils/eval_other_utils.py` at `538026089d…`, offline after the run; Long-Range Understanding: the official GPT-4o judge (`gpt-4o-2024-05-13`), pending (A.10) |
| Gemma 4 31B IT | The same settings are intended; not fixed in this document. The profile serves the kv-session protocol with thinking off, but it is in development and not GPU-validated: its runtime servers do not start on the GPU yet (README §4.14) |

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

**The data.** The cases are the harness's 32-case selection of MemoryAgentBench, already cut into chunks: a manifest (SHA-256 `010fd10971905e7611ff639c81230f12bad58acd688143cf7b6eb94e30151edf`) and one context per case. Each case carries its system prompt, its chunks, its question as the harness asks it, its reference answers, its source subset and its metric. The data are not public and are not part of this repository; the list below identifies them. A case's context SHA-256 is that of its chunks, concatenated; our harness checks every case against it at job start and refuses a mismatch.

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

The policies of this runtime on MemoryAgentBench (README, policy table, and §5):

| `--policy` | History pages used for the next turn |
| --- | --- |
| `observation_query` | Selector V2: mandatory, recent and top-cosine pages within the budget; the query is the hidden state at the end of the newest observation (the new tool message with its chunk, on the last read followed by the question) |
| `goal_query` | The same selector; the query is a maintained retrieval Goal, updated by a Qwen3-0.6B rewriter. On this protocol the Goal's observation is the new tool and user text, and its explicit requirement is the question (README §6.6) |
| `full` | Every historical page. Not part of the G4 runs: the stock Vanilla control is the exact-attention reference. It ran on this protocol in G1 (Validation status); A.4 lists its flags |
| `random_fill` | Not configured for MemoryAgentBench |

With `--active-token-budget 32768` the capacity is 2,048 pages of 16 tokens. Selection starts once a session's sealed history exceeds 8,192 tokens (README §5), but up to the budget every page fits, so a session attends to all its pages until its sealed history exceeds 32,768 tokens and to a selected subset after that. Every case exceeds the budget in its later chunks.

### A.2 Model and versions

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.6-27B`, revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, a local directory with regular files; metadata SHA-256 values in README §2.2 |
| Goal rewriter (`goal_query` only) | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca` (README §2.2) |
| vLLM | The official wheel `vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl`, unmodified, SHA-256 `700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5` |
| Python environment | CPython 3.12, `transformers==5.13.0`, `tokenizers==0.22.2`, built with `examples/env/constraints-3270721.txt` (README §2.1). The stock control and the client run from the same environment; this runtime is installed into a separate release directory and leaves the environment unchanged (README §2.4) |
| Chat template | The runtime's pinned Qwen3.6 template, SHA-256 `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` (`TEMPLATE_SHA256` in `source-overlay/qwen_observation_span.py`; the digest of the template text without one trailing newline). The model directory's template must equal it |
| Runtime | This repository at commit `f9322a6` (`f9322a65477deb7a6660c08a316f2d4c26644c26`), the commit of the G4 runs, or a later commit that leaves the kv-session protocol, the capacity gate and the vLLM arguments of A.4 unchanged. The G1 and G2 validation jobs ran `0d37346` (Validation status) |
| GPU | One NVIDIA GH200 120GB, Linux aarch64; one server per GPU at a time |
| Slurm resources | `--nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16`; `--mem=227120M` (two GH200 shares of the node's host memory, A.4) for this runtime's servers, `--mem=113560M` (one share) for the stock control |
| Client | Our runner (A.7): Python 3.12 with the standard library and Transformers from the environment above, on the GPU node in the same job; not part of this repository |
| Data | The 32 cases of A.1 (manifest SHA-256 `010fd109…`) |
| Official harness and scorer | `HUST-AI-HYZ/MemoryAgentBench` at commit `538026089d1a8a8eff05121d0db89b388f360eba` (MIT): `utils/eval_other_utils.py`, SHA-256 `d23a5ba9f2750229315cd308463ae99aef9ff36ed0859b69d4851f804976b815` (`post_process`), and `llm_based_eval/summarization_evaluate.py`, SHA-256 `0139db690c4537c8f6ca5fe766ab60caf16050d172412f18ad0535577e708801` (the judge of the summaries); its data are the Hugging Face dataset `ai-hyz/MemoryAgentBench` |

### A.3 The kv-session protocol

The harness's agent loop renders the chat template itself and parses the replies itself; its KV-session client sends token ids, and this runtime serves that client's protocol (README §6.6). The parts that matter for this configuration:

- **Who renders.** The client renders the conversation (its system prompt, its one tool, the thinking switch) and parses each reply's tool calls; the server never renders a template and needs no tool-call parser. Both servers of a comparison therefore see the client's tokenization: this runtime as one appended token stream per case, stock vLLM as the full re-render of every turn (A.5).
- **One stream per case.** A case is one session whose token stream only grows. Turn 0 sends the whole prompt; every later request sends only the new tail: the previous reply's closing delimiter (with the reply's last token in front when it was not EOS), the new messages and the next generation header. Each reply stays in the stream as the exact ids it was generated with. The server maps a request onto phase 1 (the boundary and the new messages) and phase 2 (the header and the decode) without rendering anything (README §5, §6.6).
- **The Goal source.** A request also carries the client's message list, which never makes tokens; `goal_query` takes its observation and its explicit requirement from the messages new since the previous request (README §6.6).
- **Sessions.** A kv-session server takes no `--task`: its registered tasks are slots, and a client session id binds to the next unused slot at its first request (`--kv-session-slots`, default 64). The client closes a case's session when the case ends (`POST /v1/kv-session/close`).
- **Thinking.** `--kv-session-thinking off` pins the 7-token generation header `<|im_start|>assistant\n<think>\n\n</think>\n\n`, which the template appends with `enable_thinking=False`; the server checks every request's header against it.
- **Generation, enforced by the server.** Greedy only: a request whose sampling fields are not `temperature` 0, `top_p` 1, `top_k` −1 or 0 and `repetition_penalty` 1 is refused. `max_tokens` runs from 1 to the reply bound, clamped to the context left; `finish_reason` is `stop` or `length` (at `max_tokens`); the response carries the text of every generated token, special tokens included, and its ids.
- **Refusals.** Every failure is HTTP 400 with `{"error", "type"}`. A prompt that leaves no room for a reply is refused before any model work with exactly `Logical RoPE context limit reached` (`CONTEXT_LIMIT`). A refusal found before admission leaves the session usable; a failure in the engine fails the session.

### A.4 This runtime: server flags and vLLM arguments

**Flags shared by the policies** (`serve.py`):

| Flag | Value | Effect |
| --- | --- | --- |
| `--protocol` | `kv-session`, with `--kv-session-thinking off`; `--kv-session-slots` not passed (the default 64) | The protocol of A.3 with thinking off. 64 client sessions over the server's lifetime; the 32 cases and the client's warm-up (A.7) use 33 |
| `--max-running-sessions` | `16` | K: at most 16 sessions open. vLLM gets `--max-num-seqs 16` |
| `--kv-cache-gib` | `32` | vLLM gets `--kv-cache-memory-bytes 34359738368` |
| `--page-tokens` / `--kv-layout` | `16` / `decoupled` | 16-token attention pages and a separate Gated-DeltaNet pool of 33 slots (2K + 1): 100 Mamba blocks and 27,873 usable attention blocks at 32 GiB (README §4.4, §4.6). vLLM gets `--block-size 16 --mamba-block-size 784 --additional-config ...` |
| `--max-model-len` | `262144` | The model's native context, prompt plus reply (README §4.9). The largest case ends near 128K tokens with the expected replies and near 152K with replies three times as long; 131,072 would refuse its last turns. One turn at the limit needs 17,944 of the 27,873 usable blocks |
| `--max-tokens` | `4096` | Reply bound: the client's `max_tokens` from 1 to 4096 (the client sends 4096) |
| `--dram-arena-gib` | `124` | One pinned host page arena of 124 GiB for all sessions. It holds every sealed page of every open session until the session closes; a capture beyond it fails that request (`DRAM_ARENA_EXHAUSTED`, README §4.9). The 16 largest cases at their last turn at once, with replies three times the expected lengths, need 123.35 GiB |
| `--connector-pool-gib` | `1` | The connector's unused CPU pool at 1 GiB instead of 32 (README §4.9). vLLM gets `cpu_bytes_to_use` 1073741824 |
| `--cuda-graphs` | `full` | vLLM gets `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}` instead of `--enforce-eager` (README §4.8) |
| `--decode-fastpath` | on | Less host work per decode step, the same values (README §4.11) |
| `--language-model-only` | on | No vision tower and no multimodal warmup (README §4.12). vLLM gets `--language-model-only` |
| `--release-idle-pages` | on | Under GPU pressure the capacity gate releases idle sessions' resident pages and readmits them before their next turn (README §4.13; below). No vLLM argument |
| `--port` | `8000` | The server binds `127.0.0.1` only |

The expected reply lengths are the harness's planning values, 400 tokens per read turn and 1,000 per answer.

**Idle sessions and `--release-idle-pages`.** Between turns every open session keeps its resident pages on the GPU. With 16 sessions at a 32,768-token budget an idle session pins about 2,300 pages and the 16 together about 37,000, more than the pool holds (README §4.13). With the switch the capacity gate covers a turn's shortfall by releasing whole idle sessions' resident pages, least recently used first; their host copies stay in the arena, and a released session's pages come back before its next phase 1. A turn that still does not fit waits (`CAPACITY_WAIT`, README §4.4). In this configuration releases are allowed and checked, not required.

**Host memory.** The pinned host memory is the 1-GiB connector pool, 16 Gated-DeltaNet carries (2.29 GiB) and the 124-GiB arena: 127.29 GiB, within the launcher's 128-GiB pinned-host budget (README §4.9); with the default 32-GiB connector pool it would be 158.3 GiB, and the launcher would refuse it. With about 32 GiB for the processes and a 5-GiB margin the job needs 168,237 MiB, more than one GH200's share of the node's host memory (113,560 MiB), so these servers request two shares, `--mem=227120M`.

**Flags per policy:**

| Policy | Selector flags | Admission flags | Goal rewriter |
| --- | --- | --- | --- |
| `observation_query` | `--active-token-budget 32768` | none | not started (README §5) |
| `goal_query` | `--active-token-budget 32768 --goal-rewriter-model <Qwen3-0.6B directory>` | `--admission-overlap 16` | one Qwen3-0.6B process on the same GPU |
| `full` (not in the G4 runs) | none | none | none |

The budget is one value for both query policies. No server sets `--gdn-device-handoff`, `--phase1-in-step`, `--active-ratio`, `--goal-query-variant`, `--goal-rewriter-batch`, `--fill-seed`, `--decode-fastpath2` or `--decode-fastpath3`. The admission flag follows README §4.10 (`--admission-overlap 16` for `goal_query` at 16 sessions). Selector, rewriter and admission flags pin the startup configuration but add no vLLM argument, so the servers start vLLM with the same arguments.

**Commands.** Dry runs of the two servers (add `--start` to serve; `LABEL` is a new label per attempt):

```sh
SHARED=(--protocol kv-session --kv-session-thinking off --max-running-sessions 16 --kv-cache-gib 32
        --page-tokens 16 --kv-layout decoupled --max-model-len 262144 --max-tokens 4096 --dram-arena-gib 124
        --connector-pool-gib 1 --cuda-graphs full --decode-fastpath --language-model-only --release-idle-pages)
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

### A.5 Stock vLLM control (Vanilla)

The control is plain `vllm.entrypoints.openai.api_server` from the same environment (A.2), without any code of this runtime, reached through the harness's Vanilla transport (A.7).

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
| `--language-model-only` | on | Same (README §4.12) |
| `--mamba-cache-mode` | `align` | Same |
| `--gdn-prefill-backend` | `triton` | Same |
| CUDA graphs | vLLM's default: neither `--enforce-eager` nor `--compilation-config`, which gives `FULL_AND_PIECEWISE` with `VLLM_COMPILE` (README §4.8) | Same mode |
| Asynchronous scheduling | vLLM's default (enabled); no flag | Differs: the launcher always passes `--no-async-scheduling` (A.14) |
| Attention backend | Not set; vLLM selects it (A.6) | Same |
| Route | `POST /v1/completions` with the prompt as token ids | Differs: this runtime serves `/v1/kv-session/complete` (A.3) |
| Thinking | Off, in the client's render (A.7) | Same generation header |

Every value is set explicitly, so that the control does not depend on vLLM's defaults for this hybrid model. The arguments that serve the runtime's own route (`--revision`, the reasoning and tool parsers, `--chat-template`, `--kv-transfer-config`, `--block-size`, `--mamba-block-size`, `--additional-config`) are not passed: the client sends token ids and parses the replies itself.

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
| CUDA graphs | `FULL_AND_PIECEWISE`, capture sizes 1, 2, 4, 8, 16, 24 and 32 at 16 sequences on both (README §4.8) | `'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE` and `'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32]` in the engine configuration line |
| Asynchronous scheduling | This runtime: off. Stock: on | `Asynchronous scheduling is disabled.` (this runtime) or `Asynchronous scheduling is enabled.` (stock) |
| Text only | `--language-model-only` on both (README §4.12) | vLLM's `running in text-only mode`; neither `Encoder cache will be initialized` nor `Multi-modal warmup completed` may appear (a `Using ... for MMEncoderAttention` line may) |

Confirm them from each server's log after READY:

```sh
grep -oE "Using FLASH_ATTN attention backend|Using FlashAttention version [0-9]+|Using Triton/FLA GDN prefill kernel \(requested=[a-z]+|FlashInfer top-p/top-k sampling disabled|Asynchronous scheduling is [a-z]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|'cudagraph_capture_sizes': \[[0-9, ]*\]|running in text-only mode|Encoder cache will be initialized|Multi-modal warmup completed" \
  server.log | sort | uniq -c
```

Every line of the table must appear, and no other attention backend, FlashAttention version, prefill kernel or CUDA-graph mode, and no multimodal startup line. A server whose log differs is not part of the comparison.

### A.7 The runner

Our runner is this project's own re-implementation of the harness's MemAgent agent loop and of its two transports: the KV-session client (this runtime) and the Vanilla transport (stock vLLM). Nothing of the harness is executed or imported, and the runner is not part of this repository. The loop's instruction texts are the harness's; the runner reproduces them, and this document does not. A baseline uses the harness itself or a client that sends the same requests. The runner runs on the GPU node, in its own process, against `127.0.0.1`.

| Item | Value |
| --- | --- |
| Tokenizer and template | The model directory's tokenizer through Transformers' `apply_chat_template` with `enable_thinking=False` and the harness's one tool, `bash`, with one string parameter `command`; tokens by `encode(text, add_special_tokens=False)`. At start the runner checks the template's SHA-256 (A.2) and the generation header (A.3) |
| Messages | A system message (the case's system prompt followed by the harness's `read_next` instruction) and the harness's fixed first user message; then the replies and the tool and user messages below |
| Chunks | A reply with a valid `bash` call whose command is `read_next` gets the next chunk as a tool message with return code 0 and the chunk's index (`k/N`); several such calls in one reply get several chunks, in order. The last chunk carries the harness's end note, and the case's question follows as a user message in the same request |
| Format errors | A reply without a tool call before the last chunk gets the harness's reminder as a user message; an invalid call (another command, or `read_next` after the last chunk) gets a tool message with return code 1. Two consecutive turns with such an error end the case (`format_error`) |
| Answer | The first reply without a tool call after the last chunk; its content is the prediction (`answered`). An answer that stops at `max_tokens` is `unfinished_thinking` with an empty prediction when the reply has no reasoning part (always with thinking off), else `answer_length_limit` with its text (the harness's rule) |
| Turns | At most 120 requests per case (`turn_limit`). A case of N chunks takes N + 1 requests when every reply reads one chunk |
| Sampling fields | `temperature` 0, `top_p` 1, `top_k` −1, `repetition_penalty` 1, `seed` 42, `max_tokens` 4096 |
| kv mode (this runtime) | `GET /v1/kv-session/info` first: the method, the budget (32768) and the options must be those of A.4. A new 32-hex session id per case. Turn 0: the whole render (`prefix_tokens` + `header_tokens`, `boundary_tokens` 0). Later turns: the tail after the newest reply, found by rendering the conversation with a marker in place of that reply, with and without the generation prompt: the reply's closing delimiter, the new messages and the header (`header_tokens` 7); `boundary_tokens` counts the tokens before the tail's first `<|im_start|>`, plus 1 when the previous reply's last output id is restored in front (when it is not `<|im_end|>`, after a pad or length stop). The body also carries the normalized message list, the method `dram_backed_kv` and the server's budget. At the case's end `POST /v1/kv-session/close`, retried on 409 every 2 s for up to 120 s |
| vanilla mode (stock) | Every turn the whole conversation re-rendered and sent to `/v1/completions` as token ids (A.5); a new `cache_salt` per case; no close |
| Warm-up | Before the first case, one request of 2 tokens in its own session, closed at once; not part of any case or metric |
| Timeout and retries | 3600 s per request (less when the agent time left plus one reply's drain is shorter, at least 30 s); no retry: a refused or failed request ends its case |
| Records | `cases.jsonl`: one row per case (`case`, `split`, `source`, `metric`, `status`, `prediction`, `chunks_read`, `turns`, `agent_output_tokens` = the completion tokens of its requests, `started_ns` and `finished_ns` on the node's monotonic clock); `requests.jsonl`: one row per request (the body without the message list, the response, the send and receive times); `runner.json` |

### A.8 Limits

| Limit | Value |
| --- | --- |
| Turns per case | 120 requests (`turn_limit`) |
| Format errors | Two consecutive turns end the case (`format_error`) |
| Reply bound | 4096 new tokens per request: the client's `max_tokens` (this runtime: within `--max-tokens 4096`, clamped to the context left; stock: from the client) |
| Context limit | 262,144 tokens, prompt plus reply. This runtime refuses a prompt that leaves no room for a reply (`CONTEXT_LIMIT`, A.3); the Vanilla client ends a case whose prompt plus 4096 exceeds the limit (A.5) |
| Client timeout | 3600 s per request |
| Client retries | None |
| Sessions per server | At most 16 open (K); 64 slots over the server's lifetime, 33 used; lifetime request bound max(1,024, 256 × 64) = 16,384 (README §4.9), against at most 32 × 120 = 3,840 case requests |
| Pinned page arena | 124 GiB (this runtime); a capture beyond it fails that request (`DRAM_ARENA_EXHAUSTED`, a failure under A.12) |
| Agent time | The harness's caps: at most 3,000 s (`observation_query`), 4,200 s (`goal_query`) and 6,000 s (stock), less what the job needs for start-up and stop. After the cap no request starts, requests in flight may finish one reply, and every open session is closed; cases cut this way end `stopped_cap`, and the run is partial (A.12) |

**Why a 3600-s client timeout and no retries.** A request can wait for its admission before its reply of up to 4096 tokens starts: with `--admission-overlap 16` the frontend waits up to 2,100 s for an admission (README §4.10), and a `goal_query` admission can wait for the single Goal rewriter process; a turn that does not fit waits for capacity (`CAPACITY_WAIT`, README §4.4). The 3600-s timeout keeps such waits from being cut. A retry of a request still in flight would be a second request for a busy session, which the protocol refuses. The stock control uses the same settings, so that the two sides differ only in the server.

### A.9 Concurrency and session lifecycle

- 16 client workers take the 32 cases in the order of A.1, one case each; when a case ends (its session closed), its worker takes the next case. The largest cases go first, so that the 16 workers pack the cases well (list scheduling).
- **Sessions on this runtime.** A case's session id binds to the next unused slot at its turn-0 request (README §6.6); at most 16 are open. Between turns a session keeps its resident pages on the GPU, its carry and its sealed pages on the host; `--release-idle-pages` can release an idle session's resident pages and readmits them before its next turn (A.4). At the case's end the client closes the session, which frees its GPU and host state (README §4.4, session pool) before the next case opens.
- **Stock control.** No session and no close; the per-case `cache_salt` lets a case reuse its own prefix blocks and never another case's, and stock vLLM keeps them as an evictable prefix cache.
- **No sharing between cases.** The 8 Conflict Resolution cases share one history, but each reads it in its own session; no server shares KV between cases.

### A.10 Evaluation

| Item | Value |
| --- | --- |
| Evaluator | The official `post_process` of `utils/eval_other_utils.py` (A.2), applied offline after the run, never on the GPU node |
| Input | Per case its final prediction (empty for a case without an answer), its reference answers and its source subset |
| Call | `metrics, parsed = post_process({"output": prediction}, answers, {"sub_dataset": source})`; the case's score is `metrics[metric]` with the metric of A.1 |
| Metrics | Test-Time Learning (ICL): `exact_match` on the parsed answer. Accurate Retrieval (RULER QA) and Conflict Resolution (FactConsolidation): `substring_exact_match`, the larger of the raw and the parsed prediction. Long-Range Understanding (InfBench summaries): `llm_judge_f1` |
| Long-Range Understanding | The official judge `llm_based_eval/summarization_evaluate.py` with `gpt-4o-2024-05-13` (fluency times the F1 of key-point recall and sentence precision); it needs an OpenAI API key. Our offline pipeline has not run it: these 8 cases and the Score stay pending until it runs |
| Dependencies | The scorer file imports numpy, nltk, tiktoken, rouge_score (with absl-py) and editdistance. tiktoken and editdistance are not reached for these four subsets, and rouge_score only adds `rouge*` fields that no metric here uses |
| Cases without an answer | A case that ended `format_error`, `turn_limit`, `unfinished_thinking` or `answer_length_limit` is scored on the prediction it left (an empty one scores 0) |

### A.11 Metrics

**Score.** A case's score is in [0, 1] (A.10). An ability's score is the mean over its 8 cases × 100; the **Score** is the equally weighted mean of the four ability scores, on a 0–100 scale. One case is 12.5 points of its ability and 3.125 points of the Score. For binary metrics at 50 %, the binomial standard error of one ability in one run is about 17.7 points, that of the Score about 8.8 points, and that of a difference of two independent Scores about 12.5 points. Keep the per-case scores so that runs can be compared case by case.

**Throughput** = output tokens / the run's span:

| Part | Definition |
| --- | --- |
| Numerator | The completion tokens of every request of the 32 cases, read and answer turns alike (`agent_output_tokens` in `cases.jsonl`), as the server reported them (this runtime: the response's `completion_tokens`; stock: `usage.completion_tokens`) |
| Denominator | The span on the GPU node's monotonic clock from the first case's start to the last case's end (after its session close): `max(finished_ns) − min(started_ns)` over `cases.jsonl` |
| Included | Everything between those two moments: admissions and their waits, capacity waits, releases and readmissions, the Goal rewriter's time (`goal_query`), the client's rendering, the session closes |
| Excluded | Server start-up and shutdown, the warm-up request, scoring, and the Goal rewriter's own tokens (an auxiliary model) |

Record also, per run: per case status, chunks read, turns and wall time; per server the requests, the context refusals, the releases and readmissions (`capacity.jsonl`), the `CAPACITY_WAIT` count (this runtime), and the preemption counter (stock, A.5).

### A.12 Run rules

- **One server per job,** run to the end; the client runs in the same job on the GPU node and shares its 16 CPUs with the server. Use the same node type and job size for every server that is compared.
- **A fresh server and a new output directory for every attempt.** `serve.py` refuses an existing output directory.
- **No resume across interrupted runs.** A run that ends early (the job's time limit, the agent cap, a server failure) is partial: its unfinished cases are not continued in another job, the run is excluded from comparisons and gives no Score and no throughput, and the whole server reruns from the start under a new label.
- **Failures.** A case that ends `server_error`, `transport_error`, `context_limit`, `stopped_cap` or `error` makes the run partial. On this runtime a server also fails its checks when a turn is not token-exact (the server's prompt equals the client's construction, token for token), a request has a `failures.jsonl` row, anything is preempted, a release is not of an idle session or a readmission does not restore its release, or the shutdown is not clean (README §8).
- **Capacity waits** (`CAPACITY_WAIT`) are normal; a stall that lasts until the agent cap makes the run partial.
- **Transport.** The client reaches the server on the node's loopback; the server never listens beyond loopback (README §6.4).
- **Placement.** On a two-share job the pinned arena can span two Grace sockets; the harness records the job's NUMA placement as a finding, not a failure.

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
| 15 | Asynchronous scheduling | Off (always) | vLLM's default (on); a known difference (A.14) |
| 16 | Attention | FLASH_ATTN with FlashAttention 3, chosen by vLLM | The same, no `--attention-backend` |
| 17 | Gated-DeltaNet prefill | Triton | `--gdn-prefill-backend triton` |
| 18 | Sampler | Native (`VLLM_USE_FLASHINFER_SAMPLER=0`) | The same variable |
| 19 | Prefix caching | On, sha256 | `--enable-prefix-caching --prefix-caching-hash-algo sha256`, a `cache_salt` per case |
| 20 | Mamba cache mode | `align` | `--mamba-cache-mode align` |
| 21 | Text only | `--language-model-only` | vLLM's `--language-model-only` |
| 22 | Client | The harness's MemAgent loop (A.7), rendered with Transformers 5.13.0 and the template `e84f32a2…` | The same loop and template |
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

1. **The agent loop.** The official harness feeds the chunks of a history to its memory agent and then asks every question of that history ("inject once, query multiple times"). The harness here gives every question its own session, which reads the whole history through `read_next` tool calls and then gets the question. The 8 Conflict Resolution questions share one history, which is read 8 times.
2. **Chunks.** The chunks come with the case data (at most 4,096 Qwen3.6 tokens each). The official harness cuts histories with its own sentence chunker at `chunk_size` 4096, counted with tiktoken (`configs/data_conf`, `chunk_text_into_sentences` in `utils/eval_other_utils.py`), so its chunk boundaries can differ.
3. **Reply length.** Every turn here may generate 4096 tokens; the official data configurations give each subset its own `generation_max_length` (for example 10 in `Factconsolidation_sh_32k.yaml`, 20 in `ICL_trec_coarse.yaml`, 1,200 in `InfBench_sum.yaml`).
4. **Thinking.** Off here. The harness's own configuration runs with thinking on; results with thinking on are a different configuration and are not directly comparable.
5. **Size.** 8 questions per ability and one run. The official configurations ask more questions per subset.
6. **Scoring.** The official `post_process` runs offline on each case's final answer; the summaries need the official GPT-4o judge (A.10).

**Between this runtime and stock vLLM:**

1. **History.** This runtime keeps every reply as the exact ids it was generated with, behind the empty think block of the thinking-off header. The Vanilla control re-renders the whole conversation from text every turn: the template drops the empty think block of every earlier reply (4 tokens each, as README §6.7 describes for the chat-text control) and re-encodes the reply text, so the two prompts first differ at the first earlier reply.
2. **Asynchronous scheduling.** This runtime runs without it (the launcher always passes `--no-async-scheduling`); stock vLLM runs with its default, enabled.
3. **Sampling control.** This runtime refuses non-greedy sampling fields and decodes greedily; stock vLLM takes the sampling from the request (`temperature` 0 gives greedy decoding).
4. **Context edge.** This runtime refuses a prompt only when no room for a reply is left and clamps `max_tokens` to the room left (A.3); the Vanilla client ends a case itself when the prompt plus 4096 exceeds 262,144 tokens. At the expected reply lengths no case reaches either edge (A.4).
5. **KV management.** Both have 32 GiB of GPU KV, in different layouts. This runtime uses 16-token attention pages, a separate Gated-DeltaNet pool, a 1-GiB CPU pool and a 124-GiB pinned page arena; it never preempts, a turn that does not fit waits (`CAPACITY_WAIT`), and idle sessions' pages are released and readmitted under pressure (A.4). Stock vLLM uses its own hybrid layout (README §4.6), keeps a finished request's blocks only as an evictable prefix cache, prefills again what it evicted, and can preempt when the pool is full. With 16 long sessions in 32 GiB a session's prefix can be evicted before its next turn.
6. **Admission and scheduling.** This runtime admits one request at a time in arrival order (`goal_query`: up to 16 in flight with `--admission-overlap 16`), runs each admission's phase 1 as its own forward pass, then decodes in batches (README §4.4, §4.10, §5). Stock vLLM schedules prefill chunks together with decode steps within the 16,384-token step budget.
7. **Session lifecycle.** This runtime frees a case's state when the client closes its session; stock vLLM keeps cached prefixes until it evicts them.
8. **Selective policies.** `observation_query` and `goal_query` attend to a selected subset of history pages once a session's sealed history exceeds the 32,768-token budget (A.1).
9. **Goal rewriter.** `goal_query` runs an auxiliary Qwen3-0.6B on the same GPU. Its time counts in the throughput, its tokens do not.
10. **Route.** This runtime serves `/v1/kv-session/complete` with the appended tail; stock vLLM serves `/v1/completions` with the whole prompt.
11. **Run-to-run variation.** Batched runs are not token-identical from run to run on either server: a reply can change with the batch composition, which depends on timing (README §4.4). The token identity of G1 (Validation status) holds for one eager, synchronous session replaying the same prompts, not for the batched runs of this configuration.

---

## Part B. Procedure

This part runs the configuration of Part A with the tools of this repository: one server and the client in one Slurm job on a GH200 node, the client on the node's loopback. Every step applies to the two policies and to the stock control unless marked. It needs the case data (A.1) and a client that implements A.7; neither is part of this repository.

### B.1 Inputs

- **Model, environment and release.** README §2.1 to §2.4 for the release of the runtime commit (A.2); Qwen3-0.6B for `goal_query` (README §2.2).
- **Cases.** The 32 cases of A.1, from the harness's maintainers. Save the 32 ids of A.1, in order, as `$CASES/cases.txt`, and check every case before any run: its chunk count and history tokens (pinned tokenizer) as in A.1, every chunk at most 4,096 tokens, and the SHA-256 of its chunks, concatenated, equal to its context SHA-256. A case that differs is not part of this configuration.
- **Client.** An executable `$CLIENT` that implements A.7: `$CLIENT <kv|vanilla> <base url> <case directory> <output directory>`. It runs the cases of `cases.txt` in order on 16 workers against the base URL and writes `<output directory>/cases.jsonl` with the fields of A.7 (B.6 and B.7 read them). Our runner sends these requests and writes these records, but it takes a job specification instead of these arguments and is not part of this repository; the harness's own clients send the same requests.
- **Scorer.** A checkout of `HUST-AI-HYZ/MemoryAgentBench` at commit `538026089d1a8a8eff05121d0db89b388f360eba` with the two file hashes of A.2, and a Python environment with the dependencies of A.10. For B.6, an answers file: a JSON object from case id to the case's list of reference answers, made from the case data.

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

**Dry runs.** The dry runs of A.4 (without `--start`) must print the vLLM arguments of A.4; `serve.py` refuses a configuration whose pinned host memory or attention pool does not fit (README §4.9).

**Scorer.** `sha256sum utils/eval_other_utils.py llm_based_eval/summarization_evaluate.py` in the checkout must print the two hashes of A.2.

### B.3 Slurm job script

The script below runs one server of the benchmark in one Slurm job: the server, READY, the kernel lines, the client on the node, and the shutdown. It is derived from `examples/slurm/serve_gh200.sh`, which does not pass the kv-session flags, `--cuda-graphs`, `--active-token-budget` or the switches of A.4. It is an example, **not itself part of the validated runs, and untested**. Save it as `mab_job.sh`:

```bash
#!/usr/bin/env bash
# One server of the MemoryAgentBench benchmark (docs/MEMORYAGENTBENCH_BENCHMARK.md) in one Slurm job.
# POLICY: observation_query | goal_query (this runtime) | stock (stock vLLM 0.24.0, the Vanilla control).
set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}" || { echo "run this under sbatch" >&2; exit 2; }
: "${REPO:?}" "${VENV:?}" "${POLICY:?}" "${RUN_ROOT:?}" "${AGENT_SECONDS:?}" "${CLIENT:?}" "${CASES:?}"
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

# The server: Part A.4 (this runtime) or Part A.5 (stock).
SHARED=(--protocol kv-session --kv-session-thinking off --max-running-sessions 16 --kv-cache-gib 32
        --page-tokens 16 --kv-layout decoupled --max-model-len 262144 --max-tokens 4096 --dram-arena-gib 124
        --connector-pool-gib 1 --cuda-graphs full --decode-fastpath --language-model-only --release-idle-pages)
case $POLICY in
  observation_query) POLICY_FLAGS=(--active-token-budget 32768); MODE=kv ;;
  goal_query)        POLICY_FLAGS=(--active-token-budget 32768 --goal-rewriter-model "${REWRITER:?}"
                                   --admission-overlap 16); MODE=kv ;;
  stock)             POLICY_FLAGS=(); MODE=vanilla ;;
  *) echo "unknown POLICY: $POLICY" >&2; exit 2 ;;
esac
OUTPUT="$STAGE/$POLICY"
if [[ $POLICY == stock ]]; then
  : "${MODEL:?}"
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
grep -oE "Using FLASH_ATTN attention backend|Using FlashAttention version [0-9]+|Using Triton/FLA GDN prefill kernel \(requested=[a-z]+|FlashInfer top-p/top-k sampling disabled|Asynchronous scheduling is [a-z]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|'cudagraph_capture_sizes': \[[0-9, ]*\]|running in text-only mode|Encoder cache will be initialized|Multi-modal warmup completed" \
  "$STAGE/server.log" | sort | uniq -c > "$STAGE/kernels.txt" || true

# The client (Part A.7 to A.9): the 32 cases, 16 at a time, on this node's loopback.
client_rc=0
timeout "$AGENT_SECONDS" "$CLIENT" "$MODE" "http://127.0.0.1:$PORT" "$CASES" "$STAGE/client" \
  > "$STAGE/client.log" 2>&1 || client_rc=$?
if (( client_rc == 124 )); then echo "agent time limit reached: the run is partial (B.5)" >&2; fi

# Stop: SIGTERM to the server PID only; this runtime must exit 0 with clean receipts (README section 8).
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

Submit one job per server, with the Quickstart placeholders of the README and settings in the submitting shell (`--export=ALL`). The example below submits `observation_query`:

```sh
POLICY=observation_query
export REPO VENV RELEASE POLICY
export MODEL="$MODELS/qwen3.6-27b" REWRITER="$MODELS/Qwen3-0.6B-c1899de"
export RUN_ROOT=/abs/short/jobs AGENT_SECONDS=1800
export CLIENT=/abs/path/to/mab-client CASES=/abs/path/to/cases
sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus=1 \
  --cpus-per-task=16 --mem=227120M --time=00:47:00 --job-name="mab-$POLICY" \
  --output="$RUN_ROOT/slurm-%j.out" --export=ALL --mail-type=BEGIN,END,FAIL --mail-user="$MAIL" \
  /abs/path/to/mab_job.sh
```

- **`goal_query`:** `POLICY=goal_query`, `--time=01:01:00`, for example `AGENT_SECONDS=2400`.
- **Stock control:** `POLICY=stock`, `--mem=113560M` (one share: it pins no arena), `--time=01:27:00`, for example `AGENT_SECONDS=3900`.
- **Time limits.** `--time` covers server start-up (READY 204–231 s after the start for the first server of a job in the validated jobs, README §4.1, plus about a minute of torch.compile on a cold cache, README §4.8), the client (`AGENT_SECONDS`) and the shutdown. The harness's G4 jobs use these limits, about 1.3 times their expected wall times of 2,164 s (`observation_query`), 2,804 s (`goal_query`) and 4,024 s (stock); the `AGENT_SECONDS` values above are examples within them. The expected times rest on the harness's planning reply lengths (A.4). A run cut by `AGENT_SECONDS` or by the job's time limit is partial (A.12).
- **The client's own cap.** `timeout` is the hard stop. The client should stop starting requests itself somewhat earlier, let the requests in flight finish and close its sessions (our runner does, A.8), so that the shutdown's checks see no open session.
- `RUN_ROOT` must be short: `TMPDIR` lives under it.

### B.4 Readiness check

- **This runtime.** READY is `server_checks.py ready` exit 0 (README §4.1); `/v1/models` answering is not READY. `dry-run.json` holds the vLLM command, which must match A.4. `GET /v1/kv-session/info` returns the method, the budget and the options (policy, page size, budget mode and selection contract, reply bound, context limit, thinking, sessions); the client checks them before its first case (A.7). **Never send a complete request outside the run:** its session id binds one of the server's 64 slots and opens a session that belongs to no case, which the checks then count.
- **Stock control.** `GET /health` returns 200 and `GET /v1/models` lists `selective-prefix`.
- **Kernel lines.** `kernels.txt` in the job directory must show the lines of A.6.

### B.5 Running

- Run the three servers one after another, one job each.
- Progress: one line per finished case in `$STAGE/client/cases.jsonl` (with our runner). On this runtime, `capacity.jsonl` in the server's output directory lists the releases and readmissions of idle sessions' pages, and `CAPACITY_WAIT` lines in `server.log` show turns waiting for room; `GET /v1/kv-session/info` may be read at any time.
- **A cut run.** The script has no stall watchdog: a run that stalls lasts until `AGENT_SECONDS` or the job's time limit. A run cut either way is partial; rerun the whole server in a new job (A.12).
- The client's records and the server's output directory hold prompts and history text from the case data; keep them out of public releases.

### B.6 Evaluation

Offline, after the run, with the scorer of B.1 (untested). The answers file maps each case id to its reference answers; an optional judge file maps each Long-Range Understanding case id to its `llm_judge_f1` from the official judge:

```sh
python3 - "$SCORER" "$STAGE/client/cases.jsonl" "$ANSWERS" ${JUDGE:+"$JUDGE"} <<'PY'
import hashlib, json, sys
from pathlib import Path
scorer, cases, answers = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
judge = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else {}
source = scorer / 'utils' / 'eval_other_utils.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == \
    'd23a5ba9f2750229315cd308463ae99aef9ff36ed0859b69d4851f804976b815', 'not the pinned scorer'
sys.path.insert(0, str(source.parent))
import eval_other_utils as official
rows = [json.loads(line) for line in cases.read_text().splitlines() if line.strip()]
complete = ('answered', 'answer_length_limit', 'unfinished_thinking', 'format_error', 'turn_limit')
abilities = {}
for row in rows:
    if row['metric'] == 'llm_judge_f1':
        score = judge.get(row['case'])
    else:
        metrics, _ = official.post_process({'output': row['prediction'] or ''}, answers[row['case']],
                                           {'sub_dataset': row['source']})
        score = float(metrics[row['metric']])
    abilities.setdefault(row['split'], []).append(score)
    print(json.dumps(dict(case=row['case'], status=row['status'], score=score)))
partial = len(rows) != 32 or any(row['status'] not in complete for row in rows)
means = {k: (100 * sum(v) / 8 if len(v) == 8 and None not in v else None) for k, v in sorted(abilities.items())}
score = sum(means.values()) / 4 if len(means) == 4 and None not in means.values() and not partial else None
print(json.dumps(dict(cases=len(rows), partial=partial, abilities=means, score=score)))
PY
```

The last line holds the four ability scores and the Score (A.11); an ability without all 8 scores, or a partial run, gives none. The judge needs an OpenAI API key and runs with the official `llm_based_eval/summarization_evaluate.py`; without it the Long-Range Understanding score and the Score stay pending.

### B.7 Throughput

The throughput of A.11 from one run's `cases.jsonl` (untested):

```sh
python3 - "$STAGE/client/cases.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding='utf-8') if line.strip()]
tokens = sum(row['agent_output_tokens'] for row in rows)
span = (max(row['finished_ns'] for row in rows) - min(row['started_ns'] for row in rows)) / 1e9
print(json.dumps(dict(cases=len(rows), output_tokens=tokens, span_s=round(span, 3), tokens_per_s=round(tokens / span, 3))))
PY
```

A complete run has 32 rows; a run with fewer is partial (A.12).
