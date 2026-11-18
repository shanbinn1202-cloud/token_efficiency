# Status

English | [简体中文](status.zh-CN.md)

The only page that changes with GPU jobs: which commit `main` is and which one is stable, the current configuration, and every GPU job by model, benchmark and policy, with its verdict, whether its numbers are audited and the date it was verified offline. Other pages point here. Limits are in [docs/known-limits.md](known-limits.md), the rules a number must meet in [docs/release-gates.md](release-gates.md). The validation record up to runtime `cf2b28d`, with the review notes for the Goal owner, is frozen in [docs/history/validation-cf2b28d.md](history/validation-cf2b28d.md); the job narratives that the other pages carried until 2026-11-16 are frozen under [docs/history/](history/) as well (the README's [documentation table](../README.md#documentation) lists them). Every DeltaAI job from 33xxxxx on that a page of docs/ or the README names has a row here, the jobs without a verification record included (they say so); the earlier jobs (the clean-install acceptance 3270721 and the runs before it) are described only on the pages that name them, [History](history.md#history) among them. Updated 2026-11-17.

## `main` and the stable commit

- **`main`** is integrate-1117, merged on 2026-11-17 with the previous `main` `7892db3`: runtime `2fb7f62` (integrate-1116 `7a8c36e`'s runtime plus the session pool's task cap 256, `06c1e90`, and `examples/swebench/verified-150.txt`); release check of `2fb7f62`: 2,287 tests, 141 skipped. Parts of it have not run on a GPU ([docs/known-limits.md](known-limits.md) 5.7). The previous `main`, `7892db3` (runtime `4ebff97`), is the tag `main-integrate-1112`.
- **Stable tag:** none yet. No commit has passed all six model × benchmark cells; the closest is runtime `cf2b28d` (three cells itself: jobs 3461703, 3461661, 3462014).
- **Stable configuration** (Qwen on SWE-bench Verified with thinking on): runtime `ba4d1ba`, job 3576565 (the README's command).

## Current configuration

The configuration of the coming 64-task runs (decided 2026-11-16). The configurations of record of the three benchmark pages (Part A, all with thinking off: SWE-bench 32 GiB and 131,072 tokens, MemoryAgentBench 32 GiB and 262,144 tokens, Terminal-Bench 25 GiB and 262,144 tokens) are historical.

| Setting | Value |
| --- | --- |
| Model | Qwen3.6-27B, text only |
| Thinking | on, keep mode (`--enable-thinking`) |
| Scheduling | asynchronous (`--async-scheduling`) |
| KV cache | 25 GiB (`--kv-cache-gib 25`), 16-token pages, the decoupled layout |
| Context, reply | 262,144 tokens, 4,096-token replies |
| Sessions | 16 per server; concurrency 24 is a separate planned sweep (more than 16 on the XML and kv-session protocols has not run on a GPU; the Terminal-Bench dry runs ran 20 chat-text sessions on earlier runtimes: [docs/known-limits.md](known-limits.md) 2.1) |
| Budget | 8,192 tokens (`--active-token-budget 8192`) on every benchmark; MemoryAgentBench is not scored for now |
| Speed switches | the speed defaults (CUDA graphs `full`, the three decode fast paths, `--deferred-completion-io`, `--language-model-only`, a 1-GiB connector pool) |

## Status by model, benchmark and policy

One row per job and policy. **Verdict**: the job harness's verdict, verified offline (`..._VERIFIED`) unless the row says otherwise. **Audited**: `yes (date)` when the row's numbers were re-derived independently from the raw evidence; `verifier only (not audited)` when only the job harness's offline verifier checked them (no row is covered by name by the documentation audit of `7b70e14`, History, so none is marked audited by it). **Date**: the day the job was verified offline. The three reference checks of [docs/release-gates.md](release-gates.md#3-when-a-number-may-be-used) section 3 gate the numbers of jobs on integrate-1116 and later runtimes (the second, full against stock, is deferred); every row below is of an earlier job and was not gated by them, except the three functional checks of integrate-1116 (jobs 3613280, 3613408 and 3613537), which report no benchmark numbers (the thinking-on runs before keep-rebuild-1116 would fail the keep render after a session's first format error, [docs/known-limits.md](known-limits.md) 1.4).

### Qwen3.6-27B, SWE-bench Verified (the first 32 instances, 16 sessions)

| Policy | Job | Commit | Thinking | Verdict | Resolved | Audited | Date |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `observation_query` | 3461703 | `cf2b28d` | off | `SWE32_PASS_VERIFIED` | 20 / 32 | verifier only (not audited) | 2026-10-30 |
| `goal_query` | 3461703 | `cf2b28d` | off | `SWE32_PASS_VERIFIED` | 19 / 32 | verifier only (not audited) | 2026-10-30 |
| `full` (`--release-idle-pages`) | 3583081 | `ba4d1ba` | on | `SWE32_PASS_VERIFIED` | 21 / 32 | yes (2026-11-16) | 2026-11-12 |
| `observation_query` | 3539260 | `60c5d21` | on | `SWE32_PASS_VERIFIED` | 21 / 32 | yes (2026-11-16) | 2026-11-10 |
| `recent_fill` | 3539242 | `60c5d21` | on | `SWE32_PASS_VERIFIED` | 22 / 32 | yes (2026-11-16) | 2026-11-08 |
| `goal_query` (the stable configuration) | 3576565 | `ba4d1ba` | on | `SWE32_PASS_VERIFIED` | 19 / 32 | yes (2026-11-16) | 2026-11-11 |
| `akv_rule` | 3577833 | `ba4d1ba` | on | `SWE32_PARTIAL_VERIFIED` (one instance cut at the time limit) | 20 / 32 | yes (2026-11-16) | 2026-11-12 |
| `goal_query_markers` | 3602728 | `325cdc5` (integrate-1114) | on | `SWE32_PASS_VERIFIED` | 20 / 32 | yes (2026-11-16) | 2026-11-14 |
| `goal_query` with `--token-gather markers` | 3603443 | `2b2a752` (integrate-1114) | on | `SWE32_PASS_VERIFIED` | 21 / 32 | yes (2026-11-16) | 2026-11-14 |
| `ledger_query` | 3611601 | `51b4085` (ledger-int-1112) | on | `SWE32_PASS_VERIFIED` | 21 / 32 | verifier only (not audited) | 2026-11-16 |
| `recent_fill`, `--async-scheduling` (rfillasync32ti) | 3611917 | `4ebff97` | on | `SWE32_PASS_VERIFIED` | 22 / 32 | verifier only (not audited) | 2026-11-16 |
| `recent_fill`, synchronous twin (rfillsync32ti) | 3612333 | `4ebff97` | on | `SWE32_PASS_VERIFIED` | 20 / 32 | verifier only (not audited) | 2026-11-16 |
| `goal_query` with `--goal-rewriter-engine vllm` and `--goal-overlap` (vllm25, 25 GiB) | 3500565 | `c07d42a` | off | `SWE32_FAIL_VERIFIED` (`CUDA_GRAPH_CONFIG_DIFFERS`, `GOAL_FALLBACK`); the failed checks `capture-sizes` and `context-limits` read the vLLM Goal child's log lines as the main engine's: re-evaluated with fixed checks only `GOAL_FALLBACK` remains (harness snapshot perf-swe32, information only) | 20 / 32 | verifier only (not audited) | 2026-11-02 |
| stock vLLM 0.24.0 (`reasoning_dropped`) | 3583822 | — | on | `SWE32_PASS_VERIFIED` | 21 / 32 | yes (2026-11-16) | 2026-11-12 |
| `full` (without `--release-idle-pages`) | 3579273 | `ba4d1ba` | on | `SWE32_FAIL_VERIFIED` (the server stalled at the admission gate for 55.6 min, [docs/known-limits.md](known-limits.md) 2.4) | 21 / 32 | verifier only (not audited) | 2026-11-11 |
| stock vLLM 0.24.0 (25 GiB, 262,144 tokens) | 3507294 | — | off | `SWE32_PASS_VERIFIED` | 21 / 32 | verifier only (not audited) | 2026-11-03 |
| stock vLLM 0.24.0 with `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0` (25 GiB, 262,144 tokens) | 3526362 | — | off | `SWE32_PASS_VERIFIED` | 21 / 32 | verifier only (not audited) | 2026-11-06 |
| stock vLLM 0.24.0 (25 GiB, 262,144 tokens) | 3506513 | — | off | `SWE32_PARTIAL_VERIFIED` | 14 / 32 | verifier only (not audited) | 2026-11-02 |
| stock vLLM 0.24.0 with vLLM's CPU KV offload (16 instances) | 3525442 | — | off | `SWE32_PARTIAL_VERIFIED` | 8 / 16 | verifier only (not audited) | 2026-11-04 |
| `goal_query` (Part A: 32 GiB, 131,072 tokens) | 3404696 | `5b15b2f` | off | `SWE32_PASS_VERIFIED` | 19 / 32 | verifier only (not audited) | 2026-10-19 |
| `goal_query` with `--goal-rewriter-batch 4` (Part A) | 3404230 | `5b15b2f` | off | `SWE32_PASS_VERIFIED` | 20 / 32 | verifier only (not audited) | 2026-10-19 |
| `goal_query` with `--goal-rewriter-batch 4` (Part A, with the decode fast paths and `--deferred-completion-io`) | 3428631 | `120bc3c` | off | `SWE32_FAIL_VERIFIED` (one harness check, goal-async, did not yet accept the Cancel acknowledgement that `--deferred-completion-io` holds until the request's files are written; that request was served correctly) | 19 / 32 | verifier only (not audited) | 2026-10-21 |
| `observation_query` (Part A) | 3429500 | `120bc3c` | off | `SWE32_PASS_VERIFIED` | 21 / 32 | verifier only (not audited) | 2026-10-22 |
| stock vLLM 0.24.0 (Part A) | 3429193 | — | off | `SWE32_PASS_VERIFIED` | 20 / 32 | verifier only (not audited) | 2026-10-22 |
| `full` (32 GiB, 131,072 tokens, 4,096-token replies, step limit 120, an 88-GiB arena) | 3357655 | `e877051` | off | `SWE16_PASS_VERIFIED` | 22 / 32 | verifier only (not audited) | 2026-10-11 |
| `random_fill` (the same settings) | 3360287 | `e877051` | off | `SWE16_PASS_VERIFIED` | 19 / 32 | verifier only (not audited) | 2026-10-12 |
| `goal_query` (the same job harness and commit) | 3363871 | `e877051` | off | `SWE16_PASS_VERIFIED` | 19 / 32 | verifier only (not audited) | 2026-10-12 |
| `random_fill`, `recent_tools` | — | | on | not run (`recent_tools`: job 3540110 was cancelled before it started; no verification record) | | | |

The thinking-on runtime runs before keep-rebuild-1116 lost each session's earlier reasoning after its first format error ([docs/known-limits.md](known-limits.md) 1.4). Since these runs integrate-1116 changed two policies' selections, not yet run on a GPU: `ledger_query` v2 (job 3611601 ran v1; [docs/policy-registry.md](policy-registry.md#8-the-query-hook-and-ledger_query-ledger-policy-1112) section 8) and the Goal of `goal_query` and `goal_query_markers` from a session's first format-error retry on (known-limits 1.3). `recent_tools`, which has not run with thinking on, also orders a thinking-on reply cut before `</think>` differently since ledger-fixes-1116 (its Pages are REASONING, not CALL; [docs/policy-registry.md](policy-registry.md#8-the-query-hook-and-ledger_query-ledger-policy-1112) section 8). The async pair (same commit and payload; the async flag is the only difference): 267.9 against 225.7 output tokens/s (+18.7 %), turns per minute +21 %, the steady decode step per batch size 0.892 of the synchronous one (about 11 % faster); the trajectories diverged (541K against 630K generated tokens); not audited. The stable configuration's run, job 3576565: 46.3 min from the first instance's start to the last one's end, 238.6 output tokens/s over that span (from its verification record's tables; not audited). The Qwen smoke of `4ebff97` (smoket, job 3586847) was cancelled on 2026-11-16: not run on `4ebff97` (deferred). The small `akv_rule` run with thinking on (job 3549065: 8 of the 32 tasks on 4 sessions) was cancelled before it started (no verification record). The stock rows run no code of this runtime (no commit); the Part A rows are the configuration of record of [docs/SWEBENCH_BENCHMARK.md](SWEBENCH_BENCHMARK.md), their narrative in [docs/history/swebench-validation.md](history/swebench-validation.md).

### Qwen3.6-27B, Terminal-Bench 2.0 (16-task dry runs)

| Policy | Job | Commit | Thinking | Verdict | Solved | Audited | Date |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `full` (32 GiB) | 3398129 | `0d37346` | off | archived verification `NOT_VERIFIED` (failed `pre-claim-gates`); the engine died: a decode took the last free block and vLLM preempted it (the capacity gate of `68d19b4` fixed it) | 9 of 16 (not verified) | not audited | 2026-10-17 |
| `full` (K = 20, 32 GiB, 65,536 tokens, `--release-idle-pages`) | 3399075 | `68d19b4` | off | `TB16_FAIL` (two harness checks, later corrected); its archived offline verification is `NOT_VERIFIED` | 9 of 16 (not verified) | not audited | 2026-10-18 |
| `observation_query` (K = 20, 32 GiB, 65,536 tokens) | 3399437 | `68d19b4` | off | `TB16_PASS_VERIFIED` | 8 of 16 | verifier only (not audited) | 2026-10-18 |
| `goal_query` (K = 20, 32 GiB, 65,536 tokens) | 3429670 | `120bc3c` | off | `TB16_FAIL_VERIFIED` (the server stalled in the Goal features; fixed in `53e3b7a`) | 8 of 16 | verifier only (not audited) | 2026-10-22 |
| `goal_query` (K = 20, 32 GiB, 65,536 tokens) | 3434533 | `53e3b7a` | off | `TB16_PASS_VERIFIED` | 8 of 16 | verifier only (not audited) | 2026-10-24 |
| `observation_query`, `goal_query` (K = 20, 262,144 tokens) | 3461661 | `cf2b28d` | off | `TB16_PASS_VERIFIED` | 9 and 8 of 16 | verifier only (not audited) | 2026-10-26 |
| `observation_query`, `goal_query`, stock (`--thinking-history strip`) | 3499759 | `8ca8511` | on | `TB16_PASS_VERIFIED` (finding `GOAL_FALLBACK` on `stripgoal262`) | 8, 6 and 8 of 16 | verifier only (not audited) | 2026-11-02 |

The smoke of `4ebff97` with thinking on and `--async-scheduling` (smokeqt, job 3586720) was cancelled on 2026-11-16: not run on `4ebff97` (deferred). Throughput of the 65,536-token dry runs, from their verification records (not audited): 169.1, 181.03 and 177.63 tokens/s (jobs 3399075, 3399437, 3434533).

### Qwen3.6-27B, MemoryAgentBench (kv-session; not scored)

| Policy | Job | Commit | Thinking | Verdict | Audited | Date |
| --- | --- | --- | --- | --- | --- | --- |
| `observation_query` (G4, 32 cases) | 3405844 | `f9322a6` | off | `PMAB_PARTIAL_PASS_VERIFIED` | verifier only (not audited) | 2026-10-19 |
| `goal_query` (G4) | 3405845 | `f9322a6` | off | `PMAB_FAIL_VERIFIED`; a partial pass with the corrected route-B check | verifier only (not audited) | 2026-10-19 |
| `observation_query`, `random_fill`, `goal_query` (`--async-scheduling`) | 3568311 | `ba4d1ba` | on | `PMAB_FAIL_VERIFIED`: crashed: the async resolve failed at a reply cut at `max_tokens` while another request decoded (fixed in `4ebff97`) | verifier only (not audited) | 2026-11-12 |
| `observation_query`, `random_fill`, `goal_query` (`--async-scheduling`) | 3584105 | `4ebff97` | on | `PMAB_PASS_VERIFIED` | verifier only (not audited) | 2026-11-14 |
| `observation_query` (`--async-scheduling`, the async-cut case on 16 of 16 probe turns) | 3604146 | `4ebff97` | on | `PMAB_PASS_VERIFIED` | verifier only (not audited) | 2026-11-15 |

Earlier runs, from [the validation record](history/validation-cf2b28d.md): G1 job 3397844 `PMAB_PASS`, G2 job 3397842 `PMAB_FAIL` from one harness check, a pass when re-judged (both archived verifications, made with a later harness, say `NOT_VERIFIED`); G4 job 3405855 `PMAB_FAIL_VERIFIED`, re-judged offline as `PMAB_PASS`.

### Gemma 4 31B IT

| Benchmark | Policy | Job | Commit | Verdict | Result | Audited | Date |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SWE-bench Verified | `observation_query` (4,096-token forwards, `--max-num-batched-tokens 4096`; server obs4k) | 3462014 | `cf2b28d` | `GSWE32_PASS_VERIFIED` | 18 / 32 | verifier only (not audited) | 2026-10-30 |
| SWE-bench Verified | `goal_query` | 3429808 | `9ef0d6d` | `GSWE32_PASS_VERIFIED` | 15 / 32 | verifier only (not audited) | 2026-10-24 |
| SWE-bench Verified | `observation_query` | 3429830 | `9ef0d6d` | `GSWE32_PARTIAL_VERIFIED` (re-verified with the corrected check of a request cancelled during its admission; the job's own verdict was a FAIL; partial because the pool's 2,760-s cap cut 7 of the 32 instances) | 17 / 32 (25 done) | verifier only (not audited) | 2026-10-24 |
| Terminal-Bench 2.0, 65,536 tokens | `observation_query` | 3433706 | `9ef0d6d` | `TB16_PASS_VERIFIED` (re-verified) | 4 of 16 | verifier only (not audited) | 2026-10-24 |
| Terminal-Bench 2.0, 65,536 tokens | `goal_query` | 3434544 | `53e3b7a` | `TB16_PASS_VERIFIED` (re-verified) | 5 of 16 | verifier only (not audited) | 2026-10-24 |
| Terminal-Bench 2.0, 262,144 tokens | `observation_query`, `goal_query`, `random_fill`, stock | 3560498 | `f5a5c36` | `TB16_FAIL_VERIFIED` (`goal_query` `CAPACITY_EXCEEDED`) | 4, 4, 3, 3 of 16 | verifier only (not audited) | 2026-11-15 |
| MemoryAgentBench (8 cases, 8 sessions) | `observation_query` | 3429503 | `9ef0d6d` | `PMAB_PASS_VERIFIED` | not scored | verifier only (not audited) | 2026-10-22 |
| MemoryAgentBench | `goal_query` | 3429507 | `9ef0d6d` | `PMAB_PASS_VERIFIED` | not scored | verifier only (not audited) | 2026-10-22 |
| MemoryAgentBench (`--async-scheduling`, thinking off) | `goal_query`, `observation_query`, `random_fill` | 3601174 | `4ebff97` | `PMAB_PASS_VERIFIED` | not scored | verifier only (not audited) | 2026-11-15 |

The Gemma SWE-bench smoke of `4ebff97` (smokea, job 3586987) was cancelled on 2026-11-16: not run on `4ebff97` (deferred). In job 3560498 the `goal_query` server's Goal also fell back on 2 of its 219 turns (finding `GOAL_FALLBACK`), and the join flags one `random_fill` trial cut at its agent timeout: the server answered the request in flight, which the client never received (one server row more than its client calls); verifier only (not audited).

### Not run yet

What no GPU job has run, per benchmark, at any commit (as of 2026-11-16; besides the smokes of `4ebff97` cancelled above):

- **SWE-bench Verified.** Qwen with thinking on: `random_fill` and `recent_tools` (the row above). Gemma 4: `full`, `random_fill` and the stock control (only `observation_query` and `goal_query` ran).
- **Terminal-Bench 2.0.** The benchmark as a whole (all 89 tasks; only the 16-task dry runs ran). Qwen: `full` at 262,144 tokens, `random_fill`, and the stock control with thinking off at 262,144 tokens (there it ran with thinking on only, job 3499759). Gemma 4: GG2 (`--thinking-history strip` on Terminal-Bench) has not run; its job harness plan is not built.
- **MemoryAgentBench.** Gemma 4: the 32-case configuration (its runs used 8 cases on 8 sessions) and the stock (Vanilla) control. No run is scored ([docs/known-limits.md](known-limits.md) 5.1).

### Functional GPU checks (not benchmark runs)

| Check | Job | Commit | Verdict | Audited | Date |
| --- | --- | --- | --- | --- | --- |
| 24 sessions with idle release (sess24ri r2): `goal_query` (the vLLM Goal engine, `--goal-overlap`), 24 sessions, `--release-idle-pages`, `--admission-overlap 24`, thinking on, the scripted edge-case client | 3616345 | `7a8c36e` (integrate-1116) | `SWE32_FAIL_VERIFIED` (Slurm `FAILED`), from a defect of the job harness's idle-release check, not of the runtime: the pool completed (24 of 24 tasks, no stall; 181 evictions, 113 readmissions, equal to the engine's totals), and the one failed check counted a session's second eviction as having no rebuild in between because it ordered requests by arrival, not admission. Re-checked offline with the corrected rule (job harness payload `269a00e6`): every server check passes (53 of 53 second evictions follow a rebuild), `JOIN_PASS`, the Goal text `VERIFIED`, finding `GOAL_FALLBACK`; the verdict stays FAIL because it must equal the job's own record | verifier only (not audited) | 2026-11-17 |
| 24 sessions (sess24i): `goal_query` (the vLLM Goal engine, `--goal-overlap`), 24 sessions, no idle release, thinking on, the scripted edge-case client | 3613537 | `7a8c36e` (integrate-1116) | `SWE32_FAIL_VERIFIED` (verified on tbench, Slurm `FAILED`; the Goal text `VERIFIED`, the join `JOIN_FAIL` as expected of a stalled run): `CAPACITY_STALL` after format-error rollbacks ([docs/known-limits.md](known-limits.md) 2.6) | verifier only (not audited) | 2026-11-17 |
| Scripted edge cases with the Goal (keepedgegoalpi): `goal_query` (the vLLM Goal engine) with `--goal-overlap` off and on, 16 sessions, thinking on | 3613408 | `7a8c36e` (integrate-1116) | `SWE32_PASS_VERIFIED`, the Goal text `VERIFIED` on 192 of 192 calls on each server, `JOIN_PASS` (the off / on pair is not token-equal: 132 of 192 requests differ, 60 are equal; by the job harness's analysis the first differences are in generation from equal prompts) | verifier only (not audited) | 2026-11-17 |
| Scripted edge cases (keepedgei r2): `full`, 16 sessions, thinking on, format errors and rollbacks | 3613280 | `7a8c36e` (integrate-1116) | `SWE32_PASS_VERIFIED`, release gate 1 (keep render) on 192 of 192 calls; the join `JOIN_PASS` after the join fix (its first run `JOIN_FAIL`, a defect of the join, not of the runtime) | verifier only (not audited) | 2026-11-17 |
| GG1: Gemma 4 thinking on, token alignment with stock (`--thinking-history strip`, `--enable-thinking`) | 3583846 | `4ebff97` | `GG1_PASS_VERIFIED` (readings A1, A2 not met) | verifier only (not audited) | 2026-11-14 |
| ASG1: Gemma 4 `--async-scheduling` | 3560784 | `4ed1013` | `ASG1_PARTIAL_PASS_VERIFIED` | verifier only (not audited) | 2026-11-12 |
| rw2: replay of the vLLM Goal engine on Qwen | 3488202 | `cd42995` | `RWR2_PASS_VERIFIED` (every lm-format-enforcer call failed without a token) | verifier only (not audited) | 2026-11-01 |
| rw3: replay of the vLLM Goal engine on Qwen after the lm-format-enforcer fix | 3499550 | `c07d42a` | `RWR3_FAIL_VERIFIED` (lm-format-enforcer alone equal to the Hugging Face rewriter on 99 of 160 calls, below 0.7; 4 calls timed out) | verifier only (not audited) | 2026-11-01 |
| THG1 (G1): Qwen `--thinking-history strip` on `chat-text` | 3493649 | `8ca8511` | `THG1_PASS_VERIFIED` | verifier only (not audited) | 2026-11-01 |
| THG1B (G1b): Qwen `--thinking-history strip` on `chat-text` | 3500646 | `8ca8511` | `THG1B_PASS_VERIFIED` (the token differences against stock were CUDA-graph near ties) | verifier only (not audited) | 2026-11-02 |
| CMG1: Qwen `--goal-overlap` with thinking on (a `--thinking-history strip` pair and a keep-mode pair) | 3500604 | `c07d42a` | `CMG1_PASS_VERIFIED` (both pairs EXACT) | verifier only (not audited) | 2026-11-02 |
| CMG1: Gemma 4 `--goal-overlap`, one server with `--goal-rewriter-engine vllm` | 3508345 | `c07d42a` | `CMG1_FAIL_VERIFIED` (the two servers with the flag never became READY) | verifier only (not audited) | 2026-11-05 |
| CMG1: Gemma 4 `--goal-overlap`, one server with `--goal-rewriter-engine vllm` | 3538523 | `c07d42a` | `CMG1_PASS_VERIFIED` (both pairs EXACT) | verifier only (not audited) | 2026-11-08 |
| ASG1: Qwen `--async-scheduling` | 3519531 | `4a498dc` | `ASG1_FAIL_VERIFIED` (KeyError when the second turns began; fixed by async-fix2) | verifier only (not audited) | 2026-11-05 |
| ASG1: Qwen `--async-scheduling` | 3532746 | `4ed1013` | `ASG1_PARTIAL_PASS_VERIFIED` (every b = 1 pair token-exact; partial from the comparator's `state_digest` check) | verifier only (not audited) | 2026-11-07 |
| ASG1: Gemma 4 `--async-scheduling` | 3519554 | `4a498dc` | `ASG1_FAIL_VERIFIED` (no asynchronous server ran: the first server did not become READY while the weights stalled loading) | verifier only (not audited) | 2026-11-06 |
| ASG1: Gemma 4 `--async-scheduling` | 3549350 | `4ed1013` | `ASG1_FAIL_VERIFIED` (token-exact at b = 1; 12 of 97 requests per server timed out in the harness's client) | verifier only (not audited) | 2026-11-09 |
| `--phase1-graph-tokens` | 3569412 | `b6f6b9f` | `SPD3_PARTIAL_PASS_VERIFIED` | verifier only (not audited) | 2026-11-14 |
| gbench: the gather probe and `tools/token_gather_bench.py` | 3611486 | `bdc1ca8` (integrate-1114) | `SWE32_PASS_VERIFIED`, `GATHER_PROBE_PASS`, `TOKEN_GATHER_BENCH_PASS` (114 of 114 rows; decision `DEVICE_DIRECT_ADVISED`) | verifier only (not audited) | 2026-11-16 |
| Token gather, one SWE-bench task: `observation_query` with `--token-gather markers` and without, thinking on | 3603240 | `2b2a752` (integrate-1114) | `SWE32_FAIL_VERIFIED` (a harness check that did not count a session's gather blocks until its close, not the runtime; the gather probe `GATHER_PROBE_PASS`, decision `UVA_ADVISED`) | verifier only (not audited) | 2026-11-14 |
| Token gather, the first GPU job: the gather probe, then a gather server | 3602601 | | no verification record: the gather probe passed, the first gather server refused at startup on vLLM 0.24.0's FlashAttention view layout (fixed by `74c1d32`, `eec08cf`) | not verified offline | |
| ASG1 (G1 r1): Qwen `--async-scheduling` | 3499729 | `f162120` | `ASG1_FAIL_VERIFIED` (every runtime asynchronous server died at the Scheduler's construction; fixed by async-fix) | verifier only (not audited) | 2026-11-03 |

### Earlier GPU checks, pilots and measurements

The jobs that the pages of docs/ name for a single feature, a measurement or a pilot, up to runtime `cf2b28d` and the release repository's acceptance, oldest first. Verdict from each job's verification record (the last one, where a job was verified again); "Audited" as above.

| Check | Job | Commit | Verdict | Audited | Date |
| --- | --- | --- | --- | --- | --- |
| A 4-session synthetic run on the clean environment of job 3270721 | 3300301 | `6e52a94` | `MS2A_MEASURE_PASS_VERIFIED` | verifier only (not audited) | 2026-10-03 |
| SWE-bench pilot: 10 Astropy instances × `full`, `goal_query`, `observation_query`, 5 sessions per server, 784-token pages | 3300434 | `b1e6988` | `V12_PILOT_PASS_VERIFIED` | verifier only (not audited) | 2026-10-04 |
| smoke-dp1 of the decoupled layout: `full` in three layouts; the four policies at 16-token pages; 4 sessions at 24 GiB | 3308015, 3308414, 3308482 | `8f1af6a` | no verification record (not re-verified) | not verified offline | |
| SWE-bench pilot at 16-token pages, 10 instances: `full` | 3313802 | `780fe54` | `V14_PILOT_PASS_VERIFIED` | verifier only (not audited) | 2026-10-05 |
| The same pilot: `observation_query` | 3314010 | `780fe54` | `V14_PILOT_PASS_VERIFIED` | verifier only (not audited) | 2026-10-06 |
| The same pilot: `goal_query` | 3314319 | `780fe54` | `V14_PILOT_PASS_VERIFIED` | verifier only (not audited) | 2026-10-06 |
| The same pilot: `random_fill` | 3314623 | `780fe54` | `V14_PILOT_PASS_VERIFIED` | verifier only (not audited) | 2026-10-06 |
| Decoupled layout at P = 16: performance, `full` token-identical in both layouts | 3321538 | `b37044f` | `SPF1_PERF_PASS_VERIFIED` | verifier only (not audited) | 2026-10-06 |
| CUDA graph modes | 3338893 | `f19f464` | `SCG2_FAIL_VERIFIED` (its plan's steps were incomplete, finding `STEPS_INCOMPLETE`) | verifier only (not audited) | 2026-10-08 |
| 16 sessions on one server (synthetic sessions) | 3339019 | `afffd2b` | `SMS1_PASS_VERIFIED` | verifier only (not audited) | 2026-10-08 |
| CUDA graphs | 3339448 | `f19f464` | `SCG2_PASS_VERIFIED` | verifier only (not audited) | 2026-10-08 |
| 16 sessions with `--cuda-graphs full`; the session pool (24 tasks, K = 8) with `full-decode` | 3343069 | `613a7d5` | `SINT1_PASS_VERIFIED` | verifier only (not audited) | 2026-10-09 |
| The 131,072-token context, 4,096-token replies and the arena cap, 16 `observation_query` sessions | 3348705 | `710bfc6` | `SCTX1_PARTIAL_PASS_VERIFIED` | verifier only (not audited) | 2026-10-10 |
| Admission optimizations: `observation_query` at one session and at 16 | 3349078 | `4eae441` | `SADM1_PASS_VERIFIED` | verifier only (not audited) | 2026-10-10 |
| Admission optimizations: `goal_query` at 5 and 16 sessions | 3349504 | `4eae441` | `SADM1_PASS_VERIFIED` | verifier only (not audited) | 2026-10-10 |
| A per-request comparison of Transformers 5.13 and the pinned `tokenizer.json` (refused such requests with HTTP 500) | 3354504 | | no verification record | not verified offline | |
| Gemma 4 stage 0: stock vLLM, one session of 10 turns | 3364469 | — | `GEM0_PASS_VERIFIED` | verifier only (not audited) | 2026-10-12 |
| Gemma 4 stage 0: stock vLLM, the agent run | 3364911 | — | `GEM0_PASS_VERIFIED` | verifier only (not audited) | 2026-10-12 |
| Decode fast path (64-token pages once, `full` in a decode measurement) | 3366128 | `43fd9c3` | `SPD1_PARTIAL_PASS_VERIFIED` | verifier only (not audited) | 2026-10-12 |
| The 262,144-token context | 3360375 | `e3efc22` | `SCTX2_PASS_VERIFIED` | verifier only (not audited) | 2026-10-13 |
| Decode measurement (SPD2) | 3379206 | `227dedb` | `SPD2_PARTIAL_PASS_VERIFIED_WITH_VERIFIER_FIX` (first verification `SPD2_FAIL_VERIFIED`) | verifier only (not audited) | 2026-10-16 |
| Gemma 4: the two-pool layout, selective runtime off | 3397695 | `e0df9d5` | `GEM1_PASS_VERIFIED` | verifier only (not audited) | 2026-10-18 |
| Gemma 4 runtime servers (g1a): every runtime server stopped at startup | 3398198 | `0d37346` | `GEMR_PARTIAL_PASS_VERIFIED` | verifier only (not audited) | 2026-10-19 |
| Gemma 4 runtime servers (g1b): every runtime server stopped at startup | 3398614 | `4eb9f9d` | `GEMR_PARTIAL_PASS_VERIFIED` | verifier only (not audited) | 2026-10-19 |
| Host work per completion: one request at a time, histories of 2K to 257K tokens | 3410181 | `f9322a6` | `SHBM_PASS_VERIFIED` | verifier only (not audited) | 2026-10-19 |
| Gemma 4 runtime servers (g1a) after the assembly fix | 3410750 | `316a1b4` | `GEMR_PASS_VERIFIED` | verifier only (not audited) | 2026-10-21 |
| Gemma 4 runtime servers (g1b) after the assembly fix | 3410948 | `316a1b4` | `GEMR_PARTIAL_PASS_VERIFIED` (its `goal_query` server failed) | verifier only (not audited) | 2026-10-21 |
| Gemma 4 runtime servers (g1c) | 3429673 | `9ef0d6d` | `GEMR_PASS_VERIFIED` | verifier only (not audited) | 2026-10-22 |
| Gemma 4 runtime servers (g1b), 16 sessions | 3429672 | `9ef0d6d` | `GEMR_PASS_VERIFIED` | verifier only (not audited) | 2026-10-24 |
| Decode check on Qwen: `--decode-fastpath3` and `--deferred-completion-io` against run-to-run spread | 3429022 | `120bc3c` | `SPD3_PARTIAL_PASS_VERIFIED` (some rounds lack the stock servers' measurement windows) | verifier only (not audited) | 2026-10-24 |
| Replay of the recorded Goal rewriter calls, one at a time and batched | 3443061 | `53e3b7a` | `RWRP_PARTIAL_PASS_VERIFIED` (its timing arm was cut) | verifier only (not audited) | 2026-10-24 |
| Gemma 4 runtime servers (g1d) | 3461663 | `cf2b28d` | `GEMR_PASS_VERIFIED` | verifier only (not audited) | 2026-10-26 |
| Decode check with `TRITON_ATTN` | 3461668 | `cf2b28d` | `SPD3_PARTIAL_PASS_VERIFIED` | verifier only (not audited) | 2026-10-28 |
| The release repository's GPU acceptance (its fast tests) | 3504770 | — | `RLAC_PASS_VERIFIED` | verifier only (not audited) | 2026-11-02 |
| Decode step measurement (8 requests decoding at 2K-token contexts) | 3561459 | `51ecc30` | `SPD3_PARTIAL_PASS_VERIFIED` | verifier only (not audited) | 2026-11-09 |

## Branches

| Branch | Commit | Contents | Checks |
| --- | --- | --- | --- |
| `main` | the merge of integrate-1117 and `7892db3` (2026-11-17) | runtime `2fb7f62` (integrate-1117) | release check of `2fb7f62`: 2,287 tests, 141 skipped; `tools/policy_parity.py` against `7a8c36e`: only `serve.py --help` differs; GPU: the tables above, and [docs/known-limits.md](known-limits.md) 5.7 for what has not run |
| `main-integrate-1112` (tag) | `7892db3` | runtime `4ebff97` (integrate-1112), `main` from 2026-11-15 to 2026-11-17 | release check 2,059 tests; GPU smokes per model (the tables above) |
| integrate-1114, integrate-1116, integrate-1117 | `a6d3ef7`, `21f69e4`, and integrate-1117's last commit before the merge | merged into `main` on 2026-11-17 and deleted from GitHub; their entries are in [History](history.md#revisions) | release checks: `bdc1ca8` 2,175 tests, `b88a6a7` 2,283, `2fb7f62` 2,287 |
| `data-swe32` (on GitHub) | `947e9fb` | the run archives and per-run tables of eight SWE-bench runs (3583822, 3583081, 3539242, 3539260, 3576565, 3577833, 3461703, 3611601) and two figures; no code | |

Planned GPU work: the 150-instance runs of the README's [24-session configuration](../README.md#24-session-configuration-the-groups-150-instance-runs-2026-11-17-not-yet-gpu-validated) on `main`'s runtime `2fb7f62`, queued on 2026-11-17: job 3616690 (`full`), then 3616705 (the stock control), then 3616729 (`ledger_query`); then the concurrency sweep.
