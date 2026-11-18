# Release gates

English | [简体中文](release-gates.zh-CN.md)

For whoever changes this runtime or reports its numbers: how a change reaches `main`, what a GPU job must show before its numbers are used, and what each metric means. Rules only; what has passed is in [docs/status.md](status.md), what does not work yet in [docs/known-limits.md](known-limits.md). The process is that of the repository's members; the maintainer decides.

## 1. Branches and `main`

- A change is built on a feature branch, merged into an integration branch `integrate-MMDD` and tested there; it reaches `main` after the GPU tests of section 2 pass and the maintainer approves the push.
- On GitHub this repository carries `main`, the integration branches, the data branch `data-swe32` (run archives) and the branches that collaborators push themselves; the feature branches that go into an integration branch stay local.
- `main` names its stable commit: the first section of [docs/status.md](status.md#main-and-the-stable-commit) and the README's first lines say which commit `main` is, which runtime commit it holds and which commit is the stable tag's.
- **Stable tag.** The last commit that passed a run in all six cells of the model × benchmark matrix (Qwen3.6-27B and Gemma 4 31B IT; SWE-bench Verified, Terminal-Bench 2.0, MemoryAgentBench), for whoever needs the most stability. It is created once a commit has; until then [docs/status.md](status.md) names the closest one.

## 2. GPU tests by change type

The rule for `main` of 2026-11-15 (it replaced the full model × benchmark smoke matrix as the gate):

| Change | Before `main` |
| --- | --- |
| Opt-in (a new switch, off by default; a plug-in policy) | its own GPU test, and the CPU proof that the default path is unchanged: `tools/policy_parity.py` byte-identical against the base, `check_git_release.py` passes |
| Default-on or on a shared path | the release check and at least one quick GPU smoke per model (Qwen and Gemma) |
| Client default, documentation | the release check (code), or the documentation checks (section 5) |

Each push to `main` needs the maintainer's approval, and [docs/status.md](status.md) records which cells and policies the change has run.

## 3. When a number may be used

A number from a GPU job enters [docs/status.md](status.md) as a result, or a paper or a group report, only when the job's verdict is a pass verified offline and these three independent reference checks pass on it. Where a harness does not run one yet, the status row says so. The checks gate the jobs on integrate-1116 and later runtimes (decided 2026-11-16); the rows of earlier jobs in [docs/status.md](status.md) were not gated by them (check 1 would not hold for the thinking-on runs before keep-rebuild-1116 after a session's first format error, [docs/known-limits.md](known-limits.md) 1.4). Check 2 is deferred for now: until it runs, the rows say that it has not run.

1. **Keep render on every call** (thinking on, the XML protocol). `tools/keep_render_check.py` renders each call's request with the pinned template (`preserve_thinking`) and compares it with the served prompt: `equal`, `text_equal` and `exact_equal` per call. Pass: the verdict (`exact_equal` and `text_equal`) holds on every call of every session.
2. **Full against stock, token for token, on a short stretch, periodically.** On a fixed short stretch (the same prompts, a window of decode steps) the runtime's `full` server and stock vLLM 0.24.0 with the same kernels give the same token ids. Run whenever the shared path changes, and at least once per runtime commit that produces numbers. Pass: equal token ids in the window (a difference is explained, for example by CUDA-graph near ties, or the numbers wait).
3. **Pool invariants after every close.** After each session close the close receipt balances (`released.blocks` = `resident_pages` + `tail` + `gather_blocks`, [docs/design.md](design.md#8-evidence-per-server) section 8), and with no session open the pools are back at their startup sizes. Pass: every close of the run balances and nothing is left pinned.

## 4. Scripted edge-case smokes

Each edge case has a CPU script that runs in the release check and, where noted, a GPU smoke; a result goes into [docs/status.md](status.md).

| Case | CPU script | Expected | On the GPU |
| --- | --- | --- | --- |
| The pool is full: a turn waits (`CAPACITY_WAIT`) until a close or a release | `tests/test_ms16.py`, `tests/test_adm_overlap.py`, `tests/test_idle_release.py` | no preemption; every turn equal to a server that never waits | the idle release in job 3583081 |
| A session closes and its pages return | `tests/test_ms_pool.py`, `tests/test_token_gather_e2e.py` | the close receipt balances (section 3) | every harness run |
| A format-error rollback | `tests/test_keep_rebuild.py`, `tests/test_inplace_32.py` | the exact earlier history; the abandoned retry restores the session | not yet scripted |
| 32 sessions | `tests/test_ms32.py`, `tests/test_inplace_32.py` | the launcher's limits; no preemption | not run |
| One Gated-DeltaNet slot per request (`--gdn-inplace-crossing`) | `tests/test_gdn_inplace.py`, `tests/test_inplace_32.py` | outputs equal to the flag off | not run |
| A reply cut at `max_tokens` while another request decodes (`--async-scheduling`) | `tests/test_async_sched.py` (`ResolveTests`) | no failure; its stop decided as on the synchronous server | the async-cut check of job 3604146 |

## 5. Checks before a commit

- **Code, pins, manifests or anything in the payload:** the full release check, `python -B check_git_release.py --ref HEAD`, one full suite at a time on a machine; a timing failure under load is run again alone before any code change.
- **Documentation only:** the tests that read the documentation (`test_installer`, `test_chat_text_server`, `test_kv_session_server`, `test_gemma_chat_text`, `test_gemma_kv_session`, `test_ctx_256k`, `test_speed_defaults`, `test_tb_client`, `test_policy_registry`) and `tests/test_docs_parity.py` (links and the English / Chinese parity, `tools/docs_check.py`); the release check of the code it rests on still stands and is named.
- **On a cluster's login node:** light checks only (the harness's own self-test and dry run, about 3 to 5 minutes); the full suite runs elsewhere.

## 6. Numbers for the group or a paper

- Every number is audited before it is shown to the group or used publicly: re-derived from the raw evidence by someone (or an agent) other than its drafter, and compared with stock vLLM through an in-window decode step of the same job.
- Latency and throughput in the paper come from the GPU-validated optimized runtime against stock vLLM; the optimizations themselves are not described there.
- A number that is not audited yet is marked "not audited" wherever it appears.

## 7. Metrics

The definitions the harnesses use; the source lines are those of the snapshot `57887c7` of `harnesses/`.

| Metric | Definition | Source |
| --- | --- | --- |
| SWE-bench resolved | instances whose evaluation report says resolved, of the planned instances (32) | `harnesses/perf-swe32/README.md` |
| SWE-bench throughput | `tok/s (runner span)` = the client's generated tokens / the span from the first instance's start to the last instance's end; `throughput_all` adds the server's tokens of instances that left no trajectory | `harnesses/perf-swe32/README.md` 451-455 |
| Terminal-Bench accuracy | solved trials (Harbor's verifier reward 1.0) / the planned 89 (or the subset's size) | `harnesses/perf-tb16-strip/README.md` 1135 |
| Terminal-Bench throughput | ATIF main-agent completion tokens / the agents' span (first agent start to last agent end), summarization sub-calls apart (approved 2026-10-17) | `harnesses/perf-tb16-strip/README.md` 1135-1139 |
| Unrecorded tokens | a reply delivered but not in the trajectory (a trial cut by its agent timeout: `Unrecorded tok (timeout-cut)`), short summaries of the context-limit fallback (`Short summary tok (unrecorded)`) and replies cut at the length bound (`Length-cut tok (unrecorded)`) are counted beside the numerator, never in it; the client-vs-server cross-check allows exactly them | `harnesses/perf-tb16-strip/README.md` 1140-1173 |
| Clean ITL | per batch size b, the steady intervals whose ITL stays below bound(b) = the smallest ITL histogram bound ≥ 2 × the lower quartile of the steady intervals' mean ITL at b; read `clean_itl_ms` / `clean_tok_s` for a plain decode step | `harnesses/perf-tb16-strip/README.md` 134-144 |
| Decode time lost | per sample interval sum(ITL) − observations × the clean step at the interval's batch size, summed over the run, as a share of vLLM's ITL sum; low / high from the interval ends' clean steps | `harnesses/perf-tb16-strip/README.md` 146-151; `harnesses/perf-swe32/README.md` 445-450 |
| MemoryAgentBench score | per case the metric of its ability: `exact_match` (in-context learning), `substring_exact_match` (RULER QA, FactConsolidation), an LLM judge for the InfBench summaries (`judge_pending` while no judge runs); the composite: the mean over the abilities, without a judge `composite_score_without_judge`, the mean over the other three (scoring is deferred, [docs/known-limits.md](known-limits.md) 5.1) | `harnesses/perf-mab/README.md` 850-852; [docs/MEMORYAGENTBENCH_BENCHMARK.md](MEMORYAGENTBENCH_BENCHMARK.md#a11-metrics) A.11 |

## 8. Documentation

- English and Chinese pages change in the same commit; `tools/docs_check.py` keeps their headings, code blocks, commit ids, numbers and linked pages equal and every link valid.
- Job results go to [docs/status.md](status.md) only; other pages point there. A new limit goes to [docs/known-limits.md](known-limits.md). [docs/history.md](history.md) is appended to, never rewritten; frozen records go to `docs/history/`.
- Page names and the section numbers that code and the harness snapshots cite stay unchanged.
