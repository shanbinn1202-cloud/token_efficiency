# Known limits

English | [简体中文](known-limits.zh-CN.md)

For whoever runs or measures this runtime: what does not work yet, or works only with care. Each limit gives what happens, whom it affects, the workaround, its source and when it is lifted. Only limits that hold on this branch are listed; a lifted one moves to an entry of [History](history.md#revisions). Job results are in [docs/status.md](status.md); numbers marked "not audited" have not been re-derived independently ([docs/release-gates.md](release-gates.md#3-when-a-number-may-be-used)). As of 2026-11-17 (integrate-1117).

## 1. History, rollback and reasoning (thinking on)

### 1.1 A format-error rollback re-prefills the whole session

- **What happens.** When Mini-SWE-Agent rejects a reply (a format error, most often a reply cut at the 4,096-token bound while still reasoning) and retries with the previous request plus one user message, the server renders the exact earlier history, every earlier reasoning and generated token id kept (the rollback, history label `rollback`, section [6.8](running.md#68-thinking-on-stage-1---enable-thinking)). On the engine side it is still a rebuild at local turn 0: the whole session is prefilled again, and the selector, the ledger, the marker and the gather state of that session start again. The Goal of `goal_query` and `goal_query_markers` goes on (1.3).
- **Affects.** Thinking on with `--protocol mswea-xml`, every policy; the cost is one full prefill per format error (in the five audited runs 1.70 % to 2.65 % of the calls were format errors, entry 1.4).
- **Workaround.** None in the runtime; fewer replies cut at the bound mean fewer rollbacks.
- **Source.** keep-rebuild-1116 (`eb43675`, `51af079`); sections [6.1](running.md#61-mini-swe-agent-xml-protocol) and 6.8; `tests/test_keep_rebuild.py`.
- **Lifted when** the KV is kept across a rollback: that needs the Gated-DeltaNet carry at the rejected reply's header start, held after every completion (and the Gemma window at that point), a worker and engine change that needs its own GPU test.

### 1.2 On Gemma 4 a rebuild that is not a rollback drops the earlier reasoning

- **What happens.** A rebuild other than the rollback of 1.1 re-renders the history: on Qwen with the template's `preserve_thinking`, so the reasoning stays; on Gemma 4 with its template as it is, which drops the reasoning of earlier replies.
- **Affects.** Gemma 4 with thinking on in the keep mode (`--enable-thinking`), after any rebuild that is not a rollback.
- **Workaround.** None; the rollback (the common case) keeps the exact history on both profiles.
- **Source.** keep-rebuild-1116; section [6.8](running.md#68-thinking-on-stage-1---enable-thinking).
- **Lifted when** the Gemma rebuild renders the kept reasoning.

### 1.3 The Goal carried across a rollback changes `goal_query`'s results from a session's first retry

- **What happens.** Before goal-rollback-1116 every rebuild, a format-error retry included, started the maintained Goal again from the task. Since then a retry that repeats the previous request plus one user message keeps the Goal of `goal_query` and `goal_query_markers`: the retry turn reuses the Goal of the turn whose reply was dropped, recorded as a `KEEP` that the runtime writes without running the Goal gate or the rewriter (trigger `rollback`, `query.goal.carry`), and the next turns update it as if the dropped reply had never been sent (section [6.1](running.md#61-mini-swe-agent-xml-protocol)). Up to a session's first retry nothing changes; from it on the Goals differ from an earlier runtime's: in the offline replay of job 3576565's requests (not audited) the Goal was carried at all 40 rollbacks, and the request after a carried rollback got a different Goal in all 38 cases (earlier runtime `INITIALIZE` 5, `PATCH` 31, `REWRITE` 2; now `KEEP` 13, `PATCH` 23, `REWRITE` 2). A carried turn calls no rewriter and builds no Goal: it does not park at a Goal and its native record has no Goal timing fields. The Goal owner has not confirmed the change yet. Not run on a GPU ([docs/status.md](status.md)).
- **Affects.** `goal_query` and `goal_query_markers` on the XML protocol (Qwen and Gemma 4, thinking on or off); a comparison of `goal_query` results across this change; a decision histogram (the carried `KEEP`s are not the gate's); a job harness that requires every `goal_query` admission of a parking server to park at its Goal (entry 5.5 for Gemma).
- **Workaround.** Compare `goal_query` results from one runtime version only; count rows with trigger `rollback` (or `query.goal.carry.carried`) apart from the gate's `KEEP`s; in harness checks treat a carried row as a turn that built no Goal.
- **Source.** goal-rollback-1116 (`f9ada25`, `a39dd08`, `43ff7b0`; merged as `7a8c36e`); section 6.1; `tests/test_goal_rollback.py`.
- **Lifted when** the Goal owner confirms the change and a GPU run of `goal_query` on this runtime passes with a harness that accepts carried rows; results across the change stay not comparable.

### 1.4 Runs before the keep-rebuild fix lost the earlier reasoning after a format error

- **What happens.** Before keep-rebuild-1116 a rejected reply made the XML protocol re-encode every earlier message from its content: from a session's first format error on, every earlier reply's reasoning was gone from the KV history for the rest of the session. These runs are keep-mode baselines only up to each session's first format error.
- **Affects.** Every thinking-on SWE-bench run on runtimes `60c5d21`, `ba4d1ba` and `4ebff97` and the builds with them. Share of calls at or after the session's first rebuild, per job (audited, 2026-11-16):

| Job | Policy | Runtime | Calls | Format errors | Sessions hit | Calls after the first error |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 3583081 | `full` | `ba4d1ba` | 1,262 | 26 | 12 / 32 | 351 (27.8 %) |
| 3539242 | `recent_fill` | `60c5d21` | 1,613 | 32 | 12 / 32 | 683 (42.3 %) |
| 3539260 | `observation_query` | `60c5d21` | 1,548 | 41 | 16 / 32 | 640 (41.3 %) |
| 3576565 | `goal_query` | `ba4d1ba` | 1,843 | 40 | 14 / 32 | 869 (47.2 %) |
| 3577833 | `akv_rule` | `ba4d1ba` | 1,411 | 24 | 11 / 32 | 429 (30.4 %) |

  The `ledger_query` run 3611601 (`51b4085`) shows the same (35 rollbacks, 519 of 1,548 calls after the first error; replayed offline, not audited); the async pair 3611917 / 3612333 ran on `4ebff97`, without the fix (not measured). In job 3583081, 25 of the 26 rejected replies were reasoning cut at the 4,096-token bound.
- **Workaround.** Compare these runs with each other, not with a run that keeps its reasoning; the stock reference (job 3583822) drops every earlier reasoning by design (`reasoning_dropped`).
- **Source.** The audit of 2026-11-16 (per-run numbers from the runtime rows, the trajectories and the pinned tokenizer and template); the replay with `tools/keep_render_check.py`.
- **Lifted** for runs on a runtime with keep-rebuild-1116 (integrate-1116 and later).

## 2. Capacity, sessions and pools

<a id="21-more-than-16-sessions-and---gdn-inplace-crossing-are-cpu-tested-only"></a>

### 2.1 More than 16 sessions (XML and kv-session) and `--gdn-inplace-crossing` are CPU-tested only

- **What happens.** A server takes up to 32 sessions (`--max-running-sessions`, `--admission-overlap`; chat-text pools keep 24), and `--gdn-inplace-crossing` holds one Gated-DeltaNet slot per running request. Neither more than 16 sessions on the XML and kv-session protocols (sessions-32-1116) nor `--gdn-inplace-crossing` has run on a GPU. Chat-text servers, whose cap was 24 before, did run more than 16 sessions on the GPU, on earlier runtimes: K = 20 in the Terminal-Bench dry runs (for example jobs 3399075, 3399437, 3461661 and 3560498, [docs/status.md](status.md)).
- **Affects.** Concurrency sweeps above 16; at 25 GiB and 262,144 tokens with 4,096-token replies the launcher takes K ≤ 25 without the flag (section [4.4](running.md#44-several-sessions-on-one-server)). The job harnesses still pin 16 (their open cap, task builder and capture-size check) until they are re-pinned.
- **Workaround.** Stay at 16 sessions for results; run a concurrency sweep as its own plan.
- **Source.** sessions-32-1116 (`7d62df4`, `67b601e`), gdn-inplace-1116 (`ab5d8be`, `dcd4c45`); `tests/test_ms32.py`, `tests/test_gdn_inplace.py`, `tests/test_inplace_32.py`; section [4.6](reference.md#46-decoupled-kv-layout-and-16-token-pages).
- **Lifted when** a GPU smoke per setting passes (the release gates' opt-in rule).

### 2.2 The Gated-DeltaNet pool reserves 2S + 1 slots before the attention pool

- **What happens.** Without `--gdn-inplace-crossing` the decoupled layout reserves K = 2S + 1 Mamba slots for S sessions (1 + 3K blocks of 51,314,688 bytes) out of `--kv-cache-gib` before the attention pool gets the rest: at 25 GiB and 16-token pages 100 / 124 / 148 / 196 blocks (4.78 / 5.93 / 7.07 / 9.37 GiB) at 16 / 20 / 24 / 32 sessions, leaving 20,706 / 19,531 / 18,357 / 16,008 attention blocks. Stock vLLM shares one pool. An investigation of six runs at 16 sessions (not audited) saw at most 17 slots in use at request ends (18 once, in an async run); the reservation covers the worst case the capacity gate assumes. At 20 and 24 sessions the CPU fakes showed one Mamba `CAPACITY_WAIT` each (a conservative wait, no preemption).
- **Affects.** The attention pool of every Qwen server on the decoupled layout, more so above 16 sessions.
- **Workaround.** `--gdn-inplace-crossing` (K = S + 1: 2,349 / 3,523 / 4,698 more attention blocks at 16 / 24 / 32 sessions), CPU-tested only (2.1).
- **Source.** Sections [4.6](reference.md#46-decoupled-kv-layout-and-16-token-pages) and [4.4](running.md#44-several-sessions-on-one-server); [docs/exp-decouple-pages/DESIGN.md](exp-decouple-pages/DESIGN.md) D4.
- **Lifted when** `--gdn-inplace-crossing` is GPU-validated and made the default.

### 2.3 At K = 20 decode steps of more than 16 rows fall back to piecewise graphs

- **What happens.** vLLM 0.24's dispatcher captures FULL decode graphs only for the capture sizes up to `max_num_seqs`. At K = 20 a step of 17 to 20 requests is padded to 24 and replays the piecewise graph in `--cuda-graphs full` (at K = 24 and 32, K itself is a size).
- **Affects.** Servers at K = 20 (the Terminal-Bench servers of A.3 and A.14) and other K that are not capture sizes (12, for example).
- **Workaround.** K = 16, 24 or 32 where the configuration allows it.
- **Source.** Section [4.8](reference.md#48-cuda-graphs); `tests/test_ms32.py`.
- **Lifted when** the capture sizes include K (a vLLM configuration change, then a decode check on the GPU).

### 2.4 `full` at 16 sessions in 25 GiB needs `--release-idle-pages`

- **What happens.** Without it job 3579273's `full` server stalled at the admission gate for 55.6 min: no turn ran and the free pages held no waiting turn. With it `full` passed (job 3583081, two releases of 1,726 pages).
- **Affects.** `full`; `akv_rule` (no budget) should use it too, although job 3577833 did not stall without it; `akv_rule` with `--release-idle-pages` has not run on a GPU.
- **Workaround.** Start these servers with `--release-idle-pages`.
- **Source.** Section [4.13](reference.md#413-releasing-idle-sessions-pages-opt-in); [docs/status.md](status.md).
- **Lifted when** never: a capacity rule, not a defect.

### 2.5 Gemma 4 kv-session: 16 sessions do not fit 25 GiB at a 32,768-token budget

- **What happens.** The selective rule would need 5,699 global pages of the 4,957 usable, so the Gemma MemoryAgentBench servers run 8 sessions.
- **Affects.** Gemma 4 on MemoryAgentBench at the 32,768-token budget of its configuration of record.
- **Workaround.** 8 sessions, or a smaller budget (the current configuration's 8,192 tokens, [docs/status.md](status.md#current-configuration)).
- **Source.** [docs/MEMORYAGENTBENCH_BENCHMARK.md](MEMORYAGENTBENCH_BENCHMARK.md#a15-gemma-4-31b-it) A.15.
- **Lifted when** never: a capacity rule.

### 2.6 More than 16 sessions without `--release-idle-pages` can stall after a format-error rollback

- **What happens.** A format-error rollback re-prefills the session's whole history in phase 1 (1.1). With many idle sessions holding their resident pages, that turn may need more attention blocks than are free while no request runs that could free any: the server stalls for good (`CAPACITY_STALL`). Job 3613537 (sess24i, 2026-11-16, runtime `7a8c36e`: 24 sessions, `goal_query`, no idle release, the scripted client) stopped so: after rollbacks in every session the next turn needed about 1,075 attention blocks in phase 1, 211 were free and nothing was running.
- **Affects.** Session pools above 16 sessions on the XML protocol with thinking on (format errors occur), every policy; `full` at 16 sessions in 25 GiB already needs the flag (2.4).
- **Workaround.** Start such servers with `--release-idle-pages` (section [4.13](reference.md#413-releasing-idle-sessions-pages-opt-in)): the capacity gate then releases idle sessions' resident pages (their DRAM copies stay) for exactly this turn. The README's [24-session configuration](../README.md#24-session-configuration-the-groups-150-instance-runs-2026-11-17-not-yet-gpu-validated) uses it on every server.
- **Source.** Job 3613537 (`SWE32_FAIL_VERIFIED`, Slurm `FAILED`, [docs/status.md](status.md)); section [4.4](running.md#44-several-sessions-on-one-server).
- **Lifted when** never: a capacity rule, not a defect (`--release-idle-pages` is planned as a runtime default).

## 3. The Goal

### 3.1 The vLLM Goal engine child has no reserved GPU memory

- **What happens.** With `--goal-rewriter-engine vllm` the Qwen3-0.6B child takes the GPU memory left next to the 27B server (weights, 3 GiB of KV, activations, its graph pool and a CUDA context: about 5–6 GiB; in the replay job 3499550 it peaked at 5,550 to 5,786 MiB across the four engine arms, 5,786 and 5,718 MiB in the two memory-fit arms mem25 and mem32; verifier only, not audited); vLLM's startup gate needs 5 % of the GPU free. The launcher does not budget it.
- **Affects.** `goal_query` with the vLLM engine; on Gemma 4 less memory is left: in job 3538523 the Gemma server with the vLLM engine (gvll) peaked at 92,500 MiB of the GPU's 97,871, the one with the Hugging Face child (govl) at 89,874 MiB (verifier only, not audited).
- **Workaround.** Keep the stable command's sizes (25 GiB KV, `--goal-rewriter-vllm-kv-gib 3`).
- **Source.** Section [4.10](reference.md#410-admission-optimizations-opt-in); [docs/rewriter-vllm/DESIGN.md](rewriter-vllm/DESIGN.md).
- **Lifted when** the launcher reserves the child's memory.

### 3.2 The vLLM Goal rewriter is validated only inside the stable command

- **What happens.** `--goal-rewriter-engine vllm` passed in the stable configuration (job 3576565); on its own its Goals equalled the Hugging Face child's in 99 of 160 calls of a replay (job 3499550, below the 0.7 bar).
- **Affects.** Every other combination (other benchmarks, Gemma 4, without `--goal-overlap`).
- **Workaround.** The default Hugging Face rewriter, or the stable command as it is.
- **Source.** Section 4.10.
- **Lifted when** a replay or run of that combination passes.

### 3.3 The Goal features' regular expressions run through a runtime replacement

- **What happens.** The upstream quoted-literal pattern takes exponential time on an unclosed quote (job 3429670 stalled on it); the runtime replaces it at startup with a linear-time object that returns the same result and refuses another pattern. Three observation patterns of the same file are quadratic on whitespace-heavy text (about 1 s each at 8 KB).
- **Affects.** `goal_query` on every protocol.
- **Workaround.** The replacement (on by default since `53e3b7a`).
- **Source.** The review notes for the Goal owner in [the validation record](history/validation-cf2b28d.md); History, fix-engine-stall-1022.
- **Lifted when** the Goal owner fixes the patterns upstream.

### 3.4 The Terminal-Bench Goal inputs are provisional

- **What happens.** `goal_query` on the chat-text protocol uses inputs for Terminus 2 that the team chose on 2026-10-18 and the Goal owner has not confirmed.
- **Affects.** `goal_query` on Terminal-Bench, both profiles.
- **Source.** Section [6.7](running.md#67-chat-text-protocol-terminal-bench-20-with-terminus-2); [the validation record](history/validation-cf2b28d.md).
- **Lifted when** the Goal owner confirms or changes them.

## 4. Policies and protocols

### 4.1 `ledger_query`'s command parser covers common shell forms only

- **What happens.** The ledger reads the reply's one `<mswea_bash_command>` block as reads, edits and diffs; any other command is neither (no ledger item, no read coverage). `ledger_query` v2 (ledger-fixes-1116, order `ledger_dedupe_reasoning_light_v2`) fixed the findings of the policy's review: repeated commands across an edit, `.write()` without a write-mode open, truncated reads, full-path file matching, comments, subshells and `pushd` / `popd`, `git -C` / `rm` / `mv` and more option forms. What the parser still does not see:
  - Reads are cat, nl, less, more, bat, head, tail, sed without `-i`, awk line ranges and grep / rg with file operands (filtered downstream by head, tail, sed, grep, awk, nl or cat only), each operand a path with a file suffix and without a glob or variable: `cat Makefile`, `cat $f`, a directory search (`grep -rn foo src/`), find, ls and `cat a.py | wc -l` are no reads.
  - An edit whose files are not known clears every file's read history: an editor, patch, `git apply` / `am` / `stash` / `reset --hard` and the like without file operands, `sed -i` inside a loop or through xargs, `perl -i` inside a loop, a redirect to a path with a variable, a Python body that writes with no quoted file path (`Path(p).write_text(...)`).
  - A Python body (`-c` or a heredoc) is judged by its text: a write is an `open()` / `Path.open()` with a write mode, `write_text` / `write_bytes`, a shutil copy or move or an os rename or removal (any other way of writing is not seen), and the files written are its quoted paths (other than /tmp, /dev and output files). Running a script is an edit only when the session wrote it earlier with such a body; a repository's own script, or one written another way, edits nothing for the ledger.
  - After `cd $VAR` (an unknown directory) a relative path matches every file whose path ends with it.
  - A command that cannot be split into words (an unbalanced quote) is neither a read nor an edit.
- **Affects.** `ledger_query` (Qwen, the XML protocol only; kv-session, chat-text and Gemma 4 are refused).
- **Source.** [docs/policy-registry.md](policy-registry.md#8-the-query-hook-and-ledger_query-ledger-policy-1112) section 8 (ledger-fixes-1116 `dc49e7a`, `8b81f35`; merged as `a2f9e36`); `source-overlay/page_policies/ledger_query.py`; `tests/test_ledger_query.py`.
- **Lifted when** never as a whole: the parser's scope.

### 4.2 Marker pages and token gather serve the Qwen XML path only and do not combine

- **What happens.** `goal_query_markers`, `observation_query_markers` and `--token-gather` serve Qwen3.6-27B on `--protocol mswea-xml` with 16-token pages; Gemma 4 and the other protocols are refused. `--token-gather` serves the built-in `goal_query` and `observation_query` only, so the launcher refuses it with the marker policies and `ledger_query`. The runtime does not record the gather's time.
- **Source.** [docs/policy-registry.md](policy-registry.md#9-marker-pages-goal_query_markers-and-observation_query_markers-marker-pages-1114) sections 6 and 9; [docs/token-gather/DESIGN.md](token-gather/DESIGN.md).
- **Lifted when** a further design covers them.

### 4.3 Thinking-on results are not comparable with thinking-off baselines

- **What happens.** Every benchmark configuration of record (the Part A pages) is thinking off; the current runs are thinking on. A thinking-on result is compared only with thinking-on runs and the thinking-on stock reference.
- **Source.** [docs/status.md](status.md#current-configuration).

## 5. Benchmarks and measurement

### 5.1 MemoryAgentBench is not scored

- **What happens.** The MemoryAgentBench runs report verdicts, not scores. The Long-Range Understanding (InfBench summary) cases need an LLM judge that is not run, so they stay `judge_pending`. When the runs are scored, the composite is reported without them, as the mean over the other three abilities (`composite_score_without_judge`); the four-ability `composite_score` stays null while judge cases are pending ([docs/benchmarks/memoryagentbench.md](benchmarks/memoryagentbench.md#procedure) step 5). Scoring is deferred.
- **Source.** [docs/MEMORYAGENTBENCH_BENCHMARK.md](MEMORYAGENTBENCH_BENCHMARK.md#a10-evaluation) A.10 and Part B.
- **Lifted when** scoring is scheduled with a judge.

### 5.2 Gemma 4 on Terminal-Bench at 262,144 tokens failed for `goal_query`

- **What happens.** In job 3560498 (`TB16_FAIL_VERIFIED`) one `goal_query` turn needed 1,581 global pages while other sessions pinned 2,175 of the pool's 3,637, and no release frees pinned pages (`CAPACITY_EXCEEDED`).
- **Affects.** Gemma 4 `goal_query` on Terminal-Bench at the client's default context.
- **Workaround.** The 65,536-token runs that passed (`make_job.py --context 65536`).
- **Source.** [docs/status.md](status.md); [docs/TERMINALBENCH_BENCHMARK.md](TERMINALBENCH_BENCHMARK.md#a14-gemma-4-31b-it) A.14.

### 5.3 Gemma 4 with `--async-scheduling` is a partial pass

- **What happens.** Job 3549350 was `ASG1_FAIL_VERIFIED` (token-exact at b = 1, but 12 of 97 requests per server timed out in the harness's client); job 3560784 was `ASG1_PARTIAL_PASS_VERIFIED`; the Gemma MemoryAgentBench smoke with it passed (job 3601174).
- **Source.** [docs/async-sched/GEMMA.md](async-sched/GEMMA.md).

### 5.4 The Part B job scripts of the benchmark pages are untested

- **What happens.** The example job scripts of the benchmark pages (Part B: the Slurm scripts, and Terminal-Bench's `render_job.py` and `tb_agents.sh`) have not run as written; the GPU runs used the job harnesses.
- **Source.** [docs/TERMINALBENCH_BENCHMARK.md](TERMINALBENCH_BENCHMARK.md#part-b-procedure), [docs/SWEBENCH_BENCHMARK.md](SWEBENCH_BENCHMARK.md#part-b-procedure), [docs/MEMORYAGENTBENCH_BENCHMARK.md](MEMORYAGENTBENCH_BENCHMARK.md#part-b-procedure).

### 5.5 The Gemma job harness is not updated for the integrate-1116 runtime

- **What happens.** The Gemma SWE-bench job harness (perf-gemma-swe32) does not know the evidence that the integrate-1116 runtime writes (the `rollback` history rows, the rows of a Goal carried across a rollback, the session cap of 32), so a Gemma job on integrate-1116 would fail its verification.
- **Affects.** Gemma 4 GPU jobs on integrate-1116 and later; the current experiments use Qwen ([docs/status.md](status.md#current-configuration)).
- **Workaround.** Update the harness before a Gemma job on integrate-1116.
- **Source.** [harnesses/README.md](../harnesses/README.md); the decision of 2026-11-16.
- **Lifted when** the Gemma harness is updated and re-pinned to an integrate-1116 runtime.

### 5.6 SWE-bench: the agent prompt is written for Astropy, and the agent's commands run outside the repository's environment

- **What happens.** Every SWE-bench run without `REPO_ENV=testbed` (every run before the 150-instance runs of 2026-11-17) uses the PoC's agent prompt (`examples/swebench/upstream/swebench_prompt_override.yaml`) unchanged. Its test section is written for Astropy ("The testbed is an Astropy repository", `python -m pytest`), also for the Django instances (10 of the first 32, 128 of the first 150). The agent's commands run under `bash -c` (Mini-SWE-Agent 2.4.2's SWE-bench configuration), which does not read the image's `~/.bashrc`, so `python` is the base conda interpreter and the repository does not import until the agent activates the `testbed` environment itself. Checked on the agent host on 2026-11-17: `./tests/runtests.py` in django-11099 fails with "No module named 'django'" and `import astropy` in astropy-12907 with "No module named 'erfa'"; both work after `source /opt/miniconda3/bin/activate testbed`. The prompt's rule for a repository that cannot import (one standalone reproduction instead of repairing the environment) is the PoC's way around it. How often runs met this was not counted.
- **Affects.** The absolute resolve rates, and the steps and tokens per instance, of every SWE-bench run without `REPO_ENV=testbed`: they are not comparable with SWE-bench results obtained with the repository's environment active. Comparisons between configurations (the stock control, `full`, the selective policies) are not affected: all use the same prompt and environment.
- **Workaround.** `REPO_ENV=testbed` in the environment of `examples/swebench/run_agents.sh` (since 2026-11-17; the group runs the 150-instance runs with it), next to `MAX_RUNNING`, `MAX_TOKENS`, `STEP_LIMIT` and `STOCK` ([benchmarks/swe-bench.md](benchmarks/swe-bench.md#agents)); `examples/slurm/serve_gh200.sh` passes it on. Unset, everything runs as before. It does two things:
  1. The agent's commands run with the repository's environment active: the model override's interpreter becomes `["bash", "-lc"]` (Mini-SWE-Agent's own default; its SWE-bench configuration sets `["bash", "-c"]`), and the image's login shell activates `testbed`. Checked on the agent host on 2026-11-17 in containers started as Mini-SWE-Agent starts them (django-10097, django-11099, django-14315, astropy-7166, astropy-12907 and astropy-14995): `python` is `/opt/miniconda3/envs/testbed/bin/python`, the working directory stays `/testbed`, commands print the same output as under `bash -c` (no banner; only the `CONDA_*` variables and `PATH` differ), each command takes about 0.09 s longer, and `./tests/runtests.py` and `python -m pytest` run.
  2. The prompt is `examples/swebench/swebench_prompt_testbed.yaml`: the PoC prompt with its test section chosen by the image name (a Jinja condition on `image`, which Mini-SWE-Agent 2.4.2 passes to the prompt template). A Django image gets `cd /testbed && ./tests/runtests.py --settings=test_sqlite --parallel 1 <test label>` (the label is the module path under `tests/`, for example `utils_tests.test_text`) and "Django" in place of the Astropy dependency list; every other image gets the PoC's Astropy text byte for byte (rendered with Mini-SWE-Agent 2.4.2's own code for the first 150 instances on 2026-11-17: the 22 Astropy task messages equal the PoC prompt's, the 128 Django ones differ only in the test section).

  Use it for every configuration of a comparison (the stock control, every policy, the baselines): results with it are not comparable with runs without it.
- **Source.** [docs/SWEBENCH_BENCHMARK.md](SWEBENCH_BENCHMARK.md) (the properties of the PoC files); the check of 2026-11-17 above.
- **Lifted when** a run sets `REPO_ENV=testbed` (a new configuration, not comparable with the runs without it); the runs without it keep this limit.

### 5.7 Parts of `main` that no GPU job has run (since 2026-11-17)

- **What happens.** `main` took integrate-1114, integrate-1116 and integrate-1117 on 2026-11-17 before every part had run on a GPU. Not run on a GPU yet: the session pool's 256-task cap and a pool of 150 tasks; `ledger_query` v2; more than 16 sessions as a verified run (the 24-session smoke, job 3616345, completed all 24 tasks without a stall; its FAIL verdict comes from a defect of the job harness's check, [docs/status.md](status.md)); `--gdn-inplace-crossing` (off by default, CPU-tested only, 2.1); the Gemma 4 path of integrate-1116 and later (5.5).
- **Affects.** Whoever runs these parts from `main`.
- **Workaround.** The tag `main-integrate-1112` (`7892db3`, runtime `4ebff97`) is the previous `main`, whose runtime ran one GPU smoke per model. [docs/status.md](status.md) lists which jobs each part has.
- **Source.** The decision of 2026-11-17 to merge before these runs (Qwen only for now; `--gdn-inplace-crossing` stays opt-in and CPU-tested).
- **Lifted when** the 150-instance runs of `full` and `ledger_query` (jobs 3616690, 3616729) pass, for the task cap, `ledger_query` v2 and 24 sessions; a GPU pair for `--gdn-inplace-crossing`; a Gemma smoke on an updated harness.

## 6. Records

### 6.1 A cancelled admission's failure row has no request id, and a cancelled backoff misses one refusal

- **What happens.** The failure row of a request cancelled during its admission has no `request_id`, and a request cancelled during its capacity backoff misses its last counted refusal (the routes count a refusal only after the backoff returns): in job 3429830 the log holds 12,408 refusals and the counter 12,406 (from a harness snapshot, not audited). Bookkeeping only; the requests themselves are served or cancelled correctly.
- **Source.** History, runtime `cf2b28d` ("Known issue"); [harnesses/README.md](../harnesses/README.md) (the Terminal-Bench harness's status).
- **Lifted when** the routes count the refusal before the backoff and the failure row carries the id.

### 6.2 Events the runtime does not count

- **What happens.** The runtime has no counter for a reply cut at `max_tokens` while another request decodes under `--async-scheduling` (the async-cut case of job 3604146 was inferred from the request records), and none for the token gather's time.
- **Lifted when** counters are added to the evidence ([docs/design.md](design.md#8-evidence-per-server) section 8).
