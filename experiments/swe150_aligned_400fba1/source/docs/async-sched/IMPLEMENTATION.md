# --async-scheduling, option C: what is implemented (Qwen checked on the GPU at b = 1)

English | [简体中文](IMPLEMENTATION.zh-CN.md)

Date: 2026-10-31 (sections 8-10: 2026-11-03 to 2026-11-05; section 11: 2026-11-12). Branch `async-c` on thinking-strip `8ca8511` (the
design note `DESIGN.md` is cherry-picked from `async-sched` `3e5715e`). Nothing here had run on a GPU at that date.
The parts that decide correctness on the device (the void in the persistent buffers, the kernels skipping a voided
row, the stop predicate agreeing with vLLM's, the held updates sent from inside `execute_model`) were G1's to show
(what G1 measured on Qwen: the end of section 6). Code: `source-overlay/remote_ra_live_v2/async_sched.py`; tests:
`tests/test_async_sched.py`.

As built: async-c `f162120`, async-fix `a0798a8` (section 8), async-gemma `4a498dc` (section 9) and async-fix2
`4ed1013` (section 10) are on main; async-maxtok-1112 `13ee618` (section 11) came with integrate-1112 `4ebff97`,
`main`'s runtime since 2026-11-15. The GPU checks of each part (G1 on Qwen and on Gemma 4, the MemoryAgentBench jobs
of section 11) and their verdicts are in [docs/status.md](../status.md) and [History](../history.md#history); the
limit that holds on Gemma 4 is [docs/known-limits.md](../known-limits.md) 5.3. The rest of this page is the
implementation record of its dates.

## 1. The switch and its refusals

- `serve.py --async-scheduling` (default off). It requires `--decode-fastpath2`.
- It is refused together with `--gdn-device-handoff` or `--phase1-in-step`. async-c refused the Gemma 4 profile;
  async-gemma takes it (section 9, [GEMMA.md](GEMMA.md)).
- When on:
  - the config pins `async_scheduling: true` and `effective_server_config.async_scheduling: true`;
  - vLLM gets `--async-scheduling` in place of `--no-async-scheduling`;
  - the frontend compares vLLM's parsed flag (off: vLLM must not have parsed it on);
  - the engine sends the worker record `async_scheduling: true` and compares the worker's echo.
- The synchronous-only gates now require vLLM's scheduler to be asynchronous exactly when the config pins it:
  - `max_concurrent_batches` is 2 when pinned, 1 otherwise;
  - the batch queue exists and is empty at assembly;
  - the worker gate and the per-forward `runtime_mode` check follow the same rule.
  - The gates are in `bootstrap_contract`, `policy_runtime/bootstrap`, `native_worker_registration`,
    `startup_runtime` and `positions.runtime_mode`.
- Unchanged:
  - `private_authority.Topology`, a declaration of the authority's own process topology;
  - the legacy `remote_ra_growing_worker/plan.py` and `remote_ra_qwen_semantics/hooks.py`, which stay synchronous only.
- Gemma: refused in async-c (the window layout's owner, rows and fast path 3 journal were not checked, and the void
  required a Gated-DeltaNet state index); taken since async-gemma, section 9.

## 2. The step (DESIGN 4.3), as built

1. **vLLM.** `step_with_batch_queue` at two batches runs `schedule(N+1)` and `execute_model(N+1)` (its native
   preparation) while forward N runs.
2. **Held updates.** `async_sched.attach_engine` unwraps fast path 2's flushing `schedule`, so the step's
   allocation updates stay queued (`EngineRpcBoundary.async_hold`).
3. **Placeholders.** The allocation and interval checks accept one output placeholder on a RUNNING decode
   (`placeholders=True` in `allocation_guard` and `chunk_protocol`). Synchronously nothing changes.
4. **Resolve.** The guard (`async_sched.guarded`) does not drain the device first (a connector check only) and
   does not enter the step. The runner's `before_forward` hook calls `resolve`:
   1. `settle`: the issued step N's deferred after-forward work (`finish_step`) closes its leases, then
      `leave_issued`.
   2. N's sampled ids are read once from vLLM's `prev_sampled_token_ids` / `prev_req_id_to_index`. This is one
      device read; DESIGN said "the copy event", see section 4. Only the requests of N that N+1 schedules are read
      (async-maxtok-1112, section 11): vLLM's `schedule(N+1)` skips a request whose token of N is its
      `max_tokens`-th or reaches `max_model_len` (and may leave out another one: the token budget, a preemption),
      and the runner then takes it out of the batch and of `prev_req_id_to_index`. Its stop is decided by
      `update_from_output(N)`, as on the synchronous server, and it has no row of N+1 to void.
   3. The stops are decided by `EngineLink.stops` (async-gemma: only for tokens the engine has not applied yet,
      `EngineLink.applied`, section 9):
      - vLLM's own `check_stop` on `_Shadow(request, token)`: counts and output ids advanced by one, writes kept
        on the shadow;
      - then `marker_stops`: MarkerStop's answer after appending, without writing its per-request table.
      - Nothing is mutated.
   4. Each row of N+1 whose request stopped at N is voided (`void_rows`), on the metadata the runner stashed
      after the builders (`gpu_model_runner`: `_remote_ra_step_metadata`, only when `use_async_scheduling`):
      - every GDN metadata's `non_spec_state_indices_tensor[row] = NULL_BLOCK_ID`;
      - the token's slot is -1 in every FA layer's metadata, in its per-layer slot mapping and in each FA
        group's mapping;
      - Mamba groups' slot mappings are never written.
   5. `EngineLink.flush(voided)`:
      - sends the held updates without the stopped requests' entries;
      - restores their `native_allocation` and `native_tables` (H2).
   6. The continuing requests enter the step (`enter_execute`). `scheduled_rows` skips the voided rows, as it
      skips a probe row, so no lease, vector or update is made for them (H3, H4, D4).
5. **After execute.** The guard marks the entered requests `issued` (`qwen.issue_execute`); their after-forward
   work waits for the next resolve.
6. **Update and check.** `update_from_output(N)` then runs during forward N+1. Afterwards the engine requires
   every predicted stop to have finished (`EngineLink.check` in `step_with_batch_queue` and in the drain). A
   predicted stop that did not stop is a lost step and fails closed.
7. **Unpredicted stops.** A stop that was not predicted ran in N+1 with an open lease. Its completion fails
   closed: the worker's cursor is C'+1 against the computed end C', or its lease is still open.

**Pass boundaries.**

- The engine drains first (`EngineCore._remote_ra_async_drain`):
  - **when:** before every client request (registered adds, aborts, utility calls), a Goal resume and shutdown;
  - **what:** the batch in flight is popped and updated, with no further step scheduled.
- The lifecycle sweep does not drain. It expires registrations and non-live records only, and any worker RPC it
  makes settles the issued step.
- On the worker, `worker_dispatch` settles the issued step:
  - before any RPC other than `allocation_batch` / `allocation_update`;
  - and before a terminal (`complete`, `cancel`) of a request in that step.

## 3. Evidence contract (relaxed)

- **Exact:** tokens, selections, Page vectors, carries and cursors.
- **Renamed:** block ids and incarnations, and identities (request ids, generations, bindings, nonces, lease ids)
  may differ by a consistent renaming.
- **Set aside:** the digests and MACs over renamed values (`lease_digest`, `receipt_digest`, `request_digest`, `mac`,
  `plan`) are not compared, and their consistency is not checked.
- **Comparator** (`test_async_sched.relaxed_differences`), within one native file:
  - each frame, field and KV group's block renaming is one to one, and so is the identity renaming;
  - released slots are compared as the same (group, kind) multiset with the same number of distinct blocks per
    group;
  - free counts and the other fields that depend on the interleaving (`kv_pools`, `http_measurement`,
    `admission_engine_ms`, `free_at_terminal`) are not compared.
- **The GPU harness's comparator** (perf-async-g1's `ag1checks.relaxed_differences`, outside this repository) is
  this one widened to two server processes: more identity keys (among them `server_run_id`, `process_start_id` and
  the carry and Page `digest` / `content` tokens), more digest keys, and timing fields set aside.
  - On the Qwen layouts it does not list the worker frames' `state_digest`, a per-process identity token, so it
    compares it by value. A difference outside the content fields is the partial finding NATIVE_RELAXED_DIFF.
  - That alone made job 3532746 PARTIAL: `worker_frames[].state_digest` and
    `worker_frames[].worker_receipt.body.state_digest` differed in 20 of the 37 b = 1 requests of fe, fg, te and tg,
    and in 3 of the 4 of se and xe.
  - Every one of those differences is a one-to-one renaming (0 conflicts in every pair). A local re-comparison with
    `state_digest` renamed like the other identity tokens gives ASG1_PASS.
  - This is a gap in the harness, not in the runtime. The fix is for the next re-pin of perf-async-g1: add
    `state_digest` to its identity keys for the Qwen layouts, with a test. (This page's comparator does not list
    `state_digest` either.)
- **Not implemented: H6 ("free counts as of the step boundary").** Free counts in native files and rows are read
  as they are, i.e. after the next step's schedule took its blocks. The G3 verifier must either correct them or
  set them aside.
- **H1 (exact rollback of speculative blocks):** not built, by the relaxed contract.

## 4. Deviations from DESIGN.md

1. **Token read.** The sampled-id read is a device read of `prev_sampled_token_ids` in the resolve, not a wait on
   vLLM's async copy event.
2. **Stop predicate.** It is vLLM's `check_stop` on a shadow request, plus an engine-side check that every
   predicted stop finished. A predicate read of an attribute the shadow does not advance would disagree; both
   directions fail closed.
3. **Held-update flush.** The worker's resolve calls the engine boundary on the same thread (UniProc), which RPCs
   back into the worker inside `execute_model`. GPU check.
4. **No drain before the lifecycle sweep** (section 2).
5. **H6 not implemented** (section 3).
6. **Commits.** Stages 1-7 are one runtime commit: they are interdependent and were tested together.
7. **Profiler.** The step profiler (`REMOTE_RA_STEP_PROFILE=1`) was not adapted; with async it is untested.
8. **Structured outputs.** vLLM's deferred structured-output sampling path is not refused explicitly; this runtime
   sends no structured-output requests to the main engine.

## 5. --thinking-history strip under async (the composition rule)

- **Sources.** The strip's positioned carry is:
  - `phase1`: the phase-1 state at e, held apart from the running slot; or
  - `split`: captured by the local turn 0 prefill forward that ends at X.
- **Why the extra row cannot reach it.** Both are taken before the request's first decode step and never from
  the running slot, so a stopped request's extra row cannot reach the carry. No Page past X is made.
- **The two options compose.** The void still applies: it keeps the slot and the KV at C' untouched, and stops
  the row making anything.
- **Snapshot source.** A third source, `snapshot` (option S), is not built and stays refused. Its lineage rule is
  `async_sched.SNAPSHOT_RULE`:
  - the position is the commit end;
  - it was captured from the request's own running slot by the forward whose lease ended there, before any later
    forward of the request was launched;
  - the slot was not voided, relocated or released in between;
  - the allocation generation and identity token are the same;
  - otherwise the session rebuilds.

## 6. What G1-G3 must measure

**G1** — strict, one GH200. Thinking off, plus one strip run. Async vs sync on the same node. Both eager and
`--cuda-graphs full`, and both FA3 and `--triton-attn`. Qwen with the XML 11-turn session (g1c-like) and a
chat-text session. b = 1, then b = 4 and 16.

1. **Per turn, against sync:**
   - tokens, selections, Page vectors and carry digests are byte-identical;
   - native files are equal under `relaxed_differences`.
2. **Counters:**
   - `link.check` failures: 0;
   - unpredicted stops: 0;
   - `voids` equals the stops vLLM scheduled past (EOS and marker; `max_tokens` stops are skipped by vLLM).
3. **The void on the device:**
   - a voided row leaves its GDN slot unchanged (the carry digest equals sync's);
   - KV at C' is untouched, i.e. the FA3 and TRITON_ATTN KV update and every GDN kernel path skip slot -1 and
     NULL_BLOCK_ID, also in FULL graphs' persistent buffers;
   - Mamba crossing stops (C' % 784 == 0) and Page-start stops (C' % 16 == 0) are covered.
4. **Held updates:** the flush from inside `execute_model` (reentrant RPC) works, with its frames and measured
   bytes.

**G2** — spd3's decode replay (2K, 20K, 117K, W2; b = 1-16; FA3 and Triton):

- ITL of our async against our sync, and against stock sync and async;
- reply parity within stock's own async-vs-sync spread;
- the resolve wait (the token read) and settle times per step;
- drains per 1,000 steps.

**G3** — a TB16 or SWE16 dry run (the benchmark flags plus `--async-scheduling`):

- admissions, Goal resumes, aborts and session closes under drains;
- `--release-idle-pages` and `--admission-overlap 16`;
- the counters (voids, resolves, drains);
- per-turn server time against sync;
- the free-count evidence (H6) read correctly.

**What G1 measured on Qwen** (perf-async-g1 g1 r3, job 3532746, `4ed1013`; its verdict is in
[docs/status.md](../status.md); the figures are from the job's offline verification, not audited):

- Every server answered every request (fea, fga, tea and tga 97 of 97, xea 12 of 12, sea 4 of 4), and every b = 1
  pair was token-exact (fe, fg, te and tg 37 of 37; xe and se 4 of 4). Its PARTIAL came only from the harness
  comparator's `state_digest` gap (section 3); a local re-comparison with it renamed gives ASG1_PASS.
- At b = 1 this showed item 1, the GDN void of item 3 (eager and FULL graphs, FA3 and TRITON_ATTN), the stop
  predicate and item 4's held updates sent from inside `execute_model`. The void probes stopped on a Page start
  (C' = 608) and on a Mamba crossing (C' = 1568) on every chat server.
- b > 1 is a measurement: reading B1 was met on every cell with a stock reference except te's b = 4 (7 of 12
  identical, against stock's 12 of 12).
- Not measured: item 2's counters. The EngineLink counters (voids, resolves, drains) are written to no evidence, so
  G1 relied on the b = 1 exactness of the turn after every stop. Not observed: item 3's KV-update skip of slot -1,
  because the next turn's phase 1 rewrites C' before any query reads it.

## 7. Open risks

- **GPU-only correctness:** the void writes, the kernels' skip semantics, the shadow `check_stop` against the
  real vLLM 0.24.0 request object, and the reentrant held-update RPC.
- **Smaller gain than estimated:** every admission drains one step's overlap, and the resolve waits for N's
  sampling before N+1's forward launches (inherent in C). DESIGN 4.1 estimated 5-11 % per step; nothing was
  measured at that date. The SWE-bench pair 3611917 / 3612333 (`4ebff97`) measured it since: the steady decode step
  per batch size 0.892 of the synchronous one ([docs/status.md](../status.md), not audited).
- **Fakes only:** vLLM's async scheduler internals (placeholders, the `max_tokens` skip, dropping a finished
  request's in-flight token) are modelled in fakes only. Until async-maxtok-1112 the fakes scheduled every running
  request in N+1 and left a removed request in `prev_req_id_to_index`, so they missed job 3568311's failure (section
  11).
- **H6 not implemented.**

## 7a. With --goal-overlap (async-int)

- **Why they compose.** `--goal-overlap` changes where a goal_query admission's work runs, never into a decode step:
  - phase 1 runs in the admission's pass (a registered add); the Goal is built on its Goal thread;
  - the continuation (the Goal join, the Goal forward as its own lease, phase 1's joined drain, `plan_turn`, the
    adoption and the reservation) runs in `EngineCoreProc._remote_ra_resume`'s pass.
  - Both passes are pass boundaries: the engine drains the batch in flight first (section 2). The worker also
    settles the issued step before any RPC other than the allocation updates, so the Goal forward's RPC and phase 1
    never meet a step in flight.
  - `--phase1-in-step`, the one option that puts admission work inside a step, stays refused with `--async-scheduling`.
- **Test** (`test_async_goal_overlap.py`): test_adm_overlap's three goal_query sessions (16-token Pages,
  `--admission-overlap 3`, the stand-in rewriter, Goals parked on their threads) with `--goal-overlap`, served
  synchronously (fast paths 1 and 2) and in vLLM's asynchronous order. Every turn's outputs, selection, carry, Page
  vectors, Goal, Goal hash and worker operations are equal; each native file is equal under the relaxed contract
  (clock fields and the goals.jsonl reference set aside: they hold times and request ids). Stops are voided, Goal
  resumes meet a batch in flight, Goal forwards run as their own leases.
- **Mutations.** The Goal forward skipped is refused (`Goal forward missing or foreign`); a missing void is caught.
  Without the engine's drain before a Goal resume a run is refused (`complete computed/sealed end mismatch`) or, on
  some interleavings, still equal: the drain is required, and its absence fails closed.
- **Launcher.** `--async-scheduling` with `--goal-overlap` (and `--admission-overlap`) on mswea-xml and chat-text,
  thinking off, keep and strip, is the async server's launch with the admission record added; the loader and
  `compose_engine` take it. The vLLM Goal rewriter's child keeps its own engine arguments: its record does not change
  with `--async-scheduling`.
- **Not covered.** No end-to-end run of the chat-text servers (strip or keep) or of the HTTP mswea-xml servers in the
  asynchronous order: their engine loops have no asynchronous fake (as for `--async-scheduling` alone). Keep mode
  changes the transcript layer only; the strip carry's sources are section 5's.

## 7b. H6 (free counts as of the step boundary): needed for evidence only

- **Not a correctness input.** Every decision that reads free counts (the capacity gate, `--release-idle-pages`, the
  admission ledger) runs at a pass boundary, after the drain, when no scheduled step holds blocks: the counts are the
  synchronous server's. The end-to-end tests compare every decision with free counts set aside and find them equal.
- **What differs.** A native file's `kv_pools`, `free_at_terminal` and the HTTP measurement's free counts are read at
  the completion, inside `update_from_output(N)`, after `schedule(N+1)` took its blocks. They count N+1's new blocks
  as used.
- **Design (not built).** At the resolve, record per pool the blocks the held updates of N+1 newly allocate (the
  entries `EngineLink.flush` sends, minus the voided requests'); a completion during `update_from_output(N)` adds them
  back to the free counts it writes, and records the correction as `free_correction` so a verifier can undo it. Size:
  about 60-100 lines in `async_sched` and `deferred_completion` / `evidence`, plus a test that the corrected counts of
  the asynchronous run equal the synchronous run's.
- **Until then** a verifier sets the free counts aside (`perf-async-g1` does).

## 8. async-fix: deferred block frees on the decoupled layouts (GPU job 3499729)

- **What failed.** In perf-async-g1 (job 3499729, runtime `f162120`) every runtime server with `--async-scheduling`
  (fea, fga, tea, xea, sea, tga) died while EngineCore constructed the Scheduler; the synchronous servers and stock
  vLLM's async server ran. `vllm/v1/core/sched/scheduler.py` raised `decoupled KV layout does not support deferred
  block frees (overlapping batches with a consumer KV connector)`.
- **Why.** vLLM sets `defer_block_free` when batches overlap (`max_concurrent_batches > 1`: async scheduling) and the
  KV connector is a consumer (`is_kv_consumer`: our OffloadingConnector is `kv_both`), so a freed request's blocks
  return to the pool only after the step that may still write them was processed (`processed_step_seq >= fence`).
  The decoupled layout's work (exp-decouple-pages, DESIGN.md item 4) refused that combination because a deferred
  entry held the blocks of every group in one list and `_drain_deferred_frees` returned them all to pool 0
  (`kv_cache_manager.block_pool`), and it was off in this runtime then. async-c turned it on; its CPU tests used fake
  engines that never constructed the real Scheduler (and test_kv_pools asserted the refusal).
- **Fix.** The refusal is gone. On a decoupled layout (`kv_cache_config.group_pool`, the Qwen FA / Mamba pools and the
  Gemma window layout's global / sliding pools) a deferred entry holds (block pool, block) pairs in
  `pop_blocks_for_free`'s order (`KVCacheManager.pop_blocks_for_free_by_pool`, coordinator), and the drain returns
  each block to its own pool behind the same fence, in the shared pool's order: the entry reversed, one `free_blocks`
  call per pool. The shared pool's path is unchanged. Requests the runtime's adapter owns (every generation request)
  still free at once (`_free_request_blocks`): their fence is the authenticated terminal drain, and the connector is
  idle in this runtime (no load can race a write).
- **Tests** (`test_deferred_free_pools.py`): the real overlay `Scheduler.__init__` executed from its source (the KV
  cache modules the overlay's own, the rest of vLLM stubbed): construction with `defer_block_free` on the Qwen
  decoupled layout, the Gemma window layout and the shared pool; the constructor path of every server of
  perf-async-g1's plan (serve.prepare's launch, the overlay planner's KV cache config, two batches with the async
  servers, the connector's role); ea18256's scheduler refusing it (from Git); deferred frees returning nothing before
  their fence, every block to its own pool exactly once and in the shared pool's order, FIFO fences, the same pools
  as an immediate free, owned requests bypassing; the shared pool byte-identical to ea18256's frozen methods.
  test_kv_pools and test_gemma_kv_pools follow the change.
- **Other construction-time guards on the async path** (read, not run on a GPU): vLLM's own config checks (UniProc
  supports async scheduling; passed on the GPU), the batch queue of two (`EngineCore`, `startup_runtime`
  `StartupTicket.quiescent`, `async_sched.attach_engine`), the startup gates that require async exactly when pinned
  (`bootstrap_contract`, `policy_runtime/bootstrap`, `native_worker_registration`, `positions.runtime_mode`), the
  worker's `async_sched.attach_worker` (`runner.use_async_scheduling`), and fast path 2's schedule wrapper. None
  refuses the decoupled layout. One more refuses async outright: the step profiler (`REMOTE_RA_STEP_PROFILE=1`,
  `remote_ra_step_profile.py`: "needs the plain EngineCoreProc loop"), so a GPU check must not set it.

## 9. async-gemma: the Gemma 4 profile, and drains without a worker RPC

**The Gemma 4 profile** ([GEMMA.md](GEMMA.md) holds the design). `serve.py prepare_gemma4` and
`async_sched.check_config` take `--async-scheduling` with Qwen's pins and refusals (`--decode-fastpath2` required;
`--gdn-device-handoff` and `--phase1-in-step` refused, which the profile refuses anyway); `window_worker` checks the
worker record once, attaches `async_sched.attach_worker` after the fast paths and before the generation guard, and
echoes the record. The window layout has no Gated-DeltaNet state: `void_rows(..., sliding_groups)` refuses a GDN state
index there and writes the voided token's slot -1 in every layer's metadata and forward-context mapping and in the
mapping of the global group and each sliding-window group (`owner.async_sliding`, the window owner's sliding groups,
recorded by `attach_worker`). Nothing else in the window layout changes: a stopped request's speculative allocation of
step N+1 (a fresh block for C' when C' % P == 0, the window's front block freed one step early when
(C' - W + 1) % P == 0) lies outside `kept_blocks(C')`, the window the commit pins and the worker's carry names (from
the rows of step N, as its held update is never sent), and inside the request's window slot and reply reservation.

**A claim held by a voided request (fixed on both profiles).** vLLM 0.24's `BlockPool.free_blocks` puts a freed
unhashed block at the head of the free queue (`prepend_n`; this runtime's allocations never cache, so none is
hashed). When a stopped request's speculative allocation of step N+1 frees one of its blocks (on Gemma a sliding
window's front block, one stop in 16; on Qwen a Mamba block in align mode, rare), the next request allocated in
the same schedule takes it at once. The stopped request is voided in N+1, so the worker does not observe it and its
claims stay until its completion later in that iteration: the other request's claim met a foreign one and the step
failed (`allocation owned by another generation`). `RunnerAllocationOwner._claim_full` now moves a claim whose
holder's row of this step is voided (`_voided_holder`) and refuses every other foreign claim as before. The voided
request never touches the block again (its row writes nothing; its completion reads only what vLLM still holds for
it). Test: `test_async_gemma.test_a_freed_window_block_taken_in_the_same_schedule` (deterministic; refused without
the move); a CPU run with Goal threads had met it at random.

**Drains without a worker RPC (fixed on both profiles).** A pass boundary drains the batch in flight: the engine
updates step N (its stops finish, its tokens are appended). The worker's issued step N is settled by the next worker
RPC other than the allocation updates, which an admission or a completion always makes; a CAPACITY_WAIT refusal, a
utility request, a wakeup of the engine loop or a cancel of another request make none. Then the resolve of step N+1
met N still issued and decided its stops again on requests that already held N's token: the token was counted twice,
a continuing request with one token left was predicted to stop, its row of N+1 was voided and `EngineLink.check`
failed the step (`a predicted stop did not stop`), i.e. the engine stopped, closed but not silently wrong. Under GPU
pressure every retry of a refused admission drains, and with `--goal-overlap` every Goal thread's wakeup does.
`EngineLink.applied(rid, cursor)` now tells the two cases apart: before N's update vLLM's request holds the tokens
up to the worker's cursor (`num_tokens == cursor`), after it one more (`cursor + 1`); applied tokens are not decided
again (their stopped requests finished before N+1 was scheduled), any other count fails closed. Tests: the Gemma
pressure run with idle releases at the selective pool (`test_async_gemma`, which found it) and its mutation; on Qwen
`test_async_sched.test_a_drain_with_no_worker_rpc_before_the_next_step` (a drain at every pass: every turn equal to
the synchronous server, no void needed; without the check the run fails closed).

**Tests** (`test_async_gemma.py`, CPU): the pin and launcher on mswea-xml, chat-text and kv-session (with
`--goal-overlap`, CUDA graphs, 4,096-token forwards), the loader, the frontend's parsed-flag check and the gates, the
refusals, the window worker's record, attachment order and echo; the void on window metadata (eager and persistent
buffers) and its refusals; `EngineLink.applied`; end to end on the window engine and worker fakes in vLLM's
asynchronous order against the synchronous server with the same fast paths: mswea-xml at b = 1 (both speculative
window steps covered) and b > 1, idle releases under pressure, `--goal-overlap` with `--admission-overlap` 4 and 1,
kv-session (MemAgent lanes, Gemma's stop ids), and chat-text through the API server, whose engine loop runs
`core.py`'s own input loop and drains (a length stop that vLLM does not schedule past included). Mutations: a missing
void, a sliding group left out of it, a missed stop, a wrong stop, the applied check removed. On the window layout the
relaxed comparator also renames the digests over block ids (the window carry's `digest`, its echo `boundary`, phase
1's `state_digest`) and names released slots by their (block, incarnation) pair: a sliding block that left a request's
window can come back to it under a new incarnation, depending on the free queues' order (H1).

## 10. async-fix2: the requests a phase-1 pass takes out of the runner's batch (GPU job 3519531)

- **What failed.** In perf-async-g1 g1 r2 (job 3519531, runtime `4a498dc`) every synchronous server passed; the async
  servers fea, fga, tea and tga died after 41 of 97 requests, when the b4 cell's second turn began, and xea after 8 of
  12, when x4's did; sea (b = 1 only) and stock vLLM's async server passed. Every b = 1 request equals sync (37 of 37
  for fe, fg, te and tg; 4 of 4 for se and xe); of the b = 4 requests compared, fe and fg had 3 of 4 identical, te,
  tg and xe's x4 4 of 4 (b > 1 is a measurement, not an exactness check). EngineCore's step raised KeyError in
  vLLM's GPU runner (`_update_states`, `gpu_model_runner.py:1426`) for the cached request of the step, which had one
  output token and was not in the persistent batch.
- **Why.** An admission's phase 1 runs as its own pass (`native_probe.NativeQueryForward._execute`): the runner
  executes a SchedulerOutput that schedules only the probe, and `_update_states` takes every request it does not
  schedule out of the persistent batch (`gpu_model_runner.py:1211-1217`); the probe's cleanup (`_cleanup_request`, an
  empty output) does the same. vLLM's scheduler still counts those requests as scheduled in the previous step
  (`prev_step_scheduled_req_ids`, `scheduler.py:1202-1204`), so its next step sends no `all_token_ids` for them
  (`scheduler.py:1439-1440`). Synchronously the runner adds such a request back from its own cached output ids, the
  real tokens. Under async scheduling those ids are -1 placeholders (`_bookkeeping_sync`), so a request with output
  tokens is rebuilt from `all_token_ids` alone (`gpu_model_runner.py:1423-1427`): KeyError. At b = 1 nothing runs
  during an admission and a session's first turn has no phase 1; at b4's second turn the first continuing turn is
  scheduled and samples, and the next admission's phase 1 (after the drain) takes it out of the batch.
- **Fix.** With `--async-scheduling` the probe driver runs its forward and its cleanup through
  `NativeQueryForward._runner_pass`, which reports the requests the pass took out of the batch to the engine link;
  `EngineLink.unbatched` drops them from `prev_step_scheduled_req_ids` (as vLLM's `reset_prefix_cache` clears that set
  when it flushes the batch), so the next step sends their token ids, the scheduler's own. It requires that no batch is
  in flight (`core.batch_queue` empty: the pass boundary's drain updated every scheduled step, so no output placeholder
  is left); otherwise the ids would miss a token, and it fails closed. Without the switch nothing changes (no report,
  no check). The batch is not kept across the pass: vLLM builds a step's inputs from every row of the persistent batch,
  the probe must run alone (`probe mixed with generation batch`), and removing and adding back is vLLM's own path (a
  request not scheduled in a step, preemption, `reset_prefix_cache`), the one the synchronous servers ran token-exact.
  It covers both profiles (the Gemma window worker uses the same driver), phase 1, its chunked forwards, the Goal
  forward (`--goal-overlap`) and their drains.
- **Tests** (`test_async_probe_batch.py`, CPU, no torch): the real overlay `_update_states` and `CachedRequestState`,
  the real Scheduler (two batches, the Qwen decoupled layout) and its `_make_cached_request_data`, the real
  `NativeQueryForward` and `EngineLink`, in `step_with_batch_queue`'s order on the GPU's sequence (step N schedules A
  and samples; B's admission drains and runs its phase 1; step N+1 schedules B new and A cached): A's token ids are
  sent and every input id the runner feeds equals the sampled tokens; 4a498dc's native_probe (from Git) and the report
  removed raise the GPU's KeyError at line 1426; refused with a batch in flight; synchronous scheduling unchanged. The
  asynchronous end-to-end fakes (`test_async_sched`, `test_async_gemma`, `test_async_goal_overlap`) now model
  `all_token_ids` (`test_async_sched.cached_token_ids`, `require_token_ids`): their phase-1 passes take running
  requests out, and without the report the Qwen and window harnesses raise the KeyError. The asynchronous window
  harness's completion leaves the runner's finished row to the next step's `finished_req_ids`, as vLLM's scheduler
  sends it: the synchronous fakes' shortcut (an empty out-of-band update) also took every running request out of the
  batch, which no production path does but a phase-1 pass.
- **GPU check:** perf-async-g1 g1 r3 (Qwen) ran as job 3532746: ASG1_PARTIAL_PASS_VERIFIED, every server passed and
  every b = 1 pair was token-exact; the PARTIAL came only from the harness comparator's `state_digest` gap (section
  3), and a local re-comparison gives ASG1_PASS (section 6). The Gemma 4 checks (jobs 3549350 and 3560784) and their
  verdicts are in [docs/status.md](../status.md).

## 11. async-maxtok-1112: the requests step N+1 does not schedule (GPU job 3568311)

Branch async-maxtok-1112 `13ee618` on integrate-1111 `4bf2c40`, merged into integrate-1112. CPU-tested only up to
2026-11-12; since then its case has run on the GPU (job 3604146, the last item below).

- **What failed.** perf-mab smqt (job 3568311, runtime `ba4d1ba`: Qwen3.6-27B, thinking on, `--async-scheduling`,
  MemoryAgentBench kv-session; the servers smqot, smqrt and smqgt back to back) lost each of its three servers at the
  first reply cut at `max_tokens` (4,096) while another request decoded. `before_forward(N+1)` -> `resolve` raised
  `LiveV2Error: async resolve: a request sampled without its token` (`async_sched.py:525`, `contract.py:23`). With
  thinking on, a reasoning loop runs a reply to its bound.
- **Why.** The resolve collected every issued request of step N that was `prompt_complete` and required a sampled
  token for each. vLLM 0.24.0's async scheduler does not schedule a running request in N+1 when its token of N is its
  `max_tokens`-th (`scheduler.py:490-504`) or reaches `max_model_len` (527-534, 562-578); a token budget or a
  preemption leaves a request out too. `_update_states(N+1)` then takes it out of the persistent batch
  (`gpu_model_runner.py:1197-1217`), and `InputBatch.remove_request` pops it from `prev_req_id_to_index`
  (`gpu_input_batch.py:556-557`), so `sampled_tokens` found only the other requests. A request that reaches
  `max_tokens` alone is safe: N+1 schedules nothing, and the guard returns before any resolve.
- **Fix.** The resolve's `sampled` keeps only the issued requests that N+1 schedules
  (`scheduler_output.num_scheduled_tokens`). A request of N outside N+1 has no row to void; `update_from_output(N)`
  decides its stop, as on the synchronous server.
  - `settle()` still closes every lease of N.
  - `schedule(N+1)` allocated nothing for it, so the held updates (`flush`) hold no entry of it.
  - It is not entered, so its completion meets no step lease (`WorkerBridge.terminal`).
  - The predicted stops stay a subset of N's scheduled requests (`EngineLink.check`).
  - A preemption aborts a tagged request (`scheduler._preempt_request`), whose token vLLM discards: predicting a stop
    from it would have been wrong.
  - The check keeps its force for the requests N+1 schedules: one of them without its token still fails closed.
  - Docstrings: the module's step 3.2, `sampled_tokens`, the window layout's speculative allocation. Release pins
    regenerated (only `async_sched.py` changed).
- **Tests** (CPU):
  - `test_async_sched.ResolveTests`: B skipped at N+1 (`max_tokens`, `max_model_len`, or no stop) and popped from
    `prev_req_id_to_index`: no error, nothing voided or predicted, only A entered, N's leases closed for both; A's EOS
    next to a skipped B is still voided; a request N+1 schedules without its token still fails closed.
  - The async harnesses now follow vLLM: `test_async_sched.async_harness` and `test_async_gemma.async_harness` apply
    the `max_tokens` skip and, with nothing scheduled, vLLM's early return (an empty step);
    `test_stageb_worker.FakeRunner._remove` and `test_async_gemma.AsyncRunner._remove` pop `prev_req_id_to_index`
    as `remove_request` does. Turns stop alternately on an EOS (scheduled once more, voided) and on `max_tokens`, so
    voids stay above 0; the multi-session runs assert a `max_tokens` skip next to a scheduled request.
  - On `4bf2c40`'s `async_sched.py` these tests fail with the GPU's `LiveV2Error` (14 errors and 1 failure in
    `test_async_sched`, `test_async_gemma` and `test_async_goal_overlap`); with the fix they pass.
- **Both profiles.** The resolve is shared: the Gemma 4 window layout takes the same path ([GEMMA.md](GEMMA.md)
  section 1).
- **GPU check:** the direct check, perf-mab smqt rerun on integrate-1112 `4ebff97` (job 3584105,
  `PMAB_PASS_VERIFIED`: `observation_query`, `random_fill` and `goal_query`), passed but did not reach this state:
  each server's one reply cut at `max_tokens` ended with no other request in flight. Since then (2026-11-15):
  perf-mab plan smqc (job 3604146, `4ebff97`, `PMAB_PASS_VERIFIED`) forces it. Its one server, `observation_query`
  with smqt's settings and four MemoryAgentBench cases, also runs a cut cell: a carrier request decodes a long reply
  while 8 probes, one at a time, send two turns each, cut at their `max_tokens` (24 to 136 tokens, then 32).
  Its async-cut check (at least 12 of 16) found this state on 16 of the 16 probe turns, inferred from the request
  records (the runtime has no counter for it); every request of the job was answered, with no engine error. The
  other `--async-scheduling` release smokes (perf-swe32 smoket, perf-tb16-strip smokeqt, perf-gemma-swe32 smokea,
  perf-mab smga) run the same resolve and reach this state whenever a reply hits `max_tokens` while another request
  decodes. Of these, smoket, smokeqt and smokea were cancelled on 2026-11-16 and did not run; smga ran (job 3601174,
  [docs/status.md](../status.md)).
