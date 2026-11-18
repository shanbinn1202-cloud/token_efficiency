# --async-scheduling on the Gemma 4 31B IT profile (option C, the window layout)

English | [简体中文](GEMMA.zh-CN.md)

Date: 2026-11-03. Branch `async-gemma` on `integrate-1103` (`ea18256`: async-c, the async x `--goal-overlap`
composition, the speed defaults, the Gemma profile with `--goal-overlap`). Paths are relative to `source-overlay/`
unless they start with a top-level file. Line numbers are those of `ea18256` unless a file is named as changed here.
Its GPU runs and their verdicts are in [docs/status.md](../status.md) and [History](../history.md#history); the limit
that holds (a partial pass) is [docs/known-limits.md](../known-limits.md) 5.3; what the GPU checks measured is in
sections 8 and 9. Read with [DESIGN.md](DESIGN.md) (the hazards D1-D5, H1-H7 and option C) and
[IMPLEMENTATION.md](IMPLEMENTATION.md) (what async-c built for Qwen; its section 11, async-maxtok-1112 `13ee618`,
has the shared resolve of section 1, item 4, which the window layout takes too).

## 0. Summary

1. **Why Gemma was refused.** async-c built option C for Qwen and refused the Gemma 4 profile in three places
   (`serve.py` `prepare`, `remote_ra_live_v2/async_sched.check_config`, and implicitly the window worker, which never
   read the record) because "the window layout's owner, rows and fast path 3 journal were not adapted"
   (IMPLEMENTATION.md section 1). One piece of code really was Qwen-only: the void writer required a Gated-DeltaNet
   state index (`async_sched.void_rows`: `require(states > 0 ...)`), which the window layout does not have, so every
   Gemma stop would have failed closed; of the per-group slot mappings it named only the full-attention groups.
2. **What is Gemma-specific** (section 3): no recurrent state, so the only device hazard is KV at C' (D3/D5), in six
   KV groups (the global group in pool 0, five sliding-window groups in pool 1); the speculative allocation of a
   stopped request moves its *window* (a front block freed one step early, a fresh block per sliding group), which
   must stay outside the window the commit pins; the engine follows each window incrementally (fast paths 2 and 3);
   the `--goal-overlap` park holds a net window across steps; chunked forwards of 4,096 or 8,192 tokens.
3. **Verdict: safe to enable, with two Gemma-specific runtime changes beyond the gates.** Every Gemma-specific path
   is either a pass boundary (drained first, as for Qwen) or provably unaffected by the extra row (section 3) once:
   (a) the void on the window layout writes the token's slot -1 in the global group and in every sliding group, and
   refuses a GDN state index there; (b) a worker claim held by a voided request moves to the request that vLLM gave
   the block to in the same schedule (a window's front block freed by the speculative step goes to the head of
   vLLM's free queue; section 3.1). Without (b) such a step failed closed (`allocation owned by another
   generation`): one stop in 16 frees a front block, and at 16 decoding sessions about one of those in three meets a
   request scheduled after it that takes a new sliding block in that step (one in 16 per request, about 7.5 of them
   after it), more during a prefill.
4. **Refusals kept** (section 5): without `--decode-fastpath2`, with `--gdn-device-handoff` or `--phase1-in-step`
   (neither applies to Gemma anyway: refused twice, by the profile and by async), with `--thinking-history strip`
   (gemma-thinking: the keep mode of `--enable-thinking` composes). Not a speed default (integrate-1103).
5. **Exactness contract: Qwen's** (section 4): at b = 1 the asynchronous server gives the synchronous server's tokens
   and evidence under the relaxed contract; on Gemma the window carry's block ids, incarnations and digest are part of
   the renaming. At b > 1: structural checks plus a measurement, as in perf-async-g1's verdict rule.
6. **A shared bug, fixed for both profiles** (section 6; IMPLEMENTATION.md section 9): after a pass boundary that
   drained the step in flight and made no worker RPC (a CAPACITY_WAIT refusal, a utility request, a wakeup), the next
   resolve decided that step's stops a second time, counted a token twice and stopped the engine on a lost step. The
   Gemma pressure test found it; the resolve now skips tokens the engine already applied (`EngineLink.applied`).

## 1. What asynchronous scheduling changes in vLLM 0.24's loop

`EngineCore.step_with_batch_queue` (`vllm/v1/engine/core.py:536-652`) with `max_concurrent_batches == 2`:

1. `schedule(N+1)` runs while forward N is on the GPU. For a request that sampled in step N the scheduler does not
   know the token yet: it counts one **output placeholder** (`num_output_placeholders`) and has already advanced
   `num_computed_tokens` at `_update_after_schedule(N)`. So step N+1 allocates for position C' + 1 tokens of a request
   that may have stopped at N.
2. `execute_model(N+1)` (UniProc: inline, non-blocking) prepares the inputs, metadata and slot mappings; the input ids
   of decoding requests come from the device (`prev_sampled_token_ids`). Our `before_forward` hook runs here
   (`gpu_model_runner.py:4605-4610`), after vLLM's builders filled the step's metadata (and, under FULL graphs, the
   persistent buffers).
3. Only then is step N's output popped and `update_from_output(N)` run (stops decided, completions, frees).
4. vLLM itself skips the extra step when the stop is certain: `max_tokens` (`vllm/v1/core/sched/scheduler.py:489-503`).
   An EOS, a stop id, the XML action marker or the context limit are known only after N's sampling.
   A request that `schedule(N+1)` leaves out (its `max_tokens`-th token, a token that reaches `max_model_len`,
   `scheduler.py:527-534` and `562-578`; or the token budget, a preemption) leaves the runner's batch in
   `_update_states(N+1)`, and `InputBatch.remove_request` pops it from `prev_req_id_to_index`: N+1 has no token of
   N for it (async-maxtok-1112, IMPLEMENTATION.md section 11).
5. A finished request's token of N+1 is dropped by vLLM (`scheduler.py:1723-1732`).
6. **Sliding windows.** `KVCacheManager.allocate_slots` frees the blocks a sliding group no longer reads *at
   schedule time*, from `num_computed_tokens` (`vllm/v1/core/kv_cache_manager.py:495-499`,
   `single_type_kv_cache_manager.py:479-511`; `SlidingWindowManager.get_num_skipped_tokens(n) = n - W + 1`, line 770).
   Under async that count already includes the in-flight step: a window's front moves one step ahead of the GPU, for
   every request (stock vLLM behaviour; the stream orders the in-flight step's reads before the next step's writes).

Option C (DESIGN.md 4.3, built by async-c): our `before_forward(N+1)` resolves N first (settle N's after-forward work,
read once the sampled ids of the requests of N that N+1 schedules, decide their stops with vLLM's `check_stop` on a shadow request plus `MarkerStop`, void the
stopped rows of N+1 on the device, send N+1's held allocation updates without the stopped requests' entries, enter the
continuing requests), and every pass boundary drains the batch in flight first.

## 2. Our hooks that assume synchronous steps (both profiles)

| Hook | Assumes | async-c (Qwen) | Window layout (Gemma) |
| --- | --- | --- | --- |
| Gates: `bootstrap_contract`, `policy_runtime/bootstrap`, `native_worker_registration`, `startup_runtime`, `positions.runtime_mode` | `max_concurrent_batches == 1`, no batch queue, `use_async_scheduling` false | asynchronous exactly when pinned (`async_sched.allowed`) | the same code; `positions.runtime_mode` already treats the window layout (no Mamba align mode) |
| Launcher, config loader (`serve.py`, `remote_ra_http/config.py`) | `--no-async-scheduling` | pinned key + `effective_server_config.async_scheduling` | was refused (`serve.py:497`, `check_config:79`); now the same pin (`prepare_gemma4`) |
| Engine allocation (`engine_boundary.dispatch_mixed_request`, `allocation_guard`, `chunk_protocol`) | the decode's fresh token is on the host | one placeholder on a RUNNING decode | the same code path (the window layout's allocation goes through it) |
| Held updates (`EngineRpcBoundary.flush_updates`, fast path 2) | sent when `schedule` returns | held until the resolve; a stopped request's entry dropped, its `native_allocation` / `native_tables` restored (H2) | the same; the engine's window follower (`window_rows.grow`) for that request advanced and is dropped at its completion (section 3.3) |
| Worker step (`WorkerBridge.before_forward` / `after_forward`, `finish_step`) | the step joined inside its own `execute_model` | issued, settled at the next resolve | `WindowLifecycle` inherits `issue_execute` / `leave_issued`; the window owner inherits `scheduled_rows`, which skips voided rows |
| Generation guard (`generation_guard.guarded_execute`) | a device drain before the step | a connector check, then `async_sched.guarded` | the same object (`window_worker` installs the same guard) |
| Void (`async_sched.void_rows`) | n/a | GDN state index `NULL_BLOCK_ID` + FA slot -1 | **changed**: no GDN; slot -1 in the global and every sliding group (section 3.2) |
| Completion (`allocation_owner._complete` cursor check, `WorkerBridge.terminal`) | the worker's cursor is C' | an unvoided stop fails closed | the same; `WindowWorkerBackend.drain` commits `final_window` from the worker's rows of step N |
| Pass boundaries (admission, Goal resume, client requests, close, shutdown) | no step in flight | `_remote_ra_async_drain` before each (`core.py:664-690`, `1430`, `1459`, `1479`, `1515`); the worker settles before any non-steady RPC | the same; Gemma's phase 1, Goal forward, trim, gate and ledger, idle release and window close all run there |
| Legacy (`remote_ra_growing_worker/plan.py`, `remote_ra_qwen_semantics/hooks.py`), `private_authority.Topology` | synchronous only | unchanged (not on the serving path) | unchanged |
| Step profiler (`REMOTE_RA_STEP_PROFILE=1`) | one step at a time | not adapted, untested with async | the same |

## 3. What is Gemma-specific

Geometry ([docs/reference.md section 4.14](../reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)): 10 global layers (one KV group, pool 0, P-token Pages, P = 16), 50 sliding layers (W = 1,024)
in five KV groups sharing pool 1. Pool 1 is reserved at startup: per sliding group S window slots of
`ceil(W / P) + 1` = 65 blocks, one forward's transient and, for `goal_query`, the Goal transient
(`kv_layout.WindowLayout.window_blocks` / `transient_blocks`, `window_pool_blocks`). Between turns a session pins
`kept_blocks(C')` = `[(C' - W + 1) // P, ceil(C' / P))` per sliding group (`sliding_window.py:56`,
`remote_ra_native_growing/window_pins.py:118`).

### 3.1 The speculative step of a stopped request R (stop at N, computed end C')

`schedule(N+1)` allocates for R as for a decode at position C' (num_computed_tokens = C'). R is a stop that N+1
still schedules (an EOS, a stop id, the output stop); a `max_tokens` or `max_model_len` stop allocates nothing
there (section 1, item 4):

| Event | When | Effect on the window | Why it is harmless |
| --- | --- | --- | --- |
| Front free | (C' - W + 1) % P == 0 (for W = 1,024, P = 16: (C' + 1) % 16 == 0), C' >= W | each sliding group frees block `(C' - W + 1) // P - 1` one step early | it lies before `kept_blocks(C').first`: the commit never pins it; the synchronous server frees it at R's completion |
| Fresh blocks | C' % P == 0 | one fresh global block (pool 0) and one fresh block per sliding group (pool 1) for position C' | index `C' // P = ceil(C' / P)` is past `kept_blocks(C').end`: never pinned, freed with R; the void writes nothing into them |
| Window follower (fast path 2/3, engine) | every step | `record['incremental']['windows']` advanced for R | the held entry is dropped; `complete_and_drain` pops `incremental` before anything reads it (`engine_boundary.py:442`) |
| Pool journal (fast path 3) | every allocation / free | R's speculative events are journaled | the journal only makes another request's window check *more* thorough (`window_rows._unchanged`: a named block is checked in full) |
| Worker claims | a front free, and another request allocating after R in the same `schedule(N+1)` | vLLM 0.24 puts a freed unhashed block at the **head** of the free queue (`block_pool.free_blocks`: `prepend_n`), so that request takes R's freed block at once, while R (voided, not observed this step) still holds the worker's claim on it until its completion later in the iteration | **changed**: the claim moves when its holder's row of this step is voided (`RunnerAllocationOwner._voided_holder`); before, the step failed (`allocation owned by another generation`). R never touches the block again (3.2), and R's completion reads only blocks vLLM still holds for it |

- **The commit is the synchronous server's.** The engine pins `kept_blocks(C')` from R's row during
  `update_from_output(N)` (`WindowPins.pin_after`): every index in `[first, end)` is still R's live block with one
  reference, exactly the blocks the synchronous server pins. The worker's carry (`WindowAllocationOwner.final_window`)
  is built from the worker's native rows, which are those of step N because R's update of N+1 is never sent
  (`EngineLink.flush(voided)`); the same indices, the same blocks.
- **Capacity.** R's speculative step is a decode at position C', i.e. one the synchronous server would run for a reply
  one token longer: in pool 1 it holds at most `ceil(W / P) + 1` blocks per group (R's own window slot; vLLM frees the
  front before it allocates), in pool 0 position C' < prompt + max_tokens (vLLM skips the step at a `max_tokens` stop)
  and < `max_model_len`. Both are inside what the gate and the sliding ledger already counted for R as a running
  request (`capacity.headroom_by_pool`, `window_capacity`), and both are freed in the same engine iteration
  (`update_from_output(N)`), before the next pass boundary. Every gate decision runs after a drain, so it sees the
  synchronous server's free counts (up to block identity).
- **Block identity (H1).** The early free and the fresh blocks change the order of pool 0's and pool 1's free queues, so
  later block ids and incarnation counts differ from the synchronous server's, at b = 1 too (one stop in 16 for
  each of the two cases above). The relaxed contract renames them (section 4).
- **Prefix cache.** Every generation allocation goes through `dispatch_mixed_request`, which sets `skip_cache=True`
  (`engine_boundary.py:213`), so `allocate_slots` never caches a block (`kv_cache_manager.py:561`): a block whose last
  position C' was counted as computed but voided is never offered as a prefix hit (perf-async-g1's open risk on the
  (C' + 1) % 16 == 0 stop holds for neither profile).

### 3.2 The void on the window layout

- **No recurrent state.** The extra row's only device writes are KV at C': the global group's tail Page and each
  sliding group's block `C' // P` (the last kept block, or the fresh block when C' % P == 0). Without a void these
  writes are benign (DESIGN.md D3, D5: the next turn's phase 1 starts at C = C' and writes C' before any query reads
  it). With the void the KV bytes are the synchronous server's, which the exactness contract wants.
- **What it writes.** On the window layout `void_rows` (changed) refuses a GDN state index, and writes slot -1 at the
  voided tokens in every layer's metadata `slot_mapping`, every per-layer mapping of the forward context and the
  mapping of each of the six KV groups (`owner.attention_groups` + `owner.async_sliding`, the window owner's sliding
  groups, set by `attach_worker`); each group's mapping must be present. vLLM's mappings are views of the block
  tables' persistent `slot_mapping.gpu` buffers (`gpu_model_runner._get_slot_mappings`, `4085-4157`), so the write
  reaches a FULL graph's replay; TRITON_ATTN's metadata and the KV-cache update (a splitting op on the window layout,
  `cuda_graphs.WINDOW_SPLITTING_OPS`) read the same tensors.
- **Slot -1 on TRITON_ATTN.** vLLM pads unused tokens with -1 for every backend under FULL graphs
  (`gpu_model_runner.py:4131`: "Needed for reshape_and_cache in full cuda graph mode"), so the Triton KV-update kernel
  skips -1 as the FA3 one does. Not in the overlay: a GPU check confirms it (section 8).
- The voided row still reads (attention over R's blocks, including the unwritten slot C'); its output is never used:
  `scheduled_rows` skips it (no lease, no Page vector, no finiteness verdict of its own), vLLM drops its token.

### 3.3 Engine and worker records (H2, H3, H4)

- Engine: R's held entry is dropped and its `native_allocation` / `native_tables` restored as for Qwen. The window
  follower and the global row's `fa`/`ids` of R's `incremental` record advance in `window_rows.grow` and are never
  restored: nothing reads them before `complete_and_drain` pops them. `WindowPins` state changes only at the commit and
  at pass boundaries.
- Worker: R's rows, claims, observation (fast paths 1-3: `_WindowObserved`, the steady `native3` rows), leases and
  `window_ids` stay those of step N. Continuing requests advance exactly as on the synchronous server: their deltas
  are computed against the engine follower and applied to the rows the worker last accepted, both one step per
  scheduled step.
- `WindowLifecycle` (a `QwenAlignLifecycle`) takes `issue_execute` / `leave_issued` unchanged; `before_forward3`
  (fast path 3) sees only the entered requests.

### 3.4 The phase-1 park (`--goal-overlap`, effa85e)

A `goal_query` admission runs phase 1 in its own pass, trims the session's old window at phase 1's drain
(`WindowPins.trim`), parks on its Goal thread holding its net window and fresh blocks in pool 1, and continues in the
Goal resume's pass (Goal join, the Goal forward as its own cold lease on the Goal transient, the adoption, the
reservation).

- Both passes are pass boundaries: the engine drains the batch in flight before the registered add and before the Goal
  resume (`core.py:1430`, `1515`), and the worker settles the issued step before the phase-1 and Goal-forward RPCs
  (`worker_dispatch`). Phase 1, the trim and the Goal forward never meet a step in flight.
- While it is parked, decode steps of other sessions run asynchronously. Their speculative allocations stay inside
  their own window slots and reply reservations (3.1), so the sliding ledger's invariant (free >= sum of kept + the
  largest transient + the running requests' headroom; goal-overlap DESIGN.md 9.3) holds between passes as it does
  synchronously. The parked admission's holding (`WindowPhase1Coordinator.held`) is not touched by a decode step.
- `--phase1-in-step`, the only option that would put admission work inside a step, is not available on Gemma and stays
  refused with `--async-scheduling`.

### 3.5 Chunked forwards (`--max-num-batched-tokens` 8,192 or 4,096)

- **Phase 1** in chunks (`window_phase1`, `phase1_chunks`, at most N - W - P tokens each) runs inside its admission's
  pass: drained, one lease, no step in flight.
- **Phase-2 prefill chunks** (a cold turn, a rebuild): vLLM schedules chunk k+1 while chunk k is in flight, without
  sampling in between. The sliding front moves at schedule time (section 1, item 6); the stream orders chunk k's reads
  before chunk k+1's writes. The worker reads a token only for requests whose prompt was complete at the issued step
  (`RequestState.prompt_complete`, vLLM's `prev_req_id_to_index` leaves partial prefills out) and that the next
  step schedules (async-maxtok-1112). The engine's
  `scheduled_cursor` of chunk k is advanced by the resolve of step k, which runs before `schedule(k+1)` checks chunk
  k+1's interval.
- **Token budget at b > 1.** A voided row takes one token of step N+1's `max_num_batched_tokens`, so a concurrent
  prefill chunk may end one token earlier than on the synchronous server: a different batch composition, which is why
  b > 1 is structural (section 4). It never exceeds the forward's transient: the voided row's blocks are R's own slot.

### 3.6 The decode fast paths on Gemma

`--async-scheduling` needs `--decode-fastpath2` (the held `allocation_batch`); with the speed defaults all three are
on. Fast path 1's incremental window observation, fast path 2's deltas and compact rows and fast path 3's steady
observation, journal and `before_forward3` run on the entered (non-voided) requests only, which are observed exactly
as on the synchronous server (3.3).

### 3.7 What needs no change, and why

`window_owner`, `window_lifecycle`, `window_rows`, `window_pins`, `window_phase1`, `window_capacity`, `capacity`,
`idle_release`, `goal_overlap`, `deferred_completion`: their per-step work runs on entered rows, and every other call
runs at a pass boundary or at a completion inside `update_from_output(N)`, after the resolve of N+1 (3.1, 3.3).
`--deferred-completion-io`: the window layout's completion work (the carry, the pin before vLLM's free, the check
after it) stays on the engine thread inside `update_from_output(N)`, as before; the pinned blocks' reference counts
are the synchronous server's there (R's speculative blocks are not in `kept_blocks(C')`).

## 4. Exactness contract (Qwen's)

The verdict rule of perf-async-g1 (the GPU harness; it and its README are in the maintainer's workspace, not in
this repository, so the rule is restated here), unchanged:

- **b = 1, exact.** Every request equals the synchronous server's: prompt ids, finish reason, completion count, content,
  the generated ids, the worker frames' operation sequence, and every content field of the native file and the runtime
  row under the relaxed contract: tokens, positions, sealed Pages, selections, Page vectors, Goal vectors; block ids,
  incarnations and identities up to a consistent renaming; free counts and timings set aside (H6 is not implemented).
  **Gemma adds:** the committed window is part of the content (the same groups, the same block indices
  `kept_blocks(C')`, the same positions in each slot); its block ids and incarnations are renamed and the digests over
  them are set aside like the other digests over renamed values: the window carry's (`window_lifecycle.window_digest`;
  `digest` of `drain.carry` and of a frame's `carry_restored`), its echo as the session `boundary`, and phase 1's
  window `state_digest` (`WindowWorkerBackend.capture_phase1_state`). Released slots are renamed as (block,
  incarnation) pairs: a sliding block that left a request's window can come back to the same request under a new
  incarnation, depending on the free queues' order (H1), so the number of distinct blocks per group is not invariant,
  the number of distinct slots is. The CPU tests also require that no KV is written at a position >= C' of a finished
  request (exactly-once computation of [C, C')).
- **b > 1, structural plus a measurement.** Batch composition differs (a voided row in the next step, the token budget
  of 3.5), so tokens are not required to match on a GPU. Required: every request answered; `length` exactly at the
  bound, `stop` within it on a stop id or the marker; C' = N + g - 1 in the row and the carry; the committed window is
  `kept_blocks(C')` in every sliding group, the engine pin equals the worker carry; frames from `prepare` to
  `complete`; every continuing turn extends its prompt by the reply's ids. Measured: identical share against the stock
  pair's async-vs-sync spread. (On the CPU fakes, which have no batch variance, b > 1 is also checked equal.)

## 5. Refusals kept

| Combination | Where | Message |
| --- | --- | --- |
| `--async-scheduling` without `--decode-fastpath2` | `serve.prepare_gemma4`, `async_sched.check_config`, `window_worker`, `async_sched.attach_worker` | `serve.prepare_gemma4`: `--async-scheduling requires --decode-fastpath2`; `check_config` and `attach_worker`: `--async-scheduling needs --decode-fastpath2`; `window_worker`: `async_scheduling needs decode_fastpath2` (with the speed defaults it is on; with `--no-speed-defaults` give `--decode-fastpath --decode-fastpath2`, as on Qwen) |
| with `--gdn-device-handoff` | `serve.prepare` (before the profile branch), `check_config`, `attach_worker`; the Gemma profile refuses it on its own | `--async-scheduling refuses --gdn-device-handoff and --phase1-in-step (v1)` |
| with `--phase1-in-step` | the same | the same |
| with `--thinking-history strip` (gemma-thinking; `--enable-thinking` alone composes) | `serve.prepare_gemma4`, `thinking_cut.check_config` | `--thinking-history strip on the Gemma 4 profile refuses --async-scheduling (v1)` |
| as a speed default | `serve.SPEED_SWITCHES` | not a default on either profile until a GPU check passes |

## 6. Implementation (async-gemma)

- `remote_ra_live_v2/async_sched.py`: `check_config` no longer refuses `model_profile`; `attach_worker` records the
  window owner's sliding groups (`owner.async_sliding`, None on the Qwen layouts) after checking that owner, lifecycle
  and backend agree on the layout; `void_rows(..., sliding_groups)` as in 3.2; `resolve` passes them.
- `remote_ra_live_v2/window_worker.py`: reads and checks the worker record (needs fast path 2), attaches
  `async_sched.attach_worker` after the fast paths and before the generation guard (as `attach_loaded_worker` does),
  echoes `async_scheduling` in the evidence (compose_engine compares it).
- `serve.py`: `prepare` passes `async_scheduling` to `prepare_gemma4`, which pins it as for Qwen (the config key,
  `effective_server_config.async_scheduling`, `--async-scheduling` in place of `--no-async-scheduling`; nothing else);
  help text and docstrings. `remote_ra_http/config.py`: docstring only.
- `remote_ra_live_v2/allocation_owner.py`: `_claim_full` lets a claim move from a request whose row of this step is
  voided (`_voided_holder`, reading the resolve's `runner._remote_ra_live_v2_voided`) and refuses every other foreign
  claim as before (section 3.1). Shared: on Qwen a Mamba block freed by a stopped request's speculative step (align
  mode) meets the same rule, but needs a Mamba crossing of both requests in one step (rare).
- **A shared bug, found by the Gemma pressure test and fixed for both profiles** (`async_sched.EngineLink.applied`,
  `resolve`): after a pass boundary that made no worker RPC, the resolve of N+1 decided N's stops a second time
  (section 0, item 6); it now decides only tokens the engine has not applied. The mechanism and its tests:
  IMPLEMENTATION.md section 9 ("Drains without a worker RPC").
- Release pins regenerated (333 sources, 242 payload).

## 7. CPU tests (`test_async_gemma.py`, 23 tests at `4a498dc`; 24 on main)

The 24th test on main (`tests/test_async_gemma.py`) came with async-fix2 (`4ed1013`):
`test_without_the_unbatched_report_a_phase1_pass_loses_the_token_ids`.

- `PinTests`, `LauncherTests`: the Gemma config check; `prepare` on a Gemma release with and without the switch on
  mswea-xml, chat-text and kv-session, with `--goal-overlap` (and `--admission-overlap`), CUDA graphs, fast path 3,
  `--deferred-completion-io`, `--release-idle-pages` and 4,096-token forwards: one config key, one effective key,
  `--async-scheduling` in place of `--no-async-scheduling`, and `async_scheduling=False` is the former launch; the
  loader, the frontend's parsed-flag check, the worker flag and the startup gates on that config; the refusals.
- `WindowWorkerAttachTests`: the window worker's record check, `attach_worker` after fast paths 2 and 3 and before the
  guard, the echo, the refusals; `attach_worker` on the real window owner, lifecycle and backend (the sliding groups
  recorded; a Qwen-shaped lifecycle next to the window owner refused).
- `WindowVoidTests`: the void on window metadata (eager and persistent buffers): the six groups, each mapping and
  forward-context copy, nothing else; async-c's void refuses the layout; the refusals.
- `AsyncWindowEndToEndTests` (mswea-xml, test_gemma_release's scheduler, in vLLM's asynchronous order against the
  synchronous server with the same fast paths): one session over six turns (fast paths 1-2 and 1-3; every stop voided;
  both speculative window steps happened and were checked on the rows), three sessions (`full`, `random_fill`,
  `goal_query`), four sessions with idle releases at the selective pool (CAPACITY_WAIT, evictions and readmissions),
  and the claim move made deterministic: a front-free stop while another session's first prefill chunks take the
  freed blocks in the same schedule (`reused`), equal to the synchronous server, refused without the move;
  mutations: a missing void, a sliding group left out of it, a missed stop, a wrong stop, the applied check removed.
- `AppliedTests`: `EngineLink.applied` on the counts.
- `AsyncGoalOverlapTests`: `--goal-overlap` with `--admission-overlap` 4 (fast paths 1-3) and 1: turns, Goal vectors,
  frame operations and native files equal; Goals parked while others decoded, Goal resumes met a batch in flight.
- `AsyncKvSessionTests`: MemAgent lanes at 1, 4 (idle releases) and 3 (`goal_query`) sessions.
- `AsyncChatTextTests`: the chat-text API server with Terminus clients, its engine loop the stub of test_ms2_e2e with
  `core.py`'s own input loop (its drains are the shipped ones) in vLLM's asynchronous order: one client (and a turn cut
  at its bound, which vLLM does not schedule past) and four clients decoding together.
- Elsewhere: `test_speed_defaults.async_problems` runs its checks on both profiles (a bare `--async-scheduling` on
  Gemma with the defaults is the launch of every switch given; with `--no-speed-defaults` it needs
  `--decode-fastpath --decode-fastpath2`; the refusals); `test_async_sched` (the pin test, and
  `test_a_drain_with_no_worker_rpc_before_the_next_step` for the shared fix on Qwen).
- The Scheduler's construction (async-fix, merged): GPU job 3499729 showed that the fakes above never build vLLM's
  real Scheduler, where every async Qwen server failed (`defer_block_free` refused on the decoupled layouts, the
  window layout's two pools included). `test_deferred_free_pools.GemmaServerKindTests` runs the overlay's
  `Scheduler.__init__` on the planner's KV cache config of every Gemma server kind (chat-text eager and with FULL
  graphs, mswea-xml, kv-session, 4,096-token forwards; each with and without `--async-scheduling`) and a deferred
  free on its pools.

## 8. What a GPU check must show (the g1g plan)

A Gemma G1 in perf-async-g1's form (pairs sync then async on one GH200, the same verdict rule), after Qwen's G1
has verified (it did in job 3532746, g1 r3; job 3499729, g1 r1, failed at the Scheduler's construction):

- void probes at b = 1 that stop on C' % 16 == 0 with C' > W (a fresh global and sliding block for C') and on
  (C' + 1) % 16 == 0 with C' >= W (the early front free), each with a continuing turn that reads the committed window
  and the global tail;
- the TRITON_ATTN KV update skipping slot -1 in the global and the sliding layers, eager and under FULL graphs
  (the six persistent slot mappings);
- XML marker stops (Gemma's grammar) and chat-text stop ids (0, 1, 50, 106);
- b = 4 and 16 structural, with stock Gemma's async-vs-sync spread as the reading; at b = 16 front-free stops meet
  other sessions' new sliding blocks in the same schedule (the claim move of 3.1), which no Qwen cell exercises;
- since this branch changes shared code (`EngineLink.applied`, the claim move), one Qwen pair on the same commit.

Pairs as planned in perf-async-g1's g1g (sync first, then async, same node): `ge` chat-text eager, `gg` chat-text
`--cuda-graphs full`, `gx` mswea-xml eager, then g1's Qwen pair `fe` (FA3 eager: the shared code on the same
commit) (required); `gst` stock vLLM 0.24.0 Gemma with TRITON_ATTN (optional, last). `gk` (kv-session eager) is not
planned: the harness has no kv-session client. Shared flags: `--kv-cache-gib 25 --page-tokens 16 --max-model-len
65536 --language-model-only --connector-pool-gib 1` and the three fast paths with `--deferred-completion-io`. The
plan's time fit: one ghx4 batch job of 02:00:00 on one GH200 (`--mem 113560M`), required worst case 7,093 s (107 s
spare), expected about 3,606 s; at most 2.0 GPU-hours, expected about 1.0. perf-async-g1 added what g1g needs on
2026-11-03: the Gemma pins and plan lengths, the window probes at C' = 1104 and 1167, the window layout's relaxed
contract (its digests set aside, released slots renamed), the READY checks and the stock Gemma pair.

What the GPU runs measured (from each job's offline verification, not audited):
- g1g r2 (job 3549350, runtime `4ed1013`): every server became READY, and at b = 1 every compared request of the pairs ge, gg and gx was
  token-exact; 12 of 97 requests per ge / gg server timed out in the harness's client.
- g1g r3 (job 3560784, runtime `4ed1013`): every server that ran answered every request, gg and gx were token-exact at b = 1, and ge's
  synchronous server was skipped for the job's time budget.
- Gemma's MemoryAgentBench smoke smga (job 3601174, `4ebff97`: `goal_query`, `observation_query`, `random_fill`;
  kv-session, thinking off) ran with `--async-scheduling`; whether it met the case of section 1, item 4 (a reply at
  `max_tokens` while another request decodes) is not known. On Qwen, job 3604146 met it 16 times without error
  (IMPLEMENTATION.md section 11).

## 9. Open risks

- GPU-only: TRITON_ATTN's slot -1 skip, the void in the persistent buffers, the shadow `check_stop` against vLLM's
  request object, the reentrant held-update RPC (all shared with Qwen except the sliding groups' mappings). On Qwen,
  g1 r3 (job 3532746, `4ed1013`) showed at b = 1 the void in FULL graphs' persistent buffers, the shadow
  `check_stop` and the held-update RPC (every b = 1 pair token-exact). b = 1 exactness cannot show the KV-update skip
  of slot -1 on either backend: the next turn's phase 1 rewrites C' before any query reads it.
- Gemma-specific, open on 2026-11-08 (before g1g r2 and r3): the void in the six KV groups and the claim move at
  b = 16. g1g r2 and r3 sent 48 requests at b = 16 to each Gemma chat-text server (their offline verification
  records, not audited), but section 8 records token exactness at b = 1 only, so this note does not count the two as
  checked.
- The probe passes between asynchronous steps (phase 1 and the Goal forward through `NativeQueryForward`, after a
  drain): the runner's `prev_sampled_token_ids` bookkeeping across a probe-only forward is vLLM's; Qwen's G1 exercises
  it first (shared). It did: in g1 r2 (job 3519531, `4a498dc`) vLLM's GPU runner raised a KeyError after a phase-1
  pass (`gpu_model_runner.py:1426`); async-fix2 (`4ed1013`) fixed it (IMPLEMENTATION.md section 10).
- H6 not implemented (free counts set aside), the step profiler not adapted, the EngineLink counters not written to the
  evidence (shared).
- The CPU fakes model vLLM's scheduler, runner and kernels; they do not run them.
