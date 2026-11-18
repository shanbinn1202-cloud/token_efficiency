# Goal overlap: phase 1 while the Goal is built (design, 2026-10-30)

English | [简体中文](DESIGN.zh-CN.md)

Design of 2026-10-30 (sections 1-8 at `cf2b28d`, section 9 at `5c8bc24`); the implementation and the Gemma follow-up are on `main`, and `--goal-overlap` is opt-in (default off). It is part of the stable configuration (with the vLLM Goal engine, job 3576565). Its GPU runs and their verdicts are in [docs/status.md](../status.md) and [History](../history.md#history), the limits of the Goal path in [docs/known-limits.md](../known-limits.md) section 3; what the GPU checks of section 7 measured is in section 8.

Branch `goal-overlap`, cut at `cf2b28d`. SO = `source-overlay/`. Every statement about today's code was checked in the sources of `cf2b28d`; line numbers are those of that commit.

Symbols (the turn geometry of `remote_ra_native_growing/contract.py` `turn_geometry`): P is the Page size; C the session's computed end, where phase 1 starts; o the observation start (the stable end); e the observation end, so phase 1 computes `T[C:e)`; N the prompt length; Q = floor(C/P)·P the sealed end; R the resident Pages; n = floor(o/P) the candidate Pages, L = n·P. G is the number of Goal tokens. A and M are the numbers of attention and Mamba groups (`capacity.py`; this M is not the Mamba block of docs/exp-decouple-pages). K is the `--admission-overlap` value.

## 0. Summary

**Question.** A `goal_query` admission builds its Goal (the Qwen3-0.6B rewriter on the turns that call it: seconds), and only then runs phase 1 (the prefill of the turn's new tokens and the Page and observation vectors, about 150-250 ms per turn on the 10-26 Terminal-Bench run). The Goal query vector is computed inside the phase-1 forward pass. Can phase 1 run while the Goal is built, with the Goal vector computed afterwards, and give the same vector?

**Verdict: yes, exactly.** The Goal forward is already a separate, cold forward. It runs in the same worker RPC as phase 1 but after it, on its own blocks, from position 0, without the history KV, the turn's new tokens or the GDN carry. It reads nothing that phase 1 computes or changes (section 2). Run later in its own probe-only pass, it is the same `execute_model` call: the same tokens, the same positions, the same empty context, the same one-request eager batch and the same row widths. Only the block ids differ, and they change where the KV is stored, not the values. The Goal vector is therefore bit-identical, up to the run-to-run nondeterminism of the GPU kernels, which affects today's runs in the same way (job 3349078 repeated a server and got identical phase-1 vectors for all 2,866 Pages). The Page scores (cosine against that vector), the ranking and the selection are then identical too.

**Design.** Behind a new opt-in flag `--goal-overlap` (default off), on a server that parks `goal_query` admissions:
1. the admission starts its Goal on its Goal thread and does not park;
2. it runs phase 1 without the Goal forward (in place, or inside a decode step with `--phase1-in-step`);
3. it then joins the Goal (it parks only if the Goal is not built yet);
4. when the turn needs a Goal forward (a cache miss on an active turn), it runs it as its own probe-only pass with its own lease, whose reply carries its drain (no per-request hashing beyond today's: 3.11);
5. then come the selection (`plan_turn`, unchanged), the adoption and the reservation.

Goal construction (its input, the constructor state, the rewriter call) is untouched: the TurnContext is built at the same point as today, and the construction starts at that same point.

**Gain.** Per turn whose Goal takes time: about min(Goal time, phase-1 time) less admission latency. That is about 0.15-0.25 s per rewriter turn on the 10-26 run (a few percent of those turns' admission), and up to about half of the Goal-plus-phase-1 critical path once a faster rewriter makes the Goal as fast as phase 1 (the vLLM rewriter of `rewriter-vllm` is not that fast: section 4). With `--phase1-in-step`, the phase 1 of a turn that needs a Goal forward can also join a decode step (today it stays a separate pass), so only the short Goal forward still stalls the decodes.

**Implementation and GPU checks.** Section 8. The Gemma 4 profile (window layout): section 9.

## 1. Today (verified)

### 1.1 The order of an admission

`SplitPrivateEndpoint.reserve_steps` (SO/remote_ra_live_v2/private_endpoint.py:221-283):
1. `geo = turn_geometry(...)`;
2. `query = yield from selection.begin_turn_steps(binding, geo, conversation)` (237-240): builds the TurnContext, yields one `GoalWork` when the Goal may call the rewriter (query_selection.py:906-909), then builds the Goal query text, looks up the Goal cache (keyed by `retrieval_goal_hash`) and sets `goal_payload` when the turn needs a Goal forward (912-921);
3. for a local turn > 0: `--release-idle-pages` readmission, `--admission-overlap`'s `lease_turn`, then phase 1 with `goal_payload=query.goal_payload` (242-259), in place (`run`) or, with `--phase1-in-step`, as a `Phase1Work` (`run_steps`);
4. `selection.plan_turn(binding, geo, generation, query, phase1)` (261), `phase1.adopt`, `adapter.reserve`.

Failure: once phase 1 has run, the session is poisoned, and a phase 1 that has not been adopted is aborted (270-280).

### 1.2 Where the Goal forward runs

Engine, `Phase1Coordinator.run` (phase1_coordinator.py:135-289):
- tokenizes `goal_payload` with `PLAIN_GOAL_SCHEMA` (raw, no special tokens; 167-176), bounded by one native forward;
- allocates the Goal's own blocks next to phase 1's: `ceil(G/P)` FA blocks per attention group and one Mamba block per Mamba group (202-203);
- puts them into the phase-1 lease as `lease['goal']`: rows (FA rows of `width` blocks, null-padded Mamba rows), owned, zero (the FA part), incarnations, `payload_sha256` (224-233);
- sets `query_tokens` (evidence only) to the Goal tokens (279-280).

The capacity gate counts the Goal blocks in the phase-1 peak: `A·(fresh + goal + evicted)` FA blocks and `2·M` GDN blocks, i.e. two GDN slots of one block per Mamba group (capacity.py:211-222). The gate cannot know G, so `goal` is the Goal bound, `pages(phase1.max_tokens, P)` (one native forward; the window layout's `goal_tokens`), counted when selection may be active on a `goal_query` turn.

Worker, `NativeQueryForward.phase1` (native_probe.py:144-193):
1. runs the phase-1 forward(s) on `[R][tail][fresh]`;
2. `_cleanup_request(rid)`: the phase-1 request leaves the runner, its KV stays;
3. runs the Goal forward: `self._execute(goal.lease_id, goal.tokens, 0, goal.rows, goal.zero)` with `phase='goal'` (179-186).

`model_queries.phase1` normalizes the last Goal row (pre-final-norm, `pre_final_norm_last_token_l2`; 126-139) and signs it with the phase-1 receipt (`goal_vector`, `goal_token_count`, `goal_tokens_sha256`, `goal_payload_sha256`). `plan_turn` (query_selection.py:1072-1082) uses `phase1.goal_vector` as the query (or the cached vector on a hit) and caches it.

### 1.3 What the Goal rows attend to and read

| Input | Phase 1 | Goal forward |
| --- | --- | --- |
| tokens | `T[C:e)` (the new tokens) | the Goal query text, tokenized alone |
| positions | logical `C … e-1` (`SparsePlan` maps FA slots to `x - (Q - |R|P)`, RoPE stays logical) | `0 … G-1` (`start=0`; `text_positions` with the same code) |
| FA KV read | the resident Pages R, the live tail, its own fresh blocks | **only its own fresh blocks**, zeroed through `new_block_ids_to_zero` and written by this forward. No `SparsePlan` (the phase-1 plan was popped by `_cleanup_request`), so no history KV |
| GDN state | the committed carry restored into the fresh slot (`before_forward`, native_probe.py:279-288, only for `phase == 'phase1'`) | a fresh slot. vLLM starts a request with no computed tokens from a zero state (`has_initial_state = context_lens > 0`, SO/vllm/v1/attention/backends/gdn_attn.py:390), so the slot's content is never read |
| state written | the GDN state at `e` (captured), the tail capture, fresh FA KV kept for phase 2 | its own blocks only, freed at the joined drain |
| output | Page vectors, the observation vector (row `e-1`) | the last row, pre-final-norm, L2-normalized |
| batch | one request, probe-only, eager (`force_eager`), or one row of a decode step with `--phase1-in-step` | one request, probe-only, eager; a lease with a Goal never goes into a step (phase1_step.py:133-139) |

The Goal forward reads nothing that phase 1 writes, and nothing that phase 1 reads either, except the model weights.

### 1.4 Parking today

`goal_async.GoalAdmissions` drives the admission generator. A `GoalWork` parks: it is submitted to the Goal thread, or with `--admission-overlap K` to the session's own Goal thread. The engine keeps stepping, and the admission resumes, one per engine pass, the oldest done first. Phase 1 then runs on the engine thread. Evidence: `goal_async_ms`, `goal_construct_ms`, the rewriter totals, `phase1_park_ms`, `lease_wait_ms`; parked times are subtracted from `admission_engine_ms` (registered_route.py:65-70).

## 2. Equivalence of a separate Goal forward

**Same input.**
- Tokens: the same `goal_payload`, tokenized by the same renderer call.
- Positions: the same, 0 to G-1.
- Context: the same, none (no history KV, no new tokens, no carry).
- Rows: of the same widths (`ceil(G/P)` FA blocks, one Mamba slot).
- Zeroing: the same zero list (the FA blocks).
- Representation: the same, the pre-final-norm last row, L2-normalized.

**Same computation.** The worker runs it through the same `_execute` call, again as a probe-only pass. The runner first drops the other requests from its persistent batch (vLLM's `_update_states` drops unscheduled requests), as it does for phase 1 today, so the batch is again exactly one request of G tokens at `num_computed_tokens = 0`. A probe-only pass is eager whatever the CUDA-graph mode, so the kernels, their launch shapes and the split heuristics that depend on shapes are those of today's Goal forward. The block ids differ (different allocations), but paged attention reads values through the block table, so the ids change where the KV lives, not what is computed.

**Exact, not only "up to numerics".** On the same server process and GPU, the separate forward computes the same floating-point operations in the same order as today's in-RPC Goal forward. What can make two runs differ is the GPU's own run-to-run nondeterminism (for example a Triton autotuner choice per process). It affects today's runs alike and was not observed (job 3349078, identical phase-1 vectors on a repeated server). Running the rewriter on the same GPU at the same time does not change the results of deterministic kernels, only their speed.

**Selection.** Page scores are the cosine between each Page vector and the query vector, after the same projection and int8 compaction (`query_vector`, `plan`). The Page vectors come from phase 1, which is unchanged: the same forward, now without a Goal forward after it in the same RPC, with which it shared nothing. With the same query vector, every score, the ranking, the selected count and the signed receipt are the same. The Goal cache stores the same vector.

**What can still differ, as today.**
- The Goal itself can differ only where a parked Goal can already differ (goal_async.py:24-33): a rewriter call that is slower because it shares the GPU, now also with a phase-1 prefill, can reach its 120-s timeout and fall back (`fallback_reason`).
- With `--goal-rewriter-batch`, which calls are batched together depends on timing, and batched greedy decoding is not bit-identical ([docs/reference.md section 4.10](../reference.md#410-admission-optimizations-opt-in)). This is true of any change in timing.
- The phase-1 numerics change only where `--phase1-in-step` already changes them: a phase 1 that today stays a separate pass because of its Goal forward may now run inside a step, as every other turn's phase 1 already does with that flag. A phase 1 inside a step changes the phase-1 vectors: in job 3349078, s-all vs s-off (`mixed_share` 0.9375: most phase 1s joined a step) differ on 51 of 51 compared turns (cosine below 1 on 5,752 Pages, median 0.9986); the one-session pairs, where no phase 1 joined a step, are identical (2,866 of 2,866 Pages).

## 3. Design (`--goal-overlap`)

### 3.1 The order under the flag

`reserve_steps` with `endpoint.goal_overlap` on a `goal_query` session:
1. `turn = yield from selection.begin_turn_overlap(binding, geo, conversation)` runs the first half of `begin_turn_steps`: its checks, the TurnContext, a summarization sub-call's Goal (`subcall_goal`, no rewriter), the hand-off release. A Goal that may call the rewriter is yielded as one `GoalStart(work)` and does not park. The driver submits it to the Goal thread and the admission continues at once.
2. Readmission, `lease_turn` and phase 1, with `goal_payload=None`: in place, or as a `Phase1Work` that parks with `--phase1-in-step`.
3. `query = yield from turn.finish()`: a `GoalJoin` returns the GoalRecord. It parks only while the Goal is not built yet. Then comes the second half of `begin_turn_steps`, unchanged: the Goal text, the cache lookup, `goal_payload`, `pending`.
4. If phase 1 ran and `query.goal_payload` is set: `phase1 = yield from self.phase1.goal_forward_steps(...)` (3.3).
5. `plan_turn`, `adopt`, `reserve` as today.

`begin_turn_steps` becomes `_begin` + `_finish` with the same statements in the same order, so the flag-off path is a refactor, not a change.

### 3.2 Goal start and join (goal_async)

- `GoalStart(work)` (kind `goal_start`).
  - `GoalAdmissions._advance` submits it (`_start`): to the Goal thread, or with `--admission-overlap` to the session's thread, with the same `_build` timer and wake-up callback as a parked Goal. It then sends nothing back and continues.
  - `run_inline` (no park table) builds it in place, which is today's order.
- `GoalJoin(start)` (kind `goal`).
  - Parks only when the Goal's future is not done (`_parks`). Parked, its future is the Goal's: `ready`/`resume` are unchanged.
  - Done at the join: it runs in place and returns the record or raises the construction's error.
  - Abort while parked: the future is cancelled, and `GoalAborted` is thrown at the join.
- A started Goal that is never joined (phase 1 failed, or an abort while parked at the in-step `Phase1Work` or a `LeaseWait`) is discarded in `reserve_steps`'s failure path (`future.cancel()`; a running construction finishes on its thread and its result is dropped), exactly as an aborted parked Goal is today. The session is poisoned by the failure anyway, and a session close releases its Goal state on its Goal thread, behind any Goal still running there (`after_goals`).

### 3.3 Phase 1 without the Goal; the separate Goal forward

Phase 1 is today's lease with `goal=None`: the same code path that every turn without a Goal forward takes today.

`Phase1Coordinator.goal_forward_steps(binding, geo, phase1, goal_payload, allocation_generation)`:
1. With `--admission-overlap`: `lease_turn` (3.4).
2. Tokenizes as `run` does, and allocates the Goal's blocks as `run` does (FA `ceil(G/P)` per attention group, one Mamba block per Mamba group).
3. Builds a Goal-forward lease: the Goal lease of today (`GOAL_LEASE_KEYS`) plus the identity of the turn (request, binding, session, namespace, allocation generation, epoch, source, local turn, Page size) and the id of this turn's drained phase-1 lease. It records it in `goal_records`. The lease is **not signed or digested** (3.11).
4. Calls the worker's new `goal_forward` operation (`query_tokens` = the Goal tokens, as the phase-1 call does today). Its reply carries the worker's joined drain of the lease. The engine checks it, frees the Goal blocks and returns `dataclasses.replace(phase1, goal_vector=…, goal_payload_sha256=…, goal_token_count=…)`.

If the call fails (no reply), the engine asks for the drain through `phase1_drain` with the Goal request (idempotent) before it frees anything: it never infers a release from an error, as for phase 1. A failed drain keeps the Goal blocks referenced, and `abort` retries it. `plan_turn` and the reservation are unchanged.

Worker:
- `RunnerAllocationOwner._goal_forward`: accepted only with the pinned flag (`owner.goal_overlap`), after this turn's phase 1 and before `reserve`, once.
- `model_queries.goal_forward`: the phase-1 identity checks, and the lease (`contract.parse_goal_lease`, structural only) checked field by field against the prepared turn and its phase-1 lease (whose epoch phase 1 checked).
  - It drives `NativeQueryForward.goal_forward`, which runs today's Goal branch: `_execute(lease_id, tokens, 0, rows, zero)`, `phase='goal'`.
  - The forward drains in its `finally` (`drain_goal`).
  - The reply is unsigned, as the `prepare` and `reserve` replies are. It holds today's Goal fields (`goal_vector`, `goal_token_count`, `goal_tokens_sha256`, `goal_payload_sha256`), the position evidence, `goal_ns`, `drained` and the drain record.
- The driver registers the Goal lease in its own lease table, with phase-1 leases' alias checks against claims, held slots and undrained leases. So `request_drained`, which `reserve`, `complete` and `cancel` require, covers it, and so do `close_session`'s lease cleanup and `evict`'s drain check. `phase1_drain` of a Goal-forward lease returns that drain record (a never-started lease is tombstoned), unsigned.

### 3.4 The lease: one undrained lease at a time

Today the engine allows one phase-1 lease record at a time, until adoption or abort (`require(not self.records)`, phase1_coordinator.py:153, and `LeaseWait`). With the split, an admission holds its drained phase-1 record while it waits for its Goal (seconds). Under the old rule, every other admission's phase 1 would wait for that Goal, which would undo what `--admission-overlap` gains. The worker only requires that every earlier lease be drained (native_probe.py:96-98).

So under the flag the engine's rule becomes **no undrained lease** (`Phase1Coordinator.busy()`):
- A drained record holds fresh FA run blocks (and with the hand-off the probe slot), which no other allocation can alias: they are referenced and absent from the free queue.
- B's phase 1 is for another session (one admission per session).
- A's Goal-forward lease is again a lease: it waits (`LeaseWait`) while B's in-step phase 1 is undrained (from its lease to its resume), and B waits while A's Goal forward runs. That forward runs within one engine-thread call and is drained there.

Flag off: `busy()` is `bool(self.records)`, today's rule.

### 3.5 `--gdn-device-handoff`

The probe slot holds the GDN state at `e` from phase 1 to phase 2's first step (`HeldSlots`). Under the flag it is held while the admission waits for its Goal too. It is the same slot, the same claims (`held_claims` refuse any alias, including the Goal lease's blocks) and the same release rules; only longer. The Goal forward's own Mamba slot is fresh and never held.

### 3.6 `--phase1-in-step`

The phase-1 lease never carries a Goal forward under the flag, so `Phase1Steps.accepts` takes it into a step like any other turn's phase 1. The Goal forward is a short probe-only pass at the admission's resume: one more separate pass on turns with a Goal forward, but only for the Goal tokens. Keeping it out of a step keeps it exact (2). An in-step Goal forward would change its batch shape, so it would only be equal up to numerics; it is out of scope (5).

### 3.7 `--admission-overlap K` and capacity

- Blocks: the turn's peak does not grow. Today phase 1 holds the fresh FA run, the probe slot, the Goal FA blocks and the Goal slot at once. Under the flag the Goal blocks come after phase 1's drain: the run is still held, but the probe slot is freed, except with the hand-off. So the existing phase-1 figure, which counts the Goal blocks (capacity.py:211-222), covers it.
- Decodes: running requests decode while the admission waits for its Goal after phase 1 instead of before it. The gate already counts their headroom next to the phase-1 figure when Goals park (`later = headroom`).
- Ledger: what the drained record holds comes off the admission's reservation (`held`), and `held` also counts a Goal record's blocks.
- Deadlock: nothing waits for blocks while it holds blocks. A parked join needs only its Goal thread, and the Goal forward needs only the reservation it already holds.

### 3.8 Cancellation, expiry, teardown

- Parked at the join: `GoalAdmissions.abort` cancels the future and throws `GoalAborted` there. `reserve_steps`' failure path poisons the session, because phase 1 ran, and aborts phase 1 (it frees the run and moves the hand-off slot) and any Goal record. The lifecycle then sends the worker `cancel` as for any partial admission. The native file of such a request has phase-1 frames, unlike today's cancel at the Goal.
- Parked at an in-step phase 1 or a `LeaseWait` with the Goal started: the same, plus the discard of 3.2.
- Teardown (`close`) aborts what is parked, then shuts the Goal threads down with `cancel_futures`, as today.

### 3.9 Gemma 4 (window layout)

Phase B refused `--goal-overlap` on the Gemma 4 profile, because the window ledger relies on "an admission parks at its Goal before phase 1 and nowhere after it" (`window_capacity.ledger_ahead`). The follow-up (branch `gemma-overlap`) enables it there; section 9 has the analysis. The claim made here before, that a drained phase 1 waiting for its Goal is over-reserved by the ledger's own arithmetic, was wrong: without a change of the window layout such a wait runs pool 1 out of blocks (9.3).

### 3.10 Evidence

Only under the flag:
- worker frame `goal_forward`, after phase 1's frames, on a turn with a Goal forward. Its drain record is inside it (`drain`), so there is no separate `phase1_drain` frame for it; one appears only after a failed Goal forward;
- native file: `goal_overlap_ms` (from the Goal's start to its join: the engine-thread work done meanwhile, phase 1 included); `goal_forward_ms` (the separate Goal forward's `goal_forward` RPC on the engine thread, when one ran; the worker's own time is the reply's `goal_ns`); `goal_async_ms`, which keeps its meaning (time parked at the Goal, 0 when the Goal was built by the join; subtracted from `admission_engine_ms` as before); `goal_construct_ms` and the rewriter totals as today.

`runtime.jsonl` rows stay the same as without the flag: `stage_b.phase1.goal_forward` and `query.query_token_sha256` come from the Goal-forward frame when there is one, and the prepare H2D sum also counts that frame (0 bytes: the forward is cold). `goals.jsonl` is unchanged.

Without the flag nothing changes: no config key, no worker or READY key, no frame, no field.

### 3.11 Flag plumbing

- `serve.py --goal-overlap`: Qwen, `goal_query` with `--goal-rewriter-model` only.
- It pins `admission.goal_overlap = true` (`admission_flags.FLAGS`), which the worker echoes and READY records; it is checked once at startup and adds no vLLM argument.
- The engine attaches it after `goal_async` (`goal_overlap.attach_engine`). On a server without a Goal park table (one session, `max_num_seqs` 1) it is inert, as `--phase1-in-step` is: every admission keeps today's order.
- The worker accepts `goal_forward` only with the pinned flag.
- **No per-request hashing beyond a flag-off turn's** (the rule: integrity checks at startup only).
  - The Goal-forward lease has no MAC and no lease or request digest. The worker checks it by equality against the prepared turn and its phase-1 lease, which keep today's MAC and digests. The epoch is compared with that lease's instead of hashed again.
  - The `goal_forward` reply and its drain record are not signed receipts.
  - In the CPU harness every SHA-256/HMAC call of the runtime sources is the flag-off server's, in total and per function, with two exceptions per Goal forward, each moving one call. The Goal payload digest moves from phase 1's lease (`run`) to the Goal forward's (`_goal_lease`). The Goal-token digests move from phase 1's receipt and `_result` to the Goal forward's reply and `_goal_result`, both inside `contract.digest`, so the count there is unchanged. `test_goal_overlap.py` pins this.
  - The first implementation (`230f9a9`) made 20 more SHA-256/HMAC calls per Goal forward, as the CPU harness counted them (commit `5c8bc24`).

## 4. Expected gain and costs

**Gain**, per turn whose Goal is not built when phase 1 would start: the admission completes about min(T_goal, T_phase1') earlier, where T_phase1' is the phase-1 work of the admission (readmission, lease wait, phase 1).
- On the 10-26 Terminal-Bench run (phase 1 150-250 ms, rewriter seconds) that is the phase-1 time per rewriter turn: a few percent of those turns' admission latency. It is not engine time: phase 1 and the Goal forward still run on the engine thread.
- With `--admission-overlap 16` the other admissions already proceed while a Goal is built, so the throughput effect is small. The saving is the rewriter turn's own latency.
- Once the rewriter is fast, the saving approaches min(T_goal, T_phase1), up to half of the Goal-plus-phase-1 path when the two are equal. The vLLM rewriter is not that fast: in rw3 (job 3499550) a call took a wall-clock median of about 975 ms alone and 16.4 s with 16 calls in flight.
- With `--phase1-in-step`, turns with a Goal forward also stall the decodes less (only the Goal forward stays a separate pass).

**Costs.**
- On a turn with a Goal forward, one more probe-only pass and RPC pair: the running decodes are dropped and re-added once more. The forward itself is the same work.
- The phase-1 record and, with the hand-off, the probe slot are held during the Goal wait. They are inside the turn's existing reservation.
- The rewriter shares the GPU with a phase-1 prefill, not only with decode steps. Its calls may be slower; measure `call_ms` and the fallbacks.

## 5. Out of scope: overlaps that change Goal behaviour

These belong to the Goal owner. Each changes which Goal a turn uses, or when it is built:
- **Start the rewriter earlier.** For example at the frontend when a request arrives (before the admission gate or a `CAPACITY_WAIT`), or during the previous turn's decode from a partial observation. Either changes the TurnContext or the time at which constructor state is read.
- **Use the previous Goal for this turn's selection** and update the Goal asynchronously for the next turn: a stale query by design.
- **Skip or speculate the rewriter** (for example select with the deterministic Goal and re-select if the rewrite differs).
- **A Goal forward inside a decode step.** It does not change Goal semantics, but it is equal only up to numerics (different batch shape), so it would need its own flag and a numerics comparison.

## 6. Risks

- Rewriter calls slowed by a concurrent phase-1 prefill (timeouts fall back and are recorded).
- Longer-held phase-1 records with `--admission-overlap`: the relaxed lease rule (3.4) is new. The worker already enforced only "all drained"; CPU tests cover two admissions interleaving, and on the GPU job 3500565 ran up to 16 admissions in flight (section 8).
- A failure at the Goal join now comes after phase 1 has run: the session is poisoned either way, but the failure frames differ from today's.
- The Goal forward's lease and reply carry no MAC or digest (3.11). Their integrity rests on the in-process engine–worker channel, the worker's equality checks against the turn's signed phase-1 lease, and the alias checks. This is the same standing as the unsigned `prepare` and `reserve` replies.
- Gemma 4: section 9.7.

## 7. What a GPU check should measure

On one GH200, Qwen3.6-27B, `goal_query` with the Qwen3-0.6B rewriter: the same script and sessions with and without `--goal-overlap` (and once with `--phase1-in-step` too).
1. **Equality.** Per turn: the Goal vector bytes (the `goal_forward` frame against the flag-off phase-1 frame), the Page vectors, the selections and the output tokens. They should be equal turn for turn where the Goals are equal. Run unbatched rewriter calls, or accept batch-composition noise.
2. **Gain.** Per rewriter turn: `goal_overlap_ms`, `goal_async_ms`, `goal_construct_ms` and the admission latency (ACK wait). Expect a saving of about min(`goal_construct_ms`, `goal_overlap_ms`).
3. **Costs.** `goal_forward_ms` and the decode stall (`admission_engine_ms`, inter-token latency) on turns with a Goal forward; rewriter `call_ms` and fallbacks with and without the flag.
4. **Capacity.** With `--admission-overlap 16`: `CAPACITY_WAIT` counts and `lease_wait_ms`, no preemption, both pools back at baseline after the run.

## 8. Implementation and GPU checks

Phase A: this design (commit `d8774fd`). Phase B: implemented on this branch behind `--goal-overlap` (default off). `230f9a9` added the implementation. The follow-up commit after it removed the Goal forward's extra hashing (3.11): its drain is folded into its reply, and its lease and reply are no longer signed or digested. Gemma 4 refused the flag at first (3.9); branch `gemma-overlap` enables it there (section 9). GPU runs and verdicts: [docs/status.md](../status.md).

Code:
- engine: `goal_overlap.py` (attach, `GoalForward`), `goal_async.py` (`GoalStart`, `GoalJoin`), `query_selection.py` (`_begin` / `_finish`, `begin_turn_overlap`, `StartedTurn`), `private_endpoint.py`, `phase1_coordinator.py` (`busy`, `goal_forward_steps`, `recover_goal`), `composition.py`;
- worker: `allocation_owner.py`, `model_queries.goal_forward`, `native_probe.goal_forward`, `gdn_handoff.attach_worker`;
- shared: `contract.parse_goal_lease`;
- evidence and launcher: `remote_ra_http/evidence.py`, `remote_ra_http/config.py`, `serve.py`.

With the flag off, every new branch is skipped or empty (`goal_records` stays empty, `busy()` is `bool(records)`), and the existing suites run unchanged.

`test_goal_overlap.py`, 23 tests:
- launcher, config, attach and worker refusals;
- the drivers;
- the lease rule;
- the Goal-forward lease;
- end to end over the batched HTTP fakes: the equivalence with each session alone on the flag-off server (Goal, Goal vector, ranking, selection, tokens, `runtime.jsonl`, plan, `goals.jsonl`); phase 1 while a rewrite is held; a disconnect at the join; a failed Goal forward drained through `phase1_drain`; `--admission-overlap 2` with a drained lease waiting;
- end to end over the decoupled 16-token fakes with `--admission-overlap 3`, alone, with `--gdn-device-handoff` and with `--phase1-in-step`: every decision equals the all-off server's, and Goal-forward turns run their phase 1 mixed in a step;
- the hashing: every SHA-256/HMAC call equals the flag-off server's, only the Goal payload digest moving from `run` to `_goal_lease`.

What the GPU checks of section 7 measured (cmg1, runtime `c07d42a`; from each job's offline verification, not audited):
- **Qwen, job 3500604**, both pairs with thinking on. The strip pair qref / qovl (`--thinking-history strip`) was EXACT over 30 lockstep turns: Goal equal on 30 of 30, 18 Goal-forward turns compared. The keep pair kref / kovl (`--protocol mswea-xml`, `--enable-thinking` alone) was EXACT over 18 turns, 8 with a Goal forward; that pair compares the flag on and off, so it does not check the keep mode itself.
- **Qwen with `--admission-overlap 16`.** In the SWE-bench job 3500565 (server goalvllm, with the [vLLM Goal engine](../rewriter-vllm/DESIGN.md)) `--goal-overlap` ran with up to 16 admissions in flight: 2,093 admissions, 1,149 Goal forwards, no preemption.
- **Gemma 4 (section 9), job 3538523.** gref, govl and gvll answered 49 of 49 requests each; the pair gref / govl was EXACT over 25 lockstep turns (Goal equal on 25, 15 Goal-forward turns compared) and govl / gvll (gvll with the vLLM Goal engine) over 23 (Goal equal on 22: one Goal differed, which ends that session's lockstep; 12 Goal-forward turns compared).

## 9. Gemma 4 31B IT (window layout; branch `gemma-overlap`, 2026-10-30)

Checked in the sources of `5c8bc24`. The figures are Gemma's: W = 1,024 (the sliding window), P = 16, N = 8,192 (one native forward), G = 2,048 (the Goal transient, `model_profile.GEMMA4_GOAL_TOKENS`), five sliding groups; per sliding group a window slot is ⌈W/P⌉ + 1 = 65 blocks, one forward's transient 512 and the Goal transient 128.

**Verdict: exact, and the change is contained.** The Gemma Goal vector is bit-identical when the Goal forward runs on its own (9.2). The park after phase 1 needs two changes on the window layout, both behind the flag: phase 1's drain frees the session's pinned window blocks that the adoption would free (`WindowPins.trim`), and the sliding ledger takes what a waiting admission still holds off its base (`window_capacity.ledger_ahead_split`). Without them it is not safe (9.3). Pool 1's size, the gate's figures and every flag-off path are unchanged.

### 9.1 The window layout today

- **Pools.** Pool 0 holds the global (full-attention) layers in P-token blocks: the Pages. Pool 1 holds the sliding layers and is reserved at startup: a null block plus, per sliding group, S window slots, one forward's transient and, on a `goal_query` server, the Goal transient (`kv_layout.window_pool_blocks`).
- **Window pin (W-GPU).** Between turns a session pins, per sliding group, the blocks its next query reads, `kept_blocks(C)` (≤ 65; `WindowPins`).
- **Phase 1.** It borrows that window (null head, the pinned blocks, read only) and appends fresh pool-1 blocks for [C, e) (`sliding_window.phase1_row`). A run longer than one forward runs in chunks of 4,096 on one lease, the fresh blocks of tokens that left the window reused (`phase1_ring`).
  - At the drain, the fresh blocks before the window phase 1 ended in (`kept_blocks(e)`) are freed. The others (`stay`) wait for the adoption.
  - The adoption hands the window phase 1 ended in (the pinned blocks from `kept_blocks(e)`'s first index on, plus `stay`) to the phase-2 reservation and frees the older pinned blocks (`WindowPins.handover`).
- **Goal forward (today).** It runs inside the phase-1 RPC, after phase 1's forwards, as a cold forward. Its rows are its own blocks: ⌈G/P⌉ FA blocks (zeroed) and ⌈G/P⌉ sliding blocks per sliding group, with no null head, its tokens at most G (`WindowPhase1Coordinator.run`). Its sliding blocks come out of the Goal transient, which pool 1 reserves because the Goal forward runs while phase 1's fresh blocks are still held.
- **Gate and ledger.** The gate takes the global figures of the Qwen rules and the sliding ones from `window_capacity`. With `--admission-overlap K`, each in-flight admission reserves its turn's sliding *base* (what it keeps once accepted beyond its session's old pin: max(adopted window, phase-2 peak) − old window), plus the largest *transient* (peak − base: the fresh ring and the Goal forward) once for all admissions (`ledger_ahead`). This is sound because today an admission parks only at its Goal, before phase 1, holding nothing. From its phase-1 lease to its acceptance it runs in one engine-thread call. `--phase1-in-step` is refused on this layout.

### 9.2 Equivalence: the Gemma Goal vector

The separate Goal forward is the same `_execute` call as today's in-RPC one:
- the same Goal tokens (`<bos>` and the `<retrieval_query>` text, the same renderer call);
- positions 0 … G−1;
- rows of the same widths (⌈G/P⌉ blocks in every group, sliding rows without a null head);
- the same zero list (the pool-0 ids);
- a one-request, probe-only, eager batch.

The new code path is `WindowPhase1Coordinator._goal_lease` and the worker's `NativeQueryForward.goal_forward`, which has no window-specific code.

What it reads:
- **Global layer:** only its own FA blocks, zeroed and written by this forward. There is no SparsePlan for its request id and no history KV.
- **Sliding layers:** only its own pool-1 blocks.
  - The query at position p reads [max(0, p − W + 1), p], all written earlier in the same forward. For G > W the window moves inside the forward, as it does today.
  - The blocks are not zeroed (the KV zeroer writes FA tensors only), today as before, and no position past the forward's own is read.
  - It reads no window state at all: not the session's pinned window (which only phase 1 borrows), not phase 1's fresh or chunk-ring blocks, not the trimmed blocks (9.3).

Only the block ids differ. So the vector is bit-identical, up to the GPU's run-to-run nondeterminism, as for Qwen (section 2). The vector is the last row before the final norm (`ProbeRows.last`, fp32), L2-normalized, as in today's receipt.

The Page vectors come from phase 1, now without a Goal forward after it in the same RPC. That Goal forward shared nothing with phase 1, so they are unchanged. With chunks, the Goal forward ran after the last chunk and still does. The CPU tests compare the Goal-vector bytes turn by turn: the worker's `goal_forward` reply against the flag-off phase-1 receipt.

### 9.3 What the park after phase 1 needs

**Removing the refusal alone is unsafe.** A turn waiting for its Goal after its drain holds the session's whole old window (≤ 64 blocks per group) and the `stay` blocks of its new one (up to 65). Before, both existed only inside one engine-thread call. The ledger counts that holding as the admission's lease (`held`) and takes it off its *transient*, which is reserved only once, the largest. So it hides other admissions' needs: the next admission's phase 1 or a waiting admission's Goal forward can then find too few free blocks. On the GPU, that phase 1 fails its admission (`insufficient native blocks`, the session poisoned), and a decode that finds no block is a preemption.
- A simulation of pool 1 on the Gemma geometry (`test_gemma_goal_overlap.simulate`) shows it: 16 sessions, K = 16, random runs (1 to 12,000 tokens), Goals in 70 % of the turns, decodes and completions, every admission gated as `capacity.ledger_gate` gates it.
- With the refusal simply removed, 40 of 40 random runs ran out of pool 1 (`phase 1 needs 422 of 392 free`, `Goal forward needs 128 of 123 free`).
- With today's order, 0 of 40 did.

**Change 1: free the old window at the drain (`WindowPins.trim`).** Under the flag, once phase 1's drain is proven, the session's pinned blocks before `kept_blocks(e)`'s first index are freed (`WindowPhase1Coordinator._run_split`). These are the blocks the adoption frees today.
- Why freeing them is safe: no later forward reads them. The session's next query reads [e − W + 1, e], phase 2 is installed with `kept_blocks(e)`, and the Goal forward is cold. Phase 1 has left the runner (its drain). `handover` then drops nothing.
- The worker sees no change. Its committed window carry still names the old blocks, but only that session's next phase 1 would read it, after a commit has replaced it. A failed admission poisons the session as today, and its close drops the carry. `WindowPins.close` frees what stays pinned. `idle_release` never picks a session with a phase-1 record (`phase1.pending`).
- After the trim, a waiting admission holds, beyond its session's old pin, only `stay − trimmed` = adopted window − old window blocks per group. That is what its turn keeps once accepted, and never more than its base:
  - 1 block late in a session (a 4,300-token observation: stay 65, trimmed 64; base 2);
  - more early in a session, when the window still grows (C = 300, e = 1,400: 46, base 47);
  - possibly negative, when e falls in the last pinned block.

**Change 2: what a waiting admission holds comes off its base (`ledger_ahead_split`).** `WindowPhase1Coordinator.held` reports that net figure for a drained record (`_held_split`), and `capacity.reserved` uses the split rule when `phase1.split` is set: kept = base − held and transient = peak − base. The invariant is free ≥ Σ kept + the largest transient + the running requests' headroom. The gate establishes it, and each step preserves it:
- **Phase 1** (one call): it needs its ring ≤ peak = its kept + its transient.
- **Drain and trim:** free drops by the net and Σ kept by the same; the transient stays.
- **Goal forward** (one call, freed at its drain): it needs ⌈G/P⌉ ≤ peak − held. This holds because peak ≥ ring + ⌈G/P⌉ and held ≤ stay ≤ ring.
- **Acceptance:** the admission leaves the ledger and its growth becomes headroom. kept = base − net ≥ phase-2 peak − adopted window.
- **Decodes:** they take what the headroom counted.

With every held figure ≥ 0, today's rule reserves at least as much as the split rule, so it is safe but more conservative. With a negative figure it reserves that many blocks too few next to a larger transient. The split rule is exact in both cases.

**Leases.** Every Gemma lease, phase 1 and the Goal forward, still runs and drains inside one engine-thread call (no `--phase1-in-step`). So at a gate no lease is undrained, and the transients are still never allocated together.
- `busy()` (undrained leases only, 3.4) lets the next admission's phase 1 run beside drained, waiting records, as for Qwen.
- The window `run` checks `busy()` under the flag.
- The engine refuses the flag on the window layout with an in-step phase 1 (`goal_overlap.attach_engine`).

**Without `--admission-overlap`** (K = 1, Goals parked): the gate refuses every other add while one admission is parked, and its phase-1 figure counts the running requests' headroom. The turn's sliding peak becomes max(ring, net + ⌈G/P⌉) instead of ring + ⌈G/P⌉, which is no more than the figure the gate checked. Phase 2 is unchanged.

**Not chosen: keep the old window pinned during the wait** and reserve max(base, stay) per admission. That needs no trim, but costs admissions in flight. At Terminal-Bench observation sizes (2,600 to 4,300 tokens: stay 65, base 2, transient 226 or more), pool 1 at 16 sessions would take about 6 admissions in flight instead of 16. With the trim, the reservations are today's. In the CPU geometry as many turns wait at once as wait at their Goal without the flag, and in the simulation up to 16 admissions waited after their phase 1 at once with no shortfall in 40 runs.

### 9.4 The other parts of the window layout

- **Goal transient (G = 2,048).** It is unchanged. The separate Goal forward takes its ⌈G/P⌉ sliding blocks per group, which are freed at its drain, while its turn holds only its net window. A longer Goal fails with `PHASE1_TOO_LONG` before anything is allocated, as today. The Goal-forward lease is checked as a phase-1 lease's Goal (`contract.parse_goal_lease`: FA and sliding rows of ⌈G/P⌉ blocks, tokens ≤ the layout's `goal_tokens`, zero list = the pool-0 ids).
- **Chunked phase 1.** It is still one lease, whose chunks run in one RPC and drain once. `stay` is the window its last forward ends in, and the trim and the Goal forward come after the drain. Nothing in the chunk ring outlives the drain.
- **`--release-idle-pages`.** It works on global Pages only, and the windows never leave pool 1. A readmission happens before phase 1 as today. A waiting session is never a victim (`phase1.pending`), and its readmitted Pages stay in its FA reservation (over-reserved, as for Qwen).
- **`--decode-fastpath3`.** The trim frees through pool 1 as `handover` does, so the window pool's journal records it (`window_rows.Journal`). Before the Goal forward's RPC, the proxy sends the queued allocation updates, as before phase 1 (`decode_fastpath2`). The probe-only pass drops and re-adds the running requests once more, as phase 1 does.
- **Worker.** `window_worker` takes the admission record `{goal_overlap: true}` with or without `admission_overlap`, and `gdn_handoff.attach_worker` sets `owner.goal_overlap`. The rest of the worker path (`model_queries.goal_forward`, `NativeQueryForward.goal_forward`, `drain_goal`) is the profile-independent one.
- **Launcher and config.** `serve.py` takes `--goal-overlap` on the Gemma profile with `goal_query` and `--goal-rewriter-model`, and pins only `admission.goal_overlap`. `remote_ra_http.config` takes it beside `--admission-overlap`. `--phase1-in-step` and `--gdn-device-handoff` stay refused.
- **Hashing.** As for Qwen (3.11), the Goal payload digest moves from `WindowPhase1Coordinator.run` to `_goal_lease`, and nothing else changes.

### 9.5 Gain

The same as for Qwen (section 4): about min(T_goal, T_phase1) per rewriter turn. Gemma's phase 1 was not measured for this design. The second benefit of 3.6 is absent, because Gemma refuses `--phase1-in-step`. The cost is one more probe-only pass per turn with a Goal forward, as for Qwen.

### 9.6 Code and tests

- Engine:
  - `window_phase1.py`: `run` (`busy()` under the flag), `_run_split`, `held` / `_held_split`, `_goal_lease`, and `release` (the Goal-forward lease first);
  - `remote_ra_native_growing/window_pins.py`: `trim`;
  - `window_capacity.ledger_ahead_split` and `capacity.reserved`;
  - `goal_overlap.attach_engine`.
- Shared: `contract.parse_goal_lease`.
- Worker: `window_worker.py`.
- Launcher and config: `serve.py prepare_gemma4`, `remote_ra_http/config.py`.
- With the flag off: `split` is False, so `run` and `held` take today's code, `reserved` uses `ledger_ahead`, and nothing is trimmed.

`test_gemma_goal_overlap.py`, 19 tests:
- the launcher, config and worker records and refusals; the engine attachment;
- the window Goal-forward lease;
- `trim`; `ledger_ahead_split`; the simulation (9.3);
- end to end over test_gemma_release's engine and worker fakes with test_gemma_overlap's rewriter stand-in. Every turn's tokens, replies, selections, attended Pages and Goal vectors equal the flag-off K = 1 server's:
  - with `--admission-overlap 8`;
  - with K = 1 and parked Goals;
  - with idle releases and the three decode fast paths;
  - with phase 1 in chunks of 256;
- held Goals: every admitted turn ran its phase 1 and waits at its join with only its net window. Pool 1 holds exactly those blocks, the ledger reserves Σ (base − held) plus the largest transient, and as many turns wait as without the flag;
- a cancel at the join: the session is poisoned, its phase 1 is cleaned up, and its trimmed window is freed at close;
- the hashing: only the Goal payload digest moves.

`test_goal_overlap.py` now checks the Gemma refusals of a policy without a rewriter instead of the profile's refusal.

### 9.7 Risks and the GPU check

- **The trim** is new: it frees pool-1 blocks earlier than today, after phase 1's drain instead of at the adoption. The CPU fakes check every sliding read and both pools' baselines; on the GPU the Gemma servers with the flag ran in job 3538523 (section 8).
- **The split ledger** is new arithmetic. The proof (9.3) and the simulation cover it, but the gate's figures assume the turn geometry the gate predicts (C, e from the transcript), as today.
- **A failure at the join** comes after phase 1 and the trim. The session is poisoned and its window freed at close, as for any failure after phase 1.
- **GPU check.** Section 7's check on the Gemma launch (`goal_query`, `--admission-overlap 16`, `--release-idle-pages`, the Qwen3-0.6B rewriter), with and without the flag:
  - equal Goal-vector bytes turn by turn (the `goal_forward` frame against the flag-off phase-1 frame) and equal selections;
  - no `insufficient native blocks`, no preemption, and both pools back at baseline;
  - `CAPACITY_WAIT` counts and admissions in flight against the server without the flag.
