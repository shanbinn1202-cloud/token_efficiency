# Async scheduling for the selective-prefix runtime: design update (option C since built)

English | [简体中文](DESIGN.zh-CN.md)

Date: 2026-10-30. Updates the note of 2026-10-16 (base `integrate-1015` @ `6c5c9ab`; that note is not in this
repository, and this one restates what it uses from it). Base read here: branch `async-sched` @ `cf2b28d` (143
commits after `6c5c9ab`). Paths are relative to `source-overlay/` unless they start with a top-level file
(`serve.py`, `README.md`) or a drive. Line numbers are those of `cf2b28d`. Nothing was implemented at this date.
This note decides nothing; section 6 lists what the maintainer had to decide.

As built: option C (section 4.3) is on main as the opt-in `--async-scheduling` (default off, not a speed
default): async-c `f162120`, async-fix `a0798a8`, async-gemma `4a498dc` (the Gemma 4 profile) and async-fix2
`4ed1013`; what was built, and where it deviates from this note, is in [IMPLEMENTATION.md](IMPLEMENTATION.md) and
[GEMMA.md](GEMMA.md). Its GPU checks (G1 on Qwen and on Gemma 4) and their verdicts are in
[docs/status.md](../status.md) and [History](../history.md#history); the limit that holds on Gemma 4 is
[docs/known-limits.md](../known-limits.md) 5.3. The gains of sections 0, 4 and 6 are this note's estimates. What was
built settled section 6's items 1 and 2 (option C), 3 (the relaxed contract: H1's rollback and H6 are not built) and
4 (both profiles; `--phase1-in-step` and `--gdn-device-handoff` stay refused with async), and superseded section 5's
"merge to main only after G1 and G2 pass" (option C is on main as an opt-in switch). As built, the lifecycle sweep
does not drain (H7 lists it): IMPLEMENTATION.md, sections 2 and 4. The rest of this page is the design record of
2026-10-30.

## 0. Summary

1. **What async buys, measured on stock vLLM 0.24.0** (one GH200, the decode replay of [docs/reference.md section 4.11](../reference.md#411-decode-fast-path-on-by-default)):
   1.4-2.0 ms per decode step. That is 4.8-9.2 % with FlashAttention 3 (job 3429022) and 5.1-10.2 % with
   TRITON_ATTN (job 3461668). The rounds with a full measurement window give 7.4-8.3 % (FA3, b = 1) and
   5.1-10.2 % (Triton, b = 1 and 4). Section 4.1 has the table.
2. **What it is worth per turn.** In the Qwen Terminal-Bench dry runs of 10-26 (job 3461661), decode
   including stalls is 80.9 % (`goal_query`) and 98.3 % (`observation_query`) of the server time per
   request. A 5-8 % shorter decode step is therefore at most 4.0-6.5 % (0.72-1.15 s of 17.7 s) and
   4.9-7.9 % (0.62-1.00 s of 12.7 s) of the server time per turn.
3. **The hazard has not changed in kind since 10-16.** Async launches step N+1 before it reads step N's
   tokens. Every request that stops at N therefore runs one extra row in N+1.
   - Qwen: that row advances the request's Gated-DeltaNet slot in place, and the carry is captured from
     this slot at completion. The carry then holds the stop token, and nothing notices. This is the one
     device hazard that affects correctness.
   - KV at the stop position: the extra row also writes the KV of position C′ (FA, and Gemma's sliding
     and global layers). The next turn's phase 1 rewrites C′ before anything reads it.
   - Host state: the extra step allocates blocks that the synchronous server never allocates (1 stop in 16
     at P = 16), and it advances our records by one token.
4. **What changed since 10-16 makes an exact design more feasible, and more useful.**
   - `--decode-fastpath2` already defers each step's after-forward work into one `finish_step`.
   - `--decode-fastpath3` and `--deferred-completion-io` cut the host work that async would hide.
   - Gemma 4 has no recurrent state.
   - Our synchronous server now decodes 2.3-5.5 ms per token slower than stock *async*, the group's
     baseline. Async is the largest remaining decode item.
5. **What removing the gates alone does, read from our code.** We did not read the collaborator's fork;
   this is what *our* code does if only the `require`, `allocation_guard` and `chunk_protocol` gates go.
   - Every EOS or marker completion fails closed at the worker's cursor check
     (`remote_ra_live_v2/allocation_owner.py:488-491`).
   - If that check is relaxed as well, the carry silently includes the stop token
     (`worker_backend.py:261-269`).
   - Either way, every forward is still joined inside its own `execute_model`
     (`generation_guard.py:35-38`, `worker_composition.py:172-173`). So it overlaps only the sampling wait and
     `update_from_output`, about 0.5-1 ms by our step profile.
6. **Recommendation.** If async is built, build option C (section 4.3): prepare step N+1 during forward N,
   then launch it only after N's stop decisions, with the true stops voided on the device. Do not build a
   gates-only form. Option F is exact but small. Option S comes closest to stock and carries the highest risk.

## 1. What changed since 10-16

| Change | Where (cf2b28d) | Effect on an async design |
| --- | --- | --- |
| `--decode-fastpath2` (one `allocation_batch` RPC per step; after-forward work deferred to `finish_step`) | `remote_ra_live_v2/decode_fastpath2.py:1-48`; `engine_boundary.py:287-319` (`flush_updates`); `worker_composition.py:129-148` (`_defer`), `155-186` (`finish_step`, `event.synchronize()` at 173); `generation_guard.py:40-42` | The per-step worker work is already one deferred unit, and the batch is one message that can be held until the stop decisions (option C). But `finish_step` still joins forward N inside `execute_model(N)`, so nothing overlaps until it moves to step N+1's pre-launch point. |
| `--decode-fastpath3` (steady-step reuse of leases, key tuples and claim differences; rowless batch entries; one-copy position read) | `decode_fastpath3.py:1-49`; `allocation_owner.py:1149` (`materialize_steady`); `qwen.py:237-305` (`before_forward3`); `positions.py:47-57` (`step_positions`: `.tolist()` of a device tensor) | The caches key on "the rows the worker last accepted". If the entries of stopped requests never reach the worker (C), nothing needs rolling back. If they do (S), a stopped request's caches are one step ahead, and the completion must cut them back. The device read of positions is a stream synchronization: harmless in C (it runs after N is resolved), a blocker in S. |
| `--deferred-completion-io` (file encoding and writing on one writer thread after `post_step`) | `deferred_completion.py:1-80`, `682-708` (`attach_engine` wraps `post_step`); [docs/reference.md section 4.11](../reference.md#411-decode-fast-path-on-by-default) | It works per busy-loop iteration, so the async step function keeps it. The native record is still built on the engine thread inside `update_from_output(N)`. Under async this happens while step N+1's allocations are in place, so the record's free-block counts differ (H6). The worker's `complete` RPC (about 40 ms at 517 new Pages, [docs/reference.md section 4.11](../reference.md#411-decode-fast-path-on-by-default)) stays on the engine thread, so a completion is a bubble under async as it is under sync. |
| Decoupled layout at P = 16 (FA pool of 16-token blocks; Mamba pool of K = 2S + 1 slots, M = 784) | `kv_layout.py:184-185`; [docs/reference.md section 4.6](../reference.md#46-decoupled-kv-layout-and-16-token-pages); `pool_incarnations.py:8-19` (incarnations per pool) | A stop whose C′ lands on a Page start (C′ % 16 == 0) makes step N+1 allocate an FA block the synchronous server never has. That is 1 stop in 16, 49 times as often as with 784-token pages. A Mamba crossing (C′ % 784 == 0) allocates a Mamba block. |
| `--cuda-graphs full` (FULL decode graphs; physical seq-len binding; persistent metadata buffers) | `remote_ra_live_v2/cuda_graphs.py:165-196`; `vllm/v1/attention/backends/gdn_attn.py:464-476` (state indices copied into the persistent buffer, padding filled with `NULL_BLOCK_ID`) | A void must write the persistent buffers after the builders' copies and before the replay. Our `before_forward` hook sits exactly there (`vllm/v1/worker/gpu_model_runner.py:4605-4606`). The graphs' hidden-state output is a static buffer: N's Page compaction and GDN captures must be queued before N+1's replay overwrites it. |
| Gemma 4 31B IT profile (window layout: 50 sliding layers W = 1,024 in pool 1, 10 global layers; TRITON_ATTN; refuses `--phase1-in-step` and `--gdn-device-handoff`) | `serve.py:951` (also `--no-async-scheduling`); `sliding_window.py:56-76`; `remote_ra_native_growing/window_pins.py:102-122`; `window_rows.py:78-108` (fastpath3's pool journal); `window_owner.py:48` (inherits `_complete`); [docs/reference.md section 4.14](../reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d) (`refused` row, line 841) | No recurrent state: the extra row's only device writes are KV at C′, which phase 1 rewrites (`sliding_window.phase1_row` appends into the last kept block). The host effects are in pool 1: a fresh block when C′ % 16 == 0, a block before the window freed one step early, and both recorded in the incarnation counts and fastpath3's journal. The completion's cursor check is shared with Qwen. |
| `--triton-attn` for Qwen (ef62aa6) | [docs/reference.md section 4.6](../reference.md#46-decoupled-kv-layout-and-16-token-pages) | The void's slot −1 must be skipped by the KV-update op of both backends. These kernels are not in the overlay: verify on the GPU. |
| `--phase1-in-step` (unchanged since 10-16) | `vllm/v1/engine/core.py:495-516` (probe hooks in `step` only); `phase1_step.py:158-230` | `step_with_batch_queue` (`core.py:536-649`) has no probe hooks. Refuse the flag with async in version 1; it is not in the benchmark configuration (README line 41). |
| `--admission-overlap K`, parked Goals, Gemma's sliding ledger, release before the gate | `admission_overlap.py:1-30`; `goal_async.py:285` (`resume`); `core.py:1366-1434`, `1454-1468`; `capacity.py:312-418`, `550` (`ledger_gate`); `idle_release.py:206/283/310`; `window_capacity.py:262/302` | These are all pass-boundary operations on the engine thread. Each needs the pipeline drained first (H7). |
| Measurements | the decode tables in the verification folders of jobs 3429022 (FA3) and 3461668 (Triton); step profile `step-profile.csv` of job 3429022; the turn profile of job 3461661 (an analysis outside this repository; section 4.1 quotes its figures) | Section 4.1. |

The synchronous scheduler is still required at every one of these places:
- `serve.py:614`, `951`;
- `native_worker_registration.py:57-63`, `195-196`;
- `bootstrap_contract.py:21-22`, `29-30` (`max_concurrent_batches == 1`);
- `startup_runtime.py:73` ("batch queue unsupported");
- `policy_runtime/bootstrap.py:39-45`;
- `private_authority/protocol.py:90`, `99-100`;
- `remote_ra_live_v2/positions.py:18`, run on every generation forward (`hooks.py:20`);
- `remote_ra_growing_worker/plan.py:106`;
- `remote_ra_qwen_semantics/hooks.py:32`;
- `allocation_guard.py:35-38` (output placeholders);
- `chunk_protocol.py:20-21` (`num_tokens == start + 1`) and `38-44` (`worker_interval`);
- the step profiler, `remote_ra_step_profile.py:899-902`, `973-976`.

vLLM's own mechanics (section 1 of the 10-16 note) are these: `core.py:536-649`, the placeholders at
`vllm/v1/core/sched/scheduler.py:490-522`, finished outputs dropped at `1723-1732`, the async runner output at
`gpu_model_runner.py:4952-5009`, and input ids taken from `prev_sampled_token_ids` at `1748-1855`.

## 2. The extra forward, step by step (current code)

Request R stops at step N, at computed end C′ = prompt + output − 1. Under stock async:

1. `schedule(N+1)` still holds R. The placeholder skip only covers a certain `max_tokens` stop
   (`scheduler.py:490-504`).
2. `EngineRpcBoundary.dispatch_mixed_request` advances R's records (`engine_boundary.py:112-155`, `273-283`).
   When C′ % 16 == 0, vLLM allocates a block for page C′/16.
3. `execute_model(N+1)` runs inline (UniProc). Our `before_forward` opens R's lease for [C′, C′+1). The
   forward writes KV at C′ and advances R's GDN slot. `finish_step` closes the lease, so `state.cursor`
   becomes C′+1 (`qwen.py:419`), and it may compact a vector for page C′/16 (`qwen.py:399-417`).
4. Only then does `update_from_output(N)` find the stop and call `lifecycle.before_success`
   (`scheduler.py:1848-1866`). That leads to the adapter's `complete` (`remote_ra_native_growing/adapter.py:409-414`,
   C′ from vLLM's output) and the worker's `_complete`, which requires
   `state.cursor == len(tokens) + len(output) − 1` (`allocation_owner.py:488-491`).
   - Unchanged code: this check fails, and the request ends with `FINISHED_ERROR`.
   - Check relaxed: `qwen.finish` captures the carry from the current slot (`qwen.py:349-352`,
     `worker_backend.py:261-269`). That slot already holds the state after C′, and the digest is computed
     from those bytes.

## 3. Hazards per owner, and how each can be handled

Complexity: S is under about 100 lines, M is 100-300, L is above 300 (runtime plus overlay, without tests).

### 3.1 Device

| # | Owner | What the extra row does | Void / rollback | Avoid the row | Complexity |
| --- | --- | --- | --- | --- | --- |
| D1 | Qwen GDN running slot (48 layers' conv and SSM state, 3 Mamba groups, 153.9 MB) | Advanced in place by the stop token. The carry (`worker_backend.py:261-269`) then equals the state after C′. The next turn's phase 1 restores it and computes C′ again: a **silent wrong carry**. | (a) **Void**: before the forward, set R's entries of every GDN group's `non_spec_state_indices_tensor` (the persistent buffer under FULL graphs) to `NULL_BLOCK_ID` = 0. The kernels then skip the row (`fused_recurrent.py:295-299`, `fused_sigmoid_gating.py:113/162`, `causal_conv1d.py:814-817`). The host must know the true stops before the launch (option C). (b) **Snapshot** (option S): a device predicate on `prev_sampled_token_ids` (stop ids, plus the tokens whose decode holds '>' on the XML protocol) copies each flagged row's slot into a reserved area before forward N+1, and the carry is taken from that area. It needs 154 MB of GPU memory per snapshot row and an overflow fallback (a stop without a snapshot forces a rebuild of that session). | Only for some stops (3.3). | (a) S, plus GPU checks of every GDN kernel path. (b) M-L. |
| D2 | Qwen Mamba crossing (C′ % 784 == 0, about 1 stop in 784) | Step N+1 takes a new Mamba block, and vLLM's preprocess copies R's state into it. The old slot survives: align mode frees the block of two steps ago (`vllm/v1/core/single_type_kv_cache_manager.py:1086-1103`). | With D1(a) the copy is wasted and harmless; the host block is H1. Under S, the carry is the latest capture at the M-aligned end (`qwen.py:337-347`), which is already taken in step N. | n/a | S |
| D3 | FA KV at C′ (the session's live tail Page; Gemma: the global tail and the 50 sliding layers' last kept block) | Writes the stop token's KV at C′. | Slot −1 in each group's slot mapping (persistent under FULL graphs) skips it. Verify that the FA3 and TRITON_ATTN KV-update kernels skip −1. Without a void the write is **benign** as long as the invariant holds: the next turn's phase 1 starts at C = C′ and rewrites C′ before any query reads it (adapter C2; `sliding_window.py:69-76`), and the live tail is never released ([docs/reference.md section 4.13](../reference.md#413-releasing-idle-sessions-pages-opt-in)). Nothing hashes the tail's bytes. | n/a | S |
| D4 | Page vectors and the static hidden-state buffer | When (C′+1) % 16 == 0 (1 in 16), the extra row closes page C′/16 and a vector the synchronous server never makes is compacted. Under FULL graphs, N+1's replay overwrites N's hidden rows. | C: R gets no lease in N+1, so no vector is made, and `finish_step(N)` (compaction, the M-aligned capture) runs before N+1 is launched. S: queue N's GPU part before the launch, verify later, drop R's vector when the completion truncates. | n/a | S (C) / M (S) |
| D5 | Gemma sliding windows | No state crosses turns except the pinned window, which is identified by its blocks and their incarnations (`window_lifecycle.py:348-373`). `WindowPins.pin_after` reads [C′ − W + 1, C′) from R's row, and that range stays live. | Only D3; the rest is host (H1). | n/a | S |

### 3.2 Host

| # | Owner | What diverges | Void / rollback | Complexity |
| --- | --- | --- | --- | --- |
| H1 | vLLM block pools and our incarnation counters (FA pool, Mamba pool, Gemma pool 1; fastpath3's journal) | R's speculative blocks: the FA block at C′ % 16 == 0; the Mamba block at a crossing; on Gemma, one fresh sliding block per group and one block before the window freed a step early. R's free returns them to the free queue: `BlockPool.free_blocks` puts unhashed blocks at its head (`prepend_n`) and hashed ones at its tail (`vllm/v1/core/block_pool.py:614-635`; the queue has `popleft`, `popleft_n`, `remove`, `append`, `prepend_n` and `append_n`, `vllm/v1/core/kv_cache_utils.py:232-366`). Generation blocks are never hashed (`skip_cache`, `engine_boundary.py:198`), so R's blocks go to the head, but with a block the synchronous server never took among them. The queue's order then differs from the synchronous server's, so later block ids can differ, and so do the incarnation counts (`pool_incarnations.py:8-19`) and the journal (`window_rows.py:78-108`). Tokens are unaffected. | Exact: after the stop decision, put R's speculative blocks back at the head of the queue in their former order (the queue's existing `prepend_n`; no new method), take back a block freed early (`remove`) into R's row, and decrement the incarnation counts. This is exact at b = 1 only. At b > 1, requests that allocate after R in the same `schedule(N+1)` have already taken shifted ids, and their block tables are in flight. Relaxed: compare evidence up to a consistent renaming of block ids and incarnations. That is an offline-verifier change with no runtime rollback, and it also covers b > 1 (a decision for the maintainer, section 6). | M (exact) / none (relaxed) |
| H2 | Engine records (`EngineRpcBoundary`: `scheduled_cursor`, intervals, `native_tables`, `accepted_rows`, seen ids, Mamba snapshots; Gemma's `window_rows` records) | Advanced for R by `schedule(N+1)` (`engine_boundary.py:273-319`). | R is terminal, so restore its record from a copy taken before `schedule(N+1)` (or prove that the completion never reads it). | S |
| H3 | Worker allocation owner (claims, `native_rows`, `last_tables`, fastpath3's steady caches) | Advanced by R's `allocation_batch` entry for step N+1. | C: hold the step's batch and send it without the stopped requests' entries, so nothing needs rolling back. S: truncate at completion. | S (C) / M (S) |
| H4 | Qwen and window lease state (`state.active`, cursor, `latest_hybrid`, `page_vectors`, `forwards`/`sequence`, execution states) | Advanced by one token. Also the pipeline itself: `enter_execute` requires `idle`/`joined` (`qwen.py:104-113`), `leave_execute` requires every lease closed (`115-142`), and `worker_interval` requires `active is None` (`chunk_protocol.py:38-44`). | C: R's row gets no lease. A new `issued` state between `execute_model(N)` and `before_forward(N+1)` lets `finish_step(N)` run at N+1's pre-launch point, and the guard's pre-execute drain (`generation_guard.py:35-38` → `remote_ra_growing_worker/tensors.py:474-480`, `torch.cuda.synchronize`) becomes a connector check. S: lease N+1 opens while lease N is only issued, and the completion accepts a worker one token ahead and cuts it back (cursor, vector, carry, frames, `worker_backend.py:350-354`). | M (C) / L (S) |
| H5 | Transcript and commit (`after_drain` → `commit_transcript`, `composition.py:207-212`) | Nothing. It commits vLLM's output ids, and vLLM drops the N+1 token of a finished request (`scheduler.py:1723-1732`). | None needed. | none |
| H6 | Native evidence and `runtime.jsonl` (frames, `kv_pools.free_blocks_by_pool` after the drain; also with `--deferred-completion-io`) | (a) S only: one more `allocation_update` frame for R. (b) Free counts are read in `update_from_output(N)`, after `schedule(N+1)` allocated for *all* running requests. They differ at b > 1 even with H1's rollback. | (a) C: the entry is held, so there is no frame. (b) Report the counts as of the step boundary: add back what `schedule(N+1)` took per pool, counted in `pool_incarnations`' wrapper. | S |
| H7 | Capacity ledger and every pass-boundary operation: admission, Goal resume, phase-1 settle, cancel and abort, lifecycle tick, idle release and readmission, session close, shutdown | While a batch is in flight, the free counts include step N+1 and lack N's frees. Leases are open. A separate phase-1 forward through the runner would disturb the async runner's input batch. | **Pass drain**: before any of these, finish the in-flight batch (read its output, `finish_step`, `update_from_output`) without scheduling another, then proceed exactly as the synchronous engine does. vLLM's aborts queue is drained the same way. The gate's decisions are then those of a synchronous server one step later. Each drain costs one step's overlap. At 16 sessions with replies of about 500 tokens there are about 16-32 drains per about 500 steps (one admission per session-turn, plus a Goal resume for `goal_query`), so the gain shrinks by about 3-6 %. | M |

### 3.3 Avoiding the extra row: stop prediction

| Stop | Known before `schedule(N+1)`? | Use |
| --- | --- | --- |
| `max_tokens` (the reply bound) | Yes. Stock already skips the step (`scheduler.py:490-504`). | Keep as is. |
| Context limit (`max_model_len`) | Yes (`num_tokens + 1 ≥ max_model_len`). | Add to the same rule in the overlay. S. |
| Action marker `</mswea_bash_command>` (XML protocol only, `composition.py:430-450`, `mswea_protocol.py:117-147`) | Exactly, from the text through N−1. A stop at N needs that text to end with a non-empty proper prefix of the marker, or one vocab token to hold the whole marker (check once at startup). | Hold flagged rows out of step N+1, or flag them for the void or snapshot. This reduces how often D1 applies; it does not replace it. |
| EOS, pad and stop ids (all chat-text and kv-session stops, some XML turns) | No: they are known only after N's sampling, on the device. | No host-side avoidance exists. Any design that launches N+1 before reading N's tokens needs D1(a) with true stops (C) or D1(b) (S). |

Avoiding speculative allocation altogether (not scheduling a request in a step that would allocate) delays
1 step in 16 of every request. That costs about as much as async gains: one step in 16 is 6.25 % more steps per
request, against stock's 5-8 % per step (section 4.1).

## 4. Options

### 4.1 The value, measured and estimated

Steady decode ITL, stock vLLM 0.24.0 synchronous against asynchronous scheduling (ms; window `ok` = at
least about 150 steps, `short`/`missing` = 23-119 steps; recomputed from each job's `itl-by-batch.csv`):

| (L, b) | FA3, job 3429022: sync → async | Triton, job 3461668: sync → async |
| --- | --- | --- |
| 2K b=1 | 18.59 → 17.05: −1.53 ms, −8.25 % (ok) | 18.61 → 17.14: −1.47 ms, −7.92 % (ok) |
| 20K b=1 | 18.95 → 17.40: −1.55, −8.20 % (ok) | 20.29 → 18.80: −1.49, −7.35 % (ok) |
| 117K b=1 | 20.70 → 19.16: −1.54, −7.44 % (ok) | 28.99 → 27.51: −1.48, −5.10 % (ok) |
| 2K b=4 | 19.26 → 17.55: −1.71, −8.88 % (missing) | 19.69 → 17.68: −2.01, −10.19 % (ok) |
| 20K b=4 | 20.53 → 18.87: −1.66, −8.08 % (short) | 21.55 → 19.96: −1.60, −7.40 % (ok) |
| 2K b=16 | 21.36 → 19.40: −1.96, −9.17 % (missing) | 21.51 → 19.55: −1.96, −9.10 % (missing) |
| 20K b=16 | 26.76 → 25.06: −1.70, −6.33 % (missing) | 27.11 → 25.40: −1.71, −6.32 % (missing) |
| W2 (SWE-like) b=16 | 29.48 → 28.06: −1.42, −4.80 % (missing/short) | 30.94 → 28.99: −1.94, −6.28 % (missing) |

**Our runtime (FA3, job 3429022).**
- Steady ITL with the three fast paths, no profiler: 19.40 ms (2K b=1), 24.85 (2K b=16), 31.88 (W2 b=16).
  That is +2.35, +5.45 and +3.82 ms against stock async (on Triton +2.32 to +5.22 ms, job 3461668).
- The step profile of the fast3 server (with the profiler on) gives, per step:
  - b=1: GPU forward 16.2 ms and host 3.4 ms. The host work is schedule 0.18, vLLM's preparation 1.79
    (`update_states`, `prepare_inputs`, attention metadata), `worker.before` 0.18, after-forward 0.08,
    sampling 0.67 and update 0.05.
  - b=16, 2K: forward 18.6 ms and host 9.0 ms. The host work is schedule 1.20, preparation 2.61,
    `worker.before` 1.14, after-forward 1.94, sampling 0.90 and update 0.19.
  - The other rounds have 6.3-6.4 ms of host time at b=16 (6.37 ms at 20K, 6.27 ms at W2).

**Per turn (job 3461661, Qwen Terminal-Bench dry runs, FA3).** Decode including stalls is 80.9 % of
17.73 s per request for `goal_query` and 98.3 % of 12.67 s for `observation_query`; the per-request ITL is
29.7 and 29.0 ms. A step gain g shortens the server time per turn by at most g × that share, because the
stalls behind other sessions' admissions inside the residual do not shrink:

| Step gain | `goal_query` per turn | `observation_query` per turn |
| --- | --- | --- |
| 5-8 % (stock's measured gain) | 4.0-6.5 % (0.72-1.15 s) | 4.9-7.9 % (0.62-1.00 s) |
| F: 1-4 % | 0.8-3.2 % | 1.0-3.9 % |
| C: 5-11 % | 4.0-8.9 % | 4.9-10.8 % |
| S: up to 12-22 % | up to 9.7-17.8 % | up to 11.8-21.6 % |

The F, C and S rows are estimates from the profile, not measurements (sections 4.2-4.4).

### 4.2 Option F: lookahead scheduling (exact, small)

**What it does.**
- `schedule(N+1)` runs while forward N runs. Execute N+1 starts only after `update_from_output(N)`.
- The stopped requests are removed from N+1's scheduler output before the execute. The allocations they
  got are rolled back (H1, H2).
- The GPU op order is the synchronous server's.

**Scope.**
- Overlay `core.py`: a step function and the scheduler-output edit, about 150 lines.
- The pool rollback (`kv_cache_utils.py` / `block_pool.py`, with the queue's existing `prepend_n` and `remove`) plus
  the incarnation and journal rollback, about 150-250 lines.
- The gates and flags of section 1, about 150 lines.
- In total about 500-700 runtime lines and about 500 test lines.

**Risk.** Low. No device change; the schedule's host state is the only new owner.

**Tests.**
- CPU: the stop removal against "never scheduled" on the MultiHarness fakes; the pool rollback against
  "never allocated" on random sequences of both pools and Gemma's.
- GPU: token- and evidence-exact against the synchronous server at b = 1 (bitwise) and at b = 16
  (identical whenever the step compositions are).

**Gain.** Only `step.schedule`: 0.18 ms at b=1, 1.2-1.26 ms at b=16 (profiled). That is under 1 % at b=1
and about 4 % at b=16, about 1-4 % of a turn.

### 4.3 Option C: prepare ahead, launch after the stop decisions (exact; recommended form)

**The step.**
1. vLLM async on: `schedule(N+1)`, `execute_model(N+1)` and vLLM's native preparation run while forward N
   runs.
2. Our `before_forward(N+1)` hook (`gpu_model_runner.py:4605`) first resolves N:
   1. Wait for N's sampled-id copy event.
   2. Decide the true stops with the same predicate `update_from_output` will apply (`check_stop`, then
      `MarkerStop`; pure, nothing mutated).
   3. Run the deferred `finish_step(N)`, still before N+1's replay (D4).
   4. Void the stopped rows (D1(a), D3) in the persistent buffers.
   5. Apply the held `allocation_batch` without their entries (H3).
   6. Open the leases of the continuing rows, and launch.
3. `update_from_output(N)` then runs during forward N+1, so completions see untouched slots and tails.

**Rollback.** Host rollback is needed only for vLLM's speculative allocation of the stopped requests (H1,
exact or relaxed) and our engine record (H2). Every pass-boundary operation drains first (H7).

**Refused in version 1.** `--phase1-in-step` and `--gdn-device-handoff`, as Gemma does already.

**Scope (rough lines).**

| Part | Files | Lines |
| --- | --- | --- |
| Flag, pins, the synchronous-only gates of section 1 made conditional, launcher | `serve.py`; `native_worker_registration.py`; `bootstrap_contract.py`; `startup_runtime.py`; `policy_runtime/bootstrap.py`; `private_authority/protocol.py`; `positions.py`; `plan.py`; `qwen_semantics/hooks.py`; `remote_ra_step_profile.py` | 200-300 |
| Async step function with pass drains (registered add, resume, settle, ticks, aborts, evict, close, shutdown) | `core.py` overlay, a new `remote_ra_live_v2/async_steps.py` | 250-350 |
| Placeholder-aware allocation; the batch held until the stop decisions | `allocation_guard.py`, `chunk_protocol.py`, `engine_boundary.py`, `decode_fastpath2.py`, `window_rows.py` | 150-250 |
| Deferred `finish_step`, the `issued` state, the guard without its drain | `generation_guard.py`, `worker_composition.py`, `qwen.py`, `window_lifecycle.py` | 250-350 |
| Resolve hook: token wait, stop predicate, void (3 GDN groups and 1 FA group; Gemma's 6 groups; eager and FULL) | new `remote_ra_live_v2/async_void.py`, `hooks.py`, `mswea_protocol.py` | 350-500 |
| H1 rollback (exact) and H2 | overlay `kv_cache_utils.py` / `block_pool.py`, `pool_incarnations.py`, `window_rows.py`, `engine_boundary.py` | 250-400 |
| H6 counts as of the step boundary; deferred-IO check | `remote_ra_http/evidence.py`, `deferred_completion.py` | 50-100 |
| **Runtime total** | | **about 1,500-2,250** |
| Tests | below | about 1,200-1,800 |

**Risk.** Medium. The parts that decide correctness can only run on the GPU:
- the void inside the persistent buffers under FULL graphs, for both attention backends and every GDN
  kernel path;
- the deferred join on the stream;
- the stop decision taken twice, which must agree.

The CPU fakes imitate vLLM's scheduler and runner; they do not run them.

**Tests.**
- CPU, unit level:
  - the stop predicate equals `update_from_output`'s decision on random outputs, EOS and marker cases,
    with the pinned tokenizer;
  - the void writer on eager and FULL metadata fakes;
  - the pool rollback equals "never allocated" on random allocation, free and touch sequences (FA, Mamba,
    Gemma pool 1, the journal).
- CPU, an async MultiHarness: a fake runner with device-side tokens and a copy event, and the batch-queue
  order with pass drains.
  - Setups: Qwen and Gemma fakes, fast paths 1-3, `--deferred-completion-io`, `--admission-overlap 16`,
    `--release-idle-pages`.
  - Stop cases: C′ % 16 == 0, (C′+1) % 16 == 0, C′ % 784 == 0, marker, EOS, `max_tokens`, cancel and
    abort in flight.
  - Pass: replies, plans, attended Pages, carry digests and evidence equal the synchronous fakes at b = 1,
    and at b > 1 whenever the compositions are equal.
  - Identity probe: with the flag off the launch, the startup config and the evidence are unchanged.
- GPU, ghx4 batch, thinking off:
  - G1: the strict check, like g1c. One session of the stage-0 XML append session, 11 turns, eager and
    `--cuda-graphs full`, both profiles, async against sync on the same node. Tokens, selections, Page
    vectors, carry digests and native files must be byte-identical (relaxed contract: equal up to the
    block-id renaming).
  - G2: the spd3 decode replay (2K, 20K, 117K, W2; b = 1-16; FA3 and Triton), our async and sync next to
    stock sync and async. Report ITL and reply parity within stock's own async-vs-sync spread (202 of 243
    replies identical in job 3429022).
  - G3: a TB16 or SWE16 dry run for admissions, drains, idle release and the counters (voided rows,
    rollbacks, drains).

**Gain (estimate, not measured).** At least stock's 1.5-2 ms if what C overlaps is what stock hides. By
the profile (`step-profile.csv` of job 3429022, profiler on; the step profiler itself costs about 0.2-0.5 ms per
step), C overlaps schedule, preparation and update: about 2.0 ms at b=1 (about 10 % of the step) and 3.5-4.0 ms at
b=16 (W2 3.52, 20K 3.61, 2K 4.01 ms: 11-15 % of the profiled step of 31.91, 30.19 and 27.59 ms). The work that stays
serial is the after-forward work, `worker.before`, the sampling wait and the void: about 1 ms at b=1 and 2.0-4.0 ms
at b=16 (W2 1.98, 20K 2.01, 2K 3.98 ms, the void not included). The overlap bounds the gain from above. This note's
estimate of 2026-10-30, kept as recorded and not measured (G2 has not run), was about 5-11 % per step and 4-11 % per
turn. It would put ours near stock async at b=1 and near stock sync at b=16.

### 4.4 Option S: stock-order pipeline (the most gain, the most risk)

**What it does.**
- N+1 is launched before N's tokens are read, as stock does.
- Qwen: D1(b), a snapshot of the slots of rows that may stop. Gemma needs no device change (D3, D5).
- The lease model is fully pipelined (H4 large).
- Host reads of device tensors leave `before_forward`: `positions.py:47-57`, the compaction's
  device-to-host copy.
- The completion truncates a worker that is one token ahead.
- H1, H2 and H6 as in C, plus R's extra frame.

**Scope.** About 3,000-4,000 runtime lines and about 2,000 test lines. GPU memory of 154 MB per snapshot
row, which is tight next to 25-32 GiB of KV and the rewriter.

**Risk.** High. False positives cost only a copy, but an overflow of the snapshot area needs a fallback
that rebuilds the session. Every after-forward check moves later.

**Tests.** As in C, plus snapshot overflow and the carry taken from the snapshot.

**Gain.** Up to our gap to stock async: 2.3-2.4 ms at b=1 (about 12 %) and 3.8-5.5 ms at b=16 (12-22 %),
less the snapshot copies and the drains.

### 4.5 Gates only (the form described for the collaborator's fork): not an option

Section 0, item 5, and section 2 describe what our code does. The result is either a completion failure at every
EOS or marker stop, or a silent wrong carry, depending on the completion check. The joins stay in place
(H4), so the gain is about 0.5-1 ms. Their runs are not evidence for async on this runtime.

## 5. Recommendation

1. Build async only as option C, opt-in (`--async-scheduling`; default off, with the identity probe that
   leaves everything unchanged without it), for both profiles. Refuse `--phase1-in-step` and
   `--gdn-device-handoff` with it.
2. Order the work so that each stage can be checked on the CPU before the GPU:
   1. the gates and the flag;
   2. the deferred `finish_step` and the drain-free guard, with async still off (the evidence must stay
      byte-identical);
   3. pass drains;
   4. placeholder-aware allocation and the held batch;
   5. the resolve hook and the void;
   6. the H1 and H2 rollback;
   7. the async MultiHarness;
   8. GPU G1 (b = 1, both profiles), then G2 and G3.
3. Merge to main only after G1 and G2 pass, with approval (as for every runtime change).
4. Consider S only if C's measured G2 gain falls well short and parity with stock async is required.
5. Do not merge or benchmark the gates-only form. Tell the collaborator what section 2 shows.

## 6. What the maintainer had to decide

1. **Whether to build it.** About 5-11 % per decode step (estimated; stock measured 5-10 %), about 4-11 %
   of the server time per turn. The cost is about 1,500-2,250 runtime lines plus 1,200-1,800 test lines
   for C, and three ghx4 batch jobs (G1-G3).
2. **Which option:** C (recommended), F (exact, about 1-4 %), or S (up to 12-22 %, high risk).
3. **Evidence contract.**
   - Exact: H1 rollback, and native files byte-identical to the synchronous server at b = 1. At b > 1 block
     ids shift anyway (H1).
   - Relaxed: tokens, selections, Page vectors and carries exact; block ids and incarnations equal up to a
     consistent renaming; free counts reported as of the step boundary (H6). This saves about 250-400
     runtime lines and moves the comparison into the offline verifier.
4. **Profiles and refusals:** both profiles in version 1 (recommended), or Qwen first. Confirm that
   `--phase1-in-step` and `--gdn-device-handoff` stay refused with async.
5. **Benchmarks and paper.** Whether Table 1 servers switch to async once G1-G3 pass. The stock control
   already runs stock async. Paper latency stays "optimized runtime against stock", from the GPU-validated
   commit only.
6. **The collaborator's fork:** whether to send them this note (section 2), and whether their runs are to
   be set aside.
