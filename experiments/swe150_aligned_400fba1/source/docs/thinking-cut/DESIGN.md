# Thinking ON, option (a): cut the last reply and re-prefill (design, 2026-10-30; form E is `--thinking-history strip`)

English | [简体中文](DESIGN.zh-CN.md)

Design of 2026-10-30 (branch `thinking-cut` at `cf2b28d`); form E is on `main` (`8ca8511`, with stage 1 `94888e1`;
merge `34f274b`) as the opt-in `--thinking-history strip`. Its GPU runs and their verdicts are in
[docs/status.md](../status.md) and [History](../history.md#history), the limits of thinking on in
[docs/known-limits.md](../known-limits.md) (sections 1 and 4.3); what the GPU checks G1 and G2 measured is in
section 6. Below: form E as built (2026-10-30), then the design.

## Form E as built (2026-10-30)

Form E ("commit at the cut") is implemented on branch `thinking-strip` (over stage 1, `94888e1`) as
`serve.py --thinking-history strip` (with `--enable-thinking`; [docs/running.md section 6.9](../running.md#69-thinking-on-with-the-reasoning-dropped---thinking-history-strip); rules and attachments in
`remote_ra_live_v2/thinking_cut.py`). The GPU checks G1 and G2 (section 6) ran later (their results: section 6).
The sealed verifier (`perf-tb16/`) is unchanged: section 4.2's rules V1-V5 were built into a copy of it,
`perf-tb16-strip` (outside this repository), which verifies strip runs offline.
The maintainer's decisions of section 8: (1) form E, not L; (2) `--thinking-history strip` pinned per server,
chat-text on the Qwen profile only, refused on Gemma 4, kv-session and mswea-xml (without it stage 1's keep
mode); (3) the next turn's stable end at X + 3, CANDIDATE_RANGE unchanged; (4) the dropped token ids in the
evidence (`truncate.dropped`); (5) `--gdn-device-handoff` and `--phase1-in-step` refused with strip in v1;
(6) a re-prefill above one native forward fails the turn (`PHASE1_TOO_LONG`, unchanged); no per-request
hashing beyond a turn without the option.

As built, against the sections below:
- 3.1: the transcript (`chat_text_protocol.ChatTextTranscript(strip=True)`) drops every reasoning field a
  client sends (the reference drops R), so `extends` compares role and content only. The stable end is
  X + K with K = |encode('<|im_start|>assistant\n')|, the header's shared head: 3 on the pinned Qwen3.6
  tokenizer (checked in `test_thinking_cut`), 11 in the character-tokenizer fakes. A head mismatch refuses
  the request (HTTP 500 `STRIP_REPLY_HEAD`, nothing changed); the pinned BPE never produced one (every
  content tried, a `\r` after the last `</think>` included).
- 3.3: no new RenderAssignment field. The plan carries `commit_end` (and the seal rule `floor(X/P)`), set
  at `adapter.reserve`; the worker checks it equals the assignment's observation end e at reserve. The
  prepare payload is unchanged.
- 2.3: `worker_backend.PositionedCarry` with `hold_carry`, `hold_phase1_carry` (held at finish, the
  phase-1 state's payload and identity token) and `capture_split_carry`, committed by
  `drain(..., positioned=True)`; a strip backend refuses any other completion. A split capture is also
  recorded as the latest state at X, so a next chunk that crosses an M block from X relocates it (only
  possible when the header is longer than the distance from X to the next M multiple: with Qwen's 5-token
  header when 1 <= N mod M <= 4, as in G1's cold probes with N mod 784 = 1, 3 and 4; the 19-token header of the
  fakes does it too).
- 3.4: `EngineBridge.scheduler_split` (local turn 0 only) after vLLM's aligned split, as designed. Reading
  `vllm/v1/core/sched/scheduler.py` `_mamba_block_aligned_split` and `vllm/v1/worker/mamba_utils.py`
  `preprocess_mamba` / `utils.mamba_get_block_table_tensor`: the chunk that ends at X uses the align
  block of X - 1, and the next chunk (cut at the last M multiple by vLLM's own rule when X precedes it)
  continues in the same block, as chunks of the last partial block do today; no approximation is needed on
  the CPU side, and the state's exactness on the GPU is G1.
- 4.1: `truncate.dropped.think_end` is the index of the one-token `</think>` id in the generated ids (null
  when absent or when `</think>` is not one token), with no decode; the chat-text runtime row's `cut` and
  `reprefill` are in its `thinking` record; the row adds `truncate` counts and `stage_b.phase2_end`.
- Tests (`test_thinking_cut.py`): T1 (template, pinned tokenizer), T2 (the 178 ATIF trials with reasoning
  inserted: 1,684 requests, every prompt the stock render), T3 (engine and worker fakes: commit at X, tail
  none/live/candidate, the local turn 0 split under one-Page, vLLM-aligned and whole-prompt chunks, cancel
  in phase 2, rebuild after a cut; the dense recomputation of every selection input), T4 (the decode fast
  paths, `--deferred-completion-io`, `--admission-overlap 4`, `--release-idle-pages` under pressure with
  goal_query; identity probes with the option off), mutations. T5 (the verifier) is not done: the verifier
  is sealed.

Date: 2026-10-30. Branch `thinking-cut` @ `cf2b28d`. Builds on stage 1 (branch `thinking-on`: server-side reasoning
split, MarkerStop after `</think>`, append when the client sends R back), which was not in the tree yet (it was
merged with form E: `94888e1`); section 3.0 lists what this design takes from it. Read first:
[docs/history/thinking-on-analysis.md](../history/thinking-on-analysis.md) (the 10-30 addendum, sections 2-3).

Paths are relative to `source-overlay/` unless they start with a top-level file (`serve.py`, `README.md`) or with
`perf-tb16/` (the sealed offline verifier, `development/exp-decouple-pages/perf-tb16/` of the project workspace, not in this repository).
Line numbers are those of `cf2b28d`. Short names: a bare module name is under `remote_ra_live_v2/`, except
`adapter.py`, `ownership.py` and `contract.py`, which are under `remote_ra_native_growing/` (the live-v2
`contract.py` is written out); `tensors.py` is `remote_ra_growing_worker/tensors.py`, `evidence.py` and
`chat_text_route.py` are under `remote_ra_http/`, `scheduler.py` is `vllm/v1/core/sched/scheduler.py`.
When this note was written nothing was implemented and it decided nothing; section 8 lists the decisions (taken: see "Form E as built" above).

## 0. Summary

1. **The cut point is the observation end e_t, which on chat-text is the header start h_t.** The chat-text transcript
   renders `tokens = stream + observed + header` with span `(stable, stable + len(observed))`
   (`remote_ra_live_v2/chat_text_protocol.py:582-586`), so e_t = N_t - |H| exactly, at every local turn (local
   turn 0 too: o = e = stable_end, `remote_ra_native_growing/contract.py:248-249`). The tokens diverge 3 tokens later
   (h_t + 3, after `<|im_start|>assistant\n`), but the runtime has a GDN state only at e_t (phase 1 ends there), and
   GDN cannot be recomputed without a full forward that also writes the FA KV. So the cut is at e_t, and the 3
   header tokens are recomputed.
2. **The state at the cut already exists**: the phase-1 GDN state at e (`Phase1State`, `remote_ra_live_v2/worker_backend.py:49-62`,
   captured at `remote_ra_live_v2/native_probe.py:300-307`). Today it is dropped at the complete drain
   (`worker_backend.py:365`) and the carry is captured at C' instead (`qwen.py:349-352`, `worker_backend.py:261-269`).
3. **Recommended form E, "commit at the cut"**: under the stock-default template (`enable_thinking` true, no
   `preserve_thinking`; taken to be the collaborator's Vanilla on Terminal-Bench, not re-verified against the
   collaborator's repository), every assistant message before the last user
   message renders as `<|im_start|>assistant\n` + C, whatever the client sends back (template, assistant branch:
   the `<think>` block is rendered only with `preserve_thinking` or after the last user query). So on a server pinned
   to that history, *every* continuing request diverges at e_t. The completion can therefore commit the session at
   e_t directly: phase 1 is committed, phase 2 (header, R, C, EOS) is turn-local scratch. Nothing is sealed past the
   cut, so nothing has to be rolled back: no unseal, no Page release, no arena drop, no vector or index truncation, no
   idle-release interplay, no second carry (0 extra pinned bytes instead of ~154 MB per session), and the admission
   geometry (`turn_geometry`) is unchanged.
4. **Local turn 0** (fresh slot, rebuild, rebind) has no phase 1: its prefill must end a forward at e_0 (the existing
   `scheduler_split` hook, `remote_ra_live_v2/composition.py:249-252`) and capture the GDN state there.
5. **Re-prefill**: the next turn's phase 1 computes [e_t, e_{t+1}) = `<|im_start|>assistant\n` + trim(C) + `<|im_end|>\n`
   + the user message, on the resident set and the live tail (page e_t//P, valid_end e_t), from the committed carry
   (the state at e_t). Today's phase 1 is 2 + |U| tokens (EOS, `\n`, the user message); it becomes |C| + 5 + |U|.
6. **Alternative L, "roll back at the next request"** (the design ANALYSIS sketched) keeps today's completion and
   undoes it when a request diverges. It is needed only if one server must serve both histories (strip and keep) in
   one session. It costs a second carry per session (~154 MB pinned), the per-owner rollback of section 1 and about
   twice the code.
7. **Effort (E)**: about 550-900 runtime lines in ~14 files, 800-1,200 test lines, 150-250 verifier lines in the sealed
   harness; about 1-1.5 weeks plus two ghx4 batch jobs (G1 token alignment against stock thinking-on eager, G2 TB16
   dry runs). L: about +700-1,200 runtime lines, 2.5-3.5 weeks. Main risks: the non-M-aligned prefill split at local
   turn 0 under vLLM's Mamba align mode (GPU check), and the verifier rework.

## 1. The cut point and what each owner holds past it

### 1.1 Token geometry (Qwen3.6, thinking on, chat-text, P = 16)

Header H = `<|im_start|>`, `assistant`, `\n`, `<think>`, `\n` (5 tokens). Turn t: prompt T_t of N_t tokens,
e_t = N_t - 5, phase 1 [C_t, e_t), phase 2 [e_t, C'_t) with C'_t = N_t + g_t - 1. The model writes
`R \n</think>\n\n C <|im_end|>`.

| Quantity | Value |
| --- | --- |
| Cut X_t (committed end of turn t) | e_t = N_t - 5 (the header start) |
| Token divergence with T_{t+1} | X_t + 3 (`<think>`) |
| Dropped branch | [X_t, C'_t + 1): `<|im_start|>assistant\n<think>\n` + the g_t generated ids |
| Sealed end after the cut | Q_X = floor(X_t/P)·P |
| Tail after the cut | page X_t//P, start Q_X, valid_end X_t (none when X_t % P == 0) |
| Next turn | C_{t+1} = X_t, next_token = T_t[X_t] = `<|im_start|>`, o_{t+1} = X_t + 3 (section 3.2) |

Every Page below X_t//P holds only committed tokens; it was closed either in turn t's phase 1 (its vector is a
phase-1 vector, `phase1_coordinator.py`/`plan_turn` at `query_selection.py:1054-1059`) or earlier. Every Page from
X_t//P on was produced by turn t's phase 2. **The cut is exactly "undo turn t's phase 2"** (or, at local turn 0, the
part of the prefill past e_0).

### 1.2 Per owner: what turn t's completion leaves today, form E, form L

| Owner | Today after turn t (file:line) | E: what the completion commits instead | L: what a rollback at turn t+1's admission must undo |
| --- | --- | --- | --- |
| **GDN carry** (device state, host copy) | Captured at C' from the running slot (`qwen.py:349-352` → `worker_backend.py:261-269`; committed `worker_backend.py:346-363`); the phase-1 state at e dropped (`:365`) | The state at e becomes the carry (section 2); no C' capture | Keep the phase-1 state as a second carry (+154 MB pinned per session); at the cut, make it `backend.carry[session]` so `prepare` (`allocation_owner.py:233-244`) and `verify_carry` (`checkpoint.py:27-34`) match the rolled-back mirror |
| **Tail page across the cut** (page X//P) | Sealed at completion when C' ≥ (X//P + 1)P: arena capture (`allocation_owner.py:501-507`), pinned as a resident Page (`allocation_owner.py:520-522`, `ownership.py:263-291`), phase-2 vector (`qwen.py:413-417` → `adapter.py:457` → `query_selection.py:1121-1132`) | Committed as the tail: `owner.commit(tail=(X//P, Q_X, X))` (`ownership.py:265-269` already accepts any page ≥ L/P); never sealed, never captured, no vector | Unseal: move its blocks from `owner.sessions[ns]` to `owner.tails[ns]` as `TailPin(X//P, Q_X, X)` (`ownership.py:22-27`), set the worker tail (`allocation_owner.py:518-519` shape), drop its catalog entry (engine `adapter.py:465`, worker `allocation_owner.py:542`; the arena slot returns through the `PagePayload` finalizer, `remote_ra_growing_worker/tensors.py:321-332`), drop its vector, rebuild `page_index[sid]` (the index refuses a changed vector for a page id, `dram_kv/page_vectors.py:245-261`), truncate `page_columns` (`fast_selection.py:90-128`); if the page was idle-evicted (`idle_release.py:283-308`), first restore it from its DRAM copy into a fresh block |
| **KV slots [X, (X//P + 1)P) of the tail block** | Header/R KV | Dirty slots past valid_end, allowed (`ownership.py:26`); phase 1 rewrites from X | Same |
| **Pages sealed after the cut** ([X//P + 1, Q'/P)) | GPU pins (`resident2`), arena copies, vectors, engine and worker catalogs (`adapter.py:465-467`, `allocation_owner.py:542-545`), possibly `s.evicted` (`adapter.py:85`) | Never sealed: their row blocks are freed with the request at native free (`ownership.py:298-318` checks the counts) | Free the session pins (`pool.free_blocks`, `block_namespaces` pop, as `ownership.evict` 347-360), drop catalog entries / arena slots / vectors / index / columns, filter `s.evicted` (the readmit refuses pages ≥ the tail page, `ownership.py:372`) |
| **Running GDN slot, phase-2 Mamba blocks** | Freed by vLLM at request free | Same | Same |
| **Block pool, incarnation counters** (`pool_incarnations.py:8-19`) | Monotone count per block id | Nothing: freed blocks re-enter the free queue; the next allocation counts a new incarnation | Nothing to rewind (counters are identities, not state); the free-queue order differs from a server that never sealed R, as after any request |
| **Engine session mirror** (`adapter.py:66-93`, set at `:459-469`) | sealed/computed tokens to Q'/C', next_token = full[C'], tail at C', resident, boundary | `computed = full[:X]`, `sealed = full[:Q_X]`, `next_token = full[X]`, tail at X, resident = selected ∪ [L/P, Q_X/P), boundary = carry at X | Rewrite the same fields at the cut (a new adapter method beside `reset_session`, `adapter.py:229-243`) |
| **Worker session mirror** (`allocation_owner.py:173-176`, set at `:542-545`) | catalog [0, Q'/P), resident, tail, computed/sealed ends, carry digest | Set at X in `_complete` | A new worker RPC `cut` that rewrites it (and must run before `prepare`, which compares it with the assignment) |
| **Selection tables** (`query_selection.py:360-390`) | `vector_catalog` = [0, Q'/P) and `catalog_version` (`:1121-1132`); `page_index`, `page_columns`, `compact_cache`; `turn_ends` (`:1116`); `previous_active` / `ever_selected` (diagnostics, every page < n_t) | `publish_completion` gets the phase-1 vectors and the phase-2 vectors of `closed_pages(S2, X)` (empty at local turn > 0), sealed end Q_X; nothing else changes | Delete catalog keys ≥ X//P, bump `catalog_version`, rebuild `page_index[sid]`, truncate `page_columns`; `turn_ends` and diagnostics need nothing (their entries are < e_t) |
| **Transcript** (`chat_text_protocol.py:617-634`) | `stream = prompt + generated + boundary` | `stream = prompt[:X]`; the turn record keeps C, R and the dropped ids (section 3.1) | Keep the stream; in `plan`, cut it at the last turn's header start |
| **Goal records** (`dram_kv/goals/maintained.py:134-148`, called from `query_selection.py:861-929`) | Built at the next admission from `previous_action()` (`chat_text_protocol.py:636-638`, read at `query_selection.py:951, 971, 990, 1040`) | Nothing: the render (and with it the cut decision) precedes `begin_turn` (`private_endpoint.py:163-179` before `:237-240`); with stage 1 the previous action is C | Same: nothing to undo, provided the cut is decided at render |
| **Evidence** | Native file and runtime row (section 4) | New fields; no rollback | New fields; no rollback |

## 2. Where the state at the cut comes from

### 2.1 Sources

| Turn | Source of the GDN state at X = e | Today | E |
| --- | --- | --- | --- |
| Local turn > 0, default | `Phase1State`: a pinned host payload from `io.capture` (`worker_backend.py:183-187`, `tensors.py:513-534`), restored into the phase-2 slot at phase 2's first step (`worker_backend.py:208-214`) | Dropped at the drain | Promoted to the committed `CarryState` (the same payload object and identity token, no copy); the C' capture (`qwen.finish` → `_final_state`) is skipped |
| Local turn > 0, `--gdn-device-handoff` | `HeldPhase1State`: no host bytes, the probe slot until phase 2 copies it on the device and releases it (`worker_backend.py:173-182`, `195-207`) | Released after the copy | At capture time also issue a D2H of the probe slot into the session's free carry buffer (the buffer `_capture_carry` would use at C', `worker_backend.py:278-293`), before `release_probe`; the complete drain joins it (`:337-339`). Same bytes as today's C' capture, earlier |
| Local turn 0 | None (phase 2 runs [0, N) from a zero state) | n/a | Split the prefill at e (section 3.4) and capture the state at the end of the forward that ends there (non-handoff: `io.capture`; handoff: issued into a carry buffer) |

The local-turn > 0 state is the one every continuing turn's phase 2 already starts from, so its correctness is what
today's runtime already relies on; E only commits it.

### 2.2 Costs of the alternatives

| Option | Pinned host | GPU | Extra compute | Notes |
| --- | --- | --- | --- | --- |
| **E: commit the state at e** (recommended) | 0 (it replaces the C' carry) | 0 | 0; saves the 154 MB C' capture per turn (with the hand-off the same D2H moves to phase-1 end) | Needs the strip history pinned per server |
| L: keep a rollback carry next to the C' carry | +154 MB per session (`GDN_CARRY_BYTES`, `serve.py:95`; the budget `pinned_host_bytes` counts `sessions*carries`, `serve.py:156-167`, `:505`: carries + 1). 20 sessions: +3.1 GB of the 128 GiB budget | 0 | 0 | Plus the whole L column of section 1.2 |
| Recompute: keep the carry at C_t instead and rerun [C_t, e_t) | 0 with the handoff's second buffer (it still holds the previous carry after a commit), +154 MB otherwise | 0 | +\|obs_t\| tokens per turn | The cut moves to C_t: the Pages of [C_t//P, e_t//P) (phase-1 Pages, valid) must be rolled back or rewritten in place (their vectors and DRAM copies could then differ from the recomputed KV); rejected |
| Keep the state in a device Mamba slot | 0 | +154 MB per session (a third slot per session in the Mamba pool, `kv_layout.py`, K = 2S + 1) | 0 | 2.5-3 GB at 16-20 sessions next to 25-32 GiB of KV; rejected |
| Carry at the start of the last reply's header | = E: on chat-text the header start is e | | | |

### 2.3 The positioned carry (the reusable mechanism)

Today the committed carry is, by construction, the running slot at the request's cursor (`drain` requires
`latest[0] == computed_end`, `worker_backend.py:347-355`). E introduces a **positioned carry candidate**: a GDN state
at a named end X ≤ cursor with a named source, which the completion commits instead of the slot.

- Worker (`worker_backend.py`): `hold_candidate(identity, end, source, payload | issued buffer)`; sources `phase1`
  (the probe state, end == the lease end == the assignment's e), `split` (a capture at a forward end == X at local
  turn 0). `drain(complete, computed_end=X)` commits the candidate with that end, with its lineage proved
  (`phase1_restored` / `zero_verified`, `:355-356`) instead of `latest`.
- Engine (`adapter.complete`, `adapter.py:395-480`): the plan names `commit_end` X; the completion derives seal,
  tail, resident and vectors from X instead of C'; the receipt carries both `computed_end` (C', what was computed)
  and the carry's `computed_end` (X).

Async scheduling ([docs/async-sched/DESIGN.md](../async-sched/DESIGN.md)) can reuse both halves:
- Option S's D1(b) (a snapshot of a row's slot before step N+1) is a third source, `snapshot`, with end C'. Its
  "completion truncates a worker that is one token ahead" (H4) is the same "commit at X < cursor" path.
- Option C can use the same path as a fallback for a stop decided after the launch (instead of failing the request).
  As built, it does not: a stop async C did not predict fails closed (`async_sched.py`).
- On a strip server, async needs no GDN void at all: the committed carry and every committed Page precede the reply,
  so the extra row at C' writes only phase-2 scratch (D1, D3 and D4 do not reach committed state; H1, H2, H6 and H7
  remain). As built, async C keeps the void on strip servers too; the positioned carry is out of its reach
  (`async_sched.py`).

## 3. The re-prefill

### 3.0 What stage 1 provides (assumed interfaces)

- The thinking-on header `<|im_start|>assistant\n<think>\n` (5 tokens) on chat-text, and a reply bound for thinking
  (stage 1's `MAX_TOKENS_CHOICES` decision).
- The split of the generated ids into R and C (first `</think>`), the response `content` = C and `reasoning_content`
  = R (today both reasoning fields are blanked, `remote_ra_http/chat_text_route.py:178-180`).
- A transcript turn record whose `text` is C (so `previous_action()` and `extends` see C only) and that keeps R.
- Stage 1's keep path (append the exact ids when the client sends R back, `preserve_thinking`) stays as is; this
  design adds a second, pinned history mode.

### 3.1 The transcript in strip mode (`chat_text_protocol.py`, frontend and engine)

A startup-pinned mode `history = 'strip'` with reference kwargs `{enable_thinking: true}` (the stock default: no
`preserve_thinking`), refused for the Gemma grammar.

- `commit(prompt, generated)` (`:617-634`): `X = len(prompt) - len(header)`; require `prompt[X:] == header`;
  `stream = prompt[:X]`; append `dict(observation, text=C, reasoning=R, ids=prompt[X:] + generated, cut=X)`. No
  boundary is appended: the template's `<|im_end|>\n` is part of the next piece.
- `extends(messages)` (`:550-555`): unchanged rule (history equal, then assistant, then one user message;
  `content.strip() == text.strip()` with text = C).
- `plan(messages)`, continuing turn (`:583-587`): `A = encode('<|im_start|>assistant\n' + strip_content(c) +
  '<|im_end|>\n')` with `c` the client's echoed content and `strip_content` the template's own transform (`|trim`,
  then the part after the last `</think>` with leading `\n` removed); require `A[:3] == ids[:3]` of the last turn (a
  3-token compare, no hashing); `tokens = stream + A + U(obs) + header`; `stable = len(stream) + 3`;
  `span = (stable, len(stream) + len(A) + len(U))`. Each piece starts and ends at a special token, so it encodes
  alone exactly as inside a whole render.
- Local turn 0 and rebuilds: `render_chat` with the strip kwargs; past replies come out as C only, consistent with
  the stream.
- `check_template` (`:225-245`) gains a strip variant: a probe conversation with a reply `R</think>\n\n{"a": 1}`
  must render as `user_piece + A('{"a": 1}') + user_piece + header` (startup only).
- The re-prefill uses the client's echo, i.e. exactly stock's input; a different echo fails `extends` and rebuilds,
  as today.

### 3.2 Admission: unchanged geometry

`turn_geometry` (`contract.py:235-274`) holds as is: T[:X] == computed, T[X] == next_token (`<|im_start|>`),
C = X < o = X + 3 < e < N, n = floor((X + 3)/P) ∈ [Q_X/P, Q_X/P + 1] (P ≥ 3), and the tail flags follow from
X % P: no tail at 0, tail live at 1..P-4, tail candidate (completed by the 3 header tokens) at P-3..P-1. Phase 1 (`phase1_coordinator.py:135-289`) runs [X, e_{t+1}) on the resident
Pages and the live tail with the committed carry restored into the probe slot (`native_probe.py:279-288`), as for
any turn. Its vectors cover the Pages it closes, including page X//P when it fills.

**Why o = X + 3 and not after C.** Placing the stable end after the re-encoded reply would make n = floor(o/P) jump
past Q/P + 1 whenever |C| + 5 > P, which `CANDIDATE_RANGE` refuses (`contract.py:264`, `208-210`): the Pages of C
would be candidates computed by this very phase 1, with no CPU copy before their release (today only the single
tail candidate is captured at phase-1 end, `native_probe.py:308-312`, `allocation_owner.py:508-513`). With o = X + 3,
the re-encoded C sits inside the observation span: it is attended densely in phase 2 of turn t+1 (run Pages) and
becomes a candidate at turn t+2. The alternative (many phase-1 candidates) is decision D3.

**Length.** Phase 1 is one native forward (`PHASE1_TOO_LONG`, `phase1_coordinator.py:154`; 16,384 tokens on Qwen).
It grows by |C| + 3 tokens per turn. A turn whose [X, e) exceeds the forward either fails as today's over-long
observations do or plans a rebuild instead (decision D6).

### 3.3 Completion in strip mode

Engine (`adapter.complete`, `adapter.py:395-480`) and worker (`allocation_owner._complete`, `:478-557`), with
X = `plan["commit_end"]` = e (set at `adapter.reserve`, `adapter.py:300-319`, and in the worker assignment,
`remote_ra_live_v2/contract.py:216-235`, a new field), Q_X = floor(X/P)·P:

- seal [L/P, Q_X/P), plus the phase-1 tail candidate as today; catalog becomes [0, Q_X/P);
- tail = page X//P, valid_end X, its blocks from the row (`block(p)`, `allocation_owner.py:495-500`);
- resident = selected ∪ [L/P, Q_X/P); `owner.commit(keep_pages=resident, tail=(X//P, Q_X, X))`;
- vectors: phase 2 publishes `closed_pages(S2, X)` (empty at local turn > 0; [0, Q_X/P) at local turn 0), and the
  lifecycle skips the compaction of Pages ≥ X//P (`qwen.py:399-417`; with `--decode-fastpath2` inside
  `finish_step`), so `publish_pages` (`qwen.py:436-445`) checks that set;
- carry = the positioned candidate at X (section 2.3); mirrors as in section 1.2;
- the receipt reports `computed_end` C' and `carry.computed_end` X; the engine checks
  `C' == N + g - 1` as today and the seal/tail/vector sets against X.

### 3.4 Local turn 0

- `EngineBridge.scheduler_split` (`composition.py:249-252`, called after vLLM's Mamba-aligned split,
  `vllm/v1/core/sched/scheduler.py:556-560` and `:974`) returns `X - start` when `start < X < start + count` for a
  strip-mode request at local turn 0 (X = o = e = N - 5).
- `QwenAlignLifecycle._after_state` (`qwen.py:337-347`) captures the candidate when `lease.end == X`.
- The completion commits as in 3.3 with L = 0, S2 = 0.
- vLLM's align mode cuts the prefill chunks before the last M multiple to M multiples (`scheduler.py:383-401`); the
  chunk that ends at X does not end at one. Decode steps already end mid-block, so the runner should handle it, but
  this is the main GPU check (G1). When the header straddles an M boundary (1 <= N mod M <= 4), vLLM then splits [X, N) at
  that boundary by its own rule (`scheduler.py:402-408`), an M-aligned end the runtime already handles.

### 3.5 What the next turn's selection sees

- Candidates [0, n_{t+1}) = the committed history with every past reply as C (re-encoded), exactly the Vanilla
  prompt's history, cut into Pages; vectors of each Page from its single committed computation (phase 1, or the
  local-turn-0 prefill).
- Turn buckets (`turn_ends`, `query_selection.py:1116`; attribution `selection_contract.py:274-281`): bucket k+1 =
  [X_{k-1} + 3, X_k + 3) = C_{k-1}'s piece + obs_k + 3 header tokens, so a reply is grouped with the *following*
  observation (thinking off: with the preceding one). The recent set (the last two buckets, 64 Pages at P = 16,
  `selection_contract.py:82-83`) therefore covers C_{t-1}, obs_t, obs_{t-1}, while C_t is attended densely.
- Mandatory (ceil(initial_length/P), `selection_contract.py:268-271`), cold start (local turn 0, ≤ 8,192 tokens) and
  the Goal inputs are unchanged. `random_fill` is keyed by the request ordinal (unchanged); `random50` by the prompt's
  block hash (content, `private_endpoint.py:158-159`).

## 4. Evidence and the verifier

### 4.1 Records

- Native file (`remote_ra_http/evidence.py:198-235`, built on the engine thread, also under
  `--deferred-completion-io`, `remote_ra_live_v2/deferred_completion.py:465-496`):
  - `plan.commit_end` = X and `plan.seal_at_completion.rule` = `floor(X/P)`;
  - `truncate = {cut: X, dropped: {start: X, end: C' + 1, header_ids: T[X:N], generated_ids: G, think_end: k | null}}`.
    The generated ids must be recorded here: today the verifier reads them from the next prompt
    (`perf-tb16/tb_checks.py:1167-1175`), which no longer contains them;
  - for a continuing turn, `reprefill = {start: C, piece_end: C + |A|}` (the ids are in `prompt_token_ids`).
- Runtime row (`evidence.py:456-464`): `stage_b.computed_end` keeps coming from the complete frame's carry (now X,
  the committed end), plus `phase2_end` = the frame's `computed_end` (C') and `truncate` counts.
- Chat-text row (`chat_text_route.py:346-351`): `cut`, reasoning and content token counts.

### 4.2 Verifier rules (`perf-tb16/`)

The rule "every token is computed exactly once" (`perf-tb16/smoke_checks.py:808-815`, `README.md:997`) becomes
**"each stream version is counted once"**: the committed stream after turn t is v_t = T_t[:X_t]; every position of
v_t is computed exactly once in its lineage, and each turn's phase 2 is a dropped branch computed exactly once.

- V1, lineage: `T_{t+1}[:X_t] == T_t[:X_t]`; `C_{t+1} == X_t`, `Q_{t+1} == Q_{X_t}` (`check_stage_geometry`,
  `perf-tb16/tb_checks.py:1219-1226`, keeps reading `computed_end` / `sealed_end`, now the committed ends). The
  bare `:NNNN` references of V2-V4 and of the local turn 0 rule below are in the same file.
- V2, committed computation: phase 1 == [X_t, e_{t+1}) and `e_{t+1} == N_{t+1} - 5 == X_{t+1}`; at local turn 0 the
  prefill is [0, X_0) and then [X_0, C'_0) (the "one plain prefill" rule, `:1214-1217`, gains the split).
- V3, dropped branch: `truncate.dropped == [X_t, C'_t + 1)`, its ids == `T_t[X_t:N_t] + G_t`,
  `|G_t| == completion_tokens` and `phase2_end == N + g - 1` (replacing `:1230-1234`); EOS / length rule as today
  (`:1172-1176`).
- V4, re-prefill (replacing `check_chat_stream`'s append rule, `:1165-1185`): `stable == X_t + 3`;
  `T_{t+1}[X_t:X_t+3] == T_t[X_t:X_t+3]`; `T_{t+1}[X_t:X_t+|A|] == encode(A(content(G_t)))` with the pinned
  tokenizer; then one user message and the header.
- V5, accounting: per turn, the computed tokens are (|v_t| - |v_{t-1}|) in phase 1 (or the prefill before X_0) plus
  (|dropped_t| - 1) in phase 2 (vLLM never computes the last sampled token).
- Local turn 0 render: the past replies without any thought block (strip), not "with the empty thought block"
  (`:1162-1163`).
- `join_tb16.py` compares token counts; the completion tokens are unchanged, the prompt tokens shrink.
- `test_chat_text_replay.py:84-119`: the reference is the strip render.

## 5. Interactions

| Feature | Effect of E | Change |
| --- | --- | --- |
| `--decode-fastpath` 1-3 | Decode steps unchanged; the per-request steady caches die with the request (`allocation_owner.py:547`) | Skip phase-2 Page compaction (3.3); nothing else |
| `--cuda-graphs full` | No new device op in decode; the local-turn-0 capture follows an eager prefill forward; the handoff capture runs in the eager probe | None |
| `--admission-overlap K`, parked Goals | Admission unchanged; phase 1 is \|C\| + 3 tokens longer, so the one phase-1 lease is held slightly longer | None |
| `--deferred-completion-io` | `truncate` and the dropped ids built on the engine thread in `terminal_value`; the writer and its join points (`deferred_completion.py:575-623`, `746-767`) unchanged | ~+16-40 KB JSON per native file |
| `--release-idle-pages` | Pages past the cut are never pinned; the tail stays pinned through eviction (`ownership.py:347-360`) | None (L would need `s.evicted` filtering and a restore of an evicted cut page) |
| Capacity gate (`capacity.py:186-222`) | Reads the committed (cut) session; phase 1 needs about (\|C\| + 3)/P more fresh Pages, which `fresh = ceil(N/P) - base` already counts; the rebuild test at `:198` is the transcript's unchanged `extends` | None beyond stage 1's reply bound; pinned budget unchanged |
| `--gdn-device-handoff` | Needs the issued D2H at phase-1 end (2.1) | Recommend refusing with strip mode in v1 (not in the benchmark configuration) |
| `--phase1-in-step` | The same capture call through arm/collect (`native_probe.py:195-224`) | Recommend refusing in v1 (as async C does) |
| `--goal-overlap` (branch goal-overlap) | The TurnContext is built at the same point and reads `previous_action()` = C; a longer phase 1 gives the overlap more to hide | None |
| vLLM rewriter (branch rewriter-vllm) | Separate engine process, its prompt `enable_thinking=False` | None |
| Async C (branch async-sched) | Section 2.3 | Reuses the positioned carry (planned; as built it does not, `async_sched.py`) |
| Rebuild, rebind, `close_session` (`private_endpoint.py:188-206`, `session_pool.py:112`) | Unchanged; the next local turn 0 splits at e | None |
| Cancel / abort in phase 1 or 2 | Unchanged: the committed carry (at X_{t-1}) and mirrors stay; the session is poisoned (`allocation_owner.py:559-567`) | None |
| Gemma 4 | Window layout: the state that crosses turns is the pinned window after C' (`adapter.py:461-464`); thinking on changes the mandatory prefix | Strip mode refused; Gemma stays thinking off (superseded on 2026-11-09: [docs/thinking-gemma/DESIGN.md](../thinking-gemma/DESIGN.md), the held cut window) |
| kv-session (MemoryAgentBench), XML protocol (SWE-bench) | Their baselines keep R (ANALYSIS addendum) | Strip mode refused; stage 1's append |

## 6. Tests

CPU (fakes, pinned tokenizer):
- T1 template: `A(c)` equals the template's history render for random contents (whitespace, Unicode, a `</think>`
  inside C, empty C, a reply cut at `max_tokens` without `</think>`); `A[:3]` equals the header's first 3 ids; the
  strip `check_template`.
- T2 transcript replay: the 178 TB ATIF trials of `test_chat_text_replay` with synthetic reasoning inserted into each
  reply: every served prompt equals the stock strip render; after each commit the stream equals `prompt[:X]`;
  summary hand-offs rebuild from text.
- T3 engine and worker fakes (the stage-B / chat-text e2e harnesses): commit at X for P = 16 and legacy 784, X % P == 0,
  tail candidate (X % P ≥ P - 3), tail live; catalog, resident, tail, pins, block namespaces, freed row blocks, worker
  mirror == engine mirror; the carry's identity token == the phase-1 state's; no C' capture issued; the local-turn-0
  split and capture; cancel in phase 2; rebuild after a cut; capacity gate counts.
- T4 flag matrix: fast paths 1-3, `--deferred-completion-io`, `--admission-overlap 16`, `--release-idle-pages`:
  plans, tokens and evidence equal the flags-off fakes; identity probe: with strip off, the startup record and every
  evidence file are byte-identical to `cf2b28d`.
- T5 verifier units on synthetic evidence: positive runs, and negatives (a skipped cut, a position computed twice in
  one version, a wrong re-prefill piece, dropped ids not matching the completion tokens).

GPU (ghx4 batch, job mail BEGIN,END,FAIL; after the CPU suite and the release-check receipt):
- G1 token alignment (one session, b = 1, `full`, eager and `--cuda-graphs full`, Qwen): a scripted Terminus-shaped
  client with fixed observations; stock vLLM 0.24.0 eager with the default template and `enable_thinking` true is fed
  the same messages. Every prompt's ids must be identical; replies compared token by token (report the first
  divergence and the share identical, against stock's own spread). Also on our server: the turn after a cut against
  a forced rebuild of the same prompt (checks the carry at X and the tail). Include cold prompts with
  N mod 784 ≥ 5 and < 5, and a reply cut at the bound.
- G2 TB16 dry runs, thinking on, strip mode, per policy (`full`, `observation_query`, `random_fill`, `goal_query`
  with `--admission-overlap 16` and the rewriter batch), with the benchmark flags, plus the stock thinking-on control;
  verified offline with the new rules.

What they measured (every runtime server with `--enable-thinking --thinking-history strip` at runtime `8ca8511`; from each job's offline verification, not audited):
- **G1, job 3493649.** Prompt ids equal stock's on every server (20 of 20). The turn after a cut was compared with its
  forced rebuild (12 pairs per server). The readings A1 and A2 were not met (readings, not outcomes). The cold probes
  include N mod 784 = 1, 3 and 4, where the header straddles an M multiple.
- **G1b, job 3500646** (decision GRAPH_NUMERICS): G1's two early divergences (reading A1) are CUDA-graph near ties
  (margins 0.125 and 0).
- **G2, job 3499759** (plan strip25, the Terminal-Bench dry run): stripobs262 solved 8 of 16 tasks, stripgoal262 6 of 16,
  and the stock thinking-on control stockthink262 8 of 16; verified with the rules V1-V5 in `perf-tb16-strip`.

## 7. Effort and risks

| Part (E) | Files | Lines |
| --- | --- | --- |
| Strip transcript: commit, plan, piece, startup check | `chat_text_protocol.py`, `runtime_renderer.py` | 130-200 |
| Flag, pins, refusals, startup record | `serve.py`, `remote_ra_http/config.py`, `chat_text_route.py` | 80-130 |
| `commit_end` in plan and assignment; completion at X | `adapter.py`, `contract.py` (live v2), `allocation_owner.py` | 120-200 |
| Positioned carry, drain lineage, skipped C' capture, compaction skip, local-turn-0 capture | `worker_backend.py`, `checkpoint.py`, `qwen.py` | 120-200 |
| Local-turn-0 split | `composition.py` | 15-30 |
| Selection publication, capacity rule | `query_selection.py`, `capacity.py` | 10-30 |
| Evidence fields | `evidence.py`, `chat_text_route.py` | 50-100 |
| **Runtime total** | ~14 files | **550-900** |
| Tests (T1-T5) | new `test_thinking_cut*.py`, evidence tests | 800-1,200 |
| Verifier (sealed harness) | `perf-tb16/tb_checks.py`, `smoke_checks.py`, `join_tb16.py` | 150-250 |

About 1-1.5 weeks of CPU work plus G1 and G2. Form L adds about 700-1,200 runtime and 500-800 test lines (the L
column of section 1.2, a `cut` RPC, a second carry and its pinned budget): 2.5-3.5 weeks, in line with ANALYSIS's
1.5-3 weeks.

Risks:
1. Local-turn-0 split at a non-M-aligned position under vLLM's Mamba align mode (medium; G1). If it fails, local
   turn 0 needs another capture point (for example a probe prefill of [0, e) when e fits one forward).
2. Verifier rework in the sealed harness (medium effort, low technical risk).
3. Template exactness of the re-encoded piece: trim, `</think>` inside C, length-cut replies (low with T1/T2; the
   per-turn 3-token check fails closed).
4. Selection semantics shift (3.5): no correctness risk, but strip runs are comparable only with strip runs.
5. Phase 1 longer by |C| + 3 tokens: more `PHASE1_TOO_LONG` exposure, +tens of ms per turn; offset by the skipped
   C' carry D2H (154 MB) and the arena D2H of R's Pages (about |R|/16 MiB) per turn.
6. Stage 1 is not merged yet; its turn record and header interfaces may move (it was merged with form E, `94888e1`).

## 8. Decisions for the maintainer

1. **E (commit at the cut, recommended) or L (roll back at the next request).** L is needed only to serve both
   histories in one session; it costs ~154 MB pinned per session and about twice the code.
2. **Pin the history per server**: `--thinking-history strip` (Terminal-Bench, chat-text, Qwen only; reference = stock
   default template) next to stage 1's keep mode (SWE-bench, MemoryAgentBench). Confirm that strip is refused on
   Gemma, kv-session and the XML protocol.
3. **Stable end at X + 3** (recommended: geometry unchanged; C is attended densely in the next turn and grouped with
   the following observation) **or several phase-1 candidates** (C as candidates of the turn that wrote it, as
   ANALYSIS sketched: CANDIDATE_RANGE relaxed, phase-1 captures of every candidate Page, +300-500 lines, more GPU risk).
4. **Evidence**: the dropped generated ids in the native file (recommended) or a digest and count only (then the
   verifier cannot re-derive C); who changes the sealed verifier, and the verdict names for strip runs.
5. **v1 refusals**: `--gdn-device-handoff` and `--phase1-in-step` with strip mode (recommended; neither is in the
   benchmark configuration).
6. **An over-long re-prefill** ([X, e) above one native forward): fail the turn as today, or rebuild.
7. **Order**: after the rewriter in vLLM and before async C (as planned on 10-29); async C then reuses the positioned
   carry, and on strip servers needs no GDN void. (As built, async C does neither: `async_sched.py`.)
