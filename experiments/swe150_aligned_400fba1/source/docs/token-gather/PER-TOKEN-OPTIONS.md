# Per-token scoring for option B (token gather): design options for runtime support

English | [简体中文](PER-TOKEN-OPTIONS.zh-CN.md)

Investigation of 2026-11-15, read-only: no code was changed and no GPU job was run for it; the GPU readings it cites come from earlier verified jobs, named by job id. The GPU runs of token gather are in [docs/status.md](../status.md).

> **Revision (2026-11-15).** Revised after an independent audit (2026-11-15) [S34]. The audit listed 11 required fixes and 20 recommended ones: all 11 required fixes were made and all 20 recommendations adopted, none skipped. Every changed number comes from the audit's re-measurement (its 1-thread and 4-thread runs and its fill check) or from a re-derivation in this note; its source is in Appendix C. Appendix B lists the changes item by item.

- **Basis.** Branch `integrate-1114` of this repository (HEAD `a565e84`, code `b388033`); the `main` docs at `f18b8e5`.
- **Sources.** Every `file:line` is relative to the repository root at `integrate-1114` unless its row names another commit. The `[S#]` after a fact is a row of the sources table (Appendix C).
- **Three labels on numbers:**
  - **[GPU]**: a verified GH200 job reading, with its job id.
  - **[local CPU proxy]**: measured on one x86 Windows machine (Python 3.12.7, numpy 1.26.4; Appendix A, [S32]). Not a GH200 number: it only compares orders of magnitude. The thread count was not limited: numpy's BLAS used its default, all 32 logical cores. The audit re-measured at 1 and 4 threads [S34]: the integer scoring at 262,144 tokens was 1.3–2.3 times slower, the other quantities within 1.0–1.4 times of the original values, except the resident-f32 score at 30,496 tokens on 1 thread (1.8 times: 0.527 against 0.286 ms).
  - **[estimate]**: computed from the two kinds above, with the formula beside it.
- **Who owns the rule.** The goal_query study sets the exact per-token scoring rule; the runtime supports it. Every "rule" below is a question for the algorithm side (section 11).
- **Common premise.** In this runtime every option acts on the 16 full-attention (FA) layers only and keeps the logical RoPE positions; the 48 Gated DeltaNet (GDN) layers carry a recurrent state of the whole history, which no selection touches [S18 §2][S19 §2].

---

## 0. Summary

**The gap.** Under an 8,192-token budget the goal_query study first protects the prefix and the last two turns, then "remaining slots go to tokens with the highest Goal scores" [S28]. Today the runtime has one vector per page and fills the remaining budget with whole pages by page cosine [S18 §9][S19 §4].

**Main findings:**

1. **The fill logic needs no change.**
   - Token gather's existing picker interface plus the `shared_blocks_v1` budget can already express "the remaining slots go to tokens". When the picks are room × 16 tokens sorted by score, the fill takes as whole pages only the pages whose 16 tokens are all selected, gathers every other token, and the pad is 0.
   - This note checked it with the runtime's own code on 2,000 random trials [S33]; the audit checked it independently on 3,000 trials × 4 variants (through `plan` and `turn`, a markers-first list and a quota included) [S34].
   - What is really missing is three things: a score per token, the throughput of a large gather, and a way for the picker to get the room (the picker's `view` has no room, see 2.1).
2. **Which cost is largest depends on the option.**
   - **For option 1:** the gather (25–141 ms per turn [estimate]) is far larger than the scoring.
   - **For option 2:** reading K (54–868 ms [estimate]) is of the same order as the gather or larger.
   - Gather volume: filling the remaining budget by token gathers at most about 5,051 tokens per turn [estimate: mean room 315.7 pages × 16, S27]. This is an upper bound, because a page whose 16 tokens are all selected is restored whole instead. For comparison, the markers gather only 110.6 per turn today [GPU, job 3603443, S24].
   - H2D upper bound: 315.7 MiB per turn [estimate].
   - The 141 ms is a linear extrapolation of G1's two measured points (144 and 1,024 tokens) to 5,051 tokens, about 5 times the largest measured set, beyond the measured range [estimate].
   - The gather runs synchronously in the first phase-2 step, and every session of the batch waits for that step [S18 §4][S26][S36].
   - G1's `UVA_ADVISED` is only a threshold rule (C set's DRAM gather median > 10 ms); G1 did not measure a UVA path [GPU, job 3603240, S23, S38].
3. **Option 1: one vector per token, in the page vectors' space.**
   - **Storage and scoring speed go together:**
     - Storing only the binary int8 + scale (132 B per token; 33 MiB for a 262,144-token session), the integer-exact cosine converts from int8 every turn: 3.2 / 28.7 ms at 30,496 / 262,144 tokens [local CPU proxy, 32 cores].
     - With a resident float32 copy of 512 B per token as well (14.9 MiB for a typical session, 128 MiB at 262,144 tokens): 0.29 / 4.2 ms [local CPU proxy, 32 cores].
     - At 1–4 threads, at 262,144 tokens: 37.8–39.1 ms and 6.2–9.7 ms respectively [audit re-measurement, S34].
   - **Reproducibility:** given the int8 query and token vectors, the scores are bit-identical on every implementation that accumulates in int32 or fp32. But the int8 query itself is compressed in fp32 on the engine CPU, and the GH200 and x86 differ in the last bit in a few cases; this is why the parity tool needs `recover_query` [S11][S30].
   - **Today's page-vector path of Python objects, JSON and signing cannot be copied:**
     - that is about 8 KB per token, 1.96 GiB per session at 262,144 tokens;
     - about 190 Python objects per token [S34];
     - about 71 ms of ingestion per turn [estimate: local CPU proxy 47.3 µs per vector × an assumed 1,500 new tokens per turn];
     - the upstream bit-identical cosine takes 179 ms at 262,144 tokens [local CPU proxy].
4. **Option 2: score directly with the K cache.**
   - No new storage, but the Goal vector is a last-layer hidden state, not the query of any FA layer. The q of the 16 FA layers must be captured in the Goal forward (192 KiB per Goal), and RoPE handled.
   - The bandwidth per turn grows linearly with the history: reading the K of 16 layers takes about 54–76 ms for a typical history and about 620–868 ms at 262,144 tokens [estimate; at today's restore path's effective throughput; a direct device read is not measured].
   - K is not in the evidence: offline, the fill can only be replayed on recorded scores.
5. **Option 3: two stages.**
   - Meaningful only with option 2: it limits the K reads to the top S pages by page cosine. At S = 2 × room, 16 layers take about 25–35 ms, independent of the history length [estimate, upper bound].
   - The cost is coverage. Coverage with option 2 needs q·k scores: from option 2's probe, or per-token scores from the author.
6. **Option 4: reduced dimensions or precomputed scalars.**
   - On a cache miss the Goal changes; and at capture the Goal of later turns does not exist yet, so a Goal-dependent score cannot be precomputed at capture.
   - Fewer dimensions matter only for evidence size or to fit in HBM; kept as a parameter of option 1.
7. **Option 5: contract, receipt, evidence and parity.**
   - Following the precedents of token gather and exp-subpage-score: a new contract key appears only when enabled; the per-token vectors stay outside the signed body. The price is that these vectors are not MAC-authenticated (the page vectors are); this is a deliberate trade-off, consistent with "integrity checks at startup only".
   - The receipt records only the gathered tokens (already there), the threshold and the int8 query.
   - The per-token scores do not go into the signed body: 5.2 MB per request at 262,144 tokens, at least 184 ms to build and serialize [local CPU proxy]; the engine's signing and the verification at reservation add about 110 ms each [audit re-measurement].
   - A full offline replay of goal_query needs the parity tool extended, because today it refuses goal_query [S30].

**Recommendations (section 10):**

1. **Prototype A: option 1 in shadow mode.**
   - Its capture, transport and storage do not depend on the rule; the statistics it records (top-k, threshold, overlap) apply to one rule only, the same-space cosine.
   - It measures how far a per-token same-space cosine selection is from today's page fill (overlap, tokens outside the pages), and the two-stage coverage under that score.
   - Whether it is "better" can only be answered by runs that actually change the selection.
2. **Prototype B1 (independent of the rule, now):** a GPU probe of large-gather throughput. With a real distribution: 5,051 tokens scattered over 1,000+ pages and several arena chunks; timed in a step with concurrent decode.
3. **Prototype B2 (after A):** the reuse rate of an incremental gather. Which tokens repeat from turn to turn depends on the rule, so it has to be measured with A's shadow selections (or a replay of a rule); synthetic data cannot measure it.
4. Wire the picker once the rule is confirmed. Start options 2 and 3 only if the study's score is attention-based (q·k).

---

## 1. Current state

### 1.1 Model geometry (Qwen3.6-27B, [S1][S2])

| Item | Value |
| --- | --- |
| hidden | 5,120 |
| Layers | 64: 16 FA layers (one in every 4, layers 3, 7, …, 63, counting from 0) + 48 GDN layers |
| Attention heads | 24 q heads, 4 KV heads (every 6 q heads share one KV head), head_dim 256, scaling 1/√256 |
| RoPE | partial_rotary_factor 0.25: 64 of each head's 256 dimensions rotate; MRoPE interleaved [11, 11, 10]; theta 10,000,000 |
| Before the cache write | q and k get a per-head RMSNorm first, then RoPE; K is stored after RoPE |
| Data type / longest position | bf16 / 262,144 |

**Derived quantities [estimate: products of the table]:**

- Per token per FA layer: K = 4 × 256 × 2 B = 2 KiB; K+V = 4 KiB.
- Per token, 16 layers: K = 32 KiB; K+V = 64 KiB.
- Per 16-token page: 1 MiB. This matches `token_gather.cost` and G1's `page_bytes` 1,048,576 [S15][S23].

### 1.2 Page vectors: where they are made, where they are kept, how they are scored

**Representation**

- The output of the last decoder layer, before the final RMSNorm. The hook gets the row after the norm and divides by the gain to recover the direction [S3][S4].
- Each page takes the row of its last token.
- A 128-d random ±1 projection with seed 20260904 (5,120→128), then L2 normalization, then int8 quantization with one fp32 scale [S13][S21].

**Production (on the GPU, temporary HBM only)**

- **Phase 1 (the observation [C, e))**: `ProbeRows` keeps only the page-end rows (one device clone) and the last row (the observation query, copied to the host in fp32) [S6]. `PageCompactor.compact` projects, normalizes and quantizes in float64 on the device, then `.cpu().tolist()` copies back to the host [S5].
- **Phase 2 (generation header, cold prefill, decode)**: `after_forward` compresses the pages closed in the step the same way. In decode that happens only once every 16 steps [S7].

**Transport**

- Through the worker's phase1 receipt and publish receipt. The wire form is `{page: [[128 int8], scale]}`, about 0.6 KB of JSON per page.
- The receipt body is signed: canonical JSON plus HMAC-SHA256. Phase 1 verifies it once more right after signing [S8][S9].

**Storage (engine-side host memory, no resident HBM)**

- `vector_catalog[session]`: a map from page to CompactVector Python object, with `unit_values` memoized.
- `PageColumns`: one 128 × n float64 column matrix per session, grown by doubling, so 1,024 B per vector is a lower bound [S11][S12].
- A page vector is computed once when its page is sealed and never changes [S11].

**Scoring**

- `fast_selection.cosines`: a float64 compensated summation bit-identical to the upstream `compact_cosine`, vectorized over the dimensions [S12].
- Each page's score is rounded to 8 decimals and written to the evidence's `scores`. The receipt body the engine signs holds both the `ranking` of all n pages and the `selection` (which includes the 8-decimal `scores` of all n pages). That body is HMACed once at issue and verified once at reservation [S11][S37].

### 1.3 The Goal vector

- **goal_query:**
  - The Goal text gets one cold forward in the phase-1 lease, `start=0`, on its own blocks [S6].
  - The pre-final-norm value of its last row is taken and normalized, and goes into the phase-1 receipt as 5,120 floats [S8].
  - At plan time it is projected in fp32 on the engine CPU into the same 128-d int8 space [S11].
  - It is cached by `retrieval_goal_hash`; a hit runs no Goal forward, and the receipt's `goal.cache_hit` records whether it hit [S11].
- **observation_query:** the last row of phase 1, that is, the observation's last token [S8].
- **Conclusion:** the Goal vector is a point on the last layer's residual stream, in the same space as the page vectors. It is not the query of any FA layer.

### 1.4 What token gather has today

**Selection**

- **Picker:** `TokenPicker.picks(view)` returns token positions in the candidate pages (outside the mandatory and recent ones), in priority order. There are `markers_v1` and `separators_v1`; `register_picker` registers new ones [S15].
- **What `view` holds:** `stream`, `end`, `page_tokens`, `eligible`, `state`, `encode`, `separator`, `pin`; **no room** [S15].
- **Budget `shared_blocks_v1`:**
  - room = capacity − mandatory − recent, in blocks; the picks are cut to room × 16.
  - It walks Selector V2's cosine order page by page: a page is taken if the page plus the gather blocks needed by "the picks not covered by the taken pages" still fit.
  - When the last block is not full, it is filled (the pad) with the newest tokens that nothing attends [S15][S18 §3].

**Restore**

- Tokens of resident pages are indexed on the device; tokens of DRAM pages are first collected on the host into one bounce buffer, then each FA tensor gets one H2D and one `index_put_`, and `drain()` ends it [S16].
- This call runs synchronously in the first phase-2 step [S36]. Each token moves 64 KiB [S15].
- **Why it is exact:** K is stored after RoPE, q uses logical positions, and the order of the keys in the prefix does not matter; every slot is a recorded real token [S18 §2].

**GPU readings**

- **G1 (job 3603240, GH200)** [S23]:

  | Set | Tokens | DRAM gather median | Of which host index | Restoring the pages that hold these tokens | Resident-page gather |
  | --- | --- | --- | --- | --- | --- |
  | E | 144 | 53.40 ms | 49.84 ms | 84 pages 9.19 ms | 3.46 ms |
  | C | 1,024 | 69.06 ms | 59.27 ms | 498 pages 38.88 ms | 15.41 ms |

  What `UVA_ADVISED` means: G1's rule is "give `UVA_ADVISED` when the C set's DRAM gather median > 10 ms"; it did not measure the UVA path itself [S38].
- **gath32ti (job 3603443, goal_query + markers, 32 tasks)** [S24][S26]:
  - 1,210 selective requests; picks per request mean 133.8, gathered mean 110.6 (p95 317).
  - 126,474 tokens gathered from DRAM, 14,358 on the device.
  - admission_engine_ms p50: 230.2 for all requests, 245.0 for selective requests; latency p50 8.549 s; pool wall 2,845.1 s.
  - No per-request gather time, because the gather runs inside the first phase-2 step.
- **Today's restore path's effective throughput, from G1 [estimate: measured bytes ÷ median time]:**
  - 88,080,384 B ÷ 9.1912 ms = 9.58 GB/s;
  - 522,190,848 B ÷ 38.8763 ms = 13.43 GB/s.
  - This is the effective rate of the existing path (staging plus one `index_copy_` per tensor), not a link limit. Every "DRAM → HBM time" below uses the range 13.43 to 9.58 GB/s.

### 1.5 History sizes in the SWE runs

- **The 1,559 calls of gath32ti:** 1,257 with a prompt under 32K, 237 at 32K–64K, 65 at 64K–98K, the longest 97,778 [S25]. Each call generates 358.5 tokens on average [S26].
- **Budget:** capacity 512 pages, mean fill room 315.7 pages [S27].
- **Concurrency and resources:** 16 sessions run at once; Slurm `--mem 113560M`, 16 CPUs, 1 GPU [S26].
- **Mean candidate pages:** goalvllmt 1,906 pages (about 30,496 tokens), obs262t 1,492 pages; mandatory about 140 pages, recent about 48 pages [S28]. Note that goalvllmt and gath32ti are not the same run.
- **Upper bound:** 262,144 tokens, that is 16,384 pages [S1].
- **The two sizes used below:**
  - "typical" = 1,906 pages / 30,496 tokens;
  - "upper bound" = 16,384 pages / 262,144 tokens.

### 1.6 The gap as the docs state it

- `docs/token-selection` on main, section 4 "Not covered": per-token Goal scoring needs one vector and one score per token (16 times today's), and the exact scoring rule of the goal_query work; section 5, question 4 [S19].
- `docs/token-gather` on `integrate-1114`, section 9 item 1 [S18].
- The study's setting comes from the author's weekly report of 11-11, as summarized in our note of 11-13 [S28]:
  - protect the system prompt and the first user message (1,568–2,688 tokens) and the last two turns (up to 1,024 tokens), the three sharing the 8,192 budget; the remaining slots go to the tokens with the highest Goal scores;
  - the experiments ran on the author's own native Mini-SWE stack (H200), not on our runtime.
- The policy registry lists token-level selection as out of scope, its doc lists "per-token Goal scores (one vector per Page)" as not done, and the README says a new page-vector score and token-level policies need more runtime support [S17][S20][S22].
- **None of our docs states two things:**
  - how "the per-token Goal score" is computed (section 11, question 1);
  - how the study applies a token selection in its own stack: whether the original positions are kept, and whether the GDN state still sees the whole history (section 11, question 3).

  These two are premises of every option.

---

## 2. What every option needs

### 2.1 The fill: today's `shared_blocks_v1` already expresses "the remaining slots go to tokens"

Swap the picker for "the candidate tokens by score, descending", and take the first room × 16.

**Derivation:** `fill` takes a page only when taken + 1 + ceil((left − c)/16) ≤ room, with c the number of picks on that page [S15]. Initially left = 16·room. As long as every taken page has c = 16, left = 16·(room − taken) holds throughout, and the condition reduces to c = 16.

**Conclusion:**

- only pages whose 16 tokens are all selected are restored as whole pages;
- every other pick is gathered; their total is a multiple of 16, and the pad is 0;
- the selected token set is exactly the first room × 16 tokens, which is the study's rule.

**Checks:**

- This note checked it with the runtime's own `token_gather.plan` on 2,000 random trials [S33].
- The audit checked it independently on 3,000 trials, each with a random geometry and five score distributions (ties and adversarial pages with "15 of 16 tokens high" included) [S34]:
  - it holds through both `plan` and `turn`;
  - it holds for a markers-first list too;
  - with a quota, the result equals a re-implementation of the documented rule.

**Two side benefits:**

- With fewer than room × 16 picks (for example with a quota), the rest of the budget is filled with whole pages by page cosine. The mixed form "N Goal tokens + whole pages for the rest" needs no extra change.
- Combined with markers, the pick list is "all markers (newest first), then the Goal tokens": one combined picker is enough.

**What the runtime has to add:**

- **The picker gets the scores:** `view` must carry the scores, or the token catalog plus the query.
- **The picker gets the room:** `view` has no room ([S15] `token_gather.py:468-474`). Two ways:
  - add the room (or the capacity) to `view`, and the picker returns only the first room × 16;
  - let the picker return every candidate sorted by score, and the fill cut it. Then `truncated` counts the whole tail, and the per-pick check in `turn()` grows with the number of candidates: 6.7 / 56.5 ms at 30,496 / 259,136 candidates, against 4.1 / 7.1 ms when only room × 16 are returned [local CPU proxy, audit re-measurement, S34].
- **Pin:** the rule has to be pinned in the contract (section 7).

### 2.2 Gather volume and throughput

**Per turn [estimate]**

- At most about 5,051 tokens gathered per turn (mean room 315.7 × 16). Pages selected in full are restored whole instead: the same H2D bytes, but such pages can stay resident across turns. The markers' mean today is 110.6, about 1/46 of it.
- H2D upper bound: 5,051 × 64 KiB = 315.7 MiB per turn. Whatever the mix of pages and tokens, the upper bound is room × 1 MiB.
- For comparison: in gath32ti the mean non-phase-1 H2D per request is 226.4 MiB, which includes integer multiples of 146.8 MiB of GDN state [S26].

**Time [estimate]**

| Path | Per turn | Method and confidence |
| --- | --- | --- |
| Existing host-index path | about 141 ms | G1's two points (144, 1,024 tokens) extrapolated linearly to 5,051 tokens, about 5 times the largest measured set. Slope (69.06 − 53.40)/880 = 0.0178 ms per token, intercept 50.8 ms (95 % of the 144-token time). The intercept is unexplained: the audit timed the host-index step alone on x86 (16 FA tensors, one unpinned 256-slot chunk) at 4.2 / 15.6 / 67.9 ms for 144 / 1,024 / 5,051 tokens, linear, with an intercept of only about 2 ms [local CPU proxy, S34]. So the slope is credible and the fixed part cannot be explained: one more reason to run B1 rather than trust this line. |
| At today's restore effective throughput | about 25–35 ms | 331,022,336 B ÷ (13.43 to 9.58 GB/s) |
| Device read of pinned memory (UVA) | not measured | G1 gave only a threshold decision [S38] |

**Impact**

- **ITL:** the gather runs synchronously in the first phase-2 step [S18 §4][S26][S36], so its time counts directly against every running session of the batch.
  - At gath32ti's pace [estimate]: 1,210 selective requests ÷ 2,845.1 s = 0.425 per s.
  - At 141 ms per request, every running session is held up for about 6.0 % of the wall time; at 25–35 ms, about 1.0–1.5 %.
- **HBM** [estimate]:
  - A gathered block is 16 tokens × 64 KiB = 1 MiB (16 FA layers). About 316 gathered blocks per session, about 316 MiB. Between turns they replace today's resident selected pages, of a similar size.
  - The previous turn's gathered blocks are released only at the commit [S18 §4], so during a reservation a session holds at most about 2 × room gathered blocks at once, about 632 MiB.
- **Capacity gate:**
  - With a gather pin, the capacity gate already reserves fresh blocks as min(largest, non-resident + most) (`most = largest` at quota 0, about 512 pages). This is the same worst case as a long history without gather [S35].
  - gath32ti ran under this estimate, with CAPACITY_WAIT 0 times [S24].
  - What has no data yet: the behaviour when about 316 gathered blocks are really allocated every turn.

**Countermeasures (none depends on the scoring rule)**

- the device reads pinned host memory directly for the gather (not measured; B1's subject);
- an incremental gather (token gather section 9 item 3): tokens that recur from the previous turn's gathered blocks are not copied again (the reuse rate is measured in B2);
- pages selected in full are restored whole: the fill already does this (2.1).

### 2.3 The rule must be pinned

All of the following go into the contract pin (section 7):

- the definition of the score;
- the tie rule (suggested: by (−score, position));
- which tokens can be selected (in the candidate pages, outside the mandatory and recent ones; reasoning tokens or not);
- the order relative to the markers.

---

## 3. Option 1: one vector per token, in the page vectors' space

### 3.1 How

Give every token the same representation as the page vectors: pre-final-norm row → 128-d projection → L2 → int8 + scale.

A page vector is the vector of the page's last token. The implementation follows exp-subpage-score [S29]:

- token 16 of each page reuses the page vector directly, and the page-end row keeps today's path;
- only the other 15 tokens of a page get new vectors.

**Why row 16 stays on the old path:** the compression result is independent of the batch shape only up to the rounding edges ([S5] `page_compaction.py:14-17`). Keeping the page-end row out of the new batch keeps the page-level selection, the page vectors' bytes and the digests unchanged.

### 3.2 Rules the algorithm side has to give

- Is the token score just cos(Goal vector, token vector)? That is the most direct definition in one space. Or does it add neighbourhood smoothing, a page-score weight, or score only some classes of tokens?
- Is the query goal_query's Goal vector or the observation vector? The same mechanism supports both.
- How are ties handled; do the picks fill all of room × 16, or is there a quota with whole pages for the rest?

### 3.3 Where to capture

| Stage | Today | Instead |
| --- | --- | --- |
| Phase 1 (the observation [C, e)) | `ProbeRows` clones only the page-end rows [S6] | compress every row on the device in chunks, return only int8 + scale (132 B per token) |
| Phase-2 prefill (cold prompt, rebuild, generation header) | `after_forward` compresses only the page-end rows [S7] | the same, over every row of the step |
| Decode | one D2H when a page closes (once every 16 steps) [S7] | project the row into a 16 × 128 device buffer each step, without a sync; quantize and D2H once when the page closes. The same number of syncs as today |
| Can be skipped | — | mandatory pages (the initial prompt, always attended) need no token vectors; pages past the `--thinking-history strip` cut are not published under the existing rule [S7] |

**Device memory** [estimate: rows × 5,120 × 4 B for the fp32 copy, × 8 B for the float64 buffer]:

- `pre_final_norm` already makes one fp32 copy of every row of a step today: a Qwen release has at most 16,384 rows per step, that is 320 MiB [S29][S39].
- What is new is only the float64 temporary buffer of the compression: 640 MiB for 16,384 rows at once, 40 MiB in chunks of 1,024 rows. Chunking limits only this part.

**Paths to cover as well:** `--phase1-in-step`, rows voided by `--async-scheduling`, rebuild, cancel. The Goal forward of `--goal-overlap` needs no token vectors.

### 3.4 Memory

| Representation | Per token | Typical, 30,496 tokens | Upper bound, 262,144 tokens | 16 sessions × typical |
| --- | --- | --- | --- | --- |
| Binary: int8 + fp32 scale (plus an int32 squared norm) | 132 B (136 B) | 3.8 MiB | 33.0 MiB | 61.4 MiB |
| Resident float32 copy (for fast scoring, in addition) | 512 B | 14.9 MiB | 128 MiB | 238 MiB |
| Today's Python representation copied (CompactVector + memoized `unit_values` + float64 column) | 7,001 + 1,024 = 8,025 B [local CPU proxy] | 233 MiB | 1.96 GiB | 3.65 GiB |
| For comparison: one vector per page today (Python representation) | — | 1,906 pages, about 14.6 MiB | — | — |

The values in the table are all "bytes per token × tokens" [estimate]. The audit re-measured the Python representation at 6,937 + 1,024 B, within 1 % [S34].

- **HBM:** no resident use, only the temporary use at capture (3.3).
- **Against the DRAM arena:** 16 sessions × 1,906 pages × 1 MiB ≈ 29.8 GiB [estimate]. The binary token vectors are only 132 B / 64 KiB ≈ 0.2 % of the KV.
- **Why not the Python representation:** not because of the bytes: 3.65 GiB is only about 3.3 % of the job's `--mem 113560M`. The real reasons are:
  - time: about 71 ms of ingestion per turn, and 179 ms for the upstream cosine at the upper bound (3.5);
  - object count: about 190 Python objects per token, about 9.3 × 10^7 for 16 typical sessions [estimate: 16 × 30,496 × 190]. The per-token object count is the audit's [S34], and this note's breakdown agrees: 1 value tuple, about 58 ints outside the small-int cache (−127..−6), the object itself and its `__dict__`, the memoized tuple, 128 floats.
- **Conclusion:** the binary representation is feasible; whether the resident f32 copy is needed depends on the scoring time required.

### 3.5 Time per turn

**New tokens per turn:** the typical value of exp-subpage-score, 1,500 [estimate]. That number dates from 2026-10, at P = 784 [S29]. In gath32ti (thinking on) each call generates 358.5 tokens on average [S26], and the observation length is not in the local data, so 1,500 is only an assumption.

| Step | Today's way copied (Python, JSON, signing, upstream cosine) | New representation (binary, numpy, integer-exact cosine) |
| --- | --- | --- |
| Ingestion | about 71 ms [estimate: local CPU proxy 47.3 µs per vector × 1,500] | read the binary directly with `frombuffer`, < 1 ms [estimate] |
| Sign + verify (phase 1) | 45.7 ms to sign 6,500 vectors once [local CPU proxy]; about 10.6 ms × 2 for 1,500 [estimate] | outside the signed body: 0 |
| Scoring, typical 30,496 | 13.49 ms (upstream bit-identical) [local CPU proxy] | storing only 132 B per token, converted from int8 every turn: 3.22 ms; with the resident f32 copy (another 512 B per token): 0.29 ms [local CPU proxy, 32 cores] |
| Scoring, upper bound 262,144 | 179.2 ms [local CPU proxy, 32 cores]; 207 ms at 1 thread [audit re-measurement] | from int8: 28.66 ms; f32 copy: 4.25 ms [local CPU proxy, 32 cores]. At 1–4 threads 37.8–39.1 ms and 6.2–9.7 ms respectively [audit re-measurement] |
| Top 5,040 (= 315 × 16) and sort | — | 0.71 ms (typical), 3.33 ms (upper bound) [local CPU proxy] |
| All scores into the evidence | typical 19.4 ms, 0.58 MB; upper bound 183.6 ms, 5.23 MB [local CPU proxy]. Inside the signed body, the engine's signing and the reservation's verification add about 110 ms each [audit re-measurement] | only the picks, the threshold and the int8 query, close to 0 |
| GPU capture | not measured. Phase 1 projects about 5.4 GFLOP (float64) per 4,096 rows [estimate: 4,096 × 5,120 × 128 × 2] | same |
| Gather (2.2) | 25–141 ms [estimate; 141 ms is an extrapolation] | same |

**Integer-exact cosine**

- **Why exact:** the absolute value of a 128-d int8 dot product is at most 128 × 127 × 127 = 2,064,512, below 2^24. So int32 or float32 accumulation, in any order (GPU, BLAS, numpy), equals the integer dot product exactly; bf16 or TF32 inputs with fp32 accumulation are exact too, but fp16 or bf16 accumulation is not allowed. Checks:
  - locally, float32 dot products equal int64 dot products element by element on 30,496 and 262,144 vectors [local CPU proxy, S32];
  - the audit also checked permuted dimensions, sequential accumulation and torch CPU; a matmul with bf16 output is not equal [S34].
- **Score:** score = dot / sqrt(‖q‖²·‖t‖²), computed in float64. ‖q‖²·‖t‖² is below 2^53, so the product is exact too; sqrt and division are correctly rounded.
- **Qualifier:** **given the int8 query and token vectors**, the scores are bit-identical on every implementation. The query's int8 is itself compressed in fp32 on the engine CPU, and the GH200 and x86 may differ in the last bit in a few cases [S11][S30]. So the receipt should record the int8 query (section 7).
- **Relation to upstream:**
  - Mathematically it is the same real number as the upstream `compact_cosine` (`unit_values` = v/‖v‖, and the scale cancels [S14]); only the last bits round differently.
  - The audit measured a largest difference of 1.1e-16: at 30,496 the ranking is unchanged; at 262,144 one pair swaps, and the top-5,040 set is unchanged [S34].
  - So it is pinned separately, as the new per-token scoring rule; the page level keeps the upstream cosine.

### 3.6 Code changes

**Worker side**

- `remote_ra_live_v2/page_compaction.py`: compress every row, keep the result as a tensor; add a binary wire form.
- `native_probe.py`: `ProbeRows` keeps the per-token compression results.
- `qwen.py`: a decode buffer in `after_forward`; `publish_pages`.
- `model_queries.py`, `worker_backend.py`: the vectors outside the signed body [S29].
- `phase1_coordinator.py`: `Phase1Result` carries the token vectors; today it carries the page vectors, the observation vector and the Goal vector [S10].

**Wiring** (exp-subpage-score changed these too)

- `private_endpoint.py`: the engine-side class selection;
- `worker_composition.py`: the worker-side class selection;
- `composition.py`: the startup record and the one-time projection check [S29].

**Engine side**

- `query_selection.py`: one numpy token catalog per session. Appended at commit, cleared at rebuild or close; the same integrity checks as the page catalog; scored at plan time and handed to the picker.
- `token_gather.py`: a new picker (working name `goal_scores_v1`); `view` gains the scores and the room; the pin gains the rule fields.
- `selection_contract.py`: pin checks.
- `serve.py`: startup options.
- `remote_ra_http/evidence.py`: records.

**Harness**

- `postrun/selection_parity.py`: needs a per-token replay and goal_query support (section 7).

**Size [estimate]:** comparable to exp-subpage-score, plus a picker and the parity-tool extension. exp-subpage-score changed 6 files in `remote_ra_live_v2`, added 1 module and 19 tests, and also changed `serve.py` and the README [S29].

### 3.7 Risks

1. **Rule mismatch:** if the study's score is not a hidden-state cosine (for example attention q·k), option 1's score differs from the study's. This is the largest risk, and only question 1 of section 11 can answer it.
2. **Projection precision:** the 128-d int8 projection has an error relative to the 5,120-d cosine. The page level uses it today too; if the study uses the full dimension, a comparison is needed.
3. **The cost of the full dimension:** if the rule requires the full dimension, bf16 at 5,120 dimensions per token is 10 KiB: about 298 MiB for a typical session, about 4.65 GiB for 16 sessions (about 16 % of the 16-session DRAM arena's 29.8 GiB), 2.5 GiB for an upper-bound session [estimate]. Feasible but not small (the job's `--mem 113560M` [S26]), and the D2H becomes 10 KiB per token as well.
4. **Semantics (not validated):** the last-layer representation at each position is the model's state after reading up to that token, and the page vector (the page's last token) is the same kind of representation. Whether it is a good retrieval key for a single token has no data yet.
5. **Decode path:** every step of the fast path, async scheduling and cudagraph gets one more small kernel; the ITL needs measuring.
6. **Temporary device memory at capture:** the float64 temporary buffer takes HBM; chunking solves it (3.3).

### 3.8 Validation

**CPU tests**

1. The compression of all rows equals a per-row float64 reference, independent of the chunking and the batch shape.
2. Every token's vector is published exactly once, covering phase 1, prefill chunks, decode, rebuild, cancel, the thinking cut and rows voided by async.
3. The page vectors' bytes are unchanged and reused as the vector of token 16.
4. The integer-exact cosine is bit-identical on numpy, on torch CPU and with a permuted summation order.
5. The picker's top-k and tie rule are right; with picks = room × 16 the fill takes as whole pages only the pages selected in full (the S33/S34 property as a test).
6. The signed body is unchanged and the number of hash calls is unchanged (the test method of exp-subpage-score).
7. Disabled, `tools/policy_parity.py` is byte-identical.
8. Passes on the decoupled end-to-end harness of `test_token_gather_e2e`.

**GPU smoke**

1. G-probe: the decode step increment (b = 1..16), the phase-1 capture time at 1K–8K tokens, the D2H bytes.
2. Shadow mode on 1 task: each request's selection must equal the selection in the same state with the feature off; offline, recompute every score and the picks from the frames.
3. After the picker is wired, 32 tasks.

---

## 4. Option 2: score directly with the K cache

### 4.1 How

Take one query's dot product with the stored K of the 16 FA layers (or some of them), then aggregate by the rule into one score per token.

### 4.2 The Goal vector cannot be the query as it is

**Why not:** layer l's K lives in a 256-d space per head (after W_k, k_norm, RoPE) [S2], while the Goal vector is the last layer's residual (5,120-d, pre-final-norm). Layer l's query needs layer l's input and W_q^l, q_norm and RoPE; it cannot be derived back from the last layer's vector.

**What works:**

- **goal_query:** the cold Goal forward runs all 64 layers anyway; capture the q of the last Goal token at every FA layer (post-norm, post-RoPE): 16 × 24 × 256 × 2 B = 192 KiB per Goal [estimate]. A Goal cache hit runs no Goal forward [S11], so q has to be cached by `retrieval_goal_hash` too (192 KiB per Goal).
- **observation_query:** use the q of phase 1's last observation token. This q is computed in the history context that phase 1 sees.
- **Approximation (not recommended):** multiply the last layer's vector by the last FA layer's W_q directly. The input is from the wrong layer, so it is only an approximation.

### 4.3 RoPE

- K is stored after RoPE, with the phase of the logical position p_k [S18 §2]. The cold Goal forward's positions are 0..g−1, so the Goal's q carries the phase of position g−1.
- Only 64 dimensions of each head rotate; the other 192 do not depend on the position. For pure text the three MRoPE sections have the same position, which is equivalent to 1-D RoPE [S1][S18 §2].
- How positions are handled is itself part of the rule, in one of three ways:
  1. **Rotate q again to a reference position** (for example the current end L), as if "the Goal were the next query". Rotations compose, exactly over the reals.
  2. **Use only the 192 unrotated dimensions**, for a position-independent score.
  3. **Rotate K back by each token's position**, one more 64-d rotation per key.

**Also for the rule to decide:**

- how the 24 q heads are aggregated (every 6 share one KV head): max, mean, or by softmax mass;
- which layers;
- whether to softmax-normalize: if so, two passes over all keys, or an online logsumexp.

### 4.4 Bandwidth and time

The number of non-resident pages is approximated by n − 512 (512 is the capacity).

| History | Non-resident pages | K read, 1 layer | Time, 1 layer | K read, 16 layers | Time, 16 layers |
| --- | --- | --- | --- | --- | --- |
| Typical, 1,906 pages | 1,394 | 43.6 MiB | 3.4–4.8 ms | 697 MiB | 54–76 ms |
| Upper bound, 16,384 pages | 15,872 | 496 MiB | 39–54 ms | 7.75 GiB | 620–868 ms |

**How the table is computed [estimate]:** pages × 32 KiB per layer, divided by 13.43 to 9.58 GB/s. In the arena each page's K of a layer is a contiguous 32 KiB, with a stride of 64 KiB [S16].

**Limits of the table:**

- 13.43 to 9.58 GB/s is the effective rate of today's restore path (staging plus one `index_copy_` per tensor), not a link limit.
- The table also assumes that these 32 KiB pieces reach the rate of a whole-page restore.
- The path where the device reads the pinned arena directly (the other path B1 measures) has not been measured, and could move these numbers a lot in either direction.

**Other:**

- **Compute:** 24 × 256 = 6,144 multiply-adds per token per layer; about 2.6 × 10^10 multiply-adds over 16 layers at the upper bound [estimate], not measured on a GPU.
- **Resident pages:** their K is read directly in HBM; the staging is sliced by `MAX_STAGING_BYTES` 64 MiB [S16]. The scoring could also read the pinned arena directly on the host CPU and save the H2D, but the CPU throughput has not been measured.
- **Timing:** the scoring must finish before the plan. Two placements:
  - **inside the phase-1 worker call:** the Goal forward is in that call anyway [S6], and the scoring follows it. No extra RPC, but phase 1 gets longer.
  - **a separate worker pass plus one RPC.**

  Either way the scores have to return to the engine: one float32 per token, 1 MiB at 262,144 tokens. For comparison: the admission_engine_ms p50 of selective requests is 245 ms [S26].

### 4.5 Memory

- no new resident storage, only the Goal q cache (192 KiB per cached Goal);
- only temporary HBM use: the staged K (sliced at 64 MiB) and the Goal's q.

### 4.6 Code changes

- **Model file:** an overlay of `vllm/model_executor/models/qwen3_next.py` that captures q in `Qwen3NextAttention.forward` only when a probe flag is on.
  - Both the fused and the eager path must be covered [S2], and the default path must stay exactly unchanged. This is vLLM's model file: the highest risk.
  - With `--phase1-graph-tokens` the probe forward may run under a CUDA graph [S4], so the capture must be graph-safe.
- `native_probe.py`: collect q in the Goal forward or in phase 1.
- **Worker scorer:** a new one, beside `tensors.py`: read K from the resident blocks and the arena, score layer by layer.
- `model_queries.py`: the scores return with the receipt, outside the signed body.
- **Engine and the rest:** `query_selection.py` receives the scores and hands them to the picker; `token_gather.py`; the contract and startup options; evidence.

### 4.7 Risks

- The change is in model code that every forward goes through.
- The bandwidth grows linearly with the history; the admission critical path or phase 1 gets longer.
- **Parity:** K is not in the evidence (K+V is 64 KiB per token, about 1.86 GiB for a 30,496-token session [estimate]).
  - So the per-token scores must be recorded in the evidence to replay the fill: float32 binary is 119 KiB per request (typical) or 1 MiB (upper bound); as JSON, about 0.58 MB or 5.2 MB [estimate / local CPU proxy].
  - The scorer itself can only be validated by a GPU probe against a dense reference. This is the same situation as goal_query today, which "replays only the gather, not the Goal query" [S31].

### 4.8 Validation

**CPU tests**

- On a toy RoPE attention layer (the `GatheredAttentionTests` set): the re-rotated q gives the same logits as dense attention on the recorded tokens, within 1e-12.
- Cover partial rotation and the GQA head mapping.
- K read from the arena equals the rows on the device.

**GPU probe (G1 style)**

- Capture the q of 16 layers on the real model and compare with a torch dense reference, with K from both resident and DRAM pages.
- Time each layer at 30K and 262K tokens; also validate in graph mode.
- Then shadow mode on 1 task.

---

## 5. Option 3: two stages (a page-cosine shortlist, then per-token scores inside it)

### 5.1 How

1. Stage one: the top S pages by today's page cosine (1.2 ms for 1,906 pages [local CPU proxy]).
2. Stage two: score only the tokens of these S pages. Picks = the room × 16 highest-scoring tokens in the list; the fill still uses `shared_blocks_v1`.

### 5.2 When it makes sense

- **With option 2:** the K read is bounded by S × 32 KiB per layer, independent of the history length. At S = 2 × room ≈ 631 pages, 19.7 MiB for 1 layer (1.5–2.2 ms), 315.5 MiB for 16 layers (25–35 ms) [estimate: the method of 4.4]. Part of the list is resident, so this is an upper bound.
- **With option 1:** it saves scoring time only, not storage: the vectors must be captured in the forward, when it is not known yet which pages will be on the list. Integer-exact scoring at the upper bound takes about 30–40 ms (converted from int8 every turn: 28.7 ms at 32 cores, 37.8–39.1 ms at 1–4 threads [local CPU proxy / audit re-measurement]), less with the resident f32 copy. So it is not worth doing.

### 5.3 The cost: coverage

Tokens outside the list can never be selected. Coverage = the share of the full rule's top room × 16 tokens that fall in the top S pages.

- **With option 2** (the combination that really makes sense): coverage needs q·k scores, from option 2's probe or per-token scores from the author (section 11, questions 7 and 10).
- **Prototype A (option 1 in shadow mode):** gives coverage only under the same-space cosine score.

### 5.4 Rules the algorithm side has to give

- how to choose S;
- which score the second stage uses (as in option 1 or option 2);
- whether the list includes the pages already taken whole.

### 5.5 Code and validation

- **Code:** on top of option 1 or 2, a cut in the plan, with S and the list recorded.
- **CPU test:** at S = n the result is byte-identical to the full one.
- **GPU:** as for option 1 or 2.

---

## 6. Option 4: reduced dimensions, quantization or precomputed scalars

**Lower-dimensional vectors** (for example 32-d or 64-d int8, 36 or 68 B per token against 132 B)

- A parameter of option 1. The binary 132 B per token is already small (only 33 MiB at 262,144 tokens), so fewer dimensions matter only for evidence size or to fit in HBM.
- The cost is a larger cosine error, which the rule has to accept.

**A few vectors per page** (2 or 4 per page, the last token of each sub-segment)

- 2 or 4 times the storage, not 16 times.
- Only contiguous sub-segments of 8 or 4 tokens can be selected, never single tokens.
- If the algorithm side accepts "short segments", this is a middle step between pages and tokens: the idea of exp-subpage-score taken below 16 [S29].

**Scalars precomputed at capture**

- The Goal is cached by `retrieval_goal_hash` and reused on a hit [S11]; but on a cache miss the Goal changes. More fundamentally, at capture the Goal of later turns does not exist yet. So a Goal-dependent score cannot be precomputed at capture. As a way to implement "the per-token Goal score", **this does not hold, and is dropped.**
- Only Goal-independent priors can be precomputed, for example a token's class, or its cosine with its page's vector. If the rule is "the page's Goal cosine + a prior of the tokens in the page", it can be done with 4 B per token. Whether that is usable is the algorithm side's call.

**Caching the scores while the Goal does not change**

- On a hit the old tokens' scores can be reused, and only the new tokens scored.
- The integer-exact scoring is already fast, so this optimization gains little. The hit rate can be counted offline from existing evidence (the receipt's `goal.cache_hit`) [S11].

---

## 7. Option 5: contract, receipt, evidence and parity (needed by every option)

**Contract**

- A new key, or fields added to the `token_gather` pin: `{vector, score, query, picker, quota, pad, budget}`, with `vector` the representation and dimension, `score` a rule id (such as `int_cosine_v1`), `query` `goal|observation`.
- It appears only when enabled; every other contract and digest keeps its bytes. This is what token gather and page_score do [S15][S29].

**Engine receipt (signed body)**

- Keep `gather {tokens, pad}`: at most about 5,051 positions per turn, about 30–35 KB of JSON [estimate: about 6–7 bytes per position].
- `ranking` stays the page ranking; `selection.scores` stays the scores of the n pages [S11].
- New: the threshold (the score of the last selected token), counts, and the int8 query (132 B). With the int8 query, a per-token replay needs no query recovery.
- **Do not** put the per-token scores into the signed body:
  - 5.2 MB per request at 262,144 tokens, at least 184 ms to build and serialize [local CPU proxy];
  - the signed body is HMACed once at issue and verified once at reservation [S37], and the audit measured about 110 ms more per pass of "sorted JSON + HMAC" for such a body [audit re-measurement, S34].

**Worker frames (evidence)**

- The per-token vectors stay outside the signed body. The precedent is exp-subpage-score's `subpage_vectors`: outside the HMAC body, with no new per-request hash [S29].
- **Trade-off:** these vectors are therefore not authenticated, while the page vectors are. A deliberate trade-off, consistent with "integrity checks at startup only".
- **Format:** binary is suggested (base64-encoded int8 + fp32, about 176 bytes per token [estimate: 132 × 4/3]) rather than a JSON list (456 B per token [local CPU proxy]).
- **Evidence growth per session** [estimate]:
  - 13.9 MB (JSON) or 5.4 MB (base64) for a typical session; 44.6 MB (JSON) for a 97,778-token session;
  - republished at every rebuild (gath32ti had 35 rebuilds [S25]);
  - for comparison, a comment in the parity tool says a single native receipt was at most about 1 MB at perf-swe16 [S30].
- **Option 2:** also records the per-token scores (4.7).

**Parity and replay**

- **Option 1:**
  - **observation_query:** rebuild each session's token catalog from the frames, recompute the scores with the pinned integer-exact cosine, get the picks, run `token_gather.turn`, and compare with the signed gather. For the observation query the existing tool already has `recover_query` for the GH200 fp32 last-bit differences [S30]; with the int8 query in the receipt no recovery is needed.
  - **goal_query:** today's parity tool refuses it outright (`selection_parity.py:167-168`: "is not replayed (no rewriter here)") [S30]; the gather harness, too, replays the gather only on the signed scores [S31]. A full goal_query replay needs this tool extended first. The Goal vector itself is in the phase-1 frame, and a cache hit refers to an earlier frame; with the int8 query in the receipt it can be replayed directly.
- **Option 2:** K is not in the evidence, so only the fill can be replayed, on the recorded scores; the scorer is validated by a GPU probe.
- **`tools/policy_parity.py`:** byte-identical when disabled.

**Hashing:** no new per-request or per-decode-step hash; integrity checks run at startup only.

---

## 8. Approaches considered and rejected

| Approach | Why it does not hold |
| --- | --- |
| Smaller pages (P < 16) | FA's smallest kernel block is 16 [S19 §2] |
| Per-head selection | one layer group shares one block table [S18 §9] |
| Recomputing hidden states afterwards | needs a rerun of 64 layers over the whole context before the position (GDN state included): redoing the prefill |
| Letting the Goal forward attend with the whole history's KV | needs every page restored to HBM: no selection at all |
| Precomputing Goal-dependent token scores at capture | at capture the Goal of later turns does not exist yet (section 6) |

---

## 9. Comparison

| | Option 1 (binary) | Option 2 (K, 16 layers) | Option 3 (K + shortlist, S = 631) | Option 4 (fewer dimensions / prior) |
| --- | --- | --- | --- | --- |
| Needs from the algorithm side | whether the rule is the same-space cosine | layers, heads, RoPE, aggregation | S, the second stage's score | whether a precision loss or a prior is acceptable |
| New storage (host memory, per session) | 132 B per token: typical 3.8 MiB, upper bound 33 MiB; plus a 512 B per token f32 copy for fast scoring: 14.9 MiB / 128 MiB | none (Goal q cache 192 KiB per Goal) | as option 2 | 36–68 B per token, or 4 B per token |
| Scoring per turn, typical | only 132 B stored: 3.2 ms; with the f32 copy: 0.3 ms; plus top-k 0.7 ms [local CPU proxy, 32 cores] | K read 54–76 ms + compute (not measured) [estimate, at today's restore path's rate] | 25–35 ms (upper bound) [estimate] | as option 1 |
| Scoring per turn, upper bound | only 132 B stored: 28.7 ms (1–4 threads 37.8–39.1); with the f32 copy: 4.2 ms (1–4 threads 6.2–9.7); plus top-k 3.3 ms [local CPU proxy / audit re-measurement] | 620–868 ms [estimate] | 25–35 ms [estimate] | as option 1 |
| Capture | every forward: phase 1, prefill, decode | q captured in the Goal forward (or phase 1) | as option 2 | as option 1 |
| Gather (common to all options) | at most about 5,051 tokens per turn: about 141 ms on the existing path (G1 two-point extrapolation, about 5 times the measured range), 25–35 ms at restore throughput [estimate] | same | same | same |
| Largest single cost | the gather | the K read (same order as the gather or larger) | the gather and the K read, same order | as option 1 |
| Change surface | worker capture, engine catalog, picker, wiring, parity tool (medium) | model-file overlay, worker scorer, score return (large) | option 2 + cut (large) | a parameter of option 1 (small) |
| Offline replay | complete; goal_query needs the parity tool extended | only the fill, on the recorded scores | only the fill, on the recorded scores | complete |
| Main risk | rule mismatch | bandwidth, changing the model forward, validation | coverage | precision |

---

## 10. Recommendation: what to prototype first

1. **Now (no GPU needed)**
   - Send the questions of section 11 to the goal_query study's author.
   - Ask the author for per-token scores of a few requests in the format of question 10, to check our implementation later and to compute the coverage with option 2.
2. **Prototype A: option 1 in shadow mode**
   - **What:** capture, binary transport, numpy catalog, integer-exact cosine; record only the top-k, the threshold, the overlap with the page fill, and the coverage at S = room, 2 × room, 4 × room. The selection is not changed.
   - **What does not depend on the rule:** capture, transport and storage do not; the recorded statistics apply only to one rule, the same-space cosine.
   - **Reasons:**
     - It is isomorphic to the existing page-vector path: a page vector is the vector of each page's last token.
     - It can be replayed offline completely (goal_query needs the parity tool extended first).
     - The local CPU proxy costs of host memory and engine scoring are small (3.4, 3.5); the GPU capture cost (phase 1, every decode step) is not measured yet, and A measures exactly that.
   - **Questions it answers:** how far a per-token same-space cosine selection is from today's page fill (overlap, tokens outside the pages), and the two-stage coverage under that score.
   - **Questions it does not answer:** whether it is "better": only runs that change the selection can answer that.
3. **Prototype B1: a GPU probe of large-gather throughput (independent of the rule, now, can run beside A)**
   - Extend G1 to 5,051 tokens with a real distribution: scattered over 1,000+ pages and several arena chunks, instead of G1's 100–600 pages.
   - Compare the existing path with the path where the device reads pinned memory directly.
   - Time it in a step with concurrent decode.
   - Reason: whatever the scoring, a per-token fill may gather a few thousand tokens per turn (2.2).
4. **Prototype B2: the reuse rate of an incremental gather (after A)**
   - Which tokens repeat across turns depends on the rule; synthetic data cannot measure it.
   - Count the turn-to-turn repeat rate with A's recorded shadow selections (or an offline replay of a rule), then decide whether to implement the incremental gather.
5. **Once the rule is confirmed**
   - If the rule is the same-space cosine, or can be computed from the token vectors: add the picker on top of prototype A, wire it into token gather (it can combine with markers), run a 1-task smoke first, then 32 tasks.
   - If the rule is attention-based (q·k): only then start option 2. First a Goal q capture plus dense-reference probe (which also gives the coverage under q·k scores), and bound the bandwidth with option 3.
6. **Option 4 is not a project of its own**; when needed it is a parameter of option 1.

---

## 11. Questions for the goal_query study's author

1. How exactly is a token's Goal score computed in "Ours-token"?
   - (a) The cosine of the Goal vector with the token's hidden state: which layer? Before or after the final norm? Full dimension or projected?
   - (b) The attention of the Goal (or its last token) as a query over the history's K: which layers, which heads? How aggregated? Softmax or not? At which position is the Goal placed (RoPE)?
   - (c) Something else (an external embedding model, etc.)?
2. Are the study's Goal text and ours the same (the maintained Goal's `<retrieval_query>` text and the rewriter)? Is it rescored every turn?
3. How does the study apply a token selection in its own stack?
   - Do the selected tokens keep their original positions (the original RoPE phase), or is a shorter prompt re-prefilled from the selected tokens?
   - Does the state of the linear-attention (GDN) layers still see the whole history? In our runtime the selection acts only on the 16 FA layers and keeps logical positions, and the 48 GDN layers keep the state of the whole history.
   - Also: single tokens only, or with a neighbourhood (±k-token smoothing, keeping short segments)? Is attending to the K/V of single selected tokens on their own the study's setting? (We do a single-token gather.)
4. The order relative to conditions E and C: is it prefix → recent → markers → Goal tokens? Do the Goal tokens exclude markers and separators?
5. Was "a Goal score per 16-token block" (one score per block, the block kept whole) compared with per-token scoring? If the difference is small, today's page fill is enough. This is the same kind of question as question 3 of section 5 of main's token-selection note [S19].
6. Precision: is the 128-d int8 projection acceptable, or is the full dimension required? The full dimension is 10 KiB per token; the cost is in 3.7.
7. Is an approximation acceptable: per-token selection only within the top S pages by page cosine (two stages)? How large is S? Could the coverage be measured on the study's own stack?
8. How are ties handled? We need to pin it for determinism.
9. Is thinking text (reasoning tokens) a candidate?
10. Could you provide per-token scores for a few requests as check data? For each request please give:
    - the request id;
    - the tokenizer used;
    - the positions in the full prompt's token ids;
    - the score of each position;
    - the finally selected set.

    Only then can they be aligned with our tokenization.

---

## 12. Candidate approaches to discuss

These are general descriptions of published methods, not named. A new citation needs the maintainer's approval.

- **Accumulated attention as a prior:** the accumulated attention of the last few queries on each key as a per-token importance; select the high-scoring tokens. This is a Goal-independent prior: option 4's prior, or option 2 with the observation's q.
- **Block-level upper-bound filtering:** keep per block the element-wise minimum and maximum of the keys (or one representative vector), compute an upper bound of q·k from it to pick blocks, then select tokens inside the blocks. This amounts to replacing option 3's first stage with a K-based upper bound; it stores 2 × 4 × 256 × 2 B = 4 KiB more per page per layer [estimate].
- **Approximate retrieval:** cluster or product-quantize the keys and retrieve the high-attention tokens with approximate nearest-neighbour search. It corresponds to option 4's reduced dimensions.
- **Per-layer, per-head token selection:** out of the runtime's reach, because one block table is shared by all heads of the same group [S18 §9].

---

## Appendix A: Local CPU proxy measurements

Measured for this note on one x86 Windows machine (Python 3.12.7, numpy 1.26.4, 32 logical cores; not a GH200), for orders of magnitude only. The thread count was not limited, so numpy's BLAS used all 32 logical cores; the independent audit (2026-11-15) re-measured on the same machine at 1 and 4 threads [S34].

- **What was timed** [S32]: `dram_kv/page_vectors.py` and `remote_ra_live_v2/fast_selection.py` of `integrate-1114`, loaded by file path (no package import; nothing written to a worktree): per vector, the wire form, the Python representation and its ingestion (`json.loads`, construction and checks, `unit_values`); signing 406 and 6,500 vectors; `fast_selection.cosines` over 1,906, 16,384, 30,496 and 262,144 columns; building and serializing the evidence `scores` dict; the integer-exact cosine from int8 and from a resident f32 copy, with the top-5,040 sort and a full sort; float32 against int64 dot products.
- **Fill check** [S33]: the property of section 2.1 on the runtime's own `token_gather.plan`, 2,000 random trials.

## Appendix B: Changes after the independent audit [S34]

**Required (11, all done)**

| # | What the audit required | Where it changed |
| --- | --- | --- |
| 1 | "The gather is larger than the scoring" holds for option 1 only | section 0 item 2 stated per option; section 9 gains a "largest single cost" row |
| 2 | Prototype A can measure only the difference and the coverage under the same-space cosine; coverage under q·k needs option 2's probe or the author's scores; "how much better" removed | section 0 recommendations, 5.3, section 10 item 2 |
| 3 | Prototype B split: the throughput probe now, the reuse rate after A | section 0 recommendations, 2.2 countermeasures, section 10 items 3 and 4 |
| 4 | 0.29 / 4.2 ms needs the resident f32 copy of 512 B per token | section 0 item 3, a new row in 3.4, 3.5, section 9 |
| 5 | Bit-identical reproducibility presupposes the int8 query | section 0 item 3, the qualifier in 3.5 |
| 6 | "At most about 1 MB" | section 7 |
| 7 | The goal_query replay needs the parity tool extended (`selection_parity.py:167-168` refuses it) | section 7, 3.6, section 9; S30 gains these lines |
| 8 | 4.3 changed to 4.2 in section 9 | section 9 |
| 9 | 141 ms marked as a two-point extrapolation, about 5 times the measured range | section 0 item 2, 2.2, 3.5, section 9 |
| 10 | Ask how the study applies the selection and whether GDN sees the whole history; state that every option acts on the FA layers only | the header's "common premise", 1.6, section 11 question 3 |
| 11 | State the BLAS thread count; "at most 29 ms" changed to about 30–40 ms | the header, 3.5, 5.2, section 9; S32 gains the thread note |
| — | S29's commit note (feature commit `32f36a1`, branch HEAD `5a8453f`) | S29 in Appendix C |

**Recommended (20, all adopted, none skipped)**

| # | Content | Where it changed |
| --- | --- | --- |
| 1 | The picker's `view` has no room; two ways and the cost of `turn()` | section 0 item 1, 1.4, 2.1, 3.6 |
| 2 | HBM with 16 sessions; the share of time the whole batch is held up | 2.2 impact |
| 3 | The capacity gate already reserves for the worst case | 2.2 capacity gate |
| 4 | `UVA_ADVISED` is only a threshold rule | section 0, 1.4, 2.2 |
| 5 | The G1 line's intercept is unexplained; the audit's host-index proxy measurement | 2.2 time table |
| 6 | Uniform units in each table; the real reason the Python path is not feasible | 3.4 |
| 7 | `pre_final_norm` already has an fp32 copy; chunking limits only the float64 temporary | 3.3 |
| 8 | 71 ms and < 1 ms marked as estimates; where 1,500 comes from | section 0, 3.5 |
| 9 | The rate behind the K read times, and its limits | section 0 item 4, 1.4, 4.4 |
| 10 | Option 2 can join the phase-1 call; the q cache; graph-safe | 4.2, 4.4, 4.5, 4.6 |
| 11 | 5,051 is an upper bound | section 0, 2.2 |
| 12 | "At least 184 ms", plus about 110 ms each for signing and verification | section 0 item 7, 3.5, section 7 |
| 13 | Option 2 must record the scores, and their size | 4.7, section 7 |
| 14 | "The Goal changes every turn" rewritten; aligned with the "fewer dimensions" sentence | section 0 item 6, section 6, section 8 |
| 15 | Vectors outside the signed body are not authenticated | section 0 item 7, section 7 |
| 16 | Record the int8 query in the receipt | section 0, 3.5, section 7 |
| 17 | The signed receipt holds `selection.scores` too | 1.2, section 7 |
| 18 | Prototype A's wording (local CPU proxy costs; GPU capture not measured; statistics for one rule only) | section 10 item 2 |
| 19 | Wiring files and the parity tool; S29's commit note | 3.6, S29 in Appendix C |
| 20 | B1 with a real distribution, timed in a step; question 10 gives the data format | section 10 item 3, section 11 question 10 |

## Appendix C: Sources

The `[S#]` after each fact above is a row here. Code and doc locations are relative to the repository root at `integrate-1114` (HEAD `a565e84`, code `b388033`) unless the row says otherwise; the `main` docs are at `f18b8e5`. GPU readings come only from verified results, named by job id; the local measurements are S32 and S33 (Appendix A); the independent audit and its re-measurement are S34. Sources outside this repository are named without a path.

Revision (2026-11-15, after the audit): line numbers or notes added to S4, S11, S15, S26, S30, S32; S29's commit note corrected; S34–S39 added (every new code location checked line by line).

**Model and attention**

| ID | Fact | Location |
| --- | --- | --- |
| S1 | Qwen3.6-27B: `hidden_size` 5120; `num_hidden_layers` 64; `full_attention_interval` 4, 16 `full_attention` in `layer_types` (layers 3, 7, …, 63), 48 `linear_attention`; `num_attention_heads` 24; `num_key_value_heads` 4; `head_dim` 256; `partial_rotary_factor` 0.25; `mrope_interleaved` true, `mrope_section` [11, 11, 10]; `rope_theta` 10,000,000; `dtype` bfloat16; `max_position_embeddings` 262,144; `attn_output_gate` true | `source-overlay/remote_ra_linux/model-evidence/model-config.json:11, 13, 15, 16, 18, 21-86, 93, 97, 98, 99, 102, 105-110, 112` (the FA layer count is counted from `layer_types`) |
| S2 | `Qwen3NextAttention`: `scaling = head_dim**-0.5` (240); `get_rope` (264); per-head `q_norm` / `k_norm` (297-298); when the fused qk-norm-RoPE path is on (304-309); `_project_qkv_gate` returns the post-norm, post-RoPE q, k, v (311-365); `forward`: `qkv_proj` → `_project_qkv_gate` → `self.attn(q, k, v)` writes the paged cache (367-377) | `source-overlay/vllm/model_executor/models/qwen3_next.py:240, 264, 297-298, 304-309, 311-365, 367-377` |

**Page vectors and the Goal vector**

| ID | Fact | Location |
| --- | --- | --- |
| S3 | The vectors take the output of the last decoder layer, before the final RMSNorm; the hook gets the row after the norm and divides by the per-channel gain to recover the direction; only vector consumers see these rows | `source-overlay/remote_ra_live_v2/vector_space.py:1-9, 46-51` |
| S4 | `after_forward` hands the `pre_final_norm` rows to the bridge and the probe (40-58); `force_eager` / `probe_piecewise`: with `--phase1-graph-tokens` a probe-only pass can replay a piecewise CUDA graph (61-77) | `source-overlay/remote_ra_live_v2/hooks.py:40-58, 61-77` |
| S5 | Decoupled layout: the rows of the pages one forward closes are compressed together on the device (projection 5120→128, L2, int8 per-vector scale), and only 128 int8 + 1 fp32 scale come back; wire form `{page: [[128 ints], scale]}`; at P = 16 a 6.5K observation closes about 406 pages, and the old per-page D2H + `tolist` + Python normalization costs about 1.5-5 ms per page; the result does not depend on the batch shape or the device, except within the rounding error where a float64 value sits at an int8 or fp32 rounding edge (about 1e-9 per page) (14-17); `compact` uses a float64 linear and `.cpu().tolist()` back to the host; `closing` takes the page-end rows with a strided view | `source-overlay/remote_ra_live_v2/page_compaction.py:1-25, 97-113, 115-128` |
| S6 | `ProbeRows`: decoupled, it keeps only the page-end rows (one device clone) and the last row (host fp32); legacy copies every row, about 133 MB for a 6.5K observation; the Goal forward starts cold from `start=0` on its own lease | `source-overlay/remote_ra_live_v2/native_probe.py:12-35, 184-191, 385` |
| S7 | The pages each phase-2 forward closes (generation header, cold prefill, decode): decoupled, compressed together on the device (`compactor.closing`); pages past the `--thinking-history strip` cut are not published; `publish_pages` publishes the pages closed in [phase2_start, cursor) | `source-overlay/remote_ra_live_v2/qwen.py:444-466, 485-497` |
| S8 | Phase 1: the closing pages' vectors = `compactor.compact(rows.closing)`; the observation vector = the last row, normalized; the Goal vector = the last row of the Goal forward, normalized (5,120 floats); the receipt body holds `page_vectors`, `observation_vector`, `goal_vector` and is verified once right after signing | `source-overlay/remote_ra_live_v2/model_queries.py:109-132, 133-146, 156-176` |
| S9 | `publish_vectors`: decoupled, int8 vectors, "~0.6 KB of JSON per Page", the receipt MAC covers those bytes; `sign_receipt`: `json.dumps(sort_keys=True)` + HMAC-SHA256 | `source-overlay/remote_ra_live_v2/worker_backend.py:369-386, 396-402` |
| S10 | The engine's `Phase1Result` carries `page_vectors`, `observation_vector`, `goal_vector` | `source-overlay/remote_ra_live_v2/phase1_coordinator.py:353-374` |
| S11 | `QuerySelectionAuthority`: page vectors cached per session (never changed once sealed; the comment records the old per-page per-turn cost) (495-500); decoupled, the worker has compressed and the engine does not project; the numpy fast path and the per-session column matrix (504-517); `memo_types` memoizes `unit_values` (183-226); `page_vector` / `query_vector` (605-618); the plan takes the fast path first, then `gathered` (662-668); `gathered`: calls `token_gather.turn` in Selector V2's cosine order (674-703); `plan_fast`: the float Goal compressed to int8 on the CPU (947-953), `fast_selection.cosines` (966), sorted by (-cosine, Page) (972), the evidence `scores` rounded to 8 decimals per page (1007); `plan_turn`: the Goal cached by `retrieval_goal_hash`, no Goal forward on a hit (1565-1577), the signed receipt body holds the `ranking` of all pages and the `selection` (the evidence dict, the 8-decimal `scores` included) (1584-1598), with `goal.cache_hit` (1593), and `gather` (1599-1602); `publish_completion`: a sealed page's vector committed once (1622-1633); the receipt MAC verified at reservation (1635-1645). A float query is compressed on the engine CPU by the fp32 `selection_contract.compact` (952-953) | `source-overlay/remote_ra_live_v2/query_selection.py` (lines as on the left) |
| S12 | `fast_selection`: a vectorized cosine bit-identical to upstream (the reference path about 10 µs per page, 0.3-0.5 s per admission at 8-16K pages); `cosines` is a compensated summation over the 128 dimensions in order (step by step equal to CPython's `sum()`); `PageColumns`: one 128×n float64 column matrix per session | `source-overlay/remote_ra_live_v2/fast_selection.py:1-27, 45-87, 90-128` |
| S13 | Contract: a 128-d random ±1 projection with seed 20260904 ("seeded Rademacher"), L2, int8 per-vector quantization (14-18); decoupled, the worker compresses; the wire form (20-28); `PROJECTION_DIMENSION`, `PROJECTION_SEED`, SHA (92-99); `capacity_pages` (380-391); `mandatory_pages` (394-397); `projection` / `compact` (626-648); `compact_batch` (651-670) | `source-overlay/remote_ra_live_v2/selection_contract.py` (lines as on the left) |
| S14 | Upstream `CompactVector`: int8 values + a positive scale; `unit_values` = the values × scale divided by their L2 norm (the scale cancels mathematically); `compact_cosine` = the dot product of the two unit vectors, clamped to [-1, 1] | `source-overlay/dram_kv/page_vectors.py:94-132, 443-453` |

**Token gather**

| ID | Fact | Location |
| --- | --- | --- |
| S15 | `token_gather.py`: constants (`PAGE_TOKENS` 16, `QUOTA_LIMIT` 65,536 and others, 59-72); the `TokenPicker` interface (`picks(view)` returns positions in priority order) (76-100); `newest` (124-133); `markers_v1` / `separators_v1` (136-200); `register_picker` (205-210); `fill` (shared_blocks_v1, 399-423); `pad` (426-441); `plan` and the record (444-459); `turn` (462-479; the picker's `view` holds only stream, end, page_tokens, eligible, state, encode, separator, pin, no room, 468-474); the checks of `plan_entry` (484-497); `sources`: resident pages read on the device, the rest from DRAM pages (506-520); `cost`: 64 KiB per token, 1 MiB per page (523-526) | `source-overlay/remote_ra_live_v2/token_gather.py` (lines as on the left) |
| S16 | `PageArena`: preallocated pinned host chunks, one [slots, block shape without the block axis] buffer per FA tensor (so each page's layer is a contiguous `[2, 16, 4, 256]`, K the first 32 KiB), `CHUNK_BYTES` 256 MiB (335-377); `MAX_STAGING_BYTES` 64 MiB (612-618); `restore_pages` (662-710); `gather_tokens`: device rows indexed once, host rows collected on the host into one bounce buffer, one H2D per FA tensor, one `index_put_`, one join at the end (731-824) | `source-overlay/remote_ra_growing_worker/tensors.py` (lines as on the left) |
| S17 | Policy registry: the query plug-in hook, `FillContext` (with `page_vectors`, `tokens`, `encode`), `hooked_fill`; "Token-level selection is out of scope" | `source-overlay/policy_registry.py:1-50 (47), 107-155, 173-230` |
| S18 | The token gather design note: section 1 what it does (7-23); section 2 why it is exact: K stored after RoPE, q at logical positions, prefix order does not matter, the 48 GDN layers untouched, no single slot can be masked (25-33); section 3 the shared_blocks_v1 budget, contract, receipt, evidence (35-43); section 4 the gather inside the first phase-2 step, holding and release across turns, phase 1, the capacity gate (45-53); section 5 cost (55-61); section 6 refusals (63-68); section 7 offline replay (70-83); section 8 parity and tests (85-89); section 9 open questions: per-token Goal scoring, per-head selection, incremental gather (91-96) | [`docs/token-gather/DESIGN.md`](DESIGN.md) (its Chinese twin, [DESIGN.zh-CN.md](DESIGN.zh-CN.md), has the same sections) |
| S19 | The token-level selection note on main: section 1 the study's setting (14-21); section 2 constraints: 16-token pages, FA accepts only 16/32/64 blocks, only one vector per page (23-33); section 4 "Not covered. Per-token Goal scoring…16 times today's count, and the exact scoring rule of the goal_query work" (66-67); section 5 question 3 (the "keep the marker's 16-token block" ablation) and question 4 (75-77) | `docs/token-selection/DESIGN.md` on `main` (`f18b8e5`; line numbers as of that commit; not on this branch: its content is now in [docs/policy-registry.md](../policy-registry.md) section 6) |
| S20 | Policy registry doc, section 6: three ways to token-level policies and their costs; section 9, not done: "per-token Goal scores (one vector per Page)" | `docs/policy-registry.md:249-278, 458` as of `a565e84` (since rewritten: the token-level note is now [docs/policy-registry.md](../policy-registry.md) section 6, the not-done list section 9) |
| S21 | Design doc: phase 1 produces the closing pages' vectors and the observation query (19); the vector definition (pre-final-norm, 128-d projection, seed, SHA, int8) (34); GPU batched against CPU per-vector compression: about 0.35 % of vectors differ in 1 element (47) | `docs/design.md:19, 34, 47` |
| S22 | README: a new page-vector score and token-level policies need more runtime support | `README.md:192` as of `a565e84` (the README has since been rewritten; it points to [docs/policy-registry.md](../policy-registry.md) section 6 for token-level selection) |

**GPU readings (verified)**

| ID | Fact | Location |
| --- | --- | --- |
| S23 | G1 gather probe (NVIDIA GH200 120GB, HND): arena `chunk_slots` 256, `page_bytes` 1,048,576, pinned; `c_dram_gather_median_ms` 69.0626, `c_dram_host_index_median_ms` 59.2726, `decision` `UVA_ADVISED`; E set from DRAM: 144 tokens, `gather_bytes` 9,437,184, gather median 53.4005 ms, host index 49.8355 ms, restore of 84 pages 9.1912 ms, `restore_bytes` 88,080,384; E set resident, 73 pages: gather 3.4555 ms; C set from DRAM: 1,024 tokens, gather 69.0626 ms, host index 59.2726 ms, restore of 498 pages 38.8763 ms, `restore_bytes` 522,190,848; C set resident, 506 pages: 15.4129 ms; `threshold_ms` 10; the device | GPU job 3603240: the probe's `gather.json` |
| S24 | gath32ti (goal_query + `--token-gather markers`): counts: 1,210 selective requests, picks mean 133.755, gathered mean 110.581 / p95 317 / max 462, blocks mean 7.274, host_tokens 126,474, device_tokens 14,358; guardrails: admission_engine_ms p50 230.212, `gather_time` NOT_RECORDED, latency_s p50 8.549, restore_h2d_mib mean 226.351; capacity waits 0 | GPU job 3603443: its `gather.json` |
| S25 | The prompt-length distribution of gath32ti: [0, 32,768) 1,257 calls, [32,768, 65,536) 237, [65,536, 98,304) 65, longer 0; table 1: 1,559 calls, 1,210 selective turns, longest prompt 97,778 | GPU job 3603443: its prompt-length table and table 1 |
| S26 | The audit of gath32ti: selection_ms; selective-request admission_engine_ms p50 244.99; latency; W1: non-phase-1 H2D = k×146.8 MiB (GDN state) + restored pages × 1 MiB + DRAM-gathered tokens × 64 KiB, the gather 2.24 %, 5.1 MiB per request; W5: the gather runs inside the first phase-2 step, no per-request gather time, each call generates 358.5 tokens; Slurm `--mem 113560M`, 16 CPUs, 1 GPU; pool wall 2,845.1 s; 16 sessions at once, thinking on, max_tokens 4096; 35 rebuilds, longest prompt 97,778 | GPU job 3603443: the audited readings of its evidence |
| S27 | gath32ti: capacity 512 pages per request; mean fill room 315.7 pages | GPU job 3603443: derived from the same audit |

**Our own earlier notes and experiments**

| ID | Fact | Location |
| --- | --- | --- |
| S28 | The token-level design note of 11-13: the study's setting (10-23 discusses per-head selection; 11-04 the author on their own native Mini-SWE stack, H200, not our runtime; 11-11 "Ours-token": 8,192 budget, thinking on, protect the system prompt + the first user message and the last two turns up to 1,024 tokens, "remaining slots go to tokens with the highest Goal scores", conditions B–E); goalvllmt (job 3576565) mean candidate pages 1,906, obs262t 1,492; mandatory mean 143.5 / 138.9 pages, recent 49 / 48, room 319 / 325 | our token-level design note of 2026-11-13 (not in this repository) |
| S29 | exp-subpage-score (feature commit `32f36a1`; the HEAD of the local branch `exp-subpage-score` is `5a8453f`, which only changes the timeout of `check_git_release.py`; not pushed, CPU only; designed in 2026-10, P = 784): each 784-token page is scored by all its 16-token sub-pages; the last sub-page is the page vector; the sub-page vectors are projected on the device and put outside the signed body of the phase1/complete receipts, with no new per-request hash; why project on the GPU (a row is 20 KiB, Python normalization about 2 ms per row); engine cost, a typical turn about 1,500 tokens, evidence growth, "No hashing is added"; one step's `pre_final_norm` copy at most 16,384 rows; files changed: 6 in `remote_ra_live_v2`, the new module `subpage_score.py`, `serve.py`, README; 19 tests; a local measurement | commit `32f36a1` and the exp-subpage-score design note (not in this repository) |
| S30 | The selection parity tool: rebuilds the page catalog from the worker frames, takes the query from the phase-1 frame, runs both the fast and the reference path and compares with the record; the GH200 fp32 last-bit differences are recovered with a neighbourhood search of the query (1-19); replays only observation_query and random_fill, and raises "is not replayed (no rewriter here)" on any other policy, goal_query included (163-168); a comment says a real-agent pool server at perf-swe16 held about 2,000 native receipts of up to about 1 MB each (171-172) | `harnesses/perf-swe32/postrun/selection_parity.py:1-19, 163-168, 171-172` |
| S31 | The parity replay of the gather harness: goal_query's Goal query is not replayed; `token_gather.turn` is replayed only over the cosine order of the signed 8-decimal scores (`gather_only`) | the token gather harness's `postrun/selection_parity.py` (not in this repository) |

**Local measurements of this note (x86 Windows, not GH200)**

| ID | Fact | Location |
| --- | --- | --- |
| S32 | Local CPU proxy measurements (Python 3.12.7, numpy 1.26.4, 32 logical cores; the thread count was not limited, numpy's BLAS used its default, all 32 logical cores; the 1 / 4-thread re-measurement is S34): `dram_kv/page_vectors.py` and `remote_ra_live_v2/fast_selection.py` of integrate-1114 loaded by path (no package import). Wire form 455.9 B per vector (compact separators); Python representation 2,800.9 B per vector, 7,001.0 B with the memoized `unit_values`; `json.loads` 8.46 µs, construction + checks 16.91 µs, `unit_values` 21.95 µs (per vector); signing 406 vectors 187,620 B / 2.766 ms, 6,500 vectors 3,007,475 B / 45.741 ms; `fast_selection.cosines`: 1,906 columns 1.195 ms, 16,384 columns 6.368 ms, 30,496 columns 13.486 ms, 262,144 columns 179.195 ms; the evidence `scores` dict (build / JSON / bytes): 1,906: 0.63 / 0.60 ms / 33,944 B, 30,496: 10.47 / 8.96 ms / 580,173 B, 262,144: 106.15 / 77.48 ms / 5,233,706 B; integer-exact cosine: 30,496 columns int8 3.72 MiB, from int8 every turn 3.221 ms, resident f32 copy 0.286 ms, top-5,040 sort 0.71 ms, full sort 1.83 ms; 262,144 columns int8 32.0 MiB, 28.663 / 4.248 ms, 3.33 / 20.98 ms; float32 dot = int64 dot at both sizes (element by element); `PageColumns` 1,024 B per vector | Appendix A |
| S33 | A randomized check with the runtime's own `token_gather.plan`, 2,000 trials: with picks = the top room × 16 tokens, the taken pages are exactly the pages whose 16 tokens are all selected, every other pick is gathered, and the pad is 0 | Appendix A (output "2000 trials PASS") |

**The independent audit and its re-measurement (2026-11-15)**

| ID | Fact | Location |
| --- | --- | --- |
| S34 | Independent audit: 11 required fixes, 20 recommended. Re-measured on the same machine (x86, 1 / 4 threads, not GH200): wire form 453.7 B per vector; Python representation 2,737 / 6,937 B (with memoization); ingestion 54.0 µs (4 threads) / 60.0 µs (1 thread); signing 6,500 vectors 53.1 / 63.7 ms; `fast_selection.cosines` 262,144 columns 178.96 / 207.15 ms; integer cosine (from int8) 30,496 columns 3.60 / 4.02 ms, 262,144 columns 39.15 / 37.80 ms; resident f32 copy 0.269 / 0.527 ms and 6.152 / 9.655 ms, copies 14.89 MiB and 128.0 MiB; top-5,040 sort 2.72 / 2.83 ms (262,144); evidence scores at 262,144: build + JSON 118.8 + 87.8 ms, their sorted JSON + HMAC about 112.0 / 114.1 ms; float32 dot equals int64 dot (BLAS, permuted dimensions, sequential accumulation, torch f32, and bf16 inputs + f32 accumulation), a matmul with bf16 output does not; the integer cosine differs from upstream by at most 1.1e-16, the ranking unchanged at 30,496 columns, 2 positions (one swapped pair) at 262,144 columns, the top-5,040 set unchanged; host-index proxy (16 FA tensors, one unpinned 256-slot chunk) 144 / 1,024 / 5,051 tokens at 4.156 / 15.587 / 67.87 ms; about 190 Python objects per token. The fill check, 3,000 trials × 4 variants (`plan`, `turn` returning every candidate, markers first, quota), all hold; the cost of `turn()`: 30,496 candidates 6.7 ms (4.08 ms with only room × 16), 259,136 candidates 56.47 ms (7.11 ms) | an independent audit (2026-11-15), its re-measurement and its fill check (not in this repository) |
| S35 | Capacity gate: with a gather pin, `most = largest` (quota 0) or `min(largest, pages(quota, P))`, and restores = min(largest, non-resident pages + most); the comment says the gathered blocks are fresh blocks too and share the budget with the pages, and the previous turn's are released only at the commit | `source-overlay/remote_ra_live_v2/capacity.py:220-226` |
| S36 | In the first phase-2 step, `self.io.gather_tokens(...)` is called after the restores (synchronously); `gather_tokens` calls `self.drain()` at the end | `source-overlay/remote_ra_live_v2/allocation_owner.py:1281-1290`; `source-overlay/remote_ra_growing_worker/tensors.py:821-824` |
| S37 | The engine's native authority: `sign` = HMAC-SHA256 over the canonical digest `digest(value)`, `verify` computes it again and compares | `source-overlay/remote_ra_native_growing/contract.py:393-399` |
| S38 | The decision rule of the G1 probe: `UVA_ADVISED` when the C set's DRAM gather median exceeds 10 ms, else `ARENA_GATHER_OK`; a measurement only, it never fails the probe | GPU job 3603240 (the probe is in the token gather harness, not in this repository) |
| S39 | A Qwen release keeps 16,384 as max_num_batched_tokens (one native forward holds at most 16,384 tokens) | `serve.py:562-567` |
