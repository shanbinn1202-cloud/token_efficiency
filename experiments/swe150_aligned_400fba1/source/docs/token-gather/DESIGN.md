# Token gather: single history tokens beside the selected pages

English | [简体中文](DESIGN.zh-CN.md)

Branch token-gather-1114, from ledger-int-1112 (`110632b`); design of 2026-11-14. Off by default; a server without `--token-gather` is unchanged (its selection contract, receipts, plans and evidence keep their bytes; `tools/policy_parity.py` against `110632b`, section 8). Target: Qwen3.6-27B, thinking on, an 8,192-token budget; Gemma 4 is refused. Its GPU runs (jobs 3602601, 3603240, 3603443 and the bench 3611486) and their verdicts are in [docs/status.md](../status.md); the layout fix after job 3602601 is in section 6.

## 1. What it does

Selector V2 selects whole 16-token pages. The goal_query author's study keeps single tokens as well: in condition E every historical occurrence of four protocol markers, in condition C the newest 1,024 separator tokens. With `--token-gather` a `goal_query` or `observation_query` server keeps such tokens beside its page selection: a *token picker* names them, and at restore their K/V rows are copied (*gathered*) into fresh 16-token full-attention blocks that the request's row reads after the selected pages and before the run.

```sh
python -B serve.py --release-dir "$RELEASE" --task task.json --output "$OUT" --policy observation_query \
  --page-tokens 16 --active-token-budget 8192 --enable-thinking --token-gather markers
```

| Option | Meaning |
| --- | --- |
| `--token-gather markers` | picker `markers_v1`: every occurrence of each marker in the candidate pages outside the mandatory and recent ones, newest first |
| `--token-gather-marker TEXT` | a marker (repeatable; default `</think>`, `<\|im_end\|>` and `<\|im_start\|>user`). A marker is the text of a token sequence, encoded with the session's tokenizer; an occurrence keeps its first token, so `<\|im_start\|>user` keeps the `<\|im_start\|>` of every user message (an observation in `mswea_xml_v1`, which has no `<tool_response>`) |
| `--token-gather separators` | picker `separators_v1`: the newest separator tokens of those pages (decoded text non-empty and only spaces, tabs, newlines and `. , ? ! ; :`) |
| `--token-gather-quota N` | at most N picks per turn, newest first (markers: default 0, no quota; separators: default 1,024) |

Scope: `goal_query` and `observation_query` (the Goal or the observation cosine order), `--protocol mswea-xml`, `--page-tokens 16` (the decoupled layout). The launcher and the startup config refuse every other policy (query plug-ins included), protocol and layout, the Gemma 4 profile and `--release-idle-pages` (section 6).

## 2. Why the gathered attention is exact

- **K is stored after RoPE.** `qwen3_next.py` `_project_qkv_gate` returns the post-norm, post-RoPE q, k and v, and `forward` writes k to the paged cache (Qwen3.5 and 3.6 reuse this attention class). A gathered key therefore keeps the RoPE phase of its logical position. The query is rotated at its own logical position (`positions.py`: compact slots never replace RoPE positions). So q·k depends on the logical distance only; partial rotary (0.25) and the interleaved MRoPE sections change nothing.
- **Prefix order does not matter.** FlashAttention reads a request's keys through its block table up to the physical sequence length; the causal mask only separates the prefix (every block before the run) from the run, and softmax attention is invariant to the order of its keys. A gathered block anywhere in the prefix gives the same output.
- **Physical positions** move only the full-attention slot mapping and sequence lengths: the row is `[selected pages][gathered blocks][run]` and `SparsePlan(L, L − |A_t|·P − G)`, with G the gathered tokens.
- **The 48 Gated-DeltaNet layers** keep their recurrent state of the whole history (the carry). No selection, page or token, touches them.
- **No slot is masked.** FlashAttention with a block table attends every slot of a prefix block. An empty, zero or duplicated slot changes the softmax (a zero key adds exp(0) to the denominator), so the gathered count is padded to whole blocks with real history tokens: the *pad*, the newest candidate tokens that nothing else attends (`newest_unattended_v1`). The receipt records the pad with the picks; every slot of a gathered block holds a recorded token. A partial block before the run is impossible as well: the run's blocks were written in phase 1 at offsets fixed by their logical positions.

The CPU tests check this on a toy RoPE attention layer whose cache the runtime's own gather fills (`tests/test_token_gather.py`, `GatheredAttentionTests`): the gathered row equals dense attention over exactly the recorded tokens (selected pages, picks, pad, run) to 1e-12; moving prefix blocks changes nothing; a zero pad slot or a key rotated at its physical slot changes the output.

## 3. Selection, receipt and evidence

**Budget (`shared_blocks_v1`).** The gathered blocks share the fill's room with the pages, in blocks: room = capacity − mandatory − recent. The picks are cut to room × 16 tokens (lowest priority first). The fill walks Selector V2's cosine order and takes a page when the page and the gather of the picks outside the taken pages still fit: `taken + 1 + ceil(picks left / 16) ≤ room`. A pick inside a taken page costs nothing more; without picks the fill is Selector V2's own. The mandatory and recent pages are unchanged, and their tokens are never picked.

**Contract.** The selection contract gains the key `token_gather` only on such a server: `{picker, markers, quota, pad: newest_unattended_v1, budget: shared_blocks_v1}`. Its digest (`contract_sha256`) covers the pin; every other contract keeps its bytes and digest.

**Receipt and plan.** The signed receipt's body gains `gather: {tokens, pad}` (the gathered tokens ascending, picks and pad; the pad a subset) on a turn that gathers; `ranking` and `selected_count` stay the page selection. The native plan gains `gather: {tokens, blocks}` and its `sparse` drops L − |A_t|·P − G. The adapter takes the gather from the authenticated receipt; the worker checks it (whole blocks, below L, outside the selected pages). The native file therefore holds both.

**Evidence.** Each active turn's `query.selection.gather` in `runtime.jsonl` and in the receipt: `picker`, `picks` (after the quota and the room), `truncated` (picks dropped by the room), `covered` (picks inside taken pages), `gathered`, `pad`, `blocks`, `plain_retrieved` (the plain cosine fill of the same turn), `retrieved`, `displaced` (plain-fill pages not taken), `added` (taken pages beyond the plain fill). The `selected_pages`, `retrieved_pages` and `added/retained/removed_pages` fields are the final ones.

## 4. Restore and phase 1

**Phase 2.** The engine allocates the gathered blocks after the restore targets (one per 16 tokens per full-attention group); like restore targets they join the zero queue and are installed in the row as `[A_t][gathered][run]`. At the first phase-2 step the worker zeroes them with the restore targets and transients, restores the pages, then gathers every slot (`TensorIO.gather_tokens`): a token of a page on the GPU (a resident page, or the tail page phase 1 completed) is read on the device; any other from its DRAM page (an arena page or the phase-1 tail capture), gathered on the host into one buffer and copied once per full-attention tensor; one `index_put_` per tensor writes every slot, and one join ends it. The worker seals the run past the gathered blocks.

**Between turns.** At the commit the session pins the turn's gathered blocks (engine `PrivateBlockOwner.gathers`, worker session mirror) and releases the previous turn's. The next reservation gathers again into fresh blocks (the previous ones are released at its commit). A session therefore holds its last turn's gathered blocks until it closes (none if that turn gathered nothing). On a session pool the close receipt (the `POST /v1/sessions/close` response and its `sessions.jsonl` row) itemizes them as `released.gather_blocks` beside `resident_pages` and `tail` (gather-receipt-1115), so `released.blocks` = `resident_pages` + `tail` + `gather_blocks` (one full-attention group on Qwen3.6-27B); `gather_blocks` is 0 without `--token-gather`, and the other fields and the schema (`remote-ra-session-event-v1`) are unchanged.

**Phase 1.** Phase 1 reads `[R][gathered][tail][fresh]` with `SparsePlan(Q, Q − |R|·P − G)`: it attends exactly what the previous phase 2 attended plus the pages that turn sealed, so the observation vector is computed on the same context as before. The phase-1 lease gains `gathered: {tokens, block_ids}` only then; the gathered blocks are borrowed (never zeroed or freed by the lease).

**Capacity.** The admission gate counts the gathered blocks as fresh phase-2 blocks within the budget: restores + gathered ≤ min(the budget, the sealed pages not resident + the quota's blocks).

## 5. Cost

**Model.** A gathered token moves its K and V in each of the 16 full-attention layers: 16 × 2 × 4 KV heads × 256 × 2 bytes = 64 KiB (a 16-token page is 1 MiB). From DRAM that is one host gather plus one H2D per layer tensor; from a resident page a device index. A page restore moves 1 MiB per page whatever it holds, so gathering k tokens of k different pages moves k × 64 KiB where restoring their pages moves k MiB. The gathered blocks cost HBM only: G/16 blocks of the budget per turn.

**CPU proxy** (one thread, unpinned arena, 4 of 16 layers; a proxy of the code path, not a GPU number): 16 tokens of 16 different DRAM pages, gather 0.53 ms against 1.23 ms to restore the 16 pages; 128 tokens of 128 pages 2.7 ms against 10.9 ms; packed 16 per page the restore is cheaper on the CPU (128 tokens of 8 pages: gather 2.6 ms, restore 0.7 ms), because the CPU indexes element by element where the restore copies runs. The GPU measurement (copy time against restore time, per layer and per turn) is the first item of the GPU plan. At the scale of a per-token fill (section 9, item 1; up to 8,192 tokens per turn), `tools/token_gather_bench.py` measures today's gather, a device-direct read of the pinned arena and the page restore of the same bytes on one GPU without a model (its GPU run: [docs/status.md](../status.md)).

**Offline replay** (section 7): E markers gather 4.5 blocks per turn on average (obs262t, thinking on) and 9.9 (goalvllm, thinking off), 0.9 % and 1.9 % of the 8,192-token budget, about 4.5 and 10 pages displaced from the fill. The page approximation of the same markers (protecting every page that holds one) cost 19–23 % of the budget and displaced 51–98 pages.

## 6. Refusals

- **Gemma 4.** Its global layers' RoPE and K/V layout are not confirmed on the 0.24 source, and its window layout has its own restore path. Later.
- **Other policies and protocols.** The pickers read the Qwen XML transcript's token stream; random, window and rule policies and query plug-ins have their own fills.
- **Other layouts.** P must be 16: a gathered block is one kernel block, and FlashAttention's block table cannot hold a partial block. The gather indexes vLLM 0.24.0 FlashAttention's KV layout `[blocks, 2, 16, heads, head size]` (block axis 0, token axis 2), in either stride order (NHD, or HND, which vLLM uses here because the runtime's OffloadingConnector asks for it), by the block and token axes the worker reads from the loaded backend's `get_kv_cache_shape` at attach, where it also refuses any other layout or stride order; `--triton-attn` is refused. (Until the fix after GPU job 3602601 the check and the CPU fixtures assumed `[2, blocks, 16, heads, head size]`, block axis 1; the fixtures are now built with vLLM's own shape, stride order and reshape code.)
- **`--release-idle-pages`.** A gathered block has no DRAM copy of its own: evicting an idle session's pages would also have to drop its gathered blocks and gather them again from the DRAM pages at readmission, and the evict/readmit records, capacity ledger and readmission restore would change with it. No Qwen SWE-bench run of the policies token gather serves uses the option (their 32-task runs run without it; only `full` ran with it, job 3583081, [known limit 2.4](../known-limits.md#24-full-at-16-sessions-in-25-gib-needs---release-idle-pages)), so the combination is refused at startup (launcher and startup config) and the engine owner refuses an eviction of a session that pins gathered blocks.

## 7. Offline replay of recorded runs

`token_gather.turn` (the runtime's code) on every selective request of two recorded SWE-32 runs at P = 16 and the 8,192-token budget: the receipt's cosine order, capacity and safety partition, the request's committed token stream and the pinned Qwen3.6 tokenizer. Counts and shares only. In every request the replayed plain fill equals the recorded one.

| Run (requests) | Pin | Picks (mean / p90 / max) | Covered by taken pages | Gathered + pad | Blocks (mean / p90 / max) | Share of budget (mean / p90 / max) | Pages displaced (mean / p90) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| obs262t, observation_query, thinking on (1,198) | E markers | 129 / 266 / 453 | 63 | 66 + 6 | 4.5 / 12 / 20 | 0.9 / 2.3 / 3.9 % | 4.6 / 12 |
| | E markers, quota 256 | 122 / 256 / 256 | 62 | 60 + 6 | 4.1 / 10 / 15 | 0.8 / 2.0 / 2.9 % | 4.2 / 10 |
| | C separators 1,024 | 1,023 / 1,024 / 1,024 | 361 | 663 + 2 | 41.6 / 59 / 63 | 8.1 / 11.5 / 12.3 % | 41.7 / 59 |
| goalvllm, goal_query, thinking off (1,677) | E markers | 180 / 341 / 464 | 28 | 152 + 6 | 9.9 / 20 / 29 | 1.9 / 3.9 / 5.7 % | 9.9 / 20 |
| | E markers, quota 256 | 161 / 256 / 256 | 27 | 134 + 6 | 8.7 / 15 / 16 | 1.7 / 2.9 / 3.1 % | 8.8 / 15 |
| | C separators 1,024 | 1,023 / 1,024 / 1,024 | 292 | 731 + 2 | 45.9 / 60 / 63 | 9.0 / 11.7 / 12.3 % | 46.0 / 60 |

No request needed more than its room (`truncated` 0 everywhere). The E picks per request (129 and 180) equal the per-token design note's counts for the same mapping.

## 8. Parity and tests

- `tools/policy_parity.py` between `110632b` and this branch: every outcome byte-identical (contracts, digests, refusals, launches, plans, `test_decouple_e2e` runs) except `serve.py --help` and one usage error, which differ only by the three new options.
- `tests/test_token_gather.py`: the gather copy on CPU tensors (device rows, arena pages across chunks, a captured page; nothing else moves; refusals before any copy), the attention reference, the pickers (incremental index against a full scan), the fill, the pad, the record, the contract key, the launcher and startup-config refusals, the capacity count.
- `tests/test_token_gather_e2e.py`: the decoupled end-to-end harness at (16, 784) for `observation_query` and `goal_query`, the default markers, a parameterised marker set with a quota, and separators: every forward's attended set (phase 2: the selected pages, the gathered tokens, the run; phase 1: R, the previous gathered tokens, the run), the contents of the gathered blocks, the pins on both sides, the budget, the receipt, plan, lease and evidence, a rebuild, a cancel and the close (every block back); and three sessions decoding together with the three decode fast paths of the stable configuration, alone, with `--gdn-device-handoff` and with `--phase1-in-step`, and with `--async-scheduling`; and a session pool (six tasks, three open at a time) whose close receipts itemize the gathered blocks: a session that gathers on its last turn, two that gathered earlier but not on their last turn, three that never gathered, and the same pool without token gather (`gather_blocks` 0). `tests/test_ms_pool.py` checks `released.blocks` = `resident_pages` + `tail` + `gather_blocks` on every close.

## 9. Open questions

1. **Per-token Goal scoring.** Selector V2 has one vector per page (the page's last-token hidden state). The study fills its remaining slots with the tokens of the highest Goal scores; here the remaining budget is filled with whole pages by page cosine. A per-token score needs per-token vectors (16× the page vectors) or another score. The runtime-side design options and their costs (audited 2026-11-15): [PER-TOKEN-OPTIONS.md](PER-TOKEN-OPTIONS.md).
2. **Per-head selection** (the study's earlier note) is out of reach: one block table serves every head of a layer group.
3. **Incremental gather.** Every reservation gathers into fresh blocks; keeping the previous turn's blocks when the gathered tokens repeat would save the copy (most markers persist from turn to turn).
4. **Marker mapping.** `mswea_xml_v1` has no `<tool_call>` or `<tool_response>`; E is mapped onto `</think>`, `<|im_end|>` and the observation's `<|im_start|>`. Whether this mapping, and the separator definition, match the study is the study author's call.
