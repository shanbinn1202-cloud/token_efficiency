# IntentKV on the local vLLM engine

The original repository at `baselines/IntentKV` is unchanged (commit
`065f394dbb14f317d2b4dba252e8e66b91afa8c7`). The external adapter imports its
`IntentKVCluster.update_kv` directly. It does not copy or replace the scores.

## Preserved algorithm

- Defaults match the original efficiency and RULER evaluation: last 64 queries,
  16-token blocks, configured prompt capacity of 16384, 12288, or 8192.
- Original BF16 attention, FP32 softmax followed by the original dtype cast,
  three-row cumulative attention, JSD, second differences, query starting point,
  block sums, top-k, sort, and remainder handling all execute in the original file.
- K/V are repeated conceptually to query heads before selection, as upstream.
  Every query head selects independently. No averaging across a GQA group.
- The original patched `generate` creates a fresh cache for every request.
  Initial prefill attention uses the full prompt. Compression runs after that
  prefill, before the second generated token; decode KV is then appended without
  repeated eviction. Each new agent turn renders the full history as Vanilla does.
- Qwen3.6 projections, Q/K normalization, RoPE, attention gating, recurrent layers,
  sampling, and logical positions stay in the native model implementation.

`upstream.py` provides only the last query window to the original selector through
a read-only shape/slice interface: upstream never accesses earlier Q rows.
`TorchFunctionMode` observes the original gather indices without changing them.
V is irrelevant to scoring; the original returned indices subsequently gather the
actual native K **and** V. Tests compare both with a full original Q/K/V call.

## Storage and scheduling

The Qwen model has 24 query heads and 4 KV heads. Instead of storing six identical
copies of a retained GQA token, native KV-head storage holds the union of those
six heads' selected blocks. Small per-query-head page tables recover each original
ordered selection. Layers/heads with smaller unions have unused padding, which is
never attended. Decode is appended after the shared tail.

Consequently **the budget is the original per-query-head prompt capacity**.
Physical native token slots can exceed that capacity because of the unions and
padding; they never exceed the original uncompressed prompt allocation. Event
logs distinguish `selected_tokens` from `stored_tokens`. All persistent K/V still
occupies the same fixed 5 GiB native pool. Selection scratch and index metadata
are separately allocated, as ordinary attention/selection workspaces are.

The adapter uses native HND cache layout and FlashAttention 3 with virtual
one-head sequences and 16-token page views. Prefix caching is disabled because
the final prompt queries determine a new selection for each request. CUDA Graphs,
torch compilation, chunked prefill, eight clients, and native request scheduling
remain enabled. There is no whole-task admission reservation.

Native RECOMPUTE preemption releases the original native pages. On replay, the
scheduler stops at the initial prompt boundary, reapplies the saved discrete
selection, and replays subsequent tokens against the same head-specific history.
Only integer decisions are retained on CPU; K/V and recurrent state are recomputed.
The HF cache's allocator hints (`empty_cache`) are replaced by native page ownership;
they are not part of selection or attention arithmetic.

## Validation and experiment status

Artifacts: `/u/ssun4/token_efficiency_results/intentkv_swe20_20260921`.

- CPU checks compare original K/V exactly, including boundary lengths and
  remainders, independent heads, cross-layer padding, decode appends, and replay.
- Real GH200 tests passed all three capacities, mixed query lengths, native
  784-token blocks viewed as 16-token pages, and CUDA Graph capture/replay.
  K/V selection matched exactly. Maximum attention error against an FP32
  reference was 0.000244140625 in each case.
- The previous synchronous full-model test subsequently passed eight clients
  and native preemption. Its 16K run completed 20 tasks (13 resolved); one
  context-limit failure stopped the old launcher before 12K/8K.
- New common-async validation and reruns are tracked separately in
  `/u/ssun4/token_efficiency_results/async_swe20_20260925`.
  The initial-selection/replay boundary now commits before subsequent batches;
  ordinary decode continues through the native async batch queue.


`scripts/run_intentkv_experiment.py --bundle <path> --stage all` first requires the
kernel result, starts the aligned full model, checks eight clients and forced
native preemption, then runs 16K, 12K and 8K with official grading. It stops on
failed validation and verifies frozen source hashes and the exact Vanilla task IDs.
