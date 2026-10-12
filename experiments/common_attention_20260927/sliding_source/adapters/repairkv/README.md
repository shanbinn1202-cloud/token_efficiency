# RepairKV on vLLM

Run with `--method repairkv` through the ordinary `tebench.job` entry point.
The upstream checkout at `baselines/RepairKV` is imported, never patched.
Reference commit: `554b7ce`.

Example from the project root (two-instance integration trial):

```bash
sbatch scripts/job.sbatch --method repairkv --config configs/repairkv_vllm.yaml \
  --set dataset.count=2 --output /path/to/new/repairkv-trial
```

Use `--set cache.kv_budget=12288` or `8192` for other context budgets.

## Algorithm and budget

This port uses the repository's original SnapKV-style base policy and exact-query
RepairKV score/select functions. Like the Phase 10 multi-turn runner, every
base partition is recomputed from the complete committed pool. The evictable
range covers accumulated history rather than the controlled experiment's
fixed initial document:

1. Keep a model-device pool of each token's original, committed KV rows.
   The original partition function moves only evicted rows to pinned CPU memory.
2. After committing the previous turn, call the original
   `build_turn_n_keep_plan` on the complete history pool, including accumulated
   dialogue and previously evicted rows. Use the original
   `materialize_context_partition` to retain at most `B_base` active rows.
   Previously evicted rows can return through this base selection without
   consuming the repair budget. All secondary scores are freshly computed.
   If a GPU allocation fails, release scratch and unused allocator cache,
   retry, then run these same upstream functions on CPU if necessary.
3. Probe the incoming turn against that base cache and capture actual queries
   **after Q normalization and RoPE**, using the model's native projections.
4. Call the original `score_evicted_positions` with its default CPU path,
   then call the original `select_idlekv_positions`. No custom scorer,
   approximate normalization, candidate sampling, or deadline truncation is used.
   Scoring normalizes over all active plus evicted keys, uses max pooling over
   query tokens and means over heads/layers, and packs `(left=2, right=20)`
   bursts. The earlier turn's scores and original positions break ties.
   Every offloaded row uses its freshly computed base-policy score for ties.
5. Call the original runner's `_restore_positions`: select the CPU evicted
   fragment, transfer it to the active cache device, and call `inject_kv`.
   Load that repaired cache into native pages in original position order. Roll back
   the probe's entire recurrent state and logical token history, then compute
   the incoming turn again against the repaired cache before decoding.
   Promoted rows leave the evicted store. GPU allocation failures in base
   partitioning or restoration retry after allocator cleanup, then execute
   the same upstream functions on CPU. The CPU result can be copied to native
   pages one layer at a time. Counters report these recovery events.

`cache.kv_budget = B_base + K`; `K=96` by default. Before replaying each new
turn's signal, repaired **historical** active KV is at most this budget.
New signal tokens and generated tokens grow the active cache during the turn;
they become eligible for compression at the next boundary. The first turn
uses the full prompt. This is a boundary budget, not a per-token hard cap or a
total GPU memory cap. A short initial prompt can now trigger repair after
later dialogue grows beyond the base budget.

Compression runs at the next request boundary, after committing the previous
final sampled token and before probing the incoming signal. Base selection
uses only the completed turn, so this scheduling does not expose the new
signal to the base policy. Repair is synchronous; no idle-time overlap is used.
The upstream SnapKV-style base scores recent **keys** as proxy queries; this
is intentionally preserved. Repair uses actual post-RoPE queries.
Original CPU operations temporarily use the CPU cores allocated to the Slurm
job (capped by process affinity). The previous vLLM thread count is restored
after each call, including exceptions. The scoring implementation is unchanged.

## Engine changes

New code: `vllm/repairkv.py`, `vllm/repairkv_kernels.py`, and this adapter.
Small hooks reuse the existing KV machinery for block allocation,
physical attention lengths and logical RoPE positions.
Standard vLLM handles attention, batching, sampling, chunked prefill,
compilation and CUDA graphs. Each stage uses an ordinary, non-resumable
request. The native KVConnector interface loads the base or repaired cache returned
by upstream functions. At other stage boundaries it reloads committed history;
request pages are freed normally at finish.
The algorithm's pool retains the original committed rows on the model device.

Current support: BF16, greedy generation, TP=1, Qwen3.5/3.6 hybrid models,
MRV1 with native `AsyncScheduler`/FCFS (or explicit synchronous scheduling),
native recurrent cache mode `none`. Only full-attention
layers have token KV to repair. Recurrent layers are retained, with exact
snapshot/rollback around the temporary probe. Prefix caching is disabled
because sessions modify their own KV pages in place.

Scheduling and preemption use native vLLM defaults. There is no RepairKV
output reservation, whole-instance admission gate, or turn-level CPU offload.
On preemption vLLM releases request pages, requeues the request, and recomputes
the prefix. At the current stage's original boundary, RepairKV reinstates the
original selected rows and recurrent state before the current question and
generated suffix are recomputed. This bookkeeping preserves the selected
cache context while leaving preemption decisions to the native scheduler.
Scoring and transfer currently run synchronously between turns, and their
full time is included in the request/batch wall time. No overlap with tool
execution or production throughput improvement is claimed.

The configured native GPU KV pool excludes the original algorithm's complete
on-device history pool, recurrent snapshots, and temporary scoring/query buffers.
For the 27B model at the aligned 8192-token prefill chunk size, the capture
buffer uses 1.5 GiB in addition to the 5 GiB native KV pool. The full history
pool grows with total committed history. CPU-evicted KV storage, original CPU scoring,
host-to-GPU promotion, native cache copies, recomputation, and probes are real method costs.

## Validation

`tests/test_repairkv.py` checks the boundary budget, growing-dialogue eviction,
full-pool reselection, old-row readmission, score refresh, C/W disjointness, committed history
integrity, probe rollback and restoration into recycled pages.

With a running server, run:

```bash
python -B scripts/validate_repairkv.py --config CONFIG --output smoke.json
```

This exercises short, initially over-budget, and later-growing histories,
using the server's unchanged pool/compilation settings.
Synthetic outputs are limited to 32 tokens; this is not a benchmark score.
`scripts/validate_repairkv_scoring.py` compares direct upstream scores and
selections, and induces actual allocator OOMs in its own diagnostic process.
It verifies base-partition and restoration recovery using the original
functions, CPU scoring under GPU pressure, and continued GPU use.

Set `VLLM_REPAIRKV_AUDIT=/path/to/audits` on the server to save Q/K tensors
and selected positions, and to check restored native KV bit for bit. Re-run
the original CPU selector on those real tensors with:

```bash
python -B scripts/audit_repairkv.py --input /path/to/audits --output audit.json
```

Tensor audits are diagnostic and should be disabled in performance runs.
`VLLM_REPAIRKV_VERIFY=1` checks restored KV values without saving tensors.
`scripts/validate_repairkv_preemption.py` uses the native cache-reset endpoint
to force running-request preemption and checks successful continuation.

The September 25 async validation bundle is at
`/u/ssun4/token_efficiency_results/async_swe20_20260925`.
Eight concurrent clients completed three turns each; early EOS and forced native
preemption checks also completed. Forced preemption preserved repair selections;
the generated text was not byte-identical to the uninterrupted control. This
check establishes successful restoration and selection preservation, not exact
text determinism after recomputation. Original CPU scoring and rollback remain
synchronous algorithm dependencies inside the native async request scheduler.
