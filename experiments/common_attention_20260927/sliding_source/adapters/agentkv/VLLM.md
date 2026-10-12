# AgentKV on project-local vLLM

Reference: https://github.com/LiuTaowen-Tony/agentkv/tree/b5280b7c42bd90962712c19f8e0047c89ca40d6d
(the remote main revision checked for this implementation). The copied upstream
checkout is read-only; executable policy code is imported directly from it.

Run using the existing shared configuration:

```bash
source scripts/env.sh
python -B -m tebench.job --method agentkv --backend vllm \
  --set cache.kv_budget=16384 --set generation.max_tokens=4096 --output results/agentkv_vllm_swe20
```

The job launches `engines/vllm`, not the environment's stock vLLM. Common defaults
remain thinking=true, workers=8, max_tokens=4096, max_turns=120,
max_window_size=131072, temperature=0, CUDA Graph=true, torch.compile=true.

| Concern | Original algorithm / port |
|---|---|
| Policy | Original `create_eviction_plan("agentkv-B")`, `StageQSnapKVEvictionPlan.select_layer` |
| Observation Q | Post-QK-normalization/post-RoPE queries, as in upstream attention.py |
| Phase assignment | Original prompt offset classifier and generated-token rolling-tail classifier; legacy mapping |
| Query memory | Original QueryRingBufferPool metadata and read order; 32 Q entries split into 4 rings of 8 |
| Importance | Original SnapKV BF16 attention scores, GQA mean, pooling kernel size 1 |
| Mandatory positions | First 16 physical KV entries and latest 8; no additional fixed task-prefix protection |
| Selection | Per-layer/per-KV-head top-k; sorted selected indices; no head/layer averaging of selections |
| Trigger | Original `_should_compact`: generated_len > 1 and a multiple of 128, reset per response |
| Prefill / park | No unconditional after-prefill or before-park compression |
| Budget | Compacted KV target, not a hard per-request input cap; prefill and inter-eviction decode may exceed it |
| Storage bridge | vLLM FA3 packed pages; exact per-head token gather/copy; page padding does not alter selections |
| Positions | Logical RoPE positions unchanged; physical addresses and attention lengths shrink |
| Admission | Native vLLM FCFS admission; no whole-task limit or extra output reservation |
| Scheduling | The same native AsyncScheduler as Vanilla; mixed prefill/decode and native RECOMPUTE are allowed |
| Context boundary | Clamp requested output to remaining logical context; reject an already full input |
| Sessions | One native request per turn; exact KV, recurrent state and Q-ring checkpoints survive on CPU between turns |
| Cleanup | Native GPU request pages and ring slots are released at turn completion; session CPU checkpoints are released at close |

The independent CPU score reference and live GPU checks are in
`tests/test_agentkv_vllm.py` and `scripts/validate_agentkv_vllm.py`.

Session continuation preserves a length-limited final raw token even when it is
an incomplete UTF-8 byte. Only newly appended template text is encoded for
prompt phase offsets. The restored token keeps the upstream rolling generation
phase, and parked text decodes the complete generated span (dropping only EOS),
matching upstream `_session_full_text`. Retokenizing an isolated final byte can
otherwise reject a valid continuation or change its phase.

For diagnostics only, `VLLM_AGENTKV_MODEL_AUDIT=1` together with
`VLLM_AGENTKV_MODEL_AUDIT_PATH=/absolute/path/audit.json` enables
`adapters/agentkv/vllm_audit.py`. It captures actual model Q/K/V and attention
outputs, maintains independent dense KV and stage-query histories, and checks
page contents, query rings, selected indices and FP32 reference attention after
real forwards, including graph replay and compaction. It uses a separate compile
cache key. This expensive oracle must be disabled for timed benchmarks; a partial
audit report alone is not evidence that the client workload completed.

Shared modules: `vllm/kv_session.py` owns physical coordinate translation, exact
KV gathers and page-release helpers; `vllm/kv_session_server.py` owns native
turn transport; `tebench/backends/kv_session.py` supplies `KVSessionBackend`
for both policies. `kv_window.py` retains only sliding's selection policy.
AgentKV-specific model capture / original-policy invocation live in `agentkv.py`.
The Q-ring copy operator has fixed buffers and masked writes for CUDA Graph
replay. Original ring metadata is prepared on CPU; Q storage lives on GPU.
Selection and compaction execute outside the graph after sampling, before any
subsequent forward. Scheduler page release follows worker copy completion.

Scope: Qwen3.6/Qwen3.5 hybrid full attention, TP=1, MRV1, BF16/FA3,
without speculative decoding. All six benchmark methods use native async FCFS
scheduling with the same eight-request limit and 5 GiB native KV pool. AgentKV
uses a native KVConnector to import its exact turn checkpoint. Prefix caching
is disabled for in-place compression. The original CPU phase classifier needs
real sampled IDs, so its token-copy dependency synchronizes the worker; the
engine's async batch queue remains enabled. Only physical compaction boundaries
drain the queue before page addresses change. Other decode batches overlap.

At EOS, checkpoint the last committed state before any native async lookahead.
Lookahead output is discarded by native request completion and never replaces
that checkpoint. Idle checkpoints live on CPU so waiting for a tool does not
hold GPU blocks or scheduler slots. Copy/restore preserves tensor values, phases,
ring order, logical positions, and original per-response eviction timing.

This is algorithm-level alignment on the same Q/K inputs, not a promise of
bit-identical model execution across inference engines. Different prefill chunks,
attention/GEMM kernels and hybrid-state execution may change floating-point
results and generated trajectories. Copy/gather kernels and the allocator are
necessarily vLLM-specific; the original Mini-SGL page-size-one allocator and
session objects do not own vLLM's pages. The storage bridge does not re-prefill
retained KV, quantize it, or substitute a token-window method.


Recomputation replays full token history in segments ending at historical
compaction boundaries. Each boundary retains CPU int32 per-layer, per-head
selected indices. The worker applies those decisions before the next segment;
original phase IDs rebuild the Q ring and native execution rebuilds recurrent
state. Replayed prefills do not count as newly generated tokens. Native
preemption uses the same queue and victim policy as Vanilla.

The current async implementation is being validated in
`/u/ssun4/token_efficiency_results/async_swe20_20260925`. CPU tests cover common
scheduler selection, native preemption, overlapping ordinary batches, commit
fences, exact KV/ring checkpoint restoration, raw-byte continuation and original
scores. Full-model tests additionally check eight concurrent multi-turn clients,
original selector output on captured Q/K, recurrent restoration and forced reset.
A partial oracle report alone does not establish completed validation.

Historical evidence in `results/agent_alignment_20260914/` applies to the former
synchronous prefill-first adapter. Those timings are not results for the new
common async scheduler.
