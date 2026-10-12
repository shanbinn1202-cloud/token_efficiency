# Project-local vLLM

## Current routing (2026-09-14)

Vanilla and Sliding Window now use ordinary vLLM completions with the same prefix-cache and scheduling configuration. Sliding Window selects a fixed prefix plus recent input tokens before submission; it does not use persistent window sessions or idle-session reclamation. The persistent-window implementation described below is retained as legacy code, not the active Sliding Window benchmark path.

AgentKV uses its persistent-session adapter with upstream default prefill-first admission, full remaining output reservations and retained parked KV. It has no fixed 64K lifetime session-capacity gate, client semaphore or idle-session reclamation. See `../../adapters/agentkv/VLLM.md` and `../../results/agent_alignment_20260914/对齐清单.md`.

## Local extension history

This tree starts from the installed **vLLM 0.28.0 ARM64 wheel**, including its
existing CUDA binaries. `UPSTREAM.json` records the original Python file hashes.
The system installation and copied baseline repositories are not modified.

`python -m tebench.job --method sliding_window --backend vllm` automatically
launches this local package. AgentKV also uses this package via `--method agentkv`; other vLLM methods use
it with both stateful policies disabled. No editable install or global package replacement is
needed. The client checks `/v1/kv-window/info` to verify the actual engine path.

## KV window semantics

- One native resumable streaming request per benchmark instance, closed at task
  completion. Worker threads have separate client sessions.
- Fixed system/task prefix plus the newest token suffix, exactly filling
  `cache.kv_budget` when enough history exists. Cuts may occur inside turns.
- Budget applies to per-turn input; generation may temporarily exceed it.
  Oversized incoming feedback is truncated from its beginning, preserving the
  generation header. The fixed prefix is never truncated. Omitted fresh tokens
  are not processed; positions of already computed retained KV never change.
- Gather selected full-attention KV into the retained pages, then release surplus
  pages. Boundaries need not be page aligned. Retained KV is not re-prefilled.
- Preserve original logical RoPE positions. Full-attention slot mapping and
  sequence lengths use compact physical coordinates. GatedDeltaNet keeps its
  latest recurrent state and logical counters.
- Reuse raw generated tokens, including thinking/tool-call syntax, across turns.
  Re-render only new delimiters, feedback and the next assistant header.

## Scope and execution settings

The extension targets this project's Qwen3.6-27B, TP=1, MRV1, FA3 packed BF16 KV,
without speculative decoding. CUDA Graph and torch.compile remain enabled by
the common configuration. Per-request physical sequence buffers have stable GPU
addresses during graph replay. Compaction runs outside the captured forward.

For this method, ordinary prefix caching and asynchronous scheduling are
explicitly disabled and recorded in `run.json`'s policy. Prefix hashes cannot
describe pruned KV, and compaction must run before freed pages are zeroed/reused.
These are material execution differences from the existing vanilla run.

Preemption/recomputation of a stateful window request is unsupported and fails
explicitly. The logical RoPE context still obeys `max_window_size`; an 8K physical
window does not make logical positions unlimited. The 8K budget is a token count
for each full-attention layer, not a bound on the entire GPU allocation: page
padding, recurrent state, and the shared vLLM pool remain allocated separately.

Shared infrastructure is in `kv_session.py` (physical coordinates, copy and page
ownership) and `kv_session_server.py` (streaming transport). `kv_window.py` is
the token-suffix policy; `agentkv.py` invokes the original phase/ring/selection
code and collects post-RoPE Q through a graph-compatible operator. See
[AgentKV algorithm contract](../../adapters/agentkv/VLLM.md). AgentKV compacts
at per-response decode intervals; its budget is a retention target, not the
sliding policy's per-round input cap.

Original sliding upstream edits: `v1/core/single_type_kv_cache_manager.py`,
`v1/core/sched/scheduler.py`, `v1/core/sched/output.py`, and
`v1/worker/gpu_model_runner.py`. Most new logic lives in `kv_window.py` and
`kv_window_server.py`.

CPU invariants: `python -m pytest tests/test_kv_window_engine.py` from the project
root. Live validation: start `python -m tebench.serve --method sliding_window`,
then run `python scripts/validate_kv_window.py --output results/kv_validation.json`
on the server node. GPU validation results, when completed, are saved separately;
CPU checks alone do not establish end-to-end correctness.

AgentKV additionally integrates `model_executor/models/qwen3_next.py` (Q capture),
`v1/outputs.py` (worker-to-scheduler compaction acknowledgement) and
`v1/worker/gpu_worker.py` (session statistics). `PATCHES.json` and `PATCHES.diff`
record the complete delta from the original wheel.
