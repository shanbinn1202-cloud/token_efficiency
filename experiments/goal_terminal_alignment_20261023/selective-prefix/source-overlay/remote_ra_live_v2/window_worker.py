"""Loaded-worker attachment of the window layout (Gemma 4 31B IT, gemma-s1 WP5).

``worker_composition.attach_loaded_worker`` hands a worker whose startup KV layout is the
window layout here; every other worker attaches exactly as before. The steps are its
own, with the window classes:

* tensor views of the global full-attention layers only (``views_from_runner(...,
  skip_sliding=True)``), P-token blocks in pool 0, and the pinned Page arena;
* ``WindowAllocationOwner``, ``WindowWorkerBackend`` and ``WindowLifecycle``
  (window_owner, window_lifecycle) instead of the Mamba owner, the GDN backend and
  the Qwen lifecycle; the phase-1 driver, the bridge, the step guard, the Page
  compactor and the vector space are the same objects;
* the startup checks of the Qwen worker that apply (CUDA graphs with the window
  layout's splitting ops and its global group bound, context limits, text only with
  Gemma's tower names, the source closure, the connector guard).
  --phase1-in-step, --gdn-device-handoff and --decode-fastpath3 (Qwen only) are refused:
  the Gemma profile runs without them in stage 1; --admission-overlap (gemma-goal), an engine-side flag, is checked
  against the pinned config and echoed as for Qwen. The connector's CPU pool is checked
  against its pin (connector_guard.check_worker_pool), as for Qwen;
* --decode-fastpath (gemma-perf) as for Qwen (claim counts, one position read and one
  drain per step, the incremental marker check) plus the window owner's incremental
  observation (window_owner.attach_fastpath); --decode-fastpath2 as for Qwen (one
  allocation_batch RPC per step, the after-forward work behind the logits) plus the
  incremental sliding rows (window_owner.attach_fastpath2, window_rows);
* --release-idle-pages (idle_release) as for Qwen: the startup record checked once
  against the pinned config, then the owner accepts ``evict``/``readmit`` of the
  global Pages (the sliding windows never leave pool 1) and the evidence echoes it.

Startup only; nothing here runs per request or step.
"""
from __future__ import annotations

from .contract import require


def arena_pages(runner, startup, layout):
    """Pinned Page arena capacity: ``dram_arena_pages``, else every window slot's
    session at the context limit, S * (ceil(max_model_len / P) + 1)."""
    pages = startup.get("dram_arena_pages")
    if pages is None:
        length = getattr(runner, "max_model_len", None)
        if length is None:
            length = runner.model_config.max_model_len
        pages = layout.window_slots * (-(-int(length) // layout.page_tokens) + 1)
    require(type(pages) is int and pages > 0, "Page arena capacity required")
    return pages


def check_view_geometry(io, layout, group_pool):
    """Full-attention views only: P-token blocks, groups in pool 0."""
    require(io.views and all(v.kind == "attention" and v.allocation_tokens == layout.page_tokens
                             for v in io.views),
            "window layout requires full-attention views of P-token blocks only")
    require(group_pool is not None and
            all(0 <= v.group < len(group_pool) and group_pool[v.group] == 0 for v in io.views),
            "window layout requires the full-attention groups in pool 0")


def attach_window_worker(worker, startup, layout, inputs):
    """``attach_loaded_worker`` for the window layout; returns the same evidence keys."""
    from .kv_layout import is_window
    require(is_window(layout), "window layout required")
    runner = worker.model_runner
    from remote_ra_growing_worker.tensors import PageArena, TensorIO, views_from_runner
    io = TensorIO(views_from_runner(runner, *inputs, skip_sliding=True), decoupled=True)
    check_view_geometry(io, layout, getattr(runner.kv_cache_config, "group_pool", None))
    from .cuda_graphs import WINDOW_SPLITTING_OPS, attach as attach_cuda_graphs
    graphs = attach_cuda_graphs(runner, startup.get("cuda_graphs"),
                                sorted({v.group for v in io.views if v.kind == "attention"}),
                                WINDOW_SPLITTING_OPS)
    from .context_limits import RECORD_KEY, check_worker
    context = check_worker(runner, startup.get(RECORD_KEY))
    from .admission_flags import OVERLAP, worker_flags
    admission = worker_flags(startup)
    # --admission-overlap K (gemma-goal) is engine-side only; the other admission flags are refused.
    require(admission is None or set(admission) == {OVERLAP},
            "the window layout runs with --admission-overlap alone of the admission flags")
    from .decode_fastpath import CONFIG_KEY as FASTPATH_KEY, worker_flag as fastpath_flag
    fastpath = fastpath_flag(startup)
    from .decode_fastpath2 import CONFIG_KEY as FASTPATH2_KEY, worker_flag as fastpath2_flag
    fastpath2 = fastpath2_flag(startup)
    require(not fastpath2 or fastpath, "decode_fastpath2 needs decode_fastpath")
    from .decode_fastpath3 import worker_flag as fastpath3_flag
    require(not fastpath3_flag(startup), "the window layout runs without --decode-fastpath3")
    from .idle_release import CONFIG_KEY as RELEASE_KEY, worker_flag as release_flag
    release = release_flag(startup)
    from .language_model_only import CONFIG_KEY as TEXT_ONLY_KEY, worker_check as text_only_check
    text_only = text_only_check(runner, startup)
    io.attach_arena(PageArena(io.views, max_pages=arena_pages(runner, startup, layout)))
    from .loaded_sources import verify_closure
    source_evidence = verify_closure(runner)
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from .connector_guard import quiescent
    connector = get_kv_transfer_group()
    io.connector_check = lambda: quiescent(connector)
    from .connector_guard import check_worker_pool
    check_worker_pool(connector)
    io.drain()
    from .vector_space import attach as attach_vector_space
    vector_space = attach_vector_space(runner)
    startup = dict(startup, source_closure_verified=True)
    from .window_owner import WindowAllocationOwner
    from .window_lifecycle import WindowLifecycle, WindowWorkerBackend
    owner = WindowAllocationOwner(runner, io, startup, layout)
    if release:
        owner.release_idle = True       # evict/readmit of global Pages accepted
    runner._remote_ra_live_v2_allocation_owner = owner
    backend = WindowWorkerBackend(io, owner, source=startup["source"])
    owner.bind_backend(backend)
    from .page_compaction import PageCompactor
    backend.compactor = PageCompactor(vector_space["hidden_size"], layout.page_tokens)
    from .native_probe import NativeQueryForward
    runner._remote_ra_native_query_forward = NativeQueryForward(runner, owner, backend)
    qwen = WindowLifecycle(backend, page_tokens=startup["page_tokens"])
    from .worker_composition import WorkerBridge
    bridge = WorkerBridge(qwen, owner)
    owner.bind_bridge(bridge)
    runner._remote_ra_live_v2_bridge = bridge
    from .async_support import attach as attach_async
    attach_async(runner)
    if fastpath2:
        # Before the step profile wraps the bridge (its deferred after-forward names).
        from .decode_fastpath2 import attach_worker as attach_fastpath2
        attach_fastpath2(owner, bridge, qwen, io)
    from .generation_guard import guarded_execute
    import remote_ra_step_profile as step_profile
    if step_profile.ENABLED:
        guarded_execute = step_profile.install_worker(runner, bridge, guarded_execute)
    runner.execute_model = guarded_execute(runner, runner.execute_model)
    if admission is not None:
        # Records the pinned flags (no hand-off, no in-step phase 1), as for Qwen.
        from .gdn_handoff import attach_worker
        attach_worker(admission, io=io, owner=owner, backend=backend, qwen=qwen)
    if fastpath:
        from .decode_fastpath import attach_worker as attach_fastpath
        attach_fastpath(owner, bridge, qwen)
        from .window_owner import attach_fastpath as attach_window_fastpath
        attach_window_fastpath(owner, qwen)
    if fastpath2:
        from .window_owner import attach_fastpath2 as attach_window_fastpath2
        attach_window_fastpath2(owner)
    worker._remote_ra_live_v2_backend = backend
    worker._remote_ra_live_v2_qwen = qwen
    from .kv_layout import to_additional_config
    evidence = {"evidence_kind": backend.io.evidence_kind,
                "loaded_sources_verified": startup["loaded_sources_verified"],
                "copies_drained": True, "source": startup["source"],
                "source_closure": source_evidence, "vector_space": vector_space,
                "tensor_geometry": [dict(layer=v.name, group=v.group, kind=v.kind,
                    allocation_tokens=v.allocation_tokens, kernel_tokens=v.kernel_tokens, ratio=v.ratio,
                    page_bytes=v.page_bytes, page_stride=v.page_stride, cohort=list(v.cohort),
                    shapes=[list(t.shape) for t in v.tensors], strides=[list(t.stride()) for t in v.tensors])
                    for v in io.views],
                "kv_layout": to_additional_config(layout),
                "page_vectors": backend.compactor.evidence(), "dram_arena": io.arena.stats()}
    if graphs is not None:
        evidence.update(cuda_graphs=graphs)
    if fastpath:
        evidence[FASTPATH_KEY] = True    # additive, on only
    if fastpath2:
        evidence[FASTPATH2_KEY] = True
    if context is not None:
        evidence[RECORD_KEY] = context
    if text_only is not None:
        evidence[TEXT_ONLY_KEY] = text_only
    if release:
        evidence[RELEASE_KEY] = True    # additive, on only
    if admission is not None:
        evidence.update(admission=dict(admission))
    return evidence
