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
  --phase1-in-step and --gdn-device-handoff are refused:
  the Gemma profile runs without them in stage 1; --admission-overlap (gemma-goal), an engine-side flag, is checked
  against the pinned config and echoed as for Qwen, and so is --goal-overlap (the owner then accepts the separate
  Goal forward, goal_forward). The connector's CPU pool is checked
  against its pin (connector_guard.check_worker_pool), as for Qwen;
* --decode-fastpath (gemma-perf) as for Qwen (claim counts, one position read and one
  drain per step, the incremental marker check) plus the window owner's incremental
  observation (window_owner.attach_fastpath); --decode-fastpath2 as for Qwen (one
  allocation_batch RPC per step, the after-forward work behind the logits) plus the
  incremental sliding rows (window_owner.attach_fastpath2, window_rows); --decode-fastpath3
  (gemma-fastpath3) as for Qwen (rowless allocation entries, the byte compare of the global
  device row, the steady materialize and IO scope) plus the window owner's steady observation
  and the window lifecycle's steady and moved leases (window_owner.attach_fastpath3);
* --release-idle-pages (idle_release) as for Qwen: the startup record checked once
  against the pinned config, then the owner accepts ``evict``/``readmit`` of the
  global Pages (the sliding windows never leave pool 1) and the evidence echoes it;
* --async-scheduling (async-gemma, docs/async-sched/GEMMA.md) as for Qwen: the startup record
  checked once against the pinned config (it needs --decode-fastpath2), the guard's issued step
  and the resolve's void (async_sched.attach_worker; on this layout the void writes the global
  and every sliding-window group's slot mapping, there is no GDN state), the evidence echo;
* --thinking-history strip (gemma-thinking, docs/thinking-gemma/DESIGN.md section 4.4) as for Qwen:
  the startup record checked once against the pinned config (thinking_cut.worker_flag), then the
  window owner, lifecycle and backend commit at the cut with the window held at X
  (thinking_cut.attach_worker -> window_hold.attach_worker) and the evidence echoes it. Refused with
  --async-scheduling (v1) and without hold slots in the layout.

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
    from .phase1_graphs import CONFIG_KEY as PHASE1_GRAPHS_KEY
    require(startup.get(PHASE1_GRAPHS_KEY) is None,
            "--phase1-graph-tokens is not available for the window layout (Gemma 4, v1)")
    graphs = attach_cuda_graphs(runner, startup.get("cuda_graphs"),
                                sorted({v.group for v in io.views if v.kind == "attention"}),
                                WINDOW_SPLITTING_OPS)
    from .context_limits import RECORD_KEY, check_worker
    context = check_worker(runner, startup.get(RECORD_KEY))
    from .admission_flags import OVERLAP, worker_flags
    GOAL_OVERLAP = "goal_overlap"     # goal_overlap.FLAG, one of admission_flags.FLAGS
    admission = worker_flags(startup)
    # --admission-overlap K (gemma-goal) is engine-side only; --goal-overlap (gemma-overlap)
    # makes the owner accept goal_forward (gdn_handoff.attach_worker); the other admission
    # flags are refused.
    require(admission is None or set(admission) <= {OVERLAP, GOAL_OVERLAP},
            "the window layout runs with --admission-overlap alone of the admission flags, besides --goal-overlap")
    from .decode_fastpath import CONFIG_KEY as FASTPATH_KEY, worker_flag as fastpath_flag
    fastpath = fastpath_flag(startup)
    from .decode_fastpath2 import CONFIG_KEY as FASTPATH2_KEY, worker_flag as fastpath2_flag
    fastpath2 = fastpath2_flag(startup)
    require(not fastpath2 or fastpath, "decode_fastpath2 needs decode_fastpath")
    from .decode_fastpath3 import CONFIG_KEY as FASTPATH3_KEY, worker_flag as fastpath3_flag
    fastpath3 = fastpath3_flag(startup)
    require(not fastpath3 or fastpath2, "decode_fastpath3 needs decode_fastpath2")
    from .idle_release import CONFIG_KEY as RELEASE_KEY, worker_flag as release_flag
    release = release_flag(startup)
    # --async-scheduling (async-gemma): the startup record, checked once against the pinned config.
    from .async_sched import WORKER_KEY as ASYNC_KEY, worker_flag as async_flag
    asynchronous = async_flag(startup)
    require(not asynchronous or fastpath2, "async_scheduling needs decode_fastpath2")
    # --thinking-history strip (gemma-thinking): the startup record, checked once against the pinned
    # config; it holds the window at the cut in the layout's hold slots, and v1 refuses async with it.
    from .thinking_cut import STRIP, WORKER_KEY as STRIP_KEY, worker_flag as strip_flag
    strip = strip_flag(startup)
    require(not strip or layout.hold_slots >= 1,
            "thinking_history strip on the window layout needs hold slots (kv_layout.hold_slots >= 1)")
    require(not strip or not asynchronous,
            "thinking_history strip on the window layout refuses --async-scheduling (v1)")
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
    if fastpath2:
        # Before the step profile wraps the bridge (its deferred after-forward names).
        from .decode_fastpath2 import attach_worker as attach_fastpath2
        attach_fastpath2(owner, bridge, qwen, io)
    if fastpath3:
        # After the second fast path, before the step profile wraps the bridge (as for Qwen).
        from .decode_fastpath3 import attach_worker as attach_fastpath3
        attach_fastpath3(owner, bridge, qwen)
    if asynchronous:
        # --async-scheduling: after the fast paths (it defers fast path 2's finish_step), as for Qwen.
        from .async_sched import attach_worker as attach_async
        attach_async(owner, bridge, qwen, runner)
    from .generation_guard import guarded_execute
    import remote_ra_step_profile as step_profile
    if step_profile.ENABLED:
        guarded_execute = step_profile.install_worker(runner, bridge, guarded_execute)
    runner.execute_model = guarded_execute(runner, runner.execute_model)
    if admission is not None:
        # Records the pinned flags (no hand-off, no in-step phase 1; goal_forward with --goal-overlap), as for Qwen.
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
    if fastpath3:
        from .window_owner import attach_fastpath3 as attach_window_fastpath3
        attach_window_fastpath3(owner, qwen, backend)
    if strip:
        # --thinking-history strip: completions commit the session at the cut X = e with the window
        # held at X (window_hold.attach_worker), after the fast paths as for Qwen.
        from .thinking_cut import attach_worker as attach_strip
        attach_strip(owner, qwen, backend)
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
    if fastpath3:
        evidence[FASTPATH3_KEY] = True
    if context is not None:
        evidence[RECORD_KEY] = context
    if text_only is not None:
        evidence[TEXT_ONLY_KEY] = text_only
    if release:
        evidence[RELEASE_KEY] = True    # additive, on only
    if admission is not None:
        evidence.update(admission=dict(admission))
    if asynchronous:
        evidence[ASYNC_KEY] = True      # additive, on only: compose_engine compares the echo
    if strip:
        evidence[STRIP_KEY] = STRIP     # additive, on only: compose_engine compares the echo
    return evidence
