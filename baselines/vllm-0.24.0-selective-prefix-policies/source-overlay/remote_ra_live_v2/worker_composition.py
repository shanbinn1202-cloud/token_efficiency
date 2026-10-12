"""Loaded-worker composition: runner callbacks, CUDA attachment and dispatch.

Split from ``composition.py`` so the worker half has one owner.  The engine
half (proxy, endpoint, engine bridge, ``compose_*``) stays there and
re-exports :func:`attach_loaded_worker` and :func:`worker_dispatch`, so the
overlay ``gpu_worker.py`` hooks keep importing ``composition``.
"""
from __future__ import annotations

from .contract import require
from .qwen import QwenAlignLifecycle
from .worker_backend import CudaWorkerBackend
from .allocation_owner import RunnerAllocationOwner


class WorkerBridge:
    """Runner callback owner: allocation step, Qwen forward and terminal."""

    def __init__(self, qwen=None, owner=None):
        self.qwen = qwen
        self.owner = owner
        self.assignments = {}

    def bind_qwen(self, qwen):
        require(self.qwen is None and type(qwen) is QwenAlignLifecycle,
                "Qwen lifecycle attachment collision")
        self.qwen = qwen

    def register(self, assignment, tokens, allocation_generation):
        require(self.qwen is not None, "loaded worker Qwen lifecycle absent")
        state = self.qwen.open(assignment, tuple(tokens), allocation_generation)
        self.assignments[assignment.identity.request] = state
        return state

    def before_forward(self, runner, scheduler_output, positions):
        """Every owned request of the step, each on its own token slice.

        The owner claims the whole step first (two passes); then each request
        zeroes/restores, proves its positions [offset, offset + count) and
        opens its Qwen lease. That request's H2D is attributed to it alone.
        One request returns its lease, several the tuple in batch order.
        """
        if self.owner is not None:
            from remote_ra_growing_worker.tensors import attributed
            from .positions import text_positions
            steps = self.owner.step(runner, scheduler_output)
            leases = []
            for s in steps:
                state = self.assignments.get(s.request)
                require(state is not None, "worker assignment absent")
                with attributed(self.owner.io, s.request):
                    self.owner.materialize(s)
                    logical, evidence = text_positions(runner, positions, s.start, s.count,
                                                       s.request, s.offset)
                    leases.append(self.qwen.before_forward(state.assignment.identity,
                        start=s.start, count=s.count,
                        logical_positions=logical, position_evidence=evidence,
                        attention_allocations=s.attention, hybrid_allocations=s.hybrid,
                        allocation_generation=s.generation))
            return leases[0] if len(leases) == 1 else tuple(leases)
        callback = getattr(scheduler_output, "remote_ra_live_v2_before", None)
        if callback is not None:
            return callback(self.qwen, positions)

    def after_forward(self, runner, scheduler_output, hidden_states):
        """Hidden rows [offset, offset + count) to each request's lease, batch order.

        The finiteness of every scheduled row is read with one device join for
        the step; each request still checks its own rows (formerly one join per
        request, plus a full host copy of its rows).
        """
        if self.owner is not None:
            import torch
            rows = self.owner.scheduled_rows(runner, scheduler_output)
            finite = None
            if isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 2:
                total = rows[-1][2] + rows[-1][3]
                finite = torch.isfinite(hidden_states[:total]).all(dim=1).cpu()
            receipts = []
            for request, _, offset, count in rows:
                state = self.assignments[request]
                # Pages are sealed and published at terminal complete (spec 3.8).
                receipts.append(self.qwen.after_forward(state.assignment.identity,
                    hidden_states[offset:offset + count],
                    finite=None if finite is None else finite[offset:offset + count]))
            return receipts[0] if len(receipts) == 1 else tuple(receipts)
        callback = getattr(scheduler_output, "remote_ra_live_v2_after", None)
        if callback is not None:
            return callback(self.qwen, hidden_states)

    def terminal(self, request_id, operation):
        state = self.assignments.pop(request_id, None)
        if state is not None:
            return self.qwen.finish(state.assignment.identity, operation)


def worker_layout(runner, startup):
    """The server KV layout (kv_layout I0), once at attach.

    vLLM's ``additional_config`` is the source (WP1 reads the same key); the
    startup record's ``kv_layout`` (absent or None in legacy; the whole
    additional_config dict or its value) must state the same layout and its
    Page size. Decoupled also needs vLLM's per-group pool map; legacy has none.
    """
    from types import SimpleNamespace
    from .kv_layout import LAYOUT_KEY, from_vllm_config
    declared = startup.get("kv_layout")
    if isinstance(declared, dict) and LAYOUT_KEY not in declared:
        declared = {LAYOUT_KEY: declared}
    stated = from_vllm_config(SimpleNamespace(additional_config=declared))
    config = getattr(runner, "vllm_config", None)
    layout = from_vllm_config(config) if config is not None else stated
    require(layout == stated and layout.page_tokens == startup["page_tokens"],
            "startup KV layout differs from the loaded vLLM layout")
    group_pool = getattr(getattr(runner, "kv_cache_config", None), "group_pool", None)
    require((group_pool is not None) == layout.decoupled,
            "decoupled KV layout requires the two-pool KV cache config")
    return layout


def check_view_geometry(io, layout, group_pool=None):
    """FA views are P-token blocks (one block per Page); Mamba views M-token blocks.

    Decoupled: every FA view's group allocates from pool 0 and every Mamba
    view's group from pool 1 (``kv_cache_config.group_pool``; groups are
    looked up by kind, never by position).
    """
    require(len({v.allocation_tokens for v in io.views if v.kind=='hybrid'})==1,
            'native scalar Mamba state index requires common Mamba block width')
    if not layout.decoupled:
        # Stage B w = 1: one FA block per Page, Mamba align block == Page (spec 0).
        require(all(v.allocation_tokens==layout.page_tokens for v in io.views),
                'Stage B requires allocation_tokens == page_tokens for every view')
        return
    require(all(v.allocation_tokens == (layout.page_tokens if v.kind == 'attention'
                                        else layout.mamba_tokens) for v in io.views),
            'decoupled layout requires FA allocation_tokens == P and Mamba == M')
    require(group_pool is not None and
            all(0 <= v.group < len(group_pool) and
                group_pool[v.group] == (0 if v.kind == 'attention' else 1) for v in io.views),
            'decoupled layout requires FA groups in pool 0 and Mamba groups in pool 1')


def arena_pages(runner, startup, layout):
    """Pinned Page arena capacity: the startup's ``dram_arena_pages``, else every
    session's whole context, S * ceil(max_model_len / P) + S (S from K = 2S + 1)."""
    pages = startup.get("dram_arena_pages")
    if pages is None:
        length = getattr(runner, "max_model_len", None)
        if length is None:
            length = runner.model_config.max_model_len
        sessions = (layout.mamba_slots - 1) // 2
        pages = sessions * -(-int(length) // layout.page_tokens) + sessions
    require(type(pages) is int and pages > 0, "Page arena capacity required")
    return pages


def attach_loaded_worker(worker, startup):
    """Construct the physical backend from the loaded worker after KV init."""
    require(startup is not None and startup.get("enabled") is True,
            "trusted worker startup record required")
    runner = worker.model_runner
    require(not hasattr(runner, "_remote_ra_live_v2_bridge"),
            "loaded worker already attached")
    from remote_ra_growing_worker.tensors import TensorIO, views_from_runner
    inputs = getattr(runner, "_remote_ra_growing_view_inputs", None)
    if inputs is None:
        inputs = (runner.kv_caches, runner._kernel_block_sizes)
        runner._remote_ra_growing_view_inputs = inputs
    require(type(inputs) is tuple and len(inputs) == 2,
            "physical runner views unavailable")
    layout = worker_layout(runner, startup)
    io = TensorIO(views_from_runner(runner, *inputs), decoupled=layout.decoupled)
    check_view_geometry(io, layout, getattr(runner.kv_cache_config, "group_pool", None))
    if layout.decoupled:
        # FA Pages are P tokens: captures/restores go through a pinned arena.
        from remote_ra_growing_worker.tensors import PageArena
        io.attach_arena(PageArena(io.views, max_pages=arena_pages(runner, startup, layout)))
    from .loaded_sources import verify_closure
    source_evidence=verify_closure(runner)
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from .connector_guard import quiescent
    connector=get_kv_transfer_group()
    io.connector_check=lambda:quiescent(connector)
    io.drain()
    from .vector_space import attach as attach_vector_space
    vector_space=attach_vector_space(runner)
    startup=dict(startup,source_closure_verified=True)
    owner = RunnerAllocationOwner(runner, io, startup, layout)
    runner._remote_ra_live_v2_allocation_owner = owner
    backend = CudaWorkerBackend(io,
                                owner, source=startup["source"])
    owner.bind_backend(backend)
    if layout.decoupled:
        # Batched device-side Page-vector compaction (interface I3).
        from .page_compaction import PageCompactor
        backend.compactor = PageCompactor(vector_space["hidden_size"], layout.page_tokens)
    from .native_probe import NativeQueryForward
    runner._remote_ra_native_query_forward=NativeQueryForward(runner,owner,backend)
    qwen = QwenAlignLifecycle(backend, page_tokens=startup["page_tokens"],
                              mamba_tokens=layout.mamba_tokens)
    bridge = WorkerBridge(qwen, owner)
    owner.bind_bridge(bridge)
    runner._remote_ra_live_v2_bridge = bridge
    from .generation_guard import guarded_execute
    runner.execute_model = guarded_execute(runner, runner.execute_model)
    worker._remote_ra_live_v2_backend = backend
    worker._remote_ra_live_v2_qwen = qwen
    evidence = {"evidence_kind": backend.io.evidence_kind,
            "loaded_sources_verified": startup["loaded_sources_verified"],
            "copies_drained": True, "source": startup["source"],
            "source_closure":source_evidence,
            "vector_space":vector_space,
            "tensor_geometry":[dict(layer=v.name,group=v.group,kind=v.kind,
                allocation_tokens=v.allocation_tokens,kernel_tokens=v.kernel_tokens,ratio=v.ratio,
                page_bytes=v.page_bytes,page_stride=v.page_stride,cohort=list(v.cohort),
                shapes=[list(t.shape) for t in v.tensors],strides=[list(t.stride()) for t in v.tensors])
                for v in io.views]}
    if layout.decoupled:
        # Additive READY evidence; legacy evidence keeps its exact key set.
        from .kv_layout import to_additional_config
        evidence.update(kv_layout=to_additional_config(layout),
                        page_vectors=backend.compactor.evidence(), dram_arena=io.arena.stats())
    return evidence


def worker_dispatch(worker, operation, payload):
    backend = worker._remote_ra_live_v2_backend
    backend.verify_identity()
    # The allocation owner translates the authenticated engine command to the
    # concrete NativeIdentity/RenderAssignment/Allocation set.  This is a
    # required physical API, never reconstructed from public request fields.
    owner = backend.owner
    command = owner.resolve_worker_command(operation, payload)
    from remote_ra_http.evidence import measured_command
    import os
    # Frames carry the request's own IO: requests overlap in batched decode.
    receipt = (measured_command(backend, operation, command, worker._remote_ra_live_v2_qwen,
                                request=payload.get("request"))
               if os.environ.get("REMOTE_RA_HTTP_RUNTIME", "0") == "1"
               else command(backend, worker._remote_ra_live_v2_qwen))
    require(type(receipt) is dict and receipt.get("operation") == operation and
            (operation == "close_session" or receipt.get("binding") == payload["binding"]),
            "physical worker command receipt mismatch")
    return receipt
