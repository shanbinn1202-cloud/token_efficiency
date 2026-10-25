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

    # --decode-fastpath (decode_fastpath.attach_worker): the step's positions are read
    # from the device once, and the forward is joined by one drain for the step.
    decode_fastpath = False
    # --decode-fastpath2 (decode_fastpath2.attach_worker): ``after_forward`` starts the
    # step's finiteness read and the generation guard runs the per-request after-forward
    # work once the native execute_model returned (``finish_step``).
    decode_fastpath2 = False
    # --decode-fastpath3 (decode_fastpath3.attach_worker): ``before_forward`` runs
    # decode_fastpath3.before_forward.
    decode_fastpath3 = False
    _deferred = None
    _finite_host = None

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
            if self.decode_fastpath3:
                from .decode_fastpath3 import before_forward
                return before_forward(self, runner, scheduler_output, positions)
            from remote_ra_growing_worker.tensors import attributed
            from .positions import text_positions
            steps = self.owner.step(runner, scheduler_output)
            host = None
            if self.decode_fastpath and steps:
                # One device read of the step's columns; each request below still
                # checks its own slice in order (positions are not touched by materialize).
                from .positions import step_positions
                host = step_positions(runner, positions, max(s.offset + s.count for s in steps))
            leases = []
            for s in steps:
                state = self.assignments.get(s.request)
                require(state is not None, "worker assignment absent")
                with attributed(self.owner.io, s.request):
                    self.owner.materialize(s)
                    logical, evidence = text_positions(runner, positions, s.start, s.count,
                                                       s.request, s.offset,
                                                       **({} if host is None else dict(host=host)))
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
            if (self.decode_fastpath2 and isinstance(hidden_states, torch.Tensor) and
                    hidden_states.ndim == 2):
                from .native_probe import mixed_probe
                if mixed_probe(runner) is None:
                    # With an in-step phase-1 probe the work runs in place, as before.
                    return self._defer(rows, hidden_states)
            finite = None
            if isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 2:
                total = rows[-1][2] + rows[-1][3]
                finite = torch.isfinite(hidden_states[:total]).all(dim=1).cpu()
            drained = {}
            if (self.decode_fastpath and rows and
                    not getattr(self.qwen.backend, 'handoff', False)):
                # One drain joins the step's forward for every request (each
                # forward_complete drained it again; nothing is issued between).
                self.qwen.backend.io.drain()
                drained = dict(drained=True)
            receipts = []
            for request, _, offset, count in rows:
                state = self.assignments[request]
                # Pages are sealed and published at terminal complete (spec 3.8).
                receipts.append(self.qwen.after_forward(state.assignment.identity,
                    hidden_states[offset:offset + count],
                    finite=None if finite is None else finite[offset:offset + count], **drained))
            return receipts[0] if len(receipts) == 1 else tuple(receipts)
        callback = getattr(scheduler_output, "remote_ra_live_v2_after", None)
        if callback is not None:
            return callback(self.qwen, hidden_states)

    # ---- --decode-fastpath2: the after-forward work after the native execute_model ----

    def _defer(self, rows, hidden_states):
        """Start the step's finiteness read (no wait) and keep the step for ``finish_step``."""
        import torch
        require(self._deferred is None, 'an earlier step kept its after-forward work')
        total = rows[-1][2] + rows[-1][3]
        finite = torch.isfinite(hidden_states[:total]).all(dim=1)
        event = None
        if finite.device.type == 'cuda':
            # One pinned buffer, reused: the copy is joined before the next step starts.
            host = self._finite_host
            if host is None or host.shape[0] < total:
                host = self._finite_host = torch.empty(max(total, 256), dtype=torch.bool, pin_memory=True)
            host = host[:total]
            host.copy_(finite, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        else:
            host = finite.cpu()
        self._deferred = (rows, hidden_states, host, event)
        return None

    def take_deferred(self):
        """The kept step (rows, hidden rows, host finiteness, its event), once; None when none."""
        deferred, self._deferred = self._deferred, None
        return deferred

    def finish_step(self):
        """The generation guard, after the native execute_model returned: the step's rows are
        checked finite (the read started after the forward; the logits launched after it may
        still run) and each request's lease closes on its rows, in batch order.

        The read's completion joins the forward on the stream; with the connector checks of
        a drain around it, it stands for the drain of the step (``drained``), and the Qwen
        lifecycle's ``leave_execute`` then skips its own (``step_joined``)."""
        deferred = self.take_deferred()
        if deferred is None:
            return None
        rows, hidden_states, host, event = deferred
        backend = self.qwen.backend
        handoff = getattr(backend, 'handoff', False)
        check = getattr(backend.io, 'connector_check', None)
        if check is not None and not handoff:
            check()
        if event is not None:
            event.synchronize()
        if check is not None and not handoff:
            check()
        every = bool(host.all())
        drained = {} if handoff else dict(drained=True)
        receipts = []
        for request, _, offset, count in rows:
            state = self.assignments[request]
            receipts.append(self.qwen.after_forward(state.assignment.identity,
                hidden_states[offset:offset + count],
                finite=None if every else host[offset:offset + count], finite_ok=every, **drained))
        if not handoff:
            self.qwen.step_joined = True
        return receipts[0] if len(receipts) == 1 else tuple(receipts)

    def discard_step(self):
        """A failed execute: nothing kept survives it (the guard's failure path joins and poisons)."""
        self._deferred = None
        self.qwen.step_joined = False

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
    if getattr(layout, "state", None) == "sliding":
        # The window layout of a sliding-window model (gemma-s1): its own attachment.
        from .window_worker import attach_window_worker
        return attach_window_worker(worker, startup, layout, inputs)
    io = TensorIO(views_from_runner(runner, *inputs), decoupled=layout.decoupled)
    check_view_geometry(io, layout, getattr(runner.kv_cache_config, "group_pool", None))
    # CUDA graphs: the loaded runner's resolved compilation config must be the
    # pinned mode (None: eager, and a runner with graphs is refused).
    from .cuda_graphs import attach as attach_cuda_graphs
    graphs = attach_cuda_graphs(runner, startup.get("cuda_graphs"),
                                sorted({v.group for v in io.views if v.kind == "attention"}))
    # The pinned reply bound and context limit (None: the defaults, nothing sent):
    # the runner's max_model_len must be the pinned one. Startup only.
    from .context_limits import RECORD_KEY, check_worker
    context = check_worker(runner, startup.get(RECORD_KEY))
    # Admission flags: the startup record, checked once against the pinned config.
    from .admission_flags import worker_flags
    admission = worker_flags(startup)
    # --decode-fastpath: the startup record, checked once against the pinned config.
    from .decode_fastpath import CONFIG_KEY as FASTPATH_KEY, worker_flag as fastpath_flag
    fastpath = fastpath_flag(startup)
    # --decode-fastpath2: the same, once; it needs --decode-fastpath.
    from .decode_fastpath2 import CONFIG_KEY as FASTPATH2_KEY, worker_flag as fastpath2_flag
    fastpath2 = fastpath2_flag(startup)
    require(not fastpath2 or fastpath, "decode_fastpath2 needs decode_fastpath")
    # --decode-fastpath3: the same, once; it needs --decode-fastpath2.
    from .decode_fastpath3 import CONFIG_KEY as FASTPATH3_KEY, worker_flag as fastpath3_flag
    fastpath3 = fastpath3_flag(startup)
    require(not fastpath3 or fastpath2, "decode_fastpath3 needs decode_fastpath2")
    # --language-model-only: the startup record against the pinned config and the loaded
    # runner (text-only limits, no vision-tower tensor), once.
    from .language_model_only import CONFIG_KEY as TEXT_ONLY_KEY, worker_check as text_only_check
    text_only = text_only_check(runner, startup)
    # --release-idle-pages: the startup record, checked once against the pinned config.
    from .idle_release import CONFIG_KEY as RELEASE_KEY, worker_flag as release_flag
    release = release_flag(startup)
    require(not release or layout.decoupled, "release_idle_pages needs the decoupled KV layout")
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
    # --connector-pool-gib: the loaded connector's CPU pool against the pin, once (nothing without it).
    from .connector_guard import check_worker_pool
    check_worker_pool(connector)
    io.drain()
    from .vector_space import attach as attach_vector_space
    vector_space=attach_vector_space(runner)
    startup=dict(startup,source_closure_verified=True)
    owner = RunnerAllocationOwner(runner, io, startup, layout)
    if release:
        owner.release_idle = True       # evict/readmit accepted
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
    if fastpath2:
        # Before the step profile wraps the bridge (its deferred after-forward names).
        from .decode_fastpath2 import attach_worker as attach_fastpath2
        attach_fastpath2(owner, bridge, qwen, io)
    if fastpath3:
        # After the second fast path, before the step profile wraps the bridge.
        from .decode_fastpath3 import attach_worker as attach_fastpath3
        attach_fastpath3(owner, bridge, qwen)
    from .generation_guard import guarded_execute
    import remote_ra_step_profile as step_profile
    if step_profile.ENABLED:
        # REMOTE_RA_STEP_PROFILE=1 only: per-step timers; the guard is timed around
        # the timed native execute_model. Unset, this check is all that runs.
        guarded_execute = step_profile.install_worker(runner, bridge, guarded_execute)
    runner.execute_model = guarded_execute(runner, runner.execute_model)
    if admission is not None:
        from .gdn_handoff import attach_worker
        attach_worker(admission, io=io, owner=owner, backend=backend, qwen=qwen)
    if fastpath:
        from .decode_fastpath import attach_worker as attach_fastpath
        attach_fastpath(owner, bridge, qwen)
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
    if graphs is not None:
        # Additive READY evidence, once at startup; eager evidence keeps its key set.
        evidence.update(cuda_graphs=graphs)
    if context is not None:
        # Additive, non-default only: the engine compares this echo with its pin.
        evidence[RECORD_KEY] = context
    if admission is not None:
        evidence.update(admission=dict(admission))
    if fastpath:
        # Additive, on only: the engine and write_ready compare it with the pin.
        evidence[FASTPATH_KEY] = True
    if fastpath2:
        evidence[FASTPATH2_KEY] = True
    if fastpath3:
        evidence[FASTPATH3_KEY] = True
    if text_only is not None:
        # Additive, on only: the engine and write_ready compare it with the pin.
        evidence[TEXT_ONLY_KEY] = text_only
    if release:
        evidence[RELEASE_KEY] = True    # additive, on only
    return evidence


def worker_dispatch(worker, operation, payload):
    backend = worker._remote_ra_live_v2_backend
    backend.verify_identity()
    # The allocation owner translates the authenticated engine command to the
    # concrete NativeIdentity/RenderAssignment/Allocation set.  This is a
    # required physical API, never reconstructed from public request fields.
    owner = backend.owner
    if operation == "allocation_batch":
        # --decode-fastpath2: the step's allocation updates, each as its own RPC was.
        from .decode_fastpath2 import dispatch_batch
        return dispatch_batch(worker, backend, owner, payload)
    command = owner.resolve_worker_command(operation, payload)
    from remote_ra_http.evidence import measured_command
    import os
    # Frames carry the request's own IO: requests overlap in batched decode.
    if os.environ.get("REMOTE_RA_HTTP_RUNTIME", "0") == "1":
        receipt = measured_command(backend, operation, command, worker._remote_ra_live_v2_qwen,
                                   request=payload.get("request"))
    elif getattr(backend.io, "handoff", False) is True:
        # --gdn-device-handoff: join issued copies at the frame boundaries.
        backend.io.poll_joins()
        receipt = command(backend, worker._remote_ra_live_v2_qwen)
        backend.io.poll_joins()
    else:
        receipt = command(backend, worker._remote_ra_live_v2_qwen)
    # phase1_collect (--phase1-in-step) answers with today's phase-1 receipt.
    expected = "phase1" if operation == "phase1_collect" else operation
    require(type(receipt) is dict and receipt.get("operation") == expected and
            (operation in ("close_session", "evict") or receipt.get("binding") == payload["binding"]),
            "physical worker command receipt mismatch")
    return receipt
