"""Loaded-process composition for the live four-policy runtime.

This module is the only place where the source contracts are turned into
process attachments.  It deliberately has no environment based fallbacks:
the trusted startup ticket owns the engine attachment and the loaded
``GPUWorker`` owns the CUDA attachment.  The feature-OFF path is a no-op.
"""
from __future__ import annotations

from types import SimpleNamespace

from .contract import require
from .lifecycle import EngineLifecycle
from .allocation_owner import RunnerAllocationOwner
# Worker half lives in worker_composition; gpu_worker.py imports it from here.
from .worker_composition import attach_loaded_worker, worker_dispatch


class EngineWorkerProxy:
    """Engine-side handle for the one loaded UniProc worker.

    Tensor objects never cross this boundary.  Every operation is an exact
    collective RPC and is correlated by the worker's signed receipt.
    """

    def __init__(self, executor, receipt):
        require(type(receipt) is dict and
                receipt.get("evidence_kind") == "CUDA_PHYSICAL_TENSOR_OPERATIONS" and
                receipt.get("loaded_sources_verified") is True,
                "loaded CUDA worker attachment receipt required")
        self.executor, self.receipt = executor, dict(receipt)
        self.http_receipts = {}
        self.io = SimpleNamespace(evidence_kind=receipt["evidence_kind"])

    OPERATIONS = frozenset({"prepare", "phase1", "phase1_drain", "reserve",
                            "allocation_update", "complete", "cancel", "close_session"})

    def call(self, operation, payload, *, query_tokens=None):
        """One worker RPC. ``query_tokens`` (phase1 only) is evidence, never sent:
        the Goal tokens of a Goal forward or the observation T[o:e]."""
        require(operation in self.OPERATIONS and type(payload) is dict and
                (query_tokens is None or operation == "phase1"),
                "bounded worker operation required")
        values = self.executor.collective_rpc(
            "remote_ra_live_v2_dispatch", args=(operation, payload), timeout=30)
        require(type(values) is list and len(values) == 1 and
                type(values[0]) is dict, "one loaded worker receipt required")
        from remote_ra_http.evidence import copy_receipt
        item = copy_receipt(values[0])
        if query_tokens is not None:
            from remote_ra_http.config import digest
            item["query_token_sha256"] = digest(list(query_tokens))
        self.http_receipts.setdefault(payload.get("request", "session"), []).append(item)
        return values[0]


class LiveEndpoint:
    """Exact engine endpoint consumed by :class:`EngineLifecycle`.

    The endpoint is installed only after the existing trusted renderer and
    native adapter have become active.  ``render_claim`` records the binding
    before worker reservation so partial admission remains cancellable.
    """

    def __init__(self, core, assembly, worker):
        self.core, self.assembly, self.worker = core, assembly, worker
        self.claims = {}
        self.prepared_payloads, self.prepared_drains = {}, {}

    def verify_identity(self):
        owner = self.core._remote_ra_growing_owner
        owner.owner_check(); owner.verify_identity()
        require(self.assembly is self.core._remote_ra_assembly and
                self.assembly.state == "active", "active engine assembly changed")

    @staticmethod
    def frame_identity(body):
        require(type(body) is dict, "private v2 add body required")
        return tuple(body[k] for k in ("request", "session", "turn", "generation"))

    @staticmethod
    def binding_identity(binding):
        return (binding.request, binding.session, binding.turn,
                binding.generation)

    def render_claim(self, body):
        # Decode/preprocess/render are owned by the installed private-v2
        # endpoint.  It returns before reserve, which is the required cleanup
        # ownership boundary.
        endpoint = self.core._remote_ra_private_v2_endpoint
        request, binding = endpoint.render_claim(body)
        require(binding.request not in self.claims, "binding replay")
        self.claims[binding.request] = binding
        return request, binding

    def reserve_worker(self, request, binding):
        from .goal_async import run_inline
        return run_inline(self.reserve_worker_steps(request, binding))

    def reserve_worker_steps(self, request, binding):
        """``reserve_worker`` as an admission generator: the private reserve may
        yield its goal_query Goal once (goal_async); a failure at any point,
        the Goal included, takes the same cleanup."""
        endpoint = self.core._remote_ra_private_v2_endpoint
        payload = {
            "request": binding.request, "session": binding.session,
            "turn": binding.turn, "generation": binding.generation,
            "binding": binding.identity,
        }
        payload.update(endpoint.worker_payload(binding))
        engine_bridge = self.core._remote_ra_live_v2_bridge
        self.prepared_payloads[binding.request] = payload
        try:
            receipt = self.worker.call("prepare", payload)
            require(receipt.get("binding") == binding.identity and
                    receipt.get("operation") == "prepare", "prepare receipt mismatch")
            assignment = RunnerAllocationOwner._assignment(payload["assignment"])
            engine_bridge.publish_assignment(assignment,receipt["allocation_generation"])
            endpoint.claims[binding.request]['allocation_generation']=receipt['allocation_generation']
            steps = getattr(endpoint, "reserve_steps", None)
            if steps is None:
                endpoint.reserve(request, binding)
            else:
                yield from steps(request, binding)
            self.core._remote_ra_live_v2_boundary.submit(binding)
        except BaseException as failure:
            cleanup_errors = []
            try:
                self.recover_probes(binding)
                if self.core._remote_ra_growing_adapter.owns(binding.request):
                    drained = self.core._remote_ra_growing_adapter.cancel(binding)
                else:
                    drained = self.worker.call("cancel", payload)
                    require(drained.get("operation") == "cancel" and
                            drained.get("binding") == binding.identity and
                            drained.get("drained") is True,
                            "prepared worker cleanup receipt mismatch")
            except BaseException as cleanup:
                cleanup_errors.append(cleanup)
            if cleanup_errors:
                raise BaseExceptionGroup("reserve and prepared-worker cleanup failed",
                                         [failure, *cleanup_errors]) from None
            self.prepared_drains[binding.request] = drained
            self.prepared_payloads.pop(binding.request, None)
            engine_bridge.assignments.pop(binding.request, None)
            raise

    def enqueue(self, request, binding):
        self.core.add_request(request, 0)

    def recover_probes(self, binding):
        # Drain any undrained phase-1 lease and free un-adopted phase-1 blocks.
        # Adopted blocks belong to the reservation and drain with it.
        phase1 = getattr(self.core._remote_ra_private_v2_endpoint, 'phase1', None)
        if phase1 is not None:
            phase1.release(binding.request, binding.identity)

    def drain(self, binding, operation):
        self.recover_probes(binding)
        worker_operation = "complete" if operation == "complete" else "cancel"
        prior = self.prepared_drains.get(binding.request)
        if prior is not None:
            require(operation == "cancel", "prepared request cannot complete")
            return prior
        # Engine adapter owns the one RPC drain and native BlockPool release.
        if operation == "complete":
            receipt = self.core._remote_ra_growing_adapter.complete(binding)
        elif self.core._remote_ra_growing_adapter.owns(binding.request):
            receipt = self.core._remote_ra_growing_adapter.cancel(binding)
        else:
            payload = self.prepared_payloads[binding.request]
            receipt = self.worker.call("cancel", payload)
            require(receipt.get("operation") == "cancel" and
                    receipt.get("binding") == binding.identity and
                    receipt.get("drained") is True,
                    "prepared worker retry cleanup receipt mismatch")
            self.prepared_drains[binding.request] = receipt
            self.prepared_payloads.pop(binding.request, None)
        self.core._remote_ra_live_v2_bridge.assignments.pop(binding.request, None)
        return receipt

    def after_drain(self, binding, operation):
        if operation == "complete":
            self.core._remote_ra_private_v2_endpoint.commit_transcript(binding)
        adapter = self.core._remote_ra_growing_adapter
        if operation != "complete" and adapter.owns(binding.request):
            # Entry is terminal before abort synchronously re-enters scheduler
            # cleanup, so compat never tries to drain the same lease twice.
            self.core.abort_requests([binding.request])
            if adapter.owns(binding.request):
                # Never enqueued (reserve failed after submit): the installed
                # rows exist only in the manager tables; free them natively.
                claim = self.core._remote_ra_private_v2_endpoint.claims[binding.request]
                self.core.scheduler.kv_cache_manager.free(claim["request"])

    @staticmethod
    def verify_drain(binding, operation, receipt):
        expected = "complete" if operation == "complete" else "cancel"
        require(type(receipt) is dict and receipt.get("operation") == expected and
                receipt.get("binding") == binding.identity and
                receipt.get("drained") is True, "worker drain receipt mismatch")


class EngineBridge:
    """Engine scheduler callback owner; the runner half is WorkerBridge."""

    def __init__(self):
        self.assignments = {}

    def publish_assignment(self, assignment, allocation_generation):
        """Engine scheduler view; Qwen/tensor state stays on the worker."""
        require(assignment.identity.request not in self.assignments and
                type(allocation_generation) is int and allocation_generation > 0,
                "engine assignment replay/invalid generation")
        state = SimpleNamespace(assignment=assignment,
                                allocation_generation=allocation_generation)
        self.assignments[assignment.identity.request] = state
        return state

    def scheduler_split(self, request, start, count):
        # Stage B has no seal at admission: the seal is decided at completion,
        # so phase 2 never splits (the native Mamba-aligned split still applies).
        return count


def trusted_tasks(task):
    """Multi-session stage 1: the startup task list, one engine session per task.

    The list is pinned in the same startup config as ``task`` (the
    ``legacy_geometry`` task), and ``task`` is its first entry, session 0.
    Without the HTTP runtime there is no list and the server has one session.
    """
    import os
    tasks=[task]
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME','0')=='1':
        from remote_ra_http.config import load
        tasks=[dict(t) for t in load()['tasks']]
    from .sessions import validate_tasks
    validate_tasks(tasks)
    require(tasks[0]==task,'session 0 must serve the trusted startup issue')
    return tasks


def compose_engine(core, ticket):
    """Attach the active protocol-v2 lifecycle before B0/IO threads."""
    if ticket is None:
        return None
    state = core._remote_ra_live_v2
    assembly = core._remote_ra_assembly
    if not hasattr(core,'_remote_ra_trusted_issue'):
        task=dict(core._remote_ra_growing_owner.smoke_config['task'])
        require(set(task)=={'instance_id','problem_statement'},'trusted startup task record required')
        core._remote_ra_trusted_issue=task
    if not hasattr(core,'_remote_ra_trusted_issues'):
        # Indexed like the sessions; _remote_ra_trusted_issue stays tasks[0].
        core._remote_ra_trusted_issues=trusted_tasks(core._remote_ra_trusted_issue)
    startup = {"enabled": True,
               "source": core._remote_ra_growing_owner.identity.manifest,
               "page_tokens": assembly.provider.provenance.page_tokens,
               "loaded_sources_verified": True}
    # The decoupled KV layout (interface I0/I2) travels once, in this startup
    # record, never per step; the legacy record is unchanged.
    layout = getattr(core._remote_ra_growing_adapter, "layout", None)
    if layout is not None and layout.decoupled:
        from .kv_layout import LAYOUT_KEY, to_additional_config
        require(layout.page_tokens == startup["page_tokens"], "KV layout Page size differs")
        startup["kv_layout"] = dict(to_additional_config(layout)[LAYOUT_KEY])
    replies = core.model_executor.collective_rpc(
        "remote_ra_live_v2_attach", args=(startup,), timeout=30)
    require(type(replies) is list and len(replies) == 1,
            "one loaded worker attachment required")
    proxy = EngineWorkerProxy(core.model_executor, replies[0])
    from .engine_boundary import EngineRpcBoundary
    boundary = EngineRpcBoundary(proxy, core._remote_ra_growing_adapter)
    core._remote_ra_growing_adapter.backend = boundary
    core._remote_ra_live_v2_boundary = boundary
    from .session_authority import LiveSessionAuthority
    if not hasattr(core, "_remote_ra_private_v2_endpoint"):
        core._remote_ra_live_session_authority = LiveSessionAuthority(core, assembly, proxy)
    if not hasattr(core, "_remote_ra_private_v2_endpoint"):
        from .private_endpoint import SplitPrivateEndpoint
        core._remote_ra_private_v2_endpoint = SplitPrivateEndpoint(core, assembly)
    transcript = core._remote_ra_private_v2_endpoint.transcript
    if transcript is not None:
        # XML protocol: the engine stops on the action marker itself, so the
        # request finishes as complete instead of a frontend stop-string abort.
        # One policy stops every session's request: it decodes only the
        # request's own output, through the decoder all transcripts share.
        from .mswea_protocol import MarkerStop
        require(all(t.decode == transcript.decode
                    for t in core._remote_ra_private_v2_endpoint.transcripts.values()),
                "engine transcripts must share one decoder")
        require(getattr(core.scheduler, "_remote_ra_output_stop", None) is None,
                "output stop policy attachment collision")
        core.scheduler._remote_ra_output_stop = MarkerStop(transcript.decode)
    endpoint = LiveEndpoint(core, assembly, proxy)
    lifecycle = EngineLifecycle(endpoint)
    bridge = EngineBridge()
    core._remote_ra_live_v2_endpoint = endpoint
    core._remote_ra_live_v2_lifecycle = lifecycle
    core._remote_ra_live_v2_bridge = bridge
    core.scheduler._remote_ra_live_v2_bridge = bridge
    # EngineAssembly itself is constructed with the v3 lifecycle owner (not
    # lifecycle_hooks.EngineLifecycle).  Bind its protocol-v2 half now, after
    # the loaded worker receipt and before B0/IO threads.
    assembly.lifecycle.bind_native(lifecycle)
    from .registered_route import bind_registered_dispatcher
    bind_registered_dispatcher(assembly.dispatcher, lifecycle)
    # Multi-session stage 2: when the sessions decode together, a goal_query
    # admission waits for its Goal off the engine thread (goal_async); a parked
    # Goal wakes the idle engine loop through its input queue.
    from vllm.v1.engine import EngineCoreRequestType
    from .goal_async import attach
    attach(core, lifecycle, max_num_seqs=core.vllm_config.scheduler_config.max_num_seqs,
           wake=lambda: core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None)))
    bind = __import__("remote_ra_live_v2.startup", fromlist=["bind_loaded_worker"])
    from remote_ra_http.connector import isolate_connector
    import os
    if os.environ.get("REMOTE_RA_HTTP_RUNTIME", "0") == "1":
        isolate_connector(core)
    bind.bind_loaded_worker(core, worker_backend=proxy, qwen_lifecycle=bridge,
        engine_lifecycle=lifecycle, cuda_evidence=replies[0])
    return state


def compose_frontend(serving, controller, *, policy, instance_id, readiness,
                     tasks=None, transcripts=None, max_num_seqs=1):
    """Attach startup-owned four-policy authority; disabled routes are untouched.

    One live session per startup task, with ids ``sessions.session_ids`` of the
    server id ``instance_id``, each with its own frontend transcript.
    ``max_num_seqs`` is the pinned engine batch size (stage 2: one per session).
    """
    from .controller import LiveFourPolicyController
    live = LiveFourPolicyController(controller, startup_policy=policy,
        instance_id=instance_id, readiness=readiness, session_id=instance_id,
        tasks=tasks, transcripts=transcripts, max_num_seqs=max_num_seqs)
    existing = getattr(serving, "_remote_ra_live_four_policy", None)
    require(existing is None or existing is live,
            "four-policy frontend attachment collision")
    return live
