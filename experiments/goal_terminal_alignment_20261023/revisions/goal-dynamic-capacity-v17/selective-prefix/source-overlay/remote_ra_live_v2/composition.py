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
                            "allocation_update", "complete", "cancel", "close_session",
                            "phase1_arm", "phase1_collect",      # --phase1-in-step
                            "allocation_batch",                  # --decode-fastpath2
                            "evict", "readmit"})                 # --release-idle-pages
    # --decode-fastpath2 (decode_fastpath2.attach_engine): sends the queued allocation
    # updates before any other worker RPC; None (off).
    before_call = None

    def call(self, operation, payload, *, query_tokens=None):
        """One worker RPC. ``query_tokens`` (phase1 / phase1_collect only) is evidence,
        never sent: the Goal tokens of a Goal forward or the observation T[o:e]."""
        require(operation in self.OPERATIONS and type(payload) is dict and
                (query_tokens is None or operation in ("phase1", "phase1_collect")),
                "bounded worker operation required")
        if self.before_call is not None and operation != "allocation_batch":
            self.before_call()
        values = self.executor.collective_rpc(
            "remote_ra_live_v2_dispatch", args=(operation, payload), timeout=30)
        require(type(values) is list and len(values) == 1 and
                type(values[0]) is dict, "one loaded worker receipt required")
        from remote_ra_http.evidence import copy_receipt
        if operation == "allocation_batch":
            # One frame per entry under its request, as its own allocation_update RPC
            # filed it (one JSON copy of the list gives each entry's copy).
            receipts, entries = values[0].get("receipts"), payload.get("entries")
            require(type(receipts) is list and type(entries) is list and len(receipts) == len(entries),
                    "one worker receipt per allocation batch entry required")
            for entry, item in zip(entries, copy_receipt(receipts)):
                self.http_receipts.setdefault(entry["request"], []).append(item)
            return values[0]
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
                self.release_held(binding)
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

    def release_held(self, binding):
        """--gdn-device-handoff: after the worker's drain, a held phase-1 GDN slot goes."""
        phase1 = getattr(self.core._remote_ra_private_v2_endpoint, 'phase1', None)
        held = getattr(phase1, 'handoff', None)
        if held is not None:
            held.release(binding.request)

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
        self.release_held(binding)
        return receipt

    def after_drain(self, binding, operation):
        if operation == "complete":
            self.core._remote_ra_private_v2_endpoint.commit_transcript(binding)
            release = getattr(self.core, "_remote_ra_idle_release", None)
            if release is not None:
                release.touch(binding.session)      # --release-idle-pages: LRU order
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


def session_pool():
    """K of a session-pool server (the startup config's ``max_running_sessions``),
    else None: every startup task's session opens at startup, as before."""
    import os
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME','0')!='1':
        return None
    from remote_ra_http.config import load
    return load().get('max_running_sessions')


def trusted_tasks(task):
    """Multi-session stage 1: the startup task list, one engine session per task.

    The list is pinned in the same startup config as ``task`` (the
    ``legacy_geometry`` task), and ``task`` is its first entry, session 0.
    Without the HTTP runtime there is no list and the server has one session.
    A session-pool server registers up to ``sessions.MAX_TASKS`` tasks.
    """
    import os
    tasks=[task]
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME','0')=='1':
        from remote_ra_http.config import load
        tasks=[dict(t) for t in load()['tasks']]
    from .sessions import task_limit, validate_tasks
    validate_tasks(tasks, task_limit(session_pool() is not None, startup_protocol()))
    require(tasks[0]==task,'session 0 must serve the trusted startup issue')
    return tasks


def startup_protocol():
    """The startup config's agent protocol (HTTP runtime), else None. Startup only."""
    import os
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME','0')!='1':
        return None
    from remote_ra_http.config import load
    return load().get('protocol')


def engine_lifecycle(core, endpoint, **lease):
    """The engine lifecycle, its lifetime request bound sized once, at startup,
    from the trusted task list (``sessions.request_limit``: 256 per session, at least 1,024).
    A session pool sizes it from its registered tasks and admits up to that many
    sessions over its lifetime; at most K are open (LiveSessionAuthority).
    ``lease`` (--admission-overlap): ``lease_seconds``, pinned from K."""
    from .sessions import request_limit, task_limit
    from .session_pool import configured
    tasks = core._remote_ra_trusted_issues
    if configured(core) is None:
        return EngineLifecycle(endpoint, max_requests=request_limit(len(tasks)), **lease)
    # A chat-text server registers up to MAX_CHAT_TEXT_SLOTS slots, every other pool MAX_TASKS.
    return EngineLifecycle(endpoint, max_requests=request_limit(len(tasks), task_limit(True, startup_protocol())),
                           max_sessions=len(tasks), **lease)


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
    if not hasattr(core,'_remote_ra_session_pool'):
        # Session pool: K sessions open at most (None: every task opens at startup).
        core._remote_ra_session_pool=session_pool()
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
    # The pinned CUDA-graph mode travels the same way; off (eager) adds nothing.
    from .cuda_graphs import engine_mode
    graphs = engine_mode()
    if graphs is not None:
        startup["cuda_graphs"] = graphs
    # The pinned reply bound and context limit travel the same way, only when
    # they differ from the defaults (context_limits); the default record is unchanged.
    from .context_limits import ARENA_KEY, RECORD_KEY, engine_arena, engine_record, loaded, pin_request_caps
    context = engine_record()
    if context is not None:
        startup[RECORD_KEY] = context
    # The request caps of a 262144-token server (the source, core request, frame and
    # EngineCore's registered-add message bounds), set once here before the input
    # thread starts; at 65536 and 131072 nothing is set.
    pin_request_caps(loaded(), core)
    # A pinned Page arena (--dram-arena-gib) replaces the worker's default capacity.
    arena = engine_arena()
    if arena is not None:
        startup[ARENA_KEY] = arena
    # The admission flags (admission_flags): read once here, sent once; all off adds nothing.
    from .admission_flags import check_receipt, engine_flags
    admission = engine_flags()
    if admission is not None:
        startup["admission"] = dict(admission)
    # --decode-fastpath (decode_fastpath): read once here, sent once; off adds nothing.
    from .decode_fastpath import CONFIG_KEY as FASTPATH_KEY, loaded as fastpath_pin
    fastpath = fastpath_pin()
    if fastpath:
        startup[FASTPATH_KEY] = True
    # --decode-fastpath2 (decode_fastpath2): read once here, sent once; off adds nothing.
    from .decode_fastpath2 import CONFIG_KEY as FASTPATH2_KEY, loaded as fastpath2_pin
    fastpath2 = fastpath2_pin()
    if fastpath2:
        startup[FASTPATH2_KEY] = True
    # --decode-fastpath3 (decode_fastpath3): read once here, sent once; off adds nothing.
    from .decode_fastpath3 import CONFIG_KEY as FASTPATH3_KEY, loaded as fastpath3_pin
    fastpath3 = fastpath3_pin()
    if fastpath3:
        startup[FASTPATH3_KEY] = True
    # --language-model-only (language_model_only): read once here, sent once; off adds nothing.
    from .language_model_only import CONFIG_KEY as TEXT_ONLY_KEY, RECORD as TEXT_ONLY_RECORD, loaded as text_only_pin
    text_only = text_only_pin()
    if text_only:
        startup[TEXT_ONLY_KEY] = True
    # --release-idle-pages (idle_release): read once here, sent once; off adds nothing.
    from .idle_release import CONFIG_KEY as RELEASE_KEY, loaded as release_pin
    release = release_pin()
    if release:
        startup[RELEASE_KEY] = True
    replies = core.model_executor.collective_rpc(
        "remote_ra_live_v2_attach", args=(startup,), timeout=30)
    require(type(replies) is list and len(replies) == 1,
            "one loaded worker attachment required")
    require(type(replies[0]) is dict and replies[0].get(RECORD_KEY) == context,
            "worker context record differs from the pinned one")
    check_receipt(admission, replies[0])
    require(replies[0].get(FASTPATH_KEY) == fastpath, "worker attached with another decode_fastpath")
    require(replies[0].get(FASTPATH2_KEY) == fastpath2, "worker attached with another decode_fastpath2")
    require(replies[0].get(FASTPATH3_KEY) == fastpath3, "worker attached with another decode_fastpath3")
    require(replies[0].get(TEXT_ONLY_KEY) == (TEXT_ONLY_RECORD if text_only else None),
            "worker attached with another language_model_only")
    require(replies[0].get(RELEASE_KEY) == release, "worker attached with another release_idle_pages")
    proxy = EngineWorkerProxy(core.model_executor, replies[0])
    from .engine_boundary import EngineRpcBoundary
    boundary = EngineRpcBoundary(proxy, core._remote_ra_growing_adapter)
    core._remote_ra_growing_adapter.backend = boundary
    core._remote_ra_live_v2_boundary = boundary
    if fastpath2:
        # One allocation_batch RPC per step, sent when scheduler.schedule returns.
        from .decode_fastpath2 import attach_engine as attach_fastpath2
        attach_fastpath2(core, boundary, proxy)
    if fastpath3:
        # The per-request allocation's imports, bound once (after the second fast path).
        from .decode_fastpath3 import attach_engine as attach_fastpath3
        attach_fastpath3(boundary)
    from .session_authority import LiveSessionAuthority
    if not hasattr(core, "_remote_ra_private_v2_endpoint"):
        core._remote_ra_live_session_authority = LiveSessionAuthority(core, assembly, proxy)
    if not hasattr(core, "_remote_ra_private_v2_endpoint"):
        from .private_endpoint import SplitPrivateEndpoint
        core._remote_ra_private_v2_endpoint = SplitPrivateEndpoint(core, assembly)
    if admission is not None:
        from .gdn_handoff import attach_engine
        attach_engine(core, admission)
    transcript = core._remote_ra_private_v2_endpoint.transcript
    from .kv_session_protocol import kv
    from .chat_text_protocol import chat
    # chat-text stops on EOS/pad only, like kv-session: no action marker stop.
    if transcript is not None and not kv(transcript) and not chat(transcript):
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
        if fastpath2:
            # --decode-fastpath2: also remember per token id whether its decode holds a '>'.
            core.scheduler._remote_ra_output_stop = MarkerStop(transcript.decode, incremental=True,
                                                               token_cache=True)
        elif fastpath:
            # --decode-fastpath: decode the new token, not the whole output, when that decides (MarkerStop).
            core.scheduler._remote_ra_output_stop = MarkerStop(transcript.decode, incremental=True)
        else:
            core.scheduler._remote_ra_output_stop = MarkerStop(transcript.decode)
    endpoint = LiveEndpoint(core, assembly, proxy)
    # --admission-overlap K: the engine lease pinned from K (admission_flags.timeouts).
    from .admission_flags import permits, pinned_timeouts
    overlap = permits(admission)
    lease = {} if overlap == 1 else dict(lease_seconds=pinned_timeouts(admission)[1])
    lifecycle = engine_lifecycle(core, endpoint, **lease)
    bridge = EngineBridge()
    core._remote_ra_live_v2_endpoint = endpoint
    core._remote_ra_live_v2_lifecycle = lifecycle
    core._remote_ra_live_v2_bridge = bridge
    core.scheduler._remote_ra_live_v2_bridge = bridge
    if release:
        # --release-idle-pages: the capacity gate releases idle sessions' Pages.
        from .idle_release import IdleRelease
        from remote_ra_http.config import load as load_config
        core._remote_ra_idle_release = IdleRelease(core, load_config()['output_directory'])
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
    from .admission_flags import on
    # --phase1-in-step: the park table also holds phase-1 works (phase1_step).
    in_step = dict(phase1_in_step=True) if on(admission, "phase1_in_step") else {}
    # --admission-overlap: the table parks several admissions at once.
    several = {} if overlap == 1 else dict(overlap=overlap)
    attach(core, lifecycle, max_num_seqs=core.vllm_config.scheduler_config.max_num_seqs,
           wake=lambda: core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None)), **in_step, **several)
    if in_step:
        from .phase1_step import attach_engine as attach_phase1_step
        attach_phase1_step(core, admission)
    if overlap > 1:
        # --admission-overlap K: the ledger, the lease wait, per-session Goal constructors.
        from .admission_overlap import attach_engine as attach_overlap
        attach_overlap(core, lifecycle, admission)
    bind = __import__("remote_ra_live_v2.startup", fromlist=["bind_loaded_worker"])
    from remote_ra_http.connector import isolate_connector
    import os
    if os.environ.get("REMOTE_RA_HTTP_RUNTIME", "0") == "1":
        isolate_connector(core)
        # Startup only: the engine's vLLM context limit is the pinned one.
        from remote_ra_http.config import load
        from .context_limits import check_engine
        check_engine(core.vllm_config, load())
        # Startup only: the engine's resolved multimodal config agrees with --language-model-only.
        from .language_model_only import check_engine as check_text_only
        check_text_only(core.vllm_config, text_only)
        # Startup only: vLLM's connector CPU pool is the pinned one (--connector-pool-gib).
        from .connector_guard import check_engine_pool
        check_engine_pool(core.vllm_config, load())
    bind.bind_loaded_worker(core, worker_backend=proxy, qwen_lifecycle=bridge,
        engine_lifecycle=lifecycle, cuda_evidence=replies[0])
    import remote_ra_step_profile as step_profile
    if step_profile.ENABLED:
        # REMOTE_RA_STEP_PROFILE=1 only: the busy loop with step marks, flushed at
        # shutdown. Unset, this check is all that runs.
        step_profile.attach_engine(core)
    return state


def admission_permits():
    """--admission-overlap K pinned in the startup config (frontend startup); 1 (off)
    without the HTTP runtime or without the flag."""
    import os
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME','0')!='1':
        return 1
    from .admission_flags import engine_flags, permits
    return permits(engine_flags())


def kv_session_frontend(controller, tasks, transcripts):
    """Frontend startup of a kv-session server (startup config protocol kv_session_v1):
    one token transcript per slot, given to the capture factory that live_frontend
    built without transcripts (its XML-only branch), and the client-session state.
    (None, ``transcripts``) on every other server and without the HTTP runtime."""
    import os
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None, transcripts
    from remote_ra_http.config import load
    from .kv_session_protocol import PROTOCOL
    config = load()
    if config.get('protocol') != PROTOCOL:
        return None, transcripts
    capture = controller.registry.capture_factory
    require(transcripts is None and capture.transcripts == () and tasks == config['tasks'],
            "kv-session frontend: the capture factory must not hold XML transcripts")
    make = capture.source_renderer.kv_transcripts(config['kv_session'])
    transcripts = tuple(make() for _ in tasks)
    capture.transcripts = transcripts
    from remote_ra_http.kv_session_route import frontend
    return frontend(config), transcripts


def chat_text_frontend(controller, tasks, transcripts):
    """Frontend startup of a chat-text server (startup config protocol chat_text_v1):
    one chat transcript per slot, given to the capture factory that live_frontend
    built without transcripts, and the client-session state (ChatTextFrontend).
    (None, ``transcripts``) on every other server and without the HTTP runtime."""
    import os
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return None, transcripts
    from remote_ra_http.config import load
    from .chat_text_protocol import PROTOCOL
    config = load()
    if config.get('protocol') != PROTOCOL:
        return None, transcripts
    capture = controller.registry.capture_factory
    require(transcripts is None and capture.transcripts == () and tasks == config['tasks'],
            "chat-text frontend: the capture factory must not hold XML transcripts")
    make = capture.source_renderer.chat_transcripts(config['chat_text'])
    transcripts = tuple(make() for _ in tasks)
    capture.transcripts = transcripts
    from remote_ra_http.chat_text_route import frontend
    return frontend(config), transcripts


def compose_frontend(serving, controller, *, policy, instance_id, readiness,
                     tasks=None, transcripts=None, max_num_seqs=1, max_running=None,
                     max_tokens=None, admission_overlap=None, kv_session=None, chat_text=None):
    """Attach startup-owned four-policy authority; disabled routes are untouched.

    One live session per startup task, with ids ``sessions.session_ids`` of the
    server id ``instance_id``, each with its own frontend transcript.
    ``max_num_seqs`` is the pinned engine batch size (stage 2: one per session).
    ``max_running`` is a session pool's K; None reads it from the startup config
    (``session_pool``), where its absence keeps every session open.
    ``max_tokens`` is the server's reply bound; None reads it from the startup
    config (``context_limits``; the default 1536 without one).
    ``admission_overlap`` (--admission-overlap K): the admission permits; None
    reads them from the startup config (``admission_permits``; 1 when absent).
    ``kv_session``: a kv-session server's frontend state
    (remote_ra_http.kv_session_route.KvFrontend: client sessions -> slots, info);
    its sessions are the pool's slots and /v1/chat/completions is refused. None
    reads the startup config: on a kv_session_v1 server ``kv_session_frontend``
    builds it and the slots' token transcripts (live_frontend passes none).
    ``chat_text``: a chat-text server's frontend state
    (remote_ra_http.chat_text_route.ChatTextFrontend: client sessions -> slots, the
    idle closes); its sessions are the pool's slots, and /v1/chat/completions takes
    the chat-text route. None reads the startup config (``chat_text_frontend``).
    """
    from .controller import LiveFourPolicyController
    from .context_limits import loaded, pin_request_caps
    if max_running is None:
        max_running = session_pool()
    if max_tokens is None:
        from .context_limits import max_tokens as pinned
        max_tokens = pinned(loaded())
    # The request caps of a 262144-token server (the source and frame bounds of the
    # frontend's render and codec), set once at startup; at 65536 and 131072 nothing is set.
    pin_request_caps(loaded())
    if admission_overlap is None:
        admission_overlap = admission_permits()
    permits = {} if admission_overlap == 1 else dict(admission_overlap=admission_overlap)
    if kv_session is None:
        kv_session, transcripts = kv_session_frontend(controller, tasks, transcripts)
    if chat_text is None and kv_session is None:
        chat_text, transcripts = chat_text_frontend(controller, tasks, transcripts)
    if chat_text is not None:
        from .sessions import CHAT_TEXT_PROTOCOL
        permits['task_protocol'] = CHAT_TEXT_PROTOCOL     # its slot cap (sessions.task_limit)
    live = LiveFourPolicyController(controller, startup_policy=policy,
        instance_id=instance_id, readiness=readiness, session_id=instance_id,
        tasks=tasks, transcripts=transcripts, max_num_seqs=max_num_seqs,
        max_running=max_running, max_tokens=max_tokens, **permits)
    if kv_session is not None:
        require(live.pooled and all(s.transcript is not None for s in live.sessions),
                "a kv-session frontend needs the slot pool and one transcript per slot")
        live.kv = kv_session
    if chat_text is not None:
        from .chat_text_protocol import chat
        require(kv_session is None and live.pooled and all(chat(s.transcript) for s in live.sessions),
                "a chat-text frontend needs the slot pool and one chat transcript per slot")
        live.chat = chat_text
    existing = getattr(serving, "_remote_ra_live_four_policy", None)
    require(existing is None or existing is live,
            "four-policy frontend attachment collision")
    if admission_overlap > 1:
        # The ACK wait and the engine lease pinned from K (admission_overlap).
        from .admission_overlap import attach_frontend
        attach_frontend(controller, admission_overlap)
    return live
