"""OFFLINE CANDIDATE: private UniProc registration, never request authority.

The caller must already own a quiescent, source-authenticated engine. This file
does not install itself or relax the composed candidate's NOT_INSTALLABLE gate.
An ACK is a capability receipt, NOT a publication/completion/GPU-drain receipt.
"""
from copy import deepcopy
from dataclasses import asdict
import secrets
import threading


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class WorkerGate:
    def __init__(self, executor, verify_loaded, request):
        from policy_runtime.worker_transport import RuntimeWorkerOutbox

        self.executor = executor
        self.wrapper = executor.driver_worker
        self.worker = self.wrapper.worker
        self.runner = self.worker.model_runner
        self.sink = self.runner._policy_runtime_publication_sink
        require(type(self.sink) is RuntimeWorkerOutbox, "canonical outbox required")
        self.thread = threading.get_ident()
        self.parallel = executor.parallel_config
        self.runner_parallel = self.runner.parallel_config
        self.scheduling = executor.scheduler_config
        self.verify_loaded = verify_loaded
        self.rpc = executor.collective_rpc
        self.execute = self.worker.execute_model
        self.probe = self.worker.remote_ra_worker_readiness
        self.request = deepcopy(request)
        self.phase = "ARMED"
        self.state = None
        self.receipt = None

    def _owners(self, *, admission=True):
        require(threading.get_ident() == self.thread, "owner thread changed")
        require(self.executor.driver_worker is self.wrapper
                and self.wrapper.worker is self.worker
                and self.worker.model_runner is self.runner
                and self.runner._policy_runtime_publication_sink is self.sink
                and self.executor.collective_rpc == self.rpc
                and self.worker.execute_model == self.execute
                and self.worker.remote_ra_worker_readiness == self.probe
                and self.executor.parallel_config is self.parallel
                and self.runner.parallel_config is self.runner_parallel
                and self.executor.scheduler_config is self.scheduling
                and self.wrapper.__dict__.get("_remote_ra_dynamic_gate") is self,
                "native worker ownership changed")
        if not admission:return
        require(self.verify_loaded() is True, "loaded source identity unverified")
        parallel = self.runner.parallel_config
        require(parallel.tensor_parallel_size == parallel.pipeline_parallel_size == 1
                and self.parallel.tensor_parallel_size == self.parallel.pipeline_parallel_size == 1
                and getattr(self.parallel, "data_parallel_size", 1) == 1
                and getattr(self.parallel, "worker_extension_cls", None) in (None, "")
                and (not self.scheduling.async_scheduling or
                     getattr(self.runner, "_remote_ra_async_ready", False)),
                "TP=PP=1 required")

    def register(self, request):
        """Invoked through the real wrapper string-RPC endpoint exactly once."""
        try:
            self._owners()
            require(self.phase == "ARMED", "registration is single use")
            require(type(request) is dict and request == self.request,
                    "registration request differs from engine-owned request")
            receipt = self.probe(deepcopy(request))
            self.state = self.worker._remote_ra_worker_readiness
            self.receipt = deepcopy(receipt)
            self.phase = "PROBED"
            return receipt
        except BaseException:
            self.phase = "FAILED"
            raise

    def acknowledge(self, replies):
        from policy_runtime.goal_contract_v13 import ProducerProvenance

        self._owners()
        require(self.phase == "PROBED", "worker registration did not execute")
        require(type(replies) is list and len(replies) == 1
                and type(replies[0]) is dict, "one native worker ACK required")
        reply = replies[0]
        require(set(reply) == {"channel_epoch", "coordinator_epoch", "challenge",
                "worker_nonce", "provenance_hash", "transport_capable",
                "physical_capture_verified", "gpu_drain_verified"}, "ACK schema differs")
        require(reply == self.receipt, "ACK differs from worker-owned receipt")
        for name in ("channel_epoch", "coordinator_epoch", "challenge"):
            require(reply[name] == self.request[name], "stale worker ACK")
        require(reply["provenance_hash"] == ProducerProvenance(
                    **self.request["provenance"]).provenance_hash
                and reply["transport_capable"] is True
                and reply["physical_capture_verified"] is False
                and reply["gpu_drain_verified"] is False, "invalid capability ACK")
        nonce = reply["worker_nonce"]
        require(type(nonce) is str and len(nonce) == 64
                and all(c in "0123456789abcdef" for c in nonce), "invalid worker nonce")
        self.phase = "READY"

    def before_execute(self, scheduler_output):
        from policy_runtime.worker_transport import RuntimeCaptureBatch, CaptureDescriptor

        try:
            self._owners()
            require(self.phase == "READY", "dynamic worker is not registered")
            state = self.worker._remote_ra_worker_readiness
            identity = {name: self.request[name] for name in (
                "channel_epoch", "coordinator_epoch", "provenance", "projection_seed")}
            require(state is self.state and state["identity"] == identity
                    and state["runner"] is self.runner and state["sink"] is self.sink
                    and state["nonce"] == self.receipt["worker_nonce"],
                    "worker registration identity changed")
            self.sink.assert_empty()
            batch = getattr(scheduler_output, "policy_runtime_capture", None)
            # Ordinary/no-capture steps still check registration and ownership.
            if batch is None:
                return
            require(type(batch) is RuntimeCaptureBatch, "canonical capture batch required")
            require(batch.provenance == self.request["provenance"]
                    and type(batch.projection_seed) is int
                    and batch.projection_seed == self.request["projection_seed"],
                    "capture provenance or projection seed differs")
            require(type(batch.descriptors) is tuple, "capture descriptors must be tuple")
            seen_requests, seen_tickets = set(), set()
            for descriptor in batch.descriptors:
                require(type(descriptor) is CaptureDescriptor, "canonical descriptor required")
                rid, ticket, rows = descriptor.request_id, descriptor.ticket, descriptor.expected_rows
                require(type(rid) is str and rid and type(ticket) is str and ticket,
                        "empty capture request/ticket")
                require(rid not in seen_requests and ticket not in seen_tickets,
                        "duplicate capture request/ticket")
                require(type(rows) is tuple and rows
                        and all(type(row) is int and row >= 0 for row in rows)
                        and rows == tuple(sorted(set(rows))), "invalid capture row indices")
                require(type(scheduler_output.num_scheduled_tokens.get(rid)) is int
                        and scheduler_output.num_scheduled_tokens[rid] > 0,
                        "capture request is not scheduled")
                seen_requests.add(rid)
                seen_tickets.add(ticket)
            if batch.descriptors:
                require(scheduler_output.total_num_scheduled_tokens > 0
                        and not getattr(self.runner, "is_pooling_model", False),
                        "capture requires generation forward")
            # Tickets remain coordinator-owned. Do not mint, authenticate, publish,
            # drain or mark successful here; the real runner/scheduler do that.
        except BaseException:
            self.phase = "FAILED"
            raise

    def close(self):
        from remote_ra_http.shutdown import once
        def cleanup():
            self.phase='CLOSING'
            try:
                self._owners(admission=False)
                self.sink.assert_empty()
            except BaseException:
                self.phase='FAILED';raise
            self.phase='CLOSED'
        return once(self,'_cleanup_owner',cleanup)


def register_uniproc(*, executor, channel_epoch, coordinator_epoch, provenance,
                     projection_seed, verify_loaded, enabled=False):
    """Offline assembly seam; no environment variable/HTTP field enables it.

    A failed registration leaves a sticky FAILED gate, never static fallback.
    Native UniProc ignores timeout; this synchronous call promises no deadline.
    """
    require(type(enabled) is bool, "explicit boolean opt-in required")
    if not enabled:
        return None
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from policy_runtime.goal_contract_v13 import ProducerProvenance

    require(type(executor) is UniProcExecutor, "only native UniProcExecutor supported")
    wrapper = executor.driver_worker
    require(type(wrapper) is WorkerWrapperBase, "only native WorkerWrapperBase supported")
    require("_remote_ra_dynamic_gate" not in wrapper.__dict__, "worker already armed")
    require(type(provenance) is ProducerProvenance, "canonical producer provenance required")
    require(callable(verify_loaded) and verify_loaded() is True, "loaded source verifier required")
    require(callable(getattr(type(wrapper), "remote_ra_register_dynamic", None)),
            "native wrapper registration overlay missing")
    parallel = executor.parallel_config
    require(parallel.tensor_parallel_size == parallel.pipeline_parallel_size == 1
            and getattr(parallel, "data_parallel_size", 1) == 1
            and getattr(parallel, "worker_extension_cls", None) in (None, ""),
            "single worker without class extension required")
    if executor.scheduler_config.async_scheduling:
        from remote_ra_live_v2.async_support import attach
        attach(wrapper.worker.model_runner)
    request = dict(channel_epoch=channel_epoch, coordinator_epoch=coordinator_epoch,
                   challenge=secrets.token_hex(32), provenance=asdict(provenance),
                   projection_seed=projection_seed)
    gate = WorkerGate(executor, verify_loaded, request)
    wrapper._remote_ra_dynamic_gate = gate
    try:
        # Native string dispatch only; no pickled callback or non-block future.
        replies = executor.collective_rpc("remote_ra_register_dynamic", args=(deepcopy(request),))
        gate.acknowledge(replies)
        return gate
    except BaseException:
        gate.phase = "FAILED"
        raise
