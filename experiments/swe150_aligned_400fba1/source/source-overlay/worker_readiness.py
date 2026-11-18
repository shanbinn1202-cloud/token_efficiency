"""UniProc capability proof, not a proof of a model forward or GPU drain.

The engine must authenticate the loaded endpoint/executor via source inventory.
This protocol supplies fresh epoch/nonce continuity on that private channel; its
nonce does NOT authenticate an otherwise untrusted RPC transport.
"""
from dataclasses import asdict
from copy import deepcopy
import secrets
from startup_runtime import require


def worker_probe(worker, request):
    from policy_runtime.goal_contract_v13 import ProducerProvenance
    from policy_runtime.worker_transport import (
        RuntimeCaptureBatch, RuntimeWorkerOutbox, attach_runner_publications,
    )
    from vllm.v1.outputs import ModelRunnerOutput

    require(type(request) is dict and set(request) == {
        "channel_epoch", "coordinator_epoch", "challenge", "provenance", "projection_seed"},
        "worker probe schema mismatch")
    for name in ("channel_epoch", "coordinator_epoch", "challenge"):
        require(type(request[name]) is str and 16 <= len(request[name]) <= 128,
                "worker epoch/challenge missing")
    require(type(request["projection_seed"]) is int and request["projection_seed"] >= 0,
            "invalid projection seed")
    runner = worker.model_runner
    parallel = runner.parallel_config
    require(parallel.tensor_parallel_size == parallel.pipeline_parallel_size == 1,
            "single native worker required")
    provenance = ProducerProvenance(**request["provenance"])
    model = runner.model_config
    revision = (getattr(model.hf_text_config, "_commit_hash", None)
                or getattr(model, "revision", None))
    require(revision not in (None, "", "local", "main", "master")
            and str(revision) == provenance.revision and str(model.model) == provenance.model,
            "worker immutable model identity differs")
    require(provenance.hidden_layer == model.hf_text_config.num_hidden_layers - 1,
            "worker capture layer differs")
    sink = getattr(runner, "_policy_runtime_publication_sink", None)
    require(type(sink) is RuntimeWorkerOutbox, "canonical runtime outbox unavailable")
    sink.assert_empty()
    identity = {k: request[k] for k in ("channel_epoch", "coordinator_epoch", "provenance", "projection_seed")}
    state = getattr(worker, "_remote_ra_worker_readiness", None)
    if state is None:
        state = {"identity": deepcopy(identity), "runner": runner, "sink": sink,
                 "nonce": secrets.token_hex(32)}
        worker._remote_ra_worker_readiness = state
    require(state["identity"] == identity and state["runner"] is runner and state["sink"] is sink,
            "worker epoch, runner, outbox, or provenance changed")
    # Empty control roundtrip, using actual typed transport and actual outbox.
    # Never fabricate a model vector/publication and never call a GPU forward.
    batch = RuntimeCaptureBatch(asdict(provenance), request["projection_seed"], ())
    require(RuntimeCaptureBatch.from_wire(batch.to_wire()) == batch, "capture schema roundtrip failed")
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    attach_runner_publications(runner, output)
    require(output.policy_runtime_publications == (), "unexpected startup publications")
    hook = getattr(runner, "_policy_runtime_capture_hook", None)
    if hook is not None:
        require(hook.provenance == provenance, "existing capture provenance differs")
    return dict(channel_epoch=request["channel_epoch"], coordinator_epoch=request["coordinator_epoch"],
                challenge=request["challenge"], worker_nonce=state["nonce"],
                provenance_hash=provenance.provenance_hash, transport_capable=True,
                physical_capture_verified=False, gpu_drain_verified=False)


class WorkerReadiness:
    """Readiness callback for EngineAssembly; rechecks the same actual RPC owner."""
    def __init__(self, *, core, channel_epoch, coordinator_epoch, provenance, projection_seed,
                 verify_source):
        self.core = core
        self.executor = core.model_executor
        self.rpc = core.collective_rpc
        require(getattr(self.rpc, "__self__", None) is core and callable(verify_source),
                "native engine RPC and loaded source verifier required")
        self.verify_source = verify_source
        self.request = dict(channel_epoch=channel_epoch, coordinator_epoch=coordinator_epoch,
                            provenance=asdict(provenance), projection_seed=projection_seed)
        self.provenance_hash = provenance.provenance_hash
        self.worker_nonce = None

    def __call__(self):
        require(self.core.model_executor is self.executor and self.core.collective_rpc == self.rpc,
                "native worker RPC owner changed")
        require(self.verify_source() is True, "loaded worker/runner/RPC hooks unverified")
        challenge = secrets.token_hex(32)
        replies = self.rpc("remote_ra_worker_readiness", timeout=10.0,
                           args=({**self.request, "challenge": challenge},))
        require(type(replies) is list and len(replies) == 1, "exactly one worker response required")
        reply = replies[0]
        require(type(reply) is dict and set(reply) == {"channel_epoch", "coordinator_epoch",
            "challenge", "worker_nonce", "provenance_hash", "transport_capable",
            "physical_capture_verified", "gpu_drain_verified"}, "unexpected worker response")
        require(reply["challenge"] == challenge
                and reply["channel_epoch"] == self.request["channel_epoch"]
                and reply["coordinator_epoch"] == self.request["coordinator_epoch"]
                and reply["provenance_hash"] == self.provenance_hash
                and reply["transport_capable"] is True
                and reply["physical_capture_verified"] is False
                and reply["gpu_drain_verified"] is False, "worker capability proof mismatch")
        nonce = reply["worker_nonce"]
        require(type(nonce) is str and len(nonce) == 64, "worker nonce missing")
        require(self.worker_nonce is None or nonce == self.worker_nonce, "worker restarted during epoch")
        self.worker_nonce = nonce
        return True
