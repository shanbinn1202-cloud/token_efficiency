"""Internal scheduler/worker transport types; never OpenAI request fields.

The executor must carry RuntimeCaptureBatch with SchedulerOutput and return
WorkerPublicationEnvelope with its worker result. The scheduler alone owns
the publication receiver and ticket authority. No process-global singleton,
HTTP endpoint or shared-file publication is used by this module.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy


@dataclass(frozen=True)
class CaptureDescriptor:
    request_id: str
    ticket: str
    expected_rows: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeCaptureBatch:
    provenance: dict
    projection_seed: int
    descriptors: tuple[CaptureDescriptor, ...]

    def to_wire(self):
        return asdict(self)

    @classmethod
    def from_wire(cls, value):
        if set(value) != {"provenance", "projection_seed", "descriptors"}:
            raise ValueError("unexpected runtime capture transport fields")
        return cls(dict(value["provenance"]), value["projection_seed"], tuple(
            CaptureDescriptor(item["request_id"], item["ticket"], tuple(item["expected_rows"]))
            for item in value["descriptors"]))


@dataclass(frozen=True)
class WorkerPublicationEnvelope:
    publications: tuple[dict, ...]

    def deliver(self, coordinator):
        # Called by scheduler-side result consumption, never the GPU worker.
        return tuple(coordinator.publish_worker(**publication) for publication in self.publications)


class RuntimeWorkerOutbox:
    """Worker-result adapter; executor drains this into its result envelope."""

    def __init__(self):
        self._publications = []

    def __call__(self, publications):
        self._publications.extend(deepcopy(publications))

    def assert_empty(self):
        if self._publications:
            raise RuntimeError("undrained policy runtime publication from previous worker step")

    def drain(self):
        result = WorkerPublicationEnvelope(tuple(deepcopy(self._publications)))
        self._publications.clear()
        return result


def initialize_runner_transport(runner):
    """Called only inside the runtime feature branch of runner construction."""
    if runner.parallel_config.tensor_parallel_size != 1 or runner.parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError("policy runtime worker transport requires TP=1 and PP=1")
    runner._policy_runtime_publication_sink = RuntimeWorkerOutbox()


def prepare_runner_step(runner, scheduler_output):
    outbox = getattr(runner, "_policy_runtime_publication_sink", None)
    if not isinstance(outbox, RuntimeWorkerOutbox):
        raise RuntimeError("policy runtime worker transport was not initialized")
    outbox.assert_empty()
    hook = getattr(runner, "_policy_runtime_capture_hook", None)
    if hook is not None:
        hook.release_requests(set(scheduler_output.finished_req_ids or ()) |
                              set(scheduler_output.preempted_req_ids or ()))
    batch = getattr(scheduler_output, "policy_runtime_capture", None)
    if batch is not None and batch.descriptors:
        if (not scheduler_output.total_num_scheduled_tokens or
                getattr(runner, "is_pooling_model", False)):
            raise RuntimeError("policy capture requires a scheduled generation-model forward")


def attach_runner_publications(runner, output):
    """Before sync return or async wrapping: transfer ownership to this output."""
    outbox = getattr(runner, "_policy_runtime_publication_sink", None)
    if not isinstance(outbox, RuntimeWorkerOutbox):
        raise RuntimeError("policy runtime worker transport was not initialized")
    if not hasattr(output, "policy_runtime_publications"):
        raise RuntimeError("ModelRunnerOutput policy transport schema is not installed")
    if output.policy_runtime_publications:
        raise RuntimeError("worker output already owns policy publications")
    output.policy_runtime_publications = outbox.drain().publications
