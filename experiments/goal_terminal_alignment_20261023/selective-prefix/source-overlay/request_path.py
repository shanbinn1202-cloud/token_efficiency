"""Executable local request-path contract; no vLLM IPC or renderer implementation.

Public HTTP input must never construct AssignedId or call EngineThreadBoundary.
Those values are trusted in-process call results, not network authentication.
The default transport deliberately refuses execution until actual vLLM plumbing
and a real renderer span producer have been implemented and reviewed.
"""
from dataclasses import dataclass
import threading

from contracts import PolicyRuntimeAbort, canonical_prompt_hash
from dynamic_runtime import detached, detached_source, digest, require, source_digest


LIVE_STATUS = dict(system_valid=False, task_accuracy_valid=False,
                   live_multi_turn_supported=False, cross_process_implemented=False,
                   actual_renderer_implemented=False)


class IntegrationMissing(PolicyRuntimeAbort):
    pass


@dataclass(frozen=True)
class AssignedId:
    """Trusted frontend snapshot AFTER InputProcessor.assign_request_id()."""
    internal_id: str
    external_id: str
    prompt_hash: str


def observe_assigned_id(core_request):
    """Read the actual post-assignment core object, never predict an ID suffix.

    Refuse randomization-disabled IDs for this slice; do not change vLLM env vars.
    Caller must be on the trusted AsyncLLM path. This type is not an IPC proof.
    """
    internal = core_request.request_id
    external = getattr(core_request, "external_req_id", None)
    require(type(internal) is str and type(external) is str and internal and external
            and internal != external, "post-assignment unique engine request ID required")
    require(not getattr(core_request, "mm_features", None)
            and getattr(core_request, "prompt_embeds", None) is None
            and getattr(core_request, "prompt_is_token_ids", None) is None
            and getattr(core_request, "lora_request", None) is None
            and not getattr(core_request, "resumable", False),
            "only text-token nonstreaming non-LoRA input is supported")
    params = getattr(core_request, "sampling_params", None)
    require(params is not None and getattr(params, "n", None) == 1
            and type(getattr(params, "n", None)) is int
            and getattr(core_request, "pooling_params", None) is None,
            "single n=1 generation required before child fanout")
    return AssignedId(internal, external, canonical_prompt_hash(core_request.prompt_token_ids))


class MissingProcessTransport:
    """Safe default: caller cannot accidentally call ordinary ADD as fallback."""
    def submit(self, **_):
        raise IntegrationMissing("trusted API-server to engine registration/ADD IPC is not implemented")


class EngineThreadBoundary:
    """Local contract seam at EngineCore.add_request BEFORE scheduler.add_request.

    Requires a preprocessed scheduler Request plus the actual assigned frontend
    ID snapshot. Construct on the scheduler thread. Enqueue and abort callbacks
    must act on the same real scheduler if this is eventually wired; tests inject
    explicit local fakes. This object does not install hooks or send any IPC.
    """
    def __init__(self, *, provider, enqueue, abort_enqueued, verify_source_identity):
        require(all(callable(f) for f in (enqueue, abort_enqueued, verify_source_identity)),
                "explicit scheduler callbacks and source identity check required")
        self.provider = provider
        self._enqueue, self._abort = enqueue, abort_enqueued
        self._verify_identity = verify_source_identity
        self._thread = threading.get_ident()
        self._submissions = {}

    def _engine_thread(self):
        require(threading.get_ident() == self._thread, "must run on scheduler engine thread")

    def _source_identity(self):
        require(self._verify_identity() is True, "loaded source identity not verified")

    def submit(self, *, assigned, request, session_id, source, metadata=None):
        self._engine_thread()
        try:
            self._source_identity()
        except Exception as original:
            if request.request_id in self._submissions:
                self._raise_after_cleanup(request.request_id, original)
            raise
        require(type(assigned) is AssignedId, "trusted assigned-ID snapshot required")
        require(request.request_id == assigned.internal_id
                and canonical_prompt_hash(request.prompt_token_ids) == assigned.prompt_hash,
                "preprocessed engine request differs from actual frontend assignment")
        require(not getattr(request, "abort_immediately", False), "pre-rejected request cannot register")
        source = detached_source(source)
        metadata = detached({} if metadata is None else metadata)
        fingerprint = digest(dict(assigned=assigned.__dict__, session_id=session_id,
            source_sha256=source_digest(source), metadata=metadata,
            params=getattr(request, "kv_transfer_params", None)))
        rid = request.request_id
        with self.provider.coordinator.lock:
            old = self._submissions.get(rid)
            if old:
                require(old["state"] == "enqueued" and old["fingerprint"] == fingerprint,
                        "conflicting or terminal engine submission retry")
                try:
                    self.provider(request)  # Recheck tokens/tag even on exact retry.
                except Exception as original:
                    # Admission can revoke provider authority on a mutated retry,
                    # but the earlier request may still be in the engine queue.
                    self._raise_after_cleanup(rid, original)
                return detached(old["receipt"])
            receipt = self.provider.register(session_id=session_id, request_id=rid,
                                             source=source, metadata=metadata)
            record = dict(session_id=session_id, fingerprint=fingerprint,
                          receipt=receipt, state="registering")
            self._submissions[rid] = record
            try:
                self.provider(request)  # Actual engine token equality before ADD.
                record["state"] = "enqueue_attempted"
                self._enqueue(request)
                record["state"] = "enqueued"
            except Exception as original:
                # Enqueue may mutate the waiting queue before raising. The abort
                # callback must tolerate an absent ID and drain engine resources.
                self._raise_after_cleanup(rid, original)
            return detached(receipt)

    def _raise_after_cleanup(self, request_id, original):
        try:
            self._cancel_record(request_id)
        except Exception as cleanup:
            raise ExceptionGroup("request verification/execution and cleanup failed",
                                 [original, cleanup]) from None
        raise original

    def _cancel_record(self, request_id):
        record = self._submissions[request_id]
        if record["state"] in ("cancelled", "complete"):
            return
        errors = []
        try:
            self._abort(request_id)
        except Exception as exc:
            errors.append(exc)
        try:
            self.provider.cancel_session(record["session_id"], "engine_request_path_cancelled")
        except Exception as exc:
            errors.append(exc)
        record["state"] = "cleanup_failed" if errors else "cancelled"
        if errors:
            raise ExceptionGroup("engine request-path cleanup failed", errors)

    def cancel(self, request_id):
        self._engine_thread()
        require(request_id in self._submissions, "unknown registered engine ID")
        with self.provider.coordinator.lock:
            self._cancel_record(request_id)

    def finalize_before_output(self, request_id, *, successful):
        """Wire before final output creation/free; never trust an HTTP success."""
        self._engine_thread()
        require(type(successful) is bool, "engine terminal status required")
        record = self._submissions.get(request_id)
        require(record is not None and record["state"] in ("enqueued", "complete"),
                "no live engine submission")
        if not successful:
            self.cancel(request_id)
            return
        if record["state"] == "complete":
            return
        try:
            self._source_identity()
            self.provider.complete(request_id)
            record["state"] = "complete"
        except Exception as original:
            self._raise_after_cleanup(request_id, original)


def submit_vllm_request(*, transport=None, **submission):
    """Explicit transport injection; never route missing integration to Full/ADD."""
    return (transport if transport is not None else MissingProcessTransport()).submit(**submission)
