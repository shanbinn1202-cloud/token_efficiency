"""Development glue for copied-source hooks. No bootstrap or live enabling API.

Imports the separately reviewed local protocol. Deployment packaging, renderer,
worker completion and private-channel provisioning remain missing gates.
"""
import asyncio
from hashlib import sha256

from dynamic_runtime import require
from registered_add_protocol import (ADMISSION_SECONDS, LEASE_SECONDS, Ack, Cancel,
                                     RegisteredAddClient, RegisteredAddDispatcher)
from request_path import EngineThreadBoundary, observe_assigned_id
from registered_add_timing import timed


class FrontendBridge:
    # Multi-session stage 2: a completion's Cancel can wait behind another
    # session's admission on the engine thread, so it gets 60 s, not 5 s. An
    # admission's ACK can wait for its parked goal_query Goal (goal_async), so
    # it gets up to ADMISSION_SECONDS (300 s), not 60 s. --admission-overlap K
    # (remote_ra_live_v2.admission_overlap.attach_frontend): the ACK wait and the
    # frame's engine lease become the pair pinned from K, once, at startup.
    def __init__(self, *, codec, encoder, resolve_source, verify_identity, timeout=60,
                 admission_timeout=ADMISSION_SECONDS):
        require(callable(resolve_source) and callable(verify_identity), "trusted frontend callbacks required")
        require(type(timeout) in (int, float) and 0 < timeout <= 120, "bounded reply timeout required")
        require(type(admission_timeout) in (int, float) and 0 < admission_timeout <= ADMISSION_SECONDS,
                "bounded admission timeout required")
        self.codec, self.encoder = codec, encoder
        self.resolve_source, self.verify_identity = resolve_source, verify_identity
        self.timeout = timeout
        self.admission_timeout = admission_timeout
        self.lease_seconds = LEASE_SECONDS

    def pin_admission(self, admission_timeout, lease_seconds):
        """Frontend startup, --admission-overlap K: the ACK wait and the engine lease of
        every registered add, exactly a pair pinned from K (admission_flags.timeouts)."""
        from remote_ra_live_v2.admission_flags import allowed_timeouts
        require(type(admission_timeout) is int and type(lease_seconds) is int and
                (admission_timeout, lease_seconds) in allowed_timeouts(),
                "admission timeout and lease must be a pair pinned from --admission-overlap")
        self.admission_timeout, self.lease_seconds = admission_timeout, lease_seconds

    @property
    def control_timeout(self):
        # With overlapping admissions, another session's bounded phase-1 work
        # can hold the engine thread longer than the ordinary 60-second RPC.
        return (max(self.timeout, self.admission_timeout)
                if self.admission_timeout > ADMISSION_SECONDS else self.timeout)

    @staticmethod
    def handles(request):
        params = getattr(request, "sampling_params", None)
        extra = getattr(params, "extra_args", None) or {}
        tag = (extra.get("kv_transfer_params") or {}).get("policy_runtime")
        return type(tag) is dict and tag.get("policy") in ("full", "random50", "goal_query", "observation_query", "random_fill")

    def validate_before_output(self, request, parent_req):
        require(parent_req is None, "registered-add child fanout is unsupported")
        require(self.verify_identity() is True, "frontend loaded identity is not verified")
        observe_assigned_id(request)

    def _serialize(self, request):
        frames = self.encoder.encode(request)
        require(len(frames) == 1, "registered-add only supports one text core frame")
        return bytes(frames[0])

    def _ack(self, frame, command_id, rid, states):
        ack = self.codec.decode(frame)
        require(type(ack) is Ack and ack.command_id == command_id and ack.request_id == rid,
                "registered-add acknowledgement correlation mismatch")
        if ack.state == "rejected" and "accepted" in states:
            from remote_ra_live_v2.capacity import CapacityWait, refused
            if refused(ack.error):
                raise CapacityWait(ack.error)
        require(ack.state in states, "registered-add rejected: " + ack.error)
        return ack

    async def close_session(self, client, session_id):
        """Session pool: one authenticated CloseSession frame for ``session_id``;
        returns the engine's close receipt. Retrying the same session sends the
        same command, which returns the first close's receipt."""
        from registered_add_protocol import CLOSE_PREFIX, CloseSession
        require(self.verify_identity() is True, "frontend loaded identity is not verified")
        require(client.client_index == 0 and client.client_count == 1, "one frontend is required")
        command_id = "close-" + sha256(session_id.encode()).hexdigest()
        rid = CLOSE_PREFIX + session_id
        frame = self.codec.encode(CloseSession(command_id, rid, session_id))
        with timed("frontend_session_close", request_sha256=command_id[6:], timeout_seconds=self.control_timeout):
            reply = await client.registered_add_async(frame, self.control_timeout)
        ack = self.codec.decode(reply)
        require(type(ack) is Ack and ack.command_id == command_id and ack.request_id == rid,
                "session close acknowledgement correlation mismatch")
        require(ack.state == "complete", "session close rejected: " + ack.error)
        return ack.receipt

    async def submit(self, request, client):
        self.validate_before_output(request, None)
        require(client.client_index == 0 and client.client_count == 1,
                "one frontend is required")
        request.client_index = 0
        assignment = observe_assigned_id(request)
        command_id = "add-" + sha256(assignment.internal_id.encode()).hexdigest()
        # This resolver is trusted server state, not sampling_params/HTTP extras.
        registration = self.resolve_source(request)
        require(type(registration) is dict and set(registration) == {"session_id", "source", "metadata"},
                "exact trusted registration source required")
        preparer = RegisteredAddClient(codec=self.codec, exchange=lambda *_: None)
        frame = preparer.prepare(command_id=command_id, core_request=request,
            serialize_core=self._serialize, lease_seconds=self.lease_seconds, **registration)
        try:
            # Rendering and semantic verification execute on the engine owner
            # thread before ACK, and a goal_query admission's Goal is built
            # (parked) before it. Give that work a separate bounded budget;
            # cancellation retains the short existing timeout.
            with timed("frontend_admission", request_sha256=command_id[4:],
                       timeout_seconds=self.admission_timeout):
                reply = await client.registered_add_async(frame, self.admission_timeout)
            require(self.verify_identity() is True, "frontend identity changed during admission")
            return self._ack(reply, command_id, assignment.internal_id, {"accepted"})
        except BaseException as original:
            from remote_ra_live_v2.capacity import CapacityWait
            if type(original) is CapacityWait:
                # The engine's authenticated CAPACITY_WAIT refusal precedes every
                # engine record of this ID: nothing to cancel, and a Cancel would
                # only spend a request tombstone.
                raise
            # Sending or awaiting admission may already have queued the request.
            # Cancel the SAME ID, shield compensation from the initial cancellation.
            cancel_id = "cancel-" + sha256(assignment.internal_id.encode()).hexdigest()
            cancel = self.codec.encode(Cancel(cancel_id, assignment.internal_id))
            try:
                with timed("frontend_compensation", request_sha256=command_id[4:],
                           timeout_seconds=self.timeout):
                    reply = await asyncio.shield(client.registered_add_async(cancel, self.control_timeout))
                self._ack(reply, cancel_id, assignment.internal_id, {"cancelled", "complete"})
            except BaseException as cleanup:
                raise BaseExceptionGroup("registered-add submission and cancellation failed",
                                         [original, cleanup]) from None
            raise


def build_engine_dispatcher(*, engine, provider, codec, core_type, decoder_type,
                            verify_identity, lifecycle_ready):
    """Build only; caller owns real lifecycle installation on the engine thread.

    Tests use explicit synthetic readiness. This function cannot establish that
    worker completion/expiry/disconnect hooks are installed in a live server.
    """
    require(callable(lifecycle_ready) and lifecycle_ready() is True,
            "renderer/completion/cancel/expiry/bootstrap readiness required")
    decoder = decoder_type(core_type)

    def preprocess(core):
        require(core.client_index == 0 and core.current_wave == 0 and
                core.data_parallel_rank in (None, 0), "single frontend/engine wave required")
        request, wave = engine.preprocess_add_request(core)
        require(wave == 0, "unexpected engine wave")
        return request

    boundary = EngineThreadBoundary(provider=provider,
        enqueue=lambda request: engine.add_request(request, 0),
        abort_enqueued=lambda rid: engine.abort_requests([rid]),
        verify_source_identity=verify_identity)
    return RegisteredAddDispatcher(codec=codec, boundary=boundary,
        decode_core=lambda raw: decoder.decode([raw]), preprocess=preprocess)
