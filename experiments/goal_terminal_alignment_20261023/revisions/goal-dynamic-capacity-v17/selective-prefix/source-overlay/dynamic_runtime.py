"""Trusted in-process registration prototype; deliberately no HTTP endpoint.

All entry points except __call__ are scheduler/control-plane only. Python object
access is not a security boundary. A future IPC server must authenticate callers,
marshal commands onto the scheduler thread, and allocate actual engine IDs there.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import secrets
import time

from contracts import ManifestContextProvider, PolicyRuntimeAbort, canonical_prompt_hash, GoalRuntimeAdapter
from two_span_observation import TwoSpanObservationRuntime


def detached(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def source_json(value):
    """Executable rendering source: object insertion order is significant.

    Keep metadata/receipt canonicalization on detached()/digest(). Only rendering
    source uses this encoding; nested tool argument order changes prompt tokens.
    """
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def detached_source(value):
    return json.loads(source_json(value))


def source_digest(value):
    return sha256(source_json(value).encode("utf-8")).hexdigest()


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise PolicyRuntimeAbort(message)


@dataclass(frozen=True)
class RenderedPrompt:
    """Output of configured server renderer, NOT a client receipt schema.

    Span includes the latest tool message and its actual template suffix. The
    renderer must derive it from the source/template, never infer it by flooring
    a token count. Token equality at admission proves only equality to this output;
    it does not independently prove renderer semantics or tokenizer provenance.
    """
    token_ids: tuple[int, ...]
    observation_start: int | None = None
    observation_end: int | None = None


@dataclass(frozen=True)
class ObservationGeometry:
    semantic_start: int
    semantic_end: int
    recompute_start: int
    historical_pages: int
    overlapping_logical_pages: tuple[int, ...]
    baseline_runtime_supported: bool


class IdempotentGoalRuntimeAdapter(GoalRuntimeAdapter):
    """Local fix for release before the first authenticated Goal publication.

    The baseline coordinator releases every adapter on generation invalidation;
    the baseline Goal adapter assumes its session has already been opened.
    """
    def release_session(self, session_id, generation):
        with self._lock:
            if self._generations.get(session_id) != generation:
                return
            super().release_session(session_id, generation)


def observation_geometry(start, end, prompt_length, page_tokens):
    require(all(type(x) is int for x in (start, end, prompt_length, page_tokens))
            and page_tokens > 0 and 0 <= start < end == prompt_length,
            "observation must be an exact nonempty rendered suffix")
    boundary = start // page_tokens * page_tokens
    return ObservationGeometry(start, end, boundary, boundary // page_tokens,
        tuple(range(start // page_tokens, (end - 1) // page_tokens + 1)), start == boundary)


class DynamicContextProvider(ManifestContextProvider):
    """Bounded, linearized dynamic stages reusing real baseline worker tickets.

    One frozen-history cycle per session. New turns need new sessions and a full
    cold capture: carrying page vectors/rankings into growing histories is NOT
    implemented. This provider cannot be installed using the old bootstrap alone;
    the scheduler must wire explicit complete/cancel notifications.
    """
    def __init__(self, *, provenance, selected_count, projection_seed, coordinator,
                 renderer, renderer_identity, ttl_seconds=120, clock=time.monotonic,
                 max_sessions=128, max_registrations=1024):
        require(callable(renderer), "configured server renderer required")
        require(isinstance(renderer_identity, str) and len(renderer_identity) == 64
                and all(c in "0123456789abcdef" for c in renderer_identity),
                "renderer code/config identity must be SHA256")
        require(type(ttl_seconds) in (int, float) and math.isfinite(ttl_seconds)
                and ttl_seconds > 0, "finite positive TTL required")
        require(all(type(x) is int and x > 0 for x in (max_sessions, max_registrations)),
                "positive capacity limits required")
        goal_adapter = coordinator._adapters.get("goal_query")
        require(goal_adapter is None or isinstance(goal_adapter, IdempotentGoalRuntimeAdapter),
                "dynamic lifecycle requires the idempotent development Goal adapter")
        super().__init__(dict(version=1, provenance=asdict(provenance),
            selected_count=selected_count, projection_seed=projection_seed, requests=[]), coordinator)
        coordinator._remote_ra_selector_required = True
        self._render = renderer
        self.renderer_identity = renderer_identity
        self._clock, self._ttl = clock, ttl_seconds
        self._max_sessions, self._max_registrations = max_sessions, max_registrations
        self._sessions, self._entries, self._bindings = {}, {}, {}
        # Tombstones are bounded by lifetime capacity, never silently evicted.
        # Restart creates a new authority; stale IDs cannot obtain new authority.
        self.events = []

    def _event(self, name, **fields):
        self.events.append(detached(dict(event=name, **fields)))

    def open_session(self, *, session_id, policy, num_pages):
        with self.coordinator.lock:
            require(type(session_id) is str and 0 < len(session_id) <= 128,
                    "bounded session ID required")
            require(policy in ("full", "random50", "goal_query", "observation_query"), "unsupported policy")
            require(type(num_pages) is int and num_pages > self.selected_count,
                    "nonselective or invalid historical page count")
            require(session_id not in self._sessions, "session ID cannot be reused")
            require(len(self._sessions) < self._max_sessions, "session capacity reached; restart authority")
            generation = secrets.token_hex(32)
            self._sessions[session_id] = dict(policy=policy, num_pages=num_pages,
                generation=generation, next_stage=0, active=None, state="open")
            return generation

    def register(self, *, session_id, request_id, source, metadata=None):
        """Trusted dispatcher registers the NEXT stage against its actual engine ID.

        Same ID + identical inputs is an idempotent control-plane retry while live.
        No client-selected stage, turn, generation, nonce, span or token hash exists.
        The configured renderer is invoked here; source and output are detached.
        """
        from policy_runtime.tagged_selector import require_private_admission
        require_private_admission(self.coordinator)
        source, metadata = detached_source(source), detached({} if metadata is None else metadata)
        require(type(source) is dict and type(metadata) is dict, "source and metadata must be objects")
        fingerprint = digest(dict(session_id=session_id, source_sha256=source_digest(source), metadata=metadata))
        with self.coordinator.lock:
            self.sweep_expired()
            require(type(request_id) is str and 0 < len(request_id) <= 256, "engine request ID required")
            old = self._entries.get(request_id)
            if old:
                require(old["state"] in ("registered", "admitted") and old["fingerprint"] == fingerprint,
                        "conflicting or terminal registration retry")
                return detached(old["receipt"])
            require(len(self._entries) < self._max_registrations,
                    "registration capacity reached; restart authority")
            state = self._sessions.get(session_id)
            require(state is not None and state["state"] == "open", "session is not open")
            require(state["active"] is None, "previous stage has not completed")
            stages = (("capture", "query", "restore") if state["policy"] == "goal_query"
                      else ("capture", "observation", "generation"))
            require(state["next_stage"] < len(stages), "session cycle is complete")
            stage = stages[state["next_stage"]]
            from registered_add_timing import timed
            with timed("engine_render", request_sha256=sha256(request_id.encode()).hexdigest()):
                rendered = self._render(detached_source(source))
            require(isinstance(rendered, RenderedPrompt), "server renderer returned no token binding")
            tokens = list(rendered.token_ids)
            prompt_hash = canonical_prompt_hash(tokens)
            require(bool(tokens), "empty rendered prompt")
            n = 0 if stage == "query" else state["num_pages"]
            require(len(tokens) > n * self.provenance.page_tokens, "prompt needs a nonempty suffix")
            if stage != "capture" and stage != "query":
                require(tokens[:n * self.provenance.page_tokens] == state["history_tokens"],
                        "registered prompt changed frozen historical tokens")
            if state["policy"] == "goal_query" and stage == "query":
                expected_query_hash = sha256(json.dumps(dict(prompt_token_ids=tokens, add_special_tokens=False),
                    sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                require(metadata.get("query_payload_hash") == expected_query_hash,
                        "Goal query payload hash differs from actual rendered probe tokens")
            if state["policy"] == "goal_query" and stage == "restore":
                require(all(metadata.get(k) == state["goal_lineage"][k]
                    for k in ("goal_hash", "query_payload_hash")), "Goal restore lineage changed")
            geometry = None
            if state["policy"] == "observation_query" and stage in ("observation", "generation"):
                geometry = observation_geometry(rendered.observation_start, rendered.observation_end,
                                                len(tokens), self.provenance.page_tokens)
                require(geometry.baseline_runtime_supported or isinstance(
                    self.coordinator._adapters.get("observation_query"), TwoSpanObservationRuntime),
                    "unaligned semantic observation requires development two-span adapter")
                require(geometry.historical_pages == n, "observation changed historical page universe")
                if stage == "generation":
                    require(asdict(geometry) == state["observation_geometry"],
                            "generation renderer changed sealed observation spans")
            turns = (1, 2, 3) if state["policy"] == "goal_query" else (1, 2, 2)
            entry = dict(policy=state["policy"], stage=stage, session_id=session_id,
                generation=state["generation"], turn=turns[state["next_stage"]],
                prompt_hash=prompt_hash, num_pages=n, metadata=metadata)
            # Exact existing per-stage schemas and provenance validation.
            child = ManifestContextProvider(dict(version=1, provenance=asdict(self.provenance),
                selected_count=self.selected_count, projection_seed=self.projection_seed,
                requests=[entry]), self.coordinator)
            receipt = dict(request_id=request_id, source_sha256=source_digest(source),
                renderer_identity_sha256=self.renderer_identity, prompt_sha256=prompt_hash,
                token_binding="CONFIGURED_RENDERER_OUTPUT_SCHEDULER_EQUALITY_REQUIRED",
                system_valid=False, task_accuracy_valid=False, live_multi_turn_supported=False,
                geometry=asdict(geometry) if geometry else None, stage=stage)
            self._entries[request_id] = dict(fingerprint=fingerprint, receipt=receipt,
                session_id=session_id, policy=state["policy"], tokens=tokens, child=child,
                state="registered", expires=self._clock() + self._ttl)
            state["active"] = request_id
            if stage == "capture":
                state["history_tokens"] = tokens[:n * self.provenance.page_tokens]
            if stage == "query":
                state["goal_lineage"] = {k: metadata[k] for k in ("goal_hash", "query_payload_hash")}
            if stage == "observation":
                state["observation_geometry"] = asdict(geometry)
            self._event("DYNAMIC_REGISTERED", **receipt)
            return detached(receipt)

    def __call__(self, request):
        with self.coordinator.lock:
            self.sweep_expired()
            entry = self._entries.get(request.request_id)
            require(entry is not None and entry["state"] in ("registered", "admitted"),
                    "unregistered, expired or retired engine request ID")
            # Recheck every retry, including client tag mutation after admission.
            try:
                require(request.kv_transfer_params == {"policy_runtime": {"version": 1,
                        "policy": entry["policy"]}}, "only the exact routing tag is allowed")
                require(type(request.kv_transfer_params["policy_runtime"]["version"]) is int,
                        "routing tag version must be integer")
                require(isinstance(request.prompt_token_ids, (list, tuple))
                        and list(request.prompt_token_ids) == entry["tokens"]
                        and all(type(t) is int for t in request.prompt_token_ids),
                        "actual scheduler tokens differ from registered renderer output")
                context = entry["child"](request)
                if (entry["policy"] == "observation_query"
                        and context["stage"] in ("observation", "generation")
                        and isinstance(self.coordinator._adapters.get("observation_query"), TwoSpanObservationRuntime)):
                    # Legacy child owns strict stage schema/nonces/tickets only.
                    # Replace its inferred span with the actual renderer's S and
                    # explicitly provide B; only the two-span adapter accepts it.
                    geometry = entry["receipt"]["geometry"]
                    context["metadata"].update(observation_start=geometry["semantic_start"],
                        observation_end=geometry["semantic_end"], recompute_start=geometry["recompute_start"])
                self._contexts[request.request_id] = detached(context)
                entry["state"] = "admitted"
                return context
            except Exception:
                self.cancel_session(entry["session_id"], "admission_failed")
                raise

    def register_binding(self, binding, handle, context):
        """Exact retry is legal only before commit, while reserve() remains live.

        After commit, allocation deferral must use the unchanged coordinator's
        reuse_committed_plan() snapshot/plan recheck, not reissue capture tickets.
        """
        with self.coordinator.lock:
            entry = self._entries.get(binding.request_id)
            require(entry is not None and entry["state"] == "admitted", "binding has no admission")
            require(self.coordinator.resolve(handle) == binding, "binding is not scheduler reserved")
            expected = self._contexts[binding.request_id]
            require(context == expected and binding.canonical_prompt_hash == expected["prompt_hash"]
                    and all(getattr(binding, k) == expected[k] for k in
                        ("session_id", "generation", "policy", "stage", "turn", "nonce", "num_pages",
                         "model_revision", "producer_provenance"))
                    and binding.page_tokens == self.provenance.page_tokens
                    and detached(binding.metadata) == expected["metadata"]
                    and binding.cumulative_prefix_identities == tuple(canonical_prompt_hash(
                        entry["tokens"][:(i + 1) * binding.page_tokens]) for i in range(binding.num_pages)),
                    "scheduler binding differs from admission")
            old = self._bindings.get(binding.request_id)
            require(old is None or old == (binding, handle), "binding retry changed authority")
            if old is None:
                from policy_runtime.tagged_selector import private_registered_binding
                private_registered_binding(self.coordinator, binding, context)
            super().register_binding(binding, handle, context)
            self._bindings[binding.request_id] = (binding, handle)

    def capture_batch(self, scheduled_request_ids):
        with self.coordinator.lock:
            return super().capture_batch(scheduled_request_ids)

    def complete(self, request_id):
        """Scheduler invokes BEFORE releasing successful final output.

        Baseline coordinator verifies worker captures and Observation recompute.
        A completion callback without an actual registered binding is rejected.
        """
        with self.coordinator.lock:
            entry = self._entries.get(request_id)
            require(entry is not None, "unknown completion")
            if entry["state"] == "complete":
                return
            require(entry["state"] == "admitted" and request_id in self._bindings,
                    "completion lacks a scheduler binding")
            binding, handle = self._bindings[request_id]
            try:
                if binding.stage in ("capture", "query"):
                    require(self.coordinator.resolve(handle) == binding, "capture reservation is not live")
                else:
                    require(self.coordinator.resolve_committed(handle) == binding, "restore plan is not committed")
                self.coordinator.finalize_success(request_id)
                self.coordinator.finish_request(request_id)
            except Exception:
                self.cancel_session(entry["session_id"], "completion_failed")
                raise
            entry["state"] = "complete"
            state = self._sessions[entry["session_id"]]
            state["active"] = None
            state["next_stage"] += 1
            if state["next_stage"] == 3:
                state["state"] = "complete"
            self._drop_request(request_id)
            self._event("DYNAMIC_STAGE_COMPLETE", request_id=request_id, stage=binding.stage)

    def _drop_request(self, request_id):
        super().retire(request_id)
        self._bindings.pop(request_id, None)
        entry = self._entries[request_id]
        entry.pop("child", None)
        entry.pop("tokens", None)

    def retire(self, request_id):
        """Legacy bare retirement cannot assert success; fail closed and cancel."""
        with self.coordinator.lock:
            entry = self._entries.get(request_id)
            if entry and entry["state"] not in ("complete", "cancelled"):
                self.cancel_session(entry["session_id"], "retired_without_completion")

    def cancel_session(self, session_id, reason="cancelled"):
        with self.coordinator.lock:
            state = self._sessions.get(session_id)
            require(state is not None, "unknown session")
            if state["state"] == "cancelled":
                return
            state["state"] = "cancelling"
            cleanup_succeeded = False
            try:
                self.coordinator.invalidate_generation(session_id, state["generation"])
                cleanup_succeeded = True
            finally:
                state["state"] = "cancelled" if cleanup_succeeded else "cleanup_failed"
                for rid, entry in self._entries.items():
                    if entry["session_id"] == session_id:
                        entry["state"] = "cancelled"
                        self._drop_request(rid)
                state["active"] = None
                state.pop("history_tokens", None)
                state.pop("goal_lineage", None)
                state.pop("observation_geometry", None)
                self._event("DYNAMIC_SESSION_CANCELLED" if cleanup_succeeded else "DYNAMIC_CLEANUP_FAILED",
                            session_id=session_id, reason=reason)

    def release_session(self, session_id):
        """Release producer data after completion; keep bounded anti-replay tombstone."""
        with self.coordinator.lock:
            state = self._sessions.get(session_id)
            require(state is not None and state["state"] == "complete", "session is not complete")
            self.coordinator.invalidate_generation(session_id, state["generation"])
            state["state"] = "released"
            state.pop("history_tokens", None)
            state.pop("goal_lineage", None)
            state.pop("observation_geometry", None)

    def sweep_expired(self):
        with self.coordinator.lock:
            expired = {e["session_id"] for e in self._entries.values()
                if e["state"] == "registered" and self._clock() >= e["expires"]}
            for session_id in expired:
                self.cancel_session(session_id, "registration_timeout")
            return len(expired)
