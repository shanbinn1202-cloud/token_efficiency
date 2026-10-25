"""Pluggable, bounded retrieval-Goal rewrite backends.

The rewrite boundary intentionally carries only the immutable task anchor and a
compact view of the latest turn.  In particular, it has no trajectory field and
does not expose an execution-plan schema to a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import select
import subprocess
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from ..schema import RetrievalGoal


MAX_ACTION_CHARS = 4_000
MAX_OBSERVATION_CHARS = 8_000
MAX_FEATURES_JSON_CHARS = 16_000
MAX_STATS_JSON_CHARS = 4_000
MAX_DESCRIPTION_CHARS = 1_200
MAX_ITEMS_PER_FIELD = 64
MAX_ITEM_CHARS = 512
MAX_CANDIDATE_JSON_CHARS = 32_000

_GOAL_FIELDS = frozenset(RetrievalGoal.__dataclass_fields__)
_TUPLE_FIELDS = (
    "files",
    "symbols",
    "tests",
    "errors",
    "evidence_needed",
    "constraints",
)
_PROTECTED_FIELDS = frozenset({
    "root_goal",
    "task_objective",
    "execution_plan",
    "plan",
    "steps",
    "next_action",
    "actions",
    "commands",
    "reasoning",
})


class RewriteValidationError(ValueError):
    """The SLM returned a structurally unsafe or invalid Goal."""


class RewriteCommandError(RuntimeError):
    """A command-backed rewrite process failed."""

    def __init__(
        self,
        message: str,
        *,
        latency_ms: float = 0.0,
        startup_latency_ms: float = 0.0,
        generation_latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        super().__init__(message)
        self.latency_ms = latency_ms
        self.startup_latency_ms = startup_latency_ms
        self.generation_latency_ms = generation_latency_ms
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def _freeze_json(value: Any, *, path: str = "$", depth: int = 0) -> Any:
    """Validate JSON safety while making nested request data immutable."""

    if depth > 12:
        raise ValueError(f"{path} exceeds the maximum JSON nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string mapping key")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}", depth=depth + 1)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    raise ValueError(f"{path} contains non-JSON value {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


@dataclass(frozen=True)
class RewriteRequest:
    """The complete and deliberately bounded context visible to an SLM."""

    root_goal: str
    current_goal: RetrievalGoal
    latest_action: str
    latest_observation: str
    features: Mapping[str, Any]
    trigger: str
    stats: Mapping[str, Any]
    allow_constraint_updates: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.root_goal, str) or not self.root_goal.strip():
            raise ValueError("root_goal is required")
        if not isinstance(self.current_goal, RetrievalGoal):
            raise TypeError("current_goal must be a RetrievalGoal")
        if not isinstance(self.latest_action, str):
            raise TypeError("latest_action must be a string")
        if not isinstance(self.latest_observation, str):
            raise TypeError("latest_observation must be a string")
        if not isinstance(self.trigger, str) or not self.trigger.strip():
            raise ValueError("trigger is required")
        if not isinstance(self.allow_constraint_updates, bool):
            raise TypeError("allow_constraint_updates must be a bool")

        object.__setattr__(self, "latest_action", self.latest_action[:MAX_ACTION_CHARS])
        object.__setattr__(
            self,
            "latest_observation",
            self.latest_observation[:MAX_OBSERVATION_CHARS],
        )
        frozen_features = _freeze_json(self.features, path="features")
        frozen_stats = _freeze_json(self.stats, path="stats")
        if not isinstance(frozen_features, Mapping) or not isinstance(frozen_stats, Mapping):
            raise TypeError("features and stats must be mappings")
        if _json_size(_thaw_json(frozen_features)) > MAX_FEATURES_JSON_CHARS:
            raise ValueError("features exceed the compact rewrite-request bound")
        if _json_size(_thaw_json(frozen_stats)) > MAX_STATS_JSON_CHARS:
            raise ValueError("stats exceed the compact rewrite-request bound")
        object.__setattr__(self, "features", frozen_features)
        object.__setattr__(self, "stats", frozen_stats)

    def to_payload(self) -> dict[str, Any]:
        """Return the exact JSON-safe model input (never a full trajectory)."""

        return {
            "root_goal": self.root_goal,
            "current_goal": asdict(self.current_goal),
            "latest_action": self.latest_action,
            "latest_observation": self.latest_observation,
            "features": _thaw_json(self.features),
            "trigger": self.trigger,
            "stats": _thaw_json(self.stats),
            "allow_constraint_updates": self.allow_constraint_updates,
        }


@dataclass(frozen=True)
class RewriteResult:
    candidate: RetrievalGoal
    latency_ms: float = 0.0
    startup_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model_cost: float = 0.0
    raw_response: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, RetrievalGoal):
            raise TypeError("candidate must be a validated RetrievalGoal")
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if self.startup_latency_ms < 0:
            raise ValueError("startup_latency_ms cannot be negative")
        if self.generation_latency_ms < 0:
            raise ValueError("generation_latency_ms cannot be negative")
        if isinstance(self.input_tokens, bool) or self.input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")
        if isinstance(self.output_tokens, bool) or self.output_tokens < 0:
            raise ValueError("output_tokens cannot be negative")
        if isinstance(self.model_cost, bool) or self.model_cost < 0:
            raise ValueError("model_cost cannot be negative")
        frozen = _freeze_json(self.raw_response, path="raw_response")
        if not isinstance(frozen, Mapping):
            raise TypeError("raw_response must be a mapping")
        object.__setattr__(self, "raw_response", frozen)

    @property
    def cost(self) -> float:
        """Short alias useful to generic experiment telemetry consumers."""

        return self.model_cost


class GoalRewriter(Protocol):
    """Dependency-free interface implemented by fake and real SLM adapters."""

    def rewrite(self, request: RewriteRequest) -> RewriteResult:
        ...


def _bounded_tuple(name: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RewriteValidationError(f"{name} must be a JSON array of strings")
    if len(value) > MAX_ITEMS_PER_FIELD:
        raise RewriteValidationError(
            f"{name} exceeds the {MAX_ITEMS_PER_FIELD}-item bound"
        )
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise RewriteValidationError(f"{name} must contain non-empty strings")
        if len(item) > MAX_ITEM_CHARS:
            raise RewriteValidationError(
                f"{name} contains an item over {MAX_ITEM_CHARS} characters"
            )
        result.append(item)
    return tuple(result)


def validate_rewrite_candidate(
    candidate: RetrievalGoal | Mapping[str, Any],
    current_goal: RetrievalGoal,
    *,
    trigger: str,
    allow_constraint_updates: bool = False,
) -> RetrievalGoal:
    """Validate and normalize an SLM candidate without ever accepting a root.

    Missing retrieval fields inherit their current values.  Unknown fields are
    rejected instead of ignored, preventing accidental execution-plan or root
    fields from crossing this interface.
    """

    if isinstance(candidate, RetrievalGoal):
        raw: Mapping[str, Any] = asdict(candidate)
    elif isinstance(candidate, Mapping):
        raw = candidate
    else:
        raise RewriteValidationError("rewrite candidate must be a JSON object")

    if any(not isinstance(key, str) for key in raw):
        raise RewriteValidationError("rewrite candidate keys must be strings")
    present = set(raw)
    protected = present & _PROTECTED_FIELDS
    if protected:
        raise RewriteValidationError(
            f"rewrite candidate contains protected fields: {sorted(protected)}"
        )
    unknown = present - _GOAL_FIELDS
    if unknown:
        raise RewriteValidationError(
            f"rewrite candidate contains unknown fields: {sorted(unknown)}"
        )
    try:
        raw_json_size = _json_size(_thaw_json(raw))
    except (TypeError, ValueError) as error:
        raise RewriteValidationError("rewrite candidate is not JSON-safe") from error
    if raw_json_size > MAX_CANDIDATE_JSON_CHARS:
        raise RewriteValidationError("rewrite candidate exceeds the output-size bound")

    values = asdict(current_goal)
    values.update(raw)
    description = values["description"]
    if not isinstance(description, str) or not description.strip():
        raise RewriteValidationError("description must be a non-empty string")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise RewriteValidationError(
            f"description exceeds {MAX_DESCRIPTION_CHARS} characters"
        )
    phase_hint = values["phase_hint"]
    status = values["status"]
    if not isinstance(phase_hint, str) or not isinstance(status, str):
        raise RewriteValidationError("phase_hint and status must be strings")
    for name in _TUPLE_FIELDS:
        values[name] = _bounded_tuple(name, values[name])

    constraints_changed = values["constraints"] != current_goal.constraints
    constraints_allowed = trigger == "user_request" and allow_constraint_updates
    if constraints_changed and not constraints_allowed:
        raise RewriteValidationError(
            "constraints may change only for an authorized user_request rewrite"
        )
    try:
        return RetrievalGoal(**values)
    except (TypeError, ValueError) as error:
        raise RewriteValidationError(str(error)) from error


def _merge_feature_strings(
    existing: tuple[str, ...], value: Any
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return existing
    additions = tuple(item for item in value if isinstance(item, str) and item.strip())
    return tuple(dict.fromkeys((*existing, *additions)))[:MAX_ITEMS_PER_FIELD]


class DeterministicFakeRewriter:
    """Network-free rewrite backend for replay and unit tests."""

    _DESCRIPTIONS = {
        "initialization": "Retrieve evidence that establishes the initial task-relevant working set.",
        "hypothesis_invalidated": "Retrieve evidence for the changed failure signature and invalidated hypothesis.",
        "user_request": "Retrieve evidence relevant to the latest user-requested requirement.",
        "goal_invalid": "Retrieve evidence for the newly observed goal boundary.",
    }
    _EVIDENCE = {
        "initialization": "task anchor, relevant repository structure, and initial failure evidence",
        "hypothesis_invalidated": "changed error signature and earlier diagnostic evidence",
        "user_request": "evidence for the explicit user requirement",
        "goal_invalid": "evidence around the detected goal boundary",
    }

    def __init__(self) -> None:
        self.calls: list[RewriteRequest] = []

    def rewrite(self, request: RewriteRequest) -> RewriteResult:
        self.calls.append(request)
        current = request.current_goal
        trigger = request.trigger
        values = asdict(current)
        values["description"] = self._DESCRIPTIONS.get(
            trigger, "Retrieve evidence relevant to the triggered retrieval-goal boundary."
        )
        phase_by_trigger = {
            "initialization": "exploration",
            "hypothesis_invalidated": "hypothesis",
            "user_request": "exploration",
            "goal_invalid": "exploration",
        }
        values["phase_hint"] = phase_by_trigger.get(trigger, current.phase_hint)
        for name in ("files", "symbols", "tests", "errors", "evidence_needed"):
            values[name] = _merge_feature_strings(
                tuple(values[name]), request.features.get(name)
            )
        evidence = self._EVIDENCE.get(trigger)
        if evidence:
            values["evidence_needed"] = _merge_feature_strings(
                tuple(values["evidence_needed"]), (evidence,)
            )
        if trigger == "user_request" and request.allow_constraint_updates:
            values["constraints"] = _merge_feature_strings(
                current.constraints, request.features.get("constraints")
            )
        candidate = validate_rewrite_candidate(
            values,
            current,
            trigger=trigger,
            allow_constraint_updates=request.allow_constraint_updates,
        )
        return RewriteResult(
            candidate=candidate,
            raw_response={"backend": "deterministic_fake", "trigger": trigger},
        )


def _nonnegative_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RewriteCommandError("reported token counts must be non-negative integers")
    return value


def _nonnegative_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise RewriteCommandError("reported model cost must be non-negative")
    value = float(value)
    if not math.isfinite(value):
        raise RewriteCommandError("reported model cost must be finite")
    return value


class CommandGoalRewriter:
    """Run an optional local SLM command using one JSON request/response.

    The command receives the :meth:`RewriteRequest.to_payload` object on stdin.
    It may print either a retrieval-Goal object directly, or an envelope with a
    ``candidate`` object and optional ``input_tokens``, ``output_tokens`` and
    ``model_cost`` telemetry.  No shell is involved.
    """

    def __init__(self, argv: tuple[str, ...], timeout_seconds: float = 30.0) -> None:
        if not isinstance(argv, tuple) or not argv or any(
            not isinstance(item, str) or not item for item in argv
        ):
            raise ValueError("argv must be a non-empty tuple of strings")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.argv = argv
        self.timeout_seconds = timeout_seconds

    def rewrite(self, request: RewriteRequest) -> RewriteResult:
        payload_text = json.dumps(
            request.to_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                self.argv,
                input=payload_text,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RewriteCommandError(f"rewrite command failed: {error}") from error
        latency_ms = (time.perf_counter() - started) * 1_000
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-1_000:]
            raise RewriteCommandError(
                f"rewrite command exited with {completed.returncode}: {detail}"
            )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RewriteCommandError("rewrite command did not return valid JSON") from error
        if not isinstance(response, dict):
            raise RewriteCommandError("rewrite command response must be a JSON object")

        if "candidate" in response:
            envelope_fields = {
                "candidate", "input_tokens", "output_tokens", "model_cost",
                "generation_latency_ms",
            }
            unknown_envelope = set(response) - envelope_fields
            if unknown_envelope:
                raise RewriteCommandError(
                    "rewrite command envelope contains unknown fields: "
                    f"{sorted(unknown_envelope)}"
                )
            raw_candidate = response["candidate"]
            input_tokens = _nonnegative_int(response.get("input_tokens"), 0)
            output_tokens = _nonnegative_int(response.get("output_tokens"), 0)
            model_cost = _nonnegative_float(response.get("model_cost"), 0.0)
            generation_latency_ms = _nonnegative_float(
                response.get("generation_latency_ms"), 0.0
            )
        else:
            raw_candidate = response
            input_tokens = 0
            output_tokens = 0
            model_cost = 0.0
            generation_latency_ms = 0.0
        candidate = validate_rewrite_candidate(
            raw_candidate,
            request.current_goal,
            trigger=request.trigger,
            allow_constraint_updates=request.allow_constraint_updates,
        )
        return RewriteResult(
            candidate=candidate,
            latency_ms=latency_ms,
            generation_latency_ms=generation_latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_cost=model_cost,
            raw_response=response,
        )


_PERSISTENT_PROTOCOL = "dram_kv_rewriter_ndjson"
_PERSISTENT_PROTOCOL_VERSION = 1


class PersistentCommandGoalRewriter:
    """Reuse one line-oriented rewrite process across all SLM requests.

    The child must first emit a readiness record and then accept one JSON
    request envelope per line.  This keeps heavyweight model dependencies and
    state outside the core package while avoiding model reloads at every Goal
    boundary.  Responses still cross the exact same fail-closed validator as
    :class:`CommandGoalRewriter`.
    """

    def __init__(
        self,
        argv: tuple[str, ...],
        timeout_seconds: float = 30.0,
        startup_timeout_seconds: float = 300.0,
    ) -> None:
        if not isinstance(argv, tuple) or not argv or any(
            not isinstance(item, str) or not item for item in argv
        ):
            raise ValueError("argv must be a non-empty tuple of strings")
        if timeout_seconds <= 0 or startup_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        self.argv = argv
        self.timeout_seconds = timeout_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._next_request_id = 1
        self._lock = threading.Lock()
        self._pending_startup_latency_ms = 0.0

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _readline(self, timeout_seconds: float) -> str:
        process = self._process
        if process is None or process.stdout is None:
            raise RewriteCommandError("persistent rewrite process is not running")
        try:
            readable, _, _ = select.select(
                (process.stdout,), (), (), timeout_seconds
            )
        except (OSError, ValueError) as error:
            raise RewriteCommandError(
                f"persistent rewrite process wait failed: {error}"
            ) from error
        if not readable:
            raise RewriteCommandError("persistent rewrite process timed out")
        line = process.stdout.readline()
        if not line:
            code = process.poll()
            raise RewriteCommandError(
                f"persistent rewrite process closed stdout (exit={code})"
            )
        return line

    def _start(self) -> None:
        if self.is_running:
            return
        self.close()
        started = time.perf_counter()
        try:
            self._process = subprocess.Popen(
                self.argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            ready = json.loads(self._readline(self.startup_timeout_seconds))
        except (OSError, json.JSONDecodeError, RewriteCommandError) as error:
            self.close()
            if isinstance(error, RewriteCommandError):
                raise
            raise RewriteCommandError(
                f"persistent rewrite process failed to start: {error}"
            ) from error
        expected = {
            "protocol": _PERSISTENT_PROTOCOL,
            "version": _PERSISTENT_PROTOCOL_VERSION,
            "ready": True,
        }
        if ready != expected:
            self.close()
            raise RewriteCommandError(
                "persistent rewrite process returned an invalid readiness record"
            )
        self._pending_startup_latency_ms = (time.perf_counter() - started) * 1_000

    def rewrite(self, request: RewriteRequest) -> RewriteResult:
        with self._lock:
            self._start()
            process = self._process
            assert process is not None and process.stdin is not None
            request_id = str(self._next_request_id)
            self._next_request_id += 1
            envelope = {
                "request_id": request_id,
                "request": request.to_payload(),
            }
            started = time.perf_counter()
            try:
                process.stdin.write(json.dumps(
                    envelope,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n")
                process.stdin.flush()
                response = json.loads(self._readline(self.timeout_seconds))
            except (BrokenPipeError, OSError, json.JSONDecodeError, RewriteCommandError) as error:
                self.close()
                if isinstance(error, RewriteCommandError):
                    raise
                raise RewriteCommandError(
                    f"persistent rewrite request failed: {error}"
                ) from error
            latency_ms = (time.perf_counter() - started) * 1_000
            if not isinstance(response, dict):
                raise RewriteCommandError("persistent response must be a JSON object")
            if response.get("request_id") != request_id:
                raise RewriteCommandError("persistent response request_id mismatch")
            fields = set(response)
            if fields in (
                {"request_id", "error"},
                {"request_id", "error", "telemetry"},
            ):
                detail = response["error"]
                if not isinstance(detail, str):
                    detail = "child returned a non-string error"
                telemetry = response.get("telemetry", {})
                if not isinstance(telemetry, dict):
                    raise RewriteCommandError("persistent error telemetry must be an object")
                telemetry_fields = {
                    "input_tokens", "output_tokens", "generation_latency_ms",
                }
                if set(telemetry) - telemetry_fields:
                    raise RewriteCommandError(
                        "persistent error telemetry contains unknown fields"
                    )
                startup_latency_ms = self._pending_startup_latency_ms
                self._pending_startup_latency_ms = 0.0
                raise RewriteCommandError(
                    f"persistent rewrite failed: {detail[:1_000]}",
                    latency_ms=latency_ms,
                    startup_latency_ms=startup_latency_ms,
                    generation_latency_ms=_nonnegative_float(
                        telemetry.get("generation_latency_ms"), 0.0
                    ),
                    input_tokens=_nonnegative_int(
                        telemetry.get("input_tokens"), 0
                    ),
                    output_tokens=_nonnegative_int(
                        telemetry.get("output_tokens"), 0
                    ),
                )
            if fields != {"request_id", "result"}:
                raise RewriteCommandError(
                    "persistent response contains unknown or missing fields"
                )
            result = response["result"]
            if not isinstance(result, dict):
                raise RewriteCommandError("persistent result must be a JSON object")
            if "candidate" in result:
                envelope_fields = {
                    "candidate", "input_tokens", "output_tokens", "model_cost",
                    "generation_latency_ms",
                }
                unknown = set(result) - envelope_fields
                if unknown:
                    raise RewriteCommandError(
                        "rewrite command envelope contains unknown fields: "
                        f"{sorted(unknown)}"
                    )
                raw_candidate = result["candidate"]
                input_tokens = _nonnegative_int(result.get("input_tokens"), 0)
                output_tokens = _nonnegative_int(result.get("output_tokens"), 0)
                model_cost = _nonnegative_float(result.get("model_cost"), 0.0)
                generation_latency_ms = _nonnegative_float(
                    result.get("generation_latency_ms"), 0.0
                )
            else:
                raw_candidate = result
                input_tokens = output_tokens = 0
                model_cost = 0.0
                generation_latency_ms = 0.0
            candidate = validate_rewrite_candidate(
                raw_candidate,
                request.current_goal,
                trigger=request.trigger,
                allow_constraint_updates=request.allow_constraint_updates,
            )
            startup_latency_ms = self._pending_startup_latency_ms
            self._pending_startup_latency_ms = 0.0
            return RewriteResult(
                candidate=candidate,
                latency_ms=latency_ms,
                startup_latency_ms=startup_latency_ms,
                generation_latency_ms=generation_latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_cost=model_cost,
                raw_response=result,
            )

    def close(self) -> None:
        process, self._process = self._process, None
        self._pending_startup_latency_ms = 0.0
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout is not None:
            process.stdout.close()

    def __enter__(self) -> "PersistentCommandGoalRewriter":
        self._start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup for replay APIs that do not own a context manager.
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "CommandGoalRewriter",
    "DeterministicFakeRewriter",
    "GoalRewriter",
    "PersistentCommandGoalRewriter",
    "RewriteCommandError",
    "RewriteRequest",
    "RewriteResult",
    "RewriteValidationError",
    "validate_rewrite_candidate",
]
