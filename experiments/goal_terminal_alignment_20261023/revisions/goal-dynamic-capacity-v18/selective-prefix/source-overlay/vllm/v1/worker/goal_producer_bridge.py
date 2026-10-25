# SPDX-License-Identifier: Apache-2.0
"""Fail-closed control plane for a vLLM Goal-query vector producer.

This module deliberately has no vLLM or torch import at module import time.
The GPU runner supplies compact vectors produced from its existing forward;
this control plane owns request identity, page-boundary bookkeeping, Goal
vector caching, semantic ranking, and sealed ranking artifacts.

It is not an online endpoint and does not claim an end-to-end integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import secrets
from typing import Iterable, Mapping, Sequence
from uuid import uuid4


_SESSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_KINDS = frozenset({"agent", "goal_probe"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VLLM_INTERNAL_REQUEST_SUFFIX_RE = re.compile(
    r"^(?P<external>spcv1:.*)-[0-9a-f]{8}$", re.IGNORECASE
)
_VLLM_INDEXED_INTERNAL_REQUEST_SUFFIX_RE = re.compile(
    r"^(?P<external>spcv1:.*)-(?P<prompt_index>[0-9]+)-[0-9a-f]{8}$",
    re.IGNORECASE,
)


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_hash(value: object) -> str:
    return sha256(_stable_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ProducerProvenance:
    model: str
    revision: str
    hidden_layer: int
    projection_hash: str
    dimension: int = 128
    page_tokens: int = 784
    pooling: str = "last_token"
    dtype: str = "int8"
    vector_source: str = "vllm_existing_forward_final_hidden"
    producer_version: str = "goal-producer-bridge-v1"

    def __post_init__(self) -> None:
        if not self.model or not self.revision:
            raise ValueError("model and revision are required")
        if self.hidden_layer < 0:
            raise ValueError("hidden_layer cannot be negative")
        if not _SHA256_RE.fullmatch(self.projection_hash.casefold()):
            raise ValueError("projection_hash must be SHA-256 hex")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.page_tokens <= 0:
            raise ValueError("page_tokens must be positive")
        if self.pooling != "last_token" or self.dtype != "int8":
            raise ValueError("bridge v1 requires last_token int8 vectors")

    @property
    def provenance_hash(self) -> str:
        return _stable_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class CompactVector:
    values: tuple[int, ...]
    scale: float

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("compact vector cannot be empty")
        if any(isinstance(value, bool) or value < -127 or value > 127 for value in self.values):
            raise ValueError("int8 compact values must lie in [-127, 127]")
        if not any(self.values):
            raise ValueError("compact vector cannot be all zero")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")

    @property
    def content_hash(self) -> str:
        return _stable_hash({"dtype": "int8", "scale": self.scale, "values": self.values})

    def unit_values(self) -> tuple[float, ...]:
        values = tuple(value * self.scale for value in self.values)
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            raise ValueError("compact vector cannot be all zero")
        return tuple(value / norm for value in values)


def compact_projected_vector(values: Sequence[float]) -> CompactVector:
    """Mirror Xuanxue's L2-normalize then symmetric-int8 compaction."""

    raw = tuple(float(value) for value in values)
    if not raw or any(not math.isfinite(value) for value in raw):
        raise ValueError("projected vector must contain finite values")
    norm = math.sqrt(sum(value * value for value in raw))
    if norm == 0:
        raise ValueError("projected vector cannot be all zero")
    unit = tuple(value / norm for value in raw)
    peak = max(abs(value) for value in unit)
    scale = peak / 127.0
    quantized = tuple(
        max(-127, min(127, int(round(value / scale)))) for value in unit
    )
    return CompactVector(quantized, scale)


def compact_cosine(left: CompactVector, right: CompactVector) -> float:
    if len(left.values) != len(right.values):
        raise ValueError("vector dimensions differ")
    return sum(a * b for a, b in zip(left.unit_values(), right.unit_values()))


@dataclass(frozen=True, slots=True)
class ProducerRequestTag:
    session_id: str
    turn_id: int
    kind: str
    goal_hash: str | None = None

    def __post_init__(self) -> None:
        if not _SESSION_RE.fullmatch(self.session_id):
            raise ValueError("session_id must be 1-64 URL-safe characters")
        if self.turn_id <= 0:
            raise ValueError("turn_id must be positive")
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {sorted(_KINDS)}")
        if self.kind == "goal_probe":
            if self.goal_hash is None or not _SHA256_RE.fullmatch(self.goal_hash.casefold()):
                raise ValueError("goal_probe requires a SHA-256 goal_hash")
        elif self.goal_hash is not None:
            raise ValueError("agent request cannot carry goal_hash")

    def request_id(self) -> str:
        goal = self.goal_hash or "-"
        return f"spcv1:{self.kind}:{self.session_id}:{self.turn_id}:{goal}"

    def openai_request_fields(self) -> dict[str, str]:
        """Fields for a vLLM 0.24.0 OpenAI request body.

        vLLM prefixes this value with ``chatcmpl-`` before passing it to the
        engine.  Callers must not also set a conflicting ``X-Request-Id``
        header because that header has precedence in ``BaseServing``.
        """

        return {"request_id": self.request_id()}

    @classmethod
    def parse(cls, request_id: str) -> "ProducerRequestTag | None":
        normalized = request_id
        for prefix in ("chatcmpl-", "cmpl-"):
            if normalized.startswith(prefix):
                normalized = normalized.removeprefix(prefix)
                break
        if not normalized.startswith("spcv1:"):
            return None
        # vLLM 0.24.0 preserves the caller-provided ID as external_req_id but
        # appends "-<8 random hex>" to the internal request_id.  The GPU model
        # runner sees that internal ID, so recover the authenticated external
        # tag before parsing its fields while retaining vLLM's uniqueness.
        # The completions endpoint first adds ``-<prompt_index>`` and the
        # engine then adds its normal ``-<8 random hex>`` suffix.  Strip the
        # indexed form first so the prompt index cannot be mistaken for part
        # of a SHA-256 goal hash.
        internal_match = _VLLM_INDEXED_INTERNAL_REQUEST_SUFFIX_RE.fullmatch(
            normalized
        )
        if internal_match is None:
            internal_match = _VLLM_INTERNAL_REQUEST_SUFFIX_RE.fullmatch(normalized)
        if internal_match is not None:
            normalized = internal_match.group("external")
        parts = normalized.split(":")
        if len(parts) != 5:
            raise ValueError("malformed spcv1 request id")
        _, kind, session_id, turn_text, goal = parts
        try:
            turn_id = int(turn_text)
        except ValueError as exc:
            raise ValueError("malformed spcv1 turn id") from exc
        return cls(session_id, turn_id, kind, None if goal == "-" else goal.casefold())


def ranking_artifact_path(
    artifact_dir: Path, tag: ProducerRequestTag
) -> Path:
    if tag.kind != "goal_probe" or tag.goal_hash is None:
        raise ValueError("ranking artifacts require a goal_probe tag")
    return Path(artifact_dir) / (
        f"{tag.session_id}.turn{tag.turn_id}.{tag.goal_hash}.ranking.json"
    )


def read_sealed_ranking_artifact(
    path: Path,
    *,
    expected_tag: ProducerRequestTag,
    expected_provenance_hash: str | None = None,
) -> dict[str, object]:
    """Read and validate a ranking artifact before exposing it to a client."""

    expected = ranking_artifact_path(path.parent, expected_tag)
    if path != expected:
        raise ValueError("ranking artifact path does not match request identity")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("ranking artifact is not a complete sealed record")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("ranking artifact must be a JSON object")
    assert expected_tag.goal_hash is not None
    identity = (
        value.get("session_id"),
        value.get("turn_id"),
        value.get("goal_hash"),
    )
    expected_identity = (
        expected_tag.session_id,
        expected_tag.turn_id,
        expected_tag.goal_hash,
    )
    if identity != expected_identity:
        raise ValueError("ranking artifact identity mismatch")
    if value.get("schema_name") != "vllm_goal_query_ranking":
        raise ValueError("unexpected ranking artifact schema")
    if expected_provenance_hash is not None and (
        value.get("provenance_hash") != expected_provenance_hash
    ):
        raise ValueError("ranking artifact provenance mismatch")
    params = value.get("kv_transfer_params")
    if not isinstance(params, dict):
        raise ValueError("ranking artifact lacks kv_transfer_params")
    if params.get("selective_page_policy") != "goal_query":
        raise ValueError("ranking artifact is not a goal_query payload")
    if params.get("selective_page_query_hash") != value.get("query_hash"):
        raise ValueError("ranking query hash mismatch")
    if params.get("selective_page_ranked_pages") != value.get(
        "ranked_logical_pages"
    ):
        raise ValueError("ranking page list mismatch")
    return value


async def selective_goal_ranking_middleware(request: object, call_next: object):
    """Authenticated read-only bridge from the API process to sealed files.

    Configure vLLM with ``--middleware
    vllm.v1.worker.goal_producer_bridge.selective_goal_ranking_middleware``
    and set both
    ``VLLM_SELECTIVE_GOAL_ARTIFACT_DIR`` and a high-entropy
    ``VLLM_SELECTIVE_GOAL_TOKEN``.  Only ``GET /selective_goal_ranking`` is
    intercepted; every other route is delegated unchanged.
    """

    if request.url.path != "/selective_goal_ranking":
        return await call_next(request)
    from starlette.responses import JSONResponse

    if request.method != "GET":
        return JSONResponse({"error": "method_not_allowed"}, status_code=405)
    configured_token = os.environ.get("VLLM_SELECTIVE_GOAL_TOKEN", "")
    supplied_token = request.headers.get("X-Selective-Producer-Token", "")
    if (
        len(configured_token) < 32
        or not supplied_token
        or not secrets.compare_digest(configured_token, supplied_token)
    ):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    artifact_dir_text = os.environ.get("VLLM_SELECTIVE_GOAL_ARTIFACT_DIR", "")
    if not artifact_dir_text:
        return JSONResponse({"error": "artifact_dir_not_configured"}, status_code=503)
    try:
        tag = ProducerRequestTag(
            request.query_params["session_id"],
            int(request.query_params["turn_id"]),
            "goal_probe",
            request.query_params["goal_hash"].casefold(),
        )
        path = ranking_artifact_path(Path(artifact_dir_text), tag)
    except (KeyError, TypeError, ValueError) as exc:
        return JSONResponse({"error": "invalid_identity", "detail": str(exc)}, status_code=400)
    if not path.is_file():
        return JSONResponse({"status": "pending"}, status_code=404)
    try:
        value = read_sealed_ranking_artifact(path, expected_tag=tag)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return JSONResponse({"error": "invalid_artifact", "detail": str(exc)}, status_code=500)
    return JSONResponse(value, status_code=200)


def page_boundary_rows(
    num_computed_tokens: int,
    num_scheduled_tokens: int,
    page_tokens: int,
) -> tuple[tuple[int, int], ...]:
    """Return ``(flattened-request-row, logical-page-index)`` boundaries."""

    if num_computed_tokens < 0 or num_scheduled_tokens < 0 or page_tokens <= 0:
        raise ValueError("token counts must be non-negative and page_tokens positive")
    result: list[tuple[int, int]] = []
    for offset in range(num_scheduled_tokens):
        logical_position = num_computed_tokens + offset
        if (logical_position + 1) % page_tokens == 0:
            result.append((offset, logical_position // page_tokens))
    return tuple(result)


def batch_page_boundary_rows(
    request_ids: Sequence[str],
    num_computed_tokens: Mapping[str, int],
    num_scheduled_tokens: Mapping[str, int],
    page_tokens: int,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Map each request's Page boundary to the flattened hidden-state row.

    vLLM concatenates request chunks in ``input_batch.req_ids`` order.  The
    returned row index is therefore global within the unpadded hidden tensor,
    while the Page index remains logical within that request.
    """

    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request_ids contain duplicates")
    if set(request_ids) != set(num_computed_tokens) or set(request_ids) != set(
        num_scheduled_tokens
    ):
        raise ValueError("batch token maps must exactly cover request_ids")
    cursor = 0
    result: dict[str, tuple[tuple[int, int], ...]] = {}
    for request_id in request_ids:
        count = num_scheduled_tokens[request_id]
        local = page_boundary_rows(
            num_computed_tokens[request_id], count, page_tokens
        )
        result[request_id] = tuple(
            (cursor + row, page_index) for row, page_index in local
        )
        cursor += count
    return result


@dataclass(frozen=True, slots=True)
class GoalPolicyRankingArtifact:
    session_id: str
    turn_id: int
    goal_hash: str
    query_hash: str
    provenance_hash: str
    fraction: float
    ranked_logical_pages: tuple[int, ...]
    selected_count: int
    mandatory_pages: tuple[int, ...]
    recent_pages: tuple[int, ...]
    schema_name: str = "vllm_goal_query_ranking"
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.ranked_logical_pages:
            raise ValueError("ranking cannot be empty")
        if len(set(self.ranked_logical_pages)) != len(self.ranked_logical_pages):
            raise ValueError("ranking contains duplicate pages")
        if self.selected_count <= 0 or self.selected_count > len(self.ranked_logical_pages):
            raise ValueError("invalid selected_count")

    def kv_transfer_params(self) -> dict[str, object]:
        return {
            "selective_page_fraction": self.fraction,
            "selective_page_policy": "goal_query",
            "selective_page_ranked_pages": list(self.ranked_logical_pages),
            "selective_page_query_hash": self.query_hash,
        }

    def as_json(self) -> dict[str, object]:
        value = asdict(self)
        value["ranked_logical_pages"] = list(self.ranked_logical_pages)
        value["mandatory_pages"] = list(self.mandatory_pages)
        value["recent_pages"] = list(self.recent_pages)
        value["kv_transfer_params"] = self.kv_transfer_params()
        return value


class GoalProducerStore:
    """In-worker sealed vector index and Goal-ranking artifact producer."""

    def __init__(
        self,
        provenance: ProducerProvenance,
        *,
        fraction: float,
        artifact_dir: Path | None = None,
    ) -> None:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        self.provenance = provenance
        self.fraction = float(fraction)
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self._pages: dict[tuple[str, int], CompactVector] = {}
        self._page_turns: dict[tuple[str, int], int] = {}
        self._goals: dict[tuple[str, str], CompactVector] = {}

    def publish_page(
        self,
        session_id: str,
        turn_id: int,
        page_index: int,
        vector: CompactVector,
    ) -> None:
        ProducerRequestTag(session_id, turn_id, "agent")
        if page_index < 0:
            raise ValueError("page_index cannot be negative")
        if len(vector.values) != self.provenance.dimension:
            raise ValueError("page vector dimension differs from provenance")
        key = (session_id, page_index)
        previous = self._pages.get(key)
        previous_turn = self._page_turns.get(key)
        if previous is not None:
            if previous != vector:
                raise ValueError("sealed logical page vector cannot be overwritten")
            if previous_turn is None or turn_id < previous_turn:
                raise ValueError("logical page cannot move to an earlier turn")
            # A growing conversation can expose the same immutable logical
            # page boundary again. Treat an exact later observation as an
            # idempotent no-op and retain the first-seen turn for recency.
            return
        self._pages[key] = vector
        self._page_turns[key] = turn_id

    def publish_scheduled_rows(
        self,
        tag: ProducerRequestTag,
        *,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
        boundary_vectors: Mapping[int, CompactVector],
    ) -> tuple[int, ...]:
        if tag.kind != "agent":
            raise ValueError("only agent requests publish page vectors")
        boundaries = page_boundary_rows(
            num_computed_tokens,
            num_scheduled_tokens,
            self.provenance.page_tokens,
        )
        expected_rows = {row for row, _ in boundaries}
        if set(boundary_vectors) != expected_rows:
            raise ValueError("boundary vector rows do not match scheduled page boundaries")
        published: list[int] = []
        for row, page_index in boundaries:
            self.publish_page(tag.session_id, tag.turn_id, page_index, boundary_vectors[row])
            published.append(page_index)
        return tuple(published)

    def publish_goal(self, tag: ProducerRequestTag, vector: CompactVector) -> bool:
        if tag.kind != "goal_probe" or tag.goal_hash is None:
            raise ValueError("Goal vector requires a goal_probe tag")
        if len(vector.values) != self.provenance.dimension:
            raise ValueError("Goal vector dimension differs from provenance")
        key = (tag.session_id, tag.goal_hash)
        previous = self._goals.get(key)
        if previous is not None and previous != vector:
            raise ValueError("same Goal hash produced a different vector")
        cache_hit = previous is not None
        self._goals[key] = vector
        return cache_hit

    def build_ranking(
        self,
        *,
        session_id: str,
        turn_id: int,
        goal_hash: str,
        mandatory_pages: Iterable[int] = (),
        recent_turns: int = 0,
        recent_quota: int = 0,
    ) -> GoalPolicyRankingArtifact:
        ProducerRequestTag(session_id, turn_id, "goal_probe", goal_hash)
        if recent_turns < 0 or recent_quota < 0:
            raise ValueError("recent safety settings cannot be negative")
        query = self._goals.get((session_id, goal_hash))
        if query is None:
            raise ValueError("Goal vector is missing")
        available = sorted(
            page_index
            for (candidate_session, page_index), page_turn in self._page_turns.items()
            if candidate_session == session_id and page_turn < turn_id
        )
        if not available:
            raise ValueError("no historical page vectors are available")
        mandatory = tuple(sorted(set(int(page) for page in mandatory_pages)))
        if any(page not in available for page in mandatory):
            raise ValueError("mandatory page is unavailable")
        mandatory_set = set(mandatory)
        cutoff = max(0, turn_id - recent_turns)
        recent_candidates = [
            page for page in available
            if page not in mandatory_set and self._page_turns[(session_id, page)] >= cutoff
        ]
        recent_candidates.sort(
            key=lambda page: (-self._page_turns[(session_id, page)], -page)
        )
        recent = tuple(recent_candidates[:recent_quota])
        safety = mandatory_set | set(recent)
        semantic = sorted(
            (page for page in available if page not in safety),
            key=lambda page: (
                -compact_cosine(query, self._pages[(session_id, page)]),
                page,
            ),
        )
        ranking = tuple((*mandatory, *recent, *semantic))
        selected_count = int(len(available) * self.fraction)
        if selected_count <= 0:
            raise ValueError("fraction would select zero pages")
        if len(safety) > selected_count:
            raise ValueError("safety pages exceed selective budget")
        query_hash = sha256(
            (
                "goal_query:"
                + self.provenance.provenance_hash
                + ":"
                + goal_hash.casefold()
                + ":"
                + query.content_hash
            ).encode("ascii")
        ).hexdigest()
        return GoalPolicyRankingArtifact(
            session_id=session_id,
            turn_id=turn_id,
            goal_hash=goal_hash.casefold(),
            query_hash=query_hash,
            provenance_hash=self.provenance.provenance_hash,
            fraction=self.fraction,
            ranked_logical_pages=ranking,
            selected_count=selected_count,
            mandatory_pages=mandatory,
            recent_pages=recent,
        )

    def write_artifact(self, artifact: GoalPolicyRankingArtifact) -> Path:
        if self.artifact_dir is None:
            raise ValueError("artifact_dir is not configured")
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        tag = ProducerRequestTag(
            artifact.session_id,
            artifact.turn_id,
            "goal_probe",
            artifact.goal_hash,
        )
        final = ranking_artifact_path(self.artifact_dir, tag)
        content = (_stable_json(artifact.as_json()) + "\n").encode("utf-8")
        if final.exists():
            if final.read_bytes() != content:
                raise ValueError("sealed ranking artifact cannot be overwritten")
            return final
        temporary = final.with_name(final.name + f".tmp.{os.getpid()}.{uuid4().hex}")
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, final)
            except FileExistsError:
                if final.read_bytes() != content:
                    raise ValueError("sealed ranking artifact race changed content")
            return final
        finally:
            temporary.unlink(missing_ok=True)

    def release_session(self, session_id: str) -> None:
        self._pages = {
            key: value for key, value in self._pages.items() if key[0] != session_id
        }
        self._page_turns = {
            key: value for key, value in self._page_turns.items() if key[0] != session_id
        }
        self._goals = {
            key: value for key, value in self._goals.items() if key[0] != session_id
        }


@dataclass(frozen=True, slots=True)
class ProducerHookRequestResult:
    request_id: str
    tag: ProducerRequestTag
    published_pages: tuple[int, ...] = ()
    goal_published: bool = False
    goal_cache_hit: bool = False
    ranking_artifact: GoalPolicyRankingArtifact | None = None
    artifact_path: str | None = None


@dataclass(frozen=True, slots=True)
class ProducerHookBatchResult:
    processed: tuple[ProducerHookRequestResult, ...]
    ignored_request_ids: tuple[str, ...]
    selected_hidden_rows: tuple[int, ...]


class ExistingForwardGoalProducerHook:
    """Consume final hidden rows from one existing vLLM model forward.

    This adapter intentionally accepts plain request IDs and token-count maps
    instead of importing vLLM types.  A narrow patch in ``gpu_model_runner``
    can construct these values from ``input_batch`` and ``SchedulerOutput``.

    Goal probes are published only by the chunk that contains the final prompt
    token.  Decode chunks are ignored, which prevents a partial/chunked Goal
    prefill from silently becoming a query vector.
    """

    def __init__(
        self,
        torch: object,
        projection: object,
        store: GoalProducerStore,
        *,
        recent_turns: int = 0,
        recent_quota: int = 0,
    ) -> None:
        if recent_turns < 0 or recent_quota < 0:
            raise ValueError("recent safety settings cannot be negative")
        if getattr(projection, "ndim", None) != 2:
            raise ValueError("projection must be a two-dimensional tensor")
        if int(projection.shape[0]) != store.provenance.dimension:
            raise ValueError("projection output dimension differs from provenance")
        self.torch = torch
        self.projection = projection
        self.store = store
        self.recent_turns = recent_turns
        self.recent_quota = recent_quota

    def process_existing_forward(
        self,
        *,
        request_ids: Sequence[str],
        num_computed_tokens: Mapping[str, int],
        num_scheduled_tokens: Mapping[str, int],
        num_prompt_tokens: Mapping[str, int],
        hidden_states: object,
        num_unpadded_tokens: int,
        mandatory_pages: Mapping[str, Iterable[int]] | None = None,
    ) -> ProducerHookBatchResult:
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_ids contain duplicates")
        expected = set(request_ids)
        for name, values in (
            ("num_computed_tokens", num_computed_tokens),
            ("num_scheduled_tokens", num_scheduled_tokens),
            ("num_prompt_tokens", num_prompt_tokens),
        ):
            if set(values) != expected:
                raise ValueError(f"{name} must exactly cover request_ids")
        if getattr(hidden_states, "ndim", None) != 2:
            raise ValueError("hidden_states must be a two-dimensional tensor")
        if num_unpadded_tokens < 0 or num_unpadded_tokens > int(hidden_states.shape[0]):
            raise ValueError("invalid num_unpadded_tokens")
        scheduled_total = sum(int(num_scheduled_tokens[rid]) for rid in request_ids)
        if scheduled_total != num_unpadded_tokens:
            raise ValueError("scheduled token sum differs from num_unpadded_tokens")
        if int(hidden_states.shape[1]) != int(self.projection.shape[1]):
            raise ValueError("hidden dimension differs from projection input")
        if mandatory_pages is None:
            mandatory_pages = {}
        unknown_mandatory = set(mandatory_pages) - expected
        if unknown_mandatory:
            raise ValueError("mandatory_pages contains unknown request IDs")

        parsed: dict[str, ProducerRequestTag] = {}
        ignored: list[str] = []
        for request_id in request_ids:
            tag = ProducerRequestTag.parse(request_id)
            if tag is None:
                ignored.append(request_id)
            else:
                parsed[request_id] = tag

        boundary_rows = batch_page_boundary_rows(
            request_ids,
            num_computed_tokens,
            num_scheduled_tokens,
            self.store.provenance.page_tokens,
        )
        request_starts: dict[str, int] = {}
        cursor = 0
        for request_id in request_ids:
            request_starts[request_id] = cursor
            cursor += int(num_scheduled_tokens[request_id])

        # Select all needed rows first so the implementation can perform one
        # vectorized projection and one device-to-host copy per model forward.
        requested_rows: dict[tuple[str, str, int], int] = {}
        for request_id, tag in parsed.items():
            if tag.kind == "agent":
                for global_row, page_index in boundary_rows[request_id]:
                    requested_rows[(request_id, "page", page_index)] = global_row
                continue

            computed = int(num_computed_tokens[request_id])
            scheduled = int(num_scheduled_tokens[request_id])
            prompt = int(num_prompt_tokens[request_id])
            if computed < 0 or scheduled < 0 or prompt <= 0:
                raise ValueError("invalid request token counts")
            # Capture exactly the final Goal prompt token.  A decode-only
            # chunk has computed >= prompt and therefore publishes nothing.
            if computed < prompt <= computed + scheduled:
                local_row = prompt - computed - 1
                requested_rows[(request_id, "goal", 0)] = (
                    request_starts[request_id] + local_row
                )

        ordered_keys = sorted(requested_rows, key=lambda key: requested_rows[key])
        selected_rows = tuple(requested_rows[key] for key in ordered_keys)
        compact_vectors: tuple[CompactVector, ...]
        if selected_rows:
            indices = self.torch.tensor(
                selected_rows, dtype=self.torch.long, device=hidden_states.device
            )
            selected_hidden = hidden_states.index_select(0, indices)
            projection = self.projection.to(
                device=hidden_states.device, dtype=self.torch.float32
            )
            compact_vectors = compact_hidden_rows_with_torch(
                self.torch, selected_hidden, projection
            )
        else:
            compact_vectors = ()
        vectors_by_key = dict(zip(ordered_keys, compact_vectors, strict=True))

        results: list[ProducerHookRequestResult] = []
        # Publish agent pages before Goal rankings so a same-forward Goal probe
        # sees all historical pages with a strictly earlier turn ID.
        for request_id in request_ids:
            tag = parsed.get(request_id)
            if tag is None or tag.kind != "agent":
                continue
            published: list[int] = []
            for _global_row, page_index in boundary_rows[request_id]:
                key = (request_id, "page", page_index)
                vector = vectors_by_key.get(key)
                if vector is None:
                    raise RuntimeError("selected page vector is missing")
                self.store.publish_page(
                    tag.session_id, tag.turn_id, page_index, vector
                )
                published.append(page_index)
            results.append(
                ProducerHookRequestResult(
                    request_id=request_id,
                    tag=tag,
                    published_pages=tuple(published),
                )
            )

        for request_id in request_ids:
            tag = parsed.get(request_id)
            if tag is None or tag.kind != "goal_probe":
                continue
            key = (request_id, "goal", 0)
            vector = vectors_by_key.get(key)
            if vector is None:
                results.append(ProducerHookRequestResult(request_id, tag))
                continue
            cache_hit = self.store.publish_goal(tag, vector)
            assert tag.goal_hash is not None
            artifact = self.store.build_ranking(
                session_id=tag.session_id,
                turn_id=tag.turn_id,
                goal_hash=tag.goal_hash,
                mandatory_pages=mandatory_pages.get(request_id, ()),
                recent_turns=self.recent_turns,
                recent_quota=self.recent_quota,
            )
            artifact_path = None
            if self.store.artifact_dir is not None:
                artifact_path = str(self.store.write_artifact(artifact))
            results.append(
                ProducerHookRequestResult(
                    request_id=request_id,
                    tag=tag,
                    goal_published=True,
                    goal_cache_hit=cache_hit,
                    ranking_artifact=artifact,
                    artifact_path=artifact_path,
                )
            )
        return ProducerHookBatchResult(
            processed=tuple(results),
            ignored_request_ids=tuple(ignored),
            selected_hidden_rows=selected_rows,
        )


def build_xuanxue_projection(torch: object, hidden_size: int, dimension: int, seed: int):
    """Build the exact seeded Rademacher projection used by the reference code."""

    if hidden_size <= 0 or dimension <= 0 or seed < 0:
        raise ValueError("invalid projection configuration")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.randint(
        0, 2, (dimension, hidden_size), generator=generator, dtype=torch.int8
    ).to(torch.float32)
    values.mul_(2.0).sub_(1.0).mul_(1.0 / math.sqrt(dimension))
    digest = sha256(values.contiguous().numpy().tobytes()).hexdigest()
    return values, digest


def compact_hidden_with_torch(torch: object, hidden: object, projection: object) -> CompactVector:
    """Project one existing-forward hidden row without another model forward."""

    vector = torch.nn.functional.linear(hidden.float(), projection)
    vector = torch.nn.functional.normalize(vector, p=2, dim=-1)
    peak = vector.abs().max()
    if not bool(torch.isfinite(peak)) or float(peak) == 0.0:
        raise ValueError("projection produced an invalid zero vector")
    scale = peak / 127.0
    values = torch.clamp(torch.round(vector / scale), -127, 127).to(torch.int8)
    return CompactVector(
        tuple(int(value) for value in values.cpu().tolist()),
        float(scale.cpu()),
    )


def compact_hidden_rows_with_torch(
    torch: object, hidden_rows: object, projection: object
) -> tuple[CompactVector, ...]:
    """Vectorized form of Xuanxue's projection/compaction for many rows."""

    if getattr(hidden_rows, "ndim", None) != 2 or int(hidden_rows.shape[0]) <= 0:
        raise ValueError("hidden_rows must be a non-empty two-dimensional tensor")
    vectors = torch.nn.functional.linear(hidden_rows.float(), projection)
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
    peaks = vectors.abs().amax(dim=-1)
    if not bool(torch.isfinite(peaks).all()) or bool((peaks == 0).any()):
        raise ValueError("projection produced an invalid zero vector")
    scales = peaks / 127.0
    values = torch.clamp(
        torch.round(vectors / scales.unsqueeze(-1)), -127, 127
    ).to(torch.int8)
    cpu_values = values.cpu().tolist()
    cpu_scales = scales.cpu().tolist()
    return tuple(
        CompactVector(tuple(int(value) for value in row), float(scale))
        for row, scale in zip(cpu_values, cpu_scales, strict=True)
    )


__all__ = [
    "CompactVector",
    "ExistingForwardGoalProducerHook",
    "GoalPolicyRankingArtifact",
    "GoalProducerStore",
    "ProducerHookBatchResult",
    "ProducerHookRequestResult",
    "ProducerProvenance",
    "ProducerRequestTag",
    "batch_page_boundary_rows",
    "build_xuanxue_projection",
    "compact_cosine",
    "compact_hidden_with_torch",
    "compact_hidden_rows_with_torch",
    "compact_projected_vector",
    "page_boundary_rows",
    "ranking_artifact_path",
    "read_sealed_ranking_artifact",
    "selective_goal_ranking_middleware",
]
