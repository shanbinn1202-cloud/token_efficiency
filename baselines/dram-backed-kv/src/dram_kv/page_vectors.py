"""Compact vectors published from the main-model prefill/decode path.

The production boundary starts at :class:`PageVectorPublisher`: an engine
publishes only compact vectors and never transfers complete token hidden
states into this CPU index.  :class:`SimulatedPrefillPageVectorPublisher` is a
dependency-free CPU test double for pooling/projection/quantization; it is not
an embedding model and does not perform a Transformer forward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from threading import RLock
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence


_DTYPES = frozenset({"int8", "float16", "float32"})
_POOLING = frozenset({"masked_mean", "last_token"})


def _stable_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    normalized = value.casefold()
    return len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )


@dataclass(frozen=True)
class PageVectorProvenance:
    """Identity of one mutually compatible Page/query vector space."""

    model: str
    revision: str
    hidden_layer: int
    pooling: str
    projection_hash: str
    dimension: int
    dtype: str
    vector_source: str = "main_model_hidden_states"
    page_size: int = 16
    normalization: str = "l2"
    quantization: str = "symmetric_per_vector"
    selector_version: str = "v2"

    def __post_init__(self) -> None:
        text_fields = (
            self.model,
            self.revision,
            self.projection_hash,
            self.dtype,
            self.vector_source,
            self.normalization,
            self.quantization,
            self.selector_version,
        )
        if any(not str(value).strip() for value in text_fields):
            raise ValueError("complete Page vector provenance is required")
        if self.hidden_layer < 0:
            raise ValueError("hidden_layer cannot be negative")
        if not _is_sha256(self.projection_hash):
            raise ValueError("projection_hash must be a SHA-256 hex digest")
        if self.pooling not in _POOLING:
            raise ValueError(f"unsupported pooling {self.pooling!r}")
        if self.dimension <= 0 or self.dimension > 65_536:
            raise ValueError("Page vector dimension must be in [1, 65536]")
        if self.dtype not in _DTYPES:
            raise ValueError(f"unsupported compact vector dtype {self.dtype!r}")
        if self.page_size != 16:
            raise ValueError("Selector V2 requires 16-token physical Pages")
        if self.normalization != "l2":
            raise ValueError("Selector V2 requires L2-normalized vectors")
        expected_quantization = "symmetric_per_vector" if self.dtype == "int8" else "none"
        if self.quantization != expected_quantization:
            raise ValueError(
                f"{self.dtype} vectors require quantization={expected_quantization!r}"
            )
        if self.selector_version != "v2":
            raise ValueError("Page vector provenance must identify Selector V2")

    @property
    def provenance_hash(self) -> str:
        return _stable_hash(asdict(self))


@dataclass(frozen=True)
class CompactVector:
    """A bounded dense vector; int8 values carry a per-vector scale."""

    values: tuple[int | float, ...]
    dtype: str
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise ValueError(f"unsupported compact vector dtype {self.dtype!r}")
        if not self.values or len(self.values) > 65_536:
            raise ValueError("compact vector must have bounded non-zero dimension")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("compact vector scale must be finite and positive")
        if self.dtype == "int8":
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < -127
                or value > 127
                for value in self.values
            ):
                raise ValueError("int8 compact values must be integers in [-127, 127]")
        elif any(not math.isfinite(float(value)) for value in self.values):
            raise ValueError("compact vector values must be finite")
        if not any(float(value) != 0.0 for value in self.values):
            raise ValueError("compact vector cannot be all zero")

    @property
    def dimension(self) -> int:
        return len(self.values)

    def unit_values(self) -> tuple[float, ...]:
        values = tuple(float(value) * self.scale for value in self.values)
        norm = math.sqrt(sum(value * value for value in values))
        if not norm:  # guarded by __post_init__
            raise ValueError("compact vector cannot be all zero")
        return tuple(value / norm for value in values)

    @property
    def content_hash(self) -> str:
        return _stable_hash({
            "dtype": self.dtype,
            "scale": self.scale,
            "values": self.values,
        })


@dataclass(frozen=True)
class PageVectorEntry:
    session_id: str
    page_id: str
    vector: CompactVector
    provenance: PageVectorProvenance

    def __post_init__(self) -> None:
        if not self.session_id or not self.page_id:
            raise ValueError("Page vector entry requires session_id and page_id")
        if self.vector.dimension != self.provenance.dimension:
            raise ValueError("Page vector dimension differs from provenance")
        if self.vector.dtype != self.provenance.dtype:
            raise ValueError("Page vector dtype differs from provenance")

    @property
    def vector_ref(self) -> str:
        return "page-vector-v2:" + _stable_hash({
            "session_id": self.session_id,
            "page_id": self.page_id,
            "provenance_hash": self.provenance.provenance_hash,
            "content_hash": self.vector.content_hash,
        })


@dataclass(frozen=True)
class QueryVectorEntry:
    session_id: str
    turn_id: int
    vector: CompactVector
    provenance: PageVectorProvenance

    def __post_init__(self) -> None:
        if not self.session_id or self.turn_id <= 0:
            raise ValueError("query vector entry requires session_id and positive turn_id")
        if self.vector.dimension != self.provenance.dimension:
            raise ValueError("query vector dimension differs from provenance")
        if self.vector.dtype != self.provenance.dtype:
            raise ValueError("query vector dtype differs from provenance")


@dataclass(frozen=True)
class PageVectorSnapshot:
    """One atomic availability view used for a single selection decision."""

    query: QueryVectorEntry | None
    pages: Mapping[str, PageVectorEntry]
    missing_page_ids: tuple[str, ...]
    mismatched_page_ids: tuple[str, ...]
    query_mismatch: bool = False

    @property
    def complete(self) -> bool:
        return (
            self.query is not None
            and not self.query_mismatch
            and not self.missing_page_ids
            and not self.mismatched_page_ids
        )


class PageVectorIndex:
    """Thread-safe in-memory V2 index with immutable sealed publications."""

    def __init__(self) -> None:
        self._pages: dict[tuple[str, str], PageVectorEntry] = {}
        self._queries: dict[tuple[str, int], QueryVectorEntry] = {}
        self._lock = RLock()

    def snapshot(
        self,
        session_id: str,
        turn_id: int,
        page_ids: Iterable[str],
        expected_provenance: PageVectorProvenance,
    ) -> PageVectorSnapshot:
        requested = tuple(dict.fromkeys(str(page_id) for page_id in page_ids))
        with self._lock:
            raw_query = self._queries.get((session_id, turn_id))
            query_mismatch = (
                raw_query is not None and raw_query.provenance != expected_provenance
            )
            query = None if query_mismatch else raw_query
            pages: dict[str, PageVectorEntry] = {}
            missing: list[str] = []
            mismatched: list[str] = []
            for page_id in requested:
                entry = self._pages.get((session_id, page_id))
                if entry is None:
                    missing.append(page_id)
                elif entry.provenance != expected_provenance:
                    mismatched.append(page_id)
                else:
                    pages[page_id] = entry
        return PageVectorSnapshot(
            query=query,
            pages=MappingProxyType(pages),
            missing_page_ids=tuple(missing),
            mismatched_page_ids=tuple(mismatched),
            query_mismatch=query_mismatch,
        )

    def release_session(self, session_id: str) -> None:
        with self._lock:
            self._pages = {
                key: value for key, value in self._pages.items() if key[0] != session_id
            }
            self._queries = {
                key: value for key, value in self._queries.items() if key[0] != session_id
            }

    def _publish_page(self, entry: PageVectorEntry) -> str:
        key = (entry.session_id, entry.page_id)
        with self._lock:
            previous = self._pages.get(key)
            if previous is not None and previous != entry:
                raise ValueError("a sealed physical Page vector cannot be overwritten")
            self._pages[key] = entry
        return entry.vector_ref

    def _publish_query(self, entry: QueryVectorEntry) -> None:
        key = (entry.session_id, entry.turn_id)
        with self._lock:
            previous = self._queries.get(key)
            if previous is not None and previous != entry:
                raise ValueError("an observation query vector cannot be overwritten")
            self._queries[key] = entry


class PageVectorPublisher:
    """Engine-side compact-vector publication boundary."""

    def __init__(
        self,
        index: PageVectorIndex,
        provenance: PageVectorProvenance,
    ) -> None:
        self.index = index
        self.provenance = provenance

    def publish_page(
        self,
        session_id: str,
        page_id: str,
        vector: CompactVector,
    ) -> str:
        return self.index._publish_page(PageVectorEntry(
            session_id, page_id, vector, self.provenance
        ))

    def publish_query(
        self,
        session_id: str,
        turn_id: int,
        vector: CompactVector,
    ) -> None:
        self.index._publish_query(QueryVectorEntry(
            session_id, turn_id, vector, self.provenance
        ))


class SimulatedPrefillPageVectorPublisher:
    """CPU-only test double for the future in-runner GPU implementation.

    It accepts already-produced hidden states, accumulates chunks, filters
    rejected speculative decode tokens, and publishes only after a Page is
    explicitly sealed.  It never invokes a model.
    """

    def __init__(
        self,
        index: PageVectorIndex,
        *,
        model: str,
        revision: str,
        hidden_layer: int,
        projection: Sequence[Sequence[float]],
        pooling: str = "masked_mean",
        dtype: str = "int8",
    ) -> None:
        rows = tuple(tuple(float(value) for value in row) for row in projection)
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("projection must be a non-empty rectangular matrix")
        if any(not math.isfinite(value) for row in rows for value in row):
            raise ValueError("projection values must be finite")
        projection_hash = _stable_hash(rows)
        self.provenance = PageVectorProvenance(
            model=model,
            revision=revision,
            hidden_layer=hidden_layer,
            pooling=pooling,
            projection_hash=projection_hash,
            dimension=len(rows),
            dtype=dtype,
            vector_source="simulated_main_model_hidden_states",
            quantization="symmetric_per_vector" if dtype == "int8" else "none",
        )
        self.publisher = PageVectorPublisher(index, self.provenance)
        self._projection = rows
        self._hidden_dimension = len(rows[0])
        self._page_chunks: dict[tuple[str, str], list[tuple[float, ...]]] = {}
        self._query_chunks: dict[tuple[str, int], list[tuple[float, ...]]] = {}

    def observe_page_chunk(
        self,
        session_id: str,
        page_id: str,
        hidden_states: Sequence[Sequence[float]],
        *,
        accepted: Sequence[bool] | None = None,
    ) -> int:
        states = self._states(hidden_states)
        if accepted is not None:
            flags = tuple(bool(flag) for flag in accepted)
            if len(flags) != len(states):
                raise ValueError("accepted mask must match hidden-state count")
            states = tuple(state for state, keep in zip(states, flags) if keep)
        key = (session_id, page_id)
        accumulated = self._page_chunks.get(key, [])
        if len(accumulated) + len(states) > self.provenance.page_size:
            raise ValueError("hidden-state chunks exceed one 16-token physical Page")
        accumulated.extend(states)
        self._page_chunks[key] = accumulated
        return len(accumulated)

    def seal_page(self, session_id: str, page_id: str) -> str:
        key = (session_id, page_id)
        states = self._page_chunks.get(key, [])
        if len(states) != self.provenance.page_size:
            raise ValueError("only a complete 16-token physical Page can be sealed")
        vector = self._compact(states)
        reference = self.publisher.publish_page(session_id, page_id, vector)
        del self._page_chunks[key]
        return reference

    def observe_query_chunk(
        self,
        session_id: str,
        turn_id: int,
        hidden_states: Sequence[Sequence[float]],
    ) -> int:
        states = self._states(hidden_states)
        key = (session_id, turn_id)
        values = self._query_chunks.setdefault(key, [])
        values.extend(states)
        return len(values)

    def seal_query(self, session_id: str, turn_id: int) -> None:
        key = (session_id, turn_id)
        states = self._query_chunks.get(key, [])
        if not states:
            raise ValueError("observation query requires at least one hidden state")
        self.publisher.publish_query(session_id, turn_id, self._compact(states))
        del self._query_chunks[key]

    def release_session(self, session_id: str) -> None:
        self._page_chunks = {
            key: value for key, value in self._page_chunks.items() if key[0] != session_id
        }
        self._query_chunks = {
            key: value for key, value in self._query_chunks.items() if key[0] != session_id
        }

    def _states(
        self, hidden_states: Sequence[Sequence[float]]
    ) -> tuple[tuple[float, ...], ...]:
        states = tuple(tuple(float(value) for value in state) for state in hidden_states)
        if any(len(state) != self._hidden_dimension for state in states):
            raise ValueError("hidden-state dimension differs from projection input")
        if any(not math.isfinite(value) for state in states for value in state):
            raise ValueError("hidden states must be finite")
        return states

    def _compact(self, states: Sequence[Sequence[float]]) -> CompactVector:
        if self.provenance.pooling == "last_token":
            pooled = tuple(float(value) for value in states[-1])
        else:
            pooled = tuple(
                sum(float(state[column]) for state in states) / len(states)
                for column in range(self._hidden_dimension)
            )
        projected = tuple(
            sum(weight * value for weight, value in zip(row, pooled))
            for row in self._projection
        )
        norm = math.sqrt(sum(value * value for value in projected))
        if not norm:
            raise ValueError("projection produced an all-zero vector")
        normalized = tuple(value / norm for value in projected)
        if self.provenance.dtype == "int8":
            peak = max(abs(value) for value in normalized)
            scale = peak / 127.0
            quantized = tuple(
                max(-127, min(127, int(round(value / scale))))
                for value in normalized
            )
            return CompactVector(quantized, "int8", scale)
        return CompactVector(normalized, self.provenance.dtype)


def compact_cosine(left: CompactVector, right: CompactVector) -> float:
    """Cosine over dequantized values with strict dimensionality checks."""

    if left.dimension != right.dimension:
        raise ValueError("compact vector dimensionality changed")
    left_values = left.unit_values()
    right_values = right.unit_values()
    return max(-1.0, min(1.0, sum(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values)
    )))


__all__ = [
    "CompactVector",
    "PageVectorEntry",
    "PageVectorIndex",
    "PageVectorProvenance",
    "PageVectorPublisher",
    "PageVectorSnapshot",
    "QueryVectorEntry",
    "SimulatedPrefillPageVectorPublisher",
    "compact_cosine",
]
