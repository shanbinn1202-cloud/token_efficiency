# SPDX-License-Identifier: Apache-2.0
"""Goal v1.3 contract core for the feature-gated runtime integration.

This is intentionally not a drop-in replacement for the v1.2 runner hook.
Only a trusted scheduler adapter may create PrefixHandoff objects. Their
hashes cannot be synthesized from the client prompt or a session identifier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import secrets

SCHEMA_VERSION = "1.3.0"
CONTRACT_VERSION = 2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def hash64(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def block_hash(value):
    # vLLM hashing configuration determines length. Preserve actual bytes.hex().
    if not isinstance(value, str) or not re.fullmatch(r"(?:[0-9a-f]{2})+", value):
        raise ValueError("prefix_identity must be actual nonempty block-hash hex")
    return value


@dataclass(frozen=True)
class ProducerProvenance:
    model: str
    revision: str
    hidden_layer: int
    projection_hash: str
    dimension: int = 128
    page_tokens: int = 784
    pooling: str = "last_token"
    dtype: str = "int8"
    vector_source: str = "vllm_existing_forward_final_hidden_postnorm"
    producer_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model or not isinstance(self.revision, str) or not self.revision:
            raise ValueError("model and immutable revision required")
        integer(self.hidden_layer, "hidden_layer")
        integer(self.dimension, "dimension", 1)
        integer(self.page_tokens, "page_tokens", 1)
        hash64(self.projection_hash, "projection_hash")
        if (self.pooling, self.dtype, self.vector_source, self.producer_version) != (
            "last_token", "int8", "vllm_existing_forward_final_hidden_postnorm", SCHEMA_VERSION
        ):
            raise ValueError("unsupported producer provenance")

    @property
    def provenance_hash(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class CompactVector:
    values: tuple[int, ...]
    scale: float

    def __post_init__(self):
        if type(self.values) is not tuple or not self.values or any(type(v) is not int or not -127 <= v <= 127 for v in self.values):
            raise ValueError("vector must contain int8 integers")
        if not any(self.values) or type(self.scale) not in (float, int) or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("invalid vector scale or zero vector")

    @property
    def content_hash(self):
        return digest(asdict(self))

    def unit_values(self):
        # Positive scale cancels; normalize integers to avoid overflow.
        norm = math.sqrt(sum(v * v for v in self.values))
        return tuple(v / norm for v in self.values)


def compact_cosine(left, right):
    if len(left.values) != len(right.values):
        raise ValueError("vector dimensions differ")
    return sum(a * b for a, b in zip(left.unit_values(), right.unit_values()))


@dataclass(frozen=True)
class PrefixHandoff:
    """Trusted scheduler record, bound to one *upcoming* agent request.

    page_prefix_hashes are hashes of cumulative rendered token prefixes at
    each page boundary, including all attention-affecting context. They are
    not hashes of individual pages. prefix_identity is the scheduler's actual
    matched-prefix block hash, with its native bytes converted to hex.
    """
    session_id: str
    generation: str
    turn_id: int
    request_nonce: str
    prompt_hash: str
    prefix_identity: str
    num_pages: int
    page_tokens: int
    page_prefix_hashes: tuple[str, ...]
    provenance_hash: str

    def __post_init__(self):
        if not isinstance(self.session_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", self.session_id):
            raise ValueError("invalid session_id")
        for name in ("generation", "request_nonce", "prompt_hash", "provenance_hash"):
            hash64(getattr(self, name), name)
        block_hash(self.prefix_identity)
        integer(self.turn_id, "turn_id", 1)
        integer(self.num_pages, "num_pages", 1)
        integer(self.page_tokens, "page_tokens", 1)
        if type(self.page_prefix_hashes) is not tuple or len(self.page_prefix_hashes) != self.num_pages:
            raise ValueError("complete page-prefix universe required")
        for value in self.page_prefix_hashes:
            hash64(value, "page_prefix_hash")
        if len(set(self.page_prefix_hashes)) != self.num_pages:
            raise ValueError("duplicate page-prefix identity")

    @property
    def binding_hash(self):
        return digest(asdict(self))


def fraction_value(value):
    # Count is authoritative. Fractions, if supplied, must be exact rationals.
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", value):
        raise ValueError("fraction must be an exact rational string")
    result = Fraction(value)
    if not 0 < result <= 1:
        raise ValueError("fraction outside (0,1]")
    return result


def query_identity(binding_hash, goal_hash, query_payload_hash, query_vector_hash, provenance_hash):
    return digest(dict(binding_hash=binding_hash, goal_hash=goal_hash,
                       query_payload_hash=query_payload_hash, query_vector_hash=query_vector_hash,
                       provenance_hash=provenance_hash))


def validate_ranking(value, *, expected_handoff, expected_provenance,
                     expected_goal_hash, expected_query_payload_hash, expected_selected_count):
    """Validate every sealed field against caller/scheduler expectations.

    SHA-256 is integrity, not authentication. Artifacts must travel through
    authenticated transport or a trusted local directory. No optional
    provenance or identity checks are permitted.
    """
    if not isinstance(expected_handoff, PrefixHandoff) or not isinstance(expected_provenance, ProducerProvenance):
        raise ValueError("trusted scheduler handoff and provenance required")
    required = {"schema_name", "schema_version", "handoff", "binding_hash", "provenance",
                "provenance_hash", "goal_hash", "query_payload_hash", "query_vector_hash", "query_hash",
                "num_pages", "page_tokens", "selected_count", "ranked_logical_pages",
                "mandatory_pages", "recent_pages", "page_vector_hashes", "kv_transfer_params", "seal_hash"}
    if type(value) is not dict or set(value) - {"fraction"} != required:
        raise ValueError("artifact schema fields differ")
    if value["schema_name"] != "vllm_goal_query_ranking" or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported artifact schema/version")
    hash64(value["seal_hash"], "seal_hash")
    if value["seal_hash"] != digest({k: v for k, v in value.items() if k != "seal_hash"}):
        raise ValueError("artifact seal mismatch")
    if canonical(value["handoff"]) != canonical(asdict(expected_handoff)) or value["binding_hash"] != expected_handoff.binding_hash:
        raise ValueError("stale prompt, session, turn, or prefix binding")
    if canonical(value["provenance"]) != canonical(asdict(expected_provenance)) or value["provenance_hash"] != expected_provenance.provenance_hash:
        raise ValueError("stale producer provenance")
    if expected_handoff.provenance_hash != expected_provenance.provenance_hash or expected_handoff.page_tokens != expected_provenance.page_tokens:
        raise ValueError("handoff provenance mismatch")
    for name in ("goal_hash", "query_payload_hash", "query_vector_hash", "query_hash", "binding_hash", "provenance_hash"):
        hash64(value[name], name)
    if value["goal_hash"] != hash64(expected_goal_hash, "expected_goal_hash") or value["query_payload_hash"] != hash64(expected_query_payload_hash, "expected_query_payload_hash"):
        raise ValueError("stale Goal query")
    expected_query = query_identity(value["binding_hash"], value["goal_hash"], value["query_payload_hash"], value["query_vector_hash"], value["provenance_hash"])
    if value["query_hash"] != expected_query:
        raise ValueError("query hash does not bind lineage and provenance")
    n = integer(value["num_pages"], "num_pages", 1)
    page_tokens = integer(value["page_tokens"], "page_tokens", 1)
    count = integer(value["selected_count"], "selected_count", 1)
    integer(expected_selected_count, "expected_selected_count", 1)
    if n != expected_handoff.num_pages or page_tokens != expected_handoff.page_tokens or count != expected_selected_count or count > n:
        raise ValueError("universe or selected count mismatch")
    ranking = value["ranked_logical_pages"]
    if type(ranking) is not list or any(type(p) is not int for p in ranking) or sorted(ranking) != list(range(n)):
        raise ValueError("ranking must be an exact complete page permutation")
    for name in ("mandatory_pages", "recent_pages"):
        pages = value[name]
        if type(pages) is not list or any(type(p) is not int for p in pages) or len(set(pages)) != len(pages) or not set(pages) <= set(ranking[:count]):
            raise ValueError("safety pages invalid or outside selected budget")
    hashes = value["page_vector_hashes"]
    if type(hashes) is not list or len(hashes) != n:
        raise ValueError("incomplete page-vector provenance")
    for item in hashes:
        hash64(item, "page_vector_hash")
    params = value["kv_transfer_params"]
    expected_params = dict(selective_page_contract_version=CONTRACT_VERSION,
                           selective_page_policy="goal_query", selective_page_num_pages=n,
                           selective_page_page_tokens=page_tokens, selective_page_selected_count=count,
                           selective_page_prefix_identity=expected_handoff.prefix_identity,
                           selective_page_ranked_pages=ranking, selective_page_query_hash=value["query_hash"])
    if type(params) is not dict:
        raise ValueError("kv_transfer_params must be an object")
    if "selective_page_seed" in params:
        integer(params["selective_page_seed"], "seed")
        expected_params["selective_page_seed"] = params["selective_page_seed"]
    if "fraction" in value or "selective_page_fraction" in params:
        if "fraction" not in value or "selective_page_fraction" not in params:
            raise ValueError("fraction fields must agree")
        if fraction_value(value["fraction"]) != Fraction(count, n) or params["selective_page_fraction"] != value["fraction"]:
            raise ValueError("fraction inconsistent with authoritative count")
        expected_params["selective_page_fraction"] = value["fraction"]
    if canonical(params) != canonical(expected_params):
        raise ValueError("consumer v2 contract mismatch")
    return value


class GoalProducerStore:
    def __init__(self, provenance):
        self.provenance = provenance
        self._sessions = {}

    def open_session(self, session_id, *, generation=None):
        if session_id in self._sessions:
            raise ValueError("session already live; release before reusing")
        if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", session_id):
            raise ValueError("invalid session_id")
        generation = secrets.token_hex(32) if generation is None else hash64(generation, "generation")
        self._sessions[session_id] = dict(generation=generation, handoff=None, pages={}, goals={}, consumed=False)
        return generation

    def _session(self, handoff, *, current=True):
        if not isinstance(handoff, PrefixHandoff):
            raise ValueError("trusted scheduler handoff required")
        state = self._sessions.get(handoff.session_id)
        if state is None or state["generation"] != handoff.generation:
            raise ValueError("released or stale session generation")
        if handoff.provenance_hash != self.provenance.provenance_hash or handoff.page_tokens != self.provenance.page_tokens:
            raise ValueError("handoff producer provenance mismatch")
        if current and state["handoff"] != handoff:
            raise ValueError("handoff is not current")
        if current and state["consumed"]:
            raise ValueError("request handoff already consumed")
        return state

    def register_handoff(self, handoff, *, lifecycle="append"):
        state = self._session(handoff, current=False)
        previous = state["handoff"]
        if lifecycle not in {"append", "replace"}:
            raise ValueError("branch must use a new session; lifecycle must be append/replace")
        if previous == handoff:
            return
        if previous is not None:
            if handoff.turn_id <= previous.turn_id or handoff.request_nonce == previous.request_nonce:
                raise ValueError("turn/nonce replay")
            if lifecycle == "append":
                if handoff.num_pages < previous.num_pages or handoff.page_prefix_hashes[:previous.num_pages] != previous.page_prefix_hashes:
                    raise ValueError("mutation/truncation requires explicit replace")
        if lifecycle == "replace":
            state["pages"].clear()
        # Reuse only vectors with exactly matching cumulative prefix identity.
        state["pages"] = {p: v for p, v in state["pages"].items()
                          if p < handoff.num_pages and v[0] == handoff.page_prefix_hashes[p]}
        state["goals"].clear()
        state["handoff"] = handoff
        state["consumed"] = False

    def publish_page(self, handoff, page_index, vector):
        state = self._session(handoff)
        integer(page_index, "page_index")
        if page_index >= handoff.num_pages or not isinstance(vector, CompactVector) or len(vector.values) != self.provenance.dimension:
            raise ValueError("page outside declared universe or vector dimension mismatch")
        record = (handoff.page_prefix_hashes[page_index], vector)
        if page_index in state["pages"] and state["pages"][page_index] != record:
            raise ValueError("sealed page vector conflict")
        state["pages"][page_index] = record

    def publish_goal(self, handoff, *, goal_hash, query_payload_hash, vector):
        state = self._session(handoff)
        key = (hash64(goal_hash, "goal_hash"), hash64(query_payload_hash, "query_payload_hash"))
        if not isinstance(vector, CompactVector) or len(vector.values) != self.provenance.dimension:
            raise ValueError("Goal vector dimension mismatch")
        if key in state["goals"] and state["goals"][key] != vector:
            raise ValueError("sealed Goal vector conflict")
        state["goals"][key] = vector

    def build_ranking(self, handoff, *, goal_hash, query_payload_hash, selected_count, mandatory_pages=(), recent_pages=()):
        state = self._session(handoff)
        n = handoff.num_pages
        integer(selected_count, "selected_count", 1)
        if selected_count > n:
            raise ValueError("selected count exceeds universe")
        if set(state["pages"]) != set(range(n)):
            raise ValueError("incomplete page universe, including cached-page gaps")
        try:
            query = state["goals"][(goal_hash, query_payload_hash)]
        except KeyError as exc:
            raise ValueError("Goal vector missing for current handoff") from exc
        safety = []
        for pages in (mandatory_pages, recent_pages):
            if len(set(pages)) != len(pages) or any(type(p) is not int or not 0 <= p < n for p in pages):
                raise ValueError("invalid safety pages")
            safety.extend(p for p in pages if p not in safety)
        if len(safety) > selected_count:
            raise ValueError("safety pages exceed budget")
        ranking = safety + sorted((p for p in range(n) if p not in safety),
                                  key=lambda p: (-compact_cosine(state["pages"][p][1], query), p))
        qhash = query_identity(handoff.binding_hash, goal_hash, query_payload_hash, query.content_hash, self.provenance.provenance_hash)
        value = dict(schema_name="vllm_goal_query_ranking", schema_version=SCHEMA_VERSION,
                     handoff=asdict(handoff), binding_hash=handoff.binding_hash,
                     provenance=asdict(self.provenance), provenance_hash=self.provenance.provenance_hash,
                     goal_hash=goal_hash, query_payload_hash=query_payload_hash, query_vector_hash=query.content_hash,
                     query_hash=qhash, num_pages=n, page_tokens=handoff.page_tokens, selected_count=selected_count,
                     ranked_logical_pages=ranking, mandatory_pages=list(mandatory_pages), recent_pages=list(recent_pages),
                     page_vector_hashes=[state["pages"][p][1].content_hash for p in range(n)],
                     kv_transfer_params=dict(selective_page_contract_version=CONTRACT_VERSION,
                                             selective_page_policy="goal_query", selective_page_num_pages=n,
                                             selective_page_page_tokens=handoff.page_tokens,
                                             selective_page_selected_count=selected_count,
                                             selective_page_prefix_identity=handoff.prefix_identity,
                                             selective_page_ranked_pages=ranking, selective_page_query_hash=qhash))
        value["seal_hash"] = digest(value)
        return validate_ranking(value, expected_handoff=handoff, expected_provenance=self.provenance,
                                expected_goal_hash=goal_hash, expected_query_payload_hash=query_payload_hash,
                                expected_selected_count=selected_count)

    def release_session(self, session_id, generation):
        state = self._sessions.get(session_id)
        if state is None or state["generation"] != generation:
            raise ValueError("session release generation mismatch")
        del self._sessions[session_id]

    def validate_and_consume(self, handoff, ranking, *, actual_prefix_identity,
                             actual_prompt_hash, goal_hash, query_payload_hash, selected_count):
        """Scheduler-side one-use commit, under its scheduling lock.

        Called immediately before restore planning, using the *current* actual
        scheduler values. The caller must keep match+commit+planning atomic.
        A stale match consumes nothing and causes no selective restore.
        """
        state = self._session(handoff)
        if block_hash(actual_prefix_identity) != handoff.prefix_identity or hash64(actual_prompt_hash, "actual_prompt_hash") != handoff.prompt_hash:
            raise ValueError("scheduler match changed; obtain a new handoff")
        validated = validate_ranking(ranking, expected_handoff=handoff,
                                     expected_provenance=self.provenance,
                                     expected_goal_hash=goal_hash,
                                     expected_query_payload_hash=query_payload_hash,
                                     expected_selected_count=selected_count)
        state["consumed"] = True
        return dict(validated["kv_transfer_params"])


def artifact_path(directory, handoff, goal_hash, query_payload_hash):
    hash64(goal_hash, "goal_hash")
    hash64(query_payload_hash, "query_payload_hash")
    # Hash the compound identity to stay below Windows path/component limits.
    identity = digest([handoff.binding_hash, goal_hash, query_payload_hash])
    return Path(directory) / f"{identity}.ranking.json"


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def read_sealed_ranking_artifact(path, **expectations):
    expected = artifact_path(Path(path).parent, expectations["expected_handoff"],
                             expectations["expected_goal_hash"], expectations["expected_query_payload_hash"])
    if Path(path) != expected:
        raise ValueError("artifact filename identity mismatch")
    raw = Path(path).read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("incomplete sealed artifact")
    value = json.loads(raw, object_pairs_hook=_no_duplicate_keys,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    return validate_ranking(value, **expectations)


def write_sealed_ranking_artifact(directory, value, **expectations):
    validate_ranking(value, **expectations)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final = artifact_path(directory, expectations["expected_handoff"], expectations["expected_goal_hash"], expectations["expected_query_payload_hash"])
    content = (canonical(value) + "\n").encode()
    temporary = final.with_name(final.name + ".tmp." + secrets.token_hex(16))
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, final)
        except FileExistsError:
            if final.read_bytes() != content:
                raise ValueError("sealed artifact cannot be overwritten")
    finally:
        temporary.unlink(missing_ok=True)
    return final

