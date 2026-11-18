"""Canonical renderer/catalog geometry and existing Random50 semantics.

Both development layers import this module. A rendered prompt includes the
current tool Observation and its template suffix. Only complete Pages strictly
before that suffix can enter persistent backing; the crossing partial Page,
Observation and generation header remain request-private for this turn.
"""
from dataclasses import dataclass
import re
from . import selective_policy as _legacy


@dataclass(frozen=True)
class HistoryGeometry:
    stable_end: int
    sealable_end: int
    num_pages: int
    observation_span: tuple[int, int] | None
    request_private_span: tuple[int, int]


def history_geometry(prompt_tokens, observation_span, page_tokens, previous_sealed=()):
    if (type(prompt_tokens) is not tuple or not prompt_tokens
            or any(type(t) is not int or t < 0 for t in prompt_tokens)
            or type(page_tokens) is not int or page_tokens <= 0):
        raise ValueError("actual rendered prompt tokens and positive Page geometry required")
    if (type(previous_sealed) is not tuple or len(previous_sealed) % page_tokens
            or prompt_tokens[:len(previous_sealed)] != previous_sealed):
        raise ValueError("sealed history changed instead of append-only growth")
    end = len(prompt_tokens)
    if observation_span is None:
        stable_end = end
    else:
        if (type(observation_span) is not tuple or len(observation_span) != 2
                or any(type(x) is not int for x in observation_span)
                or not 0 < observation_span[0]
                or not len(previous_sealed) <= observation_span[0] < observation_span[1] == end):
            raise ValueError("Observation must be the exact new rendered tool-output/template suffix")
        stable_end = observation_span[0]
    sealed_end = stable_end // page_tokens * page_tokens
    return HistoryGeometry(stable_end, sealed_end, sealed_end // page_tokens,
                           observation_span, (sealed_end, end))


def sparse_count(n):
    if type(n) is not int or n < 0:
        raise ValueError("nonnegative Page count required")
    return n // 2


def select_random50(n, native_prefix_identity, seed=42):
    if type(seed) is not int or seed != 42:
        raise ValueError("Random50 contract requires seed 42")
    if type(native_prefix_identity) is not str or not re.fullmatch(r"[a-f0-9]{64}", native_prefix_identity):
        raise ValueError("actual native prefix SHA256 identity required")
    if n < 2:
        raise ValueError("short histories require explicit Full bootstrap")
    result = _legacy.select_logical_pages(policy="random", num_pages=n, fraction="1/2",
        seed=seed, prefix_identity=bytes.fromhex(native_prefix_identity)).logical_pages
    if len(result) != sparse_count(n):
        raise ValueError("existing Random50 budget changed")
    return tuple(result)
