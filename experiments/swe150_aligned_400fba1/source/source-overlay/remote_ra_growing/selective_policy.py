# SPDX-License-Identifier: Apache-2.0
"""CPU-only selection and validation for request contract version 2."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import math
import re
from typing import Mapping, Protocol

CONTRACT_VERSION = 2
POLICY_NAMES = frozenset({"random", "observation_query", "goal_query"})
FractionInput = int | float | str | Fraction


class PageSelectionPolicyError(ValueError):
    """Unsafe or inconsistent input; caller must emit FALLBACK."""


def parse_fraction(value: object) -> Fraction:
    # JSON floats use their decimal spelling, never their binary expansion.
    if type(value) is float and not math.isfinite(value):
        raise PageSelectionPolicyError("fraction must be finite")
    if type(value) is str and not re.fullmatch(
        r"(?:[0-9]+/[1-9][0-9]*|(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)",
        value,
    ):
        raise PageSelectionPolicyError("fraction must be a decimal or integer ratio")
    if type(value) not in (int, float, str, Fraction):
        raise PageSelectionPolicyError("fraction must be numeric or an exact string")
    try:
        result = value if type(value) is Fraction else Fraction(str(value))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise PageSelectionPolicyError("invalid fraction") from exc
    if not 0 < result <= 1:
        raise PageSelectionPolicyError("fraction must be in (0, 1]")
    return result


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise PageSelectionPolicyError(f"{name} must be a positive int")
    return value


def exact_selected_count(num_pages: int, fraction: FractionInput) -> int:
    _positive_int(num_pages, "num_pages")
    ratio = parse_fraction(fraction)
    return num_pages * ratio.numerator // ratio.denominator


def _hex(value: object, name: str, length: int | None = None) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) % 2
        or (length is not None and len(value) != length)
        or any(c not in "0123456789abcdefABCDEF" for c in value)
    ):
        raise PageSelectionPolicyError(f"{name} must be a valid hex digest")
    return value.lower()


@dataclass(frozen=True, slots=True)
class SelectivePageControls:
    requested: bool = False
    policy: str = "random"
    fraction: Fraction | None = None
    seed: int = 42
    ranked_logical_pages: tuple[int, ...] | None = None
    query_hash: str | None = None
    contract_version: int | None = None
    expected_num_pages: int | None = None
    expected_page_tokens: int | None = None
    selected_count: int | None = None
    expected_prefix_identity: str | None = None
    dry_run_fraction: Fraction | None = None


def parse_request_controls(
    params: Mapping[str, object] | None
) -> SelectivePageControls:
    if params is None:
        return SelectivePageControls()
    if not isinstance(params, Mapping):
        raise PageSelectionPolicyError("kv_transfer_params must be an object")
    known = {
        "selective_page_policy",
        "selective_page_fraction",
        "selective_page_seed",
        "selective_page_ranked_pages",
        "selective_page_query_hash",
        "selective_page_contract_version",
        "selective_page_num_pages",
        "selective_page_page_tokens",
        "selective_page_selected_count",
        "selective_page_prefix_identity",
        "selective_page_dry_run_fraction",
    }
    unknown = [
        key
        for key in params
        if str(key).startswith("selective_page_") and key not in known
    ]
    if unknown:
        raise PageSelectionPolicyError(f"unknown selective controls: {unknown}")
    supplied = known.intersection(params)
    if not supplied:
        return SelectivePageControls()
    dry = (
        parse_fraction(params["selective_page_dry_run_fraction"])
        if "selective_page_dry_run_fraction" in params
        else None
    )
    active = bool(supplied - {"selective_page_dry_run_fraction", "selective_page_seed"})
    if not active and dry is None:
        raise PageSelectionPolicyError("selective controls require a selection budget")
    policy = params.get("selective_page_policy", "random")
    if type(policy) is not str or policy not in POLICY_NAMES:
        raise PageSelectionPolicyError("unknown selective_page_policy")
    seed = params.get("selective_page_seed", 42)
    if type(seed) is not int or seed < 0:
        raise PageSelectionPolicyError("selective_page_seed must be a non-negative int")
    version = params.get("selective_page_contract_version")
    if "selective_page_contract_version" in params and (
        type(version) is not int or version != CONTRACT_VERSION
    ):
        raise PageSelectionPolicyError("selective_page_contract_version must be int 2")
    ranked = policy != "random"
    required = {
        "selective_page_contract_version",
        "selective_page_num_pages",
        "selective_page_page_tokens",
        "selective_page_selected_count",
        "selective_page_prefix_identity",
        "selective_page_ranked_pages",
        "selective_page_query_hash",
    }
    if ranked and not required.issubset(params):
        raise PageSelectionPolicyError(
            f"ranked policy missing {sorted(required - params.keys())}"
        )
    fraction = (
        parse_fraction(params["selective_page_fraction"])
        if "selective_page_fraction" in params
        else None
    )
    values = {}
    for wire, local in (
        ("selective_page_num_pages", "expected_num_pages"),
        ("selective_page_page_tokens", "expected_page_tokens"),
        ("selective_page_selected_count", "selected_count"),
    ):
        values[local] = _positive_int(params[wire], wire) if wire in params else None
    if active and fraction is None and values["selected_count"] is None:
        raise PageSelectionPolicyError(
            "selective controls require fraction or selected_count"
        )
    ranking = None
    if "selective_page_ranked_pages" in params:
        raw = params["selective_page_ranked_pages"]
        if type(raw) not in (list, tuple) or any(
            type(i) is not int or i < 0 for i in raw
        ):
            raise PageSelectionPolicyError(
                "selective_page_ranked_pages must contain non-negative ints"
            )
        ranking = tuple(raw)
        if len(set(ranking)) != len(ranking):
            raise PageSelectionPolicyError("ranked page index appears more than once")
    query_hash = (
        _hex(params["selective_page_query_hash"], "selective_page_query_hash", 64)
        if "selective_page_query_hash" in params
        else None
    )
    prefix = (
        _hex(params["selective_page_prefix_identity"], "selective_page_prefix_identity")
        if "selective_page_prefix_identity" in params
        else None
    )
    if not ranked and (ranking is not None or query_hash is not None):
        raise PageSelectionPolicyError(
            "random policy cannot consume ranked/query controls"
        )
    n, count = values["expected_num_pages"], values["selected_count"]
    if n is not None:
        if count is not None and count > n:
            raise PageSelectionPolicyError("selected_count exceeds num_pages")
        if (
            fraction is not None
            and count is not None
            and exact_selected_count(n, fraction) != count
        ):
            raise PageSelectionPolicyError("fraction and selected_count disagree")
        if ranking is not None and (len(ranking) != n or set(ranking) != set(range(n))):
            raise PageSelectionPolicyError(
                "ranking must be a complete permutation of num_pages"
            )
    return SelectivePageControls(
        requested=active,
        policy=policy,
        fraction=fraction,
        seed=seed,
        ranked_logical_pages=ranking,
        query_hash=query_hash,
        contract_version=version,
        expected_prefix_identity=prefix,
        dry_run_fraction=dry,
        **values,
    )


@dataclass(frozen=True, slots=True)
class PageSelectionContext:
    num_pages: int
    fraction: FractionInput | None
    seed: int
    prefix_identity: bytes
    ranked_logical_pages: tuple[int, ...] | None = None
    query_hash: str | None = None
    page_tokens: int | None = None
    contract_version: int | None = None
    expected_num_pages: int | None = None
    expected_page_tokens: int | None = None
    expected_selected_count: int | None = None
    expected_prefix_identity: str | None = None

    @property
    def selected_count(self) -> int:
        if self.expected_selected_count is not None:
            return self.expected_selected_count
        if self.fraction is None:
            raise PageSelectionPolicyError("missing selection budget")
        return exact_selected_count(self.num_pages, self.fraction)


@dataclass(frozen=True, slots=True)
class PageSelection:
    policy: str
    logical_pages: tuple[int, ...]


class PageSelectionPolicy(Protocol):
    name: str

    def select(self, context: PageSelectionContext) -> PageSelection: ...


def _validate_context(context: PageSelectionContext) -> None:
    _positive_int(context.num_pages, "num_pages")
    if type(context.seed) is not int or context.seed < 0:
        raise PageSelectionPolicyError("seed must be a non-negative int")
    if not isinstance(context.prefix_identity, bytes) or not context.prefix_identity:
        raise PageSelectionPolicyError("prefix_identity must be nonempty bytes")
    if context.fraction is not None:
        parse_fraction(context.fraction)
    if context.page_tokens is not None:
        _positive_int(context.page_tokens, "page_tokens")
    if context.contract_version is not None and (
        type(context.contract_version) is not int
        or context.contract_version != CONTRACT_VERSION
    ):
        raise PageSelectionPolicyError("contract_version must be int 2")
    for expected, actual, name in (
        (context.expected_num_pages, context.num_pages, "num_pages"),
        (context.expected_page_tokens, context.page_tokens, "page_tokens"),
    ):
        if expected is not None and (_positive_int(expected, name) != actual):
            raise PageSelectionPolicyError(f"expected {name} does not match scheduler")
    if (
        context.expected_prefix_identity is not None
        and _hex(context.expected_prefix_identity, "prefix_identity")
        != context.prefix_identity.hex()
    ):
        raise PageSelectionPolicyError(
            "expected prefix_identity does not match scheduler"
        )
    count = context.selected_count
    _positive_int(count, "selected_count (zero-page selection is unsupported)")
    if count > context.num_pages:
        raise PageSelectionPolicyError("selected_count exceeds num_pages")
    if (
        context.fraction is not None
        and exact_selected_count(context.num_pages, context.fraction) != count
    ):
        raise PageSelectionPolicyError("fraction and selected_count disagree")


class RandomPageSelectionPolicy:
    """Original SHA-256 ordering, with exact selected-count arithmetic."""

    name = "random"

    def select(self, context: PageSelectionContext) -> PageSelection:
        _validate_context(context)
        if context.ranked_logical_pages is not None or context.query_hash is not None:
            raise PageSelectionPolicyError(
                "random policy cannot consume ranked/query controls"
            )
        seed_bytes = str(context.seed).encode("ascii")

        def score(page_idx: int) -> bytes:
            return sha256(
                seed_bytes
                + b"\0"
                + context.prefix_identity
                + page_idx.to_bytes(8, "big", signed=False)
            ).digest()

        ranked = sorted(range(context.num_pages), key=lambda idx: (score(idx), idx))
        return PageSelection(self.name, tuple(sorted(ranked[: context.selected_count])))


class _RankedQueryPageSelectionPolicy:
    name: str

    def select(self, context: PageSelectionContext) -> PageSelection:
        _validate_context(context)
        if (
            context.contract_version != CONTRACT_VERSION
            or context.expected_num_pages is None
            or context.expected_page_tokens is None
            or context.expected_selected_count is None
            or context.expected_prefix_identity is None
        ):
            raise PageSelectionPolicyError(
                "ranked policies require version, expected num_pages/page_tokens/selected_count/prefix_identity"
            )
        _hex(context.query_hash, "selective_page_query_hash", 64)
        ranking = context.ranked_logical_pages
        if type(ranking) not in (tuple, list) or any(
            type(i) is not int or i < 0 for i in ranking
        ):
            raise PageSelectionPolicyError(
                "selective_page_ranked_pages must contain non-negative ints"
            )
        if len(set(ranking)) != len(ranking):
            raise PageSelectionPolicyError("ranked page index appears more than once")
        if len(ranking) != context.num_pages or set(ranking) != set(
            range(context.num_pages)
        ):
            raise PageSelectionPolicyError(
                "ranking must be a complete permutation of num_pages"
            )
        return PageSelection(
            self.name, tuple(sorted(ranking[: context.selected_count]))
        )


class ObservationQueryPageSelectionPolicy(_RankedQueryPageSelectionPolicy):
    name = "observation_query"


class GoalQueryPageSelectionPolicy(_RankedQueryPageSelectionPolicy):
    name = "goal_query"


_POLICIES = {
    p.name: p
    for p in (
        RandomPageSelectionPolicy(),
        ObservationQueryPageSelectionPolicy(),
        GoalQueryPageSelectionPolicy(),
    )
}


def get_page_selection_policy(name: str) -> PageSelectionPolicy:
    if type(name) is not str or name not in _POLICIES:
        raise PageSelectionPolicyError(f"unknown selective_page_policy {name!r}")
    return _POLICIES[name]


def select_logical_pages(
    *,
    policy: str,
    num_pages: int,
    fraction: FractionInput | None = None,
    seed: int = 42,
    prefix_identity: bytes,
    ranked_logical_pages: tuple[int, ...] | None = None,
    query_hash: str | None = None,
    page_tokens: int | None = None,
    contract_version: int | None = None,
    expected_num_pages: int | None = None,
    expected_page_tokens: int | None = None,
    expected_selected_count: int | None = None,
    expected_prefix_identity: str | None = None,
) -> PageSelection:
    return get_page_selection_policy(policy).select(
        PageSelectionContext(
            num_pages,
            fraction,
            seed,
            prefix_identity,
            ranked_logical_pages,
            query_hash,
            page_tokens,
            contract_version,
            expected_num_pages,
            expected_page_tokens,
            expected_selected_count,
            expected_prefix_identity,
        )
    )
