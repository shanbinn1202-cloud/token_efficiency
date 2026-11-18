"""Deterministic, dependency-free features for Goal maintenance.

The extractor deliberately operates on only the latest action and tool
observation.  Action and observation evidence stay separate so downstream
rules cannot accidentally treat text printed by a tool as an agent action (or
as a user constraint).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


_SOURCE_EXTENSIONS = (
    "py|pyi|js|jsx|ts|tsx|java|go|rs|c|cc|cpp|cxx|h|hpp|"
    "json|ya?ml|toml|md|rst|sh|bash|sql"
)
_PATH_RE = re.compile(
    rf"(?<![\w/.-])((?:[A-Za-z]:)?[A-Za-z0-9_@+.,/\\-]+\.(?:{_SOURCE_EXTENSIONS}))"
    r"(?=$|[^\w/-])",
    re.IGNORECASE,
)
_TEST_NODE_RE = re.compile(
    r"(?<![\w/.-])([A-Za-z0-9_./\\-]+\.py(?:::[A-Za-z0-9_\[\].:/-]+)+)"
)
_NAMED_TEST_RE = re.compile(
    r"\b((?:test|should)_[A-Za-z0-9_]+|[A-Za-z_][A-Za-z0-9_]*Test\b)"
)
_ERROR_LINE_RE = re.compile(
    r"(?m)^\s*(?:E\s+)?([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Failure|Fault)"
    r"(?:\s*:\s*[^\n]*)?)\s*$"
)
_ERROR_TYPE_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Failure|Fault))\b"
)
_PYTHON_FRAME_RE = re.compile(
    r'File\s+["\']([^"\']+)["\']\s*,\s*line\s+\d+\s*,\s*in\s+([A-Za-z_][A-Za-z0-9_.<>]*)'
)
_GENERIC_FRAME_RE = re.compile(
    r"(?m)^\s*(?:at\s+|#\d+\s+)([A-Za-z_][A-Za-z0-9_.$:<>-]*)"
    r"(?:\s+\([^\n]*\)|\s+at\s+[^\n]*)?\s*$"
)
_DEF_SYMBOL_RE = re.compile(
    r"\b(?:def|class|function|func|fn)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_CALL_SYMBOL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*)\s*\(")
_BACKTICK_SYMBOL_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_.:]*)`")

_EDIT_RE = re.compile(
    r"(?:\bapply_patch\b|\bpatch\s+-p\d*\b|\bsed\s+[^\n]*\s-i(?:\s|$)|"
    r"\b(?:vim|vi|nano)\s+|\bperl\s+[^\n]*\s-i(?:\s|$)|"
    r"(?:^|[;&|]\s*)cat\s+[^\n]*(?:>|>>)|(?:^|[;&|]\s*)tee\s+)",
    re.IGNORECASE,
)
_TEST_RE = re.compile(
    r"(?:\bpytest\b|\bpython\s+-m\s+(?:pytest|unittest)\b|\bunittest\b|"
    r"\b(?:cargo|go)\s+test\b|\bnpm\s+(?:run\s+)?test\b|\bctest\b|"
    r"\b(?:make|ninja)\s+(?:check|test)\b)",
    re.IGNORECASE,
)
_INSPECT_RE = re.compile(
    r"(?:^|[;&|]\s*)(?:cat|head|tail|less|more|rg|grep|find|ls|tree|wc|"
    r"git\s+(?:show|log|status|blame)|sed\s+-n)\b",
    re.IGNORECASE,
)
_DIAGNOSE_RE = re.compile(
    r"(?:\bgdb\b|\blldb\b|\bstrace\b|\bpdb\b|\bdebug\b|\btraceback\b|"
    r"\broot[ -]?cause\b|\bdiagnos(?:e|is|tic)\b|\bgit\s+diff\b)",
    re.IGNORECASE,
)
_DIFF_RE = re.compile(
    r"(?m)^(?:diff --git |index [0-9a-f]+\.\.[0-9a-f]+|--- (?:a/|/dev/null)|"
    r"\+\+\+ (?:b/|/dev/null)|@@\s+-\d)",
)
_PATCH_SUCCESS_RE = re.compile(
    r"(?im)^\s*(?:Done!|Applied patch|Patch applied successfully)\s*$"
)
_RETURN_CODE_RES = (
    re.compile(r"(?im)^\s*(?:return\s*code|returncode|exit\s*code)\s*[:=]\s*(-?\d+)\s*$"),
    re.compile(r"(?im)^\s*process\s+(?:exited|finished)\s+with\s+(?:code|status)\s+(-?\d+)\s*$"),
)
_FAILED_TEST_RE = re.compile(
    r"(?im)(?:^FAILED\b|\b\d+\s+failed\b|\bFAILURES?\b|\btests?\s+failed\b)"
)
_PASSED_TEST_RE = re.compile(
    r"(?im)(?:\b\d+\s+passed\b|^OK\s*$|\btests?\s+passed\b|\bBUILD SUCCESSFUL\b)"
)

_SHELL_PREFIX_RE = re.compile(r"^\s*(?:\$|#)\s+")
_QUOTED_LITERAL_RE = re.compile(r"(?s)(['\"])(?:\\.|(?!\1).)*\1")
_NUMBER_RE = re.compile(r"(?<![A-Za-z_])(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)?)(?![A-Za-z_])")
_SPACE_RE = re.compile(r"\s+")

_SYMBOL_STOPWORDS = frozenset(
    {
        "if", "for", "while", "return", "print", "len", "str", "int",
        "list", "dict", "set", "tuple", "assert", "raise", "super",
    }
)


@dataclass(frozen=True)
class ExtractionLimits:
    """Hard bounds that make feature size independent of tool output size."""

    max_files: int = 16
    max_symbols: int = 16
    max_tests: int = 12
    max_errors: int = 8
    max_top_frames: int = 3
    max_item_chars: int = 240
    max_action_chars: int = 512
    max_observation_chars: int = 1024

    def __post_init__(self) -> None:
        if any(value <= 0 for value in asdict(self).values()):
            raise ValueError("all extraction limits must be positive")


@dataclass(frozen=True)
class FeatureEvidence:
    """Entities extracted from one explicitly identified provenance."""

    provenance: str
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    top_frames: tuple[str, ...] = ()
    excerpt: str = ""

    def __post_init__(self) -> None:
        if self.provenance not in {"action", "tool_observation"}:
            raise ValueError(f"unsupported feature provenance {self.provenance!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureSnapshot:
    """Frozen features for one turn boundary.

    No constraint field exists by design: only an authenticated user/system
    message may add constraints, while this extractor sees agent/tool text.
    """

    action: FeatureEvidence
    observation: FeatureEvidence
    normalized_action: str
    action_kind: str
    return_code: int | None
    test_verdict: str
    has_diff: bool
    tool_name: str | None = None

    def __post_init__(self) -> None:
        if self.action.provenance != "action":
            raise ValueError("action evidence must have action provenance")
        if self.observation.provenance != "tool_observation":
            raise ValueError("observation evidence must have tool_observation provenance")
        if self.action_kind not in {"edit", "test", "diagnose", "inspect", "other"}:
            raise ValueError(f"unsupported action kind {self.action_kind!r}")
        if self.test_verdict not in {"passed", "failed", "unknown", "not_run"}:
            raise ValueError(f"unsupported test verdict {self.test_verdict!r}")

    @property
    def files(self) -> tuple[str, ...]:
        return _merge(self.action.files, self.observation.files)

    @property
    def symbols(self) -> tuple[str, ...]:
        return _merge(self.action.symbols, self.observation.symbols)

    @property
    def tests(self) -> tuple[str, ...]:
        return _merge(self.action.tests, self.observation.tests)

    @property
    def errors(self) -> tuple[str, ...]:
        return _merge(self.action.errors, self.observation.errors)

    @property
    def confirmed_errors(self) -> tuple[str, ...]:
        """Failure-signature errors, excluding successful-output code mentions."""

        if self.failure_evidence_confirmed:
            return self.observation.errors
        return ()

    @property
    def failure_evidence_confirmed(self) -> bool:
        """Whether observation failures are strong enough to drive Goal gates.

        A structured successful return code is authoritative.  When no return
        code is available, require a complete traceback shape rather than an
        exception-name or stack-frame mention in inspected source text.
        """

        if self.return_code is not None:
            return self.return_code != 0
        if self.test_verdict == "failed":
            return True
        excerpt = self.observation.excerpt.casefold()
        return bool(
            self.observation.errors
            and self.observation.top_frames
            and "traceback (most recent call last)" in excerpt
        )

    @property
    def top_frames(self) -> tuple[str, ...]:
        return _merge(self.action.top_frames, self.observation.top_frames)

    @property
    def top_frame(self) -> str | None:
        frames = self.top_frames
        return frames[0] if frames else None

    @property
    def confirmed_top_frames(self) -> tuple[str, ...]:
        return self.observation.top_frames if self.failure_evidence_confirmed else ()

    @property
    def confirmed_top_frame(self) -> str | None:
        frames = self.confirmed_top_frames
        return frames[0] if frames else None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["files"] = list(self.files)
        result["symbols"] = list(self.symbols)
        result["tests"] = list(self.tests)
        result["errors"] = list(self.errors)
        result["confirmed_errors"] = list(self.confirmed_errors)
        result["failure_evidence_confirmed"] = self.failure_evidence_confirmed
        result["top_frames"] = list(self.top_frames)
        result["top_frame"] = self.top_frame
        result["confirmed_top_frames"] = list(self.confirmed_top_frames)
        result["confirmed_top_frame"] = self.confirmed_top_frame
        return result


class DeterministicFeatureExtractor:
    """Extract a bounded :class:`FeatureSnapshot` from the latest boundary."""

    def __init__(self, limits: ExtractionLimits | None = None) -> None:
        self.limits = limits or ExtractionLimits()

    def extract(
        self,
        action: str | None,
        observation: str | None,
        *,
        tool_name: str | None = None,
        return_code: int | None = None,
    ) -> FeatureSnapshot:
        action_text = str(action or "")
        observation_text = str(observation or "")
        normalized = normalize_action(action_text)
        action_kind = classify_action(action_text, tool_name=tool_name)
        resolved_return_code = (
            return_code if return_code is not None else _extract_return_code(observation_text)
        )
        action_evidence = _extract_evidence(
            action_text, "action", self.limits, self.limits.max_action_chars
        )
        observation_evidence = _extract_evidence(
            observation_text,
            "tool_observation",
            self.limits,
            self.limits.max_observation_chars,
        )
        test_verdict = _test_verdict(
            action_kind, observation_text, resolved_return_code
        )
        has_diff = bool(
            _DIFF_RE.search(action_text)
            or _DIFF_RE.search(observation_text)
            or (action_kind == "edit" and _PATCH_SUCCESS_RE.search(observation_text))
        )
        return FeatureSnapshot(
            action=action_evidence,
            observation=observation_evidence,
            normalized_action=normalized,
            action_kind=action_kind,
            return_code=resolved_return_code,
            test_verdict=test_verdict,
            has_diff=has_diff,
            tool_name=tool_name,
        )


def extract_features(
    action: str | None,
    observation: str | None,
    *,
    tool_name: str | None = None,
    return_code: int | None = None,
    limits: ExtractionLimits | None = None,
) -> FeatureSnapshot:
    """Convenience wrapper for callers that do not retain an extractor."""

    return DeterministicFeatureExtractor(limits).extract(
        action, observation, tool_name=tool_name, return_code=return_code
    )


def normalize_action(action: str | None) -> str:
    """Normalize volatile shell literals without erasing file/test identity."""

    text = _SHELL_PREFIX_RE.sub("", str(action or "").strip())
    text = _QUOTED_LITERAL_RE.sub("<str>", text)
    text = _NUMBER_RE.sub("<num>", text)
    return _SPACE_RE.sub(" ", text).strip().lower()


def classify_action(action: str | None, *, tool_name: str | None = None) -> str:
    """Classify the action, with modification deliberately taking precedence."""

    text = " ".join(part for part in (tool_name or "", str(action or "")) if part)
    if _EDIT_RE.search(text) or (tool_name or "").lower() in {
        "apply_patch", "write_file", "edit_file", "replace_file"
    }:
        return "edit"
    if _TEST_RE.search(text):
        return "test"
    if _DIAGNOSE_RE.search(text):
        return "diagnose"
    if _INSPECT_RE.search(text):
        return "inspect"
    return "other"


def _extract_evidence(
    text: str, provenance: str, limits: ExtractionLimits, excerpt_limit: int
) -> FeatureEvidence:
    files = _bounded_unique(
        (_normalize_path(match.group(1)) for match in _PATH_RE.finditer(text)),
        limits.max_files,
        limits.max_item_chars,
    )
    tests = _bounded_unique(
        (
            *(_normalize_path(match.group(1)) for match in _TEST_NODE_RE.finditer(text)),
            *(match.group(1) for match in _NAMED_TEST_RE.finditer(text)),
        ),
        limits.max_tests,
        limits.max_item_chars,
    )
    # Python prints frames outermost-to-innermost; the innermost frame is the
    # useful "top frame" for invalidation checks.
    python_frames_in_trace_order = tuple(
        f"{_normalize_path(match.group(1))}:{match.group(2)}"
        for match in _PYTHON_FRAME_RE.finditer(text)
    )
    python_frames = tuple(reversed(python_frames_in_trace_order))
    generic_frames = tuple(match.group(1) for match in _GENERIC_FRAME_RE.finditer(text))
    top_frames = _bounded_unique(
        (*python_frames, *generic_frames),
        limits.max_top_frames,
        limits.max_item_chars,
    )

    error_lines = tuple(_compact(match.group(1)) for match in _ERROR_LINE_RE.finditer(text))
    error_types = tuple(match.group(1) for match in _ERROR_TYPE_RE.finditer(text))
    errors = _bounded_unique(
        (*error_lines, *error_types), limits.max_errors, limits.max_item_chars
    )

    frame_symbols = tuple(frame.rsplit(":", 1)[-1] for frame in python_frames)
    candidates = (
        *frame_symbols,
        *(match.group(1) for match in _DEF_SYMBOL_RE.finditer(text)),
        *(match.group(1) for match in _BACKTICK_SYMBOL_RE.finditer(text)),
        *(match.group(1) for match in _CALL_SYMBOL_RE.finditer(text)),
    )
    symbols = _bounded_unique(
        (symbol for symbol in candidates if symbol.rsplit(".", 1)[-1] not in _SYMBOL_STOPWORDS),
        limits.max_symbols,
        limits.max_item_chars,
    )
    return FeatureEvidence(
        provenance=provenance,
        files=files,
        symbols=symbols,
        tests=tests,
        errors=errors,
        top_frames=top_frames,
        excerpt=_clip(_compact(text), excerpt_limit),
    )


def _extract_return_code(observation: str) -> int | None:
    for pattern in _RETURN_CODE_RES:
        match = pattern.search(observation)
        if match:
            return int(match.group(1))
    return None


def _test_verdict(action_kind: str, observation: str, return_code: int | None) -> str:
    looks_like_test = action_kind == "test" or bool(
        _FAILED_TEST_RE.search(observation) or _PASSED_TEST_RE.search(observation)
    )
    if not looks_like_test:
        return "not_run"
    if _FAILED_TEST_RE.search(observation) or (return_code is not None and return_code != 0):
        return "failed"
    if _PASSED_TEST_RE.search(observation) or return_code == 0:
        return "passed"
    return "unknown"


def _normalize_path(path: str) -> str:
    value = path.strip(" \t\r\n'\"`.,:;()[]{}<>").replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    if value.startswith(("a/", "b/")):
        value = value[2:]
    return value


def _bounded_unique(values: Any, limit: int, max_chars: int) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = _clip(_compact(str(raw)), max_chars)
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= limit:
            break
    return tuple(result)


def _merge(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*first, *second)))


def _compact(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."
