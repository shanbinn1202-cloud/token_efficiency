"""Pinned Qwen3.5 template trace and fail-closed final-token span verifier.

Input is the trusted *normalized* vLLM conversation and final renderer outputs,
not an HTTP body or client-provided span. This module installs no server hooks.
"""
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import secrets

from contracts import PolicyRuntimeAbort, canonical_prompt_hash
from dynamic_runtime import RenderedPrompt, source_digest, observation_geometry, require


PINNED_FILES = {
    "tokenizer-config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "model-config.json": "69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c",
}
TEMPLATE_SHA256 = "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
TOOL_BRANCH = '{%- elif message.role == "tool" %}'


@dataclass(frozen=True)
class TemplateTrace:
    character_start: int
    first_tool_message: int
    tool_message_count: int
    source_hash: str
    rendered_text_hash: str


@dataclass(frozen=True)
class SpanProof:
    rendered: RenderedPrompt
    recompute_start: int
    historical_pages: int
    trace: TemplateTrace
    prompt_hash: str
    tokenizer_sha256: str
    template_sha256: str
    schema: str = "qwen-terminal-tool-run-v1"
    page_tokens: int = field(kw_only=True)


def _load_pinned_artifacts(evidence_dir):
    """Keep verified immutable tokenizer bytes; never reopen them for loading."""
    blobs = {}
    for name, expected in PINNED_FILES.items():
        blob = (Path(evidence_dir) / name).read_bytes()
        require(sha256(blob).hexdigest() == expected, f"pinned file changed: {name}")
        blobs[name] = blob
    config = json.loads(blobs["tokenizer-config.json"])
    template = config["chat_template"]
    require(sha256(template.encode()).hexdigest() == TEMPLATE_SHA256,
            "pinned template changed")
    require(config["tokenizer_class"] == "Qwen2Tokenizer", "unexpected tokenizer class")
    return template, blobs["tokenizer.json"]


def load_pinned_template(evidence_dir):
    """Verify copied bytes, not the identity of a running server."""
    return _load_pinned_artifacts(evidence_dir)[0]


def trace_terminal_tools(*, template, conversation, render_template,
                         final_rendered_text, template_kwargs):
    """Instrument a specific template branch; never search for tool output text.

    render_template(template, conversation, kwargs) must use the trusted server's
    template environment. A local Jinja renderer is useful only for CPU tests.
    A terminal consecutive tool run is ONE observation. The span includes the
    enclosing user header and all trailing control/generation-prompt text.
    """
    require(type(template) is str and sha256(template.encode()).hexdigest() == TEMPLATE_SHA256,
            "only the pinned Qwen template is supported")
    require(type(conversation) is list and conversation,
            "normalized conversation required")
    require(type(template_kwargs) is dict and type(final_rendered_text) is str,
            "trusted rendered text and template kwargs required")
    require(set(template_kwargs) <= {"tools", "add_generation_prompt", "enable_thinking",
                                     "preserve_thinking"},
            "unsupported template options")
    require(template_kwargs.get("add_generation_prompt") is True,
            "generation prompt must be explicit and enabled")
    for message in conversation:
        require(type(message) is dict and message.get("role") in
                {"system", "user", "assistant", "tool"}, "unsupported normalized message")
        require(type(message.get("content")) is str,
                "normalized plain text required; raw arrays/multimodal are unsupported")
    require(conversation[-1]["role"] == "tool", "observation must end in real tool messages")
    first = len(conversation) - 1
    while first > 0 and conversation[first - 1]["role"] == "tool":
        first -= 1
    require(first > 0 and conversation[first - 1]["role"] == "assistant",
            "terminal tool run must follow an assistant message")
    # JSON roundtrip snapshots the source so a rendering callback cannot silently
    # mutate the source between original and traced execution.
    snapshot = json.loads(json.dumps(conversation, ensure_ascii=False, allow_nan=False))
    options = json.loads(json.dumps(template_kwargs, ensure_ascii=False, allow_nan=False))
    source_hash = source_digest(dict(conversation=snapshot, template_kwargs=options))
    original = render_template(template, snapshot, options)
    require(original == final_rendered_text, "final renderer text mismatch")
    require(source_digest(dict(conversation=snapshot, template_kwargs=options)) == source_hash,
            "template renderer mutated inputs")
    marker = "REMOTE_RA_TRACE_" + secrets.token_hex(32)
    require(marker not in original and marker not in json.dumps(snapshot), "trace marker collision")
    require(template.count(TOOL_BRANCH) == 1, "tool branch is not unique")
    insertion = TOOL_BRANCH + "{{- '" + marker + "' if loop.index0 == " + str(first) + " else '' }}"
    traced = render_template(template.replace(TOOL_BRANCH, insertion), snapshot, options)
    require(type(traced) is str and traced.count(marker) == 1, "missing/ambiguous branch trace")
    start = traced.index(marker)
    require(traced[:start] + traced[start + len(marker):] == original,
            "instrumentation changed final rendered text")
    require(source_digest(dict(conversation=snapshot, template_kwargs=options)) == source_hash,
            "template renderer mutated inputs")
    return TemplateTrace(start, first, len(snapshot) - first, source_hash,
                         sha256(original.encode()).hexdigest())


def prove_offset_boundary(*, text, character_start, token_ids, offsets, final_token_ids):
    """Check whole-prompt offset mapping supplied by a trusted tokenizer backend.

    This low-level function alone does NOT authenticate a tokenizer. Tests may
    supply synthetic maps, but the public producer never accepts an encoder from
    a request. Overlapping Unicode offsets are allowed; crossing S is not.
    """
    require(type(text) is str and type(character_start) is int
            and 0 <= character_start < len(text), "invalid semantic character boundary")
    require(isinstance(token_ids, (list, tuple)) and isinstance(final_token_ids, (list, tuple))
            and token_ids and all(type(t) is int and t >= 0 for t in token_ids)
            and all(type(t) is int and t >= 0 for t in final_token_ids), "invalid token array")
    require(tuple(token_ids) == tuple(final_token_ids),
            "final token mismatch: truncation, padding, special-token or tokenizer drift")
    require(isinstance(offsets, (list, tuple)) and len(offsets) == len(token_ids),
            "NO_VERIFIED_OFFSET_MAPPING: token/offset counts differ")
    previous_start, covered = -1, 0
    boundary = None
    for index, pair in enumerate(offsets):
        require(isinstance(pair, (list, tuple)) and len(pair) == 2
                and all(type(x) is int for x in pair), "invalid offset pair")
        left, right = pair
        require(0 <= left < right <= len(text) and previous_start <= left <= covered
                and right >= covered, "NO_VERIFIED_OFFSET_MAPPING: gap/reorder/empty offset")
        require(not left < character_start < right,
                "tokenizer token crosses semantic observation boundary")
        if left == character_start and boundary is None:
            boundary = index
        previous_start, covered = left, right
    require(covered == len(text) and boundary is not None,
            "NO_VERIFIED_OFFSET_MAPPING: incomplete coverage or no exact boundary")
    require(all(right <= character_start for _, right in offsets[:boundary])
            and all(left >= character_start for left, _ in offsets[boundary:]),
            "ambiguous token ownership at observation boundary")
    return boundary


def validate_tokenization_options(options):
    """Snapshot these fields from real TokenizeParams, not request JSON."""
    expected = dict(truncate_prompt_tokens=None, pad_prompt_tokens=None,
                    truncation_side=None, do_lower_case=False, add_special_tokens=False)
    require(type(options) is dict and set(options) == set(expected)
            and all(type(options[k]) is type(v) and options[k] == v for k, v in expected.items()),
            "tokenizer truncation/padding/case/special-token transformations are unsupported")


def validate_identity_normalization(*, text, normalizer):
    """Support only prompts unchanged by the pinned backend's normalizer.

    NFC can consume a source combining mark without giving it an offset. Such a
    map is not a proof of complete original-character coverage. Do not repair
    offsets, normalize the prompt, or infer a different semantic boundary.
    Use the backend's Unicode implementation, not Python's Unicode tables.
    """
    require(type(text) is str, "trusted rendered text required")
    normalize = getattr(normalizer, "normalize_str", None)
    require(callable(normalize),
            "NO_VERIFIED_OFFSET_MAPPING: pinned normalizer is unavailable")
    try:
        normalized = normalize(text)
    except Exception as exc:
        raise PolicyRuntimeAbort(
            "NO_VERIFIED_OFFSET_MAPPING: pinned normalization check failed") from exc
    require(type(normalized) is str,
            "NO_VERIFIED_OFFSET_MAPPING: invalid pinned normalization result")
    require(normalized == text,
            "NO_VERIFIED_OFFSET_MAPPING: NORMALIZATION_CHANGED_TEXT; "
            "only prompts unchanged by the pinned NFC normalizer are supported")


class QwenObservationSpanProducer:
    """Offline verifier ready for a trusted renderer seam, not installed plumbing.

    Exact final IDs provide per-prompt equality; the separate live-loaded source
    and model identity gate is still mandatory. No default synthetic fallback.
    """
    def __init__(self, evidence_dir, *, render_template):
        self.template, tokenizer_bytes = _load_pinned_artifacts(evidence_dir)
        require(callable(render_template), "trusted template renderer required")
        self._render_template = render_template
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise PolicyRuntimeAbort(
                "NO_VERIFIED_OFFSET_MAPPING: tokenizers backend is unavailable; "
                "no span was produced") from exc
        self._tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
        require(self._tokenizer.truncation is None and self._tokenizer.padding is None,
                "tokenizer truncation/padding is unsupported")

    def produce(self, *, conversation, template_kwargs, final_rendered_text,
                final_token_ids, page_tokens, tokenization_options, truncation_applied=False):
        validate_tokenization_options(tokenization_options)
        require(truncation_applied is False, "truncated prompts are unsupported")
        trace = trace_terminal_tools(template=self.template, conversation=conversation,
            render_template=self._render_template, final_rendered_text=final_rendered_text,
            template_kwargs=template_kwargs)
        validate_identity_normalization(text=final_rendered_text,
                                        normalizer=getattr(self._tokenizer, "normalizer", None))
        encoded = self._tokenizer.encode(final_rendered_text, add_special_tokens=False)
        require(not encoded.overflowing, "tokenizer overflow is unsupported")
        start = prove_offset_boundary(text=final_rendered_text,
            character_start=trace.character_start, token_ids=encoded.ids,
            offsets=encoded.offsets, final_token_ids=final_token_ids)
        tokens = tuple(final_token_ids)
        geometry = observation_geometry(start, len(tokens), len(tokens), page_tokens)
        return SpanProof(RenderedPrompt(tokens, start, len(tokens)), geometry.recompute_start,
            geometry.historical_pages, trace, canonical_prompt_hash(tokens),
            PINNED_FILES["tokenizer.json"], TEMPLATE_SHA256, page_tokens=page_tokens)
