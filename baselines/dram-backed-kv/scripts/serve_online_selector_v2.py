#!/usr/bin/env python3
"""Single-session OpenAI-compatible server for online Selector V2 trials.

This is deliberately an experiment runner, not a vLLM patch.  It owns one
Qwen session, publishes vectors from a fixed decoder layer, enforces the
selected physical Pages in the real Transformers cache, and returns freely
generated text to Mini-SWE-Agent.  Mini-SWE-Agent remains responsible for
parsing and executing shell actions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
import traceback
from threading import RLock
from time import perf_counter, time
from typing import Any, Mapping, Sequence
import uuid

from dram_kv.backends.dram import DramRuntimeBackend
from dram_kv.goals import MaintainedGoalConstructor, PersistentCommandGoalRewriter
from dram_kv.goals.features import DeterministicFeatureExtractor
from dram_kv.integrations.qwen35 import install_qwen35_logical_rope
from dram_kv.page_vectors import (
    CompactVector,
    PageVectorIndex,
    PageVectorProvenance,
    PageVectorPublisher,
)
from dram_kv.runtime import DramKVRuntime
from dram_kv.runtime.qwen import (
    QwenKVPage,
    QwenKVPageAdapter,
    extract_suffix_page,
    full_attention_layers,
)
from dram_kv.runtime.qwen_forward import QwenForwardBridge
from dram_kv.schema import (
    GoalRecord,
    PageCatalog,
    PageRecord,
    SelectionPlan,
    SelectionReason,
    TurnContext,
)
from dram_kv.selectors import PrefillPageEmbeddingSelectorV2


PAGE_SIZE = 16
_HANDLE_RE = re.compile(r"[A-Za-z0-9_./:$-]{2,}")
_RETURN_CODE_RE = re.compile(r"<returncode>(-?\d+)</returncode>")
_TASK_RE = re.compile(r"<pr_description>\s*(.*?)\s*</pr_description>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="selector-v2-qwen")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--active-ratio", type=float, default=0.40)
    parser.add_argument("--switch-threshold", type=int, default=8192)
    parser.add_argument("--recent-turns", type=int, default=2)
    parser.add_argument("--recent-quota", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--hidden-layer", type=int, default=-1)
    parser.add_argument("--projection-dimension", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=20260904)
    parser.add_argument(
        "--selection-mode",
        choices=("selector", "full_context"),
        default="selector",
        help="Run Page selection or keep the complete logical history active",
    )
    parser.add_argument(
        "--query-source", choices=("goal", "observation"), default="goal"
    )
    parser.add_argument("--goal-rewriter-model")
    parser.add_argument("--goal-rewriter-max-new-tokens", type=int, default=384)
    parser.add_argument("--goal-rewriter-timeout", type=float, default=120.0)
    parser.add_argument("--metrics", type=Path, required=True)
    return parser.parse_args()


def _create_goal_rewriter(args: argparse.Namespace):
    if args.selection_mode == "full_context" or not args.goal_rewriter_model:
        return None
    rewriter_script = Path(__file__).with_name("qwen_goal_rewriter.py")
    return PersistentCommandGoalRewriter(
        (
            sys.executable,
            "-u",
            str(rewriter_script),
            "--model",
            args.goal_rewriter_model,
            "--device",
            "cuda",
            "--dtype",
            "bfloat16",
            "--max-new-tokens",
            str(args.goal_rewriter_max_new_tokens),
            "--local-files-only",
            "--serve",
        ),
        timeout_seconds=args.goal_rewriter_timeout,
        startup_timeout_seconds=300.0,
    )


def _extract_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def _sha_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()


def _projection(torch: Any, hidden_size: int, dimension: int, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.randint(
        0, 2, (dimension, hidden_size), generator=generator, dtype=torch.int8
    ).to(torch.float32)
    values.mul_(2.0).sub_(1.0).mul_(1.0 / math.sqrt(dimension))
    digest = sha256(values.contiguous().numpy().tobytes()).hexdigest()
    return values, digest


def _model_layers(model: Any) -> Sequence[Any]:
    base = model.model
    text = getattr(base, "language_model", base)
    layers = getattr(text, "layers", None)
    if layers is None:
        raise TypeError("main model does not expose decoder layers")
    return layers


def _root_objective(messages: Sequence[Mapping[str, Any]]) -> str:
    if len(messages) < 2:
        raise ValueError("a system message and task message are required")
    content = str(messages[1].get("content", "")).strip()
    match = _TASK_RE.search(content)
    return (match.group(1) if match else content).strip()


def _goal_query_text(goal: GoalRecord) -> str:
    """Render only the maintained selector-visible Goal, never diagnostics."""

    return "<retrieval_query>\n" + json.dumps(
        {"retrieval_goal": asdict(goal.retrieval_goal)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n</retrieval_query>"


def _disabled_query_metrics() -> dict[str, Any]:
    return {
        "source": None,
        "required": False,
        "published": False,
        "skip_reason": "full_context_mode",
        "cache_hit": False,
        "cache_entries": 0,
        "goal_hash": None,
        "goal_chars": 0,
        "goal_payload_hash": None,
        "token_count": None,
        "tokenize_ms": 0.0,
        "forward_ms": 0.0,
        "projection_ms": 0.0,
        "publish_ms": 0.0,
    }


def _goal_metrics(goal: GoalRecord | None) -> dict[str, Any]:
    if goal is None:
        return {
            "enabled": False,
            "gate_decision": "DISABLED",
            "applied_decision": "DISABLED",
            "gate_trigger": "full_context_mode",
            "reused_previous": False,
            "changed_fields": [],
            "retrieval_goal_hash": None,
            "retrieval_goal_chars": 0,
            "slm_attempted": False,
            "slm_succeeded": False,
            "slm_fallback_reason": None,
            "slm_input_tokens": 0,
            "slm_output_tokens": 0,
            "constructor_total_latency_ms": 0.0,
            "rule_latency_ms": 0.0,
            "slm_total_latency_ms": 0.0,
            "slm_startup_latency_ms": 0.0,
            "slm_request_latency_ms": 0.0,
            "slm_generation_latency_ms": 0.0,
        }
    return {
        "enabled": True,
        "gate_decision": goal.gate_decision,
        "applied_decision": goal.applied_decision,
        "gate_trigger": goal.triggers[0] if goal.triggers else "initialization",
        "reused_previous": goal.reused_previous,
        "changed_fields": list(goal.changed_fields),
        "retrieval_goal_hash": goal.retrieval_goal_hash,
        "retrieval_goal_chars": len(json.dumps(
            asdict(goal.retrieval_goal),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )),
        "slm_attempted": goal.slm_attempted,
        "slm_succeeded": goal.slm_succeeded,
        "slm_fallback_reason": goal.fallback_reason,
        "slm_input_tokens": goal.input_tokens,
        "slm_output_tokens": goal.output_tokens,
        "constructor_total_latency_ms": goal.latency_ms,
        "rule_latency_ms": goal.diagnostics.get("rule_latency_ms", 0.0),
        "slm_total_latency_ms": goal.diagnostics.get("slm_latency_ms", 0.0),
        "slm_startup_latency_ms": goal.diagnostics.get(
            "slm_startup_latency_ms", 0.0
        ),
        "slm_request_latency_ms": goal.diagnostics.get(
            "slm_request_latency_ms", 0.0
        ),
        "slm_generation_latency_ms": goal.diagnostics.get(
            "slm_generation_latency_ms", 0.0
        ),
    }


class OnlineSelectorSession:
    """One prefix-stable Mini-SWE conversation backed by real selected KV."""

    def __init__(
        self,
        args: argparse.Namespace,
        model: Any,
        tokenizer: Any,
        torch: Any,
        *,
        goal_rewriter: Any | None = None,
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.full_context = args.selection_mode == "full_context"
        self.session_id = f"online-{uuid.uuid4().hex}"
        self.instance_id = "online-swe"
        self.turn_id = 0
        self.logical_tokens: list[int] = []
        self.pages: list[PageRecord] = []
        self.page_turns: dict[int, int] = {}
        self.mandatory_end = 0
        self.cataloged_end = 0
        self.sealed_end = 0
        self.live_tail: QwenKVPage | None = None
        self.skeleton: Any = None
        self.previous_active: tuple[str, ...] = ()
        self.ever_selected: set[str] = set()
        self.previous_action: str | None = None
        self.assistant_serializations: list[tuple[str, tuple[int, ...]]] = []
        self.root_goal: str | None = None
        self.resume_reason = "fresh_instance"
        self.previous_session_id: str | None = None
        self.last_acceptance_diagnostic: dict[str, Any] | None = None
        self.index = PageVectorIndex()
        self._owns_goal_rewriter = goal_rewriter is None
        self.goal_rewriter = (
            None if self.full_context else goal_rewriter or _create_goal_rewriter(args)
        )
        self.goal_constructor = None if self.full_context else MaintainedGoalConstructor(
            self.goal_rewriter,
            entity_drift_steps=3,
            initialize_with_rewriter=self.goal_rewriter is not None,
        )
        text_config = getattr(model.config, "text_config", model.config)
        self.bytes_per_token = (
            sum(kind == "full_attention" for kind in text_config.layer_types)
            * 2
            * int(text_config.num_key_value_heads)
            * int(text_config.head_dim)
            * torch.tensor([], dtype=torch.bfloat16).element_size()
        )
        self.projection = None
        self.provenance = None
        self.publisher = None
        self.goal_control_hash = None
        if not self.full_context:
            projection_cpu, projection_hash = _projection(
                torch,
                int(text_config.hidden_size),
                args.projection_dimension,
                args.projection_seed,
            )
            self.projection = projection_cpu.to(model.device)
            self.provenance = PageVectorProvenance(
                model=str(args.model),
                revision="local",
                hidden_layer=self._resolve_hidden_layer(args.hidden_layer),
                pooling="last_token",
                projection_hash=projection_hash,
                dimension=args.projection_dimension,
                dtype="int8",
                vector_source="online_main_model_hidden_states",
            )
            self.publisher = PageVectorPublisher(self.index, self.provenance)
            assert self.goal_constructor is not None
            self.goal_control_hash = _sha_json({
                "constructor": self.goal_constructor.name,
                "config": self.goal_constructor._config(),
            })
        self.runtime = DramKVRuntime(QwenKVPageAdapter(model.device), pin_cpu=True)
        self.backend = DramRuntimeBackend(self.runtime)
        self.bridge = QwenForwardBridge(self.backend)
        self._captured: list[Any] = []
        self._hook = None
        if self.provenance is not None:
            layer = _model_layers(model)[self.provenance.hidden_layer]
            self._hook = layer.register_forward_hook(self._capture_layer)
        self._page_vectors: dict[int, CompactVector] = {}
        self._goal_query_cache: dict[str, tuple[CompactVector, int, str]] = {}
        self._feature_extractor = DeterministicFeatureExtractor()

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
        self.backend.release_session(self.session_id)
        self.index.release_session(self.session_id)
        if self.goal_constructor is not None:
            self.goal_constructor.release_session(self.session_id)
        if self._owns_goal_rewriter and self.goal_rewriter is not None:
            self.goal_rewriter.close()

    def _resolve_hidden_layer(self, requested: int) -> int:
        count = len(_model_layers(self.model))
        resolved = requested if requested >= 0 else count + requested
        if resolved < 0 or resolved >= count:
            raise ValueError("hidden layer is outside the decoder stack")
        return resolved

    def _capture_layer(self, _module: Any, _inputs: Any, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        self._captured.append(tensor.detach())

    def _encode(self, text: str) -> list[int]:
        return _extract_ids(self.tokenizer(text, add_special_tokens=False))

    def _render(self, messages: Sequence[Mapping[str, Any]], add_prompt: bool) -> list[int]:
        if len(messages) < 2:
            raise ValueError("conversation needs system and user messages")
        kwargs = {"tokenize": True, "add_generation_prompt": False}
        try:
            ids = _extract_ids(self.tokenizer.apply_chat_template(
                messages[:2], **kwargs, enable_thinking=False
            ))
        except TypeError:
            ids = _extract_ids(self.tokenizer.apply_chat_template(messages[:2], **kwargs))
        # A rebuilt session may begin with assistant messages that were produced
        # by an older session.  Only substitute exact generated token IDs when
        # the response text itself matches; positional indexing would attach the
        # first new serialization to the first old assistant message.
        exact_assistants = list(self.assistant_serializations)
        for message in messages[2:]:
            role = str(message["role"])
            content = str(message.get("content", ""))
            if role not in {"assistant", "user"}:
                raise ValueError(f"text action runner emitted unsupported role {role!r}")
            exact_index = next(
                (
                    index
                    for index, (returned_content, _exact_ids) in enumerate(exact_assistants)
                    # LiteLLM/Mini-SWE may trim transport-only leading or
                    # trailing whitespace before storing an assistant message.
                    # The generated token IDs remain the authoritative cache
                    # serialization when the trimmed payload is unchanged.
                    if returned_content.strip() == content.strip()
                ),
                None,
            ) if role == "assistant" else None
            if exact_index is None:
                ids.extend(self._encode(f"<|im_start|>{role}\n{content}<|im_end|>\n"))
                continue
            _returned_content, exact_ids = exact_assistants.pop(exact_index)
            ids.extend(exact_ids)
        if add_prompt:
            if messages[-1]["role"] != "user":
                raise ValueError("generation prompt must follow a user observation")
            # Qwen3.6's chat template emits an explicit empty thinking block
            # when thinking is disabled.  It must be part of the prefetched
            # prompt, not generated response text: LiteLLM removes that block
            # from assistant content before sending the next request, which
            # otherwise makes a valid transcript appear prefix-unstable.
            ids.extend(self._encode(
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ))
        return ids

    def accepts(self, messages: Sequence[Mapping[str, Any]]) -> bool:
        """Return whether ``messages`` can extend the currently cached transcript.

        Mini-SWE-Agent may compact an old transcript once it reaches its context
        budget.  That is a valid request, but its token stream can no longer use
        this session's logical positions or KV.  The server handles it by opening
        a fresh cold-prefill session for the compacted transcript.
        """
        if not self.logical_tokens:
            self.last_acceptance_diagnostic = {"accepted": True, "reason": "empty_session"}
            return True
        complete = self._render(messages, False)
        # A retry can resend an already-consumed user observation without the
        # assistant response that this session generated for it.  There is no
        # new suffix to prefill in that case, so rebuild from the request just
        # like a compacted/rewritten transcript instead of entering query() with
        # an empty observation.
        accepted = (
            len(complete) > len(self.logical_tokens)
            and complete[: len(self.logical_tokens)] == self.logical_tokens
        )
        mismatch = next(
            (
                index
                for index, (incoming, cached) in enumerate(
                    zip(complete, self.logical_tokens)
                )
                if incoming != cached
            ),
            min(len(complete), len(self.logical_tokens)),
        )
        incoming_assistants = [
            str(message.get("content", ""))
            for message in messages
            if str(message.get("role", "")) == "assistant"
        ]
        cached_assistants = [text for text, _ids in self.assistant_serializations]
        self.last_acceptance_diagnostic = {
            "accepted": accepted,
            "reason": (
                "strict_extension"
                if accepted
                else "not_longer"
                if len(complete) <= len(self.logical_tokens)
                else "prefix_mismatch"
            ),
            "cached_token_count": len(self.logical_tokens),
            "incoming_token_count": len(complete),
            "first_token_mismatch": mismatch,
            "message_roles": [str(message.get("role", "")) for message in messages],
            "latest_content_prefix": str(messages[-1].get("content", ""))[:240],
            "incoming_assistants": [
                {
                    "chars": len(text),
                    "sha256": sha256(text.strip().encode("utf-8")).hexdigest(),
                    "prefix": text[:160],
                    "suffix": text[-160:],
                    "open_tags": text.count("<mswea_bash_command>"),
                    "close_tags": text.count("</mswea_bash_command>"),
                }
                for text in incoming_assistants
            ],
            "cached_assistants": [
                {
                    "chars": len(text),
                    "sha256": sha256(text.strip().encode("utf-8")).hexdigest(),
                    "prefix": text[:160],
                    "suffix": text[-160:],
                    "open_tags": text.count("<mswea_bash_command>"),
                    "close_tags": text.count("</mswea_bash_command>"),
                }
                for text in cached_assistants
            ],
        }
        return accepted

    def _compact_last(self, hidden: Any) -> CompactVector:
        if self.projection is None:
            raise RuntimeError("full-context mode does not create vectors")
        vector = self.torch.nn.functional.linear(hidden.float(), self.projection)
        vector = self.torch.nn.functional.normalize(vector, p=2, dim=-1)
        peak = vector.abs().max()
        if not bool(self.torch.isfinite(peak)) or float(peak) == 0.0:
            raise ValueError("projection produced an invalid zero vector")
        scale = peak / 127.0
        values = self.torch.clamp(
            self.torch.round(vector / scale), -127, 127
        ).to(self.torch.int8)
        return CompactVector(
            tuple(int(value) for value in values.cpu().tolist()),
            "int8",
            float(scale.cpu()),
        )

    def _record_hidden(self, logical_start: int, hidden: Any) -> None:
        if self.full_context:
            return
        if hidden.ndim != 3 or int(hidden.shape[0]) != 1 or int(hidden.shape[1]) <= 0:
            raise RuntimeError("unexpected fixed-layer hidden-state shape")
        count = int(hidden.shape[1])
        boundary_offsets = [
            offset for offset in range(count)
            if (logical_start + offset + 1) % PAGE_SIZE == 0
        ]
        if not boundary_offsets:
            return
        selected = hidden[0, boundary_offsets, :]
        for offset, row in zip(boundary_offsets, selected):
            page_index = (logical_start + offset) // PAGE_SIZE
            self._page_vectors[page_index] = self._compact_last(row)

    def _forward(
        self,
        input_ids: Sequence[int],
        kwargs: Mapping[str, Any],
        *,
        use_cache: bool = True,
    ):
        if not input_ids:
            raise ValueError("model suffix cannot be empty")
        self._captured.clear()
        tensor = self.torch.tensor([input_ids], dtype=self.torch.long, device=self.model.device)
        with self.torch.inference_mode():
            output = self.model(
                input_ids=tensor,
                use_cache=use_cache,
                return_dict=True,
                logits_to_keep=1,
                **kwargs,
            )
        if self.full_context:
            return output, None
        if len(self._captured) != 1:
            raise RuntimeError(f"fixed-layer hook fired {len(self._captured)} times")
        return output, self._captured.pop()

    def _publish_selection_query(
        self,
        goal: GoalRecord,
        observation_hidden: Any,
        *,
        required: bool,
    ) -> dict[str, Any]:
        """Publish a same-space query and return isolated cost telemetry."""

        if self.full_context or self.publisher is None:
            raise RuntimeError("full-context mode cannot publish a selection query")

        rendered = _goal_query_text(goal)
        rendered_hash = sha256(rendered.encode("utf-8")).hexdigest()
        metrics: dict[str, Any] = {
            "source": self.args.query_source,
            "required": required,
            "published": False,
            "skip_reason": None,
            "cache_hit": False,
            "cache_entries": len(self._goal_query_cache),
            "goal_hash": goal.retrieval_goal_hash,
            "goal_chars": len(rendered),
            "goal_payload_hash": rendered_hash,
            "token_count": None,
            "tokenize_ms": 0.0,
            "forward_ms": 0.0,
            "projection_ms": 0.0,
            "publish_ms": 0.0,
        }
        if not required:
            metrics["skip_reason"] = "selector_inactive_before_threshold"
            return metrics

        if self.args.query_source == "observation":
            self.torch.cuda.synchronize()
            projection_started = perf_counter()
            vector = self._compact_last(observation_hidden)
            self.torch.cuda.synchronize()
            metrics["projection_ms"] = (perf_counter() - projection_started) * 1_000
        else:
            cached = self._goal_query_cache.get(goal.retrieval_goal_hash)
            if cached is not None:
                vector, token_count, cached_payload_hash = cached
                if cached_payload_hash != rendered_hash:
                    raise RuntimeError("Goal hash collision changed the query payload")
                metrics["cache_hit"] = True
                metrics["token_count"] = token_count
            else:
                tokenize_started = perf_counter()
                goal_ids = self._encode(rendered)
                metrics["tokenize_ms"] = (perf_counter() - tokenize_started) * 1_000
                if not goal_ids:
                    raise ValueError("rendered Goal query cannot be empty")
                metrics["token_count"] = len(goal_ids)

                self.torch.cuda.synchronize()
                forward_started = perf_counter()
                _goal_output, goal_hidden = self._forward(
                    goal_ids,
                    self._cold_kwargs(len(goal_ids)),
                    use_cache=False,
                )
                self.torch.cuda.synchronize()
                metrics["forward_ms"] = (perf_counter() - forward_started) * 1_000

                projection_started = perf_counter()
                vector = self._compact_last(goal_hidden[0, -1, :])
                self.torch.cuda.synchronize()
                metrics["projection_ms"] = (
                    perf_counter() - projection_started
                ) * 1_000
                self._goal_query_cache[goal.retrieval_goal_hash] = (
                    vector, len(goal_ids), rendered_hash
                )
                del _goal_output, goal_hidden

        publish_started = perf_counter()
        self.publisher.publish_query(self.session_id, self.turn_id, vector)
        metrics["publish_ms"] = (perf_counter() - publish_started) * 1_000
        metrics["published"] = True
        metrics["cache_entries"] = len(self._goal_query_cache)
        return metrics

    def _cold_kwargs(self, token_count: int, logical_start: int = 0) -> dict[str, Any]:
        positions = self.torch.arange(
            logical_start, logical_start + token_count,
            dtype=self.torch.long, device=self.model.device,
        )
        return {
            "past_key_values": None,
            "attention_mask": self.torch.ones(
                (1, token_count), dtype=self.torch.long, device=self.model.device
            ),
            "cache_position": self.torch.arange(
                token_count, dtype=self.torch.long, device=self.model.device
            ),
            "position_ids": self.torch.arange(
                token_count, dtype=self.torch.long, device=self.model.device
            ).unsqueeze(0),
            "dram_kv_logical_position_ids": positions.unsqueeze(0),
        }

    def _catalog(self) -> PageCatalog:
        return PageCatalog(self.session_id, tuple(self.pages))

    def _context(self, observation: str) -> TurnContext:
        catalog = self._catalog()
        history = catalog.historical_for_turn(self.turn_id)
        match = _RETURN_CODE_RE.search(observation)
        return TurnContext(
            session_id=self.session_id,
            instance_id=self.instance_id,
            turn_id=self.turn_id,
            task_objective=self.root_goal or "",
            previous_goal_text=None,
            previous_action=self.previous_action,
            current_observation=observation,
            structured_state={
                "tool_name": "bash",
                "return_code": int(match.group(1)) if match else None,
            },
            historical_turn_ids=tuple(range(1, self.turn_id)),
            historical_page_ids=tuple(page.page_id for page in history),
            hbm_resident_page_ids=tuple(
                page_id for page_id in self.runtime.resident_page_ids(self.session_id)
                if page_id in {page.page_id for page in history}
            ),
            message_source="user" if self.turn_id == 1 else "tool",
        )

    def _page_plan(
        self,
        context: TurnContext,
        label: str,
        page_ids: Sequence[str] | None = None,
    ) -> SelectionPlan:
        wanted = set(context.historical_page_ids if page_ids is None else page_ids)
        ordered = tuple(
            page.page_id
            for page in self.pages
            if page.page_id in wanted and page.turn_id < self.turn_id
        )
        return SelectionPlan(
            session_id=self.session_id,
            turn_id=self.turn_id,
            goal_hash=label,
            selector=label,
            selector_config={
                "mode": label,
                "selection_mode": self.args.selection_mode,
            },
            ordered_page_ids=ordered,
            reasons={page_id: SelectionReason(label, 1.0) for page_id in ordered},
            mandatory_page_ids=(),
            recent_page_ids=(),
            retrieved_page_ids=ordered,
        )

    def _plan(self, context: TurnContext, goal: Any) -> SelectionPlan:
        if self.full_context:
            return self._page_plan(context, "full_context")
        history_tokens = len(context.historical_page_ids) * PAGE_SIZE
        if history_tokens <= self.args.switch_threshold:
            return self._page_plan(context, "full_active_before_threshold")
        history_ids = set(context.historical_page_ids)
        mandatory_tokens = sum(
            page.token_count
            for page in self.pages
            if page.page_id in history_ids and page.mandatory
        )
        # Safety pages are constraints, not retrieval candidates.  A ratio can
        # shrink the optional working set, but must never make the plan itself
        # infeasible by budgeting fewer tokens than the mandatory prefix.
        capacity = max(
            PAGE_SIZE,
            mandatory_tokens,
            math.floor(history_tokens * self.args.active_ratio),
        )
        assert self.provenance is not None and self.goal_control_hash is not None
        selector = PrefillPageEmbeddingSelectorV2(
            self.index,
            self.provenance,
            frozen_goal_trace_hash=self.goal_control_hash,
            active_token_budget=capacity,
            recent_turns=self.args.recent_turns,
            recent_quota=self.args.recent_quota,
            query_source=f"{self.args.query_source}_hidden",
        )
        return selector.select(
            context,
            goal,
            self._catalog(),
            previous_active=self.previous_active,
            ever_selected=tuple(self.ever_selected),
        )

    def _active_before_observation(self, context: TurnContext) -> SelectionPlan:
        if self.full_context:
            return self._page_plan(context, "full_context_before_observation")
        # Pages produced in the preceding action stay live until this turn's
        # query exists and the new selection can replace them atomically.
        return self._page_plan(
            context,
            "previous_working_set_plus_last_turn",
            self.runtime.resident_page_ids(self.session_id),
        )

    def _prepared_kwargs(
        self,
        context: TurnContext,
        plan: SelectionPlan,
        suffix_start: int,
        suffix_count: int,
        live_tail: QwenKVPage | None,
    ) -> tuple[dict[str, Any], int]:
        prepared = self.bridge.prepare_before_prefill(
            context,
            self._catalog(),
            plan,
            self.skeleton,
            suffix_logical_start=suffix_start,
            suffix_token_count=suffix_count,
            live_tail=live_tail,
        )
        return prepared.model_kwargs(
            self.torch, device=self.model.device
        ), prepared.physical_prefix_length

    def _add_page_records(self, through: int) -> None:
        while self.cataloged_end + PAGE_SIZE <= through:
            start = self.cataloged_end
            end = start + PAGE_SIZE
            page_index = start // PAGE_SIZE
            page_id = f"p{page_index:08d}:{start}-{end}"
            vector_ref = None
            files: tuple[str, ...] = ()
            symbols: tuple[str, ...] = ()
            tests: tuple[str, ...] = ()
            errors: tuple[str, ...] = ()
            handles: tuple[str, ...] = ()
            if not self.full_context:
                vector = self._page_vectors.pop(page_index)
                assert self.publisher is not None
                vector_ref = self.publisher.publish_page(
                    self.session_id, page_id, vector
                )
                text = self.tokenizer.decode(self.logical_tokens[start:end])
                features = self._feature_extractor.extract(None, text)
                files = features.files
                symbols = features.symbols
                tests = features.tests
                errors = features.errors
                handles = tuple(dict.fromkeys(_HANDLE_RE.findall(text)))[:64]
            self.pages.append(PageRecord(
                page_id=page_id,
                turn_id=self.page_turns.get(page_index, self.turn_id),
                role="mixed",
                token_start=start,
                token_end=end,
                logical_position_start=start,
                logical_position_end=end,
                byte_size=PAGE_SIZE * self.bytes_per_token,
                mandatory=end <= self.mandatory_end,
                files=files,
                symbols=symbols,
                tests=tests,
                errors=errors,
                lexical_handles=handles,
                page_vector_ref=vector_ref,
                kv_provenance=(
                    "exact" if self.full_context or start < self.args.switch_threshold
                    else "selection_conditioned"
                ),
            ))
            self.cataloged_end = end

    def _seal_kv_pages(
        self, cache: Any, through: int, *, suffix_start: int, physical_prefix: int
    ) -> None:
        target = through // PAGE_SIZE * PAGE_SIZE
        for start in range(self.sealed_end, target, PAGE_SIZE):
            page_index = start // PAGE_SIZE
            page_id = f"p{page_index:08d}:{start}-{start + PAGE_SIZE}"
            payload = extract_suffix_page(
                cache,
                start,
                start + PAGE_SIZE,
                suffix_logical_start=suffix_start,
                physical_prefix_length=physical_prefix,
            )
            self.backend.seal_after_generation(
                self.session_id, page_id, payload, keep_resident=True
            )
        self.sealed_end = target
        self._add_page_records(through)

    def _clear_attention(self, cache: Any) -> None:
        for layer in full_attention_layers(cache).values():
            layer.keys = None
            layer.values = None
            if hasattr(layer, "is_initialized"):
                layer.is_initialized = False

    def _decode_one(self, cache: Any, logical_position: int):
        physical = next(iter(full_attention_layers(cache).values())).keys.shape[-2]
        kwargs = {
            "past_key_values": cache,
            "attention_mask": self.torch.ones(
                (1, physical + 1), dtype=self.torch.long, device=self.model.device
            ),
            "cache_position": self.torch.tensor(
                [physical], dtype=self.torch.long, device=self.model.device
            ),
            "position_ids": self.torch.tensor(
                [[physical]], dtype=self.torch.long, device=self.model.device
            ),
            "dram_kv_logical_position_ids": self.torch.tensor(
                [[logical_position]], dtype=self.torch.long, device=self.model.device
            ),
        }
        return kwargs

    def _generate(
        self, output: Any, cache: Any, logical_start: int
    ) -> tuple[str, Any, int, bool]:
        generated: list[int] = []
        next_token = int(output.logits[0, -1].argmax().item())
        close_marker = "</mswea_bash_command>"
        stop_ids = {
            int(token_id)
            for token_id in (
                getattr(self.tokenizer, "eos_token_id", None),
                getattr(self.tokenizer, "pad_token_id", None),
            )
            if token_id is not None
        }
        for _ in range(self.args.max_new_tokens):
            generated.append(next_token)
            logical_position = logical_start + len(generated) - 1
            self.logical_tokens.append(next_token)
            self.page_turns[logical_position // PAGE_SIZE] = self.turn_id
            output, hidden = self._forward(
                [next_token], self._decode_one(cache, logical_position)
            )
            cache = output.past_key_values
            self._record_hidden(logical_position, hidden)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if close_marker in text:
                return text, cache, len(generated), False
            if next_token in stop_ids:
                return text, cache, len(generated), True
            next_token = int(output.logits[0, -1].argmax().item())
        return (
            self.tokenizer.decode(generated, skip_special_tokens=True),
            cache,
            len(generated),
            False,
        )

    def _append_boundary(self, cache: Any, *, eos_already_cached: bool) -> tuple[Any, int]:
        # The response omits chat boundary tokens. Prefill them now so the next
        # request's rendered history is byte-for-byte prefix stable.
        boundary = self._encode("\n" if eos_already_cached else "<|im_end|>\n")
        logical_start = len(self.logical_tokens)
        output = None
        for offset, token in enumerate(boundary):
            position = logical_start + offset
            self.logical_tokens.append(token)
            self.page_turns[position // PAGE_SIZE] = self.turn_id
            output, hidden = self._forward([token], self._decode_one(cache, position))
            cache = output.past_key_values
            self._record_hidden(position, hidden)
        return cache, len(boundary)

    def query(self, messages: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
        started = perf_counter()
        self.turn_id += 1
        if self.root_goal is None:
            self.root_goal = _root_objective(messages)
            self.instance_id = sha256(self.root_goal.encode("utf-8")).hexdigest()[:16]
        complete = self._render(messages, False)
        with_prompt = self._render(messages, True)
        if complete[: len(self.logical_tokens)] != self.logical_tokens:
            raise ValueError("Mini-SWE transcript is not prefix-stable with online KV")
        observation_ids = complete[len(self.logical_tokens) :]
        prompt_ids = with_prompt[len(complete) :]
        if not observation_ids or not prompt_ids:
            raise ValueError("current observation or assistant prompt is empty")
        observation = str(messages[-1].get("content", ""))
        turn_suffix_start = self.sealed_end
        old_logical_end = len(self.logical_tokens)
        query_metrics: dict[str, Any]
        goal: GoalRecord | None

        if self.turn_id == 1:
            # A context-rebuild cold-prefills the whole request, but only the
            # stable system + root-task prefix is mandatory.  Marking the
            # reconstructed history mandatory would silently defeat the active
            # ratio on every later V2 plan.
            self.mandatory_end = len(self._render(messages[:2], False))
            feed = observation_ids + prompt_ids
            output, hidden = self._forward(feed, self._cold_kwargs(len(feed)))
            self.logical_tokens.extend(feed)
            for position in range(len(feed)):
                self.page_turns[position // PAGE_SIZE] = self.turn_id
            self._record_hidden(0, hidden)
            context = self._context(observation)
            if self.full_context:
                goal = None
                query_metrics = _disabled_query_metrics()
                plan = self._plan(context, goal)
            else:
                assert hidden is not None and self.goal_constructor is not None
                query_hidden = hidden[0, len(observation_ids) - 1, :]
                goal = self.goal_constructor.construct(context)
                query_metrics = self._publish_selection_query(
                    goal, query_hidden, required=False
                )
                plan = self._page_plan(context, "cold_start")
            physical_prefix = 0
            generation_start = len(self.logical_tokens)
            assistant_start = len(observation_ids)
        else:
            context = self._context(observation)
            pre_plan = self._active_before_observation(context)
            kwargs, _pre_active_prefix = self._prepared_kwargs(
                context,
                pre_plan,
                old_logical_end,
                len(observation_ids),
                self.live_tail,
            )
            output, hidden = self._forward(observation_ids, kwargs)
            observation_cache = output.past_key_values
            self.logical_tokens.extend(observation_ids)
            for position in range(old_logical_end, len(self.logical_tokens)):
                self.page_turns[position // PAGE_SIZE] = self.turn_id
            self._record_hidden(old_logical_end, hidden)
            # Publish vectors/catalog entries now so the selector sees the
            # completed physical Pages on the next turn. Keep their KV in the
            # current live suffix until generation ends; sealing them into the
            # runtime early would make the current selection evict them and
            # force an avoidable H2D restore at the next observation.
            self._add_page_records(len(self.logical_tokens))
            context = self._context(observation)
            if self.full_context:
                goal = None
                query_metrics = _disabled_query_metrics()
            else:
                assert hidden is not None and self.goal_constructor is not None
                query_hidden = hidden[0, -1, :]
                goal = self.goal_constructor.construct(context)
                query_metrics = self._publish_selection_query(
                    goal,
                    query_hidden,
                    required=(
                        len(context.historical_page_ids) * PAGE_SIZE
                        > self.args.switch_threshold
                    ),
                )
            plan = self._plan(context, goal)
            current_live = extract_suffix_page(
                observation_cache,
                turn_suffix_start,
                len(self.logical_tokens),
                suffix_logical_start=turn_suffix_start,
                physical_prefix_length=len(pre_plan.ordered_page_ids) * PAGE_SIZE,
            )
            kwargs, _formal_active_prefix = self._prepared_kwargs(
                context,
                plan,
                len(self.logical_tokens),
                len(prompt_ids),
                current_live,
            )
            prompt_start = len(self.logical_tokens)
            output, hidden = self._forward(prompt_ids, kwargs)
            self.logical_tokens.extend(prompt_ids)
            for position in range(prompt_start, len(self.logical_tokens)):
                self.page_turns[position // PAGE_SIZE] = self.turn_id
            self._record_hidden(prompt_start, hidden)
            generation_start = len(self.logical_tokens)
            assistant_start = prompt_start
            physical_prefix = len(plan.ordered_page_ids) * PAGE_SIZE

        text, cache, generated_count, eos_cached = self._generate(
            output, output.past_key_values, generation_start
        )
        cache, boundary_count = self._append_boundary(
            cache, eos_already_cached=eos_cached
        )
        self.assistant_serializations.append(
            (text, tuple(self.logical_tokens[assistant_start:]))
        )
        self._seal_kv_pages(
            cache,
            len(self.logical_tokens),
            suffix_start=turn_suffix_start,
            physical_prefix=physical_prefix,
        )
        self.live_tail = (
            extract_suffix_page(
                cache,
                self.sealed_end,
                len(self.logical_tokens),
                suffix_logical_start=turn_suffix_start,
                physical_prefix_length=physical_prefix,
            )
            if self.sealed_end < len(self.logical_tokens)
            else None
        )
        self.skeleton = cache
        self._clear_attention(self.skeleton)
        resident = set(self.runtime.resident_page_ids(self.session_id))
        self.previous_active = tuple(
            page.page_id for page in self.pages if page.page_id in resident
        )
        self.ever_selected.update(plan.ordered_page_ids)
        self.previous_action = text
        stats = asdict(self.runtime.session_metrics(self.session_id))
        record = {
            "session_id": self.session_id,
            "resume_reason": self.resume_reason,
            "previous_session_id": self.previous_session_id,
            "instance_key": self.instance_id,
            "root_goal_hash": sha256((self.root_goal or "").encode("utf-8")).hexdigest(),
            "turn_id": self.turn_id,
            "logical_tokens": len(self.logical_tokens),
            "sealed_pages": len(self.pages),
            "historical_pages": len(context.historical_page_ids),
            "selected_pages": len(plan.ordered_page_ids),
            "selected_tokens": len(plan.ordered_page_ids) * PAGE_SIZE,
            "selection_mode": self.args.selection_mode,
            "selector": plan.selector,
            "selector_config": dict(plan.selector_config),
            "query_embedding": query_metrics,
            "fallback": plan.selector_config.get("effective_mode") == "lexical_deterministic_fallback",
            "generated_tokens": generated_count,
            "boundary_tokens": boundary_count,
            "goal": _goal_metrics(goal),
            "runtime": stats,
            "latency_ms": (perf_counter() - started) * 1000.0,
        }
        self.args.metrics.parent.mkdir(parents=True, exist_ok=True)
        with self.args.metrics.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return text, record


def main() -> None:
    args = parse_args()
    if not 0 < args.active_ratio <= 1:
        raise ValueError("active ratio must be in (0, 1]")
    if args.goal_rewriter_timeout <= 0:
        raise ValueError("goal rewriter timeout must be positive")
    import atexit
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    patched = install_qwen35_logical_rope(model)
    shared_goal_rewriter = _create_goal_rewriter(args)
    sessions: dict[str, OnlineSelectorSession | None] = {"current": None}

    def write_transcript_event(event: Mapping[str, Any]) -> None:
        path = args.metrics.with_name("transcript_events.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def current_session(messages: Sequence[Mapping[str, Any]]) -> OnlineSelectorSession:
        session = sessions["current"]
        # The batch runner is deliberately single-worker. A fresh two-message
        # transcript after at least one completed turn marks the next instance.
        new_instance = session is not None and len(messages) == 2 and session.turn_id > 0
        accepted = True
        if session is not None and not new_instance and session.turn_id > 0:
            accepted = session.accepts(messages)
        compacted_context = session is not None and not new_instance and not accepted
        previous_session_id = session.session_id if compacted_context else None
        rebuild_reason = "transcript_rewrite_rebuild"
        if compacted_context and messages:
            latest_content = str(messages[-1].get("content", "")).lstrip()
            if latest_content.startswith("Format error:"):
                rebuild_reason = "format_retry_rebuild"
            elif latest_content.startswith("Your previous response reached the output token limit"):
                rebuild_reason = "length_retry_rebuild"
            write_transcript_event({
                "event": "session_rebuild",
                "reason": rebuild_reason,
                "session_id": previous_session_id,
                "turn_id": session.turn_id,
                "diagnostic": session.last_acceptance_diagnostic,
            })
        if new_instance or compacted_context:
            session.close()
            sessions["current"] = None
            del session
            gc.collect()
            torch.cuda.empty_cache()
        if sessions["current"] is None:
            sessions["current"] = OnlineSelectorSession(
                args,
                model,
                tokenizer,
                torch,
                goal_rewriter=shared_goal_rewriter,
            )
            if compacted_context:
                sessions["current"].resume_reason = rebuild_reason
                sessions["current"].previous_session_id = previous_session_id
        return sessions["current"]

    def close_current_session() -> None:
        session = sessions["current"]
        if session is not None:
            session.close()
        if shared_goal_rewriter is not None:
            shared_goal_rewriter.close()

    atexit.register(close_current_session)
    lock = RLock()
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "patched_full_attention_layers": patched}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{
            "id": args.served_model_name, "object": "model", "owned_by": "dram-kv"
        }]}

    @app.post("/v1/chat/completions")
    def completions(request: dict[str, Any]):
        try:
            with lock:
                messages = request.get("messages", [])
                content, record = current_session(messages).query(messages)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}")
        created = int(time())
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": args.served_model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": record["logical_tokens"] - record["generated_tokens"],
                "completion_tokens": record["generated_tokens"],
                "total_tokens": record["logical_tokens"],
            },
        })

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
