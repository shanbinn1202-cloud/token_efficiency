#!/usr/bin/env python3
"""Schema-constrained local Qwen adapter for Goal rewrite JSON I/O.

By default this remains compatible with the one-shot ``CommandGoalRewriter``.
``--serve`` loads the model once, emits a readiness handshake, and then serves
the NDJSON protocol used by ``PersistentCommandGoalRewriter``.  The model sees
only the bounded request supplied by the core Goal boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from time import perf_counter
from typing import Any, Mapping


PROTOCOL = "dram_kv_rewriter_ndjson"
PROTOCOL_VERSION = 1
ALLOWED_FIELDS = {
    "description", "phase_hint", "files", "symbols", "tests", "errors",
    "evidence_needed", "status", "constraints",
}
PHASES = ("exploration", "hypothesis", "implementation", "verification")
STATUSES = ("active", "blocked", "verifying", "done")
MAX_ACTION_CHARS = 4_000
MAX_OBSERVATION_CHARS = 8_000
MAX_ROOT_CHARS = 8_000


class GenerationFailed(RuntimeError):
    """Generation completed, but its candidate could not cross validation."""

    def __init__(self, message: str, telemetry: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.telemetry = dict(telemetry)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--serve", action="store_true",
        help="load once and serve the versioned line-delimited JSON protocol",
    )
    return parser.parse_args()


def candidate_schema(*, constraints_allowed: bool) -> dict[str, Any]:
    """Return the exact JSON Schema enforced during token generation."""

    string_array = {
        "type": "array",
        "items": {"type": "string", "minLength": 1, "maxLength": 512},
        "maxItems": 64,
        "uniqueItems": True,
    }
    properties: dict[str, Any] = {
        "description": {"type": "string", "minLength": 1, "maxLength": 1200},
        "phase_hint": {"type": "string", "enum": list(PHASES)},
        "files": string_array,
        "symbols": string_array,
        "tests": string_array,
        "errors": string_array,
        "evidence_needed": string_array,
        "status": {"type": "string", "enum": list(STATUSES)},
    }
    if constraints_allowed:
        properties["constraints"] = string_array
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
        "minProperties": 1,
    }


def bounded_model_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Reassert the online context boundary before prompt construction."""

    allowed = {
        "root_goal", "current_goal", "latest_action", "latest_observation",
        "features", "trigger", "stats", "allow_constraint_updates",
    }
    unknown = set(request) - allowed
    if unknown:
        raise ValueError(f"request contains unknown fields: {sorted(unknown)}")
    return {
        "root_goal": str(request.get("root_goal", ""))[:MAX_ROOT_CHARS],
        "current_goal": request.get("current_goal", {}),
        "latest_action": str(request.get("latest_action", ""))[:MAX_ACTION_CHARS],
        "latest_observation": str(
            request.get("latest_observation", "")
        )[:MAX_OBSERVATION_CHARS],
        "features": request.get("features", {}),
        "trigger": request.get("trigger"),
        "stats": request.get("stats", {}),
        "allow_constraint_updates": request.get("allow_constraint_updates", False),
    }


def parse_exact_candidate(text: str) -> dict[str, Any]:
    """Require one JSON object with no prose before or after it."""

    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("model response must be one JSON object")
    unknown = set(value) - ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"model returned unknown/protected fields: {sorted(unknown)}")
    return value


class QwenGoalGenerator:
    """Optional-dependency model runtime, deliberately outside ``dram_kv``."""

    def __init__(self, args: argparse.Namespace) -> None:
        # These imports stay lazy so CPU Shadow and the core package have no ML
        # dependency. Missing LM Format Enforcer fails closed at startup.
        import torch
        from lmformatenforcer import JsonSchemaParser
        # lm-format-enforcer<=0.11 imports this base class from the pre-v5
        # Transformers module.  Transformers 5 moved it to
        # tokenization_utils_base; install a process-local compatibility alias
        # before importing the integration rather than weakening constraints.
        try:
            from transformers.tokenization_utils import PreTrainedTokenizerBase
        except ImportError:
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase
            import transformers.tokenization_utils as tokenization_utils

            tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.JsonSchemaParser = JsonSchemaParser
        self.build_prefix_fn = build_transformers_prefix_allowed_tokens_fn
        self.device = args.device
        self.max_new_tokens = args.max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            trust_remote_code=True,
        )
        dtype: Any = args.dtype
        if dtype != "auto":
            if not hasattr(torch, dtype):
                raise ValueError(f"unsupported torch dtype {dtype!r}")
            dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
            trust_remote_code=True,
        ).to(args.device)
        self.model.eval()

    def generate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValueError("request must be a RewriteRequest object")
        constraints_allowed = (
            request.get("trigger") == "user_request"
            and request.get("allow_constraint_updates") is True
        )
        schema = candidate_schema(constraints_allowed=constraints_allowed)
        system = (
            "Rewrite a selector-side retrieval Goal as exactly one JSON object "
            "matching the supplied schema. This is not an execution plan. Never "
            "emit root_goal, commands, steps, actions, next_action, reasoning, or "
            "unknown fields. Preserve constraints unless their field is present in "
            "the schema. Prefer concise existing entities and unique array items."
        )
        user = json.dumps(
            {
                "request": bounded_model_request(request),
                "output_json_schema": schema,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        # Qwen3 must not spend tokens on its thinking channel. Qwen2.5's Jinja
        # template safely ignores the extra keyword.
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.device)
        parser = self.JsonSchemaParser(schema)
        prefix_fn = self.build_prefix_fn(self.tokenizer, parser)
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        generation_started = perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_fn,
            )
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        generation_latency_ms = (perf_counter() - generation_started) * 1_000
        suffix = generated[0, inputs["input_ids"].shape[1]:]
        response_text = self.tokenizer.decode(suffix, skip_special_tokens=True)
        telemetry = {
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "output_tokens": int(suffix.shape[0]),
            "generation_latency_ms": generation_latency_ms,
        }
        try:
            candidate = parse_exact_candidate(response_text)
        except Exception as error:
            raise GenerationFailed(
                f"{type(error).__name__}: {error}", telemetry
            ) from error
        return {"candidate": candidate, "model_cost": 0.0, **telemetry}


def _write_json(value: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n")
    sys.stdout.flush()


def serve(generator: QwenGoalGenerator) -> None:
    _write_json({"protocol": PROTOCOL, "version": PROTOCOL_VERSION, "ready": True})
    for line in sys.stdin:
        request_id = ""
        try:
            envelope = json.loads(line)
            if not isinstance(envelope, dict) or set(envelope) != {"request_id", "request"}:
                raise ValueError("request envelope has unknown or missing fields")
            request_id = envelope["request_id"]
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("request_id must be a non-empty string")
            result = generator.generate(envelope["request"])
            _write_json({"request_id": request_id, "result": result})
        except Exception as error:  # the parent will deterministically fall back
            response = {
                "request_id": request_id,
                "error": f"{type(error).__name__}: {error}"[:1_000],
            }
            telemetry = getattr(error, "telemetry", None)
            if isinstance(telemetry, Mapping):
                response["telemetry"] = dict(telemetry)
            _write_json(response)


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    generator = QwenGoalGenerator(args)
    if args.serve:
        serve(generator)
        return
    request = json.load(sys.stdin)
    _write_json(generator.generate(request))


if __name__ == "__main__":
    main()
