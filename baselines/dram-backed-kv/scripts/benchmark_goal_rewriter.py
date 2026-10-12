#!/usr/bin/env python3
"""Run a fixed RewriteRequest set through one persistent rewriter command."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shlex
from typing import Any, Mapping

from dram_kv.goals import PersistentCommandGoalRewriter
from dram_kv.goals.rewriter import (
    RewriteCommandError,
    RewriteRequest,
    RewriteValidationError,
)
from dram_kv.schema import RetrievalGoal


TUPLE_FIELDS = (
    "files", "symbols", "tests", "errors", "evidence_needed", "constraints",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--command", required=True, help="quoted persistent server argv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=300.0)
    return parser.parse_args()


def load_requests(path: Path) -> tuple[Mapping[str, Any], ...]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
        return tuple(value)
    raise ValueError("requests must be one object, an object array, or JSONL objects")


def request_from_payload(payload: Mapping[str, Any]) -> RewriteRequest:
    allowed = {
        "root_goal", "current_goal", "latest_action", "latest_observation",
        "features", "trigger", "stats", "allow_constraint_updates",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"request contains unknown fields: {sorted(unknown)}")
    current = payload.get("current_goal")
    if not isinstance(current, Mapping):
        raise ValueError("current_goal must be an object")
    current_values = dict(current)
    for field in TUPLE_FIELDS:
        current_values[field] = tuple(current_values.get(field, ()))
    return RewriteRequest(
        root_goal=payload["root_goal"],
        current_goal=RetrievalGoal(**current_values),
        latest_action=payload.get("latest_action", ""),
        latest_observation=payload.get("latest_observation", ""),
        features=payload.get("features", {}),
        trigger=payload["trigger"],
        stats=payload.get("stats", {}),
        allow_constraint_updates=payload.get("allow_constraint_updates", False),
    )


def main() -> None:
    args = parse_args()
    argv = tuple(shlex.split(args.command))
    backend = PersistentCommandGoalRewriter(
        argv,
        timeout_seconds=args.timeout_seconds,
        startup_timeout_seconds=args.startup_timeout_seconds,
    )
    records: list[dict[str, Any]] = []
    payloads = load_requests(args.requests)
    try:
        for index, payload in enumerate(payloads, 1):
            request = request_from_payload(payload)
            try:
                result = backend.rewrite(request)
                records.append({
                    "record_type": "rewrite_result",
                    "request_index": index,
                    "success": True,
                    "candidate": asdict(result.candidate),
                    "latency_ms": result.latency_ms,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "model_cost": result.model_cost,
                })
            except (RewriteCommandError, RewriteValidationError) as error:
                records.append({
                    "record_type": "rewrite_result",
                    "request_index": index,
                    "success": False,
                    "error_type": type(error).__name__,
                    "error": str(error)[:1_000],
                })
    finally:
        backend.close()
    successes = [record for record in records if record["success"]]
    latencies = sorted(record["latency_ms"] for record in successes)
    p95_index = max(0, (95 * len(latencies) + 99) // 100 - 1)
    manifest = {
        "record_type": "rewrite_benchmark_manifest",
        "schema_version": "1.0.0",
        "requests": str(args.requests),
        "requests_sha256": sha256(args.requests.read_bytes()).hexdigest(),
        "request_count": len(payloads),
        "command_argv": list(argv),
        "command_sha256": sha256(
            json.dumps(argv, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "persistent": True,
    }
    summary = {
        "record_type": "rewrite_benchmark_summary",
        "requests": len(records),
        "successes": len(successes),
        "failures": len(records) - len(successes),
        "valid_rate": len(successes) / len(records) if records else 0.0,
        "latency_ms": {
            "p50": latencies[(len(latencies) - 1) // 2] if latencies else 0.0,
            "p95": latencies[p95_index] if latencies else 0.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in (manifest, *records, summary)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
