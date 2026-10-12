#!/usr/bin/env python3
"""Compare validated Goal candidates from two fixed-request benchmarks."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from benchmark_goal_rewriter import load_requests, request_from_payload
from dram_kv.goals import DeterministicFakeRewriter


ENTITY_FIELDS = ("files", "symbols", "tests", "errors")
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:$-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object record")
            result.append(value)
    return result


def result_index(path: Path, request_count: int) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    records = read_jsonl(path)
    summaries = [item for item in records if item.get("record_type") == "rewrite_benchmark_summary"]
    results = {
        int(item["request_index"]): item
        for item in records if item.get("record_type") == "rewrite_result"
    }
    if len(summaries) != 1 or set(results) != set(range(1, request_count + 1)):
        raise ValueError(f"{path} does not cover the fixed request set exactly")
    return results, summaries[0]


def normalized(values: Sequence[Any]) -> set[str]:
    return {str(item).strip().casefold() for item in values if str(item).strip()}


def ratio(found: int, total: int) -> float | None:
    return found / total if total else None


def candidate_metrics(
    payload: Mapping[str, Any], candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    features = payload.get("features")
    current = payload.get("current_goal")
    if not isinstance(features, Mapping) or not isinstance(current, Mapping):
        raise ValueError("fixed request lacks features/current_goal")
    feature_total = feature_found = current_total = current_found = 0
    field_metrics: dict[str, Any] = {}
    for field in ENTITY_FIELDS:
        desired = normalized(features.get(field, ()))
        retained = normalized(current.get(field, ()))
        actual = normalized(candidate.get(field, ()))
        feature_total += len(desired)
        feature_found += len(desired & actual)
        current_total += len(retained)
        current_found += len(retained & actual)
        field_metrics[field] = {
            "feature_entities": len(desired),
            "feature_entities_found": len(desired & actual),
            "current_entities": len(retained),
            "current_entities_retained": len(retained & actual),
        }
    focus_tokens = set()
    for field in ENTITY_FIELDS:
        for value in features.get(field, ()):
            focus_tokens.update(token.casefold() for token in TOKEN_RE.findall(str(value)))
    description_tokens = {
        token.casefold() for token in TOKEN_RE.findall(str(candidate.get("description", "")))
    }
    return {
        "feature_entity_total": feature_total,
        "feature_entity_found": feature_found,
        "feature_entity_recall": ratio(feature_found, feature_total),
        "current_entity_total": current_total,
        "current_entity_retained": current_found,
        "current_entity_retention": ratio(current_found, current_total),
        "description_chars": len(str(candidate.get("description", ""))),
        "description_focus_token_coverage": ratio(
            len(focus_tokens & description_tokens), len(focus_tokens)
        ),
        "phase_matches_deterministic_reference": (
            candidate.get("phase_hint") == reference.get("phase_hint")
        ),
        "status_matches_deterministic_reference": (
            candidate.get("status") == reference.get("status")
        ),
        "constraints_preserved": (
            tuple(candidate.get("constraints", ())) == tuple(current.get("constraints", ()))
        ),
        "field_metrics": field_metrics,
    }


def aggregate(records: Sequence[Mapping[str, Any]], raw_summary: Mapping[str, Any]) -> dict[str, Any]:
    successful = [item for item in records if item.get("success")]
    metrics = [item["quality"] for item in successful]
    feature_total = sum(int(item["feature_entity_total"]) for item in metrics)
    feature_found = sum(int(item["feature_entity_found"]) for item in metrics)
    current_total = sum(int(item["current_entity_total"]) for item in metrics)
    current_found = sum(int(item["current_entity_retained"]) for item in metrics)
    focus_values = [
        float(item["description_focus_token_coverage"])
        for item in metrics if item["description_focus_token_coverage"] is not None
    ]
    return {
        "requests": len(records),
        "valid_outputs": len(successful),
        "valid_rate": ratio(len(successful), len(records)),
        "failure_types": dict(sorted(Counter(
            str(item.get("error_type", "unknown"))
            for item in records if not item.get("success")
        ).items())),
        "latency_ms": raw_summary.get("latency_ms"),
        "feature_entity_recall_micro": ratio(feature_found, feature_total),
        "current_entity_retention_micro": ratio(current_found, current_total),
        "description_focus_token_coverage_macro": (
            sum(focus_values) / len(focus_values) if focus_values else None
        ),
        "phase_reference_agreement": ratio(sum(
            bool(item["phase_matches_deterministic_reference"]) for item in metrics
        ), len(metrics)),
        "status_reference_agreement": ratio(sum(
            bool(item["status_matches_deterministic_reference"]) for item in metrics
        ), len(metrics)),
        "constraint_preservation_rate": ratio(sum(
            bool(item["constraints_preserved"]) for item in metrics
        ), len(metrics)),
        "mean_description_chars": (
            sum(int(item["description_chars"]) for item in metrics) / len(metrics)
            if metrics else None
        ),
    }


def main() -> None:
    args = parse_args()
    payloads = load_requests(args.requests)
    left, left_summary = result_index(args.left, len(payloads))
    right, right_summary = result_index(args.right, len(payloads))
    references = []
    for payload in payloads:
        request = request_from_payload(payload)
        references.append(asdict(DeterministicFakeRewriter().rewrite(request).candidate))

    per_request = []
    model_records: dict[str, list[dict[str, Any]]] = {
        args.left_name: [], args.right_name: [],
    }
    for index, (payload, reference) in enumerate(zip(payloads, references), 1):
        item: dict[str, Any] = {"request_index": index, "trigger": payload["trigger"]}
        for name, source in ((args.left_name, left), (args.right_name, right)):
            raw = source[index]
            record = {
                "request_index": index,
                "success": bool(raw["success"]),
            }
            if raw["success"]:
                record["quality"] = candidate_metrics(payload, raw["candidate"], reference)
            else:
                record["error_type"] = raw.get("error_type")
                record["error"] = raw.get("error")
            model_records[name].append(record)
            item[name] = record
        per_request.append(item)

    output = {
        "schema_name": "dram_kv_goal_rewriter_quality_comparison",
        "schema_version": "1.0.0",
        "requests": str(args.requests),
        "request_count": len(payloads),
        "interpretation": {
            "validated_output": "constrained output accepted by the core validator",
            "feature_entity_recall": "exact normalized bounded-feature entities retained in the candidate",
            "current_entity_retention": "existing Goal entities retained in the candidate",
            "reference_agreement": "agreement with deterministic fallback, not semantic ground truth",
            "limitation": "the cohort has one REWRITE gold label; these are structural and retrieval-focus proxies",
        },
        "models": {
            args.left_name: aggregate(model_records[args.left_name], left_summary),
            args.right_name: aggregate(model_records[args.right_name], right_summary),
        },
        "per_request": per_request,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["models"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
