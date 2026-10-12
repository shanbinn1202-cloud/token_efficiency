#!/usr/bin/env python3
"""Run the validated Qwen replay engine through new framework contracts."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dram_kv.adapters import build_legacy_replay_bundle
from dram_kv.artifacts import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-repo", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--framework-artifacts", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--switch-threshold-tokens", type=int)
    parser.add_argument("--active-ratio-budget", type=float)
    parser.add_argument("--force-full-active", action="store_true")
    parser.add_argument("--drop-reactivated-at-turn", action="append", type=int, default=[])
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backing-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--kl-chunk-tokens", type=int, default=64)
    parser.add_argument("--full-active-atol", type=float, default=0.02)
    parser.add_argument("--dense-full-check-turn", type=int, default=2)
    parser.add_argument("--dense-incremental-atol", type=float, default=2.0)
    parser.add_argument("--dense-incremental-nll-atol", type=float, default=0.1)
    parser.add_argument("--dense-incremental-top1-min", type=float, default=0.99)
    parser.add_argument("--measure-before-switch", action="store_true")
    parser.add_argument("--max-turns", type=int)
    return parser.parse_args()


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.legacy_repo))
    from scripts import kv_drift_replay as engine
    from scripts.kv_drift_core import ReplayPlan, build_replay_plan, pages_for_ratio_budget
    from transformers import AutoTokenizer

    annotation = load_object(args.annotation)
    trajectory = load_object(args.trajectory)
    messages = trajectory["messages"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    threshold = args.switch_threshold_tokens
    if threshold is None:
        threshold = int(annotation["oracle_policy"]["switch_threshold_tokens"])
    engine_plan = build_replay_plan(
        messages,
        annotation,
        engine._chat_renderer(tokenizer),
        page_size=args.page_size,
        switch_threshold_tokens=threshold,
    )
    if args.max_turns is not None:
        engine_plan = ReplayPlan(
            canonical_policy=engine_plan.canonical_policy,
            page_size=engine_plan.page_size,
            switch_threshold_tokens=engine_plan.switch_threshold_tokens,
            message_spans=engine_plan.message_spans,
            complete_prefix_lengths=engine_plan.complete_prefix_lengths,
            turns=engine_plan.turns[:args.max_turns],
        )
    session_id = str(annotation["trajectory"]["instance_id"])
    bundle = build_legacy_replay_bundle(
        engine_plan,
        messages,
        session_id=session_id,
        page_size=args.page_size,
        active_ratio_budget=args.active_ratio_budget,
        force_full_active=args.force_full_active,
        drop_reactivated_at_turn=args.drop_reactivated_at_turn,
        budget_function=pages_for_ratio_budget,
    )
    artifact_records = []
    for turn in bundle.turns:
        artifact_records.extend((
            {"record_type": "turn_context", **asdict(turn.context)},
            {"record_type": "page_catalog", **asdict(turn.catalog)},
            {"record_type": "goal_record", **asdict(turn.goal)},
            {"record_type": "selection_plan", **asdict(turn.selection)},
        ))
    write_jsonl(args.framework_artifacts, artifact_records)

    header = {
        "record_type": "framework_regression_metadata",
        "framework": "dram-backed-kv",
        "session_id": session_id,
        "trajectory": str(args.trajectory),
        "annotation": str(args.annotation),
        "page_size": args.page_size,
        "active_ratio_budget": args.active_ratio_budget,
        "force_full_active": args.force_full_active,
        "drop_reactivated_at_turn": args.drop_reactivated_at_turn,
        "selection_plan_consumed_by_engine": True,
        "legacy_engine_repo": str(args.legacy_repo),
        "backing_device": args.backing_device,
        "dry_run": args.dry_run,
    }
    records = [header]
    if args.dry_run:
        records.extend(engine._plan_records(bundle.engine_plan))
    else:
        engine_args = copy.copy(args)
        # Transformations are already captured in the framework SelectionPlans.
        engine_args.active_ratio_budget = None
        engine_args.drop_reactivated_at_turn = []
        records.extend(engine.run_model(engine_args, tokenizer, messages, bundle.engine_plan))
    engine._write_jsonl(args.output, records)


if __name__ == "__main__":
    main()

