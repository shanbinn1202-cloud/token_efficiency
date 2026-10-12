"""Dependency-free legacy smoke and Goal Constructor Shadow replay CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
from typing import Sequence

from .artifacts import write_jsonl
from .backends import ShadowBackend
from .evaluation import calibrate_recorded_thresholds, write_shadow_trace
from .goals import HeuristicGoalConstructor, RawObjectiveGoalConstructor
from .paging import TokenSegment, build_page_catalog
from .pipeline import ExperimentPipeline
from .replay.goal_shadow import (
    SHADOW_CONFIGS,
    load_shadow_replay,
    run_shadow_matrix,
)
from .replay.swe_agent import build_semantic_shadow_payload, load_swe_agent_bundle
from .schema import TurnContext
from .selectors import RecentSelector


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal", choices=("raw", "heuristic"), default="raw")
    parser.add_argument(
        "--shadow-config",
        choices=("all", *SHADOW_CONFIGS),
        help="run the Goal Constructor replay instead of the legacy smoke path",
    )
    parser.add_argument(
        "--slm-command",
        help=(
            "optional quoted argv for heuristic_slm JSON stdin/stdout adapter; "
            "the deterministic fake is used when omitted"
        ),
    )
    parser.add_argument(
        "--no-slm-init",
        action="store_true",
        help="skip the optional one-time initializer call in heuristic_slm",
    )
    parser.add_argument("--slm-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--slm-persistent", action="store_true",
        help="treat --slm-command as a persistent NDJSON server command",
    )
    parser.add_argument("--slm-startup-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--recent-turns", type=int, default=2)
    parser.add_argument("--recent-quota", type=int, default=2)
    parser.add_argument("--page-budget", type=int, default=8)
    parser.add_argument(
        "--calibrate-thresholds", action="store_true",
        help="write a bounded per-trajectory heuristic threshold sweep",
    )
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--bytes-per-token", type=int, default=0)
    return parser.parse_args(argv)


def _run_legacy(args: argparse.Namespace) -> None:
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    segments = tuple(TokenSegment(**segment) for segment in payload["segments"])
    catalog = build_page_catalog(
        payload["session_id"], segments,
        page_size=args.page_size,
        bytes_per_token=args.bytes_per_token,
    )
    constructor = (
        RawObjectiveGoalConstructor() if args.goal == "raw" else HeuristicGoalConstructor()
    )
    pipeline = ExperimentPipeline(
        constructor,
        RecentSelector(recent_turns=args.recent_turns),
        ShadowBackend(),
    )
    records = []
    for raw in payload["turns"]:
        context = TurnContext(**raw)
        result = pipeline.run_turn(context, catalog)
        records.extend((result.goal, result.selection, result.backend_metrics))
    write_jsonl(args.output, records)


def _run_goal_shadow(args: argparse.Namespace) -> None:
    if args.input.suffix.lower() == ".zip":
        payload = build_semantic_shadow_payload(
            load_swe_agent_bundle(args.input),
            page_budget=args.page_budget,
            recent_turns=args.recent_turns,
            recent_quota=args.recent_quota,
        )
    else:
        payload = load_shadow_replay(args.input)
    names = SHADOW_CONFIGS if args.shadow_config == "all" else (args.shadow_config,)
    slm_command = shlex.split(args.slm_command) if args.slm_command else None
    runs = run_shadow_matrix(
        payload,
        configs=names,
        slm_command=slm_command,
        initialize_with_rewriter=not args.no_slm_init,
        slm_timeout_seconds=args.slm_timeout_seconds,
        slm_persistent=args.slm_persistent,
        slm_startup_timeout_seconds=args.slm_startup_timeout_seconds,
    )
    if args.shadow_config == "all":
        args.output.mkdir(parents=True, exist_ok=True)
        for run in runs:
            write_shadow_trace(
                args.output / f"{run.config_name}.jsonl",
                run.turns,
                run.summary,
            )
        comparison = {
            "record_type": "shadow_comparison",
            "schema_version": "1.0.0",
            "input": str(args.input),
            "configs": [run.config_name for run in runs],
            "summaries": {run.config_name: run.summary for run in runs},
        }
        (args.output / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if args.calibrate_thresholds:
            (args.output / "calibration.json").write_text(
                json.dumps(
                    calibrate_recorded_thresholds(payload),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
    else:
        if args.calibrate_thresholds:
            raise SystemExit("--calibrate-thresholds requires --shadow-config all")
        run = runs[0]
        write_shadow_trace(args.output, run.turns, run.summary)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.slm_persistent and not args.slm_command:
        raise SystemExit("--slm-persistent requires --slm-command")
    if args.shadow_config:
        _run_goal_shadow(args)
    else:
        if args.slm_command or args.no_slm_init or args.slm_persistent:
            raise SystemExit(
                "--slm-command/--no-slm-init/--slm-persistent require --shadow-config"
            )
        _run_legacy(args)


if __name__ == "__main__":
    main()
