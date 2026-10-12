"""Small deterministic threshold sweep for recorded Goal Shadow traces."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Mapping

from ..goals.state import PhaseTrackerConfig


def calibrate_recorded_thresholds(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rank a bounded grid by Oracle Recall and control-plane churn.

    This is per-trajectory calibration, not a claim that one SWE-bench instance
    estimates globally valid thresholds.
    """

    # Lazy import keeps the replay -> evaluation.shadow dependency acyclic.
    from ..replay.goal_shadow import run_shadow_configuration

    base = PhaseTrackerConfig()
    candidates: list[dict[str, Any]] = []
    for entity_drift_steps in (2, 3, 4):
        for hysteresis_steps in (2, 3):
            for stuck_scale in ("default", "sensitive"):
                phase = replace(
                    base,
                    hysteresis_steps=hysteresis_steps,
                    failure_threshold=3 if stuck_scale == "sensitive" else 5,
                    same_file_edit_threshold=4 if stuck_scale == "sensitive" else 6,
                    repeated_action_threshold=5 if stuck_scale == "sensitive" else 8,
                    read_without_edit_threshold=10 if stuck_scale == "sensitive" else 15,
                )
                run = run_shadow_configuration(
                    payload,
                    "heuristic_only",
                    phase_config=phase,
                    entity_drift_steps=entity_drift_steps,
                )
                summary = run.summary
                recall = summary["shadow_evidence_recall_at_budget"]
                update_rate = float(summary["effective_retrieval_goal_update_rate"])
                rewrite_rate = float(summary["decision_counts"].get("REWRITE", 0)) / max(
                    1, int(summary["maintenance_boundaries"])
                )
                score = float(recall["micro"]) - 0.10 * update_rate - 0.02 * rewrite_rate
                candidates.append({
                    "candidate_id": (
                        f"entity{entity_drift_steps}-hyst{hysteresis_steps}-{stuck_scale}"
                    ),
                    "entity_drift_steps": entity_drift_steps,
                    "phase": asdict(phase),
                    "micro_recall": recall["micro"],
                    "macro_recall": recall["macro"],
                    "goal_update_rate": update_rate,
                    "rewrite_rate": rewrite_rate,
                    "decision_counts": summary["decision_counts"],
                    "trigger_counts": summary["trigger_counts"],
                    "score": score,
                })
    candidates.sort(key=lambda item: (-item["score"], item["candidate_id"]))
    return {
        "record_type": "threshold_calibration",
        "schema_version": "1.0.0",
        "scope": "single_recorded_trajectory",
        "objective": "maximize Oracle older-evidence Recall@Budget with churn penalties",
        "score_formula": "micro_recall - 0.10*goal_update_rate - 0.02*rewrite_rate",
        "candidate_count": len(candidates),
        "recommended": candidates[0],
        "candidates": candidates,
        "limitations": [
            "one trajectory cannot establish globally valid thresholds",
            "Oracle labels retrieval evidence, not ground-truth KEEP/PATCH/REWRITE boundaries",
        ],
    }


__all__ = ["calibrate_recorded_thresholds"]
