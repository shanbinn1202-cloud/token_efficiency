"""Session-local phase detection and bounded stuck statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .features import FeatureSnapshot


PHASES = ("exploration", "hypothesis", "implementation", "verification", "stuck")
_PHASE_RANK = {
    "exploration": 0,
    "hypothesis": 1,
    "implementation": 2,
    "verification": 3,
}


@dataclass(frozen=True)
class PhaseTrackerConfig:
    """Auditable defaults taken from the convergence design."""

    hysteresis_steps: int = 2
    stuck_floor: int = 10
    failure_threshold: int = 5
    same_file_edit_threshold: int = 6
    read_without_edit_threshold: int = 15
    repeated_action_threshold: int = 8
    cooldown_steps: int = 8
    max_tracked_files: int = 32

    def __post_init__(self) -> None:
        if any(value <= 0 for value in asdict(self).values()):
            raise ValueError("all phase tracker thresholds must be positive")


@dataclass(frozen=True)
class StuckStats:
    step: int
    consecutive_failures: int
    same_file_edits: int
    same_file: str | None
    reads_without_edit: int
    repeated_action: int
    normalized_action: str
    cooldown_remaining: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseTransition:
    previous: str
    current: str
    mode: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseSnapshot:
    """Complete, JSON-compatible state-detector output for one boundary."""

    raw_phase: str
    stable_phase: str
    raw_trigger: str
    transition: PhaseTransition | None
    trigger: str | None
    triggers: tuple[str, ...]
    stats: StuckStats

    def __post_init__(self) -> None:
        if self.raw_phase not in PHASES or self.stable_phase not in PHASES:
            raise ValueError("invalid phase snapshot")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PhaseTracker:
    """Track one session's phase without retaining its trajectory.

    Construct one instance per session.  Counters saturate at their thresholds,
    the per-file table is capped, and only the last normalized action is kept.
    """

    def __init__(
        self,
        config: PhaseTrackerConfig | None = None,
        *,
        initial_phase: str = "exploration",
    ) -> None:
        if initial_phase not in PHASES or initial_phase == "stuck":
            raise ValueError("initial_phase must be one of the four working phases")
        self.config = config or PhaseTrackerConfig()
        self._stable_phase = initial_phase
        self._pending_phase: str | None = None
        self._pending_count = 0
        self._step = 0
        self._consecutive_failures = 0
        self._reads_without_edit = 0
        self._last_action = ""
        self._repeated_action = 0
        self._edit_counts: dict[str, int] = {}
        self._edit_recency: dict[str, int] = {}
        self._cooldown_remaining = 0
        self._reset_after_trigger = False

    @property
    def stable_phase(self) -> str:
        return self._stable_phase

    def update(self, features: FeatureSnapshot) -> PhaseSnapshot:
        """Consume exactly one latest-boundary feature snapshot."""

        if self._reset_after_trigger:
            self._reset_stuck_counters()
            self._reset_after_trigger = False

        self._step += 1
        cooldown_active = self._cooldown_remaining > 0
        self._update_counters(features)
        raw_phase, raw_trigger = self._detect_working_phase(features)

        triggers = self._stuck_triggers()
        if self._step < self.config.stuck_floor or cooldown_active:
            triggers = ()
        if triggers:
            raw_phase = "stuck"
            raw_trigger = f"stuck:{triggers[0]}"

        transition = self._stabilize(raw_phase)
        if triggers:
            self._cooldown_remaining = self.config.cooldown_steps
            self._reset_after_trigger = True
        elif cooldown_active:
            self._cooldown_remaining -= 1

        stats = self._stats()
        return PhaseSnapshot(
            raw_phase=raw_phase,
            stable_phase=self._stable_phase,
            raw_trigger=raw_trigger,
            transition=transition,
            trigger=triggers[0] if triggers else None,
            triggers=triggers,
            stats=stats,
        )

    def _update_counters(self, features: FeatureSnapshot) -> None:
        cfg = self.config
        if features.return_code is not None and features.return_code != 0:
            self._consecutive_failures = min(
                cfg.failure_threshold, self._consecutive_failures + 1
            )
        else:
            self._consecutive_failures = 0

        if features.action_kind == "edit":
            self._reads_without_edit = 0
            edit_file = next(iter(features.action.files or features.files), None)
            if edit_file:
                if (
                    edit_file not in self._edit_counts
                    and len(self._edit_counts) >= cfg.max_tracked_files
                ):
                    oldest = min(self._edit_recency, key=self._edit_recency.__getitem__)
                    del self._edit_counts[oldest]
                    del self._edit_recency[oldest]
                self._edit_counts[edit_file] = min(
                    cfg.same_file_edit_threshold,
                    self._edit_counts.get(edit_file, 0) + 1,
                )
                self._edit_recency[edit_file] = self._step
        elif features.action_kind == "inspect":
            self._reads_without_edit = min(
                cfg.read_without_edit_threshold, self._reads_without_edit + 1
            )

        action = features.normalized_action
        if action and action == self._last_action:
            self._repeated_action = min(
                cfg.repeated_action_threshold, self._repeated_action + 1
            )
        elif action:
            self._last_action = action
            self._repeated_action = 1
        else:
            self._last_action = ""
            self._repeated_action = 0

    def _detect_working_phase(self, features: FeatureSnapshot) -> tuple[str, str]:
        # Modification wins even if patch content mentions reads/tests.
        if features.action_kind == "edit":
            return "implementation", "action_kind:edit"
        if features.action_kind == "test" or features.test_verdict != "not_run":
            return "verification", "test_evidence"
        if (
            features.action_kind == "diagnose"
            or features.confirmed_errors
            or features.confirmed_top_frame
        ):
            return "hypothesis", "diagnostic_evidence"
        if features.action_kind == "inspect":
            return "exploration", "action_kind:inspect"
        return self._working_fallback(), "no_phase_signal"

    def _working_fallback(self) -> str:
        return "exploration" if self._stable_phase == "stuck" else self._stable_phase

    def _stuck_triggers(self) -> tuple[str, ...]:
        cfg = self.config
        triggers: list[str] = []
        if self._consecutive_failures >= cfg.failure_threshold:
            triggers.append("failure_return")
        if (
            self._edit_counts
            and max(self._edit_counts.values()) >= cfg.same_file_edit_threshold
        ):
            triggers.append("same_file_edit")
        if self._reads_without_edit >= cfg.read_without_edit_threshold:
            triggers.append("read_without_edit")
        if self._repeated_action >= cfg.repeated_action_threshold:
            triggers.append("repeated_action")
        return tuple(triggers)

    def _stabilize(self, raw_phase: str) -> PhaseTransition | None:
        previous = self._stable_phase
        if raw_phase == previous:
            self._pending_phase = None
            self._pending_count = 0
            return None

        if raw_phase == "stuck":
            self._stable_phase = raw_phase
            self._pending_phase = None
            self._pending_count = 0
            return PhaseTransition(previous, raw_phase, "stuck_immediate")

        is_forward = previous != "stuck" and (
            _PHASE_RANK[raw_phase] > _PHASE_RANK[previous]
        )
        if is_forward:
            self._stable_phase = raw_phase
            self._pending_phase = None
            self._pending_count = 0
            return PhaseTransition(previous, raw_phase, "forward_immediate")

        if raw_phase == self._pending_phase:
            self._pending_count += 1
        else:
            self._pending_phase = raw_phase
            self._pending_count = 1
        if self._pending_count < self.config.hysteresis_steps:
            return None

        self._stable_phase = raw_phase
        self._pending_phase = None
        self._pending_count = 0
        return PhaseTransition(previous, raw_phase, "hysteresis")

    def _stats(self) -> StuckStats:
        if self._edit_counts:
            same_file = min(
                self._edit_counts,
                key=lambda path: (-self._edit_counts[path], -self._edit_recency[path], path),
            )
            same_file_edits = self._edit_counts[same_file]
        else:
            same_file = None
            same_file_edits = 0
        return StuckStats(
            step=self._step,
            consecutive_failures=self._consecutive_failures,
            same_file_edits=same_file_edits,
            same_file=same_file,
            reads_without_edit=self._reads_without_edit,
            repeated_action=self._repeated_action,
            normalized_action=self._last_action,
            cooldown_remaining=self._cooldown_remaining,
        )

    def _reset_stuck_counters(self) -> None:
        self._consecutive_failures = 0
        self._reads_without_edit = 0
        self._last_action = ""
        self._repeated_action = 0
        self._edit_counts.clear()
        self._edit_recency.clear()
