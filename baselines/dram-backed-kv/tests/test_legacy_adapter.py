from __future__ import annotations

import unittest
from dataclasses import dataclass

from dram_kv.adapters import build_legacy_replay_bundle


@dataclass(frozen=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True)
class Turn:
    turn_id: int
    assistant_message_id: int
    sealed_history: Span
    selection_enabled: bool
    selected_page_ids: tuple[int, ...]
    selected_token_indices: tuple[int, ...]


@dataclass(frozen=True)
class Plan:
    message_spans: tuple[Span, ...]
    turns: tuple[Turn, ...]


def budget(selected, *, history_end, mandatory_end, page_size, target_ratio):
    del selected, mandatory_end, target_ratio
    pages = tuple(range((history_end + page_size - 1) // page_size))[-2:]
    tokens = tuple(
        token for page in pages
        for token in range(page * page_size, min((page + 1) * page_size, history_end))
    )
    return pages, tokens


class LegacyAdapterTests(unittest.TestCase):
    def test_framework_plan_is_the_rebuilt_engine_selection(self) -> None:
        plan = Plan(
            message_spans=(Span(0, 4), Span(4, 8)),
            turns=(Turn(2, 4, Span(0, 24), True, (0, 2), tuple(range(8)) + tuple(range(16, 24))),),
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "action"},
            {"role": "user", "content": "observation"},
            {"role": "assistant", "content": "next"},
        ]
        bundle = build_legacy_replay_bundle(
            plan, messages, session_id="s", page_size=8,
            active_ratio_budget=0.5, budget_function=budget,
        )
        self.assertEqual(bundle.engine_plan.turns[0].selected_page_ids, (1, 2))
        self.assertEqual(
            bundle.turns[0].selection.ordered_page_ids,
            ("p00000001", "p00000002"),
        )
        self.assertEqual(bundle.turns[0].context.current_observation, "observation")

    def test_drop_reactivation_is_applied_before_engine_rebuild(self) -> None:
        plan = Plan(
            message_spans=(Span(0, 4), Span(4, 8)),
            turns=(
                Turn(2, 4, Span(0, 16), True, (0, 1), tuple(range(16))),
                Turn(3, 6, Span(0, 24), True, (0, 2), tuple(range(8)) + tuple(range(16, 24))),
                Turn(4, 8, Span(0, 32), True, (0, 1, 3), tuple(range(16)) + tuple(range(24, 32))),
            ),
        )
        messages = [{"content": str(index)} for index in range(9)]
        bundle = build_legacy_replay_bundle(
            plan, messages, session_id="s", page_size=8,
            drop_reactivated_at_turn=(4,), budget_function=budget,
        )
        self.assertEqual(bundle.engine_plan.turns[2].selected_page_ids, (0, 3))
        self.assertNotIn("p00000001", bundle.turns[2].selection.ordered_page_ids)


if __name__ == "__main__":
    unittest.main()
