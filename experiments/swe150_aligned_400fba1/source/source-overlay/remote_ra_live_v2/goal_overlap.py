"""Phase 1 while the Goal is built (``serve.py --goal-overlap``; docs/goal-overlap/DESIGN.md).

Without the flag a goal_query admission builds its Goal (the Qwen3-0.6B rewriter on
the turns that call it: seconds), then runs phase 1 with the Goal forward in the same
worker RPC. The Goal forward is a cold forward: the Goal query text alone, from
position 0, on its own fresh blocks, with no history KV, none of the turn's new tokens
and no GDN carry (a request without computed tokens starts from a zero state). It
reads nothing phase 1 reads or writes.

With the flag, on a server that parks goal_query admissions (sessions decoding
together, a Goal rewriter):

1. ``QuerySelectionAuthority.begin_turn_overlap`` builds the TurnContext where
   begin_turn does and yields the Goal as a ``goal_async.GoalStart``: it goes to the
   Goal thread (the session's with --admission-overlap) and the admission goes on;
2. phase 1 runs without the Goal forward: in place, or with --phase1-in-step inside
   a decode step (a lease without a Goal forward is accepted there);
3. ``StartedTurn.finish`` joins the Goal (a ``goal_async.GoalJoin``, parked only while
   the Goal is still being built), then the rest of begin_turn: the Goal text, the
   Goal cache, the Goal payload;
4. a turn that needs a Goal forward runs it as its own lease
   (``Phase1Coordinator.goal_forward_steps``): one ``GoalForward`` work run in place
   (a separate probe-only pass on the worker's ``goal_forward`` operation), then its
   joined drain; ``plan_turn``, the adoption and the reservation follow as today.

The Goal forward is today's: the same tokens, positions 0..G-1, no context, its own
zeroed blocks of the same widths, the same eager one-request batch and the same
pre-final-norm last row; only its block ids differ. Its vector, the Page scores
against it and the selection are bit-identical to the flag-off server's. Only the
time at which phase 1 runs changes, and a lease then waits only for undrained leases
(``Phase1Coordinator.busy``): a drained phase-1 record that waits for its Goal holds
back no other admission's phase 1.

The window layout (Gemma 4, docs/goal-overlap/DESIGN.md section 9): the same order and
the same cold Goal forward, on its own FA and sliding blocks in the layout's Goal
transient (``WindowPhase1Coordinator._goal_lease``). A drained phase 1 that waits for its
Goal first frees the session's pinned blocks its adoption would free
(``WindowPins.trim``), and the sliding ledger takes what it still holds off its base
(``window_capacity.ledger_ahead_split``). A server without a Goal park table (one
session) keeps today's order: the flag is inert there. Startup only; nothing here runs
per step.
"""
from __future__ import annotations

import time

from .contract import require
import policy_registry

FLAG = 'goal_overlap'


class GoalForward:
    """The separate Goal forward of one turn, as goal_forward_steps yields it: run in place
    on the engine thread (never parked), its RPC time recorded as ``goal_forward_ms``."""
    kind = 'goal_forward'

    def __init__(self, coordinator, request, payload, query_tokens):
        self.coordinator, self.request, self.payload = coordinator, request, payload
        self.query_tokens = query_tokens
        self.ns = None

    def run(self):
        started = time.perf_counter_ns()
        try:
            return self.coordinator.proxy.call('goal_forward', self.payload, query_tokens=self.query_tokens)
        finally:
            self.ns = time.perf_counter_ns() - started


def attach_engine(core, admission):
    """Engine startup (compose_engine), after goal_async.attach: with the flag on, a
    goal_query server whose Goals park overlaps phase 1 with them. Returns True when
    attached, None when the flag is off or no park table exists (one session)."""
    from .admission_flags import on
    if not on(admission, FLAG):
        return None
    endpoint = core._remote_ra_private_v2_endpoint
    selection, phase1 = endpoint.selection, endpoint.phase1
    require(selection is not None and policy_registry.goal_engine(selection.policy) and selection.rewriter is not None,
            '--goal-overlap needs goal_query with a Goal rewriter')
    # The window layout (Gemma 4): its ledger lets an admission wait after phase 1 only
    # while no phase 1 is leased across a park (no --phase1-in-step on this layout).
    require(getattr(phase1, 'window', None) is None or not getattr(phase1, 'in_step', False),
            '--goal-overlap on the window layout runs without --phase1-in-step')
    require(not endpoint.goal_overlap and not phase1.split and not phase1.records,
            'goal overlap attached twice or after phase 1 started')
    goals = getattr(core, '_remote_ra_goal_admissions', None)
    if goals is None or goals.goals is not True:
        return None             # no Goal park table: every admission keeps today's order
    endpoint.goal_overlap = phase1.split = True
    return True
