"""Full-history selector and model adapter."""

from baselines.common.history import Policy, split_turns
from baselines.common.model import HistoryModel


POLICIES = {"full": Policy("full")}
class FullMemory:
    def __init__(self, policy):
        self.policy = policy
        self.goal_changes = 0
    def select(self, messages, gate=None):
        _, turns = split_turns(messages)
        return messages, {"kept": list(range(len(turns))), "available": len(turns), "goal": 0, "gate": "warmup"}


class FullHistoryModel(HistoryModel):
    def __init__(self, *, variant="full", policy=None, **kwargs):
        super().__init__(variant=variant, memory=FullMemory(policy or POLICIES[variant]), **kwargs)
