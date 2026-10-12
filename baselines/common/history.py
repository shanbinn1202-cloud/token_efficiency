"""Consolidated history implementation."""

from dataclasses import dataclass

@dataclass(frozen=True)
class Policy:
    mode: str
    recent: int = 3
    recall: int = 2
    recall_chars: int = 16000
    gate_limit: int = 8
    repeat_hint: bool = False
    adaptive: bool = False
    growth_every: int = 16
    max_recall: int = 6


def split_turns(messages: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    """Keep assistant calls and ALL their tool replies atomic.

    User-role format-error feedback remains attached to the preceding turn.
    If no tool call has occurred yet it stays in the initial prefix instead.
    """
    prefix, turns = [], []
    for message in messages:
        if message["role"] == "assistant":
            turns.append([message])
        elif turns:
            turns[-1].append(message)
        else:
            prefix.append(message)
    return prefix, turns
