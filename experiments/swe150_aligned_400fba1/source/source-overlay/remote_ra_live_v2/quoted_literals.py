"""The upstream Goal features' quoted-literal pass in linear time (same result).

Why. Every goal_query Goal construction runs the upstream extractor
(``dram_kv/goals/features.py``, unchanged) on the session's previous action, the
full decoded reply of its prior turn. ``normalize_action`` replaces quoted
literals with ``_QUOTED_LITERAL_RE.sub("<str>", text)``, pattern
``(?s)(['"])(?:\\.|(?!\1).)*\1``. Its body may take a backslash with the next
character (``\\.``) or alone (``(?!\1).``). When a quote has no closing quote on
the greedy path, the regex engine tries every split of the backslashes behind
the quote it finally closes at (or behind the opening quote, when there is none)
before it answers: the time doubles with each backslash. CPython's regex engine
holds the GIL for the whole call, so every thread of the engine process stops
with it: no decode step, no reply to the manager's heartbeat.

- Slurm job 3429670 (runtime 120bc3c, Qwen goal_query, Terminus 2 on chat-text):
  the one reply of the run that hit max_tokens (4096 tokens, cut inside a JSON
  string) was the previous action of its session's next turn. That admission
  parked at its Goal, the GPU went idle within seconds, no engine thread logged
  again, and the heartbeat's 90-s deadline revoked the server.
- A reply cut like that, locally: 22 backslash escapes after the open quote take
  11 s, each further one doubles it; a 4096-token reply holds hundreds.
- Slurm job 3428631 (SWE goal, mswea-xml): the two Goals without a rewriter call
  that took longest (156 ms and 15 ms; median 0.6 ms) follow the two replies
  with the most such backslashes (14 and 10; the pass alone takes 139 ms and
  12 ms on them locally).

What. ``QuotedLiterals.sub`` returns what the upstream pattern's ``sub`` returns,
in time linear in the text. Scanning as ``re.sub`` does, at each quote ``q``:

- the greedy path: a backslash takes the next character, any other character
  but ``q`` is taken alone; the first ``q`` it reaches closes the match (the
  upstream regex's first try, which it keeps);
- no such ``q`` before the end: every ``q`` behind the opening one is then the
  second character of a backslash pair, and the backtracking keeps the longest
  match: up to the last ``q`` of the text, its backslash taken alone;
- no ``q`` behind the opening one at all: no match there; the scan goes on at
  the next character.

A greedy scan that reaches the end happens at most once per quote character, so
the pass reads the text a bounded number of times. test_goal_quoted_literals
compares both on every short text over the characters that matter and on random
ones.

Where. ``install`` replaces the module attribute ``_QUOTED_LITERAL_RE`` of the
loaded upstream features module (``query_selection.upstream``, at startup), after
checking that it is the upstream pattern and that both agree on probe texts; the
upstream file and its pin are unchanged, and so are every normalized action,
feature, Goal and hash. Anything else asked of the attribute goes to the upstream
pattern.
"""
from __future__ import annotations

import re

from .contract import require

# dram_kv/goals/features.py, as packaged (the raw string literal, byte for byte).
UPSTREAM_PATTERN = r"(?s)(['\"])(?:\\.|(?!\1).)*\1"
UPSTREAM_FLAGS = re.compile(UPSTREAM_PATTERN).flags
REPLACEMENT = '<str>'       # normalize_action's replacement (no group reference)

_QUOTE = re.compile(r"['\"]")
_STOP = {"'": re.compile(r"[\\']"), '"': re.compile(r'[\\"]')}

# Startup self-check texts: greedy closes, escaped quotes, backtracking to the
# last quote, no closing quote, both quote kinds, a trailing backslash. Short, so
# the upstream regex answers them at once.
PROBES = (
    '', 'plain', "echo 'a b' \"c\"", "'it''s'", r"'a\'b'", r"'a\'b", r"x 'a\'b\'c d",
    r'"a\"b\"c', r'"\\"x"', "'unclosed", "\"say 'hi\" 'x", r"'a\\", "'\\",
    '{"k": "v\\n\\"w\\"\\nx', "p 'q \"r\\\"s\" t\\'u",
)


class QuotedLiterals:
    """``_QUOTED_LITERAL_RE`` of the upstream features module, linear in the text."""

    def __init__(self, upstream):
        self.upstream = upstream

    def __getattr__(self, name):
        return getattr(self.upstream, name)        # pattern, flags, search, ...

    @staticmethod
    def spans(text):
        """(start, end) of each match ``re.sub`` replaces, in order."""
        n, position = len(text), 0
        while True:
            found = _QUOTE.search(text, position)
            if found is None:
                return
            start = found.start()
            quote, stop, j, close = text[start], _STOP[text[start]], start + 1, -1
            while j < n:
                hit = stop.search(text, j)
                if hit is None:
                    break
                k = hit.start()
                if text[k] == quote:
                    close = k
                    break
                j = k + 2               # a backslash takes the next character
            if close < 0:
                close = text.rfind(quote, start + 1)
                if close < 0:
                    position = start + 1
                    continue
            yield start, close + 1
            position = close + 1

    def sub(self, repl, string, count=0):
        if type(repl) is not str or '\\' in repl or type(string) is not str or type(count) is not int or count < 0:
            return self.upstream.sub(repl, string, count)   # not normalize_action's call
        pieces, last = [], 0
        for number, (start, end) in enumerate(self.spans(string), 1):
            pieces += (string[last:start], repl)
            last = end
            if number == count:
                break
        pieces.append(string[last:])
        return ''.join(pieces)


def install(features):
    """Startup: the upstream features module's quoted-literal pass, linear. Idempotent."""
    current = getattr(features, '_QUOTED_LITERAL_RE', None)
    if isinstance(current, QuotedLiterals):
        return current
    require(isinstance(current, re.Pattern) and current.pattern == UPSTREAM_PATTERN and
            current.flags == UPSTREAM_FLAGS, 'the upstream Goal features quoted-literal pattern changed')
    linear = QuotedLiterals(current)
    require(all(linear.sub(REPLACEMENT, text) == current.sub(REPLACEMENT, text) for text in PROBES),
            'the linear quoted-literal pass differs from the upstream pattern')
    features._QUOTED_LITERAL_RE = linear
    return linear
