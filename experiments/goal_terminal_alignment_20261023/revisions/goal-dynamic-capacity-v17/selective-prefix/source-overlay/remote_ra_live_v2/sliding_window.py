"""Sliding-window geometry of the Gemma 4 profile (gemma-s1): pure integer arithmetic, no vLLM.

Selection (design 2.1-2.5) chooses Pages of the global full-attention layers only:
Pages, budgets, mandatory and recent Pages, vectors and restores are the Qwen ones,
applied to the global layers' KV. The sliding-window layers always attend to the
exact last W - 1 logical tokens plus the current one, from their own window blocks
in pool 1 (``kv_layout.WindowLayout``), whatever the global Pages' residency; the
window is outside the active-token budget (D-1).

Positions of a continuing turn (README 5, the runtime row's ``stage_b``): sealed
history before the candidate end L (the candidate Pages, of which A_t is selected),
the live tail [L, C), the phase-1 run [C, e) (boundary, user turn, observation),
then the generation header and decode from e.

* At the first phase-2 step (the query at position e) the sliding layers read
  [e - W + 1, e]. If the phase-1 run holds at least W - 1 tokens the window lies
  inside it; otherwise it reaches back into sealed history, possibly into Pages the
  global layers do not see this turn (``window_evidence``).
* vLLM's SlidingWindowManager frees a block once every token in it lies before the
  window of the next query (``get_num_skipped_tokens(n) = max(0, n - W + 1)``), so a
  request that has computed n tokens keeps blocks [floor((n - W + 1) / B), ceil(n / B)),
  at most ceil((W - 1) / B) + 1 of them (65 at W = 1024, B = 16), behind a null head.
* Window lifecycle with the window on the GPU between turns (W-GPU, D-3): the drain
  of a turn that computed C' tokens pins ``kept_blocks(C')``; phase 1 of the next turn
  borrows them behind its null head and appends fresh blocks for [C, e)
  (``phase1_row``); its drain keeps ``kept_blocks(e)`` for phase 2 (adopted, installed
  at admission behind the null head ``phase2_head``) and releases the rest
  (``phase1_drain``).
"""
from __future__ import annotations

from .contract import require


def _cdiv(a, b):
    return -(-a // b)


def _counts(*values):
    require(all(type(v) is int and v >= 0 for v in values), 'non-negative token counts required')


def skipped_tokens(computed, window):
    """vLLM ``SlidingWindowManager.get_num_skipped_tokens``: tokens no later query reads."""
    _counts(computed, window)
    require(window > 0, 'positive window required')
    return max(0, computed - window + 1)


def window_span(position, window):
    """Half-open token span the sliding layers attend for the query at ``position``."""
    _counts(position, window)
    return max(0, position - window + 1), position + 1


def kept_blocks(computed, window, block):
    """Block indices [first, end) a sliding group keeps after ``computed`` tokens: the
    blocks of the tokens the next query reads from the cache, [computed - W + 1, computed)."""
    _counts(computed, window, block)
    require(block > 0, 'positive block size required')
    return skipped_tokens(computed, window) // block, _cdiv(computed, block)


def max_kept_blocks(window, block):
    """The most blocks ``kept_blocks`` can name: ceil((W - 1) / B) + 1."""
    return _cdiv(window - 1, block) + 1


def phase1_row(C, e, window, block):
    """Phase 1 over [C, e) of one sliding group: (null head, borrowed, fresh) where
    ``borrowed`` are the window blocks the session pinned after computing C (the last one
    may be partial; phase 1 appends into it) and ``fresh`` the new blocks for [C, e)."""
    _counts(C, e)
    require(C < e, 'phase 1 needs a non-empty run')
    head, end = kept_blocks(C, window, block)
    return head, (head, end), (end, _cdiv(e, block))


def phase1_drain(C, e, window, block):
    """Blocks of the phase-1 row after its drain: (adopted by phase 2, unpinned old
    window blocks, freed fresh blocks), each a list of block indices."""
    head, (b0, b1), (f0, f1) = phase1_row(C, e, window, block)
    k0, k1 = kept_blocks(e, window, block)
    require(k1 == f1 or (f0 == f1 and k1 == b1), 'phase 1 keeps the window it ends in')
    adopted = list(range(k0, k1))
    unpinned = [i for i in range(b0, b1) if i < k0]
    freed = [i for i in range(f0, f1) if i < k0]
    return adopted, unpinned, freed


def phase1_chunks(C, e, forward, chunk):
    """Phase 1's native forwards over [C, e) (gemma-goal, design WP10): one forward when the
    run fits one native forward of ``forward`` tokens, else forwards of ``chunk`` tokens
    (the last one shorter); a list of (start, end)."""
    _counts(C, e, forward, chunk)
    require(C < e and 0 < chunk <= forward, 'phase-1 chunks need a non-empty run and 0 < chunk <= forward')
    if e - C <= forward:
        return [(C, e)]
    return [(s, min(e, s + chunk)) for s in range(C, e, chunk)]


def phase1_ring(C, chunks, window, block):
    """Fresh sliding blocks per group that phase 1's forwards hold at once: in the forward
    over [s, t) the blocks of [s - W + 1, t) are live; fresh blocks (index >= ceil(C / B))
    of tokens before that window are free for the next forward. One forward: every fresh
    block, ceil(e / B) - ceil(C / B)."""
    first_fresh = _cdiv(C, block)
    return max(_cdiv(t, block) - max(first_fresh, kept_blocks(s, window, block)[0]) for s, t in chunks)


def phase2_head(e, window, block):
    """(null head length, adopted window blocks [first, end)) installed at admission:
    without it vLLM's first allocation would cover every token of [0, e)."""
    first, end = kept_blocks(e, window, block)
    return first, (first, end)


def window_evidence(*, L, C, e, window, page_tokens, selected, mandatory=(), recent=()):
    """Runtime-row evidence of one turn: the window the sliding layers read at the first
    phase-2 step and its overlap with the candidate Pages [0, L / P).

    ``history_tokens``: sealed history tokens inside the window; ``history_pages``: the
    candidate Pages they touch; ``unselected_pages``: those the global layers do not
    see this turn (the sliding layers still do, exactly); ``recent_pages`` /
    ``mandatory_pages``: the overlap with those quotas."""
    _counts(L, C, e, page_tokens)
    require(L <= C < e and L % page_tokens == 0, 'stage_b geometry L <= C < e with L on a Page boundary')
    start, stop = window_span(e, window)
    history = max(0, L - start)
    first = start // page_tokens
    pages = list(range(first, L // page_tokens)) if history else []
    selected, mandatory, recent = set(selected), set(mandatory), set(recent)
    return dict(window_tokens=window, span=[start, stop], run_tokens=e - C,
                inside_run=start >= C, history_tokens=history, history_pages=pages,
                unselected_pages=[p for p in pages if p not in selected],
                recent_pages=[p for p in pages if p in recent],
                mandatory_pages=[p for p in pages if p in mandatory])
