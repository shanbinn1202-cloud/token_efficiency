"""Capacity of the window layout (gemma-s1 WP4): the sliding pool's needs, and the launcher's sizing.

Pure integer arithmetic over ``kv_layout.WindowLayout``; nothing here touches vLLM.

GPU, per pool (design 3.1, 3.4). Pool 0 (global Pages) follows the Qwen rules of
``capacity`` with one full-attention group. Pool 1 (sliding windows) holds, per
sliding group, at most one window slot per open session (W-GPU: the session pins the
blocks its next query reads) plus what one forward adds:

* a cold request (turn 0, a rebuild, a session opening) prefills [0, n) in native
  forwards of at most N tokens; vLLM frees the blocks that leave the window before
  each allocation, so it holds at most min(ceil(n / P), slot + transient) blocks;
* a continuing turn's phase 1 borrows the pinned window and appends
  ceil(e / P) - ceil(C / P) fresh blocks; a goal_query Goal forward is a separate cold
  forward of at most G tokens (``WindowLayout.goal_tokens``, gemma-goal) that runs while
  those fresh blocks are held: pool 1 reserves its ceil(G / P) blocks per group as a
  second transient on a goal_query server (``kv_layout.window_pool_blocks``);
* phase 2 adopts the window phase 1 ends in (``sliding_window.phase1_drain``); its
  first step computes the render suffix [e, N) (Gemma's generation header, 7 tokens),
  which can hold one block more than a slot (``phase2_peak``), and then it never holds
  more than a slot: vLLM frees the oldest block before it allocates the next. A
  running request may still grow to its peak (its headroom).

The engine's admission gate (``capacity``) takes the FA figures of the Qwen rules and
the sliding ones from ``gate_turn_need`` and ``gate_headroom``. As for Qwen, a turn while
nothing runs (or is in flight) is checked when another open session pins global Pages, a
tail or a window (``pins_window``; integrate-1018) and waits when it does not fit after
the releases --release-idle-pages makes. With --admission-overlap
(gemma-goal) its ledger reserves, per admission in flight, the sliding blocks the turn
keeps once accepted, and the phase-1 transient once for all of them (``ledger_ahead``:
phase 1 runs one lease at a time); the FA pool keeps the Qwen reservations.

The launcher sizes pool 0 by one of three rules (serve.py):

* strict (the default): one session's worst turn at the context limit
  (``one_session_pages``), as for Qwen;
* selective (--release-idle-pages with a query policy and a fixed
  --active-token-budget, the 16-session SWE-bench line): one session's worst
  *selective* turn (``selective_turn_pages``: its resident Pages, one selection of
  restores, one forward and one reply) plus the other sessions' tails and the
  watermark (``selective_pages``). Idle sessions' Pages are released under pressure
  and readmitted at their next turn (``idle_release``); the windows stay on the GPU.
  A first turn or a rebuild needs its whole row; one that cannot fit even with every
  other session released is refused (CAPACITY_EXCEEDED) instead of waiting forever;
* release (--release-idle-pages without a fixed budget: ``full``, the 16-session line of
  the four policies, ``random50`` or --active-ratio; gemma-goal): one session's worst turn
  at the context limit plus the other sessions' tails and the watermark
  (``release_pages``). Idle sessions release all their Pages under pressure and readmit
  them at their next turn: for ``full`` that is the session's whole history.

Host (the job's memory, D-5 memory accounting): the pinned budget of a Gemma server is
the job's memory less a process overhead, ``HOST_MEMORY_BYTES - PROCESS_OVERHEAD_BYTES``
= 113,560 MiB - 30 GiB (one GH200's share of a ghx4 node, which bills as one GPU):
the connector pool plus the pinned arena of global Pages; with serve.py
--host-memory-shares 2 (gemma-mab) a job of two shares, 2 x 113,560 MiB - 30 GiB
(``pinned_budget_bytes``). There is no Gated-DeltaNet carry, and with the window on
the GPU no window carry either. The OffloadingConnector's
CPU pool is never used by this runtime (the HTTP runtime isolates the connector), so
the Gemma profile gives it 1 GiB (``CONNECTOR_BYTES``), what ``--connector-pool-gib 1``
of the host-pool line sets for Qwen; a 32 GiB pool would leave 48 GiB of arena.
"""
from __future__ import annotations

from .contract import require

GIB = 1024 ** 3
MIB = 1024 ** 2
HOST_MEMORY_BYTES = 113560 * MIB          # --mem=113560M, one GH200's share (110.9 GiB)
PROCESS_OVERHEAD_BYTES = 30 * GIB         # weights through the page cache, Python heaps, the rewriter
PINNED_BUDGET_BYTES = HOST_MEMORY_BYTES - PROCESS_OVERHEAD_BYTES   # 80.9 GiB
CONNECTOR_GIB = 1                         # the OffloadingConnector's CPU pool when --connector-pool-gib is left out
CONNECTOR_BYTES = CONNECTOR_GIB * GIB
ARENA_CHUNK_BYTES = 256 << 20
WATERMARK = .01


def _cdiv(a, b):
    return -(-a // b)


def _window(layout):
    from .kv_layout import is_window
    require(is_window(layout), 'window layout required')
    return layout


def slot_and_transient(layout):
    _window(layout)
    return layout.window_blocks, layout.transient_blocks


def cold_peak(layout, tokens):
    """Blocks per sliding group a cold request of ``tokens`` prompt tokens holds at most."""
    slot, transient = slot_and_transient(layout)
    require(type(tokens) is int and tokens >= 0, 'token count required')
    return min(_cdiv(tokens, layout.page_tokens), slot + transient)


def phase2_peak(layout, suffix):
    """Blocks per sliding group phase 2 holds at most: its first step computes the
    ``suffix`` tokens [e, N) on the adopted window (vLLM's bound for a forward of that
    size, never less than a slot), and decoding keeps one slot."""
    _window(layout)
    require(type(suffix) is int and 0 <= suffix <= layout.forward_tokens, 'phase-2 suffix out of range')
    return max(layout.window_blocks, _cdiv(layout.window_tokens - 1 + suffix, layout.page_tokens) + 1)


def phase1_chunk_tokens(layout):
    """The forwards of a chunked phase 1: model_profile.GEMMA4_PHASE1_CHUNK_TOKENS, at most N."""
    from .model_profile import GEMMA4_PHASE1_CHUNK_TOKENS
    return min(GEMMA4_PHASE1_CHUNK_TOKENS, _window(layout).forward_tokens)


def sliding_turn_need(layout, groups, *, cold, prompt_tokens=0, C=None, e=None, goal_tokens=0,
                      adopted=None, suffix=0, chunk=None):
    """(phase-1, phase-2) sliding blocks of one turn beyond the session's window pin.

    ``cold``: turn 0, a rebuild or an opening (phase 1 none; phase 2 the cold peak).
    Otherwise phase 1 runs [C, e) on the pinned window (fresh blocks, plus a Goal
    forward of ``goal_tokens``), and phase 2 needs what lifts the adopted window
    (``adopted`` blocks per group, ``sliding_window.kept_blocks(e)``) to its peak
    (``phase2_peak`` of the ``suffix`` [e, N); a full slot for 0 or 1 token).

    A run longer than one native forward runs in forwards of ``chunk`` tokens
    (``phase1_chunk_tokens`` when omitted; gemma-goal, design WP10) that reuse the fresh
    blocks which left the window: phase 1 holds ``sliding_window.phase1_ring`` of them."""
    _window(layout)
    require(type(groups) is int and groups > 0, 'sliding group count required')
    P = layout.page_tokens
    if cold:
        return 0, groups * cold_peak(layout, prompt_tokens)
    require(type(C) is int and type(e) is int and 0 < C < e, 'phase 1 [C, e) required')
    require(type(goal_tokens) is int and 0 <= goal_tokens <= layout.goal_tokens,
            'Goal forward exceeds the Goal transient')
    from .sliding_window import phase1_chunks, phase1_ring
    chunk = phase1_chunk_tokens(layout) if chunk is None else chunk
    fresh = phase1_ring(C, phase1_chunks(C, e, layout.forward_tokens, chunk), layout.window_tokens, P)
    if adopted is None:
        from .sliding_window import kept_blocks
        first, end = kept_blocks(e, layout.window_tokens, P)
        adopted = end - first
    require(type(adopted) is int and 0 <= adopted <= layout.window_blocks, 'adopted window out of range')
    return (groups * (fresh + cold_peak(layout, goal_tokens)),
            groups * max(0, phase2_peak(layout, suffix) - adopted))


def sliding_headroom(layout, groups, held, peaks=None):
    """Sliding blocks the running requests may still allocate: each grows to its peak
    (``peaks``, per request; a full slot when omitted). ``held``: live blocks per group
    of each running request."""
    _window(layout)
    bound = layout.window_blocks + layout.transient_blocks
    require(all(type(h) is int and 0 <= h <= bound for h in held), 'held blocks out of range')
    peaks = [layout.window_blocks] * len(held) if peaks is None else list(peaks)
    require(len(peaks) == len(held) and all(type(p) is int and 0 <= p <= bound for p in peaks),
            'peak blocks out of range')
    return groups * sum(max(0, p - h) for p, h in zip(peaks, held))


# ---------------------------------------------------------------- the engine's admission gate

def gate_turn_need(core, session_id, tokens, messages, generation):
    """(phase-1, phase-2) sliding blocks of one admission, for ``capacity.turn_need_by_pool``.

    A session's first turn, an opening and a rebuild (its render frees the window) are
    cold: the prefill's peak, or the decode's slot. A continuing turn's phase 1 runs
    [C, e) with e = N - |generation header|; a goal_query Goal forward is bounded by the
    layout's Goal transient G (phase 1 refuses a longer one). A run longer than one
    forward runs in chunks (gemma-goal): its fresh blocks are the chunks' ring."""
    layout, G, turn = _gate_turn(core, session_id, tokens, messages, generation)
    if turn is None:
        return 0, 0
    if turn[0] == 'cold':
        return 0, turn[1]
    _, C, e, goal, n, chunk = turn
    return sliding_turn_need(layout, G, cold=False, C=C, e=e, goal_tokens=goal, suffix=n - e, chunk=chunk)


def held_window(adapter, session):
    """Sliding blocks the session's pinned window holds now, over every sliding group (0 without one)."""
    window = getattr(adapter, 'window', None)
    pinned = None if window is None else window.window(session.namespace)
    return sum(len(blocks) for blocks in (pinned or {}).values())


def pins_window(adapter, session):
    """Whether the session pins a window in pool 1 now (``held_window``): for the gate's rule
    that a turn admitted while nothing runs is checked when other sessions pin blocks
    (``capacity.pinned_elsewhere``, integrate-1018)."""
    return getattr(session, 'namespace', None) is not None and held_window(adapter, session) > 0


def _gate_turn(core, session_id, tokens, messages, generation):
    """(layout, sliding groups, the turn): ('cold', its sliding peak), ('turn', C, e, Goal
    tokens, prompt tokens, phase-1 chunk tokens) of a continuing turn, or None without a run."""
    adapter = core._remote_ra_growing_adapter
    endpoint = core._remote_ra_private_v2_endpoint
    layout = _window(adapter.layout)
    _, win, _ = endpoint.phase1.groups()
    G, P, n = len(win), layout.page_tokens, len(tokens)
    s = adapter.sessions.get(session_id)
    transcript = endpoint.transcripts.get(session_id)
    if (s is None or s.local_turn == 0 or transcript is None or
            (transcript.turn > 0 and not transcript.extends(messages))):
        cold = G * max(cold_peak(layout, n), min(_cdiv(n + generation, P), layout.window_blocks))
        if s is not None and s.local_turn > 0:
            # A rebuild's render closes the session's namespace, which frees its pinned
            # window before the cold prefill takes its blocks: credit them, as the global
            # figures credit its R and tail (capacity._need_by_pool). Without the credit a
            # rebuild with every window slot taken looked short of pool 1 by its own window,
            # so IdleRelease released no idle Pages for its pool-0 shortfall and, admitted
            # once nothing ran, the rebuild waited for global Pages forever (gemma-tb).
            cold = max(0, cold - held_window(adapter, s))
        return layout, G, ('cold', cold)
    # A kv-session prompt names its own phase-2 suffix (gemma-mab: Gemma's header, or one
    # token after tool responses); the XML grammar's is always its header.
    suffix = getattr(transcript, 'phase2_suffix', None)
    C, e = s.computed_end, n - (suffix(tokens) if suffix is not None else len(transcript.header))
    if not 0 < C < e:
        return layout, G, None
    goal = 0
    if s.policy == 'goal_query':
        contract = getattr(endpoint.selection, 'contract', None)
        if contract is not None:
            from .selection_contract import selection_active
            Q = s.sealed_end
            if any(selection_active(k, s.local_turn, contract) for k in {Q // P, Q // P + (s.tail is not None)}):
                goal = layout.goal_tokens
    chunk = getattr(endpoint.phase1, 'chunk_tokens', None) or phase1_chunk_tokens(layout)
    return layout, G, ('turn', C, e, goal, n, chunk)


# ---- --admission-overlap: the sliding pool's part of the reservation ledger (gemma-goal) ----

def ledger_parts(layout, groups, *, C, e, goal_tokens=0, suffix=0, chunk=None):
    """(base, peak) sliding blocks of one continuing turn in the ledger (capacity.ledger_gate).

    ``base``: what the turn holds beyond the session's pins once it is accepted, at most
    max(adopted window, phase-2 peak) less the pinned window it borrows (``kept_blocks(C)``),
    per sliding group. ``peak``: max(its phase-1 need, base); the part above ``base`` is the
    phase-1 transient (fresh blocks before the window it ends in, the Goal forward), freed
    or adopted before the admission is accepted."""
    _window(layout)
    from .sliding_window import kept_blocks
    W, P = layout.window_tokens, layout.page_tokens
    phase1, _ = sliding_turn_need(layout, groups, cold=False, C=C, e=e, goal_tokens=goal_tokens, suffix=suffix,
                                  chunk=chunk)
    first, end = kept_blocks(e, W, P)
    old0, old1 = kept_blocks(C, W, P)
    base = groups * max(0, max(end - first, phase2_peak(layout, suffix)) - (old1 - old0))
    return base, max(phase1, base)


def gate_ledger(core, session_id, tokens, messages, generation):
    """(base, peak) of one admission for the ledger (``ledger_parts``); a cold turn's base
    and peak are its sliding peak, and a phase 1 longer than one forward asks nothing."""
    layout, G, turn = _gate_turn(core, session_id, tokens, messages, generation)
    if turn is None:
        return 0, 0
    if turn[0] == 'cold':
        return turn[1], turn[1]
    _, C, e, goal, n, chunk = turn
    return ledger_parts(layout, G, C=C, e=e, goal_tokens=goal, suffix=n - e, chunk=chunk)


def ledger_ahead(items):
    """(sum of bases, largest transient) of the in-flight admissions' sliding reservations,
    ``items`` = ((base, peak), sliding blocks its phase-1 lease holds now) each.

    Phase 1 runs one lease at a time, within one engine-thread call from its lease to the
    admission's acceptance: an admission parks at its Goal before phase 1 and nowhere
    after it, and --phase1-in-step (which parks a leased phase 1) is refused for this
    layout. So the in-flight admissions' phase-1 transients are never allocated together:
    the ledger keeps every base and only the largest transient. Pool 1 holds one forward's
    transient and the Goal transient beyond the window slots for exactly that."""
    total = largest = 0
    for (base, peak), held in items:
        # What a lease holds comes off its transient first: the rest it may still take
        # counts as base (summed) up to its base, the remainder as transient.
        rest = max(0, peak - held)
        kept = min(rest, base)
        total += kept
        largest = max(largest, rest - kept)
    return total, largest


def ledger_phase1(reservation, largest):
    """What an admission's sliding reservation (base, peak) adds to the in-flight ones whose
    largest transient is ``largest``: its base, and its transient beyond ``largest``."""
    base, peak = reservation
    return base + max(0, peak - base - largest)


def gate_headroom(lifecycle):
    """Sliding blocks the running requests may still allocate, for ``capacity.headroom_by_pool``.

    Each accepted request grows to its peak: a cold one still prefilling to the cold
    peak, a continuing one before its first phase-2 step to ``phase2_peak``, and every
    one decoding to min(its row, a slot)."""
    core = lifecycle.endpoint.core
    endpoint, adapter = core._remote_ra_private_v2_endpoint, core._remote_ra_growing_adapter
    layout = _window(adapter.layout)
    _, win, _ = endpoint.phase1.groups()
    P, slot = layout.page_tokens, layout.window_blocks
    manager = adapter.kv_manager.coordinator.single_type_managers[win[0]]
    from .capacity import max_tokens
    held, peaks = [], []
    for rid, entry in lifecycle.entries.items():
        request = entry['request']
        if entry['state'] != 'accepted' or request is None:
            continue
        geometry = endpoint.claims.get(rid, {}).get('geometry')
        n, done = len(request.prompt_token_ids), request.num_computed_tokens
        peak = min(_cdiv(n + max_tokens(request.sampling_params), P), slot)
        if geometry is None or geometry.local_turn == 0:
            if done < n:
                peak = max(peak, cold_peak(layout, n))
        elif done <= geometry.phase2_start:
            peak = max(peak, phase2_peak(layout, n - geometry.e))
        held.append(sum(1 for block in manager.req_to_blocks.get(rid, ()) if not block.is_null))
        peaks.append(peak)
    return sliding_headroom(layout, len(win), held, peaks)


# ---------------------------------------------------------------- launcher sizing

def pool_one_bytes(layout, groups, block_bytes):
    """Bytes of the statically reserved pool 1 (``kv_layout.window_pool_blocks``)."""
    from .kv_layout import window_pool_blocks
    return window_pool_blocks(_window(layout), groups) * block_bytes


def pool_zero_blocks(kv_bytes, layout, groups, block_bytes, page_bytes):
    """Global Pages vLLM gives pool 0 out of ``kv_bytes`` after pool 1."""
    return max(kv_bytes - pool_one_bytes(layout, groups, block_bytes), 0) // page_bytes


def one_session_pages(layout, max_model_len, max_tokens, blocks):
    """One session's worst turn at the context limit in pool 0 (the Qwen rule with
    N-token forwards): every Page plus the tail, one forward's Pages, one reply's
    Pages plus one, the 1 % watermark. A goal_query phase 1 holds a Goal forward's
    ceil(G / P) Pages instead of the reply's: the larger of the two counts."""
    P = layout.page_tokens
    return (_cdiv(max_model_len, P) + 1 + _cdiv(layout.forward_tokens, P)
            + max(_cdiv(max_tokens, P) + 1, layout.goal_blocks) + int(WATERMARK * blocks))


def selective_turn_pages(layout, active_tokens, max_tokens):
    """One session's worst continuing turn in pool 0 under a fixed active budget: its
    resident Pages before the turn (R, the last turn's selection plus the Pages sealed
    in it), this turn's restores (at most one selection), the row past its tail (one
    native forward, the generation header and one reply: ``capacity.turn_need_by_pool``
    end - base) and the tail. ``active_tokens``: the largest selection in tokens,
    max(--active-token-budget, the switch threshold) (up to the threshold every Page
    is active). Phase 1 holds R, one forward's fresh Pages, a goal_query Goal forward's
    Pages (ceil(G / P), freed at its drain) and the tail; the phase-2 peak is the larger
    for every Gemma setting, but the rule takes the larger of the two."""
    P = _window(layout).page_tokens
    require(type(active_tokens) is int and active_tokens >= P, 'active budget of at least one Page required')
    run = _cdiv(layout.forward_tokens + max_tokens, P) + 1
    resident = active_tokens // P + run
    phase1 = resident + _cdiv(layout.forward_tokens, P) + layout.goal_blocks + 1
    return max(2 * resident + 1, phase1)


def selective_pages(layout, active_tokens, max_tokens, blocks):
    """Pool 0 under the selective rule (--release-idle-pages, a query policy with
    --active-token-budget): one session's worst selective turn while every idle session
    has released its resident Pages, the other sessions' tails (never released) and
    the 1 % watermark. A first turn or a rebuild whose whole row does not fit then is
    refused (``idle_release``: CAPACITY_EXCEEDED)."""
    return (selective_turn_pages(layout, active_tokens, max_tokens) + layout.window_slots - 1
            + int(WATERMARK * blocks))


def pinned_request_pages(layout, active_tokens, max_tokens, kv_bytes):
    """The selective rule's figure the launcher pins as ``request_pages`` (gemma-mab):
    ``selective_pages`` of a Gemma 4 server's pool 0 out of ``kv_bytes``, the layout
    taken without the field. Startup only (serve.py and remote_ra_http.config)."""
    import dataclasses
    from .model_profile import GEMMA4_GEOMETRY as g
    layout = dataclasses.replace(_window(layout), request_pages=0)
    blocks = pool_zero_blocks(kv_bytes, layout, len(g.sliding_groups()), g.sliding_block_bytes(layout.page_tokens),
                              g.page_bytes(layout.page_tokens))
    return selective_pages(layout, active_tokens, max_tokens, blocks)


def release_pages(layout, max_model_len, max_tokens, blocks):
    """Pool 0 under --release-idle-pages without a fixed active budget (``full``, ``random50``
    or a query policy with --active-ratio; gemma-goal): one session's worst turn at the
    context limit (``one_session_pages``: its whole row, as ``full`` keeps every Page) while
    every idle session has released its resident Pages, the other sessions' tails (never
    released) and the watermark. Each released session readmits its Pages before its next
    phase 1 (``idle_release``)."""
    return one_session_pages(layout, max_model_len, max_tokens, blocks) + _window(layout).window_slots - 1


def max_sessions(kv_bytes, *, page_tokens, window_tokens, forward_tokens, groups, block_bytes, page_bytes,
                 max_model_len, max_tokens, limit=64, active_tokens=None, goal_tokens=0, release=False):
    """The most window slots whose pools still hold one session's worst turn (0 if none):
    at the context limit, or (``active_tokens``) the selective rule's (``selective_pages``),
    or (``release``, no active budget) the context limit with the other sessions' tails
    (``release_pages``). ``goal_tokens``: a goal_query server's Goal transient (pool 1)."""
    from .kv_layout import WindowLayout
    best = 0
    for sessions in range(1, limit + 1):
        layout = WindowLayout(page_tokens, window_tokens, sessions, forward_tokens, goal_tokens)
        blocks = pool_zero_blocks(kv_bytes, layout, groups, block_bytes, page_bytes)
        needed = (selective_pages(layout, active_tokens, max_tokens, blocks) if active_tokens is not None else
                  release_pages(layout, max_model_len, max_tokens, blocks) if release else
                  one_session_pages(layout, max_model_len, max_tokens, blocks))
        if blocks - 1 >= needed:
            best = sessions
    return best


def arena_bytes(pages, page_bytes):
    """The pinned arena's bytes, in whole 256 MiB chunks."""
    chunk = max(1, ARENA_CHUNK_BYTES // page_bytes)
    return _cdiv(pages, chunk) * chunk * page_bytes


def pinned_host_bytes(sessions, page_tokens, page_bytes, max_model_len, arena_pages=None,
                      connector_bytes=CONNECTOR_BYTES):
    """Worst-case pinned host bytes of a Gemma server: the connector pool (--connector-pool-gib,
    1 GiB when left out) and the arena (every running session at the context limit, or the
    --dram-arena-gib cap)."""
    pages = sessions * (_cdiv(max_model_len, page_tokens) + 1) if arena_pages is None else arena_pages
    return connector_bytes + arena_bytes(pages, page_bytes)


def pinned_budget_bytes(shares=1):
    """The pinned budget of a job of ``shares`` GH200 memory shares (serve.py
    --host-memory-shares, gemma-mab): the job's memory less the process overhead."""
    require(shares in (1, 2), 'one or two GH200 memory shares')
    return shares * HOST_MEMORY_BYTES - PROCESS_OVERHEAD_BYTES


def host_fits(pinned, shares=1):
    """Pinned bytes plus the process overhead within the job's memory (``shares`` GH200
    memory shares of 113,560 MiB: one by default, two with --host-memory-shares 2)."""
    return pinned <= pinned_budget_bytes(shares)


def largest_arena_gib(page_tokens, page_bytes, max_model_len, connector_bytes=CONNECTOR_BYTES, shares=1):
    """The largest --dram-arena-gib whose pinned bytes fit the job's memory (0 if none
    holds one session at the context limit)."""
    one = _cdiv(max_model_len, page_tokens) + 1
    for gib in range(shares * HOST_MEMORY_BYTES // GIB, 0, -1):
        pages = gib * GIB // page_bytes
        if pages >= one and host_fits(pinned_host_bytes(0, page_tokens, page_bytes, max_model_len, pages,
                                                        connector_bytes), shares):
            return gib
    return 0
