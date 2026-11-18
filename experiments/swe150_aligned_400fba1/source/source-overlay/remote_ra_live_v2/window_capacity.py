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
phase 1 runs one lease at a time); the FA pool keeps the Qwen reservations. With
--goal-overlap an admission may wait for its Goal after phase 1; what it holds then
comes off its base (``ledger_ahead_split``).

--thinking-history strip (option G, window_hold): each strip request holds its session's
window at the cut X through phase 2 (WindowPins.hold), in one of the layout's H hold slots
(``WindowLayout.hold_slots``, one window slot each in pool 1). The gate admits at most H
requests that hold or will hold one (``hold_wait``: CAPACITY_WAIT, never preemption), and counts
per strip turn and sliding group the blocks its hold may keep beyond its row (``hold_term``:
the window at X, at most ceil((suffix + max_tokens) / P) + 1 of which leave the row while phase
2 runs) in its phase-2 need, its ledger base and, less what it already keeps
(``WindowPins.held_left``), in the running requests' headroom. The launcher's selective rule
adds the reply such a turn re-prefills (``strip_reprefill_pages``). H = 0 adds nothing anywhere.

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
import policy_registry

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


def phase1_chunk_tokens(layout, forward=None):
    """The forwards of a chunked phase 1: model_profile.GEMMA4_PHASE1_CHUNK_TOKENS, at most one
    native forward (``forward``, the layout's N when omitted) and at most N - W - P, the largest
    chunk whose ring (``sliding_window.phase1_ring``: at most ceil((W - 1 + chunk) / P) + 1 fresh
    blocks per sliding group) fits one forward's transient (N / P blocks): 4,096 at N = 8,192
    (unchanged), 3,056 at N = 4,096 and P = 16, 3,040 at P = 32 (gemma-forward4k). A layout too
    small for such a chunk (N - W - P < P, CPU fakes only) keeps min(4,096, N)."""
    from .model_profile import GEMMA4_PHASE1_CHUNK_TOKENS
    layout = _window(layout)
    chunk = min(GEMMA4_PHASE1_CHUNK_TOKENS, layout.forward_tokens if forward is None else forward)
    fits = layout.forward_tokens - layout.window_tokens - layout.page_tokens
    return min(chunk, fits) if fits >= layout.page_tokens else chunk


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
    hold = _turn_hold(core, layout, G, session_id, tokens, generation)
    if turn[0] == 'cold':
        return 0, turn[1] + hold
    _, C, e, goal, n, chunk = turn
    phase1, phase2 = sliding_turn_need(layout, G, cold=False, C=C, e=e, goal_tokens=goal, suffix=n - e, chunk=chunk)
    return phase1, phase2 + hold


# ---- --thinking-history strip: the window held at the cut (option G, window_hold) ----

def hold_slots(adapter):
    """The hold slots H of a server that holds the window at the strip cut (WindowPins.hold_slots,
    set by window_hold.attach_engine), else 0."""
    window = getattr(adapter, 'window', None)
    return getattr(window, 'hold_slots', 0) if window is not None else 0


def hold_term(layout, end, suffix, max_tokens):
    """Blocks per sliding group a strip request's hold may keep beyond its row: the window at the
    cut ``end`` = X (``sliding_window.kept_blocks(X)``, at most a slot), of which at most
    ceil((suffix + max_tokens) / P) + 1 leave the row while phase 2 computes the ``suffix``
    (the generation header [X, N)) and up to ``max_tokens`` replies."""
    _window(layout)
    require(type(end) is int and end > 0 and type(suffix) is int and suffix >= 0 and
            type(max_tokens) is int and max_tokens >= 0, 'cut, suffix and reply bound required')
    from .sliding_window import kept_blocks
    first, last = kept_blocks(end, layout.window_tokens, layout.page_tokens)
    return min(last - first, _cdiv(suffix + max_tokens, layout.page_tokens) + 1)


def _suffix(transcript, tokens):
    """The generation header after the cut, [X, N): a kv-session's own suffix, else the header;
    None without a transcript (an opening session-pool session)."""
    if transcript is None:
        return None
    suffix = getattr(transcript, 'phase2_suffix', None)
    return suffix(tokens) if suffix is not None else len(transcript.header)


def _turn_hold(core, layout, groups, session_id, tokens, generation):
    """A strip turn's hold term over every sliding group (0 without hold slots); a whole slot per
    group when the header is not known yet."""
    adapter = core._remote_ra_growing_adapter
    if not hold_slots(adapter):
        return 0
    suffix = _suffix(core._remote_ra_private_v2_endpoint.transcripts.get(session_id), tokens)
    if suffix is None or not 0 < len(tokens) - suffix:
        return groups * layout.window_blocks
    return groups * hold_term(layout, len(tokens) - suffix, suffix, generation)


def hold_wait(lifecycle):
    """Raise CapacityWait when every hold slot is taken: the requests whose phase 2 holds (or will
    hold) a window at the strip cut, i.e. the admissions in flight or parked, the running requests
    and the engine's holds (a failed cleanup's too, while it holds), number H already. The turn then
    waits and retries; nothing changed and nothing is preempted. No-op without hold slots."""
    core = lifecycle.endpoint.core
    adapter = core._remote_ra_growing_adapter
    H = hold_slots(adapter)
    if not H:
        return
    holders = {rid for rid, entry in lifecycle.entries.items()
               if entry['state'] in ('rendered', 'reserving', 'accepted')}
    holders.update(adapter.window.holds)
    if len(holders) >= H:
        from .capacity import CAPACITY_WAIT, CapacityWait
        raise CapacityWait('%s: window hold slots: %d of %d in use (a strip turn takes one from its admission '
                           'to its commit; at most %d in flight)' % (CAPACITY_WAIT, len(holders), H, H))


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
    if policy_registry.query_vector(s.policy) == 'goal':
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
    # --thinking-history strip: the hold term is kept once accepted (base and peak).
    hold = _turn_hold(core, layout, G, session_id, tokens, generation)
    if turn[0] == 'cold':
        return turn[1] + hold, turn[1] + hold
    _, C, e, goal, n, chunk = turn
    base, peak = ledger_parts(layout, G, C=C, e=e, goal_tokens=goal, suffix=n - e, chunk=chunk)
    return base + hold, peak + hold


def ledger_ahead(items):
    """(sum of bases, largest transient) of the in-flight admissions' sliding reservations,
    ``items`` = ((base, peak), sliding blocks its phase-1 lease holds now) each.

    Phase 1 runs one lease at a time, within one engine-thread call from its lease to the
    admission's acceptance: an admission parks at its Goal before phase 1 and nowhere
    after it, and --phase1-in-step (which parks a leased phase 1) is refused for this
    layout. So the in-flight admissions' phase-1 transients are never allocated together:
    the ledger keeps every base and only the largest transient. Pool 1 holds one forward's
    transient and the Goal transient beyond the window slots for exactly that.
    --goal-overlap (an admission may also wait after phase 1): ``ledger_ahead_split``."""
    total = largest = 0
    for (base, peak), held in items:
        # What a lease holds comes off its transient first: the rest it may still take
        # counts as base (summed) up to its base, the remainder as transient.
        rest = max(0, peak - held)
        kept = min(rest, base)
        total += kept
        largest = max(largest, rest - kept)
    return total, largest


def ledger_ahead_split(items):
    """``ledger_ahead`` with --goal-overlap (goal_overlap; docs/goal-overlap/DESIGN.md section 9).

    An admission may now wait for its Goal after phase 1's drain and before its adoption.
    Every lease still runs inside one engine-thread call (phase 1, then the Goal forward,
    each drained there; no --phase1-in-step on this layout), so at a gate no lease is
    undrained and the transients are still never allocated together. What an admission
    holds at a gate is what a drained phase 1 kept: its fresh kept blocks less the session's
    pinned blocks its drain freed (``WindowPhase1Coordinator.held``), which its turn keeps
    once accepted and which never exceeds its base. So it comes off the base, not the
    transient: the base left is what the turn may still take for good (possibly more than
    its base when the drain freed more than it kept), and the transient (the Goal forward
    still ahead) stays reserved, the largest once. ``ledger_ahead`` takes held blocks off the
    transient first: while they are not negative it reserves at least as much as this, but a
    negative figure (the drain freed more than it kept) it reserves too little by, next to a
    larger transient. Without the drain's trim a waiting admission would hold the session's
    old window too, more than its base: that is what ran pool 1 out of blocks."""
    total = largest = 0
    for (base, peak), held in items:
        rest = max(0, peak - held)
        kept = min(rest, max(0, base - held))
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
    room = sliding_headroom(layout, len(win), held, peaks)
    if hold_slots(adapter):
        # --thinking-history strip: each running request's hold may still keep blocks beyond its
        # row: its hold term less those that already left the row (WindowPins.held_left).
        for rid, entry in lifecycle.entries.items():
            request = entry['request']
            geometry = endpoint.claims.get(rid, {}).get('geometry')
            if entry['state'] != 'accepted' or request is None:
                continue
            n = len(request.prompt_token_ids)
            term = (slot if geometry is None else
                    hold_term(layout, geometry.e, n - geometry.e, max_tokens(request.sampling_params)))
            room += len(win) * max(0, term - adapter.window.held_left(rid))
    return room


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
    for every Gemma setting, but the rule takes the larger of the two.

    --thinking-history strip (a layout with hold slots): ``strip_reprefill_pages`` more, in
    both peaks."""
    P = _window(layout).page_tokens
    require(type(active_tokens) is int and active_tokens >= P, 'active budget of at least one Page required')
    run = _cdiv(layout.forward_tokens + max_tokens, P) + 1
    resident = active_tokens // P + run
    phase1 = resident + _cdiv(layout.forward_tokens, P) + layout.goal_blocks + 1
    return max(2 * resident + 1, phase1) + strip_reprefill_pages(layout, max_tokens)


def strip_reprefill_pages(layout, max_tokens):
    """Pages the selective rule adds for the reply a Gemma strip turn re-prefills (0 without hold
    slots, i.e. on every server without --thinking-history strip: their figures are unchanged).

    A strip turn's phase 1 computes the previous reply's content again (``<|turn>model\\n`` + C +
    ``<turn|>\\n``) before the observation, in chunks when the two exceed one forward, so its run
    is an observation of one forward plus up to one reply bound of content. The rule's two runs
    (the one sealed in the last turn, resident; this turn's row) count one forward and one reply
    each. Under strip the last reply is not resident (the commit at the cut drops it), which
    leaves room for one re-prefilled content; the turn holds two (c' in the run sealed last turn,
    c in this turn's row) and this turn's reply: forward + c', forward + c + reply. So the rule
    as it was covers a strip turn while c' + c <= max_tokens, and is short by up to
    ceil(max_tokens / P) Pages when two replies in a row are nearly all content at the bound.
    That is the term, in the phase-2 peak and (the run of phase 1) in the phase-1 peak."""
    return _cdiv(max_tokens, layout.page_tokens) if _window(layout).hold_slots else 0


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


def most_hold_slots(kv_bytes, layout, groups, block_bytes, page_bytes, needed):
    """The most hold slots (window_hold: one window slot each in pool 1) with which pool 0 still
    holds the server rule's Pages (``needed(layout, blocks)``: ``selective_pages``,
    ``release_pages`` or ``one_session_pages``), at most the layout's window slots; 0 if none."""
    import dataclasses
    best = 0
    for hold in range(1, _window(layout).window_slots + 1):
        held = dataclasses.replace(layout, hold_slots=hold)
        blocks = pool_zero_blocks(kv_bytes, held, groups, block_bytes, page_bytes)
        if blocks - 1 >= needed(held, blocks):
            best = hold
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
