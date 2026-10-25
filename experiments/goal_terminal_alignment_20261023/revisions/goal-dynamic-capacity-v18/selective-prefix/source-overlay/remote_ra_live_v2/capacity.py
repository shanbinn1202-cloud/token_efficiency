"""Multi-session stage 2: the admission capacity gate. No preemption, ever.

Several sessions decode in one vLLM step. vLLM preempts a running request when
a step finds no free block for it, and a preemption is fatal here. So before a
new request changes anything, the engine checks that the pool's free blocks
cover (capacity/CAPACITY.md, survey 4.3; one block = one Page of one
FullAttention group, a GDN slot = one block per Mamba group):

* this turn's worst-case peak beyond what its session already pins
  (I = |R| + tail), P = max(P1, P2 + 3) - I:
  - phase 1, while nothing else runs: the fresh run Pages and one GDN slot,
    plus for goal_query a Goal forward of at most one native forward's Pages
    and its own GDN slot;
  - phase 2: the restores |A_t minus R'|, bounded by min(sealed Pages not
    resident, the policy's largest selected count), the row from the session's
    tail to ceil((N + max_tokens) / P), the running GDN slot and one
    page-crossing slot. A rebuild frees the session's pins and needs the whole
    row; a local turn 0 needs the whole row;
* the headroom every running request may still allocate until it ends: its row
  Pages up to ceil((N + max_tokens) / P), one page-crossing GDN slot (vLLM's
  align mode frees the previous state block one step after allocating the
  next, so crossings never stack) and, before its first step, its running slot.

Released Pages (R minus A_t) stay pinned until commit: they are already in use
and never added. Completions only free. The vLLM watermark is added once.

Placement: RegisteredRouteBridge.add calls ``admission_gate`` after the codec
authenticated the command and before the dispatcher record (request
tombstone), the session's turn and generation check, the lifecycle entry, the
transcript render, phase 1 or any pin. The gate only reads, so its CapacityWait
leaves the engine exactly as it was. RegisteredAddDispatcher.handle returns it
as an authenticated ``rejected`` Ack whose error starts with ``CapacityWait:
CAPACITY_WAIT``; the frontend sends no compensation Cancel, undoes its capture
and authorization, releases admission, keeps the session slot and retries from
capture after a completion or a short backoff.

With no other request running no completion can free a block: the gate admits and
the turn proceeds exactly as on a one-task server, unless other open sessions pin
blocks (``pinned_elsewhere``). Then the turn is checked as above (no headroom: nothing
runs) and, when it does not fit, waits (CAPACITY_WAIT) until a session close or
--release-idle-pages makes room. Admitting it unchecked let its decode take the last
free block and vLLM preempt it (tb16 job 3398129: 16 chat-text sessions of up to 60K
tokens, `full`, 32 GiB, the idle sessions' pins 26.6K of 26.7K blocks; fatal). The
window layout (Gemma) holds the same rule since integrate-1018, where a pinned window
in pool 1 is a pin too; before, it admitted such a turn unchecked whatever was pinned.

Session pool (session_pool.py): only open sessions pin blocks, so the free
blocks the gate reads already include every closed session's; the first turn of
a registered session that is not open yet is assessed as a local turn 0.

--gdn-device-handoff: the probe's GDN slot stays allocated until the first
phase-2 step has copied it, so a continuing turn's phase-2 peak holds three
GDN slots (running, crossing, held probe) instead of two. A held slot whose
request's first phase-2 step has executed is freed at that request's next
allocation; vLLM schedules running requests before waiting ones, so that free
precedes any new request's first phase-2 allocation and the phase-2 inequality
counts it as free (``releasing``). Phase 1 allocates at once and is checked
against the blocks free now.

Decoupled KV layout (exp-decouple-pages D7): FullAttention blocks (P-token
Pages) and GDN slots (one block per Mamba group) live in two pools. The same
inequalities hold per pool: FA needs against FA free blocks (the watermark is
on FA only, as in vLLM), GDN slots against Mamba free blocks, where a running
request's crossing slot is counted while it may still cross an M-token Mamba
block. In the legacy shared pool the gate is the single sum, as before.

--admission-overlap K (admission design (f), section 2.3): up to K admissions
are in flight (gated, not yet accepted) at once, so a parked admission no
longer refuses every other add. Instead each in-flight admission X holds a
reservation, stored at its gate (``lifecycle.reservations``; the gate and the
lifecycle entry are created in one engine-thread call before any yield):
r_X = max(P1_X, P2_X) - held_X per pool, where P2_X of a rebuild also counts
the pins its render frees and held_X is what X's phase-1 lease holds now
(``Phase1Coordinator.held``; nothing else is allocated while X is parked).
Per pool, a new admission needs

    phase 1:  P1 + sum r + headroom <= free
    phase 2:  P2 + sum r + headroom + watermark <= free + releasing

(the running requests decode while it waits, so phase 1 counts their headroom
too). "Nothing accepted: admit" holds only when nothing is in flight either (and,
as above, no other session pins blocks). A
reservation ends when its admission is accepted (its rest is then headroom) or
fails; it is taken only at the gate and nothing waits while holding one except
for its Goal or the one phase-1 lease, neither of which needs blocks: no
deadlock, and a refused admission only retries (CAPACITY_WAIT, unchanged).
With K = 1 (the default) none of this runs: the gate is the one above.

The window layout (Gemma, gemma-goal) reserves its sliding pool differently: each
in-flight admission reserves what its turn keeps once accepted (its base) and the
phase-1 transients (fresh blocks, the Goal forward) are reserved once, the largest of
them, because phase 1 runs one lease at a time within one engine-thread call there
(no --phase1-in-step); ``window_capacity.ledger_ahead``. Its FA pool is reserved as above.
"""
from __future__ import annotations

import base64

from .contract import LiveV2Error

CAPACITY_WAIT = 'CAPACITY_WAIT'
BACKOFF_SECONDS = 0.5
QUERY = frozenset({'goal_query', 'observation_query', 'random_fill'})


class CapacityWait(LiveV2Error):
    """Distinct, non-poisoning refusal: the engine changed nothing; retry from capture."""


def refused(error):
    """Whether an authenticated rejection's error text is the gate's CAPACITY_WAIT."""
    return type(error) is str and error.startswith('CapacityWait: ' + CAPACITY_WAIT)


def waiting(failure):
    """Whether a frontend failure is exactly a CAPACITY_WAIT refusal, possibly
    behind vLLM's ``raise ... from`` wrappers. A group (a second failure) is not."""
    for _ in range(16):
        if isinstance(failure, CapacityWait):
            return True
        if failure is None or isinstance(failure, BaseExceptionGroup):
            return False
        failure = failure.__cause__
    return False


def pages(tokens, page_tokens):
    return -(-tokens // page_tokens)


def max_tokens(params):
    """Generation bound. The XML protocol pins 1536 and check_sampling refuses any
    other value before allocation; a larger value, if present, is used instead."""
    from .mswea_protocol import MAX_TOKENS
    value = getattr(params, 'max_tokens', None)
    return max(MAX_TOKENS, value) if type(value) is int else MAX_TOKENS


def selected_max(policy, n, local_turn, contract, initial):
    """The largest selected Page count of a turn with ``n`` candidates (select_pages)."""
    if policy == 'random50':
        from remote_ra_growing.contract import sparse_count
        return n if n < 2 else sparse_count(n)
    if policy in QUERY:
        from . import selection_contract as sc
        if contract is None or not sc.selection_active(n, local_turn, contract):
            return n
        mandatory = sc.mandatory_pages(initial, n, contract) if initial > 0 else n
        return sc.capacity_pages(n, mandatory, contract)
    return n


def decoupled(adapter):
    """Whether the adapter runs the decoupled two-pool KV layout."""
    return getattr(adapter, 'decoupled', False) is True


def turn_need(core, session_id, tokens, messages, generation, *, freed=None):
    """(phase-1 peak, phase-2 peak) in blocks beyond the session's current pins.

    Decoupled layout: each peak is a (FA blocks, Mamba blocks) pair. ``freed``
    (a list, --admission-overlap): gets the FA blocks a rebuild's render frees."""
    (fa1, gdn1), (fa2, gdn2) = turn_need_by_pool(core, session_id, tokens, messages, generation, freed=freed)
    if decoupled(core._remote_ra_growing_adapter):
        return (fa1, gdn1), (fa2, gdn2)
    return fa1 + gdn1, fa2 + gdn2


def turn_need_by_pool(core, session_id, tokens, messages, generation, *, freed=None):
    """((phase-1 FA, phase-1 GDN), (phase-2 FA, phase-2 GDN)) blocks beyond the pins.

    ``freed`` (a list): a rebuild appends the FA blocks its render frees (its R and
    tail), which the turn then needs again; nothing is appended otherwise.

    The window layout (gemma-s1 WP4): the FA blocks as below, the sliding-window
    blocks of ``window_capacity.gate_turn_need`` in place of the GDN slots."""
    need = _need_by_pool(core, session_id, tokens, messages, generation, freed=freed)
    if getattr(core._remote_ra_growing_adapter, 'window', None) is None:
        return need
    from .window_capacity import gate_turn_need
    (fa1, _), (fa2, _) = need
    sliding1, sliding2 = gate_turn_need(core, session_id, tokens, messages, generation)
    return (fa1, sliding1), (fa2, sliding2)


def _need_by_pool(core, session_id, tokens, messages, generation, *, freed=None):
    adapter = core._remote_ra_growing_adapter
    endpoint = core._remote_ra_private_v2_endpoint
    s, P = adapter.sessions.get(session_id), adapter.page_tokens
    att, mam, _ = endpoint.phase1.groups()
    A, M = len(att), len(mam)
    end = pages(len(tokens) + generation, P)
    if s is None or s.local_turn == 0:      # None: a session-pool session opening now
        return (0, 0), (A * end, 2 * M)
    # --release-idle-pages: an evicted session's Pages are readmitted (pinned again)
    # before its phase 1 and stay pinned through phase 2.
    evicted = len(getattr(s, 'evicted', ()))
    transcript = endpoint.transcripts.get(session_id)
    if transcript is not None and transcript.turn > 0 and not transcript.extends(messages):
        # Rebuild: render closes the namespace (frees R and the tail) first.
        if freed is not None:
            freed.append(A * (len(s.resident) + (s.tail is not None)))
        return (0, 0), (A * (end - len(s.resident) - (s.tail is not None)), 2 * M)
    Q = s.sealed_end
    base = Q // P + (s.tail is not None)          # first Page past the pinned tail
    candidates = {Q // P, base}                   # n = floor(o/P) is one of them
    contract = getattr(endpoint.selection, 'contract', None)
    initial = transcript.initial_length if transcript is not None else 0
    largest = max(selected_max(s.policy, n, s.local_turn, contract, initial) for n in candidates)
    restores = min(Q // P - len(s.resident) - evicted, largest)
    goal = 0
    if s.policy == 'goal_query' and contract is not None:
        from .selection_contract import selection_active
        if any(selection_active(n, s.local_turn, contract) for n in candidates):
            # Phase 1 refuses a longer Goal: one native forward, or the window layout's
            # Goal transient (WindowPhase1Coordinator.goal_tokens, gemma-goal).
            goal = pages(getattr(endpoint.phase1, 'goal_tokens', endpoint.phase1.max_tokens), P)
    fresh = max(0, pages(len(tokens), P) - base)
    # --gdn-device-handoff: the held probe slot is the third GDN slot of phase 2.
    slots = 3 if getattr(endpoint.phase1, 'handoff', None) is not None else 2
    return ((A * (fresh + goal + evicted), M * (1 + (goal > 0))),
            (A * (end - base + restores + evicted), slots * M))


def headroom(lifecycle):
    """Blocks the running (accepted, undrained) requests may still allocate.

    Decoupled layout: a (FA blocks, Mamba blocks) pair."""
    fa, gdn = headroom_by_pool(lifecycle)
    if decoupled(lifecycle.endpoint.core._remote_ra_growing_adapter):
        return fa, gdn
    return fa + gdn


def headroom_by_pool(lifecycle):
    """(FA blocks, GDN blocks) the running requests may still allocate.

    The window layout (gemma-s1 WP4): (FA blocks, sliding-window blocks), the latter
    from ``window_capacity.gate_headroom``."""
    fa, gdn = _headroom_by_pool(lifecycle)
    if getattr(lifecycle.endpoint.core._remote_ra_growing_adapter, 'window', None) is None:
        return fa, gdn
    from .window_capacity import gate_headroom
    return fa, gate_headroom(lifecycle)


def _headroom_by_pool(lifecycle):
    endpoint = lifecycle.endpoint.core._remote_ra_private_v2_endpoint
    adapter = lifecycle.endpoint.core._remote_ra_growing_adapter
    att, mam, _ = endpoint.phase1.groups()
    A, M, P = len(att), len(mam), adapter.page_tokens
    # The Mamba align block: a GDN crossing slot is needed while the request may
    # still cross an M-token boundary (M == P in the shared pool).
    Mt = getattr(adapter, 'mamba_tokens', P) if decoupled(adapter) else P
    # --gdn-device-handoff: a request's GDN peak (running + crossing) less the slots it
    # holds now; vLLM frees a crossing's old slot before it allocates the next one.
    held = held_slots(adapter, mam) if getattr(endpoint.phase1, 'handoff', None) is not None else None
    fa = gdn = 0
    for rid, entry in lifecycle.entries.items():
        request = entry['request']
        if entry['state'] != 'accepted' or request is None:
            continue
        geometry = endpoint.claims.get(rid, {}).get('geometry')
        start = 0 if geometry is None else geometry.phase2_start
        done = request.num_computed_tokens
        total = len(request.prompt_token_ids) + max_tokens(request.sampling_params)
        left = max(0, pages(total, P) - pages(done, P))
        cross = left if Mt == P else max(0, pages(total, Mt) - pages(done, Mt))
        fa += A * left
        if held is not None:
            gdn += M * max(0, 1 + (cross > 0) - held(rid))
            continue
        gdn += M * (cross > 0) + M * (done <= start)
    return fa, gdn


def held_slots(adapter, mamba_groups):
    """GDN slots (live Mamba blocks of the first Mamba group) a request holds now."""
    managers = adapter.kv_manager.coordinator.single_type_managers
    manager = managers[mamba_groups[0]]

    def count(rid):
        return sum(1 for block in manager.req_to_blocks.get(rid, ()) if not block.is_null)
    return count


def releasing(lifecycle):
    """--gdn-device-handoff: GDN blocks of held probe slots whose request's first
    phase-2 step has executed (freed at its next allocation); 0 when off."""
    endpoint = lifecycle.endpoint.core._remote_ra_private_v2_endpoint
    held = getattr(endpoint.phase1, 'handoff', None)
    if held is None:
        return 0
    blocks = 0
    for rid, (_, slot) in held.slots.items():
        entry = lifecycle.entries.get(rid)
        if entry is None or entry['state'] != 'accepted' or entry['request'] is None:
            continue
        geometry = endpoint.claims.get(rid, {}).get('geometry')
        if geometry is not None and entry['request'].num_computed_tokens > geometry.phase2_start:
            blocks += len(slot)
    return blocks


def opening(core, session_id):
    """Whether ``session_id`` is a session-pool session that its next turn 0 opens."""
    authority = getattr(core, '_remote_ra_live_session_authority', None)
    return (getattr(authority, 'limit', None) is not None and session_id in authority.ids and
            session_id not in authority.namespaces and session_id not in authority.ended)


def assess(lifecycle, command, *, freed=None):
    """The gate's numbers for one authenticated add, or None when the request's
    own checks will refuse it anyway (unknown, busy, poisoned or stale session,
    undecodable core or source). ``freed``: see ``turn_need_by_pool``."""
    core = lifecycle.endpoint.core
    adapter = core._remote_ra_growing_adapter
    meta = command.metadata
    s = adapter.sessions.get(command.session_id)
    if s is None:
        # Session pool: a registered session that this turn 0 opens.
        if not (opening(core, command.session_id) and type(meta) is dict and meta.get('turn') == 0):
            return None
    elif (s.poisoned or s.pending is not None or type(meta) is not dict
            or meta.get('turn') != s.turn):
        return None
    try:
        native = core._remote_ra_assembly.dispatcher._decode_core(
            base64.b64decode(command.core_b64, validate=True))
        tokens = list(native.prompt_token_ids)
        messages = None
        transcript = core._remote_ra_private_v2_endpoint.transcripts.get(command.session_id)
        from .kv_session_protocol import kv
        from .chat_text_protocol import chat
        if chat(transcript):
            # chat-text: the messages its engine transcript renders (rebuild or not).
            from runtime_renderer import validate_chat_source
            messages = [dict(role=m['role'], content=m['content'])
                        for m in validate_chat_source(command.source)['messages']]
        elif transcript is not None and not kv(transcript):
            # XML protocol: the messages the engine transcript renders (rebuild or not);
            # a kv-session never rebuilds (its transcript extends every request).
            from runtime_renderer import validate_xml_source
            messages = [dict(role=m['role'], content=m['content'])
                        for m in validate_xml_source(command.source)['messages']]
    except Exception:
        return None
    phase1, phase2 = turn_need(core, command.session_id, tokens, messages,
                               max_tokens(getattr(native, 'sampling_params', None)), freed=freed)
    manager = adapter.kv_manager
    watermark = getattr(manager, 'watermark_blocks', 0)
    watermark = watermark if type(watermark) is int else 0
    soon = releasing(lifecycle)
    if decoupled(adapter):
        # One (FA, Mamba) pair per figure; the vLLM watermark is on the FA pool only.
        free = tuple(manager.get_num_free_blocks_by_pool())
        pools = POOLS if getattr(adapter, 'window', None) is None else WINDOW_POOLS
        need = dict(pools=pools, free=free, phase1=phase1, phase2=phase2,
                    headroom=headroom(lifecycle), watermark=(watermark, 0))
        if soon:
            need['releasing'] = (0, soon)
        if pools is WINDOW_POOLS and getattr(lifecycle, 'reservations', None) is not None:
            # --admission-overlap on the window layout (gemma-goal): the sliding (base, peak)
            # of the ledger (window_capacity.gate_ledger).
            from .window_capacity import gate_ledger
            need['sliding_ledger'] = gate_ledger(core, command.session_id, tokens, messages,
                                                 max_tokens(getattr(native, 'sampling_params', None)))
        return need
    need = dict(free=manager.block_pool.get_num_free_blocks(), phase1=phase1, phase2=phase2,
                headroom=headroom(lifecycle), watermark=watermark)
    if soon:
        need['releasing'] = soon
    return need


def admission_gate(lifecycle, command):
    """The capacity gate; with --release-idle-pages a refused turn first releases idle
    sessions' GPU Pages (remote_ra_live_v2.idle_release)."""
    from .idle_release import attached
    release = attached(lifecycle)
    if release is None:
        return gate(lifecycle, command)
    return release.admit(lifecycle, command, gate)


def gate(lifecycle, command):
    """Raise CapacityWait unless free blocks cover this turn and every running request.

    On a server that parks goal_query admissions at their Goal (goal_async), the
    running requests keep decoding between this gate and phase 1, so phase 1 must
    fit even after they have taken all their headroom (a refusal only delays the
    turn: the retry is the same turn). And no add starts while one is parked: it
    has reserved nothing yet, so the pool does not show it (the frontend's
    admission lock, held until acceptance, already keeps that from happening).

    --admission-overlap (``lifecycle.reservations``): the ledger gate instead."""
    if getattr(lifecycle, 'reservations', None) is not None:
        return ledger_gate(lifecycle, command)
    goals = getattr(lifecycle, 'goals', None)
    if goals is not None and goals.parked:
        raise CapacityWait('%s: another admission waits for its Goal' % CAPACITY_WAIT)
    # Nothing runs: admitted unchecked (None), unless other sessions' pins may leave it no room.
    alone = not any(e['state'] == 'accepted' for e in lifecycle.entries.values())
    if alone and not pinned_elsewhere(lifecycle, getattr(command, 'session_id', None)):
        return None
    need = assess(lifecycle, command)
    if need is None:
        return None
    if 'pools' in need:
        _two_pool_gate(need, goal_parks(goals))
        return None if alone else need
    later = need['headroom'] if goal_parks(goals) else 0    # decode while the Goal is built
    if (need['phase1'] + later > need['free'] or
            need['phase2'] + need['headroom'] + need['watermark'] > need['free'] + need.get('releasing', 0)):
        raise CapacityWait('%s: turn needs %d (phase 1) / %d (phase 2) blocks, running requests '
                           '%d more, watermark %d; %d free' % (CAPACITY_WAIT, need['phase1'],
                           need['phase2'], need['headroom'], need['watermark'], need['free']))
    return None if alone else need


def pinned_elsewhere(lifecycle, session_id):
    """Whether another open session pins blocks (resident Pages or a live tail; on the window
    layout also its pinned window in pool 1): blocks no completion frees, so a turn admitted
    while nothing runs may find no room, but a session close or --release-idle-pages may free
    them later. The window layout (Gemma) admitted such a turn unchecked until integrate-1018,
    the hole tb16 job 3398129 found in the Qwen gate (``window_capacity.pins_window``)."""
    core = getattr(getattr(lifecycle, 'endpoint', None), 'core', None)
    adapter = getattr(core, '_remote_ra_growing_adapter', None)
    if adapter is None:
        return False
    sessions = getattr(adapter, 'sessions', None) or {}
    if getattr(adapter, 'window', None) is None:
        return any(sid != session_id and (getattr(s, 'resident', ()) or getattr(s, 'tail', None) is not None)
                   for sid, s in sessions.items())
    from .window_capacity import pins_window
    return any(sid != session_id and (getattr(s, 'resident', ()) or getattr(s, 'tail', None) is not None
                                      or pins_window(adapter, s))
               for sid, s in sessions.items())


POOLS = ('FA', 'Mamba')
WINDOW_POOLS = ('FA', 'sliding')      # the window layout (gemma-s1)


def goal_parks(goals):
    """Whether admissions may park at their Goal before phase 1 (decodes then run on).
    A park table built for --phase1-in-step alone (no rewriter) never parks a Goal,
    and an in-step phase 1 leases its blocks before it parks."""
    return goals is not None and getattr(goals, 'goals', True) is not False


def _two_pool_gate(need, parked_goals):
    """The legacy inequalities, once per pool (decoupled layout)."""
    for i, pool in enumerate(need['pools']):
        free, headroom_i = need['free'][i], need['headroom'][i]
        later = headroom_i if parked_goals else 0
        soon = need['releasing'][i] if 'releasing' in need else 0
        if (need['phase1'][i] + later > free or
                need['phase2'][i] + headroom_i + need['watermark'][i] > free + soon):
            raise CapacityWait('%s: %s pool: turn needs %d (phase 1) / %d (phase 2) blocks, running '
                               'requests %d more, watermark %d; %d free (%s)' % (
                               CAPACITY_WAIT, pool, need['phase1'][i], need['phase2'][i], headroom_i,
                               need['watermark'][i], free, _pool_summary(need)))
    return need


def _pool_summary(need):
    return ', '.join('%s %d free' % (pool, need['free'][i]) for i, pool in enumerate(need['pools']))


# ---- --admission-overlap: the reservation ledger (admission design (f), section 2.3) ----------

# An admission is in flight from its gate until it is accepted or fails: its
# lifecycle entry is 'rendered' or 'reserving' (parked at its Goal, waiting for
# the phase-1 lease or parked at its in-step phase 1). A 'rejected' entry never
# rendered and is terminal, as are 'cancelled' and 'drain_pending' (blocks held
# by a failed cleanup are already allocated and never grow).
IN_FLIGHT = frozenset({'rendered', 'reserving'})


def reservation(need, freed):
    """An admission's reservation at its gate: max(P1, P2) per pool, where a
    rebuild's P2 also counts the FA blocks its render frees (``freed``). The window
    layout's sliding part is its (base, peak) pair (``sliding_ledger``, gemma-goal)."""
    extra = sum(freed)
    if 'pools' in need:
        (fa1, gdn1), (fa2, gdn2) = need['phase1'], need['phase2']
        sliding = need.get('sliding_ledger')
        return max(fa1, fa2 + extra), (max(gdn1, gdn2) if sliding is None else tuple(sliding))
    return max(need['phase1'], need['phase2'] + extra)


def reserved(lifecycle, transient=None):
    """(sum of r_X over the in-flight admissions, how many are in flight). r_X =
    X's reservation less what its phase-1 lease holds now, per pool: a (FA, Mamba)
    pair when the stored reservations are pairs (decoupled), else blocks.
    Reservations whose admission was accepted or failed are dropped here.

    The window layout (gemma-goal): the sliding part sums every base and adds the
    largest phase-1 transient once (window_capacity.ledger_ahead), which ``transient``
    (a list) receives."""
    reservations = lifecycle.reservations
    phase1 = lifecycle.endpoint.core._remote_ra_private_v2_endpoint.phase1
    fa = gdn = flying = 0
    window = []
    for rid in list(reservations):
        entry = lifecycle.entries.get(rid)
        if entry is None or entry['state'] not in IN_FLIGHT:
            del reservations[rid]
            continue
        flying += 1
        held_fa, held_gdn = phase1.held(rid)
        r = reservations[rid]
        if type(r) is tuple and type(r[1]) is tuple:
            fa += max(0, r[0] - held_fa)
            window.append((r[1], held_gdn))
        elif type(r) is tuple:
            fa, gdn = fa + max(0, r[0] - held_fa), gdn + max(0, r[1] - held_gdn)
        else:
            fa += max(0, r - held_fa - held_gdn)
    if window:
        from .window_capacity import ledger_ahead
        total, largest = ledger_ahead(window)
        gdn += total + largest
        if transient is not None:
            transient.append(largest)
    return ((fa, gdn) if decoupled(lifecycle.endpoint.core._remote_ra_growing_adapter) else fa), flying


def ledger_phase1(need, transient):
    """An admission's phase-1 figures on top of the ledger's ``reserved``: its phase-1
    need per pool; on the window layout (``sliding_ledger``) the sliding one is its base
    plus what its transient adds to the largest in flight (``transient``, as ``reserved``
    filled it; window_capacity.ledger_phase1)."""
    sliding = need.get('sliding_ledger')
    if sliding is None:
        return need['phase1']
    from .window_capacity import ledger_phase1 as window
    return need['phase1'][0], window(sliding, max(transient, default=0))


def release_reservation(lifecycle, request_id):
    """An admission ended (accepted: its rest is now headroom; or failed): its
    reservation leaves the ledger. No-op without --admission-overlap."""
    reservations = getattr(lifecycle, 'reservations', None)
    if reservations is not None:
        reservations.pop(request_id, None)


def ledger_gate(lifecycle, command):
    """``admission_gate`` with admissions in flight: raise CapacityWait unless the
    free blocks cover this turn, every in-flight admission's reservation and every
    running request; else record this admission's reservation and return the
    numbers. The running requests decode while it is parked, so phase 1 counts
    their headroom too. Only reads until it admits: a CapacityWait changes nothing."""
    freed = []
    need = assess(lifecycle, command, freed=freed)
    if need is None:
        return None                 # the request's own checks refuse it: it reserves nothing
    transient = []
    ahead, flying = reserved(lifecycle, transient=transient)
    need.update(reserved=ahead, in_flight=flying, reservation=reservation(need, freed))
    # Nothing runs or is in flight: admitted unchecked, unless other sessions' pins may leave it
    # no room (``gate``, tb16 job 3398129).
    if (flying or any(e['state'] == 'accepted' for e in lifecycle.entries.values())
            or pinned_elsewhere(lifecycle, getattr(command, 'session_id', None))):
        pools = need.get('pools', ('blocks',))
        pair = lambda value: value if 'pools' in need else (value,)
        free, phase1, phase2 = pair(need['free']), pair(ledger_phase1(need, transient)), pair(need['phase2'])
        room, watermark, ahead = pair(need['headroom']), pair(need['watermark']), pair(ahead)
        soon = pair(need.get('releasing', (0, 0) if 'pools' in need else 0))
        for i, pool in enumerate(pools):
            if (phase1[i] + ahead[i] + room[i] > free[i] or
                    phase2[i] + ahead[i] + room[i] + watermark[i] > free[i] + soon[i]):
                raise CapacityWait('%s: %s: turn needs %d (phase 1) / %d (phase 2) blocks, %d admissions in '
                                   'flight %d more, running requests %d more, watermark %d; %d free' % (
                                   CAPACITY_WAIT, pool, phase1[i], phase2[i], flying, ahead[i], room[i],
                                   watermark[i], free[i]))
    lifecycle.reservations[command.request_id] = need['reservation']
    return need
