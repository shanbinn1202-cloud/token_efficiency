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

With no other request running nothing could free a block: the gate admits and
the turn proceeds exactly as on a one-task server.

Decoupled KV layout (exp-decouple-pages D7): FullAttention blocks (P-token
Pages) and GDN slots (one block per Mamba group) live in two pools. The same
inequalities hold per pool: FA needs against FA free blocks (the watermark is
on FA only, as in vLLM), GDN slots against Mamba free blocks, where a running
request's crossing slot is counted while it may still cross an M-token Mamba
block. In the legacy shared pool the gate is the single sum, as before.
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


def turn_need(core, session_id, tokens, messages, generation):
    """(phase-1 peak, phase-2 peak) in blocks beyond the session's current pins.

    Decoupled layout: each peak is a (FA blocks, Mamba blocks) pair."""
    (fa1, gdn1), (fa2, gdn2) = turn_need_by_pool(core, session_id, tokens, messages, generation)
    if decoupled(core._remote_ra_growing_adapter):
        return (fa1, gdn1), (fa2, gdn2)
    return fa1 + gdn1, fa2 + gdn2


def turn_need_by_pool(core, session_id, tokens, messages, generation):
    """((phase-1 FA, phase-1 GDN), (phase-2 FA, phase-2 GDN)) blocks beyond the pins."""
    adapter = core._remote_ra_growing_adapter
    endpoint = core._remote_ra_private_v2_endpoint
    s, P = adapter.sessions[session_id], adapter.page_tokens
    att, mam, _ = endpoint.phase1.groups()
    A, M = len(att), len(mam)
    end = pages(len(tokens) + generation, P)
    if s.local_turn == 0:
        return (0, 0), (A * end, 2 * M)
    transcript = endpoint.transcripts.get(session_id)
    if transcript is not None and transcript.turn > 0 and not transcript.extends(messages):
        # Rebuild: render closes the namespace (frees R and the tail) first.
        return (0, 0), (A * (end - len(s.resident) - (s.tail is not None)), 2 * M)
    Q = s.sealed_end
    base = Q // P + (s.tail is not None)          # first Page past the pinned tail
    candidates = {Q // P, base}                   # n = floor(o/P) is one of them
    contract = getattr(endpoint.selection, 'contract', None)
    initial = transcript.initial_length if transcript is not None else 0
    largest = max(selected_max(s.policy, n, s.local_turn, contract, initial) for n in candidates)
    restores = min(Q // P - len(s.resident), largest)
    goal = 0
    if s.policy == 'goal_query' and contract is not None:
        from .selection_contract import selection_active
        if any(selection_active(n, s.local_turn, contract) for n in candidates):
            goal = pages(endpoint.phase1.max_tokens, P)   # phase 1 refuses a longer Goal
    fresh = max(0, pages(len(tokens), P) - base)
    return (A * (fresh + goal), M * (1 + (goal > 0))), (A * (end - base + restores), 2 * M)


def headroom(lifecycle):
    """Blocks the running (accepted, undrained) requests may still allocate.

    Decoupled layout: a (FA blocks, Mamba blocks) pair."""
    fa, gdn = headroom_by_pool(lifecycle)
    if decoupled(lifecycle.endpoint.core._remote_ra_growing_adapter):
        return fa, gdn
    return fa + gdn


def headroom_by_pool(lifecycle):
    """(FA blocks, GDN blocks) the running requests may still allocate."""
    endpoint = lifecycle.endpoint.core._remote_ra_private_v2_endpoint
    adapter = lifecycle.endpoint.core._remote_ra_growing_adapter
    att, mam, _ = endpoint.phase1.groups()
    A, M, P = len(att), len(mam), adapter.page_tokens
    # The Mamba align block: a GDN crossing slot is needed while the request may
    # still cross an M-token boundary (M == P in the shared pool).
    Mt = getattr(adapter, 'mamba_tokens', P) if decoupled(adapter) else P
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
        gdn += M * (cross > 0) + M * (done <= start)
    return fa, gdn


def assess(lifecycle, command):
    """The gate's numbers for one authenticated add, or None when the request's
    own checks will refuse it anyway (unknown, busy, poisoned or stale session,
    undecodable core or source)."""
    core = lifecycle.endpoint.core
    adapter = core._remote_ra_growing_adapter
    meta = command.metadata
    s = adapter.sessions.get(command.session_id)
    if (s is None or s.poisoned or s.pending is not None or type(meta) is not dict
            or meta.get('turn') != s.turn):
        return None
    try:
        native = core._remote_ra_assembly.dispatcher._decode_core(
            base64.b64decode(command.core_b64, validate=True))
        tokens = list(native.prompt_token_ids)
        messages = None
        if core._remote_ra_private_v2_endpoint.transcripts.get(command.session_id) is not None:
            # XML protocol: the messages the engine transcript renders (rebuild or not).
            from runtime_renderer import validate_xml_source
            messages = [dict(role=m['role'], content=m['content'])
                        for m in validate_xml_source(command.source)['messages']]
    except Exception:
        return None
    phase1, phase2 = turn_need(core, command.session_id, tokens, messages,
                               max_tokens(getattr(native, 'sampling_params', None)))
    manager = adapter.kv_manager
    watermark = getattr(manager, 'watermark_blocks', 0)
    watermark = watermark if type(watermark) is int else 0
    if decoupled(adapter):
        # One (FA, Mamba) pair per figure; the vLLM watermark is on the FA pool only.
        free = tuple(manager.get_num_free_blocks_by_pool())
        return dict(pools=POOLS, free=free, phase1=phase1, phase2=phase2,
                    headroom=headroom(lifecycle), watermark=(watermark, 0))
    return dict(free=manager.block_pool.get_num_free_blocks(), phase1=phase1, phase2=phase2,
                headroom=headroom(lifecycle), watermark=watermark)


def admission_gate(lifecycle, command):
    """Raise CapacityWait unless free blocks cover this turn and every running request.

    On a server that parks goal_query admissions at their Goal (goal_async), the
    running requests keep decoding between this gate and phase 1, so phase 1 must
    fit even after they have taken all their headroom (a refusal only delays the
    turn: the retry is the same turn). And no add starts while one is parked: it
    has reserved nothing yet, so the pool does not show it (the frontend's
    admission lock, held until acceptance, already keeps that from happening)."""
    goals = getattr(lifecycle, 'goals', None)
    if goals is not None and goals.parked:
        raise CapacityWait('%s: another admission waits for its Goal' % CAPACITY_WAIT)
    if not any(e['state'] == 'accepted' for e in lifecycle.entries.values()):
        return None
    need = assess(lifecycle, command)
    if need is None:
        return None
    if 'pools' in need:
        return _two_pool_gate(need, goals is not None)
    later = need['headroom'] if goals is not None else 0    # decode while the Goal is built
    if (need['phase1'] + later > need['free'] or
            need['phase2'] + need['headroom'] + need['watermark'] > need['free']):
        raise CapacityWait('%s: turn needs %d (phase 1) / %d (phase 2) blocks, running requests '
                           '%d more, watermark %d; %d free' % (CAPACITY_WAIT, need['phase1'],
                           need['phase2'], need['headroom'], need['watermark'], need['free']))
    return need


POOLS = ('FA', 'Mamba')


def _two_pool_gate(need, parked_goals):
    """The legacy inequalities, once per pool (decoupled layout)."""
    for i, pool in enumerate(need['pools']):
        free, headroom_i = need['free'][i], need['headroom'][i]
        later = headroom_i if parked_goals else 0
        if (need['phase1'][i] + later > free or
                need['phase2'][i] + headroom_i + need['watermark'][i] > free):
            raise CapacityWait('%s: %s pool: turn needs %d (phase 1) / %d (phase 2) blocks, running '
                               'requests %d more, watermark %d; %d free (%s)' % (
                               CAPACITY_WAIT, pool, need['phase1'][i], need['phase2'][i], headroom_i,
                               need['watermark'][i], free, _pool_summary(need)))
    return need


def _pool_summary(need):
    return ', '.join('%s %d free' % (pool, need['free'][i]) for i, pool in enumerate(need['pools']))
