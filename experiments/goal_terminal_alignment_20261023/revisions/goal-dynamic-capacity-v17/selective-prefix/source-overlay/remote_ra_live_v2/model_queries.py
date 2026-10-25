"""Phase-1 worker receipt (spec 2): every field the engine's Phase1Result (spec 1.8) needs."""
from hashlib import sha256
import time

from .contract import closed_pages, digest, hex64, parse_lease, require


def phase1_arm(owner, backend, rec, payload):
    """--phase1-in-step: ``phase1``'s checks and lease registration; the probe then
    runs as one row of the engine's next step (phase1_step)."""
    return phase1(owner, backend, rec, payload, mode='arm')


def phase1_collect(owner, backend, rec, payload):
    """--phase1-in-step: the step ran the armed probe row; today's phase-1 receipt."""
    return phase1(owner, backend, rec, payload, mode='collect')


def phase1(owner, backend, rec, payload, mode='run'):
    """Check the engine phase-1 request against the prepared turn, run it, sign it.

    The request names the prepared binding (its minted identity, unique per
    binding object, which is frozen with an immutable token tuple), the
    allocation generation and identity, and carries the tokens phase 1 computes,
    compared here with the prepared ones. Its prompt_digest is that binding's
    canonical prompt hash, computed by the engine in this process (the UniProc
    worker) from the same tokens the prepare payload carried; it is not hashed
    again here (it travels on, inside request_digest, into the receipt).

    ``mode`` (--phase1-in-step): 'arm' checks and registers the lease, and
    'collect' signs what the engine's step computed with the lease armed then;
    'run' (the default) is the separate pass."""
    a, identity = rec['assignment'], rec['identity']
    C, e, P = a.computed_start, a.observation_end, a.page_tokens
    if mode == 'collect':
        # --phase1-in-step: the engine's step ran the probe row armed below.
        armed = rec.get('phase1_armed')
        require(armed is not None and rec['phase1'] is None and 'plan' not in rec and
                armed['lease'].lease_id == payload['lease']['lease_id'] and
                armed['lease'].lease_digest == payload['lease_digest'] and payload['binding'] == rec['binding'],
                'in-step phase-1 collect without its arm')
        lease, start = armed['lease'], armed['start']
        output = owner.runner._remote_ra_native_query_forward.collect(payload)
    else:
        require(mode in ('run', 'arm'), 'unknown phase-1 mode')
        require(a.local_turn > 0 and rec['phase1'] is None and 'plan' not in rec,
                'phase 1 runs once, after prepare and before reserve')
        require(payload['allocation_generation'] == rec['generation'] and
                payload['binding'] == rec['binding'] and payload['source'] == identity.source and
                payload['model'] == identity.model and
                (payload['session'], payload['turn'], payload['generation'], payload['local_turn'])
                == (identity.session, identity.turn, identity.generation, a.local_turn) and
                list(payload['tokens']) == list(rec['tokens'][:e]) and
                hex64(payload['prompt_digest']) and hex64(payload['request_digest']),
                'stale phase-1 allocation/render')
        # Decoupled: Mamba rows of ceil(e/M) and pool-keyed blocks (WP2's parse_lease
        # takes the layout); legacy calls it exactly as before.
        lease = parse_lease(payload['lease'], attention_groups=owner.attention_groups,
                            **({'layout': owner.layout} if getattr(owner, 'decoupled', False) else {}))
        s = owner.sessions[rec['namespace']]
        tail = s['tail']
        require(lease.lease_digest == payload['lease_digest'] and
                (lease.request, lease.binding, lease.session, lease.namespace, lease.generation,
                 lease.source, lease.epoch) ==
                (identity.request, rec['binding'], identity.session, rec['namespace'],
                 rec['generation'], identity.source, payload['epoch']) and
                sha256(lease.epoch.encode()).hexdigest() == identity.epoch and
                (lease.local_turn, lease.page_tokens, lease.start, lease.end, lease.sealed_start,
                 lease.candidate_end, lease.tail_candidate) ==
                (a.local_turn, P, C, e, a.sealed_start, a.candidate_end, a.tail_candidate) and
                lease.carry_digest == a.carry_digest == s['carry_digest'] and
                lease.resident_pages == tuple(s['resident']) and
                (lease.tail is None) == (tail is None), 'phase-1 lease differs from the prepared turn')
        # Borrowed = the session's resident Page blocks in R order, then the tail
        # block: exactly what phase 1 reads, never zeroed or freed by the lease.
        for fi, g in enumerate(owner.attention_groups):
            expect = tuple(s['resident_blocks'][p][g][0] for p in lease.resident_pages)
            if tail is not None:
                require(lease.tail.block_ids[fi] == tail['blocks'][g], 'phase-1 tail block changed')
                expect += tail['blocks'][g]
            require(lease.rows[g][:len(expect)] == expect, 'phase-1 borrowed blocks differ from pins')
        window = getattr(owner, 'check_phase1_window', None)
        if window is not None:
            # The window layout (gemma-s1): the sliding rows borrow the committed window.
            window(rec, lease, s)
        driver = getattr(owner.runner, '_remote_ra_native_query_forward', None)
        require(callable(getattr(driver, 'phase1', None)),
                'LOADED_QUERY_FORWARD_UNAVAILABLE: native phase-1 driver is not attached')
        if mode == 'arm':
            # --phase1-in-step: register the lease; the probe runs in the engine's next step.
            require(getattr(owner, 'phase1_in_step', False) is True and lease.goal is None and
                    'phase1_armed' not in rec, 'phase 1 in a step needs --phase1-in-step, no Goal forward, once')
            driver.arm(payload)
            from remote_ra_growing_worker.tensors import request_counters
            # The collect frame is measured from here: the carry restore runs in the step.
            rec['phase1_armed'] = dict(lease=lease, start=time.perf_counter_ns(),
                                       counters=request_counters(backend.io, identity.request))
            return dict(operation='phase1_arm', binding=rec['binding'], lease_id=lease.lease_id,
                        lease_digest=lease.lease_digest, drained=False)
        start = time.perf_counter_ns()
        output = driver.phase1(payload)
    import torch
    rows, goal_rows = output['rows'], output['goal_rows']
    compactor = getattr(backend, 'compactor', None)
    from .query_selection import normalized
    if compactor is None:
        require(isinstance(rows, torch.Tensor) and rows.ndim == 2 and rows.shape[0] == e - C and
                bool(torch.isfinite(rows).all()), 'exact phase-1 forward rows required')
        rows = rows.detach().to(device='cpu', dtype=torch.float32)
        # Pages whose last token lies in [C, e): first computation, used now and
        # committed at completion; the observation query is row e-1.
        vectors = {p: normalized(rows[(p + 1) * P - 1 - C].tolist()) for p in closed_pages(C, e, P)}
        observation = normalized(rows[e - 1 - C].tolist())
        wire = {str(p): list(v) for p, v in vectors.items()}
    else:
        # Decoupled: the probe kept only the closing rows (on their device,
        # compacted there together) and the observation query row (host fp32).
        from .native_probe import ProbeRows
        pages = closed_pages(C, e, P)
        require(type(rows) is ProbeRows and rows.start == C and rows.count == e - C and
                rows.finite and rows.pages == tuple(pages), 'exact phase-1 forward rows required')
        vectors = dict(zip(pages, compactor.compact(rows.closing))) if pages else {}
        observation = normalized(rows.last.tolist())
        from .page_compaction import to_wire
        wire = to_wire(vectors)
    goal = None
    if lease.goal is not None:
        if compactor is None:
            require(isinstance(goal_rows, torch.Tensor) and goal_rows.ndim == 2 and
                    goal_rows.shape[0] == len(lease.goal.tokens) and
                    bool(torch.isfinite(goal_rows).all()), 'exact Goal forward rows required')
            goal = normalized(goal_rows[-1].detach().to(device='cpu', dtype=torch.float32).tolist())
        else:
            require(type(goal_rows) is ProbeRows and goal_rows.start == 0 and
                    goal_rows.count == len(lease.goal.tokens) and goal_rows.finite and
                    goal_rows.closing is None, 'exact Goal forward rows required')
            goal = normalized(goal_rows.last.tolist())
    else:
        require(goal_rows is None, 'Goal rows without a Goal lease')
    width = rows.shape[1] if compactor is None else rows.hidden_size
    require(len(observation) == width and (goal is None or len(goal) == len(observation)),
            'query/model hidden dimension mismatch')
    state, captured = output['state'], output['tail_capture']
    require(state.request == identity.request and state.lease_id == lease.lease_id and
            state.end == e and (captured is not None) == lease.tail_candidate,
            'phase-1 state/tail capture incomplete')
    tail_digest = backend.io.payload_digest(captured) if captured is not None else None
    backend.io.drain()
    body = dict(operation='phase1', binding=rec['binding'], request_digest=payload['request_digest'],
        lease_id=lease.lease_id, lease_digest=lease.lease_digest,
        allocation_generation=rec['generation'], start=C, end=e, forward_tokens=e - C,
        page_vectors=wire,
        observation_vector=list(observation), goal_vector=None if goal is None else list(goal),
        goal_token_count=None if goal is None else len(lease.goal.tokens),
        goal_tokens_sha256=None if goal is None else digest(list(lease.goal.tokens)),
        goal_payload_sha256=None if goal is None else lease.goal.payload_sha256,
        state_digest=state.digest, tail_capture_digest=tail_digest,
        carry_restored=dict(start=C, digest=lease.carry_digest),
        position_evidence=output['position_evidence'],
        vector_space=dict(getattr(owner.runner, '_remote_ra_vector_space', None) or {}),
        representation='pre_final_norm_last_token_l2', phase1_ns=time.perf_counter_ns() - start,
        drained=True)
    if compactor is not None:
        # Interface I3: Page vectors compact; the observation/Goal query
        # vectors keep the float representation.
        from .page_compaction import QUERY_REPRESENTATION, REPRESENTATION
        body.update(representation=REPRESENTATION, query_representation=QUERY_REPRESENTATION)
    receipt = backend.sign_receipt(identity, 'phase1', body)
    backend.verify_receipt(identity, 'phase1', receipt)
    rec['phase1'] = dict(lease=lease, vectors=vectors, obs=observation, goal=goal,
        tail_capture=captured, tail_capture_digest=tail_digest, state_digest=state.digest,
        receipt=receipt)
    return dict(body, worker_receipt=receipt)

