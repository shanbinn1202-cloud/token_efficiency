"""Stage B engine side (WP-E M1) against a fake worker answering the spec 2 ops.

CPU only. FakePool follows vllm/v1/core/block_pool.py (ref counts, free queue,
get_new_blocks/touch/free_blocks) and a fake KV manager allocates FA rows by the
physical cursor and null-padded Mamba rows. The real EngineWorkerProxy,
EngineRpcBoundary, LiveEndpoint, SplitPrivateEndpoint.reserve/rebuild,
Phase1Coordinator, LiveSessionAuthority close, EngineCoreAdapter and
PrivateBlockOwner run unmodified. Nothing here is CUDA evidence or READY.
"""
from collections import deque
from hashlib import sha256
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))

from remote_ra_live_v2.contract import (LiveV2Error, NativeIdentity, RenderAssignment, ZERO_DIGEST,
                                        closed_pages, digest, parse_lease)
from remote_ra_live_v2.composition import EngineBridge, EngineWorkerProxy, LiveEndpoint
from remote_ra_live_v2.engine_boundary import EngineRpcBoundary
from remote_ra_live_v2.phase1_coordinator import Phase1Coordinator
from remote_ra_live_v2.pool_incarnations import install
from remote_ra_live_v2.private_endpoint import SplitPrivateEndpoint
from remote_ra_live_v2.session_authority import LiveSessionAuthority
from remote_ra_native_growing.adapter import EngineCoreAdapter
from remote_ra_native_growing.contract import Abort, Authority, digest as native_digest

P = 16
H = lambda c: c * 64
SOURCE = H('c')
ATT, MAMBA = (0,), (1, 2, 3)
HEADER, BOUNDARY = (7001, 7002, 7003), (7101, 7102)
GOAL_TEXT = '<retrieval_query>\nfix the bug\n</retrieval_query>'
GOAL_TOKENS = tuple(range(8000, 8020))  # 20 tokens: two Goal Pages
QUERY = {'goal_query', 'observation_query'}


def vec(page):
    return [1.0, (page % 5 + 1) / 8.0]


class Block:
    def __init__(self, block_id):
        self.block_id, self.ref_cnt, self.block_hash, self.is_null = block_id, 0, None, block_id == 0

    def __repr__(self):
        return 'B%d:%d' % (self.block_id, self.ref_cnt)


class FakePool:
    """BlockPool semantics; freed blocks are reused first, like prepend_n."""
    def __init__(self, count):
        self.blocks = [Block(i) for i in range(count)]
        self.null_block = self.blocks[0]
        self.free = deque(self.blocks[1:])

    def get_num_free_blocks(self):
        return len(self.free)

    def get_new_blocks(self, n):
        if n > len(self.free):
            raise ValueError('Cannot get %d free blocks from the pool' % n)
        out = [self.free.popleft() for _ in range(n)]
        for b in out:
            assert b.ref_cnt == 0
            b.ref_cnt += 1
        return out

    def touch(self, blocks):
        for b in blocks:
            if b.ref_cnt == 0 and not b.is_null:
                self.free.remove(b)
            b.ref_cnt += 1

    def free_blocks(self, ordered):
        for b in ordered:
            b.ref_cnt -= 1
            assert b.ref_cnt >= 0, 'double free'
            if b.ref_cnt == 0 and not b.is_null:
                self.free.appendleft(b)


class FullAttentionManager:
    def __init__(self, pool):
        self.block_pool, self.block_size = pool, P
        self.req_to_blocks, self.num_cached_block, self.new_block_ids = {}, {}, []


class MambaManager:
    def __init__(self, pool):
        self.block_pool, self.block_size, self.req_to_blocks = pool, P, {}


def spec(name):
    return type(name, (), {'block_size': P})()


class KVManager:
    def __init__(self, pool):
        self.block_pool = pool
        self.fa = FullAttentionManager(pool)
        self.mamba = [MambaManager(pool) for _ in MAMBA]
        self.coordinator = NS(single_type_managers=[self.fa, *self.mamba])
        self.kv_cache_config = NS(kv_cache_groups=[NS(kv_cache_spec=spec('FullAttentionSpec'))] +
                                  [NS(kv_cache_spec=spec('MambaSpec')) for _ in MAMBA])
        self.calls = []

    def _growing_original_allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0,
            new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
            delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=False,
            reserved_blocks=0, has_scheduled_reqs=True, external_computed_tokens_by_group=None,
            computed_tokens_by_group=None, skip_cache=False):
        rid, cursors = request.request_id, tuple(computed_tokens_by_group)
        self.calls.append(dict(start=request.num_computed_tokens, count=num_new_tokens,
                               cursors=cursors, skip_cache=skip_cache,
                               floor=hasattr(self, '_remote_ra_chunk_capacity')))
        need = [-(-(c + num_new_tokens) // P) for c in cursors]
        row = self.fa.req_to_blocks.setdefault(rid, [])
        fresh = sum(1 for m in self.mamba if not any(not b.is_null for b in m.req_to_blocks.get(rid, ())))
        if max(0, need[0] - len(row)) + fresh > self.block_pool.get_num_free_blocks():
            return None
        while len(row) < need[0]:
            row.extend(self.block_pool.get_new_blocks(1))
        for g, m in zip(MAMBA, self.mamba):
            live = [b for b in m.req_to_blocks.get(rid, ()) if not b.is_null] or self.block_pool.get_new_blocks(1)
            m.req_to_blocks[rid] = [self.block_pool.null_block] * (need[g] - 1) + live
        return NS(blocks=())

    def native_free(self, rid):
        for m in self.coordinator.single_type_managers:
            blocks = m.req_to_blocks.pop(rid, [])
            self.block_pool.free_blocks(reversed([b for b in blocks if not b.is_null]))


class Req:
    def __init__(self, rid, tokens):
        self.request_id, self.prompt_token_ids, self.output_token_ids = rid, list(tokens), []
        self.num_computed_tokens, self.status = 0, NS(name='WAITING')
        self.num_preemptions = self.num_output_placeholders = 0
        self.spec_token_ids = []

    @property
    def num_tokens(self):
        return len(self.prompt_token_ids) + len(self.output_token_ids)


class FakeWorker:
    """Spec 2 ops with synthetic receipts; keeps the worker-side session mirrors."""
    def __init__(self):
        self.sessions, self.session_ns, self.records = {}, {}, {}
        self.generation, self.log, self.fail, self.leases, self.drained = 0, [], {}, [], set()

    def __call__(self, operation, payload):
        self.log.append(operation)
        if operation in self.fail:
            raise RuntimeError(self.fail.pop(operation))
        return getattr(self, operation)(payload)

    def prepare(self, p):
        ident = NativeIdentity(**p['identity'])
        a = RenderAssignment(**dict(p['assignment'], identity=ident))
        assert p['request'] not in self.records and ident.request == p['request']
        assert set(p) == {'request', 'session', 'turn', 'local_turn', 'generation', 'binding',
                          'namespace', 'ns_generation', 'identity', 'assignment', 'tokens'}
        if p['local_turn'] == 0:
            assert self.session_ns.get(p['session']) is None and a.local_turn == 0
        else:
            ns = self.session_ns[p['session']]
            w = self.sessions[ns]
            assert ns == p['namespace'] and not w['poisoned']
            assert (w['computed_end'], w['sealed_end'], w['carry']) == (a.computed_start, a.sealed_start, a.carry_digest)
            assert (w['tail'] is not None) == a.tail
        self.generation += 1
        self.records[p['request']] = dict(binding=p['binding'], a=a, tokens=list(p['tokens']),
            ns=p['namespace'], session=p['session'], generation=self.generation)
        return dict(operation='prepare', binding=p['binding'], drained=True,
                    allocation_generation=self.generation)

    def phase1(self, p):
        rec = self.records[p['request']]
        a = rec['a']
        lease = parse_lease(p['lease'], attention_groups=ATT)
        self.leases.append(p['lease'])
        assert p['lease_digest'] == lease.lease_digest
        assert p['request_digest'] == digest({k: v for k, v in p.items() if k != 'request_digest'})
        assert (lease.start, lease.end, lease.sealed_start, lease.candidate_end) == (
            a.computed_start, a.observation_end, a.sealed_start, a.candidate_end)
        assert p['tokens'] == rec['tokens'][:a.observation_end]
        assert p['allocation_generation'] == rec['generation'] == lease.generation
        w = self.sessions[rec['ns']]
        borrowed = [w['resident_blocks'][pg][0][0] for pg in lease.resident_pages]
        borrowed += [w['tail']['block_ids'][0][0]] if w['tail'] else []
        assert list(lease.borrowed) == borrowed and list(lease.resident_pages) == w['resident']
        assert lease.carry_digest == w['carry']
        C, e, goal = lease.start, lease.end, lease.goal
        rec['phase1'] = lease
        body = dict(operation='phase1', binding=p['binding'], request_digest=p['request_digest'],
            lease_digest=p['lease_digest'], allocation_generation=p['allocation_generation'],
            start=C, end=e, forward_tokens=e - C,
            page_vectors={str(pg): vec(pg) for pg in closed_pages(C, e, P)},
            observation_vector=[1.0, 0.25], goal_vector=[0.5, 1.0] if goal else None,
            goal_token_count=len(goal.tokens) if goal else None,
            goal_tokens_sha256=digest(list(goal.tokens)) if goal else None,
            state_digest=digest(['state', e]),
            tail_capture_digest=digest(['tail', rec['ns'], C]) if lease.tail_candidate else None,
            carry_restored=dict(start=C, digest=w['carry']),
            position_evidence=[dict(start=C, count=e - C)], vector_space=dict(decoder_layers=64),
            representation='pre_final_norm_last_token_l2', drained=True)
        return dict(body, worker_receipt=dict(body=digest(body), mac=H('f')))

    def phase1_drain(self, p):
        lease = p['phase1_request']['lease']
        assert lease['lease_id'] == p['lease_id'] and digest(lease) == p['lease_digest']
        self.drained.add(p['lease_id'])
        return dict(operation='phase1_drain', binding=p['binding'], lease_id=p['lease_id'],
                    lease_digest=p['lease_digest'], drained=True, worker_receipt=dict(mac=H('f')))

    def reserve(self, p):
        rec = self.records[p['request']]
        a, plan = rec['a'], p['plan']
        assert 'plan' not in rec and plan['namespace'] == rec['ns']
        w = self.sessions.get(rec['ns'])
        target = {int(k): v for k, v in p['target'].items()}
        handover = {int(k): v for k, v in p['handover'].items()}
        assert set(target) == set(plan['selected']) and set(handover) == set(plan['run_pages'])
        for pg in plan['retain']:
            tail_page = a.tail_candidate and pg == a.sealed_start // P
            assert target[pg] == (w['tail']['block_ids'] if tail_page else w['resident_blocks'][pg])
        assert all(pg in w['catalog'] for pg in plan['restore'])
        for pg, ids in handover.items():
            live = a.tail_live and pg == a.sealed_start // P
            fresh = rec['phase1'].run_fresh.get(pg, ()) if 'phase1' in rec else ()
            assert ids == (w['tail']['block_ids'] if live else [list(g) for g in fresh])
        rec.update(plan=plan, target=target)
        return dict(operation='reserve', binding=p['binding'], drained=True)

    def allocation_update(self, p):
        rec = self.records[p['request']]
        prior = rec.get('interval')
        assert (p['start'] == rec['a'].phase2_start) if prior is None else (sum(prior) == p['start'])
        rec['interval'], rec['rows'] = (p['start'], p['count']), p['rows']
        return dict(operation='allocation_update', binding=p['binding'], drained=True)

    def complete(self, p):
        rec = self.records.pop(p['request'])
        a, plan = rec['a'], rec['plan']
        out = list(p['output_token_ids'])
        C2 = a.prompt_tokens + len(out) - 1
        Q2 = C2 // P * P
        assert (p['computed_end'], p['sealed_end']) == (C2, Q2) and sum(rec['interval']) == C2
        full, ns = rec['tokens'] + out, rec['ns']
        row = [b for b, _ in rec['rows'][ATT[0]]]
        a_count, base = len(plan['selected']), a.candidate_end // P
        block = lambda pg: [[row[a_count + pg - base]]]
        w = self.sessions.setdefault(ns, dict(catalog={}, resident=[], resident_blocks={}, tail=None,
            computed_end=0, sealed_end=0, carry=ZERO_DIGEST, poisoned=False))
        new = set(range(base, Q2 // P)) | ({a.sealed_start // P} if a.tail_candidate else set())
        new_pages = {str(pg): dict(token_sha256=native_digest(tuple(full[pg*P:(pg+1)*P])),
                                   namespace=ns, binding=p['binding'], source=SOURCE,
                                   content=digest(['page', ns, pg])) for pg in sorted(new)}
        resident = sorted(set(plan['selected']) | set(range(base, Q2 // P)))
        blocks = {pg: rec['target'][pg] if pg in rec['target'] else block(pg) for pg in resident}
        tail = dict(page=Q2 // P, start=Q2, valid_end=C2, block_ids=block(Q2 // P)) if C2 > Q2 else None
        carry = digest(['carry', ns, C2])
        w.update(resident=resident, resident_blocks=blocks, tail=tail, computed_end=C2,
                 sealed_end=Q2, carry=carry)
        w['catalog'].update({pg: new_pages[str(pg)]['content'] for pg in new})
        self.session_ns[rec['session']] = ns
        return dict(operation='complete', binding=p['binding'], drained=True, computed_end=C2,
            sealed_end=Q2, new_pages=new_pages,
            page_vectors={str(pg): vec(pg) for pg in closed_pages(a.phase2_start, C2, P)},
            resident=resident, resident_blocks={str(k): v for k, v in blocks.items()}, tail=tail,
            carry=dict(computed_end=C2, digest=carry), boundary=carry, dram=sorted(w['catalog']),
            release=[], publication=None)

    def cancel(self, p):
        rec = self.records.pop(p['request'], None)
        w = self.sessions.get(rec['ns']) if rec else None
        if w is not None:
            w['poisoned'] = True
        return dict(operation='cancel', binding=p['binding'], drained=True, poisoned=True,
            boundary=w['carry'] if w else ZERO_DIGEST, resident=list(w['resident']) if w else [],
            dram=sorted(w['catalog']) if w else [], tail=w['tail'] if w else None)

    def close_session(self, p):
        assert not any(r['session'] == p['session'] for r in self.records.values())
        assert self.session_ns.get(p['session']) in (None, p['namespace'])
        self.sessions.pop(p['namespace'], None)
        self.session_ns.pop(p['session'], None)
        return dict(operation='close_session', closed=True, drained=True, namespace=p['namespace'],
                    reason=p['reason'])


class FakeSelection:
    """Spec 4.2 selection API stand-in: A_t from ``choose(n, local_turn)``, signed receipts."""
    def __init__(self, adapter, choose=None, goal=False):
        self.adapter, self.goal = adapter, goal
        self.choose = choose or (lambda n, local_turn: range(n))
        self.catalog, self.authorized, self.used, self.pending, self.resets = {}, {}, set(), {}, []
        self.text_sha256 = sha256(GOAL_TEXT.encode()).hexdigest()

    def begin_turn(self, binding, geo, conversation):
        q = NS(goal_payload=GOAL_TEXT if self.goal and geo.local_turn else None,
               text_sha256=self.text_sha256, cache_hit=False)
        self.pending[binding.request] = (geo, q)
        return q

    def plan_turn(self, binding, geo, allocation_generation, query, phase1):
        assert self.pending.pop(binding.request) == (geo, query)
        assert (phase1 is None) == (geo.local_turn == 0)
        assert set(self.catalog.get(binding.session, {})) == set(range(geo.Q // P))
        if phase1 is not None:
            assert set(phase1.page_vectors) == set(closed_pages(geo.C, geo.e, P))
            assert (phase1.goal_vector is not None) == (query.goal_payload is not None)
            if query.goal_payload is not None:
                assert phase1.goal_payload_sha256 == self.text_sha256
        chosen = tuple(sorted(self.choose(geo.n, geo.local_turn)))
        ranking = chosen + tuple(p for p in range(geo.n) if p not in chosen)
        body = dict(binding=binding.identity, ranking=list(ranking), selected_count=len(chosen),
                    local_turn=geo.local_turn)
        receipt = dict(body=body, mac=self.adapter.authority.sign(body))
        self.authorized[binding.request] = receipt
        return ranking, len(chosen), receipt

    def verify_reservation(self, binding, ranking, selected_count, receipt):
        self.adapter.authority.verify(receipt['body'], receipt['mac'])
        assert self.authorized[binding.request] == receipt and binding.request not in self.used
        self.used.add(binding.request)

    def publish_completion(self, binding, geo, phase1_vectors, completion_vectors, sealed_end, computed_end):
        catalog = self.catalog.setdefault(binding.session, {})
        assert not set(phase1_vectors) & set(completion_vectors)
        catalog.update(phase1_vectors)
        catalog.update(completion_vectors)
        assert set(catalog) == set(range(sealed_end // P))

    def reset_session(self, session_id):
        self.resets.append(session_id)
        self.catalog.pop(session_id, None)


class Renderer:
    def __call__(self, source):
        assert source['text'] == GOAL_TEXT
        return NS(token_ids=list(GOAL_TOKENS))


class Harness:
    def __init__(self, policy='full', *, choose=None, goal=False, blocks=160, max_tokens=4096):
        self.policy = policy
        self.pool = FakePool(blocks)
        install(self.pool)
        self.kv = KVManager(self.pool)
        self.authority = Authority(epoch=H('e'), loaded_source=SOURCE)
        self.adapter = EngineCoreAdapter(self.authority, self.kv, page_tokens=P,
                                         full_attention_managers=(self.kv.fa,))
        self.worker = FakeWorker()
        executor = NS(collective_rpc=lambda method, args, timeout: [self.worker(*args)])
        self.proxy = EngineWorkerProxy(executor, dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS',
                                                      loaded_sources_verified=True))
        self.boundary = EngineRpcBoundary(self.proxy, self.adapter)
        self.adapter.backend = self.boundary
        sessions = LiveSessionAuthority.__new__(LiveSessionAuthority)
        sessions.adapter, sessions.worker, sessions.session, sessions.epoch = self.adapter, self.proxy, 's', H('e')
        sessions.closing = sessions.closed = False
        sessions.namespace = self.adapter.open('s', policy, 'm', 'r')
        self.adapter._live_session_authority = sessions
        self.sessions = sessions
        core = NS(_remote_ra_growing_adapter=self.adapter, _remote_ra_live_v2_boundary=self.boundary,
                  _remote_ra_live_v2_bridge=EngineBridge(), abort_requests=self.abort,
                  scheduler=NS(connector=NS(has_pending_push_work=lambda: False),
                               max_num_scheduled_tokens=max_tokens,
                               kv_cache_manager=NS(free=self.free)))
        endpoint = SplitPrivateEndpoint.__new__(SplitPrivateEndpoint)
        endpoint.core, endpoint.adapter, endpoint.sessions = core, self.adapter, sessions
        endpoint.assembly = NS(renderer=Renderer(), provider=NS(provenance=NS(page_tokens=P)))
        endpoint.owner = NS(identity=NS(manifest=SOURCE))
        endpoint.claims, endpoint.transcript, endpoint.protocol = {}, None, None
        endpoint.phase1 = Phase1Coordinator(core, self.proxy, endpoint.assembly.renderer)
        endpoint.selection = FakeSelection(self.adapter, choose, goal)
        self.boundary.selection = endpoint.selection
        core._remote_ra_private_v2_endpoint = endpoint
        self.core, self.endpoint, self.selection = core, endpoint, endpoint.selection
        self.live = LiveEndpoint(core, endpoint.assembly, self.proxy)
        self.baseline = self.pool.get_num_free_blocks()
        self.ordinal, self.stream, self.requests, self.enqueued = 0, None, {}, set()

    @property
    def session(self):
        return self.adapter.sessions['s']

    # ---- one request -------------------------------------------------------
    def render(self, observation=(), rebuild=False):
        s = self.session
        if self.stream is None or rebuild:
            complete = tuple(range(300 if rebuild else 100, 340 if rebuild else 140))
            tokens, stable, span, local = complete + HEADER, len(complete), None, 0
        else:
            stable = len(self.stream)
            tokens = self.stream + tuple(observation) + HEADER
            span, local = (stable, stable + len(observation)), s.local_turn
        rid = 'r%d' % self.ordinal
        request = Req(rid, tokens)
        binding, mac = self.authority.render(request, session='s', turn=self.ordinal, local_turn=local,
            rebuild=rebuild, generation='g%d' % self.ordinal, model='m', renderer='r', tokens=tokens,
            stable_prefix_tokens=tokens[:stable], observation_span=span, native_prefix='p')
        if rebuild:
            self.endpoint.rebuild(binding)
        self.endpoint.claims[rid] = dict(request=request, binding=binding, mac=mac, conversation=[])
        self.requests[rid] = request
        return request, binding

    def admit(self, request, binding):
        self.kv.fa.new_block_ids.clear()  # the scheduler takes the zero queue every step
        self.live.reserve_worker(request, binding)
        self.enqueued.add(binding.request)  # LiveEndpoint.enqueue -> core.add_request
        return self.boundary.prepared[binding.request]['plan']

    def step(self, request, count):
        full = request.status.name == 'WAITING'
        result = self.adapter.allocate_slots(self.kv, request, count, num_new_computed_tokens=0,
            new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
            delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=full,
            reserved_blocks=0, has_scheduled_reqs=False)
        assert result is not None
        request.num_computed_tokens += count
        request.status.name = 'RUNNING'

    def generate(self, request, outputs):
        N = len(request.prompt_token_ids)
        while request.num_computed_tokens < N:  # Mamba-aligned prefill chunks
            start = request.num_computed_tokens
            self.step(request, min(N, (start // P + 1) * P) - start)
        request.output_token_ids.append(outputs[0])
        for token in outputs[1:]:
            self.step(request, 1)
            request.output_token_ids.append(token)

    def free(self, request):
        self.adapter.before_free(request)
        self.kv.native_free(request.request_id)
        self.adapter.after_free(request)

    def abort(self, request_ids):
        # Scheduler abort frees only requests it holds.
        for rid in request_ids:
            if rid in self.enqueued:
                self.enqueued.discard(rid)
                self.free(self.requests[rid])

    def finish(self, request, binding):
        receipt = self.live.drain(binding, 'complete')
        LiveEndpoint.verify_drain(binding, 'complete', receipt)
        self.live.after_drain(binding, 'complete')
        self.free(request)
        self.stream = tuple(request.prompt_token_ids) + tuple(request.output_token_ids) + BOUNDARY
        self.ordinal += 1
        return receipt

    def cancel(self, binding):
        receipt = self.live.drain(binding, 'cancel')
        LiveEndpoint.verify_drain(binding, 'cancel', receipt)
        self.live.after_drain(binding, 'cancel')
        return receipt

    def turn(self, observation, outputs, rebuild=False):
        request, binding = self.render(observation, rebuild)
        plan = self.admit(request, binding)
        geo = self.endpoint.claims[binding.request]['geometry']
        self.generate(request, outputs)
        self.finish(request, binding)
        return geo, plan

    def close(self):
        self.sessions.close()

    # ---- invariants ----------------------------------------------------------
    def check_boundary(self, test):
        """#30 and spec 3.7 at a turn boundary."""
        s, owner = self.session, self.adapter.owner
        ns, Q = s.namespace, s.sealed_end
        pins = owner.pinned(ns)
        test.assertEqual(tuple(sorted(owner.sessions.get(ns, {}))), s.resident)
        test.assertEqual(self.adapter.tail_mirror(owner.tails.get(ns)), s.tail)
        test.assertEqual(set(s.catalog), set(range(Q // P)))
        if self.policy in QUERY:
            test.assertEqual(set(self.selection.catalog.get('s', {})), set(range(Q // P)))
        w = self.worker.sessions.get(ns)
        if w is not None:
            test.assertEqual((w['resident'], w['tail'], set(w['catalog']), w['carry']),
                             (list(s.resident), s.tail, set(s.catalog), s.boundary))
        test.assertTrue(all(b.ref_cnt == 1 for b in pins))
        test.assertEqual(self.pool.get_num_free_blocks(), self.baseline - len(pins))
        test.assertEqual(len(pins), len(s.resident) + (s.tail is not None))
        test.assertTrue(s.pending is None and not self.adapter.requests and
                        not self.endpoint.phase1.records and not owner.leases)


def outputs(count, base=5000):
    return tuple(range(base, base + count))


def obs(count, base=6000):
    return tuple(range(base, base + count))


# Full-policy script (P = 16, 40 template tokens, header 3, boundary 2):
# t0 C'=48 (C == Q); t1 C'=85 (tail live); t2 C'=124 (tail live, short next);
# t3 C'=141 (tail candidate next); t4 C'=161.
SCRIPT = [((), 6), (obs(20), 12), (obs(30), 4), (obs(1), 11), (obs(10), 5)]


class GeometryScriptTests(unittest.TestCase):
    def test_five_turn_script_covers_every_tail_case(self):
        h = Harness('full')
        seen = []
        for number, (observation, count) in enumerate(SCRIPT):
            geo, plan = h.turn(observation, outputs(count, 5000 + 100 * number))
            seen.append((geo.local_turn, geo.C, geo.Q, geo.o, geo.e, geo.n, geo.L, geo.tail,
                         geo.tail_candidate, geo.tail_live, tuple(geo.fresh_pages)))
            h.check_boundary(self)
            self.assertEqual(plan['selected'], list(range(geo.n)))            # Full
            self.assertEqual(plan['sparse']['dropped_full_attention_tokens'], 0)
        self.assertEqual(seen, [
            (0, 0, 0, 40, 40, 0, 0, False, False, False, ()),
            (1, 48, 48, 51, 71, 3, 48, False, False, False, (3, 4)),       # C == Q
            (2, 85, 80, 88, 118, 5, 80, True, False, True, (6, 7)),        # tail live
            (3, 124, 112, 127, 128, 7, 112, True, False, True, ()),        # short observation
            (4, 141, 128, 144, 154, 9, 144, True, True, False, (9,)),      # tail candidate
        ])
        s = h.session
        self.assertEqual((s.turn, s.local_turn, s.computed_end, s.sealed_end), (5, 5, 161, 160))
        self.assertEqual(s.next_token, 5404)
        self.assertEqual(s.resident, tuple(range(10)))
        self.assertEqual((s.tail['page'], s.tail['start'], s.tail['valid_end']), (10, 160, 161))
        h.close()
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        self.assertNotIn('s', h.adapter.sessions)

    def test_selective_script_plan_sets_and_refcounts(self):
        # observation_query A_t per local turn: below threshold, first selective
        # turn (restore empty), tail candidate unselected plus a restore.
        chosen = {1: (0, 1, 2), 2: (0, 1, 2, 3, 4), 3: (0, 5, 6), 4: (1, 7)}
        h = Harness('observation_query', choose=lambda n, t: chosen.get(t, range(n)))
        for number, (observation, count) in enumerate(SCRIPT[:3]):
            geo, plan = h.turn(observation, outputs(count, 5000 + 100 * number))
            h.check_boundary(self)
        self.assertEqual((plan['retain'], plan['restore'], plan['release']), ([0, 1, 2, 3, 4], [], []))
        # Local turn 3: first selective turn, tail live; R' = R.
        request, binding = h.render(SCRIPT[3][0])
        plan = h.admit(request, binding)
        self.assertEqual((plan['resident_before'], plan['retain'], plan['restore'], plan['release']),
                         ([0, 1, 2, 3, 4, 5, 6], [0, 5, 6], [], [1, 2, 3, 4]))
        self.assertEqual((plan['run_pages'], plan['run_tail_page'], plan['tail_live']), ([7], 7, True))
        self.assertEqual(plan['sparse'], dict(logical_matched_tokens=112, dropped_full_attention_tokens=64))
        owner, ns = h.adapter.owner, h.session.namespace
        tail = owner.tails[ns].groups[0][0]
        row = h.kv.fa.req_to_blocks[binding.request]
        self.assertEqual(row, [owner.sessions[ns][p][0][0] for p in (0, 5, 6)] + [tail])
        self.assertEqual([b.ref_cnt for b in row], [2, 2, 2, 2])            # S + Q, live tail too
        self.assertEqual({owner.sessions[ns][p][0][0].ref_cnt for p in (1, 2, 3, 4)}, {1})
        self.assertEqual(h.kv.fa.new_block_ids, [])                         # nothing to zero
        h.generate(request, outputs(11, 5300))
        h.finish(request, binding)
        h.check_boundary(self)
        self.assertEqual(h.session.resident, (0, 5, 6, 7))
        # Local turn 4: tail candidate (page 8) not selected; page 1 restored.
        request, binding = h.render(SCRIPT[4][0])
        before = h.pool.get_num_free_blocks()
        plan = h.admit(request, binding)
        self.assertEqual((plan['tail_candidate'], plan['resident_before']), (True, [0, 5, 6, 7]))
        self.assertEqual((plan['retain'], plan['restore'], plan['release']), ([7], [1], [0, 5, 6, 8]))
        self.assertEqual(plan['seal_at_completion']['phase1_sealed'], [8])
        tail = owner.tails[ns].groups[0][0]
        restored = h.kv.fa.req_to_blocks[binding.request][0]
        self.assertEqual(h.kv.fa.new_block_ids, [restored.block_id])        # restore targets only
        self.assertEqual((restored.ref_cnt, tail.ref_cnt), (1, 1))         # tail untouched, not in row
        self.assertNotIn(tail, h.kv.fa.req_to_blocks[binding.request])
        # Held now: restore target + one adopted fresh FA block (Mamba freed at drain).
        self.assertEqual(h.pool.get_num_free_blocks(), before - 2)
        h.generate(request, outputs(5, 5400))
        receipt = h.finish(request, binding)
        self.assertIn('8', receipt['new_pages'])                            # phase-1 tail capture
        self.assertEqual(tail.ref_cnt, 0)                                  # released at commit
        self.assertEqual(h.session.resident, (1, 7, 9))
        h.check_boundary(self)
        h.close()
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)


class Phase1LeaseTests(unittest.TestCase):
    def run_to(self, h, turns):
        for number, (observation, count) in enumerate(SCRIPT[:turns]):
            h.turn(observation, outputs(count, 5000 + 100 * number))

    def test_lease_rows_for_tail_cases(self):
        h = Harness('full')
        expected = {1: False, 2: True, 3: True, 4: True}
        self.run_to(h, 1)
        for turns in range(1, 5):
            request, binding = h.render(SCRIPT[turns][0])
            s, owner = h.session, h.adapter.owner
            R = sorted(owner.sessions[s.namespace])
            pin = owner.tails.get(s.namespace)
            h.admit(request, binding)
            raw = h.worker.leases[-1]
            lease = parse_lease(raw, attention_groups=ATT)
            h.authority.verify({k: v for k, v in raw.items() if k != 'mac'}, raw['mac'])
            geo = h.endpoint.claims[binding.request]['geometry']
            fresh = [raw['run_fresh'][str(p)][0][0] for p in geo.fresh_pages]
            head = [owner.sessions[s.namespace][p][0][0].block_id for p in R]
            self.assertEqual(lease.rows[0], tuple(head + ([pin.groups[0][0].block_id] if pin else []) + fresh))
            self.assertEqual((pin is not None), expected[turns])
            M = -(-geo.e // P)
            for g in MAMBA:
                self.assertEqual(lease.rows[g][:-1], (0,) * (M - 1))
            self.assertFalse(set(lease.zero) & set(lease.borrowed))
            self.assertEqual(dict(lease.sparse), dict(logical_matched_tokens=geo.Q,
                                                     dropped_full_attention_tokens=geo.Q - len(R) * P))
            self.assertIn(lease.lease_id, h.worker.drained)
            h.generate(request, outputs(SCRIPT[turns][1], 5000 + 100 * turns))
            h.finish(request, binding)
            h.check_boundary(self)

    def test_phase1_too_long_refuses_before_allocation(self):
        h = Harness('full', max_tokens=16)
        self.run_to(h, 1)
        request, binding = h.render(obs(20))       # e - C = 23 > 16
        pinned = h.pool.get_num_free_blocks()
        with self.assertRaisesRegex(LiveV2Error, 'PHASE1_TOO_LONG'):
            h.admit(request, binding)
        self.assertTrue(h.session.poisoned)
        self.assertEqual(h.pool.get_num_free_blocks(), pinned)
        self.assertEqual(h.worker.log[-2:], ['prepare', 'cancel'])
        h.close()
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)

    def test_goal_lease_uses_own_blocks_and_frames(self):
        h = Harness('goal_query', goal=True)
        self.run_to(h, 1)
        request, binding = h.render(obs(20))
        before = h.pool.get_num_free_blocks()
        h.admit(request, binding)
        raw = h.worker.leases[-1]
        lease = parse_lease(raw, attention_groups=ATT)
        self.assertEqual(lease.goal.tokens, GOAL_TOKENS)
        self.assertEqual(len(lease.goal.rows[0]), 2)                     # ceil(20/16) Goal Pages
        self.assertFalse(set(lease.goal.owned) & (set(lease.owned) | set(lease.borrowed)))
        # Goal and Mamba blocks are back in the pool; only the adopted run blocks remain.
        self.assertEqual(h.pool.get_num_free_blocks(), before - len(lease.run_fresh))
        frames = h.proxy.http_receipts[binding.request]
        self.assertEqual([f['operation'] for f in frames], ['prepare', 'phase1', 'phase1_drain', 'reserve'])
        from remote_ra_http.config import digest as http_digest
        self.assertEqual(frames[1]['query_token_sha256'], http_digest(list(GOAL_TOKENS)))
        h.generate(request, outputs(12, 5100))
        h.finish(request, binding)
        h.check_boundary(self)

    def test_observation_query_frame_digest(self):
        h = Harness('observation_query')
        self.run_to(h, 1)
        request, binding = h.render(obs(20))
        h.admit(request, binding)
        from remote_ra_http.config import digest as http_digest
        geo = h.endpoint.claims[binding.request]['geometry']
        frame = h.proxy.http_receipts[binding.request][1]
        self.assertEqual(frame['query_token_sha256'], http_digest(list(binding.tokens[geo.o:geo.e])))
        self.assertIsNone(h.worker.leases[-1]['goal'])


class DispatchTests(unittest.TestCase):
    def test_phase2_cursors_and_waiting_start(self):
        h = Harness('observation_query', choose=lambda n, t: (1,) if t else range(n))
        h.turn((), outputs(6))
        request, binding = h.render(obs(20))
        h.admit(request, binding)
        geo = h.endpoint.claims[binding.request]['geometry']
        self.assertEqual(request.num_computed_tokens, geo.e)
        h.step(request, 1)
        call = h.kv.calls[-1]
        # A_t = [1], L = 48: FA physical = |A_t|P + (e - L); Mamba logical e.
        self.assertEqual(call['cursors'], (P + geo.e - geo.L, geo.e, geo.e, geo.e))
        self.assertEqual((call['skip_cache'], call['floor']), (True, False))
        row = [b.block_id for b in h.kv.fa.req_to_blocks[binding.request]]
        plan = h.boundary.prepared[binding.request]
        self.assertEqual(row[:2], [plan['target'][1][0][0].block_id, plan['run'][3][0][0].block_id])
        self.assertEqual(h.worker.records[binding.request]['interval'], (geo.e, 1))

    def test_waiting_request_must_start_at_e(self):
        h = Harness('full')
        h.turn((), outputs(6))
        request, binding = h.render(obs(20))
        h.admit(request, binding)
        request.num_computed_tokens += 1   # WAITING must start exactly at e
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(LiveV2Error, 'phase-2 start'):
            h.step(request, 1)
        self.assertTrue(h.session.poisoned)
        h.cancel(binding)
        pins = h.adapter.owner.pinned(h.session.namespace)
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline - len(pins))
        h.close()
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)


class CompletionTests(unittest.TestCase):
    def test_completion_commit_and_token_digests(self):
        h = Harness('full')
        h.turn((), outputs(6))
        request, binding = h.render(obs(20))
        h.admit(request, binding)
        h.generate(request, outputs(12, 5100))
        receipt = h.finish(request, binding)
        full = tuple(request.prompt_token_ids) + tuple(request.output_token_ids)
        self.assertEqual((receipt['computed_end'], receipt['sealed_end']), (85, 80))
        self.assertEqual(sorted(map(int, receipt['new_pages'])), [3, 4])
        for page, record in receipt['new_pages'].items():
            p = int(page)
            self.assertEqual(record['token_sha256'], native_digest(full[p*P:(p+1)*P]))
        s = h.session
        self.assertEqual((s.computed_tokens, s.sealed_tokens, s.next_token), (full[:85], full[:80], full[85]))
        self.assertEqual(s.boundary, receipt['carry']['digest'])
        h.check_boundary(self)

    def test_live_tail_continues_on_the_same_block(self):
        h = Harness('full')
        h.turn((), outputs(6))
        h.turn(obs(20), outputs(12, 5100))                   # tail [80, 85) on page 5
        tail = h.adapter.owner.tails[h.session.namespace].groups[0][0]
        geo, plan = h.turn(obs(1), outputs(1, 5200))         # C' = 92 < Q + P
        self.assertTrue(geo.tail_live)
        s = h.session
        self.assertEqual((s.computed_end, s.sealed_end, s.tail['page'], s.tail['valid_end']), (92, 80, 5, 92))
        self.assertIs(h.adapter.owner.tails[s.namespace].groups[0][0], tail)
        self.assertEqual(tail.ref_cnt, 1)
        h.check_boundary(self)

    def test_foreign_completion_receipt_poisons_without_commit(self):
        h = Harness('full')
        h.turn((), outputs(6))
        request, binding = h.render(obs(20))
        h.admit(request, binding)
        h.generate(request, outputs(12, 5100))
        original = h.worker.complete
        def tampered(p):
            r = original(p); r['resident'] = r['resident'][:-1]; return r
        h.worker.complete = tampered
        with self.assertRaisesRegex(Abort, 'working set'):
            h.live.drain(binding, 'complete')
        s = h.session
        self.assertTrue(s.poisoned)
        self.assertEqual((s.turn, s.local_turn, s.sealed_end), (1, 1, 48))


class CancelTests(unittest.TestCase):
    def test_cancel_before_adopt_frees_phase1_blocks(self):
        h = Harness('observation_query')
        h.turn((), outputs(6))
        h.turn(obs(20), outputs(12, 5100))
        request, binding = h.render(obs(30))
        pinned = h.pool.get_num_free_blocks()
        h.selection.plan_turn = lambda *a: (_ for _ in ()).throw(RuntimeError('selection failed'))
        with self.assertRaisesRegex(RuntimeError, 'selection failed'):
            h.admit(request, binding)
        self.assertTrue(h.session.poisoned)
        self.assertEqual(h.pool.get_num_free_blocks(), pinned)            # F blocks freed
        self.assertIn(h.worker.leases[-1]['lease_id'], h.worker.drained)
        self.assertEqual(h.worker.log[-1], 'cancel')
        self.assertFalse(h.endpoint.phase1.records)
        tail = h.adapter.owner.tails[h.session.namespace].groups[0][0]
        self.assertEqual(tail.ref_cnt, 1)
        h.close()
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)

    def test_cancel_after_admission_restores_refs(self):
        h = Harness('observation_query', choose=lambda n, t: (0, 4) if t == 2 else range(n))
        h.turn((), outputs(6))
        h.turn(obs(20), outputs(12, 5100))
        s = h.session
        committed = (s.resident, dict(s.tail), dict(s.catalog), s.boundary)
        request, binding = h.render(obs(30))
        h.admit(request, binding)
        h.step(request, 3)                                              # the header [e, N)
        request.output_token_ids.append(5500)
        h.step(request, 1)                                              # one decode step
        receipt = h.cancel(binding)
        self.assertTrue(receipt['poisoned'] and h.session.poisoned)
        self.assertEqual((s.resident, s.tail, s.catalog, s.boundary), committed)
        pins = h.adapter.owner.pinned(s.namespace)
        self.assertTrue(all(b.ref_cnt == 1 for b in pins))
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline - len(pins))
        h.close()
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        self.assertEqual(h.worker.log[-1], 'close_session')

    def test_reserve_rpc_failure_drains_through_adapter(self):
        h = Harness('full')
        h.turn((), outputs(6))
        request, binding = h.render(obs(20))
        h.worker.fail['reserve'] = 'worker reserve lost'
        with self.assertRaisesRegex(RuntimeError, 'worker reserve lost'):
            h.admit(request, binding)
        self.assertTrue(h.session.poisoned)
        # The lifecycle drain returns the prepared cancel; after_drain frees the
        # never-enqueued rows natively.
        h.cancel(binding)
        self.assertFalse(h.adapter.owns(binding.request))
        pins = h.adapter.owner.pinned(h.session.namespace)
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline - len(pins))
        h.close()
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)


class RebuildTests(unittest.TestCase):
    def test_rebuild_opens_fresh_namespace_generation(self):
        h = Harness('goal_query', goal=True)
        h.turn((), outputs(6))
        h.turn(obs(20), outputs(12, 5100))
        old = h.session.namespace
        geo, plan = h.turn((), outputs(9, 5200), rebuild=True)
        s = h.session
        self.assertEqual(h.selection.resets, ['s'])
        self.assertNotEqual(s.namespace, old)
        self.assertEqual((s.ns_generation, s.turn, s.local_turn, geo.local_turn), (1, 3, 1, 0))
        self.assertEqual(h.sessions.namespace, s.namespace)
        self.assertNotIn(old, h.adapter.owner.sessions)
        self.assertEqual(plan['namespace'], s.namespace)
        self.assertIn('close_session', h.worker.log)
        h.check_boundary(self)
        h.turn(obs(20), outputs(12, 5300))                          # local turn 1 after rebuild
        h.check_boundary(self)
        self.assertEqual((h.session.turn, h.session.local_turn), (4, 2))
        h.close()
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)

    def test_rebuild_refused_for_poisoned_session(self):
        h = Harness('full')
        h.turn((), outputs(6))
        h.session.poisoned = True
        with self.assertRaisesRegex(LiveV2Error, 'rebuild requires'):
            h.render((), rebuild=True)


class WorkerPayloadTests(unittest.TestCase):
    def test_prepare_payload_is_v2_assignment(self):
        h = Harness('full')
        h.turn((), outputs(6))
        request, binding = h.render(obs(20))
        payload = h.endpoint.worker_payload(binding)
        a = RenderAssignment(**dict(payload['assignment'], identity=NativeIdentity(**payload['identity'])))
        self.assertEqual((a.local_turn, a.computed_start, a.sealed_start, a.candidate_end,
                          a.observation_start, a.observation_end, a.phase2_start, a.prompt_tokens),
                         (1, 48, 48, 48, 51, 71, 71, 74))
        self.assertEqual(a.carry_digest, h.session.boundary)
        self.assertEqual((payload['local_turn'], payload['namespace'], payload['ns_generation']),
                         (1, h.session.namespace, 0))


if __name__ == '__main__':
    unittest.main()
