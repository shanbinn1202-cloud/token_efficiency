"""Token gather end to end (token-gather-1114). CPU only.

test_decouple_e2e's harness at (P, M) = (16, 784), whose engine, worker, Selector V2 and evidence writers are the
real ones, with a token gather pin in the selection contract: goal_query and observation_query, the markers_v1
picker (the default markers and a parameterised set with a quota) and the separators_v1 picker. The fake runner
checks what every forward read through its block table: each slot must hold the K/V of the logical token it
claims (store entries are (position, token)), the run must be [L, cursor), and here the prefix may hold single
tokens besides whole Pages (test_decouple_e2e's runner requires whole Pages only). The fake IO's gather_tokens
copies store slots from device blocks or DRAM payloads, as TensorIO.gather_tokens copies rows.

Checked on every turn: the receipt's gather against the native plan, the worker's reservation, the SparsePlan
(L - |A_t|P - G) and the evidence row; phase 2 attends exactly the selected Pages, the gathered tokens and the
run, and phase 1 exactly R, the previous turn's gathered tokens and its run; the gathered tokens are the picker's
picks outside the selected Pages plus the pad (real unattended tokens), in whole blocks; the shared budget
(selected Pages + gathered blocks <= capacity); the session's pins (resident Pages, tail, gathered blocks) on both
sides and in the FA pool; the gathered blocks hold their tokens' K/V. Then the rebuild, the cancel and the close
leave every block free.

Session pool (gather-receipt-1115): with --max-running-sessions the close receipt (and its sessions.jsonl row)
itemizes the gathered blocks a session pins at its close as ``released.gather_blocks``, so released.blocks ==
resident_pages + tail + gather_blocks: a session that gathers on its last turn, one that gathered earlier but not
on its last turn, ones that never gathered, and the same pool without token gather.
"""
from contextlib import ExitStack
import json
import unittest
from unittest.mock import patch

import test_decouple_e2e as D
import test_stageb_e2e as S
import test_stageb_engine as E
import test_stageb_worker as W
import test_mswea_protocol as X

from remote_ra_live_v2 import selection_contract
from remote_ra_live_v2 import token_gather as TG
from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner

try:
    import torch
except ImportError:
    torch = None

CASE = next(c for c in D.CASES if c.name == '16-784')


class GatherRunner(D.Runner):
    """D.Runner whose prefix may hold single tokens: every slot must still hold its token's K/V, no token twice."""

    def __init__(self, store, case):
        super().__init__(store, case)
        self.prefix_seen = {}

    def forward(self, rid, req, start, count, idx):
        P = W.P
        tokens = req.prompt_token_ids + req.output_token_ids
        plan = self._selective_load_plans.get(rid)
        Lm, dropped = (plan.logical_matched_tokens, plan.dropped_full_attention_tokens) if plan else (0, 0)
        assert start >= Lm, 'logical cursor precedes the matched boundary'
        self.plans[rid] = (Lm, dropped)
        state = self.chain(tokens[:start])
        if start or not self.decoupled:
            assert all(self.store.mamba.get((g, req.block_ids[g][idx])) == state for g in W.MAMBA), \
                f'GDN discontinuity: {rid} at {start}'
        row = req.block_ids[W.FA]
        phys = start - dropped
        seen = [self.store.kv.get((W.FA, row[j // P]), [None] * P)[j % P] for j in range(phys)]
        assert all(v is not None and tokens[v[0]] == v[1] for v in seen), f'{rid}: KV hole or stale KV before {start}'
        pos = [v[0] for v in seen]
        prefix = [p for p in pos if p < Lm]
        assert [p for p in pos if p >= Lm] == list(range(Lm, start)), f'{rid}: the run is not [L, cursor)'
        assert len(set(prefix)) == len(prefix), f'{rid}: a prefix token attended twice'
        if rid in self.prefix_seen:
            assert self.prefix_seen[rid] == sorted(prefix), f'{rid}: the prefix moved within the request'
        self.prefix_seen[rid] = sorted(prefix)
        self.visible[rid] = sorted({p // P for p in prefix})
        out = []
        for x in range(start, start + count):
            phys = x - dropped
            kv = self.store.kv.setdefault((W.FA, row[phys // P]), [None] * P)
            kv[phys % P] = (x, tokens[x])
            state = W.chain([tokens[x]], state, x)
            self.computed.append((rid, x))
            out.append([x + 1., tokens[x] + 1., 1.])
        for g in W.MAMBA:
            self.store.mamba[(g, req.block_ids[g][idx])] = state
        self._chains = (self._chains + [(tuple(tokens[:start + count]), state)])[-6:]
        return torch.tensor(out)


class GatherIO(D.IO):
    """D.IO plus the token gather entry point (store semantics): slot k of the gathered blocks gets the store
    slot of its source (a device block's, or a DRAM payload's single Page)."""

    def gather_tokens(self, blocks, sources):
        P = W.P
        assert blocks and len(sources) == len(blocks) * P
        targets = [groups[W.FA][0] for groups in blocks]
        assert len(set(targets)) == len(targets)
        for k, (kind, source, offset) in enumerate(sources):
            if kind == 'device':
                assert source[W.FA][0] not in targets
                value = self.store.read('attention', W.FA, source[W.FA][0])[offset]
            else:
                assert kind == 'host' and source['kind'] == 'attention' and len(source['blocks']) == 1
                value = source['blocks'][0][offset]
            assert value is not None, 'gathered an empty slot'
            self.store.kv.setdefault((W.FA, targets[k // P]), [None] * P)[k % P] = value
        self.log.append(('gather', 'attention', [(W.FA, b) for b in targets]))


def gather_contract(pin):
    def contract(case):
        value = D.contract.__wrapped__(case) if hasattr(D.contract, '__wrapped__') else ORIGINAL(case)
        value[TG.KEY] = pin
        return selection_contract.validate(value)
    return contract


ORIGINAL = D.contract


def marker_positions(stream, patterns):
    return {t for t in range(len(stream)) for ids in patterns if tuple(stream[t:t + len(ids)]) == ids}


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class TokenGatherEndToEndTests(unittest.TestCase):
    maxDiff = None

    def play(self, policy, pin):
        checked = []

        def check(h, info):
            self.check_turn(h, info, pin)
            checked.append(info['rid'])
        with ExitStack() as stack:
            stack.enter_context(patch.object(D, 'contract', gather_contract(pin)))
            stack.enter_context(patch.object(D, 'Runner', GatherRunner))
            stack.enter_context(patch.object(D, 'IO', GatherIO))
            stack.enter_context(patch.object(RunnerAllocationOwner, 'token_gather', True))
            h, infos, cancel, files = D.run(policy, CASE, check=check)
        self.assertEqual(len(checked), len(D.TURNS) + 2)
        # Rebuild, cancel and close: every block back, no gathered pin left.
        self.assertEqual(cancel['receipt_cancel']['state'], 'cancelled')
        self.assertEqual(cancel['committed_before'], cancel['committed_after'])
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
        self.assertEqual(h.mamba_pool.get_num_free_blocks(), h.mamba_baseline)
        self.assertTrue(all(block.ref_cnt == 0 for pool in h.kv.block_pools for block in pool.blocks[1:]))
        self.assertFalse(h.adapter.owner.gathers or h.adapter.owner.block_namespaces)
        rows = [json.loads(line) for line in files['runtime.jsonl'].splitlines()]
        gathered = [r for r in rows if (r['query']['selection'].get('gather') or {}).get('blocks')]
        return h, infos, rows, gathered

    def check_turn(self, h, info, pin):
        P = CASE.P
        rid, geo, plan, b = info['rid'], info['geo'], info['plan'], info['binding']
        s, ns = h.session, info['namespace']
        body = h.selection.authorized[rid]['body']
        selection = body['selection']
        selected = plan['selected']
        gather = body.get('gather')
        tokens = [] if gather is None else gather['tokens']
        G = len(tokens)
        # Receipt, native plan, worker reservation, SparsePlan.
        if gather is None:
            self.assertNotIn('gather', plan)
        else:
            self.assertEqual(plan['gather'], dict(tokens=tokens, blocks=G // P))
            self.assertEqual(G % P, 0)
            self.assertFalse({t // P for t in tokens} & set(selected))
            self.assertTrue(set(gather['pad']) <= set(tokens))
        self.assertEqual(plan['sparse']['dropped_full_attention_tokens'], geo.L - len(selected) * P - G)
        self.assertEqual(h.runner.plans[rid], (geo.L, geo.L - len(selected) * P - G))
        # Phase 2 attends the selected Pages, the gathered tokens and the run; phase 1 R, the previous gather.
        pages = lambda ps: [t for p in ps for t in range(p * P, (p + 1) * P)]
        self.assertEqual(h.runner.prefix_seen[rid], sorted(pages(selected) + tokens))
        previous = getattr(h, 'gather_pinned', ())
        lease_id = plan['phase1']['lease_id'] if plan['phase1'] else None
        if lease_id:
            lease = h.driver.leases[lease_id]['payload']['lease']
            self.assertEqual(h.runner.prefix_seen[lease_id], sorted(pages(plan['resident_before']) + list(previous)))
            self.assertEqual('gathered' in lease, bool(previous))
            if previous:
                self.assertEqual(lease['gathered']['tokens'], list(previous))
                gathered_ids = {g[0] for v in lease['gathered']['block_ids'] for g in v}
                self.assertTrue(gathered_ids <= set(lease['borrowed']))
                self.assertFalse(gathered_ids & set(lease['zero']))
            self.assertEqual(lease['sparse']['dropped_full_attention_tokens'],
                             geo.Q - len(plan['resident_before']) * P - len(previous))
        # The picks: what the picker names, outside the selected Pages; the pad: unattended, not picked.
        record = selection.get('gather')
        active = selection['reason'] is None
        self.assertEqual(record is not None, active)
        if active:
            capacity = selection['capacity_pages']
            self.assertLessEqual(len(selected) + G // P, capacity)
            self.assertEqual(record['blocks'], G // P)
            self.assertEqual(record['gathered'] + record['pad'], G)
            self.assertEqual(record['retrieved'], len(selection['retrieved_pages']))
            stream = h.endpoint.transcript.stream[:geo.L] if False else tuple(b.tokens[:geo.L])
            picks = sorted(set(tokens) - set(gather['pad'])) if gather else []
            attended = set(selection['mandatory_pages']) | set(selection['recent_pages'])
            self.assertFalse({t // P for t in tokens} & attended)
            if pin['picker'] == 'markers_v1':
                patterns = [tuple(X.TOK.encode(m)) for m in pin['markers']]
                found = marker_positions(tuple(b.tokens), patterns)
                self.assertTrue(set(picks) <= found)
                eligible = {t for t in found if t < geo.L and t // P not in attended | set(selected)}
                if not pin['quota'] and not record['truncated']:
                    self.assertEqual(set(picks), eligible)        # every occurrence outside the attended Pages
            else:
                text = lambda t: X.TOK.decode([stream[t]])
                self.assertTrue(all(TG.separator_text(text(t)) for t in picks))
            for t in gather['pad'] if gather else ():
                self.assertNotIn(t // P, attended | set(selected))
        # Pins after the commit: resident Pages, the tail and this turn's gathered blocks, both sides.
        owner = h.adapter.owner
        self.assertEqual(owner.gather_tokens(ns), tuple(tokens))
        self.assertEqual(s.gather, tuple(tokens))
        w = h.owner.sessions[ns]
        self.assertEqual(tuple((w.get('gather') or {}).get('tokens', ())), tuple(tokens))
        pins = owner.pinned(ns)
        self.assertEqual(len(pins), len(s.resident) + (s.tail is not None) + G // P)
        self.assertTrue(all(block.ref_cnt == 1 for block in pins))
        self.assertEqual(h.pool.get_num_free_blocks(), h.baseline - len(pins))
        full = tuple(b.tokens) + tuple(info['request'].output_token_ids)
        pin_ = owner.gather_pin(ns)
        if pin_ is not None:
            for k, groups in enumerate(pin_.blocks):
                block = groups[0][0].block_id
                self.assertEqual(h.store.kv[(W.FA, block)], [(x, full[x]) for x in tokens[k * P:(k + 1) * P]])
        # Evidence: the runtime row carries the record; the native file the plan's gather.
        row = info['row']
        self.assertEqual(row['query']['selection'].get('gather'), record)
        self.assertEqual(info['native']['plan'].get('gather'), plan.get('gather'))
        h.gather_pinned = tuple(tokens)

    def test_observation_query_default_markers(self):
        pin = TG.pin('markers_v1')
        h, infos, rows, gathered = self.play('observation_query', pin)
        self.assertGreaterEqual(len(gathered), 2)
        self.assertTrue(any(r['query']['selection']['gather']['pad'] for r in gathered))

    def test_goal_query_parameterised_markers_with_a_quota(self):
        pin = TG.pin('markers_v1', markers=['<|im_end|>', '<|im_start|>user', ';'], quota=40)
        h, infos, rows, gathered = self.play('goal_query', pin)
        self.assertGreaterEqual(len(gathered), 2)
        self.assertTrue(all(r['query']['selection']['gather']['picks'] <= 40 for r in gathered))
        self.assertTrue(any(r['query']['selection']['gather']['picks'] == 40 for r in gathered))

    def test_observation_query_separators(self):
        pin = TG.pin('separators_v1', quota=64)
        h, infos, rows, gathered = self.play('observation_query', pin)
        self.assertGreaterEqual(len(gathered), 2)
        self.assertTrue(all(r['query']['selection']['gather']['picks'] <= 64 for r in gathered))



@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class MultiSessionFastPathTests(unittest.TestCase):
    """Three sessions decoding together (test_decouple_e2e's MultiHarness at (16, 784)) with the three decode fast
    paths of the stable configuration (test_decode_fastpath3.attach): admissions between batched steps, the batched
    allocation updates and the steady observation over rows that hold gathered blocks."""

    def test_three_sessions_with_the_decode_fast_paths(self):
        self.serve('fast')

    def test_three_sessions_with_the_fast_paths_and_the_gdn_hand_off(self):
        # --gdn-device-handoff: the zero and the restores are issued on the stream; the gather joins them.
        self.serve('handoff')

    def test_three_sessions_with_the_fast_paths_and_phase1_in_a_step(self):
        # --phase1-in-step: phase 1 (reading the previous turn's gathered blocks) runs inside a decode step.
        self.serve('in_step')

    def test_three_sessions_with_asynchronous_scheduling(self):
        # --async-scheduling (with fast paths 1 and 2): placeholder decodes, voided rows, the held updates.
        self.serve('async')

    def serve(self, mode):
        import tempfile
        from pathlib import Path
        import test_decode_fastpath3 as FP3
        P = CASE.P
        multi = D.Case(**dict(vars(CASE), sessions=len(D.TASKS)))
        pin = TG.pin('markers_v1', markers=['<|im_end|>', '<|im_start|>user', ';'], quota=40)
        gathered, active = [], []

        def check(h, info):
            rid, plan = info['rid'], info['plan']
            body = h.selection.authorized[rid]['body']
            gather = body.get('gather')
            tokens = [] if gather is None else gather['tokens']
            pages = [t for p in plan['selected'] for t in range(p * P, (p + 1) * P)]
            self.assertEqual(h.runner.prefix_seen[rid], sorted(pages + tokens))
            self.assertEqual(plan.get('gather'), None if gather is None else dict(tokens=tokens, blocks=len(tokens) // P))
            self.assertEqual(h.runner.plans[rid], (info['geo'].L, info['geo'].L - len(pages) - len(tokens)))
            if body['selection']['reason'] is None:
                active.append(rid)
                self.assertLessEqual(len(plan['selected']) + len(tokens) // P, body['selection']['capacity_pages'])
            if tokens:
                gathered.append(rid)
        with ExitStack() as stack:
            D.enter_case(stack, multi)
            stack.enter_context(patch.object(D, 'contract', gather_contract(pin)))
            stack.enter_context(patch.object(D, 'Runner', GatherRunner))
            stack.enter_context(patch.object(D, 'IO', GatherIO))
            stack.enter_context(patch.object(RunnerAllocationOwner, 'token_gather', True))
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = dict(D.config('observation_query', output, multi, D.TASKS), **FP3.TWO)
            harness = D.MultiHarness
            if mode == 'handoff':
                import test_gdn_handoff as G
                value['admission'] = dict(G.ON)
                harness = G.handoff_harness(D.MultiHarness)
            elif mode == 'in_step':
                import test_adm_phase1_step as IS
                value['admission'] = dict(phase1_in_step=True)
                harness = IS.in_step_harness(False)
            elif mode == 'async':
                import test_async_sched as AS
                import test_stageb_worker as WW
                stack.enter_context(patch.object(D, 'Runner', AS.async_runner(D.Runner, WW, D)))
                harness = lambda policy, output, case: AS.async_harness(D)(policy, output, case, D.TASKS)
            if mode != 'async':
                value['decode_fastpath3'] = True
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            h = harness('observation_query', output, multi)
            if mode == 'async':
                AS.attach_async(h, AS.scripted_stop)
                stack.callback(AS.A.detach_engine)
            else:
                FP3.attach(h, True)
            h.serve(check)
            self.assertEqual(len(h.done), 5 * len(D.TASKS))
            self.assertEqual(max(h.widths), len(D.TASKS))
            self.assertGreater(len(active), 0)
            self.assertGreater(len(gathered), 0)
            h.close()
            D.engine_closed(h.core)
            self.assertEqual(h.pool.get_num_free_blocks(), h.baseline)
            self.assertEqual(h.mamba_pool.get_num_free_blocks(), h.mamba_baseline)
            self.assertFalse(h.adapter.owner.gathers)


class AlternateMarkers(TG.MarkerPicker):
    """A test picker (gather-receipt-1115): markers_v1's picks on a session's 2nd, 4th, ... active turn, none on
    the others, so a session's last turn may gather nothing after an earlier turn gathered. Turns are told apart
    by the length of the session's committed stream, which grows every turn."""
    name = 'alternate_markers_test'

    def picks(self, view):
        seen = view.state.setdefault('stream_lengths', [])
        if len(view.stream) not in seen:
            seen.append(len(view.stream))
        return super().picks(view) if seen.index(len(view.stream)) % 2 else []


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class SessionPoolCloseTests(unittest.TestCase):
    """The close receipt of a session pool (--max-running-sessions) with token gather (gather-receipt-1115).

    test_ms_pool's pool harness at (16, 784), observation_query, six registered tasks of 8, 6, 3, 1, 5 and 2 turns
    with K = 3 (selection is active from a session's fourth turn on). With AlternateMarkers the 8- and 6-turn
    sessions gather on their fifth (and seventh) turn but not on their last, the 5-turn session gathers on its last
    turn and the others never select. Every close passes test_ms_pool's check_close (exactly the session's pins
    come back, released.blocks == resident_pages + tail + gather_blocks, gather_blocks the gathered blocks the
    session pinned); gather_blocks is the last turn's gathered blocks, and the sessions.jsonl rows carry the same."""
    maxDiff = None
    TURNS = (8, 6, 3, 1, 5, 2)
    LIMIT = 3
    KEYS = {'resident_pages', 'tail', 'gather_blocks', 'blocks', 'requests', 'worker'}

    def setUp(self):
        TG.register_picker(AlternateMarkers())
        self.addCleanup(TG.unregister_picker, AlternateMarkers.name)

    def serve(self, gather):
        import tempfile
        from pathlib import Path
        import test_ms_pool as MS
        P = CASE.P
        listed = list(D.TASKS) + MS.tasks(len(self.TURNS) - len(D.TASKS))
        scripts = [list(D.TURNS[:t]) for t in self.TURNS]
        blocks = {}                 # lane index -> the gathered blocks of each of its turns

        def check(h, info):
            rid, plan = info['rid'], info['plan']
            body = h.selection.authorized[rid]['body']
            tokens = (body.get('gather') or {}).get('tokens', [])
            pages = [t for p in plan['selected'] for t in range(p * P, (p + 1) * P)]
            self.assertEqual(h.runner.prefix_seen[rid], sorted(pages + tokens))
            self.assertEqual(plan.get('gather'), dict(tokens=tokens, blocks=len(tokens) // P) if tokens else None)
            # After the commit the session pins exactly this turn's gathered blocks.
            ns = info['namespace']
            self.assertEqual(h.adapter.owner.gather_tokens(ns), tuple(tokens))
            blocks.setdefault(info['lane'].index, []).append(len(tokens) // P)

        PoolHarness = MS.pool_harness_class()
        with ExitStack() as stack:
            if gather:
                pin = TG.pin(AlternateMarkers.name, markers=['<|im_end|>', '<|im_start|>user', ';'], quota=40)
                stack.enter_context(patch.object(D, 'contract', gather_contract(pin)))
                stack.enter_context(patch.object(RunnerAllocationOwner, 'token_gather', True))
            stack.enter_context(patch.object(D, 'Runner', GatherRunner))
            stack.enter_context(patch.object(D, 'IO', GatherIO))
            case = D.Case(**dict(vars(CASE), blocks=self.LIMIT * 400))
            multi, output = MS.EnginePoolTests.enter(self, stack, case, 'observation_query', listed, self.LIMIT)
            h = PoolHarness('observation_query', output, multi, listed, self.LIMIT, scripts)
            self.assertEqual(MS.token_gather_server(h), gather)
            h.serve(check, lambda h, record: MS.EnginePoolTests.check_close(self, h, record))
            self.assertEqual(len(h.done), sum(self.TURNS))
            self.assertEqual(sorted(r.lane.index for r in h.closes), list(range(len(self.TURNS))))
            # Everything released: both pools at their baselines, no gathered pin or gather state left.
            self.assertEqual((h.pool.get_num_free_blocks(), h.mamba_pool.get_num_free_blocks()),
                             (h.baseline, h.mamba_baseline))
            self.assertFalse(h.adapter.owner.gathers or h.adapter.owner.block_namespaces)
            self.assertFalse(h.selection.__dict__.get('gather_state'))
            rows = [json.loads(line) for line in (output / 'sessions.jsonl').read_bytes().splitlines()]
            closes = [r for r in rows if r['event'] == 'close']
            self.assertEqual([(r['session_id'], r['released']) for r in closes],
                             [(r.lane.sid, r.receipt['released']) for r in h.closes])
            self.assertEqual({r['schema'] for r in rows}, {'remote-ra-session-event-v1'})
            h.close()
            D.engine_closed(h.core)
        for record in h.closes:
            released = record.receipt['released']
            self.assertEqual(set(released), self.KEYS)
            # The last turn's gathered blocks are what the close released besides the Pages and the tail.
            self.assertEqual(released['gather_blocks'], blocks[record.lane.index][-1])
        return h, blocks

    def test_close_receipts_itemize_the_gathered_blocks(self):
        h, blocks = self.serve(True)
        kind = {k: 'last' if turns[-1] else 'earlier' if any(turns) else 'never' for k, turns in blocks.items()}
        self.assertEqual(kind, {0: 'earlier', 1: 'earlier', 2: 'never', 3: 'never', 4: 'last', 5: 'never'})
        released = {r.lane.index: r.receipt['released'] for r in h.closes}
        self.assertGreater(released[4]['gather_blocks'], 0)
        self.assertEqual(released[4]['blocks'],
                         released[4]['resident_pages'] + 1 + released[4]['gather_blocks'])
        for k in (0, 1, 2, 3, 5):
            self.assertEqual(released[k]['gather_blocks'], 0)
            self.assertEqual(released[k]['blocks'], released[k]['resident_pages'] + released[k]['tail'])
        # The 8- and 6-turn sessions gathered on their fifth turn (and the 8-turn one on its seventh) only.
        self.assertEqual([turn for turn, count in enumerate(blocks[0]) if count], [4, 6])
        self.assertEqual([turn for turn, count in enumerate(blocks[1]) if count], [4])

    def test_without_token_gather_the_field_is_zero(self):
        h, blocks = self.serve(False)
        self.assertFalse(any(count for turns in blocks.values() for count in turns))
        self.assertTrue(all(r.receipt['released']['gather_blocks'] == 0 for r in h.closes))


if __name__ == '__main__':
    unittest.main()
