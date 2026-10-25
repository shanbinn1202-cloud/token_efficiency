"""Route B capacity on CPU: serve.py --release-idle-pages (remote_ra_live_v2.idle_release).

Under GPU pressure the capacity gate releases idle sessions' resident Pages (their
DRAM copies stay in the worker's catalog) instead of waiting, and an evicted
session's next turn readmits exactly those Pages before its phase 1. Nothing here
is CUDA evidence. Covered:

- the launcher and the startup config: the pinned switch (decoupled layout only),
  off adds nothing;
- PrivateBlockOwner.evict/readmit: the resident Pages leave and come back, the
  tail stays, borrowed blocks and sessions with native work are refused; the
  worker refuses evict/readmit without the switch;
- end to end on the engine (test_decouple_e2e's batched engine and worker at P = 16):
  sixteen sessions whose idle pins exceed the FullAttention pool. Without the
  switch the run cannot proceed (with nothing running, every turn waits: the idle
  pins leave it no room; before tb16 job 3398129's fix such a turn was admitted
  and found no blocks); with it every turn completes, never preempting; every forward reads
  exactly the Pages and KV it reads with a pool too large to release anything
  (tokens, selections, attended Pages, replies equal per session); victims are
  idle sessions only, least recently used first; readmission restores exactly the
  released Pages; capacity.jsonl, the native records and engine-cleanup.json;
  both pools back at their baselines. Two scenarios: test_decouple_e2e's script
  (900 vs 9,000 blocks, full and observation_query), and Budget32kTests (about
  5 minutes; REMOTE_RA_SLOW_TESTS=1): eleven ~3.6K-token observations per session
  at a 32,768-token budget, 25,600 vs 48,000 blocks.
"""
from contextlib import ExitStack
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install  # noqa: E402
import serve  # noqa: E402
import test_installer as INSTALL  # noqa: E402

from remote_ra_http import config as http_config  # noqa: E402
from remote_ra_live_v2 import idle_release as R  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


def tasks(count):
    return [dict(instance_id='repo__r%02d-%d' % (k, 300 + k),
                 problem_statement='Resolve the defect %02d of module r%02d.' % (k, k)) for k in range(count)]


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with contextlib.redirect_stdout(io.StringIO()):
            package, site, model, cls.release = INSTALL.fixture(cls.root)
            install.install(site, model, cls.release, package_root=package, apply=True)
        cls.count = 0

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def prepare(self, page_tokens=16, **options):
        type(self).count += 1
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            return serve.prepare(self.release, tasks(4), 'full', self.root / ('out%d' % self.count), 8231,
                                 None, 'mswea-xml', 24, page_tokens, None, max_running=4, **options)

    def load(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'startup.json'
            path.write_bytes(serve.wire(config))
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                            REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
                return http_config.load()

    def test_switch(self):
        _, off, argv, env = self.prepare()
        _, on, argv_on, env_on = self.prepare(release_idle_pages=True)
        self.assertNotIn('release_idle_pages', off)
        self.assertEqual(on['release_idle_pages'], True)
        self.assertEqual({k: v for k, v in on.items() if k != 'release_idle_pages'},
                         json.loads(json.dumps(off).replace('out%d' % (self.count - 1), 'out%d' % self.count)))
        self.assertEqual(argv_on[1:], json.loads(json.dumps(argv).replace('out%d' % (self.count - 1),
                                                                           'out%d' % self.count))[1:])
        self.assertEqual((R.pinned(self.load(on)), R.pinned(self.load(off))), (True, None))
        with self.assertRaisesRegex(ValueError, 'needs the decoupled KV layout'):
            self.prepare(page_tokens=784, release_idle_pages=True)
        for bad in (dict(on, release_idle_pages=False), dict(on, release_idle_pages=1),
                    {k: v for k, v in on.items() if k != 'kv_layout'}):
            with self.assertRaises(RuntimeError):
                self.load(bad)
        with self.assertRaisesRegex(ValueError, 'is a switch'):
            self.prepare(release_idle_pages=1)
        # The worker reads the record once and compares it with the pinned config.
        self.assertEqual((R.worker_flag({}), R.worker_flag({R.CONFIG_KEY: True})), (None, True))
        with self.assertRaises(LiveV2Error):
            R.worker_flag({R.CONFIG_KEY: False})


class OwnerTests(unittest.TestCase):
    def test_evict_and_readmit(self):
        import test_stageb_engine as E
        from remote_ra_native_growing.ownership import PrivateBlockOwner, TailPin
        from remote_ra_native_growing.contract import Abort
        pool = E.FakePool(40)
        owner = PrivateBlockOwner(pool)
        ns = 'sparse/x'
        resident = {p: ((pool.get_new_blocks(1)[0],),) for p in (0, 2, 3)}
        tail_block = pool.get_new_blocks(1)[0]
        for groups in resident.values():
            owner.block_namespaces[groups[0][0].block_id] = ns
        owner.block_namespaces[tail_block.block_id] = ns
        owner.sessions[ns], owner.tails[ns] = dict(resident), TailPin(5, 80, 85, ((tail_block,),))
        free = pool.get_num_free_blocks()
        # Borrowed blocks (a request reference) and native work are refused.
        pool.touch([resident[2][0][0]])
        with self.assertRaises(Abort):
            owner.evict(ns)
        pool.free_blocks([resident[2][0][0]])
        released = owner.evict(ns)
        self.assertEqual((sorted(released), owner.sessions[ns], pool.get_num_free_blocks()), ([0, 2, 3], {}, free + 3))
        self.assertIs(owner.tails[ns].groups[0][0], tail_block)                 # the tail stays
        self.assertEqual(set(owner.block_namespaces), {tail_block.block_id})
        back = owner.readmit(ns, (0, 2, 3), 1)
        self.assertEqual((sorted(back), pool.get_num_free_blocks()), ([0, 2, 3], free))
        self.assertTrue(all(groups[0][0].ref_cnt == 1 and owner.block_namespaces[groups[0][0].block_id] == ns
                            for groups in back.values()))
        self.assertEqual(sorted(owner.pinned(ns), key=lambda b: b.block_id),
                         sorted([g[0][0] for g in back.values()] + [tail_block], key=lambda b: b.block_id))
        for bad in (((0,), 1), ((), 1), ((6,), 1)):
            with self.assertRaises(Abort):
                owner.readmit(ns, *bad)            # the session has resident Pages again
        owner.evict(ns)
        with self.assertRaises(Abort):
            owner.readmit(ns, (6,), 1)            # at or after the tail Page
        with self.assertRaises(Abort):
            owner.readmit(ns, tuple(range(5)), 9)  # more blocks than the pool has
        owner.leases['r'] = NS(namespace=ns)
        with self.assertRaises(Abort):
            owner.readmit(ns, (0,), 1)
        with self.assertRaises(Abort):
            owner.evict(ns)


class WorkerOperationTests(unittest.TestCase):
    def test_evict_and_readmit_need_the_switch(self):
        from remote_ra_live_v2.allocation_owner import OPERATIONS, RunnerAllocationOwner
        from remote_ra_live_v2.composition import EngineWorkerProxy
        self.assertTrue({'evict', 'readmit'} <= OPERATIONS and {'evict', 'readmit'} <= EngineWorkerProxy.OPERATIONS)
        owner = RunnerAllocationOwner.__new__(RunnerAllocationOwner)
        owner.records, owner.terminals = {'r': dict(binding='b')}, {}
        self.assertFalse(owner.release_idle)
        for operation, payload in (('evict', dict(session='s', namespace='n', pages=[0])),
                                   ('readmit', dict(request='r', binding='b', session='s', namespace='n', pages={}))):
            with self.assertRaisesRegex(LiveV2Error, 'needs --release-idle-pages'):
                owner.resolve_worker_command(operation, payload)


class AdmitOrderTests(unittest.TestCase):
    """IdleRelease.admit's order of release and gate, on either layout (integrate-1016). With one
    admission at a time the gate admits a turn unchecked (None) when nothing runs, and the
    release of what it needs follows. With --admission-overlap K (``lifecycle.reservations``)
    the ledger gate admits such a turn (nothing runs, no admission in flight) unchecked too but
    returns its figures, so the release runs first; with an admission in flight or a request
    running the ledger gate checks the turn and nothing is released before it."""

    def admit(self, reservations, entries, flying, result):
        import importlib
        IR = importlib.import_module('remote_ra_live_v2.idle_release')
        CAP = importlib.import_module('remote_ra_live_v2.capacity')
        order = []

        def gate(lifecycle, command):
            order.append('gate')
            return result
        with patch.object(CAP, 'reserved', lambda lifecycle: (0, flying)), \
                patch.object(IR.IdleRelease, 'make_room', lambda self, lifecycle, command: order.append('release')):
            out = IR.IdleRelease(NS()).admit(NS(reservations=reservations, entries=entries), NS(request_id='r'), gate)
        return order, out

    def test_one_admission_at_a_time(self):
        need = dict(phase1=(1, 0))
        self.assertEqual(self.admit(None, {}, 0, None), (['gate', 'release'], None))
        self.assertEqual(self.admit(None, {}, 0, need), (['gate'], need))

    def test_overlap_releases_before_an_unchecked_admission(self):
        need = dict(phase1=(1, 0))
        self.assertEqual(self.admit({}, {}, 0, need), (['release', 'gate'], need))
        self.assertEqual(self.admit({'a': (1, 0)}, {'a': dict(state='reserving')}, 1, need), (['gate'], need))
        self.assertEqual(self.admit({}, {'b': dict(state='accepted')}, 0, need), (['gate'], need))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class PressureTests(unittest.TestCase):
    """Sixteen sessions on test_decouple_e2e's batched engine at P = 16 (its five-turn
    script, cut to four turns, its own contract): a pool of 900 FullAttention blocks
    against one of 9,000 that never releases anything."""
    maxDiff = None
    SESSIONS, TURNS = 16, 4
    BLOCKS, REFERENCE = 900, 9000

    def contract(self, case):
        """The selection contract of the query policies (None: test_decouple_e2e's)."""
        return None

    def script(self):
        import test_decouple_e2e as DE
        return list(DE.TURNS[:self.TURNS])

    def enter(self, stack, policy, blocks):
        import test_decouple_e2e as DE
        case = DE.Case(**dict(vars(DE.CASES[2]), sessions=self.SESSIONS, blocks=blocks))
        DE.enter_case(stack, case)
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        value = DE.config(policy, output, case, tasks(self.SESSIONS))
        contract = self.contract(case)
        if contract is not None and value['selection_contract'] is not None:
            value['selection_contract'] = contract
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        return case, output

    def harness(self, policy, case, output, release):
        import test_decouple_e2e as DE
        h = DE.MultiHarness(policy, output, case, tasks=tasks(self.SESSIONS))
        for lane in h.lanes:
            lane.script = self.script()
        h.active = h.restores = 0          # MultiSessionMatrixTests.check counts these
        if release:
            h.owner.release_idle = True
            h.core._remote_ra_idle_release = R.IdleRelease(h.core, output)
        return h

    def run_case(self, policy, blocks, release, observe=None):
        """Serve every lane's script; returns each lane's turns (prompt, reply, selected and
        attended Pages), the harness, capacity.jsonl, the native records and the cleanup."""
        import test_decouple_e2e as DE
        from remote_ra_http.evidence import engine_closed
        with ExitStack() as stack:
            case, output = self.enter(stack, policy, blocks)
            h = self.harness(policy, case, output, release)
            if observe is not None:
                observe(h)
            check = lambda h, info: DE.MultiSessionMatrixTests.check(self, h, info)
            h.serve(check)
            result = {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'],
                                    h.runner.visible[i['rid']]) for i in h.done if i['lane'] is lane]
                      for lane in h.lanes}
            self.assertEqual(len(h.done), self.SESSIONS * self.TURNS)
            events = ([json.loads(line) for line in (output / R.EVENTS).read_bytes().splitlines()]
                      if (output / R.EVENTS).exists() else [])
            natives = [i['native'] for i in h.done]
            # What the idle sessions pin at the end, before shutdown closes them.
            h.final_pins = sum(len(s.resident) + (s.tail is not None) for s in h.adapter.sessions.values())
            h.close()
            engine_closed(h.core)
            cleanup = json.loads((output / 'engine-cleanup.json').read_bytes())
            self.assertEqual((h.pool.get_num_free_blocks(), h.mamba_pool.get_num_free_blocks()),
                             (h.baseline, h.mamba_baseline))
            return result, h, events, natives, cleanup

    def pressure(self, policy):
        # Large enough that nothing is ever released: the reference.
        reference, big, events, natives, cleanup = self.run_case(policy, self.REFERENCE, True)
        self.assertEqual(events, [])
        self.assertEqual(cleanup['idle_release'], dict(evictions=0, evicted_pages=0, readmissions=0,
                                                       readmitted_pages=0))
        self.assertTrue(all(n['idle_release'] == dict(evicted_sessions=[], evicted_pages=0, readmitted_pages=[])
                            for n in natives))
        # The sixteen sessions' idle pins exceed the small pool: the gate must release.
        pins = big.final_pins
        self.assertGreater(pins, self.BLOCKS)
        plans = []

        def observe(h):
            release = h.core._remote_ra_idle_release
            victims = release.victims

            def watched(lifecycle, command, short):
                chosen = victims(lifecycle, command, short)
                idle = [sid for sid, _ in release.idle(lifecycle, command.session_id)]
                plans.append(dict(chosen=chosen, idle=idle, used=[release.used.get(sid, 0) for sid in idle],
                                  admitting=command.session_id, short=short,
                                  running={info['lane'].sid for info in h.running}))
                return chosen
            release.victims = watched
        result, h, events, natives, cleanup = self.run_case(policy, self.BLOCKS, True, observe)
        # Every session saw, decided and attended exactly what it did with the large pool.
        self.assertEqual(result, reference)
        evictions = [e for e in events if e['event'] == 'evict']
        readmissions = [e for e in events if e['event'] == 'readmit']
        self.assertTrue(evictions and readmissions)
        # Victims: idle sessions only (never running, never the admitting one), least
        # recently used first, and only as many as the shortfall needs.
        self.assertTrue(plans)
        for plan in plans:
            self.assertEqual(plan['used'], sorted(plan['used']))
            if plan['chosen']:
                self.assertEqual(plan['chosen'], plan['idle'][:len(plan['chosen'])])
                self.assertFalse(set(plan['chosen']) & (plan['running'] | {plan['admitting']}))
        self.assertEqual(sum(len(p['chosen'] or ()) for p in plans), len(evictions))
        for e in evictions:
            self.assertGreater(e['free_blocks_by_pool'][0], e['free_blocks_by_pool_before'][0])
            self.assertEqual(e['free_blocks_by_pool'][1], e['free_blocks_by_pool_before'][1])   # FA only
        # Each readmission restores exactly what that session's last eviction released.
        last = {}
        for e in events:
            if e['event'] == 'evict':
                self.assertNotIn(e['session_id'], last)
                last[e['session_id']] = e['pages']
            else:
                self.assertEqual(e['pages'], last.pop(e['session_id']))
        self.assertEqual(cleanup['idle_release'], dict(evictions=len(evictions),
            evicted_pages=sum(e['pages'] for e in evictions), readmissions=len(readmissions),
            readmitted_pages=sum(e['pages'] for e in readmissions)))
        rows = [n['idle_release'] for n in natives]
        self.assertEqual(sum(len(r['evicted_sessions']) for r in rows), len(evictions))
        self.assertEqual(sum(len(r['readmitted_pages']) for r in rows), sum(e['pages'] for e in readmissions))
        # Without the switch the same pool cannot serve the sixteen sessions: once every
        # session is idle and pins, nothing runs and the next turn does not fit. The gate
        # refuses it (CAPACITY_WAIT), and every other lane's turn too: the run stalls, and no
        # turn is admitted into a pool it cannot get its blocks from (tb16 job 3398129: the gate
        # admitted such a turn unchecked and vLLM preempted it, fatally; here it failed with
        # "insufficient native blocks" or "found no free block").
        import test_decouple_e2e as DE
        with self.assertRaises(DE.CapacityStall):
            self.run_case(policy, self.BLOCKS, False)
        return dict(events=events, plans=plans, harness=h, pins=pins)

    # The slow pass (README section 2.3); observation_query's sixteen sessions run in the default suite.
    @unittest.skipUnless(os.environ.get('REMOTE_RA_SLOW_TESTS') == '1',
                         'about 25 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_sessions_full(self):
        self.pressure('full')

    def test_sixteen_sessions_observation_query(self):
        self.pressure('observation_query')


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
@unittest.skipUnless(os.environ.get('REMOTE_RA_SLOW_TESTS') == '1',
                     'about 5 minutes of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
class Budget32kTests(PressureTests):
    """Sixteen read_next-like sessions at the MemoryAgentBench budget, 32,768 tokens
    (2,048 Pages of 16 tokens): eleven ~3.6K-token observations each, so every session
    ends past 39K tokens of history and its idle pins approach the budget plus the
    Pages sealed in its turn. A pool of 25,600 FullAttention blocks (25 GiB of 1 MiB
    Pages) cannot hold sixteen of them; 48,000 never releases anything."""
    SESSIONS, TURNS = 16, 12
    BLOCKS, REFERENCE = 25600, 48000

    def contract(self, case):
        import test_decouple_e2e as DE
        from remote_ra_live_v2 import selection_contract as sc
        return sc.default_contract(case.P, active_ratio=None, active_token_budget=32768,
                                   vector_format=sc.vector_format_for(DE.layout_record(case)))

    def script(self):
        return [(None, 0, None, True)] + [('history chunk %d' % k, 3500, None, True)
                                          for k in range(self.TURNS - 1)]

    def test_sixteen_sessions_full(self):
        self.skipTest('the 32K budget applies to the query policies')

    def test_sixteen_sessions_observation_query(self):
        self.skipTest('see test_sixteen_sessions_at_a_32k_budget')

    def test_sixteen_sessions_at_a_32k_budget(self):
        found = self.pressure('observation_query')
        h = found['harness']
        # The budget binds: the last turns select 2,048 Pages (32,768 tokens) out of more.
        last = [i for i in h.done if i['binding'].turn == self.TURNS - 1]
        self.assertTrue(all(len(i['plan']['selected']) == 2048 and i['geo'].n > 2048 for i in last))


if __name__ == '__main__':
    unittest.main()
