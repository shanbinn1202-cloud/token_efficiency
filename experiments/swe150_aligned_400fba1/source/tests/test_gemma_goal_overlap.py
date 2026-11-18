"""Gemma 4 31B IT with --goal-overlap (gemma-overlap; docs/goal-overlap/DESIGN.md section 9). CPU only.

The Gemma profile takes --goal-overlap as Qwen does: a goal_query turn's phase 1 runs while its
Goal is built, the admission joins the Goal after phase 1's drain (it parks only while the Goal
is still being built), and a turn that needs a Goal forward runs it as its own lease, today's
cold forward on its own FA and sliding blocks in the layout's Goal transient. The window layout
adds two things (window_phase1, window_capacity): at phase 1's drain the session's pinned
blocks before the window phase 1 ended in are freed (``WindowPins.trim``: the adoption would
free them), and the sliding ledger of --admission-overlap takes what a waiting admission still
holds off its base (``ledger_ahead_split``), not off the shared transient. Covered:

- the launcher pins ``admission.goal_overlap`` (with or without --admission-overlap) and no vLLM
  argument; goal_query with a rewriter only; remote_ra_http.config.load and the window worker
  take it; the engine attaches it on the window layout;
- the Goal-forward lease of the window layout (contract.parse_goal_lease): cold rows of P-token
  blocks in every group, within the Goal transient;
- the ledger: ``WindowPins.trim``; ``ledger_ahead_split``; a simulation of pool 1 on the Gemma
  geometry (16 sessions, K = 16, random runs, Goals, decodes and completions) where every phase 1,
  Goal forward and decode finds its blocks with the trim and the split ledger, while the ledger of
  today with the refusal simply removed leaves a Goal forward or a phase 1 without its blocks;
- end to end on test_gemma_release's engine and worker fakes with the rewriter stand-in of
  test_gemma_overlap (Goals parked on their sessions' threads): every session's tokens, replies,
  selections, attended Pages and Goal vectors (the worker's Goal-forward reply against the
  flag-off phase-1 receipt) equal the flag-off server's, with --admission-overlap 8 and without it,
  with idle releases and the three decode fast paths, and with phase 1 in chunks; held Goals: every
  continuing turn runs its phase 1 and waits at its Goal holding only its net window; a cancel at
  the Goal join cleans its phase 1 up; no SHA-256/HMAC call beyond the flag-off server's but the
  Goal payload digest moving from phase 1's lease to the Goal forward's.
"""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_gemma_chunked as GC
import test_gemma_e2e as GE
import test_gemma_fastpath3 as F3T
import test_gemma_overlap as GOV
import test_gemma_release as GR
from test_gemma_install import gemma_fixture

from remote_ra_http import config as HC
from remote_ra_live_v2 import admission_flags as AF
from remote_ra_live_v2 import kv_layout as I0
from remote_ra_live_v2 import model_profile as MP
from remote_ra_live_v2 import sliding_window as SW
from remote_ra_live_v2 import window_capacity as WC
from remote_ra_live_v2.contract import LiveV2Error, block_key, parse_goal_lease
from remote_ra_native_growing.contract import Abort

try:
    import torch
except ImportError:
    torch = None

ON = {'goal_overlap': True}
SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
live = GOV.live


def tasks(n):
    return [dict(instance_id='repo__g%02d-%d' % (k, k), problem_statement='Fix goal %02d.' % k) for k in range(n)]


# ---- flag plumbing ---------------------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    """serve.py's Gemma path and config.load take --goal-overlap as Qwen does."""
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with redirect_stdout(io.StringIO()):
            package, site, model, gemma, cls.release = gemma_fixture(cls.root)
            install.install(site, gemma, cls.release, package_root=package, apply=True, model_profile=MP.GEMMA4)
        cls.rewriter = cls.root / 'qwen3-0.6b'
        cls.rewriter.mkdir()
        (cls.rewriter / 'config.json').write_bytes(b'{"model_type":"qwen3"}')
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.enterContext(patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path))

    def prepare(self, policy='goal_query', selection=None, **kw):
        """serve.prepare of a 16-session Gemma server (docs/reference.md 4.14's goal_query launch): the
        config and command with the output directory named OUT, and the config as written."""
        kw = dict(dict(kv_cache_gib=24, page_tokens=16, max_tokens=4096, max_model_len=131072,
                       language_model_only=True, release_idle_pages=True, dram_arena_gib=79), **kw)
        if selection is None and policy != 'full':
            selection = (dict(active_token_budget=8192, goal_rewriter_model=str(self.rewriter))
                         if policy == 'goal_query' else dict(active_token_budget=8192))
        type(self).n += 1
        output = self.root / ('out%d' % self.n)
        with redirect_stdout(io.StringIO()):
            _, config, command, _ = serve.prepare(self.release, tasks(16), policy, output, 8000, selection, **kw)
        named = json.loads(json.dumps([config, command]).replace(json.dumps(str(output))[1:-1], 'OUT'))
        return named, config

    def load(self, config):
        self.n += 1
        path = self.root / ('startup%d.json' % self.n)
        path.write_bytes(serve.wire(config))
        env = dict(REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                   REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config))
        with patch.dict(os.environ, env):
            return HC.load()

    def test_the_flag_pins_its_record_and_no_vllm_argument(self):
        (plain, command), _ = self.prepare()
        self.assertNotIn('admission', plain)
        self.assertEqual(self.prepare(goal_overlap=False)[0], [plain, command])
        (alone, flagged), config = self.prepare(goal_overlap=True)
        self.assertEqual(alone.pop('admission'), ON)
        self.assertEqual([alone, flagged], [plain, command])
        self.assertEqual(HC.admission(self.load(config)), ON)
        (both, flagged), config = self.prepare(goal_overlap=True, admission_overlap=16)
        (overlap, overlap_command), _ = self.prepare(admission_overlap=16)
        self.assertEqual(both.pop('admission'), dict(AF.record(admission_overlap=16), **ON))
        overlap.pop('admission')
        self.assertEqual([both, flagged], [overlap, overlap_command])
        self.assertEqual(HC.admission(self.load(config)), dict(AF.record(admission_overlap=16), **ON))
        # The window layout (its Goal transient) and the pools are the flag-off server's.
        self.assertEqual(json.loads(json.dumps(config['kv_layout'])), plain['kv_layout'])

    def test_the_refusals_are_qwens(self):
        for policy in ('full', 'observation_query'):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, '--goal-overlap needs --policy goal_query'):
                self.prepare(policy, goal_overlap=True)
        with self.assertRaisesRegex(ValueError, '--goal-overlap needs --policy goal_query with --goal-rewriter-model'):
            self.prepare(selection=dict(active_token_budget=8192, no_goal_rewriter=True), goal_overlap=True)
        with self.assertRaisesRegex(ValueError, '--goal-overlap is a switch'):
            self.prepare(goal_overlap=1)
        with self.assertRaisesRegex(ValueError, 'phase1-in-step is not available'):
            self.prepare(goal_overlap=True, phase1_in_step=True)
        _, config = self.prepare(goal_overlap=True, admission_overlap=4)
        for admission in (dict(config['admission'], phase1_in_step=True), dict(ON, gdn_device_handoff=True)):
            with self.subTest(admission=sorted(admission)), \
                    self.assertRaisesRegex(RuntimeError, 'takes --admission-overlap only, besides --goal-overlap'):
                self.load(dict(config, admission=admission))
        rule = dict(config['selection_contract'], goal_rewriter=None)
        with self.assertRaisesRegex(RuntimeError, 'goal_overlap needs goal_query with a pinned Goal rewriter'):
            HC.admission(dict(config, selection_contract=rule))

    def test_the_window_worker_takes_the_record(self):
        attach = GR.WorkerAttachTests('test_release_record').attach
        overlap = GOV.record(3)
        for record in (ON, dict(overlap, **ON)):
            evidence, owner = attach({AF.CONFIG_KEY: record})
            self.assertEqual(evidence[AF.CONFIG_KEY], record)
            self.assertIs(owner.goal_overlap, True)
            self.assertIs(owner.phase1_in_step, False)
        evidence, owner = attach({AF.CONFIG_KEY: overlap})
        self.assertIs(owner.goal_overlap, False)
        for flags in (dict(ON, phase1_in_step=True), dict(overlap, gdn_device_handoff=True, **ON)):
            with self.assertRaisesRegex(LiveV2Error, 'admission-overlap alone of the admission flags, besides --goal-overlap'):
                attach({AF.CONFIG_KEY: flags})


class LeaseTests(unittest.TestCase):
    """The Goal-forward lease of the window layout: group 0 global, groups 1 and 2 sliding."""
    layout = I0.WindowLayout(16, 64, 2, 1024, 512)

    def raw(self, tokens=20, rows=None, zero=None):
        rows = rows or [[3, 4], [7, 8], [9, 10]]
        pools = (0, 1, 1)
        ids = [(pools[g], b) for g, row in enumerate(rows) for b in row if b]
        goal = dict(lease_id='goal-' + 'a' * 32, tokens=list(range(tokens)), rows=rows, owned=[b for _, b in ids],
                    zero=[b for p, b in ids if p == 0] if zero is None else zero,
                    incarnations={block_key(p, b): 1 for p, b in ids}, payload_sha256='b' * 64)
        return dict(lease_id=goal['lease_id'], request='r1', binding='c' * 64, session='s1', namespace='ns/1',
                    generation=3, epoch='e1', source='d' * 64, local_turn=2, page_tokens=16,
                    phase1_lease='phase1-' + 'e' * 32, goal=goal)

    def test_cold_rows_of_p_token_blocks_in_every_group(self):
        lease = parse_goal_lease(self.raw(), attention_groups=(0,), layout=self.layout)
        self.assertEqual(lease.forward.rows, ((3, 4), (7, 8), (9, 10)))
        self.assertEqual(lease.forward.zero, (3, 4))                    # the global group's blocks only
        self.assertEqual(lease.block_keys(), frozenset({(0, 3), (0, 4), (1, 7), (1, 8), (1, 9), (1, 10)}))
        # The Goal transient bounds the tokens (G = 512 here), as in a phase-1 lease.
        parse_goal_lease(self.raw(tokens=512, rows=[list(range(1, 33)), list(range(40, 72)),
                                                    list(range(80, 112))]),
                         attention_groups=(0,), layout=self.layout)
        with self.assertRaisesRegex(LiveV2Error, 'exceeds the Goal transient'):
            parse_goal_lease(self.raw(tokens=513), attention_groups=(0,), layout=self.layout)
        for rows, zero, message in (([[3, 4], [0, 8], [9, 10]], None, 'goal FullAttention row layout'),
                                    ([[3, 4], [7], [9, 10]], None, 'goal FullAttention row layout'),
                                    (None, [3, 4, 7], 'owned/zero')):
            with self.subTest(message=message), self.assertRaisesRegex(LiveV2Error, message):
                parse_goal_lease(self.raw(rows=rows, zero=zero), attention_groups=(0,), layout=self.layout)


class AttachTests(unittest.TestCase):
    def test_the_engine_attaches_on_the_window_layout(self):
        from remote_ra_live_v2.window_phase1 import WindowPhase1Coordinator
        GO = live('goal_overlap')
        for in_step in (False, True):
            phase1 = WindowPhase1Coordinator.__new__(WindowPhase1Coordinator)
            phase1.records, phase1.goal_records, phase1.window, phase1.in_step = {}, {}, object(), in_step
            endpoint = NS(selection=NS(policy='goal_query', rewriter=object()), phase1=phase1, goal_overlap=False)
            core = NS(_remote_ra_private_v2_endpoint=endpoint, _remote_ra_goal_admissions=NS(goals=True))
            if in_step:
                with self.assertRaisesRegex(LiveV2Error, 'without --phase1-in-step'):
                    GO.attach_engine(core, ON)
                continue
            self.assertTrue(GO.attach_engine(core, ON))
            self.assertEqual((endpoint.goal_overlap, phase1.split, phase1.busy()), (True, True, False))


# ---- the ledger --------------------------------------------------------------------------------

class Block:
    def __init__(self, block_id):
        self.block_id, self.ref_cnt, self.is_null = block_id, 1, False


class TrimTests(unittest.TestCase):
    def pins(self):
        from remote_ra_native_growing.window_pins import WindowPins
        freed = []
        blocks = [None] + [Block(b) for b in range(1, 40)]
        pool = NS(blocks=blocks, null_block=NS(is_null=True), free_blocks=lambda bs: freed.extend(b.block_id for b in bs))
        managers = [type('SlidingWindowManager', (), dict(block_pool=pool, req_to_blocks={}, num_cached_block={}))()
                    for _ in range(2)]
        pins = WindowPins(pool, managers, (1, 2), window_tokens=64, page_tokens=16)
        pins.windows['ns'] = {1: tuple((i, blocks[i]) for i in range(3, 8)),
                              2: tuple((i, blocks[10 + i]) for i in range(3, 8))}
        return pins, blocks, freed

    def test_trim_frees_the_blocks_before_the_window_and_the_handover_drops_nothing(self):
        pins, blocks, freed = self.pins()
        self.assertEqual(pins.trim('ns', 5), 4)
        self.assertEqual(sorted(freed), [3, 4, 13, 14])
        self.assertEqual({g: [i for i, _ in v] for g, v in pins.windows['ns'].items()}, {1: [5, 6, 7], 2: [5, 6, 7]})
        self.assertEqual(pins.trim('ns', 5), 0)                         # nothing more before index 5
        head = pins.handover('ns', 'r1', 5, {1: ((8, blocks[30]),), 2: ((8, blocks[31]),)})
        self.assertEqual(sorted(freed), [3, 4, 13, 14])
        self.assertEqual({g: (first, [b.block_id for b in row]) for g, (first, row) in head.items()},
                         {1: (5, [5, 6, 7, 30]), 2: (5, [15, 16, 17, 31])})
        with self.assertRaisesRegex(Abort, 'window trim without a pinned window'):
            pins.trim('ns', 6)

    def test_trim_refuses_a_block_with_another_owner(self):
        pins, blocks, freed = self.pins()
        blocks[4].ref_cnt = 2
        with self.assertRaisesRegex(Abort, 'changed owner before the trim'):
            pins.trim('ns', 5)
        self.assertEqual(freed, [])


class LedgerTests(unittest.TestCase):
    layout = I0.for_window_server(16, 16, 1024, 8192, MP.GEMMA4_GOAL_TOKENS)

    def test_what_a_waiting_admission_holds_comes_off_its_base(self):
        # Nothing held: the rule of today (every base, the largest transient once).
        items = [((10, 1985), 0), ((10, 1500), 0), ((235, 345), 0)]
        self.assertEqual(WC.ledger_ahead_split(items), WC.ledger_ahead(items))
        # A drained phase 1 waiting for its Goal holds 5 of its base of 10: 5 left for good, its
        # transient (the Goal forward still ahead) unchanged; today's rule took it off the transient.
        self.assertEqual(WC.ledger_ahead_split([((10, 1985), 5)]), (5, 1975))
        self.assertEqual(WC.ledger_ahead([((10, 1985), 5)]), (10, 1970))
        # A drain that freed more than it kept: the turn may take that much more for good.
        self.assertEqual(WC.ledger_ahead_split([((10, 1985), -5)]), (15, 1975))
        self.assertEqual(WC.ledger_ahead_split([((10, 1985), 10), ((10, 600), 3)]), (7, 1975))
        # There today's rule reserves 5 blocks too few beside another admission's larger transient
        # (with nothing negative held it reserves at least as much as the split rule).
        items = [((10, 1985), 0), ((10, 600), -5)]
        self.assertEqual(sum(WC.ledger_ahead_split(items)), sum(WC.ledger_ahead(items)) + 5)
        items = [((10, 1985), 0), ((10, 600), 5), ((30, 900), 30)]
        self.assertGreaterEqual(sum(WC.ledger_ahead(items)), sum(WC.ledger_ahead_split(items)))

    def test_a_continuing_turn_holds_its_net_window_while_it_waits(self):
        # A 4,300-token observation late in a session: phase 1 keeps the 65 fresh blocks of the
        # window it ends in, and its drain frees the session's 64 old ones: it holds 1 block per
        # group, within its base of 2 (ledger_parts), while its Goal is built.
        C, e = 50000, 54300
        head, (b0, b1), (f0, f1) = SW.phase1_row(C, e, 1024, 16)
        k0, k1 = SW.kept_blocks(e, 1024, 16)
        stay, trimmed = f1 - max(k0, f0), min(k0, b1) - b0
        self.assertEqual((stay, trimmed, stay - trimmed), (65, 64, 1))
        base, peak = WC.ledger_parts(self.layout, 5, C=C, e=e, goal_tokens=2048, suffix=7)
        self.assertEqual((base, peak), (5 * 2, 5 * (269 + 128)))
        self.assertLessEqual(5 * (stay - trimmed), base)


def simulate(seed, *, trim, rule, atomic=False, sessions=16, permits=16, steps=3000):
    """Pool 1 of a Gemma server (W = 1,024, P = 16, N = 8,192, G = 2,048), one sliding group, under
    the admission gate's sliding inequalities with --admission-overlap ``permits``.

    Every admission is gated as capacity.ledger_gate gates it (phase 1 and phase 2 against the
    in-flight reservations, ``rule`` over (base, peak) and held, and the running requests'
    headroom), then runs phase 1 (its ring, its blocks before the window it ends in freed at the
    drain; ``trim``: the session's pinned blocks before that window too), waits, runs its Goal
    forward (G tokens, when its turn's selection is active) and is accepted (its window becomes the
    request's row; without ``trim`` the adoption frees the old pinned blocks); ``atomic``: today's
    order, the admission waits (for its Goal) before phase 1, and phase 1, the Goal forward and the
    acceptance run in one step. Running requests decode one block
    at a time up to their phase-2 peak and complete (the session pins the window after its reply).
    Returns (None or the first allocation that found too few free blocks, the most admissions that
    waited after their phase 1 at once)."""
    rng = random.Random(seed)
    L = I0.for_window_server(16, sessions, 1024, 8192, MP.GEMMA4_GOAL_TOKENS)
    P, W, N, G = L.page_tokens, L.window_tokens, L.forward_tokens, L.goal_tokens
    chunk = WC.phase1_chunk_tokens(L)
    total_blocks = sessions * L.window_blocks + L.transient_blocks + L.goal_blocks

    def kept(x):
        first, end = SW.kept_blocks(x, W, P)
        return end - first
    sess = []
    for _ in range(sessions):
        C = rng.randrange(300, 60000)
        sess.append(dict(C=C, pin=kept(C), busy=False))
    free = total_blocks - sum(s['pin'] for s in sess)
    inflight, running, waited = [], [], 0

    def conserved():
        held = sum(x['stay'] for x in inflight if x['stage'] == 'B') + sum(r['held'] for r in running)
        assert free + sum(s['pin'] for s in sess) + held == total_blocks and free >= 0

    def accept(x):
        nonlocal free
        if not trim:
            free += x['dropped']            # the handover frees them at the adoption
        sess[x['sid']]['pin'] = 0           # the kept pinned blocks move to the request's row
        running.append(dict(sid=x['sid'], held=x['adopted'], peak=WC.phase2_peak(L, 7), e=x['e']))
        inflight.remove(x)

    def phase1(x):
        nonlocal free
        if x['ring'] > free:
            return 'phase 1 needs %d of %d free' % (x['ring'], free)
        free -= x['stay']                   # the ring, less its blocks before the window freed at the drain
        if trim:
            free += x['dropped']
            sess[x['sid']]['pin'] -= x['dropped']
        x['stage'] = 'B'
        return None

    for _ in range(steps):
        idle = [i for i, s in enumerate(sess) if not s['busy']]
        actions = (['gate'] * 2 if idle and len(inflight) < permits else []) + \
            (['phase1'] if any(x['stage'] == 'A' for x in inflight) else []) + \
            (['goal'] if any(x['stage'] == 'B' for x in inflight) else []) + \
            (['decode', 'decode', 'complete'] if running else [])
        if not actions:
            break
        action = rng.choice(actions)
        if action == 'gate':
            i = rng.choice(idle)
            C = sess[i]['C']
            run = rng.choice((rng.randrange(1, 200), rng.randrange(200, 1300), rng.randrange(1300, 4500),
                              rng.randrange(4500, 12000)))
            e = C + run
            goal = G if rng.random() < 0.7 else 0
            base, peak = WC.ledger_parts(L, 1, C=C, e=e, goal_tokens=goal, suffix=7)
            phase2 = WC.sliding_turn_need(L, 1, cold=False, C=C, e=e, goal_tokens=goal, suffix=7)[1]
            held = [((x['base'], x['peak']), x['held'] if x['stage'] == 'B' else 0) for x in inflight]
            ahead, largest = rule(held)
            room = sum(r['peak'] - r['held'] for r in running)
            need = WC.ledger_phase1((base, peak), largest)
            if need + ahead + largest + room > free or phase2 + ahead + largest + room > free:
                continue                    # CAPACITY_WAIT
            head, (b0, b1), (f0, f1) = SW.phase1_row(C, e, W, P)
            k0, k1 = SW.kept_blocks(e, W, P)
            stay, dropped = max(0, f1 - max(k0, f0)), max(0, min(k0, b1) - b0)
            x = dict(sid=i, e=e, base=base, peak=peak, stay=stay, dropped=dropped, adopted=k1 - k0,
                     ring=SW.phase1_ring(C, SW.phase1_chunks(C, e, N, chunk), W, P),
                     g=WC.cold_peak(L, goal), stage='A')
            x['held'] = stay - dropped if trim else stay       # WindowPhase1Coordinator.held at a gate
            inflight.append(x)
            sess[i]['busy'] = True
        elif action == 'phase1':
            x = rng.choice([x for x in inflight if x['stage'] == 'A'])
            failure = phase1(x)
            if failure is None and atomic:
                # Today's order: the Goal was built before phase 1, whose RPC runs the Goal
                # forward too, and the admission is accepted in the same engine-thread call.
                if x['g'] > free:
                    failure = 'Goal forward needs %d of %d free' % (x['g'], free)
                else:
                    accept(x)
            if failure is not None:
                return failure, waited
            waited = max(waited, sum(1 for x in inflight if x['stage'] == 'B'))
        elif action == 'goal':
            x = rng.choice([x for x in inflight if x['stage'] == 'B'])
            if x['g'] > free:
                return 'Goal forward needs %d of %d free' % (x['g'], free), waited
            accept(x)
        elif action == 'decode':
            r = rng.choice(running)
            if r['held'] < r['peak']:
                if free < 1:
                    return 'a decode needs a block, none free (preemption)', waited
                r['held'] += 1
                free -= 1
        else:
            r = rng.choice(running)
            C2 = r['e'] + rng.randrange(1, 400)
            pin = kept(C2)
            grow = max(0, pin - r['held'])
            if grow > free:
                return 'a decode needs %d of %d free (preemption)' % (grow, free), waited
            free += r['held'] - pin
            running.remove(r)
            sess[r['sid']].update(C=C2, pin=pin, busy=False)
        conserved()
    return None, waited


class LedgerSimulationTests(unittest.TestCase):
    SEEDS = range(40) if SLOW else range(12)

    def test_todays_order_fits_with_todays_ledger(self):
        for seed in self.SEEDS:
            self.assertEqual(simulate(seed, trim=False, rule=WC.ledger_ahead, atomic=True)[0], None, seed)

    def test_waiting_after_phase_1_fits_with_the_trim_and_the_split_ledger(self):
        most = 0
        for seed in self.SEEDS:
            failure, waited = simulate(seed, trim=True, rule=WC.ledger_ahead_split)
            self.assertIsNone(failure, seed)
            most = max(most, waited)
        self.assertGreaterEqual(most, 8)            # many admissions waited after their phase 1 at once

    def test_waiting_after_phase_1_with_todays_ledger_runs_out_of_pool_1(self):
        # The refusal removed and nothing else changed: a waiting admission keeps the session's
        # old window and its new fresh blocks, and the ledger takes them off the shared transient.
        failures = [simulate(seed, trim=False, rule=WC.ledger_ahead)[0] for seed in self.SEEDS]
        self.assertTrue(any(failures), failures)
        self.assertTrue(all(f is None or 'needs' in f for f in failures))


# ---- end to end ---------------------------------------------------------------------------------

class GoalOverlapPressure(GOV.OverlapPressure):
    """test_gemma_overlap.OverlapPressure (K >= 2) or a K = 1 server whose Goals park
    (goal_async.GoalAdmissions), with --goal-overlap attached as compose_engine attaches it
    (``goal_overlap``: the engine's goal_overlap.attach_engine and the worker owner's switch)."""

    def __init__(self, policy, output, task_list, fa_blocks, release, turns, permits, goal_overlap=True):
        GA = live('goal_async')
        if permits > 1:
            super().__init__(policy, output, task_list, fa_blocks, release, turns, permits)
        else:
            GR.Pressure.__init__(self, policy, output, task_list, fa_blocks, release, turns)
            self.permits, self.parked_adds, self.busy, self.parked_peak = 1, {}, set(), 0
            self.continuing_peak = 0
            self.table = GA.GoalAdmissions(self.lifecycle, wake=lambda: None, goals=True)
            self.lifecycle.goals = self.table
        self.core._remote_ra_goal_admissions = self.table
        if goal_overlap:
            record = dict(AF.record(admission_overlap=permits) or {}, **ON)
            assert live('goal_overlap').attach_engine(self.core, record) is True
            self.owner.goal_overlap = True

    def accepted(self, info, ack, before):
        info = super().accepted(info, ack, before)
        # Phase 1 ran before the admission waited for its Goal: its probe's positions by its lease.
        leases = {lid for lid, record in self.driver.leases.items()
                  if lid.startswith('phase1-') and record['payload']['request'] == info['rid']}
        info['probe'] = [x for r, x in self.runner.computed if r in leases]
        return info


def long_script(index, turns):
    """Observations longer than one native forward (test_gemma_chunked's)."""
    return [(None if k == 0 else GC.LONG[(k - 1 + index) % len(GC.LONG)], GE.TARGETS[(k + index) % len(GE.TARGETS)],
             (k + index) % 3 == 2) for k in range(turns)]


def goals_of(h):
    """(lane, turn) -> the turn's Goal vector, its token count and payload digest (the phase-1
    receipt's without the flag, the Goal-forward reply's with it), or None without a Goal forward."""
    out = {}
    for info in h.done:
        frames = h.proxy.http_receipts[info['rid']]
        goal = [f for f in frames if f['operation'] in ('phase1', 'goal_forward') and f.get('goal_vector') is not None]
        out[info['lane'].index, info['turn']] = (None if not goal else (goal[0]['goal_vector'], goal[0]['goal_token_count'],
                                                                         goal[0]['goal_payload_sha256']))
    return out


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class GoalOverlapEndToEndTests(unittest.TestCase):
    maxDiff = None
    TURNS = 3
    SESSIONS = 8
    TEST_GOAL = GOV.OverlapEndToEndTests.TEST_GOAL
    setUp = GOV.OverlapEndToEndTests.setUp
    check, boundary, window_of = GR.SixteenSessionTests.check, GR.SixteenSessionTests.boundary, \
        GR.SixteenSessionTests.window_of
    _references = {}

    def server(self, stack, blocks, permits, *, goal_overlap=True, release=True, sessions=SESSIONS, turns=TURNS,
               chunk=None, fast=False):
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        layout = GE.window_layout('goal_query', sessions)
        value = GR.config('goal_query', output, GR.tasks(sessions), layout, release)
        record = dict(AF.record(admission_overlap=permits or 1) or {}, **(ON if goal_overlap else {}))
        if record:
            value = dict(value, admission=record)
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        if permits is None:
            h = GR.Pressure('goal_query', output, GR.tasks(sessions), blocks, release, turns)
        else:
            h = GoalOverlapPressure('goal_query', output, GR.tasks(sessions), blocks, release, turns, permits,
                                    goal_overlap)
        self.assertIsNotNone(h.selection.rewriter)
        if chunk is not None:
            h.endpoint.phase1.chunk_tokens = chunk
            for lane in h.lanes:
                lane.script = long_script(lane.index, turns)
        if fast:
            F3T.attach_fast3(h)
        h.output, h.turns = output, turns
        return h

    def finish(self, h):
        """test_gemma_overlap's finish (every turn served, both pools back at their baselines)
        and the Goal vectors of every turn."""
        goals = goals_of(h)
        frames = [f for info in h.done for f in h.proxy.http_receipts[info['rid']]]
        result, events = GOV.OverlapEndToEndTests.finish(self, h, None)
        self.assertEqual((h.endpoint.phase1.records, h.endpoint.phase1.goal_records), ({}, {}))
        if getattr(h, 'table', None) is not None:
            self.assertEqual(h.table.parked, {})
            self.assertIn(h.lifecycle.reservations, (None, {}))
        return result, goals, frames, events

    def run_case(self, blocks, permits, **kw):
        with ExitStack() as stack:
            h = self.server(stack, blocks, permits, **kw)
            h.serve(self.check)
            return self.finish(h) + (h,)

    def reference(self, **kw):
        """The flag-off K = 1 server (Goals built inline, one admission at a time)."""
        key = tuple(sorted(kw.items()))
        if key not in self._references:
            result, goals, frames, events, _ = self.run_case(GR.REFERENCE_BLOCKS, None, goal_overlap=False, **kw)
            self.assertEqual(events, [])
            self._references[key] = result, goals, frames
        return self._references[key]

    def same(self, flagged, reference):
        """Every turn equals the flag-off server's, its Goal vector included; with the flag no
        phase-1 receipt carries a Goal vector and each Goal forward has its own frame after phase 1."""
        result, goals, frames = flagged[:3]
        self.assertEqual(result, reference[0])
        self.assertEqual(goals, reference[1])
        forwards = sum(1 for g in reference[1].values() if g is not None)
        self.assertGreater(forwards, 0)
        self.assertTrue(all(f['goal_vector'] is None for f in frames if f['operation'] == 'phase1'))
        self.assertEqual(sum(1 for f in frames if f['operation'] == 'goal_forward'), forwards)
        self.assertTrue(all(f['drained'] is True and f['drain']['drained'] is True
                            for f in frames if f['operation'] == 'goal_forward'))

    def test_overlapped_admissions_decide_as_the_flag_off_server(self):
        reference = self.reference()
        flagged = self.run_case(GR.REFERENCE_BLOCKS, self.SESSIONS)
        self.same(flagged, reference)
        h = flagged[-1]
        self.assertGreaterEqual(h.parked_peak, 2)
        self.assertTrue(any(r.calls for r in self.rewriters))

    def test_without_admission_overlap_a_parked_goal_waits_after_phase_1(self):
        flagged = self.run_case(GR.REFERENCE_BLOCKS, 1)
        self.same(flagged, self.reference())
        self.assertEqual(flagged[-1].parked_peak, 1)

    def test_with_idle_releases_and_the_decode_fast_paths(self):
        small = GR.selective_blocks(GE.window_layout('goal_query', self.SESSIONS))
        flagged = self.run_case(small, self.SESSIONS, fast=True)
        self.same(flagged, self.reference())
        events = flagged[3]
        self.assertTrue([e for e in events if e['event'] == 'evict'] and
                        [e for e in events if e['event'] == 'readmit'])

    def test_phase_1_in_chunks(self):
        reference = self.reference(sessions=4, chunk=256)
        flagged = self.run_case(GR.REFERENCE_BLOCKS, 4, sessions=4, chunk=256)
        self.same(flagged, reference)
        chunked = [f for f in flagged[2] if f['operation'] == 'phase1' and f['end'] - f['start'] > GE.N]
        self.assertTrue(chunked)

    def held_round(self, stack, sessions=SESSIONS, turns=TURNS, goal_overlap=True):
        """The first turns of every lane, then every lane's last turn admitted while every Goal is
        held: with the flag each admitted turn runs its phase 1 at once and waits at its Goal join
        (without it at its Goal, before phase 1); the gate refuses the others (CAPACITY_WAIT: they
        are admitted again once the Goals return)."""
        h = self.server(stack, GR.REFERENCE_BLOCKS, sessions, release=False, sessions=sessions, turns=turns,
                        goal_overlap=goal_overlap)
        later = [lane.script[turns - 1:] for lane in h.lanes]
        for lane in h.lanes:
            lane.script = lane.script[:turns - 1]
        h.serve(self.check)
        self.assertEqual(len(h.done), sessions * (turns - 1))
        # serve closed the park table: a new one, as test_gemma_overlap's scripted round makes.
        h.table = type(h.table)(h.lifecycle, wake=lambda: None, goals=True, permits=sessions)
        h.lifecycle.goals = h.table
        for lane, script in zip(h.lanes, later):
            lane.script = script
        self.goals.clear()
        h.before = h.window_pool.get_num_free_blocks()
        h.outcomes = [h.admit(lane) for lane in h.lanes]
        return h

    def test_held_goals_every_turn_waits_after_its_phase_1_holding_its_net_window(self):
        CAP = live('capacity')
        with ExitStack() as stack:
            off = self.held_round(stack, goal_overlap=False)
            outcomes = off.outcomes
            self.goals.set()
            off.serve(self.check)
        self.goals.set()
        with ExitStack() as stack:
            h = self.held_round(stack)
            # As many turns wait as without the flag (this geometry's long runs bound them: the
            # largest transient), each after its phase 1.
            self.assertEqual(h.outcomes, outcomes)
            self.assertGreaterEqual(h.outcomes.count('parked'), 2)
            phase1 = h.endpoint.phase1
            self.assertEqual(sorted(phase1.records), sorted(h.table.parked))
            self.assertFalse(phase1.busy())
            held, items = 0, []
            for rid, record in phase1.records.items():
                self.assertTrue(record['drained'])
                self.assertIsInstance(h.table.parked[rid].work, live('goal_async').GoalJoin)
                window = record['window']
                pinned = h.adapter.window.window(window['namespace'])
                # The drain freed the session's pinned blocks before the window phase 1 ended in.
                self.assertTrue(all(i >= window['first'] for v in pinned.values() for i, _ in v))
                fa, sliding = phase1.held(rid)
                self.assertEqual(sliding, sum(len(v) for v in window['fresh'].values()) - window['trimmed'])
                self.assertEqual(fa, sum(len(groups) for groups in record['fa'].values()))
                held += sliding
                items.append((h.lifecycle.reservations[rid][1], sliding))
                self.assertLessEqual(sliding, h.lifecycle.reservations[rid][1][0])   # within its base
            # Pool 1 holds exactly what the waiting turns hold beyond their sessions' old windows.
            self.assertEqual(h.window_pool.get_num_free_blocks(), h.before - held)
            total, largest = WC.ledger_ahead_split(items)
            self.assertEqual(CAP.reserved(h.lifecycle)[0][1], total + largest)
            self.goals.set()
            h.serve(self.check)
            result = self.finish(h)
        self.same(result, self.reference(release=False))
        # The held round's turns ran Goal forwards after their wait.
        self.assertTrue(any(f['operation'] == 'goal_forward' for f in result[2]))

    def test_a_cancel_at_the_goal_join_cleans_its_phase_1_up(self):
        with ExitStack() as stack:
            h = self.held_round(stack, sessions=4)
            self.assertIn('parked', h.outcomes)
            rid = sorted(h.table.parked)[0]
            info, replies = h.parked_adds.pop(rid)
            lane = info['lane']
            entry = h.lifecycle.entries[rid]
            record = h.endpoint.phase1.records[rid]
            namespace = record['window']['namespace']
            h.lifecycle.cancel(entry['identity'])
            (frame, failure), = replies
            self.assertIsNone(frame)
            self.assertIsInstance(failure, live('goal_async').GoalAborted)
            self.assertEqual(entry['state'], 'cancelled')
            self.assertEqual(h.lifecycle.sessions[lane.sid][1], 'poisoned')
            self.assertNotIn(rid, h.endpoint.phase1.records)
            self.assertEqual([f['operation'] for f in h.proxy.http_receipts[rid]][-3:],
                             ['phase1', 'phase1_drain', 'cancel'])
            # Its fresh blocks are freed; the session keeps its trimmed window until it closes.
            self.assertTrue(all(i >= record['window']['first']
                                for v in h.adapter.window.window(namespace).values() for i, _ in v))
            h.busy.discard(lane.index)
            lane.frontend.abandon()
            lane.script = []
            self.goals.set()
            h.serve(self.check)
            self.assertEqual(len(h.done), 4 * 2 + 3)
            others = {i['lane'].index: i['committed']['text'] for i in h.done if i['turn'] == 2}
            h.lifecycle.disconnect()
            h.sessions.close()
            from remote_ra_http.evidence import engine_closed
            engine_closed(h.core)
            self.assertEqual(h.adapter.window.pinned_blocks(), 0)
            self.assertEqual((h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks()),
                             (h.baseline, h.window_baseline))
            self.assertEqual((h.table.parked, h.endpoint.phase1.records, h.endpoint.phase1.goal_records), ({}, {}, {}))
        reference = self.reference(release=False, sessions=4)[0]
        self.assertEqual(others, {k: rows[2][1] for k, rows in reference.items() if k != lane.index})

    def test_no_hashing_beyond_the_flag_off_server(self):
        """Every SHA-256/HMAC call of the runtime sources, every thread counted, is the flag-off
        server's (the same harness with --admission-overlap 4, Goals parked before phase 1), except
        the Goal payload digest, which moves from phase 1's lease (``run``) to the Goal forward's
        (``_goal_lease``), once per Goal forward."""
        import test_adm_overlap as OV
        counts = {}
        for flag in (False, True):
            hashing = OV.CountedHashing()
            with ExitStack() as stack:
                h = self.server(stack, GR.REFERENCE_BLOCKS, 4, goal_overlap=flag, release=False, sessions=4)
                hashing.run(lambda: h.serve(self.check))
                forwards = sum(1 for g in goals_of(h).values() if g is not None)
                self.finish(h)
            counts[flag] = hashing.inner.calls, forwards
        (before, forwards), (after, again) = counts[False], counts[True]
        self.assertTrue(forwards and before and forwards == again)
        delta = {key: after[key] - before[key] for key in set(before) | set(after) if after[key] != before[key]}
        self.assertEqual(delta, {('window_phase1.py', 'run'): -forwards, ('window_phase1.py', '_goal_lease'): forwards})


if __name__ == '__main__':
    unittest.main()
