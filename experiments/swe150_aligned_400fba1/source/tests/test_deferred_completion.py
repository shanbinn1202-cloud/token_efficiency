"""--deferred-completion-io (remote_ra_live_v2.deferred_completion). CPU only; not GPU evidence.

- The pin, the launcher and startup: on every base of test_integrate_flags (the XML protocol on
  the legacy and decoupled layouts, a session pool with CUDA graphs, the admission flags with
  goal_query, kv-session at 262144 tokens, chat-text with CUDA graphs) and with the decode fast
  paths, --language-model-only and --release-idle-pages, and next to --goal-rewriter-batch, the
  switch adds exactly its startup key and changes no vLLM argument; the other switches' refusals
  hold with it; config.load and ready.json (the Gemma 4 profile's window layout:
  test_gemma_deferred_completion).
- compose_engine attaches the writer only when the switch is pinned (off: no writer, no wrapper,
  the same worker record); shutdown_engine closes it before the request drain.
- The bytes: iter_encoded equals encoded on random values (string and other keys, tuples,
  subclasses, floats, deep nesting, small pieces), the copy composes (encoded(copy_receipt(v))),
  the snapshot is taken at completion (later changes to the receipt, the plan or the query do
  not reach the file), pickle's fallback, the encoder's own refusals.
- The writer: jobs wait for post_step, run in order, the goals.jsonl row before its native file,
  with the same calls (open modes, one write, flush, fsync) and the same bytes as the synchronous
  path; joins; held acknowledgements; a write failure (its acknowledgement fails, every later
  call raises, nothing more is written, the failure diagnostic); cancellations and retries run
  the synchronous path after a flush; close and interpreter exit; the pause and its skip while
  the engine waits.
- The engine touch points (attach_engine): post_step releases, a Cancel of a completed request
  whose files are pending answers ``Held`` and its acknowledgement goes out after the files,
  delivered by the lifecycle tick; admission, close and an idle release join the session.
- End to end (CPU fakes): test_decouple_e2e's three sessions decoding together (mswea-xml;
  16-token and 784-token pages; full, observation_query, goal_query, random50; with the three
  decode fast paths) and test_idle_release's sixteen sessions under pool pressure with
  --release-idle-pages and the three fast paths write byte-identical evidence files, send the
  same RPCs and commit the same replies with the switch on and off. The batched API servers of
  test_ms2_e2e (mswea-xml), test_kv_session_e2e (kv-session) and test_chat_text_e2e (chat-text)
  run their scenarios with the switch through the packaged router and the frontend's
  authenticated Cancel: every check of those tests holds, every file the writer wrote equals the
  synchronous path's bytes taken at the same moment (a shadow directory), and with a slow writer
  the acknowledgements are held until the files are complete.
- The step profile's completion sub-scopes and the writer's totals.
The longer cases run with REMOTE_RA_SLOW_TESTS=1.
"""
import asyncio
import contextlib
from contextlib import ExitStack
import copy
import enum
from hashlib import sha256
import io
import itertools
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import serve  # noqa: E402

from remote_ra_http import config as http_config  # noqa: E402
from remote_ra_http import evidence as EV  # noqa: E402
from remote_ra_live_v2 import deferred_completion as D  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
KEY = D.CONFIG_KEY
encoded, copy_receipt = http_config.encoded, EV.copy_receipt


def wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('timed out')
        time.sleep(0.002)


# ---- 1. the pin, the launcher and startup -------------------------------------------------------

class PinTests(unittest.TestCase):
    def test_pin_values(self):
        self.assertIsNone(D.pinned(None))
        self.assertIsNone(D.pinned({}))
        self.assertIs(D.pinned({KEY: True}), True)
        for bad in (False, 1, 'true', None):
            with self.subTest(bad=bad), self.assertRaisesRegex(LiveV2Error, 'must be true'):
                D.pinned({KEY: bad})
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            self.assertIsNone(D.loaded())
        # config.load's check raises the runtime's RuntimeError.
        self.assertIs(http_config.deferred_completion_io({KEY: True}), True)
        with self.assertRaisesRegex(RuntimeError, 'must be true'):
            http_config.deferred_completion_io({KEY: False})

    def test_the_encoder_options_are_the_evidence_ones(self):
        values = [dict(b=[1, 2.5, 'é\n"', None, True], a={'z': (1, 2)}), [], {}, 'x', 1e300, -0.0, 2 ** 70]
        for value in values:
            self.assertEqual(D._enc(value), encoded(value))
        with self.assertRaises(ValueError):
            D._enc(float('nan'))


class LauncherTests(unittest.TestCase):
    """test_integrate_flags' bases and switches, with --deferred-completion-io."""
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        import test_integrate_flags as IF
        cls.IF = IF
        cls.fixture = type('Fixture', (IF.Release,), {})
        cls.fixture.setUpClass()
        cls.release = cls.fixture()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def test_every_base_and_switch_combination_adds_only_its_key(self):
        IF, release = self.IF, self.release
        combinations = [(), ('decode_fastpath', 'decode_fastpath2', 'decode_fastpath3'), ('release_idle_pages',),
                        ('language_model_only', 'release_idle_pages'), IF.SWITCHES]
        if SLOW:
            combinations = IF.COMBINATIONS
        accepted = 0
        for base in IF.BASES:
            for combination in combinations:
                refused = IF.refusal(base, combination)
                with self.subTest(base=base, combination=combination):
                    on = release.launched(base, combination, deferred_completion_io=True)
                    if refused is not None:
                        self.assertEqual(on[0], 'ValueError')
                        self.assertIn(refused, on[1])
                        continue
                    off = release.launched(base, combination)
                    self.assertEqual((on[0], off[0]), ('ok', 'ok'), (on, off))
                    m1, c1, a1, e1 = on[1]
                    m0, c0, a0, e0 = off[1]
                    self.assertEqual((m1, a1), (m0, a0))                   # no vLLM argument
                    self.assertIs(c1.pop(KEY), True)
                    self.assertNotIn(KEY, c0)
                    self.assertEqual(c1, c0)
                    self.assertEqual(e1, dict(e0, REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(dict(c1, **{KEY: True}))))
                    accepted += 1
        self.assertGreaterEqual(accepted, 25)

    def test_refused_as_a_non_switch_and_with_the_other_refusals(self):
        release = self.release
        status, message = release.launched('xml-legacy', deferred_completion_io=1)
        self.assertEqual((status, message), ('ValueError', '--deferred-completion-io is a switch'))
        for base, combination, refused in (('xml-legacy', ('decode_fastpath2',), 'requires --decode-fastpath'),
                                           ('xml-legacy', ('release_idle_pages',), 'needs the decoupled KV layout')):
            outcome = release.launched(base, combination, deferred_completion_io=True)
            self.assertEqual(outcome[0], 'ValueError')
            self.assertIn(refused, outcome[1])

    def test_next_to_the_goal_rewriter_batch(self):
        IF, release = self.IF, self.release
        model = release.root / 'qwen3-0.6b'
        model.mkdir(exist_ok=True)
        (model / 'config.json').write_bytes(b'{"model_type":"qwen3"}')
        tasks = IF.tasks(4)
        selection = dict(goal_rewriter_model=str(model))
        kw = dict(admission_overlap=4, goal_rewriter_batch=4)
        _, on, on_command, _ = release.prepare(tasks, 'goal_query', 24, 16, None, dict(selection), out='grb-on',
                                               deferred_completion_io=True, **kw)
        _, off, off_command, _ = release.prepare(tasks, 'goal_query', 24, 16, None, dict(selection), out='grb-off',
                                                 **kw)
        strip = lambda value, out: json.loads(json.dumps(value).replace(json.dumps(str(release.root / out))[1:-1], 'OUT'))
        self.assertEqual((on['goal_rewriter_batch'], on.pop(KEY)), (4, True))
        self.assertEqual(strip(on, 'grb-on'), strip(off, 'grb-off'))
        self.assertEqual(strip(on_command, 'grb-on'), strip(off_command, 'grb-off'))

    def test_dry_run_main_and_load(self):
        IF, release = self.IF, self.release
        task = release.root / 'task.json'
        task.write_bytes(json.dumps(IF.tasks(1)[0]).encode())
        commands = []
        for extra in ([], ['--deferred-completion-io']):
            argv = ['serve.py', '--no-speed-defaults', '--release-dir', str(release.release), '--task', str(task), '--policy', 'full',
                    '--output', str(release.root / ('main%d' % len(commands))), '--port', '8231'] + extra
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()) as out, \
                    patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                serve.main()
            value = json.loads(out.getvalue().strip().splitlines()[-1])
            self.assertEqual(value['status'], 'SERVICE_DRY_RUN')
            commands.append([str(c).replace(str(release.root / ('main%d' % len(commands))), '<out>')
                             for c in value['command']])
        self.assertEqual(commands[0], commands[1])
        _, config, _, _ = release.launch('xml-pool-graphs', IF.SWITCHES, deferred_completion_io=True)
        with release.startup(config):
            loaded = http_config.load()
            self.assertIs(loaded[KEY], True)
            self.assertIs(D.loaded(), True)
        for bad in (False, 0, 'yes'):
            with self.subTest(bad=bad), release.startup(dict(config, **{KEY: bad})), \
                    self.assertRaisesRegex(RuntimeError, 'deferred_completion_io must be true'):
                http_config.load()

    def test_ready_records_the_switch(self):
        IF = self.IF
        written = {}
        for on in (False, True):
            config = self.release.load(self.release.launch('kv-obs-262144', IF.SWITCHES, out='ready%d' % on,
                                                           **({KEY: True} if on else {}))[1])
            written[on] = ready_json(config)
        self.assertIs(written[True].pop(KEY), True)
        self.assertNotIn(KEY, written[False])
        differ = ('server_run_id', 'common_config_sha256', 'input_pins_sha256')
        self.assertEqual({k: v for k, v in written[True].items() if k not in differ},
                         {k: v for k, v in written[False].items() if k not in differ})


def ready_json(config):
    """FrontendEvidence.write_ready's ready.json for a loaded startup config
    (test_integrate_flags.ReadyTests' stand-ins: the authority, /proc, the worker evidence)."""
    import pathlib
    import test_integrate_flags as IF
    import test_ms_frontend as MF
    readiness = dict(native_ready=True)
    authority = NS(identity=NS(epoch=MF.EPOCH, manifest='m' * 64), policy=config['policy'],
                   session_id=MF.SERVER, verify_admission=lambda: True, live_v2_readiness=readiness,
                   loaded=NS(manifest=dict(qualification_parent_manifest='q' * 64)))
    read_text = pathlib.Path.read_text

    def proc(path, *args, **kwargs):
        name = str(path).replace('\\', '/')
        if name == '/proc/sys/kernel/random/boot_id':
            return 'boot\n'
        if name == '/proc/self/stat':
            return '1 (python) ' + ' '.join(['0'] * 30)
        return read_text(path, *args, **kwargs)
    record = IF.engine_record(config)[0]
    receipt = dict(tensor_geometry={}, source_closure={}, context=record['context'],
                   **{name: IF.ECHOED[name] for name in IF.SWITCHES})
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        value = dict(config, output_directory=str(output))
        (output / 'worker-attached.json').write_bytes(json.dumps(dict(
            epoch=MF.EPOCH, report=readiness, receipt=receipt,
            kv_pools=dict(kv_layout=config['kv_layout'], num_blocks_by_pool=[4, 4],
                          free_blocks_by_pool=[4, 4]))).encode())
        with patch('remote_ra_http.evidence.load', lambda: value), patch.object(pathlib.Path, 'read_text', proc):
            evidence = EV.FrontendEvidence(authority, None, dict(config['effective_server_config']))
            evidence.write_ready(NS(routes=[NS(path='/v1/chat/completions', methods={'POST'})]), NS(started=True))
        return json.loads((output / 'ready.json').read_bytes())


# ---- 2. exact bytes in pieces --------------------------------------------------------------------

class Color(enum.IntEnum):
    RED = 1


class Text(str):
    pass


def random_value(rng, depth=0):
    """A random JSON-able value: dicts (string keys mostly, sometimes ints, floats, bools or
    None), lists and tuples, long and short runs of scalars, unicode and escapes, floats."""
    pick = rng.random()
    if depth > 5 or pick < 0.35:
        kind = rng.randrange(9)
        if kind == 0:
            return rng.randrange(-10 ** 6, 10 ** 6)
        if kind == 1:
            return rng.choice([0.0, -0.0, 1e-300, 1e300, 0.1, 2.5, -7.25e-5, float(rng.randrange(10 ** 9))])
        if kind == 2:
            return ''.join(rng.choice('ab"\\\n\t\x01é€😀 /<>') for _ in range(rng.randrange(0, 90)))
        if kind == 3:
            return rng.choice([True, False, None])
        if kind == 4:
            return 2 ** rng.randrange(60, 80)
        if kind == 5:
            return Text('sub')
        if kind == 6:
            return Color.RED
        return rng.randrange(100)
    # Large containers near the top only (the total stays some tens of thousands of leaves).
    n = rng.choice([0, 1, 2, 3, 5, 8, 40, 300, 3000] if depth == 0 else [0, 1, 2, 3, 5, 8, 40, 300]
                   if depth == 1 else [0, 1, 2, 3, 5])
    if pick < 0.6:
        items = [random_value(rng, depth + 1) for _ in range(n)]
        if rng.random() < 0.3:
            items = [rng.randrange(1000) for _ in range(n)]           # a run of scalars
        return tuple(items) if rng.random() < 0.2 else items
    keys = rng.random()
    out = {}
    for k in range(n):
        if keys < 0.85:
            key = rng.choice(['k%d' % k, 'é%d' % k, 'a"b%d' % k, '%05d' % k, 'Z%d' % k])
        else:
            key = rng.choice([k, float(k) + 0.5, None, True, 'x%d' % k])
        out[key] = random_value(rng, depth + 1)
    return out


class PieceTests(unittest.TestCase):
    def check(self, value, caps=(1, 2, 7, 64, D.PIECE)):
        try:
            whole = encoded(value)
        except (TypeError, ValueError) as error:
            for cap in caps:
                with self.assertRaises(type(error)):
                    b''.join(D.iter_encoded(value, cap))
            return
        for cap in caps:
            self.assertEqual(b''.join(D.iter_encoded(value, cap)), whole, cap)
        try:
            copied = encoded(copy_receipt(value))
        except (TypeError, ValueError):
            return
        for cap in caps:
            self.assertEqual(b''.join(D.iter_encoded(value, cap, leaf=D._copy_enc)), copied, cap)

    def test_random_values(self):
        rng = random.Random(20261019)
        for _ in range(300 if SLOW else 30):
            self.check(random_value(rng))

    def test_shapes_of_a_native_record(self):
        # The pieces of real records: long token lists, per-step frames, a phase-1 frame with
        # vectors and Page maps, a drain receipt with sealed Pages and releases.
        prompt = list(range(5000))
        frame = dict(operation='allocation_update', http_measurement=dict(before=dict(h2d_issued=1), after={},
                                                                          elapsed_ns=12, active_owned_kv_bytes=3))
        phase1 = dict(operation='phase1', observation_vector=[0.125 * k for k in range(5120)],
                      page_vectors={str(p): [p, 0.5, -1.5] for p in range(600)},
                      worker_receipt=dict(body=dict(position_evidence=[dict(rows=[list(range(3000))] * 3)])))
        drain = dict(dram=list(range(16000)), release=[[k, k + 1] for k in range(2000)],
                     new_pages={str(p): dict(token_span=[16 * p, 16 * p + 16], content='c' * 64) for p in range(600)},
                     resident=tuple(range(50)), carry=dict(computed_end=99, digest='d' * 64))
        value = dict(prompt_token_ids=prompt, worker_frames=[phase1] + [frame] * 1100, drain=drain,
                     plan=dict(selected=list(range(512))), query=None, turn=3)
        self.check(value)
        self.check(drain)

    def test_other_keys_subclasses_and_refusals_are_the_encoder_own(self):
        cases = [{1: 'a', 10: 'b', 2: 'c'}, {None: 1, True: 2, 1.5: 3}, {'b': {3: [1] * 3000, 20: 'x'}},
                 [Text('s')] * 3000, {Text('k%d' % k): k for k in range(3000)}, [Color.RED] * 2500,
                 [(1, 2)] * 2100, {'a': float('nan')}, {'a': [float('inf')] * 5000}, [object()] * 3,
                 {('t', 1): 2}, [[]] * 5000, [{}] * 5000]
        for value in cases:
            with self.subTest(value=repr(value)[:60]):
                self.check(value)

    def test_deep_and_circular(self):
        deep = []
        node = deep
        for _ in range(60):
            child = [list(range(3000))]
            node.append(child)
            node = child
        self.check(deep)
        loop = [list(range(5000))]
        loop.append(loop)
        with self.assertRaisesRegex(ValueError, 'Circular'):
            b''.join(D.iter_encoded(loop))

    def test_long_strings(self):
        for value in (['x' * 200000, 'y' * 300000] + list(range(10)), [['x' * 200000], {'y': 'z' * 300000}] * 3):
            self.check(value, caps=(2, D.PIECE))


class SnapshotTests(unittest.TestCase):
    def record(self):
        receipt = dict(operation='complete', drained=True, dram=list(range(4000)),
                       new_pages={str(p): dict(token_span=[p, p + 1], binding='b') for p in range(2500)},
                       release=((1, 2), (3, 4)), int_keys={3: 'x', 20: 'y'})
        value = dict(schema='remote-ra-native-http-terminal-v1', request_id='r1', turn=2, rebuild=False,
                     prompt_token_ids=list(range(9000)),
                     worker_frames=[dict(operation='allocation_update', n=k) for k in range(3000)],
                     plan=dict(selected=list(range(700)), restore=[1, 2]), query=dict(body=dict(scores={
                         str(p): p / 7 for p in range(3000)}), mac='m'),
                     idle_release=dict(evicted_sessions=['s1'], evicted_pages=4, readmitted_pages=[]),
                     observation_span=(5, 9), admission_engine_ms=1.25)
        return value, receipt

    def expected(self, value, receipt):
        return encoded(dict(value, drain=copy_receipt(receipt)))

    def test_snapshot_assembles_the_synchronous_bytes(self):
        value, receipt = self.record()
        want = self.expected(value, receipt)
        parts = D.snapshot(dict(value, drain=D.drain_snapshot(receipt)))
        ticks = []
        self.assertEqual(D.assemble(parts, tick=lambda: ticks.append(1)), want)
        self.assertGreater(len(ticks), 10)
        self.assertEqual([k for k, _ in parts], [encoded(k) for k in sorted(dict(value, drain=0))])

    def test_later_changes_do_not_reach_the_file(self):
        value, receipt = self.record()
        want = self.expected(value, receipt)
        parts = D.snapshot(dict(value, drain=D.drain_snapshot(receipt)))
        # The engine's owners change what they hold after the completion; only the record's own
        # frames and prompt list are not copied (nothing else holds them).
        receipt['dram'].append(-1)
        receipt['new_pages']['0']['binding'] = 'changed'
        receipt['int_keys'][4] = 'z'
        value['plan']['selected'].clear()
        value['query']['body']['scores']['0'] = 99
        value['idle_release']['evicted_sessions'].append('s2')
        self.assertEqual(D.assemble(parts), want)

    def test_drain_with_other_keys_and_pickle_fallback(self):
        class Local(int):                                    # JSON encodes it; pickle cannot find it
            pass
        for receipt in ({1: [1, 2], 10: 3, 2: [4] * 3000}, [dict(x=1)] * 3000, dict(a=Local(5), b=[Local(1)] * 3000),
                        dict(a=(1, {2: 3}), b='s')):
            with self.subTest(receipt=repr(receipt)[:50]):
                value = dict(request_id='r', plan=dict(local=Local(3)), worker_frames=[1], prompt_token_ids=[2])
                want = encoded(dict(value, drain=copy_receipt(receipt)))
                self.assertEqual(D.assemble(D.snapshot(dict(value, drain=D.drain_snapshot(receipt)))), want)

    def test_a_refusal_of_the_encoder_raises_when_its_part_is_taken_or_written(self):
        class Opaque:
            pass
        with self.assertRaises(TypeError):
            D.snapshot(dict(plan=dict(x=Opaque())))           # unpicklable and not JSON: raised now
        parts = D.snapshot(dict(worker_frames=[float('nan')]))
        with self.assertRaises(ValueError):
            D.assemble(parts)                                 # a private part: raised on the writer


# ---- 3. the writer -----------------------------------------------------------------------------------

class Rig:
    """A writer over a fake endpoint: evidence.terminal_value patched to give each request's
    record (with ``drain`` of its receipt, as the real one); ``expected(rid)`` is the synchronous
    path's native bytes."""

    def __init__(self, case, *, decoupled=False, goals=False, **writer):
        self.case = case
        self.output = Path(case.enterContext(tempfile.TemporaryDirectory()))
        self.values, self.receipts, self.expected_bytes = {}, {}, {}
        self.texts = NS(pending={}, written={}) if goals else None
        selection = NS(goal_texts=self.texts) if goals else NS()
        self.writer = D.CompletionWriter(str(self.output), decoupled, **writer)
        self.core = NS(_remote_ra_private_v2_endpoint=NS(selection=selection),
                       _remote_ra_completion_writer=self.writer)
        self.endpoint = NS(core=self.core)
        self.config = dict(output_directory=str(self.output))
        values = self.values

        def terminal_value(endpoint, entry, operation, drain=None, frames=None):
            # frames: the record's worker_frames are plain lists here (host-trims-1111 takes them as filed)
            rid = entry['binding'].request
            value = copy.deepcopy(values[rid])
            receipt = entry['drain_receipt']
            value.update(operation=operation, terminal_state=entry['state'],
                         drain=copy_receipt(receipt) if drain is None else drain(receipt))
            return value
        case.enterContext(patch.object(EV, 'terminal_value', terminal_value))
        case.enterContext(patch.object(EV, 'load', lambda: self.config))
        case.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        case.addCleanup(self.close)

    def close(self):
        if self.writer.thread is not None and not self.writer.closed:
            try:
                self.writer.close()
            except Exception:
                pass

    def entry(self, rid, session='s0', size=200, state='complete'):
        self.values[rid] = dict(schema='remote-ra-native-http-terminal-v1', request_id=rid, session_id=session,
                                session_index=int(session[1:]), epoch='e' * 64, local_turn=0,
                                turn=len(self.values), prompt_token_ids=list(range(size)),
                                worker_frames=[dict(operation='allocation_update', step=k) for k in range(size)],
                                plan=dict(selected=list(range(size // 4))), query=None)
        self.receipts[rid] = dict(operation='complete', drained=True, dram=list(range(size)), binding='b-' + rid)
        if self.texts is not None:
            self.texts.pending[rid] = dict(goal='g-' + rid, text='t' * 50)
        return dict(binding=NS(request=rid, session=session), drain_receipt=self.receipts[rid], state=state)

    def path(self, rid):
        return self.output / ('native-' + sha256(rid.encode()).hexdigest() + '.json')

    def expected(self, rid, goal=None):
        value = dict(self.values[rid], operation='complete', terminal_state='complete',
                     drain=copy_receipt(self.receipts[rid]))
        if goal is not None:
            value['goal_text'] = goal
        return encoded(value)

    def goal_line(self, rid, record, state='complete'):
        from remote_ra_live_v2.goal_text import SCHEMA, KEYS
        value = dict(self.values[rid], operation=state, terminal_state=state,
                     drain=copy_receipt(self.receipts[rid]))
        raw = encoded(dict({k: value[k] for k in KEYS}, schema=SCHEMA, **record))
        return raw, dict(file='goals.jsonl', sha256=sha256(raw).hexdigest())


class WriterTests(unittest.TestCase):
    maxDiff = None

    def test_staged_until_released_then_written_in_order_with_the_same_bytes(self):
        rig = Rig(self, goals=True)
        writer = rig.writer.start()
        rids = ['r%d' % k for k in range(5)]
        lines = {}
        for k, rid in enumerate(rids):
            entry = rig.entry(rid, session='s%d' % (k % 2), size=300 + 50 * k)
            lines[rid] = rig.goal_line(rid, rig.texts.pending[rid])
            EV.native_terminal(rig.endpoint, entry, 'complete')        # the dispatch in native_terminal
            self.assertEqual(rig.texts.written[rid], lines[rid][1])     # named now, as goal_text names it
        time.sleep(0.05)
        self.assertEqual(sorted(p.name for p in rig.output.iterdir()), [])      # nothing before post_step
        self.assertEqual(writer.stats()['staged'], 5)
        writer.release()
        writer.flush()
        for rid in rids:
            self.assertEqual(rig.path(rid).read_bytes(), rig.expected(rid, lines[rid][1]))
        self.assertEqual((rig.output / 'goals.jsonl').read_bytes(), b''.join(lines[r][0] + b'\n' for r in rids))
        stats = writer.stats()
        self.assertEqual((stats['jobs'], stats['pending'], stats['errors']), (5, 0, 0))
        self.assertEqual(stats['bytes'], sum(len(rig.expected(r, lines[r][1])) for r in rids))

    def test_the_same_calls_as_goal_text_and_save(self):
        rig = Rig(self, goals=True)
        writer = rig.writer.start()
        calls, original_open, original_fsync = [], Path.open, os.fsync

        def recording_open(path, mode='r', *args, **kwargs):
            stream = original_open(path, mode, *args, **kwargs)
            if threading.current_thread() is writer.thread:
                write = stream.write
                calls.append(('open', path.name, mode))
                stream = Recorded(stream, lambda data: calls.append(('write', path.name, len(data))) or write(data),
                                  lambda: calls.append(('flush', path.name)))
            return stream

        def recording_fsync(fd):
            if threading.current_thread() is writer.thread:
                calls.append(('fsync',))
            return original_fsync(fd)
        rid = 'r0'
        EV.native_terminal(rig.endpoint, rig.entry(rid, size=50), 'complete')
        line, reference = rig.goal_line(rid, dict(goal='g-r0', text='t' * 50))
        with patch.object(Path, 'open', recording_open), patch.object(D.os, 'fsync', recording_fsync):
            writer.release()
            writer.flush()
        native = rig.path(rid).name
        self.assertEqual(calls, [('open', 'goals.jsonl', 'ab'), ('write', 'goals.jsonl', len(line) + 1),
                                 ('flush', 'goals.jsonl'), ('fsync',),
                                 ('open', native, 'xb'), ('write', native, len(rig.expected(rid, reference))),
                                 ('flush', native), ('fsync',)])

    def test_joins_wait_for_their_request_or_session_only(self):
        gate = threading.Event()
        rig = Rig(self, pause=lambda: gate.wait(), slice_s=0.0)
        writer = rig.writer.start()
        EV.native_terminal(rig.endpoint, rig.entry('a', session='s1', size=3000), 'complete')
        writer.release()
        wait_until(lambda: writer.stats()['pauses'] > 0)     # 'a' is being encoded, held at a pause
        EV.native_terminal(rig.endpoint, rig.entry('b', session='s2', size=10), 'complete')
        writer.join_session('s3')                            # nothing pending there: returns at once
        joined = []
        thread = threading.Thread(target=lambda: (writer.join_session('s1'), joined.append('s1')))
        thread.start()
        time.sleep(0.05)
        self.assertEqual(joined, [])
        gate.set()
        thread.join(5)
        self.assertEqual(joined, ['s1'])
        writer.join_request('b')                             # releases 'b' first
        self.assertEqual(rig.path('b').read_bytes(), rig.expected('b'))

    def test_a_join_skips_the_pauses(self):
        pauses = []
        rig = Rig(self, pause=lambda: pauses.append(1) or time.sleep(0.001), slice_s=0.0)
        writer = rig.writer.start()
        EV.native_terminal(rig.endpoint, rig.entry('a', size=20000), 'complete')
        writer.release()
        writer.join_request('a')
        self.assertLess(len(pauses), 5)            # the engine thread waited: no pause after it did
        EV.native_terminal(rig.endpoint, rig.entry('b', size=20000), 'complete')
        writer.release()
        wait_until(lambda: not writer.pending)
        self.assertGreater(len(pauses), 10)
        self.assertEqual(rig.path('b').read_bytes(), rig.expected('b'))

    def test_a_held_acknowledgement_goes_out_after_its_files(self):
        gate = threading.Event()
        woken = []
        rig = Rig(self, pause=lambda: gate.wait(), slice_s=0.0, wake=lambda: woken.append(threading.get_ident()))
        writer = rig.writer.start()
        EV.native_terminal(rig.endpoint, rig.entry('r', size=3000), 'complete')
        self.assertTrue(writer.hold('r'))          # releases it too
        self.assertFalse(writer.hold('other'))
        sent = []
        send = lambda call_id, frame, failure: sent.append((call_id, frame, failure, rig.path('r').exists()))
        writer.held_ack(send, 7, D.Held('r', b'frame'), None)
        time.sleep(0.05)
        writer.deliver()
        self.assertEqual(sent, [])
        gate.set()
        wait_until(lambda: writer.ready)
        self.assertEqual(woken, [writer.thread.ident])       # the writer woke the engine loop
        writer.deliver()
        self.assertEqual(sent, [(7, b'frame', None, True)])
        # Written already: sent at once, on this thread.
        writer.held_ack(send, 8, D.Held('r', b'again'), None)
        self.assertEqual(sent[-1], (8, b'again', None, True))
        self.assertEqual(writer.stats()['held'], 2)

    def test_a_write_failure_fails_its_acknowledgement_and_everything_after_it(self):
        failures = []
        rig = Rig(self, goals=True)
        writer = rig.writer.start()
        rig.path('bad').write_bytes(b'{}')                   # 'xb' refuses: the write fails
        with patch('remote_ra_http.diagnostics.failure', lambda event, exc, owner=None: failures.append(event)):
            EV.native_terminal(rig.endpoint, rig.entry('bad'), 'complete')
            EV.native_terminal(rig.endpoint, rig.entry('next'), 'complete')
            sent = []
            self.assertTrue(writer.hold('bad'))
            writer.held_ack(lambda *a: sent.append(a), 1, D.Held('bad', b'frame'), None)
            wait_until(lambda: not writer.pending)
            writer.deliver()
        self.assertEqual(failures, ['COMPLETION_WRITE_FAILED'])
        (call_id, frame, failure), = sent
        self.assertEqual((call_id, frame), (1, None))
        self.assertRegex(str(failure), 'COMPLETION_WRITE_FAILED: FileExistsError')
        self.assertFalse(rig.path('next').exists())          # nothing written after the failure
        self.assertEqual((rig.output / 'goals.jsonl').read_bytes().count(b'\n'), 1)   # 'bad''s row, then nothing
        for call in (lambda: EV.native_terminal(rig.endpoint, rig.entry('later'), 'complete'),
                     writer.flush, lambda: writer.join_session('s0'), lambda: writer.hold('next'), writer.close):
            with self.assertRaisesRegex(RuntimeError, 'COMPLETION_WRITE_FAILED'):
                call()
        self.assertFalse(writer.thread.is_alive())
        self.assertEqual(writer.stats()['errors'], 1)

    def test_cancellations_and_retries_run_the_synchronous_path_after_every_write(self):
        gate = threading.Event()
        rig = Rig(self, goals=True, pause=lambda: gate.wait(), slice_s=0.0)
        writer = rig.writer.start()
        EV.native_terminal(rig.endpoint, rig.entry('done', size=3000), 'complete')
        writer.release()
        wait_until(lambda: writer.stats()['pauses'] > 0)
        line, reference = rig.goal_line('done', dict(goal='g-done', text='t' * 50))
        cancel = rig.entry('cancelled', state='cancelled')
        threading.Timer(0.05, gate.set).start()
        EV.native_terminal(rig.endpoint, cancel, 'cancel')   # waits for 'done', then writes itself
        self.assertEqual(rig.path('done').read_bytes(), rig.expected('done', reference))
        value = dict(rig.values['cancelled'], operation='cancel', terminal_state='cancelled',
                     drain=copy_receipt(rig.receipts['cancelled']))
        cancel_line, cancel_reference = rig.goal_line('cancelled', dict(goal='g-cancelled', text='t' * 50), 'cancelled')
        value = dict(value, goal_text=cancel_reference)
        self.assertEqual(rig.path('cancelled').read_bytes(), encoded(value))
        # goals.jsonl keeps the order of the terminals.
        self.assertEqual((rig.output / 'goals.jsonl').read_bytes().splitlines(), [line, cancel_line])
        # A second terminal of a completed request: the synchronous retry check (unchanged: accepted).
        retry = dict(binding=NS(request='done', session='s0'), drain_receipt=rig.receipts['done'], state='complete')
        EV.native_terminal(rig.endpoint, retry, 'complete')
        changed = dict(retry, drain_receipt=dict(rig.receipts['done'], drained=True, extra=1))
        with self.assertRaisesRegex(RuntimeError, 'terminal receipt changed on retry'):
            EV.native_terminal(rig.endpoint, changed, 'complete')

    def test_close_stops_the_writer_and_later_terminals_are_synchronous(self):
        rig = Rig(self)
        writer = rig.writer.start()
        EV.native_terminal(rig.endpoint, rig.entry('a', size=2000), 'complete')
        import atexit
        with patch.object(atexit, 'unregister', wraps=atexit.unregister) as unregister:
            writer.close()
        unregister.assert_called_once_with(writer._at_exit)
        self.assertEqual(rig.path('a').read_bytes(), rig.expected('a'))      # staged, flushed by close
        self.assertFalse(writer.thread.is_alive())
        EV.native_terminal(rig.endpoint, rig.entry('b'), 'complete')
        self.assertEqual(rig.path('b').read_bytes(), rig.expected('b'))      # written before it returned
        writer.close()                                                         # again: nothing

    def test_interpreter_exit_writes_what_is_pending(self):
        rig = Rig(self)
        writer = rig.writer.start()
        EV.native_terminal(rig.endpoint, rig.entry('a', size=2000), 'complete')
        writer._at_exit()                     # what atexit runs if the process ends without shutdown
        self.assertEqual(rig.path('a').read_bytes(), rig.expected('a'))

    def test_engine_thread_only_and_the_decoupled_pools(self):
        rig = Rig(self, decoupled=True)
        writer = rig.writer.start()
        free = [[7, 3]]
        with patch.object(EV, 'kv_pools', lambda core: dict(free_blocks_by_pool=free[0])):
            EV.native_terminal(rig.endpoint, rig.entry('a'), 'complete')
            free[0] = [1, 1]                  # read at the completion, not when written
            writer.flush()
        value = dict(rig.values['a'], operation='complete', terminal_state='complete',
                     drain=copy_receipt(rig.receipts['a']), kv_pools=dict(free_blocks_by_pool=[7, 3]))
        self.assertEqual(rig.path('a').read_bytes(), encoded(value))
        errors = []
        thread = threading.Thread(target=lambda: errors.append(
            self.assertRaises(LiveV2Error, writer.defer, rig.endpoint, rig.entry('b'), 'complete')))
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)


class Recorded:
    """A file stream whose write and flush are observed (WriterTests)."""

    def __init__(self, stream, write, flush):
        self.stream, self.write_hook, self.flush_hook = stream, write, flush

    def write(self, data):
        return self.write_hook(data)

    def flush(self):
        self.flush_hook()
        return self.stream.flush()

    def fileno(self):
        return self.stream.fileno()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self.stream.__exit__(*exc)


# ---- 4. the engine touch points --------------------------------------------------------------------

class FakeBridge:
    """RegisteredRouteBridge's entry points (the calls are recorded)."""

    def __init__(self, lifecycle, calls):
        self.lifecycle, self.calls = lifecycle, calls

    def cancel(self, dispatcher, command):
        self.calls.append(('cancel', command.request_id))
        return NS(state=command.state)

    def add_steps(self, dispatcher, command, fingerprint):
        self.calls.append(('add', command.session_id))
        if False:
            yield
        return ('accepted', command.session_id)

    def close_session(self, dispatcher, command):
        self.calls.append(('close', command.session_id))
        return ('closed', command.session_id)


class FakeDispatcher:
    """RegisteredAddDispatcher.handle_steps for three commands; ``handle`` drives it."""

    def __init__(self, bridge):
        self._remote_ra_live_v2 = bridge

    def handle_steps(self, frame):
        if False:
            yield
        if frame.kind == 'cancel':
            return ('ack', self._remote_ra_live_v2.cancel(self, frame).state)
        if frame.kind == 'add':
            return (yield from self._remote_ra_live_v2.add_steps(self, frame, 'fp'))
        return self._remote_ra_live_v2.close_session(self, frame)

    def handle(self, frame):
        steps = self.handle_steps(frame)
        try:
            next(steps)
        except StopIteration as done:
            return done.value
        raise AssertionError('nothing parks here')


class AttachTests(unittest.TestCase):
    maxDiff = None

    def attach(self, **writer):
        rig = Rig(self, **writer)
        self.calls, self.sent, self.ticks, self.evicted = [], [], [], []
        test = self

        class Loop:
            def post_step(self, executed):
                test.ticks.append(('post', executed))

            def _remote_ra_ack(self, call_id, frame, failure):
                test.sent.append((call_id, frame, failure, sorted(p.name for p in rig.output.iterdir())))
        self.loop, self.ticker = Loop(), NS(tick=lambda *a: self.ticks.append('tick'))
        lifecycle = NS()
        self.dispatcher = FakeDispatcher(FakeBridge(lifecycle, self.calls))
        rig.core._remote_ra_completion_writer = None
        rig.core._remote_ra_idle_release = NS(evict=lambda sid, rid, short: self.evicted.append(sid) or ('pages',))
        self.woken = []
        writer = D.attach_engine(rig.core, lifecycle, self.dispatcher, ticker=self.ticker,
                                 wake=lambda: self.woken.append(1), loop=self.loop,
                                 config=dict(output_directory=str(rig.output), **{KEY: True}), **writer)
        rig.writer = writer
        self.assertIs(rig.core._remote_ra_completion_writer, writer)
        self.assertIsInstance(writer, D.CompletionWriter)
        self.assertTrue(writer.thread.is_alive())
        return rig, writer

    def test_post_step_releases_after_the_outputs(self):
        rig, writer = self.attach()
        EV.native_terminal(rig.endpoint, rig.entry('r'), 'complete')
        time.sleep(0.05)
        self.assertFalse(rig.path('r').exists())
        self.loop.post_step(True)
        self.assertEqual(self.ticks, [('post', True)])          # the loop's own post_step ran too
        wait_until(lambda: not writer.pending)
        self.assertEqual(rig.path('r').read_bytes(), rig.expected('r'))

    def test_a_cancel_of_a_completed_request_is_acknowledged_after_its_files(self):
        gate = threading.Event()
        rig, writer = self.attach(pause=lambda: gate.wait(), slice_s=0.0)
        EV.native_terminal(rig.endpoint, rig.entry('r', size=3000), 'complete')
        self.loop.post_step(True)
        reply = self.dispatcher.handle(NS(kind='cancel', request_id='r', state='complete'))
        self.assertIsInstance(reply, D.Held)
        self.assertEqual((reply.request, reply.reply), ('r', ('ack', 'complete')))
        self.loop._remote_ra_ack(9, reply, None)
        self.ticker.tick()
        self.assertEqual(self.sent, [])                         # its files are not written yet
        gate.set()
        wait_until(lambda: writer.ready)
        self.assertEqual(self.woken, [1])
        self.ticker.tick()                                      # the input pass delivers it
        self.assertEqual(self.sent, [(9, ('ack', 'complete'), None, [rig.path('r').name])])
        self.assertEqual(self.ticks[-1], 'tick')
        # Nothing pending (or not complete): the reply as it is, at once.
        for state in ('complete', 'cancelled'):
            reply = self.dispatcher.handle(NS(kind='cancel', request_id='r', state=state))
            self.assertEqual(reply, ('ack', state))
        self.loop._remote_ra_ack(10, ('ack', 'cancelled'), None)
        self.assertEqual(self.sent[-1][:3], (10, ('ack', 'cancelled'), None))
        # Other commands never hold.
        self.assertEqual(self.dispatcher.handle(NS(kind='close', session_id='s0')), ('closed', 's0'))

    def test_admission_close_and_release_join_their_session(self):
        gate = threading.Event()
        rig, writer = self.attach(pause=lambda: gate.wait(), slice_s=0.0)
        EV.native_terminal(rig.endpoint, rig.entry('r', session='s1', size=3000), 'complete')
        self.loop.post_step(True)
        wait_until(lambda: writer.stats()['pauses'] > 0)
        self.assertEqual(self.dispatcher.handle(NS(kind='add', session_id='s2')), ('accepted', 's2'))
        rig.core._remote_ra_idle_release.evict('s2', 'x', 1)       # another session: at once
        done = []
        actions = (lambda: self.dispatcher.handle(NS(kind='add', session_id='s1')),
                   lambda: self.dispatcher.handle(NS(kind='close', session_id='s1')),
                   lambda: rig.core._remote_ra_idle_release.evict('s1', 'y', 2))
        threads = [threading.Thread(target=lambda a=a: done.append(a())) for a in actions]
        for thread in threads:
            thread.start()
        time.sleep(0.05)
        self.assertEqual(done, [])
        gate.set()
        for thread in threads:
            thread.join(5)
        self.assertEqual(sorted(map(str, done)), sorted(map(str, [('accepted', 's1'), ('closed', 's1'), ('pages',)])))
        self.assertEqual(self.evicted, ['s2', 's1'])

    def test_a_held_acknowledgement_of_a_failed_write_fails(self):
        rig, writer = self.attach()
        rig.path('r').write_bytes(b'x')
        gate = threading.Event()
        writer.pause, writer.slice_s = (lambda: gate.wait()), 0.0
        with patch('remote_ra_http.diagnostics.failure', lambda *a, **k: None), \
                contextlib.redirect_stderr(io.StringIO()):
            EV.native_terminal(rig.endpoint, rig.entry('r', size=3000), 'complete')
            self.loop.post_step(True)
            reply = self.dispatcher.handle(NS(kind='cancel', request_id='r', state='complete'))
            self.loop._remote_ra_ack(3, reply, None)
            gate.set()
            wait_until(lambda: writer.ready)
            self.ticker.tick()
        (call_id, frame, failure, _), = self.sent
        self.assertEqual((call_id, frame), (3, None))
        self.assertRegex(str(failure), 'COMPLETION_WRITE_FAILED')
        # A later Cancel is refused (the dispatcher turns it into a rejected acknowledgement).
        with self.assertRaisesRegex(RuntimeError, 'COMPLETION_WRITE_FAILED'):
            self.dispatcher.handle(NS(kind='cancel', request_id='r', state='complete'))


class ShutdownTests(unittest.TestCase):
    def test_the_writer_closes_before_the_drain_and_its_failure_is_raised(self):
        from remote_ra_http.shutdown import shutdown_engine
        for fails in (False, True):
            order = []

            def close():
                order.append('writer')
                if fails:
                    raise RuntimeError('COMPLETION_WRITE_FAILED: x')
            core = NS(_remote_ra_dynamic_mode=True, _remote_ra_completion_writer=NS(close=close),
                      _remote_ra_assembly=NS(close=lambda: order.append('drain')))
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
                if fails:
                    with self.assertRaises(BaseExceptionGroup):
                        shutdown_engine(core, lambda: order.append('native'))
                else:
                    shutdown_engine(core, lambda: order.append('native'))
            self.assertEqual(order, ['writer', 'drain', 'native'])
        # Off: no writer, the stages of before.
        order = []
        core = NS(_remote_ra_dynamic_mode=True, _remote_ra_assembly=NS(close=lambda: order.append('drain')))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            shutdown_engine(core, lambda: order.append('native'))
        self.assertEqual(order, ['drain', 'native'])


class ComposeTests(unittest.TestCase):
    """compose_engine attaches the writer exactly when the switch is pinned; the worker startup
    record is the same either way (the switch is the engine's alone)."""
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        import test_integrate_flags as IF
        cls.fixture = type('Fixture', (IF.Release,), {})
        cls.fixture.setUpClass()
        cls.release = cls.fixture()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def compose(self, on):
        import queue
        import types
        from remote_ra_live_v2 import composition
        _, config, _, _ = self.release.launch('xml-legacy', out='compose%d' % on, **({KEY: True} if on else {}))
        sent = []
        echo = dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', loaded_sources_verified=True)

        def collective_rpc(method, args, timeout):
            sent.append(args[0])
            return [dict(echo)]

        class Core:
            def post_step(self, executed):
                pass

            def _remote_ra_ack(self, call_id, frame, failure):
                pass
        task = dict(config['task'])
        core = Core()
        assembly = NS(provider=NS(provenance=NS(page_tokens=784)), state='active',
                      lifecycle=NS(bind_native=lambda native: None, tick=lambda force=False: None),
                      dispatcher=NS(_max_records=1, handle_steps=lambda frame: iter(())))
        vars(core).update(_remote_ra_live_v2={}, _remote_ra_assembly=assembly, _remote_ra_trusted_issue=task,
                          _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                          _remote_ra_growing_adapter=NS(layout=None), model_executor=NS(collective_rpc=collective_rpc),
                          scheduler=NS(), vllm_config=NS(scheduler_config=NS(max_num_seqs=1)),
                          input_queue=queue.Queue())
        engine = types.ModuleType('vllm.v1.engine')
        engine.EngineCoreRequestType = NS(WAKEUP='WAKEUP')
        stubs = {'vllm': types.ModuleType('vllm'), 'vllm.v1': types.ModuleType('vllm.v1'), 'vllm.v1.engine': engine}
        with ExitStack() as stack:
            stack.enter_context(self.release.startup(config))
            stack.enter_context(patch.dict(sys.modules, stubs))
            for target, value in (
                    ('remote_ra_live_v2.engine_boundary.EngineRpcBoundary', lambda proxy, adapter: NS(proxy=proxy)),
                    ('remote_ra_live_v2.session_authority.LiveSessionAuthority', lambda *a: NS()),
                    ('remote_ra_live_v2.private_endpoint.SplitPrivateEndpoint',
                     lambda *a: NS(transcript=None, transcripts={}, selection=None)),
                    ('remote_ra_live_v2.goal_async.attach', lambda *a, **k: None),
                    ('remote_ra_http.connector.isolate_connector', lambda core: None),
                    ('remote_ra_live_v2.context_limits.check_engine', lambda *a: None),
                    ('remote_ra_live_v2.context_limits.pin_request_caps', lambda *a, **k: None),
                    ('remote_ra_live_v2.language_model_only.check_engine', lambda *a: None),
                    ('remote_ra_live_v2.connector_guard.check_engine_pool', lambda *a: None),
                    ('remote_ra_live_v2.startup.bind_loaded_worker', lambda *a, **k: None)):
                stack.enter_context(patch(target, value))
            composition.compose_engine(core, object())
        writer = vars(core).get('_remote_ra_completion_writer')
        if writer is not None:
            self.addCleanup(writer.close)
        return core, sent, writer

    def test_attached_only_when_pinned(self):
        off, off_sent, none = self.compose(False)
        on, on_sent, writer = self.compose(True)
        self.assertEqual(off_sent, on_sent)                        # the worker startup record
        self.assertIsNone(none)
        self.assertNotIn('post_step', vars(off))                   # the class's own seams
        self.assertNotIn('_remote_ra_ack', vars(off))
        self.assertEqual(off._remote_ra_assembly.dispatcher._remote_ra_live_v2.cancel.__qualname__,
                         'RegisteredRouteBridge.cancel')
        self.assertIsInstance(writer, D.CompletionWriter)
        self.assertTrue(writer.thread.is_alive())
        self.assertEqual(writer.output, self.release.root / 'compose1')     # the pinned output directory
        for name in ('post_step', '_remote_ra_ack'):
            self.assertIn(name, vars(on))
        bridge = on._remote_ra_assembly.dispatcher._remote_ra_live_v2
        self.assertEqual(bridge.cancel.__name__, 'holding_cancel')
        self.assertEqual(on._remote_ra_assembly.lifecycle.tick.__name__, 'delivering_tick')
        self.assertFalse(writer.decoupled)


# ---- 5. end to end ---------------------------------------------------------------------------------

MISSING = object()


class ShadowWriter(D.CompletionWriter):
    """The writer, and at each completion the synchronous path's files of the same moment in
    ``shadow`` (native_terminal with no writer: the bytes the switch off writes); each file
    the writer writes is compared with them (``compared``, ``mismatches``)."""

    def __init__(self, output, decoupled, *, shadow, **kwargs):
        super().__init__(output, decoupled, **kwargs)
        self.shadow = Path(shadow)
        self.shadow.mkdir(parents=True, exist_ok=True)
        self.expected, self.compared, self.mismatches = {}, 0, []

    def defer(self, endpoint, entry, operation):
        rid = entry['binding'].request
        path = self.output / ('native-' + sha256(rid.encode()).hexdigest() + '.json')
        if operation == 'complete' and not self.closed and path not in self.paths and self.error is None:
            self.expected[path] = self.synchronous(endpoint, entry, operation)
        return super().defer(endpoint, entry, operation)

    def synchronous(self, endpoint, entry, operation):
        core, rid = endpoint.core, entry['binding'].request
        texts = getattr(core._remote_ra_private_v2_endpoint.selection, 'goal_texts', None)
        saved = None if texts is None else (texts.pending.get(rid, MISSING), texts.written.get(rid, MISSING))
        goals = self.shadow / 'goals.jsonl'
        before = goals.read_bytes() if goals.exists() else b''
        load = EV.load
        config = dict(load(), output_directory=str(self.shadow))
        # host-trims-1111 (perf 2.4): the terminal reads the config once per engine
        # (evidence.terminal_config, kept on the core): the shadow's is read for this call only.
        held = getattr(core, '_remote_ra_terminal_config', None)
        core._remote_ra_completion_writer, EV.load = None, (lambda: config)
        core._remote_ra_terminal_config = None
        try:
            EV.native_terminal(endpoint, entry, operation)
        finally:
            core._remote_ra_completion_writer, EV.load = self, load
            core._remote_ra_terminal_config = held
            if texts is not None:
                for table, value in zip((texts.pending, texts.written), saved):
                    if value is MISSING:
                        table.pop(rid, None)
                    else:
                        table[rid] = value
        after = goals.read_bytes() if goals.exists() else b''
        native = (self.shadow / ('native-' + sha256(rid.encode()).hexdigest() + '.json')).read_bytes()
        return native, after[len(before):] or None

    def _write(self, job):
        super()._write(job)
        native, line = self.expected.pop(job.path)
        self.compared += 1
        if job.path.read_bytes() != native or (None if job.row is None else job.row[1]) != line:
            self.mismatches.append(job.path.name)


def deferred_harness(base):
    """A synchronous e2e harness (test_decouple_e2e's MultiHarness family) with the switch, as
    EngineCoreProc carries it: its batched step ends in post_step (the staged jobs go), and a
    native file is read after its request's writes (the frontend reads after the Cancel's
    acknowledgement, which waits for them)."""

    class Harness(base):
        writer = None

        def batch_step(self):
            base.batch_step(self)
            if self.writer is not None:
                self.post_step(True)

        def native_file(self, rid):
            if self.writer is not None:
                self.writer.join_request(rid)
            return base.native_file(self, rid)

        def post_step(self, executed):
            pass

        def _remote_ra_ack(self, call_id, frame, failure):
            raise AssertionError('no Cancel acknowledgement in this harness')

        def attach_deferred(self, config, shadow=None, **writer):
            self.dispatcher._remote_ra_live_v2 = self.route
            factory = None if shadow is None else (lambda *a, **k: ShadowWriter(*a, shadow=shadow, **k))
            self.writer = D.attach_engine(self.core, self.lifecycle, self.dispatcher, ticker=NS(tick=lambda: None),
                                          wake=None, loop=self, config=dict(config, **{KEY: True}), factory=factory,
                                          **writer)
            return self.writer
    return Harness


SLOW_WRITER = dict(pause=lambda: time.sleep(0.002), slice_s=0.0)


def multi_run(case, policy, deferred, fast=0, writer=None):
    """test_decouple_e2e's three sessions decoding together (its script, deterministic) with the
    decode fast paths ``fast`` (test_integrate_flags.attach_fast); ``deferred``: the switch, with
    a shadow writer. Everything the run sent, kept and wrote."""
    import test_decouple_e2e as DE
    import test_host_trim as HT
    import test_integrate_flags as IF
    multi = DE.Case(**dict(vars(case), sessions=len(DE.TASKS)))
    with ExitStack() as stack:
        DE.enter_case(stack, multi)
        HT.deterministic(stack)
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        value = DE.config(policy, output, multi, DE.TASKS)
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        h = deferred_harness(DE.MultiHarness)(policy, output, multi)
        IF.attach_fast(h, fast)
        trace, rpc = [], h.proxy.executor.collective_rpc

        def collective_rpc(method, args, timeout):
            operation, payload = args
            sent = HT.plain(payload)
            reply = rpc(method, args, timeout)
            trace.append(dict(operation=operation, payload=sent, receipt=HT.plain(reply[0])))
            return reply
        h.proxy.executor = NS(collective_rpc=collective_rpc)
        out = None
        if deferred:
            out = h.attach_deferred(value, shadow=Path(stack.enter_context(tempfile.TemporaryDirectory())),
                                    **(writer or {}))
        h.serve()
        frames = {rid: HT.plain(items) for rid, items in h.proxy.http_receipts.items()}
        if out is not None:
            out.close()                    # shutdown_engine's first stage
        h.close()
        DE.engine_closed(h.core)
        files = {p.name: p.read_bytes() for p in sorted(output.iterdir())}
        return dict(trace=trace, files=files, frames=frames, computed=list(h.runner.computed),
                    done=[(i['rid'], list(i['tokens']), list(i['request'].output_token_ids), i['committed']['text'],
                           list(i['plan']['selected'])) for i in h.done],
                    widths=list(h.widths), writer=out)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class MultiSessionTests(unittest.TestCase):
    """Three mswea-xml sessions decoding together: the switch on equals it off."""
    maxDiff = None

    def compare(self, off, on):
        import test_decode_fastpath3 as TF3
        for key in ('files', 'frames', 'computed', 'done', 'widths', 'trace'):
            self.assertTrue(on[key] == off[key], (key, TF3.first_difference(off[key], on[key])))
        natives = [name for name in on['files'] if name.startswith('native-')]
        writer = on['writer']
        self.assertEqual((writer.compared, writer.mismatches, writer.stats()['jobs']), (len(natives), [], len(natives)))
        self.assertGreater(max(off['widths']), 1)

    def check(self, case, policy, fast=0, writer=None):
        off = multi_run(case, policy, False, fast)
        on = multi_run(case, policy, True, fast, writer)
        self.compare(off, on)
        return on

    def test_16_token_pages_every_policy(self):
        import test_decouple_e2e as DE
        for policy in (('full', 'goal_query') if not SLOW else ('full', 'observation_query', 'goal_query', 'random50')):
            with self.subTest(policy=policy):
                on = self.check(DE.CASES[2], policy)
                if policy == 'goal_query':
                    self.assertIn('goals.jsonl', on['files'])

    def test_with_the_three_decode_fast_paths(self):
        import test_decouple_e2e as DE
        on = self.check(DE.CASES[2], 'observation_query', fast=3, writer=SLOW_WRITER)
        # This harness reads each native file inside the step that completes its request (the
        # servers below read it after the acknowledgement): every write is joined at once.
        self.assertEqual(on['writer'].stats()['errors'], 0)

    @unittest.skipUnless(SLOW, 'about a minute more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_784_token_pages(self):
        import test_decouple_e2e as DE
        for case in DE.CASES[:2]:
            for policy in ('full', 'goal_query'):
                with self.subTest(case=case.name, policy=policy):
                    self.check(case, policy, fast=3 if policy == 'full' else 0)


def pressure_run(deferred, fast=3, policy='observation_query'):
    """test_idle_release's sixteen sessions under pool pressure (900 FullAttention blocks) with
    --release-idle-pages and the decode fast paths ``fast``, deterministic; ``deferred``: the
    switch (a shadow writer). (the run's outputs, its files, the writer)."""
    import test_decode_fastpath3 as TF3
    import test_decouple_e2e as DE
    import test_host_trim as HT
    import test_idle_release as IR
    import test_integrate_flags as IF
    captured = {}

    class Rows16(TF3.NpTable):
        def __init__(self, rows, width, block_size):
            super().__init__(max(rows, 16), width, block_size)

    class Pressure(IR.PressureTests):
        def enter(self, stack, policy, blocks):
            HT.deterministic(stack)
            stack.enter_context(patch.object(TF3, 'NpTable', Rows16))
            stack.enter_context(patch.object(DE, 'Runner', TF3.vllm_runner(True)))
            case, output = IR.PressureTests.enter(self, stack, policy, blocks)
            captured['output'] = output
            captured['shadow'] = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            # run_case's last files (engine-cleanup.json included), before its directory goes.
            stack.callback(lambda: captured.update(files={p.name: p.read_bytes() for p in sorted(output.iterdir())}))
            return case, output

        def harness(self, policy, case, output, release):
            import test_decouple_e2e as DE_
            with patch.object(DE_, 'MultiHarness', deferred_harness(DE_.MultiHarness)):
                h = IR.PressureTests.harness(self, policy, case, output, release)
            IF.attach_fast(h, fast)
            if deferred:
                import remote_ra_http.config as C
                writer = captured['writer'] = h.attach_deferred(C.load(), shadow=captured['shadow'])
                close = h.close

                def closed():
                    writer.close()             # shutdown_engine's first stage, then the drain
                    close()
                h.close = closed
            return h
    run = Pressure('test_sixteen_sessions_full')
    result = run.run_case(policy, run.BLOCKS, True)
    return result, captured['files'], captured.get('writer')


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class PressureTests(unittest.TestCase):
    """Sixteen sessions under pool pressure, --release-idle-pages and the three fast paths."""
    maxDiff = None

    def test_releases_and_readmissions_with_the_switch(self):
        import test_decode_fastpath3 as TF3
        off, off_files, _ = pressure_run(False)
        on, on_files, writer = pressure_run(True)
        self.assertEqual(on[0], off[0])                          # prompts, replies, selections, attended Pages
        self.assertTrue(on_files == off_files, TF3.first_difference(off_files, on_files))
        events = [json.loads(line) for line in on_files['capacity.jsonl'].splitlines()]
        self.assertTrue(any(e['event'] == 'evict' for e in events) and any(e['event'] == 'readmit' for e in events))
        natives = [n for n in on_files if n.startswith('native-')]
        self.assertEqual((writer.compared, writer.mismatches, len(natives)), (64, [], 64))


def batch_engine(base, engines, *, shadow_root, writer=None):
    """test_ms2_e2e's BatchEngine family (also test_kv_session_e2e's and test_chat_text_e2e's)
    with the switch attached as compose_engine attaches it to EngineCoreProc: post_step after
    each step, the core's acknowledgement seam ``_remote_ra_ack(call_id, frame, failure)``
    (call_id: the frame and the frontend's future), the lifecycle tick at each input pass, and
    the writer closed first at shutdown. A shadow writer compares every file it writes."""

    class Engine(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            loop = asyncio.get_running_loop()
            self.ticker = NS(tick=lambda: None)
            shadow = Path(tempfile.mkdtemp(dir=shadow_root))
            self.writer = D.attach_engine(
                self.core, self.lifecycle, self.dispatcher, ticker=self.ticker,
                wake=lambda: loop.call_soon_threadsafe(self.wakeup.set), loop=self,
                config=dict(EV.load(), **{KEY: True}),
                factory=lambda *a, **k: ShadowWriter(*a, shadow=shadow, **k), **(writer or {}))
            engines.append(self)

        def post_step(self, executed):
            pass

        def step(self):
            ran = base.step(self)
            self.post_step(ran)
            return ran

        def _process_input_queue(self):
            self.ticker.tick()
            return base._process_input_queue(self)

        def _handle_client_request(self, kind, payload):
            assert kind == 'REGISTERED_ADD', kind
            frame, future = payload
            if self.watch:
                self.watch('handle', frame)
            goals = self._remote_ra_goal_admissions
            try:
                if goals is None:
                    reply = self.dispatcher.handle(frame)
                else:
                    import functools
                    reply = goals.handle(self.dispatcher, frame, functools.partial(self._remote_ra_ack, (frame, future)))
                    if reply is goals.PARKED:
                        return
            except BaseException as failure:
                if not future.done():
                    future.set_exception(failure)
                raise
            self._remote_ra_ack((frame, future), reply, None)

        def _remote_ra_ack(self, call_id, frame, failure):
            wire, future = call_id
            base._remote_ra_ack(self, wire, future, frame, failure)

        def shutdown(self):
            self.writer.close()
            return base.shutdown(self)
    return Engine


def run_with_switch(module, cls, names, *, writer=None, patch_name='BatchEngine', base=None):
    """Run ``cls``'s tests ``names`` (test_integrate_flags.run_tests) with the switch on their
    batched engine; returns (outcomes, engines)."""
    import test_integrate_flags as IF
    engines = []
    root = Path(tempfile.mkdtemp())
    try:
        Engine = batch_engine(base or getattr(module, patch_name), engines, shadow_root=root, writer=writer)
        with patch.object(module, patch_name, Engine):
            outcomes = IF.run_tests(cls, names)
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    return outcomes, engines


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ServerTests(unittest.TestCase):
    """The batched API servers (the packaged router, the frontend, the authenticated Cancel)
    with the switch: every check of those tests holds and every file the writer wrote equals
    the synchronous path's of the same moment."""
    maxDiff = None

    def check(self, outcomes, engines, held=False):
        for name, *outcome in outcomes:
            with self.subTest(name):
                self.assertEqual(outcome, [1, [], [], []])
        # At least one engine per test: a test may serve on several servers (test_ms2_e2e's
        # two- and three-session runs build one batched server each in the slow pass).
        self.assertGreaterEqual(len(engines), len(outcomes))
        for k, engine in enumerate(engines):
            stats = engine.writer.stats()
            self.assertEqual((engine.writer.mismatches, stats['errors'], stats['pending']), ([], 0, 0), k)
        compared = sum(e.writer.compared for e in engines)
        self.assertGreater(compared, 4)
        if held:
            self.assertGreater(sum(e.writer.counters['held'] for e in engines), 0)
        return compared

    @unittest.skipUnless(SLOW, 'about 15 s more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_mswea_xml_sessions_decoding_together(self):
        import test_ms2_e2e as B2
        names = ['test_two_and_three_sessions_decode_together_goal_query',
                 'test_two_completions_in_one_step_then_a_probe_before_the_next',
                 'test_cancel_of_one_session_between_batched_steps_leaves_the_other_decoding']
        if SLOW:
            names += ['test_two_and_three_sessions_decode_together_full',
                      'test_failed_completion_in_a_batched_step_fails_only_its_session',
                      'test_shutdown_with_requests_in_flight_drains_and_closes_everything']
        self.check(*run_with_switch(B2, B2.BatchedEndToEndTests, names))

    def test_mswea_xml_with_a_slow_writer_holds_the_acknowledgements(self):
        import test_ms2_e2e as B2
        self.check(*run_with_switch(B2, B2.BatchedEndToEndTests,
                                    ['test_two_completions_in_one_step_then_a_probe_before_the_next'],
                                    writer=SLOW_WRITER),
                   held=True)

    @unittest.skipUnless(SLOW, 'about 25 s more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_kv_session_memagent(self):
        import test_kv_session_e2e as KE
        names = ['test_memagent_sessions_goal_query']
        if SLOW:
            names += ['test_memagent_sessions_full', 'test_memagent_sessions_observation_query',
                      'test_refusals_and_closes', 'test_busy_session_and_close_while_in_flight',
                      'test_engine_failure_fails_only_its_session']
        self.check(*run_with_switch(KE, KE.KvSessionEndToEndTests, names, patch_name='KvEngine'))

    def test_kv_session_with_a_slow_writer(self):
        import test_kv_session_e2e as KE
        self.check(*run_with_switch(KE, KE.KvSessionEndToEndTests, ['test_memagent_sessions_goal_query'],
                                    patch_name='KvEngine', writer=SLOW_WRITER), held=True)

    @unittest.skipUnless(SLOW, 'about 10 s more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_chat_text_sessions(self):
        import test_chat_text_e2e as CT
        import test_kv_session_e2e as KE
        names = ['test_sixteen_sessions_decode_together']
        if SLOW:
            names += ['test_header_aside_handoff_and_close', 'test_cancel_retry_rebinds_and_the_failed_slot_closes',
                      'test_idle_close_eviction_turn_bound_and_lifetime']
        self.check(*run_with_switch(KE, CT.ChatTextEndToEndTests, names, patch_name='KvEngine'))

    @unittest.skipUnless(SLOW, 'about 10 s more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_chat_text_with_a_slow_writer(self):
        import test_chat_text_e2e as CT
        import test_kv_session_e2e as KE
        self.check(*run_with_switch(KE, CT.ChatTextEndToEndTests, ['test_sixteen_sessions_decode_together'],
                                    patch_name='KvEngine', writer=SLOW_WRITER), held=True)

    def replies(self, server, method, cls, names, deferred, writer=None):
        """Every HTTP exchange of the run's client sessions: {session: [(request, status, reply)]},
        each session keyed by its first request and its uuids numbered in order of appearance
        (both are the sessions' own, whatever the interleaving of the sessions)."""
        import re
        import test_integrate_flags as IF
        import test_kv_session_e2e as KE
        records = []
        original = getattr(server, method)

        async def recorded(srv, *args, **kwargs):
            result = await original(srv, *args, **kwargs)
            records.append((args, {k: v for k, v in kwargs.items() if k != 'receive'}, result))
            return result
        with patch.object(server, method, recorded):
            if deferred:
                outcomes, engines = run_with_switch(KE, cls, names, patch_name='KvEngine', writer=writer)
                self.check(outcomes, engines, held=writer is not None)
            else:
                for name, *outcome in IF.run_tests(cls, names):
                    self.assertEqual(outcome, [1, [], [], []], name)
        token = re.compile(r'[0-9a-f]{32}|call_[0-9a-f]{12}')

        def plain(value):
            # Not the replies' content: ids, creation times and wall-clock waits (queue_wait_s).
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()
                        if k not in ('id', 'created') and not k.endswith(('_s', '_ms', '_ns'))}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            return value
        sessions = {}
        for args, kwargs, (status, reply) in records:
            request = plain([args, kwargs])
            body = args[1] if method == 'send' else None
            sid = (body or {}).get('session_id') if method == 'send' else (args[1] if len(args) > 1 else None)
            key = sid if sid is not None else json.dumps(plain(args[0][:2]), sort_keys=True)
            sessions.setdefault(key, []).append(json.dumps([request, status, plain(reply)], sort_keys=True))
        out = {}
        for exchanges in sessions.values():
            names_seen = {}
            text = token.sub(lambda m: names_seen.setdefault(m.group(0), '<u%d>' % len(names_seen)),
                             '\n'.join(exchanges))
            out[text.split('\n', 1)[0]] = text
        return out

    def test_replies_equal_with_the_switch_off_and_on(self):
        import test_chat_text_e2e as CT
        import test_kv_session_e2e as KE
        # chat-text in the suite (its replies with a slow writer: the acknowledgements are held);
        # kv-session in the slow pass (test_kv_session_with_a_slow_writer holds them in the suite).
        cases = [(CT.ChatServer, 'chat_request', CT.ChatTextEndToEndTests, 'test_sixteen_sessions_decode_together')]
        if SLOW:
            cases.append((KE.KvServer, 'send', KE.KvSessionEndToEndTests, 'test_memagent_sessions_goal_query'))
        for server, method, cls, name in cases:
            with self.subTest(cls.__name__):
                off = self.replies(server, method, cls, [name], False)
                on = self.replies(server, method, cls, [name], True, writer=SLOW_WRITER)
                self.assertGreaterEqual(len(off), 4)
                self.assertEqual(sorted(on), sorted(off))
                for key in off:
                    self.assertEqual(on[key], off[key])


# ---- 6. the step profile -----------------------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ProfileTests(unittest.TestCase):
    """The completion's sub-scopes on test_step_profile's three profiled sessions, off and on."""
    maxDiff = None

    def run_profiled(self, deferred):
        import test_decouple_e2e as DE
        import test_step_profile as TSP
        import remote_ra_step_profile as SP
        lines = []
        prof = SP.Profiler(every=10, emit=lines.append)
        Harness = deferred_harness(TSP.ProfiledMultiHarness)
        attached = []
        original = Harness.__init__

        def init(h, *args, **kwargs):
            original(h, *args, **kwargs)
            if deferred:
                import remote_ra_http.config as C
                attached.append(h.attach_deferred(C.load()))
        with patch.object(Harness, '__init__', init), patch.object(TSP, 'ProfiledMultiHarness', Harness):
            closed = EV.engine_closed

            def engine_closed(core):
                for writer in attached:
                    writer.close()
                return closed(core)
            with patch.object(TSP.DE, 'engine_closed', engine_closed):
                run = TSP.serve_sessions(DE.CASES[2], 'goal_query', prof)
        return run, [json.loads(line[len('REMOTE_RA_STEP '):]) for line in lines]

    def test_sub_scopes_off_and_on(self):
        import remote_ra_step_profile as SP
        names = {name for name, _ in SP.TREE}
        runs = {}
        for deferred in (False, True):
            run, values = self.run_profiled(deferred)
            runs[deferred] = run
            total = values[-1]
            self.assertEqual((total['kind'], total['errors']), ('total', 0))
            seen = {}
            for group in total['groups']:
                for name, stat in group['c'].items():
                    seen[name] = seen.get(name, 0) + stat[1]
            completions = seen['sched.complete']
            for name in ('sched.complete.drain', 'rpc.complete', 'rpc.complete.copy', 'sched.complete.after',
                         'sched.complete.transcript', 'sched.complete.terminal', 'sched.complete.publish'):
                self.assertEqual(seen.get(name), completions, name)
            if deferred:
                self.assertEqual(seen.get('sched.complete.terminal.snapshot'), completions)
                for name in ('sched.complete.terminal.save', 'sched.complete.terminal.config',
                             'sched.complete.terminal.copy'):
                    self.assertNotIn(name, seen)
                self.assertEqual(total['completion_writer']['jobs'], completions)
            else:
                for name in ('sched.complete.terminal.save', 'sched.complete.terminal.encode',
                             'sched.complete.terminal.copy', 'sched.complete.terminal.goal'):
                    self.assertEqual(seen.get(name), completions, name)
                # host-trims-1111 (perf 2.4): the startup config is read once per engine (this
                # harness has no compose_engine: at its first terminal), not at every completion.
                self.assertEqual(seen.get('sched.complete.terminal.config'), 1)
                self.assertNotIn('completion_writer', total)
            self.assertTrue(all(n in names or n in SP.OUTSIDE or n.startswith('rpc.complete') for n in seen),
                            set(seen) - names)
        for key in ('done', 'trace', 'files', 'widths'):
            self.assertEqual(runs[True][key], runs[False][key], key)


if __name__ == '__main__':
    unittest.main()
