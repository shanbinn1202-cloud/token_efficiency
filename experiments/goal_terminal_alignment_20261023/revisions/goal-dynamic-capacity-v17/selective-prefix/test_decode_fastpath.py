"""serve.py --decode-fastpath (remote_ra_live_v2.decode_fastpath). CPU only.

Off: no config key, worker record, evidence key or vLLM argument; the code paths are
the former ones. On: the same values with less per-decode-step host work, checked
here piece by piece against the former computations and end to end over the
decoupled multi-session fakes (test_decouple_e2e's MultiHarness, P = 16 and the
legacy 784): identical RPC payloads and receipts, evidence files, attended Pages,
computed positions and committed replies, with fewer device drains and no scan
of other requests' claims.
"""
from contextlib import ExitStack
import itertools
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

from remote_ra_live_v2 import decode_fastpath as F
from remote_ra_live_v2.contract import Allocation, HybridLease, LiveV2Error, NativeIdentity, StepLease
from remote_ra_live_v2.mswea_protocol import MARKER, MarkerStop
# The runner hooks import these lazily; importing them here keeps them out of the
# end-to-end runs' sys.modules patch (enter_case), which drops modules first
# imported inside it and would make later tests' module patches miss them.
from remote_ra_live_v2 import (generation_guard, hooks, native_probe, page_compaction,  # noqa: F401
                               phase1_step, positions, vector_space, worker_rows)

try:
    import torch
except ImportError:
    torch = None


def scan(claims, request=None):
    """remote_ra_http.evidence.measured_command's former count, verbatim."""
    blocks = {}
    for (group, block), owner in claims.items():
        if request is None or owner[0] == request:
            blocks[group] = blocks.get(group, 0) + 1
    return blocks


class ClaimTableTests(unittest.TestCase):
    def test_counts_equal_the_scan_under_random_mutations(self):
        rng = random.Random(7)
        table, plain = F.ClaimTable(), {}
        requests = ['r%d' % i for i in range(5)]
        for _ in range(4000):
            op = rng.random()
            key = (rng.randrange(4), rng.randrange(60))
            owner = (rng.choice(requests), rng.randrange(3))
            if op < .45:
                table[key] = owner
                plain[key] = owner
            elif op < .65:
                self.assertEqual(table.pop(key, None), plain.pop(key, None))
            elif op < .72 and plain:
                k = rng.choice(sorted(plain))
                del table[k]
                del plain[k]
            elif op < .78:
                self.assertEqual(table.setdefault(key, owner), plain.setdefault(key, owner))
            elif op < .84:
                batch = {(rng.randrange(4), rng.randrange(60)): (rng.choice(requests), 0) for _ in range(3)}
                table.update(batch)
                plain.update(batch)
            elif op < .86 and plain:
                k, v = table.popitem()
                self.assertEqual(plain.pop(k), v)
            elif op < .87:
                table.clear()
                plain.clear()
            self.assertEqual(table, plain)
            for request in requests + [None, 'absent']:
                self.assertEqual(table.groups_of(request), scan(plain, request))
        self.assertIsInstance(table, dict)

    def test_pop_without_default_raises_and_ior_counts(self):
        table = F.ClaimTable({(0, 1): ('a', 1)})
        with self.assertRaises(KeyError):
            table.pop((0, 2))
        table |= {(1, 3): ('b', 1)}
        self.assertEqual((table.groups_of('a'), table.groups_of('b')), ({0: 1}, {1: 1}))

    def test_measured_command_reads_the_same_active_bytes(self):
        from remote_ra_http.evidence import measured_command
        rng = random.Random(3)
        claims = {(g, b): ('r%d' % rng.randrange(4), 1) for g in range(4) for b in range(1, 200)
                  if rng.random() < .6}
        views = [NS(group=g, page_bytes=1000 + g) for g in (0, 0, 1, 2, 3)]
        out = []
        for table in (dict(claims), F.ClaimTable(claims)):
            backend = NS(io=NS(views=views, handoff=False), owner=NS(claims=table, records={}))
            for request in ('r0', 'r1', 'r9', None):
                with patch('time.monotonic_ns', itertools.count(1).__next__):
                    out.append(measured_command(backend, 'allocation_update',
                                                lambda b, q: dict(operation='allocation_update'), None,
                                                request=request))
        self.assertEqual(out[:4], out[4:])
        self.assertTrue(any(r['http_measurement']['active_owned_kv_bytes'] for r in out))

    def test_attach_worker_replaces_the_empty_claims_only(self):
        owner, bridge, qwen = NS(claims={}), NS(), NS()
        F.attach_worker(owner, bridge, qwen)
        self.assertIsInstance(owner.claims, F.ClaimTable)
        self.assertTrue(bridge.decode_fastpath and qwen.decode_fastpath)
        with self.assertRaises(LiveV2Error):
            F.attach_worker(NS(claims={(0, 1): ('r', 1)}), NS(), NS())


class PinTests(unittest.TestCase):
    def test_pin_values(self):
        self.assertIsNone(F.pinned({}))
        self.assertIsNone(F.pinned(None))
        self.assertTrue(F.pinned({'decode_fastpath': True}))
        for bad in (False, 1, 'true', None):
            with self.assertRaises(LiveV2Error):
                F.pinned({'decode_fastpath': bad})

    def test_worker_flag_without_http_runtime_checks_the_value_only(self):
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            self.assertIsNone(F.worker_flag({}))
            self.assertTrue(F.worker_flag({'decode_fastpath': True}))
            with self.assertRaises(LiveV2Error):
                F.worker_flag({'decode_fastpath': False})

    def test_worker_flag_with_http_runtime_must_equal_the_pin(self):
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'):
            with patch('remote_ra_http.config.load', lambda: {'decode_fastpath': True}):
                self.assertTrue(F.worker_flag({'decode_fastpath': True}))
                with self.assertRaises(LiveV2Error):
                    F.worker_flag({})
            with patch('remote_ra_http.config.load', lambda: {}):
                self.assertIsNone(F.worker_flag({}))
                with self.assertRaises(LiveV2Error):
                    F.worker_flag({'decode_fastpath': True})


class StepLeaseTests(unittest.TestCase):
    def setUp(self):
        self.identity = NativeIdentity('r', 's', 1, 'g', 'e' * 64, 'f' * 64, 'm')
        slots = tuple(Allocation(g, 100 + g, 3, 'mamba', 1) for g in range(3))
        self.hybrid = HybridLease(self.identity, slots, 0, 0)
        self.attention = tuple(Allocation(3, b, 3, 'decode', 1) for b in range(1, 50))

    def test_extending_builds_the_same_lease(self):
        args = (self.identity, 3, 40, 41, 'decode', self.attention, self.hybrid)
        for k in (0, 1, 25, 49):
            lease = StepLease.extending(*args, k)
            self.assertEqual(lease, StepLease(*args))
            self.assertEqual(vars(lease), vars(StepLease(*args)))
            self.assertEqual(hash(lease.identity), hash(self.identity))

    def test_extending_checks_everything_but_the_prefix(self):
        bad = self.attention + ('not an allocation',)
        with self.assertRaises(LiveV2Error):
            StepLease.extending(self.identity, 3, 40, 41, 'decode', bad, self.hybrid, 49)
        with self.assertRaises(LiveV2Error):
            StepLease(self.identity, 3, 40, 41, 'decode', bad, self.hybrid)
        for args in ((self.identity, -1, 40, 41, 'decode'), (self.identity, 3, 41, 41, 'decode'),
                     (self.identity, 3, 40, 41, 'other')):
            with self.assertRaises(LiveV2Error):
                StepLease.extending(*args, self.attention, self.hybrid, 49)
        other = NativeIdentity('q', 's', 1, 'g', 'e' * 64, 'f' * 64, 'm')
        with self.assertRaises(LiveV2Error):
            StepLease.extending(other, 3, 40, 41, 'decode', self.attention, self.hybrid, 49)
        with self.assertRaises(LiveV2Error):
            StepLease.extending(self.identity, 3, 40, 41, 'decode', self.attention, self.hybrid, 50)


@unittest.skipUnless(torch is not None, 'CPU torch required')
class PositionTests(unittest.TestCase):
    """text_positions(host=step_positions(...)) returns and refuses as the per-request read."""

    def runner(self, mrope, logical):
        tokens = list(range(500))
        req = NS(mm_features=None, prompt_embeds=None, prompt_token_ids=tokens, mrope_position_delta=0,
                 mrope_positions=torch.zeros((3, len(tokens)), dtype=torch.int64))
        return NS(uses_mrope=mrope, requests={'a': req, 'b': req, 'c': req},
                  positions=torch.tensor(logical, dtype=torch.int64))

    def spans(self):
        return (('a', 100, 1, 0), ('b', 7, 1, 1), ('c', 300, 3, 2))

    def flat(self):
        return [100, 7, 300, 301, 302, 0, 0]          # padded step columns

    def test_one_dimensional_and_mrope_reads_are_identical(self):
        from remote_ra_live_v2.positions import step_positions, text_positions
        for mrope in (False, True):
            runner = self.runner(mrope, self.flat())
            positions = (torch.tensor([self.flat()] * 3, dtype=torch.int64) if mrope
                         else torch.tensor(self.flat(), dtype=torch.int64))
            host = step_positions(runner, positions, 5)
            for rid, start, count, offset in self.spans():
                self.assertEqual(text_positions(runner, positions, start, count, rid, offset, host=host),
                                 text_positions(runner, positions, start, count, rid, offset))

    def test_refusals_are_the_same(self):
        from remote_ra_live_v2.positions import step_positions, text_positions
        for mrope in (False, True):
            wrong = list(self.flat())
            wrong[3] = 999
            runner = self.runner(mrope, wrong)
            positions = (torch.tensor([wrong] * 3, dtype=torch.int64) if mrope
                         else torch.tensor(wrong, dtype=torch.int64))
            host = step_positions(runner, positions, 5)
            for kwargs in ({}, dict(host=host)):
                with self.assertRaises(LiveV2Error) as caught:
                    text_positions(runner, positions, 300, 3, 'c', 2, **kwargs)
                if not kwargs:
                    message = str(caught.exception)
                else:
                    self.assertEqual(str(caught.exception), message)
            # A host read too short for a request falls back to the device read.
            short = step_positions(runner, positions, 2)
            self.assertEqual(text_positions(runner, positions, 100, 1, 'a', 0, host=short),
                             text_positions(runner, positions, 100, 1, 'a', 0))
        self.assertIsNone(step_positions(self.runner(False, self.flat()), torch.zeros(3, dtype=torch.int64), 5))


class Decoder:
    """Byte-level decode like Qwen2's: token bytes joined, UTF-8 with replacement,
    special tokens skipped (skip_special_tokens=True). Counts calls and tokens."""

    def __init__(self, vocab, special):
        self.vocab, self.special = vocab, special
        self.calls = self.tokens = 0

    def __call__(self, ids):
        ids = list(ids)
        self.calls += 1
        self.tokens += len(ids)
        return b''.join(self.vocab[i] for i in ids if i not in self.special).decode('utf-8', 'replace')


def vocabulary():
    pieces = [b'<', b'/', b'mswea', b'_', b'bash', b'_command', b'>', b'</', b'mswea_bash_command',
              b'>\n', b' ', b'\n', b'ls', b' -la', b'cd', b'THOUGHT', b':', b'a', b'b', b'->', b'>>',
              'é'.encode()[:1], 'é'.encode()[1:], '中'.encode()[:2], '中'.encode()[2:],
              '中'.encode(), b'', b'<|im_end|>', b'command>', b'</mswea_bash_command>']
    vocab = dict(enumerate(pieces))
    special = {pieces.index(b'<|im_end|>')}
    return vocab, special


class MarkerStopTests(unittest.TestCase):
    def requests(self, rng, vocab, n, length):
        out = []
        for i in range(n):
            out.append([rng.randrange(len(vocab)) for _ in range(rng.randrange(1, length))])
        return out

    def play(self, stop, tokens, rid='r'):
        """The scheduler's calls: one per appended token until the stop says True."""
        request = NS(request_id=rid, output_token_ids=[], status=None, stop_reason=None)
        for k, t in enumerate(tokens, 1):
            request.output_token_ids.append(t)
            if stop(request):
                return k
        return None

    def test_incremental_decides_exactly_as_the_full_decode(self):
        vocab, special = vocabulary()
        rng = random.Random(11)
        streams = self.requests(rng, vocab, 3000, 60)
        for tokens in streams[:200]:
            # Make some streams spell the marker across token boundaries.
            tokens[rng.randrange(len(tokens)):] = [7, 2, 3, 4, 5, 6]
        streams += [[16, 17, 10] * 30 + [8, 6], [28] * 3, [7, 2, 3, 4, 5, 9]]
        full, fast = Decoder(vocab, special), Decoder(vocab, special)
        with patch.dict(sys.modules, {'vllm.v1.request': NS(RequestStatus=NS(FINISHED_STOPPED='stopped'))}):
            for i, tokens in enumerate(streams):
                expected = self.play(MarkerStop(full), tokens)
                self.assertEqual(self.play(MarkerStop(fast, incremental=True), tokens), expected, tokens)
        self.assertLess(fast.tokens, full.tokens / 3)

    def test_interleaved_requests_and_a_skipped_call_decode_in_full(self):
        vocab, special = vocabulary()
        decode = Decoder(vocab, special)
        stop = MarkerStop(decode, incremental=True)
        with patch.dict(sys.modules, {'vllm.v1.request': NS(RequestStatus=NS(FINISHED_STOPPED='stopped'))}):
            a = NS(request_id='a', output_token_ids=[16, 17], status=None, stop_reason=None)
            b = NS(request_id='b', output_token_ids=[7, 2, 3, 4], status=None, stop_reason=None)
            self.assertFalse(stop(a))
            self.assertFalse(stop(b))
            b.output_token_ids += [5, 6]           # two tokens since b's last call: full decode
            before = decode.tokens
            self.assertTrue(stop(b))
            self.assertEqual(decode.tokens - before, 6)
            self.assertEqual((b.status, b.stop_reason), ('stopped', MARKER))
            a.output_token_ids.append(10)
            before = decode.tokens
            self.assertFalse(stop(a))
            self.assertEqual(decode.tokens - before, 1)     # the new token alone
            # The table starts over when full; answers do not change.
            stop.INCREMENTAL_REQUESTS = 1
            c = NS(request_id='c', output_token_ids=[16], status=None, stop_reason=None)
            self.assertFalse(stop(c))
            self.assertNotIn(('a', id(a)), stop._checked)

    def test_switch_must_be_a_bool(self):
        with self.assertRaises(LiveV2Error):
            MarkerStop(lambda ids: '', incremental=1)


def real_tokenizer():
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    evidence = ROOT / 'source-overlay/remote_ra_linux/model-evidence'
    if not (evidence / 'tokenizer.json').is_file():
        return None
    import shutil
    directory = tempfile.mkdtemp()
    for target, source in (('config.json', 'model-config.json'), ('tokenizer.json', 'tokenizer.json'),
                           ('tokenizer_config.json', 'tokenizer-config.json')):
        shutil.copyfile(evidence / source, Path(directory) / target)
    try:
        return AutoTokenizer.from_pretrained(directory, local_files_only=True)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@unittest.skipUnless(real_tokenizer() is not None, 'transformers and the pinned tokenizer files required')
class RealTokenizerMarkerStopTests(unittest.TestCase):
    def test_real_qwen_tokens(self):
        tok = real_tokenizer()
        decode = lambda ids: tok.decode(list(ids), skip_special_tokens=True)
        texts = ['THOUGHT: check the file.\n<mswea_bash_command>ls -la src/ && cat a->b >> c</mswea_bash_command>',
                 'Résumé 中文 → done: </mswea_bash_command> trailing',
                 'if a > b: print("<mswea_bash_command>")\n</mswea_bash_command',
                 '<think>\nplan</think>\nTHOUGHT: x\n<mswea_bash_command>echo ok</mswea_bash_command>\n',
                 'no marker here >>> >= -> <|im_end|> </mswea_bash_comman d>']
        rng = random.Random(5)
        with patch.dict(sys.modules, {'vllm.v1.request': NS(RequestStatus=NS(FINISHED_STOPPED='stopped'))}):
            for text in texts:
                ids = tok.encode(text, add_special_tokens=False)
                streams = [ids, ids + [tok.convert_tokens_to_ids('<|im_end|>')]]
                for _ in range(20):
                    cut = rng.randrange(1, len(ids))
                    streams.append(ids[cut:] + ids[:cut])
                for tokens in streams:
                    expected = MarkerStopTests.play(None, MarkerStop(decode), tokens)
                    got = MarkerStopTests.play(None, MarkerStop(decode, incremental=True), tokens)
                    self.assertEqual(got, expected, text)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(unittest.TestCase):
    """Fast path on vs off over test_decouple_e2e's multi-session harness."""

    def serve(self, case, policy, fast):
        import test_decouple_e2e as D
        import test_host_trim as HT
        multi = D.Case(**dict(vars(case), sessions=len(D.TASKS)))
        with ExitStack() as stack:
            D.enter_case(stack, multi)
            HT.deterministic(stack)
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = D.config(policy, output, multi, D.TASKS)
            if fast:
                value['decode_fastpath'] = True
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            h = D.MultiHarness(policy, output, multi)
            if fast:
                F.attach_worker(h.owner, h.bridge, h.qwen)
            drains = []
            h.io.drain = lambda: drains.append(1)
            trace, rpc = [], h.collective_rpc

            def collective_rpc(method, args, timeout):
                operation, payload = args
                sent = HT.plain(payload)
                reply = rpc(method, args, timeout)
                trace.append(dict(operation=operation, payload=sent, receipt=HT.plain(reply[0])))
                return reply
            h.proxy.executor = NS(collective_rpc=collective_rpc)
            scans = []
            items = dict.items

            class Watched(type(h.owner.claims)):
                def items(self):
                    # Only measured_command's scan counts (the owner's full claim
                    # passes, which these fakes take every step, scan as before).
                    if sys._getframe(1).f_globals.get('__name__') == 'remote_ra_http.evidence':
                        scans.append(len(self))
                    return items(self)
            h.owner.claims = Watched(h.owner.claims)
            h.serve()
            h.close()
            D.engine_closed(h.core)
            files = {p.name: p.read_bytes() for p in sorted(output.iterdir())}
            return dict(trace=trace, files=files, computed=list(h.runner.computed),
                        visible=dict(h.runner.visible), plans=dict(h.runner.plans),
                        done=[(i['rid'], list(i['request'].output_token_ids)) for i in h.done],
                        widths=list(h.widths)), len(drains), scans

    def test_on_equals_off(self):
        import test_decouple_e2e as D
        # Default suite: full on legacy-784 and observation_query at P = 16; the slow pass
        # (README section 2.3) both on both.
        for case in (D.CASES[0], D.CASES[2]):
            for policy in ('full', 'observation_query'):
                if not (D.SLOW or (case.name, policy) in (('legacy-784', 'full'), ('16-784', 'observation_query'))):
                    continue
                with self.subTest(case=case.name, policy=policy):
                    off, drains_off, scans_off = self.serve(case, policy, False)
                    on, drains_on, scans_on = self.serve(case, policy, True)
                    self.assertEqual(on['trace'], off['trace'])
                    self.assertEqual(on['files'], off['files'])
                    for key in ('computed', 'visible', 'plans', 'done', 'widths'):
                        self.assertEqual(on[key], off[key], key)
                    self.assertGreater(max(off['widths']), 1)        # several requests per step
                    self.assertLess(drains_on, drains_off)
                    self.assertTrue(scans_off)                        # measured_command scanned...
                    self.assertFalse(scans_on)                        # ...and no longer does


class LauncherTests(unittest.TestCase):
    def test_switch_pins_one_config_key_and_changes_no_vllm_argument(self):
        import contextlib
        import io
        import install
        import serve
        import test_installer as INSTALL
        from remote_ra_http import config as http_config
        tasks = [dict(instance_id='repo__a-1', problem_statement='Fix a.'),
                 dict(instance_id='repo__b-2', problem_statement='Fix b.')]
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            package, site, model, release = INSTALL.fixture(root)
            install.install(site, model, release, package_root=package, apply=True)
            kw = dict(kv_cache_gib=24, page_tokens=16, cuda_graphs='full', max_model_len=131072)
            off = serve.prepare(release, [dict(t) for t in tasks], 'full', root/'off', 8231, **kw)
            on = serve.prepare(release, [dict(t) for t in tasks], 'full', root/'on', 8231,
                               decode_fastpath=True, **kw)
            default = serve.prepare(release, [dict(t) for t in tasks], 'full', root/'default', 8231,
                                    decode_fastpath=False, **kw)
            with self.assertRaises(ValueError):
                serve.prepare(release, [dict(t) for t in tasks], 'full', root/'bad', 8231, decode_fastpath=1, **kw)
            # The startup loader accepts the pin and refuses any other value.
            for name, config in (('on', on[1]), ('off', off[1])):
                path = root/(name + '.json')
                raw = serve.wire(config)
                path.write_bytes(raw)
                with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                                REMOTE_RA_HTTP_CONFIG_SHA256=serve.digest(raw)):
                    self.assertEqual(http_config.load().get('decode_fastpath'), True if name == 'on' else None)
                    self.assertEqual(F.loaded(), True if name == 'on' else None)
            bad = dict(on[1], decode_fastpath=False)
            path, raw = root/'bad.json', serve.wire(bad)
            path.write_bytes(raw)
            with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                            REMOTE_RA_HTTP_CONFIG_SHA256=serve.digest(raw)):
                with self.assertRaises(RuntimeError):
                    http_config.load()
        strip = lambda value, name: json.loads(json.dumps(value).replace(str(root/name).replace('\\', '\\\\'), '<out>'))
        (_, on_config, on_command, on_env), (_, off_config, off_command, off_env) = on, off
        self.assertEqual(strip(off_command, 'off'), strip(on_command, 'on'))            # vLLM argv unchanged
        self.assertEqual(strip(default[1], 'default'), strip(off_config, 'off'))       # False == absent
        self.assertNotIn('decode_fastpath', off_config)
        self.assertEqual({k: v for k, v in strip(on_config, 'on').items() if k != 'decode_fastpath'},
                         strip(off_config, 'off'))
        self.assertIs(on_config['decode_fastpath'], True)
        output = ('REMOTE_RA_HTTP_CONFIG', 'REMOTE_RA_HTTP_CONFIG_SHA256')
        self.assertEqual({k: v for k, v in on_env.items() if k not in output},
                         {k: v for k, v in off_env.items() if k not in output})

    def test_attach_loaded_worker_echoes_the_switch_and_attaches_the_fast_paths(self):
        import test_cudagraph as CG
        from remote_ra_live_v2 import worker_composition as WC

        class Owner(CG.FakeOwner):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.claims = {}
        tests = CG.AttachLoadedWorkerTests()
        with patch.object(CG, 'FakeOwner', Owner), patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            off, off_runner, _ = tests.attach(None)
            on, on_runner, _ = tests.attach(None, {'decode_fastpath': True})
        self.assertNotIn('decode_fastpath', off)
        self.assertEqual(dict(on), dict(off, decode_fastpath=True))
        owner, bridge = on_runner._remote_ra_live_v2_allocation_owner, on_runner._remote_ra_live_v2_bridge
        self.assertIsInstance(owner.claims, F.ClaimTable)
        self.assertTrue(bridge.decode_fastpath and bridge.qwen.decode_fastpath)
        off_owner, off_bridge = off_runner._remote_ra_live_v2_allocation_owner, off_runner._remote_ra_live_v2_bridge
        self.assertIs(type(off_owner.claims), dict)
        self.assertFalse(off_bridge.decode_fastpath or getattr(off_bridge.qwen, 'decode_fastpath', False))
        self.assertIs(WC.WorkerBridge.decode_fastpath, False)

    def test_compose_engine_sends_the_pin_once_and_checks_the_worker_echo(self):
        import contextlib
        import io
        import install
        import serve
        import test_installer as INSTALL
        from remote_ra_live_v2 import composition
        task = dict(instance_id='repo__a-1', problem_statement='Fix a.')
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            package, site, model, release = INSTALL.fixture(root)
            install.install(site, model, release, package_root=package, apply=True)
            for on in (False, True):
                config = serve.prepare(release, dict(task), 'full', root/('ce-%s' % on), 8231,
                                       decode_fastpath=on)[1]
                path = root/('ce-%s.json' % on)
                path.write_bytes(serve.wire(config))
                sent = []

                def collective_rpc(method, args, timeout):
                    sent.append(args[0])
                    return [{}]                  # a worker that echoes nothing
                core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=784))),
                          _remote_ra_trusted_issue=dict(task), _remote_ra_trusted_issues=[dict(task)],
                          _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                          _remote_ra_growing_adapter=NS(layout=None), model_executor=NS(collective_rpc=collective_rpc))
                with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                                REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
                    with self.assertRaises(LiveV2Error) as caught:
                        composition.compose_engine(core, object())
                startup, = sent
                self.assertEqual(startup, dict(enabled=True, source='m' * 64, page_tokens=784,
                                               loaded_sources_verified=True,
                                               **({'decode_fastpath': True} if on else {})))
                # On: refused at the missing echo; off: the echo check passes and the next
                # check (the attach receipt) refuses the empty reply.
                self.assertIn('decode_fastpath' if on else 'loaded CUDA worker attachment receipt',
                              str(caught.exception))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ProfiledFastPathTests(unittest.TestCase):
    """The step profile (REMOTE_RA_STEP_PROFILE) on top of the fast path: the same outputs
    as both off, and complete profile lines (test_step_profile's checks)."""

    def test_profile_and_fast_path_together(self):
        import test_decouple_e2e as D
        import test_step_profile as TSP
        import remote_ra_step_profile as SP
        real_init = D.MultiHarness.__init__

        def init(self, *args, **kwargs):
            real_init(self, *args, **kwargs)
            F.attach_worker(self.owner, self.bridge, self.qwen)
        case = D.CASES[2]
        off = TSP.serve_sessions(case, 'full')
        lines = []
        with patch.object(D.MultiHarness, '__init__', init):
            on = TSP.serve_sessions(case, 'full', SP.Profiler(every=10, emit=lines.append))
        self.assertIsInstance(on['h'].owner.claims, F.ClaimTable)
        for name in ('done', 'trace', 'allocations', 'widths', 'files'):
            self.assertEqual(on[name], off[name], name)
        TSP.StepProfileEndToEndTests().check_lines(lines, on, case)


if __name__ == '__main__':
    unittest.main()
