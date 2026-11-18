"""host-trims-1111: engine-thread host work removed without changing a value (perf review 1110). CPU only.

- perf 1.5 (RequestDigestTextTests): the phase-1 request digest hashes the canonical JSON text
  of the prompt that render_claim hashed (request_path.prompt_hash keeps it, the binding takes
  it, ``phase1_request_digest`` cuts it at e) instead of encoding the prompt again; its value
  is ``digest(request)`` byte for byte. test_digest_once counts the call sites on the e2e plays.
- perf 2.2 (CaptureQueuedTests): TensorIO.capture queues each D2H copy into a pinned host buffer
  (non_blocking) between its two drains, one host wait per capture instead of one per tensor per
  block; a pageable buffer copies synchronously as before. The bytes are the same. Only a GPU run
  shows the time and that the buffers are pinned there (CPU torch never pins).
- perf 2.3 (WorkerFramesTests): EngineWorkerProxy files every worker frame as the JSON text
  copy_receipt parsed (bytes: not tracked by the garbage collector), and composition.WorkerFrames
  parses it when the frames are read; every read equals the former copies, and test_decouple_e2e's
  play writes the same evidence files, byte for byte, as with the former proxy (620b9e4's call).
- perf 2.4 (DeferredFramesTests): with --deferred-completion-io the engine thread takes a request's
  frames as filed and the writer parses them (deferred_completion.filed_frames, _Frames): the
  bytes equal the encoding of the parsed frames; without it the native terminal reads the startup
  config once per engine (evidence.terminal_config) and writes what 620b9e4's native_terminal wrote.
  test_deferred_completion's plays (switch on equals off, byte for byte) run the writer path.
- perf 2.5 (ReceiptDigestTests): the phase-1 result's text is made member by member and the signed
  worker receipt's member text gives the plan's receipt_digest (digest(dict(receipt)), byte for
  byte, when the text is ASCII without DEL) without encoding the receipt again.
"""
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import random
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / 'source-overlay'
sys.path.insert(0, str(OVERLAY))
sys.path.insert(0, str(ROOT))

from remote_ra_live_v2 import contract as live
from remote_ra_native_growing import contract as native
from remote_ra_native_growing.contract import RenderBinding, phase1_request_digest, remember_prompt_digest

try:
    import torch
except ImportError:
    torch = None


def binding(tokens):
    """A RenderBinding with these tokens (the other fields are not read here)."""
    values = {f.name: None for f in native.fields(RenderBinding)}
    values['tokens'] = tuple(tokens)
    return RenderBinding(**values)


def text_of(tokens):
    return json.dumps(list(tokens), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def other_fields(rng):
    """The phase-1 request's other members, with what the encoder escapes (non-ASCII, DEL,
    quotes, control characters), floats, nested dicts and lists."""
    words = ['séance', 'x\x7fy', 'q"uote', 'tab\t', '中文', 'plain', '']
    lease = dict(lease_id='phase1-' + '%032x' % rng.getrandbits(128), rows=[[rng.randrange(9)] * 3] * 2,
                 goal=None if rng.random() < .5 else dict(tokens=[rng.randrange(99) for _ in range(5)]),
                 incarnations={'0:%d' % k: rng.randrange(1 << 40) for k in range(4)},
                 sparse=dict(logical_matched_tokens=rng.randrange(1000), ratio=rng.random()))
    return dict(request='r-' + rng.choice(words), binding='%064x' % rng.getrandbits(256),
                session=rng.choice(words), turn=rng.randrange(50), local_turn=rng.randrange(50),
                generation=rng.choice(words), allocation_generation=rng.randrange(1 << 20),
                epoch=rng.choice(words), source='%064x' % rng.getrandbits(256), model=rng.choice(words),
                policy=rng.choice(['full', 'goal_query', 'observation_query']),
                prompt_digest='%064x' % rng.getrandbits(256), lease=lease,
                lease_digest='%064x' % rng.getrandbits(256))


def request_for(rng, b, end):
    request = other_fields(rng)
    request['tokens'] = list(b.tokens[:end])
    return request


class RequestDigestTextTests(unittest.TestCase):
    def setUp(self):
        native._PROMPT_TEXTS.clear()

    def kept(self, tokens):
        b = binding(tokens)
        remember_prompt_digest(b, hashlib.sha256(text_of(b.tokens)).hexdigest(), text=text_of(b.tokens))
        return b

    def test_the_value_is_digest_of_the_request(self):
        rng = random.Random(1111)
        for case in range(400):
            n = rng.choice([0, 1, 2, 17, rng.randrange(3, 3000)])
            tokens = [rng.choice([0, 7, 10, 99, 248000, rng.randrange(1 << 18)]) for _ in range(n)]
            end = rng.choice([0, n, max(n - 1, 0), max(n - 7, 0), rng.randrange(n + 1)])
            b = self.kept(tokens)
            request = request_for(rng, b, end)
            expected = live.digest(request)
            self.assertEqual(expected, native.digest(request))
            self.assertEqual(native._PROMPT_TEXTS[-1][0], b.tokens, case)
            self.assertEqual(phase1_request_digest(request, b), expected, (case, n, end))
            # Taken once: a second call (no kept text) encodes the request, the same value.
            self.assertTrue(all(t is not b.tokens for t, _ in native._PROMPT_TEXTS))
            self.assertEqual(phase1_request_digest(request, b), expected)

    def test_the_text_is_used_only_for_these_tokens(self):
        rng = random.Random(7)
        tokens = [rng.randrange(1 << 18) for _ in range(500)]
        b = self.kept(tokens)
        # Another binding with equal tokens (another tuple object): no text, digest(request).
        other = binding(tokens)
        request = request_for(rng, other, 493)
        calls = []
        real = native.digest
        try:
            native.digest = lambda value: calls.append(1) or real(value)
            self.assertEqual(phase1_request_digest(request, other), real(request))
            self.assertEqual(len(calls), 1)
            # The latest render's binding: its text, no digest(request).
            b2 = self.kept(tokens)
            request = request_for(rng, b2, 493)
            self.assertEqual(phase1_request_digest(request, b2), real(request))
            self.assertEqual(len(calls), 1)       # (the expected values call the real digest)
            # Overlapping admissions: renders r0..r9, then their phase 1s in another order; the
            # last HELD renders' texts are kept, the two oldest are encoded.
            renders = [self.kept([k] * 300 + tokens) for k in range(native.HELD + 2)]
            native._PROMPT_TEXTS[:] = native._PROMPT_TEXTS[-native.HELD:]
            order = renders[2:][::-1] + renders[:2]
            before = len(calls)
            for k, r in enumerate(order):
                request = request_for(rng, r, 790)
                self.assertEqual(phase1_request_digest(request, r), real(request))
                self.assertEqual(len(calls) - before, max(0, k - (native.HELD - 1)), k)
            # A cut longer than TEXT_CUT_LIMIT, a non-string key: digest(request).
            big = self.kept([3] * 6000)
            request = request_for(rng, big, 6000 - native.TEXT_CUT_LIMIT - 1)
            before = len(calls)
            self.assertEqual(phase1_request_digest(request, big), real(request))
            self.assertEqual(len(calls), before + 1)
            odd = self.kept(tokens)
            request = request_for(rng, odd, 400)
            request[1] = 'not a string key'
            with self.assertRaises(TypeError):
                real(request)                     # the encoder refuses mixed keys when sorting
            with self.assertRaises(TypeError):
                phase1_request_digest(request, odd)
        finally:
            native.digest = real

    def test_the_render_keeps_the_text_it_hashed(self):
        import request_path
        rng = random.Random(3)
        tokens = [rng.randrange(1 << 18) for _ in range(800)]
        value = request_path.prompt_hash(tokens)
        self.assertEqual(request_path.take_prompt_text([*tokens]), None)    # another list: nothing
        value2 = request_path.prompt_hash(tokens)
        self.assertEqual(value, value2)
        text = request_path.take_prompt_text(tokens)
        self.assertEqual(text, text_of(tokens))
        self.assertEqual(hashlib.sha256(text).hexdigest(), value)
        self.assertIsNone(request_path.take_prompt_text(tokens))           # handed over once
        # remember_prompt_digest without a text keeps nothing; at most HELD texts are kept.
        native._PROMPT_TEXTS.clear()
        remember_prompt_digest(binding(tokens), value)
        self.assertEqual(native._PROMPT_TEXTS, [])
        for k in range(native.HELD + 3):
            self.kept(tokens)
        self.assertEqual(len(native._PROMPT_TEXTS), native.HELD)


class Host:
    """A host buffer that says whether it is pinned and logs its copies (CPU torch cannot pin)."""
    def __init__(self, tensor, pinned, log):
        self.tensor, self.pinned, self.log = tensor, pinned, log

    def is_pinned(self):
        return self.pinned

    def copy_(self, source, non_blocking=False):
        self.log.append(('copy', non_blocking))
        self.tensor.copy_(source, non_blocking=non_blocking)
        return self

    def __getattr__(self, name):
        return getattr(self.tensor, name)


@unittest.skipUnless(torch is not None, 'CPU torch required')
class CaptureQueuedTests(unittest.TestCase):
    GDN = 1

    def io(self, pinned, log):
        from remote_ra_growing_worker.tensors import CpuTensorTestIO, LayerView

        class Logged(CpuTensorTestIO):
            def drain(self):
                log.append(('drain',))
                return super().drain()

            def _empty_host(self, sample):
                return Host(super()._empty_host(sample), pinned, log)
        g = torch.Generator().manual_seed(5)
        views = [LayerView('gdn0', self.GDN, 'hybrid', (torch.randn(9, 3, generator=g),
                                                         torch.randn(9, 2, 2, generator=g)), (0, 0)),
                 LayerView('gdn1', self.GDN, 'hybrid', (torch.randn(9, 4, generator=g),
                                                         torch.randn(9, 2, generator=g)), (0, 0))]
        return Logged(views, decoupled=True)

    def test_pinned_copies_are_queued_between_the_two_drains(self):
        for pinned in (True, False):
            with self.subTest(pinned=pinned):
                log = []
                io = self.io(pinned, log)
                payload = io.capture({self.GDN: (2, 5, 7)}, 'hybrid')
                pieces = 3 * 4
                self.assertEqual(log, [('drain',)] + [('copy', pinned)] * pieces + [('drain',)])
                for v in io.views:
                    for bid, got in zip((2, 5, 7), payload[v.name]):
                        for tensor, host in zip(v.tensors, got):
                            self.assertTrue(torch.equal(host.tensor, tensor[bid]))
                self.assertEqual(io._http_d2h_joined, sum(t[0].numel() * 4 for v in io.views for t in v.tensors) * 3)

    def test_the_pin_test(self):
        from remote_ra_growing_worker.tensors import _pinned
        self.assertFalse(_pinned(torch.empty(2)))           # CPU torch: never pinned
        self.assertFalse(_pinned(object()))
        self.assertTrue(_pinned(Host(None, True, [])))
        self.assertFalse(_pinned(Host(None, 1, [])))        # only a True answer queues


def former_call(self, operation, payload, *, query_tokens=None):
    """EngineWorkerProxy.call at 620b9e4, verbatim."""
    from remote_ra_live_v2.contract import require
    require(operation in self.OPERATIONS and type(payload) is dict and
            (query_tokens is None or operation in ("phase1", "phase1_collect", "goal_forward")),
            "bounded worker operation required")
    if self.before_call is not None and operation != "allocation_batch":
        self.before_call()
    values = self.executor.collective_rpc(
        "remote_ra_live_v2_dispatch", args=(operation, payload), timeout=30)
    require(type(values) is list and len(values) == 1 and
            type(values[0]) is dict, "one loaded worker receipt required")
    from remote_ra_http.evidence import copy_receipt
    if operation == "allocation_batch":
        receipts, entries = values[0].get("receipts"), payload.get("entries")
        require(type(receipts) is list and type(entries) is list and len(receipts) == len(entries),
                "one worker receipt per allocation batch entry required")
        for entry, item in zip(entries, copy_receipt(receipts)):
            self.http_receipts.setdefault(entry["request"], []).append(item)
        return values[0]
    item = copy_receipt(values[0])
    if query_tokens is not None:
        from remote_ra_http.config import digest
        item["query_token_sha256"] = digest(list(query_tokens))
    self.http_receipts.setdefault(payload.get("request", "session"), []).append(item)
    return values[0]


def former_proxy(stack):
    """EngineWorkerProxy as at 620b9e4: the copy taken at the call, a plain dict of copies."""
    from unittest.mock import patch
    from remote_ra_live_v2.composition import EngineWorkerProxy
    init = EngineWorkerProxy.__init__

    def plain_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.http_receipts = {}
    stack.enter_context(patch.object(EngineWorkerProxy, '__init__', plain_init))
    stack.enter_context(patch.object(EngineWorkerProxy, 'call', former_call))


def containers(value):
    """Objects the garbage collector tracks in a parsed frame."""
    if type(value) is dict:
        return 1 + sum(containers(v) for v in value.values())
    if type(value) is list:
        return 1 + sum(containers(v) for v in value)
    return 0


class WorkerFramesTests(unittest.TestCase):
    RECEIPTS = [
        ('prepare', dict(request='a'), None,
         dict(operation='prepare', binding='b' * 64, pages={2: 'x', 10: 'y', 1: (3, 4)}, ratio=0.1 + 0.2,
              text='s\u00e9ance \u4e2d\x7f"', none=None, flag=True, nested=[{'k': [1, (2, 3)]}])),
        ('phase1', dict(request='a'), (5, 6, 7),
         dict(operation='phase1', observation_vector=[0.5, -1e-300, 3.25e12], worker_receipt=dict(mac='m'))),
        ('allocation_batch', dict(entries=[dict(request='a'), dict(request='b'), dict(request='a')]), None,
         dict(operation='allocation_batch', receipts=[dict(operation='allocation_update', n=k, ids={3: k})
                                                      for k in range(3)])),
        ('goal_forward', dict(request='b'), [9],
         dict(operation='goal_forward', query_token_sha256='replaced', vector=[1.0])),
        ('close_session', dict(session='s'), None, dict(operation='close_session', reason='shutdown')),
    ]

    def proxies(self):
        from remote_ra_live_v2.composition import EngineWorkerProxy
        answers = []
        executor = SimpleNamespace(collective_rpc=lambda name, args, timeout: [answers.pop(0)])
        receipt = dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', loaded_sources_verified=True)
        new, old = EngineWorkerProxy(executor, receipt), EngineWorkerProxy(executor, receipt)
        old.http_receipts = {}
        return new, old, answers

    def test_reads_equal_the_former_copies(self):
        new, old, answers = self.proxies()
        for operation, payload, query, answer in self.RECEIPTS:
            for proxy, call in ((new, type(new).call), (old, former_call)):
                answers.append(answer)
                self.assertIs(call(proxy, operation, payload, query_tokens=query), answer)
        self.assertEqual(sorted(new.http_receipts), sorted(old.http_receipts))
        for key in old.http_receipts:
            self.assertEqual(new.http_receipts[key], old.http_receipts[key], key)
            self.assertEqual(new.http_receipts.get(key), old.http_receipts[key])
            # The same JSON, key order included (encoded sorts; the query hash comes last).
            self.assertEqual([json.dumps(f) for f in new.http_receipts[key]],
                             [json.dumps(f) for f in old.http_receipts[key]])
            # Fresh objects on every read.
            self.assertIsNot(new.http_receipts[key][0], new.http_receipts[key][0])
        self.assertEqual(dict(new.http_receipts.items()), old.http_receipts)
        self.assertEqual(new.http_receipts.values(), list(old.http_receipts.values()))
        self.assertIsNone(new.http_receipts.get('missing'))
        self.assertEqual(new.http_receipts.get('missing', []), [])
        self.assertEqual(new.http_receipts['a'][1]['query_token_sha256'], live.digest([5, 6, 7]))
        # Filed as bytes (or bytes with the query hash): nothing the collector tracks.
        gc.collect()
        for items in dict.values(new.http_receipts):
            for item in items:
                self.assertIn(type(item), (bytes, tuple))
                self.assertFalse(gc.is_tracked(item))
        self.assertGreater(sum(containers(f) for frames in old.http_receipts.values() for f in frames), 15)
        # pop (session_pool.release_requests) is dict's.
        self.assertEqual(len(new.http_receipts.pop('b')), 2)
        self.assertNotIn('b', new.http_receipts)

    def test_a_receipt_json_refuses_is_refused_at_the_call_as_before(self):
        for bad in (dict(operation='x', v=float('nan')), dict(operation='x', v=object())):
            new, old, answers = self.proxies()
            errors = []
            for proxy, call in ((new, type(new).call), (old, former_call)):
                answers.append(bad)
                with self.assertRaises((ValueError, TypeError)) as caught:
                    call(proxy, 'prepare', dict(request='a'))
                errors.append((type(caught.exception), str(caught.exception)))
                self.assertEqual(len(proxy.http_receipts), 0)
            self.assertEqual(errors[0], errors[1])

    @unittest.skipUnless(torch is not None, 'CPU torch required')
    def test_the_e2e_play_writes_the_same_files(self):
        import test_decouple_e2e as DE
        with ExitStack() as stack:
            former_proxy(stack)
            before = DE.play('goal_query', DE.CASES[2])
        after = DE.play('goal_query', DE.CASES[2])
        self.assertEqual(after['trace'], before['trace'])
        self.assertEqual(after['semantics'], before['semantics'])
        self.assertEqual(sorted(after['files']), sorted(before['files']))
        natives = [name for name in after['files'] if name.startswith('native-')]
        self.assertGreater(len(natives), 5)
        for name, raw in after['files'].items():
            self.assertEqual(raw, before['files'][name], name)


def former_native_terminal(endpoint, entry, operation):
    """remote_ra_http.evidence.native_terminal at 620b9e4, verbatim (module names prefixed)."""
    from remote_ra_http import evidence as ev
    from remote_ra_http.evidence import Path, decoupled, goal_text, kv_pools, require, save, sha256
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME', '0') != '1':
        return
    writer = getattr(endpoint.core, '_remote_ra_completion_writer', None)
    if writer is not None and writer.defer(endpoint, entry, operation):
        return
    binding = entry['binding']; core = endpoint.core
    value = ev.terminal_value(endpoint, entry, operation)
    selection = core._remote_ra_private_v2_endpoint.selection
    config = ev.load()
    output = Path(config['output_directory'])
    p = output / ('native-' + sha256(binding.request.encode()).hexdigest() + '.json')
    reference = goal_text(selection, output, value)
    if reference is not None: value['goal_text'] = reference
    if decoupled(config):
        value['kv_pools'] = dict(free_blocks_by_pool=ev.kv_pools(core)['free_blocks_by_pool'])
        if p.exists():
            old = json.loads(p.read_bytes())
            require(type(old.get('kv_pools')) is dict and
                    {k: v for k, v in old.items() if k != 'kv_pools'} == {k: v for k, v in value.items() if k != 'kv_pools'},
                    'terminal receipt changed on retry')
            return
    if p.exists(): require(json.loads(p.read_bytes()) == value, 'terminal receipt changed on retry')
    else: save(p, value)


class DeferredFramesTests(unittest.TestCase):
    def filed(self):
        from remote_ra_live_v2.composition import WorkerFrames
        from remote_ra_http.config import encoded
        rng = random.Random(24)
        frames = WorkerFrames()
        frames.file('a', encoded(dict(operation='prepare', ids={2: 1, 10: 2}, tokens=list(range(3000)))))
        frames.file('a', (encoded(dict(operation='phase1', v=[rng.random() for _ in range(5000)],
                                       pages={str(p): dict(values=[p] * 128, scale=.5) for p in range(40)})), 'f' * 64))
        for k in range(30):
            frames.file('a', encoded(dict(operation='allocation_update', step=k, text='\u00e9\x7f')))
        frames.file('a', encoded(dict(operation='complete', release=[[p] for p in range(2500)])))
        return frames

    def test_the_writer_parses_the_filed_frames_into_the_same_bytes(self):
        from remote_ra_live_v2 import composition, deferred_completion as D
        from remote_ra_http.config import encoded
        frames = self.filed()
        parsed = []
        real = composition.worker_frame

        def counted(item):
            parsed.append(1)
            return real(item)
        with patch.object(composition, 'worker_frame', counted):
            taken = D.filed_frames(frames, 'a')
            value = dict(request_id='a', worker_frames=taken, prompt_token_ids=list(range(5000)),
                         plan=dict(selected=[1, 2]), terminal_state='complete', admission_engine_ms=1.5)
            parts = D.snapshot(value)
            self.assertEqual(parsed, [])                  # the engine thread parses nothing
            raw = D.assemble(parts, cap=512)
            self.assertEqual(len(parsed), len(frames.raw('a')))
        self.assertEqual(raw, encoded(dict(value, worker_frames=frames['a'])))
        self.assertEqual(len(taken), 33)
        # Frames filed later do not reach a taken record; a plain dict's frames are taken as they are.
        frames.file('a', encoded(dict(operation='late')))
        self.assertEqual(D.assemble(parts, cap=512), raw)
        plain = {'a': [dict(operation='x')]}
        self.assertEqual(D.filed_frames(plain, 'a'), [dict(operation='x')])

    def test_the_native_terminal_reads_the_config_once(self):
        from remote_ra_http import evidence as ev
        for layout in (None, dict(kind='decoupled')):
            with self.subTest(decoupled=layout is not None), tempfile.TemporaryDirectory() as a, \
                    tempfile.TemporaryDirectory() as b:
                written = {}
                for name, terminal, out in (('former', former_native_terminal, a), ('after', ev.native_terminal, b)):
                    config = dict(output_directory=out, kv_layout=layout)
                    loads = []
                    core = SimpleNamespace(_remote_ra_private_v2_endpoint=SimpleNamespace(selection=None))
                    endpoint = SimpleNamespace(core=core)
                    with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'), \
                            patch.object(ev, 'load', lambda: loads.append(1) or config), \
                            patch.object(ev, 'kv_pools', lambda core: dict(free_blocks_by_pool=[3, 4])), \
                            patch.object(ev, 'terminal_value', lambda endpoint, entry, operation, **kw: dict(
                                request_id=entry['binding'].request, operation=operation, frames=[1, 'x'])):
                        for rid in ('r1', 'r2', 'r3', 'r2'):            # r2 again: a retry
                            terminal(endpoint, dict(binding=SimpleNamespace(request=rid)), 'complete')
                    written[name] = {p.name: p.read_bytes() for p in Path(out).iterdir()}
                    self.assertEqual(len(loads), 4 if name == 'former' else 1, name)
                self.assertEqual(written['after'], written['former'])
                self.assertEqual(len(written['after']), 3)


def json_value(rng, depth=0):
    kind = rng.randrange(9 if depth < 3 else 5)
    if kind == 0:
        return rng.randrange(-10**6, 10**6)
    if kind == 1:
        return rng.random() * 10 ** rng.randrange(-5, 15)
    if kind == 2:
        return rng.choice(['x', 'caf\u00e9', 'd\x7fl', 'q"\\', 'nl\n', '\u4e2d', ''])
    if kind == 3:
        return rng.choice([None, True, False])
    if kind == 4:
        return 'h' * rng.randrange(64)
    if kind in (5, 6):
        return [json_value(rng, depth + 1) for _ in range(rng.randrange(5))]
    return {rng.choice(['a', 'b', 'k%d' % rng.randrange(20), '\u00e9']): json_value(rng, depth + 1)
            for _ in range(rng.randrange(5))}


class ReceiptDigestTests(unittest.TestCase):
    def setUp(self):
        native._RECEIPT_DIGESTS.clear()       # a former-digest play leaves its receipts kept

    def test_members_make_the_same_text(self):
        from remote_ra_http.config import encoded
        from remote_ra_http.evidence import receipt_text_members
        rng = random.Random(25)
        for case in range(300):
            value = {k: json_value(rng) for k in rng.sample(['body', 'mac', 'worker_receipt', 'z', '\u00e9', 'a'], 4)}
            value['worker_receipt'] = dict(body=json_value(rng), mac='m' * 64)
            raw, member = receipt_text_members(value, 'worker_receipt')
            self.assertEqual(raw, encoded(value), case)
            self.assertEqual(member, encoded(value['worker_receipt']))

    def test_the_digest_is_digest_of_the_copy(self):
        from remote_ra_http.evidence import receipt_text
        rng = random.Random(26)
        encodes = []
        real = native.digest
        with patch.object(native, 'digest', lambda value: encodes.append(1) or real(value)):
            for case in range(300):
                receipt = dict(body=json_value(rng), mac=json_value(rng))
                text = receipt_text(receipt)
                native.remember_receipt_digest(receipt, text)
                ascii_safe = text.isascii() and b'\x7f' not in text
                before = len(encodes)
                self.assertEqual(native.receipt_digest(dict(receipt)), real(dict(receipt)), case)
                self.assertEqual(len(encodes) - before, 0 if ascii_safe else 1, case)
                # Used once.
                self.assertEqual(native._RECEIPT_DIGESTS, [])
            # Another member object (an equal copy): encoded.
            receipt = dict(body=dict(a=[1, 2]), mac='m')
            native.remember_receipt_digest(receipt, receipt_text(receipt))
            before = len(encodes)
            self.assertEqual(native.receipt_digest(dict(body=dict(a=[1, 2]), mac='m')), real(receipt))
            self.assertEqual(len(encodes) - before, 1)
            self.assertEqual(len(native._RECEIPT_DIGESTS), 1)
            extra = dict(receipt, more=1)
            self.assertEqual(native.receipt_digest(extra), real(extra))
            # Several kept (overlapping admissions): each found by its members, in any order; at most HELD.
            receipts = [dict(body=dict(k=[k]), mac='m%d' % k) for k in range(native.HELD + 2)]
            native._RECEIPT_DIGESTS.clear()
            for r in receipts:
                native.remember_receipt_digest(r, receipt_text(r))
            self.assertEqual(len(native._RECEIPT_DIGESTS), native.HELD)
            before = len(encodes)
            for r in reversed(receipts[2:]):
                self.assertEqual(native.receipt_digest(dict(r)), real(dict(r)))
            self.assertEqual(len(encodes) - before, 0)
            self.assertEqual(native.receipt_digest(dict(receipts[0])), real(dict(receipts[0])))
            self.assertEqual(len(encodes) - before, 1)      # dropped: the oldest beyond HELD

    @unittest.skipUnless(torch is not None, 'CPU torch required')
    def test_every_phase1_plan_digest_comes_from_its_rpc_text(self):
        import test_decouple_e2e as DE
        from remote_ra_native_growing import adapter
        runs = {}
        for name in ('former', 'after'):
            seen = []

            def taken(receipt, real=native.receipt_digest):
                held = len(native._RECEIPT_DIGESTS)
                value = real(receipt)
                seen.append((held > 0, len(native._RECEIPT_DIGESTS) == held - 1,
                             value == native.digest(dict(receipt))))
                return value
            replacement = (lambda receipt: native.digest(dict(receipt))) if name == 'former' else taken
            with patch.object(adapter, 'receipt_digest', replacement):
                runs[name] = DE.play('goal_query', DE.CASES[2])
            if name == 'after':
                self.assertGreater(len(seen), 5)
                self.assertTrue(all(held and used and equal for held, used, equal in seen), seen)
        self.assertEqual(runs['after']['trace'], runs['former']['trace'])
        self.assertEqual(runs['after']['files'], runs['former']['files'])


if __name__ == '__main__':
    unittest.main()
