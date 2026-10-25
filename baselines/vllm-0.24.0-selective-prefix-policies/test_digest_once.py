"""One prompt digest per binding, no repeated phase-1 payload digests (int-1008). CPU only.

Before this change a phase-1 admission (local turn > 0) hashed its prompt
eight times: the native request's canonical prompt hash at render, the
assignment's token digest, the phase-1 request's prompt_digest and
request_digest, the worker's re-check of prompt_digest, the probe's payload
digest at registration and twice more when the lease drained (once in the
phase-1 call itself, once on the engine's drain RPC). Now:

- the canonical prompt hash computed and checked at render is the binding's
  prompt digest (``remote_ra_native_growing.contract.prompt_digest``: kept with
  the binding's immutable token tuple), so phase 1 sends it without hashing;
- the worker no longer re-hashes its prepared prompt to compare it with
  prompt_digest (the request names the prepared binding by its minted
  identity and carries the phase-1 tokens, compared exactly; in the UniProc
  worker both values come from that binding, in one process);
- the probe compares a drained or replayed payload with its registered copy
  by atom identity (``contract.same_atoms``), and falls back to the digest it
  computed at registration when the payload is not made of those atoms
  (a payload that went through JSON).

The c0a74c2 source of every changed function is kept below, verbatim (FORMER,
generated with ast from ``git show c0a74c2:<file>``), so one process runs both:

- the values: test_stageb_e2e's script (legacy 784-token pool, JSON worker RPC)
  and test_decouple_e2e's (16-token Pages, in-process worker RPC as in the
  UniProc server) for full, goal_query and observation_query send identical
  worker RPC payloads (prompt_digest, request_digest, lease_digest), receive
  identical receipts and write identical evidence files, byte for byte;
  every prompt_digest is the canonical prompt hash of the prepared prompt and
  every request_digest the digest of its request;
- the counts: SHA-256 and HMAC calls during admissions (sys.setprofile, by
  call site) differ exactly by the removed digests, per phase-1 admission the
  prompt digest of Phase1Coordinator.run, the one of model_queries.phase1 and
  the two payload digests of NativeQueryForward.recover; with the in-process
  RPC nothing is hashed in their place, with the JSON RPC the drain's payload
  (new int objects) is compared by its digest once, as before;
- prompt_digest's memo, same_atoms (never true when the canonical JSON differs)
  and the probe's refusals of changed payloads.
"""
import __future__
import ast
from collections import Counter
from contextlib import ExitStack
import copy
import dataclasses
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
OVERLAY = ROOT / 'source-overlay'
POLICIES = (('full', 'goal_query', 'observation_query') if os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
            else ('goal_query',))
sys.path.insert(0, str(OVERLAY))
sys.path.insert(0, str(ROOT))

from remote_ra_live_v2 import contract as live
from remote_ra_live_v2.contract import LiveV2Error, canonical, plain_copy, same_atoms
from remote_ra_live_v2.lifecycle import EngineLifecycle
from remote_ra_live_v2.native_probe import NativeQueryForward
from remote_ra_native_growing import contract as native
from remote_ra_native_growing.contract import RenderBinding, prompt_digest, remember_prompt_digest

try:
    import torch
except ImportError:
    torch = None

FLAGS = __future__.annotations.compiler_flag
FORMER_LABEL = '<c0a74c2 %s>'


def compiled(module_name, source):
    """The former function, defined against the live module namespace."""
    module = importlib.import_module(module_name)
    scope = {}
    exec(compile(source, FORMER_LABEL % module_name, 'exec', FLAGS, dont_inherit=True), module.__dict__, scope)
    (function,) = scope.values()
    return module, function


def former(stack):
    """Every function this change touched, as it was at c0a74c2."""
    for module_name, target, source in FORMER:
        module, function = compiled(module_name, source)
        owner, _, name = target.rpartition('.')
        stack.enter_context(patch.object(getattr(module, owner) if owner else module, name, function))


def source_of(module_name, dotted):
    path = OVERLAY / (module_name.replace('.', '/') + '.py')
    text = path.read_text(encoding='utf-8')
    body, node = ast.parse(text).body, None
    for part in dotted.split('.'):
        (node,) = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part]
        body = node.body
    lines = text.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]
    indent = len(lines[0]) - len(lines[0].lstrip(' '))
    return ''.join(line[indent:] if line.startswith(' ' * indent) else line for line in lines)


class FormerTableTests(unittest.TestCase):
    def test_former_sources_compile_and_differ_from_current(self):
        self.assertEqual(len({(m, t) for m, t, _ in FORMER}), len(FORMER))
        for module_name, target, source in FORMER:
            with self.subTest(target=target):
                compiled(module_name, source)
                self.assertNotEqual(source, source_of(module_name, target))


# ---- what a run hashes during its admissions ------------------------------------------------

def label(code):
    """A code object's source file relative to the overlay (a former function: its module's file)."""
    name = code.co_filename
    if name.startswith('<c0a74c2 '):
        return name[len('<c0a74c2 '):-1].replace('.', '/') + '.py'
    path = Path(name)
    try:
        return path.relative_to(OVERLAY).as_posix()
    except ValueError:
        return None


class Hashing:
    """SHA-256 and HMAC calls made from runtime code during admissions, by call site:
    (file:function that called hashlib/hmac, file:function that called it).
    hashlib.sha256/new and hmac.digest are seen at their C call, hmac.new at its
    Python call (whose own C calls run in hmac.py and are not counted again)."""

    def __init__(self):
        self.calls, self.on = Counter(), False

    @staticmethod
    def site(frame):
        where = label(frame.f_code)
        if where is None:
            return None
        back = frame.f_back
        caller = label(back.f_code) if back is not None else None
        return ('%s:%s' % (where, frame.f_code.co_name),
                None if caller is None else '%s:%s' % (caller, back.f_code.co_name))

    def __call__(self, frame, event, arg):
        if not self.on:
            return
        if event == 'c_call':
            name = getattr(arg, '__name__', '')
            if (name in ('openssl_sha256', 'sha256', 'hmac_digest', 'hmac_new') or
                    (name == 'new' and getattr(arg, '__module__', '') in ('_hashlib', 'hashlib'))):
                site = self.site(frame)
                if site is not None:
                    self.calls[('sha256' if 'hmac' not in name else 'hmac',) + site] += 1
        elif event == 'call' and frame.f_code.co_name in ('new', 'digest') and \
                frame.f_code.co_filename.endswith('hmac.py') and frame.f_back is not None:
            site = self.site(frame.f_back)
            if site is not None:
                self.calls[('hmac',) + site] += 1

    def admissions(self, stack):
        """Count during every EngineLifecycle.admit (the engine and its in-process worker)."""
        admit = EngineLifecycle.admit
        hashing = self

        def counted(self, *args, **kwargs):
            previous = sys.getprofile()
            sys.setprofile(hashing)
            hashing.on = True
            try:
                return admit(self, *args, **kwargs)
            finally:
                hashing.on = False
                sys.setprofile(previous)
        stack.enter_context(patch.object(EngineLifecycle, 'admit', counted))


DIGEST = 'remote_ra_live_v2/contract.py:digest'


def removed(phase1):
    """The calls the former code made and this code does not, in a run with ``phase1``
    phase-1 admissions: the prompt digest of Phase1Coordinator.run, the worker's
    re-hash in model_queries.phase1 and the probe's two drain digests."""
    return Counter({('sha256', DIGEST, 'remote_ra_live_v2/phase1_coordinator.py:run'): phase1,
                    ('sha256', DIGEST, 'remote_ra_live_v2/model_queries.py:phase1'): phase1,
                    ('sha256', DIGEST, 'remote_ra_live_v2/native_probe.py:recover'): 2 * phase1})


def fallbacks(count):
    """The probe's digest comparisons of payloads not made of the registered atoms."""
    return Counter({('sha256', DIGEST, 'remote_ra_live_v2/native_probe.py:_same'): count} if count else {})


def check_digests(test, trace):
    """Every phase-1 request: prompt_digest is the canonical prompt hash of its prepared
    prompt, request_digest the digest of the rest of the request."""
    from contracts import canonical_prompt_hash
    prompts = {e['payload']['request']: e['payload']['tokens'] for e in trace if e['operation'] == 'prepare'}
    phase1 = [e['payload'] for e in trace if e['operation'] == 'phase1']
    for p in phase1:
        test.assertEqual(p['prompt_digest'], canonical_prompt_hash(prompts[p['request']]))
        test.assertEqual(p['prompt_digest'], live.digest(prompts[p['request']]))
        test.assertEqual(p['request_digest'], live.digest({k: v for k, v in p.items() if k != 'request_digest'}))
        test.assertEqual(p['lease_digest'], live.digest(p['lease']))
    return len(phase1)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EquivalenceTests(unittest.TestCase):
    """The former and the current code on the same deterministic scripts."""
    maxDiff = None

    def stage_b(self, policy, old):
        import test_host_trim as HT
        hashing = Hashing()
        with ExitStack() as stack:
            hashing.admissions(stack)
            stack.enter_context(patch.object(HT, 'former', former))
            run = HT.play(policy, old=old)
        return run, hashing.calls

    def decoupled(self, policy, old):
        import test_decouple_e2e as DE

        class InProcess(DE.Harness):
            """The UniProc worker: payloads and receipts are passed, not copied through JSON."""
            def __init__(self, policy, output, case, rpc='json', tasks=None):
                super().__init__(policy, output, case, rpc='inproc', tasks=tasks)
        hashing = Hashing()
        with ExitStack() as stack:
            hashing.admissions(stack)
            stack.enter_context(patch.object(DE, 'Harness', InProcess))
            if old:
                former(stack)
            run = DE.play(policy, DE.CASES[2])
        return run, hashing.calls

    def compare(self, before, after, keys):
        for key in keys:
            self.assertEqual(before[key], after[key], key)
        self.assertEqual(sorted(after['files']), sorted(before['files']))
        for name, raw in after['files'].items():
            self.assertEqual(raw, before['files'][name], name)

    # Default suite: goal_query (phase 1, Goal forwards: every digest); full and
    # observation_query in the slow pass (REMOTE_RA_SLOW_TESTS=1, README section 2.3).
    def test_stage_b_script_with_json_worker_rpc(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                (before, old), (after, new) = self.stage_b(policy, True), self.stage_b(policy, False)
                self.assertGreater(len(before['trace']), 20)
                self.compare(before, after, ('outputs', 'identities', 'trace', 'prepared', 'receipts',
                                             'allocations', 'views', 'forwards'))
                phase1 = check_digests(self, after['trace'])
                self.assertGreater(phase1, 5)
                # The JSON RPC hands the drain RPC a payload of new int objects: the probe
                # compares it with its registration digest there, as before (one digest
                # per phase-1 admission instead of two); nothing else is hashed more.
                self.assertEqual(old - new, removed(phase1))
                self.assertEqual(new - old, fallbacks(phase1))

    # Default suite: goal_query (phase 1, Goal forwards: every digest); full and
    # observation_query in the slow pass (REMOTE_RA_SLOW_TESTS=1, README section 2.3).
    def test_decoupled_script_with_in_process_worker_rpc(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                (before, old), (after, new) = self.decoupled(policy, True), self.decoupled(policy, False)
                self.compare(before, after, ('semantics', 'trace', 'allocations'))
                phase1 = check_digests(self, after['trace'])
                self.assertGreater(phase1, 5)
                self.assertEqual(old - new, removed(phase1))
                self.assertEqual(new - old, fallbacks(0))
                # What is left per phase-1 admission on the probe: its one payload digest.
                self.assertEqual(new[('sha256', DIGEST, 'remote_ra_live_v2/native_probe.py:_register')], phase1)
                self.assertNotIn(('sha256', DIGEST, 'remote_ra_live_v2/native_probe.py:recover'), new)
                # The engine: the native request's canonical prompt hash once per admission.
                admissions = sum(1 for e in after['trace'] if e['operation'] == 'prepare')
                self.assertEqual(new[('sha256', 'swebench_baseline/coordinator.py:canonical_prompt_hash',
                                      'request_path.py:observe_assigned_id')], admissions)


# ---- prompt_digest, same_atoms, the probe's comparisons ---------------------------------------

def binding(tokens, **fields):
    values = dict(epoch='e' * 64, session='s', request='r', turn=1, local_turn=1, rebuild=False,
                  generation='g', model='m', renderer='x', tokens=tuple(tokens), stable_end=1,
                  observation_span=(1, len(tokens)), native_prefix='p', nonce='n')
    values.update(fields)
    return RenderBinding(**values)


class Hashes:
    """Counts native_growing's digest calls (its module-level sha256)."""
    def __init__(self, stack):
        self.count = 0
        real = native.sha256

        def sha256(data=b''):
            self.count += 1
            return real(data)
        stack.enter_context(patch.object(native, 'sha256', sha256))


class PromptDigestTests(unittest.TestCase):
    def test_the_value_is_the_canonical_prompt_hash(self):
        from contracts import canonical_prompt_hash
        from remote_ra_http.config import digest as config_digest
        rng = random.Random(5)
        for n in (1, 2, 17, 4096):
            tokens = [rng.randrange(0, 151_000) for _ in range(n)]
            b = binding(tokens)
            value = prompt_digest(b)
            self.assertEqual(value, canonical_prompt_hash(tokens))
            self.assertEqual(value, live.digest(list(tokens)))
            self.assertEqual(value, config_digest(list(tokens)))

    def test_once_per_binding_and_never_stale(self):
        tokens = list(range(1000, 1300))
        with ExitStack() as stack:
            hashes = Hashes(stack)
            b = binding(tokens)
            value = prompt_digest(b)
            self.assertEqual((prompt_digest(b), prompt_digest(b), hashes.count), (value, value, 1))
            # Copies and pickles keep it, with the tuple it describes.
            for other in (copy.copy(b), copy.deepcopy(b), pickle.loads(pickle.dumps(b))):
                self.assertEqual(prompt_digest(other), value)
            self.assertEqual(hashes.count, 1)
            # A new object, or a replaced tokens field, is hashed again.
            replaced = dataclasses.replace(b, tokens=tuple(tokens[:-1]))
            self.assertEqual(prompt_digest(replaced), live.digest(tokens[:-1]))
            object.__setattr__(b, 'tokens', tuple(tokens[1:]))
            self.assertEqual(prompt_digest(b), live.digest(tokens[1:]))
            self.assertEqual(hashes.count, 3)
            # Equality, hash and asdict ignore the memo.
            self.assertEqual(binding(tokens), dataclasses.replace(binding(tokens)))
            self.assertNotIn('_prompt_digest', dataclasses.asdict(b))
            # Objects that are not frozen bindings of a tuple are hashed every time, unmarked.
            loose = NS(tokens=list(tokens))
            self.assertEqual(prompt_digest(loose), value)
            self.assertEqual((vars(loose), hashes.count), (dict(tokens=list(tokens)), 4))

    def test_the_render_seam_hands_its_checked_hash_over(self):
        tokens = list(range(5, 400))
        with ExitStack() as stack:
            hashes = Hashes(stack)
            b = binding(tokens)
            remember_prompt_digest(b, live.digest(tokens))
            self.assertEqual((prompt_digest(b), hashes.count), (live.digest(tokens), 0))
            loose = NS(tokens=tuple(tokens))
            remember_prompt_digest(loose, 'a' * 64)       # nothing kept on another object
            self.assertEqual(vars(loose), dict(tokens=tuple(tokens)))
        for bad in ('a' * 63, None, b'a' * 64):
            with self.assertRaises(native.Abort):
                remember_prompt_digest(binding(tokens), bad)


def json_value(rng, depth=0):
    kind = rng.randrange(10 if depth < 3 else 6)
    if kind == 0:
        return rng.randrange(0, 300)                       # cached small ints
    if kind == 1:
        return rng.randrange(10 ** 5, 10 ** 6)             # distinct int objects
    if kind == 2:
        return rng.choice((None, True, False, 'ab', '', 0.5, -0.0, 1.0))
    if kind == 3:
        return [rng.randrange(0, 10 ** 6) for _ in range(rng.randrange(0, 6))]
    if kind == 4:
        return tuple(rng.randrange(0, 9) for _ in range(rng.randrange(0, 3)))
    if kind == 5:
        return 'x%d' % rng.randrange(10 ** 6)
    if kind in (6, 7):
        return {'k%d' % rng.randrange(20): json_value(rng, depth + 1) for _ in range(rng.randrange(0, 4))}
    return [json_value(rng, depth + 1) for _ in range(rng.randrange(0, 4))]


def leaves(value, path=()):
    if type(value) in (list, tuple):
        for i, item in enumerate(value):
            yield from leaves(item, path + (i,))
    elif type(value) is dict:
        for key, item in value.items():
            yield from leaves(item, path + (key,))
    else:
        yield path


def replaced(value, path, new):
    if not path:
        return new
    head, rest = path[0], path[1:]
    if type(value) is dict:
        return {k: (replaced(v, rest, new) if k == head else v) for k, v in value.items()}
    items = [replaced(v, rest, new) if i == head else v for i, v in enumerate(value)]
    return items if type(value) is list else tuple(items)


class SameAtomsTests(unittest.TestCase):
    def test_true_only_when_the_canonical_json_is_the_same(self):
        rng = random.Random(29)
        seen = Counter()
        for _ in range(3000):
            a = dict(payload=json_value(rng), tokens=[rng.randrange(10 ** 5, 10 ** 6) for _ in range(50)])
            for b in (plain_copy(a), copy.deepcopy(a)):
                self.assertTrue(same_atoms(a, b))
            through_json = json.loads(json.dumps(a))
            self.assertFalse(same_atoms(a, through_json))       # new int objects: not shown, digest decides
            self.assertEqual(canonical(a), canonical(through_json))
            # One leaf replaced by an equal or a different value of another object.
            path = rng.choice(list(leaves(a)))
            old = a
            for key in path:
                old = old[key]
            for new in (True if old == 1 else 1, float(old) if type(old) is int else 7, -0.0, 0.0,
                        int(str(old)) if type(old) is int else 'y', str(old) + '', None):
                b = replaced(plain_copy(a), path, new)
                if same_atoms(a, b):
                    seen['same'] += 1
                    self.assertEqual(canonical(a), canonical(b))
                elif canonical(a) == canonical(b):
                    seen['digest'] += 1
        self.assertGreater(seen['same'], 0)
        self.assertGreater(seen['digest'], 0)
        # Container shapes and keys must match.
        self.assertTrue(same_atoms([1, (2, 3)], [1, (2, 3)]))
        self.assertFalse(same_atoms([1, 2], (1, 2)))           # equal JSON; the digest decides
        self.assertFalse(same_atoms({1: 'a'}, {1: 'a'}))       # non-str keys: the digest decides
        self.assertFalse(same_atoms({'a': 1}, {'a': 1, 'b': 2}))
        self.assertFalse(same_atoms([1], [1, 2]))


class ProbeComparisonTests(unittest.TestCase):
    """NativeQueryForward.recover and the replay check on a registered payload."""

    def payload(self, n=4096):
        rng = random.Random(3)
        lease = dict(lease_id='phase1-' + 'a' * 32, rows=[[rng.randrange(1, 9000) for _ in range(40)] for _ in range(4)],
                     incarnations={str(k): rng.randrange(10 ** 6) for k in range(30)}, goal=None)
        return dict(request='r1', binding='b' * 64, tokens=[rng.randrange(1000, 151000) for _ in range(n)],
                    prompt_digest='d' * 64, lease=lease, lease_digest='l' * 64, request_digest='q' * 64)

    def driver(self, payload):
        probe = NativeQueryForward.__new__(NativeQueryForward)
        probe.leases = {payload['lease']['lease_id']: dict(payload=plain_copy(payload), digest=live.digest(payload),
                                                          identity='identity')}
        probe.drains = {payload['lease']['lease_id']: 'receipt'}
        return probe

    def test_a_drain_of_the_registered_payload_is_accepted_without_hashing(self):
        payload = self.payload()
        probe = self.driver(payload)
        with patch.object(live, 'canonical', side_effect=AssertionError('hashed')):
            self.assertEqual(probe.recover(payload), ('receipt', 'identity'))
            self.assertEqual(probe.recover(plain_copy(payload)), ('receipt', 'identity'))   # the engine's copy
        # A payload that went through JSON is checked by its digest.
        self.assertEqual(probe.recover(json.loads(json.dumps(payload))), ('receipt', 'identity'))

    def test_a_changed_payload_is_refused(self):
        payload = self.payload()
        probe = self.driver(payload)
        changes = [lambda p: p['tokens'].__setitem__(17, p['tokens'][17] + 1),
                   lambda p: p['tokens'].append(5),
                   lambda p: p['lease']['rows'][2].__setitem__(0, 1),
                   lambda p: p.__setitem__('prompt_digest', 'e' * 64),
                   lambda p: p['lease']['incarnations'].__setitem__('3', 0),
                   lambda p: p['tokens'].__setitem__(0, 1) or p['tokens'].__setitem__(1, True)]
        for change in changes:
            for through_json in (False, True):
                bad = plain_copy(payload) if not through_json else json.loads(json.dumps(payload))
                change(bad)
                with self.assertRaisesRegex(LiveV2Error, 'phase-1 drain payload changed'):
                    probe.recover(bad)
        # True where the registered payload holds 1: equal in Python, not in JSON.
        ones = self.payload()
        ones['tokens'][5] = 1
        probe = self.driver(ones)
        bad = plain_copy(ones)
        bad['tokens'][5] = True
        with self.assertRaisesRegex(LiveV2Error, 'phase-1 drain payload changed'):
            probe.recover(bad)

    def test_the_replay_check_compares_alike(self):
        payload = self.payload(64)
        prior = dict(payload=plain_copy(payload), digest=live.digest(payload))
        self.assertTrue(NativeQueryForward._same(prior, payload))
        self.assertTrue(NativeQueryForward._same(prior, json.loads(json.dumps(payload))))
        changed = plain_copy(payload)
        changed['tokens'][3] += 1
        self.assertFalse(NativeQueryForward._same(prior, changed))


# ---- the c0a74c2 code of every changed function ------------------------------------------------

FORMER = (
    ('remote_ra_live_v2.private_endpoint', 'SplitPrivateEndpoint.render_claim', r'''
def render_claim(self, body):
    self.verify_identity()
    self.sessions.verify()
    require(set(body) == {"request", "session", "turn", "generation", "core", "source",
                         "external_id", "prompt_hash"} and
            type(body["session"]) is str and body["session"] in self.sessions.ids,
            "exact server-owned render envelope required")
    transcript = self.transcripts[body["session"]]
    if len(self.sessions.ids) > 1:
        # Several tasks: the request's task message must route to its session.
        from .sessions import route
        source = body["source"]
        require(route(self.tasks, source.get("messages") if type(source) is dict else None)
                == self.sessions.index(body["session"]), "request task differs from its session")
    native = self.assembly.dispatcher._decode_core(
        base64.b64decode(body["core"], validate=True))
    rid = body.get("request", body.get("assigned_request"))
    require(native.request_id == rid and native.client_index == 0 and
            native.current_wave == 0 and native.data_parallel_rank in (None, 0),
            "native assigned request mismatch")
    from request_path import AssignedId, observe_assigned_id
    require(observe_assigned_id(native) == AssignedId(rid, body["external_id"],
                                                    body["prompt_hash"]),
            "registered fields differ from native assignment")
    request, wave = self.core.preprocess_add_request(native)
    # Policy/session already authenticated in RegisteredRouteBridge.
    # Legacy connector metadata must not initiate competing stores.
    request.kv_transfer_params = None
    require(wave == 0 and request.request_id == rid, "preprocess identity changed")
    if transcript is not None:
        from .mswea_protocol import check_sampling
        check_sampling(request.sampling_params, transcript.stop_ids)
    session = self.adapter.sessions[body["session"]]
    # Random50 seed: vLLM's chained hash of the last full hash block, computed
    # by the engine core anyway (no extra hashing here). Its block is the
    # hash_block_size: 784 tokens on the legacy pool, the FA Page P (16) on a
    # decoupled one, so the seed names the prefix up to its last full P-token
    # block there (still an actual native prefix identity; F3 would need
    # another seed source if owned requests stop hashing).
    hashes = getattr(request, "block_hashes", ())
    native_prefix = (hashes[-1].hex() if hashes else sha256(b"").hexdigest())
    # turn is the session's request ordinal (it survives rebuilds); the
    # renderer checks the session's transcript against the session-local
    # turn and sets rebuild on a transcript reset.
    binding, mac = self.assembly.renderer.render_growing(body["source"], request,
        self.adapter.authority, session=body["session"], turn=body["turn"],
        local_turn=session.local_turn, generation=body["generation"],
        model=session.model, renderer=session.renderer, native_prefix=native_prefix,
        transcript=transcript)
    require(binding.request not in self.claims, "private binding replay")
    if binding.rebuild:
        self.rebuild(binding)
    source=body['source']
    self.claims[binding.request] = dict(request=request, binding=binding, mac=mac,
        conversation=source.get('conversation',source.get('messages')))
    return request, binding
'''),
    ('remote_ra_live_v2.phase1_coordinator', 'Phase1Coordinator.run', r'''
def run(self, binding, geo, *, allocation_generation, goal_payload=None):
    """Lease, run and drain phase 1 for local turn > 0; return its Phase1Result."""
    adapter = self.adapter
    adapter.authority.owner()
    P = adapter.page_tokens
    s = adapter.sessions[binding.session]
    require(geo.local_turn > 0 and geo.page_tokens == P, 'phase 1 runs only after local turn 0')
    require(not s.poisoned and s.pending is None and s.local_turn == geo.local_turn and
            s.turn == binding.turn and binding.request not in adapter.requests,
            'phase 1 must precede the generation reserve')
    connector = self.core.scheduler.connector
    require(connector is not None and not connector.has_pending_push_work(),
            'native connector must be quiescent before phase-1 allocation')
    # Process-wide, like the worker: no lease starts while any session's
    # lease lacks its joined drain (native_probe refuses the alias).
    require(not self.records, 'prior phase-1 lease lacks a drain, adoption or abort')
    require(geo.e - geo.C <= self.max_tokens, 'PHASE1_TOO_LONG: e - C exceeds one native forward')
    require(type(allocation_generation) is int and allocation_generation > 0,
            'worker allocation generation required')
    require(hex64(s.boundary) and s.boundary != ZERO_DIGEST, 'phase 1 needs a committed carry')
    owner, ns = adapter.owner, s.namespace
    resident = owner.sessions.get(ns, {})
    R = tuple(sorted(resident))
    pin = owner.tails.get(ns)
    require(R == tuple(s.resident) and all(p < geo.Q // P for p in R) and
            (pin is not None) == geo.tail and
            (pin is None or (pin.page, pin.start, pin.valid_end) == (geo.Q // P, geo.Q, geo.C)),
            'phase-1 geometry differs from the session pins')
    att, mam, count = self.groups()
    goal_tokens = None
    if goal_payload is not None:
        require(s.policy == 'goal_query' and type(goal_payload) is str,
                'a Goal forward belongs to goal_query only')
        # Upstream _goal_query_text, tokenized raw without special tokens.
        from runtime_renderer import PLAIN_GOAL_SCHEMA
        goal_tokens = [int(t) for t in self.renderer(dict(schema=PLAIN_GOAL_SCHEMA,
                                                         text=goal_payload)).token_ids]
        require(0 < len(goal_tokens) <= self.max_tokens,
                'PHASE1_TOO_LONG: Goal forward exceeds one native forward')
    fresh = list(geo.fresh_pages)
    Mt = mamba_tokens(P, self.layout)
    M = -(-geo.e // Mt)                      # Mamba row length (ceil(e/P) shared)
    width = -(-len(goal_tokens) // P) if goal_tokens else 0
    mwidth = -(-len(goal_tokens) // Mt) if goal_tokens else 0
    shared = self.mamba_pool is self.pool
    if shared:
        needed = len(att) * len(fresh) + len(mam) + (len(att) * width + len(mam) if goal_tokens else 0)
        require(needed <= self.pool.get_num_free_blocks(), 'insufficient native blocks for phase 1')
    else:
        fa, gdn = len(att) * (len(fresh) + width), len(mam) * (1 + bool(goal_tokens))
        require(fa <= self.pool.get_num_free_blocks() and gdn <= self.mamba_pool.get_num_free_blocks(),
                'insufficient native blocks for phase 1: FA %d of %d free, Mamba %d of %d free' %
                (fa, self.pool.get_num_free_blocks(), gdn, self.mamba_pool.get_num_free_blocks()))
    from .pool_incarnations import install
    versions = install(self.pool)
    tables = (versions, versions if shared else install(self.mamba_pool))
    allocated = []
    def new(n, pool=self.pool):
        blocks = pool.get_new_blocks(n)
        allocated.extend(blocks)
        return list(blocks)
    try:
        fresh_fa = {g: new(len(fresh)) for g in att}
        fresh_mam = {g: new(1, self.mamba_pool)[0] for g in mam}
        goal_fa = {g: new(width) for g in att} if goal_tokens else {}
        goal_mam = {g: new(1, self.mamba_pool)[0] for g in mam} if goal_tokens else {}
    except BaseException:
        self._free(allocated)
        raise
    try:
        tail_ids = [pin.groups[i][0].block_id for i in range(len(att))] if pin else []
        rows = []
        for g in range(count):
            if g in att:
                i = att.index(g)
                rows.append([resident[p][i][0].block_id for p in R] + tail_ids[i:i + 1] +
                            [b.block_id for b in fresh_fa[g]])
            else:
                rows.append([0] * (M - 1) + [fresh_mam[g].block_id])
        pools = group_pools(att, count, self.layout)
        ids = [(pools[g], b) for g, row in enumerate(rows) for b in row if b]
        require(all(b in tables[pool] for pool, b in ids),
                'phase-1 block allocation incarnation missing')
        fa_owned = [b.block_id for g in att for b in fresh_fa[g]]
        owned = fa_owned + [fresh_mam[g].block_id for g in mam]
        goal = None
        if goal_tokens:
            grows = [[b.block_id for b in goal_fa[g]] if g in att else
                     [0] * (mwidth - 1) + [goal_mam[g].block_id] for g in range(count)]
            gids = [(pools[g], b) for g, row in enumerate(grows) for b in row if b]
            # The KV zeroer writes FullAttention tensors only: the zero list holds
            # pool-0 ids (every id in the shared pool).
            goal = dict(lease_id='goal-' + secrets.token_hex(16), tokens=goal_tokens, rows=grows,
                        owned=[b for _, b in gids], zero=[b for pool, b in gids if shared or pool == 0],
                        incarnations={block_key(pool, b): tables[pool][b] for pool, b in gids},
                        payload_sha256=sha256(goal_payload.encode()).hexdigest())
        lease = dict(lease_id='phase1-' + secrets.token_hex(16), request=binding.request,
            binding=binding.identity, session=binding.session, namespace=ns,
            generation=allocation_generation, epoch=binding.epoch,
            source=adapter.authority.source, local_turn=geo.local_turn, page_tokens=P,
            start=geo.C, end=geo.e, sealed_start=geo.Q, candidate_end=geo.L,
            resident_pages=list(R),
            tail=(dict(page=pin.page, valid_end=pin.valid_end, block_ids=[[b] for b in tail_ids])
                  if pin else None),
            tail_candidate=geo.tail_candidate, rows=rows,
            borrowed=[resident[p][i][0].block_id for p in R for i in range(len(att))] + tail_ids,
            owned=owned, zero=list(owned if shared else fa_owned),
            run_fresh={str(p): [[fresh_fa[g][k].block_id] for g in att] for k, p in enumerate(fresh)},
            incarnations={block_key(pool, b): tables[pool][b] for pool, b in ids},
            sparse=dict(logical_matched_tokens=geo.Q,
                        dropped_full_attention_tokens=geo.Q - len(R) * P),
            carry_digest=s.boundary, goal=goal)
        lease['mac'] = adapter.authority.sign(lease)
        # The worker parses the same shape.
        parse_lease(lease, attention_groups=att, layout=self.layout)
        request = dict(request=binding.request, binding=binding.identity,
            session=binding.session, turn=binding.turn, local_turn=geo.local_turn,
            generation=binding.generation, allocation_generation=allocation_generation,
            epoch=binding.epoch, source=adapter.authority.source, model=binding.model,
            policy=s.policy, tokens=list(binding.tokens[:geo.e]),
            prompt_digest=digest(list(binding.tokens)), lease=lease, lease_digest=digest(lease))
        request['request_digest'] = digest(request)
    except BaseException:
        self._free(allocated)
        raise
    # Private copies of what was sent (deepcopy semantics; the prompt list is
    # copied in one call).
    self.records[binding.request] = dict(binding=binding.identity, lease=plain_copy(lease),
        request=plain_copy(request), drained=False,
        fa={p: tuple((fresh_fa[g][k],) for g in att) for k, p in enumerate(fresh)},
        transient=tuple(fresh_mam.values()) + tuple(b for g in att for b in goal_fa.get(g, ()))
                  + tuple(goal_mam.values()))
    # Evidence: the query tokens are the Goal tokens when a Goal forward ran,
    # else the observation T[o:e] for observation_query.
    query_tokens = (goal_tokens if goal_tokens else
                    list(binding.tokens[geo.o:geo.e]) if s.policy == 'observation_query' else None)
    try:
        result = self.proxy.call('phase1', request, query_tokens=query_tokens)
        return self._result(result, binding, geo, s, goal_tokens)
    finally:
        # A failed forward must expose its own signed drain through the
        # separate idempotent RPC; never infer release from an error.
        self.recover(binding.request)
'''),
    ('remote_ra_live_v2.model_queries', 'phase1', r'''
def phase1(owner, backend, rec, payload):
    """Check the engine phase-1 request against the prepared turn, run it, sign it."""
    a, identity = rec['assignment'], rec['identity']
    C, e, P = a.computed_start, a.observation_end, a.page_tokens
    require(a.local_turn > 0 and rec['phase1'] is None and 'plan' not in rec,
            'phase 1 runs once, after prepare and before reserve')
    require(payload['allocation_generation'] == rec['generation'] and
            payload['binding'] == rec['binding'] and payload['source'] == identity.source and
            payload['model'] == identity.model and
            (payload['session'], payload['turn'], payload['generation'], payload['local_turn'])
            == (identity.session, identity.turn, identity.generation, a.local_turn) and
            list(payload['tokens']) == list(rec['tokens'][:e]) and
            payload['prompt_digest'] == digest(list(rec['tokens'])) and
            hex64(payload['request_digest']), 'stale phase-1 allocation/render')
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
    driver = getattr(owner.runner, '_remote_ra_native_query_forward', None)
    require(callable(getattr(driver, 'phase1', None)),
            'LOADED_QUERY_FORWARD_UNAVAILABLE: native phase-1 driver is not attached')
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
'''),
    ('remote_ra_live_v2.native_probe', 'NativeQueryForward._register', r'''
def _register(self,payload):
    from .contract import parse_lease
    lease=parse_lease(payload['lease'],attention_groups=self.owner.attention_groups,layout=self.layout)
    rid=lease.lease_id
    record=self.owner.records[payload['request']]
    identity=record['identity']
    require(record['binding']==payload['binding']
            and record['generation']==payload['allocation_generation']
            and lease.lease_digest==payload['lease_digest']
            and lease.request==payload['request']
            and lease.session==identity.session
            and lease.generation==record['generation']
            and lease.source==identity.source
            and lease.epoch==payload['epoch']
            and sha256(lease.epoch.encode()).hexdigest()==identity.epoch,
            'foreign native phase-1 lease')
    prior=self.leases.get(rid)
    if prior is not None:
        require(prior['digest']==digest(payload),'phase-1 replay changed payload')
        return prior
    # Every (pool, id) the lease names: borrowed pins are read-only, owned
    # blocks are zeroed. Neither may be claimed by a generation or held by
    # another undrained lease. Ids are per pool (one pool unless decoupled).
    blocks=lease.block_keys()
    require(not any(key not in self.drains for key in self.leases) and
            not any((lease.pool_of(g),b) in blocks for g,b in self.owner.claims),
            'phase-1 physical allocation alias')
    prior=dict(payload=plain_copy(payload),digest=digest(payload),lease=lease,
               identity=identity,generation=record['generation'])
    self.leases[rid]=prior
    return prior
'''),
    ('remote_ra_live_v2.native_probe', 'NativeQueryForward.recover', r'''
def recover(self,payload):
    """Join/clean or tombstone a never-started RPC. Delayed work is rejected."""
    rid=payload['lease']['lease_id']
    prior=self.leases.get(rid)
    if prior is None:prior=self._register(payload)
    require(prior['digest']==digest(payload),'phase-1 drain payload changed')
    if rid in self.drains:return self.drains[rid],prior['identity']
    require(not self.executing and (self.active is None or self.active['lease'].lease_id==rid),
            'probe execution not joined: references must remain held')
    lease=prior['lease']
    self.backend.io.drain()
    self._cleanup_request(rid)
    if lease.goal is not None:self._cleanup_request(lease.goal.lease_id)
    self.backend.io.drain()
    require(self.runner.execute_model_state is None,'probe left sampling state')
    body=dict(operation='phase1_drain',binding=payload['binding'],lease_id=rid,
              lease_digest=lease.lease_digest,drained=True)
    receipt=self.backend.sign_receipt(prior['identity'],'phase1_drain',body)
    self.drains[rid]=receipt;self.active=None
    return receipt,prior['identity']
'''),
)


if __name__ == '__main__':
    unittest.main()
