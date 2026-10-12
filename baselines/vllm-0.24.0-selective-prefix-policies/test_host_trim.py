"""Host-side trims of the scheduler and worker hot path (ms2-host-trim). CPU only.

Per request (b):
- RenderBinding.identity is an opaque token minted once per binding (a per-process
  random prefix and a sequence number, 64 lowercase hex), not digest(asdict(b)).
- Authority.render returns a 256-bit random render capability instead of an HMAC
  over that digest, and authenticate accepts the binding object it minted, with
  every field still the object it was minted with, without reading its tokens;
  any other object must equal the minted fields exactly (types and values, every
  token). The binding never leaves the engine process (render -> claim -> reserve,
  on the engine thread), so no full-prompt digest is left on this path.
Per step (a):
- The worker forward record is neither hashed nor signed (SHA-256 of the hidden
  rows, an HMAC and two NativeIdentity digests per request per step); it carries
  a forward sequence number. Only Page-closing rows are copied to the host; one
  device join per step reads the finiteness of every scheduled row and each
  request still checks its own rows.
- The allocator Signature is built once per function; the allocation_update row
  check and Allocation's validator are plain loops; native_terminal keeps the
  frames' JSON copies instead of copying them again.

The 40e8b33 code of every changed function is kept below, verbatim apart from
imports (FORMER_*), so one process runs both:
- test_stageb_e2e's script (seven turns, a rebuild, a continuing turn and a cancel;
  full, goal_query, observation_query) gives identical generated tokens, worker RPC
  payloads and receipts, scheduled intervals, native allocations, lifecycle views
  and evidence files (byte for byte) with the former code and with this code when
  the identity is aligned to the former digest; the forward records are equal
  apart from the removed hidden_sha256/identity/mac and the added sequence.
- With the minted identity, the evidence and RPCs differ from the aligned run only
  in the binding identity and the values derived from it (CHANGED).
- test_ms2_worker's interleaved three-session batched script: the same with the
  former worker functions.
- After warm-up no SHA-256 or HMAC runs per decode step (calls from runtime
  source files counted with sys.setprofile), and no binding is hashed.
"""
from collections import Counter
from contextlib import ExitStack
import copy
import dataclasses
from dataclasses import asdict
from hashlib import sha256
import hashlib
import inspect
import itertools
import json
import operator
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import test_stageb_e2e as S        # the stage-B end-to-end harness and script
import test_stageb_worker as W
import test_ms2_worker as MW

from remote_ra_growing_worker import tensors
from remote_ra_http import evidence
from remote_ra_http.evidence import engine_closed
from remote_ra_live_v2 import engine_boundary
from remote_ra_live_v2.allocation_owner import RunnerAllocationOwner
from remote_ra_live_v2.contract import Allocation, closed_pages, require as live_require
from remote_ra_live_v2.qwen import QwenAlignLifecycle
from remote_ra_live_v2.worker_composition import WorkerBridge
from remote_ra_native_growing import contract
from remote_ra_native_growing.contract import Abort, Authority, RenderBinding

try:
    import torch
except ImportError:
    torch = None

OVERLAY = str(ROOT/'source-overlay')
FIELDS = tuple(f.name for f in dataclasses.fields(RenderBinding))
HEX64 = re.compile(r'[0-9a-f]{64}')


# ---- the 40e8b33 code of every changed function ------------------------------------------

_ATOMS = frozenset((str, int, bool, float, type(None)))


def _immutable(value):
    """No later mutation can change value's JSON: an atom or a flat tuple of atoms."""
    kind = type(value)
    return kind in _ATOMS or (kind is tuple and set(map(type, value)) <= _ATOMS)


def former_identity(self):
    """digest(asdict(self)), hashed once per binding, not on every read.

    The scheduler reads it several times per request and engine step; the
    digest covers the whole prompt (35-60K tokens, 6-11 ms). The value is
    kept with the field objects it was computed from, and only when they
    are all immutable (atoms, flat tuples of atoms: what binding_shape
    admits). A read returns it while each field is still that same object,
    so the frozen fields cannot change it; anything else is recomputed as
    before. Not a field: equality, hash, repr and asdict ignore it.
    """
    values = contract._field_values(self)
    memo = self.__dict__.get("_identity_memo")
    if memo is not None and all(map(operator.is_, memo[0], values)):
        return memo[1]
    identity = contract.digest(asdict(self))
    if all(map(_immutable, values)):
        object.__setattr__(self, "_identity_memo", (values, identity))
    return identity


FORMER_IDENTITY = property(former_identity)


def former_render(self, native_request, *, session, turn, local_turn, rebuild, generation,
                  model, renderer, tokens, stable_prefix_tokens, observation_span,
                  native_prefix):
    require = contract.require
    self.owner()
    rid = native_request.request_id
    require(type(rid) is str and rid and rid not in self._used and
            len(self._used) < self.max_attempts, "request collision/replay/limit")
    require(type(stable_prefix_tokens) is tuple and
            tuple(tokens[:len(stable_prefix_tokens)]) == stable_prefix_tokens,
            "canonical stable render is not an exact token prefix")
    require(tuple(native_request.prompt_token_ids) == tuple(tokens),
            "renderer tokens differ from actual assigned native request")
    require(type(turn) is int and turn >= 0 and all(type(x) is str and x for x in
            (session, generation, model, renderer, native_prefix)), "missing identity")
    b = RenderBinding(epoch=self.epoch, session=session, request=rid, turn=turn,
                      local_turn=local_turn, rebuild=rebuild, generation=generation,
                      model=model, renderer=renderer, tokens=tuple(tokens),
                      stable_end=len(stable_prefix_tokens),
                      observation_span=observation_span, native_prefix=native_prefix,
                      nonce=contract.secrets.token_hex(16))
    contract.binding_shape(b)
    self._used.add(rid)
    self._requests[rid] = (native_request, b)
    return b, self.sign(asdict(b))


def former_authenticate(self, request, binding, mac):
    self.owner()
    self.verify(asdict(binding), mac)
    contract.require(binding.epoch == self.epoch and
                     self._requests.get(binding.request) == (request, binding) and
                     self._requests[binding.request][0] is request and
                     tuple(request.prompt_token_ids) == binding.tokens,
                     "foreign native request or changed render")


def former_after_forward(self, identity, hidden_states):
    require = live_require
    state = self._state(identity)
    lease, a = state.active, state.assignment
    require(lease is not None and not state.poisoned,
            "missing active Qwen step")
    try:
        import torch
        require(isinstance(hidden_states, torch.Tensor) and
                hidden_states.ndim == 2 and
                hidden_states.shape[0] == lease.end - lease.start and
                hidden_states.shape[1] > 0 and
                bool(torch.isfinite(hidden_states).all()),
                "exact finite unpadded hidden rows required")
        self.backend.forward_complete(identity, lease)
        state.latest_hybrid = lease.hybrid.slots
        state.latest_state_step = lease.end
        rows = hidden_states.detach().to(device="cpu", dtype=torch.float32).clone()
        P = a.page_tokens
        from remote_ra_live_v2.query_selection import normalized
        # Every phase (header, cold prefill, decode) closes Pages; keep
        # only each closing row, at its first computation (spec 3.9).
        for page in closed_pages(lease.start, lease.end, P):
            require(page not in state.page_vectors, "Page vector recomputed")
            state.page_vectors[page] = normalized(
                rows[(page + 1) * P - 1 - lease.start].tolist())
        # Native preprocess moves the align slot only after a Page-aligned
        # end, so latest is needed only there; finish captures the carry
        # candidate at the cursor otherwise (spec 7.3 follow-up).
        if lease.end % P == 0:
            self.backend.capture_latest_hybrid(
                identity, lease.hybrid.slots, lease.end,
                state.allocation_generation)
        state.cursor = lease.end
        state.prompt_complete = state.cursor >= a.prompt_tokens
        body = dict(request=identity.request,
                    allocation_generation=state.allocation_generation,
                    start=lease.start, end=lease.end, phase=lease.phase,
                    hidden_sha256=hashlib.sha256(rows.numpy().tobytes()).hexdigest(),
                    position_evidence=state.position_evidence,
                    system_valid=False)
        receipt = self.backend.sign_receipt(identity, "forward", body)
        state.receipts.append(receipt)
        state.active = None
        return receipt
    except BaseException:
        state.poisoned = True
        raise


def former_bridge_after_forward(self, runner, scheduler_output, hidden_states):
    """Hidden rows [offset, offset + count) to each request's lease, batch order."""
    if self.owner is not None:
        receipts = []
        for request, _, offset, count in self.owner.scheduled_rows(runner, scheduler_output):
            state = self.assignments[request]
            # Pages are sealed and published at terminal complete (spec 3.8).
            receipts.append(self.qwen.after_forward(state.assignment.identity,
                                                    hidden_states[offset:offset + count]))
        return receipts[0] if len(receipts) == 1 else tuple(receipts)
    callback = getattr(scheduler_output, "remote_ra_live_v2_after", None)
    if callback is not None:
        return callback(self.qwen, hidden_states)


def former_allocation_update(self, rec, payload):
    require = live_require
    def update(backend,qwen):
        rows=tuple(tuple(tuple(v) for v in group) for group in payload['rows'])
        require(all(len(v)==2 and all(type(x) is int and x>=0 for x in v)
                    and ((v[0]==0)==(v[1]==0)) for row in rows for v in row),
                'invalid native allocation incarnation row')
        ids=[b for row in rows for b,_ in row if b]
        require(len(ids)==len(set(ids)),'native live pool IDs alias across groups')
        from remote_ra_live_v2.chunk_protocol import worker_interval
        worker_interval(rec,qwen.requests[rec['identity'].request],payload['start'],payload['count'])
        rec['native_rows']=rows
        rec['native_interval']=(payload['start'],payload['count'])
        return dict(operation='allocation_update',binding=payload['binding'],drained=True)
    return update


def former_allocation_post_init(self):
    require = live_require
    require(all(type(v) is int and v >= 0 for v in
                (self.group, self.block, self.generation)),
            "non-negative allocation identity required")
    require(self.block > 0, "null/kernel IDs cannot be allocation leases")
    require(type(self.incarnation) is int and self.incarnation>=0,'invalid allocation incarnation')
    require(self.kind in {"attention", "mamba", "suffix", "decode"},
            "unknown allocation kind")


def former_native_terminal(endpoint,entry,operation):
    ev = evidence
    require = ev.require
    if os.environ.get('REMOTE_RA_HTTP_RUNTIME','0')!='1':return
    binding=entry['binding'];core=endpoint.core
    require(entry['drain_receipt'].get('drained') is True,'terminal without native drain')
    proxy=endpoint.worker
    frames=ev.copy_receipt(proxy.http_receipts.get(binding.request,[]))
    require(frames,'native worker receipts absent')
    boundary=core._remote_ra_live_v2_boundary
    prepared=boundary.prepared.get(binding.request,{})
    plan=prepared.get('plan')
    selection=core._remote_ra_private_v2_endpoint.selection
    query=selection.authorized.get(binding.request)
    value=dict(schema='remote-ra-native-http-terminal-v1',request_id=binding.request,
        session_id=binding.session,session_index=core._remote_ra_live_session_authority.index(binding.session),
        epoch=binding.epoch,turn=binding.turn,
        local_turn=binding.local_turn,rebuild=binding.rebuild,
        generation=binding.generation,operation=operation,terminal_state=entry['state'],
        binding=binding.identity,prompt_token_ids=list(binding.tokens),observation_span=binding.observation_span,
        stable_end=binding.stable_end,page_tokens=core._remote_ra_growing_adapter.page_tokens,
        plan=plan,query=query,worker_frames=frames,drain=ev.copy_receipt(entry['drain_receipt']),
        admission_engine_ms=entry.get('admission_engine_ms'),  # stage 2: decode stall of this admission
        # stage 2, goal_async: a parked goal_query Goal's wait and construction time
        **{k:entry[k] for k in ('goal_async_ms','goal_construct_ms') if k in entry})
    p=Path(ev.load()['output_directory'])/('native-'+sha256(binding.request.encode()).hexdigest()+'.json')
    if p.exists():require(json.loads(p.read_bytes())==value,'terminal receipt changed on retry')
    else:ev.save(p,value)


def former(stack):
    """Every changed function as it was at 40e8b33."""
    stack.enter_context(patch.object(RenderBinding, 'identity', FORMER_IDENTITY))
    stack.enter_context(patch.object(Authority, 'render', former_render))
    stack.enter_context(patch.object(Authority, 'authenticate', former_authenticate))
    stack.enter_context(patch.object(QwenAlignLifecycle, 'after_forward', former_after_forward))
    stack.enter_context(patch.object(WorkerBridge, 'after_forward', former_bridge_after_forward))
    stack.enter_context(patch.object(engine_boundary, '_signature', inspect.signature))
    stack.enter_context(patch.object(RunnerAllocationOwner, '_allocation_update', former_allocation_update))
    stack.enter_context(patch.object(Allocation, '__post_init__', former_allocation_post_init))
    stack.enter_context(patch.object(evidence, 'native_terminal', former_native_terminal))


FORMER_CODE = frozenset(f.__code__ for f in (
    former_identity, former_render, former_authenticate, former_after_forward,
    former_bridge_after_forward, former_allocation_update, former_allocation_post_init,
    former_native_terminal))


# ---- deterministic runs ------------------------------------------------------------------

def plain(value):
    """The JSON form of what a run sent or kept."""
    return json.loads(json.dumps(value, default=lambda o: getattr(o, 'block_id', repr(o))))


def deterministic(stack):
    """secrets.token_bytes/token_hex, the evidence clocks (perf_counter_ns,
    monotonic_ns), the payload and identity sequences from counters: runs that make
    the same calls in the same order draw the same keys, nonces, lease ids, tokens
    and timings. The render capability (secrets.token_hex(32) in the native
    contract; the former render drew none) has a stream of its own."""
    draws, ticks, capabilities = itertools.count(), itertools.count(1), itertools.count(1)

    def token_bytes(n=32):
        out = b''
        while len(out) < n:
            out += sha256(b'ms2-host-trim %d' % next(draws)).digest()
        return out[:n]

    def token_hex(n=32):
        return token_bytes(n).hex()
    clock = lambda: 1_000_000 * next(ticks)
    stack.enter_context(patch('secrets.token_bytes', token_bytes))
    stack.enter_context(patch('secrets.token_hex', token_hex))
    stack.enter_context(patch.object(contract, 'secrets', NS(
        token_bytes=token_bytes,
        token_hex=lambda n=32: token_hex(n) if n != 32 else '%064x' % (0xcafe << 200 | next(capabilities)))))
    stack.enter_context(patch('time.perf_counter_ns', clock))
    stack.enter_context(patch('time.monotonic_ns', clock))
    stack.enter_context(patch.object(tensors, '_PAYLOAD_SEQUENCE', itertools.count(1)))
    stack.enter_context(patch.object(contract, '_IDENTITY_PREFIX', 'd' * 48))
    stack.enter_context(patch.object(contract, '_IDENTITY_SEQUENCE', itertools.count(1)))


class Hashing:
    """SHA-256/HMAC calls made from runtime source files while ``on`` (sys.setprofile).

    hashlib.sha256 (and hashlib.new, hmac.digest) are counted at their C call when
    the calling frame is in source-overlay; hmac.new at its Python call when its
    caller is."""

    def __init__(self):
        self.calls, self.on = Counter(), False

    @staticmethod
    def runtime(code):
        """A runtime source file, or one of the 40e8b33 functions kept above."""
        return code.co_filename.startswith(OVERLAY) or code in FORMER_CODE

    def __call__(self, frame, event, arg):
        if not self.on:
            return
        if event == 'c_call':
            name = getattr(arg, '__name__', '')
            if (name in ('openssl_sha256', 'sha256', 'hmac_digest', 'hmac_new') or
                    (name == 'new' and getattr(arg, '__module__', '') in ('_hashlib', 'hashlib'))):
                if self.runtime(frame.f_code):
                    self.calls[(Path(frame.f_code.co_filename).name, frame.f_code.co_name)] += 1
        elif event == 'call' and frame.f_code.co_name in ('new', 'digest') and \
                frame.f_code.co_filename.endswith('hmac.py'):
            caller = frame.f_back
            if caller is not None and self.runtime(caller.f_code):
                self.calls[(Path(caller.f_code.co_filename).name, caller.f_code.co_name)] += 1


def play(policy, *, old=False, aligned=False, count=False):
    """test_stageb_e2e.run_policy's requests: the script, a rebuild, a continuing turn
    and a cancel, then shutdown. ``old``: the 40e8b33 functions; ``aligned``: this
    code with the identity the former digest. Returns what the run sent and wrote."""
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, W.vllm_stubs()))
        stack.enter_context(patch.object(W, 'P', S.P))
        stack.enter_context(patch.object(W.Table, 'block_size', S.P))
        stack.enter_context(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        value = S.config(policy, output)
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        deterministic(stack)
        if old:
            former(stack)
        elif aligned:
            stack.enter_context(patch.object(RenderBinding, 'identity', FORMER_IDENTITY))
        binding_digests = Counter()
        real_digest = contract.digest

        def counted(value):
            if type(value) is dict and set(value) == set(FIELDS):
                binding_digests[value['request']] += 1
            return real_digest(value)
        stack.enter_context(patch.object(contract, 'digest', counted))
        h = S.Harness(policy, output)
        trace, hashing, decode_steps, forwards = [], Hashing(), [0], []
        after_forward = h.qwen.after_forward

        def forward(identity, hidden_states, **kwargs):
            receipt = after_forward(identity, hidden_states, **kwargs)
            forwards.append(dict(receipt))
            return receipt
        h.qwen.after_forward = forward

        def collective_rpc(method, args, timeout):
            operation, payload = args
            sent = plain(payload)
            reply = h.collective_rpc(method, args, timeout)
            trace.append(dict(operation=operation, payload=sent, receipt=plain(reply[0])))
            return reply
        h.proxy.executor = NS(collective_rpc=collective_rpc)
        if count:
            # Decode steps after each request's first: one allocation_update RPC and one
            # execute_model each, the steps counted while sys.setprofile watches.
            step = h.step

            def watched(request, count_, first):
                decode = (request.num_computed_tokens >= len(request.prompt_token_ids) and
                          len(request.output_token_ids) > 1)
                if not decode:
                    return step(request, count_, first)
                decode_steps[0] += 1
                sys.setprofile(hashing)
                hashing.on = True
                try:
                    return step(request, count_, first)
                finally:
                    hashing.on = False
                    sys.setprofile(None)
            h.step = watched
        infos = [h.turn(observation, target=target, eos=eos, chunk=chunk)
                 for observation, target, eos, chunk, _ in S.SCRIPT]
        observation, target, _ = S.REBUILD
        infos.append(h.turn(messages=h.messages[:-1] + [dict(role='user', content=observation)],
                            target=target))
        observation, target, _ = S.AFTER
        infos.append(h.turn(observation, target=target))
        cancel = h.submit(h.messages + [dict(role='user', content=S.CANCEL)])
        h.generate(cancel, S.X.reply('abc'), stop_after=2)
        views = [i['view'] for i in infos] + [cancel['view'], h.lifecycle.cancel(cancel['ident'])]
        h.evidence.failed(NS(internal_id=cancel['rid']), RuntimeError('client disconnected'))
        bindings = [i['binding'] for i in infos + [cancel]]
        prepared = {rid: {k: v for k, v in record.items()
                          if k in ('plan', 'ranking', 'selected_count', 'submitted', 'native_tables',
                                   'native_allocation', 'scheduled_cursor', 'scheduled_intervals')}
                    for rid, record in h.boundary.prepared.items()}
        receipts = {'%s/%s' % key: value for key, value in h.boundary.receipts.items()}
        outputs = {i['rid']: list(i['request'].output_token_ids) for i in infos + [cancel]}
        h.close()
        engine_closed(h.core)
        files = {path.name: path.read_bytes() for path in sorted(output.iterdir())}
        return dict(trace=trace, prepared=plain(prepared), receipts=plain(receipts),
                    allocations=plain(h.kv.calls), views=plain(views), files=files,
                    identities=[b.identity for b in bindings], bindings=bindings,
                    formers=[real_digest(asdict(b)) for b in bindings],
                    forwards=forwards, outputs=outputs, binding_digests=dict(binding_digests),
                    hashing=hashing.calls, decode_steps=decode_steps[0])


def walk(a, b, path, out):
    """Every path where two JSON values differ: (path, left, right)."""
    if type(a) is not type(b):
        out.append((path, a, b))
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append((path + '/' + str(k), a.get(k, '<missing>'), b.get(k, '<missing>')))
            else:
                walk(a[k], b[k], path + '/' + str(k), out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append((path, 'len %d' % len(a), 'len %d' % len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, '%s[]' % path, out)
    elif a != b:
        out.append((path, a, b))
    return out


# The values that change with the binding identity: the identity itself wherever it
# is named, and what is computed over a value that holds it (a MAC, a digest).
CHANGED = {
    # worker RPC payloads (trace) and the plan
    'payload/binding', 'payload/plan/binding', 'plan/binding',
    'payload/lease/binding', 'payload/lease/mac', 'payload/lease_digest', 'payload/request_digest',
    'payload/phase1_request/binding', 'payload/phase1_request/lease/binding',
    'payload/phase1_request/lease/mac', 'payload/phase1_request/lease_digest',
    'payload/phase1_request/request_digest', 'payload/plan/phase1/lease_digest',
    'payload/catalog/*/binding',
    'payload/plan/phase1/receipt_digest', 'plan/phase1/lease_digest', 'plan/phase1/receipt_digest',
    # worker receipts (trace) and the frames in the evidence
    'receipt/binding', 'receipt/request_digest', 'receipt/lease_digest', 'receipt/worker_receipt/mac',
    'receipt/worker_receipt/body/binding', 'receipt/worker_receipt/body/request_digest',
    'receipt/worker_receipt/body/lease_digest', 'receipt/new_pages/*/binding',
    'receipt/publication/mac', 'receipt/plan', 'receipt/worker_binding',
    'worker_frames[]/binding', 'worker_frames[]/request_digest', 'worker_frames[]/lease_digest',
    'worker_frames[]/worker_receipt/mac', 'worker_frames[]/worker_receipt/body/binding',
    'worker_frames[]/worker_receipt/body/request_digest', 'worker_frames[]/worker_receipt/body/lease_digest',
    'worker_frames[]/new_pages/*/binding', 'worker_frames[]/publication/mac',
    'drain/binding', 'drain/plan', 'drain/worker_binding', 'drain/new_pages/*/binding', 'drain/publication/mac',
    # EngineRpcBoundary's terminal receipts
    'plan', 'worker_binding', 'new_pages/*/binding', 'publication/mac',
    # the native terminal evidence and the query (Selector V2) receipt
    'binding', 'query/body/binding', 'query/body/phase1/lease_digest', 'query/mac',
    'native_allocation/binding', 'scheduled_intervals[]/binding',
}


def shape(path):
    """A difference path with Page numbers folded (new_pages/12 -> new_pages/*) and
    its leading list marker and container key dropped."""
    path = re.sub(r'/\d+(?=/|$)', '/*', path)
    return re.sub(r'^(\[\])?/', '', path)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class FormerEquivalenceTests(unittest.TestCase):
    maxDiff = None

    def check_aligned(self, before, after):
        self.assertEqual(before['outputs'], after['outputs'])
        self.assertEqual(before['identities'], after['identities'])
        for key in ('trace', 'prepared', 'receipts', 'allocations', 'views'):
            self.assertEqual(before[key], after[key], key)
        self.assertEqual(sorted(after['files']), sorted(before['files']))
        for name, raw in after['files'].items():
            self.assertEqual(raw, before['files'][name], name)
        # The worker forward records, every one in order: equal apart from what they
        # no longer carry (hidden_sha256, identity, mac) and the added sequence.
        old, new = before['forwards'], after['forwards']
        self.assertGreater(len(old), 100)
        self.assertEqual(len(old), len(new))
        for o, n in zip(old, new):
            self.assertEqual(set(o), {'identity', 'operation', 'body', 'mac'})
            self.assertEqual(set(n), {'operation', 'body'})
            self.assertEqual(o['operation'], n['operation'])
            self.assertTrue(HEX64.fullmatch(o['body']['hidden_sha256']))
            self.assertEqual({k: v for k, v in o['body'].items() if k != 'hidden_sha256'},
                             {k: v for k, v in n['body'].items() if k != 'sequence'})
        self.assertEqual([n['body']['sequence'] for n in new], list(range(1, len(new) + 1)))

    def test_former_and_new_code_send_and_write_the_same(self):
        for policy in ('full', 'goal_query', 'observation_query'):
            with self.subTest(policy=policy):
                before = play(policy, old=True)
                aligned = play(policy, aligned=True)
                self.assertGreater(Counter(t['operation'] for t in before['trace'])['allocation_update'], 100)
                self.check_aligned(before, aligned)
                # Former: three digests of each binding (render MAC, authenticate,
                # identity); now none, whatever reads the aligned identity.
                self.assertEqual(set(before['binding_digests'].values()), {3})

    def test_minted_identity_changes_only_the_identity_and_what_is_computed_over_it(self):
        for policy in ('full', 'goal_query', 'observation_query'):
            with self.subTest(policy=policy):
                aligned, now = play(policy, aligned=True), play(policy)
                self.assertEqual(aligned['outputs'], now['outputs'])
                self.assertEqual(now['binding_digests'], {})
                ids = now['identities']
                self.assertTrue(all(HEX64.fullmatch(i) for i in ids) and len(set(ids)) == len(ids))
                self.assertEqual(ids, ['d' * 48 + '%016x' % k for k in range(1, len(ids) + 1)])
                self.assertEqual(aligned['identities'], now['formers'])
                swap = dict(zip(now['identities'], aligned['identities']))
                found, unexpected = set(), set()

                def compare(a, n, allowed, origin):
                    for path, left, right in walk(a, n, '', []):
                        where = shape(path)
                        found.add(where)
                        if where not in allowed:
                            unexpected.add((origin, where))
                        self.assertTrue(type(left) is str and type(right) is str and
                                        HEX64.fullmatch(left) and HEX64.fullmatch(right), (origin, path))
                        if where.endswith('binding') and right in swap:
                            self.assertEqual(swap[right], left, (origin, path))
                for key in ('trace', 'prepared', 'receipts', 'views'):
                    # The run's own containers: a list of RPCs, request ids, receipt keys.
                    a = aligned[key] if type(aligned[key]) is list else list(aligned[key].values())
                    n = now[key] if type(now[key]) is list else list(now[key].values())
                    compare(a, n, CHANGED, key)
                self.assertEqual(aligned['allocations'], now['allocations'])
                self.assertEqual(sorted(aligned['files']), sorted(now['files']))
                for name in aligned['files']:
                    a, n = aligned['files'][name], now['files'][name]
                    if name.endswith('.jsonl'):
                        a = [json.loads(line) for line in a.splitlines()]
                        n = [json.loads(line) for line in n.splitlines()]
                    else:
                        a, n = json.loads(a), json.loads(n)
                    # The runtime row names the native file by its SHA-256.
                    compare(a, n, {'native_evidence_sha256'} if name == 'runtime.jsonl' else CHANGED, name)
                self.assertEqual(unexpected, set())
                self.assertTrue({'binding', 'payload/binding', 'receipt/binding', 'worker_frames[]/binding',
                                 'plan/binding'} <= found, found)
                # Every evidence file names each binding by its minted identity.
                natives = [json.loads(raw) for name, raw in now['files'].items() if name.startswith('native-')]
                self.assertEqual(sorted(v['binding'] for v in natives), sorted(ids))
                for v in natives:
                    self.assertTrue(all(f.get('binding', v['binding']) == v['binding'] for f in v['worker_frames']))

    def test_batched_worker_steps_are_unchanged(self):
        # test_ms2_worker: three sessions, interleaved turns, several rows per step.
        sids = ('sa', 'sb', 'sc')
        turns = MW.interleave(*[(sid, MW.script(k)) for k, sid in enumerate(sids)])

        def run(old):
            with ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, W.vllm_stubs()))
                deterministic(stack)
                if old:
                    former(stack)
                h = MW.BatchHarness(mrope=True)
                for sid in sids:
                    h.open(sid)
                h.drive(list(turns))
                return dict(summary={sid: v for sid, v in MW.by_session(h).items()},
                            receipts=plain(h.receipts), frames=list(h.frames), io=plain(h.io.log),
                            forwards=[[{k: v for k, v in r['body'].items() if k not in ('hidden_sha256', 'sequence')}
                                       for r in info['receipts_raw']] for info in h.done],
                            widths=max(len(s) for s in h.runner.steps))
        with patch.object(MW.BatchHarness, 'finish', finish_keeping_records):
            before, after = run(True), run(False)
        self.assertEqual(before['widths'], 3)
        for key in ('summary', 'receipts', 'frames', 'io', 'forwards'):
            self.assertEqual(before[key], after[key], key)


def finish_keeping_records(self, rid):
    """BatchHarness.finish, keeping the request's full forward records."""
    records = [dict(r) for r in self.qwen.requests[rid].receipts]
    info = ORIGINAL_FINISH(self, rid)
    info['receipts_raw'] = records
    return info


ORIGINAL_FINISH = MW.BatchHarness.finish


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class NoHashingPerStepTests(unittest.TestCase):
    def test_no_sha256_or_hmac_per_decode_step(self):
        for policy in ('full', 'goal_query'):
            with self.subTest(policy=policy):
                now = play(policy, count=True)
                self.assertGreater(now['decode_steps'], 50)
                self.assertEqual(now['hashing'], Counter())
                # The former code hashed four times per request per decode step:
                # the hidden rows, the receipt MAC and two NativeIdentity digests.
                before = play(policy, old=True, count=True)
                self.assertEqual(before['decode_steps'], now['decode_steps'])
                self.assertEqual(sum(before['hashing'].values()), 4 * before['decode_steps'], before['hashing'])

    def test_rendering_and_authenticating_hash_no_binding(self):
        authority = Authority(epoch='e' * 64, loaded_source='c' * 64)
        tokens = tuple((i * 7919 + 13) % 248_000 for i in range(60_000))
        hashing = Hashing()
        native = NS(request_id='r', prompt_token_ids=list(tokens))
        sys.setprofile(hashing)
        hashing.on = True
        try:
            b, mac = authority.render(native, session='s', turn=0, local_turn=0, rebuild=False,
                                      generation='g', model='m', renderer='x', tokens=tokens,
                                      stable_prefix_tokens=tokens[:100], observation_span=None,
                                      native_prefix='p')
            authority.authenticate(native, b, mac)
            for _ in range(10):
                identity = b.identity
        finally:
            hashing.on = False
            sys.setprofile(None)
        self.assertEqual(hashing.calls, Counter())
        self.assertTrue(HEX64.fullmatch(identity) and HEX64.fullmatch(mac))


# ---- the identity and the render capability ------------------------------------------------

def render(authority, rid, tokens, *, turn=0, local_turn=0, rebuild=False, stable=1, span=None):
    native = NS(request_id=rid, prompt_token_ids=list(tokens))
    b, mac = authority.render(native, session='s', turn=turn, local_turn=local_turn, rebuild=rebuild,
                              generation='g', model='m', renderer='x', tokens=tokens,
                              stable_prefix_tokens=tuple(tokens[:stable]), observation_span=span,
                              native_prefix='p')
    return native, b, mac


class IdentityTests(unittest.TestCase):
    def test_identity_is_a_minted_hex64_token_kept_by_the_binding(self):
        authority = Authority(epoch='e' * 64, loaded_source='c' * 64)
        tokens = tuple(range(100, 400))
        bindings = [render(authority, 'r%d' % k, tokens, stable=40)[1] for k in range(5)]
        ids = [b.identity for b in bindings]
        self.assertTrue(all(HEX64.fullmatch(i) for i in ids))
        self.assertEqual(len(set(ids)), 5)
        prefix = {i[:48] for i in ids}
        self.assertEqual(len(prefix), 1)                     # one process prefix
        self.assertEqual([int(i[48:], 16) for i in ids],
                         list(range(int(ids[0][48:], 16), int(ids[0][48:], 16) + 5)))
        b = bindings[0]
        self.assertIs(b.identity, b.identity)
        self.assertNotEqual(b.identity, contract.digest(asdict(b)))
        # Not a field: equality, hash, repr and asdict ignore it.
        fresh = RenderBinding(**{name: getattr(b, name) for name in FIELDS})
        self.assertEqual(asdict(b), asdict(fresh))
        self.assertTrue(b == fresh and hash(b) == hash(fresh))
        self.assertEqual(repr(b), repr(fresh))
        self.assertNotIn('_identity', repr(b))
        # A new object has its own identity; a copy or a pickle keeps it.
        self.assertNotEqual(fresh.identity, b.identity)
        self.assertNotEqual(dataclasses.replace(b).identity, b.identity)
        for clone in (copy.copy(b), copy.deepcopy(b), pickle.loads(pickle.dumps(b))):
            self.assertEqual(clone, b)
            self.assertEqual(clone.identity, b.identity)

    def test_a_binding_built_elsewhere_gets_one_identity_at_its_first_read(self):
        b = RenderBinding('e' * 64, 's', 'm', 0, 0, False, 'g', 'm', 'x', (1, 2, 3), 1, None, 'p', 'n')
        self.assertNotIn('_identity', b.__dict__)
        first = b.identity
        self.assertTrue(HEX64.fullmatch(first))
        self.assertEqual([b.identity for _ in range(3)], [first] * 3)
        self.assertNotEqual(RenderBinding(*dataclasses.astuple(b)).identity, first)


class AuthenticateTests(unittest.TestCase):
    def setUp(self):
        self.authority = Authority(epoch='e' * 64, loaded_source='c' * 64)
        self.tokens = tuple(range(100, 400))
        self.native, self.b, self.mac = render(self.authority, 'r0', self.tokens, stable=40)

    def test_the_minted_binding_and_its_capability_authenticate(self):
        self.authority.authenticate(self.native, self.b, self.mac)
        self.assertTrue(HEX64.fullmatch(self.mac))
        other = render(self.authority, 'r1', self.tokens, stable=40)
        self.assertNotEqual(other[2], self.mac)

    def test_a_wrong_or_foreign_capability_is_refused(self):
        _, b1, mac1 = render(self.authority, 'r1', self.tokens, stable=40)
        for mac in (mac1, 'f' * 64, self.mac[:-1], None, 7):
            with self.subTest(mac=mac), self.assertRaisesRegex(Abort, 'MAC mismatch'):
                self.authority.authenticate(self.native, self.b, mac)
        other = Authority(epoch='e' * 64, loaded_source='c' * 64)
        with self.assertRaisesRegex(Abort, 'MAC mismatch'):
            other.authenticate(self.native, self.b, self.mac)

    def test_an_equal_copy_is_checked_field_by_field(self):
        for clone in (copy.copy(self.b), copy.deepcopy(self.b), pickle.loads(pickle.dumps(self.b)),
                      RenderBinding(**{name: getattr(self.b, name) for name in FIELDS})):
            self.authority.authenticate(self.native, clone, self.mac)
        bad = [dict(tokens=self.tokens[:-1] + (7,)), dict(tokens=self.tokens[:-1]), dict(nonce='0' * 32),
               dict(turn=1, local_turn=0), dict(stable_end=41), dict(session='t'),
               dict(tokens=tuple(float(x) for x in self.tokens)), dict(tokens=list(self.tokens)),
               dict(rebuild=0), dict(epoch='f' * 64)]
        for change in bad:
            with self.subTest(change=sorted(change)):
                clone = dataclasses.replace(self.b, **change)
                with self.assertRaisesRegex(Abort, 'foreign native request or changed render'):
                    self.authority.authenticate(self.native, clone, self.mac)

    def test_a_field_replaced_on_the_minted_binding_is_refused_unless_equal(self):
        object.__setattr__(self.b, 'tokens', tuple(self.tokens))     # equal value, new object
        self.authority.authenticate(self.native, self.b, self.mac)
        object.__setattr__(self.b, 'tokens', self.tokens[:-1] + (7,))
        with self.assertRaisesRegex(Abort, 'changed render'):
            self.authority.authenticate(self.native, self.b, self.mac)
        object.__setattr__(self.b, 'tokens', self.tokens)
        object.__setattr__(self.b, 'stable_end', 41)
        with self.assertRaisesRegex(Abort, 'changed render'):
            self.authority.authenticate(self.native, self.b, self.mac)

    def test_the_native_request_must_be_the_assigned_one_with_the_same_tokens(self):
        with self.assertRaisesRegex(Abort, 'changed render'):
            self.authority.authenticate(NS(request_id='r0', prompt_token_ids=list(self.tokens)),
                                        self.b, self.mac)
        self.native.prompt_token_ids[5] += 1
        with self.assertRaisesRegex(Abort, 'changed render'):
            self.authority.authenticate(self.native, self.b, self.mac)

    def test_a_retired_binding_no_longer_authenticates(self):
        self.authority.retire(self.b)
        with self.assertRaisesRegex(Abort, 'MAC mismatch'):
            self.authority.authenticate(self.native, self.b, self.mac)
        with self.assertRaisesRegex(Abort, 'replay'):
            render(self.authority, 'r0', self.tokens, stable=40)

    def test_another_thread_is_refused(self):
        import threading
        failures = []
        thread = threading.Thread(target=lambda: failures.append(
            self.assertRaises(Abort, self.authority.authenticate, self.native, self.b, self.mac)))
        thread.start(); thread.join()
        self.assertEqual(len(failures), 1)


class SignatureCacheTests(unittest.TestCase):
    def test_the_kept_signature_is_the_one_inspect_builds(self):
        class Manager:
            def allocate(self, request, num_new_tokens, num_new_computed_tokens=0, *, skip_cache=False):
                return None
        first, second = Manager(), Manager()
        sig = engine_boundary._signature(first.allocate)
        self.assertEqual(sig, inspect.signature(first.allocate))
        self.assertIs(engine_boundary._signature(second.allocate), sig)       # one per function
        bound = sig.bind('req', 5); bound.apply_defaults()
        self.assertEqual(dict(bound.arguments), dict(request='req', num_new_tokens=5,
                                                     num_new_computed_tokens=0, skip_cache=False))
        # New defaults, a new code object: built again, as inspect would.
        Manager.allocate.__defaults__ = (3,)
        self.assertEqual(engine_boundary._signature(first.allocate).parameters['num_new_computed_tokens'].default, 3)
        Manager.allocate.__kwdefaults__ = dict(skip_cache=True)
        self.assertEqual(engine_boundary._signature(first.allocate).parameters['skip_cache'].default, True)
        # Wrapped or non-method callables are inspected per call.
        import functools

        def plain_function(request, n=1):
            return None
        self.assertEqual(engine_boundary._signature(plain_function), inspect.signature(plain_function))
        wrapped = functools.wraps(plain_function)(lambda *a, **k: None)
        self.assertEqual(engine_boundary._signature(wrapped), inspect.signature(wrapped))
        self.assertNotIn(plain_function, engine_boundary._SIGNATURES)


class AllocationRowCheckTests(unittest.TestCase):
    """The allocation_update row check and Allocation's validator accept and refuse
    exactly what the former generator expressions did."""

    ROWS = [
        [[[3, 1], [0, 0]], [[0, 0], [4, 2]]],
        [[[3, 1]], [[3, 2]]],                     # alias across groups
        [[[3, 0]]], [[[0, 1]]], [[[-1, 1]]], [[[3, -1]]], [[[True, 1]]], [[[3, 1.0]]],
        [[[3]]], [[[3, 1, 1]]], [[]], [], [[[5, 1], [6, 1]], [[7, 1]]], [[[None, 1]]], [[['3', 1]]],
    ]

    def outcome(self, update, rows):
        rec = dict(identity=NS(request='r'))
        payload = dict(rows=rows, start=0, count=1, binding='b')
        owner = RunnerAllocationOwner.__new__(RunnerAllocationOwner)
        qwen = NS(requests={'r': None})
        with patch('remote_ra_live_v2.chunk_protocol.worker_interval', lambda *a: None):
            try:
                result = update(owner, rec, payload)(None, qwen)
            except Exception as e:
                return (type(e).__name__, str(e))
        return (result, rec['native_rows'])

    def test_row_check(self):
        for rows in self.ROWS:
            with self.subTest(rows=rows):
                self.assertEqual(self.outcome(RunnerAllocationOwner._allocation_update, rows),
                                 self.outcome(former_allocation_update, rows))

    def test_allocation_validator(self):
        values = [(0, 1, 1, 'mamba', 0), (1, 0, 1, 'decode', 0), (-1, 1, 1, 'mamba', 0), (1, 1, -1, 'mamba', 0),
                  (True, 1, 1, 'mamba', 0), (1, 1.0, 1, 'mamba', 0), (1, 1, 1, 'other', 0), (1, 1, 1, 'mamba', -1),
                  (1, 1, 1, 'attention', 2), ('1', 1, 1, 'mamba', 0)]
        for args in values:
            with self.subTest(args=args):
                def build():
                    try:
                        return Allocation(*args).key
                    except Exception as e:
                        return (type(e).__name__, str(e))
                now = build()
                with patch.object(Allocation, '__post_init__', former_allocation_post_init):
                    self.assertEqual(build(), now)


if __name__ == '__main__':
    unittest.main()
