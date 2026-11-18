"""--async-scheduling (remote_ra_live_v2.async_sched; option C of docs/async-sched/DESIGN.md), CPU only.

- the pin: the config check and its refusals, the worker flag, the launcher (one more config key,
  --async-scheduling instead of --no-async-scheduling, the effective server config), the gates;
- the stop oracle: vLLM's check_stop on a shadow of the request and the output stop (MarkerStop)
  without touching its table, against "append, then check" on random outputs;
- the void writer on eager and persistent-buffer (FULL graph) metadata fakes;
- the held allocation updates (an excluded entry restores its record) and the placeholder-aware
  allocation and interval checks;
- the resolve when a request of step N is not in step N+1 (GPU job 3568311: vLLM's async scheduler
  skips a request whose token in flight is its max_tokens-th, and its runner takes it out of the batch
  and of ``prev_req_id_to_index``): its token is not read, nothing of it is voided or entered;
- an async MultiHarness: test_decouple_e2e's sessions served with vLLM's asynchronous order
  (schedule and execute N+1, then update N; pass drains before admissions; vLLM's max_tokens skip),
  the worker resolving each step's stops in before_forward. A turn stops on an EOS-style last output
  (scheduled once more, voided) or on max_tokens (not scheduled again), alternately. Per turn (lane,
  turn) the replies, selections, carries and Page vectors equal the synchronous fast-path-2 server's;
  the EOS-style stops are voided. Mutations: a missing void (the stop token reaches the carry) and a
  wrong block renaming are caught.
"""
from contextlib import ExitStack
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
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

from remote_ra_live_v2 import async_sched as A
from remote_ra_live_v2.contract import LiveV2Error
from remote_ra_live_v2.mswea_protocol import MarkerStop
# Imported here so the end-to-end runs' sys.modules patch (enter_case) keeps them.
from remote_ra_live_v2 import (generation_guard, hooks, loaded_sources, native_probe,  # noqa: F401
                               page_compaction, phase1_step, positions, vector_space, worker_composition,
                               worker_rows)
from remote_ra_growing_worker import tensors  # noqa: F401

try:
    import torch
except ImportError:
    torch = None

BASE = dict(decode_fastpath=True, decode_fastpath2=True)


# ---- the pin ------------------------------------------------------------------------------------------

class PinTests(unittest.TestCase):
    def test_config_check_and_refusals(self):
        self.assertIsNone(A.check_config({}))
        self.assertIsNone(A.check_config(None))
        self.assertIs(A.check_config(dict(BASE, async_scheduling=True)), True)
        # The Gemma 4 profile takes it since async-gemma (test_async_gemma), with the same refusals.
        self.assertIs(A.check_config(dict(BASE, async_scheduling=True, model_profile='gemma4-31b-it')), True)
        for bad, why in ((dict(BASE, async_scheduling=False), 'true or absent'),
                         (dict(BASE, async_scheduling=1), 'true or absent'),
                         (dict(decode_fastpath=True, async_scheduling=True), 'decode-fastpath2'),
                         (dict(decode_fastpath=True, async_scheduling=True, model_profile='gemma4-31b-it'),
                          'decode-fastpath2'),
                         (dict(BASE, async_scheduling=True, admission=dict(gdn_device_handoff=True)), 'refuses'),
                         (dict(BASE, async_scheduling=True, admission=dict(phase1_in_step=True)), 'refuses')):
            with self.subTest(why=why), self.assertRaisesRegex(LiveV2Error, why):
                A.check_config(bad)

    def test_worker_flag_against_the_pinned_config(self):
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0'):
            self.assertIsNone(A.worker_flag({}))
            self.assertIs(A.worker_flag({A.WORKER_KEY: True}), True)
            self.assertFalse(A.allowed())
            self.assertEqual(A.batches(), 1)
            with self.assertRaises(LiveV2Error):
                A.worker_flag({A.WORKER_KEY: 'yes'})
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'):
            with patch('remote_ra_http.config.load', lambda: dict(BASE, async_scheduling=True)):
                self.assertIs(A.worker_flag({A.WORKER_KEY: True}), True)
                self.assertTrue(A.allowed())
                self.assertEqual(A.batches(), 2)
                with self.assertRaisesRegex(LiveV2Error, 'differs from the pinned'):
                    A.worker_flag({})
            with patch('remote_ra_http.config.load', lambda: dict(BASE)):
                self.assertIsNone(A.worker_flag({}))
                with self.assertRaisesRegex(LiveV2Error, 'differs from the pinned'):
                    A.worker_flag({A.WORKER_KEY: True})

    def test_gates_require_the_pinned_scheduler(self):
        import bootstrap_contract
        import startup_runtime
        from policy_runtime import bootstrap as policy_bootstrap
        from remote_ra_live_v2.positions import runtime_mode

        def config(asynchronous):
            return NS(parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
                                         decode_context_parallel_size=1, prefill_context_parallel_size=1,
                                         distributed_executor_backend='uni'),
                      scheduler_config=NS(async_scheduling=asynchronous),
                      model_config=NS(runner_type='generate', is_encoder_decoder=False, enable_prompt_embeds=False),
                      speculative_config=None, max_concurrent_batches=2 if asynchronous else 1,
                      use_v2_model_runner=False)
        for pinned in (False, True):
            value = dict(BASE, async_scheduling=True) if pinned else dict(BASE)
            with self.subTest(pinned=pinned), patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'), \
                    patch('remote_ra_http.config.load', lambda value=value: value):
                bootstrap_contract.validate_topology(config(pinned))
                policy_bootstrap.validate_supported_config(config(pinned))
                with self.assertRaises(Exception):
                    bootstrap_contract.validate_topology(config(not pinned))
                with self.assertRaises(RuntimeError):
                    policy_bootstrap.validate_supported_config(config(not pinned))
                from collections import deque
                import queue
                import threading
                scheduler = NS(requests={}, waiting=[], running=[])
                core = NS(engines_running=False, scheduler=scheduler, input_queue=queue.Queue(),
                          output_queue=queue.Queue(), aborts_queue=queue.Queue(),
                          batch_queue=deque(maxlen=2) if pinned else None, batch_queue_size=2 if pinned else 1)
                ticket = startup_runtime.StartupTicket(core, None, threading.get_ident(), 'assembling') \
                    if hasattr(startup_runtime, 'StartupTicket') else None
                if ticket is not None:
                    core._remote_ra_startup_ticket = ticket
                    ticket.quiescent()
                    core.batch_queue = None if pinned else deque(maxlen=2)
                    with self.assertRaises(Exception):
                        ticket.quiescent()
        owner = NS(layout=NS(state=None))
        runner = NS(_remote_ra_live_v2_allocation_owner=owner, cache_config=NS(mamba_cache_mode='align'),
                    num_spec_tokens=0, speculative_config=None, use_async_scheduling=False, uses_xdrope_dim=0,
                    uses_mrope=False)
        runtime_mode(runner, 'forward')
        runner.use_async_scheduling = True
        with self.assertRaisesRegex(LiveV2Error, 'use_async_scheduling'):
            runtime_mode(runner, 'forward')
        owner.async_scheduling = True
        runtime_mode(runner, 'forward')
        runner.use_async_scheduling = False
        with self.assertRaisesRegex(LiveV2Error, 'use_async_scheduling'):
            runtime_mode(runner, 'forward')


class LauncherTests(unittest.TestCase):
    def test_switch_pins_one_key_and_swaps_the_scheduling_argument(self):
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
            kw = dict(kv_cache_gib=24, page_tokens=16, cuda_graphs='full', max_model_len=131072,
                      decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True)
            sync = serve.prepare(release, [dict(t) for t in tasks], 'full', root/'sync', 8231, **kw)
            on = serve.prepare(release, [dict(t) for t in tasks], 'full', root/'on', 8231, async_scheduling=True, **kw)
            refusals = (dict(kw, decode_fastpath2=False, decode_fastpath3=False, async_scheduling=True),
                        dict(kw, async_scheduling=1),
                        dict(kw, async_scheduling=True, gdn_device_handoff=True),
                        dict(kw, async_scheduling=True, phase1_in_step=True))
            for k, bad in enumerate(refusals):
                with self.subTest(refusal=k), self.assertRaises(ValueError):
                    args = dict(bad)
                    handoff, in_step = args.pop('gdn_device_handoff', False), args.pop('phase1_in_step', False)
                    serve.prepare(release, [dict(t) for t in tasks], 'goal_query' if (handoff or in_step) else 'full',
                                  root/('bad%d' % k), 8231, None, 'mswea-xml',
                                  **dict(args, gdn_device_handoff=handoff, phase1_in_step=in_step))
            for name, config, expected in (('on', on[1], True), ('sync', sync[1], None)):
                path, raw = root/(name + '.json'), serve.wire(config)
                path.write_bytes(raw)
                with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                                REMOTE_RA_HTTP_CONFIG_SHA256=serve.digest(raw)):
                    self.assertEqual(http_config.load().get('async_scheduling'), expected)
                    self.assertEqual(A.loaded(), expected)
                    args = NS(**config['effective_server_config'])
                    self.assertEqual(http_config.effective_server_config(args), config['effective_server_config'])
                    flipped = NS(**dict(config['effective_server_config'], async_scheduling=not expected))
                    with self.assertRaises(RuntimeError):
                        http_config.effective_server_config(flipped)
        strip = lambda value, name: json.loads(json.dumps(value).replace(str(root/name).replace('\\', '\\\\'), '<out>'))
        on_argv, sync_argv = strip(on[2], 'on'), strip(sync[2], 'sync')
        self.assertIn('--no-async-scheduling', sync_argv)
        self.assertNotIn('--async-scheduling', sync_argv)
        self.assertEqual([a if a != '--async-scheduling' else '--no-async-scheduling' for a in on_argv], sync_argv)
        on_config = strip(on[1], 'on')
        self.assertIs(on_config.pop('async_scheduling'), True)
        self.assertIs(on_config['effective_server_config'].pop('async_scheduling'), True)
        on_config.pop('server_config_sha256')
        sync_config = strip(sync[1], 'sync')
        sync_config.pop('server_config_sha256')
        self.assertEqual(on_config, sync_config)
        self.assertNotIn('async_scheduling', sync[1])


# ---- the stop oracle ----------------------------------------------------------------------------------

def check_stop(request, max_model_len):
    """vLLM 0.24.0's check_stop (vllm/v1/core/sched/utils.py), for the CPU tests."""
    if request.num_tokens >= max_model_len or request.num_output_tokens >= request.max_tokens:
        request.status = 'FINISHED_LENGTH_CAPPED'
        return True
    params = request.sampling_params
    if request.num_output_tokens < params.min_tokens:
        return False
    last = request.output_token_ids[-1]
    if not params.ignore_eos and last == request.eos_token_id:
        request.status = 'FINISHED_STOPPED'
        return True
    if last in (params.stop_token_ids or ()):
        request.status = 'FINISHED_STOPPED'
        request.stop_reason = last
        return True
    return False


class Request:
    def __init__(self, rid, prompt, outputs, max_tokens, eos, stops=(), min_tokens=0, ignore_eos=False):
        self.request_id, self.prompt, self.output_token_ids = rid, list(prompt), list(outputs)
        self.max_tokens, self.eos_token_id, self.status, self.stop_reason = max_tokens, eos, 'RUNNING', None
        self.sampling_params = NS(min_tokens=min_tokens, ignore_eos=ignore_eos, stop_token_ids=list(stops))

    @property
    def num_tokens(self):
        return len(self.prompt) + len(self.output_token_ids)

    @property
    def num_output_tokens(self):
        return len(self.output_token_ids)


class OracleTests(unittest.TestCase):
    def test_shadow_check_stop_equals_append_then_check_and_mutates_nothing(self):
        rng = random.Random(7)
        stops = 0
        for k in range(3000):
            outputs = [rng.randrange(20) for _ in range(rng.randrange(0, 12))]
            request = Request('r%d' % k, [1] * rng.randrange(1, 30), outputs, rng.randrange(1, 16), rng.randrange(20),
                              stops=rng.sample(range(20), rng.randrange(0, 3)), min_tokens=rng.randrange(0, 4),
                              ignore_eos=rng.random() < .2)
            token = rng.randrange(20)
            before = (list(request.output_token_ids), request.status, request.stop_reason)
            scheduler = NS(requests={request.request_id: request}, max_model_len=rng.randrange(8, 48))
            link = A.EngineLink(scheduler, None, check_stop=check_stop)
            predicted = link.stops(request.request_id, token)
            self.assertEqual((list(request.output_token_ids), request.status, request.stop_reason), before)
            request.output_token_ids.append(token)
            self.assertEqual(predicted, check_stop(request, scheduler.max_model_len))
            stops += predicted
        self.assertGreater(stops, 300)

    def test_marker_stop_without_touching_its_table(self):
        vocab = ['a', 'b', '<', '/', 'mswea_bash_command', '>', 'x>', '\n', ' ', '</think>']
        decode = lambda ids: ''.join(vocab[i] for i in ids)
        rng = random.Random(3)
        statuses = patch.dict(sys.modules, {'vllm.v1.request': NS(RequestStatus=NS(FINISHED_STOPPED='stopped'))})
        statuses.start()
        self.addCleanup(statuses.stop)
        for incremental, thinking in ((False, False), (True, False), (True, True)):
            policy = MarkerStop(decode, incremental=incremental, token_cache=incremental, thinking=thinking)
            reference = MarkerStop(decode, incremental=incremental, token_cache=incremental, thinking=thinking)
            stops = 0
            for k in range(600):
                outputs = []
                request = Request('m%d' % k, [1], outputs, 99, -1)
                stream = [rng.randrange(len(vocab)) for _ in range(rng.randrange(1, 30))]
                if rng.random() < .5:            # the marker (in pieces, or ending in 'x>'), maybe after </think>
                    at = rng.randrange(len(stream) + 1)
                    piece = [2, 3, 4, 5] if rng.random() < .7 else [2, 3, 4, 6]
                    stream[at:at] = ([9] if rng.random() < .5 else []) + piece
                for token in stream:
                    table = dict(policy._checked)
                    predicted = A.marker_stops(policy, request, token)
                    self.assertEqual(policy._checked, table)        # nothing written
                    request.output_token_ids.append(token)
                    actual = policy(request)
                    self.assertEqual(predicted, actual)
                    self.assertEqual(actual, reference(Request('r', [1], list(request.output_token_ids), 99, -1)))
                    if actual:
                        stops += 1
                        break
            with self.subTest(incremental=incremental, thinking=thinking):
                self.assertGreater(stops, 50)

    def test_link_check_requires_every_predicted_stop(self):
        link = A.EngineLink(NS(requests={}), None, check_stop=check_stop)
        link.check(['a'])                               # nothing predicted: nothing to check
        link.predicted = frozenset({'a', 'b'})
        link.check(['a', 'b', 'c'])
        self.assertIsNone(link.predicted)
        link.predicted = frozenset({'a', 'b'})
        with self.assertRaisesRegex(LiveV2Error, 'did not stop'):
            link.check(['a'])


# ---- the void writer ----------------------------------------------------------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required')
class VoidTests(unittest.TestCase):
    def metadata(self, persistent):
        rows, tokens = 4, 9                      # rows 0 (prefill 6), 1, 2, 3 (decode)
        if persistent:
            # FULL graphs: the builders' tensors are views of padded persistent buffers.
            states = [torch.arange(1, 17) * 10 for _ in range(3)]
            gdn = [NS(non_spec_state_indices_tensor=s[:rows]) for s in states]
            buffer = torch.arange(100, 132)
            fa = NS(slot_mapping=buffer[:tokens])
            group = fa.slot_mapping
        else:
            states = None
            gdn = [NS(non_spec_state_indices_tensor=torch.arange(1, rows + 1) * 10) for _ in range(3)]
            fa = NS(slot_mapping=torch.arange(100, 100 + tokens))
            group = fa.slot_mapping
        attn = {'fa.0': fa, 'fa.1': fa}
        for g, md in enumerate(gdn):
            attn['gdn.%d.a' % g] = attn['gdn.%d.b' % g] = md
        layers = {'fa.0': fa.slot_mapping, 'fa.1': fa.slot_mapping.clone()}
        mamba = torch.arange(500, 500 + tokens)
        groups = {0: group, 1: mamba, 2: mamba.clone(), 3: mamba.clone()}
        runner = NS(input_batch=NS(req_ids=['p', 'a', 'b', 'c']), _remote_ra_step_metadata=(attn, layers, groups))
        so = NS(num_scheduled_tokens=dict(p=6, a=1, b=1, c=1))
        return runner, so, gdn, fa, layers, groups, states

    def test_void_writes_the_rows_and_tokens_of_stopped_requests_only(self):
        for persistent in (False, True):
            with self.subTest(persistent=persistent):
                runner, so, gdn, fa, layers, groups, states = self.metadata(persistent)
                mamba = [groups[g].clone() for g in (1, 2, 3)]
                written = A.void_rows(runner, so, frozenset({'a', 'c'}), (0,))
                self.assertEqual(written, (3, 2))   # 3 GDN groups; the FA metadata (= group 0) and one clone
                for md in gdn:
                    self.assertEqual(md.non_spec_state_indices_tensor.tolist(), [10, 0, 30, 0])
                expect = [100, 101, 102, 103, 104, 105, -1, 107, -1]
                self.assertEqual(fa.slot_mapping.tolist(), expect)
                self.assertEqual(layers['fa.1'].tolist(), expect)
                self.assertEqual(groups[0].tolist(), expect)
                for g, before in zip((1, 2, 3), mamba):
                    self.assertTrue(torch.equal(groups[g], before))      # Mamba groups' mappings untouched
                if persistent:
                    for s in states:
                        self.assertEqual(s[:4].tolist(), [10, 0, 30, 0])
                        self.assertEqual(s[4:].tolist(), (torch.arange(5, 17) * 10).tolist())

    def test_void_refusals(self):
        runner, so, *_ = self.metadata(False)
        with self.assertRaisesRegex(LiveV2Error, 'more than one token'):
            A.void_rows(runner, so, frozenset({'p'}), (0,))
        with self.assertRaisesRegex(LiveV2Error, 'missing from the batch'):
            A.void_rows(runner, so, frozenset({'a', 'zz'}), (0,))
        bare = NS(input_batch=runner.input_batch)
        with self.assertRaisesRegex(LiveV2Error, 'not stashed'):
            A.void_rows(bare, so, frozenset({'a'}), (0,))
        attn = {'fa.0': NS(slot_mapping=torch.arange(9))}
        runner._remote_ra_step_metadata = (attn, {}, {})
        with self.assertRaisesRegex(LiveV2Error, 'no GDN state index'):
            A.void_rows(runner, so, frozenset({'a'}), (0,))

    def test_sampled_tokens_reads_once_and_skips_discarded_rows(self):
        runner = NS(input_batch=NS(prev_sampled_token_ids=torch.tensor([[5], [6], [7]]),
                                   prev_req_id_to_index={'a': 0, 'c': 2}))
        self.assertEqual(A.sampled_tokens(runner, ('a', 'b', 'c')), {'a': 5, 'c': 7})
        self.assertEqual(A.sampled_tokens(runner, ()), {})
        runner.input_batch.prev_sampled_token_ids = None
        with self.assertRaisesRegex(LiveV2Error, 'unavailable'):
            A.sampled_tokens(runner, ('a',))


# ---- the resolve: a request of step N that step N+1 does not schedule ----------------------------------

@unittest.skipUnless(torch is not None, 'CPU torch required')
class ResolveTests(unittest.TestCase):
    """async_sched.resolve on fakes, as before_forward(N+1) calls it after vLLM's _update_states(N+1)."""

    def setUp(self):
        self.addCleanup(A.detach_engine)

    def resolve(self, scheduled, b, tokens, *, a_eos=-1, max_model_len=1 << 20, popped=None):
        """Step N ran and sampled A (row 0) and B (row 1, the request ``b``); N+1 schedules ``scheduled``. The
        runner's _update_states(N+1) took every other request out of the persistent batch, and its
        InputBatch.remove_request popped it from prev_req_id_to_index (gpu_input_batch.py:556-557; ``popped``
        overrides). Returns (voided, the link, the bridge, the worker calls in order, the void_rows calls)."""
        calls, voids = [], []
        requests = dict(A=Request('A', [1] * 10, [5] * 3, 99, a_eos), B=b)
        identities = {rid: NS(request=rid) for rid in requests}
        qwen = NS(requests={rid: NS(prompt_complete=True, cursor=r.num_tokens, terminal=False)
                            for rid, r in requests.items()},
                  leave_issued=lambda *i: calls.append(('leave_issued', [x.request for x in i])),
                  enter_execute=lambda *i: calls.append(('enter_execute', [x.request for x in i])))
        bridge = NS(issued=(identities['A'], identities['B']), entered=None, qwen=qwen,
                    finish_step=lambda: calls.append(('finish_step',)),
                    owner=NS(records={rid: dict(identity=i) for rid, i in identities.items()},
                             attention_groups=(0,), async_sliding=None))
        popped = [rid for rid in requests if rid not in scheduled] if popped is None else popped
        index = {rid: row for rid, row in (('A', 0), ('B', 1)) if rid not in popped}
        runner = NS(input_batch=NS(req_ids=[rid for rid in requests if rid in scheduled], prev_req_id_to_index=index,
                                   prev_sampled_token_ids=torch.tensor([[tokens['A']], [tokens['B']]])))
        boundary = NS(flush_updates=lambda exclude: calls.append(('flush', sorted(exclude))))
        scheduler = NS(requests=requests, max_model_len=max_model_len, _remote_ra_output_stop=None,
                       prev_step_scheduled_req_ids=set())
        link = A._LINK = A.EngineLink(scheduler, boundary, check_stop=check_stop, core=NS(batch_queue=()))
        void = lambda runner, so, voided, groups, sliding=None: voids.append(sorted(voided)) or (1, 1)
        with patch.object(A, 'void_rows', void):
            voided = A.resolve(runner, bridge, NS(num_scheduled_tokens=dict(scheduled)))
        return voided, link, bridge, calls, voids

    def test_a_request_the_next_step_does_not_schedule_is_left_to_its_update(self):
        """GPU job 3568311 (smqt: Qwen3.6-27B, thinking and async on, MemoryAgentBench kv-session; all three
        servers died at the first length stop next to another request). vLLM's async scheduler skips B in N+1
        when its token of N reaches max_tokens or max_model_len (scheduler.py:490-504; 527-534 and 562-578); its
        runner then takes B out of the batch (gpu_model_runner.py:1197-1217) and so out of prev_req_id_to_index
        (gpu_input_batch.py:556-557). A request it
        leaves out for another reason (the token budget, a preemption) leaves the batch the same way. The
        resolve required B's token and raised 'a request sampled without its token'. Now B's token is not
        read: update_from_output(N) decides its stop, as on the synchronous server; B has no row of N+1 to
        void and is not entered; N's leases still close for both requests; the held updates are sent whole."""
        for why, b, max_model_len, stops in (
                ('max_tokens', Request('B', [1] * 10, [5] * 7, 8, -1), 1 << 20, True),
                ('max_model_len', Request('B', [1] * 10, [5] * 7, 99, -1), 18, True),
                ('not scheduled, no stop', Request('B', [1] * 10, [5] * 7, 99, -1), 1 << 20, False)):
            with self.subTest(why=why):
                self.assertEqual(check_stop(A._Shadow(b, 6), max_model_len), stops)
                voided, link, bridge, calls, voids = self.resolve({'A': 1}, b, dict(A=6, B=6),
                                                                  max_model_len=max_model_len)
                self.assertEqual((voided, link.predicted, voids), (frozenset(), frozenset(), []))
                self.assertEqual(calls, [('finish_step',), ('leave_issued', ['A', 'B']), ('flush', []),
                                         ('enter_execute', ['A'])])
                self.assertEqual(bridge.entered, (bridge.owner.records['A']['identity'],))
                link.check(['B'] if stops else [])        # update_from_output(N): nothing predicted is missed

    def test_a_stop_of_a_request_in_the_next_step_is_still_voided(self):
        """A's token of N is its EOS and A is in N+1: its stop is predicted, its row voided, its update held
        back, and it is not entered; B (skipped at max_tokens) is not read."""
        b = Request('B', [1] * 10, [5] * 7, 8, -1)
        voided, link, bridge, calls, voids = self.resolve({'A': 1}, b, dict(A=11, B=6), a_eos=11)
        self.assertEqual((voided, link.predicted, voids), (frozenset({'A'}), frozenset({'A'}), [['A']]))
        self.assertEqual(calls, [('finish_step',), ('leave_issued', ['A', 'B']), ('flush', ['A'])])
        self.assertEqual(bridge.entered, ())
        link.check(['A', 'B'])
        with self.assertRaisesRegex(LiveV2Error, 'did not stop'):
            link.predicted = frozenset({'A'})
            link.check(['B'])

    def test_a_request_in_the_next_step_without_its_token_still_fails_closed(self):
        """The check keeps its force for the requests N+1 does schedule: one missing from
        prev_req_id_to_index (a token never sampled) still fails closed."""
        b = Request('B', [1] * 10, [5] * 7, 99, -1)
        with self.assertRaisesRegex(LiveV2Error, 'sampled without its token'):
            self.resolve({'A': 1, 'B': 1}, b, dict(A=6, B=6), popped=('B',))


# ---- the held updates and the placeholders -------------------------------------------------------------

class HoldTests(unittest.TestCase):
    def test_flush_excludes_and_restores_the_records(self):
        from remote_ra_live_v2.engine_boundary import EngineRpcBoundary
        sent = []
        proxy = NS(call=lambda op, payload: sent.append((op, payload)) or dict(
            operation=op, receipts=[dict(operation='allocation_update', binding=e['binding'])
                                    for e in payload['entries']]))
        boundary = EngineRpcBoundary.__new__(EngineRpcBoundary)     # no pools: flush only
        boundary.proxy, boundary.adapter, boundary.prepared, boundary.receipts = proxy, NS(requests={}), {}, {}
        boundary.batched, boundary.async_hold = {}, True
        records = {}
        for rid in ('a', 'b'):
            prior = dict(logical_cursor=10)
            record = records[rid] = dict(native_allocation=dict(logical_cursor=11), native_tables={0: (1, 2, 3)},
                                         held_prior=(prior, {0: (1, 2)}), accepted_rows=None)
            boundary.batched[rid] = (record, NS(request=rid, identity='id-' + rid), 10, 1, [[1]],
                                     dict(start=10, end=11, phase='decode'))
        boundary.flush_updates(exclude=frozenset({'b'}))
        self.assertEqual([e['request'] for e in sent[0][1]['entries']], ['a'])
        self.assertEqual(records['a']['scheduled_cursor'], 11)
        self.assertEqual(records['b']['native_allocation'], dict(logical_cursor=10))
        self.assertEqual(records['b']['native_tables'], {0: (1, 2)})
        self.assertNotIn('scheduled_cursor', records['b'])
        self.assertNotIn('held_prior', records['b'])
        boundary.batched = {'b': (records['a'], NS(request='b', identity='id-b'), 10, 1, [[1]], {})}
        records['a']['held_prior'] = (None, {})
        self.assertIsNone(boundary.flush_updates(exclude=frozenset({'b'})))   # nothing left to send
        self.assertEqual(len(sent), 1)

    def test_placeholder_aware_checks(self):
        from remote_ra_live_v2.allocation_guard import validate_native_allocation
        from remote_ra_live_v2.chunk_protocol import validate_interval
        binding = NS(request='r', tokens=[0] * 8, identity='id', generation=1)
        plan = dict(phase2_start=0)
        values = dict(num_new_tokens=1, delay_cache_blocks=False, full_sequence_must_fit=False, has_scheduled_reqs=False,
                      skip_cache=False, new_computed_blocks=None)

        def request(computed, outputs, held, status='RUNNING'):
            return NS(request_id='r', num_computed_tokens=computed, num_tokens=8 + outputs, num_output_placeholders=held,
                      num_preemptions=0, spec_token_ids=[], status=NS(name=status))
        # Synchronous: the decode's token is known; no placeholder is accepted.
        validate_native_allocation(values, request(8, 1, 0), binding, plan, 0)
        validate_interval(dict(scheduled_cursor=8), request(8, 1, 0), binding, plan, 1)
        with self.assertRaises(LiveV2Error):
            validate_native_allocation(values, request(8, 0, 1), binding, plan, 0)
        with self.assertRaises(LiveV2Error):
            validate_interval(dict(scheduled_cursor=8), request(8, 0, 1), binding, plan, 1)
        # Asynchronous: one placeholder counts as the fresh token.
        validate_native_allocation(values, request(8, 0, 1), binding, plan, 0, placeholders=True)
        validate_interval(dict(scheduled_cursor=8), request(8, 0, 1), binding, plan, 1, placeholders=True)
        validate_native_allocation(values, request(9, 1, 1), binding, plan, 0, placeholders=True)
        for bad in (request(8, 0, 2), request(8, 0, 1, status='WAITING'), request(9, 0, 1)):
            with self.assertRaises(LiveV2Error):
                validate_native_allocation(values, bad, binding, plan, 0, placeholders=True)
        with self.assertRaises(LiveV2Error):
            validate_interval(dict(scheduled_cursor=8), request(8, 1, 1), binding, plan, 1, placeholders=True)


# ---- the async MultiHarness ---------------------------------------------------------------------------

DUMMY = 0           # the token a stopped request's extra row samples (never used)
# A turn whose lane index plus turn index is even stops on an EOS-style last output (max_tokens beyond it): vLLM
# schedules the step after the stop, whose row the resolve voids. An odd one stops on max_tokens: vLLM schedules
# no step after it (with no other request running, an empty step).
EOS_SLACK = 16      # an EOS turn's max_tokens beyond its last scripted output


def scripted_limits(info):
    """A turn's scripted stop on the async harnesses (``scripted_stop``): its last output, an EOS when its lane
    index plus turn index is even (``eos_at``; max_tokens EOS_SLACK beyond it), else its max_tokens-th token."""
    request, n, lane = info['request'], len(info['outputs']), info['lane']
    request.eos_at = n
    request.max_tokens = n + EOS_SLACK if (lane.index + lane.turn) % 2 == 0 else n
    return info


def length_skip(request):
    """vLLM 0.24.0's async scheduler (scheduler.py:490-504): a running request whose token in flight is its
    max_tokens-th is not scheduled; update_from_output stops it, and the runner takes it out of the persistent
    batch and of prev_req_id_to_index (gpu_model_runner.py:1197-1217, gpu_input_batch.py:556-557)."""
    held = request.num_output_placeholders
    return held > 0 and (request.num_computed_tokens + 2 - held >=
                         len(request.prompt_token_ids) + request.max_tokens)


def cached_token_ids(link, counts, news):
    """vLLM's _make_cached_request_data and schedule()'s bookkeeping (scheduler.py) on the fakes: a scheduled
    cached request that was not scheduled in the previous step gets its token ids (``all_token_ids``), and the
    step's requests become the previous step's (``prev_step_scheduled_req_ids``)."""
    scheduler = link.scheduler
    previous, new = scheduler.prev_step_scheduled_req_ids, {n.req_id for n in news}
    ids = {rid: list(scheduler.requests[rid].prompt_token_ids) + list(scheduler.requests[rid].output_token_ids)
           for rid in counts if rid not in new and rid not in previous}
    previous.clear()
    previous.update(counts)
    return ids


def require_token_ids(runner, so):
    """vLLM's GPU runner under async scheduling (_update_states): a scheduled cached request outside the
    persistent batch that has output tokens is added back from ``all_token_ids`` alone (the runner's own output
    ids are placeholders), else KeyError (GPU job 3519531: a phase-1 pass had taken it out). The ids must be its
    tokens, none in flight (test_async_probe_batch runs vLLM's own code)."""
    held = {rid for rid in runner.input_batch.req_ids if rid is not None}
    new = {n.req_id for n in so.scheduled_new_reqs}
    for rid in so.num_scheduled_tokens:
        request = runner.requests.get(rid)
        if rid in held or rid in new or request is None or not request.output_token_ids:
            continue
        ids = so.all_token_ids[rid]
        assert ids == list(request.prompt_token_ids) + list(request.output_token_ids), ('stale token ids', rid)


class InFlight:
    """attach_engine's EngineCore for the link on a harness: its batch queue is the harness's step in flight."""

    def __init__(self, h):
        self.h = h

    @property
    def batch_queue(self):
        return () if self.h.inflight is None else (self.h.inflight,)


def async_runner(base, W, D):
    """test_decouple_e2e's Runner under vLLM's async order: the step's attention metadata is stashed
    for the void (one GDN metadata per Mamba group, shared by two layers; the FA metadata, its
    per-layer and per-group slot mappings; Mamba groups' slot mappings that must stay untouched),
    the forward honours it (a row at NULL_BLOCK_ID with slot -1 is skipped by the kernels), and
    the sampled ids stay on the device (prev_sampled_token_ids) until the next step."""

    class AsyncRunner(base):
        def __init__(self, store, case):
            super().__init__(store, case)
            self.use_async_scheduling = True
            self.input_batch.prev_sampled_token_ids = None
            self.input_batch.prev_req_id_to_index = None
            self.sampled, self.voided_rows, self.extra_rows = [], [], []

        def execute_model(self, so):
            from remote_ra_live_v2.native_probe import active_probe
            if active_probe(self) is not None:
                return base.execute_model(self, so)
            assert self.execute_model_state is None
            require_token_ids(self, so)
            self._update_states(so)
            batch = list(self.input_batch.req_ids)
            assert batch and len(batch) == len(so.num_scheduled_tokens) and set(batch) == set(so.num_scheduled_tokens)
            from remote_ra_live_v2.worker_rows import install_before_prepare
            install_before_prepare(self, so)
            M, P, spans = self.M, W.P, []
            for rid in batch:
                req = self.requests[rid]
                start, count = req.num_computed_tokens, so.num_scheduled_tokens[rid]
                curr = -(-(start + count) // M) - 1
                for g in W.MAMBA:
                    row = req.block_ids[g]
                    if start > 0 and (start - 1) // M != curr:
                        self.store.mamba[(g, row[curr])] = self.store.mamba.get((g, row[(start - 1) // M]), 'garbage')
                self.mamba_state_idx[rid] = curr
                spans.append((rid, req, start, count, curr))
            self.steps.append(tuple((rid, start, count) for rid, _, start, count, _ in spans))
            gdn = {g: NS(non_spec_state_indices_tensor=torch.tensor([req.block_ids[g][curr]
                                                                     for _, req, _, _, curr in spans]))
                   for g in W.MAMBA}
            slots = []
            for rid, req, start, count, _ in spans:
                plan = self._selective_load_plans.get(rid)
                dropped = plan.dropped_full_attention_tokens if plan else 0
                row = req.block_ids[W.FA]
                slots.extend(row[(x - dropped) // P] * P + (x - dropped) % P for x in range(start, start + count))
            fa = NS(slot_mapping=torch.tensor(slots))
            attn = {'fa.0': fa, 'fa.1': fa}
            for g, md in gdn.items():
                attn['gdn.%d.a' % g] = attn['gdn.%d.b' % g] = md
            layers = {'fa.0': fa.slot_mapping, 'fa.1': fa.slot_mapping.clone()}
            groups = {W.FA: fa.slot_mapping, **{g: torch.tensor(slots) for g in W.MAMBA}}
            self._remote_ra_step_metadata = (attn, layers, groups)
            untouched = {g: groups[g].clone() for g in W.MAMBA}
            from remote_ra_live_v2 import hooks
            flat = torch.cat([torch.arange(start, start + count) for _, _, start, count, _ in spans])
            hooks.before_forward(self, so, flat)
            assert all(torch.equal(groups[g], untouched[g]) for g in W.MAMBA), 'void wrote a Mamba slot mapping'
            hidden, offset, void = [], 0, []
            for k, (rid, req, start, count, curr) in enumerate(spans):
                nulls = [int(gdn[g].non_spec_state_indices_tensor[k]) == A.NULL_BLOCK_ID for g in W.MAMBA]
                skipped = [int(m[offset + i]) == A.NULL_SLOT for m in (fa.slot_mapping, layers['fa.1'])
                           for i in range(count)]
                assert all(nulls) == any(nulls) and (all(skipped) if all(nulls) else not any(skipped)), \
                    'partial void of %s' % rid
                void.append(all(nulls))
                if all(nulls):
                    self.voided_rows.append((rid, start))
                    hidden.append(torch.zeros(count, 3))
                else:
                    if start >= len(req.prompt_token_ids) + len(req.output_token_ids):
                        self.extra_rows.append((rid, start))   # a stop the device computed past
                    hidden.append(self.forward(rid, req, start, count, curr))
                offset += count
            result = hooks.after_forward(self, so, torch.cat(hidden))
            ids, index, sampled = [], {}, {}
            for k, (rid, req, start, count, _) in enumerate(spans):
                req.num_computed_tokens = start + count
                token = DUMMY
                if start + count >= len(req.prompt_token_ids):
                    index[rid] = k
                    if not void[k] and start + count == len(req.prompt_token_ids) + len(req.output_token_ids):
                        token = next(self.samples[rid], None)
                        if token is None:
                            token = DUMMY
                        else:
                            req.output_token_ids.append(token)
                            sampled[rid] = token
                ids.append([token])
            self.input_batch.prev_sampled_token_ids = torch.tensor(ids)
            self.input_batch.prev_req_id_to_index = index
            self.sampled.append(sampled)
            return result

        def empty_step(self, so):
            """vLLM's execute_model with nothing scheduled (every running request's token in flight is its
            last, ``length_skip``): the persistent batch is updated (``_update_states``), then it returns
            before any dispatch, forward, hook or sampling; the guard passes it through (no owned request)."""
            assert self.execute_model_state is None and not so.num_scheduled_tokens
            require_token_ids(self, so)
            self._update_states(so)
            self.sampled.append({})
    return AsyncRunner


def async_harness(D):
    """test_decouple_e2e's MultiHarness in vLLM's async order (EngineCore.step_with_batch_queue at
    two batches): schedule N+1 (computed counts advance, a sampling step holds a placeholder), execute
    it (the worker resolves N in before_forward), then update N (tokens appended, stops finished, the
    link's predictions checked). Admissions are pass boundaries: the batch in flight is drained first.
    A request stops on its last scripted output, which the host learns only from the update
    (``scripted_limits``): alternately an EOS, scheduled once more and voided, and its max_tokens-th
    token, which vLLM's schedule does not schedule past (``length_skip``; GPU job 3568311).
    ``length_skips``: per schedule that skipped a request, (the skipped requests, the requests it scheduled)."""

    class AsyncMultiHarness(D.MultiHarness):
        def __init__(self, policy, output, case, tasks=None):
            super().__init__(policy, output, case, tasks)
            self.inflight, self.async_drains, self.finished_at, self.length_skips = None, 0, {}, []

        def admit(self, lane):
            info = super().admit(lane)
            return info if info is None else scripted_limits(info)

        def schedule(self):
            from vllm.v1.core.sched.output import NewRequestData
            import test_stageb_e2e as S
            M, counts, news, skipped = self.case.M, {}, [], []
            for info in self.running:
                request, rid = info['request'], info['rid']
                if length_skip(request):
                    skipped.append(rid)
                    continue
                N, start = len(request.prompt_token_ids), request.num_computed_tokens
                waiting = request.status.name == 'WAITING'
                count = min(N, (start // M + 1) * M) - start if start < N else 1
                result = self.adapter.allocate_slots(self.kv, request, count, num_new_computed_tokens=0,
                    new_computed_blocks=None, num_lookahead_tokens=0, num_external_computed_tokens=0,
                    delay_cache_blocks=False, num_encoder_tokens=0, full_sequence_must_fit=waiting,
                    reserved_blocks=0, has_scheduled_reqs=waiting and bool(counts))
                assert result is not None, 'running request %s found no free block (preemption)' % rid
                rows = tuple([b.block_id for b in m.req_to_blocks[rid]]
                             for m in self.kv.coordinator.single_type_managers)
                if info['first']:
                    news.append(NewRequestData(rid, list(request.prompt_token_ids), [], S.sampling(), None,
                                               rows, start, None))
                    info['first'] = False
                else:
                    self.runner.set_rows(rid, rows)
                self.slots[rid].append(rows[1][-1])
                counts[rid] = count
            if skipped:
                self.length_skips.append((tuple(skipped), tuple(counts)))
            for info in self.running:          # vLLM's _update_after_schedule
                request = info['request']
                if info['rid'] not in counts:
                    continue
                request.status.name = 'RUNNING'
                if request.num_computed_tokens + counts[info['rid']] >= len(request.prompt_token_ids):
                    request.num_output_placeholders += 1
                request.num_computed_tokens += counts[info['rid']]
            return counts, news

        def batch_step(self):
            counts, news = self.schedule()
            finished, self.finished = self.finished, set()
            # Nothing scheduled (length_skip): vLLM's execute_model returns before the forward (empty_step).
            (self.runner.execute_model if counts else self.runner.empty_step)(NS(num_scheduled_tokens=dict(counts),
                total_num_scheduled_tokens=sum(counts.values()), scheduled_new_reqs=news,
                all_token_ids=cached_token_ids(A.link(), counts, news),
                finished_req_ids=finished, new_block_ids_to_zero=self.kv.take_new_block_ids(),
                kv_connector_metadata=NS(load_jobs={}, store_jobs={}, jobs_to_flush=set())))
            self.widths.append(len(counts))
            previous, self.inflight = self.inflight, (counts, self.runner.sampled[-1])
            if previous is not None:
                self.update(previous)

        def update(self, step):
            counts, sampled = step
            for info in list(self.running):
                rid, request = info['rid'], info['request']
                if rid not in counts:
                    continue
                token = sampled.get(rid)
                if token is not None:
                    request.output_token_ids.append(token)
                    request.num_output_placeholders -= 1
                info['steps'] += 1
                if len(request.output_token_ids) == len(info['outputs']):
                    self.finished_at[rid] = len(self.widths)
                    self.finish(info)
            running = {info['rid'] for info in self.running}
            A.link().check([rid for rid in counts if rid not in running])

        def drain(self):
            if self.inflight is not None:
                step, self.inflight = self.inflight, None
                self.async_drains += 1
                self.update(step)

        def serve(self, check=None):
            k = refused = 0
            while any(lane.script for lane in self.lanes) or self.running or self.inflight:
                busy = {info['lane'].index for info in self.running}
                ready = [lane for lane in self.lanes if lane.script and lane.index not in busy]
                if ready:
                    self.drain()                  # a pass boundary: the batch in flight first
                    busy = {info['lane'].index for info in self.running}
                    ready = [lane for lane in self.lanes if lane.script and lane.index not in busy]
                if ready:
                    admitted = self.admit(ready[k % len(ready)]) is not None
                    k += 1
                    refused = 0 if admitted or self.running else refused + 1
                    if refused >= len(ready):
                        raise D.CapacityStall('every lane with work was refused')
                if self.running:
                    before = len(self.done)
                    self.batch_step()
                    for info in self.done[before:]:
                        if check is not None:
                            check(self, info)
                else:
                    self.drain()
            self.drain()
            self.flush()
    return AsyncMultiHarness


def attach_async(h, check_stop_fn):
    """attach_loaded_worker's and compose_engine's --async-scheduling attachments on a harness (after
    fast paths 1 and 2): the updates are held (no flush where schedule returns), the link reads the
    harness's requests."""
    from remote_ra_live_v2 import decode_fastpath as F
    from remote_ra_live_v2 import decode_fastpath2 as F2
    F.attach_worker(h.owner, h.bridge, h.qwen)
    F2.attach_worker(h.owner, h.bridge, h.qwen, h.io)
    h.boundary.batched, h.boundary.async_hold = {}, True
    h.proxy.before_call = h.boundary.flush_updates
    A.attach_worker(h.owner, h.bridge, h.qwen, h.runner)
    scheduler = NS(requests=h.requests, max_model_len=1 << 20, _remote_ra_output_stop=None,
                   prev_step_scheduled_req_ids=set())
    A.detach_engine()
    A._LINK = A.EngineLink(scheduler, h.boundary, check_stop=check_stop_fn, core=InFlight(h))
    return A._LINK


def scripted_stop(request, max_model_len):
    """The harness's stop (``scripted_limits``): the last scripted output (``eos_at``) or the max_tokens cap."""
    count = len(request.output_token_ids)
    return count >= request.max_tokens or count >= getattr(request, 'eos_at', request.max_tokens)


# The relaxed evidence contract (the maintainer's choice): a native file of the async server equals the
# synchronous server's for the same (lane, turn) up to a consistent renaming. Within one file:
# block ids (any int under a block field) map one to one; identities (request ids, generations,
# bindings, nonces) map one to one; digests and MACs over renamed values may differ; free counts
# (``kv_pools``, ``http_measurement``) depend on the interleaving of other sessions' steps and are
# not compared here (the offline verifier reads them as of the step boundary). Everything else is
# exact: tokens, selections, Page vectors, carries, cursors.
BLOCK_KEYS = frozenset({'release', 'resident_blocks', 'block_ids'})      # block ids (Pages and positions are exact)
IDENTITY_KEYS = frozenset({'binding', 'worker_binding', 'request_id', 'generation', 'allocation_generation', 'nonce',
                           'lease_id', 'id', 'internal_id', 'response_id', 'namespace', 'ns_generation', 'identity',
                           'request'})
DIGEST_KEYS = frozenset({'lease_digest', 'receipt_digest', 'request_digest', 'mac', 'plan'})   # over renamed values
INTERLEAVED = frozenset({'kv_pools', 'http_measurement', 'admission_engine_ms', 'free_at_terminal'})


def relaxed_differences(sync, asyn):
    """The differences between two native files that a consistent renaming does not explain (the
    relaxed contract). [] when they are equal under it."""
    import re
    blocks, identities, bad = {}, {}, []
    # A block renaming is one to one within a frame, a field and a KV group (the first list index
    # under the field): slots are reused across frames, and the pools number their blocks apart.
    scoped = re.compile(r'(.*?)\.(release|resident_blocks|block_ids)((?:\.[^.\[]+)?)\[(\d+)\]')

    def bijection(table, a, b, where):
        forward, backward = table.setdefault('f', {}), table.setdefault('b', {})
        if forward.setdefault(a, b) != b or backward.setdefault(b, a) != a:
            bad.append((where, 'inconsistent renaming', a, b))

    def block(a, b, where):
        m = scoped.match(where)
        bijection(blocks.setdefault((m.group(1), m.group(2), m.group(4)) if m else where, {}), a, b, where)

    def walk(x, y, path, keys):
        key = keys[-1] if keys else None
        if any(k in INTERLEAVED for k in keys):
            return
        if key == 'release' and isinstance(x, list) and isinstance(y, list) and \
                all(isinstance(e, list) and len(e) > 2 for e in x + y):
            # Released slots: [group, block, generation, kind, block incarnation], sorted by block id (so
            # a renaming reorders them): the same (group, kind) multiset and distinct blocks per group.
            from collections import Counter
            shape = lambda entries: (Counter((e[0], e[3]) for e in entries),
                                     Counter(g for g, _ in {(e[0], e[1]) for e in entries}))
            if shape(x) != shape(y):
                bad.append((path, 'released slots', len(x), len(y)))
            return
        if isinstance(x, dict) and isinstance(y, dict):
            if set(x) != set(y):
                bad.append((path, 'keys', sorted(set(x) ^ set(y), key=str)[:4], None))
            for k in set(x) & set(y):
                walk(x[k], y[k], path + '.' + str(k), keys + (str(k),))
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                bad.append((path, 'length', len(x), len(y)))
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, path + '[%d]' % i, keys)
        elif x == y:
            if type(x) is int and any(k in BLOCK_KEYS for k in keys):
                block(x, y, path)
        elif type(x) is int and type(y) is int and any(k in BLOCK_KEYS for k in keys):
            block(x, y, path)
        elif any(k in IDENTITY_KEYS for k in keys) and type(x) is type(y):
            bijection(identities, x, y, path)
        elif key in DIGEST_KEYS and type(x) is str and type(y) is str:
            pass
        else:
            bad.append((path, 'value', json.dumps(x)[:40], json.dumps(y)[:40]))
    walk(sync, asyn, '', ())
    return bad


def project(h):
    """Per (lane, turn): the reply, the selection, the attended Pages, the carry and Page vectors of
    the completion (the worker's complete receipt) - what must not depend on the step order."""
    out = {}
    for info in h.done:
        key = (info['lane'].index, info['turn'] if 'turn' in info else None)
        receipt = {k: v for k, v in info['receipt'].items()
                   if k in ('carry', 'carry_digest', 'computed_end', 'sealed_end', 'vectors', 'page_vectors',
                            'published', 'digests', 'catalog')}
        out[key] = dict(outputs=list(info['request'].output_token_ids), selected=list(info['plan']['selected']),
                        restore=list(info['plan'].get('restore', ())), receipt=receipt,
                        visible=h.runner.visible.get(info['rid']), plans=h.runner.plans.get(info['rid']))
    return out


def natives(h):
    """(lane, turn) -> the request's native file and the worker's complete receipt."""
    return {(info['lane'].index, info['turn']): dict(native=info['native'], receipt=info['receipt']) for info in h.done}


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class AsyncEndToEndTests(unittest.TestCase):
    """Synchronous fast path 2 vs --async-scheduling over test_decouple_e2e's multi-session harness."""
    maxDiff = None

    def serve(self, case, policy, asynchronous, *, tasks=None, mutate=None):
        import test_decouple_e2e as D
        import test_decode_fastpath2 as T2
        import test_host_trim as HT
        import test_stageb_worker as W
        from remote_ra_live_v2 import decode_fastpath as F
        tasks = D.TASKS if tasks is None else tasks
        multi = D.Case(**dict(vars(case), sessions=len(tasks)))
        with ExitStack() as stack:
            D.enter_case(stack, multi)
            HT.deterministic(stack)
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = dict(D.config(policy, output, multi, tasks), **BASE)
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            if asynchronous:
                stack.enter_context(patch.object(D, 'Runner', async_runner(D.Runner, W, D)))
                h = async_harness(D)(policy, output, multi, tasks)
                link = attach_async(h, scripted_stop)
                stack.callback(A.detach_engine)
                if mutate is not None:
                    mutate(stack, h)
            else:
                h = D.MultiHarness(policy, output, multi, tasks)
                F.attach_worker(h.owner, h.bridge, h.qwen)
                T2.attach_fast2(h)
                link = None
            h.active = h.restores = 0
            h.serve()
            for info in h.done:
                info['turn'] = sum(1 for other in h.done[:h.done.index(info)] if other['lane'] is info['lane'])
            result = dict(project=project(h), natives=natives(h), widths=list(h.widths))
            if asynchronous:
                result.update(voided=list(h.runner.voided_rows), extra=list(h.runner.extra_rows),
                              drains=h.async_drains, voids=link.voids, resolves=link.resolves,
                              unbatched=link.unbatched_requests, length_skips=list(h.length_skips))
            h.close()
            D.engine_closed(h.core)
            result['files'] = sorted(p.name for p in output.iterdir())
            return result

    def check(self, case, policy, tasks=None):
        sync = self.serve(case, policy, False, tasks=tasks)
        asyn = self.serve(case, policy, True, tasks=tasks)
        self.assertEqual(sorted(asyn['project']), sorted(sync['project']))
        for key in sorted(sync['project']):
            self.assertEqual(asyn['project'][key], sync['project'][key], key)
        # Native file names hash the request id (renamed with the admission order); the rest are the same.
        named = lambda files: ([n for n in files if not n.startswith('native-')],
                               sum(n.startswith('native-') for n in files))
        self.assertEqual(named(asyn['files']), named(sync['files']))
        for key in sorted(sync['natives']):
            self.assertEqual(relaxed_differences(sync['natives'][key], asyn['natives'][key]), [], key)
        self.assertEqual(asyn['extra'], [])                     # no row ran past its stop
        self.assertEqual(len(asyn['voided']), asyn['voids'])
        return sync, asyn

    def test_async_equals_sync_per_turn(self):
        import test_decouple_e2e as D
        for case in D.CASES:
            for policy in ('full', 'observation_query'):
                if not (D.SLOW or (case.name, policy) in (('legacy-784', 'full'), ('16-784', 'observation_query'))):
                    continue
                with self.subTest(case=case.name, policy=policy):
                    sync, asyn = self.check(case, policy)
                    self.assertGreater(asyn['voids'], 0)            # EOS stops were voided in a later step
                    self.assertGreater(max(asyn['widths']), 1)      # several requests per step
                    # Phase-1 passes took running requests out of the batch (async-fix2).
                    self.assertGreater(asyn['unbatched'], 0)
                    # max_tokens stops vLLM did not schedule past, next to requests it did schedule: the
                    # resolve of that step met a request of the step before outside it (GPU job 3568311).
                    self.assertTrue(any(scheduled for _, scheduled in asyn['length_skips']), asyn['length_skips'])

    def test_a_phase1_pass_between_steps_needs_the_token_ids(self):
        """async-fix2 (GPU job 3519531): an admission's phase 1 runs as its own pass, which takes the running
        requests out of the runner's batch; the next step must send their token ids (EngineLink.unbatched), as
        vLLM's async runner adds a request with output tokens back from them alone. Mutation: without the
        report the next step after such a pass raises the GPU's KeyError."""
        import test_decouple_e2e as D

        def no_report(stack, h):
            stack.enter_context(patch.object(A.EngineLink, 'unbatched', lambda self, rids: None))
        with self.assertRaises((KeyError, ExceptionGroup)) as caught:
            self.serve(D.CASES[2], 'observation_query', True, mutate=no_report)
        error = caught.exception
        if isinstance(error, ExceptionGroup):
            self.assertTrue(error.subgroup(KeyError), error)

    def test_one_session(self):
        import test_decouple_e2e as D
        sync, asyn = self.check(D.CASES[2], 'goal_query', tasks=D.TASKS[:1])
        self.assertGreater(asyn['voids'], 0)
        self.assertEqual(max(asyn['widths']), 1)

    def test_wrong_block_renaming_is_caught(self):
        """Mutation of the relaxed comparator's input: one block id of a turn's file moved onto another
        block's id (two sync blocks then share one async id) is refused, as is a changed carry."""
        import copy
        import test_decouple_e2e as D
        sync = self.serve(D.CASES[2], 'observation_query', False)
        asyn = self.serve(D.CASES[2], 'observation_query', True)
        caught = 0
        for key in sorted(sync['natives']):
            files = copy.deepcopy(asyn['natives'][key])
            resident = files['receipt'].get('resident_blocks') or {}
            pages = [p for p, groups in resident.items() if groups and groups[0]]
            if len(pages) < 2:
                continue
            resident[pages[1]][0][0] = resident[pages[0]][0][0]       # two Pages on one block
            self.assertTrue(relaxed_differences(sync['natives'][key], files), key)
            caught += 1
            files = copy.deepcopy(asyn['natives'][key])
            files['receipt']['carry'] = dict(files['receipt']['carry'], digest='0' * 64) \
                if isinstance(files['receipt'].get('carry'), dict) else 'changed'
            self.assertTrue(relaxed_differences(sync['natives'][key], files), key)
        self.assertGreater(caught, 0)

    def test_missing_void_reaches_the_carry(self):
        """Mutation: the resolve decides the stops but writes nothing on the device. The stopped
        request's extra row runs: its GDN slot advances by the stop token and its KV is written at
        C'. The completion's carry then differs from the synchronous server's (or the next turn's
        continuity check fails)."""
        import test_decouple_e2e as D

        def no_void(stack, h):
            stack.enter_context(patch.object(A, 'void_rows', lambda *args, **kwargs: (1, 1)))
        sync = self.serve(D.CASES[2], 'observation_query', False)
        try:
            broken = self.serve(D.CASES[2], 'observation_query', True, mutate=no_void)
        except (AssertionError, LiveV2Error, RuntimeError):
            return                                               # caught by a continuity check
        self.assertTrue(broken['extra'])
        self.assertNotEqual(broken['project'], sync['project'])

    def test_missed_stop_fails_closed(self):
        """Mutation: the oracle misses every stop. The stopped request keeps a lease in the next
        step and its completion is refused (or the link check fails)."""
        import test_decouple_e2e as D

        def blind(stack, h):
            stack.enter_context(patch.object(A._LINK, 'stops', lambda rid, token: False))
        with self.assertRaises((LiveV2Error, AssertionError, RuntimeError, ExceptionGroup)):
            self.serve(D.CASES[2], 'full', True, mutate=blind)

    def test_wrong_prediction_fails_closed(self):
        """Mutation: the oracle predicts a stop that is not one: the step is lost; the link refuses."""
        import test_decouple_e2e as D

        def eager(stack, h):
            stack.enter_context(patch.object(A._LINK, 'stops', lambda rid, token: True))
        with self.assertRaises((LiveV2Error, AssertionError, RuntimeError, ExceptionGroup)):
            self.serve(D.CASES[2], 'full', True, mutate=eager)

    def test_a_drain_with_no_worker_rpc_before_the_next_step(self):
        """A pass boundary that drains the batch in flight and makes no worker RPC before the next step (a
        CAPACITY_WAIT refusal, a utility request, a wakeup): the next resolve settles that step without
        deciding its stops again (EngineLink.applied, async-gemma). Every pass drains here: no stop needs a
        void and every turn equals the synchronous server's. Without the check (the mutation) a drained
        token is counted twice, a request with one token left is voided and its step is lost (fail closed)."""
        import test_decouple_e2e as D

        def drain_every_pass(stack, h):
            step = h.batch_step

            def drained():
                h.drain()
                if h.running:
                    step()
            h.batch_step = drained
        sync = self.serve(D.CASES[2], 'observation_query', False)
        asyn = self.serve(D.CASES[2], 'observation_query', True, mutate=drain_every_pass)
        self.assertEqual(asyn['project'], sync['project'])
        for key in sorted(sync['natives']):
            self.assertEqual(relaxed_differences(sync['natives'][key], asyn['natives'][key]), [], key)
        self.assertEqual((asyn['voids'], asyn['extra']), (0, []))
        self.assertGreater(asyn['drains'], 0)

        def twice(stack, h):
            drain_every_pass(stack, h)
            stack.enter_context(patch.object(A.EngineLink, 'applied', lambda self, rid, cursor: False))
        with self.assertRaisesRegex(LiveV2Error, 'predicted stop did not stop'):
            self.serve(D.CASES[2], 'observation_query', True, mutate=twice)


if __name__ == '__main__':
    unittest.main()
