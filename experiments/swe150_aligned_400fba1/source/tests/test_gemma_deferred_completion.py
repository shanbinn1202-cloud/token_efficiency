"""--deferred-completion-io on the Gemma 4 profile's window layout (gemma-deferred-io). CPU only; not GPU evidence.

The switch (remote_ra_live_v2.deferred_completion, test_deferred_completion) moves the native
terminal's file writing, which both profiles share, to one writer thread. What the window layout
adds to a completion stays where it was on the engine thread: the worker's window carry and the
release of its sliding leases in the ``complete`` RPC, WindowPins.pin_after (the next window
pinned before vLLM frees the request), phase 1's leftover blocks (recover_probes ->
WindowPhase1Coordinator.release), then, after the native terminal, vLLM's free (the request's
other sliding blocks back to pool 1) and WindowPins.after_free (the pinned window keeps the
session's pin alone). The record is the same code's; pool 1's free blocks are read with pool 0's at the same point. The native file's one
reader on the three protocols is FrontendEvidence.complete, after the engine acknowledged the
authenticated Cancel (held while the files are written); a session's admission, its close (also
a chat-text idle close or eviction, which send the same CloseSession frame) and an idle release
join its writes. Covered:

- the launcher on a Gemma release (test_gemma_flags' bases with CUDA graphs, the decode fast paths,
  --release-idle-pages and --admission-overlap; the kv-session and chat-text protocols): the switch
  adds exactly its startup key and changes no vLLM argument; it is refused as a non-switch, and the
  profile's other refusals hold with it; config.load and model_profile accept it, ready.json records
  it, compose_engine attaches the writer on the window layout (both pools; an idle release joins)
  and sends the same worker startup record as without it;
- the window layout's engine and worker (test_gemma_e2e's fakes: vLLM's KV manager with the
  sliding managers, WindowPins, WindowPhase1Coordinator, WindowAllocationOwner, a runner that checks
  every full-attention and sliding read), deterministic, with the switch off and on: one and three
  mswea-xml sessions (test_gemma_fastpath's script: Page and forward chunks, rounds decoded
  together, a rebuild), eager and with --cuda-graphs full and both decode fast paths; sessions
  under pool pressure with --release-idle-pages (idle sessions released and readmitted), also with
  graphs, both fast paths and chunked phase 1; MemAgent sessions over the kv-session protocol with
  releases. Each pair writes byte-identical evidence files (native files, goals.jsonl,
  capacity.jsonl, engine-cleanup.json) and sends the same RPCs, frames and replies; every file the
  writer writes equals the synchronous path's bytes of the same moment (a shadow directory), and
  nothing is left pending or failed; goal_query with --admission-overlap K and releases (Goals parked
  on their sessions' threads while others decode) decides as the K = 1 server without the switch,
  every file the writer wrote equal to the synchronous path's;
- the order on the window layout while a completion's files are pending: a cancellation waits for
  them and then writes synchronously, an admission of that session waits for them and one of
  another session does not, the shutdown closes the writer (its last files) before the request
  drain, whose terminals are synchronous; windows and both pools are back at their baselines;
- the API servers over the window engine (the packaged router, the frontend, the authenticated
  Cancel, test_ms2_e2e's engine loop): Gemma's chat-text sessions (sixteen decoding together, with
  releases, a cancel and its rebind, the hand-off and close) and mswea-xml sessions (decoding
  together, a cancel between batched steps, the shutdown with requests in flight) pass their checks
  with the switch, also with a slow writer whose acknowledgements are held, and reply as without
  it; so do Gemma's MemAgent sessions through the kv-session route (with their closes) over the window
  engine, and on test_kv_session_e2e's engine as test_gemma_kv_session_e2e runs them.

The suite runs the shorter cases (full, observation_query and goal_query; six sessions under
pressure); REMOTE_RA_SLOW_TESTS=1 adds every policy, sixteen sessions and every API scenario.
"""
import asyncio
from contextlib import ExitStack
import functools
from hashlib import sha256
import itertools
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import serve  # noqa: E402
import test_decode_fastpath3 as TF3  # noqa: E402
import test_deferred_completion as TD  # noqa: E402
import test_gemma_chat_text_e2e as GCE  # noqa: E402
import test_gemma_e2e as GE  # noqa: E402
import test_gemma_fastpath as TF  # noqa: E402
import test_gemma_flags as GFL  # noqa: E402
import test_gemma_full as GF  # noqa: E402
import test_gemma_goal as GG  # noqa: E402
import test_gemma_kv_session_e2e as GKS  # noqa: E402
import test_gemma_overlap as GO  # noqa: E402
import test_gemma_release as GR  # noqa: E402
import test_host_trim as HT  # noqa: E402
import test_integrate_flags as IF  # noqa: E402
import test_ms2_e2e as B2  # noqa: E402
import test_ms_e2e as E2E  # noqa: E402
import test_stageb_worker as W  # noqa: E402

from remote_ra_http import config as HC  # noqa: E402
from remote_ra_http import evidence as EV  # noqa: E402
from remote_ra_http import mswea_route  # noqa: E402
from remote_ra_http.evidence import engine_closed  # noqa: E402
from remote_ra_live_v2 import deferred_completion as D  # noqa: E402
from remote_ra_live_v2 import kv_layout as I0  # noqa: E402
from remote_ra_live_v2 import model_profile as MP  # noqa: E402
from remote_ra_live_v2.composition import compose_frontend  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
KEY = D.CONFIG_KEY
A, B, C = GE.A, GE.B, GE.C
GRAPHS = GFL.flags('full', GFL.BOTH)        # --cuda-graphs full --decode-fastpath --decode-fastpath2
SLOW_WRITER = TD.SLOW_WRITER                # a pause of 2 ms after every piece: acknowledgements wait
COMPARED = ('trace', 'files', 'frames', 'computed', 'visible', 'plans', 'windows', 'done', 'batched', 'steps')
THIS = sys.modules[__name__]


def quiet_close(writer):
    """A test's exit after a failure: stop the writer thread without raising."""
    try:
        writer.close(5)
    except BaseException:
        pass


def natives(files):
    return [name for name in files if name.startswith('native-')]


def native_name(rid):
    return 'native-' + sha256(rid.encode()).hexdigest() + '.json'


class WriterChecks:
    """Every native file came from the writer and equals the synchronous path's bytes of the same
    moment; nothing is pending or failed; the writer stopped."""

    def writer_done(self, writer, files):
        stats = writer.stats()
        self.assertEqual((writer.compared, writer.mismatches, stats['jobs'], stats['errors'], stats['pending'],
                          stats['staged'], stats['queued']), (len(natives(files)), [], len(natives(files)), 0, 0, 0, 0))
        self.assertTrue(writer.closed and not writer.thread.is_alive())


# ---- 1. the launcher and startup on a Gemma release ----------------------------------------------

LAUNCH_COMBINATIONS = [GFL.PLAIN, ('full', GFL.BOTH, True, GFL.K), ('off', (GFL.FP,), True, 1),
                       ('piecewise', GFL.BOTH, False, GFL.K), ('full-decode', (), True, 1)]
KV_SESSION = dict(protocol='kv-session', kv_session_thinking='off', max_running=16, max_model_len=262144,
                  release_idle_pages=True, dram_arena_gib=150, host_memory_shares=2, kv_cache_gib=28, page_tokens=16,
                  max_tokens=4096, language_model_only=True, connector_pool_gib=1)
CHAT_TEXT = dict(protocol='chat-text', kv_cache_gib=28, page_tokens=16, max_tokens=4096, max_model_len=65536,
                 language_model_only=True, max_running=20, dram_arena_gib=64, connector_pool_gib=1,
                 release_idle_pages=True, cuda_graphs='full', decode_fastpath=True, chat_text_slots=512,
                 chat_text_idle_timeout=600, chat_text_evict_after=30)


def worker_echo(config):
    """The records of a startup config that the worker's evidence echoes (context, text only, the admission
    flags, the decode fast paths, the release switch); absent when off."""
    from remote_ra_live_v2.context_limits import RECORD_KEY, record
    from remote_ra_live_v2.language_model_only import CONFIG_KEY as TEXT_ONLY, record as text_only
    echo = {RECORD_KEY: record(config), TEXT_ONLY: text_only(config), 'admission': config.get('admission')}
    echo.update({name: config.get(name) for name in ('decode_fastpath', 'decode_fastpath2', 'release_idle_pages')})
    return {k: v for k, v in echo.items() if v is not None}


def gemma_ready_json(config):
    """FrontendEvidence.write_ready's ready.json for a loaded Gemma startup config
    (test_deferred_completion.ready_json's stand-ins; the worker's evidence echoes the config's records)."""
    import pathlib
    import test_ms_frontend as MF
    readiness = dict(native_ready=True)
    authority = NS(identity=NS(epoch=MF.EPOCH, manifest='m' * 64), policy=config['policy'],
                   session_id=MF.SERVER, verify_admission=lambda: True, live_v2_readiness=readiness,
                   loaded=NS(manifest=dict(qualification_parent_manifest='q' * 64)))
    receipt = dict(tensor_geometry={}, source_closure={}, **worker_echo(config))
    read_text = pathlib.Path.read_text

    def proc(path, *args, **kwargs):
        name = str(path).replace('\\', '/')
        if name == '/proc/sys/kernel/random/boot_id':
            return 'boot\n'
        if name == '/proc/self/stat':
            return '1 (python) ' + ' '.join(['0'] * 30)
        return read_text(path, *args, **kwargs)
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


def compose(config, env):
    """compose_engine on a loaded Gemma startup config (``env``) as far as the writer: the engine core and
    the worker startup record it sent (test_deferred_completion.ComposeTests' stand-ins; the worker's
    echo carries the config's records)."""
    import queue
    import types
    from remote_ra_live_v2 import composition
    sent = []
    echo = dict(evidence_kind='CUDA_PHYSICAL_TENSOR_OPERATIONS', loaded_sources_verified=True, **worker_echo(config))

    def collective_rpc(method, args, timeout):
        sent.append(args[0])
        return [dict(echo)]

    class Core:
        def post_step(self, executed):
            pass

        def _remote_ra_ack(self, call_id, frame, failure):
            pass
    core = Core()
    layout = I0.from_vllm_config(NS(additional_config={I0.LAYOUT_KEY: config['kv_layout']}))
    assembly = NS(provider=NS(provenance=NS(page_tokens=16)), state='active',
                  lifecycle=NS(bind_native=lambda native: None, tick=lambda force=False: None),
                  dispatcher=NS(_max_records=1, handle_steps=lambda frame: iter(())))
    vars(core).update(_remote_ra_live_v2={}, _remote_ra_assembly=assembly, _remote_ra_trusted_issue=dict(config['task']),
                      _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
                      _remote_ra_growing_adapter=NS(layout=layout), model_executor=NS(collective_rpc=collective_rpc),
                      scheduler=NS(), vllm_config=NS(scheduler_config=NS(max_num_seqs=16)), input_queue=queue.Queue())
    engine = types.ModuleType('vllm.v1.engine')
    engine.EngineCoreRequestType = NS(WAKEUP='WAKEUP')
    stubs = {'vllm': types.ModuleType('vllm'), 'vllm.v1': types.ModuleType('vllm.v1'), 'vllm.v1.engine': engine}
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, env))
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
    return core, sent


class LauncherTests(GFL.Release, unittest.TestCase):
    """test_gemma_flags' launches with --deferred-completion-io: exactly its key is added."""
    maxDiff = None

    def same_but_the_key(self, on, off):
        (c1, m1, e1), (c0, m0, e0) = on, off
        self.assertEqual((m1, e1), (m0, e0))                    # no vLLM argument, the same environment
        c1 = dict(c1)
        self.assertIs(c1.pop(KEY), True)
        self.assertNotIn(KEY, c0)
        self.assertEqual(c1, c0)

    def test_every_base_and_combination_adds_only_its_key(self):
        combinations = GFL.COMBINATIONS if SLOW else LAUNCH_COMBINATIONS
        for base in GFL.BASES:
            for combination in combinations:
                with self.subTest(base=base, combination=combination):
                    status, on, _ = self.launch(base, combination, deferred_completion_io=True)
                    self.assertEqual(status, 'ok', on)
                    self.same_but_the_key(on, self.launched(base, combination)[0])

    def prepare(self, kw, policy, selection, deferred):
        name = self.name('proto')
        with patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            _, config, command, env = serve.prepare(self.release, None, policy, self.root / name, 8000, selection,
                                                    deferred_completion_io=deferred, **kw)
        self.assertEqual(env['REMOTE_RA_HTTP_CONFIG_SHA256'], serve.pin(config))
        env = {k: v for k, v in env.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'}
        return json.loads(json.dumps([config, command, env]).replace(name, 'OUT')), config

    def test_the_kv_session_and_chat_text_protocols(self):
        for kw, policy, selection in ((KV_SESSION, 'observation_query', dict(active_token_budget=32768)),
                                      (KV_SESSION, 'goal_query', dict(active_token_budget=32768, no_goal_rewriter=True)),
                                      (CHAT_TEXT, 'observation_query', dict(active_token_budget=8192)),
                                      (dict(CHAT_TEXT, kv_cache_gib=32), 'full', None)):
            with self.subTest(protocol=kw['protocol'], policy=policy):
                on, config = self.prepare(kw, policy, selection, True)
                off, _ = self.prepare(kw, policy, selection, False)
                self.same_but_the_key(on, off)
                value, _ = self.load(config)
                self.assertEqual((HC.model_profile(value), HC.deferred_completion_io(value)), (MP.GEMMA4, True))

    def test_refused_as_a_non_switch_and_with_the_other_refusals(self):
        for bad in (1, 'yes', None):
            status, message, _ = self.launch('full', deferred_completion_io=bad)
            self.assertEqual((status, message), ('refused', '--deferred-completion-io is a switch'))
        for extra, message in GFL.REFUSED:
            with self.subTest(extra=extra):
                status, error, _ = self.launch('goal_query', ('full', GFL.BOTH, True, GFL.K),
                                               deferred_completion_io=True, **extra)
                self.assertEqual(status, 'refused')
                self.assertIn(message, error)
        status, error, _ = self.launch('full', ('off', (GFL.FP2,), False, 1), deferred_completion_io=True)
        self.assertEqual(status, 'refused')
        self.assertIn(GFL.FP2_ALONE, error)

    def test_load_model_profile_and_ready(self):
        for base, combination in (('full', GFL.PLAIN), ('goal_query-rewriter', ('full', GFL.BOTH, True, GFL.K)),
                                  ('observation_query', ('off', (GFL.FP,), True, 1))):
            with self.subTest(base=base, combination=combination):
                _, _, config = self.launch(base, combination, deferred_completion_io=True)
                value, env = self.load(config)
                self.assertEqual(value, config)
                self.assertEqual((HC.model_profile(value), HC.deferred_completion_io(value)), (MP.GEMMA4, True))
                with patch.dict(os.environ, env):
                    self.assertIs(D.loaded(), True)
                for bad in (False, 0, 'yes'):
                    with self.assertRaisesRegex(RuntimeError, 'deferred_completion_io must be true'):
                        self.load(dict(config, **{KEY: bad}))
                # ready.json records the switch; nothing else changes.
                on, off = gemma_ready_json(config), gemma_ready_json(self.launched(base, combination)[1])
                self.assertIs(on.pop(KEY), True)
                self.assertNotIn(KEY, off)
                differ = ('server_run_id', 'common_config_sha256', 'input_pins_sha256')
                self.assertEqual({k: v for k, v in on.items() if k not in differ},
                                 {k: v for k, v in off.items() if k not in differ})

    def test_dry_run_main(self):
        import contextlib
        import io
        task = self.root / 'gemma-task.json'
        task.write_bytes(json.dumps(GFL.TASKS[0]).encode())
        commands = []
        for extra in ([], ['--deferred-completion-io']):
            out = self.root / self.name('main')
            argv = ['serve.py', '--no-speed-defaults', '--release-dir', str(self.release), '--task', str(task), '--policy', 'observation_query',
                    '--active-token-budget', '8192', '--page-tokens', '16', '--kv-cache-gib', '28',
                    '--max-model-len', '131072', '--max-tokens', '4096', '--language-model-only',
                    '--output', str(out), '--port', '8000'] + extra
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()) as printed, \
                    patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                serve.main()
            value = json.loads(printed.getvalue().strip().splitlines()[-1])
            self.assertEqual(value['status'], 'SERVICE_DRY_RUN')
            commands.append([str(c).replace(str(out), '<out>') for c in value['command']])
        self.assertEqual(commands[0], commands[1])


class ComposeTests(GFL.Release, unittest.TestCase):
    """compose_engine on a Gemma startup config: the writer is attached (the window layout's two pools; an
    idle release joins the victim's writes) and the worker startup record is the one without the switch."""
    maxDiff = None

    def test_attached_on_the_window_layout(self):
        for combination in (GFL.PLAIN, ('off', (), True, 1)):
            with self.subTest(combination=combination):
                _, _, on = self.launch('observation_query', combination, deferred_completion_io=True)
                off = self.launched('observation_query', combination)[1]
                core_on, sent_on = compose(*self.load(on))
                core_off, sent_off = compose(*self.load(off))
                self.assertEqual(sent_on, sent_off)
                self.assertNotIn(KEY, sent_on[0])
                self.assertIn('kv_layout', sent_on[0])
                self.assertIsNone(vars(core_off).get('_remote_ra_completion_writer'))
                writer = core_on._remote_ra_completion_writer
                self.addCleanup(quiet_close, writer)
                self.assertIsInstance(writer, D.CompletionWriter)
                self.assertTrue(writer.decoupled and writer.thread.is_alive())
                self.assertEqual(str(writer.output), on['output_directory'])
                bridge = core_on._remote_ra_assembly.dispatcher._remote_ra_live_v2
                self.assertEqual((bridge.cancel.__name__, bridge.add_steps.__name__, bridge.close_session.__name__),
                                 ('holding_cancel', 'joined_add_steps', 'joined_close_session'))
                release = getattr(core_on, '_remote_ra_idle_release', None)
                self.assertEqual(release is not None, combination[2])
                if release is not None:
                    self.assertEqual(release.evict.__name__, 'joined_evict')
                writer.close()


# ---- 2. the window layout's engine and worker, one and three sessions (mswea-xml) ---------------------

class WindowRun:
    """test_gemma_fastpath.Run's script over test_gemma_e2e's Harness with ``tasks`` (one session each),
    deterministic (test_host_trim.deterministic). ``flags``: startup config entries and ``attach``
    their engine and worker attachments (test_gemma_flags.attacher); ``deferred``: the switch as
    compose_engine attaches it, with a shadow writer (``writer``: its options); a native file is read
    after its request's writes, and the writer closes first at shutdown."""

    def __init__(self, policy, flags=None, *, tasks=(A, B, C), attach=None, deferred=False, writer=None):
        self.policy, self.flags, self.tasks = policy, dict(flags or {}), tuple(tasks)
        self.attach, self.deferred, self.options = attach, deferred, dict(writer or {})

    def __enter__(self):
        stack = self.stack = ExitStack()
        HT.deterministic(stack)
        output = self.output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        layout = GE.window_layout(self.policy, len(self.tasks))
        value = self.config = dict(GE.config(self.policy, output, list(self.tasks), layout), **self.flags)
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        h = self.h = TD.deferred_harness(GE.Harness)(self.policy, output, list(self.tasks))
        if self.attach is not None:
            self.attach(h)
        self.trace, rpc = [], h.collective_rpc

        def collective_rpc(method, args, timeout):
            operation, payload = args
            sent = HT.plain(payload)
            reply = rpc(method, args, timeout)
            self.trace.append(dict(operation=operation, payload=sent, receipt=HT.plain(reply[0])))
            return reply
        h.proxy.executor = NS(collective_rpc=collective_rpc)
        self.steps, step = [], h.owner.step

        def observed_step(runner, scheduler_output):
            rows = step(runner, scheduler_output)
            self.steps.append(([(s.request, s.start, s.count, TF.keys(s.attention), TF.keys(s.hybrid),
                                 [list(k) for k in s.current]) for s in rows],
                               sorted((list(k), list(v)) for k, v in h.owner.claims.items())))
            return rows
        h.owner.step = observed_step
        self.writer = None
        if self.deferred:
            shadow = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            self.writer = h.attach_deferred(value, shadow=shadow, **self.options)
            stack.callback(quiet_close, self.writer)
        self.infos = []
        return self

    def __exit__(self, *exc):
        self.stack.close()

    def script(self, rounds=4, together=3, rebuild=True):
        """test_gemma_fastpath.Run.script: turns one session at a time (Page and forward chunks), rounds
        decoded together, a rebuild of the second session (the first of one) and its next turn."""
        h, infos = self.h, self.infos
        for k in range(rounds):
            for lane in h.lanes:
                observation = None if k == 0 else GE.OBSERVATIONS[(k - 1 + lane.index) % len(GE.OBSERVATIONS)]
                infos.append(h.turn(lane, observation, target=GE.TARGETS[(k + lane.index) % len(GE.TARGETS)],
                                    eos=(k + lane.index) % 3 == 2, chunk='forward' if (k + lane.index) % 2 else 'page'))
        for k in range(together):
            batch = [self.prefilled(lane, GE.OBSERVATIONS[(k + 2 * lane.index) % len(GE.OBSERVATIONS)],
                                    GE.TARGETS[(k + lane.index + 3) % len(GE.TARGETS)]) for lane in h.lanes]
            h.decode_together(batch)
            for info in batch:
                h.complete(info)
                self.committed(info)
        if rebuild:
            lane = h.lanes[min(1, len(h.lanes) - 1)]
            retry = lane.messages[:-1] + [dict(role='user', content='Format error: expected exactly one action.')]
            infos.append(h.turn(lane, messages=retry, target=3))
            infos.append(h.turn(lane, GE.OBSERVATIONS[5], target=9))
        return self

    def prefilled(self, lane, observation, target):
        """A request of ``lane`` admitted and prefilled in Page chunks, its reply planned."""
        h = self.h
        info = h.submit(lane, lane.messages + [dict(role='user', content=observation)])
        info['outputs'] = h.reply(info, target, False)
        request = info['request']
        h.runner.samples[request.request_id] = iter(info['outputs'])
        first = True
        while request.num_computed_tokens < len(request.prompt_token_ids):
            start = request.num_computed_tokens
            h.step(request, min(len(request.prompt_token_ids), (start // GE.P + 1) * GE.P) - start, first)
            first = False
        return info

    def committed(self, info):
        info['lane'].messages = info['messages'] + [dict(role='assistant',
                                                         content=' ' + info['committed']['text'] + '\n')]
        self.infos.append(info)
        return info

    def result(self):
        h = self.h
        frames = {rid: HT.plain(items) for rid, items in h.proxy.http_receipts.items()}
        if self.writer is not None:
            self.writer.close()                  # shutdown_engine's first stage, then the drain
        h.lifecycle.disconnect()
        h.sessions.close()
        engine_closed(h.core)
        files = {p.name: p.read_bytes() for p in sorted(self.output.iterdir())}
        return dict(trace=self.trace, files=files, frames=frames, computed=list(h.runner.computed),
                    visible=dict(h.runner.visible), plans=dict(h.runner.plans), windows=dict(h.runner.windows),
                    done=[(i['rid'], list(i['request'].output_token_ids), list(i['plan']['selected']))
                          for i in self.infos],
                    batched=list(h.batched), steps=self.steps, writer=self.writer,
                    pools=(h.pool.get_num_free_blocks() - h.baseline, h.window_pool.get_num_free_blocks() - h.window_baseline,
                           h.adapter.window.pinned_blocks()))


_RUNS = {}


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class WindowSessionTests(WriterChecks, unittest.TestCase):
    """One and three mswea-xml sessions on the window layout, eager and with graphs and both decode fast
    paths: the switch on equals it off."""
    maxDiff = None

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))

    def serve(self, policy, graphs, deferred, tasks, writer=None, **script):
        key = (policy, graphs, tuple(t['instance_id'] for t in tasks), tuple(sorted(script.items())))
        if deferred or key not in _RUNS:
            attach = GFL.attacher('full', GFL.BOTH) if graphs else None
            with WindowRun(policy, GRAPHS if graphs else {}, tasks=tasks, attach=attach, deferred=deferred,
                           writer=writer) as run:
                result = run.script(**script).result()
            if deferred:
                return result
            _RUNS[key] = result
        return _RUNS[key]

    def compare(self, policy, graphs, tasks=(A, B, C), writer=None, **script):
        off = self.serve(policy, graphs, False, tasks, **script)
        on = self.serve(policy, graphs, True, tasks, writer, **script)
        for key in COMPARED:
            self.assertTrue(on[key] == off[key], (key, TF3.first_difference(off[key], on[key])))
        self.writer_done(on['writer'], on['files'])
        self.assertEqual(on['pools'], (0, 0, 0))             # windows and both pools back at their baselines
        self.assertTrue(any(n > 2 * GE.WINDOW for n in on['windows'].values()))
        if policy == 'goal_query':
            self.assertIn('goals.jsonl', on['files'])
        return on

    def test_three_sessions_eager(self):
        for policy in GE.POLICIES if SLOW else ('full', 'observation_query', 'goal_query'):
            with self.subTest(policy=policy):
                on = self.compare(policy, False, **({} if SLOW else TF.SHORT))
                self.assertTrue(any(len(step) == 3 for step in on['batched']))

    def test_three_sessions_with_graphs_both_fast_paths_and_a_slow_writer(self):
        for policy in GE.POLICIES if SLOW else ('goal_query',):
            with self.subTest(policy=policy):
                on = self.compare(policy, True, writer=SLOW_WRITER, **({} if SLOW else TF.SHORT))
                self.assertTrue(any(t['operation'] == 'allocation_batch' for t in on['trace']))
                self.assertFalse(any(t['operation'] == 'allocation_update' for t in on['trace']))

    def test_one_session(self):
        cases = [('observation_query', False), ('goal_query', True)]
        if SLOW:
            cases += [('full', False), ('goal_query', False), ('random_fill', True)]
        for policy, graphs in cases:
            with self.subTest(policy=policy, graphs=graphs):
                self.compare(policy, graphs, tasks=(A,), rounds=3, together=1)


# ---- 3. sessions under pool pressure, --release-idle-pages ---------------------------------------------

def pressure_run(test, policy, sessions, turns, *, deferred, switches=False, writer=None):
    """test_gemma_release's Pressure (``sessions`` lanes of ``turns`` turns, its checks after every turn)
    with --release-idle-pages at the selective rule's minimum pool 0 (the release rule's for ``full``);
    ``switches``: --cuda-graphs full and both decode fast paths, phase 1 in forwards of
    test_gemma_chunked's chunk; ``deferred``: the switch with a shadow writer. Deterministic.
    (per-lane result, evidence files, worker frames, the writer)."""
    layout = GE.window_layout(policy, sessions)
    blocks = GF.release_blocks(layout) if policy == 'full' else GR.selective_blocks(layout)
    with ExitStack() as stack:
        HT.deterministic(stack)
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        tasks = GR.tasks(sessions)
        if policy in GE.QUERY:
            value = GR.config(policy, output, tasks, layout, True)
        else:
            value = dict(GE.config(policy, output, tasks, layout), release_idle_pages=True)
        chunks = []
        if switches:
            value.update(GRAPHS)
            # Lane 1's third turn reads test_gemma_chunked's longest observation: its phase 1 runs in chunks.
            GFL.long_turns(stack, chunks)
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        h = TD.deferred_harness(GR.Pressure)(policy, output, tasks, blocks, True, turns)
        if switches:
            GFL.in_chunks(h)
            GFL.attacher('full', GFL.BOTH)(h)
        out = None
        if deferred:
            out = h.attach_deferred(value, shadow=Path(stack.enter_context(tempfile.TemporaryDirectory())),
                                    **(writer or {}))
            stack.callback(quiet_close, out)
        h.serve(test.check)
        test.assertEqual(len(h.done), sessions * turns)
        test.boundary(h)
        result = {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'],
                                h.runner.visible[i['rid']]) for i in h.done if i['lane'] is lane] for lane in h.lanes}
        frames = {rid: HT.plain(items) for rid, items in h.proxy.http_receipts.items()}
        if out is not None:
            out.close()
        h.lifecycle.disconnect()
        h.sessions.close()
        engine_closed(h.core)
        test.assertEqual((h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks(),
                          h.adapter.window.pinned_blocks()), (h.baseline, h.window_baseline, 0))
        files = {p.name: p.read_bytes() for p in sorted(output.iterdir())}
        test.assertEqual(any(chunks), switches)
    return result, files, frames, out


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class PressureTests(WriterChecks, unittest.TestCase):
    """test_gemma_release's sessions under pool pressure with --release-idle-pages (six sessions of four
    turns in the suite, sixteen of five with REMOTE_RA_SLOW_TESTS=1): the switch on equals it off."""
    maxDiff = None
    check, boundary, window_of = GR.SixteenSessionTests.check, GR.SixteenSessionTests.boundary, \
        GR.SixteenSessionTests.window_of

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        self.enterContext(patch.object(GE, 'GOAL', GG.SixteenGoalSessionsTests.TEST_GOAL))

    def compare(self, policy, sessions, turns, switches=False, writer=None):
        r0, f0, m0, _ = pressure_run(self, policy, sessions, turns, deferred=False, switches=switches)
        r1, f1, m1, out = pressure_run(self, policy, sessions, turns, deferred=True, switches=switches, writer=writer)
        self.assertEqual(r1, r0)                              # prompts, replies, selections, attended Pages
        self.assertTrue(m1 == m0, TF3.first_difference(m0, m1))  # every worker frame
        self.assertTrue(f1 == f0, TF3.first_difference(f0, f1))
        events = [json.loads(line) for line in f1['capacity.jsonl'].splitlines()]
        self.assertTrue(any(e['event'] == 'evict' for e in events) and any(e['event'] == 'readmit' for e in events))
        self.assertEqual(len(natives(f1)), sessions * turns)
        self.writer_done(out, f1)
        return f1

    def test_observation_and_goal_query(self):
        sessions, turns = (GR.SESSIONS, 5) if SLOW else (GG.SHORT_SESSIONS, GG.SHORT_TURNS)
        for policy in ('observation_query', 'goal_query') + (('full', 'random_fill') if SLOW else ()):
            with self.subTest(policy=policy):
                files = self.compare(policy, sessions, turns)
                self.assertEqual('goals.jsonl' in files, policy == 'goal_query')

    def test_with_graphs_both_fast_paths_and_a_slow_writer(self):
        sessions, turns = (GR.SESSIONS, 5) if SLOW else (5, 3)
        for policy in ('observation_query',) + (('goal_query', 'full') if SLOW else ()):
            with self.subTest(policy=policy):
                self.compare(policy, sessions, turns, switches=True, writer=SLOW_WRITER)


def overlap_harness(shadow_root):
    """test_gemma_overlap.OverlapPressure (--admission-overlap K: Goals built on their sessions' threads,
    the window ledger) with the switch as compose_engine attaches it (a shadow writer in ``shadow_root``),
    the writer closed when the loop ends, before the drain."""
    base = TD.deferred_harness(GO.OverlapPressure)

    class Harness(base):
        def __init__(self, *args):
            super().__init__(*args)
            self.attach_deferred(EV.load(), shadow=Path(tempfile.mkdtemp(dir=shadow_root)))

        def serve(self, check):
            try:
                base.serve(self, check)
            finally:
                self.writer.close()
    return Harness


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class OverlapTests(unittest.TestCase):
    """goal_query with --admission-overlap K and --release-idle-pages on the window layout (test_gemma_overlap's
    rewriter stand-in and slow Goals: admissions park at their Goal while others decode), with the switch:
    every turn decides as on the K = 1 server without it, and every file the writer wrote equals the
    synchronous path's of the same moment. Eight sessions of three turns in the suite, sixteen of five with
    REMOTE_RA_SLOW_TESTS=1."""
    maxDiff = None
    TURNS, TEST_GOAL = GO.OverlapEndToEndTests.TURNS, GO.OverlapEndToEndTests.TEST_GOAL
    setUp = GO.OverlapEndToEndTests.setUp
    check, boundary, window_of = GR.SixteenSessionTests.check, GR.SixteenSessionTests.boundary, \
        GR.SixteenSessionTests.window_of
    server, finish, run_case = GO.OverlapEndToEndTests.server, GO.OverlapEndToEndTests.finish, \
        GO.OverlapEndToEndTests.run_case

    def test_parked_goals_releases_and_the_switch(self):
        sessions, turns = (GR.SESSIONS, 5) if SLOW else (GO.SHORT_SESSIONS, 3)
        reference, _, _ = self.run_case(GR.REFERENCE_BLOCKS, None, sessions=sessions, turns=turns)
        small = GR.selective_blocks(GE.window_layout('goal_query', sessions))
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with patch.object(GO, 'OverlapPressure', overlap_harness(root)):
            result, h, events = self.run_case(small, sessions, sessions=sessions, turns=turns)
        self.assertEqual(result, reference)
        self.assertTrue(any(e['event'] == 'evict' for e in events) and any(e['event'] == 'readmit' for e in events))
        self.assertGreaterEqual(h.parked_peak, 2)
        writer, stats = h.writer, h.writer.stats()
        self.assertEqual((writer.compared, writer.mismatches, stats['jobs'], stats['errors'], stats['pending']),
                         (sessions * turns, [], sessions * turns, 0, 0))
        self.assertTrue(writer.closed and not writer.thread.is_alive())


# ---- 4. kv-session (MemAgent) sessions on the window layout --------------------------------------------

def memagent_run(test, policy, sessions, *, deferred, slow=False, writer=None):
    """test_gemma_kv_session_e2e's MemAgent sessions with --release-idle-pages at the selective rule's
    minimum pool 0; deterministic (the clients' ids too); ``deferred``: the switch with a shadow writer.
    (per-lane result, evidence files, the writer)."""
    layout = GE.window_layout(policy, sessions)
    blocks = GKS.GemmaMemAgentSessionTests.small(test, layout)
    ids = itertools.count(1)
    with ExitStack() as stack:
        HT.deterministic(stack)
        stack.enter_context(patch('uuid.uuid4', lambda: NS(hex='%032x' % next(ids))))
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        value = GKS.config(policy, output, layout, True, sessions)
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        h = TD.deferred_harness(GKS.Sessions)(policy, output, blocks, True, sessions, slow)
        out = None
        if deferred:
            out = h.attach_deferred(value, shadow=Path(stack.enter_context(tempfile.TemporaryDirectory())),
                                    **(writer or {}))
            stack.callback(quiet_close, out)
        h.serve(test.check)
        for lane in h.lanes:
            test.assertEqual(lane.agent.answer, 'Answer: chunk 2 names the harbour city.')
        test.boundary(h)
        result = {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'],
                                h.runner.visible[i['rid']]) for i in h.done if i['lane'] is lane] for lane in h.lanes}
        if out is not None:
            out.close()
        h.lifecycle.disconnect()
        h.sessions.close()
        engine_closed(h.core)
        test.assertEqual((h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks(),
                          h.adapter.window.pinned_blocks()), (h.baseline, h.window_baseline, 0))
        files = {p.name: p.read_bytes() for p in sorted(output.iterdir())}
    return result, files, out


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class MemAgentTests(WriterChecks, unittest.TestCase):
    """MemAgent sessions over the kv-session protocol on the window layout with releases (six sessions in
    the suite, sixteen with REMOTE_RA_SLOW_TESTS=1): the switch on equals it off."""
    maxDiff = None
    check, boundary = GKS.GemmaMemAgentSessionTests.check, GKS.GemmaMemAgentSessionTests.boundary

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))

    def test_observation_and_goal_query(self):
        sessions = GKS.SESSIONS if SLOW else GKS.SHORT
        for policy in ('observation_query', 'goal_query'):
            with self.subTest(policy=policy):
                r0, f0, _ = memagent_run(self, policy, sessions, deferred=False, slow=SLOW)
                r1, f1, out = memagent_run(self, policy, sessions, deferred=True, slow=SLOW,
                                           writer=SLOW_WRITER if policy == 'goal_query' else None)
                self.assertEqual(r1, r0)
                self.assertTrue(f1 == f0, TF3.first_difference(f0, f1))
                events = [json.loads(line) for line in f1['capacity.jsonl'].splitlines()]
                self.assertTrue({'evict', 'readmit'} <= {e['event'] for e in events})
                self.writer_done(out, f1)


# ---- 5. the order of a cancellation, admissions and the shutdown while files are pending ----------------

@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class OrderTests(unittest.TestCase):
    """The window layout's engine and worker with a writer that keeps a released job in progress (its
    pause returns only once the engine thread waits for it): who waits for a completion's files."""
    maxDiff = None

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))

    @staticmethod
    def drained(h, info):
        """The success path up to the native terminal and vLLM's free (the files staged), then post_step."""
        h.route.finalize(h.dispatcher, info['rid'], True)
        h.free(info['request'])
        h.post_step(True)

    @staticmethod
    def commit(h, info):
        """The frontend's commit of a completed turn, without reading its native file."""
        lane = info['lane']
        generated = list(info['request'].output_token_ids)
        info['committed'] = lane.frontend.commit(info['tokens'], generated)
        lane.turn += 1
        lane.messages = info['messages'] + [dict(role='assistant', content=' ' + info['committed']['text'] + '\n')]

    def test_a_cancel_admissions_and_the_shutdown(self):
        events, holder = [], {}

        def pause():
            writer = holder['writer']
            while not writer.urgent and not writer.stopping:
                time.sleep(0.0005)
        save, write = EV.save, D.CompletionWriter._write

        def saved(path, value):
            if path.parent == holder['output']:          # not the shadow writer's synchronous copies
                events.append(('sync', path.name))
            return save(path, value)

        def written(writer, job):
            write(writer, job)
            events.append(('writer', job.path.name))
        with patch.object(EV, 'save', saved), patch.object(D.CompletionWriter, '_write', written), \
                WindowRun('observation_query', deferred=True, writer=dict(pause=pause, slice_s=0.0)) as run:
            h, writer = run.h, run.writer
            holder.update(writer=writer, output=run.output)
            run.script(rounds=1, together=0, rebuild=False)        # each lane's first turn, read at once
            a, b, c = h.lanes
            admit = h.lifecycle.admit

            def admitting(body):
                events.append(('admit', body['session']))
                return admit(body)
            h.lifecycle.admit = admitting
            missing = lambda info: not (run.output / native_name(info['rid'])).exists()
            # A and B decode together; A completes: its files are released at post_step and stay in
            # progress on the writer.
            ia, ib = run.prefilled(a, GE.OBSERVATIONS[0], 5), run.prefilled(b, GE.OBSERVATIONS[1], 7)
            h.decode_together([ia, ib])
            events.clear()
            self.drained(h, ia)
            self.assertTrue(writer.hold(ia['rid']) and missing(ia))
            # C's admission does not wait for A's files.
            ic = run.prefilled(c, GE.OBSERVATIONS[2], 9)
            self.assertTrue(missing(ia))
            # B's client cancels: its terminal waits for every pending write, then is written here.
            ack = h.route.cancel(h.dispatcher, NS(command_id='x-' + ib['rid'], request_id=ib['rid']))
            self.assertEqual(ack.state, 'cancelled')
            b.frontend.abandon()
            self.assertEqual(events, [('admit', c.sid), ('writer', native_name(ia['rid'])),
                                      ('sync', native_name(ib['rid']))])
            self.assertEqual(h.lifecycle.sessions[b.sid][1], 'poisoned')
            self.commit(h, ia)
            # C completes; with its files in progress A's next admission runs at once and C's own
            # next admission waits for them.
            h.decode_together([ic])
            events.clear()
            self.drained(h, ic)
            self.commit(h, ic)
            ia2 = run.prefilled(a, GE.OBSERVATIONS[3], 4)
            self.assertTrue(missing(ic))
            ic2 = run.prefilled(c, GE.OBSERVATIONS[4], 6)
            self.assertEqual(events, [('admit', a.sid), ('writer', native_name(ic['rid'])), ('admit', c.sid)])
            # The shutdown with A's files in progress and C's request in flight: the writer closes
            # first (A's files), then the drain cancels C's request, whose terminal is synchronous.
            h.decode_together([ia2])
            events.clear()
            self.drained(h, ia2)
            self.assertTrue(missing(ia2))
            writer.close()
            h.lifecycle.disconnect()
            self.assertEqual(events, [('writer', native_name(ia2['rid'])), ('sync', native_name(ic2['rid']))])
            h.sessions.close()
            engine_closed(h.core)
            self.assertEqual((h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks(),
                              h.adapter.window.pinned_blocks()), (h.baseline, h.window_baseline, 0))
            states = {rid: json.loads((run.output / native_name(rid)).read_bytes())['terminal_state']
                      for rid in (ia['rid'], ib['rid'], ic['rid'], ia2['rid'], ic2['rid'])}
            self.assertEqual(list(states.values()), ['complete', 'cancelled', 'complete', 'complete', 'cancelled'])
            stats = writer.stats()
            self.assertEqual((writer.compared, writer.mismatches, stats['errors'], stats['pending']), (6, [], 0, 0))


# ---- 6. the API servers over the window engine, with held acknowledgements ------------------------------

class XmlWindowEngine(B2.BatchEngine, GCE.WindowBase):
    """test_ms2_e2e's engine loop over the window layout's engine and worker on the XML protocol:
    test_gemma_chat_text_e2e.GemmaEngine without the chat-text stops (E2E.Engine's MarkerStop stays)."""
    free = GCE.GemmaEngine.free


class XmlWindowServer(B2.BatchServer):
    """test_ms2_e2e.BatchServer over XmlWindowEngine (this module's, looked up at construction): the
    packaged router, the XML route and the frontend over the window layout's engine and worker."""

    def __init__(self, policy, tasks, output, router, Completion, layout, fa_blocks=GE.FA_BLOCKS):
        self.frames, self.send_holds = [], {}
        engine = functools.partial(THIS.XmlWindowEngine, batched=True)
        frontend = functools.partial(compose_frontend, max_num_seqs=len(tasks))
        with patch.object(GCE.WindowBase, 'slot_tasks', list(tasks)), patch.object(GCE.WindowBase, 'layout', layout), \
                patch.object(GCE.WindowBase, 'fa_blocks', fa_blocks), patch.object(E2E, 'Engine', engine), \
                patch.object(E2E, 'LLM', B2.BatchLLM), patch.object(E2E, 'compose_frontend', frontend):
            E2E.Server.__init__(self, policy, tasks, output, router, Completion, E2E.SERVER)
        self.seqs = len(tasks)
        self.loop = asyncio.create_task(self.engine.serve())


XML_TURNS = 4


class XmlScenarios:
    """Gemma's mswea-xml sessions through the packaged router over the window engine; each test keeps
    what its sessions were served in RECORDS (compared with the switch off and on)."""
    RECORDS = {}
    asyncSetUp = E2E.MultiSessionEndToEndTests.asyncSetUp
    boundary = GCE.GemmaChatTextEndToEndTests.boundary
    windows, window_blocks = GCE.GemmaChatTextEndToEndTests.windows, GCE.GemmaChatTextEndToEndTests.window_blocks

    def server(self, policy, tasks):
        output = Path(self.enterContext(tempfile.TemporaryDirectory()))
        layout = GE.window_layout(policy, len(tasks))
        value = GE.config(policy, output, list(tasks), layout)
        self.enterContext(patch('remote_ra_http.config.load', lambda: value))
        self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
        srv = XmlWindowServer(policy, tasks, output, self.router, self.Completion, layout)
        self.enterContext(patch.object(mswea_route, 'finish_response', srv.finish_response))
        self.addAsyncCleanup(srv.stop)
        return srv

    async def turn(self, srv, k, i):
        observation = None if i == 0 else GE.OBSERVATIONS[(i - 1 + k) % len(GE.OBSERVATIONS)]
        return await srv.turn(k, observation, target=GE.TARGETS[(i + k) % len(GE.TARGETS)], eos=(i + k) % 3 == 2,
                              chunk='page')

    def served(self, srv, info):
        """One completed turn: the reply as stock vLLM's bookkeeping has it, the runtime row and the native
        file agree (the prompt, the selection); what the session was served."""
        expected, _end, text = info['upstream']
        row = info['row']
        self.assertEqual(info['text'], text)
        self.assertEqual((row['prompt_token_ids'], info['native']['prompt_token_ids'], row['selected_logical_pages']),
                         (expected, expected, info['plan']['selected']))
        self.assertEqual(info['native']['terminal_state'], 'complete')
        return (info['lane'].index, row['turn'], text, row['selected_logical_pages'])

    def keep(self, value):
        self.RECORDS[self._testMethodName] = value

    async def together(self, policy):
        srv = self.server(policy, [A, B, C])
        h = srv.engine

        async def agent(k):
            return [self.served(srv, await self.turn(srv, k, i)) for i in range(XML_TURNS)]
        served = await asyncio.gather(*(agent(k) for k in range(3)))
        self.assertEqual(max(len(b.rows) for b in h.batches), 3)
        self.assertEqual(len(srv.rows()), 3 * XML_TURNS)
        self.assertFalse(srv.rows('failures.jsonl'))
        self.assertTrue(h.runner.windows and all(v > 0 for v in h.runner.windows.values()))
        self.boundary(srv)
        self.keep(served)

    async def test_three_sessions_decode_together_goal_query(self):
        await self.together('goal_query')

    async def test_three_sessions_decode_together_full(self):
        await self.together('full')

    async def test_cancel_of_one_session_between_batched_steps(self):
        srv = self.server('observation_query', [A, B])
        h = srv.engine
        a, b = srv.lanes
        served = [self.served(srv, await self.turn(srv, k, i)) for i in range(2) for k in range(2)]
        pause = h.park(lambda e: B2.decoding(e, a.sid, 2) and B2.decoding(e, b.sid, 2))
        ta, tb = asyncio.create_task(self.turn(srv, 0, 2)), asyncio.create_task(self.turn(srv, 1, 2))
        await asyncio.wait_for(pause.parked.wait(), 10)
        ra = B2.running(h, a.sid)
        srv.disconnect(0)
        await B2.settle(lambda: ('abort', a.sid, 2) in h.events, 10)
        pause.gate.set()
        with self.assertRaises(asyncio.CancelledError):
            await ta
        served.append(self.served(srv, await tb))
        self.assertEqual(h.native_file(ra)['terminal_state'], 'cancelled')
        self.assertEqual(h.lifecycle.sessions[a.sid], (2, 'poisoned'))
        served.append(self.served(srv, await self.turn(srv, 1, 3)))
        self.assertEqual([(f['session_index'], f['turn'], f['error']) for f in srv.rows('failures.jsonl')],
                         [(0, 2, 'CancelledError')])
        self.keep(served)

    async def test_shutdown_with_requests_in_flight(self):
        from entrypoint_runtime import ShutdownOwner
        srv = self.server('goal_query', [A, B, C])
        h = srv.engine
        a, b, c = srv.lanes
        served = [self.served(srv, await self.turn(srv, k, i)) for i in range(2) for k in range(3)]
        # A and B decode together; C's admission is accepted, its ack not yet at the frontend.
        pause = h.park(lambda e: B2.decoding(e, a.sid, 2) and B2.decoding(e, b.sid, 2))
        ta, tb = asyncio.create_task(self.turn(srv, 0, 2)), asyncio.create_task(self.turn(srv, 1, 2))
        await asyncio.wait_for(pause.parked.wait(), 10)
        ack = srv.hold_ack(2)
        tc = asyncio.create_task(self.turn(srv, 2, 2))
        await B2.settle(lambda: not h.input_queue.empty(), 10)
        pause.gate.set()
        await asyncio.wait_for(ack.parked.wait(), 10)
        owner = srv.llm._remote_ra_shutdown_owner = ShutdownOwner(srv.llm, srv.controller, lambda: True)
        await owner.close(10)
        for task in (ta, tb, tc):
            with self.assertRaises(asyncio.CancelledError):
                await task
        await asyncio.wait_for(srv.loop, 5)
        self.assertFalse(h.running or h.waiting or h.futures or h.enqueued)
        self.assertEqual((h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks(),
                          h.adapter.window.pinned_blocks()), (h.baseline, h.window_baseline, 0))
        self.assertTrue(h.sessions.closed and h.lifecycle.closed)
        last = [r for r, i in h.infos.items() if i['turn'] == 2]
        self.assertEqual([json.loads((srv.output / native_name(r)).read_bytes())['terminal_state'] for r in last],
                         ['cancelled'] * 3)
        cleanup = json.loads((srv.output / 'engine-cleanup.json').read_bytes())
        self.assertEqual(srv.llm.cleanup_receipt, sha256((srv.output / 'engine-cleanup.json').read_bytes()).hexdigest())
        self.assertEqual(len(cleanup['admitted_request_ids']), 9)
        self.assertEqual(len(srv.rows()), 6)
        self.assertEqual(sorted((f['session_index'], f['turn'], f['error']) for f in srv.rows('failures.jsonl')),
                         [(k, 2, 'CancelledError') for k in range(3)])
        self.keep(served)


def xml_case():
    """XmlScenarios as a TestCase (built here, so that discovery does not run it on its own)."""
    return type('XmlWindowServerScenarios', (XmlScenarios, unittest.IsolatedAsyncioTestCase), {'maxDiff': None})


def exchanges(server, method, run):
    """Every HTTP exchange that ``run()`` makes through ``server.method`` (test_deferred_completion.ServerTests.
    replies): {session: its exchanges}, each session keyed by its first request and its uuids numbered in
    order of appearance, without ids, creation times and wall-clock waits."""
    token = re.compile(r'[0-9a-f]{32}|call_[0-9a-f]{12}')
    records = []
    original = getattr(server, method)

    async def recorded(srv, *args, **kwargs):
        result = await original(srv, *args, **kwargs)
        records.append((args, {k: v for k, v in kwargs.items() if k != 'receive'}, result))
        return result
    with patch.object(server, method, recorded):
        run()

    def plain(value):
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()
                    if k not in ('id', 'created') and not k.endswith(('_s', '_ms', '_ns'))}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value
    sessions = {}
    for args, kwargs, (status, reply) in records:
        if method == 'send':
            sid = (args[1] if len(args) > 1 and type(args[1]) is dict else {}).get('session_id')
        else:
            sid = args[1] if len(args) > 1 else kwargs.get('header')
        key = sid if sid is not None else json.dumps(plain(args[0][:2]), sort_keys=True)
        sessions.setdefault(key, []).append(json.dumps([plain([args, kwargs]), status, plain(reply)], sort_keys=True))
    out = {}
    for items in sessions.values():
        seen = {}
        joined = token.sub(lambda m: seen.setdefault(m.group(0), '<u%d>' % len(seen)), '\n'.join(items))
        out[joined.split('\n', 1)[0]] = joined
    return out


_KV = {}


def kv_window_engine():
    """test_kv_session_e2e's KvEngine over the window layout's engine and worker (test_gemma_e2e's fakes,
    no MRoPE rows): test_gemma_kv_session_e2e's GemmaKvServer builds its engine on it once it stands for
    KvEngine. Built when first needed (test_kv_session_e2e's imports)."""
    if not _KV:
        import test_kv_session_e2e as KE

        class KvWindowEngine(KE.KvEngine, GCE.WindowBase):
            free = GCE.GemmaEngine.free

            def __init__(self, *args, mrope=False, **kwargs):
                super().__init__(*args, mrope=False, **kwargs)
        _KV['engine'] = KvWindowEngine
    return _KV['engine']


def window_route_server(test, policy, slots, limit, tokenizer):
    """test_gemma_kv_session_e2e.route_server over the window layout (``limit`` open sessions, pool 0 that
    never releases): Gemma's kv-session record (test_gemma_kv_session_e2e.config's, with test_gemma_e2e's
    contract: no fixed budget, as the route tests' clients send) on the window layout's startup config."""
    K = GKS.K
    output = Path(test.enterContext(tempfile.TemporaryDirectory()))
    layout = GE.window_layout(policy, limit)
    tasks = K.slot_tasks(slots)
    value = GE.config(policy, output, tasks, layout)
    value.update(protocol=K.PROTOCOL, task=tasks[0], model_profile=MP.GEMMA4, max_running_sessions=limit,
                 kv_session=K.record(contract=value['selection_contract'], thinking='off', slots=slots, grammar=K.GEMMA4))
    test.enterContext(patch('remote_ra_http.config.load', lambda: value))
    test.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
    with patch.object(GCE.WindowBase, 'slot_tasks', tasks), \
            patch.object(GCE.WindowBase, 'layout', layout), patch.object(GCE.WindowBase, 'fa_blocks', GE.FA_BLOCKS):
        srv = GKS.GemmaKvServer(policy, value, output, test.router, test.Completion, tokenizer)
    test.assertIsNotNone(srv.engine.adapter.window)            # the window layout's engine and worker
    test.assertEqual(type(srv.engine.owner).__name__, 'WindowAllocationOwner')
    test.addAsyncCleanup(srv.stop)
    return srv


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ServerTests(unittest.TestCase):
    """The API servers over the window engine with the switch (test_deferred_completion.ServerTests'
    engine stand-in: post_step, the acknowledgement seam, the lifecycle tick, the writer closed first at
    shutdown; a shadow writer): every check of their scenarios holds, every file the writer wrote equals
    the synchronous path's of the same moment, and the replies are those of the switch off."""
    maxDiff = None
    check = TD.ServerTests.check

    def same(self, off, on, sessions):
        self.assertGreaterEqual(len(off), sessions)
        self.assertEqual(sorted(on), sorted(off))
        for key in off:
            self.assertEqual(on[key], off[key])

    def outcomes(self, outcomes):
        for name, *outcome in outcomes:
            self.assertEqual(outcome, [1, [], [], []], name)

    # ---- chat-text (test_gemma_chat_text_e2e's server) ------------------------------------------------
    def test_chat_text_with_a_slow_writer(self):
        names = ['test_sixteen_sessions_decode_together', 'test_cancel_retry_rebinds_and_frees_the_window',
                 'test_header_aside_handoff_and_close']
        if SLOW:
            names += ['test_sixteen_sessions_release_idle_pages', 'test_context_limit_length_and_refusals',
                      'test_twenty_open_sessions_eviction_and_window_reuse',
                      'test_a_rebuild_with_every_window_slot_taken_releases_idle_pages',
                      'test_a_lone_turn_waits_for_room_instead_of_running_out']
        self.check(*TD.run_with_switch(GCE, GCE.GemmaChatTextEndToEndTests, names, writer=SLOW_WRITER,
                                       patch_name='GemmaEngine'), held=True)

    def test_chat_text_replies_equal_with_the_switch_off_and_on(self):
        cls, names = GCE.GemmaChatTextEndToEndTests, ['test_sixteen_sessions_release_idle_pages']
        off = exchanges(GCE.GemmaChatServer, 'chat_request', lambda: self.outcomes(IF.run_tests(cls, names)))
        on = exchanges(GCE.GemmaChatServer, 'chat_request', lambda: self.check(*TD.run_with_switch(
            GCE, cls, names, patch_name='GemmaEngine')))
        self.same(off, on, 16)

    # ---- mswea-xml (this module's server over the window engine) --------------------------------------
    def xml(self, names, writer=None):
        case = xml_case()
        case.RECORDS = {}
        self.outcomes(IF.run_tests(case, names))
        off = dict(case.RECORDS)
        case.RECORDS = {}
        compared = self.check(*TD.run_with_switch(THIS, case, names, writer=writer, patch_name='XmlWindowEngine'),
                              held=writer is not None)
        self.assertEqual(sorted(off), sorted(names))
        self.assertEqual(case.RECORDS, off)
        return compared

    def test_mswea_xml_with_a_slow_writer(self):
        names = ['test_three_sessions_decode_together_goal_query', 'test_cancel_of_one_session_between_batched_steps',
                 'test_shutdown_with_requests_in_flight']
        if SLOW:
            names += ['test_three_sessions_decode_together_full']
        self.xml(names, SLOW_WRITER)

    @unittest.skipUnless(SLOW, 'about 15 s more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_mswea_xml(self):
        self.xml(['test_three_sessions_decode_together_goal_query', 'test_three_sessions_decode_together_full',
                  'test_cancel_of_one_session_between_batched_steps', 'test_shutdown_with_requests_in_flight'])

    # ---- kv-session (test_gemma_kv_session_e2e's route tests) ----------------------------------------
    def test_kv_session_route_on_the_window_layout(self):
        """GemmaRouteTests' MemAgent sessions through the route (closes included) over the window engine."""
        import test_kv_session_e2e as KE
        engine = kv_window_engine()
        cls, names = GKS.GemmaRouteTests, ['test_memagent_sessions_through_the_route']
        with patch.object(GKS, 'route_server', window_route_server):
            with patch.object(KE, 'KvEngine', engine):
                off = exchanges(KE.KvServer, 'send', lambda: self.outcomes(IF.run_tests(cls, names)))
            on = exchanges(KE.KvServer, 'send', lambda: self.check(*TD.run_with_switch(
                KE, cls, names, writer=SLOW_WRITER, patch_name='KvEngine', base=engine), held=True))
        self.same(off, on, 3)

    def test_kv_session_route_with_gemma_grammar(self):
        """GemmaRouteTests as they are (test_kv_session_e2e's engine with Gemma's transcripts)."""
        import test_kv_session_e2e as KE
        names = ['test_memagent_sessions_through_the_route'] + (['test_refusals_through_the_route'] if SLOW else [])
        self.check(*TD.run_with_switch(KE, GKS.GemmaRouteTests, names, writer=SLOW_WRITER, patch_name='KvEngine'),
                   held=True)


if __name__ == '__main__':
    unittest.main()
