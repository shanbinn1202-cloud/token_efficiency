"""Gemma 4 31B IT, goal_query on the chat-text protocol (gemma-tb-goal). CPU only.

goal_query's Goal inputs on chat-text (remote_ra_live_v2.terminus2_goal, PROVISIONAL: pending
the Goal owner's review) are cut out of the request's message text: the task instruction of
Terminus 2's pinned prompt template, the newest message's terminal output, the summary text
after a hand-off, and the summarization sub-calls' rules. None of that depends on the model's
turns, so the Gemma profile takes Qwen's rules, record and Goal construction unchanged; only
the Goal forward is Gemma's (``<bos>`` and the ``<retrieval_query>`` text, in the window
layout's Goal transient, gemma-goal). Covered:

- the launcher and the startup checks at docs/reference.md 4.14's chat-text example (K = 20,
  65,536 tokens, 4,096-token replies, 28 GiB, P = 16, CUDA graphs full, --decode-fastpath,
  --release-idle-pages, an 8,192-token budget) with goal_query, its Goal rewriter and
  --admission-overlap 16: the dry run prints ``chat_text_goal`` (terminus2_goal.record(), the
  record a Qwen server pins); the config pins it next to the Goal transient and the selective
  rule's request_pages and differs from observation_query's only in goal_query's own keys;
  remote_ra_http.config.load, the frontend's argument check, the chat-text frontend and its
  options take it. The refusals that stay: --phase1-in-step, --gdn-device-handoff, a --task
  list (--goal-rewriter-batch and --decode-fastpath3 are served since gemma-rewriter-batch and
  gemma-fastpath3, integrate-1021); the startup check refuses a
  goal_query config without the record or with another one, and the record on another policy;
- the memory rule at 28 GiB with 16 sessions at 131,072 tokens and with docs/reference.md 4.14's chat-text
  example, K = 20 at 65,536: pool 0 under the selective rule, its figure pinned as request_pages, which
  vLLM's own startup estimate then asks of pool 0 (one request at the context limit would
  not fit next to pool 1 in either case);
- end to end (test_gemma_chat_text_e2e's packaged router, chat-text frontend and window
  engine and worker fakes): Terminus 2 clients with 2.0.0's prompts (odd ones send
  X-Session-ID) against a goal_query server with K = 20, --admission-overlap 16 (the overlap
  park table, the pinned lease and the ledger as compose_engine attaches them, K permits in
  the frontend), --release-idle-pages under pool pressure and --decode-fastpath, a Goal
  rewriter stand-in on the Goal threads. Every client summarizes (the three sub-calls, an
  aside, the hand-off rebuild), and one client's request is cancelled in flight and sent
  again (its key rebinds to a fresh slot, where its Goal starts again). Every prompt is the
  template's render (and stock vLLM's without the thought channels); every turn's Goal
  inputs and Goal source are checked against what the client sent (runtime.jsonl
  ``chat_text.goal_inputs``, goals.jsonl ``inputs``, the rewriter's requests); every Goal
  forward fits the Goal transient; admissions overlapped; idle sessions were released and
  readmitted, the windows never. Every client's requests and replies, and every client
  session's prompts, selections, attended Pages, Goal inputs, Goal decisions, query texts
  and rewriter requests equal those of a server with none of these flags (one admission at
  a time, a pool that never releases). The default suite runs four clients of six episodes
  (both servers, about 45 s); REMOTE_RA_SLOW_TESTS=1 runs sixteen (Harbor's -n 16 against
  K = 20; their asides outgrow the K open sessions, so the pool closes idle sessions at times
  that differ between the two servers, and the client sessions rebound in either run are
  left out of the comparison, as in test_integrate_flags);
- integrate-1021 (gemma-tb-goal with gemma-rewriter-batch and gemma-fastpath3): the launch at
  25 GiB with K = 17, --goal-rewriter-batch 4 (the token cap's 25-GiB default, 25,600),
  --release-idle-pages and each decode fast path, and the clients above with the batched
  in-process rewriter child against the same child unbatched (BatchedRewriterTests,
  BatchedEndToEndTests), also with --deferred-completion-io (gemma-deferred-io: the completion
  files on test_deferred_completion's writer, each equal to the synchronous path's bytes);
- with the pinned tokenizer and template (REMOTE_RA_GEMMA4_METADATA and Transformers): a
  Terminus 2 session through Gemma4SourceRenderer.chat_transcripts, each prompt the template's
  render, each turn's Goal built from its Goal inputs as the engine builds it, its Goal probe
  ``<bos>`` and the text's tokens within the Goal transient.
"""
import asyncio
from contextlib import redirect_stdout
import dataclasses
import functools
from hashlib import sha256
import io
import json
import os
from pathlib import Path
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

import gemma4_testing as G              # noqa: E402
import serve                            # noqa: E402
import test_chat_text_e2e as CE         # noqa: E402  the Goal sessions' outputs, the rewriter stand-in
import test_gemma_chat_text as GT       # noqa: E402  the Gemma release fixture, the kits
import test_gemma_chat_text_e2e as GCE  # noqa: E402  the Gemma chat-text server on the window fakes
import test_deferred_completion as TD   # noqa: E402  the completion writer on the engine fakes
import test_gemma_release as GR         # noqa: E402  the fixed-budget contract
import test_goal_rewriter_batch as GRB   # noqa: E402  the in-process batched rewriter child (imported
                                         # here: a test's patch.dict(sys.modules) drops what it imports)
import test_ms2_e2e as B2               # noqa: E402
import test_stageb_e2e as S             # noqa: E402  the engine renderer seam
import test_terminus2_goal as TG        # noqa: E402  Terminus 2.0.0's exact prompts

from remote_ra_http import config as HC  # noqa: E402
from remote_ra_live_v2 import admission_flags as AF  # noqa: E402
from remote_ra_live_v2 import admission_overlap as AO  # noqa: E402
from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from remote_ra_live_v2 import goal_async as GA  # noqa: E402
from remote_ra_live_v2 import idle_release as R  # noqa: E402
from remote_ra_live_v2 import kv_layout as I0  # noqa: E402
from remote_ra_live_v2 import model_profile as MP  # noqa: E402
from remote_ra_live_v2 import query_selection  # noqa: E402
from remote_ra_live_v2 import terminus2_goal as T2  # noqa: E402
from remote_ra_live_v2 import window_capacity as WC  # noqa: E402
from remote_ra_live_v2.composition import compose_frontend  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
GiB = 1 << 30


# ---- the launcher and the startup checks ----------------------------------------------------------

def tb_goal_argv(model):
    """serve.py's arguments of a Gemma chat-text goal_query server: docs/reference.md 4.14's chat-text
    example (test_gemma_chat_text.TB) with goal_query's own: the 8,192-token budget, the Goal
    rewriter ``model`` and --admission-overlap 16."""
    return ['--protocol', 'chat-text', '--policy', 'goal_query', '--chat-text-slots', '512',
            '--chat-text-idle-timeout', '600', '--chat-text-evict-after', '30', '--kv-cache-gib', '28',
            '--page-tokens', '16', '--cuda-graphs', 'full', '--max-running-sessions', '20', '--max-tokens', '4096',
            '--max-model-len', '65536', '--dram-arena-gib', '64', '--connector-pool-gib', '1', '--decode-fastpath',
            '--language-model-only', '--release-idle-pages', '--active-token-budget', '8192',
            '--goal-rewriter-model', str(model), '--admission-overlap', '16']


def window(sessions, gib, goal=MP.GEMMA4_GOAL_TOKENS, budget=8192, max_tokens=4096):
    """The window layout serve.py pins under the selective rule (16-token Pages): the Goal
    transient and the rule's request_pages."""
    base = I0.for_window_server(16, sessions, MP.GEMMA4_GEOMETRY.sliding_window, MP.GEMMA4_FORWARD_TOKENS, goal)
    return dataclasses.replace(base, request_pages=WC.pinned_request_pages(base, budget, max_tokens, gib * GiB))


class LauncherTests(unittest.TestCase):
    """serve.prepare on a Gemma release fixture (test_gemma_chat_text.LauncherTests')."""
    maxDiff = None
    setUp, load = GT.LauncherTests.setUp, GT.LauncherTests.load

    def prepare(self, policy='goal_query', selection=None, **kw):
        if selection is None:
            selection = dict(GT.BUDGET, goal_rewriter_model=str(self.rewriter()))
        return GT.LauncherTests.prepare(self, policy, selection, **kw)

    def rewriter(self):
        model = self.root / 'qwen3-0.6b'
        if not model.exists():
            model.mkdir()
            (model / 'config.json').write_text('{"model_type": "qwen3"}')
        return model

    def test_the_terminal_bench_goal_query_server(self):
        from remote_ra_http import chat_text_route as CR
        from remote_ra_live_v2 import cuda_graphs as CG
        text = io.StringIO()
        with patch.object(sys, 'argv', ['serve.py', '--no-speed-defaults', '--release-dir', str(self.release), '--output',
                                        str(self.root / 'tb-goal'), '--port', '8000', *tb_goal_argv(self.rewriter())]), \
                redirect_stdout(text):
            serve.main()
        dry = json.loads(text.getvalue())
        self.assertEqual((dry['status'], dry['chat_text_goal'], dry['goal_rewriter']['used']),
                         ('SERVICE_DRY_RUN', T2.record(), True))
        _, config, command, env = self.prepare(admission_overlap=16)
        out = str(self.root / ('out%d' % self.n))
        self.assertEqual([a.replace(out, str(self.root / 'tb-goal')) for a in command], dry['command'])
        self.assertEqual(env['REMOTE_RA_HTTP_CONFIG_SHA256'], serve.pin(config))
        self.assertEqual((config['protocol'], config['model_profile'], config['policy'], config['max_running_sessions']),
                         (C.PROTOCOL, MP.GEMMA4, 'goal_query', 20))
        self.assertEqual((config['chat_text'], config['chat_text_goal']), (C.record(profile=MP.GEMMA4), T2.record()))
        self.assertEqual((config['admission'], config['goal_rewriter']['used'], config['release_idle_pages']),
                         (AF.record(admission_overlap=16), True, True))
        self.assertNotIn('goal_rewriter_batch', config)
        layout = window(20, 28)
        self.assertEqual(layout.goal_tokens, 2048)
        self.assertEqual(config['kv_layout'], I0.to_additional_config(layout)[I0.LAYOUT_KEY])
        argv = command[command.index('--model'):]
        self.assertEqual(json.loads(argv[argv.index('--additional-config') + 1]), I0.to_additional_config(layout))
        self.assertEqual(json.loads(argv[argv.index('--compilation-config') + 1]), CG.compilation_config('full'))
        # Startup: the checks accept it; the frontend classifies Terminus 2's Goal inputs.
        loaded = self.load(config)
        self.assertEqual(loaded, config)
        self.assertEqual((HC.model_profile(loaded), HC.chat_text(loaded), HC.kv_layout(loaded)),
                         (MP.GEMMA4, C.record(profile=MP.GEMMA4), layout))
        self.assertEqual(T2.check_config(loaded), T2.record())
        options = C.options(loaded)
        self.assertEqual((options['goal_inputs'], options['model_profile'], options['header_text']),
                         (T2.record(), MP.GEMMA4, C.GEMMA4_HEADER_TEXT))
        self.assertTrue(CR.frontend(loaded).goal_inputs)
        effective = dict(config['effective_server_config'])
        effective['compilation_config'] = CG.observed(effective['compilation_config'])
        env = dict(REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(self.root / ('startup%d.json' % self.n)),
                   REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config))
        with patch.dict(os.environ, env):
            self.assertEqual(HC.effective_server_config(NS(**effective)), config['effective_server_config'])
        # observation_query with the same flags (docs/reference.md 4.14's chat-text example): the configs
        # differ only in goal_query's own keys, and vLLM's arguments only in the window layout (the
        # Goal transient, and the selective figure that pool 0 holds next to it).
        _, plain, plain_command, _ = self.prepare('observation_query', GT.BUDGET)
        own = {'policy', 'goal_rewriter', 'chat_text_goal', 'admission', 'output_directory', 'selection_contract',
               'kv_layout', 'effective_server_config', 'server_config_sha256'}
        self.assertEqual({k for k in set(config) | set(plain) if config.get(k) != plain.get(k)} - own, set())
        self.assertEqual(set(config) - set(plain), {'chat_text_goal', 'admission'})
        self.assertEqual(plain['kv_layout'], I0.to_additional_config(window(20, 28, goal=0))[I0.LAYOUT_KEY])
        self.assertEqual({k: v for k, v in config['selection_contract'].items() if k != 'goal_rewriter'},
                         {k: v for k, v in plain['selection_contract'].items() if k != 'goal_rewriter'})
        self.assertEqual(len(command), len(plain_command))
        differ = [k for k, (a, b) in enumerate(zip(command, plain_command)) if a != b]
        self.assertEqual({command[k - 1] for k in differ}, {'--additional-config', '--chat-template'})

    def test_goal_query_without_a_rewriter_and_the_refusals_that_stay(self):
        _, config, _, _ = self.prepare(selection=dict(GT.BUDGET, no_goal_rewriter=True))
        self.assertEqual((config['chat_text_goal'], config['goal_rewriter']['used']), (T2.record(), False))
        self.load(config)
        rewriter = dict(GT.BUDGET, goal_rewriter_model=str(self.rewriter()))
        for change, pattern in (
                (dict(selection=GT.BUDGET), 'goal_query requires --goal-rewriter-model'),
                # --goal-rewriter-batch is served since gemma-rewriter-batch (integrate-1021:
                # test_gemma_chat_text_goal.BatchedRewriterTests).
                # --decode-fastpath3 is served since gemma-fastpath3 (integrate-1021).
                (dict(phase1_in_step=True, admission_overlap=2), '--phase1-in-step is not available'),
                (dict(gdn_device_handoff=True), 'no Gated-DeltaNet state'),
                (dict(admission_overlap=33), 'must be an integer 1-32'),
                (dict(max_running=None), 'needs --max-running-sessions K')):
            change = dict(change)
            selection = change.pop('selection', rewriter)
            with self.subTest(change=sorted(change)), self.assertRaises(Exception) as caught:
                self.prepare('goal_query', selection, **change)
            self.assertIn(pattern, str(caught.exception))
        with self.assertRaisesRegex(ValueError, 'takes no --task'):
            serve.prepare(self.release, [dict(instance_id='a', problem_statement='b')], 'goal_query',
                          self.root / 'tasked', 8000, rewriter, **GT.TB)
        # Startup: goal_query needs exactly this release's record; no other policy takes one.
        _, other, _, _ = self.prepare('observation_query', GT.BUDGET)
        for value, pattern in (({k: v for k, v in config.items() if k != 'chat_text_goal'}, 'chat_text_goal'),
                               (dict(config, chat_text_goal=dict(T2.record(), status='confirmed')), 'chat_text_goal'),
                               (dict(other, chat_text_goal=T2.record()), 'applies to goal_query only')):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(RuntimeError, pattern):
                self.load(value)

    def test_memory_rule_at_28_gib(self):
        """16 sessions at 131,072 tokens and docs/reference.md 4.14's chat-text example, K = 20 at 65,536, both at
        --kv-cache-gib 28 with 4,096-token replies and an 8,192-token budget: pool 0 under the
        selective rule, its figure pinned for vLLM's own startup estimate."""
        import test_gemma_kv_pools as GKP
        block, page = MP.GEMMA4_GEOMETRY.sliding_block_bytes(16), MP.GEMMA4_GEOMETRY.page_bytes(16)
        rows = {}
        for sessions, max_len, arena in ((16, 131072, 79), (20, 65536, 64)):
            _, config, command, _ = self.prepare(max_running=sessions, max_model_len=max_len, dram_arena_gib=arena,
                                                 admission_overlap=16)
            layout = window(sessions, 28)
            self.assertEqual(HC.kv_layout(self.load(config)), layout)
            self.assertEqual(command[command.index('--max-model-len') + 1], str(max_len))
            blocks = WC.pool_zero_blocks(28 * GiB, layout, 5, block, page)
            need = WC.selective_pages(layout, 8192, 4096, blocks)
            self.assertEqual(layout.request_pages, need)
            rows[sessions] = (I0.window_pool_blocks(layout, 5), blocks, need)
            # vLLM's own startup estimate (the overlay's kv_cache_utils): pool 1 and request_pages
            # global Pages for one request, within 28 GiB; one request at the context limit instead
            # (the strict figure, no request_pages) would not fit next to pool 1.
            w = GKP.gemma_world()
            vc = GKP.vconfig(layout, max_len=max_len)
            groups = w.utils.get_kv_cache_groups(vc, GKP.gemma_specs(w, 16))
            pool1 = I0.window_pool_blocks(layout, 5) * block
            self.assertEqual(w.utils._max_memory_usage_bytes_from_groups(vc, groups), need * page + pool1)
            _, kv = GKP.plan(w, layout, 28 * GiB, max_len=max_len)
            self.assertEqual(kv.num_blocks_by_pool, ((28 * GiB - pool1) // page, I0.window_pool_blocks(layout, 5)))
            self.assertGreater(kv.num_blocks_by_pool[0] - 1, need)
            self.assertGreater(max_len // 16 * page + pool1, 28 * GiB)
            with self.assertRaisesRegex(ValueError, 'max seq len'):
                GKP.plan(w, dataclasses.replace(layout, request_pages=0), 28 * GiB, max_len=max_len)
        # 16 sessions: pool 1 of 8,401 block ids (640 of them the Goal transient, 1,600 MiB) and
        # 6,135 global Pages against the 2,639 the selective rule needs (test_gemma_goal's figures
        # for the XML protocol: the layout is the same). K = 20: 9,701 block ids and 3,535 Pages
        # against 2,617 (a margin of 917 Pages for runs longer than one forward).
        self.assertEqual((rows[16], rows[20]), ((8401, 6135, 2639), (9701, 3535, 2617)))
        common = dict(page_tokens=16, window_tokens=1024, forward_tokens=8192, groups=5, block_bytes=block,
                      page_bytes=page, max_tokens=4096, active_tokens=8192, goal_tokens=MP.GEMMA4_GOAL_TOKENS)
        self.assertEqual(WC.max_sessions(28 * GiB, max_model_len=131072, **common), 21)


# ---- end to end on the window fakes -------------------------------------------------------------

class Rewriter(CE.Rewriter):
    """test_chat_text_e2e's rewriter stand-in (the upstream DeterministicFakeRewriter, every request
    recorded), serialised like the persistent rewriter and a little slow, so that other sessions'
    admissions proceed while a Goal is parked."""

    def __init__(self, delay=0.01):
        super().__init__()
        self.delay, self.serial = delay, threading.Lock()

    def rewrite(self, request):
        with self.serial:
            time.sleep(self.delay)
            return super().rewrite(request)


def goal_config(output, slots, limit, release, overlap, pin=None, batch=None, tokens=None, deferred=False):
    """A Gemma goal_query chat-text startup config on the test geometry (test_gemma_chat_text_e2e's):
    test_gemma_release's fixed-budget contract, the provisional Goal-input record and, with
    ``overlap``, the admission record; ``pin`` a Goal rewriter pin in the contract (the selection
    does not read it), ``batch`` --goal-rewriter-batch N with its token cap ``tokens``, ``deferred``
    the --deferred-completion-io pin."""
    value, layout = GCE.config('goal_query', output, slots, limit, release)
    value['selection_contract'] = GR.contract('goal_query')
    if pin is not None:
        value['selection_contract'] = dict(value['selection_contract'], goal_rewriter=pin)
    value['chat_text_goal'] = T2.record()
    if overlap is not None:
        value['admission'] = AF.record(admission_overlap=overlap)
    if batch is not None:
        value.update(goal_rewriter_batch=batch, goal_rewriter_batch_tokens=tokens)
    if deferred:
        value[TD.KEY] = True
    return value, layout


class ChildCalls:
    """The in-process rewriter child's calls (test_goal_rewriter_batch.InProcess: one line per call
    or per batch) as check_goals counts the stand-in's requests."""

    def __init__(self, child):
        self.child = child

    @property
    def requests(self):
        return [None] * sum(self.child.sizes)


def goal_run(count, episodes, blocks, *, limit=20, overlap=None, release=False, fast=False, retry=None,
             context_limit=4600, child=False, batch=None, tokens=None, deferred=False):
    """``count`` Terminus 2 clients (test_chat_text_e2e.goal_sessions' clients) playing ``episodes``
    each against one Gemma goal_query chat-text server of ``limit`` open sessions whose pool 0 has
    ``blocks`` blocks, with ``overlap`` --admission-overlap K, ``release`` --release-idle-pages and
    ``fast`` --decode-fastpath; ``retry`` (client, episode): that request is cancelled in flight
    (its slot fails) and sent again. Returns (run, failures, errors, skipped) and, per client
    session (its initial segment and header id), each request's kind, rebuild, binding, prompt,
    selection, attended Pages, Goal inputs (frontend and engine), decision, query hash and
    rewriter requests; each client's requests and replies; the release events; the engine.

    ``fast``: True or 1 --decode-fastpath, 2 with --decode-fastpath2, 3 with --decode-fastpath3
    (integrate-1021). ``child``: the Goal rewriter is test_goal_rewriter_batch's in-process child
    (the upstream instance and the fake batched generator, on which batched and one-at-a-time
    decoding agree token for token) instead of the DeterministicFakeRewriter stand-in; ``batch``
    --goal-rewriter-batch N within the token cap ``tokens`` (that child, batched; the first call's
    answer waits until N more calls queue, so that the next line is a batch). ``deferred``
    --deferred-completion-io (gemma-deferred-io): the window engine with test_deferred_completion's
    writer attached as compose_engine attaches it, a shadow writer comparing every file it writes
    with the synchronous path's bytes; ``writers`` in the result."""
    found = {}

    class Case(GCE.GemmaChatTextEndToEndTests):
        TRACE = CE.ChatTextEndToEndTests.TRACE
        goal_output = CE.ChatTextEndToEndTests.goal_output

        async def scenario(self):
            pin, children = None, []
            if child or batch is not None:
                from remote_ra_live_v2 import selection_contract
                model = Path(self.enterContext(tempfile.TemporaryDirectory())) / 'qwen3-0.6b'
                model.mkdir()
                (model / 'config.json').write_text('{}')
                pin = selection_contract.rewriter_pin(model)

                def start(pin_, batch=None, tokens=0):
                    inner, process = GRB.unbatched() if batch is None else GRB.batched(batch, tokens)
                    children.append((inner, process))
                    return inner
                self.enterContext(patch.object(query_selection, 'goal_rewriter', start))
                self.addCleanup(lambda: [inner.close() for inner, _ in children])
            else:
                rewriter = Rewriter()
                self.enterContext(patch.object(query_selection, 'goal_rewriter', lambda pin, batch=None: rewriter))
            probes, plain = [], S.Renderer.__call__

            def probe(renderer, source):
                """The engine renderer's Goal probe (PLAIN_GOAL_SCHEMA), its length recorded."""
                out = plain(renderer, source)
                probes.append(len(out.token_ids))
                return out
            self.enterContext(patch.object(GCE.ChatRenderer, '__call__', probe))
            if overlap is not None:
                # compose_engine's and compose_frontend's attachments of --admission-overlap K.
                self.enterContext(patch.object(GA, 'attach', functools.partial(GA.attach, overlap=overlap)))
                self.enterContext(patch.object(GCE, 'compose_frontend',
                                               functools.partial(compose_frontend, admission_overlap=overlap)))
            output = Path(self.enterContext(tempfile.TemporaryDirectory()))
            value, layout = goal_config(output, count * (episodes + 1) * 3, limit, release, overlap, pin=pin,
                                        batch=batch, tokens=tokens, deferred=deferred)
            self.enterContext(patch('remote_ra_http.config.load', lambda: value))
            self.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
            writers = []
            if deferred:
                shadow = Path(self.enterContext(tempfile.TemporaryDirectory()))
                self.enterContext(patch.object(GCE, 'GemmaEngine',
                                               TD.batch_engine(GCE.GemmaEngine, writers, shadow_root=shadow)))
            srv = GCE.GemmaChatServer('goal_query', value, layout, output, self.router, self.Completion,
                                      fa_blocks=blocks, release=release, goals=True)
            self.addAsyncCleanup(srv.stop)
            h = srv.engine
            self.assertEqual(writers, [h] if deferred else [])
            goals = h._remote_ra_goal_admissions

            def close():
                for executor in list(getattr(goals, '_threads', {}).values()) + [goals._thread]:
                    executor.shutdown(wait=True, cancel_futures=True)
            self.addCleanup(close)
            if overlap is not None:
                self.assertIsInstance(goals, GA.OverlapGoalAdmissions)
                h.lifecycle.lease = AF.timeouts(overlap)[1]
                AO.attach_engine(h.core, h.lifecycle, AF.record(admission_overlap=overlap))
                # compose_frontend pinned the ACK wait from K; the fake engine client keeps vLLM's
                # former 300-s bound (test_integrate_flags.chat_goal_sessions).
                client = srv.llm.engine_core
                add = client.registered_add_async
                client.registered_add_async = lambda frame, timeout: add(frame, min(timeout, 300))
            if fast:
                import test_gemma_fastpath as TF
                import test_gemma_fastpath3 as TF3
                {1: TF.attach_fast, 2: TF.attach_fast2, 3: TF3.attach_fast3}[int(fast)](h)
            opened = None
            if pin is not None:
                (inner, process), = children
                rewriter = ChildCalls(process)
                if batch is not None:
                    process.gate.clear()
                    # The first call and two queued behind it: the next line is a batch of two (on the
                    # window fakes three clients' first turns are in flight at once).
                    want = min(count, batch + 1, 3)

                    async def opening():
                        try:
                            await B2.settle(lambda: sum(process.sizes) + len(inner._remote_ra_batch._queue) >= want,
                                            timeout=60)
                        except AssertionError:
                            seen = sum(process.sizes) + len(inner._remote_ra_batch._queue)
                            raise AssertionError('only %d of %d first calls reached the batcher' % (seen, want))
                        finally:
                            process.gate.set()
                    opened = asyncio.ensure_future(opening())
            peak = [0]
            park = goals._park

            def parked(steps, work, finish):
                park(steps, work, finish)
                peak[0] = max(peak[0], len(goals.parked))
            goals._park = parked
            srv.chat.autostart, srv.chat.tick_s = True, 0.02
            srv.chat.record.update(evict_after_s=0.05)
            count_chars = lambda messages: sum(len(m['content']) for m in messages)
            agents = []
            for k in range(count):
                instruction = 'Task %d: build /app/job%d.py and make ./run %d print ok.' % (k, k, k)
                agent = TG.Terminus2(instruction, asend=srv.send, count=count_chars, context_limit=context_limit,
                                     threshold=600, header='t%02d__agent' % k if k % 2 else None)
                agent.first, agent.outputs = TG.first_prompt(instruction, 'root@h%04d:/app# \n' % k), []
                agents.append(agent)
            retried = {}

            async def cancelled(k, prompt):
                """Client k's next request (no summary due: what its step sends next) is cancelled
                in flight, so its slot fails; the step then sends it again (test_gemma_chat_text_e2e's
                cancel and rebind)."""
                agent = agents[k]
                self.assertGreater(context_limit - count_chars(agent.chat), 600)
                messages = agent.chat + [TG.user(prompt)]
                client = next(c for key, c in srv.chat.registry.sessions.items()
                              if key[0] == agent.header and key[1][0][1] == agent.first)
                slot = client.slot
                hold = srv.hold(slot, 2)
                receive = asyncio.Queue()
                task = asyncio.create_task(srv.chat_request(messages, agent.header, receive=receive))
                await asyncio.wait_for(hold.parked.wait(), 60)
                rid = [e for e in srv.served if e['slot'] == slot][-1]['rid']
                receive.put_nowait(dict(type='http.disconnect'))
                await B2.settle(lambda: any(e[0] == 'abort' for e in h.events))
                hold.gate.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                retried.update(client=k, slot=slot, rid=rid, messages=messages)

            async def play(k):
                agent = agents[k]
                for j in range(episodes):
                    prompt = agent.first if j == 0 else self.goal_output(k, j - 1)[1]
                    if retry == (k, j):
                        await cancelled(k, prompt)
                    await agent.step(prompt)
                    agent.outputs.append(self.goal_output(k, j)[0])
            await asyncio.gather(*(play(k) for k in range(count)))
            if opened is not None:
                await opened
            cut = {retried['rid']} if retried else set()
            self.check_prompts(srv, cut=cut)
            rows, goal_rows = srv.rows(), {r['request_id']: r for r in srv.rows('goals.jsonl')}
            # The cancelled request is the one failure: the client went away, its slot failed.
            self.assertEqual([(f['error'], f['session_index']) for f in srv.rows('failures.jsonl')],
                             [('CancelledError', retried['slot'])] if retried else [])
            self.assertEqual(len(rows), sum(len(a.requests) for a in agents))
            # goals.jsonl also holds the cancelled request's Goal (built before it was cancelled).
            self.assertEqual(sorted(goal_rows), sorted([r['request_id'] for r in rows] + sorted(cut)))
            self.assertTrue(all(status == 200 for a in agents for _, status, _ in a.requests))
            self.assertTrue(all('summary' in a.events for a in agents))
            self.check_goals(srv, agents, rows, goal_rows, rewriter)
            # Goal forwards ran on the Goal probe of the engine's renderer, each within the transient.
            self.assertTrue(probes and max(probes) <= layout.goal_tokens, (probes, layout.goal_tokens))
            if retried:
                # The retry rebound the client's key to a fresh slot, rendered from text, where its
                # Goal starts again from the instruction and the same terminal output.
                row = next(r for r in rows if r['chat_text']['binding'] == 'rebind'
                           and r['chat_text']['rebind_reason'] == 'failed')
                served = next(e for e in srv.served if e['rid'] == row['request_id'])
                self.assertEqual((served['messages'], row['stage_b']['local_turn']), (retried['messages'], 0))
                self.assertEqual(goal_rows[row['request_id']]['decision']['gate_decision'], 'INITIALIZE')
                self.assertEqual(goal_rows[retried['rid']]['inputs'], goal_rows[row['request_id']]['inputs'])
            self.boundary(srv)
            self.quiescent(srv)
            sessions = {}
            for row in rows:
                ct, rid = row['chat_text'], row['request_id']
                g = goal_rows[rid]
                sessions.setdefault((ct['key_sha256'], ct['session_id']), []).append(dict(
                    kind=ct['kind'], rebuild=row['stage_b']['rebuild'], finish=ct['finish_reason'],
                    binding=ct['binding'], rebind_reason=ct['rebind_reason'],
                    prompt=row['prompt_token_ids'], selected=row['selected_logical_pages'],
                    attended=h.runner.visible[rid], goal_inputs=ct['goal_inputs'], inputs=g['inputs'],
                    decision=g['decision'], query=g['query']['sha256'],
                    rewrites=[json.loads(x['request']) for x in g['rewrites']]))
            replies = [[(messages, status, body['choices'][0]['message']['content'], body['usage'])
                        for messages, status, body in a.requests] for a in agents]
            found.update(sessions=sessions, replies=replies, events=srv.rows(R.EVENTS), engine=h, goals=goals,
                         goal_rows=goal_rows, parked_peak=peak[0], rewriter=rewriter, probes=probes,
                         width=max(len(b.rows) for b in h.batches), children=list(children), writers=writers)

        def check_goals(self, srv, agents, rows, goals, rewriter):
            """test_chat_text_e2e.goal_sessions' checks of every turn's Goal inputs and Goal source."""
            served = {entry['rid']: entry['messages'] for entry in srv.served}
            raw = (srv.output / 'goals.jsonl').read_bytes()
            lines = {json.loads(line)['request_id']: line for line in raw.splitlines()}
            seen, sources, calls, rewrote, previous = set(), set(), 0, 0, {}
            for row in rows:
                rid, ct = row['request_id'], row['chat_text']
                messages, g = served[rid], goals[rid]
                agent = next(a for a in agents if a.instruction in messages[0]['content'])
                inputs, decision = g['inputs'], g['decision']
                goal = json.loads(g['goal']['text'])
                self.assertEqual(row['query']['goal_text'],
                                 dict(file='goals.jsonl', sha256=sha256(lines[rid]).hexdigest()))
                self.assertEqual(ct['goal_inputs'], dict(definition=T2.DEFINITION, request=inputs['request'],
                    requirement=inputs['requirement']['rule'], observation=inputs['observation']['rule'],
                    rewriter=inputs['rewriter']))
                self.assertEqual(goal['root_goal'], agent.instruction)
                span = inputs['requirement']['span']
                self.assertEqual(messages[span['message']]['content'][span['start']:span['end']], agent.instruction)
                last, first = messages[-1]['content'], messages[0]['content']
                kind = ('questions_call' if first.startswith('You are picking up') else
                        'summary_call' if last.startswith('You are about to hand off') else
                        'answers_call' if last.startswith('The next agent has a few questions') else 'main')
                self.assertEqual(inputs['request'], kind)
                exchanges = [json.loads(x['request']) for x in g['rewrites']]
                calls += len(exchanges)
                before = previous.get(row['session_id'])          # this engine session's last Goal
                if kind != 'main':
                    self.assertEqual((inputs['rewriter'], inputs['observation'], decision['slm_attempted'], exchanges),
                                     ('skipped', dict(rule='subcall_none', span=None, chars=0), False, []))
                    if kind != 'questions_call' and before is not None:
                        self.assertEqual((inputs['goal_source'], decision['gate_decision'], decision['triggers'],
                                          row['query']['goal_hash'], g['goal']['text']),
                                         ('session_last', 'KEEP', ['summarization_subcall'], before[0], before[1]))
                    else:
                        self.assertEqual((inputs['goal_source'], decision['gate_decision'],
                                          goal['retrieval_goal']['description']),
                                         ('requirement_only', 'INITIALIZE',
                                          'Retrieve historical evidence relevant to: ' + agent.instruction))
                elif last.startswith('Here are the answers'):
                    # The hand-off: a rebuild of the same client session; the Goal starts again from
                    # the instruction, with the summary text as the observation of its rewriter call.
                    summary = messages[1]['content'].split('**Summary from Previous Agent:**\n')[1].split(
                        '\n\n**Current Terminal Screen:**')[0]
                    self.assertTrue(row['stage_b']['local_turn'] == 0 and summary.startswith('summary '))
                    self.assertEqual((inputs['observation']['rule'], inputs['goal_restarted'],
                                      decision['gate_decision'],
                                      [(x['trigger'], x['root_goal'], x['latest_observation']) for x in exchanges]),
                                     ('handoff_summary', True, 'INITIALIZE',
                                      [('initialization', agent.instruction, summary)]))
                elif len(messages) == 1:
                    self.assertEqual((inputs['observation']['rule'], decision['gate_decision'],
                                      [(x['trigger'], x['latest_observation']) for x in exchanges]),
                                     ('first_turn', 'INITIALIZE', [('initialization', '')]))
                else:
                    # A turn's observation: that turn's terminal output only, the wrapper removed (on the
                    # fresh slot of the retry's rebind too, where the Goal initializes from it).
                    expected = next(o for o in agent.outputs if last.endswith(o) or last.endswith(o + ' '))
                    rule = ('warnings_terminal_output' if last.startswith('Previous response had warnings') else
                            'timeout_terminal_state' if last.startswith('Previous command:') else 'terminal_output')
                    span = inputs['observation']['span']
                    self.assertEqual((inputs['observation']['rule'], messages[span['message']]['content'][
                        span['start']:span['end']]), (rule, expected))
                    for x in exchanges:
                        self.assertEqual((x['root_goal'], x['latest_observation']), (agent.instruction, expected))
                    rewrote += decision['gate_decision'] == 'REWRITE' and bool(exchanges)
                seen.add((inputs['request'], inputs['observation']['rule']))
                sources.add((inputs['request'], inputs['goal_source']))
                previous[row['session_id']] = (row['query']['goal_hash'], g['goal']['text'])
            # Every rewriter call is one of goals.jsonl's (the cancelled request's among them).
            self.assertEqual(sum(len(g['rewrites']) for g in goals.values()), len(rewriter.requests))
            self.assertTrue({('main', 'maintained'), ('summary_call', 'session_last'),
                             ('answers_call', 'session_last'), ('questions_call', 'requirement_only')} <= sources,
                            sources)
            self.assertGreater(rewrote, 0)          # a traceback's new error: the rewriter saw that turn's output
            # Goals with the rewriter were built on the Goal threads (goal_async); a sub-call's never parks.
            parked = [r['request_id'] for r in rows if 'goal_async_ms' in srv.native(r['request_id'])]
            self.assertTrue(parked and all(goals[rid]['inputs']['rewriter'] == 'allowed' for rid in parked))
            self.assertTrue({('main', 'first_turn'), ('main', 'terminal_output'), ('main', 'warnings_terminal_output'),
                             ('main', 'timeout_terminal_state'), ('main', 'handoff_summary'),
                             ('summary_call', 'subcall_none'), ('questions_call', 'subcall_none'),
                             ('answers_call', 'subcall_none')} <= seen, seen)
            self.assertEqual({r['chat_text']['kind'] for r in rows}, {'session', 'main', 'aside'})

    result = unittest.TestResult()
    Case('scenario').run(result)
    return (result.testsRun, result.failures, result.errors, result.skipped), found


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class EndToEndTests(unittest.TestCase):
    """goal_query Terminus 2 sessions on Gemma's chat-text server with --admission-overlap 16,
    --release-idle-pages and --decode-fastpath against a server without them (goal_run)."""
    maxDiff = None
    # Pool-0 blocks of the server that never releases: every open session's pins and one turn (without
    # --release-idle-pages a smaller pool fills with idle sessions' Pages, and a turn waits for a close).
    REFERENCE = 8000
    PRESSURE = 700             # every first turn fits (its whole row, the Goal forward, a reply), not every pin

    def served(self, count, episodes, retry, closes=False):
        reference = goal_run(count, episodes, self.REFERENCE, retry=retry)
        self.assertEqual(reference[0], (1, [], [], []))
        self.assertEqual(reference[1]['events'], [])
        self.assertNotIsInstance(reference[1]['goals'], GA.OverlapGoalAdmissions)
        run = goal_run(count, episodes, self.PRESSURE, overlap=16, release=True, fast=True, retry=retry)
        self.assertEqual(run[0], (1, [], [], []))
        found, large = run[1], reference[1]
        self.assertEqual(found['replies'], large['replies'])
        sessions, expected = found['sessions'], large['sessions']
        self.assertEqual(set(sessions), set(expected))
        if closes:
            # Sixteen clients and their asides outgrow the K open sessions: which idle sessions the
            # pool closes is a matter of timing, and a client session that continues on a fresh slot
            # (rebound in either run) starts its Goal and its selection again there.
            touched = {key for side in (sessions, expected) for key, rows in side.items()
                       if any(r['binding'] == 'rebind' for r in rows)}
            self.assertGreaterEqual(len(sessions) - len(touched), len(sessions) // 2)
            sessions = {k: v for k, v in sessions.items() if k not in touched}
            expected = {k: v for k, v in expected.items() if k not in touched}
        self.assertEqual(sessions, expected)
        h = found['engine']
        # --admission-overlap 16: the overlap park table, the ledger, several Goals parked at once.
        self.assertEqual((h.lifecycle.permits, h.lifecycle.reservations, h.endpoint.phase1.records), (16, {}, {}))
        self.assertGreaterEqual(found['parked_peak'], 2)
        # --decode-fastpath: the window owner's incremental observation.
        self.assertTrue(h.bridge.decode_fastpath)
        # --release-idle-pages: idle sessions' Pages released under the pressure and readmitted.
        events = found['events']
        evictions = [e for e in events if e['event'] == 'evict']
        readmissions = [e for e in events if e['event'] == 'readmit']
        self.assertTrue(evictions and readmissions)
        self.assertEqual(h.core._remote_ra_idle_release.summary(), dict(
            evictions=len(evictions), evicted_pages=sum(e['pages'] for e in evictions),
            readmissions=len(readmissions), readmitted_pages=sum(e['pages'] for e in readmissions)))
        return found

    def test_goal_query_sessions_with_the_terminal_bench_flags(self):
        found = self.served(4, 6, retry=(1, 1))
        self.assertGreater(found['width'], 1)                      # sessions decode together

    @unittest.skipUnless(SLOW, 'about 5 minutes of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_goal_query_sessions_with_the_terminal_bench_flags(self):
        self.served(16, 6, retry=(1, 1), closes=True)


# ---- Gemma chat-text goal_query with the batched Goal rewriter (integrate-1021) -------------------

def batch_argv(model, fast):
    """tb_goal_argv at --kv-cache-gib 25 with K = 17 open sessions (the most 25 GiB holds for
    goal_query under the selective rule), --goal-rewriter-batch 4 and the decode fast paths ``fast``
    (1-3)."""
    argv = tb_goal_argv(model)
    argv[argv.index('--kv-cache-gib') + 1] = '25'
    argv[argv.index('--max-running-sessions') + 1] = '17'
    extra = ['--decode-fastpath2', '--decode-fastpath3'][:fast - 1]
    return argv + extra + ['--goal-rewriter-batch', '4']


class BatchedRewriterTests(unittest.TestCase):
    """gemma-tb-goal (goal_query on Gemma's chat-text) and gemma-rewriter-batch (--goal-rewriter-batch
    on Gemma) meet in integrate-1021, with gemma-fastpath3 and the 25-GiB KV size of Gemma's
    selective servers: the launcher, the startup checks and the token cap's 25-GiB default
    (serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS[25] = 25,600, live now that 25 is a launcher choice),
    also with --deferred-completion-io (gemma-deferred-io), and Terminus 2 sessions with summaries
    under pool pressure against a server without the flags."""
    maxDiff = None
    setUp, load = GT.LauncherTests.setUp, GT.LauncherTests.load
    prepare, rewriter = LauncherTests.prepare, LauncherTests.rewriter

    def test_the_launch_at_25_gib_with_k17_and_the_fast_paths(self):
        import test_gemma_kv_pools as GKP
        self.assertIn(25, serve.GEMMA4_KV_CACHE_GIB)
        self.assertEqual(serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS[25], 25600)
        block, page = MP.GEMMA4_GEOMETRY.sliding_block_bytes(16), MP.GEMMA4_GEOMETRY.page_bytes(16)
        fast_keys = ('decode_fastpath', 'decode_fastpath2', 'decode_fastpath3')
        options = dict(kv_cache_gib=25, max_running=17, admission_overlap=16, goal_rewriter_batch=4,
                       cuda_graphs='full')
        configs = {}
        for fast in (1, 2, 3):
            with self.subTest(fast=fast):
                switches = {name: True for name in fast_keys[:fast]}
                _, config, command, _ = self.prepare(**dict(options, **switches))
                self.assertEqual((config['chat_text_goal'], config['goal_rewriter']['used'], config['goal_rewriter_batch'],
                                  config['goal_rewriter_batch_tokens'], config['release_idle_pages']),
                                 (T2.record(), True, 4, 25600, True))
                self.assertEqual({k for k in fast_keys if config.get(k)}, set(switches))
                self.assertEqual((config['admission'], config['max_running_sessions']), (AF.record(admission_overlap=16), 17))
                layout = window(17, 25)
                value = self.load(config)
                self.assertEqual((HC.kv_layout(value), HC.goal_rewriter_batch(value), value['goal_rewriter_batch_tokens']),
                                 (layout, 4, 25600))
                self.assertEqual(C.options(value)['goal_inputs'], T2.record())
                self.assertEqual(command[command.index('--max-num-seqs') + 1], '17')
                configs[fast] = config
                # The dry run prints the Goal-input record; its startup config pins the batch.
                text = io.StringIO()
                with patch.object(sys, 'argv', ['serve.py', '--no-speed-defaults', '--release-dir', str(self.release), '--output',
                                                str(self.root / ('dry-batch%d' % fast)), '--port', '8000',
                                                *batch_argv(self.rewriter(), fast)]), redirect_stdout(text):
                    serve.main()
                dry = json.loads(text.getvalue())
                self.assertEqual((dry['status'], dry['chat_text_goal']), ('SERVICE_DRY_RUN', T2.record()))
        # The three launches differ only in the fast-path switches.
        differ = lambda a, b: {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        self.assertEqual(differ(configs[1], configs[3]) - {'output_directory'}, {'decode_fastpath2', 'decode_fastpath3'})
        # --deferred-completion-io (gemma-deferred-io) with each: its startup key and nothing else,
        # no vLLM argument; config.load and the profile check take it with the batch and the fast paths.
        for fast in (1, 2, 3):
            with self.subTest(fast=fast, deferred=True):
                switches = {name: True for name in fast_keys[:fast]}
                _, plain, plain_command, _ = self.prepare(**dict(options, **switches))
                _, config, command, _ = self.prepare(**dict(options, deferred_completion_io=True, **switches))
                self.assertEqual(differ(plain, config) - {'output_directory'}, {TD.KEY})
                self.assertIs(config[TD.KEY], True)
                self.assertEqual(len(command), len(plain_command))
                paths = {command[k - 1] for k, (a, b) in enumerate(zip(command, plain_command)) if a != b}
                self.assertLessEqual(paths, {'--additional-config', '--chat-template'})
                value = self.load(config)
                self.assertEqual((HC.deferred_completion_io(value), HC.goal_rewriter_batch(value),
                                  value['goal_rewriter_batch_tokens'], HC.kv_layout(value)), (True, 4, 25600, window(17, 25)))
                text = io.StringIO()
                with patch.object(sys, 'argv', ['serve.py', '--no-speed-defaults', '--release-dir', str(self.release), '--output',
                                                str(self.root / ('dry-deferred%d' % fast)), '--port', '8000',
                                                *batch_argv(self.rewriter(), fast), '--deferred-completion-io']), \
                        redirect_stdout(text):
                    serve.main()
                self.assertEqual(json.loads(text.getvalue())['status'], 'SERVICE_DRY_RUN')
        # vLLM's own startup estimate at 25 GiB: pool 1 and request_pages for one request.
        layout = window(17, 25)
        w = GKP.gemma_world()
        _, kv = GKP.plan(w, layout, 25 * GiB, max_len=65536)
        pool1 = I0.window_pool_blocks(layout, 5)
        self.assertEqual(kv.num_blocks_by_pool, ((25 * GiB - pool1 * block) // page, pool1))
        self.assertGreater(kv.num_blocks_by_pool[0] - 1, layout.request_pages)
        # An explicit T, and 0 (no cap), are pinned as given; K = 18 does not fit 25 GiB for goal_query.
        for tokens in (8192, 0):
            _, config, _, _ = self.prepare(**dict(options, decode_fastpath=True, goal_rewriter_batch_tokens=tokens))
            self.assertEqual(config['goal_rewriter_batch_tokens'], tokens)
        with self.assertRaisesRegex(ValueError, 'use at most 17 sessions'):
            self.prepare(**dict(options, max_running=18, decode_fastpath=True))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows and the fake rewriter model')
class BatchedEndToEndTests(unittest.TestCase):
    """Terminus 2 goal_query sessions with summaries on Gemma's chat-text server at K = 17 with
    --goal-rewriter-batch 4 (the 25-GiB default cap, 25,600 tokens; test_goal_rewriter_batch's
    in-process child), --admission-overlap 16, --release-idle-pages under pool pressure and the
    decode fast paths, against the same child unbatched on a server without the flags: every
    client's replies and every client session's prompts, selections, attended Pages, Goal inputs,
    decisions, query texts and rewriter requests equal; Goals parked at once, the rewriter's calls
    in batches of at most 4 (some of more than one), idle sessions released and readmitted. With
    --deferred-completion-io (gemma-deferred-io) the completion files go through
    test_deferred_completion's writer: every file it wrote equals the synchronous path's bytes,
    nothing pending or failed. The suite runs four clients with the three fast paths and the
    switch; REMOTE_RA_SLOW_TESTS=1 adds the first and the first two fast paths with the switch,
    the three without it, and sixteen clients with the switch."""
    maxDiff = None
    # Pool-0 blocks: the first turns of three clients (their Goals parked together, so that the
    # rewriter's calls meet in a batch) fit at once, not every idle session's pins later.
    PRESSURE = 1100

    def served(self, count, episodes, fast, closes=False, deferred=True):
        reference = goal_run(count, episodes, EndToEndTests.REFERENCE, limit=17, child=True)
        self.assertEqual(reference[0], (1, [], [], []))
        self.assertEqual(reference[1]['events'], [])
        (_, plain), = reference[1]['children']
        self.assertEqual(set(plain.sizes), {1})
        run = goal_run(count, episodes, self.PRESSURE, limit=17, overlap=16, release=True, fast=fast,
                       child=True, batch=4, tokens=serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS[25], deferred=deferred)
        self.assertEqual(run[0], (1, [], [], []))
        found, large = run[1], reference[1]
        self.assertEqual(found['replies'], large['replies'])
        sessions, expected = found['sessions'], large['sessions']
        self.assertEqual(set(sessions), set(expected))
        if closes:
            touched = {key for side in (sessions, expected) for key, rows in side.items()
                       if any(r['binding'] == 'rebind' for r in rows)}
            self.assertGreaterEqual(len(sessions) - len(touched), len(sessions) // 2)
            sessions = {k: v for k, v in sessions.items() if k not in touched}
            expected = {k: v for k, v in expected.items() if k not in touched}
        self.assertEqual(sessions, expected)
        h = found['engine']
        self.assertEqual((h.lifecycle.permits, h.lifecycle.reservations), (16, {}))
        self.assertGreaterEqual(found['parked_peak'], 2)
        self.assertEqual((h.bridge.decode_fastpath, bool(getattr(h.bridge, 'decode_fastpath2', False)),
                          bool(getattr(h.bridge, 'decode_fastpath3', False))), (True, fast >= 2, fast >= 3))
        (inner, child), = found['children']
        self.assertEqual((inner._remote_ra_batch.limit, inner._remote_ra_batch.cap), (4, 25600))
        exchanges = [x for g in found['goal_rows'].values() for x in g['rewrites']]
        self.assertEqual(len(exchanges), sum(child.sizes))
        self.assertTrue(exchanges and all(1 <= x['batch_size'] <= 4 for x in exchanges))
        self.assertTrue(1 < max(child.sizes) <= 4, child.sizes)
        events = found['events']
        self.assertTrue(any(e['event'] == 'evict' for e in events) and any(e['event'] == 'readmit' for e in events))
        self.assertEqual(len(found['writers']), int(deferred))
        for engine in found['writers']:
            stats = engine.writer.stats()
            self.assertEqual((engine.writer.mismatches, stats['errors'], stats['pending']), ([], 0, 0))
            # Every request's native file came through the writer (equal to the synchronous bytes).
            self.assertGreaterEqual(engine.writer.compared, sum(len(rows) for rows in expected.values()))
        return found

    def test_goal_query_sessions_with_the_batch_the_three_fast_paths_and_deferred_io(self):
        self.served(4, 6, 3)

    @unittest.skipUnless(SLOW, 'about 3 minutes of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_the_first_and_the_first_two_fast_paths_and_the_three_without_deferred_io(self):
        for fast, deferred in ((1, True), (2, True), (3, False)):
            with self.subTest(fast=fast, deferred=deferred):
                self.served(4, 6, fast, deferred=deferred)

    @unittest.skipUnless(SLOW, 'about 5 minutes of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_clients(self):
        self.served(16, 6, 3, closes=True)


# ---- the pinned tokenizer and template ----------------------------------------------------------

HF = GT.HF


@unittest.skipUnless(HF is not None, 'pinned Gemma 4 tokenizer (%s) and transformers required' % G.ENV)
class PinnedTokenizerTests(unittest.TestCase):
    """A Terminus 2 session (2.0.0's prompts: the pinned template, terminal outputs in Terminus's
    wrappers, the summary sub-calls and the hand-off) on the pinned Gemma tokenizer and template."""
    maxDiff = None

    def test_terminus_goals_on_the_gemma_transcript(self):
        import importlib
        import test_gemma_renderer as TR
        from gemma4_renderer import Gemma4SourceRenderer
        from runtime_renderer import PLAIN_GOAL_SCHEMA
        from remote_ra_live_v2 import selection_contract as sc
        schema = importlib.import_module('dram_kv.schema')
        maintained = importlib.import_module('dram_kv.goals.maintained')
        renderer = Gemma4SourceRenderer(model_dir=G.metadata_dir(), tokenizer=HF, render_template=TR.hf_render(HF),
                                        verify_identity=lambda: True, page_tokens=16)
        kit = GT.real_kit()
        make = renderer.chat_transcripts(C.record(profile=MP.GEMMA4))
        drift = sc.default_contract(16)['entity_drift_steps']
        # The session's maintained Goal, and the requirement's of a sub-call (query_selection's
        # requirement_goal: a rewriter-less constructor on a session of its own).
        constructor = maintained.MaintainedGoalConstructor(None, entity_drift_steps=drift)
        requirement = maintained.MaintainedGoalConstructor(None, entity_drift_steps=drift,
                                                           initialize_with_rewriter=False)
        main, aside = make(), make()
        instruction = ('Fix the build of /app/src/parser.c: `make` fails with "undefined reference to '
                       '`yylex`"; keep the Makefile targets. Résumé tests live in /app/tests/test_parser.py.')
        outputs = ['root@4f0c2a9d81be:/app# make\ncc -c src/parser.c -o build/parser.o\n'
                   'src/parser.c:41: undefined reference to `yylex`\nmake: *** [Makefile:12: all] Error 1\n'
                   'root@4f0c2a9d81be:/app# ',
                   'root@4f0c2a9d81be:/app# python3 -m pytest -q tests/test_parser.py\n'
                   'Traceback (most recent call last):\n  File "/app/tests/test_parser.py", line 7, in test_ok\n'
                   'AssertionError: 2 != 3\n1 failed in 0.03s\nroot@4f0c2a9d81be:/app# ']
        wrapped = [TG.new_output(outputs[0]),
                   TG.warnings('- Extra text detected after JSON object',
                               TG.timeout('make\n', 1.0, TG.new_output(outputs[1])))]
        actions = ['{"analysis": "Build it.", "plan": "make", "commands": [{"keystrokes": "make\\n", "duration": 1.0}]}',
                   '{"analysis": "Link error.", "plan": "test", "commands": [{"keystrokes": "python3 -m pytest -q '
                   'tests/test_parser.py\\n", "duration": 1.0}]}',
                   '{"analysis": "One test fails.", "plan": "read it", "commands": []}']
        summary_text = 'make fails at the link step (yylex); tests/test_parser.py has one failure.'
        questions = '1. Which linker flags does the Makefile use?'
        answers = 'It links with -lfl missing.'
        first = TG.first_prompt(instruction)
        summary = TG.summary_prompt(instruction)
        questions_prompt = TG.question_prompt(instruction, summary_text, 'root@4f0c2a9d81be:/app# ')
        history = [TG.user(first), TG.assistant(actions[0]), TG.user(wrapped[0]), TG.assistant(actions[1]),
                   TG.user(wrapped[1])]
        requests = [  # (transcript, messages, reply)
            (main, history[:1], actions[0]), (main, history[:3], actions[1]), (main, history, actions[2]),
            (main, history + [TG.assistant(actions[2]), TG.user(summary)], summary_text),
            (aside, [TG.user(questions_prompt)], questions),
            (main, history + [TG.assistant(actions[2]), TG.user(summary), TG.assistant(summary_text),
                              TG.user(TG.ANSWERS + questions)], answers),
            (main, [TG.user(first), TG.user(questions_prompt), TG.assistant(questions),
                    TG.user(TG.HANDOFF % answers)], actions[0])]
        kinds, lengths, rebuilds = [], [], []
        for transcript, messages, reply in requests:
            tokens, _, _ = transcript.render(messages)
            rebuilds.append(transcript.pending['rebuild'])
            self.assertEqual(list(tokens), kit.reference(messages))
            self.assertEqual(kit.without_channels(tokens), kit.stock(messages))
            inputs = T2.classify(messages)
            kinds.append((inputs.request, inputs.observation_rule))
            fields = T2.context_fields(inputs)
            self.assertEqual(fields['objective'], instruction)
            # The engine's previous action is the session's decoded reply: the text the client echoes.
            action = transcript.previous_action() if inputs.rewrite else None
            if inputs.rewrite and transcript.turn:
                self.assertEqual(action, messages[-2]['content'])
            session = 'aside' if transcript is aside else 'main'
            context = schema.TurnContext(session, 'i', transcript.turn + 1, fields['objective'], None, action,
                                         fields['observation'], message_source=fields['message_source'],
                                         structured_state=fields['structured_state'])
            if not inputs.rewrite:
                goal = requirement.construct(context)
                requirement.release_session(session)
            else:
                if inputs.handoff:
                    constructor.release_session(session)       # the rebuild starts the Goal again
                goal = constructor.construct(context)
            text = sc.goal_query_text(goal, 'full_goal')
            probe = list(renderer(dict(schema=PLAIN_GOAL_SCHEMA, text=text)).token_ids)
            self.assertEqual(probe, [MP.BOS] + kit.encode(text))
            self.assertNotIn(MP.BOS, probe[1:])
            lengths.append(len(probe))
            transcript.commit(tokens, kit.encode(reply) + [MP.TURN_CLOSE])
        self.assertEqual(kinds, [('main', 'first_turn'), ('main', 'terminal_output'),
                                 ('main', 'warnings_terminal_output'), ('summary_call', 'subcall_none'),
                                 ('questions_call', 'subcall_none'), ('answers_call', 'subcall_none'),
                                 ('main', 'handoff_summary')])
        self.assertEqual(rebuilds, [False] * 6 + [True])
        # Every Goal probe fits the window layout's Goal transient (2,048 tokens): 117 to 171 tokens
        # here (the deterministic Goal; a rewriter's Goals are longer, test_gemma_goal).
        self.assertLessEqual(max(lengths), MP.GEMMA4_GOAL_TOKENS)


if __name__ == '__main__':
    unittest.main()
