"""The Goal rewriter's timing and --goal-rewriter-batch N against the Gemma 4 profile, CPU only.

integrate-1015 refused --goal-rewriter-batch for the Gemma 4 stage 1 (not checked with the
window layout). gemma-rewriter-batch takes it as Qwen does: goal_query with a Goal rewriter
and --admission-overlap K >= N, on the XML and kv-session protocols (and on chat-text, whose
goal_query gemma-tb-goal serves: test_gemma_chat_text_goal's batched tests). The batch changes how the rewriter's calls run (the sessions' Goal threads
queue them for one child that runs them as one batched generate); everything the window layout
adds for goal_query runs on the engine thread after a parked Goal is done, one resumed admission
per pass, and reads only the GoalRecord: the Goal forward's transient (``goal_tokens`` in the
layout record), the selective rule's ``request_pages``, the WindowPhase1Coordinator's Goal
forward and the sliding ledger of --admission-overlap. Covered:

* the launcher (serve.py's Gemma path): the batch pins ``goal_rewriter_batch`` and
  ``goal_rewriter_batch_tokens`` (the profile's default by --kv-cache-gib,
  serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS: 28,672 at 24 GiB, 25,600 at 25, 13,312 at 28, none
  at 32, where T must be given; Qwen keeps 16,384) and nothing else: the window layout, the vLLM command and the
  other keys are those of the launch without it, on mswea-xml and kv-session; the refusals are
  Qwen's (another policy, --no-goal-rewriter, K < N, the values); remote_ra_http.config.load
  checks the pin as for Qwen; compose_engine's worker startup record does not carry it and
  ready.json records it;
* the defaults' derivation from job 3410948, recomputed for every KV size (an estimate until a
  GPU run);
* end to end on the window layout's engine and worker fakes with test_goal_rewriter_batch's
  in-process child (a fake model whose rows decode independently, so batched and unbatched
  answers agree token for token): sixteen goal_query sessions with --admission-overlap 16 and
  --release-idle-pages, N = 4 and the 28-GiB default cap:
  every turn decides, selects, attends and replies exactly as with the unbatched rewriter,
  the child receives the same requests, every batch holds at most N calls within the cap,
  every exchange records its batch, the Goal forwards stay within their transient, idle
  sessions' Pages are released and readmitted and both pools return to their baselines; the
  kv-session protocol's MemAgent sessions likewise (six sessions, K = 6);
* the rewriter's timing evidence: a Gemma server without a pinned rewriter and the policies
  other than goal_query write none of it (the Qwen control writes it).
"""
from contextlib import ExitStack, redirect_stdout
import importlib
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

import install
import serve
import test_gemma_e2e as GE
import test_gemma_goal as GG
import test_gemma_kv_session_e2e as KE
import test_gemma_overlap as GO
import test_gemma_release as GR
import test_goal_rewriter_batch as RB
import test_goal_text as GT
import test_host_trim as HT
import test_stageb_worker as W
from test_gemma_install import gemma_fixture
from remote_ra_http import config as HC
from remote_ra_live_v2 import admission_flags as AF
from remote_ra_live_v2 import goal_text
from remote_ra_live_v2 import model_profile as MP

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
QWEN_OTHER = ('full', 'random50', 'observation_query', 'random_fill')
GEMMA_SERVED = tuple(p for p in GE.POLICIES if p != 'goal_query')
# Every key the Goal rewriter work adds to the evidence: per exchange, per native file
# (the totals of a parked Goal) and the startup pin of the batch.
TIMING_KEYS = (set(goal_text.TIMING) | set(goal_text.TELEMETRY) | set(goal_text.BATCH) |
               {native for native, _ in goal_text.TOTALS} | {'goal_rewriter_batch', 'goal_rewriter_batch_tokens'})
BATCH_KEYS = ('goal_rewriter_batch', 'goal_rewriter_batch_tokens')
GEMMA_T = serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS[28]     # the launcher tests' KV size; the e2e cap
LIMIT = serve.GOAL_REWRITER_BATCH_TOKENS_LIMIT


def tasks(n):
    return [dict(instance_id='t%02d' % k, problem_statement='fix check %02d' % k) for k in range(n)]


def keys(value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            out.add(k)
            keys(v, out)
    elif isinstance(value, list):
        for v in value:
            keys(v, out)
    return out


def evidence_keys(files):
    """Every JSON key of the evidence files (JSON and JSON-lines), nested keys included."""
    found = set()
    for name, raw in files.items():
        if name.endswith('.jsonl'):
            for line in raw.splitlines():
                if line.strip():
                    keys(json.loads(line), found)
        elif name.endswith('.json'):
            keys(json.loads(raw), found)
    return found


def without_output(value, output):
    """A launch's (config, command) with its output directory named OUT."""
    return json.loads(json.dumps(value).replace(json.dumps(str(output))[1:-1], 'OUT'))


class GemmaLauncherTests(unittest.TestCase):
    """serve.py's Gemma path and config.load take --goal-rewriter-batch as Qwen does."""
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

    def prepare(self, task, policy='full', selection=None, **kw):
        """serve.prepare of a 16-session Gemma server at 131,072 tokens (the group's settings):
        (config, command) with the output directory named OUT, and the config as written."""
        kw = dict(dict(kv_cache_gib=28, page_tokens=16, max_tokens=4096, max_model_len=131072,
                       language_model_only=True, release_idle_pages=True, dram_arena_gib=79), **kw)
        type(self).n += 1
        output = self.root / ('out%d' % self.n)
        _, config, command, _ = serve.prepare(self.release, task, policy, output, 8000, selection, **kw)
        return without_output([config, command], output), config

    def kv(self, policy='goal_query', selection=None, **kw):
        """The Gemma kv-session launch of docs/reference.md 4.14 (test_gemma_kv_session's), 16 sessions."""
        kw = dict(dict(protocol='kv-session', kv_session_thinking='off', max_running=16, max_model_len=262144,
                       dram_arena_gib=150, host_memory_shares=2), **kw)
        return self.prepare(None, policy, self.selection(policy) if selection is None else selection, **kw)

    def selection(self, policy, budget=8192):
        if policy == 'goal_query':
            return dict(active_token_budget=budget, goal_rewriter_model=str(self.rewriter))
        return dict(active_token_budget=budget) if policy in ('observation_query', 'random_fill') else None

    def test_goal_query_pins_the_batch_and_nothing_else(self):
        for name, launch in (('mswea-xml', lambda **kw: self.prepare(tasks(16), 'goal_query',
                                                                    self.selection('goal_query'), **kw)),
                             ('kv-session', lambda **kw: self.kv(**kw))):
            with self.subTest(protocol=name):
                (plain, plain_command), _ = launch(admission_overlap=16)
                self.assertTrue(set(BATCH_KEYS).isdisjoint(plain))
                self.assertEqual(plain['kv_layout']['goal_tokens'], MP.GEMMA4_GOAL_TOKENS)
                self.assertIn('request_pages', plain['kv_layout'])
                for n, tokens, expected in ((4, None, GEMMA_T), (16, None, GEMMA_T), (4, 0, 0), (2, 5000, 5000),
                                            (4, LIMIT, LIMIT)):
                    kw = dict(admission_overlap=16, goal_rewriter_batch=n)
                    if tokens is not None:
                        kw.update(goal_rewriter_batch_tokens=tokens)
                    (config, command), written = launch(**kw)
                    self.assertEqual((config.pop('goal_rewriter_batch'), config.pop('goal_rewriter_batch_tokens')),
                                     (n, expected))
                    # The window layout (goal_tokens, request_pages), the vLLM command and every other key
                    # are the plain launch's.
                    self.assertEqual((config, command), (plain, plain_command))
                    value = self.load(written)
                    self.assertEqual((HC.goal_rewriter_batch(value), HC.kv_layout(value)),
                                     (n, HC.kv_layout(self.load(launch(admission_overlap=16)[1]))))
        # The profile's default differs from Qwen's; Qwen's is unchanged.
        self.assertEqual((serve.GOAL_REWRITER_BATCH_TOKENS, GEMMA_T), (16384, 13312))

    def test_the_default_follows_the_kv_size(self):
        # serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS: each KV size's own default, pinned and loaded; at
        # 32 GiB no batch fits next to the server, so the batch needs an explicit T there.
        goal = self.selection('goal_query')
        for gib in (24, 28):
            with self.subTest(gib=gib):
                _, written = self.prepare(tasks(16), 'goal_query', goal, kv_cache_gib=gib, admission_overlap=4,
                                          goal_rewriter_batch=4)
                self.assertEqual(written['goal_rewriter_batch_tokens'], serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS[gib])
                self.assertEqual(self.load(written)['goal_rewriter_batch_tokens'], written['goal_rewriter_batch_tokens'])
        with self.assertRaisesRegex(ValueError, '--goal-rewriter-batch at --kv-cache-gib 32 needs an explicit '
                                                '--goal-rewriter-batch-tokens T'):
            self.prepare(tasks(16), 'goal_query', goal, kv_cache_gib=32, admission_overlap=4, goal_rewriter_batch=4)
        for tokens in (0, 4096):
            _, written = self.prepare(tasks(16), 'goal_query', goal, kv_cache_gib=32, admission_overlap=4,
                                      goal_rewriter_batch=4, goal_rewriter_batch_tokens=tokens)
            self.assertEqual(HC.goal_rewriter_batch(self.load(written)), 4)
            self.assertEqual(written['goal_rewriter_batch_tokens'], tokens)
        # 25 GiB (the selective servers' size; the integration adds it to the choices) takes 25,600,
        # on both protocols; a size without an entry needs T.
        with patch.object(serve, 'GEMMA4_KV_CACHE_GIB', tuple(sorted(set(serve.GEMMA4_KV_CACHE_GIB) | {25, 26}))):
            for name, launch in (('mswea-xml', lambda **kw: self.prepare(tasks(16), 'goal_query', goal, **kw)),
                                 ('kv-session', lambda **kw: self.kv(**kw))):
                with self.subTest(protocol=name):
                    (plain, plain_command), _ = launch(kv_cache_gib=25, admission_overlap=16)
                    (config, command), written = launch(kv_cache_gib=25, admission_overlap=16, goal_rewriter_batch=4)
                    self.assertEqual((config.pop('goal_rewriter_batch'), config.pop('goal_rewriter_batch_tokens')),
                                     (4, 25600))
                    self.assertEqual((config, command), (plain, plain_command))
                    self.assertEqual(config['effective_server_config']['kv_cache_memory_bytes'], 25 << 30)
                    self.assertEqual(HC.goal_rewriter_batch(self.load(written)), 4)
            with self.assertRaisesRegex(ValueError, 'at --kv-cache-gib 26 needs an explicit'):
                self.prepare(tasks(16), 'goal_query', goal, kv_cache_gib=26, admission_overlap=4, goal_rewriter_batch=4)

    def test_the_refusals_are_qwens(self):
        goal = self.selection('goal_query')
        cases = [(policy, self.selection(policy), dict(admission_overlap=4, goal_rewriter_batch=4),
                  'needs --policy goal_query with --goal-rewriter-model') for policy in GEMMA_SERVED]
        cases += [
            ('goal_query', dict(active_token_budget=8192, no_goal_rewriter=True),
             dict(admission_overlap=4, goal_rewriter_batch=4), 'needs --policy goal_query with --goal-rewriter-model'),
            ('goal_query', goal, dict(goal_rewriter_batch=2), 'needs --admission-overlap K >= 2'),
            ('goal_query', goal, dict(admission_overlap=2, goal_rewriter_batch=4), 'needs --admission-overlap K >= 4'),
            ('goal_query', goal, dict(admission_overlap=4, goal_rewriter_batch=1), 'integer 2-16'),
            ('goal_query', goal, dict(admission_overlap=4, goal_rewriter_batch=17), 'integer 2-16'),
            ('goal_query', goal, dict(admission_overlap=4, goal_rewriter_batch='4'), 'integer 2-16'),
            ('goal_query', goal, dict(admission_overlap=4, goal_rewriter_batch_tokens=5000),
             'applies to --goal-rewriter-batch only'),
            ('goal_query', goal, dict(admission_overlap=4, goal_rewriter_batch=4, goal_rewriter_batch_tokens=-1),
             'integer 0-4194304'),
            ('goal_query', goal, dict(admission_overlap=4, goal_rewriter_batch=4, goal_rewriter_batch_tokens=LIMIT + 1),
             'integer 0-4194304'),
            ('goal_query', goal, dict(admission_overlap=4, goal_rewriter_batch=4, goal_rewriter_batch_tokens=True),
             'integer 0-4194304')]
        for policy, selection, kw, message in cases:
            with self.subTest(policy=policy, kw=kw), self.assertRaisesRegex(ValueError, message):
                self.prepare(tasks(16), policy, selection, **kw)
        with self.assertRaisesRegex(ValueError, 'needs --admission-overlap K >= 4'):
            self.kv(goal_rewriter_batch=4)
        # chat-text serves goal_query since gemma-tb-goal (integrate-1021): the batch pins as on the
        # XML protocol, T by --kv-cache-gib; with another policy the batch has nothing to batch.
        chat = dict(protocol='chat-text', max_running=20, max_model_len=65536, dram_arena_gib=64,
                    admission_overlap=4, goal_rewriter_batch=4)
        _, config = self.prepare(None, 'goal_query', goal, **chat)
        gib = config['effective_server_config']['kv_cache_memory_bytes'] >> 30
        self.assertEqual((config['protocol'], config['goal_rewriter_batch'], config['goal_rewriter_batch_tokens']),
                         ('chat_text_v1', 4, serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS[gib]))
        with self.assertRaisesRegex(ValueError, 'needs --policy goal_query with --goal-rewriter-model'):
            self.prepare(None, 'observation_query', self.selection('observation_query'), **chat)

    def test_load_checks_the_pin_as_for_qwen(self):
        _, config = self.prepare(tasks(16), 'goal_query', self.selection('goal_query'), admission_overlap=4,
                                 goal_rewriter_batch=4)
        value = self.load(config)
        self.assertEqual((value['model_profile'], HC.goal_rewriter_batch(value), value['goal_rewriter_batch_tokens']),
                         (MP.GEMMA4, 4, GEMMA_T))
        _, full = self.prepare(tasks(16), 'full', kv_cache_gib=32)
        self.assertIsNone(HC.goal_rewriter_batch(self.load(full)))
        for bad, message in ((dict(goal_rewriter_batch=8), 'admission-overlap K >= N'),
                             (dict(goal_rewriter_batch=1), 'integer 2-16'),
                             (dict(goal_rewriter_batch_tokens=-1), 'integer 0-4194304'),
                             (dict(goal_rewriter_batch_tokens=None), 'integer 0-4194304')):
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, message):
                self.load(dict(config, **bad))
        alone = {k: v for k, v in config.items() if k != 'goal_rewriter_batch'}
        with self.assertRaisesRegex(RuntimeError, 'goal_rewriter_batch_tokens needs goal_rewriter_batch'):
            self.load(alone)
        with self.assertRaisesRegex(RuntimeError, 'goal_query with a pinned Goal rewriter'):
            self.load(dict(full, goal_rewriter_batch=2, goal_rewriter_batch_tokens=GEMMA_T))

    def test_the_worker_record_carries_nothing_and_ready_records_the_pin(self):
        import test_gemma_flags as TFL
        launched = {}
        for batch in (None, 4):
            kw = dict(admission_overlap=4) if batch is None else dict(admission_overlap=4, goal_rewriter_batch=batch)
            _, config = self.prepare(tasks(16), 'goal_query', self.selection('goal_query'), **kw)
            value, env = self.load(config), self.environment(config)
            record, outcome = TFL.engine_record(value, env)
            self.assertIsNone(outcome)
            launched[batch] = config, record, self.ready(value, record)
        (plain, plain_record, plain_ready), (config, record, ready) = launched[None], launched[4]
        # The batch is the engine's: compose_engine sends the worker what it sends without it.
        self.assertTrue(set(BATCH_KEYS).isdisjoint(record))
        self.assertEqual(record, plain_record)
        self.assertTrue(set(BATCH_KEYS).isdisjoint(plain_ready))
        self.assertEqual((ready['goal_rewriter_batch'], ready['goal_rewriter_batch_tokens']), (4, GEMMA_T))
        differ = ('server_run_id', 'process_start_id') + BATCH_KEYS
        self.assertEqual({k: v for k, v in ready.items() if k not in differ},
                         {k: v for k, v in plain_ready.items() if k not in differ})

    def ready(self, config, record):
        """ready.json of a Gemma server with this startup config (FrontendEvidence.write_ready, its
        worker's receipt echoing the record as the window worker does)."""
        import pathlib
        import test_ms_frontend as MF
        from remote_ra_http.evidence import FrontendEvidence
        from remote_ra_live_v2 import language_model_only as T
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
        receipt = dict(tensor_geometry={}, source_closure={}, context=record['context'], admission=config['admission'],
                       release_idle_pages=True, **{T.CONFIG_KEY: T.RECORD})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            value = dict(config, output_directory=str(output))
            (output / 'worker-attached.json').write_bytes(json.dumps(dict(
                epoch=MF.EPOCH, report=readiness, receipt=receipt,
                kv_pools=dict(kv_layout=config['kv_layout'], num_blocks_by_pool=[4, 4],
                              free_blocks_by_pool=[4, 4]))).encode())
            with patch('remote_ra_http.evidence.load', lambda: value), patch.object(pathlib.Path, 'read_text', proc):
                evidence = FrontendEvidence(authority, None, dict(config['effective_server_config']))
                evidence.write_ready(NS(routes=[NS(path='/v1/chat/completions', methods={'POST'})]), NS(started=True))
            return json.loads((output / 'ready.json').read_bytes())

    def environment(self, config):
        type(self).n += 1
        path = self.root / ('startup%d.json' % self.n)
        path.write_bytes(serve.wire(config))
        return dict(REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                    REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config))

    def load(self, config):
        """config.load() of a startup config written as serve.py writes it."""
        with patch.dict(os.environ, self.environment(config)):
            return HC.load()


class DefaultCapTests(unittest.TestCase):
    """serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS by --kv-cache-gib, from job 3410948 (an estimate
    until a GPU run)."""

    MIB, KIB = 1 << 20, 1 << 10

    def peak(self, T, B, new=384):
        """MiB of one batched generate of B rows of T / B padded prompt tokens (the Qwen3-0.6B model of
        serve.py: 152 KiB per padded prompt token, 112 KiB per generated token per row, 3 B L^2 bytes of
        attention mask)."""
        L = T / B
        return (152 * self.KIB * T + 112 * self.KIB * new * B + 3 * B * L * L) / self.MIB

    def worst(self, T):
        return max(self.peak(T, B) for B in (2, 3, 4))

    def bound(self, budget, slack):
        """The largest T whose widest batch fits ``budget`` MiB with ``slack`` (0: none fits)."""
        if self.worst(1) * slack > budget:
            return 0
        low, high = 1, 1 << 23
        while low < high:
            middle = (low + high + 1) // 2
            low, high = (middle, high) if self.worst(middle) * slack <= budget else (low, middle - 1)
        return low

    def budget(self, gib):
        """MiB left for one batch next to a Gemma server with ``gib`` GiB of KV: the EngineCore's
        largest report at 32 GiB (92.24 GiB) less the KV it does not hold, the largest admission
        allocation that failed (258 MiB), the rewriter child's weights, context, kernels and
        workspaces (1,137 + 1,382 MiB) and a 1 GiB margin, of the 95.08 GiB (97,362 MiB) CUDA can
        allocate."""
        engine = round(92.24 * 1024) - (32 - gib) * 1024
        return engine, 97362 - engine - 258 - (1137 + 1382) - 1024

    def test_each_default_is_the_largest_multiple_of_1024_within_the_slack_bound(self):
        table = serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS
        # Every KV size the profile takes has an entry, and so does 25 GiB (the selective servers'
        # size, the user's decision of 10-21, which the integration adds to the choices).
        self.assertLessEqual(set(serve.GEMMA4_KV_CACHE_GIB) | {25}, set(table))
        self.assertEqual(table, {24: 28672, 25: 25600, 28: 13312, 32: None})
        figures = {}
        for gib in table:
            engine, budget = self.budget(gib)
            bounds = self.bound(budget, 1.0), self.bound(budget, 1.3)
            expected = bounds[1] // 1024 * 1024 or None
            figures[gib] = (engine, budget) + bounds
            with self.subTest(gib=gib):
                self.assertEqual(table[gib], expected)
                if expected:
                    # The widest batch is worst as 2 x T/2; it fits the budget with the slack.
                    self.assertEqual(max(range(2, 5), key=lambda B: self.peak(expected, B)), 2)
                    self.assertLessEqual(1.3 * self.worst(expected), budget)
        self.assertEqual(figures, {24: (86262, 7299, 36068, 29098), 25: (87286, 6275, 31900, 25624),
                                   28: (90358, 3203, 17918, 14113), 32: (94454, -893, 0, 0)})
        self.assertEqual({gib: (round(self.worst(t)), round(1.3 * self.worst(t))) for gib, t in table.items() if t},
                         {24: (5516, 7171), 25: (4822, 6268), 28: (2314, 3008)})
        # Qwen's default (job 3404230, 27B server) needs more than the 28-GiB headroom.
        self.assertGreater(1.3 * self.worst(serve.GOAL_REWRITER_BATCH_TOKENS), self.budget(28)[1])


# ---- end to end: the window layout's engine and worker with a batched rewriter -------------------

DELAY = 0.2                    # seconds per line the in-process child takes (a batch as long as one call)
ERRORS = ('ValueError', 'KeyError', 'TypeError', 'IndexError', 'RuntimeError', 'OSError')
# Every other lane's task statement is this much longer: its rewriter prompts are longer than a
# quarter of the default cap (at most three of them fit a batch), the others shorter.
DETAILS = ' Details: ' + ' '.join('detail%03d' % k for k in range(140))


def rewrite_script(index, turns):
    """test_gemma_release.script's lanes with a new error type in every observation after the first
    turn: each turn's Goal is a rewrite (one rewriter call), the outputs padded lane by lane."""
    out = []
    for k in range(turns):
        observation = None
        if k:
            error = ERRORS[(k + index) % len(ERRORS)]
            body = ('Traceback (most recent call last):\n  File "pkg/g%02d.py", line %d, in run\n%s: bad %d'
                    % (index, 3 + k, error, k))
            observation = GR.S.OUT % (1, body + ' ' + 'w' * (60 * (index % 5)))
        out.append((observation, GE.TARGETS[(k + index) % len(GE.TARGETS)], (k + index) % 4 == 3))
    return out


def long_tasks(count):
    """test_gemma_release.tasks with every other statement longer (DETAILS)."""
    out = GR.tasks(count)
    for k, task in enumerate(out):
        if k % 2 == 0:
            task['problem_statement'] += DETAILS
    return out


class RewriterChild:
    """test_goal_rewriter_batch's in-process child behind the authority's Goal rewriter
    (query_selection.goal_rewriter patched): unbatched, or batched by N within a token cap,
    built from the modules the engine imports (an earlier test's sys.modules patch may have
    dropped the copies imported at the top)."""

    def __init__(self, test):
        self.test, self.children, self.pin = test, [], None

    def __call__(self, pin, batch=None, tokens=0):
        self.test.assertEqual(pin, self.pin)
        B, R = GO.live('goal_rewriter_batch'), importlib.import_module('dram_kv.goals.rewriter')
        make = lambda: R.PersistentCommandGoalRewriter(('in-process-child',), timeout_seconds=60.0,
                                                       startup_timeout_seconds=60.0)
        if batch is None:
            rewriter = make()
            child = RB.InProcess(rewriter, delay=lambda size: DELAY)
            rewriter.__enter__()
        else:
            rewriter, holder = make(), make()
            child = RB.InProcess(holder, delay=lambda size: DELAY)
            B.BatchedRewriter(rewriter, holder, batch, tokens, counter() if tokens else None).install()
        self.test.addCleanup(rewriter.close)
        self.children.append(child)
        return rewriter


def counter():
    """The parent's prompt counter with the fake child's tokenizer (one token per character)."""
    return GO.live('goal_rewriter_batch').counter(RB.Tokenizer(), RB.MAX_NEW, torch)


def prompts(line):
    """The prompt tokens of each call of one child line."""
    count, canonical = counter(), GO.live('goal_rewriter_batch').canonical_text
    value = json.loads(line)
    return [count(canonical(e)) for e in (value['batch'] if 'batch' in value else [value])]


def with_rewriter(test, value, batch, tokens, permits):
    """A startup config of the e2e harnesses with the pinned rewriter, --admission-overlap K and the
    batch (None: off); the batch pin as remote_ra_http.config.load checks it."""
    SC = GO.live('selection_contract')
    value = dict(value, selection_contract=SC.validate(dict(value['selection_contract'], goal_rewriter=test.rewriter.pin)),
                 admission=AF.record(admission_overlap=permits))
    if batch is not None:
        value.update(goal_rewriter_batch=batch, goal_rewriter_batch_tokens=tokens)
    test.assertEqual(GO.live('goal_rewriter_batch').pinned(value), batch)
    return value


def instrument(h):
    """The batches the batched rewriter forms (``taken``: size and calls left queued) and a hold:
    ``hold(n)`` keeps the child's answer to its next line until n calls queue behind it."""
    batcher = h.selection.rewriter._remote_ra_batch
    taken, take = [], batcher._take

    def taking():
        batch = take()
        taken.append((len(batch), len(batcher._queue)))
        return batch
    batcher._take = taking

    def hold(queued, child):
        child.gate.clear()

        def release():
            deadline = time.monotonic() + 60
            while len(batcher._queue) < queued and time.monotonic() < deadline:
                time.sleep(0.002)
            child.gate.set()
        thread = threading.Thread(target=release, daemon=True)
        thread.start()
        return thread
    return NS(batcher=batcher, taken=taken, hold=hold)


def exchanges_of(rows):
    """Per (session, turn): its Goal texts and each exchange's request, answer and error, without the
    request ids, times and batch keys (which differ between servers)."""
    out = {}
    for row in rows:
        value = {k: v for k, v in row.items() if k not in ('request_id', 'epoch', 'rewrites')}
        calls = []
        for x in row['rewrites']:
            answer = None if x['response'] is None else RB.timeless(json.loads(x['response']))
            if answer is not None:
                answer.pop('request_id', None)
            calls.append(dict(request=x['request'], answer=answer, error=x['error'],
                              tokens=(x['input_tokens'], x['output_tokens'])))
        value['rewrites'] = calls
        out[row['session_id'], row['turn']] = value
    return out


def batch_checks(test, rows, child, limit, cap):
    """The batched child's lines: at most ``limit`` calls each, a batch of several within ``cap``
    (a call over it alone), and every exchange recording its batch. Returns the batch sizes."""
    sizes = []
    for line in child.lines:
        tokens = prompts(line)
        sizes.append(len(tokens))
        test.assertLessEqual(len(tokens), limit)
        if len(tokens) > 1 and cap:
            test.assertLessEqual(len(tokens) * max(tokens), cap)
    test.assertEqual(sizes, child.sizes)
    exchanges = [x for row in rows for x in row['rewrites']]
    test.assertEqual(len(exchanges), sum(sizes))
    batches = {}
    for x in exchanges:
        batches.setdefault(x['batch_id'], []).append(x['batch_size'])
    test.assertEqual(sorted(len(v) for v in batches.values()), sorted(sizes))
    test.assertTrue(all(set(v) == {len(v)} for v in batches.values()))
    return sizes


def same_as_unbatched(test, batched, alone):
    """Every turn as with the unbatched rewriter: tokens, replies, selections, attended Pages; the Goals
    and their exchanges; the requests the child received, byte for byte."""
    test.assertEqual(batched.result, alone.result)
    test.assertEqual(exchanges_of(batched.rows), exchanges_of(alone.rows))
    test.assertEqual(RB.envelope_texts(batched.child), RB.envelope_texts(alone.child))
    test.assertEqual(max(alone.child.sizes), 1)
    test.assertTrue(all(set(goal_text.BATCH).isdisjoint(x) for row in alone.rows for x in row['rewrites']))


class Checkpoint:
    """A Goal rewriter checkpoint directory (its config.json is what the pin hashes)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.directory = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.directory.name) / 'qwen3-0.6b'
        cls.checkpoint.mkdir()
        (cls.checkpoint / 'config.json').write_bytes(b'{"model_type":"qwen3"}')

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()
        super().tearDownClass()

    def patch_rewriter(self):
        self.rewriter = RewriterChild(self)
        self.rewriter.pin = GO.live('selection_contract').rewriter_pin(self.checkpoint)
        self.enterContext(patch.object(GO.live('query_selection'), 'goal_rewriter', self.rewriter))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows and the fake rewriter model')
class SixteenBatchedSessionsTests(Checkpoint, unittest.TestCase):
    """test_gemma_overlap's sixteen goal_query sessions (--admission-overlap 16, the window ledger,
    Goals parked on their sessions' threads, --release-idle-pages) with every turn's Goal a rewrite,
    the pinned rewriter unbatched and batched (N = 4, the default cap)."""
    maxDiff = None
    TURNS = 3
    TEST_GOAL = GO.OverlapEndToEndTests.TEST_GOAL       # 512 tokens: the Goal forward's transient here
    check, boundary, window_of = GR.SixteenSessionTests.check, GR.SixteenSessionTests.boundary, \
        GR.SixteenSessionTests.window_of
    finish = GO.OverlapEndToEndTests.finish
    watch, goal_checks = GG.SixteenGoalSessionsTests.watch, GG.SixteenGoalSessionsTests.goal_checks
    _alone = None

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        self.enterContext(patch.object(GE, 'GOAL', self.TEST_GOAL))
        self.enterContext(patch.object(GR, 'script', rewrite_script))
        self.patch_rewriter()

    @staticmethod
    def pool():
        """Twice the selective rule's minimum pool 0: admissions park several at a time, and the idle
        sessions' Pages are still released and readmitted."""
        return 2 * GR.selective_blocks(GE.window_layout('goal_query', GR.SESSIONS))

    def server(self, stack, blocks, batch, tokens, records):
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        sessions = GR.SESSIONS
        layout = GE.window_layout('goal_query', sessions)
        value = with_rewriter(self, GR.config('goal_query', output, long_tasks(sessions), layout, True),
                              batch, tokens, sessions)
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        h = GO.OverlapPressure('goal_query', output, long_tasks(sessions), blocks, True, self.TURNS, sessions)
        h.output, h.turns = output, self.TURNS
        self.assertEqual(h.selection.rewriter_batch, batch)
        self.watch(records)(h)
        return h

    def done(self, h, records, child, taken=None):
        rows = [json.loads(line) for line in (h.output / goal_text.FILE).read_bytes().splitlines()]
        result, events = self.finish(h, GR.SESSIONS)
        return NS(result=result, h=h, events=events, rows=rows, records=records, child=child, taken=taken)

    def run_case(self, blocks, batch, tokens, hold=0):
        """One server's run; ``hold``: the child holds its first answer until that many calls queue."""
        records = []
        with ExitStack() as stack:
            h = self.server(stack, blocks, batch, tokens, records)
            child = self.rewriter.children[-1]
            taken = None
            if batch is not None:
                probe = instrument(h)
                taken = probe.taken
                if hold:
                    probe.hold(hold, child)
            h.serve(self.check)
            return self.done(h, records, child, taken)

    def alone(self):
        """The unbatched rewriter's run (once per class)."""
        cls = type(self)
        if cls._alone is None:
            cls._alone = self.run_case(self.pool(), None, 0)
            self.assertEqual(len(cls._alone.child.sizes), GR.SESSIONS * self.TURNS)   # every Goal a rewrite
        return cls._alone

    def test_sixteen_sessions_with_releases_decide_as_unbatched(self):
        alone = self.alone()
        batched = self.run_case(self.pool(), 4, GEMMA_T, hold=2)
        same_as_unbatched(self, batched, alone)
        sizes = batch_checks(self, batched.rows, batched.child, 4, GEMMA_T)
        self.assertGreaterEqual(max(sizes), 2)
        self.assertLess(len(sizes), len(alone.child.sizes))
        for run in (alone, batched):
            # Admissions parked together; idle sessions' Pages were released and readmitted.
            self.assertGreaterEqual(run.h.parked_peak, 2)
            self.assertTrue([e for e in run.events if e['event'] == 'evict'] and
                            [e for e in run.events if e['event'] == 'readmit'])
            # The Goal forwards ran in their transient, each query vector the Goal forward's last row.
            self.goal_checks(run.records, GR.SESSIONS)
        forwards = lambda run: sorted(r['tokens'] for r in run.records if r['kind'] == 'goal')
        self.assertEqual(forwards(batched), forwards(alone))

    def test_sixteen_held_rewrites_batch_by_n_within_the_cap(self):
        # test_gemma_overlap's scripted round: every lane's turn 0, then all sixteen turn-1 admissions
        # parked at once (the window ledger reserves one phase-1 transient for them), their rewriter
        # calls held until all have queued. Prompts: the long statements' lanes above a quarter of the
        # cap and within a third of it, the others within a quarter: a batch holds three or four calls,
        # and one holding a long call stops at three, a cut by the cap (N = 4).
        alone = self.alone()
        records = []

        def turns(scripts):
            """The lanes' next turns, served by a fresh park table (serve closes the last one)."""
            h.table = type(h.table)(h.lifecycle, wake=lambda: None, goals=True, permits=GR.SESSIONS)
            h.lifecycle.goals = h.table
            for lane, script in zip(h.lanes, scripts):
                lane.script = script
        with ExitStack() as stack:
            h = self.server(stack, GR.REFERENCE_BLOCKS, 4, GEMMA_T, records)
            child, probe = self.rewriter.children[-1], instrument(h)
            held_turn, rest = [lane.script[1:2] for lane in h.lanes], [lane.script[2:] for lane in h.lanes]
            for lane in h.lanes:
                lane.script = lane.script[:1]
            h.serve(self.check)
            turns(held_turn)
            lines, formed = len(child.lines), len(probe.taken)
            child.gate.clear()
            before = h.window_pool.get_num_free_blocks()
            self.assertEqual([h.admit(lane) for lane in h.lanes], ['parked'] * GR.SESSIONS)
            # Every held call reaches the batcher before the gate opens (a late one would batch apart
            # from the others); the wait ends as soon as they have, and is long for loaded machines.
            queued = lambda: len(probe.batcher._queue) + sum(len(prompts(line)) for line in child.lines[lines:])
            deadline = time.monotonic() + 300
            while queued() < GR.SESSIONS and time.monotonic() < deadline:
                time.sleep(0.002)
            self.assertEqual(queued(), GR.SESSIONS, 'not every held rewriter call reached the batcher')
            # Parked admissions hold reservations only: pool 1 is untouched until a Goal returns.
            self.assertEqual(h.window_pool.get_num_free_blocks(), before)
            child.gate.set()
            # The held round alone: a lane whose turn 1 ends early cannot queue its turn-2 call
            # behind the last held calls, where the batcher (FIFO, within N and the cap) would put
            # it in their batch (seen under load: the dispatcher took three calls before the gate
            # closed, the thirteen left split 3+3+3+3+1, and two turn-2 calls joined the last).
            h.serve(self.check)
            held_lines, taken = child.lines[lines:], probe.taken[formed:]
            turns(rest)
            h.serve(self.check)
            run = self.done(h, records, child, probe.taken)
        same_as_unbatched(self, run, alone)
        batch_checks(self, run.rows, child, 4, GEMMA_T)
        held = [prompts(line) for line in held_lines]
        self.assertEqual(sum(map(len, held)), GR.SESSIONS)
        tokens = [t for batch in held for t in batch]
        self.assertEqual(sum(t > GEMMA_T // 4 for t in tokens), GR.SESSIONS // 2, held)
        self.assertTrue(all(t <= GEMMA_T // 3 for t in tokens), held)
        self.assertIn(max(map(len, held)), (3, 4))
        self.assertEqual([size for size, _ in taken], list(map(len, held)))
        self.assertTrue(any(size < 4 and left for size, left in taken))           # the cap cut a batch
        self.goal_checks(run.records, GR.SESSIONS)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows and the fake rewriter model')
class KvSessionBatchedTests(Checkpoint, unittest.TestCase):
    """test_gemma_kv_session_e2e's MemAgent sessions with goal_query, --admission-overlap K = 6 and
    the pinned rewriter, unbatched and batched (N = 4, the default cap), with --release-idle-pages
    on twice the selective rule's minimum pool 0."""
    maxDiff = None
    SESSIONS = KE.SHORT
    check, boundary, window_of = KE.GemmaMemAgentSessionTests.check, KE.GemmaMemAgentSessionTests.boundary, \
        KE.GemmaMemAgentSessionTests.window_of
    small = KE.GemmaMemAgentSessionTests.small

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        self.patch_rewriter()

    def run_case(self, blocks, batch, tokens, hold=0):
        from remote_ra_http.evidence import engine_closed
        sessions = self.SESSIONS
        layout = GE.window_layout('goal_query', sessions)
        with ExitStack() as stack:
            output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            value = with_rewriter(self, KE.config('goal_query', output, layout, True, sessions), batch, tokens,
                                  sessions)
            stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
            stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            h = OverlapSessions('goal_query', output, blocks, True, sessions, sessions)
            self.assertEqual(h.selection.rewriter_batch, batch)
            child = self.rewriter.children[-1]
            if hold:
                instrument(h).hold(hold, child)
            h.serve(self.check)
            for lane in h.lanes:
                self.assertEqual(lane.agent.answer, 'Answer: chunk 2 names the harbour city.')
            self.boundary(h)
            result = {lane.index: [(list(i['tokens']), i['committed']['text'], i['plan']['selected'],
                                    h.runner.visible[i['rid']]) for i in h.done if i['lane'] is lane]
                      for lane in h.lanes}
            rows = [json.loads(line) for line in (output / goal_text.FILE).read_bytes().splitlines()]
            path = output / KE.R.EVENTS
            events = [json.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []
            h.lifecycle.disconnect()
            h.sessions.close()
            engine_closed(h.core)
            self.assertEqual((h.table.parked, h.lifecycle.reservations, h.endpoint.phase1.records), ({}, {}, {}))
            self.assertEqual(h.adapter.window.pinned_blocks(), 0)
            self.assertEqual((h.pool.get_num_free_blocks(), h.window_pool.get_num_free_blocks()),
                             (h.baseline, h.window_baseline))
        return NS(result=result, h=h, rows=rows, events=events, child=child)

    def test_memagent_sessions_batch_their_rewrites_and_answer_as_unbatched(self):
        pool = 2 * self.small(GE.window_layout('goal_query', self.SESSIONS))
        alone = self.run_case(pool, None, 0)
        batched = self.run_case(pool, 4, GEMMA_T, hold=2)
        same_as_unbatched(self, batched, alone)
        self.assertGreaterEqual(max(batch_checks(self, batched.rows, batched.child, 4, GEMMA_T)), 2)
        for run in (alone, batched):
            self.assertGreaterEqual(run.h.parked_peak, 2)
            self.assertTrue([e for e in run.events if e['event'] == 'evict'] and
                            [e for e in run.events if e['event'] == 'readmit'])
            # Gemma's kv geometry ran: continuations without a header and headers after a user turn.
            headers = [i['request_kv'].header_tokens for i in run.h.done]
            self.assertIn(0, headers)
            self.assertIn(len(KE.KIT.header()), headers)


class OverlapSessions(KE.Sessions):
    """test_gemma_kv_session_e2e.Sessions with --admission-overlap K, wired as test_gemma_overlap's
    OverlapPressure (the park table, the pinned lease, the ledger, the per-session constructors):
    each pass resumes one admission whose Goal is done or starts the next lane's add, then runs one
    batched step."""

    def __init__(self, policy, output, fa_blocks, release, sessions, permits):
        AO, GA = GO.live('admission_overlap'), GO.live('goal_async')
        super().__init__(policy, output, fa_blocks, release, sessions, False)
        self.permits, self.parked_adds, self.busy, self.parked_peak = permits, {}, set(), 0
        self.table = GA.OverlapGoalAdmissions(self.lifecycle, wake=lambda: None,
                                              goals=self.selection.rewriter is not None, permits=permits)
        self.lifecycle.goals = self.table
        self.lifecycle.lease = AF.timeouts(permits)[1]
        AO.attach_engine(self.core, self.lifecycle, AF.record(admission_overlap=permits))

    def admit(self, lane):
        """KE.Sessions.admit with the add started in the park table."""
        from remote_ra_live_v2.capacity import CapacityWait
        K = KE.K
        if lane.body is None:
            lane.body = lane.agent.client.build(lane.agent.messages)
        body, normalized = lane.body
        request = K.parse_complete(body, budget=KE.BUDGET, bound=KE.BOUND, max_model_len=KE.CONTEXT,
                                   empty_header=True)
        t = lane.frontend
        t.check(request)
        effective = K.effective_max_tokens(request, t.prompt_length(request), KE.CONTEXT)
        first = t.turn == 0
        source = K.source(request, K.new_messages(request, t.history, first),
                          K.objective_of(request.messages) if first else None)
        t.stage(request)
        tokens, stable, span = t.render()
        command = self.kv_command(lane, tokens, source, effective)
        rid = command.request_id
        before = len(self.runner.computed)
        info = dict(rid=rid, lane=lane, tokens=tokens, request_kv=request, effective=effective, span=span,
                    stable=stable, before_stream=tuple(t.stream), mark=self.mark(), turn=lane.turn,
                    body=body, normalized=normalized, messages=[],
                    client_stream=list(KE.KP.client_stream(lane.agent.client.bodies + [body],
                                                           lane.agent.client.responses)))
        replies = []
        try:
            out = self.table.start(self.route.add_steps(self.dispatcher, command, 'fp-' + rid),
                                   lambda frame, failure: replies.append((frame, failure)))
        except CapacityWait:
            t.abandon()
            self.natives.pop(rid.encode())
            self.waits += 1
            return None
        if out is self.table.PARKED:
            self.parked_adds[rid] = (info, replies)
            self.busy.add(lane.index)
            return 'parked'
        return self.accepted(info, out, before)

    def accepted(self, info, ack, before):
        GK = KE.GK
        lane, rid = info['lane'], info['rid']
        self.busy.discard(lane.index)
        info['ack'] = ack
        claim = self.endpoint.claims[rid]
        info.update(request=self.requests[rid], binding=claim['binding'], geo=claim['geometry'],
                    namespace=self.adapter.sessions[lane.sid].namespace)
        info['probe'] = [x for r, x in self.runner.computed[before:] if r.startswith('phase1-')]
        text, end = lane.replies.pop(0)
        outputs = KE.KIT.encode(text) + {'call': [GK.TOOL_OPEN], 'turn': [GK.TURN_END], 'eos': [GK.EOS],
                                         'pad': [GK.PAD]}[end]
        info['outputs'] = outputs[:info['effective']]
        self.runner.samples[rid] = iter(info['outputs'])
        info.update(first=True, produced=0)
        lane.body = None
        self.running.append(info)
        return info

    def resume(self):
        waiting, before = set(self.parked_adds), len(self.runner.computed)
        self.table.resume()
        for rid in waiting:
            info, replies = self.parked_adds[rid]
            if rid in self.table.parked or not replies:
                continue
            del self.parked_adds[rid]
            (frame, failure), = replies
            if failure is not None:
                raise failure
            self.accepted(info, frame, before)

    def serve(self, check):
        k = passes = 0
        while any(lane.agent.answer is None for lane in self.lanes) or self.running or self.table.parked:
            passes += 1
            assert passes < 400000, 'the overlap loop does not finish'
            if self.table.ready():
                self.resume()
            else:
                busy = {info['lane'].index for info in self.running} | self.busy
                ready = [lane for lane in self.lanes if lane.agent.answer is None and lane.index not in busy]
                started = None
                if ready and len(self.busy) < self.permits:
                    started = self.admit(ready[k % len(ready)])
                    k += 1
                if started is None and not self.running:
                    time.sleep(0.002)       # only a Goal thread can move now
            self.parked_peak = max(self.parked_peak, len(self.table.parked))
            if self.running:
                before = len(self.done)
                self.batch_step()
                for info in self.done[before:]:
                    check(self, info)
        self.table.close()


# ---- the rewriter's timing evidence --------------------------------------------------------------

class GemmaEvidenceTests(unittest.TestCase):
    """A Gemma server (test_gemma_e2e's engine, worker and evidence, no rewriter pinned) writes no
    rewriter timing."""
    maxDiff = None
    setUp = GE.GemmaEndToEndTests.setUp
    harness = GE.GemmaEndToEndTests.harness
    windows = GE.GemmaEndToEndTests.windows
    check = GE.GemmaEndToEndTests.check
    play = GE.GemmaEndToEndTests.play
    close = GE.GemmaEndToEndTests.close

    def files(self, policy):
        h, _ = self.play(policy, turns=2)
        self.close(h)
        return {p.name: p.read_bytes() for p in sorted(Path(h.output).iterdir())}

    def test_served_policies_write_no_rewriter_timing(self):
        for policy in GEMMA_SERVED:
            with self.subTest(policy=policy):
                files = self.files(policy)
                self.assertTrue(any(name.startswith('native-') for name in files))
                self.assertNotIn(goal_text.FILE, files)
                self.assertEqual(evidence_keys(files) & TIMING_KEYS, set())

    def test_goal_query_without_a_rewriter_records_no_exchange(self):
        # The engine's goal_query path without a rewriter (the e2e contract pins none):
        # its goals.jsonl rows have no exchange and nothing is timed.
        files = self.files('goal_query')
        rows = [json.loads(line) for line in files[goal_text.FILE].splitlines()]
        self.assertTrue(rows)
        self.assertEqual([row['rewrites'] for row in rows], [[]] * len(rows))
        self.assertEqual(evidence_keys(files) & TIMING_KEYS, set())


class QwenEvidenceTests(unittest.TestCase):
    """Qwen: the other policies write no rewriter timing; goal_query with the rewriter does."""

    def test_other_policies_write_no_rewriter_timing(self):
        for policy in QWEN_OTHER:
            with self.subTest(policy=policy):
                files = HT.play(policy)['files']
                self.assertTrue(any(name.startswith('native-') for name in files))
                self.assertNotIn(goal_text.FILE, files)
                self.assertEqual(evidence_keys(files) & TIMING_KEYS, set())

    def test_control_goal_query_with_the_rewriter_writes_the_fields(self):
        case = GT.EndToEndTests('test_one_row_per_terminal_goal_query_request')
        files = case.play('goal_query')['files']
        exchanges = [x for line in files[goal_text.FILE].splitlines() for x in json.loads(line)['rewrites']]
        self.assertTrue(exchanges)
        self.assertTrue(all(set(goal_text.TIMING) | set(goal_text.TELEMETRY) <= set(x) for x in exchanges))
        # One session builds its Goals in place: no batch, no parked totals.
        found = evidence_keys(files)
        self.assertEqual(found & TIMING_KEYS, set(goal_text.TIMING) | set(goal_text.TELEMETRY))


if __name__ == '__main__':
    unittest.main()
