"""Gemma 4 31B IT, the options of gemma-goal and gemma-perf together (integrate-1016). CPU only.

gemma-goal (goal_query with the Goal forward's own sliding transient, --admission-overlap K
with the window ledger, chunked phase 1, full at 16 sessions under the release rule),
gemma-perf (--cuda-graphs piecewise/full/full-decode, --decode-fastpath, --decode-fastpath2
on the window layout) and gemma-fastpath3 (--decode-fastpath3) test each option alone; this
file covers them together:

- the launcher (serve.prepare on a Gemma release fixture: 16 sessions, 131,072 tokens,
  4,096-token replies, --kv-cache-gib 32, a 79 GiB arena): for full, goal_query (with
  --no-goal-rewriter and with a Goal rewriter), observation_query and random_fill, every
  combination of --cuda-graphs (off, piecewise, full, full-decode), --decode-fastpath
  (alone, with --decode-fastpath2 or with both and --decode-fastpath3), --release-idle-pages
  and --admission-overlap 4 (64 per policy, 320 launches) adds exactly the startup keys and
  vLLM arguments of its options to the plain launch; --goal-rewriter-batch 4 with an explicit T
  (gemma-rewriter-batch: at 32 GiB the profile has no default) adds exactly its two keys to
  each of goal_query-with-a-rewriter's combinations with --admission-overlap 4 (the three fast
  paths too), is refused there without T and with every other policy (and without a rewriter)
  and without the overlap; --decode-fastpath2 without --decode-fastpath, --decode-fastpath3
  without --decode-fastpath2, --phase1-in-step, --gdn-device-handoff, the kv-session protocol
  with thinking on (served with thinking off since gemma-mab) and the chat-text protocol with a
  --task list (served since gemma-tb) are refused whatever else is on;
- startup: remote_ra_http.config.load and the frontend's argument check accept every
  accepted combination (the batch's too), and refuse it with phase1_in_step or
  gdn_device_handoff pinned, decode_fastpath2 without decode_fastpath, decode_fastpath3
  without decode_fastpath2, or goal_rewriter_batch on another policy; for every combination on the plain layout (full) and on goal_query's
  (with the rewriter): compose_engine sends each option once in the worker startup
  record, the window worker (attach_window_worker, its loaded seams faked as in
  test_gemma_graphs) attaches that record with every switch, its evidence adds exactly
  each option's echo, and compose_engine accepts it (and refuses it without any one
  switch); the worker refuses --decode-fastpath2 alone, --decode-fastpath3 without
  --decode-fastpath2 and the other admission flags whatever else is on;
- at run time (test_gemma_e2e's engine and worker fakes, each run compared with the run
  without the switches): goal_query with CUDA graphs (the runner's record attached: phase
  1 and the Goal forward, and only they, run eager) and both decode fast paths, and with
  the three, its three sessions decoding together (test_gemma_fastpath's script); chunked
  phase 1 of goal_query (its Goal forward after the chunks) with graphs and both or the
  three fast paths equal to the plain chunked run; --admission-overlap 8 with
  --release-idle-pages, the three fast paths and graphs on eight goal_query sessions
  (Goals parked at once, idle Pages released and
  readmitted, one turn's phase 1 in chunks) equal to the K = 1 server without any switch,
  and, after every session's first turn, with nothing running or in flight, a turn that
  fits only once idle sessions are released: the ledger gate admits it unchecked and the
  release runs first, as on the K = 1 server (integrate-1016 fixed
  idle_release.IdleRelease.admit: such a turn found its phase 1 short of the blocks idle
  sessions held); the release rule's pressure
  run of full (one turn's phase 1 in chunks) and the selective rule's of observation_query
  (five sessions of three turns at the minimum pool 0) with the three fast paths and graphs:
  the same turns, the same evictions and readmissions as --release-idle-pages alone,
  allocation_batch RPCs instead of allocation_update.

REMOTE_RA_SLOW_TESTS=1 adds goal_query's whole script in every graph mode (with both and with
the three fast paths; full graphs also with none and with the first), chunked full and
observation_query, sixteen sessions with --admission-overlap 16 and the three fast paths,
eight with both, and the pressure runs of full, observation_query, goal_query and random_fill
at six sessions of four turns (each with one chunked phase 1) with both fast paths and with
the three.
"""
from contextlib import ExitStack, redirect_stdout
import copy
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import install  # noqa: E402
import serve  # noqa: E402
import test_gemma_chunked as GC  # noqa: E402
import test_gemma_e2e as GE  # noqa: E402
import test_gemma_fastpath as TF  # noqa: E402
import test_gemma_fastpath3 as TF3  # noqa: E402
import test_gemma_full as GF  # noqa: E402
import test_gemma_goal as GG  # noqa: E402
import test_gemma_graphs as TG  # noqa: E402
import test_gemma_overlap as GO  # noqa: E402
import test_gemma_release as GR  # noqa: E402
import test_stageb_worker as W  # noqa: E402
from test_gemma_install import gemma_fixture  # noqa: E402
from remote_ra_http import config as HC  # noqa: E402
from remote_ra_live_v2 import cuda_graphs as CG  # noqa: E402
from remote_ra_live_v2 import decode_fastpath as F  # noqa: E402
from remote_ra_live_v2 import kv_layout as I0  # noqa: E402
from remote_ra_live_v2 import language_model_only as T  # noqa: E402
from remote_ra_live_v2 import model_profile as MP  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402
# Imported here, outside any sys.modules patch, so the patches the tests enter keep them
# (test_decode_fastpath explains why).
import remote_ra_step_profile  # noqa: E402,F401
from remote_ra_live_v2 import (composition, context_limits, gdn_handoff, hooks, idle_release,  # noqa: E402,F401
                               native_probe, window_worker)

try:
    import torch
except ImportError:
    torch = None

SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'

# ---- the launcher -------------------------------------------------------------------------------

SESSIONS, K = 16, 4
TASKS = [dict(instance_id='g%02d' % k, problem_statement='fix check %02d' % k) for k in range(SESSIONS)]
COMMON = dict(kv_cache_gib=32, page_tokens=16, max_tokens=4096, max_model_len=131072, language_model_only=True,
              dram_arena_gib=79)
GRAPHS = ('off',) + TG.ON
FP, FP2, FP3 = 'decode_fastpath', 'decode_fastpath2', 'decode_fastpath3'
BOTH = (FP, FP2)
ALL3 = (FP, FP2, FP3)
FAST = ((), (FP,), BOTH, ALL3)
# (cuda_graphs, decode fast paths, release_idle_pages, admission_overlap)
COMBINATIONS = [(graphs, fast, release, overlap) for graphs in GRAPHS for fast in FAST
                for release in (False, True) for overlap in (1, K)]
PLAIN = ('off', (), False, 1)
# The options of each combination besides the decode fast paths: graphs x release x overlap.
OTHERS = [(graphs, release, overlap) for graphs in GRAPHS for release in (False, True) for overlap in (1, K)]
# name: (policy, selection); 'goal_query-rewriter' names the fixture's rewriter checkpoint.
BASES = {'full': ('full', None),
         'goal_query': ('goal_query', dict(active_token_budget=8192, no_goal_rewriter=True)),
         'goal_query-rewriter': ('goal_query', dict(active_token_budget=8192)),
         'observation_query': ('observation_query', dict(active_token_budget=8192)),
         'random_fill': ('random_fill', dict(active_token_budget=8192))}
SWITCH_KEYS = ('cuda_graphs', FP, FP2, FP3, 'release_idle_pages', 'admission', 'goal_rewriter_batch',
               'goal_rewriter_batch_tokens')
# --goal-rewriter-batch K (gemma-rewriter-batch): goal_query with a rewriter and --admission-overlap K.
# At COMMON's 32 GiB the profile has no default token cap (serve.GEMMA4_GOAL_REWRITER_BATCH_TOKENS):
# the batch takes an explicit T.
BATCH_T = 8192
BATCH = dict(goal_rewriter_batch=K, goal_rewriter_batch_tokens=BATCH_T)
BATCH_KEYS = dict(goal_rewriter_batch=K, goal_rewriter_batch_tokens=BATCH_T)
BATCH_POLICY = '--goal-rewriter-batch needs --policy goal_query with --goal-rewriter-model'
BATCH_OVERLAP = '--goal-rewriter-batch %d needs --admission-overlap K >= %d' % (K, K)
BATCH_NO_DEFAULT = '--goal-rewriter-batch at --kv-cache-gib 32 needs an explicit --goal-rewriter-batch-tokens T'
# Refused whatever else is on: (extra launcher options, the refusal).
REFUSED = (
    (dict(phase1_in_step=True), '--phase1-in-step is not available for the Gemma 4 profile'),
    (dict(gdn_device_handoff=True), 'it has no Gated-DeltaNet state'),
    # kv-session runs since gemma-mab (test_gemma_kv_session), with --kv-session-thinking on too since
    # gemma-thinking (test_gemma_thinking_on); it takes no --task list.
    (dict(protocol='kv-session', max_running=SESSIONS), 'protocol kv-session takes no --task'),
    # gemma-tb serves chat-text (test_gemma_chat_text); it takes no --task list.
    (dict(protocol='chat-text', max_running=SESSIONS), 'protocol chat-text takes no --task'))
FP2_ALONE = '--decode-fastpath2 requires --decode-fastpath'
FP3_ALONE = '--decode-fastpath3 requires --decode-fastpath2'


def options(combination):
    graphs, fast, release, overlap = combination
    return dict(cuda_graphs=graphs, release_idle_pages=release, admission_overlap=overlap,
                **{name: True for name in fast})


def expected(plain, combination, base):
    """The (config, command, env) of a combination: the plain launch plus each option's pins."""
    config, command, env = copy.deepcopy(plain)
    graphs, fast, release, overlap = combination
    if selective(base, release):
        # The selective rule (gemma-mab): the layout pins its figure for vLLM's startup check.
        additional = I0.to_additional_config(window(base, release))
        effective = dict(config['effective_server_config'], additional_config=additional)
        config.update(kv_layout=additional[I0.LAYOUT_KEY], effective_server_config=effective,
                      server_config_sha256=serve.pin(effective))
        at = command.index('--additional-config')
        command[at + 1] = json.dumps(additional, sort_keys=True, separators=(',', ':'))
    if graphs != 'off':
        effective = dict(config['effective_server_config'], enforce_eager=False,
                         compilation_config=CG.compilation_config(graphs))
        config.update(cuda_graphs=graphs, effective_server_config=effective, server_config_sha256=serve.pin(effective))
        at = command.index('--enforce-eager')
        command[at:at + 1] = ['--compilation-config', CG.argument(graphs)]
    config.update({name: True for name in fast})
    if release:
        config['release_idle_pages'] = True
    if overlap > 1:
        config['admission'] = GO.record(overlap)
    return [config, command, env]


def selective(base, release):
    """Whether serve.py sizes pool 0 by the selective rule: --release-idle-pages with a fixed budget."""
    return release and (BASES[base][1] or {}).get('active_token_budget') is not None


def window(base, release=False):
    """The window layout serve.py pins for a base (goal_query's with the Goal transient; under the
    selective rule, gemma-mab, with its request_pages)."""
    goal = MP.GEMMA4_GOAL_TOKENS if BASES[base][0] == 'goal_query' else 0
    layout = I0.for_window_server(16, SESSIONS, MP.GEMMA4_GEOMETRY.sliding_window, MP.GEMMA4_FORWARD_TOKENS, goal)
    if selective(base, release):
        import dataclasses
        from remote_ra_live_v2.window_capacity import pinned_request_pages
        layout = dataclasses.replace(layout, request_pages=pinned_request_pages(
            layout, BASES[base][1]['active_token_budget'], COMMON['max_tokens'], COMMON['kv_cache_gib'] << 30))
    return layout


_FIXTURE = {}


def fixture():
    """One Gemma release fixture for the module (test_gemma_graphs' LauncherTests), and a Goal
    rewriter checkpoint; the launches are shared by the tests."""
    if not _FIXTURE:
        directory = tempfile.TemporaryDirectory()
        unittest.addModuleCleanup(directory.cleanup)
        unittest.addModuleCleanup(_FIXTURE.clear)
        root = Path(directory.name)
        with redirect_stdout(io.StringIO()):
            package, site, model, gemma, release = gemma_fixture(root)
            install.install(site, gemma, release, package_root=package, apply=True, model_profile=MP.GEMMA4)
        rewriter = root / 'qwen3-0.6b'
        rewriter.mkdir()
        (rewriter / 'config.json').write_text('{"model_type": "qwen3"}')
        _FIXTURE.update(root=root, release=release, rewriter=rewriter, n=[0], launches={}, startups={})
    return _FIXTURE


class Release:
    """The module's release fixture, its launches and their startup files."""

    def setUp(self):
        super().setUp()
        value = fixture()
        self.root, self.release, self.rewriter = value['root'], value['release'], value['rewriter']
        self.counter, self.launches, self.startups = value['n'], value['launches'], value['startups']

    def name(self, prefix):
        self.counter[0] += 1
        return '%s%05dz' % (prefix, self.counter[0])

    def selection(self, base):
        selection = copy.deepcopy(BASES[base][1])
        if base == 'goal_query-rewriter':
            selection['goal_rewriter_model'] = str(self.rewriter)
        return selection

    def launch(self, base, combination=PLAIN, **extra):
        """('ok', [config, command, env] with the output directory named OUT, the config) or
        ('refused', the message, None)."""
        name = self.name('launch')
        kw = dict(COMMON, **options(combination))
        kw.update(extra)
        with redirect_stdout(io.StringIO()), patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            try:
                manifest, config, command, env = serve.prepare(self.release, copy.deepcopy(TASKS), BASES[base][0],
                                                               self.root / name, 8000, self.selection(base), **kw)
            except ValueError as error:
                return 'refused', str(error), None
        self.assertEqual(env['REMOTE_RA_HTTP_CONFIG_SHA256'], serve.pin(config))
        env = {k: v for k, v in env.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'}
        return 'ok', json.loads(json.dumps([config, command, env]).replace(name, 'OUT')), config

    def launched(self, base, combination=PLAIN):
        """The launch of an accepted combination, once per module."""
        key = (base, combination)
        if key not in self.launches:
            status, value, config = self.launch(base, combination)
            self.assertEqual(status, 'ok', (key, value))
            self.launches[key] = value, config
        return self.launches[key]

    def launched_batch(self, combination):
        """goal_query with a rewriter, a combination with --admission-overlap K and
        --goal-rewriter-batch K (gemma-rewriter-batch), once per module."""
        key = ('goal_query-rewriter+batch', combination)
        if key not in self.launches:
            status, value, config = self.launch('goal_query-rewriter', combination, **BATCH)
            self.assertEqual(status, 'ok', (key, value))
            self.launches[key] = value, config
        return self.launches[key]

    def load(self, config, key=None):
        """config.load() of a startup config written as serve.py writes it, and the environment it
        ran in; ``key``: the (base, combination) of an accepted launch, its file written once."""
        env = self.startups.get(key)
        if env is None:
            path = self.root / (self.name('startup') + '.json')
            path.write_bytes(serve.wire(config))
            env = dict(REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                       REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config))
            if key is not None:
                self.startups[key] = env
        with patch.dict(os.environ, env):
            return HC.load(), env


class LauncherTests(Release, unittest.TestCase):
    maxDiff = None

    def test_every_combination_adds_exactly_its_options(self):
        accepted = 0
        for base in BASES:
            plain = self.launched(base)[0]
            self.assertFalse(set(SWITCH_KEYS) & set(plain[0]))
            self.assertIn('--enforce-eager', plain[1])
            self.assertEqual(plain[0]['kv_layout'], I0.to_additional_config(window(base))[I0.LAYOUT_KEY])
            for combination in COMBINATIONS:
                with self.subTest(base=base, combination=combination):
                    self.assertEqual(self.launched(base, combination)[0], expected(plain, combination, base))
                    accepted += 1
        self.assertEqual(accepted, len(BASES) * 4 * 4 * 2 * 2)
        # The rewriter is pinned on goal_query-rewriter only; full has no contract.
        self.assertIs(self.launched('goal_query-rewriter')[0][0]['goal_rewriter']['used'], True)
        self.assertIs(self.launched('goal_query')[0][0]['goal_rewriter']['used'], False)
        self.assertIsNone(self.launched('full')[0][0]['selection_contract'])

    def test_refusals_hold_whatever_else_is_on(self):
        refused = 0
        for base in BASES:
            cases = []
            for k, (graphs, release, overlap) in enumerate(OTHERS):
                # --decode-fastpath2 alone and --decode-fastpath3 without --decode-fastpath2, whatever
                # else is on.
                cases.append(((graphs, (FP2,), release, overlap), {}, FP2_ALONE))
                cases.append(((graphs, (FP, FP3), release, overlap), {}, FP3_ALONE))
                # --goal-rewriter-batch K (with each decode fast-path choice in turn): every base but
                # goal_query with a rewriter refuses it, and that one without --admission-overlap K
                # and, at 32 GiB, without an explicit T.
                if base != 'goal_query-rewriter':
                    cases.append(((graphs, FAST[k % 4], release, overlap), BATCH, BATCH_POLICY))
                elif overlap == 1:
                    cases.append(((graphs, FAST[k % 4], release, overlap), BATCH, BATCH_OVERLAP))
                else:
                    cases.append(((graphs, FAST[k % 4], release, overlap), dict(goal_rewriter_batch=K),
                                  BATCH_NO_DEFAULT))
            # The other options Gemma does not take, with every switch on in each graph mode.
            cases += [((graphs, ALL3, True, K), extra, message) for graphs in GRAPHS for extra, message in REFUSED]
            for combination, extra, message in cases:
                with self.subTest(base=base, combination=combination, extra=extra):
                    status, error, _ = self.launch(base, combination, **extra)
                    self.assertEqual(status, 'refused')
                    self.assertIn(message, error)
                    refused += 1
        self.assertEqual(refused, len(BASES) * (len(OTHERS) * 3 + len(GRAPHS) * 4))

    def test_the_batch_adds_its_two_keys_to_every_combination_with_the_overlap(self):
        # gemma-rewriter-batch: goal_query with a rewriter takes --goal-rewriter-batch K with
        # --admission-overlap K (and, at 32 GiB, an explicit T), in every combination: exactly
        # goal_rewriter_batch and goal_rewriter_batch_tokens, no vLLM argument, the window layout unchanged.
        plain = self.launched('goal_query-rewriter')[0]
        accepted = 0
        for combination in COMBINATIONS:
            if combination[3] == 1:
                continue
            with self.subTest(combination=combination):
                want = expected(plain, combination, 'goal_query-rewriter')
                want[0].update(BATCH_KEYS)
                self.assertEqual(self.launched_batch(combination)[0], want)
                accepted += 1
        self.assertEqual(accepted, len(COMBINATIONS) // 2)


# ---- startup: config.load, compose_engine, the window worker --------------------------------------

def parsed(effective):
    """vLLM's parsed arguments of a Gemma server pinning ``effective`` (test_gemma_graphs)."""
    value = dict(effective, attention_backend=NS(name=MP.GEMMA4_ATTENTION_BACKEND), mamba_cache_mode='none',
                 mamba_block_size=None, gdn_prefill_backend=None)
    if 'compilation_config' in effective:
        pinned = effective['compilation_config']
        value['compilation_config'] = NS(mode=NS(name=pinned['mode']), cudagraph_mode=NS(name=pinned['cudagraph_mode']))
    return NS(**value)


class Stop(Exception):
    pass


def engine_record(config, env, worker=None):
    """compose_engine on a Gemma startup config (written for ``env``): the worker startup record
    it sends, and its refusal of the reply ``worker(record)`` gives (None: no worker replies;
    'accepted': the echo checks passed, the engine reached its worker proxy)."""
    sent = []

    def collective_rpc(method, args, timeout):
        sent.append(args[0])
        if worker is None:
            raise Stop()
        return [dict(worker(copy.deepcopy(args[0])))]

    def proxy(executor, receipt):
        raise Stop('accepted')
    layout = I0.from_vllm_config(NS(additional_config={I0.LAYOUT_KEY: config['kv_layout']}))
    core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(page_tokens=16))),
              _remote_ra_trusted_issue=dict(config['tasks'][0]),
              _remote_ra_trusted_issues=[dict(t) for t in config['tasks']],
              _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
              _remote_ra_growing_adapter=NS(layout=layout), model_executor=NS(collective_rpc=collective_rpc))
    outcome = None
    with patch.dict(os.environ, env), patch.object(composition, 'EngineWorkerProxy', proxy), \
            patch.object(context_limits, 'pin_request_caps', lambda config, core=None: None):
        try:
            composition.compose_engine(core, object())
        except Stop as stop:
            outcome = str(stop) or None
        except LiveV2Error as caught:
            outcome = str(caught)
    startup, = sent
    return startup, outcome


def compiled(graphs):
    """The loaded runner's compilation config and seq-len-bound groups for a mode (test_gemma_graphs)."""
    mode = None if graphs == 'off' else graphs
    return TG.compilation(mode), (0,) if mode in CG.FULL_MODES else ()


class StartupTests(Release, unittest.TestCase):
    maxDiff = None

    def test_load_and_the_frontend_check_accept_every_combination(self):
        count = 0
        for base in BASES:
            for combination in COMBINATIONS:
                graphs, fast, release, overlap = combination
                with self.subTest(base=base, combination=combination):
                    _, config = self.launched(base, combination)
                    value, env = self.load(config, (base, combination))
                    self.assertEqual(value, config)
                    self.assertEqual((HC.model_profile(value), HC.kv_layout(value)), (MP.GEMMA4, window(base, release)))
                    self.assertEqual((HC.cuda_graphs(value), HC.decode_fastpath(value), HC.decode_fastpath2(value),
                                      HC.decode_fastpath3(value), HC.release_idle_pages(value), HC.admission(value)),
                                     (None if graphs == 'off' else graphs, True if FP in fast else None,
                                      True if FP2 in fast else None, True if FP3 in fast else None,
                                      True if release else None, GO.record(overlap) if overlap > 1 else None))
                    with patch.dict(os.environ, env):
                        effective = HC.effective_server_config(parsed(config['effective_server_config']))
                    self.assertEqual(effective, config['effective_server_config'])
                    count += 1
        self.assertEqual(count, len(BASES) * len(COMBINATIONS))

    def test_load_refuses_the_options_gemma_does_not_take(self):
        # Every switch on: full graphs for every policy, and each other graph mode for goal_query
        # (model_profile refuses these keys before it reads anything else).
        cases = [(base, 'full') for base in BASES] + [('goal_query-rewriter', g) for g in GRAPHS if g != 'full']
        for base, graphs in cases:
            _, config = self.launched(base, (graphs, ALL3, True, K))
            admission = config['admission']
            for edited, pattern in (({k: v for k, v in config.items() if k != FP2}, 'requires decode_fastpath2'),
                                    (dict(config, admission=dict(admission, phase1_in_step=True)),
                                     'takes --admission-overlap only'),
                                    (dict(config, admission=dict(admission, gdn_device_handoff=True)),
                                     'takes --admission-overlap only'),
                                    ({k: v for k, v in config.items() if k != FP}, 'requires decode_fastpath')):
                with self.subTest(base=base, graphs=graphs, pattern=pattern), \
                        self.assertRaisesRegex(RuntimeError, pattern):
                    self.load(edited)

    def test_load_checks_the_batch_as_for_qwen(self):
        # gemma-rewriter-batch: every combination of goal_query with a rewriter and the batch loads,
        # the frontend's argument check passes; another policy's config with the batch is refused.
        for combination in COMBINATIONS:
            if combination[3] == 1:
                continue
            with self.subTest(combination=combination):
                _, config = self.launched_batch(combination)
                value, env = self.load(config, ('goal_query-rewriter+batch', combination))
                self.assertEqual(value, config)
                self.assertEqual((HC.goal_rewriter_batch(value), value['goal_rewriter_batch_tokens']),
                                 (K, BATCH_T))
                self.assertEqual(HC.kv_layout(value), window('goal_query-rewriter', combination[2]))
                with patch.dict(os.environ, env):
                    effective = HC.effective_server_config(parsed(config['effective_server_config']))
                self.assertEqual(effective, config['effective_server_config'])
        for base in BASES:
            _, config = self.launched(base, ('full', BOTH, True, K))
            pattern = ('admission-overlap K >= N' if base == 'goal_query-rewriter' else
                       'goal_query with a pinned Goal rewriter')
            with self.subTest(base=base), self.assertRaisesRegex(RuntimeError, pattern):
                self.load(dict(config, goal_rewriter_batch=8, goal_rewriter_batch_tokens=0))

    def test_engine_worker_and_echo_with_every_combination(self):
        """launch -> compose_engine's worker startup record -> attach_window_worker (its loaded
        seams faked) -> its evidence back to compose_engine, for every combination on the plain
        and on goal_query's layout (the record does not depend on the policy otherwise)."""
        attach = TG.WorkerAttachTests().attach
        for base in ('full', 'goal_query-rewriter'):
            plain = {}
            for combination in COMBINATIONS:           # the plain launch first
                graphs, fast, release, overlap = combination
                layout = window(base, release)
                with self.subTest(base=base, combination=combination):
                    value, env = self.load(self.launched(base, combination)[1], (base, combination))
                    attached = []

                    def worker(record):
                        attached.append(attach(record, *compiled(graphs), layout=layout))
                        return attached[-1][0]
                    record, outcome = engine_record(value, env, worker)
                    self.assertEqual(outcome, 'accepted')
                    (evidence, runner), = attached
                    if combination == PLAIN:
                        self.assertFalse(set(SWITCH_KEYS) & (set(record) | set(evidence)))
                        self.assertEqual(record['kv_layout'], I0.to_additional_config(layout)[I0.LAYOUT_KEY])
                        self.assertEqual((evidence['kv_layout'], evidence['context'], evidence[T.CONFIG_KEY]),
                                         (I0.to_additional_config(layout), record['context'], T.RECORD))
                        plain.update(record=record, evidence=evidence)
                    switches = {name: True for name in fast}
                    if release:
                        switches['release_idle_pages'] = True
                    if overlap > 1:
                        switches['admission'] = GO.record(overlap)
                    # The selective rule's layout (gemma-mab) in the record and the worker's echo.
                    layouts = {} if not selective(base, release) else dict(
                        kv_layout=I0.to_additional_config(layout)[I0.LAYOUT_KEY])
                    echoes = {} if not layouts else dict(kv_layout=I0.to_additional_config(layout))
                    # compose_engine sends each option once, in the worker startup record; the
                    # worker's evidence adds exactly each option's echo (its CUDA-graph record).
                    self.assertEqual(record, dict(plain['record'], **switches, **layouts,
                                                  **({} if graphs == 'off' else dict(cuda_graphs=graphs))))
                    if base == 'goal_query-rewriter' and overlap > 1:
                        # --goal-rewriter-batch K is the engine's: the worker startup record is the same.
                        key = ('goal_query-rewriter+batch', combination)
                        self.assertEqual(engine_record(*self.load(self.launched_batch(combination)[1], key))[0],
                                         record)
                    self.assertEqual({k: v for k, v in evidence.items() if k != 'cuda_graphs'},
                                     dict(plain['evidence'], **switches, **echoes))
                    self.assertEqual(evidence.get('cuda_graphs', {}).get('mode'), None if graphs == 'off' else graphs)
                    # ... and attaches every switch.
                    owner, bridge = runner._remote_ra_live_v2_allocation_owner, runner._remote_ra_live_v2_bridge
                    self.assertEqual(getattr(runner, '_remote_ra_cuda_graphs', {}).get('mode'),
                                     None if graphs == 'off' else graphs)
                    self.assertEqual((type(owner.claims) is F.ClaimTable, owner.window_fast, bridge.decode_fastpath),
                                     (FP in fast,) * 3)
                    self.assertEqual((owner.window_rows, getattr(owner, 'decode_fastpath2', False),
                                      bridge.decode_fastpath2), (FP2 in fast,) * 3)
                    self.assertEqual((owner.window_steady, getattr(owner, 'decode_fastpath3', False),
                                      bridge.decode_fastpath3, getattr(owner.backend, 'decode_fastpath3', False)),
                                     (FP3 in fast,) * 4)
                    self.assertIs(owner.release_idle, release)
                    self.assertEqual(getattr(owner.backend, 'admission', None),
                                     GO.record(overlap) if overlap > 1 else None)
            # Every switch on: the engine refuses the echo without any one of them.
            for name, pattern in ((FP, 'another decode_fastpath'), (FP2, 'another decode_fastpath2'),
                                  (FP3, 'another decode_fastpath3'),
                                  ('release_idle_pages', 'another release_idle_pages'),
                                  ('admission', 'other admission flags')):
                with self.subTest(base=base, missing=name):
                    partial = {k: v for k, v in evidence.items() if k != name}
                    self.assertIn(pattern, engine_record(value, env, lambda record: partial)[1])

    def test_the_worker_refuses_what_gemma_does_not_take(self):
        attach = TG.WorkerAttachTests().attach
        context = dict(max_model_len=131072, max_tokens=4096)
        for layout in {window(base, release) for base in BASES for release in (False, True)}:
            for graphs, release, overlap in OTHERS:
                startup = dict(context=context, language_model_only=True)
                if graphs != 'off':
                    startup['cuda_graphs'] = graphs
                if release:
                    startup['release_idle_pages'] = True
                if overlap > 1:
                    startup['admission'] = GO.record(overlap)
                for extra, pattern in (({FP2: True}, 'decode_fastpath2 worker record needs decode_fastpath'),
                                       ({FP: True, FP3: True}, 'decode_fastpath3 worker record needs decode_fastpath2'),
                                       ({FP: True, FP2: True, 'admission': dict(startup.get('admission') or {},
                                                                                phase1_in_step=True)},
                                        'admission-overlap alone'),
                                       ({FP: True, FP2: True, 'admission': dict(startup.get('admission') or {},
                                                                                gdn_device_handoff=True)},
                                        'admission-overlap alone')):
                    with self.subTest(layout=layout, combination=(graphs, release, overlap), extra=extra), \
                            self.assertRaisesRegex(LiveV2Error, pattern):
                        attach(dict(startup, **extra), *compiled(graphs), layout=layout)


# ---- run time -------------------------------------------------------------------------------------

_RUNS = {}


def recorder(steps):
    """The runner's execute_model recording (forced eager, the probe pass's phase or None, the
    scheduled counts) of every step, as the patched runner asks force_eager."""
    real = GE.Runner.execute_model

    def execute(runner, so):
        probe = native_probe.active_probe(runner)
        steps.append((hooks.force_eager(runner), None if probe is None else probe.active['phase'],
                      tuple(so.num_scheduled_tokens.values())))
        return real(runner, so)
    return patch.object(GE.Runner, 'execute_model', execute)


def attacher(graphs, fast):
    """Attach a mode's CUDA-graph record (what cuda_graphs.attach keeps) and the decode fast
    paths to a harness, as attach_window_worker and compose_engine attach them."""
    def attach(h):
        if graphs is not None:
            h.runner._remote_ra_cuda_graphs = dict(mode=graphs)
        if FP3 in fast:
            TF3.attach_fast3(h)
        elif FP2 in fast:
            TF.attach_fast2(h)
        elif FP in fast:
            TF.attach_fast(h)
    return attach


def flags(graphs, fast):
    return dict({name: True for name in fast}, **({} if graphs is None else dict(cuda_graphs=graphs)))


def operations(h):
    """Record the name of every worker RPC of a harness in ``h.operations``."""
    h.operations, rpc = [], h.proxy.executor.collective_rpc

    def collective_rpc(method, args, timeout):
        h.operations.append(args[0])
        return rpc(method, args, timeout)
    h.proxy.executor = NS(collective_rpc=collective_rpc)
    return h


def recording(make):
    """A harness constructor whose harnesses record their worker RPC names (``operations``)."""
    return lambda *args: operations(make(*args))


def long_turns(stack, chunks):
    """test_gemma_release's scripts with lane 1's third turn reading test_gemma_chunked's longest
    observation (about 2,600 tokens, above one native forward of the test geometry), so that its
    phase 1 runs in chunks (``in_chunks`` sets a harness's forwards to test_gemma_chunked.CHUNK
    tokens); ``chunks`` receives the later forwards of every phase-1 lease."""
    real = GR.script

    def script(index, turns):
        out = real(index, turns)
        if index == 1 and turns > 2:
            out[2] = (GC.LONG[0],) + out[2][1:]
        return out
    stack.enter_context(patch.object(GR, 'script', script))
    WP = importlib.import_module('remote_ra_live_v2.window_phase1')
    parse = WP.parse_lease

    def recorded(raw, **kw):
        chunks.append(len(raw.get('chunks', ())))
        return parse(raw, **kw)
    stack.enter_context(patch.object(WP, 'parse_lease', recorded))


def in_chunks(h):
    """A harness whose phase 1 runs in forwards of test_gemma_chunked.CHUNK tokens above one native forward."""
    h.endpoint.phase1.chunk_tokens = GC.CHUNK
    return h


class Switched:
    """Checks of a run with graphs and decode fast paths against the run without them."""

    def eager_exactly_for_probes(self, steps, phases):
        self.assertEqual([forced for forced, _, _ in steps], [phase is not None for _, phase, _ in steps])
        self.assertEqual({phase for _, phase, _ in steps}, phases)
        self.assertTrue(any(not forced and n == (1,) for forced, _, n in steps))


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class GoalQueryTests(Switched, unittest.TestCase):
    """test_gemma_fastpath's script for goal_query (Page and forward chunks, rounds of the three
    sessions decoding together, a rebuild) with graphs and the decode fast paths."""
    maxDiff = None
    setUp = TF.EndToEndTests.setUp

    def serve(self, graphs, fast, **script):
        key = (graphs, fast, tuple(sorted(script.items())))
        if key not in _RUNS:
            steps = []
            with TF.Run(self, 'goal_query', flags(graphs, fast), attach=attacher(graphs, fast),
                        tamper=recorder(steps)) as run:
                _RUNS[key] = dict(run.script(**script).result(), graph_steps=steps)
        return _RUNS[key]

    def compare(self, graphs, fast, **script):
        off, on = self.serve(None, (), **script), self.serve(graphs, fast, **script)
        for key in TF.EndToEndTests.COMPARED:
            if key != 'trace':
                self.assertEqual(on[key], off[key], key)
        trace, kinds, _ = TF.expand(on['trace'])
        self.assertEqual(trace, off['trace'])
        self.assertEqual(kinds['full'] + kinds['delta'] + kinds['unchanged'] > 0, FP2 in fast)
        self.assertFalse(any(t['operation'] == 'allocation_update' for t in on['trace']) and FP2 in fast)
        # The three sessions decode together; with graphs only phase 1 and the Goal forwards run eager.
        self.assertTrue(any(len(step) == 3 for step in on['batched']))
        self.assertTrue(any(n == (1, 1, 1) for _, _, n in on['graph_steps']))
        if graphs is not None:
            self.eager_exactly_for_probes(on['graph_steps'], {None, 'phase1', 'goal'})
        else:
            self.assertFalse(any(forced for forced, _, _ in on['graph_steps']))
        if FP in fast:
            self.assertGreater(sum(on['extended']), 0.6 * len(on['extended']))
        return on

    def test_graphs_and_both_fast_paths_decoding_together(self):
        on = self.compare('full', BOTH, **TF.SHORT)
        self.assertTrue(any(phase == 'goal' for _, phase, _ in on['graph_steps']))

    def test_graphs_and_the_three_fast_paths_decoding_together(self):
        on = self.compare('full', ALL3, **TF.SHORT)
        self.assertTrue(any(phase == 'goal' for _, phase, _ in on['graph_steps']))

    @unittest.skipUnless(SLOW, 'about 35 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_the_whole_script_in_every_mode(self):
        for graphs, fast in [(mode, f) for mode in TG.ON for f in (BOTH, ALL3)] + [('full', ()), ('full', (FP,))]:
            with self.subTest(graphs=graphs, fast=fast):
                self.compare(graphs, fast)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ChunkedTests(Switched, unittest.TestCase):
    """test_gemma_chunked's long observations (phase 1 in forwards of 256 tokens) with graphs and
    both decode fast paths, against the same chunked run without them."""
    maxDiff = None
    setUp, play = GC.ChunkedEndToEndTests.setUp, GC.ChunkedEndToEndTests.play
    check, windows = GE.GemmaEndToEndTests.check, GE.GemmaEndToEndTests.windows

    def run_chunked(self, policy, graphs=None, fast=()):
        steps, leases = [], []

        def harness(test, policy, tasks, n=None):
            output = Path(test.enterContext(tempfile.TemporaryDirectory()))
            value = dict(GE.config(policy, output, list(tasks), GE.window_layout(policy, len(tasks))),
                         **flags(graphs, fast))
            test.enterContext(patch('remote_ra_http.config.load', lambda: value))
            test.enterContext(patch('remote_ra_http.evidence.load', lambda: value))
            h = GE.Harness(policy, output, list(tasks))
            attacher(graphs, fast)(h)
            return h
        with patch.object(GC, 'harness', harness), recorder(steps):
            result, h = self.play(policy, chunk=GC.CHUNK, leases=leases, turns=3)
        shape = [(l['start'], l['end'], [c[:2] for c in l.get('chunks', ())], l['goal'] is not None) for l in leases]
        return result, steps, shape

    def chunked(self, policy, fasts=(BOTH, ALL3)):
        plain, plain_steps, plain_leases = self.run_chunked(policy)
        self.assertFalse(any(forced for forced, _, _ in plain_steps))
        for fast in fasts:
            result, steps, leases = self.run_chunked(policy, 'full', fast)
            self.assertEqual(result, plain)
            self.assertEqual(leases, plain_leases)
            chunked = [lease for lease in leases if lease[2]]
            self.assertTrue(chunked)
            self.eager_exactly_for_probes(steps, {None, 'phase1', 'goal'} if policy == 'goal_query' else
                                          {None, 'phase1'})
            # Every forward of a chunked phase 1 is a probe pass of its own.
            self.assertEqual(sum(1 for _, phase, _ in steps if phase == 'phase1'), sum(1 + len(l[2]) for l in leases))
            if policy == 'goal_query':
                self.assertTrue(any(lease[3] for lease in chunked))

    def test_goal_query(self):
        self.chunked('goal_query')

    @unittest.skipUnless(SLOW, 'about 7 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_full_and_observation_query(self):
        for policy in ('full', 'observation_query'):
            with self.subTest(policy=policy):
                self.chunked(policy)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class OverlapTests(Switched, unittest.TestCase):
    """test_gemma_overlap's goal_query sessions (a rewriter stand-in, Goals parked on their session
    threads; lane 1's third turn a chunked phase 1, ``long_turns``) with --admission-overlap K,
    --release-idle-pages at the selective rule's minimum pool 0, both decode fast paths and
    graphs, against the K = 1 server without any switch with a pool that never releases. After
    every lane's turn 0 (nothing running or in flight, every session idle), a turn whose peak
    the free blocks do not hold: the ledger gate admits it unchecked and the idle sessions it
    needs are released first, as on the K = 1 server (idle_release, fixed in integrate-1016:
    before, its phase 1 could find too few blocks); it is not sent."""
    maxDiff = None
    TURNS, TEST_GOAL = GO.OverlapEndToEndTests.TURNS, GO.OverlapEndToEndTests.TEST_GOAL
    setUp, finish, run_case = GO.OverlapEndToEndTests.setUp, GO.OverlapEndToEndTests.finish, \
        GO.OverlapEndToEndTests.run_case
    check, boundary, window_of = GR.SixteenSessionTests.check, GR.SixteenSessionTests.boundary, \
        GR.SixteenSessionTests.window_of
    switches = None             # (graphs, fast) of the next server, None: none

    def server(self, stack, blocks, permits, release=True, sessions=GR.SESSIONS, turns=None):
        self.chunks = []
        long_turns(stack, self.chunks)
        if self.switches is None:
            return in_chunks(GO.OverlapEndToEndTests.server(self, stack, blocks, permits, release, sessions, turns))
        graphs, fast = self.switches
        config = GR.config
        stack.enter_context(patch.object(GR, 'config', lambda *a: dict(config(*a), **flags(graphs, fast))))
        self.steps = []
        stack.enter_context(recorder(self.steps))
        h = in_chunks(GO.OverlapEndToEndTests.server(self, stack, blocks, permits, release, sessions, turns))
        attacher(graphs, fast)(h)
        if permits is not None and release:
            # Every lane's turn 0, the lone turn (nothing runs, every session idle with its
            # Pages pinned), then the rest of the script.
            serve = h.serve

            def serving(check):
                later = [lane.script[1:] for lane in h.lanes]
                for lane in h.lanes:
                    lane.script = lane.script[:1]
                serve(check)
                self.lone_turn(h)
                h.table = type(h.table)(h.lifecycle, wake=lambda: None, goals=True, permits=permits)
                h.lifecycle.goals = h.table
                for lane, script in zip(h.lanes, later):
                    lane.script = script
                serve(check)
            h.serve = serving
        return operations(h)

    def lone_turn(self, h):
        """Nothing runs and no admission is in flight; hold FA blocks so that lane 0's next turn
        fits only once idle sessions are released: the gate releases them first."""
        CAP = GO.live('capacity')
        lane = h.lanes[0]
        self.assertFalse(h.running or h.table.parked or h.lifecycle.reservations)
        messages = lane.messages + [dict(role='user', content=GE.OBSERVATIONS[0])]
        tokens = lane.frontend.render(messages)[0]
        lane.frontend.abandon()
        command = h.command(lane, tokens, messages)
        need = CAP.assess(h.lifecycle, command)
        peak = max(need['phase1'][0], need['phase2'][0] + need['watermark'][0])
        idle = sum(len(s.resident) for sid, s in h.adapter.sessions.items() if sid != lane.sid)
        self.assertGreater(idle, 2)
        held = h.pool.get_new_blocks(max(0, need['free'][0] - max(0, peak - idle + 2)))
        free = h.pool.get_num_free_blocks()
        self.assertLess(free, peak)
        evicted, evict = [], h.core._remote_ra_idle_release.evict
        h.core._remote_ra_idle_release.evict = lambda *args: evicted.append(args[0]) or evict(*args)
        gated = CAP.admission_gate(h.lifecycle, command)
        h.core._remote_ra_idle_release.evict = evict
        # Released before the gate recorded its reservation: the turn now fits.
        self.assertTrue(evicted)
        self.assertNotIn(lane.sid, evicted)
        self.assertGreaterEqual(gated['free'][0], peak)
        self.assertEqual(h.pool.get_num_free_blocks(), gated['free'][0])
        self.assertEqual(list(h.lifecycle.reservations), [command.request_id])
        # The turn is never sent: its reservation and the held blocks go.
        CAP.release_reservation(h.lifecycle, command.request_id)
        h.pool.free_blocks(held)
        h.lone_evictions = evicted

    def overlap(self, sessions, turns, permits, fast=ALL3):
        self.switches = None
        reference, _, events = self.run_case(GR.REFERENCE_BLOCKS, None, sessions=sessions, turns=turns)
        self.assertEqual(events, [])
        self.assertTrue(any(self.chunks))
        self.switches = ('full', fast)
        small = GR.selective_blocks(GE.window_layout('goal_query', sessions))
        result, h, events = self.run_case(small, permits, sessions=sessions, turns=turns)
        self.assertEqual(result, reference)
        self.assertTrue(any(self.chunks))
        self.assertTrue([e for e in events if e['event'] == 'evict'] and
                        [e for e in events if e['event'] == 'readmit'])
        self.assertTrue(h.lone_evictions)
        self.assertGreaterEqual(h.parked_peak, 2)
        self.assertIn('allocation_batch', h.operations)
        self.assertNotIn('allocation_update', h.operations)
        self.assertTrue(h.owner.window_rows and h.bridge.decode_fastpath2)
        self.assertEqual(h.owner.window_steady and h.bridge.decode_fastpath3, FP3 in fast)
        self.eager_exactly_for_probes(self.steps, {None, 'phase1', 'goal'})
        return h

    def test_eight_sessions(self):
        self.overlap(GO.SHORT_SESSIONS, 3, GO.SHORT_SESSIONS)

    @unittest.skipUnless(SLOW, 'about 10 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_eight_sessions_with_both_fast_paths(self):
        self.overlap(GO.SHORT_SESSIONS, 3, GO.SHORT_SESSIONS, fast=BOTH)

    @unittest.skipUnless(SLOW, 'about 25 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_sessions(self):
        h = self.overlap(GR.SESSIONS, self.TURNS, GR.SESSIONS)
        self.assertGreater(h.waits, 0)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ReleasePressureTests(Switched, unittest.TestCase):
    """test_gemma_full's (and test_gemma_release's) pressure runs at the release rule's (the
    selective rule's) minimum pool 0 (lane 1's third turn a chunked phase 1, ``long_turns``,
    but in the archive suite's observation_query run) with the decode fast paths (the three in
    the archive suite, both too in the slow pass) and graphs, against --release-idle-pages alone."""
    maxDiff = None
    TEST_GOAL = GG.SixteenGoalSessionsTests.TEST_GOAL
    check, boundary, window_of = GR.SixteenSessionTests.check, GR.SixteenSessionTests.boundary, \
        GR.SixteenSessionTests.window_of

    def setUp(self):
        self.enterContext(patch.dict(sys.modules, W.vllm_stubs()))
        self.enterContext(patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1'))
        self.enterContext(patch.object(GE, 'GOAL', self.TEST_GOAL))

    def pressure(self, policy, sessions, turns, long=True, fasts=(ALL3,)):
        """``long``: lane 1's third turn a chunked phase 1 (``long_turns``); ``fasts``: the decode fast
        paths of the runs compared with --release-idle-pages alone."""
        layout = GE.window_layout(policy, sessions)
        small = GF.release_blocks(layout) if policy == 'full' else GR.selective_blocks(layout)
        chunks = []
        with ExitStack() as stack:
            if long:
                long_turns(stack, chunks)
            alone = GG.pressure_run(self, policy, sessions, turns, small,
                                    make=recording(lambda *args: in_chunks(GR.Pressure(*args))))
        for switched in fasts:
            self.switched(policy, sessions, turns, long, small, alone, chunks, switched)

    def switched(self, policy, sessions, turns, long, small, alone, chunks0, switched):
        """One run with the decode fast paths ``switched`` and graphs against ``alone``."""
        steps, chunks1 = [], []

        def fast(*args):
            h = in_chunks(GR.Pressure(*args))
            attacher('full', switched)(h)
            return h
        with ExitStack() as stack:
            if long:
                long_turns(stack, chunks1)
            stack.enter_context(recorder(steps))
            both = GG.pressure_run(self, policy, sessions, turns, small, make=recording(fast),
                                   extra=flags('full', switched))
        chunks = (chunks0, chunks1)
        self.assertEqual(any(chunks[1]), long)
        self.assertEqual(chunks[1], chunks[0])
        (result, h, events, cleanup), (released, h0, events0, cleanup0) = both, alone
        # The same turns, the same releases and readmissions as --release-idle-pages alone.
        self.assertEqual(result, released)
        strip = lambda events: [{k: v for k, v in e.items() if k != 'monotonic_ns'} for e in events]
        self.assertEqual(strip(events), strip(events0))
        self.assertTrue(any(e['event'] == 'evict' for e in events) and any(e['event'] == 'readmit' for e in events))
        self.assertEqual(cleanup['idle_release'], cleanup0['idle_release'])
        self.assertEqual([i['native']['idle_release'] for i in h.done], [i['native']['idle_release'] for i in h0.done])
        # The fast paths ran: allocation_batch RPCs, no allocation_update; every evict and readmit as alone.
        self.assertIn('allocation_batch', h.operations)
        self.assertNotIn('allocation_update', h.operations)
        self.assertNotIn('allocation_batch', h0.operations)
        for name in ('evict', 'readmit', 'phase1'):
            self.assertEqual(h.operations.count(name), h0.operations.count(name), name)
        self.assertLess(h.operations.count('allocation_batch'), h0.operations.count('allocation_update'))
        self.eager_exactly_for_probes(steps, {None, 'phase1'} | ({'goal'} if policy == 'goal_query' else set()))
        if policy == 'full':
            self.assertIsNone(h.selection.rewriter)
        self.assertEqual(h.owner.window_steady, FP3 in switched)

    def test_full_and_observation_query(self):
        # Five sessions of three turns: releases and readmissions in both runs; full's with a
        # chunked phase 1.
        for policy, long in (('full', True), ('observation_query', False)):
            with self.subTest(policy=policy):
                self.pressure(policy, 5, 3, long)

    @unittest.skipUnless(SLOW, 'about 30 s of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_every_policy_six_sessions(self):
        # test_gemma_goal's and test_gemma_full's scale: six sessions of four turns.
        for policy in ('full', 'observation_query', 'goal_query', 'random_fill'):
            with self.subTest(policy=policy):
                self.pressure(policy, GG.SHORT_SESSIONS, GG.SHORT_TURNS, fasts=(BOTH, ALL3))


if __name__ == '__main__':
    unittest.main()
