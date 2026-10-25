"""Options that meet for the first time on integrate-1014 and integrate-1016. CPU only; not GPU evidence.

integrate-1014 is main 4eaf159 (``--language-model-only``, the 262144-token context,
goals.jsonl) plus decode-speed2 (``--decode-fastpath2``, which needs
``--decode-fastpath``), mab-kvsession (``--protocol kv-session``,
``--release-idle-pages``) and host-pool (``--connector-pool-gib N``). integrate-1016 adds
fastpath3 (``--decode-fastpath3``, which needs ``--decode-fastpath2``) and tb-runtime
(``--protocol chat-text``: a session pool of up to 24 running sessions whose slots bind
lazily to client sessions). Each branch tests its own option; this file covers the
combinations of the five switches (``--decode-fastpath``, ``--decode-fastpath2``,
``--decode-fastpath3``, ``--language-model-only``, ``--release-idle-pages``) on the three
protocols, and ``--connector-pool-gib`` with them:

- the launcher (serve.prepare and the dry-run main), on the XML protocol, on kv-session and
  on chat-text, with the legacy and the decoupled layouts (XML and chat-text on both), a
  session pool (20 chat-text sessions), CUDA graphs, the admission flags (XML and
  chat-text), 4096-token replies, a Page arena, the 262144-token context and the chat-text
  slots and timers: every combination adds exactly the startup keys of its switches
  (and effective_server_config.language_model_only), the vLLM command gains only
  ``--language-model-only``, and each refusal holds whatever else is on
  (``--decode-fastpath2`` without ``--decode-fastpath``, ``--decode-fastpath3`` without
  ``--decode-fastpath2``, ``--release-idle-pages`` on the legacy layout); kv-session and
  chat-text keep their own checks with every switch on;
- ``--connector-pool-gib N`` next to ``--release-idle-pages``, kv-session, chat-text and the
  three decode fast paths: only vLLM's cpu_bytes_to_use and startup.json
  connector_pool_bytes change, the worker startup record does not, and the pinned-host
  budget counts N GiB (16 kv-session sessions at 262144 tokens fit a 124 GiB arena with
  1 GiB, 93 GiB with 32);
- startup: remote_ra_http.config.load accepts every combination and refuses a fast path
  without the one it needs and release on the legacy layout; the frontend's argument
  check; compose_engine sends each switch once in the worker startup record
  and accepts only the worker echo that carries every switch; attach_loaded_worker
  (legacy and decoupled layouts) echoes and attaches each switch and refuses the third
  fast path without the second; ready.json records each (kv-session and chat-text);
- at run time, with the decode fast paths attached to the engine and worker as
  compose_engine and attach_loaded_worker attach them, on vLLM-shaped runners (numpy
  block tables whose rows grow in place: test_decode_fastpath3's for test_decouple_e2e's
  harness, the same tables for test_stageb_worker's FakeRunner), where the incremental
  observation of the second fast path and the steady paths of the third run (on the
  harnesses' own runners they never do):
  - sixteen sessions under GPU pressure (test_idle_release's script, 900 against 9,000
    FullAttention blocks, P = 16) with the first two and with all three decode fast paths
    and ``--release-idle-pages``: every session's prompts, selections, attended Pages and
    replies equal those of ``--release-idle-pages`` alone; so do the releases and
    readmissions; the allocation updates go out as allocation_batch RPCs, before every
    evict and readmit; the third fast path's steady paths and rowless entries run;
  - test_kv_session_e2e's MemAgent sessions with the three decode fast paths;
  - test_chat_text_e2e's sixteen sessions decoding together with the three decode fast
    paths (every check of that test), and chat-text sessions under GPU pressure (eight
    Terminus clients, 260 against 8,000 blocks) with ``--release-idle-pages`` and the three
    fast paths: every client's requests and replies and every client session's prompts
    (the template's render of its messages), selections and attended Pages equal those
    of a pool that never releases; idle sessions are released and readmitted, each
    readmission restores exactly what its session's last eviction released;
  - ``--admission-overlap K`` with ``--release-idle-pages`` (OverlapReleaseTests): eight
    goal_query sessions of four turns (Goals parked on their session threads, several
    admissions in flight) under pool pressure (600 FullAttention blocks), with and without
    the three decode fast paths: every turn equals the K = 1 server's with a pool that never
    releases (9,000 blocks); and the lone turn (nothing runs, nothing is in flight, every
    session idle with its Pages pinned, a turn whose phase 1 fits only once idle Pages are
    released), which the ledger gate admits unchecked: the release runs first (0535f79's
    idle_release released nothing there and the turn's phase 1 failed, as does the pressure
    run).
  The suite runs observation_query (pressure), goal_query (MemAgent), full (sixteen
  chat-text sessions), observation_query (chat-text pressure) and, with K = 4, the overlap
  pressure run with the three fast paths and the lone turn without them; the slow pass
  (REMOTE_RA_SLOW_TESTS=1, README section 2.3) adds full, the large-pool reference, the
  runs on test_decouple_e2e's and test_kv_session_e2e's own runners (the integrate-1014
  runs), the MemAgent sessions with the first two fast paths and the full and
  observation_query ones, test_chat_text_e2e's other scenarios with the three fast paths,
  chat-text pressure with ``--release-idle-pages`` alone and with Terminus's summary
  hand-offs (asides, the reaper, and rebuilds of released sessions, which are not
  readmitted), and the overlap runs' other halves (pressure without and with the first two
  fast paths, K = 1 and K = 8, the lone turn with the three, sixteen sessions with K = 16),
  which keeps the archive suite within check_git_release.py's time limit.
"""
import asyncio
import contextlib
from contextlib import ExitStack
import copy
import io
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source-overlay'))
sys.path.insert(0, str(ROOT))

import serve  # noqa: E402
import test_align_ctx as AC  # noqa: E402
import test_cudagraph as TCG  # noqa: E402
import test_language_model_only as LMO  # noqa: E402

from remote_ra_http import config as http_config  # noqa: E402
from remote_ra_live_v2 import chat_text_protocol as C  # noqa: E402
from remote_ra_live_v2 import decode_fastpath as F  # noqa: E402
from remote_ra_live_v2 import decode_fastpath2 as F2  # noqa: E402
from remote_ra_live_v2 import decode_fastpath3 as F3  # noqa: E402
from remote_ra_live_v2 import idle_release as R  # noqa: E402
from remote_ra_live_v2 import kv_layout as I0  # noqa: E402
from remote_ra_live_v2 import kv_session_protocol as K  # noqa: E402
from remote_ra_live_v2 import language_model_only as T  # noqa: E402
from remote_ra_live_v2.contract import LiveV2Error  # noqa: E402
# Imported before any test patches sys.modules (see test_align_ctx).
from remote_ra_live_v2 import loaded_sources, native_probe, vector_space, worker_composition as WC  # noqa: E402,F401

try:
    import torch
except ImportError:
    torch = None

tasks = AC.tasks
SLOW = os.environ.get('REMOTE_RA_SLOW_TESTS') == '1'
SWITCHES = ('decode_fastpath', 'decode_fastpath2', 'decode_fastpath3', 'language_model_only', 'release_idle_pages')
FLAG = {name: '--' + name.replace('_', '-') for name in SWITCHES}
TEXT_ONLY = '--language-model-only'
COMBINATIONS = [c for n in range(len(SWITCHES) + 1) for c in itertools.combinations(SWITCHES, n)]
FAST = ('decode_fastpath', 'decode_fastpath2', 'decode_fastpath3')

# name: task, policy, protocol, KV GiB, P, layout, selection, other launcher options
BASES = {
    'xml-legacy': (tasks(1)[0], 'full', 'mswea-xml', 8, 784, None, None, {}),
    'xml-pool-graphs': (tasks(8), 'observation_query', 'mswea-xml', 24, 16, 'decoupled',
                        dict(active_token_budget=8192), dict(max_running=4, cuda_graphs='full')),
    'xml-admission': (tasks(4), 'goal_query', 'mswea-xml', 24, 16, 'decoupled', dict(no_goal_rewriter=True),
                      dict(gdn_device_handoff=True, phase1_in_step=True, admission_overlap=4, max_tokens=4096)),
    'kv-obs-262144': (None, 'observation_query', 'kv-session', 32, 16, 'decoupled',
                      dict(active_token_budget=32768),
                      dict(max_running=16, max_tokens=4096, max_model_len=262144, dram_arena_gib=88)),
    'kv-goal': (None, 'goal_query', 'kv-session', 24, 16, None, dict(no_goal_rewriter=True, active_token_budget=32768),
                dict(max_running=4, kv_session_slots=8, kv_session_thinking='off')),
    # README section 6.7's Terminal-Bench launch (its --connector-pool-gib 1 is ConnectorPoolTests'), with CUDA graphs.
    'chat-k20': (None, 'observation_query', 'chat-text', 32, 16, 'decoupled', dict(active_token_budget=8192),
                 dict(max_running=20, max_tokens=4096, dram_arena_gib=64, cuda_graphs='full')),
    'chat-legacy-admission': (None, 'random50', 'chat-text', 16, 784, None, None,
                              dict(max_running=4, chat_text_slots=8, chat_text_idle_timeout=120, chat_text_evict_after=5,
                                   gdn_device_handoff=True, phase1_in_step=True, admission_overlap=4)),
}


def fast_refusal(combination):
    """The launcher's refusal of a combination's decode fast paths, else None (in its order)."""
    if 'decode_fastpath2' in combination and 'decode_fastpath' not in combination:
        return '--decode-fastpath2 requires --decode-fastpath'
    if 'decode_fastpath3' in combination and 'decode_fastpath2' not in combination:
        return '--decode-fastpath3 requires --decode-fastpath2'
    return None


def refusal(base, combination):
    """The launcher's refusal of a combination, else None."""
    refused = fast_refusal(combination)
    if refused is None and 'release_idle_pages' in combination and BASES[base][4] == 784 and BASES[base][5] is None:
        refused = '--release-idle-pages needs the decoupled KV layout'
    return refused


def expected_config(config, combination):
    """The startup config of a combination: the base config plus each switch's own pins."""
    value = dict(config, **{name: True for name in combination})
    if 'language_model_only' in combination:
        effective = dict(config['effective_server_config'], language_model_only=True)
        value.update(effective_server_config=effective, server_config_sha256=serve.pin(effective))
    return value


def pins(config):
    """Each switch's pin of a loaded startup config (True, or None when off)."""
    return (F.pinned(config), F2.pinned(config), F3.pinned(config), T.pinned(config), R.pinned(config))


class Release(AC.Release):
    """test_align_ctx's release fixture with the protocol of each base."""

    def launch(self, base, combination=(), out=None, **extra):
        task, policy, protocol, gib, P, layout, selection, options = BASES[base]
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            return serve.prepare(self.release, copy.deepcopy(task), policy, self.root / (out or base), 8231,
                                 copy.deepcopy(selection), protocol, gib, P, layout,
                                 **dict(options, **extra, **{name: True for name in combination}))

    def launched(self, base, combination=(), **extra):
        try:
            return ('ok', self.launch(base, combination, **extra))
        except Exception as error:
            return (type(error).__name__, str(error))

    def configs(self):
        """(base, combination, config) of every combination the launcher accepts (launched once
        per class: a dry launch creates nothing)."""
        cls = type(self)
        if '_configs' not in cls.__dict__:
            cls._configs = [(base, combination, self.launch(base, combination)[1])
                            for base in BASES for combination in COMBINATIONS if refusal(base, combination) is None]
        return cls._configs


def engine_record(config, echo=None):
    """compose_engine's worker startup record for a startup config, and its refusal of the
    worker's reply ``echo`` (None: the record is captured and nothing replies)."""
    from remote_ra_live_v2 import composition

    class Stop(Exception):
        pass
    sent = []

    def collective_rpc(method, args, timeout):
        sent.append(args[0])
        if echo is None:
            raise Stop()
        return [dict(echo)]
    listed = config.get('tasks') or [config['task']]
    layout = None
    if 'kv_layout' in config:
        record = config['kv_layout']
        layout = I0.KvLayout(record['page_tokens'], record['mamba_tokens'], record['mamba_slots'], True)
    core = NS(_remote_ra_live_v2={}, _remote_ra_assembly=NS(provider=NS(provenance=NS(
                  page_tokens=config['kv_layout']['page_tokens'] if layout else 784))),
              _remote_ra_trusted_issue=dict(listed[0]), _remote_ra_trusted_issues=[dict(t) for t in listed],
              _remote_ra_growing_owner=NS(identity=NS(manifest='m' * 64)),
              _remote_ra_growing_adapter=NS(layout=layout), model_executor=NS(collective_rpc=collective_rpc))
    error = None
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'startup.json'
        path.write_bytes(serve.wire(config))
        # The 262144-token request caps are module state (test_ctx_256k covers them): not set here.
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                        REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)), \
                patch('remote_ra_live_v2.context_limits.pin_request_caps', lambda config, core=None: None):
            try:
                composition.compose_engine(core, object())
            except Stop:
                pass
            except LiveV2Error as caught:
                error = str(caught)
    startup, = sent
    return startup, error


# ---- 1. the launcher ----------------------------------------------------------------------------

XML_ARGV = ['--policy', 'observation_query', '--page-tokens', '16', '--kv-layout', 'decoupled', '--kv-cache-gib', '24',
            '--max-running-sessions', '4', '--active-token-budget', '8192']
KV_ARGV = ['--protocol', 'kv-session', '--policy', 'observation_query', '--page-tokens', '16', '--kv-layout',
           'decoupled', '--active-token-budget', '32768', '--max-running-sessions', '16', '--kv-cache-gib', '32',
           '--max-tokens', '4096', '--max-model-len', '262144', '--dram-arena-gib', '88']
CHAT_ARGV = ['--protocol', 'chat-text', '--policy', 'observation_query', '--page-tokens', '16', '--kv-layout',
             'decoupled', '--active-token-budget', '8192', '--max-running-sessions', '20', '--kv-cache-gib', '32',
             '--max-tokens', '4096', '--dram-arena-gib', '64', '--cuda-graphs', 'full']


class LauncherTests(Release, unittest.TestCase):
    maxDiff = None

    def test_every_combination_adds_exactly_its_switches(self):
        accepted = 0
        for base in BASES:
            status, plain = self.launched(base)
            self.assertEqual(status, 'ok', (base, plain))
            manifest, config, command, env = plain
            for name in SWITCHES:
                self.assertNotIn(name, config)
            self.assertNotIn(TEXT_ONLY, command)
            for combination in COMBINATIONS[1:]:
                with self.subTest(base=base, combination=combination):
                    outcome = self.launched(base, combination)
                    refused = refusal(base, combination)
                    if refused is not None:
                        self.assertEqual(outcome[0], 'ValueError')
                        self.assertIn(refused, outcome[1])
                        continue
                    self.assertEqual(outcome[0], 'ok', outcome)
                    m1, c1, a1, e1 = outcome[1]
                    self.assertEqual(m1, manifest)
                    self.assertEqual(c1, expected_config(config, combination))
                    self.assertEqual(a1, command + ([TEXT_ONLY] if 'language_model_only' in combination else []))
                    self.assertEqual(e1, dict(env, REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(c1)))
                    accepted += 1
        # 16 valid combinations (4 of the fast paths' 8, times 4); 8 on the two legacy bases (release refused).
        self.assertEqual(accepted, 5 * 15 + 2 * 7)

    def test_kv_session_keeps_its_own_checks_with_every_switch(self):
        everything = SWITCHES
        _, config, _, _ = self.launch('kv-obs-262144', everything)
        self.assertEqual((config['protocol'], config['tasks'], config['max_running_sessions']),
                         ('kv_session_v1', K.slot_tasks(64), 16))
        self.assertEqual(config['kv_session'], dict(method='dram_backed_kv', budget=32768, thinking='on',
                                                    header_text='<|im_start|>assistant\n<think>\n', slots=64))
        task, policy, protocol, gib, P, layout, selection, options = BASES['kv-obs-262144']
        switches = {name: True for name in everything}
        for change, message in ((dict(task=tasks(1)[0]), 'takes no --task'),
                                (dict(max_running=None), 'needs --max-running-sessions'),
                                (dict(kv_session_slots=65), '--kv-session-slots must be')):
            with self.subTest(message=message):
                values = dict(options, **switches)
                values.update({k: v for k, v in change.items() if k != 'task'})
                with self.assertRaisesRegex(ValueError, message), contextlib.redirect_stdout(io.StringIO()), \
                        patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                    serve.prepare(self.release, change.get('task'), policy, self.root / 'kv-refused', 8231,
                                  dict(selection), protocol, gib, P, layout, **values)
        with self.assertRaisesRegex(ValueError, 'apply to --protocol kv-session only'), \
                contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
            serve.prepare(self.release, tasks(1)[0], 'full', self.root / 'xml-kv', 8231, None, 'mswea-xml', 24, 16,
                          None, kv_session_slots=8, **switches)

    def test_chat_text_keeps_its_own_checks_with_every_switch(self):
        _, config, command, _ = self.launch('chat-k20', SWITCHES)
        self.assertEqual((config['protocol'], config['tasks'], config['task'], config['max_running_sessions']),
                         ('chat_text_v1', C.slot_tasks(512), C.slot_tasks(1)[0], 20))
        self.assertEqual(config['chat_text'], C.record())
        self.assertEqual((command[command.index('--max-num-seqs') + 1], config['kv_layout']['mamba_slots']),
                         ('20', 41))
        task, policy, protocol, gib, P, layout, selection, options = BASES['chat-k20']
        switches = {name: True for name in SWITCHES}
        # The session pool's bound is sessions.running_limit's (a LiveV2Error), the rest the launcher's.
        for change, error, message in ((dict(task=tasks(1)[0]), ValueError, 'takes no --task'),
                                       (dict(max_running=None), ValueError, 'needs --max-running-sessions'),
                                       (dict(max_running=25), LiveV2Error, 'max_running_sessions must be 1..24'),
                                       (dict(chat_text_slots=1025), ValueError, r'--chat-text-slots must be 1\.\.1024'),
                                       (dict(chat_text_slots=8), LiveV2Error, 'at most the number of tasks'),
                                       (dict(chat_text_idle_timeout=20, chat_text_evict_after=21), ValueError,
                                        'evict-after'),
                                       (dict(policy='goal_query'), ValueError, 'SWE-centric')):
            with self.subTest(message=message):
                values = dict(options, **switches)
                values.update({k: v for k, v in change.items() if k not in ('task', 'policy')})
                chosen = dict(selection, no_goal_rewriter=True) if 'policy' in change else dict(selection)
                with self.assertRaisesRegex(error, message), contextlib.redirect_stdout(io.StringIO()), \
                        patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                    serve.prepare(self.release, change.get('task'), change.get('policy', policy),
                                  self.root / 'chat-refused', 8231, chosen, protocol, gib, P, layout, **values)
        for protocol_, task_, running in (('mswea-xml', tasks(1)[0], None), ('kv-session', None, 4)):
            with self.subTest(protocol=protocol_), \
                    self.assertRaisesRegex(ValueError, 'apply to --protocol chat-text only'), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                serve.prepare(self.release, task_, 'full', self.root / 'other-chat', 8231, None, protocol_, 24, 16,
                              None, max_running=running, chat_text_slots=8, **switches)

    def test_dry_run_main(self):
        task = self.root / 'tasks8.json'
        task.write_text(json.dumps(tasks(8)))
        xml = ['--task', str(task)] + XML_ARGV

        def main(*argv):
            out = io.StringIO()
            with patch.object(sys, 'argv', ['serve.py', '--release-dir', str(self.release), '--output',
                                            str(self.root / 'dry'), '--port', '8231', *argv]), \
                    contextlib.redirect_stdout(out), \
                    patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                serve.main()
            return json.loads(out.getvalue())
        for name, argv in (('xml', xml), ('kv-session', KV_ARGV), ('chat-text', CHAT_ARGV)):
            plain = main(*argv)
            self.assertEqual(plain['status'], 'SERVICE_DRY_RUN')
            for combination in COMBINATIONS[1:]:
                flags = [FLAG[n] for n in combination]
                with self.subTest(protocol=name, flags=flags):
                    refused = fast_refusal(combination)
                    if refused is not None:
                        with self.assertRaisesRegex(ValueError, refused):
                            main(*argv, *flags)
                        continue
                    dry = main(*argv, *flags)
                    text_only = [TEXT_ONLY] if 'language_model_only' in combination else []
                    self.assertEqual(dry, dict(plain, command=plain['command'] + text_only))


# ---- 1b. --connector-pool-gib (host-pool) with the other options ----------------------------------

GIB = 1024 ** 3
TRANSFER = '--kv-transfer-config'
POOL_COMBINATIONS = ((), ('release_idle_pages',), ('decode_fastpath', 'decode_fastpath2'), ('decode_fastpath2',),
                     ('decode_fastpath', 'decode_fastpath2', 'release_idle_pages'), FAST,
                     ('decode_fastpath', 'decode_fastpath3'), SWITCHES)


def with_pool(argv, gib):
    """A vLLM command with the connector's CPU pool set to ``gib`` GiB (--kv-transfer-config only)."""
    at = argv.index(TRANSFER) + 1
    value = json.loads(argv[at])
    value['kv_connector_extra_config'] = {'cpu_bytes_to_use': gib * GIB}
    return argv[:at] + [json.dumps(value, separators=(',', ':'))] + argv[at + 1:]


class ConnectorPoolTests(Release, unittest.TestCase):
    """--connector-pool-gib N changes only the connector's pool (vLLM's cpu_bytes_to_use, startup.json
    connector_pool_bytes, the launcher's pinned-host budget) next to --release-idle-pages, the
    kv-session and chat-text protocols and the three decode fast paths; it is not part of the
    worker startup record."""
    maxDiff = None

    def test_pool_with_the_other_switches(self):
        from remote_ra_live_v2 import connector_guard as G
        for base in BASES:
            m0, c0, a0, e0 = self.launch(base, out='pool')
            for combination in POOL_COMBINATIONS:
                refused = refusal(base, combination)
                if refused is None:
                    # The worker startup record of the combination without the pool.
                    plain = engine_record(self.launch(base, combination, out='pool')[1])
                for n in (1, 31):
                    with self.subTest(base=base, combination=combination, n=n):
                        outcome = self.launched(base, combination, out='pool', connector_pool_gib=n)
                        if refused is not None:
                            self.assertEqual(outcome[0], 'ValueError')
                            self.assertIn(refused, outcome[1])
                            continue
                        self.assertEqual(outcome[0], 'ok', outcome)
                        m1, c1, a1, e1 = outcome[1]
                        self.assertEqual(m1, m0)
                        self.assertEqual(c1, dict(expected_config(c0, combination), connector_pool_bytes=n * GIB))
                        text_only = [TEXT_ONLY] if 'language_model_only' in combination else []
                        self.assertEqual(a1, with_pool(a0 + text_only, n))
                        self.assertEqual(e1, dict(e0, REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(c1)))
                        loaded = self.load(c1)
                        self.assertEqual(G.pinned_pool(loaded), n * GIB)
                        self.assertEqual(pins(loaded), tuple(True if s in combination else None for s in SWITCHES))
                        # The engine sends the worker the same startup record as without the pool.
                        self.assertEqual(engine_record(c1), plain)

    def test_the_budget_with_kv_session_and_release(self):
        # 16 kv-session sessions at 262144 tokens with every switch: the largest arena that fits
        # grows by the pool's 31 GiB (93 -> 124 GiB), as without the switches.
        with self.assertRaisesRegex(ValueError, r'\(the 32 GiB connector pool, .*\(at most 93 GiB\)$'):
            self.launch('kv-obs-262144', SWITCHES, dram_arena_gib=124)
        config = self.launch('kv-obs-262144', SWITCHES, dram_arena_gib=124, connector_pool_gib=1)[1]
        self.assertEqual((config['connector_pool_bytes'], config['release_idle_pages'], config['protocol']),
                         (GIB, True, 'kv_session_v1'))
        with self.assertRaisesRegex(ValueError, r'\(the 1 GiB connector pool, .*\(at most 124 GiB\)$'):
            self.launch('kv-obs-262144', SWITCHES, dram_arena_gib=126, connector_pool_gib=1)
        with self.assertRaisesRegex(ValueError, '--connector-pool-gib must be an integer 1-31'):
            self.launch('kv-goal', ('release_idle_pages', 'decode_fastpath', 'decode_fastpath2'),
                        connector_pool_gib=32)
        with self.assertRaisesRegex(ValueError, '--connector-pool-gib must be an integer 1-31'):
            self.launch('chat-k20', SWITCHES, connector_pool_gib=0)

    def test_dry_run_main(self):
        task = self.root / 'pool-tasks8.json'
        task.write_text(json.dumps(tasks(8)))
        xml = ['--task', str(task)] + XML_ARGV

        def main(*argv):
            out = io.StringIO()
            with patch.object(sys, 'argv', ['serve.py', '--release-dir', str(self.release), '--output',
                                            str(self.root / 'pool-dry'), '--port', '8231', *argv]), \
                    contextlib.redirect_stdout(out), \
                    patch.object(sys, 'path', [str(ROOT / 'source-overlay')] + sys.path):
                serve.main()
            return json.loads(out.getvalue())
        pool = ('--connector-pool-gib', '1')
        fast = tuple(FLAG[n] for n in FAST)
        for argv, flags in ((KV_ARGV, ('--release-idle-pages',)),
                            (KV_ARGV, ('--decode-fastpath', '--decode-fastpath2', '--release-idle-pages',
                                       TEXT_ONLY)),
                            (KV_ARGV, fast + ('--release-idle-pages',)),
                            (xml, ('--decode-fastpath', '--decode-fastpath2')),
                            (xml, ('--decode-fastpath', '--decode-fastpath2', '--release-idle-pages')),
                            (xml, fast),
                            (CHAT_ARGV, ('--release-idle-pages',)),
                            (CHAT_ARGV, fast),
                            (CHAT_ARGV, fast + ('--release-idle-pages', TEXT_ONLY))):
            with self.subTest(protocol=argv[1], flags=flags):
                plain = main(*argv, *flags)
                dry = main(*argv, *pool, *flags)
                self.assertEqual(dry, dict(plain, command=with_pool(plain['command'], 1)))
        with self.assertRaisesRegex(ValueError, 'requires --decode-fastpath'):
            main(*xml, *pool, '--decode-fastpath2')
        with self.assertRaisesRegex(ValueError, '--decode-fastpath3 requires --decode-fastpath2'):
            main(*CHAT_ARGV, *pool, '--decode-fastpath', '--decode-fastpath3')


# ---- 2. startup ---------------------------------------------------------------------------------

class StartupTests(Release, unittest.TestCase):
    maxDiff = None

    def test_load_and_the_frontend_check_accept_every_combination(self):
        from remote_ra_live_v2 import cuda_graphs as CG
        count = 0
        for base, combination, config in self.configs():
            with self.subTest(base=base, combination=combination):
                with self.startup(config):
                    loaded = http_config.load()
                    effective = dict(config['effective_server_config'])
                    if 'compilation_config' in effective:
                        effective['compilation_config'] = CG.observed(effective['compilation_config'])
                    self.assertEqual(http_config.effective_server_config(NS(**effective)),
                                     config['effective_server_config'])
                    self.assertEqual((F.loaded(), F2.loaded(), F3.loaded(), T.loaded(), R.loaded()),
                                     tuple(True if n in combination else None for n in SWITCHES))
                # load() also lists a one-task config's task under 'tasks'.
                self.assertEqual(loaded, dict(config, tasks=config.get('tasks') or [config['task']]))
                self.assertEqual(pins(loaded), tuple(True if n in combination else None for n in SWITCHES))
                if BASES[base][2] == 'kv-session':
                    self.assertEqual(K.check_config(loaded), loaded['kv_session'])
                if BASES[base][2] == 'chat-text':
                    self.assertEqual(C.check_config(loaded), loaded['chat_text'])
                count += 1
        self.assertEqual(count, 5 * 16 + 2 * 8)

    def test_load_refuses_what_the_launcher_refuses(self):
        # Startup configs the launcher never writes (their pin recomputed, so only the check
        # refuses them): a fast path without the one it needs; release on the legacy layout.
        for base in BASES:
            config = self.launch(base, FAST)[1]
            for dropped, message in ((('decode_fastpath',), 'decode_fastpath2 requires decode_fastpath'),
                                     (('decode_fastpath2',), 'decode_fastpath3 requires decode_fastpath2'),
                                     (FAST[:2], 'decode_fastpath3 requires decode_fastpath2')):
                with self.subTest(base=base, dropped=dropped), \
                        self.startup({k: v for k, v in config.items() if k not in dropped}), \
                        self.assertRaisesRegex(RuntimeError, message):
                    http_config.load()
            if refusal(base, ('release_idle_pages',)) is not None:
                with self.subTest(base=base, release=True), self.startup(dict(config, release_idle_pages=True)), \
                        self.assertRaisesRegex(RuntimeError, 'release_idle_pages needs the decoupled KV layout'):
                    http_config.load()

    def test_compose_engine_sends_each_switch_once(self):
        records = {(base, combination): engine_record(config) for base, combination, config in self.configs()}
        for (base, combination), (record, error) in records.items():
            with self.subTest(base=base, combination=combination):
                self.assertIsNone(error)
                self.assertEqual(record, dict(records[base, ()][0], **{name: True for name in combination}))

    def test_compose_engine_accepts_only_the_echo_of_every_switch(self):
        receipt_check = 'loaded CUDA worker attachment receipt'     # the first check after the echoes
        for base in ('xml-pool-graphs', 'kv-obs-262144', 'chat-k20'):
            config = self.launch(base, SWITCHES)[1]
            record = engine_record(config)[0]
            echo = {key: record[key] for key in ('context', 'admission') if key in record}
            echo.update({name: True for name in SWITCHES}, language_model_only=T.RECORD)
            with self.subTest(base=base):
                self.assertIn(receipt_check, engine_record(config, echo)[1])
            for name in SWITCHES:
                with self.subTest(base=base, missing=name):
                    partial = {k: v for k, v in echo.items() if k != name}
                    self.assertIn('another %s' % name, engine_record(config, partial)[1])


class Arena:
    """remote_ra_growing_worker.tensors.PageArena as attach_loaded_worker uses it."""

    def __init__(self, views, max_pages):
        self.max_pages = max_pages

    def stats(self):
        return dict(max_pages=self.max_pages)


class ArenaIO(TCG.FakeIO):
    def attach_arena(self, arena):
        self.arena = arena


def attach(startup_extra=None, decoupled=False, **runner_extra):
    """test_language_model_only's attach_loaded_worker fixture; ``decoupled``: a P = 16 layout with
    sixteen sessions' Mamba slots and a pinned arena of 64 Pages (the arena itself faked)."""
    from remote_ra_growing_worker import tensors
    views = TCG.AttachLoadedWorkerTests().views()
    layout = I0.KvLayout(16, 784, 33, True) if decoupled else I0.legacy()
    runner = NS(_remote_ra_growing_view_inputs=('kv', (784,)), kv_cache_config=NS(group_pool=None),
                execute_model=lambda *args: None, compilation_config=TCG.resolved_config(None),
                _selective_cudagraph_groups=frozenset(), **runner_extra)
    worker = NS(model_runner=runner)
    startup = dict(enabled=True, source='m' * 64, page_tokens=layout.page_tokens, loaded_sources_verified=True,
                   **(startup_extra or {}))
    if decoupled:
        startup.update(kv_layout=I0.to_additional_config(layout)[I0.LAYOUT_KEY], dram_arena_pages=64)
    transfer = ModuleType('vllm.distributed.kv_transfer')
    transfer.get_kv_transfer_group = lambda: 'connector'
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {'vllm.distributed.kv_transfer': transfer}))
        stack.enter_context(patch.object(tensors, 'TensorIO', ArenaIO))
        stack.enter_context(patch.object(tensors, 'PageArena', Arena))
        stack.enter_context(patch.object(tensors, 'views_from_runner', lambda runner, kv, sizes: views))
        stack.enter_context(patch.object(WC, 'worker_layout', lambda runner, startup: layout))
        stack.enter_context(patch.object(WC, 'check_view_geometry', lambda *args: None))
        stack.enter_context(patch.object(WC, 'RunnerAllocationOwner', LMO.FakeOwner))
        stack.enter_context(patch.object(WC, 'CudaWorkerBackend', TCG.FakeBackend))
        stack.enter_context(patch.object(WC, 'QwenAlignLifecycle', TCG.FakeQwen))
        stack.enter_context(patch.object(loaded_sources, 'verify_closure', lambda runner: {'closure': 'ok'}))
        stack.enter_context(patch.object(vector_space, 'attach', lambda runner: {'hidden_size': 8}))
        stack.enter_context(patch.object(native_probe, 'NativeQueryForward', lambda *args: 'probe'))
        evidence = WC.attach_loaded_worker(worker, startup)
    return evidence, runner


ECHOED = dict(decode_fastpath=True, decode_fastpath2=True, decode_fastpath3=True, language_model_only=T.RECORD,
              release_idle_pages=True)


class WorkerTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        patcher = patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='0')
        patcher.start()
        self.addCleanup(patcher.stop)

    @unittest.skipUnless(torch is not None, 'CPU torch required for the Page compactor')
    def test_attach_echoes_and_attaches_every_switch(self):
        for decoupled in (False, True):
            plain = attach(decoupled=decoupled)[0]
            for combination in COMBINATIONS:
                with self.subTest(decoupled=decoupled, combination=combination):
                    runner_extra = LMO.text_runner() if 'language_model_only' in combination else {}
                    startup = {name: True for name in combination}
                    if fast_refusal(combination) is not None:
                        # The record check (decode_fastpath2/3.worker_flag) or attach_loaded_worker's.
                        with self.assertRaisesRegex(LiveV2Error, 'needs decode_fastpath'):
                            attach(startup, decoupled, **runner_extra)
                        continue
                    if 'release_idle_pages' in combination and not decoupled:
                        with self.assertRaisesRegex(LiveV2Error, 'needs the decoupled KV layout'):
                            attach(startup, decoupled, **runner_extra)
                        continue
                    evidence, runner = attach(startup, decoupled, **runner_extra)
                    self.assertEqual(evidence, dict(plain, **{name: ECHOED[name] for name in combination}))
                    owner, bridge = runner._remote_ra_live_v2_allocation_owner, runner._remote_ra_live_v2_bridge
                    self.assertEqual(getattr(owner, 'release_idle', False), 'release_idle_pages' in combination)
                    self.assertEqual(bridge.decode_fastpath, 'decode_fastpath' in combination)
                    self.assertEqual(bridge.decode_fastpath2, 'decode_fastpath2' in combination)
                    self.assertEqual((bridge.decode_fastpath3, getattr(owner, 'decode_fastpath3', False),
                                      getattr(bridge.qwen, 'decode_fastpath3', False)),
                                     ('decode_fastpath3' in combination,) * 3)
                    self.assertEqual(isinstance(owner.claims, F.ClaimTable), 'decode_fastpath' in combination)


class ReadyTests(Release, unittest.TestCase):
    maxDiff = None

    def test_ready_records_every_switch(self):
        import pathlib
        import test_ms_frontend as MF
        from remote_ra_http.evidence import FrontendEvidence
        readiness = dict(native_ready=True)
        authority = NS(identity=NS(epoch=MF.EPOCH, manifest='m' * 64), policy='observation_query',
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
        for base in ('kv-obs-262144', 'chat-k20'):
            written = {}
            for combination in ((), SWITCHES):
                config = self.load(self.launch(base, combination, out='ready')[1])
                record = engine_record(config)[0]
                receipt = dict(tensor_geometry={}, source_closure={}, context=record['context'],
                               **{name: ECHOED[name] for name in combination})
                missing = [None] + list(combination)
                for drop in missing:
                    with self.subTest(base=base, combination=combination, drop=drop), \
                            tempfile.TemporaryDirectory() as directory:
                        output = Path(directory)
                        value = dict(config, output_directory=str(output))
                        sent = {k: v for k, v in receipt.items() if k != drop}
                        (output / 'worker-attached.json').write_bytes(json.dumps(dict(
                            epoch=MF.EPOCH, report=readiness, receipt=sent,
                            kv_pools=dict(kv_layout=config['kv_layout'], num_blocks_by_pool=[4, 4],
                                          free_blocks_by_pool=[4, 4]))).encode())
                        with patch('remote_ra_http.evidence.load', lambda: value), \
                                patch.object(pathlib.Path, 'read_text', proc):
                            evidence = FrontendEvidence(authority, None, dict(config['effective_server_config']))
                            app = NS(routes=[NS(path='/v1/chat/completions', methods={'POST'})])
                            if drop is not None:
                                with self.assertRaisesRegex(Exception, 'worker/parent %s mismatch' % drop):
                                    evidence.write_ready(app, NS(started=True))
                                continue
                            evidence.write_ready(app, NS(started=True))
                        written[combination] = json.loads((output / 'ready.json').read_bytes())
            plain, everything = written[()], written[SWITCHES]
            for name in SWITCHES:
                self.assertNotIn(name, plain)
                self.assertEqual(everything[name], ECHOED[name])
            self.assertIs(everything['effective_server_config']['language_model_only'], True)
            self.assertEqual((everything['protocol'], len(everything['sessions'])),
                             (config['protocol'], len(config['tasks'])))
            differ = ('effective_server_config', 'server_config_sha256', 'server_run_id', 'worker_receipt_sha256')
            self.assertEqual({k: v for k, v in everything.items() if k not in differ + SWITCHES},
                             {k: v for k, v in plain.items() if k not in differ})


# ---- 3. run time: the decode fast paths with --release-idle-pages, kv-session and chat-text ------

def attach_fast(h, fast):
    """attach_loaded_worker's and compose_engine's decode fast-path attachments on a harness:
    ``fast`` 2, the first two (test_decode_fastpath2); 3, all three (test_decode_fastpath3);
    0, none. With the third, ``h.fast3`` counts its steady paths (materialize skipped, the
    step lease kept)."""
    import test_decode_fastpath2 as TF2
    import test_decode_fastpath3 as TF3
    if fast == 2:
        F.attach_worker(h.owner, h.bridge, h.qwen)
        TF2.attach_fast2(h)
    elif fast == 3:
        TF3.attach(h, True)
        h.fast3 = counts = dict(steady=0, lease=0)
        steady, lease, fallback = h.owner.materialize_steady, h.qwen.before_forward3, h.qwen.before_forward

        def counted_steady(s):
            done = steady(s)
            counts['steady'] += bool(done)
            return done

        def counted_lease(*args):
            calls = []
            h.qwen.before_forward = lambda *a, **k: calls.append(1) or fallback(*a, **k)
            try:
                return lease(*args)
            finally:
                h.qwen.before_forward = fallback
                counts['lease'] += not calls
        h.owner.materialize_steady, h.qwen.before_forward3 = counted_steady, counted_lease
    else:
        assert fast == 0, fast


def record_rpcs(h):
    """``h.operations``: every engine-to-worker RPC's operation, in order."""
    h.operations = operations = []
    rpc = h.proxy.executor.collective_rpc

    def collective_rpc(method, args, timeout):
        operations.append(args[0])
        return rpc(method, args, timeout)
    h.proxy.executor = NS(collective_rpc=collective_rpc)


@contextlib.contextmanager
def rowless_counted():
    """A count of the allocation_batch entries decode_fastpath3.rowless_entry answered."""
    counts = dict(rowless=0)
    rowless = F3.rowless_entry

    def counted(*args):
        receipt = rowless(*args)
        counts['rowless'] += receipt is not None
        return receipt
    with patch.object(F3, 'rowless_entry', counted):
        yield counts


def pressure(fast, shaped=True):
    """test_idle_release's sixteen-session pressure run with the decode fast paths ``fast``
    (attach_fast); ``shaped``: on test_decode_fastpath3's vLLM-shaped runner (numpy block
    tables grown in place, MRoPE positions), else on test_decouple_e2e's own."""
    import test_decode_fastpath3 as TF3
    import test_decouple_e2e as DE
    import test_idle_release as IR

    class Rows16(TF3.NpTable):
        """A row for each of the sixteen sessions decoding together."""

        def __init__(self, rows, width, block_size):
            super().__init__(max(rows, 16), width, block_size)

    class Pressure(IR.PressureTests):
        def enter(self, stack, policy, blocks):
            if shaped:
                stack.enter_context(patch.object(TF3, 'NpTable', Rows16))
                stack.enter_context(patch.object(DE, 'Runner', TF3.vllm_runner(True)))
            self.rowless = stack.enter_context(rowless_counted())
            return IR.PressureTests.enter(self, stack, policy, blocks)

        def harness(self, policy, case, output, release):
            h = IR.PressureTests.harness(self, policy, case, output, release)
            attach_fast(h, fast)
            record_rpcs(h)
            return h
    return Pressure('test_sixteen_sessions_full')


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class PressureTests(unittest.TestCase):
    maxDiff = None

    def compare(self, fast, released):
        """``fast`` (a run with decode fast paths and --release-idle-pages) against ``released``
        (--release-idle-pages alone on the same runner)."""
        # Every session saw, decided and attended what it did with release alone (which
        # equals the large pool in test_idle_release; the slow pass runs the large pool here too).
        self.assertEqual(fast[0], released[0])
        # The same releases and readmissions as --release-idle-pages alone.
        strip = lambda events: [{k: v for k, v in e.items() if k != 'monotonic_ns'} for e in events]
        self.assertEqual(strip(fast[2]), strip(released[2]))
        self.assertTrue(any(e['event'] == 'evict' for e in fast[2]) and
                        any(e['event'] == 'readmit' for e in fast[2]))
        self.assertEqual([n['idle_release'] for n in fast[3]], [n['idle_release'] for n in released[3]])
        self.assertEqual(fast[4]['idle_release'], released[4]['idle_release'])
        # The fast paths ran: one allocation_batch per step, no allocation_update RPC; every
        # evict and readmit follows a sent batch (EngineWorkerProxy.before_call).
        ops = fast[1].operations
        self.assertIn('allocation_batch', ops)
        self.assertNotIn('allocation_update', ops)
        self.assertNotIn('allocation_batch', released[1].operations)
        self.assertEqual(ops.count('evict'), released[1].operations.count('evict'))
        self.assertEqual(ops.count('readmit'), released[1].operations.count('readmit'))
        self.assertLess(ops.count('allocation_batch'), released[1].operations.count('allocation_update'))

    def check(self, policy):
        plain = pressure(0)
        released = plain.run_case(policy, plain.BLOCKS, True)
        for fast in (2, 3):
            with self.subTest(fast=fast):
                run = pressure(fast)
                found = run.run_case(policy, run.BLOCKS, True)
                self.compare(found, released)
                if fast == 3:
                    # The third fast path's own paths ran under the releases: rowless
                    # allocation_batch entries, steps that skip materialize and keep the step lease.
                    self.assertGreater(run.rowless['rowless'], 100)
                    self.assertGreater(min(found[1].fast3.values()), 100, found[1].fast3)
        if SLOW:
            reference = plain.run_case(policy, plain.REFERENCE, True)
            self.assertEqual(released[0], reference[0])
            if policy == 'observation_query':
                # test_decouple_e2e's own runner (the integrate-1014 runs; no steady paths there).
                plain0 = pressure(0, False)
                released0 = plain0.run_case(policy, plain0.BLOCKS, True)
                self.assertEqual(released0[0], reference[0])
                for fast in (2, 3):
                    with self.subTest(runner='test_decouple_e2e', fast=fast):
                        run = pressure(fast, False)
                        self.compare(run.run_case(policy, run.BLOCKS, True), released0)

    def test_observation_query(self):
        self.check('observation_query')

    @unittest.skipUnless(SLOW, 'about 50 s more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_full(self):
        self.check('full')


def shaped_runner():
    """test_ms2_e2e's DecodeFirstRunner (test_stageb_worker's FakeRunner) with vLLM 0.24's block
    tables, as test_decode_fastpath3.vllm_runner gives test_decouple_e2e's runner: numpy rows
    and row counts (no get_row), a running request's FullAttention id list extended in place
    and its Mamba lists rewritten in place (``_update_states``), so the incremental observation
    of --decode-fastpath2 and the steady paths of --decode-fastpath3 run."""
    import numpy as np
    import test_ms2_e2e as B2
    import test_stageb_worker as W

    class Rows(dict):
        """The fake's ``Table.rows`` (it adds, moves, swaps and drops rows by key), written
        through to the numpy table and its row counts."""

        def __init__(self, table):
            super().__init__()
            self.table = table

        def __setitem__(self, row, ids):
            super().__setitem__(row, ids)
            self.table.block_table.np[row, :len(ids)] = ids
            self.table.num_blocks_per_row[row] = len(ids)

        def pop(self, row, *default):
            value = super().pop(row, *default)
            self.table.num_blocks_per_row[row] = 0
            return value

    class Table:
        blocks_per_kv_block = 1

        def __init__(self, rows=32, width=4096):
            self.block_size = W.P
            self.block_table = NS(np=np.zeros((rows, width), dtype=np.int32))
            self.num_blocks_per_row = np.zeros(rows, dtype=np.int32)
            self.rows = Rows(self)

        def add_row(self, ids, row):
            self.rows[row] = list(ids)

    class Runner(B2.DecodeFirstRunner):
        def __init__(self, store):
            super().__init__(store)
            tables = self.input_batch.block_table.block_tables
            for group in list(tables):
                tables[group] = Table()

        def set_rows(self, rid, block_ids):
            req = self.requests[rid]
            row = self.input_batch.req_ids.index(rid) if rid in self.input_batch.req_ids else None
            for group, ids in zip(W.GROUPS, block_ids):
                current = req.block_ids[group]
                if group == W.FA and len(ids) >= len(current) and list(ids[:len(current)]) == current:
                    current.extend(ids[len(current):])
                else:
                    current[:] = list(ids)
                if row is not None:
                    self.input_batch.block_table.block_tables[group].add_row(current, row)
    return Runner


@contextlib.contextmanager
def kv_engines(fast, shaped=True, release=False):
    """test_kv_session_e2e's batched engine (also test_chat_text_e2e's) with the decode fast
    paths ``fast`` (attach_fast) and, with ``release``, --release-idle-pages attached as
    attach_loaded_worker and compose_engine attach it (test_idle_release's harness);
    ``shaped``: on shaped_runner. Yields the list of engines built."""
    import test_kv_session_e2e as KE
    import test_ms2_e2e as B2
    engines = []

    class Engine(KE.KvEngine):
        def __init__(self, policy, output, server, **kwargs):
            super().__init__(policy, output, server, **kwargs)
            attach_fast(self, fast)
            if release:
                self.owner.release_idle = True
                self.core._remote_ra_idle_release = R.IdleRelease(self.core, output)
            record_rpcs(self)
            # Every rebuild (a session's transcript reset), with the Pages its session has released.
            self.rebuilds = rebuilds = []
            rebuild = self.endpoint.rebuild

            def recorded(binding):
                session = self.adapter.sessions[binding.session]
                rebuilds.append((time.monotonic_ns(), binding.session, len(getattr(session, 'evicted', ()))))
                return rebuild(binding)
            self.endpoint.rebuild = recorded
            engines.append(self)
    with ExitStack() as stack:
        stack.enter_context(patch.object(KE, 'KvEngine', Engine))
        if shaped:
            stack.enter_context(patch.object(B2, 'DecodeFirstRunner', shaped_runner()))
        yield engines


def run_tests(case, names):
    """Run ``case``'s tests ``names``, each as its own TestCase: (name, run, failures, errors, skipped)."""
    outcomes = []
    for name in names:
        result = unittest.TestResult()
        case(name).run(result)
        outcomes.append((name, result.testsRun, result.failures, result.errors, result.skipped))
    return outcomes


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class KvSessionFastPathTests(unittest.TestCase):
    """test_kv_session_e2e's MemAgent sessions (the packaged router, the frontend, the
    batched engine and worker) with the decode fast paths attached to the engine and
    worker: every check of those tests holds (each turn's prompt equals the client's
    construction, the token geometry, the evidence rows, the pools back at baseline).
    goal_query with the three fast paths in the suite; with REMOTE_RA_SLOW_TESTS=1 also
    full and observation_query, and the first two fast paths alone (also on
    test_kv_session_e2e's own runner)."""

    def test_goal_query_sessions_with_the_three_fast_paths(self):
        self.memagent(3, True, 'test_memagent_sessions_goal_query')

    @unittest.skipUnless(SLOW, 'about 2.5 minutes more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_more_sessions_and_the_first_two_fast_paths(self):
        self.memagent(3, True, 'test_memagent_sessions_full', 'test_memagent_sessions_observation_query')
        self.memagent(2, True, 'test_memagent_sessions_goal_query')
        self.memagent(2, False, 'test_memagent_sessions_full', 'test_memagent_sessions_observation_query')

    def memagent(self, fast, shaped, *names):
        import test_kv_session_e2e as KE
        with kv_engines(fast, shaped) as engines:
            for name, *outcome in run_tests(KE.KvSessionEndToEndTests, names):
                with self.subTest(name, fast=fast, shaped=shaped):
                    self.assertEqual(outcome, [1, [], [], []])
        self.assertEqual(len(engines), len(names))
        for engine in engines:
            self.assertIn('allocation_batch', engine.operations)
            self.assertNotIn('allocation_update', engine.operations)
            self.assertTrue(engine.owner.decode_fastpath2 and engine.bridge.decode_fastpath)
            self.assertEqual(engine.bridge.decode_fastpath3, fast == 3)
            if fast == 3:
                self.assertGreater(min(engine.fast3.values()), 20, engine.fast3)


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ChatTextFastPathTests(unittest.TestCase):
    """test_chat_text_e2e's scenarios (the packaged router, the chat-text route and session pool,
    the frontend, the batched engine and worker) with the three decode fast paths attached
    to the engine and worker on shaped_runner: every check of those tests holds (every
    served prompt is the template's render of its messages, at the engine and the
    frontend; the evidence rows; the session events; the pool quiescent). The suite runs
    sixteen sessions decoding together; REMOTE_RA_SLOW_TESTS=1 the other scenarios
    (summary hand-off and asides, refusals, cancel and rebind, idle closes and evictions,
    20 open sessions, random_fill)."""

    def test_sixteen_sessions_with_the_three_fast_paths(self):
        self.chat('test_sixteen_sessions_decode_together')

    @unittest.skipUnless(SLOW, 'about 40 s more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_the_other_scenarios_with_the_three_fast_paths(self):
        self.chat('test_header_aside_handoff_and_close', 'test_context_limit_length_and_refusals',
                  'test_cancel_retry_rebinds_and_the_failed_slot_closes',
                  'test_idle_close_eviction_turn_bound_and_lifetime',
                  'test_spare_sessions_above_sixteen_and_no_close_while_one_is_spare',
                  'test_frontend_from_the_config_and_the_background_reaper', 'test_sixteen_sessions_random_fill')

    def chat(self, *names):
        import test_chat_text_e2e as CT
        with kv_engines(3) as engines:
            for name, *outcome in run_tests(CT.ChatTextEndToEndTests, names):
                with self.subTest(name):
                    self.assertEqual(outcome, [1, [], [], []])
        self.assertEqual(len(engines), len(names))
        for engine in engines:
            self.assertIn('allocation_batch', engine.operations)
            self.assertNotIn('allocation_update', engine.operations)
            self.assertTrue(engine.bridge.decode_fastpath3 and engine.owner.decode_fastpath3)
        # The steady paths ran (the scenario with the most decode steps).
        self.assertGreater(max(min(e.fast3.values()) for e in engines), 100)


def chat_sessions(count, episodes, policy, blocks, *, summaries=False, fast=0, release=True, shaped=True):
    """``count`` Terminus clients of test_chat_text_e2e playing ``episodes`` each against one
    chat-text server whose pool has ``blocks`` blocks (16 open sessions), as its
    test_sixteen_sessions_* tests play them (``summaries``: a small client context, so that
    each client hands off with a summary, a rebuild of its session, and asks its questions
    in an aside; the reaper closes idle sessions); the engine with the decode fast paths
    ``fast`` and --release-idle-pages ``release``. The test's checks: every served prompt is
    the template's render of its messages (engine and frontend), every request answered
    200, one evidence row each, no failure, the pool quiescent. Returns each client's
    requests and replies, each client session's prompts, selections and attended Pages,
    capacity.jsonl and the engine."""
    import test_chat_text_e2e as CT
    found = {}

    class Case(CT.ChatTextEndToEndTests):
        async def scenario(self):
            slots = count * (episodes + 1) * (4 if summaries else 1)
            srv = self.server(policy, slots=slots, limit=16, blocks=blocks)
            if summaries:
                srv.chat.autostart, srv.chat.tick_s = True, 0.02
                srv.chat.record.update(evict_after_s=0.05)
            agents = [self.agent(srv, k, header='t%02d__agent' % k if summaries and k % 2 else None,
                                 context_limit=1700 if summaries else 65536, threshold=600, short=not summaries)
                      for k in range(count)]
            await asyncio.gather(*(self.play(a, episodes) for a in agents))
            h = srv.engine
            self.check_prompts(srv)
            rows = srv.rows()
            self.assertFalse(srv.rows('failures.jsonl'))
            self.assertEqual(len(rows), sum(len(a.requests) for a in agents))
            for agent in agents:
                self.assertTrue(all(status == 200 for _, status, _ in agent.requests))
            if summaries:
                # As test_sixteen_sessions_with_summaries_observation_query: asides, rebuilds.
                self.assertEqual({r['chat_text']['kind'] for r in rows}, {'session', 'main', 'aside'})
                self.assertTrue(any(r['stage_b']['rebuild'] for r in rows))
            self.quiescent(srv)
            sessions = {}
            for row in rows:
                ct, rid = row['chat_text'], row['request_id']
                self.assertIsNotNone(h.runner.visible.get(rid), rid)
                # A client session by what it is (its initial segment and header id), not by its
                # name: names follow the bind order, which timing may change (the asides of two
                # clients that summarize at the same time).
                sessions.setdefault((ct['key_sha256'], ct['session_id']), []).append(dict(
                    kind=ct['kind'], rebuild=row['stage_b']['rebuild'], finish=ct['finish_reason'],
                    prompt=row['prompt_token_ids'], selected=row['selected_logical_pages'],
                    attended=h.runner.visible[rid]))
            replies = [[(messages, status, body['choices'][0]['message']['content'], body['usage'])
                        for messages, status, body in a.requests] for a in agents]
            found.update(sessions=sessions, replies=replies, events=srv.rows(R.EVENTS), engine=h,
                         width=max(len(b.rows) for b in h.batches))
    with kv_engines(fast, shaped, release) as engines:
        result = unittest.TestResult()
        Case('scenario').run(result)
    return (result.testsRun, result.failures, result.errors, result.skipped), found


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class ChatTextReleaseTests(unittest.TestCase):
    """chat-text sessions under GPU pressure with --release-idle-pages (test_chat_text_e2e's
    engine, the stage-B shared pool at P = 16, on shaped_runner): the idle sessions' pins
    exceed the pool, so the capacity gate releases them and readmits them before their next
    turn, and every client session's requests, replies, prompts, selections and attended
    Pages equal those of a pool that never releases (that pool with the switch on, which
    then releases nothing). The pool holds one turn's worst case (its prompt and the
    1536-token reply bound) next to the other sessions' tails, as the launcher's rule
    demands; smaller pools fail as without the switch."""
    maxDiff = None

    def check(self, run, reference):
        """A run under pressure against the reference; returns the run and the number of
        evicted sessions whose next turn was a rebuild."""
        self.assertEqual(run[0], (1, [], [], []))
        found, large = run[1], reference[1]
        self.assertEqual(found['replies'], large['replies'])
        self.assertEqual(found['sessions'], large['sessions'])
        events, h = found['events'], found['engine']
        evictions = [e for e in events if e['event'] == 'evict']
        readmissions = [e for e in events if e['event'] == 'readmit']
        self.assertTrue(evictions and readmissions)
        # Each readmission restores exactly what that session's last eviction released; an
        # evicted session whose next turn rebuilds (a summary hand-off) is not readmitted:
        # the rebuild closes its namespace, released Pages and all.
        last, rebuilt = {}, 0
        stream = sorted([(e['monotonic_ns'], 0, e) for e in events] +
                        [(t, 1, dict(event='rebuild', session_id=sid, pages=n)) for t, sid, n in h.rebuilds],
                        key=lambda item: item[:2])
        for _, _, e in stream:
            if e['event'] == 'evict':
                self.assertNotIn(e['session_id'], last)
                self.assertGreater(e['free_blocks_by_pool'][0], e['free_blocks_by_pool_before'][0])
                last[e['session_id']] = e['pages']
            elif e['event'] == 'rebuild':
                self.assertEqual(e['pages'], last.pop(e['session_id'], 0))
                rebuilt += e['pages'] > 0
            else:
                self.assertEqual(e['pages'], last.pop(e['session_id']))
        self.assertEqual(h.core._remote_ra_idle_release.summary(), dict(
            evictions=len(evictions), evicted_pages=sum(e['pages'] for e in evictions),
            readmissions=len(readmissions), readmitted_pages=sum(e['pages'] for e in readmissions)))
        self.assertEqual((h.operations.count('evict'), h.operations.count('readmit')),
                         (len(evictions), len(readmissions)))
        return found, rebuilt

    def reference(self, *args, release=True, **kwargs):
        """The large pool (8,000 blocks): with the switch on it releases nothing."""
        large = chat_sessions(*args, 8000, release=release, **kwargs)
        self.assertEqual(large[0], (1, [], [], []))
        self.assertEqual(large[1]['events'], [])
        if release:
            self.assertEqual(large[1]['engine'].core._remote_ra_idle_release.summary(),
                             dict(evictions=0, evicted_pages=0, readmissions=0, readmitted_pages=0))
        return large

    def test_eight_sessions_with_release_and_the_three_fast_paths(self):
        reference = self.reference(8, 2, 'observation_query')
        found, _ = self.check(chat_sessions(8, 2, 'observation_query', 260, fast=3), reference)
        h = found['engine']
        self.assertGreater(found['width'], 1)                     # sessions still decode together
        self.assertIn('allocation_batch', h.operations)
        self.assertNotIn('allocation_update', h.operations)
        self.assertGreater(min(h.fast3.values()), 20, h.fast3)

    @unittest.skipUnless(SLOW, 'about a minute more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_release_alone_and_summary_hand_offs(self):
        reference = self.reference(8, 2, 'observation_query')
        plain = self.reference(8, 2, 'observation_query', release=False)
        self.assertEqual((plain[1]['replies'], plain[1]['sessions']),
                         (reference[1]['replies'], reference[1]['sessions']))
        self.check(chat_sessions(8, 2, 'observation_query', 260), reference)
        # Terminus's summary hand-offs under pressure: rebuilt sessions, asides, the reaper.
        reference = self.reference(4, 5, 'observation_query', summaries=True)
        for fast in (0, 3):
            with self.subTest(fast=fast):
                _, rebuilt = self.check(chat_sessions(4, 5, 'observation_query', 300, summaries=True, fast=fast),
                                        reference)
                self.assertGreater(rebuilt, 0)       # hand-offs of released sessions (3 or 4 of 4 here)


# ---- 4. --admission-overlap K with --release-idle-pages (idle_release, fixed in integrate-1016) ---

def overlap_release(sessions, turns, permits, fast, *, lone_after=None):
    """test_idle_release's pressure run of goal_query sessions with --admission-overlap K
    (``permits``; None: one admission at a time, test_decouple_e2e's MultiHarness), as
    test_adm_overlap's AllFlagsEndToEndTests serves them: its harness (the overlap park table,
    a Goal thread per session, the pinned lease and the ledger; EngineCoreProc's loop with up
    to K admissions in flight), the rewriter stand-in (serialised, staggered), every Goal
    taking 20 ms, two failing observations so that the gate REWRITEs past turn 0; the decode
    fast paths ``fast`` (attach_fast); on test_decode_fastpath3's vLLM-shaped runner (numpy
    block tables grown in place). ``lone_after`` n: after every lane's first n turns
    (nothing runs, nothing is in flight, every session idle with its Pages pinned) lane 0's
    next turn alone, with the free FullAttention blocks held down so that its phase 1 fits
    only once idle sessions are released (``h.lone``: its figures), then the rest."""
    import test_adm_overlap as AOT
    import test_decode_fastpath3 as TF3
    import test_decouple_e2e as DE
    import test_idle_release as IR

    class Rows16(TF3.NpTable):
        def __init__(self, rows, width, block_size):
            super().__init__(max(rows, 16), width, block_size)

    class OverlapRelease(IR.PressureTests):
        SESSIONS, TURNS = sessions, turns

        def enter(self, stack, policy, blocks):
            from remote_ra_live_v2 import goal_async as GA, query_selection
            self.rewriters = []

            def start(pin):
                self.rewriters.append(AOT.LockedRewriter(lambda request: 0.002 * (1 + len(request.root_goal) % 4)))
                return self.rewriters[-1]
            stack.enter_context(patch.object(query_selection, 'goal_rewriter', start))
            stack.enter_context(patch.object(DE, 'OUT', AOT.FailingOut(DE.OUT)))
            stack.enter_context(patch.object(DE, 'TURNS', AOT.failing_turns(DE.TURNS)))
            build = GA.GoalWork.run

            def slow(work):
                time.sleep(0.02)            # every Goal takes a while: other admissions start meanwhile
                return build(work)
            stack.enter_context(patch.object(GA.GoalWork, 'run', slow))
            stack.enter_context(patch.object(TF3, 'NpTable', Rows16))
            stack.enter_context(patch.object(DE, 'Runner', TF3.vllm_runner(False)))
            self.rowless = stack.enter_context(rowless_counted())
            case, output = IR.PressureTests.enter(self, stack, policy, blocks)
            if permits is not None:
                value = dict(http_config.load(), admission=AOT.AF.record(admission_overlap=permits))
                stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
                stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
            return case, output

        def harness(self, policy, case, output, release):
            if permits is None:
                h = DE.MultiHarness(policy, output, case, tasks=IR.tasks(sessions))
            else:
                h = AOT.overlap_harness(permits, in_step=False, handoff=False)(policy, output, case,
                                                                             tasks=IR.tasks(sessions))
            for lane in h.lanes:
                lane.script = self.script()
            h.active = h.restores = 0
            if release:
                h.owner.release_idle = True
                h.core._remote_ra_idle_release = R.IdleRelease(h.core, output)
            attach_fast(h, fast)
            record_rpcs(h)
            serve = h.serve
            if lone_after is not None:
                serve = lone_turn(h, serve, lone_after)
            if permits is not None:
                plain = serve

                def serve(check=None):
                    try:
                        plain(check)
                    finally:
                        h.table.close()     # engine teardown: the Goal threads
            h.serve = serve
            return h
    return OverlapRelease('test_sixteen_sessions_full')


def lone_turn(h, serve, after):
    """``h.serve`` with lane 0's turn ``after`` served alone between the others (overlap_release)."""
    from remote_ra_live_v2 import capacity as CAP

    def serving(check=None):
        later = [lane.script[after:] for lane in h.lanes]
        for lane in h.lanes:
            lane.script = lane.script[:after]
        serve(check)
        assert not (h.running or h.table.parked or h.lifecycle.reservations)
        release = h.core._remote_ra_idle_release
        admit, held, h.lone = release.admit, [], {}

        def first(lifecycle, command, gate):
            release.admit = admit
            need = CAP.assess(lifecycle, command)
            peak = max(need['phase1'][0], need['phase2'][0] + need['watermark'][0])
            idle = release.groups() * sum(len(s.resident) for sid, s in h.adapter.sessions.items()
                                          if sid != command.session_id)
            held.extend(h.pool.get_new_blocks(max(0, need['free'][0] - max(0, peak - idle + 2))))
            h.lone.update(request=command.request_id, session=command.session_id, peak=peak, idle=idle,
                          phase1=need['phase1'][0], free=h.pool.get_num_free_blocks())
            return admit(lifecycle, command, gate)
        release.admit = first
        for lane in h.lanes:
            lane.script = []
        h.lanes[0].script, later[0] = later[0][:1], later[0][1:]
        serve(check)
        h.pool.free_blocks(held)
        for lane, script in zip(h.lanes, later):
            lane.script = script
        serve(check)
    return serving


@unittest.skipUnless(torch is not None, 'CPU torch required for hidden rows')
class OverlapReleaseTests(unittest.TestCase):
    """goal_query sessions on the decoupled 16-token layout with --admission-overlap K (Goals
    parked on their session threads, several admissions in flight) and --release-idle-pages
    under pool pressure, with and without the three decode fast paths (overlap_release):
    every turn's prompt, reply, selection and attended Pages equal those of the K = 1 server
    with a pool that never releases (9,000 blocks); idle sessions are released and
    readmitted, each readmission restoring what its session's last eviction released. And
    the lone turn: nothing runs, nothing is in flight, every session is idle with its Pages
    pinned, and the turn's phase 1 fits only once idle Pages are released. The ledger gate
    admits it unchecked and IdleRelease.admit releases what it needs first (integrate-1016:
    0535f79's idle_release released nothing there, and its phase 1 failed with "insufficient
    native blocks for phase 1"). The suite runs eight sessions of four turns, K = 4, the
    pressure run (600 blocks) with the three fast paths and the lone turn without them;
    REMOTE_RA_SLOW_TESTS=1 adds the other halves, the first two fast paths, K = 1 and K = 8
    under pressure, and sixteen sessions with K = 16."""
    maxDiff = None
    SESSIONS, TURNS, BLOCKS, REFERENCE = 8, 4, 600, 9000
    _references = {}

    def reference(self, sessions, turns):
        """The K = 1 server's turns with a pool that never releases (once per class)."""
        key = sessions, turns
        if key not in self._references:
            run = overlap_release(sessions, turns, None, 0)
            result, h, events, natives, cleanup = run.run_case('goal_query', self.REFERENCE, True)
            self.assertEqual(events, [])
            self._references[key] = result
        return self._references[key]

    def served(self, permits, fast, blocks, *, sessions=None, turns=None, lone_after=None):
        sessions, turns = sessions or self.SESSIONS, turns or self.TURNS
        reference = self.reference(sessions, turns)
        run = overlap_release(sessions, turns, permits, fast, lone_after=lone_after)
        result, h, events, natives, cleanup = run.run_case('goal_query', blocks, True)
        self.assertEqual(result, reference)
        evictions = [e for e in events if e['event'] == 'evict']
        readmissions = [e for e in events if e['event'] == 'readmit']
        self.assertTrue(evictions and readmissions)
        last = {}
        for e in events:
            if e['event'] == 'evict':
                self.assertNotIn(e['session_id'], last)
                last[e['session_id']] = e['pages']
            else:
                self.assertEqual(e['pages'], last.pop(e['session_id']))
        self.assertEqual(cleanup['idle_release'], dict(evictions=len(evictions),
            evicted_pages=sum(e['pages'] for e in evictions), readmissions=len(readmissions),
            readmitted_pages=sum(e['pages'] for e in readmissions)))
        self.assertEqual(sum(len(n['idle_release']['evicted_sessions']) for n in natives), len(evictions))
        if permits is not None:
            self.assertGreaterEqual(h.parked_peak, 2)               # admissions overlapped
            self.assertGreater(len(run.rewriters[0].calls), 2 * sessions - 1)
        ops = h.operations
        self.assertEqual(('allocation_batch' in ops, 'allocation_update' in ops), (fast > 0, fast == 0))
        if fast == 3:
            self.assertGreater(min(h.fast3.values()), 100, h.fast3)
            self.assertGreater(run.rowless['rowless'], 100)
        if lone_after is not None:
            lone = h.lone
            # It could not run its phase 1 on the free blocks; the release came before the gate.
            self.assertLess(lone['free'], lone['phase1'])
            mine = [e for e in evictions if e['request_id'] == lone['request']]
            self.assertTrue(mine)
            self.assertNotIn(lone['session'], [e['session_id'] for e in mine])
            self.assertGreaterEqual(lone['free'] + sum(e['blocks'] for e in mine), lone['peak'])
            record = next(n['idle_release'] for n in natives if n['request_id'] == lone['request'])
            self.assertEqual(record['evicted_sessions'], [e['session_id'] for e in mine])
        return h

    def test_pressure_with_the_three_fast_paths(self):
        self.served(4, 3, self.BLOCKS)

    def test_the_lone_turn(self):
        # On the large pool: the first two turns release nothing, so every session holds its Pages.
        self.served(4, 0, self.REFERENCE, lone_after=2)

    @unittest.skipUnless(SLOW, 'about 30 s more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_more_overlaps_and_fast_paths(self):
        for permits, fast, blocks, lone_after in ((4, 0, self.BLOCKS, None), (4, 2, self.BLOCKS, None),
                                                  (None, 3, self.BLOCKS, None), (8, 3, self.BLOCKS, None),
                                                  (4, 3, self.REFERENCE, 2)):
            with self.subTest(permits=permits, fast=fast, blocks=blocks, lone_after=lone_after):
                self.served(permits, fast, blocks, lone_after=lone_after)

    @unittest.skipUnless(SLOW, 'about 20 s more of CPU fakes: set REMOTE_RA_SLOW_TESTS=1 (not in the archive suite)')
    def test_sixteen_sessions(self):
        self.served(16, 3, 1200, sessions=16)


if __name__ == '__main__':
    unittest.main()
