"""The Goal rewriter's timing and --goal-rewriter-batch N against the Gemma 4 profile, CPU only.

The two features met in one revision (integrate-1015); what was decided:

* --goal-rewriter-batch N batches goal_query's Goal rewriter calls. The Gemma 4 stage 1
  serves goal_query (gemma-goal) but refuses the flag, for every policy: the batched
  rewriter is not checked with the window layout yet; serve.py (the Gemma launcher path)
  and remote_ra_http.config.load (a Gemma startup config that pins goal_rewriter_batch).
  A Qwen release keeps the flag.
* The rewriter's timing evidence (goals.jsonl rewrites[]: wait_ms, start_ms, call_ms,
  input_tokens, output_tokens, generation_latency_ms, and batch_id, batch_size when
  batched; the native goal_rewrite_wait_ms and goal_rewrite_call_ms) comes with a goal_query
  rewriter call only. A Gemma server, whatever policy it serves, and a Qwen server of the
  other four policies write none of it; the Gemma engine's goal_query path writes no
  exchange without a rewriter. The control: a Qwen goal_query
  server with the scripted rewriter writes the fields, so the scan below sees them.
"""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import install
import serve
import test_gemma_e2e as GE
import test_goal_text as GT
import test_host_trim as HT
from test_gemma_install import gemma_fixture
from remote_ra_http import config as HC
from remote_ra_live_v2 import goal_text
from remote_ra_live_v2 import model_profile as MP

ROOT = Path(__file__).resolve().parent
QWEN_OTHER = ('full', 'random50', 'observation_query', 'random_fill')
GEMMA_SERVED = tuple(p for p in GE.POLICIES if p != 'goal_query')
# Every key the Goal rewriter work adds to the evidence: per exchange, per native file
# (the totals of a parked Goal) and the startup pin of the batch.
TIMING_KEYS = (set(goal_text.TIMING) | set(goal_text.TELEMETRY) | set(goal_text.BATCH) |
               {native for native, _ in goal_text.TOTALS} | {'goal_rewriter_batch'})


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


class GemmaLauncherTests(unittest.TestCase):
    """serve.py and config.load refuse --goal-rewriter-batch for a Gemma release."""

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
        kw = dict(dict(kv_cache_gib=28, page_tokens=16, max_tokens=4096, max_model_len=131072,
                       language_model_only=True), **kw)
        type(self).n += 1
        return serve.prepare(self.release, task, policy, self.root / ('out%d' % self.n), 8000, selection, **kw)

    def selection(self, policy):
        if policy == 'goal_query':
            return dict(active_token_budget=8192, goal_rewriter_model=str(self.rewriter))
        return dict(active_token_budget=8192) if policy in ('observation_query', 'random_fill') else None

    def test_the_launcher_refuses_the_batch_for_every_policy(self):
        for policy in GE.POLICIES:
            for kw in (dict(goal_rewriter_batch=4, admission_overlap=4), dict(goal_rewriter_batch=2)):
                with self.subTest(policy=policy, kw=kw), \
                        self.assertRaisesRegex(ValueError, 'goal-rewriter-batch is not available for the Gemma 4'):
                    self.prepare(tasks(4), policy, self.selection(policy), **kw)
            # Without the flag every policy starts (goal_query since gemma-goal).
            _, config, _, _ = self.prepare(tasks(4), policy, self.selection(policy))
            self.assertNotIn('goal_rewriter_batch', config)
            self.assertTrue(TIMING_KEYS.isdisjoint(config))

    def test_load_refuses_a_gemma_config_that_pins_the_batch(self):
        _, config, _, _ = self.prepare(tasks(2), 'full')
        self.assertEqual(self.load(config)['model_profile'], MP.GEMMA4)
        for batch in (2, 4, 16):
            with self.subTest(batch=batch), \
                    self.assertRaisesRegex(RuntimeError, 'Goal rewriter batch is not in the Gemma 4 stage 1'):
                self.load(dict(config, goal_rewriter_batch=batch))
        self.assertIsNone(HC.goal_rewriter_batch(self.load(config)))

    def load(self, config):
        """config.load() of a startup config written as serve.py writes it."""
        type(self).n += 1
        path = self.root / ('startup%d.json' % self.n)
        path.write_bytes(serve.wire(config))
        env = dict(REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                   REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config))
        with patch.dict(os.environ, env):
            return HC.load()


class GemmaEvidenceTests(unittest.TestCase):
    """A Gemma server (test_gemma_e2e's engine, worker and evidence) writes no rewriter timing."""
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

    def test_the_refused_goal_query_path_records_no_exchange(self):
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
