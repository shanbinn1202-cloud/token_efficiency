"""--goal-rewriter-engine vllm together with --goal-overlap (integrate-1030). CPU only; not GPU evidence.

rewriter-vllm (``--goal-rewriter-engine vllm``: the Goal rewriter's calls go to one vLLM child,
several in flight; docs/rewriter-vllm/DESIGN.md) and goal-overlap (``--goal-overlap``: phase 1 runs
while the Goal is built, the Goal forward afterwards as its own exact pass; docs/goal-overlap/DESIGN.md)
each test their own option (test_goal_rewriter_vllm.py, test_goal_overlap.py). This file runs them
together on a Qwen goal_query server:

- the launcher pins both records, admission (with --admission-overlap K and goal_overlap) and
  goal_rewriter_engine, and nothing else: the vLLM command, the environment and every other key
  are those of the server with --admission-overlap K alone; the dry run records the engine record
  and the same command; remote_ra_http.config.load accepts the startup config and returns both;
- end to end over test_adm_overlap's three goal_query sessions decoding together at 16-token
  Pages (test_decouple_e2e's fakes: the real engine, worker, Selector V2 and evidence writers;
  test_adm_overlap's turns whose new error signatures REWRITE), the rewriter pinned in the
  selection contract: the all-off server with the Hugging Face child's in-process fakes
  (test_goal_rewriter_batch.InProcess, one call at a time) against the server with
  --admission-overlap 3, --goal-overlap and the vLLM rewriter (test_goal_rewriter_vllm's real child
  subprocess on its fake vllm site, the authority starting it from the pinned record). Every
  turn's outputs, selection, ranking, carry, Page vectors, signed Goal, Goal and query hashes, the
  worker frames apart from the Goal forward's own, and every goals.jsonl row with its exchanges
  (requests, answers, errors, token counts; ids and times aside) equal the all-off server's; each
  turn that needs a Goal forward ran it as its own pass; the vLLM child answered every rewriter
  call, with several in flight.
"""
from contextlib import ExitStack, redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source-overlay'))
sys.path.insert(0, str(ROOT))

import install
import serve
import test_goal_overlap as TGO
import test_goal_rewriter_vllm as TV
import test_installer as INSTALL

RB = TV.RB
from remote_ra_http import config as HC     # noqa: E402
from remote_ra_live_v2 import admission_flags as AF, goal_async as GA, goal_text     # noqa: E402
from remote_ra_live_v2 import goal_overlap as GO, goal_rewriter_vllm as V     # noqa: E402

TORCH = importlib.util.find_spec('torch') is not None
PERMITS = 3


def setUpModule():
    TV.setUpModule()            # the fake vllm and transformers site, the rewriter checkpoint


def tearDownModule():
    TV.tearDownModule()


# ---- the launcher and the startup check ---------------------------------------------------------------

class LauncherTests(unittest.TestCase):
    maxDiff = None
    TASKS = [dict(instance_id='repo__v%02d-%d' % (k, k), problem_statement='Fix check %02d.' % k) for k in range(4)]

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with redirect_stdout(io.StringIO()):
            package, site, model, cls.release = INSTALL.fixture(cls.root)
            install.install(site, model, cls.release, package_root=package, apply=True)
        cls.rewriter = cls.root / 'qwen3-0.6b'
        cls.rewriter.mkdir()
        (cls.rewriter / 'config.json').write_bytes(b'{"model_type":"qwen3"}')
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.enterContext(patch.object(sys, 'path', [str(ROOT/'source-overlay')] + sys.path))

    def prepare(self, **kwargs):
        """serve.prepare of a 4-session goal_query server at 16-token Pages: (config with its output
        named OUT, command, env, config as written)."""
        type(self).n += 1
        out = self.root / ('out%d' % self.n)
        with redirect_stdout(io.StringIO()):
            _, config, command, env = serve.prepare(self.release, [dict(t) for t in self.TASKS], 'goal_query', out,
                                                    8000, dict(goal_rewriter_model=str(self.rewriter)), 'mswea-xml',
                                                    24, 16, None, **kwargs)
        named = json.loads(json.dumps(config).replace(json.dumps(str(out))[1:-1], 'OUT'))
        return named, [a.replace(str(out), 'OUT') for a in command], \
            {k: v.replace(str(out), 'OUT') for k, v in env.items() if k != 'REMOTE_RA_HTTP_CONFIG_SHA256'}, config

    def load(self, config):
        type(self).n += 1
        path = self.root / ('startup%d.json' % self.n)
        path.write_bytes(serve.wire(config))
        with patch.dict(os.environ, REMOTE_RA_HTTP_RUNTIME='1', REMOTE_RA_HTTP_CONFIG=str(path),
                        REMOTE_RA_HTTP_CONFIG_SHA256=serve.pin(config)):
            return HC.load()

    def test_both_flags_pin_both_records_and_nothing_else(self):
        plain, command, env, _ = self.prepare(admission_overlap=4)
        record = V.record(str(self.rewriter.resolve()))
        admission = dict(AF.record(admission_overlap=4), goal_overlap=True)
        for kw, value in ((dict(), record),
                          (dict(goal_rewriter_vllm_kv_gib=5), V.record(str(self.rewriter.resolve()), kv_gib=5))):
            with self.subTest(kw=kw):
                config, flagged, flagged_env, written = self.prepare(admission_overlap=4, goal_overlap=True,
                                                                     goal_rewriter_engine='vllm', **kw)
                self.assertEqual(config.pop(V.CONFIG_KEY), value)
                self.assertEqual(config.pop('admission'), admission)
                plain_rest = dict(plain)
                self.assertEqual(plain_rest.pop('admission'), AF.record(admission_overlap=4))
                self.assertEqual((config, flagged, flagged_env), (plain_rest, command, env))   # no vLLM argument
                loaded = self.load(written)
                self.assertEqual((loaded['admission'], loaded[V.CONFIG_KEY]), (admission, value))
                self.assertEqual((HC.admission(written), HC.goal_rewriter_engine(written)), (admission, value))
        # Each refusal of either option holds with the other on.
        with self.assertRaisesRegex(ValueError, 'excludes --goal-rewriter-batch'):
            self.prepare(admission_overlap=4, goal_overlap=True, goal_rewriter_engine='vllm', goal_rewriter_batch=4)
        with self.assertRaisesRegex(ValueError, 'needs --admission-overlap K >= 2'):
            self.prepare(goal_overlap=True, goal_rewriter_engine='vllm')

    def test_the_dry_run_records_the_engine_and_the_same_command(self):
        task = self.root / 'tasks.json'
        task.write_text(json.dumps(self.TASKS))

        def dry(*flags):
            type(self).n += 1
            output = self.root / ('dry%d' % self.n)
            argv = ['serve.py', '--no-speed-defaults', '--release-dir', str(self.release), '--task', str(task), '--policy', 'goal_query',
                    '--output', str(output), '--goal-rewriter-model', str(self.rewriter), '--page-tokens', '16',
                    '--kv-cache-gib', '24', '--admission-overlap', '4', *flags]
            stdout = io.StringIO()
            with patch.object(sys, 'argv', argv), redirect_stdout(stdout):
                serve.main()
            return json.loads(stdout.getvalue().replace(json.dumps(str(output))[1:-1], 'OUT'))
        plain = dry()
        value = dry('--goal-overlap', '--goal-rewriter-engine', 'vllm')
        self.assertEqual(value.pop(V.CONFIG_KEY), V.record(str(self.rewriter.resolve())))
        self.assertEqual(value, plain)


# ---- end to end: three goal_query sessions with both options ------------------------------------------

class Rewriters:
    """query_selection.goal_rewriter as the harness patches it: without an engine record the Hugging
    Face child's in-process fakes (test_goal_rewriter_batch.InProcess, the upstream instance's own
    lock: one call at a time); with one the vLLM transport and its real child on the fake vllm site
    (as test_goal_rewriter_vllm.EngineRewriter starts it; max_new_tokens as the in-process child's).
    Built from the modules the engine imports (the harness patches sys.modules)."""

    def __init__(self, test, log, pin):
        self.test, self.log, self.pin = test, log, pin
        self.children, self.engines = [], []

    def __call__(self, pin, batch=None, tokens=0, engine=None):
        self.test.assertEqual((pin, batch, tokens), (self.pin, None, 0))
        R = importlib.import_module('dram_kv.goals.rewriter')
        if engine is None:
            rewriter = R.PersistentCommandGoalRewriter(('in-process-child',), timeout_seconds=60.0,
                                                       startup_timeout_seconds=60.0)
            self.children.append(RB.InProcess(rewriter, delay=lambda size: 0.02))
            rewriter.__enter__()
            self.test.addCleanup(rewriter.close)
            return rewriter
        live = importlib.import_module('remote_ra_live_v2.goal_rewriter_vllm')
        stderr = open(TV.fresh('e2e-stderr'), 'w', encoding='utf-8')
        self.test.addCleanup(stderr.close)
        with patch.dict(os.environ, FAKE_VLLM_LOG=str(self.log)):
            rewriter = live.start(R, dict(pin, max_new_tokens=RB.MAX_NEW), engine, TV.OVERLAY, TV.STATE.site,
                                  sys.executable, stderr=stderr)
        self.test.addCleanup(rewriter.close)
        self.engines.append(engine)
        return rewriter


def exchanges_of(rows):
    """Per (session, turn): the goals.jsonl row without its request id and epoch, each exchange's
    request, answer, error and token counts without ids and times (test_gemma_goal_rewriter's)."""
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


@unittest.skipUnless(TORCH, 'CPU torch required for hidden rows and the fake rewriter model')
class VllmGoalOverlapEndToEndTests(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.log = TV.fresh('overlap-vllm').with_suffix('.jsonl')
        self.rewriters = Rewriters(self, self.log, TV.pin())

    def harness(self, stack, *, combined):
        """test_adm_overlap.AllFlagsEndToEndTests.harness's server with the rewriter pinned and
        ``self.rewriters`` behind it: all off, or (``combined``) --admission-overlap 3 --goal-overlap
        with goal_rewriter_engine pinned, wired as compose_engine wires them."""
        import test_adm_overlap as OV
        import test_decouple_e2e as DE
        from remote_ra_live_v2 import query_selection, selection_contract as SC
        multi = DE.Case(**dict(vars(DE.CASES[2]), sessions=len(DE.TASKS)))
        DE.enter_case(stack, multi)
        output = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        value = DE.config('goal_query', output, multi, DE.TASKS)
        value['selection_contract'] = SC.validate(dict(value['selection_contract'], goal_rewriter=self.rewriters.pin))
        record = None
        if combined:
            record = V.record(self.rewriters.pin['model'])
            value.update(admission=AF.record(admission_overlap=PERMITS, goal_overlap=True),
                         **{V.CONFIG_KEY: record})
        # The startup checks of remote_ra_http.config.load on this config.
        self.assertEqual(V.pinned(value), record)
        stack.enter_context(patch('remote_ra_http.config.load', lambda: value))
        stack.enter_context(patch('remote_ra_http.evidence.load', lambda: value))
        stack.enter_context(patch.object(query_selection, 'goal_rewriter', self.rewriters))
        stack.enter_context(patch.object(DE, 'OUT', OV.FailingOut(DE.OUT)))
        # Four turns per lane: INITIALIZE, PATCH, REWRITE, PATCH (test_adm_overlap's).
        stack.enter_context(patch.object(DE, 'TURNS', OV.failing_turns(DE.TURNS)[:4]))
        build = GA.GoalWork.run

        def slow(work):
            time.sleep(0.02)            # every Goal takes a while: phase 1 and other lanes run meanwhile
            return build(work)
        stack.enter_context(patch.object(GA.GoalWork, 'run', slow))
        if combined:
            h = OV.overlap_harness(PERMITS, in_step=False, handoff=False)('goal_query', output, multi)
            self.assertEqual((h.selection.rewriter_engine, h.selection.rewriter_batch), (record, None))
            h.core._remote_ra_goal_admissions = h.table      # compose_engine's goal_async attachment
            self.assertTrue(GO.attach_engine(h.core, value['admission']))
            h.owner.goal_overlap = True
        else:
            h = DE.MultiHarness('goal_query', output, multi)
            self.assertIsNone(h.selection.rewriter_engine)
        h.active = h.restores = 0
        h.output = output
        return h

    def serve(self, *, combined):
        import test_adm_overlap as OV
        import test_decouple_e2e as DE
        with ExitStack() as stack:
            h = self.harness(stack, combined=combined)
            checks = DE.MultiSessionMatrixTests('test_three_sessions_over_the_matrix')
            h.serve(checks.check)
            self.assertEqual(len(h.done), 4 * len(DE.TASKS))
            self.assertEqual((h.endpoint.phase1.records, h.endpoint.phase1.goal_records), ({}, {}))
            rows = [json.loads(line) for line in (h.output / goal_text.FILE).read_bytes().splitlines()]
            result = dict(decisions=OV.AllFlagsEndToEndTests.decisions(h), rows=rows,
                          natives={i['rid']: i['native'] for i in h.done},
                          parked_peak=getattr(h, 'parked_peak', 0))
            h.close()
            DE.engine_closed(h.core)
            self.assertEqual((h.pool.get_num_free_blocks(), h.mamba_pool.get_num_free_blocks()),
                             (h.baseline, h.mamba_baseline))
            return result

    def test_every_turn_equals_the_all_off_server_with_the_hugging_face_child(self):
        off = self.serve(combined=False)
        on = self.serve(combined=True)
        self.assertEqual(len(self.rewriters.children), 1)
        self.assertEqual(self.rewriters.engines, [V.record(self.rewriters.pin['model'])])
        # Goals, Goal vectors, selections, Page vectors and tokens: the all-off server's.
        self.assertEqual(set(on['decisions']), set(off['decisions']))
        normalized = TGO.AllFlagsGoalOverlapTests.normalized(on['decisions'])
        for key in sorted(off['decisions']):
            self.assertEqual(normalized[key], off['decisions'][key], key)
        # The same Goal text rows and rewriter exchanges, without batch keys.
        self.assertEqual(exchanges_of(on['rows']), exchanges_of(off['rows']))
        exchanges = [x for row in on['rows'] for x in row['rewrites']]
        self.assertTrue(exchanges and all(set(goal_text.BATCH).isdisjoint(x) for x in exchanges))
        goals = [d['goal'] for d in off['decisions'].values()]
        self.assertTrue(any(g['gate_decision'] == 'REWRITE' and g['slm_succeeded'] for g in goals))
        # Each turn with a Goal forward ran it as its own pass after phase 1; the others none.
        forwards = {key for key, value in off['decisions'].items() if value['goal']['forward']}
        self.assertTrue(forwards)
        for key, value in on['decisions'].items():
            operations = value['operations']
            self.assertEqual('goal_forward' in operations, key in forwards, key)
            if key in forwards:
                start = operations.index('phase1')
                self.assertEqual(operations[start:start + len(TGO.GOAL_FRAMES)], TGO.GOAL_FRAMES)
        natives = list(on['natives'].values())
        self.assertEqual(sum(1 for n in natives if 'goal_forward_ms' in n), len(forwards))
        self.assertTrue(all(n['goal_overlap_ms'] >= 0 and n['goal_async_ms'] >= 0 for n in natives))
        self.assertTrue(all('goal_overlap_ms' not in n for n in off['natives'].values()))
        # The vLLM child answered every rewriter call, several in flight; admissions overlapped.
        self.assertEqual(len(TV.events(self.log, 'engine')), 1)
        self.assertEqual(len(TV.events(self.log, 'request')), len(exchanges))
        self.assertGreaterEqual(max(e['peak'] for e in TV.events(self.log, 'finish')), 2)
        self.assertGreaterEqual(on['parked_peak'], 2)


if __name__ == '__main__':
    unittest.main()
