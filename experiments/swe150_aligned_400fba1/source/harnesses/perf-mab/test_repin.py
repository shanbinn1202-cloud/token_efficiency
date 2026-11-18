"""The re-pin to integrate-1112 4ebff97 (2026-11-12; D:\\Remote-RA-temp\\scratch\\integrate-1112\\GPU-VALIDATION-PLAN.md
Q1 / E): the harness's feature readers and guards on the previous pin and on the new one, each read from its own sealed
bundle (CPU only, local; never delivered):

- the previous pin main ba4d1ba (the bundle of the frozen copy ..\\perf-mab-36eaf3a7, which smqt r1 job 3568311 ran):
  no page-selection policy registry (check_features refuses the build: the plans read their policies from it) and
  the async resolve without async-maxtok-1112 13ee618 (async_fix None: 3568311's EngineCore died in it), so even with
  the registry's rows supplied every async plan (smq, smqt, smga) is NOT_SUBMITTABLE there while the async-off plans
  pass; the receipt of 4ebff97 refuses that commit; its kv-session protocol module is b79d5498;
- the pin integrate-1112 4ebff97 (this folder's bundle): the registry (describe()'s rows), the three async fixes,
  async on both profiles, kv_session_protocol.py 289d3c60 (the registry's pins), the same chat template
  (KV_SESSION_TEMPLATE_SHA256); its features equal the pinned ones.
The earlier plans' arguments and dry runs are compared before / after by the scratch record (plan_fingerprint.py).
"""
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import localtemp  # noqa: E402

localtemp.ensure()
import controller as c  # noqa: E402
import git_bundle  # noqa: E402
import mab_workload as mw  # noqa: E402

PINS = c.load(ROOT / 'pins.json')
PREVIOUS = dict(commit='ba4d1ba4fc37ed9d3d593cdc6333b43628b36903', bundle=ROOT.parent / 'perf-mab-36eaf3a7' /
                git_bundle.BUNDLE_NAME, protocol='b79d5498')
CURRENT = dict(commit='4ebff978661f4c5e2e06da0ee68a5217f92dfa82', bundle=ROOT / git_bundle.BUNDLE_NAME, protocol='289d3c60')
ASYNC_PLANS = ('smq', 'smqt', 'smga', 'smqc')   # smqc (11-15): the async-maxtok case cell, async like smqt


def clone(bundle, commit, target):
    git_bundle.git(['-c', 'core.autocrlf=false', 'clone', '-q', '--branch', 'sealed-release', bundle, target],
                   stderr=subprocess.STDOUT)
    git_bundle.git(['-C', target, '-c', 'core.autocrlf=false', 'checkout', '-q', '--detach', commit],
                   stderr=subprocess.STDOUT)
    return target


class Repin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.features = {}
        for name, pin in (('previous', PREVIOUS), ('current', CURRENT)):
            if not pin['bundle'].is_file():
                continue
            repo = clone(pin['bundle'], pin['commit'], Path(cls.tmp.name) / name)
            cls.features[name] = dict(features=c.runtime_features(repo), protocol=hashlib.sha256(
                (repo / c.KV_SESSION_MODULE).read_bytes()).hexdigest())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pinned_commit_and_receipt(self):
        self.assertEqual((PINS['commit'], PINS['release_receipt']['commit']), (CURRENT['commit'], CURRENT['commit']))
        self.assertEqual((PINS['release_receipt']['tests_run'], PINS['release_receipt']['skipped'],
                          PINS['release_receipt']['status']), (2059, 139, 'GIT_ARCHIVE_TESTS_PASS'))
        c.check_release_receipt(PINS, CURRENT['commit'])
        with self.assertRaisesRegex(RuntimeError, 'not the sealed'):     # the previous pin with this receipt: refused
            c.check_release_receipt(dict(PINS, commit=PREVIOUS['commit']))
        with self.assertRaisesRegex(RuntimeError, 'cloned HEAD'):
            c.check_release_receipt(PINS, PREVIOUS['commit'])

    def test_current_pin(self):
        current = self.features['current']
        self.assertEqual(current['features'], PINS['runtime_features'])
        self.assertEqual(sorted(current['features'][c.ASYNC_FIX_KEY]), ['native_probe', 'resolve', 'scheduler'])
        self.assertEqual(current['features'][c.ASYNC_KEY]['profiles'], [c.QWEN_MODEL, c.GEMMA_MODEL])
        self.assertEqual([r['name'] for r in current['features'][c.REGISTRY_KEY]], current['features']['policies'])
        self.assertEqual(current['protocol'][:8], CURRENT['protocol'])
        self.assertEqual(current['features']['kv_session']['template_sha256'], c.KV_SESSION_TEMPLATE_SHA256)
        c.check_features(current['features'], PINS['config'])
        for name in ASYNC_PLANS + ('smg',):
            c.validate_plan(PINS, name)

    def test_previous_pin_is_refused(self):
        """ba4d1ba: no policy registry (the build and every plan refused); with the registry's rows supplied, the async
        resolve of job 3568311 still makes every async plan NOT_SUBMITTABLE (decision D-G), and the async-off plans
        (smg, smq without async_scheduling) pass the guard."""
        if 'previous' not in self.features:
            self.skipTest('the frozen copy %s is absent' % PREVIOUS['bundle'].parent)
        old = self.features['previous']
        self.assertIsNone(old['features'][c.REGISTRY_KEY])
        self.assertIsNone(old['features']['policies'])
        self.assertIsNone(old['features'][c.ASYNC_FIX_KEY])
        self.assertEqual(old['features'][c.ASYNC_KEY]['profiles'], [c.QWEN_MODEL, c.GEMMA_MODEL])
        self.assertEqual(old['protocol'][:8], PREVIOUS['protocol'])
        self.assertEqual(old['features']['kv_session']['template_sha256'], c.KV_SESSION_TEMPLATE_SHA256)
        with self.assertRaisesRegex(RuntimeError, 'no page-selection policy registry'):
            c.check_features(old['features'], PINS['config'])
        pins = copy.deepcopy(PINS)
        pins['runtime_features'] = old['features']
        for name in ASYNC_PLANS + ('smg',):
            with self.assertRaisesRegex(RuntimeError, 'no page-selection policy registry'):
                c.validate_plan(pins, name)
        # the registry's rows supplied: only the async guard is left
        supplied = dict(old['features'], policies=PINS['runtime_features']['policies'],
                        policy_registry=PINS['runtime_features'][c.REGISTRY_KEY])
        self.assertEqual({k: v for k, v in supplied.items() if k != c.ASYNC_FIX_KEY},
                         {k: v for k, v in PINS['runtime_features'].items() if k != c.ASYNC_FIX_KEY})
        with self.assertRaisesRegex(RuntimeError, 'NOT_SUBMITTABLE'):
            c.check_features(supplied, PINS['config'])
        pins['runtime_features'] = supplied
        for name in ASYNC_PLANS:
            with self.assertRaises(RuntimeError) as caught:
                c.validate_plan(pins, name)
            self.assertRegex(str(caught.exception), 'NOT_SUBMITTABLE.*async-maxtok-1112 13ee618', name)
        sync = copy.deepcopy(PINS['config'])
        for plan in sync['plans'].values():
            plan.pop('async_scheduling', None)
        sync['servers'] = [s for s in sync['servers'] if not s.get(c.ASYNC_GEMMA_KEY) and not s.get(c.CUT_CELL_KEY)]
        for name in ('smga', 'smqc'):      # smqc's cut cell is refused without --async-scheduling
            sync['plans'].pop(name)
            sync['plan_order'].remove(name)
        c.check_features(supplied, sync)
        pins.update(config=sync)
        pins['workload_plan'] = mw.build_plan(sync, pins.get('real_data'))
        c.validate_plan(pins, 'smg')
        c.validate_plan(pins, 'smq')      # smq without async_scheduling: the sync fallback is not guarded


if __name__ == '__main__':
    unittest.main()
