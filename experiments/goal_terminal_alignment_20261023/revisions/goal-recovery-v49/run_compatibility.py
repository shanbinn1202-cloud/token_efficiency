"""Exercise existing CPU assertions with the integration's scheduler-config fixture.

The original synchronous fake scheduler predates async_support.after_probe and
has no scheduler_config. Supply that missing attribute on the fake only; no
production code or test assertions are patched. Real GPU async checks run before
formal measurement in goal_recovery_probe.py.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'selective-prefix'), str(ROOT/'selective-prefix/source-overlay')]
import test_gemma_release as release
import test_gemma_e2e as geometry
import test_kv_session_protocol as protocol
import test_goal_protocol as pinned

original = geometry.Harness.__init__
def init(self, *args, **kwargs):
    original(self, *args, **kwargs)
    self.core.scheduler.scheduler_config = SimpleNamespace(async_scheduling=False)

suite = unittest.TestSuite()
for cls in [release.SixteenSessionTests, release.SelectiveRuleTests,
            protocol.FakeEpisodeTests, protocol.GoalInputTests,
            pinned.GemmaThinking, pinned.WindowRetention]:
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
if len(sys.argv) > 1:
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name, release)
                               for name in sys.argv[1:])
with patch.object(geometry.Harness, '__init__', init):
    result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
