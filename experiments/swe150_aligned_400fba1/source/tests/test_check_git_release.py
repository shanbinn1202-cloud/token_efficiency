"""check_git_release.py: the parallel run (one process per test module) reports what the single-process run
reports - the same test ids and outcomes, unittest's own closing lines - and fails closed where it could not."""
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import check_git_release as R

ALPHA = '''import unittest
class A(unittest.TestCase):
    def test_pass(self): pass
    def test_skip(self): self.skipTest('not here')
    @unittest.expectedFailure
    def test_expected(self): self.assertEqual(1, 2)
    def test_subtests(self):
        for k in range(3):
            with self.subTest(k=k): self.assertLess(k, 3)
'''
BETA = '''import unittest
import helper
class B(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.value = helper.VALUE
    def test_value(self): self.assertEqual(self.value, 7)
    @unittest.skip('decorated')
    def test_decorated(self): pass
'''
GAMMA = '''"""No tests here."""
'''
FAILING = '''import unittest
class F(unittest.TestCase):
    def test_wrong(self): self.assertEqual('expected', 'got')
'''
BROKEN = '''import unittest
import no_such_module_anywhere
'''
SLOW = '''import time, unittest
class S(unittest.TestCase):
    def test_sleeps(self): time.sleep(60)
'''
HOOKED = '''import unittest
def load_tests(loader, tests, pattern): return tests
class H(unittest.TestCase):
    def test_x(self): pass
'''
GOOD = {'test_alpha.py': ALPHA, 'test_beta.py': BETA, 'test_gamma.py': GAMMA, 'helper.py': 'VALUE = 7\n'}


def repository(directory, files):
    """A Git repository with `files` in tests/, the directory check_git_release discovers (TESTS)."""
    repo = Path(directory) / 'repo'
    repo.mkdir()
    for name, text in files.items():
        path = repo / R.TESTS / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    def git(*args):
        subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)
    git('init', '--quiet')
    git('add', '--all')
    git('-c', 'user.name=Archive Test', '-c', 'user.email=archive-test@example.invalid', 'commit', '--quiet', '-m', 'x')
    return repo


class Parallel(unittest.TestCase):
    def test_parallel_reports_what_the_serial_run_reports(self):
        with TemporaryDirectory() as directory:
            repo = repository(directory, GOOD)
            serial = R.check(repo, serial=True)
            parallel = R.check(repo, jobs=2, outcomes=Path(directory) / 'parallel.json')
            record = R.check(repo, record=True, outcomes=Path(directory) / 'record.json')
            ran = re.compile(r'Ran (\d+) tests? in ([0-9.]+)s')
            for result in (serial, parallel, record):
                self.assertEqual(result['status'], 'GIT_ARCHIVE_TESTS_PASS')
                self.assertEqual(ran.search(result['tests']).group(1), '6')
                self.assertTrue(result['tests'].endswith('\nOK (skipped=2, expected failures=1)'), result['tests'])
            a = json.loads((Path(directory) / 'parallel.json').read_text('utf-8'))
            b = json.loads((Path(directory) / 'record.json').read_text('utf-8'))
            self.assertEqual(a['outcomes'], b['outcomes'])
            self.assertEqual(a['totals'], b['totals'])
            self.assertEqual(a['totals'], dict(run=6, failures=0, errors=0, skipped=2, expected_failures=1,
                                               unexpected_successes=0))
            self.assertEqual(sorted(a['modules']), ['test_alpha.py', 'test_beta.py', 'test_gamma.py'])
            self.assertIn(['test_alpha.A.test_skip', 'skip'], a['outcomes'])
            self.assertIn(['test_alpha.A.test_expected', 'expected_failure'], a['outcomes'])
            self.assertIn(['test_beta.B.test_value', 'pass'], a['outcomes'])
            self.assertIn('archive tree unchanged', parallel['tests'])

    def test_the_receipt_parsers_read_the_parallel_result(self):
        # The regular expressions of the make_receipt.py scripts and of the plan folders' build.py: the first
        # "Ran N tests in S" and the closing OK line.
        plain = ALPHA.replace('    @unittest.expectedFailure\n    def test_expected(self): self.assertEqual(1, 2)\n', '')
        with TemporaryDirectory() as directory:
            repo = repository(directory, {'test_alpha.py': plain, 'test_beta.py': BETA, 'helper.py': 'VALUE = 7\n',
                                          'test_more.py': plain.replace('class A', 'class M')})
            tests = R.check(repo, jobs=3)['tests']
            self.assertEqual(len(re.findall(r'Ran \d+ tests? in', tests)), 1)
            ran = re.search(r'Ran (\d+) tests in ([0-9.]+)s', tests)
            self.assertEqual(ran.group(1), '8')
            self.assertEqual(re.search(r'OK \(skipped=(\d+)\)', tests).group(1), '3')
            self.assertEqual(re.search(r'\nOK(?: \(skipped=(\d+)\))?$', tests).group(1), '3')

    def test_a_failure_shows_the_failing_modules_whole_output(self):
        with TemporaryDirectory() as directory:
            repo = repository(directory, dict(GOOD, **{'test_failing.py': FAILING, 'test_broken.py': BROKEN}))
            with self.assertRaises(RuntimeError) as caught:
                R.check(repo, jobs=2)
            text = str(caught.exception)
            self.assertIn('==== test_failing.py: returncode 1', text)
            self.assertIn("AssertionError: 'expected' != 'got'", text)
            self.assertIn('==== test_broken.py: returncode 1', text)
            self.assertIn('no_such_module_anywhere', text)
            self.assertTrue(text.endswith('\nFAILED (failures=1, errors=1, skipped=2, expected failures=1)'), text[-200:])
            with self.assertRaises(RuntimeError):
                R.check(repo, serial=True)

    def test_a_module_past_its_time_is_killed(self):
        with TemporaryDirectory() as directory:
            repo = repository(directory, dict(GOOD, **{'test_slow.py': SLOW}))
            began = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, 'test_slow.py: returncode -?\\d+, timed out'):
                R.check(repo, jobs=2, module_seconds=3)
            self.assertLess(time.monotonic() - began, 45)

    def test_serial_only_modules_run_after_the_pool(self):
        with TemporaryDirectory() as directory:
            repo = repository(directory, GOOD)
            with patch.dict(R.SERIAL_ONLY, {'test_alpha.py': 'here for the test'}):
                tests = R.check(repo, jobs=2)['tests']
            self.assertIn('2 workers, 1 alone after the pool', tests)
            self.assertIn('module test_alpha.py: 4 run, 1 skipped', tests)

    def test_what_one_process_per_module_cannot_reproduce_fails_closed(self):
        with TemporaryDirectory() as directory:
            repo = repository(directory, dict(GOOD, **{'pkg/__init__.py': '', 'pkg/test_inner.py': ALPHA}))
            with self.assertRaisesRegex(ValueError, 'package directory pkg'):
                R.check(repo, jobs=2)
        with TemporaryDirectory() as directory:
            repo = repository(directory, dict(GOOD, **{'test_hooked.py': HOOKED}))
            with self.assertRaisesRegex(RuntimeError, 'load_tests in test_hooked'):
                R.check(repo, jobs=2)

    def test_a_module_answering_any_attribute_is_no_load_tests_hook(self):
        # As torch.ops does: a module __getattr__ that returns something for every name.
        dynamic = {'dynamic.py': 'def __getattr__(name):\n    return object()\n',
                   'test_uses.py': 'import unittest\nimport dynamic\nclass U(unittest.TestCase):\n'
                                   '    def test_x(self): self.assertIsNotNone(dynamic.anything)\n'}
        with TemporaryDirectory() as directory:
            repo = repository(directory, dict(GOOD, **dynamic))
            self.assertEqual(R.check(repo, jobs=2)['status'], 'GIT_ARCHIVE_TESTS_PASS')
            self.assertEqual(R.check(repo, record=True)['status'], 'GIT_ARCHIVE_TESTS_PASS')


if __name__ == '__main__':
    unittest.main()
