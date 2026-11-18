"""Local only (never delivered): the Terminus 2 template hash check of goal_query's Goal inputs against the agent host's
Harbor files (2026-10-21).

The runtime's PROVISIONAL Terminus 2 Goal inputs (remote_ra_live_v2/terminus2_goal.py) pin four files of Harbor 0.21.0
by SHA-256 (templates/terminus-json-plain.txt, templates/timeout.txt, terminus_2.py, tmux_session.py). This check runs
the sealed commit's own test_terminus2_goal (every test; with REMOTE_RA_HARBOR_TERMINUS2_DIR also HarborSourceTests:
the pins and every literal against Harbor's files) on a read-only copy of the agent host's
harbor/agents/terminus_2 (fetched from tbench with `ssh tbench tar -C .../harbor/agents/terminus_2 -cf - ...`, never
modified there), after checking that the copy's files are the ones pins.json pins (config.harbor.terminus2_goal_files,
the RECORD entries of the pinned Harbor install, which the tbench preflight verifies before every claim) and the
sealed runtime's pins (runtime_features.chat_text_goal). Writes the receipt (--out) and prints
TERMINUS2_GOAL_CHECK_PASS (exit 0) or TERMINUS2_GOAL_CHECK_FAIL.
Usage: python -B terminus2_check.py --repo <clean checkout of the pinned commit> --harbor-dir <copy of terminus_2>
       [--out receipt.json] [--python D:\\Anaconda\\python.exe]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import controller as c  # noqa: E402

RUNNER = r'''
import os, sys, unittest
repo = sys.argv[1]
sys.dont_write_bytecode = True
# (82f23e2 and later: the runtime's tests in tests/)
sys.path[:0] = [repo, os.path.join(repo, 'tests'), os.path.join(repo, 'source-overlay')]
os.chdir(repo)
import test_terminus2_goal as TG
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(TG))
print('RUNNER tests=%d failures=%d errors=%d skipped=%d' % (result.testsRun, len(result.failures), len(result.errors),
                                                            len(result.skipped)))
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--harbor-dir', required=True)
    parser.add_argument('--out')
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args(argv)
    pins = c.load(ROOT / 'pins.json')
    repo, harbor = Path(args.repo).resolve(), Path(args.harbor_dir).resolve()
    problems = []
    files = {}
    pinned = c.cfg(pins)['harbor'].get('terminus2_goal_files') or {}
    for name in c.TERMINUS2_GOAL_FILES:
        path = harbor / name[len('harbor/agents/terminus_2/'):]
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if files != pinned:
        problems.append('the agent host copy %s differs from config.harbor.terminus2_goal_files %s' % (files, pinned))
    if c.terminus2_goal_pins(pins['runtime_features']) != pinned:
        problems.append('the sealed Goal-input pins differ from config.harbor.terminus2_goal_files')
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if head != pins['commit']:
        problems.append('the checkout is %s, pins.json pins %s' % (head, pins['commit']))
    started = time.time()
    env = dict(os.environ, REMOTE_RA_HARBOR_TERMINUS2_DIR=str(harbor), PYTHONDONTWRITEBYTECODE='1')
    done = subprocess.run([args.python, '-B', '-c', RUNNER, str(repo)], cwd=str(repo), env=env, capture_output=True,
                          text=True, timeout=900)
    text = done.stdout + done.stderr
    harbor_tests = [line for line in text.splitlines() if 'HarborSourceTests' in line and line.rstrip().endswith('ok')]
    if done.returncode != 0 or len(harbor_tests) != 2:
        problems.append('test_terminus2_goal exit %s, HarborSourceTests passed %d of 2: %s' % (
            done.returncode, len(harbor_tests), text[-1500:]))
    runner = next((line for line in text.splitlines() if line.startswith('RUNNER ')), '')
    value = dict(kind='terminus2-goal agent-host check', status='PASS' if not problems else 'FAIL', commit=head,
                 files=files, pinned=pinned, harbor_dir=str(harbor), runner=runner, harbor_source_tests=harbor_tests,
                 definition=(pins['runtime_features'].get('chat_text_goal') or {}).get('DEFINITION'),
                 seconds=round(time.time() - started, 1), problems=problems)
    if args.out:
        Path(args.out).write_text(json.dumps(value, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    print('%s %s files=%d harbor_source_tests=%d commit=%s' % (
        'TERMINUS2_GOAL_CHECK_PASS' if not problems else 'TERMINUS2_GOAL_CHECK_FAIL', runner, len(files),
        len(harbor_tests), head[:12]))
    for problem in problems:
        print('  PROBLEM ' + problem[:2000])
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
