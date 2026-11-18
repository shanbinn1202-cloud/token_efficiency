"""Run the CPU test suite of tests/: the fast default (tests/FAST_MODULES.txt) or every module, one process each.

  python -B run_tests.py                          the fast modules (a few minutes; the CI's job)
  python -B run_tests.py --all                    every tests/test_*.py module (the release check's suite)
  python -B run_tests.py --all --timings t.json   also each module's wall time
  python -B run_tests.py test_http test_lease     only the modules named
  python -B run_tests.py --fast-from t.json       rewrite tests/FAST_MODULES.txt from a timing run (no tests)
  REMOTE_RA_SLOW_TESTS=1 python -B run_tests.py --all   also the slow variants inside the modules

Modelled on the release repository's run_tests.py, with one process per module: each module runs as
`python -s -B -m unittest <module>` from the repository root, one after the other, with the root and tests/ on
PYTHONPATH (the inherited PYTHON* variables dropped), PYTHONDONTWRITEBYTECODE=1 and, unless set, one OpenMP
thread, so no module sees another's state. A module fails when its process exits non-zero or prints no OK result
line. The runner prints one line per module and one JSON summary (status, mode, seconds, failed_modules, run,
failures, errors, skipped) and exits 1 when a module failed. The full release check, on the bytes Git exports,
is check_git_release.py.

The fast default: every module that ran in at most FAST_SECONDS in a full timing run, plus test_installer (pins,
package, Git-archive check), as the release's list is made; the timing run is tests/FAST_MODULES.timings.json.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / 'tests'
FAST = TESTS / 'FAST_MODULES.txt'
FAST_SECONDS = 10.0
ALWAYS_FAST = ('test_installer',)


def all_modules():
    return sorted(p.stem for p in TESTS.glob('test_*.py'))


def fast_modules():
    names = [line.strip() for line in FAST.read_text(encoding='utf-8').splitlines()]
    return [n for n in names if n and not n.startswith('#')]


def counts(text):
    """unittest's closing lines: (ok, run, failures, errors, skipped), or None when they are missing."""
    ran = re.findall(r'^Ran (\d+) tests? in ', text, re.M)
    last = re.findall(r'^(OK|FAILED)(?: \(([^)]*)\))?\s*$', text, re.M)
    if not ran or not last:
        return None
    fields = dict(failures=0, errors=0, skipped=0)
    for part in (last[-1][1] or '').split(','):
        key, _, value = part.strip().partition('=')
        if key in fields:
            fields[key] = int(value)
    return last[-1][0] == 'OK', int(ran[-1]), fields['failures'], fields['errors'], fields['skipped']


def environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env.update(PYTHONPATH=os.pathsep.join((str(ROOT), str(TESTS))), PYTHONDONTWRITEBYTECODE='1')
    env.setdefault('OMP_NUM_THREADS', '1')
    return env


def write_fast(timings_path):
    """tests/FAST_MODULES.txt from a timing run of every module."""
    timings = json.loads(Path(timings_path).read_text(encoding='utf-8'))
    seconds = timings['modules']
    if timings['summary']['mode'] != 'all' or sorted(seconds) != all_modules():
        raise SystemExit('the timing run must cover every module of tests/ (run_tests.py --all --timings)')
    chosen = sorted(set([n for n, s in seconds.items() if s <= FAST_SECONDS] + list(ALWAYS_FAST)))
    expected = round(sum(seconds[n] for n in chosen), 1)
    FAST.write_text('# The fast default of run_tests.py: every module that ran in at most %g s in the timing run '
                    '%s (run_tests.py --all --timings, one process per module, CPU, one OpenMP thread), plus %s; '
                    'expected %.0f s.\n' % (FAST_SECONDS, Path(timings_path).name, ', '.join(ALWAYS_FAST), expected)
                    + ''.join(n + '\n' for n in chosen), encoding='utf-8', newline='\n')
    print(json.dumps(dict(fast_modules=len(chosen), expected_seconds=expected, of=len(seconds)), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('names', nargs='*', help='modules to run (default: tests/FAST_MODULES.txt)')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--timings', type=Path)
    parser.add_argument('--timeout', type=int, default=1200, help='seconds per module (default 1200)')
    parser.add_argument('--fast-from', type=Path, help='rewrite tests/FAST_MODULES.txt from this timing run and stop')
    args = parser.parse_args()
    if args.fast_from:
        write_fast(args.fast_from)
        return 0
    if args.all and args.names:
        parser.error('--all takes no module names')
    known = all_modules()
    chosen = known if args.all else (args.names or fast_modules())
    unknown = [n for n in chosen if n not in known]
    if unknown:
        parser.error('no such module in tests/: %s' % ', '.join(unknown))
    env = environment()
    totals = dict(run=0, failures=0, errors=0, skipped=0)
    timings, failed = {}, []
    start = time.monotonic()
    for name in chosen:
        t0 = time.monotonic()
        try:
            done = subprocess.run([sys.executable, '-s', '-B', '-m', 'unittest', name], cwd=ROOT, env=env,
                                  capture_output=True, text=True, encoding='utf-8', errors='replace',
                                  timeout=args.timeout)
            output, code = done.stdout + done.stderr, done.returncode
        except subprocess.TimeoutExpired as error:
            output = '\n'.join(x.decode('utf-8', 'replace') if isinstance(x, bytes) else (x or '')
                               for x in (error.stdout, error.stderr)) + '\nTIMEOUT after %d s' % args.timeout
            code = None
        timings[name] = round(time.monotonic() - t0, 2)
        result = counts(output)
        if result:
            for key, value in zip(('run', 'failures', 'errors', 'skipped'), result[1:]):
                totals[key] += value
        if code != 0 or not result or not result[0]:
            failed.append(name)
            sys.stderr.write('==== %s: exit %s\n%s\n' % (name, code, output[-8000:]))
        print('%-36s %8.2f s  %s' % (name, timings[name], 'FAILED' if name in failed else 'ok'), flush=True)
    summary = dict(status='TESTS_PASS' if not failed else 'TESTS_FAIL',
                   mode='all' if args.all else ('named' if args.names else 'fast'), modules=len(timings),
                   seconds=round(time.monotonic() - start, 1), failed_modules=failed, **totals)
    if args.timings:
        args.timings.write_text(json.dumps(dict(summary=summary, modules=timings), indent=1, sort_keys=True) + '\n',
                                encoding='utf-8', newline='\n')
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
