"""Tests of perf-gemma-swe32's shared tbench locks (local only): tbench/bin/lockset.py takes gswe32's own lock plus
tb16's, swe32's, swe16's and swe10's (byte-identical copies of their lock modules) all or nothing: acquire / handoff / release /
status with live and stale owners, the rollback, absent roots, the roots from the environment or gswe32.env (none: the
set refuses); the copies against their halves' originals, gswe32.env and build.py's roots; host_check.py, preflight.py
and the link's lock wait on the set's status; and the real scripts under Git Bash on fake_tbench's half: run_pool.sh
and start_eval.sh refused beside a held tb16 or swe32 lock, a pool and the detached evaluation holding the whole set
until they end (the other halves' own modules find their lock held meanwhile), stop_pool.sh releasing it.

Run: python -B -m unittest test_tbench_lockset -v   (TEMP/TMP/TMPDIR on D:, localtemp.py)
"""
import ast
import base64
import contextlib
import hashlib
import http.server
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT)]
import localtemp  # noqa: E402

localtemp.ensure()
import fake_tbench  # noqa: E402
import tbench_link  # noqa: E402

BIN = ROOT / 'tbench' / 'bin'
SHARED = ('tb16', 'swe32', 'swe16', 'swe10')  # acquisition order after gswe32's own lock
ROOTS_LINE = 'GSWE32_SHARED_LOCKS="tb16=/home/ubuntu/tb16 swe32=/home/ubuntu/swe32 swe16=/home/ubuntu/swe16 ' \
             'swe10=/home/ubuntu/swe10"'
# where each copied module's original lives (the half that owns it) and perf-tb16's copies (the copies' source)
ORIGINALS = {'tb16': 'perf-tb16', 'swe32': 'perf-swe32', 'swe16': 'perf-swe16', 'swe10': 'perf-swe10'}
INSTANCE = 'astropy__astropy-12907'
TBENCH_CONFIG = json.loads((ROOT / 'pins.json').read_text(encoding='utf-8'))['tbench']
_loaded = [0]


def load(path):
    """A module from a file under a private name (never cached in sys.modules)."""
    _loaded[0] += 1
    spec = importlib.util.spec_from_file_location('lockset_test_%d' % _loaded[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def temp():
    return tempfile.TemporaryDirectory(prefix='gls-', ignore_cleanup_errors=True)


# ---------------------------------------------------------------- lockset.py in process (owners' liveness by pid set)


class Lockset(unittest.TestCase):
    def setUp(self):
        self.tmp = temp()
        base = Path(self.tmp.name)
        self.ls, self.live, cache = load(BIN / 'lockset.py'), set(), {}
        original = self.ls.module

        def module(name):  # one copy per name, its liveness from self.live (no processes needed)
            if name not in cache:
                cache[name] = original(name)
                cache[name].alive = lambda owner: owner.get('pid') in self.live
            return cache[name]
        self.ls.module, self.mod = module, module
        self.root = base / 'gswe32'
        self.roots = {n: base / n for n in SHARED}
        for path in [self.root] + list(self.roots.values()):
            path.mkdir()
        self.env = {'GSWE32_SHARED_LOCKS': ' '.join('%s=%s' % (n, fake_tbench.posix(p)) for n, p in self.roots.items())}

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, name):
        return self.root if name == 'gswe32' else self.roots[name]

    def owner(self, name):
        path = self.path(name) / 'lock' / 'tbench.lock' / 'owner.json'
        return json.loads(path.read_text()) if path.is_file() else None

    def test_acquire_status_release_every_lock(self):
        self.live.add(101)
        code, value = self.ls.acquire(self.root, 'pool', 'gswe32-1-r1', 101, self.env)
        self.assertEqual((code, value['event'], value['acquired'], value['skipped']),
                         (0, 'gswe32_locks_acquired', ['gswe32'] + list(SHARED), []))
        for name in ('gswe32',) + SHARED:
            self.assertEqual((self.owner(name)['kind'], self.owner(name)['run_id'], self.owner(name)['pid']),
                             ('pool', 'gswe32-1-r1', 101))
        status = self.ls.status(self.root, self.env)
        self.assertEqual((status['event'], status['held'], status['alive'], status['kind'], status['run_id'],
                          status['pid'], status['holder'], status['eval_only'], len(status['blocking'])),
                         ('gswe32_locks', True, True, 'pool', 'gswe32-1-r1', 101, 'gswe32', False, 5))
        self.assertEqual(sorted(status['locks']), sorted(('gswe32',) + SHARED))
        self.assertIsNotNone(status['age_s'])
        code, value = self.ls.acquire(self.root, 'eval', 'gswe32-1-r1', 102, self.env)  # never a second holder
        self.assertEqual((code, value['event'], value['holder'], value['released']),
                         (3, 'gswe32_locks_held', 'gswe32', []))
        self.assertEqual(self.ls.release(self.root, 'pool', 'gswe32-9-r1', self.env)[1]['released'], [])
        self.assertEqual(self.ls.release(self.root, 'eval', 'gswe32-1-r1', self.env)[1]['released'], [])
        code, value = self.ls.release(self.root, 'pool', 'gswe32-1-r1', self.env)
        self.assertEqual((code, value['released']), (0, list(reversed(SHARED)) + ['gswe32']))
        status = self.ls.status(self.root, self.env)
        self.assertEqual((status['held'], status['kind'], status['eval_only'], status['blocking']),
                         (False, None, False, []))
        self.assertFalse(any((self.path(n) / 'lock' / 'tbench.lock').exists() for n in ('gswe32',) + SHARED))

    def test_all_or_nothing_beside_a_held_lock(self):
        self.live.update((101, 202, 303))
        self.assertEqual(self.mod('swe16').acquire(str(self.roots['swe16']), 'eval', 'swe16-1-r1', 202)[0], 0)
        code, value = self.ls.acquire(self.root, 'pool', 'gswe32-1-r1', 101, self.env)
        self.assertEqual((code, value['holder'], value['released'], value['held']['run_id']),
                         (3, 'swe16', ['gswe32', 'tb16', 'swe32'], 'swe16-1-r1'))
        for name in ('gswe32', 'tb16', 'swe32', 'swe10'):  # rolled back (or never reached)
            self.assertIsNone(self.owner(name), name)
        self.assertEqual(self.owner('swe16')['pid'], 202)  # the holder's lock untouched
        status = self.ls.status(self.root, self.env)  # an evaluation only: a job waits for it
        self.assertEqual((status['held'], status['kind'], status['holder'], status['eval_only'], status['blocking']),
                         (True, 'eval', 'swe16', True, []))
        self.assertEqual(self.mod('tb16').acquire(str(self.roots['tb16']), 'pool', 'tb16-2-r1', 303)[0], 0)
        status = self.ls.status(self.root, self.env)  # a live TB run besides: refused at once
        self.assertEqual((status['holder'], status['kind'], status['eval_only'], status['blocking']),
                         ('tb16', 'pool', False, [dict(name='tb16', kind='pool', run_id='tb16-2-r1')]))
        self.assertEqual(self.ls.acquire(self.root, 'eval', 'gswe32-1-r1', 101, self.env)[1]['holder'], 'tb16')
        self.assertIsNone(self.owner('gswe32'))

    def test_stale_owners_are_removed(self):
        self.live.add(101)
        self.assertEqual(self.mod('swe32').acquire(str(self.roots['swe32']), 'pool', 'swe32-1-r1', 999)[0], 0)
        self.assertEqual(self.mod('gswe32').acquire(str(self.root), 'eval', 'gswe32-0-r1', 998)[0], 0)
        status = self.ls.status(self.root, self.env)
        self.assertEqual((status['held'], status['locks']['swe32']['held'], status['locks']['swe32']['alive']),
                         (False, True, False))
        code, value = self.ls.acquire(self.root, 'pool', 'gswe32-1-r1', 101, self.env)
        self.assertEqual((code, value['acquired']), (0, ['gswe32'] + list(SHARED)))
        self.assertEqual((self.owner('swe32')['pid'], self.owner('gswe32')['run_id']), (101, 'gswe32-1-r1'))

    def test_absent_roots_are_skipped_and_reported(self):
        self.live.add(101)
        self.roots['swe10'].rmdir()
        code, value = self.ls.acquire(self.root, 'eval', 'gswe32-1-r1', 101, self.env)
        self.assertEqual((code, value['acquired'], value['skipped']),
                         (0, ['gswe32', 'tb16', 'swe32', 'swe16'], ['swe10']))
        status = self.ls.status(self.root, self.env)
        self.assertEqual(status['locks']['swe10'], dict(root=fake_tbench.posix(self.roots['swe10']), absent=True,
                                                        held=False, alive=False))
        self.assertEqual(self.ls.handoff(self.root, 'eval', 'gswe32-1-r1', 102, self.env)[1]['skipped'], ['swe10'])
        self.assertEqual(self.ls.release(self.root, 'eval', 'gswe32-1-r1', self.env)[1]['released'],
                         ['swe16', 'swe32', 'tb16', 'gswe32'])
        self.assertFalse(self.roots['swe10'].exists())  # an absent half is never created

    def test_handoff_passes_every_lock_or_none(self):
        self.live.update((101, 404))
        self.assertEqual(self.ls.acquire(self.root, 'eval', 'gswe32-1-r1', 101, self.env)[0], 0)
        code, value = self.ls.handoff(self.root, 'eval', 'gswe32-1-r1', 404, self.env)
        self.assertEqual((code, value['handed']), (0, ['gswe32'] + list(SHARED)))
        self.assertEqual({self.owner(n)['pid'] for n in ('gswe32',) + SHARED}, {404})
        self.assertEqual(self.ls.handoff(self.root, 'eval', 'gswe32-2-r1', 505, self.env)[0], 2)
        self.assertEqual(self.ls.handoff(self.root, 'pool', 'gswe32-1-r1', 505, self.env)[0], 2)
        self.mod('swe32').write_owner(str(self.roots['swe32']), 'eval', 'swe32-7-r1', 606)  # one lock not ours
        code, value = self.ls.handoff(self.root, 'eval', 'gswe32-1-r1', 505, self.env)
        self.assertEqual((code, value['lock']), (2, 'swe32'))
        self.assertEqual(self.owner('gswe32')['pid'], 404)  # nothing passed on
        self.assertEqual(self.ls.release(self.root, 'eval', 'gswe32-1-r1', self.env)[1]['released'],
                         ['swe10', 'swe16', 'tb16', 'gswe32'])
        self.assertEqual(self.owner('swe32')['run_id'], 'swe32-7-r1')

    def test_an_error_rolls_back(self):
        self.live.add(101)
        (self.roots['swe32'] / 'lock').write_text('not a directory')  # the lock directory cannot be made
        code, value = self.ls.acquire(self.root, 'pool', 'gswe32-1-r1', 101, self.env)
        self.assertEqual((code, value['event'], value['holder'], value['released']),
                         (2, 'gswe32_locks_error', 'swe32', ['gswe32', 'tb16']))
        self.assertIsNone(self.owner('gswe32'))
        self.assertIsNone(self.owner('tb16'))

    def test_roots_from_the_environment_or_gswe32_env(self):
        names = lambda value: [n for n, _ in value]
        self.assertEqual(names(self.ls.shared_roots(self.env, self.root)), list(SHARED))
        with self.assertRaises(ValueError):  # neither: never only our own lock by accident
            self.ls.shared_roots({}, self.root)
        (self.root / 'gswe32.env').write_text('# x\nHOST_PYTHON=/usr/bin/python3\nexport GSWE32_SHARED_LOCKS="%s"\n'
                                              % self.env['GSWE32_SHARED_LOCKS'])
        self.assertEqual(self.ls.shared_roots({}, self.root), self.ls.shared_roots(self.env))
        self.assertEqual(self.ls.shared_roots({'GSWE32_SHARED_LOCKS': ''}, self.root), [])  # the environment wins
        for bad in ('swe32=relative/path', 'swe32=/a swe32=/b', 'gswe32=/home/ubuntu/gswe32', '=/a', 'a-b=/x'):
            with self.assertRaises(ValueError, msg=bad):
                self.ls.shared_roots({'GSWE32_SHARED_LOCKS': bad})
        build = ast.parse((ROOT / 'build.py').read_text(encoding='utf-8'))
        roots = next(ast.literal_eval(n.value) for n in build.body if isinstance(n, ast.Assign)
                     and any(getattr(t, 'id', None) == 'SHARED_LOCK_ROOTS' for t in n.targets))
        self.assertEqual(tuple(n for n, _ in roots), SHARED)
        self.assertEqual('GSWE32_SHARED_LOCKS="%s"' % ' '.join('%s=%s' % kv for kv in roots), ROOTS_LINE)

    def test_cli_refuses_without_roots_and_bad_kinds(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            saved = os.environ.pop('GSWE32_SHARED_LOCKS', None)
            try:
                code = self.ls.main(['lockset.py', str(self.root), 'acquire', 'pool', 'gswe32-1-r1', '101'])
                bad = self.ls.main(['lockset.py', str(self.root), 'acquire', 'smoke', 'gswe32-1-r1', '101'])
            finally:
                if saved is not None:
                    os.environ['GSWE32_SHARED_LOCKS'] = saved
        self.assertEqual((code, bad), (2, 2))
        self.assertEqual(json.loads(out.getvalue().splitlines()[0])['event'], 'gswe32_locks_error')
        self.assertIsNone(self.owner('gswe32'))  # nothing taken


# ---------------------------------------------------------------- the copies, the env file, the deployed file set


class Copies(unittest.TestCase):
    def test_lock_modules_are_byte_identical_copies(self):
        for name in SHARED:
            mine = BIN / ('%s_lock.py' % name)
            self.assertTrue(mine.is_file(), name)
            for harness in ('perf-tb16', ORIGINALS[name]):
                other = ROOT.parent / harness / 'tbench' / 'bin' / mine.name
                if other.is_file():
                    self.assertEqual(sha(mine), sha(other), '%s vs %s' % (mine, other))
        self.assertEqual(load(BIN / 'lockset.py').OWN, 'gswe32')
        self.assertTrue((BIN / 'gswe32_lock.py').is_file())

    def test_env_file_and_tbench_files(self):
        lines = (ROOT / 'tbench' / 'gswe32.env').read_text().splitlines()
        self.assertEqual([l for l in lines if l.startswith('GSWE32_SHARED_LOCKS=')], [ROOTS_LINE])
        self.assertEqual(fake_tbench.SHARED, SHARED)
        for path in (ROOT / 'tbench').rglob('*'):
            self.assertNotEqual(path.name, '__pycache__', path)
            if path.is_file():
                self.assertNotIn(b'\r', path.read_bytes(), path)
        for name in ('run_pool.sh', 'stop_pool.sh', 'start_eval.sh', 'evaluate_run.sh', 'package_run.sh',
                     'lock_status.sh', 'stress_test.sh', 'preflight.py', 'host_check.py'):
            text = (BIN / name).read_text()
            self.assertNotIn('gswe32_lock.py', text, name)  # the set only, never our own lock alone
        self.assertIn('lock() { hpy "$GSWE32_BIN/lockset.py"', (BIN / 'lib.sh').read_text())


# ---------------------------------------------------------------- host_check.py and the link on the set's status


def set_status(**holders):
    """lockset.py status for {name: kind} live holders (in the set's order)."""
    locks, live = {}, []
    for name in ('gswe32',) + SHARED:
        kind = holders.get(name)
        locks[name] = dict(held=bool(kind), alive=bool(kind), kind=kind, run_id='%s-1-r1' % name if kind else None,
                           pid=7 if kind else None)
        if kind:
            live.append(dict(locks[name], name=name))
    blocking = [dict(name=v['name'], kind=v['kind'], run_id=v['run_id']) for v in live if v['kind'] != 'eval']
    first = live[0] if live else {}
    return dict(event='gswe32_locks', held=bool(live), alive=bool(live), kind=first.get('kind'),
                run_id=first.get('run_id'), pid=first.get('pid'), age_s=1.0 if live else None, holder=first.get('name'),
                eval_only=bool(live) and not blocking, blocking=blocking, locks=locks)


class HostCheckOnTheSet(unittest.TestCase):
    def check(self, containers, mode, lock, available_kb=5000000):
        module = load(BIN / 'host_check.py')
        with temp() as tmp:
            root = Path(tmp)
            proc = root / 'proc'
            proc.mkdir()
            (proc / 'loadavg').write_text('1.0 1.0 1.0 1/1 1\n')
            (proc / 'meminfo').write_text('MemTotal: 30000000 kB\nMemAvailable: %d kB\n' % available_kb)
            (root / 'gswe32.env').write_text('GSWE32_MIN_MEM_GB=12\nGSWE32_MIN_FREE_GB=0\nGSWE32_HOST_PROC=%s\n' % proc)
            table = ''.join('%s\t%s\timg\tUp\n' % ('c%011d' % i, n) for i, n in enumerate(containers))
            module.run = lambda argv, timeout=60: (0, table, '') if 'ps' in argv[:3] else (1, '', '')
            module.lock_state = lambda root: lock
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = module.main(str(root), mode)
            return code, json.loads(out.getvalue())

    def test_evaluations_of_any_half_and_our_containers(self):
        ours = 'sweb.eval.%s.gswe32-1-r1-full' % INSTANCE
        code, value = self.check([ours], 'preflight', set_status(gswe32='eval', tb16='eval', swe32='eval'))
        self.assertEqual((code, value['status'], value['our_evaluation_running'], value['evaluation_running']),
                         (0, 'HOST_FREE', True, True))
        # another half's evaluation: low memory reported only (the job waits for it), our evaluator's container is a
        # leftover (no evaluation of ours holds our own lock)
        code, value = self.check([], 'preflight', set_status(swe32='eval'))
        self.assertEqual((value['status'], value['our_evaluation_running'], value['evaluation_running']),
                         ('HOST_FREE', False, True))
        code, value = self.check([ours], 'preflight', set_status(swe32='eval'))
        self.assertEqual((value['status'], value['our_leftovers']), ('HOST_BUSY', 1))
        # a live TB run among the holders, or the strict check: memory refuses
        self.assertEqual(self.check([], 'preflight', set_status(swe32='eval', tb16='pool'))[1]['status'], 'HOST_BUSY')
        self.assertEqual(self.check([], 'strict', set_status(swe32='eval'))[1]['status'], 'HOST_BUSY')
        self.assertEqual(self.check([], 'preflight', None)[1]['status'], 'HOST_BUSY')
        self.assertEqual(self.check([], 'strict', set_status(), 25000000)[1]['status'], 'HOST_FREE')


class LinkWaitOnTheSet(unittest.TestCase):
    class Link(tbench_link.Link):
        def __init__(self, answers, event='gswe32_locks'):
            super().__init__('/u/ycao11/gswe32-x', TBENCH_CONFIG)
            self.answers, self.t, self.event = list(answers), [0.0], event
            self.clock = lambda: self.t[0]
            self.sleep = lambda s: self.t.__setitem__(0, self.t[0] + s)

        def call(self, label, remote, timeout):
            value = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
            stdout = json.dumps(dict(value, event=self.event))
            return dict(label=label, argv=remote, returncode=0, timed_out=False, seconds=0.0, stdout=stdout, stderr='')

    def test_waits_while_only_evaluations_hold_the_set(self):
        link = self.Link([set_status(swe32='eval'), set_status(gswe32='eval', swe16='eval'), set_status()])
        lines = []
        value = link.wait_for_lock(100.0, 30, log=lines.append)
        self.assertEqual((value['status'], value['waited_s'], len(value['polls'])), ('LOCK_FREE', 60.0, 3))
        self.assertEqual([(p['holder'], p['eval_only']) for p in value['polls']],
                         [('swe32', True), ('gswe32', True), (None, False)])
        self.assertEqual(json.loads(lines[0])['holder'], 'swe32')
        self.assertEqual(self.Link([set_status(swe32='eval')]).wait_for_lock(50.0, 30, log=lines.append)['status'],
                         'LOCK_HELD_TIMEOUT')

    def test_refuses_at_once_beside_a_live_run(self):
        for holders in (dict(tb16='pool'), dict(swe10='stress'), dict(gswe32='pool'), dict(swe32='eval', swe16='pool')):
            value = self.Link([set_status(**holders)]).wait_for_lock(5000.0, 30, log=lambda line: None)
            self.assertEqual((value['status'], len(value['polls'])), ('LOCK_HELD_BY_LIVE_RUN', 1), holders)
        stale = dict(set_status(), locks=dict(set_status()['locks'], swe32=dict(held=True, alive=False, kind='pool')))
        self.assertEqual(self.Link([stale]).wait_for_lock(0.0, 30, log=lambda line: None)['status'], 'LOCK_FREE')

    def test_reads_only_the_set_event(self):
        self.assertEqual(self.Link([set_status()]).lock_status()['status'], 'LOCK_READ')
        self.assertEqual(self.Link([set_status()], event='gswe32_lock').lock_status()['status'], 'LOCK_UNKNOWN')
        self.assertEqual(self.Link([set_status()], event='gswe32_lock').wait_for_lock(9.0, 1, log=print)['status'],
                         'LOCK_UNKNOWN')


# ---------------------------------------------------------------- the real scripts under Git Bash (fake_tbench's half)


class Model(http.server.BaseHTTPRequestHandler):
    """A model endpoint for the fake agents: /v1/models, chat completions, session close."""
    count = 0

    def log_message(self, *args):
        pass

    def reply(self, value, code=200):
        body = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(dict(object='list', data=[dict(id='selective-prefix')]) if self.path.startswith('/v1/models')
                   else dict(error='not found'), 200 if self.path.startswith('/v1/models') else 404)

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length') or 0))
        if self.path.endswith('/chat/completions'):
            Model.count += 1
            self.reply(dict(id='chatcmpl-%d' % Model.count, created=0, usage=dict(prompt_tokens=1, completion_tokens=1),
                            choices=[dict(index=0, message=dict(role='assistant', content='THOUGHT: look.'))]))
        elif self.path.endswith('/sessions/close'):
            self.reply(dict(closed=True))
        else:
            self.reply(dict(error='not found'), 404)


def pool_spec(port, steps, label='full'):
    """run_pool.sh's spec (controller.pool_spec's schema) against 127.0.0.1:<port>, K = 1."""
    up = ROOT / 'tbench' / 'upstream'
    override = '\n'.join(['model:', '  model_name: "openai/selective-prefix"', '  model_kwargs:',
                          '    api_base: "http://127.0.0.1:%d/v1"' % port, '    timeout: 60', '', 'agent:',
                          '  step_limit: %d' % steps, ''])
    spec = dict(schema='gswe32-pool-spec-v1', label=label, kind='runtime', max_running=1, step_limit=steps,
                deadline_epoch=int(time.time()) + 3600, runner_sha256=sha(up / 'swebench_runner.py'),
                prompt_file='swebench_prompt_override.yaml', prompt_sha256=sha(up / 'swebench_prompt_override.yaml'),
                mini_retry_attempts=1, request_timeout=60, close=True, model_override=override)
    raw = json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip('=')


def last_json(text):
    lines = [l for l in text.strip().splitlines() if l.startswith('{')]
    return json.loads(lines[-1]) if lines else {}


class Scripts(unittest.TestCase):
    def setUp(self):
        self.tmp = temp()
        self.base = Path(self.tmp.name)
        self.holders, self.server, self.background = [], None, []
        tasks = self.base / 'tasks.json'
        tasks.write_text(json.dumps([dict(instance_id=INSTANCE, problem_statement='Fix the separability matrix.')]))
        self.tasks = tasks

    def tearDown(self):
        for process in self.holders:
            self.kill_tree(process)
        for process in self.background:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        self.tmp.cleanup()

    def deploy(self, **options):
        return fake_tbench.deploy(self.base / 'tbench', self.tasks, **options)

    def bash(self, root, script, *args, env=None, timeout=300):
        environ = dict(os.environ, GSWE32_ROOT=fake_tbench.posix(root), GSWE32_ALLOW_TCP='1', **(env or {}))
        argv = [fake_tbench.BASH, fake_tbench.posix(Path(root) / 'bin' / script)] + [str(a) for a in args]
        return subprocess.run(argv, env=environ, capture_output=True, text=True, timeout=timeout)

    def status(self, root):
        done = self.bash(root, 'lock_status.sh')
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return last_json(done.stdout)

    def half(self, root, name):
        """(that half's root, its own lock module) as the half itself would use it."""
        path = Path(root) if name == 'gswe32' else fake_tbench.shared_root(root, name)
        return str(path), load(path / 'bin' / ('%s_lock.py' % name))

    def hold(self, root, name, kind, run_id, seconds=None):
        """A live owner of one half's lock through that half's own module (a bash sleep; None: until killed)."""
        path, module = self.half(root, name)
        py = fake_tbench.msys(sys.executable)
        mod = fake_tbench.posix(Path(path) / 'bin' / ('%s_lock.py' % name))
        script = ('"%s" -I -B "%s" "%s" acquire %s %s $$ > /dev/null && sleep %s && "%s" -I -B "%s" "%s" release %s %s'
                  % (py, mod, fake_tbench.posix(path), kind, run_id, seconds or 3600, py, mod, fake_tbench.posix(path),
                     kind, run_id))
        process = subprocess.Popen([fake_tbench.BASH, '-c', script], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.holders.append(process)
        deadline = time.time() + 30
        while time.time() < deadline:
            value = module.status(path)
            if value.get('held') and value.get('alive') and value.get('run_id') == run_id:
                process.msys_pid = value['pid']
                return process
            time.sleep(0.2)
        raise RuntimeError('the holder of %s did not take the lock' % name)

    @staticmethod
    def kill_tree(process):
        """The holder and its MSYS `sleep` (a plain kill of the bash leaves the sleep; e2e_gswe32.kill_tree)."""
        msys_pid = getattr(process, 'msys_pid', None)
        if os.name == 'nt' and msys_pid:
            script = ("for p in $(ps | awk -v parent=%d '$2 == parent {print $1}'); do kill -9 $p; done; kill -9 %d"
                      % (msys_pid, msys_pid))
            subprocess.run([fake_tbench.BASH, '-c', script], capture_output=True, timeout=60)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=30)

    def lock_dirs(self, root):
        return {n: (Path(self.half(root, n)[0]) / 'lock' / 'tbench.lock').exists() for n in ('gswe32',) + SHARED}

    def pool_output(self, root, run, label='full'):
        out = Path(root) / 'runs' / run / label
        out.mkdir(parents=True)
        (out / 'instances.txt').write_text(INSTANCE + '\n')
        (out / 'preds.json').write_text(json.dumps({INSTANCE: dict(model_name_or_path='gswe32', instance_id=INSTANCE,
                                                                   model_patch='diff --git a/f b/f\n')}))

    def assert_refused(self, root, holder):
        done = self.bash(root, 'run_pool.sh', 'gswe32-5-r1', 'full', 'runtime', 25001, '-', pool_spec(25001, 4),
                         INSTANCE)
        value = last_json(done.stdout)
        self.assertEqual((done.returncode, value.get('event')), (4, 'gswe32_pool_refused'), done.stdout + done.stderr)
        self.assertIn('tbench lock held', value['reason'])
        refused = json.loads((Path(root) / 'runs' / 'gswe32-5-r1' / 'full' / 'lock-refused.json').read_text())
        self.assertEqual((refused['event'], refused['holder']), ('gswe32_locks_held', holder))
        self.assertFalse((Path(root) / 'runs' / 'gswe32-5-r1' / 'full' / 'logs').exists())  # no agent started
        self.pool_output(root, 'gswe32-6-r1')
        done = self.bash(root, 'start_eval.sh', 'gswe32-6-r1', 2, 60, 600, 'full')
        value = last_json(done.stdout)
        self.assertEqual((done.returncode, value.get('event'), value.get('reason')),
                         (4, 'gswe32_eval_refused', 'tbench lock held'), done.stdout + done.stderr)
        self.assertEqual(value['holder']['holder'], holder)
        dirs = self.lock_dirs(root)
        self.assertEqual([n for n in dirs if dirs[n]], [holder])  # every lock taken meanwhile rolled back

    def test_status_with_an_absent_root(self):
        root = self.deploy(absent=('swe10',))
        value = self.status(root)
        self.assertEqual((value['event'], value['held'], value['holder'], value['eval_only']),
                         ('gswe32_locks', False, None, False))
        self.assertEqual(sorted(value['locks']), sorted(('gswe32',) + SHARED))  # (sort_keys)
        self.assertTrue(value['locks']['swe10']['absent'])
        self.assertFalse(any(value['locks'][n].get('absent') for n in ('gswe32', 'tb16', 'swe32', 'swe16')))

    def test_refused_beside_a_tb16_run(self):
        root = self.deploy()
        self.hold(root, 'tb16', 'pool', 'tb16-3000001-r1')
        value = self.status(root)
        self.assertEqual((value['held'], value['holder'], value['kind'], value['eval_only'], value['blocking']),
                         (True, 'tb16', 'pool', False, [dict(name='tb16', kind='pool', run_id='tb16-3000001-r1')]))
        self.assert_refused(root, 'tb16')

    def test_refused_beside_a_swe32_evaluation_which_a_job_waits_for(self):
        root = self.deploy()
        holder = self.hold(root, 'swe32', 'eval', 'swe32-3000002-r1')
        value = self.status(root)
        self.assertEqual((value['holder'], value['kind'], value['eval_only'], value['blocking']),
                         ('swe32', 'eval', True, []))
        self.assert_refused(root, 'swe32')
        self.kill_tree(holder)
        self.hold(root, 'swe32', 'eval', 'swe32-3000003-r1', 8)  # an evaluation ending in ~8 s
        link = tbench_link.Link(self.base / 'stage', TBENCH_CONFIG, env=dict(os.environ, GSWE32_FAKE_ROOT=str(root)),
                                ssh=[sys.executable, '-B', str(ROOT / 'fake_tbench.py'), 'ssh'])
        waited = link.wait_for_lock(time.monotonic() + 120, 1, log=lambda line: None)
        self.assertEqual(waited['status'], 'LOCK_FREE', waited)
        self.assertEqual((waited['polls'][0]['holder'], waited['polls'][0]['eval_only']), ('swe32', True))
        self.hold(root, 'swe16', 'pool', 'gemma0-smoke-1')  # perf-gemma0's smoke takes swe16's lock
        self.assertEqual(link.wait_for_lock(time.monotonic() + 120, 1, log=lambda line: None)['status'],
                         'LOCK_HELD_BY_LIVE_RUN')

    def test_the_evaluation_holds_the_whole_set_until_it_ends(self):
        root = self.deploy()
        run = 'gswe32-4000001-r1'
        self.pool_output(root, run)
        done = self.bash(root, 'start_eval.sh', run, 2, 60, 600, 'full', env=dict(GSWE32_FAKE_EVAL_SECONDS='25'))
        started = last_json(done.stdout)
        self.assertEqual((done.returncode, started.get('event')), (0, 'gswe32_eval_started'), done.stdout + done.stderr)
        value = self.status(root)
        self.assertEqual((value['holder'], value['kind'], value['run_id'], value['eval_only']),
                         ('gswe32', 'eval', run, True))
        self.assertEqual({(n, v['kind'], v['run_id'], v['pid'], v['alive']) for n, v in value['locks'].items()},
                         {(n, 'eval', run, started['pid'], True) for n in ('gswe32',) + SHARED})
        for name in ('gswe32',) + SHARED:  # each half's own scripts find their lock held by a live owner
            path, module = self.half(root, name)
            code, held = module.acquire(path, 'pool', '%s-9-r1' % name, os.getpid())
            self.assertEqual((code, held['holder']['run_id']), (3, run), name)
        self.assert_refused_by_eval(root, run)
        package = self.bash(root, 'package_run.sh', run)
        self.assertEqual((package.returncode, last_json(package.stdout).get('status')), (3, 'EVAL_RUNNING'),
                         package.stdout)
        deadline = time.time() + 240
        state = None
        while time.time() < deadline:
            status_file = Path(root) / 'runs' / run / 'eval-status.json'
            state = json.loads(status_file.read_text()).get('state') if status_file.is_file() else None
            if state in ('done', 'failed') and not self.status(root)['held']:
                break
            time.sleep(1)
        self.assertEqual(state, 'done')
        self.assertFalse(any(self.lock_dirs(root).values()), self.lock_dirs(root))

    def assert_refused_by_eval(self, root, run):
        done = self.bash(root, 'run_pool.sh', 'gswe32-5-r1', 'full', 'runtime', 25001, '-', pool_spec(25001, 4),
                         INSTANCE)
        value = last_json(done.stdout)
        self.assertEqual(done.returncode, 4, done.stdout + done.stderr)
        self.assertIn('tbench lock held', value['reason'])
        refused = json.loads((Path(root) / 'runs' / 'gswe32-5-r1' / 'full' / 'lock-refused.json').read_text())
        self.assertEqual((refused['holder'], refused['held']['run_id']), ('gswe32', run))

    def test_a_pool_holds_the_set_until_stop_pool_releases_it(self):
        root = self.deploy()
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Model)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        port = self.server.server_address[1]
        run = 'gswe32-4000002-r1'
        environ = dict(os.environ, GSWE32_ROOT=fake_tbench.posix(root), GSWE32_ALLOW_TCP='1', GSWE32_FAKE_STEPS='60',
                       GSWE32_FAKE_TOOL_SECONDS='1.0')
        with (self.base / 'pool.stdout').open('wb') as log:  # a file: runners left behind never hold a pipe open
            pool = subprocess.Popen([fake_tbench.BASH, fake_tbench.posix(Path(root) / 'bin' / 'run_pool.sh'), run,
                                     'full', 'runtime', str(port), '-', pool_spec(port, 60), INSTANCE], env=environ,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        self.background.append(pool)
        out = Path(root) / 'runs' / run / 'full'
        deadline = time.time() + 120
        while time.time() < deadline and not ((out / 'pids').is_dir() and list((out / 'pids').glob('*.pid'))):
            self.assertIsNone(pool.poll(), (self.base / 'pool.stdout').read_text(errors='replace'))
            time.sleep(0.5)
        value = self.status(root)
        pid = int((out / 'pool.pid').read_text())
        self.assertEqual({(n, v['kind'], v['run_id'], v['pid']) for n, v in value['locks'].items()},
                         {(n, 'pool', run, pid) for n in ('gswe32',) + SHARED})
        self.assertEqual((value['eval_only'], len(value['blocking'])), (False, 5))
        path, module = self.half(root, 'tb16')
        self.assertEqual(module.acquire(path, 'pool', 'tb16-9-r1', os.getpid())[0], 3)  # a TB run finds tb16's held
        self.pool_output(root, 'gswe32-6-r1')
        self.assertEqual(self.bash(root, 'start_eval.sh', 'gswe32-6-r1', 2, 60, 600, 'full').returncode, 4)
        stopped = self.bash(root, 'stop_pool.sh', run, 'full')
        value = last_json(stopped.stdout)
        self.assertEqual((stopped.returncode, value.get('event'), value.get('pool_found')),
                         (0, 'gswe32_pool_stopped', True), stopped.stdout + stopped.stderr)
        release = json.loads((out / 'lock-release-job.json').read_text())
        # the set's release: by stop_pool.sh, or already by the pool's own EXIT trap on its SIGTERM (either way all)
        self.assertEqual((release['event'], release['kind'], release['run_id']), ('gswe32_locks_release', 'pool', run))
        self.assertIn(release['released'], ([], list(reversed(SHARED)) + ['gswe32']))
        try:
            pool.wait(timeout=120)
        except subprocess.TimeoutExpired:
            pool.kill()
        self.assertFalse(self.status(root)['held'])
        self.assertFalse(any(self.lock_dirs(root).values()), self.lock_dirs(root))

    def test_host_check_and_preflight_read_the_set(self):
        root = self.deploy()
        (Path(root) / 'fake' / 'proc' / 'meminfo').write_text('MemTotal: 30000000 kB\nMemAvailable: 5000000 kB\n')
        self.hold(root, 'swe32', 'eval', 'swe32-3000003-r1')
        done = self.bash(root, 'host_check.sh', 'preflight')
        value = last_json(done.stdout)
        self.assertEqual((done.returncode, value['status'], value['evaluation_running'],
                          value['our_evaluation_running'], value['lock']['event'], value['lock']['holder']),
                         (0, 'HOST_FREE', True, False, 'gswe32_locks', 'swe32'), done.stdout + done.stderr)
        self.assertEqual(last_json(self.bash(root, 'host_check.sh', 'strict').stdout)['status'], 'HOST_BUSY')
        value = self.preflight(root)
        self.assertEqual(value['lock_modules'], {n: 'equal' for n in SHARED})
        self.assertEqual([n for n, _ in value['shared_roots']], list(SHARED))
        self.assertEqual((value['evaluation_running'], value['lock']['holder']), (True, 'swe32'))
        self.assertFalse([f for f in value['failures'] if 'tbench lock' in f or '_lock.py' in f or 'SHARED_LOCKS' in f],
                         value['failures'])
        swe16 = fake_tbench.shared_root(root, 'swe16') / 'bin' / 'swe16_lock.py'
        swe16.write_bytes(swe16.read_bytes() + b'# another protocol\n')
        (fake_tbench.shared_root(root, 'swe10') / 'lock').rmdir()
        (fake_tbench.shared_root(root, 'swe10') / 'bin' / 'swe10_lock.py').unlink()
        (fake_tbench.shared_root(root, 'swe10') / 'bin').rmdir()
        fake_tbench.shared_root(root, 'swe10').rmdir()
        self.hold(root, 'tb16', 'pool', 'tb16-3000004-r1')
        value = self.preflight(root)
        self.assertEqual(value['lock_modules'], dict(tb16='equal', swe32='equal', swe16='DIFFERENT', swe10='absent'))
        self.assertTrue(any('swe16_lock.py differs' in f for f in value['failures']), value['failures'])
        self.assertTrue(any('tbench lock held by a live run' in f and 'tb16' in f for f in value['failures']))
        self.assertTrue(value['lock']['locks']['swe10']['absent'])

    def preflight(self, root):
        """preflight.py's main in process (its subprocesses stubbed: no docker, no agent env here)."""
        saved_path, saved = list(sys.path), sys.modules.pop('lockset', None)
        try:
            module = load(Path(root) / 'bin' / 'preflight.py')
            module.run = lambda argv, timeout=120, env=None: (1, '', 'stubbed')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                module.main(str(root), 'a' * 64, 'b' * 64, ['%s=%s' % (INSTANCE, 'c' * 64)])
        finally:
            sys.path[:] = saved_path
            sys.modules.pop('lockset', None)
            if saved is not None:
                sys.modules['lockset'] = saved
        return last_json(out.getvalue())


if __name__ == '__main__':
    unittest.main()
