"""Tests of postrun/verify_on_tbench.py and tbench_verify/remote_verify.py (no connection; a local directory stands
in for the tbench root). Byte-identical in perf-decode3, perf-mab, perf-thinking-gg1 and perf-gdn-probe.

- the harness's kind (its marker) and its spec against its own fetch_and_verify.ps1: stage prefix, folder name, core
  (and bulk) archive, the verifier's Python flags and its options in the ps1's order, the synthetic flag, the tables;
- the code tree: every file the verifier compares or imports, nothing else (no docs, dist\\, validation\\, bundle);
- refusals before any connection; the DeltaAI-side copy command, also run in bash with stand-in ssh / scp (every kind,
  perf-mab --with-bulk with 0, 1 and 3 bulk archives; `bash -n -c` on each remote command and on the whole line);
  safe extraction of the pulled outputs;
- the whole manual path on the stand-in: code install, evidence upload, the detached run, the poll, the pull, every
  file against tbench's list (a non-archive core: the harness's own verifier says NOT_VERIFIED, exit 1), and the
  refusals of a different file already on tbench and of a core that is not the check block's.
"""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import verify_on_tbench as vot  # noqa: E402

rv = vot.remote_verify
KIND = vot.kind_of(ROOT)
SPEC = rv.SPECS[KIND]
PS1 = (HERE / 'fetch_and_verify.ps1').read_text(encoding='utf-8')
STAGE = '/u/ycao11/%s-%s' % (SPEC['prefix'], 'a' * 32)


def scratch():
    base = Path(os.environ.get('TMP') or r'D:\Remote-RA-temp\tmp')
    if str(base)[:2].upper() == 'C:':
        base = Path(r'D:\Remote-RA-temp\tmp')
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='tbv-test-', dir=str(base)))


class Spec(unittest.TestCase):
    def test_kind_and_marker(self):
        self.assertTrue(ROOT.name.startswith(SPEC['name']), (ROOT.name, SPEC['name']))
        others = [k for k, s in rv.SPECS.items() if k != KIND and (ROOT / s['marker']).exists()]
        self.assertEqual(others, [])
        self.assertEqual(vot.remote_root(KIND), '/home/ubuntu/%s-verify' % SPEC['prefix'])

    def test_spec_against_fetch_and_verify(self):
        self.assertIn("'^/u/ycao11/%s-[0-9a-f]{32}$'" % SPEC['prefix'], PS1)
        self.assertIn("('%s'+$JobId+'-'+(Get-Date -Format 'yyyyMMddTHHmmss'))" % SPEC['folder'], PS1)
        self.assertIn("'%s'" % SPEC['core'], PS1)
        if SPEC['bulk'] and SPEC['bulk'][0] == 'required':
            self.assertIn("'%s'" % SPEC['bulk'][1], PS1)
        if SPEC['bulk'] and SPEC['bulk'][0] == 'optional-glob':
            self.assertIn("'pmab-bulk-*.tar.xz'", PS1)
        flags = re.search(r"\$Arguments = @\(((?:'-[IB]',\s*)+)\$Verifier", PS1)
        self.assertIsNotNone(flags)
        self.assertEqual(tuple(re.findall(r"'(-[IB])'", flags.group(1))), SPEC['flags'])
        self.assertIn("'%s'" % SPEC['synthetic'], PS1)
        # the verifier's options in the ps1's order (perf-mab's --parity-python is not offered on tbench)
        block = PS1[PS1.index('$Arguments = @('):PS1.index('& $Python @Arguments')]
        wanted = [o for o in re.findall(r"'(--[a-z0-9-]+)'", block) if o != '--parity-python']
        a = mock.Mock(job_id='7', stage=STAGE, core_sha256='0' * 64, bulk_sha256='1' * 64, slurm_state='COMPLETED',
                      slurm_exit_code='0:0', synthetic=True, with_bulk=True)
        got = [str(c) for c in rv.verifier_command(KIND, 'python', ROOT, a, Path('e'), Path('c'))
               if str(c).startswith('--')]
        self.assertEqual(got, wanted)
        # the tables / analysis: the script, its flags, run on the check folder
        scripts = re.findall(r"\$(\w+) = Join-Path \$PSScriptRoot '(\w+\.py)'", PS1)
        calls = {name: re.search(r"& \$Python ((?:-[IB] )+)\$%s \$Check" % var, PS1) for var, name in scripts
                 if name != 'verify_evidence.py'}
        self.assertEqual(sorted(calls), sorted(s for _, s, _ in SPEC['tables']))
        for flags_, script, word in SPEC['tables']:
            self.assertIsNotNone(calls[script], script)
            self.assertEqual(tuple(calls[script].group(1).split()), flags_)
            self.assertIn(word, PS1)
            self.assertTrue((HERE / script).is_file())

    def test_specs_are_complete(self):
        for kind, spec in rv.SPECS.items():
            self.assertEqual(sorted(spec), sorted(('name', 'marker', 'prefix', 'folder', 'core', 'bulk', 'flags',
                                                   'synthetic', 'tables', 'extracted')), kind)
            self.assertIn('check/files', spec['extracted'])


class CodeTree(unittest.TestCase):
    def test_code_files(self):
        files = dict(vot.code_files(ROOT))
        prefix = ROOT.name + '/'
        self.assertIn('tbench_verify/remote_verify.py', files)
        names = {n[len(prefix):] for n in files if n.startswith(prefix)}
        for path in HERE.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                self.assertIn('postrun/' + path.relative_to(HERE).as_posix(), names)
        self.assertIn('pins.json', names)
        tree = ast.parse((HERE / 'verify_evidence.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') in ('LOCAL_FILES', 'SEALED_MODULES')
                                                    for t in node.targets):
                for item in ast.literal_eval(node.value):
                    if (ROOT / item).is_file():
                        self.assertIn(item, names)
        imported = set()
        for path in [HERE / 'verify_evidence.py'] + [HERE / s for _, s, _ in SPEC['tables']]:
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.Import):
                    imported.update(a.name.split('.')[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    imported.add(node.module.split('.')[0])
        for module in imported:
            if (ROOT / (module + '.py')).is_file():
                self.assertIn(module + '.py', names)
        for name in names:
            self.assertFalse(name.startswith(('dist/', 'validation/', 'fixtures/', 'tokenizer/')), name)
            self.assertFalse(name.endswith(('.md', '.txt', '.bundle')), name)

    def test_bundle_is_deterministic(self):
        base = scratch()
        try:
            one = vot.bundle(ROOT, base / 'a')
            two = vot.bundle(ROOT, base / 'b')
            self.assertEqual(one['tag'], two['tag'])
            self.assertEqual(vot.sha_file(one['tar']), vot.sha_file(two['tar']))
            with tarfile.open(one['tar']) as archive:
                self.assertEqual(len(archive.getmembers()), one['count'])
        finally:
            shutil.rmtree(base, ignore_errors=True)


class Refusals(unittest.TestCase):
    def setUp(self):
        self.base = scratch()
        self.common = ['--cache-root', str(self.base / 'cache'), '--out-root', str(self.base / 'out')]

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def refused(self, argv, pattern):
        with self.assertRaisesRegex(vot.Refused, pattern):
            vot.main(argv + self.common)

    def test_before_any_connection(self):
        good = ['--stage', STAGE, '--job-id', '7']
        self.refused(['--stage', '/u/ycao11/xx-' + 'a' * 32, '--job-id', '7', '--print-copy'], 'stage')
        self.refused(['--stage', STAGE + 'a', '--job-id', '7', '--print-copy'], 'stage')
        self.refused(['--stage', STAGE, '--job-id', '7x', '--print-copy'], 'job id')
        self.refused(good + ['--slurm-state', 'COMPLETED'], 'both')
        self.refused(good + ['--slurm-state', 'completed', '--slurm-exit-code', '0:0'], 'Slurm state')
        self.refused(good + ['--core-sha256', 'abc'], 'invalid --core-sha256')
        self.refused(good + ['--core-sha256', '0' * 64, '--synthetic'], 'local-root')
        if KIND != 'pmab':
            self.refused(good + ['--with-bulk'], 'perf-mab only')
        if KIND != 'gg1':
            self.refused(good + ['--bulk-sha256', '0' * 64], 'gg1 only')
        else:
            self.refused(good + ['--core-sha256', '0' * 64], 'bulk-sha256')
        self.refused(good, 'core-sha256')
        self.refused(good + ['--candidate-dir', str(self.base)], 'not exactly one known harness')

    def test_remote_arguments(self):
        good = dict(kind=KIND, harness=ROOT.name, job_id='7', stage=STAGE, stamp='20261114T120000',
                    core_sha256='0' * 64, bulk_sha256='1' * 64 if KIND == 'gg1' else '', with_bulk=False,
                    slurm_state='', slurm_exit_code='', code_tag='0123456789abcdef')
        # the working tree is not a code/<tag> tree: everything else passes, the tag is refused
        self.assertIn('code tag', rv.check_run_args(SimpleNamespace(**good)) or '')
        for field, value, word in (('stage', '/tmp/x', 'stage'), ('job_id', '7;', 'job id'), ('stamp', 'x', 'stamp'),
                                   ('core_sha256', 'x', 'core'), ('slurm_state', 'COMPLETED', 'both'),
                                   ('with_bulk', KIND != 'pmab', 'perf-mab only' if KIND != 'pmab' else 'code tag'),
                                   ('kind', 'nope', 'unknown kind')):
            self.assertIn(word, rv.check_run_args(SimpleNamespace(**dict(good, **{field: value}))) or '', field)


class CopyCommand(unittest.TestCase):
    def test_print_copy(self):
        argv = ['--stage', STAGE, '--job-id', '3999999', '--print-copy', '--cache-root', r'D:\Remote-RA-temp\tmp\x']
        out = io.StringIO()
        with mock.patch('sys.stdout', out):
            self.assertEqual(vot.main(argv), 0)
        text = out.getvalue().strip()
        self.assertTrue(text.startswith('( set -e -o pipefail; S=%s; J=3999999; D=/home/ubuntu/%s-verify/evidence/'
                                        'job$J;' % (STAGE, SPEC['prefix'])), text)
        self.assertTrue(text.endswith('sacct -j $J -X -n -o JobID,State,ExitCode )'))
        self.assertIn(vot.TBENCH['host_key'], text)
        self.assertIn('-i /u/ycao11/.ssh/tunnel_key', text)
        self.assertIn('StrictHostKeyChecking=yes', text)
        self.assertIn(SPEC['core'], text)
        self.assertNotIn(' exit', text.replace('EXIT', ''))
        self.assertEqual(text.count('('), text.count(')'))
        if SPEC['bulk'] and SPEC['bulk'][0] == 'required':
            self.assertIn(SPEC['bulk'][1], text)
        if KIND == 'pmab':
            self.assertNotIn('pmab-bulk', text)
            out = io.StringIO()
            with mock.patch('sys.stdout', out):
                vot.main(argv + ['--with-bulk'])
            self.assertIn('N="pmab-core.tar.xz $(ls pmab-bulk-*.tar.xz | tr \'\\n\' \' \')";', out.getvalue())

    def test_copy_line_runs_in_bash(self):
        """2026-11-15 (job 3601174, three bulk archives): the line runs in bash with stand-in ssh / scp / sacct /
        mktemp (a local directory for tbench); every remote command string the stand-in ssh gets and the whole line pass
        `bash -n -c`, and the archives arrive renamed. Every kind without bulk, perf-mab --with-bulk with 1 and 3 bulk
        archives, and with 0 (refused before any copy: ls fails under pipefail)."""
        bash = find_bash()
        self.assertTrue(bash, 'no bash found (Git for Windows bash.exe next to git, or bash on PATH)')
        # the check catches the 3601174 break: newlines between the names in the remote command
        self.assertNotEqual(self.syntax(bash, 'cd /x && for n in a.tar.xz\nb.tar.xz\nc.tar.xz; do mv $n.part $n; '
                                              'done && sha256sum a.tar.xz\nb.tar.xz\nc.tar.xz'), 0)
        bulk = ['pmab-bulk-smgga.tar.xz', 'pmab-bulk-smgoa.tar.xz', 'pmab-bulk-smgra.tar.xz']
        cases = [(kind, False, bulk[:2] if kind == 'pmab' else []) for kind in sorted(rv.SPECS)]
        cases += [('pmab', True, bulk[:1]), ('pmab', True, bulk), ('pmab', True, [])]
        for kind, with_bulk, present in cases:
            with self.subTest(kind=kind, with_bulk=with_bulk, bulk=len(present)):
                self.run_case(bash, kind, with_bulk, present)

    def syntax(self, bash, command):
        # the string goes in on stdin: Git bash's parsing of a Windows command line does not keep every quote
        return subprocess.run([bash, '-c', 'c=$(cat); exec bash -n -c "$c"'], input=command.encode('utf-8'),
                              capture_output=True, timeout=60).returncode

    def run_case(self, bash, kind, with_bulk, present):
        spec = rv.SPECS[kind]
        base = scratch()
        self.addCleanup(shutil.rmtree, base, True)
        fetch, root, tmp = base / 'stage' / 'fetch', base / 'root', base / 'tmp'
        for folder in (fetch, root, tmp):
            folder.mkdir(parents=True)
        names = [spec['core']] + ([spec['bulk'][1]] if spec['bulk'] and spec['bulk'][0] == 'required' else [])
        for name in names + present:
            (fetch / name).write_bytes(('%s %s\n' % (kind, name)).encode() * 64)
        expected = names + (sorted(present) if with_bulk else [])
        stage = '/u/ycao11/%s-%s' % (spec['prefix'], 'a' * 32)
        line = vot.copy_command(SimpleNamespace(stage=stage, job_id='3999999', with_bulk=with_bulk), kind)
        self.assertNotIn('\n', line)
        self.assertEqual(self.syntax(bash, line), 0, line)
        for old, new in (('S=%s;' % stage, 'S=%s;' % posix(base / 'stage')),
                         ('D=%s/evidence/' % vot.remote_root(kind), 'D=%s/evidence/' % posix(root))):
            self.assertEqual(line.count(old), 1, old)
            line = line.replace(old, new)
        prelude = ('export PATH="/usr/bin:/bin:$PATH"\n'
                   'mktemp() { : > "$TBV_TMP/known_hosts"; echo "$TBV_TMP/known_hosts"; }\n'
                   'ssh() { local c="${!#}"; printf \'%s\\0\' "$c" >> "$TBV_LOG"; bash -n -c "$c" || return 97; '
                   'bash -c "$c"; }\n'
                   'scp() { local a=("$@"); local n=${#a[@]}; cp -- "${a[n-2]}" "${a[n-1]#*:}"; }\n'
                   'sacct() { echo "$2 COMPLETED 0:0"; }\n')
        log, script = tmp / 'remote-commands', tmp / 'copy-line.sh'
        script.write_bytes((prelude + line + '\n').encode('utf-8'))
        done = subprocess.run([bash, posix(script)], cwd=str(base), capture_output=True, timeout=120,
                              stdin=subprocess.DEVNULL, env=dict(os.environ, TBV_LOG=posix(log), TBV_TMP=posix(tmp)))
        out, err = done.stdout.decode('utf-8', 'replace'), done.stderr.decode('utf-8', 'replace')
        remote = [c for c in (log.read_bytes().decode('utf-8') if log.is_file() else '').split('\0') if c]
        evidence = root / 'evidence' / 'job3999999'
        if with_bulk and not present:
            self.assertNotEqual(done.returncode, 0, out + err)
            self.assertIn('pmab-bulk-', err)
            self.assertEqual(remote, [])
            self.assertFalse(evidence.exists())
            return
        self.assertEqual(done.returncode, 0, out + err)
        self.assertEqual(len(remote), 2, remote)
        for command in remote:
            self.assertNotIn('\n', command)
            self.assertEqual(self.syntax(bash, command), 0, command)
        self.assertEqual(remote[1].split(' && ')[-1].split(), ['sha256sum'] + expected)
        self.assertEqual(sorted(p.name for p in evidence.iterdir()), sorted(expected))
        sums = ['%s  %s' % (hashlib.sha256((fetch / n).read_bytes()).hexdigest(), n) for n in expected]
        lines = [re.sub(r'^([0-9a-f]{64}) \*', r'\1  ', x) for x in out.splitlines()]   # Git bash: binary-mode '*'
        self.assertEqual(lines, sums + sums + ['3999999 COMPLETED 0:0'], err)


def find_bash():
    """Git for Windows' bin\\bash.exe next to git (never System32's WSL bash), or bash on PATH elsewhere."""
    if os.name != 'nt':
        return shutil.which('bash')
    git = shutil.which('git')
    for up in (Path(git).resolve().parents if git else ()):
        if (up / 'bin' / 'bash.exe').is_file() and (up / 'usr' / 'bin').is_dir():
            return str(up / 'bin' / 'bash.exe')
    return None


def posix(path):
    """A local path as bash sees it (Git for Windows: D:\\x -> /d/x)."""
    text = str(path)
    if os.name == 'nt':
        text = '/' + text[0].lower() + text[2:].replace('\\', '/')
    return text


class Extract(unittest.TestCase):
    def test_safe_extract(self):
        base = scratch()
        try:
            for name in ('../evil', '/abs', 'a\\b', 'c:x'):
                archive = base / 'bad.tar.xz'
                with tarfile.open(archive, 'w:xz') as stream:
                    info = tarfile.TarInfo(name)
                    info.size = 1
                    stream.addfile(info, io.BytesIO(b'x'))
                with self.assertRaises(vot.Refused):
                    vot.safe_extract(archive, base / 'out')
                archive.unlink()
            archive = base / 'link.tar.xz'
            with tarfile.open(archive, 'w:xz') as stream:
                info = tarfile.TarInfo('l')
                info.type, info.linkname = tarfile.SYMTYPE, '/etc/passwd'
                stream.addfile(info)
            with self.assertRaises(vot.Refused):
                vot.safe_extract(archive, base / 'out')
            self.assertFalse((base / 'out').exists())
        finally:
            shutil.rmtree(base, ignore_errors=True)


class Compare(unittest.TestCase):
    def test_compare_folders(self):
        base = scratch()
        try:
            def folder(name, archive, verdict, table, extra=b'x'):
                root = base / name
                (root / 'check' / 'files').mkdir(parents=True)
                (root / 'check' / 'tables').mkdir()
                (root / 'check' / 'files' / 'a.json').write_bytes(extra)
                (root / 'check' / 'tables' / 't.md').write_bytes(table)
                rows = [dict(check='one', status='PASS', detail=archive), dict(check='two', ok=True, detail=None)]
                (root / 'check' / 'local-verification.json').write_text(json.dumps(dict(
                    archive=archive, verdict=verdict, checks=rows)), encoding='utf-8')
                return root
            local = folder('local', r'I:\x\a.tar.xz', 'X_PASS_VERIFIED', b'same')
            same = folder('tb1', '/home/ubuntu/a.tar.xz', 'X_PASS_VERIFIED', b'same')
            shutil.rmtree(same / 'check' / 'files')
            (same / 'tbench-extracted.json').write_text(json.dumps(dict(files={'check/files/a.json': hashlib.sha256(
                b'x').hexdigest()})), encoding='utf-8')
            value = vot.compare_folders(local, same)
            self.assertTrue(value['same'], value)
            self.assertEqual(value['report_keys_differing'], ['archive', 'checks'])
            self.assertEqual(value['full_rows_differing'], ['one'])
            self.assertTrue(value['extracted_equal'])
            self.assertFalse(vot.compare_folders(local, folder('tb2', 'p', 'X_PASS_VERIFIED', b'other'))['same'])
            crlf = folder('crlf', r'I:\x\a.tar.xz', 'X_PASS_VERIFIED', b'a\r\nb\r\n')
            lf = folder('lf', '/home/ubuntu/a.tar.xz', 'X_PASS_VERIFIED', b'a\nb\n')
            value = vot.compare_folders(crlf, lf)
            self.assertTrue(value['same'], value)
            self.assertEqual(value['files_differing_in_line_endings_only'], ['check/tables/t.md'])
            self.assertFalse(vot.compare_folders(crlf, folder('lf2', 'p', 'X_PASS_VERIFIED', b'a\nc\n'))['same'])
            self.assertFalse(vot.compare_folders(local, folder('tb3', 'p', 'NOT_VERIFIED', b'same'))['same'])
            changed = folder('tb4', 'p', 'X_PASS_VERIFIED', b'same')
            shutil.rmtree(changed / 'check' / 'files')
            (changed / 'tbench-extracted.json').write_text(json.dumps(dict(files={})), encoding='utf-8')
            self.assertFalse(vot.compare_folders(local, changed)['same'])
            out = io.StringIO()
            with mock.patch('sys.stdout', out):
                self.assertEqual(vot.main(['--compare', str(local), str(same)]), 0)
            self.assertIn('TBENCH_LOCAL_COMPARE {', out.getvalue())
        finally:
            shutil.rmtree(base, ignore_errors=True)


class StandIn(unittest.TestCase):
    """The manual path end to end on a local stand-in root (the driver runs with this Python, detached)."""

    def setUp(self):
        self.base = scratch()
        self.core = self.base / 'in' / SPEC['core']
        self.core.parent.mkdir()
        self.core.write_bytes(b'not an archive\n')
        self.bulk = self.base / 'in' / 'gg1-bulk.tar.xz'
        self.bulk.write_bytes(b'not a bulk archive\n')
        self.argv = ['--stage', STAGE, '--job-id', '3999999', '--local-root', str(self.base / 'root'),
                     '--cache-root', str(self.base / 'cache'), '--out-root', str(self.base / 'out'),
                     '--local-core', str(self.core), '--core-sha256', vot.sha_file(self.core),
                     '--slurm-state', 'COMPLETED', '--slurm-exit-code', '0:0']
        if KIND == 'gg1':
            self.argv += ['--local-bulk', str(self.bulk), '--bulk-sha256', vot.sha_file(self.bulk)]

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_manual_path(self):
        out = io.StringIO()
        with mock.patch('sys.stdout', out):
            code = vot.main(self.argv)
        text = out.getvalue()
        self.assertEqual(code, 1, text[-3000:])
        self.assertIn('EVIDENCE_UPLOAD %s' % SPEC['core'], text)
        folders = list((self.base / 'out').iterdir())
        self.assertEqual(len(folders), 1)
        job = folders[0]
        self.assertTrue(job.name.startswith(SPEC['folder'] + '3999999-'))
        record = json.loads((job / 'tbench-verification.json').read_bytes())
        # the harness's own verifier on a file that is not an archive: a NOT_VERIFIED report (perf-decode3,
        # perf-mab) or a traceback without one (gg1, gdnp); exit 1 either way, as fetch_and_verify.ps1
        if (job / 'check' / 'local-verification.json').is_file():
            report = json.loads((job / 'check' / 'local-verification.json').read_bytes())
            self.assertEqual(report['verdict'], 'NOT_VERIFIED')
            self.assertEqual(record['verdict'], 'NOT_VERIFIED')
            self.assertEqual(record['tbench_status'], 'DONE')
            self.assertIn('VERDICT=NOT_VERIFIED', text)
        else:
            self.assertEqual(record['tbench_status'], 'VERIFY_NO_REPORT')
            self.assertIn('VERDICT=None', text)
        self.assertIn('verify_evidence.py', (job / 'tbench-run.log').read_text(encoding='utf-8'))
        self.assertEqual(text.count('==== verify: '), 1)  # the console once, never doubled
        self.assertEqual(record['exit_code'], 1)
        self.assertEqual(record['kind'], KIND)
        self.assertEqual(record['check_block']['core_sha256'], vot.sha_file(self.core))
        self.assertTrue((job / 'tbench-extracted.json').is_file())
        for d in SPEC['extracted']:
            self.assertFalse((job / d).exists(), d)
        remote_job = Path(record['remote_job_dir'])
        self.assertTrue(remote_job.is_dir())  # NOT_VERIFIED: the stand-in keeps what the verifier extracted
        marker = json.loads((self.base / 'cache' / 'job3999999' / 'tbench-verified.json').read_bytes())
        self.assertEqual((marker['verdict'], marker['exit_code']), (record['verdict'], 1))
        # a second run reuses the code tree and the evidence (no upload), in a new folder
        out = io.StringIO()
        with mock.patch('sys.stdout', out), mock.patch('time.strftime', return_value='20991231T235959'):
            self.assertEqual(vot.main(self.argv), 1)
        self.assertNotIn('EVIDENCE_UPLOAD', out.getvalue())
        second = self.base / 'out' / (SPEC['folder'] + '3999999-20991231T235959')
        self.assertTrue(second.is_dir())
        # --resume-stamp: no new run; polls the record of that stamp and pulls it again (the folder moved away first)
        with self.assertRaisesRegex(vot.Refused, 'exists'):
            vot.main(self.argv + ['--resume-stamp', '20991231T235959'])
        second.rename(self.base / 'moved')
        out = io.StringIO()
        with mock.patch('sys.stdout', out):
            self.assertEqual(vot.main(self.argv + ['--resume-stamp', '20991231T235959']), 1)
        self.assertNotIn('TBENCH_RUN_STARTED', out.getvalue())
        self.assertEqual(vot.sha_file(second / 'tbench-extracted.json'),
                         vot.sha_file(self.base / 'moved' / 'tbench-extracted.json'))
        with self.assertRaisesRegex(vot.Refused, 'no run of stamp'):
            vot.main(self.argv + ['--resume-stamp', '20991231T235958'])

    def test_refused_archives(self):
        other = self.base / 'in' / 'other'
        other.write_bytes(b'another file\n')
        argv = list(self.argv)
        argv[argv.index('--core-sha256') + 1] = vot.sha_file(other)
        with self.assertRaisesRegex(vot.Refused, 'differs from the check block'):
            vot.main(argv)
        # a different file already on tbench is never replaced
        folder = self.base / 'root' / 'evidence' / 'job3999999'
        folder.mkdir(parents=True)
        (folder / SPEC['core']).write_bytes(b'already there\n')
        with self.assertRaisesRegex(vot.Refused, 'keep both'):
            vot.main(self.argv)
        self.assertEqual((folder / SPEC['core']).read_bytes(), b'already there\n')
        self.assertFalse((self.base / 'out').exists())


if __name__ == '__main__':
    unittest.main()
