"""perf-tb16-strip postrun on tbench (runs on tbench under the system python3 -I -B; stdlib only).

perf-swe32's tbench verification (2026-11-11), ported 2026-11-12. The tbench half of postrun/verify_on_tbench.py: the
same steps as postrun/fetch_run.ps1 after its downloads, on the archives in /home/ubuntu/tb16-verify/evidence/<run_id>/
and with the harness copy the local side uploaded (code/<tag>/, SHA-256-checked before anything runs from it). The root
is this file's code tree's parent (<root>/code/<tag>/tbench_verify/remote_verify.py), /home/ubuntu/tb16-verify on
tbench.

`run` (the local entry point calls it over SSH; the waiter calls it in-process):
1. the two archives against the check block's (or the job's) SHA-256 values;
2. verify_evidence.py (the tokenizer venv; --gemma-meta the pinned Gemma 4 files of the code tree) into
   out/<run_id>/tb16-job<J>-<stamp>/check, the evidence extracted to work/<run_id>/<stamp>/evidence-<stamp>;
3. the tbench run package of <tbench bin>/package_run.sh (as fetch_run.ps1 calls it; it packages a finished run once,
   HARBOR_RUNNING while a Harbor run of it is still going), its SHA-256, unpack.py into
   work/<run_id>/<stamp>/tbench-<state>-<stamp>;
4. join_tb16.py and tb16_tables.py next to the check directory (fetch_run.ps1's arguments), from --join-harness when
   given (2026-11-15, the manual mode's --join-candidate-dir: a second harness copy of the code tree; default the
   verifying harness);
5. the small outputs: out/<run_id>/tb16-job<J>-<stamp>.tar.xz (the job folder without check/tokenizer and
   check/tokenizer-gemma, which the local side restores from its pinned copies) and tb16-job<J>-<stamp>.json (every
   output file's SHA-256, the step codes and seconds, host, the venv freeze, courtesy settings), whose content is also
   the last line printed.
`trigger` (the GPU job calls it over SSH at its end, after pushing its archives, through code/by-payload/<payload
members id>): checks this code tree against its SHA256SUMS and its harness copy's sealed payload against the job's
delivered members, writes evidence/<run_id>/request.json (the job's archive SHA-256 values, its end record: status and
the Slurm state it implies, "from the job; confirm with the END mail") and starts `wait` detached (new session, nice 19,
output to out/<run_id>/waiter.log); prints one JSON line and returns at once.
`wait`: one waiter per run (lock/wait-<run_id>.lock), at most `wait_seconds` from the trigger: polls until both archives
are in evidence/<run_id>/ with the request's SHA-256 (a file there with another SHA-256 is refused at once) and the run's
Harbor runs ended on tbench (package_run.sh's rule: no pool.pid / harbor.pid of the run alive without its .done, the
shared tbench locks not held by that run), then runs `run` with the job's Slurm state (retried while another
verification holds lock/verify.lock) and writes out/<run_id>/RESULT.json (verdict, join verdict, key counts, every
output file's SHA-256, the archive to pull; any failure a clean record with its reason); work/<run_id>/<stamp> is removed
after a verified run (evidence/ and out/ are kept).
Courtesy (shared host, README-tbench-sharing.md): always nice 19; when a selective-prefix experiment lock is IN USE,
2 cores (sched_setaffinity, inherited by every step) and 2 threads per library. One verification at a time (flock).
Writes only under its root.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parents[1]
EXPERIMENT_ROOTS = ('swe10', 'swe16', 'swe32', 'gswe32', 'tb16')
TBENCH_BIN = '/home/ubuntu/tb16/bin'
TBENCH_RUNS = '/home/ubuntu/tb16/runs'
GEMMA_META = 'gemma4-meta'  # the code tree's copy of the pinned Gemma 4 tokenizer files (verify_on_tbench.code_files)
THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
RUN_RE = re.compile(r'tb16-[0-9]+-r[0-9]{1,2}')
HEX64 = re.compile(r'[0-9a-f]{64}')
STAGE_RE = re.compile(r'/u/ycao11/tb16-[0-9a-f]{32}')
PATH_RE = re.compile(r'(/|[A-Za-z]:/)[A-Za-z0-9_./-]{1,200}')
STATE_RE = re.compile(r'[A-Z_+]{1,40}')
EXIT_RE = re.compile(r'[0-9]{1,3}:[0-9]{1,3}')
JOB_STATUSES = ('TB16_PASS', 'TB16_PARTIAL', 'TB16_REFUSED', 'TB16_FAIL')
ARCHIVES = (('core', 'tb16-core.tar.xz'), ('bulk', 'tb16-bulk.tar.xz'))
TOKENIZER_DIRS = ('check/tokenizer', 'check/tokenizer-gemma')  # restored locally from the pinned copies
SLURM_NOTE = 'from the job; confirm with the END mail'
BUSY = 75
# package_run.sh's rule without the packaging (lib.sh: the run directory, the shared locks' status)
HARBOR_RULE = ('. "$1/lib.sh"; RUN=$2; alive=0; for f in "$RUNS/$RUN"/*/pool.pid "$RUNS/$RUN"/*/harbor.pid; do '
               '[ -f "$f" ] || continue; d=$(dirname "$f"); [ -f "$d/.done" ] && continue; '
               'kill -0 "$(cat "$f")" 2>/dev/null && alive=1; done; echo "HARBOR_ALIVE $alive"; locks status')
# local Windows tests only: no console window per child; nothing changes on tbench
NO_WINDOW = dict(creationflags=0x08000000) if os.name == 'nt' else {}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def members_id(files):
    """The payload members id: SHA-256 of '<sha256>  <name>\\n' per member, by name ({name: sha256}); the same value
    the job computes from its delivery.json and verify_on_tbench.py --deploy from dist/payload.json."""
    return hashlib.sha256(''.join('%s  %s\n' % (files[n], n) for n in sorted(files)).encode()).hexdigest()


def lock_file(path):
    """An exclusive non-blocking lock held while the returned handle is open (None: someone else holds it)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handle = Path(path).open('a')
    try:
        import fcntl
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
    except ImportError:  # local Windows tests only
        import msvcrt
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            handle.close()
            return None
    return handle


def experiment_locks():
    """The README check: a lock directory whose owner process is alive is IN USE."""
    found = {}
    for name in EXPERIMENT_ROOTS:
        lock = Path('/home/ubuntu') / name / 'lock' / 'tbench.lock'
        if not lock.is_dir():
            continue
        try:
            pid = int(json.loads((lock / 'owner.json').read_text()).get('pid'))
            os.kill(pid, 0)
            found[name] = 'IN USE'
        except PermissionError:
            found[name] = 'IN USE'
        except Exception:
            found[name] = 'stale'
    return found


def be_nice():
    if hasattr(os, 'nice'):
        os.nice(19 - os.nice(0))


class Courtesy:
    def __init__(self):
        be_nice()
        self.limited = False
        self.cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
        self.history = []

    def check(self, step):
        locks = experiment_locks()
        busy = sorted(k for k, v in locks.items() if v == 'IN USE')
        if busy and not self.limited and hasattr(os, 'sched_setaffinity'):
            self.cpus = self.cpus[-2:]
            os.sched_setaffinity(0, set(self.cpus))
            self.limited = True
        self.history.append(dict(step=step, locks=locks, limited=self.limited, cpus=list(self.cpus)))
        return busy

    def env(self, tmp):
        value = dict(PATH='/usr/local/bin:/usr/bin:/bin', HOME=str(ROOT / 'home'), LANG='C.UTF-8', LC_ALL='C.UTF-8',
                     TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp), HF_HOME=str(tmp / 'hf'),
                     XDG_CACHE_HOME=str(tmp / 'cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                     HF_HUB_DISABLE_TELEMETRY='1')
        if os.name == 'nt':  # local Windows tests only: the interpreter needs its own PATH and SYSTEMROOT
            value.update({k: os.environ[k] for k in ('PATH', 'SYSTEMROOT') if k in os.environ})
        if self.limited:
            value.update({k: '2' for k in THREAD_VARS})
        return value


def run_step(name, command, log, env, cwd, steps):
    """Runs one step, its output streamed to stdout and appended to the log; returns the exit code."""
    started = time.time()
    with log.open('a', encoding='utf-8') as stream:
        stream.write('==== %s: %s\n' % (name, ' '.join(map(str, command))))
        stream.flush()
        process = subprocess.Popen([str(c) for c in command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   env=env, cwd=str(cwd), text=True, encoding='utf-8', errors='replace', **NO_WINDOW)
        lines = []
        for line in process.stdout:
            stream.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line.rstrip('\n'))
        code = process.wait()
        stream.write('==== %s exit %s after %.1f s\n' % (name, code, time.time() - started))
    steps.append(dict(step=name, code=code, seconds=round(time.time() - started, 1)))
    return code, lines


def outputs_manifest(job_dir):
    return {p.relative_to(job_dir).as_posix(): sha_file(p) for p in sorted(job_dir.rglob('*')) if p.is_file()}


def pack_outputs(job_dir, target):
    """tar.xz of the job folder (names relative to it), the tokenizer copies left out; deterministic member order."""
    with tarfile.open(target, 'x:xz') as archive:
        for path in sorted(job_dir.rglob('*')):
            rel = path.relative_to(job_dir).as_posix()
            if any(rel == d or rel.startswith(d + '/') for d in TOKENIZER_DIRS):
                continue
            info = archive.gettarinfo(str(path), arcname=rel)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            if path.is_file():
                with path.open('rb') as stream:
                    archive.addfile(info, stream)
            elif path.is_dir():
                archive.addfile(info)


def venv_python(venv):
    path = Path(venv) / 'bin' / 'python'
    return path if path.exists() or os.name != 'nt' else Path(venv) / 'Scripts' / 'python.exe'


def venv_record(venv):
    ready = json.loads((Path(venv) / 'READY.json').read_text())
    return {k: ready.get(k) for k in ('name', 'venv', 'requirements_sha256', 'freeze_sha256', 'python')}


def venv_paths():
    """The pinned venv of this code tree's requirements file (verify_on_tbench.ensure_venvs' name)."""
    return {'tok': ROOT / 'venv' / ('tok-%s' % sha_file(CODE / 'tbench_verify' / 'requirements-tok.txt')[:12])}


def exit_code(verify_code, join_code, harbor_running):
    """fetch_run.ps1's exit code."""
    if harbor_running:
        return 3
    if verify_code not in (0, 2, 3, 4):
        return verify_code
    if join_code != 0:
        return 1
    return verify_code


def bash():
    return os.environ.get('TB16_VERIFY_BASH') or 'bash'


def run(args):
    started = time.time()
    if not RUN_RE.fullmatch(args.run_id) or not args.job_id.isdigit() or ('-%s-' % args.job_id) not in args.run_id:
        raise SystemExit('invalid run id / job id')
    if not re.fullmatch(r'[0-9]{8}T[0-9]{6}', args.stamp):
        raise SystemExit('invalid stamp')
    for value in (args.core_sha256, args.bulk_sha256):
        if not HEX64.fullmatch(value):
            raise SystemExit('invalid sha256')
    harness = (CODE / args.harness).resolve(strict=True)
    if CODE not in harness.parents:
        raise SystemExit('harness outside the code tree')
    join_harness = harness
    if getattr(args, 'join_harness', None):
        join_harness = (CODE / args.join_harness).resolve(strict=True)
        if CODE not in join_harness.parents or not (join_harness / 'postrun' / 'join_tb16.py').is_file():
            raise SystemExit('join harness outside the code tree or without postrun/join_tb16.py')
    synthetic = bool(getattr(args, 'synthetic_pins', None))
    if synthetic:  # local tests only (fetch_run.ps1 -Synthetic): the fakes' char tokenizer
        tok = Path(sys.executable)
    else:
        tok = venv_python(args.tok_venv)
        if Path(args.tok_venv).parent != ROOT / 'venv' or not (Path(args.tok_venv) / 'READY.json').is_file():
            raise SystemExit('venv not ready: %s' % args.tok_venv)
    package_run = str(PurePosixPath(getattr(args, 'tbench_bin', None) or TBENCH_BIN) / 'package_run.sh')
    lock = lock_file(ROOT / 'lock' / 'verify.lock')
    if lock is None:
        print(json.dumps(dict(tb16_tbench_verify='BUSY', detail='another verification is running on tbench')))
        return BUSY
    courtesy = Courtesy()
    evidence_in = ROOT / 'evidence' / args.run_id
    core, bulk = evidence_in / 'tb16-core.tar.xz', evidence_in / 'tb16-bulk.tar.xz'
    base = ROOT / 'out' / args.run_id
    job_name = 'tb16-job%s-%s' % (args.job_id, args.stamp)
    job_dir, result_path = base / job_name, base / (job_name + '.json')
    work, tmp = ROOT / 'work' / args.run_id / args.stamp, ROOT / 'tmp' / args.stamp
    for path in (job_dir, work, tmp, result_path):
        if path.exists():
            raise SystemExit('exists (never reused): %s' % path)
    base.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True)
    tmp.mkdir(parents=True)
    (ROOT / 'home').mkdir(exist_ok=True)
    log = base / (job_name + '.log')
    steps = []
    result = dict(tb16_tbench_verify='RUNNING', run_id=args.run_id, job_id=args.job_id, stage=args.stage,
                  stamp=args.stamp, job_dir=str(job_dir), log=str(log), code_tag=args.code_tag, code_dir=str(CODE),
                  harness=str(harness), join_harness=str(join_harness), host=platform.node(),
                  system_python=sys.version.split()[0],
                  venvs=dict(synthetic=True) if synthetic else dict(tok=venv_record(args.tok_venv)),
                  work_dir=str(work), steps=steps)
    if getattr(args, 'slurm_source', None):
        result['slurm_source'] = args.slurm_source
    if synthetic:
        result['synthetic'] = True

    def finish(status, code):
        result.update(tb16_tbench_verify=status, exit_code=code, courtesy=courtesy.history,
                      seconds=round(time.time() - started, 1))
        if job_dir.is_dir():
            result['outputs'] = outputs_manifest(job_dir)
            archive = base / (job_name + '.tar.xz')
            pack_outputs(job_dir, archive)
            result['archive'] = dict(path=str(archive), sha256=sha_file(archive), bytes=archive.stat().st_size)
        with result_path.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, indent=1)
            stream.write('\n')
        shutil.rmtree(tmp, ignore_errors=True)
        print(json.dumps(result, sort_keys=True))
        lock.close()
        return code

    # 1. the archives
    got = dict(core=sha_file(core) if core.is_file() else None, bulk=sha_file(bulk) if bulk.is_file() else None)
    result['archives'] = dict(got, core_bytes=core.stat().st_size if core.is_file() else None,
                              bulk_bytes=bulk.stat().st_size if bulk.is_file() else None)
    if got['core'] != args.core_sha256 or got['bulk'] != args.bulk_sha256:
        result['detail'] = 'archives on tbench differ from the check block (or are missing): keep both and report'
        return finish('ARCHIVE_MISMATCH', 1)
    # 2. verify_evidence.py (fetch_run.ps1's arguments, in its order; the Gemma tokenizer files of the code tree)
    courtesy.check('verify')
    check = job_dir / 'check'
    command = [tok, '-B', harness / 'postrun' / 'verify_evidence.py', '--core', core, '--job-id', args.job_id,
               '--stage', args.stage, '--out', check, '--evidence', work / ('evidence-' + args.stamp), '--bulk', bulk,
               '--core-sha256', args.core_sha256, '--bulk-sha256', args.bulk_sha256]
    if args.slurm_state:
        command += ['--slurm-state', args.slurm_state, '--slurm-exit-code', args.slurm_exit_code]
    if synthetic:
        command += ['--fake-tokens', '--synthetic-pins', args.synthetic_pins]
    else:
        command += ['--gemma-meta', CODE / GEMMA_META]
    verify_code, _ = run_step('verify', command, log, courtesy.env(tmp), tmp, steps)
    report_path = check / 'local-verification.json'
    if not report_path.is_file():
        return finish('VERIFY_NO_REPORT', verify_code or 1)
    report = json.loads(report_path.read_bytes())
    result.update(verify_code=verify_code, verdict=report.get('verdict'), report_run_id=report.get('run_id'))
    if report.get('run_id') != args.run_id:
        result['detail'] = 'the archives are run %s, not %s' % (report.get('run_id'), args.run_id)
        return finish('RUN_ID_MISMATCH', 1)
    # 3. the tbench run package (package_run.sh, as fetch_run.ps1 calls it over SSH)
    courtesy.check('package')
    code, lines = run_step('package', [bash(), package_run, args.run_id], log, dict(os.environ), tmp, steps)
    last = [line for line in lines if line.startswith('{')]
    if not last:
        return finish('PACKAGE_NO_JSON', 1)
    info = json.loads(last[-1])
    result['package'] = info
    if info.get('status') == 'HARBOR_RUNNING':
        result.update(harbor_state='running', detail='the DeltaAI half is verified; a Harbor run of this job is still '
                      'going on tbench: run verify_on_tbench.py again later (a new stamp)')
        return finish('HARBOR_RUNNING', exit_code(verify_code, 0, True))
    if info.get('status') == 'NO_RUN':
        result['detail'] = 'tbench holds no run directory %s (no Harbor run started: the DeltaAI half is verified)' % (
            args.run_id,)
        return finish('NO_TBENCH_RUN', verify_code or 1)
    if info.get('status') != 'PACKAGED' or sha_file(info['path']) != info.get('sha256'):
        return finish('PACKAGE_FAILED', 1)
    state = 'done'
    result['harbor_state'] = state
    unpacked = work / ('tbench-%s-%s' % (state, args.stamp))
    code, _ = run_step('unpack', [tok, '-I', '-B', harness / 'postrun' / 'unpack.py', info['path'], unpacked], log,
                       courtesy.env(tmp), tmp, steps)
    runs_dir = unpacked / args.run_id
    if code != 0 or not runs_dir.is_dir():
        return finish('UNPACK_FAILED', 1)
    # 4. join and tables (fetch_run.ps1's arguments)
    courtesy.check('join')
    join_out = job_dir / ('join-%s-%s' % (state, args.stamp))
    join_command = [tok, '-B', join_harness / 'postrun' / 'join_tb16.py', check, runs_dir, '--out', join_out]
    if synthetic:
        join_command += ['--synthetic-pins', args.synthetic_pins]
    join_code, _ = run_step('join', join_command, log, courtesy.env(tmp), tmp, steps)
    tables_out = job_dir / ('tables-%s-%s' % (state, args.stamp))
    tables_code, _ = run_step('tables', [tok, '-B', join_harness / 'postrun' / 'tb16_tables.py', '--out', tables_out,
                                         join_out], log, courtesy.env(tmp), tmp, steps)
    result.update(join_code=join_code, tables_code=tables_code, join_dir=join_out.name, tables_dir=tables_out.name,
                  runs_dir=str(runs_dir))
    if tables_code != 0:
        return finish('TABLES_FAILED', 1)
    return finish('DONE', exit_code(verify_code, join_code, False))


# ---------------------------------------------------------------- the job's trigger and the waiter


def event(status, code, **fields):
    print(json.dumps(dict(event='tb16_verify_trigger', status=status, **fields), sort_keys=True), flush=True)
    return code


def tree_check():
    """This code tree against code/<tag>.SHA256SUMS: the list hashes to the tag, every listed file is there with its
    SHA-256 and nothing else is (verify_on_tbench.ensure_code's check, in Python). Returns (ok, detail)."""
    tag = CODE.name
    sums = CODE.parent / (tag + '.SHA256SUMS')
    if not re.fullmatch(r'[0-9a-f]{16}', tag) or not sums.is_file() or sha_file(sums)[:16] != tag:
        return False, 'code/%s.SHA256SUMS missing or not this tag' % tag
    listed = {}
    for line in sums.read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        listed[name] = digest
    present = sorted(p.relative_to(CODE).as_posix() for p in CODE.rglob('*') if not p.is_dir())
    if present != sorted(listed):
        return False, 'files differ from SHA256SUMS: %s' % sorted(set(present) ^ set(listed))[:5]
    bad = [n for n in present if sha_file(CODE / n) != listed[n]]
    return (not bad), (bad[:5] if bad else 'files=%d' % len(present))


def find_harness():
    found = [p.parent.parent for p in CODE.rglob('verify_evidence.py')
             if p.parent.name == 'postrun' and (p.parent.parent / 'dist' / 'payload.json').is_file()]
    if len(found) != 1:
        raise SystemExit('expected one harness copy in the code tree, found %d' % len(found))
    return found[0]


def check_args(a):
    if not RUN_RE.fullmatch(a.run_id) or not a.job_id.isdigit() or ('-%s-' % a.job_id) not in a.run_id:
        return 'invalid run id / job id'
    if not STAGE_RE.fullmatch(a.stage):
        return 'invalid stage'
    if not all(HEX64.fullmatch(v) for v in (a.core_sha256, a.bulk_sha256, a.payload_files)):
        return 'invalid sha256'
    if not (STATE_RE.fullmatch(a.slurm_state) and EXIT_RE.fullmatch(a.slurm_exit_code) and a.job_status in JOB_STATUSES):
        return 'invalid end record'
    if not (PATH_RE.fullmatch(a.tbench_bin) and PATH_RE.fullmatch(a.tbench_runs)) or '..' in a.tbench_bin + a.tbench_runs:
        return 'invalid tbench paths'
    if not (1 <= a.poll_seconds <= 3600 and 60 <= a.wait_seconds <= 172800 or a.synthetic_pins and a.wait_seconds >= 1):
        return 'invalid poll / wait seconds'
    return None


def trigger(a):
    bad = check_args(a)
    if bad:
        return event('REFUSED_ARGUMENTS', 2, detail=bad)
    ok, detail = tree_check()
    if not ok:
        return event('CODE_TREE_MISMATCH', 6, detail=detail, code_tag=CODE.name)
    harness = find_harness()
    payload = json.loads((harness / 'dist' / 'payload.json').read_bytes())
    have = members_id({k: v['sha256'] for k, v in payload['files'].items()})
    if have != a.payload_files:
        return event('PAYLOAD_MISMATCH', 6, detail='the code tree verifies payload members %s, the job delivered %s'
                     % (have, a.payload_files), code_tag=CODE.name)
    request = dict(run_id=a.run_id, job_id=a.job_id, stage=a.stage, archives=dict(core=a.core_sha256, bulk=a.bulk_sha256),
                   archive_bytes=dict(core=a.core_bytes, bulk=a.bulk_bytes), payload_files=a.payload_files,
                   end=dict(job_status=a.job_status, slurm_state=a.slurm_state, slurm_exit_code=a.slurm_exit_code,
                            note=SLURM_NOTE),
                   pushed=a.pushed, code_tag=CODE.name, harness=harness.relative_to(CODE).as_posix(),
                   tbench_bin=a.tbench_bin, tbench_runs=a.tbench_runs, poll_seconds=a.poll_seconds,
                   wait_seconds=a.wait_seconds)
    if a.synthetic_pins:
        request['synthetic_pins'] = a.synthetic_pins
    folder, out = ROOT / 'evidence' / a.run_id, ROOT / 'out' / a.run_id
    folder.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    path = folder / 'request.json'
    if path.is_file():
        previous = json.loads(path.read_bytes())
        if {k: v for k, v in previous.items() if k not in ('triggered_wall', 'pushed')} != {
                k: v for k, v in request.items() if k != 'pushed'}:
            return event('REQUEST_EXISTS_DIFFERENT', 3, request=str(path))
        request = previous
    else:
        request['triggered_wall'] = round(time.time(), 3)
        part = folder / ('.request.json.%d' % os.getpid())
        part.write_text(json.dumps(request, sort_keys=True, indent=1) + '\n', encoding='utf-8')
        os.replace(part, path)
    if (out / 'RESULT.json').is_file():
        return event('ALREADY_DONE', 0, result=str(out / 'RESULT.json'))
    log = (out / 'waiter.log').open('ab')
    kwargs = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True, cwd=str(ROOT))
    if os.name == 'nt':  # local Windows tests only: its own hidden console, which its children share
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs.update(start_new_session=True, preexec_fn=be_nice)
    process = subprocess.Popen([sys.executable, '-I', '-B', str(Path(__file__).resolve()), 'wait', '--run-id', a.run_id],
                               **kwargs)
    log.close()
    return event('WAITER_STARTED', 0, pid=process.pid, request=str(path), request_sha256=sha_file(path),
                 code_tag=CODE.name, log=str(out / 'waiter.log'))


def evidence_state(run_id, want, cache):
    out = {}
    for key, name in ARCHIVES:
        path = ROOT / 'evidence' / run_id / name
        if not path.is_file():
            out[key] = 'missing'
            continue
        stat = path.stat()
        sig = (stat.st_size, stat.st_mtime_ns)
        if cache.get(key, (None,))[0] != sig:
            cache[key] = (sig, sha_file(path))
        out[key] = 'ok' if cache[key][1] == want[key] else 'mismatch:' + cache[key][1]
    return out


def harbor_state(request):
    """package_run.sh's rule (HARBOR_RULE): a Harbor run of the run is going while a pool.pid / harbor.pid of it is
    alive without its .done marker or the shared tbench locks are held by that run. Returns (running, detail); an
    unreadable state counts as running (the waiter waits; its timeout bounds it)."""
    detail = None
    try:
        done = subprocess.run([bash(), '-c', HARBOR_RULE, 'harbor-state', request['tbench_bin'], request['run_id']],
                              capture_output=True, text=True, timeout=120, **NO_WINDOW)
        alive = [line.split()[1] for line in done.stdout.splitlines() if line.startswith('HARBOR_ALIVE ')]
        lines = [line for line in done.stdout.splitlines() if line.startswith('{')]
        locks = json.loads(lines[-1]) if lines else None
        detail = dict(alive=alive[-1] if alive else None, locks=locks)
        if not alive or locks is None:
            detail['error'] = (done.stderr or done.stdout)[-300:]
            return True, detail
        held = bool(locks.get('held') and locks.get('run_id') == request['run_id'])
        return alive[-1] == '1' or held, detail
    except Exception as exc:
        return True, dict(detail or {}, error='%s: %s' % (type(exc).__name__, exc))


def read_json(path):
    try:
        return json.loads(Path(path).read_bytes())
    except Exception:
        return None


def wait(a):
    if not RUN_RE.fullmatch(a.run_id):
        raise SystemExit('invalid run id')
    request_path = ROOT / 'evidence' / a.run_id / 'request.json'
    request = json.loads(request_path.read_bytes())
    out = ROOT / 'out' / a.run_id
    result_path = out / 'RESULT.json'
    holder = lock_file(ROOT / 'lock' / ('wait-%s.lock' % a.run_id))
    if holder is None:
        print('WAITER_EXISTS %s' % a.run_id, flush=True)
        return 0
    if result_path.is_file():
        print('RESULT_EXISTS %s' % result_path, flush=True)
        return 0
    be_nice()
    started = time.time()
    deadline = request['triggered_wall'] + request['wait_seconds']
    record = dict(tb16_tbench_auto='WAITING', run_id=a.run_id, job_id=request['job_id'], stage=request['stage'],
                  request=str(request_path), request_sha256=sha_file(request_path), archives=request['archives'],
                  slurm_from_job=request['end'], pushed=request.get('pushed'), code_tag=request['code_tag'],
                  harness=request['harness'], host=platform.node(), pid=os.getpid(), waiter_started_wall=round(started, 3),
                  polls=0, harbor_running_at_start=None, busy_retries=0, synthetic=bool(request.get('synthetic_pins')))
    cache, last_harbor = {}, None

    def write(status, **fields):
        record.pop('_seen', None)
        record.update(fields, tb16_tbench_auto=status, waited_seconds=round(time.time() - started, 1),
                      finished_wall=round(time.time(), 3))
        part = out / ('.RESULT.json.%d' % os.getpid())
        part.write_text(json.dumps(record, sort_keys=True, indent=1) + '\n', encoding='utf-8')
        os.replace(part, result_path)
        print('RESULT %s %s' % (status, result_path), flush=True)
        return 0

    try:
        while True:
            have = evidence_state(a.run_id, request['archives'], cache)
            record['evidence'] = have
            if any(v.startswith('mismatch') for v in have.values()):
                return write('ARCHIVE_MISMATCH', detail='an archive in evidence/%s differs from the job\'s SHA-256 '
                             '(archives.json): refused; keep both and report' % a.run_id)
            running, detail = harbor_state(request)
            last_harbor = dict(running=running, detail=detail)
            record['polls'] += 1
            seen = (sorted(have.items()), running, (detail or {}).get('error'))
            if seen != record.get('_seen'):  # one line per change of the evidence or the Harbor state
                record['_seen'] = seen
                print('POLL %s t=%.0fs evidence=%s harbor_running=%s detail=%s' % (
                    time.strftime('%H:%M:%S'), time.time() - started, have, running,
                    json.dumps(detail, sort_keys=True)[:200]), flush=True)
            if record['harbor_running_at_start'] is None:
                record['harbor_running_at_start'] = running
                record['evidence_at_start'] = dict(have)
            if all(v == 'ok' for v in have.values()) and not running:
                stamp = time.strftime('%Y%m%dT%H%M%S')
                venvs = venv_paths()
                if not request.get('synthetic_pins') and not all((v / 'READY.json').is_file() for v in venvs.values()):
                    return write('VENV_MISSING', detail='the pinned venv %s is not built (verify_on_tbench.py '
                                 '--deploy --setup-venv)' % [str(v) for v in venvs.values()], harbor=last_harbor)
                args = argparse.Namespace(
                    harness=request['harness'], run_id=a.run_id, job_id=request['job_id'], stage=request['stage'],
                    core_sha256=request['archives']['core'], bulk_sha256=request['archives']['bulk'],
                    slurm_state=request['end']['slurm_state'], slurm_exit_code=request['end']['slurm_exit_code'],
                    stamp=stamp, tok_venv=str(venvs['tok']), code_tag=request['code_tag'],
                    tbench_bin=request['tbench_bin'], synthetic_pins=request.get('synthetic_pins'),
                    slurm_source=SLURM_NOTE)
                print('RUN stamp=%s' % stamp, flush=True)
                code = run(args)
                if code == BUSY:
                    record['busy_retries'] += 1
                else:
                    return finish_result(write, record, out, a.run_id, request, stamp, code, last_harbor)
            if time.time() + request['poll_seconds'] > deadline:
                return write('WAIT_TIMEOUT', detail='evidence %s, Harbor %s after %d s' % (
                    have, last_harbor, request['wait_seconds']), harbor=last_harbor)
            time.sleep(request['poll_seconds'])
    except BaseException as exc:
        write('ERROR', detail='%s: %s' % (type(exc).__name__, exc), traceback=traceback.format_exc()[-4000:],
              harbor=last_harbor)
        raise


def finish_result(write, record, out, run_id, request, stamp, code, last_harbor):
    result_path = out / ('tb16-job%s-%s.json' % (request['job_id'], stamp))
    result = read_json(result_path) or {}
    status = result.get('tb16_tbench_verify') or 'NO_RESULT'
    job_dir = Path(result.get('job_dir') or out / 'missing')
    report = read_json(job_dir / 'check' / 'local-verification.json') or {}
    join = read_json(job_dir / result['join_dir'] / 'join.json') if result.get('join_dir') else None
    counts = dict(checks=len(report.get('checks') or ()), failed=report.get('failed'),
                  servers=[{k: s.get(k) for k in ('label', 'policy', 'instances', 'resolved', 'accuracy', 'timeouts',
                                                  'status', 'note')}
                           for s in (join or {}).get('servers') or ()],
                  join_instances=len((join or {}).get('instances') or ()),
                  join_failed=len((join or {}).get('failed') or ()),
                  killed_in_flight_rows=(join or {}).get('killed_in_flight_rows'))
    verified = status == 'DONE' and str(result.get('verdict')).endswith('_VERIFIED') and result.get('join_code') == 0
    cleaned = False
    if verified and result.get('work_dir'):
        shutil.rmtree(result['work_dir'], ignore_errors=True)
        try:
            Path(result['work_dir']).parent.rmdir()
        except OSError:
            pass
        cleaned = not Path(result['work_dir']).exists()
    return write('DONE' if status == 'DONE' else status, verify_status=status, exit_code=code, stamp=stamp,
                 verdict=result.get('verdict'), verify_code=result.get('verify_code'),
                 join_verdict=(join or {}).get('verdict'), join_code=result.get('join_code'),
                 tables_code=result.get('tables_code'), harbor_state=result.get('harbor_state'), counts=counts,
                 outputs=result.get('outputs'), archive=result.get('archive'),
                 result_json=dict(path=str(result_path), sha256=sha_file(result_path)) if result_path.is_file() else None,
                 job_dir=result.get('job_dir'), join_dir=result.get('join_dir'), tables_dir=result.get('tables_dir'),
                 detail=result.get('detail'), work_dir=result.get('work_dir'), work_cleaned=cleaned,
                 seconds_verify=result.get('seconds'), harbor=last_harbor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('run')
    one.add_argument('--harness', required=True, help='the harness copy, relative to the code tree')
    one.add_argument('--join-harness', default='',
                     help='the harness copy (relative to the code tree) whose join_tb16 / tb16_tables run; default --harness')
    one.add_argument('--run-id', required=True)
    one.add_argument('--job-id', required=True)
    one.add_argument('--stage', required=True)
    one.add_argument('--core-sha256', required=True)
    one.add_argument('--bulk-sha256', required=True)
    one.add_argument('--slurm-state', default='')
    one.add_argument('--slurm-exit-code', default='')
    one.add_argument('--stamp', required=True)
    one.add_argument('--tok-venv', required=True)
    one.add_argument('--code-tag', required=True)
    one.add_argument('--tbench-bin', default=TBENCH_BIN)
    trig = sub.add_parser('trigger')
    for name in ('--run-id', '--job-id', '--stage', '--core-sha256', '--bulk-sha256', '--payload-files', '--job-status',
                 '--slurm-state', '--slurm-exit-code'):
        trig.add_argument(name, required=True)
    trig.add_argument('--core-bytes', type=int, default=None)
    trig.add_argument('--bulk-bytes', type=int, default=None)
    trig.add_argument('--pushed', default='')
    trig.add_argument('--tbench-bin', default=TBENCH_BIN)
    trig.add_argument('--tbench-runs', default=TBENCH_RUNS)
    trig.add_argument('--poll-seconds', type=int, default=60)
    trig.add_argument('--wait-seconds', type=int, default=28800)
    trig.add_argument('--synthetic-pins', default='', help='local tests only (the fake tbench adds it)')
    waiter = sub.add_parser('wait')
    waiter.add_argument('--run-id', required=True)
    sub.add_parser('locks')
    args = parser.parse_args(argv)
    if args.command == 'locks':
        print(json.dumps(experiment_locks(), sort_keys=True))
        return 0
    if args.command == 'trigger':
        return trigger(args)
    if args.command == 'wait':
        return wait(args)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
