"""A harness's postrun on tbench (runs on tbench under the system python3 -I -B; stdlib only; never delivered).

The tbench half of postrun/verify_on_tbench.py (2026-11-14; the user's rule of that day: every harness verifies on
tbench and only the results come back over the slow local link). One file for the four harnesses that verify one
evidence archive with postrun/fetch_and_verify.ps1 (perf-decode3 = spd3, perf-mab = pmab, perf-thinking-gg1 = gg1,
perf-gdn-probe = gdnp; byte-identical in each of their postrun/tbench_verify/): SPECS below holds what differs, and
`steps` runs exactly the commands of that harness's fetch_and_verify.ps1 after its download (the verifier, then its
tables / analysis script when the verifier wrote local-verification.json; the same Python flags, arguments and order).

Layout (root = this file's code tree's parent: <root>/code/<tag>/tbench_verify/remote_verify.py; on tbench
/home/ubuntu/<kind>-verify, one root per harness): code/<tag>/ (the harness copy that verifies, sha256sum-checked by
the local side before every run), evidence/job<J>/ (the archives: copied there from DeltaAI by the user's command or
uploaded by the local side), out/job<J>/ (per run, <name> = <folder><J>-<stamp>: <name>/ = the verification folder,
<name>.log = the steps' commands and output (pulled as <name>/tbench-run.log), <name>.console = the detached run's
output, <name>.json = the record, <name>.tar.xz = the small outputs the local side pulls), tmp/<stamp>/, home/,
lock/verify.lock.

`run`: 1. one verification per root at a time (lock/verify.lock; BUSY otherwise) and at least MIN_FREE_GIB free on the
disk (the experiment jobs refuse below 50 GB); 2. the archives in evidence/job<J>/ against the SHA-256 values of the
DeltaAI check block (a different file is refused: keep both and report); 3. the verifier and the tables (`steps`) into
out/job<J>/<folder><J>-<stamp>/check, as fetch_and_verify.ps1 writes <verification>/<folder><J>-<stamp>/check; 4. the
outputs: the folder without the evidence the verifier extracted (EXTRACTED: check/files, perf-mab's check/bulk; their
files' SHA-256 go to <folder>/tbench-extracted.json), packed as <name>.tar.xz with every output file's SHA-256 in the
record; the extracted evidence is removed from tbench after a run that verified (any verdict but NOT_VERIFIED; the
archives in evidence/ stay); 5. the record <name>.json (also the last line printed). The exit code is the verifier's,
as fetch_and_verify.ps1's (the tables never change it).
`start`: `run` detached (new session, nice 19, output to out/job<J>/<name>.console); prints one JSON line and returns.
`status`: the record when written, whether the run is alive, and the console from a byte offset (one JSON line).
`preflight`: python version, free disk, the experiment locks (one JSON line).
Courtesy (shared host, README-tbench-sharing.md): nice 19 always; while a selective-prefix experiment lock is IN USE,
2 cores (sched_setaffinity, inherited by every step) and 2 threads per library. Writes only under its root.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parents[1]
DRIVER_VERSION = '2026-11-14.1'
EXPERIMENT_ROOTS = ('swe10', 'swe16', 'swe32', 'gswe32', 'tb16')
THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
MIN_FREE_GIB = 60
BUSY = 75
HEX64 = re.compile(r'[0-9a-f]{64}')
STAMP_RE = re.compile(r'[0-9]{8}T[0-9]{6}')
STATE_RE = re.compile(r'[A-Z_+]{1,40}')
EXIT_RE = re.compile(r'[0-9]{1,3}:[0-9]{1,3}')
# local Windows tests only: no console window per child; nothing changes on tbench
NO_WINDOW = dict(creationflags=0x08000000) if os.name == 'nt' else {}

# What differs between the harnesses (fetch_and_verify.ps1 of each, read 2026-11-14; test_tbench_verify.py checks
# these against the harness's own script):
#   marker      a module only that harness has (verify_on_tbench.py finds its kind by it)
#   prefix      the stage prefix (/u/ycao11/<prefix>-<32 hex>) and the root's name (/home/ubuntu/<prefix>-verify)
#   folder      the verification folder's name before <J>-<stamp> (the ps1's $Out)
#   core        the core archive (fetch/<core> on the stage)
#   bulk        None; ('optional-glob', regex) perf-mab: pmab-bulk-<label>.tar.xz, used with --with-bulk (the ps1's
#               -WithBulk); ('required', name) gg1: always passed (the ps1 always fetches it)
#   flags       the Python flags of the verifier call
#   synthetic   the verifier's flag for local synthetic archives (local tests only)
#   tables      (flags, script, failure word) per table / analysis step, run when local-verification.json exists
#   extracted   directories of the check folder that hold the evidence the verifier extracted (not pulled back)
SPECS = {
    'spd3': dict(name='perf-decode3', marker='decode_checks.py', prefix='spd3', folder='spd3-decode-job',
                 core='decode-evidence.tar.xz', bulk=None, flags=('-B',), synthetic='--synthetic-pins',
                 tables=((('-B',), 'decode_tables.py', 'DECODE_TABLES_FAILED'),), extracted=('check/files',)),
    'pmab': dict(name='perf-mab', marker='mab_checks.py', prefix='pmab', folder='pmab-job', core='pmab-core.tar.xz',
                 bulk=('optional-glob', r'pmab-bulk-[A-Za-z0-9_.-]{1,64}\.tar\.xz'), flags=('-B',),
                 synthetic='--synthetic-pins', tables=((('-B',), 'mab_tables.py', 'MAB_TABLES_FAILED'),),
                 extracted=('check/files', 'check/bulk')),
    'gg1': dict(name='perf-thinking-gg1', marker='g1checks.py', prefix='gg1', folder='gg1-job',
                core='gg1-evidence.tar.xz', bulk=('required', 'gg1-bulk.tar.xz'), flags=('-I', '-B'),
                synthetic='--synthetic', tables=((('-I', '-B'), 'analyze.py', 'GG1_ANALYSIS_NOT_WRITTEN'),),
                extracted=('check/files',)),
    'gdnp': dict(name='perf-gdn-probe', marker='probe.py', prefix='gdnp', folder='gdnp-job',
                 core='gdnp-evidence.tar.xz', bulk=None, flags=('-I', '-B'), synthetic='--synthetic', tables=(),
                 extracted=('check/files',)),
}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run_name(spec, job_id, stamp):
    return '%s%s-%s' % (spec['folder'], job_id, stamp)


def stage_re(spec):
    return re.compile(r'/u/ycao11/%s-[0-9a-f]{32}' % re.escape(spec['prefix']))


def bulk_names(spec, folder):
    """The bulk archives present in an evidence folder (perf-mab: every pmab-bulk-<label>.tar.xz; gg1: its one)."""
    if not spec['bulk']:
        return []
    kind, value = spec['bulk']
    if kind == 'required':
        return [value] if (Path(folder) / value).is_file() else []
    return sorted(p.name for p in Path(folder).glob('*.tar.xz') if re.fullmatch(value, p.name))


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
    """README-tbench-sharing.md's check: a lock directory whose owner process is alive is IN USE."""
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


def free_gib():
    return round(shutil.disk_usage(str(ROOT if ROOT.exists() else ROOT.parent)).free / 2 ** 30, 1)


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
        self.history.append(dict(step=step, locks=locks, limited=self.limited, cpus=list(self.cpus),
                                 nice=os.nice(0) if hasattr(os, 'nice') else None))
        return busy

    def env(self, tmp):
        value = dict(PATH='/usr/local/bin:/usr/bin:/bin', HOME=str(ROOT / 'home'), LANG='C.UTF-8', LC_ALL='C.UTF-8',
                     TMPDIR=str(tmp), TEMP=str(tmp), TMP=str(tmp), PYTHONDONTWRITEBYTECODE='1')
        if os.name == 'nt':  # local Windows tests only: the interpreter needs its own PATH and SYSTEMROOT
            value.update({k: os.environ[k] for k in ('PATH', 'SYSTEMROOT') if k in os.environ})
            value['PYTHONIOENCODING'] = 'utf-8'
        if self.limited:
            value.update({k: '2' for k in THREAD_VARS})
        return value


def run_step(name, command, log, env, cwd, steps):
    """Runs one step, its output streamed to stdout and appended to the log; returns (exit code, lines)."""
    started = time.time()
    with log.open('a', encoding='utf-8') as stream:
        header = '==== %s: %s\n' % (name, ' '.join(map(str, command)))
        stream.write(header)
        stream.flush()
        sys.stdout.write(header)
        sys.stdout.flush()
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
    steps.append(dict(step=name, code=code, seconds=round(time.time() - started, 1),
                      command=[str(c) for c in command]))
    return code, lines


def verifier_command(kind, python, harness, a, evidence, check):
    """The verifier call of the harness's fetch_and_verify.ps1 ($Arguments), argument for argument and in its order;
    the archives are read where they lie in evidence/job<J>/ (the ps1 copies them into its output folder first)."""
    spec = SPECS[kind]
    verifier = harness / 'postrun' / 'verify_evidence.py'
    command = [python] + list(spec['flags']) + [verifier, '--archive', evidence / spec['core']]
    if kind == 'gg1':
        command += ['--bulk', evidence / spec['bulk'][1]]
    command += ['--job-id', a.job_id, '--stage', a.stage, '--out', check]
    if a.core_sha256:
        command += ['--remote-archive-sha256', a.core_sha256]
    if kind == 'gg1' and a.bulk_sha256:
        command += ['--remote-bulk-sha256', a.bulk_sha256]
    if a.slurm_state:
        command += ['--slurm-state', a.slurm_state, '--slurm-exit-code', a.slurm_exit_code]
    if a.synthetic:
        command += [spec['synthetic']]
    if kind == 'pmab' and a.with_bulk:
        command += ['--bulk-dir', evidence]
    return command


def manifest(folder, skip=()):
    out = {}
    for path in sorted(Path(folder).rglob('*')):
        rel = path.relative_to(folder).as_posix()
        if path.is_file() and not any(rel == d or rel.startswith(d + '/') for d in skip):
            out[rel] = sha_file(path)
    return out


def pack_outputs(job_dir, target, skip):
    """tar.xz of the verification folder (names relative to it), the extracted evidence left out; sorted members."""
    with tarfile.open(target, 'x:xz', format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(job_dir.rglob('*')):
            rel = path.relative_to(job_dir).as_posix()
            if any(rel == d or rel.startswith(d + '/') for d in skip):
                continue
            info = archive.gettarinfo(str(path), arcname=rel)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            if path.is_file():
                with path.open('rb') as stream:
                    archive.addfile(info, stream)
            elif path.is_dir():
                archive.addfile(info)


def check_run_args(a):
    if a.kind not in SPECS:
        return 'unknown kind'
    spec = SPECS[a.kind]
    if not a.job_id.isdigit() or len(a.job_id) > 12:
        return 'invalid job id'
    if not stage_re(spec).fullmatch(a.stage):
        return 'invalid stage (expected /u/ycao11/%s-<32 hex>)' % spec['prefix']
    if not STAMP_RE.fullmatch(a.stamp):
        return 'invalid stamp'
    if not HEX64.fullmatch(a.core_sha256 or ''):
        return 'invalid core sha256'
    if a.bulk_sha256 and not HEX64.fullmatch(a.bulk_sha256):
        return 'invalid bulk sha256'
    if a.kind == 'gg1' and not a.bulk_sha256:
        return 'gg1 needs --bulk-sha256'
    if a.kind != 'gg1' and a.bulk_sha256:
        return '--bulk-sha256 is for gg1 only'
    if a.with_bulk and a.kind != 'pmab':
        return '--with-bulk is for perf-mab only'
    if bool(a.slurm_state) != bool(a.slurm_exit_code):
        return 'give both --slurm-state and --slurm-exit-code, or neither'
    if a.slurm_state and not (STATE_RE.fullmatch(a.slurm_state) and EXIT_RE.fullmatch(a.slurm_exit_code)):
        return 'invalid Slurm state / exit code'
    if not re.fullmatch(r'[0-9a-f]{16}', a.code_tag) or a.code_tag != CODE.name:
        return 'code tag %s is not this code tree (%s)' % (a.code_tag, CODE.name)
    if not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', a.harness):
        return 'invalid harness name'
    return None


def run(a):
    started = time.time()
    bad = check_run_args(a)
    if bad:
        print(json.dumps(dict(tbench_verify='REFUSED_ARGUMENTS', detail=bad)))
        return 2
    spec = SPECS[a.kind]
    name = run_name(spec, a.job_id, a.stamp)
    base = ROOT / 'out' / ('job' + a.job_id)
    job_dir, result_path, log = base / name, base / (name + '.json'), base / (name + '.log')
    tmp = ROOT / 'tmp' / a.stamp
    for path in (job_dir, result_path, tmp):
        if path.exists():
            print(json.dumps(dict(tbench_verify='EXISTS', detail='exists (a stamp is never reused): %s' % path)))
            return 2
    base.mkdir(parents=True, exist_ok=True)
    (ROOT / 'home').mkdir(exist_ok=True)
    steps = []
    result = dict(tbench_verify='RUNNING', kind=a.kind, harness_name=spec['name'], job_id=a.job_id, stage=a.stage,
                  stamp=a.stamp, name=name, job_dir=str(job_dir), log=str(log), console=str(base / (name + '.console')),
                  code_tag=a.code_tag, code_dir=str(CODE),
                  root=str(ROOT), host=platform.node(), system_python=sys.version.split()[0],
                  python=sys.executable, driver_version=DRIVER_VERSION, steps=steps, synthetic=bool(a.synthetic),
                  with_bulk=bool(a.with_bulk), slurm=dict(state=a.slurm_state or None,
                                                          exit_code=a.slurm_exit_code or None))
    lock = None

    def finish(status, code, **fields):
        result.update(fields, tbench_verify=status, exit_code=code, seconds=round(time.time() - started, 1))
        if job_dir.is_dir():
            if log.is_file():  # the steps' commands and output (fetch_and_verify.ps1's console) come back too
                shutil.copyfile(log, job_dir / 'tbench-run.log')
            extracted = {k: v for k, v in manifest(job_dir).items()
                         if any(k.startswith(d + '/') for d in spec['extracted'])}
            record = dict(note='the evidence the verifier extracted on tbench (not pulled back): every file\'s SHA-256',
                          dirs=list(spec['extracted']), files=extracted)
            (job_dir / 'tbench-extracted.json').write_text(json.dumps(record, sort_keys=True, indent=1) + '\n',
                                                           encoding='utf-8', newline='\n')
            result['extracted'] = dict(files=len(extracted), bytes=sum(
                (job_dir / k).stat().st_size for k in extracted))
            result['outputs'] = manifest(job_dir, skip=spec['extracted'])
            archive = base / (name + '.tar.xz')
            pack_outputs(job_dir, archive, spec['extracted'])
            result['archive'] = dict(path=str(archive), sha256=sha_file(archive), bytes=archive.stat().st_size)
            if result.get('verdict') and result['verdict'] != 'NOT_VERIFIED':
                for d in spec['extracted']:
                    shutil.rmtree(job_dir / d, ignore_errors=True)
                result['extracted']['removed'] = not any((job_dir / d).exists() for d in spec['extracted'])
            else:
                result['extracted']['removed'] = False
        with result_path.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, indent=1)
            stream.write('\n')
        shutil.rmtree(tmp, ignore_errors=True)
        print(json.dumps(result, sort_keys=True), flush=True)
        if lock is not None:
            lock.close()
        return code

    lock = lock_file(ROOT / 'lock' / 'verify.lock')
    if lock is None:
        return finish('BUSY', BUSY, detail='another verification is running in this root: run the same command again '
                      'later (a new stamp)')
    free = free_gib()
    result['free_gib_at_start'] = free
    if free < MIN_FREE_GIB:
        return finish('DISK_LOW', 1, detail='%.1f GiB free on tbench, below %d: nothing ran' % (free, MIN_FREE_GIB))
    harness = (CODE / a.harness).resolve()
    if CODE not in harness.parents or not (harness / spec['marker']).is_file() or not (
            harness / 'postrun' / 'verify_evidence.py').is_file():
        return finish('NO_HARNESS', 1, detail='%s is not a %s copy in the code tree' % (a.harness, spec['name']))
    result['harness'] = str(harness)
    evidence = ROOT / 'evidence' / ('job' + a.job_id)
    # 2. the archives against the check block's SHA-256 values
    core = evidence / spec['core']
    got = dict(core=sha_file(core) if core.is_file() else None)
    names = bulk_names(spec, evidence)
    if a.kind == 'gg1':
        bulk = evidence / spec['bulk'][1]
        got['bulk'] = sha_file(bulk) if bulk.is_file() else None
    elif a.with_bulk:
        got['bulk_archives'] = {n: sha_file(evidence / n) for n in names}
    result['archives'] = got
    if got['core'] != a.core_sha256 or (a.kind == 'gg1' and got['bulk'] != a.bulk_sha256):
        return finish('ARCHIVE_MISMATCH', 1, detail='the archives in %s differ from the check block (or are missing): '
                      'keep both and report' % evidence)
    if a.with_bulk and not names:
        return finish('NO_BULK', 1, detail='--with-bulk but no %s archive in %s' % (spec['bulk'][1], evidence))
    tmp.mkdir(parents=True)
    courtesy = Courtesy()
    result['courtesy'] = courtesy.history
    # 3. the verifier and the tables (fetch_and_verify.ps1's commands)
    check = job_dir / 'check'
    job_dir.mkdir(parents=True)
    courtesy.check('verify')
    verify_code, _ = run_step('verify', verifier_command(a.kind, sys.executable, harness, a, evidence, check), log,
                              courtesy.env(tmp), tmp, steps)
    report_path = check / 'local-verification.json'
    if not report_path.is_file():
        return finish('VERIFY_NO_REPORT', verify_code or 1)
    report = json.loads(report_path.read_bytes())
    result.update(verify_code=verify_code, verdict=report.get('verdict'), status=report.get('status'),
                  failed=report.get('failed'))
    tables = []
    for flags, script, word in spec['tables']:
        courtesy.check(script)
        code, _ = run_step(script, [sys.executable] + list(flags) + [harness / 'postrun' / script, check], log,
                           courtesy.env(tmp), tmp, steps)
        tables.append(dict(script=script, code=code))
        if code != 0:
            print('%s exit=%s' % (word, code), flush=True)
    result['tables'] = tables
    return finish('DONE', verify_code)


# ---------------------------------------------------------------- detached runs (the local side polls)


def run_paths(kind, job_id, stamp):
    """The record, the console (the detached run's stdout: what the local side prints) and the pid file."""
    base = ROOT / 'out' / ('job' + job_id)
    name = run_name(SPECS[kind], job_id, stamp)
    return base / (name + '.json'), base / (name + '.console'), base / (name + '.pid')


def start(a):
    """`run` detached: its own session, nice 19, stdout to the console file; one JSON line."""
    bad = check_run_args(a)
    if bad:
        print(json.dumps(dict(event='start', status='REFUSED_ARGUMENTS', detail=bad)))
        return 2
    result_path, log_path, pid_path = run_paths(a.kind, a.job_id, a.stamp)
    if result_path.exists() or log_path.exists() or log_path.with_suffix('.log').exists():
        print(json.dumps(dict(event='start', status='EXISTS', detail=str(log_path))))
        return 2
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), 'run'] + run_argv(a)
    log = log_path.open('ab')
    kwargs = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True, cwd=str(ROOT))
    if os.name == 'nt':  # local Windows tests only: its own hidden console, which its children share
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs.update(start_new_session=True, preexec_fn=be_nice)
    process = subprocess.Popen(command, **kwargs)
    log.close()
    pid_path.write_text('%d\n' % process.pid, encoding='utf-8')
    print(json.dumps(dict(event='start', status='STARTED', pid=process.pid, log=str(log_path),
                          result=str(result_path)), sort_keys=True))
    return 0


def alive(pid):
    if os.name == 'nt':  # local Windows tests only
        done = subprocess.run(['tasklist', '/FI', 'PID eq %d' % pid, '/NH'], capture_output=True, text=True,
                              **NO_WINDOW)
        return str(pid) in done.stdout
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def status(a):
    result_path, log_path, pid_path = run_paths(a.kind, a.job_id, a.stamp)
    text = ''
    if log_path.is_file():
        with log_path.open('rb') as stream:
            stream.seek(a.offset)
            data = stream.read(1 << 20)
        text = data.decode('utf-8', errors='replace')
        offset = a.offset + len(data)
    else:
        offset = a.offset
    result = json.loads(result_path.read_bytes()) if result_path.is_file() else None
    pid = int(pid_path.read_text().strip()) if pid_path.is_file() else None
    print(json.dumps(dict(event='status', result=result, alive=bool(pid and alive(pid)), pid=pid, log=text,
                          offset=offset, started=log_path.is_file()), sort_keys=True))
    return 0


def preflight(a):
    print(json.dumps(dict(event='preflight', python=sys.version.split()[0], python_ok=sys.version_info >= (3, 12),
                          free_gib=free_gib(), min_free_gib=MIN_FREE_GIB, locks=experiment_locks(), root=str(ROOT),
                          host=platform.node(), driver_version=DRIVER_VERSION), sort_keys=True))
    return 0


def run_argv(a):
    argv = ['--kind', a.kind, '--harness', a.harness, '--job-id', a.job_id, '--stage', a.stage,
            '--core-sha256', a.core_sha256, '--stamp', a.stamp, '--code-tag', a.code_tag]
    if a.bulk_sha256:
        argv += ['--bulk-sha256', a.bulk_sha256]
    if a.with_bulk:
        argv += ['--with-bulk']
    if a.slurm_state:
        argv += ['--slurm-state', a.slurm_state, '--slurm-exit-code', a.slurm_exit_code]
    if a.synthetic:
        argv += ['--synthetic']
    return argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'start'):
        one = sub.add_parser(name)
        one.add_argument('--kind', required=True, choices=sorted(SPECS))
        one.add_argument('--harness', required=True, help='the harness copy, relative to the code tree')
        one.add_argument('--job-id', required=True)
        one.add_argument('--stage', required=True)
        one.add_argument('--core-sha256', required=True)
        one.add_argument('--bulk-sha256', default='')
        one.add_argument('--with-bulk', action='store_true')
        one.add_argument('--slurm-state', default='')
        one.add_argument('--slurm-exit-code', default='')
        one.add_argument('--stamp', required=True)
        one.add_argument('--code-tag', required=True)
        one.add_argument('--synthetic', action='store_true', help='local synthetic archives only (tests)')
    state = sub.add_parser('status')
    state.add_argument('--kind', required=True, choices=sorted(SPECS))
    state.add_argument('--job-id', required=True)
    state.add_argument('--stamp', required=True)
    state.add_argument('--offset', type=int, default=0)
    sub.add_parser('preflight')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):  # one encoding for the log and the local side (Windows tests: not cp936)
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if args.command == 'status':
        if not args.job_id.isdigit() or not STAMP_RE.fullmatch(args.stamp):
            raise SystemExit('invalid job id / stamp')
        return status(args)
    if args.command == 'preflight':
        return preflight(args)
    if args.command == 'start':
        return start(args)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
