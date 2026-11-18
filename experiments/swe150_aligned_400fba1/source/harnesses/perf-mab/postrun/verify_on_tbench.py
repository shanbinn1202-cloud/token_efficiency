"""This harness's postrun on tbench (local entry point; no DeltaAI connection): the alternative to
postrun\\fetch_and_verify.ps1 that never pulls the evidence archives over the slow local link (2026-11-14; the user's
rule of that day: verify on tbench, bring back only the results). Ported from perf-tb16-strip / perf-swe32's
verify_on_tbench.py (manual mode); byte-identical in perf-decode3, perf-mab, perf-thinking-gg1 and perf-gdn-probe:
the harness is found by its marker module (tbench_verify/remote_verify.py SPECS), its tbench root is
/home/ubuntu/<prefix>-verify (spd3, pmab, gg1, gdnp: one root per harness).

1. The DeltaAI side copies the finished job's <stage>/fetch archives to tbench:<root>/evidence/job<J>/ (the user runs
   the command --print-copy prints, in the DeltaAI shell: the jobs' tunnel key, the pinned tbench host key, .part names
   renamed after the copy, then sha256sum on both sides and the job's sacct row). Or --local-core / --local-bulk upload
   archives that are already local (about 300 KB/s up).
2. This script uploads the harness copy that verifies the job (--candidate-dir: default this postrun's harness; for a
   job of an older payload its frozen copy <harness>-<8 hex>) with tbench_verify/remote_verify.py as one tar under
   code/<tag>/ (tag = first 16 hex of the SHA-256 of its SHA256SUMS; top-level .py / .json / .sh / .ps1 / .in / .patch
   files and postrun\\), once per tag; every run checks the whole tree on tbench against SHA256SUMS (sha256sum -c
   --strict, the file count, nothing but files and directories) before anything runs from it.
3. It starts remote_verify.py `run` detached on tbench (nice 19; 2 cores while a selective-prefix lock is IN USE): the
   archives against the check block's SHA-256 values, then exactly the commands of the candidate's
   fetch_and_verify.ps1 (the verifier with its flags and arguments, then the tables / analysis), and polls it,
   printing its log. A dropped connection does not stop it: --resume-stamp <stamp> polls and pulls again.
4. It pulls the small outputs (the verification folder without the evidence the verifier extracted: check\\files and
   perf-mab's check\\bulk stay on tbench, their SHA-256 list comes back as tbench-extracted.json) into
   <Remote-RA>\\verification\\<folder><J>-<stamp>\\ (fetch_and_verify.ps1's folder name: spd3-decode-job, pmab-job,
   gg1-job, gdnp-job), checks every file against tbench's SHA-256 list and writes tbench-verification.json next to
   check\\ (host, code tag, candidate and its payload, the commands, the archives, the courtesy record). The marker
   <cache>\\job<J>\\tbench-verified.json records the run; no archive is written locally.
Exit codes: the verifier's, as fetch_and_verify.ps1 (0 PASS, 3 PARTIAL, 2 Slurm exit unconfirmed, 1 FAIL / NOT_VERIFIED
or refused). Usage (stdlib only; -I -B):
  D:\\Anaconda\\python.exe -I -B postrun\\verify_on_tbench.py --print-copy --stage /u/ycao11/<prefix>-<32 hex> --job-id J
  D:\\Anaconda\\python.exe -I -B postrun\\verify_on_tbench.py --stage /u/ycao11/<prefix>-<32 hex> --job-id J
      --core-sha256 <hex> [--bulk-sha256 <hex> (gg1)] [--with-bulk (perf-mab)] --slurm-state COMPLETED
      --slurm-exit-code 0:0 [--candidate-dir <frozen copy>] [--local-core <archive> [--local-bulk <archive>]]
  D:\\Anaconda\\python.exe -I -B postrun\\verify_on_tbench.py --compare <fetch_and_verify folder> <tbench folder>
      (no connection: the same verdict, check rows, tables and extracted files; exit 0 when the same)
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'tbench_verify'
sys.path.insert(0, str(DRIVER))
import remote_verify  # noqa: E402

SPECS = remote_verify.SPECS
HEX64 = re.compile(r'[0-9a-fA-F]{64}')
STAMP_RE = re.compile(r'[0-9]{8}T[0-9]{6}')
SSH = ['-o', 'BatchMode=yes', '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=30', '-o', 'ConnectTimeout=30']
CODE_SUFFIXES = ('.py', '.json', '.sh', '.ps1', '.in', '.patch')
DRIVER_FILES = ('remote_verify.py',)
# The DeltaAI side of the copy: the jobs' tunnel key (perf-tb16-strip pins.json `tbench`) and tbench's pinned host key
# (perf-tb16-strip\tbench_known_hosts, the same line as the local known_hosts on 2026-11-14)
TBENCH = dict(user='ubuntu', host='ec2-3-22-234-78.us-east-2.compute.amazonaws.com',
              identity_file='/u/ycao11/.ssh/tunnel_key', connect_timeout_seconds=20,
              host_key='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIETFfBzgld0WOns2CR4duoysPGbz3FjxYKId7FHXWlog')
POLL_SECONDS = 15


class Refused(Exception):
    pass


def sha_file(path):
    return remote_verify.sha_file(path)


def keep_temp_off_c():
    for name in ('TEMP', 'TMP', 'TMPDIR'):
        value = os.environ.get(name) or ''
        if not value or value[:2].upper() == 'C:' or value.startswith('/'):
            Path(r'D:\Remote-RA-temp\tmp\tbench-verify').mkdir(parents=True, exist_ok=True)
            os.environ[name] = r'D:\Remote-RA-temp\tmp\tbench-verify'


def kind_of(folder):
    """The harness kind of a folder: the one spec whose marker module it has."""
    found = [k for k, s in SPECS.items() if (Path(folder) / s['marker']).is_file()
             and (Path(folder) / 'postrun' / 'verify_evidence.py').is_file()]
    if len(found) != 1:
        raise Refused('%s is not exactly one known harness (markers found: %s)' % (folder, found))
    return found[0]


def remote_root(kind):
    return '/home/ubuntu/%s-verify' % SPECS[kind]['prefix']


def code_files(candidate):
    """(posix path in the code tree, local file): the candidate's top-level code files and postrun\\ under
    <candidate name>/, and this script's tbench_verify driver."""
    files = []
    for path in sorted(candidate.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        parts = path.relative_to(candidate).parts
        if (len(parts) == 1 and path.suffix in CODE_SUFFIXES) or parts[0] == 'postrun':
            files.append((candidate.name + '/' + '/'.join(parts), path))
    for name in DRIVER_FILES:
        files.append(('tbench_verify/' + name, DRIVER / name))
    names = [n for n, _ in files]
    if len(set(names)) != len(names):
        raise Refused('duplicate code path')
    return sorted(files)


def bundle(candidate, cache):
    """The code tar (deterministic) and its SHA256SUMS; returns dict(tag, harness, count, tar, sums, sums_sha256)."""
    files = code_files(candidate)
    for name, path in files:
        if path.suffix in ('.sh', '.py') and b'\r' in path.read_bytes():
            raise Refused('CR in %s (LF only)' % path)
    sums = ''.join('%s  %s\n' % (sha_file(path), name) for name, path in files).encode()
    sums_sha = hashlib.sha256(sums).hexdigest()
    tag = sums_sha[:16]
    folder = cache / 'tbench-code'
    folder.mkdir(parents=True, exist_ok=True)
    tar, sums_path = folder / (tag + '.tar.xz'), folder / (tag + '.SHA256SUMS')
    if not sums_path.is_file() or sha_file(sums_path) != sums_sha:
        sums_path.write_bytes(sums)
    if not tar.is_file():
        part = tar.with_name(tar.name + '.part')
        if part.exists():
            part.unlink()
        with tarfile.open(part, 'x:xz', format=tarfile.PAX_FORMAT) as archive:
            for name, path in files:
                data = path.read_bytes()
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.uid, info.gid, info.mode = len(data), 0, 0, 0, 0o644
                archive.addfile(info, io.BytesIO(data))
        part.rename(tar)
    return dict(tag=tag, harness=candidate.name, count=len(files), tar=tar, sums=sums_path, sums_sha256=sums_sha,
                files=dict((name, sha_file(path)) for name, path in files
                           if name.startswith(candidate.name + '/postrun/') or name.startswith('tbench_verify/')))


def candidate_payload(candidate):
    path = candidate / 'dist' / 'payload.json'
    return sha_file(path) if path.is_file() else None


# ---------------------------------------------------------------- tbench access (SSH) or a local stand-in (tests)


def ssh(host, command, check=True):
    done = subprocess.run(['ssh'] + SSH + [host, command], capture_output=True, text=True, encoding='utf-8',
                          errors='replace')
    if check and done.returncode != 0:
        raise Refused('ssh %s failed (exit %s): %s' % (command[:160], done.returncode, (done.stderr or done.stdout)[-800:]))
    return done.returncode, done.stdout


def scp(sources, target):
    done = subprocess.run(['scp', '-q'] + SSH + [str(s) for s in sources] + [str(target)])
    if done.returncode != 0:
        raise Refused('scp failed with exit %s (rerun is safe)' % done.returncode)


def json_line(text, what):
    lines = [line for line in text.splitlines() if line.startswith('{')]
    if not lines:
        raise Refused('%s printed no JSON line: %s' % (what, text[-400:]))
    return json.loads(lines[-1])


class SshRemote:
    """tbench over the local `tbench` alias; everything under remote_root(kind)."""

    def __init__(self, host, kind):
        self.host, self.kind, self.root, self.local = host, kind, remote_root(kind), None

    def code_dir(self, tag):
        return '%s/code/%s' % (self.root, tag)

    def driver(self, tag, args):
        return 'nice -n 19 python3 -I -B %s/tbench_verify/remote_verify.py %s' % (
            self.code_dir(tag), ' '.join(shlex.quote(a) for a in args))

    def preflight(self):
        ssh(self.host, 'mkdir -p %s && cd %s && mkdir -p code evidence out lock tmp home' % (self.root, self.root))
        rc, out = ssh(self.host, 'python3 -I -B -c %s' % shlex.quote(
            'import json,shutil,sys; print(json.dumps(dict(python=sys.version.split()[0], python_ok=sys.version_info '
            '>= (3, 12), free_gib=round(shutil.disk_usage("/home/ubuntu").free / 2 ** 30, 1))))'))
        return json_line(out, 'preflight')

    def ensure_code(self, info):
        """Install code/<tag> once (extract, sha256sum -c --strict, exact file count, read only); check it again."""
        tag, code = info['tag'], self.code_dir(info['tag'])
        sums = '%s/code/%s.SHA256SUMS' % (self.root, tag)
        verify = ('cd %s && printf \'%%s  %%s\\n\' %s %s | sha256sum -c --quiet && sha256sum -c --strict --quiet %s '
                  '&& test "$(find . -type f | wc -l)" -eq %d && test -z "$(find . ! -type f ! -type d)"'
                  % (code, info['sums_sha256'], sums, sums, info['count']))
        if ssh(self.host, 'test -d %s && test -f %s' % (code, sums), check=False)[0] != 0:
            print('CODE_UPLOAD tag=%s files=%d bytes=%d' % (tag, info['count'], info['tar'].stat().st_size), flush=True)
            scp([info['tar'], info['sums']], '%s:%s/tmp/' % (self.host, self.root))
            ssh(self.host, 'set -e; cd %(R)s/code; test ! -e %(T)s; rm -rf %(T)s.part; mkdir %(T)s.part; '
                'nice -n 19 tar -xJf %(R)s/tmp/%(T)s.tar.xz -C %(T)s.part --no-same-owner; '
                'mv %(R)s/tmp/%(T)s.SHA256SUMS %(R)s/code/%(T)s.SHA256SUMS; mv %(T)s.part %(T)s; '
                'chmod -R a-w %(T)s; rm -f %(R)s/tmp/%(T)s.tar.xz; echo CODE_INSTALLED %(T)s' % dict(R=self.root, T=tag))
        rc, out = ssh(self.host, verify, check=False)
        if rc != 0:
            raise Refused('the code tree on tbench differs from the local files (tag %s): %s' % (tag, out[-400:]))
        print('CODE_VERIFIED tag=%s files=%d root=%s' % (tag, info['count'], self.root), flush=True)
        return code

    def evidence(self, job_id):
        folder = '%s/evidence/job%s' % (self.root, job_id)
        rc, out = ssh(self.host, 'cd %s 2>/dev/null && for f in *.tar.xz; do test -f "$f" && sha256sum "$f"; done; true'
                      % folder, check=False)
        return {line.split()[1]: line.split()[0] for line in out.splitlines() if len(line.split()) == 2}

    def upload(self, job_id, name, path, digest):
        folder = '%s/evidence/job%s' % (self.root, job_id)
        ssh(self.host, 'mkdir -p %s' % folder)
        scp([path], '%s:%s/%s.part' % (self.host, folder, name))
        ssh(self.host, 'cd %s && test "$(sha256sum %s.part | cut -d\' \' -f1)" = %s && mv %s.part %s'
            % (folder, name, digest, name, name))

    def start(self, tag, argv):
        return json_line(ssh(self.host, self.driver(tag, ['start'] + argv), check=False)[1], 'remote_verify start')

    def status(self, tag, kind, job_id, stamp, offset):
        return json_line(ssh(self.host, self.driver(tag, ['status', '--kind', kind, '--job-id', job_id, '--stamp',
                                                          stamp, '--offset', str(offset)]))[1], 'remote_verify status')

    def fetch(self, absolute, target):
        if not str(absolute).startswith(self.root + '/out/'):
            raise Refused('the outputs archive lies outside %s/out: %s' % (self.root, absolute))
        scp(['%s:%s' % (self.host, absolute)], target)


class LocalRemote:
    """Local tests only: a directory standing in for the tbench root; the driver runs with this Python."""

    def __init__(self, root, kind):
        self.local, self.kind = Path(root), kind
        self.root = str(self.local)

    def code_dir(self, tag):
        return str(self.local / 'code' / tag)

    def preflight(self):
        for name in ('code', 'evidence', 'out', 'lock', 'tmp', 'home'):
            (self.local / name).mkdir(parents=True, exist_ok=True)
        return dict(python=sys.version.split()[0], python_ok=sys.version_info >= (3, 12),
                    free_gib=round(shutil.disk_usage(str(self.local)).free / 2 ** 30, 1))

    def ensure_code(self, info):
        code = self.local / 'code' / info['tag']
        if not code.is_dir():
            part = code.with_name(info['tag'] + '.part')
            if part.exists():
                shutil.rmtree(part)
            with tarfile.open(info['tar']) as archive:
                archive.extractall(part, filter='data')
            shutil.copyfile(info['sums'], self.local / 'code' / (info['tag'] + '.SHA256SUMS'))
            part.rename(code)
        listed = {}
        for line in (self.local / 'code' / (info['tag'] + '.SHA256SUMS')).read_text(encoding='utf-8').splitlines():
            digest, name = line.split('  ', 1)
            listed[name] = digest
        present = {p.relative_to(code).as_posix(): sha_file(p) for p in code.rglob('*') if p.is_file()
                   and '__pycache__' not in p.parts}
        if present != listed:
            raise Refused('the local code tree differs from its SHA256SUMS')
        print('CODE_VERIFIED tag=%s files=%d root=%s' % (info['tag'], info['count'], self.root), flush=True)
        return str(code)

    def evidence(self, job_id):
        folder = self.local / 'evidence' / ('job' + job_id)
        return {p.name: sha_file(p) for p in sorted(folder.glob('*.tar.xz'))} if folder.is_dir() else {}

    def upload(self, job_id, name, path, digest):
        folder = self.local / 'evidence' / ('job' + job_id)
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, folder / (name + '.part'))
        if sha_file(folder / (name + '.part')) != digest:
            raise Refused('copied %s differs' % name)
        os.replace(folder / (name + '.part'), folder / name)

    def driver(self, tag, argv):
        done = subprocess.run([sys.executable, '-I', '-B', str(self.local / 'code' / tag / 'tbench_verify' /
                                                                'remote_verify.py')] + argv,
                              capture_output=True, text=True, encoding='utf-8', errors='replace',
                              **remote_verify.NO_WINDOW)
        return done.stdout + done.stderr

    def start(self, tag, argv):
        return json_line(self.driver(tag, ['start'] + argv), 'remote_verify start')

    def status(self, tag, kind, job_id, stamp, offset):
        return json_line(self.driver(tag, ['status', '--kind', kind, '--job-id', job_id, '--stamp', stamp, '--offset',
                                           str(offset)]), 'remote_verify status')

    def fetch(self, absolute, target):
        source = Path(absolute)
        if (self.local / 'out').resolve() not in source.resolve().parents:
            raise Refused('the outputs archive lies outside the stand-in root: %s' % absolute)
        shutil.copyfile(source, Path(target) / source.name)


# ---------------------------------------------------------------- the local half


def print_copy(args, kind):
    print(copy_command(args, kind))


def copy_command(args, kind):
    """The DeltaAI-side command (the user runs it in the DeltaAI shell, bash; a subshell, so a failure never ends the
    shell): the archives to tbench (.part, renamed after the copy), their SHA-256 on both sides, the job's sacct row.
    perf-mab's bulk names are joined with spaces (2026-11-15: `ls` puts one per line, and with several bulk archives
    the newlines in $N broke the remote command of the second ssh, job 3601174); pipefail keeps `ls`'s failure when
    there is no bulk archive (the subshell stops before any copy, as the line without `tr` did)."""
    spec = SPECS[kind]
    names = [spec['core']]
    if spec['bulk'] and spec['bulk'][0] == 'required':
        names.append(spec['bulk'][1])
    listing = ' '.join(names)
    if kind == 'pmab' and args.with_bulk:
        listing += ' $(ls pmab-bulk-*.tar.xz | tr \'\\n\' \' \')'
    return ('( set -e -o pipefail; S=%s; J=%s; D=%s/evidence/job$J; H=%s@%s; K=$(mktemp); '
            'trap \'rm -f "$K"\' EXIT; printf \'%%s\\n\' \'%s %s\' > "$K"; O="-i %s -o IdentitiesOnly=yes '
            '-o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$K -o GlobalKnownHostsFile=/dev/null '
            '-o CheckHostIP=no -o UpdateHostKeys=no -o ConnectTimeout=%d"; cd "$S/fetch"; N="%s"; '
            'ssh $O $H "mkdir -p $D"; for n in $N; do scp $O "$n" "$H:$D/$n.part"; done; '
            'ssh $O $H "cd $D && for n in $N; do mv \\$n.part \\$n; done && sha256sum $N"; sha256sum $N; '
            'sacct -j $J -X -n -o JobID,State,ExitCode )'
            % (args.stage, args.job_id, remote_root(kind), TBENCH['user'], TBENCH['host'], TBENCH['host'],
               TBENCH['host_key'], TBENCH['identity_file'], TBENCH['connect_timeout_seconds'], listing))


def safe_extract(archive, target):
    """Regular files and directories only, relative names without '..' or ':'; the target must not exist."""
    target = Path(target)
    if target.exists():
        raise Refused('%s exists (move it away to pull again)' % target)
    with tarfile.open(archive, 'r:xz') as stream:
        members = stream.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or ':' in member.name or '\\' in member.name or not (
                    member.isfile() or member.isdir()):
                raise Refused('unsafe member in the pulled outputs: %r' % member.name)
        target.mkdir(parents=True)
        stream.extractall(target, filter='data')


def ensure_evidence(remote, args, kind):
    """The archives in evidence/job<J>/: the core (and gg1's bulk) with the check block's SHA-256; perf-mab's bulk
    archives with --with-bulk (checked by the verifier against the core's evidence-sha256.json). A missing one is
    uploaded from --local-core / --local-bulk; a file there with another SHA-256 is never replaced."""
    spec = SPECS[kind]
    have = remote.evidence(args.job_id)
    wanted = {spec['core']: (args.core_sha256.lower(), args.local_core)}
    if kind == 'gg1':
        wanted[spec['bulk'][1]] = (args.bulk_sha256.lower(), (args.local_bulk or [''])[0])
    if kind == 'pmab':
        for path in args.local_bulk or ():
            name = Path(path).name
            if not re.fullmatch(spec['bulk'][1], name):
                raise Refused('not a perf-mab bulk archive name: %s' % name)
            wanted[name] = (sha_file(path), path)
    for name, (digest, local) in wanted.items():
        if have.get(name) == digest:
            continue
        if have.get(name):
            raise Refused('evidence/job%s/%s on tbench has SHA-256 %s, not %s: keep both and report'
                          % (args.job_id, name, have[name], digest))
        if not local:
            raise Refused('evidence/job%s/%s is missing on tbench: copy it from DeltaAI first (--print-copy) or give '
                          '--local-core / --local-bulk' % (args.job_id, name))
        if sha_file(local) != digest:
            raise Refused('local %s differs from the check block\'s SHA-256' % local)
        print('EVIDENCE_UPLOAD %s (%d bytes)' % (name, Path(local).stat().st_size), flush=True)
        remote.upload(args.job_id, name, local, digest)
    have = remote.evidence(args.job_id)
    bulk = sorted(n for n in have if spec['bulk'] and spec['bulk'][0] == 'optional-glob' and re.fullmatch(
        spec['bulk'][1], n))
    if kind == 'pmab' and args.with_bulk and not bulk:
        raise Refused('--with-bulk: no pmab-bulk-*.tar.xz in evidence/job%s on tbench' % args.job_id)
    print('EVIDENCE_ON_TBENCH %s/evidence/job%s %s' % (remote.root, args.job_id, ' '.join(
        '%s=%s' % (n, have[n][:16]) for n in sorted(have))), flush=True)
    return have


def poll(remote, tag, kind, job_id, stamp, max_wait):
    """Prints the run's log as it grows; returns its record."""
    offset, started, quiet, lost = 0, time.time(), 0, 0
    while True:
        try:
            state = remote.status(tag, kind, job_id, stamp, offset)
            lost = 0
        except Refused as exc:  # the slow link dropped: the run goes on on tbench
            lost += 1
            if lost >= 5:
                raise Refused('%s; the run goes on on tbench: poll and pull later with --resume-stamp %s'
                              % (exc, stamp))
            time.sleep(POLL_SECONDS if remote.local is None else 1)
            continue
        if state.get('log'):
            sys.stdout.write(state['log'])
            sys.stdout.flush()
        offset = state.get('offset', offset)
        if state.get('result'):
            return state['result']
        if not state.get('alive'):
            quiet += 1
            if quiet >= 3 and not state.get('started'):
                raise Refused('no run of stamp %s on tbench (out/job%s/)' % (stamp, job_id))
            if quiet >= 3:
                raise Refused('the run of stamp %s ended without a record (see out/job%s/ on tbench)' % (stamp, job_id))
        else:
            quiet = 0
        if time.time() - started > max_wait:
            raise Refused('still running after %d s: poll again with --resume-stamp %s' % (max_wait, stamp))
        time.sleep(POLL_SECONDS if remote.local is None else 1)


def finish_local(args, kind, result, remote, info, cache, mode):
    """Pull the outputs archive, unpack it into the verification folder, check every file, record."""
    spec = SPECS[kind]
    if not result.get('archive'):
        raise Refused('tbench verification %s (exit %s): %s' % (result.get('tbench_verify'), result.get('exit_code'),
                                                                 result.get('detail')))
    pulled = cache / ('job' + args.job_id) / ('tbench-verify-' + result['stamp'])
    pulled.mkdir(parents=True, exist_ok=True)
    archive = pulled / Path(result['archive']['path']).name
    if not archive.is_file() or sha_file(archive) != result['archive']['sha256']:
        if archive.exists():
            archive.unlink()
        remote.fetch(result['archive']['path'], pulled)
    if sha_file(archive) != result['archive']['sha256']:
        raise Refused('the pulled outputs differ from tbench\'s SHA-256')
    job_dir = Path(args.out_root) / remote_verify.run_name(spec, args.job_id, result['stamp'])
    safe_extract(archive, job_dir)
    local = {p.relative_to(job_dir).as_posix(): sha_file(p) for p in sorted(job_dir.rglob('*')) if p.is_file()}
    if local != result.get('outputs'):
        bad = sorted(k for k in set(local) | set(result.get('outputs') or {}) if local.get(k) != result['outputs'].get(k))
        raise Refused('local outputs differ from tbench\'s list: %s' % bad[:10])
    candidate = Path(args.candidate_dir)
    record = dict(
        host='tbench', hostname=result.get('host'), root=result.get('root'), mode=mode, kind=kind,
        harness_name=spec['name'], candidate_dir=str(candidate), candidate_payload_sha256=candidate_payload(candidate),
        code_tag=info['tag'], code_sha256sums_sha256=info['sums_sha256'], code_files=info['count'],
        code_dir=result.get('code_dir'), harness=result.get('harness'), postrun_sha256=info['files'],
        driver_version=result.get('driver_version'), system_python=result.get('system_python'),
        python=result.get('python'), job_id=args.job_id, stage=result.get('stage'), stamp=result['stamp'],
        check_block=dict(core_sha256=args.core_sha256.lower() or None, bulk_sha256=(args.bulk_sha256 or '').lower()
                         or None, slurm_state=args.slurm_state or None, slurm_exit_code=args.slurm_exit_code or None),
        archives_on_tbench=result.get('archives'), with_bulk=result.get('with_bulk'),
        tbench_status=result.get('tbench_verify'), detail=result.get('detail'), verdict=result.get('verdict'),
        status=result.get('status'), verify_code=result.get('verify_code'), exit_code=result.get('exit_code'),
        tables=result.get('tables'), steps=result.get('steps'), courtesy=result.get('courtesy'),
        seconds=result.get('seconds'), extracted=result.get('extracted'), outputs_archive=result['archive'],
        remote_job_dir=result.get('job_dir'), remote_log=result.get('log'), synthetic=result.get('synthetic'),
        free_gib_at_start=result.get('free_gib_at_start'), record_sha256=hashlib.sha256(
            json.dumps(result, sort_keys=True).encode()).hexdigest(),
        note='verified on tbench by postrun\\verify_on_tbench.py; the evidence archives stay on tbench '
             '(evidence/job%s/), the extracted evidence is listed in tbench-extracted.json' % args.job_id)
    (job_dir / 'tbench-verification.json').write_text(json.dumps(record, sort_keys=True, indent=1) + '\n',
                                                      encoding='utf-8', newline='\n')
    marker = dict(verdict=result.get('verdict'), exit_code=result.get('exit_code'), job_dir=str(job_dir),
                  stamp=result['stamp'], code_tag=info['tag'], host='tbench', mode=mode,
                  core=(result.get('archives') or {}).get('core'))
    (cache / ('job' + args.job_id) / 'tbench-verified.json').write_text(json.dumps(marker, sort_keys=True) + '\n',
                                                                        encoding='utf-8')
    return job_dir


def check_rows(report_value):
    """(check, PASS / FAIL) per row: perf-decode3 / perf-mab rows carry `status`, gg1 / gdnp rows `ok`."""
    return [(r.get('check'), r['status'] if 'status' in r else ('PASS' if r.get('ok') else 'FAIL'))
            for r in report_value.get('checks') or ()]


def compare_folders(local, tbench):
    """A tbench verification folder against a fetch_and_verify.ps1 folder of the same job: the verdict, the check rows
    (name and result; then the full rows), every file of check\\ except the extracted evidence (byte for byte; a text
    file that differs only in CRLF / LF line endings is listed apart), and tbench-extracted.json against the local
    check\\files / check\\bulk when those are there. `same` = the same verdict, the same rows and no file but
    local-verification.json (it names the archive's path) differing beyond line endings."""
    local, tbench = Path(local), Path(tbench)
    extracted = SPECS['pmab']['extracted']

    def files(root):
        out = {}
        for path in sorted((root / 'check').rglob('*')):
            rel = path.relative_to(root).as_posix()
            if path.is_file() and not any(rel.startswith(d + '/') for d in extracted):
                out[rel] = sha_file(path)
        return out

    a = json.loads((local / 'check' / 'local-verification.json').read_bytes())
    b = json.loads((tbench / 'check' / 'local-verification.json').read_bytes())
    fa, fb = files(local), files(tbench)
    common = sorted(set(fa) & set(fb))
    differing = [k for k in common if fa[k] != fb[k]]
    # a text file a script wrote in text mode on Windows (CRLF) and on tbench (LF): the same lines
    newlines = [k for k in differing if k != 'check/local-verification.json' and (local / k).read_bytes().replace(
        b'\r\n', b'\n') == (tbench / k).read_bytes().replace(b'\r\n', b'\n')]
    differing = [k for k in differing if k not in newlines]
    value = dict(verdict_local=a.get('verdict'), verdict_tbench=b.get('verdict'),
                 same_verdict=a.get('verdict') == b.get('verdict'), same_check_rows=check_rows(a) == check_rows(b),
                 rows=len(check_rows(b)), full_rows_differing=[x.get('check') for x, y in zip(
                     a.get('checks') or (), b.get('checks') or ()) if x != y],
                 report_keys_differing=sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)),
                 files_compared=len(common), files_differing=differing, files_differing_in_line_endings_only=newlines,
                 only_local=sorted(set(fa) - set(fb)),
                 only_tbench=sorted(set(fb) - set(fa)), extracted_equal=None)
    listed = tbench / 'tbench-extracted.json'
    if listed.is_file() and (local / 'check' / 'files').is_dir():
        here = {}
        for d in extracted:
            for path in sorted((local / d).rglob('*')) if (local / d).is_dir() else ():
                if path.is_file():
                    here[path.relative_to(local).as_posix()] = sha_file(path)
        value['extracted_equal'] = here == json.loads(listed.read_bytes())['files']
        value['extracted_files'] = len(here)
    value['same'] = bool(value['same_verdict'] and value['same_check_rows'] and not value['only_local']
                         and not value['only_tbench'] and set(differing) <= {'check/local-verification.json'}
                         and value['extracted_equal'] is not False)
    return value


def report(kind, result, job_dir):
    check = job_dir / 'check'
    print('VERDICT=%s status=%s exit=%s CHECK_DIR=%s' % (result.get('verdict'), result.get('status'),
                                                         result.get('exit_code'), check))
    for row in result.get('tables') or ():
        print('TABLES %s exit=%s' % (row['script'], row['code']))
    if kind == 'spd3':
        print('DECODE_TABLES_DIR=%s' % (check / 'decode-tables'))
    if kind == 'pmab':
        print('MAB_TABLES_DIR=%s' % (check / 'mab-tables'))
    if kind == 'gg1':
        print('GG1_ANALYSIS_DIR=%s' % (check / 'analysis'))
    print('EVIDENCE_DIR=%s' % job_dir)
    print('VERIFIED_ON=tbench status=%s remote_seconds=%s pulled_bytes=%s extracted_kept_on_tbench=%s' % (
        result.get('tbench_verify'), result.get('seconds'), result['archive']['bytes'],
        not (result.get('extracted') or {}).get('removed')))


def check_args(args, kind):
    spec = SPECS[kind]
    if not remote_verify.stage_re(spec).fullmatch(args.stage or ''):
        raise Refused('invalid %s stage path (expected /u/ycao11/%s-<32 hex>)' % (spec['name'], spec['prefix']))
    if not args.job_id.isdigit():
        raise Refused('invalid Slurm job id')
    if args.with_bulk and kind != 'pmab':
        raise Refused('--with-bulk is for perf-mab only')
    if args.bulk_sha256 and kind != 'gg1':
        raise Refused('--bulk-sha256 is for perf-thinking-gg1 only')
    if args.local_bulk and kind not in ('gg1', 'pmab'):
        raise Refused('--local-bulk: %s has no bulk archive' % spec['name'])
    if kind == 'pmab' and args.local_bulk and not args.with_bulk:
        raise Refused('--local-bulk needs --with-bulk')
    if kind == 'gg1' and len(args.local_bulk or ()) > 1:
        raise Refused('gg1 has one bulk archive')
    if args.slurm_state and not re.fullmatch(r'[A-Z_+]+', args.slurm_state):
        raise Refused('invalid Slurm state')
    if args.slurm_exit_code and not re.fullmatch(r'[0-9]+:[0-9]+', args.slurm_exit_code):
        raise Refused('invalid Slurm exit code')
    if bool(args.slurm_state) != bool(args.slurm_exit_code):
        raise Refused('give both --slurm-state and --slurm-exit-code, or neither')
    if args.synthetic and not args.local_root:
        raise Refused('--synthetic is for local synthetic archives in a --local-root stand-in only')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', default='')
    parser.add_argument('--job-id', default='')
    parser.add_argument('--core-sha256', default='', help='the check block\'s sha256sum of the core archive')
    parser.add_argument('--bulk-sha256', default='', help='gg1: the check block\'s sha256sum of gg1-bulk.tar.xz')
    parser.add_argument('--with-bulk', action='store_true', help='perf-mab: also verify the pmab-bulk-*.tar.xz')
    parser.add_argument('--slurm-state', default='')
    parser.add_argument('--slurm-exit-code', default='')
    parser.add_argument('--candidate-dir', default=str(HERE.parent), help='the harness copy whose code verifies the job')
    parser.add_argument('--local-core', default='', help='upload this local core archive when tbench lacks it')
    parser.add_argument('--local-bulk', action='append', default=[], help='upload this local bulk archive (gg1 / '
                        'perf-mab --with-bulk) when tbench lacks it')
    parser.add_argument('--out-root', default=str(HERE.parents[3] / 'verification'))
    parser.add_argument('--cache-root', default='')
    parser.add_argument('--tbench-host', default='tbench')
    parser.add_argument('--print-copy', action='store_true', help='print the DeltaAI-side copy command and exit')
    parser.add_argument('--self-test', action='store_true', help='no connection: inputs, code bundle, tools')
    parser.add_argument('--deploy', action='store_true', help='only install the code tree on tbench (and check it)')
    parser.add_argument('--resume-stamp', default='', help='poll and pull the run of this stamp (a dropped connection)')
    parser.add_argument('--max-wait', type=int, default=4 * 3600)
    parser.add_argument('--local-root', default='', help='local tests only: a directory standing in for the tbench root')
    parser.add_argument('--synthetic', action='store_true', help='local tests only: a synthetic archive')
    parser.add_argument('--compare', nargs=2, metavar=('LOCAL_FOLDER', 'TBENCH_FOLDER'),
                        help='no connection: compare a tbench verification folder with a fetch_and_verify.ps1 folder '
                        'of the same job (exit 0 when the same)')
    args = parser.parse_args(argv)
    keep_temp_off_c()
    if args.compare:
        value = compare_folders(*args.compare)
        print('TBENCH_LOCAL_COMPARE %s' % json.dumps(value, sort_keys=True))
        return 0 if value['same'] else 1
    kind = kind_of(HERE.parent)
    candidate = Path(args.candidate_dir).resolve(strict=True)
    if kind_of(candidate) != kind:
        raise Refused('--candidate-dir %s is not a %s copy' % (candidate, SPECS[kind]['name']))
    args.candidate_dir = str(candidate)
    cache = Path(args.cache_root or (r'D:\Remote-RA-temp\scratch\tbench-verify\%s' % kind))
    if str(cache.resolve())[:2].upper() == 'C:' or str(Path(args.out_root).resolve())[:2].upper() == 'C:':
        raise Refused('the cache and the output root must not be on C:')
    if args.self_test and not args.stage:
        args.stage, args.job_id = '/u/ycao11/%s-%s' % (SPECS[kind]['prefix'], '0' * 32), '1'
        args.core_sha256 = '0' * 64
        args.bulk_sha256 = '0' * 64 if kind == 'gg1' else ''
    if args.deploy:
        remote = LocalRemote(args.local_root, kind) if args.local_root else SshRemote(args.tbench_host, kind)
        info = bundle(candidate, cache)
        found = remote.preflight()
        remote.ensure_code(info)
        print('TBENCH_VERIFY_DEPLOYED kind=%s root=%s code_tag=%s files=%d harness=%s python=%s free_gib=%s' % (
            kind, remote.root, info['tag'], info['count'], info['harness'], found.get('python'), found.get('free_gib')))
        return 0
    check_args(args, kind)
    if args.print_copy:
        print_copy(args, kind)
        return 0
    for value, name in ((args.core_sha256, '--core-sha256'), (args.bulk_sha256, '--bulk-sha256')):
        if value and not HEX64.fullmatch(value):
            raise Refused('invalid %s (the 64-hex value of the DeltaAI check block)' % name)
    if not args.core_sha256:
        raise Refused('give --core-sha256 (the 64-hex sha256sum of fetch/%s from the DeltaAI check block)'
                      % SPECS[kind]['core'])
    if kind == 'gg1' and not args.bulk_sha256:
        raise Refused('give --bulk-sha256 (the 64-hex sha256sum of fetch/gg1-bulk.tar.xz)')
    info = bundle(candidate, cache)
    if args.self_test:
        for tool in ('ssh', 'scp'):
            if not shutil.which(tool):
                raise Refused(tool + ' not found')
        print('TBENCH_VERIFY_SELFTEST_PASS connections=0 kind=%s root=%s code_tag=%s files=%d harness=%s payload=%s '
              'out_root=%s' % (kind, remote_root(kind), info['tag'], info['count'], info['harness'],
                               (candidate_payload(candidate) or '')[:16], args.out_root))
        return 0
    remote = LocalRemote(args.local_root, kind) if args.local_root else SshRemote(args.tbench_host, kind)
    started = time.time()
    found = remote.preflight()
    if not found.get('python_ok'):
        raise Refused('python3 on tbench is %s (3.12+ needed: tarfile data filter)' % found.get('python'))
    if found.get('free_gib', 0) < remote_verify.MIN_FREE_GIB:
        raise Refused('%.1f GiB free on tbench, below %d' % (found.get('free_gib', 0), remote_verify.MIN_FREE_GIB))
    remote.ensure_code(info)
    mode = 'manual: archives uploaded from local' if (args.local_core or args.local_bulk) else \
        'manual: archives copied DeltaAI->tbench by the user'
    if args.resume_stamp:
        if not STAMP_RE.fullmatch(args.resume_stamp):
            raise Refused('invalid --resume-stamp')
        stamp = args.resume_stamp
    else:
        ensure_evidence(remote, args, kind)
        stamp = time.strftime('%Y%m%dT%H%M%S')
        argv_run = ['--kind', kind, '--harness', info['harness'], '--job-id', args.job_id, '--stage', args.stage,
                    '--core-sha256', args.core_sha256.lower(), '--stamp', stamp, '--code-tag', info['tag']]
        if args.bulk_sha256:
            argv_run += ['--bulk-sha256', args.bulk_sha256.lower()]
        if args.with_bulk:
            argv_run += ['--with-bulk']
        if args.slurm_state:
            argv_run += ['--slurm-state', args.slurm_state, '--slurm-exit-code', args.slurm_exit_code]
        if args.synthetic:
            argv_run += ['--synthetic']
        begun = remote.start(info['tag'], argv_run)
        if begun.get('status') != 'STARTED':
            raise Refused('remote_verify start: %s' % begun)
        print('TBENCH_RUN_STARTED stamp=%s pid=%s log=%s (a dropped connection: run this command again with '
              '--resume-stamp %s)' % (stamp, begun.get('pid'), begun.get('log'), stamp), flush=True)
    result = poll(remote, info['tag'], kind, args.job_id, stamp, args.max_wait)
    if result.get('code_tag') != info['tag'] or result.get('job_id') != args.job_id or result.get('kind') != kind:
        raise Refused('the record names code %s job %s kind %s' % (result.get('code_tag'), result.get('job_id'),
                                                                     result.get('kind')))
    print('TBENCH_RESULT status=%s verdict=%s verify_code=%s remote_seconds=%s' % (
        result.get('tbench_verify'), result.get('verdict'), result.get('verify_code'), result.get('seconds')),
        flush=True)
    job_dir = finish_local(args, kind, result, remote, info, cache, mode)
    report(kind, result, job_dir)
    print('TOTAL_SECONDS=%.1f' % (time.time() - started))
    return int(result.get('exit_code', 1))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Refused as exc:
        print('TBENCH_VERIFY_REFUSED ' + str(exc))
        sys.exit(1)
