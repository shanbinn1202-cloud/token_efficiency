"""perf-tb16-strip postrun on tbench (local entry point; no DeltaAI connection): perf-swe32's tbench verification of
2026-11-11, ported 2026-11-12. The alternative to postrun\\fetch_run.ps1 that does not pull the evidence archives over
the local link.

Since 2026-11-12 the job pushes its archives itself and starts the verification on tbench (remote_verify.py trigger /
wait; config.tbench_verify). Two modes for that:
  --deploy [--setup-venv]: before a job of this payload is submitted: the code tree (below, 1.) and the venv (2.),
      then the pointer code/by-payload/<payload members id> (line 1 the tag, line 2 the harness path, line 3 the payload
      SHA-256) that the job's trigger follows (the members id: SHA-256 of '<sha256>  <name>\\n' per dist/payload.json
      member, by name; the job computes it from its delivery.json);
  --pull --job-id <J> --slurm-state S --slurm-exit-code E [--core-sha256 H --bulk-sha256 H]: after the job's END mail
      (its Slurm state and exit code are REQUIRED: the pull confirms the job's own end record with them; a different
      state is refused: verify again with the manual command below and the END mail's values): reads
      out/<run_id>/RESULT.json (absent: the waiter's state, exit 3), pulls the outputs archive it names and writes the
      verification folder exactly as 4. (check, join, tables; every file's SHA-256 against tbench's list). A pull of an
      already pulled result only records the confirmation. --core-sha256 / --bulk-sha256 (the DeltaAI check block),
      when given, must equal the job's pushed values.

The manual path (a job whose push failed: fetch/tbench-push.json says why). The DeltaAI side copies a finished job's
<stage>/fetch/tb16-core.tar.xz and tb16-bulk.tar.xz to tbench:/home/ubuntu/tb16-verify/evidence/<run_id>/ (the command
is printed by --print-copy); this script then
1. uploads the harness copy that verifies the job (default: this postrun's harness; --candidate-dir another frozen copy;
   --join-candidate-dir a second harness copy under the same root whose join_tb16.py / tb16_tables.py join instead of
   the candidate's: 2026-11-15, a job verified by its own frozen copy's sealed verifier and joined with a later join,
   e.g. job 3560498 with the killed in-flight rule) together with the three pinned Qwen tokenizer files, the five pinned Gemma 4 tokenizer files (pins.json
   gemma_model.pins) and postrun\\tbench_verify, as one tar under code/<tag>/ (tag = first 16 hex of the SHA-256 of its
   SHA256SUMS), once per tag; every run checks the whole tree on tbench against SHA256SUMS (sha256sum -c --strict, no
   extra file) before anything runs from it;
2. checks the pinned venv (venv/tok-<12 hex> of the requirements file; --setup-venv builds it once with
   tbench_verify/setup_venvs.sh);
3. runs tbench_verify/remote_verify.py on tbench: the archives against the check block's SHA-256 values,
   verify_evidence.py, package_run.sh + unpack.py, join_tb16.py, tb16_tables.py, exactly as fetch_run.ps1 runs them
   (courtesy: nice 19; 2 cores while a selective-prefix lock is IN USE);
4. pulls back the small outputs (a tar.xz of the job folder without check\\tokenizer and check\\tokenizer-gemma) into
   <Remote-RA>\\verification\\tb16-job<J>-<stamp>\\ (check, join-done-<stamp>, tables-done-<stamp>: the folder
   fetch_run.ps1 writes), restores the tokenizer copies from the local pinned files, checks every file against the
   SHA-256 list tbench computed, and adds `verified_on` (host, code tag and file hashes, the venv freeze) to
   check\\local-verification.json. The marker <cache>\\job<J>\\tbench-verified.json records the run; no fetched.json is
   written.
Exit codes as fetch_run.ps1 (0 PASS, 3 partial / Harbor still running, 4 refused, 2 Slurm exit unconfirmed (manual
mode without the END mail), 1 otherwise). Usage (tokenizer venv, -I -B):
  python -I -B postrun\\verify_on_tbench.py --pull --job-id <J> --slurm-state COMPLETED --slurm-exit-code 0:0
  python -I -B postrun\\verify_on_tbench.py --stage /u/ycao11/tb16-<32 hex> --job-id <J> --core-sha256 <hex>
      --bulk-sha256 <hex> --slurm-state COMPLETED --slurm-exit-code 0:0 [--run-id tb16-<J>-r1]
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

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'tbench_verify'
sys.path.insert(0, str(HERE.parent))
import localtemp  # noqa: E402

localtemp.hide_windows()  # (ssh / scp / unpack without a console window of their own on Windows)
DRIVER_FILES = ('remote_verify.py', 'setup_venvs.sh', 'requirements-tok.txt')
REMOTE = '/home/ubuntu/tb16-verify'
SHARE = 'share/vllm-0.24.0-selective-prefix-policies/source-overlay/remote_ra_linux/model-evidence'
# verify_evidence.TOKENIZER_FILES: check/tokenizer/<name> = model-evidence/<source>
TOKENIZER = {'config.json': ('model-config.json', '69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c'),
             'tokenizer.json': ('tokenizer.json', '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'),
             'tokenizer_config.json': ('tokenizer-config.json',
                                       '5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b')}
# verify_evidence.GEMMA_META / GEMMA_TOKENIZER_FILES: check/tokenizer-gemma/<name>, each pinned in pins.json
# gemma_model.pins; the code tree carries them under gemma4-meta/ (remote_verify passes --gemma-meta)
GEMMA_META = os.environ.get('REMOTE_RA_GEMMA4_METADATA') or r'D:\Remote-RA-temp\scratch\gemma4-meta'
GEMMA_FILES = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json', 'generation_config.json')
GEMMA_TREE = 'gemma4-meta'
HARNESS_DIRS = ('dist', 'postrun', 'tasks', 'tbench')
STAGE_RE = re.compile(r'/u/ycao11/tb16-[0-9a-f]{32}')
HEX64 = re.compile(r'[0-9a-fA-F]{64}')
SSH = ['-o', 'BatchMode=yes', '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=30']
CORE, BULK = 'tb16-core.tar.xz', 'tb16-bulk.tar.xz'


class Refused(Exception):
    pass


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def keep_temp_off_c():
    for name in ('TEMP', 'TMP', 'TMPDIR'):
        value = os.environ.get(name) or ''
        if not value or value[:2].upper() == 'C:' or value.startswith('/'):
            Path(r'D:\Remote-RA-temp\tmp\perf-tb16-strip').mkdir(parents=True, exist_ok=True)
            os.environ[name] = r'D:\Remote-RA-temp\tmp\perf-tb16-strip'


def members_id(files):
    """The payload members id (remote_verify.members_id; the job's tbench_link.members_id): SHA-256 of
    '<sha256>  <name>\\n' per member, by name."""
    return hashlib.sha256(''.join('%s  %s\n' % (files[n], n) for n in sorted(files)).encode()).hexdigest()


def payload_members(candidate):
    payload = json.loads((Path(candidate) / 'dist' / 'payload.json').read_bytes())
    return {k: v['sha256'] for k, v in payload['files'].items()}


def gemma_pins(candidate):
    """{name: sha256} of the pinned Gemma 4 tokenizer files (the candidate's pins.json gemma_model.pins)."""
    pins = json.loads((Path(candidate) / 'pins.json').read_bytes())
    value = ((pins.get('gemma_model') or {}).get('pins') or {})
    return {name: value[name] for name in GEMMA_FILES if name in value}


def harness_files(candidate, root):
    """(the harness copy's path under the Remote-RA root, [(posix path in the code tree, local file)])."""
    rel = PurePosixPath(*candidate.relative_to(root).parts)
    files = []
    for path in sorted(candidate.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        parts = path.relative_to(candidate).parts
        if (len(parts) == 1 and path.suffix in ('.py', '.json', '.bundle')) or parts[0] in HARNESS_DIRS:
            files.append((str(rel.joinpath(*parts)), path))
    return str(rel), files


def code_files(candidate, join_candidate=None):
    """(posix path in the code tree, local file): the harness copy at its path under the Remote-RA root (the verifier
    reads the tokenizer files at CANDIDATE.parents[2]/share/...), the pinned Qwen tokenizer files, the pinned Gemma 4
    tokenizer files (gemma4-meta/), the tbench driver. `join_candidate` (manual mode, 2026-11-15): a second harness copy
    under the same root whose join_tb16 / tb16_tables run instead of the candidate's (a job verified by its own sealed
    verifier, joined with a later join)."""
    root = candidate.parents[2]
    rel, files = harness_files(candidate, root)
    if join_candidate is not None and Path(join_candidate) != candidate:
        if Path(join_candidate).parents[2] != root:
            raise Refused('the join candidate %s does not lie under the Remote-RA root %s' % (join_candidate, root))
        files += harness_files(Path(join_candidate), root)[1]
    for name, (source, pin) in TOKENIZER.items():
        path = root.joinpath(*SHARE.split('/'), source)
        if sha_file(path) != pin:
            raise Refused('pinned tokenizer evidence changed: %s' % path)
        files.append((SHARE + '/' + source, path))
    for name, pin in gemma_pins(candidate).items():
        path = Path(GEMMA_META) / name
        if not path.is_file() or sha_file(path) != pin:
            raise Refused('pinned Gemma 4 metadata changed or missing: %s' % path)
        files.append((GEMMA_TREE + '/' + name, path))
    for name in DRIVER_FILES:
        files.append(('tbench_verify/' + name, DRIVER / name))
    names = [n for n, _ in files]
    if len(set(names)) != len(names):
        raise Refused('duplicate code path')
    return rel, sorted(files)


def bundle(candidate, cache, join_candidate=None):
    """The code tar (deterministic) and its SHA256SUMS; returns dict(tag, harness, join_harness, count, tar, sums,
    sums_sha256, files)."""
    harness, files = code_files(candidate, join_candidate)
    join = harness if join_candidate is None or Path(join_candidate) == candidate else \
        harness_files(Path(join_candidate), candidate.parents[2])[0]
    for name, path in files:
        if path.suffix == '.sh' and b'\r' in path.read_bytes():
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
                info.size, info.mtime, info.uid, info.gid = len(data), 0, 0, 0
                info.mode = 0o755 if path.suffix == '.sh' else 0o644
                archive.addfile(info, io.BytesIO(data))
        part.rename(tar)
    return dict(tag=tag, harness=harness, join_harness=join, count=len(files), tar=tar, sums=sums_path,
                sums_sha256=sums_sha,
                files={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files
                       if name.startswith((harness + '/postrun/', join + '/postrun/', 'tbench_verify/'))})


def ssh(host, command, check=True, stream=False):
    argv = ['ssh'] + SSH + [host, command]
    if not stream:
        done = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if check and done.returncode != 0:
            raise Refused('ssh %s failed (exit %s): %s' % (command[:120], done.returncode, (done.stderr or done.stdout)[-800:]))
        return done.returncode, done.stdout
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                               errors='replace')
    lines = []
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line.rstrip('\n'))
    return process.wait(), lines


def scp(sources, target):
    done = subprocess.run(['scp', '-q'] + SSH + [str(s) for s in sources] + [str(target)])
    if done.returncode != 0:
        raise Refused('scp failed with exit %s (rerun is safe)' % done.returncode)


def ensure_code(host, info):
    """Install code/<tag> once (extract, sha256sum -c --strict, exact file count, read only); then check it again."""
    tag, code = info['tag'], REMOTE + '/code/' + info['tag']
    sums = '%s/code/%s.SHA256SUMS' % (REMOTE, tag)
    verify = ('cd %s && printf \'%%s  %%s\\n\' %s %s | sha256sum -c --quiet && sha256sum -c --strict --quiet %s && '
              'test "$(find . -type f | wc -l)" -eq %d && test -z "$(find . ! -type f ! -type d)"'
              % (code, info['sums_sha256'], sums, sums, info['count']))
    code_rc, _ = ssh(host, 'test -d %s && test -f %s' % (code, sums), check=False)
    if code_rc != 0:
        print('CODE_UPLOAD tag=%s files=%d bytes=%d' % (tag, info['count'], info['tar'].stat().st_size))
        ssh(host, 'mkdir -p %s/code %s/tmp' % (REMOTE, REMOTE))
        scp([info['tar'], info['sums']], '%s:%s/tmp/' % (host, REMOTE))
        install = ('set -e; cd %(R)s/code; test ! -e %(T)s; rm -rf %(T)s.part; mkdir %(T)s.part; '
                   'nice -n 19 tar -xJf %(R)s/tmp/%(T)s.tar.xz -C %(T)s.part --no-same-owner; '
                   'mv %(R)s/tmp/%(T)s.SHA256SUMS %(R)s/code/%(T)s.SHA256SUMS; mv %(T)s.part %(T)s; '
                   'chmod -R a-w %(T)s; rm -f %(R)s/tmp/%(T)s.tar.xz; echo CODE_INSTALLED %(T)s'
                   % dict(R=REMOTE, T=tag))
        ssh(host, install)
    rc, out = ssh(host, verify, check=False)
    if rc != 0:
        raise Refused('the code tree on tbench differs from the local files (tag %s): %s' % (tag, out[-400:]))
    print('CODE_VERIFIED tag=%s files=%d' % (tag, info['count']))
    return code


def ensure_venvs(host, code, setup):
    names = {'tok': '%s/venv/tok-%s' % (REMOTE, sha_file(DRIVER / 'requirements-tok.txt')[:12])}
    ready = 'test -f %s/READY.json' % names['tok']
    if ssh(host, ready, check=False)[0] != 0:
        if not setup:
            raise Refused('the pinned venv is not on tbench yet: run once with --setup-venv')
        rc, _ = ssh(host, 'bash %s/tbench_verify/setup_venvs.sh' % code, check=False, stream=True)
        if rc != 0 or ssh(host, ready, check=False)[0] != 0:
            raise Refused('setup_venvs.sh failed (exit %s)' % rc)
    records = {}
    for name, venv in names.items():
        records[name] = json.loads(ssh(host, 'cat %s/READY.json' % venv)[1])
    return names, records


def ensure_evidence(host, run_id, wanted, local):
    """The two archives in evidence/<run_id>/ with the check block's SHA-256 values (upload --local-core/--local-bulk
    when tbench lacks one; never over a different file)."""
    folder = '%s/evidence/%s' % (REMOTE, run_id)
    rc, out = ssh(host, 'cd %s 2>/dev/null && for f in %s %s; do test -f $f && sha256sum $f; done; true'
                  % (folder, CORE, BULK), check=False)
    have = {line.split()[1]: line.split()[0] for line in out.splitlines() if len(line.split()) == 2}
    for name, digest in wanted.items():
        if have.get(name) == digest:
            continue
        if have.get(name):
            raise Refused('%s/%s has SHA-256 %s, not the check block\'s %s: keep both and report'
                          % (folder, name, have[name], digest))
        path = local.get(name)
        if not path:
            raise Refused('%s/%s is missing: copy it from DeltaAI first (--print-copy)' % (folder, name))
        if sha_file(path) != digest:
            raise Refused('local %s differs from the check block\'s SHA-256' % path)
        print('EVIDENCE_UPLOAD %s (%d bytes)' % (name, Path(path).stat().st_size))
        ssh(host, 'mkdir -p %s' % folder)
        scp([path], '%s:%s/%s.part' % (host, folder, name))
        ssh(host, 'cd %s && test "$(sha256sum %s.part | cut -d\' \' -f1)" = %s && mv %s.part %s'
            % (folder, name, digest, name, name))
    print('EVIDENCE_ON_TBENCH %s core=%s bulk=%s' % (folder, wanted[CORE], wanted[BULK]))


def print_copy(args, run_id, tbench):
    """The DeltaAI-side command (the user runs it in the DeltaAI shell): the jobs' tunnel key and SSH options
    (tbench_link.common_options) with the stage's pinned known_hosts copy; .part names renamed only after the copy;
    then the SHA-256 values on both sides (the local command takes the DeltaAI ones)."""
    print('S=%s; D=%s/evidence/%s; H=%s@%s; O="-i %s -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes '
          '-o UserKnownHostsFile=$S/%s -o GlobalKnownHostsFile=/dev/null -o CheckHostIP=no -o UpdateHostKeys=no '
          '-o ConnectTimeout=%d"; ssh $O $H "mkdir -p $D" && scp $O $S/fetch/%s $H:$D/%s.part '
          '&& scp $O $S/fetch/%s $H:$D/%s.part && ssh $O $H "cd $D && '
          'mv %s.part %s && mv %s.part %s && sha256sum %s %s" && (cd $S/fetch && sha256sum %s %s) '
          '&& sacct -j %s -X -n -o JobID,State,ExitCode'
          % (args.stage, REMOTE, run_id, tbench['user'], tbench['host'], tbench['identity_file'],
             tbench['known_hosts_file'], tbench['connect_timeout_seconds'], CORE, CORE, BULK, BULK, CORE, CORE, BULK,
             BULK, CORE, BULK, CORE, BULK, args.job_id))


def restore_tokenizers(job_dir, listed, candidate):
    """check/tokenizer (the pinned Qwen files) and check/tokenizer-gemma (the pinned Gemma 4 files) where tbench's list
    names them: remote_verify.pack_outputs leaves them out of the outputs archive."""
    if any(k.startswith('check/tokenizer/') for k in listed):
        target = job_dir / 'check' / 'tokenizer'
        target.mkdir()
        share = Path(candidate).resolve().parents[2].joinpath(*SHARE.split('/'))
        for name, (source, pin) in TOKENIZER.items():
            data = (share / source).read_bytes()
            if hashlib.sha256(data).hexdigest() != pin:
                raise Refused('pinned tokenizer evidence changed: ' + source)
            (target / name).write_bytes(data)
    if any(k.startswith('check/tokenizer-gemma/') for k in listed):
        target = job_dir / 'check' / 'tokenizer-gemma'
        target.mkdir()
        for name, pin in gemma_pins(candidate).items():
            data = (Path(GEMMA_META) / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != pin:
                raise Refused('pinned Gemma 4 metadata changed: ' + name)
            (target / name).write_bytes(data)


def finish_local(args, result, pulled, out_root, info, venvs, extra=None):
    """Unpack the pulled outputs into the verification folder, restore the tokenizer copies, check every file, record."""
    archive = pulled / Path(result['archive']['path']).name
    if sha_file(archive) != result['archive']['sha256']:
        raise Refused('pulled outputs differ from tbench\'s SHA-256')
    job_dir = out_root / ('tb16-job%s-%s' % (args.job_id, args.stamp))
    done = subprocess.run([sys.executable, '-I', '-B', str(HERE / 'unpack.py'), str(archive), str(job_dir)],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise Refused('the pulled outputs could not be unpacked safely: ' + (done.stdout + done.stderr)[-400:])
    listed = result.get('outputs') or {}
    restore_tokenizers(job_dir, listed, args.candidate_dir)
    local = {p.relative_to(job_dir).as_posix(): sha_file(p) for p in sorted(job_dir.rglob('*')) if p.is_file()}
    if local != listed:
        bad = sorted(k for k in set(local) | set(listed) if local.get(k) != listed.get(k))
        raise Refused('local outputs differ from tbench\'s list: %s' % bad[:10])
    report_path = job_dir / 'check' / 'local-verification.json'
    original = report_path.read_bytes()
    report = json.loads(original)
    same = (json.dumps(report, sort_keys=True, indent=2, default=str) + '\n').encode() == original
    report['verified_on'] = dict(
        host='tbench', hostname=result.get('host'), root=REMOTE, code_tag=info['tag'], code_dir=result.get('code_dir'),
        code_sha256sums_sha256=info['sums_sha256'], code_files=info['count'], harness=result.get('harness'),
        join_harness=result.get('join_harness') or result.get('harness'),
        postrun_sha256=info['files'], system_python=result.get('system_python'),
        venvs={k: dict(path=v['venv'], python=v['python'], freeze_sha256=v['freeze_sha256'],
                       requirements_sha256=v['requirements_sha256']) for k, v in venvs.items()},
        report_sha256_as_written=hashlib.sha256(original).hexdigest(), remote_job_dir=result.get('job_dir'),
        remote_runs_dir=result.get('runs_dir'), package=result.get('package'), steps=result.get('steps'),
        courtesy=result.get('courtesy'), seconds=result.get('seconds'), stamp=args.stamp,
        outputs_archive=result['archive'], **(extra or {}))
    if same:
        report_path.write_bytes((json.dumps(report, sort_keys=True, indent=2, default=str) + '\n').encode())
    else:  # never rewrite a report this script cannot reproduce byte for byte
        (job_dir / 'check' / 'verified-on.json').write_text(json.dumps(report['verified_on'], sort_keys=True, indent=2)
                                                            + '\n', encoding='utf-8', newline='\n')
    return job_dir


# ---------------------------------------------------------------- deploy before the job, pull after it


class Remote:
    """tbench access for --deploy / --pull: the `tbench` alias over SSH, or (local tests) a directory standing in for
    the root."""

    def __init__(self, host, local_root=None):
        self.host, self.local = host, Path(local_root) if local_root else None

    def path(self, rel):
        return self.local.joinpath(*PurePosixPath(rel).parts) if self.local else None

    def read(self, rel):
        """The text of <root>/<rel>, or None when absent."""
        if self.local:
            path = self.path(rel)
            return path.read_text(encoding='utf-8') if path.is_file() else None
        rc, out = ssh(self.host, 'cat %s/%s 2>/dev/null' % (REMOTE, shlex.quote(rel)), check=False)
        return out if rc == 0 else None

    def fetch(self, absolute, target):
        """Copy a file the remote side named by its absolute path (it must lie under the root)."""
        if self.local:
            source = Path(absolute)
            if self.local.resolve() not in source.resolve().parents:
                raise Refused('the outputs archive lies outside the verification root: %s' % absolute)
            shutil.copyfile(source, target / source.name)
            return
        if not str(absolute).startswith(REMOTE + '/out/'):
            raise Refused('the outputs archive lies outside %s/out: %s' % (REMOTE, absolute))
        scp(['%s:%s' % (self.host, absolute)], target)


def install_local(candidate, root):
    """Local tests only: the code tree code/<tag>/ (files copied, read-only not enforced) and its SHA256SUMS under a
    directory standing in for the root, as ensure_code leaves it on tbench."""
    harness, files = code_files(candidate)
    sums = ''.join('%s  %s\n' % (sha_file(path), name) for name, path in files).encode()
    tag = hashlib.sha256(sums).hexdigest()[:16]
    code = Path(root) / 'code' / tag
    if not code.is_dir():
        part = code.with_name(tag + '.part')
        if part.exists():
            shutil.rmtree(part)
        for name, path in files:
            target = part.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        (Path(root) / 'code' / (tag + '.SHA256SUMS')).write_bytes(sums)
        part.rename(code)
    return dict(tag=tag, harness=harness, count=len(files), sums_sha256=hashlib.sha256(sums).hexdigest())


def deploy(args, candidate, cache):
    """The code tree (ensure_code), the venv (ensure_venvs; --setup-venv builds it) and the payload pointer."""
    members = payload_members(candidate)
    ident = members_id(members)
    payload_sha = sha_file(candidate / 'dist' / 'payload.json')
    remote = Remote(args.tbench_host, args.local_root)
    if remote.local:
        info = install_local(candidate, remote.local)
        venvs = 'local test root: not checked'
    else:
        info = bundle(candidate, cache)
        code = ensure_code(args.tbench_host, info)
        names, records = ensure_venvs(args.tbench_host, code, args.setup_venv)
        venvs = {k: dict(path=v['venv'], freeze_sha256=v['freeze_sha256']) for k, v in records.items()}
    text = '%s\n%s\n%s\n' % (info['tag'], info['harness'], payload_sha)
    if remote.local:
        folder = remote.local / 'code' / 'by-payload'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (ident + '.part')).write_text(text, encoding='utf-8', newline='\n')
        os.replace(folder / (ident + '.part'), folder / ident)
    else:
        ssh(args.tbench_host, 'set -e; mkdir -p %(R)s/code/by-payload; cd %(R)s/code/by-payload; printf %%s %(T)s > '
            '%(I)s.part; mv %(I)s.part %(I)s' % dict(R=REMOTE, T=shlex.quote(text), I=ident))
        if remote.read('code/by-payload/' + ident) != text:
            raise Refused('the pointer on tbench does not read back')
    print('TBENCH_VERIFY_DEPLOYED payload=%s members_id=%s code_tag=%s files=%d harness=%s venvs=%s' % (
        payload_sha, ident, info['tag'], info['count'], info['harness'], json.dumps(venvs, sort_keys=True)))
    return 0


def pull_info(remote, tag, harness):
    """The code record of the tree that verified (finish_local's `info`), from its SHA256SUMS on tbench."""
    sums = remote.read('code/%s.SHA256SUMS' % tag)
    if sums is None or hashlib.sha256(sums.encode()).hexdigest()[:16] != tag:
        raise Refused('code/%s.SHA256SUMS missing on tbench or not that tag' % tag)
    files = {}
    for line in sums.splitlines():
        digest, name = line.split('  ', 1)
        files[name] = digest
    return dict(tag=tag, sums_sha256=hashlib.sha256(sums.encode()).hexdigest(), count=len(files),
                files={n: d for n, d in files.items() if n.startswith(harness + '/postrun/') or n.startswith('tbench_verify/')})


def slurm_confirmation(args, from_job):
    """(record, refusal or None): the END mail's state (required by --pull) against the job's own end record."""
    record = dict(source='job end record', note='from the job; confirmed with the END mail',
                  state=from_job.get('slurm_state'), exit_code=from_job.get('slurm_exit_code'),
                  job_status=from_job.get('job_status'), end_mail=dict(state=args.slurm_state,
                                                                       exit_code=args.slurm_exit_code))
    if (args.slurm_state, args.slurm_exit_code) != (record['state'], record['exit_code']):
        return dict(record, confirmed=False), (
            'the END mail says %s %s, the job\'s end record %s %s: the verification on tbench used the job\'s state; '
            'verify again with the manual command (--stage ... --core-sha256 ... --slurm-state %s --slurm-exit-code '
            '%s)' % (args.slurm_state, args.slurm_exit_code, record['state'], record['exit_code'], args.slurm_state,
                     args.slurm_exit_code))
    return dict(record, confirmed=True), None


def rewrite_slurm(job_dir, slurm):
    """An already pulled folder: record the END mail's confirmation in verified_on (only that field changes)."""
    report_path = job_dir / 'check' / 'local-verification.json'
    side = job_dir / 'check' / 'verified-on.json'
    if side.is_file():
        value = json.loads(side.read_bytes())
        value['slurm'] = slurm
        side.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8', newline='\n')
        return
    report = json.loads(report_path.read_bytes())
    report['verified_on']['slurm'] = slurm
    report_path.write_bytes((json.dumps(report, sort_keys=True, indent=2, default=str) + '\n').encode())


def pull(args, run_id, cache):
    remote = Remote(args.tbench_host, args.local_root)
    text = remote.read('out/%s/RESULT.json' % run_id)
    if text is None:
        request = remote.read('evidence/%s/request.json' % run_id)
        log = remote.read('out/%s/waiter.log' % run_id) or ''
        if request is None:
            print('TBENCH_NO_REQUEST run_id=%s: the job did not trigger a verification on tbench (fetch/tbench-push.json '
                  'says why); use the manual command (--print-copy, then --stage ...) or fetch_run.ps1.' % run_id)
            return 1
        print('TBENCH_VERIFY_PENDING run_id=%s: the waiter has not finished (evidence or Harbor run incomplete); last '
              'waiter lines: %s' % (run_id, ' | '.join(log.strip().splitlines()[-3:])))
        return 3
    result = json.loads(text)
    if result.get('run_id') != run_id or result.get('job_id') != args.job_id or not STAGE_RE.fullmatch(
            str(result.get('stage'))):
        raise Refused('RESULT.json names run %s job %s stage %s' % (result.get('run_id'), result.get('job_id'),
                                                                     result.get('stage')))
    if result.get('synthetic') and not args.local_root:
        raise Refused('a synthetic (test) result on tbench')
    for key, value in (('core', args.core_sha256), ('bulk', args.bulk_sha256)):
        if value and value.lower() != (result.get('archives') or {}).get(key):
            raise Refused('the check block\'s %s SHA-256 %s differs from the job\'s pushed %s'
                          % (key, value, (result.get('archives') or {}).get(key)))
    status = result.get('tb16_tbench_auto')
    print('TBENCH_RESULT status=%s verdict=%s join=%s harbor=%s code_tag=%s waited_s=%s harbor_running_at_start=%s' % (
        status, result.get('verdict'), result.get('join_verdict'), result.get('harbor_state'), result.get('code_tag'),
        result.get('waited_seconds'), result.get('harbor_running_at_start')))
    if status != 'DONE' or not result.get('archive'):
        print('TBENCH_VERIFY_REFUSED status=%s detail=%s' % (status, result.get('detail')))
        return 4 if status == 'ARCHIVE_MISMATCH' else 1
    slurm, refusal = slurm_confirmation(args, result.get('slurm_from_job') or {})
    if refusal:
        raise Refused(refusal)
    stamp = result['stamp']
    job_dir = Path(args.out_root) / ('tb16-job%s-%s' % (args.job_id, stamp))
    if job_dir.is_dir():  # pulled before: check it, record the confirmation
        local = {p.relative_to(job_dir).as_posix(): sha_file(p) for p in sorted(job_dir.rglob('*')) if p.is_file()}
        listed = dict(result.get('outputs') or {})
        report = local.pop('check/local-verification.json', None) and listed.pop('check/local-verification.json', None)
        local.pop('check/verified-on.json', None)
        if local != listed or report is None:
            raise Refused('%s exists and differs from tbench\'s list (move it away to pull again)' % job_dir)
        rewrite_slurm(job_dir, slurm)
        print('PULLED_BEFORE %s: Slurm CONFIRMED_BY_END_MAIL' % job_dir)
    else:
        info = pull_info(remote, result['code_tag'], result['harness'])
        named = str((result.get('result_json') or {}).get('path'))
        if remote.local:
            raw = Path(named).read_bytes() if remote.local.resolve() in Path(named).resolve().parents else None
        else:
            text_rj = remote.read(named[len(REMOTE) + 1:]) if named.startswith(
                '%s/out/%s/' % (REMOTE, run_id)) else None
            raw = text_rj.encode() if text_rj is not None else None
        if raw is None or hashlib.sha256(raw).hexdigest() != result['result_json']['sha256']:
            raise Refused('the verification record %s does not match RESULT.json' % named)
        verify_result = json.loads(raw)
        pulled = cache / ('job' + args.job_id) / ('tbench-auto-' + stamp)
        pulled.mkdir(parents=True, exist_ok=True)
        remote.fetch(result['archive']['path'], pulled)
        args.stamp = stamp
        venvs = verify_result.get('venvs') or {}
        if verify_result.get('synthetic'):
            venvs = {}
        extra = dict(mode='auto: pushed and triggered by the job, verified by the tbench waiter', slurm=slurm,
                     result_sha256=hashlib.sha256(text.encode()).hexdigest(), waited_seconds=result.get('waited_seconds'),
                     harbor_running_at_start=result.get('harbor_running_at_start'), pushed=result.get('pushed'),
                     work_cleaned=result.get('work_cleaned'), synthetic=bool(result.get('synthetic')))
        job_dir = finish_local(args, verify_result, pulled, Path(args.out_root), info, venvs, extra)
        marker = dict(core=result['archives']['core'], bulk=result['archives']['bulk'], check=str(job_dir / 'check'),
                      verdict=result.get('verdict'), code=result.get('verify_code'), status='DONE', host='tbench',
                      mode='auto', remote_job_dir=result.get('job_dir'), code_tag=result['code_tag'], stamp=stamp,
                      harbor_state=result.get('harbor_state'), slurm_confirmed=slurm['confirmed'])
        (cache / ('job' + args.job_id) / 'tbench-verified.json').write_text(json.dumps(marker), encoding='utf-8')
    check = job_dir / 'check'
    print('VERDICT=%s RUN_ID=%s CHECK_DIR=%s' % (result.get('verdict'), run_id, check))
    print('JOIN_DIR=%s TABLES_DIR=%s HARBOR=%s JOIN=%s KILLED_IN_FLIGHT_ROWS=%s' % (
        job_dir / result['join_dir'], job_dir / result['tables_dir'], result.get('harbor_state'),
        result.get('join_verdict'), (result.get('counts') or {}).get('killed_in_flight_rows')))
    print('SLURM=CONFIRMED_BY_END_MAIL state=%s exit=%s' % (slurm['state'], slurm['exit_code']))
    return int(result.get('exit_code', 1))


def checked_slurm(args, required):
    if args.slurm_state and not re.fullmatch(r'[A-Z_+]+', args.slurm_state):
        raise Refused('invalid Slurm state')
    if args.slurm_exit_code and not re.fullmatch(r'[0-9]+:[0-9]+', args.slurm_exit_code):
        raise Refused('invalid Slurm exit code')
    if bool(args.slurm_state) != bool(args.slurm_exit_code):
        raise Refused('give both --slurm-state and --slurm-exit-code, or neither')
    if required and not args.slurm_state:
        raise Refused('--pull needs the END mail\'s --slurm-state and --slurm-exit-code (e.g. COMPLETED 0:0)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', default='')
    parser.add_argument('--job-id', default='')
    parser.add_argument('--run-id', default='', help='default tb16-<job id>-r1')
    parser.add_argument('--core-sha256', default='')
    parser.add_argument('--bulk-sha256', default='')
    parser.add_argument('--slurm-state', default='')
    parser.add_argument('--slurm-exit-code', default='')
    parser.add_argument('--candidate-dir', default=str(HERE.parent), help='the harness copy whose code verifies the job')
    parser.add_argument('--join-candidate-dir', default='',
                        help='manual mode only: the harness copy whose join_tb16 / tb16_tables join the job (default: '
                        'the candidate), e.g. a later join on a job verified by its own frozen copy')
    parser.add_argument('--local-core', default='', help='upload this local archive when tbench lacks it (tests)')
    parser.add_argument('--local-bulk', default='')
    parser.add_argument('--out-root', default=str(HERE.parents[3] / 'verification'))
    parser.add_argument('--cache-root', default=r'D:\Remote-RA-temp\scratch\perf-tb16-strip\fetch')
    parser.add_argument('--tbench-host', default='tbench')
    parser.add_argument('--setup-venv', action='store_true', help='build the pinned venv on tbench if missing')
    parser.add_argument('--print-copy', action='store_true', help='print the DeltaAI-side copy command and exit')
    parser.add_argument('--self-test', action='store_true', help='no connection: inputs, code bundle, tools')
    parser.add_argument('--deploy', action='store_true', help='install the code tree and the payload pointer')
    parser.add_argument('--pull', action='store_true', help='pull the job-triggered verification (END mail values)')
    parser.add_argument('--local-root', default='', help='local tests only: a directory standing in for the tbench root')
    args = parser.parse_args(argv)
    keep_temp_off_c()
    cache = Path(args.cache_root)
    if str(cache.resolve())[:2].upper() == 'C:':
        raise Refused('the cache must not be on C:')
    if args.join_candidate_dir and (args.deploy or args.pull or args.print_copy):
        raise Refused('--join-candidate-dir is for the manual mode only (a deploy and a job-triggered verification '
                      'join with the candidate\'s own join)')
    if args.deploy:
        return deploy(args, Path(args.candidate_dir).resolve(strict=True), cache)
    if args.pull:
        if not args.job_id.isdigit():
            raise Refused('invalid Slurm job id')
        run_id = args.run_id or 'tb16-%s-r1' % args.job_id
        if not re.fullmatch(r'tb16-%s-r[0-9]{1,2}' % args.job_id, run_id):
            raise Refused('invalid run id %s' % run_id)
        for value in (args.core_sha256, args.bulk_sha256):
            if value and not HEX64.fullmatch(value):
                raise Refused('invalid SHA-256 %s' % value)
        checked_slurm(args, required=True)
        args.candidate_dir = str(Path(args.candidate_dir).resolve(strict=True))
        return pull(args, run_id, cache)
    if args.self_test and not args.stage:
        args.stage, args.job_id = '/u/ycao11/tb16-' + '0' * 32, '1'
        args.core_sha256 = args.bulk_sha256 = '0' * 64
    if not STAGE_RE.fullmatch(args.stage):
        raise Refused('invalid tb16 stage path (expected /u/ycao11/tb16-<32 hex>)')
    if not args.job_id.isdigit():
        raise Refused('invalid Slurm job id')
    run_id = args.run_id or 'tb16-%s-r1' % args.job_id
    if not re.fullmatch(r'tb16-%s-r[0-9]{1,2}' % args.job_id, run_id):
        raise Refused('invalid run id %s' % run_id)
    if args.print_copy:
        pins = json.loads((Path(args.candidate_dir) / 'pins.json').read_bytes())
        print_copy(args, run_id, pins['tbench'])
        return 0
    for value in (args.core_sha256, args.bulk_sha256):
        if not HEX64.fullmatch(value):
            raise Refused('give --core-sha256 and --bulk-sha256 (the 64-hex values of the DeltaAI check block)')
    checked_slurm(args, required=False)
    candidate = Path(args.candidate_dir).resolve(strict=True)
    args.candidate_dir = str(candidate)
    join_candidate = Path(args.join_candidate_dir).resolve(strict=True) if args.join_candidate_dir else None
    if join_candidate is not None and not (join_candidate / 'postrun' / 'join_tb16.py').is_file():
        raise Refused('the join candidate %s has no postrun/join_tb16.py' % join_candidate)
    wanted = {CORE: args.core_sha256.lower(), BULK: args.bulk_sha256.lower()}
    info = bundle(candidate, cache, join_candidate)
    if args.self_test:
        for tool in ('ssh', 'scp'):
            if not shutil.which(tool):
                raise Refused(tool + ' not found')
        print('TBENCH_VERIFY_SELFTEST_PASS connections=0 code_tag=%s files=%d harness=%s join_harness=%s run_id=%s '
              'out_root=%s' % (info['tag'], info['count'], info['harness'], info['join_harness'], run_id, args.out_root))
        return 0
    host = args.tbench_host
    started = time.time()
    code = ensure_code(host, info)
    names, venvs = ensure_venvs(host, code, args.setup_venv)
    ensure_evidence(host, run_id, wanted, {CORE: args.local_core, BULK: args.local_bulk})
    args.stamp = time.strftime('%Y%m%dT%H%M%S')
    command = ['python3', '-I', '-B', code + '/tbench_verify/remote_verify.py', 'run', '--harness', info['harness'],
               '--run-id', run_id, '--job-id', args.job_id, '--stage', args.stage, '--core-sha256', wanted[CORE],
               '--bulk-sha256', wanted[BULK], '--stamp', args.stamp, '--tok-venv', names['tok'],
               '--code-tag', info['tag']]
    if args.slurm_state:
        command += ['--slurm-state', args.slurm_state, '--slurm-exit-code', args.slurm_exit_code]
    if info['join_harness'] != info['harness']:
        command += ['--join-harness', info['join_harness']]
    rc, lines = ssh(host, ' '.join(shlex.quote(c) for c in command), check=False, stream=True)
    last = [line for line in lines if line.startswith('{"')]
    if not last:
        raise Refused('remote_verify.py printed no result (exit %s)' % rc)
    result = json.loads(last[-1])
    status = result.get('tb16_tbench_verify')
    remote_seconds = result.get('seconds')
    if not result.get('archive'):
        raise Refused('tbench verification %s (exit %s): %s' % (status, rc, result.get('detail')))
    pulled = cache / ('job' + args.job_id) / ('tbench-verify-' + args.stamp)
    pulled.mkdir(parents=True)
    scp(['%s:%s' % (host, result['archive']['path'])], pulled)
    job_dir = finish_local(args, result, pulled, Path(args.out_root), info, venvs)
    check = job_dir / 'check'
    marker = dict(core=wanted[CORE], bulk=wanted[BULK], check=str(check), verdict=result.get('verdict'),
                  code=result.get('verify_code'), status=status, host='tbench', remote_job_dir=result.get('job_dir'),
                  code_tag=info['tag'], stamp=args.stamp, harbor_state=result.get('harbor_state'),
                  harness=info['harness'], join_harness=info['join_harness'])
    (cache / ('job' + args.job_id) / 'tbench-verified.json').write_text(json.dumps(marker), encoding='utf-8')
    print('VERDICT=%s RUN_ID=%s CHECK_DIR=%s' % (result.get('verdict'), run_id, check))
    if result.get('join_dir'):
        print('JOIN_DIR=%s TABLES_DIR=%s HARBOR=%s' % (job_dir / result['join_dir'], job_dir / result['tables_dir'],
                                                       result.get('harbor_state')))
        join_json = job_dir / result['join_dir'] / 'join.json'
        joined = json.loads(join_json.read_bytes()) if join_json.is_file() else {}
        print('JOIN=%s KILLED_IN_FLIGHT_ROWS=%s JOIN_HARNESS=%s' % (joined.get('verdict'),
                                                                   joined.get('killed_in_flight_rows'),
                                                                   info['join_harness']))
    if status == 'HARBOR_RUNNING':
        print('TBENCH_HARBOR_RUNNING run_id=%s: run this same command again later.' % run_id)
    print('VERIFIED_ON=tbench status=%s remote_seconds=%s total_seconds=%.1f code_tag=%s' % (
        status, remote_seconds, time.time() - started, info['tag']))
    return int(result.get('exit_code', 1))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Refused as exc:
        print('TBENCH_VERIFY_REFUSED ' + str(exc))
        sys.exit(1)
