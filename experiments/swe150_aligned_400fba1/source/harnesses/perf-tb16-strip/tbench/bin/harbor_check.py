"""Read-only check of the pinned Harbor install and the Terminal-Bench task packages on tbench (stdlib only; perf-tb16).
Run by preflight.py (before every claim) and by run_harbor.sh (before every run), under the host python.

- Harbor: <site>/harbor-<version>.dist-info/RECORD has the pinned SHA-256; every file it lists with a hash (harbor/*,
  the dist-info files, ../../../bin/harbor, hb, hr) exists with that hash and size; no other file under <site>/harbor
  except __pycache__/*.pyc (bytecode is checked against its source by Python itself); the venv's python version.
- litellm and openai: exactly one dist-info each, at the pinned version, with the pinned RECORD SHA-256.
- The environment: no MSWEA_CTX_* variable (the former ctx patch's switches; run_configs.sh must not be sourced).
- Tasks: for each pinned task, <cache>/<name>/ holds exactly the pinned content-hash directory and its tree SHA-256
  (sha256 over "<file sha256>  <relative path>\\n" of every file, sorted by path; a symlink as "link:<target>") equals
  the pin.

Usage: harbor_check.py <harbor env> <task cache> <pins.json file>   (pins: {harbor: {...}, tasks: [{name,
content_hash, tree_sha256}, ...]}). One JSON line {"event": "tb16_harbor_check", "status": "PASS" | "FAIL",
"failures": [...], ...}; exit 0 on PASS, 1 on FAIL, 2 on a usage error.
"""
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h


def record_digest(path):
    """The RECORD form of a file's hash: sha256=<urlsafe base64 without padding>."""
    return 'sha256=' + base64.urlsafe_b64encode(sha256_file(path).digest()).decode().rstrip('=')


def site_of(env):
    found = sorted(Path(env, 'lib').glob('python3*/site-packages'))
    return found[0] if len(found) == 1 else None


def venv_version(env):
    path = Path(env) / 'pyvenv.cfg'
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        key, _, value = line.partition('=')
        if key.strip() == 'version_info':
            return value.strip()
    return None


def check_record(site, version, record_sha256, record_files, allowed_extra=None):
    """The wheel RECORD check. `allowed_extra` {relative path: sha256 hex}: files outside the RECORD tolerated with
    exactly that content (reported under allowed_extra)."""
    allowed_extra = allowed_extra or {}
    failures, info = [], {}
    dist = site / ('harbor-%s.dist-info' % version)
    record = dist / 'RECORD'
    if not record.is_file():
        return ['harbor %s is not installed (%s missing)' % (version, record)], info
    info['record_sha256'] = sha256_file(record).hexdigest()
    if info['record_sha256'] != record_sha256:
        failures.append('harbor RECORD %s differs from the pin %s' % (info['record_sha256'][:16], record_sha256[:16]))
    listed, bad, missing = set(), [], []
    rows = list(csv.reader(io.StringIO(record.read_text(encoding='utf-8'))))
    for row in rows:
        if len(row) != 3:
            bad.append('malformed RECORD row %r' % (row,))
            continue
        name, digest, size = row
        path = (site / name).resolve() if name.startswith('../') else site / name
        listed.add(path.resolve() if path.exists() else path)
        if not digest:
            continue  # RECORD itself
        if not path.is_file():
            missing.append(name)
            continue
        if not size.isdigit() or path.stat().st_size != int(size) or record_digest(path) != digest:
            bad.append(name)
    hashed = sum(1 for row in rows if len(row) == 3 and row[1])
    info.update(record_files=hashed, mismatched=bad[:20], missing=missing[:20])
    if hashed != record_files:
        failures.append('harbor RECORD lists %d hashed files, the pin %d' % (hashed, record_files))
    if bad:
        failures.append('%d harbor files differ from the wheel RECORD: %s' % (len(bad), bad[:6]))
    if missing:
        failures.append('%d harbor files of the wheel RECORD are missing: %s' % (len(missing), missing[:6]))
    extra, tolerated = [], []
    for path in sorted((site / 'harbor').rglob('*')):
        if not path.is_file() and not path.is_symlink():
            continue
        if path.suffix == '.pyc' and path.parent.name == '__pycache__':
            continue
        if path.resolve() not in listed and path not in listed:
            rel = path.relative_to(site).as_posix()
            if rel in allowed_extra and path.is_file() and not path.is_symlink() \
                    and sha256_file(path).hexdigest() == allowed_extra[rel]:
                tolerated.append(rel)
            else:
                extra.append(rel)
    info['extra'] = extra[:20]
    info['allowed_extra'] = tolerated
    if extra:
        failures.append('%d files under harbor/ are not in the wheel RECORD: %s' % (len(extra), extra[:6]))
    return failures, info


def check_dist(site, name, version, record_sha256):
    found = sorted(p for p in site.glob('%s-*.dist-info' % name) if p.name.split('-')[0] == name)
    if len(found) != 1:
        return ['%s: %d dist-info directories (%s)' % (name, len(found), [p.name for p in found])], None
    seen = found[0].name[len(name) + 1:-len('.dist-info')]
    failures = []
    if seen != version:
        failures.append('%s is %s, the pin %s' % (name, seen, version))
    record = found[0] / 'RECORD'
    digest = sha256_file(record).hexdigest() if record.is_file() else None
    if digest != record_sha256:
        failures.append('%s RECORD %s differs from the pin %s' % (name, (digest or 'absent')[:16], record_sha256[:16]))
    return failures, dict(version=seen, record_sha256=digest)


def tree_sha256(folder):
    lines = []
    for path in sorted(Path(folder).rglob('*')):
        rel = path.relative_to(folder).as_posix()
        if path.is_symlink():
            lines.append('link:%s  %s\n' % (os.readlink(path), rel))
        elif path.is_file():
            lines.append('%s  %s\n' % (sha256_file(path).hexdigest(), rel))
    return hashlib.sha256(''.join(lines).encode()).hexdigest()


def check_tasks(cache, tasks):
    failures, bad = [], []
    for task in tasks:
        folder = Path(cache) / task['name']
        digests = sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
        if digests != [task['content_hash']]:
            bad.append('%s: cached %s, pinned %s' % (task['name'], [d[:12] for d in digests], task['content_hash'][:12]))
            continue
        if tree_sha256(folder / task['content_hash']) != task['tree_sha256']:
            bad.append('%s: the package files differ from the pin' % task['name'])
    if bad:
        failures.append('%d task packages differ from the pins: %s' % (len(bad), bad[:4]))
    return failures, dict(tasks=len(tasks), bad=bad[:20])


def main(env, cache, pins_path):
    value = check(env, cache, json.loads(Path(pins_path).read_text()))
    print(json.dumps(value, sort_keys=True))
    return 0 if value['status'] == 'PASS' else 1


def check(env, cache, pins):
    """The check as a dict (preflight.py imports it)."""
    harbor = pins['harbor']
    failures, info = [], dict(env=str(env), cache=str(cache))
    site = site_of(env)
    if site is None:
        failures.append('no site-packages under %s/lib' % env)
    else:
        found, record = check_record(site, harbor['version'], harbor['record_sha256'], harbor['record_files'],
                                     harbor.get('allowed_extra'))
        failures += found
        info['harbor'] = record
        info['dependencies'] = {}
        for name, pin in sorted((harbor.get('dependencies') or {}).items()):
            found, value = check_dist(site, name, pin['version'], pin['record_sha256'])
            failures += found
            info['dependencies'][name] = value
    version = venv_version(env)
    info['python_version'] = version
    if harbor.get('python_version') and version != harbor['python_version']:
        failures.append('the Harbor env runs python %s, the pin %s' % (version, harbor['python_version']))
    ctx = sorted(k for k in os.environ if k.startswith('MSWEA_CTX'))
    info['mswea_ctx_env'] = ctx
    if ctx:
        failures.append('MSWEA_CTX_* variables are set (%s): run_configs.sh must not be sourced' % ', '.join(ctx))
    found, value = check_tasks(cache, pins.get('tasks') or [])
    failures += found
    info['task_packages'] = value
    status = 'PASS' if not failures else 'FAIL'
    return dict(event='tb16_harbor_check', status=status, failures=failures, **info)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(*sys.argv[1:4]))
