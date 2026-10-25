"""READY and clean-shutdown checks of one selective-prefix server (README sections 4.1 and 8); stdlib only.

  server_checks.py ready <output dir> <policy> <release dir>
      exit 0: READY; 1: not yet (ready.json absent or still being written); 2: READY for something else.
  server_checks.py cleanup <output dir>
      exit 0: the shutdown receipts are clean; 2: they are not. Run it after the server exited with code 0.

The validated jobs applied the same conditions (READY before any request; the receipts after every server).
"""
import hashlib
import json
import sys
from pathlib import Path

HASHES = ('task_sha256', 'tasks_sha256', 'input_pins_sha256', 'common_config_sha256', 'server_config_sha256')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(output, policy, release):
    output = Path(output)
    try:
        value = json.loads((output / 'ready.json').read_bytes())
    except (FileNotFoundError, ValueError):   # created first, written after: retry
        return 1, 'not ready yet'
    startup = json.loads((output / 'startup.json').read_bytes())
    problems = []
    if value.get('native_http_ready') is not True:
        problems.append('native_http_ready is not true')
    if value.get('evidence_kind') != 'NATIVE_HTTP_RUNTIME':
        problems.append('evidence_kind is %r' % value.get('evidence_kind'))
    if value.get('policy') != policy:
        problems.append('policy is %r' % value.get('policy'))
    if value.get('runtime_manifest') != sha256(Path(release) / 'runtime-manifest.json'):
        problems.append('runtime_manifest differs from this release')
    for name in HASHES:
        if name in startup and value.get(name) != startup[name]:
            problems.append(name + ' differs from startup.json')
    return (2, '; '.join(problems)) if problems else (0, 'READY')


def cleanup(output):
    output = Path(output)
    raw = (output / 'engine-cleanup.json').read_bytes()
    engine, parent = json.loads(raw), json.loads((output / 'parent-cleanup.json').read_bytes())
    value = json.loads((output / 'ready.json').read_bytes())
    problems = []
    if parent.get('cleanup_sha256') != hashlib.sha256(raw).hexdigest():
        problems.append('parent.cleanup_sha256 is not the SHA-256 of engine-cleanup.json')
    for name, receipt in (('engine', engine), ('parent', parent)):
        if receipt.get('joined') is not True:
            problems.append(name + '.joined is not true')
        for key in ('epoch', 'session_id'):
            if receipt.get(key) != value.get(key):
                problems.append('%s.%s differs from ready.json' % (name, key))
    if engine.get('runtime_manifest') != value.get('runtime_manifest'):
        problems.append('engine.runtime_manifest differs from ready.json')
    if engine.get('pending_operations') != 0:
        problems.append('engine.pending_operations is %r' % engine.get('pending_operations'))
    if parent.get('child_exitcode') != 0 or parent.get('natural_exit') is not True:
        problems.append('parent: child_exitcode %r, natural_exit %r' % (parent.get('child_exitcode'),
                                                                         parent.get('natural_exit')))
    return (2, '; '.join(problems)) if problems else (0, 'CLEAN_SHUTDOWN')


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == 'ready':
        code, message = ready(*sys.argv[2:])
    elif len(sys.argv) == 3 and sys.argv[1] == 'cleanup':
        code, message = cleanup(sys.argv[2])
    else:
        sys.exit(__doc__)
    print(message, flush=True)
    sys.exit(code)
