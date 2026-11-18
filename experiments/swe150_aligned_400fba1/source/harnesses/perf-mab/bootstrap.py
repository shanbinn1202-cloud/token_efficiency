"""Executed with -I -S: stdlib-only receiver (v6 mechanics), never imports vLLM."""
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

# vLLM binds ZMQ IPC sockets at $TMPDIR/<uuid4>; $STAGE/tmp must leave room under
# AF_UNIX sun_path (107 usable bytes). v4 job 3260831 failed at 109 bytes.
STAGE_PREFIX = 'pmab-'
IPC_PATH_BUDGET = 99  # len($STAGE/tmp) + '/' + 36-byte uuid4; 8 bytes below the limit


def ipc_path_bytes(stage):
    return len(str(Path(stage) / 'tmp').encode()) + 1 + 36


def without_bom(raw):
    """A UTF-8 console stdin writer on Windows may prepend one BOM; the JSON is unchanged."""
    return raw[3:] if raw.startswith(b'\xef\xbb\xbf') else raw


def receive(root, nonce, raw):
    if len(nonce) != 32 or any(c not in '0123456789abcdef' for c in nonce):
        raise ValueError('invalid nonce')
    root = Path(root).resolve(strict=True)
    stage = root / (STAGE_PREFIX + nonce)
    if ipc_path_bytes(stage) > IPC_PATH_BUDGET:
        raise ValueError('stage path too long for AF_UNIX IPC sockets')
    stage.mkdir(mode=0o700)  # Exclusive; a site-managed parent is allowed.
    info = stage.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stage.resolve().parent != root:
        raise ValueError('unsafe new stage')
    stage.chmod(0o700)  # Normalize setgid inherited from the parent.
    if stat.S_IMODE(stage.lstat().st_mode) != 0o700:
        raise ValueError('stage mode is not 0700')
    print('CLEAN_STAGE=' + str(stage), flush=True)
    for name in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[name] = str(stage)
    raw = without_bom(raw)
    envelope = json.loads(raw)
    for name, item in envelope['files'].items():
        if Path(name).name != name or name in ('.', '..') or '\\' in name:
            raise ValueError('unsafe payload name')
        data = base64.b64decode(item['base64'], validate=True)
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('payload hash mismatch')
        with (stage / name).open('xb') as stream:
            stream.write(data)
    with (stage / 'delivery.json').open('xb') as stream:
        stream.write(raw)
    options = envelope['options']
    # perf-mab: the phrase names one pinned plan (pins.json config.plans.<plan>.authorization).
    plan = json.loads((stage / 'pins.json').read_bytes())['config']['plans'].get(options.get('plan'))
    if plan is None or options.get('authorization') != plan['authorization']:
        raise ValueError('authorization mismatch')
    with (stage / 'options.json').open('x') as stream:
        json.dump(options, stream, sort_keys=True)
    # This interpreter is a stdlib bootstrap only: `setup` hands `qualify` to the
    # reused v9 clean environment's own interpreter; nothing here is imported from it.
    return subprocess.run([sys.executable, '-I', '-S', '-B', str(stage / 'controller.py'),
                           'setup', str(stage)], check=False).returncode


if __name__ == '__main__':
    raw = sys.stdin.buffer.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError('payload limit exceeded')
    sys.exit(receive('/u/ycao11', sys.argv[1], raw))
