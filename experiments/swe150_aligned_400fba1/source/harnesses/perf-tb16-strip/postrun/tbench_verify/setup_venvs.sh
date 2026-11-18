#!/usr/bin/env bash
# perf-tb16-strip tbench verification (perf-swe32's, 2026-11-11; ported 2026-11-12): build the pinned verifier venv
# "tok" under /home/ubuntu/tb16-verify/venv (once per requirements file; a built venv is never changed). Run by
# postrun/verify_on_tbench.py --deploy --setup-venv from the uploaded, SHA-256-checked code tree. tbench's Python 3.12 has
# no pip and no ensurepip: pip 24.2 runs from its pinned wheel (sha256 checked), the venv is made with --without-pip,
# every wheel is downloaded and installed with --require-hashes --no-deps --only-binary=:all: (wheels perf-swe32's
# verification root already holds are copied first; pip checks each against its pinned hash). Writes
# venv/tok-<12 hex of the requirements sha256>/{freeze.txt, pip-check.txt, READY.json}. Only /home/ubuntu/tb16-verify
# is written; low priority (nice 19). The verifier needs no selection replay, so no "parity" venv.
set -euo pipefail
ROOT=/home/ubuntu/tb16-verify
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIP_WHL=$ROOT/wheels/pip-24.2-py3-none-any.whl
PIP_SHA=2cd581cf58ab7fcfca4ce8efa6dcacd0de5bf8d0a3eb9ec927e07405f4d9e2a2
PIP_URL=https://files.pythonhosted.org/packages/d4/55/90db48d85f7689ec6f81c0db0622d704306c5284850383c090e6c7195a5c/pip-24.2-py3-none-any.whl
SHARED_WHEELS=/home/ubuntu/swe32-verify/wheels
export TMPDIR=$ROOT/tmp PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$ROOT/wheels" "$ROOT/venv" "$TMPDIR"
if [[ ! -f $PIP_WHL ]]; then
  if [[ -f $SHARED_WHEELS/pip-24.2-py3-none-any.whl ]]; then
    cp "$SHARED_WHEELS/pip-24.2-py3-none-any.whl" "$PIP_WHL.part"
  else
    curl -sSfL -o "$PIP_WHL.part" "$PIP_URL"
  fi
  mv "$PIP_WHL.part" "$PIP_WHL"
fi
printf '%s  %s\n' "$PIP_SHA" "$PIP_WHL" | sha256sum -c --quiet
for name in tok; do
  req=$HERE/requirements-$name.txt
  req_sha=$(sha256sum "$req" | cut -d' ' -f1)
  venv=$ROOT/venv/$name-${req_sha:0:12}
  if [[ -f $venv/READY.json ]]; then
    echo "VENV_READY $venv"
    continue
  fi
  rm -rf "$venv"  # an unfinished build of this same requirements file (READY.json is written last)
  mkdir -p "$ROOT/wheels/$name"
  if [[ -d $SHARED_WHEELS/$name ]]; then
    cp -n "$SHARED_WHEELS/$name"/*.whl "$ROOT/wheels/$name/" 2>/dev/null || true
  fi
  nice -n 19 python3 -I "$PIP_WHL/pip" download -q --require-hashes --no-deps --only-binary=:all: \
    -d "$ROOT/wheels/$name" -r "$req"
  python3 -I -m venv --without-pip "$venv"
  nice -n 19 "$venv/bin/python" -I "$PIP_WHL/pip" install -q --require-hashes --no-deps --only-binary=:all: \
    --no-index --find-links "$ROOT/wheels/$name" -r "$req"
  "$venv/bin/python" -I "$PIP_WHL/pip" freeze --all > "$venv/freeze.txt"
  "$venv/bin/python" -I "$PIP_WHL/pip" check > "$venv/pip-check.txt" 2>&1 || true
  "$venv/bin/python" -I -B -c 'import tokenizers, transformers, numpy; print(tokenizers.__version__, transformers.__version__, numpy.__version__)'
  python3 -I -B - "$venv" "$name" "$req_sha" <<'EOF'
import hashlib, json, platform, subprocess, sys
from pathlib import Path
venv, name, req_sha = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
freeze = (venv / 'freeze.txt').read_bytes()
version = subprocess.check_output([str(venv / 'bin' / 'python'), '-I', '-c', 'import sys; print(sys.version)']).decode().strip()
value = dict(name=name, venv=str(venv), requirements_sha256=req_sha, freeze_sha256=hashlib.sha256(freeze).hexdigest(),
             freeze=freeze.decode().splitlines(), pip_check=(venv / 'pip-check.txt').read_text().splitlines(),
             python=version, host=platform.node(), machine=platform.machine())
with (venv / 'READY.json').open('x') as stream:
    json.dump(value, stream, indent=1, sort_keys=True)
    stream.write('\n')
print('VENV_BUILT %s freeze_sha256=%s' % (venv, value['freeze_sha256']))
EOF
done
