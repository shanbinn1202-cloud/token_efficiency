#!/usr/bin/env bash
# perf-swe32 tbench verification: build the two pinned verifier venvs under /home/ubuntu/swe32-verify/venv (once per
# requirements file; a built venv is never changed). Run by postrun/verify_on_tbench.py --setup-venv from the uploaded,
# SHA-256-checked code tree. tbench's Python 3.12 has no pip and no ensurepip: pip 24.2 runs from its pinned wheel
# (sha256 checked), the venvs are made with --without-pip, every wheel is downloaded and installed with
# --require-hashes --no-deps --only-binary=:all:. Writes venv/<name>-<12 hex of the requirements sha256>/{freeze.txt,
# pip-check.txt, READY.json}. Only /home/ubuntu/swe32-verify is written; low priority (nice 19).
set -euo pipefail
ROOT=/home/ubuntu/swe32-verify
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PIP_WHL=$ROOT/wheels/pip-24.2-py3-none-any.whl
PIP_SHA=2cd581cf58ab7fcfca4ce8efa6dcacd0de5bf8d0a3eb9ec927e07405f4d9e2a2
PIP_URL=https://files.pythonhosted.org/packages/d4/55/90db48d85f7689ec6f81c0db0622d704306c5284850383c090e6c7195a5c/pip-24.2-py3-none-any.whl
export TMPDIR=$ROOT/tmp PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$ROOT/wheels" "$ROOT/venv" "$TMPDIR"
if [[ ! -f $PIP_WHL ]]; then
  curl -sSfL -o "$PIP_WHL.part" "$PIP_URL"
  mv "$PIP_WHL.part" "$PIP_WHL"
fi
printf '%s  %s\n' "$PIP_SHA" "$PIP_WHL" | sha256sum -c --quiet
for name in tok parity; do
  req=$HERE/requirements-$name.txt
  req_sha=$(sha256sum "$req" | cut -d' ' -f1)
  venv=$ROOT/venv/$name-${req_sha:0:12}
  if [[ -f $venv/READY.json ]]; then
    echo "VENV_READY $venv"
    continue
  fi
  rm -rf "$venv"  # an unfinished build of this same requirements file (READY.json is written last)
  extra=()
  [[ $name == parity ]] && extra=(--extra-index-url https://download.pytorch.org/whl/cpu)
  mkdir -p "$ROOT/wheels/$name"
  nice -n 19 python3 -I "$PIP_WHL/pip" download -q --require-hashes --no-deps --only-binary=:all: \
    -d "$ROOT/wheels/$name" -r "$req" "${extra[@]}"
  python3 -I -m venv --without-pip "$venv"
  nice -n 19 "$venv/bin/python" -I "$PIP_WHL/pip" install -q --require-hashes --no-deps --only-binary=:all: \
    --no-index --find-links "$ROOT/wheels/$name" -r "$req"
  "$venv/bin/python" -I "$PIP_WHL/pip" freeze --all > "$venv/freeze.txt"
  "$venv/bin/python" -I "$PIP_WHL/pip" check > "$venv/pip-check.txt" 2>&1 || true
  if [[ $name == tok ]]; then
    "$venv/bin/python" -I -B -c 'import tokenizers, transformers, numpy; print(tokenizers.__version__, transformers.__version__, numpy.__version__)'
  else
    "$venv/bin/python" -I -B -c 'import torch, numpy; print(torch.__version__, numpy.__version__, torch.get_num_threads())'
  fi
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
