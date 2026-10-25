#!/bin/bash
# Harbor server bootstrap — essential plumbing then start server.
# First arg is WORKDIR (container cwd), rest are server args.
export WORKDIR="${1:-/app}"; shift
export HARBOR_STAGING="/staging/env_files"
mkdir -p "$WORKDIR"

export DEBIAN_FRONTEND=noninteractive

# --- Refresh package index once (apt-based images) ---
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq 2>/dev/null || true
fi

# --- Ensure /usr/bin/python3 exists ---
_SYS_PY=/usr/bin/python3
if [ ! -x "$_SYS_PY" ]; then
  echo "[runtime] /usr/bin/python3 not found, installing..." >&2
  if command -v apt-get >/dev/null 2>&1; then
    apt-get install -y -qq python3 python3-venv 2>/dev/null || true
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache python3 2>/dev/null || true
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y python3 2>/dev/null || true
  elif command -v yum >/dev/null 2>&1; then
    yum install -y python3 2>/dev/null || true
  fi
  if [ ! -x "$_SYS_PY" ]; then
    echo "[runtime] FATAL: cannot install /usr/bin/python3" >&2
    exit 1
  fi
fi

# Debian's minimal Python 3.9 omits the stdlib distutils needed by its supported pip.
if "$_SYS_PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] < (3,10) else 1)' && ! "$_SYS_PY" -c 'import distutils.cmd' 2>/dev/null; then
  apt-get install -y -qq python3-distutils || { echo '[runtime] Cannot install existing Python stdlib distutils' >&2; exit 1; }
fi

# --- Create an isolated venv for the Harbor server at /opt/harbor-server ---
_HARBOR_VENV=/opt/harbor-server
_HARBOR_PY="$_HARBOR_VENV/bin/python3"
if [ ! -x "$_HARBOR_PY" ]; then
  echo "[runtime] Creating server venv at $_HARBOR_VENV..." >&2
  "$_SYS_PY" -m venv --without-pip "$_HARBOR_VENV" \
    || { echo "[runtime] FATAL: cannot create server venv" >&2; exit 1; }
  # Bootstrap pip into the venv (try multiple strategies)
  if "$_HARBOR_PY" -c "import ensurepip" 2>/dev/null; then
    "$_HARBOR_PY" -m ensurepip --default-pip 2>/dev/null || true
  fi
  if ! "$_HARBOR_PY" -m pip --version 2>/dev/null; then
    # Try using system pip to install pip into the venv
    if "$_SYS_PY" -m pip --version 2>/dev/null; then
      echo "[runtime] Bootstrapping pip from system pip..." >&2
      "$_SYS_PY" -m pip install --prefix="$_HARBOR_VENV" --no-deps --force-reinstall pip 2>/dev/null || true
    fi
  fi
  if ! "$_HARBOR_PY" -m pip --version 2>/dev/null; then
    echo "[runtime] Bootstrapping pip via get-pip.py..." >&2
    "$_HARBOR_PY" -c "
import urllib.request, socket
socket.setdefaulttimeout(15)
import sys
url = 'https://bootstrap.pypa.io/' + ('pip/%d.%d/' % sys.version_info[:2] if sys.version_info[:2] < (3, 10) else '') + 'get-pip.py'
urllib.request.urlretrieve(url, '/tmp/get-pip.py')
" \
      && "$_HARBOR_PY" /tmp/get-pip.py \
      || { echo "[runtime] FATAL: cannot bootstrap pip" >&2; exit 1; }
  fi
fi

if ! "$_HARBOR_PY" -c "import uvicorn; import fastapi" 2>/dev/null; then
  echo "[runtime] Installing server dependencies (uvicorn/fastapi)..." >&2
  "$_HARBOR_PY" -m pip install --upgrade pip 2>/dev/null || true
  "$_HARBOR_PY" -m pip install uvicorn fastapi 2>/dev/null \
    || { echo "[runtime] WARNING: failed to install uvicorn/fastapi, server may fail" >&2; }
fi

export HARBOR_PYTHON="$_HARBOR_PY"

# --- Install tmux & asciinema (for terminal-based agents) ---
export TMUX_TMPDIR="${TMUX_TMPDIR:-/tmp/.runtime-tmux}"
mkdir -p "$TMUX_TMPDIR"

# TMUX_TMPDIR is inherited by tmux; preserve the image shell startup files.

if ! command -v tmux >/dev/null 2>&1; then
  echo "[runtime] Installing tmux..." >&2
  if command -v apt-get >/dev/null 2>&1; then
    apt-get install -y -qq tmux 2>/dev/null || true
  elif command -v dnf >/dev/null 2>&1; then dnf install -y tmux 2>/dev/null || true
  elif command -v yum >/dev/null 2>&1; then yum install -y tmux 2>/dev/null || true
  elif command -v apk >/dev/null 2>&1; then apk add --no-cache tmux 2>/dev/null || true
  fi
fi
if ! command -v asciinema >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get install -y -qq asciinema 2>/dev/null || true
  elif command -v pip3 >/dev/null 2>&1; then
    pip3 install --break-system-packages asciinema 2>/dev/null || pip3 install asciinema 2>/dev/null || true
  fi
fi

# --- Run task-specific setup (sourced so it can export/override HARBOR_PYTHON) ---
if [ -f "$HARBOR_STAGING/setup.sh" ]; then
  echo "[runtime] Running task setup.sh..." >&2
  source "$HARBOR_STAGING/setup.sh"
fi

# Re-verify after setup.sh (the /opt/harbor-server venv should be untouched,
# but check anyway in case something unusual happened)
if ! "$HARBOR_PYTHON" -c "import uvicorn; import fastapi" 2>/dev/null; then
  echo "[runtime] WARNING: uvicorn/fastapi lost after setup.sh, server may fail" >&2
fi

# PyStan 3.10.0 imports pkg_resources, removed in setuptools 82.
printf 'setuptools==80.9.0\n' > /opt/pystan-build-constraints.txt
export PIP_CONSTRAINT=/opt/pystan-build-constraints.txt
# Preserve stanc version; use its official native ARM64 release asset.
cat > /usr/local/bin/curl <<'CURLWRAP'
#!/bin/bash
args=()
for arg in "$@"; do
  if [[ "$arg" == https://github.com/stan-dev/stanc3/releases/download/v2.35.0/linux-stanc ]]; then
    arg=https://github.com/stan-dev/stanc3/releases/download/v2.35.0/linux-arm64-stanc
  fi
  args+=("$arg")
done
exec /usr/bin/curl "${args[@]}"
CURLWRAP
chmod 755 /usr/local/bin/curl
exec "$HARBOR_PYTHON" "$@"
