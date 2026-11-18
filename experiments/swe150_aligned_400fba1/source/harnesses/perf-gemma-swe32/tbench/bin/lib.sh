# perf-gemma-swe32 tbench half: settings and helpers shared by every script (sourced, never run).
# The deployed root is the directory above bin/ (/home/ubuntu/gswe32, next to the v12 pilot's half, which is never
# touched, and the other halves /home/ubuntu/tb16, swe32, swe16, swe10, which are never written by this half except
# through their own lock protocol: lockset.py takes their tbench locks with byte-identical copies of their lock
# modules). gswe32.env (written by deploy_tbench.ps1) names the agent env, the evaluator env, the host python and the
# shared lock roots (GSWE32_SHARED_LOCKS).
GSWE32_BIN=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GSWE32_ROOT=${GSWE32_ROOT:-$(dirname "$GSWE32_BIN")}
if [[ -f $GSWE32_ROOT/gswe32.env ]]; then
  # shellcheck disable=SC1091
  . "$GSWE32_ROOT/gswe32.env"
fi
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3}
MSWEA_CONFIG_DIR=${MSWEA_CONFIG_DIR:-$GSWE32_ROOT/mswea-config}
export GSWE32_SHARED_LOCKS
RUNS=$GSWE32_ROOT/runs
SOCKS=$GSWE32_ROOT/sock
UPSTREAM=$GSWE32_ROOT/upstream
PACKAGES=$GSWE32_ROOT/packages
RUN_ID_RE='^[A-Za-z0-9_.-]{1,80}$'
LABEL_RE='^[a-z][a-z0-9-]{0,15}$'
INSTANCE_RE='^[A-Za-z0-9_-]{1,120}$'

hpy() { "$HOST_PYTHON" -I -B "$@"; }
# the shared tbench locks (lockset.py: gswe32's own lock and every other half's, all or nothing); never one lock alone
lock() { hpy "$GSWE32_BIN/lockset.py" "$GSWE32_ROOT" "$@"; }
image_of() {   # image_of <instance id>: the SWE-bench Verified evaluation image (README 7.2)
  local x="docker.io/swebench/sweb.eval.x86_64.${1//__/_1776_}:latest"
  echo "${x,,}"
}
json_line() {  # json_line <event> <extra JSON object>: one final JSON line
  hpy -c 'import json,sys; v=json.loads(sys.argv[2]); v["event"]=sys.argv[1]; print(json.dumps(v, sort_keys=True))' "$1" "$2"
}
