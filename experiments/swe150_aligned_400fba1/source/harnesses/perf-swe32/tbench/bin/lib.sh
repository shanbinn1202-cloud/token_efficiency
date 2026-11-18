# perf-swe32 tbench half: settings and helpers shared by every script (sourced, never run).
# The deployed root is the directory above bin/ (/home/ubuntu/swe32, next to the v12 pilot's half, which is never
# touched). swe32.env (written by deploy_tbench.ps1) names the agent env, the evaluator env and the host python.
SWE32_BIN=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SWE32_ROOT=${SWE32_ROOT:-$(dirname "$SWE32_BIN")}
if [[ -f $SWE32_ROOT/swe32.env ]]; then
  # shellcheck disable=SC1091
  . "$SWE32_ROOT/swe32.env"
fi
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3}
MSWEA_CONFIG_DIR=${MSWEA_CONFIG_DIR:-$SWE32_ROOT/mswea-config}
RUNS=$SWE32_ROOT/runs
SOCKS=$SWE32_ROOT/sock
UPSTREAM=$SWE32_ROOT/upstream
PACKAGES=$SWE32_ROOT/packages
RUN_ID_RE='^[A-Za-z0-9_.-]{1,80}$'
LABEL_RE='^[a-z][a-z0-9-]{0,15}$'
INSTANCE_RE='^[A-Za-z0-9_-]{1,120}$'

hpy() { "$HOST_PYTHON" -I -B "$@"; }
lock() { hpy "$SWE32_BIN/swe32_lock.py" "$SWE32_ROOT" "$@"; }
image_of() {   # image_of <instance id>: the SWE-bench Verified evaluation image (README 7.2)
  local x="docker.io/swebench/sweb.eval.x86_64.${1//__/_1776_}:latest"
  echo "${x,,}"
}
json_line() {  # json_line <event> <extra JSON object>: one final JSON line
  hpy -c 'import json,sys; v=json.loads(sys.argv[2]); v["event"]=sys.argv[1]; print(json.dumps(v, sort_keys=True))' "$1" "$2"
}
