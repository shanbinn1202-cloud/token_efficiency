#!/usr/bin/env bash
# Stop one tb16 Harbor run (perf-tb16): Harbor first (SIGTERM to its process: Harbor cancels its trials and takes their
# docker compose projects down, "up to a minute"), then after the grace its whole process group (SIGKILL), the run's
# relay and run_harbor.sh itself, then only the containers and networks of THIS run's trials (their compose projects;
# never anyone else's), and it releases the shared locks. Idempotent. Called over SSH by the GPU job at the harbor cap,
# on a tunnel drop or on an error, and by the run's own watchdog at its remote deadline. The final line reports every
# planned task as Harbor left it (done / cancelled / missing).
# Usage: stop_harbor.sh <run_id> <label> [watchdog]      (TB16_STOP_GRACE: seconds Harbor gets, default 120)
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUN=${1:-}; LABEL=${2:-}; WHO=${3:-job}
if ! [[ $RUN =~ $RUN_ID_RE && $LABEL =~ $LABEL_RE ]]; then echo "TB16_ERROR bad arguments" >&2; exit 2; fi
OUT=$RUNS/$RUN/$LABEL
SOCK=$SOCKS/$RUN-$LABEL.sock
if [[ ! -d $OUT ]]; then
  json_line tb16_harbor_stopped "{\"run_id\": \"$RUN\", \"label\": \"$LABEL\", \"by\": \"$WHO\", \"run_found\": false, \"signalled\": 0, \"removed\": 0}"
  exit 0
fi
touch "$OUT/.stopping"
grace=${TB16_STOP_GRACE:-120}
signalled=0
harbor=$( [[ -f $OUT/harbor.pid ]] && cat "$OUT/harbor.pid" )
if [[ -n $harbor ]] && kill -0 "$harbor" 2>/dev/null; then
  kill -TERM "$harbor" 2>/dev/null && signalled=$((signalled + 1))
  for _ in $(seq 1 "$grace"); do kill -0 "$harbor" 2>/dev/null || break; sleep 1; done
  # anything of Harbor's group still alive (a hung compose call) goes now
  kill -KILL -- "-$harbor" 2>/dev/null
  kill -KILL "$harbor" 2>/dev/null
fi
pids=()
[[ -f $OUT/relay.pid ]] && pids+=("$(cat "$OUT/relay.pid")")
[[ -f $OUT/pool.pid ]] && pids+=("$(cat "$OUT/pool.pid")")
if command -v pgrep > /dev/null 2>&1; then
  for p in $(pgrep -f -- "run -c $OUT/job.json" 2>/dev/null) $(pgrep -f -- "relay.py $SOCK " 2>/dev/null); do pids+=("$p"); done
fi
for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null && signalled=$((signalled + 1)); done
sleep 2
for pid in "${pids[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
# This run's trials' compose projects (Harbor: <trial>__env, <trial>__verifier__<key>, sanitized): their containers and
# networks, if Harbor could not remove them.
removed=$(hpy - "$OUT" <<'PY'
import json, os, re, subprocess, sys
from pathlib import Path
out = Path(sys.argv[1])
spec = json.loads((out / 'spec.json').read_text()) if (out / 'spec.json').is_file() else {}
job = out / 'jobs' / str(spec.get('job_name') or '')
def sanitize(name):
    name = name.lower()
    if not re.match(r'^[a-z0-9]', name):
        name = '0' + name
    return re.sub(r'[^a-z0-9_-]', '-', name)
prefixes = [sanitize(p.name) + '__' for p in job.iterdir() if p.is_dir()] if job.is_dir() else []
client = os.environ.get('TB16_DOCKER', 'docker')
base = [os.environ.get('TB16_BASH', 'bash'), client] if os.name == 'nt' and not client.endswith('.exe') else [client]
def docker(*args):
    try:
        return subprocess.run(base + list(args), capture_output=True, text=True, timeout=120)
    except Exception:
        return None
removed = 0
for kind in ('container', 'network'):
    listing = docker(*(['ps', '-a'] if kind == 'container' else ['network', 'ls']),
                     '--format', '{{.ID}}\t{{.Label "com.docker.compose.project"}}')
    for line in (listing.stdout if listing and listing.returncode == 0 else '').splitlines():
        ident, _, project = line.partition('\t')
        if ident and project and any(project.startswith(p) for p in prefixes):
            done = docker(*(['rm', '-f', ident] if kind == 'container' else ['network', 'rm', ident]))
            removed += 1 if done is not None and done.returncode == 0 else 0
print(removed)
PY
)
touch "$OUT/.sampler-stop" "$OUT/.done"
locks release pool "$RUN" > "$OUT/lock-release-$WHO.json"
hpy "$TB16_BIN/harbor_report.py" tb16_harbor_stopped "$OUT" \
  "{\"run_id\": \"$RUN\", \"label\": \"$LABEL\", \"by\": \"$WHO\", \"run_found\": true, \"signalled\": $signalled, \"removed\": ${removed:-0}}"
