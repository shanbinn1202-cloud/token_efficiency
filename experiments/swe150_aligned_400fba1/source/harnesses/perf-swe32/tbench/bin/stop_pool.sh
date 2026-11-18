#!/usr/bin/env bash
# Stop one swe32 pool (perf-swe32): its pool script, its runners, its relay and only the containers it started.
# Idempotent. Called over SSH by the GPU job at the agent cap, on a tunnel drop or on an error, and by the pool's own
# watchdog at its remote deadline. It reports which instances had finished (trajectory written) before the stop.
# Usage: stop_pool.sh <run_id> <label> [watchdog]
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUN=${1:-}; LABEL=${2:-}; WHO=${3:-job}
if ! [[ $RUN =~ $RUN_ID_RE && $LABEL =~ $LABEL_RE ]]; then echo "SWE32_ERROR bad arguments" >&2; exit 2; fi
OUT=$RUNS/$RUN/$LABEL
SOCK=$SOCKS/$RUN-$LABEL.sock
if [[ ! -d $OUT ]]; then
  json_line swe32_pool_stopped "{\"run_id\": \"$RUN\", \"label\": \"$LABEL\", \"by\": \"$WHO\", \"pool_found\": false, \"processes_signalled\": 0, \"containers_removed\": 0, \"instances\": {}}"
  exit 0
fi
touch "$OUT/.stopping"
pids=()
for f in "$OUT"/pids/*.pid "$OUT/relay.pid"; do [[ -f $f ]] && pids+=("$(cat "$f")"); done
if [[ -f $OUT/pool.pid ]]; then pids+=("$(cat "$OUT/pool.pid")"); fi
if command -v pgrep > /dev/null 2>&1; then
  for p in $(pgrep -f -- "--output $OUT/runs/" 2>/dev/null) $(pgrep -f -- "relay.py $SOCK " 2>/dev/null); do pids+=("$p"); done
fi
killed=0
for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null && killed=$((killed + 1)); done
sleep 5
for pid in "${pids[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
# Containers this pool started, as Mini-SWE-Agent logs them ("Started container minisweagent-xxxxxxxx").
names=$(grep -rhoE 'Started container minisweagent-[0-9a-f]{8}' "$OUT/logs" "$OUT/runs" 2>/dev/null | awk '{print $3}' | sort -u)
removed=0
for name in $names; do
  if docker ps -q --filter "name=^${name}$" | grep -q .; then
    docker rm -f "$name" > /dev/null 2>&1 && removed=$((removed + 1))
  fi
done
touch "$OUT/.sampler-stop" "$OUT/.done"
if [[ ! -f $OUT/preds.json && -n ${MINI_PYTHON:-} ]]; then
  "$MINI_PYTHON" - "$OUT" <<'PY' || true
import json, sys
from pathlib import Path
out, merged = Path(sys.argv[1]), {}
for preds in sorted((out / 'runs').glob('*/preds.json')):
    merged.update(json.loads(preds.read_text()))
(out / 'preds.json').write_text(json.dumps(merged, indent=2))
PY
fi
lock release pool "$RUN" > "$OUT/lock-release-$WHO.json"
hpy "$SWE32_BIN/pool_report.py" swe32_pool_stopped "$OUT" \
  "{\"run_id\": \"$RUN\", \"label\": \"$LABEL\", \"by\": \"$WHO\", \"pool_found\": true, \"processes_signalled\": $killed, \"containers_removed\": $removed}"
