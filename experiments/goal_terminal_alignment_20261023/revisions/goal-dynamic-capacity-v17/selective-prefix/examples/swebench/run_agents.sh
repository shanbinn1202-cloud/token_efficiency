#!/usr/bin/env bash
# Run Mini-SWE-Agent 2.4.2 on one group of SWE-bench Verified instances against one selective-prefix server, on
# the agent (Docker) host. Settings, config stack and checks are those of the multi-session SWE-bench pilots
# (Slurm jobs 3299922 and 3300434): one runner, one worker thread per instance, all in parallel.
#
# Usage: run_agents.sh <socket|-> <port> <output dir> <instance_id>...
#   <socket>  the GPU job's reverse tunnel ends in this Unix socket on this host (the validated form);
#             relay.py exposes it at 127.0.0.1:<port>.
#   -         something already serves http://127.0.0.1:<port>/v1 on this host, for example a TCP reverse
#             forward (not part of the validated runs).
#   The instances must be exactly the server's --task list (same ids; any order).
#
# Environment (required):
#   MINI_PYTHON       python of an environment with mini-swe-agent 2.4.2 and datasets
#   UPSTREAM_DIR      (optional) directory with swebench_runner.py and swebench_prompt_override.yaml from the
#                     dram-backed-kv PoC (scripts/ at commit a70701e); default: the copies in ./upstream next to
#                     this script. Their SHA-256 must equal the pins below.
#   MSWEA_CONFIG_DIR  an empty directory, used as Mini-SWE-Agent's global config directory
#   MAX_RUNNING       (optional) a session-pool server (serve.py --max-running-sessions K, the same K here):
#                     instead of one runner for the whole group, a pool of K agents works through the instances
#                     in the order given, one runner (one worker) per instance; when an instance's runner ends,
#                     its session is closed (POST /v1/sessions/close) and the next instance starts. Per instance:
#                     <output dir>/runs/<instance>/ (the runner's output), logs/<instance>.log, closes/<instance>.json
#                     and a line in timeline.jsonl (start, end, runner exit, close status); preds.json merges them.
#   STEP_LIMIT        (optional) Mini-SWE-Agent's step limit, 1-256 (default 90, the pilots' value; for example
#                     120). At most 256: the server's per-session request bound.
#   MAX_TOKENS        (optional) the model's max_tokens, 1536 (default, the pilots' value) or 4096; set it to the
#                     server's --max-tokens (the server forces its own bound and ignores this field).
#   With both unset (or at their defaults) the model override is byte for byte the pilots' file.
# The SWE-bench Verified dataset and the instance images must already be cached on this host (the pilots ran
# offline: HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOCK=${1:-}; PORT=${2:-}; OUT=${3:-}
if (( $# < 4 )) || ! [[ $PORT =~ ^[0-9]{4,5}$ && -n $SOCK && -n $OUT ]]; then
  echo "usage: $0 <socket|-> <port> <output dir> <instance_id>..." >&2; exit 2
fi
shift 3
INSTANCES=("$@")
for i in "${INSTANCES[@]}"; do
  [[ $i =~ ^[A-Za-z0-9_-]{1,120}$ ]] || { echo "bad instance id: $i" >&2; exit 2; }
done
[[ $(printf '%s\n' "${INSTANCES[@]}" | sort -u | wc -l) == "${#INSTANCES[@]}" ]] || { echo "repeated instance" >&2; exit 2; }
UPSTREAM_DIR=${UPSTREAM_DIR:-$HERE/upstream}
: "${MINI_PYTHON:?set MINI_PYTHON}" "${MSWEA_CONFIG_DIR:?set MSWEA_CONFIG_DIR}"
MAX_RUNNING=${MAX_RUNNING:-}
if [[ -n $MAX_RUNNING ]] && ! [[ $MAX_RUNNING =~ ^[1-9][0-9]?$ ]] || [[ -n $MAX_RUNNING ]] && (( MAX_RUNNING > 16 )); then
  echo "MAX_RUNNING must be 1-16" >&2; exit 2
fi
STEP_LIMIT=${STEP_LIMIT:-90}; MAX_TOKENS=${MAX_TOKENS:-1536}
# Checked before anything starts; render_override.py fills the model override below.
python3 -I -B "$HERE/render_override.py" "$HERE/model_override.yaml" 10000 "$MAX_TOKENS" "$STEP_LIMIT" > /dev/null || exit 2
version=$("$MINI_PYTHON" -c 'from importlib.metadata import version; print(version("mini-swe-agent"))' 2>/dev/null)
[[ $version == 2.4.2 ]] || { echo "mini-swe-agent is ${version:-absent}, the validated version is 2.4.2" >&2; exit 2; }
# The pinned PoC files (LF bytes), as checked before every pilot run.
(cd "$UPSTREAM_DIR" && sha256sum --check --quiet) <<'EOF' || { echo "PoC files differ from the pins" >&2; exit 2; }
34ca6d9288c72830e61bd00ad8b15790cab4c29515db24b8df30775cfc0880a7  swebench_runner.py
1b6a48666ad528775bd11225fa4d5c002721c25c0cdbc05b99cc15b2de31336a  swebench_prompt_override.yaml
EOF
mkdir "$OUT" || { echo "output exists or parent missing: $OUT" >&2; exit 2; }   # never reuse an output
printf '%s\n' "${INSTANCES[@]}" > "$OUT/instances.txt"   # the planned instances, for evaluate.sh

if [[ $SOCK != - ]]; then
  # The loopback relay to the tunnel's socket; it dies with this script.
  python3 -I -B "$HERE/relay.py" "$SOCK" "$PORT" > "$OUT/relay.log" 2>&1 &
  RELAY=$!
  trap 'kill $RELAY 2>/dev/null; wait $RELAY 2>/dev/null' EXIT
fi
ready=0
for _ in $(seq 1 60); do
  if curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" -o "$OUT/models.json"; then ready=1; break; fi
  [[ $SOCK == - ]] || kill -0 "$RELAY" 2>/dev/null || break
  sleep 5
done
[[ $ready == 1 ]] || { echo "endpoint 127.0.0.1:$PORT not reachable" >&2; exit 3; }
# Only loopback may listen on that port: an exposed endpoint must never serve.
listeners=$(ss -ltnH "sport = :$PORT" | awk '{print $4}' | sort -u | tr '\n' ' ')
[[ -n $listeners ]] || { echo "no listener on $PORT" >&2; exit 3; }
for addr in $listeners; do
  [[ $addr == "127.0.0.1:$PORT" ]] || { echo "port $PORT also listens on $addr" >&2; exit 3; }
done

python3 -I -B "$HERE/render_override.py" "$HERE/model_override.yaml" "$PORT" "$MAX_TOKENS" "$STEP_LIMIT" \
  > "$OUT/model_override.yaml" || exit 2
runner() {   # runner <workers> <filter> <output dir> <log>: the pinned PoC runner and config stack
  env -u MSWEA_CTX_PATCH AGENTDIET_MODE=off HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    MSWEA_GLOBAL_CONFIG_DIR="$MSWEA_CONFIG_DIR" PYTHONDONTWRITEBYTECODE=1 \
    "$MINI_PYTHON" "$UPSTREAM_DIR/swebench_runner.py" \
      --subset verified --split test --workers "$1" --filter "$2" --output "$3" \
      -c swebench_xml.yaml -c "$UPSTREAM_DIR/swebench_prompt_override.yaml" -c "$OUT/model_override.yaml" \
      > "$4" 2>&1
}
close_session() {   # close_session <instance>: retried while its last request is still being torn down (409)
  local code=000
  for _ in $(seq 1 30); do
    code=$(curl -s -o "$OUT/closes/$1.json" -w '%{http_code}' --max-time 120 -X POST \
           -H 'Content-Type: application/json' -d "{\"instance_id\": \"$1\"}" \
           "http://127.0.0.1:$PORT/v1/sessions/close")
    [[ $code == 409 ]] || break
    sleep 2
  done
  echo "$code"
}
agent() {   # agent <instance>: its own runner, then its session closes and frees the slot
  local t0 t1 rc code
  t0=$(date +%s.%N)
  runner 1 "^$1\$" "$OUT/runs/$1" "$OUT/logs/$1.log"
  rc=$?
  t1=$(date +%s.%N)
  code=$(close_session "$1")
  printf '{"instance_id": "%s", "start": %s, "end": %s, "runner_exit": %d, "close_http": %d}\n' \
    "$1" "$t0" "$t1" "$rc" "$((10#${code:-0}))" >> "$OUT/timeline.jsonl"
  [[ $rc == 0 && $code == 200 ]]
}
start=$(date +%s)
if [[ -z $MAX_RUNNING ]]; then
  # One group: one runner, one worker thread per instance, all in parallel (the pilots' form).
  filter="^($(IFS='|'; echo "${INSTANCES[*]}"))\$"
  runner "${#INSTANCES[@]}" "$filter" "$OUT" "$OUT/runner.log"
  rc=$?
else
  # Session pool: at most MAX_RUNNING agents at once, refilled in the order given as each one ends.
  mkdir "$OUT/runs" "$OUT/logs" "$OUT/closes"
  agents=() failed=0
  for i in "${INSTANCES[@]}"; do
    # Running agents: this shell's running jobs except the relay.
    while (( $(jobs -rp | grep -cvx "${RELAY:-none}") >= MAX_RUNNING )); do sleep 1; done
    agent "$i" &
    agents+=($!)
  done
  for p in "${agents[@]}"; do wait "$p" || failed=$((failed + 1)); done
  "$MINI_PYTHON" - "$OUT" "${INSTANCES[@]}" <<'PY'
import json, sys
from pathlib import Path
out, merged = Path(sys.argv[1]), {}
for instance in sys.argv[2:]:
    preds = out / 'runs' / instance / 'preds.json'
    if preds.is_file():
        merged.update(json.loads(preds.read_text()))
(out / 'preds.json').write_text(json.dumps(merged, indent=2))
PY
  rc=$(( failed > 0 ))
fi
end=$(date +%s)
# Mini-SWE-Agent stops its containers asynchronously; the next server's agents must start clean.
left=0
for _ in $(seq 1 60); do
  left=0
  for i in "${INSTANCES[@]}"; do
    image=docker.io/swebench/sweb.eval.x86_64.${i//__/_1776_}:latest
    left=$(( left + $(docker ps -q --filter "ancestor=${image,,}" | wc -l) ))
  done
  (( left == 0 )) && break
  sleep 2
done
echo "{\"runner_exit\": $rc, \"seconds\": $((end - start)), \"containers_left\": $left, \"preds\": \"$OUT/preds.json\"}"
exit $rc
