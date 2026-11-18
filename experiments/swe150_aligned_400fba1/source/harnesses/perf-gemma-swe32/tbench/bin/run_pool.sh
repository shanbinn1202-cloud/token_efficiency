#!/usr/bin/env bash
# The agent pool of one gswe32 server (perf-gemma-swe32), started over SSH by the GPU job once its reverse tunnel is up.
# Mini-SWE-Agent 2.4.2 runs one runner (--workers 1) per instance, at most K at once, refilled in list order as each
# ends; when an instance's runner ends its session is closed (POST /v1/sessions/close, 409 retried; skipped for the
# stock control, which has no close route) and the next instance starts (the runtime's examples/swebench/run_agents.sh
# pool mode, with the shared tbench locks, pid files for stop_pool.sh, a remote deadline, a host sampler and a final
# line).
#
# Usage: run_pool.sh <run_id> <label> <runtime|stock> <port> <socket|-> <spec> <instance_id>...
#   <socket>  the job's reverse tunnel ends in this Unix socket (sock/<run_id>-<label>.sock); relay.py exposes it at
#             127.0.0.1:<port> and `ss` must show nothing else listening there. `-` (tests only, GSWE32_ALLOW_TCP=1):
#             something already serves 127.0.0.1:<port>.
#   <spec>    base64url JSON from the job (controller.pool_spec): K, step limit, remote deadline (epoch s), the pinned
#             runner/prompt SHA-256, mini's retry attempts, litellm's REQUEST_TIMEOUT, close or not, and the whole
#             model_override.yaml rendered from the sealed config.
# Output: runs/<run_id>/<label>/ (never reused): spec.json, model_override.yaml, instances.txt, runs/<id>/ (runner
# output: <id>/<id>.traj.json, preds.json), logs/<id>.log, closes/<id>.json, timeline.jsonl, host-samples.jsonl,
# preds.json; final JSON line {"event": "gswe32_pool_done", ...} (or gswe32_pool_refused, exit 4: one of the shared
# tbench locks is held by a live owner, or the host check refused; exit 2: the locks could not be read).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUN=${1:-}; LABEL=${2:-}; KIND=${3:-}; PORT=${4:-}; SOCK=${5:-}; SPEC=${6:-}
refuse() { json_line gswe32_pool_refused "$(hpy -c 'import json,sys; print(json.dumps(dict(run_id=sys.argv[1], label=sys.argv[2], reason=sys.argv[3])))' "$RUN" "$LABEL" "$1")"; exit "${2:-2}"; }
if (( $# < 7 )) || ! [[ $RUN =~ $RUN_ID_RE && $LABEL =~ $LABEL_RE && $KIND =~ ^(runtime|stock)$ && $PORT =~ ^[0-9]{4,5}$ \
        && $SPEC =~ ^[A-Za-z0-9_=-]+$ ]]; then
  refuse "bad arguments"
fi
if [[ $SOCK == - ]]; then
  [[ ${GSWE32_ALLOW_TCP:-0} == 1 ]] || refuse "socket '-' is for tests (GSWE32_ALLOW_TCP=1)"
elif [[ $SOCK != "$SOCKS/$RUN-$LABEL.sock" ]]; then
  refuse "socket path is not $SOCKS/$RUN-$LABEL.sock"
fi
shift 6
INSTANCES=("$@")
for i in "${INSTANCES[@]}"; do [[ $i =~ $INSTANCE_RE ]] || refuse "bad instance id: $i"; done
[[ $(printf '%s\n' "${INSTANCES[@]}" | sort -u | wc -l) == "${#INSTANCES[@]}" ]] || refuse "repeated instance"
: "${MINI_PYTHON:?MINI_PYTHON unset (gswe32.env)}"
OUT=$RUNS/$RUN/$LABEL
mkdir -p "$RUNS/$RUN" || refuse "cannot create $RUNS/$RUN"
mkdir "$OUT" || refuse "output exists (never reused): $OUT"
# The spec: decoded, checked, the model override written as the job rendered it.
hpy - "$SPEC" "$OUT" "$LABEL" "$KIND" <<'PY' || refuse "spec rejected"
import base64, hashlib, json, re, sys
from pathlib import Path
text, out, label, kind = sys.argv[1:]
spec = json.loads(base64.urlsafe_b64decode(text + '=' * (-len(text) % 4)))
assert spec.get('schema') == 'gswe32-pool-spec-v1' and spec['label'] == label and spec['kind'] == kind
assert 1 <= spec['max_running'] <= 16 and 1 <= spec['step_limit'] <= 256 and spec['deadline_epoch'] > 0
assert len(spec['runner_sha256']) == 64 and len(spec['prompt_sha256']) == 64 and spec['mini_retry_attempts'] >= 1
assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,80}', spec['prompt_file']) and spec['prompt_file'] != 'SHA256SUMS'
assert 'step_limit: %d' % spec['step_limit'] in spec['model_override']
Path(out, 'spec.json').write_text(json.dumps(spec, sort_keys=True, indent=1))
Path(out, 'model_override.yaml').write_bytes(spec['model_override'].encode())
values = {k.upper(): spec[k] for k in ('max_running', 'step_limit', 'deadline_epoch', 'runner_sha256', 'prompt_sha256',
                                       'prompt_file', 'mini_retry_attempts', 'request_timeout')}
values.update(CLOSE=int(bool(spec['close'])), OVERRIDE_SHA256=hashlib.sha256(spec['model_override'].encode()).hexdigest())
Path(out, 'spec.env').write_bytes(''.join('%s=%s\n' % kv for kv in sorted(values.items())).encode())
PY
# shellcheck disable=SC1091
. "$OUT/spec.env"
# The pinned runner and prompt (config.agent: the PoC runner and the one prompt entry), and mini-swe-agent 2.4.2.
RUNNER_FILE=${RUNNER_FILE:-swebench_runner.py}
(cd "$UPSTREAM" && printf '%s  %s\n%s  %s\n' "$RUNNER_SHA256" "$RUNNER_FILE" "$PROMPT_SHA256" "$PROMPT_FILE" \
  | sha256sum --check --quiet) || refuse "upstream runner/prompt differ from the pins"
version=$("$MINI_PYTHON" -c 'from importlib.metadata import version; print(version("mini-swe-agent"))' 2>/dev/null)
[[ $version == 2.4.2 ]] || refuse "mini-swe-agent is ${version:-absent}, need 2.4.2"
# The shared tbench locks (lockset.py: this half's and every other half's, all or nothing): never beside another
# gswe32 run, a TB run or a SWE pool, stress test or evaluation.
held=$(lock acquire pool "$RUN" $$); rc=$?
if (( rc != 0 )); then
  echo "$held" > "$OUT/lock-refused.json"
  (( rc == 3 )) && refuse "tbench lock held: $held" 4
  refuse "tbench locks unreadable: $held" 2
fi
echo "$held" > "$OUT/lock.json"
# Light host check (host_check.py strict): no leftover container of our previous runs, MemAvailable and free disk
# within gswe32.env's limits (other containers and the load are reported only); refused before any agent starts.
if ! "$HOST_PYTHON" -I -B "$GSWE32_BIN/host_check.py" "$GSWE32_ROOT" strict > "$OUT/host-check.json" 2> "$OUT/host-check.err"; then
  lock release pool "$RUN" > /dev/null
  refuse "tbench host busy: $(hpy -c 'import json,sys; print("; ".join(json.load(open(sys.argv[1])).get("reasons") or ["unreadable"]))' "$OUT/host-check.json" 2>/dev/null)" 4
fi
echo $$ > "$OUT/pool.pid"
RELAY=""; SAMPLER=""; WATCHDOG=""
cleanup() {
  touch "$OUT/.done" "$OUT/.sampler-stop" 2>/dev/null
  [[ -n $RELAY ]] && { kill "$RELAY" 2>/dev/null; wait "$RELAY" 2>/dev/null; }
  [[ -n $WATCHDOG ]] && { kill "$WATCHDOG" 2>/dev/null; wait "$WATCHDOG" 2>/dev/null; }
  [[ -n $SAMPLER ]] && { wait "$SAMPLER" 2>/dev/null; }
  lock release pool "$RUN" > /dev/null
}
trap cleanup EXIT
printf '%s\n' "${INSTANCES[@]}" > "$OUT/instances.txt"
hpy - "$OUT/pool.json" "$RUN" "$LABEL" "$KIND" "$PORT" "$SOCK" "${INSTANCES[@]}" <<'PY'
import json, sys, time
path, run, label, kind, port, sock, *instances = sys.argv[1:]
with open(path, 'x') as stream:
    json.dump(dict(run_id=run, label=label, kind=kind, port=int(port), socket=sock, started=time.time(),
                   instances=instances), stream, sort_keys=True)
PY
if [[ $SOCK != - ]]; then
  "$HOST_PYTHON" -I -B "$GSWE32_BIN/relay.py" "$SOCK" "$PORT" > "$OUT/relay.log" 2>&1 &
  RELAY=$!; echo "$RELAY" > "$OUT/relay.pid"
fi
ready=0
for _ in $(seq 1 60); do
  if curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" -o "$OUT/models.json"; then ready=1; break; fi
  [[ -z $RELAY ]] || kill -0 "$RELAY" 2>/dev/null || break
  sleep 5
done
[[ $ready == 1 ]] || refuse "endpoint 127.0.0.1:$PORT not reachable" 3
if [[ $SOCK != - ]]; then
  listeners=$(ss -ltnH "sport = :$PORT" | awk '{print $4}' | sort -u | tr '\n' ' ')
  [[ -n $listeners ]] || refuse "no listener on $PORT" 3
  for addr in $listeners; do [[ $addr == "127.0.0.1:$PORT" ]] || refuse "port $PORT also listens on $addr" 3; done
fi
hpy "$GSWE32_BIN/host_sampler.py" "$OUT/host-samples.jsonl" 5 "$OUT/.sampler-stop" $$ > "$OUT/sampler.log" 2>&1 &
SAMPLER=$!
# Remote deadline: if the job could not stop this pool (a dropped connection, a killed job), it stops itself.
( while sleep 10; do
    [[ -f $OUT/.done ]] && exit 0
    if (( $(date +%s) >= DEADLINE_EPOCH )); then
      echo "deadline $DEADLINE_EPOCH reached at $(date +%s)" > "$OUT/watchdog.txt"
      "$GSWE32_BIN/stop_pool.sh" "$RUN" "$LABEL" watchdog > "$OUT/watchdog-stop.log" 2>&1
      exit 0
    fi
  done ) &
WATCHDOG=$!
mkdir "$OUT/runs" "$OUT/logs" "$OUT/closes" "$OUT/pids"
runner() {   # runner <instance>: the pinned PoC runner and config stack, one worker, no client retry
  # exec: the background subshell becomes the runner itself, so its pid file names the process stop_pool.sh must
  # kill (killing only a subshell left the runner alive and sending requests after its session was closed).
  exec env -u MSWEA_CTX_PATCH AGENTDIET_MODE=off HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    MSWEA_GLOBAL_CONFIG_DIR="$MSWEA_CONFIG_DIR" PYTHONDONTWRITEBYTECODE=1 \
    MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT="$MINI_RETRY_ATTEMPTS" REQUEST_TIMEOUT="$REQUEST_TIMEOUT" \
    "$MINI_PYTHON" "$UPSTREAM/$RUNNER_FILE" \
      --subset verified --split test --workers 1 --filter "^$1\$" --output "$OUT/runs/$1" \
      -c swebench_xml.yaml -c "$UPSTREAM/$PROMPT_FILE" -c "$OUT/model_override.yaml" \
      > "$OUT/logs/$1.log" 2>&1
}
close_session() {   # close_session <instance>: retried while its last request is still being torn down (409)
  local code=000 n=0
  for _ in $(seq 1 30); do
    n=$((n + 1))
    code=$(curl -s -o "$OUT/closes/$1.json" -w '%{http_code}' --max-time 120 -X POST \
           -H 'Content-Type: application/json' -d "{\"instance_id\": \"$1\"}" \
           "http://127.0.0.1:$PORT/v1/sessions/close")
    [[ $code == 409 ]] || break
    sleep 2
  done
  echo "$code $n"
}
agent() {   # agent <instance>: its own runner, then its session closes and frees the slot
  local t0 t1 rc code=0 attempts=0 out
  t0=$(date +%s.%N)
  runner "$1" &
  echo $! > "$OUT/pids/$1.pid"
  wait $!
  rc=$?
  t1=$(date +%s.%N)
  # stopped: the runner ended while the pool was being stopped (killed, or racing the stop): its trajectory, if
  # Mini-SWE-Agent left one, is not a finished instance (conservative: such an instance counts as truncated)
  local stopped=false
  [[ -f $OUT/.stopping ]] && stopped=true
  rm -f "$OUT/pids/$1.pid"
  if [[ $CLOSE == 1 ]]; then out=$(close_session "$1"); code=${out%% *}; attempts=${out##* }; fi
  printf '{"instance_id": "%s", "start": %s, "end": %s, "runner_exit": %d, "stopped": %s, "close_http": %d, "close_attempts": %d}\n' \
    "$1" "$t0" "$t1" "$rc" "$stopped" "$((10#${code:-0}))" "$attempts" >> "$OUT/timeline.jsonl"
  [[ $rc == 0 && ( $CLOSE == 0 || $code == 200 ) ]]
}
start=$(date +%s)
agents=() failed=0
running() { local n=0 p; for p in "${agents[@]}"; do kill -0 "$p" 2>/dev/null && n=$((n + 1)); done; echo $n; }
for i in "${INSTANCES[@]}"; do
  while (( $(running) >= MAX_RUNNING )); do sleep 1; done
  [[ -f $OUT/.stopping ]] && break
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
end=$(date +%s)
touch "$OUT/.done" "$OUT/.sampler-stop"
# Mini-SWE-Agent stops its containers asynchronously; the next run must start clean.
# Only this pool's own containers (the names its runners logged), never anyone else's on the shared host.
names=$(grep -rhoE 'Started container minisweagent-[0-9a-f]{8}' "$OUT/logs" "$OUT/runs" 2>/dev/null | awk '{print $3}' | sort -u)
left=0
for _ in $(seq 1 60); do
  left=0
  for name in $names; do left=$(( left + $(docker ps -q --filter "name=^${name}\$" | wc -l) )); done
  (( left == 0 )) && break
  sleep 2
done
pool_exit=$(( failed > 0 ))
hpy "$GSWE32_BIN/pool_report.py" gswe32_pool_done "$OUT" \
  "{\"run_id\": \"$RUN\", \"label\": \"$LABEL\", \"kind\": \"$KIND\", \"pool_exit\": $pool_exit, \"failed_agents\": $failed, \"seconds\": $((end - start)), \"containers_left\": $left, \"model_override_sha256\": \"$OVERRIDE_SHA256\", \"max_running\": $MAX_RUNNING, \"step_limit\": $STEP_LIMIT, \"mini_version\": \"$version\", \"host\": \"$(hostname)\"}"
exit $pool_exit
