#!/usr/bin/env bash
# The Harbor run of one tb16 server (perf-tb16), started over SSH by the GPU job once its reverse tunnel is up.
# Harbor 0.21.0's Terminus 2, unchanged, runs the server's task set at -n 16 against the job's model server through a
# loopback relay; Harbor runs each trial's verifier itself. Order: the spec (the job config rendered on DeltaAI from the
# sealed config) is checked; Harbor's install and the task packages are checked against their pins (harbor_check.py);
# the shared tbench locks (lockset.py: this half's and every SWE half's) are taken or the run is refused (exit 4); the
# light host check (exit 4 when busy); the relay 127.0.0.1:<port> -> the tunnel's Unix socket (`ss` must show nothing
# else listening there), the host sampler, the remote deadline watchdog; then `harbor run -c job.json -y` in its own
# process group (HARBOR_TELEMETRY=0, no MSWEA_CTX_* variable, no bytecode written); at the end the containers of our
# trials must be gone and the final line reports every planned task.
#
# Usage: run_harbor.sh <run_id> <label> <runtime|stock> <port> <socket|-> <spec>
#   <socket>  sock/<run_id>-<label>.sock; `-` (tests only, TB16_ALLOW_TCP=1): something already serves 127.0.0.1:<port>.
#   <spec>    base64url of zlib-compressed JSON from the job (controller.harbor_spec): K, the remote deadline (epoch s), the job name, the
#             task names with their package pins, the Harbor pins, the session close mode and the whole job config.
# Output: runs/<run_id>/<label>/ (never reused): spec.json, job.json, harbor-check.json, lock.json, host-check.json,
# harbor.log, jobs/<job name>/ (Harbor's job directory: config.json, lock.json, result.json, job.log, one directory per
# trial with result.json, agent/trajectory.json, verifier/, trial.log), host-samples.jsonl; final JSON line
# {"event": "tb16_harbor_done", ...} (or tb16_harbor_refused: exit 4 tbench busy, 2 / 3 a refusal before Harbor).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUN=${1:-}; LABEL=${2:-}; KIND=${3:-}; PORT=${4:-}; SOCK=${5:-}; SPEC=${6:-}
refuse() { json_line tb16_harbor_refused "$(hpy -c 'import json,sys; print(json.dumps(dict(run_id=sys.argv[1], label=sys.argv[2], reason=sys.argv[3])))' "$RUN" "$LABEL" "$1")"; exit "${2:-2}"; }
if (( $# != 6 )) || ! [[ $RUN =~ $RUN_ID_RE && $LABEL =~ $LABEL_RE && $KIND =~ ^(runtime|stock)$ && $PORT =~ ^[0-9]{4,5}$ \
        && $SPEC =~ ^[A-Za-z0-9_=-]+$ ]]; then
  refuse "bad arguments"
fi
if [[ $SOCK == - ]]; then
  [[ ${TB16_ALLOW_TCP:-0} == 1 ]] || refuse "socket '-' is for tests (TB16_ALLOW_TCP=1)"
elif [[ $SOCK != "$SOCKS/$RUN-$LABEL.sock" ]]; then
  refuse "socket path is not $SOCKS/$RUN-$LABEL.sock"
fi
: "${HARBOR_BIN:?HARBOR_BIN unset (tb16.env)}" "${HARBOR_ENV:?HARBOR_ENV unset (tb16.env)}"
: "${TB16_TASK_CACHE:?TB16_TASK_CACHE unset (tb16.env)}"
OUT=$RUNS/$RUN/$LABEL
mkdir -p "$RUNS/$RUN" || refuse "cannot create $RUNS/$RUN"
mkdir "$OUT" || refuse "output exists (never reused): $OUT"
# The spec: decoded and checked against this run (labels, port, the job directory, the task filter); the job config
# is written exactly as the job rendered it.
hpy - "$SPEC" "$OUT" "$RUN" "$LABEL" "$KIND" "$PORT" <<'PY' || refuse "spec rejected"
import base64, hashlib, json, re, sys, zlib
from pathlib import Path
text, out, run, label, kind, port = sys.argv[1:]
spec = json.loads(zlib.decompress(base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))))
assert spec.get('schema') == 'tb16-harbor-spec-v1' and spec['run_id'] == run and spec['label'] == label
assert spec['kind'] == kind and 1 <= spec['n_concurrent'] <= 16 and spec['deadline_epoch'] > 0
# session_close: `none` only (Terminus 2 sends no close; a harness-side close is PENDING_INTERFACE)
assert spec['job_name'] == '%s-%s' % (run, label) and spec['session_close'] == 'none'
names = [t['name'] for t in spec['tasks']]
assert names and len(set(names)) == len(names) and all(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', n) for n in names)
config = spec['job_config']
assert config['job_name'] == spec['job_name'] and Path(config['jobs_dir']).as_posix() == (Path(out) / 'jobs').as_posix()
assert config['n_concurrent_trials'] == spec['n_concurrent'] and config['n_attempts'] == 1
assert config['retry']['max_retries'] == 0 and config['timeout_multiplier'] == 1.0
assert len(config['agents']) == 1 and config['agents'][0]['name'] == 'terminus-2'
assert config['agents'][0]['kwargs']['api_base'] == 'http://127.0.0.1:%s/v1' % port
(dataset,) = config['datasets']
org = dataset['name'].split('/')[0]
assert dataset['task_names'] == ['%s/%s' % (org, n) for n in names]
Path(out, 'spec.json').write_text(json.dumps(spec, sort_keys=True, indent=1))
raw = (json.dumps(config, sort_keys=True, indent=2) + '\n').encode()
Path(out, 'job.json').write_bytes(raw)
Path(out, 'check-pins.json').write_text(json.dumps(dict(harbor=spec['harbor'], tasks=spec['tasks']), sort_keys=True))
values = dict(DEADLINE_EPOCH=spec['deadline_epoch'], N_CONCURRENT=spec['n_concurrent'], JOB_NAME=spec['job_name'],
              JOB_CONFIG_SHA256=hashlib.sha256(raw).hexdigest(), SESSION_CLOSE=spec['session_close'],
              TELEMETRY=spec.get('telemetry', 'off'), PLANNED=len(names))
Path(out, 'spec.env').write_bytes(''.join('%s=%s\n' % kv for kv in sorted(values.items())).encode())
PY
# shellcheck disable=SC1091
. "$OUT/spec.env"
# Harbor's install and the task packages against their pins (read-only; the same check as the preflight's).
if ! "$HOST_PYTHON" -I -B "$TB16_BIN/harbor_check.py" "$HARBOR_ENV" "$TB16_TASK_CACHE" "$OUT/check-pins.json" \
    > "$OUT/harbor-check.json" 2> "$OUT/harbor-check.err"; then
  refuse "the Harbor install or the task packages differ from the pins: $(hpy -c 'import json,sys; print("; ".join(json.load(open(sys.argv[1])).get("failures") or ["unreadable"])[:600])' "$OUT/harbor-check.json" 2>/dev/null)" 2
fi
# The shared tbench locks: never beside another TB run, a SWE pool, stress test or evaluation.
held=$(locks acquire pool "$RUN" $$); rc=$?
if (( rc != 0 )); then
  echo "$held" > "$OUT/lock-refused.json"
  refuse "tbench lock held: $held" 4
fi
echo "$held" > "$OUT/lock.json"
# Light host check (strict): no leftover container of our previous runs, MemAvailable and free disk within tb16.env's
# limits (other containers and the load are reported only); refused before any trial starts.
if ! "$HOST_PYTHON" -I -B "$TB16_BIN/host_check.py" "$TB16_ROOT" strict > "$OUT/host-check.json" 2> "$OUT/host-check.err"; then
  locks release pool "$RUN" > /dev/null
  refuse "tbench host busy: $(hpy -c 'import json,sys; print("; ".join(json.load(open(sys.argv[1])).get("reasons") or ["unreadable"]))' "$OUT/host-check.json" 2>/dev/null)" 4
fi
echo $$ > "$OUT/pool.pid"
RELAY=""; SAMPLER=""; WATCHDOG=""
cleanup() {
  touch "$OUT/.done" "$OUT/.sampler-stop" 2>/dev/null
  [[ -n $RELAY ]] && { kill "$RELAY" 2>/dev/null; wait "$RELAY" 2>/dev/null; }
  [[ -n $WATCHDOG ]] && { kill "$WATCHDOG" 2>/dev/null; wait "$WATCHDOG" 2>/dev/null; }
  [[ -n $SAMPLER ]] && { wait "$SAMPLER" 2>/dev/null; }
  locks release pool "$RUN" > /dev/null
}
trap cleanup EXIT
if [[ $SOCK != - ]]; then
  "$HOST_PYTHON" -I -B "$TB16_BIN/relay.py" "$SOCK" "$PORT" > "$OUT/relay.log" 2>&1 &
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
hpy "$TB16_BIN/host_sampler.py" "$OUT/host-samples.jsonl" 5 "$OUT/.sampler-stop" $$ > "$OUT/sampler.log" 2>&1 &
SAMPLER=$!
# Remote deadline: if the job could not stop this run (a dropped connection, a killed job), it stops itself.
( while sleep 10; do
    [[ -f $OUT/.done ]] && exit 0
    if (( $(date +%s) >= DEADLINE_EPOCH )); then
      echo "deadline $DEADLINE_EPOCH reached at $(date +%s)" > "$OUT/watchdog.txt"
      "$TB16_BIN/stop_harbor.sh" "$RUN" "$LABEL" watchdog > "$OUT/watchdog-stop.log" 2>&1
      exit 0
    fi
  done ) &
WATCHDOG=$!
# Harbor: no MSWEA_CTX_* variable (harbor_check.py refused one already), no telemetry, no bytecode; its own process
# group (setsid), so stop_harbor.sh can end it and nothing else.
unset_ctx=()
for name in $(compgen -e); do [[ $name == MSWEA_CTX* ]] && unset_ctx+=(-u "$name"); done
setsid_cmd=()
command -v setsid > /dev/null 2>&1 && setsid_cmd=(setsid -w)
mkdir "$OUT/jobs"
start=$(date +%s.%N)
( cd "$OUT" && exec env "${unset_ctx[@]}" HARBOR_TELEMETRY=0 PYTHONDONTWRITEBYTECODE=1 "${setsid_cmd[@]}" \
    "$HARBOR_BIN" run -c "$OUT/job.json" -y > "$OUT/harbor.log" 2>&1 < /dev/null ) &
HARBOR=$!
echo "$HARBOR" > "$OUT/harbor.pid"
wait "$HARBOR"
harbor_exit=$?
end=$(date +%s.%N)
touch "$OUT/.done" "$OUT/.sampler-stop"
# Harbor removes each trial's containers as the trial ends; the next run must start clean. Only our trials' compose
# projects are looked at (host_check.py's rule), never anyone else's.
left=0
for _ in $(seq 1 30); do
  left=$("$HOST_PYTHON" -I -B "$TB16_BIN/host_check.py" "$TB16_ROOT" strict 2>/dev/null | hpy -c 'import json,sys; print(json.loads(sys.stdin.read().strip().splitlines()[-1]).get("our_leftovers", 0))' 2>/dev/null || echo 0)
  (( left == 0 )) && break
  sleep 2
done
hpy "$TB16_BIN/harbor_report.py" tb16_harbor_done "$OUT" \
  "{\"run_id\": \"$RUN\", \"label\": \"$LABEL\", \"kind\": \"$KIND\", \"harbor_exit\": $harbor_exit, \"start\": $start, \"end\": $end, \"containers_left\": ${left:-0}, \"job_config_sha256\": \"$JOB_CONFIG_SHA256\", \"n_concurrent\": $N_CONCURRENT, \"telemetry\": \"$TELEMETRY\", \"host\": \"$(hostname)\"}"
(( harbor_exit == 0 )) && exit 0
exit 1
