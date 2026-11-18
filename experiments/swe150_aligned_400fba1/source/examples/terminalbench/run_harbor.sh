#!/usr/bin/env bash
# Run one Harbor 0.21.0 Terminal-Bench 2.0 job (Terminus 2, unchanged) against one server, on the agent (Docker)
# host. Settings and order are those of the Terminal-Bench trial runs (earlier GPU runs).
#
# Usage: run_harbor.sh <socket|-> <port> <output dir> <profile> <task set> [stock]
#   <socket>    the GPU job's reverse tunnel ends in this Unix socket on this host (the validated form);
#               examples/swebench/relay.py exposes it at 127.0.0.1:<port>.
#   -           something already serves http://127.0.0.1:<port>/v1 on this host, for example the server itself on
#               this host (an x86_64 host with the GPU; not part of the validated runs).
#   <output dir> a new directory (its parent must exist); its name is the Harbor job name.
#   <profile>   qwen3.6-27b or gemma4-31b-it (the server's release); <task set> trial16, smoke32 or all.
#   stock       the server is a stock vLLM control (make_job.py --stock; no chat-text routes, no session close).
#
# Environment: HARBOR_VENV (required), the virtual environment with Harbor 0.21.0, litellm 1.97.0 and openai
# 2.54.0 (docs/benchmarks/terminal-bench.md, "Agent host setup"). TB_CONTEXT (optional): 65536 writes the job at
# 65,536 tokens (make_job.py --context 65536), the context of the earlier dry runs, for a server at
# --max-model-len 65536; unset or empty, the profile's 262,144.
#
# Order: the Harbor install against its pins (the three versions, Terminus 2's four files: upstream/
# terminus-2.SHA256SUMS); the output directory (never reused); make_job.py writes job.json (jobs directory
# <output dir>/jobs); check_images.py (added after the trial runs): every task image already present must have
# the digest tasks.json records (Harbor pulls an absent image by its tag); the loopback relay; the endpoint must
# answer, and only 127.0.0.1 may listen on the port; close_sessions.py check (the server serves this job and has
# no session yet); `harbor run -c job.json -y` (HARBOR_TELEMETRY=0, in its own process group, whose id is in
# harbor.pgid); close_sessions.py close (the sessions Terminus 2 leaves open); the containers of this job's trials
# must be gone (a failing `docker ps` fails this check); report.py.
# Output: <output dir>/ make-job.json, job.json, images.json, models.json, check.json, harbor.log, harbor.pgid,
# jobs/<job name>/ (Harbor's job directory), close.json, report.json, relay.log; the last line is a JSON summary.
# Exit 0 only when Harbor exited 0, the sessions closed, Docker listed no container of the job and report.py
# accepted the job.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOCK=${1:-}; PORT=${2:-}; OUT=${3:-}; PROFILE=${4:-}; TASK_SET=${5:-}; KIND=${6:-runtime}
if (( $# < 5 || $# > 6 )) || ! [[ $PORT =~ ^[0-9]{4,5}$ && -n $SOCK && -n $OUT \
        && $PROFILE =~ ^(qwen3\.6-27b|gemma4-31b-it)$ && $TASK_SET =~ ^(trial16|smoke32|all)$ \
        && $KIND =~ ^(runtime|stock)$ ]] || (( 10#$PORT < 1024 || 10#$PORT > 65535 )); then
  echo "usage: $0 <socket|-> <port> <output dir> <qwen3.6-27b|gemma4-31b-it> <trial16|smoke32|all> [stock]" >&2
  exit 2
fi
: "${HARBOR_VENV:?set HARBOR_VENV}"
[[ $HARBOR_VENV == /* ]] || { echo "HARBOR_VENV must be an absolute path" >&2; exit 2; }
NAME=$(basename "$OUT")
[[ $NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$ ]] || { echo "bad output directory name: $NAME" >&2; exit 2; }
# The Harbor install, as the runs had it.
versions=$("$HARBOR_VENV/bin/python" -I -c 'from importlib.metadata import version as v
print(v("harbor"), v("litellm"), v("openai"))' 2>/dev/null)
[[ $versions == "0.21.0 1.97.0 2.54.0" ]] || {
  echo "harbor litellm openai are '${versions:-absent}', the validated versions are 0.21.0 1.97.0 2.54.0" >&2; exit 2; }
site=$("$HARBOR_VENV/bin/python" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
(cd "$site" && sha256sum --check --quiet "$HERE/upstream/terminus-2.SHA256SUMS") \
  || { echo "Terminus 2's files differ from the pins" >&2; exit 2; }
docker compose version > /dev/null 2>&1 || { echo "docker compose is not available" >&2; exit 2; }
mkdir "$OUT" || { echo "output exists or parent missing: $OUT" >&2; exit 2; }   # never reuse an output
OUT=$(cd "$OUT" && pwd)
stock=(); [[ $KIND == stock ]] && stock=(--stock)
context=(); [[ -n ${TB_CONTEXT:-} ]] && context=(--context "$TB_CONTEXT")
python3 -I -B "$HERE/make_job.py" --profile "$PROFILE" --tasks "$TASK_SET" --port "$PORT" --jobs-dir "$OUT/jobs" \
  --job-name "$NAME" ${stock[@]+"${stock[@]}"} ${context[@]+"${context[@]}"} --out "$OUT/job.json" \
  > "$OUT/make-job.json" || exit 2
python3 -I -B "$HERE/check_images.py" check --tasks "$TASK_SET" > "$OUT/images.json" \
  || { echo "a task image is not the one tasks.json records, or docker failed: $OUT/images.json" >&2; exit 2; }

if [[ $SOCK != - ]]; then
  # The loopback relay to the tunnel's socket; it dies with this script.
  python3 -I -B "$HERE/../swebench/relay.py" "$SOCK" "$PORT" > "$OUT/relay.log" 2>&1 &
  RELAY=$!
  trap 'kill $RELAY 2>/dev/null; wait $RELAY 2>/dev/null' EXIT
fi
URL=http://127.0.0.1:$PORT
ready=0
for _ in $(seq 1 60); do
  if curl -sf --noproxy '*' --max-time 5 "$URL/v1/models" -o "$OUT/models.json"; then ready=1; break; fi
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
python3 -I -B "$HERE/close_sessions.py" check --base-url "$URL" --job "$OUT/job.json" --profile "$PROFILE" \
  > "$OUT/check.json" || { echo "the server does not serve this job: $OUT/check.json" >&2; exit 3; }

group=(); command -v setsid > /dev/null 2>&1 && group=(setsid -w)
start=$(date +%s)
( cd "$OUT" && exec env HARBOR_TELEMETRY=0 PYTHONDONTWRITEBYTECODE=1 ${group[@]+"${group[@]}"} \
    "$HARBOR_VENV/bin/harbor" run -c "$OUT/job.json" -y > "$OUT/harbor.log" 2>&1 < /dev/null ) &
HARBOR=$!
echo "$HARBOR" > "$OUT/harbor.pgid"
wait "$HARBOR"
harbor_rc=$?
end=$(date +%s)

close_rc=0
if [[ $KIND == runtime ]]; then
  python3 -I -B "$HERE/close_sessions.py" close --base-url "$URL" > "$OUT/close.json"
  close_rc=$?
fi
# Harbor removes each trial's containers as the trial ends (environment.delete); the next run must start clean.
# Only the docker compose projects of this job's trials are looked at: <trial>__env and <trial>__verifier__*.
left=0 docker_ok=1
for _ in $(seq 1 30); do
  # A docker error is not "no container left": the check fails (docker's own message goes to stderr).
  if ! projects=$(docker ps -a --format '{{.Label "com.docker.compose.project"}}' | sort -u); then
    echo "cannot verify the Docker cleanup: docker ps failed" >&2; docker_ok=0; break
  fi
  left=0
  for d in "$OUT/jobs/$NAME"/*/; do
    [[ -d $d ]] || continue
    p=$(basename "$d"); p=${p,,}; p=${p//[^a-z0-9_-]/-}
    left=$(( left + $(grep -c -e "^${p}__env\$" -e "^${p}__verifier__" <<< "$projects") ))
  done
  (( left == 0 )) && break
  sleep 2
done
python3 -I -B "$HERE/report.py" --job "$OUT/job.json" > "$OUT/report.json"
report_rc=$?
(( docker_ok == 1 )) || left=unknown
python3 -I -B - "$OUT" "$harbor_rc" "$close_rc" "$left" "$((end - start))" "$KIND" <<'PY'
import json, sys
out, harbor, close, left, seconds, kind = sys.argv[1:]
def read(name):
    try:
        with open('%s/%s' % (out, name), encoding='utf-8') as f:
            return json.loads(f.read().strip().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return {}
report = read('report.json')
print(json.dumps(dict(job=report.get('job'), harbor_exit=int(harbor), seconds=int(seconds), kind=kind,
                      close=read('close.json').get('status') if kind == 'runtime' else None,
                      containers_left=None if left == 'unknown' else int(left), report=report.get('status'),
                      images=read('images.json').get('status'), planned=report.get('planned'),
                      n_exceptions=report.get('n_exceptions'), solved=report.get('solved'),
                      mean_reward=report.get('mean_reward')), sort_keys=True))
PY
(( harbor_rc == 0 && close_rc == 0 && docker_ok == 1 && report_rc == 0 )) && [[ $left == 0 ]]
