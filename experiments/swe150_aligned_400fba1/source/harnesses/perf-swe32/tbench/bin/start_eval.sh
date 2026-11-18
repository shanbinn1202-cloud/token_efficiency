#!/usr/bin/env bash
# Launch the SWE-bench evaluation of one finished job's pools, detached (perf-swe32). Called over SSH by the GPU job
# after it archived its evidence. It takes the tbench lock (kind eval) or refuses (exit 4), starts bin/evaluate_run.sh
# under setsid/nohup (it survives the SSH session), waits until the evaluator owns the lock and prints one line
# {"event": "swe32_eval_started", "pid": ...}. The evaluation writes runs/<run_id>/<label>/eval/ (never overwritten) and
# runs/<run_id>/eval-status.json; it releases the lock when it ends. Never two evaluations, and never one beside a pool.
# Usage: start_eval.sh <run_id> <workers> <timeout s> <overall timeout s> <label>...  (config.evaluation)
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUN=${1:-}; WORKERS=${2:-}; ETIMEOUT=${3:-}; OVERALL=${4:-}
if (( $# < 5 )) || ! [[ $RUN =~ $RUN_ID_RE && $WORKERS =~ ^[1-8]$ && $ETIMEOUT =~ ^[0-9]{2,6}$ && $OVERALL =~ ^[0-9]{3,6}$ ]]; then
  echo "SWE32_ERROR bad arguments" >&2; exit 2
fi
shift 4
LABELS=("$@")
for label in "${LABELS[@]}"; do
  [[ $label =~ $LABEL_RE && -d $RUNS/$RUN/$label ]] || { json_line swe32_eval_refused "{\"run_id\": \"$RUN\", \"reason\": \"no pool output $label\"}"; exit 2; }
  [[ ! -e $RUNS/$RUN/$label/eval ]] || { json_line swe32_eval_refused "{\"run_id\": \"$RUN\", \"reason\": \"evaluation output exists for $label (never overwritten)\"}"; exit 2; }
done
held=$(lock acquire eval "$RUN" $$); rc=$?
if (( rc != 0 )); then
  json_line swe32_eval_refused "{\"run_id\": \"$RUN\", \"reason\": \"tbench lock held\", \"holder\": $held}"
  exit 4
fi
log=$RUNS/$RUN/eval-driver.log
if command -v setsid > /dev/null 2>&1; then
  EVAL_WORKERS=$WORKERS EVAL_TIMEOUT=$ETIMEOUT EVAL_OVERALL_TIMEOUT=$OVERALL     setsid nohup "$SWE32_BIN/evaluate_run.sh" "$RUN" "${LABELS[@]}" > "$log" 2>&1 < /dev/null &
else
  EVAL_WORKERS=$WORKERS EVAL_TIMEOUT=$ETIMEOUT EVAL_OVERALL_TIMEOUT=$OVERALL     nohup "$SWE32_BIN/evaluate_run.sh" "$RUN" "${LABELS[@]}" > "$log" 2>&1 < /dev/null &
fi
child=$!
for _ in $(seq 1 100); do
  owner=$(lock status)
  if hpy -c 'import json,sys; v=json.loads(sys.argv[1]); sys.exit(0 if v.get("pid")==int(sys.argv[2]) and v.get("kind")=="eval" else 1)' "$owner" "$child"; then
    json_line swe32_eval_started "{\"run_id\": \"$RUN\", \"pid\": $child, \"labels\": $(hpy -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${LABELS[@]}"), \"log\": \"$log\"}"
    exit 0
  fi
  kill -0 "$child" 2>/dev/null || break
  sleep 0.2
done
lock release eval "$RUN" > /dev/null
json_line swe32_eval_refused "{\"run_id\": \"$RUN\", \"reason\": \"the evaluator did not take over the lock (see $log)\"}"
exit 3
