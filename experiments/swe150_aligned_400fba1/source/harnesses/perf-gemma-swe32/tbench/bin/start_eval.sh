#!/usr/bin/env bash
# Launch the SWE-bench evaluation of one finished job's pools, detached (perf-gemma-swe32). Called over SSH by the GPU job
# after it archived its evidence. It takes the shared tbench locks (lockset.py, kind eval: this half's and every other
# half's) or refuses (exit 4), starts bin/evaluate_run.sh under setsid/nohup (it survives the SSH session), waits until
# the evaluator owns every lock of the set and prints one line {"event": "gswe32_eval_started", "pid": ...}. The
# evaluation writes runs/<run_id>/<label>/eval/ (never overwritten) and runs/<run_id>/eval-status.json; it holds the
# whole set until it ends, then releases it. Never two evaluations, and never one beside a pool, a TB run or another
# half's pool, stress test or evaluation.
# Usage: start_eval.sh <run_id> <workers> <timeout s> <overall timeout s> <label>...  (config.evaluation)
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUN=${1:-}; WORKERS=${2:-}; ETIMEOUT=${3:-}; OVERALL=${4:-}
if (( $# < 5 )) || ! [[ $RUN =~ $RUN_ID_RE && $WORKERS =~ ^[1-8]$ && $ETIMEOUT =~ ^[0-9]{2,6}$ && $OVERALL =~ ^[0-9]{3,6}$ ]]; then
  echo "GSWE32_ERROR bad arguments" >&2; exit 2
fi
shift 4
LABELS=("$@")
for label in "${LABELS[@]}"; do
  [[ $label =~ $LABEL_RE && -d $RUNS/$RUN/$label ]] || { json_line gswe32_eval_refused "{\"run_id\": \"$RUN\", \"reason\": \"no pool output $label\"}"; exit 2; }
  [[ ! -e $RUNS/$RUN/$label/eval ]] || { json_line gswe32_eval_refused "{\"run_id\": \"$RUN\", \"reason\": \"evaluation output exists for $label (never overwritten)\"}"; exit 2; }
done
held=$(lock acquire eval "$RUN" $$); rc=$?
if (( rc != 0 )); then
  reason="tbench lock held"; (( rc == 3 )) || reason="tbench locks unreadable"
  json_line gswe32_eval_refused "{\"run_id\": \"$RUN\", \"reason\": \"$reason\", \"holder\": ${held:-null}}"
  exit $(( rc == 3 ? 4 : 2 ))
fi
log=$RUNS/$RUN/eval-driver.log
if command -v setsid > /dev/null 2>&1; then
  EVAL_WORKERS=$WORKERS EVAL_TIMEOUT=$ETIMEOUT EVAL_OVERALL_TIMEOUT=$OVERALL     setsid nohup "$GSWE32_BIN/evaluate_run.sh" "$RUN" "${LABELS[@]}" > "$log" 2>&1 < /dev/null &
else
  EVAL_WORKERS=$WORKERS EVAL_TIMEOUT=$ETIMEOUT EVAL_OVERALL_TIMEOUT=$OVERALL     nohup "$GSWE32_BIN/evaluate_run.sh" "$RUN" "${LABELS[@]}" > "$log" 2>&1 < /dev/null &
fi
child=$!
for _ in $(seq 1 100); do
  owner=$(lock status)
  # every present lock of the set handed to the evaluator (evaluate_run.sh: lockset.py handoff), not only the first
  if hpy -c 'import json,sys; v=json.loads(sys.argv[1]); n=[x for x in (v.get("locks") or {}).values() if not x.get("absent")]; sys.exit(0 if n and all(x.get("pid")==int(sys.argv[2]) and x.get("kind")=="eval" and x.get("run_id")==sys.argv[3] for x in n) else 1)' "$owner" "$child" "$RUN"; then
    json_line gswe32_eval_started "{\"run_id\": \"$RUN\", \"pid\": $child, \"labels\": $(hpy -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${LABELS[@]}"), \"log\": \"$log\"}"
    exit 0
  fi
  kill -0 "$child" 2>/dev/null || break
  sleep 0.2
done
lock release eval "$RUN" > /dev/null
json_line gswe32_eval_refused "{\"run_id\": \"$RUN\", \"reason\": \"the evaluator did not take over the locks (see $log)\"}"
exit 3
