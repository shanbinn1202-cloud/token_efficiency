#!/usr/bin/env bash
# The detached SWE-bench evaluation of one job's pools (perf-swe32), started by start_eval.sh: it takes over the tbench
# lock (kind eval), evaluates each label in turn (bin/evaluate_pool.py: swebench 5.0.2 run_evaluation on the pool's
# predictions, an empty patch for an instance without one) into runs/<run_id>/<label>/eval/, writes
# runs/<run_id>/eval-status.json (state running -> done | failed) and releases the lock when it ends.
# Usage: evaluate_run.sh <run_id> <label>...
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUN=${1:-}
shift
LABELS=("$@")
lock handoff eval "$RUN" $$ > /dev/null || exit 3
trap 'lock release eval "$RUN" > /dev/null' EXIT
status=$RUNS/$RUN/eval-status.json
hpy "$SWE32_BIN/evaluate_pool.py" status "$status" running "$RUN" "${LABELS[@]}"
overall=${EVAL_OVERALL_TIMEOUT:-14400}
failed=0
for label in "${LABELS[@]}"; do
  timeout "$overall" "$HOST_PYTHON" -I -B "$SWE32_BIN/evaluate_pool.py" evaluate "$RUNS/$RUN/$label" "$RUN-$label" \
    "${EVAL_PYTHON:?EVAL_PYTHON unset (swe32.env)}" "${EVAL_WORKERS:-2}" "${EVAL_TIMEOUT:-1800}" \
    > "$RUNS/$RUN/eval-$label.log" 2>&1 || failed=$((failed + 1))
done
hpy "$SWE32_BIN/evaluate_pool.py" status "$status" "$([[ $failed == 0 ]] && echo done || echo failed)" "$RUN" "${LABELS[@]}"
