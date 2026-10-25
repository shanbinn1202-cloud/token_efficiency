#!/usr/bin/env bash
# Official SWE-bench evaluation (swebench 5.0.2) of one policy's agent runs, on the agent (Docker) host, as the
# multi-session pilots did after each job (jobs 3299922 and 3300434).
#
# One predictions file holds every planned instance of the given run_agents.sh output directories (their
# instances.txt), with the patch from that directory's preds.json and model_name_or_path = <label>. An instance with
# no prediction (its agent failed or was stopped) is evaluated with an empty patch and reported as missing.
# Everything the evaluator writes (it writes logs/ under its working directory) stays in <eval dir>, which must
# not exist yet.
#
# Usage: evaluate.sh <run_id> <label> <eval dir> <run_agents output dir>...
# Environment: EVAL_PYTHON = python of an environment with swebench 5.0.2 (required); WORKERS (default 2).
set -euo pipefail
RUN_ID=${1:-}; LABEL=${2:-}; OUT=${3:-}
if (( $# < 4 )) || ! [[ $RUN_ID =~ ^[A-Za-z0-9_.-]{1,80}$ && $LABEL =~ ^[A-Za-z0-9_.-]{1,80}$ && -n $OUT ]]; then
  echo "usage: $0 <run_id> <label> <eval dir> <run_agents output dir>..." >&2; exit 2
fi
shift 3
: "${EVAL_PYTHON:?set EVAL_PYTHON}"
WORKERS=${WORKERS:-2}
[[ $WORKERS =~ ^[1-8]$ ]] || { echo "WORKERS must be 1-8" >&2; exit 2; }
GROUPS_IN=()
for d in "$@"; do GROUPS_IN+=("$(cd "$d" && pwd)"); done
mkdir "$OUT" || { echo "evaluation output exists (never overwritten): $OUT" >&2; exit 2; }
cd "$OUT"
"$EVAL_PYTHON" - "$LABEL" "$OUT/predictions.jsonl" "${GROUPS_IN[@]}" <<'PY'
import json, sys
from pathlib import Path
label, target, *groups = sys.argv[1:]
rows = []
for group in map(Path, groups):
    planned = (group / 'instances.txt').read_text().split()
    preds = json.loads((group / 'preds.json').read_text()) if (group / 'preds.json').is_file() else {}
    for instance in planned:
        row = preds.get(instance)
        rows.append(dict(instance_id=instance, model_name_or_path=label,
                         model_patch=(row or {}).get('model_patch') or ''))
        if row is None:
            print(json.dumps(dict(event='missing_prediction', group=str(group), instance_id=instance)))
with open(target, 'x') as out:
    for row in rows:
        out.write(json.dumps(row) + '\n')
PY
set +e
"$EVAL_PYTHON" -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Verified --split test \
  --predictions_path "$OUT/predictions.jsonl" --max_workers "$WORKERS" --timeout 1800 \
  --run_id "$RUN_ID" > "$OUT/eval.log" 2>&1
rc=$?
set -e
"$EVAL_PYTHON" - "$OUT" "$RUN_ID" "$rc" <<'PY'
import json, sys
from pathlib import Path
out, run_id, rc = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
# swebench 5.0.2 layout (as read in the pilots): logs/evaluation/<run_id>/results.json
report = out / 'logs' / 'evaluation' / run_id / 'results.json'
data = json.loads(report.read_text()) if report.is_file() else {}
predictions = [json.loads(l) for l in (out / 'predictions.jsonl').read_text().splitlines() if l]
for row in predictions:
    i = row['instance_id']
    print(json.dumps(dict(instance_id=i, patch_chars=len(row['model_patch']),
                          resolved=i in (data.get('resolved_ids') or []) if data else None,
                          empty_patch=i in (data.get('empty_patch_ids') or []) if data else None,
                          error=i in (data.get('error_ids') or []) if data else None), sort_keys=True))
print(json.dumps(dict(run_id=run_id, evaluator_exit=rc, report=str(report), report_missing=not data,
                      instances=len(predictions), resolved=len(data.get('resolved_ids') or [])), sort_keys=True))
PY
exit $rc
