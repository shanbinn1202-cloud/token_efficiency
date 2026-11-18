#!/usr/bin/env bash
# Official SWE-bench evaluation (SWE-bench's evaluator at commit 02e7a74, version 5.0.2) of one policy's agent runs,
# on the agent (Docker) host, as the multi-session pilots did after each job (jobs 3299922 and 3300434).
#
# One predictions file holds every planned instance of the given run_agents.sh output directories (their
# instances.txt), with the patch from that directory's preds.json and model_name_or_path = <label>. An instance with
# no prediction (its agent failed or was stopped) is evaluated with an empty patch and reported as missing.
# Everything the evaluator writes (it writes logs/ under its working directory) stays in <eval dir>, which must
# not exist yet.
#
# The dataset is SWE-bench/SWE-bench_Verified at the pinned revision below: its test split file is fetched (or
# taken from the Hugging Face cache), checked against its SHA-256 and passed to the evaluator as a local file, so the
# evaluator never resolves the dataset's HEAD. Every instance's image must be its pinned digest (images.py, as in
# run_agents.sh).
#
# Usage: evaluate.sh <run_id> <label> <eval dir> <run_agents output dir>...
# Environment: EVAL_PYTHON = python of an environment with the SWE-bench evaluator at commit 02e7a74ffd0b707aab73d203fe87bdc7c76afc8e
#   (version 5.0.2; docs/benchmarks/swe-bench.md), which reads a local Parquet dataset (required); WORKERS (default 2);
#   IMAGE_PINS (optional; default verified-32.txt next to this script).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# The evaluation dataset: the revision the evaluations of the validated runs used (its HEAD since 2026-08-16) and
# the SHA-256 of its test split file at that revision (the Hugging Face API's LFS sha256).
DATASET=SWE-bench/SWE-bench_Verified DATASET_REVISION=78f471bf655a3137b2e8a75af1501690ec009ec3
DATASET_FILE=data/test-00000-of-00001.parquet
DATASET_SHA256=030cfd7f2a704c4c0226e7f104c725a3b41230b1d3517f9c915ad7ea5be3fa25
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
# The evaluator must read a local Parquet dataset (load_swebench_dataset of SWE-bench at 02e7a74).
"$EVAL_PYTHON" -I -B -c 'import inspect, swebench.harness.utils as utils; assert ".parquet" in inspect.getsource(utils.load_swebench_dataset)' 2> /dev/null \
  || { echo "the evaluator reads no local Parquet dataset: install SWE-bench at commit 02e7a74" >&2; exit 2; }
INSTANCES=$("$EVAL_PYTHON" -I -B -c 'import json, sys; print(" ".join(json.loads(l)["instance_id"] for l in open(sys.argv[1])))' \
  "$OUT/predictions.jsonl")
python3 -I -B "$HERE/images.py" "${IMAGE_PINS:-$HERE/verified-32.txt}" "$OUT/images.txt" $INSTANCES
"$EVAL_PYTHON" -I -B -c 'import shutil, sys; from huggingface_hub import hf_hub_download; shutil.copyfile(hf_hub_download(sys.argv[1], sys.argv[3], repo_type="dataset", revision=sys.argv[2]), sys.argv[4])' \
  "$DATASET" "$DATASET_REVISION" "$DATASET_FILE" "$OUT/dataset.parquet"
echo "$DATASET_SHA256  $OUT/dataset.parquet" | sha256sum --check --quiet
set +e
"$EVAL_PYTHON" -m swebench.harness.run_evaluation --dataset_name "$OUT/dataset.parquet" --split test \
  --predictions_path "$OUT/predictions.jsonl" --max_workers "$WORKERS" --timeout 1800 \
  --run_id "$RUN_ID" > "$OUT/eval.log" 2>&1
rc=$?
set -e
"$EVAL_PYTHON" - "$OUT" "$RUN_ID" "$rc" <<'PY'
import json, sys
from pathlib import Path
out, run_id, rc = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
# The evaluator's layout (version 5.0.2, as read in the pilots): logs/evaluation/<run_id>/results.json
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
