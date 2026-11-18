#!/usr/bin/env bash
# Build the sealed task statements of the gswe32 task list on tbench (one-time; again after the list changes).
# Runs the runtime's examples/swebench/build_tasks.py (deployed unchanged as bin/build_tasks.py) with the agent env
# offline against the cached dataset: --pool writes one tasks-pool.json with every instance in list order. Copy that
# file to the local perf-gemma-swe32\tasks\ and re-pin with `build.py --tasks` (README "Task statements").
# Usage: build_tasks.sh [instance_id ...]   (default: the deployed tasks/gswe32-tasks.json, in order)
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
: "${MINI_PYTHON:?MINI_PYTHON unset (gswe32.env)}"
INSTANCES=("$@")
if (( ${#INSTANCES[@]} == 0 )); then
  mapfile -t INSTANCES < <(hpy -c 'import json,sys; [print(i) for i in json.load(open(sys.argv[1]))["instances"]]' "$GSWE32_ROOT/tasks/gswe32-tasks.json")
fi
out=$GSWE32_ROOT/tasks/build-$(date -u +%Y%m%dT%H%M%S)
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "$MINI_PYTHON" "$GSWE32_BIN/build_tasks.py" --output-dir "$out" --pool "${INSTANCES[@]}" \
  > "$out.report.json" || { echo "GSWE32_ERROR build_tasks.py failed (see $out.report.json)" >&2; exit 1; }
mv "$out.report.json" "$out/report.json"
echo "tasks-pool.json: $out/tasks-pool.json"
echo "instances: ${#INSTANCES[@]} (first ${INSTANCES[0]}, last ${INSTANCES[-1]})"
sha256sum "$out/tasks-pool.json"
stat -c '%s bytes' "$out/tasks-pool.json"
