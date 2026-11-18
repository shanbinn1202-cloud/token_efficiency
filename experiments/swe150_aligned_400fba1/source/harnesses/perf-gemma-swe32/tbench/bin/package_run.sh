#!/usr/bin/env bash
# Package one job's tbench run for the local fetch (perf-gemma-swe32; read by postrun/fetch_run.ps1 over SSH).
# While the run's evaluation is still running (its eval-status.json says running, or a live owner of the run holds a
# lock of the shared tbench lock set) it packages nothing and says so (exit 3: fetch again later). Otherwise
# it writes packages/<run_id>.tar.xz (the run directory: pool outputs, trajectories, logs, timelines, close receipts,
# host samples, evaluation reports and logs; the pools' pid files and markers left out) and its SHA-256, once per
# evaluation state (a repeated call returns the same package).
# Usage: package_run.sh <run_id>
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUN=${1:-}
[[ $RUN =~ $RUN_ID_RE && -d $RUNS/$RUN ]] || { json_line gswe32_package "{\"status\": \"NO_RUN\", \"run_id\": \"$RUN\"}"; exit 2; }
state=$(hpy -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text()).get("state") if p.is_file() else "absent")' "$RUNS/$RUN/eval-status.json")
owner=$(lock status)
running=$(hpy -c 'import json,sys; v=json.loads(sys.argv[1]); print(1 if any(x.get("held") and x.get("alive") and x.get("run_id")==sys.argv[2] for x in (v.get("locks") or {}).values()) else 0)' "$owner" "$RUN")
if [[ $running == 1 || $state == running ]]; then
  json_line gswe32_package "{\"status\": \"EVAL_RUNNING\", \"run_id\": \"$RUN\", \"eval_state\": \"$state\", \"lock\": ${owner:-null}}"
  exit 3
fi
mkdir -p "$PACKAGES"
target=$PACKAGES/$RUN.$state.tar.xz
if [[ ! -f $target ]]; then
  tmp=$target.tmp.$$
  tar -C "$RUNS" --exclude='*/pids' --exclude='*.pid' --exclude='*/.done' --exclude='*/.stopping' \
      --exclude='*/.sampler-stop' -cf - "$RUN" | xz -T4 -6 > "$tmp" && mv "$tmp" "$target" \
    || { rm -f "$tmp"; json_line gswe32_package "{\"status\": \"PACKAGE_FAILED\", \"run_id\": \"$RUN\"}"; exit 1; }
fi
sum=$(sha256sum "$target" | cut -d' ' -f1)
bytes=$(stat -c %s "$target" 2>/dev/null || wc -c < "$target")
json_line gswe32_package "{\"status\": \"PACKAGED\", \"run_id\": \"$RUN\", \"eval_state\": \"$state\", \"path\": \"$target\", \"sha256\": \"$sum\", \"bytes\": $bytes}"
