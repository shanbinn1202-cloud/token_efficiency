#!/usr/bin/env bash
# Package one job's tbench run for the local fetch (perf-tb16; read by postrun/fetch_run.ps1 over SSH).
# While a Harbor run of it is still going (its run_harbor.sh or Harbor alive, or the shared locks held by it) it
# packages nothing and says so (exit 3: fetch again later). Otherwise it writes packages/<run_id>.tar.xz (the run
# directory: specs, job configs, Harbor's job directories with every trial's result, ATIF trajectories, verifier
# output and logs, the host samples; pid files and markers left out) and its SHA-256, once (a repeated call returns
# the same package).
# Usage: package_run.sh <run_id>
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUN=${1:-}
[[ $RUN =~ $RUN_ID_RE && -d $RUNS/$RUN ]] || { json_line tb16_package "{\"status\": \"NO_RUN\", \"run_id\": \"$RUN\"}"; exit 2; }
alive=0
for f in "$RUNS/$RUN"/*/pool.pid "$RUNS/$RUN"/*/harbor.pid; do
  [[ -f $f ]] || continue
  dir=$(dirname "$f")
  [[ -f $dir/.done ]] && continue
  kill -0 "$(cat "$f")" 2>/dev/null && alive=1
done
owner=$(locks status)
held=$(hpy -c 'import json,sys; v=json.loads(sys.argv[1]); print(1 if v.get("held") and v.get("run_id")==sys.argv[2] else 0)' "$owner" "$RUN")
if [[ $alive == 1 || $held == 1 ]]; then
  json_line tb16_package "{\"status\": \"HARBOR_RUNNING\", \"run_id\": \"$RUN\", \"locks\": $owner}"
  exit 3
fi
mkdir -p "$PACKAGES"
target=$PACKAGES/$RUN.tar.xz
if [[ ! -f $target ]]; then
  tmp=$target.tmp.$$
  tar -C "$RUNS" --exclude='*.pid' --exclude='*/.done' --exclude='*/.stopping' --exclude='*/.sampler-stop' \
      -cf - "$RUN" | xz -T4 -6 > "$tmp" && mv "$tmp" "$target" \
    || { rm -f "$tmp"; json_line tb16_package "{\"status\": \"PACKAGE_FAILED\", \"run_id\": \"$RUN\"}"; exit 1; }
fi
sum=$(sha256sum "$target" | cut -d' ' -f1)
bytes=$(stat -c %s "$target" 2>/dev/null || wc -c < "$target")
json_line tb16_package "{\"status\": \"PACKAGED\", \"run_id\": \"$RUN\", \"path\": \"$target\", \"sha256\": \"$sum\", \"bytes\": $bytes}"
