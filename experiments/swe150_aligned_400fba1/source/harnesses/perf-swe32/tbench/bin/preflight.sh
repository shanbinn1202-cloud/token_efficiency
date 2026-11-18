#!/usr/bin/env bash
# Read-only readiness check of the swe32 tbench half for one plan (perf-swe32), run over SSH by the DeltaAI login node
# before the claim. Prints one final JSON line {"event": "swe32_preflight", "status": "PASS" | "FAIL", ...}; exit 0 on
# PASS only. Checks: the deployed files against their manifest, the pinned runner and prompt, mini-swe-agent 2.4.2,
# the evaluator env (swebench), every instance's evaluation image, the cached dataset's statements equal to the sealed
# ones, free disk, the socket directory 0700, the tbench lock (a live pool or stress test fails; a running evaluation
# is reported: the job waits for it, bounded) and host_check.py (no leftover container of our previous runs, enough
# MemAvailable / disk; other containers and the load are reported, never refused, never touched).
# Usage: preflight.sh <runner sha256> <prompt sha256> <instance_id>=<statement sha256>...
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUNNER=${1:-}; PROMPT=${2:-}
shift 2 2>/dev/null
exec "$HOST_PYTHON" -I -B "$SWE32_BIN/preflight.py" "$SWE32_ROOT" "$RUNNER" "$PROMPT" "$@"
