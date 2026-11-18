#!/usr/bin/env bash
# Read-only readiness check of the gswe32 tbench half for one plan (perf-gemma-swe32), run over SSH by the DeltaAI login node
# before the claim. Prints one final JSON line {"event": "gswe32_preflight", "status": "PASS" | "FAIL", ...}; exit 0 on
# PASS only. Checks: the deployed files against their manifest, the pinned runner and prompt, mini-swe-agent 2.4.2,
# the evaluator env (swebench), every instance's evaluation image, the cached dataset's statements equal to the sealed
# ones, free disk, the socket directory 0700, every shared lock module copy equal to the deployed half's own, the
# shared tbench locks (lockset.py status: a live pool, stress test or TB run of any half fails; a running evaluation of
# any half is reported: the job waits for it, bounded) and host_check.py (no leftover container of our previous runs,
# enough MemAvailable / disk; other containers and the load are reported, never refused, never touched).
# Usage: preflight.sh <runner sha256> <prompt sha256> <instance_id>=<statement sha256>...
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RUNNER=${1:-}; PROMPT=${2:-}
shift 2 2>/dev/null
exec "$HOST_PYTHON" -I -B "$GSWE32_BIN/preflight.py" "$GSWE32_ROOT" "$RUNNER" "$PROMPT" "$@"
