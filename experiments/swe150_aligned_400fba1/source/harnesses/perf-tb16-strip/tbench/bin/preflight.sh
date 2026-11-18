#!/usr/bin/env bash
# Read-only readiness check of the tb16 tbench half for one plan (perf-tb16), run over SSH by the DeltaAI login node
# before the claim. Prints one final JSON line {"event": "tb16_preflight", "status": "PASS" | "FAIL", ...}; exit 0 on
# PASS only. Checks: the deployed files, the shared lock modules, Harbor's files against the wheel RECORD, litellm /
# openai, no MSWEA_CTX_* variable, the plan's task packages and images, the shared locks, host_check.py (preflight.py).
# Usage: preflight.sh <base64url of zlib-compressed JSON pins: {harbor: {...}, tasks: [{name, content_hash, tree_sha256,
#        docker_image}]}>
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
exec "$HOST_PYTHON" -I -B "$TB16_BIN/preflight.py" "$TB16_ROOT" "${1:-}"
