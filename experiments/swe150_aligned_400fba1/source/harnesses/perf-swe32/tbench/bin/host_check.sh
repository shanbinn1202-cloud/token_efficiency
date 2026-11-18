#!/usr/bin/env bash
# Light read-only host check before a swe32 pool (host_check.py): refuses only for leftover containers of OUR previous
# runs or too little MemAvailable / free disk (swe32.env limits); other containers and the load are reported.
# One JSON line {"event": "swe32_host", ...}; exit 0 free, 5 busy. Never stops, prunes or removes anything.
# Usage: host_check.sh [strict|preflight]
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
exec "$HOST_PYTHON" -I -B "$SWE32_BIN/host_check.py" "$SWE32_ROOT" "${1:-strict}"
