#!/usr/bin/env bash
# The shared tbench locks of perf-tb16, read-only: one JSON line {"event": "tb16_locks", held, alive, kind, run_id,
# eval_only, blocking, locks: {tb16, swe32, swe16, swe10: each lock's status}} (lockset.py status).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
locks status
