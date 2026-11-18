#!/usr/bin/env bash
# The tbench lock of perf-swe32, read-only: one JSON line {"event": "swe32_lock", held, alive, kind, run_id, pid, age_s}.
# kind pool (a GPU job's agents), eval (a finished job's detached evaluation) or stress (the stress test).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
lock status
