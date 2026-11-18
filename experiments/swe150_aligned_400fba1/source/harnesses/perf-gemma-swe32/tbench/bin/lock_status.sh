#!/usr/bin/env bash
# The shared tbench locks of perf-gemma-swe32, read-only: one JSON line {"event": "gswe32_locks", held, alive,
# kind, run_id, pid, age_s, holder (of the first live holder), eval_only, blocking, locks: {gswe32, tb16, swe32,
# swe16, swe10: each lock's status}} (lockset.py status). kind pool (a GPU job's agents), eval (a finished job's
# detached evaluation) or stress (the stress test); exit 2 with {"event": "gswe32_locks_error"} when the set cannot be
# read.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
lock status
