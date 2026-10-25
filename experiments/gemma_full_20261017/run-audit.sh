#!/bin/bash -l
set -euo pipefail
B=/projects/bcjw/ssun4/token_efficiency/experiments/gemma_full_20261017
source "$B/env.sh"
export TMPDIR="$B/tmp"
mkdir -p "$TMPDIR"
cd "$B/harness"
exec "$R/venv/bin/python" -B -u "$B/audit.py" "$1"
