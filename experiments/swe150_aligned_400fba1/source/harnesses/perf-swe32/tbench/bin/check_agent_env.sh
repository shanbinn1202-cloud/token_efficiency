#!/usr/bin/env bash
# Read-only: what the agents do on a slow or failed request (litellm timeout and retries, mini's retry), from the
# installed agent env (perf-swe32 design R9). One JSON line; send it back with the stress-test summary.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
: "${MINI_PYTHON:?MINI_PYTHON unset (swe32.env)}"
MSWEA_GLOBAL_CONFIG_DIR="$MSWEA_CONFIG_DIR" HF_HUB_OFFLINE=1 "$MINI_PYTHON" -I -B "$SWE32_BIN/check_agent_env.py"
