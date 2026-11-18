#!/usr/bin/env bash
# Read-only: what Terminus 2 does with the tb16 settings (retry layers, timeouts, defaults), from the installed Harbor
# env (check_agent_env.py under its python, -I -B, HARBOR_TELEMETRY=0). One JSON line; send it back with the stress
# test's summary.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
: "${HARBOR_ENV:?HARBOR_ENV unset (tb16.env)}"
HARBOR_TELEMETRY=0 "$HARBOR_ENV/bin/python" -I -B "$TB16_BIN/check_agent_env.py" "$@"
