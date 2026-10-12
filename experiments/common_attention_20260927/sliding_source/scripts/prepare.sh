#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
exec "$AGENT_ENV/bin/python" -B -m baselines.common.prepare "$@"
