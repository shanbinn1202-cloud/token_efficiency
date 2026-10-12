#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export PYTHONPATH="$PROJECT_ROOT/engines/vllm:$PYTHONPATH"
exec "$AGENT_ENV/bin/python" -B -m pytest -q -p no:cacheprovider tests "$@"
