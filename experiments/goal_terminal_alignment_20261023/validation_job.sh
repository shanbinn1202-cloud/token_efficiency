#!/bin/bash -l
set -euo pipefail
D=/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023
export EXPERIMENT_NODE_STARTED=${EXPERIMENT_NODE_STARTED:-$(date +%s)}
export EXPERIMENT_WALL_SECONDS=${EXPERIMENT_WALL_SECONDS:-7200}
exec /usr/bin/python3 -B -u "$D/shared_validation_allocation.py"
