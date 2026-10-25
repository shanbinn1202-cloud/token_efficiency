#!/bin/bash -l
set -euo pipefail
export EXPERIMENT_NODE_STARTED=${EXPERIMENT_NODE_STARTED:-$(date +%s)}
export EXPERIMENT_WALL_SECONDS=7200
export PYTHONPATH=/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023
exec /usr/bin/python3 -B -u /projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023/revisions/goal-dynamic-capacity-v18/controller.py
