#!/bin/bash -l
set -euo pipefail
D=/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023
export EXPERIMENT_NODE_STARTED=$(date +%s)
export EXPERIMENT_WALL_SECONDS=7200
export SLURM_JOB_END_TIME=${SLURM_JOB_END_TIME:-$((EXPERIMENT_NODE_STARTED+7200))}
export EXPERIMENT_VALIDATION_TAG=terminal-session-drain-v48
export TERMINAL_LIFECYCLE_PROBE=1
bash "$D/terminal_validation_job.sh"
exec bash "$D/revisions/terminal-session-drain-v48/after-validation.sh"
