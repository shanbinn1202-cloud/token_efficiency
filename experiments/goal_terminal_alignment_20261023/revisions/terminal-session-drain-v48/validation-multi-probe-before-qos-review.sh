#!/bin/bash -l
set -euo pipefail
D=/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023
export EXPERIMENT_NODE_STARTED=$(date +%s)
export EXPERIMENT_WALL_SECONDS=7200
export SLURM_JOB_END_TIME=${SLURM_JOB_END_TIME:-$((EXPERIMENT_NODE_STARTED+7200))}
export EXPERIMENT_VALIDATION_TAG=terminal-session-drain-v48
export TERMINAL_LIFECYCLE_PROBE=1
bash "$D/terminal_validation_job.sh"
# Formal continuations now have their own owners. Reuse this allocation for
# additional real-model validation rather than launch a duplicate formal run.
source "$D/../gemma_full_20261017/env.sh"
E="/tmp/ssun4-vllm024-$SLURM_JOB_ID/venv"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9
export PATH="$E/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$E/lib/python3.12/site-packages/nvidia/cu13/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/math_libs/12.9/include${CPATH:+:$CPATH}"
export PYTHONPATH="$D/../../engines/vllm_024:$D/../gemma_full_20261017/harness:$D/../../mini-swe-agent/src"
export TMPDIR="/tmp/ssun4-terminal-validation-$SLURM_JOB_ID"
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_SERVER_DEV_MODE=1
for pair in gemma:sliding_window qwen:agentkv qwen:random_attention qwen:intentkv; do
  if [ $((SLURM_JOB_END_TIME - $(date +%s))) -lt 1200 ]; then break; fi
  model=${pair%%:*}
  method=${pair#*:}
  export EXPERIMENT_VALIDATION_TAG="terminal-session-drain-v48-$model-$method"
  "$E/bin/python" -B -u "$D/terminal_stage.py" --model "$model" --method "$method" --validation
done
