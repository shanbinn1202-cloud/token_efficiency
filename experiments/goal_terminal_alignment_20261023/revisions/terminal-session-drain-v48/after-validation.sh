#!/bin/bash -l
set -euo pipefail
D=/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023
MARKER="$D/revisions/terminal-session-drain-v48/allow-continuation-$SLURM_JOB_ID.json"
while [ ! -f "$MARKER" ]; do
  if [ $((SLURM_JOB_END_TIME-$(date +%s))) -lt 180 ]; then exit 0; fi
  sleep 10
done
"$D/harbor-env/bin/python" -B "$D/check_qualification.py" terminal all
scontrol update JobId="$SLURM_JOB_ID" JobName=terminal-qwen-vanilla
unset EXPERIMENT_VALIDATION_TAG TERMINAL_LIFECYCLE_PROBE
# The validation already copied and checked this allocation's environment.
# Reuse it without another full rsync traversal or resetting its Slurm deadline.
source "$D/../gemma_full_20261017/env.sh"
E="/tmp/ssun4-vllm024-$SLURM_JOB_ID/venv"
test -f "/tmp/ssun4-vllm024-$SLURM_JOB_ID/venv-ready"
export TEBENCH_VLLM_ROOT="$D/../../engines/vllm_024"
export PYTHONPATH="$TEBENCH_VLLM_ROOT:$D/../gemma_full_20261017/harness:$D/../../mini-swe-agent/src"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9
export PATH="$E/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$E/lib/python3.12/site-packages/nvidia/cu13/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/math_libs/12.9/include${CPATH:+:$CPATH}"
export TMPDIR="/tmp/ssun4-terminal-validation-$SLURM_JOB_ID"
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_SERVER_DEV_MODE=1
exec "$E/bin/python" -B -u "$D/terminal_stage.py" --model qwen --method vanilla
