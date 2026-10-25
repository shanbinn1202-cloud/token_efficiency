#!/bin/bash -l
set -euo pipefail
umask 077
export EXPERIMENT_NODE_STARTED=${EXPERIMENT_NODE_STARTED:-$(date +%s)}
export EXPERIMENT_WALL_SECONDS=${EXPERIMENT_WALL_SECONDS:-7200}
D=/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023
source /projects/bcjw/ssun4/token_efficiency/experiments/gemma_full_20261017/env.sh
export TEBENCH_VLLM_ROOT=/projects/bcjw/ssun4/token_efficiency/engines/vllm_024
export PYTHONPATH="$TEBENCH_VLLM_ROOT:/projects/bcjw/ssun4/token_efficiency/experiments/gemma_full_20261017/harness:/projects/bcjw/ssun4/token_efficiency/mini-swe-agent/src"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/math_libs/12.9/include${CPATH:+:$CPATH}"
export TMPDIR="/tmp/ssun4-terminal-validation-$SLURM_JOB_ID"
TAG="${EXPERIMENT_VALIDATION_TAG:-}"
if [[ -n "$TAG" && ! "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]]; then exit 2; fi
ATTEMPT="$SLURM_JOB_ID${TAG:+-$TAG}"
mkdir -p "$TMPDIR" "$D/terminal/validation-jobs/$ATTEMPT"
OUT="$D/terminal/validation-jobs/$ATTEMPT"
exec >> "$OUT/job.log" 2>&1
nvidia-smi --query-gpu=name,memory.total,uuid --format=csv > "$OUT/gpu.csv"
"$R/venv/bin/python" -B "$R/stage_local_runtime.py" > "$OUT/staging.log" 2>&1
E="/tmp/ssun4-vllm024-$SLURM_JOB_ID/venv"
export PATH="$E/bin:$PATH"
export LD_LIBRARY_PATH="$E/lib/python3.12/site-packages/nvidia/cu13/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
"$E/bin/python" -B "$D/preflight_cuda_libraries.py" > "$OUT/cuda-libraries.json"
if [ ! -e "$E/bin/CC" ]; then ln -s /opt/cray/pe/craype/2.7.35/bin/CC "$E/bin/CC"; fi
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_SERVER_DEV_MODE=1
exec "$E/bin/python" -B -u "$D/terminal_stage.py" --model qwen --method vanilla --validation
