#!/bin/bash -l
set -euo pipefail
B=/projects/bcjw/ssun4/token_efficiency/experiments/gemma_full_20261017
source "$B/env.sh"
key=${1:?key}
D="$B/cases/$key"
mkdir -p "$D"
exec >> "$D/node.log" 2>&1
cd "$B/harness"
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/math_libs/12.9/include${CPATH:+:$CPATH}"
export TMPDIR="/tmp/ssun4-gemma-full-$SLURM_JOB_ID"
export HF_DATASETS_OFFLINE=1
mkdir -p "$TMPDIR"
ulimit -c 0
printf '%s %s\n' "$SLURM_JOB_ID" "$(hostname)" > "$D/node.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader > "$D/gpu.txt"
"$R/venv/bin/python" -B "$R/stage_local_runtime.py" > "$D/staging.log" 2>&1
export TEBENCH_PYTHON="/tmp/ssun4-vllm024-$SLURM_JOB_ID/venv/bin/python"
export PATH="$(dirname "$TEBENCH_PYTHON"):$PATH"
compiler_alias="$(dirname "$TEBENCH_PYTHON")/CC"
test -x /opt/cray/pe/craype/2.7.35/bin/CC
if [ ! -e "$compiler_alias" ]; then ln -s /opt/cray/pe/craype/2.7.35/bin/CC "$compiler_alias"; fi
"$compiler_alias" --version > "$D/compiler.log" 2>&1
"$TEBENCH_PYTHON" -B -u /projects/bcjw/ssun4/swe500_qwen_20261009/toolchain-check.py > "$D/toolchain.log" 2>&1
exec "$TEBENCH_PYTHON" -B -u "$B/node.py" "$key"
