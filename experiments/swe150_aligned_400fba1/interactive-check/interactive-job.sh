#!/bin/bash
set -euo pipefail
umask 077
F=/projects/bcjw/ssun4/swe150_aligned_400fba1
test "${SLURM_JOB_PARTITION:?}" = ghx4-interactive
OUT="$F/interactive-check/runs/${SLURM_JOB_ID:?}"
mkdir -p "$OUT"
exec >> "$OUT/job.log" 2>&1
on_error() {
 local exit_code=$? line=$1
 printf '{"exit_code":%s,"line":%s,"phase":"job_shell"}\n' "$exit_code" "$line" > "$OUT/job-failure.json"
 exit "$exit_code"
}
trap 'on_error "$LINENO"' ERR
test "$(uname -m)" = aarch64
/usr/bin/python3 -B "$F/verify_manifest.py"
/usr/bin/python3 -B "$F/check_local_assets.py" > "$OUT/local-assets.json"
export GOAL_LOCAL_ROOT="/tmp/swe150-$SLURM_JOB_ID"
export TMPDIR="$GOAL_LOCAL_ROOT/tmp"
mkdir -p "$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9
export CPATH=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/math_libs/12.9/include
export PATH="$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export HF_HOME=/work/nvme/bcjw/ssun4/cache/huggingface
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MAX_JOBS=4
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_SERVER_DEV_MODE=1
unset PYTHONPATH SWE64_VERIFY_AGENTKV_RESTORE
/projects/bcjw/ssun4/selective_prefix_first32_20261006/environment/venv/bin/python -S -B "$F/stage_goal_environment.py"
E="$GOAL_LOCAL_ROOT/venv"
if [ ! -e "$E/bin/CC" ]; then ln -s /opt/cray/pe/craype/2.7.35/bin/CC "$E/bin/CC"; fi
export PATH="$E/bin:$PATH"
export LD_LIBRARY_PATH="$E/lib/python3.12/site-packages/nvidia/cu13/lib:$CUDA_HOME/lib64"
export XDG_CACHE_HOME="$GOAL_LOCAL_ROOT/cache" TRITON_CACHE_DIR="$GOAL_LOCAL_ROOT/triton"
export VLLM_CACHE_ROOT="$GOAL_LOCAL_ROOT/vllm-cache" TORCHINDUCTOR_CACHE_DIR="$GOAL_LOCAL_ROOT/inductor"
export FLASHINFER_WORKSPACE_BASE="$GOAL_LOCAL_ROOT/flashinfer-home" VLLM_RPC_BASE_PATH="$GOAL_LOCAL_ROOT/rpc"
mkdir -p "$VLLM_RPC_BASE_PATH"
"$E/bin/python" -B "$F/stage.py"
"$E/bin/python" -B "$F/engines_adapter/prepare.py" --local "$GOAL_LOCAL_ROOT"
exec "$E/bin/python" -B "$F/interactive-check/verify_methods.py" "$OUT" "$GOAL_LOCAL_ROOT"
