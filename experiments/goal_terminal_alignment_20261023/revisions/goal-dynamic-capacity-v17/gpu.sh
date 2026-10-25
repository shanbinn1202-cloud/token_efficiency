#!/bin/bash -l
set -euo pipefail
umask 077
D=/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023
R="$D/revisions/goal-dynamic-capacity-v17"
export GOAL_LOCAL_ROOT="/tmp/ssun4-goal-align-$SLURM_JOB_ID"
E="$GOAL_LOCAL_ROOT/venv"
OUT="$D/validation/$SLURM_JOB_ID-goal-capacity-v17"
mkdir -p "$OUT" "$GOAL_LOCAL_ROOT/tmp"
exec >> "$OUT/job.log" 2>&1
export TMPDIR="$GOAL_LOCAL_ROOT/tmp" PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MAX_JOBS=4
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9
export CPATH="/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/math_libs/12.9/include${CPATH:+:$CPATH}"
export XDG_CACHE_HOME="$GOAL_LOCAL_ROOT/cache" TRITON_CACHE_DIR="$GOAL_LOCAL_ROOT/triton" VLLM_CACHE_ROOT="$GOAL_LOCAL_ROOT/vllm-cache" TORCHINDUCTOR_CACHE_DIR="$GOAL_LOCAL_ROOT/inductor"
export PATH="$E/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$E/lib/python3.12/site-packages/nvidia/cu13/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
unset PYTHONPATH
RELEASE="$GOAL_LOCAL_ROOT/releases/gemma-capacity-v17"
"$E/bin/python" -B "$R/selective-prefix/install.py" --site-packages "$E/lib/python3.12/site-packages" --model-dir /projects/bcjw/ssun4/models/gemma-4-31B-it-842da379 --release-dir "$RELEASE" --apply --model-profile gemma4-31b-it > "$OUT/install.log" 2>&1
cp "$RELEASE/runtime-manifest.json" "$OUT/runtime-manifest.json"
cp "$R/selective-prefix/INSTALL_RELEASE.json" "$OUT/candidate-source.json"
"$E/bin/python" -B -u "$R/validate32.py" gemma "$RELEASE" "$OUT/gemma"
