#!/bin/bash -l
set -euo pipefail
umask 077
export EXPERIMENT_NODE_STARTED=$(date +%s)
export EXPERIMENT_WALL_SECONDS=${EXPERIMENT_WALL_SECONDS:-36000}
D=/projects/bcjw/ssun4/token_efficiency/experiments/goal_terminal_alignment_20261023
export GOAL_LOCAL_ROOT="/tmp/ssun4-goal-align-$SLURM_JOB_ID"
export TMPDIR="$GOAL_LOCAL_ROOT/tmp"
mkdir -p "$TMPDIR" "$D/goal/allocation-logs/$SLURM_JOB_ID"
OUT="$D/goal/allocation-logs/$SLURM_JOB_ID"
mkdir -p "$OUT"
exec >> "$OUT/job.log" 2>&1
printf 'JOB %s NODE %s\n' "$SLURM_JOB_ID" "$(hostname)"
nvidia-smi --query-gpu=name,memory.total,uuid,driver_version --format=csv > "$OUT/gpu.csv"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MAX_JOBS=4
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/cuda/12.9
export CPATH="/opt/nvidia/hpc_sdk/Linux_aarch64/25.5/math_libs/12.9/include${CPATH:+:$CPATH}"
export XDG_CACHE_HOME="$GOAL_LOCAL_ROOT/cache"
export TRITON_CACHE_DIR="$GOAL_LOCAL_ROOT/triton"
export VLLM_CACHE_ROOT="$GOAL_LOCAL_ROOT/vllm-cache"
export TORCHINDUCTOR_CACHE_DIR="$GOAL_LOCAL_ROOT/inductor"
/projects/bcjw/ssun4/selective_prefix_first32_20261006/environment/venv/bin/python -B "$D/stage_goal_runtime.py" > "$OUT/staging.log" 2>&1
E="$GOAL_LOCAL_ROOT/venv"
export PATH="$E/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$E/lib/python3.12/site-packages/nvidia/cu13/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
unset PYTHONPATH
if [ ! -e "$E/bin/CC" ]; then ln -s /opt/cray/pe/craype/2.7.35/bin/CC "$E/bin/CC"; fi
mkdir -p "$GOAL_LOCAL_ROOT/releases"
MODEL=${1:?model}
BENCH=${2:?benchmark}
"$E/bin/python" -B "$D/check_qualification.py" goal "$MODEL"
if [ "${EXPERIMENT_GRADING_ONLY:-0}" = 1 ]; then
  exec "$E/bin/python" -B -u "$D/grade_goal.py" "$MODEL" "$BENCH"
fi
ARGS=()
if [ "$MODEL" = qwen ]; then
  WEIGHTS=/projects/bcjw/ssun4/selective_prefix_first32_20261006/model
else
  WEIGHTS=/projects/bcjw/ssun4/models/gemma-4-31B-it-842da379
  ARGS+=(--model-profile gemma4-31b-it)
fi
RELEASE="$GOAL_LOCAL_ROOT/releases/$MODEL"
"$E/bin/python" -B "$D/selective-prefix/install.py" --site-packages "$E/lib/python3.12/site-packages" --model-dir "$WEIGHTS" --release-dir "$RELEASE" --apply "${ARGS[@]}" > "$OUT/install-$MODEL.log" 2>&1
cp "$RELEASE/runtime-manifest.json" "$OUT/runtime-manifest-$MODEL.json"
exec "$E/bin/python" -B -u "$D/goal_stage.py" --model "$MODEL" --benchmark "$BENCH" --release "$RELEASE"
