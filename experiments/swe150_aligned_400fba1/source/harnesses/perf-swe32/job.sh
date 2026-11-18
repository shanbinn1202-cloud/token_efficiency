#!/usr/bin/env bash
set -euo pipefail
umask 077
stage="$1"
# Reused read-only: the v9 clean environment of Slurm job 3270721 (pins.json clean_env.prefix).
clean="/u/ycao11/spq9-277fa5f1390947e08caefecc8e928a3d/clean-env"
test -n "${SLURM_JOB_ID:-}"
test -d "$stage" && test ! -L "$stage"
export TMPDIR="$stage/tmp" TMP="$stage/tmp" TEMP="$stage/tmp"
export HOME="$stage/home" XDG_CACHE_HOME="$stage/cache"
export HF_HOME="$stage/cache/hf" TORCH_HOME="$stage/cache/torch"
export TRITON_CACHE_DIR="$stage/cache/triton" CUDA_CACHE_PATH="$stage/cache/cuda"
export TORCH_EXTENSIONS_DIR="$stage/cache/torch-extensions"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_POLICY_RUNTIME_V1=0 REMOTE_RA_DYNAMIC_STARTUP=0
# v6 backend profile (controller.ACCEPTANCE_PROFILE): native sampler, no FlashInfer
# sampling JIT against the stage-local cache. serve.py passes it on.
export VLLM_USE_FLASHINFER_SAMPLER=0
unset VLLM_RPC_BASE_PATH
# DeltaAI login shells export CC=cc (Cray PE). Triton uses $CC before gcc/clang,
# so drop it: Triton JIT then uses gcc on the job PATH, as the compiler gate checked.
unset CC
export PATH="$clean/bin:/usr/local/bin:/usr/bin:/bin"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_PRELOAD
export LD_LIBRARY_PATH="$clean/lib/python3.12/site-packages/nvidia/cu13/lib"
exec "$clean/bin/python" -I -B "$stage/job.py" "$stage"
