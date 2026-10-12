#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
AGENT_ENV="${AGENT_ENV:-/u/ssun4/miniconda3/envs/agent}"
export PATH="$AGENT_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/mini-swe-agent/src:$PROJECT_ROOT/external/SWE-bench:$PROJECT_ROOT/experiments/agentkv_qwen36/vendor${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/work/nvme/bcjw/ssun4/cache/huggingface}"
export GOMAXPROCS="${GOMAXPROCS:-2}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$PROJECT_ROOT/data/apptainer/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}/token-efficiency-$UID/build}"
export TMPDIR="${TMPDIR:-/tmp/token-efficiency-$UID/sandboxes}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$TMPDIR" "$PROJECT_ROOT/logs"
test -x "$AGENT_ENV/bin/python"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
