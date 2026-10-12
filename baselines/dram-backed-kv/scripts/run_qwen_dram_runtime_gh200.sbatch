#!/bin/bash
#SBATCH --job-name=qwen-dram-kv
#SBATCH --account=bcjw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=results/raw/qwen_dram_runtime_gh200_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DRAM_KV_ENV_FILE:-${SCRIPT_DIR}/env.local.sh}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

DEFAULT_REPO="${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
REPO="${DRAM_KV_REPO_ROOT:-${DEFAULT_REPO}}"
PYTHON="${DRAM_KV_PYTHON:-${VLLM_PYTHON:-python3}}"
MODEL="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
OUTPUT_ROOT="${DRAM_KV_OUTPUT_ROOT:-${REPO}/results}"

[[ -x "${PYTHON}" ]] || { echo "Missing Python executable: ${PYTHON}" >&2; exit 2; }
[[ -e "${MODEL}" || "${MODEL}" != /* ]] || { echo "Missing model: ${MODEL}" >&2; exit 2; }

cd "${REPO}"
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_ROOT}/raw"
"$PYTHON" scripts/validate_qwen_dram_runtime.py \
  --model "$MODEL" \
  --tokens 8192 \
  --page-size 512 \
  --selected-pages 4 \
  --output "${OUTPUT_ROOT}/raw/qwen_dram_runtime_gh200_${SLURM_JOB_ID}.json"
