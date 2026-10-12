#!/bin/bash
# Single restored-Oracle baseline paired with job3056285's turn-30 omission.
#SBATCH --account=bcjw-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128g
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --job-name=dram_kv_restore
#SBATCH --output=dram_kv_restore_%j.out
#SBATCH --error=dram_kv_restore_%j.err

set -euo pipefail
module load python/miniforge3_pytorch/2.10.0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DRAM_KV_ENV_FILE:-${SCRIPT_DIR}/env.local.sh}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

DEFAULT_REPO="${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
REPO="${DRAM_KV_REPO_ROOT:-${DEFAULT_REPO}}"
LEGACY_REPO="${DRAM_KV_LEGACY_REPO:?Set DRAM_KV_LEGACY_REPO in ${ENV_FILE}}"
PYTHON="${DRAM_KV_PYTHON:-${VLLM_PYTHON:-python3}}"
MODEL="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
TRAJECTORY="${DRAM_KV_TRAJECTORY_13579:?Set DRAM_KV_TRAJECTORY_13579 in ${ENV_FILE}}"
ANNOTATION="${DRAM_KV_ANNOTATION_13579:?Set DRAM_KV_ANNOTATION_13579 in ${ENV_FILE}}"
OUTPUT_DIR="${DRAM_KV_OUTPUT_ROOT:-${REPO}/results}/job${SLURM_JOB_ID}"

for required in "${PYTHON}" "${MODEL}" "${LEGACY_REPO}" "${TRAJECTORY}" "${ANNOTATION}"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 2; }
done

mkdir -p "${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "${REPO}"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

"${PYTHON}" scripts/run_legacy_regression.py \
  --legacy-repo "${LEGACY_REPO}" \
  --model "${MODEL}" \
  --trajectory "${TRAJECTORY}" \
  --annotation "${ANNOTATION}" \
  --no-dry-run --device cuda:0 --backing-device cuda \
  --page-size 128 --kl-chunk-tokens 64 \
  --output "${OUTPUT_DIR}/oracle_restored.metrics.jsonl" \
  --framework-artifacts "${OUTPUT_DIR}/oracle_restored.framework.jsonl"

echo "RESTORED_BASELINE_COMPLETE output=${OUTPUT_DIR}"
