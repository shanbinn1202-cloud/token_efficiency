#!/bin/bash
#SBATCH --job-name=online-sel-v2-swe
#SBATCH --account=bcjw-delta-gpu
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=results/raw/online_selector_v2_swe_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="${SLURM_SUBMIT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
REPO="${DRAM_KV_REPO_ROOT:-${DEFAULT_REPO}}"
ENV_FILE="${DRAM_KV_ENV_FILE:-${REPO}/scripts/env.local.sh}"
source "${ENV_FILE}"
AGENT_ROOT="${AGENT_ROOT:?Set AGENT_ROOT in ${ENV_FILE}}"
source "${AGENT_ROOT}/agentdiet.env"

# Array tasks and parallel job steps can share a physical node. Give each one
# private Apptainer scratch/cache directories so image extraction cannot collide.
RUN_SLOT="${RUN_SLOT:-${SLURM_ARRAY_TASK_ID:-${SLURM_PROCID:-0}}}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/apptainer_tmp_xzheng5_${SLURM_JOB_ID:-local}_${RUN_SLOT}}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/tmp/apptainer_cache_xzheng5_${SLURM_JOB_ID:-local}_${RUN_SLOT}}"

SERVER_PYTHON="${DRAM_KV_PYTHON:?DRAM_KV_PYTHON is required}"
MODEL="${MODEL_NAME:?MODEL_NAME is required}"
MINI_PYTHON="${MINI_PYTHON:?MINI_PYTHON is required}"
INSTANCE="${INSTANCE:-astropy__astropy-14995}"
INSTANCE_FILTER="${INSTANCE_FILTER:-^${INSTANCE}$}"
PORT="${PORT:-8099}"
# Compatibility guard for cohort-2 allocations submitted before the wrapper
# began deriving ports itself. The wrapper is copied by sbatch, while this base
# runner is opened at execution time, so pending jobs also receive the fix.
if [[ -n "${EXPERIMENT_MODE:-}" && -n "${RUN_GROUP:-}" && "${PORT}" =~ ^1811[01]$ ]]; then
  PORT=$((20000 + (SLURM_JOB_ID % 10000) * 2 + RUN_SLOT))
fi
STEP_LIMIT="${STEP_LIMIT:-30}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
SWITCH_THRESHOLD="${SWITCH_THRESHOLD:-8192}"
SELECTION_MODE="${SELECTION_MODE:-selector}"
QUERY_SOURCE="${QUERY_SOURCE:-goal}"
GOAL_REWRITER_MODEL="${GOAL_REWRITER_MODEL:?GOAL_REWRITER_MODEL is required}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO}/results/raw/online_selector_v2_${INSTANCE}_${SLURM_JOB_ID}}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"

export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${SERVER_PYTHON}"):${PATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${SERVER_PYTHON}" "${REPO}/scripts/serve_online_selector_v2.py" \
  --model "${MODEL}" \
  --served-model-name selector-v2-qwen \
  --port "${PORT}" \
  --active-ratio 0.40 \
  --selection-mode "${SELECTION_MODE}" \
  --switch-threshold "${SWITCH_THRESHOLD}" \
  --recent-turns 2 \
  --recent-quota 64 \
  --query-source "${QUERY_SOURCE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --goal-rewriter-model "${GOAL_REWRITER_MODEL}" \
  --metrics "${OUTPUT_DIR}/selector_metrics.jsonl" \
  > "${OUTPUT_DIR}/selector_server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _attempt in $(seq 1 120); do
  if "${MINI_PYTHON}" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=2)" 2>/dev/null; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -100 "${OUTPUT_DIR}/selector_server.log" || true
    exit 1
  fi
  sleep 5
done
"${MINI_PYTHON}" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=2)"

DYNAMIC_CONFIG="${OUTPUT_DIR}/model_override.yaml"
cat > "${DYNAMIC_CONFIG}" <<EOF
model:
  model_name: "openai/selector-v2-qwen"
  model_class: litellm_textbased
  cost_tracking: ignore_errors
  model_kwargs:
    api_base: "http://127.0.0.1:${PORT}/v1"
    api_key: "dummy"
    temperature: 0.0
    max_tokens: ${MAX_NEW_TOKENS}

environment:
  environment_class: singularity
  pull_timeout: 600
  cwd: "/testbed"
  timeout: 120
  interpreter: ["bash", "-c"]
  env:
    PAGER: cat
    MANPAGER: cat
    LESS: -R
    PIP_PROGRESS_BAR: "off"
    TQDM_DISABLE: "1"

agent:
  step_limit: ${STEP_LIMIT}
  cost_limit: 0
EOF

export SIF_BASE
"${MINI_PYTHON}" "${AGENT_ROOT}/swebench_agentdiet_runner.py" \
  --subset verified \
  --split test \
  --workers 1 \
  --filter "${INSTANCE_FILTER}" \
  --output "${OUTPUT_DIR}" \
  -c swebench_xml.yaml \
  -c "${AGENT_ROOT}/swebench_prompt_override.yaml" \
  -c "${DYNAMIC_CONFIG}"

if [[ ! -s "${OUTPUT_DIR}/selector_metrics.jsonl" ]]; then
  echo "ERROR: Mini-SWE completed without any online model turn" >&2
  exit 3
fi

echo "OUTPUT_DIR=${OUTPUT_DIR}"
