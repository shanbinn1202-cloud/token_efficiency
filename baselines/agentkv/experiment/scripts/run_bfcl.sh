#!/usr/bin/env bash
# Run BFCL against one mini-sglang server with a given eviction plan.
# Env overrides: PYTHON_BIN, MODEL, TASK, EVICTION_PLAN, MEMORY_RATIO, EXPERIMENT_NAME.
# Example (no-compression baseline):
#   MODEL=qwen3-8b EVICTION_PLAN=none TASK=multi_turn_base bash experiment/scripts/run_bfcl.sh
# Example (AgentKV @2048):
#   MODEL=qwen3-8b EVICTION_PLAN=agentkv-2048 \
#     TASK=multi_turn_base bash experiment/scripts/run_bfcl.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-agentkv_bfcl}"
MODEL="${MODEL:-qwen3-8b}"
TASK="${TASK:-multi_turn_base}"
EVICTION_PLAN="${EVICTION_PLAN:-none}"

echo "[run_bfcl] model=$MODEL task=$TASK eviction_plan=$EVICTION_PLAN"

exec "$PYTHON_BIN" -u -m experiment.runners.bfcl run \
  --experiment "$EXPERIMENT_NAME" \
  --model "$MODEL" \
  --task "$TASK" \
  --eviction-plan "$EVICTION_PLAN" \
  --memory-ratio "${MEMORY_RATIO:-0.7}" \
  "$@"
