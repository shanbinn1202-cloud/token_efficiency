#!/usr/bin/env bash
# Run tau2-bench domains against one mini-sglang server with a given eviction plan.
# Env overrides: PYTHON_BIN, MODEL, DOMAINS, EVICTION_PLAN, PARALLEL_SEQUENCES, EXPERIMENT_NAME.
# The tau2 user-simulator (--user-llm) can point at the same local server, so no external
# API key is required for a self-contained run.
# Example:
#   MODEL=qwen3-8b DOMAINS=airline EVICTION_PLAN=agentkv-512 \
#     bash experiment/scripts/run_tau2.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-agentkv_tau2}"
MODEL="${MODEL:-qwen3-8b}"
DOMAINS="${DOMAINS:-airline}"
EVICTION_PLAN="${EVICTION_PLAN:-none}"
PARALLEL_SEQUENCES="${PARALLEL_SEQUENCES:-16}"
read -r -a DOMAIN_ARGS <<< "$DOMAINS"

echo "[run_tau2] model=$MODEL domains=$DOMAINS eviction_plan=$EVICTION_PLAN"

exec "$PYTHON_BIN" -u -m experiment.runners.tau2 run \
  --experiment "$EXPERIMENT_NAME" \
  --model "$MODEL" \
  --memory-ratio "${MEMORY_RATIO:-0.75}" \
  --eviction-plan "$EVICTION_PLAN" \
  --parallel-sequences "$PARALLEL_SEQUENCES" \
  --split base \
  --domains "${DOMAIN_ARGS[@]}" \
  "$@"
