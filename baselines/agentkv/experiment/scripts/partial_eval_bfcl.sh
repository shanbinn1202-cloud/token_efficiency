#!/usr/bin/env bash
# Score one BFCL multi_turn_base run.
# Usage: RESULT_DIR=<the run results/ dir> [BFCL_MODELS=...] bash experiment/scripts/partial_eval_bfcl.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${RESULT_DIR:?set RESULT_DIR to the run results/ dir}"
RESULT_DIR="$(cd "$RESULT_DIR" && pwd)"   # bfcl_eval resolves --result-dir relative to its repo; pass absolute
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$ROOT/gorilla/berkeley-function-call-leaderboard${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m bfcl_eval evaluate \
  --model "${BFCL_MODELS:-qwen3-8b-minisglang-prompt}" \
  --test-category "${BFCL_TEST_CATEGORY:-multi_turn_base}" \
  --result-dir "$RESULT_DIR" --score-dir "$RESULT_DIR/../scores" --partial-eval
