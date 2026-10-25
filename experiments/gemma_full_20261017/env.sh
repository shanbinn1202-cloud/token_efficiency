#!/bin/bash
source /projects/bcjw/ssun4/vllm024_triton_six_20261008/env.sh
export GEMMA_BUNDLE=/projects/bcjw/ssun4/token_efficiency/experiments/gemma_full_20261017
export TEBENCH_VLLM_ROOT=/projects/bcjw/ssun4/token_efficiency/engines/vllm_024_gemma_20261017
export PYTHONPATH="$TEBENCH_VLLM_ROOT:$GEMMA_BUNDLE/harness:/projects/bcjw/ssun4/token_efficiency/mini-swe-agent/src:/projects/bcjw/ssun4/token_efficiency/external/SWE-bench:/projects/bcjw/ssun4/token_efficiency/experiments/agentkv_qwen36/vendor"
export PYTHONDONTWRITEBYTECODE=1
