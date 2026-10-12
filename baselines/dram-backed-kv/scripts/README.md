# Script index

Scripts remain in one directory for now because several Slurm launchers and tests refer to them by path. Use these groups when choosing an entry point.

## Current online path

- `serve_online_selector_v2.py`: online OpenAI-compatible server and session orchestrator.
- `run_online_selector_v2_swe_h200.sbatch`: one instance/mode on H200.
- `run_online_selector_v2_cohort2_triplet_h200.sbatch`: current five-instance, three-mode cohort wrapper.
- `qwen_goal_rewriter.py`: persistent Qwen3-0.6B structured Retrieval Goal service.

Completed pilot, smoke, one-instance retry and earlier cohort Slurm wrappers were removed after their job IDs and outcomes were captured in result summaries. They remain recoverable from Git history.

## Retained validation

- `run_13579_regression_gh200.sbatch`, `run_13579_restored_gh200.sbatch`: KV-drift compatibility regression.
- `validate_qwen_dram_runtime.py`, `validate_qwen_forward_bridge.py` and matching Slurm launchers: runtime/forward integration checks.
- `run_oracle_budget_sweep.py`: retained because recorded chat-renderer regression tests import its canonical renderer.

## Historical offline support

The dependency-free Shadow CLI remains available through `python -m dram_kv.cli`. One-off recorded-Oracle, frozen-Goal, embedding-corpus, retention and proxy-replay pipelines were removed after their outputs were summarized; recover them from Git history or the validated V2 tag when historical reproduction is required.
