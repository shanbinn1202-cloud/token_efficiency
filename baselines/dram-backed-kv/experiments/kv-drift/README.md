# KV-drift compatibility regression

This retained regression validates the legacy Qwen numerical replay/runtime bridge. It is independent of Goal maintenance and of the removed Goal embedding semantic-drift signal.

Entry points:

- `scripts/run_13579_regression_gh200.sbatch`
- `scripts/run_13579_restored_gh200.sbatch`
- compatibility adapter: `src/dram_kv/adapters/legacy_kv_drift.py`
- validated result: [`result_2026-08-31.md`](result_2026-08-31.md)

The checked-in `results/job3056285/` and `results/job3056342/` JSONL files are retained golden evidence for this historical numerical path.
