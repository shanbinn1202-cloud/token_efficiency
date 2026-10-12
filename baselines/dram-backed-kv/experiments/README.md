# Experiment map

| Experiment | Status | Entry point | Results |
|---|---|---|---|
| Online Selector V2 | current | `scripts/run_online_selector_v2_swe_h200.sbatch`; cohort wrapper `scripts/run_online_selector_v2_cohort2_triplet_h200.sbatch` | `results/current/` and ignored `results/raw/` |
| KV-drift compatibility | retained regression | [`kv-drift/README.md`](kv-drift/README.md) | `experiments/kv-drift/result_2026-08-31.md` |
| Recorded replay / Selector V1 | archived baseline | see `scripts/README.md` | historical summaries in `results/` and docs in `docs/archive/` |

“KV-drift compatibility” checks numerical replay/runtime behavior. It is unrelated to the removed Goal embedding semantic-drift gate.
