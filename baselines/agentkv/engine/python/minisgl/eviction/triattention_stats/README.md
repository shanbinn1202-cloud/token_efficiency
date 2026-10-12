# TriAttention calibration stats (not bundled)

The TriAttention baseline needs offline calibration statistics (per-layer, per-head query
statistics) that are **third-party artifacts** produced by the official TriAttention project.
They are **not distributed** with this repository.

To run any `triattn-*` eviction plan:

1. Download the official TriAttention stats for your model from the official TriAttention
   repository, e.g. `qwen3_8b.pt` (Qwen3-8B) and `qwen3_32b_int4_stats.pt` (Qwen3-32B).
2. Point the engine at them by either:
   - setting `MINISGL_TRIATTN_STATS_PATH=/path/to/stats.pt`, **or**
   - dropping the files in this directory using these exact names:
       - 36-layer model (Qwen3-8B):  `qwen3_8b.pt`
       - 64-layer model (Qwen3-32B): `qwen3_32b_int4_stats.pt`

Without these files, `triattn-*` plans raise a descriptive error. All other eviction plans
(identity/snapkv/streamingllm/rkv/agentkv) need no external stats.
