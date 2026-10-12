# DRAM-Backed KV: Online Selector V2

This repository's active path is an online 16-token Page working set for Qwen 27B. Sealed KV history is backed by CPU DRAM; a `SelectionPlan` restores the active subset to GPU HBM for a fresh Mini-SWE-Agent rollout.

The controlled modes are full context, observation-hidden-state query, and cached Retrieval-Goal-hidden-state query. Start with the [Chinese README](README.zh-CN.md) and the [current architecture document](docs/current/Online_Selector_V2_Code_Architecture_2026-09-06.zh-CN.md).

```bash
PYTHONPATH=src python3.12 -m unittest discover -s tests -v
```

Historical recorded-replay and Selector V1 documents are retained under `docs/archive/`.
