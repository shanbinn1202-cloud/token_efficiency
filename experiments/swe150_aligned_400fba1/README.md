# Qwen3.6-27B on SWE-bench Verified 150

This directory records the four configurations used with the selective-prefix vLLM 0.24.0 stack. The complete upstream source in `source/` is pinned to `400fba1ca414904d8a48222de4315ff678e18446`. Its runtime is `2fb7f6241fe00517a472f8703125de153ec715d2`. The local AgentKV overlay and Claude Code adapter are included below. [Machine-readable configuration](release.json) · [Cluster file hashes](files.json)

| Method | Version and behavior |
| --- | --- |
| Random | Upstream `random_fill` selects **16-token pages**. It keeps the required initial-prompt and recent pages, then fills the available budget with a seeded random selection. The active budget is 8,192 tokens and the seed is 42. |
| Sliding Window | Upstream `recent_fill` uses the same page size, protection rules and budget. It fills the remaining capacity with the most recent eligible pages. |
| AgentKV | Per-KV-head StageQSnapKV with `AgentKVConnector`, applied as a hash-pinned overlay on the same vLLM release. The budget is 8,192 tokens, the eviction interval is 128, and the sink, recent window and query-anchor budgets are each 32. [Adapter and settings](engines_adapter/) |
| Claude Code compression | Mini-SWE-Agent calls the official Claude Code **2.1.325** `/compact` command when the rendered input exceeds 8,192 tokens at the start of a turn. The same Qwen service generates the summary and the subsequent agent response. Summary tokens are counted separately. [Compaction adapter](method_support/official_compaction.py) |

The experiments share the first 150 Verified tasks in the pinned order (22 Astropy and 128 Django), 24 concurrent sessions, asynchronous scheduling, a 262,144-token context, a 16,384-token batch limit and a 25 GiB GPU KV pool. Thinking is enabled, temperature is 0, the seed is 42, and each response is capped at 4,096 tokens. Formal runs allow 120 steps per task. On DeltaAI, vLLM and the ARM64 Apptainer workers run in two disjoint 16-core steps within a single 145 GiB GH200 allocation. End-to-end output throughput uses main-model completion tokens over the agent evaluation interval. [Timing, memory and launch details](RUN.md)

## Interactive qualification

The November 17 interactive run passed for **Sliding Window** and for the independent reference-patch scoring control. Random completed 24/24 short agent trials and exercised page eviction, then parallel grading encountered a truncated ZIP header. AgentKV's forced-compaction probe generated 16,647 input tokens and stopped at its 16,384-token probe bound. Claude Code's history probe observed token deltas of 0 and 18 (expected 14 and 4), before reaching the live `/compact` probe. The four-method gate therefore failed. These checks do not supply accuracy or throughput for the formal 150-task runs. [Qualification details](qualification.json)

## Inspect or restore the snapshot

Run these commands in a clone of this repository.

```bash
cd experiments/swe150_aligned_400fba1
python3 verify_snapshot.py
```

To recreate the two benchmark inputs, run the commands below in an environment with Hugging Face `datasets`. For an offline export, add `--dataset-json /path/to/verified-test.json`. The builder checks both generated files against the original SHA-256 digests.

```bash
python3 build_assets.py --output-dir assets --roster source/examples/swebench/verified-150.txt
python3 verify_snapshot.py --assets assets
```

The DeltaAI launch scripts are preserved at their measured cluster paths. They expect the site's pinned Qwen checkpoint, the prepared Python environment archive listed in `runtime-archive.json`, and the 150 ARM64 SIF images listed in `local_images/manifest.json`. The client also imports its frozen SWE-64 support tree. `method_support/` contains the local source changes and a pinned restoration tool for that tree, including the official Claude Code executable. Runtime dependency manifests are in `runtime-dependencies/`. New runs need their own interactive qualification receipt and matching Slurm dependencies. [Support installation](method_support/SUPPORT.md) · [Interactive checks](interactive-check/) · [Production launcher](benchmark.py)
