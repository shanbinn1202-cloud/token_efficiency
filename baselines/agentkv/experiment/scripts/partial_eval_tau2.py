#!/usr/bin/env python3
"""Print the reward summary for one tau2 run.

Usage: python experiment/scripts/partial_eval_tau2.py <path/to/results.json>
(results.json is written under tau2-bench/data/simulations/<experiment>/<run>/.../).
"""
import json
import sys

data = json.loads(open(sys.argv[1]).read())
sims = data.get("simulations") or []
rewards = [
    (s.get("reward_info") or {}).get("reward")
    for s in sims
    if isinstance((s.get("reward_info") or {}).get("reward"), (int, float))
]
total = len(data.get("tasks") or []) or len(sims)
avg = sum(rewards) / len(rewards) if rewards else 0.0
print(f"scored={len(rewards)}/{total}  avg_reward={avg:.4f}  solved~{avg * total:.0f}/{total}")
