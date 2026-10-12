"""Reject incomparable or failed runs before producing a metric table."""
import argparse
from copy import deepcopy
import json
from pathlib import Path


def alignment(run):
    c = run["configuration"]
    return {"model": c["model"], "generation": c["generation"], "dataset": c["dataset"],
        "workers": c["evaluation"]["workers"], "image_manifest": c["evaluation"]["image_manifest"],
        "timeout": c["evaluation"]["timeout"], "kv_budget": c["cache"]["kv_budget"],
        "tp": c["backend"]["tensor_parallel_size"], "instance_ids": run["instance_ids"],
        "prefix_caching": c["backend"]["prefix_caching"],
        "execution": {k: c["backend"][k] for k in ("cuda_graph", "torch_compile", "full_attention", "sdpa_kernel", "linear_attention", "prefill_chunk_size")},
        "model_files": run.get("model_files_sha256"), "architecture": run.get("architecture"),
        "versions": run.get("versions"),
        "external": run["external"], "sources": run["source_sha256"]}


def compare(paths):
    runs = [(p, json.loads((p / "run.json").read_text()), json.loads((p / "metrics.json").read_text())) for p in map(Path, paths)]
    reference = alignment(runs[0][1])
    policies = {}
    runtime_reference = None
    for p, run, metrics in runs:
        if not metrics["eligible_for_comparison"]:
            raise ValueError(f"{p}: failed/incomplete run is not eligible for comparison")
        current = alignment(run)
        mismatch = [k for k in reference if reference[k] != current[k]]
        if mismatch:
            raise ValueError(f"{p}: comparison settings differ: {mismatch}")
        runtime_path = p / "runtime.json"
        if run["configuration"]["backend"]["name"] == "transformers" and runtime_path.exists():
            runtime = json.loads(runtime_path.read_text())
            if runtime_reference is None:
                runtime_reference = runtime
            elif runtime != runtime_reference:
                raise ValueError(f"{p}: actual runtime kernels, precision or GPU differ")
        method = run["method"]
        if method in policies and policies[method] != run["policy"]:
            raise ValueError(f"{p}: method policy differs across runs/backends")
        policies[method] = run["policy"]
    keys = ("accuracy", "e2e_latency_seconds_mean", "throughput_instances_per_hour",
            "inference_output_tokens_per_second", "avg_turns", "avg_context_tokens")
    print("| Method | Backend | Accuracy | E2E s/task | Tasks/hour | Model output tok/s | Avg turns | Avg context tokens |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, run, metrics in runs:
        values = [f"{metrics[k]:.4f}" if metrics[k] is not None else "N/A" for k in keys]
        print("| " + " | ".join([run["method"], run["configuration"]["backend"]["name"], *values]) + " |")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    compare(p.parse_args().runs)


if __name__ == "__main__":
    main()
