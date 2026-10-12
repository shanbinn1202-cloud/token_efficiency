"""Unified method × backend runner; model loading and grading excluded from E2E."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import time
import uuid

from tebench.config import ROOT, load, runner_config
from tebench.methods import METHODS, check_support, create_model, policy_snapshot


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="Optional YAML overlay on configs/default.yaml")
    p.add_argument("--method", choices=METHODS, default="vanilla")
    p.add_argument("--backend", choices=("transformers", "vllm"))
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Validate and print fully resolved configuration; no GPU or dataset access")
    p.add_argument("--smoke", action="store_true", help="One 32-token request; no benchmark output written")
    p.add_argument("--list", action="store_true", help="Print actual backend capabilities")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    os.chdir(ROOT)
    config = load(args.config, args.set + ([f"backend.name={args.backend}"] if args.backend else []))
    if args.list:
        for method in METHODS:
            print(f"{method:20} transformers={'no' if method in {'repairkv', 'intentkv'} else 'yes'}  vllm=yes")
        return
    check_support(args.method, config)
    resolved_policy = policy_snapshot(args.method, config)
    if args.dry_run:
        print(json.dumps({"method": args.method, "policy": resolved_policy, "configuration": config}, indent=2))
        return
    if args.smoke:
        from tebench.backends import create
        backend = create(config, args.method)
        try:
            result = backend.complete(model=config["model"]["name"], messages=[
                {"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "Reply OK."}], max_tokens=32)
            print(result.model_dump_json(indent=2))
        finally:
            backend.close()
        return
    from datasets import load_dataset
    from baselines.common.environment import resolve_instance_image
    from baselines.common.runner import run_instance, save_json
    from baselines.common.harness import run_apptainer_harness
    from tebench.metrics import aggregate
    from tebench.provenance import snapshot

    output = args.output or ROOT / config["evaluation"]["output_root"] / f"{args.method}__{config['backend']['name']}"
    output = output.resolve()
    if any(output == ROOT / x or ROOT / x in output.parents for x in ("baselines", "mini-swe-agent", "external", "adapters", "configs", "tebench", "experiments")):
        raise ValueError("Output must not be placed inside source or copied dependency directories")
    rc = runner_config(config, output)
    data = config["dataset"]
    rows = list(load_dataset(data["name"], split=data["split"]))[data["start"]:data["start"] + data["count"]]
    if len(rows) != data["count"]:
        raise ValueError("Dataset contains fewer instances than requested")
    ids = [r["instance_id"] for r in rows]
    for row in rows:
        row["image_name"] = resolve_instance_image(row, "", rc.evaluation.apptainer_image_manifest)
    current = snapshot(config, args.method, ids)
    saved = output / "run.json"
    if saved.exists():
        old = json.loads(saved.read_text())
        if old != current:
            raise ValueError("Configuration, policy, sources, instance IDs or package versions changed; use a new output directory")
        if not (args.resume or args.eval_only):
            raise ValueError("Output already exists; use --resume or a new directory")
    elif args.resume or args.eval_only:
        raise ValueError("--resume/--eval-only requires a previous run.json")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is nonempty and has no run.json")
    output.mkdir(parents=True, exist_ok=True)
    if not saved.exists():
        save_json(saved, current)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(output / "runner.log")], force=True)
    os.environ.setdefault("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "3")
    pred_path, time_path = output / "predictions.json", output / "timings.json"
    predictions = {x["instance_id"]: x for x in json.loads(pred_path.read_text())} if pred_path.exists() else {}
    timing = json.loads(time_path.read_text()) if time_path.exists() else {"instances": {}, "segments": {}}
    if args.eval_only and (not pred_path.exists() or not time_path.exists()):
        raise ValueError("--eval-only needs saved predictions and timings")
    pending = [r for r in rows if predictions.get(r["instance_id"], {}).get("exit_status", "error") == "error"]
    if pending and not args.eval_only:
        from tebench.backends import create
        backend = create(config, args.method)
        try:
            save_json(output / "runtime.json", getattr(backend, "execution_info", {}))
            # Warmup uses the same runtime but is excluded from measured instance latency.
            backend.complete(model=config["model"]["name"], messages=[{"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "Reply OK."}], max_tokens=2)
            backend.reset()
            segment, started = uuid.uuid4().hex, time.perf_counter()
            def run(row):
                start = time.perf_counter()
                # vLLM is stateless client transport; local KV adapters must reset per instance.
                if config["backend"]["name"] == "transformers":
                    backend.reset()
                def factory(**kwargs):
                    return create_model(args.method, config, backend, **kwargs)
                try:
                    result = run_instance(row, rc, output, model_factory=factory)
                    return result, time.perf_counter() - start
                finally:
                    if config["backend"]["name"] == "vllm":
                        backend.reset()
            with ThreadPoolExecutor(max_workers=config["evaluation"]["workers"]) as pool:
                futures = [pool.submit(run, row) for row in pending]
                for future in as_completed(futures):
                    prediction, elapsed = future.result()
                    iid = prediction["instance_id"]
                    predictions[iid] = prediction
                    timing["instances"][iid] = {"seconds": elapsed, "segment": segment}
                    timing["segments"][segment] = time.perf_counter() - started
                    save_json(pred_path, [predictions[i] for i in ids if i in predictions])
                    save_json(time_path, timing)
                    print(f"{len(predictions)}/{len(ids)} {iid}: {prediction['exit_status']} ({elapsed:.1f}s)", flush=True)
        finally:
            backend.close()
    run_apptainer_harness(predictions=list(predictions.values()), dataset_name=data["name"], split=data["split"],
        instance_ids=ids, max_workers=rc.evaluation.swebench_max_workers, timeout=rc.evaluation.instance_timeout,
        run_id=output.name, report_path=output / "report.json", executable=rc.evaluation.apptainer_executable,
        image_manifest=rc.evaluation.apptainer_image_manifest)
    metrics = aggregate(output, ids)
    save_json(output / "metrics.json", metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != "cases"}, indent=2))
    if not metrics["eligible_for_comparison"]:
        raise SystemExit("Run has missing data or infrastructure/request errors; inspect metrics.json before comparison")


if __name__ == "__main__":
    main()
