from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from .common import (
    ROOT,
    RUNS_DIR,
    MiniSGLangServer,
    auto_job_name,
    ensure_env_file,
    load_env_file,
    model_spec,
    run_logged,
    timestamp,
    write_manifest,
)


def run_generate(
    *,
    bfcl_model: str,
    task: str,
    result_dir: Path,
    base_url: str,
    log_path: Path,
    code_root: Path,
    num_threads: str,
    web_search_backend: str,
    run_ids: bool,
    include_input_log: bool,
) -> int:
    env_file = ensure_env_file()
    env = load_env_file(env_file)
    if not env.get("SERPAPI_API_KEY") and env.get("SERP_API_KEY"):
        env["SERPAPI_API_KEY"] = env["SERP_API_KEY"]
    env["BFCL_WEB_SEARCH_BACKEND"] = web_search_backend
    env["REMOTE_OPENAI_BASE_URL"] = base_url
    env["REMOTE_OPENAI_API_KEY"] = "EMPTY"
    repo = code_root / "gorilla" / "berkeley-function-call-leaderboard"
    env["PYTHONPATH"] = str(repo) + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bfcl] generate model={bfcl_model} task={task} backend={env['BFCL_WEB_SEARCH_BACKEND']}")
    cmd = [
        sys.executable,
        "-m",
        "bfcl_eval",
        "generate",
        "--model",
        bfcl_model,
        "--test-category",
        task,
        "--num-threads",
        num_threads,
        "--skip-server-setup",
        "--result-dir",
        str(result_dir),
        "--allow-overwrite",
    ]
    if run_ids:
        cmd.append("--run-ids")
    if include_input_log:
        cmd.append("--include-input-log")
    return run_logged(
        cmd,
        log_path,
        cwd=repo,
        env=env,
    )


def run_job(
    *,
    experiment: str,
    run: str,
    job_name: str,
    model_name: str,
    task: str,
    gpu: str | None,
    memory_ratio: str,
    max_seq_len: str,
    max_extend_tokens: str | None,
    max_running_requests: str,
    cuda_graph_max_bs: str,
    eviction_plan: str,
    compact_interval: str,
    compact_after_prefill: bool,
    compact_before_park: bool,
    bfcl_test_ids_file: str | None,
    include_input_log: bool,
    num_threads: str,
    web_search_backend: str,
) -> int:
    spec = model_spec(model_name)
    run_dir = RUNS_DIR / experiment / run
    logs = run_dir / "logs"
    result_dir = run_dir / "results"
    logs.mkdir(parents=True, exist_ok=True)
    code_root = ROOT
    if bfcl_test_ids_file:
        src = Path(bfcl_test_ids_file)
        if not src.is_absolute():
            src = ROOT / src
        if not src.is_file():
            raise RuntimeError(f"missing BFCL test ids file: {src}")
        dst = code_root / "gorilla" / "berkeley-function-call-leaderboard" / "test_case_ids_to_generate.json"
        shutil.copyfile(src, dst)
    write_manifest(
        run_dir / "run_manifest.json",
        {
            "kind": "bfcl",
            "experiment": experiment,
            "run": run,
            "job": job_name,
            "model": model_name,
            "bfcl_model": spec.bfcl_model,
            "task": task,
            "gpu": gpu,
            "result_dir": str(result_dir),
            "server": {
                "memory_ratio": memory_ratio,
                "max_seq_len": max_seq_len,
                "max_extend_tokens": max_extend_tokens or max_seq_len,
                "attention_backend": "fi",
                "max_running_requests": max_running_requests,
                "cuda_graph_max_bs": cuda_graph_max_bs,
                "eviction_plan": eviction_plan,
                "compact_interval": compact_interval,
                "compact_after_prefill": compact_after_prefill,
                "compact_before_park": compact_before_park,
            },
            "bfcl": {
                "num_threads": num_threads,
                "web_search_backend": web_search_backend,
                "run_ids": bool(bfcl_test_ids_file),
                "test_ids_file": bfcl_test_ids_file,
                "include_input_log": include_input_log,
            },
        },
    )
    print(f"[job] name={job_name} model={model_name} gpu={gpu or 'unset'}")
    with MiniSGLangServer(
        model_name=model_name,
        log_path=logs / f"minisglang_{timestamp()}_{job_name}.log",
        code_root=code_root,
        cuda_visible_devices=gpu,
        memory_ratio=memory_ratio,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        max_seq_len=max_seq_len,
        max_extend_tokens=max_extend_tokens,
        eviction_plan=eviction_plan,
        compact_interval=compact_interval,
        compact_after_prefill=compact_after_prefill,
        compact_before_park=compact_before_park,
    ) as server:
        return run_generate(
            bfcl_model=spec.bfcl_model,
            task=task,
            result_dir=result_dir,
            base_url=server.base_url,
            log_path=logs / f"bfcl_{job_name}.log",
            code_root=code_root,
            num_threads=num_threads,
            web_search_backend=web_search_backend,
            run_ids=bool(bfcl_test_ids_file),
            include_input_log=include_input_log,
        )


def _add_common_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--memory-ratio", default="0.75")
    parser.add_argument("--max-seq-len", default="40960")
    parser.add_argument("--max-extend-tokens", default=None)
    parser.add_argument("--max-running-requests", default="16")
    parser.add_argument("--cuda-graph-max-bs", default="16")
    parser.add_argument("--eviction-plan", default="none")
    parser.add_argument("--compact-interval", default="128")
    parser.add_argument("--compact-after-prefill", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compact-before-park", action=argparse.BooleanOptionalAction, default=False)


def run_from_args(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run BFCL against one mini-sglang server.")
    parser.add_argument("--experiment", default="bfcl_minisglang_run")
    parser.add_argument("--run", default=timestamp())
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="web_search_base,multi_turn_base")
    parser.add_argument("--num-threads", default="8")
    parser.add_argument("--web-search-backend", default="serpapi")
    parser.add_argument("--bfcl-test-ids-file", default=None)
    parser.add_argument("--include-input-log", action="store_true")
    _add_common_server_args(parser)
    args = parser.parse_args(argv)
    return run_job(
        experiment=args.experiment,
        run=args.run,
        job_name=auto_job_name(args.model, args.task),
        model_name=args.model,
        task=args.task,
        gpu=args.gpu,
        memory_ratio=args.memory_ratio,
        max_seq_len=args.max_seq_len,
        max_extend_tokens=args.max_extend_tokens,
        max_running_requests=args.max_running_requests,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        eviction_plan=args.eviction_plan,
        compact_interval=args.compact_interval,
        compact_after_prefill=args.compact_after_prefill,
        compact_before_park=args.compact_before_park,
        bfcl_test_ids_file=args.bfcl_test_ids_file,
        include_input_log=args.include_input_log,
        num_threads=args.num_threads,
        web_search_backend=args.web_search_backend,
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    command = argv[0] if argv else "run"
    if command == "run":
        return run_from_args(argv[1:])
    raise SystemExit(f"unknown bfcl command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
