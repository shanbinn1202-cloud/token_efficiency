from __future__ import annotations

import argparse
import json
import os
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
    slugify_name,
    timestamp,
    write_manifest,
)

def task_summary(split: str, domains: list[str]) -> list[dict]:
    rows = []
    for domain in domains:
        split_path = ROOT / "tau2-bench" / "data" / "tau2" / "domains" / domain / "split_tasks.json"
        task_path = ROOT / "tau2-bench" / "data" / "tau2" / "domains" / domain / "tasks.json"
        splits = json.loads(split_path.read_text())
        tasks = json.loads(task_path.read_text())
        if split not in splits:
            raise RuntimeError(f"{domain} has no split {split}")
        rows.append({"domain": domain, "split": split, "tasks": len(set(splits[split])), "total": len(tasks)})
    return rows


def run_tau2_once(
    *,
    experiment: str,
    run: str,
    domain: str,
    split: str,
    agent_llm: str,
    base_url: str,
    log_dir: Path,
    code_root: Path,
    agent: str,
    user_llm: str,
    max_concurrency: str,
    timeout: str,
    num_tasks: str | None,
) -> int:
    env_file = ensure_env_file()
    file_env = load_env_file(env_file)
    if not file_env.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is not set in {env_file}")
    env = os.environ.copy()
    env.update(file_env)
    tau2_repo = code_root / "tau2-bench"
    env["PYTHONPATH"] = str(tau2_repo / "src") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["TAU2_DATA_DIR"] = str(ROOT / "tau2-bench" / "data")
    result_dir = ROOT / "tau2-bench" / "data" / "simulations" / experiment / run
    run_dir = log_dir.parent
    result_link = run_dir / "tau2_results"
    log_dir.mkdir(parents=True, exist_ok=True)
    if result_link.exists() and not result_link.is_symlink():
        raise RuntimeError(f"result link path exists and is not a symlink: {result_link}")
    result_link.unlink(missing_ok=True)
    result_link.symlink_to(result_dir)
    agent_args = {"temperature": 0, "api_base": base_url, "api_key": "EMPTY", "timeout": 600, "max_tokens": 8192}
    user_args = {"temperature": 0}
    cmd = [
        str(Path(sys.executable).with_name("tau2")),
        "run",
        "--domain",
        domain,
        "--task-split-name",
        split,
        "--agent",
        agent,
        "--agent-llm",
        agent_llm,
        "--agent-llm-args",
        json.dumps(agent_args, separators=(",", ":")),
        "--user-llm",
        user_llm,
        "--user-llm-args",
        json.dumps(user_args, separators=(",", ":")),
        "--num-trials",
        "1",
        "--max-concurrency",
        max_concurrency,
        "--timeout",
        timeout,
        "--save-to",
        f"{experiment}/{run}",
        "--auto-resume",
        "--verbose-logs",
        "--llm-log-mode",
        "all",
        "--log-level",
        "INFO",
    ]
    if num_tasks:
        cmd[cmd.index("--num-trials"):cmd.index("--num-trials")] = ["--num-tasks", num_tasks]
    print(f"[tau2] run={run} domain={domain} agent_llm={agent_llm} base_url={base_url}")
    return run_logged(cmd, log_dir / "tau2.log", cwd=tau2_repo, env=env)


def run_model_domains(
    *,
    experiment: str,
    run_root: str,
    model_name: str,
    gpu: str | None,
    split: str,
    domains: list[str],
    memory_ratio: str,
    max_seq_len: str,
    max_extend_tokens: str | None,
    parallel_sequences: str,
    max_running_requests: str,
    cuda_graph_max_bs: str,
    page_size: str | None,
    eviction_plan: str,
    compact_interval: str,
    compact_after_prefill: bool,
    compact_before_park: bool,
    agent: str,
    user_llm: str,
    max_concurrency: str,
    timeout: str,
    num_tasks: str | None,
    job_timestamp: str | None,
) -> int:
    if parallel_sequences:
        max_concurrency = parallel_sequences
        max_running_requests = parallel_sequences
        cuda_graph_max_bs = parallel_sequences
    spec = model_spec(model_name)
    root_run_dir = RUNS_DIR / experiment / run_root
    root_logs = root_run_dir / "logs"
    root_run_dir.mkdir(parents=True, exist_ok=True)
    code_root = ROOT
    jobs = {
        domain: {
            "name": (
                "_".join(
                    [
                        job_timestamp,
                        *(slugify_name(part) for part in (model_name, domain, split)),
                    ]
                )
                if job_timestamp
                else auto_job_name(model_name, domain, split)
            ),
            "domain": domain,
            "split": split,
        }
        for domain in domains
    }
    write_manifest(
        root_run_dir / "run_manifest.json",
        {
            "kind": "tau2_run",
            "experiment": experiment,
            "run": run_root,
            "jobs_dir": str(root_run_dir / "jobs"),
        },
    )
    print(f"[model-run] run_root={run_root} model={model_name} gpu={gpu or 'unset'} domains={' '.join(domains)}")
    status = 0
    with MiniSGLangServer(
        model_name=model_name,
        log_path=root_logs / f"minisglang_{timestamp()}_{model_name}_{split}.log",
        code_root=code_root,
        cuda_visible_devices=gpu,
        memory_ratio=memory_ratio,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        max_seq_len=max_seq_len,
        max_extend_tokens=max_extend_tokens,
        page_size=page_size,
        eviction_plan=eviction_plan,
        compact_interval=compact_interval,
        compact_after_prefill=compact_after_prefill,
        compact_before_park=compact_before_park,
    ) as server:
        for domain in domains:
            job = jobs[domain]
            job_name = job["name"]
            job_dir = root_run_dir / "jobs" / job_name
            result_dir = ROOT / "tau2-bench" / "data" / "simulations" / experiment / run_root / job_name
            write_manifest(
                job_dir / "job_manifest.json",
                {
                    "kind": "tau2_job",
                    "experiment": experiment,
                    "run": run_root,
                    "job": job_name,
                    "model": model_name,
                    "gpu": gpu,
                    "domain": domain,
                    "domain_summary": task_summary(split, [domain])[0],
                    "result_dir": str(result_dir),
                    "server": {
                        "memory_ratio": memory_ratio,
                        "max_seq_len": max_seq_len,
                        "max_extend_tokens": max_extend_tokens or max_seq_len,
                        "attention_backend": "fi",
                        "max_running_requests": max_running_requests,
                        "cuda_graph_max_bs": cuda_graph_max_bs,
                        "page_size": page_size,
                        "eviction_plan": eviction_plan,
                        "compact_interval": compact_interval,
                        "compact_after_prefill": compact_after_prefill,
                        "compact_before_park": compact_before_park,
                    },
                    "tau2": {
                        "agent": agent,
                        "user_llm": user_llm,
                        "max_concurrency": max_concurrency,
                        "timeout": timeout,
                        "num_tasks": num_tasks,
                        "split": split,
                    },
                },
            )
            status = status or run_tau2_once(
                experiment=experiment,
                run=f"{run_root}/{job_name}",
                domain=domain,
                split=split,
                agent_llm=spec.tau2_agent_llm,
                base_url=server.base_url,
                log_dir=job_dir / "logs",
                code_root=code_root,
                agent=agent,
                user_llm=user_llm,
                max_concurrency=max_concurrency,
                timeout=timeout,
                num_tasks=num_tasks,
            )
    return status


def _add_common_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--memory-ratio", default="0.75")
    parser.add_argument("--max-seq-len", default="40960")
    parser.add_argument("--max-extend-tokens", default=None)
    parser.add_argument("--parallel-sequences", default="8")
    parser.add_argument("--max-running-requests", default="16")
    parser.add_argument("--cuda-graph-max-bs", default="16")
    parser.add_argument("--page-size", default=None)
    parser.add_argument("--eviction-plan", default="none")
    parser.add_argument("--compact-interval", default="128")
    parser.add_argument("--compact-after-prefill", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compact-before-park", action=argparse.BooleanOptionalAction, default=False)


def run_from_args(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run Tau2 domains against one mini-sglang server.")
    parser.add_argument("--experiment", default="tau2_minisglang_run")
    parser.add_argument("--run", default=timestamp())
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="base")
    parser.add_argument("--domains", nargs="+", default=["airline"])
    parser.add_argument("--agent", default="qwen3_preserve_think_llm_agent")
    parser.add_argument("--user-llm", default="openai/gpt-4.1-mini")
    parser.add_argument("--max-concurrency", default="8")
    parser.add_argument("--timeout", default="900")
    parser.add_argument("--num-tasks", default=None)
    parser.add_argument("--job-timestamp", default=None)
    _add_common_server_args(parser)
    args = parser.parse_args(argv)
    return run_model_domains(
        experiment=args.experiment,
        run_root=args.run,
        model_name=args.model,
        gpu=args.gpu,
        split=args.split,
        domains=args.domains,
        memory_ratio=args.memory_ratio,
        max_seq_len=args.max_seq_len,
        max_extend_tokens=args.max_extend_tokens,
        parallel_sequences=args.parallel_sequences,
        max_running_requests=args.max_running_requests,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        page_size=args.page_size,
        eviction_plan=args.eviction_plan,
        compact_interval=args.compact_interval,
        compact_after_prefill=args.compact_after_prefill,
        compact_before_park=args.compact_before_park,
        agent=args.agent,
        user_llm=args.user_llm,
        max_concurrency=args.max_concurrency,
        timeout=args.timeout,
        num_tasks=args.num_tasks,
        job_timestamp=args.job_timestamp,
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    command = argv[0] if argv else "run"
    if command == "run":
        return run_from_args(argv[1:])
    raise SystemExit(f"unknown tau2 command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
