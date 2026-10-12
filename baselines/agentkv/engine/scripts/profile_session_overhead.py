#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from transformers import AutoTokenizer

MINISGL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MINISGL_ROOT.parent
MINISGL_PYTHON = MINISGL_ROOT / "python"
if str(MINISGL_PYTHON) not in sys.path:
    sys.path.insert(0, str(MINISGL_PYTHON))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minisgl.benchmark.client import generate_prompt, get_model_name  # noqa: E402


try:
    from experiment.runners.common import model_spec, timestamp
except Exception:  # pragma: no cover - standalone use outside agentkv
    model_spec = None

    def timestamp() -> str:
        return time.strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class TurnProfile:
    session_index: int
    session_id: str
    turn: int
    input_tokens_estimate: int
    output_tokens_estimate: int
    elapsed_s: float
    request_tok_s: float
    output_chars: int
    error: str | None = None


class MiniSGLangProcess:
    def __init__(self, args: argparse.Namespace, model_path: str, log_dir: Path) -> None:
        self.args = args
        self.model_path = model_path
        self.log_dir = log_dir
        self.proc: subprocess.Popen[str] | None = None
        self.port = args.port or find_free_port_pair(args.port_start, args.port_end)
        self.base_url = f"http://{args.host}:{self.port}/v1"
        self.log_path = log_dir / "server.log"
        self.compact_debug_dir = log_dir / "compact_debug"

    def start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(MINISGL_PYTHON) + (
            ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        if self.args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.args.gpu

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "minisgl",
            "--model-path",
            self.model_path,
            "--host",
            self.args.host,
            "--port",
            str(self.port),
            "--max-running-requests",
            str(self.args.max_running_requests),
            "--cuda-graph-max-bs",
            str(self.args.cuda_graph_max_bs),
            "--max-seq-len-override",
            str(self.args.max_seq_len),
            "--max-extend-length",
            str(self.args.max_extend_tokens),
            "--memory-ratio",
            str(self.args.memory_ratio),
            "--attention-backend",
            self.args.attention_backend,
            "--page-size",
            str(self.args.page_size),
            "--eviction-plan",
            self.args.eviction_plan,
            "--compact-debug-dir",
            str(self.compact_debug_dir),
            "--no-debug-rendered-prompt",
        ]
        if self.args.compact_debug:
            cmd.append("--compact-debug")
        if self.args.compact_debug_jsonl:
            cmd.append("--compact-debug-jsonl")

        with self.log_path.open("w", encoding="utf-8") as log:
            self.proc = subprocess.Popen(
                cmd,
                cwd=MINISGL_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,
            )

        print(f"[profile] mini-sglang pid={self.proc.pid} base_url={self.base_url}")
        print(f"[profile] server_log={self.log_path}")
        self._wait_ready()

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        print(f"[profile] stopping mini-sglang pid={self.proc.pid}")
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=20)
        except Exception:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=10)

    def _wait_ready(self) -> None:
        assert self.proc is not None
        deadline = time.time() + self.args.startup_timeout
        last_error: Exception | None = None
        models_url = self.base_url.rstrip("/") + "/models"
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"mini-sglang exited during startup; see {self.log_path}\n"
                    f"{tail_file(self.log_path)}"
                )
            try:
                with urllib.request.urlopen(models_url, timeout=5) as response:
                    if response.status == 200:
                        print(f"[profile] server ready: {self.base_url}")
                        return
            except Exception as exc:
                last_error = exc
            time.sleep(2)
        raise RuntimeError(
            f"mini-sglang did not become ready: {last_error}; see {self.log_path}\n"
            f"{tail_file(self.log_path)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile mini-sglang session-cache end-to-end generated token throughput."
    )
    parser.add_argument("--base-url", default="", help="Use an existing OpenAI server.")
    parser.add_argument("--model-name", default="", help="Known model name from experiment config.")
    parser.add_argument("--model-path", default="", help="Model/tokenizer path.")
    parser.add_argument("--tokenizer-path", default="", help="Tokenizer path when using --base-url.")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for the server process.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 means auto-select a free port pair.")
    parser.add_argument("--port-start", type=int, default=1919)
    parser.add_argument("--port-end", type=int, default=2999)
    parser.add_argument("--startup-timeout", type=float, default=600)

    parser.add_argument("--eviction-plan", default="identity")
    parser.add_argument(
        "--eviction-plans",
        default="",
        help="Comma-separated plans to run sequentially, e.g. none,identity,streamingllm.",
    )
    parser.add_argument("--attention-backend", default="fi")
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--memory-ratio", type=float, default=0.75)
    parser.add_argument("--max-seq-len", type=int, default=40960)
    parser.add_argument("--max-extend-tokens", type=int, default=40960)
    parser.add_argument("--max-running-requests", type=int, default=8)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=8)
    parser.add_argument("--compact-debug-jsonl", action="store_true")
    parser.add_argument("--compact-debug", action="store_true")

    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=4096)
    parser.add_argument("--turn-prompt-len", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--log-dir", default="")
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> str:
    if args.model_path:
        return args.model_path
    if args.model_name and model_spec is not None:
        return model_spec(args.model_name).path
    raise RuntimeError("--model-path is required unless --model-name is known")


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


def find_free_port_pair(start: int, end: int) -> int:
    for port in range(start, end):
        if port_is_free(port) and port_is_free(port + 1):
            return port
    raise RuntimeError(f"no free mini-sglang port pair found in [{start}, {end}]")


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def tail_file(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


def estimate_chat_tokens(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    try:
        return len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    except Exception:
        text = "\n".join(str(msg.get("content") or "") for msg in messages)
        return len(tokenizer.encode(text, add_special_tokens=False))


def message_raw_content(message: Any) -> str:
    extra = getattr(message, "model_extra", None) or {}
    raw = extra.get("raw_content")
    if raw is not None:
        return raw
    raw = getattr(message, "raw_content", None)
    if raw is not None:
        return raw
    return message.content or ""


async def close_session(client: AsyncOpenAI, model: str, session_id: str, timeout: float) -> None:
    await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": ""}],
            max_tokens=1,
            temperature=0.0,
            extra_body={"session_id": session_id, "close_session": True},
        ),
        timeout=timeout,
    )


async def run_one_session(
    *,
    client: AsyncOpenAI,
    tokenizer: Any,
    model: str,
    args: argparse.Namespace,
    session_index: int,
    prompt: str,
    turn_prompt: str,
) -> list[TurnProfile]:
    session_id = f"profile:{timestamp()}:{os.getpid()}:{session_index}"
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    results: list[TurnProfile] = []
    try:
        for turn in range(args.turns):
            input_tokens = estimate_chat_tokens(tokenizer, messages)
            start = time.perf_counter()
            error = None
            output_text = ""
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        extra_body={
                            "session_id": session_id,
                            "ignore_eos": True,
                            "top_k": 1,
                        },
                    ),
                    timeout=args.timeout,
                )
                output_text = message_raw_content(response.choices[0].message)
            except Exception as exc:
                error = repr(exc)
            elapsed = time.perf_counter() - start
            output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))
            request_tok_s = output_tokens / elapsed if elapsed > 0 else 0.0
            results.append(
                TurnProfile(
                    session_index=session_index,
                    session_id=session_id,
                    turn=turn,
                    input_tokens_estimate=input_tokens,
                    output_tokens_estimate=output_tokens,
                    elapsed_s=elapsed,
                    request_tok_s=request_tok_s,
                    output_chars=len(output_text),
                    error=error,
                )
            )
            if error is not None:
                break
            messages.append({"role": "assistant", "content": output_text})
            if turn + 1 < args.turns:
                messages.append(
                    {
                        "role": "user",
                        "content": f"{turn_prompt}\nTurn {turn + 2}: continue.",
                    }
                )
    finally:
        try:
            await close_session(client, model, session_id, timeout=min(args.timeout, 30))
        except Exception as exc:
            results.append(
                TurnProfile(
                    session_index=session_index,
                    session_id=session_id,
                    turn=-1,
                    input_tokens_estimate=0,
                    output_tokens_estimate=0,
                    elapsed_s=0.0,
                    request_tok_s=0.0,
                    output_chars=0,
                    error=f"close_session_failed: {exc!r}",
                )
            )
    return results


async def run_profile(
    *,
    base_url: str,
    tokenizer_path: str,
    args: argparse.Namespace,
) -> tuple[list[TurnProfile], str]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompt = generate_prompt(tokenizer, args.prompt_len)
    turn_prompt = generate_prompt(tokenizer, args.turn_prompt_len)
    async with AsyncOpenAI(base_url=base_url, api_key="EMPTY") as client:
        model = await get_model_name(client)
        semaphore = asyncio.Semaphore(args.concurrency)

        async def run_limited(session_index: int) -> list[TurnProfile]:
            async with semaphore:
                return await run_one_session(
                    client=client,
                    tokenizer=tokenizer,
                    model=model,
                    args=args,
                    session_index=session_index,
                    prompt=prompt,
                    turn_prompt=turn_prompt,
                )

        nested = await asyncio.gather(*(run_limited(i) for i in range(args.sessions)))
    return [item for group in nested for item in group], model


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def summarize_turns(results: list[TurnProfile]) -> dict[str, Any]:
    request_results = [result for result in results if result.turn >= 0]
    ok = [result for result in request_results if result.error is None]
    errors = [result for result in results if result.error is not None]
    elapsed = [result.elapsed_s for result in ok]
    output_tokens = sum(result.output_tokens_estimate for result in ok)
    input_tokens = sum(result.input_tokens_estimate for result in ok)
    duration = max(elapsed) if elapsed else 0.0
    return {
        "requests": len(request_results),
        "successful_requests": len(ok),
        "errors": len(errors),
        "input_tokens_estimate": input_tokens,
        "output_tokens_estimate": output_tokens,
        "avg_request_s": sum(elapsed) / len(elapsed) if elapsed else None,
        "p50_request_s": percentile(elapsed, 0.50),
        "p90_request_s": percentile(elapsed, 0.90),
        "max_request_s": max(elapsed) if elapsed else None,
        "mean_request_output_tok_s": (
            sum(result.request_tok_s for result in ok) / len(ok) if ok else None
        ),
        "slowest_single_request_s": duration,
        "by_turn": summarize_by_turn(ok),
    }


def summarize_by_turn(results: list[TurnProfile]) -> dict[str, dict[str, Any]]:
    by_turn: dict[int, list[TurnProfile]] = {}
    for result in results:
        by_turn.setdefault(result.turn, []).append(result)
    summary: dict[str, dict[str, Any]] = {}
    for turn, group in sorted(by_turn.items()):
        elapsed = [result.elapsed_s for result in group]
        output_tokens = sum(result.output_tokens_estimate for result in group)
        summary[str(turn)] = {
            "requests": len(group),
            "input_tokens_estimate": sum(result.input_tokens_estimate for result in group),
            "output_tokens_estimate": output_tokens,
            "avg_request_s": sum(elapsed) / len(elapsed) if elapsed else None,
            "p50_request_s": percentile(elapsed, 0.50),
            "p90_request_s": percentile(elapsed, 0.90),
            "mean_request_output_tok_s": (
                sum(result.request_tok_s for result in group) / len(group) if group else None
            ),
        }
    return summary


def summarize_compact_events(compact_debug_dir: Path) -> dict[str, Any]:
    event_path = compact_debug_dir / "compact_events.jsonl"
    if not event_path.exists():
        return {"compact_event_path": str(event_path), "compact_events": 0}
    events = []
    for line in event_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    committed = [event for event in events if event.get("status") == "committed"]
    failed = [event for event in events if event.get("status") == "failed"]
    cpu = [event["compact_cpu_ms"] for event in committed if event.get("compact_cpu_ms") is not None]
    gpu = [event["compact_gpu_ms"] for event in committed if event.get("compact_gpu_ms") is not None]
    return {
        "compact_event_path": str(event_path),
        "compact_events": len(events),
        "compact_committed": len(committed),
        "compact_failed": len(failed),
        "compact_cpu_ms_total": sum(cpu),
        "compact_cpu_ms_avg": sum(cpu) / len(cpu) if cpu else None,
        "compact_gpu_ms_total": sum(gpu),
        "compact_gpu_ms_avg": sum(gpu) / len(gpu) if gpu else None,
    }


def write_jsonl(path: Path, results: list[TurnProfile]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def run_once(args: argparse.Namespace, *, log_dir: Path) -> dict[str, Any]:
    random.seed(args.seed)
    if args.base_url:
        base_url = normalize_base_url(args.base_url)
        model_path = args.tokenizer_path or args.model_path
        if not model_path:
            raise RuntimeError("--tokenizer-path or --model-path is required with --base-url")
        server = None
    else:
        model_path = resolve_model_path(args)
        server = MiniSGLangProcess(args, model_path, log_dir)
        server.start()
        base_url = server.base_url

    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        results, model = asyncio.run(
            run_profile(base_url=base_url, tokenizer_path=model_path, args=args)
        )
    finally:
        if server is not None:
            server.stop()

    total_s = time.perf_counter() - started
    write_jsonl(log_dir / "turns.jsonl", results)
    compact_dir = server.compact_debug_dir if server is not None else log_dir / "compact_debug"
    summary = {
        "model": model if "model" in locals() else None,
        "model_path": model_path,
        "base_url": base_url,
        "eviction_plan": args.eviction_plan,
        "sessions": args.sessions,
        "concurrency": args.concurrency,
        "turns": args.turns,
        "prompt_len": args.prompt_len,
        "turn_prompt_len": args.turn_prompt_len,
        "max_tokens": args.max_tokens,
        "total_wall_s": total_s,
        "results_path": str(log_dir / "turns.jsonl"),
        "server_log": str(server.log_path) if server is not None else None,
        "request_summary": summarize_turns(results),
        "compact_summary": summarize_compact_events(compact_dir),
    }
    output_tokens = summary["request_summary"]["output_tokens_estimate"]
    input_tokens = summary["request_summary"]["input_tokens_estimate"]
    summary["end_to_end_output_tok_s"] = output_tokens / total_s if total_s > 0 else 0.0
    summary["end_to_end_total_tok_s_estimate"] = (
        (input_tokens + output_tokens) / total_s if total_s > 0 else 0.0
    )
    (log_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[profile] summary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["request_summary"]["errors"] and not args.allow_errors:
        raise RuntimeError(f"profile run failed for eviction_plan={args.eviction_plan}")
    return summary


def parse_plan_list(args: argparse.Namespace) -> list[str]:
    if args.eviction_plans.strip():
        plans = [plan.strip() for plan in args.eviction_plans.split(",") if plan.strip()]
    else:
        plans = [args.eviction_plan]
    if not plans:
        raise RuntimeError("at least one eviction plan is required")
    if args.base_url and len(plans) > 1:
        raise RuntimeError("--base-url can only profile one already-running eviction plan")
    return plans


def main() -> int:
    args = parse_args()
    plans = parse_plan_list(args)
    base_log_dir = Path(args.log_dir or (MINISGL_ROOT / "profile_runs" / timestamp()))
    summaries: list[dict[str, Any]] = []

    for plan in plans:
        run_args = copy.copy(args)
        run_args.eviction_plan = plan
        plan_log_dir = base_log_dir if len(plans) == 1 else base_log_dir / plan
        print(f"[profile] running eviction_plan={plan} log_dir={plan_log_dir}")
        summaries.append(run_once(run_args, log_dir=plan_log_dir))

    if len(summaries) > 1:
        comparison = {
            "plans": [
                {
                    "eviction_plan": summary["eviction_plan"],
                    "end_to_end_output_tok_s": summary["end_to_end_output_tok_s"],
                    "end_to_end_total_tok_s_estimate": summary[
                        "end_to_end_total_tok_s_estimate"
                    ],
                    "successful_requests": summary["request_summary"]["successful_requests"],
                    "errors": summary["request_summary"]["errors"],
                    "compact_events": summary["compact_summary"].get("compact_events", 0),
                    "summary_path": str(base_log_dir / summary["eviction_plan"] / "summary.json"),
                }
                for summary in summaries
            ]
        }
        (base_log_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("[profile] comparison")
        print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
