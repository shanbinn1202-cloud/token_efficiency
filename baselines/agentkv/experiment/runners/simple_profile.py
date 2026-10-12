from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from .common import ROOT, RUNS_DIR, MiniSGLangServer, model_spec, timestamp, write_manifest

MINISGL_PYTHON = ROOT / "engine" / "python"
if str(MINISGL_PYTHON) not in sys.path:
    sys.path.insert(0, str(MINISGL_PYTHON))

from minisgl.benchmark.client import generate_prompt, get_model_name  # noqa: E402


@dataclass(frozen=True)
class ProfileCaseResult:
    batch_size: int
    successful_requests: int
    failed_requests: int
    prompt_len: int
    max_tokens: int
    input_tokens: int
    output_tokens: int
    elapsed_s: float
    output_tok_s: float
    total_tok_s: float
    error: str | None = None


def estimate_chat_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))


def response_text(message: Any) -> str:
    extra = getattr(message, "model_extra", None) or {}
    raw = extra.get("raw_content")
    if raw is not None:
        return raw
    return message.content or ""


async def close_session(client: AsyncOpenAI, model: str, session_id: str) -> None:
    await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": ""}],
        max_tokens=1,
        temperature=0.0,
        extra_body={"session_id": session_id, "close_session": True},
        timeout=30,
    )


async def run_case(
    *,
    client: AsyncOpenAI,
    tokenizer: Any,
    model: str,
    prompt_len: int,
    max_tokens: int,
    batch_size: int,
    timeout: float,
) -> ProfileCaseResult:
    messages_list = [
        [{"role": "user", "content": generate_prompt(tokenizer, prompt_len)}]
        for _ in range(batch_size)
    ]
    input_tokens = sum(estimate_chat_tokens(tokenizer, messages) for messages in messages_list)
    start = time.perf_counter()
    errors: list[str] = []

    async def run_one(index: int, messages: list[dict[str, str]]) -> int | None:
        session_id = f"simple-profile:{timestamp()}:{prompt_len}:{max_tokens}:{index}"
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                extra_body={"session_id": session_id, "ignore_eos": True, "top_k": 1},
                timeout=timeout,
            )
            text = response_text(response.choices[0].message)
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception as exc:
            errors.append(repr(exc))
            return None
        finally:
            try:
                await close_session(client, model, session_id)
            except Exception:
                pass

    outputs = await asyncio.gather(
        *(run_one(index, messages) for index, messages in enumerate(messages_list))
    )
    successful_requests = sum(output is not None for output in outputs)
    output_tokens = sum(output or 0 for output in outputs)

    elapsed = time.perf_counter() - start
    return ProfileCaseResult(
        batch_size=batch_size,
        successful_requests=successful_requests,
        failed_requests=batch_size - successful_requests,
        prompt_len=prompt_len,
        max_tokens=max_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_s=elapsed,
        output_tok_s=output_tokens / elapsed if elapsed > 0 else 0.0,
        total_tok_s=(input_tokens + output_tokens) / elapsed if elapsed > 0 else 0.0,
        error="; ".join(errors) if errors else None,
    )


async def run_cases(
    base_url: str,
    tokenizer_path: str,
    *,
    cases: list[tuple[int, int]],
    batch_size: int,
    timeout: float,
) -> tuple[str, list[ProfileCaseResult]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    async with AsyncOpenAI(base_url=base_url, api_key="EMPTY") as client:
        model = await get_model_name(client)
        results = []
        for prompt_len, max_tokens in cases:
            print(
                f"[profile] case batch_size={batch_size} "
                f"prompt_len={prompt_len} max_tokens={max_tokens}"
            )
            result = await run_case(
                client=client,
                tokenizer=tokenizer,
                model=model,
                prompt_len=prompt_len,
                max_tokens=max_tokens,
                batch_size=batch_size,
                timeout=timeout,
            )
            print(
                "[profile] done "
                f"prompt_len={result.prompt_len} ok={result.successful_requests}/{result.batch_size} "
                f"output_tokens={result.output_tokens} "
                f"elapsed={result.elapsed_s:.2f}s output_tok/s={result.output_tok_s:.2f} "
                f"total_tok/s={result.total_tok_s:.2f}"
            )
            if result.error:
                print(f"[profile] error prompt_len={result.prompt_len}: {result.error}")
            results.append(result)
    return model, results


def summarize_compact_events(compact_debug_dir: Path) -> dict[str, Any]:
    path = compact_debug_dir / "compact_events.jsonl"
    if not path.exists():
        return {"compact_event_path": str(path), "compact_events": 0}
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    committed = [event for event in events if event.get("status") == "committed"]

    def timing_stats(key: str) -> dict[str, Any]:
        values = [event[key] for event in committed if event.get(key) is not None]
        return {
            f"{key}_total": sum(values),
            f"{key}_avg": sum(values) / len(values) if values else None,
            f"{key}_max": max(values) if values else None,
        }

    summary = {
        "compact_event_path": str(path),
        "compact_events": len(events),
        "compact_committed": len(committed),
    }
    for key in (
        "select_cpu_ms",
        "select_gpu_ms",
        "prepare_cpu_ms",
        "prepare_normalize_cpu_ms",
        "prepare_representative_cpu_ms",
        "prepare_alloc_cpu_ms",
        "prepare_request_state_cpu_ms",
        "prepare_position_state_cpu_ms",
        "compact_cpu_ms",
        "compact_gpu_ms",
        "total_cpu_ms",
    ):
        summary.update(timing_stats(key))
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple mini-sglang tok/s profile.")
    parser.add_argument("--experiment", default="minisglang_simple_profile")
    parser.add_argument("--run", default=f"{timestamp()}_simple_profile")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--memory-ratio", default="0.75")
    parser.add_argument("--max-seq-len", default="40960")
    parser.add_argument("--max-extend-tokens", default="40960")
    parser.add_argument("--page-size", default="1")
    parser.add_argument("--max-running-requests", default="1")
    parser.add_argument("--cuda-graph-max-bs", default="1")
    parser.add_argument("--eviction-plan", default="none")
    parser.add_argument("--compact-interval", default="128")
    parser.add_argument("--compact-after-prefill", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compact-before-park", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile-compact", action="store_true")
    parser.add_argument("--compact-debug-artifacts", action="store_true")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Profile case as prompt_len:max_tokens, e.g. 4096:1024. Repeat for multiple cases.",
    )
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def parse_cases(raw_cases: list[str]) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    for raw in raw_cases:
        parts = raw.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"invalid --case {raw!r}; expected prompt_len:max_tokens")
        cases.append((int(parts[0]), int(parts[1])))
    return cases


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cases = parse_cases(args.case)
    if int(args.max_running_requests) < args.batch_size:
        raise ValueError(
            f"--max-running-requests ({args.max_running_requests}) must be >= "
            f"--batch-size ({args.batch_size})"
        )
    random.seed(args.seed)
    spec = model_spec(args.model)
    run_dir = RUNS_DIR / args.experiment / args.run
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    compact_debug_dir = logs / "compact_debug"
    server_log = logs / f"minisglang_{timestamp()}_{args.model}.log"
    write_manifest(
        run_dir / "run_manifest.json",
        {
            "kind": "simple_profile",
            "experiment": args.experiment,
            "run": args.run,
            "model": args.model,
            "gpu": args.gpu,
            "batch_size": args.batch_size,
            "cases": [{"prompt_len": prompt, "max_tokens": output} for prompt, output in cases],
            "server": {
                "memory_ratio": args.memory_ratio,
                "max_seq_len": args.max_seq_len,
                "max_extend_tokens": args.max_extend_tokens,
                "page_size": args.page_size,
                "attention_backend": "fi",
                "max_running_requests": args.max_running_requests,
                "cuda_graph_max_bs": args.cuda_graph_max_bs,
                "eviction_plan": args.eviction_plan,
                "compact_interval": args.compact_interval,
                "compact_after_prefill": args.compact_after_prefill,
                "compact_before_park": args.compact_before_park,
                "profile_compact": args.profile_compact,
                "compact_debug_artifacts": args.compact_debug_artifacts,
                "compact_debug_dir": str(compact_debug_dir),
            },
        },
    )

    started = time.perf_counter()
    with MiniSGLangServer(
        model_name=args.model,
        log_path=server_log,
        cuda_visible_devices=args.gpu,
        memory_ratio=args.memory_ratio,
        max_running_requests=args.max_running_requests,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        max_seq_len=args.max_seq_len,
        max_extend_tokens=args.max_extend_tokens,
        page_size=args.page_size,
        eviction_plan=args.eviction_plan,
        compact_interval=args.compact_interval,
        compact_after_prefill=args.compact_after_prefill,
        compact_before_park=args.compact_before_park,
        profile_compact=args.profile_compact,
        compact_debug_artifacts=args.compact_debug_artifacts,
        compact_debug_dir=compact_debug_dir,
    ) as server:
        model_id, results = asyncio.run(
            run_cases(
                server.base_url,
                spec.path,
                cases=cases,
                batch_size=args.batch_size,
                timeout=args.timeout,
            )
        )

    total_wall_s = time.perf_counter() - started
    rows = [asdict(result) for result in results]
    (run_dir / "profile_cases.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "model": model_id,
        "model_name": args.model,
        "model_path": spec.path,
        "batch_size": args.batch_size,
        "run_dir": str(run_dir),
        "server_log": str(server_log),
        "total_wall_s": total_wall_s,
        "cases": rows,
        "compact_summary": summarize_compact_events(compact_debug_dir),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("[profile] summary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if any(result.error for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
