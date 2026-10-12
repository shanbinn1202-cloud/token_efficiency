#!/usr/bin/env python3
"""Recorded Qwen turn loop with native DRAM backing and Oracle budgets."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from dram_kv.backends.dram import DramRuntimeBackend
from dram_kv.integrations.qwen35 import install_qwen35_logical_rope
from dram_kv.replay import (
    ActiveBudget,
    RecordedTrajectory,
    RecordedTurn,
    RecordedTurnLoop,
    RenderedSpan,
    build_selection_plan,
    oracle_evidence_from_annotation,
)
from dram_kv.goals.features import DeterministicFeatureExtractor
from dram_kv.replay.swe_agent import assistant_action, tool_observation
from dram_kv.runtime import DramKVRuntime
from dram_kv.runtime.qwen import QwenKVPageAdapter, extract_page, full_attention_layers
from dram_kv.runtime.qwen_forward import QwenForwardBridge
from dram_kv.schema import GoalRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--ratios", type=float, nargs="+", default=(0.25, 0.40, 0.60))
    parser.add_argument("--eval-turns", type=int, nargs="+", default=(11, 23, 29, 36, 43, 48))
    parser.add_argument("--switch-threshold", type=int, default=8192)
    return parser.parse_args()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def extract_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def chat_renderer(tokenizer: Any):
    """Prefix-stable reconstruction of the saved mini-SWE-agent transcript."""

    def encode(text: str) -> list[int]:
        return extract_ids(tokenizer(text, add_special_tokens=False))

    def render(messages: Sequence[Mapping[str, Any]], add_generation_prompt: bool):
        if len(messages) < 2:
            return []
        kwargs = {"tokenize": True, "add_generation_prompt": False}
        try:
            ids = extract_ids(
                tokenizer.apply_chat_template(messages[:2], **kwargs, enable_thinking=False)
            )
        except TypeError:
            ids = extract_ids(tokenizer.apply_chat_template(messages[:2], **kwargs))
        for message in messages[2:]:
            role = str(message["role"])
            content = str(message.get("content", ""))
            if role in {"assistant", "user"}:
                rendered = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            elif role == "tool":
                rendered = (
                    "<|im_start|>user\n<tool_response>\n"
                    f"{content}\n</tool_response><|im_end|>\n"
                )
            else:
                raise ValueError(f"unsupported recorded role {role!r}")
            ids.extend(encode(rendered))
        if add_generation_prompt:
            if messages[-1].get("role") not in {"user", "tool"}:
                raise ValueError("generation prompt must follow a user/tool observation")
            # Deliberately omit Qwen3's empty <think> wrapper: the recorded
            # assistant continuation does not contain it, and cumulative KV
            # replay requires every completed prefix to remain byte-identical.
            ids.extend(encode("<|im_start|>assistant\n"))
        return ids

    return render


def require_prefix(prefix: Sequence[int], full: Sequence[int], label: str) -> None:
    if list(full[: len(prefix)]) != list(prefix):
        raise ValueError(f"canonical tokens are not prefix-stable at {label}")


def build_recorded_trajectory(
    messages: Sequence[Mapping[str, Any]],
    annotation: Mapping[str, Any],
    render: Any,
) -> RecordedTrajectory:
    terminal = int(annotation["message_map"]["terminal_exit_message_id"])
    if terminal < 0 or terminal >= len(messages):
        raise ValueError("terminal message lies outside trajectory")
    complete: list[list[int]] = [[], []]
    for count in range(2, terminal + 1):
        ids = list(render(messages[:count], False))
        require_prefix(complete[-1], ids, f"message-count-{count}")
        complete.append(ids)

    entries = {int(item["turn_id"]): item for item in annotation["turns"]}
    extractor = DeterministicFeatureExtractor()
    turns = []
    previous_full: list[int] = []
    for turn_id in sorted(entries):
        entry = entries[turn_id]
        assistant_id = int(entry["assistant_message_id"])
        context = list(render(messages[:assistant_id], True))
        full = complete[assistant_id + 1]
        require_prefix(previous_full, context, f"turn-{turn_id}-context")
        require_prefix(previous_full, full, f"turn-{turn_id}-action-history")
        # A BPE token can straddle the generation-prompt/completion boundary
        # (for example, one token may encode the prompt newline plus leading
        # completion newlines).  Keep the canonical full-transcript token
        # stream and use the standalone-context length as the ownership cut.
        # Only that single boundary token is allowed to differ.
        common = 0
        for context_token, full_token in zip(context, full):
            if context_token != full_token:
                break
            common += 1
        if common < len(context) - 1:
            raise ValueError(
                f"canonical tokens diverge before action boundary at turn {turn_id}"
            )
        prefill_count = len(context) - len(previous_full)
        turn_delta = full[len(previous_full) :]
        if prefill_count > len(turn_delta):
            raise ValueError(f"action boundary lies beyond canonical turn {turn_id}")
        prefill_ids = tuple(turn_delta[:prefill_count])
        generated_ids = tuple(turn_delta[prefill_count:])
        source_ids = tuple(int(value) for value in entry.get("input_message_ids", ()))
        if not source_ids:
            source_ids = (0, 1) if turn_id == 1 else (assistant_id - 1,)
        action_text = assistant_action(messages[assistant_id])
        if turn_id == 1:
            observation_text, return_code = "", None
        else:
            observations = [tool_observation(messages[value]) for value in source_ids]
            observation_text = "\n".join(text for text, _code in observations)
            return_code = next(
                (code for _text, code in reversed(observations) if code is not None),
                None,
            )
        action_features = extractor.extract(action_text, "", tool_name="shell")
        observation_features = extractor.extract(
            "", observation_text, tool_name="shell", return_code=return_code
        )
        turns.append(
            RecordedTurn(
                turn_id,
                (
                    RenderedSpan(
                        "bootstrap" if turn_id == 1 else "current_observation",
                        prefill_ids,
                        mandatory=turn_id == 1,
                        source_message_ids=source_ids,
                        summary=observation_features.observation.excerpt or None,
                        files=observation_features.files,
                        symbols=observation_features.symbols,
                        tests=observation_features.tests,
                        errors=observation_features.errors,
                    ),
                ),
                (
                    RenderedSpan(
                        "assistant",
                        generated_ids,
                        source_message_ids=(assistant_id,),
                        summary=action_features.action.excerpt,
                        files=action_features.files,
                        symbols=action_features.symbols,
                        tests=action_features.tests,
                        errors=action_features.errors,
                    ),
                ),
                action_features.action.excerpt,
                action_text,
                observation_text,
                return_code,
                "shell",
            )
        )
        previous_full = full
    return RecordedTrajectory(
        "astropy-14995-oracle-sweep",
        str(annotation["trajectory"]["instance_id"]),
        str(messages[1].get("content", "")),
        tuple(turns),
    )


def clone_state(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state(item) for item in value)
    return value


def fork_recurrent_skeleton(cache: Any) -> Any:
    fork = copy.copy(cache)
    layers = []
    for layer in cache.layers:
        child = copy.copy(layer)
        for name in (
            "conv_states",
            "recurrent_states",
            "has_previous_state",
            "is_conv_states_initialized",
            "is_recurrent_states_initialized",
        ):
            if hasattr(layer, name):
                setattr(child, name, clone_state(getattr(layer, name)))
        if hasattr(child, "keys") and hasattr(child, "values"):
            child.keys = None
            child.values = None
            if hasattr(child, "is_initialized"):
                child.is_initialized = False
        layers.append(child)
    fork.layers = layers
    return fork


def clear_full_attention(cache: Any) -> None:
    for layer in full_attention_layers(cache).values():
        layer.keys = None
        layer.values = None
        if hasattr(layer, "is_initialized"):
            layer.is_initialized = False


def cache_length(cache: Any) -> int:
    lengths = {int(layer.keys.shape[-2]) for layer in full_attention_layers(cache).values()}
    if len(lengths) != 1:
        raise RuntimeError(f"full-attention cache lengths differ: {sorted(lengths)}")
    return lengths.pop()


@dataclass(frozen=True)
class ColdPrepared:
    token_count: int

    def model_kwargs(self, torch: Any, *, device: Any) -> dict[str, Any]:
        positions = torch.arange(self.token_count, dtype=torch.long, device=device)
        return {
            "past_key_values": None,
            "attention_mask": torch.ones(
                (1, self.token_count), dtype=torch.long, device=device
            ),
            "cache_position": positions,
            "position_ids": positions.unsqueeze(0),
            "dram_kv_logical_position_ids": positions.unsqueeze(0),
        }


def turn_token_ids(turn: RecordedTurn) -> tuple[tuple[int, ...], tuple[int, ...]]:
    prefill = tuple(token for span in turn.prefill_spans for token in span.token_ids)
    generated = tuple(token for span in turn.generated_spans for token in span.token_ids)
    return prefill + generated, generated


def model_forward(model: Any, torch: Any, prepared: Any, turn: RecordedTurn):
    feed, generated = turn_token_ids(turn)
    input_ids = torch.tensor([feed], dtype=torch.long, device=model.device)
    torch.cuda.synchronize()
    started = perf_counter()
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
            logits_to_keep=len(generated) + 1,
            **prepared.model_kwargs(torch, device=model.device),
        )
    torch.cuda.synchronize()
    elapsed = (perf_counter() - started) * 1000.0
    if int(output.logits.shape[1]) != len(generated) + 1:
        raise RuntimeError("model returned an unexpected logit suffix")
    action_logits = output.logits[:, : len(generated), :].detach()
    labels = torch.tensor(generated, dtype=torch.long, device=model.device)
    return output, action_logits, labels, elapsed


def comparison_metrics(torch: Any, branch: Any, dense: Any, labels: Any) -> dict[str, Any]:
    import torch.nn.functional as functional

    branch_f = branch.squeeze(0).float()
    dense_f = dense.squeeze(0).float()
    delta = (branch_f - dense_f).abs()
    branch_nll = functional.cross_entropy(branch_f, labels).item()
    dense_nll = functional.cross_entropy(dense_f, labels).item()
    return {
        "action_nll": branch_nll,
        "dense_action_nll": dense_nll,
        "delta_nll_vs_dense": branch_nll - dense_nll,
        "top1_agreement_vs_dense": float(
            (branch_f.argmax(-1) == dense_f.argmax(-1)).float().mean().item()
        ),
        "max_abs_logit_delta": float(delta.max().item()),
        "mean_abs_logit_delta": float(delta.mean().item()),
    }


def main() -> None:
    args = parse_args()
    if args.page_size <= 0 or any(not 0 < ratio < 1 for ratio in args.ratios):
        raise ValueError("page size must be positive and ratios must be in (0, 1)")
    trajectory_json = load_object(args.trajectory)
    annotation = load_object(args.annotation)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    trajectory = build_recorded_trajectory(
        trajectory_json["messages"], annotation, chat_renderer(tokenizer)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    patched_layers = install_qwen35_logical_rope(model)
    text_config = getattr(model.config, "text_config", model.config)
    bytes_per_token = (
        sum(kind == "full_attention" for kind in text_config.layer_types)
        * 2
        * int(text_config.num_key_value_heads)
        * int(text_config.head_dim)
        * torch.tensor([], dtype=torch.bfloat16).element_size()
    )
    catalog = trajectory.catalog(
        page_size=args.page_size, bytes_per_token=bytes_per_token
    )
    page_sources = trajectory.page_source_message_ids(
        page_size=args.page_size, bytes_per_token=bytes_per_token
    )
    evidence = oracle_evidence_from_annotation(annotation)

    adapter = QwenKVPageAdapter(model.device)
    runtime = DramKVRuntime(adapter, pin_cpu=True)
    backend = DramRuntimeBackend(runtime)
    bridge = QwenForwardBridge(backend)
    loop = RecordedTurnLoop(trajectory, catalog, backend)
    skeleton = None
    artifacts = []
    sweep_records = []
    selector_state = {
        ratio: {"previous": (), "ever": set()} for ratio in args.ratios
    }
    eval_turns = set(args.eval_turns)

    for turn in trajectory.turns:
        resident = runtime.resident_page_ids(trajectory.session_id)
        context = loop.context_for_turn(
            turn.turn_id, resident_page_ids=resident
        )
        goal = GoalRecord(
            context.session_id,
            context.turn_id,
            "recorded_action_summary",
            {"future_leakage": True},
            turn.action_summary,
        )
        _, prefill_end, generated_end = trajectory.turn_bounds(turn.turn_id)
        selection_enabled = prefill_end > args.switch_threshold
        plans: dict[float, tuple[Any, Any]] = {}
        if selection_enabled and turn.turn_id in eval_turns:
            for ratio in args.ratios:
                state = selector_state[ratio]
                plan, decision = build_selection_plan(
                    context,
                    goal,
                    catalog,
                    policy="oracle",
                    evidence=evidence,
                    budget=ActiveBudget("ratio", ratio),
                    previous_active=state["previous"],
                    ever_selected=state["ever"],
                    page_source_message_ids=page_sources,
                )
                state["previous"] = plan.ordered_page_ids
                state["ever"].update(plan.ordered_page_ids)
                plans[ratio] = (plan, decision)

        branch_results = []
        if turn.turn_id in eval_turns:
            if not selection_enabled:
                raise RuntimeError(f"eval turn {turn.turn_id} precedes selection switch")
            for ratio in args.ratios:
                plan, decision = plans[ratio]
                branch_cache = fork_recurrent_skeleton(skeleton)
                prepare_started = perf_counter()
                prepared = bridge.prepare_before_prefill(
                    context,
                    catalog,
                    plan,
                    branch_cache,
                    suffix_logical_start=trajectory.turn_bounds(turn.turn_id)[0],
                    suffix_token_count=turn.appended_token_count,
                )
                torch.cuda.synchronize()
                prepare_ms = (perf_counter() - prepare_started) * 1000.0
                output, logits, labels, forward_ms = model_forward(
                    model, torch, prepared, turn
                )
                branch_results.append(
                    (ratio, plan, decision, logits, labels, prepare_ms, forward_ms)
                )
                del output, prepared, branch_cache

        # The canonical trajectory itself advances through full DRAM restore.
        resident = runtime.resident_page_ids(trajectory.session_id)
        main_context = loop.context_for_turn(
            turn.turn_id, resident_page_ids=resident
        )
        full_plan, full_decision = build_selection_plan(
            main_context, goal, catalog, policy="full_active"
        )
        holder: dict[str, Any] = {}

        def restore_and_bind(ctx: Any, cat: Any, plan: Any):
            if turn.turn_id == 1:
                return ColdPrepared(turn.appended_token_count)
            return bridge.prepare_before_prefill(
                ctx,
                cat,
                plan,
                skeleton,
                suffix_logical_start=trajectory.turn_bounds(turn.turn_id)[0],
                suffix_token_count=turn.appended_token_count,
            )

        def prefill_and_generate(prepared: Any, recorded_turn: RecordedTurn):
            output, logits, labels, elapsed = model_forward(
                model, torch, prepared, recorded_turn
            )
            holder.update(
                output=output, logits=logits, labels=labels, forward_ms=elapsed
            )
            return output

        def extract_payload(output: Any, page: Any):
            return extract_page(
                output.past_key_values,
                page.logical_position_start,
                page.logical_position_end,
            )

        artifact = loop.run_turn(
            main_context,
            full_plan,
            full_decision,
            restore_and_bind=restore_and_bind,
            prefill_and_generate=prefill_and_generate,
            extract_payload=extract_payload,
        )
        if cache_length(holder["output"].past_key_values) != generated_end:
            raise RuntimeError(f"turn {turn.turn_id} full cache boundary mismatch")

        dense_logits = holder["logits"]
        labels = holder["labels"]
        if turn.turn_id in eval_turns:
            baseline = comparison_metrics(torch, dense_logits, dense_logits, labels)
            sweep_records.append(
                {
                    "turn_id": turn.turn_id,
                    "branch": "full_active",
                    "history_tokens": artifact.history_token_end,
                    "active_tokens": artifact.history_token_end,
                    "active_ratio": 1.0,
                    "pages": len(full_plan.ordered_page_ids),
                    "forward_ms": holder["forward_ms"],
                    **baseline,
                }
            )
            for ratio, plan, decision, logits, branch_labels, prepare_ms, forward_ms in branch_results:
                if not bool(torch.equal(labels, branch_labels)):
                    raise RuntimeError("branch labels differ from full-active labels")
                metrics = comparison_metrics(torch, logits, dense_logits, labels)
                sweep_records.append(
                    {
                        "turn_id": turn.turn_id,
                        "branch": f"oracle_ratio_{ratio:.2f}",
                        "history_tokens": decision.historical_tokens,
                        "active_tokens": decision.selected_tokens,
                        "active_ratio": (
                            decision.selected_tokens / decision.historical_tokens
                        ),
                        "pages": decision.selected_pages,
                        "capacity_tokens": decision.capacity,
                        "unused_capacity": decision.unused_capacity,
                        "omitted_recent_pages": len(decision.omitted_recent_page_ids),
                        "omitted_oracle_pages": len(decision.omitted_oracle_page_ids),
                        "filler_pages": len(decision.filler_page_ids),
                        "added_pages": len(plan.added_page_ids),
                        "retained_pages": len(plan.retained_page_ids),
                        "removed_pages": len(plan.removed_page_ids),
                        "prepare_ms": prepare_ms,
                        "forward_ms": forward_ms,
                        **metrics,
                    }
                )
                del logits, branch_labels

        artifacts.append(asdict(artifact))
        skeleton = holder["output"].past_key_values
        clear_full_attention(skeleton)
        del holder["output"], holder["logits"], holder["labels"]
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "progress": "turn_complete",
                    "turn_id": turn.turn_id,
                    "history_tokens": artifact.generated_token_end,
                    "evaluated": turn.turn_id in eval_turns,
                }
            ),
            flush=True,
        )

    result = {
        "config": {
            "model": args.model,
            "trajectory": str(args.trajectory),
            "annotation": str(args.annotation),
            "page_size": args.page_size,
            "bytes_per_token": bytes_per_token,
            "ratios": args.ratios,
            "eval_turns": args.eval_turns,
            "switch_threshold": args.switch_threshold,
            "teacher_forced": True,
            "oracle_future_leakage": True,
            "selected_branches_are_reset_from_full_recurrent_state": True,
            "selector_plans_built_only_at_eval_turns": True,
        },
        "environment": {
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "patched_full_attention_layers": patched_layers,
        },
        "trajectory": {
            "turns": len(trajectory.turns),
            "tokens": trajectory.turn_bounds(len(trajectory.turns))[2],
            "pages": len(catalog.pages),
        },
        "runtime_stats": asdict(runtime.session_metrics(trajectory.session_id)),
        "sweep_records": sweep_records,
        "acceptance": {
            "completed_all_turns": len(artifacts) == len(trajectory.turns),
            "event_order_valid": all(
                item["event_order"]
                == (
                    "plan",
                    "restore_bind_complete",
                    "prefill_generation_complete",
                    "seal_complete",
                )
                for item in artifacts
            ),
            "final_cache_boundary": trajectory.turn_bounds(len(trajectory.turns))[2],
            "sweep_record_count": len(sweep_records),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output.with_name(
        f"{args.output.stem}.turns.jsonl.gz"
    )
    with gzip.open(artifact_path, "wt", encoding="utf-8") as handle:
        for artifact in artifacts:
            handle.write(json.dumps(artifact, sort_keys=True) + "\n")
    result["turn_artifact_gzip"] = artifact_path.name
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
