#!/usr/bin/env python3
"""Weight-loaded Qwen validation of the pre-prefill DRAM bridge."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

from dram_kv.backends.dram import DramRuntimeBackend
from dram_kv.integrations.qwen35 import install_qwen35_logical_rope
from dram_kv.runtime import DramKVRuntime
from dram_kv.runtime.qwen import QwenKVPageAdapter, extract_page, full_attention_layers
from dram_kv.runtime.qwen_forward import QwenForwardBridge
from dram_kv.schema import (
    PageCatalog,
    PageRecord,
    SelectionPlan,
    SelectionReason,
    TurnContext,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefix-tokens", type=int, default=512)
    parser.add_argument("--suffix-tokens", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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


def fork_cache(cache: Any) -> Any:
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
        layers.append(child)
    fork.layers = layers
    return fork


def clear_full_attention(cache: Any) -> None:
    for layer in full_attention_layers(cache).values():
        layer.keys = None
        layer.values = None
        if hasattr(layer, "is_initialized"):
            layer.is_initialized = False


def catalog_and_payloads(cache: Any, prefix_tokens: int, page_size: int):
    adapter = QwenKVPageAdapter("cuda")
    pages = []
    payloads = []
    for index, start in enumerate(range(0, prefix_tokens, page_size)):
        end = min(prefix_tokens, start + page_size)
        payload = extract_page(cache, start, end)
        page_id = f"p{index:04d}"
        pages.append(
            PageRecord(
                page_id=page_id,
                turn_id=1,
                role="prefix",
                token_start=start,
                token_end=end,
                logical_position_start=start,
                logical_position_end=end,
                byte_size=adapter.nbytes(payload),
                dram_location=f"qwen-forward/{page_id}",
            )
        )
        payloads.append((page_id, payload))
    return PageCatalog("qwen-forward-validation", tuple(pages)), payloads


def context_for(catalog: PageCatalog, resident: tuple[str, ...]) -> TurnContext:
    return TurnContext(
        session_id=catalog.session_id,
        instance_id="qwen-forward-validation",
        turn_id=2,
        task_objective="validate pre-prefill DRAM restore",
        previous_goal_text=None,
        previous_action=None,
        current_observation="fixed suffix",
        historical_turn_ids=(1,),
        historical_page_ids=tuple(page.page_id for page in catalog.pages),
        hbm_resident_page_ids=resident,
    )


def plan_for(catalog: PageCatalog, selected: tuple[str, ...], name: str) -> SelectionPlan:
    return SelectionPlan(
        session_id=catalog.session_id,
        turn_id=2,
        goal_hash="qwen-forward-validation",
        selector=name,
        selector_config={},
        ordered_page_ids=selected,
        reasons={page_id: SelectionReason(name, 1.0) for page_id in selected},
        mandatory_page_ids=(),
        recent_page_ids=(),
        retrieved_page_ids=selected,
        active_page_budget=len(selected),
    )


def forward(model: Any, torch: Any, input_ids: Any, kwargs: dict[str, Any]):
    torch.cuda.synchronize()
    started = perf_counter()
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
            logits_to_keep=0,
            **kwargs,
        )
    torch.cuda.synchronize()
    return output, (perf_counter() - started) * 1000.0


def dense_kwargs(cache: Any, torch: Any, prefix_tokens: int, suffix_tokens: int, device: Any):
    physical = torch.arange(
        prefix_tokens, prefix_tokens + suffix_tokens, dtype=torch.long, device=device
    )
    return {
        "past_key_values": cache,
        "attention_mask": torch.ones(
            (1, prefix_tokens + suffix_tokens), dtype=torch.long, device=device
        ),
        "cache_position": physical,
        "position_ids": physical.unsqueeze(0),
        "dram_kv_logical_position_ids": physical.unsqueeze(0),
    }


def tensor_delta(torch: Any, left: Any, right: Any) -> dict[str, Any]:
    delta = (left.float() - right.float()).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "top1_agreement": float(
            (left.argmax(dim=-1) == right.argmax(dim=-1)).float().mean().item()
        ),
    }


def cache_delta(torch: Any, left: Any, right: Any) -> dict[str, Any]:
    left_layers = full_attention_layers(left)
    right_layers = full_attention_layers(right)
    if tuple(left_layers) != tuple(right_layers):
        raise AssertionError("full-attention layer IDs differ")
    max_abs = 0.0
    exact = True
    for layer_id in left_layers:
        for name in ("keys", "values"):
            a = getattr(left_layers[layer_id], name)
            b = getattr(right_layers[layer_id], name)
            if tuple(a.shape) != tuple(b.shape):
                raise AssertionError(f"layer {layer_id} {name} shape differs")
            exact = exact and bool(torch.equal(a, b))
            max_abs = max(max_abs, float((a.float() - b.float()).abs().max().item()))
    return {"exact": exact, "max_abs": max_abs}


def state_tensor_deltas(torch: Any, left: Any, right: Any) -> dict[str, Any]:
    maximum = 0.0
    exact = True
    count = 0

    def visit(a: Any, b: Any) -> None:
        nonlocal maximum, exact, count
        if hasattr(a, "detach") and hasattr(b, "detach"):
            count += 1
            exact = exact and bool(torch.equal(a, b))
            maximum = max(maximum, float((a.float() - b.float()).abs().max().item()))
        elif isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                raise AssertionError("recurrent state dictionaries differ")
            for key in a:
                visit(a[key], b[key])
        elif isinstance(a, (list, tuple)) and isinstance(b, type(a)):
            if len(a) != len(b):
                raise AssertionError("recurrent state sequences differ")
            for item_a, item_b in zip(a, b):
                visit(item_a, item_b)
        elif a != b:
            exact = False

    for left_layer, right_layer in zip(left.layers, right.layers):
        for name in ("conv_states", "recurrent_states"):
            if hasattr(left_layer, name) or hasattr(right_layer, name):
                visit(getattr(left_layer, name, None), getattr(right_layer, name, None))
    return {"tensor_count": count, "exact": exact, "max_abs": maximum}


def main() -> None:
    args = parse_args()
    if args.prefix_tokens <= 0 or args.suffix_tokens <= 0:
        raise ValueError("prefix and suffix lengths must be positive")
    if args.prefix_tokens % args.page_size:
        raise ValueError("prefix-tokens must be divisible by page-size")

    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(7)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    patched_layers = install_qwen35_logical_rope(model)
    device = model.device
    vocab_size = int(model.config.vocab_size)
    all_ids = (
        torch.arange(
            1000,
            1000 + args.prefix_tokens + args.suffix_tokens,
            dtype=torch.long,
            device=device,
        )
        % vocab_size
    ).unsqueeze(0)
    prefix_ids = all_ids[:, : args.prefix_tokens]
    suffix_ids = all_ids[:, args.prefix_tokens :]

    empty_before_prefix = int(torch.cuda.memory_allocated())
    prefix_positions = torch.arange(args.prefix_tokens, dtype=torch.long, device=device)
    prefix_output, prefix_ms = forward(
        model,
        torch,
        prefix_ids,
        {
            "attention_mask": torch.ones_like(prefix_ids),
            "cache_position": prefix_positions,
            "position_ids": prefix_positions.unsqueeze(0),
            "dram_kv_logical_position_ids": prefix_positions.unsqueeze(0),
        },
    )
    boundary_cache = prefix_output.past_key_values
    del prefix_output
    dense_cache = fork_cache(boundary_cache)
    restored_cache = fork_cache(boundary_cache)
    selected_cache = fork_cache(boundary_cache)
    clear_full_attention(restored_cache)
    clear_full_attention(selected_cache)

    catalog, payloads = catalog_and_payloads(
        boundary_cache, args.prefix_tokens, args.page_size
    )
    adapter = QwenKVPageAdapter(device)
    runtime = DramKVRuntime(adapter, pin_cpu=True)
    backend = DramRuntimeBackend(runtime)
    for page_id, payload in payloads:
        backend.seal_after_generation(catalog.session_id, page_id, payload)
    del payloads, boundary_cache
    gc.collect()
    torch.cuda.empty_cache()

    all_page_ids = tuple(page.page_id for page in catalog.pages)
    full_plan = plan_for(catalog, all_page_ids, "full_restore")
    bridge = QwenForwardBridge(backend)
    prepare_started = perf_counter()
    full_prepared = bridge.prepare_before_prefill(
        context_for(catalog, ()),
        catalog,
        full_plan,
        restored_cache,
        suffix_logical_start=args.prefix_tokens,
        suffix_token_count=args.suffix_tokens,
    )
    torch.cuda.synchronize()
    full_prepare_ms = (perf_counter() - prepare_started) * 1000.0

    dense_output, dense_ms = forward(
        model,
        torch,
        suffix_ids,
        dense_kwargs(dense_cache, torch, args.prefix_tokens, args.suffix_tokens, device),
    )
    restored_output, restored_ms = forward(
        model,
        torch,
        suffix_ids,
        full_prepared.model_kwargs(torch, device=device),
    )
    full_logit_delta = tensor_delta(torch, dense_output.logits, restored_output.logits)
    full_cache_delta = cache_delta(
        torch, dense_output.past_key_values, restored_output.past_key_values
    )
    full_state_delta = state_tensor_deltas(
        torch, dense_output.past_key_values, restored_output.past_key_values
    )
    if full_logit_delta["max_abs"] > 1e-5:
        raise AssertionError(f"100% restored logits differ: {full_logit_delta}")
    if not full_cache_delta["exact"] or not full_state_delta["exact"]:
        raise AssertionError("100% restored cache/state differs from Dense")

    selected_ids = (all_page_ids[0], all_page_ids[-1])
    selected_plan = plan_for(catalog, selected_ids, "first_last_pages")
    selected_prepare_started = perf_counter()
    selected_prepared = bridge.prepare_before_prefill(
        context_for(catalog, all_page_ids),
        catalog,
        selected_plan,
        selected_cache,
        suffix_logical_start=args.prefix_tokens,
        suffix_token_count=args.suffix_tokens,
    )
    torch.cuda.synchronize()
    selected_prepare_ms = (perf_counter() - selected_prepare_started) * 1000.0
    selected_output, selected_ms = forward(
        model,
        torch,
        suffix_ids,
        selected_prepared.model_kwargs(torch, device=device),
    )
    selected_logit_delta = tensor_delta(torch, dense_output.logits, selected_output.logits)

    result = {
        "model": args.model,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "page_size": args.page_size,
        "page_count": len(all_page_ids),
        "selected_page_ids": selected_ids,
        "patched_full_attention_layers": patched_layers,
        "model_allocated_bytes_before_prefix": empty_before_prefix,
        "model_allocated_bytes_final": int(torch.cuda.memory_allocated()),
        "prefix_forward_ms": prefix_ms,
        "dense_suffix_forward_ms": dense_ms,
        "full_restore_prepare_ms": full_prepare_ms,
        "full_restore_suffix_forward_ms": restored_ms,
        "selected_prepare_ms": selected_prepare_ms,
        "selected_suffix_forward_ms": selected_ms,
        "full_restore_logit_delta_vs_dense": full_logit_delta,
        "full_restore_attention_cache_delta_vs_dense": full_cache_delta,
        "full_restore_recurrent_state_delta_vs_dense": full_state_delta,
        "selected_logit_delta_vs_dense": selected_logit_delta,
        "full_active_logical_positions": full_prepared.active_logical_positions,
        "selected_active_logical_positions": selected_prepared.active_logical_positions,
        "suffix_logical_positions": full_prepared.suffix_logical_positions,
        "full_physical_cache_positions": full_prepared.cache_positions,
        "selected_physical_cache_positions": selected_prepared.cache_positions,
        "runtime_stats": asdict(runtime.session_metrics(catalog.session_id)),
        "acceptance": {
            "restore_before_prefill": True,
            "full_logits_match_dense": full_logit_delta["max_abs"] <= 1e-5,
            "full_attention_cache_exact": full_cache_delta["exact"],
            "recurrent_state_exact": full_state_delta["exact"],
            "selected_forward_completed": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
