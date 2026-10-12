#!/usr/bin/env python3
"""GH200 validation for Qwen full-attention KV Page offload and restore.

The benchmark uses the model's real Transformers ``DynamicCache`` layout and
the exact K/V shape from its config, while avoiding a 27B weight load.  This
isolates allocator behavior and byte-exact tensor transfer from model-weight
memory.  Semantic/logit equivalence remains covered by the recorded Qwen
replay regression.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dram_kv.runtime import DramKVRuntime
from dram_kv.runtime.qwen import (
    QwenKVPageAdapter,
    assemble_pages,
    extract_page,
    full_attention_layers,
    write_active_cache,
)
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
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--page-size", type=int, default=512)
    parser.add_argument("--selected-pages", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def allocator_snapshot(torch: Any, label: str, *, trim: bool = True) -> dict[str, Any]:
    gc.collect()
    torch.cuda.synchronize()
    if trim:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return {
        "label": label,
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "cumulative_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cumulative_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def clear_full_attention(cache: Any) -> None:
    for layer in full_attention_layers(cache).values():
        layer.keys = None
        layer.values = None
        if hasattr(layer, "is_initialized"):
            layer.is_initialized = False


def make_records(tokens: int, page_size: int, page_bytes: list[int]) -> PageCatalog:
    pages = []
    for index, byte_size in enumerate(page_bytes):
        start = index * page_size
        end = min(tokens, start + page_size)
        pages.append(
            PageRecord(
                page_id=f"p{index:04d}",
                turn_id=1,
                role="history",
                token_start=start,
                token_end=end,
                logical_position_start=start,
                logical_position_end=end,
                byte_size=byte_size,
                dram_location=f"session/page-{index}",
            )
        )
    return PageCatalog("qwen-dram-validation", tuple(pages))


def context_for(catalog: PageCatalog, resident: tuple[str, ...] = ()) -> TurnContext:
    return TurnContext(
        session_id=catalog.session_id,
        instance_id="synthetic-qwen-cache",
        turn_id=2,
        task_objective="validate DRAM-backed Qwen full-attention KV",
        previous_goal_text=None,
        previous_action=None,
        current_observation="allocator validation",
        historical_turn_ids=(1,),
        historical_page_ids=tuple(page.page_id for page in catalog.pages),
        hbm_resident_page_ids=resident,
    )


def plan_for(catalog: PageCatalog, selected: tuple[str, ...]) -> SelectionPlan:
    return SelectionPlan(
        session_id=catalog.session_id,
        turn_id=2,
        goal_hash="allocator-validation",
        selector="fixed_validation",
        selector_config={},
        ordered_page_ids=selected,
        reasons={page_id: SelectionReason("validation", 1.0) for page_id in selected},
        mandatory_page_ids=(),
        recent_page_ids=(),
        retrieved_page_ids=selected,
        active_page_budget=len(selected),
    )


def assert_expected(active: Any) -> None:
    import torch

    for layer in active.layers:
        expected_k = float(layer.layer_id + 1)
        expected_v = -expected_k
        if not bool(torch.all(layer.keys == expected_k)):
            raise AssertionError(f"layer {layer.layer_id} K changed during transfer")
        if not bool(torch.all(layer.values == expected_v)):
            raise AssertionError(f"layer {layer.layer_id} V changed during transfer")


def main() -> None:
    args = parse_args()
    if args.tokens <= 0 or args.page_size <= 0 or args.tokens % args.page_size:
        raise ValueError("tokens must be positive and divisible by page-size")
    page_count = args.tokens // args.page_size
    if not 0 < args.selected_pages < page_count:
        raise ValueError("selected-pages must be between zero and the full Page count")

    import torch
    from transformers import AutoConfig
    from transformers.cache_utils import DynamicCache

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    text_config = getattr(config, "text_config", config)
    cache = DynamicCache(config=text_config)
    full_layer_ids = tuple(
        index for index, kind in enumerate(text_config.layer_types)
        if kind == "full_attention"
    )
    recurrent_ids = tuple(index for index in range(len(cache.layers)) if index not in full_layer_ids)
    recurrent_sentinels = {}
    for index in recurrent_ids:
        # Non-empty, framework-shaped sentinels make the isolation assertion
        # meaningful even though this allocator-only test does not run forward.
        recurrent = {0: object()}
        convolution = {0: object()}
        cache.layers[index].recurrent_states = recurrent
        cache.layers[index].conv_states = convolution
        recurrent_sentinels[index] = (recurrent, convolution)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    snapshots = [allocator_snapshot(torch, "empty")]
    shape = (1, int(text_config.num_key_value_heads), args.tokens, int(text_config.head_dim))
    for layer_id in full_layer_ids:
        layer = cache.layers[layer_id]
        layer.keys = torch.full(shape, float(layer_id + 1), dtype=torch.bfloat16, device=device)
        layer.values = torch.full(shape, -float(layer_id + 1), dtype=torch.bfloat16, device=device)
        layer.is_initialized = True
    snapshots.append(allocator_snapshot(torch, "dense_cache"))

    adapter = QwenKVPageAdapter(device=device)
    runtime = DramKVRuntime(adapter, pin_cpu=True)
    page_bytes: list[int] = []
    for index, start in enumerate(range(0, args.tokens, args.page_size)):
        payload = extract_page(cache, start, start + args.page_size)
        page_bytes.append(adapter.nbytes(payload))
        runtime.offload_sealed("qwen-dram-validation", f"p{index:04d}", payload)
        del payload
    catalog = make_records(args.tokens, args.page_size, page_bytes)
    expected_full_bytes = sum(page_bytes)
    dram_stats = runtime.session_metrics(catalog.session_id)
    if dram_stats.dram_bytes != expected_full_bytes:
        raise AssertionError("runtime DRAM accounting differs from exact tensor bytes")

    clear_full_attention(cache)
    snapshots.append(allocator_snapshot(torch, "dram_backed_no_full_attention_hbm"))
    for layer_id, (recurrent, convolution) in recurrent_sentinels.items():
        layer = cache.layers[layer_id]
        if getattr(layer, "recurrent_states", None) is not recurrent:
            raise AssertionError(f"recurrent layer {layer_id} was modified")
        if getattr(layer, "conv_states", None) is not convolution:
            raise AssertionError(f"convolution layer {layer_id} was modified")

    context = context_for(catalog)
    full_result = runtime.restore_all(context, catalog)
    snapshots.append(allocator_snapshot(torch, "full_pages_restored"))
    full_active = assemble_pages(full_result.ordered_payloads)
    assert_expected(full_active)
    positions = write_active_cache(cache, full_active)
    if positions != tuple(range(args.tokens)):
        raise AssertionError("full restore lost logical positions")
    snapshots.append(allocator_snapshot(torch, "full_cache_assembled", trim=False))
    full_operation = asdict(full_result.operation)

    # Drop the packed cache before selecting a smaller Page set.  The runtime
    # then owns exactly the selected restored Pages and can physically release
    # the remainder on its transactional commit.
    clear_full_attention(cache)
    del full_active, full_result
    selected_indices = tuple(
        round(i * (page_count - 1) / (args.selected_pages - 1))
        if args.selected_pages > 1 else 0
        for i in range(args.selected_pages)
    )
    selected_ids = tuple(f"p{index:04d}" for index in selected_indices)
    selected_context = context_for(
        catalog, tuple(page.page_id for page in catalog.pages)
    )
    selected_result = runtime.restore_selected(
        selected_context, catalog, plan_for(catalog, selected_ids)
    )
    snapshots.append(allocator_snapshot(torch, "selected_pages_resident"))
    selected_active = assemble_pages(selected_result.ordered_payloads)
    assert_expected(selected_active)
    selected_positions = write_active_cache(cache, selected_active)
    expected_positions = tuple(
        position
        for index in selected_indices
        for position in range(index * args.page_size, (index + 1) * args.page_size)
    )
    if selected_positions != expected_positions:
        raise AssertionError("selected restore lost non-contiguous logical positions")
    snapshots.append(allocator_snapshot(torch, "selected_cache_assembled", trim=False))
    selected_operation = asdict(selected_result.operation)

    clear_full_attention(cache)
    del selected_active, selected_result
    reactivated = runtime.restore_all(context_for(catalog, selected_ids), catalog)
    assert_expected(assemble_pages(reactivated.ordered_payloads))
    snapshots.append(allocator_snapshot(torch, "full_pages_reactivated"))
    reactivation_operation = asdict(reactivated.operation)
    final_stats = asdict(runtime.session_metrics(catalog.session_id))

    del reactivated
    runtime.release_session(catalog.session_id)
    snapshots.append(allocator_snapshot(torch, "released"))
    result = {
        "model": str(args.model),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "tokens": args.tokens,
        "page_size": args.page_size,
        "page_count": page_count,
        "selected_page_ids": selected_ids,
        "full_attention_layer_ids": full_layer_ids,
        "recurrent_layer_count": len(recurrent_ids),
        "kv_shape_per_layer": shape,
        "expected_full_kv_bytes": expected_full_bytes,
        "expected_selected_kv_bytes": sum(page_bytes[index] for index in selected_indices),
        "full_restore_operation": full_operation,
        "selected_restore_operation": selected_operation,
        "reactivation_operation": reactivation_operation,
        "final_runtime_stats": final_stats,
        "byte_exact_validation": True,
        "logical_positions_validation": True,
        "recurrent_state_identity_preserved": True,
        "allocator_snapshots": snapshots,
        "scope": {
            "uses_real_transformers_cache_layout": True,
            "loads_model_weights": False,
            "tests_logits": False,
            "runs_model_forward": False,
            "uses_nonempty_recurrent_state_sentinels": True,
            "assembly_is_contiguous_concat": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
