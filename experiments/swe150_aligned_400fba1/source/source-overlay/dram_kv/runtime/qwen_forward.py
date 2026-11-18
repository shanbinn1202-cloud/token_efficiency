"""Safe pre-prefill bridge from a DRAM selection to a Qwen hybrid cache.

The module is intentionally dependency-free at import time.  The top-level
Qwen model continues to see dense *physical* cache positions for causal-mask
construction.  A patched full-attention implementation receives a separate
logical-position keyword for RoPE.  Recurrent/conv cache layers are neither
assembled nor written by this bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ..schema import PageCatalog, SelectionPlan, TurnContext
from .qwen import QwenKVPage, assemble_pages, write_active_cache

if TYPE_CHECKING:
    from ..backends.dram import DramRuntimeBackend


@dataclass(frozen=True)
class QwenPreparationReceipt:
    """Capability identifying one exact, synchronously restored selection."""

    session_id: str
    turn_id: int
    preparation_id: str
    ordered_page_ids: tuple[str, ...]


@dataclass(frozen=True)
class PreparedQwenPrefill:
    """A cache already bound to one working set, ready for formal prefill."""

    cache: Any
    receipt: QwenPreparationReceipt
    active_logical_positions: tuple[int, ...]
    suffix_logical_positions: tuple[int, ...]
    cache_positions: tuple[int, ...]
    attention_mask_length: int
    logical_position_kwarg: str

    @property
    def physical_prefix_length(self) -> int:
        return len(self.active_logical_positions)

    @property
    def suffix_token_count(self) -> int:
        return len(self.suffix_logical_positions)

    def model_kwargs(
        self,
        torch: Any,
        *,
        device: Any,
        batch_size: int = 1,
        attention_mask_dtype: Any | None = None,
    ) -> dict[str, Any]:
        """Materialize kwargs accepted by a patched Transformers Qwen model.

        ``position_ids`` and ``cache_position`` are physical and contiguous.
        Only ``logical_position_kwarg`` contains original trajectory positions.
        The latter must be consumed exclusively by the full-attention RoPE
        patch; it must not replace the top model's causal-mask positions.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        dtype = attention_mask_dtype
        if dtype is None:
            dtype = torch.long
        physical_rows = [list(self.cache_positions) for _ in range(batch_size)]
        logical_rows = [list(self.suffix_logical_positions) for _ in range(batch_size)]
        return {
            "past_key_values": self.cache,
            "attention_mask": torch.ones(
                (batch_size, self.attention_mask_length), dtype=dtype, device=device
            ),
            "cache_position": torch.tensor(
                list(self.cache_positions), dtype=torch.long, device=device
            ),
            "position_ids": torch.tensor(
                physical_rows, dtype=torch.long, device=device
            ),
            self.logical_position_kwarg: torch.tensor(
                logical_rows, dtype=torch.long, device=device
            ),
        }


class QwenForwardBridge:
    """Restore, validate, assemble, and bind selected Pages before prefill."""

    def __init__(
        self,
        backend: DramRuntimeBackend[QwenKVPage],
        *,
        logical_position_kwarg: str = "dram_kv_logical_position_ids",
        cat: Callable[[tuple[Any, ...], int], Any] | None = None,
    ) -> None:
        if not logical_position_kwarg or not logical_position_kwarg.isidentifier():
            raise ValueError("logical_position_kwarg must be a Python identifier")
        self.backend = backend
        self.logical_position_kwarg = logical_position_kwarg
        self.cat = cat

    def restore_selection(
        self,
        context: TurnContext,
        catalog: PageCatalog,
        plan: SelectionPlan,
    ) -> QwenPreparationReceipt:
        """Synchronously restore an exact plan and return its capability."""

        plan.validate_against(context, catalog)
        result = self.backend.prepare_before_prefill(context, catalog, plan)
        if result.ordered_page_ids != plan.ordered_page_ids:
            raise RuntimeError("runtime restored Pages in an order different from the plan")
        return QwenPreparationReceipt(
            context.session_id,
            context.turn_id,
            self.backend.preparation_id(context.session_id, context.turn_id),
            result.ordered_page_ids,
        )

    def prepare_before_prefill(
        self,
        context: TurnContext,
        catalog: PageCatalog,
        plan: SelectionPlan,
        cache: Any,
        *,
        suffix_logical_start: int,
        suffix_token_count: int,
        live_tail: QwenKVPage | None = None,
    ) -> PreparedQwenPrefill:
        """One-step production boundary; returns only after cache installation."""

        receipt = self.restore_selection(context, catalog, plan)
        return self.bind_prepared(
            context,
            catalog,
            plan,
            receipt,
            cache,
            suffix_logical_start=suffix_logical_start,
            suffix_token_count=suffix_token_count,
            live_tail=live_tail,
        )

    def bind_prepared(
        self,
        context: TurnContext,
        catalog: PageCatalog,
        plan: SelectionPlan,
        receipt: QwenPreparationReceipt,
        cache: Any,
        *,
        suffix_logical_start: int,
        suffix_token_count: int,
        live_tail: QwenKVPage | None = None,
    ) -> PreparedQwenPrefill:
        """Bind one still-current receipt to ``cache`` before model execution.

        This public split phase exists for orchestrators that restore and bind in
        adjacent stages.  The backend capability check makes a stale/superseded
        receipt fail before any cache layer is modified.
        """

        plan.validate_against(context, catalog)
        _validate_receipt(receipt, context, plan)
        if suffix_token_count <= 0:
            raise ValueError("formal prefill suffix must contain at least one token")
        pages_by_id = catalog.by_id()
        unknown_history = set(context.historical_page_ids) - set(pages_by_id)
        if unknown_history:
            raise ValueError(
                f"context history is absent from Page catalog: {sorted(unknown_history)}"
            )
        historical = tuple(pages_by_id[page_id] for page_id in context.historical_page_ids)
        if not historical:
            raise ValueError("Qwen working-set prefill requires sealed history")
        if any(page.turn_id >= context.turn_id for page in historical):
            raise ValueError("context history contains an unsealed current/future Page")
        sealed_history_end = max(page.logical_position_end for page in historical)
        if suffix_logical_start < sealed_history_end:
            raise ValueError(
                "formal suffix cannot start before sealed-history end: "
                f"sealed_end={sealed_history_end} actual={suffix_logical_start}"
            )
        live_positions = () if live_tail is None else live_tail.logical_positions
        expected_live_positions = tuple(range(sealed_history_end, suffix_logical_start))
        if live_positions != expected_live_positions:
            raise ValueError(
                "live tail must exactly bridge sealed history to the formal suffix: "
                f"expected={expected_live_positions} actual={live_positions}"
            )

        # Capability validation happens before assembly or cache mutation.
        payloads = self.backend.prepared_payloads(
            context.session_id,
            context.turn_id,
            preparation_id=receipt.preparation_id,
        )
        if len(payloads) != len(receipt.ordered_page_ids):
            raise RuntimeError("prepared payload count differs from selected Page count")
        for page_id, payload in zip(receipt.ordered_page_ids, payloads):
            record = pages_by_id[page_id]
            expected = tuple(
                range(record.logical_position_start, record.logical_position_end)
            )
            if payload.logical_positions != expected:
                raise RuntimeError(
                    f"restored Page {page_id!r} logical positions differ from catalog"
                )

        active_payloads = payloads + (() if live_tail is None else (live_tail,))
        active = assemble_pages(active_payloads, cat=self.cat)
        expected_selected = tuple(
            position
            for page_id in receipt.ordered_page_ids
            for position in range(
                pages_by_id[page_id].logical_position_start,
                pages_by_id[page_id].logical_position_end,
            )
        )
        expected_active = expected_selected + live_positions
        if active.logical_positions != expected_active:
            raise RuntimeError("assembled active logical positions differ from selection")

        recurrent_snapshot = _snapshot_unmanaged_layers(cache, active.layers)
        written_positions = write_active_cache(cache, active)
        _assert_unmanaged_layers_unchanged(cache, recurrent_snapshot)
        if written_positions != expected_active:
            raise RuntimeError("cache writer returned unexpected logical positions")

        physical_prefix = len(expected_active)
        suffix_logical = tuple(
            range(suffix_logical_start, suffix_logical_start + suffix_token_count)
        )
        cache_positions = tuple(
            range(physical_prefix, physical_prefix + suffix_token_count)
        )
        attention_length = physical_prefix + suffix_token_count
        if attention_length != cache_positions[-1] + 1:
            raise RuntimeError("physical attention-mask length invariant failed")
        return PreparedQwenPrefill(
            cache=cache,
            receipt=receipt,
            active_logical_positions=expected_active,
            suffix_logical_positions=suffix_logical,
            cache_positions=cache_positions,
            attention_mask_length=attention_length,
            logical_position_kwarg=self.logical_position_kwarg,
        )


def _validate_receipt(
    receipt: QwenPreparationReceipt,
    context: TurnContext,
    plan: SelectionPlan,
) -> None:
    if receipt.session_id != context.session_id:
        raise RuntimeError("preparation receipt belongs to another session")
    if receipt.turn_id != context.turn_id:
        raise RuntimeError("preparation receipt belongs to another turn")
    if receipt.ordered_page_ids != plan.ordered_page_ids:
        raise RuntimeError("preparation receipt belongs to another selection plan")


def _snapshot_unmanaged_layers(
    cache: Any, active_layers: tuple[Any, ...]
) -> tuple[tuple[int, Any, tuple[tuple[str, Any], ...]], ...]:
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError(f"cache {type(cache).__name__} has no mutable layers")
    managed = {layer.layer_id for layer in active_layers}
    snapshots = []
    for layer_id, layer in enumerate(layers):
        if layer_id in managed:
            continue
        attributes = tuple(sorted(vars(layer).items())) if hasattr(layer, "__dict__") else ()
        snapshots.append((layer_id, layer, attributes))
    return tuple(snapshots)


def _assert_unmanaged_layers_unchanged(
    cache: Any,
    snapshot: tuple[tuple[int, Any, tuple[tuple[str, Any], ...]], ...],
) -> None:
    layers = cache.layers
    for layer_id, original_layer, attributes in snapshot:
        if layer_id >= len(layers) or layers[layer_id] is not original_layer:
            raise RuntimeError(f"bridge replaced unmanaged cache layer {layer_id}")
        if hasattr(original_layer, "__dict__"):
            current = vars(original_layer)
            if set(current) != {name for name, _ in attributes}:
                raise RuntimeError(f"bridge changed unmanaged cache layer {layer_id}")
            if any(current[name] is not value for name, value in attributes):
                raise RuntimeError(
                    f"bridge reset recurrent/conv state in cache layer {layer_id}"
                )
