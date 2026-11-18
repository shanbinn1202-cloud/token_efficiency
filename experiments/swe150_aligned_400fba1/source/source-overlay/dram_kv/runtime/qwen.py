"""Qwen hybrid-cache payloads for the DRAM runtime.

This module deliberately uses duck typing and does not import torch or
transformers at import time.  A Qwen KV Page contains only token-growing
full-attention K/V tensors.  Linear/recurrent cache state has a different
lifecycle and is never copied into, sliced as, or restored from a Page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class FullAttentionKV:
    """K/V for one full-attention layer in one logical Page."""

    layer_id: int
    keys: Any
    values: Any


@dataclass(frozen=True)
class QwenKVPage:
    """An independently owned Page cut from a dense Qwen hybrid cache."""

    logical_positions: tuple[int, ...]
    layers: tuple[FullAttentionKV, ...]

    @property
    def token_count(self) -> int:
        return len(self.logical_positions)

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(layer.layer_id for layer in self.layers)


@dataclass(frozen=True)
class ActiveQwenKV:
    """Ordered physical K/V and their possibly non-contiguous RoPE positions."""

    logical_positions: tuple[int, ...]
    layers: tuple[FullAttentionKV, ...]

    @property
    def token_count(self) -> int:
        return len(self.logical_positions)


def full_attention_layers(cache: Any) -> dict[int, Any]:
    """Return initialized token-growing layers, excluding recurrent layers."""

    result: dict[int, Any] = {}
    for layer_id, layer in enumerate(getattr(cache, "layers", ())):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if (keys is None) != (values is None):
            raise RuntimeError(
                f"cache layer {layer_id} has only one of full-attention K/V"
            )
        if keys is None or values is None:
            continue
        if _ndim(keys) >= 3 and _ndim(values) >= 3:
            result[layer_id] = layer
    if not result:
        raise RuntimeError(
            f"cache {type(cache).__name__} exposes no initialized full-attention layers"
        )
    return result


def extract_page(
    cache: Any,
    logical_start: int,
    logical_end: int,
    *,
    cache_logical_start: int = 0,
) -> QwenKVPage:
    """Clone a contiguous logical span from every full-attention cache layer.

    ``cache_logical_start`` identifies the logical position represented by
    physical cache index zero.  Selective/non-contiguous caches should not be
    passed here; seal Pages while their source span is still contiguous.
    """

    if logical_start < cache_logical_start or logical_end <= logical_start:
        raise ValueError("Page logical span must be non-empty and inside the cache")
    physical_start = logical_start - cache_logical_start
    physical_end = logical_end - cache_logical_start
    source_layers = full_attention_layers(cache)
    lengths = {_seq_len(layer.keys) for layer in source_layers.values()}
    lengths.update(_seq_len(layer.values) for layer in source_layers.values())
    if len(lengths) != 1:
        raise RuntimeError(f"full-attention K/V lengths differ: {sorted(lengths)}")
    cache_length = lengths.pop()
    if physical_end > cache_length:
        raise ValueError(
            f"Page span ends outside cache: physical_end={physical_end} "
            f"cache_length={cache_length}"
        )

    payloads = []
    for layer_id, layer in source_layers.items():
        _validate_pair(layer_id, layer.keys, layer.values)
        payloads.append(
            FullAttentionKV(
                layer_id,
                _slice_clone(layer.keys, physical_start, physical_end),
                _slice_clone(layer.values, physical_start, physical_end),
            )
        )
    return QwenKVPage(tuple(range(logical_start, logical_end)), tuple(payloads))


def extract_suffix_page(
    cache: Any,
    logical_start: int,
    logical_end: int,
    *,
    suffix_logical_start: int,
    physical_prefix_length: int,
) -> QwenKVPage:
    """Extract a newly appended Page from a selectively assembled cache.

    A selective prefill cache is physically ``selected history + new suffix``
    even though the suffix retains its original, usually much larger, logical
    positions.  This helper translates that uniform suffix offset before
    delegating to :func:`extract_page`.  It must never be used for Pages in the
    non-contiguous selected prefix.
    """

    if physical_prefix_length < 0:
        raise ValueError("physical_prefix_length cannot be negative")
    if logical_start < suffix_logical_start:
        raise ValueError("requested Page starts before the contiguous suffix")
    cache_logical_start = suffix_logical_start - physical_prefix_length
    return extract_page(
        cache,
        logical_start,
        logical_end,
        cache_logical_start=cache_logical_start,
    )


class QwenKVPageAdapter:
    """Synchronous tensor-tree transfers used by :class:`DramKVRuntime`."""

    def __init__(self, device: Any = "cuda") -> None:
        self.device = device

    def nbytes(self, payload: QwenKVPage) -> int:
        _validate_page(payload)
        return sum(
            _tensor_nbytes(tensor)
            for layer in payload.layers
            for tensor in (layer.keys, layer.values)
        )

    def to_cpu(self, payload: QwenKVPage, *, pin: bool) -> QwenKVPage:
        _validate_page(payload)
        return _map_page(payload, lambda tensor: _copy_to(tensor, "cpu", pin=pin))

    def to_device(self, payload: QwenKVPage) -> QwenKVPage:
        _validate_page(payload)
        return _map_page(payload, lambda tensor: _copy_to(tensor, self.device, pin=False))


def assemble_pages(
    pages: Iterable[QwenKVPage],
    *,
    cat: Callable[[tuple[Any, ...], int], Any] | None = None,
) -> ActiveQwenKV:
    """Concatenate selected Pages in logical order along the sequence axis."""

    ordered = tuple(pages)
    if not ordered:
        raise ValueError("cannot assemble an empty Page sequence")
    for page in ordered:
        _validate_page(page)
    layer_ids = ordered[0].layer_ids
    positions: list[int] = []
    previous: int | None = None
    for page in ordered:
        if page.layer_ids != layer_ids:
            raise RuntimeError(
                f"Page full-attention layers differ: expected={layer_ids} "
                f"actual={page.layer_ids}"
            )
        if previous is not None and page.logical_positions[0] <= previous:
            raise ValueError("Pages must be strictly ordered by logical position")
        positions.extend(page.logical_positions)
        previous = page.logical_positions[-1]

    concatenate = cat or _torch_cat
    active_layers = []
    for offset, layer_id in enumerate(layer_ids):
        keys = tuple(page.layers[offset].keys for page in ordered)
        values = tuple(page.layers[offset].values for page in ordered)
        _validate_compatible(layer_id, keys, "K")
        _validate_compatible(layer_id, values, "V")
        active_layers.append(
            FullAttentionKV(
                layer_id,
                concatenate(keys, -2),
                concatenate(values, -2),
            )
        )
    return ActiveQwenKV(tuple(positions), tuple(active_layers))


def write_active_cache(cache: Any, active: ActiveQwenKV) -> tuple[int, ...]:
    """Install active K/V into their original cache layers.

    Other layers, including all recurrent state, remain untouched.  The return
    value is the exact logical-position vector the caller must use for RoPE/
    cache positions during the following prefill.
    """

    _validate_active(active)
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError(f"cache {type(cache).__name__} has no mutable layers")
    for payload in active.layers:
        if payload.layer_id < 0 or payload.layer_id >= len(layers):
            raise RuntimeError(f"cache is missing full-attention layer {payload.layer_id}")
        layer = layers[payload.layer_id]
        if not hasattr(layer, "keys") or not hasattr(layer, "values"):
            raise RuntimeError(
                f"cache layer {payload.layer_id} is not a mutable full-attention layer"
            )
        _validate_pair(payload.layer_id, payload.keys, payload.values)
        current_k = getattr(layer, "keys", None)
        current_v = getattr(layer, "values", None)
        if current_k is not None and current_v is not None:
            if _ndim(current_k) < 3 or _ndim(current_v) < 3:
                raise RuntimeError(
                    f"refusing to overwrite recurrent cache layer {payload.layer_id}"
                )
            _validate_target(payload.layer_id, current_k, payload.keys, "K")
            _validate_target(payload.layer_id, current_v, payload.values, "V")
        layer.keys = payload.keys
        layer.values = payload.values
        if hasattr(layer, "is_initialized"):
            layer.is_initialized = True
    return active.logical_positions


def _validate_page(page: QwenKVPage) -> None:
    if not page.logical_positions:
        raise ValueError("Qwen KV Page cannot be empty")
    if tuple(sorted(set(page.logical_positions))) != page.logical_positions:
        raise ValueError("Page logical positions must be sorted and unique")
    if not page.layers:
        raise ValueError("Qwen KV Page has no full-attention layers")
    ids = page.layer_ids
    if tuple(sorted(set(ids))) != ids:
        raise ValueError("full-attention layer IDs must be sorted and unique")
    for layer in page.layers:
        _validate_pair(layer.layer_id, layer.keys, layer.values)
        if _seq_len(layer.keys) != page.token_count:
            raise RuntimeError(
                f"layer {layer.layer_id} token count does not match logical Page"
            )


def _validate_active(active: ActiveQwenKV) -> None:
    if not active.logical_positions:
        raise ValueError("active logical positions cannot be empty")
    if tuple(sorted(set(active.logical_positions))) != active.logical_positions:
        raise ValueError("active logical positions must be sorted and unique")
    if not active.layers:
        raise ValueError("active K/V has no full-attention layers")
    ids = tuple(layer.layer_id for layer in active.layers)
    if tuple(sorted(set(ids))) != ids:
        raise ValueError("active full-attention layer IDs must be sorted and unique")
    for layer in active.layers:
        _validate_pair(layer.layer_id, layer.keys, layer.values)
        if _seq_len(layer.keys) != len(active.logical_positions):
            raise RuntimeError(
                f"active layer {layer.layer_id} token count differs from logical positions"
            )


def _validate_pair(layer_id: int, keys: Any, values: Any) -> None:
    if _ndim(keys) < 3 or _ndim(values) < 3:
        raise RuntimeError(f"layer {layer_id} is not token-growing full attention")
    if tuple(keys.shape) != tuple(values.shape):
        raise RuntimeError(f"layer {layer_id} K/V shapes differ")
    if getattr(keys, "dtype", None) != getattr(values, "dtype", None):
        raise RuntimeError(f"layer {layer_id} K/V dtypes differ")
    if getattr(keys, "device", None) != getattr(values, "device", None):
        raise RuntimeError(f"layer {layer_id} K/V devices differ")


def _validate_compatible(layer_id: int, tensors: tuple[Any, ...], kind: str) -> None:
    first = tensors[0]
    first_shape = tuple(first.shape)
    for tensor in tensors[1:]:
        shape = tuple(tensor.shape)
        if shape[:-2] + shape[-1:] != first_shape[:-2] + first_shape[-1:]:
            raise RuntimeError(f"layer {layer_id} {kind} non-sequence shapes differ")
        if getattr(tensor, "dtype", None) != getattr(first, "dtype", None):
            raise RuntimeError(f"layer {layer_id} {kind} dtypes differ across Pages")
        if getattr(tensor, "device", None) != getattr(first, "device", None):
            raise RuntimeError(f"layer {layer_id} {kind} devices differ across Pages")


def _validate_target(layer_id: int, target: Any, source: Any, kind: str) -> None:
    if getattr(target, "dtype", None) != getattr(source, "dtype", None):
        raise RuntimeError(f"layer {layer_id} {kind} dtype differs from target cache")
    if getattr(target, "device", None) != getattr(source, "device", None):
        raise RuntimeError(f"layer {layer_id} {kind} device differs from target cache")


def _map_page(page: QwenKVPage, copy: Callable[[Any], Any]) -> QwenKVPage:
    return QwenKVPage(
        page.logical_positions,
        tuple(FullAttentionKV(layer.layer_id, copy(layer.keys), copy(layer.values))
              for layer in page.layers),
    )


def _slice_clone(tensor: Any, start: int, end: int) -> Any:
    slices = [slice(None)] * _ndim(tensor)
    slices[-2] = slice(start, end)
    detached = tensor.detach() if hasattr(tensor, "detach") else tensor
    result = detached[tuple(slices)]
    if not hasattr(result, "clone"):
        raise TypeError(f"tensor {type(tensor).__name__} cannot be cloned")
    return result.clone()


def _copy_to(tensor: Any, device: Any, *, pin: bool) -> Any:
    detached = tensor.detach() if hasattr(tensor, "detach") else tensor
    try:
        copied = detached.to(device, copy=True)
    except TypeError:
        copied = detached.to(device)
        if copied is detached:
            copied = detached.clone()
    if pin:
        if not hasattr(copied, "pin_memory"):
            raise TypeError(f"tensor {type(tensor).__name__} cannot be pinned")
        copied = copied.pin_memory()
    return copied


def _tensor_nbytes(tensor: Any) -> int:
    value = getattr(tensor, "nbytes", None)
    if value is not None:
        return int(value)
    return int(tensor.numel()) * int(tensor.element_size())


def _ndim(tensor: Any) -> int:
    return int(getattr(tensor, "ndim", len(getattr(tensor, "shape", ()))))


def _seq_len(tensor: Any) -> int:
    return int(tensor.shape[-2])


def _torch_cat(tensors: tuple[Any, ...], dim: int) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production dependency boundary
        raise RuntimeError("assembling real Qwen tensors requires torch") from exc
    return torch.cat(tensors, dim=dim)
