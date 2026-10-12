"""Tensor transfer boundary without importing a tensor framework."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar


PayloadT = TypeVar("PayloadT")


class TensorAdapter(Protocol[PayloadT]):
    """Operations the runtime needs from a model-specific tensor adapter.

    Transfer methods must finish before returning and must return independently
    owned storage. An asynchronous adapter must synchronize or expose a higher-
    level completion boundary before it is used with this runtime.
    """

    def nbytes(self, payload: PayloadT) -> int:
        ...

    def to_cpu(self, payload: PayloadT, *, pin: bool) -> PayloadT:
        ...

    def to_device(self, payload: PayloadT) -> PayloadT:
        ...


class DuckTensorAdapter:
    """Best-effort adapter for tensor-like objects (including torch tensors).

    Torch is deliberately not imported. Model integrations may instead provide
    a specialized adapter for nested layer KV structures or asynchronous copies.
    """

    def __init__(self, device: Any = "cuda") -> None:
        self.device = device

    def nbytes(self, payload: Any) -> int:
        value = getattr(payload, "nbytes", None)
        if value is not None:
            return int(value)
        return int(payload.numel()) * int(payload.element_size())

    def to_cpu(self, payload: Any, *, pin: bool) -> Any:
        try:
            # Force distinct backing even when the input already lives on CPU.
            copied = payload.to("cpu", copy=True)
        except TypeError:
            copied = payload.to("cpu")
            if copied is payload and hasattr(copied, "clone"):
                copied = copied.clone()
        if pin and hasattr(copied, "pin_memory"):
            copied = copied.pin_memory()
        return copied

    def to_device(self, payload: Any) -> Any:
        return payload.to(self.device)
