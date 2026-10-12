"""Minimal text-embedding contract for archived offline selectors."""

from __future__ import annotations

from typing import Protocol, Sequence


class TextEmbedder(Protocol):
    def encode(self, text: str) -> Sequence[float]: ...


__all__ = ["TextEmbedder"]
