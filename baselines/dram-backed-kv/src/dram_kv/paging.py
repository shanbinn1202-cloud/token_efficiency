"""Logical Page construction and active-set accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import PageCatalog, PageRecord


@dataclass(frozen=True)
class TokenSegment:
    turn_id: int
    role: str
    token_start: int
    token_end: int
    mandatory: bool = False
    files: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    tests: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    summary: str | None = None
    lexical_handles: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActiveSetDelta:
    added: tuple[str, ...]
    retained: tuple[str, ...]
    removed: tuple[str, ...]
    reactivated: tuple[str, ...]


def build_page_catalog(
    session_id: str,
    segments: Iterable[TokenSegment],
    *,
    page_size: int,
    bytes_per_token: int,
    resident_page_ids: Iterable[str] = (),
) -> PageCatalog:
    if page_size <= 0 or bytes_per_token < 0:
        raise ValueError("page_size must be positive and bytes_per_token non-negative")
    resident = set(resident_page_ids)
    pages: list[PageRecord] = []
    for segment in sorted(segments, key=lambda item: item.token_start):
        if segment.token_start < 0 or segment.token_end <= segment.token_start:
            raise ValueError("TokenSegment span must be non-empty")
        first_page = segment.token_start // page_size
        last_page = (segment.token_end - 1) // page_size
        for logical_page in range(first_page, last_page + 1):
            start = max(segment.token_start, logical_page * page_size)
            end = min(segment.token_end, (logical_page + 1) * page_size)
            page_id = f"p{logical_page:08d}:{start}-{end}"
            pages.append(PageRecord(
                page_id=page_id,
                turn_id=segment.turn_id,
                role=segment.role,
                token_start=start,
                token_end=end,
                logical_position_start=start,
                logical_position_end=end,
                byte_size=(end - start) * bytes_per_token,
                mandatory=segment.mandatory,
                files=segment.files,
                symbols=segment.symbols,
                tests=segment.tests,
                errors=segment.errors,
                lexical_handles=segment.lexical_handles,
                summary=segment.summary,
                hbm_resident=page_id in resident,
            ))
    return PageCatalog(session_id=session_id, pages=tuple(pages))


def active_set_delta(
    current: Sequence[str],
    previous: Sequence[str],
    ever_selected: Iterable[str] = (),
) -> ActiveSetDelta:
    current_set = set(current)
    previous_set = set(previous)
    ever = set(ever_selected)
    return ActiveSetDelta(
        added=tuple(page for page in current if page not in previous_set),
        retained=tuple(page for page in current if page in previous_set),
        removed=tuple(page for page in previous if page not in current_set),
        reactivated=tuple(page for page in current if page not in previous_set and page in ever),
    )
