from __future__ import annotations

from contextlib import contextmanager
from typing import List, Tuple

import torch
from minisgl.core import Req
from minisgl.utils import div_ceil


class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.device = device
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size

    @property
    def available_size(self) -> int:
        return len(self.free_slots) * self.page_size

    @property
    def available_pages(self) -> int:
        return len(self.free_slots)

    def needed_pages(self, reqs: List[Req]) -> int:
        needed_pages = 0
        for req in reqs:
            first_page = div_ceil(req.physical_cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
        return needed_pages

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.physical_cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def free_req_pages(self, req: Req) -> None:
        indices = self.page_table[req.table_idx, : req.physical_cached_len]
        self._free(indices)

    def allocate_token_slots(self, num_tokens: int) -> torch.Tensor:
        if self.page_size != 1:
            raise NotImplementedError("token-slot allocation requires page_size == 1")
        return self._allocate(num_tokens)

    def free_token_slots(self, indices: torch.Tensor) -> None:
        self._free(indices)

    def check_integrity(self, *, owned_pages: int = 0) -> None:
        if len(self.free_slots) + owned_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" owned_pages({owned_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free
            yield
        finally:
            del self._free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            raise RuntimeError(
                f"Not enough KV cache pages: need {needed_pages}, have {free_pages}"
            )
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset : offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated
