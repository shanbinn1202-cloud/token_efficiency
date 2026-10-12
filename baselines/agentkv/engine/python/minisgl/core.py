from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.eviction.query_ring import QueryRingBufferPool, QueryRingMetadata
    from minisgl.kvcache import BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    # Req has two coordinate systems once session compaction is enabled:
    # physical indices address this request's compacted token/page-table row,
    # while logical positions are the original un-compacted context positions
    # used for RoPE and session continuation checks.
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    # Physical cached prefix length. page_table[table_idx, :physical_cached_len] has
    # valid KV pages. After StreamingLLM compaction this can be smaller than
    # logical_cached_len because only selected cached tokens are retained.
    physical_cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    # Public HTTP requests are session-cache-only and should always set this.
    # None is kept only for internal dummy requests.
    session_id: str | None = None
    # Rendered full prompt text for this turn. Used to reconstruct and validate
    # append-only session text when clients detokenize and re-tokenize history.
    session_text: str | None = None
    # Logical prompt length before this turn's generated tokens. Tokens after
    # this offset are decoded back into full_text when the session parks.
    session_prompt_len: int | None = None
    # Full logical token history for this session turn. For compacted sessions,
    # input_ids only stores retained physical tokens plus the current suffix.
    full_input_ids: torch.Tensor | None = None
    # Phase labels mirror input_ids/full_input_ids. They drive legacy KV phase
    # logic and are ignored when the configured query ring has a single stage.
    phase_ids: torch.Tensor | None = None
    full_phase_ids: torch.Tensor | None = None
    # Semantic phases are canonical labels. Stage-Q plans route them to query
    # ring stages, while KV phase logic continues to use phase_ids.
    semantic_phase_ids: torch.Tensor | None = None
    # Logical end position of the cached KV prefix before the current extend.
    # This drives the next RoPE position and may exceed physical_cached_len after compact.
    logical_cached_len: int | None = None
    # Logical end position of the tokens currently present on device.
    logical_device_len: int | None = None
    # retained_positions[i] is the original logical position for physical
    # cached token input_ids[i]. Its length must stay equal to physical_cached_len.
    retained_positions: torch.Tensor | None = None
    # Optional per-layer/per-head retained logical positions for eviction
    # policies. Attention does not consume this metadata.
    retained_positions_by_layer_head: torch.Tensor | None = None
    # Number of tokens generated in this turn; used for remain_len and compact
    # interval scheduling.
    generated_len: int = 0
    phase_text_tail: str = ""
    # Intermediate chunk-prefill requests materialize prompt KV only. They must
    # not enter decode or return sampled output tokens until the final chunk.
    prefill_only: bool = False
    _full_input_ids_tail: list[int] = field(default_factory=list, init=False, repr=False)
    _full_phase_ids_tail: list[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.physical_cached_len < self.device_len
        if self.session_id is not None and self.session_prompt_len is None:
            base_input_ids = (
                self.full_input_ids if self.full_input_ids is not None else self.input_ids
            )
            self.session_prompt_len = len(base_input_ids)
        if self.full_input_ids is not None:
            assert self.full_input_ids.is_cpu
        if self.phase_ids is None:
            self.phase_ids = torch.zeros(
                len(self.input_ids),
                dtype=torch.int32,
                device="cpu",
            )
        if self.full_phase_ids is not None:
            assert self.full_phase_ids.is_cpu
        elif self.full_input_ids is not None:
            self.full_phase_ids = torch.zeros(
                len(self.full_input_ids),
                dtype=torch.int32,
                device="cpu",
            )
        assert self.phase_ids.is_cpu
        assert len(self.phase_ids) == len(self.input_ids)
        if self.full_phase_ids is not None:
            assert len(self.full_phase_ids) == len(self.full_input_ids)
        if self.semantic_phase_ids is None:
            self.semantic_phase_ids = self.phase_ids.clone()
        assert self.semantic_phase_ids.is_cpu
        assert len(self.semantic_phase_ids) == len(self.input_ids)
        if self.logical_cached_len is None:
            self.logical_cached_len = self.physical_cached_len
        if self.logical_device_len is None:
            self.logical_device_len = self.logical_cached_len + self.extend_len
        if self.retained_positions is None:
            self.retained_positions = torch.arange(
                self.logical_cached_len,
                dtype=torch.int32,
                device="cpu",
            )
        assert len(self.retained_positions) == self.physical_cached_len
        if self.retained_positions_by_layer_head is not None:
            assert self.retained_positions_by_layer_head.is_cpu
            assert self.retained_positions_by_layer_head.ndim == 3
            assert self.retained_positions_by_layer_head.shape[2] == self.physical_cached_len

    @property
    def remain_len(self) -> int:
        return self.output_len - self.generated_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.physical_cached_len

    def complete_one(self) -> None:
        assert self.logical_cached_len is not None
        assert self.logical_device_len is not None
        assert self.retained_positions is not None
        self._append_new_cached_positions()
        self.physical_cached_len = self.device_len
        self.device_len += 1
        self.logical_cached_len = self.logical_device_len
        self.logical_device_len += 1
        self.generated_len += 1

    def complete_prefill_chunk(self) -> None:
        assert self.logical_cached_len is not None
        assert self.logical_device_len is not None
        assert self.retained_positions is not None
        self._append_new_cached_positions()
        self.physical_cached_len = self.device_len
        self.logical_cached_len = self.logical_device_len

    def _append_new_cached_positions(self) -> None:
        assert self.logical_cached_len is not None
        assert self.logical_device_len is not None
        assert self.retained_positions is not None
        new_positions = torch.arange(
            self.logical_cached_len,
            self.logical_device_len,
            dtype=torch.int32,
            device="cpu",
        )
        self.retained_positions = torch.cat([self.retained_positions, new_positions])
        if self.retained_positions_by_layer_head is not None:
            num_layers, num_heads, _ = self.retained_positions_by_layer_head.shape
            new_positions_by_head = new_positions.view(1, 1, -1).expand(
                num_layers, num_heads, -1
            )
            self.retained_positions_by_layer_head = torch.cat(
                [self.retained_positions_by_layer_head, new_positions_by_head],
                dim=2,
            )

    def append_host(
        self,
        next_token: torch.Tensor,
        *,
        phase_id: int | None = None,
        semantic_phase_id: int | None = None,
        phase_text_tail: str | None = None,
    ) -> None:
        self.input_ids = torch.cat([self.input_ids, next_token])
        assert self.phase_ids is not None
        assert self.semantic_phase_ids is not None
        if phase_id is None:
            next_phase = self.phase_ids[-1:].clone()
        else:
            next_phase = torch.tensor([phase_id], dtype=torch.int32, device="cpu")
        if semantic_phase_id is None:
            next_semantic_phase = self.semantic_phase_ids[-1:].clone()
        else:
            next_semantic_phase = torch.tensor(
                [semantic_phase_id],
                dtype=torch.int32,
                device="cpu",
            )
        self.phase_ids = torch.cat([self.phase_ids, next_phase])
        self.semantic_phase_ids = torch.cat([self.semantic_phase_ids, next_semantic_phase])
        if phase_text_tail is not None:
            self.phase_text_tail = phase_text_tail
        if self.full_input_ids is not None:
            self._full_input_ids_tail.append(int(next_token.item()))
            if self.full_phase_ids is not None:
                self._full_phase_ids_tail.append(int(next_phase.item()))

    def materialize_full_host(self) -> None:
        if self.full_input_ids is not None and self._full_input_ids_tail:
            tail = torch.tensor(
                self._full_input_ids_tail,
                dtype=self.full_input_ids.dtype,
                device="cpu",
            )
            self.full_input_ids = torch.cat([self.full_input_ids, tail])
            self._full_input_ids_tail.clear()
        if self.full_phase_ids is not None and self._full_phase_ids_tail:
            tail = torch.tensor(
                self._full_phase_ids_tail,
                dtype=self.full_phase_ids.dtype,
                device="cpu",
            )
            self.full_phase_ids = torch.cat([self.full_phase_ids, tail])
            self._full_phase_ids_tail.clear()

    @property
    def can_decode(self) -> bool:
        return (not self.prefill_only) and self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"physical_cached_len={self.physical_cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    stage_ids: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    query_ring_metadata: QueryRingMetadata | None = field(default=None, init=False)
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    query_ring: QueryRingBufferPool = field(init=False)
    _batch: Batch | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
