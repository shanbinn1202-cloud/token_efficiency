from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QueryRingMetadata:
    table_indices_by_token: torch.Tensor
    stage_ids_by_token: torch.Tensor
    ring_locations_by_token: torch.Tensor
    positions_by_token: torch.Tensor
    source_token_indices: torch.Tensor


class QueryRingBufferPool:
    def __init__(
        self,
        *,
        max_table_slots: int,
        num_layers: int,
        capacity: int,
        num_q_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        num_stages: int = 1,
    ) -> None:
        self.capacity = capacity
        self.num_stages = num_stages
        self.q = torch.empty(
            (max_table_slots, num_layers, num_stages, capacity, num_q_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.positions = torch.full(
            (max_table_slots, num_stages, capacity),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.write_pos = torch.zeros(
            (max_table_slots, num_stages),
            dtype=torch.int32,
            device=device,
        )
        self.size = torch.zeros(
            (max_table_slots, num_stages),
            dtype=torch.int32,
            device=device,
        )
        self._max_read_entries = num_stages * capacity
        self._flat_locations_by_stage = torch.arange(
            self._max_read_entries,
            dtype=torch.long,
            device=device,
        ).view(num_stages, capacity)
        self._unordered_read_indices = torch.empty(
            (max_table_slots, self._max_read_entries),
            dtype=torch.long,
            device=device,
        )
        self._unordered_read_lengths = [0 for _ in range(max_table_slots)]
        self._unordered_full_rings = [False for _ in range(max_table_slots)]
        self.profile_enabled = False
        self._profile_counters: dict[str, int] = {}

    def set_profile_enabled(self, enabled: bool) -> None:
        self.profile_enabled = enabled

    def record_write_metadata_stats(
        self,
        *,
        phase: str,
        active_rows: int,
        padded_rows: int,
        written_rows: int,
    ) -> None:
        if not self.profile_enabled:
            return
        if phase not in {"prefill", "decode"}:
            phase = "unknown"
        layers = int(self.q.shape[1])
        self._profile_increment(f"{phase}_batches", 1)
        self._profile_increment(f"{phase}_active_rows", active_rows)
        self._profile_increment(f"{phase}_padded_rows", padded_rows)
        self._profile_increment(f"{phase}_written_rows", written_rows)
        self._profile_increment(f"{phase}_estimated_layer_write_rows", written_rows * layers)

    def profile_snapshot(self) -> dict[str, int]:
        return dict(self._profile_counters)

    def profile_delta_since(self, previous: dict[str, int]) -> dict[str, int]:
        current = self.profile_snapshot()
        keys = sorted(set(current) | set(previous))
        return {key: current.get(key, 0) - previous.get(key, 0) for key in keys}

    def _profile_increment(self, key: str, value: int) -> None:
        self._profile_counters[key] = self._profile_counters.get(key, 0) + int(value)

    def reset(self, table_idx: int) -> None:
        self.positions[table_idx].fill_(-1)
        self.write_pos[table_idx].zero_()
        self.size[table_idx].zero_()
        self._unordered_read_lengths[table_idx] = 0
        self._unordered_full_rings[table_idx] = False

    def prepare_write_metadata_from_reqs(
        self,
        reqs,
        positions_by_token: torch.Tensor,
        stage_ids_by_token: torch.Tensor | None = None,
    ) -> QueryRingMetadata:
        total_tokens = sum(req.extend_len for req in reqs)
        if positions_by_token.numel() != total_tokens:
            raise ValueError("positions_by_token length does not match request extend lengths")
        if stage_ids_by_token is not None and stage_ids_by_token.numel() != total_tokens:
            raise ValueError("stage_ids_by_token length does not match request extend lengths")
        if self.capacity <= 0:
            empty = torch.empty(0, dtype=torch.long, device=positions_by_token.device)
            return QueryRingMetadata(
                table_indices_by_token=empty,
                stage_ids_by_token=empty,
                ring_locations_by_token=empty,
                positions_by_token=empty.to(dtype=torch.int32),
                source_token_indices=empty,
            )

        table_indices = torch.empty(
            total_tokens,
            dtype=torch.long,
            device=positions_by_token.device,
        )
        ring_locations = torch.empty_like(table_indices)
        keep_mask = torch.zeros(total_tokens, dtype=torch.bool, device=positions_by_token.device)
        stage_ids = self._normalize_stage_ids(
            stage_ids_by_token,
            total_tokens=total_tokens,
            device=positions_by_token.device,
        )
        affected_slots: set[int] = set()
        all_token_indices = torch.arange(
            total_tokens,
            dtype=torch.long,
            device=positions_by_token.device,
        )
        offset = 0
        for req in reqs:
            length = req.extend_len
            if length <= 0:
                continue
            token_slice = slice(offset, offset + length)
            req_stage_ids = stage_ids[token_slice]
            token_indices = all_token_indices[token_slice]
            for stage in range(self.num_stages):
                mask = req_stage_ids == stage
                count = int(mask.sum().item())
                if count <= 0:
                    continue
                selected_tokens = token_indices[mask]
                old_write_pos = int(self.write_pos[req.table_idx, stage].item())
                kept_count = min(count, self.capacity)
                kept_tokens = selected_tokens[-kept_count:]
                table_indices[kept_tokens] = req.table_idx
                ring_locations[kept_tokens] = (
                    old_write_pos
                    + count
                    - kept_count
                    + torch.arange(
                        kept_count,
                        dtype=torch.long,
                        device=positions_by_token.device,
                    )
                ) % self.capacity
                keep_mask[kept_tokens] = True
                self.write_pos[req.table_idx, stage] = (old_write_pos + count) % self.capacity
                new_size = min(
                    self.capacity,
                    int(self.size[req.table_idx, stage].item()) + count,
                )
                self.size[req.table_idx, stage] = new_size
                affected_slots.add(req.table_idx)
            offset += length

        source_token_indices = all_token_indices[keep_mask]
        if source_token_indices.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=positions_by_token.device)
            return QueryRingMetadata(
                table_indices_by_token=empty,
                stage_ids_by_token=empty,
                ring_locations_by_token=empty,
                positions_by_token=empty.to(dtype=torch.int32),
                source_token_indices=empty,
            )
        writable_table_indices = table_indices[source_token_indices]
        writable_stage_ids = stage_ids[source_token_indices]
        writable_ring_locations = ring_locations[source_token_indices]
        positions = positions_by_token.to(dtype=torch.int32)[source_token_indices]
        self.positions[writable_table_indices, writable_stage_ids, writable_ring_locations] = (
            positions
        )
        for slot in affected_slots:
            self._refresh_unordered_read_indices(slot)
        return QueryRingMetadata(
            table_indices_by_token=writable_table_indices,
            stage_ids_by_token=writable_stage_ids,
            ring_locations_by_token=writable_ring_locations,
            positions_by_token=positions,
            source_token_indices=source_token_indices,
        )

    def prepare_write_metadata(
        self,
        *,
        table_indices_by_token: torch.Tensor,
        positions_by_token: torch.Tensor,
        stage_ids_by_token: torch.Tensor | None = None,
    ) -> QueryRingMetadata:
        if self.capacity <= 0:
            empty = torch.empty(0, dtype=torch.int64, device=table_indices_by_token.device)
            return QueryRingMetadata(
                table_indices_by_token=empty,
                stage_ids_by_token=empty,
                ring_locations_by_token=empty,
                positions_by_token=empty.to(dtype=torch.int32),
                source_token_indices=empty,
            )
        if table_indices_by_token.numel() != positions_by_token.numel():
            raise ValueError("table_indices_by_token and positions_by_token must match")
        if (
            stage_ids_by_token is not None
            and table_indices_by_token.numel() != stage_ids_by_token.numel()
        ):
            raise ValueError("table_indices_by_token and stage_ids_by_token must match")
        if table_indices_by_token.numel() == 0:
            empty = torch.empty(0, dtype=torch.int64, device=table_indices_by_token.device)
            return QueryRingMetadata(
                table_indices_by_token=empty,
                stage_ids_by_token=empty,
                ring_locations_by_token=empty,
                positions_by_token=empty.to(dtype=torch.int32),
                source_token_indices=empty,
            )

        table_indices = table_indices_by_token.to(dtype=torch.long)
        positions = positions_by_token.to(dtype=torch.int32)
        stage_ids = self._normalize_stage_ids(
            stage_ids_by_token,
            total_tokens=table_indices.numel(),
            device=table_indices.device,
        )
        write_mask = stage_ids >= 0
        source_token_indices = torch.arange(
            table_indices.numel(),
            dtype=torch.long,
            device=table_indices.device,
        )[write_mask]
        if source_token_indices.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=table_indices.device)
            return QueryRingMetadata(
                table_indices_by_token=empty,
                stage_ids_by_token=empty,
                ring_locations_by_token=empty,
                positions_by_token=empty.to(dtype=torch.int32),
                source_token_indices=empty,
            )
        table_indices = table_indices[source_token_indices]
        positions = positions[source_token_indices]
        stage_ids = stage_ids[source_token_indices]
        ring_locations = torch.empty_like(table_indices)
        affected_slots: set[int] = set()

        unique_slots = torch.unique_consecutive(table_indices)
        for slot_tensor in unique_slots:
            slot = int(slot_tensor.item())
            slot_mask = table_indices == slot
            for stage in range(self.num_stages):
                mask = slot_mask & (stage_ids == stage)
                count = int(mask.sum().item())
                if count <= 0:
                    continue
                offsets = torch.arange(count, dtype=torch.long, device=table_indices.device)
                old_write_pos = int(self.write_pos[slot, stage].item())
                ring_locations[mask] = (old_write_pos + offsets) % self.capacity
                self.write_pos[slot, stage] = (old_write_pos + count) % self.capacity
                new_size = min(self.capacity, int(self.size[slot, stage].item()) + count)
                self.size[slot, stage] = new_size
                affected_slots.add(slot)

        self.positions[table_indices, stage_ids, ring_locations] = positions
        for slot in affected_slots:
            self._refresh_unordered_read_indices(slot)
        return QueryRingMetadata(
            table_indices_by_token=table_indices,
            stage_ids_by_token=stage_ids,
            ring_locations_by_token=ring_locations,
            positions_by_token=positions,
            source_token_indices=source_token_indices,
        )

    def write_layer(
        self,
        *,
        layer_id: int,
        q: torch.Tensor,
        metadata: QueryRingMetadata | None,
    ) -> None:
        if metadata is None or self.capacity <= 0:
            return
        if q.shape[0] == metadata.table_indices_by_token.numel():
            q_to_write = q
        elif metadata.source_token_indices.numel() == metadata.table_indices_by_token.numel():
            q_to_write = q[metadata.source_token_indices]
        else:
            raise ValueError("q token dimension does not match query ring metadata")
        if q_to_write.shape[0] == 0:
            return
        self.q[
            metadata.table_indices_by_token,
            layer_id,
            metadata.stage_ids_by_token,
            metadata.ring_locations_by_token,
        ] = q_to_write

    def get_unordered(
        self,
        *,
        table_idx: int,
        layer_id: int,
        stages: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._read(table_idx=table_idx, layer_id=layer_id, ordered=False, stages=stages)

    def get_ordered(
        self,
        *,
        table_idx: int,
        layer_id: int,
        stages: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._read(table_idx=table_idx, layer_id=layer_id, ordered=True, stages=stages)

    def _normalize_stage_ids(
        self,
        stage_ids_by_token: torch.Tensor | None,
        *,
        total_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        if stage_ids_by_token is None:
            return torch.zeros(total_tokens, dtype=torch.long, device=device)
        stage_ids = stage_ids_by_token.to(device=device, dtype=torch.long)
        if self.num_stages == 1:
            return torch.where(stage_ids < 0, stage_ids, torch.zeros_like(stage_ids))
        return torch.where(
            stage_ids < 0,
            stage_ids,
            stage_ids.clamp(max=self.num_stages - 1),
        )

    def _read(
        self,
        *,
        table_idx: int,
        layer_id: int,
        ordered: bool,
        stages: tuple[int, ...] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not ordered and self._reads_all_stages(stages):
            return self._read_unordered_all_stages(table_idx=table_idx, layer_id=layer_id)

        q_parts = []
        position_parts = []
        read_stages = range(self.num_stages) if stages is None else stages
        for stage in read_stages:
            size = int(self.size[table_idx, stage].item())
            if size <= 0:
                continue
            if size < self.capacity:
                indices = torch.arange(size, dtype=torch.long, device=self.q.device)
            elif ordered:
                start = int(self.write_pos[table_idx, stage].item())
                indices = (
                    torch.arange(self.capacity, dtype=torch.long, device=self.q.device) + start
                ) % self.capacity
            else:
                indices = torch.arange(self.capacity, dtype=torch.long, device=self.q.device)
            q_parts.append(self.q[table_idx, layer_id, stage, indices])
            position_parts.append(self.positions[table_idx, stage, indices])
        if not q_parts:
            return (
                self.q[table_idx, layer_id, 0, :0],
                self.positions[table_idx, 0, :0],
            )
        return torch.cat(q_parts, dim=0), torch.cat(position_parts, dim=0)

    def _reads_all_stages(self, stages: tuple[int, ...] | None) -> bool:
        if stages is None:
            return True
        if len(stages) != self.num_stages:
            return False
        return all(stage == expected for expected, stage in enumerate(stages))

    def _refresh_unordered_read_indices(self, table_idx: int) -> None:
        if self.capacity <= 0:
            self._unordered_read_lengths[table_idx] = 0
            self._unordered_full_rings[table_idx] = False
            return

        read_len = 0
        full = True
        for stage in range(self.num_stages):
            size = int(self.size[table_idx, stage].item())
            if size != self.capacity:
                full = False
            if size <= 0:
                continue
            next_len = read_len + size
            self._unordered_read_indices[table_idx, read_len:next_len] = (
                self._flat_locations_by_stage[stage, :size]
            )
            read_len = next_len
        self._unordered_read_lengths[table_idx] = read_len
        self._unordered_full_rings[table_idx] = full

    def _read_unordered_all_stages(
        self,
        *,
        table_idx: int,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_q = self.q[table_idx, layer_id].reshape(
            self._max_read_entries,
            self.q.shape[-2],
            self.q.shape[-1],
        )
        flat_positions = self.positions[table_idx].reshape(self._max_read_entries)
        read_len = self._unordered_read_lengths[table_idx]
        if read_len <= 0:
            return flat_q[:0], flat_positions[:0]
        if self._unordered_full_rings[table_idx]:
            return flat_q, flat_positions
        indices = self._unordered_read_indices[table_idx, :read_len]
        return flat_q[indices], flat_positions[indices]
