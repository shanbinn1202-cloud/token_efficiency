"""Native R-SWA block recycling with a fixed initial prefix across requests."""

from vllm.tebench.native_window import BUDGET, PREFIXES, gap_blocks
from vllm.v1.core.single_type_kv_cache_manager import RSWAManager


class NativeWindowManager(RSWAManager):
    def remove_skipped_blocks(
        self, request_id, processed_computed_tokens, num_prompt_tokens=None
    ):
        first, last = gap_blocks(
            PREFIXES[request_id], processed_computed_tokens, BUDGET, self.block_size
        )
        self._remove_blocks_in_range(request_id, first, last)

    def get_num_blocks_to_allocate(
        self,
        request_id,
        num_tokens,
        new_computed_blocks,
        total_computed_tokens,
        num_local_computed_tokens,
        num_tokens_main_model,
        apply_admission_cap=False,
    ):
        count = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        if not self.req_to_blocks.get(request_id):
            assert not new_computed_blocks and not num_local_computed_tokens
            first, last = gap_blocks(
                PREFIXES[request_id], total_computed_tokens, BUDGET, self.block_size
            )
            count -= last - first
        assert count >= 0
        return count

    def allocate_external_computed_blocks(
        self, request_id, num_local_computed_tokens, num_external_computed_tokens
    ):
        assert num_local_computed_tokens == 0 and not self.enable_caching
        if not num_external_computed_tokens:
            return
        blocks = self.req_to_blocks[request_id]
        assert not blocks
        count = (num_external_computed_tokens + self.block_size - 1) // self.block_size
        first, last = gap_blocks(
            PREFIXES[request_id], num_external_computed_tokens, BUDGET, self.block_size
        )
        allocated = self.block_pool.get_new_blocks(count - (last - first))
        iterator = iter(allocated)
        blocks.extend(
            self._null_block if first <= i < last else next(iterator)
            for i in range(count)
        )
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in allocated)
