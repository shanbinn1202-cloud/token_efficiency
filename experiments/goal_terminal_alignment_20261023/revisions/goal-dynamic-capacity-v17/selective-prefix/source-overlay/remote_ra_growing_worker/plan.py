"""Compact FullAttention dispatch: logical positions are never renumbered.

The forward cursor is the OLD sealed boundary. Only old selected Pages can be
restored; every new suffix token is forwarded once, including new unselected
Pages which must be published into DRAM before their transient slots are freed.
"""
from dataclasses import asdict, dataclass
from .contract import require, digest


@dataclass(frozen=True)
class MixedPlan:
    page_tokens: int
    old_seal: int
    new_seal: int
    prompt_tokens: int
    selected: tuple[int, ...]
    retained: tuple[int, ...]
    restored: tuple[int, ...]
    group_ids: tuple[int, ...]
    block_sizes: tuple[int, ...]
    prefix_blocks: tuple[tuple[int, ...], ...]
    suffix_blocks: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        require(type(self.page_tokens) is int and self.page_tokens > 0,
                "positive Page width required")
        require(all(type(n) is int for n in (self.old_seal, self.new_seal, self.prompt_tokens))
                and 0 <= self.old_seal <= self.new_seal <= self.prompt_tokens <= 131072
                and self.new_seal // self.page_tokens <= 4096
                and self.old_seal % self.page_tokens == self.new_seal % self.page_tokens == 0,
                "invalid canonical sealed boundaries")
        for pages in (self.selected, self.retained, self.restored):
            require(type(pages) is tuple and all(type(p) is int and p >= 0 for p in pages)
                    and tuple(sorted(set(pages))) == pages, "noncanonical logical Pages")
        old_selected = tuple(p for p in self.selected if p < self.old_seal // self.page_tokens)
        require(not set(self.retained) & set(self.restored) and
                tuple(sorted(self.retained + self.restored)) == old_selected and
                all(p < self.new_seal // self.page_tokens for p in self.selected),
                "retained/restored partition does not match backed selected Pages")
        require(all(type(x) is tuple for x in (self.group_ids, self.block_sizes, self.prefix_blocks, self.suffix_blocks))
                and all(type(x) is tuple for x in self.prefix_blocks + self.suffix_blocks)
                and self.group_ids and all(type(g) is int and 0 <= g < 128 for g in self.group_ids)
                and tuple(sorted(set(self.group_ids))) == self.group_ids and
                len(self.group_ids) == len(self.block_sizes) == len(self.prefix_blocks)
                == len(self.suffix_blocks), "FullAttention group layout mismatch")
        all_ids = []
        for size, prefix, suffix in zip(self.block_sizes, self.prefix_blocks, self.suffix_blocks):
            require(type(size) is int and size > 0 and self.page_tokens % size == 0,
                    "Page/kernel allocation geometry mismatch")
            require(len(prefix) == len(old_selected) * (self.page_tokens // size),
                    "compact prefix physical capacity mismatch")
            require(len(suffix) * size >= self.prompt_tokens - self.old_seal,
                    "insufficient private suffix capacity")
            require(all(type(b) is int and b > 0 for b in prefix + suffix),
                    "null/invalid native blocks")
            all_ids.extend(prefix + suffix)
        require(len(all_ids) == len(set(all_ids)), "physical block alias across groups")

    @property
    def identity(self):
        return digest(asdict(self))

    @property
    def logical_matched_tokens(self):
        return self.old_seal

    @property
    def full_attention_group_ids(self):
        return self.group_ids

    @property
    def dropped_full_attention_tokens(self):
        return self.old_seal - (len(self.retained) + len(self.restored)) * self.page_tokens

    @property
    def append(self):
        return tuple(range(self.old_seal // self.page_tokens, self.new_seal // self.page_tokens))

    def positions(self, logical_cursor, count):
        require(type(logical_cursor) is int and type(count) is int and count > 0
                and logical_cursor >= self.old_seal, "invalid forward cursor")
        physical = logical_cursor - self.dropped_full_attention_tokens
        require(all((physical + count + size - 1) // size <= len(pre) + len(suf)
                    for size, pre, suf in zip(self.block_sizes, self.prefix_blocks, self.suffix_blocks)),
                "decode growth requires newly owned native suffix slots")
        return tuple(range(logical_cursor, logical_cursor + count)), tuple(range(physical, physical + count))

    def page_blocks(self, page):
        require(page in self.append, "delta store must come from newly sealed suffix")
        start = (page * self.page_tokens - self.old_seal)
        return tuple(tuple(blocks[start // size:(start + self.page_tokens) // size])
                     for size, blocks in zip(self.block_sizes, self.suffix_blocks))

    def mixed_tables(self):
        return tuple(pre + suf for pre, suf in zip(self.prefix_blocks, self.suffix_blocks))

    def install(self, runner, request_id):
        """Use native BlockTable.add_row, which performs kernel-block splitting.

        Must run after _update_states and before _prepare_inputs. This method
        writes physical tables only; input_batch logical token counters and RoPE
        buffers are deliberately untouched.
        """
        require(not runner.uses_mrope and runner.uses_xdrope_dim == 0 and
                runner.num_spec_tokens == 0 and not runner.use_async_scheduling,
                "unsupported MRoPE/XDRoPE/speculative/async configuration")
        require(request_id in runner.input_batch.req_ids, "native request row absent")
        row = runner.input_batch.req_ids.index(request_id)
        logical_cursor = int(runner.input_batch.num_computed_tokens_cpu[row])
        self.positions(logical_cursor, 1)
        tables = runner.input_batch.block_table.block_tables
        for group, size, ids in zip(self.group_ids, self.block_sizes, self.mixed_tables()):
            require(group < len(tables) and tables[group].block_size * tables[group].blocks_per_kv_block == size
                    and len(ids) * tables[group].blocks_per_kv_block <= tables[group].max_num_blocks_per_req,
                    "loaded native block table layout/capacity mismatch")
        for group, ids in zip(self.group_ids, self.mixed_tables()):
            tables[group].add_row(list(ids), row)
        runner._selective_load_plans[request_id] = self
