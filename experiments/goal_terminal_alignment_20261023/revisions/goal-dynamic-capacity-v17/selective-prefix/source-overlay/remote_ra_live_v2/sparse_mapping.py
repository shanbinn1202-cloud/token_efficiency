"""Exact compact-table plan; logical positions remain native token positions."""
from collections.abc import Mapping
from dataclasses import dataclass
from .contract import require


def _int(value, low=0):
    return type(value) is int and value >= low


@dataclass(frozen=True)
class SparsePlan:
    """Duck-type of the connector SelectiveLoadPlan the runner consumes.

    FullAttention groups use physical = logical - dropped; other groups and
    every RoPE row stay logical (gpu_model_runner.py:1938-1959).
    """
    logical_matched_tokens: int
    dropped_full_attention_tokens: int
    full_attention_group_ids: tuple

    def __post_init__(self):
        groups = self.full_attention_group_ids
        # An empty group tuple would silently leave FA slots logical.
        require(_int(self.logical_matched_tokens) and _int(self.dropped_full_attention_tokens)
                and self.dropped_full_attention_tokens <= self.logical_matched_tokens
                and type(groups) is tuple and groups and all(_int(g) for g in groups)
                and len(set(groups)) == len(groups), "invalid sparse plan")


@dataclass(frozen=True)
class RowLayout:
    """Fixed head of one request's FA rows: [prefix Pages][run Pages].

    Logical x >= run_start sits at physical |prefix|*P + (x - run_start), row
    index |prefix| + (x - run_start)//P. Phase 2 uses run_start = L with prefix
    A_t and the handover run; phase 1 is the same shape with run_start = Q,
    prefix R and run [tail][fresh]. Local turn 0 is empty(P, groups).
    """
    page_tokens: int
    run_start: int                                   # L (phase 2) / Q (phase 1)
    prefix_pages: tuple[int, ...]                    # ascending, all < run_start/P
    prefix_blocks: Mapping[int, tuple[int, ...]]     # FA group -> ids aligned with prefix_pages
    run_blocks: Mapping[int, tuple[int, ...]]        # FA group -> ids of run Pages, logical order

    def __post_init__(self):
        P, start = self.page_tokens, self.run_start
        require(_int(P, 1) and _int(start) and start % P == 0, "row layout geometry")
        pages = self.prefix_pages
        require(type(pages) in (tuple, list) and all(_int(p) for p in pages) and
                list(pages) == sorted(set(pages)) and all(p < start // P for p in pages),
                "prefix Pages must be ascending, unique and before the run")
        require(isinstance(self.prefix_blocks, Mapping) and isinstance(self.run_blocks, Mapping)
                and self.prefix_blocks and set(self.prefix_blocks) == set(self.run_blocks)
                and all(_int(g) for g in self.prefix_blocks), "FullAttention group set mismatch")
        prefix = {g: tuple(v) for g, v in sorted(self.prefix_blocks.items())}
        run = {g: tuple(v) for g, v in sorted(self.run_blocks.items())}
        require(all(len(v) == len(pages) for v in prefix.values()) and
                len({len(v) for v in run.values()}) == 1, "prefix/run width differs across groups")
        ids = [b for v in (*prefix.values(), *run.values()) for b in v]
        # All KV groups share one BlockPool: an id may appear once per request.
        require(all(_int(b, 1) for b in ids) and len(set(ids)) == len(ids),
                "fixed sparse block null/aliased")
        object.__setattr__(self, "prefix_pages", tuple(pages))
        object.__setattr__(self, "prefix_blocks", prefix)
        object.__setattr__(self, "run_blocks", run)

    @classmethod
    def empty(cls, page_tokens, groups):
        """Local turn 0: no fixed blocks, physical == logical."""
        return cls(page_tokens, 0, (), {g: () for g in groups}, {g: () for g in groups})


def compose_rows(layout, raw):
    """Pure phase-2 (and phase-1) FA row: [prefix][run][native transients].

    raw: {FA group: (native ids, block size)}. The engine installs the fixed
    head at admission, so native allocation only grows ``rest`` at the end and
    the published prefix never moves (engine_boundary prefix immutability).
    """
    require(type(layout) is RowLayout and isinstance(raw, Mapping) and
            set(raw) == set(layout.prefix_blocks), "native FA groups differ from the layout")
    out = {}
    for g, (ids, size) in sorted(raw.items()):
        ids = tuple(ids)
        require(size == layout.page_tokens, "Stage B requires one FA block per Page")
        require(all(_int(b, 1) for b in ids) and len(set(ids)) == len(ids),
                "native FA row null/aliased")
        fixed = layout.prefix_blocks[g] + layout.run_blocks[g]
        pinned = set(fixed)
        require(pinned <= set(ids), "fixed sparse block missing from the native row")
        rest = tuple(b for b in ids if b not in pinned)       # native order = allocation order
        mixed = fixed + rest
        require(len(mixed) == len(ids) and set(mixed) == set(ids),
                "sparse row changed native allocation ownership")
        out[g] = (mixed, size)
    n = len(layout.prefix_pages)
    return out, SparsePlan(layout.run_start, layout.run_start - n * layout.page_tokens,
                           tuple(sorted(raw)))


def install_before_prepare(runner, scheduler_output):
    """Runner hook (gpu_model_runner.py:4415): the worker half lives in worker_rows."""
    from .worker_rows import install_before_prepare as install
    return install(runner, scheduler_output)
