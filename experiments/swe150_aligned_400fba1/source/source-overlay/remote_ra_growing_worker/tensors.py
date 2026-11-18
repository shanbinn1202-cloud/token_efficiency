"""Real torch copies of native shaped KV views, never guessed raw offsets.

FullAttention storage can be [2, blocks, ...] with noncontiguous block slices;
hybrid layout can change strides. select(axis, id).copy_ preserves both. Mamba
views are a separate tuple of convolution/recurrent states, always block axis 0
in the pinned GPUModelRunner._reshape_kv_cache_tensors implementation.
"""
from dataclasses import dataclass
import itertools
import secrets
from typing import NamedTuple
from .contract import require, digest

# payload_digest identity tokens: a per-process random prefix and a sequence number,
# 64 hex characters like the SHA-256 they replace.
_PAYLOAD_PREFIX = secrets.token_hex(24)
_PAYLOAD_SEQUENCE = itertools.count(1)

# A startup/layout proof must remain bounded even for hostile strided views.
# Dense tensors collapse to one span; packed layouts use one span per packing
# gap. Unsupported larger proofs fail closed before any native tensor operation.
MAX_STORAGE_SPANS = 262144


def _storage_metadata(tensor):
    """Byte-address metadata only; no tensor reads, writes, copies or CUDA work."""
    import torch
    require(tensor.layout == torch.strided and tensor.device.type in {"cpu", "cuda"},
            "native storage non-overlap requires a concrete strided tensor")
    storage = tensor.untyped_storage()
    shape, strides = tuple(tensor.shape), tuple(tensor.stride())
    itemsize, offset = tensor.element_size(), tensor.storage_offset()
    require(itemsize > 0 and len(shape) == len(strides) and shape and
            all(type(n) is int and n > 0 for n in shape), "empty/unknown native storage layout")
    base, capacity = storage.data_ptr(), storage.nbytes()
    require(base > 0 and tensor.data_ptr() == base + offset * itemsize,
            "native tensor/storage address mismatch")
    return (str(tensor.device), base, capacity, offset, itemsize, str(tensor.dtype), shape, strides)


def _storage_spans(metadata):
    """Prove internal element (therefore block) disjointness and exact byte spans.

    Sort nontrivial dimensions by absolute byte stride. Each next stride must
    cover the entire previous address extent. This is a sufficient proof; more
    exotic overlapping-extent layouts are deliberately refused. Negative strides
    normalize to their minimum address if a torch implementation supports them.
    Current torch rejects creation of negative-stride tensors itself.
    """
    device, base, capacity, offset, itemsize, dtype, shape, strides = metadata
    dimensions = sorted((abs(step) * itemsize, n) for n, step in zip(shape, strides) if n > 1)
    extent = itemsize
    for step, n in dimensions:
        require(step >= extent, "native storage overlap or non-overlap proof unavailable")
        extent += (n - 1) * step
    first = offset * itemsize + sum(min(0, (n - 1) * step * itemsize)
                                   for n, step in zip(shape, strides))
    require(0 <= first and first + extent <= capacity, "native view exceeds underlying storage")
    width, sparse = itemsize, []
    for step, n in dimensions:
        if not sparse and step == width:
            width *= n
        else:
            sparse.append((step, n))
    span_count = 1
    for _, n in sparse:
        span_count *= n
        require(span_count <= MAX_STORAGE_SPANS, "native storage non-overlap proof span limit")
    spans = [(base + first, base + first + width)]
    for step, n in sparse:
        spans = [(start + i * step, end + i * step) for i in range(n) for start, end in spans]
    return device, spans


@dataclass(frozen=True)
class LayerView:
    name: str
    group: int
    kind: str
    tensors: tuple
    axes: tuple[int, ...]
    ratio: int = 1
    allocation_tokens: int = 0
    kernel_tokens: int = 0
    page_bytes: int = 0
    page_stride: int = 0
    page_offset: int = 0
    cohort: tuple = ()

    def __post_init__(self):
        require(type(self.name) is str and self.name and type(self.group) is int and self.group >= 0
                and self.kind in {"attention", "hybrid"} and self.tensors
                and len(self.tensors) == len(self.axes), "invalid physical layer descriptor")
        for tensor, axis in zip(self.tensors, self.axes):
            require(type(axis) is int and 0 <= axis < tensor.ndim and tensor.shape[axis] > 1,
                    "invalid native block dimension")
            require(not tensor.requires_grad, "native KV must not require gradients")
        require(self.kind != "hybrid" or (len(self.tensors) >= 2 and set(self.axes) == {0}),
                "hybrid checkpoint must include convolution and recurrent states")
        require(type(self.ratio) is int and self.ratio>0 and
                all(t.shape[a]%self.ratio==0 for t,a in zip(self.tensors,self.axes)),
                'invalid allocation/kernel tensor geometry')

    def signature(self):
        return (self.name, self.group, self.kind, self.ratio, self.allocation_tokens,
                self.kernel_tokens,self.page_bytes,self.page_stride,self.page_offset,self.cohort,
                tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype),
                       t.storage_offset(), t.element_size(), a)
                      for t, a in zip(self.tensors, self.axes)))


def views_from_runner(runner, caches, kernel_sizes, skip_sliding=False):
    """Called after the pinned hybrid reshape, with actual backend shapes.

    Infer block dimension from two backend shape calls, not from ambiguous actual
    sizes (e.g. 2 blocks and a K/V dimension both equal 2). Split allocations
    cover all B/K kernel slices; compressed storage remains unsupported.

    ``skip_sliding`` (the window layout, gemma-s1): sliding-window groups get no view;
    their KV is never captured, restored or zeroed by the runtime, and their tensors
    are pool 1's own (no storage shared with the full-attention views).
    """
    result = []
    configs={}
    for item in runner.kv_cache_config.kv_cache_tensors:
        for name in item.shared_by:
            require(name not in configs,'ambiguous native tensor cohort')
            configs[name]=item
    for group in runner._kv_cache_spec_attn_group_iterator():
        gid, spec = group.kv_cache_group_id, group.kv_cache_spec
        if gid == len(kernel_sizes):
            continue
        if skip_sliding and type(spec).__name__ == "SlidingWindowSpec":
            continue
        for name in group.layer_names:
            if name in runner.runner_only_attn_layers:
                continue
            cache = caches[name]
            require(name in configs,'native tensor sharing declaration absent')
            config=configs[name]
            stride=int(config.block_stride or spec.page_size_bytes)
            offset=int(config.offset if config.block_stride else 0)
            cohort=tuple(sorted(config.shared_by))
            common=dict(allocation_tokens=spec.block_size,page_bytes=spec.page_size_bytes,
                        page_stride=stride,page_offset=offset,cohort=cohort)
            if type(spec).__name__ == "MambaSpec":
                require(type(cache) in (tuple, list), "Mamba state tuple missing")
                require(len(cache)==len(spec.shapes)==len(spec.dtypes),'Mamba state count mismatch')
                require(all(tuple(t.shape[1:])==tuple(s) and t.dtype==d
                            for t,s,d in zip(cache,spec.shapes,spec.dtypes)),
                        'Mamba native conv/recurrent shape or dtype mismatch')
                result.append(LayerView(name, gid, "hybrid", tuple(cache), (0,) * len(cache),
                                        kernel_tokens=spec.block_size,**common))
            else:
                require(type(spec).__name__ == "FullAttentionSpec" and
                        spec.block_size == spec.storage_block_size and
                        spec.block_size%kernel_sizes[gid]==0,
                        "unsupported attention spec/compressed/nonintegral storage")
                ratio=spec.block_size//kernel_sizes[gid]
                shapes = [group.backend.get_kv_cache_shape(n, kernel_sizes[gid], spec.num_kv_heads,
                          spec.head_size, cache_dtype_str=runner.cache_config.cache_dtype) for n in (17, 19)]
                axes = [i for i, (a, b) in enumerate(zip(*shapes)) if a != b]
                require(len(shapes[0]) == len(shapes[1]) == cache.ndim and len(axes) == 1
                        and shapes[0][axes[0]] == 17 and shapes[1][axes[0]] == 19,
                        "backend block axis is not uniquely identifiable")
                expected=group.backend.get_kv_cache_shape(cache.shape[axes[0]],kernel_sizes[gid],
                    spec.num_kv_heads,spec.head_size,cache_dtype_str=runner.cache_config.cache_dtype)
                require(tuple(cache.shape)==tuple(expected),'actual backend cache shape mismatch')
                result.append(LayerView(name, gid, "attention", (cache,), (axes[0],),ratio=ratio,
                                        kernel_tokens=kernel_sizes[gid],**common))
    require(result and len({v.name for v in result}) == len(result), "empty/duplicate layer inventory")
    return tuple(result)


# The fourth counter (device-to-device bytes) moves only with --gdn-device-handoff;
# request_counters reports it only then, so frames keep their three keys when off.
_COUNTERS = ("_http_h2d_issued", "_http_h2d_joined", "_http_zero_joins", "_http_d2d_bytes")


class attributed:
    """Per-request IO: H2D bytes and zero joins inside the scope belong to ``request``.

    Several requests share one step, so the process counters alone cannot say
    whose bytes they are. The scope reads only those counters (any IO that keeps
    them is attributed); a nested scope's IO belongs to the inner request only.
    Integer bookkeeping on the owner thread: no copy, sync or hash.
    """
    def __init__(self, io, request):
        require(type(request) is str and request, "IO attribution needs a request id")
        self.io, self.request = io, request

    def __enter__(self):
        stack = self.io.__dict__.setdefault("_http_scopes", [])
        self.start = tuple(getattr(self.io, name, 0) for name in _COUNTERS)
        self.inner = [0] * len(_COUNTERS)
        stack.append(self)
        return self

    def __exit__(self, *exc):
        stack = self.io._http_scopes
        require(stack and stack[-1] is self, "IO attribution scopes interleaved")
        stack.pop()
        delta = [getattr(self.io, name, 0) - v for name, v in zip(_COUNTERS, self.start)]
        own = self.io.__dict__.setdefault("_http_requests", {}).setdefault(
            self.request, [0] * len(_COUNTERS))
        for i, (d, inner) in enumerate(zip(delta, self.inner)):
            own[i] += d - inner
        if stack:
            stack[-1].inner = [a + d for a, d in zip(stack[-1].inner, delta)]
        return False


def request_counters(io, request):
    """IO counters attributed to one request (zeros before its first scope).

    With --gdn-device-handoff (``io.handoff``) also its device-to-device bytes."""
    values = getattr(io, "_http_requests", {}).get(request, (0,) * len(_COUNTERS))
    out = dict(h2d_issued=values[0], h2d_joined=values[1], zero_joins=values[2])
    if getattr(io, "handoff", False) is True:
        out["d2d_bytes"] = values[3]
    return out


class _Pending(NamedTuple):
    """One issued batch of copies (--gdn-device-handoff): its event on the current
    stream, what joins with it, the request whose IO scope issued it and the
    host objects (pinned sources, arena Pages) kept alive until it is joined."""
    event: object
    h2d: int
    zeros: int
    d2h: int
    request: object
    holds: tuple


class _DoneEvent:
    """CPU tensors (CpuTensorTestIO): every copy has completed when it returns."""
    def query(self):
        return True

    def synchronize(self):
        return None


# ---------------------------------------------------------------------------
# Pinned Page arena (decoupled layout, H7).  Legacy never builds one.
# ---------------------------------------------------------------------------

def _token():
    return _PAYLOAD_PREFIX + "%016x" % next(_PAYLOAD_SEQUENCE)


def identity_token():
    """A fresh 64-hex identity in payload_digest's sequence (no bytes are read)."""
    return _token()


class SlotTable:
    """Slot bookkeeping of the Page arena; pure Python, no torch.

    A slot holds one FA Page (every FA layer).  Slots live in fixed-size
    chunks that are allocated on demand up to ``max_slots``; the lowest free
    slots are handed out first, so one batch usually gets consecutive slots.
    Each allocation gets a fresh identity token (the payload digest): a slot
    released and reused can never be mistaken for the Page it held before.
    """

    def __init__(self, chunk_slots, max_slots, grow):
        require(type(chunk_slots) is int and chunk_slots > 0 and type(max_slots) is int and
                max_slots > 0 and callable(grow), "invalid Page arena geometry")
        self.chunk_slots, self.max_slots, self._grow = chunk_slots, max_slots, grow
        self.capacity, self.free, self.owner = 0, [], {}

    def allocate(self, n):
        import heapq
        require(type(n) is int and n > 0, "positive Page count required")
        while len(self.free) < n:
            require(self.capacity < self.max_slots,
                    "DRAM_ARENA_EXHAUSTED: %d Pages held, %d more requested, limit %d"
                    % (len(self.owner), n, self.max_slots))
            self._grow()
            self.free.extend(range(self.capacity, self.capacity + self.chunk_slots))
            heapq.heapify(self.free)
            self.capacity += self.chunk_slots
        slots = [heapq.heappop(self.free) for _ in range(n)]
        tokens = [_token() for _ in slots]
        self.owner.update(zip(slots, tokens))
        return list(zip(slots, tokens))

    def release(self, slot, token):
        import heapq
        if self.owner.get(slot) == token:
            del self.owner[slot]
            heapq.heappush(self.free, slot)

    def holds(self, slot, token):
        return self.owner.get(slot) == token

    def segments(self, slots):
        """Ascending ``slots`` -> [(chunk, batch offset, count, chunk indices)].

        One segment per chunk; ``indices`` is a range when the slots are
        consecutive (one copy), else a list (host gather/scatter).
        """
        out = []
        for offset, slot in enumerate(slots):
            chunk, index = divmod(slot, self.chunk_slots)
            if out and out[-1][0] == chunk:
                out[-1][3].append(index)
            else:
                out.append([chunk, offset, 0, [index]])
        result = []
        for chunk, offset, _, indices in out:
            first = indices[0]
            run = indices == list(range(first, first + len(indices)))
            result.append((chunk, offset, len(indices),
                           range(first, first + len(indices)) if run else indices))
        return result


class PagePayload:
    """One FA Page held in an arena slot; ``token`` is its payload digest.

    The slot returns to the arena when the payload is no longer referenced
    (e.g. its session's catalog is dropped), like a pinned tensor today.
    """
    __slots__ = ("arena", "slot", "token", "__weakref__")

    def __init__(self, arena, slot, token):
        import weakref
        self.arena, self.slot, self.token = arena, slot, token
        weakref.finalize(self, arena.slots.release, slot, token)


class PageArena:
    """Preallocated pinned host chunks holding FA Pages for every FA layer view.

    A chunk holds ``chunk_slots`` Pages; per FA tensor its host buffer is
    [chunk_slots * ratio, *block shape without the block axis], so a run of
    consecutive slots is one contiguous host range per layer.
    """
    CHUNK_BYTES = 256 << 20

    def __init__(self, views, *, max_pages, pin=True, chunk_bytes=None):
        import torch
        self.torch, self.pin, self.layout, page_bytes = torch, pin is True, [], 0
        for v in views:
            if v.kind != "attention":
                continue
            for ti, (t, a) in enumerate(zip(v.tensors, v.axes)):
                rest = tuple(n for i, n in enumerate(t.shape) if i != a)
                self.layout.append((v.name, ti, v.ratio, rest, t.dtype))
                page_bytes += v.ratio * (t.numel() // t.shape[a]) * t.element_size()
        require(self.layout and page_bytes > 0, "Page arena needs FullAttention views")
        chunk_slots = max(1, (chunk_bytes or self.CHUNK_BYTES) // page_bytes)
        require(type(max_pages) is int and max_pages > 0, "Page arena capacity required")
        self.page_bytes, self.chunks = page_bytes, []
        self.slots = SlotTable(chunk_slots, -(-max_pages // chunk_slots) * chunk_slots, self._grow)

    def _grow(self):
        n = self.slots.chunk_slots
        self.chunks.append({(name, ti): self.torch.empty((n * r,) + rest, dtype=dtype, device="cpu",
                                                         pin_memory=self.pin)
                            for name, ti, r, rest, dtype in self.layout})

    def allocate(self, n):
        return [PagePayload(self, slot, token) for slot, token in self.slots.allocate(n)]

    def host(self, chunk, name, ti):
        return self.chunks[chunk][(name, ti)]

    def stats(self):
        return dict(page_bytes=self.page_bytes, chunk_slots=self.slots.chunk_slots,
                    max_pages=self.slots.max_slots, chunks=len(self.chunks),
                    pinned_bytes=len(self.chunks) * self.slots.chunk_slots * self.page_bytes,
                    pages_held=len(self.slots.owner), pinned=self.pin)


def _expand(indices, ratio):
    """Block/slot indices -> their ratio kernel rows."""
    return [i * ratio + k for i in indices for k in range(ratio)]


def _pinned(host):
    """A page-locked host buffer (TensorIO._empty_host on CUDA): a D2H copy into it may be
    queued (non_blocking) and joined by a later device synchronize (perf 2.2)."""
    is_pinned = getattr(host, "is_pinned", None)
    return is_pinned is not None and is_pinned() is True


class TensorIO:
    """Synchronous, device-drained baseline for physical storage correctness.

    No background copy stream means a receipt is issued only after a device join.
    This costs synchronization and makes no throughput claim. A failed join keeps
    leases quarantined; no caller may infer drainage from an exception.
    """
    evidence_kind = "CUDA_PHYSICAL_TENSOR_OPERATIONS"
    # Decoupled KV layout (kv_layout.KvLayout.decoupled): FA and Mamba blocks
    # come from two pools whose ids may coincide, so FA and GDN storage must
    # be disjoint. The pinned Page arena (attach_arena) serves batched FA
    # capture/restore; legacy keeps the per-block path and has no arena.
    decoupled = False
    arena = None
    # --gdn-device-handoff (gdn_handoff.attach_worker -> enable_handoff): the issue
    # variants below record an event instead of draining; off, nothing issues.
    handoff = False
    _pending = ()

    def __init__(self, views, *, decoupled=False):
        import torch
        self.torch = torch
        self.views = tuple(views)
        require(self.views and all(type(v) is LayerView for v in self.views), "physical views required")
        devices = {str(t.device) for v in self.views for t in v.tensors}
        require(len(devices) == 1 and next(iter(devices)).startswith("cuda"),
                "CUDA_REQUIRED: CPU tensor tests cannot create physical worker")
        self.device = self.views[0].tensors[0].device
        self.decoupled = decoupled is True
        self._validate_aliases()

    def _validate_aliases(self):
        # Prove whole writable views disjoint, not just descriptor inequality.
        # This covers all block IDs including currently unclaimed ones, as the
        # same physical views persist through future allocations and recycling.
        metadata = [(id(t), _storage_metadata(t)) for v in self.views for t in v.tensors]
        if all(v.page_stride for v in self.views):
            # Native hybrid groups intentionally alias complete tensors. Prove
            # allocation page boundaries, then allow ONLY declared cross-group
            # cohorts. Live block IDs are globally exclusive at the owner.
            entries=[]
            for v in self.views:
                require(v.page_bytes>0 and 0<=v.page_offset and
                        v.page_offset+v.page_bytes<=v.page_stride,
                        'native page extent exceeds allocation stride')
                for t,a in zip(v.tensors,v.axes):
                    base=t.untyped_storage().data_ptr();capacity=t.untyped_storage().nbytes()
                    count=t.shape[a]//v.ratio
                    require(count>1 and count*v.page_stride<=capacity,
                            'allocation capacity exceeds shared backing')
                    require(t.stride()[a]*v.ratio*t.element_size()==v.page_stride,
                            'native kernel-to-allocation stride differs from declared page stride')
                    device,spans=_storage_spans(_storage_metadata(self._slice(v,t,a,0)))
                    require(all(base+v.page_offset<=lo<hi<=base+v.page_offset+v.page_bytes
                                for lo,hi in spans),'allocation slice escapes declared native page')
                    for other,other_base,other_device,other_spans in entries:
                        if device!=other_device or base!=other_base:continue
                        overlap=any(lo<end and start<hi for lo,hi in spans for start,end in other_spans)
                        if overlap:
                            # Two pools: an FA id and a Mamba id may be equal,
                            # so FA and GDN storage must never share bytes.
                            require(not self.decoupled or v.kind==other.kind,
                                    'decoupled FA/GDN native storage alias')
                            require(v.group!=other.group and v.cohort==other.cohort and
                                    v.name in v.cohort and other.name in v.cohort,
                                    'undeclared or same-group native storage alias')
                    entries.append((v,base,device,spans))
        else:
            require(not any(v.page_stride for v in self.views),'mixed proved/unproved native layout')
            occupied=[]
            for _,entry in metadata:
                device,spans=_storage_spans(entry)
                require(len(occupied)+len(spans)<=MAX_STORAGE_SPANS,'native storage proof span limit')
                occupied.extend((device,start,end) for start,end in spans)
            occupied.sort()
            for left,right in zip(occupied,occupied[1:]):
                require(left[0]!=right[0] or left[2]<=right[1],
                        'overlapping native storage across layer/state/group views')
        self._storage_inventory = tuple(metadata)
        self._view_inventory = tuple(v.signature() for v in self.views)
        self.layout = digest([v.signature() for v in self.views])

    def _assert_storage_unchanged(self):
        # Tensor metadata is mutable even when LayerView is frozen. Recheck at
        # each IO entry before drain/zero/copy, so set_/as_strided_ cannot turn a
        # previously proved layout into a write alias after construction.
        require(tuple(v.signature() for v in self.views) == self._view_inventory and
                tuple((id(t), _storage_metadata(t)) for v in self.views for t in v.tensors)
                == self._storage_inventory, "native storage layout changed after non-overlap proof")

    def drain(self):
        check=getattr(self,'connector_check',None)
        if check is not None:check()
        self.torch.cuda.synchronize(self.device)
        if check is not None:check()
        if self._pending:
            self.poll_joins()   # --gdn-device-handoff: the device join completed every issued copy

    def validate(self, groups, kind):
        # One call names one kind, i.e. one pool (legacy: the shared pool;
        # decoupled: FA pool 0 or Mamba pool 1), so the id uniqueness below
        # is per pool in both layouts.
        self._assert_storage_unchanged()
        relevant = tuple(v for v in self.views if v.kind == kind)
        require(relevant and set(groups) == {v.group for v in relevant}, "incomplete physical group coverage")
        for v in relevant:
            ids = groups[v.group]
            require(type(ids) is tuple and ids and len(set(ids)) == len(ids)
                    and all(type(b) is int and b > 0 for b in ids), "invalid physical block IDs")
            for t, a in zip(v.tensors, v.axes):
                require(max(ids) < t.shape[a]//v.ratio, "allocation block ID out of range")
        all_ids=[bid for values in groups.values() for bid in values]
        require(len(set(all_ids))==len(all_ids),'shared pool allocation used by multiple groups')
        return relevant

    @staticmethod
    def _slice(view,tensor,axis,bid):
        if view.ratio==1:return tensor.select(axis,bid)
        return tensor.narrow(axis,bid*view.ratio,view.ratio)

    def _empty_host(self, sample):
        return self.torch.empty(tuple(sample.shape), dtype=sample.dtype,
                                device="cpu", pin_memory=True)

    def byte_count(self, groups, kind):
        relevant = self.validate(groups, kind)
        return sum(len(groups[v.group]) * v.ratio * (t.numel() // t.shape[a]) * t.element_size()
                   for v in relevant for t, a in zip(v.tensors, v.axes))

    def capture(self, groups, kind):
        relevant = self.validate(groups, kind)
        self.drain()  # includes the native forward/postprocess which produced KV
        result = {}
        copied = 0
        for v in relevant:
            per_block = []
            for bid in groups[v.group]:
                pieces = []
                for tensor, axis in zip(v.tensors, v.axes):
                    sample = self._slice(v,tensor,axis,bid)
                    host = self._empty_host(sample)
                    # host-trims-1111 (perf 2.2): into a pinned buffer the copy is queued on the
                    # stream (as issue_capture_into does) and the drain below joins every one
                    # before the payload is returned; one host wait per capture, not one per
                    # tensor per block. A pageable buffer copies synchronously, as before.
                    host.copy_(sample, non_blocking=_pinned(host))
                    pieces.append(host)
                    copied += host.numel() * host.element_size()
                per_block.append(tuple(pieces))
            result[v.name] = tuple(per_block)
        self.drain()
        # Joined D2H bytes (per-step latest/carry cost; smoke items 3 and 12).
        self._http_d2h_joined = getattr(self, "_http_d2h_joined", 0) + copied
        return result

    def copy(self, source, target, kind):
        """Device-to-device block copy, e.g. a GPU-resident GDN state; no host bytes."""
        relevant = self.validate(source, kind)
        require(set(target) == set(source) and self.validate(target, kind) == relevant and
                all(len(target[g]) == len(ids) for g, ids in source.items()),
                "copy source/target group coverage differs")
        # Overlap would read partially overwritten blocks: pool IDs are shared.
        require(not {b for ids in source.values() for b in ids} &
                {b for ids in target.values() for b in ids}, "copy source/target alias")
        self.drain()
        for v in relevant:
            for src, dst in zip(source[v.group], target[v.group]):
                for tensor, axis in zip(v.tensors, v.axes):
                    self._slice(v,tensor,axis,dst).copy_(self._slice(v,tensor,axis,src),
                                                         non_blocking=False)
        self.drain()

    def zero(self, groups, kind):
        relevant = self.validate(groups, kind)
        self.drain()
        if self.arena is not None and kind == "attention":
            # Decoupled: one index_fill_ per FA tensor instead of one zero_
            # per block per layer (P = 16 rows hold ~49x more blocks).
            index = self._kernel_index(groups, relevant)
            for v in relevant:
                for tensor, axis in zip(v.tensors, v.axes):
                    tensor.index_fill_(axis, index[(v.group, v.ratio)], 0)
        else:
            for v in relevant:
                for bid in groups[v.group]:
                    for tensor, axis in zip(v.tensors, v.axes):
                        self._slice(v,tensor,axis,bid).zero_()
        self.drain()
        self._http_zero_joins = getattr(self, "_http_zero_joins", 0) + 1

    # ---- Pinned Page arena: batched FA capture/restore (decoupled) --------

    def attach_arena(self, arena):
        require(self.decoupled and self.arena is None and type(arena) is PageArena and
                [(n, ti, r, rest, dt) for n, ti, r, rest, dt in arena.layout] ==
                [(v.name, ti, v.ratio, tuple(s for i, s in enumerate(t.shape) if i != a), t.dtype)
                 for v in self.views if v.kind == "attention"
                 for ti, (t, a) in enumerate(zip(v.tensors, v.axes))],
                "Page arena must match the decoupled FullAttention views")
        self.arena = arena

    def _kernel_index(self, groups, relevant):
        """{(group, ratio): device LongTensor of the kernel rows of groups[group]}."""
        index = {}
        for v in relevant:
            key = (v.group, v.ratio)
            if key not in index:
                index[key] = self.torch.tensor(_expand(groups[v.group], v.ratio),
                                               dtype=self.torch.long, device=self.device)
        return index

    def _page_groups(self, pages):
        """[{FA group: (block,)}] in slot order -> {FA group: blocks} (one validate)."""
        fa = {v.group for v in self.views if v.kind == "attention"}
        require(all(type(p) is dict and set(p) == fa and
                    all(type(p[g]) is tuple and len(p[g]) == 1 for g in fa) for p in pages),
                "one block per FullAttention group per Page required")
        groups = {g: tuple(p[g][0] for p in pages) for g in sorted(fa)}
        return groups, self.validate(groups, "attention")

    # Device staging per FA tensor per copy batch; bigger batches go in slices.
    MAX_STAGING_BYTES = 64 << 20

    def _slices(self, relevant, n):
        layer = max(v.ratio * (t.numel() // t.shape[a]) * t.element_size()
                    for v in relevant for t, a in zip(v.tensors, v.axes))
        step = max(1, self.MAX_STAGING_BYTES // layer)
        return [(i, min(n, i + step)) for i in range(0, n, step)]

    def capture_pages(self, pages):
        """FA Pages -> PagePayloads: per FA tensor one gather and one D2H, one join.

        Bytes and device join match ``capture`` per Page; only the number of
        pinned allocations, copies and synchronisations changes.
        """
        require(self.arena is not None, "Page arena not attached")
        pages = list(pages)
        if not pages:
            return []
        groups, relevant = self._page_groups(pages)
        self.drain()  # includes the native forward/postprocess which produced KV
        payloads = self.arena.allocate(len(pages))      # ascending slots, batch order
        copied, scatter = 0, []
        for lo, hi in self._slices(relevant, len(pages)):
            index = self._kernel_index({g: ids[lo:hi] for g, ids in groups.items()}, relevant)
            segments = self.arena.slots.segments([p.slot for p in payloads[lo:hi]])
            for v in relevant:
                r = v.ratio
                for ti, (tensor, axis) in enumerate(zip(v.tensors, v.axes)):
                    staging = tensor.index_select(axis, index[(v.group, r)])
                    if axis:
                        staging = staging.movedim(axis, 0)
                    staging = staging.contiguous()
                    for chunk, offset, count, indices in segments:
                        host = self.arena.host(chunk, v.name, ti)
                        source = staging.narrow(0, offset * r, count * r)
                        if type(indices) is range:
                            host.narrow(0, indices.start * r, count * r).copy_(source, non_blocking=True)
                        else:
                            bounce = self.torch.empty(tuple(source.shape), dtype=source.dtype,
                                                      device="cpu", pin_memory=self.arena.pin)
                            bounce.copy_(source, non_blocking=True)
                            scatter.append((host, _expand(indices, r), bounce))
                    copied += staging.numel() * staging.element_size()
        self.drain()
        for host, rows, bounce in scatter:
            host.index_copy_(0, self.torch.tensor(rows, dtype=self.torch.long), bounce)
        # Joined D2H bytes (per-step latest/carry cost; smoke items 3 and 12).
        self._http_d2h_joined = getattr(self, "_http_d2h_joined", 0) + copied
        return payloads

    def restore_pages(self, items):
        """[(targets {FA group: (block,)}, payload)] -> device, one H2D + index_copy_ per FA tensor.

        Arena payloads restore together (in slot order); any other payload (a
        single-Page ``capture``, e.g. the phase-1 tail) restores through
        ``restore``. The arena targets and payloads are validated before the
        first copy.
        """
        items = list(items)
        others = [(g, p) for g, p in items if type(p) is not PagePayload]
        batch = sorted(((g, p) for g, p in items if type(p) is PagePayload), key=lambda x: x[1].slot)
        require(not batch or self.arena is not None, "Page arena not attached")
        if batch:
            groups, relevant = self._page_groups([g for g, _ in batch])
            require(len({p.slot for _, p in batch}) == len(batch) and
                    all(p.arena is self.arena and self.arena.slots.holds(p.slot, p.token)
                        for _, p in batch), "foreign or released Page arena payload")
        for g, p in others:
            self.restore(g, "attention", p)
        if not batch:
            return
        issued = 0
        for lo, hi in self._slices(relevant, len(batch)):
            index = self._kernel_index({g: ids[lo:hi] for g, ids in groups.items()}, relevant)
            segments = self.arena.slots.segments([p.slot for _, p in batch[lo:hi]])
            for v in relevant:
                r = v.ratio
                for ti, (tensor, axis) in enumerate(zip(v.tensors, v.axes)):
                    rest = tuple(s for i, s in enumerate(tensor.shape) if i != axis)
                    staging = self.torch.empty(((hi - lo) * r,) + rest, dtype=tensor.dtype,
                                               device=self.device)
                    for chunk, offset, count, indices in segments:
                        host = self.arena.host(chunk, v.name, ti)
                        if type(indices) is range:
                            source = host.narrow(0, indices.start * r, count * r)
                        else:
                            source = self.torch.empty((count * r,) + rest, dtype=tensor.dtype,
                                                      device="cpu", pin_memory=self.arena.pin)
                            self.torch.index_select(host, 0, self.torch.tensor(
                                _expand(indices, r), dtype=self.torch.long), out=source)
                        staging.narrow(0, offset * r, count * r).copy_(source, non_blocking=True)
                    tensor.index_copy_(axis, index[(v.group, r)],
                                       staging.movedim(0, axis) if axis else staging)
                    issued += staging.numel() * staging.element_size()
        self._http_h2d_issued = getattr(self, "_http_h2d_issued", 0) + issued
        self.drain()
        self._http_h2d_joined = getattr(self, "_http_h2d_issued", 0)

    # ---- Token gather (token-gather-1114, remote_ra_live_v2.token_gather) ----

    def _host_pin(self):
        return self.arena.pin if self.arena is not None else self.device.type == "cuda"

    # Token gather: {(FA view name, tensor index): (block axis, token axis)}, set once at worker startup from the
    # loaded backend's layout (token_gather.check_views -> attach_token_gather); None: gather_tokens refuses.
    gather_axes = None

    def attach_token_gather(self, axes):
        """Worker startup, once: the block and token axis of every FA tensor (token_gather.check_views). The block
        axis must be the view's own; the token axis another axis of that tensor."""
        fa = {(v.name, ti): (t, a) for v in self.views if v.kind == "attention"
              for ti, (t, a) in enumerate(zip(v.tensors, v.axes))}
        require(self.gather_axes is None and type(axes) is dict and fa and set(axes) == set(fa) and
                all(type(value) is tuple and len(value) == 2 and all(type(x) is int for x in value) and
                    value[0] == fa[key][1] and 0 <= value[1] < fa[key][0].ndim and value[1] != value[0]
                    for key, value in axes.items()),
                "token gather needs the block and token axis of every FullAttention tensor")
        self.gather_axes = dict(axes)

    def gather_tokens(self, blocks, sources):
        """Fill gathered FA blocks with single tokens' K/V rows, slot by slot in order.

        ``blocks``: [{FA group: (block,)}], the fresh gathered blocks; ``sources``: one per slot of those
        blocks in order (len(blocks) x P): ('device', {FA group: (block,)}, offset) for a token of a block on
        the GPU (a resident Page or the phase-1 tail), or ('host', payload, offset) for a token of a DRAM Page
        (an arena PagePayload, or a ``capture`` payload). Every FA tensor is indexed by its block and token axis
        (``attach_token_gather``; vLLM 0.24.0 FlashAttention: [blocks, 2, P, heads, head size], block axis 0,
        token axis 2, NHD or HND strides) through views, so its own strides hold; a token's row is the rest
        (K/V, heads, head size) in the tensor's axis order, as in a Page payload (the tensor without its block
        axis). Per FA tensor: the device rows in one index, the host rows gathered on the host into one buffer
        and copied in one H2D, then one index_put_ writes every slot. The targets, sources and payloads are
        validated before the first copy; one device join at the end. The H2D bytes count as restore bytes;
        device-to-device rows are not counted."""
        blocks, sources = list(blocks), list(sources)
        require(blocks, "token gather needs at least one gathered block")
        require(self.gather_axes is not None,
                "token gather needs the FullAttention block and token axes (attach_token_gather at worker startup)")
        groups, relevant = self._page_groups(blocks)
        tokens = {t.shape[self.gather_axes[(v.name, ti)][1]] for v in relevant for ti, t in enumerate(v.tensors)}
        require(all(v.ratio == 1 for v in relevant) and len(tokens) == 1,
                "token gather needs one kernel block per Page and one token count in every FullAttention view")
        P = tokens.pop()
        require(len(sources) == len(blocks) * P and
                all(type(s) is tuple and len(s) == 3 and s[0] in ("device", "host") and
                    type(s[2]) is int and 0 <= s[2] < P for s in sources),
                "token gather needs one (device|host, source, offset) per gathered slot")
        device = [k for k, s in enumerate(sources) if s[0] == "device"]
        host = [k for k, s in enumerate(sources) if s[0] == "host"]
        targets = {b for ids in groups.values() for b in ids}
        if device:
            seen = []
            for k in device:
                if sources[k][1] not in seen:
                    seen.append(sources[k][1])
            dev_groups, _ = self._page_groups(seen)
            require(not {b for ids in dev_groups.values() for b in ids} & targets,
                    "token gather source block aliases a gathered block")
        for k in host:
            payload = sources[k][1]
            if type(payload) is PagePayload:
                require(self.arena is not None and payload.arena is self.arena and
                        self.arena.slots.holds(payload.slot, payload.token), "foreign or released Page arena payload")
            else:
                require(type(payload) is dict and set(payload) == {v.name for v in relevant} and
                        all(len(payload[v.name]) == 1 and len(payload[v.name][0]) == len(v.tensors)
                            for v in relevant), "token gather host payload must be one captured Page")
        torch = self.torch
        long = dict(dtype=torch.long, device=self.device)
        issued = 0
        for v in relevant:
            g = v.group
            dst_b = torch.tensor([blocks[k // P][g][0] for k in range(len(sources))], **long)
            dst_o = torch.tensor([k % P for k in range(len(sources))], **long)
            if device:
                dev_index = torch.tensor(device, **long)
                src_b = torch.tensor([sources[k][1][g][0] for k in device], **long)
                src_o = torch.tensor([sources[k][2] for k in device], **long)
            if host:
                host_index = torch.tensor(host, **long)
            for ti, tensor in enumerate(v.tensors):
                axis, token_axis = self.gather_axes[(v.name, ti)]
                # The cache as [blocks, P, rest...] (a view: same storage, its own strides); a Page payload (a
                # capture piece, an arena slot) is the tensor without its block axis, so its token axis is page_axis.
                cache = tensor.movedim((axis, token_axis), (0, 1))
                page_axis = token_axis - (token_axis > axis)
                rest = tuple(cache.shape[2:])
                out = torch.empty((len(sources),) + rest, dtype=tensor.dtype, device=self.device)
                if device:
                    out.index_copy_(0, dev_index, cache[src_b, src_o])
                if host:
                    bounce = torch.empty((len(host),) + rest, dtype=tensor.dtype, device="cpu",
                                         pin_memory=self._host_pin())
                    arena_rows = {}
                    for i, k in enumerate(host):
                        payload, offset = sources[k][1], sources[k][2]
                        if type(payload) is PagePayload:
                            chunk, slot = divmod(payload.slot, self.arena.slots.chunk_slots)
                            arena_rows.setdefault(chunk, []).append((i, slot, offset))
                        else:
                            bounce[i].copy_(payload[v.name][0][ti].select(page_axis, offset))
                    for chunk, rows in arena_rows.items():
                        # arena rows: [chunk slots, the tensor's axes without the block axis] -> [slots, P, rest...]
                        buffer = self.arena.host(chunk, v.name, ti).movedim(1 + page_axis, 1)
                        at = torch.tensor([i for i, _, _ in rows], dtype=torch.long)
                        slots = torch.tensor([s for _, s, _ in rows], dtype=torch.long)
                        offsets = torch.tensor([o for _, _, o in rows], dtype=torch.long)
                        bounce.index_copy_(0, at, buffer[slots, offsets])
                    out.index_copy_(0, host_index, bounce.to(self.device, non_blocking=True))
                    issued += bounce.numel() * bounce.element_size()
                cache[dst_b, dst_o] = out
        self._http_h2d_issued = getattr(self, "_http_h2d_issued", 0) + issued
        self.drain()
        self._http_h2d_joined = getattr(self, "_http_h2d_issued", 0)

    def restore(self, groups, kind, payload):
        relevant = self.validate(groups, kind)
        require(set(payload) == {v.name for v in relevant}, "incomplete tensor payload")
        # Validate every target/payload before issuing the first copy.
        for v in relevant:
            require(len(payload[v.name]) == len(groups[v.group]), "block payload count mismatch")
            for bid, pieces in zip(groups[v.group], payload[v.name]):
                require(len(pieces) == len(v.tensors), "state payload count mismatch")
                for tensor, axis, piece in zip(v.tensors, v.axes, pieces):
                    target = self._slice(v,tensor,axis,bid)
                    require(piece.device.type == "cpu" and piece.dtype == target.dtype
                            and tuple(piece.shape) == tuple(target.shape), "tensor payload layout mismatch")
        for v in relevant:
            for bid, pieces in zip(groups[v.group], payload[v.name]):
                for tensor, axis, piece in zip(v.tensors, v.axes, pieces):
                    self._slice(v,tensor,axis,bid).copy_(piece, non_blocking=False)
                    self._http_h2d_issued = getattr(self, "_http_h2d_issued", 0) + piece.numel()*piece.element_size()
        self.drain()
        self._http_h2d_joined = getattr(self, "_http_h2d_issued", 0)

    # ---- --gdn-device-handoff: issue variants, events and lazy joins -------
    #
    # Each issue variant validates exactly like its synchronous twin, issues the
    # same copies on the current stream without a device drain and records one
    # event after them. H2D bytes count as issued at once and as joined when the
    # event is joined (poll_joins at every RPC frame boundary, any drain, or
    # ``join`` before a host read); each join is credited to the request whose
    # IO scope issued it, never to a scope that is open at the join. Pinned
    # sources and arena Pages stay referenced until their event is joined.
    # Device-to-device bytes need no join and count at issue. Nothing here runs
    # when the flag is off.

    def enable_handoff(self):
        """Worker startup, once: the issue variants become available."""
        require(not self.handoff, "GDN device hand-off already enabled")
        self.handoff, self._pending = True, []

    def _event(self):
        """An event recorded on the current stream, after the copies just issued."""
        event = self.torch.cuda.Event()
        event.record(self.torch.cuda.current_stream(self.device))
        return event

    def _issued(self, h2d=0, zeros=0, d2h=0, holds=()):
        require(self.handoff, "issue variants need --gdn-device-handoff")
        event = self._event()
        scopes = self.__dict__.get("_http_scopes")
        self._pending.append(_Pending(event, h2d, zeros, d2h, scopes[-1].request if scopes else None,
                                      tuple(holds)))
        return event

    def _credit(self, entry):
        """A joined entry: process counters and the issuing request; open scopes never see it."""
        self._http_h2d_joined = getattr(self, "_http_h2d_joined", 0) + entry.h2d
        self._http_zero_joins = getattr(self, "_http_zero_joins", 0) + entry.zeros
        if entry.d2h:
            self._http_d2h_joined = getattr(self, "_http_d2h_joined", 0) + entry.d2h
        if entry.request is not None:
            own = self.__dict__.setdefault("_http_requests", {}).setdefault(
                entry.request, [0] * len(_COUNTERS))
            own[1] += entry.h2d
            own[2] += entry.zeros
        for scope in self.__dict__.get("_http_scopes", ()):
            start = list(scope.start)
            start[1] += entry.h2d
            start[2] += entry.zeros
            scope.start = tuple(start)

    def poll_joins(self):
        """Join every issued copy, in issue order (RPC frame boundaries and drains)."""
        if self._pending:
            self.join(self._pending[-1].event)

    def join(self, event):
        """Join ``event`` and every copy issued before it (one stream: they complete in order)."""
        index = next((i for i, entry in enumerate(self._pending) if entry.event is event), None)
        if index is None:
            return          # joined before
        self._pending[index].event.synchronize()
        done = self._pending[:index + 1]
        del self._pending[:index + 1]
        for entry in done:
            self._credit(entry)

    def joined(self, event):
        return all(entry.event is not event for entry in self._pending)

    def wait(self, event):
        """Before a pinned buffer is reused: its last event has completed."""
        if event is None:
            return
        if self.joined(event):
            event.synchronize()
        else:
            self.join(event)

    def check_connector(self):
        """The drain's connector check without the device join (a step under the hand-off)."""
        check = getattr(self, "connector_check", None)
        if check is not None:
            check()

    def record_step(self):
        """A step event instead of the step's device drain (joined lazily)."""
        return self._issued()

    def issue_restore(self, groups, kind, payload):
        """``restore`` without the drain: non-blocking H2D from the pinned payload."""
        require(self.handoff, "issue variants need --gdn-device-handoff")
        relevant = self.validate(groups, kind)
        require(set(payload) == {v.name for v in relevant}, "incomplete tensor payload")
        for v in relevant:
            require(len(payload[v.name]) == len(groups[v.group]), "block payload count mismatch")
            for bid, pieces in zip(groups[v.group], payload[v.name]):
                require(len(pieces) == len(v.tensors), "state payload count mismatch")
                for tensor, axis, piece in zip(v.tensors, v.axes, pieces):
                    target = self._slice(v,tensor,axis,bid)
                    require(piece.device.type == "cpu" and piece.dtype == target.dtype
                            and tuple(piece.shape) == tuple(target.shape), "tensor payload layout mismatch")
        issued = 0
        for v in relevant:
            for bid, pieces in zip(groups[v.group], payload[v.name]):
                for tensor, axis, piece in zip(v.tensors, v.axes, pieces):
                    self._slice(v,tensor,axis,bid).copy_(piece, non_blocking=True)
                    issued += piece.numel()*piece.element_size()
        self._http_h2d_issued = getattr(self, "_http_h2d_issued", 0) + issued
        return self._issued(h2d=issued, holds=(payload,))

    def issue_copy(self, source, target, kind):
        """``copy`` without the drains: device-to-device on the current stream."""
        require(self.handoff, "issue variants need --gdn-device-handoff")
        relevant = self.validate(source, kind)
        require(set(target) == set(source) and self.validate(target, kind) == relevant and
                all(len(target[g]) == len(ids) for g, ids in source.items()),
                "copy source/target group coverage differs")
        require(not {b for ids in source.values() for b in ids} &
                {b for ids in target.values() for b in ids}, "copy source/target alias")
        copied = 0
        for v in relevant:
            for src, dst in zip(source[v.group], target[v.group]):
                for tensor, axis in zip(v.tensors, v.axes):
                    piece = self._slice(v,tensor,axis,src)
                    self._slice(v,tensor,axis,dst).copy_(piece, non_blocking=True)
                    copied += piece.numel()*piece.element_size()
        self._http_d2d_bytes = getattr(self, "_http_d2d_bytes", 0) + copied
        return self._issued()

    def issue_zero(self, groups, kind):
        """``zero`` without the drains; it counts as a zero join once joined."""
        require(self.handoff, "issue variants need --gdn-device-handoff")
        relevant = self.validate(groups, kind)
        if self.arena is not None and kind == "attention":
            index = self._kernel_index(groups, relevant)
            for v in relevant:
                for tensor, axis in zip(v.tensors, v.axes):
                    tensor.index_fill_(axis, index[(v.group, v.ratio)], 0)
        else:
            for v in relevant:
                for bid in groups[v.group]:
                    for tensor, axis in zip(v.tensors, v.axes):
                        self._slice(v,tensor,axis,bid).zero_()
        return self._issued(zeros=1)

    def issue_restore_pages(self, items):
        """``restore_pages`` without the drain; non-arena payloads go through issue_restore."""
        require(self.handoff, "issue variants need --gdn-device-handoff")
        items = list(items)
        others = [(g, p) for g, p in items if type(p) is not PagePayload]
        batch = sorted(((g, p) for g, p in items if type(p) is PagePayload), key=lambda x: x[1].slot)
        require(not batch or self.arena is not None, "Page arena not attached")
        if batch:
            groups, relevant = self._page_groups([g for g, _ in batch])
            require(len({p.slot for _, p in batch}) == len(batch) and
                    all(p.arena is self.arena and self.arena.slots.holds(p.slot, p.token)
                        for _, p in batch), "foreign or released Page arena payload")
        events = [self.issue_restore(g, "attention", p) for g, p in others]
        if not batch:
            return events[-1] if events else None
        issued, holds = 0, [p for _, p in batch]
        for lo, hi in self._slices(relevant, len(batch)):
            index = self._kernel_index({g: ids[lo:hi] for g, ids in groups.items()}, relevant)
            segments = self.arena.slots.segments([p.slot for _, p in batch[lo:hi]])
            for v in relevant:
                r = v.ratio
                for ti, (tensor, axis) in enumerate(zip(v.tensors, v.axes)):
                    rest = tuple(s for i, s in enumerate(tensor.shape) if i != axis)
                    staging = self.torch.empty(((hi - lo) * r,) + rest, dtype=tensor.dtype,
                                               device=self.device)
                    for chunk, offset, count, indices in segments:
                        host = self.arena.host(chunk, v.name, ti)
                        if type(indices) is range:
                            source = host.narrow(0, indices.start * r, count * r)
                        else:
                            source = self.torch.empty((count * r,) + rest, dtype=tensor.dtype,
                                                      device="cpu", pin_memory=self.arena.pin)
                            self.torch.index_select(host, 0, self.torch.tensor(
                                _expand(indices, r), dtype=self.torch.long), out=source)
                            holds.append(source)        # pinned bounce: alive until joined
                        staging.narrow(0, offset * r, count * r).copy_(source, non_blocking=True)
                    tensor.index_copy_(axis, index[(v.group, r)],
                                       staging.movedim(0, axis) if axis else staging)
                    issued += staging.numel() * staging.element_size()
        self._http_h2d_issued = getattr(self, "_http_h2d_issued", 0) + issued
        return self._issued(h2d=issued, holds=holds)

    def host_buffer(self, groups, kind):
        """A pinned payload shaped like ``capture(groups, kind)``'s, for issue_capture_into."""
        require(self.handoff, "issue variants need --gdn-device-handoff")
        relevant = self.validate(groups, kind)
        return {v.name: tuple(tuple(self._empty_host(self._slice(v,tensor,axis,bid))
                                    for tensor, axis in zip(v.tensors, v.axes))
                              for bid in groups[v.group])
                for v in relevant}

    def issue_capture_into(self, groups, kind, payload):
        """``capture`` into a preallocated pinned payload, without the drains.

        The payload gets a fresh identity token (``payload_digest``), so bytes
        written again are never taken for the state they held before. The
        caller has waited for the payload's previous event (``wait``)."""
        require(self.handoff, "issue variants need --gdn-device-handoff")
        relevant = self.validate(groups, kind)
        require(set(payload) == {v.name for v in relevant}, "incomplete capture buffer")
        copied = 0
        for v in relevant:
            require(len(payload[v.name]) == len(groups[v.group]), "capture buffer block count")
            for bid, pieces in zip(groups[v.group], payload[v.name]):
                require(len(pieces) == len(v.tensors), "capture buffer piece count")
                for tensor, axis, host in zip(v.tensors, v.axes, pieces):
                    sample = self._slice(v,tensor,axis,bid)
                    require(host.device.type == "cpu" and host.dtype == sample.dtype and
                            tuple(host.shape) == tuple(sample.shape), "capture buffer layout mismatch")
                    host.copy_(sample, non_blocking=True)
                    copied += host.numel() * host.element_size()
        self.retoken(payload)
        return self._issued(d2h=copied, holds=(payload,))

    @staticmethod
    def retoken(payload):
        """A fresh payload_digest identity for a payload whose bytes are rewritten."""
        first = next((t for name in sorted(payload) for pieces in payload[name] for t in pieces), None)
        require(first is not None, "empty tensor payload")
        first._remote_ra_payload_token = _PAYLOAD_PREFIX + "%016x" % next(_PAYLOAD_SEQUENCE)
        return first._remote_ra_payload_token

    @staticmethod
    def payload_digest(payload):
        """Identity token of a captured CPU payload, not a content hash.

        Unique in this process and stable for the object's lifetime: the first
        tensor carries it. Every caller compares it with the token recorded when
        that same private CPU copy was captured, so equality still proves the
        same payload is used. Hashing the bytes cost ~0.85 s per turn (about
        8.5 attention pages of 51.4 MB plus six 154 MB GDN states, v10d spans).
        An arena Page (decoupled) carries the token of its slot allocation.
        """
        if type(payload) is PagePayload:
            require(payload.arena.slots.holds(payload.slot, payload.token),
                    "Page arena slot no longer holds this payload")
            return payload.token
        first = next((t for name in sorted(payload) for pieces in payload[name] for t in pieces), None)
        require(first is not None, "empty tensor payload")
        token = getattr(first, "_remote_ra_payload_token", None)
        if token is None:
            token = _PAYLOAD_PREFIX + "%016x" % next(_PAYLOAD_SEQUENCE)
            first._remote_ra_payload_token = token
        return token


class CpuTensorTestIO(TensorIO):
    """Explicit real-CPU-tensor test substitute; cannot attain native readiness."""
    evidence_kind = "CPU_TENSOR_TEST_ONLY"

    def __init__(self, views, *, decoupled=False):
        import torch
        self.torch, self.views = torch, tuple(views)
        require(self.views and all(type(v) is LayerView for v in self.views) and
                all(t.device.type == "cpu" for v in self.views for t in v.tensors),
                "CPU test tensors required")
        self.device = torch.device("cpu")
        self.decoupled = decoupled is True
        self._validate_aliases()

    def drain(self):
        if self._pending:
            self.poll_joins()   # --gdn-device-handoff: CPU copies completed when issued
        return None

    def _event(self):
        return _DoneEvent()

    def _empty_host(self, sample):
        return self.torch.empty(tuple(sample.shape), dtype=sample.dtype, device="cpu")
