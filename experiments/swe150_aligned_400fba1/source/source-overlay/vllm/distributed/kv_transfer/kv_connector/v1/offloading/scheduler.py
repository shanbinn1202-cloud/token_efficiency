# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from fractions import Fraction
from itertools import islice
from typing import Any, NamedTuple

from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    SelectiveLoadPlan,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
    get_offloading_event_group_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    _TransferMetricName,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.selective_policy import (
    PageSelectionPolicyError,
    SelectivePageControls,
    parse_request_controls,
    select_logical_pages,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    make_offload_key,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)
_FORCED_OFFLOAD_DETAILED = (
    os.environ.get("VLLM_FORCED_OFFLOAD_TRACE", "compact") == "detailed"
)


def _select_logical_page_indices(
    num_pages: int,
    fraction: float,
    seed: int,
    prefix_identity: bytes,
) -> tuple[int, ...]:
    """Backward-compatible entry point for the random baseline policy."""

    return select_logical_pages(
        policy="random",
        num_pages=num_pages,
        fraction=fraction,
        seed=seed,
        prefix_identity=prefix_identity,
    ).logical_pages


def _prompt_token_ids_sha256(request: Request) -> str:
    """Return a canonical digest covering every prompt token, including tail."""
    cached = getattr(request, "_forced_offload_prompt_token_ids_sha256", None)
    if cached is not None:
        return cached
    token_ids = request.prompt_token_ids
    if token_ids is None:
        return "none"
    digest = sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, byteorder="big", signed=False))
    result = digest.hexdigest()
    setattr(request, "_forced_offload_prompt_token_ids_sha256", result)
    return result


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    # Number of workers still pending. Starts at num_workers,
    # decremented as each worker reports completion. Job is done at 0.
    pending_count: int
    # Offload keys this job covers; passed to manager.complete_*().
    keys: set[OffloadKey]
    is_store: bool
    # Store src block IDs whose ref_cnt protects them while the request
    # runs. Only registered in _block_id_to_pending_jobs on request_finished.
    non_sliding_window_block_ids: list[int] | None = None
    # Store src block IDs that may be freed before the request finishes.
    # Registered in _block_id_to_pending_jobs at store creation time.
    sliding_window_block_ids: list[int] | None = None


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    gpu_block_size: int
    offloaded_block_size: int
    hash_block_size_factor: int
    # KV cache spec metadata propagated onto emitted BlockStored events so
    # KV-aware consumers can classify and filter the group.
    kv_event_group_spec: OffloadingEventGroupSpec
    # None below means full attention
    sliding_window_size_in_blocks: int | None
    # Number of this group's offloaded blocks per full-attention alignment
    # segment. Used to skip storing SWA blocks that can never serve a load
    # hit (e.g. DeepSeek V4 where SWA groups have much smaller block sizes
    # than the MLA full-attention group).
    # None for full-attention groups or when the optimization doesn't apply.
    alignment_block_count: int | None = None
    # True for EAGLE/MTP draft-model attention groups. The trailing block
    # of these groups is volatile and lacks a stable hash, so it must
    # be excluded from store and load scheduling.
    is_eagle_group: bool = False
    is_mamba_group: bool = False


def get_sliding_window_size_in_blocks(
    kv_cache_spec: KVCacheSpec, offloaded_block_size: int
) -> int | None:
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        assert kv_cache_spec.sliding_window > 0
        return cdiv(kv_cache_spec.sliding_window, offloaded_block_size)

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1

    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None


def resolve_mamba_align_size(spec: "OffloadingSpec") -> int | None:
    """Scan all KV cache groups in *spec* and return the single mamba alignment
    size, or None if no group requires mamba alignment.

    For MambaSpec groups in "align" cache mode the hit window must be rounded
    down to a multiple of the offloaded block size. Asserts that all such
    groups agree on the same value.
    """
    mamba_align_size: int | None = None
    for idx, gpu_block_size in enumerate(spec.gpu_block_size):
        kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec
        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode == "align":
            offload_block_size = gpu_block_size * spec.block_size_factor
            assert mamba_align_size is None or mamba_align_size == offload_block_size
            mamba_align_size = offload_block_size
    return mamba_align_size


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    block_size_factor: int
    num_workers: int
    offload_prompt_only: bool

    @classmethod
    def from_spec(cls, spec: OffloadingSpec) -> "SchedulerOffloadConfig":
        # Determine the alignment token count from the full-attention group(s).
        # This is the offloaded_block_size of the full-attention group; load
        # hits are always aligned to this boundary, so SWA blocks earlier in
        # each segment can never serve a load hit. Relevant for hybrid
        # architectures like DeepSeek V4 (MLA + SWA groups).
        full_attn_offloaded_block_sizes: set[int] = set()
        for idx, gpu_block_size in enumerate(spec.gpu_block_size):
            kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_blocks(
                kv_spec, gpu_block_size * spec.block_size_factor
            )
            if sw is None:
                full_attn_offloaded_block_sizes.add(
                    gpu_block_size * spec.block_size_factor
                )

        # Only apply the optimization if there's a single consistent
        # full-attention alignment size.
        alignment_tokens: int | None = None
        if len(full_attn_offloaded_block_sizes) == 1:
            alignment_tokens = full_attn_offloaded_block_sizes.pop()

        def _alignment_block_count(
            offloaded_block_size: int,
            sliding_window_size_in_blocks: int | None,
        ) -> int | None:
            if alignment_tokens is None or sliding_window_size_in_blocks is None:
                return None
            if alignment_tokens <= offloaded_block_size:
                return None
            per_segment = alignment_tokens // offloaded_block_size
            if sliding_window_size_in_blocks >= per_segment:
                return None
            return per_segment

        eagle_groups = {
            idx
            for idx, g in enumerate(spec.kv_cache_config.kv_cache_groups)
            if g.is_eagle_group
        }

        use_eagle = (
            spec.vllm_config.speculative_config is not None
            and spec.vllm_config.speculative_config.use_eagle()
        )
        if use_eagle and not eagle_groups:
            eagle_groups = set(range(len(spec.kv_cache_config.kv_cache_groups)))

        if eagle_groups:
            logger.info(
                "KV offloading: EAGLE/MTP draft attention groups %s "
                "detected. The trailing block of these groups will be "
                "excluded from offloading due to volatility.",
                sorted(eagle_groups),
            )

        return cls(
            num_workers=spec.vllm_config.parallel_config.world_size,
            kv_group_configs=tuple(
                GroupOffloadConfig(
                    group_idx=idx,
                    gpu_block_size=gpu_block_size,
                    offloaded_block_size=gpu_block_size * spec.block_size_factor,
                    hash_block_size_factor=(
                        (gpu_block_size * spec.block_size_factor)
                        // spec.hash_block_size
                    ),
                    sliding_window_size_in_blocks=(
                        sw := get_sliding_window_size_in_blocks(
                            spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec,
                            gpu_block_size * spec.block_size_factor,
                        )
                    ),
                    alignment_block_count=_alignment_block_count(
                        gpu_block_size * spec.block_size_factor, sw
                    ),
                    kv_event_group_spec=get_offloading_event_group_spec(
                        spec.kv_cache_config.kv_cache_groups[idx]
                    ),
                    is_eagle_group=idx in eagle_groups,
                    is_mamba_group=isinstance(
                        spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec,
                        MambaSpec,
                    ),
                )
                for idx, gpu_block_size in enumerate(spec.gpu_block_size)
            ),
            block_size_factor=spec.block_size_factor,
            offload_prompt_only=spec.offload_prompt_only,
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # index of next block (of size offloaded_block_size) to offload
    next_stored_block_idx: int = 0
    # number of offloaded blocks hit (including GPU prefix cache)
    # when the request first started
    num_hit_blocks: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    req_context: ReqContext
    offloading_context: RequestOffloadingContext
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    # upper bound on tokens to offload for this request; None means no cap
    max_offload_tokens: int | None = None
    # Upper bound on the request-prefix boundary eligible for external load.
    # None means no cap. This is intentionally a contiguous-prefix control:
    # it does not represent arbitrary/non-contiguous block selection.
    max_load_tokens: int | None = None
    # Dry-run-only control. This computes and logs a deterministic subset of
    # matched full-attention pages but intentionally does not change loading.
    selective_page_dry_run_fraction: Fraction | None = None
    # Actual experimental random-omit mode. Unlike the dry-run field above,
    # this changes allocation, transfer, and attention behavior.
    selective_page_fraction: Fraction | None = None
    selective_page_controls: SelectivePageControls = field(
        default_factory=SelectivePageControls
    )
    selective_page_error: str | None = None
    selective_page_seed: int = 42
    selective_page_policy: str = "random"
    selective_page_ranked_pages: tuple[int, ...] | None = None
    selective_page_query_hash: str | None = None
    selective_load_plan: SelectiveLoadPlan | None = None
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0
    # In-flight job IDs. Per the connector's invariant, at any given time
    # this contains either a single load job, or one or more store jobs.
    transfer_jobs: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
        params = self.req.kv_transfer_params
        if params is not None and not isinstance(params, dict):
            self.selective_page_error = "kv_transfer_params must be an object"
            logger.warning(
                "SELECTIVE_PAGE_FALLBACK request=%s reason=%s; using exact full load",
                self.req.request_id,
                self.selective_page_error,
            )
            return

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_offload_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_offload_tokens = raw
            logger.debug(
                "Request %s: max_offload_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_offload_tokens must be a non-negative int, got %r; ignoring", raw
            )

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_load_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_load_tokens = raw
            logger.info(
                "Request %s: max_load_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_load_tokens must be a non-negative int, got %r; ignoring", raw
            )

        try:
            controls = parse_request_controls(params)
        except PageSelectionPolicyError as exc:
            self.selective_page_error = str(exc)
            logger.warning(
                "SELECTIVE_PAGE_FALLBACK request=%s reason=invalid_controls:%s; "
                "using exact full load",
                self.req.request_id,
                exc,
            )
            return
        self.selective_page_controls = controls
        self.selective_page_dry_run_fraction = controls.dry_run_fraction
        self.selective_page_fraction = controls.fraction
        self.selective_page_seed = controls.seed
        self.selective_page_policy = controls.policy
        self.selective_page_ranked_pages = controls.ranked_logical_pages
        self.selective_page_query_hash = controls.query_hash

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hash_block_size_factor * len(group_state.offload_keys)
                + group_config.hash_block_size_factor
                - 1,
                None,
                group_config.hash_block_size_factor,
            ):
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
            group_state.next_stored_block_idx = num_blocks

    def update_num_hit_blocks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_blocks = (
                num_cached_tokens // group_config.offloaded_block_size
            )


def _create_req_context(req: Request) -> ReqContext:
    return ReqContext(
        req_id=req.request_id,
        kv_transfer_params=req.kv_transfer_params,
    )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        spec: OffloadingSpec,
    ):
        self.config = SchedulerOffloadConfig.from_spec(spec)
        self.manager: OffloadingManager = spec.get_manager()
        # remote-ra decoupled KV layout: the worker registers pool 0 only
        # (Mamba groups have no CPU copy), so the connector must stay isolated
        # and emit no jobs. Group configs keep one entry per KV cache group.
        self._remote_ra_decoupled = spec.kv_cache_config.group_pool is not None
        self._connector_stats: OffloadingConnectorStats | None = None

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_blocks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self.config.kv_group_configs[i].sliding_window_size_in_blocks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self._full_attention_groups: tuple[int, ...] = tuple(full_attention_groups)
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)
        self._lookup_groups = self._full_attention_groups + self._sliding_window_groups
        self._mamba_align_size: int | None = resolve_mamba_align_size(spec)

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        # Explicit trusted bootstrap only. No client field can enable or supply
        # the producer authority. Untagged requests keep the v3.2 path.
        self._policy_runtime = None
        self._policy_runtime_context = None
        self._policy_runtime_handles: dict[str, str] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        # GPU block IDs allocated in the current engine step
        self._current_batch_allocated_block_ids: set[int] = set()
        # if GPU prefix caching is enabled,
        # track loaded blocks to avoid redundant loads
        self._blocks_being_loaded: set[OffloadKey] | None = (
            set() if spec.vllm_config.cache_config.enable_prefix_caching else None
        )

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        # Threshold value for stale jobs. All job ids >= _stale_job_threshold are
        # active jobs.
        self._stale_job_threshold: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}
        self._forced_offload_last_pending_signature: (
            tuple[int, bool, int, int] | None
        ) = None

        # block_id -> pending store job_ids. Used to track jobs that needs
        # flushing in case a block is re-allocated by the KV cache manager.
        # Populated only for finished requests (running-request blocks are
        # protected by their ref_cnt) and for sliding window blocks (which can
        # be freed before a request finishes).
        self._block_id_to_pending_jobs: dict[int, set[int]] = {}

        self._events_tracker = OffloadingEventsTracker(spec.kv_events_config)
        logger.info(
            "FORCED_OFFLOAD_LAYOUT block_size_factor=%d mamba_align_tokens=%s "
            "groups=%s",
            self.config.block_size_factor,
            self._mamba_align_size,
            [
                {
                    "group": group.group_idx,
                    "gpu_block_tokens": group.gpu_block_size,
                    "offloaded_block_tokens": group.offloaded_block_size,
                    "hash_block_size_factor": group.hash_block_size_factor,
                    "sliding_window_blocks": group.sliding_window_size_in_blocks,
                    "is_mamba": group.is_mamba_group,
                }
                for group in self.config.kv_group_configs
            ],
        )

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def _remove_pending_job(self, job_id: int, block_ids: list[int] | None) -> None:
        for bid in block_ids or ():
            pending = self._block_id_to_pending_jobs[bid]
            pending.remove(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[bid]

    def _maximal_prefix_lookup(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> int | None:
        """Return the number of consecutive offloaded blocks from the start,
        or None if the backend deferred a lookup."""
        hit_count = 0
        defer_lookup = False
        for key in keys:
            result = self.manager.lookup(key, req_context)
            if result is None:
                defer_lookup = True
                # continue lookup to allow manager to kick-off async lookups
                # for all blocks (until a miss is detected)
                result = True
            if not result:
                break
            hit_count += 1
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
    ) -> int | None:
        """Return the end index (in `keys`) of the last run of
        `sliding_window_size` consecutive hits, scanning from the end.
        Returns 0 on miss, None if the backend deferred a lookup."""
        defer_lookup = False
        consecutive_hits = 0
        for idx in range(len(keys) - 1, -1, -1):
            result = self.manager.lookup(keys[idx], req_context)
            if result is None:
                defer_lookup = True
                # continue lookup to allow manager to kick-off async lookups
                # for all blocks (until a hit is detected)
                result = False
            if not result:
                consecutive_hits = 0
            else:
                consecutive_hits += 1
                if consecutive_hits == sliding_window_size:
                    return idx + sliding_window_size if not defer_lookup else None
        return consecutive_hits if not defer_lookup else None

    def _touch(self, req_status: RequestOffloadState):
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.sliding_window_size_in_blocks is None:
                self.manager.touch(group_state.offload_keys, req_status.req_context)
            else:
                # we aim to keep just blocks that are necessary to hit
                # the original request (+ decoded blocks)
                blocks_to_skip = max(
                    0,
                    group_state.num_hit_blocks
                    - group_config.sliding_window_size_in_blocks,
                )
                self.manager.touch(
                    group_state.offload_keys[blocks_to_skip:],
                    req_status.req_context,
                )

    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        """
        Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Iterates full-attention groups first (prefix lookup), then sliding-window
        groups (suffix lookup). Each group may tighten max_hit_size_tokens, which
        can invalidate an earlier group's result, so the loop re-runs when that
        happens until num_hit_tokens converges.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        max_hit_size_tokens: int = req_status.req.num_tokens
        block_chain_sha256 = sha256(b"".join(req_status.req.block_hashes)).hexdigest()
        prompt_token_ids_sha256 = _prompt_token_ids_sha256(req_status.req)
        logger.info(
            "FORCED_OFFLOAD_LOOKUP_START request=%s num_prompt_tokens=%d "
            "num_tokens=%d prompt_token_ids_sha256=%s block_chain_sha256=%s "
            "full_hash_blocks=%d local_tokens=%d lookup_groups=%s",
            req_status.req.request_id,
            req_status.req.num_prompt_tokens,
            req_status.req.num_tokens,
            prompt_token_ids_sha256,
            block_chain_sha256,
            len(req_status.req.block_hashes),
            num_computed_tokens,
            self._lookup_groups,
        )
        if self._sliding_window_groups:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            max_hit_size_tokens -= 1
            if self._mamba_align_size is not None:
                # Constrain hit-window to the mamba block size.
                max_hit_size_tokens = round_down(
                    max_hit_size_tokens, self._mamba_align_size
                )

        # Apply the experimental load boundary before looking up any cache
        # group. Every full-attention, sliding-window, and Mamba group must
        # therefore validate keys at the same boundary. Applying this cap only
        # to the scalar returned by _lookup() is unsafe: windowed/recurrent
        # groups may have validated a different boundary, and prepare_load()
        # can then receive keys that were never confirmed present.
        max_load_tokens = req_status.max_load_tokens
        if max_load_tokens is not None:
            uncapped_max_hit_size_tokens = max_hit_size_tokens
            max_hit_size_tokens = min(max_hit_size_tokens, max_load_tokens)
            if self._mamba_align_size is not None:
                max_hit_size_tokens = round_down(
                    max_hit_size_tokens, self._mamba_align_size
                )
            max_hit_size_tokens = max(num_computed_tokens, max_hit_size_tokens)
            logger.info(
                "Request %s: external-load boundary capped from %d to %d tokens "
                "(requested=%d, locally_computed=%d)",
                req_status.req.request_id,
                uncapped_max_hit_size_tokens,
                max_hit_size_tokens,
                max_load_tokens,
                num_computed_tokens,
            )

        num_hit_tokens: int = 0
        defer_lookup = False
        lookup_groups = self._lookup_groups

        # Tracks which eagle groups have already popped their volatile trailing block
        # in the current convergence iteration. Reset when a non-eagle group
        # tightens the hit boundary, requiring a fresh pop.
        eagle_verified: set[int] = set()
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self.config.kv_group_configs[
                    group_idx
                ]
                group_state: RequestGroupState = req_status.group_states[group_idx]
                offloaded_block_size = group_config.offloaded_block_size
                offload_keys = group_state.offload_keys

                assert (
                    len(offload_keys)
                    >= req_status.req.num_tokens // offloaded_block_size
                )

                is_eagle_unverified = (
                    group_config.is_eagle_group and group_idx not in eagle_verified
                )

                # Constrain to block-aligned boundary for this group
                max_hit_size_tokens = min(
                    max_hit_size_tokens, len(offload_keys) * offloaded_block_size
                )
                if max_hit_size_tokens - num_computed_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    logger.info(
                        "FORCED_OFFLOAD_LOOKUP_RESULT request=%s result=miss "
                        "reason=less_than_one_block group=%d candidate_tokens=%d "
                        "block_tokens=%d",
                        req_status.req.request_id,
                        group_idx,
                        max_hit_size_tokens - num_computed_tokens,
                        offloaded_block_size,
                    )
                    return 0

                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )

                # For eagle groups, query one extra block that will be popped.
                # We only need to increase the query size for sliding window groups.
                query_max = max_hit_size_tokens
                if is_eagle_unverified and sliding_window_size_in_blocks is not None:
                    query_max = min(
                        max_hit_size_tokens + offloaded_block_size,
                        len(offload_keys) * offloaded_block_size,
                    )

                num_blocks = min(
                    cdiv(query_max, offloaded_block_size), len(offload_keys)
                )
                start_block_idx = num_computed_tokens // offloaded_block_size
                offload_keys = offload_keys[start_block_idx:num_blocks]

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_blocks: int | None
                if sliding_window_size_in_blocks is None:
                    num_hit_blocks = self._maximal_prefix_lookup(
                        offload_keys, req_status.req_context
                    )
                else:
                    required_window = sliding_window_size_in_blocks
                    if is_eagle_unverified:
                        required_window += 1
                    num_hit_blocks = self._sliding_window_lookup(
                        offload_keys,
                        required_window,
                        req_status.req_context,
                    )
                if _FORCED_OFFLOAD_DETAILED:
                    logger.info(
                        "FORCED_OFFLOAD_GROUP_LOOKUP request=%s group=%d "
                        "kind=%s candidate_blocks=%d start_block=%d "
                        "block_tokens=%d hit_blocks=%s",
                        req_status.req.request_id,
                        group_idx,
                        "prefix" if sliding_window_size_in_blocks is None else "window",
                        len(offload_keys),
                        start_block_idx,
                        offloaded_block_size,
                        num_hit_blocks,
                    )
                if num_hit_blocks == 0:
                    logger.info(
                        "FORCED_OFFLOAD_LOOKUP_RESULT request=%s result=miss "
                        "reason=group_miss group=%d",
                        req_status.req.request_id,
                        group_idx,
                    )
                    return 0

                if num_hit_blocks is None:
                    defer_lookup = True
                else:
                    if is_eagle_unverified:
                        num_hit_blocks -= 1
                        eagle_verified.add(group_idx)

                    max_hit_size_tokens = min(
                        max_hit_size_tokens,
                        offloaded_block_size * (start_block_idx + num_hit_blocks),
                    )

                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    logger.info(
                        "FORCED_OFFLOAD_LOOKUP_RESULT request=%s result=miss "
                        "reason=converged_less_than_one_block group=%d "
                        "candidate_tokens=%d block_tokens=%d",
                        req_status.req.request_id,
                        group_idx,
                        new_num_hit_tokens,
                        offloaded_block_size,
                    )
                    return 0

                if new_num_hit_tokens < num_hit_tokens:
                    if not group_config.is_eagle_group:
                        eagle_verified.clear()
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = self._lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self._sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_blocks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            logger.info(
                "FORCED_OFFLOAD_LOOKUP_RESULT request=%s result=deferred "
                "local_tokens=%d",
                req_status.req.request_id,
                num_computed_tokens,
            )
            return None

        # possibly delay request if any of the hit blocks is already being loaded
        if self._blocks_being_loaded:
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                offloaded_block_size = group_config.offloaded_block_size
                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )
                offload_keys = group_state.offload_keys
                num_blocks = cdiv(
                    num_computed_tokens + num_hit_tokens, offloaded_block_size
                )
                start_block_idx = num_computed_tokens // offloaded_block_size
                offload_keys = offload_keys[start_block_idx:num_blocks]
                if sliding_window_size_in_blocks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_blocks:]
                if any(key in self._blocks_being_loaded for key in offload_keys):
                    # hit blocks are being loaded, delay request
                    logger.debug(
                        "Delaying request %s since some of its"
                        " blocks are already being loaded",
                        req_status.req.request_id,
                    )
                    logger.info(
                        "FORCED_OFFLOAD_LOOKUP_RESULT request=%s result=deferred "
                        "reason=shared_keys_being_loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )
        logger.info(
            "FORCED_OFFLOAD_LOOKUP_RESULT request=%s result=hit "
            "local_tokens=%d external_tokens=%d",
            req_status.req.request_id,
            num_computed_tokens,
            num_hit_tokens,
        )

        return num_hit_tokens

    def _log_selective_page_dry_run(
        self, req_status: RequestOffloadState, num_hit_tokens: int
    ) -> None:
        """Log a deterministic full-attention page plan without applying it."""

        fraction = req_status.selective_page_dry_run_fraction
        if fraction is None or num_hit_tokens <= 0:
            return

        for group_idx in self._full_attention_groups:
            group_config = self.config.kv_group_configs[group_idx]
            num_pages = num_hit_tokens // group_config.offloaded_block_size
            if num_pages == 0:
                continue

            # Identify the matched prefix independently of the KV group. The
            # final request block hash at this boundary is stable for an exact
            # prompt replay, while an OffloadKey also contains the group index.
            hash_idx = num_pages * group_config.hash_block_size_factor - 1
            if hash_idx < 0 or hash_idx >= len(req_status.req.block_hashes):
                logger.warning(
                    "SELECTIVE_PAGE_FALLBACK request=%s reason=dry_run_prefix_hash_missing; using exact full load",
                    req_status.req.request_id,
                )
                return
            prefix_identity = req_status.req.block_hashes[hash_idx]
            try:
                selected = _select_logical_page_indices(
                    num_pages,
                    fraction,
                    req_status.selective_page_seed,
                    prefix_identity,
                )
            except PageSelectionPolicyError as exc:
                logger.warning(
                    "SELECTIVE_PAGE_FALLBACK request=%s reason=dry_run:%s; using exact full load",
                    req_status.req.request_id,
                    exc,
                )
                return
            logger.info(
                "SELECTIVE_PAGE_DRY_RUN request=%s group=%d matched_tokens=%d "
                "page_tokens=%d matched_pages=%d selected_pages=%d "
                "fraction=%.6f seed=%d logical_pages=%s "
                "(NO BEHAVIOR CHANGE)",
                req_status.req.request_id,
                group_idx,
                num_hit_tokens,
                group_config.offloaded_block_size,
                num_pages,
                len(selected),
                fraction,
                req_status.selective_page_seed,
                list(selected),
            )

    def _create_selective_load_plan(
        self, req_status: RequestOffloadState, num_hit_tokens: int
    ) -> SelectiveLoadPlan | None:
        """Create a policy-selected load plan after normal lookup converges.

        Unsupported layouts deliberately fall back to the exact full-load path.
        The mechanism supports one full-attention group, zero local GPU prefix
        hits, and Mamba-align groups for every remaining group.
        """

        controls = req_status.selective_page_controls
        fraction = controls.fraction
        req_status.selective_load_plan = None
        # Invalid controls already emitted FALLBACK during request admission.
        if req_status.selective_page_error is not None or not controls.requested:
            return None

        reason: str | None = None
        if req_status.num_locally_computed_tokens != 0:
            reason = "local GPU prefix hits are not supported"
        elif num_hit_tokens <= 0:
            reason = "no external prefix hit"
        elif len(self._full_attention_groups) != 1:
            reason = "the MVP requires exactly one full-attention KV group"
        elif any(
            not self.config.kv_group_configs[idx].is_mamba_group
            for idx in self._sliding_window_groups
        ):
            reason = "only Mamba-align non-full-attention groups are supported"
        elif self._sliding_window_groups and self._mamba_align_size is None:
            reason = "Mamba groups must use align mode"

        if reason is not None:
            logger.warning(
                "SELECTIVE_PAGE_FALLBACK request=%s reason=%s; using exact full load",
                req_status.req.request_id,
                reason,
            )
            return None

        group_idx = self._full_attention_groups[0]
        group_config = self.config.kv_group_configs[group_idx]
        page_tokens = group_config.offloaded_block_size
        if num_hit_tokens % page_tokens:
            logger.warning(
                "SELECTIVE_PAGE_FALLBACK request=%s matched_tokens=%d is not "
                "aligned to page_tokens=%d; using exact full load",
                req_status.req.request_id,
                num_hit_tokens,
                page_tokens,
            )
            return None

        num_pages = num_hit_tokens // page_tokens
        hash_idx = num_pages * group_config.hash_block_size_factor - 1
        if (
            num_pages == 0
            or hash_idx < 0
            or hash_idx >= len(req_status.req.block_hashes)
        ):
            logger.warning(
                "SELECTIVE_PAGE_FALLBACK request=%s reason=matched_prefix_hash_missing; using exact full load",
                req_status.req.request_id,
            )
            return None
        prefix_identity = req_status.req.block_hashes[hash_idx]
        try:
            selection = select_logical_pages(
                policy=req_status.selective_page_policy,
                num_pages=num_pages,
                fraction=fraction,
                seed=req_status.selective_page_seed,
                prefix_identity=prefix_identity,
                ranked_logical_pages=req_status.selective_page_ranked_pages,
                query_hash=req_status.selective_page_query_hash,
                page_tokens=page_tokens,
                contract_version=controls.contract_version,
                expected_num_pages=controls.expected_num_pages,
                expected_page_tokens=controls.expected_page_tokens,
                expected_selected_count=controls.selected_count,
                expected_prefix_identity=controls.expected_prefix_identity,
            )
        except PageSelectionPolicyError as exc:
            logger.warning(
                "SELECTIVE_PAGE_FALLBACK request=%s policy=%s reason=%s; "
                "using exact full load",
                req_status.req.request_id,
                req_status.selective_page_policy,
                exc,
            )
            return None
        selected = selection.logical_pages
        if len(selected) == num_pages:
            logger.warning(
                "SELECTIVE_PAGE_FALLBACK request=%s policy=%s reason=all_pages_selected; using exact full load",
                req_status.req.request_id,
                selection.policy,
            )
            return None
        if not selected:
            logger.warning(
                "SELECTIVE_PAGE_FALLBACK request=%s selected zero pages; "
                "using exact full load",
                req_status.req.request_id,
            )
            return None

        plan = SelectiveLoadPlan(
            req_id=req_status.req.request_id,
            logical_matched_tokens=num_hit_tokens,
            physical_full_attention_tokens=len(selected) * page_tokens,
            page_tokens=page_tokens,
            selected_logical_pages=selected,
            full_attention_group_ids=self._full_attention_groups,
            policy=selection.policy,
            query_hash=req_status.selective_page_query_hash,
        )
        req_status.selective_load_plan = plan
        logger.info(
            "SELECTIVE_PAGE_PLAN request=%s policy=%s query_hash=%s "
            "logical_tokens=%d physical_fa_tokens=%d dropped_fa_tokens=%d "
            "page_tokens=%d prefix_identity=%s fraction=%.6f seed=%d "
            "logical_pages=%s",
            plan.req_id,
            plan.policy,
            plan.query_hash,
            plan.logical_matched_tokens,
            plan.physical_full_attention_tokens,
            plan.dropped_full_attention_tokens,
            plan.page_tokens,
            prefix_identity.hex(),
            float(fraction) if fraction is not None else len(selected) / num_pages,
            req_status.selective_page_seed,
            list(plan.selected_logical_pages),
        )
        return plan

    def on_new_request(self, request: Request) -> None:
        """Called when a new request is added to the scheduler."""
        req_context = _create_req_context(request)
        offloading_context = self.manager.on_new_request(req_context)
        req_status = RequestOffloadState(
            config=self.config,
            req=request,
            req_context=req_context,
            offloading_context=offloading_context,
        )
        self._req_status[request.request_id] = req_status

    def configure_policy_runtime(self, coordinator, context_provider) -> None:
        """Install scheduler-owned producers and trusted per-request context.

        context_provider runs in the scheduler and must resolve server-owned
        session/capture state, never accept client-supplied ranked controls.
        """
        if self._req_status:
            raise RuntimeError("configure policy runtime before admitting requests")
        self._policy_runtime = coordinator
        self._policy_runtime_context = context_provider

    def _create_runtime_load_plan(self, req_status, num_hit_tokens):
        from dataclasses import replace
        import secrets
        from policy_runtime.coordinator import (
            RuntimeBinding, CaptureSnapshot, QuerySnapshot, PolicyRuntimeAbort, canonical_prompt_hash,
        )
        request = req_status.req
        params = request.kv_transfer_params or {}
        tag = params.get("policy_runtime")
        coordinator = getattr(self, "_policy_runtime", None)
        if coordinator is None:
            raise PolicyRuntimeAbort("tagged runtime request without trusted bootstrap")
        if (not isinstance(tag, dict) or set(tag) != {"version", "policy"}
                or type(tag.get("version")) is not int or tag["version"] != 1
                or tag.get("policy") not in ("goal_query", "observation_query")
                or any(str(key).startswith("selective_page_") for key in params)):
            raise PolicyRuntimeAbort("invalid runtime tag or client-ranked controls")
        context = dict(self._policy_runtime_context(request))
        query = context.get("stage") == "query"
        capture = context.get("stage") in ("capture", "query")
        if ((num_hit_tokens != 0 if capture else num_hit_tokens <= 0)
                or req_status.num_locally_computed_tokens != 0
                or len(self._full_attention_groups) != 1
                or any(not self.config.kv_group_configs[i].is_mamba_group
                       for i in self._sliding_window_groups)
                or (self._sliding_window_groups and self._mamba_align_size is None)):
            raise PolicyRuntimeAbort("runtime requires aligned external hit and supported layout")
        group = self.config.kv_group_configs[self._full_attention_groups[0]]
        page_tokens = group.offloaded_block_size
        if num_hit_tokens % page_tokens:
            raise PolicyRuntimeAbort("external hit is not page aligned")
        n = context["num_pages"] if capture else num_hit_tokens // page_tokens
        if type(n) is not int or (n != 0 if query else n <= 0) or context.get("num_pages", n) != n:
            raise PolicyRuntimeAbort("actual external universe differs from trusted history")
        hash_idx = n * group.hash_block_size_factor - 1
        if not query and hash_idx >= len(request.block_hashes):
            raise PolicyRuntimeAbort("matched native prefix hash unavailable")
        # This context is supplied by bootstrap, not kv_transfer_params.
        required = ("session_id", "generation", "stage", "turn", "nonce",
                    "model_revision", "producer_provenance")
        if any(not context.get(k) for k in required):
            raise PolicyRuntimeAbort("trusted runtime context incomplete")
        prompt = list(request.prompt_token_ids)
        if len(prompt) < n * page_tokens:
            raise PolicyRuntimeAbort("matched tokens exceed rendered prompt")
        def snapshot():
            current_prompt = list(request.prompt_token_ids)
            current_context = dict(self._policy_runtime_context(request))
            return replace(binding,
                canonical_prompt_hash=canonical_prompt_hash(current_prompt),
                native_prefix_identity="" if query else request.block_hashes[hash_idx].hex(),
                cumulative_prefix_identities=tuple(canonical_prompt_hash(
                    current_prompt[:(i + 1) * page_tokens]) for i in range(n)),
                local_hit_tokens=req_status.num_locally_computed_tokens,
                page_tokens=group.offloaded_block_size,
                external_hit_tokens=self._lookup(req_status),
                metadata=current_context.get("metadata", {}),
                policy=tag["policy"],
                **{k: current_context[k] for k in required})
        existing_handle = self._policy_runtime_handles.get(request.request_id)
        if existing_handle is not None:
            if capture:
                binding = coordinator.resolve(existing_handle)
                if snapshot() != binding:
                    coordinator.cancel(existing_handle, "cold_capture_binding_changed")
                    raise PolicyRuntimeAbort("cold capture binding changed during allocation deferral")
                return None
            binding = coordinator.resolve_committed(existing_handle)
            return coordinator.reuse_committed_plan(existing_handle,
                recheck=snapshot, plan=req_status.selective_load_plan)
        binding_type = QuerySnapshot if query else (CaptureSnapshot if capture else RuntimeBinding)
        binding = binding_type(
            reservation_id=secrets.token_hex(24), request_id=request.request_id,
            policy=tag["policy"], canonical_prompt_hash=canonical_prompt_hash(prompt),
            native_prefix_identity="" if query else request.block_hashes[hash_idx].hex(),
            cumulative_prefix_identities=tuple(canonical_prompt_hash(
                prompt[:(i + 1) * page_tokens]) for i in range(n)),
            num_pages=n, page_tokens=page_tokens, external_hit_tokens=num_hit_tokens,
            metadata=context.get("metadata", {}), **{k: context[k] for k in required})
        handle = coordinator.reserve(binding)
        self._policy_runtime_handles[request.request_id] = handle
        register = getattr(self._policy_runtime_context, "register_binding", None)
        if register is not None:
            register(binding, handle, context)
        if capture:
            logger.info("POLICY_RUNTIME_COLD_CAPTURE request=%s reservation=%s historical_tokens=%d",
                        request.request_id, binding.reservation_id, n * page_tokens)
            return None
        def validate(params, live):
            controls = parse_request_controls(params)
            if (not controls.requested or controls.policy != live.policy
                    or controls.expected_num_pages != live.num_pages
                    or controls.expected_page_tokens != live.page_tokens
                    or controls.expected_prefix_identity != live.native_prefix_identity
                    or controls.selected_count is None
                    or not 0 < controls.selected_count < live.num_pages):
                raise PolicyRuntimeAbort("producer controls mismatch or Full selection")
            return controls
        def plan(controls):
            req_status.selective_page_controls = controls
            req_status.selective_page_error = None
            req_status.selective_page_fraction = controls.fraction
            req_status.selective_page_seed = controls.seed
            req_status.selective_page_policy = controls.policy
            req_status.selective_page_ranked_pages = controls.ranked_logical_pages
            req_status.selective_page_query_hash = controls.query_hash
            return self._create_selective_load_plan(req_status, num_hit_tokens)
        return coordinator.prepare_and_commit(handle, context.get("metadata", {}),
            recheck=snapshot, control_validator=validate, planner=plan)

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()

        if req_status.transfer_jobs:
            logger.debug(
                "Delaying request %s since it still has in-flight transfers",
                request.request_id,
            )
            logger.info(
                "FORCED_OFFLOAD_LOOKUP_RESULT request=%s result=deferred "
                "reason=request_transfer_jobs pending_jobs=%d",
                request.request_id,
                len(req_status.transfer_jobs),
            )
            return None, False

        req_status.update_offload_keys()
        req_status.num_locally_computed_tokens = num_computed_tokens

        if isinstance(request.kv_transfer_params, dict) and "policy_runtime" in request.kv_transfer_params:
            from policy_runtime.coordinator import PolicyRuntimeAbort
            if self._policy_runtime_context is None:
                raise PolicyRuntimeAbort("runtime request requires configured scheduler")
            context = self._policy_runtime_context(request)
            if "num_pages" in context:
                if len(self._full_attention_groups) != 1:
                    raise PolicyRuntimeAbort("runtime requires exactly one full-attention group")
                page_tokens = self.config.kv_group_configs[self._full_attention_groups[0]].offloaded_block_size
                history_tokens = context["num_pages"] * page_tokens
                # Probe suffixes are never stored or externally restored.
                req_status.max_offload_tokens = history_tokens
                req_status.max_load_tokens = 0 if context["stage"] in ("capture", "query") else history_tokens

        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache:
            logger.info(
                "FORCED_OFFLOAD_LOOKUP_RESULT request=%s result=miss "
                "reason=skip_reading_prefix_cache",
                request.request_id,
            )
            num_hit_tokens = 0
        else:
            num_hit_tokens = self._lookup(req_status)
        if num_hit_tokens is not None:
            if isinstance(request.kv_transfer_params, dict) and "policy_runtime" in request.kv_transfer_params:
                self._create_runtime_load_plan(req_status, num_hit_tokens)
            else:
                self._log_selective_page_dry_run(req_status, num_hit_tokens)
                self._create_selective_load_plan(req_status, num_hit_tokens)
        req_status.update_num_hit_blocks(num_computed_tokens + (num_hit_tokens or 0))

        self._touch(req_status)

        return num_hit_tokens, bool(num_hit_tokens)

    def get_selective_load_plan(self, request_id: ReqId) -> SelectiveLoadPlan | None:
        req_status = self._req_status.get(request_id)
        return None if req_status is None else req_status.selective_load_plan

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens

        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        selective_plan = req_status.selective_load_plan
        for group_idx, (group_config, group_state, group_blocks) in enumerate(
            zip(
                self.config.kv_group_configs,
                req_status.group_states,
                blocks.blocks,
            )
        ):
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            gpu_block_size = group_config.gpu_block_size
            offloaded_block_size = group_config.offloaded_block_size
            offload_keys = group_state.offload_keys
            selective_full_attention = (
                selective_plan is not None
                and group_idx in selective_plan.full_attention_group_ids
            )
            group_cached_tokens = (
                selective_plan.physical_full_attention_tokens
                if selective_full_attention
                else num_cached_tokens
            )
            num_gpu_blocks = cdiv(group_cached_tokens, gpu_block_size)

            assert len(group_blocks) >= num_gpu_blocks
            num_locally_computed_gpu_blocks = num_gpu_blocks
            # Skip null placeholder blocks (used for sliding window or mamba padding).
            for i, block in enumerate(group_blocks[:num_gpu_blocks]):
                if not block.is_null and block.block_hash is None:
                    num_locally_computed_gpu_blocks = i
                    break

            assert (
                num_locally_computed_tokens
                <= num_locally_computed_gpu_blocks * gpu_block_size
            )
            num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks

            if group_config.sliding_window_size_in_blocks is not None:
                assert (
                    num_pending_gpu_blocks
                    <= group_config.sliding_window_size_in_blocks
                    * self.config.block_size_factor
                )

            num_blocks = cdiv(group_cached_tokens, offloaded_block_size)
            if num_pending_gpu_blocks:
                if selective_full_attention:
                    assert selective_plan is not None
                    assert num_locally_computed_gpu_blocks == 0
                    selected_keys = [
                        offload_keys[page_idx]
                        for page_idx in selective_plan.selected_logical_pages
                    ]
                    assert len(selected_keys) == num_blocks
                    keys_to_load.extend(selected_keys)
                else:
                    assert len(offload_keys) >= num_blocks
                    start_block_idx = (
                        num_locally_computed_gpu_blocks // self.config.block_size_factor
                    )
                    keys_to_load.extend(offload_keys[start_block_idx:num_blocks])

            dst_block_ids.extend(
                block.block_id
                for block in group_blocks[
                    num_locally_computed_gpu_blocks:num_gpu_blocks
                ]
            )
            group_sizes.append(num_pending_gpu_blocks)
            block_indices.append(num_locally_computed_gpu_blocks)

            # Skip prefix-hit blocks for block-level policy; for
            # request-level, next_stored_block_idx stays at 0 so all
            # blocks (including hits) are offloaded.
            if req_status.offloading_context.policy == OffloadPolicy.BLOCK_LEVEL:
                group_state.next_stored_block_idx = (
                    cdiv(num_cached_tokens, offloaded_block_size)
                    if selective_full_attention
                    else num_blocks
                )

        if selective_plan is not None:
            logger.info(
                "SELECTIVE_PAGE_TRANSFER request=%s selected_pages=%d "
                "source_keys=%d destination_gpu_blocks=%d",
                request.request_id,
                len(selective_plan.selected_logical_pages),
                len(keys_to_load),
                len(dst_block_ids),
            )

        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )

        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            transfer_spec=(src_spec, dst_spec),
        )
        # a load can only be issued when no other jobs are pending.
        assert not req_status.transfer_jobs
        req_status.transfer_jobs.add(load_job_id)
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
        )

        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(keys_to_load)

    def _update_req_states(self, scheduler_output: SchedulerOutput) -> None:
        """
        Update request states from the Scheduler's output.
        """

        # new_block_ids_end[req_id][i] = end of pre-existing block_ids for
        # the i-th sliding window group (before this step's extend).
        # Used to detect sliding window blocks that got re-allocated.
        new_block_ids_end: dict[str, tuple[int, ...]] = {}

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                if self._sliding_window_groups:
                    new_block_ids_end[req_id] = tuple(
                        len(req_status.group_states[grp_idx].block_ids)
                        for grp_idx in self._sliding_window_groups
                    )
                req_status.update_block_id_groups(new_block_id_groups)
                for new_blocks in new_block_id_groups:
                    for bid in new_blocks:
                        if bid != 0:
                            self._current_batch_allocated_block_ids.add(bid)

        # Zero out stale block_ids in sliding window groups' pending-store
        # positions. Only sliding window groups can have stale entries (blocks
        # freed by remove_skipped_blocks then reallocated). Only positions in
        # [next_stored_block_idx * bsf, end) need checking where end is the
        # pre-extend length: earlier positions were already offloaded, later
        # ones are fresh allocations from this step.
        if self._sliding_window_groups and self._current_batch_allocated_block_ids:
            block_size_factor = self.config.block_size_factor
            for req_id, req_status in self._req_status.items():
                ends = new_block_ids_end.get(req_id)
                for i, grp_idx in enumerate(self._sliding_window_groups):
                    group_state = req_status.group_states[grp_idx]
                    start = group_state.next_stored_block_idx * block_size_factor
                    end = ends[i] if ends is not None else len(group_state.block_ids)
                    for j in range(start, end):
                        if (
                            group_state.block_ids[j]
                            in self._current_batch_allocated_block_ids
                        ):
                            group_state.block_ids[j] = 0

    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        block_size_factor = self.config.block_size_factor
        store_jobs: dict[int, TransferJob] = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            # KV produced after a random-omit load is conditioned on an
            # approximate history. Publishing it under the original exact
            # request hashes would silently poison later prefix-cache hits.
            if req_status.selective_load_plan is not None:
                if _FORCED_OFFLOAD_DETAILED:
                    logger.info(
                        "FORCED_OFFLOAD_STORE_SKIP request=%s "
                        "reason=selective_load_plan",
                        req_id,
                    )
                continue
            req = req_status.req

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens
            # with async scheduling, some tokens may be missing
            num_offloadable_tokens = min(num_tokens_after_batch, req.num_tokens)
            max_offload_tokens = req_status.max_offload_tokens
            if max_offload_tokens is not None:
                num_offloadable_tokens = min(num_offloadable_tokens, max_offload_tokens)

            # Skip decode-phase blocks: clamp to the prompt length so only
            # prefill (prompt) blocks become eligible for store. next_stored_idx
            # never advances past this boundary, so decode blocks are never
            # queued in this or any later step.
            if self.config.offload_prompt_only:
                num_offloadable_tokens = min(
                    num_offloadable_tokens, req.num_prompt_tokens
                )

            # Filter out blocks skipped due to sliding window attention / SSM
            # or unreachable by the load path's alignment constraints.
            new_offload_keys: list[OffloadKey] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                if group_config.is_eagle_group:
                    num_blocks = max(0, num_blocks - 1)

                start_block_idx = group_state.next_stored_block_idx
                if num_blocks <= start_block_idx:
                    continue
                offload_keys = group_state.offload_keys[start_block_idx:num_blocks]
                # For each block to offload, take the last corresponding GPU block.
                # e.g. if block size factor is 3 and GPU block IDs are
                # 1 5 6 7 2 4 9 3 8 then we'll take blocks 6 4 8.
                # A block_id of 0 means either a sliding window / SSM skip
                # or a stale entry that was zeroed out — skip it either way.
                offload_block_ids = group_state.block_ids[
                    start_block_idx * block_size_factor
                    + block_size_factor
                    - 1 : num_blocks * block_size_factor : block_size_factor
                ]
                assert len(offload_keys) == len(offload_block_ids)

                alignment_block_count = group_config.alignment_block_count
                tail = group_config.sliding_window_size_in_blocks

                for key_idx, (offload_key, block_id) in enumerate(
                    zip(offload_keys, offload_block_ids)
                ):
                    if block_id == 0:
                        continue
                    # Skip SWA blocks that can never serve a load hit:
                    # within each full-attention alignment segment, only the
                    # trailing `tail` blocks are reachable by
                    # _sliding_window_lookup. For DeepSeek V4 with 100K
                    # tokens this reduces SWA stores by ~78%.
                    if alignment_block_count is not None:
                        assert tail is not None
                        abs_block_idx = start_block_idx + key_idx
                        pos_in_segment = abs_block_idx % alignment_block_count
                        if pos_in_segment < alignment_block_count - tail:
                            continue
                    new_offload_keys.append(offload_key)

            if not new_offload_keys:
                if _FORCED_OFFLOAD_DETAILED:
                    logger.info(
                        "FORCED_OFFLOAD_STORE_PREPARE request=%s candidate_keys=0 "
                        "result=no_eligible_keys",
                        req_id,
                    )
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            store_output = self.manager.prepare_store(
                new_offload_keys, req_status.req_context
            )
            if store_output is None:
                logger.info(
                    "FORCED_OFFLOAD_STORE_PREPARE request=%s candidate_keys=%d "
                    "result=rejected",
                    req_id,
                    len(new_offload_keys),
                )
                logger.warning("Request %s: cannot store blocks", req_id)
                continue

            logger.info(
                "FORCED_OFFLOAD_STORE_PREPARE request=%s candidate_keys=%d "
                "keys_to_store=%d evicted_keys=%d",
                req_id,
                len(new_offload_keys),
                len(store_output.keys_to_store),
                len(store_output.evicted_keys),
            )

            if not store_output.keys_to_store:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            self._touch(req_status)

            keys_to_store = set(store_output.keys_to_store)

            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            sliding_window_block_ids: list[int] = []
            non_sliding_window_block_ids: list[int] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                is_sliding_window = (
                    group_config.sliding_window_size_in_blocks is not None
                )
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                start_block_idx = group_state.next_stored_block_idx
                block_ids = group_state.block_ids
                num_group_blocks = 0
                start_gpu_block_idx: int | None = None
                for idx, offload_key in enumerate(
                    group_state.offload_keys[start_block_idx:num_blocks]
                ):
                    if offload_key not in keys_to_store:
                        continue

                    offloaded_block_idx = start_block_idx + idx

                    self._events_tracker.record_store(
                        req, group_config, offloaded_block_idx, offload_key
                    )

                    gpu_block_idx = offloaded_block_idx * block_size_factor
                    for i in range(block_size_factor):
                        block_id = block_ids[gpu_block_idx + i]
                        if block_id == 0:
                            continue
                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        num_group_blocks += 1
                        if is_sliding_window:
                            sliding_window_block_ids.append(block_id)
                        else:
                            non_sliding_window_block_ids.append(block_id)

                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)
                group_state.next_stored_block_idx = num_blocks

            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=group_sizes, block_indices=block_indices
            )
            dst_spec = store_output.store_spec

            job_id = self._generate_job_id()
            # a store can only be issued when no load is pending.
            if req_status.transfer_jobs:
                any_jid = next(iter(req_status.transfer_jobs))
                assert self._jobs[any_jid].is_store
            req_status.transfer_jobs.add(job_id)

            # Watch sliding window blocks as they may get evicted
            # before the request finishes
            for bid in sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

            # the non-sliding window blocks will be watched only
            # when the request finishes
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(keys_to_store),
                is_store=True,
                non_sliding_window_block_ids=non_sliding_window_block_ids,
                sliding_window_block_ids=sliding_window_block_ids or None,
            )

            store_jobs[job_id] = TransferJob(
                req_id=req_id, transfer_spec=(src_spec, dst_spec)
            )

            logger.debug(
                "Request %s offloading %s blocks upto %d tokens (job %d)",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
                job_id,
            )

        return store_jobs

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        self._update_req_states(scheduler_output)
        self.manager.on_schedule_end()

        # Flush jobs for preempted requests.
        for req_id in scheduler_output.preempted_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None or not req_status.transfer_jobs:
                continue
            any_jid = next(iter(req_status.transfer_jobs))
            assert self._jobs[any_jid].is_store
            self._current_batch_jobs_to_flush.update(req_status.transfer_jobs)

        # Flush jobs that contain re-allocated blocks.
        if (
            self._block_id_to_pending_jobs
            and not self._block_id_to_pending_jobs.keys().isdisjoint(
                self._current_batch_allocated_block_ids
            )
        ):
            self._current_batch_jobs_to_flush.update(
                jid
                for bid in self._current_batch_allocated_block_ids
                if bid in self._block_id_to_pending_jobs
                for jid in self._block_id_to_pending_jobs[bid]
            )

        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=self._build_store_jobs(scheduler_output),
            jobs_to_flush=self._current_batch_jobs_to_flush,
            selective_load_plans={
                req_id: req_status.selective_load_plan
                for req_id in scheduler_output.num_scheduled_tokens
                if (req_status := self._req_status.get(req_id)) is not None
                and req_status.selective_load_plan is not None
            },
        )
        self._current_batch_load_jobs = {}
        self._current_batch_jobs_to_flush = set()
        self._current_batch_allocated_block_ids = set()
        if self._remote_ra_decoupled and (
            meta.load_jobs or meta.store_jobs or meta.jobs_to_flush
        ):
            raise RuntimeError(
                "decoupled KV layout: offloading jobs are unsupported "
                "(the connector must stay isolated)"
            )
        return meta

    def has_pending_push_work(self) -> bool:
        """Whether the engine must keep stepping.

        While True, build_connector_meta() and update_connector_output()
        continue to be called even when no requests are scheduled.
        """
        # Preserve the original short-circuit behavior exactly. The manager is
        # queried only when scheduler-owned jobs are empty.
        if self._jobs:
            manager_pending = False
            has_pending = True
        else:
            manager_pending = self.manager.has_pending_work()
            has_pending = manager_pending
        signature = (
            len(self._jobs),
            manager_pending,
            len(self._req_status),
            len(self._blocks_being_loaded or ()),
        )
        if signature != self._forced_offload_last_pending_signature:
            logger.info(
                "FORCED_OFFLOAD_PENDING_STATE jobs=%d manager_pending=%s "
                "request_states=%d blocks_being_loaded=%d",
                *signature,
            )
            self._forced_offload_last_pending_signature = signature
        return has_pending

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
        if not meta.transfer_stats.is_empty():
            transfer_stats = OffloadingConnectorStats()
            if not meta.transfer_stats.load.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_BYTES,
                    meta.transfer_stats.load.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_TIME,
                    meta.transfer_stats.load.time,
                )
                for size in meta.transfer_stats.load.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.LOAD_SIZE, size
                    )
            if not meta.transfer_stats.store.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_BYTES,
                    meta.transfer_stats.store.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_TIME,
                    meta.transfer_stats.store.time,
                )
                for size in meta.transfer_stats.store.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.STORE_SIZE, size
                    )
            if self._connector_stats is None:
                self._connector_stats = transfer_stats
            else:
                self._connector_stats.aggregate(transfer_stats)

        for job_id, count in meta.completed_jobs.items():
            assert count > 0
            if job_id < self._stale_job_threshold:
                logger.debug(
                    "Skipping stale completed job %d (pre-reset counter: %d)",
                    job_id,
                    self._stale_job_threshold,
                )
                continue
            job_status = self._jobs[job_id]
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0

            req_status = self._req_status[job_status.req_id]
            if job_status.is_store:
                self.manager.complete_store(job_status.keys, req_status.req_context)
            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                if self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(job_status.keys)
            logger.info(
                "FORCED_OFFLOAD_TRANSFER_COMPLETE request=%s job_id=%d "
                "direction=%s keys=%d remaining_jobs_before_cleanup=%d",
                job_status.req_id,
                job_id,
                "store" if job_status.is_store else "load",
                len(job_status.keys),
                len(self._jobs),
            )
            if self._block_id_to_pending_jobs:
                # Sliding window blocks are tracked from store creation
                # and must be cleaned up unconditionally.
                self._remove_pending_job(job_id, job_status.sliding_window_block_ids)
                # Non-sliding-window blocks are only tracked after
                # request_finished, so only clean up for finished requests.
                if req_status.req.is_finished():
                    self._remove_pending_job(
                        job_id, job_status.non_sliding_window_block_ids
                    )

            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if not req_status.transfer_jobs and req_status.req.is_finished():
                # Deferred from request_finished: the request's last in-flight
                # job is now done, so fire the finalize hook here, after the
                # final complete_store/complete_load above (and any submit_store
                # the complete_store cascade issued).
                self.manager.on_request_finished(req_status.req_context)
                del self._req_status[job_status.req_id]

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = self._connector_stats
        self._connector_stats = None

        manager_stats = self.manager.get_stats()
        if manager_stats is not None:
            if stats is None:
                stats = manager_stats
            else:
                stats.aggregate(manager_stats)

        return stats

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        coordinator = getattr(self, "_policy_runtime", None)
        if coordinator is not None:
            if request.status.name in ("FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED"):
                coordinator.finish_request(request.request_id)
            else:
                coordinator.cancel_request(request.request_id, "request_aborted")
            self._policy_runtime_handles.pop(request.request_id, None)
            retire = getattr(self._policy_runtime_context, "retire", None)
            if retire is not None:
                retire(request.request_id)
        # TODO(orozery): possibly kickoff offload for last block
        # which may have been deferred due to async scheduling
        req_status = self._req_status.get(request.request_id)

        if req_status is None:
            # Untracked request (offloading never started): no in-flight jobs,
            # nothing was deferred, so finalize immediately.
            self.manager.on_request_finished(_create_req_context(request))
            return False, None

        if not req_status.transfer_jobs:
            # No in-flight jobs: all per-request calls are done, finalize now.
            self.manager.on_request_finished(req_status.req_context)
            del self._req_status[request.request_id]
            return False, None

        # In-flight jobs remain, so defer on_request_finished to
        # update_connector_output, which fires it once the last job completes
        # (after the final complete_store and any cascade submit_store it
        # issues). These pending stores outlive the request's block ownership;
        # register them so future reuse of those blocks triggers a flush.
        for job_id in req_status.transfer_jobs:
            job_status = self._jobs[job_id]
            for bid in job_status.non_sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Drain pending KV cache events.

        Complete metadata is available only when self-describing KV events
        are enabled, and only for full-attention groups. Other shapes retain
        the previous placeholder payload so consumers can ignore them.

        Yields:
            ``BlockStored`` or ``BlockRemoved`` events corresponding to
            the underlying :class:`OffloadingEvent` stream.
        """
        yield from self._events_tracker.take_events(self.manager.take_events())

    def reset_cache(self) -> None:
        """Reset the offloading manager cache, evicting all stored blocks."""

        # reset_cache cannot be called in the middle of a schedule step
        assert not self._current_batch_load_jobs
        assert not self._current_batch_jobs_to_flush
        assert not self._current_batch_allocated_block_ids

        # Flush all in-flight jobs
        self._current_batch_jobs_to_flush.update(self._jobs.keys())

        # A finished request may still be tracked here with in-flight jobs that
        # this reset discards, so its deferred on_request_finished() would never
        # fire (completions are skipped as stale) and its _req_status entry would
        # leak. Finalize such requests now, before resetting the manager.
        # list() snapshots because we delete while iterating.
        for req_id, status in list(self._req_status.items()):
            if status.req.is_finished():
                self.manager.on_request_finished(status.req_context)
                del self._req_status[req_id]

        # Reset offloading manager cache
        self.manager.reset_cache()

        # Reset store progress so active requests re-offload from block 0
        for status in self._req_status.values():
            for group_state in status.group_states:
                group_state.next_stored_block_idx = 0

        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        self._jobs.clear()
        self._block_id_to_pending_jobs.clear()

        # The manager pool is empty; pending event payloads and announced
        # reference counts are stale.
        self._events_tracker.reset()

        # Note: _current_batch_jobs_to_flush is intentionally NOT cleared.
        # The load flush IDs collected above must be delivered to workers.
        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.clear()

    def shutdown(self) -> None:
        self.manager.shutdown()
