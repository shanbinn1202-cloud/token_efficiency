"""Transformers tensor layout bridge to the original MiniSGL cache and sessions.

Only full-attention layers have paged KV. Hybrid recurrent states stay in the
session's Transformers cache. This adapter is single-session/single-sequence,
matching the benchmark's serialized model calls.
"""
from types import SimpleNamespace
from uuid import uuid4

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from .upstream import (CacheManager, EvictionManager, MHAKVCache,
    QueryRingBufferPool, Req, SamplingParams, SessionManager, UserMsg,
    get_tp_info, set_tp_info, try_get_tp_info)


class PagedLayer(DynamicLayer):
    """Expose temporary dense attention inputs; the page pool owns persistent KV."""
    def __init__(self, owner, layer_id):
        super().__init__()
        self.owner, self.layer_id = owner, layer_id

    def update(self, key_states, value_states, *args, **kwargs):
        owner, req = self.owner, self.owner.req
        if key_states.shape[0] != 1 or key_states.shape[-2] != req.extend_len:
            raise ValueError("AgentKV paged bridge requires one sequence and its full extend")
        # Native store kernel accepts token-major contiguous rows.
        owner.kv_cache.store_kv(
            key_states[0].transpose(0, 1).contiguous(),
            value_states[0].transpose(0, 1).contiguous(),
            owner.out_loc, self.layer_id,
        )
        self.is_initialized = True
        slots = owner.page_table[req.table_idx, :req.device_len].long()
        # HF SDPA consumes dense tensors. These views are not retained or used
        # for compaction; upstream EvictionManager reads the page pool directly.
        return tuple(pool(self.layer_id)[:, 0][slots].transpose(0, 1).unsqueeze(0)
                     for pool in (owner.kv_cache.k_cache, owner.kv_cache.v_cache))

    def get_seq_length(self):
        return self.owner.req.physical_cached_len if self.owner.req is not None else 0

    def get_mask_sizes(self, query_length):
        return self.get_seq_length() + query_length, 0


class PagedSession:
    def __init__(self, config, plan, *, max_window_size, eviction_interval,
                 device="cuda", dtype=torch.bfloat16):
        self.config, self.plan = config, plan
        self.device = torch.device(device)
        if try_get_tp_info() is None:
            set_tp_info(0, 1)
        if get_tp_info().size != 1:
            raise ValueError("Transformers AgentKV bridge requires tensor_parallel_size=1")
        self.layer_ids = [i for i, kind in enumerate(config.layer_types) if kind == "full_attention"]
        self.layer_map = {index: i for i, index in enumerate(self.layer_ids)}
        # Enough room for a legal uncompressed prompt and copy-to-new-page
        # compaction; budget is a retention target, not the pool's capacity.
        self.num_pages = max_window_size + plan.budget
        self.page_table = torch.empty((1, max_window_size + 1), dtype=torch.int32, device=device)
        self.token_pool = torch.empty_like(self.page_table)
        self.kv_cache = MHAKVCache(num_kv_heads=config.num_key_value_heads,
            num_layers=len(self.layer_ids), head_dim=config.head_dim,
            num_pages=self.num_pages, page_size=1, dtype=dtype, device=self.device)
        self.query_ring = QueryRingBufferPool(max_table_slots=1, num_layers=len(self.layer_ids),
            capacity=plan.query_ring_capacity, num_stages=plan.query_ring_num_stages,
            num_q_heads=config.num_attention_heads, head_dim=config.head_dim,
            dtype=dtype, device=self.device)
        self.manager_config = SimpleNamespace(eviction_plan=f"agentkv-{plan.budget}",
            page_size=1, compact_interval=eviction_interval, compact_after_prefill=False,
            compact_before_park=False, compact_debug=False, compact_debug_dir="",
            compact_debug_jsonl=False, compact_debug_max_items=0)
        self.reset()

    def reset(self):
        if hasattr(self, "sessions"):
            self.close()
        self.sessions = SessionManager(page_size=1)
        self.session_id = uuid4().hex
        self.req = None
        self.cache_manager = CacheManager(self.num_pages, 1, self.page_table)
        self.eviction = EvictionManager(config=self.manager_config, engine=self,
            cache_manager=self.cache_manager, token_pool=self.token_pool, device=self.device)
        self.eviction.plan = self.plan
        self.query_ring.reset(0)
        self.cache = DynamicCache(config=self.config)
        for index, layer_id in self.layer_map.items():
            self.cache.layers[index] = PagedLayer(self, layer_id)
        self.compactions = self.peak_physical = self.uid = 0

    def close(self):
        # A failed forward may own pages beyond Req.physical_cached_len. Reset
        # reconstructs the allocator only after all sessions have been removed.
        parked = self.sessions.close(self.session_id)
        if parked is not None:
            self.cache_manager.free_req_pages(parked)
            self.query_ring.reset(0)
            self.cache_manager.check_integrity()
        self.sessions.close_all_parked()
        self.req = None

    def begin(self, ids, stages, prompt, max_tokens, temperature, encode_suffix, *, semantic_stages=None):
        self.uid += 1
        msg = UserMsg(uid=self.uid, input_ids=torch.tensor(ids, dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=max_tokens, temperature=temperature),
            session_id=self.session_id, rendered_text=prompt,
            phase_ids=torch.tensor(stages, dtype=torch.int32),
            semantic_phase_ids=torch.tensor(semantic_stages if semantic_stages is not None else stages,
                                            dtype=torch.int32))
        pending = self.sessions.prepare_request(msg, encode_suffix=encode_suffix)
        try:
            self.req = Req(input_ids=pending.input_ids, table_idx=0,
                physical_cached_len=pending.physical_cached_len, output_len=max_tokens,
                uid=self.uid, sampling_params=pending.sampling_params,
                session_id=pending.session_id, session_text=pending.rendered_text,
                session_prompt_len=pending.session_prompt_len,
                full_input_ids=pending.full_input_ids, phase_ids=pending.phase_ids,
                full_phase_ids=pending.full_phase_ids, semantic_phase_ids=pending.semantic_phase_ids,
                retained_positions=pending.retained_positions,
                retained_positions_by_layer_head=pending.retained_positions_by_layer_head,
                logical_cached_len=pending.logical_cached_len,
                phase_text_tail=pending.phase_text_tail)
        except Exception:
            self.sessions.cancel_prepare(pending)
            raise
        return self.req

    def prepare_forward(self):
        req = self.req
        self.cache_manager.allocate_paged([req])
        self.out_loc = self.page_table[0, req.physical_cached_len:req.device_len]
        self.token_pool[0, req.physical_cached_len:req.device_len].copy_(
            req.input_ids[req.physical_cached_len:req.device_len])
        positions = torch.arange(req.logical_cached_len, req.logical_device_len,
                                 dtype=torch.int32, device=self.device)
        self.query_metadata = self.query_ring.prepare_write_metadata_from_reqs([req], positions,
            req.phase_ids[req.physical_cached_len:req.device_len].to(self.device))
        self.peak_physical = max(self.peak_physical, req.device_len)
        return positions

    def compact(self, *, finished):
        old_length = self.req.physical_cached_len
        error = self.eviction.maybe_compact(self.req, finished=finished)
        if error is not None:
            raise RuntimeError(error.error)
        self.compactions += self.req.physical_cached_len < old_length

    def park(self, *, drop_last_token, full_text):
        if self.sessions.finish_request(self.req, drop_last_token=drop_last_token, full_text=full_text):
            self.cache_manager.free_req_pages(self.req)
            self.query_ring.reset(0)
        self.cache_manager.check_integrity(owned_pages=self.sessions.owned_pages())
