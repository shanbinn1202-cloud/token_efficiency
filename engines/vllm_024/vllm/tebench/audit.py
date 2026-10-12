"""Opt-in live model oracle. Never enabled during benchmark timing.

Observe projections and attention output independently of the eviction writer.
Maintain dense per-head KV and per-stage Q histories, and check the live paged
cache, Q ring, original selector and attention against those observations.
"""

import json
import os
from pathlib import Path

import torch

ENABLED = os.environ.get("VLLM_AGENTKV_MODEL_AUDIT") == "1"
STATE = None


def _copy(src: torch.Tensor, dst: torch.Tensor) -> None:
    dst[: src.shape[0]].copy_(src.reshape(src.shape[0], -1))


def _fake(src, dst):
    pass


if ENABLED:
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="agentkv_audit_copy",
        op_func=_copy,
        mutates_args=["dst"],
        fake_impl=_fake,
    )


def initialize(config, device, layers):
    global STATE
    if ENABLED and STATE is None:
        STATE = Oracle(config, device, layers)


def capture(kind, tensor, layer):
    torch.ops.vllm.agentkv_audit_copy(tensor, STATE.buffers[kind][layer])


class Oracle:
    def __init__(self, config, device, layers):
        c = config.model_config.hf_text_config
        self.heads, self.kv_heads, self.dim = (
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
        )
        self.layer_map = {layer: i for i, layer in enumerate(layers)}
        n = max(8192, config.scheduler_config.max_num_batched_tokens)
        self.buffers = {
            kind: {
                layer: torch.empty(
                    n, heads * self.dim, dtype=config.model_config.dtype, device=device
                )
                for layer in layers
            }
            for kind, heads in [
                ("q", self.heads),
                ("k", self.kv_heads),
                ("v", self.kv_heads),
                ("o", self.heads),
            ]
        }
        self.cache, self.queries = {}, {}
        self.checks = self.compactions = self.batches = 0
        self.phase_counts = {"prefill": 0, "decode": 0}
        self.max_abs = self.max_relative_l2 = 0.0
        self.path = Path(os.environ["VLLM_AGENTKV_MODEL_AUDIT_PATH"])
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def observe(self, runner, scheduled, state):
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm.tebench.kv_session import OFFSETS, gather_kv
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        batch_phases = {
            "prefill"
            if runner.requests[rid].num_computed_tokens
            < len(runner.requests[rid].prompt_token_ids)
            else "decode"
            for rid, count in scheduled.num_scheduled_tokens.items()
            if count
        }
        for phase in batch_phases:
            self.phase_counts[phase] += 1
        live = set(state.sessions)
        for mapping in [self.cache, self.queries]:
            for key in list(mapping):
                if key[0] not in live:
                    del mapping[key]
        offset = 0
        for rid in runner.input_batch.req_ids:
            count = scheduled.num_scheduled_tokens[rid]
            req = runner.requests[rid]
            session = state.sessions[rid]
            if session.stage_finished:
                offset += count
                continue
            start = req.num_computed_tokens
            phases = session.phases[start : start + count]
            physical = start + count - OFFSETS.get(rid, 0)
            for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
                if type(group.kv_cache_spec) is not FullAttentionSpec:
                    continue
                for name in group.layer_names:
                    layer = extract_layer_index(name)
                    key = rid, layer
                    q, k, v, o = [
                        self.buffers[kind][layer][offset : offset + count].view(
                            count, heads, self.dim
                        )
                        for kind, heads in [
                            ("q", self.heads),
                            ("k", self.kv_heads),
                            ("v", self.kv_heads),
                            ("o", self.heads),
                        ]
                    ]
                    fresh = torch.cat([k, v], dim=-1)
                    dense = (
                        torch.cat([self.cache[key], fresh])
                        if key in self.cache
                        else fresh.clone()
                    )
                    assert dense.shape[0] == physical, (
                        "oracle length",
                        rid,
                        layer,
                        dense.shape,
                        physical,
                    )
                    self.cache[key] = dense
                    cache = runner.compilation_config.static_forward_context[
                        name
                    ].kv_cache
                    actual = gather_kv(
                        cache,
                        req.block_ids[gid],
                        group.kv_cache_spec.block_size,
                        physical,
                    )
                    assert torch.equal(actual, dense), (
                        "paged KV differs from model projections",
                        rid,
                        layer,
                        (actual - dense).abs().max().item(),
                    )
                    histories = self.queries.setdefault(
                        key, [[] for _ in range(state.plan.num_stages)]
                    )
                    for stage in range(state.plan.num_stages):
                        indices = [i for i, p in enumerate(phases) if p == stage]
                        histories[stage] = (
                            histories[stage]
                            + [
                                (start + i, q[i].clone())
                                for i in indices[-state.plan.query_ring_capacity :]
                            ]
                        )[-state.plan.query_ring_capacity :]
                    expected_by_position = dict(
                        item for history in histories for item in history
                    )
                    actual_q, positions = state.ring.get_unordered(
                        table_idx=session.slot,
                        layer_id=self.layer_map[layer],
                        stages=state.plan.query_ring_read_stages,
                    )
                    assert set(positions.tolist()) == set(expected_by_position), (
                        "Q positions",
                        rid,
                        layer,
                    )
                    expected_q = torch.stack(
                        [expected_by_position[p] for p in positions.tolist()]
                    )
                    assert torch.equal(actual_q, expected_q), (
                        "Q values",
                        rid,
                        layer,
                        (actual_q - expected_q).abs().max().item(),
                    )
                    # Last query in each request sees the whole physical prefix.
                    # FP32 dense softmax is independent of paged FA3 metadata.
                    q_last = (
                        q[-1]
                        .float()
                        .reshape(self.kv_heads, self.heads // self.kv_heads, self.dim)
                    )
                    logits = (
                        torch.einsum(
                            "hgd,thd->hgt", q_last, dense[..., : self.dim].float()
                        )
                        / self.dim**0.5
                    )
                    expected_o = torch.einsum(
                        "hgt,thd->hgd",
                        logits.softmax(-1),
                        dense[..., self.dim :].float(),
                    ).reshape(self.heads, self.dim)
                    diff = o[-1].float() - expected_o
                    absolute = diff.abs().max().item()
                    relative = (diff.norm() / expected_o.norm().clamp_min(1e-6)).item()
                    self.max_abs = max(self.max_abs, absolute)
                    self.max_relative_l2 = max(self.max_relative_l2, relative)
                    assert relative < 0.03, (
                        "attention output",
                        rid,
                        layer,
                        absolute,
                        relative,
                    )
                    self.checks += 1
            offset += count
        self.batches += 1
        if self.batches % 16 == 0:
            self.report()

    def compact(self, rid, layer, keep, state):
        from adapters.agentkv.upstream import LayerEvictionView

        key = rid, layer
        dense = self.cache[key]
        session = state.sessions[rid]
        _, pos = state.ring.get_unordered(
            table_idx=session.slot,
            layer_id=self.layer_map[layer],
            stages=state.plan.query_ring_read_stages,
        )
        history = dict(item for h in self.queries[key] for item in h)
        expected = state.plan.select_layer(
            LayerEvictionView(
                layer_id=self.layer_map[layer],
                physical_cached_len=len(dense),
                num_kv_heads=self.kv_heads,
                device=dense.device,
                k=dense[..., : self.dim],
                q_observation=torch.stack([history[p] for p in pos.tolist()]),
            )
        )
        assert torch.equal(keep, expected), ("original selector differs", rid, layer)
        self.cache[key] = torch.stack(
            [dense[keep[h], h] for h in range(self.kv_heads)], dim=1
        )
        self.compactions += 1

    def report(self):
        self.path.write_text(
            json.dumps(
                dict(
                    passed=True,
                    batches=self.batches,
                    phase_counts=self.phase_counts,
                    layer_request_checks=self.checks,
                    layer_compactions=self.compactions,
                    max_attention_absolute_error=self.max_abs,
                    max_attention_relative_l2_error=self.max_relative_l2,
                ),
                indent=2,
            )
            + "\n"
        )
