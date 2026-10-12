#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from minisgl.kvcache import compact_layer_kv_heads, compact_layer_kv_heads_triton
from minisgl.kvcache.base import BaseKVCachePool


@dataclass
class ProfileKVCache(BaseKVCachePool):
    k: torch.Tensor
    v: torch.Tensor

    @classmethod
    def create(
        cls,
        *,
        layers: int,
        pages: int,
        heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> ProfileKVCache:
        shape = (layers, pages, 1, heads, head_dim)
        k = torch.randn(shape, dtype=dtype, device=device)
        v = torch.randn(shape, dtype=dtype, device=device)
        return cls(k=k, v=v)

    def clone(self) -> ProfileKVCache:
        return ProfileKVCache(k=self.k.clone(), v=self.v.clone())

    def k_cache(self, index: int) -> torch.Tensor:
        return self.k[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self.v[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return self.k.device

    @property
    def dtype(self) -> torch.dtype:
        return self.k.dtype

    @property
    def num_layers(self) -> int:
        return self.k.shape[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile layer/head KV compaction kernels.")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--pages", type=int, default=32768)
    parser.add_argument("--cached-len", type=int, default=8192)
    parser.add_argument("--compact-len", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--block-dim", type=int, default=64)
    parser.add_argument(
        "--per-head-shift",
        action="store_true",
        help="Shift keep indices per head to exercise per-head source selections.",
    )
    parser.add_argument("--skip-torch", action="store_true")
    return parser.parse_args()


def make_keep_indices(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    if args.compact_len > args.cached_len:
        raise ValueError("--compact-len must be <= --cached-len")
    sink = min(4, args.compact_len)
    recent = args.compact_len - sink
    base = torch.cat(
        [
            torch.arange(sink, dtype=torch.long),
            torch.arange(args.cached_len - recent, args.cached_len, dtype=torch.long),
        ]
    )
    keep = base.unsqueeze(0).repeat(args.heads, 1)
    if args.per_head_shift and recent > 0:
        recent_start = sink
        span = args.cached_len - sink
        for head in range(args.heads):
            shifted = sink + ((keep[head, recent_start:] - sink + head) % span)
            keep[head, recent_start:] = shifted
    return keep.to(device)


def time_cuda(fn: Callable[[], None], *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiling script")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda")
    if args.pages < args.cached_len + args.compact_len:
        raise ValueError("--pages must be at least --cached-len + --compact-len")

    src_slots = torch.arange(args.cached_len, dtype=torch.int32, device=device)
    new_slots = torch.arange(
        args.pages - args.compact_len,
        args.pages,
        dtype=torch.int32,
        device=device,
    )
    page_table = torch.zeros((1, args.cached_len), dtype=torch.int32, device=device)
    page_table[0, : args.cached_len] = src_slots
    keep = make_keep_indices(args, device)

    base_cache = ProfileKVCache.create(
        layers=args.layers,
        pages=args.pages,
        heads=args.heads,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
    )
    triton_cache = base_cache.clone()

    def run_triton() -> None:
        for layer_id in range(args.layers):
            compact_layer_kv_heads_triton(
                kv_cache=triton_cache,
                page_table=page_table,
                table_idx=0,
                physical_cached_len=args.cached_len,
                layer_id=layer_id,
                keep_relative_indices_by_head=keep,
                new_slots=new_slots,
                block_tokens=args.block_tokens,
                block_dim=args.block_dim,
            )

    torch_ms = None
    if not args.skip_torch:
        torch_cache = base_cache.clone()

        def run_torch() -> None:
            for layer_id in range(args.layers):
                compact_layer_kv_heads(
                    kv_cache=torch_cache,
                    page_table=page_table,
                    table_idx=0,
                    physical_cached_len=args.cached_len,
                    layer_id=layer_id,
                    keep_relative_indices_by_head=keep,
                    new_slots=new_slots,
                )

        run_torch()
        run_triton()
        torch.cuda.synchronize()
        if not torch.equal(torch_cache.k, triton_cache.k) or not torch.equal(
            torch_cache.v, triton_cache.v
        ):
            raise RuntimeError("Triton output does not match torch reference")
        torch_ms = time_cuda(run_torch, warmup=args.warmup, repeat=args.repeat)

    triton_ms = time_cuda(run_triton, warmup=args.warmup, repeat=args.repeat)
    copied_bytes = (
        args.layers
        * args.heads
        * args.compact_len
        * args.head_dim
        * 2
        * torch.empty((), dtype=dtype).element_size()
    )
    triton_gbps = copied_bytes / triton_ms / 1e6

    print("KV compact profile")
    print(f"  layers={args.layers} heads={args.heads} head_dim={args.head_dim} dtype={args.dtype}")
    print(f"  cached_len={args.cached_len} compact_len={args.compact_len} pages={args.pages}")
    print(f"  per_head_shift={args.per_head_shift}")
    if torch_ms is not None:
        torch_gbps = copied_bytes / torch_ms / 1e6
        print(f"  torch_ms={torch_ms:.4f} approx_GBps={torch_gbps:.2f}")
    print(f"  triton_ms={triton_ms:.4f} approx_GBps={triton_gbps:.2f}")
    if torch_ms is not None:
        print(f"  speedup={torch_ms / triton_ms:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
