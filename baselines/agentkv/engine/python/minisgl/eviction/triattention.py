from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from minisgl.eviction.base import LayerEvictionView


# Filenames of the official TriAttention calibration stats, keyed by model layer count
# (36 = Qwen3-8B, 64 = Qwen3-32B). These files are third-party artifacts and are NOT
# bundled with this repo. Obtain them from the official TriAttention repository, then
# either set MINISGL_TRIATTN_STATS_PATH, pass an explicit path in the plan string, or drop
# the files (named as below) under `_STATS_ASSET_DIR`. See eviction/triattention_stats/README.md.
_DEFAULT_STATS_FILENAMES = {
    36: "qwen3_8b.pt",
    64: "qwen3_32b_int4_stats.pt",
}
_STATS_ASSET_DIR = Path(__file__).resolve().parent / "triattention_stats"


@dataclass(frozen=True)
class TriAttentionEvictionPlan:
    name: str = "triattn"
    budget: int = 2048
    window_tokens: int = 0
    score_aggregation: str = "mean"
    offset_max_length: int = 65536
    normalize_scores: bool = True
    stats_path: str | None = None
    requires_page_size_one: bool = True
    requires_logical_positions: bool = False
    requires_kv_view: bool = True
    requires_query_observation: bool = False
    requires_kv_phase_ids: bool = False

    def select_layer(self, view: LayerEvictionView) -> torch.Tensor:
        identity = _identity_indices(view)
        if view.physical_cached_len <= self.budget:
            return identity
        if view.k is None:
            return identity
        self._validate_parameters()

        window = min(self.window_tokens, self.budget, view.physical_cached_len)
        dynamic_keep = self.budget - window
        dynamic_len = view.physical_cached_len - window
        if dynamic_keep <= 0:
            recent = torch.arange(
                view.physical_cached_len - self.budget,
                view.physical_cached_len,
                dtype=torch.long,
                device=view.device,
            )
            return recent.unsqueeze(0).expand(view.num_kv_heads, -1)
        if dynamic_len <= dynamic_keep:
            return identity

        bundle = _load_stats_bundle(
            stats_path=self._stats_path_for_view(view),
            device=view.device,
            num_model_layers=view.num_layers,
            num_kv_heads=view.num_kv_heads,
            head_dim=view.k.shape[-1],
        )
        scores = _score_layer(
            k=view.k[:dynamic_len],
            layer_id=view.layer_id,
            bundle=bundle,
            num_kv_heads=view.num_kv_heads,
            round_start=view.logical_current_position or view.physical_cached_len,
            offset_max_length=self.offset_max_length,
            score_aggregation=self.score_aggregation,
        )
        if self.normalize_scores:
            scores = _normalize_scores(scores)
        selected = torch.topk(scores, k=dynamic_keep, dim=-1).indices
        if window > 0:
            recent = torch.arange(
                view.physical_cached_len - window,
                view.physical_cached_len,
                dtype=torch.long,
                device=view.device,
            )
            recent = recent.unsqueeze(0).expand(view.num_kv_heads, -1)
            keep = torch.cat([selected, recent], dim=1)
        else:
            keep = selected
        return torch.sort(keep, dim=1).values

    def _stats_path_for_view(self, view: LayerEvictionView) -> Path:
        if self.stats_path:
            return Path(self.stats_path)
        env_path = os.environ.get("MINISGL_TRIATTN_STATS_PATH") or os.environ.get(
            "TRIATTN_RUNTIME_SPARSE_STATS_PATH"
        )
        if env_path:
            return Path(env_path)
        fname = _DEFAULT_STATS_FILENAMES.get(view.num_layers)
        if fname is not None:
            candidate = _STATS_ASSET_DIR / fname
            if candidate.exists():
                return candidate
        raise RuntimeError(
            "TriAttention requires offline calibration stats for this model, which are NOT "
            "bundled with this repository. Download the official TriAttention stats "
            f"(expected filename for a {view.num_layers}-layer model: "
            f"{_DEFAULT_STATS_FILENAMES.get(view.num_layers, '<model>.pt')}) from the official "
            "TriAttention repo, then either set MINISGL_TRIATTN_STATS_PATH=/path/to/stats.pt, "
            f"or place the file under {_STATS_ASSET_DIR}/. See eviction/triattention_stats/README.md."
        )

    def _validate_parameters(self) -> None:
        if self.budget <= 0:
            raise RuntimeError("TriAttention budget must be positive")
        if self.window_tokens < 0:
            raise RuntimeError("TriAttention window_tokens must be non-negative")
        if self.window_tokens >= self.budget:
            raise RuntimeError("TriAttention window_tokens must be less than budget")
        if self.score_aggregation not in {"mean", "max"}:
            raise RuntimeError("TriAttention score_aggregation must be 'mean' or 'max'")
        if self.offset_max_length < 1:
            raise RuntimeError("TriAttention offset_max_length must be positive")


@dataclass(frozen=True)
class _StatsBundle:
    metadata: dict[str, Any]
    head_stats: dict[int, dict[str, torch.Tensor]]
    omega: torch.Tensor
    num_attention_heads: int
    gqa_group_size: int


_STATS_CACHE: dict[tuple[str, str, int, int, int], _StatsBundle] = {}


def _load_stats_bundle(
    *,
    stats_path: Path,
    device: torch.device,
    num_model_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> _StatsBundle:
    cache_key = (str(stats_path), str(device), num_model_layers, num_kv_heads, head_dim)
    cached = _STATS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not stats_path.exists():
        raise FileNotFoundError(f"TriAttention stats file not found: {stats_path}")

    payload = torch.load(stats_path, map_location="cpu")
    raw_stats = payload.get("stats", {})
    metadata = _metadata_with_stats_path_defaults(
        dict(payload.get("metadata", {})),
        stats_path,
    )
    layer_ids: set[int] = set()
    head_ids: set[int] = set()
    for key in raw_stats:
        if key.startswith("layer") and "_head" in key:
            layer_part, head_part = key.split("_", maxsplit=1)
            layer_ids.add(int(layer_part.removeprefix("layer")))
            head_ids.add(int(head_part.removeprefix("head")))
    if not layer_ids or not head_ids:
        raise RuntimeError("TriAttention stats must use layerXX_headYY entries")

    num_layers = max(layer_ids) + 1
    num_attention_heads = max(head_ids) + 1
    if num_layers != num_model_layers:
        raise RuntimeError(
            f"TriAttention stats layer mismatch: stats={num_layers}, model={num_model_layers}"
        )
    if num_attention_heads % num_kv_heads != 0:
        raise RuntimeError(
            f"TriAttention stats heads ({num_attention_heads}) must be a multiple of "
            f"model KV heads ({num_kv_heads})"
        )
    freq_count = head_dim // 2
    if int(metadata.get("head_dim", head_dim)) != head_dim:
        raise RuntimeError(
            f"TriAttention stats head_dim mismatch: stats={metadata.get('head_dim')}, "
            f"model={head_dim}"
        )

    omega = _rope_frequencies(metadata, freq_count, device)
    head_stats: dict[int, dict[str, torch.Tensor]] = {}
    for layer_id in sorted(layer_ids):
        q_mean_real = []
        q_mean_imag = []
        q_abs_mean = []
        for head_id in sorted(head_ids):
            entry = raw_stats.get(f"layer{layer_id:02d}_head{head_id:02d}")
            if entry is None:
                raise RuntimeError(
                    f"TriAttention stats missing layer{layer_id:02d}_head{head_id:02d}"
                )
            q_mean_real.append(_stat_vector(entry, "q_mean_real", freq_count, device))
            q_mean_imag.append(_stat_vector(entry, "q_mean_imag", freq_count, device))
            if "q_abs_mean" in entry:
                q_abs_mean.append(_stat_vector(entry, "q_abs_mean", freq_count, device))
            else:
                q_abs_mean.append(torch.ones(freq_count, dtype=torch.float32, device=device))
        head_stats[layer_id] = {
            "q_mean_real": torch.stack(q_mean_real, dim=0),
            "q_mean_imag": torch.stack(q_mean_imag, dim=0),
            "q_abs_mean": torch.stack(q_abs_mean, dim=0),
            "freq_scale_sq": torch.ones(freq_count, dtype=torch.float32, device=device),
        }

    bundle = _StatsBundle(
        metadata=metadata,
        head_stats=head_stats,
        omega=omega,
        num_attention_heads=num_attention_heads,
        gqa_group_size=num_attention_heads // num_kv_heads,
    )
    _STATS_CACHE[cache_key] = bundle
    return bundle


def _stat_vector(
    entry: dict[str, torch.Tensor],
    key: str,
    freq_count: int,
    device: torch.device,
) -> torch.Tensor:
    value = entry[key].to(device=device, dtype=torch.float32)
    if value.numel() != freq_count:
        raise RuntimeError(
            f"TriAttention stat {key} has {value.numel()} frequencies; expected {freq_count}"
        )
    return value.contiguous()


def _metadata_with_stats_path_defaults(
    metadata: dict[str, Any],
    stats_path: Path,
) -> dict[str, Any]:
    if "inv_freq" in metadata or "rope_theta" in metadata:
        return metadata
    model_name = str(metadata.get("model_name", "")).lower()
    stats_name = str(stats_path).lower()
    if "qwen3" in model_name or "qwen3" in stats_name:
        metadata["rope_theta"] = 1000000.0
    return metadata


def _rope_frequencies(
    metadata: dict[str, Any],
    freq_count: int,
    device: torch.device,
) -> torch.Tensor:
    inv_freq = metadata.get("inv_freq")
    if isinstance(inv_freq, torch.Tensor):
        return inv_freq.to(device=device, dtype=torch.float32)[:freq_count].contiguous()
    model_name = str(metadata.get("model_name", ""))
    rope_theta = float(metadata.get("rope_theta", 1000000.0 if "Qwen3" in model_name else 10000.0))
    return 1.0 / (
        rope_theta
        ** (torch.arange(0, freq_count * 2, 2, device=device, dtype=torch.float32) / (freq_count * 2))
    )


def _score_layer(
    *,
    k: torch.Tensor,
    layer_id: int,
    bundle: _StatsBundle,
    num_kv_heads: int,
    round_start: int,
    offset_max_length: int,
    score_aggregation: str,
) -> torch.Tensor:
    layer_stats = bundle.head_stats.get(layer_id)
    if layer_stats is None:
        raise RuntimeError(f"TriAttention stats missing layer {layer_id}")
    if k.ndim != 3:
        raise RuntimeError(f"TriAttention expected K view [tokens, heads, dim], got {k.shape}")
    if k.shape[1] != num_kv_heads:
        raise RuntimeError(
            f"TriAttention K head mismatch: k={k.shape[1]}, view={num_kv_heads}"
        )

    offsets = _scoring_offsets(offset_max_length, k.device)
    scores_by_kv_head = []
    gqa = bundle.gqa_group_size
    for kv_head in range(num_kv_heads):
        start = kv_head * gqa
        end = start + gqa
        scores_by_q_head = _score_kv_head(
            k_head=k[:, kv_head, :],
            q_mean_real=layer_stats["q_mean_real"][start:end],
            q_mean_imag=layer_stats["q_mean_imag"][start:end],
            q_abs_mean=layer_stats["q_abs_mean"][start:end],
            freq_scale_sq=layer_stats["freq_scale_sq"],
            omega=bundle.omega,
            offsets=offsets,
            round_start=round_start,
            score_aggregation=score_aggregation,
        )
        scores_by_kv_head.append(scores_by_q_head.max(dim=0).values)
    return torch.stack(scores_by_kv_head, dim=0)


def _score_kv_head(
    *,
    k_head: torch.Tensor,
    q_mean_real: torch.Tensor,
    q_mean_imag: torch.Tensor,
    q_abs_mean: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    omega: torch.Tensor,
    offsets: torch.Tensor,
    round_start: int,
    score_aggregation: str,
) -> torch.Tensor:
    half_dim = k_head.shape[-1] // 2
    k_float = k_head.to(dtype=torch.float32)
    k_real = k_float[:, :half_dim]
    k_imag = k_float[:, half_dim:]
    k_abs = torch.sqrt(k_real.square() + k_imag.square())

    prod_real = q_mean_real[:, None, :] * k_real[None, :, :] + q_mean_imag[:, None, :] * k_imag[None, :, :]
    prod_imag = q_mean_imag[:, None, :] * k_real[None, :, :] - q_mean_real[:, None, :] * k_imag[None, :, :]
    phase = (float(round_start) + offsets).unsqueeze(1) * omega.unsqueeze(0)
    cos_vals = torch.cos(phase)
    sin_vals = torch.sin(phase)
    weighted_real = prod_real * freq_scale_sq.view(1, 1, -1)
    weighted_imag = prod_imag * freq_scale_sq.view(1, 1, -1)
    offset_scores = torch.einsum("htf,of->hto", weighted_real, cos_vals) - torch.einsum(
        "htf,of->hto", weighted_imag, sin_vals
    )
    if score_aggregation == "max":
        scores = offset_scores.max(dim=-1).values
    else:
        scores = offset_scores.mean(dim=-1)

    q_mean_abs = torch.sqrt(q_mean_real.square() + q_mean_imag.square() + 1e-8)
    extra_coef = q_abs_mean - q_mean_abs
    extra = (k_abs[None, :, :] * extra_coef[:, None, :] * freq_scale_sq.view(1, 1, -1)).sum(dim=-1)
    return scores + extra


def _scoring_offsets(offset_max_length: int, device: torch.device) -> torch.Tensor:
    offsets = []
    value = 1
    while value <= offset_max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, dtype=torch.float32, device=device)


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    mean = scores.mean(dim=-1, keepdim=True)
    std = scores.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (scores - mean) / std


def _identity_indices(view: LayerEvictionView) -> torch.Tensor:
    keep = torch.arange(view.physical_cached_len, dtype=torch.long, device=view.device)
    return keep.unsqueeze(0).expand(view.num_kv_heads, -1)
