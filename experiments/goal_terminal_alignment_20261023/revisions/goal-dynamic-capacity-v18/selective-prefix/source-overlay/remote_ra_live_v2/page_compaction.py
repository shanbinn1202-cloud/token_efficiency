"""Batched Page-vector compaction where the hidden rows are (decoupled layout).

Legacy (P = 784, no layout key) never uses this module: the worker sends each
closing row as 5,120 Python floats (``query_selection.normalized``) and the
engine compacts them (``selection_contract.compact``).  At P = 16 a 6.5K
observation closes ~406 Pages, and that per-Page D2H + ``tolist`` + Python
normalisation costs ~1.5-5 ms each (x86 CPython).  Decoupled, the closing
rows of one forward are compacted together on their device and only the
compact vectors (128 int8 + one fp32 scale per Page) leave it:

    project (seeded Rademacher, 5,120 -> 128) -> L2 normalise -> int8 per-vector scale

which is ``selection_contract.compact`` / upstream ``_compact_last`` math.  The
projection and normalisation run in float64 and the scale is reported in fp32.
The result then does not depend on the batch shape or the device, except
where a float64 value sits within its rounding error of an int8 or fp32
rounding edge (~1e-9 per Page).  In fp32, a batched GEMM and a single-row GEMV
round differently: one element of ~0.2 % of vectors moves by 1 (synthetic
rows, x86 CPU).  The result is not bit-identical to today's float path either
(fp32 projection of the double-normalised row): one element of ~0.35 % of
vectors differs by 1.  That is why legacy keeps its own path.

Wire form (interface I3, ``representation = 'compact_int8_v1'``):
``page_vectors = {str(page): [[128 ints in -127..127], scale]}``.
"""
from __future__ import annotations

import math
from typing import NamedTuple

from .contract import require

REPRESENTATION = 'compact_int8_v1'
QUERY_REPRESENTATION = 'pre_final_norm_last_token_l2'
ACCUMULATE = 'float64'
DIMENSION = 128


class CompactPage(NamedTuple):
    """One Page vector: int8 values (as Python ints) and its positive scale."""
    values: tuple
    scale: float


def _valid(values, scale):
    return (type(values) in (tuple, list) and len(values) == DIMENSION and
            all(type(v) is int and -127 <= v <= 127 for v in values) and any(values) and
            type(scale) is float and math.isfinite(scale) and scale > 0)


def to_wire(pages):
    """{page: CompactPage} -> the JSON body form, pages ascending."""
    require(type(pages) is dict and all(type(p) is int and p >= 0 and type(v) is CompactPage
                                        and _valid(v.values, v.scale) for p, v in pages.items()),
            'compact Page vectors required')
    return {str(p): [list(v.values), v.scale] for p, v in sorted(pages.items())}


def from_wire(value):
    """The JSON body form -> {page: CompactPage}; refuses anything else."""
    require(type(value) is dict, 'compact Page vector map required')
    out = {}
    for key, item in value.items():
        require(type(key) is str and key.isdecimal() and type(item) in (list, tuple) and
                len(item) == 2 and _valid(item[0], item[1]), 'invalid compact Page vector')
        out[int(key)] = CompactPage(tuple(item[0]), item[1])
    return out


class PageCompactor:
    """Compacts hidden rows on their own device; one per worker, built at attach."""

    def __init__(self, hidden_size, page_tokens):
        import torch
        from . import selection_contract
        require(type(hidden_size) is int and hidden_size > 0 and
                type(page_tokens) is int and page_tokens > 0, 'compactor geometry required')
        # The projection depends only on the contract's seed and dimension.
        # Refuses a matrix that differs from the upstream vector space (5,120).
        matrix, digest = selection_contract.projection(hidden_size, selection_contract.default_contract())
        require(tuple(matrix.shape) == (DIMENSION, hidden_size), 'projection shape differs')
        self.torch, self.hidden_size, self.page_tokens = torch, hidden_size, page_tokens
        self.projection_sha256 = digest
        self._matrices = {str(matrix.device): matrix.to(torch.float64)}

    def evidence(self):
        return dict(representation=REPRESENTATION, projection_sha256=self.projection_sha256,
                    dimension=DIMENSION, accumulate=ACCUMULATE)

    def _matrix(self, device):
        key = str(device)
        matrix = self._matrices.get(key)
        if matrix is None:
            matrix = self._matrices[key] = next(iter(self._matrices.values())).to(device)
        return matrix

    def compact(self, rows):
        """rows: [k, hidden] tensor (any float dtype/device) -> k CompactPage, two small D2H."""
        torch = self.torch
        require(isinstance(rows, torch.Tensor) and rows.ndim == 2 and rows.shape[0] > 0 and
                rows.shape[1] == self.hidden_size, 'hidden rows [k, hidden] required')
        vectors = torch.nn.functional.linear(rows.detach().to(torch.float64),
                                             self._matrix(rows.device))
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=-1)
        scale = vectors.abs().amax(dim=-1) / 127.0
        quantized = torch.clamp(torch.round(vectors / scale.unsqueeze(-1)), -127, 127).to(torch.int8)
        # The reported scale is fp32: its float64 last bits follow the device's
        # summation order, its fp32 rounding does not (int8 + fp32 per Page).
        scales = scale.to(torch.float32).cpu().tolist()
        require(all(math.isfinite(s) and s > 0 for s in scales),
                'projection produced an invalid zero vector')
        return tuple(CompactPage(tuple(values), s)
                     for values, s in zip(quantized.cpu().tolist(), scales))

    def closing(self, hidden, start, pages):
        """Vectors of ``pages`` (consecutive) from rows [start, ...) of one forward.

        Page p closes at row (p + 1) * P - 1 - start; the rows are a strided
        view, so no index tensor is built or uploaded.
        """
        pages = tuple(pages)
        if not pages:
            return ()
        P, n = self.page_tokens, len(pages)
        require(list(pages) == list(range(pages[0], pages[0] + n)), 'consecutive closing Pages required')
        first = (pages[0] + 1) * P - 1 - start
        require(first >= 0 and first + (n - 1) * P < hidden.shape[0], 'closing rows outside the forward')
        return self.compact(hidden[first:first + (n - 1) * P + 1:P])
