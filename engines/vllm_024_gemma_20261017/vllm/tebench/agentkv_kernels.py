"""Layout-only Q-ring writer; selection and ring locations come from upstream."""

import triton as tr
import triton.language as tl


@tr.jit
def _write(
    Q,
    R,
    S,
    D,
    N,
    q_stride: tl.constexpr,
    width: tl.constexpr,
    layer_offset: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = row < tl.load(N)
    src = tl.load(S + row, active, other=0)
    dst = tl.load(D + row, active, other=0) + layer_offset
    val = tl.load(Q + src * q_stride + cols, active & (cols < width), other=0)
    tl.store(R + dst * width + cols, val, active & (cols < width))


def write_queries(q, ring, sources, destinations, count, layer, anchors):
    assert q.ndim == 2 and q.stride(1) == 1
    width = q.shape[1]
    _write[(sources.numel(), tr.cdiv(width, 256))](
        q, ring, sources, destinations, count, q.stride(0), width, layer * anchors, 256
    )
