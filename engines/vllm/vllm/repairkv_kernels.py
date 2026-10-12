"""Copy post-RoPE Q during probes; no scoring or policy logic in this kernel."""
import triton as tr
import triton.language as tl


@tr.jit
def _copy(Q, T, N, stride: tl.constexpr, width: tl.constexpr,
          offset: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if row < tl.load(N):
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(Q + row * stride + cols, cols < width, other=0)
        tl.store(T + offset + row * width + cols, values, cols < width)


def copy_queries(q, target, count, layer):
    assert q.ndim == 2 and q.stride(1) == 1
    width = q.shape[1]
    _copy[(q.shape[0], tr.cdiv(width, 256))](q, target, count, q.stride(0),
        width, layer * target.shape[1] * width, 256)
