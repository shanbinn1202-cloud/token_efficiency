"""Storage/address transforms only. All IntentKV scoring stays upstream."""
import triton as tr
import triton.language as tl


@tr.jit
def _metadata(BT, Maps, Slots, Offsets, Seq, Cu, OutBT, OutSeq, OutCu,
              bt_stride: tl.constexpr, map_stride: tl.constexpr,
              HEADS: tl.constexpr, KVHEADS: tl.constexpr, PAGES: tl.constexpr,
              NATIVE: tl.constexpr, NREQ: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    req, head = row // HEADS, row % HEADS
    slot = tl.load(Slots + req)
    length = tl.load(Seq + req) - tl.load(Offsets + tl.maximum(slot, 0))
    length = tl.where(slot >= 0, length, 0)
    page = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(Maps + (tl.maximum(slot, 0)*HEADS+head)*PAGES+page,
                  (page < PAGES) & (slot >= 0), other=0)
    native_page = pos // NATIVE
    physical = tl.load(BT + req*bt_stride+native_page,
                       (page*16 < length) & (page < PAGES), other=0)
    fine = (physical*KVHEADS + head//(HEADS//KVHEADS))*NATIVE + pos % NATIVE
    tl.store(OutBT + row*PAGES+page, fine, page < PAGES)
    if tl.program_id(1) == 0:
        start = tl.load(Cu+req)
        end = tl.load(Cu+req+1)
        tl.store(OutCu+row, start*HEADS+head*(end-start))
        tl.store(OutSeq+row, length)
        if row == NREQ*HEADS-1:
            tl.store(OutCu+row+1, end*HEADS)


def metadata(block_table, maps, slots, offsets, seq, cu, out_table, out_seq, out_cu,
             heads, kv_heads, native_block):
    assert native_block % 16 == 0
    n = cu.numel()-1
    pages = maps.shape[-1]
    _metadata[(n*heads, tr.cdiv(pages, 256))](block_table, maps, slots, offsets, seq, cu,
        out_table, out_seq, out_cu, block_table.stride(0), maps.stride(0),
        heads, kv_heads, pages, native_block//16, n, 256)


@tr.jit
def _pack(Q, Packed, Cu, HEADS: tl.constexpr, DIM: tl.constexpr,
          Q0: tl.constexpr, Q1: tl.constexpr, MAXQ: tl.constexpr,
          REVERSE: tl.constexpr, BLOCK: tl.constexpr):
    req, head = tl.program_id(0), tl.program_id(1)
    x = tl.program_id(2)*BLOCK+tl.arange(0, BLOCK)
    token, col = x // DIM, x % DIM
    start, end = tl.load(Cu+req), tl.load(Cu+req+1)
    valid = token < end-start
    original = (start+token)*Q0+head*Q1+col
    packed = (start*HEADS+head*(end-start)+token)*DIM+col
    if REVERSE:
        value = tl.load(Packed+packed, valid, other=0)
        tl.store(Q+original, value, valid)
    else:
        value = tl.load(Q+original, valid, other=0)
        tl.store(Packed+packed, value, valid)


def pack(q, packed, cu, maxq, reverse=False):
    _pack[(cu.numel()-1, q.shape[1], tr.cdiv(maxq*q.shape[2], 256))](
        q, packed, cu, q.shape[1], q.shape[2], q.stride(0), q.stride(1), maxq, reverse, 256)
