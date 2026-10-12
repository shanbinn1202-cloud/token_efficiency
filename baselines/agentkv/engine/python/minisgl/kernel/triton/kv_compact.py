import triton
import triton.language as tl


@triton.jit
def compact_kv_heads_kernel(
    k_ptr,
    v_ptr,
    table_row_ptr,
    keep_ptr,
    new_slots_ptr,
    lengths_ptr,
    compact_len: tl.constexpr,
    head_dim: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_token: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_dim: tl.constexpr,
    keep_stride_head: tl.constexpr,
    keep_stride_token: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    head_id = tl.program_id(0)
    token_block_id = tl.program_id(1)
    dim_block_id = tl.program_id(2)

    token_offsets = token_block_id * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    dim_offsets = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    token_mask = token_offsets < compact_len
    dim_mask = dim_offsets < head_dim
    copy_mask = token_mask[:, None] & dim_mask[None, :]

    keep_offsets = head_id * keep_stride_head + token_offsets * keep_stride_token
    keep_relative_indices = tl.load(keep_ptr + keep_offsets, mask=token_mask, other=0).to(tl.int64)
    src_locs = tl.load(table_row_ptr + keep_relative_indices, mask=token_mask, other=0).to(
        tl.int64
    )
    dst_locs = tl.load(new_slots_ptr + token_offsets, mask=token_mask, other=0).to(tl.int64)

    k_src_offsets = (
        src_locs[:, None] * k_stride_token
        + head_id * k_stride_head
        + dim_offsets[None, :] * k_stride_dim
    )
    k_dst_offsets = (
        dst_locs[:, None] * k_stride_token
        + head_id * k_stride_head
        + dim_offsets[None, :] * k_stride_dim
    )
    v_src_offsets = (
        src_locs[:, None] * v_stride_token
        + head_id * v_stride_head
        + dim_offsets[None, :] * v_stride_dim
    )
    v_dst_offsets = (
        dst_locs[:, None] * v_stride_token
        + head_id * v_stride_head
        + dim_offsets[None, :] * v_stride_dim
    )

    k_values = tl.load(k_ptr + k_src_offsets, mask=copy_mask)
    v_values = tl.load(v_ptr + v_src_offsets, mask=copy_mask)
    tl.store(k_ptr + k_dst_offsets, k_values, mask=copy_mask)
    tl.store(v_ptr + v_dst_offsets, v_values, mask=copy_mask)

    if dim_block_id == 0 and token_block_id == 0:
        tl.store(lengths_ptr + head_id, compact_len)


@triton.jit
def compact_kv_layers_heads_kernel(
    k_ptr,
    v_ptr,
    table_row_ptr,
    keep_ptr,
    new_slots_ptr,
    compact_len: tl.constexpr,
    head_dim: tl.constexpr,
    k_stride_layer: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_layer: tl.constexpr,
    v_stride_token: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_dim: tl.constexpr,
    keep_stride_layer: tl.constexpr,
    keep_stride_head: tl.constexpr,
    keep_stride_token: tl.constexpr,
    NUM_DIM_BLOCKS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    layer_id = tl.program_id(0).to(tl.int64)
    head_id = tl.program_id(1).to(tl.int64)
    block_id = tl.program_id(2)
    token_block_id = block_id // NUM_DIM_BLOCKS
    dim_block_id = block_id - token_block_id * NUM_DIM_BLOCKS

    token_offsets = (token_block_id * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)).to(tl.int64)
    dim_offsets = (dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)).to(tl.int64)

    token_mask = token_offsets < compact_len
    dim_mask = dim_offsets < head_dim
    copy_mask = token_mask[:, None] & dim_mask[None, :]

    keep_offsets = (
        layer_id * keep_stride_layer
        + head_id * keep_stride_head
        + token_offsets * keep_stride_token
    )
    keep_relative_indices = tl.load(keep_ptr + keep_offsets, mask=token_mask, other=0).to(tl.int64)
    src_locs = tl.load(table_row_ptr + keep_relative_indices, mask=token_mask, other=0).to(
        tl.int64
    )
    dst_locs = tl.load(new_slots_ptr + token_offsets, mask=token_mask, other=0).to(tl.int64)

    k_src_offsets = (
        layer_id * k_stride_layer
        + src_locs[:, None] * k_stride_token
        + head_id * k_stride_head
        + dim_offsets[None, :] * k_stride_dim
    )
    k_dst_offsets = (
        layer_id * k_stride_layer
        + dst_locs[:, None] * k_stride_token
        + head_id * k_stride_head
        + dim_offsets[None, :] * k_stride_dim
    )
    v_src_offsets = (
        layer_id * v_stride_layer
        + src_locs[:, None] * v_stride_token
        + head_id * v_stride_head
        + dim_offsets[None, :] * v_stride_dim
    )
    v_dst_offsets = (
        layer_id * v_stride_layer
        + dst_locs[:, None] * v_stride_token
        + head_id * v_stride_head
        + dim_offsets[None, :] * v_stride_dim
    )

    k_values = tl.load(k_ptr + k_src_offsets, mask=copy_mask)
    v_values = tl.load(v_ptr + v_src_offsets, mask=copy_mask)
    tl.store(k_ptr + k_dst_offsets, k_values, mask=copy_mask)
    tl.store(v_ptr + v_dst_offsets, v_values, mask=copy_mask)


@triton.jit
def compact_kv_layers_heads_inplace_sorted_kernel(
    k_ptr,
    v_ptr,
    table_row_ptr,
    keep_ptr,
    compact_len: tl.constexpr,
    head_dim: tl.constexpr,
    k_stride_layer: tl.constexpr,
    k_stride_token: tl.constexpr,
    k_stride_head: tl.constexpr,
    k_stride_dim: tl.constexpr,
    v_stride_layer: tl.constexpr,
    v_stride_token: tl.constexpr,
    v_stride_head: tl.constexpr,
    v_stride_dim: tl.constexpr,
    keep_stride_layer: tl.constexpr,
    keep_stride_head: tl.constexpr,
    keep_stride_token: tl.constexpr,
    NUM_TOKEN_BLOCKS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    layer_id = tl.program_id(0).to(tl.int64)
    head_id = tl.program_id(1).to(tl.int64)
    dim_block_id = tl.program_id(2)
    dim_offsets = (dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)).to(tl.int64)
    dim_mask = dim_offsets < head_dim

    # Keep indices must be sorted. Copying destination blocks in ascending
    # order is then safe because every source token is at or after its
    # destination token and has not been overwritten yet.
    token_block_id = 0
    while token_block_id < NUM_TOKEN_BLOCKS:
        token_offsets = (token_block_id * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)).to(
            tl.int64
        )
        token_mask = token_offsets < compact_len
        copy_mask = token_mask[:, None] & dim_mask[None, :]

        keep_offsets = (
            layer_id * keep_stride_layer
            + head_id * keep_stride_head
            + token_offsets * keep_stride_token
        )
        keep_relative_indices = tl.load(keep_ptr + keep_offsets, mask=token_mask, other=0).to(
            tl.int64
        )
        src_locs = tl.load(table_row_ptr + keep_relative_indices, mask=token_mask, other=0).to(
            tl.int64
        )
        dst_locs = tl.load(table_row_ptr + token_offsets, mask=token_mask, other=0).to(tl.int64)

        k_src_offsets = (
            layer_id * k_stride_layer
            + src_locs[:, None] * k_stride_token
            + head_id * k_stride_head
            + dim_offsets[None, :] * k_stride_dim
        )
        k_dst_offsets = (
            layer_id * k_stride_layer
            + dst_locs[:, None] * k_stride_token
            + head_id * k_stride_head
            + dim_offsets[None, :] * k_stride_dim
        )
        v_src_offsets = (
            layer_id * v_stride_layer
            + src_locs[:, None] * v_stride_token
            + head_id * v_stride_head
            + dim_offsets[None, :] * v_stride_dim
        )
        v_dst_offsets = (
            layer_id * v_stride_layer
            + dst_locs[:, None] * v_stride_token
            + head_id * v_stride_head
            + dim_offsets[None, :] * v_stride_dim
        )

        k_values = tl.load(k_ptr + k_src_offsets, mask=copy_mask)
        v_values = tl.load(v_ptr + v_src_offsets, mask=copy_mask)
        tl.store(k_ptr + k_dst_offsets, k_values, mask=copy_mask)
        tl.store(v_ptr + v_dst_offsets, v_values, mask=copy_mask)
        token_block_id += 1
