# Decoupling full-attention Pages from the GDN state page (design, 2026-10-04)

Branch `exp-decouple-pages`, cut from `main` at `0f91d78` (runtime `902690a`).
Sources read: the overlay copies under `source-overlay/vllm/`. Files missing from `patches/v0.24.0.diff` are byte-identical upstream 0.24.0 "anchors". Also read: the runtime modules, `development/multi-session/capacity/CAPACITY.md`, `development/selection-contract-v7/stageB-*.md` and the PoC `reference/dram-backed-kv/scripts/serve_online_selector_v2.py`.
Measurement scripts are in `development/exp-decouple-pages/tools/` (system `python3`, x86 Windows, CPython; **not** GH200/Grace numbers).

## 0. Summary

**Decision.** Implement **Option A**. The full-attention (FA) KV and the Gated-DeltaNet (GDN, vLLM "Mamba") state get **two block pools with independent page sizes**:
- FA pool: `P`-token blocks. One FA block is still exactly one runtime Page, so the "one FA block per Page" invariant stays true.
- Mamba pool: one block holds one unpadded GDN layer state for each Mamba group, and is sized to a fixed number of GDN slots per server.

Group structure, the GDN kernels, the attention kernel and align mode stay as they are. The legacy shared pool (`P = 784`) remains the default code path, unchanged, and byte-identical.

**Page size.** Target `P = 16`.
- It is the PoC's `PAGE_SIZE`, so the Selector V2 contract becomes upstream-exact: budgets `floor(0.4·history)//16`, recent quota 64 Pages, a live tail under 16 tokens, and one vector per 16-token Page.
- It costs nothing in the attention kernel. With an fp32 GDN state, FlashAttention already runs with **16-token kernel blocks**: each 784-token block is split into 49 kernel blocks, so the kernel block table is already ~4,100 entries per request at 64K.
- What does not come for free: every runtime structure that is O(Pages) grows 49×. These have to be vectorized (section 5).
- `P` is a parameter in {16, 32, 64, 784}. P = 64 is the measured fallback if the host-side budget cannot be met at 16.

**GDN block size.** It is kept separate from `P` as `M` (the Mamba align block). Default `M = P·⌈784/P⌉`, which is 784 at P = 16.
- The GDN crossing cadence, captures and H2D accounting therefore stay exactly as today.
- `M = P = 16` is not viable in this runtime: each align-slot crossing costs a 154 MB D2H plus a 154 MB H2D (`qwen.py:246-252`, `worker_backend.relocate_latest_hybrid`).
- `M ≥ max_model_len` (a single GDN block per request, no crossings) is a follow-up (D5).

**Effort.** Four work packages, about 10–14 agent-days in total, roughly 4–5 days wall time with four parallel agents. After that come integration and 1–2 serial GPU smoke cycles.

## 1. How vLLM 0.24 lays out KV and GDN state today (verified in source unless marked)

### 1.1 Model geometry (Qwen3.6-27B, `remote_ra_linux/model-evidence/model-config.json`)
- 64 layers: 16 `full_attention` (every 4th) and 48 `linear_attention` (GDN).
- FA: 4 KV heads × 256 dim, bf16. That is **4,096 B per token per layer**, and 65,536 B per token over the 16 FA layers.
- GDN per layer:
  - conv state: (16·128·2 + 48·128) × 3 × bf16 = 61,440 B;
  - recurrent state: 48 × 128 × 128 × **fp32** = 3,145,728 B (`mamba_ssm_dtype: float32`);
  - total **3,207,168 B**, which equals 783 tokens of FA per layer.
  - Note: one fp32 head state is exactly 64 KiB, one 16-token FA page per layer.

### 1.2 Where 784 comes from (`platforms/interface.py:594-802`)
`update_block_size_for_backend` is called from `uniproc_executor.py:69` after the model loads. For hybrid models, `_align_hybrid_block_size` does three things:
1. Computes `attn_block_size = 16·cdiv(3,207,168, 16·4096) = 784` (`:766-769`) and raises `cache_config.block_size` to it (`:771-777`). This overrides a user `--block-size`.
2. In align mode, sets `mamba_block_size = block_size` (`:779-780`).
3. Pads the Mamba page to 784·4096 = 3,211,264 B (`:782-802`, +0.13 %).

The 784 therefore exists only so that one attention page and one Mamba page have the same size in one pool.

### 1.3 Groups, tensors, one pool
- **Page-size unification.** `get_kv_cache_groups` (`v1/core/kv_cache_utils.py:1697`) calls `unify_kv_cache_spec_page_size` (`:1049`), then `_get_kv_cache_groups_uniform_page_size` (`:1108`).
  - Result: 4 groups of 16 layers. Group 0 is FA; groups 1–3 are Mamba, with the 48 GDN layers split by `layers[i::3]`. Group 0 is FA in every test fixture.
- **Shared tensors.** `get_kv_cache_config_from_groups` (`:1318-1400`), general branch:
  - `group_size = 16` and one uniform `page_size` → `num_blocks = mem // page // 16`;
  - **16 `KVCacheTensor`s**, each shared by one FA layer and one layer from each Mamba group (4 layers per tensor).
  - At 8 GiB: 167 blocks × 51,380,224 B per block-row.
- **One pool.** `KVCacheConfig` has one `num_blocks` (`kv_cache_interface.py:880`). `KVCacheCoordinator` builds **one `BlockPool`** (`kv_cache_coordinator.py:91-97`) and hands it to every manager. A block id is a single namespace across all groups. The runtime relies on that in several places (section 2, B).
- **Packed layout exists but does not help.** Upstream also has a packed multi-page-size layout (`_get_kv_cache_config_packed`, `:1277`, DeepSeek-V4 and `enable_cross_layers_blocks`). It still uses one block-id space, and every id reserves bytes for every page size, so it cannot decouple the pools.

### 1.4 Runner layout and kernel blocks
- **Allocation.** `_allocate_kv_cache_tensors` (`gpu_model_runner.py:7325`) allocates one int8 buffer per `KVCacheTensor`.
- **Reshape.** `_reshape_kv_cache_tensors` (`:7377-7486`) derives `num_blocks = numel // page_size_bytes` **per layer**:
  - FA views: `(kernel_num_blocks, 2, kernel_bs, 4, 256)`;
  - Mamba views: `as_strided((num_blocks, *state_shape), stride=page_size_bytes)`, conv first, then ssm, inside one page.
  - Because the reshape is per layer, it already works if FA and Mamba layers sit in different tensors of different sizes.
- **Kernel block size.**
  - `FlashAttentionBackend.get_supported_kernel_block_sizes` returns `[16, 32, 64]` when the Mamba (ssm) cache dtype is fp32 (`flash_attn.py:77-94`, FA issue #1974).
  - `select_common_block_size(784, …)` (`v1/worker/utils.py:262`) therefore picks **16**.
  - `BlockTable` splits every 784-token allocation block into 49 kernel blocks (`block_table.py:50-65, 174+`).
  - *Inferred from the dtype; confirm with `tensor_geometry.kernel_tokens` in a READY receipt.*
  - Consequence: at 64K the kernel block table already has 84·49 = 4,116 entries per request. With 16-token allocation blocks it would have 4,096. **Attention decode cost and block-table upload size do not change.**

### 1.5 Mamba align mode, prefix caching, connector
- `serve.py:83-84, 99-100` passes `--enable-prefix-caching --mamba-cache-mode align`. **Prefix caching is on, not off.**
  - The reason is the OffloadingConnector: `kv_offload/base.py:486-498` requires every group's block size to be divisible by `hash_block_size` and recommends prefix caching for hybrids.
  - Owned (Stage-B) requests never use prefix caching: `get_computed_blocks` returns empty and `cache_blocks` is a no-op (`kv_cache_manager.py:679-705`).
  - The connector is isolated: lookups return 0 and there are no jobs (`remote_ra_http/connector.py:5-32`).
  - It still allocates a **32 GiB pinned CPU pool** at registration (`kv_offload/cpu/gpu_worker.py:493-519`), and that pool is unused.
- **Align mode** (`single_type_kv_cache_manager.py:958-1291`):
  - The Mamba block table is `cdiv(max_model_len, M)` wide (84 entries), mostly null blocks.
  - The running state sits at index `(seq_len-1)//M` (`attention/backends/utils.py:899-939`).
  - On a block crossing, `preprocess_mamba` copies the state into the new block (`v1/worker/mamba_utils.py:678-742`), and the old block is freed one step later (`MambaManager.remove_skipped_blocks`).
  - So a request holds 1–2 blocks per Mamba group, i.e. one GDN "slot" = 3 blocks = 153,944,064 B (+1 transient slot at a crossing).
- **Chunk splitting.** `_mamba_block_aligned_split` (`sched/scheduler.py:342-397`) splits prefill chunks at multiples of **`cache_config.block_size`** (`:369`).
  - Upstream assumes FA block = Mamba block in align mode. After `core.py:316` sets `block_size = min(group block sizes)`, a decoupled config would split at P instead of M (must patch, WP1).
- **Hashing.** `resolve_kv_cache_block_sizes` (`kv_cache_utils.py:607-670`):
  - `scheduler_block_size = lcm(group sizes)`;
  - `hash_block_size = gcd(group sizes)`, **unless a Mamba group's block size ≠ `cache_config.block_size`**, in which case it backs off to the lcm (`:653-658`). That back-off would break the connector assert above.
  - Request block hashes (sha256 of the pickled 784-token chunks) are computed per request in the engine core (`core.py:225`). Random50 seeds from the last one (`private_endpoint.py:109-110`).

### 1.6 What the runtime builds on top (from the two surveys)
- **Pages.**
  - `P = 784` is pinned in `serve.py:41,87` and `remote_ra_http/config.py:45`. It is only *checked* against vLLM: `worker_composition.py:112-116`, `native_growing/adapter.py:98-101`, `phase1_coordinator.py:52-54`, `sparse_mapping.py:86`.
  - One FA block per Page per FA group everywhere: `ownership.py`, `allocation_owner.py`, `contract.py`, `engine_boundary.py`.
  - FA rows are `[selected][run][native transients]` with `SparsePlan(L, L-|A_t|·P)`. Positions stay logical; only FA slots compact.
- **GDN.** The GDN state is carried **continuously** across turns through pinned host memory (`worker_backend.py`, `remote_ra_growing_worker/tensors.py`):
  - carry captured at C′ (end of turn);
  - restored H2D into the phase-1 slot (`prepare.h2d_bytes` = 153,944,064 B);
  - phase-1 state captured and restored into the phase-2 slot;
  - latest state captured at every **Page-aligned** forward end, and restored from host when `preprocess_mamba` moves the slot (`qwen.py:179-192, 246-252`).
  - All of this assumes Mamba block == P. Phase-1 and Goal leases build Mamba rows of length ⌈e/P⌉ and require slot index ⌈e/P⌉−1 (`contract.py:344-347, 405, 445-449`; `native_probe.py:147-150`).
  - No Mamba block is held between turns (CAPACITY.md §Method 1).
- **DRAM catalog.** Worker-private, keyed `(session namespace, Page)`. Each payload is one block per FA group with all 16 layers. Every block piece gets **its own pinned allocation and a synchronous copy** (`tensors.py:310-386`).
- **Capacity gate** (`remote_ra_live_v2/capacity.py`): `A·pages + M·slots` against **one** `block_pool.get_num_free_blocks()`.
- **Turn geometry.** It needs `n = ⌊o/P⌋ ∈ [Q/P, Q/P+1]` (`native_growing/contract.py:218`, `live_v2/contract.py:143-145`).
  - `o − C = 1 + |boundary| = 2 or 3` tokens by construction (`mswea_protocol.py:117-119, 186-195`), so this holds for every P ≥ 4, including 16. *Derived; WP2 adds a test at C ≡ P−1 (mod P).*

## 2. Requirements and invariants
1. **R1.** FA Pages are P tokens (target 16). The GDN state stays one continuous per-request state: not paged, not split, carried across turns as today.
2. **R2.** `P = 784` without the new flag is the legacy path:
   - identical vLLM argv and `effective_server_config`;
   - identical KV config, tokens and evidence;
   - all existing tests unchanged.
3. **R3.** Multi-session serving (≤ 8 sessions, FIFO admission, batched decode, gate, **no preemption**) keeps working. The gate must cover both pools.
4. **R4.** No new per-request or per-step hashing. Existing per-Page digests must not grow into a hot-path cost (section 5).
5. **R5.** Per-step host overhead must not grow with the number of Pages. Steady-state target: ≤ +0.1 ms per request per step at 64K versus `main`.
6. **R6.** Phase-1/2 semantics, selection policy semantics (in Pages of size P), the evidence schema and the clean-shutdown receipts are unchanged. New fields are additive.

## 3. Options

| | A. Two pools (recommended) | B. One pool, GDN state spans many small pages | C. Keep 784 physical blocks, runtime sub-allocates 16-token kernel blocks | D. 16-token attention rows over 784-token residency | E. Paged per-head GDN state (B done properly) |
|---|---|---|---|---|---|
| Idea | FA blocks of P tokens in pool 0; GDN slots in pool 1, sized per session | GDN state = 49 *contiguous, aligned* 16-token blocks (or a reserved region) | Each 784 block is 49 contiguous 64 KiB kernel blocks; the runtime packs logical 16-token Pages into slots and writes kernel-level rows | Rows list only the selected 16-token kernel blocks; residency, DRAM and restore stay per 784 Page | 48 head-pages + 1 conv page per layer; kernels take a per-head state index table |
| vLLM files | `platforms/interface.py`, `v1/core/kv_cache_utils.py`, `v1/kv_cache_interface.py`, `kv_cache_coordinator.py`, `kv_cache_manager.py`, `sched/scheduler.py`, offloading `worker.py` + `cpu/spec.py` (+ connector scheduler group filter); runner reshape unchanged | Same as A, plus a contiguous allocator in `block_pool.py` and Mamba views over aliased ranges | None, except a kernel-level row install that bypasses `BlockTable.map_to_kernel_blocks` (`block_table.py`, `worker_rows`) | None | All GDN Triton/CuTe kernels (`fla/ops/*`, `fused_recurrent`, `chunk_delta_h`, `causal_conv1d`, `gdn_chunk_cutedsl/*`), `gdn_attn.py`, `mamba_utils` copy |
| Kernel constraints | FA kernel block 16 (already the case); GDN kernels untouched | Contiguity needed by the GDN `[num_blocks,…]` views | Kernel block 16 (already); slot mapping from kernel rows | Same as C | Kernel rewrite |
| Block table at 64K | FA 4,096 allocation = kernel entries (today 84 allocation / 4,116 kernel); Mamba 84 | as A | 4,096 kernel entries | ≤ 4,116 | as A |
| Memory overhead | Mamba pool reserved statically: (2S+1) slots. S=1: 0.51 GB, S=8: 2.67 GB of the KV budget. Padding gone. FA tail waste < 16 tokens | Fragmentation, or a reserved region, which amounts to A | Slot fragmentation inside partly used 784 blocks (needs compaction copies) | No memory benefit: a 784 Page stays resident if any of its 49 sub-pages is selected | as A |
| Multi-session capacity | Same total FA tokens (−1–2K for the spare slot, table §4.4). Finer residency; two-pool gate | as A | Depends on fragmentation | Same as today | as A |
| DRAM / restore | Needs a P-granular pinned arena and batched copies (needed by A, B, C, E alike) | as A | as A, plus compaction | Unchanged (784) | as A |
| Risk | Medium: well-delimited vLLM patch; runtime O(Pages) hot spots | High (allocator, fragmentation) | High (new allocator; native row growth stays 784-granular; mixed kernel/allocation rows) | Low/medium, but delivers only attention-granularity semantics | Very high |
| Effort | 10–14 agent-days | > A | ≈ A or more, and more novel invariants | 4–6 agent-days | weeks |

**Verdict.**
- **A** is the only option that gives P-granular residency, restore and selection without kernel work or a new allocator. It also keeps vLLM's own allocator, scheduler and row growth in Pages.
- **D** is a cheaper interim if the algorithm side only needs PoC attention semantics soon. It wastes restore bytes, and `exp-subpage-score` is already exploring adjacent ground.
- **C** is the fallback if a vLLM pool patch is ever ruled out.
- **B** and **E** are rejected.

## 4. Recommended design (decisions)

### D1. Activation and the legacy path
- **Flag.** A new `--additional-config '{"remote_ra_kv_layout": {...}}'` (standard `VllmConfig.additional_config`, `config/vllm.py:358`). It reaches the engine core and the worker unchanged.
- **No key means legacy:** every patched function takes the upstream branch, and `serve.py` emits today's argv byte for byte.
- **Shared helper.** A new pure-Python module `source-overlay/remote_ra_live_v2/kv_layout.py` holds the layout (interface I0, section 6.0). Owner: WP1; the API is frozen by this document. vLLM patches import it lazily, as `kv_cache_manager.py` already imports `remote_ra_live_v2.chunk_protocol`.

### D2. P = 16, parameterised P ∈ {16, 32, 64, 784}
Why 16:
- PoC parity: `PAGE_SIZE = 16`, live tail < 16, mandatory/recent/budget in 16-token units (`selection_contract.default_recent_quota_pages(16) = 64`).
- No kernel cost (§1.4).
- It removes the documented deviations "live tail up to 783 tokens" and "budgets floored to 784" (`stageB-design.md` D2/Q4).

Why not smaller: FlashAttention kernel blocks must be multiples of 16.

Why 32/64 only as fallbacks: they cut the O(Pages) host work 2–4×, but deviate from the PoC.

Constraints: P must be a multiple of 16 (FA and `selection_contract.validate`). P ≠ 784 implies the decoupled layout. `P = 784` + decoupled is allowed as a **validation mode** that isolates the pool split.

### D3. Mamba align block `M` decoupled from P
- **Rule:** `M = P·⌈784/P⌉`, so M = 784 / 800 / 832 / 784 for P = 16 / 32 / 64 / 784.
  - M is a multiple of P, so `hash_block_size = P` passes `kv_offload/base.py:493`.
  - M is a multiple of 16, which causal_conv1d needs (multiple of 8).
  - M ≤ `max_num_batched_tokens`.
- **At P = 16**, M = 784 keeps the GDN behaviour bit-for-bit:
  - same crossings and prefill chunk boundaries (`_mamba_block_aligned_split` must use M);
  - same latest-capture cadence (`qwen.py` must use `end % M`);
  - same H2D accounting, same Mamba table width (84).
  - So `full` at P = 16 must be token-exact against legacy (section 8).
- **Rejected: `M = P = 16`.** That is a crossing every 16 tokens, each costing a 154 MB D2H capture plus a 154 MB H2D relocation in this runtime.
- **Deferred: `M ≥ max_model_len`** (D5).

### D4. Pools and tensors
- **Groups.** Exactly as today: same specs and layer→group assignment. The only difference is that the decoupled branch **skips `unify_kv_cache_spec_page_size`**.
- **Pool 0 (FA):**
  - 16 `KVCacheTensor`s, one per FA layer, each `N_fa × P·4096` B;
  - `N_fa = (budget − mamba bytes) // (P·65,536)`;
  - `num_gpu_blocks_override` applies here.
- **Pool 1 (Mamba):**
  - 16 tensors, each shared by the i-th layer of the three Mamba groups (as today), each `N_m × 3,207,168` B, **unpadded** (`mamba_page_size_padded = None`);
  - `N_m = 1 + 3·K` with `K = 2S + 1` slots for S = `max_num_seqs`: per running request one running slot plus one crossing slot, plus one spare. That covers phase 1 plus a Goal forward (2 slots) while S−1 others hold 2 each.
- **Block ids** are per pool and may coincide across pools. Each pool has its own null block 0.
- `KVCacheConfig` gains `group_pool: tuple[int,…] | None` and `num_blocks_by_pool: tuple[int,…] | None`. None means legacy. `num_blocks` keeps meaning pool 0.

### D5. Follow-ups, out of scope for the first implementation
- **F1. `M ≥ max_model_len`.** One GDN block per group per request and zero crossings. Saves ~308 MB of copies per crossing (about 1–2 per turn at 784).
  - Needs: scheduler block size handling (`lcm` becomes ≥ 65536), hash back-off removal, the align-split edge case at `num_tokens == M`, and the runtime cadence.
  - Needs its own validation, because GDN evidence changes.
- **F2.** Drop the OffloadingConnector in decoupled mode (frees its 32 GiB pinned pool). It changes the startup contract (`isolate_connector`, `connector_guard`), so it is an open question.
- **F3.** Skip vLLM block hashing for owned requests. Random50's seed would need another source.

### D6. Offloading connector in decoupled mode
It stays registered and isolated, but **Mamba groups and tensors are excluded** from its registration, CPU sizing and group configs. Otherwise:
- `offloading/worker.py:54,116-129` builds `(num_blocks_fa, page)` views over the Mamba storage, which raises;
- `cpu/gpu_worker.py:493-519` allocates `num_cpu_blocks × 3.2 MB` pinned per Mamba tensor. With CPU blocks sized from 1 MiB FA blocks, that is about 1.6 TB, and fails.

### D7. Two-pool capacity gate
Same structure as `capacity.py`, evaluated per pool:
- FA: `A·pages` against FA free blocks.
- Mamba: `Mg·slots` against Mamba free blocks.
- Headroom and watermark per pool (the watermark is on FA only, as in vLLM).

With N_m as in D4, the Mamba inequality never binds for ≤ S sessions, but it is still checked. In legacy mode the gate is the current single sum, with the same messages.

### D8. Host-side data path at small P (mandatory for P = 16; section 5)
- pinned DRAM arena with batched gather/scatter copies;
- batched device-side Page-vector compaction;
- compact int8 vectors end to end;
- incremental FA row install.

### 4.4 Capacity numbers (`tools/capacity_numbers.py`)

| KV budget | S | legacy blocks (×784) | FA tokens, legacy, at peak GDN use | decoupled Mamba pool | decoupled FA tokens (P=16) |
|---|---|---|---|---|---|
| 8 GiB | 1 | 167 | 125.4K | 10 blk, 0.51 GB | 123.2K |
| 8 GiB | 8 | 167 | 92.5K | 52 blk, 2.67 GB | 90.3K |
| 24 GiB | 4 | 501 | 373.2K | 28 blk, 1.44 GB | 371.3K |
| 32 GiB | 8 | 668 | 485.3K | 52 blk, 2.67 GB | 483.6K |

- **Raw FA token capacity** is unchanged within about 2K tokens (the spare slot).
- **Capacity gains come from granularity.**
  - A partly filled Page wastes < 16 tokens instead of < 784. Per running session that is about 1–3 run/tail Pages × 51 MB today.
  - Released Pages pinned until commit (CAPACITY.md) are finer.
  - Selection spends the 40 % budget exactly.
  - Churn between consecutive selections at 16 tokens may raise or lower restores. **Not predictable; measure.**
- **The main motivation is PoC fidelity, not capacity.**

## 5. Hot spots that scale with the Page count (must be addressed for P = 16)

| # | Where | P = 784 | P = 16 | Required change (WP) |
|---|---|---|---|---|
| H1 | FA row recompose every step: `worker_rows.install_row` → `verify_native_row`×2 + `compose_rows` | measured 0.39 ms/request/step at 64K, 40 % selection (`tools/rowbench.py`, x86) | 0.81 ms/request/step, so ~3 ms per step at 4 sessions | Incremental: cache the composed row per (request, row); on native growth append and verify the new tail only; full verify only when the layout changes (WP3). Lowers today's cost too |
| H2 | Page vectors: `qwen.py:237-245`, `model_queries.py:58`: per closed Page, `row.to(cpu).tolist()` of 5,120 floats + Python `normalized()` | ~8 Pages per 6.5K observation | ~406 Pages per observation; estimate 0.3–1 s per turn (not measured) | One device gather of all closing rows per forward; batched project→L2→int8 on device; D2H 128 B + scale per Page (WP3) |
| H3 | Vector storage (`RequestState.page_vectors`, `QuerySelectionAuthority.vector_catalog`): tuples of 5,120 Python floats | ~14 MB per 64K session | ~0.6–0.7 GB per 64K session (×8 sessions) | Store `CompactVector` (int8[128] + scale) only for P < 784 (WP3/WP4) |
| H4 | `worker_backend.publish_vectors`: JSON + **HMAC-SHA256** over all phase-2 vectors per request | ~8 Pages × 5,120 floats | ×49 bytes, so per-request hashing grows (R4) | Publish compact vectors (64 KB per 500 Pages); keep the MAC over that, or replace it with an identity token (open question 7) (WP3) |
| H5 | `query_selection.plan` compaction (torch CPU, per Page, cached) | — | ~500 new Pages per turn | Consume worker-compacted vectors; no compaction on the engine path (WP4) |
| H6 | Selector V2 loop over n Pages per turn (upstream records, publisher, cosine, evidence `scores`/`page_turns`) | n ≤ 84 | n ≤ 4,095 (the PoC does the same) | Keep the upstream algorithm. Measure. Bound evidence size (open question 6) (WP4) |
| H7 | `TensorIO.capture/restore`: pinned alloc + synchronous copy per block per layer | seal ~8 Pages × 16 copies per turn | seal ~400 × 16 = 6,400 pinned allocs and sync D2H per turn; restore up to ~1,600 × 16 sync H2D | Preallocated pinned arena of P-Pages; per layer, one device gather into staging + one D2H (and one H2D + one scatter). Identity tokens per arena slot (WP3) |
| H8 | Per-Page `token_sha256` (`allocation_owner.py:341`, `native_growing/adapter.py:321`) | ~8 per turn | ~500 per turn, ~1 µs each | Count ×49, bytes unchanged. Acceptable; optionally replace with an identity token (open question 7) |
| H9 | vLLM request block hashes (`core.py:225`) | 83 per 64K | 4,096 per 64K. Emulation: 1.2 → 4.2 ms per 64K tokens, ~0.5 ms per 8K-token turn (`tools/hashbench.py`) | Acceptable (existing vLLM mechanism). F3 removes it for owned requests |
| H10 | `runtime.jsonl` per turn (selected Pages, per-Page scores and turns) | ~84 entries | ~4K entries (~100–200 KB per row) | Decide format (open question 6) (WP4) |
| H11 | Engine catalog (`Session.catalog[page]` dicts), `adapter.py:177` bound `n ≤ 4096` | 84 | 4,095: fits exactly at 64K | Replace the bound with `cdiv(max_model_len, P)` (WP2) |

## 6. Work packages

All WPs branch from `exp-decouple-pages`. The file ownership below is disjoint, so the four can run concurrently. The **integration owner** (orchestrator) regenerates release metadata only at merge time:
- `patches/v0.24.0.diff` (anchor files becoming modified);
- `INSTALL_RELEASE.json` and `SOURCE_SHA256SUMS.txt` via `development/selection-contract-v7/regen_release.py`;
- any source-closure pins (`remote_ra_live_v2/loaded_sources.py`).

**Conflict note.** `exp-subpage-score` (another agent) modifies `serve.py`, `composition.py`, `native_probe.py`, `private_endpoint.py`, `query_selection.py`, `selection_contract.py` and `worker_composition.py`. Keep edits to these files small and localized. Rebase only after that branch lands on main.

### 6.0 Shared interface I0: `source-overlay/remote_ra_live_v2/kv_layout.py` (pure Python; owner WP1, API frozen here)
```python
LAYOUT_KEY = 'remote_ra_kv_layout'          # key in VllmConfig.additional_config
LEGACY_PAGE = 784
@dataclass(frozen=True)
class KvLayout:
    page_tokens: int     # P: FA block size == runtime Page, multiple of 16
    mamba_tokens: int    # M: Mamba align block (GDN crossing cadence), multiple of P and 16
    mamba_slots: int     # K: GDN slots in the Mamba pool (1 slot = 1 block per Mamba group); 0 in legacy
    decoupled: bool      # False only for the legacy shared pool (P == M == 784, K == 0)
def legacy() -> KvLayout
def default_mamba_tokens(P: int) -> int                 # P * ceil(784 / P)
def for_server(P: int, sessions: int, decoupled: bool | None = None) -> KvLayout
                                                        # decoupled defaults to P != 784; K = 2*sessions + 1
def to_additional_config(layout) -> dict | None         # None for legacy -> serve.py emits no --additional-config
def from_vllm_config(vllm_config) -> KvLayout           # legacy if the key is absent; validates
def mamba_pool_blocks(layout, mamba_groups: int) -> int # 1 + mamba_groups * K
def mamba_row_len(end: int, layout) -> int              # ceil(end / M)
def mamba_index(end: int, layout) -> int                # ceil(end / M) - 1
```
Validation:
- P % 16 == 0, M % P == 0, M % 16 == 0, 0 < M ≤ 16384.
- Decoupled ⇒ K ≥ 3. Legacy ⇒ P == M == 784.
- `additional_config` value: `{"page_tokens": P, "mamba_tokens": M, "mamba_slots": K}`.

### WP1 — vLLM split pools (overlay `source-overlay/vllm/**` + `kv_layout.py`)
1. **`platforms/interface.py` `_align_hybrid_block_size`.**
   - If `from_vllm_config(...).decoupled`: require a user-specified `block_size == P` (supported by the backend: multiple of 16) and a user-specified `mamba_block_size == M`.
   - Set `mamba_page_size_padded = None`, and **return without raising `block_size`**.
   - Legacy: untouched.
2. **`v1/kv_cache_interface.py`.** `KVCacheConfig.group_pool` and `num_blocks_by_pool` (default None; legacy objects compare equal to today's).
3. **`v1/core/kv_cache_utils.py`.**
   - `get_kv_cache_groups`: decoupled → no page unification; same grouping.
   - `get_kv_cache_config_from_groups`: decoupled branch per D4. With an override (`_init_minimal_kv_cache_for_profiling` passes `available_memory=0`): `N_fa = override`, `N_m = 1 + 3K`.
   - `get_kv_cache_configs`: shrink tensors per pool.
   - `_check_enough_kv_cache_memory` and the concurrency log: per pool.
   - `resolve_kv_cache_block_sizes`: decoupled → `(lcm(all), gcd(non-Mamba groups))`, i.e. (784, 16).
4. **`v1/core/kv_cache_coordinator.py`.**
   - `self.block_pools = tuple(BlockPool(n,…) for n in num_blocks_by_pool or (num_blocks,))`, `self.block_pool = block_pools[0]`.
   - Managers get `block_pools[group_pool[i]]`.
   - `find_longest_cache_hit` uses each group's pool.
   - Add `get_num_blocks_to_allocate_by_pool(...) -> tuple[int, ...]`; the scalar method stays and returns the sum.
5. **`v1/core/kv_cache_manager.py`.**
   - `block_pools`, `pool_of_group(g)`, `get_num_free_blocks_by_pool()`.
   - `allocate_slots`: both checks (full-sequence and main) per pool.
   - `usage` reports pool 0, plus a new `usage_by_pool`.
   - `evict_blocks` / `reset_prefix_cache` / `take_events` across pools.
   - `take_new_block_ids` stays FA-only (zeroing FA tensors, `v1/worker/utils.py:78-218`; harmless).
6. **`v1/core/sched/scheduler.py`.**
   - `_mamba_block_aligned_split` uses the Mamba groups' `block_size` (equals `cache_config.block_size` in legacy).
   - `_remote_ra_native_free_request` eviction (`:2330-2360`) and `:2410` free: per group pool.
   - Connector binds pool 0.
7. **Offloading** (`offloading/worker.py`, `kv_offload/cpu/spec.py`, `offloading/scheduler.py` group configs). Decoupled → only non-Mamba groups and tensors are registered. CPU bytes per block come from pool-0 tensors and `num_blocks_by_pool[0]`.
8. **Runner and worker.**
   - Verify there is **no change** in `_reshape_kv_cache_tensors` (per-layer `numel // page`).
   - `_update_hybrid_attention_mamba_layout` is a no-op for FA `block_dim == 0`.
   - Check `gpu_worker.py:604` (`num_gpu_blocks` = pool 0) and the KV zeroer.

**Exports to WP2/WP3 (interface I1):**
- `kv_manager.block_pools`, `kv_manager.pool_of_group(g)`, `kv_manager.get_num_free_blocks_by_pool()`;
- `kv_cache_config.group_pool` / `num_blocks_by_pool`;
- FA spec `block_size == P`, Mamba spec `block_size == M`, unpadded Mamba page;
- worker-side `runner.kv_cache_config` carries the same fields.

**Acceptance.**
- Legacy: CPU tests show `KVCacheConfig` and scheduler/hash sizes identical to the unpatched functions on the Qwen3.6 fake (golden: 167 blocks, 16 × 167 × 3,211,264 B, (784, 784)).
- Decoupled: the planner, pools, allocation/free and `resolve` (784, 16) behave as in D4.

### WP2 — engine-side runtime geometry, ownership and capacity
Files: `serve.py`, `remote_ra_http/config.py`, `remote_ra_native_growing/{adapter,ownership,contract}.py`, `remote_ra_live_v2/{contract,phase1_coordinator,native_probe,engine_boundary,pool_incarnations,capacity,composition(startup wiring only),startup}.py`, `growing_startup.py`.
1. **Startup.**
   - `serve.py --page-tokens {784,16,32,64}` (default 784 → argv, config and `effective_server_config` byte-identical), plus a test-only `--kv-layout decoupled` for P=784 validation.
   - P ≠ 784 adds `--block-size P --mamba-block-size M --additional-config {...}` and `effective_server_config` keys `block_size`, `mamba_block_size`, `additional_config`. The startup config gets `page_tokens=P` and `kv_layout` (I0 dict). The selection contract becomes `default_contract(P)`.
   - `remote_ra_http/config.py:45`: accept the allowed P with a matching layout and effective config.
   - Build `KvLayout` once at startup and cross-check it with `kv_manager` (FA managers `block_size == P`, Mamba `== M`, `group_pool` present iff decoupled), replacing `adapter.py:98-101`.
2. **Mamba arithmetic P → M.** `Phase1Lease.mamba_index`, `parse_lease`/`_parse_goal` row lengths, `Phase1Coordinator.run` (`M = ceil(e/M)`, Goal width), `NativeQueryForward._mamba_slot`. Use I0 helpers.
3. **Pools.**
   - `Phase1Coordinator` allocates FA from pool 0 and Mamba from pool 1, and frees each to its own pool.
   - `PrivateBlockOwner` owns pool-0 blocks only. Assert that it never sees a Mamba block.
   - `pool_incarnations` / `engine_boundary.install` on every pool.
   - Alias and uniqueness checks keyed `(pool, id)`: `contract.py:367-368, 408-413`, `native_probe.py:49-51`, `RowLayout` (FA-only; unchanged).
4. **Capacity (D7).** `turn_need`/`headroom`/`assess`/`admission_gate` return per-pool needs. The CapacityWait text names both pools; the legacy text is unchanged.
5. **Bounds.** `adapter.py:177` becomes `n ≤ cdiv(max_model_len, P)`. Add a test that `o − C ≤ 3` keeps `CANDIDATE_RANGE` valid at P=16 for C ≡ 0, 1, P−1 (mod P).

**Interface I2 (to WP3):** the lease and `allocation_update` wire shapes are unchanged:
- FA rows: one id per Page per FA group (pool 0);
- Mamba rows: length `mamba_row_len(e)`, null-padded, ids from pool 1;
- every frame carries the startup `kv_layout` dict once (startup record), never per step.

**Acceptance.**
- All existing tests pass unchanged (legacy).
- New contract, capacity and ownership tests are parameterised over (P, M) ∈ {(784,784 legacy), (784,784 decoupled), (16,784), (64,832)}, with a random turn-sequence property test: alloc/free balance per pool and no in-pool aliasing.

### WP3 — worker-side data path at small P
Files: `remote_ra_growing_worker/tensors.py`, `remote_ra_live_v2/{worker_backend,allocation_owner,worker_rows,sparse_mapping,geometry,qwen,worker_composition,model_queries}.py`.
1. **Gates.**
   - `worker_composition.py:112-116`: FA views `allocation_tokens == P`, hybrid views `== M`.
   - `TensorIO._validate_aliases`: decoupled → no cross-kind aliasing at all (FA and GDN tensors disjoint).
   - `validate()` uniqueness per pool; `allocation_owner.py:205, 469-470` per pool.
2. **GDN cadence.** `qwen.py` captures latest at `lease.end % M == 0` (M from the startup layout). Nothing else in the GDN path changes.
3. **DRAM arena (H7).**
   - Preallocated pinned arena of FA Page slots, all 16 FA layers per slot, `P·4096` B per layer.
   - Capacity from the startup config. Default: enough for S sessions × `cdiv(max_model_len, P)` Pages, about 4 GiB per 64K session. The existing 32 GiB connector pool stays separate (open question 9).
   - Batched `capture(pages)` / `restore(pages)`: per FA layer one device `index_select` into a staging buffer + one D2H (H2D + `index_copy_`), then drain once.
   - Identity tokens per slot replace per-payload tokens. Byte counters keep their meaning.
   - GDN payloads stay as today.
   - Legacy P=784 can keep the old path. Equivalence is tested on GPU.
4. **Vectors (H2–H4).**
   - Decoupled: closing rows are gathered once per forward and compacted on device in one batch, using `selection_contract.projection` and the `compact` math (fp32 linear → L2 → int8 with scale).
   - `page_vectors` holds `CompactVector`. `publish_vectors` sends `{page: (int8 list, scale)}`.
   - Legacy keeps the float path, byte-identical.
5. **Rows (H1).** Incremental `install_row` with the same invariants: prefix immutability, exact expansion. Budget ≤ 0.05 ms/request/step steady state at 64K.

**Interface I3 (to WP4):** `publish_vectors` body adds `representation="compact_int8_v1"` with `page_vectors={page: [128 int8], scale}` in decoupled mode. The legacy representation is unchanged.

**Acceptance.**
- CPU tests: arena slot bookkeeping (pure Python), incremental row install equal to full recompose on random sequences (numpy only), qwen cadence with M ≠ P.
- Torch-dependent tests skip locally (no torch on this machine, no downloads allowed) and run in the DeltaAI environment.

### WP4 — selection contract, evidence, tests, GPU harness
Files: `remote_ra_live_v2/{selection_contract,query_selection,private_endpoint(seed note only),vector_space}.py`, `remote_ra_growing/contract.py`, `remote_ra_http/evidence.py`, new `test_decouple_*.py`, `development/exp-decouple-pages/gpu-smoke/`.
1. **Contract.**
   - `default_contract(16)` gives recent quota 64 and capacity `floor(0.4·n·16)//16`.
   - Keep the `initial_prompt_pages` (cover) semantics unless the algorithm side chooses the PoC's "fully inside" (open question 2). That would be a new enum value, not a change of the old one.
2. **Selection.**
   - `query_selection.plan` accepts compact vectors (I3) and skips compaction.
   - At P=16 the PageRecord token span equals the native Page (the formula already yields this).
   - Evidence size per open question 6.
3. **Evidence.**
   - `ready.json`/`startup.json` record `kv_layout` and `num_blocks_by_pool`.
   - The per-turn row adds FA/Mamba pool free counts.
   - `restore.h2d_bytes` / `prepare.h2d_bytes` keep their definitions.
4. **Tests.** Parameterise the stage-B suites over legacy and (16,784): `test_stageb_{contract,engine,worker,e2e}.py`, `test_ms_*`, `test_host_trim.py`. Their fakes currently hard-wire FA = Mamba = P. New `test_decouple_layout.py` (I0 + WP1 AST tests).
5. **GPU harness.** sbatch scripts for section 8, with `--mail-type=BEGIN,END,FAIL` to the DeltaAI job mail address, serial on `ghx4-interactive` (one job at a time, new `-RunLabel` on reruns). They reuse `development/stage-b/tools` and `development/multi-session/tools/step_profile.py`.

### Integration order
1. WP1 and WP2/3/4 code against I0–I3 using fakes.
2. Merge WP1 → WP2 → WP3 → WP4.
3. Regenerate release metadata; run `python -B -m unittest discover` and `check_git_release.py --ref HEAD`.
4. Run the GPU smoke.

## 7. CPU test plan (local, torch-free unless noted)
- **WP1.** Extract the patched functions with `ast` (the style of `test_allocation.py`) and feed them fake specs: 16 FA layers at P, 48 Mamba layers with `page_size_bytes = 3,207,168`, block M.
  - Legacy golden equality against the original functions taken from `main`'s bytes.
  - Decoupled: groups identical to legacy; tensor list and sizes; `N_fa`/`N_m` for 8/24 GiB and S = 1/4/8; profiling override; `resolve` (784, 16); two-pool allocate/free; the MambaManager align sentinel against its own pool; ids that coincide across pools; `_mamba_block_aligned_split` at M with `cache_config.block_size = 16`.
  - Offload registration excludes Mamba.
- **WP2.**
  - Contract and lease geometry over the (P, M) matrix; `mamba_index`; Goal rows.
  - Ownership `row_groups`; two-pool capacity (worked examples from CAPACITY.md recomputed for P=16).
  - `serve.py` dry-run argv: legacy byte-identical (existing `test_ms2_admission` checks), decoupled adds exactly the three flags.
  - Config pin.
- **WP3.** Arena slot allocator; incremental row install equal to full compose; qwen capture cadence; `validate()` per pool. (Torch on DeltaAI: batched compaction equal to per-Page `compact()` up to the agreed tolerance, open question 3; arena copy round-trip on CPU tensors.)
- **WP4.** Selection numbers at P=16 against PoC formulas; evidence schema; random50 `sparse_count`.
- **Release.** `check_git_release.py` at the end.

## 8. GPU validation plan (first smoke, serial jobs on one GH200)
Same commit throughout, the same synthetic multi-turn workload (6 turns of ~6.5K-token observations, up to ~40–60K prompt tokens), `temperature 0`.

| Job | Config | Must show |
|---|---|---|
| A | legacy (P=784, no flag) | Regression: identical tokens and evidence vs the `main` runtime (902690a) on the same inputs |
| B | decoupled, P=784, M=784 | Startup logs two pools (FA = Mamba counts per D4), no padding log. `full` tokens **exactly equal** to A. `prepare.h2d_bytes` = 153,944,064 per continuing turn. Crossing relocations equal to A |
| C | decoupled, P=16, M=784, `full` | Tokens **exactly equal** to A (same kernel, same KV order, same chunking). FA pool counts as computed. Per-step decode overhead within +0.1 ms/request of A at 1 session; admission (phase 1 + reserve) not slower than A by more than 10 % |
| D | P=16, four policies, 10 turns | Selection active after 8,192 tokens. `capacity_pages = floor(0.4·n·16)//16`, recent ≤ 64, restores happen. Clean shutdown receipts. Host RSS and pinned bytes per session within the arena estimate. No CAPACITY or ownership failures |
| E | P=16, 4 sessions × 6 turns, 24 GiB | Batched decode, no preemption. Both pools return to baseline after drain. Pool usage samples within the two-pool model. CAPACITY_WAIT appears only when the FA gate binds |
| F | `step_profile.py`, P=16 vs A at 1–4 sessions | Per-step host overhead ≤ A + 0.1 ms per request (R5); `install_row` time; vector pipeline time per turn |

**Token-exactness.** Also compare C against stock vLLM via the in-window decode step, per the established audit practice.

**If C is not exact**, report the first divergent position. A divergence with B exact points at P-dependent code (rows, slot mapping, chunking); B not exact points at the pool split.

## 9. Risks
- **Hidden single-pool assumptions** in upstream code paths not visible here (metrics/Prometheus usage, KV events consumers, the CPU offload manager). Mitigation: the file list in section 11; fail-fast asserts on `group_pool` in decoupled mode. Inside the visible coordinator, every `self.block_pool` use (including `find_longest_cache_hit` of the hybrid coordinator) must be reviewed.
- **Host-side O(Pages) costs** (section 5) eating the decode budget. Mitigation: the R5 budget is enforced by job F; P=64 fallback.
- **Vector arithmetic.** Batched device compaction changes int8 rounding against today's per-Page CPU path. This only affects decoupled mode, but it is a semantic change for the algorithm side.
- **Pinned host memory.** The 32 GiB connector pool (unused) plus the arena (~4 GiB per 64K session). Watch node RAM with 8 sessions.
- **Merge conflicts** with `exp-subpage-score`.
- **Release metadata churn.** Anchors (`interface.py`, `kv_cache_utils.py`, `kv_cache_interface.py`, the offload files) become modified files. The installer's upstream map handles this (`expected_live` against `payload`), but `patches/v0.24.0.diff` grows.

## 10. Open questions (user / algorithm side)
1. **P.** Is P = 16 (PoC parity) the target, with P = 64 accepted as a fallback if the host overhead budget (R5) cannot be met? Should decoupled P=16 become the default after qualification, or stay opt-in?
2. **Mandatory Pages at P = 16.** Keep "every Page the system+task prompt touches" (cover, today) or switch to the PoC's "Pages fully inside" (floor)? At P=16 they differ by at most one Page.
3. **Page vectors.**
   - Must the int8 vectors be bit-identical to the PoC's `_compact_last` (batched, from the fp32 hidden), or to today's runtime path (Python float64 pre-normalisation + per-vector torch CPU compaction)?
   - Is equal cosine ranking enough?
4. **GDN cadence.**
   - Is "one continuous GDN state per request" satisfied by today's align semantics: one state, moved between slots every M = 784 tokens with a host round trip?
   - Or is the no-crossing variant (F1) wanted in the first version?
5. **Mamba pool.** Is a static reservation of (2S+1) slots (0.51 GB at S=1, 2.67 GB at S=8) out of `--kv-cache-gib` acceptable? Or should `--kv-cache-gib` mean FA only, with the Mamba pool added on top?
6. **Evidence volume at P=16.** Per-turn rows would carry ~4K selected/score/turn entries. Keep them in full, or record selected Pages plus a top-k of scores?
7. **Existing per-Page and per-request digests.** May the following be replaced at P < 784, in the spirit of the "no hashing on the hot path" rule?
   - `token_sha256` per Page;
   - the HMAC over published vectors.
   - Candidate replacements: identity tokens, or a MAC over compact vectors only.
8. **vLLM block hashes at 16 tokens.** 4,096 per 64K request; ~4 ms per 64K in a CPython emulation here, not measured on Grace. Acceptable? Or skip hashing for owned requests (F3)? F3 needs a new Random50 seed, because the seed is currently the last block hash.
9. **OffloadingConnector in decoupled mode.** Keep it registered (isolated, FA only) or drop it (F2)? Dropping it frees its unused 32 GiB pinned pool for the DRAM arena, but changes the startup contract.
10. **Interaction with `exp-subpage-score`.** If P = 16 lands, is sub-page scoring of 784 Pages still needed? Which branch rebases on which?

## 11. Upstream vLLM 0.24.0 files not available locally but needed (from the wheel on DeltaAI)
- **`vllm/engine/arg_utils.py`.** EngineArgs → CacheConfig wiring: does `--block-size` accept 16 for this model? Is `--mamba-block-size` honoured with `--mamba-cache-mode align`? How is `--additional-config` parsed? Any block-size choice list?
- **`vllm/model_executor/models/config.py`.** Qwen3.5/Qwen3Next `verify_and_update_config`:
  - where `mamba_ssm_cache_dtype = float32` is set from `mamba_ssm_dtype`;
  - the defaults for `mamba_cache_mode` / `mamba_block_size`;
  - any second block-size alignment.
- **`vllm/config/model.py`, `vllm/config/scheduler.py`.** `is_hybrid`, the default `mamba_block_size`, `disable_hybrid_kv_cache_manager`, `long_prefill_token_threshold`, `watermark`.
- **`vllm/v1/request.py`.** `Request.block_hashes` and block-hasher invocation points (H9, Random50 seed).
- **`vllm/utils/hashing.py`.** The exact sha256 serialization, to measure H9 for real.
- **`vllm/v1/core/kv_cache_metrics.py`, `vllm/v1/metrics/{stats,loggers}.py`.** Per-pool usage and Prometheus `kv_cache_usage`, used by the CAPACITY.md cross-checks.
- **CPU offload internals:**
  - `vllm/v1/kv_offload/cpu/manager.py` (`CPUOffloadingManager`) and the eviction policies;
  - `vllm/v1/kv_offload/factory.py`;
  - `vllm/distributed/kv_transfer/kv_connector/factory.py`.
  - Needed for Mamba exclusion and for dropping the connector.
- **`vllm/_custom_ops.py`** (`swap_blocks_batch`, cache ops) and its csrc counterparts, if the arena reuses vLLM's batched copy ops instead of torch indexing.
- **`vllm/v1/attention/backends/fa_utils.py`** and the `vllm_flash_attn` interface: block-table limits; confirm the FA2/FA3 path on GH200 with kernel block 16.
- **`vllm/v1/executor/abstract.py`**, plus `gpu_worker` memory determination with `kv_cache_memory_bytes` (already visible), to confirm the budget is split exactly once.
- **`vllm/v1/core/sched/interface.py`, `vllm/v1/core/sched/utils.py`.** Minor, for completeness.

## Appendix: key facts with sources
- **FA kernel sizes** `[16,32,64]` with an fp32 Mamba state: `vllm/v1/attention/backends/flash_attn.py:77-94`. Kernel split: `vllm/v1/worker/block_table.py:50-65`.
- **Align GDN index** `(seq_len-1)//M`: `vllm/v1/attention/backends/utils.py:927-939`. Crossing copy: `vllm/v1/worker/mamba_utils.py:678-742`. Two blocks per group: `vllm/v1/core/single_type_kv_cache_manager.py:1111-1252`.
- **Runtime GDN cadence**: `remote_ra_live_v2/qwen.py:179-192, 246-252`. Carry/phase-1 H2D: `remote_ra_live_v2/worker_backend.py:111-177`.
- **Single-pool checks**: `remote_ra_live_v2/capacity.py:175-178`, `phase1_coordinator.py:35,102-118`, `remote_ra_native_growing/ownership.py:53-81,136-137`, `remote_ra_growing_worker/tensors.py:296-302`, `remote_ra_live_v2/allocation_owner.py:205,469-470`, `pool_incarnations.py:27-28`.
- **"One FA block per Page" gates**: `sparse_mapping.py:86`, `worker_composition.py:112-116`, `remote_ra_native_growing/adapter.py:98-101`, `phase1_coordinator.py:52-54`, `remote_ra_live_v2/contract.py:264-269`.

## Deviations

### WP1 (branch `exp-decouple-wp1`)
1. **Interface readings (no API change).**
   - `to_additional_config` returns the whole `--additional-config` dict, `{"remote_ra_kv_layout": {"page_tokens", "mamba_tokens", "mamba_slots"}}`, or None; `from_vllm_config` inverts it. The key's presence means decoupled, P = 784 included. `mamba_pool_blocks` raises for the legacy layout.
   - `kv_manager.pool_of_group(g)` returns the `BlockPool`; its index is `kv_cache_config.group_pool[g]` (0 = attention, 1 = Mamba). `get_num_free_blocks_by_pool()` is a tuple, pool 0 first. Legacy: `block_pools == (block_pool,)`, all groups in pool 0.
   - Pools follow the spec type, not the group index. The real model may list the Mamba groups first (layer 0 is GDN); the §7 fixtures put FA first. WP1 does not depend on the order.
2. **Test files.** `test_kv_layout.py` (I0) and `test_kv_pools.py` (WP1) instead of `test_decouple_layout.py`, which is in WP4's file list. `test_kv_pools.py` executes the overlay sources with stub imports. Its legacy comparisons against the base commit 8586dbd need the Git history and skip in the Git-archive run.
3. **Offloading (D6).** The scheduler-side group configs keep one entry per KV group: block ids are zipped per group and the GPU transfer asserts one ref list per group. The worker registers pool 0 only and gives the Mamba groups empty refs. The native offloading scheduler raises if it would emit jobs in decoupled mode; only the isolated connector is supported.
4. **Deferred frees.** Decoupled mode refuses `defer_block_free` (overlapping batches with a consumer connector; off in this runtime), because the deferred list would mix blocks of both pools.
5. **Manager details.** `evict_blocks(ids)` acts on pool 0 (connector ids). `reset_prefix_cache` and `take_events` cover every pool. `usage` stays pool 0; `usage_by_pool` is new. The residency metrics collector is bound to pool 0. `get_kv_cache_capacity` (concurrency log, `kv_cache_size_tokens`) is the minimum over the pools.
6. **Refused at startup in decoupled mode:** hidden-state cache layers, packed or cross-layer layouts, configs without both attention and Mamba groups, padded Mamba specs, `--mamba-cache-mode` other than align.
7. **Runner and worker: no change** (item 8 verified). The reshape is per layer (`numel // page`), FA has `block_dim == 0`, the zeroer touches FA layers only, `num_gpu_blocks` is pool 0, and the Mamba kernel block is M.
8. **Release metadata on the branch.** Pins come from `regen_release.py` plus the source-closure pins of the edited closure files, which `regen_release.py` leaves alone; `kv_layout.py` is not added to the closure set. Installer counts are (291, 200). `patches/v0.24.0.diff` is not regenerated (integration owner). Newly modified anchors: `platforms/interface.py`, `v1/kv_cache_interface.py`, `v1/core/kv_cache_utils.py`, `offloading/worker.py`, `kv_offload/cpu/spec.py`.
9. **Needs GPU job B.** Not visible locally: that nothing upstream (`models/config.py`, `arg_utils.py`) overrides a user `--mamba-block-size` in align mode or re-pads the Mamba page. The startup log line "Decoupled KV layout: attention block …" marks the decoupled branch.

### WP2 (branch `exp-decouple-wp2`)
1. **Startup.** `serve.py --page-tokens {784,16,32,64}` (default 784) and the validation-only `--kv-layout decoupled`. Without either, `prepare()` takes no new branch: argv, config, `effective_server_config` and pins are byte-identical (proved against the 8586dbd code). Decoupled appends exactly `--block-size P --mamba-block-size M --additional-config <to_additional_config, sorted keys>` at the end of argv; the config gains `kv_layout` (the inner I0 dict) and `page_tokens = P`; the selection contract is `default_contract(P)`. `remote_ra_http/config.py` requires a `kv_layout` record to equal `for_server(P, sessions)` and the three pinned vLLM arguments; `effective_server_config` compares them only in decoupled mode (a JSON-text `additional_config` is normalized). `legacy_geometry()` passes `kv_layout` on.
2. **Layout plumbing.** `growing_startup` builds the layout once (`contract.startup_layout(config geometry) or legacy()`), requires it to equal `from_vllm_config(vllm_config)`, verifies `kv_layout.py` as a loaded module, and passes it to `EngineCoreAdapter(layout=...)`, which cross-checks the manager (FA block = P, Mamba block = M by spec kind; legacy: `group_pool is None`, `block_pools == (block_pool,)`; decoupled: `group_pool`, two pools, `pool_of_group(g) is block_pools[group_pool[g]]`, Mamba pool = `mamba_pool_blocks`). Fixtures pass no layout and keep M = P in one pool. `compose_engine` adds `kv_layout` to the worker startup record in decoupled mode only (I2: once, never per step); `startup_layout(record)` reads it on the worker.
3. **Leases (I2 made precise; shapes unchanged).** `parse_lease(raw, attention_groups=..., layout=...)`: shared pool exactly as before. Decoupled: Mamba rows ceil(e/M) and the Goal Mamba row ceil(goal/M) long; ids unique per pool and keyed (pool, id); incarnation keys `str(id)` for pool 0 and `"1:id"` for pool 1; `owned` lists the fresh FA ids (FA group order) then the Mamba ids (group order) and may repeat an id across pools; **`zero` is the pool-0 part only** (Goal: its FA ids): vLLM's `KVBlockZeroer` writes FullAttention tensors only, so a Mamba id there would zero the FA block with the same id. `attention_groups` is required. `Phase1Lease` keeps M and the group pools (not compared, not on the wire): `mamba_index` = ceil(e/M) - 1, `block_keys()`. `NativeQueryForward` parses with the startup layout and refuses claims that alias the lease per (pool, id).
4. **Ownership.** `Phase1Coordinator` allocates FA from pool 0 and GDN slots from the Mamba pool, checks and frees per pool, records incarnations per pool. `PrivateBlockOwner` stays on pool 0 and checks handover/adopted blocks are its pool's objects before pinning or freeing. `pool_incarnations.rows` takes per-group pools; `EngineRpcBoundary` tracks every pool.
5. **No per-Page digest at P < 784.** Decoupled completions publish sealed Pages as `{token_span: [pP, (p+1)P], namespace, binding, source, content}` instead of `token_sha256` (the binding identity plus the span names the tokens; about 500 SHA-256 per turn at P = 16 removed). The engine checks the span. **WP3** must emit `token_span` in `allocation_owner._complete` when the startup record has `kv_layout`; **WP4**: worker frames and native evidence carry `token_span` in decoupled runs. Legacy records are unchanged. No other hashing was added; the existing per-request digests (lease digest and MAC, request and plan digests) stay per request, though their inputs grow with the Page count.
6. **Capacity (D7).** `turn_need_by_pool`/`headroom_by_pool` give (FA, GDN) blocks; legacy `turn_need`/`headroom` return their sums and `assess`/`admission_gate` keep the scalar dict and text. Decoupled `assess` returns `pools=('FA','Mamba')` and pairs for `free` (`get_num_free_blocks_by_pool`), `phase1`, `phase2`, `headroom`, `watermark` (FA only); the gate applies the legacy inequalities per pool and names the pool and both free counts. A running request's crossing slot is counted while it may still cross an M block.
7. **Bounds.** The catalog bound is `cdiv(max_model_len, P)` (84 at 784, 4096 at 16; 4096 when a fixture manager has no `max_model_len`). CANDIDATE_RANGE holds at P = 16 for C = 0, 1, P-1 (mod P) (tested).
8. **Engine per-step rows (addition, H1's engine half).** `dispatch_mixed_request` rebuilt the whole FA row every step (compose, reorder, prefix check, incarnation rows): 2.5 ms per request per step at P = 16 / 64K here (0.12 ms at 784). Decoupled only: the first allocation composes and publishes as before; later steps check and append the new transients, spot-check the published head, rebuild the short Mamba rows, and send identical `allocation_update` rows: 0.08 ms. **Open (cross-WP):** I2 still sends the full rows every step, so the worker side stays O(row) per step (`_allocation_update`, `_observe`); a delta form (appended FA entries + Mamba rows) would need WP2 and WP3 together.
9. **Unchanged:** `remote_ra_live_v2/startup.py`, `query_selection.py`, `selection_contract.py` (budgets already scale with `page_tokens`), `RowLayout`.
10. **Tests.** `test_decouple_wp2.py` ((P, M) matrix x both group orders, two-pool engine harness, property tests, launcher/config) and `test_decouple_legacy.py` (the 8586dbd source of every changed function, verbatim, compiled into its module: stage-B e2e for four policies, batched worker, two-session engine with capacity needs, gate, launcher, config, startup wiring identical). Both run without Git history; the e2e parts need CPU torch. Installer counts on this branch: (291, 195).
11. **Needs GPU.** vLLM's parsing of `--additional-config` (dict expected; text tolerated), acceptance of `--block-size 784` / `--mamba-block-size` in decoupled mode (WP1 item 9).

## Integration (branch `exp-decouple-int`)

1. **Merges.** WP1 → WP2 → WP3 → WP4, each `--no-ff`. The disjoint file ownership held: conflicts only in the release metadata (re-derived with the closure-aware regen), the installer's exact counts (now (292, 201): `kv_layout.py` and `page_compaction.py` added, WP1's five anchors in the payload) and this file's Deviations (both sections kept). The full suite passed after the WP2 merge (388 tests) and after all four (468); the integrated head runs 482.
2. **Interface fixes.**
   - `serve.py`: a decoupled query server's contract carries `vector_format = vector_format_for(kv_layout)` (`compact_int8_v1`); without it the worker-attach check refused it. Legacy builds the contract once, unchanged.
   - Phase 1 and completion accept WP3's compact Page vectors when decoupled: `Phase1Result.representation` (default the legacy float rows), `Phase1Coordinator._result` (compact Pages plus float queries, `query_representation`), the adapter's completion vectors (`page_compaction.from_wire`). Either layout refuses the other representation.
   - `NativeQueryForward.after_forward` keeps a `ProbeRows` when decoupled: a device copy of the Page-closing rows (compacted there by `model_queries`), the last row on the host and the finiteness of every row; legacy still copies every row (about 133 MB per 6.5K observation at 5,120 fp32).
   - The phase-1 alias check was already keyed by (pool, id) in WP2 (`fc69eb6`, tested); no change.
   - `check_git_release.py` allows the suite 1200 s and runs it with `OMP_NUM_THREADS=1` (840 s, as on `exp-subpage-score`, before the admission merge).
3. **Release metadata.** Pins regenerated once at the end. `patches/v0.24.0.diff` is not regenerated: the repository has no generator for it, and it is already stale on `main` (the `core.py` / `core_client.py` edits of 35bcf77, 9b18421 and b1e6988 are missing).
4. **CPU end to end** (`test_decouple_e2e.py`): the real engine, worker, Selector V2 and evidence over the stage-B fakes generalised to (P, M) (two pools when decoupled, Mamba slot ceil(end/M) - 1, FA-only zeroing, M-token Mamba views, the owner's batched Page entry points), (P, M) ∈ {(784, 784) legacy, (784, 784) decoupled, (16, 784)}, four policies, rebuild, cancel, and three sessions decoding in one batch. Legacy and decoupled (784, 784) decide, restore and attend identically; their evidence differs only in `kv_pools`, `token_span` for `token_sha256`, the compact vectors and representation fields, FA block ids, the digests and MACs over those, and the contract digest. On this workload the int8 vectors of both layouts are identical (the fp32 scales differ by ≤ 2e-7 relative); WP3 measured one element of ~0.35 % of vectors at ±1 on 5,120-wide rows, which can reorder near-ties in the query policies. The legacy case on 902690a's overlay (no `kv_layout` module: the pre-decoupling wiring) produces the same RPC payloads, receipts and evidence files as this tree, as do the existing stage-B, batched-worker and two-session scripts (checked across processes, pid-derived run ids aside).
5. **For the GPU smoke.** Not covered on CPU: vLLM's parsing of the three flags (WP1 9, WP2 11), the real TensorIO/PageArena and the device compaction, CUDA ordering. Decoupled Goal forwards and cold requests start on a Mamba slot nobody zeroes (the FA-only zero list): correct only because vLLM ignores the initial state at position 0 (`has_initial_state` false), which job B/C confirms. Random50 at P = 16 seeds from a 16-token block hash, so its selections are not comparable with legacy.
