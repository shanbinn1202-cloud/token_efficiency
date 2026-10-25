# WP4: selection contract, evidence and the GPU smoke (decisions, 2026-10-04)

Branch `exp-decouple-wp4` (from `exp-decouple-pages` 8586dbd; WP1's I0 `360ebd9` cherry-picked). Design: `DESIGN.md` in this directory. The GPU smoke harness lives outside the repository: `development/exp-decouple-pages/smoke-dp1/` (README there).

## 1. Selection contract at P

- The Page size is the contract parameter it already was. `default_contract(16)` is the PoC contract: recent quota `ceil(64·16/16) = 64` Pages, capacity `min(n, max(1, mandatory, floor(0.4·16n)//16))`, switch threshold 8,192 tokens. For every n ≤ 4,096 this equals the PoC's `_plan` (`max(16, 16·mandatory, floor(0.4·16n)) // 16`, tested).
- Mandatory Pages keep the current "every Page the system + task prompt touches" (cover). The PoC's "fully inside" would be a new enum value (open question 2).
- At P = 16 the "final 16-token upstream Page" of a native Page is the Page itself (PageRecord spans tested). The live tail is below P (`n = floor(o/P)`).
- **Vector format.** A decoupled-layout contract carries `vector_format: "compact_int8_v1"`; the legacy contract has no such key, so its bytes and digest are unchanged (`fe0b47e0…`, pinned in `test_decouple_selection.py`). `selection_contract.vector_format_for(kv_layout)` gives the value; `check_layout(contract, kv_layout, P)` ties format and Page size to the layout and runs once at worker attach (startup only).

## 2. Compact vectors (interface I3)

- Wire form per Page: `[[q_0, …, q_127], scale]` (128 ints in [−127, 127], not all zero; finite positive float scale), as WP3's `page_compaction.to_wire` writes it (`publish_vectors` and the phase-1 receipt, `representation: "compact_int8_v1"`).
- `QuerySelectionAuthority` under a compact contract parses Page vectors with `selection_contract.parse_compact` and never projects a Page; the catalog holds only `CompactVector`s (H3, H5). A float Page row is refused in compact mode, a compact one in legacy mode (no silent mixing).
- Query vectors (observation, Goal) may stay float (WP3 keeps them float): the engine compacts that one vector per turn (`selection_contract.compact`, fp32 CPU). A compact query is used as is (projection hash from `PROJECTION_SHA256[hidden]`).
- `selection_contract.compact_batch` / `compact_wire` are a batched fp32 helper (equal to the per-row path within ±1 per element, same top ranking); WP3 implemented its own float64 batch compaction, which the runtime uses.

## 3. Hashing (R4: no new per-request or per-step hashing)

| Digest | At P = 16 | Choice |
|---|---|---|
| Worker receipt HMAC over published vectors (`publish_vectors`, phase-1 receipt; WP3) | one MAC per publication, key minted at worker startup | Keep it as the per-turn, startup-keyed MAC over the compact batch. No per-Page MAC. Bytes MAC'd per turn fall: ~406 Pages × ~0.6 KB ≈ 0.25 MB for a 6.5K observation, against ~0.8 MB of float rows for 8 Pages at 784. |
| Query receipt HMAC (`plan_turn` / `verify_reservation`) | one per turn, body ≤ 4,096 scores + Page turns (~120 KB) | Keep (per turn, authority key from startup). |
| Upstream Selector V2 refs (`vector_ref`, `content_hash`, `provenance_hash`) | were ~7 SHA-256 + JSON per Page per plan | `query_selection.memo_types`: content and provenance hashes once per vector object, one index per session with each Page published once. Left per plan: one `vector_ref` per Page in the upstream snapshot check and one per selected Page in its reasons (upstream algorithm, kept). Plan time at 600 / 1,500 / 4,096 Pages: 26 / 44 / 118 ms instead of 94 / 201 / 554 ms (x86 desktop, CPython; Grace is slower per Python step). |
| Per-Page `token_sha256` in `new_pages` (WP3 `allocation_owner`, WP2 adapter) | ×49 calls, same bytes per turn | WP2 records `token_span: [pP, (p+1)P]` instead in the decoupled layout (legacy keeps `token_sha256`); the smoke's `page-evidence` check requires exactly that per layout. |
| Random50 ordering (one SHA-256 per Page per turn) | 4,096 at 64K | Keep: it is the Random50 policy. |
| Random50 seed = vLLM's last block hash | block = P tokens when decoupled | Comment in `private_endpoint.py`; no extra hashing. |

## 4. Evidence (additive, decoupled only; legacy byte-identical)

| File | Field | Value |
|---|---|---|
| `worker-attached.json` | `kv_pools` | `{kv_layout, num_blocks_by_pool, free_blocks_by_pool}` at attach |
| `ready.json` | `kv_layout`, `num_blocks_by_pool`, `free_blocks_by_pool` | the startup layout (inner I0 dict), both pool sizes (FA, Mamba), the readiness baseline |
| `native-*.json`, `runtime.jsonl` row | `kv_pools.free_blocks_by_pool` | both pools after that request's drain (a retried terminal keeps its first sample) |
| `engine-cleanup.json` | `kv_pools` | sizes and free blocks once every session closed (the smoke requires the readiness baseline) |

Counts come from `kv_manager.kv_cache_config.num_blocks_by_pool` and `kv_manager.get_num_free_blocks_by_pool()` (interface I1, pool 0 = FA, pool 1 = Mamba): two integers per pool at startup, per terminal and at teardown; nothing is hashed. `test_decouple_evidence.py` pins the SHA256 of every file of one deterministic legacy server life as 8586dbd's module writes them. `restore.h2d_bytes` / `prepare.h2d_bytes` keep their definitions.

**Evidence volume (open question 6).** Kept in full: `scores` and `page_turns` of every candidate Page in the signed query receipt (native file only), `selected_pages` in the row. Bounded by n ≤ `cdiv(max_model_len, P)` = 4,096 entries; about 120 KB per native file at 64K. A top-k would change the schema (R6); left to the algorithm side.

## 5. Integration notes

1. **serve.py (WP2)** must build the query policies' contract with the layout's vector format: `contract.default_contract(page_tokens, …, vector_format=contract.vector_format_for(kv_layout_record))`. WP2's final `exp-decouple-wp2` (1738f4e) still calls `default_contract(page_tokens, …)` only, so the worker-attach check refuses a decoupled query server ("selection contract vector format None differs from the KV layout"). One line at integration.
2. **Phase-1 receipt (WP2)**: WP3 sends `representation: "compact_int8_v1"` (plus `query_representation`) and compact `page_vectors` in decoupled mode. WP2's final `phase1_coordinator._result` still requires `pre_final_norm_last_token_l2` and `contract.Phase1Result._vector` floats (and equal dimensions for Page and query vectors); both must accept the compact Page form in decoupled mode. `QuerySelectionAuthority.plan_turn` parses the compact Page vectors itself (`[list, scale]` pairs or tuples), so `Phase1Result` can pass them through.
3. **Completion vectors (WP2 adapter)**: `tuple(v)` of a wire pair gives `([q…], scale)`, which `publish_completion` parses; no change needed beyond the representation check if any.
4. The startup config's `kv_layout` is the inner I0 dict (WP2); `selection_contract.layout_fields` also accepts the whole `additional_config` form.
5. Release pins: `regen_pins.py` of WP1 (closure-aware) was used after the cherry-pick; WP4 touches no closure file.
6. The stage-B / multi-session end-to-end suites (`test_stageb_e2e.py`, `test_ms_*`, `test_host_trim.py`) run the real worker and engine fakes, whose vectors and pools are WP2/WP3 code; their (P, M) parameterisation with compact vectors belongs to the integration (it needs items 1-2). WP4's parameterised coverage is the contract, Selector V2 and evidence (`test_decouple_selection.py`, `test_decouple_evidence.py`).

## 6. Tests

- `test_decouple_selection.py`: I0 stub and its conformance with WP1's `kv_layout`; the contract over (P, M) ∈ {(784, 784) legacy, (784, 784) decoupled, (16, 784), (64, 832)}; compact vs float selections identical over the matrix (goal_query and observation_query, float and compact queries); the P = 16 numbers; one publication per Page and the hash bound; live tail < P; Random50 at 4,096 Pages.
- `test_decouple_evidence.py`: legacy files byte-identical (pinned), the decoupled fields over the matrix, pools at baseline after teardown, the startup refusals (contract format vs layout, two native pools), the terminal-retry rule.
