# WP4：选页合同、证据和 GPU 冒烟测试（决定，2026-10-04）

[English](WP4.md) | 简体中文

**状态（2026-11-08）。** 本页其余部分是 2026-10-04 的 WP4 记录。此后：
- WP4 已与其他 WP 一起合并，位于 `main` 上（`DESIGN.md` 的 Integration 一节）。
- 第 5 节的集成说明已在集成时完成：第 1 项（`serve.py` 用 `vector_format=contract.vector_format_for(record)` 构建合同）、第 2 项（`Phase1Result.representation`；`Phase1Coordinator._result` 接受 compact Page）和第 6 项（`test_decouple_e2e.py` 运行 legacy-784、decoupled-784 和 16-784）。
- GPU 冒烟测试以 smoke-dp1 作业 3308015、3308414 和 3308482 运行（未复核：这些作业的证据从未取回）。之后经过核验的作业 3321538 运行了 P = 16 的服务（`DESIGN.md` 的 Status 一节）。

分支 `exp-decouple-wp4`（从 `exp-decouple-pages` 的 8586dbd 切出；cherry-pick 了 WP1 的 I0 `360ebd9`）。设计：本目录下的 `DESIGN.md`。GPU 冒烟测试的 harness 不在仓库里：`development/exp-decouple-pages/smoke-dp1/`（README 在那里）。

## 1. P 下的选页合同

- 页大小仍是它原本就是的合同参数。`default_contract(16)` 就是 PoC 的合同：recent 配额 `ceil(64·16/16) = 64` 页，容量 `min(n, max(1, mandatory, floor(0.4·16n)//16))`，切换阈值 8,192 token。对每个 n ≤ 4,096，它都等于 PoC 的 `_plan`（`max(16, 16·mandatory, floor(0.4·16n)) // 16`，已测试）。
- 必保留页沿用现在的“system + 任务 prompt 触及的每一页”（覆盖）。PoC 的“完全落在其中”将是一个新的枚举值（开放问题 2）。
- P = 16 时，native Page 的“最后一个 16-token 上游 Page”就是该 Page 自己（PageRecord 的区间已测试）。live tail 小于 P（`n = floor(o/P)`）。
- **向量格式。** 解耦布局的合同带有 `vector_format: "compact_int8_v1"`；legacy 合同没有这个键，所以它的字节和摘要不变（`fe0b47e0…`，固定在 `test_decouple_selection.py` 中）。`selection_contract.vector_format_for(kv_layout)` 给出这个值；`check_layout(contract, kv_layout, P)` 把格式和页大小与布局绑定，只在 worker 接入时运行一次（仅在启动时）。

## 2. 压缩向量（接口 I3）

- 每个 Page 的传输形式：`[[q_0, …, q_127], scale]`（128 个取值在 [−127, 127] 内、不全为零的整数；有限的正浮点 scale），与 WP3 的 `page_compaction.to_wire` 写出的一样（`publish_vectors` 和阶段 1 回执，`representation: "compact_int8_v1"`）。
- 在 compact 合同下，`QuerySelectionAuthority` 用 `selection_contract.parse_compact` 解析 Page 向量，从不对 Page 做投影；catalog 中只有 `CompactVector`（H3、H5）。compact 模式拒绝浮点的 Page 行，legacy 模式拒绝 compact 的 Page 行（不会悄悄混用）。
- query 向量（observation、Goal）可以保持浮点（WP3 让它们保持浮点）：引擎每轮把这一个向量压缩（`selection_contract.compact`，CPU 上 fp32）。compact 的 query 原样使用（投影哈希取自 `PROJECTION_SHA256[hidden]`）。
- `selection_contract.compact_batch` / `compact_wire` 是一个成批的 fp32 辅助函数（每个元素与逐行路径相差不超过 ±1，排在前面的排序相同）；WP3 实现了自己的 float64 成批压缩，runtime 使用的是它。

## 3. 哈希（R4：不新增逐请求或逐步的哈希）

| 摘要 | P = 16 时 | 选择 |
|---|---|---|
| 对发布的向量做的 worker 回执 HMAC（`publish_vectors`，阶段 1 回执；WP3） | 每次发布一个 MAC，密钥在 worker 启动时生成 | 保留为对 compact 批次、每轮一次、密钥在启动时确定的 MAC。不做逐 Page 的 MAC。每轮做 MAC 的字节数下降：6.5K 的 observation 约 406 个 Page × 约 0.6 KB ≈ 0.25 MB，而 784 时 8 个 Page 的浮点行约 0.8 MB。 |
| query 回执 HMAC（`plan_turn` / `verify_reservation`） | 每轮一个，body 最多 4,096 个分数加 Page 轮次（约 120 KB） | 保留（每轮一次，authority 密钥来自启动时）。 |
| 上游 Selector V2 的引用（`vector_ref`、`content_hash`、`provenance_hash`） | 原为每次 plan 每个 Page 约 7 次 SHA-256 加 JSON | `query_selection.memo_types`：content 和 provenance 哈希每个向量对象只算一次，每个 session 一个索引，每个 Page 只发布一次。每次 plan 仍剩下：上游 snapshot 检查中每个 Page 一个 `vector_ref`，以及其 reasons 中每个选中 Page 一个（上游算法，保留）。4,096 个 Page 时 plan 用时：118 ms，而不是 554 ms（x86 台式机，CPython，commit `e5b926f`；Grace 上每个 Python 步更慢）。 |
| `new_pages` 中逐 Page 的 `token_sha256`（WP3 `allocation_owner`，WP2 adapter） | ×49 次调用，每轮都是相同的字节 | 解耦布局下 WP2 改为记录 `token_span: [pP, (p+1)P]`（legacy 保留 `token_sha256`）；冒烟测试的 `page-evidence` 检查按布局恰好要求这一点。 |
| Random50 排序（每轮每个 Page 一次 SHA-256） | 64K 时 4,096 次 | 保留：这就是 Random50 策略。 |
| Random50 种子 = vLLM 的最后一个 block 哈希 | 解耦时 block = P 个 token | 在 `private_endpoint.py` 中加注释；没有额外哈希。 |

## 4. 证据（只增不改，仅解耦布局；legacy 逐字节相同）

| 文件 | 字段 | 值 |
|---|---|---|
| `worker-attached.json` | `kv_pools` | 接入时的 `{kv_layout, num_blocks_by_pool, free_blocks_by_pool}` |
| `ready.json` | `kv_layout`、`num_blocks_by_pool`、`free_blocks_by_pool` | 启动时的布局（内层的 I0 dict）、两个池的大小（FA、Mamba）、READY 时的基线 |
| `native-*.json`、`runtime.jsonl` 行 | `kv_pools.free_blocks_by_pool` | 该请求 drain 之后的两个池（重试的终态保留它的第一次采样） |
| `engine-cleanup.json` | `kv_pools` | 所有 session 关闭后的大小和空闲 block（冒烟测试要求回到 READY 时的基线） |

计数来自 `kv_manager.kv_cache_config.num_blocks_by_pool` 和 `kv_manager.get_num_free_blocks_by_pool()`（接口 I1，pool 0 = FA，pool 1 = Mamba）：启动时、每个终态和关停时，每个池两个整数；不做任何哈希。`test_decouple_evidence.py` 固定了一次确定性的 legacy 服务运行中每个文件的 SHA256，即 8586dbd 的模块写出的样子。`restore.h2d_bytes` / `prepare.h2d_bytes` 的定义不变。

**证据量（开放问题 6）。** 完整保留：签名的 query 回执中每个候选 Page 的 `scores` 和 `page_turns`（只在 native 文件中），以及行中的 `selected_pages`。上限为 n ≤ `cdiv(max_model_len, P)` = 4,096 项；64K 时每个 native 文件约有 110 KB 的 `scores` 和 `page_turns`（作业 3321538：3,346 个 Page 时 87,924 B，按比例换算到 4,096 时为 107,632 B；那整个 native 文件为 1.87 MB）。top-k 会改变 schema（R6）；留给算法一方。

## 5. 集成说明

1. **serve.py（WP2）** 必须用布局的向量格式构建 query 策略的合同：`contract.default_contract(page_tokens, …, vector_format=contract.vector_format_for(kv_layout_record))`。WP2 最终的 `exp-decouple-wp2`（1738f4e）仍只调用 `default_contract(page_tokens, …)`，所以 worker 接入检查会拒绝解耦布局的 query 服务（"selection contract vector format None differs from the KV layout"）。集成时改一行。
2. **阶段 1 回执（WP2）**：解耦模式下 WP3 发送 `representation: "compact_int8_v1"`（外加 `query_representation`）和 compact 的 `page_vectors`。WP2 最终的 `phase1_coordinator._result` 仍要求 `pre_final_norm_last_token_l2` 和 `contract.Phase1Result._vector` 为浮点（并要求 Page 向量和 query 向量维度相同）；两者在解耦模式下都必须接受 compact 的 Page 形式。`QuerySelectionAuthority.plan_turn` 自己解析 compact 的 Page 向量（`[list, scale]` 对或元组），所以 `Phase1Result` 可以原样传过去。
3. **完成向量（WP2 adapter）**：对一个传输对取 `tuple(v)` 得到 `([q…], scale)`，`publish_completion` 会解析它；除了可能有的 representation 检查之外，无需改动。
4. 启动配置中的 `kv_layout` 是内层的 I0 dict（WP2）；`selection_contract.layout_fields` 也接受完整的 `additional_config` 形式。
5. Release pin：cherry-pick 之后使用了 WP1 自己的临时脚本 `regen_pins.py`（考虑 closure；不在本仓库或项目文件夹中保存）；WP4 不改动任何 closure 文件。
6. stage-B / 多 session 的端到端测试集（`test_stageb_e2e.py`、`test_ms_*`、`test_host_trim.py`）运行真实的 worker 和引擎假实现，其中的向量和池是 WP2/WP3 的代码；用 compact 向量对它们做 (P, M) 参数化属于集成（需要第 1-2 项）。WP4 的参数化覆盖的是合同、Selector V2 和证据（`test_decouple_selection.py`、`test_decouple_evidence.py`）。

## 6. 测试

- `test_decouple_selection.py`：I0 stub 及其与 WP1 的 `kv_layout` 的一致性；(P, M) ∈ {(784, 784) legacy、(784, 784) 解耦、(16, 784)、(64, 832)} 上的合同；在整个矩阵上 compact 与浮点的选页相同（goal_query 和 observation_query，浮点和 compact 的 query）；P = 16 的数字；每个 Page 一次发布以及哈希上限；live tail < P；4,096 个 Page 时的 Random50。
- `test_decouple_evidence.py`：legacy 文件逐字节相同（已固定），整个矩阵上的解耦字段，关停后池回到基线，启动时的拒绝（合同格式与布局不符，两个 native 池），终态重试规则。
