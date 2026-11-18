# 全注意力 Page 与 GDN 状态页解耦（设计，2026-10-04）

[English](DESIGN.md) | 简体中文

**状态（2026-11-08）。** 本页其余部分是 2026-10-04 的设计记录，之后是各 WP 和集成的说明。此后：
- **在 `main` 上。** 四个 WP 和集成已经合并（`5bfd77a`）。解耦布局仍需显式开启：`--page-tokens` 的默认值仍是 784。
- **GPU 上。** 已离线复核的性能作业 3321538（runtime `b37044f`，SPF1_PERF_PASS_VERIFIED）在 legacy 服务旁边运行了 P = 16、M = 784 的服务。READY 中的 `num_blocks_by_pool`：8 GiB、S = 1 时为 [7702, 10]；24 GiB、S = 4 时为 [23205, 28]。它的 P = 16 Goal 服务（`b-C-goal`、`l-C-goal`）通过了。它的作业 F 比较（R5）对 batch 段和 length 段报告了 OVERHEAD_ABOVE_BUDGET，对 length 段报告了 ADMISSION_SLOWER，所以 +0.1 ms 的目标在那里没有达到。之后的 GPU 作业使用同样的布局（例如作业 3493649：`--block-size 16 --mamba-block-size 784`）。第 8 节的第一次冒烟测试，即 `8f1af6a` 上的 smoke-dp1 作业 3308015、3308414 和 3308482，未复核：这些作业的证据从未取回。
- **开放问题（第 10 节）。** 已解决的：
  - Q1：P = 16 成为之后 benchmark 服务的布局，并且仍需显式开启。
  - Q3：解耦模式以 float64 成批压缩，与两条路径都不逐位相同（`page_compaction.py`）；legacy 保留自己的路径。
  - Q4：采用 M = 784 的 align 语义；F1 没有实现。
  - Q5：Mamba 池从 `--kv-cache-gib` 中划出（见上面的 READY 计数）。
  - Q6：分数全部保留（WP4.md 第 4 节）。
  - Q7：`token_span` 取代 `token_sha256`；每次发布对压缩向量做一次 HMAC（密钥在启动时确定），这一项保留。
  - Q9：connector 仍然注册（只注册池 0）；`--connector-pool-gib` 决定它那个不使用的池的大小，默认 1 GiB。
  - Q10：`exp-subpage-score` 从未合并，`--page-score` 不在 `main` 上。
  - 仍未解决：Q2（`main` 保留 cover 规则）和 Q8（F3 没有实现；Random50 的种子仍来自最后一个 block 哈希）。
- **偏差中需要 GPU 的几项**（WP1 第 9 项、WP2 第 11 项、集成第 5 项）已由这些运行解决：服务接受 `--block-size 16 --mamba-block-size 784 --additional-config ...`，打印“unpadded Mamba page”并到达 READY。
- **第 11 节**是一份历史清单。

分支 `exp-decouple-pages`，从 `main` 的 `0f91d78`（runtime `902690a`）切出。
读过的源码：`source-overlay/vllm/` 下的 overlay 副本。`patches/v0.24.0.diff` 中没有的文件是与上游 0.24.0 逐字节相同的“anchor”文件。另外读过：runtime 模块、`development/multi-session/capacity/CAPACITY.md`、`development/selection-contract-v7/stageB-*.md` 和 PoC 的 `reference/dram-backed-kv/scripts/serve_online_selector_v2.py`。
测量脚本在 `development/exp-decouple-pages/tools/`（系统 `python3`，x86 Windows，CPython；**不是** GH200/Grace 上的数字）。
`development/` 和 `reference/` 路径在项目文件夹中，不在本仓库里：只拿到本仓库（没有项目文件夹）的协作者打不开它们。模块路径（`remote_ra_linux/...`、`remote_ra_http/...`，以及 1.6 节和第 5 节中的简短模块名）都在 `source-overlay/` 下；`serve.py` 在顶层目录。

## 0. 概要

**决定。** 实现**方案 A**。全注意力（FA）KV 和 Gated-DeltaNet（GDN，vLLM 中称为“Mamba”）状态使用**两个 block 池，页大小互相独立**：
- FA 池：`P` token 的 block。一个 FA block 仍然恰好是一个 runtime Page，所以“每个 Page 一个 FA block”这一不变量仍然成立。
- Mamba 池：对每个 Mamba group，一个 block 存放一份不填充的 GDN 层状态；池的大小按每个服务固定数量的 GDN slot 确定。

group 结构、GDN kernel、注意力 kernel 和 align 模式都保持不变。legacy 共享池（`P = 784`）仍是默认代码路径，不做改动，逐字节相同。

**Page 大小。** 目标是 `P = 16`。
- 它就是 PoC 的 `PAGE_SIZE`，所以 Selector V2 合同与上游完全一致：预算 `floor(0.4·history)//16`，recent 配额 64 页，live tail 不足 16 token，每个 16 token 的 Page 一个向量。
- 它在注意力 kernel 中没有任何代价。GDN 状态为 fp32 时，FlashAttention 本来就以 **16 token 的 kernel block** 运行：每个 784 token 的 block 拆成 49 个 kernel block，所以 64K 时每个请求的 kernel block table 已经有约 4,100 项。
- 不是免费的部分：runtime 中每个 O(Pages) 的结构都变成 49 倍。这些结构必须向量化（第 5 节）。
- `P` 是取值于 {16, 32, 64, 784} 的参数。如果 16 时达不到主机侧预算，P = 64 是经测量的后备。

**GDN block 大小。** 它与 `P` 分开，记作 `M`（Mamba 对齐 block）。默认 `M = P·⌈784/P⌉`，P = 16 时为 784。
- 因此 GDN 的跨块节奏、捕获和 H2D 计数都与现在完全相同。
- `M = P = 16` 在本 runtime 中不可行：每次跨越 align slot 都要做一次 154 MB 的 D2H 加一次 154 MB 的 H2D（`qwen.py:246-252`、`worker_backend.relocate_latest_hybrid`）。
- `M ≥ max_model_len`（每个请求一个 GDN block，不跨块）是后续工作（D5）。

**工作量。** 四个工作包，合计约 10–14 个 agent 日；四个 agent 并行时实际耗时约 4–5 天。之后是集成和 1–2 轮依次进行的 GPU 冒烟测试。

## 1. vLLM 0.24 在 0f91d78 上如何布置 KV 和 GDN 状态（除非另有标注，均已在源码中核实）

本节的行号是 `0f91d78` 的行号。在 `main` 上它们已经移动（例如 `get_kv_cache_groups` 1697 → 1907，`unify_kv_cache_spec_page_size` 1049 → 1099）。

### 1.1 模型几何（Qwen3.6-27B，`remote_ra_linux/model-evidence/model-config.json`）
- 64 层：16 个 `full_attention` 层（每 4 层一个）和 48 个 `linear_attention` 层（GDN）。
- FA：4 个 KV head × 256 维，bf16。即**每层每 token 4,096 B**，16 个 FA 层合计每 token 65,536 B。
- GDN 每层：
  - 卷积状态：(16·128·2 + 48·128) × 3 × bf16 = 61,440 B；
  - 递归状态：48 × 128 × 128 × **fp32** = 3,145,728 B（`mamba_ssm_dtype: float32`）；
  - 合计 **3,207,168 B**，相当于每层 783 个 token 的 FA。
  - 注：一个 fp32 head 的状态恰好是 64 KiB，即每层一个 16 token 的 FA 页。

### 1.2 784 从哪里来（`platforms/interface.py:594-802`）
模型加载后，`uniproc_executor.py:69` 调用 `update_block_size_for_backend`。对混合模型，`_align_hybrid_block_size` 做三件事：
1. 计算 `attn_block_size = 16·cdiv(3,207,168, 16·4096) = 784`（`:766-769`），并把 `cache_config.block_size` 提高到这个值（`:771-777`）。这会覆盖用户给的 `--block-size`。
2. 在 align 模式下设 `mamba_block_size = block_size`（`:779-780`）。
3. 把 Mamba 页填充到 784·4096 = 3,211,264 B（`:782-802`，+0.13 %）。

所以 784 存在的唯一原因，是让同一个池里的一个注意力页和一个 Mamba 页大小相同。

### 1.3 group、tensor 和单一的池
- **统一页大小。** `get_kv_cache_groups`（`v1/core/kv_cache_utils.py:1697`）先调用 `unify_kv_cache_spec_page_size`（`:1049`），再调用 `_get_kv_cache_groups_uniform_page_size`（`:1108`）。
  - 结果：4 个各 16 层的 group：一个 FA group 和三个 Mamba group，48 个 GDN 层按 `layers[i::3]` 划分。在真实模型上 Mamba group 排在前面（group 0–2），FA 是 group 3，因为第 0 层是 GDN（legacy READY 的 `tensor_geometry`，作业 3300301；`model-config.json` 的 `layer_types`）。测试 fixture 把 FA 放在最前面（group 0）。
- **共享 tensor。** `get_kv_cache_config_from_groups`（`:1318-1400`）的通用分支：
  - `group_size = 16`，统一的 `page_size` → `num_blocks = mem // page // 16`；
  - **16 个 `KVCacheTensor`**，每个由一个 FA 层和每个 Mamba group 中的一层共享（每个 tensor 4 层）。
  - 8 GiB 时：167 个 block × 每个 block 行 51,380,224 B。
- **只有一个池。** `KVCacheConfig` 只有一个 `num_blocks`（`kv_cache_interface.py:880`）。`KVCacheCoordinator` 只建**一个 `BlockPool`**（`kv_cache_coordinator.py:91-97`），并把它交给每个 manager。block id 在所有 group 之间是同一个命名空间。runtime 在好几处依赖这一点（第 2 节，B）。
- **packed 布局存在，但没有帮助。** 上游还有一种多页大小的 packed 布局（`_get_kv_cache_config_packed`，`:1277`，DeepSeek-V4 和 `enable_cross_layers_blocks`）。它仍只用一个 block id 空间，每个 id 都为每种页大小预留字节，所以无法把池拆开。

### 1.4 runner 布局与 kernel block
- **分配。** `_allocate_kv_cache_tensors`（`gpu_model_runner.py:7325`）为每个 `KVCacheTensor` 分配一个 int8 缓冲区。
- **reshape。** `_reshape_kv_cache_tensors`（`:7377-7486`）**逐层**推出 `num_blocks = numel // page_size_bytes`：
  - FA 视图：`(kernel_num_blocks, 2, kernel_bs, 4, 256)`；
  - Mamba 视图：`as_strided((num_blocks, *state_shape), stride=page_size_bytes)`，在一个页内先放 conv，再放 ssm。
  - 因为 reshape 是逐层的，即使 FA 层和 Mamba 层位于大小不同的不同 tensor 中，它也已经能工作。
- **kernel block 大小。**
  - Mamba（ssm）cache 的 dtype 为 fp32 时，`FlashAttentionBackend.get_supported_kernel_block_sizes` 返回 `[16, 32, 64]`（`flash_attn.py:77-94`，FA issue #1974）。
  - 因此 `select_common_block_size(784, …)`（`v1/worker/utils.py:262`）选的是 **16**。
  - `BlockTable` 把每个 784 token 的分配 block 拆成 49 个 kernel block（`block_table.py:50-65, 174+`）。
  - *这里是根据 dtype 推断的。legacy READY 回执（作业 3300301，`tensor_geometry`）证实了这一点：注意力视图的 `allocation_tokens` 为 784，`kernel_tokens` 为 16，`ratio` 为 49。*
  - 结果：64K 时每个请求的 kernel block table 已经有 84·49 = 4,116 项。改用 16 token 的分配 block 后是 4,096 项。**注意力 decode 的开销和 block table 的上传大小都不变。**

### 1.5 Mamba align 模式、prefix caching、connector
- `serve.py:83-84, 99-100` 传入 `--enable-prefix-caching --mamba-cache-mode align`。**prefix caching 是开着的，不是关着的。**
  - 原因是 OffloadingConnector：`kv_offload/base.py:486-498` 要求每个 group 的 block 大小都能被 `hash_block_size` 整除，并建议混合模型开启 prefix caching。
  - 自有（Stage-B）请求从不使用 prefix caching：`get_computed_blocks` 返回空，`cache_blocks` 什么都不做（`kv_cache_manager.py:679-705`）。
  - connector 是隔离的：查找返回 0，也不产生任何 job（`remote_ra_http/connector.py:5-32`）。
  - 它在注册时仍会分配一个**锁页 CPU 池**（`kv_offload/cpu/gpu_worker.py:493-519`），而这个池不被使用。这里它是 32 GiB。在 `main` 上，速度默认值把它设为 1 GiB（`--connector-pool-gib`）；只有加 `--no-speed-defaults` 或 `--no-connector-pool-gib` 时才是 32 GiB。
- **align 模式**（`single_type_kv_cache_manager.py:958-1291`）：
  - Mamba block table 宽 `cdiv(max_model_len, M)`（84 项），大部分是 null block。
  - 运行中的状态位于下标 `(seq_len-1)//M`（`attention/backends/utils.py:899-939`）。
  - 跨块时，`preprocess_mamba` 把状态拷进新 block（`v1/worker/mamba_utils.py:678-742`），旧 block 在一步之后释放（`MambaManager.remove_skipped_blocks`）。
  - 所以一个请求在每个 Mamba group 中占 1–2 个 block，即一个 GDN “slot” = 3 个 block（填充后的池中 154,140,672 B），存放 153,944,064 B 的状态（跨块时另加 1 个临时 slot）。
- **chunk 切分。** `_mamba_block_aligned_split`（`sched/scheduler.py:342-397`）在 **`cache_config.block_size`** 的整数倍处切分 prefill chunk（`:369`）。
  - 上游假设 align 模式下 FA block = Mamba block。`core.py:316` 设置 `block_size = min(group block sizes)` 之后，解耦配置会在 P 而不是 M 处切分（必须打补丁，WP1）。
- **哈希。** `resolve_kv_cache_block_sizes`（`kv_cache_utils.py:607-670`）：
  - `scheduler_block_size = lcm(group sizes)`；
  - `hash_block_size = gcd(group sizes)`，**除非某个 Mamba group 的 block 大小 ≠ `cache_config.block_size`**，这时它退回到 lcm（`:653-658`）。这种退回会让上面 connector 的断言失败。
  - 请求的 block 哈希（pickle 后的 784 token chunk 的 sha256）在引擎核心中逐请求计算（`core.py:225`）。Random50 的种子来自最后一个哈希（`private_endpoint.py:109-110`）。

### 1.6 runtime 在其上构建了什么（来自两份调研）
- **Page。**
  - `P = 784` 固定在 `serve.py:41,87` 和 `remote_ra_http/config.py:45` 中。它只是与 vLLM *核对*：`worker_composition.py:112-116`、`native_growing/adapter.py:98-101`、`phase1_coordinator.py:52-54`、`sparse_mapping.py:86`。
  - 所有地方都是每个 FA group 每个 Page 一个 FA block：`ownership.py`、`allocation_owner.py`、`contract.py`、`engine_boundary.py`。
  - FA 行是 `[selected][run][native transients]`，配 `SparsePlan(L, L-|A_t|·P)`。位置保持逻辑位置；只有 FA slot 被压紧。
- **GDN。** GDN 状态通过锁页主机内存在轮次之间**连续地**传递（`worker_backend.py`、`remote_ra_growing_worker/tensors.py`）：
  - carry 在 C′（轮次结束）处捕获；
  - 以 H2D 恢复到阶段 1 的 slot 中（`prepare.h2d_bytes` = 153,944,064 B）；
  - 阶段 1 的状态被捕获，并恢复到阶段 2 的 slot 中；
  - 每个**与 Page 对齐**的 forward 结束时捕获最新状态，`preprocess_mamba` 移动 slot 时从主机恢复（`qwen.py:179-192, 246-252`）。
  - 这些都假设 Mamba block == P。阶段 1 和 Goal 的 lease 构造长度为 ⌈e/P⌉ 的 Mamba 行，并要求 slot 下标为 ⌈e/P⌉−1（`contract.py:344-347, 405, 445-449`；`native_probe.py:147-150`）。
  - 轮次之间不占用任何 Mamba block（CAPACITY.md 的 §Method 1）。
- **DRAM catalog。** worker 私有，键为 `(session namespace, Page)`。每个 payload 是每个 FA group 一个 block，含全部 16 层。每一个 block 片段都有**自己的锁页分配和一次同步拷贝**（`tensors.py:310-386`）。
- **容量 gate**（`remote_ra_live_v2/capacity.py`）：`A·pages + M·slots` 对比**一个** `block_pool.get_num_free_blocks()`。
- **轮次几何。** 它要求 `n = ⌊o/P⌋ ∈ [Q/P, Q/P+1]`（`native_growing/contract.py:218`、`live_v2/contract.py:143-145`）。
  - 按构造，`o − C = 1 + |boundary| = 2 or 3` 个 token（`mswea_protocol.py:117-119, 186-195`），所以它对每个 P ≥ 4 都成立，包括 16。*这是推导结果；WP2 在 C ≡ 0、1 和 P−1 (mod P) 时测试它（`tests/test_decouple_wp2.py`）。*

## 2. 需求与不变量
1. **R1。** FA Page 是 P token（目标 16）。GDN 状态仍是每个请求一份连续的状态：不分页，不拆分，像现在一样在轮次之间传递。
2. **R2。** 不加新参数的 `P = 784` 就是 legacy 路径：
   - vLLM argv 和 `effective_server_config` 相同；
   - KV 配置、token 和证据相同；
   - 所有现有测试不变。
3. **R3。** 多 session 服务（当时 ≤ 8 个 session；在 `main` 上为 16 个，chat-text 服务上为 24 个；FIFO admission、成批 decode、gate、**无抢占**）继续可用。gate 必须覆盖两个池。
4. **R4。** 不增加逐请求或逐步的哈希。现有的逐 Page 摘要不能变成热路径上的开销（第 5 节）。
5. **R5。** 每步的主机开销不能随 Page 数增长。稳态目标：64K 时相对 `main` 每个请求每步 ≤ +0.1 ms。
6. **R6。** 阶段 1/2 的语义、选页策略的语义（以大小为 P 的 Page 计）、证据 schema 和干净关停回执都不变。新字段只做增加。

## 3. 方案

| | A. 两个池（推荐） | B. 一个池，GDN 状态跨越许多小页 | C. 保留 784 的物理 block，runtime 再细分出 16 token 的 kernel block | D. 在 784 token 的常驻之上使用 16 token 的注意力行 | E. 分页的逐 head GDN 状态（把 B 做彻底） |
|---|---|---|---|---|---|
| 思路 | 池 0 中是 P token 的 FA block；池 1 中是 GDN slot，按 session 数确定大小 | GDN 状态 = 49 个*连续且对齐*的 16 token block（或一块预留区域） | 每个 784 block 是 49 个连续的 64 KiB kernel block；runtime 把逻辑上 16 token 的 Page 装进 slot，并写 kernel 级的行 | 行只列出选中的 16 token kernel block；常驻、DRAM 和恢复仍按 784 的 Page 进行 | 每层 48 个 head 页 + 1 个 conv 页；kernel 接受逐 head 的状态下标表 |
| vLLM 文件 | `platforms/interface.py`、`v1/core/kv_cache_utils.py`、`v1/kv_cache_interface.py`、`kv_cache_coordinator.py`、`kv_cache_manager.py`、`sched/scheduler.py`、offloading 的 `worker.py` + `cpu/spec.py`（+ connector scheduler 的 group 过滤）；runner 的 reshape 不变 | 与 A 相同，另加 `block_pool.py` 中的连续分配器，以及覆盖别名区间的 Mamba 视图 | 无，只有一个绕过 `BlockTable.map_to_kernel_blocks` 的 kernel 级行安装（`block_table.py`、`worker_rows`） | 无 | 所有 GDN Triton/CuTe kernel（`fla/ops/*`、`fused_recurrent`、`chunk_delta_h`、`causal_conv1d`、`gdn_chunk_cutedsl/*`）、`gdn_attn.py`、`mamba_utils` 的拷贝 |
| kernel 约束 | FA kernel block 16（现在已经如此）；GDN kernel 不动 | GDN 的 `[num_blocks,…]` 视图需要连续性 | kernel block 16（已经如此）；slot mapping 来自 kernel 行 | 与 C 相同 | 重写 kernel |
| 64K 时的 block table | FA 4,096 个分配项 = kernel 项（现在 84 个分配项 / 4,116 个 kernel 项）；Mamba 84 | 同 A | 4,096 个 kernel 项 | ≤ 4,116 | 同 A |
| 内存开销 | Mamba 池静态预留：(2S+1) 个 slot。S=1 时占 KV 预算中的 0.51 GB，S=8 时 2.67 GB。没有填充。FA 尾部浪费 < 16 token | 碎片，或一块预留区域，那就等于 A | 部分使用的 784 block 内部的 slot 碎片（需要压紧拷贝） | 没有内存收益：一个 784 Page 的 49 个子页中只要有一个被选中，它就要常驻 | 同 A |
| 多 session 容量 | FA token 总数相同（备用 slot 占去 −1–2K，见 4.4 节的表）。常驻粒度更细；双池 gate | 同 A | 取决于碎片 | 与现在相同 | 同 A |
| DRAM / 恢复 | 需要一个以 P 为粒度的锁页 arena 和成批拷贝（A、B、C、E 都需要） | 同 A | 同 A，另加压紧 | 不变（784） | 同 A |
| 风险 | 中：边界清楚的 vLLM 补丁；runtime 中 O(Pages) 的热点 | 高（分配器、碎片） | 高（新的分配器；native 行增长仍以 784 为粒度；kernel 行与分配行混合） | 低/中，但只提供注意力粒度的语义 | 很高 |
| 工作量 | 10–14 个 agent 日 | > A | ≈ A 或更多，且有更多新的不变量 | 4–6 个 agent 日 | 数周 |

**结论。**
- **A** 是唯一一个不需要改 kernel、也不需要新分配器，就能以 P 为粒度做常驻、恢复和选页的方案。它还保留 vLLM 自己的分配器、scheduler 和以 Page 计的行增长。
- 如果算法方只是近期需要 PoC 的注意力语义，**D** 是更便宜的过渡方案。它浪费恢复字节，而且 `exp-subpage-score` 已经在探索相邻的方向。
- 如果 vLLM 的池补丁被排除，**C** 是后备方案。
- **B** 和 **E** 被否决。

## 4. 推荐设计（决定）

### D1. 启用方式与 legacy 路径
- **参数。** 一个新的 `--additional-config '{"remote_ra_kv_layout": {...}}'`（标准的 `VllmConfig.additional_config`，`config/vllm.py:358`）。它原样到达引擎核心和 worker。
- **没有这个键就是 legacy：** 每个打过补丁的函数都走上游分支，`serve.py` 输出的 argv 与现在逐字节相同。
- **共享 helper。** 新的纯 Python 模块 `source-overlay/remote_ra_live_v2/kv_layout.py` 保存布局（接口 I0，6.0 节）。负责方：WP1；API 由本文档冻结。vLLM 补丁延迟导入它，就像 `kv_cache_manager.py` 已经导入 `remote_ra_live_v2.chunk_protocol` 那样。

### D2. P = 16，参数化为 P ∈ {16, 32, 64, 784}
为什么是 16：
- 与 PoC 一致：`PAGE_SIZE = 16`，live tail < 16，必保留页/recent 页/预算都以 16 token 为单位（`selection_contract.default_recent_quota_pages(16) = 64`）。
- 没有 kernel 开销（1.4 节）。
- 它消除了文档记载的偏差“live tail 最多 783 token”（`stageB-design.md` D2，`stageB-native-map.md` Q4），预算也改以 16 token 而不是 784 token 的 Page 计。

为什么不更小：FlashAttention 的 kernel block 必须是 16 的倍数。

为什么 32/64 只作后备：它们把 O(Pages) 的主机工作减少 2–4 倍，但偏离 PoC。

约束：P 必须是 16 的倍数（FA 和 `selection_contract.validate`）。P ≠ 784 意味着解耦布局。`P = 784` + 解耦可以作为**验证模式**，用来单独检验池的拆分。

### D3. 与 P 解耦的 Mamba 对齐 block `M`
- **规则：** `M = P·⌈784/P⌉`，所以 P = 16 / 32 / 64 / 784 时 M = 784 / 800 / 832 / 784。
  - M 是 P 的倍数，所以 `hash_block_size = P` 能通过 `kv_offload/base.py:493`。
  - M 是 16 的倍数，满足 causal_conv1d 的要求（8 的倍数）。
  - M ≤ `max_num_batched_tokens`。
- **P = 16 时**，M = 784 让 GDN 的行为逐位不变：
  - 跨块和 prefill chunk 边界相同（`_mamba_block_aligned_split` 必须使用 M）；
  - 捕获最新状态的节奏相同（`qwen.py` 必须使用 `end % M`）；
  - H2D 计数相同，Mamba table 宽度相同（84）。
  - 所以 P = 16 下的 `full` 必须与 legacy 逐 token 一致（第 8 节）。
- **否决：`M = P = 16`。** 那样每 16 token 就跨一次块，在本 runtime 中每次都要做一次 154 MB 的 D2H 捕获加一次 154 MB 的 H2D 搬移。
- **推迟：`M ≥ max_model_len`**（D5）。

### D4. 池与 tensor
- **group。** 与现在完全相同：spec 和层→group 的分配都一样。唯一的区别是解耦分支**跳过 `unify_kv_cache_spec_page_size`**。
- **池 0（FA）：**
  - 16 个 `KVCacheTensor`，每个 FA 层一个，每个 `N_fa × P·4096` B；
  - `N_fa = (budget − mamba bytes) // (P·65,536)`；
  - `num_gpu_blocks_override` 作用于这个池。
- **池 1（Mamba）：**
  - 16 个 tensor，每个由三个 Mamba group 的第 i 层共享（与现在相同），每个 `N_m × 3,207,168` B，**不填充**（`mamba_page_size_padded = None`）；
  - `N_m = 1 + 3·K`，其中 S = `max_num_seqs` 时 `K = 2S + 1` 个 slot：每个运行中的请求一个运行 slot 加一个跨块 slot，另加一个备用。这足够容纳阶段 1 加一次 Goal forward（2 个 slot），同时其余 S−1 个请求各占 2 个。
- **block id** 按池编号，不同池之间可以重复。每个池有自己的 null block 0。
- `KVCacheConfig` 增加 `group_pool: tuple[int,…] | None` 和 `num_blocks_by_pool: tuple[int,…] | None`。None 表示 legacy。`num_blocks` 仍表示池 0。

### D5. 后续工作（不在第一版实现范围内）
- **F1. `M ≥ max_model_len`。** 每个请求每个 group 一个 GDN block，不跨块。每次跨块省下约 308 MB 的拷贝（2 × 153,944,064 B）。在测过的负载上跨块很少：作业 3300301 的 20 个延续轮次中有 2 次，作业 3321538 中 legacy `full` 服务的 9 轮中 0 次，大约十轮一次。这个比例取决于这些合成 session 的轮次长度：回复长时更多。
  - 需要：scheduler 对 block 大小的处理（`lcm` 变成 ≥ 65536）、去掉哈希退回、`num_tokens == M` 时 align 切分的边界情况，以及 runtime 的节奏。
  - 它需要单独验证，因为 GDN 的证据会变。
- **F2。** 在解耦模式下去掉 OffloadingConnector（释放它的锁页池：这里是 32 GiB，`main` 上默认 1 GiB）。它改变启动约定（`isolate_connector`、`connector_guard`），所以是一个开放问题。
- **F3。** 对自有请求跳过 vLLM 的 block 哈希。Random50 的种子就需要另找来源。

### D6. 解耦模式下的 offloading connector
它仍然注册并保持隔离，但其注册、CPU 容量计算和 group 配置都**排除 Mamba group 和 tensor**。否则：
- `offloading/worker.py:54,116-129` 会在 Mamba 存储上建 `(num_blocks_fa, page)` 视图，这会抛出异常；
- `cpu/gpu_worker.py:493-519` 会为每个 Mamba tensor 分配 `num_cpu_blocks × 3.2 MB` 的锁页内存。CPU block 数按 1 MiB 的 FA block 计算时，这约为 1.7 TB（32,768 个 CPU block × 16 个 tensor × 3,207,168 B = 1.68 TB），分配会失败。

### D7. 双池容量 gate
结构与 `capacity.py` 相同，按池分别计算：
- FA：`A·pages` 对比 FA 的空闲 block。
- Mamba：`Mg·slots` 对比 Mamba 的空闲 block。
- headroom 和 watermark 按池计算（与 vLLM 一样，watermark 只作用于 FA）。

N_m 按 D4 取值时，≤ S 个 session 下 Mamba 的不等式从不起约束作用，但仍然检查。legacy 模式下 gate 就是现在的单一总和，消息也相同。

### D8. 小 P 下的主机侧数据通路（P = 16 时必需；第 5 节）
- 锁页 DRAM arena，配合成批的 gather/scatter 拷贝；
- 在 device 上成批压缩 Page 向量；
- 端到端使用压缩的 int8 向量；
- 增量安装 FA 行。

### 4.4 容量数字（`tools/capacity_numbers.py`）

| KV 预算 | S | legacy block 数（×784） | legacy 在 GDN 用量峰值时的 FA token 数 | 解耦的 Mamba 池 | 解耦的 FA token 数（P=16） |
|---|---|---|---|---|---|
| 8 GiB | 1 | 167 | 125.4K | 10 个 block，0.51 GB | 123.2K |
| 8 GiB | 8 | 167 | 92.5K | 52 个 block，2.67 GB | 90.3K |
| 24 GiB | 4 | 501 | 373.2K | 28 个 block，1.44 GB | 371.3K |
| 32 GiB | 8 | 668 | 485.3K | 52 个 block，2.67 GB | 483.6K |

- **FA token 的原始容量**不变，差距在约 2K token 以内（备用 slot）。
- **容量收益来自粒度。**
  - 一个部分填充的 Page 浪费 < 16 token，而不是 < 784。按每个运行中的 session 计，现在约是 1–3 个 run/tail Page × 51 MB。
  - 已释放但在 commit 之前仍被钉住的 Page（CAPACITY.md）粒度更细。
  - 选页正好用完 40 % 的预算。
  - 16 token 下相邻两次选页之间的变动可能增加也可能减少恢复量。**无法预测；需要测量。**
- **主要动机是忠实于 PoC，而不是容量。**

## 5. 随 Page 数增长的热点（P = 16 时必须解决）

| # | 位置 | P = 784 | P = 16 | 需要的改动（WP） |
|---|---|---|---|---|
| H1 | 每步重组 FA 行：`worker_rows.install_row` → `verify_native_row`×2 + `compose_rows` | 64K、40 % 选页时测得每个请求每步 0.39 ms（`tools/rowbench.py`，x86；输出没有保存） | 每个请求每步 0.81 ms（同一次运行，未保存），所以 4 个 session 时每步约 3 ms | 增量：按（请求，行）缓存组好的行；native 增长时只追加并校验新的尾部；只在布局改变时完整校验（WP3）。也能降低现在的开销 |
| H2 | Page 向量：`qwen.py:237-245`、`model_queries.py:58`：每个写满的 Page 做一次 5,120 个 float 的 `row.to(cpu).tolist()`，再加 Python 的 `normalized()` | 每个 6.5K 的 observation 约 8 个 Page | 每个 observation 约 406 个 Page；估计每轮 0.3–1 s（未测量） | 每次 forward 对所有写满 Page 的行做一次 device gather；在 device 上成批做投影→L2→int8；每个 Page 的 D2H 为 128 B + scale（WP3） |
| H3 | 向量存储（`RequestState.page_vectors`、`QuerySelectionAuthority.vector_catalog`）：5,120 个 Python float 组成的 tuple | 每个 64K session 约 14 MB | 每个 64K session 约 0.6–0.7 GB（×8 个 session） | 只在 P < 784 时存 `CompactVector`（int8[128] + scale）（WP3/WP4） |
| H4 | `worker_backend.publish_vectors`：对每个请求的全部阶段 2 向量做 JSON + **HMAC-SHA256** | 约 8 个 Page × 5,120 个 float | 字节数 ×49，所以逐请求的哈希量增长（R4） | 发布压缩向量（每 500 个 Page 64 KB）；对它保留 MAC，或者换成 identity token（开放问题 7）（WP3） |
| H5 | `query_selection.plan` 的压缩（torch CPU，逐 Page，有缓存） | — | 每轮约 500 个新 Page | 直接使用 worker 压缩好的向量；引擎路径上不做压缩（WP4） |
| H6 | 每轮 Selector V2 对 n 个 Page 的循环（上游记录、publisher、余弦、证据 `scores`/`page_turns`） | n ≤ 84 | n ≤ 4,095（PoC 也是这样） | 保留上游算法。测量。限制证据大小（开放问题 6）（WP4） |
| H7 | `TensorIO.capture/restore`：每层每个 block 一次锁页分配 + 一次同步拷贝 | 每轮封存约 8 个 Page × 16 次拷贝 | 每轮封存约 400 × 16 = 6,400 次锁页分配和同步 D2H；恢复最多约 1,600 × 16 次同步 H2D | 预先分配、以 P token 的 Page 为单位的锁页 arena；每层一次 device gather 到暂存区 + 一次 D2H（以及一次 H2D + 一次 scatter）。每个 arena slot 一个 identity token（WP3） |
| H8 | 逐 Page 的 `token_sha256`（`allocation_owner.py:341`、`native_growing/adapter.py:321`） | 每轮约 8 个 | 每轮约 500 个，每个约 1 µs（估计值，未测量） | 次数 ×49，字节数不变。可以接受；也可以换成 identity token（开放问题 7） |
| H9 | vLLM 请求的 block 哈希（`core.py:225`） | 每 64K 83 个 | 每 64K 4,096 个。模拟：每 64K token 1.2 → 4.2 ms，每个 8K token 的轮次约 0.5 ms（`tools/hashbench.py`；输出没有保存） | 可以接受（vLLM 的现有机制）。F3 对自有请求去掉它 |
| H10 | 每轮的 `runtime.jsonl`（选中的 Page、逐 Page 的分数和轮次） | 约 84 项 | 约 4K 项（每行约 100–200 KB） | 确定格式（开放问题 6）（WP4） |
| H11 | 引擎 catalog（`Session.catalog[page]` 字典），`adapter.py:177` 的上限 `n ≤ 4096` | 84 | 4,095：64K 时正好放下 | 把上限换成 `cdiv(max_model_len, P)`（WP2） |

## 6. 工作包

所有 WP 都从 `exp-decouple-pages` 分出。下面的文件归属互不重叠，所以四个 WP 可以同时进行。**集成负责人**（orchestrator）只在合并时重新生成发布元数据：
- `patches/v0.24.0.diff`（anchor 文件变成被修改的文件）；
- 通过 `development/selection-contract-v7/regen_release.py`（在项目文件夹中，不在本仓库里）生成的 `INSTALL_RELEASE.json` 和 `SOURCE_SHA256SUMS.txt`；
- 所有 source-closure pin（`remote_ra_live_v2/loaded_sources.py`）。

**冲突说明。** `exp-subpage-score`（另一个 agent）修改 `serve.py`、`composition.py`、`native_probe.py`、`private_endpoint.py`、`query_selection.py`、`selection_contract.py` 和 `worker_composition.py`。对这些文件的修改要小而局部。等那个分支进入 main 之后再 rebase。（它从未进入：解耦工作在没有它的情况下合并进了 `main`，`--page-score` 不在 `main` 上。）

### 6.0 共享接口 I0：`source-overlay/remote_ra_live_v2/kv_layout.py`（纯 Python；负责方 WP1，API 在此冻结）
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
校验：
- P % 16 == 0，M % P == 0，M % 16 == 0，0 < M ≤ 16384。
- 解耦 ⇒ K ≥ 3。legacy ⇒ P == M == 784。
- `additional_config` 的值：`{"page_tokens": P, "mamba_tokens": M, "mamba_slots": K}`。

### WP1 — vLLM 拆分池（overlay `source-overlay/vllm/**` + `kv_layout.py`）
1. **`platforms/interface.py` 的 `_align_hybrid_block_size`。**
   - 如果 `from_vllm_config(...).decoupled`：要求用户指定 `block_size == P`（后端支持：16 的倍数），并要求用户指定 `mamba_block_size == M`。
   - 设 `mamba_page_size_padded = None`，并**直接返回，不提高 `block_size`**。
   - legacy：不动。
2. **`v1/kv_cache_interface.py`。** `KVCacheConfig.group_pool` 和 `num_blocks_by_pool`（默认 None；legacy 对象与现在的对象比较相等）。
3. **`v1/core/kv_cache_utils.py`。**
   - `get_kv_cache_groups`：解耦 → 不统一页大小；分组相同。
   - `get_kv_cache_config_from_groups`：解耦分支按 D4。有 override 时（`_init_minimal_kv_cache_for_profiling` 传入 `available_memory=0`）：`N_fa = override`，`N_m = 1 + 3K`。
   - `get_kv_cache_configs`：按池缩小 tensor。
   - `_check_enough_kv_cache_memory` 和并发日志：按池。
   - `resolve_kv_cache_block_sizes`：解耦 → `(lcm(all), gcd(non-Mamba groups))`，即 (784, 16)。
4. **`v1/core/kv_cache_coordinator.py`。**
   - `self.block_pools = tuple(BlockPool(n,…) for n in num_blocks_by_pool or (num_blocks,))`，`self.block_pool = block_pools[0]`。
   - 各 manager 拿到 `block_pools[group_pool[i]]`。
   - `find_longest_cache_hit` 使用每个 group 自己的池。
   - 增加 `get_num_blocks_to_allocate_by_pool(...) -> tuple[int, ...]`；标量方法保留，返回总和。
5. **`v1/core/kv_cache_manager.py`。**
   - `block_pools`、`pool_of_group(g)`、`get_num_free_blocks_by_pool()`。
   - `allocate_slots`：两项检查（完整序列检查和主检查）都按池进行。
   - `usage` 报告池 0，另加新的 `usage_by_pool`。
   - `evict_blocks` / `reset_prefix_cache` / `take_events` 覆盖所有池。
   - `take_new_block_ids` 仍只针对 FA（清零 FA tensor，`v1/worker/utils.py:78-218`；无害）。
6. **`v1/core/sched/scheduler.py`。**
   - `_mamba_block_aligned_split` 使用 Mamba group 的 `block_size`（legacy 下等于 `cache_config.block_size`）。
   - `_remote_ra_native_free_request` 的淘汰（`:2330-2360`）和 `:2410` 处的释放：按 group 所属的池。
   - connector 绑定池 0。
7. **Offloading**（`offloading/worker.py`、`kv_offload/cpu/spec.py`、`offloading/scheduler.py` 的 group 配置）。解耦 → 只注册非 Mamba 的 group 和 tensor。每个 block 的 CPU 字节数来自池 0 的 tensor 和 `num_blocks_by_pool[0]`。
8. **runner 和 worker。**
   - 核实 `_reshape_kv_cache_tensors` **不需要改动**（逐层 `numel // page`）。
   - 对 FA 的 `block_dim == 0`，`_update_hybrid_attention_mamba_layout` 什么都不做。
   - 检查 `gpu_worker.py:604`（`num_gpu_blocks` = 池 0）和 KV zeroer。

**导出给 WP2/WP3 的内容（接口 I1）：**
- `kv_manager.block_pools`、`kv_manager.pool_of_group(g)`、`kv_manager.get_num_free_blocks_by_pool()`；
- `kv_cache_config.group_pool` / `num_blocks_by_pool`；
- FA spec 的 `block_size == P`，Mamba spec 的 `block_size == M`，不填充的 Mamba 页；
- worker 侧的 `runner.kv_cache_config` 带有同样的字段。

**验收。**
- legacy：CPU 测试表明，在 Qwen3.6 的 fake 上，`KVCacheConfig` 以及 scheduler/哈希的大小与未打补丁的函数相同（golden 值：167 个 block，16 × 167 × 3,211,264 B，(784, 784)）。
- 解耦：planner、池、分配/释放和 `resolve` (784, 16) 的行为都符合 D4。

### WP2 — 引擎侧 runtime 几何、归属与容量
文件：`serve.py`、`remote_ra_http/config.py`、`remote_ra_native_growing/{adapter,ownership,contract}.py`、`remote_ra_live_v2/{contract,phase1_coordinator,native_probe,engine_boundary,pool_incarnations,capacity,composition(startup wiring only),startup}.py`、`growing_startup.py`。
1. **启动。**
   - `serve.py --page-tokens {784,16,32,64}`（默认 784 → argv、配置和 `effective_server_config` 逐字节相同），另加一个只用于测试的 `--kv-layout decoupled`，用于 P=784 的验证。
   - P ≠ 784 时增加 `--block-size P --mamba-block-size M --additional-config {...}`，以及 `effective_server_config` 的键 `block_size`、`mamba_block_size`、`additional_config`。启动配置得到 `page_tokens=P` 和 `kv_layout`（I0 的 dict）。选页合同变成 `default_contract(P)`。
   - `remote_ra_http/config.py:45`：接受允许的 P，前提是布局和有效配置与之匹配。
   - 启动时构建一次 `KvLayout`，并与 `kv_manager` 交叉核对（FA manager 的 `block_size == P`，Mamba 的 `== M`，当且仅当解耦时有 `group_pool`），取代 `adapter.py:98-101`。
2. **Mamba 算术 P → M。** `Phase1Lease.mamba_index`、`parse_lease`/`_parse_goal` 的行长度、`Phase1Coordinator.run`（`M = ceil(e/M)`、Goal 宽度）、`NativeQueryForward._mamba_slot`。使用 I0 的 helper。
3. **池。**
   - `Phase1Coordinator` 从池 0 分配 FA，从池 1 分配 Mamba，并把它们各自释放回自己的池。
   - `PrivateBlockOwner` 只拥有池 0 的 block。断言它从不会看到 Mamba block。
   - `pool_incarnations` / `engine_boundary.install` 作用于每个池。
   - 别名和唯一性检查以 `(pool, id)` 为键：`contract.py:367-368, 408-413`、`native_probe.py:49-51`、`RowLayout`（只有 FA；不变）。
4. **容量（D7）。** `turn_need`/`headroom`/`assess`/`admission_gate` 返回每个池的需求。CapacityWait 的文字写明两个池；legacy 的文字不变。
5. **上限。** `adapter.py:177` 变成 `n ≤ cdiv(max_model_len, P)`。加一个测试：C ≡ 0、1、P−1 (mod P) 时，`o − C ≤ 3` 让 `CANDIDATE_RANGE` 在 P=16 下仍然有效。

**接口 I2（给 WP3）：** lease 和 `allocation_update` 的传输格式不变：
- FA 行：每个 FA group 每个 Page 一个 id（池 0）；
- Mamba 行：长度为 `mamba_row_len(e)`，用 null 填充，id 来自池 1；
- 每个 frame 携带一次启动时的 `kv_layout` dict（启动记录），从不逐步携带。

**验收。**
- 所有现有测试不加修改地通过（legacy）。
- 新的合同、容量和归属测试在 (P, M) ∈ {(784,784 legacy), (784,784 解耦), (16,784), (64,832)} 上参数化，并有一个随机轮次序列的性质测试：每个池的分配/释放平衡，池内没有别名。

### WP3 — 小 P 下的 worker 侧数据通路
文件：`remote_ra_growing_worker/tensors.py`、`remote_ra_live_v2/{worker_backend,allocation_owner,worker_rows,sparse_mapping,geometry,qwen,worker_composition,model_queries}.py`。
1. **gate。**
   - `worker_composition.py:112-116`：FA 视图 `allocation_tokens == P`，混合视图 `== M`。
   - `TensorIO._validate_aliases`：解耦 → 完全没有跨类别的别名（FA 和 GDN 的 tensor 互不相交）。
   - `validate()` 的唯一性按池检查；`allocation_owner.py:205, 469-470` 按池。
2. **GDN 节奏。** `qwen.py` 在 `lease.end % M == 0` 时捕获最新状态（M 来自启动布局）。GDN 路径的其他部分都不变。
3. **DRAM arena（H7）。**
   - 预先分配的锁页 arena，由 FA Page slot 组成，每个 slot 包含全部 16 个 FA 层，每层 `P·4096` B。
   - 容量来自启动配置。默认：足够容纳 S 个 session × `cdiv(max_model_len, P)` 个 Page，每个 64K session 约 4 GiB。connector 的池仍然独立（开放问题 9）；这里它是 32 GiB，`main` 上默认 1 GiB（`--connector-pool-gib`）。
   - 成批的 `capture(pages)` / `restore(pages)`：每个 FA 层一次 device 上的 `index_select` 到暂存缓冲区 + 一次 D2H（H2D + `index_copy_`），然后 drain 一次。
   - 每个 slot 的 identity token 取代每个 payload 的 token。字节计数的含义不变。
   - GDN 的 payload 与现在相同。
   - legacy 的 P=784 可以保留旧路径。等价性在 GPU 上测试。
4. **向量（H2–H4）。**
   - 解耦：每次 forward 把写满 Page 的行收集一次，在 device 上一批压缩，使用 `selection_contract.projection` 和 `compact` 的算法（fp32 线性 → L2 → 带 scale 的 int8）。
   - `page_vectors` 存放 `CompactVector`。`publish_vectors` 发送 `{page: (int8 list, scale)}`。
   - legacy 保留 float 路径，逐字节相同。
5. **行（H1）。** 增量的 `install_row`，不变量相同：前缀不可变，展开精确。预算：64K 稳态下每个请求每步 ≤ 0.05 ms。

**接口 I3（给 WP4）：** 解耦模式下，`publish_vectors` 的消息体增加 `representation="compact_int8_v1"`，以及 `page_vectors={page: [128 int8], scale}`。legacy 的表示不变。

**验收。**
- CPU 测试：arena slot 的记账（纯 Python），在随机序列上增量行安装等于完整重组（只用 numpy），M ≠ P 时的 qwen 节奏。
- 依赖 torch 的测试当时在本地跳过（开发机上没有 torch，也不允许下载），在 DeltaAI 环境中运行。之后装上了 CPU torch：集成测试套件运行了它们（482 个测试，没有跳过）。

### WP4 — 选页合同、证据、测试、GPU harness
文件：`remote_ra_live_v2/{selection_contract,query_selection,private_endpoint(seed note only),vector_space}.py`、`remote_ra_growing/contract.py`、`remote_ra_http/evidence.py`、新的 `test_decouple_*.py`、`development/exp-decouple-pages/gpu-smoke/`（实际建成 `smoke-dp1/`，见 WP4.md）。
1. **合同。**
   - `default_contract(16)` 给出 recent 配额 64 和容量 `floor(0.4·n·16)//16`。
   - 保留 `initial_prompt_pages`（cover）的语义，除非算法方选择 PoC 的“完全落在其中”（开放问题 2）。那会是一个新的枚举值，而不是修改旧值。
2. **选页。**
   - `query_selection.plan` 接受压缩向量（I3），跳过压缩步骤。
   - P=16 时 PageRecord 的 token 区间等于 native Page（公式本来就得出这个结果）。
   - 证据大小按开放问题 6 决定。
3. **证据。**
   - `ready.json`/`startup.json` 记录 `kv_layout` 和 `num_blocks_by_pool`。
   - 每轮的行增加 FA/Mamba 池的空闲 block 数。
   - `restore.h2d_bytes` / `prepare.h2d_bytes` 的定义不变。
4. **测试。** 把 stage-B 的测试套件在 legacy 和 (16,784) 上参数化：`test_stageb_{contract,engine,worker,e2e}.py`、`test_ms_*`、`test_host_trim.py`。它们的 fake 目前把 FA = Mamba = P 写死了。新增 `test_decouple_layout.py`（I0 + WP1 的 AST 测试）。
5. **GPU harness。** 第 8 节的 sbatch 脚本，用 `--mail-type=BEGIN,END,FAIL` 发到 DeltaAI 的作业通知邮箱，在 `ghx4-interactive` 上依次运行（一次一个作业，重跑时用新的 `-RunLabel`；之后的 GPU 作业使用 `ghx4` 批处理分区）。它们复用 `development/stage-b/tools` 和 `development/multi-session/tools/step_profile.py`（在项目文件夹中，不在本仓库里；`step_profile.py` 现在也以 `tools/step_profile.py` 的形式在仓库里）。

### 集成顺序
1. WP1 和 WP2/3/4 用 fake 针对 I0–I3 编写代码。
2. 依次合并 WP1 → WP2 → WP3 → WP4。
3. 重新生成发布元数据；运行 `python -B run_tests.py --all`（自 integrate-1107 起测试在 `tests/` 中；此前是 `python -B -m unittest discover`）和 `check_git_release.py --ref HEAD`。
4. 运行 GPU 冒烟测试。

## 7. CPU 测试计划（本地；除非注明，不需要 torch）
- **WP1。** 用 `ast` 提取打过补丁的函数（`test_allocation.py` 的做法），喂给它们 fake spec：16 个 block 为 P 的 FA 层，48 个 `page_size_bytes = 3,207,168`、block 为 M 的 Mamba 层。
  - legacy：与取自 `main` 字节的原始函数做 golden 相等比较。
  - 解耦：group 与 legacy 相同；tensor 列表和大小；8/24 GiB 和 S = 1/4/8 时的 `N_fa`/`N_m`；profiling 的 override；`resolve` (784, 16)；双池分配/释放；MambaManager 的 align 哨兵对照它自己的池；在不同池之间重复的 id；`cache_config.block_size = 16` 时 `_mamba_block_aligned_split` 在 M 处切分。
  - offload 注册排除 Mamba。
- **WP2。**
  - 在 (P, M) 矩阵上测合同和 lease 的几何；`mamba_index`；Goal 行。
  - 归属 `row_groups`；双池容量（CAPACITY.md 中的算例按 P=16 重新计算）。
  - `serve.py` dry run 的 argv：legacy 逐字节相同（现有的 `test_ms2_admission` 检查），解耦时恰好多出那三个参数。
  - 配置的 pin。
- **WP3。** arena slot 分配器；增量行安装等于完整组合；qwen 的捕获节奏；按池的 `validate()`。（在 DeltaAI 上用 torch：成批压缩在约定的容差内等于逐 Page 的 `compact()`，开放问题 3；在 CPU tensor 上做 arena 拷贝往返。）
- **WP4。** P=16 时的选页数字对照 PoC 公式；证据 schema；random50 的 `sparse_count`。
- **发布。** 最后运行 `check_git_release.py`。

## 8. GPU 验证计划（第一次冒烟测试，在一张 GH200 上依次运行的作业）
全程同一个 commit，同一个合成的多轮负载（6 轮，每轮约 6.5K token 的 observation，prompt 最多约 40–60K token），`temperature 0`。

| 作业 | 配置 | 必须显示 |
|---|---|---|
| A | legacy（P=784，不加参数） | 回归：在相同输入上，token 和证据与 `main` runtime（902690a）相同 |
| B | 解耦，P=784，M=784 | 启动日志显示两个池（FA = Mamba 的 block 数按 D4），没有填充日志。`full` 的 token 与 A **完全相同**。每个延续轮次 `prepare.h2d_bytes` = 153,944,064。跨块搬移与 A 相同 |
| C | 解耦，P=16，M=784，`full` | token 与 A **完全相同**（同样的 kernel、同样的 KV 顺序、同样的 chunk 切分）。FA 池的 block 数与计算值一致。1 个 session 时每步 decode 开销比 A 每个请求最多多 +0.1 ms；admission（阶段 1 + reserve）比 A 慢不超过 10 % |
| D | P=16，四个策略，10 轮 | 8,192 token 之后选页生效。`capacity_pages = floor(0.4·n·16)//16`，recent ≤ 64，发生恢复。干净关停回执。每个 session 的主机 RSS 和锁页字节数在 arena 估计之内。没有 CAPACITY 或归属失败 |
| E | P=16，4 个 session × 6 轮，24 GiB | 成批 decode，无抢占。drain 之后两个池都回到基线。池用量的采样符合双池模型。只有 FA gate 起约束作用时才出现 CAPACITY_WAIT |
| F | `step_profile.py`，1–4 个 session 下 P=16 对比 A | 每步主机开销 ≤ A + 每个请求 0.1 ms（R5）；`install_row` 的用时；每轮向量流水线的用时 |

**逐 token 一致性。** 按既定的审计做法，还要通过窗口内 decode 步把 C 与 stock vLLM 比较。

**如果 C 不一致**，报告第一个出现分歧的位置。B 一致而 C 有分歧，说明问题在依赖 P 的代码（行、slot mapping、chunk 切分）；B 不一致，说明问题在池的拆分。

## 9. 风险
- **隐藏的单池假设**，位于这里看不到的上游代码路径中（metrics/Prometheus 用量、KV event 的消费者、CPU offload manager）。缓解：第 11 节的文件清单；解耦模式下对 `group_pool` 做 fail-fast 断言。在看得到的 coordinator 里，每一处 `self.block_pool` 的使用（包括混合 coordinator 的 `find_longest_cache_hit`）都必须审查。
- **主机侧 O(Pages) 的开销**（第 5 节）吃掉 decode 预算。缓解：作业 F 强制检查 R5 预算；P=64 作为后备。
- **向量算术。** 在 device 上成批压缩会改变 int8 的舍入，与现在逐 Page 的 CPU 路径不同。这只影响解耦模式，但对算法方来说是语义上的变化。
- **锁页主机内存。** connector 池（不使用；这里 32 GiB，`main` 上默认 1 GiB）加上 arena（每个 64K session 约 4 GiB）。8 个 session 时要注意节点内存。
- 与 `exp-subpage-score` 的**合并冲突**。
- **发布元数据的变动。** anchor 文件（`interface.py`、`kv_cache_utils.py`、`kv_cache_interface.py`、offload 文件）变成被修改的文件。安装器的上游映射能处理这一点（`expected_live` 对比 `payload`），但 `patches/v0.24.0.diff` 会变大。

## 10. 开放问题（维护者 / 算法方）
1. **P。** 目标是否为 P = 16（与 PoC 一致），并在达不到主机开销预算（R5）时接受 P = 64 作为后备？解耦的 P=16 在验收合格后应成为默认，还是保持需显式开启？
2. **P = 16 时的必保留页。** 保留“system + 任务 prompt 触及的每一页”（cover，现在的做法），还是改用 PoC 的“完全落在其中的页”（floor）？P=16 时两者最多差一页。
3. **Page 向量。**
   - int8 向量必须与 PoC 的 `_compact_last`（成批，从 fp32 hidden 计算）逐位相同，还是与现在的 runtime 路径（Python float64 预归一化 + 逐向量的 torch CPU 压缩）逐位相同？
   - 余弦排序相同是否就够了？
4. **GDN 节奏。**
   - 现在的 align 语义是否满足“每个请求一份连续的 GDN 状态”：一份状态，每 M = 784 token 经一次主机往返在 slot 之间移动？
   - 还是第一版就要不跨块的变体（F1）？
5. **Mamba 池。** 从 `--kv-cache-gib` 中静态预留 (2S+1) 个 slot（S=1 时 0.51 GB，S=8 时 2.67 GB）是否可以接受？还是 `--kv-cache-gib` 应只指 FA，Mamba 池另外加在上面？
6. **P=16 时的证据量。** 每轮的行会带约 4K 个选中/分数/轮次条目。全部保留，还是记录选中的 Page 加上分数的 top-k？
7. **现有的逐 Page 和逐请求摘要。** 本着“热路径上不做哈希”这条规则的精神，P < 784 时能否替换下面这些？
   - 逐 Page 的 `token_sha256`；
   - 对发布向量的 HMAC。
   - 候选替代：identity token，或只对压缩向量做 MAC。
8. **16 token 时的 vLLM block 哈希。** 每个 64K 请求 4,096 个；在这里的 CPython 模拟中每 64K 约 4 ms（输出没有保存），没有在 Grace 上测量。能否接受？还是对自有请求跳过哈希（F3）？F3 需要一个新的 Random50 种子，因为现在的种子是最后一个 block 哈希。
9. **解耦模式下的 OffloadingConnector。** 保持注册（隔离，只有 FA），还是去掉（F2）？去掉它能把那个不使用的 32 GiB 锁页池（`main` 上默认 1 GiB）腾给 DRAM arena，但会改变启动约定。
10. **与 `exp-subpage-score` 的关系。** 如果 P = 16 落地，还需要对 784 的 Page 做子页打分吗？哪个分支 rebase 到哪个分支上？

## 11. 本地没有的上游 vLLM 0.24.0 文件（历史清单）
- **`vllm/engine/arg_utils.py`。** EngineArgs → CacheConfig 的连接：对这个模型 `--block-size` 是否接受 16？`--mamba-cache-mode align` 下 `--mamba-block-size` 是否生效？`--additional-config` 如何解析？有没有 block 大小的可选值列表？
- **`vllm/model_executor/models/config.py`。** Qwen3.5/Qwen3Next 的 `verify_and_update_config`：
  - 在哪里由 `mamba_ssm_dtype` 设置 `mamba_ssm_cache_dtype = float32`；
  - `mamba_cache_mode` / `mamba_block_size` 的默认值；
  - 是否还有第二次 block 大小对齐。
- **`vllm/config/model.py`、`vllm/config/scheduler.py`。** `is_hybrid`、默认的 `mamba_block_size`、`disable_hybrid_kv_cache_manager`、`long_prefill_token_threshold`、`watermark`。
- **`vllm/v1/request.py`。** `Request.block_hashes` 和 block hasher 的调用点（H9，Random50 种子）。
- **`vllm/utils/hashing.py`。** 确切的 sha256 序列化方式，用于实际测量 H9。
- **`vllm/v1/core/kv_cache_metrics.py`、`vllm/v1/metrics/{stats,loggers}.py`。** 每个池的用量和 Prometheus 的 `kv_cache_usage`，CAPACITY.md 的交叉核对会用到。
- **CPU offload 的内部实现：**
  - `vllm/v1/kv_offload/cpu/manager.py`（`CPUOffloadingManager`）和淘汰策略；
  - `vllm/v1/kv_offload/factory.py`；
  - `vllm/distributed/kv_transfer/kv_connector/factory.py`。
  - 排除 Mamba 和去掉 connector 都需要它们。
- **`vllm/_custom_ops.py`**（`swap_blocks_batch`、cache op）及其 csrc 对应实现，前提是 arena 复用 vLLM 的成批拷贝 op 而不用 torch 索引。
- **`vllm/v1/attention/backends/fa_utils.py`** 和 `vllm_flash_attn` 接口：block table 的限制；确认 GH200 上 kernel block 16 时的 FA2/FA3 路径。
- **`vllm/v1/executor/abstract.py`**，以及 `gpu_worker` 用 `kv_cache_memory_bytes` 确定内存的部分（已经能看到），用来确认预算只被拆分一次。
- **`vllm/v1/core/sched/interface.py`、`vllm/v1/core/sched/utils.py`。** 次要，只为完整。

## 附录：关键事实及其出处
- **FA kernel 大小** `[16,32,64]`（Mamba 状态为 fp32 时）：`vllm/v1/attention/backends/flash_attn.py:77-94`。kernel 拆分：`vllm/v1/worker/block_table.py:50-65`。
- **align 模式下的 GDN 下标** `(seq_len-1)//M`：`vllm/v1/attention/backends/utils.py:927-939`。跨块拷贝：`vllm/v1/worker/mamba_utils.py:678-742`。每个 group 两个 block：`vllm/v1/core/single_type_kv_cache_manager.py:1111-1252`。
- **runtime 的 GDN 节奏**：`remote_ra_live_v2/qwen.py:179-192, 246-252`。carry/阶段 1 的 H2D：`remote_ra_live_v2/worker_backend.py:111-177`。
- **单池检查**：`remote_ra_live_v2/capacity.py:175-178`、`phase1_coordinator.py:35,102-118`、`remote_ra_native_growing/ownership.py:53-81,136-137`、`remote_ra_growing_worker/tensors.py:296-302`、`remote_ra_live_v2/allocation_owner.py:205,469-470`、`pool_incarnations.py:27-28`。
- **“每个 Page 一个 FA block”的 gate**：`sparse_mapping.py:86`、`worker_composition.py:112-116`、`remote_ra_native_growing/adapter.py:98-101`、`phase1_coordinator.py:52-54`、`remote_ra_live_v2/contract.py:264-269`。

## 偏差

### WP1（分支 `exp-decouple-wp1`）
1. **对接口的理解（API 不变）。**
   - `to_additional_config` 返回整个 `--additional-config` dict，即 `{"remote_ra_kv_layout": {"page_tokens", "mamba_tokens", "mamba_slots"}}`，或 None；`from_vllm_config` 做逆变换。有这个键就表示解耦，P = 784 也一样。`mamba_pool_blocks` 对 legacy 布局抛出异常。
   - `kv_manager.pool_of_group(g)` 返回 `BlockPool`；它的下标是 `kv_cache_config.group_pool[g]`（0 = 注意力，1 = Mamba）。`get_num_free_blocks_by_pool()` 是一个 tuple，池 0 在前。legacy：`block_pools == (block_pool,)`，所有 group 都在池 0。
   - 池按 spec 类型确定，而不是按 group 下标。真实模型可能把 Mamba group 列在前面（第 0 层是 GDN）；第 7 节的 fixture 把 FA 放在前面。WP1 不依赖这个顺序。
2. **测试文件。** 用 `test_kv_layout.py`（I0）和 `test_kv_pools.py`（WP1），而不是 WP4 文件清单中的 `test_decouple_layout.py`。`test_kv_pools.py` 用 stub import 执行 overlay 源码。它对照基础 commit 8586dbd 的 legacy 比较需要 Git 历史，在 Git archive 运行中跳过。
3. **Offloading（D6）。** scheduler 侧的 group 配置每个 KV group 保留一项：block id 按 group zip 在一起，GPU 传输断言每个 group 一个 ref 列表。worker 只注册池 0，给 Mamba group 空的 ref。原生 offloading scheduler 如果要在解耦模式下发出 job 就抛出异常；只支持隔离的 connector。
4. **延迟释放。** 解耦模式曾拒绝 `defer_block_free`（与 consumer connector 重叠的 batch），因为延迟列表会混入两个池的 block。`--async-scheduling`（async-c）会开启它：GPU 作业 3499729 中本 runtime 的每个异步服务都在构造 Scheduler 时失败（stock vLLM 的异步服务通过了）。自 async-fix 起，一个延迟项保存 (block pool, block) 对（`KVCacheManager.pop_blocks_for_free_by_pool`），`_drain_deferred_frees` 在同一个 fence 之后按共享池的顺序（该项倒序，每个池一次 `free_blocks` 调用）把每个 block 还给它自己的池；共享池的路径不变（`test_deferred_free_pools.py`）。
5. **manager 的细节。** `evict_blocks(ids)` 作用于池 0（connector 的 id）。`reset_prefix_cache` 和 `take_events` 覆盖每个池。`usage` 仍是池 0；`usage_by_pool` 是新增的。常驻指标的 collector 绑定到池 0。`get_kv_cache_capacity`（并发日志，`kv_cache_size_tokens`）取各池的最小值。
6. **解耦模式下启动时拒绝：** hidden-state cache 层、packed 或跨层布局、不同时含有注意力 group 和 Mamba group 的配置、填充的 Mamba spec、align 以外的 `--mamba-cache-mode`。
7. **runner 和 worker：不改动**（第 8 项已核实）。reshape 是逐层的（`numel // page`），FA 的 `block_dim == 0`，zeroer 只碰 FA 层，`num_gpu_blocks` 是池 0，Mamba 的 kernel block 是 M。
8. **分支上的发布元数据。** pin 来自 `regen_release.py`，另加被编辑的 closure 文件的 source-closure pin（`regen_release.py` 不处理这些）；`kv_layout.py` 没有加入 closure 集合。安装器计数为 (291, 200)。`patches/v0.24.0.diff` 没有重新生成（由集成负责人负责）。新被修改的 anchor：`platforms/interface.py`、`v1/kv_cache_interface.py`、`v1/core/kv_cache_utils.py`、`offloading/worker.py`、`kv_offload/cpu/spec.py`。
9. **需要 GPU 作业 B。** 本地看不到的一点：上游（`models/config.py`、`arg_utils.py`）没有任何地方在 align 模式下覆盖用户的 `--mamba-block-size`，或重新填充 Mamba 页。启动日志行“Decoupled KV layout: attention block …”标志着走的是解耦分支。

### WP2（分支 `exp-decouple-wp2`）
1. **启动。** `serve.py --page-tokens {784,16,32,64}`（默认 784）和只用于验证的 `--kv-layout decoupled`。两者都不给时，`prepare()` 不走任何新分支：argv、配置、`effective_server_config` 和 pin 都逐字节相同（对照 8586dbd 的代码证明）。解耦时恰好在 argv 末尾追加 `--block-size P --mamba-block-size M --additional-config <to_additional_config, sorted keys>`（在 WP2 上如此；在 `main` 上其后还有后端、torch-profile 和纯文本参数）；配置增加 `kv_layout`（内层的 I0 dict）和 `page_tokens = P`；选页合同是 `default_contract(P)`。`remote_ra_http/config.py` 要求 `kv_layout` 记录等于 `for_server(P, sessions)`，并要求有那三个固定的 vLLM 参数；`effective_server_config` 只在解耦模式下比较它们（JSON 文本形式的 `additional_config` 会被规范化）。`legacy_geometry()` 把 `kv_layout` 传下去。
2. **布局的传递。** `growing_startup` 构建一次布局（`contract.startup_layout(config geometry) or legacy()`），要求它等于 `from_vllm_config(vllm_config)`，把 `kv_layout.py` 作为已加载模块校验，然后传给 `EngineCoreAdapter(layout=...)`；后者交叉核对 manager（按 spec 类型，FA block = P，Mamba block = M；legacy：`group_pool is None`，`block_pools == (block_pool,)`；解耦：有 `group_pool`，两个池，`pool_of_group(g) is block_pools[group_pool[g]]`，Mamba 池 = `mamba_pool_blocks`）。fixture 不传布局，并在一个池中保持 M = P。`compose_engine` 只在解耦模式下把 `kv_layout` 加进 worker 的启动记录（I2：一次，从不逐步）；worker 上由 `startup_layout(record)` 读取。
3. **lease（把 I2 说明确；格式不变）。** `parse_lease(raw, attention_groups=..., layout=...)`：共享池与之前完全相同。解耦：Mamba 行长 ceil(e/M)，Goal 的 Mamba 行长 ceil(goal/M)；id 在每个池内唯一，以 (pool, id) 为键；incarnation 的键在池 0 为 `str(id)`，在池 1 为 `"1:id"`；`owned` 先列新的 FA id（按 FA group 顺序），再列 Mamba id（按 group 顺序），同一个 id 可以在不同池中重复出现；**`zero` 只含池 0 的部分**（Goal：它的 FA id）：vLLM 的 `KVBlockZeroer` 只写 FullAttention tensor，所以那里出现 Mamba id 会把同 id 的 FA block 清零。`attention_groups` 是必需的。`Phase1Lease` 保存 M 和各 group 的池（不参与比较，也不上线传输）：`mamba_index` = ceil(e/M) - 1，`block_keys()`。`NativeQueryForward` 用启动布局解析，并拒绝按 (pool, id) 与 lease 构成别名的 claim。
4. **归属。** `Phase1Coordinator` 从池 0 分配 FA，从 Mamba 池分配 GDN slot，按池检查和释放，按池记录 incarnation。`PrivateBlockOwner` 仍在池 0 上，在钉住或释放之前检查移交/接收的 block 是它的池中的对象。`pool_incarnations.rows` 接受逐 group 的池；`EngineRpcBoundary` 跟踪每个池。
5. **P < 784 时没有逐 Page 摘要。** 解耦模式的 completion 把封存的 Page 发布为 `{token_span: [pP, (p+1)P], namespace, binding, source, content}`，而不是 `token_sha256`（binding 身份加上区间就指明了这些 token；P = 16 时每轮约 500 次 SHA-256 被去掉）。引擎检查这个区间。启动记录带有 `kv_layout` 时，**WP3** 必须在 `allocation_owner._complete` 中发出 `token_span`；**WP4**：解耦运行中 worker frame 和 native 证据都带 `token_span`。legacy 记录不变。没有增加其他哈希；现有的逐请求摘要（lease 摘要和 MAC、请求摘要和 plan 摘要）仍是逐请求的，只是它们的输入随 Page 数增长。
6. **容量（D7）。** `turn_need_by_pool`/`headroom_by_pool` 给出 (FA, GDN) 的 block 数；legacy 的 `turn_need`/`headroom` 返回它们的和，`assess`/`admission_gate` 保留标量 dict 和文字。解耦的 `assess` 返回 `pools=('FA','Mamba')`，并为 `free`（`get_num_free_blocks_by_pool`）、`phase1`、`phase2`、`headroom`、`watermark`（只有 FA）给出数对；gate 按池应用 legacy 的不等式，并写明是哪个池以及两个空闲计数。运行中的请求只要还可能跨越一个 M block，它的跨块 slot 就计入。
7. **上限。** catalog 的上限是 `cdiv(max_model_len, P)`（784 时为 84，16 时为 4096；fixture 的 manager 没有 `max_model_len` 时为 4096）。P = 16 时，对 C = 0、1、P-1 (mod P)，CANDIDATE_RANGE 成立（已测试）。
8. **引擎的逐步行（新增内容，H1 的引擎一半）。** `dispatch_mixed_request` 原来每步重建整个 FA 行（组合、重排、前缀检查、incarnation 行）：这里在 P = 16 / 64K 时每个请求每步 2.5 ms（784 时 0.12 ms；一个临时的 `stepbench.py`，输出没有保存）。仅限解耦：第一次分配照旧组合并发布；之后各步检查并追加新的 transient，抽查已发布的开头部分，重建短的 Mamba 行，并发送相同的 `allocation_update` 行：0.08 ms（同一次运行）。**未解决（跨 WP）：** I2 仍然每步发送完整的行，所以 worker 侧每步仍是 O(row)（`_allocation_update`、`_observe`）；增量形式（追加的 FA 项 + Mamba 行）需要 WP2 和 WP3 一起改。后来部分解决：`--decode-fastpath2`（`main` 上的速度默认值）在一步既没有追加 FA block、Mamba 行也没变时不发送行（`engine_boundary.py` 的 `_grow`）；追加了 block 的步仍发送完整的行。
9. **不变：** `remote_ra_live_v2/startup.py`、`query_selection.py`、`selection_contract.py`（预算本来就随 `page_tokens` 缩放）、`RowLayout`。
10. **测试。** `test_decouple_wp2.py`（(P, M) 矩阵 x 两种 group 顺序、双池引擎 harness、性质测试、launcher/配置）和 `test_decouple_legacy.py`（每个改动函数在 8586dbd 上的源码，原样编译进它的模块：四个策略的 stage-B e2e、成批 worker、带容量需求的两 session 引擎、gate、launcher、配置、启动连接，结果都相同）。两者都不需要 Git 历史；e2e 部分需要 CPU torch。这个分支上的安装器计数：(291, 195)。
11. **需要 GPU。** vLLM 对 `--additional-config` 的解析（期望 dict；也容忍文本），以及解耦模式下是否接受 `--block-size 784` / `--mamba-block-size`（WP1 第 9 项）。

## 集成（分支 `exp-decouple-int`）

1. **合并。** WP1 → WP2 → WP3 → WP4，每次都用 `--no-ff`。文件归属互不重叠这一点成立：冲突只出现在发布元数据（用考虑 closure 的 regen 重新推导）、安装器的精确计数（现在是 (292, 201)：加入了 `kv_layout.py` 和 `page_compaction.py`，WP1 的五个 anchor 进入 payload）和本文件的偏差部分（两节都保留）。合并 WP2 之后完整测试套件通过（388 个测试），四个都合并之后也通过（468 个）；集成后的 head 运行 482 个。
2. **接口修正。**
   - `serve.py`：解耦的 query 服务的合同带有 `vector_format = vector_format_for(kv_layout)`（`compact_int8_v1`）；没有它，worker 接入检查会拒绝该合同。legacy 构建一次合同，不变。
   - 解耦时，阶段 1 和 completion 接受 WP3 的压缩 Page 向量：`Phase1Result.representation`（默认是 legacy 的 float 行）、`Phase1Coordinator._result`（压缩的 Page 加 float 的 query，`query_representation`）、adapter 的 completion 向量（`page_compaction.from_wire`）。每种布局都拒绝另一种表示。
   - 解耦时，`NativeQueryForward.after_forward` 保存一个 `ProbeRows`：使 Page 写满的那些行在 device 上的副本（由 `model_queries` 在那里压缩）、主机上的最后一行，以及每一行是否有限；legacy 仍拷贝每一行（5,120 维 fp32 时，每个 6.5K 的 observation 约 133 MB）。
   - 阶段 1 的别名检查在 WP2 中已经以 (pool, id) 为键（`fc69eb6`，已测试）；不改动。
   - 这次集成时 `check_git_release.py` 给测试套件 840 s（与 admission 合并之前的 `exp-subpage-score` 相同）；`7edc7b8`（10-10）把它提高到 1200 s，并设 `OMP_NUM_THREADS=1`。在 `main` 上它用 10 个并行 worker 运行各模块，每个模块 1200 s，总共 3600 s，仍设 `OMP_NUM_THREADS=1`。
3. **发布元数据。** pin 在最后重新生成一次。`patches/v0.24.0.diff` 没有重新生成：仓库里没有它的生成器，而且它在 `main` 上已经过时（缺少 35bcf77、9b18421 和 b1e6988 对 `core.py` / `core_client.py` 的修改）。
4. **CPU 端到端**（`test_decouple_e2e.py`）：在推广到 (P, M) 的 stage-B fake 上运行真实的引擎、worker、Selector V2 和证据（解耦时两个池，Mamba slot 为 ceil(end/M) - 1，只清零 FA，M token 的 Mamba 视图，owner 的成批 Page 入口），(P, M) ∈ {(784, 784) legacy, (784, 784) 解耦, (16, 784)}，四个策略、重建、取消，以及三个 session 在一个 batch 中 decode。legacy 和解耦的 (784, 784) 做出相同的决定、相同的恢复和相同的注意力计算；它们的证据只在这些地方不同：`kv_pools`、用 `token_span` 代替 `token_sha256`、压缩向量和表示字段、FA block id、对这些内容的摘要和 MAC，以及合同摘要。在这个负载上，一次单独的比较发现两种布局的 int8 向量相同，fp32 scale 的相对差在 2e-7 以内（输出没有保存；`test_decouple_e2e.py` 允许向量不同）；WP3 测得在 5,120 维的行上约 0.35 % 的向量有一个元素相差 ±1，这可能改变 query 策略中近乎并列的排序。902690a 的 overlay 上的 legacy 情形（没有 `kv_layout` 模块：解耦之前的连接方式）产生的 RPC payload、回执和证据文件与本代码树相同，现有的 stage-B、batched-worker 和 two-session 脚本也是如此（跨进程检查，由 pid 派生的 run id 除外）。
5. **留给 GPU 冒烟测试的部分。** CPU 上没有覆盖：vLLM 对这三个参数的解析（WP1 第 9 项，WP2 第 11 项）、真实的 TensorIO/PageArena 和 device 上的压缩、CUDA 的执行顺序。解耦模式的 Goal forward 和冷请求从一个没人清零的 Mamba slot 开始（清零列表只含 FA）：之所以正确，只是因为 vLLM 忽略位置 0 的初始状态（`has_initial_state` 为 false），作业 B/C 证实这一点。P = 16 时 Random50 的种子来自 16 token block 的哈希，所以它的选页结果不能与 legacy 比较。
