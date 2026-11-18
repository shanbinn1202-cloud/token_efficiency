# 服务选项

[English](reference.md) | 简体中文

给启动服务的人看：第 4.6 到 4.15 节的每个 launcher 选项做什么、默认值、何时使用、限制和对应的测试（KV 布局和 Page 大小、CUDA graph、上下文上限和回复上限、admission 优化、解码快速路径、纯文本、释放空闲 Page、Gemma 4 31B IT profile 和默认开启的速度选项）。`python -B serve.py --help` 列出所有参数；策略见 [docs/design.zh-CN.md](design.zh-CN.md#5-query-策略的选页规则) 第 5 节。GPU 作业结果见 [docs/status.zh-CN.md](status.zh-CN.md)，限制见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md)；本页在 2026-11-16 之前记载的 GPU 运行说明冻结在 [docs/history/reference-gpu-notes.zh-CN.md](history/reference-gpu-notes.zh-CN.md)。这些节在 integrate-1107 之前是 README 的第 4.6 到 4.15 节，编号不变（代码和测试引用这些编号）；各文档页面的节号互不重复，[文档地图](../README.zh-CN.md#文档地图)列出了每一节所在的页面。

### 4.6 解耦 KV 布局与 16 token 的 Page

**为什么默认 Page 是 784 token。** Qwen3.6-27B 是混合模型：16 个全注意力层（4 个 KV head × 256，bf16：每层每 token 4,096 字节），以及 48 个 Gated-DeltaNet 层，每层有一份 3,207,168 字节的递归状态加卷积状态（vLLM 把这类层叫作 Mamba 层）。vLLM 0.24 把两类状态放在同一个 block 池里，只用一种页大小。`vllm/platforms/interface.py` 中的 `_align_hybrid_block_size` 会把注意力 block 调大，直到一个注意力页至少能放下一个 Mamba 页：16 × ⌈3,207,168 / (16 × 4,096)⌉ = 784 token，并把 Mamba 页填充到 784 × 4,096 = 3,211,264 字节。runtime 的一个 Page 就是一个注意力 block，所以 Page 是 784 token，PoC 按 16 token 计的预算只能取整到整页（第 5 节）。

**解耦布局改了什么。** 它给两类状态各建一个 block 池，block 大小互相独立（`docs/exp-decouple-pages/DESIGN.md`，决定 D2–D4）：
- **注意力池：** P token 的 block。一个 block 仍然是一个 Page：16 个注意力层合计 P × 65,536 字节，P = 16 时为 1 MiB。
- **Mamba 池：** 不填充的 Gated-DeltaNet 状态。48 层分成 3 个各 16 层的 Mamba group；一个 block 存一个 group 的状态（16 × 3,207,168 = 51,314,688 字节），一个 slot 是每个 group 各一个 block。池里有 S 个 session 对应的 K = 2S + 1 个 slot（每个 session 一个运行 slot 和一个跨块 slot，另加一个备用），再加一个 null block：共 1 + 3K 个 block。它先从 `--kv-cache-gib` 中预留，剩下的给注意力池，即 ⌊(KV 字节数 − (1 + 3K) × 51,314,688) / (P × 65,536)⌋ 个 block。
- **Mamba 对齐 block：** M = P × ⌈784 / P⌉，P = 16 时为 784。因此 Gated-DeltaNet 状态的跨块、捕获和恢复都与 784 token 的 Page 完全相同（每份状态 153,944,064 字节）。

group 结构、kernel 和 Mamba 的 `align` 模式都不变。

**参数。** `--page-tokens 16`、`32` 或 `64` 总是使用解耦布局；`--kv-layout decoupled` 在 784 token 时也会选用它，这是只拆分两个池的验证模式。`--page-tokens 16 --kv-layout decoupled`（GPU 运行用的写法）与只写 `--page-tokens 16` 效果相同，只是把布局写明。此时 `serve.py` 会在 vLLM 命令末尾追加 `--block-size P --mamba-block-size M --additional-config {"remote_ra_kv_layout":{"mamba_slots":K,"mamba_tokens":M,"page_tokens":P}}`，并在 `startup.json` 里记录 `kv_layout` 和 `page_tokens` = P。两个参数都不给时，布局不增加任何 vLLM 参数、启动键或固定哈希：加 `--no-speed-defaults` 时，vLLM 命令、启动配置及其固定哈希与 `902690a` 逐字节相同（对照旧代码的 CPU 测试；不加时适用 4.15 节的速度默认值）。

**启动日志。** 解耦布局的服务会打印下面几行（作业 3313802：24 GiB、5 个 session），不会出现 vLLM 的 `Setting attention block size to 784 tokens`，也不会出现它的 Mamba 填充日志：

```text
Decoupled KV layout: attention block 16 tokens, Mamba block 784 tokens, 11 Mamba slots, unpadded Mamba page.
Decoupled KV cache pools: 22912 attention blocks, 34 Mamba blocks
GPU KV cache size: 366,592 tokens
```

`ready.json` 记录布局和两个池的大小（第 8 节）。已验证运行的池大小见 4.4 节的表。

**主机内存。** 解耦布局下，worker 把拷到主机的注意力 Page 放在一个锁页的 Page arena 里。上下文上限为 L token 时，它的容量是 S × (⌈L / P⌉ + 1) 个 Page，每个 P × 65,536 字节（`--dram-arena-gib` 可设一个更小的总容量，4.9 节），按需以 256 MiB 为单位分配：P = 16、L = 65,536 时每个 session 最多约 4 GiB（`remote_ra_live_v2/worker_composition.py` 中的 `arena_pages`，`remote_ra_growing_worker/tensors.py` 中的 `PageArena`）。这是在 connector 的锁页 CPU 池之外另加的（默认 1 GiB，加 `--no-connector-pool-gib` 时为 vLLM 的 32 GiB；4.9 和 4.15 节）。

**P = 16 时的选页。** Selector V2 合同以页大小为参数，所以 P = 16 时它就是 PoC 自己的合同：容量 `max(1, 必保留页数, floor(0.40 × 历史 token) // 16)` 页，recent 配额 64 页（1,024 token），live tail 不足 16 token，每个 16 token 的 Page 一个向量（第 5 节）。worker 在 GPU 上每次 forward 一批地把 Page 向量压缩成 int8（选页合同中的 `compact_int8_v1`），而不是在 CPU 上逐个计算。

**注意力后端：`--triton-attn`（可选）。** 在 GH200 上，vLLM 0.24 用 FlashAttention 3 运行 Qwen 的 16 个全注意力层。`--triton-attn`（只用于 P = 16、32 或 64 的解耦布局）改为传入 `--attention-backend TRITON_ATTN`，即组里 baseline 使用的后端；Gated-DeltaNet 层保留自己的后端和 `--gdn-prefill-backend triton` 的 Triton kernel。其余都不变：runner 在两种后端的 metadata builder 读取之前就改写 vLLM 的公共注意力 metadata（物理序列长度、压缩后的 block table 和 slot mapping），worker 按已加载后端的 cache shape 建立 KV 视图（两者每层都是 `(blocks, 2, P, 4, 256)`），两种后端都在同一个注意力 op 和 KV 更新 op 之后运行，所以 CUDA graph 各模式以及 FULL graph 的序列长度绑定都相同（4.8 节）。在 784 token 的 block 上，FlashAttention 用 49 个 16 token 的 kernel block，TRITON_ATTN 用一个 784 token 的，所以那里拒绝这个开关（共享池，以及 784 下的 `--kv-layout decoupled`）。它在 `startup.json` 及其 `effective_server_config` 中固定 `attention_backend`；启动时前端比较 vLLM 解析出的后端，引擎比较自己解析后的后端，worker 比较自己解析后的后端以及每个全注意力组实际加载的后端，各一次；worker 的记录写入 `worker-attached.json` 和 `ready.json`（无论哪种后端，其 `source_closure.backends` 都记下已加载后端模块的名字和哈希）。不加这个开关时什么都不变，而 vLLM 若仍为 Qwen 解析出 TRITON_ATTN 会被拒绝。输出预期不会与 FlashAttention 逐位相同。给 stock vLLM 对照组同样传 `--attention-backend TRITON_ATTN`。Gemma 4 profile 始终运行 TRITON_ATTN（4.14 节），在那里这个开关不改变任何东西。测试：`test_triton_attn.py`。

**Mamba block 原地跨界：`--gdn-inplace-crossing`（可选；只做过 CPU 测试，未经 GPU 验证）。** 在 vLLM 的 `align` 模式下，请求跨过 M 的倍数时会再分到一个 Mamba block，状态拷进去，旧 block 要到下一个引擎步才释放；所以 Mamba 池有 K = 2S + 1 个槽位，容量门控也为每个请求算一个运行槽位加一个跨界槽位。加 `--gdn-inplace-crossing`（只用于解耦布局，Qwen）后，vLLM 的 `MambaManager` 改为把请求唯一的运行 block 挪到新的行索引：不分配也不释放，旧索引变成 null，worker 的跨界拷贝读写的是同一个 block。这样每个运行中的请求只占一个 Gated-DeltaNet 槽位，池里有 K = S + 1 个（+ 1 是 phase 1 的 probe 和它旁边的 Goal 槽位），其余都给注意力池：25 GiB、P = 16 时，16 / 24 / 32 个 session 分别多出 2,349 / 3,523 / 4,698 个注意力 block（+2.29 / +3.44 / +4.59 GiB）。容量门控在 phase 2 只算一个运行槽位（加 `--gdn-device-handoff` 时连同保留的 probe 算两个），运行中请求的 headroom 里不再算跨界槽位；phase 1 不变。worker 不再在 M 对齐的位置抓取状态（槽位从不移动；槽位一旦移动就失败关闭），所以每次跨界省下第二个 block、154 MB 的主机抓取和 154 MB 的写回（开 hand-off 时是拷到新 block 的设备间拷贝）。跨界时 vLLM 仍会把运行状态拷到新索引上的 block，而那现在是同一个 block：这是对整份 Gated-DeltaNet 状态的一次设备上自拷贝（约读 154 MB、写 154 MB），开销与不加开关时那次拷贝相同。在这次拷贝之前，worker 会检查每个请求的行在上一步的索引处仍然指向它的运行 block（vLLM 的 worker 行只会增长），所以行一旦被替换就失败关闭，不会把 null block 拷到状态上；后续轮次的 `restore.h2d_bytes` 是恢复的 Page 加一份 carry，第 0 轮只有 Page（开 hand-off 时 `restore.d2d_bytes` 只有 hand-off 那一次拷贝）。这个开关在 `startup.json` 的 `kv_layout` 记录和 vLLM 的 `--additional-config` 里固定 `"inplace_crossing": true`，`ready.json`（worker 回显的布局）也会重复它；dry run 和 `launch.json` 增加带 K 的 `gdn_inplace_crossing`。不加时（默认）什么都不变：没有这个键，K = 2S + 1，分配（对照 `a6d3ef7` 的 KV manager 检查过）和记录都相同。以下情况拒绝：784 token 的共享池、Gemma 4 profile（它没有 Gated-DeltaNet 状态）、投机解码，以及任何会对 Mamba block 做前缀缓存的地方（runtime 从不缓存它：自己拥有的每个请求都带 `skip_cache` 分配，HTTP runtime 拒绝不归它管的请求；开了这个开关的 Mamba manager 从不给 block 计算哈希，跳过该请求的 Mamba 缓存而不是让引擎停下，跨界时也拒绝已缓存或被共享的运行 block）。它可以和 `--async-scheduling` 一起用（已停止请求被作废的那一步挪动的是它本来就持有的 block，carry 不变），也可以和 `--gdn-device-handoff` 一起用。现在 Mamba 池可能让门控等待，而 2S + 1 时从不会：开 hand-off 时，处在 phase 2 第一步的请求在运行槽位旁还持有它的 probe；开 `--admission-overlap` 时，每个在途的准入都预留它的 phase 2 峰值，所以重叠的准入会比不加开关时更早在 Mamba 池上等待（`CAPACITY_WAIT` 会写明 Mamba 池；不会超额分配）。例如 4 个 session、两个在途准入各持有 probe、没有请求在 decode 时，第三个准入在开关打开时（5 个槽位）要等待，不加开关时（9 个）会被接纳。测试：`test_gdn_inplace.py`（真实的 `MambaManager`、16 / 24 / 32 个 session 下的门控、一个和三个 session 的端到端、async 以及 hand-off）。在任何基准测试使用它之前，它需要自己的 GPU 测试（与不加它的服务逐 token 对照）。最多 32 个 session 时（integrate-1116；只在 CPU 上）：32 个 session 时 K = S + 1 = 33；在 262,144 token、4,096 token 回复下，24 GiB 和 25 GiB 时启动器都接受直到 32 的每个 K（不加这个开关时 25 GiB 只接受 K ≤ 25，24 GiB 只接受 K ≤ 20，见 `running.md` 第 4.4 节）；在 fake 引擎上，`goal_query` 24 个 session 不加开关时会为 Mamba 池等待一次，加了则从不等待；32 个 session 在同一个 batch 里解码，每四个 session 中有一个带格式错误回滚（局部第 0 轮的重建）（`test_inplace_32.py`）。

**GPU 上的运行。** 作业结果见 [docs/status.zh-CN.md](status.zh-CN.md)，限制见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md)（2.1、2.2）；本节在 2026-11-16 之前记述的 GPU 运行见[冻结说明 4.6](history/reference-gpu-notes.zh-CN.md#出自-46-节解耦-kv-布局与-16-token-的-page)。

### 4.7 16 token Page 的验证

16 token Page 的 GPU 冒烟测试、已验证的运行和 SWE-bench pilot（runtime `8f1af6a`、`780fe54` 和 `b37044f`）以及它们的性能说明冻结在[冻结说明 4.7](history/reference-gpu-notes.zh-CN.md#出自-47-节16-token-page-的验证)。16 token 的 Page 本身见 4.6 节；作业结果见 [docs/status.zh-CN.md](status.zh-CN.md)。

### 4.8 CUDA graph

`--cuda-graphs off`（加 `--no-speed-defaults` 时的默认；不加时默认是 `full`，4.15 节）就是上面的 eager 服务：vLLM 命令、`startup.json`、其固定哈希和证据都与不加该参数的 `--no-speed-defaults` 启动逐字节相同（对照 `b37044f` 的 CPU 测试，`test_cudagraph.py`）。

`--cuda-graphs piecewise`、`full` 或 `full-decode` 把 vLLM 命令里的 `--enforce-eager` 换成固定的 `--compilation-config`（它不改变调度方式：默认是同步调度，除非启动时加 `--async-scheduling`，开 CUDA graph 的服务同样可以加这个参数；见 4.15 节）：

| 模式 | `--compilation-config` | vLLM 0.24 捕获什么 |
| --- | --- | --- |
| `piecewise` | `{"cudagraph_mode":"PIECEWISE","mode":"VLLM_COMPILE"}` | FlashAttention、KV cache 写入和 Gated-DeltaNet 核心之间的已编译分段；这三者每步都以 eager 方式运行，读取该步的选择性序列长度、slot mapping、block table 和 Mamba 状态索引 |
| `full` | `{"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}`（stock vLLM 的默认值） | 整个 uniform decode 步，包括注意力；混合步仍是分段。runner 在捕获和重放时都把每个全注意力 group 绑定到其物理序列长度缓冲区 |
| `full-decode` | `{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"NONE"}` | 整个 uniform decode 步，使用 eager kernel，不做 torch.compile；prefill、混合步和 phase 1 与 `off` 一样以 eager 方式运行。绑定方式与 `full` 相同 |

- **固定值与检查。** `startup.json` 记录 `cuda_graphs`，有效配置记录 `enforce_eager: false` 和该编译配置。启动时 frontend 比对 vLLM 解析出的参数，worker 检查解析后的配置（编译模式、CUDA graph 模式、捕获尺寸，以及编译时留在 graph 之外的三个 op），任何差异都拒绝启动；参数为 off 时，worker 拒绝已捕获 graph 的 runner。
- **证据。** `worker-attached.json` 多一条 `cuda_graphs` 记录（模式、解析后的配置、捕获尺寸）；不做任何逐请求或逐步的检查。
- **多个 session。** vLLM 按 `max_num_seqs` 确定捕获的 batch 尺寸：每个启动 task 一个，使用 `--max-running-sessions K` 时为 K（[第 4.4 节](running.zh-CN.md#44-一个服务上的多个-session)）。尺寸为 1、2、4 以及不超过 2 × `max_num_seqs` 的 8 的倍数（16 个 session 时为 1、2、4、8、16、24、32），所以最多 16 个运行中 session 的每个 decode 步都有对应的 graph（CPU 测试 `test_int_flags.py`，使用 vLLM 0.24 自身的尺寸计算代码）。超过 16 个时（`MAX_SESSIONS` 为 32，未经 GPU 验证）：K = 20 时到 40，K = 24 时到 48，K = 32 时到 64（CPU 测试 `test_ms32.py`）。vLLM 0.24 的 dispatcher（`vllm/v1/cudagraph_dispatcher.py`）只为不超过 `max_num_seqs` 的尺寸捕获 FULL uniform-decode graph：K = 24 和 32 时 K 本身就是一个尺寸，所以 `full` 模式下每个最多 K 个请求的 decode 步都重放 FULL graph；K = 20 时（与 K = 12 时一样）17 到 20 个请求的一步会填充到 24，重放 piecewise graph。多出来的尺寸只是 vLLM 共用 graph 池里的小 graph（原版 vLLM 捕获到 512 也只用 0.13 GiB）。
- **Phase 1**（以及 Goal forward）不使用 CUDA graph，也不做填充：`piecewise` 和 `full` 下运行已编译代码，`full-decode` 下运行 eager kernel。decode 步（以及除 `full-decode` 外的小混合步）使用 graph。使用 `--phase1-in-step`（4.10 节）时，并入 decode 步的阶段 1 随该步一起运行：`piecewise` 和 `full` 下使用分段 graph，`full-decode` 下以 eager 方式运行。使用 `--phase1-graph-tokens T`（4.10 节；需显式开启，仅限 `piecewise` 和 `full`）时，vLLM 还会捕获最多 T 个 token 的分段 graph，不超过 T 个 token 的单独阶段 1 或 Goal forward 会被填充并重放其中一个。
- **结果。** torch.compile（`piecewise`、`full`）用编译生成的 kernel 代替 vLLM 的自定义 kernel，会改变 bf16 数值，所以 greedy 输出可能与 eager 服务不同。`full-decode` 保留 eager kernel，只改变捕获本身带来的变化：batch 不超过最大捕获尺寸时 FlashAttention 的 split 上限，以及填充到捕获尺寸。prompt 和 transcript 的 token 精确性与模式无关。使用 torch.compile 时，编译缓存为冷时启动大约多一分钟（单张 GH200 上的 stock vLLM 0.24.0：+58 s；捕获 4 s、0.13 GiB）。
- **对比。** graph 模式的输出，连同 query 策略的选页，都可能与 eager 服务不同，所以 CUDA graph 服务应与同一模式下的 stock vLLM 对比。即使在 eager 模式下，成批运行在两次运行之间也不是 token 完全相同的（4.4 节）。
- **Gemma 4**（4.14 节）在其 window 布局上接受相同的模式，绑定的是它的全局组。

**GPU 上的运行。** 作业结果见 [docs/status.zh-CN.md](status.zh-CN.md)，限制见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md)（2.3）；本节在 2026-11-16 之前记述的 GPU 运行见[冻结说明 4.8](history/reference-gpu-notes.zh-CN.md#出自-48-节cuda-graph)。

### 4.9 上下文上限与回复上限

三个 launcher 选项让服务能承载更长的 agent 运行。不使用它们时，vLLM 命令、`startup.json`、其中的固定值、证据文件和选页结果都与 `8c5f4d0` 逐字节相同（CPU 测试）。`--max-model-len 262144` 是后来加入的第三个上下文选项；加入它之后，默认配置和 131072 的配置都保持不变（CPU 测试，`test_ctx_256k.py`）。

| 选项 | 默认 | 另一选择 | 固定到哪里 |
| --- | --- | --- | --- |
| `--max-tokens` | 1536（PoC 的回复上限） | 4096 | `startup.json` 的 `max_tokens` |
| `--max-model-len` | 65536 | 131072 或 262144 | vLLM 的 `--max-model-len` 和 `effective_server_config.max_model_len` |
| `--dram-arena-gib N` | 为每个运行中的 session 在上下文上限处留足空间 | 所有 session 共用一个 N GiB 的锁页 Page arena，仅限解耦布局 | `startup.json` 的 `dram_arena_pages` |

- **KV cache 大小（`--kv-cache-gib N`）。** 这些上限必须放进的 GPU KV cache：默认 8 GiB，Qwen profile 还可取 16、24、25 或 32；Gemma 4 profile 取 24、25、28 或 32（4.14 节）。下面的拒绝条件和容量表说明哪些大小能承载给定的上下文上限和 session 数。
- **如何选择上限。** 上限越长，同样的 session 需要的 GPU KV 和锁页主机内存越多（见下面的拒绝条件），所以应选能容纳 agent 历史（prompt 加回复）的最短上限：4096 token 回复时，65536 下 prompt 最多 61,440 token，131072 下最多 126,976，262144 下最多 258,048。262144 是模型的原生上下文，没有更长的选项。
- **回复上限。** 这是服务端设置，不属于 agent 协议：前端的 live controller 和引擎持有固定值；Mini-SWE-Agent XML 路由在每个请求上强制使用它，引擎在分配之前拒绝任何其他值。服务端仍然贪心解码、忽略客户端的采样字段。容量门控为每个请求预留到 ⌈(prompt + max_tokens) / P⌉ 的 Page，launcher 的余量是一次回复，⌈max_tokens / P⌉ + 1 个 block（16 token Page 下 1536 时为 97，4096 时为 257）。vLLM 把回复上限截断为上下文上限减去 prompt，所以 prompt 最多为上下文上限减去回复上限（65,536、4096 token 回复时为 61,440 token；131,072 时为 126,976；262,144 时为 258,048）。
- **启动时检查，只做一次。** 前端把 vLLM 解析出的 `--max-model-len` 与固定值比较；引擎检查自己解析后的 `max_model_len`，在任一值不是默认值时，把 context 记录（`max_model_len`、`max_tokens`）放进发给 worker 的启动记录；worker 用自己的 runner 核对这条记录，并在 `worker-attached.json` 中回显；引擎和前端都比较这份回显，`ready.json` 增加 `context`。262144 token 的服务还会在前端和引擎中各设置一次自己的请求上限（见下）。不做任何逐请求或逐步的检查。Page 目录上限、注意力和 Gated-DeltaNet 的行以及锁页 Page arena 都随 vLLM 的 `max_model_len` 而定（P = 16、131072 token 时，Page 目录为 8,192 个 Page，arena 每个 session 8,193 个 Page；262144 时分别为 16,384 和 16,385 个）。
- **launcher 的拒绝条件。** 一个 session 在上下文上限处的一轮必须放得进注意力池（解耦布局：⌈L / P⌉ + 1 + ⌈16,384 / P⌉ + ⌈max_tokens / P⌉ + 1 个 block，加上 1 % watermark）或共享的 784 token 池（⌈L / 784⌉ + 1 + 6 + ⌈max_tokens / 784⌉ + 1 个 block 加 watermark，也就是 vLLM 自己启动检查的要求）；在 131072 token 下，这在所有布局里都会拒绝 `--kv-cache-gib 8`。最坏情况下的锁页主机内存（connector 的 CPU 池、每个运行中 session 一份 154 MB 的 Gated-DeltaNet carry（开 `--gdn-device-handoff` 时两份），以及所有运行中 session 在上限处每个注意力 Page 的主机副本）必须不超过 128 GiB（`serve.py` 中的 `PINNED_HOST_GIB`；已验证的作业申请了 200 GiB）。本条和下面 262144 那一条的数值都按 vLLM 的 32 GiB 池计算（`--no-connector-pool-gib`）；默认的 1 GiB 池少需 31 GiB（下一条）。在 65536 token 下，最多 16 个运行中的 session 最多需要 99.4 GiB（开 `--gdn-device-handoff` 时 101.7 GiB），都能通过。chat-text 的 session 池最多可运行 24 个（`remote_ra_live_v2/sessions.py` 中的 `MAX_CHAT_TEXT_RUNNING`）；在 65536 token、P = 16 时预算能放下其中 23 个（开 `--gdn-device-handoff` 或用 784 token Page 时为 22 个）：20 个需要 115.1 GiB，24 个需要 131.7 GiB。在 131072 token 下，11 个运行中的 session 放得下（P = 16 时 121.8 GiB），16 个需要 162.5 GiB，因此 16 个 session 需要 `--dram-arena-gib`（88 GiB 时为 122.3 GiB）。超出限额 arena 的捕获会让该请求失败（`DRAM_ARENA_EXHAUSTED`）；不会换出或淘汰任何东西。16 个 session 平均 90,000 token 时共持有 87.9 GiB 的 Page。如果 `--dram-arena-gib` 本身超出预算，拒绝信息会给出放得下的最大 arena（16 个 session 时为 93 GiB，开 `--gdn-device-handoff` 时为 91 GiB）。
- **connector 的 CPU 池（`--connector-pool-gib N`；默认为 1，4.15 节，`--no-connector-pool-gib` 恢复 vLLM 的 32 GiB；每个 benchmark 服务都用它）。** 上面 connector 的 CPU 池属于 vLLM 的 `OffloadingConnector`，`serve.py` 总是配置它（`--kv-transfer-config`）；worker 启动时 vLLM 会锁页并清零这个池，与 `--kv-cache-gib` 无关。本 runtime 从不经过这个池搬运数据。它在启动时隔离 connector（`remote_ra_http/connector.py`）：每次前缀查找都返回 0 个 token，任何从池中加载的请求都会被拒绝，结束的请求什么也不存，connector 也从不拿到传输任务。拷到主机和从主机恢复的 Page 走的是 runtime 自己的锁页内存：4.6 节的 Page arena（共享池中是每个捕获的 Page 一份锁页副本）和 Gated-DeltaNet carry。connector 本身保留，因为 runtime 的隔离和检查挂在它上面。
  - `--connector-pool-gib N`（N 为 1 到 31；默认为 1，4.15 节）把池设为 N GiB 而不是 32 GiB，预算也按 N GiB 计算。1 GiB 仍能放下 20 个 784 token 的 block，或 1,024 个 16 token 的 block，在每种 `--kv-cache-gib` 下每个服务都省下约 31 GiB 锁页主机内存。在 65536 token 下，最多 16 个运行中的 session 最多需要 68.4 GiB，而不是 99.4 GiB（开 `--gdn-device-handoff` 时 70.7 GiB，而不是 101.7 GiB），24 个 session 的 chat-text 池最多需要 105.5 GiB，所以所有配置都能通过；在 131072 token 下，不用 `--dram-arena-gib` 能放下 15 个运行中的 session，16 个 session 可用的 arena 最大为 124 GiB（开 `--gdn-device-handoff` 时 122 GiB）；在 262144 token 下，不用 arena 能放下 7 个。对只申请一张 GH200 对应份额节点主机内存的作业，这一点很重要。
  - `startup.json` 记录 `connector_pool_bytes`，vLLM 得到的 `cpu_bytes_to_use` 为 N GiB。engine 检查自己的 vLLM 配置是否就是固定的值，worker 检查分配出的池里每个 KV 张量至少有一个 block、总量不超过 N GiB，各在启动时做一次（`remote_ra_live_v2/connector_guard.py`）。加 `--no-connector-pool-gib`（或 `--no-speed-defaults`）时，vLLM 命令、启动配置和 launcher 的拒绝信息都与以前 32 GiB 池的服务相同。CPU 测试：`test_connector_pool.py`。
- **token 与帧大小上限**（在 CPU 上用固定的 tokenizer 测得，测试 `test_align_ctx.TokenFrameCapTests`）：126,976 token 的 Mini-SWE-Agent 请求，observation 为 SWE 类内容（代码清单、带填充的 `pytest -v` 输出、FITS 头卡片，每条最多 10,000 字符）时，每个 token 占 3.3 到 5.3 字节 source，不到 1 MiB source 上限中的 0.66 MiB（测试的界限）；core 请求不到 0.3 MB（上限 1 MiB；131,072 个 id 最多 0.66 MB）；帧不到 1.5 MB（上限 4 MiB）；126,976 token 也在 131,072 token 的上限之内。在 126,976 token 时，source 上限在每 token 超过 8.26 字节时生效（64,000 token 时为 16.4）：只有横幅分隔线的日志（每个 token 一条 80 列分隔线，约每 token 19 字节）在任一上下文下都会在准入前被拒绝。这些是 65536 和 131072 下的上限，没有改动；262144 token 的服务把它们加倍（下一条）。
- **262144 token。** `--max-model-len 262144` 是模型的原生上下文（`max_position_embeddings` 为 262144，rope 为默认类型）：不用 YaRN，不需要覆盖参数，也没有新的固定值。更长的上下文需要 YaRN，而 YaRN 会改变所有长度下的输出。和 131072 一样，只有 vLLM 的 `--max-model-len` 和 `effective_server_config.max_model_len` 改变，Page 目录（P = 16 时 16,384 个 Page）、各种行和 arena 都随 vLLM 的 `max_model_len` 而定。单个请求的字节和 token 上限随固定的上下文上限缩放（见下表）。在 CPU 上用固定的 tokenizer 测得（测试 `test_ctx_256k.TokenFrameCapTests256`）：258,048 token 的请求（262,144 减去 4096 token 的回复）每个 token 需要 3.3（代码清单、README 文本）到 5.2（FITS 头卡片）字节 source，即 0.84 到 1.35 MB 的 source，因此带填充的历史和 FITS 历史都超过了原来的 1 MiB。core 请求最多 0.61 MB，帧最多 2.9 MB。2 MiB 的 source 上限在每 token 超过 8.13 字节时生效，所以只有横幅分隔线的日志仍会在准入前被拒绝。
  - 262144 token 下 launcher 的拒绝条件。一个 session 的一轮需要 16,385 + 1,024 + 257 个 block 加 watermark（16 GiB、4096 token 回复时为 17,824），所以 `--kv-cache-gib 8` 和 `16` 在所有布局、所有 session 数下都会被拒绝；24、25 和 32 在 1 到 16 个 session 时都能通过。共享的 784 token 池在 4096 token 回复时需要 352 个 block（16 GiB 时可用 333 个，24 GiB 时 500 个）。
  - 262144 token 下的锁页主机内存：不设 arena 时最多放得下 5 个运行中的 session（P = 16 时 113.0 GiB）。6 个需要 129.1 GiB，16 个需要 290.5 GiB；session 池按其 K 个运行中的 session 计算。16 个 session 需要 `--dram-arena-gib`，取值在 17 GiB（上限处的一个 session，16,385 个 Page）到 93 GiB 之间（开 `--gdn-device-handoff` 时为 91 GiB）。93 GiB 平均每个 session 可容纳 95,232 token，88 GiB 为 90,112。16 个 session 都到 262,144 token 是放不下的：仅它们的 Page 就有 256 GiB。
  - 尚未改动：300 s 的准入等待和 360 s 的租约；phase 1 对每条追加消息仍最多处理 16,384 token；GPU 之外随 prompt 增长的逐请求工作（prompt 哈希、JSON 快照、反 tokenize、带全部 prompt id 的证据行）；arena 的公平性（`DRAM_ARENA_EXHAUSTED` 让下一个捕获的 session 失败，不论是谁）；严格规则下 262K 重建的容量（17,409 个 block）。
- **终身请求上限。** T 个注册任务时为 max(1,024, 256 × T)：单个 session 256 个请求的上限（前端的轮次上限、引擎的轮次序号和 generation 上限）对所有任务求和，因此它永远不会拒绝一个 session 仍可发出的请求。32 个 120 步的 agent 任务最多发出 3,840 个请求，上限为 8,192；`CAPACITY_WAIT` 重试不占用终身记录。单个 session 的上限仍是 256 个请求，所以 `run_agents.sh` 会拒绝大于 256 的步数上限。
- **SWE-bench。** `run_agents.sh` 接受 `STEP_LIMIT`（默认 90）和 `MAX_TOKENS`（默认 1536），用于 model override（7.4 节）。

**131,072 token、16 个 session、P = 16、4096 token 回复时的容量**（按 4.4 节的规则推导：15 个空闲 session 加一个处于全上下文峰值的准入轮次再加余量，必须放进可用注意力 block；Mamba 池占 100 个 block）：

| 策略与空闲模型 | 每个 session 的空闲 block | 所需 KV GiB | 24 放得下 | 25 放得下 | 32 放得下 |
| --- | ---: | ---: | :-: | :-: | :-: |
| `full`（严格规则下所有策略都如此，因为重建会回到全部驻留） | 8,193 | 135.4 | 否 | 否 | 否 |
| 40 %，只算预算 | 3,277 | 62.6 | 否 | 否 | 否 |
| 40 % + 典型的上一轮（7K token） | 3,715 | 69.1 | 否 | 否 | 否 |
| 40 % + 最坏的上一轮（16,384 + 4,096） | 4,557 | 81.6 | 否 | 否 | 否 |
| `--active-token-budget 8192`，只算预算 | 513 | 21.8 | 是 | 是 | 是 |
| 预算 8192 + 典型的上一轮 | 951 | 28.2 | 否 | 否 | 是 |
| 预算 8192 + 最坏的上一轮 | 1,793 | 40.8 | 否 | 否 | 否 |

25 GiB（integrate-1026；组里的运行用它）在 16 个 session 时有 20,706 个注意力 block，比 24 GiB 多 1,024 个：按同一规则，只算预算时能容纳 20 个 session，加典型的上一轮 13 个，加最坏的上一轮 8 个（24 GiB：18、12、7）。这些数字按 16 个 session 时的注意力 block 计算：超过 16 个时 Mamba 池占得更多（[4.4 节](running.zh-CN.md#44-一个服务上的多个-session)），而 XML 和 kv-session 协议在 GPU 上运行过的最多是 16 个（chat-text：K = 20；[已知限制 2.1](known-limits.zh-CN.md#21-超过-16-个-session-和---gdn-inplace-crossing-都只经过-cpu-测试)）。

65,536 token、4096 token 回复时，同样各行需要 70.8、34.4、40.9、53.4、17.8、24.2 和 36.6 GiB（1536 token 回复时为 4.4 节的数值）。无论上下文上限多大，16 个 `full` session 只有在上下文较短时才不会死锁（24 GiB 时每个 session 约 18K token，32 GiB 时约 26K）。即使在 Goal rewriter 旁边还能留出约 36 GiB 给 KV（只是估计，未实测；launcher 最多接受 32），131,072 token 下的 16 个 session 也只有在使用较小的固定预算、且每轮远低于 observation 上限时才不会死锁（见上表：典型的最后一轮需要 28.2 GiB，最坏的需要 40.8 GiB）。

**按上下文上限列出的单个请求上限**（`remote_ra_live_v2/context_limits.py` 中的 `request_caps`；262144 的值由前端和引擎在启动时各设置一次，65536 和 131072 下不设置任何东西）：

| 上限 | 65536 和 131072 | 262144 |
| --- | ---: | ---: |
| XML source（对话 JSON）和每条编码的消息：max(1 MiB, 8 B × L) | 1 MiB | 2 MiB |
| core 请求（msgpack，每个 prompt id 1 到 5 字节）：max(1 MiB, 8 B × L) | 1 MiB | 2 MiB |
| 认证帧：2 × (source + core) | 4 MiB | 8 MiB |
| EngineCore 的 registered-add 消息：帧 + 1 MiB | 5 MiB | 9 MiB |
| 每个请求的 prompt id：max(131,072, L) | 131,072 | 262,144 |

**262,144 token、16 个 session、P = 16、4096 token 回复时的容量**（按同一规则推导）。准入轮次的全上下文峰值本身就有 17,409 个 block（17 GiB）。最后一列是 24、25 和 32 GiB 下最多放得下的 session 数（最多数到 16 个 session，Mamba 池按 16 个 session 计；超过 16 个见 4.4 节）：

| 策略与空闲模型 | 每个 session 的空闲 block | 所需 KV GiB | 24 放得下 | 25 放得下 | 32 放得下 | 24 / 25 / 32 GiB 下的 session 数 |
| --- | ---: | ---: | :-: | :-: | :-: | :-: |
| `full`（严格规则下所有策略都如此） | 16,385 | 264.6 | 否 | 否 | 否 | 1 / 1 / 1 |
| 40 %，只算预算 | 6,554 | 119.2 | 否 | 否 | 否 | 1 / 2 / 3 |
| 40 % + 典型的上一轮（7K token） | 6,992 | 125.7 | 否 | 否 | 否 | 1 / 1 / 2 |
| 40 % + 最坏的上一轮（16,384 + 4,096） | 7,834 | 138.1 | 否 | 否 | 否 | 1 / 1 / 2 |
| `--active-token-budget 8192`，只算预算 | 513 | 29.8 | 否 | 否 | 是 | 8 / 9 / 16 |
| 预算 8192 + 典型的上一轮 | 951 | 36.3 | 否 | 否 | 否 | 5 / 6 / 12 |
| 预算 8192 + 最坏的上一轮 | 1,793 | 48.7 | 否 | 否 | 否 | 3 / 4 / 7 |

这些行中只有 8192 token 的预算、且空闲 session 只持有自己的预算时，262,144 token 下的 16 个 session 才不会死锁（32 GiB 时）。

**GPU 上的运行。** 作业结果见 [docs/status.zh-CN.md](status.zh-CN.md)，限制见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md)；本节在 2026-11-16 之前记述的 GPU 运行见[冻结说明 4.9](history/reference-gpu-notes.zh-CN.md#出自-49-节上下文上限与回复上限)。

### 4.10 Admission 优化（需显式开启）

admission 是引擎在新一轮 decode 之前要做的工作：阶段 1，`goal_query` 的 Goal，选页和预留（第 5 节）。默认情况下，admission 按到达顺序一次一个；阶段 1 是一次单独的 forward，期间正在运行的请求都要等待；Gated-DeltaNet 状态经锁页主机内存从阶段 1 交给阶段 2。四个 launcher 参数可以改变这些，默认都关闭（括号里的字母是代码注释中对应的 admission 设计编号）：

| 参数 | 默认 | 作用 | 启动时固定到哪里 |
| --- | --- | --- | --- |
| `--gdn-device-handoff`（e） | 关 | 在 device 上把 Gated-DeltaNet 状态从阶段 1 交给阶段 2，不经主机拷贝 | `startup.json` 的 `admission.gdn_device_handoff` = `true` |
| `--phase1-in-step`（b） | 关 | 把一次 admission 的阶段 1 放进下一个 decode 步里运行，而不是单独一次 | `admission.phase1_in_step` = `true` |
| `--admission-overlap K`（f） | 1（关） | 最多同时有 K 个 admission 在进行，2 ≤ K ≤ 同时运行的 session 数 | `admission.admission_overlap` = `{"permits": K, "admission_seconds": 300 + 120(K − 1), "lease_seconds": 360 + 120(K − 1)}` |
| `--goal-overlap` | 关 | `goal_query` 加 `--goal-rewriter-model`：在构造 Goal 的同时运行该轮的阶段 1，之后 Goal forward 单独运行一次 | `admission.goal_overlap` = `true` |
| `--phase1-graph-tokens T`（d） | 0（关） | Qwen 加 `--cuda-graphs piecewise` 或 `full`：vLLM 还会捕获最多 T 个 token 的分段 CUDA graph，不超过 T 个 token 的单独阶段 1 或 Goal forward 用这些 graph 运行，而不是不用 graph（未经 GPU 验证；见下面它的条目） | `phase1_graphs` = `{"capture_sizes": [...], "tokens": T}` 和编译配置的 `cudagraph_capture_sizes`（不在 `admission` 中） |

只有在开了某个参数时，`startup.json` 才有 `admission` 记录；全部关闭时，启动配置、证据文件和 vLLM 命令都不会多出任何内容。`--phase1-graph-tokens` 固定的是它自己的记录 `phase1_graphs`，与 `cuda_graphs` 并列（见下面它的条目）。引擎和 worker 在启动时各检查一次这条记录，worker 在 `worker-attached.json` 中回显它，`ready.json` 记录它；不做任何逐请求或逐步的检查（`remote_ra_live_v2/admission_flags.py`；CPU 测试 `test_gdn_handoff.py`、`test_adm_phase1_step.py`、`test_adm_overlap.py`、`test_goal_overlap.py`）。`examples/slurm/serve_gh200.sh` 不传这些参数中的任何一个（也不传 `--cuda-graphs`），所以它的服务按 4.15 节的速度默认值运行（CUDA graph `full` 和解码快速路径）。

- **`--gdn-device-handoff`。** 阶段 1 把它的 probe slot 留在 GPU 上，阶段 2 的第一步在 device 上把它拷贝到阶段 2 的 slot；跨 Mamba block 时也在 device 上拷贝。提交后的 carry 是唯一的主机副本：一轮结束时它被异步拷到主机，写入该 session 两个锁页 carry 缓冲区中没有存放已提交 carry 的那一个，并由 drain 等待完成；阶段 1 从那里恢复它（`remote_ra_live_v2/gdn_handoff.py`）。延续轮次的阶段 2 占用三个 Gated-DeltaNet slot（运行、跨块和保留的 probe slot），容量门控也按三个计算。launcher 的锁页主机内存预算（4.9 节）计入第二个 carry 缓冲区：每个运行中的 session 多 153.9 MB（16 个 session 时 2.3 GiB；131,072 token、16 个 session、`--dram-arena-gib 88` 时需要 124.6 GiB，而不是 122.3）。索引恢复时临时使用的锁页中转副本在恢复完成后释放，不计入预算。每个 `runtime.jsonl` 行增加 `restore.d2d_bytes`（第 8 节）。
- **`--phase1-in-step`。** admission 把它的阶段 1 挂起，引擎把它作为额外一行，和正在运行的请求一起放进下一个 decode 步（混合步）；该步收集到结果后，admission 继续（`remote_ra_live_v2/phase1_step.py`）。以下情况阶段 1 仍单独运行：没有其他请求在运行（`probe_only`），它的 token 放不进该步在每步 16,384 token 预算中剩下的部分，或者连续两步都没有空闲的 batch 行（`fallback`）；带有 `goal_query` Goal forward 的阶段 1 也仍单独运行，一次只运行一个 session 的服务（`max_num_seqs` 为 1）总是单独运行。阶段 1 从不分块。使用 CUDA graph 时，带有阶段 1 的步使用分段 graph（`piecewise`、`full`）或以 eager 方式运行（`full-decode`）。native 文件记录 `phase1_mode` 和挂起时间。
- **`--admission-overlap K`。** 最多同时有 K 个 admission 在进行：前端的 admission 锁变成一个有 K 个许可、先到先得的信号量，每个 session 有自己的 Goal 线程和自己的一份 Goal 构造器，所以一个 admission 在等它的 Goal（一次 rewriter 调用）或它的步内阶段 1 时，其他 admission 可以继续（`remote_ra_live_v2/admission_overlap.py`）。Goal rewriter 仍是一个一次只服务一个请求的进程，阶段 1 也仍然一次只为一个 admission 运行。每次 rewriter 调用等待该进程的时间和调用本身的用时，按调用记录在 `goals.jsonl`（`wait_ms`、`call_ms`），按请求记录在 native 文件（`goal_rewrite_wait_ms`、`goal_rewrite_call_ms`，与 `goal_construct_ms` 并列；第 8 节）。不开这个参数时同一时间只构造一个 Goal，所以 `wait_ms` 接近 0，排队发生在引擎前面的 admission 等待里。前端等待 admission 确认的时间和引擎租约随 K 增长，并在启动时固定：300 + 120(K − 1) 秒，租约再多 60 秒（K = 16 时为 2,100 秒和 2,160 秒；不开这个参数时为 300 秒和 360 秒）。不会发生抢占：每个进行中的 admission 在被接受之前都持有一份容量预留；放不下的 admission（与正在运行的请求和其他预留一起算）会以 `CAPACITY_WAIT` 被拒绝并重试，与 4.4 节相同。三个参数都开时，一个 admission 预留两个峰值中较大的那个；在 Mamba 池中，阶段 2 为 9 个 block（三个 Gated-DeltaNet slot；不开 `--gdn-device-handoff` 时为 6 个），阶段 1 为 3 个（有 Goal forward 时为 6 个）。因此 5 个 session 时（11 个 slot，33 个 block）同时只能容纳约 3 个 admission，16 个 session 时（33 个 slot，100 个 block）约 11 个，其余的重试。每次拒绝都会在服务日志中写一行带 `traceback` 字段的 `engine_rejected`（6.3 节）：这是预期行为，不是崩溃。
- **`--goal-overlap`（Gemma 4 profile 也接受它，见 4.14 节）。** 用于 `goal_query` 加 `--goal-rewriter-model`，且多个 session 一起解码的服务；一次只运行一个 session 时它不改变任何东西。
  - **顺序。** 一轮在原来的位置、在自己的 Goal 线程上开始构造 Goal，但不再等它。阶段 1 同时运行：就地运行，或开 `--phase1-in-step` 时放进 decode 步里，需要 Goal forward 的轮次现在也可以这样。之后 admission 只在 Goal 还没构造好时才等它。如果该轮需要 Goal forward（对 Goal 查询文本做的冷 forward，它给出查询向量），这次 forward 单独作为一次短的 probe-only pass 运行，然后选页。
  - **结果相同。** Goal 的构造与以前完全相同。Goal forward 一直在它自己的 block 上从位置 0 开始运行，不读历史 KV、新 token 或 Gated-DeltaNet 状态。单独运行时它给出相同的 Goal 向量，因此 Page 打分和选页也相同。
  - **收益与代价。** Goal 比阶段 1 慢的轮次，大约早一个阶段 1 的时间被接受：节省的时间最多是 `goal_construct_ms` 和 `goal_overlap_ms`（重叠窗口，从 Goal 开始到 join）中较小的那个。有 Goal forward 的轮次要多一次 probe-only pass。实测的窗口见[冻结说明 4.10](history/reference-gpu-notes.zh-CN.md#出自-410-节admission-优化)。
  - **租约。** 开 `--admission-overlap K` 时，阶段 1 不再等待那些已排空自己的阶段 1、正在等 Goal 的 admission，只等待尚未排空的租约。
  - **证据。** native 文件增加 `goal_overlap_ms`；有 Goal forward 的轮次还增加 `goal_forward_ms` 和 worker 帧 `goal_forward`，其中包含这次 forward 的 drain（只有 Goal forward 失败后才会再出现一个对应的 `phase1_drain` 帧）。`runtime.jsonl` 行与不开这个参数的服务相同（第 8 节）。
  - **不增加哈希。** Goal forward 的租约和返回既不签名也不做摘要：worker 逐字段对照本轮阶段 1 的租约检查它们。它的 SHA-256 和 HMAC 调用与不开这个参数的服务相同（CPU 测试会统计）。
  - **源码与测试。** `remote_ra_live_v2/goal_overlap.py`；设计和等价性论证见 `docs/goal-overlap/DESIGN.md`（Gemma 4 profile 见其第 9 节）。CPU 测试（`test_goal_overlap.py`；Gemma：`test_gemma_goal_overlap.py`）检查每一轮的 Goal、Goal 向量、选页和 token 都与不开这个参数的服务相同，开或不开另外三个参数都一样。
  - **开启 thinking 时（integrate-1031 和 keep-overlap）。** 它可与 `--thinking-history strip`（chat-text，Qwen）组合：overlap 只改变 phase 1 何时运行，不改变它的计算；已 drain 的 phase-1 记录保留 strip 完成时在切点提交的 phase-1 Gated-DeltaNet 状态（6.9 节）。`test_goal_overlap_thinking.py` 在开启该选项的 strip server 上运行带摘要、处于 pool 压力下的 Terminus 2 `goal_query` 会话（单独开启，以及与 `--admission-overlap 4` 一起）：每项 strip 检查都成立，每个需要 Goal forward 的 Goal 都在 phase 1 drain 之后单独运行它，每轮的 Goal、Goal hash 和选择与不开该选项的同一运行相同。它也可与只开 `--enable-thinking`（第一阶段的 keep 模式，chat-text 和 mswea-xml；keep-overlap）组合：keep 模式只改变 transcript、回复及其证据，引擎和 worker 走 thinking 关闭时的路径，Goal 的输入在同一时刻、由同一条上一轮回复（其 content）构造。`test_goal_overlap_thinking.py` 在 keep server 上运行同样的 Terminus 2 会话，其客户端把每条 reasoning 发回（单独开启，以及与 `--admission-overlap 4` 一起）；并在挂起 Goal 的 `goal_query` server 上同时运行三个 mswea-xml（SWE-bench）会话：一个每次都发回 reasoning，一个丢掉一次，一个改动一次（各引起一次 rebuild）。每个 prompt 都是保留 reasoning 时模板的渲染，发回 reasoning 的轮次为 append、其余为 rebuild，每轮的 Goal、Goal 向量、选择、prompt 和输出 token 与不开该选项的同一运行相同。变异（跳过 Goal forward、forward 时用了过期的 Goal、忽略 reasoning 的 append 规则、XML 的 marker stop 未在 `</think>` 之后启用、Goal 读到 reasoning）都会使它失败。
- **`--phase1-graph-tokens T`（方案 (d)；默认关闭，即 0；Qwen profile 加 `--cuda-graphs piecewise` 或 `full`；未经 GPU 验证：作业 3569412 在 `b6f6b9f` 上运行过它，部分通过（[docs/status.zh-CN.md](status.zh-CN.md)；其发现项见[历史](history.zh-CN.md#integrate-11122026-11-12)））。** 不开时，单独运行的阶段 1 以及 Goal forward 不使用 CUDA graph（4.8 节），超过 vLLM 最大捕获尺寸（2 × `max_num_seqs` 个 token，16 个 session 时为 32）的步也都不使用 graph。在作业 3539242 和 3461703（SWE-bench，`recent_fill` 和 `observation_query` 服务）以及 3461661（Terminal-Bench，`observation_query`）的 Qwen 运行中，延续轮次的阶段 1 RPC 在 512 token 以内的每个 token 区间（0–128、128–256、256–512）中位数为 91–104 ms，512–1,024 区间为 106–110 ms；对不超过 4,096 token 的轮次做最小二乘拟合，每 1,000 token 增加 51–84 ms：一次 200 token 的前向中，随 token 数变化的约为 10–17 ms（取自这些作业 native 文件中的 `phase1_ns` 和 `forward_tokens`；其余部分由下面的 step profile 来计时）。
  - **改变了什么。** vLLM 的捕获尺寸被固定下来：先是它按同时运行的序列数给出的尺寸（与不开该参数时它推出的列表相同，所以 decode 步和 FULL decode graph 不变），然后是不超过 T 的 64、96、128、192、256、320、384、448、512、640、768、896、1024、1280、1536、1792 和 2048（T 为 64、128、256、512、1024、2048 之一）；超过 256 token 时，一次前向的填充少于 25 %。单独运行的阶段 1 或 Goal forward 在 token 数放得下时被分派到下一个已捕获尺寸的分段 graph，超过 T 时不使用 graph；它从不重放 FULL graph。其他每一步仍按 vLLM 的方式分派，只是尺寸更大了，所以不超过 T 个 token 的混合步也按分段 graph 运行：开 `--phase1-in-step` 时带阶段 1 的 decode 步，以及任何带 prefill 块的步。probe 的 hook 在被捕获的分段之外运行，只读取它自己的行和位置，从不读取填充部分。
  - **结果。** 填充会改变已编译分段中矩阵乘法的形状，因此阶段 1 的 KV、Gated-DeltaNet 状态和向量、现在被填充的混合步中的各行（以及由此带来的选页或回复中的近似平局）可能在最后几位上与不填充、不用 graph 的前向不同，与 4.8 节的各 graph 模式一样：请在相同设置下比较各策略。管线本身是精确的：在 CPU fake 上（填充行为 NaN），每个回复、选页、carry 和 Page 向量都与不开该参数的服务相同。
  - **固定值与检查。** `startup.json` 记录 `phase1_graphs` = `{"capture_sizes": [...], "tokens": T}`，编译配置（也在 vLLM 命令的 `--compilation-config` 中）记录 `cudagraph_capture_sizes`。前端比对 vLLM 解析出的尺寸；引擎把这条记录发给 worker 并要求其回显；worker 要求 vLLM 解析后的捕获尺寸与固定值完全一致，在它的 `cuda_graphs` 记录中写入 `probe_forward: "piecewise"` 和 `phase1_graph_tokens`，并在 `worker-attached.json` 中回显 `phase1_graphs`；`ready.json` 记录它。不做任何逐请求或逐步的检查；runner 每步多读一个属性。取 0 时，vLLM 命令、启动配置、其固定值和证据都与不带该参数的启动相同，runner 对每一步的分派也与 `ba4d1ba` 相同（identity probe；CPU 测试）。
  - **组合。** 在 Qwen profile 上，launcher 只在 `--cuda-graphs off` 和 `full-decode` 时拒绝它（没有分段 graph）。CPU 测试端到端运行了以下情形：阶段 1 原地运行、`goal_query` 的 Goal forward 在阶段 1 内、`--phase1-in-step`、`--gdn-device-handoff` 和 `--async-scheduling`：异步时，单独的前向在正在运行的 batch 排空之后运行，与不开该参数时一样，每一轮的 token、选页、Page 向量和 carry 都与不开该参数的同步服务相同。`--goal-overlap`（它的 Goal forward 也是这样的一次前向）和 thinking 选项可以同时使用，但没有与它一起做端到端测试。Gemma 4 profile 拒绝它（v1：在窗口布局上既未实现也未测试；作业 3462014 的 2,710 秒运行中，它的阶段 1 前向共用 101 秒，占 3.7 %）。
  - **待在 GPU 上检查。** vLLM 是否保留固定的尺寸，并且只为 decode 的尺寸捕获 FULL graph（它的分派器不在 overlay 中；CPU 测试用的是它的模型）；启动时间和 graph 内存（每多一个尺寸就多一次分段捕获，graph 内存池占用固定 KV cache（`--kv-cache-gib`）和 Goal rewriter 之外剩余的 GPU 内存）；按 token 区间统计的阶段 1 时间（开与不开该参数）；相同 prompt 下与不开该参数的服务比较向量和选页。
  - **Step profile。** 设置 `REMOTE_RA_STEP_PROFILE=1` 时，每次单独的阶段 1 或 Goal forward 还会有一条自己的记录，带有 vLLM 选择的 graph 模式和它的 token 区间（4.11 节）。
  - **源码与测试。** `remote_ra_live_v2/phase1_graphs.py`、`cuda_graphs.py`（`probe_piecewise`）、`hooks.py`、runner 的分派（`vllm/v1/worker/gpu_model_runner.py`）、`serve.py`；`test_phase1_graphs.py`（尺寸与 vLLM 自己的尺寸代码对照，固定值与拒绝，分派与 `ba4d1ba` runner 及 vLLM 分派器模型对照，以及 fake 上三个 session、步被填充的运行：原地、`goal_query`、`--phase1-in-step`、`--gdn-device-handoff` 和 `--async-scheduling`）。
- **`--goal-rewriter-batch N`（默认关闭；需要 `goal_query` 加 `--goal-rewriter-model` 和 `--admission-overlap K`，2 ≤ N ≤ 16, N ≤ K；Gemma 4 profile 也接受它，有自己的默认 token 上限，4.14 节）。** 开 `--admission-overlap 16` 时，各 session 的 Goal rewriter 调用排队等那一个 rewriter 进程：在不批量的 16 个 session 的 SWE-bench 作业 3404696 中，rewriter 调用排队时间的中位数为 98.2 秒，有 rewriter 调用的 Goal 构造中位数为 108.8 秒。开这个参数时，排队中的调用最多 N 个一起发给一个子进程，由它用一次批量的 greedy `generate` 运行（`remote_ra_live_v2/goal_rewriter_batch.py`、`goal_rewriter_child.py`）：
  - **不变的：** 何时调用 rewriter、发送的内容（上游的请求行，逐字节相同）、prompt、JSON schema 及其约束、`max_new_tokens`、每个回答的解析和校验（每次调用都运行上游 `PersistentCommandGoalRewriter.rewrite` 的代码）以及回退。每次调用得到自己的回答；子进程的错误或被拒绝的候选只影响那一次调用。只有一个调用的批次就是上游的 `generate`。
  - **改变的：** 模型调用。各 prompt 左填充后合成一次 `generate`，每一行由它自己请求的 lm-format-enforcer prefix function 约束。**批量 greedy 解码与一次一个请求的解码并非逐位相同：** 填充和 batch 维度改变浮点归约顺序，接近平局时可能选出不同的 token，被改写的 Goal（或其回退）可能与不开这个参数的服务不同。只在设置相同的服务之间比较策略。子进程还把 lm-format-enforcer 的 tokenizer 数据只构建一次，而不是每个请求构建一次（允许的 token 相同；子进程启动时自检比较新建和缓存的结果，不一致则保留逐请求构建）。
  - **超时和失败：** 固定的 120 秒超时从调用所在批次发送时开始计算，所以发送前排队的时间与以前一样不计入。超时、子进程退出或无法读取的回答会让该批次的每个调用都以上游的错误失败（各自回退），下一个批次会重启子进程。CUDA 显存不足时批次拆成两半运行，一直拆到单个请求；批量 `generate` 的其他失败会让其中的请求逐个经上游代码运行。每次重试都要等失败的 `generate` 的异常释放之后才开始，每次尝试之前清空 CUDA 缓存，单独运行时失败的请求不保留 traceback（早先的代码在异常处理块内运行重试，而其 traceback 仍持有失败的 `generate` 的张量：作业 3404230，[冻结说明 4.10](history/reference-gpu-notes.zh-CN.md#出自-410-节admission-优化)）。单个调用的失败就是上游的回退。
  - **token 上限（`--goal-rewriter-batch-tokens T`，默认 16,384；Gemma 4 profile 随 `--kv-cache-gib` 而定，4.14 节；0：不设上限）。** 每个调用的线程在排队之前，用固定的 checkpoint 的 tokenizer 和子进程自己的 prompt 构造数出它的 prompt token 数。一个批次按顺序取排队中的调用，只要批次大小乘以其中最长的 prompt 不超过 T，遇到第一个放不下的调用就停止；单独就超过 T 的调用（或无法计数的调用）作为只有一个调用的批次运行，与不批量时相同。没有调用插队。默认值按作业 3404230 在 27B 服务旁边留出 1 GiB 的 GPU 显存：CUDA 可分配 97,362 MiB（95.08 GiB，取自子进程自己的显存不足报告；nvidia-smi 显示 97,871 MiB），服务进程占 90,102 MiB（87.99 GiB），子进程的其余部分占 2,519 MiB（权重 1,137 MiB；CUDA context、kernel 和 workspace 1,382 MiB），剩下 4,741 MiB，扣除余量后一个批次可用 3,717 MiB。按 Qwen3-0.6B 的尺寸（28 层、8 个 128 维的 KV head、bf16），每个填充后的 prompt token 需要 112 KiB 的 KV cache 和最多 40 KiB 的单层激活，每个生成的 token 需要 112 KiB，B 个长 L 的 prompt 组成的填充批次最多需要 3 B L² 字节的 attention mask：填充后 16,384 个 token 的最宽批次（最坏为 2 × 8,192，每行 384 个新 token）需要 2,900 MiB，按两次运行显示的 1.3 倍分配器 slack 为 3,770 MiB。放得下的上限不计 slack 为 20,400，计 slack 为 16,100；16,384 介于两者之间。在本地把作业 3404230 的批次按这个上限重放，估计它保留了批量节省的子进程时间中的大部分（只是估计；其输出没有存档）。
  - **启动时固定的内容：** `startup.json` 和 `ready.json` 记录 `goal_rewriter_batch: N` 和 `goal_rewriter_batch_tokens: T`，引擎在启动时检查一次它的 rewriter 以 N 和 T 批量运行。批量服务的每个 `goals.jsonl` 交互增加 `batch_id` 和 `batch_size`（第 8 节）。不开这个参数时，启动配置、rewriter 命令和证据都不变。
  - **GPU 显存：** Qwen3-0.6B 每个 prompt token 约需 112 KiB 的 KV cache，16 个 4–6K token 的 prompt 组成的批次在 27B 服务旁边需要 7–10 GiB。先用 N = 4，并从 `goals.jsonl` 读出 prompt 大小（`input_tokens`）。每次批量 `generate` 之后子进程都清空 CUDA 缓存，使批次的峰值不会在批次之间一直被保留——被保留的缓存 cuDNN、驱动和 27B 服务都用不上（rewriter-batch-fix；作业 3404230 中缓存一直被保留）。
  - **CPU 测试（`test_goal_rewriter_batch.py`）：** 使用遵循 Transformers 批量 greedy `generate` 语义的假模型和假 enforcer。每个批量回答都等于该请求单独运行时的上游回答，16 个 session 的 Goal 与不批量时相同（N = 16 和 N = 4）。测试还覆盖显存不足和失败路径、超时、不匹配的回答、被取消的 admission 和关停，并在一个假 GPU 上（张量只要还被某个 frame 或 traceback 引用就保持已分配，缓存的块 cuDNN 用不上）复现作业 3404230 的四种故障模式（每个测试在修复前都失败）。
- **`--goal-rewriter-engine vllm`（默认关闭：`hf`；需要 `goal_query` 加 `--goal-rewriter-model` 和 `--admission-overlap K ≥ 2`，不能与 `--goal-rewriter-batch` 同用；两个模型 profile 都接受；在 README 的稳定配置命令之外是实验性的，在该命令中已经验证（作业 3576565，[已知限制 3.2](known-limits.zh-CN.md#32-vllm-goal-rewriter-只在稳定配置的命令中验证过)）；[docs/rewriter-vllm/DESIGN.zh-CN.md](rewriter-vllm/DESIGN.zh-CN.md)）。** rewriter 的调用发给一个子进程，它在一个小的 vLLM `AsyncLLM` 引擎中以 continuous batching 运行 Qwen3-0.6B，于是可以同时有多个调用在进行，而不是一次一个 Hugging Face `generate`（`remote_ra_live_v2/goal_rewriter_vllm.py`、`goal_rewriter_vllm_child.py`）：
  - **不变的：** 何时调用 rewriter、发送的内容（上游的请求行）、prompt（vendored 脚本自己的构造和 Hugging Face tokenizer 给出的 id，所以 `input_tokens` 的计数相同）、JSON schema、greedy 解码、`max_new_tokens`、解码、解析和校验（每次调用都运行上游 `PersistentCommandGoalRewriter.rewrite` 的代码）、回退、120 秒的调用超时、回答行以及 `goals.jsonl` 的每个字段。
  - **改变的：** 模型调用（vLLM 的 kernel 和它的 structured-output 后端），所以 Goal 可能与 Hugging Face 服务的不同：只在引擎相同的服务之间比较策略。回答可以以任意顺序返回；`wait_ms` 接近 0（排队发生在引擎内部，计入 `call_ms`）。超时只让该调用失败，并在子进程中中止它的请求，不重启；子进程退出会让正在进行的调用失败，下一个调用启动新的子进程。超过引擎 16,384 token 上下文减去 `max_new_tokens` 的 prompt 会让该调用失败（记录到的 prompt 约 7.5K token，最多 7,544 个）。
  - **隔离和启动：** 子进程的 `sys.path` 只有干净环境的 site-packages（没有本 overlay，否则会加载它打过补丁的 `vllm`），它的环境里没有 `REMOTE_RA_*`、`VLLM_POLICY_*` 或 `SELECTIVE_*` 变量，它的 stdout 只承载协议行（vLLM 的输出进入服务日志），并且它运行在自己的 session 中：关停时结束它的整个进程组，包括 EngineCore。它在引擎进程中、27B 引擎分配完 KV cache 之后、READY 之前启动（Hugging Face 子进程也在这里启动），受固定的 300 秒启动超时约束。
  - **选项：** `--goal-rewriter-guided-backend lm-format-enforcer`（默认，Hugging Face 子进程使用的 enforcer）或 `xgrammar`（vLLM 0.24 的 xgrammar 拒绝 schema 中的 `uniqueItems`，所以这一组发送去掉它的 schema：仅用于测量；校验仍然拒绝重复项）；`--goal-rewriter-vllm-kv-gib G`（2–16，默认 3，约 28K token）；`--goal-rewriter-batch-invariant`（只在子进程中设置 `VLLM_BATCH_INVARIANT=1`）。
  - **启动时固定的内容：** `startup.json` 和 `ready.json` 记录 `goal_rewriter_engine`（引擎、后端、KV 字节数、0.05 的 utilization 门槛、batch invariance 以及子进程的引擎参数；`ready.json` 另加子进程的 `startup_ms`），引擎在启动时检查一次它的 rewriter 以该记录在 vLLM 上运行。用 `hf` 时，启动配置、rewriter 命令和证据都不变。
  - **GPU 显存：** 权重、3 GiB KV、激活、CUDA-graph 池和一个 CUDA context，在 27B 服务旁边约需 5–6 GiB（在作业 3499550 中，子进程的峰值为 5,550 到 5,786 MiB，它在只剩 7,260 MiB 空闲时也启动了）；vLLM 的启动门槛要求 GPU 有 5 % 空闲。Gemma 4 剩下的更少：在作业 3538523 中，带 vLLM 引擎的 Gemma 服务峰值为 GPU 97,871 MiB 中的 92,500 MiB（仅验证器检查，未审计；[已知限制 3.1](known-limits.zh-CN.md#31-vllm-goal-引擎子进程没有预留的-gpu-显存)）。
  - **CPU 测试（`test_goal_rewriter_vllm.py`）：** 子进程作为真实的子进程，针对假的 `vllm` 和 `transformers` 包运行。它的 prompt id、schema、解码和解析与上游 generate 相同；16 个 session 的 Goal 与 Hugging Face 子进程的假实现相同，同时有多个调用在进行，回答乱序返回；引擎错误、在 `max_new_tokens` 处被截断的回复、超时及其中止、子进程退出和引擎死亡都只让各自的调用失败；检查子进程的 `sys.path`、环境和 stdout；两个 profile 上的参数、固定内容和拒绝条件、`ready.json` 和 Goal 控制哈希；以及在引擎和 worker 的假实现上、固定了该记录的 16 个 Gemma 4 `goal_query` session，其决策与 Hugging Face 子进程的假实现完全相同。
  - **vLLM 0.24 上的 lm-format-enforcer（rewriter-lfe-fix）。** 第一次 GPU 回放（perf-rwreplay2，作业 3488202）发现 vLLM 的 lm-format-enforcer 后端无法使用：lm-format-enforcer 0.11.3 的 vLLM 集成会导入 `vllm.transformers_utils.tokenizer`，而 vLLM 0.24 已没有这个模块（其 tokenizer 移到了 `vllm.tokenizers`），所以该后端从未启动，这些组的每次调用都失败（xgrammar 不受影响）。现在子进程在 vLLM 加载之前，用自己的模块占用这个集成的模块名（在子进程里，以及它的 EngineCore 里，无论 fork 还是 spawn）。该模块按 Hugging Face 子进程的方式构造 tokenizer 数据（lm-format-enforcer 的 Transformers 构造器，遍历 tokenizer 的 id，并使用上游脚本的 Transformers 别名），宽度按 vLLM 对模型词表的 bitmask 设定；因此 tokenizer 之外的 padding id（解码为空，lm-format-enforcer 自己的 vLLM 构造器会在每一步都允许它们）始终不被允许，与 Hugging Face 子进程一致。启动时、在 readiness 记录之前，一个 schema 只有唯一取值的约束请求必须返回该值、以 stop token 结束且只用 tokenizer 的 id；否则子进程退出，server 不会启动（只在启动时检查，不按调用检查）。`test_goal_rewriter_vllm.py` 的假包采用 vLLM 0.24 的模块布局和 lm-format-enforcer 0.11.3 的导入代码：没有该模块（或只在子进程里有、spawn 出的 EngineCore 里没有）或没有别名时子进程拒绝启动；没有启动检查时每次调用都像回放中那样失败；遍历全部模型 id 的数据、未受约束的回复和越界 id 都会被发现。
  - **与 `--goal-overlap` 一起用（integrate-1030；README 的稳定配置命令同时使用这两者）。** 两者可以组合：launcher 固定两条记录（带 `goal_overlap` 的 `admission`，以及 `goal_rewriter_engine`），不加任何 vLLM 参数，启动检查也接受二者同时出现。`test_goal_overlap_vllm.py` 让三个 `goal_query` session 在 `--admission-overlap 3`、`--goal-overlap` 和 vLLM 子进程（多个调用同时进行）下运行，与所有参数关闭、使用 Hugging Face 子进程假实现的服务对比：每一轮的 Goal、Goal 向量、选页、Page 向量和 token，以及 `goals.jsonl` 的每次交换都相同；需要 Goal forward 的轮次都把它作为单独的一次运行。
  - **GPU 上的运行。** 只在 README 的稳定配置命令中验证过（作业 3576565，[docs/status.zh-CN.md](status.zh-CN.md)）；其他所有组合都是实验性的（[已知限制 3.2](known-limits.zh-CN.md#32-vllm-goal-rewriter-只在稳定配置的命令中验证过)）。在那之前该引擎的回放和运行见[冻结说明 4.10](history/reference-gpu-notes.zh-CN.md#出自-410-节admission-优化)。
- **推荐。** Qwen 在 SWE-bench Verified 上开 thinking 时：README 的[稳定配置命令](../README.zh-CN.md#稳定配置qwen-跑-swe-bench-verified)原样使用（`goal_query` 加 `--admission-overlap 16`、vLLM Goal 引擎和 `--goal-overlap`）。其他策略用同一命令，但去掉 `--goal-overlap` 和 `--goal-rewriter-*` 选项（对没有 Goal 的策略，launcher 拒绝 `--goal-overlap` 和 vLLM Goal 引擎；`goal_query` 和 `goal_query_markers` 有 Goal），`full` 另加 `--release-idle-pages`（[已知限制 2.4](known-limits.zh-CN.md#24-25-gib-中-16-个-session-的-full-需要---release-idle-pages)）；当前配置的 64 任务运行另加 `--async-scheduling`（[docs/status.zh-CN.md](status.zh-CN.md#当前配置)）。其他情况：`goal_query` 加 `--admission-overlap 16`（16 个 session 时）和 `--goal-rewriter-batch 4`，其他 admission 参数关闭，rewriter 引擎用默认的 `hf`，与 SWE-bench 正式运行（[docs/SWEBENCH_BENCHMARK.zh-CN.md](SWEBENCH_BENCHMARK.zh-CN.md) A 部分）相同；其他所有策略的 admission 参数全部关闭。

**GPU 上的运行。** 作业结果见 [docs/status.zh-CN.md](status.zh-CN.md)，限制见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md)（3.1、3.2）；本节在 2026-11-16 之前记述的 GPU 运行见[冻结说明 4.10](history/reference-gpu-notes.zh-CN.md#出自-410-节admission-优化)。

### 4.11 解码快速路径（默认开启）

默认开启（4.15 节）：`--no-decode-fastpath` 关闭全部三个阶段，`--no-decode-fastpath2` 关闭第二、第三阶段，`--no-decode-fastpath3` 关闭第三阶段，`--no-deferred-completion-io` 关闭延后写完成文件。下文按显式给出开关的情形描述。

在作业 3366128 的 decode 检查中（[冻结说明 4.11](history/reference-gpu-notes.zh-CN.md#出自-411-节解码快速路径)），16 个请求在 SWE 类历史上同时运行、Page 为 16 token 时，不开快速路径的本 runtime 解码比 stock vLLM 慢 1.8 倍：每 token 53.8 ms，stock 为 29.2 ms（两者都是同步调度）。原因是每个解码步里的 host 端工作随"运行中的请求数 × 每个请求的 Page 数"增长。

- 每个运行中的请求每步都要发一次 `allocation_update` RPC。这个 RPC 为了数出本请求的 block，会扫描所有运行中请求的全部分配记录（`remote_ra_http/evidence.py`）。
- 同样长的请求，16 token 的 Page 数是 784 token 的 49 倍，所以 Page 越小，扫描越慢。
- 在 `full` 的 SWE-bench 运行（作业 3357655）中，这些调用共耗时 245.9 s，占整场 13.8%（321,338 次，每次 0.765 ms）。

`--decode-fastpath`（默认开启，推荐所有策略都打开，4.15 节；`--no-decode-fastpath` 关闭它）用更少的工作得到相同的值：

- 用每个请求已占 block 的计数代替扫描（`remote_ra_live_v2/decode_fastpath.py`）。
- 每步的 lease 只检查上一步之后新增的 attention lease 的类型。
- 每步只从设备读一次 position，而不是每个请求读一到两次。
- worker 每步只同步一次设备，而不是每个请求一次。
- 动作标记的停止判断只解码新 token；只有新 token 含 `>` 时才解码全部输出。这一点是精确的，因为标记是 ASCII，并以 `>` 结尾。

**开关的记录。**

- 打开时，`startup.json` 记录 `decode_fastpath: true`；worker 在其证据中回显，`ready.json` 也会记录；只在启动时检查一次。
- 不开时，启动配置、证据和 vLLM 命令都不增加任何内容，代码路径与之前相同。

**CPU 测试（`test_decode_fastpath.py`）。**

- 在随机修改下，占用计数与扫描结果相同。
- lease、position、拒绝和标记判断都完全相同，包括使用固定的 Qwen3.6 tokenizer 时。
- 在多 session 的模拟环境中（16 和 784 token 的 Page，`full` 和 `observation_query`），开和关时的 RPC 内容、证据文件、参与计算的 Page 和回复都完全相同。

**逐步计时。** 环境变量 `REMOTE_RA_STEP_PROFILE=1`（默认关闭）会在服务日志里为每个解码步写一行 `REMOTE_RA_STEP {json}`。

- 每行记录该步各部分的 host 时间，以及用 CUDA event 计时的前向时间（不增加同步）。
- `tools/step_profile.py` 负责汇总这些行。
- 它只用于诊断，不属于证据。
- admission 自己的工作不计入各步，但每次单独的阶段 1 或 Goal forward 还会作为一条自己的记录写进每行的 `probes`（phase1-graphs）：按种类（阶段 1、带 Goal forward 的阶段 1、单独的 Goal forward）、vLLM 为它选择的 CUDA graph 模式和 token 区间分组，每次记录 RPC、每次前向的 runner 过程（状态更新、输入、graph 分派、元数据、带 carry 恢复的 hook、前向、带状态捕获的行和向量）、清理与 drain、Page 向量压缩、回执、前向的 device 时间，以及下一步把这次前向移出 batch 的行重新加回的时间。开 `--phase1-in-step` 时，步内的 probe 行增加 `worker.before.probe` 和 `worker.after.probe`。`tools/step_profile.py` 按标签打印它们（`--probe-csv` 写出长表）。各步的记录需要同步的引擎循环：开 `--async-scheduling` 时 vLLM 经它的 batch 队列执行各步，install 行给出这一故障，不记录任何步，但这些 probe 记录仍会写出：每 `REMOTE_RA_STEP_PROFILE_EVERY` 次前向（默认 50）在下一次前向开始时写一行；关闭前最后几次前向不会被写出。
- 每一步还会按 vLLM 的分派为它选择的 CUDA graph 模式计入每行的 `dispatch`（phase1-graphs）：FULL、PIECEWISE 或 NONE，并带有该步调度的 token 数、填充后的尺寸，以及该步和前向的时间。输入阶段处理了 Cancel 或 CloseSession 命令的迭代标记为 `cancel` 或 `close`；`admit` 只留给真正的 admission（此前三者都标记为 `admit`）。`tools/step_profile.py` 按标签和模式打印各步（`--dispatch-csv` 写出它们），这样 admission 之后的第一步就可以和请求数相同的普通 decode 步对照。

**Torch profile。** 环境变量 `REMOTE_RA_TORCH_PROFILE=<绝对目录>`（默认关闭）让 `serve.py` 传入 vLLM 自己的 `--profiler-config {"profiler":"torch","torch_profiler_dir":<目录>}`，于是服务带有 vLLM 的 `/start_profile` 和 `/stop_profile` 路由，trace 窗口和 stock vLLM 一样通过它们开启和结束（`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` 会加上 vLLM 的命名区间）。

- startup config、其 pin 和服务的环境都不变，`--language-model-only` 仍是最后一个参数；不设置时 vLLM 命令不变。
- 不能与 `REMOTE_RA_STEP_PROFILE=1` 同时使用（一次只用一个 profiler）。
- 它只用于诊断，不属于证据。只在 CPU 上测试过（`test_torch_profile.py`）。

**第二阶段：`--decode-fastpath2`（默认开启，4.15 节；需要 `--decode-fastpath`）。** 16 个请求同时运行时，作业 3366128 的解码每个 token 仍比 stock vLLM 慢 6 到 7 ms（36.1 对 29.2 ms，两边都是同步调度）；其逐步计时显示，每步 9.2 ms 的 host 时间（所有 16 个请求同时运行的 step 的均值；W2 轮里为 10.2 ms）里约有 5.5 ms 花在本 runtime 中。`--decode-fastpath2`（默认开启；`--no-decode-fastpath2` 关闭它）再去掉其中一部分工作，得到相同的值：

- 每个解码步只发一次 `allocation_batch` worker RPC，携带所有已调度请求的分配更新，而不是每个请求一次 `allocation_update` RPC。自 worker 上次接受以来行没有变化的更新不带行发送（`remote_ra_live_v2/decode_fastpath2.py`、`engine_boundary.py`）。worker 仍逐条按原来的方式处理，所以每个 native 文件里每个请求每步仍有一个字段相同的 `allocation_update` 帧。
- 每个请求每步的 host 工作更少：native 分配参数不经 `inspect` 绑定；没有变化的行、lease 和元组直接复用而不重建；请求的活跃 KV 字节按 KV 组计算；动作标记的停止判断按 token id 记住该 token 是否含 `>`。
- 前向之后每个请求的工作（关闭 lease、Page 压缩、Gated-DeltaNet 捕获）放在 vLLM 排队 logits 计算之后运行，两者重叠。

**开关的记录。** 打开时，`startup.json` 记录 `decode_fastpath2: true`（没有 `decode_fastpath: true` 时拒绝）；引擎在启动时把它发给 worker 一次，`worker-attached.json` 和 `ready.json` 都会记录。不开时不增加任何内容，代码路径与只开 `--decode-fastpath` 时相同。

**Gemma 4**（4.14 节）接受这两个开关；在其 window 布局上，它们还加上滑窗窗口的增量观测和增量的滑窗行。

**CPU 测试（`test_decode_fastpath2.py`）。** 在多 session 的模拟环境中（784 和 16 token 的 Page，`full` 和 `observation_query`；`goal_query` 分别配合 GDN 交接和步内阶段 1），回复、计划、参与计算的 Page、计算的 position、证据文件和每个请求的帧都与只开 `--decode-fastpath` 时相同；把每个 batch 逐条展开后，与其 `allocation_update` RPC 相同。

**第三阶段：`--decode-fastpath3`（默认开启，4.15 节；需要 `--decode-fastpath2`）。** 两个快速路径都打开后，本 runtime 在解码步前向之前仍然加入的 host 工作，大部分是每个运行中请求的固定开销：worker 的步骤（观察、claim、materialize、position、步 lease）、FullAttention 行检查和每个请求的分配路径。`--decode-fastpath3`（默认开启；`--no-decode-fastpath3` 关闭它）去掉其中一部分，得到相同的值：

- 若一个请求在这一步没有追加 FullAttention 块，且 Mamba 行在同一个槽位下标处保持不变（16 token 的 Page 下每 16 步有 15 步如此），则在做完相同的列表、设备行和 Mamba 检查之后，直接沿用上一步的 Mamba lease、key 元组和（为空的）claim 差异（`remote_ra_live_v2/decode_fastpath3.py`、`allocation_owner.py`）。设备行按字节比较，对同一种整数 dtype 与 `np.array_equal` 结果相同（`allocation_owner.py`、`worker_rows.py`）。
- 这样的请求在 `materialize` 只会标记这一步时跳过它；在其中不可能发起 IO 时跳过它的 IO 归属作用域。它的步 lease 在构建时不再重复检查在接受时已检查过的 Mamba 槽位（`qwen.py`）。
- 一步的 position 列从 GPU 一次拷贝读回，而不是两次。
- worker 端，不带行发送的分配更新直接应用并计量，帧不变；引擎把每个请求都要 import 的模块只绑定一次（`engine_boundary.py`）。

**开关的记录。** 打开时，`startup.json` 记录 `decode_fastpath3: true`（没有 `decode_fastpath2: true` 时拒绝）。引擎在启动时把它发给 worker 一次，`worker-attached.json` 和 `ready.json` 都会记录。不开时不增加任何内容，代码路径与只开 `--decode-fastpath2` 时相同。

**CPU 测试（`test_decode_fastpath3.py`）。**

- 在随机输入上，字节比较与 `np.array_equal` 相同，一次拷贝的 position 读取与两次读取相同。
- 在随机的步骤序列上（Page 和 Mamba 跨界、行移动、请求结束后块被复用、各种篡改），owner 的稳定观察和 materialize 与完整的逐步路径相同：结果、拒绝和状态都相同。
- 在随机的 lease 序列上，新的步 lease 与原来的相同，拒绝也相同。
- 在多 session 的模拟环境中（784 和 16 token 的 Page，`full` 和 `observation_query`；`goal_query` 分别配合 GDN 交接和步内阶段 1；还包括 vLLM 形状的块表和 MRoPE position，稳定路径在这些情况下运行），回复、计划、参与计算的 Page、计算的 position、证据文件、每个请求的帧和 RPC 都与只开 `--decode-fastpath2` 时相同。较长的用例在 `REMOTE_RA_SLOW_TESTS=1` 时运行。
- CPU 微基准（x86 模拟环境，16 个请求、每个 20K token、16 token 的 Page，7 次运行的中位数）：一个解码步（全部是 host 工作，包括模拟环境自己的前向）只开 `--decode-fastpath2` 时为 6.15 ms，打开 `--decode-fastpath3` 后为 5.25 ms（每步均值；中位步为 5.44 和 4.70 ms）。

**Gemma 4**（4.14 节）同样接受第三个开关（gemma-fastpath3）。上面不依赖 Mamba 行的削减原样适用：不带行的分配条目、全局组设备行的字节比较、跳过的 `materialize` 和 IO 作用域、只绑定一次的每次调用 import。window 布局另有自己的削减，结果不变（`remote_ra_live_v2/window_rows.py`、`window_owner.py`、`window_lifecycle.py`）：

- 引擎原来每一步都逐块检查每个滑窗窗口中保留的块（W = 1,024、P = 16 时每个请求 5 × 65 个块），并从每个滑窗行的开头走到窗口处（O(上下文)）。现在窗口池记录它改变了引用计数或 incarnation 的块（`get_new_blocks`、`touch` 和 `free_blocks`，vLLM 0.24 只在这三处改变它们），只有当这份记录在请求的窗口上次检查之后提到其中某个保留块时，才逐块检查这些窗口。行中属于窗口的部分按切片读取。
- 行没有变化的一步（每 16 个解码步中有 14 步）在做完相同的 full-attention、窗口和 runner 行检查之后，沿用上一次观测的 lease、key 元组和（为空的）差异，它的 `materialize` 只标记这一步。
- 窗口没有移动时，步 lease 的构建不再重复检查其槽位；窗口移动后的槽位用 C 层面的遍历检查；后端不再重复检查它已在同一 generation 下检查过的 lease 元组的 generation。

**CPU 测试（`test_gemma_fastpath3.py`）。** 在 vLLM 的 KV manager 上的随机引擎步骤中（其他请求的块来来去去、重复释放、被释放后分给另一个请求的块、被 touch 的块、被替换的行条目、记录重新开始），引擎的记录路径与完整检查相同；在随机序列上，稳定的 owner、lease 和后端与各自的完整路径相同，拒绝也相同；在 window 端到端模拟环境中，`--decode-fastpath2` 与 `--decode-fastpath3` 相比（三个 session，eager 以及带 CUDA graph 记录，`REMOTE_RA_SLOW_TESTS=1` 时覆盖每个策略；一个 session；16 个 session 在池压力下开 `--release-idle-pages`；`goal_query`）：RPC、证据文件、帧、position、参与计算的 Page、计划、回复、滑窗读取、每步的 lease 和 claim 都完全相同。Gemma 几何下的 CPU 微基准（x86 模拟环境：一个全局组和五个滑窗组，W = 1,024，16 token 的 Page，7 次运行的中位数，度量同上）：16 个请求、每个 20K token 时每个解码步 6.51 → 4.13 ms（中位步 5.11 → 2.88 ms），16 个请求、每个 2K token 时 5.26 → 3.70 ms，一个请求、20K token 时 0.50 → 0.37 ms（未复核：该基准的输出从未存档）。

**把完成时的 I/O 移出解码路径：`--deferred-completion-io`（默认开启，4.15 节）。** vLLM 要等 `update_from_output` 跑完本 runtime 的完成处理，才发出请求的最后一个 token：worker 的 `complete` RPC（封存的 Page 写入 arena、Gated-DeltaNet carry、它的回执）、transcript 提交，以及 native terminal——它再次读取并检查启动配置，复制 drain 回执，编码 2 到 5 MB 的 JSON 并带 fsync 写盘。在作业 3410181 中（一次一个请求，历史 2K 到 257K token），每个请求的解码时间都比 stock vLLM 多出固定的 103 到 108 ms（`goal_query`、`observation_query`），其中约 40 ms 是 worker 的 `complete` 帧；有 16 个请求在运行时，它们全都要等这段时间。在 Gemma 4 profile 上（作业 3410750：一个 session，`full`，CUDA graph `full`，开两条解码快速路径），完成处理让 699 个解码间隔中的 10 个长达 75 到 300 ms（回复平均 64.5 个 token）：按该服务的 inter-token latency 直方图，它约占每个 token 1.7 ms（1.3 到 2.1），而该服务每个 token 比 stock vLLM 慢 3.8 ms（29.30 对 25.49 ms）。`--deferred-completion-io`（默认开启；`--no-deferred-completion-io` 关闭它）只把读取引擎状态的部分留在引擎线程上，同样的文件由一个写入线程在该步的输出之后写出（`remote_ra_live_v2/deferred_completion.py`）：

- **引擎线程。** native 记录由同一段代码构建（`remote_ra_http/evidence.py` 的 `terminal_value`）。其他属主可能仍持有的容器（drain 回执、计划、query 回执、idle-release 记录）在这一时刻用 pickle 复制；`goal_query` 的 `goals.jsonl` 行在这里编码并命名；启动配置不再重新读取（输出目录和布局在启动时已读过）。请求的 worker 帧和 prompt 列表只属于这条记录。
- **写入线程。** 该步的输出进入输出队列之后，`post_step` 把暂存的任务交给它，它按顺序、用与以前相同的调用写出：该行用 `open('ab')`，native 文件用 `open('xb')`，各一次 write、flush 和 fsync。字节就是 `encoded(value)` 的字节：JSON 由约 2,048 个值的片段拼成（对子值使用同一个编码器和同样的选项），写入线程每工作 0.5 ms 就睡眠 0.1 ms。CPython 的 JSON 编码器在一次调用中始终持有 GIL，所以一个一次编码整条记录的普通线程会让引擎线程一次被卡住几十毫秒（见下面的 CPU 基准）。
- **谁等待这些文件。** 前端只在引擎确认了它的 Cancel（状态 complete）之后才读取 native 文件。请求的文件还在写的时候，这个确认会等待，由引擎线程在下一次输入处理时发出。一个 session 的下一次 admission（及其中的重新载入或重建）、为另一次 admission 释放它的 Page，以及关闭它，都先等待它的写入。取消、超时、断开连接或同一请求的第二次 terminal 会等待全部写入，然后同步写出，因此 `goals.jsonl` 保持原来的顺序。关闭服务时先等待并停止写入线程，再排空请求（`remote_ra_http/shutdown.py`）；没有正常关闭就退出解释器时，待写的内容仍会写出。
- **失败。** 一次写入失败会让该请求的确认失败（请求失败，与引擎线程自己写时一样），作为 `COMPLETION_WRITE_FAILED` 报告（一个 `failure-*.json` 和服务日志），并停止写入线程：之后的每次完成、admission 和关闭都会抛出它，不再写任何内容。

**该开关固定的内容。** `startup.json` 记录 `deferred_completion_io: true`，`ready.json` 也记录它。它只属于引擎：没有 worker 启动记录，也没有 vLLM 参数。两个模型 profile 都接受它（Gemma 4 profile 自 gemma-deferred-io 起：见下文和 4.14 节）。不开这个开关时，启动配置、证据和 vLLM 命令都不增加任何内容，代码路径与之前相同。

**在 Gemma 4 profile 的 window 布局上。** 被移动的代码是两个 profile 共用的。window 布局为完成处理增加的工作仍留在引擎线程上原来的位置，即 native terminal 的前后：worker 的 `complete` RPC 提交窗口 carry 并释放该请求的滑窗 lease；在 vLLM 释放请求之前，引擎 pin 住下一个窗口（`WindowPins.pin_after`），并释放 phase 1 剩下的块（`WindowPhase1Coordinator.release`）；在那之后，vLLM 的释放把该请求其余的滑窗块还给 pool 1，`WindowPins.after_free` 检查被 pin 的窗口只剩 session 的 pin。记录由同样的代码构建，两个池的空闲块（全局 Page 和滑窗块）在与不开该开关时相同的位置读取。在 XML、kv-session 和 chat-text 协议上，native 文件唯一的读取者是前端，在被延后的确认之后读取；session 的关闭（chat-text 的空闲关闭或驱逐也是：同一个 CloseSession 帧）、它的下一次 admission（包括重新载入或重建）以及空闲释放都会等待它的写入，与 Qwen 相同。服务运行期间，Gemma 的其他代码路径都不读取 native 文件或 `goals.jsonl`。

**Step profile。** 设置 `REMOTE_RA_STEP_PROFILE=1` 时，`sched.complete` 被拆成 `sched.complete.drain`（worker 的 `rpc.complete`、引擎对其回执的复制 `rpc.complete.copy`、query 策略的向量发布 `sched.complete.publish`，以及引擎的检查）、`sched.complete.after`（含 transcript 提交 `sched.complete.transcript`）和 `sched.complete.terminal`（`.config`、`.copy`、`.goal`，以及带 JSON 编码 `.encode` 的 `.save`；开该开关时为 `.snapshot`）；开该开关时每一行还带写入线程的累计值（`completion_writer`：任务数、字节数、编码、写入和排队时间、被延后的确认）。不设置该变量时没有任何变化。

**CPU 测试（`test_deferred_completion.py`）。**

- 在 `test_integrate_flags.py` 的每个基础配置上（两种布局、三种协议、CUDA graph、admission 选项），配合解码快速路径、`--language-model-only`、`--release-idle-pages`，以及与 `--goal-rewriter-batch` 一起，启动器只多出该开关的键；`config.load`、`ready.json`、`compose_engine`（只在固定时挂上写入线程，worker 记录相同）和关闭顺序。
- 在随机值（其他类型的键、tuple、子类、浮点数、深层嵌套、小片段）、形如 native 文件的记录以及作业 3410181 的每个 native 文件上（108 个文件，另行检查），字节都等于 `encoded(value)`；之后对回执、计划或 query 的修改不会进入文件。
- 写入线程：与同步路径相同的调用和字节、顺序、等待、被延后的确认、写入失败、取消和重试、关闭和解释器退出。
- 端到端：三个一起解码的 mswea-xml session（`full`、`goal_query`；配合三条解码快速路径；`REMOTE_RA_SLOW_TESTS=1` 时还有 `observation_query`、`random50` 和 784 token 的 Page），以及池压力下开 `--release-idle-pages` 和三条快速路径的 16 个 session，开关打开和关闭时写出逐字节相同的证据文件，发出相同的 RPC，提交相同的回复。`test_ms2_e2e.py`、`test_kv_session_e2e.py` 和 `test_chat_text_e2e.py` 的批处理 API 服务（打包的路由、前端及其经认证的 Cancel）在开该开关时通过其场景的每项检查，用一个慢写入线程（确认被延后）时也通过；写入线程写出的每个文件都等于同一时刻同步路径的字节，chat-text 的回复（kv-session 的在慢速测试中）与不开该开关时相同。
- Gemma 4 profile（`test_gemma_deferred_completion.py`）：在 Gemma release 上（`test_gemma_flags.py` 的基础配置，配合 CUDA graph、解码快速路径、`--release-idle-pages` 和 `--admission-overlap`；以及 kv-session 和 chat-text 协议），启动器只多出该开关的键；window 布局上的 `config.load`、profile 检查、`ready.json` 和 `compose_engine`。在 window 布局的引擎和 worker 模拟上，一个和三个 mswea-xml session（eager，以及 `--cuda-graphs full` 加两条解码快速路径）、池压力下开 `--release-idle-pages` 的六个 session（`REMOTE_RA_SLOW_TESTS=1` 时 16 个；也配合 graph、两条快速路径和分块 phase 1），以及六个带释放的 MemAgent kv-session session，开关打开和关闭时写出逐字节相同的证据文件，发出相同的 RPC、帧和回复；开 `--admission-overlap 8` 和释放的八个 `goal_query` session（Goal 停放时其他 session 继续解码；`REMOTE_RA_SLOW_TESTS=1` 时 16 个，K = 16）的决定与不开该开关、K = 1 的服务相同；一个完成的文件还在写入时，取消和该 session 的下一次 admission 会等待它们，另一个 session 的 admission 不等，关闭服务在请求 drain 之前写完它们。Gemma 的 chat-text API 服务和一个基于 window 引擎的 mswea-xml API 服务在用慢写入线程（确认被延后）时通过其场景（16 个 session、释放、取消及其重新绑定、交接和关闭；一起解码的 session、批处理步之间的取消、有请求在运行时关闭服务），回复与不开该开关时相同；Gemma 的 MemAgent session 在 window 引擎上经 kv-session 路由（包括关闭）也是如此。

**CPU 微基准**（Windows 台式机，Python 3.12；以作业 3410181 的 108 个 native 文件为记录，平均 2.6 MB；一个 16 session `observation_query` 服务的启动配置）：不开该开关时，一次完成的文件工作在引擎线程上的中位数为 49.6 ms（`observation_query`、`goal_query` 和 `full` 的记录分别为 41.1、47.7 和 71.6 ms；p90 90.8 ms），其中 1.6 ms 是启动配置的读取；开该开关时为 2.1 ms（1.7、1.8 和 5.7 ms，主要是 pickle 复制；p90 7.4 ms）；随后写入线程每条记录约花 72 ms 编码、5 ms 写盘。在一个解码步循环中（每步 15 ms 不持有 GIL 的等待，即 GPU 等待，加 3 ms Python），每 6 步完成一个请求：不开该开关时，一步在中位数上晚 0.36 ms 拿回 GIL，开该开关时晚 0.52 ms（p99 为 0.75 和 2.7 ms；平均步长 18.37 和 18.70 ms）；而一个一次编码整条记录的普通线程会让它在第 90 百分位晚 23 ms（p99 52 ms，平均步长 23.2 ms）。开和不开该开关时，每个文件都逐字节相同。

**限制。** 尚未测量：只开该开关时 16 session agent 的吞吐、写入线程在 GH200 的 CPU 上对解码步的影响，以及作业所用文件系统的 fsync 时间。worker 的 `complete` RPC（作业 3410181 中 517 个新 Page 时约 40 ms）仍在路径上。

**GPU 上的运行。** 作业结果见 [docs/status.zh-CN.md](status.zh-CN.md)，限制见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md)；本节在 2026-11-16 之前记述的 GPU 运行见[冻结说明 4.11](history/reference-gpu-notes.zh-CN.md#出自-411-节解码快速路径)。

### 4.12 纯文本：`--language-model-only`（默认开启）

默认开启（4.15 节）；`--no-language-model-only` 会加载视觉塔（仅 Qwen）。下文按显式给出开关的情形描述。

Qwen3.6-27B 是多模态 checkpoint，但本 runtime 只服务文本。不加这个参数时，vLLM 0.24.0 仍会加载视觉塔，并用一张假图片对 encoder cache 做 profiling。在服务报告就绪之前，它还会用按上下文上限构造的假图片和假视频预热多模态处理器（作业 3357655 的日志：`Multi-modal warmup completed in 13.884s`）。这次预热会在不到一秒的时间里占用数十 GiB 主机内存；如果作业只申请一张 GH200 对应份额的节点主机内存，服务可能在 READY 之前就被杀掉。

`serve.py --language-model-only`（默认开启，4.15 节；在 Qwen 上 `--no-language-model-only` 关闭它）把 vLLM 自己的 `--language-model-only` 传给 vLLM，它把所有多模态上限设为 0。在固定的 vLLM 0.24.0 源码中，这意味着：

- 视觉塔被占位层替换，既不构建也不加载（`vllm/model_executor/models/interfaces.py` 中的 `_mark_tower_model`；`qwen3_5.py` 在其中构建 `visual`）。
- renderer 不创建多模态处理器，所以两次多模态预热都不运行（`vllm/renderers/base.py`），也不建立 encoder cache，不对它做 profiling。
- 语言模型的权重不变。KV cache 仍由 `--kv-cache-memory-bytes` 决定，所以它的 block 和池，以及 launcher 的 KV 和主机内存检查都不变。视觉塔的权重（BF16 约 0.9 GiB，按视觉配置计算，未实测）在 GPU 上空出来。
- 有一处计算会变。纯文本模式下，vLLM 把 16 个全注意力层的 QK-norm、RoPE 和输出门拆分合成一个融合 kernel 运行，而不是分开的几个算子（`qwen3_next.py` 中的 `Qwen3NextAttention`）。算术相同（runtime 每一步都检查三行 mRoPE 位置等于逻辑位置），但输出不一定与不加这个参数的服务逐位相同。runner 也改为把 token id 而不是它自己查好的 embedding 交给模型；这一变化是精确的（模型内部做同样的查表）。

**stock vLLM 对照也要传同一个参数**，这样两边加载同样的模型、运行同样的 kernel；只比较这一设置一致的服务。[docs/SWEBENCH_BENCHMARK.zh-CN.md](SWEBENCH_BENCHMARK.zh-CN.md) 中的 stock 命令传了它；`examples/slurm/serve_gh200.sh` 不传它，所以它的服务取默认值（开启，4.15 节）。

**参数的记录。**

- `startup.json` 记录 `language_model_only: true`，其 `effective_server_config` 中也有 `language_model_only: true`；前端把 vLLM 解析出的参数与之比较。
- engine 检查自己解析出的多模态配置。worker 检查 vLLM 把 image 和 video 的上限解析为 0，并且不持有任何视觉塔张量，然后在 `worker-attached.json` 中回显 `{"mm_limits": {"image": 0, "video": 0}, "vision_tower_tensors": 0}`；`ready.json` 也会记录。每项检查只在启动时做一次（`remote_ra_live_v2/language_model_only.py`）。
- 不加参数时，vLLM 命令、启动配置和证据都不增加任何内容；如果 vLLM 仍以 language-model-only 运行，前端、engine 和 worker 都会拒绝。

**CPU 测试（`test_language_model_only.py`）：** launcher、启动检查、worker 证据和 `ready.json`；加上参数后，多 session 模拟环境产生的 RPC 内容、证据文件和 token 都相同。

**GPU 上的运行。** 作业结果见 [docs/status.zh-CN.md](status.zh-CN.md)，限制见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md)；本节在 2026-11-16 之前记述的 GPU 运行见[冻结说明 4.12](history/reference-gpu-notes.zh-CN.md#出自-412-节纯文本)。

### 4.13 释放空闲 session 的 Page（需显式开启）

两轮之间，一个 session 在 GPU 上钉住它的常驻 Page（上一轮选中的页，加上那一轮封存的页）和 live tail 页；它的 Gated-DeltaNet carry 在主机上，每个已封存的页在锁页 Page arena 里也有一份主机副本。16 个 session、32,768 token 的预算、16 token 的 Page 时，每个空闲 session 约钉住 2,300 页，16 个合计约 37,000 页，超过 25 或 32 GiB 的 KV 池（4.9 节的容量规则）。不开这个选项时，容量检查会等运行中的请求结束；一旦所有 session 都空闲，就没有任何东西能释放 block，此后每一轮都会一直等待（早先的代码因为没有请求在运行而接纳其中第一轮，它随后因缺 block 失败）。

`--release-idle-pages`（默认关闭，仅限解耦布局）让容量检查改为释放 Page（`remote_ra_live_v2/idle_release.py`）：

- **释放对象。** 一轮放不下时，容量检查释放若干空闲 session 的全部常驻 Page 来补足 attention 池的缺口，最久未用的先释放。空闲指已打开、没有已接纳或运行中的请求，且不是正在接纳的那个 session。下一个请求已到达服务端但仍在等待接纳的 session 也算空闲：它被释放的页会在该请求的阶段 1 之前载回（否则，当各 session 的下一个请求都在等待时，被它们钉满的池永远腾不出 block）。live tail 页从不释放。如果所有空闲 session 加起来也补不足缺口，就不释放任何页，这一轮照旧以 `CAPACITY_WAIT` 等待。因为没有请求在运行而被容量检查直接接纳的一轮，会先释放它需要的 Page；开 `--admission-overlap K` 时，在同时也没有 admission 进行时同样如此（早先的代码：ledger gate 直接接纳这样的一轮而不释放）。
- **重新载入。** 被释放的页在该 session 下一轮的阶段 1 之前从主机副本恢复，阶段 1 会用到它们。这一轮的运行与这些页从未离开时完全相同；容量检查在两个峰值里都计入重新载入的页。
- **接纳过程中不等待。** 释放和重新载入都是两步之间同步执行的引擎操作；拒绝仍是立即返回的 `CAPACITY_WAIT`，所以前端 300 s 的确认等待只覆盖已接纳的这一轮本身的工作。
- **开关的记录。** `startup.json` 记录 `release_idle_pages: true`。引擎在启动记录里把它发给 worker 一次；worker 检查它，之后才接受释放和重新载入操作；`worker-attached.json` 和 `ready.json` 会记录它。不开时不增加任何内容，代码路径与之前相同。
- **证据。** 输出目录里的 `capacity.jsonl` 每次释放和每次重新载入各一行（`schema` 为 `remote-ra-capacity-event-v1`：session、它的页和 block、两个池操作前后的空闲 block、正在接纳的请求及其缺口）。每个 native 文件和 `runtime.jsonl` 行增加 `idle_release`（为接纳该请求而释放的 session、它重新载入的页），`engine-cleanup.json` 增加总计。

**CPU 测试（`test_idle_release.py`）。** 在批量引擎和 worker 的假实现上，16 token 的 Page，16 个 session，`observation_query` 加 `--active-token-budget 32768`，每个 session 12 轮，每轮 observation 约 3,600 token（prompt 最长 39,845 token，最后几轮选 2,048 页）：

| Attention 池 | 不开选项 | 开选项 |
| --- | --- | --- |
| 25,600 block（16 个 session 空闲时钉住的页超过它） | 有一轮一直等待（没有请求在运行时的 `CAPACITY_WAIT`；早先的代码接纳它，随后失败） | 192 轮全部完成，期间释放并重新载入空闲 session 的页 |
| 48,000 block | 192 轮全部完成 | 192 轮全部完成，不释放任何页 |

开选项时，每个 session 的 prompt、选页、实际读取的页和回复都与 48,000 block 的运行相同。较短的用例（900 对 9,000 block）在常规测试中以 `observation_query` 运行，在慢速测试（2.3 节）中再加上 `full`；32,768 token 的用例约需 5 分钟，在慢速测试中运行。尚未测量：GPU 上重新载入的主机到设备时间，以及真实负载下一个 session 在下一轮之前被再次释放的频率（轮流发言时，最久未用的正是下一个要运行的 session）。

**GPU 上的运行。** 作业结果见 [docs/status.zh-CN.md](status.zh-CN.md)，限制见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md)（2.4）；本节在 2026-11-16 之前记述的 GPU 运行见[冻结说明 4.13](history/reference-gpu-notes.zh-CN.md#出自-413-节释放空闲-session-的-page)。

<a id="414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证"></a>

### 4.14 Gemma 4 31B IT（第一阶段）

runtime 包含第二个模型 profile 的第一阶段：[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)，revision `842da3794eaa0b77d5f08bae87a17459d91ff475`。它可以安装；其 engine 和 worker（包括滑窗层的生命周期）已在 CPU 上做过端到端测试，启动检查另有单独测试；`serve.py --start` 可以启动它。依赖已加载模型的启动检查（attention 后端的 cache 形状、final norm、视觉塔的参数名）在不符时直接拒绝启动。加入该 profile 没有改变 Qwen release：其 vLLM 命令、启动配置、pin 和证据与加入之前的版本逐字节相同（identity probe 共 555 + 219 项，0 处差异；未复核：这些 probe 的输出从未存档）。该 profile 也以 Gemma 的轮次格式服务 MemoryAgentBench 的 kv-session 协议（见下文“MemoryAgentBench”和 6.6 节）。它的 GPU 运行见 [docs/status.zh-CN.md](status.zh-CN.md)，以及本节在 2026-11-16 之前的记述：[冻结说明 4.14](history/reference-gpu-notes.zh-CN.md#出自-414-节gemma-4-31b-it)。

**Benchmark 设置**（见各 benchmark 文档的 Gemma 小节；每次运行一张 GH200，`--mem=113560M`）：`observation_query`、`random_fill` 和 `goal_query` 用 `--kv-cache-gib 25`（组里 Gemma baseline 的 KV cache），`full` 用 32；SWE-bench：16 个 session、131,072 token、`--dram-arena-gib 72`（[docs/SWEBENCH_BENCHMARK.zh-CN.md](SWEBENCH_BENCHMARK.zh-CN.md)，A.14）；Terminal-Bench：`--max-running-sessions 20`、262,144 token、`--max-num-batched-tokens 4096`、`--dram-arena-gib 72`、Harbor `-n 16`，不含 `full`（[docs/TERMINALBENCH_BENCHMARK.zh-CN.md](TERMINALBENCH_BENCHMARK.zh-CN.md)，A.14：262,144 token 的上下文和 25 GiB 的池是组里 2026-10-26 定的设置，维护者已于 2026-11-08 确认，其余为维护者当天定下的设置；它的 GPU dry run 见[已知限制 5.2](known-limits.zh-CN.md#52-gemma-4-在-262144-token-下跑-terminal-benchgoal_query-失败)）；MemoryAgentBench：8 个 session、262,144 token、`--dram-arena-gib 60`（[docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](MEMORYAGENTBENCH_BENCHMARK.zh-CN.md)，A.15）。每个服务都开 CUDA graph `full`、三条解码快速路径、`--deferred-completion-io`、`--language-model-only`、`--release-idle-pages` 和 `--connector-pool-gib 1`；`goal_query` 再加 `--admission-overlap 16`（8 个 session 时为 8）、`--goal-rewriter-batch 4`，以及按 harness 估算至少留出 2 GiB GPU 显存的显式 `--goal-rewriter-batch-tokens`：SWE-bench 15,360，Terminal-Bench 20,480（8 个 session 的 MemoryAgentBench 用 25 GiB 时的默认值 25,600）。

**模型与选页。**
- 60 个 decoder 层：50 个滑窗层（窗口 1,024 token，16 个 KV head × 256）和 10 个全局 full-attention 层（5、11、…、59；4 个 KV head × 512，K 和 V 都缓存）。
- Page、预算、mandatory 和 recent Page、向量和恢复都只作用于全局层：每个 token 81,920 字节，16 token 的 Page 为 1.25 MiB（Qwen 为 1 MiB）。
- 滑窗层始终精确关注最近的 1,023 个 token 加当前 token，与 stock vLLM 相同。窗口不计入 `--active-token-budget`。
- 选页 query 是第 59 层（最后一层，为全局层）final norm 之前的隐状态，用同一个带种子的矩阵投影：hidden size 5,376 的 SHA-256 为 `852b42e4…a1dfb7`。

**跨轮次的窗口。**
- 每个 session 在两轮之间把滑窗窗口保留在 GPU 上：一轮结束时，engine 为每个滑窗组 pin 住下一个 query 要读的块（最多 ⌈1,024 / P⌉ + 1 个）。
- 下一轮的 phase 1 在该窗口加新分配的块上计算新 token。phase 2 从 phase 1 结束时的窗口开始，前面是 null 块，因此 vLLM 不会为窗口之前的位置分配块；更早的块此时释放。
- worker 检查这两种行，并把计算终点之后的窗口作为该 session 的 carry；没有 Gated-DeltaNet 状态。重建或关闭的 session 会释放自己的窗口。

**安装。** `--model-profile gemma4-31b-it` 用 `INSTALL_RELEASE.json` 中该 profile 的 pin（`model_profiles`：`chat_template.jinja`、`config.json`、`generation_config.json`、`tokenizer.json`、`tokenizer_config.json`）检查模型目录，并在 release manifest 中记录 profile。这些文件从模型目录读取，不会被复制。

```sh
"$VENV/bin/python" -B install.py --site-packages "$VENV/lib/python3.12/site-packages" \
  --model-dir "$MODELS/gemma-4-31b-it" --release-dir "$RELEASE-gemma4" --model-profile gemma4-31b-it
"$VENV/bin/python" -B install.py --site-packages "$VENV/lib/python3.12/site-packages" \
  --model-dir "$MODELS/gemma-4-31b-it" --release-dir "$RELEASE-gemma4" --model-profile gemma4-31b-it --apply
```

**Dry run。**

```sh
python -B serve.py --release-dir "$RELEASE-gemma4" --task /abs/path/to/tasks-g0.json \
  --policy observation_query --active-token-budget 8192 --page-tokens 16 --kv-cache-gib 28 \
  --max-model-len 131072 --max-tokens 4096 --language-model-only --output "$RUNS/gemma-obs" --port 8000
```

Gemma profile 固定的内容：

| 设置 | 值 |
| --- | --- |
| attention 后端 | `TRITON_ATTN`（FlashAttention 3 不支持 512 维的 head；FA4 是 vLLM 对该模型的默认后端，但未与本 runtime 一起验证过）。stock vLLM 对照也要用同一后端。 |
| 纯文本 | 必须加 `--language-model-only`（不加载视觉塔） |
| KV 布局 | 两个块大小为 P token 的池：全局层（P = 16；`--page-tokens 32` 为后备）和滑窗层，启动时静态预留：每个滑窗组每个 session ⌈1,024 / P⌉ + 1 个块，外加一次前向的瞬态块；`goal_query` 另加 Goal 前向的瞬态块，每组 ⌈2,048 / P⌉ 个（`--additional-config {"remote_ra_kv_layout": {"page_tokens": 16, "state": "sliding", ...}}`，只有 `goal_query` 带 `"goal_tokens": 2048`） |
| 一次 native 前向 | 8,192 token（`--max-num-batched-tokens 8192`）；更长的 phase 1 分成每次最多 4,096 token 的前向（见下）。可选：4,096 token，即组里的设置（`serve.py --max-num-batched-tokens 4096`；见下文“4,096 token 的前向”） |
| 上下文 | 65,536、131,072 或 262,144 token；262,144 在选择性规则下（见下）或 32 GiB 下单个 session 时放得下：严格规则和释放规则要求全局池放下一个上下文上限处的 session |
| `--kv-cache-gib` | 24、25、28 或 32（25：组里 Gemma baseline 的 KV cache 大小；Qwen release 接受 8、16、24、25 和 32，25 自 integrate-1026 起） |
| 协议 | 使用 Gemma 轮次格式的 Mini-SWE-Agent XML，`mswea_xml_gemma4_v1`（见下文），使用 Gemma 格式的 kv-session 协议，`--protocol kv-session --kv-session-thinking off`（见下文“MemoryAgentBench”），或使用 Gemma 轮次格式的 chat-text 协议（`--protocol chat-text`，见下文） |
| parser | `gemma4` reasoning 和 tool-call parser，`--chat-template-content-format openai` |
| 主机内存 | 一张 GH200 占节点的份额（`--mem=113560M`）；两份配额的作业（`--mem=227120M`）用 `--host-memory-shares 2`（见下） |
| 异步调度 | 与 Qwen 一样可手动开启 `--async-scheduling`（async-gemma；GPU 上部分通过，见[已知限制 5.3](known-limits.zh-CN.md#53-gemma-4-开---async-scheduling-是部分通过)）：需 `--decode-fastpath2`，已停止请求在下一步中的行在全局组和每个滑动窗口组里作废；不是速度默认选项（[docs/async-sched/GEMMA.zh-CN.md](async-sched/GEMMA.zh-CN.md)，4.15 节） |
| 拒绝 | `--gdn-device-handoff` 和 `--phase1-in-step`；`--thinking-history strip` 与 `--async-scheduling` 同用，或没有 hold slot；不带 `--thinking-history strip` 的 `--window-hold-slots` |
| 开启 thinking | 未经 GPU 验证（gemma-thinking，[docs/thinking-gemma/DESIGN.zh-CN.md](thinking-gemma/DESIGN.zh-CN.md)）：`mswea-xml` 和 `chat-text` 上的 `--enable-thinking`、`--kv-session-thinking on`，以及 `chat-text` 上的 `--thinking-history strip`：它在 GPU 上用 `--window-hold-slots H` 个 hold slot 保留切点处的滑动窗口（1 到同时运行的 session 数；默认取放得下的最大值，此时池 0 只剩选择性规则的最小值；每个为一个窗口，16-token Page 下 812.5 MiB；一个 strip 轮次从准入到提交都占一个 hold slot，phase 1、停在 Goal 处的 goal_query 准入和冷 prefill 都算在内，因此同时在途的 strip 轮次最多 H 个，其余等待；固定为 `kv_layout.hold_slots`；较小的 H 换来什么见设计说明 4.3 节）。GPU 上只检查了它的 token 对齐（GG1，[docs/status.zh-CN.md](status.zh-CN.md)） |

协议即 6.1 节的协议，改用 Gemma 的轮次格式：
- system 和 task 以 OpenAI text parts 经固定模板渲染，与 vLLM 交给该模板的形式相同（比纯字符串多一个空格 token）。
- user 轮为 `<|turn>user\n{content.strip()}<turn|>\n`；生成头为 `<|turn>model\n<|channel>thought\n<channel|>`，即模板在关闭 thinking 时的生成提示。
- 历史中每个回复前都保留空的 thought channel（只追加；Gemma 自己的模板在重新渲染对话时会去掉它，每轮 4 个 token），重建时也在它之后重新编码过去的回复。
- 停止 id 为 0、1、50 和 106（pad 加上 `generation_config.json` 的结束 id，其中包括 model 轮的结束）。

**Chat-text 协议（Terminal-Bench 2.0 与 Terminus 2）。** `--protocol chat-text`（6.7 节）可在 Gemma release 上以 Gemma 的轮次格式运行（`remote_ra_live_v2/chat_text_protocol.py` 中的 `GEMMA4_CHAT`）。启动配置固定 `protocol` 为 `chat_text_v1`，并带 `model_profile`；其 `chat_text` 记录写入 Gemma 的生成头和模板参数（`header_text` 为 `<|turn>model\n<|channel>thought\n<channel|>`，`template_kwargs` 为 `{"enable_thinking": false}`），服务在启动时用固定模板检查一次下面这些片段。Qwen 的记录和轮次格式不变。
- 新的或重建的 session 把初始段（如有 system 消息则包括它，再加第一条 user 消息）以 OpenAI text parts、关闭 thinking 交给固定模板渲染：以 `<bos>` 开头，system 消息多一个空格，与 stock vLLM 的渲染相同。之后的每条消息都按续接时的 token 流写入：user 消息为 `<|turn>user\n{content.strip()}<turn|>\n`，连续的 assistant 消息合为一个 model 轮（模板也会合并），放在空的 thought channel 之后：`<|turn>model\n<|channel>thought\n<channel|>` + 每条内容经模板 `strip_thinking` 后的文本 + `<turn|>\n`。最后接生成头。
- 续接的请求追加回复生成时的 id、它的边界（停止 id 之后为 `\n`；达到上限被截断的回复之后为 `<turn|>\n`）和新的 user 消息，再接生成头。停止 id 为 0、1、50 和 106；遇到停止 id 时 `finish_reason` 为 `stop`，达到上限时为 `length`；回复解码时跳过特殊 token。
- 每个 prompt 等于模板对其消息的渲染、并在每个过去的 `<|turn>model\n` 之后保留空的 thought channel；去掉这些 channel（`<|channel>thought\n<channel|>`，每条过去的回复 4 个 token）后，只要每条回复重新编码仍是它自己的 id，就逐 token 等于 stock vLLM 的 prompt。用固定 tokenizer 时，chat-text transcript 也逐个复现了 stock vLLM 在第 0 阶段 session（作业 3364469）中服务的 10 个 prompt。因此使用 vLLM 默认模板参数的 stock Gemma 对照（Gemma 的模板默认关闭 thinking）重新渲染每条过去的回复时少 4 个 token，与 Qwen 的 stock 对照相同（6.7 节）。
- session、换绑、aside、关闭、上下文上限的拒绝和回复上限都与 6.7 节相同；`GET /v1/chat-text/info` 增加 `model_profile`。最多同时打开 24 个 session（K，即 pool 1 的窗口槽位数）；`goal_query` 使用 6.7 节暂定的 Terminus 2 Goal 输入（见下）。
- Terminal-Bench 设置下的容量（[docs/TERMINALBENCH_BENCHMARK.zh-CN.md](TERMINALBENCH_BENCHMARK.zh-CN.md)，A.14：K = 20 对应 Harbor 的 16 个 trial，262,144 token，4,096 token 的前向和回复，25 GiB，P = 16）：pool 1 容纳 20 个窗口和一次前向的瞬态块，共 7,781 个块 id（19.00 GiB），pool 0 有 4,918 个全局 Page，除去 vLLM 的空 Page 后可用 4,917 个；开 `--release-idle-pages`、8,192 token 预算时，`observation_query` 和 `random_fill` 需要 2,119 个（选择性规则；余量 2,798 个 Page，最多 24 个 session），vLLM 自己的启动估算要 21.58 GiB。用 8,192 token 的前向时 K = 20 放不进 25 GiB（最多 19 个 session）。`full` 需要 pool 0 容纳一个停在上下文上限的 session（释放规则）：32 GiB、K = 20 时需要 17,023 个 Page，只有 10,652 个（8,192 token 的前向时需要 17,253 个，只有 8,092 个），所以它不属于 Gemma 的配置。主机内存：`--dram-arena-gib 72` 可容纳 58,982 个 1.25 MiB 的 Page（943,712 token），加上 1 GiB connector 池共锁页 73.22 GiB；不设 arena 时 20 个 262,144 token 的 session 可能锁页 401.2 GiB，启动器会拒绝。下面的 dry run 示例（K = 20，65,536 token，8,192 token 的前向）：pool 1 容纳 20 个窗口和一次前向的瞬态块，共 9,061 个块 id（22.12 GiB）。`--kv-cache-gib 28` 剩 4,815 个全局 Page：开 `--release-idle-pages`、8,192 token 预算时，`observation_query` 和 `random_fill` 需要 2,630 个（选择性规则）；`full` 需要 4,933 个（释放规则），因此 28 GiB 最多容纳 19 个 `full` session，32 GiB 容纳 24 个。不开 `--release-idle-pages` 时，严格规则在 K = 20 下需要 4,914 个 Page，空闲 session 的 Page 也会占满 pool 0，所以要开这个选项。按上文的估计，K = 20 时 32 GiB 留给 GH200 的余量不足 2 GiB（峰值约 96,100 MiB，按 Qwen 运行时的开销为 97,400 MiB）；28 GiB 的峰值约 92,000 MiB（93,300）。主机内存：`--dram-arena-gib 64` 可容纳 52,428 个 1.25 MiB 的 Page（838,848 token；Qwen 的 64 GiB 容纳 1,048,576 token），加上 1 GiB connector 池共锁页 65.0 GiB；不设 arena 时 20 个 65,536 token 的 session 可能锁页 101.1 GiB，启动器会拒绝。
- `goal_query`（gemma-tb-goal）：Goal 输入就是 6.7 节暂定的 Terminus 2 规则，没有改动。它们从消息文本中截取（任务指令、最新的终端输出、交接后的摘要、各子调用的规则），与 Gemma 的轮次格式无关；启动配置固定与 Qwen 服务相同的 `chat_text_goal` 记录（`serve.py`、`remote_ra_live_v2/chat_text_protocol.py`）。Goal 的构造与 XML 协议相同（第 5 节，以及下文的 `goal_query`）：它的前向是 `<bos>` 加 `<retrieval_query>` 文本的 token，放在窗口布局的 Goal 瞬态块中（2,048 token，pool 1 的 640 个块 id，1,600 MiB）；更长的 Goal 会让该轮失败（`PHASE1_TOO_LONG`），与 XML 协议相同。用固定 tokenizer 时，一个 Terminus 2 session 的确定性 Goal 占 117 到 171 个 token。按上面的 Terminal-Bench 设置（25 GiB，4,096 token 的前向），加上 `--active-token-budget 8192`、Goal rewriter 和 `--admission-overlap 16`，pool 1 有 8,421 个块 id（20.56 GiB），pool 0 有 3,638 个全局 Page（可用 3,637 个），选择性规则需要 2,106 个（余量 1,531 个 Page；最多 22 个 session），vLLM 自己的启动估算要 23.13 GiB。按示例的设置（8,192 token 的前向，65,536 token），pool 1 有 9,701 个块 id（23.68 GiB），28 GiB 剩 3,535 个全局 Page，选择性规则需要 2,617 个（余量 917 个 Page；28 GiB 最多容纳 21 个 session）；131,072 token 下 16 个 session 的数字与 XML 协议相同（6,135 对 2,639）。vLLM 自己的启动估算因此在 K = 20 时要 26.88 GiB（16 个 session 时 23.73 GiB），而按上下文上限算一个请求则要 28.68 GiB（30.51 GiB）。`--phase1-in-step` 和 `--gdn-device-handoff` 仍被拒绝；`--goal-rewriter-batch`（gemma-rewriter-batch，见下）、`--decode-fastpath3`（gemma-fastpath3）和 `--deferred-completion-io`（gemma-deferred-io）可以与它一起使用：在 25 GiB、K = 17 下，开 `--goal-rewriter-batch 4`（默认 T = 25,600）、`--release-idle-pages`、任一条解码快速路径和 `--deferred-completion-io` 时，每个客户端 session 都与不带这些开关的服务相同，写入线程写的每个完成文件都与同步路径的字节相同（integrate-1021，`test_gemma_chat_text_goal.py`）。Qwen3-0.6B rewriter（约 2 GiB）与模型共用 GPU。Terminal-Bench 的服务在 25 GiB 下（262,144 token 时 K = 20）显式给出 `--goal-rewriter-batch-tokens 20480`，按 harness 的估算至少留出 2 GiB GPU 显存（默认的 25,600 只会留出约 0.7 GiB；[docs/TERMINALBENCH_BENCHMARK.zh-CN.md](TERMINALBENCH_BENCHMARK.zh-CN.md)，A.14）。

一个 chat-text 的 dry run（即上面的示例：K = 20，65,536 token，28 GiB；Terminal-Bench 的服务见 [docs/TERMINALBENCH_BENCHMARK.zh-CN.md](TERMINALBENCH_BENCHMARK.zh-CN.md)，A.14）：

```sh
python -B serve.py --release-dir "$RELEASE-gemma4" --protocol chat-text --policy observation_query \
  --active-token-budget 8192 --release-idle-pages --page-tokens 16 --max-running-sessions 20 \
  --kv-cache-gib 28 --dram-arena-gib 64 --connector-pool-gib 1 --max-model-len 65536 --max-tokens 4096 \
  --cuda-graphs full --decode-fastpath --language-model-only --output "$RUNS/tb-gemma-obs" --port 8000
```

**容量。**
- GPU 池以滑窗窗口为主：P = 16 时一个 session 的窗口占 325 个 2.5 MiB 的块 id（812.5 MiB），一次 8,192 token 的前向每个服务另加一次 2,560 个块 id（6.25 GiB）。
- 若全局池放不下一个 session 在上下文上限处最坏的一轮，启动器会拒绝该 KV 大小。131,072 token、4096 token 回复时，24、28 和 32 GiB 分别最多放下 8、13 和 18 个 session。
- 16 个 session（组里 Vanilla 运行的规模）需要 `--release-idle-pages`（4.13 节），Gemma profile 接受该选项：GPU 紧张时 gate 释放空闲 session 的全局 Page，并在该 session 下一次 phase 1 之前重新载入；滑窗窗口始终留在 GPU 上。配合 query 策略（`goal_query`、`observation_query` 或 `random_fill`）以及固定的 `--active-token-budget`，启动器改用选择性规则确定全局池大小：一个 session 最坏的选择性一轮（它的常驻 Page、一份选择量的恢复、一次前向和一次回复：8,192 token 预算、4,096 token 回复时为 2 × (512 + 769) + 1 = 2,563 个 Page），加上其他 session 的 tail 和 1 % 水位线。131,072 token 下 16 个 session 需要 22.13 GiB（窗口和单次前向的临时块 18.95 GiB，Page 3.18 GiB），因此 `--kv-cache-gib 24` 即可容纳，低于 stock vLLM 服务给 KV cache 的 26.41 GiB；24、28 和 32 GiB 分别最多容纳 18、23 和 28 个 session（`goal_query` 为 16、21 和 26，见下）。不加该选项时仍按上一条规则。
- 加 `--release-idle-pages` 而没有固定预算时（`full`、`random50` 或 `--active-ratio`），启动器按释放规则确定全局池大小：一个 session 在上下文上限处最坏的一轮、其他 session 的 tail 和 1 % 水位线。GPU 紧张时空闲 session 释放全部 Page，并在下一次 phase 1 之前重新载入；对 `full` 来说就是该 session 的全部历史。131,072 token、4096 token 回复时，16 个 `full` session 需要 30.02 GiB（精确为 30,738 MiB；9,083 个 Page，32 GiB 有 10,692 个）；24 和 28 GiB 分别最多容纳 8 和 13 个（65,536 token 时 28 GiB 可容纳 19 个）。`full` 与 Qwen 一样用 `--kv-cache-gib 32`（`serve.py` 的 `GEMMA4_FULL_KV_CACHE_GIB`）：比规则多 1,608 个 Page（1.96 GiB），更小的大小被拒绝时提示 32。`full` 从不启动 Goal rewriter：它没有选择契约，只有 `goal_query` 会启动。GPU 显存（估计，未实测）：57.9 GiB 权重加 32 GiB KV 为 92,068 MiB，GH200 共 97,871 MiB。按 stock Gemma 服务的开销（26.41 GiB KV 时 READY 87,601 MiB、峰值 90,099 MiB：READY 时权重和 KV 之外 1,257 MiB，峰值再多 2,498 MiB）加上运行时自身约 300 MiB（phase-1 行、重新载入的中转），READY 约 93,300 MiB，峰值约 96,100 MiB，距总量 1.7 GiB；按 Qwen 运行时在 32 GiB 时的开销（2,215 和 3,092 MiB），峰值为 97,375 MiB，只剩 0.5 GiB。16 个 session 的检查 g1b 实测了它（作业 3429672，32 GiB 的 `full`）：READY 时 93,369 MiB，峰值 97,125 MiB，剩余 746 MiB。benchmark 设置仍用 32 GiB；原计划在峰值剩余不到约 2 GiB 时退回的 30.5 GiB（比规则多 391 个 Page）需要改启动器（它只接受 24、25、28、32）。重新载入的数据量按 Qwen swe16 的 `full` 运行（job 3357655，32 个任务，同时打开 16 个，历史按 Gemma token 换算，每轮视为瞬时、按最久未用释放）在本地回放估计（回放的输出没有存档）：32 GiB 下 18 % 的轮次需要重新载入，平均 1.5 GiB（中位数 1.45 GiB，p90 2.65 GiB，最多 4.1 GiB 主机到设备拷贝），平均到所有轮次为每轮 278 MiB；28 GiB 下为 36 % 和 546 MiB，但只有上下文低于 104,880 token 时才容纳得下 16 个 `full` session。
- vLLM 自己的启动检查要求全局池在滑窗池之外再放下一个 `--max-model-len` 长的请求：131,072 token 下 16 个窗口需要 28.95 GiB（`goal_query` 为 30.51 GiB），262,144 token 下需要 38.95 GiB。因此，尽管启动器的选择性规则接受，它也会在启动时拒绝 32 GiB 以下、131,072 token 的 16 个 session，并在任何大小下拒绝 262,144 token 的 16 个 session。所以在选择性规则下，窗口布局固定该规则的数值 `--additional-config` 中的 `"request_pages": R`（`kv_layout.WindowLayout.request_pages`、`window_capacity.pinned_request_pages`，由 `remote_ra_http/config.py` 检查），vLLM 的检查改为要求全局池为一个请求放下 R 个全局 Page（`vllm/v1/core/kv_cache_utils.py`）。两个池、gate 和各种拒绝都不变；严格规则和释放规则仍用 vLLM 自己的检查。
- 在选择性规则下，第一轮或重建需要整行。若即使释放所有空闲 session、等所有运行中的请求结束也放不下，该轮会以 `CAPACITY_EXCEEDED` 被拒绝（它不是 `CAPACITY_WAIT`，客户端不会重试），而不是无限等待。
- 释放的频率取决于池的大小：8,192 token 预算下，16 个空闲的 SWE-bench session 约占 16 × (512 + 上一轮的 Page)，超过 24 GiB 的 4,138 个 Page，所以在 24 GiB 下 admission 会释放并重新载入 Page（每次重新载入几百 MiB 的主机到设备拷贝）；28 GiB 有 7,415 个 Page，32 GiB 有 10,692 个。
- engine 的 admission gate 检查两个池。全局 Page 沿用 Qwen 的规则；滑窗池中，一轮在 phase 1 需要它的新窗口块，在 phase 2 需要把接管的窗口补到其峰值的块，而每个运行中的请求仍可能增长到它自己的峰值。重建的滑窗需求扣除它的渲染所释放的窗口，与全局需求扣除它的 Page 和 tail 一样（早先的代码：窗口槽全部占满时，重建看起来缺 pool 1，缺的正是它自己的窗口，于是不为它的全局缺口释放任何空闲 session；在没有请求运行时被放行后，它会永远等待全局 Page）。与 Qwen 相同（4.4 节），没有请求运行（也没有admission 在途）时，只要另一个已打开的 session 钉住了全局 Page、tail 或窗口，就检查这一轮；在 `--release-idle-pages` 能做的释放之后仍放不下时，它等待（`CAPACITY_WAIT`）。integrate-1018 之前，窗口布局不加检查地放行这样的一轮，它的解码可能找不到 block（vLLM 会抢占，这在这里是致命的）或永远等待。没有其他 session 钉住任何东西时，照旧不加检查地放行。
- 主机内存：启动器按一张 GH200 占节点的份额（`--mem=113560M`）、进程占 30 GiB 计算，因此 connector 池加 Page arena 不得超过 80.9 GiB。OffloadingConnector 的 CPU 池在本 runtime 中从不使用，所以 Gemma 服务接受 `--connector-pool-gib N`（4.9 节），不加该选项时用 1 GiB；该池总是被固定（`cpu_bytes_to_use`，`startup.json` 中的 `connector_pool_bytes`）并计入预算。131,072 token 时，不加 `--dram-arena-gib` 最多 7 个 session；arena 最大为 79 GiB，12 个 session 时平均每个约 86K token。16 个 session 最多使用 `--dram-arena-gib 79`（见下面的 dry run）；SWE-bench 的服务用 72（见上面的 benchmark 设置）。按 Qwen swe16 运行（SWE-bench Verified 前 32 个实例，同时打开 16 个，五个策略）回放，并用 Gemma 的 tokenizer 重新计算每段历史（比 Qwen 多 3.7–16.4 % 的 token），16 个打开的 session 的 Page 峰值为 26.5–37.75 GiB，最长的 16 段历史合计最多 65.25 GiB；单段最长历史为 96,336 token（未复核：该回放的输出从未存档）。一份配额（`--mem=113560M`）对 SWE-bench 足够：Gemma 的 SWE-bench 运行（作业 3429808、3429830 和 3462014）都只用一份；不需要两份。`--host-memory-shares 2`（仅 Gemma profile；Qwen 保持 128 GiB 的预算）让启动器按两份配额的作业（`--mem=227120M`）计算：可锁页 191.8 GiB，262,144 token 时 arena 最大 190 GiB；MemoryAgentBench 最长的 16 个 case 需要它（见下）。

**分块的 phase 1。** 超过一次 native 前向（8,192 token；Mini-SWE-Agent 的 10,000 字符上限下，数字密集的输出约为 10.2K token）的 phase 1 在同一个 lease 上分成每次最多 4,096 token 的前向运行（`model_profile.GEMMA4_PHASE1_CHUNK_TOKENS`）；不超过 8,192 token 的 phase 1 仍是一次前向。
- 全局行从第一次前向起就包含整个 run，只有第一次前向把新的 Page 清零。每次前向的滑窗行包含它读取的窗口和它的新 token；token 已离开窗口的新滑窗块会被后面的 token 复用，因此 phase 1 每个滑窗组最多占 ⌈(1,023 + 4,096) / 16⌉ + 1 = 321 个新块，在一次前向的瞬态（P = 16 时 512 个）之内：pool 1 不变。lease 中列出后面的前向（`chunks`），worker 检查它们（铺满整个 run、延续窗口、从不复用借来的窗口或仍会被读取的块）并依次运行：第一次前向之前检查借来的窗口，最后一次之后捕获 phase 2 起始的窗口。Page 向量和观测 query 都是整个 run 的。
- 启动器的规则假定 run 只有一次前向。更长的 run 由池的余量承担（24 GiB、16 个 session 时，`goal_query` 余 251 个全局 Page，`observation_query` 余 1,518 个，约合 4,000 和 24,000 token 的 run；25 GiB 时余 1,063 和 2,330 个，约合 17,000 和 37,000 token）；即使释放所有空闲 session 也放不下时，以 `CAPACITY_EXCEEDED` 拒绝，不会无限等待。
- 计算的切分方式与一次 10K token 的前向不同（stock vLLM 在 8,192 token 处切分这样的 prefill），因此这种轮次的输出可能与 stock vLLM 在最后几位上不同；尚未测量。

**4,096 token 的前向（可选，gemma-forward4k）。** `--max-num-batched-tokens 4096` 让一次 native 前向为 4,096 token（组里的 `max_num_batched_tokens`），而不是 8,192。不加该选项或使用 `--max-num-batched-tokens 8192` 时，每个 Gemma 启动都与上文所述逐字节相同；Qwen release 拒绝该选项（它保持 16,384）。
- 它改变的内容：vLLM 的 `--max-num-batched-tokens 4096` 和窗口布局的 `"forward_tokens": 4096`（`--additional-config`、`startup.json` 的 `kv_layout` 与 `effective_server_config`；在选择性规则下还有 `request_pages` 的数值）。`remote_ra_http/config.py` 接受 8,192 或 4,096 token 的布局，并拒绝与布局不一致的 `--max-num-batched-tokens`。
- pool 1 预留一次 4,096 token 前向的瞬态块：每个滑窗组 ⌈(1,023 + 4,096) / 16⌉ + 1 − 65 = 256 个块（8,192 时为 512），共少 1,280 个 block id，因此在两种 Page 大小下 pool 1 都小 3.125 GiB，全局池在 P = 16 时多出 2,560 个 Page。16 个 session：6,481 个 block id，15.82 GiB（`goal_query`：7,121 个，17.39 GiB；8,192 时为 18.95 和 20.51 GiB）。
- 启动器的规则按 4,096 token 的前向计算（一次前向的 Page；8,192 token 预算、4,096 token 回复时一个选择性轮次为 2 × (512 + 513) + 1 = 2,051 个 Page，8,192 时为 2,563）。131,072 token 下 16 个 session、P = 16，依次为全局 Page 数 / 规则所需 / 最多放得下的 session 数（括号内为 8,192）：

| `--kv-cache-gib` | `observation_query`、`random_fill`（选择性规则） | `goal_query`（选择性规则） | `full`（释放规则）：所需 / 最多 |
| --- | --- | --- | --- |
| 24 | 6,698 / 2,132 / 23（4,138 / 2,619 / 18） | 5,418 / 2,120 / 21（2,858 / 2,606 / 16） | 8,787 / 12（9,018 / 8） |
| 25 | 7,518 / 2,141 / 24（4,958 / 2,627 / 19） | 6,238 / 2,128 / 22（3,678 / 2,614 / 17） | 8,796 / 14（9,026 / 9） |
| 28 | 9,975 / 2,165 / 28（7,415 / 2,652 / 23） | 8,695 / 2,152 / 26（6,135 / 2,639 / 21） | 8,820 / 17（9,051 / 13） |
| 32 | 13,252 / 2,198 / 33（10,692 / 2,684 / 28） | 11,972 / 2,185 / 31（9,412 / 2,672 / 26） | 8,853 / 22（9,083 / 18） |

- 因此用这个选项时 16 个 `full` session 在 28 GiB 下放得下（benchmark 设置仍用 32）；25 GiB 下，留给超过一次前向的 phase 1 的余量是 `observation_query` 5,376 个全局 Page、`goal_query` 4,109 个（约 86,000 和 66,000 token 的 run；8,192 时为 2,330 和 1,063）。vLLM 自己的启动检查在 25 GiB 下需要 18.44 GiB（`goal_query`：19.98；8,192 时为 22.15 和 23.70）；一个 131,072 token 的请求加 16 个窗口则需要 25.82 GiB（8,192 时为 28.95）。
- 分块的 phase 1：超过 4,096 token 的 phase 1 分成每次 N − W − P = 3,056 token 的前向（P = 16；P = 32 时为 3,040；`window_capacity.phase1_chunk_tokens`；N = 8,192 时仍是 4,096，不变），这是新滑窗块（每组最多 ⌈(1,023 + 块长) / P⌉ + 1 个）仍能放进该前向瞬态（P = 16 时 256 个）的最大块长。4,096 token 的块每组最多会占 321 个块：其他 session 的窗口都被固定时，这样的轮次永远无法准入（任何释放都不会释放窗口），会一直等待（`CAPACITY_WAIT`）。不超过 4,096 token 的 phase 1 仍是一次前向。4,097 到 8,192 token 的 observation 在 8,192 时是一次前向，现在也分块运行，因此其输出可能与 8,192 token 的服务在最后几位上不同；冷 prefill（第 0 轮、重建）按 vLLM 的 4,096 token 前向运行。
- 不变的内容：Goal 前向（最多 2,048 token）及其瞬态、容量门控、`--release-idle-pages`、`--admission-overlap`、三种协议，以及按 `--kv-cache-gib` 设定的 `--goal-rewriter-batch-tokens` 默认值（KV 字节数相同；4,096 token 的前向只会降低激活峰值，尚未测量）。
- 这些服务的 stock vLLM 对照也要加 `--max-num-batched-tokens 4096`。
- CPU 测试：`test_gemma_forward4k.py`（上面的 pool、瞬态和容量数值，vLLM 的规划器，对 12,000 token 以内每个 run 的分块规则，门控的数值，`test_gemma_flags` 中每个策略与选项组合以及 kv-session 和 chat-text 协议的启动，它们与 8,192 的启动恰好只在前向上不同，拒绝情形，启动检查；在引擎和 worker 的 fake 上、按相同的前向与窗口之比，分块运行的长 observation 与一次前向的轮次相同，而按一次 native 前向分块时会一直等待滑窗池）。

16 个 session、4,096 token 的前向，dry run：

```sh
python -B serve.py --release-dir "$RELEASE-gemma4" --task /abs/path/to/tasks-16.json \
  --policy observation_query --active-token-budget 8192 --release-idle-pages --page-tokens 16 \
  --kv-cache-gib 25 --dram-arena-gib 72 --max-model-len 131072 --max-tokens 4096 \
  --max-num-batched-tokens 4096 --language-model-only --output "$RUNS/gemma-obs16-4k" --port 8000
```

**`goal_query`。**
- Goal 即第 5 节的 Goal，没有改动：它的维护逻辑、rewriter 及其 prompt、gate 和 Goal 文本。engine 从 Gemma 的 transcript 构造每轮的 Goal 上下文，方式与 Qwen 相同：根目标（task 消息中的 `<pr_description>`）、该 session 上一轮解码出的回复（空 thought channel 之后生成的文本，不含 `<turn|>`，即客户端回传的内容）、最新的观测及其 `<returncode>`。
- Goal 前向是对 `<retrieval_query>` 文本单独做的一次冷前向，用 Gemma 的 tokenizer 编码，前面加 `<bos>`（id 2），与 Gemma 的每个序列一样开头（PoC 对 Qwen 的 Goal 不加特殊 token；Qwen 没有 BOS）。它的 query 向量与观测 query 相同：Goal 最后一个 token 在第 59 层 final norm 之前的隐状态，用 5,376 维的 pin 投影，与 Page 向量在同一空间。
- 它在 phase 1 之后运行，此时 phase 1 新分配的滑窗块仍被占用，所以 pool 1 为它单独预留瞬态块：Goal 最多 G = 2,048 token 时每个滑窗组 ⌈G / P⌉ 个块（`model_profile.GEMMA4_GOAL_TOKENS`；Qwen 在 16 session SWE-bench 运行 job 3363871 中的 Goal 前向最多 1,164 token）。更长的 Goal 会在分配任何块之前以 `PHASE1_TOO_LONG` 使该轮失败。选页生效的轮次，gate 为 Goal 前向计入 ⌈G / P⌉ 个全局 Page 和每组同样多的滑窗块。
- 内存：640 个块 id，1,600 MiB，两种 Page 大小都一样。16 个 session、131,072 token、8,192 token 预算、4,096 token 回复时，全局池在 24 GiB 下剩 2,858 个 Page（`observation_query` 为 4,138），25 GiB 下剩 3,678 个（4,958），28 GiB 下剩 6,135 个（7,415），选择性规则分别需要 2,606、2,614 和 2,639（2,619、2,627 和 2,652）：16 个 `goal_query` session 需要 23.70 GiB（不含 Goal 时 22.13），24、25、28 和 32 GiB 分别最多容纳 16、17、21 和 26 个（`observation_query`：18、19、23 和 28）。vLLM 自己的启动检查在 pool 1 之外为规则的 `request_pages` 向 pool 0 要空间，25 GiB 时 `goal_query` 需要 23.70 GiB，`observation_query` 和 `random_fill` 需要 22.15 GiB（`test_gemma_goal.py`）。释放规则下的 `full` 需要 30.02 GiB，在 25 GiB 下与 28 GiB 时一样被拒绝。若 Goal 瞬态按一整次 8,192 token 前向预留（6.25 GiB），24 和 28 GiB 都放不下 16 个 session。Goal rewriter（Qwen3-0.6B，约 2 GiB）与模型和 KV 池共用 GPU；在 Gemma 旁边，作业 3410948 中它的进程占 2.74–2.77 GiB（见下面的 token 上限）。
- 接受 `--admission-overlap K`（4.10 节），启动记录与 Qwen 相同（`--phase1-in-step` 和 `--gdn-device-handoff` 仍被拒绝）。全局池的预留与 Qwen 相同。滑窗池中，每个进行中的 admission 预留它的这一轮被接受后仍占用的块，而 phase 1 的瞬态块（新分配的块、Goal 前向）只预留一次，即进行中最大的那份：phase 1 一次只有一个 lease，且在 engine 线程的一次调用内完成（不开 `--goal-overlap` 时，admission 只在 phase 1 之前停在 Goal 处，之后不再停）。否则每个 admission 的 phase 1 峰值都要占一次前向的 pool 1，观测为 2,600–4,300 token 时进行中的 admission 只能有一到两个。在 CPU fake 上（16 个 session），16 个第 1 轮 admission 全部同时停住，而每个 admission 各预留一份 phase 1 峰值时只有 6 个。24 GiB 下进行中的 admission 数量仍受全局池限制：每个续接轮次都要预留它的行和恢复的 Page，因此会为它们释放空闲 session 的 Page（4.13 节）。16 个 session 时：`goal_query --admission-overlap 16 --release-idle-pages --active-token-budget 8192`，24 或 28 GiB。
- 接受 `--goal-overlap`（4.10 节），条件与 Qwen 相同：`goal_query` 加 `--goal-rewriter-model`，开不开 `--admission-overlap K` 都可以。它只固定 `admission.goal_overlap`，不加 vLLM 参数。Goal 前向仍是同一个在 Goal 瞬态里自有块上的冷前向，所以 Goal 向量和选页都相同。某一轮在 phase 1 之后等待 Goal 时，adoption 本会释放的 session 旧窗口块在 phase 1 的 drain 时就已释放，滑窗账本把该轮仍占用的块从它的 base 中扣除。pool 1 和 gate 的数字都不变（`docs/goal-overlap/DESIGN.md` 第 9 节）。CPU 测试：`test_gemma_goal_overlap`。
- 接受 `--goal-rewriter-batch N`（4.10 节；gemma-rewriter-batch），条件与 Qwen 相同：`goal_query` 加 `--goal-rewriter-model` 和 `--admission-overlap K ≥ N`，可用于 XML、kv-session 和 chat-text 协议（chat-text 上的 `goal_query`：gemma-tb-goal，见上），拒绝情形也与 Qwen 相同。它在 `startup.json` 和 `ready.json` 中固定 `goal_rewriter_batch` 和 `goal_rewriter_batch_tokens`，此外不改变任何东西：没有 vLLM 参数，窗口布局和 worker 启动记录都不变。批量改变的只是 rewriter 调用的执行方式，它们在各 session 的 Goal 线程上运行，此时对应的 admission 处于停住状态；窗口布局为 `goal_query` 增加的内容在 Goal 完成后在 engine 线程上运行，每个 engine 循环只恢复一个 admission，且只读取已完成的 Goal：Goal 前向的瞬态块（`goal_tokens`）、选择性规则的 `request_pages`、phase 1 之后的 Goal 前向，以及 `--admission-overlap` 的滑窗 ledger。批量得到的 Goal 可能与不批量的不同（4.10 节）；它的 Goal 前向受同一个瞬态块上限约束（`PHASE1_TOO_LONG`）。
- **token 上限：`--goal-rewriter-batch-tokens` 的默认值随 `--kv-cache-gib` 而定**（`serve.py` 的 `GEMMA4_GOAL_REWRITER_BATCH_TOKENS`，固定为 `goal_rewriter_batch_tokens`；Qwen：16,384；这是估计；benchmark 设置显式给出至少留出 2 GiB 的 T：SWE-bench 为 15,360，Terminal-Bench 为 20,480，见上）：24 GiB 时 28,672，25 GiB 时 25,600（选择性服务的大小，用户 10-21 的决定；launcher 的选项在集成分支中加入 25），28 GiB 时 13,312，32 GiB 时没有默认值，此时 `--goal-rewriter-batch` 需要显式给出 T，表中没有的大小也一样。在作业 3410948（runtime `316a1b4`：16 个 `goal_query` session，不批量的 rewriter，`--admission-overlap 16`，CUDA graphs `full`，32 GiB KV，131,072 token）中，服务在 12 次 240–258 MiB 的 admission 分配上显存不足，当时 EngineCore 占 92.22–92.24 GiB，rewriter 进程占 2.74–2.77 GiB，CUDA 可分配的总量为 95.08 GiB（97,362 MiB）。KV 为 G GiB 时，EngineCore 比它在那里最大的一次报告（94,454 MiB）少占 32 − G GiB；一个批次得到的是减去 258 MiB 的 admission 瞬态、rewriter 子进程的权重和上下文（2,519 MiB，4.10 节）以及 1 GiB 余量之后剩下的部分。按 4.10 节的 Qwen3-0.6B 模型（T 个填充 token 的最宽批次最坏情况是 2 × T/2，每行 384 个新 token），默认值取在 1.3 倍分配器余量下批次仍放得下的最大的 1,024 的倍数。24 GiB：EngineCore 86,262 MiB，批次可用 7,299 MiB，T 不计余量最多 36,068，计入余量最多 29,098，默认 28,672（5,516 MiB，计入余量 7,171）。25 GiB：87,286、6,275、31,900 和 25,624，默认 25,600（4,822 MiB，6,268）。28 GiB：90,358、3,203、17,918 和 14,113，默认 13,312（2,314 MiB，3,008）。32 GiB：94,454 和 −893 MiB，放不下任何批次（不批量的服务在那里已经显存不足）。Qwen 的 16,384 需要 3,770 MiB。显存不足的批次会拆成两半运行，直到单个调用（4.10 节），但 EngineCore 的分配没有这样的退路：它失败会拒绝该请求，所以要留余量。
- CPU 测试（`test_gemma_goal_rewriter.py`，在窗口布局的 engine 和 worker 替身上，使用进程内的批量子进程，它的各行独立解码，因此批量与不批量的回答逐 token 相同）：16 个 `goal_query` session，开 `--admission-overlap 16`、`--release-idle-pages`（空闲 session 的 Page 被释放并重新接纳）、N = 4 和 13,312 的上限（28 GiB 的默认值），每一轮的 Goal 都调用 rewriter，每一轮的决定、选择、注意的 Page 和回复都与不批量的 rewriter 完全相同；子进程收到的请求相同，每个 Goal 前向都在它的瞬态块之内；16 个第 1 轮的 rewriter 调用被扣住直到全部排队，然后以每批至多 4 个、不超过上限的批次运行，含有八个较长 prompt（超过上限的四分之一）之一的批次停在 3 个；6 个 MemAgent kv-session session（K = 6）同样如此。此外还覆盖两种协议上的 launcher 和启动固定值、拒绝情形、`ready.json` 和默认值的推导。

16 个 session 的 dry run：

```sh
python -B serve.py --release-dir "$RELEASE-gemma4" --task /abs/path/to/tasks-16.json \
  --policy observation_query --active-token-budget 8192 --release-idle-pages --page-tokens 16 \
  --kv-cache-gib 24 --dram-arena-gib 79 --max-model-len 131072 --max-tokens 4096 \
  --language-model-only --output "$RUNS/gemma-obs16" --port 8000
python -B serve.py --release-dir "$RELEASE-gemma4" --task /abs/path/to/tasks-16.json \
  --policy goal_query --goal-rewriter-model "$MODELS/qwen3-0.6b" --active-token-budget 8192 \
  --release-idle-pages --admission-overlap 16 --page-tokens 16 --kv-cache-gib 24 --dram-arena-gib 79 \
  --max-model-len 131072 --max-tokens 4096 --language-model-only --output "$RUNS/gemma-goal16" --port 8000
```

**服务配置：CUDA graph 与解码快速路径。** Gemma 服务接受 `--cuda-graphs piecewise`、`full` 或 `full-decode`、`--decode-fastpath`、`--decode-fastpath2` 和 `--decode-fastpath3`，其 pin 和启动检查与 4.8、4.11 节相同；不加这些选项时，其 vLLM 命令、启动配置和证据不变。合起来就是与 Qwen 相同的服务配置：`--cuda-graphs full --decode-fastpath --decode-fastpath2 --decode-fastpath3`（benchmark 设置三条都开）。它们可以与每个策略、`--admission-overlap K` 和 `--release-idle-pages` 一起使用（CPU：`test_gemma_flags`）。
- **CUDA graph。** 全局组是 window 布局中唯一的 FullAttentionSpec 组（vLLM 的 SlidingWindowSpec 不是 FullAttentionSpec）。使用 FULL graph 时，runner 在 capture 和 replay 时都把它绑定到物理 seq-len 缓冲区，并保留物理 slot mapping；五个滑窗组保留 vLLM 的逻辑 seq len、position 和 slot mapping。captured graph 读取的每个缓冲区都是 runner 的持久缓冲区，每次 replay 之前就地重写（CPU 测试：overlay runner 在六个组上构建 metadata 时，每个组在 capture 和 replay 时用的是同一组缓冲区）。编译模式要求 attention 和 KV-cache 更新是 splitting op（没有 GDN op）。phase 1、它的各个分块和 Goal 前向以 eager 方式运行。16 个 session 时 capture size 为 1、2、4、8、16、24 和 32。torch.compile（`piecewise`、`full`）与 Qwen 一样会改变 bf16 数值：应与同一模式、`TRITON_ATTN` 的 stock vLLM 比较。
- **`--decode-fastpath`。** Qwen 的快速路径加上 window owner 的增量观测：当完整观测会接受相同的行时，一步在上一步的观测上增量扩展。full-attention 的 lease 按新追加的块增长，每个滑窗组的 lease 按进入窗口的块和离开窗口的块变化，claim 和 allocation 集合也只按这些差异更新；窗口没有移动时，其 lease 既不重建也不重新检查。
- **`--decode-fastpath2`。** Qwen 的每步一个 `allocation_batch` RPC、把前向之后的工作放到 logits 之后，加上增量的滑窗行（`remote_ra_live_v2/window_rows.py`）：engine 逐步跟踪每个滑窗窗口，而不是每步重建整行；有变化的请求，其更新对每个 KV 组只携带窗口的起始下标、行长度和新追加的（block, incarnation）对，没有变化的请求不带行；worker 以紧凑形式保存滑窗行。131,072 token 时，一次有变化的更新原本约携带 41K 对（五个各 8,192 项的滑窗行加全局行），现在只携带新追加的那几对。请求的第一次更新仍是整行。
- **`--decode-fastpath3`**（gemma-fastpath3，需要 `--decode-fastpath2`）。Qwen 中不依赖 Mamba 行的削减，加上 window 布局自己的削减（4.11 节）：窗口池记录它改变了引用计数或 incarnation 的块，因此引擎只在请求窗口中的某个保留块发生变化时才逐块检查这些窗口；没有变化的一步在做完相同的检查之后沿用它的观测、lease 和 key 元组；窗口移动后的槽位以 C 的速度检查。CPU 测试：`test_gemma_fastpath3`。
- **vLLM runner 的行。** vLLM 的 runner 只会追加请求的新 block id；EngineCore 只在它自己的行中把离开窗口的块置为 null。因此 worker 只在 live 窗口（包含该步的全部读取）上检查 runner 的滑窗行是否等于 EngineCore 的行，而不是比较整行。第一阶段的检查比较整行，在真实 runner 上第一次窗口滑动时就会拒绝；CPU fake 现在按 vLLM 0.24 runner 的方式保存行（以及 numpy block table）。

16 个 session 使用该服务配置的 dry run：

```sh
python -B serve.py --release-dir "$RELEASE-gemma4" --task /abs/path/to/tasks-16.json \
  --policy observation_query --active-token-budget 8192 --release-idle-pages --page-tokens 16 \
  --kv-cache-gib 24 --dram-arena-gib 79 --max-model-len 131072 --max-tokens 4096 \
  --cuda-graphs full --decode-fastpath --decode-fastpath2 \
  --language-model-only --output "$RUNS/gemma-obs16-fast" --port 8000
```

**把完成文件移出引擎线程：`--deferred-completion-io`（4.11 节）。** Gemma 服务以与 Qwen 相同的 pin（`startup.json`、`ready.json`）接受它，不增加 vLLM 参数，可与每个策略、协议和上面的选项一起使用；window 布局自己的完成处理仍在引擎线程上（4.11 节；CPU：`test_gemma_deferred_completion`）。

**MemoryAgentBench：kv-session 协议（gemma-mab）。** `--protocol kv-session`（6.6 节）在 Gemma release 上使用 Gemma 的轮次格式；启动记录 `kv_session` 多一个 `"grammar": "gemma4"`，`/v1/kv-session/info` 在 options 中给出它。线上格式、slot、上下文上限的拒绝和证据都与 6.6 节相同。
- **关闭 thinking**（与 MemoryAgentBench 的决定一致；`--kv-session-thinking on` 只做过 CPU 测试，gemma-thinking：它固定模板的 thinking-on 生成头，user 轮之后为 `<|turn>model\n`，tool response 之后为 `<|channel>thought\n`，各 3 个 token，此后每个请求都带其中之一）。user 轮之后的生成头为 `<|turn>model\n<|channel>thought\n<channel|>`，7 个 token，启动时与固定模板对照检查一次（`kv_session_protocol.check_gemma4_header`、`gemma4_renderer.Gemma4SourceRenderer.kv_transcripts`）。
- **没有生成头的续接。** 在 tool response 之后，Gemma 的模板继续同一个 model 轮，不加生成头（thinking 关闭时），所以客户端发送 `header_tokens` 0。这样的 tail 以 `<tool_response|>` 结尾，phase 2 只计算这一个 token（e = N − 1）；服务端会检查它。MemAgent 的读取轮就是这种轮次；user 消息之后的轮次（格式错误提示、问题）带生成头。
- **边界与停止。** 停止 id 为 0、1、50 和 106。新消息从 `<|turn>` 开始（`input_ids[boundary_tokens]`，在文本回复的 `<turn|>` 和 `\n` 之后：边界为 2），或紧接在 `<|tool_response>` 之后（`input_ids[boundary_tokens − 1]`）：带 tool call 的回复停在 `<|tool_response>`（id 50）上，它同时也是承载其 response 的 tail 的第一个 token，所以边界为 1。当上一回复的最后一个采样不是 tail 的第一个 token 时（`<eos>`、pad 或长度停止，或停在 `<turn|>` 上的 tool call），客户端把它补在前面（+1），与 Qwen 客户端对 `<|im_end|>` 以外所有 id 的处理相同。
- **客户端的 tail。** Gemma 的模板先渲染回复的 tool call，再渲染 tool response，最后才是回复的文本；并且回复不带 tool call 时，后面的 tool 消息会被丢掉。因此 Qwen 的 marker（把最新回复换成文本 marker）找不到 tool-call 回复的 tail。重新实现的客户端（`test_gemma_kv_session.GemmaKvClient`，也是 MemoryAgentBench harness runner 的参照）改为渲染一个替身：回复的 tool call、不带文本；后面没有消息时模板以 `<|tool_response>` 收尾，tail 从这个 token 开始。不带 tool call 的回复沿用 Qwen 的 marker。
- **逐 token 一致**（固定 tokenizer 和模板，`test_gemma_kv_session`）：MemAgent 的首个 prompt 为 219 个 token（前缀 212 + 生成头 7），每个服务的 prompt 都与模板对其消息的渲染逐 token 相同，只是每个 model 轮在 `<|turn>model\n` 之后保留了空的 thought channel（回复所跟的生成头；Gemma 的模板重新渲染历史时会去掉它，每个 model 轮 4 个 token）。因此每轮重新渲染整个对话的 stock vLLM 服务（Vanilla 传输）从第一个 `<|turn>model\n` 之后、前缀 + 3 处开始不同；对服务过的 prompt（token id）做 stock 重放才是精确对照。
- **内存，16 个 session、32,768 token 预算、4,096 token 回复、P = 16**（`--release-idle-pages`，选择性规则）：滑窗池占 18.95 GiB（16 个 812.5 MiB 的窗口共 12.70 GiB，加一次 8,192 token 前向的瞬态 6.25 GiB；`goal_query` 另加 1.56 GiB 的 Goal 瞬态），28 GiB 下全局池需要 5,724 个 Page，7.0 GiB（一次选择性轮次 2 × (2,048 + 769) + 1 = 5,635、其他 15 个 session 的 tail，以及全局池的 1 % 水位线，28 GiB 下为 74 个；`goal_query` 为 5,711；水位线随池的大小变化，所以所需的 Page 数随 KV 大小而不同）：合计 25.94 GiB，带 Goal 时 27.48 GiB。28 GiB 放得下（比规则多 1,690 个 Page，`goal_query` 多 423 个），24 GiB 放不下；不需要把空闲 session 的窗口放到主机（设计中的 WP11：每个窗口 812.5 MiB，未实现）。GPU 紧张时空闲 session 释放其全局 Page，并在下一次 phase 1 之前重新载入；窗口一直留在滑窗池中。一次读两个 chunk（约 8,300 token）以分块 phase 1 从余量中运行。GPU 显存：57.9 GiB 权重加 28 GiB KV，按上文估计的开销，峰值约 92,000–93,300 MiB，比 GH200 的 97,871 MiB 低 4.5–5.8 GiB（带 `goal_query` 约 2 GiB 的 rewriter 时为 2.5–3.7 GiB）；32 GiB 时只剩 0.5–1.7 GiB。
- **上下文。** MemAgent32 的 32 个 case 在 Gemma 下为 39,333–113,764 token（首个 prompt、每个 chunk 及其 tool response 包装和问题；是 Qwen token 数的 0.99–1.01 倍）：131,072 token 下最长的 case 只剩约 17,300 token 给 28 次读取和答案；选择性规则允许的 262,144 对较长的回复也留有余地（Qwen 的计划用 262,144）。锁页 arena 保存每个打开的 session 的每个已封存 Page：仅最长的 16 个 case 就有 150 万 token，回复之前已占 114.5 GiB 的全局 Page，超过一份配额 80.9 GiB 的锁页预算，所以 16 个这样的 session 需要 `--host-memory-shares 2` 和两份配额的作业。
- **25 GiB 下** 32,768 token 预算的 16 个 session 放不进去（24 GiB 也不行；28 GiB 能容纳，见上）。该规则在 25 GiB 下的数字，以及 MemoryAgentBench 服务因此运行的 8 个 session，见 [docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](MEMORYAGENTBENCH_BENCHMARK.zh-CN.md#a15-gemma-4-31b-it) 的 A.15。

```sh
python -B serve.py --release-dir "$RELEASE-gemma4" --protocol kv-session --kv-session-thinking off \
  --policy observation_query --active-token-budget 32768 --page-tokens 16 --kv-cache-gib 28 \
  --max-running-sessions 16 --release-idle-pages --max-tokens 4096 --max-model-len 262144 \
  --dram-arena-gib 150 --host-memory-shares 2 --connector-pool-gib 1 --language-model-only \
  --output "$RUNS/gemma-mab-obs" --port 8000
```

**CPU 测试**（`test_gemma_*.py`；命令见下）。若 `REMOTE_RA_GEMMA4_METADATA` 指向包含五个固定元数据文件的目录，渲染测试会使用真实 tokenizer（需要 Transformers 5.13 和 Tokenizers 0.22.2，与 2.1 节相同）；不设置该变量时这些测试会被跳过。常规测试在 6 或 8 个 session 上运行端到端测试；设 `REMOTE_RA_SLOW_TESTS=1` 时再加 16 个 session、更多分块 phase 1 的情形，以及每个策略和模式的 graph 与快速路径运行，约多 6 分钟。每个模块的 docstring 写明它检查什么；简要如下：

| 模块 | 覆盖 |
| --- | --- |
| `test_gemma_e2e` | 在 vLLM 自己的 KV manager 上运行 engine 和 worker，每个策略三个 session：滑窗读取精确，gate 的数值是每轮取走的块数的上界 |
| `test_gemma_release` | 全局池取选择性规则最小值时的 16 个 session：只释放空闲 session，重新载入精确，窗口不受影响，`CAPACITY_EXCEEDED` |
| `test_gemma_goal` | `goal_query`：Goal 瞬态块、它的 pin 和启动器数值、gate，以及带释放的 16 个 session |
| `test_gemma_chunked` | 分块的 phase 1：算术、gate、lease 检查，每轮与一次前向的结果相同 |
| `test_gemma_full` | 释放规则下 16 个 session 的 `full` |
| `test_gemma_overlap` | `--admission-overlap`：启动记录、滑窗 ledger、停住的 Goal |
| `test_gemma_renderer` | Goal 从 Gemma transcript 得到的输入及其 Goal probe 的 token（真实 tokenizer） |
| `test_gemma_graphs` | CUDA graph 的 pin 和检查、六个组上的缓冲区绑定、挂上 graph 的端到端测试 |
| `test_gemma_fastpath`, `test_gemma_window_rows`, `test_gemma_fastpath3` | window 布局上的解码快速路径（4.11 节） |
| `test_gemma_flags` | 这些选项一起的 320 次启动；各种拒绝 |
| `test_gemma_goal_rewriter` | `--goal-rewriter-batch`（上文） |
| `test_gemma_kv_session`, `test_gemma_kv_session_e2e` | kv-session 的 Gemma 格式、它的客户端，以及端到端的 MemAgent session |
| `test_gemma_chat_text`, `test_gemma_chat_text_e2e`, `test_gemma_chat_text_goal` | chat-text 的轮次格式、端到端的 session、chat-text 上的 `goal_query` |
| `test_gemma_chat_text_replay` | 需显式开启（`REMOTE_RA_TB_ATIF_DIR`）：用 Gemma 的轮次格式回放 8 月的 Terminus 2 轨迹（6.7 节） |
| `test_gemma_deferred_completion` | window 布局上的 `--deferred-completion-io`（4.11 节） |
| `test_gemma_forward4k` | `--max-num-batched-tokens 4096`（见上文“4,096 token 的前向”） |

这些检查截至 2026-11-16 的完整描述冻结在[冻结说明 4.14](history/reference-gpu-notes.zh-CN.md#出自-414-节gemma-4-31b-it)。

```sh
REMOTE_RA_GEMMA4_METADATA=/abs/path/to/gemma-4-31b-it \
  "$VENV/bin/python" -B run_tests.py test_gemma_profile test_gemma_renderer test_gemma_kv_pools \
  test_gemma_selection test_gemma_install test_gemma_capacity test_gemma_startup \
  test_gemma_worker test_gemma_engine test_gemma_e2e test_gemma_release test_gemma_goal test_gemma_overlap \
  test_gemma_full test_gemma_chunked test_gemma_goal_rewriter test_gemma_graphs test_gemma_fastpath \
  test_gemma_window_rows test_gemma_fastpath3 test_gemma_flags test_gemma_kv_session test_gemma_kv_session_e2e \
  test_gemma_chat_text test_gemma_chat_text_e2e test_gemma_deferred_completion test_gemma_chat_text_goal \
  test_gemma_forward4k test_gemma_goal_overlap
```

**尚未完成：** 每行 `runtime.jsonl` 中的窗口记录，以及 session 池下放到主机的窗口（28 GiB 下 MemoryAgentBench 不需要，见上）。

### 4.15 默认开启的速度选项

每个 benchmark 服务都传的开关（[docs/SWEBENCH_BENCHMARK.zh-CN.md](SWEBENCH_BENCHMARK.zh-CN.md)、[docs/TERMINALBENCH_BENCHMARK.zh-CN.md](TERMINALBENCH_BENCHMARK.zh-CN.md) 和 [docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](MEMORYAGENTBENCH_BENCHMARK.zh-CN.md) 的 A.3 和 Gemma 部分）默认开启：

| 选项 | 默认 | 关闭 | 说明 |
| --- | --- | --- | --- |
| `--cuda-graphs` | `full` | `--cuda-graphs off`（eager） | 4.8 节；`piecewise` 和 `full-decode` 不变 |
| `--decode-fastpath` / `--decode-fastpath2` / `--decode-fastpath3` | 开 / 开 / 开 | `--no-decode-fastpath`、`--no-decode-fastpath2`、`--no-decode-fastpath3` | 4.11 节；每一阶段都需要前一阶段 |
| `--deferred-completion-io` | 开 | `--no-deferred-completion-io` | 4.11 节 |
| `--language-model-only` | 开 | `--no-language-model-only` | 4.12 节；Gemma 4 profile 必须带这个参数，会拒绝关闭开关 |
| `--connector-pool-gib` | `1` | `--no-connector-pool-gib` | 4.9 节；关闭即使用该 profile 自己的池：Qwen 为 vLLM 的 32 GiB，Gemma 4 为 1 GiB |
| `--max-tokens`（带 `--enable-thinking` 时） | `4096` | `--max-tokens 1536` | 4.9 和 6.8 节：与 baseline 一致的上限，含思考内容；不开 thinking 时仍为 1536，所以不开 thinking 的服务含义不变 |

- **规则。** 命令行上给出的开关优先于默认值。前置阶段关闭时，后续阶段也关闭：`--no-decode-fastpath` 同时关闭第二、第三阶段，`--no-decode-fastpath2` 关闭第三阶段。同一个选项的开关和关闭开关同时给出会被拒绝；以前 launcher 拒绝的组合，显式要求时仍然被拒绝（例如 `--decode-fastpath2 --no-decode-fastpath`）。在 CPU 测试的启动网格上（两个 profile、三种协议、四个策略：`full`、`observation_query`、`goal_query` 和 `random_fill`，两种布局、开关 thinking、admission 和 Goal rewriter 选项、`--triton-attn`、`--release-idle-pages` 和 `--async-scheduling`），默认值不会把任何原本接受的启动变成被拒绝。
- **仍需显式开启。** `--release-idle-pages`（4.13 节；25 GiB 中 16 个 session 的 `full` 需要它，见[已知限制 2.4](known-limits.zh-CN.md#24-25-gib-中-16-个-session-的-full-需要---release-idle-pages)）、`--admission-overlap K`、`--goal-rewriter-batch N`、`--goal-overlap`、`--gdn-device-handoff`、`--phase1-in-step` 和 `--triton-attn`。
- **`--async-scheduling` 不是速度默认选项**（integrate-1103）：它仍需显式开启（[docs/async-sched/IMPLEMENTATION.zh-CN.md](async-sched/IMPLEMENTATION.zh-CN.md)）。GPU 上的情况：[docs/status.zh-CN.md](status.zh-CN.md)（Qwen 和 Gemma 4；Gemma 4 见[已知限制 5.3](known-limits.zh-CN.md#53-gemma-4-开---async-scheduling-是部分通过)）；之前的记述见[冻结说明 4.15](history/reference-gpu-notes.zh-CN.md#出自-415-节默认开启的速度选项)。当前配置使用它（[docs/status.zh-CN.md](status.zh-CN.md#当前配置)）。不带它时 vLLM 得到 `--no-async-scheduling`，无论使用默认值还是 `--no-speed-defaults`，`speed_defaults` 记录也从不包含它。使用默认值时它的前置条件 `--decode-fastpath2` 已开启，所以只给 `--async-scheduling` 启动的服务与显式给出所有速度开关再加它的服务相同；用 `--no-speed-defaults` 时它需要 `--decode-fastpath --decode-fastpath2`，与以前的命令行一样。它与 `--no-decode-fastpath` 或 `--no-decode-fastpath2` 同用会被拒绝，与 `--gdn-device-handoff`、`--phase1-in-step` 同用也会被拒绝，无论是否使用默认值。两个模型 profile 规则相同：Gemma 4 profile 自 async-gemma 起接受它（[docs/async-sched/GEMMA.zh-CN.md](async-sched/GEMMA.zh-CN.md)）。
- **`--no-speed-defaults`** 让每个没有给出的速度选项回到以前的默认（关闭）：eager 服务、不开快速路径、在引擎线程上写完成文件、加载视觉塔、使用 profile 自己的 connector 池，以及带 `--enable-thinking` 时 1,536 token 的回复上限。这就是以前的命令行，逐字节相同：dry run、vLLM 命令、`startup.json` 和 `launch.json`。与它一起给出的开关仍然生效。
- **记录了什么。** 只有命令行变了；launcher 的 `prepare()` 保留以前的默认值。选项来自默认值的启动，其 vLLM 命令、`startup.json` 和证据与显式给出这些开关的同一启动相同。dry run 和 `launch.json` 增加一条 `speed_defaults` 记录，列出每个速度选项的值和来源（`default`、`switch` 或关闭的原因）。没有任何默认值生效的启动（所有选项都显式给出，或用了 `--no-speed-defaults`）没有这条记录。
- **命令。** benchmark 文档里的服务命令显式给出每个速度选项，所以含义不变。Qwen 的 MemoryAgentBench 服务的正式运行只开了第一条快速路径，因此加上 `--no-decode-fastpath2 --no-deferred-completion-io`。这些页面中不带这些开关写的服务命令会以开启它们的方式启动服务；要得到前面各节描述的 eager 服务（[docs/running.zh-CN.md](running.zh-CN.md) 的 4.1 到 4.5 节，以及 4.6 和 4.7 节），请加 `--no-speed-defaults`。CUDA graph 服务要与同一模式的 stock vLLM 比较（4.8 节）。
- **测试。** `test_speed_defaults.py`（CPU）：三个 benchmark 文档中的每条服务命令，带与不带速度开关，得到相同的 `prepare()` 参数，dry run 除记录外相同；`--no-speed-defaults` 和各关闭开关得到以前命令行的参数，并与以前选项映射的冻结副本对照；还测试前置规则、thinking 的回复上限（mswea-xml 和 chat-text，keep 和 strip，16 个 session、262,144 token、25 GiB）、拒绝、记录和 `launch.json`，以及在两种命令行上不带和带 `--async-scheduling` 的情况；对每条规则的变异都会让这些检查失败。原有的 launcher 测试都传 `--no-speed-defaults`，即它们以前的命令行。
