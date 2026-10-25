# MemoryAgentBench 正式测试配置

[English](MEMORYAGENTBENCH_BENCHMARK.md) | 简体中文

本文档规定本 runtime 的 MemoryAgentBench 正式测试的配置：MemoryAgentBench 数据中的 32 个 case，四种能力各 8 个，每个 case 是一段长历史，agent 通过 `read_next` 工具逐块读完之后回答一个问题；Qwen3.6-27B，关闭 thinking；一个服务上最多同时运行 16 个 case，使用 kv-session 协议（`--protocol kv-session`，README 6.6 节）；`observation_query` 和 `goal_query` 每个策略一个服务；另有一个经由 harness 的 Vanilla transport 访问的 stock vLLM 0.24.0 服务作为对照。**Part A** 是配置。baseline（stock vLLM 对照或其他任何系统）只有满足 A.13 检查表的每一项才可比。**Part B** 是用本仓库工具执行的手动流程，属于可选内容；它需要两样本仓库不包含的输入：case 数据（A.1），以及一个发送 A.7 所述请求的客户端。

Gemma 4 31B IT profile（README 4.14 节）在本版本中以关闭 thinking 的方式服务 kv-session 协议，但它仍在开发中，尚未经过 GPU 验证：它的 runtime 服务目前在 GPU 上还无法启动，启动问题的修复正在进行；本文档只涉及 Qwen3.6-27B。

“README 6.6 节”这类节号指 [README.zh-CN.md](../README.zh-CN.md)。“harness”指本项目用于 agent session 的 MemoryAgentBench harness，它不公开；“官方 harness”指公开的 MemoryAgentBench 仓库（A.2）。

## 验证状态

已在一张 GH200 上经过 GPU 验证：Qwen3.6-27B 跑 kv-session 协议，关闭 thinking。两个作业都于 2026-10-17 在 runtime commit `0d37346` 上运行，并于 2026-10-18 离线核验：

- **作业 3397844（G1）：`PMAB_PASS_VERIFIED`。** 一个 12 块的合成 case（约 48K token 的历史）及其问题，每个服务一个 session，262,144 token 的上下文，24 GiB GPU KV，eager 执行：先是本 runtime 的 `full`（参照组）和 32,768 token 预算的 `observation_query`，再是 eager、同步调度的 stock vLLM 0.24.0，回放 `full` 服务的每一个 prompt。两个 runtime 服务的每一轮都与客户端的构造 token 级精确一致；stock 回放在全部 13 轮上生成了与 `full` 相同的 token，`observation_query` 在全部 13 轮上给出了与 `full` 相同的回复。协议探针全部通过：重复关闭和未知 id 的关闭（200），对已关闭 id 的第 0 轮请求（400），对忙碌 session 的请求（400），一个 262,144 token 的 prompt 在任何模型计算之前以 `CONTEXT_LIMIT` 被拒绝、该 session 之后仍可用，以及 `/health`（200）。一个可选的、使用 vLLM 默认设置并经 A.5 的 Vanilla transport 访问的 stock 服务也通过了。
- **作业 3397842（G2）：runtime 正确；作业的结论 `PMAB_FAIL` 来自 harness 一项检查的 bug，该检查修正后离线重新核验为 PASS。** 16 个 `observation_query` session，32,768 token 预算，`--cuda-graphs full --decode-fastpath`，131,072 token 的上下文，`--dram-arena-gib 39` 和 `--release-idle-pages`（README 4.13 节），分别用 16 GiB 和 24 GiB 的池各跑一次：32 个 5 到 7 块的合成 case，同时运行 16 个，每个 case 都给出了回答。16 GiB 时 139 次释放、124 次重新载入，24 GiB 时 13 次和 13 次；没有任何抢占；32 个 case 在两个池之间完全相同，每次重新载入都恰好恢复了一个已完成轮次所提交的 Page。未通过的那项检查（route-b）把引擎完成一轮之后、在请求的 HTTP 尾段发生的释放也算了进去；用修正后的检查，该服务通过。一个可选的、不开 `--release-idle-pages` 的第三个服务如预期那样因 block 不足而有一轮失败；在本正式测试所用的 runtime 上，这样的一轮会改为等待（README 4.4 节）。

**已运行，评分待出：** 本配置的 32 个 case 的运行（G4）都在 runtime commit `f9322a6` 上，于 2026-10-18 运行：作业 3405844（`observation_query`）、作业 3405845（带 Goal rewriter 的 `goal_query`）和作业 3405855（stock Vanilla 对照）。每个服务都跑完了全部 32 个 case，同时运行 16 个（614 到 618 个请求），没有 runtime 错误。作业 3405844：`PMAB_PARTIAL_PASS_VERIFIED`；runtime 的检查全部通过，有一个 case 停在 agent 自己的连续格式错误上限。作业 3405845：作业结论 `PMAB_FAIL` 只来自 harness 的 route-B 检查，它把一个请求从到达服务端起就算作运行中，而该请求当时仍在等待接纳（这样的 session 算空闲，README 4.13 节）；用修正后的检查离线重新核验，作业为 `PMAB_PARTIAL_PASS`（一个 case 停在格式错误上限）。作业 3405855：作业结论 `PMAB_FAIL` 来自 harness 的指标检查，它也要求 stock 服务不发生抢占；stock vLLM 抢占了 9 个请求，修正后的检查把它记为一项发现，作业为 `PMAB_PASS`。route B 在作业 3405844 中释放空闲 session 的页 385 次，在作业 3405845 中 460 次，每次重新载入都恰好恢复了被释放的页；本 runtime 没有发生任何抢占。在 Qwen release 上，`f9322a6` 的 launcher、启动、worker 和 engine 输出与 `0d37346` 逐字节相同（README 的历史：identity probe，0 处差异）。

尚未完成：这些运行的官方评分（A.10），包括 Long-Range Understanding case 的官方评判。B.3 的 Slurm 脚本、B.2 中 dry run 和评分程序的检查以及 B.6 和 B.7 的脚本都未经测试。本文档不报告任何分数。

## 与 baseline 对齐的配置

我们的运行所用的取值，供 baseline 照样设置；每个取值都取自 Part A（完整检查表见 A.13，仍然存在的差异见 A.14）：

| 项目 | 取值 |
| --- | --- |
| 模型 | `Qwen/Qwen3.6-27B`，revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`，一张 GH200 120GB，官方 vLLM 0.24.0 wheel；chat template 的 SHA-256 为 `e84f32a2…`（A.2） |
| Thinking | 关闭：`--kv-session-thinking off`，即模板以 `enable_thinking=False` 追加的 7 个 token 的生成头；客户端对两种服务都以 `enable_thinking=False` 渲染（A.3、A.7） |
| 解码 | 贪心：客户端发送 `temperature` 0、`top_p` 1、`top_k` −1、`repetition_penalty` 1 和 `seed` 42；vLLM `--seed 42`；不设停止字符串（A.5、A.7） |
| 上下文上限 | `--max-model-len 262144`，prompt 加回复（A.4、A.5） |
| 回复上限 | 4096：客户端的 `max_tokens` 4096（A.4、A.8） |
| 每个 session 的 KV 预算 | `observation_query` 和 `goal_query` 用 `--active-token-budget 32768`；精确注意力不设预算（A.4、A.13） |
| 并发 | 32 个 case，最多 16 个同时运行，最大的先跑，按顺序补入；`--max-num-seqs 16`（A.1、A.9） |
| 本 runtime 的 KV cache | 32 GiB GPU KV：`--kv-cache-gib 32`，34,359,738,368 字节（A.4） |
| Vanilla 对照（vLLM） | `--kv-cache-memory-bytes 34359738368`（32 GiB）、`--max-model-len 262144`、`--max-num-batched-tokens 16384`、`--max-num-seqs 16`；CUDA graph：vLLM 默认（`FULL_AND_PIECEWISE`）；异步调度：vLLM 默认（开启）；`--enable-prefix-caching --prefix-caching-hash-algo sha256`，每个 case 一个 `cache_salt`；不加 `--attention-backend`（vLLM 选择 FLASH_ATTN 加 FlashAttention 3）；`--language-model-only`；prompt 以 token id 发到 `/v1/completions`（A.5、A.6） |
| 吞吐 | 32 个 case 的每个请求（读取轮和回答轮一样）由服务端报告的 completion token 之和 / GPU 节点单调时钟上从第一个 case 开始到最后一个 case 结束（关闭其 session 之后）的时间跨度。包含 admission 及其等待、容量等待、释放和重新载入、Goal rewriter 的时间、客户端的渲染和 session 关闭（`read_next` 由 runner 自己应答，A.7）；不包含服务的启动和关闭、预热请求、评分以及 Goal rewriter 自己的 token（A.11） |
| 任务 | 32 个 MemoryAgentBench case，每种能力 8 个（Accurate Retrieval、Test-Time Learning、Long-Range Understanding、Conflict Resolution），每个 case 一个问题，manifest 的 SHA-256 为 `010fd109…`；harness 的 MemAgent 循环，每个 case 最多 120 轮（A.1、A.7） |
| 评测 | `utils/eval_other_utils.py`（`538026089d…`）中官方的 `post_process`，运行结束后离线执行；Long-Range Understanding：官方的 GPT-4o 评判（`gpt-4o-2024-05-13`），尚待运行（A.10） |
| Gemma 4 31B IT | 打算使用相同的设置；本文档未固定。该 profile 以关闭 thinking 的方式提供 kv-session 协议，但仍在开发中，未经 GPU 验证：它的 runtime 服务还不能在 GPU 上启动（README 4.14 节） |

---

## Part A. 配置

### A.1 范围与 case

| 项目 | 取值 |
| --- | --- |
| 模型 | Qwen3.6-27B，一张 NVIDIA GH200 120GB，vLLM 0.24.0（A.2） |
| 基准 | MemoryAgentBench（A.2）：**32 个 case，四种能力各 8 个**，即 Accurate Retrieval、Test-Time Learning、Long-Range Understanding 和 Conflict Resolution；**每个 case 一个问题**，共 32 个问题 |
| Case | 一段 39,409 到 113,458 token 的历史，切成 10 到 28 块、每块最多 4,096 token，外加一个问题。32 个 case 共有 25 段不同的历史（8 个 Conflict Resolution 问题共用一段）、564 块、2,255,888 个历史 token |
| 并发 | 32 个 case，**最多 16 个同时运行**，按下表顺序启动（最大的先跑）；一个 case 结束后，启动下一个 |
| 服务 | 每个配置一个全新的服务，各自一个 GPU 作业：`observation_query`、`goal_query`（本 runtime），以及 `stock`（stock vLLM 0.24.0，Vanilla 对照） |
| 运行 | 一个服务把 32 个 case 各跑一次，中间不中断（A.12）。一次运行就是一个样本（A.14） |

token 数都按固定的 Qwen3.6 tokenizer 计算。

各项能力（case id 的前缀是它在 MemoryAgentBench 中的来源子集；指标是官方评分程序对它使用的指标，A.10）：

| 能力 | 来源子集 | case 数 | 历史数 | 每个 case 的块数 | 每个 case 的历史 token 数 | 指标 |
| --- | --- | ---: | ---: | --- | --- | --- |
| Accurate Retrieval | `ruler_qa2_validation_65536` | 8 | 8 | 13–15 | 53,051–58,405 | `substring_exact_match` |
| Test-Time Learning | `icl_trec_coarse_6600shot_balance` | 8 | 8 | 28 | 113,458 | `exact_match` |
| Long-Range Understanding | `infbench_sum_eng_shots2` | 8 | 8 | 17–20 | 67,635–78,885 | `llm_judge_f1`（官方的 GPT-4o 评判） |
| Conflict Resolution | `factconsolidation_sh_32k` | 8 | 1 | 10 | 39,409 | `substring_exact_match` |

**数据。** 这些 case 是 harness 从 MemoryAgentBench 中选出的 32 个 case，已经切好块：一个 manifest（SHA-256 `010fd10971905e7611ff639c81230f12bad58acd688143cf7b6eb94e30151edf`）加每个 case 一份上下文。每个 case 带有它的 system prompt、它的各块、harness 提问时使用的问题文本、参考答案、来源子集和指标。这些数据不公开，也不在本仓库中；下面的列表用来标识它们。一个 case 的上下文 SHA-256 是其各块依次拼接后的 SHA-256；我们的 harness 在作业开始时逐个 case 核对，不一致就拒绝。

32 个 case，按运行顺序（id、块数、历史 token 数、上下文 SHA-256；B.1 把这些 id 存为 `cases.txt`）：

```text
icl_trec_coarse_6600shot_balance-r311-q0  28 113458  433e4d18b8adb1faa842a5b443551cf7c95091b7c3939f2bdeda48aa73e51d01
icl_trec_coarse_6600shot_balance-r324-q0  28 113458  3f0a8bc54e5fb822888afb51502313d4c539eb72077ca817665ff92615557463
icl_trec_coarse_6600shot_balance-r337-q0  28 113458  eae0fbec7f482a695e24cd8c677776af63be4b769c5b2284768118568237c462
icl_trec_coarse_6600shot_balance-r350-q0  28 113458  95bf987db26a3ad9424beff48d2058c7397f8c1b19f91a268a337261db528674
icl_trec_coarse_6600shot_balance-r373-q0  28 113458  c0d3cf0ba25a9219769c3d98cc1356d23c6fb3ba485641e8a00e821d77485d8f
icl_trec_coarse_6600shot_balance-r375-q0  28 113458  105ff35c6fd91a9d727f95113d34b3f1c8978a14fe4739c8c576889017b12191
icl_trec_coarse_6600shot_balance-r393-q0  28 113458  1dbdfd59f8fa35ffac2507e128d4a13990a4b635a62257abc3d0154397e01479
icl_trec_coarse_6600shot_balance-r396-q0  28 113458  1d8923f5291c22a86f55985c985df51f9732abd55d8dfce8013a1f3b9d423c3f
infbench_sum_eng_shots2-r52-q0            20  78885  cb1104516284ee52e6c76e435af3a22c995f8671fb4a2989eff62402364003e6
infbench_sum_eng_shots2-r84-q0            19  77050  c0f8cf56d00012e9b78bce6891f3f9be16408bd64d509672fe5506e9264e6ea6
infbench_sum_eng_shots2-r56-q0            19  75767  8d802a1d4dfbf5f3768f321cff1f1ae0662e451634ed80939b5f7612da5cbe8f
infbench_sum_eng_shots2-r90-q0            18  73699  4c6babc357ffba2d21763fef626c6ad97594b3685a55778f4ff648357dd5de13
infbench_sum_eng_shots2-r29-q0            18  73658  1ae329374a16eea23a72cacf14d7a18f6434652a3c9211808d0cd635e7a97f81
infbench_sum_eng_shots2-r91-q0            18  70981  7b93e166866edaa603f97e7fc45763a0fc2107993453471c101d37224959a481
infbench_sum_eng_shots2-r77-q0            18  70375  d3fa1ef5e2ffc10becdfc8114abdecef784a6f354ea62d987b39065d7bb36b50
infbench_sum_eng_shots2-r58-q0            17  67635  b5ce958a6db779794a0fedfc9e5f2087593af5465654938c65b37df1a5a70eb6
ruler_qa2_validation_65536-r365-q0        15  58405  c399a4cf4ec8846816d30d2cf060c91c8bc5a28d5b7964334bc6c2488fb223fb
ruler_qa2_validation_65536-r350-q0        15  57636  ce6cc2295980d4bb7a89de1e0ea0787ff80ef23e18d551148126fb8458b2b5f7
ruler_qa2_validation_65536-r391-q0        14  56571  b0110fa82a27bb154a0242405ee62c171233d3a4c2231a4046696e6ee7a4920c
ruler_qa2_validation_65536-r370-q0        14  55876  a2386cbb9b123d6a50b71f3bd7a9656c2c4f6e4264d5bf306346c9b87bd4be26
ruler_qa2_validation_65536-r341-q0        14  55215  8d313b9a484c93a318c2be73521313eff6fb419a4a665a36eff2153dde6faf8b
ruler_qa2_validation_65536-r342-q0        14  54852  5d2a586b16062185746bac3cbaa5c34f02b5890aa31d9836258c7bc52e407b06
ruler_qa2_validation_65536-r309-q0        14  53296  7a07b7c28a4a12dec2db350f3f3fdc5cb99dfa7d30d1b3de3ccc355a74ce9e1a
ruler_qa2_validation_65536-r301-q0        13  53051  9c3e2d2798496d3f33c0cb977b6ed66a4ae332ddf4859bacbb74daa0fd8fb563
factconsolidation_sh_32k-r5-q22           10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
factconsolidation_sh_32k-r5-q37           10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
factconsolidation_sh_32k-r5-q4            10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
factconsolidation_sh_32k-r5-q5            10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
factconsolidation_sh_32k-r5-q53           10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
factconsolidation_sh_32k-r5-q60           10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
factconsolidation_sh_32k-r5-q70           10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
factconsolidation_sh_32k-r5-q93           10  39409  2a44d6ad3264a775596fe6de01994d6b348280303fdd964f82288e2c476a7409
```

本 runtime 在 MemoryAgentBench 上的策略（README 开头的策略表和第 5 节）：

| `--policy` | 下一轮使用哪些历史 Page |
| --- | --- |
| `observation_query` | Selector V2：在预算内保留必保留页、recent 页和余弦最高的页；query 是最新 observation 末尾的 hidden state（新的 tool 消息及其块；最后一次读取时，后面还跟着问题） |
| `goal_query` | 同一个 selector；query 是持续维护的检索 Goal，由 Qwen3-0.6B rewriter 更新。在本协议上，Goal 的 observation 是新的 tool 和 user 文本，显式要求是问题（README 6.6 节） |
| `full` | 所有历史 Page。不属于 G4 运行：stock Vanilla 对照是精确注意力的参照。它在 G1 中跑过本协议（见“验证状态”）；A.4 列出它的参数 |
| `random_fill` | 没有为 MemoryAgentBench 配置 |

在 `--active-token-budget 32768` 下，容量是 2,048 个 16 token 的 Page。一个 session 已封存的历史超过 8,192 token 后才开始选择（README 第 5 节），但在预算以内所有 Page 都放得下，所以 session 在已封存历史超过 32,768 token 之前关注它的全部 Page，之后只关注选中的一部分。每个 case 在其后面的块中都会超过这个预算。

### A.2 模型与版本

| 项目 | 取值 |
| --- | --- |
| 模型 | `Qwen/Qwen3.6-27B`，revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`，本地目录，必须是普通文件；元数据的 SHA-256 见 README 2.2 节 |
| Goal rewriter（仅 `goal_query`） | `Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`（README 2.2 节） |
| vLLM | 官方 wheel `vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl`，未作任何修改，SHA-256 `700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5` |
| Python 环境 | CPython 3.12、`transformers==5.13.0`、`tokenizers==0.22.2`，用 `examples/env/constraints-3270721.txt` 构建（README 2.1 节）。stock 对照和客户端从同一个环境运行；本 runtime 安装到单独的 release 目录，环境本身保持不变（README 2.4 节） |
| Chat template | runtime 固定的 Qwen3.6 模板，SHA-256 `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`（`source-overlay/qwen_observation_span.py` 中的 `TEMPLATE_SHA256`；去掉一个末尾换行后的模板文本的摘要）。模型目录中的模板必须与它相同 |
| Runtime | 本仓库的 commit `f9322a6`（`f9322a65477deb7a6660c08a316f2d4c26644c26`，G4 运行使用的 commit），或者不改变 kv-session 协议、容量门控和 A.4 中 vLLM 参数的更新 commit。G1 和 G2 两个验证作业运行的是 `0d37346`（见“验证状态”） |
| GPU | 一张 NVIDIA GH200 120GB，Linux aarch64；一张 GPU 同一时间只跑一个服务 |
| Slurm 资源 | `--nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16`；本 runtime 的服务用 `--mem=227120M`（节点主机内存中两张 GH200 的份额，A.4），stock 对照用 `--mem=113560M`（一份） |
| 客户端 | 我们的 runner（A.7）：Python 3.12，只用标准库和上述环境中的 Transformers，在同一个作业里运行于 GPU 节点上；不在本仓库中 |
| 数据 | A.1 的 32 个 case（manifest SHA-256 `010fd109…`） |
| 官方 harness 与评分程序 | `HUST-AI-HYZ/MemoryAgentBench`，commit `538026089d1a8a8eff05121d0db89b388f360eba`（MIT）：`utils/eval_other_utils.py`，SHA-256 `d23a5ba9f2750229315cd308463ae99aef9ff36ed0859b69d4851f804976b815`（`post_process`），以及 `llm_based_eval/summarization_evaluate.py`，SHA-256 `0139db690c4537c8f6ca5fe766ab60caf16050d172412f18ad0535577e708801`（摘要的评判）；其数据是 Hugging Face 数据集 `ai-hyz/MemoryAgentBench` |

### A.3 kv-session 协议

harness 的 agent 循环自己渲染 chat template、自己解析回复；它的 KV-session 客户端发送 token id，本 runtime 服务的就是这个客户端的协议（README 6.6 节）。与本配置相关的部分：

- **由谁渲染。** 客户端渲染对话（它的 system prompt、它唯一的工具、thinking 开关），并解析每条回复中的工具调用；服务端从不渲染模板，也不需要 tool-call parser。因此一组对比中的两种服务看到的都是客户端的 token 化结果：本 runtime 上是每个 case 一条不断追加的 token 流，stock vLLM 上是每一轮的完整重新渲染（A.5）。
- **每个 case 一条 token 流。** 一个 case 就是一个 session，其 token 流只增不减。第 0 轮发送完整的 prompt；之后每个请求只发送新的尾部：上一条回复的结束分隔符（上一条回复的最后一个 token 不是 EOS 时，把它放在最前面）、新消息和下一个生成头。每条回复都以它生成时的原始 id 留在流中。服务端把一个请求映射到阶段 1（边界和新消息）和阶段 2（生成头和 decode），不做任何渲染（README 第 5 节、6.6 节）。
- **Goal 的来源。** 请求还带有客户端的消息列表，它从不生成 token；`goal_query` 从上一个请求之后新增的消息中取得 observation 和显式要求（README 6.6 节）。
- **Session。** kv-session 服务不接受 `--task`：它注册的任务是槽位，客户端 session id 在第一个请求时绑定到下一个未用的槽位（`--kv-session-slots`，默认 64）。case 结束时，客户端关闭该 case 的 session（`POST /v1/kv-session/close`）。
- **Thinking。** `--kv-session-thinking off` 固定 7 个 token 的生成头 `<|im_start|>assistant\n<think>\n\n</think>\n\n`，即模板在 `enable_thinking=False` 下追加的生成头；服务端用它检查每个请求的生成头。
- **生成约定，由服务端执行。** 只支持贪心：采样字段不是 `temperature` 0、`top_p` 1、`top_k` −1 或 0 和 `repetition_penalty` 1 的请求会被拒绝。`max_tokens` 为 1 到回复上限，并截到剩余上下文以内；`finish_reason` 为 `stop` 或 `length`（达到 `max_tokens`）；响应给出每个生成 token 的文本（含特殊 token）及其 id。
- **拒绝。** 每种失败都是 HTTP 400 加 `{"error", "type"}`。没有为回复留出空间的 prompt 在任何模型计算之前被拒绝，文字恰好是 `Logical RoPE context limit reached`（`CONTEXT_LIMIT`）。admission 之前发现的拒绝不影响该 session；引擎中的失败会使该 session 失效。

### A.4 本 runtime：服务参数与 vLLM 参数

**各策略共用的参数**（`serve.py`）：

| 参数 | 取值 | 作用 |
| --- | --- | --- |
| `--protocol` | `kv-session`，加 `--kv-session-thinking off`；不传 `--kv-session-slots`（默认值 64） | A.3 的协议，关闭 thinking。服务生命周期内共 64 个客户端 session；32 个 case 加客户端的预热（A.7）用掉 33 个 |
| `--max-running-sessions` | `16` | K：同时最多打开 16 个 session。vLLM 收到 `--max-num-seqs 16` |
| `--kv-cache-gib` | `32` | vLLM 收到 `--kv-cache-memory-bytes 34359738368` |
| `--page-tokens` / `--kv-layout` | `16` / `decoupled` | 16 token 的注意力 Page，以及一个单独的、有 33 个 slot（2K + 1）的 Gated-DeltaNet 池：32 GiB 时为 100 个 Mamba block 和 27,873 个可用注意力 block（README 4.4、4.6 节）。vLLM 收到 `--block-size 16 --mamba-block-size 784 --additional-config ...` |
| `--max-model-len` | `262144` | 模型原生的上下文（prompt 加回复，README 4.9 节）。最大的 case 在预期回复长度下结束时约 128K token，回复长三倍时约 152K；131,072 会拒绝它的最后几轮。上限处的一轮需要 27,873 个可用 block 中的 17,944 个 |
| `--max-tokens` | `4096` | 回复上限：客户端的 `max_tokens` 取 1 到 4096（客户端发送 4096） |
| `--dram-arena-gib` | `124` | 所有 session 共用一个 124 GiB 的锁页主机 Page arena。它保存每个打开的 session 的每个已封存 Page，直到该 session 关闭；超出它的捕获会让该请求失败（`DRAM_ARENA_EXHAUSTED`，README 4.9 节）。16 个最大的 case 同时处于最后一轮、回复为预期长度的三倍时，需要 123.35 GiB |
| `--connector-pool-gib` | `1` | connector 闲置的 CPU 池设为 1 GiB 而不是 32 GiB（README 4.9 节）。vLLM 收到 `cpu_bytes_to_use` 1073741824 |
| `--cuda-graphs` | `full` | vLLM 收到 `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}`，取代 `--enforce-eager`（README 4.8 节） |
| `--decode-fastpath` | 开启 | 每个解码步的主机工作更少，结果相同（README 4.11 节） |
| `--language-model-only` | 开启 | 不加载视觉塔，没有多模态预热（README 4.12 节）。vLLM 收到 `--language-model-only` |
| `--release-idle-pages` | 开启 | GPU 紧张时，容量门控释放空闲 session 的常驻 Page，并在它们的下一轮之前重新载入（README 4.13 节；见下）。不增加 vLLM 参数 |
| `--port` | `8000` | 服务只绑定 `127.0.0.1` |

预期回复长度是 harness 的规划值：每个读取轮 400 token，回答 1,000 token。

**空闲 session 与 `--release-idle-pages`。** 两轮之间，每个打开的 session 都在 GPU 上保留它的常驻 Page。16 个 session、32,768 token 预算时，一个空闲 session 钉住约 2,300 个 Page，16 个合计约 37,000 个，超过池的容量（README 4.13 节）。开了这个选项，容量门控用释放空闲 session 的全部常驻 Page 来补足一轮的缺口，最久未用的先释放；它们的主机副本留在 arena 中，被释放的 session 在下一次阶段 1 之前把 Page 载回。仍然放不下的一轮会等待（`CAPACITY_WAIT`，README 4.4 节）。本配置允许并检查释放，但不要求一定发生释放。

**主机内存。** 锁页主机内存是 1 GiB 的 connector 池、16 个 Gated-DeltaNet carry（2.29 GiB）和 124 GiB 的 arena，共 127.29 GiB，在 launcher 128 GiB 的锁页主机预算之内（README 4.9 节）；若用默认的 32 GiB connector 池则为 158.3 GiB，launcher 会拒绝。再加约 32 GiB 的进程内存和 5 GiB 余量，作业需要 168,237 MiB，超过节点主机内存中一张 GH200 的份额（113,560 MiB），所以这些服务申请两份，`--mem=227120M`。

**各策略的参数：**

| 策略 | selector 参数 | admission 参数 | Goal rewriter |
| --- | --- | --- | --- |
| `observation_query` | `--active-token-budget 32768` | 无 | 不启动（README 第 5 节） |
| `goal_query` | `--active-token-budget 32768 --goal-rewriter-model <Qwen3-0.6B 目录>` | `--admission-overlap 16` | 同一张 GPU 上一个 Qwen3-0.6B 进程 |
| `full`（不在 G4 运行中） | 无 | 无 | 无 |

两个 query 策略的预算是同一个取值。任何服务都不设置 `--gdn-device-handoff`、`--phase1-in-step`、`--active-ratio`、`--goal-query-variant`、`--goal-rewriter-batch`、`--fill-seed`、`--decode-fastpath2` 或 `--decode-fastpath3`。admission 参数遵循 README 4.10 节（16 个 session 的 `goal_query` 加 `--admission-overlap 16`）。selector、rewriter 和 admission 参数会固定到启动配置里，但不增加任何 vLLM 参数，所以各服务启动 vLLM 时的参数完全相同。

**命令。** 两个服务的 dry run（加 `--start` 即启动服务；`LABEL` 是每次尝试的新标签）：

```sh
SHARED=(--protocol kv-session --kv-session-thinking off --max-running-sessions 16 --kv-cache-gib 32
        --page-tokens 16 --kv-layout decoupled --max-model-len 262144 --max-tokens 4096 --dram-arena-gib 124
        --connector-pool-gib 1 --cuda-graphs full --decode-fastpath --language-model-only --release-idle-pages)
python -B serve.py --release-dir "$RELEASE" --policy observation_query "${SHARED[@]}" \
  --active-token-budget 32768 --output "$RUNS/mab-observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --policy goal_query "${SHARED[@]}" \
  --active-token-budget 32768 --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --admission-overlap 16 \
  --output "$RUNS/mab-goal_query-$LABEL" --port 8000
```

**由此得到的 vLLM 参数**（各服务相同；以 dry run 打印的列表为准，这里按参数组换行；作业 3397842 中 G2 各服务的 dry run 打印的正是这份列表，只是上下文上限和 KV 字节数是它们自己的）：

```text
--model <model directory> --served-model-name selective-prefix --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
--host 127.0.0.1 --port 8000 --distributed-executor-backend uni --no-async-scheduling
--tensor-parallel-size 1 --pipeline-parallel-size 1
--max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 16384
--kv-cache-memory-bytes 34359738368 --seed 42 --enable-prefix-caching --prefix-caching-hash-algo sha256
--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}
--mamba-cache-mode align --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
--chat-template <output>/chat_template.jinja --chat-template-content-format string
--kv-transfer-config {"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":1073741824}}
--gdn-prefill-backend triton
--block-size 16 --mamba-block-size 784 --additional-config {"remote_ra_kv_layout":{"mamba_slots":33,"mamba_tokens":784,"page_tokens":16}}
--language-model-only
```

启动 `serve.py` 的 shell 的**环境变量**（README 第 3 节）：

| 变量 | 取值 |
| --- | --- |
| `PATH` | `$VENV/bin:/usr/local/bin:/usr/bin:/bin` |
| `LD_LIBRARY_PATH` | `$VENV/lib/python3.12/site-packages/nvidia/cu13/lib` |
| `TMPDIR` | 短路径：`len(TMPDIR) + 37` ≤ 107 字节（AF_UNIX 限制） |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0`（原生 top-k/top-p sampler） |
| `CC`、`VLLM_RPC_BASE_PATH` | unset |
| `HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE`、`PYTHONDONTWRITEBYTECODE` | `1` |
| 由 `serve.py` 设置 | `VLLM_WORKER_MULTIPROC_METHOD=spawn`、`VLLM_USE_V2_MODEL_RUNNER=0`，以及 runtime 自己的变量 |

### A.5 stock vLLM 对照（Vanilla）

对照服务就是从同一个环境（A.2）直接启动的 `vllm.entrypoints.openai.api_server`，不带本 runtime 的任何代码，经由 harness 的 Vanilla transport 访问（A.7）。

| 设置 | 取值 | 与本 runtime 的关系 |
| --- | --- | --- |
| `--model` | 同一个 Qwen3.6-27B 目录 | 相同 |
| `--served-model-name` | `selective-prefix` | 同名 |
| `--host` / `--port` | `127.0.0.1` / `8000` | 相同（只用 loopback） |
| `--distributed-executor-backend` / `--tensor-parallel-size` | `uni` / `1` | 相同 |
| `--max-num-seqs` | `16` | 与 K = 16 相同 |
| `--max-model-len` | `262144` | 相同 |
| `--max-num-batched-tokens` | `16384` | 相同（由 launcher 固定） |
| `--kv-cache-memory-bytes` | `34359738368`（32 GiB） | 字节数相同 |
| `--seed` | `42` | 相同 |
| `--enable-prefix-caching --prefix-caching-hash-algo sha256` | 开启 | 相同；客户端的 `cache_salt` 把各 case 隔开（A.9） |
| `--language-model-only` | 开启 | 相同（README 4.12 节） |
| `--mamba-cache-mode` | `align` | 相同 |
| `--gdn-prefill-backend` | `triton` | 相同 |
| CUDA graph | vLLM 默认：既不加 `--enforce-eager` 也不加 `--compilation-config`，结果是 `FULL_AND_PIECEWISE` 加 `VLLM_COMPILE`（README 4.8 节） | 模式相同 |
| 异步调度 | vLLM 默认（开启），不加参数 | 不同：launcher 总是传 `--no-async-scheduling`（A.14） |
| 注意力后端 | 不设置，由 vLLM 选择（A.6） | 相同 |
| 路由 | `POST /v1/completions`，prompt 以 token id 发送 | 不同：本 runtime 服务 `/v1/kv-session/complete`（A.3） |
| Thinking | 关闭，体现在客户端的渲染中（A.7） | 生成头相同 |

每个取值都显式设置，这样对照服务不依赖 vLLM 对这个混合模型的默认值。只服务于 runtime 自身路由的参数（`--revision`、reasoning 和 tool parser、`--chat-template`、`--kv-transfer-config`、`--block-size`、`--mamba-block-size`、`--additional-config`）不传：客户端发送 token id，并自己解析回复。

**命令**，在设置了 A.4 环境变量的 shell 中：

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
"$VENV/bin/python" -I -B -m vllm.entrypoints.openai.api_server \
  --model "$MODELS/qwen3.6-27b" --served-model-name selective-prefix --host 127.0.0.1 --port 8000 \
  --distributed-executor-backend uni --tensor-parallel-size 1 \
  --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 16384 \
  --kv-cache-memory-bytes 34359738368 --seed 42 \
  --enable-prefix-caching --prefix-caching-hash-algo sha256 \
  --mamba-cache-mode align --gdn-prefill-backend triton --language-model-only
```

- **环境变量。** 与 A.4 相同（`PATH`、`LD_LIBRARY_PATH`、短 `TMPDIR`、`VLLM_USE_FLASHINFER_SAMPLER=0`、unset `CC` 和 `VLLM_RPC_BASE_PATH`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`PYTHONDONTWRITEBYTECODE=1`），另加 `serve.py` 为本 runtime 设置的两个变量（`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`VLLM_USE_V2_MODEL_RUNNER=0`）和 `VLLM_NO_USAGE_STATS=1`。
- **生成约定，由客户端发送**（Vanilla transport，A.7）。每一轮把以 `enable_thinking=False` 渲染的完整对话作为 `prompt` token id 发给 `POST /v1/completions`，带 `max_tokens` 4096、`temperature` 0、`top_p` 1、`top_k` −1、`repetition_penalty` 1、`seed` 42、`skip_special_tokens` false、`return_token_ids` true，以及每个 case 新生成的 `cache_salt`；没有停止字符串。发送之前，如果 prompt 加 4096 超过 262,144 token，客户端自己结束该 case。
- **就绪判断。** `GET /health` 返回 200，且 `GET /v1/models` 列出 `selective-prefix`。
- **关闭 session。** stock vLLM 没有 session；客户端不发送关闭请求。
- **抢占。** KV 池满时 stock vLLM 可能抢占请求；本 runtime 从不抢占（A.14）。在运行期间和结束时记录 vLLM 的计数器 `vllm:num_preemptions_total`（`GET /metrics`）。

### A.6 kernel 后端

两种服务都不传 `--attention-backend`；注意力后端由 vLLM 在 GH200 上自动选择。两种服务使用相同的 kernel：

| 组件 | 设置 | 服务日志行 |
| --- | --- | --- |
| 注意力 | FLASH_ATTN 加 FlashAttention 3（vLLM 自动选择） | `Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']` 和 `Using FlashAttention version 3` |
| Gated-DeltaNet prefill | `--gdn-prefill-backend triton` | `Using Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128)` |
| Sampler | 原生 top-k/top-p（`VLLM_USE_FLASHINFER_SAMPLER=0`） | `FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0` |
| CUDA graph | `FULL_AND_PIECEWISE`，两边都是 16 个序列，捕获尺寸为 1、2、4、8、16、24 和 32（README 4.8 节） | 引擎配置行中的 `'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE` 和 `'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32]` |
| 异步调度 | 本 runtime：关闭。stock：开启 | `Asynchronous scheduling is disabled.`（本 runtime）或 `Asynchronous scheduling is enabled.`（stock） |
| 纯文本 | 两边都用 `--language-model-only`（README 4.12 节） | vLLM 的 `running in text-only mode`；不能出现 `Encoder cache will be initialized` 或 `Multi-modal warmup completed`（可以出现 `Using ... for MMEncoderAttention` 行） |

READY 之后从每个服务的日志中确认：

```sh
grep -oE "Using FLASH_ATTN attention backend|Using FlashAttention version [0-9]+|Using Triton/FLA GDN prefill kernel \(requested=[a-z]+|FlashInfer top-p/top-k sampling disabled|Asynchronous scheduling is [a-z]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|'cudagraph_capture_sizes': \[[0-9, ]*\]|running in text-only mode|Encoder cache will be initialized|Multi-modal warmup completed" \
  server.log | sort | uniq -c
```

表中每一行都必须出现，且不能出现其他注意力后端、FlashAttention 版本、prefill kernel 或 CUDA graph 模式，也不能出现多模态启动日志行。日志不符的服务不纳入对比。

### A.7 runner

我们的 runner 是本项目自己对 harness 的 MemAgent agent 循环及其两种 transport 的重新实现：KV-session 客户端（本 runtime）和 Vanilla transport（stock vLLM）。harness 的任何代码都不执行、不导入，runner 也不在本仓库中。循环中的指令文字是 harness 的；runner 复现了这些文字，本文档不复现。baseline 使用 harness 本身，或者使用发送相同请求的客户端。runner 在 GPU 节点上以单独的进程运行，访问 `127.0.0.1`。

| 项目 | 取值 |
| --- | --- |
| Tokenizer 与模板 | 模型目录的 tokenizer，经 Transformers 的 `apply_chat_template`，`enable_thinking=False`，带 harness 唯一的工具 `bash`（一个字符串参数 `command`）；token 由 `encode(text, add_special_tokens=False)` 得到。启动时 runner 检查模板的 SHA-256（A.2）和生成头（A.3） |
| 消息 | 一条 system 消息（case 的 system prompt 后接 harness 的 `read_next` 说明）和 harness 固定的第一条 user 消息；之后是各条回复，以及下面所述的 tool 和 user 消息 |
| 分块 | 一条回复中有效的 `bash` 调用、且命令为 `read_next` 时，以返回码 0 的 tool 消息得到下一块，并带上该块的序号（`k/N`）；一条回复中的多个这样的调用按顺序得到多块。最后一块带有 harness 的结束说明，case 的问题作为 user 消息跟在同一个请求中 |
| 格式错误 | 在最后一块之前不含工具调用的回复，会得到 harness 的提醒（一条 user 消息）；无效的调用（别的命令，或最后一块之后的 `read_next`）得到返回码 1 的 tool 消息。连续两轮出现这类错误则结束该 case（`format_error`） |
| 回答 | 最后一块之后第一条不含工具调用的回复；它的内容就是 prediction（`answered`）。停在 `max_tokens` 的回答：回复没有 reasoning 部分时（关闭 thinking 时总是如此）记为 `unfinished_thinking`，prediction 为空；否则记为 `answer_length_limit`，保留其文本（harness 的规则） |
| 轮数 | 每个 case 最多 120 个请求（`turn_limit`）。N 块的 case 在每条回复读一块时需要 N + 1 个请求 |
| 采样字段 | `temperature` 0、`top_p` 1、`top_k` −1、`repetition_penalty` 1、`seed` 42、`max_tokens` 4096 |
| kv 模式（本 runtime） | 先 `GET /v1/kv-session/info`：method、预算（32768）和各选项必须是 A.4 的。每个 case 一个新的 32 位十六进制 session id。第 0 轮：完整的渲染（`prefix_tokens` + `header_tokens`，`boundary_tokens` 为 0）。之后各轮：最新回复之后的尾部，做法是把对话中的这条回复换成一个标记，分别带和不带生成提示渲染：这条回复的结束分隔符、新消息和生成头（`header_tokens` 为 7）；`boundary_tokens` 是尾部第一个 `<|im_start|>` 之前的 token 数，若把上一条回复的最后一个输出 id 放回最前面（它不是 `<|im_end|>` 时，即 pad 或长度停止之后），再加 1。请求体还带有规范化后的消息列表、method `dram_backed_kv` 和服务端的预算。case 结束时 `POST /v1/kv-session/close`，遇到 409 每隔 2 秒重试，最多 120 秒 |
| vanilla 模式（stock） | 每一轮把完整对话重新渲染，以 token id 发给 `/v1/completions`（A.5）；每个 case 一个新的 `cache_salt`；不关闭 |
| 预热 | 第一个 case 之前，在单独的 session 里发一个 2 token 的请求，随即关闭；不属于任何 case 或指标 |
| 超时与重试 | 每个请求 3600 秒（当剩余的 agent 时间加一条回复的收尾时间更短时取后者，至少 30 秒）；不重试：被拒绝或失败的请求结束它所在的 case |
| 记录 | `cases.jsonl`：每个 case 一行（`case`、`split`、`source`、`metric`、`status`、`prediction`、`chunks_read`、`turns`、`agent_output_tokens` 即其各请求的 completion token 数之和、节点单调时钟上的 `started_ns` 和 `finished_ns`）；`requests.jsonl`：每个请求一行（不含消息列表的请求体、响应、发送和接收时间）；`runner.json` |

### A.8 限制

| 限制 | 取值 |
| --- | --- |
| 每个 case 的轮数 | 120 个请求（`turn_limit`） |
| 格式错误 | 连续两轮则结束该 case（`format_error`） |
| 回复上限 | 每个请求最多 4096 个新 token：客户端的 `max_tokens`（本 runtime：在 `--max-tokens 4096` 之内，并截到剩余上下文以内；stock：来自客户端） |
| 上下文上限 | 262,144 token（prompt 加回复）。本 runtime 拒绝没有为回复留出空间的 prompt（`CONTEXT_LIMIT`，A.3）；Vanilla 客户端在 prompt 加 4096 超过上限时结束该 case（A.5） |
| 客户端超时 | 每个请求 3600 秒 |
| 客户端重试 | 无 |
| 每个服务的 session | 同时最多打开 16 个（K）；服务生命周期内 64 个槽位，用掉 33 个；生命周期内的请求上限是 max(1,024, 256 × 64) = 16,384（README 4.9 节），而 case 的请求最多 32 × 120 = 3,840 个 |
| 锁页 Page arena | 124 GiB（本 runtime）；超出它的捕获会让该请求失败（`DRAM_ARENA_EXHAUSTED`，按 A.12 算作失败） |
| Agent 时间 | harness 的上限：最多 3,000 秒（`observation_query`）、4,200 秒（`goal_query`）和 6,000 秒（stock），再减去作业启动和关闭所需的时间。到达上限后不再发起请求，进行中的请求最多再完成一条回复，然后关闭所有打开的 session；这样被截断的 case 记为 `stopped_cap`，该次运行是不完整的运行（A.12） |

**为什么客户端超时设为 3600 秒、且不重试。** 一个请求在最多 4096 token 的回复开始之前，可能要等待它的 admission：在 `--admission-overlap 16` 下，前端最多等待一次 admission 2,100 秒（README 4.10 节），`goal_query` 的 admission 可能要等待唯一的 Goal rewriter 进程；放不下的一轮会等待容量（`CAPACITY_WAIT`，README 4.4 节）。3600 秒的超时让这类等待不会被打断。重试一个仍在进行中的请求，就是对一个忙碌的 session 发出第二个请求，协议会拒绝它。stock 对照使用相同的设置，使两边只在服务端不同。

### A.9 并发与 session 生命周期

- 16 个客户端 worker 按 A.1 的顺序领取 32 个 case，每个 worker 一次一个；一个 case 结束（其 session 关闭）后，它的 worker 领取下一个 case。最大的 case 先跑，这样 16 个 worker 能把各 case 排得紧凑（列表调度）。
- **本 runtime 上的 session。** 一个 case 的 session id 在第 0 轮请求时绑定到下一个未用的槽位（README 6.6 节）；最多 16 个同时打开。两轮之间，session 在 GPU 上保留常驻 Page，在主机上保留 carry 和已封存的 Page；`--release-idle-pages` 可以释放空闲 session 的常驻 Page，并在它的下一轮之前重新载入（A.4）。case 结束时客户端关闭 session，释放它在 GPU 和主机上的状态（README 4.4 节，session 池），然后下一个 case 才打开。
- **stock 对照。** 没有 session，也不关闭；每个 case 的 `cache_salt` 让一个 case 复用它自己的 prefix block、从不复用别的 case 的，stock vLLM 把这些 block 作为可淘汰的 prefix cache 保留。
- **case 之间不共享。** 8 个 Conflict Resolution case 共用一段历史，但每个 case 都在自己的 session 中读它；任何服务都不在 case 之间共享 KV。

### A.10 评测

| 项目 | 取值 |
| --- | --- |
| 评测程序 | `utils/eval_other_utils.py` 的官方 `post_process`（A.2），在运行之后离线执行，从不在 GPU 节点上执行 |
| 输入 | 每个 case 的最终 prediction（没有回答的 case 为空）、参考答案和来源子集 |
| 调用 | `metrics, parsed = post_process({"output": prediction}, answers, {"sub_dataset": source})`；该 case 的得分是 `metrics[metric]`，指标见 A.1 |
| 指标 | Test-Time Learning（ICL）：对解析出的答案计算 `exact_match`。Accurate Retrieval（RULER QA）和 Conflict Resolution（FactConsolidation）：`substring_exact_match`，取原始 prediction 和解析后 prediction 中较大的一个。Long-Range Understanding（InfBench 摘要）：`llm_judge_f1` |
| Long-Range Understanding | 官方评判 `llm_based_eval/summarization_evaluate.py`，使用 `gpt-4o-2024-05-13`（fluency 乘以关键点 recall 与句子 precision 的 F1）；它需要 OpenAI API key。我们的离线流程还没有运行它：这 8 个 case 和 Score 在它运行之前都处于待定状态 |
| 依赖 | 该评分文件导入 numpy、nltk、tiktoken、rouge_score（及 absl-py）和 editdistance。这四个子集用不到 tiktoken 和 editdistance，rouge_score 只增加 `rouge*` 字段，这里没有任何指标用到它们 |
| 没有回答的 case | 以 `format_error`、`turn_limit`、`unfinished_thinking` 或 `answer_length_limit` 结束的 case，按它留下的 prediction 评分（空 prediction 得 0 分） |

### A.11 指标

**Score。** 一个 case 的得分在 [0, 1] 之间（A.10）。一项能力的得分是其 8 个 case 的平均值 × 100；**Score** 是四项能力得分的等权平均，采用 0–100 分制。一个 case 是其能力的 12.5 分、Score 的 3.125 分。对 50 % 的二值指标，一次运行中一项能力的二项分布标准误约为 17.7 分，Score 的约为 8.8 分，两次独立运行的 Score 之差的约为 12.5 分。保留每个 case 的得分，以便逐个 case 比较各次运行。

**吞吐** = 输出 token 数 / 该次运行的时间跨度：

| 部分 | 定义 |
| --- | --- |
| 分子 | 32 个 case 的每个请求（读取轮和回答轮一视同仁）的 completion token 数之和（`cases.jsonl` 中的 `agent_output_tokens`），取服务端报告的值（本 runtime：响应中的 `completion_tokens`；stock：`usage.completion_tokens`） |
| 分母 | GPU 节点单调时钟上，从第一个 case 开始到最后一个 case 结束（在其 session 关闭之后）的时间跨度：`cases.jsonl` 上的 `max(finished_ns) − min(started_ns)` |
| 包含 | 这两个时刻之间的一切：admission 及其等待、容量等待、释放和重新载入、Goal rewriter 的时间（`goal_query`）、客户端的渲染、关闭 session |
| 不包含 | 服务的启动和关闭、预热请求、评分，以及 Goal rewriter 自己的 token（辅助模型） |

每次运行还要记录：每个 case 的状态、读取的块数、轮数和墙钟时间；每个服务的请求数、上下文超限拒绝次数、释放和重新载入次数（`capacity.jsonl`）、`CAPACITY_WAIT` 次数（本 runtime），以及抢占计数（stock，A.5）。

### A.12 运行规则

- **每个作业一个服务**，一直运行到结束；客户端在同一个作业里运行于 GPU 节点上，与服务共用该作业的 16 个 CPU。要比较的各个服务使用同一类节点和同样的作业规模。
- **每次尝试都用全新的服务和新的输出目录。** `serve.py` 拒绝已存在的输出目录。
- **中断的运行不续跑。** 提前结束的运行（作业时间上限、agent 时间上限、服务失败）是不完整的运行：未完成的 case 不在另一个作业里接着跑，该次运行不纳入对比，也不给出 Score 和吞吐，整个服务换一个新标签从头重跑。
- **失败的计法。** 有 case 以 `server_error`、`transport_error`、`context_limit`、`stopped_cap` 或 `error` 结束，该次运行就是不完整的运行。在本 runtime 上，以下情况还会让服务的检查不通过：某一轮不是 token 级精确（服务端的 prompt 与客户端的构造逐 token 相同），某个请求在 `failures.jsonl` 中有记录，发生了任何抢占，释放的不是空闲 session，或者重新载入没有恢复它对应的释放，或者关闭不干净（README 第 8 节）。
- **容量等待**（`CAPACITY_WAIT`）是正常的；一直持续到 agent 时间上限的停滞会使该次运行成为不完整的运行。
- **传输方式。** 客户端通过节点的 loopback 访问服务；服务从不在 loopback 之外监听（README 6.4 节）。
- **内存位置。** 在两份份额的作业中，锁页 arena 可能跨两个 Grace socket；harness 把作业的 NUMA 位置记为一项发现，不算失败。

### A.13 对齐检查表

| # | 项目 | 本 runtime | baseline 使用 |
| --- | --- | --- | --- |
| 1 | 模型与 revision | Qwen3.6-27B `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | 相同 |
| 2 | 软件 | vLLM 0.24.0 官方 wheel、Transformers 5.13.0、Tokenizers 0.22.2、`constraints-3270721.txt` | 同一个环境 |
| 3 | 硬件与作业 | 1 × GH200 120GB、16 个 CPU、227,120 MiB | 1 × GH200 120GB、16 个 CPU、其自身锁页内存所需的主机内存（stock：113,560 MiB） |
| 4 | Case | A.1 的 32 个 case 及其上下文 SHA-256，按该顺序 | 相同 |
| 5 | 并发 | 最多 16 个同时运行，按顺序补入 | 相同 |
| 6 | 上下文上限 | 262144 | `--max-model-len 262144` |
| 7 | 每步 batched token 数 | 16384 | `--max-num-batched-tokens 16384` |
| 8 | 最大序列数 | 16 | `--max-num-seqs 16` |
| 9 | GPU KV | 32 GiB（34,359,738,368 字节） | `--kv-cache-memory-bytes 34359738368` |
| 10 | 回复上限 | 4096 | `max_tokens` 4096 |
| 11 | 解码 | 贪心（服务端），seed 42 | `temperature` 0、`top_p` 1、`top_k` −1、`repetition_penalty` 1、`seed` 42 |
| 12 | Thinking | 关闭（`--kv-session-thinking off`，7 个 token 的生成头） | 以 `enable_thinking=False` 渲染 |
| 13 | 停止 | 只在 EOS/pad | 不设停止字符串 |
| 14 | CUDA graph | `FULL_AND_PIECEWISE`（`--cuda-graphs full`） | vLLM 默认（`FULL_AND_PIECEWISE`） |
| 15 | 异步调度 | 关闭（总是） | vLLM 默认（开启）；已知差异（A.14） |
| 16 | 注意力 | FLASH_ATTN 加 FlashAttention 3，由 vLLM 选择 | 相同，不加 `--attention-backend` |
| 17 | Gated-DeltaNet prefill | Triton | `--gdn-prefill-backend triton` |
| 18 | Sampler | 原生（`VLLM_USE_FLASHINFER_SAMPLER=0`） | 同一个变量 |
| 19 | Prefix caching | 开启，sha256 | `--enable-prefix-caching --prefix-caching-hash-algo sha256`，每个 case 一个 `cache_salt` |
| 20 | Mamba cache 模式 | `align` | `--mamba-cache-mode align` |
| 21 | 纯文本 | `--language-model-only` | vLLM 的 `--language-model-only` |
| 22 | 客户端 | harness 的 MemAgent 循环（A.7），用 Transformers 5.13.0 和模板 `e84f32a2…` 渲染 | 相同的循环和模板 |
| 23 | 循环限制 | 120 轮，连续 2 次格式错误，每条回复 4096 | 相同 |
| 24 | 客户端超时与重试 | 3600 秒，不重试 | 相同 |
| 25 | 每个 session 的预算 | 32,768 token（`observation_query`、`goal_query`） | 所有依赖预算的方法都用同一个预算；精确注意力不设预算 |
| 26 | 评测 | `538026089d…` 的官方 `post_process`，A.10 的指标；摘要用官方 GPT-4o 评判 | 相同 |
| 27 | Score | 四项能力得分的等权平均，0–100 | 相同 |
| 28 | 吞吐 | A.11 的定义 | 相同的分子和计时 |
| 29 | 不完整的运行 | 不纳入，不续跑 | 相同 |

### A.14 已知差异

即使检查表的每一项都一致，以下差异仍然存在。任何对比都要一并报告。

**与官方 MemoryAgentBench harness 相比：**

1. **Agent 循环。** 官方 harness 把一段历史的各块交给它的记忆 agent，然后提出这段历史的所有问题（“注入一次，多次提问”）。这里的 harness 给每个问题一个单独的 session，先通过 `read_next` 工具调用读完整段历史，然后才得到问题。8 个 Conflict Resolution 问题共用一段历史，这段历史被读了 8 遍。
2. **分块。** 各块随 case 数据给定（每块最多 4,096 个 Qwen3.6 token）。官方 harness 用自己的按句分块器以 `chunk_size` 4096 切分历史，token 用 tiktoken 计数（`configs/data_conf`，`utils/eval_other_utils.py` 中的 `chunk_text_into_sentences`），所以它的分块边界可能不同。
3. **回复长度。** 这里每一轮都可以生成 4096 token；官方的数据配置给每个子集各自的 `generation_max_length`（例如 `Factconsolidation_sh_32k.yaml` 中为 10，`ICL_trec_coarse.yaml` 中为 20，`InfBench_sum.yaml` 中为 1,200）。
4. **Thinking。** 这里关闭。harness 自己的配置开启 thinking；开启 thinking 的结果是另一种配置，不能直接比较。
5. **规模。** 每项能力 8 个问题，只跑一次。官方配置每个子集提出的问题更多。
6. **评分。** 官方 `post_process` 离线作用于每个 case 的最终回答；摘要需要官方的 GPT-4o 评判（A.10）。

**本 runtime 与 stock vLLM 之间：**

1. **历史。** 本 runtime 把每条回复保留为它生成时的原始 id，前面是关闭 thinking 的生成头放进去的空思考块。Vanilla 对照每一轮从文本重新渲染完整对话：模板会去掉每条较早回复的空思考块（每条 4 个 token，与 README 6.7 节对 chat-text 对照的描述相同），并重新编码回复文本，所以两边的 prompt 从第一条较早的回复处开始不同。
2. **异步调度。** 本 runtime 不使用（launcher 总是传 `--no-async-scheduling`）；stock vLLM 使用其默认值，即开启。
3. **采样由谁控制。** 本 runtime 拒绝非贪心的采样字段并贪心解码；stock vLLM 从请求中读取采样设置（`temperature` 0 即贪心解码）。
4. **上下文边界。** 本 runtime 只在没有为回复留出任何空间时才拒绝 prompt，并把 `max_tokens` 截到剩余空间以内（A.3）；Vanilla 客户端在 prompt 加 4096 超过 262,144 token 时自己结束该 case。在预期回复长度下，没有 case 会到达这两个边界（A.4）。
5. **KV 管理。** 两者的 GPU KV 都是 32 GiB，但布局不同。本 runtime 使用 16 token 的注意力 Page、单独的 Gated-DeltaNet 池、1 GiB 的 CPU 池和 124 GiB 的锁页 Page arena；它从不抢占，放不下的一轮会等待（`CAPACITY_WAIT`），GPU 紧张时空闲 session 的 Page 会被释放并重新载入（A.4）。stock vLLM 使用它自己的混合布局（README 4.6 节），只把已结束请求的 block 作为可淘汰的 prefix cache 保留，淘汰掉的部分要重新 prefill，池满时可能抢占。32 GiB 中有 16 个长 session 时，一个 session 的 prefix 可能在它的下一轮之前就被淘汰。
6. **admission 与调度。** 本 runtime 按到达顺序一次准入一个请求（`goal_query` 在 `--admission-overlap 16` 下最多 16 个同时进行），每次 admission 的阶段 1 是单独的一次 forward，然后成批 decode（README 4.4、4.10 节和第 5 节）。stock vLLM 在每步 16,384 token 的预算内把 prefill 分块与 decode 步一起调度。
7. **Session 生命周期。** 本 runtime 在客户端关闭 session 时释放该 case 的状态；stock vLLM 把缓存的 prefix 保留到它自行淘汰为止。
8. **选择性策略。** 一个 session 已封存的历史超过 32,768 token 的预算后，`observation_query` 和 `goal_query` 只关注选中的一部分历史 Page（A.1）。
9. **Goal rewriter。** `goal_query` 在同一张 GPU 上运行一个辅助的 Qwen3-0.6B。它的时间计入吞吐，它的 token 不计入。
10. **路由。** 本 runtime 以追加的尾部服务 `/v1/kv-session/complete`；stock vLLM 以完整的 prompt 服务 `/v1/completions`。
11. **两次运行之间的差异。** 两种服务的成批运行在两次运行之间都不是 token 完全相同的：回复可能随 batch 的组成而变，而 batch 的组成取决于时序（README 4.4 节）。G1 的 token 级一致（见“验证状态”）只对一个 eager、同步调度、回放相同 prompt 的 session 成立，不适用于本配置的成批运行。

---

## Part B. 流程

这一部分用本仓库的工具运行 Part A 的配置：一个服务和客户端在 GH200 节点上的同一个 Slurm 作业里，客户端走节点的 loopback。除标注外，每一步都同时适用于两个策略和 stock 对照。它需要 case 数据（A.1）和一个实现 A.7 的客户端；两者都不在本仓库中。

### B.1 输入

- **模型、环境和 release。** 按 README 2.1 到 2.4 节为该 runtime commit（A.2）构建 release；`goal_query` 还需要 Qwen3-0.6B（README 2.2 节）。
- **Case。** A.1 的 32 个 case，向 harness 的维护者获取。把 A.1 的 32 个 id 按顺序存为 `$CASES/cases.txt`，并在任何运行之前逐个检查：块数和历史 token 数（固定 tokenizer）与 A.1 相同，每块最多 4,096 token，各块依次拼接后的 SHA-256 等于其上下文 SHA-256。不一致的 case 不属于本配置。
- **客户端。** 一个实现 A.7 的可执行程序 `$CLIENT`：`$CLIENT <kv|vanilla> <base url> <case directory> <output directory>`。它按 `cases.txt` 的顺序、用 16 个 worker 对该 base URL 运行各 case，并写出带有 A.7 所列字段的 `<output directory>/cases.jsonl`（B.6 和 B.7 读取它们）。我们的 runner 发送的就是这些请求、写出的就是这些记录，但它接受的是一份作业说明而不是这些参数，并且不在本仓库中；harness 自己的客户端发送的是相同的请求。
- **评分程序。** `HUST-AI-HYZ/MemoryAgentBench` 在 commit `538026089d1a8a8eff05121d0db89b388f360eba` 的 checkout，两个文件的哈希与 A.2 相同，以及一个装有 A.10 所列依赖的 Python 环境。B.6 还需要一个答案文件：一个从 case id 映射到该 case 参考答案列表的 JSON 对象，由 case 数据生成。

### B.2 占用 GPU 之前的检查

**模板与生成头**（设置 A.4 的环境变量后；不需要 GPU；已在 GPU 作业之外用固定的 tokenizer 文件和 Transformers 5.13.0 运行过）：

```sh
"$VENV/bin/python" -I -B - "$MODELS/qwen3.6-27b" <<'PY'
import hashlib, sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
template = tokenizer.chat_template
template = template[:-1] if template.endswith('\n') else template
print(hashlib.sha256(template.encode('utf-8')).hexdigest())
probe = [{'role': 'user', 'content': 'x'}]
without = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=False, enable_thinking=False)
header = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=True, enable_thinking=False)
print(repr(header[len(without):]), len(tokenizer.encode(header[len(without):], add_special_tokens=False)))
PY
```

第一行必须是 `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`，第二行必须是 `'<|im_start|>assistant\n<think>\n\n</think>\n\n' 7`。

**Dry run。** A.4 的 dry run（不加 `--start`）必须打印 A.4 的 vLLM 参数；锁页主机内存或注意力池放不下的配置会被 `serve.py` 拒绝（README 4.9 节）。

**评分程序。** 在该 checkout 中执行 `sha256sum utils/eval_other_utils.py llm_based_eval/summarization_evaluate.py`，必须打印 A.2 的两个哈希。

### B.3 Slurm 作业脚本

下面的脚本在一个 Slurm 作业里运行正式测试的一个服务：服务、READY、kernel 日志行、节点上的客户端，以及关闭。它由 `examples/slurm/serve_gh200.sh` 改写而来，后者不传 kv-session 参数、`--cuda-graphs`、`--active-token-budget` 和 A.4 的各个开关。这只是示例，**本身不在已验证的运行范围内，且未经测试**。存为 `mab_job.sh`：

```bash
#!/usr/bin/env bash
# One server of the MemoryAgentBench benchmark (docs/MEMORYAGENTBENCH_BENCHMARK.md) in one Slurm job.
# POLICY: observation_query | goal_query (this runtime) | stock (stock vLLM 0.24.0, the Vanilla control).
set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}" || { echo "run this under sbatch" >&2; exit 2; }
: "${REPO:?}" "${VENV:?}" "${POLICY:?}" "${RUN_ROOT:?}" "${AGENT_SECONDS:?}" "${CLIENT:?}" "${CASES:?}"
PORT=${PORT:-8000}; READY_SECONDS=${READY_SECONDS:-900}

# Job-local directories and the environment of README section 3.
STAGE="$RUN_ROOT/$SLURM_JOB_ID"
mkdir "$STAGE" "$STAGE/tmp" "$STAGE/home" "$STAGE/cache"
(( ${#STAGE} + 4 + 37 <= 107 )) || { echo "TMPDIR too long for AF_UNIX: $STAGE/tmp" >&2; exit 2; }
export TMPDIR="$STAGE/tmp" TMP="$STAGE/tmp" TEMP="$STAGE/tmp"
export HOME="$STAGE/home" XDG_CACHE_HOME="$STAGE/cache"
export HF_HOME="$STAGE/cache/hf" TORCH_HOME="$STAGE/cache/torch"
export TRITON_CACHE_DIR="$STAGE/cache/triton" CUDA_CACHE_PATH="$STAGE/cache/cuda"
export TORCH_EXTENSIONS_DIR="$STAGE/cache/torch-extensions"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
unset CC VLLM_RPC_BASE_PATH SSH_AUTH_SOCK SSH_AGENT_PID PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_PRELOAD
export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
PY="$VENV/bin/python"

# The server: Part A.4 (this runtime) or Part A.5 (stock).
SHARED=(--protocol kv-session --kv-session-thinking off --max-running-sessions 16 --kv-cache-gib 32
        --page-tokens 16 --kv-layout decoupled --max-model-len 262144 --max-tokens 4096 --dram-arena-gib 124
        --connector-pool-gib 1 --cuda-graphs full --decode-fastpath --language-model-only --release-idle-pages)
case $POLICY in
  observation_query) POLICY_FLAGS=(--active-token-budget 32768); MODE=kv ;;
  goal_query)        POLICY_FLAGS=(--active-token-budget 32768 --goal-rewriter-model "${REWRITER:?}"
                                   --admission-overlap 16); MODE=kv ;;
  stock)             POLICY_FLAGS=(); MODE=vanilla ;;
  *) echo "unknown POLICY: $POLICY" >&2; exit 2 ;;
esac
OUTPUT="$STAGE/$POLICY"
if [[ $POLICY == stock ]]; then
  : "${MODEL:?}"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
  "$PY" -I -B -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name selective-prefix --host 127.0.0.1 --port "$PORT" \
    --distributed-executor-backend uni --tensor-parallel-size 1 \
    --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 16384 \
    --kv-cache-memory-bytes 34359738368 --seed 42 \
    --enable-prefix-caching --prefix-caching-hash-algo sha256 \
    --mamba-cache-mode align --gdn-prefill-backend triton --language-model-only > "$STAGE/server.log" 2>&1 &
else
  : "${RELEASE:?}"
  SERVE=("$PY" -B "$REPO/serve.py" --release-dir "$RELEASE" --policy "$POLICY"
         --output "$OUTPUT" --port "$PORT" "${SHARED[@]}" ${POLICY_FLAGS[@]+"${POLICY_FLAGS[@]}"})
  "${SERVE[@]}" > "$STAGE/dry-run.json"          # dry run: checks the release, prints the vLLM command
  "${SERVE[@]}" --start > "$STAGE/server.log" 2>&1 &
fi
SERVER=$!                                         # the APIServer PID (serve.py execs it)

# READY: ready.json (this runtime, README section 4.1) or /health and /v1/models (stock).
ready() {
  if [[ $POLICY == stock ]]; then
    local models
    curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" > /dev/null || return 1
    models=$(curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models") || return 1
    [[ $models == *'"selective-prefix"'* ]] || return 1
    return 0
  fi
  "$PY" -B "$REPO/examples/slurm/server_checks.py" ready "$OUTPUT" "$POLICY" "$RELEASE" > "$STAGE/ready-check.txt"
}
deadline=$((SECONDS + READY_SECONDS))
while true; do
  set +e; ready; rc=$?; set -e
  (( rc == 0 )) && break
  (( rc == 1 )) || { cat "$STAGE/ready-check.txt" >&2; kill -TERM "$SERVER"; wait "$SERVER" || true; exit 1; }
  kill -0 "$SERVER" 2>/dev/null || { echo "server exited before READY" >&2; wait "$SERVER" || true; exit 1; }
  (( SECONDS < deadline )) || { echo "no READY within $READY_SECONDS s" >&2; kill -TERM "$SERVER"; wait "$SERVER" || true; exit 1; }
  sleep 5
done
echo "READY after $SECONDS s"
grep -oE "Using FLASH_ATTN attention backend|Using FlashAttention version [0-9]+|Using Triton/FLA GDN prefill kernel \(requested=[a-z]+|FlashInfer top-p/top-k sampling disabled|Asynchronous scheduling is [a-z]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|'cudagraph_capture_sizes': \[[0-9, ]*\]|running in text-only mode|Encoder cache will be initialized|Multi-modal warmup completed" \
  "$STAGE/server.log" | sort | uniq -c > "$STAGE/kernels.txt" || true

# The client (Part A.7 to A.9): the 32 cases, 16 at a time, on this node's loopback.
client_rc=0
timeout "$AGENT_SECONDS" "$CLIENT" "$MODE" "http://127.0.0.1:$PORT" "$CASES" "$STAGE/client" \
  > "$STAGE/client.log" 2>&1 || client_rc=$?
if (( client_rc == 124 )); then echo "agent time limit reached: the run is partial (B.5)" >&2; fi

# Stop: SIGTERM to the server PID only; this runtime must exit 0 with clean receipts (README section 8).
kill -TERM "$SERVER"
set +e; wait "$SERVER"; server_rc=$?; set -e
echo "server exit $server_rc, client exit $client_rc"
if [[ $POLICY != stock ]]; then
  (( server_rc == 0 )) || exit 1
  "$PY" -B "$REPO/examples/slurm/server_checks.py" cleanup "$OUTPUT"
  "$PY" -I -B -c 'import json,sys; v=json.load(open(sys.argv[1])); sys.exit((v.get("session_pool") or {}).get("open_at_shutdown") != [])' \
    "$OUTPUT/engine-cleanup.json" || { echo "sessions open at shutdown" >&2; exit 1; }
fi
exit "$client_rc"
```

每个服务提交一个作业，使用 README 快速上手中的占位符，设置放在提交作业的 shell 里（`--export=ALL`）。下面的例子提交 `observation_query`：

```sh
POLICY=observation_query
export REPO VENV RELEASE POLICY
export MODEL="$MODELS/qwen3.6-27b" REWRITER="$MODELS/Qwen3-0.6B-c1899de"
export RUN_ROOT=/abs/short/jobs AGENT_SECONDS=1800
export CLIENT=/abs/path/to/mab-client CASES=/abs/path/to/cases
sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus=1 \
  --cpus-per-task=16 --mem=227120M --time=00:47:00 --job-name="mab-$POLICY" \
  --output="$RUN_ROOT/slurm-%j.out" --export=ALL --mail-type=BEGIN,END,FAIL --mail-user="$MAIL" \
  /abs/path/to/mab_job.sh
```

- **`goal_query`：** `POLICY=goal_query`，`--time=01:01:00`，例如 `AGENT_SECONDS=2400`。
- **stock 对照：** `POLICY=stock`，`--mem=113560M`（一份份额：它不钉住 arena），`--time=01:27:00`，例如 `AGENT_SECONDS=3900`。
- **时间上限。** `--time` 要覆盖服务启动（已验证的作业中，作业里第一个服务在启动后 204–231 秒 READY，README 4.1 节；冷缓存下 torch.compile 再多约一分钟，README 4.8 节）、客户端（`AGENT_SECONDS`）和关闭。harness 的 G4 作业使用这些时间上限，约为其预计墙钟时间的 1.3 倍：2,164 秒（`observation_query`）、2,804 秒（`goal_query`）和 4,024 秒（stock）；上面的 `AGENT_SECONDS` 是这些上限之内的示例取值。预计时间基于 harness 规划用的回复长度（A.4）。被 `AGENT_SECONDS` 或作业时间上限截断的运行是不完整的运行（A.12）。
- **客户端自己的时间上限。** `timeout` 是强制停止。客户端应当自己稍早一些停止发起新请求，让进行中的请求完成并关闭它的 session（我们的 runner 就是这样做的，A.8），这样关闭时的检查不会看到打开的 session。
- `RUN_ROOT` 必须短：`TMPDIR` 在它下面。

### B.4 就绪检查

- **本 runtime。** READY 指 `server_checks.py ready` 返回 0（README 4.1 节）；`/v1/models` 有响应并不等于 READY。`dry-run.json` 中是 vLLM 命令，必须与 A.4 一致。`GET /v1/kv-session/info` 返回 method、预算和各选项（策略、页大小、预算方式和选页合同、回复上限、上下文上限、thinking、session 数）；客户端在第一个 case 之前检查它们（A.7）。**不要在运行之外发送 complete 请求：** 它的 session id 会占用服务 64 个槽位中的一个，打开一个不属于任何 case 的 session，之后的检查都会把它算进去。
- **stock 对照。** `GET /health` 返回 200，且 `GET /v1/models` 列出 `selective-prefix`。
- **kernel 日志行。** 作业目录中的 `kernels.txt` 必须显示 A.6 的各行。

### B.5 运行

- 三个服务依次运行，每个一个作业。
- 进度：每完成一个 case，`$STAGE/client/cases.jsonl` 多一行（使用我们的 runner 时）。在本 runtime 上，服务输出目录中的 `capacity.jsonl` 列出空闲 session 的 Page 的释放和重新载入，`server.log` 中的 `CAPACITY_WAIT` 行表示有轮次在等待空间；`GET /v1/kv-session/info` 随时可以读取。
- **被截断的运行。** 这个脚本没有停滞看门狗：停滞的运行会一直持续到 `AGENT_SECONDS` 或作业时间上限。以任一种方式被截断的运行都是不完整的运行；整个服务在一个新作业里重跑（A.12）。
- 客户端的记录和服务的输出目录包含来自 case 数据的 prompt 和历史文本，不要放进公开发布。

### B.6 评测

运行之后离线进行，使用 B.1 的评分程序（未经测试）。答案文件把每个 case id 映射到其参考答案；可选的评判文件把每个 Long-Range Understanding case id 映射到官方评判给出的 `llm_judge_f1`：

```sh
python3 - "$SCORER" "$STAGE/client/cases.jsonl" "$ANSWERS" ${JUDGE:+"$JUDGE"} <<'PY'
import hashlib, json, sys
from pathlib import Path
scorer, cases, answers = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
judge = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else {}
source = scorer / 'utils' / 'eval_other_utils.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == \
    'd23a5ba9f2750229315cd308463ae99aef9ff36ed0859b69d4851f804976b815', 'not the pinned scorer'
sys.path.insert(0, str(source.parent))
import eval_other_utils as official
rows = [json.loads(line) for line in cases.read_text().splitlines() if line.strip()]
complete = ('answered', 'answer_length_limit', 'unfinished_thinking', 'format_error', 'turn_limit')
abilities = {}
for row in rows:
    if row['metric'] == 'llm_judge_f1':
        score = judge.get(row['case'])
    else:
        metrics, _ = official.post_process({'output': row['prediction'] or ''}, answers[row['case']],
                                           {'sub_dataset': row['source']})
        score = float(metrics[row['metric']])
    abilities.setdefault(row['split'], []).append(score)
    print(json.dumps(dict(case=row['case'], status=row['status'], score=score)))
partial = len(rows) != 32 or any(row['status'] not in complete for row in rows)
means = {k: (100 * sum(v) / 8 if len(v) == 8 and None not in v else None) for k, v in sorted(abilities.items())}
score = sum(means.values()) / 4 if len(means) == 4 and None not in means.values() and not partial else None
print(json.dumps(dict(cases=len(rows), partial=partial, abilities=means, score=score)))
PY
```

最后一行给出四项能力的得分和 Score（A.11）；一项能力缺少 8 个得分中的任何一个，或运行不完整，就不给出得分。评判需要 OpenAI API key，用官方的 `llm_based_eval/summarization_evaluate.py` 运行；没有它，Long-Range Understanding 的得分和 Score 都处于待定状态。

### B.7 吞吐

从一次运行的 `cases.jsonl` 计算 A.11 的吞吐（未经测试）：

```sh
python3 - "$STAGE/client/cases.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding='utf-8') if line.strip()]
tokens = sum(row['agent_output_tokens'] for row in rows)
span = (max(row['finished_ns'] for row in rows) - min(row['started_ns'] for row in rows)) / 1e9
print(json.dumps(dict(cases=len(rows), output_tokens=tokens, span_s=round(span, 3), tokens_per_s=round(tokens / span, 3))))
PY
```

完整的运行有 32 行；少于此数的运行是不完整的运行（A.12）。
