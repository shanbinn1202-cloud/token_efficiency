# MemoryAgentBench 正式测试配置

[English](MEMORYAGENTBENCH_BENCHMARK.md) | 简体中文

本文档规定本 runtime 的 MemoryAgentBench 正式测试的配置：MemoryAgentBench 数据中的 32 个 case，四种能力各 8 个，每个 case 是一段长历史，agent 通过 `read_next` 工具逐块读完之后回答一个问题；Qwen3.6-27B，关闭 thinking；一个服务上最多同时运行 16 个 case，使用 kv-session 协议（`--protocol kv-session`，[docs/running.zh-CN.md 6.6 节](running.zh-CN.md#66-memoryagentbench-kv-session-协议)）；`observation_query` 和 `goal_query` 每个策略一个服务；另有一个经由客户端的 Vanilla transport 访问的 stock vLLM 0.24.0 服务作为对照。**Part A** 是正式记录的配置。baseline（stock vLLM 对照或其他任何系统）只有满足 A.13 检查表的每一项才可比。**Part B** 是用本仓库工具执行的手动流程，这些工具包括 `examples/memoryagentbench/` 的客户端和 case 构建程序，属于可选内容；它需要两样来自本仓库之外的输入：你自己下载的公开数据集 `cxkjntmngm/MemAgent_Bench`（由 `cases.py build` 从中生成 case，A.1），以及官方 MemoryAgentBench 仓库的一个 checkout（A.2）。

**带日期的正式记录配置。** Part A 是 2026-10-18 那批 32 个 case 的 Qwen 运行（G4，runtime `f9322a6`）的配置，作为这批运行的记录保留，命令保持原样。之后的运行使用 [docs/status.zh-CN.md](status.zh-CN.md#当前配置) 中的当前配置，它与 Part A 的不同在于下列设置（262,144 token 的上下文、4,096 token 的回复上限和 16 个 session 相同）：

| 设置 | Part A（记录） | 当前配置 |
| --- | --- | --- |
| Thinking | 关闭（`--kv-session-thinking off`） | 开启 |
| 调度 | 同步（launcher 的 `--no-async-scheduling`） | 异步（`--async-scheduling`） |
| GPU KV | 32 GiB | 25 GiB |
| 每个 session 的预算 | 32,768 token | 8,192 token |
| 解码开关 | 只开 `--decode-fastpath`（`--no-decode-fastpath2 --no-deferred-completion-io`） | 速度默认选项 |

开启 thinking 的结果不能与 Part A 关闭 thinking 的运行直接比较（[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 4.3）。

Gemma 4 31B IT profile（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）以 Gemma 的轮次格式、关闭 thinking 的方式服务 kv-session 协议（[docs/running.zh-CN.md 6.6 节](running.zh-CN.md#66-memoryagentbench-kv-session-协议)）。A.15 给出它最初几次 MemoryAgentBench 运行的设置：`observation_query` 和 `goal_query`，用 32 个 case 中的 8 个（作业 3429503 和 3429507）；之后的运行及其设置见 [docs/status.zh-CN.md](status.zh-CN.md#gemma-4-31b-it)。本文档其余部分都是 Qwen3.6-27B 的配置。

节号引用都写明该节所在的页面，例如 [docs/running.zh-CN.md 6.6 节](running.zh-CN.md#66-memoryagentbench-kv-session-协议)；README 的[文档地图](../README.zh-CN.md#文档地图)列出了这些页面。本页出现三种 harness。“作业 harness”指本项目运行 GPU 作业所用的 harness（作业的计划、检查、时间上限、预热和记录），它不公开。“read_next harness”指 agent 循环及其提示词，以及 MemAgent32 的 case 选择和分块，是 Shanbin Sun 的设计，自 integrate-1107 起公开在 `examples/memoryagentbench/` 中（见下）。“官方 harness”指公开的 MemoryAgentBench 仓库（A.2）。

> **本仓库中的客户端（integrate-1107）。** A.7 的 runner、它的 case 和评分脚本现在是 `examples/memoryagentbench/`（`run_cases.py`、`kv_client.py`、`cases.py`、`score.py`），连同它的页面 [benchmarks/memoryagentbench.zh-CN.md](benchmarks/memoryagentbench.zh-CN.md)（准备、步骤、服务命令）和测试 `test_mab_client.py`（仅 CPU）一起从发布仓库同步而来。read_next harness（agent 循环及其提示词，以及 MemAgent32 的 case 选择和切块）由 Shanbin Sun 设计；`examples/memoryagentbench/` 按本项目的运行方式收录了它。它取自 MemoryAgentBench 和 mini-swe-agent 的字符串的 MIT 声明见 [THIRD_PARTY_NOTICES.txt](../THIRD_PARTY_NOTICES.txt)。

## 验证状态

本正式测试的 GPU 作业（Qwen3.6-27B 和 Gemma 4 31B IT），连同每个作业的 runtime commit、结论、审计标记和日期，见 [docs/status.zh-CN.md](status.zh-CN.md)（其中的 MemoryAgentBench 各行）；到 runtime `cf2b28d` 为止的记录冻结在 [docs/history/validation-cf2b28d.zh-CN.md](history/validation-cf2b28d.zh-CN.md)，本节原来对这些运行的叙述（截至 2026-11-16）冻结在 [docs/history/memoryagentbench-validation.zh-CN.md](history/memoryagentbench-validation.zh-CN.md)。这些运行只报告结论，不报告分数：评分推迟（[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 5.1）。Part B 的脚本是示例，未按所写的样子测试过（[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 5.4）；`examples/memoryagentbench/` 的 `run_cases.py`、`cases.py` 和 `score.py` 有 CPU 测试（`tests/test_mab_client.py`）。

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
| 吞吐 | 32 个 case 的每个请求（读取轮和回答轮一样）由服务端报告的 completion token 之和 / GPU 节点时钟上从第一个 case 开始到最后一个 case 结束（关闭其 session 之后）的时间跨度。包含 admission 及其等待、容量等待、释放和重新载入、Goal rewriter 的时间、客户端的渲染和 session 关闭（`read_next` 由 runner 自己应答，A.7）；不包含服务的启动和关闭、作业 harness 的预热请求、评分以及 Goal rewriter 自己的 token（A.11） |
| 任务 | 32 个 MemoryAgentBench case，每种能力 8 个（Accurate Retrieval、Test-Time Learning、Long-Range Understanding、Conflict Resolution），每个 case 一个问题（MemAgent32），块数、历史 token 数和上下文 SHA-256 见 A.1（GPU 运行所用 manifest 的 SHA-256 为 `010fd109…`）；read_next 循环，每个 case 最多 120 轮（A.1、A.7） |
| 评测 | `utils/eval_other_utils.py`（`538026089d…`）中官方的 `post_process`，运行结束后离线执行；Long-Range Understanding：官方的 GPT-4o 评判（`gpt-4o-2024-05-13`），没有运行：评分推迟（A.10，[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 5.1） |
| Gemma 4 31B IT | A.15：32 个 case 中的 8 个（每个来源的前两个），8 个 session，25 GiB GPU KV（`--kv-cache-gib 25`），上下文、预算、回复上限、thinking 和解码与上面相同，CUDA graph `full`、三个解码快速路径、`--deferred-completion-io`、`TRITON_ATTN`，`goal_query` 另加 `--admission-overlap 8 --goal-rewriter-batch 4`。不是 32 个 case 的配置 |

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

**数据。** 这些 case 是 MemAgent32，即 read_next harness 选出的 32 个 case：公开的 Hugging Face 数据集 `cxkjntmngm/MemAgent_Bench` 在 revision `a785de0f8d2511a43b9e524b5326e1105a4d218a` 上的若干行（不是 `ai-hyz/MemoryAgentBench` 的），由 `examples/memoryagentbench/cases.py` 切块（`cases.py build`，B.1）。该数据集没有声明许可证，本仓库不包含它的任何数据。每个 case 带有它的 system prompt、它的各块、套入 MemoryAgentBench 提问模板的问题、参考答案、来源子集和指标。一个 case 的上下文 SHA-256 是其各块依次拼接后的 SHA-256。GPU 运行使用的是作业 harness 的 case 副本，即一个 manifest（SHA-256 `010fd10971905e7611ff639c81230f12bad58acd688143cf7b6eb94e30151edf`）加每个 case 一份上下文，作业 harness 在作业开始时按下面的列表逐个 case 核对。`cases.py build` 写出它自己的 `manifest.json`，没有任何地方固定它；`cases.py verify` 只把块数和历史 token 数与 MemAgent32 比较，上下文 SHA-256 要你自己与下面的列表比较（B.1）。

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

本 runtime 在 MemoryAgentBench 上的策略（README 开头的策略表和 [docs/design.zh-CN.md 第 5 节](design.zh-CN.md#5-query-策略的选页规则)）：

| `--policy` | 下一轮使用哪些历史 Page |
| --- | --- |
| `observation_query` | Selector V2：在预算内保留必保留页、recent 页和余弦最高的页；query 是最新 observation 末尾的 hidden state（新的 tool 消息及其块；最后一次读取时，后面还跟着问题） |
| `goal_query` | 同一个 selector；query 是持续维护的检索 Goal，由 Qwen3-0.6B rewriter 更新。在本协议上，Goal 的 observation 是新的 tool 和 user 文本，显式要求是问题（[docs/running.zh-CN.md 6.6 节](running.zh-CN.md#66-memoryagentbench-kv-session-协议)） |
| `full` | 所有历史 Page。不属于 G4 运行：stock Vanilla 对照是精确注意力的参照。它在 G1 中跑过本协议（作业 3397844，[docs/status.zh-CN.md](status.zh-CN.md)）；A.4 列出它的参数 |
| `random_fill` | 两个 query 策略的对照：Selector V2 的必保留页和 recent 页，预算的其余部分按带种子的随机顺序填充，不用 query（[docs/design.zh-CN.md 第 5 节](design.zh-CN.md#5-query-策略的选页规则)）。它的服务见正式测试页面（[benchmarks/memoryagentbench.zh-CN.md](benchmarks/memoryagentbench.zh-CN.md#服务命令)）；不属于 G4 运行 |

在 `--active-token-budget 32768` 下，容量是 2,048 个 16 token 的 Page。一个 session 已封存的历史超过 8,192 token 后才开始选择（[docs/design.zh-CN.md 第 5 节](design.zh-CN.md#5-query-策略的选页规则)），但在预算以内所有 Page 都放得下，所以 session 在已封存历史超过 32,768 token 之前关注它的全部 Page，之后只关注选中的一部分。每个 case 在其后面的块中都会超过这个预算。

### A.2 模型与版本

| 项目 | 取值 |
| --- | --- |
| 模型 | `Qwen/Qwen3.6-27B`，revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`，本地目录，必须是普通文件；元数据的 SHA-256 见 [docs/install.zh-CN.md 2.2 节](install.zh-CN.md#22-模型) |
| Goal rewriter（仅 `goal_query`） | `Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`（[docs/install.zh-CN.md 2.2 节](install.zh-CN.md#22-模型)） |
| vLLM | 官方 wheel `vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl`，未作任何修改，SHA-256 `700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5` |
| Python 环境 | CPython 3.12、`transformers==5.13.0`、`tokenizers==0.22.2`，用 `examples/env/constraints-3270721.txt` 构建（[docs/install.zh-CN.md 2.1 节](install.zh-CN.md#21-基础环境)）。stock 对照和客户端从同一个环境运行；本 runtime 安装到单独的 release 目录，环境本身保持不变（[docs/install.zh-CN.md 2.4 节](install.zh-CN.md#24-安装-release)） |
| Chat template | runtime 固定的 Qwen3.6 模板，SHA-256 `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`（`source-overlay/qwen_observation_span.py` 中的 `TEMPLATE_SHA256`；去掉一个末尾换行后的模板文本的摘要）。模型目录中的模板必须与它相同 |
| Runtime | 本仓库的 commit `f9322a6`（`f9322a65477deb7a6660c08a316f2d4c26644c26`，G4 运行使用的 commit），或者不改变 kv-session 协议、容量门控和 A.4 中 vLLM 参数的更新 commit。G1 和 G2 两个验证作业运行的是 `0d37346`（[docs/history/memoryagentbench-validation.zh-CN.md](history/memoryagentbench-validation.zh-CN.md)）。Gemma 最初的运行（A.15）用的是 `9ef0d6d`（integrate-1021）；之后的运行见 [docs/status.zh-CN.md](status.zh-CN.md#gemma-4-31b-it)。新的 `goal_query` 运行使用 `53e3b7a` 或更新的 commit：它的 Goal 特征以线性时间匹配引号中的字面量，输出不变（[docs/history.zh-CN.md](history.zh-CN.md#历史)）；在它之前，一条在引号字符串中被截断的回复可能在下一次 Goal 构造时让引擎卡住 |
| GPU | 一张 NVIDIA GH200 120GB，Linux aarch64；一张 GPU 同一时间只跑一个服务 |
| Slurm 资源 | `--nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16`；本 runtime 的服务用 `--mem=227120M`（节点主机内存中两张 GH200 的份额，A.4），stock 对照用 `--mem=113560M`（一份）；A.15 的 Gemma 运行用一份，`--mem=113560M` |
| 客户端 | `examples/memoryagentbench/run_cases.py` 加 `kv_client.py`（A.7）：Python 3.12，只用标准库和上述环境中的 Transformers，在同一个作业里运行于 GPU 节点上。GPU 运行用的是作业 harness 的 runner，它发送的请求相同（A.7） |
| 数据 | A.1 的 32 个 case，用 `cases.py build` 生成（B.1）；GPU 运行用的是作业 harness 的副本（manifest SHA-256 `010fd109…`） |
| 官方 harness 与评分程序 | `HUST-AI-HYZ/MemoryAgentBench`，commit `538026089d1a8a8eff05121d0db89b388f360eba`（MIT）：`utils/eval_other_utils.py`，SHA-256 `d23a5ba9f2750229315cd308463ae99aef9ff36ed0859b69d4851f804976b815`（`post_process`），以及 `llm_based_eval/summarization_evaluate.py`，SHA-256 `0139db690c4537c8f6ca5fe766ab60caf16050d172412f18ad0535577e708801`（摘要的评判）；其数据是 Hugging Face 数据集 `ai-hyz/MemoryAgentBench` |

### A.3 kv-session 协议

read_next harness 的 agent 循环自己渲染 chat template、自己解析回复；它的 KV-session 客户端发送 token id，本 runtime 服务的就是这个客户端的协议（[docs/running.zh-CN.md 6.6 节](running.zh-CN.md#66-memoryagentbench-kv-session-协议)）。与本配置相关的部分：

- **由谁渲染。** 客户端渲染对话（它的 system prompt、它唯一的工具、thinking 开关），并解析每条回复中的工具调用；服务端从不渲染模板，也不需要 tool-call parser。因此一组对比中的两种服务看到的都是客户端的 token 化结果：本 runtime 上是每个 case 一条不断追加的 token 流，stock vLLM 上是每一轮的完整重新渲染（A.5）。
- **每个 case 一条 token 流。** 一个 case 就是一个 session，其 token 流只增不减。第 0 轮发送完整的 prompt；之后每个请求只发送新的尾部：上一条回复的结束分隔符（上一条回复的最后一个 token 不是 EOS 时，把它放在最前面）、新消息和下一个生成头。每条回复都以它生成时的原始 id 留在流中。服务端把一个请求映射到阶段 1（边界和新消息）和阶段 2（生成头和 decode），不做任何渲染（[docs/design.zh-CN.md 第 5 节](design.zh-CN.md#5-query-策略的选页规则)、[docs/running.zh-CN.md 6.6 节](running.zh-CN.md#66-memoryagentbench-kv-session-协议)）。
- **Goal 的来源。** 请求还带有客户端的消息列表，它从不生成 token；`goal_query` 从上一个请求之后新增的消息中取得 observation 和显式要求（[docs/running.zh-CN.md 6.6 节](running.zh-CN.md#66-memoryagentbench-kv-session-协议)）。
- **Session。** kv-session 服务不接受 `--task`：它注册的任务是槽位，客户端 session id 在第一个请求时绑定到下一个未用的槽位（`--kv-session-slots`，默认 64）。case 结束时，客户端关闭该 case 的 session（`POST /v1/kv-session/close`）。
- **Thinking。** `--kv-session-thinking off` 固定 7 个 token 的生成头 `<|im_start|>assistant\n<think>\n\n</think>\n\n`，即模板在 `enable_thinking=False` 下追加的生成头；服务端用它检查每个请求的生成头。
- **生成约定，由服务端执行。** 只支持贪心：采样字段不是 `temperature` 0、`top_p` 1、`top_k` −1 或 0 和 `repetition_penalty` 1 的请求会被拒绝。`max_tokens` 为 1 到回复上限，并截到剩余上下文以内；`finish_reason` 为 `stop` 或 `length`（达到 `max_tokens`）；响应给出每个生成 token 的文本（含特殊 token）及其 id。
- **拒绝。** complete 请求的每种失败都是 HTTP 400 加 `{"error", "type"}`；session 有请求排队或进行中时的关闭返回 409 `SESSION_BUSY`，客户端会重试（A.7）。没有为回复留出空间的 prompt 在任何模型计算之前被拒绝，文字恰好是 `Logical RoPE context limit reached`（`CONTEXT_LIMIT`）。admission 之前发现的拒绝不影响该 session；引擎中的失败会使该 session 失效。

### A.4 本 runtime：服务参数与 vLLM 参数

**各策略共用的参数**（`serve.py`）：

| 参数 | 取值 | 作用 |
| --- | --- | --- |
| `--protocol` | `kv-session`，加 `--kv-session-thinking off`；不传 `--kv-session-slots`（默认值 64） | A.3 的协议，关闭 thinking。服务生命周期内共 64 个客户端 session；32 个 case 用掉 32 个（GPU 运行中为 33 个，因为作业 harness 的 runner 还在单独的 session 里发了一个预热请求，A.7） |
| `--max-running-sessions` | `16` | K：同时最多打开 16 个 session。vLLM 收到 `--max-num-seqs 16` |
| `--kv-cache-gib` | `32` | vLLM 收到 `--kv-cache-memory-bytes 34359738368` |
| `--page-tokens` / `--kv-layout` | `16` / `decoupled` | 16 token 的注意力 Page，以及一个单独的、有 33 个 slot（2K + 1）的 Gated-DeltaNet 池：32 GiB 时为 100 个 Mamba block 和 27,873 个可用注意力 block（[docs/running.zh-CN.md 4.4 节](running.zh-CN.md#44-一个服务上的多个-session)、[docs/reference.zh-CN.md 4.6 节](reference.zh-CN.md#46-解耦-kv-布局与-16-token-的-page)）。vLLM 收到 `--block-size 16 --mamba-block-size 784 --additional-config ...` |
| `--max-model-len` | `262144` | 模型原生的上下文（prompt 加回复，[docs/reference.zh-CN.md 4.9 节](reference.zh-CN.md#49-上下文上限与回复上限)）。最大的 case 在预期回复长度下结束时约 128K token，回复长三倍时约 152K；131,072 会拒绝它的最后几轮。上限处的一轮需要 27,873 个可用 block 中的 17,944 个 |
| `--max-tokens` | `4096` | 回复上限：客户端的 `max_tokens` 取 1 到 4096（客户端发送 4096） |
| `--dram-arena-gib` | `124` | 所有 session 共用一个 124 GiB 的锁页主机 Page arena。它保存每个打开的 session 的每个已封存 Page，直到该 session 关闭；超出它的捕获会让该请求失败（`DRAM_ARENA_EXHAUSTED`，[docs/reference.zh-CN.md 4.9 节](reference.zh-CN.md#49-上下文上限与回复上限)）。16 个最大的 case 同时处于最后一轮、回复为预期长度的三倍时，需要 123.35 GiB |
| `--connector-pool-gib` | `1` | connector 闲置的 CPU 池设为 1 GiB 而不是 vLLM 的 32 GiB（[docs/reference.zh-CN.md 4.9 节](reference.zh-CN.md#49-上下文上限与回复上限)）；在默认开启速度选项的 runtime 上（[docs/reference.zh-CN.md 4.15 节](reference.zh-CN.md#415-默认开启的速度选项)），1 GiB 就是默认值，32 GiB 需要 `--no-connector-pool-gib` 或 `--no-speed-defaults`。vLLM 收到 `cpu_bytes_to_use` 1073741824 |
| `--cuda-graphs` | `full` | vLLM 收到 `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}`，取代 `--enforce-eager`（[docs/reference.zh-CN.md 4.8 节](reference.zh-CN.md#48-cuda-graph)） |
| `--decode-fastpath` | 开启 | 每个解码步的主机工作更少，结果相同（[docs/reference.zh-CN.md 4.11 节](reference.zh-CN.md#411-解码快速路径默认开启)） |
| `--language-model-only` | 开启 | 不加载视觉塔，没有多模态预热（[docs/reference.zh-CN.md 4.12 节](reference.zh-CN.md#412-纯文本--language-model-only默认开启)）。vLLM 收到 `--language-model-only` |
| `--release-idle-pages` | 开启 | GPU 紧张时，容量门控释放空闲 session 的常驻 Page，并在它们的下一轮之前重新载入（[docs/reference.zh-CN.md 4.13 节](reference.zh-CN.md#413-释放空闲-session-的-page需显式开启)；见下）。不增加 vLLM 参数 |
| `--port` | `8000` | 服务只绑定 `127.0.0.1` |

预期回复长度是作业 harness 的规划值：每个读取轮 400 token，回答 1,000 token。

**空闲 session 与 `--release-idle-pages`。** 两轮之间，每个打开的 session 都在 GPU 上保留它的常驻 Page。16 个 session、32,768 token 预算时，一个空闲 session 钉住约 2,300 个 Page，16 个合计约 37,000 个，超过池的容量（[docs/reference.zh-CN.md 4.13 节](reference.zh-CN.md#413-释放空闲-session-的-page需显式开启)）。开了这个选项，容量门控用释放空闲 session 的全部常驻 Page 来补足一轮的缺口，最久未用的先释放；它们的主机副本留在 arena 中，被释放的 session 在下一次阶段 1 之前把 Page 载回。仍然放不下的一轮会等待（`CAPACITY_WAIT`，[docs/running.zh-CN.md 4.4 节](running.zh-CN.md#44-一个服务上的多个-session)）。本配置允许并检查释放，但不要求一定发生释放。

**主机内存。** 锁页主机内存是 1 GiB 的 connector 池、16 个 Gated-DeltaNet carry（2.29 GiB）和 124 GiB 的 arena，共 127.29 GiB，在 launcher 128 GiB 的锁页主机预算之内（[docs/reference.zh-CN.md 4.9 节](reference.zh-CN.md#49-上下文上限与回复上限)）；若用 vLLM 的 32 GiB connector 池（`--no-connector-pool-gib`）则为 158.3 GiB，launcher 会拒绝。再加约 32 GiB 的进程内存和 5 GiB 余量，作业需要 168,237 MiB，超过节点主机内存中一张 GH200 的份额（113,560 MiB），所以这些服务申请两份，`--mem=227120M`。

**各策略的参数：**

| 策略 | selector 参数 | admission 参数 | Goal rewriter |
| --- | --- | --- | --- |
| `observation_query` | `--active-token-budget 32768` | 无 | 不启动（[docs/design.zh-CN.md 第 5 节](design.zh-CN.md#5-query-策略的选页规则)） |
| `goal_query` | `--active-token-budget 32768 --goal-rewriter-model <Qwen3-0.6B 目录>` | `--admission-overlap 16` | 同一张 GPU 上一个 Qwen3-0.6B 进程 |
| `full`（不在 G4 运行中） | 无 | 无 | 无 |

两个 query 策略的预算是同一个取值。任何服务都不设置 `--gdn-device-handoff`、`--phase1-in-step`、`--active-ratio`、`--goal-query-variant`、`--goal-rewriter-batch`、`--fill-seed`、`--decode-fastpath2` 或 `--decode-fastpath3`。admission 参数遵循 [docs/reference.zh-CN.md 4.10 节](reference.zh-CN.md#410-admission-优化需显式开启)（16 个 session 的 `goal_query` 加 `--admission-overlap 16`）。selector、rewriter 和 admission 参数会固定到启动配置里，但不增加任何 vLLM 参数，所以各服务启动 vLLM 时的参数完全相同。

以上是 Qwen G4 正式运行所用的参数，不含 SWE-bench 和 Terminal-Bench 运行所用的优化开关（`--decode-fastpath2`、`--decode-fastpath3`、`--deferred-completion-io`、`--goal-rewriter-batch 4`），A.15 的 Gemma 运行则使用了它们。在默认开启速度选项的 runtime 上（[docs/reference.zh-CN.md 4.15 节](reference.zh-CN.md#415-默认开启的速度选项)），解码快速路径和 `--deferred-completion-io` 不关闭就会开启，所以命令加上 `--no-decode-fastpath2`（第三阶段随之关闭）和 `--no-deferred-completion-io`；`--goal-rewriter-batch` 不是速度默认选项，不传就保持关闭。

**命令。** 两个服务的 dry run（加 `--start` 即启动服务；`LABEL` 是每次尝试的新标签）：

```sh
SHARED=(--protocol kv-session --kv-session-thinking off --max-running-sessions 16 --kv-cache-gib 32
        --page-tokens 16 --kv-layout decoupled --max-model-len 262144 --max-tokens 4096 --dram-arena-gib 124
        --connector-pool-gib 1 --cuda-graphs full --decode-fastpath --no-decode-fastpath2
        --no-deferred-completion-io --language-model-only --release-idle-pages)
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

启动 `serve.py` 的 shell 的**环境变量**（[docs/install.zh-CN.md 第 3 节](install.zh-CN.md#3-运行环境)）：

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

对照服务就是从同一个环境（A.2）直接启动的 `vllm.entrypoints.openai.api_server`，不带本 runtime 的任何代码，经由客户端的 Vanilla transport 访问（`run_cases.py --mode vanilla`，A.7）。

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
| `--language-model-only` | 开启 | 相同（[docs/reference.zh-CN.md 4.12 节](reference.zh-CN.md#412-纯文本--language-model-only默认开启)） |
| `--mamba-cache-mode` | `align` | 相同 |
| `--gdn-prefill-backend` | `triton` | 相同 |
| CUDA graph | vLLM 默认：既不加 `--enforce-eager` 也不加 `--compilation-config`，结果是 `FULL_AND_PIECEWISE` 加 `VLLM_COMPILE`（[docs/reference.zh-CN.md 4.8 节](reference.zh-CN.md#48-cuda-graph)） | 模式相同 |
| 异步调度 | vLLM 默认（开启），不加参数 | 不同：launcher 默认传 `--no-async-scheduling`；它的可选项 `--async-scheduling` 这里不用（A.14） |
| 注意力后端 | 不设置，由 vLLM 选择（A.6） | 相同 |
| 路由 | `POST /v1/completions`，prompt 以 token id 发送 | 不同：本 runtime 服务 `/v1/kv-session/complete`（A.3） |
| Thinking | 关闭，体现在客户端的渲染中（A.7） | 生成头相同 |

其余每个取值都显式设置，这样对照服务不依赖 vLLM 对这个混合模型的默认值；CUDA graph 和异步调度保持 vLLM 的默认值（见上）。只服务于 runtime 自身路由的参数（`--revision`、reasoning 和 tool parser、`--chat-template`、`--kv-transfer-config`、`--block-size`、`--mamba-block-size`、`--additional-config`）不传：客户端发送 token id，并自己解析回复。

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
| CUDA graph | `FULL_AND_PIECEWISE`，两边都是 16 个序列，捕获尺寸为 1、2、4、8、16、24 和 32（[docs/reference.zh-CN.md 4.8 节](reference.zh-CN.md#48-cuda-graph)） | 引擎配置行中的 `'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE` 和 `'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32]` |
| 异步调度 | 本 runtime：关闭。stock：开启 | `Asynchronous scheduling is disabled.`（本 runtime）或 `Asynchronous scheduling is enabled.`（stock） |
| 纯文本 | 两边都用 `--language-model-only`（[docs/reference.zh-CN.md 4.12 节](reference.zh-CN.md#412-纯文本--language-model-only默认开启)） | vLLM 的 `running in text-only mode`；不能出现 `Encoder cache will be initialized` 或 `Multi-modal warmup completed`（可以出现 `Using ... for MMEncoderAttention` 行） |

READY 之后从每个服务的日志中确认：

```sh
grep -oE "Using FLASH_ATTN attention backend|Using FlashAttention version [0-9]+|Using Triton/FLA GDN prefill kernel \(requested=[a-z]+|FlashInfer top-p/top-k sampling disabled|Asynchronous scheduling is [a-z]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|'cudagraph_capture_sizes': \[[0-9, ]*\]|running in text-only mode|Encoder cache will be initialized|Multi-modal warmup completed" \
  server.log | sort | uniq -c
```

表中每一行都必须出现，且不能出现其他注意力后端、FlashAttention 版本、prefill kernel 或 CUDA graph 模式，也不能出现多模态启动日志行。日志不符的服务不纳入对比。

### A.7 runner

runner 是 `examples/memoryagentbench/run_cases.py` 加 `kv_client.py` 中的客户端：按本项目的运行方式收录的 read_next harness（Shanbin Sun 的设计），带两种 transport，即 KV-session 客户端（本 runtime）和 Vanilla transport（`--mode vanilla`，stock vLLM）（[benchmarks/memoryagentbench.zh-CN.md](benchmarks/memoryagentbench.zh-CN.md) 的 Harness 一节）。循环中的指令文字是 `run_cases.py` 中的常量；本文档不复述。GPU 运行（[docs/status.zh-CN.md](status.zh-CN.md)）用的是作业 harness 自己的 runner，其循环文字和请求构造与 `examples/memoryagentbench/` 相同；下表的“Tokenizer 与模板”“kv 模式”“预热”“超时与重试”和“记录”五行说明 `run_cases.py` 与它的不同之处。baseline 使用 read_next harness 本身，或者使用发送相同请求的客户端。runner 在 GPU 节点上以单独的进程运行，访问 `127.0.0.1`。

| 项目 | 取值 |
| --- | --- |
| Tokenizer 与模板 | 模型目录的 tokenizer，经 Transformers 的 `apply_chat_template`，`enable_thinking=False`，带唯一的工具 `bash`（一个字符串参数 `command`，即 mini-swe-agent 的工具）；token 由 `encode(text, add_special_tokens=False)` 得到。启动时 runner 检查生成头（A.3），并把模板的 SHA-256 记入 `run.json`（`tokenizer.template_sha256`）；它不把它与 A.2 的固定值比较（作业 harness 的 runner 比较过），所以先用 B.2 检查模板 |
| worker 与顺序 | `--workers 16`；case 按历史 token 数从大到小、再按 id 排序（`--order largest-first`，默认值）：即 A.1 的顺序 |
| 消息 | 一条 system 消息（case 的 system prompt 后接 `read_next` 说明 `READ_INSTRUCTION`）和固定的第一条 user 消息（`FIRST_USER`）；之后是各条回复，以及下面所述的 tool 和 user 消息 |
| 分块 | 一条回复中有效的 `bash` 调用、且命令为 `read_next` 时，以返回码 0 的 tool 消息得到下一块，并带上该块的序号（`[History chunk k/N]`）；一条回复中的多个这样的调用按顺序得到多块。最后一块带有结束说明（`END_NOTE`），case 的问题作为 user 消息跟在同一个请求中 |
| 格式错误 | 在最后一块之前不含工具调用的回复，会得到提醒（`NUDGE`，一条 user 消息）；无效的调用（别的命令，或最后一块之后的 `read_next`）得到返回码 1 的 tool 消息。连续两轮出现这类错误则结束该 case（`format_error`） |
| 回答 | 最后一块之后第一条不含工具调用的回复；它的内容就是 prediction（`answered`）。停在 `max_tokens` 的回答：回复没有 reasoning 部分时（关闭 thinking 时总是如此）记为 `unfinished_thinking`，prediction 为空；否则记为 `answer_length_limit`，保留其文本（read_next harness 的规则） |
| 轮数 | 每个 case 最多 120 个请求（`turn_limit`）。N 块的 case 在每条回复读一块时需要 N + 1 个请求 |
| 采样字段 | `temperature` 0、`top_p` 1、`top_k` −1、`repetition_penalty` 1、`seed` 42、`max_tokens` 4096 |
| kv 模式（本 runtime） | 先 `GET /v1/kv-session/info`：method 必须是 `dram_backed_kv`，服务端的 thinking 设置和轮次格式必须与客户端的相同（`--thinking`、`--grammar`）；客户端从服务端取得预算并随每个请求发送（它不检查预算是否为 32768；预算是 A.4 的 `--active-token-budget 32768`，`GET /v1/kv-session/info` 会报告它，见 B.4）。每个 case 一个新的 32 位十六进制 session id。第 0 轮：完整的渲染（`prefix_tokens` + `header_tokens`，`boundary_tokens` 为 0）。之后各轮：最新回复之后的尾部，做法是把对话中的这条回复换成一个标记，分别带和不带生成提示渲染：这条回复的结束分隔符、新消息和生成头（`header_tokens` 为 7）；`boundary_tokens` 是尾部第一个 `<|im_start|>` 之前的 token 数，若把上一条回复的最后一个输出 id 放回最前面（它不是 `<|im_end|>` 时，即 pad 或长度停止之后），再加 1。请求体还带有规范化后的消息列表、method `dram_backed_kv` 和服务端的预算。case 结束时 `POST /v1/kv-session/close`，遇到 409 每隔 2 秒重试，最多 120 秒 |
| vanilla 模式（stock） | 每一轮把完整对话重新渲染，以 token id 发给 `/v1/completions`（A.5）；每个 case 一个新的 `cache_salt`；不关闭。prompt 长度加 `max_tokens` 超过 `--max-window`（262144）时，在发送之前结束该 case |
| 预热 | 没有。GPU 运行中作业 harness 的 runner 在第一个 case 之前，在单独的 session 里发一个 2 token 的请求并随即关闭；它不属于任何 case 或指标 |
| 超时与重试 | `--request-timeout`，默认每个请求 3600 秒，对每个请求都一样（作业 harness 的 runner 在剩余的 agent 时间加一条回复的收尾时间更短时取后者，至少 30 秒）。不重试：被拒绝或失败的请求结束它所在的 case，状态为 `error`，原因记在 `runner_status` 中（`context_limit`、`server_error`、`transport_error`）。用 `--agent-seconds S` 时，S 秒之后不再发起请求；进行中的请求完成，每个 case 关闭自己的 session，这样被截断的 case 为 `error`，`runner_status` 为 `stopped_cap` |
| 记录 | 在 `--out` 中：`run.json`（设置、tokenizer 的身份、服务端的 info、结果和每种 case 状态的计数）；`requests.jsonl`，每个请求一行（case、轮次、类型、状态、token 数、`finish_reason`、解析出的工具调用、节点单调时钟上的 `t_send_ns` 和 `t_recv_ns`）；每个 case 一个 `cases/<id>/result.json`（`status`、`runner_status`、`prediction`、`chunks_read`、`agent_output_tokens` 即其各请求的 completion token 数之和、逐请求的摘要，以及墙钟秒数 `started` 和 `finished`）；`summary.json`，即 `score.py` 的输入；用 `--trace` 时还有 `trajectories.jsonl`。作业 harness 的 runner 写的则是 `cases.jsonl` 和 `runner.json`，带单调时钟上的 `started_ns` 和 `finished_ns` |

### A.8 限制

| 限制 | 取值 |
| --- | --- |
| 每个 case 的轮数 | 120 个请求（`turn_limit`） |
| 格式错误 | 连续两轮则结束该 case（`format_error`） |
| 回复上限 | 每个请求最多 4096 个新 token：客户端的 `max_tokens`（本 runtime：在 `--max-tokens 4096` 之内，并截到剩余上下文以内；stock：来自客户端） |
| 上下文上限 | 262,144 token（prompt 加回复）。本 runtime 拒绝没有为回复留出空间的 prompt（`CONTEXT_LIMIT`，A.3）；Vanilla 客户端在 prompt 加 4096 超过上限时结束该 case（A.5） |
| 客户端超时 | 每个请求 3600 秒 |
| 客户端重试 | 无 |
| 每个服务的 session | 同时最多打开 16 个（K）；服务生命周期内 64 个槽位，用掉 32 个（GPU 运行中加上预热为 33 个）；生命周期内的请求上限是 max(1,024, 256 × 64) = 16,384（[docs/reference.zh-CN.md 4.9 节](reference.zh-CN.md#49-上下文上限与回复上限)），而 case 的请求最多 32 × 120 = 3,840 个 |
| 锁页 Page arena | 124 GiB（本 runtime）；超出它的捕获会让该请求失败（`DRAM_ARENA_EXHAUSTED`，按 A.12 算作失败） |
| Agent 时间 | GPU 运行中作业 harness 的上限：最多 3,000 秒（`observation_query`）、4,200 秒（`goal_query`）和 6,000 秒（stock），再减去作业启动和关闭所需的时间。到达上限后不再发起请求，进行中的请求最多再完成一条回复，然后关闭所有打开的 session；这样被截断的 case 记为 `stopped_cap`，该次运行是不完整的运行（A.12）。`run_cases.py --agent-seconds` 采用同样的规则（A.7） |

**为什么客户端超时设为 3600 秒、且不重试。** 一个请求在最多 4096 token 的回复开始之前，可能要等待它的 admission：在 `--admission-overlap 16` 下，前端最多等待一次 admission 2,100 秒（[docs/reference.zh-CN.md 4.10 节](reference.zh-CN.md#410-admission-优化需显式开启)），`goal_query` 的 admission 可能要等待唯一的 Goal rewriter 进程；放不下的一轮会等待容量（`CAPACITY_WAIT`，[docs/running.zh-CN.md 4.4 节](running.zh-CN.md#44-一个服务上的多个-session)）。3600 秒的超时让这类等待不会被打断。重试一个仍在进行中的请求，就是对一个忙碌的 session 发出第二个请求，协议会拒绝它。stock 对照使用相同的设置，使两边只在服务端不同。

### A.9 并发与 session 生命周期

- 16 个客户端 worker 按 A.1 的顺序领取 32 个 case，每个 worker 一次一个；一个 case 结束（其 session 关闭）后，它的 worker 领取下一个 case。最大的 case 先跑，这样 16 个 worker 能把各 case 排得紧凑（列表调度）。
- **本 runtime 上的 session。** 一个 case 的 session id 在第 0 轮请求时绑定到下一个未用的槽位（[docs/running.zh-CN.md 6.6 节](running.zh-CN.md#66-memoryagentbench-kv-session-协议)）；最多 16 个同时打开。两轮之间，session 在 GPU 上保留常驻 Page，在主机上保留 carry 和已封存的 Page；`--release-idle-pages` 可以释放空闲 session 的常驻 Page，并在它的下一轮之前重新载入（A.4）。case 结束时客户端关闭 session，释放它在 GPU 和主机上的状态（[docs/running.zh-CN.md 4.4 节](running.zh-CN.md#44-一个服务上的多个-session)，session 池），然后下一个 case 才打开。
- **stock 对照。** 没有 session，也不关闭；每个 case 的 `cache_salt` 让一个 case 复用它自己的 prefix block、从不复用别的 case 的，stock vLLM 把这些 block 作为可淘汰的 prefix cache 保留。
- **case 之间不共享。** 8 个 Conflict Resolution case 共用一段历史，但每个 case 都在自己的 session 中读它；任何服务都不在 case 之间共享 KV。

### A.10 评测

| 项目 | 取值 |
| --- | --- |
| 评测程序 | `utils/eval_other_utils.py` 的官方 `post_process`（A.2），在运行之后离线执行，从不在 GPU 节点上执行 |
| 输入 | 每个 case 的最终 prediction（没有回答的 case 为空）、参考答案和来源子集 |
| 调用 | `metrics, parsed = post_process({"output": prediction}, answers, {"sub_dataset": source})`；该 case 的得分是 `metrics[metric]`，指标见 A.1 |
| 指标 | Test-Time Learning（ICL）：对解析出的答案计算 `exact_match`。Accurate Retrieval（RULER QA）和 Conflict Resolution（FactConsolidation）：`substring_exact_match`，取原始 prediction 和解析后 prediction 中较大的一个。Long-Range Understanding（InfBench 摘要）：`llm_judge_f1` |
| Long-Range Understanding | 官方评判 `llm_based_eval/summarization_evaluate.py`，使用 `gpt-4o-2024-05-13`（fluency 乘以关键点 recall 与句子 precision 的 F1）；它需要 OpenAI API key。它没有运行：评分推迟（[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 5.1），这 8 个 case 和 Score 在它运行之前都处于待定状态 |
| 依赖 | 该评分文件导入 numpy、nltk、tiktoken、rouge_score（及 absl-py）和 editdistance。这四个子集用不到 tiktoken 和 editdistance，rouge_score 只增加 `rouge*` 字段，这里没有任何指标用到它们 |
| 没有回答的 case | 以 `format_error`、`turn_limit`、`unfinished_thinking` 或 `answer_length_limit` 结束的 case，按它留下的 prediction 评分（空 prediction 得 0 分） |

### A.11 指标

**Score。** 一个 case 的得分在 [0, 1] 之间（A.10）。一项能力的得分是其 8 个 case 的平均值 × 100；**Score** 是四项能力得分的等权平均，采用 0–100 分制。一个 case 是其能力的 12.5 分、Score 的 3.125 分。对 50 % 的二值指标，一次运行中一项能力的二项分布标准误约为 17.7 分，Score 的约为 8.8 分，两次独立运行的 Score 之差的约为 12.5 分。保留每个 case 的得分，以便逐个 case 比较各次运行。

**吞吐** = 输出 token 数 / 该次运行的时间跨度：

| 部分 | 定义 |
| --- | --- |
| 分子 | 32 个 case 的每个请求（读取轮和回答轮一视同仁）的 completion token 数之和（每个 case 的 `agent_output_tokens`），取服务端报告的值（本 runtime：响应中的 `completion_tokens`；stock：`usage.completion_tokens`） |
| 分母 | GPU 节点时钟上，从第一个 case 开始到最后一个 case 结束（在其 session 关闭之后）的时间跨度：各 case 的 `result.json` 上的 `max(finished) − min(started)`（`run_cases.py`，墙钟秒数）；GPU 运行中作业 harness 的 runner 记录的是单调时钟上 `cases.jsonl` 的 `max(finished_ns) − min(started_ns)` |
| 包含 | 这两个时刻之间的一切：admission 及其等待、容量等待、释放和重新载入、Goal rewriter 的时间（`goal_query`）、客户端的渲染、关闭 session |
| 不包含 | 服务的启动和关闭、作业 harness 的预热请求（GPU 运行）、评分，以及 Goal rewriter 自己的 token（辅助模型） |

每次运行还要记录：每个 case 的状态、读取的块数、轮数和墙钟时间；每个服务的请求数、上下文超限拒绝次数、释放和重新载入次数（`capacity.jsonl`）、`CAPACITY_WAIT` 次数（本 runtime），以及抢占计数（stock，A.5）。

### A.12 运行规则

- **每个作业一个服务**，一直运行到结束；客户端在同一个作业里运行于 GPU 节点上，与服务共用该作业的 16 个 CPU。要比较的各个服务使用同一类节点和同样的作业规模。
- **每次尝试都用全新的服务和新的输出目录。** `serve.py` 拒绝已存在的输出目录。
- **中断的运行不续跑。** 提前结束的运行（作业时间上限、agent 时间上限、服务失败）是不完整的运行：未完成的 case 不在另一个作业里接着跑，该次运行不纳入对比，也不给出 Score 和吞吐，整个服务换一个新标签从头重跑。
- **失败的计法。** 有 case 以 `error` 结束（`run_cases.py` 的 `runner_status`：`server_error`、`transport_error`、`context_limit`、`stopped_cap` 或其他错误），该次运行就是不完整的运行。在本 runtime 上，以下情况还会让服务的检查不通过：某一轮不是 token 级精确（服务端的 prompt 与客户端的构造逐 token 相同），某个请求在 `failures.jsonl` 中有记录，发生了任何抢占，释放的不是空闲 session，或者重新载入没有恢复它对应的释放，或者关闭不干净（[docs/design.zh-CN.md 第 8 节](design.zh-CN.md#8-每个服务的证据)）。
- **容量等待**（`CAPACITY_WAIT`）是正常的；一直持续到 agent 时间上限的停滞会使该次运行成为不完整的运行。
- **传输方式。** 客户端通过节点的 loopback 访问服务；服务从不在 loopback 之外监听（[docs/running.zh-CN.md 6.4 节](running.zh-CN.md#64-从其他主机访问服务)）。
- **内存位置。** 在两份份额的作业中，锁页 arena 可能跨两个 Grace socket；作业 harness 把作业的 NUMA 位置记为一项发现，不算失败。

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
| 15 | 异步调度 | 关闭（launcher 的默认值；这里不用） | vLLM 默认（开启）；已知差异（A.14） |
| 16 | 注意力 | FLASH_ATTN 加 FlashAttention 3，由 vLLM 选择 | 相同，不加 `--attention-backend` |
| 17 | Gated-DeltaNet prefill | Triton | `--gdn-prefill-backend triton` |
| 18 | Sampler | 原生（`VLLM_USE_FLASHINFER_SAMPLER=0`） | 同一个变量 |
| 19 | Prefix caching | 开启，sha256 | `--enable-prefix-caching --prefix-caching-hash-algo sha256`，每个 case 一个 `cache_salt` |
| 20 | Mamba cache 模式 | `align` | `--mamba-cache-mode align` |
| 21 | 纯文本 | `--language-model-only` | vLLM 的 `--language-model-only` |
| 22 | 客户端 | `run_cases.py` 的 read_next 循环（A.7），用 Transformers 5.13.0 和模板 `e84f32a2…` 渲染 | 相同的循环和模板 |
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

1. **Agent 循环。** 官方 harness 把一段历史的各块交给它的记忆 agent，然后提出这段历史的所有问题（“注入一次，多次提问”）。read_next harness 给每个问题一个单独的 session，先通过 `read_next` 工具调用读完整段历史，然后才得到问题。8 个 Conflict Resolution 问题共用一段历史，这段历史被读了 8 遍。
2. **分块。** `cases.py` 把每段历史切成连续的 4,096 个 Qwen3.6 token 的窗口（A.1）。官方 harness 用自己的按句分块器以 `chunk_size` 4096 切分历史，token 用 tiktoken 计数（`configs/data_conf`，`utils/eval_other_utils.py` 中的 `chunk_text_into_sentences`），所以它的分块边界可能不同。
3. **回复长度。** 这里每一轮都可以生成 4096 token；官方的数据配置给每个子集各自的 `generation_max_length`（例如 `Factconsolidation_sh_32k.yaml` 中为 10，`ICL_trec_coarse.yaml` 中为 20，`InfBench_sum.yaml` 中为 1,200）。
4. **Thinking。** 这里关闭。read_next harness 自己的配置开启 thinking，当前配置（[docs/status.zh-CN.md](status.zh-CN.md#当前配置)）也开启；开启 thinking 的结果是另一种配置，不能直接比较（[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 4.3）。
5. **规模。** 每项能力 8 个问题，只跑一次。官方配置每个子集提出的问题更多。
6. **评分。** 官方 `post_process` 离线作用于每个 case 的最终回答；摘要需要官方的 GPT-4o 评判（A.10）。

**本 runtime 与 stock vLLM 之间：**

1. **历史。** 本 runtime 把每条回复保留为它生成时的原始 id，前面是关闭 thinking 的生成头放进去的空思考块。Vanilla 对照每一轮从文本重新渲染完整对话：模板会去掉每条较早回复的空思考块（每条 4 个 token，与 [docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)对 chat-text 对照的描述相同），并重新编码回复文本，所以两边的 prompt 从第一条较早的回复处开始不同。
2. **异步调度。** 本 runtime 不使用（launcher 默认传 `--no-async-scheduling`；它的可选项 `--async-scheduling` 这里不用）；stock vLLM 使用其默认值，即开启。
3. **采样由谁控制。** 本 runtime 拒绝非贪心的采样字段并贪心解码；stock vLLM 从请求中读取采样设置（`temperature` 0 即贪心解码）。
4. **上下文边界。** 本 runtime 只在没有为回复留出任何空间时才拒绝 prompt，并把 `max_tokens` 截到剩余空间以内（A.3）；Vanilla 客户端在 prompt 加 4096 超过 262,144 token 时自己结束该 case。在预期回复长度下，没有 case 会到达这两个边界（A.4）。
5. **KV 管理。** 两者的 GPU KV 都是 32 GiB，但布局不同。本 runtime 使用 16 token 的注意力 Page、单独的 Gated-DeltaNet 池、1 GiB 的 CPU 池和 124 GiB 的锁页 Page arena；它从不抢占，放不下的一轮会等待（`CAPACITY_WAIT`），GPU 紧张时空闲 session 的 Page 会被释放并重新载入（A.4）。stock vLLM 使用它自己的混合布局（[docs/reference.zh-CN.md 4.6 节](reference.zh-CN.md#46-解耦-kv-布局与-16-token-的-page)），只把已结束请求的 block 作为可淘汰的 prefix cache 保留，淘汰掉的部分要重新 prefill，池满时可能抢占。32 GiB 中有 16 个长 session 时，一个 session 的 prefix 可能在它的下一轮之前就被淘汰。
6. **admission 与调度。** 本 runtime 按到达顺序一次准入一个请求（`goal_query` 在 `--admission-overlap 16` 下最多 16 个同时进行），每次 admission 的阶段 1 是单独的一次 forward，然后成批 decode（[docs/running.zh-CN.md 4.4 节](running.zh-CN.md#44-一个服务上的多个-session)、[docs/reference.zh-CN.md 4.10 节](reference.zh-CN.md#410-admission-优化需显式开启)和[docs/design.zh-CN.md 第 5 节](design.zh-CN.md#5-query-策略的选页规则)）。stock vLLM 在每步 16,384 token 的预算内把 prefill 分块与 decode 步一起调度。
7. **Session 生命周期。** 本 runtime 在客户端关闭 session 时释放该 case 的状态；stock vLLM 把缓存的 prefix 保留到它自行淘汰为止。
8. **选择性策略。** 一个 session 已封存的历史超过 32,768 token 的预算后，`observation_query` 和 `goal_query` 只关注选中的一部分历史 Page（A.1）。
9. **Goal rewriter。** `goal_query` 在同一张 GPU 上运行一个辅助的 Qwen3-0.6B。它的时间计入吞吐，它的 token 不计入。
10. **路由。** 本 runtime 以追加的尾部服务 `/v1/kv-session/complete`；stock vLLM 以完整的 prompt 服务 `/v1/completions`。
11. **两次运行之间的差异。** 两种服务的成批运行在两次运行之间都不是 token 完全相同的：回复可能随 batch 的组成而变，而 batch 的组成取决于时序（[docs/running.zh-CN.md 4.4 节](running.zh-CN.md#44-一个服务上的多个-session)）。G1 的 token 级一致（作业 3397844，[docs/status.zh-CN.md](status.zh-CN.md)）只对一个 eager、同步调度、回放相同 prompt 的 session 成立，不适用于本配置的成批运行。

### A.15 Gemma 4 31B IT

本节是本正式测试的 Gemma 4 31B IT 路径，即它最初几次运行所用的设置（作业 3429503 和 3429507；之后的运行，其中包括开 `--async-scheduling`、跑 `random_fill`、`goal_query` 服务使用 vLLM Goal 引擎加 `--goal-overlap` 的作业 3601174，及其设置见 [docs/status.zh-CN.md](status.zh-CN.md#gemma-4-31b-it)）。它不是 A.1 到 A.14 那样 32 个 case 的配置：它在 8 个 session 上跑 8 个 case。本节没有提到的都与 Qwen 相同：关闭 thinking，贪心解码及 A.7 的采样字段，4096 token 的回复上限，262,144 token 的上下文，32,768 token 的预算，A.7 的 runner（使用 Gemma 的模板），A.8 的限制和 A.10 的评测。

| 项目 | 取值 |
| --- | --- |
| 模型 | `google/gemma-4-31B-it`，revision `842da3794eaa0b77d5f08bae87a17459d91ff475`，用于以 `install.py --model-profile gemma4-31b-it` 安装的 release（`$RELEASE-gemma4`，[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)） |
| Runtime | 它的前两次运行（作业 3429503 和 3429507，[docs/status.zh-CN.md](status.zh-CN.md)）用 `9ef0d6d`（integrate-1021）；新的 `goal_query` 运行使用 `53e3b7a` 或更新的 commit（A.2） |
| Case | A.1 的 32 个 case 中的 8 个，按 A.1 的顺序取每个来源的前两个（见下）。A.1 中的 token 数是 Qwen 的；Gemma 的 tokenizer 得到的约为其 0.99–1.01（作业 harness 固定的 Gemma token 数） |
| 并发 | 8 个 case 在 8 个 session 上同时运行；每个服务跑一次 |
| 协议 | kv-session，使用 Gemma 的轮次格式，关闭 thinking：用户轮之后是模板在关闭 thinking 时的 7 个 token 的生成头，工具响应之后 `header_tokens` 为 0，停止 id 为 0、1、50 和 106，启动记录中为 `"grammar": "gemma4"`（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)、[docs/running.zh-CN.md 6.6 节](running.zh-CN.md#66-memoryagentbench-kv-session-协议)） |
| Slurm 资源 | 与 A.2 相同，但只用一份：`--mem=113560M`（主机内存见下） |

这 8 个 case，按 A.1 的顺序：

```text
icl_trec_coarse_6600shot_balance-r311-q0
icl_trec_coarse_6600shot_balance-r324-q0
infbench_sum_eng_shots2-r52-q0
infbench_sum_eng_shots2-r84-q0
ruler_qa2_validation_65536-r365-q0
ruler_qa2_validation_65536-r350-q0
factconsolidation_sh_32k-r5-q22
factconsolidation_sh_32k-r5-q37
```

**各策略共用的参数**（`serve.py`）：

| 参数 | 取值 | 作用 |
| --- | --- | --- |
| `--protocol` | `kv-session`，加 `--kv-session-thinking off` | A.3 的协议，使用 Gemma 的轮次格式（见上）；本客户端的 Gemma 尾部按关闭 thinking 渲染，所以这里不用 `--kv-session-thinking on`（gemma-thinking）。必须显式传 `off`：Gemma profile 上启动器的默认值现在是 `on`（gemma-thinking 之前，它会拒绝不带 `off` 的 Gemma kv-session 服务），保持 `on` 的服务会拒绝本客户端的每个请求（`HEADER_MISMATCH`） |
| `--max-running-sessions` | `8` | K：8 个 session。在 32,768 token 的预算下，16 个 session 放不进 25 GiB：选择性规则需要 5,699 个全局 Page，而可用的只有 4,957 个。vLLM 收到 `--max-num-seqs 8` |
| `--kv-cache-gib` | `25` | 组里 Gemma baseline 所用的 KV cache（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）。vLLM 收到 `--kv-cache-memory-bytes 26843545600`。8 个 session 时池 0 有 10,158 个全局 Page，选择性规则需要 5,743 个（`goal_query` 的 Goal forward 在池 1 中有自己的临时空间：8,878 对 5,730） |
| `--page-tokens` / `--kv-layout` | `16` / `decoupled` | Gemma profile 的窗口布局：池 0 存放全局层的 16 token Page（每个 1.25 MiB），池 1 存放各 session 的滑动窗口（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）。vLLM 收到 `--block-size 16` 和窗口布局的 `--additional-config`，其中带有选择性规则的 `request_pages` |
| `--max-model-len` | `262144` | 与 A.4 相同；在 Gemma 上选择性规则允许这个上下文（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)） |
| `--max-tokens` | `4096` | 与 A.4 相同 |
| `--dram-arena-gib` | `60` | 8 个 session 的 Gemma Page 共用一个 60 GiB 的锁页 arena；超出它的捕获会让该请求失败（`DRAM_ARENA_EXHAUSTED`） |
| `--connector-pool-gib` | `1` | 与 A.4 相同 |
| `--cuda-graphs` | `full` | 与 A.4 相同（docs/reference.zh-CN.md [4.8](reference.zh-CN.md#48-cuda-graph)、[4.14](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证) 节） |
| `--decode-fastpath`、`--decode-fastpath2`、`--decode-fastpath3` | 开启 | 三个解码快速路径：每个解码步的主机工作更少，结果相同（docs/reference.zh-CN.md [4.11](reference.zh-CN.md#411-解码快速路径默认开启)、[4.14](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证) 节）。不增加 vLLM 参数 |
| `--deferred-completion-io` | 开启 | 完成文件在每步之后由一个写入线程写出，字节相同（[docs/reference.zh-CN.md 4.11 节](reference.zh-CN.md#411-解码快速路径默认开启)）。不增加 vLLM 参数 |
| `--language-model-only` | 开启 | Gemma profile 要求必须开启（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）。vLLM 收到 `--language-model-only` |
| `--release-idle-pages` | 开启 | 与 A.4 相同：GPU 紧张时释放空闲 session 的全局 Page，并在它们的下一次阶段 1 之前重新载入；滑动窗口始终留在 GPU 上（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)） |
| `--port` | `8000` | 服务只绑定 `127.0.0.1` |

**主机内存。** 60 GiB 的 arena 加 1 GiB 的 connector 池共锁页 61.01 GiB，在 Gemma launcher 为一份主机内存允许的 80.9 GiB 之内（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）；Gemma 没有 Gated-DeltaNet carry。所以作业只申请一份，`--mem=113560M`。

**各策略的参数：**

| 策略 | selector 参数 | admission 参数 | Goal rewriter |
| --- | --- | --- | --- |
| `observation_query` | `--active-token-budget 32768` | 无 | 不启动（[docs/design.zh-CN.md 第 5 节](design.zh-CN.md#5-query-策略的选页规则)） |
| `goal_query` | `--active-token-budget 32768 --goal-rewriter-model <Qwen3-0.6B 目录> --goal-rewriter-batch 4` | `--admission-overlap 8`（K = 8） | 同一张 GPU 上一个 Qwen3-0.6B 进程；排队的调用每批最多 4 个，受 Gemma profile 在 25 GiB 时的默认 token 上限 25,600 约束（不传 `--goal-rewriter-batch-tokens`；`startup.json` 和 `ready.json` 固定它；docs/reference.zh-CN.md [4.10](reference.zh-CN.md#410-admission-优化需显式开启)、[4.14](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证) 节）。批量 greedy 解码与逐个调用不是逐位相同（[docs/reference.zh-CN.md 4.10 节](reference.zh-CN.md#410-admission-优化需显式开启)） |
| `random_fill` | `--active-token-budget 32768` | 无 | 不启动 |

任何服务都不设置 `--gdn-device-handoff` 或 `--phase1-in-step`（Gemma profile 拒绝这两个参数，[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）、`--active-ratio`、`--goal-query-variant`、`--fill-seed`、`--host-memory-shares` 或 `--goal-rewriter-batch-tokens`。

**命令。** 两个服务的 dry run（加 `--start` 即启动服务；`LABEL` 是每次尝试的新标签）：

```sh
SHARED=(--protocol kv-session --kv-session-thinking off --max-running-sessions 8 --kv-cache-gib 25
        --page-tokens 16 --kv-layout decoupled --max-model-len 262144 --max-tokens 4096 --dram-arena-gib 60
        --connector-pool-gib 1 --cuda-graphs full --decode-fastpath --decode-fastpath2 --decode-fastpath3
        --deferred-completion-io --language-model-only --release-idle-pages)
python -B serve.py --release-dir "$RELEASE-gemma4" --policy observation_query "${SHARED[@]}" \
  --active-token-budget 32768 --output "$RUNS/mab-gemma-observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE-gemma4" --policy goal_query "${SHARED[@]}" \
  --active-token-budget 32768 --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --admission-overlap 8 \
  --goal-rewriter-batch 4 --output "$RUNS/mab-gemma-goal_query-$LABEL" --port 8000
```

**vLLM 参数。** 以 dry run 打印的列表为准。与 A.4 的列表相比：`--max-num-seqs 8`、`--kv-cache-memory-bytes 26843545600`、`--max-num-batched-tokens 8192`（一次原生 forward 为 8,192 token，[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）、`--attention-backend TRITON_ATTN`、`gemma4` 的 reasoning parser 和 tool-call parser 加 `--chat-template-content-format openai`、窗口布局的 `--additional-config`，并且没有任何 Mamba 或 Gated-DeltaNet 参数（`--mamba-block-size`、`--mamba-cache-mode`、`--gdn-prefill-backend`）。

**4,096 token 的前向（可选，[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)；GPU 运行见 [docs/status.zh-CN.md](status.zh-CN.md)）。** 每个服务都加上 `--max-num-batched-tokens 4096`，就使用组里 4,096 token 的前向（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)“4,096 token 的前向”）：vLLM 收到 `--max-num-batched-tokens 4096`，窗口布局预留一次 4,096 token 前向的瞬态（pool 1 小 3.125 GiB，P = 16 时全局 Page 多 2,560 个），超过 4,096 token 的 phase 1 分成每次 3,056 token 的块（一次读取两个块约 8,300 token，[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)，所以读取一个块时就可能分成两次前向）。stock 对照也要加 `--max-num-batched-tokens 4096`。本节的运行都用 8,192。

**kernel。** 本 runtime 的 Gemma 服务和 stock 对照都传 `--attention-backend TRITON_ATTN`：FlashAttention 3 不支持 512 宽的 head，而 FA4（vLLM 对这个模型的默认选择）未在本 runtime 上验证（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）。没有 Gated-DeltaNet prefill。sampler（`VLLM_USE_FLASHINFER_SAMPLER=0`）、CUDA graph `full` 和纯文本日志行与 A.6 相同。

**stock 对照（Vanilla），按作业 harness 的计划。** 作业 harness 为 Gemma 准备的计划是直接启动的 `vllm.entrypoints.openai.api_server`，带 `--attention-backend TRITON_ATTN`、`--max-num-batched-tokens 8192`、`--kv-cache-memory-bytes 30064771072`（28 GiB）、`--max-num-seqs 16`、`--max-model-len 262144`、`--enable-prefix-caching --prefix-caching-hash-algo sha256`（每个 case 一个 `cache_salt`）、`--seed 42`、`--language-model-only` 和 `--distributed-executor-backend uni`，以 `selective-prefix` 的名字提供服务；没有任何 Mamba 或 Gated-DeltaNet 参数，也没有 parser（客户端把 token id 发到 `/v1/completions`）。Vanilla transport 每一轮用 Gemma 的模板重新渲染对话，16 个 session 跑 32 个 case。它与上面的 runtime 服务在 KV 大小（28 GiB 对 25 GiB）和 session 数（16 对 8）上不同；而且模板重新渲染时会去掉每个过去的模型轮的空的 thought channel，所以它的 prompt 从第一个 `<|turn>model\n` 之后就与 runtime 的不同（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）。

**状态。** 见 [docs/status.zh-CN.md](status.zh-CN.md) 的 Gemma 4 31B IT 表。

---

## Part B. 流程

这一部分用本仓库的工具运行 Part A 的配置：一个服务和客户端在 GH200 节点上的同一个 Slurm 作业里，客户端走节点的 loopback。除标注外，每一步都同时适用于两个策略和 stock 对照。客户端、case 构建程序和评分程序都是 `examples/memoryagentbench/` 的；数据集的下载和 MemoryAgentBench 的 checkout 来自本仓库之外（B.1）。

### B.1 输入

- **模型、环境和 release。** 按 docs/install.zh-CN.md [2.1](install.zh-CN.md#21-基础环境)到[2.4](install.zh-CN.md#24-安装-release) 节为该 runtime commit（A.2）构建 release；`goal_query` 还需要 Qwen3-0.6B（[docs/install.zh-CN.md 2.2 节](install.zh-CN.md#22-模型)）。
- **Case。** A.1 的 32 个 case，在辅助环境中用 `cases.py build` 从你下载的 `cxkjntmngm/MemAgent_Bench`（`a785de0f8d2511a43b9e524b5326e1105a4d218a`）和一个 MemoryAgentBench checkout 生成一次，再用 `cases.py verify` 检查（`counts_match` 为 32）：见 [benchmarks/memoryagentbench.zh-CN.md](benchmarks/memoryagentbench.zh-CN.md#步骤) 第 1 步。`$CASES` 就是这个 case 目录。`verify` 把块数和历史 token 数与 A.1 比较；还要把 `$CASES/manifest.json` 中每个 case 的 `context_sha256` 与 A.1 的列表比较。不一致的 case 不属于本配置。
- **客户端。** 运行时环境中的 `examples/memoryagentbench/run_cases.py`（A.7），由 B.3 的脚本在 GPU 节点上运行：本 runtime 用 `--mode kv`，stock 对照用 `--mode vanilla`。它把 A.7 所列的记录写进一个新的运行目录（B.6 和 B.7 读取它们）。
- **评分程序。** `HUST-AI-HYZ/MemoryAgentBench` 在 commit `538026089d1a8a8eff05121d0db89b388f360eba` 的 checkout，两个文件的哈希与 A.2 相同，以及正式测试页面的辅助环境（它的 `requirements.lock`，第 1 步），用于 `score.py`（B.6）。参考答案取自 `$CASES/manifest.json`。

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

**Dry run。** A.4 的 dry run（不加 `--start`）必须打印 A.4 的 vLLM 参数；锁页主机内存或注意力池放不下的配置会被 `serve.py` 拒绝（[docs/reference.zh-CN.md 4.9 节](reference.zh-CN.md#49-上下文上限与回复上限)）。

**评分程序。** 在该 checkout 中执行 `sha256sum utils/eval_other_utils.py llm_based_eval/summarization_evaluate.py`，必须打印 A.2 的两个哈希。

### B.3 Slurm 作业脚本

下面的脚本在一个 Slurm 作业里运行正式测试的一个服务：服务、READY、kernel 日志行、节点上的客户端，以及关闭。它由 `examples/slurm/serve_gh200.sh` 改写而来，后者不传 kv-session 参数、`--cuda-graphs`、`--active-token-budget` 和 A.4 的各个开关。这只是示例，**本身不属于 GPU 运行，且未经测试**（[docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 5.4）。存为 `mab_job.sh`：

```bash
#!/usr/bin/env bash
# One server of the MemoryAgentBench benchmark (docs/MEMORYAGENTBENCH_BENCHMARK.md) in one Slurm job.
# POLICY: observation_query | goal_query (this runtime) | stock (stock vLLM 0.24.0, the Vanilla control).
set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}" || { echo "run this under sbatch" >&2; exit 2; }
: "${REPO:?}" "${VENV:?}" "${POLICY:?}" "${RUN_ROOT:?}" "${AGENT_SECONDS:?}" "${MODEL:?}" "${CASES:?}"
PORT=${PORT:-8000}; READY_SECONDS=${READY_SECONDS:-900}

# Job-local directories and the environment of docs/install.md 3.
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
        --connector-pool-gib 1 --cuda-graphs full --decode-fastpath --no-decode-fastpath2
        --no-deferred-completion-io --language-model-only --release-idle-pages)
case $POLICY in
  observation_query) POLICY_FLAGS=(--active-token-budget 32768); MODE=kv ;;
  goal_query)        POLICY_FLAGS=(--active-token-budget 32768 --goal-rewriter-model "${REWRITER:?}"
                                   --admission-overlap 16); MODE=kv ;;
  stock)             POLICY_FLAGS=(); MODE=vanilla ;;
  *) echo "unknown POLICY: $POLICY" >&2; exit 2 ;;
esac
OUTPUT="$STAGE/$POLICY"
if [[ $POLICY == stock ]]; then
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

# READY: ready.json (this runtime, docs/running.md 4.1) or /health and /v1/models (stock).
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

# The client (Part A.7 to A.9): the 32 cases, 16 at a time, largest first, on this node's loopback.
client_rc=0
timeout "$AGENT_SECONDS" "$PY" -I -B "$REPO/examples/memoryagentbench/run_cases.py" \
  --base-url "http://127.0.0.1:$PORT" --mode "$MODE" --cases "$CASES" --tokenizer "$MODEL" \
  --workers 16 --out "$STAGE/client" ${CLIENT_SECONDS:+--agent-seconds "$CLIENT_SECONDS"} \
  > "$STAGE/client.log" 2>&1 || client_rc=$?
if (( client_rc == 124 )); then echo "agent time limit reached: the run is partial (B.5)" >&2; fi

# Stop: SIGTERM to the server PID only; this runtime must exit 0 with clean receipts (docs/design.md 8).
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

每个服务提交一个作业，使用 README 快速上手中的占位符（`REPO`、`VENV`、`MODELS`、`RELEASE`、`RUNS`）和 [docs/running.zh-CN.md 4.5 节](running.zh-CN.md#45-gpu-作业slurm)的 Slurm 设置 `ACCOUNT`、`PARTITION`、`MAIL`，设置放在提交作业的 shell 里（`--export=ALL`）。下面的例子提交 `observation_query`：

```sh
POLICY=observation_query
export REPO VENV RELEASE POLICY
export MODEL="$MODELS/qwen3.6-27b" REWRITER="$MODELS/Qwen3-0.6B-c1899de"
export RUN_ROOT=/abs/short/jobs AGENT_SECONDS=1800
export CASES="$RUNS/mab-cases"
sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus=1 \
  --cpus-per-task=16 --mem=227120M --time=00:47:00 --job-name="mab-$POLICY" \
  --output="$RUN_ROOT/slurm-%j.out" --export=ALL --mail-type=BEGIN,END,FAIL --mail-user="$MAIL" \
  /abs/path/to/mab_job.sh
```

- **`goal_query`：** `POLICY=goal_query`，`--time=01:01:00`，例如 `AGENT_SECONDS=2400`。
- **stock 对照：** `POLICY=stock`，`--mem=113560M`（一份份额：它不钉住 arena），`--time=01:27:00`，例如 `AGENT_SECONDS=3900`。
- **时间上限。** `--time` 要覆盖服务启动（使用 CUDA graph 的 MemoryAgentBench 作业中，作业里第一个服务在启动后 208–263 秒 READY：本 runtime 的 G4 服务为 238 秒和 263 秒，stock 对照为 208 秒，G2 为 243 秒，Gemma 作业中为 237 秒和 242 秒；[docs/running.zh-CN.md 4.1 节](running.zh-CN.md#41-单个服务)、[docs/reference.zh-CN.md 4.8 节](reference.zh-CN.md#48-cuda-graph)）、客户端（`AGENT_SECONDS`）和关闭。作业 harness 的 G4 作业使用这些时间上限，约为其预计墙钟时间的 1.3 倍：2,164 秒（`observation_query`）、2,804 秒（`goal_query`）和 4,024 秒（stock）；上面的 `AGENT_SECONDS` 是这些上限之内的示例取值。预计时间基于作业 harness 规划用的回复长度（A.4）。被 `AGENT_SECONDS` 或作业时间上限截断的运行是不完整的运行（A.12）。
- **客户端自己的时间上限。** `timeout` 是强制停止。在提交作业的 shell 中 export 了 `CLIENT_SECONDS`（比 `AGENT_SECONDS` 稍小，给进行中的请求留出时间）时，脚本传 `run_cases.py --agent-seconds "$CLIENT_SECONDS"`：客户端自己停止发起新请求，让进行中的请求完成并关闭它的 session（A.7、A.8），这样关闭时的检查不会看到打开的 session。
- `RUN_ROOT` 必须短：`TMPDIR` 在它下面。

### B.4 就绪检查

- **本 runtime。** READY 指 `server_checks.py ready` 返回 0（[docs/running.zh-CN.md 4.1 节](running.zh-CN.md#41-单个服务)）；`/v1/models` 有响应并不等于 READY。`dry-run.json` 中是 vLLM 命令，必须与 A.4 一致。`GET /v1/kv-session/info` 返回 method、预算和各选项（策略、页大小、预算方式和选页合同、回复上限、上下文上限、thinking、session 数）；客户端在第一个 case 之前检查 method、thinking 设置和轮次格式（A.7）。**不要在运行之外发送 complete 请求：** 它的 session id 会占用服务 64 个槽位中的一个，打开一个不属于任何 case 的 session，之后的检查都会把它算进去。
- **stock 对照。** `GET /health` 返回 200，且 `GET /v1/models` 列出 `selective-prefix`。
- **kernel 日志行。** 作业目录中的 `kernels.txt` 必须显示 A.6 的各行。

### B.5 运行

- 三个服务依次运行，每个一个作业。
- 进度：每完成一个 case，`$STAGE/client/` 下多一个 `cases/<id>/result.json`；每个请求在 `requests.jsonl` 中有一行。在本 runtime 上，服务输出目录中的 `capacity.jsonl` 列出空闲 session 的 Page 的释放和重新载入，`server.log` 中的 `CAPACITY_WAIT` 行表示有轮次在等待空间；`GET /v1/kv-session/info` 随时可以读取。
- **被截断的运行。** 这个脚本没有停滞看门狗：停滞的运行会一直持续到 `AGENT_SECONDS` 或作业时间上限。以任一种方式被截断的运行都是不完整的运行；整个服务在一个新作业里重跑（A.12）。
- 客户端的记录和服务的输出目录包含来自 case 数据的 prompt 和历史文本，不要放进公开发布。

### B.6 评测

运行之后离线进行，在 B.1 的辅助环境（正式测试页面第 1 步的 `MAB_VENV` 和 MemoryAgentBench checkout `MAB`）中使用 `examples/memoryagentbench/score.py`，`STAGE=$RUN_ROOT/<job id>`，即 B.3 的作业目录：

```sh
"$MAB_VENV/bin/python" -I -B "$REPO/examples/memoryagentbench/score.py" --run "$STAGE/client" \
  --cases "$CASES" --mab "$MAB" --stub-unused
```

它用 checkout 中的官方 `post_process`（文件的 SHA-256 与 `upstream/SHA256SUMS` 核对）对每个 case 的 prediction 和取自 `$CASES` 的参考答案评分，把每个 case 的指标和得分写进它的 `result.json`，把各能力的平均分写进 `accuracy.json`。8 个 Long-Range Understanding case（`llm_judge_f1`）需要官方评判（A.10），`score.py` 不运行它：这些 case 保持 `judge_pending`，`accuracy.json` 的 `composite_score`（即 A.11 的 Score，取值范围 0–1，乘以 100 即 A.11 的 0–100）保持为 null；`composite_score_without_judge` 取值范围同为 0–1，是另外三项能力的平均。评判需要 OpenAI API key，用官方的 `llm_based_eval/summarization_evaluate.py` 运行；没有它，Long-Range Understanding 的得分和 Score 都处于待定状态。

### B.7 吞吐

从 `run_cases.py` 的一个运行目录，即各 case 的 `result.json`，计算 A.11 的吞吐（`STAGE` 同 B.6；未经测试）：

```sh
python3 - "$STAGE/client" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(Path(sys.argv[1]).glob('cases/*/result.json'))]
tokens = sum(row['agent_output_tokens'] for row in rows)
span = max(row['finished'] for row in rows) - min(row['started'] for row in rows)
print(json.dumps(dict(cases=len(rows), output_tokens=tokens, span_s=round(span, 3), tokens_per_s=round(tokens / span, 3))))
PY
```

完整的运行有 32 行，且没有状态为 `error` 的 case；否则是不完整的运行（A.12）。
