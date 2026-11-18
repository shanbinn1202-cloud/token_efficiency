# Terminal-Bench 2.0 正式测试配置

[English](TERMINALBENCH_BENCHMARK.md) | 简体中文

本文档规定本 runtime 的 Terminal-Bench 2.0 正式测试的配置：数据集 `terminal-bench/terminal-bench-2` 在固定 ref 下的全部 89 个任务，Harbor 0.21.0 及其内置的 Terminus 2 agent（不作修改，thinking 关闭），最多 16 个 trial 同时运行（`-n 16`），`full`、`observation_query`、`random_fill` 和 `goal_query` 每个策略一个使用 chat-text 协议（`--protocol chat-text`，[docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)）的服务，另有一个 stock vLLM 0.24.0 服务作为对照。**Part A** 是正式记录的配置。baseline（stock vLLM 对照或其他任何系统）只有满足 A.12 检查表的每一项才可比；配置与 agent 在哪台机器上运行无关。**Part B** 是用本仓库工具执行的手动流程，属于可选内容：任何能通过 loopback 访问服务的 x86_64 Docker 主机都可以运行 Harbor。

**一份带日期的记录。** Part A 是组里在 2026-10-26 和 2026-11-08 定下的 Terminal-Bench 配置（thinking 关闭、同步调度、K = 20），属于历史配置：它保持原样，这样按它对齐的 baseline 仍然可比。当前配置（Qwen3.6-27B，开启 thinking 和 `--async-scheduling`，25 GiB，262,144 token，16 个 session，每个 benchmark 都用同一个 8,192 token 预算）见 [docs/status.zh-CN.md](status.zh-CN.md#当前配置)，每次 GPU 运行及其判定也在那里。

两个模型的上下文上限都是 262,144 token，GPU KV 池都是 25 GiB：这是组里 2026-10-26 定的设置，维护者已于 2026-11-08 确认（25 GiB 也是 baseline 的 benchmark 配置 `token_efficiency` `068efec` 的默认 KV 池）。Qwen3.6-27B 另配 70 GiB 的 Page arena，这是 harness 计划自己的设置（A.3）；262,144 token 也是 `examples/terminalbench/` 中的客户端给 Terminus 2 的上下文；更早的 Qwen dry run 使用 65,536 token 和 32 GiB（A.2）。Gemma 4 31B IT 另配 K = 20、4,096 token 的前向和 72 GiB 的 Page arena，不含 `full`，因为它在 262,144 token 下放不下（A.14；维护者 2026-11-08 定下的设置）；更早的 Gemma dry run 都使用 65,536 token，A.14 保留了它们实际使用的设置。选择性策略每个 session 的 KV 预算是 8,192 token（A.3），每个 benchmark 都用同一个预算（2026-11-16 决定，[docs/status.zh-CN.md](status.zh-CN.md#当前配置)）。上下文和预算各是配置中的一个取值（A.3、A.6），改动任何一个都是另一种配置。`goal_query` 使用 [docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)中面向 Terminus 2 的暂定 Goal 输入（待 Goal 负责人审阅）。本配置需要 runtime `cf2b28d` 或更新的 commit，即第一个对 Qwen 接受 `--kv-cache-gib 25` 的 launcher（A.2）。A.1 到 A.13 配置的是 Qwen3.6-27B；Gemma 4 31B IT profile（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）按 A.14 的设置运行同一个正式测试。

节号引用都写明该节所在的页面，例如 [docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)；README 的[文档地图](../README.zh-CN.md#文档地图)列出了这些页面。

> **本仓库中的客户端（integrate-1107）。** `examples/terminalbench/` 是发布仓库的 Terminal-Bench 客户端：`make_job.py`（每个模型 profile 和任务集的 Harbor 0.21.0 job）、`run_harbor.sh`（一个服务的 Harbor 运行）、`report.py`、`close_sessions.py`、`check_images.py`、`tasks.json`、job 模板和哈希锁文件，连同它的页面 [benchmarks/terminal-bench.zh-CN.md](benchmarks/terminal-bench.zh-CN.md) 和测试 `test_tb_client.py`（仅 CPU）。它给 Terminus 2 的上下文是 benchmark 服务的上下文，两个 profile 都是 262,144 token：Qwen 的服务带 `--kv-cache-gib 25`（A.3，作业 3461661 的配置，[docs/reference.zh-CN.md 4.9 节](reference.zh-CN.md#49-上下文上限与回复上限)），Gemma 的服务见 A.14。`make_job.py --context 65536`（在 `run_harbor.sh` 中为 `TB_CONTEXT=65536`）改为写出 65,536 token 的 job，即更早的 dry run 所用的上下文：对 Gemma 的 dry run，它逐字节写出作业 3433706 和 3434544 中作业 harness 的 `job.json`（端口、job 名称和 jobs 目录相同）。它的 `run_harbor.sh` 运行 `close_sessions.py check`，该检查拒绝上下文与 job 不同的本 runtime 服务（对 `--stock` job 只检查模型名）；B.2 的 `tb_agents.sh` 不做这项检查。对 Qwen，B.2 的 `render_job.py` 写出的 job 文件，与 `make_job.py --profile qwen3.6-27b` 在相同端口、job 名称和 jobs 目录下写出的文件逐字节相同（`tasks.txt` 对应 `--tasks all`，dry 子集对应 `--tasks trial16`；stock 对照加 `--stock`）；两个 profile 的 job 模板现在逐字节相同，所以 `make_job.py --profile gemma4-31b-it` 写出的是同一个文件。[docs/status.zh-CN.md](status.zh-CN.md) 中列出的 dry run 是通过作业 harness 自己的脚本运行 Harbor 的，job 设置相同，没有经过这个客户端。

## 验证状态

正式测试之前，用一次 16 个任务的 dry run 在一张 GH200 上检验一个策略：一个全新的服务，使用 A.3 的服务参数和 A.6 的 agent 配置，在 dry 子集（A.1；每个任务的 agent 超时都是 900 秒）的 16 个任务上以 `-n 16` 运行，经由 B.4 的隧道。对每个服务，dry run 检查：

- **Harbor 的运行。** 每个计划内的 trial 都在该次运行的时间上限内完成（包括 verifier）；没有容量停滞（A.11）。
- **服务。** READY 和启动记录：chat-text 协议 `chat_text_v1` 及其槽位数、空闲超时、驱逐等待时间、生成头和模板开关，K = 20 个同时运行的 session，KV 字节数，解耦布局及其各个池，`--release-idle-pages`，对选择性策略还有带 8,192 token 预算的选择契约，对 `goal_query` 还有 Goal rewriter 的固定值、admission 记录（`--admission-overlap 16`）、rewriter 批处理及其 token 上限，以及暂定的 Goal 输入记录 `chat_text_goal`。然后是 A.5 的 kernel 日志行、CUDA graph 模式和捕获尺寸、按配置开启的各解码快速路径和 `--deferred-completion-io`、1 GiB 的 connector 池，以及纯文本模式（有 vLLM 的纯文本日志行，没有多模态预热）。
- **chat-text 证据。** 每个 `runtime.jsonl` 行的 `chat_text` 记录；`chat_sessions.jsonl`（bind、rebind 和 close 行按顺序，槽位从不复用，最多 20 个同时打开）；引擎的 session 行；每个请求的 completion 不超过其有效上限，`finish_reason` 为 `length` 时恰好等于该上限，prompt 加回复不超过服务的上下文上限。使用 `goal_query` 时，每一轮的 Goal 输入和 Goal 来源（`chat_text.goal_inputs` 和 `goals.jsonl` 的 `inputs`，[docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)）符合固定记录中的规则。
- **空闲 Page 的释放。** 每个行和 native 文件的 `idle_release` 记录，以及 `capacity.jsonl` 中的释放和重新载入事件，与 KV 池和 `engine-cleanup.json` 中的合计对照（[docs/reference.zh-CN.md 4.13 节](reference.zh-CN.md#413-释放空闲-session-的-page需显式开启)）。
- **失败。** 只允许上下文超限的拒绝（admission 之前的 HTTP 400），以及客户端在 agent 超时时取消的请求；没有任何抢占。
- **关闭。** 作业关闭了运行遗留的打开 session（A.8）之后：关闭时没有打开的 session，清理回执已 join，进程自然退出且退出码为 0。
- **逐请求、token 级精确的回执。** 每个 prompt 的 token 流符合 chat-text 规则（新的或重建的槽位把整段对话交给固定模板渲染；续接的请求追加上一条回复的原始 id、边界、一条 user 消息和生成头），以及轮次几何、选择（预算和投影）、该请求自身的 I/O、admission 日志、KV 池、Page 证据、Gated-DeltaNet 字节数（恢复的和重新载入的 Page）和阶段 1 的模式。
- **取回之后的对账。** 每个 trial 的 ATIF trajectory 按 `key_sha256`（A.8）与服务端的行对账：调用数等于行数，每次调用的 prompt 和 completion token 数与服务端的相同（被 agent 超时打断的请求见 A.10）。

**结果。** 已经运行过的 dry run，连同每个作业的 runtime 提交、上下文、设置和判定，见 [docs/status.zh-CN.md](status.zh-CN.md)；本节到 2026-11-16 为止给出的它们当时的记述，冻结在 [docs/history/terminalbench-validation.zh-CN.md](history/terminalbench-validation.zh-CN.md)，关于这些作业的更多说明见[历史](history.zh-CN.md#历史)、[验证记录](history/validation-cf2b28d.zh-CN.md)，以及从 [running.md](history/running-gpu-notes.zh-CN.md) 和 [reference.md](history/reference-gpu-notes.zh-CN.md) 移出的作业说明。更早的 dry run 使用的大小和提交与 Part A 不同：A.2 列出它们的提交，A.14 列出 Gemma 按实际运行的设置。它们用的是作业 harness，服务参数、Harbor 作业配置、隧道和关闭 session 的方式与本文档相同；Part B 的脚本没有按所写的样子运行过（[docs/known-limits.zh-CN.md](known-limits.zh-CN.md#54-各-benchmark-页面-b-部分的作业脚本未经测试) 5.4）。本页不报告任何结果。

## 与 baseline 对齐的配置

我们的运行所用的取值，供 baseline 照样设置；每个取值都取自 Part A（完整检查表见 A.12，仍然存在的差异见 A.13）：

| 项目 | 取值 |
| --- | --- |
| 模型 | `Qwen/Qwen3.6-27B`，revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`，一张 GH200 120GB，官方 vLLM 0.24.0 wheel（A.2） |
| Thinking | 关闭：固定的 chat template，`enable_thinking=false`、`preserve_thinking=true`，生成头中为空的 think 块；stock baseline 每次调用发送 `extra_body: {"chat_template_kwargs": {"enable_thinking": false}}`（Terminus 2 的 `llm_call_kwargs`；A.3、A.4） |
| 解码 | 贪心，由服务端执行；stock baseline 发送 `temperature` 0.0。vLLM `--seed 42`。不设停止字符串：回复在 EOS/pad 或 `max_tokens` 处结束（A.3、A.4） |
| 上下文上限 | `--max-model-len 262144` 以及 Terminus 的 `max_input_tokens` 262144（A.3、A.6） |
| 回复上限 | 4096：客户端的 `max_tokens` 4096，不超过本 runtime 的 `--max-tokens 4096`（A.3、A.7） |
| 每个 session 的 KV 预算 | `observation_query`、`random_fill` 和 `goal_query` 用 `--active-token-budget 8192`，每个 benchmark 都用同一个预算（[docs/status.zh-CN.md](status.zh-CN.md#当前配置)）；`full` 保留所有 Page（A.3） |
| 并发 | 89 个 trial，最多 16 个同时运行（`-n 16`）；本 runtime 最多打开 K = 20 个 session（`--max-num-seqs 20`），stock 对照用 `--max-num-seqs 16`（A.1、A.3、A.4） |
| 本 runtime 的 KV cache | 25 GiB GPU KV：`--kv-cache-gib 25`，26,843,545,600 字节；主机上 70 GiB 的锁页 Page arena（`--dram-arena-gib 70`）和 1 GiB 的 connector 池（A.3） |
| Goal rewriter（`goal_query`） | 同一张 GPU 上的 `Qwen/Qwen3-0.6B`，每批最多 4 个调用（`--goal-rewriter-batch 4`，每批填充后的 prompt token 默认上限 16,384），配合 `--admission-overlap 16`；使用 [docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)面向 Terminus 2 的暂定 Goal 输入（A.2、A.3） |
| stock 对照（vLLM） | `--kv-cache-memory-bytes 26843545600`（25 GiB）、`--max-model-len 262144`、`--max-num-batched-tokens 16384`、`--max-num-seqs 16`；CUDA graph：vLLM 默认（`FULL_AND_PIECEWISE`）；异步调度：vLLM 默认（开启）；`--enable-prefix-caching --prefix-caching-hash-algo sha256`；不加 `--attention-backend`（vLLM 选择 FLASH_ATTN 加 FlashAttention 3）；`--language-model-only`（A.4、A.5） |
| 吞吐 | 89 个计划内任务的 trial 中主 ATIF trajectory（含续写）的 completion token 之和，摘要子调用单列 / agent 主机时钟上各 agent 的时间跨度，从最早的 `agent_execution.started_at` 到最晚的 `agent_execution.finished_at`。包含跨度内的工具执行、各种等待、摘要以及其他 trial 的环境准备和 verifier；不包含服务的启动和关闭、第一个 agent 之前的环境准备、最后一个 agent 之后的 verifier，以及 Goal rewriter 的 token（A.10） |
| 任务 | Terminal-Bench 2.0：`terminal-bench/terminal-bench-2` 在 ref `sha256:c6fc2e23…` 下的全部 89 个任务；Harbor 0.21.0 及其内置的 Terminus 2，版本 2.0.0，不作修改（litellm 1.97.0、openai 2.54.0）（A.1、A.2、A.6） |
| 评测 | 每个 trial 中紧接 agent 运行的 Harbor verifier；reward 1.0 为已解决（A.9） |
| Gemma 4 31B IT | `google/gemma-4-31B-it`（A.14）：上述设置，同样是 262,144 token 的上下文、25 GiB 的 GPU KV 和 K = 20，加上 Gemma profile 自己的 4,096 token 前向（`--max-num-batched-tokens 4096`，stock 对照也用）、72 GiB 的 arena、显式的 Goal rewriter 批上限（20,480）、`observation_query`、`random_fill` 和 `goal_query` 三个策略（没有 `full`）以及使用 `TRITON_ATTN` 的 stock 对照。它更早的 dry run 在 65,536 token、K = 17、8,192 token 的前向和 64 GiB 的 arena 下运行（A.14 保留了这些按实际运行的设置） |

---

## Part A. 配置

### A.1 范围与任务

| 项目 | 取值 |
| --- | --- |
| 模型 | Qwen3.6-27B，一张 NVIDIA GH200 120GB，vLLM 0.24.0（A.2） |
| 基准 | Terminal-Bench 2.0：数据集 `terminal-bench/terminal-bench-2` 在固定 ref（A.2）下的**全部 89 个任务**，每个任务使用自己的 agent 超时和 verifier 超时 |
| Agent | Harbor 0.21.0 及其内置的 Terminus 2，不作修改，thinking 关闭（A.6） |
| 并发 | 89 个 trial，每个任务一个，**最多 16 个同时运行**（`-n 16`）。一个 trial 结束后，Harbor 按它自己的顺序启动下一个 |
| 服务 | 每个配置一个全新的服务，各自一个 GPU 作业：`full`、`observation_query`、`random_fill`、`goal_query`（本 runtime），以及 `stock`（stock vLLM 0.24.0，对照） |
| 运行 | 一个服务把 89 个 trial 各跑一次，中间不中断（A.11）。一次运行就是一个样本（A.13） |

89 个任务名，按名称排序（B.1 把它存为 `tasks.txt`；Harbor 的过滤条件在每个名字前加 `terminal-bench/`）：

```text
adaptive-rejection-sampler
bn-fit-modify
break-filter-js-from-html
build-cython-ext
build-pmars
build-pov-ray
caffe-cifar-10
cancel-async-tasks
chess-best-move
circuit-fibsqrt
cobol-modernization
code-from-image
compile-compcert
configure-git-webserver
constraints-scheduling
count-dataset-tokens
crack-7z-hash
custom-memory-heap-crash
db-wal-recovery
distribution-search
dna-assembly
dna-insert
extract-elf
extract-moves-from-video
feal-differential-cryptanalysis
feal-linear-cryptanalysis
filter-js-from-html
financial-document-processor
fix-code-vulnerability
fix-git
fix-ocaml-gc
gcode-to-text
git-leak-recovery
git-multibranch
gpt2-codegolf
headless-terminal
hf-model-inference
install-windows-3-11
kv-store-grpc
large-scale-text-editing
largest-eigenval
llm-inference-batching-scheduler
log-summary-date-ranges
mailman
make-doom-for-mips
make-mips-interpreter
mcmc-sampling-stan
merge-diff-arc-agi-task
model-extraction-relu-logits
modernize-scientific-stack
mteb-leaderboard
mteb-retrieve
multi-source-data-merger
nginx-request-logging
openssl-selfsigned-cert
overfull-hbox
password-recovery
path-tracing
path-tracing-reverse
polyglot-c-py
polyglot-rust-c
portfolio-optimization
protein-assembly
prove-plus-comm
pypi-server
pytorch-model-cli
pytorch-model-recovery
qemu-alpine-ssh
qemu-startup
query-optimize
raman-fitting
regex-chess
regex-log
reshard-c4-data
rstan-to-pystan
sam-cell-seg
sanitize-git-repo
schemelike-metacircular-eval
sparql-university
sqlite-db-truncate
sqlite-with-gcov
torch-pipeline-parallelism
torch-tensor-parallelism
train-fasttext
tune-mjcf
video-processing
vulnerable-secret
winning-avg-corewars
write-compressor
```

agent 超时（每个任务的 `[agent] timeout_sec`）：600 秒 × 1、750 秒 × 1、900 秒 × 49、1,200 秒 × 6、1,800 秒 × 16、2,400 秒 × 2、3,600 秒 × 12、7,200 秒 × 1（`sam-cell-seg`）和 12,000 秒 × 1（`build-pov-ray`）。verifier 超时从 360 到 12,000 秒。每个任务都允许访问外网；84 个任务用 1 个 CPU，71 个任务的内存为 2 GB，没有任务使用 GPU。

**dry 子集**（16 个 agent 超时为 900 秒的任务，从此前 Terminus 2 运行中 prompt 最大的任务里挑选；是正式测试之前的检查，不属于正式测试）：

```text
build-cython-ext
build-pmars
configure-git-webserver
crack-7z-hash
db-wal-recovery
extract-elf
fix-code-vulnerability
gcode-to-text
git-multibranch
make-doom-for-mips
password-recovery
pytorch-model-cli
raman-fitting
sanitize-git-repo
tune-mjcf
vulnerable-secret
```

本 runtime 在 Terminal-Bench 上的策略（README 开头的策略表和 [docs/design.zh-CN.md 第 5 节](design.zh-CN.md#5-query-策略的选页规则)）：

| `--policy` | 下一轮使用哪些历史 Page |
| --- | --- |
| `full` | 所有历史 Page：参照组，也是最接近 stock vLLM 的配置 |
| `observation_query` | Selector V2：在预算内保留必保留页、recent 页和余弦最高的页；query 是最新 observation 末尾的 hidden state |
| `random_fill` | `observation_query` 的对照：预算、必保留页和 recent 页相同，剩余容量按带种子的随机顺序填满 |
| `goal_query` | 同一个 selector；query 是持续维护的检索 Goal，由 Qwen3-0.6B rewriter 更新，使用 [docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)面向 Terminus 2 的暂定 Goal 输入，待 Goal 负责人审阅 |

只有当一个 session 已封存的历史超过 8,192 token 时才开始选择（[docs/design.zh-CN.md 第 5 节](design.zh-CN.md#5-query-策略的选页规则)）；历史始终更短的 trial 在每个策略下都等同于 `full`。

### A.2 模型与版本

| 项目 | 取值 |
| --- | --- |
| 模型 | `Qwen/Qwen3.6-27B`，revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`，本地目录，必须是普通文件；元数据的 SHA-256 见 [docs/install.zh-CN.md 2.2 节](install.zh-CN.md#22-模型) |
| Goal rewriter（仅 `goal_query`） | `Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`，在同一张 GPU 上运行一个进程；同一组对比中每个 `goal_query` 服务使用同一个解析后的路径（[docs/install.zh-CN.md 2.2 节](install.zh-CN.md#22-模型)） |
| vLLM | 官方 wheel `vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl`，未作任何修改，SHA-256 `700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5` |
| Python 环境 | CPython 3.12、`transformers==5.13.0`、`tokenizers==0.22.2`，用 `examples/env/constraints-3270721.txt` 构建（[docs/install.zh-CN.md 2.1 节](install.zh-CN.md#21-基础环境)）。stock 对照从同一个环境运行；本 runtime 安装到单独的 release 目录，环境本身保持不变（[docs/install.zh-CN.md 2.4 节](install.zh-CN.md#24-安装-release)） |
| Runtime | 本仓库的 commit `cf2b28d`（`cf2b28d3bb5f74a477147cabe499fe53168a36f3`，integrate-1026；release check 通过，1,495 个测试），即作业 3461661 的 commit，也是 harness 为其 262,144 token 计划固定的 commit，或者不改变 chat-text 协议、容量门控、Goal 构造和 A.3 中 vLLM 参数的更新 commit。更早的 commit 对 Qwen 拒绝 `--kv-cache-gib 25`。65,536 token 下的 dry run 用的是更早的 commit：`goal_query` 用 `53e3b7a`（2026-10-24 那批 dry run 固定的 commit；更早的 commit 可能在 Goal 特征中停滞，见[历史](history.zh-CN.md#历史)），`full` 和 `observation_query` 用 `68d19b4`。`0d37346` 及更早的 commit 在没有请求运行时不经检查就接纳一轮（A.3），不再使用。在 `cf2b28d` 上（与 `53e3b7a` 相同），chat-text 协议模块 `source-overlay/remote_ra_live_v2/chat_text_protocol.py` 的 SHA-256 为 `cd13187612fe4e312b90450c95cc96763e0eed7a50b1d2adf878edd26d2533c7`（在 `68d19b4` 上为 `873ce935ddf91a4f3247469c177c7506ffabbed5649fdb4af29ecb87fe9bb338`）；commit `7b70e14`（integrate-1107）上是另一个版本 `2b492e676bb5807e101e84f8ce66afa4ff8b58220aee05294b338e3a7236c8f8`，由 thinking 开启（docs/running.zh-CN.md [6.8](running.zh-CN.md#68-开启-thinking第一阶段--enable-thinking)、[6.9](running.zh-CN.md#69-开启-thinking-并丢弃-reasoning--thinking-history-strip) 节）、`recent_fill` 和 `akv_rule` 的 commit 改动（这三项在本配置中都不开）；之后运行所用的 commit 见 [docs/status.zh-CN.md](status.zh-CN.md) |
| GPU | 一张 NVIDIA GH200 120GB，Linux aarch64；一张 GPU 同一时间只跑一个服务 |
| Slurm 资源 | `--nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16 --mem=113560M`（节点主机内存中一张 GH200 的份额；A.3 说明为什么够用） |
| Agent 软件 | Harbor `0.21.0`，从其 wheel 安装，未作任何修改（`harbor-0.21.0.dist-info/RECORD` 的 SHA-256 为 `abfaf4e154a499d7e3c8c83ba85a8ee8861a99f66b11b7d7cb035f10b85d1424`，414 个文件），其内置的 Terminus 2 版本 `2.0.0`，litellm `1.97.0`，openai `2.54.0`，Python 3.12 |
| 数据集 | `terminal-bench/terminal-bench-2`，ref `sha256:c6fc2e2382c1dbae99b2d5ecd2f4f4a60c3c01e0d84642d69b4afd92e99d078b` |
| 评测程序 | Harbor 自己的 verifier，在每个 trial 内部运行（A.9） |

### A.3 本 runtime：服务参数与 vLLM 参数

**四个策略共用的参数**（`serve.py`）：

| 参数 | 取值 | 作用 |
| --- | --- | --- |
| `--protocol` | `chat-text`，加 `--chat-text-slots 512 --chat-text-idle-timeout 600 --chat-text-evict-after 30`（均为默认值，显式固定） | 面向 Terminus 2 的普通 OpenAI chat 文本（`chat_text_v1`，[docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)）。不接受 `--task`：512 个槽位惰性绑定到客户端 session（89 个 trial、每次摘要的 "questions" 调用各一个 session，以及 rebind） |
| `--max-running-sessions` | `20` | K：同时最多打开 20 个 session。Harbor 同时运行 16 个 trial；另外 4 个空余 session 容纳结束的 trial（Terminus 不发送关闭请求）和摘要子调用（A.8）。vLLM 收到 `--max-num-seqs 20` |
| `--kv-cache-gib` | `25` | vLLM 收到 `--kv-cache-memory-bytes 26843545600`。K = 20 时：19,531 个注意力 block 和 124 个 Mamba block，可用 19,530 个；launcher 要求上下文上限处的一轮放得下，即 17,861 个 block（[docs/reference.zh-CN.md 4.9 节](reference.zh-CN.md#49-上下文上限与回复上限)） |
| `--page-tokens` / `--kv-layout` | `16` / `decoupled` | 16 token 的注意力 Page，以及一个单独的、有 41 个 slot（2K + 1）的 Gated-DeltaNet 池（[docs/reference.zh-CN.md 4.6 节](reference.zh-CN.md#46-解耦-kv-布局与-16-token-的-page)）。vLLM 收到 `--block-size 16 --mamba-block-size 784 --additional-config ...` |
| `--max-model-len` | `262144` | 上下文上限（prompt 加回复）：模型的原生上下文。请求 4096 token 回复时，prompt 最多 262,144 − 4,096 = 258,048 token（[docs/reference.zh-CN.md 4.9 节](reference.zh-CN.md#49-上下文上限与回复上限)） |
| `--max-tokens` | `4096` | 回复上限：客户端的 `max_tokens` 取 1 到 4096（Terminus 发送 4096，A.6） |
| `--dram-arena-gib` | `70` | 所有 session 共用一个 70 GiB 的锁页主机 Page arena：71,680 个 1 MiB 的 Page，例如 20 个 57K token 的 session，或 4 个处于上限的 session。超出它的捕获会让该请求以 `DRAM_ARENA_EXHAUSTED` 失败（[docs/reference.zh-CN.md 4.9 节](reference.zh-CN.md#49-上下文上限与回复上限)） |
| `--connector-pool-gib` | `1` | connector 闲置的 CPU 池设为 1 GiB 而不是 32 GiB（[docs/reference.zh-CN.md 4.9 节](reference.zh-CN.md#49-上下文上限与回复上限)）。vLLM 收到 `cpu_bytes_to_use` 1073741824 |
| `--cuda-graphs` | `full` | vLLM 收到 `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}`，取代 `--enforce-eager`（[docs/reference.zh-CN.md 4.8 节](reference.zh-CN.md#48-cuda-graph)） |
| `--decode-fastpath` / `--decode-fastpath2` / `--decode-fastpath3` | 开启 / 开启 / 开启 | 每个解码步的主机工作更少，结果相同；每一阶段都需要前一阶段（[docs/reference.zh-CN.md 4.11 节](reference.zh-CN.md#411-解码快速路径默认开启)）。不增加 vLLM 参数 |
| `--deferred-completion-io` | 开启 | 已完成请求的 native 文件和 `goals.jsonl` 行在该步的输出之后由一个写线程写入，字节相同（[docs/reference.zh-CN.md 4.11 节](reference.zh-CN.md#411-解码快速路径默认开启)）。不增加 vLLM 参数 |
| `--language-model-only` | 开启 | 不加载视觉塔，没有多模态预热（[docs/reference.zh-CN.md 4.12 节](reference.zh-CN.md#412-纯文本--language-model-only默认开启)）。vLLM 收到 `--language-model-only` |
| `--release-idle-pages` | 开启 | GPU 紧张时，容量门控释放空闲 session 的常驻 Page，并在它们的下一轮之前重新载入（[docs/reference.zh-CN.md 4.13 节](reference.zh-CN.md#413-释放空闲-session-的-page需显式开启)；见下）。不增加 vLLM 参数 |
| `--port` | `8000` | 服务只绑定 `127.0.0.1` |

**空闲 session 与 `--release-idle-pages`。** 两轮之间，每个打开的 session 都在 GPU 上保留它的常驻 Page 和 live tail（[docs/reference.zh-CN.md 4.13 节](reference.zh-CN.md#413-释放空闲-session-的-page需显式开启)）。Terminus 不发送关闭请求，所以最多 20 个 session 一直打开，其中大多数是空闲的（已结束的 trial、正在执行长命令的 trial、"questions" session），它们的 Page 会占满注意力池：在第一次 dry run 中（不开这个选项的 `full`，32 GiB，65,536 token），空闲 session 钉住了 26,698 个注意力 block 中除 69 个以外的全部（[docs/running.zh-CN.md 4.4 节](running.zh-CN.md#44-一个服务上的多个-session)）。在 262,144 token、25 GiB 下，池连两个处于上限的 session 都放不下（每个 16,385 个 Page，可用 block 为 19,530 个），所以本配置依靠这个选项和容量门控；作业 3461661 两者都没有用上（没有释放，没有容量等待）。不开这个选项时，容量门控只能等运行中的请求结束；一旦所有 session 都空闲，就没有任何东西能释放 block。开了它，门控用释放空闲 session 的全部常驻 Page 来补足一轮的缺口，最久未用的先释放（如果所有空闲 session 加起来也补不足，就什么都不释放）。它们的主机副本留在 Page arena 中，所以不增加锁页主机内存；被释放的 session 在下一次阶段 1 之前从这些副本重新载入 Page，那一轮的运行就像它们从未离开过一样。从 `68d19b4` 起，仍然放不下的一轮会等待（`CAPACITY_WAIT`），没有其他请求运行时也一样；此前，这样的一轮因为没有请求在运行而不经检查被接纳，它的 decode 用掉最后一个空闲 block，vLLM 抢占了它，引擎随之结束（[docs/running.zh-CN.md 4.4 节](running.zh-CN.md#44-一个服务上的多个-session)）。如果什么都腾不出空间，每一轮都会等待，服务停滞（A.11）。

**主机内存。** 锁页主机内存是 1 GiB 的 connector 池、20 个 Gated-DeltaNet carry（2.87 GiB）和 70 GiB 的 arena，共 73.87 GiB。不设 arena 时，20 个处于上限的 session 最多可能锁页 324.1 GiB，超过 launcher 128 GiB 的锁页主机预算，launcher 会拒绝（[docs/reference.zh-CN.md 4.9 节](reference.zh-CN.md#49-上下文上限与回复上限)）；`--release-idle-pages` 不增加任何锁页内存。再留约 32 GiB 给进程，共 105.9 GiB，放得进一张 GH200 的份额 `--mem=113560M`（110.9 GiB）。

**各策略的参数：**

| 策略 | selector 参数 | admission 参数 | Goal rewriter |
| --- | --- | --- | --- |
| `full` | 无 | 无 | 无 |
| `observation_query` | `--active-token-budget 8192` | 无 | 不启动（[docs/design.zh-CN.md 第 5 节](design.zh-CN.md#5-query-策略的选页规则)） |
| `random_fill` | `--active-token-budget 8192`（填充种子：默认值 42） | 无 | 不启动（[docs/design.zh-CN.md 第 5 节](design.zh-CN.md#5-query-策略的选页规则)） |
| `goal_query` | `--active-token-budget 8192` `--goal-rewriter-model <Qwen3-0.6B 目录> --goal-rewriter-batch 4` | `--admission-overlap 16` | 同一张 GPU 上一个 Qwen3-0.6B 进程（约 2.5–2.8 GiB：`serve.py` 把它的权重和上下文按 2,519 MiB 估算，harness 的 GPU 估算用 2,836 MiB，见 A.14）。每批最多 4 个调用，且在每批填充后的 prompt token 默认上限 16,384 之内（不传 `--goal-rewriter-batch-tokens`；`startup.json` 和 `ready.json` 固定 `goal_rewriter_batch_tokens`）。rewriter 的批量贪心解码与逐个调用并非逐位一致，所以只在该设置相同的服务之间比较策略（[docs/reference.zh-CN.md 4.10 节](reference.zh-CN.md#410-admission-优化需显式开启)）。Goal 输入：[docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)面向 Terminus 2 的暂定规则（定义 `terminus2-goal-inputs-provisional-v2`），待 Goal 负责人审阅 |

三个选择性策略的预算是同一个取值，每个 benchmark 也都相同（[docs/status.zh-CN.md](status.zh-CN.md#当前配置)）。任何服务都不设置 `--gdn-device-handoff`、`--phase1-in-step`、`--goal-rewriter-batch-tokens`、`--active-ratio` 或 `--fill-seed`；只有 `goal_query` 设置 `--admission-overlap`、`--goal-rewriter-model` 和 `--goal-rewriter-batch`（[docs/reference.zh-CN.md 4.10 节](reference.zh-CN.md#410-admission-优化需显式开启)：16 个 session 的 `goal_query` 用 `--admission-overlap 16`）。预算、各解码快速路径、`--deferred-completion-io`、`--release-idle-pages` 以及 Goal 和 admission 参数会固定到启动配置里，但不增加任何 vLLM 参数，所以四个服务启动 vLLM 时的参数完全相同。

**命令。** 四个服务的 dry run（加 `--start` 即启动服务；`LABEL` 是每次尝试的新标签）：

```sh
SHARED=(--protocol chat-text --chat-text-slots 512 --chat-text-idle-timeout 600 --chat-text-evict-after 30
        --kv-cache-gib 25 --max-model-len 262144 --page-tokens 16 --kv-layout decoupled --cuda-graphs full
        --max-running-sessions 20 --max-tokens 4096 --dram-arena-gib 70 --connector-pool-gib 1
        --decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io
        --language-model-only --release-idle-pages)
python -B serve.py --release-dir "$RELEASE" --policy full "${SHARED[@]}" \
  --output "$RUNS/tb-full-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --policy observation_query "${SHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/tb-observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --policy random_fill "${SHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/tb-random_fill-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --policy goal_query "${SHARED[@]}" \
  --active-token-budget 8192 --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --admission-overlap 16 \
  --goal-rewriter-batch 4 --output "$RUNS/tb-goal_query-$LABEL" --port 8000
```

**由此得到的 vLLM 参数**（四个服务相同；以 dry run 打印的列表为准，这里按参数组换行；作业 3461661 的 dry run 打印的就是这个列表）：

```text
--model <model directory> --served-model-name selective-prefix --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
--host 127.0.0.1 --port 8000 --distributed-executor-backend uni --no-async-scheduling
--tensor-parallel-size 1 --pipeline-parallel-size 1
--max-num-seqs 20 --max-model-len 262144 --max-num-batched-tokens 16384
--kv-cache-memory-bytes 26843545600 --seed 42 --enable-prefix-caching --prefix-caching-hash-algo sha256
--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}
--mamba-cache-mode align --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
--chat-template <output>/chat_template.jinja --chat-template-content-format string
--kv-transfer-config {"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":1073741824}}
--gdn-prefill-backend triton
--block-size 16 --mamba-block-size 784 --additional-config {"remote_ra_kv_layout":{"mamba_slots":41,"mamba_tokens":784,"page_tokens":16}}
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

**生成约定，由服务端执行**（[docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)）。服务端忽略客户端的采样字段：
- 贪心解码，最多生成客户端 `max_tokens` 个新 token（1 到 4096；没给时取 4096，并按剩余上下文截断），只在 EOS/pad 处停止（没有停止字符串）；在 EOS/pad 处 `finish_reason` 为 `stop`，达到上限时为 `length`；
- 关闭 thinking：参照渲染是固定 chat template 在 `enable_thinking=false`、`preserve_thinking=true` 下的结果；生成头为 `<|im_start|>assistant\n<think>\n\n</think>\n\n`。Terminus 不发送 system 消息，所以初始段就是它的第一条 user 消息；
- 一个请求如果等于上一个请求加上一条回复（`strip()` 后比较）再加一条新 user 消息，就续接 session：上一条回复保留生成时的原始 token ID，后接 `<|im_end|>\n`（EOS/pad 之后只接 `\n`）和新的 user 消息。其他请求（摘要交接、回退后的历史、重发的请求）都从文本重建 session；
- 响应内容是解码后的回复（跳过特殊 token），`usage` 给出服务端自己统计的 prompt 和 completion token 数；
- prompt 加上请求的回复长度超过 262,144 token 的请求在 admission 之前返回 HTTP 400，正文是 vLLM 自己的文字 `This model's maximum context length is …`；服务端什么也不变。

### A.4 stock vLLM 对照

对照服务就是从同一个环境（A.2）直接启动的 `vllm.entrypoints.openai.api_server`，不带本 runtime 的任何代码。

| 设置 | 取值 | 与本 runtime 的关系 |
| --- | --- | --- |
| `--model` | 同一个 Qwen3.6-27B 目录 | 相同 |
| `--served-model-name` | `selective-prefix` | 同名，因此 Terminus 使用相同的模型名 |
| `--host` / `--port` | `127.0.0.1` / `8000` | 相同（只用 loopback） |
| `--distributed-executor-backend` / `--tensor-parallel-size` | `uni` / `1` | 相同 |
| `--max-num-seqs` | `16` | 等于 Harbor 的 `-n`。不同：本 runtime 为 K = 20（A.13） |
| `--max-model-len` | `262144`（与 A.3 是同一个取值） | 相同 |
| `--max-num-batched-tokens` | `16384` | 相同（由 launcher 固定） |
| `--kv-cache-memory-bytes` | `26843545600`（25 GiB） | 字节数相同 |
| `--seed` | `42` | 相同 |
| `--enable-prefix-caching --prefix-caching-hash-algo sha256` | 开启 | 相同 |
| `--language-model-only` | 开启 | 相同（[docs/reference.zh-CN.md 4.12 节](reference.zh-CN.md#412-纯文本--language-model-only默认开启)） |
| `--mamba-cache-mode` | `align` | 相同 |
| `--gdn-prefill-backend` | `triton` | 相同 |
| CUDA graph | vLLM 默认：既不加 `--enforce-eager` 也不加 `--compilation-config`，结果是 `FULL_AND_PIECEWISE` 加 `VLLM_COMPILE`（[docs/reference.zh-CN.md 4.8 节](reference.zh-CN.md#48-cuda-graph)） | 模式相同；捕获尺寸最大 32，本 runtime 为 40（A.5） |
| 异步调度 | vLLM 默认（开启），不加参数 | 不同：launcher 默认传 `--no-async-scheduling`；它的可选项 `--async-scheduling` 这里不用（A.13） |
| 注意力后端 | 不设置，由 vLLM 选择（A.5） | 相同 |
| Chat template | 模型自带的模板；服务端使用 vLLM 默认的 chat template 参数 | 历史的渲染不同（A.13） |
| Thinking | 关闭，由客户端在每次调用时请求（A.6） | 生成头相同 |

其余每个取值都显式设置，这样对照服务不依赖 vLLM 对这个混合模型的默认值；CUDA graph 和异步调度保持 vLLM 的默认值（见上）。只服务于 runtime 自身路由的参数（`--revision`、reasoning 和 tool parser、`--chat-template`、`--kv-transfer-config`、`--block-size`、`--mamba-block-size`、`--additional-config`）不传：stock 服务使用模型自带的 chat template，回复以普通的 `content` 返回。

**命令**，在设置了 A.3 环境变量的 shell 中：

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
"$VENV/bin/python" -I -B -m vllm.entrypoints.openai.api_server \
  --model "$MODELS/qwen3.6-27b" --served-model-name selective-prefix --host 127.0.0.1 --port 8000 \
  --distributed-executor-backend uni --tensor-parallel-size 1 \
  --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 16384 \
  --kv-cache-memory-bytes 26843545600 --seed 42 \
  --enable-prefix-caching --prefix-caching-hash-algo sha256 --language-model-only \
  --mamba-cache-mode align --gdn-prefill-backend triton
```

- **环境变量。** 与 A.3 相同（`PATH`、`LD_LIBRARY_PATH`、短 `TMPDIR`、`VLLM_USE_FLASHINFER_SAMPLER=0`、unset `CC` 和 `VLLM_RPC_BASE_PATH`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`PYTHONDONTWRITEBYTECODE=1`），另加 `serve.py` 为本 runtime 设置的两个变量（`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`VLLM_USE_V2_MODEL_RUNNER=0`）和 `VLLM_NO_USAGE_STATS=1`。
- **生成约定，由客户端发送。** stock vLLM 从请求中读取采样设置，所以由 agent 的 kwargs 携带（A.6）：`temperature` 0.0、`max_tokens` 4096、没有停止字符串（回复在 EOS 或达到 `max_tokens` 时结束，后者的 `finish_reason` 为 `length`），以及每次调用都带 `extra_body: {"chat_template_kwargs": {"enable_thinking": false}}`（Terminus 2 的 `llm_call_kwargs`）。在 `enable_thinking=false` 下，模型的 chat template 预期以 A.3 的生成头结束 prompt，对没有 system 消息的 user 消息不添加任何文字，并且渲染历史回复时不带空的思考块；B.3 检查这几点。
- **就绪判断。** `GET /health` 返回 200，且 `GET /v1/models` 列出 `selective-prefix`。
- **关闭 session。** stock vLLM 没有关闭路由，也没有谁发送关闭请求：Terminus 不发送，作业的关闭步骤（A.8）只在本 runtime 上执行。
- **抢占。** KV 池满时 stock vLLM 可能抢占请求；本 runtime 从不抢占（A.13）。在运行期间和结束时记录 vLLM 的计数器 `vllm:num_preemptions_total`（`GET /metrics`）；harness 对 stock 服务的检查要求它保持为 0。
- **运行情况。** 已经运行过的 stock 对照及其 thinking 设置见 [docs/status.zh-CN.md](status.zh-CN.md)。

### A.5 kernel 后端

两种服务都不传 `--attention-backend`；注意力后端由 vLLM 在 GH200 上自动选择。两种服务使用相同的 kernel：

| 组件 | 设置 | 服务日志行 |
| --- | --- | --- |
| 注意力 | FLASH_ATTN 加 FlashAttention 3（vLLM 自动选择） | `Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']` 和 `Using FlashAttention version 3` |
| Gated-DeltaNet prefill | `--gdn-prefill-backend triton` | `Using Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128)` |
| Sampler | 原生 top-k/top-p（`VLLM_USE_FLASHINFER_SAMPLER=0`） | `FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0` |
| CUDA graph | `FULL_AND_PIECEWISE`。20 个序列时捕获尺寸为 1、2、4、8、16、24、32 和 40（本 runtime）；16 个序列时为 1、2、4、8、16、24 和 32（stock）（[docs/reference.zh-CN.md 4.8 节](reference.zh-CN.md#48-cuda-graph)） | 引擎配置行中的 `'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE` 和 `'cudagraph_capture_sizes': [...]` |
| 异步调度 | 本 runtime：关闭。stock：开启 | `Asynchronous scheduling is disabled.`（本 runtime）或 `Asynchronous scheduling is enabled.`（stock） |
| 纯文本 | 两边都用 `--language-model-only`（[docs/reference.zh-CN.md 4.12 节](reference.zh-CN.md#412-纯文本--language-model-only默认开启)） | vLLM 的 `running in text-only mode`；不能出现 `Encoder cache will be initialized` 或 `Multi-modal warmup completed`（可以出现 `Using ... for MMEncoderAttention` 行） |

READY 之后从每个服务的日志中确认：

```sh
grep -oE "Using FLASH_ATTN attention backend|Using FlashAttention version [0-9]+|Using Triton/FLA GDN prefill kernel \(requested=[a-z]+|FlashInfer top-p/top-k sampling disabled|Asynchronous scheduling is [a-z]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|'cudagraph_capture_sizes': \[[0-9, ]*\]|running in text-only mode|Encoder cache will be initialized|Multi-modal warmup completed" \
  server.log | sort | uniq -c
```

表中每一行都必须出现，且不能出现其他注意力后端、FlashAttention 版本、prefill kernel 或 CUDA graph 模式，也不能出现多模态启动日志行。日志不符的服务不纳入对比。

### A.6 Agent 配置

除标注外，所有服务完全相同。

| 项目 | 取值 |
| --- | --- |
| Agent | Harbor 0.21.0 内置的 Terminus 2（`terminus-2`，`harbor.agents.terminus_2:Terminus2`，版本 2.0.0），不作修改：它的 JSON 动作解析器、它的 prompt 模板、开启的摘要（主动摘要阈值 8,000 token）、它自己的重试层。它不发送 session header，也不发送关闭请求 |
| Harbor | 0.21.0，从其 wheel 安装，未作任何修改（每个文件都与 wheel 的 RECORD 一致，A.2），litellm 1.97.0，openai 2.54.0；`HARBOR_TELEMETRY=0`；不对 Harbor 或其 agent 打任何本地补丁 |
| 运行 | `harbor run -c job.json -y`，每个服务一个 Harbor 作业，作业配置见下 |
| 数据集 | `terminal-bench/terminal-bench-2`，固定 ref（A.2），过滤为 A.1 的 89 个名字（`terminal-bench/<name>`） |
| 模型 | `hosted_vllm/selective-prefix`，`api_base` `http://127.0.0.1:<port>/v1`，`temperature` 0.0，`model_info`（`max_input_tokens` 262144，即 A.3 中服务的上下文；`max_output_tokens` 4096；两项费用均为 0），`llm_kwargs`（`max_tokens` 4096、`timeout` 3600、`max_retries` 0、`num_retries` 0）；stock 对照另加 `llm_call_kwargs`（A.4） |
| Trial | `n_concurrent_trials` 16、`n_attempts` 1、`retry.max_retries` 0（Harbor 不重试 trial）、`timeout_multiplier` 1.0（每个任务用自己的超时）、开启 verifier、`quiet` |
| 工具环境 | `docker`：每个 trial 在用该任务镜像（A.1、B.1）创建的容器中运行，使用任务自己的 CPU、内存和外网设置；trial 结束后删除（`delete: true`、`force_build: false`） |

Terminus 2 中会影响结果的性质：
- Terminus 在 agent 主机上的 Harbor 进程里运行，通过 tmux 操作每个任务容器；它的模型调用从主机发出，所以 loopback 端点就够了（B.4）。
- 没有 system 消息。第一条消息是 user 消息：填入任务说明和初始终端画面的 prompt 模板，画面中显示容器的主机名，所以每个 trial 的第一条消息都不同。动作是 JSON（`analysis`、`plan`、带按键和时长的 `commands`、`task_complete`）；完成需要第二次确认。无法解析的回复留在历史中，下一个 prompt 报告解析错误。
- `finish_reason` 为 `length` 的回复会在 Harbor 的 LiteLLM 封装中引发输出超长错误；Terminus 把被截断的回复和一条错误提示追加到历史中，再次请求。
- 摘要：在剩余的 `max_input_tokens` 不足 8,000 token（按它自己的 token 估计）的 episode 之前，以及在上下文超长错误之后（它先回退历史），Terminus 用三次子调用做摘要（基于当前历史的摘要、不带历史的 "questions" 调用、回答），然后从替换掉历史的交接内容继续。
- 每个任务的 agent 超时是墙钟时间（`[agent] timeout_sec` × 1.0）。超时时 Harbor 取消 agent 及其进行中的模型调用，该 trial 记录 `AgentTimeoutError`。

**本 runtime 的 `job.json`**：`render_job.py`（B.2）和 `examples/terminalbench/make_job.py --profile qwen3.6-27b` 在端口 20000 写出的内容，这里排版更紧凑；`job_name` 和 `jobs_dir` 每次尝试都是新的，`task_names` 包含全部 89 个名字（这里有省略）：

```json
{
  "agents": [
    {
      "kwargs": {
        "api_base": "http://127.0.0.1:20000/v1",
        "llm_kwargs": {"max_retries": 0, "max_tokens": 4096, "num_retries": 0, "timeout": 3600},
        "model_info": {"input_cost_per_token": 0, "max_input_tokens": 262144, "max_output_tokens": 4096,
                       "output_cost_per_token": 0},
        "temperature": 0.0
      },
      "model_name": "hosted_vllm/selective-prefix",
      "name": "terminus-2"
    }
  ],
  "datasets": [
    {
      "name": "terminal-bench/terminal-bench-2",
      "ref": "sha256:c6fc2e2382c1dbae99b2d5ecd2f4f4a60c3c01e0d84642d69b4afd92e99d078b",
      "task_names": ["terminal-bench/adaptive-rejection-sampler", "terminal-bench/bn-fit-modify",
                     "...", "terminal-bench/write-compressor"]
    }
  ],
  "environment": {"delete": true, "force_build": false, "type": "docker"},
  "job_name": "full-a1",
  "jobs_dir": "/abs/path/on/agent/runs/full-a1/jobs",
  "n_attempts": 1,
  "n_concurrent_trials": 16,
  "quiet": true,
  "retry": {"max_retries": 0},
  "timeout_multiplier": 1.0,
  "verifier": {"disable": false}
}
```

**stock 对照的 `job.json`**：同一个文件，agent 的 `kwargs` 中多一项，即 A.4 中每次调用的 thinking 开关：

```json
"llm_call_kwargs": {"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}
```

本 runtime 自己渲染关闭 thinking 的生成头，并忽略客户端的采样字段，所以它的作业配置没有这一项；两份配置只在这里不同。

### A.7 限制

| 限制 | 取值 |
| --- | --- |
| Agent 超时 | 每个任务的 `[agent] timeout_sec` × 1.0：600 到 12,000 秒（A.1） |
| Verifier 超时 | 每个任务自己的：360 到 12,000 秒 |
| 回复上限 | 每次模型调用最多 4096 个新 token：客户端的 `max_tokens` 4096（本 runtime：在服务端上限 `--max-tokens 4096` 之内；stock：来自客户端） |
| 上下文上限 | 262,144 token（prompt 加回复）。prompt 加 4,096 超过它的请求返回 HTTP 400 和 vLLM 的文字；Terminus 随即回退历史并做摘要（A.6）。剩余不足 8,000 token 时 Terminus 也会主动做摘要 |
| 客户端超时 | 每次模型调用 3600 秒（`llm_kwargs.timeout`） |
| 客户端重试 | litellm 和 OpenAI 客户端：不重试（`max_retries` 0、`num_retries` 0）。Terminus 2 自己的重试层保持原样：它的 LiteLLM 封装最多尝试 3 次，外面它的 query 循环最多尝试 3 次。Harbor：不重试 trial，只尝试一次 |
| 每个服务的 session | 同时最多打开 20 个（K）；服务生命周期内共 512 个槽位（客户端 session、摘要 "questions" session 和 rebind 都算）；一个槽位服务 256 个请求后，该 session 换绑；生命周期内的请求上限是 max(1,024, 256 × 512) = 131,072（[docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)） |
| 锁页 Page arena | 70 GiB，71,680 个 1 MiB 的 Page；超出它的捕获会让该请求失败（`DRAM_ARENA_EXHAUSTED`，按 A.11 算作失败） |

**为什么客户端超时设为 3600 秒。** 在本 runtime 上，客户端超时会断开请求：服务端取消它并让它的槽位退役，该 trial 的下一个请求换绑到一个新槽位，做一次完整 prefill（[docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)）。一个请求在最多 4096 token 的回复开始之前，可能要等待空出的 session（20 个都已打开时）或 KV 容量（`CAPACITY_WAIT`，[docs/running.zh-CN.md 4.4 节](running.zh-CN.md#44-一个服务上的多个-session)）；3600 秒的超时让这类等待不会被打断再重发。Terminus 2 保留的重试会在出错后重发：在本 runtime 上，重发的请求会换绑或重建；如果第一个请求仍在进行中，重发的请求最多等它 30 秒，之后得到 409 `SESSION_BUSY`。stock 对照使用相同的设置，使两边只在服务端不同。

### A.8 并发与 session 生命周期

- Harbor 同时最多运行 16 个 trial（`n_concurrent_trials` 16），每个都有自己的 Terminus 2 实例和任务容器。一个 trial 结束后（包括 verifier），Harbor 按它自己的顺序启动下一个。
- **本 runtime 上的 session 绑定。** Terminus 不发送 `X-Session-ID`，所以服务端按初始段，也就是第一条 user 消息，确定每个客户端 session；这条消息每个 trial 都不同（A.6）：每个 trial 一个主 session。摘要的 "questions" 调用（不带历史发送）的初始段不同，是单独的 session；摘要和回答两次调用会续接或重建主 session。`key_sha256` = `json.dumps([{"role": "user", "content": <第一条 user 消息>}], ensure_ascii=False, sort_keys=True, separators=(",", ":"))` 按 UTF-8 编码后的 SHA-256；它把一个 trial 的 ATIF trajectory（第 1 步的 `message`）和服务端的行对上。
- **本 runtime 上的关闭。** Terminus 不发送关闭请求。在 K = 20、16 个 trial 下，结束的 trial 和 "questions" session 占用 4 个空余位置。只有 20 个都已打开时才会关闭正在使用的 session：有新 session 在等待时，关闭最久未用、且空闲至少 30 秒的 session（优先关 aside），每个等待的请求关一个；没有请求在等时，关闭最久未用、且空闲至少 600 秒的一个 session，让下一个新 session 有空位。只要还有空位，无论空闲多久都不会关闭正在使用的 session，所以正在执行长命令的 trial 会保住它的 session。失败的、已换走的和达到请求上限的槽位立即关闭（[docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)）。GPU 紧张时，`--release-idle-pages` 会释放空闲 session 的常驻 Page，但不关闭这些 session（A.3）。
- **运行结束后，在本 runtime 上。** 作业在 GPU 节点上关闭所有仍然打开的客户端 session：对 `chat_sessions.jsonl` 中最后一个槽位有 bind 或 rebind 行、但没有 close 行的每个 session，发送 `POST /v1/chat-text/close`，请求体 `{"key_sha256": "<key>"}`（带 header 的 session 用 `{"session_id": ...}`），遇到 HTTP 409 时每隔 2 秒重试，最多 60 秒。这样关闭服务时没有打开的 session，清理回执会记录这一点（[docs/design.zh-CN.md 第 8 节](design.zh-CN.md#8-每个服务的证据)）。
- **stock 对照。** 任何时候都不关闭；stock vLLM 把缓存的 prefix 保留到它自行淘汰为止。
- **Harbor 的记录**，在 agent 主机上的 `jobs/<job name>/` 下：`config.json`、`result.json`（作业本身），以及每个 trial 一个目录，其中有 `result.json`（`started_at`、`finished_at`，各阶段 `environment_setup`、`agent_setup`、`agent_execution` 和 `verifier` 的时间，`verifier_result.rewards.reward`，`exception_info`，agent 的 token 计数）、`agent/trajectory.json`（主 ATIF trajectory，续写在 `trajectory.cont-*.json` 中）、`agent/trajectory.summarization-*.json`（摘要子调用）和 `verifier/`。

### A.9 评测

| 项目 | 取值 |
| --- | --- |
| 评测程序 | Harbor 的 verifier，在每个 trial 内部紧接着 agent 运行（`verifier.disable: false`）；没有单独的评测阶段 |
| 判定 | trial 的 `result.json` 中的 `verifier_result.rewards.reward`（也在 `verifier/reward.txt` 中）；1.0 为已解决 |
| 模型 | 不需要：verifier 运行时不需要服务，但服务一直运行到 Harbor 作业结束 |
| 未完成的 trial | trial 没有完成结果（被取消或从未启动）的计划内任务计为未解决 |

### A.10 指标

**准确率** = Harbor verifier reward 为 1.0 的 trial 数 / 89 个计划内任务 × 100。一个任务是 1.12 分。准确率为 50 % 时，一次运行的二项分布标准误约为 5.3 分，两次独立运行之差的标准误约为 7.5 分。保留每个任务的 reward，以便逐个任务比较各次运行。

**吞吐** = ATIF 主 agent 的 completion token 数 / 各 agent 的时间跨度：

| 部分 | 定义 |
| --- | --- |
| 分子 | 89 个计划内任务的 trial 中，每个 trial 的主 ATIF trajectory（`agent/trajectory.json` 及其续写 `agent/trajectory.cont-*.json`）里每个带 completion token 计数的 agent 步的 `metrics.completion_tokens` 之和。Terminus 从每个响应的 `usage` 记录这些计数 |
| 单列 | 摘要子调用，`agent/trajectory.summarization-*.json`（摘要、"questions" 和回答三次调用）：在分子旁边单独报告，不计入分子 |
| 分母 | agent 主机时钟上各 agent 的时间跨度：所有 trial 的 `result.json` 中最晚的 `agent_execution.finished_at` 减去最早的 `agent_execution.started_at` |
| 包含 | 这两个时刻之间的一切：工具执行、等待模型和等待空出 session、摘要、重建和换绑、其他 trial 的环境准备和 verifier、服务端自己关闭 session |
| 不包含 | 服务的启动和关闭、第一个 agent 开始之前的环境准备、最后一个 agent 结束之后的 verifier、运行结束后作业的关闭请求，以及 Goal rewriter 自己的 token（`goal_query`；辅助模型） |
| 交叉核对 | 在本 runtime 上，对完整的运行：客户端的 completion token 数（主 agent 加摘要），加上已送达但未记录的回复（见下），等于对上的服务端行（A.8）的 `chat_text.completion_tokens` 之和；vLLM 的 `vllm:generation_tokens_total` 在运行期间的增量既不小于客户端的总数，也不小于服务端各行的总数。Harbor 作业的墙钟时间（包括环境准备和 verifier）与时间跨度并列报告 |

**被 agent 超时打断的请求。** 一个 trial 到达它的 agent 超时时，Harbor 取消进行中的模型调用。客户端永远收不到那条回复，所以它的 token 不在 ATIF trajectory 中，也不计入吞吐的分子。在本 runtime 上，被取消的请求有一行 `failures.jsonl`（`client disconnected`），没有 `runtime.jsonl` 行，但 vLLM 的计数器包含它生成的 token；服务端的统计把这部分生成单独报告为未送达（vLLM 的增量减去服务端各行的 completion token 数）。如果一条回复在取消之前刚刚到达 Harbor，但 Terminus 还没把它记录为一步，它同样不在 trajectory 中；对账根据 Harbor 自己的调用记录统计它，单独报告为已送达但未记录，同样不计入分子。同样，对账也统计 Terminus 2 没有记录为 ATIF 步骤、在 `max_tokens` 处截断的回复（在主循环中，或是其上下文超限回退中的完整 summary 调用），单独报告（“Length-cut replies (unrecorded)”），不计入分子，就像它统计 short summary 一样。

每次运行还要记录：每个 trial 的异常（`AgentTimeoutError` 等）、调用次数、摘要次数和最大的 prompt；每个服务的上下文超限拒绝次数、重建次数、换绑次数、`length` 结束次数和发生选择的轮次（本 runtime），以及抢占计数（stock，A.4）。

### A.11 运行规则

- **agent 主机上同一时间只跑一次运行**，trial 运行期间主机上不做别的事（不跑其他 benchmark、压力测试或评测）。agent 超时是墙钟时间，所以 agent 主机的负载会同时改变吞吐和准确率。要比较的各个服务使用同一类 agent 主机。
- **每次尝试都用全新的服务和新的输出目录**，以及新的 Harbor 作业名和 jobs 目录。`serve.py` 和 `tb_agents.sh`（B.2）都拒绝已存在的输出目录。
- **中断的运行不续跑。** 提前结束的运行（作业时间上限、agent 的时间上限、隧道断开、服务失败）是不完整的运行：未完成的 trial 不在另一个作业里接着跑，该次运行不纳入对比，也不给出吞吐，整个服务换一个新标签从头重跑。
- **失败的计法。** 上下文超限的拒绝（admission 之前的 HTTP 400）是正常结果：Terminus 做摘要，服务继续工作。客户端在任务的 agent 超时时取消的请求也是正常的。本 runtime 服务上的其他任何失败（引擎失败并返回 500、`DRAM_ARENA_EXHAUSTED`、`SESSION_BUSY`、`SESSION_LIMIT`、被拒绝的请求、抢占）都会让该服务的检查不通过，该次运行不纳入对比。
- **容量停滞。** 放不下的一轮会等待（`CAPACITY_WAIT`，A.3）；如果没有一轮放得下，服务就停滞。当有请求在等待、且连续 10 分钟既没有请求完成、vLLM 也没有计算任何 token 时，harness 停止 Harbor 的运行；该服务判为失败，该次运行不纳入对比。
- **传输方式。** Terminus 通过只走 loopback 的路径访问服务（B.4）。任何只走 loopback 的路径给出相同的结果；服务从不在 loopback 之外监听（[docs/running.zh-CN.md 6.4 节](running.zh-CN.md#64-从其他主机访问服务)）。

### A.12 对齐检查表

| # | 项目 | 本 runtime | baseline 使用 |
| --- | --- | --- | --- |
| 1 | 模型与 revision | Qwen3.6-27B `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | 相同 |
| 2 | 软件 | vLLM 0.24.0 官方 wheel、Transformers 5.13.0、Tokenizers 0.22.2、`constraints-3270721.txt` | 同一个环境 |
| 3 | 硬件与作业 | 1 × GH200 120GB、16 个 CPU、113,560 MiB | 相同 |
| 4 | 任务 | A.1 的 89 个任务，数据集 ref `sha256:c6fc2e23…` | 相同 |
| 5 | 并发 | `-n 16`，只尝试一次，按 Harbor 的顺序 | 相同 |
| 6 | 上下文上限 | 262144 | `--max-model-len 262144` 和 `max_input_tokens` 262144 |
| 7 | 每步 batched token 数 | 16384 | `--max-num-batched-tokens 16384` |
| 8 | 最大序列数 | 20（K；最多 16 个 trial 同时运行） | `--max-num-seqs 16`；已知差异（A.13） |
| 9 | GPU KV | 25 GiB（26,843,545,600 字节） | `--kv-cache-memory-bytes 26843545600` |
| 10 | 回复上限 | 4096（客户端和服务端） | `max_tokens` 4096（客户端） |
| 11 | 解码 | 贪心（服务端） | `temperature` 0.0 |
| 12 | Thinking | 关闭（生成头中为空的 think 块） | 每次调用的 `llm_call_kwargs` 带 `chat_template_kwargs: {"enable_thinking": false}` |
| 13 | 停止 | 只在 EOS/pad | 不设停止字符串 |
| 14 | CUDA graph | `FULL_AND_PIECEWISE`（`--cuda-graphs full`） | vLLM 默认（`FULL_AND_PIECEWISE`） |
| 15 | 异步调度 | 关闭（launcher 的默认值；这里不用） | vLLM 默认（开启）；已知差异（A.13） |
| 16 | 注意力 | FLASH_ATTN 加 FlashAttention 3，由 vLLM 选择 | 相同，不加 `--attention-backend` |
| 17 | Gated-DeltaNet prefill | Triton | `--gdn-prefill-backend triton` |
| 18 | Sampler | 原生（`VLLM_USE_FLASHINFER_SAMPLER=0`） | 同一个变量 |
| 19 | Prefix caching | 开启，sha256 | `--enable-prefix-caching --prefix-caching-hash-algo sha256` |
| 20 | Mamba cache 模式 | `align` | `--mamba-cache-mode align` |
| 21 | Seed | 42 | `--seed 42` |
| 22 | 纯文本 | `--language-model-only` | vLLM 的 `--language-model-only` |
| 23 | Agent | Harbor 0.21.0、原样的 Terminus 2 2.0.0、litellm 1.97.0、openai 2.54.0 | 相同 |
| 24 | Agent 行为 | JSON 解析器，剩余 8,000 token 时做摘要，Terminus 自己的重试，不发 session header 也不发关闭请求 | 相同（Terminus 不作修改） |
| 25 | 模型 kwargs | `temperature` 0，`model_info` 262144 / 4096 / 0 / 0，`llm_kwargs` 4096 / 3600 秒 / 0 / 0 | 相同，另加 `llm_call_kwargs`（第 12 行） |
| 26 | Harbor 作业 | `n_attempts` 1，不重试 trial，`timeout_multiplier` 1.0，Docker 并 `delete`，开启 verifier | 相同 |
| 27 | Agent 主机 | 同一时间只跑一次运行，主机上不做别的事，提前拉取任务镜像 | 同样的规则，同一类主机 |
| 28 | 评测 | 每个 trial 中 Harbor 的 verifier；reward 1.0 = 已解决 | 相同 |
| 29 | 吞吐 | A.10 的定义 | 相同的分子和计时 |
| 30 | 不完整的运行 | 不纳入，不续跑 | 相同 |

### A.13 本 runtime 与 stock vLLM 的已知差异

即使检查表的每一项都一致，以下差异仍然存在。任何对比都要一并报告。

1. **历史回复。** 本 runtime 以每条旧回复生成时的原始 token ID 回放，前面保留关闭 thinking 的生成头放进去的空思考块。stock vLLM 用模型模板的默认参数从文本重新渲染历史，这会去掉历史回复的空思考块：那里每条历史回复短 4 个 token（`<think>`、`\n\n`、`</think>`、`\n\n`），且其文本被重新编码（[docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)）。
2. **异步调度。** 本 runtime 不使用（launcher 默认传 `--no-async-scheduling`；它的可选项 `--async-scheduling` 这里不用）；stock vLLM 使用其默认值，即开启。
3. **采样由谁控制。** 本 runtime 贪心解码，忽略客户端的采样字段，只在 EOS/pad 处停止；stock vLLM 从请求中读取采样设置（`temperature` 0 即贪心解码）。
4. **上下文边界。** 本 runtime 在 admission 之前以 HTTP 400 和 vLLM 的文字 `This model's maximum context length is …` 拒绝 prompt 加 `max_tokens` 超过 262,144 token 的请求；预计 stock vLLM 0.24.0 会用自己的检查拒绝这样的请求，文字相同（`vllm/renderers/params.py`，它在 `source-overlay/` 中的副本含有这段文字）。未在 stock 对照上测量。
5. **序列数与 KV 管理。** 两者的 GPU KV 都是 25 GiB，但布局不同。本 runtime 最多打开 20 个 session（`max_num_seqs` 20，graph 捕获到 40），使用 16 token 的注意力 Page、单独的 Gated-DeltaNet 池、1 GiB 的 CPU 池和 70 GiB 的锁页 Page arena；它从不抢占，放不下的一轮会等待（`CAPACITY_WAIT`，[docs/running.zh-CN.md 4.4 节](running.zh-CN.md#44-一个服务上的多个-session)），没有其他请求运行时也一样。空闲 session 的常驻 Page 留在 GPU 上，直到 `--release-idle-pages` 在 GPU 紧张时释放它们；被释放的 session 在下一轮之前从主机内存拷回它的 Page（A.3）。stock vLLM 最多运行 16 个序列，使用它自己的混合布局（[docs/reference.zh-CN.md 4.6 节](reference.zh-CN.md#46-解耦-kv-布局与-16-token-的-page)），只把已结束请求的 block 作为可淘汰的 prefix cache 保留，淘汰掉的部分要重新计算，池满时可能抢占。
6. **admission 与调度。** 本 runtime 按到达顺序一次准入一个请求，每次 admission 的阶段 1 是单独的一次 forward，然后成批 decode（[docs/running.zh-CN.md 4.4 节](running.zh-CN.md#44-一个服务上的多个-session)和[docs/design.zh-CN.md 第 5 节](design.zh-CN.md#5-query-策略的选页规则)）。stock vLLM 在每步 16,384 token 的预算内把 prefill 分块与 decode 步一起调度。
7. **Session 生命周期。** 本 runtime 一直保留 session 的状态，直到它自己的关闭规则或作业关闭它（A.8），并保留它在 GPU 上的 Page，直到在 GPU 紧张时被释放（第 5 条）；达到 256 个请求、或槽位失败的 session 换到新槽位，做一次完整 prefill。stock vLLM 把缓存的 prefix 保留到它自行淘汰为止。
8. **重建。** 在本 runtime 上，不是上一个请求延续的请求（摘要交接、回退后的历史、重发）会从文本重建 session，做一次完整 prefill（[docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)）；stock vLLM 借助 prefix caching 直接处理。
9. **选择性策略。** 一个 session 已封存的历史超过 8,192 token 后，`observation_query`、`random_fill` 和 `goal_query` 只关注选中的一部分历史 Page；`full` 保留所有 Page。更短的 trial 等同于 `full`。
10. **墙钟超时。** agent 超时是墙钟时间，所以准确率除了取决于模型的输出，还取决于各服务的速度和 agent 主机的负载。
11. **两次运行之间的差异。** 两种服务的成批运行在两次运行之间都不是 token 完全相同的：回复可能随 batch 的组成而变，而 batch 的组成取决于时序（[docs/running.zh-CN.md 4.4 节](running.zh-CN.md#44-一个服务上的多个-session)）。
12. **Goal rewriter。** `goal_query` 在同一张 GPU 上运行一个辅助的 Qwen3-0.6B。它的时间计入吞吐，它的 token 不计入。

### A.14 Gemma 4 31B IT

用 `google/gemma-4-31B-it`（revision `842da3794eaa0b77d5f08bae87a17459d91ff475`）运行同一个正式测试，release 用 `install.py --model-profile gemma4-31b-it` 安装（`$RELEASE-gemma4`，[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）。`serve.py` 按 release manifest 分派：`TRITON_ATTN`，16 token Page 的双池窗口布局（不传 `--kv-layout`），chat-text 协议上 Gemma 的轮次格式及其停止 id，没有 Gated-DeltaNet 状态（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）。任务、agent 配置和 `job.json`（A.6）、各项限制、并发（`-n 16`）、评测、指标和运行规则（A.7 到 A.11）都与 Qwen 相同，8,192 token 的预算也与 Qwen 相同。上下文为 262,144 token，GPU KV 池为 25 GiB，与 Qwen 相同：这是组里 2026-10-26 定的设置，维护者已于 2026-11-08 确认。在此基础上，维护者于 2026-11-08 定下：K = 20（与 Qwen 相同），4,096 token 的前向（`--max-num-batched-tokens 4096`，组里的前向大小，runtime 服务和 stock 对照都用），72 GiB 的 arena，以及 `observation_query`、`random_fill` 和 `goal_query` 三个策略加 stock 对照；`full` 不属于 Gemma 的配置，因为它在 262,144 token 下放不下（见下）。客户端的 Gemma job 就是 Qwen 的 job（`max_input_tokens` 262144）。它的 GPU 运行及其判定见 [docs/status.zh-CN.md](status.zh-CN.md)；更早的 Gemma dry run 都在 65,536 token、8,192 token 的前向下运行（其设置见下文“按实际运行”）。runtime 与 A.2 相同；Gemma profile 从 `9ef0d6d` 起接受 `--decode-fastpath3` 和 `--deferred-completion-io`，从 integrate-1026 起接受 `--max-num-batched-tokens 4096`（gemma-forward4k；最早在 `cf2b28d` 上于作业 3461663 和 3462014 中在 GPU 上运行）。

**服务参数。** A.3 的共用参数，取 Qwen 的上下文、KV 大小和 K（chat-text 参数、`--kv-cache-gib 25`、`--max-model-len 262144`、`--page-tokens 16`、`--cuda-graphs full`、`--max-running-sessions 20`、`--max-tokens 4096`、`--connector-pool-gib 1`、三个解码快速路径、`--deferred-completion-io`、`--language-model-only`、`--release-idle-pages`），不带 `--kv-layout`，另加 Gemma 的 `--max-num-batched-tokens 4096` 和 `--dram-arena-gib 72`（Qwen：70），各策略另加：

| 策略 | 策略参数 |
| --- | --- |
| `observation_query` | `--active-token-budget 8192` |
| `random_fill` | `--active-token-budget 8192` |
| `goal_query` | `--active-token-budget 8192 --goal-rewriter-model <Qwen3-0.6B 目录> --admission-overlap 16 --goal-rewriter-batch 4 --goal-rewriter-batch-tokens 20480` |
| `full` | 不属于 Gemma 的配置：launcher 在 262,144 token、K = 20 时拒绝它（见下） |

- **KV 与 session 数。** 25 GiB 是组里 Gemma baseline 的 KV cache。在 262,144 token、4,096 token 的前向和回复下（P = 16），pool 1 容纳 20 个窗口和一次前向的瞬态，共 7,781 个 2.5 MiB 的块 id（19.00 GiB；`goal_query` 加上它的 Goal 临时区为 8,421 个，20.56 GiB），pool 0 得到其余的 4,918 个 1.25 MiB 的全局 Page，除去 vLLM 的空 Page 后可用 4,917 个（`goal_query`：3,638 个，可用 3,637 个）。launcher 的选择性规则（`--release-idle-pages` 配 8,192 token 的预算，[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）需要其中的 2,119 个（`goal_query`：2,106 个）：一个 session 最坏的选择性轮次 2 × (512 + 513) + 1 = 2,051，其余 19 个 session 的尾部，以及 1 % 的水位 49（36）。余量：`observation_query` 和 `random_fill` 为 2,798 个 Page，`goal_query` 为 1,531 个；最多分别放得下 24 和 22 个 session。vLLM 自己的启动检查要求 25 GiB 中的 21.58 GiB（`goal_query`：23.13 GiB）。该规则与上下文无关。用 8,192 token 的前向时 K = 20 放不下（需要 2,605 个 Page，可用的只有 2,357 个（pool 0 为 2,358 个），最多 19 个 session；`goal_query`：需要 2,592 个，可用的只有 1,077 个（pool 0 为 1,078 个），最多 17 个），所以本配置取 4,096。`full` 需要 pool 0 容纳一个停在上下文上限的 session（release 规则）：K = 20 时需要 17,023 个 Page，而 4,096 token 的前向下 32 GiB 只有 10,652 个（8,192 时需要 17,253 个，只有 8,092 个），最多 10 个 session（5 个）。
- **Goal rewriter 的 token 上限。** `goal_query` 显式传 T = 20,480。按 harness 的 GPU 估算，25 GiB 时 EngineCore 占 87,286 MiB，一次 admission 的临时开销 258 MiB，rewriter 子进程 2,836 MiB，T 个填充 token 的一批 3,724 MiB，乘以 1.3 倍的分配器余量为 4,841 MiB：共 95,221 MiB，CUDA 可分配 97,362 MiB，剩余 2,141 MiB。该 profile 在 25 GiB 时的默认值 25,600（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）只会剩约 0.7 GiB。这个估算里没有上下文、K 或前向大小的项；它是估算，不是实测。
- **主机内存。** 72 GiB 的 arena 容纳 58,982 个 1.25 MiB 的 Page（943,712 token：3 个停在上限的 session，或 20 个 47,185 token 的 session），按每块 204 个 Page 的 256 MiB 块锁页（290 块，72.22 GiB）；加上 1 GiB 的 connector 池共锁页 73.22 GiB（Gemma 没有 Gated-DeltaNet carry），在 launcher 的 80.90 GiB 之内（`--mem=113560M` 减去 30 GiB；它允许的最大 arena 是 79 GiB）。加上进程所需的约 32 GiB，共 105.22 GiB，一张 GH200 的份额为 110.90 GiB，剩余 5.68 GiB：72 GiB 是能留出 5 GiB 余量的最大 arena，与 Qwen 的 70 GiB 一样（A.3）。不设 arena 时，20 个停在 262,144 token 上限的 session 可能锁页 401.2 GiB，launcher 会拒绝。

**命令。** 三个 Gemma 服务的 dry run（加 `--start` 即启动服务；`LABEL` 是每次尝试的新标签）：

```sh
GSHARED=(--protocol chat-text --chat-text-slots 512 --chat-text-idle-timeout 600 --chat-text-evict-after 30
         --kv-cache-gib 25 --max-model-len 262144 --page-tokens 16 --cuda-graphs full --max-running-sessions 20
         --max-tokens 4096 --max-num-batched-tokens 4096 --dram-arena-gib 72 --connector-pool-gib 1
         --decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io --language-model-only
         --release-idle-pages)
python -B serve.py --release-dir "$RELEASE-gemma4" --policy observation_query "${GSHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/tb-gemma-observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE-gemma4" --policy random_fill "${GSHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/tb-gemma-random_fill-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE-gemma4" --policy goal_query "${GSHARED[@]}" \
  --active-token-budget 8192 --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --admission-overlap 16 \
  --goal-rewriter-batch 4 --goal-rewriter-batch-tokens 20480 \
  --output "$RUNS/tb-gemma-goal_query-$LABEL" --port 8000
```

dry run 会打印 vLLM 命令；该 profile 额外加入的参数见 [docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)（`--attention-backend TRITON_ATTN`、窗口布局的 `--additional-config`、gemma4 parser）。加 `--max-num-batched-tokens 4096` 时，vLLM 收到 `--max-num-batched-tokens 4096`，布局为 `"forward_tokens": 4096`，并带选择性规则的 `"request_pages"`：2,119（`observation_query`、`random_fill`）或 2,106（`goal_query`）。B.5 的脚本服务 Qwen；Gemma 作业用这些命令代替，它的 job 文件就是 Qwen 的（`max_input_tokens` 262144：`render_job.py` 原样即可，或用 `make_job.py --profile gemma4-31b-it`；未经测试）。

**4,096 token 的前向（属于本配置；[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）。** 每个服务都加上 `--max-num-batched-tokens 4096`，就使用组里 4,096 token 的前向（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)“4,096 token 的前向”）：vLLM 收到 `--max-num-batched-tokens 4096`，窗口布局预留一次 4,096 token 前向的瞬态（pool 1 小 3.125 GiB，P = 16 时全局 Page 多 2,560 个），超过 4,096 token 的 phase 1 分成每次 3,056 token 的块。它让 K = 20 放得下（见上），stock 对照在 25 GiB 和 262,144 token 下也需要它（见下）。65,536 token 的 dry run（见下）都用 8,192 运行。

**在 65,536 token 下按实际运行（作业 3433706 和 3434544，[docs/status.zh-CN.md](status.zh-CN.md)）。** Gemma 的 dry run 使用的是 2026-11-08 之前 A.14 的参数：上面的 `GSHARED` 去掉 `--kv-cache-gib 25 --max-model-len 262144 --max-running-sessions 20 --max-num-batched-tokens 4096`，并以 `--dram-arena-gib 64` 代替 72（即默认的 65,536 token 上下文和 8,192 token 的前向），每个服务另加 `--kv-cache-gib 25 --max-running-sessions 17`（没有运行过的 `full`：`--kv-cache-gib 32 --max-running-sessions 20`）。在 25 GiB、8,192 token 的前向下，K = 17 是放得下带 Goal 临时区的 `goal_query` 的最大 K（余量 418 个 Page；`observation_query`：最多 19）；64 GiB 的 arena 容纳 52,428 个 Page，锁页 65.0 GiB。它们的 job 文件中 `max_input_tokens` 为 65536：`make_job.py --profile gemma4-31b-it --context 65536` 在相同端口、job 名称和 jobs 目录下逐字节写出这个文件。它们的 stock 对照（没有运行过）是下面的命令改用 `--max-model-len 65536 --max-num-batched-tokens 8192`。

**stock 对照。** 与 A.4 一样，直接从同一个环境启动 `vllm.entrypoints.openai.api_server`，使用 Gemma 的设置：

| 设置 | 取值 | 与本 runtime 的关系 |
| --- | --- | --- |
| `--model` | Gemma 4 31B IT 目录 | 相同 |
| `--attention-backend` | `TRITON_ATTN` | Gemma profile 的后端（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)） |
| `--kv-cache-memory-bytes` | `26843545600`（25 GiB） | runtime 服务的 KV 字节数 |
| `--max-num-seqs` | `16` | 等于 Harbor 的 `-n`。不同：本 runtime 为 K = 20 |
| `--max-model-len` | `262144`（与 runtime 服务取同一个值） | 相同 |
| `--max-num-batched-tokens` | `4096` | runtime 服务的一次原生前向（`--max-num-batched-tokens 4096`） |
| Mamba 和 Gated-DeltaNet 参数 | 无（Gemma 没有这类状态） | 相同 |
| Parser | 无：回复内容是完整的解码文本，与本 runtime 的 chat-text 路由返回的相同 | 内容相同 |
| Thinking | 关闭，由客户端在每次调用时请求（`llm_call_kwargs`，A.4）；Gemma 的模板默认就关闭 thinking | 生成头相同 |

其余设置与 A.4 相同：名称 `selective-prefix`、只用 loopback、`uni` 加一个张量并行 rank、`--seed 42`、sha256 的 prefix caching、`--language-model-only`，以及 vLLM 默认的 CUDA graph 和异步调度。vLLM 0.24.0 在启动时检查一个处于 `--max-model-len` 的请求能否放进它的 KV 内存。对 Gemma，一个块 id 容纳 10 层、每层 262,144 B，共 2.5 MiB，所以 25 GiB 是 10,240 个块 id；一个 262,144 token 的请求占用全注意力组的 8,192 个 32 token 的块，以及每个滑动窗口组 ⌈(1,023 + N) / 16⌉ + 1 个块：N = 4,096 时为 8,192 + 5 × 321 = 9,797 个块 id（23.92 GiB），放得下；N = 8,192 时为 8,192 + 5 × 577 = 11,077 个（27.04 GiB），vLLM 拒绝（作业 3364469：它在 262,144 token、8,192 token 前向下的 stock 服务在启动时就停止了，需要 27.04 GiB，可用的只有 26.41 GiB）。这样池里放得下一个停在上限的请求；stock vLLM 在池满时会抢占，而 harness 对 stock 服务的检查要求没有抢占（A.4）。

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
"$VENV/bin/python" -I -B -m vllm.entrypoints.openai.api_server \
  --model "$MODELS/gemma-4-31b-it" --served-model-name selective-prefix --host 127.0.0.1 --port 8000 \
  --distributed-executor-backend uni --tensor-parallel-size 1 \
  --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 4096 \
  --kv-cache-memory-bytes 26843545600 --seed 42 \
  --enable-prefix-caching --prefix-caching-hash-algo sha256 --attention-backend TRITON_ATTN \
  --language-model-only
```

**已知差异。** 与 A.13 相同，只是历史的渲染换成 Gemma 的：stock vLLM 的模板去掉每条历史回复的空 thought channel，每条回复少 4 个 token（`<|channel>thought\n<channel|>`，[docs/running.zh-CN.md 6.7 节](running.zh-CN.md#67-chat-text-协议terminal-bench-20-与-terminus-2)），并且窗口布局取代了 Gated-DeltaNet 池（[docs/reference.zh-CN.md 4.14 节](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证)）。

---

## Part B. 流程

这一部分用本仓库的工具运行 Part A 的配置：模型在 Slurm 作业中的 GH200 节点上，Harbor 和它的任务容器在另一台 x86_64 Docker 主机（agent 主机）上，GPU 作业到 agent 主机之间是一条反向 SSH 隧道（[docs/running.zh-CN.md 6.4 节](running.zh-CN.md#64-从其他主机访问服务)）。除标注外，每一步都同时适用于四个策略和 stock 对照。

### B.1 Agent 主机

- **要求。** Linux x86_64，装有 Docker、`curl`、`ss`、`python3` 和 `uv`，一个装有 A.2 版本的 Harbor 及其依赖的 Python 3.12 环境，以及本仓库的一个 checkout。主机要同时运行 16 个 trial 及其容器（每个任务的限制：1 到 4 个 CPU，2 到 8 GB 内存）。Harbor 第一次下载任务包时需要联网（[benchmarks/terminal-bench.zh-CN.md](benchmarks/terminal-bench.zh-CN.md#agent-主机的准备)），每个任务容器也都需要访问外网。隧道需要 GPU 节点能通过 SSH 登录这台主机（B.4）。harness 使用的 agent 主机有 16 个 vCPU 和 33 GB（30.8 GiB）内存，它的主机检查要求运行前至少有 12 GB 可用内存和 50 GB 空闲磁盘。
- **镜像。** 每个任务的包指定了它的镜像：`alexgshaw/<task>:20251031`，只有 `install-windows-3-11` 的镜像是 `alexgshaw/install-windows-3.11:20251031`。

环境和目录（这些命令不在已验证的运行范围内；`examples/terminalbench/requirements.lock` 带每个包的哈希固定了 trial 运行的 Harbor 环境，见 [Agent 主机的准备](benchmarks/terminal-bench.zh-CN.md#agent-主机的准备)），本仓库的 checkout 放在 `$AGENT/vllm-0.24.0-selective-prefix-policies`：

```sh
AGENT=/abs/path/on/agent
HARBOR=$AGENT/harbor-env
uv venv --python 3.12 "$HARBOR"
uv pip install --python "$HARBOR/bin/python" harbor==0.21.0 litellm==1.97.0 openai==2.54.0
mkdir -p "$AGENT/sock" "$AGENT/runs" "$AGENT/tb"
chmod 700 "$AGENT/sock"
```

把 A.1 的列表存为 `$AGENT/tasks.txt`（dry 子集存为 `$AGENT/tasks-dry.txt`），并在任何运行之前拉取 89 个镜像，使拉取时间不进入测量：

```sh
while read -r t; do
  img="alexgshaw/${t/install-windows-3-11/install-windows-3.11}:20251031"
  if docker pull "$img" > /dev/null; then echo "ok $img"; else echo "FAILED $img"; fi
done < "$AGENT/tasks.txt"
```

### B.2 Agent 侧文件

`$AGENT/tb` 中放三个文件：仓库中的 loopback relay、`job.json` 的生成脚本，以及运行一个服务的 Harbor 作业的脚本。复制 relay：

```sh
cp "$AGENT/vllm-0.24.0-selective-prefix-policies/examples/swebench/relay.py" "$AGENT/tb/relay.py"
```

`$AGENT/tb/render_job.py` 把 A.6 的作业配置写进一个新的输出目录；它写出的文件与 `examples/terminalbench/make_job.py --profile qwen3.6-27b` 在相同端口、job 名称和 jobs 目录下写出的文件相同（A.6）。`tb_agents.sh` 不运行该客户端的 `close_sessions.py check`（它会拒绝上下文不等于 job 的 `max_input_tokens` 的本 runtime 服务），所以要按服务的上下文生成 job：

```python
"""Write Harbor's job.json of one Terminal-Bench server run (docs/TERMINALBENCH_BENCHMARK.md, A.6)."""
import json
import sys
from pathlib import Path

kind, port, out, tasks = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
names = [line.strip() for line in tasks.read_text().splitlines() if line.strip()]
assert kind in ('runtime', 'stock') and names and len(names) == len(set(names))
kwargs = dict(api_base='http://127.0.0.1:%d/v1' % port, temperature=0.0,
              model_info=dict(max_input_tokens=262144, max_output_tokens=4096, input_cost_per_token=0,
                              output_cost_per_token=0),
              llm_kwargs=dict(max_tokens=4096, timeout=3600, max_retries=0, num_retries=0))
if kind == 'stock':
    kwargs['llm_call_kwargs'] = {'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
config = dict(job_name=out.name, jobs_dir=str(out / 'jobs'), n_attempts=1, timeout_multiplier=1.0,
              n_concurrent_trials=16, quiet=True, retry=dict(max_retries=0),
              environment=dict(type='docker', delete=True, force_build=False), verifier=dict(disable=False),
              agents=[dict(name='terminus-2', model_name='hosted_vllm/selective-prefix', kwargs=kwargs)],
              datasets=[dict(name='terminal-bench/terminal-bench-2',
                             ref='sha256:c6fc2e2382c1dbae99b2d5ecd2f4f4a60c3c01e0d84642d69b4afd92e99d078b',
                             task_names=['terminal-bench/' + name for name in names])])
(out / 'job.json').write_text(json.dumps(config, sort_keys=True, indent=2) + '\n')
```

`$AGENT/tb/tb_agents.sh`（设为可执行）运行一个服务的 Harbor 作业：创建输出目录，生成 `job.json`，启动 relay，等待端点，检查监听，然后运行 Harbor。这只是示例，**未经测试**：

```bash
#!/usr/bin/env bash
# Agent side of one Terminal-Bench server run (docs/TERMINALBENCH_BENCHMARK.md, B.2).
# Usage: tb_agents.sh <socket> <port> <output dir> <runtime|stock> <task file>
set -euo pipefail
: "${AGENT:?}" "${HARBOR:?}"
SOCK=$1; PORT=$2; OUT=$3; KIND=$4; TASKS=$5
mkdir "$OUT"                                          # refuses an existing directory (A.11)
python3 -I -B "$AGENT/tb/render_job.py" "$KIND" "$PORT" "$OUT" "$TASKS"
python3 -I -B "$AGENT/tb/relay.py" "$SOCK" "$PORT" > "$OUT/relay.log" 2>&1 &
RELAY=$!
trap 'kill "$RELAY" 2> /dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" -o "$OUT/models.json" && break
  sleep 5
done
grep -q '"selective-prefix"' "$OUT/models.json"
listeners=$(ss -ltnH "sport = :$PORT" | awk '{print $4}' | sort -u | tr '\n' ' ')
[[ $listeners == "127.0.0.1:$PORT " ]] || { echo "port $PORT listens on: $listeners" >&2; exit 3; }
cd "$OUT"
HARBOR_TELEMETRY=0 PYTHONDONTWRITEBYTECODE=1 "$HARBOR/bin/harbor" run -c "$OUT/job.json" -y \
  > "$OUT/harbor.log" 2>&1 < /dev/null
```

第一次运行之前，把两种配置各生成一次到临时目录，与 A.6 对照。

### B.3 占用 GPU 之前的环境检查

**镜像。** B.1 的每个镜像都必须已在本地：

```sh
while read -r t; do
  img="alexgshaw/${t/install-windows-3-11/install-windows-3.11}:20251031"
  docker image inspect "$img" > /dev/null 2>&1 || echo "MISSING $img"
done < "$AGENT/tasks.txt"
```

**Harbor 与 Terminus 2。** A.2 的版本以及 A.6 所依赖的默认值（未经测试）：

```sh
"$HARBOR/bin/python" -I -B - <<'PY'
import inspect, json
from importlib.metadata import version
from harbor.agents.terminus_2.terminus_2 import Terminus2
params = inspect.signature(Terminus2.__init__).parameters
names = ('parser_name', 'enable_summarize', 'proactive_summarization_threshold', 'session_id', 'llm_call_kwargs')
print(json.dumps(dict(versions={n: version(n) for n in ('harbor', 'litellm', 'openai')},
                      defaults={k: repr(params[k].default) for k in names if k in params})))
PY
```

预期：`harbor` 0.21.0、`litellm` 1.97.0、`openai` 2.54.0；`parser_name` 为 `'json'`，`enable_summarize` 为 `True`，`proactive_summarization_threshold` 为 `8000`，`session_id` 为 `None`，并且有 `llm_call_kwargs` 参数（stock 对照需要它）。

**stock 对照的 thinking 关闭渲染。** 设置 A.3 的环境变量后（不需要 GPU），检查模型的 chat template 在 `enable_thinking=false` 下的渲染（未经测试）：

```sh
"$VENV/bin/python" -I -B - "$MODELS/qwen3.6-27b" <<'PY'
import sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
def render(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
print(repr(render([{'role': 'user', 'content': 'u'}])))
print(repr(render([{'role': 'user', 'content': 'u'}, {'role': 'assistant', 'content': 'a'},
                   {'role': 'user', 'content': 'v'}])))
PY
```

第一行必须是 `'<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'`（没有默认的 system 文字，以 A.3 的生成头结束）；第二行必须是 `'<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\na<|im_end|>\n<|im_start|>user\nv<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'`（历史回复不带思考块，A.13）。

### B.4 隧道

GPU 作业在 READY 之后打开一条反向 SSH 隧道，远端是 agent 主机上的一个 Unix socket；`tb_agents.sh` 通过 `relay.py` 把它暴露在 `127.0.0.1:<port>`，如果该端口上除 `127.0.0.1` 之外还有监听，它就拒绝运行（[docs/running.zh-CN.md 6.4 节](running.zh-CN.md#64-从其他主机访问服务)）。stock 对照使用同样的形式。需要：
- 一把 agent 主机为隧道用户接受、GPU 作业可读的 key，以及一个固定 agent 主机 host key 的 known-hosts 文件（作业从不使用 SSH agent）；
- 计算节点能通过 SSH 连到 agent 主机（第一次运行前从计算节点检查一次）；
- 每个服务一个新的 socket 路径，不超过 100 字节，位于 agent 主机上一个 `0700` 目录中（`$AGENT/sock`）。

隧道断开会结束这次运行（A.11）；作业不会重连。

### B.5 Slurm 作业脚本

下面的脚本在一个 Slurm 作业里运行正式测试的一个服务：服务、READY、kernel 日志行、隧道、通过 SSH 在 agent 主机上运行的 Harbor 作业、关闭遗留的 session，以及关闭服务。它由 `examples/slurm/serve_gh200.sh` 改写而来，后者不传 chat-text 参数、`--cuda-graphs`、各解码快速路径、`--deferred-completion-io`、`--connector-pool-gib`、`--language-model-only`、`--release-idle-pages`、`--admission-overlap` 和 `--goal-rewriter-batch`。这只是示例，**未经测试**。存为 `tb_job.sh`：

```bash
#!/usr/bin/env bash
# One server of the Terminal-Bench 2.0 benchmark (docs/TERMINALBENCH_BENCHMARK.md) in one Slurm job.
# POLICY: full | observation_query | random_fill | goal_query (this runtime) | stock (stock vLLM 0.24.0).
set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}" || { echo "run this under sbatch" >&2; exit 2; }
: "${REPO:?}" "${VENV:?}" "${POLICY:?}" "${RUN_ROOT:?}" "${AGENT_SECONDS:?}"
: "${AGENT_HOST:?}" "${TUNNEL_KEY:?}" "${KNOWN_HOSTS:?}" "${AGENT_SOCKET:?}" "${AGENT_PORT:?}"
: "${AGENT_RUN:?}" "${AGENT_OUT:?}" "${AGENT_ENV:?}" "${AGENT_TASKS:?}"
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

# The server: Part A.3 (this runtime) or Part A.4 (stock).
SHARED=(--protocol chat-text --chat-text-slots 512 --chat-text-idle-timeout 600 --chat-text-evict-after 30
        --kv-cache-gib 25 --max-model-len 262144 --page-tokens 16 --kv-layout decoupled --cuda-graphs full
        --max-running-sessions 20 --max-tokens 4096 --dram-arena-gib 70 --connector-pool-gib 1
        --decode-fastpath --decode-fastpath2 --decode-fastpath3 --deferred-completion-io
        --language-model-only --release-idle-pages)
case $POLICY in
  full)                          POLICY_FLAGS=(); KIND=runtime ;;
  observation_query|random_fill) POLICY_FLAGS=(--active-token-budget 8192); KIND=runtime ;;
  goal_query)                    POLICY_FLAGS=(--active-token-budget 8192 --goal-rewriter-model "${REWRITER:?}"
                                               --admission-overlap 16 --goal-rewriter-batch 4); KIND=runtime ;;
  stock)                         POLICY_FLAGS=(); KIND=stock ;;
  *) echo "unknown POLICY: $POLICY" >&2; exit 2 ;;
esac
OUTPUT="$STAGE/$POLICY"
if [[ $KIND == stock ]]; then
  : "${MODEL:?}"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
  "$PY" -I -B -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name selective-prefix --host 127.0.0.1 --port "$PORT" \
    --distributed-executor-backend uni --tensor-parallel-size 1 \
    --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 16384 \
    --kv-cache-memory-bytes 26843545600 --seed 42 \
    --enable-prefix-caching --prefix-caching-hash-algo sha256 --language-model-only \
    --mamba-cache-mode align --gdn-prefill-backend triton > "$STAGE/server.log" 2>&1 &
else
  : "${RELEASE:?}"
  SERVE=("$PY" -B "$REPO/serve.py" --release-dir "$RELEASE" --policy "$POLICY"
         --output "$OUTPUT" --port "$PORT" "${SHARED[@]}" ${POLICY_FLAGS[@]+"${POLICY_FLAGS[@]}"})
  "${SERVE[@]}" > "$STAGE/dry-run.json"          # dry run: checks the release, prints the vLLM command
  "${SERVE[@]}" --start > "$STAGE/server.log" 2>&1 &
fi
SERVER=$!                                         # the APIServer PID (serve.py execs it)
TUNNEL=""
stop_tunnel() { if [[ -n $TUNNEL ]]; then kill -TERM "$TUNNEL" 2>/dev/null || true; wait "$TUNNEL" 2>/dev/null || true; TUNNEL=""; fi; }
trap stop_tunnel EXIT

# READY: ready.json (this runtime, docs/running.md 4.1) or /health and /v1/models (stock).
ready() {
  if [[ $KIND == stock ]]; then
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

# Reverse tunnel: a Unix socket on the agent host -> this node's 127.0.0.1:$PORT (docs/running.md 6.4).
SSH_OPTS=(-i "$TUNNEL_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes
          -o UserKnownHostsFile="$KNOWN_HOSTS" -o GlobalKnownHostsFile=/dev/null -o ConnectTimeout=20)
ssh -N -R "$AGENT_SOCKET:127.0.0.1:$PORT" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 "${SSH_OPTS[@]}" "$AGENT_HOST" > "$STAGE/tunnel.log" 2>&1 &
TUNNEL=$!
up=0
for _ in $(seq 1 30); do
  kill -0 "$TUNNEL" 2>/dev/null || break
  if ssh -T "${SSH_OPTS[@]}" "$AGENT_HOST" curl -sf --max-time 5 --unix-socket "$AGENT_SOCKET" \
       http://localhost/v1/models 2>/dev/null | grep -q '"selective-prefix"'; then up=1; break; fi
  sleep 2
done

# The Harbor job (Part A.6 to A.8): the task file's trials at -n 16 on the agent host.
agents_rc=0
if (( up == 1 )); then
  timeout "$AGENT_SECONDS" ssh -T "${SSH_OPTS[@]}" "$AGENT_HOST" env $AGENT_ENV "$AGENT_RUN" "$AGENT_SOCKET" \
    "$AGENT_PORT" "$AGENT_OUT" "$KIND" "$AGENT_TASKS" > "$STAGE/agents.log" 2>&1 || agents_rc=$?
  if (( agents_rc == 124 )); then echo "agent time limit reached: the run is partial (B.7)" >&2; fi
else
  echo "tunnel not up (see $STAGE/tunnel.log)" >&2; agents_rc=1
fi

# Close the client sessions the run left open (this runtime; Terminus 2 sends no close, Part A.8).
if [[ $KIND == runtime ]]; then
  "$PY" -I -B - "$OUTPUT" "$PORT" > "$STAGE/closes.jsonl" <<'PY' || true
import json, sys, time, urllib.error, urllib.request
out, port = sys.argv[1], sys.argv[2]
last = {}
with open(out + '/chat_sessions.jsonl', encoding='utf-8') as stream:
    for line in stream:
        event = json.loads(line) if line.strip() else {}
        name = event.get('client_session')
        if event.get('event') in ('bind', 'rebind'):
            last[name] = dict(event, open=True)
        elif event.get('event') == 'close' and name in last and last[name]['slot'] == event.get('slot'):
            last[name]['open'] = False
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for session in (s for s in last.values() if s['open']):
    body = {'session_id': session['session_id']} if session.get('session_id') else {'key_sha256': session['key_sha256']}
    deadline = time.monotonic() + 60
    while True:
        request = urllib.request.Request('http://127.0.0.1:%s/v1/chat-text/close' % port, method='POST',
                                         data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with opener.open(request, timeout=120) as response:
                status = response.status
        except urllib.error.HTTPError as error:
            status = error.code
        if status != 409 or time.monotonic() >= deadline:
            break
        time.sleep(2)
    print(json.dumps(dict(client_session=session['client_session'], slot=session['slot'], http=status)))
PY
fi
stop_tunnel

# Stop: SIGTERM to the server PID only; this runtime must exit 0 with clean receipts (docs/design.md 8).
kill -TERM "$SERVER"
set +e; wait "$SERVER"; server_rc=$?; set -e
echo "server exit $server_rc, agents exit $agents_rc"
if [[ $KIND == runtime ]]; then
  (( server_rc == 0 )) || exit 1
  "$PY" -B "$REPO/examples/slurm/server_checks.py" cleanup "$OUTPUT"
  "$PY" -I -B -c 'import json,sys; v=json.load(open(sys.argv[1])); sys.exit((v.get("session_pool") or {}).get("open_at_shutdown") != [])' \
    "$OUTPUT/engine-cleanup.json" || { echo "sessions open at shutdown" >&2; exit 1; }
fi
exit "$agents_rc"
```

每个服务提交一个作业，使用 README 快速上手中的占位符（`REPO`、`VENV`、`MODELS`、`RELEASE`、`RUNS`）和 [docs/running.zh-CN.md 4.5 节](running.zh-CN.md#45-gpu-作业slurm)的 Slurm 设置 `ACCOUNT`、`PARTITION`、`MAIL`，设置放在提交作业的 shell 里（`--export=ALL`）。下面的例子提交 `observation_query`：

```sh
LABEL=a1                                        # a new label for every attempt
POLICY=observation_query
export REPO VENV RELEASE POLICY
export MODEL="$MODELS/qwen3.6-27b" REWRITER="$MODELS/Qwen3-0.6B-c1899de"
export RUN_ROOT=/abs/short/jobs AGENT_SECONDS=16380
export AGENT_HOST=user@agent-host TUNNEL_KEY=/abs/path/to/tunnel_key KNOWN_HOSTS=/abs/path/to/agent_known_hosts
export AGENT_SOCKET="/abs/path/on/agent/sock/$LABEL-$POLICY.sock" AGENT_PORT=20000
export AGENT_RUN=/abs/path/on/agent/tb/tb_agents.sh AGENT_OUT="/abs/path/on/agent/runs/$POLICY-$LABEL"
export AGENT_TASKS=/abs/path/on/agent/tasks.txt
export AGENT_ENV="AGENT=/abs/path/on/agent HARBOR=/abs/path/on/agent/harbor-env"
sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus=1 \
  --cpus-per-task=16 --mem=113560M --time=05:02:00 --job-name="tb-$POLICY" \
  --output="$RUN_ROOT/slurm-%j.out" --export=ALL --mail-type=BEGIN,END,FAIL --mail-user="$MAIL" \
  /abs/path/to/tb_job.sh
```

- **stock 对照：** `POLICY=stock`、`AGENT_SECONDS=16320`，以及 `--time=04:53:00`。
- **`goal_query`：** `POLICY=goal_query`，并设置 `REWRITER`（不设时脚本拒绝运行）；`AGENT_SECONDS` 和 `--time` 与 `full` 相同，因为 Harbor 的墙钟时间由各任务的 agent 超时决定。
- **时间上限。** `--time` 要覆盖服务启动（Terminal-Bench 的 dry run 中，作业里第一个服务在启动后 223–292 秒 READY，作业 3461661 中为 285 秒；同一作业里的第二个服务为 98–118 秒；[docs/running.zh-CN.md 4.1 节](running.zh-CN.md#41-单个服务)、[docs/reference.zh-CN.md 4.8 节](reference.zh-CN.md#48-cuda-graph)）、Harbor 作业（`AGENT_SECONDS`）、关闭 session 和关闭服务。仅 Harbor 作业本身就需要超过 `build-pov-ray` 12,000 秒的 agent 超时，再加上它的环境准备和 verifier。harness 给 89 个任务的每个服务 16,380 秒的 Harbor 时间上限（stock 为 16,320 秒），作业时间上限为 05:02:00（stock 为 04:53:00），约为预计时间的 1.3 倍（按 65,536 token 做的计划）；剩余时间不足 12,600 秒时，它拒绝启动 Harbor 作业。被 `AGENT_SECONDS` 或作业时间上限截断的运行是不完整的运行（A.11）。
- **dry 子集。** `AGENT_TASKS=/abs/path/on/agent/tasks-dry.txt` 配 `AGENT_SECONDS=1260` 运行 dry 子集（A.1）的 16 个任务。harness 在一个 01:26:00 的 dry 作业里先后运行 `full` 和 `observation_query`，或在一个 00:52:00 的作业里只运行 `observation_query` 或 `goal_query`，每个服务的 Harbor 时间上限为 1,260 秒。
- **停滞。** 这个脚本没有停滞看门狗。harness 的看门狗在有请求等待、且连续 10 分钟既没有请求完成、vLLM 也没有计算任何 token 时停止 Harbor 的运行（A.11）；没有看门狗时，停滞的运行会一直持续到 `AGENT_SECONDS` 或作业时间上限。
- `RUN_ROOT` 必须短：`TMPDIR` 在它下面。每个作业的 `AGENT_SOCKET` 和 `AGENT_OUT` 都必须是新的。

### B.6 就绪检查

- **本 runtime。** READY 指 `server_checks.py ready` 返回 0（[docs/running.zh-CN.md 4.1 节](running.zh-CN.md#41-单个服务)）；`/v1/models` 有响应并不等于 READY。`dry-run.json` 中是 vLLM 命令，必须与 A.3 一致。**不要向本 runtime 的服务发送测试 chat 请求：** 每条新的第一条消息都会绑定一个槽位，打开一个对应不上任何 trial 的客户端 session（A.8），之后的对账和检查都会把它算进去。
- **stock 对照。** `GET /health` 返回 200，且 `GET /v1/models` 列出 `selective-prefix`。
- **kernel 日志行。** 作业目录中的 `kernels.txt` 必须显示 A.5 的各行。
- **在 agent 主机上**，任何 trial 启动之前（作业脚本执行第一项检查，`tb_agents.sh` 执行第二项）：

```sh
curl -sf --unix-socket "$AGENT/sock/$LABEL-$POLICY.sock" http://localhost/v1/models   # lists selective-prefix
ss -ltn 'sport = :20000'                                                             # only 127.0.0.1 listens
```

### B.7 运行 Harbor 作业

- 五个服务依次运行，每个一个作业；agent 主机上从不同时进行两次运行（A.11）。
- 在 agent 主机上查看进度：`$AGENT_OUT/jobs/<job name>/` 下每个 trial 一个目录（`result.json` 中有 `finished_at` 即表示该 trial 已完成），`docker ps` 中最多有 16 个 trial 的容器。在本 runtime 上，在 agent 主机上执行 `curl -s http://127.0.0.1:20000/v1/chat-text/info` 可以看到打开的 session、等待打开的 session 和已用的槽位数；在 GPU 节点上，`server.log` 中的 `CAPACITY_WAIT` 行表示有轮次在等待空间，服务输出目录中的 `capacity.jsonl` 列出空闲 session 的 Page 的释放和重新载入。
- **被截断的运行。** SSH 断开或作业的 `timeout` 不会停止 agent 主机上的 Harbor。先用 `SIGTERM` 停止该次运行的 `harbor` 进程（Harbor 会取消它的 trial 并删除它们的容器），再停止它的 `relay.py`，然后只删除该次运行的 trial 遗留的容器（如果有的话；未经测试）。这次运行是不完整的运行；整个服务换一个新标签重跑（A.11）。

### B.8 准确率

Harbor 已经运行了每个 verifier（A.9）。在 agent 主机上，该次运行结束之后（未经测试）：

```sh
python3 - "$AGENT/runs/$POLICY-$LABEL" "$AGENT/tasks.txt" <<'PY'
import json, sys
from pathlib import Path
out, tasks = Path(sys.argv[1]), Path(sys.argv[2])
planned = [line.strip() for line in tasks.read_text().splitlines() if line.strip()]
trials = {}
for path in sorted(out.glob('jobs/*/*/result.json')):
    result = json.loads(path.read_text())
    name = ((result.get('task_id') or {}).get('name') or str(result.get('task_name') or '')).split('/')[-1]
    trials.setdefault(name, []).append(result)
finished = solved = 0
for name in planned:
    result = (trials.get(name) or [{}])[-1]
    exception = (result.get('exception_info') or {}).get('exception_type')
    done = bool(result.get('finished_at')) and exception != 'CancelledError'
    reward = ((result.get('verifier_result') or {}).get('rewards') or {}).get('reward')
    finished += done
    solved += done and reward == 1.0
    print(json.dumps(dict(task=name, finished=done, reward=reward, exception=exception)))
print(json.dumps(dict(planned=len(planned), finished=finished, solved=solved,
                      accuracy=round(100 * solved / len(planned), 2))))
PY
```

准确率是最后一行的 `accuracy`（A.10）。完整的运行中每个计划内任务都有一个已完成的 trial；少于此数的运行是不完整的运行（A.11）。Harbor 的作业目录包含 prompt、终端输出和 trajectory，不要放进公开发布。

### B.9 吞吐

从一次运行的输出目录计算 A.10 的吞吐（未经测试）：

```sh
python3 - "$AGENT/runs/$POLICY-$LABEL" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
def epoch(value):
    if not value:
        return None
    moment = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).timestamp()
def completion_tokens(paths):
    total = 0
    for path in paths:
        for step in json.loads(path.read_text()).get('steps') or ():
            tokens = (step.get('metrics') or {}).get('completion_tokens')
            if step.get('source') == 'agent' and isinstance(tokens, int):
                total += tokens
    return total
out = Path(sys.argv[1])
starts, ends, main, summary, trials = [], [], 0, 0, 0
for path in sorted(out.glob('jobs/*/*/result.json')):
    result, agent = json.loads(path.read_text()), path.parent / 'agent'
    trials += 1
    phase = result.get('agent_execution') or {}
    if epoch(phase.get('started_at')) is not None:
        starts.append(epoch(phase.get('started_at')))
    if epoch(phase.get('finished_at')) is not None:
        ends.append(epoch(phase.get('finished_at')))
    main += completion_tokens([p for p in [agent / 'trajectory.json'] if p.is_file()]
                              + sorted(agent.glob('trajectory.cont-*.json')))
    summary += completion_tokens(sorted(agent.glob('trajectory.summarization-*.json')))
span = max(ends) - min(starts)
print(json.dumps(dict(trials=trials, main_completion_tokens=main, summarization_completion_tokens=summary,
                      span_s=round(span, 3), tokens_per_s=round(main / span, 3))))
PY
```

完整的运行有 89 个 trial；少于此数的运行是不完整的运行（A.11）。
