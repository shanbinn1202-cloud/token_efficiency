# SWE-bench Verified 正式测试配置

[English](SWEBENCH_BENCHMARK.md) | 简体中文

本文档规定本 runtime 的 SWE-bench Verified 正式测试的配置：`test` split 的前 32 个 instance，Mini-SWE-Agent 2.4.2，最多 16 个 agent 并发、按列表顺序补入，`full`、`observation_query`、`goal_query` 和 `random_fill` 每个策略一个服务，另有一个 stock vLLM 0.24.0 服务作为对照。**Part A** 是配置。baseline（stock vLLM 对照或其他任何系统）只有满足 A.12 检查表的每一项才可比；配置与 agent 在哪台机器上运行无关。**Part B** 是用本仓库工具执行的手动流程，属于可选内容：任何能通过 loopback 访问服务的 x86_64 Docker 主机都可以运行 agent。

“README 4.9 节”这类节号指 [README.zh-CN.md](../README.zh-CN.md)。

## 验证状态

本配置已在一张 GH200 上用 `goal_query` 运行过：runtime `5b15b2f`，前 32 个 instance，16 个 session，2026-10-18（美国中部时间）：

- 作业 3404696：`goal_query`，rewriter 不做批处理。验证结果 `SWE32_PASS_VERIFIED`，合并结果 `JOIN_PASS`；32 个 instance 全部完成，解决 19 个。
- 作业 3404230：`goal_query` 加 `--goal-rewriter-batch 4`（A.3）。验证结果 `SWE32_PASS_VERIFIED`，合并结果 `JOIN_PASS`；32 个 instance 全部完成，解决 20 个。

解决数来自服务端记录与评测结果经过验证的合并（join）。`full`、`observation_query`、`random_fill` 和 `stock` 在 `5b15b2f` 上的运行尚待进行。

在这两次运行之前，本配置的以下部分已在一张 GH200 上运行过（README）：

- 一个服务上 16 个 session：作业 3339019 和 3343069（合成 session；README 4.4 节）。
- session 池（`--max-running-sessions`）：作业 3343069（24 个任务、K = 8，合成 session；README 4.4 节）。
- CUDA graph：作业 3338893、3339448 和 3343069，其中作业 3343069 以 16 个 session 运行 `--cuda-graphs full`（README 4.8 节）。
- 131072 token 的上下文、4096 token 的回复和 `--dram-arena-gib 88`，配合 16 个 `observation_query` session、`--active-token-budget 8192`、`--cuda-graphs full` 和 32 GiB GPU KV：作业 3348705（README 4.9 节）。
- `--admission-overlap 16` 配合 `goal_query`、Qwen3-0.6B Goal rewriter 和 16 个 session：作业 3349504（README 4.10 节）。
- Mini-SWE-Agent 2.4.2 对接本 runtime：10 个 Astropy instance 的 pilot，每个策略一个作业，每个服务 5 个 session，步数上限 90，回复 1536 token（作业 3313802、3314010、3314319、3314623；README 4.7 节）。

尚未运行：上述尚待进行的四个服务（其中包括 stock 对照及其在 A.6 中的两行客户端生成约定）；带 `MAX_RUNNING`、`STEP_LIMIT` 和 `MAX_TOKENS` 的 `run_agents.sh`（README 7.4 节；上面两次运行用的是其自身测试工具的 agent 池）。B.5 的 Slurm 脚本、B.3 的 gold patch 检查和 chat template 检查、B.7 的清理命令和 B.9 的吞吐脚本都未经测试。除上面两次 `goal_query` 运行外，本文档不报告任何结果。

## 与 baseline 对齐的配置

我们的运行所用的取值，供 baseline 照样设置；每个取值都取自 Part A（完整检查表见 A.12，仍然存在的差异见 A.13）：

| 项目 | 取值 |
| --- | --- |
| 模型 | `Qwen/Qwen3.6-27B`，revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`，一张 GH200 120GB，官方 vLLM 0.24.0 wheel，`--mem=113560M`（A.2）；纯文本，两种服务都加 `--language-model-only`（A.3、A.4） |
| Thinking | 关闭：chat template 用 `enable_thinking=false`，生成头中为空的 think 块；stock baseline 发送 `chat_template_kwargs: {"enable_thinking": false}`（A.3、A.4） |
| 解码 | 贪心，由服务端执行；stock baseline 发送 `temperature` 0.0。vLLM `--seed 42`。在 `</mswea_bash_command>` 处停止，标记保留在输出中（A.3、A.4） |
| 上下文上限 | `--max-model-len 131072`，prompt 加回复（A.3） |
| 回复上限 | 每次模型调用 4096 个新 token（`max_tokens` 4096；A.3、A.7） |
| 每个 session 的 KV 预算 | `observation_query`、`goal_query` 和 `random_fill` 用 `--active-token-budget 8192`；`full` 保留所有 Page（A.3） |
| 并发 | 注册 32 个任务，最多 16 个 agent 同时运行，按列表顺序补入；`--max-num-seqs 16`（A.1、A.8） |
| 本 runtime 的 KV cache | 32 GiB GPU KV：`--kv-cache-gib 32`，34,359,738,368 字节；主机侧一个 70 GiB 的锁页 Page arena（`--dram-arena-gib 70`）和 1 GiB 的 connector 池（`--connector-pool-gib 1`）（A.3） |
| stock 对照（vLLM） | `--kv-cache-memory-bytes 34359738368`（32 GiB）、`--max-model-len 131072`、`--max-num-batched-tokens 16384`、`--max-num-seqs 16`；CUDA graph：vLLM 默认（`FULL_AND_PIECEWISE`）；异步调度：vLLM 默认（开启）；`--enable-prefix-caching --prefix-caching-hash-algo sha256`；不加 `--attention-backend`（vLLM 选择 FLASH_ATTN 加 FlashAttention 3）；`--language-model-only`（A.4、A.5） |
| 吞吐 | 32 个 instance 的主 agent 每次模型调用在客户端记录的 completion token 之和 / agent 主机时钟上从第一个 agent 开始到最后一个 agent 结束的墙钟时间。包含工具执行、任务容器的启动和停止、各种等待以及补入前的 session 关闭；不包含服务的启动和关闭、评测以及 Goal rewriter 自己的 token（A.10） |
| 任务 | SWE-bench Verified `test`，按数据集顺序的前 32 个 instance（22 个 Astropy、10 个 Django）；Mini-SWE-Agent 2.4.2（litellm 1.103.1），使用 PoC 的 runner 和 prompt，步数上限 120（A.1、A.6） |
| 评测 | swebench 5.0.2，在 agent 主机（x86_64，Docker）上运行，`SWE-bench/SWE-bench_Verified` `test`，`--timeout 1800`；agent 的任务容器使用官方镜像 `docker.io/swebench/sweb.eval.x86_64.*`（A.6、A.9） |
| Gemma 4 31B IT | 打算使用相同的设置；本文档未固定。该 profile 仍在开发中，未经 GPU 验证（README 4.14 节），并且拒绝 `--goal-rewriter-batch`（README 4.10 节） |

---

## Part A. 配置

### A.1 范围与 instance

| 项目 | 取值 |
| --- | --- |
| 模型 | Qwen3.6-27B，一张 NVIDIA GH200 120GB，vLLM 0.24.0（A.2） |
| 基准 | SWE-bench Verified，`test` split，**按数据集顺序的前 32 个 instance**：先 22 个 Astropy instance，再 10 个 Django instance |
| 并发 | 注册 32 个任务，**最多 16 个 agent 同时运行**。一个 agent 结束后，按列表顺序启动下一个 instance（补入）；不是分两批、每批 16 个 |
| 服务 | 每个策略一个全新的服务，各自一个 GPU 作业：`full`、`observation_query`、`goal_query`、`random_fill`（本 runtime），以及 `stock`（stock vLLM 0.24.0，对照） |
| 运行 | 一个服务把 32 个 instance 各跑一次，中间不中断（A.11）。一次运行就是一个样本（A.13） |

32 个 instance ID，按顺序（B.1 把它存为 `ids.txt`）：

```text
astropy__astropy-12907
astropy__astropy-13033
astropy__astropy-13236
astropy__astropy-13398
astropy__astropy-13453
astropy__astropy-13579
astropy__astropy-13977
astropy__astropy-14096
astropy__astropy-14182
astropy__astropy-14309
astropy__astropy-14365
astropy__astropy-14369
astropy__astropy-14508
astropy__astropy-14539
astropy__astropy-14598
astropy__astropy-14995
astropy__astropy-7166
astropy__astropy-7336
astropy__astropy-7606
astropy__astropy-7671
astropy__astropy-8707
astropy__astropy-8872
django__django-10097
django__django-10554
django__django-10880
django__django-10914
django__django-10973
django__django-10999
django__django-11066
django__django-11087
django__django-11095
django__django-11099
```

本 runtime 的四个策略（README 开头的策略表和第 5 节）：

| `--policy` | 下一轮使用哪些历史 Page |
| --- | --- |
| `full` | 所有历史 Page：参照组，也是最接近 stock vLLM 的配置 |
| `observation_query` | Selector V2：在预算内保留必保留页、recent 页和余弦最高的页；query 是最新 observation 末尾的 hidden state |
| `goal_query` | 同一个 selector；query 是持续维护的检索 Goal，由 Qwen3-0.6B rewriter 更新 |
| `random_fill` | 两个 query 策略的对照：预算、必保留页和 recent 页相同，剩余容量按带种子的随机顺序填满 |

### A.2 模型与版本

| 项目 | 取值 |
| --- | --- |
| 模型 | `Qwen/Qwen3.6-27B`，revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`，本地目录，必须是普通文件；元数据的 SHA-256 见 README 2.2 节 |
| Goal rewriter（仅 `goal_query`） | `Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`；同一组对比中每个 `goal_query` 服务使用同一个解析后的路径（README 2.2 节） |
| vLLM | 官方 wheel `vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl`，未作任何修改，SHA-256 `700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5` |
| Python 环境 | CPython 3.12、`transformers==5.13.0`、`tokenizers==0.22.2`，用 `examples/env/constraints-3270721.txt` 构建（README 2.1 节）。stock 对照从同一个环境运行；本 runtime 安装到单独的 release 目录，环境本身保持不变（README 2.4 节） |
| Runtime | 本仓库的 commit `5b15b2f`（`5b15b2ff2d7497717dffdb4a77641781a8c1090e`，“Batch the Goal rewriter calls that queue”），即 benchmark 当前各次运行所用的 commit；或者不改变 XML 协议路径和 A.3 的 vLLM 参数的更新 commit；`main` 包含它。历史：这个 benchmark 早先的运行使用 `e877051`，它只用加载的 tokenizer 对每条消息编码，与 stock vLLM 相同；用 `3496d60` 时有一个 Django instance 被以 HTTP 500 拒绝 |
| GPU | 一张 NVIDIA GH200 120GB，Linux aarch64；一张 GPU 同一时间只跑一个服务 |
| Slurm 资源 | `--nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16 --mem=113560M`（节点主机内存中一张 GH200 的份额；A.3 说明为什么够用） |
| Agent 软件 | Mini-SWE-Agent `2.4.2`、litellm `1.103.1`、datasets `5.0.1`，Python 3.12 |
| 评测程序 | swebench `5.0.2`、datasets `5.0.1` |

### A.3 本 runtime：服务参数与 vLLM 参数

**四个策略共用的参数**（`serve.py`）：

| 参数 | 取值 | 作用 |
| --- | --- | --- |
| `--task` | `tasks-pool.json`：按顺序包含 A.1 的 32 个 instance，由 `examples/swebench/build_tasks.py --pool` 生成 | 注册 32 个任务；每个请求按其任务消息中的 problem statement 路由（README 4.4 节） |
| `--max-running-sessions` | `16` | session 池：同时最多打开 16 个 session；session 在其任务的第一个请求时打开，由 agent 池关闭（A.8）。vLLM 收到 `--max-num-seqs 16` |
| `--kv-cache-gib` | `32` | vLLM 收到 `--kv-cache-memory-bytes 34359738368` |
| `--page-tokens` / `--kv-layout` | `16` / `decoupled` | 16 token 的注意力 Page，以及一个单独的、有 33 个 slot 的 Gated-DeltaNet 池（README 4.6 节）。vLLM 收到 `--block-size 16 --mamba-block-size 784 --additional-config ...` |
| `--max-model-len` | `131072` | 上下文上限（prompt 加回复）。prompt 最多 131,072 − 4,096 = 126,976 token（README 4.9 节） |
| `--max-tokens` | `4096` | 回复上限，由服务端对每个请求强制执行（README 4.9 节） |
| `--dram-arena-gib` | `70` | 所有 session 共用一个 70 GiB 的锁页主机 Page arena：71,680 个 1 MiB 的 Page。它能容纳 8 个处于 131,072 token 上限的 session（每个 8,193 个 Page），而不是 16 个；此前 16 个 session 的 SWE-bench 运行中最大的 16 个 session 同时打开需要 58.32 GiB。超出它的捕获会让该请求以 `DRAM_ARENA_EXHAUSTED` 失败（README 4.9 节） |
| `--connector-pool-gib` | `1` | connector 闲置的 CPU 池设为 1 GiB 而不是 32 GiB（README 4.9 节）。vLLM 收到 `cpu_bytes_to_use` 1073741824 |
| `--cuda-graphs` | `full` | vLLM 收到 `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}`，取代 `--enforce-eager`（README 4.8 节） |
| `--decode-fastpath` / `--decode-fastpath2` | 开启 / 开启 | 每个解码步的主机工作更少，结果相同；第二阶段需要第一阶段（README 4.11 节）。不增加 vLLM 参数 |
| `--language-model-only` | 开启 | 不加载视觉塔，没有多模态预热（README 4.12 节）。vLLM 收到 `--language-model-only` |
| `--protocol` | `mswea-xml`（默认值，不传） | Mini-SWE-Agent XML 协议（README 6.1 节） |
| `--port` | `8000` | 服务只绑定 `127.0.0.1` |

**各策略的参数：**

| 策略 | selector 参数 | admission 参数 | Goal rewriter |
| --- | --- | --- | --- |
| `full` | 无 | 无 | 无 |
| `observation_query` | `--active-token-budget 8192` | 无 | 不启动（README 第 5 节） |
| `random_fill` | `--active-token-budget 8192`（填充种子：默认值 42） | 无 | 不启动（README 第 5 节） |
| `goal_query` | `--active-token-budget 8192 --goal-rewriter-model <Qwen3-0.6B 目录> --goal-rewriter-batch 4` | `--admission-overlap 16` | 同一张 GPU 上一个 Qwen3-0.6B 进程（约 2 GiB）。`--goal-rewriter-batch 4` 是论文配置的设置：参与比较的是批处理的那次运行。rewriter 的批量贪心解码与逐个调用并非逐位一致，所以只在该设置相同的服务之间比较策略（README 4.10 节）；作业 3404230 的 59 次 Goal fallback 中有 9 次是只在批处理时出现的失败（4 次 CUDA out-of-memory，5 次 cuDNN attention 错误；作业 3404696 中没有），其恢复逻辑正在修复（尚未进入 `main`） |

任何服务都不设置 `--gdn-device-handoff`、`--phase1-in-step`、`--active-ratio`、`--goal-query-variant`、`--fill-seed`、`--no-goal-rewriter`、`--decode-fastpath3` 或 `--release-idle-pages`。admission 参数遵循 README 4.10 节（只有 `goal_query` 加 `--admission-overlap 16`）。selector、rewriter、admission 参数和解码快速路径会固定到启动配置里，但不增加任何 vLLM 参数，所以四个服务启动 vLLM 时的参数完全相同。

**主机内存。** 锁页主机内存是 1 GiB 的 connector 池、Gated-DeltaNet carry（2.29 GiB）和 70 GiB 的 arena，在服务的 dry run 中共 73.29 GiB。再留约 32 GiB 给进程，正好放进一张 GH200 的份额 `--mem=113560M`（110.9 GiB）；`--language-model-only` 让作业避开 vLLM 的多模态预热，它会在不到一秒内占用数十 GiB 主机内存（README 4.12 节）。本配置早先的版本使用 `--dram-arena-gib 88`、默认的 32 GiB connector 池和 `--mem=200g`；作业 3348705（见“验证状态”）在申请 200 GiB 的情况下运行了 88 GiB 的 arena（README 4.9 节）。

**命令。** 四个服务的 dry run（加 `--start` 即启动服务；`TASKS` 是 `tasks-pool.json` 的路径，`LABEL` 是每次尝试的新标签）：

```sh
SHARED=(--max-running-sessions 16 --kv-cache-gib 32 --page-tokens 16 --kv-layout decoupled
        --max-model-len 131072 --max-tokens 4096 --dram-arena-gib 70 --connector-pool-gib 1 --cuda-graphs full
        --decode-fastpath --decode-fastpath2 --language-model-only)
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy full "${SHARED[@]}" \
  --output "$RUNS/full-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy observation_query "${SHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy random_fill "${SHARED[@]}" \
  --active-token-budget 8192 --output "$RUNS/random_fill-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --task "$TASKS" --policy goal_query "${SHARED[@]}" \
  --active-token-budget 8192 --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --goal-rewriter-batch 4 \
  --admission-overlap 16 --output "$RUNS/goal_query-$LABEL" --port 8000
```

**由此得到的 vLLM 参数**（四个服务相同；以 dry run 打印的列表为准，这里按参数组换行）：

```text
--model <model directory> --served-model-name selective-prefix --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
--host 127.0.0.1 --port 8000 --distributed-executor-backend uni --no-async-scheduling
--tensor-parallel-size 1 --pipeline-parallel-size 1
--max-num-seqs 16 --max-model-len 131072 --max-num-batched-tokens 16384
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

**生成约定，由服务端执行**（README 6.1 和 4.9 节）。服务端忽略客户端的采样字段：
- 贪心解码，最多 4096 个新 token；解码文本中出现 `</mswea_bash_command>` 或遇到 EOS/pad 时停止；
- 关闭 thinking：system 和任务消息走固定的 chat template，`enable_thinking=false`，生成头为 `<|im_start|>assistant\n<think>\n\n</think>\n\n`；
- 历史中的每条旧回复都换回它生成时的原始 token ID；
- 响应内容是全部生成 token 解码后的文本，包括结束标记，以及最后一个 token 越过标记携带的内容。

### A.4 stock vLLM 对照

对照服务就是从同一个环境（A.2）直接启动的 `vllm.entrypoints.openai.api_server`，不带本 runtime 的任何代码。

| 设置 | 取值 | 与本 runtime 的关系 |
| --- | --- | --- |
| `--model` | 同一个 Qwen3.6-27B 目录 | 相同 |
| `--served-model-name` | `selective-prefix` | 同名，因此 agent 使用相同的模型名 |
| `--host` / `--port` | `127.0.0.1` / `8000` | 相同（只用 loopback） |
| `--distributed-executor-backend` / `--tensor-parallel-size` | `uni` / `1` | 相同 |
| `--max-num-seqs` | `16` | 与 K = 16 相同 |
| `--max-model-len` | `131072` | 相同 |
| `--max-num-batched-tokens` | `16384` | 相同（由 launcher 固定） |
| `--kv-cache-memory-bytes` | `34359738368`（32 GiB） | 字节数相同 |
| `--seed` | `42` | 相同 |
| `--enable-prefix-caching --prefix-caching-hash-algo sha256` | 开启 | 相同 |
| `--language-model-only` | 开启 | 相同（README 4.12 节） |
| `--mamba-cache-mode` | `align` | 相同 |
| `--gdn-prefill-backend` | `triton` | 相同 |
| CUDA graph | vLLM 默认：既不加 `--enforce-eager` 也不加 `--compilation-config`，结果是 `FULL_AND_PIECEWISE` 加 `VLLM_COMPILE`（README 4.8 节） | 模式相同（`--cuda-graphs full`） |
| 异步调度 | vLLM 默认（开启），不加参数 | 不同：launcher 总是传 `--no-async-scheduling`（A.13） |
| 注意力后端 | 不设置，由 vLLM 选择（A.5） | 相同 |
| Thinking | 关闭，由客户端请求（`chat_template_kwargs`，A.6） | 效果相同，由服务端强制 |

每个取值都显式设置，这样对照服务不依赖 vLLM 对这个混合模型的默认值。只服务于 runtime 自身路由的参数（`--revision`、reasoning 和 tool parser、`--chat-template`、`--kv-transfer-config`、`--block-size`、`--mamba-block-size`、`--additional-config`）不传：stock 服务使用模型自带的 chat template，回复以普通的 `content` 返回。

**命令**，在设置了 A.3 环境变量的 shell 中：

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
"$VENV/bin/python" -I -B -m vllm.entrypoints.openai.api_server \
  --model "$MODELS/qwen3.6-27b" --served-model-name selective-prefix --host 127.0.0.1 --port 8000 \
  --distributed-executor-backend uni --tensor-parallel-size 1 \
  --max-num-seqs 16 --max-model-len 131072 --max-num-batched-tokens 16384 \
  --kv-cache-memory-bytes 34359738368 --seed 42 \
  --enable-prefix-caching --prefix-caching-hash-algo sha256 --language-model-only \
  --mamba-cache-mode align --gdn-prefill-backend triton
```

- **环境变量。** 与 A.3 相同（`PATH`、`LD_LIBRARY_PATH`、短 `TMPDIR`、`VLLM_USE_FLASHINFER_SAMPLER=0`、unset `CC` 和 `VLLM_RPC_BASE_PATH`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`PYTHONDONTWRITEBYTECODE=1`），另加 `serve.py` 为本 runtime 设置的两个变量（`VLLM_WORKER_MULTIPROC_METHOD=spawn`、`VLLM_USE_V2_MODEL_RUNNER=0`）和 `VLLM_NO_USAGE_STATS=1`。
- **生成约定，由客户端发送。** stock vLLM 从请求中读取采样设置，所以由 agent 的模型配置携带（A.6）：`temperature` 0.0、`max_tokens` 4096、`stop` `["</mswea_bash_command>"]` 加 `include_stop_str_in_output: true`，以及 `chat_template_kwargs: {"enable_thinking": false}`。在 `enable_thinking=false` 下，模型的 chat template 预期以 A.3 的生成头结束 prompt；B.3 检查这一点。
- **就绪判断。** `GET /health` 返回 200，且 `GET /v1/models` 列出 `selective-prefix`。
- **关闭 session。** stock vLLM 没有 `/v1/sessions/close` 路由。agent 池的关闭请求会得到 HTTP 404，这是预期行为（A.8）。
- **抢占。** KV 池满时 stock vLLM 可能抢占请求；本 runtime 从不抢占（A.13）。运行结束时记录 vLLM 的计数器 `vllm:num_preemptions_total`（`GET /metrics`）。

### A.5 kernel 后端

两种服务都不传 `--attention-backend`；注意力后端由 vLLM 在 GH200 上自动选择。两种服务使用相同的 kernel：

| 组件 | 设置 | 服务日志行 |
| --- | --- | --- |
| 注意力 | FLASH_ATTN 加 FlashAttention 3（vLLM 自动选择） | `Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']` 和 `Using FlashAttention version 3` |
| Gated-DeltaNet prefill | `--gdn-prefill-backend triton` | `Using Triton/FLA GDN prefill kernel (requested=triton, head_k_dim=128)` |
| Sampler | 原生 top-k/top-p（`VLLM_USE_FLASHINFER_SAMPLER=0`） | `FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0` |
| CUDA graph | `FULL_AND_PIECEWISE`，16 个序列时捕获尺寸为 1、2、4、8、16、24 和 32（README 4.8 节） | 引擎配置行中的 `'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE` 和 `'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32]` |
| 异步调度 | 本 runtime：关闭。stock：开启 | `Asynchronous scheduling is disabled.`（本 runtime）或 `Asynchronous scheduling is enabled.`（stock） |

READY 之后从每个服务的日志中确认：

```sh
grep -oE "Using FLASH_ATTN attention backend|Using FlashAttention version [0-9]+|Using Triton/FLA GDN prefill kernel \(requested=[a-z]+|FlashInfer top-p/top-k sampling disabled|Asynchronous scheduling is [a-z]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|'cudagraph_capture_sizes': \[[0-9, ]*\]" \
  server.log | sort | uniq -c
```

表中每一行都必须出现，且不能出现其他注意力后端、FlashAttention 版本、prefill kernel 或 CUDA graph 模式。日志不符的服务不纳入对比。

### A.6 Agent 配置

除标注外，所有服务完全相同。

| 项目 | 取值 |
| --- | --- |
| Agent | Mini-SWE-Agent 2.4.2（litellm 1.103.1、datasets 5.0.1） |
| Runner | PoC 的 `examples/swebench/upstream/swebench_runner.py`，SHA-256 `34ca6d9288c72830e61bd00ad8b15790cab4c29515db24b8df30775cfc0880a7` |
| Prompt | PoC 的 `examples/swebench/upstream/swebench_prompt_override.yaml`，SHA-256 `1b6a48666ad528775bd11225fa4d5c002721c25c0cdbc05b99cc15b2de31336a`（LF 字节） |
| 配置叠加 | `-c swebench_xml.yaml`（Mini-SWE-Agent 内置）`-c swebench_prompt_override.yaml -c model_override.yaml` |
| 每个 instance 一个 runner | `swebench_runner.py --subset verified --split test --workers 1 --filter '^<instance_id>$' --output <out>/runs/<instance_id>`，加上配置叠加（带 `MAX_RUNNING` 的 `run_agents.sh`） |
| Runner 用的数据集 | `princeton-nlp/SWE-Bench_Verified`（`--subset verified`），提前缓存，离线读取 |
| 模型 | `litellm_textbased`、`openai/selective-prefix`、`api_base` `http://127.0.0.1:<port>/v1`、占位 key、`temperature` 0.0、`max_tokens` 4096、`cost_tracking: ignore_errors`；客户端超时与重试见 A.7；stock 对照另加 A.4 的生成约定 |
| 工具环境 | `docker`，镜像 `docker.io/swebench/sweb.eval.x86_64.<把 instance id 中的 "__" 换成 "_1776_" 并转成小写>:latest`，`cwd` `/testbed`，单条命令 120 秒，`bash -c`，`PAGER=cat`、`MANPAGER=cat`、`LESS=-R`、`PIP_PROGRESS_BAR=off`、`TQDM_DISABLE=1` |
| 进程环境变量 | 不设置 `MSWEA_CTX_PATCH`，`AGENTDIET_MODE=off`，`HF_HUB_OFFLINE=1`，`HF_DATASETS_OFFLINE=1`，`MSWEA_GLOBAL_CONFIG_DIR` 指向一个空目录，`PYTHONDONTWRITEBYTECODE=1`，`MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1`，`REQUEST_TIMEOUT=3600` |

PoC 文件中会影响结果的性质（README 7.1 节）：
- 达到步数上限时，runner 提交 agent 最后打印过的、至少 3 行的 `git diff`（退出状态 `LimitsExceeded_InformalSubmission`）。连续 10 条相同命令时终止该 instance（`LoopDetected`）。
- prompt 是为 Astropy 写的（"The testbed is an Astropy repository"，并带有 Astropy 专用的测试建议）。32 个 instance（包括 10 个 Django instance）都原样使用它。baseline 使用同一个文件；换用其他 prompt 就是另一种配置。

**本 runtime 的 `model_override.yaml`**：由 `run_agents.sh` 以端口 20000 和 `STEP_LIMIT=120 MAX_TOKENS=4096` 渲染，模板是 `examples/swebench/model_override.yaml` 的副本，其中加了 `timeout`、`max_retries` 和 `num_retries` 三行客户端设置（B.2）：

```yaml
model:
  model_name: "openai/selective-prefix"
  model_class: litellm_textbased
  cost_tracking: ignore_errors
  model_kwargs:
    api_base: "http://127.0.0.1:20000/v1"
    api_key: "dummy"
    temperature: 0.0
    max_tokens: 4096
    timeout: 3600
    max_retries: 0
    num_retries: 0

environment:
  environment_class: docker
  cwd: "/testbed"
  timeout: 120
  interpreter: ["bash", "-c"]
  env:
    PAGER: cat
    MANPAGER: cat
    LESS: -R
    PIP_PROGRESS_BAR: "off"
    TQDM_DISABLE: "1"

agent:
  step_limit: 120
  cost_limit: 0
```

**stock 对照的 `model_override.yaml`**：同一个文件，在 `num_retries` 之后多两行，携带 A.4 的客户端生成约定：

```yaml
model:
  model_name: "openai/selective-prefix"
  model_class: litellm_textbased
  cost_tracking: ignore_errors
  model_kwargs:
    api_base: "http://127.0.0.1:20000/v1"
    api_key: "dummy"
    temperature: 0.0
    max_tokens: 4096
    timeout: 3600
    max_retries: 0
    num_retries: 0
    extra_body: {"chat_template_kwargs": {"enable_thinking": false}, "include_stop_str_in_output": true}
    stop: ["</mswea_bash_command>"]

environment:
  environment_class: docker
  cwd: "/testbed"
  timeout: 120
  interpreter: ["bash", "-c"]
  env:
    PAGER: cat
    MANPAGER: cat
    LESS: -R
    PIP_PROGRESS_BAR: "off"
    TQDM_DISABLE: "1"

agent:
  step_limit: 120
  cost_limit: 0
```

两种服务都有的客户端设置已在两次 `goal_query` 运行中用过（见“验证状态”）；stock 的两行生成约定还没有在 agent 运行中用过。

### A.7 限制

| 限制 | 取值 |
| --- | --- |
| 步数上限 / 费用上限 | `step_limit` 120 / `cost_limit` 0 |
| 回复上限 | 每次模型调用最多 4096 个新 token（本 runtime：由服务端强制；stock：客户端的 `max_tokens`） |
| 上下文上限 | 131,072 token（prompt 加回复）；本 runtime 在 admission 之前以 HTTP 400 拒绝超过 126,976 token 的 prompt（README 4.9 节） |
| 工具命令超时 | 每条命令 120 秒 |
| 客户端超时 | 每次模型调用 3600 秒（`timeout: 3600`、`REQUEST_TIMEOUT=3600`） |
| 客户端重试 | 无：`max_retries: 0` 和 `num_retries: 0`（litellm 和 OpenAI 客户端），`MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1`（Mini-SWE-Agent） |
| 每个服务的请求数 | 32 个 instance × 120 步 = 最多 3,840 个请求，在本 runtime 的生命周期上限 max(1,024, 256 × 32) = 8,192 之内（README 4.9 节） |

**为什么不做客户端重试、超时设为 3600 秒。** 在本 runtime 上，客户端重试一个仍在处理中的请求，就是对一个忙碌的 session 发出第二个请求；超时或非 2xx 状态会结束该服务的这次运行（README 6.3 节）。在 `--admission-overlap 16` 下，`goal_query` 的 admission 可能要等待唯一的 Goal rewriter 进程，前端最多等待一次 admission 2,100 秒（README 4.10 节），之后才开始最多 4096 token 的回复。3600 秒的客户端超时高于这段等待加上一次完整回复，所以慢的 admission 能完成，而不会被客户端放弃并重试。stock 对照使用相同的设置，使两边只在服务端不同。

### A.8 并发、补入与关闭 session

- 同时最多运行 16 个 agent，每个 instance 一个 runner 进程（`--workers 1`），按 A.1 的顺序启动。一个 runner 结束后，启动下一个 instance（带 `MAX_RUNNING=16` 的 `run_agents.sh`）。
- 在本 runtime 上，结束的 instance 的 session 先被关闭，然后它的位置才被复用：`POST /v1/sessions/close`，请求体 `{"instance_id": "<instance id>"}`，遇到 HTTP 409（`SESSION_BUSY`）时每隔 2 秒重试，最多 30 次（README 6.5 节）。关闭会释放该 session 在 GPU 和主机上的状态，下一个等待的任务随即打开。关闭所用的时间会让下一个 instance 晚同样长的时间启动。
- 在 stock 对照上，同样的请求会立即返回 HTTP 404；stock vLLM 把缓存的 prefix 保留到它自行淘汰为止。
- `run_agents.sh` 在 agent 主机上为每个 instance 向 `timeline.jsonl` 写一行：`instance_id`、`start`（runner 启动前一刻记录）、`end`（runner 进程退出时记录，在关闭请求之前）、`runner_exit` 和 `close_http`。它还写出 `runs/<instance>/`、`logs/<instance>.log`、`closes/<instance>.json` 和合并后的 `preds.json`。

### A.9 评测

| 项目 | 取值 |
| --- | --- |
| 评测程序 | swebench 5.0.2，在 agent（Docker）主机上运行，在该次运行的所有 agent 结束之后 |
| 数据集 | `SWE-bench/SWE-bench_Verified`，split `test` |
| 命令 | `python -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Verified --split test --predictions_path predictions.jsonl --max_workers 2 --timeout 1800 --run_id <run_id>`（`examples/swebench/evaluate.sh`，README 7.5 节） |
| 超时 | 每个 instance `--timeout 1800` |
| Worker 数 | `--max_workers 2`（`examples/swebench/evaluate.sh` 的默认值）。worker 数只决定同时评测多少个 instance，任何 worker 数结果都相同 |
| Predictions | 每个计划内的 instance 一行（32 行）。没有 prediction 的 instance 用空 patch 评测，计为未解决 |

### A.10 指标

**准确率** = 已解决的 instance 数 / 32 × 100，取自评测程序的 `resolved_ids`。一个 instance 是 3.125 分。准确率为 75 % 时，一次运行的二项分布标准误约为 7.7 分，两次独立运行之差的标准误约为 10.8 分（3.5 个 instance）。保留每个 instance 是否解决的列表，以便逐个 instance 比较各次运行。

**吞吐** = completion token 总数 / 墙钟时间：

| 部分 | 定义 |
| --- | --- |
| 分子 | 32 个 instance 的主 agent 每次模型调用的 `usage.completion_tokens` 之和，取自客户端侧 trajectory 中的记录（`extra.response.usage.completion_tokens`）。调用包括 assistant 消息和格式错误消息（`extra.interrupt_type == "FormatError"`） |
| 分母 | agent 主机时钟上，从第一个 agent 开始到最后一个 agent 结束的墙钟时间：`timeline.jsonl` 上的 `max(end) − min(start)`（A.8） |
| 包含 | 工具执行、任务容器的启动和停止、等待空闲 agent 位置或 session、每次补入之前的关闭 session 请求（A.8）、admission 等待、Goal rewriter 的时间（`goal_query`） |
| 不包含 | 服务的启动和关闭、评测、最后一个 agent 结束之后的关闭 session 请求，以及 Goal rewriter 自己的 token（辅助模型） |

每次运行还要记录：每个 instance 的墙钟时间、步数和退出状态；上下文超限被拒绝的次数；stock 的 vLLM 抢占计数（A.4）。

### A.11 运行规则

- **agent 主机上同一时间只跑一次运行**，agent 运行期间不做评测：吞吐包含工具时间，CPU 争用会改变它。要比较吞吐的各个服务，使用同一类 agent 主机。
- **每次尝试都用全新的服务和新的输出目录。** `serve.py`、`run_agents.sh` 和 `evaluate.sh` 都拒绝已存在的输出目录。
- **中断的运行不续跑。** 提前结束的运行（时间上限、隧道断开、服务失败）是不完整的运行：未完成的 instance 不在另一个作业里接着跑，该次运行不纳入对比，整个服务换一个新标签从头重跑。
- **失败的计法。** 因 prompt 超过上下文上限而被 HTTP 400 拒绝的请求只结束该 instance（计为未解决），服务继续工作（README 4.9 节）。其他任何非 2xx 状态、超时、断连或服务退出都会结束这次运行（README 6.3 节），该次运行即为不完整的运行。
- **传输方式。** agent 通过只走 loopback 的路径访问服务（B.4）。任何只走 loopback 的路径给出相同的结果；服务从不在 loopback 之外监听（README 6.4 节）。

### A.12 对齐检查表

| # | 项目 | 本 runtime | baseline 使用 |
| --- | --- | --- | --- |
| 1 | 模型与 revision | Qwen3.6-27B `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | 相同 |
| 2 | 软件 | vLLM 0.24.0 官方 wheel、Transformers 5.13.0、Tokenizers 0.22.2、`constraints-3270721.txt` | 同一个环境 |
| 3 | 硬件与作业 | 1 × GH200 120GB、16 个 CPU、113,560 MiB | 相同 |
| 4 | instance | A.1 的 32 个 ID，按该顺序 | 相同 |
| 5 | 并发 | 32 个中最多 16 个同时运行，按列表顺序补入 | 相同 |
| 6 | 上下文上限 | 131072 | `--max-model-len 131072` |
| 7 | 每步 batched token 数 | 16384 | `--max-num-batched-tokens 16384` |
| 8 | 最大序列数 | 16 | `--max-num-seqs 16` |
| 9 | GPU KV | 32 GiB（34,359,738,368 字节） | `--kv-cache-memory-bytes 34359738368` |
| 10 | 回复上限 | 4096（服务端） | `max_tokens` 4096（客户端） |
| 11 | 解码 | 贪心（服务端） | `temperature` 0.0 |
| 12 | Thinking | 关闭（`enable_thinking=false`，生成头中为空的 think 块） | `chat_template_kwargs: {"enable_thinking": false}` |
| 13 | 停止标记 | `</mswea_bash_command>`，保留在输出中 | `stop` 加 `include_stop_str_in_output: true` |
| 14 | CUDA graph | `FULL_AND_PIECEWISE`（`--cuda-graphs full`） | vLLM 默认（`FULL_AND_PIECEWISE`） |
| 15 | 异步调度 | 关闭（总是） | vLLM 默认（开启）；已知差异（A.13） |
| 16 | 注意力 | FLASH_ATTN 加 FlashAttention 3，由 vLLM 选择 | 相同，不加 `--attention-backend` |
| 17 | Gated-DeltaNet prefill | Triton | `--gdn-prefill-backend triton` |
| 18 | Sampler | 原生（`VLLM_USE_FLASHINFER_SAMPLER=0`） | 同一个变量 |
| 19 | Prefix caching | 开启，sha256 | `--enable-prefix-caching --prefix-caching-hash-algo sha256` |
| 20 | Mamba cache 模式 | `align` | `--mamba-cache-mode align` |
| 21 | Seed | 42 | `--seed 42` |
| 22 | 纯文本 | `--language-model-only` | vLLM 的 `--language-model-only` |
| 23 | Agent | Mini-SWE-Agent 2.4.2、litellm 1.103.1、datasets 5.0.1 | 相同 |
| 24 | Runner 与 prompt | PoC 文件，SHA-256 `34ca6d92…` 和 `1b6a4866…`（32 个 instance 都用 Astropy prompt） | 相同文件 |
| 25 | 配置叠加 | `swebench_xml.yaml`、prompt override、model override | 相同 |
| 26 | 步数上限 / 费用上限 | 120 / 0 | 相同 |
| 27 | 工具环境 | Docker、`/testbed`、120 秒、A.6 的环境变量 | 相同 |
| 28 | 进程环境变量 | `AGENTDIET_MODE=off`、离线缓存、空的 `MSWEA_GLOBAL_CONFIG_DIR` | 相同 |
| 29 | 客户端超时与重试 | 3600 秒，不重试 | 相同 |
| 30 | Agent 主机 | 同一时间只跑一次运行，运行期间不评测 | 同样的规则，同一类主机 |
| 31 | 评测 | swebench 5.0.2、`SWE-bench/SWE-bench_Verified`、`test`、`--timeout 1800`、2 个 worker | 相同 |
| 32 | 吞吐 | A.10 的定义 | 相同的分子和计时 |
| 33 | 不完整的运行 | 不纳入，不续跑 | 相同 |

### A.13 本 runtime 与 stock vLLM 的已知差异

即使检查表的每一项都一致，以下差异仍然存在。任何对比都要一并报告。

1. **历史回复。** 本 runtime 以每条旧回复生成时的原始 token ID 回放；stock vLLM 通过 chat template 从文本重新渲染历史（README 6.1 节）。
2. **异步调度。** 本 runtime 不使用（launcher 总是传 `--no-async-scheduling`）；stock vLLM 使用其默认值，即开启。
3. **采样由谁控制。** 本 runtime 在引擎内强制贪心解码、回复上限和停止标记，忽略客户端字段；stock vLLM 从请求中读取。在标记处，本 runtime 返回最后一个 token 解码出的全部内容，包括越过标记的字符（README 6.1 节）；stock vLLM 在停止字符串处截断文本。
4. **上下文边界。** 本 runtime 以 HTTP 400 拒绝超过 126,976 token 的 prompt（README 4.9 节）；stock vLLM 则把回复缩短到上下文剩余的长度（README 4.9 节描述了 vLLM 的这一行为；未在 stock 对照上测量）。
5. **KV 管理。** 两者的 GPU KV 都是 32 GiB，但布局不同。本 runtime 使用 16 token 的注意力 Page、单独的 Gated-DeltaNet 池、1 GiB 的 CPU 池和 70 GiB 的锁页 Page arena；它从不抢占，放不下的一轮会等待（`CAPACITY_WAIT`，README 4.4 节）。在 131,072 token 下，32 GiB 放不下 16 个都处于完整上下文的 `full` session（README 4.9 节），所以 `full` 的轮次可能需要等待。stock vLLM 使用它自己的混合布局（README 4.6 节），池满时可能抢占。
6. **admission 与调度。** 本 runtime 按到达顺序一次准入一个请求（`goal_query` 在 `--admission-overlap 16` 下最多 16 个同时进行），每次 admission 的阶段 1 是单独的一次 forward，然后成批 decode（README 4.4、4.10 节和第 5 节）。stock vLLM 在每步 16,384 token 的预算内把 prefill 分块与 decode 步一起调度。
7. **关闭 session。** 本 runtime 在 agent 结束时释放该 session 的状态；stock vLLM 把缓存的 prefix 保留到它自行淘汰为止。
8. **重建。** 在本 runtime 上，不是上一个请求延续的请求（例如丢掉被拒回复的格式错误重试）会开一个从文本重新渲染的新 session（README 6.1 节）；stock vLLM 借助 prefix caching 直接处理。
9. **选择性策略。** `observation_query`、`goal_query` 和 `random_fill` 按设计只关注选中的一部分历史 Page；`full` 保留所有 Page。
10. **Goal rewriter。** `goal_query` 在同一张 GPU 上运行一个辅助的 Qwen3-0.6B。它的时间计入吞吐，它的 token 不计入。
11. **两次运行之间的差异。** 两种服务的成批运行在两次运行之间都不是 token 完全相同的：回复可能随 batch 的组成而变，而 batch 的组成取决于时序（README 4.4 节）。在两次 10 个 instance 的 pilot 之间，`full` 在 10 个 instance 中有 2 个结果不同（README 4.7 节）。

---

## Part B. 流程

这一部分用本仓库的工具运行 Part A 的配置：模型在 Slurm 作业中的 GH200 节点上，agent 和它们的任务容器在另一台 x86_64 Docker 主机（agent 主机）上，GPU 作业到 agent 主机之间是一条反向 SSH 隧道（README 6.4 节）。除标注外，每一步都同时适用于四个策略和 stock 对照。

### B.1 Agent 主机

- **要求。** Linux x86_64，装有 Docker、`curl`、`ss`、`python3` 和 `uv`，一个 agent 环境和一个评测环境（版本见 A.2），准备缓存时需要能访问外网。主机要同时运行最多 16 个 runner 进程和 16 个任务容器。隧道需要 GPU 节点能通过 SSH 登录这台主机（B.4）。
- **镜像命名。** 每个 instance 一个镜像：`docker.io/swebench/sweb.eval.x86_64.<把 instance id 中的 "__" 换成 "_1776_" 并转成小写>:latest`，例如 `docker.io/swebench/sweb.eval.x86_64.django_1776_django-10097:latest`。

环境、数据集缓存和目录（这些命令不在已验证的运行范围内；README 7.2 节），本仓库的 checkout 放在 `$AGENT/vllm-0.24.0-selective-prefix-policies`：

```sh
AGENT=/abs/path/on/agent
MINI=$AGENT/mini-env
EVAL=$AGENT/eval-env
uv venv --python 3.12 "$MINI"
uv pip install --python "$MINI/bin/python" mini-swe-agent==2.4.2 litellm==1.103.1 datasets==5.0.1
uv venv --python 3.12 "$EVAL"
uv pip install --python "$EVAL/bin/python" swebench==5.0.2 datasets==5.0.1
"$MINI/bin/python" -c "from datasets import load_dataset; load_dataset('princeton-nlp/SWE-Bench_Verified', split='test')"
"$EVAL/bin/python" -c "from datasets import load_dataset; load_dataset('SWE-bench/SWE-bench_Verified', split='test')"
mkdir -p "$AGENT/sock" "$AGENT/runs" "$AGENT/eval" "$AGENT/mswea-config"
chmod 700 "$AGENT/sock"
```

把 A.1 的列表存为 `$AGENT/ids.txt`，并在任何运行之前拉取 32 个镜像，使拉取时间不进入测量：

```sh
while read -r i; do
  img="docker.io/swebench/sweb.eval.x86_64.${i//__/_1776_}:latest"; img=${img,,}
  if docker pull "$img" > /dev/null; then echo "ok $img"; else echo "FAILED $img"; fi
done < "$AGENT/ids.txt"
```

### B.2 任务列表与 agent 文件

从 runner 使用的数据集生成 `tasks-pool.json`（32 个 statement 原样、按顺序），并把它复制到 GPU 一侧：

```sh
REPO_A=$AGENT/vllm-0.24.0-selective-prefix-policies
mapfile -t IDS < "$AGENT/ids.txt"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "$MINI/bin/python" "$REPO_A/examples/swebench/build_tasks.py" \
  --output-dir "$AGENT/tasks" --pool "${IDS[@]}"
```

`run_agents.sh` 根据它旁边的 `model_override.yaml` 渲染 model override。为两种服务各复制一份 `examples/swebench`，并把 A.6 的新增行加在模板的 `max_tokens` 行之后（`render_override.py` 保留新增的行，只替换端口、`max_tokens` 和 `step_limit` 三行）：

```sh
cp -r "$REPO_A/examples/swebench" "$AGENT/swebench-runtime"
cp -r "$REPO_A/examples/swebench" "$AGENT/swebench-stock"
python3 - "$AGENT/swebench-runtime/model_override.yaml" "$AGENT/swebench-stock/model_override.yaml" <<'PY'
import sys
anchor = '    max_tokens: 1536\n'
client = '    timeout: 3600\n    max_retries: 0\n    num_retries: 0\n'
stock = ('    extra_body: {"chat_template_kwargs": {"enable_thinking": false}, "include_stop_str_in_output": true}\n'
         '    stop: ["</mswea_bash_command>"]\n')
for path, extra in zip(sys.argv[1:], (client, client + stock)):
    with open(path, newline='') as stream:
        text = stream.read()
    assert text.count(anchor) == 1, path
    with open(path, 'w', newline='') as stream:
        stream.write(text.replace(anchor, anchor + extra))
PY
python3 -I -B "$AGENT/swebench-runtime/render_override.py" "$AGENT/swebench-runtime/model_override.yaml" 20000 4096 120
python3 -I -B "$AGENT/swebench-stock/render_override.py" "$AGENT/swebench-stock/model_override.yaml" 20000 4096 120
```

渲染出的两个文件必须与 A.6 的两个文件相同。每份副本都保留了带固定 PoC 文件的 `upstream/`，`run_agents.sh` 在每次运行前都会检查它们。

### B.3 占用 GPU 之前的环境检查

**Gold patch。** 评测 32 个 instance 的参考 patch，可以在不用模型的情况下检查镜像、评测程序和 agent 主机；预期每个 instance 都能解决，有任何一个没有解决，都要在提交 GPU 作业之前查明原因。评测程序会在工作目录下写 `logs/`，所以在一个新目录里运行。未经测试；参数请用 swebench 5.0.2 的 `--help` 确认：

```sh
mkdir "$AGENT/eval/gold-check" && cd "$AGENT/eval/gold-check"
"$EVAL/bin/python" -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Verified \
  --split test --predictions_path gold --instance_ids $(cat "$AGENT/ids.txt") \
  --max_workers 2 --timeout 1800 --run_id gold-check
```

**stock 对照的 thinking 关闭生成头。** 在 GPU 一侧、设置 A.3 的环境变量后，模型的 chat template 在 `enable_thinking=false` 下必须以 A.3 的生成头结束 prompt（未经测试）：

```sh
"$VENV/bin/python" -I -B -c "
import sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'x'}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
print(repr(prompt[-48:]))" "$MODELS/qwen3.6-27b"
```

打印的文本必须以 `<|im_start|>assistant\n<think>\n\n</think>\n\n` 结尾。

### B.4 隧道

GPU 作业在 READY 之后打开一条反向 SSH 隧道，远端是 agent 主机上的一个 Unix socket；`run_agents.sh` 通过 `examples/swebench/relay.py` 把它暴露在 `127.0.0.1:<port>`，如果该端口上除 `127.0.0.1` 之外还有监听，它就拒绝运行（README 6.4 节）。stock 对照使用同样的形式。需要：
- 一把 agent 主机为隧道用户接受、GPU 作业可读的 key，以及一个固定 agent 主机 host key 的 known-hosts 文件（作业从不使用 SSH agent）；
- 计算节点能通过 SSH 连到 agent 主机（第一次运行前从计算节点检查一次）；
- 每个服务一个新的 socket 路径，不超过 100 字节，位于 agent 主机上一个 `0700` 目录中（`$AGENT/sock`）。

隧道断开会结束这次运行（A.11）；作业不会重连。

### B.5 Slurm 作业脚本

下面的脚本在一个 Slurm 作业里运行正式测试的一个服务：服务、READY、kernel 日志行、隧道、通过 SSH 在 agent 主机上运行的 agent 池，以及关闭。它由 `examples/slurm/serve_gh200.sh` 改写而来，后者不传 `--cuda-graphs`、`--active-token-budget`、`--connector-pool-gib`、解码快速路径、`--language-model-only` 和 A.3 的 admission 参数（README 4.10、4.12 节）。这只是示例，**未经测试**。存为 `swebench_job.sh`：

```bash
#!/usr/bin/env bash
# One server of the SWE-bench Verified benchmark (docs/SWEBENCH_BENCHMARK.md) in one Slurm job.
# POLICY: full | observation_query | goal_query | random_fill (this runtime) | stock (stock vLLM 0.24.0).
set -euo pipefail
umask 077
test -n "${SLURM_JOB_ID:-}" || { echo "run this under sbatch" >&2; exit 2; }
: "${REPO:?}" "${VENV:?}" "${TASKS:?}" "${POLICY:?}" "${RUN_ROOT:?}" "${AGENT_SECONDS:?}"
: "${AGENT_HOST:?}" "${TUNNEL_KEY:?}" "${KNOWN_HOSTS:?}" "${AGENT_SOCKET:?}" "${AGENT_PORT:?}"
: "${AGENT_RUN:?}" "${AGENT_OUT:?}" "${AGENT_ENV:?}"
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

# The server: Part A.3 (this runtime) or Part A.4 (stock).
SHARED=(--max-running-sessions 16 --kv-cache-gib 32 --page-tokens 16 --kv-layout decoupled
        --max-model-len 131072 --max-tokens 4096 --dram-arena-gib 70 --connector-pool-gib 1 --cuda-graphs full
        --decode-fastpath --decode-fastpath2 --language-model-only)
case $POLICY in
  full)                          POLICY_FLAGS=() ;;
  observation_query|random_fill) POLICY_FLAGS=(--active-token-budget 8192) ;;
  goal_query)                    POLICY_FLAGS=(--active-token-budget 8192 --goal-rewriter-model "${REWRITER:?}"
                                               --goal-rewriter-batch 4 --admission-overlap 16) ;;
  stock)                         POLICY_FLAGS=() ;;
  *) echo "unknown POLICY: $POLICY" >&2; exit 2 ;;
esac
OUTPUT="$STAGE/$POLICY"
if [[ $POLICY == stock ]]; then
  : "${MODEL:?}"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
  "$PY" -I -B -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name selective-prefix --host 127.0.0.1 --port "$PORT" \
    --distributed-executor-backend uni --tensor-parallel-size 1 \
    --max-num-seqs 16 --max-model-len 131072 --max-num-batched-tokens 16384 \
    --kv-cache-memory-bytes 34359738368 --seed 42 \
    --enable-prefix-caching --prefix-caching-hash-algo sha256 --language-model-only \
    --mamba-cache-mode align --gdn-prefill-backend triton > "$STAGE/server.log" 2>&1 &
else
  : "${RELEASE:?}"
  SERVE=("$PY" -B "$REPO/serve.py" --release-dir "$RELEASE" --task "$TASKS" --policy "$POLICY"
         --output "$OUTPUT" --port "$PORT" "${SHARED[@]}" ${POLICY_FLAGS[@]+"${POLICY_FLAGS[@]}"})
  "${SERVE[@]}" > "$STAGE/dry-run.json"          # dry run: checks the release, prints the vLLM command
  "${SERVE[@]}" --start > "$STAGE/server.log" 2>&1 &
fi
SERVER=$!                                         # the APIServer PID (serve.py execs it)
TUNNEL=""
stop_tunnel() { if [[ -n $TUNNEL ]]; then kill -TERM "$TUNNEL" 2>/dev/null || true; wait "$TUNNEL" 2>/dev/null || true; TUNNEL=""; fi; }
trap stop_tunnel EXIT

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
grep -oE "Using FLASH_ATTN attention backend|Using FlashAttention version [0-9]+|Using Triton/FLA GDN prefill kernel \(requested=[a-z]+|FlashInfer top-p/top-k sampling disabled|Asynchronous scheduling is [a-z]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+|'cudagraph_capture_sizes': \[[0-9, ]*\]" \
  "$STAGE/server.log" | sort | uniq -c > "$STAGE/kernels.txt" || true

# Reverse tunnel: a Unix socket on the agent host -> this node's 127.0.0.1:$PORT (README section 6.4).
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

# The agent pool (Part A.6 to A.8): 16 running of the 32 instances, refilled in list order.
agents_rc=0
if (( up == 1 )); then
  INSTANCES=$("$PY" -B -c 'import json,sys; print(" ".join(t["instance_id"] for t in json.load(open(sys.argv[1]))))' "$TASKS")
  timeout "$AGENT_SECONDS" ssh -T "${SSH_OPTS[@]}" "$AGENT_HOST" env $AGENT_ENV MAX_RUNNING=16 STEP_LIMIT=120 \
    MAX_TOKENS=4096 "$AGENT_RUN" "$AGENT_SOCKET" "$AGENT_PORT" "$AGENT_OUT" $INSTANCES > "$STAGE/agents.log" 2>&1 \
    || agents_rc=$?
  if (( agents_rc == 124 )); then echo "agent time limit reached: the run is partial (B.7)" >&2; fi
else
  echo "tunnel not up (see $STAGE/tunnel.log)" >&2; agents_rc=1
fi
stop_tunnel

# Stop: SIGTERM to the server PID only; this runtime must exit 0 with clean receipts (README section 8).
kill -TERM "$SERVER"
set +e; wait "$SERVER"; server_rc=$?; set -e
echo "server exit $server_rc, agents exit $agents_rc"
if [[ $POLICY != stock ]]; then
  (( server_rc == 0 )) || exit 1
  "$PY" -B "$REPO/examples/slurm/server_checks.py" cleanup "$OUTPUT"
fi
exit "$agents_rc"
```

每个服务提交一个作业，使用 README 快速上手中的占位符，设置放在提交作业的 shell 里（`--export=ALL`）。下面的例子提交 `observation_query`：

```sh
LABEL=a1                                        # a new label for every attempt
POLICY=observation_query
export REPO VENV RELEASE POLICY
export MODEL="$MODELS/qwen3.6-27b" REWRITER="$MODELS/Qwen3-0.6B-c1899de"
export TASKS=/abs/path/to/tasks/tasks-pool.json RUN_ROOT=/abs/short/jobs AGENT_SECONDS=4800
export AGENT_HOST=user@agent-host TUNNEL_KEY=/abs/path/to/tunnel_key KNOWN_HOSTS=/abs/path/to/agent_known_hosts
export AGENT_SOCKET="/abs/path/on/agent/sock/$LABEL-$POLICY.sock" AGENT_PORT=20000
export AGENT_RUN=/abs/path/on/agent/swebench-runtime/run_agents.sh AGENT_OUT="/abs/path/on/agent/runs/$POLICY-$LABEL"
export AGENT_ENV="MINI_PYTHON=/abs/path/on/agent/mini-env/bin/python MSWEA_CONFIG_DIR=/abs/path/on/agent/mswea-config MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1 REQUEST_TIMEOUT=3600"
sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus=1 \
  --cpus-per-task=16 --mem=113560M --time=02:00:00 --job-name="swebench-$POLICY" \
  --output="$RUN_ROOT/slurm-%j.out" --export=ALL --mail-type=BEGIN,END,FAIL --mail-user="$MAIL" \
  /abs/path/to/swebench_job.sh
```

- **stock 对照：** `POLICY=stock`，`AGENT_RUN=/abs/path/on/agent/swebench-stock/run_agents.sh`（带 stock model override 的那份副本，B.2）。
- **时间上限。** `--time` 要覆盖服务启动（作业中第一个服务 204–231 秒 READY，README 4.1 节；冷缓存下 torch.compile 再多约一分钟，README 4.8 节）、agent 运行（`AGENT_SECONDS`）和关闭。被 `AGENT_SECONDS` 或作业时间上限截断的运行是不完整的运行（A.11），所以两者都要留出余量。`goal_query` 比其他服务运行得更久，因为一个 Goal rewriter 进程服务所有 session，一次处理一个调用，加 `--goal-rewriter-batch 4` 时最多把 4 个排队的调用一起处理（README 4.10 节）；给它更长的作业，例如 `--time=04:30:00` 配 `AGENT_SECONDS=13500`。
- `RUN_ROOT` 必须短：`TMPDIR` 在它下面。每个作业的 `AGENT_SOCKET` 和 `AGENT_OUT` 都必须是新的。

### B.6 就绪检查

- **本 runtime。** READY 指 `server_checks.py ready` 返回 0（README 4.1 节）；`/v1/models` 有响应并不等于 READY。`dry-run.json` 中是 vLLM 命令，必须与 A.3 一致。**不要向本 runtime 的服务发送测试 chat 请求：** 任务消息不匹配任何已注册任务的请求会被拒绝，并结束该服务的这次运行；匹配的请求则会打开该任务的 session。
- **stock 对照。** `GET /health` 返回 200，且 `GET /v1/models` 列出 `selective-prefix`。
- **kernel 日志行。** 作业目录中的 `kernels.txt` 必须显示 A.5 的各行。
- **在 agent 主机上**，任何 agent 启动之前（作业脚本执行第一项检查，`run_agents.sh` 执行第二项）：

```sh
curl -sf --unix-socket "$AGENT/sock/$LABEL-$POLICY.sock" http://localhost/v1/models   # lists selective-prefix
ss -ltn 'sport = :20000'                                                             # only 127.0.0.1 listens
```

### B.7 运行 agent 池

- 五个服务依次运行，每个一个作业；agent 主机上从不同时进行两次运行（A.11）。每次运行的 agent 结束后再评测（B.8），且不与其他运行重叠。
- `run_agents.sh` 检查 Mini-SWE-Agent 版本和 PoC 文件，启动 relay，等待端点，检查监听，渲染 model override，然后运行 agent 池。在 agent 主机上查看进度：每完成一个 instance，`$AGENT_OUT/timeline.jsonl` 多一行；`docker ps` 中最多 16 个任务容器。
- **stock 对照：** 每次关闭 session 都返回 HTTP 404，所以 `run_agents.sh` 以退出码 1 结束。以 `timeline.jsonl` 中的 `runner_exit` 判断这次运行。
- **被截断的运行。** SSH 断开或作业的 `timeout` 不会停止 agent 主机上的 runner（README 7.4 节）。先停止该池的 `run_agents.sh`、`swebench_runner.py` 和 `relay.py` 进程，再只删除其日志中列出的容器（未经测试）。这次运行是不完整的运行；整个服务换一个新标签重跑（A.11）。

```sh
grep -ohE 'minisweagent-[0-9a-f]{8}' "$AGENT/runs/$POLICY-$LABEL"/logs/*.log | sort -u | xargs -r docker rm -f
```

### B.8 评测

在 agent 主机上，该次运行的 agent 全部结束之后（README 7.5 节）：

```sh
cd "$AGENT/vllm-0.24.0-selective-prefix-policies"
EVAL_PYTHON="$EVAL/bin/python" WORKERS=2 examples/swebench/evaluate.sh "swebench-$POLICY-$LABEL" "$POLICY" \
  "$AGENT/eval/$POLICY-$LABEL" "$AGENT/runs/$POLICY-$LABEL"
```

`evaluate.sh` 为每个计划内的 instance 写一条 prediction（缺失的 prediction 用空 patch），在新的评测目录中运行 A.9 的命令，为每个 instance 输出一行，最后一行给出 `resolved`。准确率 = `resolved` / 32（A.10）。trajectory、predictions 和日志都包含 prompt 和工具输出，不要放进公开发布。

### B.9 吞吐

从一个 agent 池的输出目录计算 A.10 的吞吐（未经测试）：

```sh
python3 - "$AGENT/runs/$POLICY-$LABEL" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = [json.loads(line) for line in (out / 'timeline.jsonl').read_text().splitlines() if line.strip()]
span = max(r['end'] for r in rows) - min(r['start'] for r in rows)
paths = sorted(out.glob('runs/*/*/*.traj.json'))
tokens = calls = 0
for path in paths:
    for message in json.loads(path.read_text())['messages']:
        extra = message.get('extra') or {}
        if message.get('role') == 'assistant' or (
                message.get('role') == 'user' and extra.get('interrupt_type') == 'FormatError'):
            calls += 1
            usage = (extra.get('response') or {}).get('usage') or {}
            tokens += usage.get('completion_tokens') or 0
print(json.dumps(dict(instances=len(rows), trajectories=len(paths), calls=calls, completion_tokens=tokens,
                      span_s=round(span, 3), tokens_per_s=round(tokens / span, 3))))
PY
```

完整的运行有 32 行 timeline 和 32 个 trajectory；少于此数的运行是不完整的运行（A.11）。
