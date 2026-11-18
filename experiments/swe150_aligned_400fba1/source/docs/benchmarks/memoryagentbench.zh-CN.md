# MemoryAgentBench（kv-session 协议）

[English](memoryagentbench.md) | 简体中文

> 在 integrate-1107 从发布仓库同步（release-repo `5b47394` 的 benchmark 页面）：`examples/memoryagentbench/` 中的客户端和本页都来自发布仓库，只在发布仓库中有效的链接改指本仓库的 README 和 benchmark 文档。正式记录的配置、各次运行及其作业 id 见 [MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](../MEMORYAGENTBENCH_BENCHMARK.zh-CN.md)。

## 范围

- 32 个 case（MemAgent32），MemoryAgentBench 四种能力（Accurate Retrieval、Test-Time Learning、Long-Range Understanding、Conflict Resolution）各 8 个，最多同时运行 16 个（Gemma profile 为 8 个），下面的命令关闭 thinking（正式记录的配置；当前的运行开启 thinking、预算 8,192 token：[docs/status.zh-CN.md](../status.zh-CN.md#当前配置)）。
- 策略 `observation_query` 和 `goal_query`（以及 `random_fill`，两个 query 策略的对照：它的服务见“服务命令”），另有一个 stock vLLM 0.24.0 对照：未经修改的 `vllm.entrypoints.openai.api_server`，由客户端的 vanilla 模式驱动（每轮重新渲染整个对话，以 token id 发送到 `/v1/completions`），而不是 kv-session 协议；模型为 Qwen3.6-27B 和 Gemma 4 profile。
- 协议：`--protocol kv-session`（[docs/running.zh-CN.md 6.6 节](../running.zh-CN.md#66-memoryagentbench-kv-session-协议)）：客户端（method `dram_backed_kv`）自己渲染模板，每个 session 只发送新的尾部。
- Harness：[`examples/memoryagentbench/`](../../examples/memoryagentbench/run_cases.py) 中的 read_next harness（[Harness](#harness)）。官方评分程序是公开的 MemoryAgentBench 仓库 `HUST-AI-HYZ/MemoryAgentBench` 在 commit `538026089d1a8a8eff05121d0db89b388f360eba`（MIT）上的评分程序，从你自己的 checkout 中使用。
- 分数：8 个 Long-Range Understanding case（InfBench 摘要，指标 `llm_judge_f1`）需要 MemoryAgentBench 的 GPT 评判，本发布版本不运行它：这些 case 保持 `judge_pending`，综合分数不计入它们，报告为其他三种能力的平均值（[第 5 步](#步骤)）。打分目前推迟：到目前为止的运行只报告判定，不报告分数（[docs/known-limits.zh-CN.md](../known-limits.zh-CN.md) 5.1）。

**GPU 运行：** 使用这些循环、提示词、请求构造和 case 的运行，连同每个作业的 runtime 提交和判定，见 [docs/status.zh-CN.md](../status.zh-CN.md)。发布仓库的 `5b47394` 是 runtime `5466fc2` 的导出，即 `cf2b28d` 加上 serve.py 的速度默认值（每个都有关闭开关）和测试修复。

## Harness

read_next harness 由 Shanbin Sun 设计：agent 循环及其提示词，以及 MemAgent32 的 case 选择和切块。`examples/memoryagentbench/` 按本项目的运行方式收录了它：

- `run_cases.py`：read_next 循环。历史分块到达，每次以命令 `read_next` 调用一次 `bash` 得到一块，作为以 `[History chunk k/N]` 开头的 tool 消息；问题在同一个请求中跟在最后一块之后；最后一块之前不带工具调用的回复算格式错误，连续两轮无效会结束该 case，最后一块之后第一条不带工具调用的回复就是答案，最多 120 轮。它为每个 case 写一个 `cases/<id>/result.json`，另写 `requests.jsonl`、`run.json` 和 `summary.json`。
- `kv_client.py`：客户端。支持 Qwen3.6 和 Gemma 4 轮次格式的 kv-session 客户端，以及用于 stock vLLM 对照的 stock 客户端（`--mode vanilla`）。
- `cases.py`：case。MemAgent32（`MEMAGENT32`：每个 case 在其源数据中的 split 文件、行号和问题序号，以及它的块数和上下文 token 数）、切块规则、从源数据行构建 case 的程序、加载器（每个 case 在使用前都按 SHA-256 检查）和合成 case。
- `score.py`：使用你的 checkout 中 MemoryAgentBench 官方的 `post_process`。`upstream/SHA256SUMS` 固定所用的两个文件（`utils/eval_other_utils.py`、`utils/templates.py`）；它们没有复制到本仓库中。

切块规则：用 Qwen3.6-27B tokenizer（runtime 在 `source-overlay/remote_ra_linux/model-evidence/` 下固定的副本，不添加特殊 token）编码一个 case 的上下文，切成连续的 4,096 token 窗口，每个窗口再解码回文本；各块依次拼接就是原上下文。两个模型使用相同的块。system prompt 是 MemoryAgentBench 的 `SYSTEM_MESSAGE` 加上 read_next 指令，问题是按 case 的来源填好的 MemoryAgentBench 长上下文提问模板。bash 工具和 tool 消息格式（`<returncode>` / `<output>`）沿用 mini-swe-agent（MIT）。

数据：MemAgent32 的 case id 指的是公开 Hugging Face 数据集 [`cxkjntmngm/MemAgent_Bench`](https://huggingface.co/datasets/cxkjntmngm/MemAgent_Bench) 在 revision `a785de0f8d2511a43b9e524b5326e1105a4d218a` 上的行（split 文件、行号、问题序号），不是 `ai-hyz/MemoryAgentBench` 的行；来源是 `ruler_qa2_validation_65536`（Accurate Retrieval）、`icl_trec_coarse_6600shot_balance`（Test-Time Learning）、`infbench_sum_eng_shots2`（Long-Range Understanding）和 `factconsolidation_sh_32k`（Conflict Resolution，它的 8 个问题共用一段上下文）。本仓库不分发这两个数据集的任何数据，而且 `cxkjntmngm/MemAgent_Bench` 没有声明许可证：请自己下载（第 1 步）。`cases.py build` 读取你下载的 split 文件（Parquet 或 JSON Lines；各行带 `context`、`questions`、`answers` 和 `metadata.source`），或你自己选择的文件（`--select`：一个由 `{id, split, source, file, row, question_index}` 组成的 JSON 列表）。`cases.py verify` 把一个 case 目录与 MemAgent32 固定的块数和上下文 token 数比较。

## 服务

各服务的命令见下面的“服务命令”（带 `--kv-session-thinking off`：该协议默认开启 thinking）。Qwen 服务（16 个 session，124 GiB 的 arena）最多可能锁定 127.3 GiB 主机内存：它们的作业需要两份 GH200 内存份额，即 `--mem=227120M`，与已验证的运行申请的相同；Gemma 服务（8 个 session）一份就够，`--mem=113560M`（[主机内存](../MEMORYAGENTBENCH_BENCHMARK.zh-CN.md#a4-本-runtime服务参数与-vllm-参数)，A.4）。因此，大约 110 GiB 内存的主机（一份 GH200 内存份额，`--mem=113560M`）跑不了文档中的 Qwen profile，本发布版本也没有为 MemoryAgentBench 提供更小的 Qwen profile（没有经过验证的）。主要设置：session 池（`--max-running-sessions`）、`--page-tokens 16 --kv-layout decoupled`、绝对的注意力预算（`--active-token-budget`）、`--max-tokens 4096`、模型的原生上下文（`--max-model-len 262144`）、锁页的 Page arena（`--dram-arena-gib`）和 `--release-idle-pages`（有压力时空闲 session 的 Page 离开 GPU，在下一轮重新载入）。Qwen 服务关闭了 `--decode-fastpath2` 和 `--deferred-completion-io`：它们的 GPU 运行没有使用这些选项（`--decode-fastpath2` 当时需要显式开启，`--deferred-completion-io` 是后来才有的），这些命令保留了那些运行的参数。

跑过的 stock 对照（Qwen），在[运行环境](../install.zh-CN.md#3-运行环境)中，另加 `serve.py` 为本 runtime 设置的两个变量和 `VLLM_NO_USAGE_STATS=1`；`STOCK_CMD` 是 [MEMORYAGENTBENCH_BENCHMARK.zh-CN.md 的 A.5](../MEMORYAGENTBENCH_BENCHMARK.zh-CN.md#a5-stock-vllm-对照vanilla) 中的命令；[B.3](../MEMORYAGENTBENCH_BENCHMARK.zh-CN.md#b3-slurm-作业脚本) 中作业脚本的 stock 分支就是它的生命周期：在后台启动，`/health` 应答且 `/v1/models` 列出 `selective-prefix` 时就绪，向它的 PID 发送 `SIGTERM` 停止（`serve.py` 的 dry run、`--start`、READY 文件和清理回执的检查只适用于 selective release；stock 对照用 HTTP 就绪检查和进程关闭）：

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
STOCK_CMD=("$VENV/bin/python" -I -B -m vllm.entrypoints.openai.api_server
           --model "$MODELS/qwen3.6-27b" --served-model-name selective-prefix --host 127.0.0.1 --port 8000
           --distributed-executor-backend uni --tensor-parallel-size 1
           --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 16384
           --kv-cache-memory-bytes 34359738368 --seed 42
           --enable-prefix-caching --prefix-caching-hash-algo sha256
           --mamba-cache-mode align --gdn-prefill-backend triton --language-model-only)
```

本页不包含 Gemma 4 stock 对照服务的配方，也不涵盖这条路径。

## 步骤

**1. 准备 case**，只需一次，在任何装有 CPython 3.12 并能联网的 Linux 机器（x86_64 或 aarch64）上，在辅助环境中进行：一个专用的虚拟环境（`MAB_VENV`），绝不是基础环境，基础环境保持不变（[docs/install.zh-CN.md 2.1 节](../install.zh-CN.md#21-基础环境)）。`examples/memoryagentbench/requirements.lock` 以每个文件的 SHA-256 固定辅助环境在 Linux x86_64 和 aarch64 上 CPython 3.12 的包：`transformers` 5.13.0 和 `tokenizers` 0.22.2（发布仓库的版本）、`numpy` 2.3.5 和 `tiktoken` 0.14.0（基础环境的版本）、`pyarrow` 26.0.0（用于 Parquet split 文件）、`nltk` 3.10.4、`rouge-score` 0.1.2（PyPI 上没有 wheel：pip 从带哈希的 sdist 构建它；pip 为这次构建获取的构建依赖没有锁定）和 `editdistance` 0.8.1（评分程序导入的包），以及它们的依赖。发布仓库和基础环境之外的版本都是固定的 PyPI 发布版本；GPU 运行的评分环境没有记录。合成 case（用于不需要数据集的冒烟运行：答案是历史中的一个暗号）不需要别的东西。对 MemAgent32，`ROWS` 是你下载的固定 revision 的源数据集，`MAB` 是固定 commit 的 MemoryAgentBench checkout。下载需要联网：`unset` 那一行取消[运行环境](../install.zh-CN.md#3-运行环境)的离线开关，最后一行再设回去：

```sh
MAB_VENV=/abs/path/to/mab-tools       # new: the "venv" line creates it
ROWS=/abs/path/to/MemAgent_Bench      # new: the "hf download" line creates it
MAB=/abs/path/to/MemoryAgentBench     # new: the "git clone" line creates it
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
python3.12 -m venv "$MAB_VENV"
"$MAB_VENV/bin/python" -I -B -m pip --isolated install --require-hashes -r examples/memoryagentbench/requirements.lock
"$MAB_VENV/bin/hf" download cxkjntmngm/MemAgent_Bench --repo-type dataset \
  --revision a785de0f8d2511a43b9e524b5326e1105a4d218a --local-dir "$ROWS"
git clone https://github.com/HUST-AI-HYZ/MemoryAgentBench "$MAB"
git -C "$MAB" checkout 538026089d1a8a8eff05121d0db89b388f360eba
"$MAB_VENV/bin/python" -I -B examples/memoryagentbench/cases.py synthetic --out "$RUNS/mab-synthetic"
"$MAB_VENV/bin/python" -I -B examples/memoryagentbench/cases.py build --rows "$ROWS" --mab "$MAB" \
  --out "$RUNS/mab-cases"
"$MAB_VENV/bin/python" -I -B examples/memoryagentbench/cases.py verify --cases "$RUNS/mab-cases"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

`build` 最后输出 `verify` 的汇总：每个 case 的块数和上下文 token 数都与 MemAgent32 一致时，`counts_match` 为 32。

**2. 服务：** 启动它（见下面的“服务命令”），等待 READY。stock 对照则改为：在后台启动它的 `STOCK_CMD`（[服务](#服务)），然后等到 `/health` 应答且 `/v1/models` 列出 `selective-prefix`（A.5 的就绪判断；[B.3](../MEMORYAGENTBENCH_BENCHMARK.zh-CN.md#b3-slurm-作业脚本) 的 stock 分支）。

**3. 对服务运行 case**，在同一个节点上、在运行环境中（它的 `python` 带 Transformers 5.13）：Qwen 同时运行 16 个（`--workers 16`），Gemma profile 8 个。客户端从 `GET /v1/kv-session/info` 读取服务的预算、thinking 设置和轮次格式，与自己的设置（`--thinking`，默认 `off`；`--grammar`）不同时拒绝启动。客户端的输出目录 `CLIENT_RUN` 对每个模型、策略和尝试都是新的（`POLICY` 是服务的 `--policy`，`LABEL` 是这次尝试的标签，与服务的 `--output` 中的相同）；第 5 步对同一个目录评分。用服务所用模型的那一行设置 `CLIENT_RUN`，然后运行该模型的客户端。Qwen：

```sh
POLICY=observation_query      # the server's --policy: observation_query, goal_query or random_fill
CLIENT_RUN="$RUNS/mab-$POLICY-$LABEL-client"
python -I -B examples/memoryagentbench/run_cases.py --base-url http://127.0.0.1:8000 \
  --cases "$RUNS/mab-cases" --tokenizer "$MODELS/qwen3.6-27b" --workers 16 --out "$CLIENT_RUN"
```

Gemma：

```sh
POLICY=observation_query      # the server's --policy: observation_query, goal_query or random_fill
CLIENT_RUN="$RUNS/mab-gemma-$POLICY-$LABEL-client"
python -I -B examples/memoryagentbench/run_cases.py --base-url http://127.0.0.1:8000 \
  --cases "$RUNS/mab-cases" --tokenizer "$MODELS/gemma-4-31b-it" --grammar gemma4 --workers 8 --out "$CLIENT_RUN"
```

stock 对照（`--mode vanilla`，Qwen）：

```sh
CLIENT_RUN="$RUNS/mab-stock-$LABEL-client"
python -I -B examples/memoryagentbench/run_cases.py --base-url http://127.0.0.1:8000 --mode vanilla \
  --cases "$RUNS/mab-cases" --tokenizer "$MODELS/qwen3.6-27b" --workers 16 --out "$CLIENT_RUN"
```

它输出一行，给出结果（`COMPLETED`）和每种 case 状态的数量：`answered`、`answer_length_limit`、`unfinished_thinking`、`format_error`、`turn_limit` 或 `error`（循环自己的状态，例如 `context_limit`，是该 case 的 `runner_status`）。默认最大的 case 先跑（`--order largest-first`）；`--ids` 只跑一部分。

**4. 停止**：用 `SIGTERM` 停止服务，并检查清理回执（[docs/running.zh-CN.md 4.1 节](../running.zh-CN.md#41-单个服务)）。stock 对照不写回执：向它的 PID 发送 `SIGTERM`，并看它的退出状态（[MEMORYAGENTBENCH_BENCHMARK.zh-CN.md 的 B.3](../MEMORYAGENTBENCH_BENCHMARK.zh-CN.md#b3-slurm-作业脚本) 的 stock 分支）。

**5. 评分**，离线进行，在辅助环境中用官方的 `post_process` 对 `CLIENT_RUN` 中的运行（第 3 步）评分。checkout 导入的包是 numpy、nltk、tiktoken、rouge_score 和 editdistance（都在辅助环境中）；`--stub-unused` 替换后三个，这些来源的评分用不到它们：

```sh
"$MAB_VENV/bin/python" -I -B examples/memoryagentbench/score.py --run "$CLIENT_RUN" \
  --cases "$RUNS/mab-cases" --mab "$MAB" --stub-unused
```

它把每个 case 的指标和分数写进该 case 的 `result.json`，把每种能力的平均值写进 `accuracy.json`。8 个 Long-Range Understanding case（InfBench 摘要，指标 `llm_judge_f1`）需要 MemoryAgentBench 的 GPT 评判（一个外部 API 服务），本发布版本不运行它：`score.py` 把它们留作 `judge_pending`，没有分数。`accuracy.json` 报告不含它们的综合分数：`composite_score_without_judge` 是其他三种能力（Accurate Retrieval、Test-Time Learning、Conflict Resolution）的平均值，前提是每个计划中的 case 都已运行并完成、没有基础设施错误（状态 `error`；`infrastructure_errors` 列出它们），且这三种能力都已完整评分；`judge_abilities` 给出被排除的能力。`composite_score` 是全部四种能力的平均值，在评判 case 待出期间保持 null。

## 服务命令

本 benchmark 各服务的 dry run，每个策略一条（加 `--start` 即启动服务）。`LABEL` 是每次尝试新用的标签；发布仓库的 `tests/test_docs_commands.py` 会 dry run 这些命令。

### Qwen3.6-27B

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
python -B serve.py --release-dir "$RELEASE" --policy random_fill "${SHARED[@]}" \
  --active-token-budget 32768 --output "$RUNS/mab-random_fill-$LABEL" --port 8000
```

### Gemma 4 31B IT

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
python -B serve.py --release-dir "$RELEASE-gemma4" --policy random_fill "${SHARED[@]}" \
  --active-token-budget 32768 --output "$RUNS/mab-gemma-random_fill-$LABEL" --port 8000
```
