# vLLM 0.24.0 的 Selective-prefix 策略

[English](README.md) | 简体中文

> **本版本的状态。** 已在 Qwen3.6-27B 上经过 GPU 验证：SWE-bench Verified 上的 Mini-SWE-Agent XML 协议（[docs/SWEBENCH_BENCHMARK.zh-CN.md](docs/SWEBENCH_BENCHMARK.zh-CN.md)）；`--protocol chat-text` 上的 Terminal-Bench 2.0，`full` 和 `observation_query`（16 个任务的 dry run；[docs/TERMINALBENCH_BENCHMARK.zh-CN.md](docs/TERMINALBENCH_BENCHMARK.zh-CN.md)）；`--protocol kv-session` 上的 MemoryAgentBench，`observation_query`、`goal_query` 和 stock 对照（32 个 case 的运行；[docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md)）。chat-text 上的 `goal_query` 不在本版本中（正在分支 `tb-goal-query` 上审阅）。Gemma 4 31B IT profile（4.14 节）仍在开发中：它的两个池的 KV 布局在关闭选择性 runtime 的情况下已在一张 GH200 上运行过（作业 3397695：32 token 的 Page 与 stock vLLM 回放的 10 个 trial 中 10 个 token 级一致；16 token 的 Page 从第 3 轮起出现 TRITON_ATTN 窗口布局的数值或分块差异），但在本版本中它的 runtime 服务在 GPU 上启动时就会停止（作业 3398198 和 3398614："engine renderer and dynamic provider required"）；修复在一个开发分支上，尚未经过 GPU 验证。`--decode-fastpath3` 和 torch profile 开关（`REMOTE_RA_TORCH_PROFILE`，4.11 节）还没有经过 GPU 验证。

这是面向多轮工具调用 agent 的服务端 prefix Page 选择机制，基于 vLLM 0.24.0。每个服务运行下面五种策略之一，由策略决定下一轮从 CPU 内存恢复哪些已封存的历史 Page：

| `--policy` | 选择方式 |
| --- | --- |
| `full` | 恢复全部可用历史 Page（参照组）。 |
| `random50` | 随机选一半（⌊n/2⌋）可用 Page，抽取由固定种子 42 和前缀最后一个完整 KV block 的哈希共同决定，因此结果随前缀而变。 |
| `goal_query` | Online Selector V2（第 5 节）：由冻结任务和最新 action/observation 维护检索 Goal，转成模型 query 向量；预算由必保留页、recent 页和余弦最高的页填满。 |
| `observation_query` | 同一个 selector；query 是模型在最新 observation 末尾的 hidden state。 |
| `random_fill` | 两个 query 策略的对照（第 5 节）：轮次流程、必保留页、recent 页和容量都与 `observation_query` 相同，剩余容量按带种子的随机顺序填满（SplitMix64 加 Fisher-Yates，由种子、session 序号和该 session 的请求序号决定；`--fill-seed`，默认 42）。不用 Goal rewriter，也不用 query 向量。 |

Harness 通过普通的 OpenAI Chat Completions 端点通信，每轮发送完整的、不断增长的消息历史。policy、session、Goal、向量和 Page 选择都不会出现在 HTTP 里。

本仓库包含核心 runtime、安装器和启动器，另有示例：一个冒烟客户端、一个 Slurm 作业脚本，以及我们的 SWE-bench pilot 围绕 Mini-SWE-Agent 使用的衔接脚本（第 7 节）。仓库里还带有 pilot 使用的 PoC SWE-bench runner 和 prompt（`examples/swebench/upstream/`）。仓库里没有工作负载、结果文件或 trajectory。内部的 `remote_ra_*` 模块名属于实现细节，不是公开接口。

## 当前 runtime

- **本版本（当前的 `main`）：** 五个策略，Mini-SWE-Agent XML 协议（第 6 节），每个服务最多 16 个 session，以及最多注册 64 个任务、最多 K 个同时运行的 session 池（`--max-running-sessions K`，4.4 节），CUDA graph（`--cuda-graphs`，4.8 节），可选的上下文上限（默认 65,536 token，可选 131,072 或模型原生的 262,144）、回复上限和锁页 Page arena 上限（`--max-model-len`、`--max-tokens`、`--dram-arena-gib`，4.9 节），以及三项需要显式开启的 admission 优化（`--gdn-device-handoff`、`--phase1-in-step`、`--admission-overlap K`，4.10 节），需要显式开启的解码快速路径及其可选的第二、第三阶段（`--decode-fastpath`、`--decode-fastpath2`、`--decode-fastpath3`，4.11 节），需要显式开启的纯文本模式（`--language-model-only`，4.12 节），MemoryAgentBench 的 kv-session 协议（`--protocol kv-session`，6.6 节），面向 Terminal-Bench 2.0 agent 的 chat-text 协议（`--protocol chat-text`，6.7 节），需要显式开启的、在 GPU 紧张时释放空闲 session 的 Page（`--release-idle-pages`，4.13 节），需要显式开启的批量 Goal rewriter 调用（`--goal-rewriter-batch N`，4.10 节），以及第一阶段的 Gemma 4 31B IT 模型 profile（五个策略、`--admission-overlap K`、CUDA graph、前两条解码快速路径、MemoryAgentBench 的 kv-session 协议和 chat-text 协议；其服务已完整实现并在 CPU 上测试；在 GPU 上只运行过它的两个池的布局，且关闭了选择性 runtime，作业 3397695，4.14 节）。它是 commit `780fe54`（4.6 节的解耦 KV 布局和 `random_fill` 对照）加上这些选项和更低的 admission 开销；各步见[历史](#历史)。
- **默认配置不变：** 784 token 的 Page 放在一个共享 KV 池里，eager 执行，65,536 token 的上下文，1,536 token 的回复，admission 一次一个；上面的每个选项都需要显式开启。**对 `goal_query`、`observation_query` 和 `random_fill`，推荐加 `--page-tokens 16 --kv-layout decoupled`**（4.6 节）。**admission 优化：** 16 个 session 的 `goal_query` 只加 `--admission-overlap 16`，其他策略都不加（4.10 节）。**解码快速路径：** 每个策略都加 `--decode-fastpath`（4.11 节）。
- **GPU 上的运行**（一张 GH200；每次运行使用的 runtime commit 见对应小节）：解耦布局通过了三次冒烟测试（作业 3308015、3308414、3308482），`780fe54` 以 16 token 的 Page 用 Mini-SWE-Agent 2.4.2 跑了 10 个 instance 的 SWE-bench pilot，每个策略一个作业（作业 3313802、3314010、3314319、3314623；4.7 节）。16 个 session 在作业 3339019 和 3343069 中运行过，session 池在作业 3343069 中运行过（4.4 节），CUDA graph 在作业 3338893、3339448 和 3343069 中运行过（4.8 节），131072 token 的上下文、4096 token 的回复和 arena 上限在作业 3348705 中运行过，262144 token 的上下文在作业 3360375 中运行过（4.9 节），admission 优化在作业 3349078 和 3349504 中运行过（4.10 节），解码快速路径在作业 3366128 中运行过（4.11 节）。chat-text 协议以 runtime `68d19b4` 配合 `full` 和 `observation_query` 各通过了一次 16 个任务的 Terminal-Bench 2.0 dry run，开了 `--connector-pool-gib 1`、`--language-model-only` 和 `--release-idle-pages`（6.7 节）。kv-session 协议在 MemoryAgentBench 作业 3397844 和 3397842 中运行过（Qwen，runtime `0d37346`，6.6 节），被释放 Page 的重新载入在作业 3397842 中运行过（16 个 session，4.13 节）。[docs/SWEBENCH_BENCHMARK.zh-CN.md](docs/SWEBENCH_BENCHMARK.zh-CN.md) 中 runtime `5b15b2f` 上的 SWE-bench 运行（作业 3404696 和 3404230，`goal_query`）在 GPU 上运行了 `--decode-fastpath2`，作业 3404230 还运行了 `--goal-rewriter-batch 4`（4.10 节）；`--decode-fastpath3` 只在 CPU 上跑过。本版本还没有单独做干净安装验收；最近一次干净安装验收于 2026-09-30 通过（作业 3270721，runtime `a564df3`）。
- **以下内容尚未证明：** SWE-bench 任务成功、任务准确率、加速或传输节省，以及支持边界（第 1 节）之外的任何配置。每个 instance 只跑一次，不构成 benchmark 证据。
- **逐请求编码：** 服务端只用加载的 Hugging Face tokenizer 对每条消息编码，与 stock vLLM 相同；固定 tokenizer 的语义只在启动时检查一次。Transformers 5.13 与固定的 `tokenizer.json` 对部分文本的切分不同（天城文元音符号，出现在一个 SWE-bench Django 测试文件里），之前逐请求比较两者的做法会以 HTTP 500 拒绝这类请求（作业 3354504）。
- **SWE-bench Verified 正式测试：** [docs/SWEBENCH_BENCHMARK.zh-CN.md](docs/SWEBENCH_BENCHMARK.zh-CN.md) 规定四个策略和 stock vLLM 0.24.0 对照的 32 个 instance、16 个并发的运行配置，并给出 baseline 的对齐检查表；它还记录运行状态：runtime `5b15b2f` 上的两次 `goal_query` 运行（作业 3404696 和 3404230，32 个中分别解决 19 个和 20 个），其余四个服务尚待运行。
- **Terminal-Bench 2.0 正式测试：** [docs/TERMINALBENCH_BENCHMARK.zh-CN.md](docs/TERMINALBENCH_BENCHMARK.zh-CN.md) 规定 `full`、`observation_query`、`random_fill` 和 stock vLLM 0.24.0 对照用 Harbor 0.21.0 的 Terminus 2 跑 89 个任务、16 个 trial 并发的运行配置，并给出 baseline 的对齐检查表；它不报告任何结果。
- **MemoryAgentBench 正式测试：** [docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md) 规定 `observation_query`、`goal_query` 和 stock vLLM 0.24.0 对照在关闭 thinking 的 kv-session 协议上跑 32 个 case（每项能力 8 个，最多 16 个同时运行）的运行配置，给出它的验证状态和 baseline 的对齐检查表；它不报告任何结果。
- 更早的各个 runtime、对应运行和作业号见[历史](#历史)。

## 快速上手

### 需要准备什么

本仓库加一个干净的 vLLM 0.24.0 环境，就足以完成安装和启动服务：

1. **本仓库**（Git checkout）。
2. **一个 Python 3.12 环境，装有未经修改的官方 `vllm==0.24.0` wheel，以及恰好 `transformers==5.13.0` 和 `tokenizers==0.22.2`。** 其他版本会被 `install.py` 拒绝。直接 `pip install vllm==0.24.0` 可能会解析出别的 Transformers 版本，所以两者都要固定版本（2.1 节给出经过验证的做法）。不支持从源码或 editable 方式安装的 vLLM：安装器的 wheel RECORD 检查会失败。
3. **Qwen3.6-27B 权重**，revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`；只有使用 Goal rewriter 的 `goal_query` 才需要 **Qwen3-0.6B**（也可以用 `--no-goal-rewriter` 启动 `goal_query`）（2.2 节）。
4. **Linux，带 `gcc` 或 `clang`，以及该解释器的 Python 头文件**（Triton 在运行时编译 kernel），外加一个**较短的 `TMPDIR`**（第 3 节）。只在一张 NVIDIA GH200（Linux aarch64）上验证过。安装器不检查架构和 GPU，但其他平台都未测试。

安装器从不修改 vLLM 环境。它另建一个 release 目录（已安装 `vllm/` 包的副本加上本 overlay），所以原环境仍可以当普通 vLLM 使用。启动服务不需要联网：`serve.py` 会设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。

运行 SWE-bench harness 还需要 Mini-SWE-Agent 2.4.2 和一台装有 SWE-bench 镜像的 Docker 主机，这超出了安装和启动服务的范围（第 7 节）。

### 从零到一个冒烟请求

下面的命令使用这些占位变量，后面各节沿用它们。请使用绝对路径。

```sh
REPO=/abs/path/to/vllm-0.24.0-selective-prefix-policies
VENV=/abs/path/to/vllm-env
MODELS=/abs/path/to/models
RELEASE=/abs/path/to/releases/selective-prefix
RUNS=/abs/path/to/runs
cd "$REPO"
```

1. 建好基础环境 `$VENV`（2.1 节），并把模型放到 `$MODELS`（2.2 节）。
2. 运行测试，检查环境，然后创建 release（2.3 和 2.4 节）：

```sh
"$VENV/bin/python" -B -m unittest discover -s . -p 'test_*.py'
mkdir -p "$(dirname "$RELEASE")" "$RUNS"
"$VENV/bin/python" -B install.py --site-packages "$VENV/lib/python3.12/site-packages" \
  --model-dir "$MODELS/qwen3.6-27b" --release-dir "$RELEASE"
"$VENV/bin/python" -B install.py --site-packages "$VENV/lib/python3.12/site-packages" \
  --model-dir "$MODELS/qwen3.6-27b" --release-dir "$RELEASE" --apply
```

3. 在 GPU 节点上设置运行环境（第 3 节），先 dry run，再启动，等待 READY，发送一个请求，然后用 `SIGTERM` 停止（4.1 节和第 8 节）：

```sh
export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
export TMPDIR=/abs/short/tmp
export VLLM_USE_FLASHINFER_SAMPLER=0
unset CC VLLM_RPC_BASE_PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
python -B serve.py --release-dir "$RELEASE" --task examples/task.json --policy full \
  --output "$RUNS/full" --port 8000
python -B serve.py --release-dir "$RELEASE" --task examples/task.json --policy full \
  --output "$RUNS/full" --port 8000 --start > "$RUNS/full.server.log" 2>&1 &
SERVER=$!
until python -B examples/slurm/server_checks.py ready "$RUNS/full" full "$RELEASE"; do
  kill -0 "$SERVER" || { echo "server exited before READY"; break; }
  sleep 5
done
python -B examples/http_client.py --base-url http://127.0.0.1:8000/v1 --messages examples/messages.json
kill -TERM "$SERVER"
wait "$SERVER"
python -B examples/slurm/server_checks.py cleanup "$RUNS/full"
```

`wait` 必须返回 0，最后一条命令必须输出 `CLEAN_SHUTDOWN`。在 Slurm 作业里，`examples/slurm/serve_gh200.sh` 会替你完成第 3 步（4.5 节）。

对 `goal_query`、`observation_query` 和 `random_fill`，在两条 `serve.py` 命令里都加上 `--page-tokens 16 --kv-layout decoupled`（4.6 节）。这时 dry run 显示的 vLLM 命令末尾是 `--block-size 16 --mamba-block-size 784 --additional-config ...`：

```sh
python -B serve.py --release-dir "$RELEASE" --task examples/task.json --policy observation_query \
  --page-tokens 16 --kv-layout decoupled --output "$RUNS/obs" --port 8000
```

## 1. 支持边界

- Linux、Python 3.12、从 wheel 安装的干净上游 **vLLM 0.24.0**、Transformers **5.13.0**、Tokenizers **0.22.2**。
- 模型：本地 **Qwen3.6-27B** 快照，revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`。元数据哈希见 `INSTALL_RELEASE.json`，本仓库不分发权重。三个固定哈希的元数据文件必须是普通文件，不能是 Hugging Face cache 符号链接。
- 实测硬件：Linux aarch64 上一张 NVIDIA GH200 120GB。其他 GPU、架构和 backend 未测试。
- 同步、纯文本、UniProc executor，TP=PP=DP=1，**每个启动任务一个 session，每个服务最多 16 个**（见 4.4 节；作业 3339019 和 3343069 在 GPU 上跑过 16 个 session）；使用 `--max-running-sessions K` 时最多注册 64 个任务，其中最多 K 个同时运行（4.4 节；作业 3343069 在 GPU 上跑过 24 个任务、K = 8）。已验收的运行每个服务只有一个 session。
- 启动器固定已验收的服务配置：

  | 配置 | 值 |
  | --- | --- |
  | `max_model_len` / `max_num_batched_tokens` | 65536 / 16384；`--max-model-len 131072` 选择 131072 token 的上下文（已在作业 3348705 中经 GPU 验证，4.9 节）；`--max-model-len 262144` 是模型的原生上限，选择 262144 token 的上下文（已在作业 3360375 中经 GPU 验证，4.9 节） |
  | 回复上限（`--max-tokens`） | 每轮最多生成 1536 个 token，即 PoC 的值；`--max-tokens 4096`（已在作业 3348705 中经 GPU 验证，4.9 节） |
  | Page 大小和 KV 布局 | 784 token，一个共享池（默认）；`--page-tokens {784,16,32,64}` 配合 `--kv-layout decoupled`，注意力和 Gated-DeltaNet 各用一个池（4.6 节）。在 GPU 上验证过的：784（默认）和 16 解耦；32 和 64 从未在 GPU 上运行过 |
  | GPU KV / CPU offload | 8 GiB（多个 session 时用 `--kv-cache-gib` 16、24 或 32）/ 32 GiB |
  | 执行方式 | eager（不使用 CUDA graph / torch.compile）；`--cuda-graphs piecewise`、`full` 或 `full-decode` 已在 GPU 上验证（作业 3338893、3339448、3343069），但输出可能与 eager 不同（第 4.8 节） |
  | admission | 按到达顺序一次一个；`--gdn-device-handoff`、`--phase1-in-step` 和 `--admission-overlap K` 需要显式开启，已在作业 3349504 中用 `goal_query` 经 GPU 验证（4.10 节） |
  | Mamba cache | `align` |
  | GDN prefill | Triton |
  | 解析器 | `qwen3_xml` tool-call、`qwen3` reasoning |
  | 服务模型名 | `selective-prefix` |

  **prompt 加生成内容必须在 65536 token 以内**（即上下文上限；prompt 最多为上下文上限减去回复上限，默认 64,000 token）。 已验收的运行（作业 3270721）使用的就是这套配置，prompt 最长约 33K token。8 GiB GPU KV 用来放常驻工作集和一轮的新 token：64k 时 `full` 约保留 84 个 51.4 MB 的注意力 Page（4.3 GB），另有几个 154 MB 的 Gated-DeltaNet 状态 block。16 token 的 Page 下，同样 64K token 是 4,096 个 1 MiB 的注意力 Page（4 GiB）。

## 2. 安装与卸载

安装、installed-check 和启动服务必须使用同一个目标 Python。以下命令在本仓库目录执行，沿用快速上手里的占位变量。

### 2.1 基础环境

经过验证的环境就是干净安装验收作业 3270721 建立的那个环境；[历史](#历史)中标明“在作业 3270721 的干净环境上（复用）”的每次运行，以及两次 SWE-bench pilot，都以只读方式复用了它。它在 Linux aarch64 上这样建立：

```sh
BASE_PYTHON=/abs/path/to/cpython-3.12/bin/python3.12
WHEEL=vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl
curl -fLO "https://files.pythonhosted.org/packages/9e/80/51a071305b4eed0f6f512dc1c1c6957cbb14ccce38db1be90ffcff2a2844/$WHEEL"
echo "700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5  $WHEEL" | sha256sum --check
"$BASE_PYTHON" -I -B -m venv --copies "$VENV"
"$VENV/bin/python" -I -B -m pip --isolated install --index-url https://pypi.org/simple \
  --only-binary=:all: --no-cache-dir --no-compile --retries 0 --timeout 120 \
  --constraint examples/env/constraints-3270721.txt --report "$RUNS/pip-report.json" \
  "$WHEEL" transformers==5.13.0 tokenizers==0.22.2
```

- **基础解释器。** 作业 3270721 用 uv 管理的 CPython 3.12.13（`cpython-3.12.13-linux-aarch64-gnu`）作为 `BASE_PYTHON`。它是怎样安装的（例如 `uv python install 3.12.13`）没有记录，不在已验证的运行范围内。其他带 `venv` 和头文件的 CPython 3.12 应该也可以，但未测试。
- **Wheel。** PyPI 上的官方文件，271,361,241 字节，SHA-256 如上。作业只从这个 URL 下载过一次；`curl` 只是下载方式之一。
- **约束文件。** `examples/env/constraints-3270721.txt` 与该作业使用的约束文件逐字节相同（194 个版本固定，SHA-256 `1bc47da57ac424cf332c1e3df9f1c52137a77dd5bae3b95b1471c919607504a0`）。安装能否复现就靠它；不用它的话，pip 会解析出当天的依赖版本，那些版本未经验证。
- **作业之后检查了什么：** 195 个发行包；Torch `2.11.0+cu130`（CUDA 13.0 构建）；`$VENV/bin/ninja` 1.13.2；`$VENV/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13` 存在；已安装的 `vllm/` 树与 wheel 的 RECORD 一致（4,404 个文件）。用同样的方法检查 Torch：

```sh
LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib" "$VENV/bin/python" -I -B -c \
  "import torch; print(torch.__version__, torch.version.cuda)"
"$VENV/bin/ninja" --version
```

- **`pip check`** 会报告 `nvidia-cusparselt-cu13 0.8.0 is not supported on this platform`。这是 NVIDIA aarch64 wheel 自身的问题（安装后的 `WHEEL` 元数据带有非标准标签 `py3-none-manylinux2014_sbsa`；库本身是 aarch64 的，且与 RECORD 一致）。已验证的运行只接受这一条消息。
- 之后保持环境不变：不要再 `pip install`，不要 editable 安装。release 会绑定解释器、`sys.prefix` 和 site 路径（2.4 节）。像已验证的作业一样，运行它的 Python 时加 `-B`（或设置 `PYTHONDONTWRITEBYTECODE=1`）。

### 2.2 模型

两个 checkpoint，都以本地目录形式提供：

| 模型 | Revision | 使用者 | 如何固定 |
| --- | --- | --- | --- |
| [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | 每个服务（`install.py` 的 `--model-dir`） | `install.py`（仅检查和 `--apply` 都会）按 `INSTALL_RELEASE.json` 核对 `config.json`、`tokenizer.json`、`tokenizer_config.json` 的 SHA-256；不对权重做哈希 |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | 带 `--goal-rewriter-model` 的 `goal_query` | `serve.py` 在选页合同里固定解析后的目录和其 `config.json` 的 SHA-256；`install.py` 不检查它 |

Qwen3.6-27B 的三个固定哈希的元数据文件必须是普通文件，不能是 Hugging Face cache 符号链接（安装器拒绝符号链接）。已验证的运行没有记录 checkpoint 是怎样下载的，所以下面的命令**不在已验证的运行范围内**。用较新的 `huggingface_hub`（`hf` 命令行），`--local-dir` 会写出普通文件：

```sh
hf download Qwen/Qwen3.6-27B --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --local-dir "$MODELS/qwen3.6-27b"
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir "$MODELS/Qwen3-0.6B-c1899de"
find "$MODELS/qwen3.6-27b" -maxdepth 1 -type l
(cd "$MODELS/qwen3.6-27b" && sha256sum config.json tokenizer.json tokenizer_config.json)
(cd "$MODELS/Qwen3-0.6B-c1899de" && sha256sum config.json model.safetensors tokenizer.json)
```

`find` 必须没有任何输出。期望的 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| Qwen3.6-27B `config.json` | `69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c` |
| Qwen3.6-27B `tokenizer.json` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| Qwen3.6-27B `tokenizer_config.json` | `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b` |
| Qwen3-0.6B `config.json` | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| Qwen3-0.6B `model.safetensors` | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| Qwen3-0.6B `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |

已验证的运行里，rewriter 是该 revision 的一个 Hugging Face cache 快照（对 rewriter 来说符号链接没有问题）；0.6B 的几行来自那些运行的 rewriter 检查。rewriter 解析后的路径属于选页合同的一部分，所以同一组对比里的每个服务都要用同一个路径。

### 2.3 测试

测试需要 `torch`（CPU 版即可；有几个测试模块会 import 它，其余的在没有它时跳过张量相关测试），以及 `PATH` 上的 `git`。2.1 节的基础环境两者都有；作业 3270721 就在那里运行了公开测试。本版本有 1,242 个测试；在一台 Windows 台式机上（CPU 版 Torch 2.9.1、Python 3.12.7），commit `780fe54` 的 512 个测试用时约 9 分钟。其中一个测试会在候选文件的 Git 快照上把整套测试再跑一遍。

```sh
"$VENV/bin/python" -B -m unittest discover -s . -p 'test_*.py'
"$VENV/bin/python" -B check_git_release.py --ref HEAD
```

`check_git_release.py --ref HEAD` 会对该 commit 的 `git archive` 字节运行测试，使用一个 OpenMP 线程（`OMP_NUM_THREADS=1`），超时 1800 秒；发布 commit 前必须通过。

**慢速测试。** 默认测试（也就是 `check_git_release.py` 运行的那一套）对最大的几个 CPU 端到端用例只跑简短版本：每个“布局 × 策略”矩阵只跑一两个（布局，策略）组合，与旧代码的对比只跑 `full` 和 `goal_query`，kv-session 的 MemAgent session 只跑 `goal_query`，16 个 session 只跑 `observation_query`，session 池只跑 10 个任务的用例，48 个随机选页目录只跑其中 16 个，Gemma 的端到端用例只跑 6 或 8 个 session（4.14 节），chat-text（Qwen 与 Gemma）只跑不含摘要和 aside 的 session（6.7 与 4.14 节）。设置 `REMOTE_RA_SLOW_TESTS=1` 会运行全部用例，外加 4.13 节 32,768 token 的压力用例。每次 push 前的验证都要跑这两遍测试和 `check_git_release.py --ref HEAD`：

```sh
REMOTE_RA_SLOW_TESTS=1 OMP_NUM_THREADS=1 "$VENV/bin/python" -B -m unittest discover -s . -p 'test_*.py'
```

### 2.4 安装 release

```sh
mkdir -p "$(dirname "$RELEASE")" "$RUNS"
"$VENV/bin/python" -B install.py --site-packages "$VENV/lib/python3.12/site-packages" \
  --model-dir "$MODELS/qwen3.6-27b" --release-dir "$RELEASE"
# 上面仅检查；下面显式创建。目标目录必须不存在。
"$VENV/bin/python" -B install.py --site-packages "$VENV/lib/python3.12/site-packages" \
  --model-dir "$MODELS/qwen3.6-27b" --release-dir "$RELEASE" --apply
```

仅检查时输出 `CHECK_ONLY_PASS`，`--apply` 输出 `INSTALL_PASS`（两者都带 `"site_packages_written": false`）。安装器会做这些事：
- 用 `vllm-0.24.0.dist-info/RECORD` 核对完整的已安装 `vllm/` 树。缺失、多余、无哈希或被改动的文件都会在复制前报错。RECORD 只证明 wheel 完整，不证明发布者身份，所以基础 wheel 要从可信来源安装。不支持没有 RECORD 的 editable/source checkout。
- 核对 Transformers/Tokenizers/vLLM 的精确版本，以及三个固定的模型元数据哈希（2.2 节）。
- 用 120 条路径的上游映射绑定 overlay（27 个改动、91 个未改动锚点、2 个 absent target）。所有分发文件都会核对哈希。
- 创建独立的 `<release>/source` 树并固定依赖身份。**它不会修改 site-packages、不安装依赖、不启动模型，也不启用功能。** 请预留完整 vLLM 副本的磁盘空间，复制期间保持源环境不变。
- Manifest 记录 Python 精确版本、实现、架构、解释器、`sys.prefix` 与 site 路径；环境变了就需要新的 release。
- 失败时只删除本次新建的不完整 release。已有 release 或源码漂移会被拒绝。请保留错误输出，下次换新的 release 路径；不要对缺少最终 index 的 release 执行 `restore.py`。

### 2.5 卸载

先停止所有使用该 release 的进程。

```sh
"$VENV/bin/python" -B restore.py --release-dir "$RELEASE"
"$VENV/bin/python" -B restore.py --release-dir "$RELEASE" --apply --processes-stopped
```

卸载会检查 release，并把它改名到相邻的 `.uninstalled-<token>` 恢复目录，不删除任何东西。回滚到原始 vLLM 只需停止本服务，直接使用未改动的原环境。不要移动运行中的 release，因为它的 manifest 绑定了绝对路径。

## 3. 运行环境

已通过的干净安装验收使用下面的设置，之后在它的环境上运行的作业也一样。请在启动 `serve.py` 的 shell 中设置；启动器会把环境传给服务（只去掉 `REMOTE_RA_*`，并设置它自己的 runtime 开关）。

```sh
# 目标环境的 bin：其 Python 以及 JIT 构建使用的 Ninja。
export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
# 目标环境的 CUDA 13 runtime（libcudart.so.13），供隔离启动的进程使用。
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
# vLLM 在 $TMPDIR/<36 字节 uuid> 绑定 ZMQ IPC socket：路径要短（AF_UNIX 上限 107 字节）。
export TMPDIR=/abs/short/tmp
# 使用 native top-k/top-p sampler；FlashInfer sampler 需要 JIT 工具链和已预热的缓存。
export VLLM_USE_FLASHINFER_SAMPLER=0
# Triton 优先用 $CC 编译 launcher，否则用 gcc/clang；有些集群会导出 CC=cc。
unset CC VLLM_RPC_BASE_PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
```

说明：
- `len(TMPDIR) + 37` 不能超过 107 字节。路径更长时，第一个服务会在 READY 之前退出，报错 `zmq.error.ZMQError: ipc path ... is longer than 107 characters`。
- 缓存为空时，vLLM 0.24 默认的 FlashInfer sampler 会在首次使用时 JIT 编译，需要 `nvcc` 和 Ninja，可能要几分钟。native sampler 可以避开这一步；`temperature=0` 时输出和 sampler 无关。冷缓存下的 FlashInfer sampler 不在已验收配置范围内。
- Triton 会在最初几个请求时 JIT 编译 kernel，需要可用的 `gcc`（或 `clang`）以及目标解释器的 Python 头文件。
- 每次运行可以使用独立的缓存目录（`HOME`、`XDG_CACHE_HOME`、`TRITON_CACHE_DIR` 等）。已验证的作业用的是全新的、作业本地的空缓存（见 `examples/slurm/serve_gh200.sh`）。
- 内存：模型（51.1 GiB）加 8 GiB KV 和可选的 0.6B Goal rewriter（约 2 GiB）可以放进一张 GH200（启动时空闲 94.5 GiB）；24 GiB KV 加五个 session 和 rewriter 也放得下，784 token 和 16 token 的 Page 都是如此（作业 3300434 和 3314319）。除模型加载外，主机还需要约 32 GiB 内存作为 CPU KV 池；解耦布局另有一个锁页（pinned）的 Page arena，16 token 的 Page 下每个 session 最多约 4 GiB（4.6 节；131072 token 的上下文下约 8 GiB，262144 下约 16 GiB，4.9 节）。已验证的作业申请了 200 GiB。

## 4. 启动各个策略

### 4.1 单个服务

每个服务固定**一个策略及其冻结任务**。任务文件只包含 `instance_id` 和 `problem_statement`；每个请求的任务消息（第一条 user 消息）必须原样包含这段 problem statement。输出目录必须尚不存在（其父目录要存在）。先看 dry run；只有 `--start` 才会加载模型。

```sh
python -B serve.py --release-dir "$RELEASE" \
  --task examples/task.json --policy full --output "$RUNS/full" --port 8000
python -B serve.py --release-dir "$RELEASE" \
  --task examples/task.json --policy full --output "$RUNS/full" --port 8000 --start
```

`goal_query` 必须明确选择 Goal rewriter：要么给一个本地 Qwen3-0.6B checkpoint 目录（PoC 的设置），要么用 `--no-goal-rewriter` 只用确定性的 Goal 字段。`observation_query` 和 `random_fill` 不需要选择；它们接受 `--goal-rewriter-model`，但从不使用（见第 5 节）。其余 selector 选项可选，见第 5 节。`serve.py --help` 列出所有参数。对 query 策略和 `random_fill`，我们推荐 16 token 的 Page（4.6 节）：

```sh
python -B serve.py --release-dir "$RELEASE" \
  --task examples/task.json --policy goal_query --output "$RUNS/goal" --port 8000 \
  --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --page-tokens 16 --kv-layout decoupled --start
```

`--start` 会把启动器进程替换成 APIServer，所以启动器的 PID 就是服务 PID。服务只监听 `127.0.0.1:<port>`（从其他主机访问见 6.4 节）。启动需要几分钟：加载模型、分配 KV，以及首次 Triton 编译。已验证的作业里，每个作业的第一个服务在启动后 204–231 秒 READY，之后的服务为 105–146 秒。

**READY 的判定：** `<output>/ready.json` 存在且能完整解析，并且满足：
- `native_http_ready == true`；
- `evidence_kind == "NATIVE_HTTP_RUNTIME"`；
- `policy` 与请求一致；
- `runtime_manifest == SHA256(<release>/runtime-manifest.json 原始字节)`；
- task/config 哈希与 `startup.json` 一致。

这个文件是先创建、再写入内容的，解析失败时应重试。`/v1/models` 能访问不等于 READY。`examples/slurm/server_checks.py ready <output> <policy> <release>` 正是检查这些条件（退出码 0 表示 READY，1 表示尚未就绪，2 表示不匹配）。

**停止：** 只对该 PID 发送 `SIGTERM`，不要发给整个进程组。等进程以 0 退出后，再核对清理回执（第 8 节；`examples/slurm/server_checks.py cleanup <output>`）。被强杀或超时的关停不算干净关停。

### 4.2 一张 GPU 上依次跑四个策略

一张 GH200 同一时间只能放一个服务。每个策略（以及每个任务）都要用全新的服务和全新的输出目录。下面的流程与已验收的运行一致，把 `HARNESS` 换成你的客户端：

```sh
TASK=/abs/path/to/tasks/example-array-fix.json
OUT="$RUNS/example-array-fix"
HARNESS=/abs/path/to/your-harness
REWRITER="$MODELS/Qwen3-0.6B-c1899de"
PORT=8000
mkdir -p "$OUT"
for POLICY in full random50 goal_query observation_query; do
  SELECTOR=""
  case "$POLICY" in goal_query) SELECTOR="--goal-rewriter-model $REWRITER";; esac
  python -B serve.py --release-dir "$RELEASE" --task "$TASK" --policy "$POLICY" $SELECTOR \
    --output "$OUT/$POLICY" --port "$PORT" --start > "$OUT/$POLICY.server.log" 2>&1 &
  SERVER=$!
  until python -B examples/slurm/server_checks.py ready "$OUT/$POLICY" "$POLICY" "$RELEASE"; do
    kill -0 "$SERVER" || { echo "server exited before READY: $POLICY"; exit 1; }
    sleep 5
  done
  "$HARNESS" --base-url "http://127.0.0.1:$PORT/v1" --task "$TASK" --record "$OUT/$POLICY.harness"
  kill -TERM "$SERVER"
  wait "$SERVER" || { echo "server exit not zero: $POLICY"; exit 1; }
  python -B examples/slurm/server_checks.py cleanup "$OUT/$POLICY" || exit 1
done
```

下一个服务必须在 GPU 空闲、没有残留进程时启动。已验收的运行里，依次跑四个服务、每个六轮对话（prompt 最长约 33K token），共用时约 18 分钟。

已验收的运行里每个策略都用 784 token 的 Page。按 4.6 节的推荐，给 `goal_query` 和 `observation_query` 的服务加上 `--page-tokens 16 --kv-layout decoupled`（如果在循环里加入 `random_fill` 对照，也给它加上）。在 exact 冒烟测试里，`full` 在两种布局下产生的 token 完全相同（4.7 节）；不过 16 token Page 的 SWE-bench pilot 仍然给每个服务（包括 `full`）都加了这两个参数。

### 4.3 多张 GPU

每张 GPU 并行跑一个策略（每个服务分别设置 `CUDA_VISIBLE_DEVICES=<n>`、不同的 `--port` 和不同的 `--output`），符合“一卡一服务”的设计，但**不在**已验证的运行范围内。永远不要让两个跑同一个任务的 harness 连同一个服务。

### 4.4 一个服务上的多个 session

```sh
python -B serve.py --release-dir "$RELEASE" \
  --task /abs/path/to/tasks-g0.json --policy observation_query --kv-cache-gib 24 \
  --page-tokens 16 --kv-layout decoupled --output "$RUNS/obs-g0" --port 8000 --start
```

`--task` 接受 1 到 16 个 `{instance_id, problem_statement}` 对象组成的 JSON 列表，每个对象一个 session（`max_num_seqs` = session 数；在 GPU 上跑过的配置见下表，最多 16 个 session）。instance id 必须互不相同，任何一个 problem statement 都不能包含另一个。每个 harness 跑其中一个任务，连同一个端点；服务把每个请求路由到唯一一个 problem statement 出现在该请求任务消息（`messages[1]`）中的 session，每个 session 的行为和单任务服务一样（见第 5、6 节）。匹配不到任何任务或匹配到多个任务的请求会被拒绝。admission 按到达顺序一次处理一个请求（使用 `--admission-overlap K` 时除外，4.10 节），之后 decode 在各 session 之间成批进行。各 session 共用一个 GPU KV 池，不会发生抢占：最坏情况峰值放不下的一轮会等待（`CAPACITY_WAIT`），直到正在运行的请求释放 block。

`--kv-cache-gib` 的大小要保证在预期的最长上下文下，所有 session 的常驻 Page 加上一轮仍能放下。经过验证的设置如下（“可用”= 注意力 block 数 − 1）。784 token 的 Page 下，一页 51,380,224 字节，注意力 Page 和 Gated-DeltaNet 状态共用一个池；16 token Page 的解耦布局下，一页 1,048,576 字节，Gated-DeltaNet 状态有自己的池（4.6 节）：

| `--kv-cache-gib` | 每个服务的 session 数 | Page | 运行 | 证据 |
| --- | --- | --- | --- | --- |
| 8 | 1 | 784 | 干净安装验收：4 个策略 × 6 轮，prompt 最长约 33K token | 作业 3270721；vLLM：GPU KV cache size 121,605 token |
| 24 | 4 | 784 | 合成运行：4 个 session × 6 轮、每轮 6,500 token observation，`full`/`observation_query`/`goal_query` | 作业 3299809（以及 3292739、3295203、3300301）；池 501 个 block，可用 500 |
| 24 | 3 | 784 | SWE-bench pilot，3 个 instance × 3 个策略，一组 | 作业 3299922；`kv_usable_blocks` 500，session 上限 5 |
| 24 | 5 | 784 | SWE-bench pilot，10 个 instance × 3 个策略，两组各 5 个 | 作业 3300434；`kv_usable_blocks` 500，session 上限 5；vLLM：GPU KV cache size 364,817 token |
| 8 | 1 | 16，解耦 | GPU 冒烟测试：`full`，10 轮，prompt 最长约 60K token | 作业 3308015；7,702 个注意力 block + 10 个 Mamba block |
| 24 | 4 | 16，解耦 | GPU 冒烟测试：4 个 session × 6 轮、每轮 6,500 token observation，`full`/`goal_query` | 作业 3308482；23,205 个注意力 block + 28 个 Mamba block |
| 24 | 5 | 16，解耦 | SWE-bench pilot，10 个 instance × 4 个策略，两组各 5 个 | 作业 3313802、3314010、3314319、3314623；22,912 个注意力 block + 34 个 Mamba block（11 个 Gated-DeltaNet slot），可用 22,911，session 上限 5；vLLM：GPU KV cache size 366,592 token |
| 24 | 16 | 16，解耦 | 合成运行：16 个 session × 6 轮、每轮 2,600 token observation，`full`/`observation_query`/`random_fill`，eager | 作业 3339019；19,682 个注意力 block + 100 个 Mamba block（33 个 Gated-DeltaNet slot），可用 19,681；vLLM：GPU KV cache size 314,912 token |
| 24 | 16 | 16，解耦 | 合成运行：同样的负载，`observation_query`，eager 以及 `--cuda-graphs full` | 作业 3343069；两个池与作业 3339019 相同 |
| 24 | 24 个任务，最多 8 个同时运行 | 16，解耦 | 合成 session 池：24 个任务 × 6 轮、每轮 3,000 token observation，`--max-running-sessions 8`，`observation_query`，`--cuda-graphs full-decode` | 作业 3343069；22,031 个注意力 block + 52 个 Mamba block（17 个 Gated-DeltaNet slot），可用 22,030；vLLM：GPU KV cache size 352,496 token |

pilot 对 session 数设了上限，保证即使每个 session 都停在 65,536 token 的上限，池也不会用尽：N − 1 个空闲 session 各占 I = 85 个 block，加上一轮的峰值 P = 91 个 block，再加 3 个 block 余量，必须放进可用 block 数 U，所以 N = 1 + ⌊(U − 91 − 3) / 85⌋（这是我们容量分析中 `full` 在 64K 时的占用；因为 session 重建会重新 prefill 每一页，所以对所有策略都按它算）。由此 8 GiB（U = 166）为 1 个 session，24 GiB（U = 500）为 5 个。同一规则给出 16 GiB 为 3 个、32 GiB 为 7 个；这两种大小**不在已验证的运行范围内**。

16 token Page 的 pilot 对注意力池用了同样的规则：停在上限的空闲 session 占 I = ⌈65,536 / 16⌉ + 1 = 4,097 个 block，一轮的峰值是 I + ⌈16,384 / 16⌉ = 5,121，余量是一次 1,536 token 的回复（97 个 block）加上 vLLM 的 1 % watermark（229）。24 GiB 下五个 session 需要 4 × 4,097 + 5,121 + 326 = 21,835 个 block，可用的有 22,911 个；六个放不下。Mamba 池每个 session 两个 slot，另加一个备用（4.6 节）。这条规则同样给出 8 GiB 为 1 个 session、16 GiB 为 3 个、32 GiB 为 7 个（**不在已验证的运行范围内**）。

**最多 16 个 session。** `remote_ra_live_v2/sessions.py` 中的 `MAX_SESSIONS` 是唯一的上限，路径上每个进程级的 session 限制都读取它；此前它是 8。随 session 数 S 增长的有：

- 解耦布局的 Mamba 池，1 + 3(2S + 1) 个 block：S = 16 时为 100 个 block（4.78 GiB），在注意力池之前预留。如果注意力池连一个 session 在 65,536 token 上限处的一轮都放不下（⌈65,536 / P⌉ + 1 + ⌈16,384 / P⌉ + ⌈1,536 / P⌉ + 1 个 block，加上 1 % watermark），launcher 会拒绝这个预算。8 GiB 下，16、32、64 token Page 最多允许 9 个 session，解耦的 784 token 模式最多 8 个。1 到 8 个 session 的每种配置都能通过；共享的 784 token 池没有 Mamba 池，不受影响。
- 一个服务的终身请求上限（引擎 lifecycle 条目、请求 tombstone 和 render binding，都不会被回收）：每个 session 256 个，即单个 session 的请求上限，至少 1,024（16 个时为 4,096；4.9 节）。
- 主机内存：固定（pinned）Page arena（4.6 节）在 P = 16 时最多 16 × 4,097 个 1 MiB 的 Page = 64 GiB，另有 32 GiB 的 CPU 池，以及每个 session 一份 154 MB 的 Gated-DeltaNet carry（开 `--gdn-device-handoff` 时两份，4.10 节）。

在 GPU 上，16 个 session 跑过两次：作业 3339019（`full`、`observation_query` 和 `random_fill`，eager，runtime `afffd2b`）和作业 3343069（`observation_query`，eager 以及 `--cuda-graphs full`，runtime `613a7d5`）。都是合成会话：每个 session 6 轮、每轮 2,600 token 的 observation，prompt 最长约 14K token，24 GiB、16 token 的 Page，使用作业 3270721 的干净环境（复用）和一张 GH200。16 个 session 一起 decode，没有抢占，没有一轮等待容量；离线重放每个选页轮次，都得到与记录相同的 Page。同一负载的两次 eager 运行（两个作业里的 `observation_query`）在 96 轮中有 72 轮 token 相同：成批 decode 时，回复可能随 batch 的组成而变，而 batch 的组成取决于时序。这些上下文远低于 65,536 token 的上限，所以这两次运行没有触及下面说的那些限制。

在一块 GH200 上，16 个 `full` session 在 64K 时无法满足防死锁规则：它们需要约 67,350 个 1 MiB 的注意力 block 加上 Mamba 池，约 70.6 GiB KV，而 GPU 启动时只有 94.53 GiB 空闲（作业 3313802 的 vLLM 日志），权重占 51.1 GiB。选择性策略只在两次重建之间需要更少的 block，因为重建会回到全部驻留。16 token Page 下使用固定的 `--active-token-budget 8192` 时，15 个空闲 session 各自保留预算和上一轮的 Page，加上一轮的峰值：每轮约 7,000 token 时放得进 24 GiB，达到 16,384 token 的 observation 上限时需要 34 GiB；同样两种假设下，40 % 预算分别需要 41 和 51 GiB。请据此确定 `--kv-cache-gib` 和负载。只要正在运行的请求还能释放 block，`CAPACITY_WAIT` 就会让一轮等待。没有其他请求运行时，如果空闲 session 钉住的 Page 没给这一轮留下空间，它同样等待，直到有 session 关闭，或在 `--release-idle-pages`（4.13 节）下释放空闲 session 的 Page；所有 session 都在等待时，服务停滞。在此版本之前，门控因为没有请求在运行而放行这样的一轮：在 GPU 上它的 decode 用掉最后一个空闲 block，vLLM 抢占了它，这是致命的（Slurm 作业 3398129：16 个 Terminus 2 trial，`full`，32 GiB，空闲 session 钉住了 26,698 个注意力 block 中除 69 个以外的全部）。没有其他 session 钉住 block 时，这一轮照旧放行（如同单任务服务）。`--kv-cache-gib` 的最大值仍为 32：在使用 Goal rewriter 时，估计 GPU 上约有 36 GiB 可留给 KV（未测量）。

**Session 池：注册多个任务，最多 K 个同时运行（`--max-running-sessions K`）。** 使用这个选项时，`--task` 注册 1 到 64 个任务（与任何任务列表一样固定在 `startup.json` 中，另加 `max_running_sessions` = K），任一时刻最多打开 K 个 session（1 到 16，且不超过任务数）。这样一个服务就能例如用 16 个并发 agent 跑 32 个 SWE-bench 任务，agent 结束后补入下一个任务。按任务消息路由、session 之间的隔离、轮次调度、容量门控和不抢占都不变。改由 K 决定的有：`max_num_seqs` = K、Mamba 池（2K + 1 个 slot）和 launcher 的注意力池检查、上面的防死锁规则（N = K）以及固定的 Page arena。由注册任务数 T 决定的是终身请求上限 max(1,024, 256 × T)（T = 32 时为 8,192；4.9 节）。

```sh
python -B serve.py --release-dir "$RELEASE" --task /abs/path/to/tasks-pool.json --policy full \
  --max-running-sessions 16 --kv-cache-gib 24 --page-tokens 16 --kv-layout decoupled \
  --output "$RUNS/full-pool" --port 8000 --start
```

- **打开。** 一个 session 在其任务的第一个请求（turn 0）时打开。已有 K 个 session 打开时，这个请求在 admission 之前按到达顺序等待，直到有 session 关闭；等待时间记为它在 `runtime.jsonl` 中那一行的 `open_wait_s`。等待期间离开的请求不会打开它的任务。
- **关闭。** agent 结束后，harness 关闭其任务的 session：`POST /v1/sessions/close`，请求体 `{"instance_id": "<任务 id>"}`（6.5 节）。引擎释放这个 session 持有的一切：GPU 上的驻留 Page 和 live tail，主机上的 Gated-DeltaNet carry 和固定 arena 中的 Page，它的选页表、Goal 状态和 transcript，以及它保留的逐请求记录。两个池恰好收回该 session 固定的 block，下一个等待的任务随即打开。关闭的任务不会重新打开；之后对它的请求立即失败（`session ... closed by the harness`）。
- **失败。** 失败的请求照旧关闭它的 session，但该 session 会保留它的打开名额和引擎状态，直到 harness 关闭它；因此 harness 要关闭它启动的每个 session，不论成败。
- **关停**时会关闭仍然打开的 session，与任何服务一样。

在 GPU 上（作业 3343069，runtime `613a7d5`）：24 个任务、K = 8，`observation_query` 加 `--cuda-graphs full-decode`，每个任务 6 轮、每轮 3,000 token 的 observation，24 GiB、16 token 的 Page。每个任务都打开、跑完 6 轮，并由 harness 关闭、拿到关闭回执（24 次打开、24 次关闭）；同时运行的 session 最多 8 个，没有抢占，没有一轮等待容量，离线选页重放与每个选页轮次一致，关停时两个池都回到 READY 时的基线。Mamba 池有 17 个 slot（2K + 1），由 K 而不是 24 个任务决定（见上表）。

### 4.5 GPU 作业（Slurm）

`examples/slurm/serve_gh200.sh` 在一个 Slurm 作业里运行一个服务：作业本地的 `TMPDIR` 和缓存、第 3 节的环境、dry run、`serve.py --start`、等待 READY、运行 harness、对服务 PID 发送 `SIGTERM`、检查退出码和清理回执。它参照已验证作业的 `job.sh` 和作业步骤。那些作业在命令行上传给 `sbatch` 的选项如下：

| 选项 | 已验证的值 |
| --- | --- |
| `--partition` | 一个 GH200 分区：上限 2 小时的交互分区（作业 3270721、3299809、3300301，以及 16 token Page 的 pilot 3313802、3314010、3314319、3314623），784 token Page 的 pilot 用批处理分区（作业 3299922、3300434） |
| `--gpus` / `--nodes` / `--ntasks` | `1` / `1` / `1`（其他集群上通常对应 `--gres=gpu:1`；不在已验证的运行范围内） |
| `--cpus-per-task` / `--mem` | `16` / `200g` |
| `--time` | `00:30:00`（4 个单 session 服务，作业 3270721），`00:45:00`（3 个四 session 服务，作业 3299809 和 3300301），`02:00:00`（2 个五 session 服务，作业 3313802、3314010、3314319、3314623），`03:45:00`（3 个服务，作业 3299922），`10:45:00`（6 个服务，作业 3300434） |
| `--mail-type` | `BEGIN,END,FAIL`（作业 3270721 用的是 `END,FAIL`；批处理队列可能要等很久） |
| `--export` | `ALL`（脚本随后自己重设 `PATH`、`LD_LIBRARY_PATH`、`HOME`、缓存和 `TMPDIR`） |

脚本从提交作业的 shell 读取设置（脚本开头列出了全部变量）。用快速上手的任务做一次冒烟运行：

```sh
ACCOUNT=your-account
PARTITION=your-gh200-partition
MAIL=you@example.org
export REPO VENV RELEASE
export TASKS="$REPO/examples/task.json" POLICY=full RUN_ROOT=/abs/short/jobs MODE=smoke
sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus=1 \
  --cpus-per-task=16 --mem=200g --time=00:30:00 --job-name=selective-prefix \
  --output="$RUN_ROOT/slurm-%j.out" --export=ALL --mail-type=BEGIN,END,FAIL --mail-user="$MAIL" \
  "$REPO/examples/slurm/serve_gh200.sh"
```

`RUN_ROOT` 必须短：作业把 `$RUN_ROOT/<作业号>/tmp` 用作 `TMPDIR`，超出 AF_UNIX 预算的路径会被拒绝。`MODE=swebench` 会打开 6.4 节的反向隧道，并通过 SSH 在 agent 主机上运行 `examples/swebench/run_agents.sh`（7.4 节）。`PAGE_TOKENS=16 KV_LAYOUT=decoupled` 会把 `--page-tokens 16 --kv-layout decoupled` 传给 `serve.py`（4.6 节）；两者都不设时，服务使用默认的 784 token 布局。已验证的作业由我们的私有控制器驱动，还有额外的检查（环境快照、GPU 空闲检查、逐轮证据检查）；这个脚本只保留服务端的步骤，**它本身不在已验证的运行范围内**。

### 4.6 解耦 KV 布局与 16 token 的 Page

**为什么默认 Page 是 784 token。** Qwen3.6-27B 是混合模型：16 个全注意力层（4 个 KV head × 256，bf16：每层每 token 4,096 字节），以及 48 个 Gated-DeltaNet 层，每层有一份 3,207,168 字节的递归状态加卷积状态（vLLM 把这类层叫作 Mamba 层）。vLLM 0.24 把两类状态放在同一个 block 池里，只用一种页大小。`vllm/platforms/interface.py` 中的 `_align_hybrid_block_size` 会把注意力 block 调大，直到一个注意力页至少能放下一个 Mamba 页：16 × ⌈3,207,168 / (16 × 4,096)⌉ = 784 token，并把 Mamba 页填充到 784 × 4,096 = 3,211,264 字节。runtime 的一个 Page 就是一个注意力 block，所以 Page 是 784 token，PoC 按 16 token 计的预算只能取整到整页（第 5 节）。

**解耦布局改了什么。** 它给两类状态各建一个 block 池，block 大小互相独立（`docs/exp-decouple-pages/DESIGN.md`，决定 D2–D4）：
- **注意力池：** P token 的 block。一个 block 仍然是一个 Page：16 个注意力层合计 P × 65,536 字节，P = 16 时为 1 MiB。
- **Mamba 池：** 不填充的 Gated-DeltaNet 状态。48 层分成 3 个各 16 层的 Mamba group；一个 block 存一个 group 的状态（16 × 3,207,168 = 51,314,688 字节），一个 slot 是每个 group 各一个 block。池里有 S 个 session 对应的 K = 2S + 1 个 slot（每个 session 一个运行 slot 和一个跨块 slot，另加一个备用），再加一个 null block：共 1 + 3K 个 block。它先从 `--kv-cache-gib` 中预留，剩下的给注意力池，即 ⌊(KV 字节数 − (1 + 3K) × 51,314,688) / (P × 65,536)⌋ 个 block。
- **Mamba 对齐 block：** M = P × ⌈784 / P⌉，P = 16 时为 784。因此 Gated-DeltaNet 状态的跨块、捕获和恢复都与 784 token 的 Page 完全相同（每份状态 153,944,064 字节）。

group 结构、kernel 和 Mamba 的 `align` 模式都不变。

**参数。** `--page-tokens 16`、`32` 或 `64` 总是使用解耦布局；`--kv-layout decoupled` 在 784 token 时也会选用它，这是只拆分两个池的验证模式。`--page-tokens 16 --kv-layout decoupled`（GPU 运行用的写法）与只写 `--page-tokens 16` 效果相同，只是把布局写明。此时 `serve.py` 会在 vLLM 命令末尾追加 `--block-size P --mamba-block-size M --additional-config {"remote_ra_kv_layout":{"mamba_slots":K,"mamba_tokens":M,"page_tokens":P}}`，并在 `startup.json` 里记录 `kv_layout` 和 `page_tokens` = P。两个参数都不给时，vLLM 命令、启动配置及其固定哈希与 `902690a` 逐字节相同（对照旧代码的 CPU 测试）。

**启动日志。** 解耦布局的服务会打印下面几行（作业 3313802：24 GiB、5 个 session），不会出现 vLLM 的 `Setting attention block size to 784 tokens`，也不会出现它的 Mamba 填充日志：

```text
Decoupled KV layout: attention block 16 tokens, Mamba block 784 tokens, 11 Mamba slots, unpadded Mamba page.
Decoupled KV cache pools: 22912 attention blocks, 34 Mamba blocks
GPU KV cache size: 366,592 tokens
```

`ready.json` 记录布局和两个池的大小（第 8 节）。已验证运行的池大小见 4.4 节的表。

**主机内存。** 解耦布局下，worker 把拷到主机的注意力 Page 放在一个锁页的 Page arena 里。上下文上限为 L token 时，它的容量是 S × (⌈L / P⌉ + 1) 个 Page，每个 P × 65,536 字节（`--dram-arena-gib` 可设一个更小的总容量，4.9 节），按需以 256 MiB 为单位分配：P = 16、L = 65,536 时每个 session 最多约 4 GiB（`remote_ra_live_v2/worker_composition.py` 中的 `arena_pages`，`remote_ra_growing_worker/tensors.py` 中的 `PageArena`）。这是在第 3 节的 32 GiB CPU 池之外另加的。

**P = 16 时的选页。** Selector V2 合同以页大小为参数，所以 P = 16 时它就是 PoC 自己的合同：容量 `max(1, 必保留页数, floor(0.40 × 历史 token) // 16)` 页，recent 配额 64 页（1,024 token），live tail 不足 16 token，每个 16 token 的 Page 一个向量（第 5 节）。worker 在 GPU 上每次 forward 一批地把 Page 向量压缩成 int8（选页合同中的 `compact_int8_v1`），而不是在 CPU 上逐个计算。

### 4.7 16 token Page 的验证

下面所有运行都使用作业 3270721 的干净环境（复用）和一张 GH200；每项只跑了一次。

**GPU 冒烟测试**（runtime `8f1af6a`；合成的 Mini-SWE-Agent XML 会话，每轮 observation 6,500 token；每个服务和每项计划检查都通过）：

| 作业 | 服务 | 结果 |
| --- | --- | --- |
| 3308015 | `full`，1 个 session × 10 轮，prompt 最长约 60K token，8 GiB，三种布局：默认、784 解耦、16 解耦 | 三种布局下每一轮的 prompt token、回复和生成 token 数都相同；每轮的 prepare 和 restore H2D 字节数也相同 |
| 3308414 | 16 token Page 下的 `full`、`random50`、`goal_query`、`observation_query`，1 个 session × 10 轮，8 GiB | 历史超过 8,192 token 后，query 策略选了约 40 % 的 16 token Page，选择性策略都恢复了 Page |
| 3308482 | 4 个 session × 6 轮，24 GiB：默认布局的 `full`；16 token Page 的 `full` 和 `goal_query` | 4 个 session 成批 decode，无抢占，无容量等待；关停时每个解耦服务的两个池都回到 READY 时的基线 |

`780fe54` 在 `8f1af6a` 之上只加了 `random_fill`；其余四个策略在两者上发送和写出的字节完全相同（`test_random_fill_legacy.py`）。

**16 token Page 的 SWE-bench pilot**（runtime `780fe54`；Mini-SWE-Agent 2.4.2 和第 7 节的设置，与作业 3300434 相同的 10 个 Astropy instance，分成两组、每组 5 个 session；24 GiB；每个服务都加 `--page-tokens 16 --kv-layout decoupled`；每个策略一个作业、只跑一次；agent 每组上限 3,000 秒）。每个服务都通过了服务端检查。轮数即 agent 的请求数；resolved 来自 swebench 5.0.2：

| 策略 | 作业 | 轮数 | Resolved |
| --- | --- | --- | --- |
| `full` | 3313802 | 517 | 7/10 |
| `observation_query` | 3314010 | 538 | 8/10 |
| `goal_query` | 3314319 | 643 | 7/10 |
| `random_fill` | 3314623 | 608 | 8/10 |

同样的 instance 在 784 token 的 Page 下（runtime `b1e6988`，作业 3300434）：`full` 490 轮、9/10 resolved，`goal_query` 716 轮、6/10，`observation_query` 715 轮、7/10。

这是 10 个 instance 上每个策略一次运行。在上面的冒烟测试里，`full` 的 token 与页大小无关，但它在两次 pilot 之间有 2 个 instance 的结果变了，所以这种量级的差异单靠两次运行之间的波动就可能出现。**这不是 benchmark 证据**，也不能据此给策略排序。

已离线按 token 逐一核对的服务端证据归档：作业 3313802、3314010、3314319 和 3314623（四个策略全部）。

**性能说明**（每项只跑了一次）：
- 冒烟测试 3308015，单个请求 decode 时的稳态 token 间延迟：默认布局 67.4 ms，784 解耦 68.3 ms，16 token Page 74.9 ms。每个冒烟作业只预热了一种布局，这个作业预热的是默认布局。冒烟测试 3308482 预热的是 16 token Page，结果反了过来：单个请求 decode 时默认布局 70.8 ms、16 token Page 68.1 ms，四个请求时 75.5 ms 对 73.2 ms。因此作业 3308015 中的差距大概主要来自预热。两个作业都没有单独测出 16 token Page 的每步开销，设计目标“每个请求每步最多多 0.1 ms”（DESIGN R5）尚未得到证明。
- 冒烟测试 3308482：`full` 在 16 token Page 下每分钟完成 14.63 轮，默认布局下 14.22 轮；默认布局是该作业里没有预热的那一种。

### 4.8 CUDA graph

`--cuda-graphs off`（默认）就是上面的 eager 服务：vLLM 命令、`startup.json`、其固定哈希和证据都与不加该参数时逐字节相同（对照 `b37044f` 的 CPU 测试，`test_cudagraph.py`）。

`--cuda-graphs piecewise`、`full` 或 `full-decode` 把 vLLM 命令里的 `--enforce-eager` 换成固定的 `--compilation-config`；异步调度仍然关闭：

| 模式 | `--compilation-config` | vLLM 0.24 捕获什么 |
| --- | --- | --- |
| `piecewise` | `{"cudagraph_mode":"PIECEWISE","mode":"VLLM_COMPILE"}` | FlashAttention、KV cache 写入和 Gated-DeltaNet 核心之间的已编译分段；这三者每步都以 eager 方式运行，读取该步的选择性序列长度、slot mapping、block table 和 Mamba 状态索引 |
| `full` | `{"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}`（stock vLLM 的默认值） | 整个 uniform decode 步，包括注意力；混合步仍是分段。runner 在捕获和重放时都把每个全注意力 group 绑定到其物理序列长度缓冲区 |
| `full-decode` | `{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"NONE"}` | 整个 uniform decode 步，使用 eager kernel，不做 torch.compile；prefill、混合步和 phase 1 与 `off` 一样以 eager 方式运行。绑定方式与 `full` 相同 |

- **固定值与检查。** `startup.json` 记录 `cuda_graphs`，有效配置记录 `enforce_eager: false` 和该编译配置。启动时 frontend 比对 vLLM 解析出的参数，worker 检查解析后的配置（编译模式、CUDA graph 模式、捕获尺寸，以及编译时留在 graph 之外的三个 op），任何差异都拒绝启动；参数为 off 时，worker 拒绝已捕获 graph 的 runner。
- **证据。** `worker-attached.json` 多一条 `cuda_graphs` 记录（模式、解析后的配置、捕获尺寸）；不做任何逐请求或逐步的检查。
- **多个 session。** vLLM 按 `max_num_seqs` 确定捕获的 batch 尺寸：每个启动 task 一个，使用 `--max-running-sessions K` 时为 K（第 6.5 节）。尺寸为 1、2、4 以及不超过 2 × `max_num_seqs` 的 8 的倍数（16 个 session 时为 1、2、4、8、16、24、32），所以最多 16 个运行中 session 的每个 decode 步都有对应的 graph（CPU 测试 `test_int_flags.py`，使用 vLLM 0.24 自身的尺寸计算代码；作业 3343069 在 16 个 session 时记录的正是这些尺寸，K = 8 时为 1、2、4、8、16）。
- **Phase 1**（以及 Goal forward）不使用 CUDA graph，也不做填充：`piecewise` 和 `full` 下运行已编译代码，`full-decode` 下运行 eager kernel。decode 步（以及除 `full-decode` 外的小混合步）使用 graph。使用 `--phase1-in-step`（4.10 节）时，并入 decode 步的阶段 1 随该步一起运行：`piecewise` 和 `full` 下使用分段 graph，`full-decode` 下以 eager 方式运行。
- **结果。** torch.compile（`piecewise`、`full`）用编译生成的 kernel 代替 vLLM 的自定义 kernel，会改变 bf16 数值，所以 greedy 输出可能与 eager 服务不同。`full-decode` 保留 eager kernel，只改变捕获本身带来的变化：batch 不超过最大捕获尺寸时 FlashAttention 的 split 上限，以及填充到捕获尺寸。prompt 和 transcript 的 token 精确性与模式无关。使用 torch.compile 时，编译缓存为冷时启动大约多一分钟（单张 GH200 上的 stock vLLM 0.24.0：+58 s；捕获 4 s、0.13 GiB）。
- **GPU 上的运行**（合成会话，使用作业 3270721 的干净环境（复用）和一张 GH200，每项只跑一次；测的是机制和开销，不是 benchmark 结果）：
  - 作业 3338893（runtime `f19f464`；策略 `full`，1 个 session × 3 轮，prompt 最长约 13K token，784 token 和 16 token 的 Page 都跑了）：`piecewise`、`full-decode` 和 `full` 每一轮的 token 都与 eager 相同；同一模式下的 stock vLLM 0.24.0 重放这些轮次，生成的 token 也相同。这个作业里没有任何一轮选页。
  - 作业 3339448（同一 runtime；16 token Page 下的 `observation_query`，1 个和 8 个 session × 4 轮，prompt 最长约 37K token，每种模式）：graph 模式的输出与 eager 不同。1 个 session 时，第一处差异出现在第 1 轮的回复里，早于任何一轮选页；8 个 session 时，选中的 Page 也不同（在选页生效的轮次上，与 eager 的 Jaccard 为 0.81–0.93）。8 个请求同时 decode 时的稳态 decode 步：eager 82.7 ms，`piecewise` 29.2 ms，`full-decode` 34.0 ms，`full` 28.9 ms；这是开销测量，不是端到端加速。
  - 作业 3343069（runtime `613a7d5`）：用 `full` 的 16 个 session 与 eager 下的同一负载相比，输出和选页都不同；session 池用 `full-decode` 运行（4.4 节）。
- **对比。** graph 模式的输出，连同 query 策略的选页，都可能与 eager 服务不同，所以 CUDA graph 服务应与同一模式下的 stock vLLM 对比。即使在 eager 模式下，成批运行在两次运行之间也不是 token 完全相同的（4.4 节）。
- **Gemma 4**（4.14 节）在其 window 布局上接受相同的模式，绑定的是它的全局组；未经 GPU 验证。

### 4.9 上下文上限与回复上限

三个 launcher 选项让服务能承载更长的 agent 运行，三个都已在作业 3348705 中在 GPU 上跑过（见下）。不使用它们时，vLLM 命令、`startup.json`、其中的固定值、证据文件和选页结果都与 `8c5f4d0` 逐字节相同（CPU 测试，以及一个覆盖 549 项 launcher、启动、worker 和引擎输出的一致性探针）。`--max-model-len 262144` 是后来加入的第三个上下文选项，已在作业 3360375 中在 GPU 上跑过（见下）。加入它之后，默认配置和 131072 的配置都保持逐字节相同（覆盖 531 项和 549 项输出的一致性探针，以及 131072 dry run 的 `startup.json` 和 vLLM 命令）；把它与 4.11 节的解码快速路径合并之后也是如此（同样的 549 项输出，没有差异）。

| 选项 | 默认 | 另一选择 | 固定到哪里 |
| --- | --- | --- | --- |
| `--max-tokens` | 1536（PoC 的回复上限） | 4096 | `startup.json` 的 `max_tokens` |
| `--max-model-len` | 65536 | 131072 或 262144 | vLLM 的 `--max-model-len` 和 `effective_server_config.max_model_len` |
| `--dram-arena-gib N` | 为每个运行中的 session 在上下文上限处留足空间 | 所有 session 共用一个 N GiB 的锁页 Page arena，仅限解耦布局 | `startup.json` 的 `dram_arena_pages` |

- **GPU 上的运行（作业 3348705，runtime `710bfc6`，一张 GH200；已离线复核，token 完全一致）。** 所有服务都使用 `--max-model-len 131072 --max-tokens 4096` 和 16 token 的解耦布局。单个 session（`full` 和 `observation_query`，eager，16 GiB KV）：prompt 增长到 120,864 token；128,871 和 128,873 token 的请求在准入前被前端以 HTTP 400 拒绝，之后服务继续正常工作。16 个 session（`observation_query` 加 `--active-token-budget 8192`，`--cuda-graphs full`，32 GiB KV，`--dram-arena-gib 88`）：160 轮，其中 114 轮启用了选页，各 session 达到 60K 到 111K token，没有抢占；有 6 轮正好生成了 4,096 个 token（prompt 为 33.6K 到 76.2K），其后的轮次正常继续；16 个同时运行时 ITL 为 40.3 ms；进程 RSS 最多 117.9 GiB，cgroup 峰值 134 GiB（申请了 200 GiB）。没有覆盖到：接近上下文上限时的 4,096 token 回复（计划在约 121K token 处的长回复轮次分别只生成了 69 和 215 个 token 就结束了）。锁页内存不会出现在进程的 VmPin 中，所以 128 GiB 的锁页主机内存预算依据的是 launcher 的计算，而不是实测。
- **GPU 上的运行（作业 3360375，runtime `e3efc22`，一张 GH200；已离线复核，token 完全一致，33 项检查无一失败）。** `e3efc22` 是本版本的 262144 token 各 commit 在 `175a183` 上的版本，即 rebase 到 4.11 节的 step profile 和解码快速路径之前的版本；它的 runtime 源码等于本版本去掉这两项之后的源码，而这里的服务都没有用到这两项（262144 与 `--decode-fastpath` 同时使用的情形只在 CPU 上测试过，`test_ctx_256k.FastPathTests`）。所有服务都使用 `--max-model-len 262144 --max-tokens 4096` 和 16 token 的解耦布局。
  - 单个 session，`full`，eager，24 GiB KV：21 轮。prompt 增长到 239,527 token，对这个 prompt 的回复生成了完整的 4,096 个 token（合计 243,623）。一个 259,561 token 的请求（加上 4,096 token 的回复后超过 262,144）在准入前被前端以 HTTP 400 拒绝，之后健康检查返回 200。1 个同时运行时 ITL 为 77.4 ms。同样的运行换成 `observation_query` 时达到 239,525 token，离线选页重放在全部 19 个选页轮次上都与记录的选页相同。
  - 16 个 session，`observation_query` 加 `--active-token-budget 8192`，`--cuda-graphs full`，32 GiB KV，`--dram-arena-gib 88`：273 个请求，prompt 最多 221,416 token，峰值时 16 个同时运行，16 个同时运行时 ITL 为 41.3 ms，没有抢占，也没有容量等待，GPU 显存最多 93,349 MiB。选页重放在全部 199 个选页轮次上都与记录的选页相同。一个 268,473 token 的请求被前端以 HTTP 400 拒绝，服务继续正常工作。有 5 个回复达到了 4,096 token 的上限（prompt 为 147,672 到 221,416 token）；其中有下一轮的 3 个，下一轮都 token 完全一致地继续。
  - arena 上限（2 个 session，`full`，24 GiB KV，`--dram-arena-gib 17`）：第二个 session 在第 7 轮的捕获放不进 arena，于是该请求以 HTTP 500 失败，服务日志中记录 `DRAM_ARENA_EXHAUSTED`，符合设计；第一个 session 的下一轮（197,276 token）正常完成。
  - 主机内存：16 个 session 的服务 RSS 最多 111.8 GiB，launcher 估算的最坏情况为 122.3 GiB；作业的 cgroup 峰值为 134.6 GiB（申请了 200 GiB）。唯一的发现：锁页内存不会出现在 VmPin 中（读数为 0），所以 128 GiB 的锁页主机内存预算依据的仍是 launcher 的计算。
- **如何选择上限。** 上限越长，同样的 session 需要的 GPU KV 和锁页主机内存越多（见下面的拒绝条件），所以应选能容纳 agent 历史（prompt 加回复）的最短上限：4096 token 回复时，65536 下 prompt 最多 61,440 token，131072 下最多 126,976，262144 下最多 258,048。262144 是模型的原生上下文，没有更长的选项。
- **回复上限。** 这是服务端设置，不属于 agent 协议：前端的 live controller 和引擎持有固定值；Mini-SWE-Agent XML 路由在每个请求上强制使用它，引擎在分配之前拒绝任何其他值。服务端仍然贪心解码、忽略客户端的采样字段。容量门控为每个请求预留到 ⌈(prompt + max_tokens) / P⌉ 的 Page，launcher 的余量是一次回复，⌈max_tokens / P⌉ + 1 个 block（16 token Page 下 1536 时为 97，4096 时为 257）。vLLM 把回复上限截断为上下文上限减去 prompt，所以 prompt 最多为上下文上限减去回复上限（65,536、4096 token 回复时为 61,440 token；131,072 时为 126,976；262,144 时为 258,048）。
- **启动时检查，只做一次。** 前端把 vLLM 解析出的 `--max-model-len` 与固定值比较；引擎检查自己解析后的 `max_model_len`，在任一值不是默认值时，把 context 记录（`max_model_len`、`max_tokens`）放进发给 worker 的启动记录；worker 用自己的 runner 核对这条记录，并在 `worker-attached.json` 中回显；引擎和前端都比较这份回显，`ready.json` 增加 `context`。262144 token 的服务还会在前端和引擎中各设置一次自己的请求上限（见下）。不做任何逐请求或逐步的检查。Page 目录上限、注意力和 Gated-DeltaNet 的行以及锁页 Page arena 都随 vLLM 的 `max_model_len` 而定（P = 16 时每个 session 8,193 个 Page；262144 时为 16,385 个）。
- **launcher 的拒绝条件。** 一个 session 在上下文上限处的一轮必须放得进注意力池（解耦布局：⌈L / P⌉ + 1 + ⌈16,384 / P⌉ + ⌈max_tokens / P⌉ + 1 个 block，加上 1 % watermark）或共享的 784 token 池（⌈L / 784⌉ + 1 + 6 + ⌈max_tokens / 784⌉ + 1 个 block 加 watermark，也就是 vLLM 自己启动检查的要求）；在 131072 token 下，这在所有布局里都会拒绝 `--kv-cache-gib 8`。最坏情况下的锁页主机内存（32 GiB 的 CPU 池、每个运行中 session 一份 154 MB 的 Gated-DeltaNet carry（开 `--gdn-device-handoff` 时两份），以及所有运行中 session 在上限处每个注意力 Page 的主机副本）必须不超过 128 GiB（`serve.py` 中的 `PINNED_HOST_GIB`；已验证的作业申请了 200 GiB）。所有 65536 token 的配置最多需要 99.4 GiB，都能通过。在 131072 token 下，11 个运行中的 session 放得下（P = 16 时 121.8 GiB），16 个需要 162.5 GiB，因此 16 个 session 需要 `--dram-arena-gib`（88 GiB 时为 122.3 GiB）。超出限额 arena 的捕获会让该请求失败（`DRAM_ARENA_EXHAUSTED`）；不会换出或淘汰任何东西。16 个 session 平均 90,000 token 时共持有 87.9 GiB 的 Page。如果 `--dram-arena-gib` 本身超出预算，拒绝信息会给出放得下的最大 arena（16 个 session 时为 93 GiB，开 `--gdn-device-handoff` 时为 91 GiB）。
- **connector 的 CPU 池（`--connector-pool-gib N`，需显式开启；在 GPU 上只在 6.7 节的 Terminal-Bench dry run 中运行过）。** 上面的 32 GiB CPU 池属于 vLLM 的 `OffloadingConnector`，`serve.py` 总是配置它（`--kv-transfer-config`）；worker 启动时 vLLM 会锁页并清零这个池，与 `--kv-cache-gib` 无关。本 runtime 从不经过这个池搬运数据。它在启动时隔离 connector（`remote_ra_http/connector.py`）：每次前缀查找都返回 0 个 token，任何从池中加载的请求都会被拒绝，结束的请求什么也不存，connector 也从不拿到传输任务。拷到主机和从主机恢复的 Page 走的是 runtime 自己的锁页内存：4.6 节的 Page arena（共享池中是每个捕获的 Page 一份锁页副本）和 Gated-DeltaNet carry。在 SWE-bench 作业 3354318、3357655、3357977、3360287 和 3363871（四个策略都有）的服务日志中，这个池的使用率始终是 0.0。connector 本身保留，因为 runtime 的隔离和检查挂在它上面。
  - `--connector-pool-gib N`（N 为 1 到 31；默认关闭）把池设为 N GiB 而不是 32 GiB，预算也按 N GiB 计算。1 GiB 仍能放下 20 个 784 token 的 block，或 1,024 个 16 token 的 block，在每种 `--kv-cache-gib` 下每个服务都省下约 31 GiB 锁页主机内存：每个 65536 token 的配置最多需要 68.4 GiB，而不是 99.4 GiB。对只申请一张 GH200 对应份额节点主机内存的作业，这一点很重要。
  - `startup.json` 记录 `connector_pool_bytes`，vLLM 得到的 `cpu_bytes_to_use` 为 N GiB。engine 检查自己的 vLLM 配置是否就是固定的值，worker 检查分配出的池里每个 KV 张量至少有一个 block、总量不超过 N GiB，各在启动时做一次（`remote_ra_live_v2/connector_guard.py`）。不加参数时，vLLM 命令、启动配置和 launcher 的拒绝信息都不变（与上一版本对比的一致性探针：555 + 219 项，0 处差异）。CPU 测试：`test_connector_pool.py`。
- **token 与帧大小上限**（在 CPU 上用固定的 tokenizer 测得，测试 `test_align_ctx.TokenFrameCapTests`）：126,976 token 的 Mini-SWE-Agent 请求，observation 为 SWE 类内容（代码清单、带填充的 `pytest -v` 输出、FITS 头卡片，每条最多 10,000 字符）时，每个 token 占 3.3 到 5.3 字节 source，即 1 MiB source 上限中的 424 到 666 KB；core 请求不到 0.3 MB（上限 1 MiB；131,072 个 id 最多 0.66 MB）；帧不到 1.5 MB（上限 4 MiB）；126,976 token 也在 131,072 token 的上限之内。在 126,976 token 时，source 上限在每 token 超过 8.26 字节时生效（64,000 token 时为 16.4）：只有横幅分隔线的日志（每个 token 一条 80 列分隔线，约每 token 19 字节）在任一上下文下都会在准入前被拒绝。这些是 65536 和 131072 下的上限，没有改动；262144 token 的服务把它们加倍（下一条）。
- **262144 token（已在作业 3360375 中经 GPU 验证，见上）。** `--max-model-len 262144` 是模型的原生上下文（`max_position_embeddings` 为 262144，rope 为默认类型）：不用 YaRN，不需要覆盖参数，也没有新的固定值。更长的上下文需要 YaRN，而 YaRN 会改变所有长度下的输出。和 131072 一样，只有 vLLM 的 `--max-model-len` 和 `effective_server_config.max_model_len` 改变，Page 目录（P = 16 时 16,384 个 Page）、各种行和 arena 都随 vLLM 的 `max_model_len` 而定。单个请求的字节和 token 上限随固定的上下文上限缩放（见下表）。在 CPU 上用固定的 tokenizer 测得（测试 `test_ctx_256k.TokenFrameCapTests256`）：258,048 token 的请求（262,144 减去 4096 token 的回复）每个 token 需要 3.3（代码清单、README 文本）到 5.2（FITS 头卡片）字节 source，即 0.84 到 1.35 MB 的 source，因此带填充的历史和 FITS 历史都超过了原来的 1 MiB。core 请求最多 0.61 MB，帧最多 2.9 MB。2 MiB 的 source 上限在每 token 超过 8.13 字节时生效，所以只有横幅分隔线的日志仍会在准入前被拒绝。
  - 262144 token 下 launcher 的拒绝条件。一个 session 的一轮需要 16,385 + 1,024 + 257 个 block 加 watermark（16 GiB、4096 token 回复时为 17,824），所以 `--kv-cache-gib 8` 和 `16` 在所有布局、所有 session 数下都会被拒绝；24 和 32 在 1 到 16 个 session 时都能通过。共享的 784 token 池在 4096 token 回复时需要 352 个 block（16 GiB 时可用 333 个，24 GiB 时 500 个）。
  - 262144 token 下的锁页主机内存：不设 arena 时最多放得下 5 个运行中的 session（P = 16 时 113.0 GiB）。6 个需要 129.1 GiB，16 个需要 290.5 GiB；session 池按其 K 个运行中的 session 计算。16 个 session 需要 `--dram-arena-gib`，取值在 17 GiB（上限处的一个 session，16,385 个 Page）到 93 GiB 之间（开 `--gdn-device-handoff` 时为 91 GiB）。93 GiB 平均每个 session 可容纳 95,232 token，88 GiB 为 90,112。16 个 session 都到 262,144 token 是放不下的：仅它们的 Page 就有 256 GiB。
  - 尚未改动：300 s 的准入等待和 360 s 的租约；phase 1 对每条追加消息仍最多处理 16,384 token；GPU 之外随 prompt 增长的逐请求工作（prompt 哈希、JSON 快照、反 tokenize、带全部 prompt id 的证据行）；arena 的公平性（`DRAM_ARENA_EXHAUSTED` 让下一个捕获的 session 失败，不论是谁）；严格规则下 262K 重建的容量（17,409 个 block）。
- **终身请求上限。** T 个注册任务时为 max(1,024, 256 × T)：单个 session 256 个请求的上限（前端的轮次上限、引擎的轮次序号和 generation 上限）对所有任务求和，因此它永远不会拒绝一个 session 仍可发出的请求。32 个 120 步的 agent 任务最多发出 3,840 个请求，上限为 8,192；`CAPACITY_WAIT` 重试不占用终身记录。单个 session 的上限仍是 256 个请求，所以 `run_agents.sh` 会拒绝大于 256 的步数上限。
- **SWE-bench。** `run_agents.sh` 接受 `STEP_LIMIT`（默认 90）和 `MAX_TOKENS`（默认 1536），用于 model override（7.4 节）。

**131,072 token、16 个 session、P = 16、4096 token 回复时的容量**（按 4.4 节的规则推导：15 个空闲 session 加一个处于全上下文峰值的准入轮次再加余量，必须放进可用注意力 block；Mamba 池占 100 个 block）：

| 策略与空闲模型 | 每个 session 的空闲 block | 所需 KV GiB | 24 放得下 | 32 放得下 |
| --- | ---: | ---: | :-: | :-: |
| `full`（严格规则下所有策略都如此，因为重建会回到全部驻留） | 8,193 | 135.4 | 否 | 否 |
| 40 %，只算预算 | 3,277 | 62.6 | 否 | 否 |
| 40 % + 典型的上一轮（7K token） | 3,715 | 69.1 | 否 | 否 |
| 40 % + 最坏的上一轮（16,384 + 4,096） | 4,557 | 81.6 | 否 | 否 |
| `--active-token-budget 8192`，只算预算 | 513 | 21.8 | 是 | 是 |
| 预算 8192 + 典型的上一轮 | 951 | 28.2 | 否 | 是 |
| 预算 8192 + 最坏的上一轮 | 1,793 | 40.8 | 否 | 否 |

65,536 token、4096 token 回复时，同样各行需要 70.8、34.4、40.9、53.4、17.8、24.2 和 36.6 GiB（1536 token 回复时为 4.4 节的数值）。无论上下文上限多大，16 个 `full` session 只有在上下文较短时才不会死锁（24 GiB 时每个 session 约 18K token，32 GiB 时约 26K）。在使用 Goal rewriter 时估计 GPU 上约有 36 GiB 可留给 KV，因此 131,072 token 下的 16 个 session 只有在使用较小的固定预算、且每轮远低于 observation 上限时才不会死锁。

**按上下文上限列出的单个请求上限**（`remote_ra_live_v2/context_limits.py` 中的 `request_caps`；262144 的值由前端和引擎在启动时各设置一次，65536 和 131072 下不设置任何东西）：

| 上限 | 65536 和 131072 | 262144 |
| --- | ---: | ---: |
| XML source（对话 JSON）和每条编码的消息：max(1 MiB, 8 B × L) | 1 MiB | 2 MiB |
| core 请求（msgpack，每个 prompt id 1 到 5 字节）：max(1 MiB, 8 B × L) | 1 MiB | 2 MiB |
| 认证帧：2 × (source + core) | 4 MiB | 8 MiB |
| EngineCore 的 registered-add 消息：帧 + 1 MiB | 5 MiB | 9 MiB |
| 每个请求的 prompt id：max(131,072, L) | 131,072 | 262,144 |

**262,144 token、16 个 session、P = 16、4096 token 回复时的容量**（按同一规则推导；作业 3360375 在 32 GiB 下用 8192 token 的预算运行，没有容量等待）。准入轮次的全上下文峰值本身就有 17,409 个 block（17 GiB）。最后一列是 24 和 32 GiB 下最多放得下的 session 数：

| 策略与空闲模型 | 每个 session 的空闲 block | 所需 KV GiB | 24 放得下 | 32 放得下 | 24 / 32 GiB 下的 session 数 |
| --- | ---: | ---: | :-: | :-: | :-: |
| `full`（严格规则下所有策略都如此） | 16,385 | 264.6 | 否 | 否 | 1 / 1 |
| 40 %，只算预算 | 6,554 | 119.2 | 否 | 否 | 1 / 3 |
| 40 % + 典型的上一轮（7K token） | 6,992 | 125.7 | 否 | 否 | 1 / 2 |
| 40 % + 最坏的上一轮（16,384 + 4,096） | 7,834 | 138.1 | 否 | 否 | 1 / 2 |
| `--active-token-budget 8192`，只算预算 | 513 | 29.8 | 否 | 是 | 8 / 16 |
| 预算 8192 + 典型的上一轮 | 951 | 36.3 | 否 | 否 | 5 / 12 |
| 预算 8192 + 最坏的上一轮 | 1,793 | 48.7 | 否 | 否 | 3 / 7 |

这些行中只有 8192 token 的预算、且空闲 session 只持有自己的预算时，262,144 token 下的 16 个 session 才不会死锁（32 GiB 时）。

### 4.10 Admission 优化（需显式开启）

admission 是引擎在新一轮 decode 之前要做的工作：阶段 1，`goal_query` 的 Goal，选页和预留（第 5 节）。默认情况下，admission 按到达顺序一次一个；阶段 1 是一次单独的 forward，期间正在运行的请求都要等待；Gated-DeltaNet 状态经锁页主机内存从阶段 1 交给阶段 2。三个 launcher 参数可以改变这些，默认都关闭（括号里的字母是代码注释中对应的 admission 设计编号）：

| 参数 | 默认 | 作用 | 启动时固定到哪里 |
| --- | --- | --- | --- |
| `--gdn-device-handoff`（e） | 关 | 在 device 上把 Gated-DeltaNet 状态从阶段 1 交给阶段 2，不经主机拷贝 | `startup.json` 的 `admission.gdn_device_handoff` = `true` |
| `--phase1-in-step`（b） | 关 | 把一次 admission 的阶段 1 放进下一个 decode 步里运行，而不是单独一次 | `admission.phase1_in_step` = `true` |
| `--admission-overlap K`（f） | 1（关） | 最多同时有 K 个 admission 在进行，2 ≤ K ≤ 同时运行的 session 数 | `admission.admission_overlap` = `{"permits": K, "admission_seconds": 300 + 120(K − 1), "lease_seconds": 360 + 120(K − 1)}` |

只有在开了某个参数时，`startup.json` 才有 `admission` 记录；三个都关闭时，启动配置、证据文件和 vLLM 命令都不会多出任何内容。引擎和 worker 在启动时各检查一次这条记录，worker 在 `worker-attached.json` 中回显它，`ready.json` 记录它；不做任何逐请求或逐步的检查（`remote_ra_live_v2/admission_flags.py`；CPU 测试 `test_gdn_handoff.py`、`test_adm_phase1_step.py`、`test_adm_overlap.py`）。`examples/slurm/serve_gh200.sh` 不传这三个参数中的任何一个（也不传 `--cuda-graphs`）。

- **`--gdn-device-handoff`。** 阶段 1 把它的 probe slot 留在 GPU 上，阶段 2 的第一步在 device 上把它拷贝到阶段 2 的 slot；跨 Mamba block 时也在 device 上拷贝。提交后的 carry 是唯一的主机副本：一轮结束时它被异步拷到主机，写入该 session 两个锁页 carry 缓冲区中没有存放已提交 carry 的那一个，并由 drain 等待完成；阶段 1 从那里恢复它（`remote_ra_live_v2/gdn_handoff.py`）。延续轮次的阶段 2 占用三个 Gated-DeltaNet slot（运行、跨块和保留的 probe slot），容量门控也按三个计算。launcher 的锁页主机内存预算（4.9 节）计入第二个 carry 缓冲区：每个运行中的 session 多 153.9 MB（16 个 session 时 2.3 GiB；131,072 token、16 个 session、`--dram-arena-gib 88` 时需要 124.6 GiB，而不是 122.3）。索引恢复时临时使用的锁页中转副本在恢复完成后释放，不计入预算。每个 `runtime.jsonl` 行增加 `restore.d2d_bytes`（第 8 节）。
- **`--phase1-in-step`。** admission 把它的阶段 1 挂起，引擎把它作为额外一行，和正在运行的请求一起放进下一个 decode 步（混合步）；该步收集到结果后，admission 继续（`remote_ra_live_v2/phase1_step.py`）。以下情况阶段 1 仍单独运行：没有其他请求在运行（`probe_only`），它的 token 放不进该步在每步 16,384 token 预算中剩下的部分，或者连续两步都没有空闲的 batch 行（`fallback`）；带有 `goal_query` Goal forward 的阶段 1 也仍单独运行，一次只运行一个 session 的服务（`max_num_seqs` 为 1）总是单独运行。阶段 1 从不分块。使用 CUDA graph 时，带有阶段 1 的步使用分段 graph（`piecewise`、`full`）或以 eager 方式运行（`full-decode`）。native 文件记录 `phase1_mode` 和挂起时间。
- **`--admission-overlap K`。** 最多同时有 K 个 admission 在进行：前端的 admission 锁变成一个有 K 个许可、先到先得的信号量，每个 session 有自己的 Goal 线程和自己的一份 Goal 构造器，所以一个 admission 在等它的 Goal（一次 rewriter 调用）或它的步内阶段 1 时，其他 admission 可以继续（`remote_ra_live_v2/admission_overlap.py`）。Goal rewriter 仍是一个一次只服务一个请求的进程，阶段 1 也仍然一次只为一个 admission 运行。每次 rewriter 调用等待该进程的时间和调用本身的用时，按调用记录在 `goals.jsonl`（`wait_ms`、`call_ms`），按请求记录在 native 文件（`goal_rewrite_wait_ms`、`goal_rewrite_call_ms`，与 `goal_construct_ms` 并列；第 8 节）。不开这个参数时同一时间只构造一个 Goal，所以 `wait_ms` 接近 0，排队发生在引擎前面的 admission 等待里。前端等待 admission 确认的时间和引擎租约随 K 增长，并在启动时固定：300 + 120(K − 1) 秒，租约再多 60 秒（K = 16 时为 2,100 秒和 2,160 秒；不开这个参数时为 300 秒和 360 秒）。不会发生抢占：每个进行中的 admission 在被接受之前都持有一份容量预留；放不下的 admission（与正在运行的请求和其他预留一起算）会以 `CAPACITY_WAIT` 被拒绝并重试，与 4.4 节相同。三个参数都开时，一个 admission 预留两个峰值中较大的那个；在 Mamba 池中，阶段 2 为 9 个 block（三个 Gated-DeltaNet slot；不开 `--gdn-device-handoff` 时为 6 个），阶段 1 为 3 个（有 Goal forward 时为 6 个）。因此 5 个 session 时（11 个 slot，33 个 block）同时只能容纳约 3 个 admission，16 个 session 时（33 个 slot，100 个 block）约 11 个，其余的重试（作业 3349504 中，5 个 session 时有 3 个在进行、16 个 session 时有 10 或 11 个在进行时，新的 admission 被拒绝）。每次拒绝都会在服务日志中写一行带 `traceback` 字段的 `engine_rejected`（6.3 节）：这是预期行为，不是崩溃。
- **GPU 上的运行（作业 3349504，runtime `4eae441`，即合并进本版本的 admission 工作；一张 GH200；已离线复核，token 完全一致）。** `goal_query` 带 Qwen3-0.6B rewriter，5 个和 16 个 session，`--cuda-graphs full`，24 GiB KV。离线选页重放不适用：它只覆盖 `observation_query` 和 `random_fill`。16 个 session 时，不开参数，请求平均等待 admission 48.3 秒（占请求时间的 86 %），服务每分钟完成 13.8 轮。开 `--admission-overlap 16` 时，admission 等待为 0，服务每分钟完成 17.3 轮（只跑了一次）；等待转移到了 rewriter 上，它是一个一次只服务一个请求的进程（Goal 构造平均用时：不开参数时 3.8 秒，开时 33.9 秒）。三个参数都开、K = 16 时，每分钟完成 15.7 轮，有 726 次 `CAPACITY_WAIT` 拒绝。5 个 session 时，三个服务每分钟分别完成 11.1、12.4 和 12.4 轮（各跑一次）。开了参数的服务，其输出和选页与不开参数的服务不同，成批运行本来就是如此（4.4 节）。
- **GPU 上的运行（作业 3349078，runtime `4eae441`，一张 GH200；已离线复核，token 完全一致，选页重放与记录一致）。** `observation_query`，16 token 解耦布局，24 GiB KV。单个 session、eager、8 轮 6,500 token 的 observation：分别开 `--gdn-device-handoff`、`--phase1-in-step` 以及两者都开时，每一轮的 token、选页和 Page 向量都与不开的服务相同；不开的服务重跑一次也完全相同（全部 2,866 个 Page 的阶段 1 向量一致）。16 个 session、`--cuda-graphs full`、每个 session 6 轮 2,600 token 的 observation：两个开关都开时，引擎线程上每轮的 admission 时间从 342 ms 降到 55 ms，解码中请求的停顿从运行时间的 36.8 % 降到 6.3 %（把变慢的混合 step 算进去是 13.0 %；94 % 的后续轮次在 step 内完成阶段 1），16 个同时运行时的 ITL 从 32.0 ms 升到 32.9 ms，96 个请求早 1.8 % 跑完（73.8 s 对 75.2 s）。5 个 session、150 token 的 observation：停顿从 13.5 % 降到 4.1 %（算上 step 为 12.2 %），早 3.8 % 跑完。
- **`--goal-rewriter-batch N`（默认关闭；需要 `goal_query` 加 `--goal-rewriter-model` 和 `--admission-overlap K`，2 ≤ N ≤ K ≤ 16；仅限 Qwen：Gemma 4 profile 拒绝它，4.14 节；已在 SWE-bench 作业 3404230 中以 N = 4 在 GPU 上运行，见下）。** 开 `--admission-overlap 16` 时，各 session 的 Goal rewriter 调用排队等那一个 rewriter 进程。在已核验的 16 个 session 的 SWE-bench 运行中，启动时 16 次初始化分别在 19.5、26.4、30.4、…、98.5 秒后完成，每次调用约 5.3 秒；有 rewriter 调用的 Goal 构造中位数为 81 秒。开这个参数时，排队中的调用最多 N 个一起发给一个子进程，由它用一次批量的 greedy `generate` 运行（`remote_ra_live_v2/goal_rewriter_batch.py`、`goal_rewriter_child.py`）：
  - **不变的：** 何时调用 rewriter、发送的内容（上游的请求行，逐字节相同）、prompt、JSON schema 及其约束、`max_new_tokens`、每个回答的解析和校验（每次调用都运行上游 `PersistentCommandGoalRewriter.rewrite` 的代码）以及回退。每次调用得到自己的回答；子进程的错误或被拒绝的候选只影响那一次调用。只有一个调用的批次就是上游的 `generate`。
  - **改变的：** 模型调用。各 prompt 左填充后合成一次 `generate`，每一行由它自己请求的 lm-format-enforcer prefix function 约束。**批量 greedy 解码与一次一个请求的解码并非逐位相同：** 填充和 batch 维度改变浮点归约顺序，接近平局时可能选出不同的 token，被改写的 Goal（或其回退）可能与不开这个参数的服务不同。只在设置相同的服务之间比较策略。子进程还把 lm-format-enforcer 的 tokenizer 数据只构建一次，而不是每个请求构建一次（允许的 token 相同；子进程启动时自检比较新建和缓存的结果，不一致则保留逐请求构建）。
  - **超时和失败：** 固定的 120 秒超时从调用所在批次发送时开始计算，所以发送前排队的时间与以前一样不计入。超时、子进程退出或无法读取的回答会让该批次的每个调用都以上游的错误失败（各自回退），下一个批次会重启子进程。CUDA 显存不足时批次拆成两半运行；批量 `generate` 的其他失败会让其中的请求逐个经上游代码运行。
  - **启动时固定的内容：** `startup.json` 和 `ready.json` 记录 `goal_rewriter_batch: N`，引擎在启动时检查一次它的 rewriter 以 N 批量运行。批量服务的每个 `goals.jsonl` 交互增加 `batch_id` 和 `batch_size`（第 8 节）。不开这个参数时，启动配置、rewriter 命令和证据都不变（identity probe：0 处差异）。
  - **GPU 显存：** Qwen3-0.6B 每个 prompt token 约需 112 KiB 的 KV cache，16 个 4–6K token 的 prompt 组成的批次在 27B 服务旁边需要 7–10 GiB。先用 N = 4，并从 `goals.jsonl` 读出 prompt 大小（`input_tokens`）。
  - **预期效果（未测量）：** 当一次调用的主要开销是模型的解码步而不是约束时，一个批次用约一次调用的时间服务最多 N 个调用；单独运行的调用不会变快。`goals.jsonl` 中每次调用的时间（第 8 节）把调用的准备部分（`call_ms` − `generation_latency_ms`）和生成部分分开。
  - **CPU 测试（`test_goal_rewriter_batch.py`）：** 使用遵循 Transformers 批量 greedy `generate` 语义的假模型和假 enforcer。每个批量回答都等于该请求单独运行时的上游回答，16 个 session 的 Goal 与不批量时相同（N = 16 和 N = 4）。测试还覆盖显存不足和失败路径、超时、不匹配的回答、被取消的 admission 和关停。Transformers、lm-format-enforcer 和该 checkpoint 第一次以批量方式运行是在 GPU 上的作业 3404230（下一项）。
  - **GPU 上的运行（SWE-bench 作业 3404230，runtime `5b15b2f`，N = 4，16 个 session；[docs/SWEBENCH_BENCHMARK.zh-CN.md](docs/SWEBENCH_BENCHMARK.zh-CN.md)）：** 运行通过了检查（`SWE32_PASS_VERIFIED`），32 个 instance 全部完成。59 次 Goal fallback 中有 9 次是只在批处理时出现的失败：4 次 CUDA out-of-memory 和 5 次 cuDNN attention 错误（不批处理的作业 3404696 中没有）；它们都以回退结束，其恢复逻辑的修复正在一个开发分支上进行。
- **推荐**，也是我们 SWE-bench 运行使用的配置：`goal_query` 加 `--admission-overlap 16`（16 个 session 时），另外两个参数关闭；其他所有策略三个参数都关闭。

### 4.11 解码快速路径（需显式开启）

在 SWE-bench 运行中，16 个请求同时运行、Page 为 16 token 时，本 runtime 的解码比 stock vLLM 慢 1.7 倍：token 间延迟 52.4 ms，stock 为 30.0 ms。原因是每个解码步里的 host 端工作随"运行中的请求数 × 每个请求的 Page 数"增长。

- 每个运行中的请求每步都要发一次 `allocation_update` RPC。这个 RPC 为了数出本请求的 block，会扫描所有运行中请求的全部分配记录（`remote_ra_http/evidence.py`）。
- 同样长的请求，16 token 的 Page 数是 784 token 的 49 倍，所以 Page 越小，扫描越慢。
- 在 `full` 的 SWE-bench 运行（作业 3357655）中，这些调用共耗时 245.9 s，占整场 13.8%（321,338 次，每次 0.765 ms）。

`--decode-fastpath`（默认关闭）用更少的工作得到相同的值：

- 用每个请求已占 block 的计数代替扫描（`remote_ra_live_v2/decode_fastpath.py`）。
- 每步的 lease 只检查上一步之后新增的 attention lease 的类型。
- 每步只从设备读一次 position，而不是每个请求读一到两次。
- worker 每步只同步一次设备，而不是每个请求一次。
- 动作标记的停止判断只解码新 token；只有新 token 含 `>` 时才解码全部输出。这一点是精确的，因为标记是 ASCII，并以 `>` 结尾。

**开关的记录。**

- 打开时，`startup.json` 记录 `decode_fastpath: true`；worker 在其证据中回显，`ready.json` 也会记录；只在启动时检查一次。
- 不开时，启动配置、证据和 vLLM 命令都不增加任何内容，代码路径与之前相同（一致性探针：549 项，0 处差异）。

**CPU 测试（`test_decode_fastpath.py`）。**

- 在随机修改下，占用计数与扫描结果相同。
- lease、position、拒绝和标记判断都完全相同，包括使用固定的 Qwen3.6 tokenizer 时。
- 在多 session 的模拟环境中（16 和 784 token 的 Page，`full` 和 `observation_query`），开和关时的 RPC 内容、证据文件、参与计算的 Page 和回复都完全相同。

**逐步计时。** 环境变量 `REMOTE_RA_STEP_PROFILE=1`（默认关闭）会在服务日志里为每个解码步写一行 `REMOTE_RA_STEP {json}`。

- 每行记录该步各部分的 host 时间，以及用 CUDA event 计时的前向时间（不增加同步）。
- `tools/step_profile.py` 负责汇总这些行。
- 它只用于诊断，不属于证据。

**Torch profile。** 环境变量 `REMOTE_RA_TORCH_PROFILE=<绝对目录>`（默认关闭）让 `serve.py` 传入 vLLM 自己的 `--profiler-config {"profiler":"torch","torch_profiler_dir":<目录>}`，于是服务带有 vLLM 的 `/start_profile` 和 `/stop_profile` 路由，trace 窗口和 stock vLLM 一样通过它们开启和结束（`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` 会加上 vLLM 的命名区间）。

- startup config、其 pin 和服务的环境都不变，`--language-model-only` 仍是最后一个参数；不设置时 vLLM 命令不变。
- 不能与 `REMOTE_RA_STEP_PROFILE=1` 同时使用（一次只用一个 profiler）。
- 它只用于诊断，不属于证据。只在 CPU 上测试过（`test_torch_profile.py`）。

**GPU 上的运行（作业 3366128，runtime `43fd9c3`，一张 GH200；已离线复核）。**

- **负载。** 解码微基准：16、8、4、1 个请求同时运行，上下文约 2K 和 20K token；另有 16 段类 SWE 历史，8K 到 48K token（平均 prompt 21.6K）。
- **服务。** 本 runtime 使用 `--cuda-graphs full` 和同步调度，对照为开启和关闭异步调度的 stock vLLM 0.24.0。

16 个请求同时运行时，在类 SWE 历史上的 token 间延迟：

| 服务 | 每个 token 的 ms |
| --- | ---: |
| 本 runtime，16 token 的 Page | 53.8 |
| 本 runtime，16 token 的 Page，`--decode-fastpath` | 36.1 |
| 本 runtime，784 token 的 Page | 40.8 |
| 本 runtime，`observation_query`，`--decode-fastpath` | 31.0 |
| stock vLLM，同步调度 | 29.2 |
| stock vLLM，异步调度 | 27.6 |

- 16、64、784 token 的 Page 下，前向时间相同：16 个请求时每步约 21 ms。
- 打开开关后，每步的 host 时间减半，从 18.5 ms 降到 9.2 ms。
- 只有一个请求时，开关影响很小：2K token 时开和关都是 20.3 ms。
- KV cache 布局（HND 或 NHD）在两种服务上都没有可测的差别（在约 1 ms 以内）。
- 结论为部分通过：每轮的测量窗口在第一个请求结束时截止，只有 30 到 60 个解码步，而不是计划的 150 步；所有服务都跑完了全部轮次。

**从下一次 benchmark 运行起，推荐所有策略都打开。** [docs/SWEBENCH_BENCHMARK.zh-CN.md](docs/SWEBENCH_BENCHMARK.zh-CN.md) 中的 SWE-bench 运行使用 runtime `5b15b2f`，打开了这个开关和 `--decode-fastpath2`（作业 3404696 和 3404230）；早先的 SWE-bench 运行使用 `e877051`，没有打开它。

**第二阶段：`--decode-fastpath2`（需显式开启；需要 `--decode-fastpath`）。** 16 个请求同时运行时，作业 3366128 的解码每个 token 仍比 stock vLLM 慢 6 到 7 ms（36.1 对 29.2 ms，两边都是同步调度）；其逐步计时显示，每步 9.2 ms 的 host 时间里约有 5.5 ms 花在本 runtime 中。`--decode-fastpath2`（默认关闭）再去掉其中一部分工作，得到相同的值：

- 每个解码步只发一次 `allocation_batch` worker RPC，携带所有已调度请求的分配更新，而不是每个请求一次 `allocation_update` RPC。自 worker 上次接受以来行没有变化的更新不带行发送（`remote_ra_live_v2/decode_fastpath2.py`、`engine_boundary.py`）。worker 仍逐条按原来的方式处理，所以每个 native 文件里每个请求每步仍有一个字段相同的 `allocation_update` 帧。
- 每个请求每步的 host 工作更少：native 分配参数不经 `inspect` 绑定；没有变化的行、lease 和元组直接复用而不重建；请求的活跃 KV 字节按 KV 组计算；动作标记的停止判断按 token id 记住该 token 是否含 `>`。
- 前向之后每个请求的工作（关闭 lease、Page 压缩、Gated-DeltaNet 捕获）放在 vLLM 排队 logits 计算之后运行，两者重叠。

**开关的记录。** 打开时，`startup.json` 记录 `decode_fastpath2: true`（没有 `decode_fastpath: true` 时拒绝）；引擎在启动时把它发给 worker 一次，`worker-attached.json` 和 `ready.json` 都会记录。不开时不增加任何内容，代码路径与只开 `--decode-fastpath` 时相同（一致性探针：549 项，0 处差异）。

**Gemma 4**（4.14 节）接受这两个开关；在其 window 布局上，它们还加上滑窗窗口的增量观测和增量的滑窗行（未经 GPU 验证）。

**CPU 测试（`test_decode_fastpath2.py`）。** 在多 session 的模拟环境中（784 和 16 token 的 Page，`full` 和 `observation_query`；`goal_query` 分别配合 GDN 交接和步内阶段 1），回复、计划、参与计算的 Page、计算的 position、证据文件和每个请求的帧都与只开 `--decode-fastpath` 时相同；把每个 batch 逐条展开后，与其 `allocation_update` RPC 相同。**GPU 上的运行：** 它在 SWE-bench 作业 3404696 和 3404230（runtime `5b15b2f`）中运行过，但还没有测量它对 token 间延迟的影响。

**第三阶段：`--decode-fastpath3`（需显式开启；需要 `--decode-fastpath2`）。** 两个快速路径都打开后，本 runtime 在解码步前向之前仍然加入的 host 工作，大部分是每个运行中请求的固定开销：worker 的步骤（观察、claim、materialize、position、步 lease）、FullAttention 行检查和每个请求的分配路径。`--decode-fastpath3`（默认关闭）去掉其中一部分，得到相同的值：

- 若一个请求在这一步没有追加 FullAttention 块，且 Mamba 行在同一个槽位下标处保持不变（16 token 的 Page 下每 16 步有 15 步如此），则在做完相同的列表、设备行和 Mamba 检查之后，直接沿用上一步的 Mamba lease、key 元组和（为空的）claim 差异（`remote_ra_live_v2/decode_fastpath3.py`、`allocation_owner.py`）。设备行按字节比较，对同一种整数 dtype 与 `np.array_equal` 结果相同（`allocation_owner.py`、`worker_rows.py`）。
- 这样的请求在 `materialize` 只会标记这一步时跳过它；在其中不可能发起 IO 时跳过它的 IO 归属作用域。它的步 lease 在构建时不再重复检查在接受时已检查过的 Mamba 槽位（`qwen.py`）。
- 一步的 position 列从 GPU 一次拷贝读回，而不是两次。
- worker 端，不带行发送的分配更新直接应用并计量，帧不变；引擎把每个请求都要 import 的模块只绑定一次（`engine_boundary.py`）。

**开关的记录。** 打开时，`startup.json` 记录 `decode_fastpath3: true`（没有 `decode_fastpath2: true` 时拒绝，Gemma 4 profile 也拒绝）。引擎在启动时把它发给 worker 一次，`worker-attached.json` 和 `ready.json` 都会记录。不开时不增加任何内容，代码路径与只开 `--decode-fastpath2` 时相同（一致性探针：555 + 219 项，0 处差异）。

**CPU 测试（`test_decode_fastpath3.py`）。**

- 在随机输入上，字节比较与 `np.array_equal` 相同，一次拷贝的 position 读取与两次读取相同。
- 在随机的步骤序列上（Page 和 Mamba 跨界、行移动、请求结束后块被复用、各种篡改），owner 的稳定观察和 materialize 与完整的逐步路径相同：结果、拒绝和状态都相同。
- 在随机的 lease 序列上，新的步 lease 与原来的相同，拒绝也相同。
- 在多 session 的模拟环境中（784 和 16 token 的 Page，`full` 和 `observation_query`；`goal_query` 分别配合 GDN 交接和步内阶段 1；还包括 vLLM 形状的块表和 MRoPE position，稳定路径在这些情况下运行），回复、计划、参与计算的 Page、计算的 position、证据文件、每个请求的帧和 RPC 都与只开 `--decode-fastpath2` 时相同。较长的用例在 `REMOTE_RA_SLOW_TESTS=1` 时运行。
- CPU 微基准（x86 模拟环境，16 个请求、每个 20K token、16 token 的 Page，7 次运行的中位数）：一个解码步（全部是 host 工作，包括模拟环境自己的前向）只开 `--decode-fastpath2` 时为 6.15 ms，打开 `--decode-fastpath3` 后为 5.25 ms（每步均值；中位步为 5.44 和 4.70 ms）。

**未经 GPU 验证：** 还没有测量它对 token 间延迟的影响。

### 4.12 纯文本：`--language-model-only`（需显式开启）

Qwen3.6-27B 是多模态 checkpoint，但本 runtime 只服务文本。不加这个参数时，vLLM 0.24.0 仍会加载视觉塔，并用一张假图片对 encoder cache 做 profiling。在服务报告就绪之前，它还会用按上下文上限构造的假图片和假视频预热多模态处理器（作业 3357655 的日志：`Multi-modal warmup completed in 13.884s`）。这次预热会在不到一秒的时间里占用数十 GiB 主机内存；如果作业只申请一张 GH200 对应份额的节点主机内存，服务可能在 READY 之前就被杀掉。

`serve.py --language-model-only`（默认关闭）把 vLLM 自己的 `--language-model-only` 传给 vLLM，它把所有多模态上限设为 0。在固定的 vLLM 0.24.0 源码中，这意味着：

- 视觉塔被占位层替换，既不构建也不加载（`vllm/model_executor/models/interfaces.py` 中的 `_mark_tower_model`；`qwen3_5.py` 在其中构建 `visual`）。
- renderer 不创建多模态处理器，所以两次多模态预热都不运行（`vllm/renderers/base.py`），也不建立 encoder cache，不对它做 profiling。
- 语言模型的权重不变。KV cache 仍由 `--kv-cache-memory-bytes` 决定，所以它的 block 和池，以及 launcher 的 KV 和主机内存检查都不变。视觉塔的权重（BF16 约 0.9 GiB，按视觉配置计算，未实测）在 GPU 上空出来。
- 有一处计算会变。纯文本模式下，vLLM 把 16 个全注意力层的 QK-norm、RoPE 和输出门拆分合成一个融合 kernel 运行，而不是分开的几个算子（`qwen3_next.py` 中的 `Qwen3NextAttention`）。算术相同（runtime 每一步都检查三行 mRoPE 位置等于逻辑位置），但输出不一定与不加这个参数的服务逐位相同。runner 也改为把 token id 而不是它自己查好的 embedding 交给模型；这一变化是精确的（模型内部做同样的查表）。

**stock vLLM 对照也要传同一个参数**，这样两边加载同样的模型、运行同样的 kernel；只比较这一设置一致的服务。[docs/SWEBENCH_BENCHMARK.zh-CN.md](docs/SWEBENCH_BENCHMARK.zh-CN.md) 中的 stock 命令传了它；`examples/slurm/serve_gh200.sh` 不传它。

**参数的记录。**

- `startup.json` 记录 `language_model_only: true`，其 `effective_server_config` 中也有 `language_model_only: true`；前端把 vLLM 解析出的参数与之比较。
- engine 检查自己解析出的多模态配置。worker 检查 vLLM 把 image 和 video 的上限解析为 0，并且不持有任何视觉塔张量，然后在 `worker-attached.json` 中回显 `{"mm_limits": {"image": 0, "video": 0}, "vision_tower_tensors": 0}`；`ready.json` 也会记录。每项检查只在启动时做一次（`remote_ra_live_v2/language_model_only.py`）。
- 不加参数时，vLLM 命令、启动配置和证据都不增加任何内容（与上一版本对比的一致性探针：555 + 219 项，0 处差异）；如果 vLLM 仍以 language-model-only 运行，前端、engine 和 worker 都会拒绝。

**CPU 测试（`test_language_model_only.py`）：** launcher、启动检查、worker 证据和 `ready.json`；加上参数后，多 session 模拟环境产生的 RPC 内容、证据文件和 token 都相同。

**GPU 上的运行：** `full` 的 Terminal-Bench dry run（6.7 节）是第一次带这个参数的运行；它的日志有 vLLM 的纯文本日志行，没有多模态预热。作业 3404696 和 3404230 的 SWE-bench 运行（`goal_query`）也带了它。

### 4.13 释放空闲 session 的 Page（需显式开启）

两轮之间，一个 session 在 GPU 上钉住它的常驻 Page（上一轮选中的页，加上那一轮封存的页）和 live tail 页；它的 Gated-DeltaNet carry 在主机上，每个已封存的页在锁页 Page arena 里也有一份主机副本。16 个 session、32,768 token 的预算、16 token 的 Page 时，每个空闲 session 约钉住 2,300 页，16 个合计约 37,000 页，超过 25 或 32 GiB 的 KV 池（4.9 节的容量规则）。不开这个选项时，容量检查会等运行中的请求结束；一旦所有 session 都空闲，就没有任何东西能释放 block，此后每一轮都会一直等待（在此版本之前，其中第一轮因为没有请求在运行而被接纳，随后因缺 block 失败）。

`--release-idle-pages`（默认关闭，仅限解耦布局）让容量检查改为释放 Page（`remote_ra_live_v2/idle_release.py`）：

- **释放对象。** 一轮放不下时，容量检查释放若干空闲 session 的全部常驻 Page 来补足 attention 池的缺口，最久未用的先释放。空闲指已打开、没有已接纳或运行中的请求，且不是正在接纳的那个 session。下一个请求已到达服务端但仍在等待接纳的 session 也算空闲：它被释放的页会在该请求的阶段 1 之前载回（否则，当各 session 的下一个请求都在等待时，被它们钉满的池永远腾不出 block）。live tail 页从不释放。如果所有空闲 session 加起来也补不足缺口，就不释放任何页，这一轮照旧以 `CAPACITY_WAIT` 等待。因为没有请求在运行而被容量检查直接接纳的一轮，会先释放它需要的 Page；开 `--admission-overlap K` 时，在同时也没有 admission 进行时同样如此（本版本修复：此前 ledger gate 直接接纳这样的一轮而不释放）。
- **重新载入。** 被释放的页在该 session 下一轮的阶段 1 之前从主机副本恢复，阶段 1 会用到它们。这一轮的运行与这些页从未离开时完全相同；容量检查在两个峰值里都计入重新载入的页。
- **接纳过程中不等待。** 释放和重新载入都是两步之间同步执行的引擎操作；拒绝仍是立即返回的 `CAPACITY_WAIT`，所以前端 300 s 的确认等待只覆盖已接纳的这一轮本身的工作。
- **开关的记录。** `startup.json` 记录 `release_idle_pages: true`。引擎在启动记录里把它发给 worker 一次；worker 检查它，之后才接受释放和重新载入操作；`worker-attached.json` 和 `ready.json` 会记录它。不开时不增加任何内容，代码路径与之前相同（一致性探针：549 项，0 处差异）。
- **证据。** 输出目录里的 `capacity.jsonl` 每次释放和每次重新载入各一行（`schema` 为 `remote-ra-capacity-event-v1`：session、它的页和 block、两个池操作前后的空闲 block、正在接纳的请求及其缺口）。每个 native 文件和 `runtime.jsonl` 行增加 `idle_release`（为接纳该请求而释放的 session、它重新载入的页），`engine-cleanup.json` 增加总计。

**CPU 测试（`test_idle_release.py`）。** 在批量引擎和 worker 的假实现上，16 token 的 Page，16 个 session，`observation_query` 加 `--active-token-budget 32768`，每个 session 12 轮，每轮 observation 约 3,600 token（prompt 最长 39,845 token，最后几轮选 2,048 页）：

| Attention 池 | 不开选项 | 开选项 |
| --- | --- | --- |
| 25,600 block（空闲 session 钉住的页最多到 36,394） | 192 轮中的第 129 轮一直等待（没有请求在运行时的 `CAPACITY_WAIT`；在此版本之前它被接纳后失败）：16 个空闲 session 钉住 25,406 block，空闲 193 个，阶段 1 需要 224 个 | 192 轮全部完成；释放 35 次（65,090 页），重新载入 29 次（51,446 页），1 次 `CAPACITY_WAIT` |
| 48,000 block | 192 轮全部完成 | 192 轮全部完成，不释放任何页 |

开选项时，每个 session 的 prompt、选页、实际读取的页和回复都与 48,000 block 的运行相同。较短的用例（900 对 9,000 block）在默认测试中以 `observation_query` 运行，在慢速测试（2.3 节）中再加上 `full`；32,768 token 的用例约需 5 分钟，在慢速测试中运行。尚未测量：GPU 上重新载入的主机到设备时间，以及真实负载下一个 session 在下一轮之前被再次释放的频率（轮流发言时，最久未用的正是下一个要运行的 session）。

**GPU 上的运行：** MemoryAgentBench 作业 3397842（Qwen3.6，runtime `0d37346`，16 个 session）开着这个选项在两种池大小下运行：16 GiB 的池释放 139 次、重新载入 124 次，24 GiB 释放 13 次、重新载入 13 次。两种池下全部 32 个 case 完全相同，每次重新载入都恰好恢复了一个已完成的轮次提交过的 Page。该作业唯一失败的检查（route-b）是 harness 检查本身的错误：它把引擎完成这一轮之后 HTTP 尾部中的释放也算了进去；用改正后的检查它通过（离线复核）。`full` 的 Terminal-Bench dry run（6.7 节）释放过空闲 session 的 Page，但没有重新载入。

### 4.14 Gemma 4 31B IT（第一阶段，未经 GPU 验证）

本版本包含第二个模型 profile 的第一阶段：[google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)，revision `842da3794eaa0b77d5f08bae87a17459d91ff475`。它可以安装；其 engine 和 worker（包括滑窗层的生命周期）已在 CPU 上做过端到端测试，启动检查另有单独测试；`serve.py --start` 可以启动它。**GPU 上的运行**（作业 3397695，runtime `e0df9d5`，一张 GH200，离线核验）：它的两个池的布局在关闭选择性 runtime 的情况下提供服务（经本构建的普通服务）：16 token 的 Page 启动成功（两个池分别为 15,867 和 2,886 个 block，KV 26.41 GiB），32 token 的 Page 也启动成功。与作业 3364469 的 stock vLLM 回放（10 个 trial）相比，32 token 的 Page 在 10 个 trial 中 10 个 token 级一致；16 token 的 Page 无论开不开 CUDA graph（eager）都从第 3 轮起出现分歧（第一个差异在第 3 到 5 个 token；三个服务在前 3 轮一致），这是 TRITON_ATTN 窗口布局的数值或分块差异。选页、kv-session 和 chat-text 协议以及 MemoryAgentBench 路径都还没有在 GPU 上运行过（作业已排队）。依赖已加载模型的启动检查（attention 后端的 cache 形状、final norm、视觉塔的参数名）在不符时直接拒绝启动。Qwen release 不受影响：其 vLLM 命令、启动配置、pin 和证据与上一版本逐字节相同（identity probe 共 555 + 219 项，0 处差异）。该 profile 也以 Gemma 的轮次格式服务 MemoryAgentBench 的 kv-session 协议（见下文“MemoryAgentBench”和 6.6 节），只在 CPU 上测试过。

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
| 一次 native 前向 | 8,192 token（`--max-num-batched-tokens 8192`）；更长的 phase 1 分成每次最多 4,096 token 的前向（见下） |
| 上下文 | 65,536、131,072 或 262,144 token；262,144 在选择性规则下（见下）或 32 GiB 下单个 session 时放得下：严格规则和释放规则要求全局池放下一个上下文上限处的 session |
| `--kv-cache-gib` | 24、28 或 32 |
| 协议 | 使用 Gemma 轮次格式的 Mini-SWE-Agent XML，`mswea_xml_gemma4_v1`（见下文），使用 Gemma 格式的 kv-session 协议，`--protocol kv-session --kv-session-thinking off`（见下文“MemoryAgentBench”），或使用 Gemma 轮次格式的 chat-text 协议（`--protocol chat-text`，见下文） |
| parser | `gemma4` reasoning 和 tool-call parser，`--chat-template-content-format openai` |
| 主机内存 | 一张 GH200 占节点的份额（`--mem=113560M`）；两份配额的作业（`--mem=227120M`）用 `--host-memory-shares 2`（见下） |
| 拒绝 | `--gdn-device-handoff`、`--phase1-in-step`、`--decode-fastpath3`（仅 Qwen）、`--kv-session-thinking on`，以及 `--goal-rewriter-batch`（它批量发送 `goal_query` 的 rewriter 调用；尚未与窗口布局一起检查） |

协议即 6.1 节的协议，改用 Gemma 的轮次格式：
- system 和 task 以 OpenAI text parts 经固定模板渲染，与 vLLM 交给该模板的形式相同（比纯字符串多一个空格 token）。
- user 轮为 `<|turn>user\n{content.strip()}<turn|>\n`；生成头为 `<|turn>model\n<|channel>thought\n<channel|>`，即模板在关闭 thinking 时的生成提示。
- 历史中每个回复前都保留空的 thought channel（只追加；Gemma 自己的模板在重新渲染对话时会去掉它，每轮 4 个 token），重建时也在它之后重新编码过去的回复。
- 停止 id 为 0、1、50 和 106（pad 加上 `generation_config.json` 的结束 id，其中包括 model 轮的结束）。

**Chat-text 协议（Terminal-Bench 2.0 与 Terminus 2；仅 CPU）。** `--protocol chat-text`（6.7 节）可在 Gemma release 上以 Gemma 的轮次格式运行（`remote_ra_live_v2/chat_text_protocol.py` 中的 `GEMMA4_CHAT`）。启动配置固定 `protocol` 为 `chat_text_v1`，并带 `model_profile`；其 `chat_text` 记录写入 Gemma 的生成头和模板参数（`header_text` 为 `<|turn>model\n<|channel>thought\n<channel|>`，`template_kwargs` 为 `{"enable_thinking": false}`），服务在启动时用固定模板检查一次下面这些片段。Qwen 的记录和轮次格式不变。
- 新的或重建的 session 把初始段（如有 system 消息则包括它，再加第一条 user 消息）以 OpenAI text parts、关闭 thinking 交给固定模板渲染：以 `<bos>` 开头，system 消息多一个空格，与 stock vLLM 的渲染相同。之后的每条消息都按续接时的 token 流写入：user 消息为 `<|turn>user\n{content.strip()}<turn|>\n`，连续的 assistant 消息合为一个 model 轮（模板也会合并），放在空的 thought channel 之后：`<|turn>model\n<|channel>thought\n<channel|>` + 每条内容经模板 `strip_thinking` 后的文本 + `<turn|>\n`。最后接生成头。
- 续接的请求追加回复生成时的 id、它的边界（停止 id 之后为 `\n`；达到上限被截断的回复之后为 `<turn|>\n`）和新的 user 消息，再接生成头。停止 id 为 0、1、50 和 106；遇到停止 id 时 `finish_reason` 为 `stop`，达到上限时为 `length`；回复解码时跳过特殊 token。
- 每个 prompt 等于模板对其消息的渲染、并在每个过去的 `<|turn>model\n` 之后保留空的 thought channel；去掉这些 channel（`<|channel>thought\n<channel|>`，每条过去的回复 4 个 token）后，只要每条回复重新编码仍是它自己的 id，就逐 token 等于 stock vLLM 的 prompt。用固定 tokenizer 时，chat-text transcript 也逐个复现了 stock vLLM 在第 0 阶段 session（作业 3364469）中服务的 10 个 prompt。因此使用 vLLM 默认模板参数的 stock Gemma 对照（Gemma 的模板默认关闭 thinking）重新渲染每条过去的回复时少 4 个 token，与 Qwen 的 stock 对照相同（6.7 节）。
- session、换绑、aside、关闭、上下文上限的拒绝和回复上限都与 6.7 节相同；`GET /v1/chat-text/info` 增加 `model_profile`。最多同时打开 24 个 session（K，即 pool 1 的窗口槽位数）；拒绝 `goal_query`（6.7 节）。
- Terminal-Bench 设置下的容量（K = 20 对应 Harbor 的 16 个 trial，65,536 token，4,096 token 回复，P = 16）：pool 1 容纳 20 个窗口和一次前向的瞬态块，共 9,061 个块 id（22.12 GiB）。`--kv-cache-gib 28` 剩 4,815 个全局 Page：开 `--release-idle-pages`、8,192 token 预算时，`observation_query` 和 `random_fill` 需要 2,630 个（选择性规则）；`full` 需要 4,933 个（释放规则），因此 28 GiB 最多容纳 19 个 `full` session，32 GiB 容纳 24 个。不开 `--release-idle-pages` 时，严格规则在 K = 20 下需要 4,914 个 Page，空闲 session 的 Page 也会占满 pool 0，所以要开这个选项。按上文的估计，K = 20 时 32 GiB 留给 GH200 的余量不足 2 GiB（峰值约 96,100 MiB，按 Qwen 运行时的开销为 97,400 MiB）；28 GiB 的峰值约 92,000 MiB（93,300）。主机内存：`--dram-arena-gib 64` 可容纳 52,428 个 1.25 MiB 的 Page（838,848 token；Qwen 的 64 GiB 容纳 1,048,576 token），加上 1 GiB connector 池共锁页 65.0 GiB；不设 arena 时 20 个 65,536 token 的 session 可能锁页 101.1 GiB，启动器会拒绝。

Terminal-Bench 的 dry run：

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
- 加 `--release-idle-pages` 而没有固定预算时（`full`、`random50` 或 `--active-ratio`），启动器按释放规则确定全局池大小：一个 session 在上下文上限处最坏的一轮、其他 session 的 tail 和 1 % 水位线。GPU 紧张时空闲 session 释放全部 Page，并在下一次 phase 1 之前重新载入；对 `full` 来说就是该 session 的全部历史。131,072 token、4096 token 回复时，16 个 `full` session 需要 30.02 GiB（精确为 30,738 MiB；9,083 个 Page，32 GiB 有 10,692 个）；24 和 28 GiB 分别最多容纳 8 和 13 个（65,536 token 时 28 GiB 可容纳 19 个）。`full` 与 Qwen 一样用 `--kv-cache-gib 32`（`serve.py` 的 `GEMMA4_FULL_KV_CACHE_GIB`）：比规则多 1,608 个 Page（1.96 GiB），更小的大小被拒绝时提示 32。`full` 从不启动 Goal rewriter：它没有选择契约，只有 `goal_query` 会启动。GPU 显存（估计，未实测）：57.9 GiB 权重加 32 GiB KV 为 92,068 MiB，GH200 共 97,871 MiB。按 stock Gemma 服务的开销（26.41 GiB KV 时 READY 87,601 MiB、峰值 90,099 MiB：READY 时权重和 KV 之外 1,257 MiB，峰值再多 2,498 MiB）加上运行时自身约 300 MiB（phase-1 行、重新载入的中转），READY 约 93,300 MiB，峰值约 96,100 MiB，距总量 1.7 GiB；按 Qwen 运行时在 32 GiB 时的开销（2,215 和 3,092 MiB），峰值为 97,375 MiB，只剩 0.5 GiB。由显存探测决定：若峰值剩余不到约 2 GiB，`full` 退回 30.5 GiB（比规则多 391 个 Page；启动器目前只接受 24、28、32，需要改启动器）。重新载入的数据量按 Qwen swe16 的 `full` 运行（job 3357655，32 个任务，同时打开 16 个，历史按 Gemma token 换算，每轮视为瞬时、按最久未用释放）回放：32 GiB 下 18 % 的轮次需要重新载入，平均 1.5 GiB（中位数 1.45 GiB，p90 2.65 GiB，最多 4.1 GiB 主机到设备拷贝），平均到所有轮次为每轮 278 MiB；28 GiB 下为 36 % 和 546 MiB，但只有上下文低于 104,880 token 时才容纳得下 16 个 `full` session。
- vLLM 自己的启动检查要求全局池在滑窗池之外再放下一个 `--max-model-len` 长的请求：131,072 token 下 16 个窗口需要 28.95 GiB（`goal_query` 为 30.51 GiB），262,144 token 下需要 38.95 GiB。因此在本版本之前，尽管启动器的选择性规则接受，它也会在启动时拒绝 32 GiB 以下、131,072 token 的 16 个 session（还没有 16 session 的 Gemma 服务在 GPU 上启动过），并在任何大小下拒绝 262,144 token 的 16 个 session。现在在选择性规则下，窗口布局固定该规则的数值 `--additional-config` 中的 `"request_pages": R`（`kv_layout.WindowLayout.request_pages`、`window_capacity.pinned_request_pages`，由 `remote_ra_http/config.py` 检查），vLLM 的检查改为要求全局池为一个请求放下 R 个全局 Page（`vllm/v1/core/kv_cache_utils.py`）。两个池、gate 和各种拒绝都不变；严格规则和释放规则仍用 vLLM 自己的检查。
- 在选择性规则下，第一轮或重建需要整行。若即使释放所有空闲 session、等所有运行中的请求结束也放不下，该轮会以 `CAPACITY_EXCEEDED` 被拒绝（它不是 `CAPACITY_WAIT`，客户端不会重试），而不是无限等待。
- 释放的频率取决于池的大小：8,192 token 预算下，16 个空闲的 SWE-bench session 约占 16 × (512 + 上一轮的 Page)，超过 24 GiB 的 4,138 个 Page，所以在 24 GiB 下 admission 会释放并重新载入 Page（每次重新载入几百 MiB 的主机到设备拷贝）；28 GiB 有 7,415 个 Page，32 GiB 有 10,692 个。
- engine 的 admission gate 检查两个池。全局 Page 沿用 Qwen 的规则；滑窗池中，一轮在 phase 1 需要它的新窗口块，在 phase 2 需要把接管的窗口补到其峰值的块，而每个运行中的请求仍可能增长到它自己的峰值。重建的滑窗需求扣除它的渲染所释放的窗口，与全局需求扣除它的 Page 和 tail 一样（本版本修复：窗口槽全部占满时，重建看起来缺 pool 1，缺的正是它自己的窗口，于是不为它的全局缺口释放任何空闲 session；在没有请求运行时被放行后，它会永远等待全局 Page）。与 Qwen 相同（4.4 节），没有请求运行（也没有admission 在途）时，只要另一个已打开的 session 钉住了全局 Page、tail 或窗口，就检查这一轮；在 `--release-idle-pages` 能做的释放之后仍放不下时，它等待（`CAPACITY_WAIT`）。integrate-1018 之前，窗口布局不加检查地放行这样的一轮，它的解码可能找不到 block（vLLM 会抢占，这在这里是致命的）或永远等待。没有其他 session 钉住任何东西时，照旧不加检查地放行。
- 主机内存：启动器按一张 GH200 占节点的份额（`--mem=113560M`）、进程占 30 GiB 计算，因此 connector 池加 Page arena 不得超过 80.9 GiB。OffloadingConnector 的 CPU 池在本 runtime 中从不使用，所以 Gemma 服务接受 `--connector-pool-gib N`（4.9 节），不加该选项时用 1 GiB；该池总是被固定（`cpu_bytes_to_use`，`startup.json` 中的 `connector_pool_bytes`）并计入预算。131,072 token 时，不加 `--dram-arena-gib` 最多 7 个 session；arena 最大为 79 GiB，12 个 session 时平均每个约 86K token。16 个 session 使用 `--dram-arena-gib 79`（见下）。按 Qwen swe16 运行（SWE-bench Verified 前 32 个实例，同时打开 16 个，五个策略）回放，并用 Gemma 的 tokenizer 重新计算每段历史（比 Qwen 多 3.7–16.4 % 的 token），16 个打开的 session 的 Page 峰值为 26.5–37.75 GiB，最长的 16 段历史合计最多 65.25 GiB；单段最长历史为 96,336 token。因此一份配额（`--mem=113560M`）足够，不需要两份。`--host-memory-shares 2`（仅 Gemma profile；Qwen 保持 128 GiB 的预算）让启动器按两份配额的作业（`--mem=227120M`）计算：可锁页 191.8 GiB，262,144 token 时 arena 最大 190 GiB；MemoryAgentBench 最长的 16 个 case 需要它（见下）。

**分块的 phase 1。** 超过一次 native 前向（8,192 token；Mini-SWE-Agent 的 10,000 字符上限下，数字密集的输出约为 10.2K token）的 phase 1 在同一个 lease 上分成每次最多 4,096 token 的前向运行（`model_profile.GEMMA4_PHASE1_CHUNK_TOKENS`）；不超过 8,192 token 的 phase 1 仍是一次前向。
- 全局行从第一次前向起就包含整个 run，只有第一次前向把新的 Page 清零。每次前向的滑窗行包含它读取的窗口和它的新 token；token 已离开窗口的新滑窗块会被后面的 token 复用，因此 phase 1 每个滑窗组最多占 ⌈(1,023 + 4,096) / 16⌉ + 1 = 321 个新块，在一次前向的瞬态（P = 16 时 512 个）之内：pool 1 不变。lease 中列出后面的前向（`chunks`），worker 检查它们（铺满整个 run、延续窗口、从不复用借来的窗口或仍会被读取的块）并依次运行：第一次前向之前检查借来的窗口，最后一次之后捕获 phase 2 起始的窗口。Page 向量和观测 query 都是整个 run 的。
- 启动器的规则假定 run 只有一次前向。更长的 run 由池的余量承担（24 GiB、16 个 session 时，`goal_query` 余 251 个全局 Page，`observation_query` 余 1,518 个，约合 4,000 和 24,000 token 的 run）；即使释放所有空闲 session 也放不下时，以 `CAPACITY_EXCEEDED` 拒绝，不会无限等待。
- 计算的切分方式与一次 10K token 的前向不同（stock vLLM 在 8,192 token 处切分这样的 prefill），因此这种轮次的输出可能与 stock vLLM 在最后几位上不同；尚未测量。

**`goal_query`。**
- Goal 即第 5 节的 Goal，没有改动：它的维护逻辑、rewriter 及其 prompt、gate 和 Goal 文本。engine 从 Gemma 的 transcript 构造每轮的 Goal 上下文，方式与 Qwen 相同：根目标（task 消息中的 `<pr_description>`）、该 session 上一轮解码出的回复（空 thought channel 之后生成的文本，不含 `<turn|>`，即客户端回传的内容）、最新的观测及其 `<returncode>`。
- Goal 前向是对 `<retrieval_query>` 文本单独做的一次冷前向，用 Gemma 的 tokenizer 编码，前面加 `<bos>`（id 2），与 Gemma 的每个序列一样开头（PoC 对 Qwen 的 Goal 不加特殊 token；Qwen 没有 BOS）。它的 query 向量与观测 query 相同：Goal 最后一个 token 在第 59 层 final norm 之前的隐状态，用 5,376 维的 pin 投影，与 Page 向量在同一空间。
- 它在 phase 1 之后运行，此时 phase 1 新分配的滑窗块仍被占用，所以 pool 1 为它单独预留瞬态块：Goal 最多 G = 2,048 token 时每个滑窗组 ⌈G / P⌉ 个块（`model_profile.GEMMA4_GOAL_TOKENS`；Qwen 在 16 session SWE-bench 运行 job 3363871 中的 Goal 前向最多 1,164 token）。更长的 Goal 会在分配任何块之前以 `PHASE1_TOO_LONG` 使该轮失败。选页生效的轮次，gate 为 Goal 前向计入 ⌈G / P⌉ 个全局 Page 和每组同样多的滑窗块。
- 内存：640 个块 id，1,600 MiB，两种 Page 大小都一样。16 个 session、131,072 token、8,192 token 预算、4,096 token 回复时，全局池在 24 GiB 下剩 2,858 个 Page（`observation_query` 为 4,138），28 GiB 下剩 6,135 个（7,415），选择性规则分别需要 2,606 和 2,639：16 个 `goal_query` session 需要 23.70 GiB（不含 Goal 时 22.13），24、28 和 32 GiB 分别最多容纳 16、21 和 26 个。若 Goal 瞬态按一整次 8,192 token 前向预留（6.25 GiB），24 和 28 GiB 都放不下 16 个 session。Goal rewriter（Qwen3-0.6B，约 2 GiB）与模型和 KV 池共用 GPU，Gemma 下尚未实测。
- 接受 `--admission-overlap K`（4.10 节），启动记录与 Qwen 相同（`--phase1-in-step` 和 `--gdn-device-handoff` 仍被拒绝）。全局池的预留与 Qwen 相同。滑窗池中，每个进行中的 admission 预留它的这一轮被接受后仍占用的块，而 phase 1 的瞬态块（新分配的块、Goal 前向）只预留一次，即进行中最大的那份：phase 1 一次只有一个 lease，且在 engine 线程的一次调用内完成（admission 只在 phase 1 之前停在 Goal 处，之后不再停）。否则每个 admission 的 phase 1 峰值都要占一次前向的 pool 1，观测为 2,600–4,300 token 时进行中的 admission 只能有一到两个。在 CPU fake 上（16 个 session），16 个第 1 轮 admission 全部同时停住，而每个 admission 各预留一份 phase 1 峰值时只有 6 个。24 GiB 下进行中的 admission 数量仍受全局池限制：每个续接轮次都要预留它的行和恢复的 Page，因此会为它们释放空闲 session 的 Page（4.13 节）。16 个 session 时：`goal_query --admission-overlap 16 --release-idle-pages --active-token-budget 8192`，24 或 28 GiB。

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

**服务配置：CUDA graph 与解码快速路径（未经 GPU 验证）。** Gemma 服务接受 `--cuda-graphs piecewise`、`full` 或 `full-decode`、`--decode-fastpath` 和 `--decode-fastpath2`，其 pin 和启动检查与 4.8、4.11 节相同；不加这些选项时，其 vLLM 命令、启动配置和证据不变。三者合起来就是与 Qwen 相同的服务配置：`--cuda-graphs full --decode-fastpath --decode-fastpath2`。它们可以与每个策略、`--admission-overlap K` 和 `--release-idle-pages` 一起使用（CPU：`test_gemma_flags`）。
- **CUDA graph。** 全局组是 window 布局中唯一的 FullAttentionSpec 组（vLLM 的 SlidingWindowSpec 不是 FullAttentionSpec）。使用 FULL graph 时，runner 在 capture 和 replay 时都把它绑定到物理 seq-len 缓冲区，并保留物理 slot mapping；五个滑窗组保留 vLLM 的逻辑 seq len、position 和 slot mapping。captured graph 读取的每个缓冲区都是 runner 的持久缓冲区，每次 replay 之前就地重写（CPU 测试：overlay runner 在六个组上构建 metadata 时，每个组在 capture 和 replay 时用的是同一组缓冲区）。编译模式要求 attention 和 KV-cache 更新是 splitting op（没有 GDN op）。phase 1、它的各个分块和 Goal 前向以 eager 方式运行。16 个 session 时 capture size 为 1、2、4、8、16、24 和 32。torch.compile（`piecewise`、`full`）与 Qwen 一样会改变 bf16 数值：应与同一模式、`TRITON_ATTN` 的 stock vLLM 比较。
- **`--decode-fastpath`。** Qwen 的快速路径加上 window owner 的增量观测：当完整观测会接受相同的行时，一步在上一步的观测上增量扩展。full-attention 的 lease 按新追加的块增长，每个滑窗组的 lease 按进入窗口的块和离开窗口的块变化，claim 和 allocation 集合也只按这些差异更新；窗口没有移动时，其 lease 既不重建也不重新检查。
- **`--decode-fastpath2`。** Qwen 的每步一个 `allocation_batch` RPC、把前向之后的工作放到 logits 之后，加上增量的滑窗行（`remote_ra_live_v2/window_rows.py`）：engine 逐步跟踪每个滑窗窗口，而不是每步重建整行；有变化的请求，其更新对每个 KV 组只携带窗口的起始下标、行长度和新追加的（block, incarnation）对，没有变化的请求不带行；worker 以紧凑形式保存滑窗行。131,072 token 时，一次有变化的更新原本约携带 41K 对（五个各 8,192 项的滑窗行加全局行），现在只携带新追加的那几对。请求的第一次更新仍是整行。
- **vLLM runner 的行。** vLLM 的 runner 只会追加请求的新 block id；EngineCore 只在它自己的行中把离开窗口的块置为 null。因此 worker 只在 live 窗口（包含该步的全部读取）上检查 runner 的滑窗行是否等于 EngineCore 的行，而不是比较整行。第一阶段的检查比较整行，在真实 runner 上第一次窗口滑动时就会拒绝；CPU fake 现在按 vLLM 0.24 runner 的方式保存行（以及 numpy block table）。

16 个 session 使用该服务配置的 dry run：

```sh
python -B serve.py --release-dir "$RELEASE-gemma4" --task /abs/path/to/tasks-16.json \
  --policy observation_query --active-token-budget 8192 --release-idle-pages --page-tokens 16 \
  --kv-cache-gib 24 --dram-arena-gib 79 --max-model-len 131072 --max-tokens 4096 \
  --cuda-graphs full --decode-fastpath --decode-fastpath2 \
  --language-model-only --output "$RUNS/gemma-obs16-fast" --port 8000
```

**使用该服务配置之前要在 GPU 上检查的内容：** window 布局在 `TRITON_ATTN` 上的启动和 capture（六个组的 FULL decode graph、worker 的 graph 记录、在 24 GiB KV 旁边的 capture 时间和显存）；在 stage-0 append 会话上，`full-decode` 与 eager、`full` 与 `full` 模式的 stock vLLM 的 token 一致性；131,072 token 下 `full` 和 `observation_query` 各一个 session，服务配置与 eager 服务对比（只开快速路径时必须与 eager 的 token 一致；graph 可能改变 token，见 4.8 节）；在 1 个和 16 个运行中请求时的 inter-token latency，与 Triton 上的 stock vLLM（stage 0 中为 20.6 和 27.0 ms）对比，并开启 step profile。

**MemoryAgentBench：kv-session 协议（gemma-mab，未经 GPU 验证）。** `--protocol kv-session`（6.6 节）在 Gemma release 上使用 Gemma 的轮次格式；启动记录 `kv_session` 多一个 `"grammar": "gemma4"`，`/v1/kv-session/info` 在 options 中给出它。线上格式、slot、上下文上限的拒绝和证据都与 6.6 节相同。
- **只支持关闭 thinking**（与 MemoryAgentBench 的决定一致；`--kv-session-thinking on` 被拒绝）。user 轮之后的生成头为 `<|turn>model\n<|channel>thought\n<channel|>`，7 个 token，启动时与固定模板对照检查一次（`kv_session_protocol.check_gemma4_header`、`gemma4_renderer.Gemma4SourceRenderer.kv_transcripts`）。
- **没有生成头的续接。** 在 tool response 之后，Gemma 的模板继续同一个 model 轮，不加生成头（thinking 关闭时），所以客户端发送 `header_tokens` 0。这样的 tail 以 `<tool_response|>` 结尾，phase 2 只计算这一个 token（e = N − 1）；服务端会检查它。MemAgent 的读取轮就是这种轮次；user 消息之后的轮次（格式错误提示、问题）带生成头。
- **边界与停止。** 停止 id 为 0、1、50 和 106。新消息从 `<|turn>` 开始（`input_ids[boundary_tokens]`，在文本回复的 `<turn|>` 和 `\n` 之后：边界为 2），或紧接在 `<|tool_response>` 之后（`input_ids[boundary_tokens − 1]`）：带 tool call 的回复停在 `<|tool_response>`（id 50）上，它同时也是承载其 response 的 tail 的第一个 token，所以边界为 1。当上一回复的最后一个采样不是 tail 的第一个 token 时（`<eos>`、pad 或长度停止，或停在 `<turn|>` 上的 tool call），客户端把它补在前面（+1），与 Qwen 客户端对 `<|im_end|>` 以外所有 id 的处理相同。
- **客户端的 tail。** Gemma 的模板先渲染回复的 tool call，再渲染 tool response，最后才是回复的文本；并且回复不带 tool call 时，后面的 tool 消息会被丢掉。因此 Qwen 的 marker（把最新回复换成文本 marker）找不到 tool-call 回复的 tail。重新实现的客户端（`test_gemma_kv_session.GemmaKvClient`，也是 MemoryAgentBench harness runner 的参照）改为渲染一个替身：回复的 tool call、不带文本；后面没有消息时模板以 `<|tool_response>` 收尾，tail 从这个 token 开始。不带 tool call 的回复沿用 Qwen 的 marker。
- **逐 token 一致**（固定 tokenizer 和模板，`test_gemma_kv_session`）：MemAgent 的首个 prompt 为 219 个 token（前缀 212 + 生成头 7），每个服务的 prompt 都与模板对其消息的渲染逐 token 相同，只是每个 model 轮在 `<|turn>model\n` 之后保留了空的 thought channel（回复所跟的生成头；Gemma 的模板重新渲染历史时会去掉它，每个 model 轮 4 个 token）。因此每轮重新渲染整个对话的 stock vLLM 服务（Vanilla 传输）从第一个 `<|turn>model\n` 之后、前缀 + 3 处开始不同；对服务过的 prompt（token id）做 stock 重放才是精确对照。
- **内存，16 个 session、32,768 token 预算、4,096 token 回复、P = 16**（`--release-idle-pages`，选择性规则）：滑窗池占 18.95 GiB（16 个 812.5 MiB 的窗口共 12.70 GiB，加一次 8,192 token 前向的瞬态 6.25 GiB；`goal_query` 另加 1.56 GiB 的 Goal 瞬态），全局池需要 5,724 个 Page，7.0 GiB（一次选择性轮次 2 × (2,048 + 769) + 1 = 5,635、其他 session 的 tail 和水位线；`goal_query` 为 5,711）：合计 25.94 GiB，带 Goal 时 27.48 GiB。28 GiB 放得下（比规则多 1,690 个 Page，`goal_query` 多 423 个），24 GiB 放不下；不需要把空闲 session 的窗口放到主机（设计中的 WP11：每个窗口 812.5 MiB，未实现）。GPU 紧张时空闲 session 释放其全局 Page，并在下一次 phase 1 之前重新载入；窗口一直留在滑窗池中。一次读两个 chunk（约 8,300 token）以分块 phase 1 从余量中运行。GPU 显存：57.9 GiB 权重加 28 GiB KV，按上文估计的开销，峰值约 92,000–93,300 MiB，比 GH200 的 97,871 MiB 低 4.5–5.8 GiB（带 `goal_query` 约 2 GiB 的 rewriter 时为 2.5–3.7 GiB）；32 GiB 时只剩 0.5–1.7 GiB。
- **上下文。** MemAgent32 的 32 个 case 在 Gemma 下为 39,336–113,785 token（首个 prompt、每个 chunk 及其 tool response 包装和问题；是 Qwen token 数的 0.97–1.0 倍）：131,072 token 下最长的 case 只剩约 17,300 token 给 28 次读取和答案；选择性规则允许的 262,144 对较长的回复也留有余地（Qwen 的计划用 262,144）。锁页 arena 保存每个打开的 session 的每个已封存 Page：仅最长的 16 个 case 就有 150 万 token，回复之前已占 114.5 GiB 的全局 Page，超过一份配额 80.9 GiB 的锁页预算，所以 16 个这样的 session 需要 `--host-memory-shares 2` 和两份配额的作业。

```sh
python -B serve.py --release-dir "$RELEASE-gemma4" --protocol kv-session --kv-session-thinking off \
  --policy observation_query --active-token-budget 32768 --page-tokens 16 --kv-cache-gib 28 \
  --max-running-sessions 16 --release-idle-pages --max-tokens 4096 --max-model-len 262144 \
  --dram-arena-gib 150 --host-memory-shares 2 --connector-pool-gib 1 --language-model-only \
  --output "$RUNS/gemma-mab-obs" --port 8000
```

**CPU 测试**（`test_gemma_*.py`）。若 `REMOTE_RA_GEMMA4_METADATA` 指向包含五个固定元数据文件的目录，渲染测试会使用真实 tokenizer（需要 Transformers 5.13 和 Tokenizers 0.22.2，与 2.1 节相同）：transcript 在全部 10 轮（最长 62,919 token）上逐 token 重现 stock vLLM 0.24.0 Gemma 会话的 prompt id。不设置该变量时这些测试会被跳过。`test_gemma_e2e` 用 vLLM 自己的 KV manager（一个全局组、五个滑窗组、64 token 的窗口）对每个策略跑三个 session 的 engine 和 worker：每次前向在每个滑窗组中读到的恰好是最近 63 个 token，并且 gate 的数值是每轮从滑窗池实际取走的块数的上界。`test_gemma_release` 让 16 个 session 一起运行，全局池取选择性规则的最小值：每一轮读到的内容与一个从不释放的池完全相同，被释放的只有空闲 session（最久未用的优先），重新载入恰好恢复被释放的 Page 及其 KV，释放不触碰窗口和滑窗池，而一个永远放不下的重建会以 `CAPACITY_EXCEEDED` 被拒绝。`test_gemma_goal` 覆盖 `goal_query`：Goal 瞬态的布局、pin 和启动器数值、gate，以及 16 个带释放的 `goal_query` session（每一轮与从不释放的池完全相同，每次 Goal 前向都在它的瞬态块内，其 query 向量是 final norm 之前的最后一行、位于 Page 向量的投影空间；没有该瞬态时同样的池会失败）。`test_gemma_chunked` 覆盖分块的 phase 1：算术和 gate 的数值、lease 检查，以及三个 session 上 `full`、`observation_query` 和 `goal_query` 的最长约 2,600 token 的观测，按每次 256 token 前向运行（一次 native 前向 1,024），每一轮都与一次前向的结果相同，每次滑窗读取都精确。`test_gemma_full` 覆盖释放规则下 16 个 session 的 `full`（每一轮与从不释放的池完全相同；重新载入恰好带回每个已封存的 Page 及其 KV；不加该选项时同样的池会失败）。`test_gemma_overlap` 覆盖 `--admission-overlap`：启动记录、滑窗 ledger，以及 16 个 `goal_query` session（rewriter 用替身，Goal 停在各 session 自己的线程上），有无释放时的决策都与不加该选项的服务完全相同。使用真实 tokenizer 时，`test_gemma_renderer` 检查 Goal 从 Gemma transcript 得到的输入及其 Goal probe 的 token。`test_gemma_graphs` 覆盖 CUDA graph 的 pin 和检查、overlay runner 在 Gemma 六个组上的缓冲区绑定，以及挂上 graph 的端到端测试（只有 probe 前向走 eager）；`test_gemma_fastpath` 在关闭、开启一个和开启两个快速路径时运行 window 端到端测试（把每个 batch 条目展开后 RPC 完全相同，证据文件、frame、关注的 Page、回复、滑窗读取、每步的 lease 和 claim 也完全相同）；`test_gemma_window_rows` 在带有篡改的随机步骤序列上比较增量观测与完整观测，以及增量行。`test_gemma_flags` 把这些选项放在一起运行：240 次启动（`full`、带与不带 rewriter 的 `goal_query`、`observation_query` 和 `random_fill`，各自配合 `--cuda-graphs`、第一条或两条解码快速路径、`--release-idle-pages` 和 `--admission-overlap 4` 的每一种组合，16 个 session、131,072 token）每次都只多出自己的启动键和 vLLM 参数，启动检查和 window worker 全部接受；单独的 `--decode-fastpath2`、`--decode-fastpath3`、`--goal-rewriter-batch`、`--phase1-in-step`、`--gdn-device-handoff`、开 thinking 的 kv-session 协议和带 `--task` 列表的 chat-text 协议无论其他选项如何都被拒绝。在 CPU 替身上，`--cuda-graphs full` 加两条快速路径的 `goal_query`（三个 session 一起解码；只有 phase 1 和 Goal 前向走 eager）、带两条快速路径的分块 phase 1、在 8 个 `goal_query` session 上带两条快速路径的 `--admission-overlap 8 --release-idle-pages`，以及带两条快速路径的 `full` 和 `observation_query` 释放规则压力运行，每一轮都与不开这些选项的运行相同，释放和重新载入也相同。默认测试集在 6 或 8 个 session 上运行这些端到端测试和 `test_gemma_release` 的测试（`goal_query`、`full`、`observation_query`、`random_fill`、overlap），并对 `goal_query` 运行分块的 phase 1；设 `REMOTE_RA_SLOW_TESTS=1` 时还运行 16 个 session、每个 5 轮、与从不释放的池对比的版本（上文 16 个第 1 轮 admission 全部停住的结果），`full` 和 `observation_query` 的分块 phase 1，以及每个策略和模式的 graph 与快速路径运行，约多 6 分钟。`test_gemma_goal_rewriter` 检查该 profile 拒绝 `--goal-rewriter-batch`，且其服务不写 Goal rewriter 的任何用时字段。`test_gemma_kv_session` 覆盖 kv-session 的 Gemma 格式：重新实现的客户端、transcript 的几何与拒绝、启动记录和生成头检查、启动器（262,144 token 的 16 个 session、各种拒绝、`--host-memory-shares`），以及带 `request_pages` 的 vLLM 启动检查；使用固定 tokenizer 时，MemAgent 的每种轮次（读取、一次读两个 chunk、无效调用、格式错误、问题、补回的采样）都与模板逐 token 一致。`test_gemma_kv_session_e2e` 让该客户端的 MemAgent session 经窗口布局的 engine 和 worker 运行，开 `--release-idle-pages`、全局池取选择性规则的最小值（默认测试集中 `observation_query` 和 `goal_query` 各 6 个 session；设 `REMOTE_RA_SLOW_TESTS=1` 时 16 个，并且每个回复都与从不释放的池相同，约 1.5 分钟）：每个 prompt 都是客户端的构造，Gemma 的几何成立，释放和重新载入不动窗口；另有三个 session 经打包的路由和带 `Gemma4SourceRenderer` 的前端 capture 运行。`test_gemma_chat_text` 覆盖 chat-text 的轮次格式（gemma-tb）：启动记录和模板检查；Terminus transcript 的每个 prompt 都等于参照渲染，去掉 thought channel 后等于 stock vLLM 的 prompt（用固定 tokenizer 和模板，也用替身）；chat-text transcript 复现第 0 阶段的 stock session；以及 Terminal-Bench 设置下的启动器和启动检查。`test_gemma_chat_text_e2e` 让 Gemma chat-text session 经打包的 router 和 chat-text 前端，在窗口布局的引擎和 worker 替身上运行，并开 `--release-idle-pages`：K = 20 时 16 个 session 一起解码，16 个 session 在选择性规则最小的全局池上运行（释放并重新载入空闲 session），摘要的 aside 和交接重建，回复上限和上下文拒绝，取消及其换绑，以及把窗口还给 pool 1 的关闭和驱逐；`REMOTE_RA_SLOW_TESTS=1` 再加 16 个 session × 8 个 episode 在释放压力下的摘要运行、压力下的 `random_fill` 和 `full`，以及与从不释放的池的比较。`test_gemma_chat_text_replay`（需显式开启，`REMOTE_RA_TB_ATIF_DIR`）用 Gemma 的轮次格式回放 8 月的 Terminus 2 轨迹（6.7 节）。

```sh
REMOTE_RA_GEMMA4_METADATA=/abs/path/to/gemma-4-31b-it \
  "$VENV/bin/python" -B -m unittest test_gemma_profile test_gemma_renderer test_gemma_kv_pools \
  test_gemma_selection test_gemma_install test_gemma_capacity test_gemma_startup \
  test_gemma_worker test_gemma_engine test_gemma_e2e test_gemma_release test_gemma_goal test_gemma_overlap \
  test_gemma_full test_gemma_chunked test_gemma_goal_rewriter test_gemma_graphs test_gemma_fastpath \
  test_gemma_window_rows test_gemma_flags test_gemma_kv_session test_gemma_kv_session_e2e \
  test_gemma_chat_text test_gemma_chat_text_e2e
```

**尚未完成：** 每行 `runtime.jsonl` 中的窗口记录、session 池下放到主机的窗口（28 GiB 下 MemoryAgentBench 不需要，见上），以及全部 GPU 验证。

## 5. query 策略的选页规则

`goal_query` 和 `observation_query` 执行 dram-backed-kv PoC（`scripts/serve_online_selector_v2.py`，上游 commit `a70701e`）的 Online Selector V2 合同。页大小 P 是这个合同的参数。默认 P = 784 时，vLLM 以 784 token 的 Page 恢复全注意力 KV，所以 PoC 按 16 token 计的预算要换算成整页，每个 Page 用它最后一个 16-token 子页来代表。P = 16 时（`--page-tokens 16`，4.6 节），Page、预算和向量都是 PoC 的 16 token 版本。

会话第一个请求之后的每个请求，每个策略都按 PoC 的 V0 时序执行：

1. **阶段 1（observation）。** 新的 observation 在上一轮之后仍常驻的 Page（上一轮的激活页，加上服务上一轮时封存的页）和 live tail 页上 prefill，并接续上一轮的 Gated-DeltaNet 状态。这一步得到它所封存的 Page 的向量，以及 observation query：observation 最后一个 token 的 hidden state。
2. **Goal。** 对 `goal_query`，由上一轮回复和最新 observation 构造维护中的 Goal。其 query 文本单独做一次冷 forward，并与上游一样按 retrieval-goal hash 缓存。
3. **选页**：在 observation 之前的 Page 里挑选（见下）。
4. **阶段 2（生成）。** 生成头和 decode 在选中的 Page 加上阶段 1 留下的 observation KV 上运行。

每个 token 只计算一次；本轮封存的 Page 使用这唯一一次计算得到的向量。会话的第一个请求是一次普通 prefill。

选页规则：

1. 第 0 轮，以及已封存历史不超过 **8192 token** 时，所有 Page 都保持激活（`reason` 为 `cold_start` / `below_switch_threshold`）。
2. 否则容量为 `max(1, 必保留页数, floor(0.40 × 历史 token) // P)` 页。`--active-token-budget N` 用绝对预算代替比例；`--active-ratio R` 修改比例。
3. 始终保留**必保留**页：初始 system + 任务 prompt 触及的每一页。
4. 加入 **recent** 页：前两轮里的非必保留页，从新到旧，最多 ⌈1,024 / P⌉ 页（对应 PoC 的 64 × 16 = 1,024 token）：P = 784 时 2 页，P = 16 时 64 页。
5. 剩余容量用与 query 余弦最高的页填满，使用上游的 `PrefillPageEmbeddingSelectorV2`。

向量：取模型最终 norm 之前的 last-token 隐藏状态（即 PoC 的最后一层 decoder 输出），用 PoC 的带种子 128 维 Rademacher 矩阵投影（seed 20260904，hidden size 5120 时 SHA-256 为 `20c2e1cf…87aa`），再 L2 归一化并逐向量 int8 量化。Goal query 是维护中 Goal 的 PoC `<retrieval_query>` JSON 文本；`--goal-query-variant candidate_entity_cap8` 在这段文本里只保留最后 8 个 files、symbols 和 tests。维护中的 Goal 使用 entity-drift 3；指定 `--goal-rewriter-model` 时使用 PoC 的常驻 Qwen3-0.6B rewriter。某一轮 rewrite 失败时与上游一样回退到确定性字段，并记录下来；`goals.jsonl` 保存每个 Goal、Goal query 和 rewriter 交互的文本（第 8 节）。

`observation_query` 不使用 Goal rewriter，因为它的排序只取决于 observation 向量：服务从不为它启动 rewriter，它的 Goal 只用确定性字段构建（仅用于证据），钉住 rewriter 的启动配置会被拒绝，给出的 `--goal-rewriter-model` 会在 `startup.json` 和 dry run 中记为未使用（`goal_rewriter`）。

**随机填充对照。** `--policy random_fill` 就是只改了第 5 步的 `observation_query`：阈值、容量、必保留页和 recent 页都相同，阶段 1 和确定性 Goal（仅用于证据）也相同；不使用 Goal rewriter（给出的 `--goal-rewriter-model` 记为未使用，钉住 rewriter 的配置会被拒绝），也不做 Goal forward。去掉必保留页和 recent 页之后剩下的候选页被随机排序，取前面的页填满容量；不用 query 向量、投影或分数。排序用 SplitMix64 加 Fisher-Yates（不做任何哈希），由启动时钉住的种子（`--fill-seed`，默认 42，即选页合同的 `random_fill` 键）、session 序号和该 session 的请求序号决定。每个选页生效的轮次，`query.selection.fill` 记录种子、session 序号、请求序号（`turn`）以及可选页数和抽中页数，`retrieved_pages` 按抽取顺序列出抽中的页。如果 `goal_query` 和 `observation_query` 不比这个对照好，它们的排序就没有带来有用的信息。与 `random50` 不同，它保留 query 策略的必保留页、recent 页和预算。

与 PoC 的已知差异，以及两种布局之间的差异：
- 默认 784 token 的 Page 下，预算都是整页数，live tail 是一个未写满的 Page。16 token 的 Page 下，预算、recent 配额和 live tail（不足 16 token）都与 PoC 相同。
- 必保留页覆盖 system + 任务 prompt 碰到的每一页。PoC 只保留完全落在其中的页，在 784-token 的页上可能丢掉任务的结尾；在 16 token 的页上两者最多差一页。
- 解耦布局下，worker 在 GPU 上一批地计算 int8 Page 向量；默认布局下在 CPU 上逐个计算。在 CPU 测试负载上两条路径得到的向量完全相同，但在 5,120 维的行上约 0.35 % 的向量有一个元素相差 ±1，可能改变近乎并列的排序（`docs/exp-decouple-pages/DESIGN.md`，Integration 第 4 项）。
- `random50` 保持固定的 floor(n/2) 随机抽取。16 token 的 Page 下它的种子来自 16 token block 的哈希，所以它的抽取结果不能与 784 token 的 Page 下比较。

## 6. Harness 接口

启动器的 `--protocol` 接受 `mswea-xml`（默认值）、`kv-session` 和 `chat-text`。`mswea-xml` 逐 token 复现 dram-backed-kv PoC 服务端，是 SWE-bench 实验用的协议（6.1 到 6.5 节）。`kv-session` 是 MemoryAgentBench harness 的 token 级 session 协议（6.6 节）。`chat-text` 是普通的 OpenAI chat 文本，面向 Terminal-Bench 2.0 上 Terminus 2 这样的 agent（6.7 节）。

### 6.1 Mini-SWE-Agent XML 协议

这是 PoC 服务端的约定，Mini-SWE-Agent 2.4.2 以单个 `<mswea_bash_command>` 动作使用：

- **请求。** `POST /v1/chat/completions`，只看 `messages`：完整历史，每条是 `{"role", "content"}` 文本。依次是 system 消息、任务消息（必须包含冻结的 issue），然后是 agent 的 assistant 和 user 消息，最后一条必须是 user 消息（带 `<returncode>` 的 observation、格式错误提示等）。带 `tools` 的请求会被拒绝，消息里的其他字段会被忽略。
- **采样。** 服务端和 PoC 一样忽略客户端的采样字段：贪心解码，最多生成服务端回复上限那么多个 token（1536，即 PoC 的值；`--max-tokens`，4.9 节），解码文本里出现 `</mswea_bash_command>` 或遇到 EOS/pad 就停。动作结束标记在引擎内部检测。
- **渲染。**
  - system 和任务消息走固定的 chat template，`enable_thinking=false`。
  - 之后每条消息都是 `<|im_start|>{role}\n{content}<|im_end|>\n`。
  - 生成头是 `<|im_start|>assistant\n<think>\n\n</think>\n\n`。
  - 历史里的回复会换回生成时的原始 token ID，按 `strip()` 之后的文本匹配，所以客户端裁掉首尾空白也没关系。
- **为什么关闭 thinking。** 本 runtime 的所有 benchmark 都关闭 thinking，与 PoC 对齐：PoC 的服务端（dram-backed-kv 的 `scripts/serve_online_selector_v2.py`）以 `enable_thinking=False` 渲染，因为 LiteLLM / Mini-SWE-Agent 在发送下一个请求之前会从 assistant 内容中删去 thinking 块。开启 thinking 时，下一个请求不再延续服务端持有的 KV 前缀（前缀不稳定，KV 无法复用）；关闭 thinking 时，空的 think 块属于服务端渲染的 prompt（即上面的生成头），历史保持只追加。因此开启 thinking 的 baseline 运行与这些运行不可比。
- **响应。**
  - `choices[0].message.content` 是全部生成 token 解码后的文本，包括结束标记，以及最后一个 token 越过标记的字符。
  - `finish_reason` 恒为 `stop`，没有 tool call。
  - 边界 `<|im_end|>\n`（EOS 之后是 `\n`）进入服务端的历史，但不返回。
- **Goal-query 的输入。** Goal 使用任务消息里的 PR 描述、上一轮回复、最新的 user 消息及其中的 `<returncode>`，与 PoC 相同。
- **会话。** 请求若是上一个请求，加上一轮回复，再加一条新的 user 消息，就延续当前会话（第 5 节）。
  - 其他请求和 PoC 服务端一样开一个冷启动新会话。例如丢掉被拒回复的格式错误重试（PoC cohort 的 5 个实例中有 1 个出现），以及上下文压缩。
  - 新会话从文本重新渲染整个请求，释放旧会话的 Page，并重置 selector 和 Goal 状态。它的 runtime 行记录 `stage_b.rebuild == true`。

### 6.2 最小客户端

`examples/http_client.py` 发送一个首轮请求（`python -B examples/http_client.py --base-url http://127.0.0.1:8000/v1 --messages examples/messages.json`）。它只是冒烟客户端，不是 agent。`examples/messages.json` 与 `examples/task.json` 对应，所以只能对以该任务启动的服务使用。

本版本拒绝 OpenAI tool-call 协议（`tools`、`tool_calls`、`tool` 消息）。上一个验收版本（commit `4d90616`）提供该协议，并在其 README 里有说明。`examples/http_client.py` 仍保留 `--tools` 选项，因为 `package.py` 会打包这个文件，而且有测试加载它；不要对本 runtime 使用它，本 runtime 会拒绝任何带 `tools` 的请求。

### 6.3 错误处理

出现任何非 2xx 状态、超时、断连或服务退出，都视为这个服务的这次运行已经结束：
1. 保存 HTTP 错误 body、`failures.jsonl` 和服务日志。
2. 用 `SIGTERM` 停止服务。
3. 如需重试，启动一个全新的服务。

不要静默重试 admission 状态不明的请求，不要推进轮次，也不要复用这个 session。违反 6.1 约定的请求会被 HTTP 错误拒绝。

引擎拒绝请求时，HTTP 500 的 body 是空的。原因记录在两处：
- `failures.jsonl` 里该请求那一行的 `chain` 和 `traceback` 字段；
- 服务日志里的 `REMOTE_RA_REGISTERED_ADD {"phase": "engine_rejected", ...}` 一行。

`CAPACITY_WAIT` 拒绝（4.4 节）也会写出同样的日志行，带有 traceback，但它不是错误：前端会重试这一轮，客户端只会看到延迟。

外部 prefix 借用、speculative decoding、async loading、preemption、历史改动和未知 drain 都会直接失败，不会静默回退到 Full。

### 6.4 从其他主机访问服务

服务只监听 `127.0.0.1`，没有任何认证；永远不要把它暴露到 loopback 之外。SWE-bench 需要 Docker，所以 pilot 里 agent 运行在另一台 agent 主机上，通过反向 SSH 隧道访问 GPU 作业（作业 3299922 和 3300434）。每个服务按下面的顺序进行：

1. READY 之后，GPU 作业打开一条反向隧道，远端是 agent 主机上的一个 **Unix socket**：

   ```sh
   ssh -N -R "$AGENT_SOCKET:127.0.0.1:8000" -o ExitOnForwardFailure=yes \
     -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
     -i "$TUNNEL_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes \
     -o UserKnownHostsFile="$KNOWN_HOSTS" "$AGENT_HOST"
   ```

   `AGENT_HOST` 为 `user@agent-host`，`TUNNEL_KEY` 是 agent 主机接受的密钥，`KNOWN_HOSTS` 是固定其主机密钥的文件，`AGENT_SOCKET` 是该主机上一个 `0700` 目录里的新路径（每个服务一个，最长 100 字节）。
2. 作业从 agent 主机检查隧道：`curl -sf --unix-socket "$AGENT_SOCKET" http://localhost/v1/models` 必须列出 `selective-prefix`。
3. 在 agent 主机上，`examples/swebench/relay.py "$AGENT_SOCKET" <port>` 只监听 `127.0.0.1:<port>`，并把每个连接转发到该 socket；`run_agents.sh` 会启动它，并用 `ss` 检查该端口上只有 `127.0.0.1` 在监听。agent 使用 `http://127.0.0.1:<port>/v1`。
4. agent 运行结束后，作业先关闭隧道，再用 `SIGTERM` 停止服务。

为什么用 Unix socket：如果 agent 主机的 sshd 设置了 `GatewayPorts yes`，每个 TCP 远程转发都会绑定到所有网卡，不管客户端要求的是哪个地址，这会把没有认证的服务暴露出去。Unix socket 转发没有 TCP 监听，relay 又显式绑定 loopback，所以无论 sshd 怎么设置，这种形式都只停留在 loopback 上。计算节点需要能向外 SSH 到 agent 主机；pilot 在每个作业的第一个服务之前检查过一次。

一种更简单的 TCP 形式，**不是已验证的形式**：`ssh -N -R 127.0.0.1:<p>:127.0.0.1:8000 <agent-host>`。只有当 agent 主机的 sshd 设置为 `GatewayPorts no`（OpenSSH 的默认值）时，它才只停留在 loopback 上。任何 agent 运行之前，先在 agent 主机上用 `ss -ltn` 检查；`run_agents.sh -`（不用 socket）会做这项检查，并拒绝非 loopback 的监听。

### 6.5 关闭 session（session 池）

在用 `--max-running-sessions` 启动的服务上（4.4 节），`POST /v1/sessions/close`，JSON 请求体 `{"instance_id": "<启动任务 id>"}`，会永久关闭该任务的 session。它只接受回环地址的对端（服务本来就只绑定回环地址；经过 6.4 节的隧道时对端同样是回环地址），与 chat 路由一样没有 HTTP 认证；前端把它作为私有通道上一个经过认证的帧转发给引擎。响应是引擎的关闭回执（`"object": "session.close"`、`opened`、`turns`、释放的 Page 和 block、两个池关闭前后的空闲 block、`open_sessions`）；对已关闭的任务再次关闭，会返回同一份回执并带 `"repeated": true`。拒绝的情况：403（不是回环地址）、404（不是启动任务）、409（`SESSION_BUSY`：该 session 有排队或进行中的请求；请在它结束后重试）、400（没有 session 池，或请求体不对）。从未打开过的任务关闭时不会触及引擎。

### 6.6 MemoryAgentBench kv-session 协议

`--protocol kv-session` 服务 MemoryAgentBench harness 中 KV-session 客户端（方法 `dram_backed_kv`）使用的 token 级 session 协议。客户端自己渲染 chat template、自己解析回复；服务端从不渲染模板。这样的服务需要 session 池（`--max-running-sessions K`，4.4 节），不接受 `--task`：它注册的任务是槽位，每个客户端 session 在第一个请求时绑定到下一个未用的槽位（`--kv-session-slots N`，1 到 64，默认 64，所以一个服务在其生命周期内最多服务 N 个客户端 session）。`--kv-session-thinking on|off`（默认 `on`）固定客户端发送的生成头。这样的服务拒绝 `/v1/chat/completions` 和 `/v1/sessions/close`。

```sh
python -B serve.py --release-dir "$RELEASE" --protocol kv-session --policy observation_query \
  --page-tokens 16 --kv-layout decoupled --active-token-budget 32768 \
  --max-running-sessions 16 --kv-cache-gib 32 --max-tokens 4096 --max-model-len 262144 \
  --dram-arena-gib 88 --release-idle-pages --output "$RUNS/mab-obs" --port 8000 --start
```

- **`GET /v1/kv-session/info`** 返回 `{"method": "dram_backed_kv", "budget": <token 数或 null>, "options": {...}}`：服务的注意力预算（即 `--active-token-budget`，否则为 `null`）和它实际运行的设置（策略、页大小、预算方式和选页合同字段、回复上限、上下文上限、thinking、session 数）。
- **`POST /v1/kv-session/complete`** 只接受以下字段（已缩写）：

```json
{"method": "dram_backed_kv", "session_id": "<32 lowercase hex>", "input_ids": [248046, 198, 248045, "..."],
 "header_tokens": 5, "prefix_tokens": 423, "boundary_tokens": 2, "budget": 32768, "max_tokens": 4096,
 "temperature": 0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "seed": 42, "messages": ["..."]}
```

  - session 的第一个请求发送完整的渲染后 prompt：`boundary_tokens` 为 0，`prefix_tokens` + `header_tokens` 等于 `input_ids` 的个数。
  - 之后每个请求只发送新的尾部：上一条回复的结束分隔符（上一条回复的最后一个 token 不是 EOS 时，把它放在最前面）、新消息和下一个生成头。`input_ids[0]` 必须重复上一条回复的最后一个输出 id，`prefix_tokens` 不能变，尾部必须以固定的生成头结束。服务端的 prompt 是该 session 的 token 流加上 `input_ids[1:]`；阶段 1 从上一条回复的最后一个 token 算到生成头之前（第 5 节）。
  - 采样：只支持贪心（`temperature` 0、`top_p` 1、`top_k` −1 或 0、`repetition_penalty` 1）；`max_tokens` 为 1 到服务的回复上限（`--max-tokens`），并截到剩余上下文以内。`budget` 和 `method` 必须与服务端一致。
  - `messages`（客户端的完整历史）从不生成 token。只使用上一个请求之后新增的消息，供 `goal_query` 的 Goal 使用：其中 tool 和 user 消息的文本作为 observation；user 消息（MemoryAgentBench 随最后一个历史块发送的问题，或格式错误提示）成为 Goal 的显式要求。
- **响应：** `text`（全部生成 token 的解码，含特殊 token）、`output_ids`（含最后一个 token）、`prompt_tokens`（prompt 长度）、`completion_tokens`、`cached_tokens`（`prompt_tokens` 减去 `input_ids` 的个数）、`finish_reason`（`stop` 或 `length`），另有单独命名的 `kv_slot`、`kv_turn`、`effective_max_tokens`、各项等待时间和本轮的页数统计。
- **`POST /v1/kv-session/close`**，请求体 `{"session_id": ...}`，返回 200 `{"closed": true, ...}`；服务从未见过或已关闭的 id 也返回 200（已关闭时返回第一次的结果并带 `"repeated": true`）；该 session 有排队或进行中的请求时返回 409。关闭过的 id 不会再被服务。
- **错误。** complete 请求的任何失败都是 400 `{"error": ..., "type": ...}`。prompt 没有为回复留出空间时，在任何模型计算之前拒绝，`error` 恰好是 `"Logical RoPE context limit reached"`（`type` 为 `CONTEXT_LIMIT`）。接纳之前发现的拒绝（请求体、采样字段、生成头、接续 token、prefix、上下文）不影响该 session；引擎中的失败会使该 session 失效，客户端随后关闭它。
- **证据。** 每个 `runtime.jsonl` 行增加 `kv_session`：客户端 session id 和槽位、token 几何（`header_tokens`、`prefix_tokens`、`boundary_tokens`、`input_tokens`）、客户端给的和实际生效的 `max_tokens`、`finish_reason`，以及 Goal 的消息来源和显式要求。
- **Gemma 4（4.14 节）。** 在 Gemma release 上，该协议使用关闭 thinking 的 Gemma 轮次格式：user 轮之后为 7 个 token 的生成头，tool response 之后 `header_tokens` 为 0（tail 以 `<tool_response|>` 结尾，由 phase 2 计算），新消息从 `<|turn>` 或 `<|tool_response>` 之后开始，停止 id 为 0、1、50 和 106，启动记录带 `"grammar": "gemma4"`。
- **状态。** 已在 Qwen3.6-27B 上经过 GPU 验证（关闭 thinking，runtime `0d37346`），见 MemoryAgentBench 作业 3397844（G1：`full` 和 `observation_query` 各一个 262,144 token 的 session，每一轮都 token 级精确一致，stock vLLM 对 `full` 各 prompt 的 eager 回放在全部 13 轮上 token 完全相同，协议探针全部通过）和 3397842（G2：16 个 session，开 `--release-idle-pages`，16 GiB 和 24 GiB，4.13 节；runtime 正确，作业中唯一未通过的检查是 harness 检查本身的 bug，修正后离线重新核验为通过）。真实的 32 个 case 的运行（G4：作业 3405844 `observation_query`、3405845 `goal_query`、3405855 stock，runtime `f9322a6`）跑完了每一个 case，没有 runtime 错误；它们的结论和两项修正后的 harness 检查见 [docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md)；官方评分待出。Gemma 4 profile 的 kv-session 协议（上文，4.14 节）未经 GPU 验证。CPU 测试（`test_kv_session_protocol.py`，装了 Transformers 时使用固定的 Qwen3.6 tokenizer：thinking 开启时生成头为 5 个 token，服务端的原始 token 流与模板在 `preserve_thinking` 下的渲染逐 token 相同；`test_kv_session_server.py`；`test_kv_session_e2e.py`，MemAgent session 经打包的路由和引擎假实现运行）。

MemoryAgentBench 正式测试的配置（32 个 case，每项能力 8 个，最多 16 个同时运行，关闭 thinking，`observation_query` 和 `goal_query` 两个策略，以及一个经由 harness 的 Vanilla transport 访问的 stock vLLM 0.24.0 对照）、它的验证状态和 baseline 与之对齐的检查表见 [docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md](docs/MEMORYAGENTBENCH_BENCHMARK.zh-CN.md)。

### 6.7 Chat text 协议（Terminal-Bench 2.0 与 Terminus 2）

`--protocol chat-text` 在 `/v1/chat/completions` 上服务普通的 OpenAI chat 文本，token 由服务端自己的 transcript 给出，面向每轮发送完整文本历史、自己解析回复的 agent。目标是原样的 Harbor 0.21.0 Terminus 2 跑 Terminal-Bench 2.0：JSON 动作，没有 system 消息，thinking 关闭。这样的服务需要 session 池（`--max-running-sessions K`，4.4 节；chat-text 服务为 1 到 24，其他服务为 1 到 16），不接受 `--task`：它注册的任务是槽位，惰性绑定到客户端 session（`--chat-text-slots N`，1 到 1024，默认 512）。它服务 `full`、`random50`、`observation_query` 和 `random_fill`；拒绝 `goal_query`，因为它的 Goal 是面向 SWE 的（`<pr_description>`、`<returncode>`）。这样的服务拒绝 `/v1/sessions/close` 和 kv-session 的路由。Gemma 4 profile 的 release 以 Gemma 的轮次格式服务该协议（4.14 节）：除渲染、停止 id 和容量数字外，下文对两个模型都成立。

```sh
python -B serve.py --release-dir "$RELEASE" --protocol chat-text --policy observation_query \
  --page-tokens 16 --kv-layout decoupled --active-token-budget 8192 \
  --max-running-sessions 20 --kv-cache-gib 32 --max-tokens 4096 \
  --connector-pool-gib 1 --dram-arena-gib 64 --output "$RUNS/tb-obs" --port 8000 --start
```

- **请求。** 非流式，一个 choice，`model` 为 `selective-prefix`。`messages`：可选的 system 消息，然后是 user 和 assistant 文本；system 之后的第一条是 user 消息，最后一条也是。content 必须是字符串；`tools`、`tool_calls` 和 `tool` 消息返回 400，消息的其他字段被忽略。
- **采样。** 贪心解码，忽略客户端的采样字段。`max_tokens`（或 `max_completion_tokens`）从 1 到回复上限（`--max-tokens`）；没给时取该上限，并按剩余上下文截断。回复只在 EOS/pad 处停止（没有 marker）。
- **渲染。** 参照是固定模板在 `enable_thinking=false`、`preserve_thinking=true` 下的渲染。新的或重建的 session 把整段对话交给模板渲染，再接生成头 `<|im_start|>assistant\n<think>\n\n</think>\n\n`。续接的请求追加上一条回复生成时的原始 token id 和 `<|im_end|>\n`（EOS/pad 之后只追加 `\n`），再追加模板对新 user 消息的渲染 `<|im_start|>user\n{content.strip()}<|im_end|>\n`；启动时用模板检查一次这些片段。一个请求如果等于上一个请求加上一条回复（`strip()` 后比较）再加一条新 user 消息，就续接 session；其他请求（例如 Terminus 的摘要交接、回退后的历史、重发的请求）都会重建 session（`stage_b.rebuild`）。只要每条回复重新编码后仍是它自己的 id，每个 prompt 都等于模板对该请求消息的渲染。
- **响应。** `choices[0].message.content` 是解码后的回复（跳过特殊 token），`finish_reason` 在 EOS/pad 处为 `stop`、达到上限时为 `length`，`usage` 给出 prompt、completion 和总 token 数。
- **上下文上限。** prompt 加上请求的回复长度超过上下文上限（没有 `max_tokens` 时：连一个 token 都放不下）的请求返回 400，正文是 vLLM 自己的文字 `This model's maximum context length is …`。litellm 和 Harbor 会把它映射为上下文长度错误。服务端什么也不变；该 session 的下一个请求照常继续。
- **Session。** 请求的键是它的初始段（如有 system 消息则包括它，再加第一条 user 消息），客户端发送 `X-Session-ID` header 时还包括该 header（litellm 也会把它作为 `session_id` 放进请求体）。Terminus 2 默认不发 header，它的第一个 prompt 含有容器的主机名，所以每个 trial 都是自己的 session。在同一个 header id 下，初始段不同的请求（Terminus 摘要里不带历史的 "questions" 调用）是单独的 session（类型 `aside`），不会重置主 session。一个键在第一个请求时绑定到下一个未用的槽位。如果一个键的槽位上已有排队或进行中的请求，新请求最多等它 30 s，之后返回 409 `SESSION_BUSY`。当槽位因请求失败（客户端断开或取消、引擎失败并返回 500）、被服务端关闭或已服务 256 个请求时，该键换到一个新槽位（rebind：新槽位上的第一个请求从文本渲染，做一次完整 prefill）。槽位从不复用；全部用完后，新 session 返回 400 `SESSION_LIMIT`。
- **关闭。** 默认路径上不会有关闭调用。失败的、已换走的、被客户端关闭的以及达到请求上限的槽位立即关闭。只有在没有空余的打开 session（即已有 K 个打开）时才会关闭正在使用的 session：有请求在等打开的 session 时，关闭最久未用、且空闲至少 `--chat-text-evict-after` 秒（默认 30）的 session，优先关 aside，每个等待的请求关一个；没有请求在等时，关闭最久未用、且空闲至少 `--chat-text-idle-timeout` 秒（默认 600）的一个 session，让下一个新 session 有空位。只要还有空位，无论空闲多久都不会关闭正在使用的 session。发送 `X-Session-ID` 的客户端可以自己关闭：`POST /v1/chat-text/close`，请求体 `{"session_id": ...}`（或 `{"key_sha256": ...}`，即初始段规范 JSON 的 SHA-256）关闭该 id 下的所有 session。它返回 200，未知的或已关闭的 id 也返回 200（已关闭时返回第一次的结果并带 `"repeated": true`），其中任一 session 有进行中的请求时返回 409；关闭后的 id 不再被服务。`GET /v1/chat-text/info` 返回选项、计数和各客户端 session。
- **容量。** 最多 K 个 session 同时打开。对 Harbor 的 `-n 16`，设置为 K = 20：四个空余 session 容纳结束的 trial（不会有关闭调用）和 Terminus 的摘要子调用，因此执行长命令的运行中 trial 不会被驱逐。K 决定 `max_num_seqs`、Mamba 池和锁页 Page arena 的大小（4.4 节）。在 P = 16、32 GiB KV 下，K = 20 有 41 个 Gated-DeltaNet 槽位，Mamba 池 124 个块（5.93 GiB，比 K = 16 多 24 个块即 1.15 GiB），注意力块 26,699 个（少 1,175 个）。按 4.4 节的规则、4096 token 的回复，20 个 session 在 131,072 token 时：8192 token 预算需要 24.9 GiB（K = 16 时 21.7），加上典型的最后一轮需要 33.1 GiB（28.2），加上最坏的最后一轮需要 48.9 GiB（40.7）；在 65,536 token 时分别为 20.8、29.1 和 44.8 GiB。65,536 token 时的锁页主机内存：不设 arena 为 84.1 GiB（K = 16 时 67.5），`--dram-arena-gib 64` 可保持在 67.9 GiB（64 GiB 的 1 MiB Page 副本，例如 20 个 52K token 的 session）；131,072 token 时为 164.1 GiB，超过 launcher 的 128 GiB，必须设 arena。一个槽位服务 256 个请求；一个服务最多绑定 N 个槽位（客户端 session、aside 和 rebind 都算）；生命周期内的请求上限是 max(1,024, 256 × N)。空闲 session 在两轮之间把 Page 留在 GPU 上，又不会有关闭调用，所以打开的 session 的 Page 可能占满池（在 K = 20、32 GiB、65,536 token 下，`full` 的一次 Terminal-Bench dry run 中就是这样，4.4 节）：Terminal-Bench 的服务要开 `--release-idle-pages`（4.13 节）。
- **证据。** 每个 `runtime.jsonl` 行增加 `chat_text`，`chat_sessions.jsonl` 记录每次绑定、换绑和关闭（第 8 节）。
- **状态。** 在 GPU 上，`full` 在一张 GH200 上以 runtime `68d19b4` 和 [docs/TERMINALBENCH_BENCHMARK.zh-CN.md](docs/TERMINALBENCH_BENCHMARK.zh-CN.md) 的设置（K = 20、32 GiB、`--release-idle-pages`）通过了 16 个任务的 Terminal-Bench 2.0 dry run：16 个 trial 全部结束，333 个请求，每个请求都与其 native 回执 token 级精确一致，没有抢占，也没有容量停滞。`observation_query`（8,192 token 预算）通过了同样的 dry run：384 个请求，18 次重建，其中 238 个请求的选择生效，每个请求都 token 级精确一致，没有抢占，也没有容量停滞。`random_fill` 还没有在 Terminal-Bench 上运行。CPU 测试：`test_chat_text_protocol.py`（装了 Transformers 时用固定的 Qwen3.6 tokenizer：带转义的 JSON 回复、非 ASCII 指令、有和没有 system 消息，都与 Transformers 的模板渲染相等）、`test_chat_text_server.py`、`test_chat_text_e2e.py`（16 个 session 经打包的 router 和引擎替身一起解码；取消、换绑、空闲关闭和驱逐、上下文拒绝；`REMOTE_RA_SLOW_TESTS=1` 再加 12 个 session × 8 个 episode，含摘要和 aside）和 `test_chat_text_replay.py`；`test_integrate_flags.py` 在三条解码快速路径下运行 chat-text session，并在池压力下配合 `--release-idle-pages` 运行。回放测试需要显式开启（`REMOTE_RA_TB_ATIF_DIR`；轨迹不在本仓库中）：8 月两次 Terminus 2 全量运行的 178 个 trial，每个用固定 tokenizer 回放两次（客户端 token 估计准确、以及偏低 20 %），共 3,363 个请求，其中 83 次上下文拒绝、33 次重建、7 个 aside，每个被服务的 prompt 都等于模板对其消息的渲染。stock vLLM 0.24.0 对照沿用 vLLM 默认的 chat template 参数（与 SWE-bench 的运行相同）：它重新渲染每条较早的回复时不带空的思考块，因此每条历史回复比本服务的 token 流少 4 个 token（`<think>`、`\n\n`、`</think>`、`\n\n`），并且它重新编码回复文本，而本服务保留生成时的 id。Gemma（4.14 节，仅 CPU）：`test_gemma_chat_text.py`、`test_gemma_chat_text_e2e.py` 和 `test_gemma_chat_text_replay.py`；后者用固定的 Gemma tokenizer 和模板、以 Gemma 的轮次格式把同样 178 个 trial 各回放两次：3,359 个请求，126 次上下文拒绝，43 次重建，7 个 aside，每个被服务的 prompt 都等于参照渲染，去掉 thought channel 后等于 stock vLLM 的 prompt（0 处差异）。stock Gemma 对照的 prompt 同样每条过去的回复少 4 个 token（`<|channel>thought\n<channel|>`）。

Terminal-Bench 2.0 正式测试的配置（全部 89 个任务，Harbor 的 Terminus 2 以 `-n 16` 运行，`full`、`observation_query` 和 `random_fill` 三个策略，以及一个 stock vLLM 0.24.0 对照）、它的验证状态和 baseline 与之对齐的检查表见 [docs/TERMINALBENCH_BENCHMARK.zh-CN.md](docs/TERMINALBENCH_BENCHMARK.zh-CN.md)。

## 7. 用 Mini-SWE-Agent 2.4.2 跑 SWE-bench Verified

SWE-bench Verified 正式测试的配置（前 32 个 instance，16 个 agent 并发、按列表顺序补入，`full`、`observation_query`、`goal_query` 和 `random_fill` 四个策略，以及一个 stock vLLM 0.24.0 对照）和 baseline 与之对齐的检查表见 [docs/SWEBENCH_BENCHMARK.zh-CN.md](docs/SWEBENCH_BENCHMARK.zh-CN.md)。

本节介绍多 session SWE-bench pilot 的设置：3 个 Astropy instance × `full`/`goal_query`/`observation_query`、每个服务 3 个 session（作业 3299922），以及 10 个 instance × 同样 3 个策略、每个服务 5 个 session（作业 3300434），runtime 为 `b1e6988`，784 token 的 Page。之后同样 10 个 instance 又以 16 token 的 Page × `full`/`observation_query`/`goal_query`/`random_fill` 跑了一遍，每个策略一个作业（作业 3313802、3314010、3314319、3314623，runtime `780fe54`；结果见 4.7 节），agent 主机不变。每个服务都通过了服务端检查。每个 instance 只跑一次，不构成 benchmark 证据。这两次 pilot 没有跑 `random50`。

### 7.1 设置

| 设置 | 值 |
| --- | --- |
| Agent | Mini-SWE-Agent 2.4.2（litellm 1.103.1、datasets 5.0.1） |
| Runner | PoC 的 `swebench_runner.py`，参数 `--subset verified --split test --workers <session 数> --filter '^(<id>\|<id>…)$' --output <dir>` |
| 配置叠加 | `-c swebench_xml.yaml`（Mini-SWE-Agent 内置）`-c swebench_prompt_override.yaml`（PoC 的 prompt）`-c model_override.yaml` |
| 模型 | `examples/swebench/model_override.yaml`：`litellm_textbased`、`openai/selective-prefix`、`api_base` `http://127.0.0.1:<port>/v1`、占位 key、temperature 0.0、`max_tokens` 1536（`run_agents.sh` 的 `MAX_TOKENS`；4.9 节）、`cost_tracking: ignore_errors` |
| 环境 | `docker`、`cwd` `/testbed`、超时 120 秒、`bash -c`、`PAGER`/`MANPAGER` 为 `cat`、`LESS` 为 `-R`、`PIP_PROGRESS_BAR` off、`TQDM_DISABLE` 1 |
| 限制 | `step_limit` 90（`run_agents.sh` 的 `STEP_LIMIT`；4.9 节）、`cost_limit` 0 |
| 进程环境变量 | 不设置 `MSWEA_CTX_PATCH`、`AGENTDIET_MODE=off`、`HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`、`MSWEA_GLOBAL_CONFIG_DIR` 指向一个空目录 |
| 服务 | 每个策略、每个组一个全新的服务，`--task tasks-g<k>.json`、`--kv-cache-gib 24`，`goal_query` 带 0.6B rewriter；worker 数 = session 数 = 组的大小。16 token Page 的 pilot 给每个服务（包括 `full`）都加了 `--page-tokens 16 --kv-layout decoupled`，`random_fill` 不带 rewriter 选项 |
| 评测 | swebench 5.0.2、`SWE-bench/SWE-bench_Verified`、split `test`、`--timeout 1800`、`--max_workers 2`（pilot 脚本的默认值） |

填入端口后的 `model_override.yaml` 与 10-instance pilot 使用的六个文件逐字节相同。服务端本来就忽略客户端的采样字段（6.1 节）。Mini-SWE-Agent 会写出 `<output>/<instance>/<instance>.traj.json` 和 `<output>/preds.json`。

**PoC 文件。** `examples/swebench/upstream/` 里是 dram-backed-kv PoC 仓库的 `swebench_runner.py` 和 `swebench_prompt_override.yaml`（commit `a70701e` 的 `scripts/`，LF 的 Git blob 字节；prompt override 与 cohort-2 分支 `0ab0078` 上的副本相同），与 pilot 实际运行的副本逐字节一致（见 `upstream/SOURCE.txt`、`upstream/SHA256SUMS`）。`run_agents.sh` 默认使用它们，SHA-256 与下列固定值不同的文件会被拒绝：

| 文件 | SHA-256 |
| --- | --- |
| `swebench_runner.py` | `34ca6d9288c72830e61bd00ad8b15790cab4c29515db24b8df30775cfc0880a7` |
| `swebench_prompt_override.yaml` | `1b6a48666ad528775bd11225fa4d5c002721c25c0cdbc05b99cc15b2de31336a` |

这两个文件有两点会影响结果，和其他设置比较时要注意：
- runner 替换了 Mini-SWE-Agent 的 agent 类。达到步数上限时，它会提交 agent 最后打印过的、至少 3 行的 `git diff`（退出状态 `LimitsExceeded_InformalSubmission`）；连续 10 次相同命令时终止该实例（`LoopDetected`，有没有这个补丁都会终止）。它还会尝试导入一个可选的 `agentdiet_adapter`，本仓库不包含；`AGENTDIET_MODE=off`（pilot 的设置）时不会用到它。
- prompt 是为 Astropy 写的（"The testbed is an Astropy repository"，并带有 Astropy 专用的测试建议）。pilot 只跑了 Astropy 实例；其他仓库需要改写 prompt，这**不属于已验证的运行**。

### 7.2 Agent 主机

agent 主机需要 Docker、`curl`、`ss`、`python3`、一个 agent 环境和一个评测环境。pilot 的环境是上面列出的版本；创建它们的命令没有记录，所以下面的命令**不在已验证的运行范围内**：

```sh
MINI=/abs/path/to/mini-env
EVAL=/abs/path/to/eval-env
uv venv --python 3.12 "$MINI"
uv pip install --python "$MINI/bin/python" mini-swe-agent==2.4.2 litellm==1.103.1 datasets==5.0.1
uv venv --python 3.12 "$EVAL"
uv pip install --python "$EVAL/bin/python" swebench==5.0.2 datasets==5.0.1
```

agent 离线运行，使用事先准备好的缓存：Hugging Face 数据集 `princeton-nlp/SWE-Bench_Verified`（runner 和任务构建脚本用），以及每个 instance 一个 Docker 镜像 `docker.io/swebench/sweb.eval.x86_64.<把 instance id 中的 "__" 换成 "_1776_" 并转成小写>:latest`。评测程序加载 `SWE-bench/SWE-bench_Verified`（十个 pilot instance 的 statement 与前者完全相同），运行时没有设置离线变量。在能联网时准备一次缓存，例如用 `load_dataset(...)` 和 `docker pull`（不在已验证的运行范围内）。

### 7.3 任务

每个服务的 `--task` 列表必须恰好包含其 agent 要跑的 instance，`{instance_id, problem_statement}` 原样取自 SWE-bench Verified。请求按任务消息里的 problem statement 路由，所以如果某个 agent 的 instance 不在列表里，该服务的这次运行会以 HTTP 错误结束（4.4 和 6.3 节）。`examples/swebench/build_tasks.py` 按我们 harness 的方式生成这些文件：每个 instance 一个 `task-<id>.json`，每个服务组一个 `tasks-g<k>.json`（分成最少的组，每组最多 `--max-sessions` 个，各组大小最多差一）。在能访问数据集的地方运行：

```sh
"$MINI/bin/python" examples/swebench/build_tasks.py --output-dir /abs/path/to/tasks --max-sessions 5 \
  astropy__astropy-12907 astropy__astropy-13236 astropy__astropy-14096
```

给定相同的 statement，它能逐字节复现 pilot 固定的十个 `task-*.json` 文件，以及作业 3299922 和 3300434 的 `tasks-g*.json` 列表；对这十个 instance，如果 statement 与 pilot 固定的 SHA-256 不同，它会拒绝。把 `tasks-g<k>.json` 复制到 GPU 一侧。

对 session 池服务，`--pool` 按给定顺序生成一个包含全部 instance（1 到 64 个）的 `tasks-pool.json`，不再分组；`run_agents.sh` 按这个顺序补入 agent：

```sh
"$MINI/bin/python" examples/swebench/build_tasks.py --output-dir /abs/path/to/tasks --pool <instance id>...
```

### 7.4 运行一个组

每个策略、每个组：用该组列表启动一个全新的服务，打开隧道，然后在 agent 主机上运行一个 runner（`run_agents.sh <socket> <port> <新的输出目录> <该组的 instance id>`）。使用 Slurm 脚本时：

```sh
export REPO VENV RELEASE
export TASKS=/abs/path/to/tasks/tasks-g0.json POLICY=goal_query KV_GIB=24 REWRITER="$MODELS/Qwen3-0.6B-c1899de"
export PAGE_TOKENS=16 KV_LAYOUT=decoupled
export RUN_ROOT=/abs/short/jobs MODE=swebench AGENT_SECONDS=5400
export AGENT_HOST=user@agent-host TUNNEL_KEY=/abs/path/to/tunnel_key KNOWN_HOSTS=/abs/path/to/agent_known_hosts
export AGENT_SOCKET=/abs/path/on/agent/sock/g0-goal_query.sock AGENT_PORT=20000
export AGENT_RUN=/abs/path/on/agent/examples/swebench/run_agents.sh AGENT_OUT=/abs/path/on/agent/runs/goal_query-g0
export AGENT_ENV="MINI_PYTHON=/abs/path/on/agent/mini-env/bin/python MSWEA_CONFIG_DIR=/abs/path/on/agent/mswea-config"
sbatch --account="$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 --gpus=1 \
  --cpus-per-task=16 --mem=200g --time=03:45:00 --job-name=selective-prefix \
  --output="$RUN_ROOT/slurm-%j.out" --export=ALL --mail-type=BEGIN,END,FAIL --mail-user="$MAIL" \
  "$REPO/examples/slurm/serve_gh200.sh"
```

说明：
- 每个策略一个服务，每个组都用全新的服务、全新的输出目录和 socket。pilot 让每个策略跑同样的分组，在一张 GPU 上一个服务接一个服务地运行。
- `PAGE_TOKENS=16 KV_LAYOUT=decoupled` 是 `goal_query`、`observation_query` 和 `random_fill` 的推荐设置（4.6 节）；两者都不设就是默认的 784 token 布局。跑 `random_fill` 时设 `POLICY=random_fill`；脚本只给 `goal_query` 传 rewriter。
- 时间：作业 3300434 中，一组 5 个 agent 用时 1,255–2,120 秒，它的 6 个服务（3 个策略 × 2 组）中最后一个在作业开始后 10,851 秒结束，在 10:45:00 的申请时间之内。pilot 对每次 agent 运行设了时间上限：3,300 秒（作业 3299922）或 5,400 秒（作业 3300434）。16 token Page 的 pilot（每个作业一个策略、2 组，agent 上限 3,000 秒）中，一组用时 1,215–2,045 秒，每个作业用了 02:00:00 申请时间中的 49–68 分钟。
- SSH 断开或超时不会停止远端的 runner。pilot 在这种情况下会结束该 runner，并只删除它启动的容器（日志里 `minisweagent-<8 位十六进制>` 的名字）。
- 作业 3300434 和 3314319 中有些 `goal_query` 轮次回退到了确定性的 Goal 字段（finding `GOAL_FALLBACK`；第 5 节）；服务仍然通过。
- Session 池（4.4 节）：设 `TASKS=.../tasks-pool.json` 和 `MAX_RUNNING=<K>`。Slurm 脚本会给 `serve.py` 传 `--max-running-sessions K`，给 `run_agents.sh` 传 `MAX_RUNNING`；后者随即运行一个 K 个 agent 的池，按列表顺序每个 instance 一个 runner，每个 runner 结束时关闭该 instance 的 session（其最后一个请求仍在清理时会重试），再启动下一个。它写出 `runs/<instance>/`、`logs/<instance>.log`、`closes/<instance>.json`、`timeline.jsonl`（每个 instance 的开始和结束时间、runner 退出码和关闭状态：即从第一个开始到最后一个结束的墙钟时间）以及供 `evaluate.sh` 使用的合并后的 `preds.json`。不在已验证的运行范围内。
- 步数上限和回复上限（4.9 节）：`STEP_LIMIT=<1-256>` 和 `MAX_TOKENS=<1536|4096>` 设定 `run_agents.sh` 所写 model override（`render_override.py`）中 Mini-SWE-Agent 的 `step_limit` 和 `max_tokens`；Slurm 脚本把两者传给 agent 主机，并把 `MAX_TOKENS`（以及 `MAX_MODEL_LEN`、`DRAM_ARENA_GIB`）传给 `serve.py`。不设置时，override 与 pilot 用的文件逐字节相同，`serve.py` 也不会多出任何参数。不在已验证的运行范围内。

### 7.5 评测

agent 结束后，在 agent 主机上按策略评测（同一策略的所有组输出放在一次调用里）：

```sh
EVAL_PYTHON="$EVAL/bin/python" examples/swebench/evaluate.sh my-run-goal_query goal_query \
  /abs/path/to/eval/goal_query /abs/path/to/runs/goal_query-g0 /abs/path/to/runs/goal_query-g1
```

它写出一个包含所有计划 instance 的 predictions 文件（没有 prediction 的 instance 用空 patch，并报告为缺失），在新的评测目录里运行 `python -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Verified --split test --predictions_path … --max_workers 2 --timeout 1800 --run_id …`，并根据 `logs/evaluation/<run_id>/results.json` 为每个 instance 输出一行。trajectory、predictions 和日志都包含 prompt 和工具输出，不要放进公开发布。

## 8. 每个服务的证据

每个 `--output` 目录包含：

| 文件 | 内容 |
| --- | --- |
| `startup.json`、`launch.json`、`chat_template.jinja` | 冻结的任务/配置哈希、确切的服务命令、固定模板 |
| `ready.json`、`worker-attached.json` | READY 身份（含选页合同和 norm 前向量空间）与 worker 接入回执 |
| `runtime.jsonl` | 每轮一行：响应 `id`、内部 request ID、token/顺序哈希、`candidate_page_count`、`selected_logical_pages`、`prepare`/`restore` 的 H2D 字节数、joined purge/drain、延迟。`stage_b` 记录本轮几何：已计算起点 C、候选终点、observation 区间、阶段 1 区间、阶段 2 起点、`retain`/`restore`/`release`、已计算终点 C′ = prompt + 生成 − 1、封存终点和 `rebuild`。Goal/Observation 的行另有 `query.selection`（`reason`、`capacity_pages`、`mandatory_pages`、`recent_pages`、`retrieved_pages`、合同与投影哈希）和 `query.goal`（变体、rewriter 是否尝试/成功/回退、缓存命中、冷 forward）；`goal_query` 的行另有 `query.goal_text`（见下文） |
| `goals.jsonl` | 仅 `goal_query`：每个请求的 Goal 哈希背后的文本，以及 Goal rewriter 的输入和输出（见下文） |
| `native-<sha256(request id)>.json` | native plan（`selected`、`retain`/`restore`/`release`、run 页、tail 标记、阶段 1 lease）、worker frames（`prepare`、`phase1`、`phase1_drain`、`reserve`、`allocation_update`、`complete`）和 drain 回执 |
| `failures.jsonl` | 约定/运行时失败，带异常 `chain` 和 `traceback`（成功时为空） |
| `engine-cleanup.json`、`parent-cleanup.json` | 关停回执 |

解耦布局会增加一些字段；默认布局不写其中任何一个（`docs/exp-decouple-pages/WP4.md` 第 4 节）：
- `ready.json`：`kv_layout`（P、M、K）、`num_blocks_by_pool` 和 `free_blocks_by_pool`（注意力池在前，Mamba 池在后）；`worker-attached.json`：同样的内容，记为 `kv_pools`。
- 每个 `runtime.jsonl` 行和 native 文件：该请求 drain 之后的 `kv_pools.free_blocks_by_pool`。
- `engine-cleanup.json`：所有 session 关闭后的 `kv_pools`。运行干净时两个池都回到 READY 时的基线（除每个池的 null block 外全部空闲），已验证的运行检查过这一点。
- 封存 Page 的记录用 `token_span` [pP, (p+1)P] 而不是 `token_sha256` 来指明其 token，Page 向量使用 `compact_int8_v1` 形式 `[[128 个 int8], scale]`。

session 池服务（4.4 节）还会写 `sessions.jsonl`，每次 session 打开和关闭各一行（`schema` 为 `remote-ra-session-event-v1`，含 session id、序号和任务 id、`open_sessions`、各池的空闲 block，关闭时还有其回执）；`ready.json` 增加 `session_pool`（`max_running_sessions`、`registered`）；每个 `runtime.jsonl` 行增加 `open_wait_s`；`engine-cleanup.json` 的 `sessions` 列出关停时仍打开的 session，并增加 `session_pool`（哪些由 harness 关闭、哪些未打开就关闭、哪些在关停时仍打开、哪些从未打开）。不带这个选项的服务不写其中任何一项。

开了 `--connector-pool-gib` 的服务（4.9 节）只在 `startup.json` 中记录 `connector_pool_bytes`。

开了 admission 参数的服务（4.10 节）在 `startup.json`、`worker-attached.json` 和 `ready.json` 中记录固定的 `admission` 记录；开 `--gdn-device-handoff` 时，每个 `runtime.jsonl` 行增加 `restore.d2d_bytes`。不开这些参数的服务不写其中任何一项。开 `--goal-rewriter-batch N` 的服务在 `startup.json` 和 `ready.json` 中记录 `goal_rewriter_batch`，并在每个 `goals.jsonl` 交互中增加 `batch_id` 和 `batch_size`。

开了 `--language-model-only` 的服务（4.12 节）在 `startup.json` 及其 `effective_server_config` 中记录 `language_model_only`，在 `worker-attached.json` 中记录 worker 的纯文本记录，`ready.json` 两者都记录。不开这个参数的服务不写其中任何一项。

kv-session 服务（6.6 节）在每个 `runtime.jsonl` 行增加 `kv_session`。chat-text 服务（6.7 节）在每个 `runtime.jsonl` 行增加 `chat_text`（`client_session`、`kind` session/main/aside、`key_sha256`、`session_id`（X-Session-ID 或 null）、`slot`、`slot_turn`、`client_turn`、`binding` bind/rebind、`rebind_reason`、`rebuild`、请求的和实际的 `max_tokens`、`prompt_tokens`、`completion_tokens`、`finish_reason`、`busy_wait_s`），并写 `chat_sessions.jsonl`（`schema` `remote-ra-chat-text-session-v1`：每次绑定、换绑和关闭一行，含槽位、原因、空闲时间和关闭回执里的轮数）。开 `--release-idle-pages` 的服务（4.13 节）写 `capacity.jsonl`，在每个 native 文件和 `runtime.jsonl` 行增加 `idle_release`，在 `engine-cleanup.json` 增加总计，并在 `startup.json`、`worker-attached.json` 和 `ready.json` 中记录 `release_idle_pages`。其他服务不写其中任何一项。

**Goal 文本（仅 `goal_query`）。** `goals.jsonl`（`schema` 为 `remote-ra-goal-text-v1`）对每个到达 native 终态的请求（完成或取消）写一行，保存 `runtime.jsonl` 中 Goal 哈希背后的文本：
- `request_id`、`session_id`、`session_index`、`turn`、`local_turn`、`epoch`、`terminal_state`：与该请求的 native 文件相同。
- `goal`：`text` 是 selector 实际使用的 Goal，即其 selector payload（schema 版本、root goal、检索 Goal、policy hint）的规范 JSON；`sha256` 等于该行的 `query.goal_hash`。
- `query`：`text` 是 Goal query 的 `<retrieval_query>` 文本（`variant`，第 5 节）；`sha256` 等于 `query.goal.payload_sha256`。
- `decision`：gate 决策和实际采用的决策、触发条件、变化的字段、rewriter 是否尝试和成功、`fallback_reason`。
- `rewrites`：构造 Goal 时每次调用 Qwen3-0.6B rewriter 一项：`request`（发送的确切 JSON payload）、`response`（rewriter 进程返回的原始一行；没有返回时为 `null`）和 `error`（失败调用的 `类型: 消息`，等于 `fallback_reason`）。成功时这一行包含 rewriter 从其生成结果中解析出的候选。rewriter 无法解析自己的生成结果时，这一行只有错误信息和 token 计数：随仓库提供的 rewriter 脚本不返回生成的文本。调用之前就被拒绝的请求（超出大小上限）没有对应项。每一项还记录该次调用的时间（用服务进程的 `time.perf_counter` 测量）和子进程自己报告的计数：
  - `wait_ms`：从发起调用到 rewriter 接手。rewriter 在锁内一次只服务一个调用，所以这是排在其他 session 的调用后面所花的时间。
  - `start_ms`：调用内部启动 rewriter 进程的时间；只有进程已退出或之前某次调用超时（rewriter 随后会关闭进程）时才不为 0。
  - `call_ms`：从发送到收到回答的那一行；没有回答时到失败为止（超时一直算到 120 秒上限）。
  - `input_tokens`、`output_tokens`、`generation_latency_ms`：从回答行中复制（子进程的 prompt token 数、生成 token 数和它的 `generate` 用时）；回答行中没有时为 `null`，例如超时之后。
  - `batch_id`、`batch_size`（只在开 `--goal-rewriter-batch N` 时，4.10 节）：服务内的批次序号和一起发送的调用数。这种服务上，`wait_ms` 是排队到批次形成的时间，`start_ms` 是子进程启动和批次行，`call_ms` 是批次从发送到回答，`generation_latency_ms` 是批次的 `generate` 用时；token 计数是该调用自己的。

该请求的 native 文件和 `runtime.jsonl` 行分别用 `goal_text` 和 `query.goal_text`（`{"file": "goals.jsonl", "sha256": …}`）指向这一行，`sha256` 是该行不含换行符的字节的 SHA-256。验证方对 `runtime.jsonl` 中每个 `goal_query` 行检查：`goals.jsonl` 中有一行的 SHA-256 等于该值，且 `request_id` 相同；`goal.text`（UTF-8）的 SHA-256 等于 `query.goal_hash`；`query.text` 的 SHA-256 等于 `query.goal.payload_sha256`。16 个 session × 70 轮时，每个服务的该文件约 10–12 MB，每轮约 9.5 KB，主要是 `goal.text` 中每轮重复的 root goal（用替身 rewriter 重放作业 3363871 的 32 条 trajectory 估算）。记录不改变任何 Goal、选页或 rewriter 调用，也不做逐请求检查；服务启动时检查 Goal 文本的哈希等于 GoalRecord 的 version hash。其他策略既不写该文件，也不写引用。

**Goal 用时（`goal_query`，多个 session 一起解码时）。** Goal 在 Goal 线程上构造时（4.10 节），native 文件记录 `goal_async_ms`（从挂起该 admission 到恢复它）和 `goal_construct_ms`（构造的实际用时）。如果构造调用了 rewriter，native 文件再增加 `goal_rewrite_wait_ms` 和 `goal_rewrite_call_ms`，即其各次调用 `wait_ms` 与 `call_ms` 之和；`goal_construct_ms` 包含这两部分。没有调用的构造两者都不写。其他策略的服务，无论 Qwen 还是 Gemma 4，都不写这些字段，也不写上面每次调用的用时（`test_gemma_goal_rewriter.py`）；带 rewriter 的 Gemma 4 `goal_query` 服务（4.14 节）通过与 Qwen 相同的代码写入它们，这一点没有单独测试。

干净关停需要同时满足（`examples/slurm/server_checks.py cleanup` 会检查这些回执）：
- `parent.cleanup_sha256 == SHA256(engine-cleanup.json 原始字节)`；
- 两份回执的 `joined == true`；
- `epoch`/`session_id` 一致；
- `engine.runtime_manifest == ready.runtime_manifest`，`engine.pending_operations == 0`；
- `parent.child_exitcode == 0`，`parent.natural_exit == true`；
- APIServer 进程实际以 0 退出。

如何理解传输数字：
- `restore.h2d_bytes` 统计的是这一轮模型执行期间所有 host 到 device 的拷贝，不只是注意力 Page，还包括混合模型的递归（Gated-DeltaNet）状态。对 Qwen3.6-27B，两种布局下一份递归状态都是 153,944,064 字节，一个 784-token 的注意力 Page 是 51,380,224 字节，一个 16-token 的注意力 Page 是 1,048,576 字节。
- 历史短、全部放得进 GPU KV 时，没有任何注意力 Page 需要从 CPU 恢复，所以所有策略的恢复字节数相同。
- `prepare.h2d_bytes` 是阶段 1 的开销（恢复从上一轮带过来的递归状态）；`restore.h2d_bytes` 统计本轮其余部分。
- 开 `--gdn-device-handoff`（4.10 节）时，递归状态在 device 上移动：延续轮次的 `restore.h2d_bytes` 只统计恢复的注意力 Page，`restore.d2d_bytes` 统计状态拷贝（一次，加上每次跨 Mamba block 一次）；`prepare.h2d_bytes` 仍是 carry。
- 只有在历史确实需要把 Page 换出到 CPU 时，比较各策略才有意义；并且要把以上几部分分开报告。

这些文件可能包含 prompt 和工具输出，不要放进公开的源码发布包。Joined transfers 只证明机制行为，不证明任务质量或加速。回执里 `NOT_SYSTEM_VALID`、`cuda_qualified=false` 这类历史标签是有意保守的能力描述；评判一次运行，要看它的 READY、逐轮 native 证据、清理回执和任务评测。

## 9. 发布完整性与声明

`SOURCE_SHA256SUMS.txt` 和 `INSTALL_RELEASE.json` 描述交付的核心代码及其与干净上游的精确映射。仅供审阅的 `patches/v0.24.0.diff` 不会随每个 runtime 重新生成（仓库里没有生成它的工具）：它缺少后来的若干 vLLM 改动，包括解耦布局的改动，所以审阅时请以固定哈希的源码为准。`package.py` 按固定的文件清单生成核心代码的确定性源码归档；Slurm 和 SWE-bench 示例（以及一些较新的测试）不在这个清单里，需要它们时请用 Git checkout 或 `git archive`。READY 里的 `qualification_parent_manifest` 指向历史验收来源，不是对新运行的认证。本仓库保留 Apache-2.0 [LICENSE](LICENSE) 和上游声明。模型配置/tokenizer 来源及第三方条款见 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)。`examples/swebench/upstream/` 包含 PoC 的 runner 和 prompt，从 dram-backed-kv 仓库原样拷贝而来；它们遵循该仓库的条款。

## 历史

- **当前 runtime：本版本（当前的 `main`）。** 它在下面的 runtime 上加了三项需要显式开启的 admission 优化（4.10 节）、launcher 为其中一项计算的主机内存预算、更长的发布测试超时，第三个上下文选项 262,144 token（4.9 节），`goal_query` 的 Goal 文本证据（`goals.jsonl`，第 8 节），需要显式开启的纯文本模式（4.12 节），需要显式开启的 connector 闲置 CPU 池大小（4.9 节），三条需要显式开启的解码快速路径（4.11 节），MemoryAgentBench 的 kv-session 协议（6.6 节），面向 Terminal-Bench 2.0 agent 的 chat-text 协议（6.7 节），需要显式开启的、在 GPU 紧张时释放空闲 session 的 Page（4.13 节），Goal rewriter 每次调用的用时和需要显式开启的批量调用（4.10 节和第 8 节），以及第一阶段的 Gemma 4 31B IT 模型 profile，含五个策略、`--admission-overlap K`、CUDA graph、前两条解码快速路径和 kv-session 协议（4.14 节）。默认配置、策略、选择和默认的 agent 协议都没有变。
  - `--max-model-len 262144`（原来只有 65536 或 131072）是模型的原生上下文：不用 YaRN，没有新的固定值。单个请求的字节和 token 上限随固定的上下文上限缩放，只在 262144 时于启动时设置一次（`remote_ra_live_v2/context_limits.py`、`runtime_renderer.py`、`registered_add_protocol.py`、`vllm/v1/engine/core.py`）；`--dram-arena-gib` 过大时，launcher 的拒绝信息会给出放得下的最大 arena（`serve.py`）。它在一张 GH200 上跑过（Slurm 作业 3360375，runtime `e3efc22`，即合并解码快速路径之前的版本），已离线复核，token 完全一致（4.9 节）。
  - `--gdn-device-handoff` 在 device 上把 Gated-DeltaNet 状态从阶段 1 交给阶段 2，`--phase1-in-step` 把阶段 1 放进 decode 步里运行，`--admission-overlap K` 让最多 K 个 admission 同时进行（`remote_ra_live_v2/admission_flags.py`、`gdn_handoff.py`、`phase1_step.py`、`admission_overlap.py`、`capacity.py`、`goal_async.py`、`remote_ra_growing_worker/tensors.py`）。三个都关闭时，启动配置、证据文件和 vLLM 命令都不会多出任何内容。
  - launcher 的锁页主机内存预算计入 `--gdn-device-handoff` 为每个运行中 session 保留的第二份锁页 Gated-DeltaNet carry（`serve.py`，4.9 节）。
  - `check_git_release.py` 以 `OMP_NUM_THREADS=1` 运行测试，超时为 1800 秒（原为 840 秒，后为 1200 秒，直到本版本的 1,065 个测试用了 1,191 秒）；它不属于 runtime 文件。默认测试对最大的几个 CPU 端到端用例只跑简短版本，设置 `REMOTE_RA_SLOW_TESTS=1` 时运行全部用例（慢速测试，2.3 节）。
  - `goal_query` 服务把每轮 Goal 哈希背后的文本以及 Goal rewriter 的输入和输出记录到 `goals.jsonl`，native 文件和 `runtime.jsonl` 行指向它（第 8 节；`remote_ra_live_v2/goal_text.py`）。只是证据：Goal、选页和 rewriter 都不变，其他策略写出的文件与之前相同。
  - 每次 Goal rewriter 调用还记录它等待那一个 rewriter 进程的时间、调用内部的进程重启、调用用时和子进程报告的 token 计数（`goals.jsonl` 的 `rewrites[]`：`wait_ms`、`start_ms`、`call_ms`、`input_tokens`、`output_tokens`、`generation_latency_ms`），在 Goal 线程上构造的 Goal 在其 native 文件中增加总计 `goal_rewrite_wait_ms` 和 `goal_rewrite_call_ms`（第 8 节；`remote_ra_live_v2/goal_text.py`、`goal_async.py`、`remote_ra_http/evidence.py`）。只是证据：Goal、选页和 rewriter 调用都不变（CPU 测试 `test_goal_rewriter_timing.py`、`test_goal_text.py`）。
  - `--goal-rewriter-batch N`（需显式开启，需要 `--admission-overlap K ≥ N`）把排队中的 Goal rewriter 调用一起发给一个子进程，由它用一次批量的 `generate` 运行（4.10 节；`remote_ra_live_v2/goal_rewriter_batch.py`、`goal_rewriter_child.py`、`query_selection.py`、`private_endpoint.py`、`goal_text.py`、`serve.py`、`remote_ra_http/config.py`、`remote_ra_http/evidence.py`）。gate、请求、prompt、schema、解析、校验和回退都是上游的；批量输出在数值上可能与不批量时不同。已在 SWE-bench 作业 3404230 中以 N = 4 在 GPU 上运行过（4.10 节）。不开它时，launcher、启动、worker 和引擎的输出逐字节相同（identity probe）。
  - `--language-model-only` 把 vLLM 自己的参数传给 vLLM：不加载视觉塔，不做多模态预热，并在启动时固定并检查一次（`serve.py`、`remote_ra_http/config.py`、`remote_ra_http/evidence.py`、`remote_ra_live_v2/language_model_only.py`、`composition.py`、`worker_composition.py`）。不加它时，vLLM 命令、启动配置和证据都不变。在 GPU 上只在 `full` 的 Terminal-Bench dry run 中运行过（4.12 和 6.7 节）。
  - `--connector-pool-gib N` 把本 runtime 从不使用的 connector CPU 池设为 N GiB 而不是 32 GiB，launcher 的锁页主机内存预算也按 N GiB 计算；在启动时固定并检查一次（`serve.py`、`remote_ra_http/config.py`、`remote_ra_live_v2/connector_guard.py`、`composition.py`、`worker_composition.py`）。不加它时，vLLM 命令和启动配置都不变。在 GPU 上只在 `full` 的 Terminal-Bench dry run 中运行过（4.9 和 6.7 节）。
  - `--decode-fastpath` 去掉每个解码步里随"运行中的请求数 × 每个请求的 Page 数"增长的 host 工作（`remote_ra_live_v2/decode_fastpath.py`）；它在一张 GH200 上跑过（Slurm 作业 3366128，runtime `43fd9c3`，4.11 节）。`--decode-fastpath2` 需要它，把一步的分配更新合并为一次 worker RPC，再减少每个请求的 host 工作，并把前向之后的工作放到 logits 启动之后（`remote_ra_live_v2/decode_fastpath2.py`、`engine_boundary.py`、`allocation_owner.py`、`worker_composition.py`、`qwen.py`、`generation_guard.py`）；未经 GPU 验证。`--decode-fastpath3` 需要 `--decode-fastpath2`，进一步减少解码步里每个运行中请求的固定 host 工作（`remote_ra_live_v2/decode_fastpath3.py`、`allocation_owner.py`、`qwen.py`、`worker_rows.py`、`engine_boundary.py`）；未经 GPU 验证。不开这些开关时，启动配置、证据和 vLLM 命令都不变。
  - `REMOTE_RA_TORCH_PROFILE=<绝对目录>`（诊断用，4.11 节）传入 vLLM 自己的 `--profiler-config`（torch profiler 和该 trace 目录），服务因此和 stock vLLM 一样带有 `/start_profile` 和 `/stop_profile` 路由（`serve.py`）；与 `REMOTE_RA_STEP_PROFILE=1` 同时设置会被拒绝。startup config、其 pin 和服务的环境都不变；不设置时 vLLM 命令也不变（identity probe）。只在 CPU 上测试过（`test_torch_profile.py`）。
  - admission 工作在它自己的最新 commit 上（runtime `4eae441`），在一张 GH200 上跑了 5 个和 16 个 session 的 `goal_query`（Slurm 作业 3349504），已离线复核，token 完全一致（4.10 节）。作业 3349078（`observation_query`：这些开关在单个 session 下的一致性和 16 个 session 下的效果）也已离线复核。
  - MemoryAgentBench 的 kv-session 协议（`--protocol kv-session`，6.6 节：`remote_ra_live_v2/kv_session_protocol.py`、`remote_ra_http/kv_session_route.py`）和 `--release-idle-pages`（4.13 节：`remote_ra_live_v2/idle_release.py`）：`--release-idle-pages` 在 `full` 的 Terminal-Bench dry run 中于 GPU 上释放过空闲 session 的 Page（6.7 节），带释放和重新载入的 kv-session 协议在 MemoryAgentBench 作业 3397842 中运行过（runtime `0d37346`，4.13 节）。用 `--protocol mswea-xml` 且不开该选项时，launcher、启动、worker 和引擎的输出与 `ed57675` 逐字节相同（一致性探针，549 项）。
  - 不开 `--decode-fastpath2`、`--release-idle-pages` 和 `--connector-pool-gib`、使用 `--protocol mswea-xml` 时，launcher、启动、worker 和引擎的输出与上一个 `main` `4eaf159` 逐字节相同（一致性探针：555 + 219 项，0 处差异）。
  - 第一阶段的 Gemma 4 31B IT 模型 profile（4.14 节）：`install.py --model-profile gemma4-31b-it`、`serve.py` 的 Gemma 启动路径、Gemma 的轮次格式和 renderer（`remote_ra_live_v2/mswea_protocol.py`、`gemma4_renderer.py`）、vLLM KV 规划中的双池窗口布局（`remote_ra_live_v2/kv_layout.py`、`vllm/v1/core/kv_cache_utils.py`、`vllm/platforms/interface.py`、`vllm/v1/core/sched/scheduler.py`）、5,376 维投影的固定值以及滑窗与容量的计算（`selection_contract.py`、`model_profile.py`、`sliding_window.py`、`window_capacity.py`），滑窗在 engine 和 worker 中的生命周期（`remote_ra_native_growing/window_pins.py`、`remote_ra_live_v2/window_phase1.py`、`window_lifecycle.py`、`window_owner.py`、`window_worker.py`），Gemma 启动配置的检查（`remote_ra_http/config.py`）、admission gate 中的滑窗池（`capacity.py`），以及 `--release-idle-pages` 和 16 个 session 的选择性启动规则（`window_worker.py`、`idle_release.py`、`window_capacity.py`、`serve.py`）。Qwen release 不变。未经 GPU 验证。
  - Goal rewriter 的改动与 Gemma profile 合在一起（`test_gemma_goal_rewriter.py`）：Gemma profile 对每个策略都拒绝 `--goal-rewriter-batch`（尚未与窗口布局一起检查），`serve.py` 和启动检查（`remote_ra_http/config.py`）都拒绝；Gemma 服务和 `goal_query` 以外的策略都不写 rewriter 的用时字段。
  - Gemma profile 组里运行的四个策略（4.14 节，仅 CPU）：带 Goal 前向独立滑窗瞬态（2,048 token，pool 1 的 1,600 MiB）的 `goal_query`，以及使用窗口布局滑窗 ledger 的 `--admission-overlap K`（`kv_layout.py`、`window_capacity.py`、`window_phase1.py`、`capacity.py`、`idle_release.py`、`admission_overlap.py`、`window_worker.py`、`contract.py`、`remote_ra_http/config.py`、`serve.py`），16 个 session 的 `full` 的释放规则（`window_capacity.py`、`serve.py`），以及超过一次 native 前向时按每次最多 4,096 token 分块的 phase 1（`window_phase1.py`、`native_probe.py`、`window_owner.py`、`contract.py`、`sliding_window.py`）。Goal 的维护、rewriter、prompt、gate 和内容都没有改动。
  - Gemma profile 的服务配置（4.14 节，仅 CPU）：window 布局上的 `--cuda-graphs`、`--decode-fastpath` 和 `--decode-fastpath2`（`cuda_graphs.py`、`window_owner.py`、`window_rows.py`、`window_lifecycle.py`、`window_worker.py`、`engine_boundary.py`、`qwen.py`、`remote_ra_http/config.py`、`serve.py`），worker 只在 live 窗口上检查 vLLM runner 的滑窗行。Qwen release 不变。未经 GPU 验证。
  - chat-text 协议（`--protocol chat-text`，6.7 节：`remote_ra_live_v2/chat_text_protocol.py`、`remote_ra_http/chat_text_route.py`，以及它在 `runtime_renderer.py`、`frontend_registry.py`、`route_runtime.py`、overlay 的 `api_router.py`、`composition.py`、`controller.py`、`private_endpoint.py`、`capacity.py`、`query_selection.py`、`sessions.py`、`session_authority.py`、`remote_ra_http/config.py`、`evidence.py`、`shutdown.py` 和 `serve.py` 中的分支），在 GPU 上配合 `full` 跑过一次 16 个任务的 Terminal-Bench 2.0 dry run（6.7 节），其余只在 CPU 上；最初仅 Qwen（当时 Gemma profile 服务 `mswea-xml` 和 `kv-session`；后面一条加入）。
  - Gemma profile 的 MemoryAgentBench 路径（gemma-mab，4.14 节，仅 CPU）：使用 Gemma 格式、关闭 thinking 的 `--protocol kv-session`（`remote_ra_live_v2/kv_session_protocol.py`、`gemma4_renderer.py`、`remote_ra_http/kv_session_route.py`、`window_capacity.py`、`remote_ra_http/config.py`、`serve.py`）；该 profile 的 `--max-model-len 262144`；`--host-memory-shares 2`；选择性规则下窗口布局的 `request_pages`，vLLM 自己的启动检查现在按它而不是按一个上下文上限长的请求检查（`kv_layout.py`、`vllm/v1/core/kv_cache_utils.py`；原先会拒绝 32 GiB 以下、131,072 token 的 16 个 Gemma session）；以及前端 capture 现在接受 Gemma profile 的 source renderer（`frontend_registry.py`：它的精确类型检查拒绝 `Gemma4SourceRenderer`，Gemma 服务因此无法绑定前端）。Qwen release 上的 launcher、启动、worker 和 engine 输出与 `227dedb` 逐字节相同（identity probe：555 + 219 个 mswea-xml 项和 482 个 kv-session 与 session 池项，0 处差异）。
  - 本版本中首次同时出现的选项，仅 CPU：`test_integrate_flags.py` 在 XML、kv-session 和 chat-text 三种协议上（legacy 与解耦布局，并配合 `--connector-pool-gib`）以全部组合运行 `--decode-fastpath`、`--decode-fastpath2`、`--decode-fastpath3`、`--language-model-only` 和 `--release-idle-pages`，覆盖 launcher、dry run、启动检查、worker 挂接和 `ready.json`；运行时在 vLLM 形状的 CPU runner 上：开快速路径和 `--release-idle-pages` 的 16 个 session 压力运行与只开 `--release-idle-pages` 时相同，MemAgent kv-session session 和 16 个 chat-text session 在三条快速路径下运行，池压力下开 `--release-idle-pages` 和三条快速路径的 chat-text session 与一个从不释放的池相同。`test_gemma_flags.py` 把 Gemma 的选项放在一起运行（4.14 节）。无论其他选项如何都会拒绝：没有 `--decode-fastpath` 的 `--decode-fastpath2`、没有 `--decode-fastpath2` 的 `--decode-fastpath3`，以及 legacy 布局上的 `--release-idle-pages`；对 Gemma 还有 `--decode-fastpath3`、`--goal-rewriter-batch`、`--phase1-in-step`、`--gdn-device-handoff` 和 `--kv-session-thinking on`。`--language-model-only` 和 Qwen 的 CUDA graph 只在启动和启动检查中检查（CPU 替身不模拟它们）。
  - 开 `--admission-overlap K` 和 `--release-idle-pages` 时，在没有请求运行、也没有 admission 进行时被接纳的一轮，现在会先释放它需要的空闲 session 的 Page，与一次只做一个 admission 时相同；此前 ledger gate 不经检查直接接纳它而不释放，它的 phase 1 可能以 `insufficient native blocks for phase 1` 失败（`remote_ra_live_v2/idle_release.py`，两个 profile 共用；由 `test_gemma_flags.py` 发现，`test_idle_release.py` 检查顺序；在 Qwen 上，`test_integrate_flags.py` 在池压力下以 `--admission-overlap 4` 和 `--release-idle-pages` 运行 8 个 `goal_query` session（开与不开三条解码快速路径）以及这样单独的一轮，每一轮都与一次只做一个 admission、池从不释放的服务相同）。
  - 没有请求运行时，如果其他已打开的 session 钉住了 block，容量门控现在也会检查这一轮；它们没给它留下空间的一轮会等待（`CAPACITY_WAIT`），直到有 session 关闭或 `--release-idle-pages` 释放空闲 Page；此前它不经检查直接被接纳，在 GPU 上它的 decode 用掉了最后一个空闲注意力 block，vLLM 抢占了它，引擎因此退出（Slurm 作业 3398129：Terminal-Bench 2.0 dry run，Qwen3.6 `full` 走 chat-text，16 个 Terminus 2 trial，K = 20，32 GiB，未开 `--release-idle-pages`；空闲 session 钉住了 26,698 个 block 中除 69 个以外的全部，这一轮最多需要 272 个）。`--admission-overlap K` 下规则相同；Gemma 的窗口布局保持原规则，直到 integrate-1018（见下；`remote_ra_live_v2/capacity.py`，4.4 和 4.13 节；`test_adm_overlap.py` 覆盖两种门控，`test_chat_text_e2e.py` 覆盖此前被抢占、现在等待 session 关闭的 chat-text 一轮，`test_idle_release.py` 覆盖不开该选项的压力运行，它们现在停滞而不是失败；`test_ms16.py` 的紧张池改为 3,200 个 block，因为在 3,000 时单独的一轮只靠不经检查的接纳才放得下）。
  - 使用 Qwen release、不开 `--decode-fastpath3` 和 `--goal-rewriter-batch`、使用 `--protocol mswea-xml` 或 `kv-session` 时，launcher、启动、worker 和引擎的输出与上一个 `main` `227dedb` 逐字节相同（一致性探针：mswea-xml 555 + 219 项，kv-session 与 session 池 482 项，0 处差异）；带 Goal rewriter 的 `goal_query` 只多出 `goals.jsonl` 中每次调用的用时和计数（rewriter 探针：52 项，两处差异就是这些新键）。
  - Gemma 4 profile 上的 chat-text 协议（4.14 和 6.7 节，仅 CPU）：`remote_ra_live_v2/chat_text_protocol.py` 中 Gemma 的轮次格式（`GEMMA4_CHAT`；Qwen 的轮次格式即原有代码，作为 `QWEN_CHAT`），`gemma4_renderer.py` 中的初始段渲染和启动时的模板检查，固定 `chat_text_v1` 与该 profile 记录的 Gemma 启动配置（`remote_ra_http/config.py`、`serve.py`；最多同时打开 24 个 session）。每个 prompt 等于模板的渲染、并在每条过去的回复前保留空的 thought channel，去掉这些 channel 后等于 stock vLLM 的 prompt；第 0 阶段的 stock session（作业 3364469）10 轮全部复现。所有 Gemma 协议共用的修复：重建的滑窗需求扣除它的渲染所释放的窗口（`remote_ra_live_v2/window_capacity.py`；此前在窗口槽全部占满时，重建可能在没有释放它的全局缺口所需的空闲 session 的情况下被放行，然后永远等待；由 `test_gemma_chat_text_e2e` 发现；它和 `test_gemma_release`（XML 协议）现在都检查这一点，后者在不修复时失败）。使用 Qwen release 时，launcher、启动、worker 和引擎的输出与 `227dedb` 逐字节相同（一致性探针：mswea-xml 555 + 219 项，kv-session 与 session 池 482 项，0 处差异），Qwen 的 chat-text launcher、启动配置、前端、worker 记录、模板检查和 transcript 与 `0d37346` 逐字节相同（chat-text 探针：977 项，0 处差异）。
  - integrate-1018 合并了 `terminal-bench`（上文 `68d19b4` 在其他 session 钉住 block 时对单独一轮的容量门控检查，以及 Terminal-Bench 2.0 正式测试的配置，[docs/TERMINALBENCH_BENCHMARK.zh-CN.md](docs/TERMINALBENCH_BENCHMARK.zh-CN.md)）、`gemma-window-fix`（上文重建对自身窗口的扣除）、`gemma-mab`（上文用于 MemoryAgentBench 的 Gemma kv-session 协议）、`gemma-tb`（上文用于 Terminal-Bench 的 Gemma chat-text 协议）和 `torch-profile`（上文的 `REMOTE_RA_TORCH_PROFILE`）。Gemma profile 现在服务 `mswea-xml`、`kv-session`（关闭 thinking）和 `chat-text`；在选择性规则下，它的 chat-text 布局也固定 `request_pages`（`serve.py`、`remote_ra_http/config.py`、`gemma4_renderer.py`）。两项修复：Gemma 窗口布局的门控在没有请求运行、也没有 admission 在途时，只要另一个已打开的 session 钉住了全局 Page、tail 或窗口，就检查这一轮；在 `--release-idle-pages` 能做的释放之后仍放不下时，它等待（`CAPACITY_WAIT`），与 `68d19b4` 对 Qwen 门控所做的相同；此前它被不加检查地放行，可能用尽 block（致命的抢占）或在 vLLM 中永远等待（`remote_ra_live_v2/capacity.py`、`window_capacity.py`，4.14 节；`test_adm_overlap.py` 检查两种门控，`test_gemma_chat_text_e2e.py` 检查此前会被抢占、现在等待一个 session 关闭的单独一轮，`test_gemma_goal.py` 中没有 Goal 临时块的运行现在在门控处等待，而不是在 phase 1 中失败）。另外，chat-text 和 kv-session 的 CPU 端到端测试框架只要引擎有进展就等待一个请求，不再用在负载下会失败的固定 20 秒或 30 秒上限，`test_integrate_flags.py` 按初始段而不是绑定顺序标识它的 chat-text session（仅测试）。Qwen release 上的 launcher、启动、worker 和 engine 输出与 `0d37346` 逐字节相同（identity probe：555 + 219 个 mswea-xml 项、482 个 kv-session 与 session 池项和 977 个 chat-text 项，0 处差异；这些 probe 不覆盖其他 session 钉住 block 时的门控）。未经 GPU 验证：Gemma 的运行时功能（作业 3397695 中关闭选择性 runtime 运行的两个池的布局除外，4.14 节）、`--decode-fastpath3`、`--goal-rewriter-batch` 和 `REMOTE_RA_TORCH_PROFILE`。
  - 它还没有单独做干净安装验收。
- **上一个 runtime：commit `710bfc6718cf36d22f039151fb9c8d15ca2ab0f6`（2026-10-09）。** 它在下面的 runtime 上支持更大的服务，并加了新的 launcher 选项，每个都需要显式开启；默认服务不变（4.8 和 4.9 节）。
  - 每个服务最多 16 个 session（原为 8 个；`remote_ra_live_v2/sessions.py` 中的 `MAX_SESSIONS`），以及 session 池：`--max-running-sessions K` 最多注册 64 个任务，最多 K 个同时打开，每个在其第一个请求时打开，由 harness 关闭（`POST /v1/sessions/close`；4.4 和 6.5 节）。
  - `--cuda-graphs {off,piecewise,full,full-decode}`（4.8 节）。
  - `--max-tokens {1536,4096}`、`--max-model-len {65536,131072}` 和 `--dram-arena-gib`，终身请求上限 max(1,024, 256 × T)，以及 launcher 的锁页主机内存预算（4.9 节）。
  - 更小的 admission 开销：阶段 1 的记录不再用 `deepcopy` 拷贝，Selector V2 的打分和随机填充改为向量化计算，结果逐位相同（`remote_ra_live_v2/fast_selection.py`），prompt 每个 binding 只哈希一次（`remote_ra_native_growing/contract.py`）。
  - 在作业 3270721 的干净环境上（复用）和一张 GH200 上，用这条线上的 runtime（每项只跑了一次）：CUDA graph 在 Slurm 作业 3338893 和 3339448 中运行（runtime `f19f464`），16 个 session 在作业 3339019 中运行（`afffd2b`），用 `--cuda-graphs full` 的 16 个 session 和用 `full-decode` 的 session 池在作业 3343069 中运行（`613a7d5`）；见 4.4 和 4.8 节。在一张 GH200 上，131072 token 的上下文、4096 token 的回复和 arena 上限在作业 3348705 中运行（`710bfc6`，已离线复核，token 完全一致；4.9 节）。
  - 它还没有单独做干净安装验收。
- **更早的 runtime：commit `780fe54df8124d75ea76ad752bb6692427a89ed5`（2026-10-05）。** 它在下面的 runtime 上加了 query 策略的 `random_fill` 对照（第 5 节）。
  - `random_fill` 就是只改了填充步骤的 `observation_query`：剩余容量按 SplitMix64 / Fisher-Yates 顺序填满，而不是按余弦；顺序由启动时的种子（`--fill-seed`，默认 42）、session 序号和该 session 的请求序号决定。只有它的选页合同带 `random_fill` 键（`remote_ra_live_v2/selection_contract.py`、`remote_ra_live_v2/query_selection.py`、`serve.py`）。
  - 其余四个策略发送和写出的字节与下面的 runtime 完全相同（`test_random_fill_legacy.py`）。
  - 在作业 3270721 的干净环境上（复用），它以 16 token 的 Page 用 Mini-SWE-Agent 2.4.2 跑了 10 个 instance 的 SWE-bench pilot，每个策略一个作业：`full`、`observation_query`、`goal_query` 和 `random_fill`（Slurm 作业 3313802、3314010、3314319、3314623）。所有服务都通过。每个策略只跑一次，不构成 benchmark 证据（4.7 节）。
  - 它还没有单独做干净安装验收。
- **更早的 runtime：commit `8f1af6a15356a0b83c5e90fa4258e46bea0761e5`（2026-10-04）。** 它在下面的 runtime 上加了需要显式开启的解耦 KV 布局（4.6 节）。默认布局、策略、选择和 agent 协议都没有变。
  - `--page-tokens {784,16,32,64}` 和 `--kv-layout decoupled` 让全注意力 Page 和 Gated-DeltaNet 状态各用一个 block 池（`vllm/platforms/interface.py`、`vllm/v1/core/kv_cache_utils.py`、`kv_cache_coordinator.py`、`kv_cache_manager.py`、`sched/scheduler.py`、`vllm/v1/kv_cache_interface.py`、offloading connector 的文件、`remote_ra_live_v2/kv_layout.py`）。引擎一侧按池管理 block 归属并检查容量。
  - 在这种布局下，worker 在 GPU 上压缩 Page 向量，通过锁页 arena 成批地捕获和恢复注意力 Page，并用 token 区间而不是逐页摘要来指明封存的 Page（`remote_ra_live_v2/page_compaction.py`、`remote_ra_growing_worker/tensors.py`、`remote_ra_live_v2/allocation_owner.py`）。
  - 不加这些参数时，vLLM 命令、启动配置、固定哈希和证据都与下面的 runtime 相同（对照其源码的 CPU 测试）。
  - 在作业 3270721 的干净环境上（复用），它通过了三次 GPU 冒烟测试：`full` 在默认布局、784 解耦和 16 解耦下 token 完全相同（Slurm 作业 3308015），16 token Page 下的四个策略（作业 3308414），以及 24 GiB 下的 4 个 session（作业 3308482）。每项只跑了一次。
  - 它还没有单独做干净安装验收。
- **更早的 runtime：commit `902690ae2d8e9971bf4eb59ac9d38dc5c613dec0`（2026-10-03）。** 它在下面的 runtime 上加了两处改动。策略、选择和 agent 协议都没有变。
  - 停在 Goal 上的 admission 等待确认的时间上限改为 300 秒（原为 60 秒），请求租约改为 360 秒（原为 120 秒）（`registered_add_bridge.py`、`registered_add_protocol.py`、`remote_ra_live_v2/lifecycle.py`）。超时值不进入证据。
  - decode 路径上不再有每步或每个请求的哈希：binding 身份改为生成时发放的令牌，render 和 authenticate 不再对 prompt 做摘要，worker 不再每个 decode 步对一行做哈希和签名；只有写满 Page 的那一行 hidden state 才拷回 host（`remote_ra_native_growing/contract.py`、`remote_ra_live_v2/qwen.py`、`remote_ra_live_v2/worker_composition.py`）。生成的 token 不变。
  - 在作业 3270721 的干净环境上（复用），4 个 session 的合成运行通过（Slurm 作业 3300301）。Full 在 1–4 个并发请求时每个 decode 步为 69.1、72.0、72.0、74.6 ms，比用同样方法测量的原版 vLLM 每步慢 4.3–7.7 ms（作业 3291778）；每项只跑了一次。
  - 300 秒的 admission 等待在下面的 runtime 上（commit `b1e6988`）跑了多 session 的 SWE-bench pilot，使用 Mini-SWE-Agent 2.4.2：3 个 instance × 3 个策略、每个服务 3 个 session（Slurm 作业 3299922），以及 10 个 instance × 3 个策略、每个服务 5 个 session（作业 3300434）。所有服务都通过。每个 instance 只跑一次，不构成 benchmark 证据。
  - 它还没有单独做干净安装验收。
- **更早的 runtime：commit `40e8b3382f74f255c12b78181a6bb82c03df75c6`（2026-10-03）。** 它在下面的 runtime 上加了多 session 服务（见 4.4 节）。每个 session 的策略、选择和 agent 协议都没有变。
  - 一个服务最多服务 8 个 session，每个启动任务一个（`--task` 可以是任务列表）。每个请求按它的任务消息路由到对应的 session。admission 按到达顺序一次处理一个请求；decode 在各 session 之间成批进行（`max_num_seqs` = session 数）。不会发生抢占：GPU KV 池放不下的一轮会等待（`CAPACITY_WAIT`），直到正在运行的请求释放 block。
  - `--kv-cache-gib {8,16,24,32}` 设置 GPU KV 池大小（默认 8，即已验收的配置）。
  - `observation_query` 从不启动 Goal rewriter（见第 5 节）。
  - `goal_query` 在后台线程构建 Goal，rewrite 期间其他 session 继续 decode（`remote_ra_live_v2/goal_async.py`）。
  - 请求的 binding 身份只哈希一次，不再在每个调度步重新计算（`remote_ra_native_growing/contract.py`）。
  - 在作业 3270721 的干净环境上（复用），4 个 session × 6 轮、每轮 6,500 token observation、24 GiB 的合成运行通过（Slurm 作业 3292739、3295203、3299809）。作业 3299809 中，四个请求同时 decode 时的稳态每 token 延迟为 82、82、71 ms（full、observation_query、goal_query），作业 3292739 中为 117、111、102 ms。同一模型的原版 vLLM 0.24.0 在四个请求同时 decode 时每步为 67 ms（作业 3291778，用同样方法测量；它的整轮平均为 85 ms，因为其中还包括被其他请求的 prefill 拉长的 decode 步），因此四个请求时本 runtime 每步仍慢 4–15 ms。
  - 它还没有单独做干净安装验收。
- **更早的 runtime：commit `c1ad7fdc8209a370ef229defe44d4832cd40ad85`（2026-10-02）。** 它在下面的 runtime 上去掉了每个请求的额外开销。策略、选择和 agent 协议都没有变。
  - 文件和 tokenizer 的身份只在启动时校验一次，不再每个请求都校验（`tokenizer_semantics.py`、`remote_ra_linux/live_identity.py`、`runtime_renderer.py`）。
  - 不再每轮重新投影并重新哈希整个 Page 目录（`remote_ra_live_v2/query_selection.py`）。
  - 捕获到 CPU 的 KV Page 和 Gated-DeltaNet 状态副本改用身份令牌跟踪，不再对其字节做哈希（`remote_ra_growing_worker/tensors.py`）。
  - 在作业 3270721 的干净环境上（复用），四策略、十轮的合成运行通过（Slurm 作业 3290762），并在离线按 token 逐一核对一致：
    - Full 在第 3–9 轮的 admission 从约 4.0 秒（作业 3284520）降到 1.2 秒；
    - Goal-query 每轮比 Full 多出的 admission 从 1.6–1.8 秒降到 0.11 秒。
  - 它还没有单独做干净安装验收。
- **较旧的 runtime：commit `9430c695d60bd90ff4bc588f6316e6613681d3e2`（2026-10-01）。** 它在下面已验收的 runtime 上加了一个修复。
  - 修复的问题：引擎的 120 秒 admission 租约也会取消仍在生成的请求，而前端收不到任何输出。在约 100 ms/token 的速度下，超过约 1,160 token 的回复就可能遇到；一次测量运行确实遇到了（Slurm 作业 3276233）。现在正在运行的请求只会因完成、客户端取消或断连而结束（`registered_add_protocol.py`、`remote_ra_live_v2/lifecycle.py`、`test_lease.py`）。
  - 这个 runtime 还没有单独做干净安装验收。在作业 3270721 的干净环境上（复用），它通过了：
    - 四策略、十轮的合成运行，prompt 最长约 60K token（作业 3284520）；
    - 在一个 SWE-bench Verified instance 上，每个策略跑一次真实的 Mini-SWE-Agent 2.4.2 会话（作业 3284734），其中一个 125.6 秒的请求正常完成。
- **干净安装验收已通过（2026-09-30，Slurm 作业 3270721）。** commit `a564df36ffcc4194cedc0cd2db66bcfb641a7813` 的 runtime 字节在一张 NVIDIA GH200 上通过验收：
  - 在新环境里私有安装在全新的官方 vLLM 0.24.0 aarch64 wheel 之上（见 2.1 节）；
  - 四个全新服务，每个策略一个，各跑六轮不断增长的 Mini-SWE-Agent XML 请求，prompt 最长约 33K token，有 native 回执、joined drain 和自然关停；
  - 每个 token 只计算一次，query 策略在六轮中的三轮进行了选页；
  - 源环境在验收前后逐字节一致。

  这个 runtime：
  - 每一轮都按 dram-backed-kv PoC 的 V0 时序执行（第 5 节）：先在仍常驻的 Page 上 prefill observation，再由 selector 选出生成用的 Page，Gated-DeltaNet 状态保持连续；
  - 两个 query 策略遵循 PoC 的 Online Selector V2 合同；
  - 提供 PoC 的 Mini-SWE-Agent XML 协议（第 6 节）；
  - 使用 64k 上下文配置和 8 GiB GPU KV（第 1 节）。

  已验收的后端配置见[运行环境](#3-运行环境)。在上面的修复之前，之后只改文档的 commit 没有改变 runtime 字节，可以对照 `SOURCE_SHA256SUMS.txt` 核实。同一份 runtime 字节先在复用的环境上通过了 GPU dev smoke（Slurm 作业 3265689）。
- **上一个验收版本（OpenAI tool-call 协议）：** commit `4d90616996858d614b3dbfcfcbe8664f7aa0fe7c`，2026-09-29 通过干净安装验收（Slurm 作业 3261646），每个策略跑三轮不断增长的结构化 tool 历史。当前 runtime 不再提供该协议。
- 更早的一个 runtime 版本（尚未加入 `remote_ra_live_v2/query_selection.py` 的结构化 tool-call action 提取）在托管安装上通过了同样的四策略验收（Slurm 作业 3253330）。
- **以下内容尚未证明：** SWE-bench 任务成功、任务准确率、加速或传输节省，以及支持边界（第 1 节）之外的任何配置。合成 HTTP 验收只证明机制能运行，不是 benchmark 证据。
