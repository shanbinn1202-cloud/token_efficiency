# vLLM 0.24.0 的 Selective-prefix 策略

[English](README.md) | 简体中文

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

- **commit `780fe54df8124d75ea76ad752bb6692427a89ed5`（2026-10-05）：** 五个策略，Mini-SWE-Agent XML 协议（第 6 节），每个服务最多 8 个 session（4.4 节）。它是 `902690a` 加上解耦 KV 布局（commit `8f1af6a`，4.6 节）和 `random_fill` 对照。
- **默认配置不变：** 784 token 的 Page 放在一个共享 KV 池里，与 `902690a` 相同。16 token Page 的解耦布局需要显式开启。**对 `goal_query`、`observation_query` 和 `random_fill`，推荐加 `--page-tokens 16 --kv-layout decoupled`**（4.6 节）。
- 在 Slurm 作业 3270721 的干净环境上（复用），解耦布局通过了三次 GPU 冒烟测试（作业 3308015、3308414、3308482）；`780fe54` 以 16 token 的 Page 用 Mini-SWE-Agent 2.4.2 跑了 10 个 instance 的 SWE-bench pilot，每个策略一个作业（作业 3313802、3314010、3314319、3314623）；见 4.7 节。它还没有单独做干净安装验收；最近一次干净安装验收于 2026-09-30 通过（作业 3270721，runtime `a564df3`）。
- **以下内容尚未证明：** SWE-bench 任务成功、任务准确率、加速或传输节省，以及支持边界（第 1 节）之外的任何配置。每个 instance 只跑一次，不构成 benchmark 证据。
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
- 同步、纯文本、UniProc executor，TP=PP=DP=1，**每个启动任务一个 session，每个服务最多 8 个**（见 4.4 节）。已验收的运行每个服务只有一个 session。
- 启动器固定已验收的服务配置：

  | 配置 | 值 |
  | --- | --- |
  | `max_model_len` / `max_num_batched_tokens` | 65536 / 16384 |
  | Page 大小和 KV 布局 | 784 token，一个共享池（默认）；`--page-tokens {784,16,32,64}` 配合 `--kv-layout decoupled`，注意力和 Gated-DeltaNet 各用一个池（4.6 节）。在 GPU 上验证过的：784（默认）和 16 解耦；32 和 64 从未在 GPU 上运行过 |
  | GPU KV / CPU offload | 8 GiB（多个 session 时用 `--kv-cache-gib` 16、24 或 32）/ 32 GiB |
  | 执行方式 | eager（不使用 CUDA graph / torch.compile） |
  | Mamba cache | `align` |
  | GDN prefill | Triton |
  | 解析器 | `qwen3_xml` tool-call、`qwen3` reasoning |
  | 服务模型名 | `selective-prefix` |

  **prompt 加生成内容必须在 65536 token 以内。** 已验收的运行（作业 3270721）使用的就是这套配置，prompt 最长约 33K token。8 GiB GPU KV 用来放常驻工作集和一轮的新 token：64k 时 `full` 约保留 84 个 51.4 MB 的注意力 Page（4.3 GB），另有几个 154 MB 的 Gated-DeltaNet 状态 block。16 token 的 Page 下，同样 64K token 是 4,096 个 1 MiB 的注意力 Page（4 GiB）。

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

测试需要 `torch`（CPU 版即可；有几个测试模块会 import 它，其余的在没有它时跳过张量相关测试），以及 `PATH` 上的 `git`。2.1 节的基础环境两者都有；作业 3270721 就在那里运行了公开测试。在一台 Windows 台式机上（CPU 版 Torch 2.9.1、Python 3.12.7），本版本的 512 个测试用时约 9 分钟。其中一个测试会在候选文件的 Git 快照上把整套测试再跑一遍。

```sh
"$VENV/bin/python" -B -m unittest discover -s . -p 'test_*.py'
"$VENV/bin/python" -B check_git_release.py --ref HEAD
```

`check_git_release.py --ref HEAD` 会对该 commit 的 `git archive` 字节运行测试；发布 commit 前必须通过。

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
- 内存：模型（51.1 GiB）加 8 GiB KV 和可选的 0.6B Goal rewriter（约 2 GiB）可以放进一张 GH200（启动时空闲 94.5 GiB）；24 GiB KV 加五个 session 和 rewriter 也放得下，784 token 和 16 token 的 Page 都是如此（作业 3300434 和 3314319）。除模型加载外，主机还需要约 32 GiB 内存作为 CPU KV 池；解耦布局另有一个锁页（pinned）的 Page arena，16 token 的 Page 下每个 session 最多约 4 GiB（4.6 节）。已验证的作业申请了 200 GiB。

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

`--task` 接受 1 到 8 个 `{instance_id, problem_statement}` 对象组成的 JSON 列表，每个对象一个 session（`max_num_seqs` = session 数）。instance id 必须互不相同，任何一个 problem statement 都不能包含另一个。每个 harness 跑其中一个任务，连同一个端点；服务把每个请求路由到唯一一个 problem statement 出现在该请求任务消息（`messages[1]`）中的 session，每个 session 的行为和单任务服务一样（见第 5、6 节）。匹配不到任何任务或匹配到多个任务的请求会被拒绝。admission 按到达顺序一次处理一个请求，之后 decode 在各 session 之间成批进行。各 session 共用一个 GPU KV 池，不会发生抢占：最坏情况峰值放不下的一轮会等待（`CAPACITY_WAIT`），直到正在运行的请求释放 block。

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

pilot 对 session 数设了上限，保证即使每个 session 都停在 65,536 token 的上限，池也不会用尽：N − 1 个空闲 session 各占 I = 85 个 block，加上一轮的峰值 P = 91 个 block，再加 3 个 block 余量，必须放进可用 block 数 U，所以 N = 1 + ⌊(U − 91 − 3) / 85⌋（这是我们容量分析中 `full` 在 64K 时的占用；因为 session 重建会重新 prefill 每一页，所以对所有策略都按它算）。由此 8 GiB（U = 166）为 1 个 session，24 GiB（U = 500）为 5 个。同一规则给出 16 GiB 为 3 个、32 GiB 为 7 个；这两种大小**不在已验证的运行范围内**。

16 token Page 的 pilot 对注意力池用了同样的规则：停在上限的空闲 session 占 I = ⌈65,536 / 16⌉ + 1 = 4,097 个 block，一轮的峰值是 I + ⌈16,384 / 16⌉ = 5,121，余量是一次 1,536 token 的回复（97 个 block）加上 vLLM 的 1 % watermark（229）。24 GiB 下五个 session 需要 4 × 4,097 + 5,121 + 326 = 21,835 个 block，可用的有 22,911 个；六个放不下。Mamba 池每个 session 两个 slot，另加一个备用（4.6 节）。这条规则同样给出 8 GiB 为 1 个 session、16 GiB 为 3 个、32 GiB 为 7 个（**不在已验证的运行范围内**）。

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

**主机内存。** 解耦布局下，worker 把拷到主机的注意力 Page 放在一个锁页的 Page arena 里。它的容量是 S × (⌈65,536 / P⌉ + 1) 个 Page，每个 P × 65,536 字节，按需以 256 MiB 为单位分配：P = 16 时每个 session 最多约 4 GiB（`remote_ra_live_v2/worker_composition.py` 中的 `arena_pages`，`remote_ra_growing_worker/tensors.py` 中的 `PageArena`）。这是在第 3 节的 32 GiB CPU 池之外另加的。

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

向量：取模型最终 norm 之前的 last-token 隐藏状态（即 PoC 的最后一层 decoder 输出），用 PoC 的带种子 128 维 Rademacher 矩阵投影（seed 20260904，hidden size 5120 时 SHA-256 为 `20c2e1cf…87aa`），再 L2 归一化并逐向量 int8 量化。Goal query 是维护中 Goal 的 PoC `<retrieval_query>` JSON 文本；`--goal-query-variant candidate_entity_cap8` 在这段文本里只保留最后 8 个 files、symbols 和 tests。维护中的 Goal 使用 entity-drift 3；指定 `--goal-rewriter-model` 时使用 PoC 的常驻 Qwen3-0.6B rewriter。某一轮 rewrite 失败时与上游一样回退到确定性字段，并记录下来。

`observation_query` 不使用 Goal rewriter，因为它的排序只取决于 observation 向量：服务从不为它启动 rewriter，它的 Goal 只用确定性字段构建（仅用于证据），钉住 rewriter 的启动配置会被拒绝，给出的 `--goal-rewriter-model` 会在 `startup.json` 和 dry run 中记为未使用（`goal_rewriter`）。

**随机填充对照。** `--policy random_fill` 就是只改了第 5 步的 `observation_query`：阈值、容量、必保留页和 recent 页都相同，阶段 1 和确定性 Goal（仅用于证据）也相同；不使用 Goal rewriter（给出的 `--goal-rewriter-model` 记为未使用，钉住 rewriter 的配置会被拒绝），也不做 Goal forward。去掉必保留页和 recent 页之后剩下的候选页被随机排序，取前面的页填满容量；不用 query 向量、投影或分数。排序用 SplitMix64 加 Fisher-Yates（不做任何哈希），由启动时钉住的种子（`--fill-seed`，默认 42，即选页合同的 `random_fill` 键）、session 序号和该 session 的请求序号决定。每个选页生效的轮次，`query.selection.fill` 记录种子、session 序号、请求序号（`turn`）以及可选页数和抽中页数，`retrieved_pages` 按抽取顺序列出抽中的页。如果 `goal_query` 和 `observation_query` 不比这个对照好，它们的排序就没有带来有用的信息。与 `random50` 不同，它保留 query 策略的必保留页、recent 页和预算。

与 PoC 的已知差异，以及两种布局之间的差异：
- 默认 784 token 的 Page 下，预算都是整页数，live tail 是一个未写满的 Page。16 token 的 Page 下，预算、recent 配额和 live tail（不足 16 token）都与 PoC 相同。
- 必保留页覆盖 system + 任务 prompt 碰到的每一页。PoC 只保留完全落在其中的页，在 784-token 的页上可能丢掉任务的结尾；在 16 token 的页上两者最多差一页。
- 解耦布局下，worker 在 GPU 上一批地计算 int8 Page 向量；默认布局下在 CPU 上逐个计算。在 CPU 测试负载上两条路径得到的向量完全相同，但在 5,120 维的行上约 0.35 % 的向量有一个元素相差 ±1，可能改变近乎并列的排序（`docs/exp-decouple-pages/DESIGN.md`，Integration 第 4 项）。
- `random50` 保持固定的 floor(n/2) 随机抽取。16 token 的 Page 下它的种子来自 16 token block 的哈希，所以它的抽取结果不能与 784 token 的 Page 下比较。

## 6. Harness 接口

启动器的 `--protocol` 只接受 `mswea-xml`（默认值）。它逐 token 复现 dram-backed-kv PoC 服务端，是 SWE-bench 实验用的协议。

### 6.1 Mini-SWE-Agent XML 协议

这是 PoC 服务端的约定，Mini-SWE-Agent 2.4.2 以单个 `<mswea_bash_command>` 动作使用：

- **请求。** `POST /v1/chat/completions`，只看 `messages`：完整历史，每条是 `{"role", "content"}` 文本。依次是 system 消息、任务消息（必须包含冻结的 issue），然后是 agent 的 assistant 和 user 消息，最后一条必须是 user 消息（带 `<returncode>` 的 observation、格式错误提示等）。带 `tools` 的请求会被拒绝，消息里的其他字段会被忽略。
- **采样。** 服务端和 PoC 一样忽略客户端的采样字段：贪心解码，最多生成 1536 个 token，解码文本里出现 `</mswea_bash_command>` 或遇到 EOS/pad 就停。动作结束标记在引擎内部检测。
- **渲染。**
  - system 和任务消息走固定的 chat template，`enable_thinking=false`。
  - 之后每条消息都是 `<|im_start|>{role}\n{content}<|im_end|>\n`。
  - 生成头是 `<|im_start|>assistant\n<think>\n\n</think>\n\n`。
  - 历史里的回复会换回生成时的原始 token ID，按 `strip()` 之后的文本匹配，所以客户端裁掉首尾空白也没关系。
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

## 7. 用 Mini-SWE-Agent 2.4.2 跑 SWE-bench Verified

这是多 session SWE-bench pilot 的设置：3 个 Astropy instance × `full`/`goal_query`/`observation_query`、每个服务 3 个 session（作业 3299922），以及 10 个 instance × 同样 3 个策略、每个服务 5 个 session（作业 3300434），runtime 为 `b1e6988`，784 token 的 Page。之后同样 10 个 instance 又以 16 token 的 Page × `full`/`observation_query`/`goal_query`/`random_fill` 跑了一遍，每个策略一个作业（作业 3313802、3314010、3314319、3314623，runtime `780fe54`；结果见 4.7 节），agent 主机不变。每个服务都通过了服务端检查。每个 instance 只跑一次，不构成 benchmark 证据。这两次 pilot 没有跑 `random50`。

### 7.1 设置

| 设置 | 值 |
| --- | --- |
| Agent | Mini-SWE-Agent 2.4.2（litellm 1.103.1、datasets 5.0.1） |
| Runner | PoC 的 `swebench_runner.py`，参数 `--subset verified --split test --workers <session 数> --filter '^(<id>\|<id>…)$' --output <dir>` |
| 配置叠加 | `-c swebench_xml.yaml`（Mini-SWE-Agent 内置）`-c swebench_prompt_override.yaml`（PoC 的 prompt）`-c model_override.yaml` |
| 模型 | `examples/swebench/model_override.yaml`：`litellm_textbased`、`openai/selective-prefix`、`api_base` `http://127.0.0.1:<port>/v1`、占位 key、temperature 0.0、`max_tokens` 1536、`cost_tracking: ignore_errors` |
| 环境 | `docker`、`cwd` `/testbed`、超时 120 秒、`bash -c`、`PAGER`/`MANPAGER` 为 `cat`、`LESS` 为 `-R`、`PIP_PROGRESS_BAR` off、`TQDM_DISABLE` 1 |
| 限制 | `step_limit` 90、`cost_limit` 0 |
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
| `runtime.jsonl` | 每轮一行：响应 `id`、内部 request ID、token/顺序哈希、`candidate_page_count`、`selected_logical_pages`、`prepare`/`restore` 的 H2D 字节数、joined purge/drain、延迟。`stage_b` 记录本轮几何：已计算起点 C、候选终点、observation 区间、阶段 1 区间、阶段 2 起点、`retain`/`restore`/`release`、已计算终点 C′ = prompt + 生成 − 1、封存终点和 `rebuild`。Goal/Observation 的行另有 `query.selection`（`reason`、`capacity_pages`、`mandatory_pages`、`recent_pages`、`retrieved_pages`、合同与投影哈希）和 `query.goal`（变体、rewriter 是否尝试/成功/回退、缓存命中、冷 forward） |
| `native-<sha256(request id)>.json` | native plan（`selected`、`retain`/`restore`/`release`、run 页、tail 标记、阶段 1 lease）、worker frames（`prepare`、`phase1`、`phase1_drain`、`reserve`、`allocation_update`、`complete`）和 drain 回执 |
| `failures.jsonl` | 约定/运行时失败，带异常 `chain` 和 `traceback`（成功时为空） |
| `engine-cleanup.json`、`parent-cleanup.json` | 关停回执 |

解耦布局会增加一些字段；默认布局不写其中任何一个（`docs/exp-decouple-pages/WP4.md` 第 4 节）：
- `ready.json`：`kv_layout`（P、M、K）、`num_blocks_by_pool` 和 `free_blocks_by_pool`（注意力池在前，Mamba 池在后）；`worker-attached.json`：同样的内容，记为 `kv_pools`。
- 每个 `runtime.jsonl` 行和 native 文件：该请求 drain 之后的 `kv_pools.free_blocks_by_pool`。
- `engine-cleanup.json`：所有 session 关闭后的 `kv_pools`。运行干净时两个池都回到 READY 时的基线（除每个池的 null block 外全部空闲），已验证的运行检查过这一点。
- 封存 Page 的记录用 `token_span` [pP, (p+1)P] 而不是 `token_sha256` 来指明其 token，Page 向量使用 `compact_int8_v1` 形式 `[[128 个 int8], scale]`。

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
- 只有在历史确实需要把 Page 换出到 CPU 时，比较各策略才有意义；并且要把以上几部分分开报告。

这些文件可能包含 prompt 和工具输出，不要放进公开的源码发布包。Joined transfers 只证明机制行为，不证明任务质量或加速。回执里 `NOT_SYSTEM_VALID`、`cuda_qualified=false` 这类历史标签是有意保守的能力描述；评判一次运行，要看它的 READY、逐轮 native 证据、清理回执和任务评测。

## 9. 发布完整性与声明

`SOURCE_SHA256SUMS.txt` 和 `INSTALL_RELEASE.json` 描述交付的核心代码及其与干净上游的精确映射。仅供审阅的 `patches/v0.24.0.diff` 不会随每个 runtime 重新生成（仓库里没有生成它的工具）：它缺少后来的若干 vLLM 改动，包括解耦布局的改动，所以审阅时请以固定哈希的源码为准。`package.py` 按固定的文件清单生成核心代码的确定性源码归档；Slurm 和 SWE-bench 示例（以及一些较新的测试）不在这个清单里，需要它们时请用 Git checkout 或 `git archive`。READY 里的 `qualification_parent_manifest` 指向历史验收来源，不是对新运行的认证。本仓库保留 Apache-2.0 [LICENSE](LICENSE) 和上游声明。模型配置/tokenizer 来源及第三方条款见 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)。`examples/swebench/upstream/` 包含 PoC 的 runner 和 prompt，从 dram-backed-kv 仓库原样拷贝而来；它们遵循该仓库的条款。

## 历史

- **当前 runtime：commit `780fe54df8124d75ea76ad752bb6692427a89ed5`（2026-10-05）。** 它在下面的 runtime 上加了 query 策略的 `random_fill` 对照（第 5 节）。
  - `random_fill` 就是只改了填充步骤的 `observation_query`：剩余容量按 SplitMix64 / Fisher-Yates 顺序填满，而不是按余弦；顺序由启动时的种子（`--fill-seed`，默认 42）、session 序号和该 session 的请求序号决定。只有它的选页合同带 `random_fill` 键（`remote_ra_live_v2/selection_contract.py`、`remote_ra_live_v2/query_selection.py`、`serve.py`）。
  - 其余四个策略发送和写出的字节与下面的 runtime 完全相同（`test_random_fill_legacy.py`）。
  - 在作业 3270721 的干净环境上（复用），它以 16 token 的 Page 用 Mini-SWE-Agent 2.4.2 跑了 10 个 instance 的 SWE-bench pilot，每个策略一个作业：`full`、`observation_query`、`goal_query` 和 `random_fill`（Slurm 作业 3313802、3314010、3314319、3314623）。所有服务都通过。每个策略只跑一次，不构成 benchmark 证据（4.7 节）。
  - 它还没有单独做干净安装验收。
- **上一个 runtime：commit `8f1af6a15356a0b83c5e90fa4258e46bea0761e5`（2026-10-04）。** 它在下面的 runtime 上加了需要显式开启的解耦 KV 布局（4.6 节）。默认布局、策略、选择和 agent 协议都没有变。
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
