# 安装与卸载

[English](install.md) | 简体中文

第 1、2、3 和 9 节，integrate-1107 之前在 README 中，编号不变。各文档页面的节号互不重复；README [文档](../README.zh-CN.md#文档)一节的表格列出了每一节所在的页面。

## 1. 支持边界

- Linux、Python 3.12、从 wheel 安装的干净上游 **vLLM 0.24.0**、Transformers **5.13.0**、Tokenizers **0.22.2**。
- 模型：本地 **Qwen3.6-27B** 快照，revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`。元数据哈希见 `INSTALL_RELEASE.json`，本仓库不分发权重。三个固定哈希的元数据文件必须是普通文件，不能是 Hugging Face cache 符号链接。
- 第二个模型 profile：**Gemma 4 31B IT**，用它自己的 release（`install.py --model-profile gemma4-31b-it`）；它的 revision、固定的文件、安装命令和设置见 [4.14](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证) 节。本页其余部分讲的是 Qwen 的 release。
- 实测硬件：Linux aarch64 上一张 NVIDIA GH200 120GB。其他 GPU、架构和 backend 未测试。
- 默认同步（`--async-scheduling` 需显式开启，4.15 节和 [docs/async-sched/](async-sched/)）、纯文本、UniProc executor，TP=PP=DP=1，**每个启动任务一个 session，每个服务最多 32 个**（见 4.4 节；作业 3339019 和 3343069 在 GPU 上跑过 16 个 session；超过 16 个见 [docs/known-limits.zh-CN.md](known-limits.zh-CN.md) 2.1）；使用 `--max-running-sessions K` 时最多注册 256 个任务（integrate-1117 之前为 64 个），其中最多 K 个（K 最大为 32）同时运行（4.4 节；作业 3343069 在 GPU 上跑过 24 个任务、K = 8）。chat-text 服务（6.7 节）最多同时运行 K = 24 个 session，最多 1,024 个槽位（`--chat-text-slots`）；kv-session 服务（6.6 节）最多绑定 64 个客户端 session（`--kv-session-slots`）。已验收的运行每个服务只有一个 session。
- 加 `--no-speed-defaults` 时，启动器固定下面这套已验收的服务配置。不加时，还会应用 [4.15](reference.zh-CN.md#415-默认开启的速度选项) 节的默认速度选项：CUDA graph `full`、三个解码快速路径、`--deferred-completion-io`、`--language-model-only`、1 GiB 的 connector 池，以及开启 `--enable-thinking` 时 4096 token 的回复上限：

  | 配置 | 值 |
  | --- | --- |
  | `max_model_len` / `max_num_batched_tokens` | 65536 / 16384；`--max-model-len 131072` 选择 131072 token 的上下文（已在作业 3348705 中经 GPU 验证，4.9 节）；`--max-model-len 262144` 是模型的原生上限，选择 262144 token 的上下文（已在作业 3360375 中经 GPU 验证，4.9 节） |
  | 回复上限（`--max-tokens`） | 每轮最多生成 1536 个 token，即 PoC 的值；开启 `--enable-thinking` 时为 4096，除非加 `--no-speed-defaults`（4.15 节）；`--max-tokens 4096`（已在作业 3360375 中经过 GPU 验证，那里有 7 个回复达到上限；作业 3348705 中 16 个 session 的服务有 6 个回复达到上限，4.9 节） |
  | Page 大小和 KV 布局 | 784 token，一个共享池（默认）；`--page-tokens 16`、`32` 或 `64` 总是使用解耦布局，注意力和 Gated-DeltaNet 各用一个池（4.6 节）；`--kv-layout decoupled` 在 784 时也启用这种布局，仅用于验证。在 GPU 上验证过的：784（默认）和 16 解耦；64 运行过一次（作业 3366128，SPD1_PARTIAL_PASS_VERIFIED：一次解码测量中的 `full`）；32 从未在 GPU 上用 Qwen 运行过 |
  | GPU KV / connector 池 | 8 GiB（多个 session 时用 `--kv-cache-gib` 16、24、25 或 32；25 在作业 3461661 和 3461703 中运行过）/ vLLM 的 KV connector 池：默认 1 GiB（4.15 节），加 `--no-speed-defaults` 或 `--no-connector-pool-gib` 时为 32 GiB；runtime 从不向它存取数据 |
  | 执行方式 | 默认 `--cuda-graphs full`（4.15 节）；加 `--cuda-graphs off` 或 `--no-speed-defaults` 时为 eager（不使用 CUDA graph / torch.compile），即已验收的配置；`--cuda-graphs piecewise`、`full` 或 `full-decode` 已在 GPU 上验证（作业 3339448 和 3343069；作业 3338893 运行了这些模式但没有完成），但输出可能与 eager 不同（第 4.8 节） |
  | admission | 按到达顺序一次一个；`--gdn-device-handoff`、`--phase1-in-step` 和 `--admission-overlap K` 需要显式开启，已在作业 3349504 中用 `goal_query` 经 GPU 验证（4.10 节） |
  | Mamba cache | `align` |
  | GDN prefill | Triton |
  | 解析器 | `qwen3_xml` tool-call、`qwen3` reasoning |
  | 服务模型名 | `selective-prefix` |

  **prompt 加生成内容必须在 65536 token 以内**（即上下文上限；prompt 最多为上下文上限减去回复上限，回复上限为 1536 token 时是 64,000 token）。 已验收的运行（作业 3270721）使用的就是这套配置，prompt 最长约 33K token。8 GiB GPU KV 用来放常驻工作集和一轮的新 token：64k 时 `full` 约保留 84 个 49 MiB 的注意力 Page（4.0 GiB），另有几个 147 MiB 的 Gated-DeltaNet 状态。16 token 的 Page 下，同样 64K token 是 4,096 个 1 MiB 的注意力 Page（4.0 GiB）。

## 2. 安装与卸载

安装、installed-check 和启动服务必须使用同一个目标 Python。以下命令在本仓库目录执行，沿用快速上手里的占位变量。

### 2.1 基础环境

经过验证的环境就是干净安装验收作业 3270721 建立的那个环境；[历史](history.zh-CN.md#历史)中标明“在作业 3270721 的干净环境上（复用）”的每次运行，以及两次 SWE-bench pilot，都以只读方式复用了它。它在 Linux aarch64 上这样建立：

```sh
BASE_PYTHON=/abs/path/to/cpython-3.12/bin/python3.12
WHEEL=vllm-0.24.0-cp38-abi3-manylinux_2_28_aarch64.whl
mkdir -p "$RUNS"
curl -fL -o "$RUNS/$WHEEL" "https://files.pythonhosted.org/packages/9e/80/51a071305b4eed0f6f512dc1c1c6957cbb14ccce38db1be90ffcff2a2844/$WHEEL"
echo "700db71c3cf14697d42583521f38b12fac38db1e7a8ad062e8e4d63a5dadebd5  $RUNS/$WHEEL" | sha256sum --check
"$BASE_PYTHON" -I -B -m venv --copies "$VENV"
"$VENV/bin/python" -I -B -m pip --isolated install --index-url https://pypi.org/simple \
  --only-binary=:all: --no-cache-dir --no-compile --retries 0 --timeout 120 \
  --constraint examples/env/constraints-3270721.txt --report "$RUNS/pip-report.json" \
  "$RUNS/$WHEEL" transformers==5.13.0 tokenizers==0.22.2
```

- **基础解释器。** 作业 3270721 用 uv 管理的 CPython 3.12.13 作为 `BASE_PYTHON`：它的 `sys.base_prefix` 是 uv 默认目录 `~/.local/share/uv/python/` 下的 `cpython-3.12.13-linux-aarch64-gnu`，Triton 用的也是那里的头文件（`include/python3.12`）。它是怎样安装的没有记录，不在已验证的运行范围内。用 uv 的话，先 `uv python install 3.12.13`，再 `BASE_PYTHON="$(uv python find 3.12.13)"`，应该能得到同一个解释器（未测试）。其他带 `venv` 和头文件的 CPython 3.12 应该也可以，但未测试。
- **Wheel。** PyPI 上的官方文件，271,361,241 字节，SHA-256 如上。作业只从这个 URL 下载过一次，放在它的作业目录里；用 `curl` 下载到 Git checkout 之外的 `$RUNS` 只是下载方式之一。先 `mkdir -p "$RUNS"`，因为 pip report 也写到那里。
- **磁盘空间。** 作业 3270721 的环境有 66,550 个文件，9,071,166,786 字节（8.4 GiB）。2.4 节的 release 会复制其中已安装的 `vllm/` 树（4,404 个文件，613,445,655 字节，0.57 GiB），再加上 overlay。2.2 节的 checkpoint 另需：Qwen3.6-27B 约 51 GiB（服务加载 51.1 GiB 权重，第 3 节），Qwen3-0.6B 1.4 GiB（`model.safetensors`，1,503,300,328 字节）。请把 `$VENV`、`$MODELS`、`$RELEASE` 和 `$RUNS` 放在空间足够的文件系统上。
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

Qwen release 用两个 checkpoint，都以本地目录形式提供（Gemma 4 31B IT 的 checkpoint 及其安装见 [4.14](reference.zh-CN.md#414-gemma-4-31b-it第一阶段已在-cf2b28d-上通过-gpu-验证) 节）：

| 模型 | Revision | 使用者 | 如何固定 |
| --- | --- | --- | --- |
| [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | 每个服务（`install.py` 的 `--model-dir`） | `install.py`（仅检查和 `--apply` 都会）按 `INSTALL_RELEASE.json` 核对 `config.json`、`tokenizer.json`、`tokenizer_config.json` 的 SHA-256；不对权重做哈希 |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | 带 `--goal-rewriter-model` 的 `goal_query` | `serve.py` 在选页合同里固定解析后的目录和其 `config.json` 的 SHA-256；`install.py` 不检查它 |

Qwen3.6-27B 的三个固定哈希的元数据文件必须是普通文件，不能是 Hugging Face cache 符号链接（安装器拒绝符号链接）。已验证的运行没有记录 checkpoint 是怎样下载的，所以下面的命令**不在已验证的运行范围内**。2.1 节的基础环境里已经有 huggingface-hub 1.33.0 的 `hf` 命令行（`$VENV/bin/hf`）。请按这个路径调用：`$VENV/bin` 要到第 3 节才放进 `PATH`，而且 `$VENV` 里不能再安装别的东西（2.1 节）。用 `--local-dir` 时它应当写出普通文件；`find` 那一行负责检查：

```sh
"$VENV/bin/hf" download Qwen/Qwen3.6-27B --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --local-dir "$MODELS/qwen3.6-27b"
"$VENV/bin/hf" download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
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

测试是 `tests/test_*.py` 这些模块，以及它们的辅助模块 `tests/gemma4_testing.py` 和 `tests/testdata/`（自 integrate-1107 起，与发布仓库相同；此前它们在根目录）。测试需要 `torch`（CPU 版即可；有几个测试模块会 import 它，其余的在没有它时跳过张量相关测试），以及 `PATH` 上的 `git`。2.1 节的基础环境两者都有；作业 3270721 就在那里运行了公开测试。测试数量随每个分支变化：`run_tests.py --all` 会输出它（汇总中的 `run`），每个版本的 release check 写在[历史](history.zh-CN.md#版本记录)中。在台式机 CPU 上，逐个模块运行的 `run_tests.py --all` 约需 40 分钟，十个 worker 的 `check_git_release.py --ref HEAD` 约 9 到 11 分钟（integrate-1116，2026-11-16），快速默认集合约 5 分钟（`tests/FAST_MODULES.timings.json`）。其中一个测试，即 `test_installer` 的 `test_git_archive_head_preserves_pins_and_runs_full_suite`，在核心包文件（`package.PUBLIC_FILES`，含少数几个测试模块）的 Git 快照上运行 `check_git_release`。

```sh
"$VENV/bin/python" -B run_tests.py --all
"$VENV/bin/python" -B check_git_release.py --ref HEAD
```

`run_tests.py` 在仓库根目录下把每个模块依次作为 `python -s -B -m unittest <module>` 运行，`PYTHONPATH` 包含根目录和 `tests/`，使用一个 OpenMP 线程，最后输出一行 JSON 汇总（`status`、`run`、`failures`、`errors`、`skipped`、`failed_modules`）。不带参数时运行 `tests/FAST_MODULES.txt` 中的快速默认集合：在计时运行 `tests/FAST_MODULES.timings.json` 中用时不超过 10 秒的所有模块，再加上 `test_installer`（约一分钟，该计时运行中为 58 秒；CI 运行的就是它）。给出模块名时只运行这些模块（`"$VENV/bin/python" -B run_tests.py test_http test_lease`），`--timings FILE` 写出每个模块的秒数，`--fast-from FILE` 根据新的 `--all --timings` 运行重写快速列表。

`check_git_release.py --ref HEAD` 会对该 commit 的 `git archive` 字节运行测试，每个进程使用一个 OpenMP 线程（`OMP_NUM_THREADS=1`）；发布 commit 前必须通过。默认 `tests/` 中的每个测试模块在单独的进程中运行（在 `tests/` 中以该文件为 pattern 的 unittest discover，工作目录为 archive 根目录，`PYTHONPATH` 包含根目录和 `tests/`），同时运行十个（`--jobs`），少数计时要求严格的模块在其后单独运行；每个模块限时 1200 秒，总计限时 3600 秒；结果行仍以一行 `Ran N tests in S` 和 unittest 自己的 `OK` / `FAILED` 行结尾。`--serial` 是原来的单进程运行（一次 `unittest discover`，3600 秒）；`--record` 与之一样在一个进程中运行，配合 `--outcomes FILE` 写出每个测试 id 的结果，用于比较两种方式。

**慢速测试。** 常规测试（不设 `REMOTE_RA_SLOW_TESTS`，也就是 `check_git_release.py` 和 `run_tests.py` 运行的那一套）对最大的几个 CPU 端到端用例只跑简短版本：每个“布局 × 策略”矩阵只跑一两个（布局，策略）组合，与旧代码的对比只跑 `full` 和 `goal_query`，kv-session 的 MemAgent session 只跑 `goal_query`，16 个 session 只跑 `observation_query`，session 池只跑 10 个任务的用例，48 个随机选页目录只跑其中 16 个，Gemma 的端到端用例只跑 6 或 8 个 session（4.14 节），chat-text（Qwen 与 Gemma）只跑不含摘要和 aside 的 session（6.7 与 4.14 节），Gemma 的 `goal_query` Terminus 2 客户端只跑 4 个而不是 16 个（4.14 节）。设置 `REMOTE_RA_SLOW_TESTS=1` 会运行全部用例，外加 4.13 节 32,768 token 的压力用例。每次 push 前的验证都要跑这两遍测试和 `check_git_release.py --ref HEAD`：

```sh
REMOTE_RA_SLOW_TESTS=1 OMP_NUM_THREADS=1 "$VENV/bin/python" -B run_tests.py --all
```

**CI。** `.github/workflows/tests.yml` 在 GitHub Actions 上运行 `run_tests.py`（快速默认集合），触发条件为 push 到 `main` 和 `integrate-*`、pull request 以及手动触发：ubuntu-24.04、Python 3.12、CPU 版 torch、`transformers==5.13.0`、`tokenizers==0.22.2`、一个 OpenMP 线程、限时 30 分钟；它先按 `INSTALL_RELEASE.json`（`install.verify_package`）和 `examples/swebench/upstream/SHA256SUMS` 检查交付的源文件。

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
- 创建独立的 `<release>/source` 树并固定依赖身份。**它不会修改 site-packages、不安装依赖、不启动模型，也不启用功能。** 请预留完整 vLLM 副本的磁盘空间（约 0.6 GiB，2.1 节），复制期间保持源环境不变。
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

已通过的干净安装验收使用下面的设置，之后在它的环境上运行的作业也一样。请在 GH200 节点上启动 `serve.py` 的 shell 中设置它们，并在那里重新设置快速上手的占位变量。已验证的作业都是 Slurm 批处理作业（`sbatch`；benchmark 作业用的是 GH200 批处理分区 `ghx4`），`examples/slurm/serve_gh200.sh` 会自己设置这套环境（[4.5](running.zh-CN.md#45-gpu-作业slurm) 节）。如果要在 GPU 节点上开交互式 shell，请用你所在集群分配节点的方式；这不在已验证的运行范围内。启动器把环境传给服务，但去掉所有 `REMOTE_RA_*` 变量（只保留 step-profile 的一对 `REMOTE_RA_STEP_PROFILE` 和 `REMOTE_RA_STEP_PROFILE_EVERY`，并把 `REMOTE_RA_TORCH_PROFILE` 转成一个 vLLM 参数），也去掉 vLLM 的 policy runtime manifest 变量；然后设置它自己的 runtime 开关，并用 `python -I -S -B` 启动服务。

```sh
# 目标环境的 bin：其 Python 以及 JIT 构建使用的 Ninja。
export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
# 目标环境的 CUDA 13 runtime（libcudart.so.13），供隔离启动的进程使用。
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
# vLLM 在 $TMPDIR/<36 字节 uuid> 绑定 ZMQ IPC socket：路径要短（AF_UNIX 上限 107 字节）。
export TMPDIR=/abs/short/tmp
# 该目录必须存在，并且只有你自己可以访问。
mkdir -p -m 700 "$TMPDIR"
# 使用 native top-k/top-p sampler；FlashInfer sampler 需要 JIT 工具链和已预热的缓存。
export VLLM_USE_FLASHINFER_SAMPLER=0
# Triton 优先用 $CC 编译 launcher，否则用 gcc/clang；有些集群会导出 CC=cc。
unset CC VLLM_RPC_BASE_PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
```

说明：
- `len(TMPDIR) + 37` 不能超过 107 字节，所以 `TMPDIR` 最多 70 字节（验收作业 51 字节的 `TMPDIR` 得到 88）。路径更长时，第一个服务会在 READY 之前退出，报错 `zmq.error.ZMQError: ipc path ... is longer than 107 characters`。Slurm 脚本使用 `$RUN_ROOT/<job id>/tmp`，在 `umask 077` 下创建。
- 缓存为空时，vLLM 0.24 默认的 FlashInfer sampler 会在首次使用时 JIT 编译，需要 `nvcc` 和 Ninja，可能要几分钟。native sampler 可以避开这一步；`temperature=0` 时输出和 sampler 无关。冷缓存下的 FlashInfer sampler 不在已验收配置范围内。
- Triton 会在最初几个请求时 JIT 编译 kernel，需要可用的 `gcc`（或 `clang`）以及目标解释器的 Python 头文件。
- 每次运行可以使用独立的缓存目录（`HOME`、`XDG_CACHE_HOME`、`TRITON_CACHE_DIR` 等）。已验证的作业用的是全新的、作业本地的空缓存（见 `examples/slurm/serve_gh200.sh`）。
- 内存：模型（51.1 GiB）加 8 GiB KV 和可选的 0.6B Goal rewriter（连同其 CUDA context 约 2.5 GiB）可以放进一张 GH200（启动时空闲 94.5 GiB）；24 GiB KV 加五个 session 和 rewriter 也放得下，784 token 和 16 token 的 Page 都是如此（作业 3300434 和 3314319）。主机内存：如果一个服务锁页（pinned）主机内存的最坏情况可能超过 128 GiB，启动器会拒绝它。这个最坏情况包括 vLLM 的 connector 池（默认 1 GiB，加 `--no-speed-defaults` 时 32 GiB；runtime 从不使用它）、每个 session 一份 147 MiB 的 Gated-DeltaNet carry（开 `--gdn-device-handoff` 时两份），以及每个被捕获 Page 的主机副本：65536 token 的上下文下，两种布局都是每个 session 约 4 GiB（16 token 的 Page 下即解耦布局的锁页 Page arena，4.6 节；131072 token 的上下文下约 8 GiB，262144 下约 16 GiB，4.9 节）。请在此之外再给进程留约 32 GiB（`examples/slurm/serve_gh200.sh`）。2026-10-13 及以前的作业申请 `--mem=200g`：验收作业 3270721、SWE-bench 的 pilot，以及上下文作业 3348705 和 3360375。自 2026-10-14 起，benchmark 作业申请一张 GH200 的内存份额 `--mem=113560M`（4.5 节）。MemoryAgentBench 的 Qwen 服务从 G4 运行开始申请两个，`--mem=227120M`（作业 3405844 和 3405845；更早的作业 3397842 和 3397844 申请一个）。之后有几个测量作业按选择保留了 `200g`。

## 9. 发布完整性与声明

`SOURCE_SHA256SUMS.txt` 和 `INSTALL_RELEASE.json` 描述交付的核心代码及其与干净上游的精确映射。仅供审阅的 `patches/v0.24.0.diff` 不会随每个 runtime 重新生成（仓库里没有生成它的工具）：它缺少后来的若干 vLLM 改动，包括解耦布局的改动，所以审阅时请以固定哈希的源码为准。`package.py` 按固定的文件清单生成核心代码的确定性源码归档；Slurm 和 SWE-bench 示例（以及大多数测试模块）不在这个清单里，需要它们时请用 Git checkout 或 `git archive`。READY 里的 `qualification_parent_manifest` 指向历史验收来源，不是对新运行的认证。本仓库保留 Apache-2.0 [LICENSE](../LICENSE) 和上游声明。模型配置/tokenizer 来源及第三方条款见 [THIRD_PARTY_NOTICES.txt](../THIRD_PARTY_NOTICES.txt)。`examples/swebench/upstream/` 包含 PoC 的 runner 和 prompt，从 dram-backed-kv 仓库原样拷贝而来；它们遵循该仓库的条款。
