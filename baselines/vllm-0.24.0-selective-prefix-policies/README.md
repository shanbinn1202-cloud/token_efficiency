# Selective-prefix policies for vLLM 0.24.0

English | [简体中文](README.zh-CN.md)

> **Status of this revision.** GPU-validated on Qwen3.6-27B: the Mini-SWE-Agent XML protocol on SWE-bench Verified ([docs/SWEBENCH_BENCHMARK.md](docs/SWEBENCH_BENCHMARK.md)); Terminal-Bench 2.0 on `--protocol chat-text` with `full` and `observation_query` (16-task dry runs; [docs/TERMINALBENCH_BENCHMARK.md](docs/TERMINALBENCH_BENCHMARK.md#validation-status)); MemoryAgentBench on `--protocol kv-session` with `observation_query`, `goal_query` and the stock control (32-case runs; [docs/MEMORYAGENTBENCH_BENCHMARK.md](docs/MEMORYAGENTBENCH_BENCHMARK.md)). `goal_query` on chat-text is not in this revision (it is under review on the branch `tb-goal-query`). The Gemma 4 31B IT profile (section 4.14) is in development: its two-pool KV layout ran on one GH200 with the selective runtime off (job 3397695: with 32-token pages token-exact against the stock-vLLM replay on 10 of 10 trials; with 16-token pages a numerics or tiling difference of the TRITON_ATTN window layout from turn 3 on), but in this revision its runtime servers stop at startup on the GPU (jobs 3398198 and 3398614: "engine renderer and dynamic provider required"); the fix is on a development branch and not yet GPU-validated. `--decode-fastpath3` and the torch profile switch (`REMOTE_RA_TORCH_PROFILE`, section 4.11) are not GPU-validated.

Server-side prefix-page selection for multi-turn tool-using agents on vLLM 0.24.0. Each server runs one of five policies that decide which sealed history pages are restored from CPU memory on the next turn:

| `--policy` | Selection |
| --- | --- |
| `full` | Restore every eligible historical page (reference). |
| `random50` | Random half (⌊n/2⌋) of eligible pages, drawn with the fixed seed 42 and the prefix's last full KV block hash, so the draw depends on the prefix. |
| `goal_query` | Online Selector V2 (section 5): a maintained retrieval Goal from the frozen task and the latest action/observation becomes a model query vector; mandatory, recent and top-cosine pages fill the budget. |
| `observation_query` | Same selector; the query is the model's hidden state at the end of the newest observation. |
| `random_fill` | Control for the two query policies (section 5): the turn schedule, mandatory and recent pages and capacity of `observation_query`, with the remaining capacity filled in a seeded random order (SplitMix64 and Fisher-Yates, keyed by the seed, the session index and the session's request ordinal; `--fill-seed`, default 42). No Goal rewriter, no query vector. |

The harness talks to an ordinary OpenAI Chat Completions endpoint. It sends the complete, growing message history each turn; the policy, session, Goal, vectors and page choices never appear in HTTP. This repository contains the core runtime, installer and launcher, plus examples: a smoke client, a Slurm job script and the glue our SWE-bench pilots put around Mini-SWE-Agent (section 7). It also carries the PoC's SWE-bench runner and prompt that the pilots used (`examples/swebench/upstream/`). It contains no workloads, result files or trajectories. Internal `remote_ra_*` module names are implementation-private.

## Current runtime

- **This revision (the current `main`):** five policies, the Mini-SWE-Agent XML protocol (section 6), up to 16 sessions per server and a session pool of up to 64 tasks with at most K running (`--max-running-sessions K`, section 4.4), CUDA graphs (`--cuda-graphs`, section 4.8), a selectable context limit (65,536 tokens by default, 131,072, or the model's native 262,144) and reply bound and a cap on the pinned page arena (`--max-model-len`, `--max-tokens`, `--dram-arena-gib`, section 4.9), and three opt-in admission optimizations (`--gdn-device-handoff`, `--phase1-in-step`, `--admission-overlap K`, section 4.10), an opt-in decode fast path with optional second and third stages (`--decode-fastpath`, `--decode-fastpath2`, `--decode-fastpath3`, section 4.11), an opt-in text-only mode (`--language-model-only`, section 4.12), the MemoryAgentBench kv-session protocol (`--protocol kv-session`, section 6.6), the chat-text protocol for Terminal-Bench 2.0 agents (`--protocol chat-text`, section 6.7), the opt-in release of idle sessions' pages under GPU pressure (`--release-idle-pages`, section 4.13), opt-in batched Goal rewriter calls (`--goal-rewriter-batch N`, section 4.10) and a stage-1 Gemma 4 31B IT model profile (the five policies, `--admission-overlap K`, CUDA graphs, the first two decode fast paths, the MemoryAgentBench kv-session protocol and the chat-text protocol) whose server is complete and tested on the CPU; on the GPU only its two-pool layout has run, with the selective runtime off (job 3397695, section 4.14). It is commit `780fe54` (the decoupled KV layout of section 4.6 and the `random_fill` control) plus these options and a lower admission overhead; [History](#history) lists the steps.
- **Default configuration unchanged:** 784-token pages in one shared KV pool, eager execution, a 65,536-token context, 1,536-token replies and admissions one at a time; every option above is opt-in. **Recommended for `goal_query`, `observation_query` and `random_fill`: `--page-tokens 16 --kv-layout decoupled`** (section 4.6). **Admission optimizations:** `--admission-overlap 16` alone for `goal_query` at 16 sessions, none for the other policies (section 4.10). **Decode fast path:** `--decode-fastpath` for every policy (section 4.11).
- **On the GPU** (one GH200; each run used the runtime commit given in its section): the decoupled layout passed three smokes (jobs 3308015, 3308414, 3308482), and `780fe54` with 16-token pages served a 10-instance SWE-bench pilot with Mini-SWE-Agent 2.4.2, one job per policy (jobs 3313802, 3314010, 3314319, 3314623; section 4.7). 16 sessions ran in jobs 3339019 and 3343069, the session pool in job 3343069 (section 4.4), CUDA graphs in jobs 3338893, 3339448 and 3343069 (section 4.8), the 131072-token context, 4096-token replies and the arena cap in job 3348705, the 262144-token context in job 3360375 (section 4.9), the admission optimizations in jobs 3349078 and 3349504 (section 4.10), and the decode fast path in job 3366128 (section 4.11). The chat-text protocol passed 16-task Terminal-Bench 2.0 dry runs with `full` and with `observation_query` at runtime `68d19b4`, with `--connector-pool-gib 1`, `--language-model-only` and `--release-idle-pages` (section 6.7). The kv-session protocol ran in MemoryAgentBench jobs 3397844 and 3397842 (Qwen, runtime `0d37346`, section 6.6), and the readmission of released pages in job 3397842 (16 sessions, section 4.13). The SWE-bench runs of [docs/SWEBENCH_BENCHMARK.md](docs/SWEBENCH_BENCHMARK.md) at runtime `5b15b2f` (jobs 3404696 and 3404230, `goal_query`) ran `--decode-fastpath2` on the GPU, and job 3404230 `--goal-rewriter-batch 4` (section 4.10); `--decode-fastpath3` has run on the CPU only. This revision has not had its own clean-install acceptance; the last one passed on 2026-09-30 (job 3270721, runtime `a564df3`).
- **Not established:** SWE-bench task success, task accuracy, speedup or transfer savings, and support outside the tested boundary (section 1). One run per instance is not benchmark evidence.
- **Per-request encoding:** the server encodes each message with the loaded Hugging Face tokenizer only, as stock vLLM does; the pinned tokenizer's semantics are checked once at startup. Transformers 5.13 and the pinned `tokenizer.json` split some text differently (Devanagari vowel signs, found in a SWE-bench Django test file), and an earlier per-request comparison of the two refused such requests with HTTP 500 (job 3354504).
- **SWE-bench Verified benchmark:** [docs/SWEBENCH_BENCHMARK.md](docs/SWEBENCH_BENCHMARK.md) fixes the configuration of the 32-instance, 16-concurrent run of the four policies and the stock vLLM 0.24.0 control, with the checklist for aligning baselines, and records the run status: the two `goal_query` runs at runtime `5b15b2f` (jobs 3404696 and 3404230, 19 and 20 of 32 resolved), the other four servers pending.
- **Terminal-Bench 2.0 benchmark:** [docs/TERMINALBENCH_BENCHMARK.md](docs/TERMINALBENCH_BENCHMARK.md) fixes the configuration of the 89-task run with Harbor 0.21.0's Terminus 2 at 16 concurrent trials, for `full`, `observation_query`, `random_fill` and the stock vLLM 0.24.0 control, with the checklist for aligning baselines; it reports no result.
- **MemoryAgentBench benchmark:** [docs/MEMORYAGENTBENCH_BENCHMARK.md](docs/MEMORYAGENTBENCH_BENCHMARK.md) fixes the configuration of the 32-case run (8 cases per ability, at most 16 at a time) on the kv-session protocol with thinking off, for `observation_query`, `goal_query` and the stock vLLM 0.24.0 control, with its validation status and the checklist for aligning baselines; it reports no result.
- Every earlier runtime, its runs and job ids: [History](#history).

## Quickstart

### What you need

This repository plus a clean vLLM 0.24.0 environment is enough to install and serve:

1. **This repository** (a Git checkout).
2. **A Python 3.12 environment with the official `vllm==0.24.0` wheel, unmodified, plus exactly `transformers==5.13.0` and `tokenizers==0.22.2`.** `install.py` refuses other versions. A plain `pip install vllm==0.24.0` may resolve a different Transformers version, so pin both (section 2.1 has the validated recipe). Source or editable vLLM installs are unsupported: the installer's wheel-RECORD check fails on them.
3. **The Qwen3.6-27B weights** at revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, and **Qwen3-0.6B** only for `goal_query` with the Goal rewriter (or start `goal_query` with `--no-goal-rewriter`) (section 2.2).
4. **Linux with `gcc` or `clang` and the Python headers of that interpreter** (Triton compiles kernels at run time) and a **short `TMPDIR`** (section 3). Validated only on one NVIDIA GH200 (Linux aarch64). The installer does not check the architecture or the GPU, but other platforms are untested.

The installer never modifies the vLLM environment. It builds a separate release directory (a copy of the installed `vllm/` package plus this overlay), so the environment stays usable as plain vLLM. Serving needs no network: `serve.py` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

Running the SWE-bench harness additionally needs Mini-SWE-Agent 2.4.2 and a Docker host with the SWE-bench images. That is beyond install and serve (section 7).

### From nothing to a smoke request

The commands below use these placeholders; later sections reuse them. Use absolute paths.

```sh
REPO=/abs/path/to/vllm-0.24.0-selective-prefix-policies
VENV=/abs/path/to/vllm-env
MODELS=/abs/path/to/models
RELEASE=/abs/path/to/releases/selective-prefix
RUNS=/abs/path/to/runs
cd "$REPO"
```

1. Build the base environment `$VENV` (section 2.1) and get the model(s) into `$MODELS` (section 2.2).
2. Run the tests, check the environment, then create the release (sections 2.3 and 2.4):

```sh
"$VENV/bin/python" -B -m unittest discover -s . -p 'test_*.py'
mkdir -p "$(dirname "$RELEASE")" "$RUNS"
"$VENV/bin/python" -B install.py --site-packages "$VENV/lib/python3.12/site-packages" \
  --model-dir "$MODELS/qwen3.6-27b" --release-dir "$RELEASE"
"$VENV/bin/python" -B install.py --site-packages "$VENV/lib/python3.12/site-packages" \
  --model-dir "$MODELS/qwen3.6-27b" --release-dir "$RELEASE" --apply
```

3. On the GPU node, set the runtime environment (section 3), dry-run, start, wait for READY, send one request and stop with `SIGTERM` (sections 4.1 and 8):

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

`wait` must return 0 and the last command must print `CLEAN_SHUTDOWN`. In a Slurm job, `examples/slurm/serve_gh200.sh` does step 3 for you (section 4.5).

For `goal_query`, `observation_query` and `random_fill`, add `--page-tokens 16 --kv-layout decoupled` to both `serve.py` commands (section 4.6). The dry run then shows `--block-size 16 --mamba-block-size 784 --additional-config ...` at the end of the vLLM command:

```sh
python -B serve.py --release-dir "$RELEASE" --task examples/task.json --policy observation_query \
  --page-tokens 16 --kv-layout decoupled --output "$RUNS/obs" --port 8000
```

## 1. Supported boundary

- Linux, Python 3.12, clean upstream **vLLM 0.24.0** installed from its wheel, Transformers **5.13.0**, Tokenizers **0.22.2**.
- Model: local **Qwen3.6-27B** snapshot at revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. Metadata hashes are in `INSTALL_RELEASE.json`; weights are not distributed. The three pinned metadata files must be regular files, not Hugging Face cache symlinks.
- Tested hardware: one NVIDIA GH200 120GB on Linux aarch64. Other GPUs, architectures and backends are untested.
- Synchronous, text-only, UniProc executor, TP=PP=DP=1, **one session per startup task, up to 16 per server** (section 4.4; 16 sessions ran on the GPU in jobs 3339019 and 3343069); with `--max-running-sessions K`, up to 64 registered tasks of which at most K run at once (section 4.4; on the GPU: 24 tasks with K = 8 in job 3343069). The accepted run used one session per server.
- The launcher pins the qualified server configuration:

  | Setting | Value |
  | --- | --- |
  | `max_model_len` / `max_num_batched_tokens` | 65536 / 16384; `--max-model-len 131072` selects a 131072-token context (GPU-validated in job 3348705, section 4.9); `--max-model-len 262144`, the model's native limit, selects a 262144-token context (GPU-validated in job 3360375, section 4.9) |
  | reply bound (`--max-tokens`) | 1536 new tokens per turn, the PoC's value; `--max-tokens 4096` (GPU-validated in job 3348705, section 4.9) |
  | page size and KV layout | 784 tokens in one shared pool (default); `--page-tokens {784,16,32,64}` with `--kv-layout decoupled` for separate attention and Gated-DeltaNet pools (section 4.6). Validated on the GPU: 784 (default) and 16 decoupled; 32 and 64 never ran on a GPU |
  | GPU KV / CPU offload | 8 GiB (`--kv-cache-gib` 16, 24 or 32 for several sessions) / 32 GiB |
  | execution | eager (no CUDA graphs / torch.compile); `--cuda-graphs piecewise`, `full` or `full-decode` is GPU-validated (jobs 3338893, 3339448, 3343069), but its outputs can differ from eager (section 4.8) |
  | admission | one at a time in arrival order; `--gdn-device-handoff`, `--phase1-in-step` and `--admission-overlap K` are opt-in, GPU-validated with `goal_query` in job 3349504 (section 4.10) |
  | Mamba cache | `align` |
  | GDN prefill | Triton |
  | parsers | `qwen3_xml` tool-call, `qwen3` reasoning |
  | served model name | `selective-prefix` |

  **Prompt plus completion must fit in 65536 tokens** (the context limit; a prompt holds at most the context limit minus the reply bound, 64,000 tokens by default). The accepted run (job 3270721) used this profile with prompts up to about 33K tokens. 8 GiB of GPU KV holds the resident working set plus one turn's new tokens. At 64k tokens `full` keeps about 84 attention pages of 51.4 MB (4.3 GB), next to a few 154 MB Gated-DeltaNet state blocks. With 16-token pages the same 64K tokens are 4,096 attention pages of 1 MiB (4 GiB).

## 2. Install and uninstall

Use the same target Python for installation, installed-check and serving. Commands run from this repository with the Quickstart placeholders.

### 2.1 Base environment

The validated environment is the one clean-install acceptance job 3270721 created; every run that [History](#history) lists "on the reused clean environment of job 3270721", and both SWE-bench pilots, used it read-only. It was built like this on Linux aarch64:

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

- **Base interpreter.** Job 3270721 used uv's managed CPython 3.12.13 (`cpython-3.12.13-linux-aarch64-gnu`) as `BASE_PYTHON`. How it was installed (for example `uv python install 3.12.13`) is not recorded and not part of the validated runs. Any CPython 3.12 with `venv` and its headers should work but is untested.
- **Wheel.** The official PyPI file, 271,361,241 bytes, SHA-256 as above. The job downloaded that URL once; `curl` is only one way to do it.
- **Constraints.** `examples/env/constraints-3270721.txt` is the job's constraint file, byte for byte (194 pins, SHA-256 `1bc47da57ac424cf332c1e3df9f1c52137a77dd5bae3b95b1471c919607504a0`). It is what makes the install reproducible; without it pip resolves today's dependency versions, which were not validated.
- **What the job checked afterwards:** 195 distributions; Torch `2.11.0+cu130` (CUDA build 13.0); `$VENV/bin/ninja` 1.13.2; `$VENV/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13` present; the installed `vllm/` tree equal to the wheel's RECORD (4,404 files). Check Torch the same way:

```sh
LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib" "$VENV/bin/python" -I -B -c \
  "import torch; print(torch.__version__, torch.version.cuda)"
"$VENV/bin/ninja" --version
```

- **`pip check`** reports `nvidia-cusparselt-cu13 0.8.0 is not supported on this platform`. That is a defect of NVIDIA's aarch64 wheel (its installed `WHEEL` metadata carries the nonstandard tag `py3-none-manylinux2014_sbsa`; the library itself is aarch64 and RECORD-valid). The validated runs accepted exactly this one message.
- Keep the environment unchanged afterwards: no further `pip install`, no editable installs. The release binds the interpreter, `sys.prefix` and site path (section 2.4). Use `-B` (or `PYTHONDONTWRITEBYTECODE=1`) when you run its Python, as the validated jobs did.

### 2.2 Models

Two checkpoints, both as local directories:

| Model | Revision | Used by | Pinned how |
| --- | --- | --- | --- |
| [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | every server (`--model-dir` of `install.py`) | `install.py` (check-only and `--apply`) verifies the SHA-256 of `config.json`, `tokenizer.json` and `tokenizer_config.json` against `INSTALL_RELEASE.json`; it does not hash the weights |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | `goal_query` with `--goal-rewriter-model` | `serve.py` pins the resolved directory and the SHA-256 of its `config.json` in the selection contract; `install.py` does not check it |

The three pinned Qwen3.6-27B metadata files must be regular files, not Hugging Face cache symlinks (the installer refuses symlinks). The validated runs did not record how the checkpoints were downloaded, so the commands below are **not part of the validated runs**. With a recent `huggingface_hub` (`hf` CLI), `--local-dir` writes regular files:

```sh
hf download Qwen/Qwen3.6-27B --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --local-dir "$MODELS/qwen3.6-27b"
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir "$MODELS/Qwen3-0.6B-c1899de"
find "$MODELS/qwen3.6-27b" -maxdepth 1 -type l
(cd "$MODELS/qwen3.6-27b" && sha256sum config.json tokenizer.json tokenizer_config.json)
(cd "$MODELS/Qwen3-0.6B-c1899de" && sha256sum config.json model.safetensors tokenizer.json)
```

`find` must print nothing. Expected SHA-256 values:

| File | SHA-256 |
| --- | --- |
| Qwen3.6-27B `config.json` | `69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c` |
| Qwen3.6-27B `tokenizer.json` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| Qwen3.6-27B `tokenizer_config.json` | `5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b` |
| Qwen3-0.6B `config.json` | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| Qwen3-0.6B `model.safetensors` | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| Qwen3-0.6B `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |

In the validated runs the rewriter was a Hugging Face cache snapshot of that revision (symlinks are fine for the rewriter); the 0.6B rows come from those runs' rewriter checks. The rewriter's resolved path is part of the selection contract, so use the same path for every server of one comparison.

### 2.3 Tests

The suite needs `torch` (CPU is fine; several test modules import it, others skip their tensor tests without it) and `git` on `PATH`. The base environment of section 2.1 has both; job 3270721 ran the public tests there. This revision has 1,242 tests; on a Windows desktop with CPU Torch 2.9.1 and Python 3.12.7 the 512 tests of commit `780fe54` took about 9 minutes. One of them repeats the whole suite on a Git snapshot of the candidate files.

```sh
"$VENV/bin/python" -B -m unittest discover -s . -p 'test_*.py'
"$VENV/bin/python" -B check_git_release.py --ref HEAD
```

`check_git_release.py --ref HEAD` runs the suite on the `git archive` bytes of that commit, with one OpenMP thread (`OMP_NUM_THREADS=1`) and an 1800 s timeout; it must pass before publishing a commit.

**Slow pass.** The default suite, and with it `check_git_release.py`, runs a short variant of the largest CPU end-to-end cases: one or two (layout, policy) pairs of each layout-by-policy matrix, `full` and `goal_query` in the comparisons with former code, the kv-session MemAgent sessions with `goal_query`, sixteen sessions with `observation_query` only, the 10-task session pool, 16 of the 48 random selection catalogs, the Gemma end-to-end cases on six or eight sessions (section 4.14) and the chat-text sessions without summaries and asides, Qwen's and Gemma's (sections 6.7 and 4.14). `REMOTE_RA_SLOW_TESTS=1` runs every case, plus the 32,768-token pressure case of section 4.13. Every validation before a push runs both passes and `check_git_release.py --ref HEAD`:

```sh
REMOTE_RA_SLOW_TESTS=1 OMP_NUM_THREADS=1 "$VENV/bin/python" -B -m unittest discover -s . -p 'test_*.py'
```

### 2.4 Install the release

```sh
mkdir -p "$(dirname "$RELEASE")" "$RUNS"
"$VENV/bin/python" -B install.py --site-packages "$VENV/lib/python3.12/site-packages" \
  --model-dir "$MODELS/qwen3.6-27b" --release-dir "$RELEASE"
# Check-only above; explicit creation below. Destination must not exist.
"$VENV/bin/python" -B install.py --site-packages "$VENV/lib/python3.12/site-packages" \
  --model-dir "$MODELS/qwen3.6-27b" --release-dir "$RELEASE" --apply
```

Check-only prints `CHECK_ONLY_PASS`, `--apply` prints `INSTALL_PASS` (both with `"site_packages_written": false`). What the installer does:
- It verifies the whole installed `vllm/` tree against `vllm-0.24.0.dist-info/RECORD`. Missing, extra, unhashed or changed files fail before anything is copied. RECORD proves wheel integrity, not publisher authenticity, so install the base wheel from a trusted source. Editable/source checkouts without a RECORD are unsupported.
- It checks the exact Transformers/Tokenizers/vLLM versions and the three pinned model metadata hashes (section 2.2).
- A 120-path upstream map binds the overlay (27 modified, 91 unchanged anchors, 2 absent targets). All distributed files are hash-checked.
- It creates an independent `<release>/source` tree and pins its dependencies. **It never edits site-packages, installs dependencies, starts a model or enables the feature.** Allow disk space for a full vLLM copy and keep the source environment unchanged while copying.
- The manifest records the exact Python version, implementation, architecture, interpreter, `sys.prefix` and site path. A changed environment needs a new release.
- On failure it removes only its own incomplete release. Existing releases and source drift are rejected. Preserve the error output and use a new release path next time; do not run `restore.py` on a release without its final index.

### 2.5 Uninstall

Stop every process using the release first.

```sh
"$VENV/bin/python" -B restore.py --release-dir "$RELEASE"
"$VENV/bin/python" -B restore.py --release-dir "$RELEASE" --apply --processes-stopped
```

Uninstall checks the release and renames it to an adjacent `.uninstalled-<token>` recovery directory; nothing is deleted. Rolling back to plain vLLM means stopping this service and using the untouched original environment. Do not relocate a live release: its manifest binds absolute paths.

## 3. Runtime environment

The accepted clean-install run used the settings below, and so did the later jobs on its environment. Set them in the shell that starts `serve.py`; the launcher passes the environment to the server (it only removes `REMOTE_RA_*` and sets its own runtime flags).

```sh
# The target environment's bin: its Python and the Ninja used by JIT builds.
export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
# CUDA 13 runtime of the target environment (libcudart.so.13) for the isolated launcher.
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
# vLLM binds ZMQ IPC sockets at $TMPDIR/<36-byte uuid>: keep the path short (AF_UNIX limit 107 bytes).
export TMPDIR=/abs/short/tmp
# Native top-k/top-p sampler; the FlashInfer sampler needs a JIT toolchain and a warm cache.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Triton compiles launchers with $CC, else gcc/clang; some clusters export CC=cc.
unset CC VLLM_RPC_BASE_PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
```

Notes:
- `len(TMPDIR) + 37` must stay at or below 107 bytes. A longer path makes the first server exit before READY with `zmq.error.ZMQError: ipc path ... is longer than 107 characters`.
- With an empty cache, vLLM 0.24's default FlashInfer sampler JIT-compiles on first use. That needs `nvcc` and Ninja and can take minutes. The native sampler avoids it, and with `temperature=0` the output does not depend on the sampler. The FlashInfer sampler on a cold cache is outside the accepted profile.
- Triton JIT-compiles kernels on the first requests. It needs a working `gcc` (or `clang`) and the Python headers of the target interpreter.
- Separate cache directories per run (`HOME`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, …) are fine. The validated jobs used empty, job-local caches (see `examples/slurm/serve_gh200.sh`).
- Memory: the model (51.1 GiB) plus 8 GiB KV and the optional 0.6B Goal rewriter (about 2 GiB) fit one GH200 (94.5 GiB free at start); so did 24 GiB KV with five sessions and the rewriter (job 3300434), at 784-token and at 16-token pages (job 3314319). Allow about 32 GiB of host RAM for the CPU KV pool in addition to model loading; the decoupled layout adds a pinned page arena of up to about 4 GiB per session at 16-token pages (section 4.6; about 8 GiB at a 131072-token context and 16 GiB at 262144, section 4.9). The validated jobs requested 200 GiB.

## 4. Start the policies

### 4.1 One server

Each server fixes **one policy and its frozen task(s)**. A task file has exactly `instance_id` and `problem_statement`; the problem statement must appear verbatim in the task message (the first user message) of every request. The output directory must not exist yet (its parent must). Inspect the dry run first; only `--start` loads the model.

```sh
python -B serve.py --release-dir "$RELEASE" \
  --task examples/task.json --policy full --output "$RUNS/full" --port 8000
python -B serve.py --release-dir "$RELEASE" \
  --task examples/task.json --policy full --output "$RUNS/full" --port 8000 --start
```

`goal_query` needs an explicit Goal-rewriter choice: a local Qwen3-0.6B checkpoint directory (the PoC setting), or `--no-goal-rewriter` for deterministic Goal fields only. `observation_query` and `random_fill` need none; they accept `--goal-rewriter-model` but never use it (section 5). The other selector options are optional; section 5 lists them. `serve.py --help` lists every flag. For the query policies and `random_fill` we recommend 16-token pages (section 4.6):

```sh
python -B serve.py --release-dir "$RELEASE" \
  --task examples/task.json --policy goal_query --output "$RUNS/goal" --port 8000 \
  --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --page-tokens 16 --kv-layout decoupled --start
```

`--start` replaces the launcher with the APIServer process, so the launcher's PID is the server PID. The server binds only `127.0.0.1:<port>` (section 6.4 for other hosts). Startup takes a few minutes: model load, KV allocation and the first Triton compilations. In the validated jobs READY came 204–231 s after the start for the first server of a job and 105–146 s for later ones.

**READY** means `<output>/ready.json` exists, parses completely, and has `native_http_ready == true`, `evidence_kind == "NATIVE_HTTP_RUNTIME"`, the requested `policy`, and `runtime_manifest == SHA256(raw <release>/runtime-manifest.json)`. Its task/config hashes must equal `startup.json`. The file is created before it is fully written, so retry on a parse error. `/v1/models` answering is not READY. `examples/slurm/server_checks.py ready <output> <policy> <release>` checks exactly this (exit 0 READY, 1 not yet, 2 mismatch).

**Stop** with `SIGTERM` to that PID only, never the process group. Wait for the process to exit with code 0, then check the cleanup receipts (section 8; `examples/slurm/server_checks.py cleanup <output>`). A killed or timed-out shutdown is not a clean shutdown.

### 4.2 All four policies on one GPU (sequential)

One GH200 holds one server at a time. Use a fresh server and a fresh output directory for every policy (and every task). This mirrors the accepted run; replace `HARNESS` with your client:

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

The next server must start on an idle GPU with no leftover process. The accepted run needed about 18 minutes for four sequential servers of six turns each, up to about 33K prompt tokens.

The accepted run used 784-token pages for every policy. To follow the recommendation of section 4.6, add `--page-tokens 16 --kv-layout decoupled` to the `goal_query` and `observation_query` servers (and to `random_fill`, if you add the control to the loop). `full` produced identical tokens in both layouts in the exact smoke (section 4.7); the 16-token-page SWE-bench pilot nevertheless ran every server, `full` included, with the flags.

### 4.3 Several GPUs

Running one policy per GPU in parallel (`CUDA_VISIBLE_DEVICES=<n>`, a distinct `--port` and a distinct `--output` per server) follows from the one-server-per-GPU design but was **not** part of the validated runs. Never point two harnesses with the same task at one server.

### 4.4 Several sessions on one server

```sh
python -B serve.py --release-dir "$RELEASE" \
  --task /abs/path/to/tasks-g0.json --policy observation_query --kv-cache-gib 24 \
  --page-tokens 16 --kv-layout decoupled --output "$RUNS/obs-g0" --port 8000 --start
```

`--task` takes a JSON list of 1 to 16 `{instance_id, problem_statement}` objects, one session each (`max_num_seqs` = sessions; the GPU runs, up to 16 sessions, are in the table below). Instance ids must be distinct and no problem statement may contain another. Each harness runs one of these tasks against the same endpoint; the server routes every request to the one session whose problem statement its task message (`messages[1]`) contains, and each session behaves as a one-task server (sections 5 and 6). A request that matches no task, or several, is rejected. Admission runs one request at a time in arrival order (unless `--admission-overlap K`, section 4.10), then decode is batched across sessions. The sessions share one GPU KV pool and nothing is preempted: a turn whose worst-case peak does not fit waits (`CAPACITY_WAIT`) until running requests free blocks.

Size `--kv-cache-gib` so that every session's resident pages plus one turn fit at the longest context you expect. Validated settings ("usable" = attention blocks − 1). At 784-token pages one page is 51,380,224 bytes and one pool holds both the attention pages and the Gated-DeltaNet states. In the decoupled layout at 16-token pages one page is 1,048,576 bytes, and the Gated-DeltaNet states have their own pool (section 4.6):

| `--kv-cache-gib` | Sessions per server | Pages | Run | Evidence |
| --- | --- | --- | --- | --- |
| 8 | 1 | 784 | Clean-install acceptance: 4 policies × 6 turns, prompts up to about 33K tokens | job 3270721; vLLM: GPU KV cache size 121,605 tokens |
| 24 | 4 | 784 | Synthetic: 4 sessions × 6 turns of 6,500-token observations, `full`/`observation_query`/`goal_query` | job 3299809 (also 3292739, 3295203, 3300301); pool 501 blocks, 500 usable |
| 24 | 3 | 784 | SWE-bench pilot, 3 instances × 3 policies, one group | job 3299922; `kv_usable_blocks` 500, sessions cap 5 |
| 24 | 5 | 784 | SWE-bench pilot, 10 instances × 3 policies, two groups of 5 | job 3300434; `kv_usable_blocks` 500, sessions cap 5; vLLM: GPU KV cache size 364,817 tokens |
| 8 | 1 | 16, decoupled | GPU smoke: `full`, 10 turns up to about 60K prompt tokens | job 3308015; 7,702 attention + 10 Mamba blocks |
| 24 | 4 | 16, decoupled | GPU smoke: 4 sessions × 6 turns of 6,500-token observations, `full`/`goal_query` | job 3308482; 23,205 attention + 28 Mamba blocks |
| 24 | 5 | 16, decoupled | SWE-bench pilot, 10 instances × 4 policies, two groups of 5 | jobs 3313802, 3314010, 3314319, 3314623; 22,912 attention + 34 Mamba blocks (11 Gated-DeltaNet slots), 22,911 usable, sessions cap 5; vLLM: GPU KV cache size 366,592 tokens |
| 24 | 16 | 16, decoupled | Synthetic: 16 sessions × 6 turns of 2,600-token observations, `full`/`observation_query`/`random_fill`, eager | job 3339019; 19,682 attention + 100 Mamba blocks (33 Gated-DeltaNet slots), 19,681 usable; vLLM: GPU KV cache size 314,912 tokens |
| 24 | 16 | 16, decoupled | Synthetic: the same workload, `observation_query`, eager and with `--cuda-graphs full` | job 3343069; pools as in job 3339019 |
| 24 | 8 running of 24 tasks | 16, decoupled | Synthetic session pool: 24 tasks × 6 turns of 3,000-token observations, `--max-running-sessions 8`, `observation_query`, `--cuda-graphs full-decode` | job 3343069; 22,031 attention + 52 Mamba blocks (17 Gated-DeltaNet slots), 22,030 usable; vLLM: GPU KV cache size 352,496 tokens |

The pilots capped sessions so that the pool can never run out even when every session sits at the 65,536-token limit: N − 1 idle sessions at I = 85 blocks plus one turn's peak P = 91 blocks plus a margin of 3 must fit the usable blocks U, so N = 1 + ⌊(U − 91 − 3) / 85⌋ (the `full` footprint at 64K from our capacity analysis, used for every policy because a session rebuild re-prefills every page). That gives 1 session at 8 GiB (U = 166) and 5 at 24 GiB (U = 500). The same rule gives 3 at 16 GiB and 7 at 32 GiB; those sizes are **not part of the validated runs**.

The 16-token-page pilot applied the same rule to the attention pool: an idle session at the limit holds I = ⌈65,536 / 16⌉ + 1 = 4,097 blocks, one turn's peak is I + ⌈16,384 / 16⌉ = 5,121, and the margin is one 1,536-token reply (97 blocks) plus vLLM's 1 % watermark (229). Five sessions at 24 GiB need 4 × 4,097 + 5,121 + 326 = 21,835 of the 22,911 usable blocks; six do not fit. The Mamba pool holds 2 slots per session plus a spare (section 4.6). The rule again gives 1 session at 8 GiB, 3 at 16 GiB and 7 at 32 GiB (**not part of the validated runs**).

**Up to 16 sessions.** `MAX_SESSIONS` in `remote_ra_live_v2/sessions.py` is the one cap that every per-process session limit on the path reads; it was 8 before. What grows with the number of sessions S:

- The decoupled layout's Mamba pool, 1 + 3(2S + 1) blocks: 100 blocks (4.78 GiB) at S = 16, reserved before the attention pool. The launcher refuses a budget whose attention pool cannot hold one session's turn at the 65,536-token limit (⌈65,536 / P⌉ + 1 + ⌈16,384 / P⌉ + ⌈1,536 / P⌉ + 1 blocks plus the 1 % watermark). At 8 GiB that allows at most 9 sessions at 16-, 32- and 64-token pages and 8 in the decoupled 784-token mode. Every configuration of 1 to 8 sessions passes, and the shared 784-token pool, which has no Mamba pool, is not affected.
- The lifetime request bound of one server (engine lifecycle entries, request tombstones and render bindings, none ever evicted): 256 per session, the per-session request bound, and at least 1,024 (4,096 at 16; section 4.9).
- Host memory: the pinned page arena (section 4.6) of up to 16 × 4,097 pages of 1 MiB = 64 GiB at P = 16, next to the 32 GiB CPU pool and a 154 MB Gated-DeltaNet carry per session (two with `--gdn-device-handoff`, section 4.10).

On the GPU, 16 sessions ran in job 3339019 (`full`, `observation_query` and `random_fill`, eager, runtime `afffd2b`) and job 3343069 (`observation_query`, eager and with `--cuda-graphs full`, runtime `613a7d5`): synthetic sessions of 6 turns with 2,600-token observations, prompts up to about 14K tokens, 24 GiB at 16-token pages, on the reused clean environment of job 3270721 and one GH200. All 16 sessions decoded together, nothing was preempted, no turn waited for capacity, and an offline replay of every selective turn reproduced the recorded pages. Two eager runs of this workload (`observation_query` in both jobs) gave the same tokens on 72 of 96 turns: with batched decode a reply can change with the batch composition, which depends on timing. These contexts stay far below the 65,536-token limit, so the runs do not reach the limits described next.

The never-deadlock rule cannot hold for 16 `full` sessions at 64K on one GH200. They need about 67,350 attention blocks of 1 MiB plus the Mamba pool, about 70.6 GiB of KV, while the GPU has 94.53 GiB free at start (vLLM log of job 3313802) and the weights take 51.1 GiB. The selective policies need less only between rebuilds, since a rebuild returns to full residency. With a fixed `--active-token-budget 8192` at 16-token pages, 15 idle sessions holding their budget and their last turn's pages, plus one turn's peak, fit in 24 GiB when turns are about 7,000 tokens and need 34 GiB when they reach the 16,384-token observation limit; a 40 % budget needs 41 and 51 GiB on the same two assumptions. Size `--kv-cache-gib` and the workload accordingly. `CAPACITY_WAIT` holds a turn back while running requests can still free blocks. When nothing else runs, a turn that the idle sessions' pins leave no room for waits too, until a session closes or, with `--release-idle-pages` (section 4.13), idle sessions' pages are released; once every session waits, the server stalls. Before this revision the gate admitted such a turn because nothing ran: on the GPU its decode took the last free block and vLLM preempted it, which is fatal (Slurm job 3398129: 16 Terminus 2 trials, `full`, 32 GiB, the idle pins held all but 69 of 26,698 attention blocks). When no other session pins blocks, the turn is admitted as before (as on a one-task server). The largest `--kv-cache-gib` stays 32: with the Goal rewriter, about 36 GiB of the GPU is estimated to be left for KV (not measured).

**Session pool: many tasks, at most K running (`--max-running-sessions K`).** With this option `--task` registers 1 to 64 tasks (pinned in `startup.json` like any task list, plus `max_running_sessions` = K), and at most K (1 to 16, at most the number of tasks) sessions are open at a time, so that one server runs, for example, 32 SWE-bench tasks with 16 concurrent agents, refilled as agents finish. Routing by the task message, per-session isolation, the turn schedule, the capacity gate and no-preemption are unchanged. What K sizes instead of the task count: `max_num_seqs` = K, the Mamba pool (2K + 1 slots) and the launcher's attention-pool check, the never-deadlock rule above (N = K) and the pinned page arena. What the registered task count T sizes: the lifetime request bound, max(1,024, 256 × T) (8,192 at T = 32; section 4.9).

```sh
python -B serve.py --release-dir "$RELEASE" --task /abs/path/to/tasks-pool.json --policy full \
  --max-running-sessions 16 --kv-cache-gib 24 --page-tokens 16 --kv-layout decoupled \
  --output "$RUNS/full-pool" --port 8000 --start
```

- **Open.** A session opens at its task's first request (turn 0). While K sessions are open, that request waits before admission, first come first served, until one closes; the wait is `open_wait_s` in its `runtime.jsonl` row. A request that goes away while it waits leaves its task unopened.
- **Close.** When an agent is done, the harness closes its task's session: `POST /v1/sessions/close` with `{"instance_id": "<task id>"}` (section 6.5). The engine frees everything the session holds: its resident pages and live tail on the GPU, its Gated-DeltaNet carry and pinned-arena pages on the host, its selection tables, Goal state and transcripts, and the per-request records it kept. Both pools return by exactly the session's pins, and the next waiting task opens. A closed task never reopens; later requests for it fail at once (`session ... closed by the harness`).
- **Failures.** A failed request closes its session as before, but the session keeps its open slot and engine state until the harness closes it, so the harness closes every session it started, failed or not.
- **Shutdown** closes the sessions still open, as on any server.

On the GPU (job 3343069, runtime `613a7d5`): 24 tasks with K = 8, `observation_query` with `--cuda-graphs full-decode`, 6 turns of 3,000-token observations per task, 24 GiB at 16-token pages. Every task opened, ran its 6 turns and was closed by the harness with a close receipt (24 opens, 24 closes); at most 8 sessions ran at once, nothing was preempted, no turn waited for capacity, the offline selection replay matched every selective turn, and both pools were back at their readiness baseline at shutdown. The Mamba pool held 17 slots (2K + 1): K sizes it, not the 24 tasks (table above).

### 4.5 GPU job (Slurm)

`examples/slurm/serve_gh200.sh` runs one server in one Slurm job: job-local `TMPDIR` and caches, the section 3 environment, the dry run, `serve.py --start`, the READY wait, a harness, `SIGTERM` to the server PID, the exit code and the cleanup receipts. It follows the validated jobs' `job.sh` and job steps. They passed these `sbatch` options on the command line:

| Option | Validated value |
| --- | --- |
| `--partition` | a GH200 partition: an interactive one with a 2 h limit (jobs 3270721, 3299809, 3300301, and the 16-token-page pilot 3313802, 3314010, 3314319, 3314623), the batch one for the 784-token-page pilots (jobs 3299922, 3300434) |
| `--gpus` / `--nodes` / `--ntasks` | `1` / `1` / `1` (`--gres=gpu:1` is the usual equivalent elsewhere; not part of the validated runs) |
| `--cpus-per-task` / `--mem` | `16` / `200g` |
| `--time` | `00:30:00` (4 one-session servers, job 3270721), `00:45:00` (3 four-session servers, jobs 3299809 and 3300301), `02:00:00` (2 five-session servers, jobs 3313802, 3314010, 3314319, 3314623), `03:45:00` (3 servers, job 3299922), `10:45:00` (6 servers, job 3300434) |
| `--mail-type` | `BEGIN,END,FAIL` (job 3270721 used `END,FAIL`; batch queue waits can be long) |
| `--export` | `ALL` (the script then resets `PATH`, `LD_LIBRARY_PATH`, `HOME`, caches and `TMPDIR` itself) |

The script takes its settings from the submitting shell (its header lists them). A smoke run of the Quickstart task:

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

`RUN_ROOT` must be short: the job uses `$RUN_ROOT/<job id>/tmp` as `TMPDIR` and refuses a path over the AF_UNIX budget. `MODE=swebench` opens the reverse tunnel of section 6.4 and runs `examples/swebench/run_agents.sh` on the agent host over SSH (section 7.4). `PAGE_TOKENS=16 KV_LAYOUT=decoupled` passes `--page-tokens 16 --kv-layout decoupled` to `serve.py` (section 4.6); with both unset the server uses the default 784-token layout. The validated jobs were driven by our private controller with extra gates (environment snapshots, GPU idle checks, per-turn evidence checks); this script keeps the server-side steps only and is **not itself part of the validated runs**.

### 4.6 Decoupled KV layout and 16-token pages

**Why pages are 784 tokens by default.** Qwen3.6-27B is a hybrid model: 16 full-attention layers (4 KV heads × 256, bf16: 4,096 bytes per token per layer) and 48 Gated-DeltaNet layers, each with a 3,207,168-byte recurrent and convolution state (vLLM calls these Mamba layers). vLLM 0.24 keeps both kinds in one block pool with one page size. `_align_hybrid_block_size` in `vllm/platforms/interface.py` raises the attention block size until one attention page holds at least one Mamba page, 16 × ⌈3,207,168 / (16 × 4,096)⌉ = 784 tokens, and pads the Mamba page to 784 × 4,096 = 3,211,264 bytes. A runtime page is one attention block, so pages are 784 tokens, and the PoC's 16-token budgets have to be rounded to whole pages (section 5).

**What the decoupled layout changes.** It gives the two kinds separate block pools with independent block sizes (`docs/exp-decouple-pages/DESIGN.md`, decisions D2-D4):
- **Attention pool:** blocks of P tokens. One block is still one page: P × 65,536 bytes over the 16 attention layers, 1 MiB at P = 16.
- **Mamba pool:** unpadded Gated-DeltaNet states. The 48 layers form 3 Mamba groups of 16; one block holds one group's states (16 × 3,207,168 = 51,314,688 bytes), and one slot is one block per group. The pool holds K = 2S + 1 slots for S sessions (a running and a crossing slot per session plus a spare) and a null block: 1 + 3K blocks. It is reserved first out of `--kv-cache-gib`; the attention pool gets the rest, ⌊(KV bytes − (1 + 3K) × 51,314,688) / (P × 65,536)⌋ blocks.
- **Mamba align block:** M = P × ⌈784 / P⌉, which is 784 at P = 16. The Gated-DeltaNet state therefore crosses blocks, is captured and is restored exactly as at 784-token pages (153,944,064 bytes per state).

Group structure, kernels and the Mamba `align` mode stay as they are.

**Flags.** `--page-tokens 16`, `32` or `64` always uses the decoupled layout; `--kv-layout decoupled` also selects it at 784 tokens, a validation mode that splits only the pools. `--page-tokens 16 --kv-layout decoupled` (the form the GPU runs used) is the same as `--page-tokens 16` alone and names the layout explicitly. `serve.py` then appends `--block-size P --mamba-block-size M --additional-config {"remote_ra_kv_layout":{"mamba_slots":K,"mamba_tokens":M,"page_tokens":P}}` to the vLLM command and records `kv_layout` and `page_tokens` = P in `startup.json`. Without either flag, the vLLM command, the startup config and its pins are byte-identical to those of `902690a` (CPU tests against the earlier code).

**Startup log.** A decoupled server logs these lines (job 3313802: 24 GiB, 5 sessions) and neither vLLM's `Setting attention block size to 784 tokens` nor its Mamba padding line:

```text
Decoupled KV layout: attention block 16 tokens, Mamba block 784 tokens, 11 Mamba slots, unpadded Mamba page.
Decoupled KV cache pools: 22912 attention blocks, 34 Mamba blocks
GPU KV cache size: 366,592 tokens
```

`ready.json` records the layout and both pool sizes (section 8). Pool sizes of the validated runs are in the table of section 4.4.

**Host memory.** In the decoupled layout the worker keeps the attention pages it copies to the host in a pinned page arena. Its capacity is S × (⌈L / P⌉ + 1) pages of P × 65,536 bytes for a context limit of L tokens (`--dram-arena-gib` sets one smaller capacity, section 4.9), allocated on demand in 256 MiB chunks: up to about 4 GiB per session at P = 16 and L = 65,536 (`arena_pages` in `remote_ra_live_v2/worker_composition.py`, `PageArena` in `remote_ra_growing_worker/tensors.py`). This comes on top of the 32 GiB CPU pool of section 3.

**Selection at P = 16.** The Selector V2 contract takes the page size as a parameter, so at P = 16 it is the PoC's own contract: capacity `max(1, mandatory pages, floor(0.40 × history tokens) // 16)` pages, a recent quota of 64 pages (1,024 tokens), a live tail under 16 tokens and one vector per 16-token page (section 5). The worker compacts the page vectors to int8 on the GPU in one batch per forward (`compact_int8_v1` in the selection contract) instead of one at a time on the CPU.

### 4.7 Validation at 16-token pages

All runs below used the reused clean environment of job 3270721 and one GH200; one run each.

**GPU smokes** (runtime `8f1af6a`; synthetic Mini-SWE-Agent XML sessions with 6,500-token observations; every server and every plan check passed):

| Job | Servers | Result |
| --- | --- | --- |
| 3308015 | `full`, 1 session × 10 turns up to about 60K prompt tokens, 8 GiB, in three layouts: default, decoupled at 784 and decoupled at 16 | Identical prompt tokens, replies and completion counts on every turn across the three layouts; identical prepare and restore H2D bytes per turn |
| 3308414 | `full`, `random50`, `goal_query`, `observation_query` at 16-token pages, 1 session × 10 turns, 8 GiB | The query policies selected about 40 % of the 16-token pages once the history passed 8,192 tokens, and the selective policies restored pages |
| 3308482 | 4 sessions × 6 turns, 24 GiB: `full` at the default layout; `full` and `goal_query` at 16-token pages | Batched decode across the 4 sessions, no preemption, no capacity wait; at shutdown both pools of each decoupled server back at their readiness baseline |

`780fe54` adds only `random_fill` to `8f1af6a`; the four other policies send and write the same bytes on both (`test_random_fill_legacy.py`).

**SWE-bench pilot at 16-token pages** (runtime `780fe54`; Mini-SWE-Agent 2.4.2 and the setup of section 7, the same 10 Astropy instances as job 3300434 in two groups of 5 sessions; 24 GiB; `--page-tokens 16 --kv-layout decoupled` for every server; one job and one run per policy; agent cap 3,000 s per group). Every server passed its server-side checks. Turns are agent requests; resolved is from swebench 5.0.2:

| Policy | Job | Turns | Resolved |
| --- | --- | --- | --- |
| `full` | 3313802 | 517 | 7/10 |
| `observation_query` | 3314010 | 538 | 8/10 |
| `goal_query` | 3314319 | 643 | 7/10 |
| `random_fill` | 3314623 | 608 | 8/10 |

The same instances at 784-token pages (runtime `b1e6988`, job 3300434): `full` 490 turns and 9/10 resolved, `goal_query` 716 and 6/10, `observation_query` 715 and 7/10.

This is one run per policy on 10 instances. `full`, whose tokens did not depend on the page size in the smoke above, changed its outcome on 2 of the 10 instances between the two pilots, so differences of this size can come from run-to-run variation alone. **This is not benchmark evidence** and does not rank the policies.

Offline token-exact verification of the archived server evidence: jobs 3313802, 3314010, 3314319 and 3314623 (all four policies).

**Performance notes** (single runs):
- Smoke 3308015, steady inter-token latency with one request decoding: 67.4 ms default layout, 68.3 ms decoupled at 784, 74.9 ms at 16-token pages. Each smoke job warmed up one layout only, the default one in this job. In smoke 3308482, warmed up at 16-token pages, the order reversed: 70.8 ms default against 68.1 ms at 16-token pages with one request decoding, 75.5 against 73.2 ms with four. The gap in job 3308015 is therefore probably mostly a warm-up effect. Neither job isolates the per-step cost of 16-token pages, and the design target of at most +0.1 ms per request per step (DESIGN R5) has not been shown.
- Smoke 3308482: `full` completed 14.63 turns per minute at 16-token pages and 14.22 at the default layout, which was the one not warmed up in that job.

### 4.8 CUDA graphs

`--cuda-graphs off` (the default) is the eager server above: the vLLM command, `startup.json`, its pins and the evidence are byte-identical to those without the flag (CPU tests against `b37044f`, `test_cudagraph.py`).

`--cuda-graphs piecewise`, `full` or `full-decode` replaces `--enforce-eager` in the vLLM command with a pinned `--compilation-config`; async scheduling stays off:

| Mode | `--compilation-config` | What vLLM 0.24 captures |
| --- | --- | --- |
| `piecewise` | `{"cudagraph_mode":"PIECEWISE","mode":"VLLM_COMPILE"}` | the compiled pieces between FlashAttention, the KV-cache update and the Gated-DeltaNet core, which run eagerly every step and read that step's selective sequence lengths, slot mapping, block tables and Mamba state indices |
| `full` | `{"cudagraph_mode":"FULL_AND_PIECEWISE","mode":"VLLM_COMPILE"}` (stock vLLM's default) | whole uniform-decode steps, attention included; mixed steps stay piecewise. The runner binds every full-attention group to its physical sequence-length buffer at capture and at replay |
| `full-decode` | `{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"NONE"}` | whole uniform-decode steps of the eager kernels, without torch.compile; prefill, mixed steps and phase 1 run eagerly as with `off`. The same binding as `full` |

- **Pins and checks.** `startup.json` records `cuda_graphs` and the effective config `enforce_eager: false` plus the compilation config. At startup the frontend compares vLLM's parsed arguments, and the worker checks the resolved configuration (compilation mode, CUDA-graph mode, capture sizes and, when compiled, the three ops kept outside the graphs) and refuses any difference; with the flag off it refuses a runner that captured graphs.
- **Evidence.** `worker-attached.json` gains a `cuda_graphs` record (mode, resolved configuration, capture sizes); nothing is checked per request or step.
- **With several sessions.** vLLM sizes the captured batches from `max_num_seqs`: one per startup task, or K with `--max-running-sessions K` (section 6.5). The sizes are 1, 2, 4 and multiples of 8 up to 2 × `max_num_seqs` (1, 2, 4, 8, 16, 24, 32 at 16 sessions), so every decode step of up to 16 running sessions has a graph (CPU test `test_int_flags.py`, vLLM 0.24's own sizing code; job 3343069 recorded exactly these sizes at 16 sessions, and 1, 2, 4, 8, 16 at K = 8).
- **Phase 1** (and the Goal forward) runs without a CUDA graph and unpadded: compiled code in `piecewise` and `full`, the eager kernels in `full-decode`. Decode steps (and, except in `full-decode`, small mixed steps) use the graphs. With `--phase1-in-step` (section 4.10) a phase 1 that joins a decode step runs with that step: with the piecewise graphs in `piecewise` and `full`, eagerly in `full-decode`.
- **Results.** torch.compile (`piecewise`, `full`) replaces vLLM's custom kernels with compiled ones and changes bf16 numerics, so greedy outputs can differ from the eager server. `full-decode` keeps the eager kernels; it changes only what capture itself changes: FlashAttention's split limit for batches up to the largest capture size, and padding to a capture size. Prompt and transcript token exactness does not depend on the mode. With torch.compile, startup takes about a minute longer on a cold compile cache (stock vLLM 0.24.0 on one GH200: +58 s; capture 4 s, 0.13 GiB).
- **GPU runs** (synthetic sessions on the reused clean environment of job 3270721 and one GH200, one run each; mechanism and cost, not benchmark results):
  - Job 3338893 (runtime `f19f464`; policy `full`, 1 session × 3 turns, prompts up to about 13K tokens, at 784-token and at 16-token pages): `piecewise`, `full-decode` and `full` gave the same tokens as eager on every turn, and stock vLLM 0.24.0 in the same mode generated the same tokens for the replayed turns. No turn of this job selected pages.
  - Job 3339448 (same runtime; `observation_query` at 16-token pages, 1 and 8 sessions × 4 turns, prompts up to about 37K tokens, every mode): the graph modes' outputs differ from eager. At 1 session the first difference is in turn 1's reply, before any turn selects; at 8 sessions the selected pages differ as well (Jaccard 0.81–0.93 against eager on the active turns). Steady decode step with 8 requests decoding: 82.7 ms eager, 29.2 ms `piecewise`, 34.0 ms `full-decode`, 28.9 ms `full`; a cost measurement, not an end-to-end speedup.
  - Job 3343069 (runtime `613a7d5`): 16 sessions with `full` differ from the same workload in eager, in outputs and in selections; the session pool ran with `full-decode` (section 4.4).
- **Comparing.** Graph-mode outputs, and with them the query policies' page choices, can differ from the eager server's, so compare a CUDA-graph server with stock vLLM in the same mode. Even in eager mode, batched runs are not token-identical from run to run (section 4.4).
- **Gemma 4** (section 4.14) takes the same modes on its window layout, with its global group bound; not GPU-validated.

### 4.9 Context limit and reply bound

Three launcher options size a server for longer agent runs; all three ran on the GPU in job 3348705 (below). Without them the vLLM command, `startup.json`, its pins, the evidence and the page selections are byte-identical to `8c5f4d0` (CPU tests and an identity probe over 549 launcher, startup, worker and engine outputs). `--max-model-len 262144` is a later third context choice; it ran on the GPU in job 3360375 (below). Adding it left the defaults and the 131072 profile byte-identical (identity probes over 531 and 549 outputs, and the 131072 dry run's `startup.json` and vLLM command), and so did merging it with the decode fast path of section 4.11 (the same 549 outputs, no difference).

| Option | Default | Other choice | What it pins |
| --- | --- | --- | --- |
| `--max-tokens` | 1536 (the PoC's reply bound) | 4096 | `startup.json` `max_tokens` |
| `--max-model-len` | 65536 | 131072 or 262144 | vLLM's `--max-model-len` and `effective_server_config.max_model_len` |
| `--dram-arena-gib N` | room for every running session at the context limit | one pinned page arena of N GiB for all sessions, decoupled layout only | `startup.json` `dram_arena_pages` |

- **On the GPU (job 3348705, runtime `710bfc6`, one GH200; verified offline, token-exact).** Every server ran with `--max-model-len 131072 --max-tokens 4096` and the 16-token decoupled layout. One session (`full` and `observation_query`, eager, 16 GiB KV): prompts grew to 120,864 tokens; requests of 128,871 and 128,873 tokens were refused with HTTP 400 at the frontend before admission, and the server kept serving. 16 sessions (`observation_query` with `--active-token-budget 8192`, `--cuda-graphs full`, 32 GiB KV, `--dram-arena-gib 88`): 160 turns, 114 with active selection, sessions reaching 60K to 111K tokens, no preemptions; six turns generated exactly 4,096 tokens (prompts 33.6K to 76.2K) and the next turns continued; ITL with 16 running 40.3 ms; process RSS at most 117.9 GiB and cgroup peak 134 GiB (of 200 GiB requested). Not exercised: a 4,096-token reply near the context limit (the planned long reply at about 121K tokens ended after 69 and 215 tokens). Pinned memory does not show in the process's VmPin, so the 128 GiB pinned-host budget rests on the launcher's arithmetic, not a measurement.
- **On the GPU (job 3360375, runtime `e3efc22`, one GH200; verified offline, token-exact, 33 checks and none failed).** `e3efc22` holds this revision's 262144-token commits on `175a183`, before they were rebased onto the step profile and the decode fast path of section 4.11; its runtime sources are this revision's without those two, which no server here used (262144 together with `--decode-fastpath` has run only on the CPU, `test_ctx_256k.FastPathTests`). Every server ran with `--max-model-len 262144 --max-tokens 4096` and the 16-token decoupled layout.
  - One session, `full`, eager, 24 GiB KV: 21 turns. Prompts grew to 239,527 tokens, and the reply to that prompt generated the full 4,096 tokens (243,623 in all). A request of 259,561 tokens (above 262,144 with the 4,096-token reply) was refused with HTTP 400 at the frontend before admission, and the health check answered 200 afterwards. ITL with one running: 77.4 ms. The same run with `observation_query` reached 239,525 tokens, and the offline selection replay equalled the recorded selections on all 19 selective turns.
  - 16 sessions, `observation_query` with `--active-token-budget 8192`, `--cuda-graphs full`, 32 GiB KV, `--dram-arena-gib 88`: 273 requests, prompts up to 221,416 tokens, 16 running at the peak, ITL with 16 running 41.3 ms, no preemptions and no capacity waits, GPU memory at most 93,349 MiB. The selection replay equalled the recorded selections on all 199 selective turns. A request of 268,473 tokens was refused with HTTP 400 at the frontend, and the server kept serving. Five replies reached the 4,096-token bound (prompts of 147,672 to 221,416 tokens); the three that had a next turn continued token-exact.
  - The arena cap (2 sessions, `full`, 24 GiB KV, `--dram-arena-gib 17`): the second session's capture at its turn 7 did not fit the arena, so that request failed with HTTP 500 and `DRAM_ARENA_EXHAUSTED` in the server log, as designed; the first session's next turn (197,276 tokens) was served.
  - Host memory: the 16-session server's RSS was at most 111.8 GiB, against the launcher's worst case of 122.3 GiB; the job's cgroup peak was 134.6 GiB (of 200 GiB requested). The only finding: pinned memory does not show in VmPin (it reads 0), so the 128 GiB pinned-host budget still rests on the launcher's arithmetic.
- **Choosing the limit.** A longer limit needs more GPU KV and pinned host memory for the same sessions (the refusals below), so use the shortest one that holds the agent's histories, prompt plus reply: with 4096-token replies, prompts of up to 61,440 tokens at 65536, 126,976 at 131072 and 258,048 at 262144. 262144 is the model's native context; there is no longer choice.
- **Reply bound.** A server setting, not part of the agent protocol: the frontend's live controller and the engine hold the pinned value; the Mini-SWE-Agent XML route fixes it on every request and the engine refuses any other value before allocation. The server still decodes greedily and ignores the client's sampling fields. The capacity gate reserves every request's pages up to ⌈(prompt + max_tokens) / P⌉, and the launcher's margin is one reply, ⌈max_tokens / P⌉ + 1 blocks (97 at 1536, 257 at 4096 with 16-token pages). vLLM caps a reply at the context limit minus the prompt, so a prompt holds at most the context limit minus the reply bound (61,440 tokens at 65,536 with 4096-token replies; 126,976 at 131,072; 258,048 at 262,144).
- **Startup checks, once.** The frontend compares vLLM's parsed `--max-model-len` with the pin; the engine checks its resolved `max_model_len` and sends the worker a context record (`max_model_len`, `max_tokens`) in its startup record when either is not the default; the worker checks the record against its runner and echoes it in `worker-attached.json`; the engine and the frontend compare the echo, and `ready.json` adds `context`. A 262144-token server also sets its request caps (below) once, in the frontend and in the engine. Nothing is checked per request or step. The page catalog bound, the attention and Gated-DeltaNet rows and the pinned page arena follow vLLM's `max_model_len` (8,193 pages per session at P = 16; 16,385 at 262144).
- **Launcher refusals.** One session's turn at the context limit must fit the attention pool (decoupled layouts: ⌈L / P⌉ + 1 + ⌈16,384 / P⌉ + ⌈max_tokens / P⌉ + 1 blocks plus the 1 % watermark) or the shared 784-token pool (⌈L / 784⌉ + 1 + 6 + ⌈max_tokens / 784⌉ + 1 blocks plus the watermark, what vLLM's own startup check demands); at 131072 tokens this refuses `--kv-cache-gib 8` in every layout. The worst-case pinned host memory (the 32 GiB CPU pool, one 154 MB Gated-DeltaNet carry per running session (two with `--gdn-device-handoff`) and a host copy of every attention page of every running session at the limit) must stay within 128 GiB (`PINNED_HOST_GIB` in `serve.py`; the validated jobs requested 200 GiB). Every 65536-token configuration needs at most 99.4 GiB and passes. At 131072 tokens 11 running sessions fit (121.8 GiB at P = 16) and 16 need 162.5 GiB, so 16 sessions need `--dram-arena-gib` (88 GiB gives 122.3 GiB). A capture beyond a capped arena fails that request (`DRAM_ARENA_EXHAUSTED`); nothing is spilled or evicted. 16 sessions at an average of 90,000 tokens hold 87.9 GiB of pages. When `--dram-arena-gib` itself is too large for the budget, the refusal names the largest arena that fits (93 GiB for 16 sessions, 91 GiB with `--gdn-device-handoff`).
- **Connector CPU pool (`--connector-pool-gib N`, opt-in; on the GPU only in the Terminal-Bench dry run of section 6.7).** The 32 GiB CPU pool above belongs to vLLM's `OffloadingConnector`, which `serve.py` always configures (`--kv-transfer-config`); vLLM pins and zeroes it when the worker starts, whatever `--kv-cache-gib` is. This runtime never moves data through it. At startup it isolates the connector (`remote_ra_http/connector.py`): every prefix lookup returns 0 tokens, any load from the pool is refused, finished requests store nothing, and the connector never gets a transfer job. Pages copied to the host and restored from it use the runtime's own pinned memory instead: the page arena of section 4.6 (in the shared pool, one pinned copy per captured page) and the Gated-DeltaNet carries. In the server logs of SWE-bench jobs 3354318, 3357655, 3357977, 3360287 and 3363871 (all four policies), the pool's usage read 0.0 throughout. The connector itself stays, because the runtime's isolation and checks hook into it.
  - `--connector-pool-gib N` (N from 1 to 31; off by default) gives the pool N GiB instead of 32, and the budget counts N GiB. One GiB still holds 20 blocks of 784 tokens, or 1,024 of 16 tokens, and saves about 31 GiB of pinned host memory per server at every `--kv-cache-gib`: every 65536-token configuration then needs at most 68.4 GiB instead of 99.4 GiB. This matters for a job that requests only one GH200's share of the node's host memory.
  - `startup.json` records `connector_pool_bytes`, and vLLM gets `cpu_bytes_to_use` = N GiB. The engine checks that its vLLM configuration has the pinned value, and the worker checks that the allocated pool holds at least one block per KV tensor and at most N GiB, once each at startup (`remote_ra_live_v2/connector_guard.py`). Without the option, the vLLM command, the startup config and the launcher's refusals are unchanged (identity probe against the previous revision: 555 + 219 items, 0 differences). CPU tests: `test_connector_pool.py`.
- **Token and frame caps** (measured on the CPU with the pinned tokenizer, test `test_align_ctx.TokenFrameCapTests`): Mini-SWE-Agent requests of 126,976 tokens with SWE-like observations (code listings, padded `pytest -v` output, FITS header cards, at most 10,000 characters each) need 3.3 to 5.3 source bytes per token, 424 to 666 KB of the 1 MiB source limit, under 0.3 MB of the 1 MiB core request limit (131,072 ids are at most 0.66 MB) and under 1.5 MB of the 4 MiB frame limit; 126,976 tokens are within the 131,072-token bound. The source limit binds above 8.26 bytes per token at 126,976 tokens (16.4 at 64,000): banner-only logs (one 80-column rule per token, about 19 bytes per token) are refused before admission, at either context. These are the caps at 65536 and 131072, unchanged; a 262144-token server doubles them (next item).
- **262144 tokens (GPU-validated in job 3360375, above).** `--max-model-len 262144` is the model's native context (`max_position_embeddings` 262144 with the default rope): no YaRN, no override and no new pin. Anything longer would need YaRN, which changes outputs at every length. As with 131072, only vLLM's `--max-model-len` and `effective_server_config.max_model_len` change, and the page catalog (16,384 pages at P = 16), the rows and the arena follow vLLM's `max_model_len`. One request's byte and token caps scale with the pinned limit (table below). On the CPU with the pinned tokenizer (test `test_ctx_256k.TokenFrameCapTests256`), 258,048-token requests (262,144 minus a 4096-token reply) need 3.3 (code listings, README text) to 5.2 (FITS header cards) source bytes per token: 0.84 to 1.35 MB of source, so the padded and FITS histories exceed the former 1 MiB. They need at most 0.61 MB of core request and 2.9 MB of frame. The 2 MiB source cap binds above 8.13 bytes per token, so banner-only logs are still refused before admission.
  - Launcher refusals at 262144 tokens. One session's turn needs 16,385 + 1,024 + 257 blocks plus the watermark (17,824 at 16 GiB with 4096-token replies), so `--kv-cache-gib 8` and `16` are refused in every layout and at every session count; 24 and 32 pass at 1 to 16 sessions. The shared 784-token pool needs 352 blocks with 4096-token replies (333 usable at 16 GiB, 500 at 24 GiB).
  - Pinned host memory at 262144 tokens: at most 5 running sessions fit without an arena (113.0 GiB at P = 16). 6 need 129.1 GiB and 16 need 290.5 GiB; a session pool counts its K running sessions. 16 sessions need `--dram-arena-gib` between 17 GiB (one session at the limit, 16,385 pages) and 93 GiB (91 GiB with `--gdn-device-handoff`). 93 GiB holds 95,232 tokens per session on average, 88 GiB 90,112. 16 sessions all at 262,144 tokens cannot fit: their pages alone are 256 GiB.
  - Not changed yet: the 300-s admission wait and 360-s lease; phase 1 still takes at most 16,384 tokens per appended message; the per-request work outside the GPU that grows with the prompt (prompt hashes, JSON snapshots, detokenization, evidence rows with every prompt id); arena fairness (`DRAM_ARENA_EXHAUSTED` fails whichever session captures next); the capacity of a 262K rebuild under the strict rule (17,409 blocks).
- **Lifetime request bound.** max(1,024, 256 × T) for T registered tasks: the per-session bound of 256 requests (the frontend's turn bound, the engine's turn ordinal and generation bound) summed over the tasks, so it never refuses a request a session could still make. 32 tasks of 120-step agents make at most 3,840 requests against 8,192; a `CAPACITY_WAIT` retry takes no lifetime record. Per-session limits stay at 256 requests, so a step limit above 256 is refused by `run_agents.sh`.
- **SWE-bench.** `run_agents.sh` takes `STEP_LIMIT` (default 90) and `MAX_TOKENS` (default 1536) for the model override (section 7.4).

**Capacity at 131,072 tokens, 16 sessions, P = 16, 4096-token replies** (derived from the rule of section 4.4: 15 idle sessions plus one admitting turn at the full-context peak plus the margin within the usable attention blocks; the Mamba pool takes 100 blocks):

| Policy and idle model | Idle blocks per session | KV GiB needed | Fits 24 | Fits 32 |
| --- | ---: | ---: | :-: | :-: |
| `full` (and every policy under the strict rule, because a rebuild returns to full) | 8,193 | 135.4 | no | no |
| 40 %, budget only | 3,277 | 62.6 | no | no |
| 40 % + typical last turn (7K tokens) | 3,715 | 69.1 | no | no |
| 40 % + worst last turn (16,384 + 4,096) | 4,557 | 81.6 | no | no |
| `--active-token-budget 8192`, budget only | 513 | 21.8 | yes | yes |
| budget 8192 + typical last turn | 951 | 28.2 | no | yes |
| budget 8192 + worst last turn | 1,793 | 40.8 | no | no |

At 65,536 tokens with 4096-token replies the same rows need 70.8, 34.4, 40.9, 53.4, 17.8, 24.2 and 36.6 GiB (with 1536-token replies, the values of section 4.4). 16 `full` sessions stay deadlock-free only at short contexts whatever the limit (about 18K tokens per session at 24 GiB and 26K at 32 GiB). With about 36 GiB of the GPU estimated to be left for KV next to the Goal rewriter, 16 sessions at 131,072 tokens are deadlock-free only with a small fixed budget and turns well below the observation limit.

**One request's caps by context limit** (`request_caps` in `remote_ra_live_v2/context_limits.py`; the frontend and the engine set the 262144 values once at startup, and nothing is set at 65536 or 131072):

| Cap | 65536 and 131072 | 262144 |
| --- | ---: | ---: |
| XML source (the conversation JSON) and each encoded message: max(1 MiB, 8 B × L) | 1 MiB | 2 MiB |
| core request (msgpack, 1 to 5 bytes per prompt id): max(1 MiB, 8 B × L) | 1 MiB | 2 MiB |
| authenticated frame: 2 × (source + core) | 4 MiB | 8 MiB |
| EngineCore's registered-add message: frame + 1 MiB | 5 MiB | 9 MiB |
| prompt ids per request: max(131,072, L) | 131,072 | 262,144 |

**Capacity at 262,144 tokens, 16 sessions, P = 16, 4096-token replies** (derived by the same rule; job 3360375 ran the 8192-token budget at 32 GiB with no capacity wait). The admitting turn's full-context peak alone is 17,409 blocks (17 GiB). The last column is the most sessions that fit at 24 and 32 GiB:

| Policy and idle model | Idle blocks per session | KV GiB needed | Fits 24 | Fits 32 | Sessions at 24 / 32 GiB |
| --- | ---: | ---: | :-: | :-: | :-: |
| `full` (and every policy under the strict rule) | 16,385 | 264.6 | no | no | 1 / 1 |
| 40 %, budget only | 6,554 | 119.2 | no | no | 1 / 3 |
| 40 % + typical last turn (7K tokens) | 6,992 | 125.7 | no | no | 1 / 2 |
| 40 % + worst last turn (16,384 + 4,096) | 7,834 | 138.1 | no | no | 1 / 2 |
| `--active-token-budget 8192`, budget only | 513 | 29.8 | no | yes | 8 / 16 |
| budget 8192 + typical last turn | 951 | 36.3 | no | no | 5 / 12 |
| budget 8192 + worst last turn | 1,793 | 48.7 | no | no | 3 / 7 |

Of these rows only the 8192-token budget, with idle sessions that hold no more than their budget, keeps 16 sessions at 262,144 tokens deadlock-free (at 32 GiB).

### 4.10 Admission optimizations (opt-in)

An admission is the engine's work on a new turn before it decodes: phase 1, for `goal_query` the Goal, the selection and the reservation (section 5). By default admissions run one at a time in arrival order, phase 1 runs as a separate forward pass while the running requests wait, and the Gated-DeltaNet state goes from phase 1 to phase 2 through pinned host memory. Three launcher flags change this, each off by default (the letters are the admission designs that the code comments name):

| Flag | Default | What it does | What it pins at startup |
| --- | --- | --- | --- |
| `--gdn-device-handoff` (e) | off | hands the Gated-DeltaNet state from phase 1 to phase 2 on the device instead of through host copies | `startup.json` `admission.gdn_device_handoff` = `true` |
| `--phase1-in-step` (b) | off | runs an admission's phase 1 inside the next decode step instead of a separate pass | `admission.phase1_in_step` = `true` |
| `--admission-overlap K` (f) | 1 (off) | keeps up to K admissions in flight, 2 ≤ K ≤ the sessions running together | `admission.admission_overlap` = `{"permits": K, "admission_seconds": 300 + 120(K − 1), "lease_seconds": 360 + 120(K − 1)}` |

`startup.json` has an `admission` record only when a flag is on; with all three off the startup config, the evidence and the vLLM command gain nothing. The engine and the worker check the record once at startup, the worker echoes it in `worker-attached.json` and `ready.json` records it; nothing is checked per request or step (`remote_ra_live_v2/admission_flags.py`; CPU tests `test_gdn_handoff.py`, `test_adm_phase1_step.py`, `test_adm_overlap.py`). `examples/slurm/serve_gh200.sh` passes none of the three flags (nor `--cuda-graphs`).

- **`--gdn-device-handoff`.** Phase 1 keeps its probe slot on the GPU, and phase 2's first step copies it device to device into the phase-2 slot; Mamba-block crossings copy on the device too. The committed carry is the only host copy: at the end of a turn it is copied to the host asynchronously, into whichever of the session's two pinned carry buffers does not hold the committed carry, and joined by the drain; phase 1 restores it from there (`remote_ra_live_v2/gdn_handoff.py`). A continuing turn's phase 2 holds three Gated-DeltaNet slots (running, crossing and the held probe slot), and the capacity gate counts three. The launcher's pinned-host budget (section 4.9) counts the second carry buffer: 153.9 MB more per running session (2.3 GiB at 16 sessions; 16 sessions at 131,072 tokens with `--dram-arena-gib 88` need 124.6 GiB instead of 122.3). The transient pinned bounce copies of indexed restores, freed when the restore is joined, are not counted. Every `runtime.jsonl` row adds `restore.d2d_bytes` (section 8).
- **`--phase1-in-step`.** The admission parks its phase 1, and the engine adds it to the next decode step as one more row next to the running requests (a mixed step); the admission resumes once that step has collected it (`remote_ra_live_v2/phase1_step.py`). Phase 1 keeps the separate pass when nothing else runs (`probe_only`), when its tokens do not fit what the step leaves of the 16,384-token step budget, or when no batch row is free for two steps (`fallback`); a phase 1 that carries `goal_query`'s Goal forward also stays a separate pass, and a server that runs one session at a time (`max_num_seqs` 1) always takes it. Phase 1 is never chunked. With CUDA graphs, a step that carries a phase 1 runs with the piecewise graphs (`piecewise`, `full`) or eagerly (`full-decode`). Native files record `phase1_mode` and the time parked.
- **`--admission-overlap K`.** Up to K admissions are in flight at once: the frontend's admission lock becomes a first-come-first-served semaphore with K permits, and every session gets its own Goal thread and its own copy of the Goal constructor, so while one admission waits for its Goal (a rewriter call) or its in-step phase 1, the others proceed (`remote_ra_live_v2/admission_overlap.py`). The Goal rewriter is still one process that serves one request at a time, and phase 1 still runs for one admission at a time. How long each rewriter call waited for that process and how long it took are recorded per call in `goals.jsonl` (`wait_ms`, `call_ms`) and per request in the native file (`goal_rewrite_wait_ms`, `goal_rewrite_call_ms`, next to `goal_construct_ms`; section 8). Without the flag only one Goal is built at a time, so `wait_ms` stays near 0 and the queueing is the admission wait in front of the engine. The frontend's wait for an admission's acknowledgement and the engine lease grow with K and are pinned at startup: 300 + 120(K − 1) s and 60 s more (2,100 s and 2,160 s at K = 16; 300 s and 360 s without the flag). Nothing is preempted: every admission in flight holds a capacity reservation until it is accepted, and an admission that does not fit next to the running requests and the other reservations is refused with `CAPACITY_WAIT` and retried, as in section 4.4. With all three flags an admission reserves the larger of its two peaks; in the Mamba pool that is 9 blocks for phase 2 (three Gated-DeltaNet slots; 6 without `--gdn-device-handoff`) against 3 for phase 1 (6 with a Goal forward), so about 3 admissions fit at once at 5 sessions (11 slots, 33 blocks) and about 11 at 16 sessions (33 slots, 100 blocks); the others retry (job 3349504 refused further admissions with 3 in flight at 5 sessions and with 10 or 11 in flight at 16). Each refusal writes an `engine_rejected` line with a `traceback` field to the server log (section 6.3): expected, not a crash.
- **On the GPU (job 3349504, runtime `4eae441`, the admission work merged into this revision; one GH200; verified offline, token-exact).** `goal_query` with the Qwen3-0.6B rewriter at 5 and 16 sessions, `--cuda-graphs full`, 24 GiB KV. The offline selection replay does not apply; it covers `observation_query` and `random_fill` only. At 16 sessions, without the flags a request waited 48.3 s on average for admission (86 % of its request time) and the server completed 13.8 turns per minute. With `--admission-overlap 16` the admission wait was 0 and the server completed 17.3 turns per minute (one run); the wait moved into the rewriter, one process serving one request at a time (Goal construction 3.8 s on average without the flag, 33.9 s with it). With all three flags and K = 16 it completed 15.7 turns per minute, with 726 `CAPACITY_WAIT` refusals. At 5 sessions the three servers completed 11.1, 12.4 and 12.4 turns per minute (one run each). The outputs and selections of the flagged servers differ from those of the unflagged server, as batched runs do anyway (section 4.4).
- **On the GPU (job 3349078, runtime `4eae441`, one GH200; verified offline, token-exact, the selection replay equal to the recorded selections).** `observation_query`, 16-token decoupled layout, 24 GiB KV. One session, eager, 8 turns of 6,500-token observations: with `--gdn-device-handoff`, with `--phase1-in-step` and with both, every turn's tokens, selections and Page vectors equal those of the server without the flags, and a repeat of that server equals it too (the phase-1 vectors of all 2,866 Pages identical). 16 sessions, `--cuda-graphs full`, 6 turns of 2,600-token observations: with both flags the engine thread's admission time fell from 342 ms to 55 ms per turn and the decoding requests' stall from 36.8 % to 6.3 % of the run (13.0 % counting the slower mixed steps; 94 % of the continuing turns ran their phase 1 inside a step), the inter-token latency with 16 running rose from 32.0 to 32.9 ms, and the 96 requests finished 1.8 % sooner (73.8 s against 75.2 s). Five sessions with 150-token observations: stall 13.5 % to 4.1 % (12.2 % with the steps), 3.8 % sooner.
- **`--goal-rewriter-batch N` (off by default; `goal_query` with `--goal-rewriter-model` and `--admission-overlap K`, 2 ≤ N ≤ K ≤ 16; Qwen only: the Gemma 4 profile refuses it, section 4.14; on the GPU in SWE-bench job 3404230 with N = 4, below).** With `--admission-overlap 16` the sessions' Goal rewriter calls queue for the one rewriter process. In the verified 16-session SWE-bench run the 16 initializations at startup finished after 19.5, 26.4, 30.4, … 98.5 s, about 5.3 s per call, and a Goal with a rewriter call took 81 s at the median. With this flag the calls that queue are sent together, up to N at a time, to a child that runs them as one batched greedy `generate` (`remote_ra_live_v2/goal_rewriter_batch.py`, `goal_rewriter_child.py`):
  - **Unchanged:** when the rewriter is called, what it is sent (the upstream request line, byte for byte), the prompt, the JSON schema and its constraint, `max_new_tokens`, the parsing and validation of each answer (the upstream `PersistentCommandGoalRewriter.rewrite` code runs for every call) and the fallbacks. Each call gets its own answer; a child error or a refused candidate concerns that call only. A batch of one is the upstream `generate`.
  - **Changed:** the model call. The prompts are left-padded into one `generate`, and each row is constrained by its own request's lm-format-enforcer prefix function. **Batched greedy decoding is not bit-identical to one request at a time:** padding and the batch dimension change the order of floating-point reductions, so a near-tie can flip a token, and a rewritten Goal (or its fallback) can differ from the unbatched server's. Compare policies only between servers with the same setting. The child also builds lm-format-enforcer's tokenizer data once instead of for every request (the same allowed tokens; a self-check at the child's start compares a fresh and a cached build, else the per-request build stays).
  - **Timeouts and failures:** the pinned 120-s timeout runs from the send of the call's batch, so the time queued before it is not charged, as before. A timeout, a child exit or an unreadable answer fails every call of that batch with the upstream error (each falls back), and the next batch restarts the child. A CUDA out-of-memory error runs the batch in halves; any other failure of a batched `generate` runs its requests one by one through the upstream code.
  - **What it pins:** `startup.json` and `ready.json` record `goal_rewriter_batch: N`, and the engine checks once, at startup, that its rewriter runs batched with N. Each `goals.jsonl` exchange of a batched server adds `batch_id` and `batch_size` (section 8). Without the flag the startup config, the rewriter command and the evidence are unchanged (identity probe: 0 differences).
  - **GPU memory:** Qwen3-0.6B needs about 112 KiB of KV cache per prompt token, so a batch of 16 prompts of 4–6K tokens needs 7–10 GiB next to the 27B server. Start with N = 4 and read the prompt sizes (`input_tokens`) in `goals.jsonl`.
  - **Expected effect, not measured:** a batch serves up to N calls in about the time of one when the model's decode step, not the constraint, dominates a call; it does not speed up a call that runs alone. The per-call times in `goals.jsonl` (section 8) separate the call's setup (`call_ms` − `generation_latency_ms`) from its generation.
  - **CPU tests (`test_goal_rewriter_batch.py`):** a fake model that follows Transformers' batched greedy `generate` and a fake enforcer. Each batched answer equals the upstream answer of that request alone, and sixteen sessions' Goals equal the unbatched ones (N = 16 and N = 4). The tests also cover the out-of-memory and failure paths, a timeout, a mismatched answer, a cancelled admission and shutdown. Transformers, lm-format-enforcer and the checkpoint first ran batched on the GPU in job 3404230 (next item).
  - **On the GPU (SWE-bench job 3404230, runtime `5b15b2f`, N = 4, 16 sessions; [docs/SWEBENCH_BENCHMARK.md](docs/SWEBENCH_BENCHMARK.md)):** the run passed its checks (`SWE32_PASS_VERIFIED`), all 32 instances done. 9 of its 59 Goal fallbacks were batch-only failures, 4 CUDA out-of-memory and 5 cuDNN attention errors (none in the unbatched job 3404696); they ended as fallbacks, and a fix of their recovery is in progress on a development branch.
- **Recommended**, and the configuration of our SWE-bench runs: `goal_query` with `--admission-overlap 16` (at 16 sessions) and the other two flags off; every other policy with all three off.

### 4.11 Decode fast path (opt-in)

In the SWE-bench runs, with 16 requests running and 16-token pages, this runtime decoded 1.7 times slower than stock vLLM: the inter-token latency was 52.4 ms against 30.0 ms. The cause is host work in each decode step that grows with the number of running requests times the Pages per request.

- Each running request's per-step `allocation_update` RPC counted its own blocks by scanning every allocation claim of every running request (`remote_ra_http/evidence.py`).
- A long request holds 49 times as many Pages at 16 tokens as at 784 tokens, so the scan grew with the smaller page.
- In the `full` SWE-bench run (job 3357655) these calls took 245.9 s, 13.8 % of the run (321,338 calls, 0.765 ms each).

`--decode-fastpath` (off by default) produces the same values with less work:

- A per-request count of claimed blocks replaces the scan (`remote_ra_live_v2/decode_fastpath.py`).
- The step lease type-checks only the attention leases added since the last step.
- The step positions are read from the device once per step instead of once or twice per request.
- The worker drains the device once per step instead of once per request.
- The action-marker stop decodes only the new token unless that token contains `>`. This is exact because the marker is ASCII and ends in `>`.

**What the flag pins.**

- With the flag, `startup.json` records `decode_fastpath: true`. The worker echoes it in its evidence, `ready.json` records it, and it is checked once at startup.
- Without the flag, the startup config, the evidence and the vLLM command gain nothing, and the code paths are the previous ones (identity probe: 549 items, 0 differences).

**CPU tests (`test_decode_fastpath.py`).**

- The claim counts equal the scan under random changes.
- Leases, positions, refusals and marker decisions are identical, including with the pinned Qwen3.6 tokenizer.
- On the multi-session fakes (16- and 784-token pages, `full` and `observation_query`), the RPC payloads, evidence files, attended Pages and replies are identical with the flag on and off.

**Step profile.** `REMOTE_RA_STEP_PROFILE=1` (an environment variable, off by default) writes one `REMOTE_RA_STEP {json}` line per decode step to the server log.

- Each line holds the host time of each part of the step, and the forward pass timed with CUDA events (no added synchronization).
- `tools/step_profile.py` summarizes the lines.
- It is a diagnostic, not part of the evidence.

**Torch profile.** `REMOTE_RA_TORCH_PROFILE=<absolute directory>` (an environment variable, off by default) makes `serve.py` pass vLLM's own `--profiler-config {"profiler":"torch","torch_profiler_dir":<directory>}`, so the server has vLLM's `/start_profile` and `/stop_profile` routes, and a trace window is opened and closed through them as on stock vLLM (`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` adds vLLM's named ranges).

- The startup config, its pins and the server's environment are unchanged, and `--language-model-only` stays the last argument; unset, the vLLM command is unchanged.
- It cannot be combined with `REMOTE_RA_STEP_PROFILE=1` (one profiler at a time).
- It is a diagnostic, not part of the evidence. CPU-tested only (`test_torch_profile.py`).

**On the GPU (job 3366128, runtime `43fd9c3`, one GH200; verified offline).**

- **Workload.** A decode micro-benchmark: 16, 8, 4 and 1 running requests at about 2K and 20K tokens, and 16 SWE-like histories of 8K to 48K tokens (mean prompt 21.6K).
- **Servers.** This runtime with `--cuda-graphs full` and synchronous scheduling, against stock vLLM 0.24.0 with and without asynchronous scheduling.

Inter-token latency with 16 requests running, on the SWE-like histories:

| Server | ms per token |
| --- | ---: |
| This runtime, 16-token pages | 53.8 |
| This runtime, 16-token pages, `--decode-fastpath` | 36.1 |
| This runtime, 784-token pages | 40.8 |
| This runtime, `observation_query`, `--decode-fastpath` | 31.0 |
| Stock vLLM, synchronous scheduling | 29.2 |
| Stock vLLM, asynchronous scheduling | 27.6 |

- The forward pass took the same time at 16-, 64- and 784-token pages: about 21 ms per step with 16 running.
- The flag halved the host time per step, from 18.5 ms to 9.2 ms.
- With one running request the flag changes little: 20.3 ms at 2K tokens with and without it.
- The KV-cache layout (HND or NHD) made no measurable difference on either server (within about 1 ms).
- The verdict is a partial pass. Each round's measurement window ended when its first request finished, after 30 to 60 decode steps instead of the planned 150. Every server ran all rounds.

**Recommended for every policy** from the next benchmark run on. The SWE-bench runs of [docs/SWEBENCH_BENCHMARK.md](docs/SWEBENCH_BENCHMARK.md) use runtime `5b15b2f` with this flag and `--decode-fastpath2` (jobs 3404696 and 3404230); earlier SWE-bench runs used `e877051`, without it.

**Second stage: `--decode-fastpath2` (opt-in; needs `--decode-fastpath`).** With 16 requests running, job 3366128 still decoded 6 to 7 ms per token slower than stock vLLM (36.1 against 29.2 ms, both with synchronous scheduling), and its step profile put about 5.5 ms of the 9.2 ms host time per step in this runtime. `--decode-fastpath2` (off by default) removes more of that work and produces the same values:

- One `allocation_batch` worker RPC per decode step carries the allocation updates of every scheduled request, instead of one `allocation_update` RPC per request. An update whose rows did not change since the worker last accepted them is sent without rows (`remote_ra_live_v2/decode_fastpath2.py`, `engine_boundary.py`). The worker applies each entry as before, so every native file keeps one `allocation_update` frame per request per step, with the same fields.
- Less host work per request and step: the native allocation arguments are bound without `inspect`, unchanged rows, leases and tuples are reused instead of rebuilt, a request's active KV bytes are counted per KV group, and the action-marker stop remembers per token id whether the token contains `>`.
- The per-request work after the forward pass (lease close, Page compaction, Gated-DeltaNet capture) runs after vLLM has queued the logits computation, so the two overlap.

**What the flag pins.** With the flag, `startup.json` records `decode_fastpath2: true` (refused without `decode_fastpath: true`). The engine sends it to the worker once at startup; `worker-attached.json` and `ready.json` record it. Without the flag, nothing is added and the code paths are those of `--decode-fastpath` alone (identity probe: 549 items, 0 differences).

**Gemma 4** (section 4.14) takes both switches; on its window layout they add the incremental observation of the sliding windows and incremental sliding rows (not GPU-validated).

**CPU tests (`test_decode_fastpath2.py`).** On the multi-session fakes (784- and 16-token pages, `full` and `observation_query`; `goal_query` with the GDN hand-off and with in-step phase 1), the replies, plans, attended Pages, computed positions, evidence files and per-request frames equal those of `--decode-fastpath` alone, and the batches, expanded entry by entry, equal its `allocation_update` RPCs. **On the GPU:** it ran in the SWE-bench jobs 3404696 and 3404230 (runtime `5b15b2f`), but its effect on the inter-token latency has not been measured.

**Third stage: `--decode-fastpath3` (opt-in; needs `--decode-fastpath2`).** With both fast paths on, most of the host work this runtime still adds to a decode step before the forward is a fixed cost per running request: the worker's step (observation, claims, materialize, positions, the step lease), the FullAttention row check and the per-request allocation path. `--decode-fastpath3` (off by default) cuts part of it and produces the same values:

- A request whose step appends no FullAttention block and keeps its Mamba rows at the same slot index (15 of 16 steps at 16-token pages) keeps its last step's Mamba leases, key tuples and (empty) claim differences, after the same list, device-row and Mamba checks (`remote_ra_live_v2/decode_fastpath3.py`, `allocation_owner.py`). The device row is compared by its bytes, the same answer as `np.array_equal` for one integer dtype (`allocation_owner.py`, `worker_rows.py`).
- Such a request skips `materialize` when it would only mark the step, and skips its IO-attribution scope when nothing in it can issue IO. Its step lease is built without re-checking Mamba slots that were checked when they were accepted (`qwen.py`).
- The step's position columns are read from the GPU in one copy instead of two.
- On the worker, an allocation update sent without rows is applied and measured directly, with the same frame; the engine binds the modules it imports per request once (`engine_boundary.py`).

**What the flag pins.** With the flag, `startup.json` records `decode_fastpath3: true` (refused without `decode_fastpath2: true`, and for the Gemma 4 profile). The engine sends it to the worker once at startup; `worker-attached.json` and `ready.json` record it. Without it nothing is added and the code paths are those of `--decode-fastpath2` (identity probe: 555 + 219 items, 0 differences).

**CPU tests (`test_decode_fastpath3.py`).**

- The byte compare equals `np.array_equal`, and the one-copy position read equals the two reads, on random inputs.
- The owner's steady observation and materialize equal the full per-step path on randomized step sequences (Page and Mamba crossings, row moves, finishes with block reuse, corruptions): the same results, refusals and state.
- The new step lease equals the old one on random lease sequences, refusals included.
- On the multi-session fakes (784- and 16-token pages, `full` and `observation_query`; `goal_query` with the GDN hand-off and with in-step phase 1; also with vLLM-shaped block tables and MRoPE positions, where the steady paths run), the replies, plans, attended Pages, computed positions, evidence files, per-request frames and RPCs equal those of `--decode-fastpath2` alone. The longer cases run with `REMOTE_RA_SLOW_TESTS=1`.
- CPU micro-benchmark (x86 fakes, 16 requests at 20K tokens, 16-token pages, median of 7 runs): a decode step, all host work including the fakes' own forward, takes 6.15 ms with `--decode-fastpath2` alone and 5.25 ms with `--decode-fastpath3` (the per-step mean; 5.44 and 4.70 ms at the median step).

**Not GPU-validated:** its effect on the inter-token latency has not been measured.

### 4.12 Text only: `--language-model-only` (opt-in)

Qwen3.6-27B is a multimodal checkpoint, but this runtime serves text only. Without this flag, vLLM 0.24.0 still loads the vision tower and profiles its encoder cache with a dummy image. Before the server reports ready, it also warms up the multimodal processor with fake image and video inputs sized to the context limit (job 3357655 logged `Multi-modal warmup completed in 13.884s`). For under a second this warmup takes tens of GiB of host memory, enough to get a server killed before READY in a job that requests only one GH200's share of the node's host memory.

`serve.py --language-model-only` (off by default) passes vLLM's own `--language-model-only`, which sets every multimodal limit to 0. In the pinned vLLM 0.24.0 sources this means:

- The vision tower is replaced by a placeholder and is never built or loaded (`_mark_tower_model` in `vllm/model_executor/models/interfaces.py`; `qwen3_5.py` builds `visual` inside it).
- The renderer creates no multimodal processor, so neither multimodal warmup runs (`vllm/renderers/base.py`), and no encoder cache is set up or profiled.
- The language model's weights are the same. The KV cache is still `--kv-cache-memory-bytes`, so its blocks and pools and the launcher's KV and host-memory checks are unchanged. The tower's weights (about 0.9 GiB in BF16, computed from the vision config, not measured) stay free on the GPU.
- One computation changes. In text-only mode vLLM runs the QK-norm, RoPE and output-gate split of the 16 full-attention layers as one fused kernel instead of separate operations (`Qwen3NextAttention` in `qwen3_next.py`). The arithmetic is the same (every step the runtime checks that the three mRoPE rows equal the logical positions), but outputs need not be bit-identical to those of a server without the flag. The runner also passes token ids to the model instead of the embeddings it looked up itself; that change is exact (the model does the same lookup).

**Pass the same flag to the stock vLLM control**, so that both sides load the same model and run the same kernels, and compare only servers that agree on it. The stock command in [docs/SWEBENCH_BENCHMARK.md](docs/SWEBENCH_BENCHMARK.md) passes it; `examples/slurm/serve_gh200.sh` does not.

**What the flag pins.**

- `startup.json` records `language_model_only: true`, and its `effective_server_config` holds `language_model_only: true`; the frontend compares vLLM's parsed argument with it.
- The engine checks its resolved multimodal configuration. The worker checks that vLLM resolved the image and video limits to 0 and holds no vision-tower tensor, and echoes `{"mm_limits": {"image": 0, "video": 0}, "vision_tower_tensors": 0}` in `worker-attached.json`; `ready.json` records it. Each check runs once, at startup (`remote_ra_live_v2/language_model_only.py`).
- Without the flag, the vLLM command, the startup config and the evidence gain nothing (identity probe against the previous revision: 555 + 219 items, 0 differences), and the frontend, the engine and the worker refuse a vLLM that runs language-model-only anyway.

**CPU tests (`test_language_model_only.py`):** the launcher, the startup checks, the worker evidence and `ready.json`; with the flag, the multi-session fakes produce the same RPC payloads, evidence files and tokens.

**On the GPU:** the Terminal-Bench dry run of `full` (section 6.7) was the first run with the flag; its log had vLLM's text-only line and no multimodal warmup. The SWE-bench runs of jobs 3404696 and 3404230 (`goal_query`) ran with it too.

### 4.13 Releasing idle sessions' pages (opt-in)

Between turns a session pins its resident pages on the GPU (the previous turn's selected pages plus the pages sealed in it) and its live tail page; its Gated-DeltaNet carry is on the host, and every sealed page also has a host copy in the pinned page arena. With 16 sessions at a 32,768-token budget and 16-token pages, an idle session pins about 2,300 pages and the 16 together about 37,000, more than a 25 or 32 GiB KV pool holds (the capacity rule of section 4.9). Without this option the capacity gate then waits for a running request to finish. Once every session is idle nothing can free a block, and every next turn waits for good (before this revision the first of them was admitted because nothing ran, and failed for lack of blocks).

`--release-idle-pages` (off by default; decoupled layout only) lets the capacity gate release pages instead (`remote_ra_live_v2/idle_release.py`):

- **Victims.** When a turn does not fit, the gate covers its attention-pool shortfall by releasing whole idle sessions' resident pages, least recently used first. Idle means open, with no request admitted or running, and not the session being admitted. A session whose next request has reached the server but still waits for admission is idle: its released pages come back before that request's phase 1 (otherwise a pool pinned by sessions whose next requests all wait could never free a block). The live tail page is never released. If the idle sessions together cannot cover the shortfall, nothing is released and the turn waits with `CAPACITY_WAIT`, as before. A turn the gate admits because nothing runs first releases what it needs; with `--admission-overlap K` the same holds when, in addition, no admission is in flight (fixed in this revision: the ledger gate admitted such a turn without the release).
- **Readmission.** The released pages come back from their host copies before the session's next phase 1, which attends to them. That turn then runs exactly as if they had never left; the gate counts the readmitted pages in both of its peaks.
- **No waiting inside an admission.** A release and a readmission are synchronous engine operations between steps. A refusal is the usual immediate `CAPACITY_WAIT`, so the frontend's 300-s acknowledgement wait covers only an admitted turn's own work.
- **What it pins.** `startup.json` records `release_idle_pages: true`. The engine sends it to the worker once in its startup record; the worker checks it and only then accepts the release and readmission operations; `worker-attached.json` and `ready.json` record it. Without the option nothing is added and the code paths are the previous ones (identity probe: 549 items, 0 differences).
- **Evidence.** `capacity.jsonl` in the output directory has one row per release and per readmission (`schema` `remote-ra-capacity-event-v1`: the session, its pages and blocks, both pools' free blocks before and after, the admitting request and its shortfall). Every native file and `runtime.jsonl` row adds `idle_release` (the sessions released to admit that request, its readmitted pages), and `engine-cleanup.json` adds the totals.

**CPU tests (`test_idle_release.py`).** 16 sessions on the batched engine and worker fakes at 16-token pages, `observation_query` with `--active-token-budget 32768`, 12 turns each with observations of about 3,600 tokens (prompts up to 39,845 tokens, 2,048 selected pages on the last turns):

| Attention pool | Without the option | With it |
| --- | --- | --- |
| 25,600 blocks (the idle pins reach 36,394) | turn 129 of 192 waits for good (`CAPACITY_WAIT` with nothing running; before this revision it was admitted and failed): the 16 idle sessions pin 25,406 blocks, 193 are free, phase 1 needs 224 | all 192 turns; 35 releases (65,090 pages), 29 readmissions (51,446 pages), 1 `CAPACITY_WAIT` |
| 48,000 blocks | all 192 turns | all 192 turns, nothing released |

With the option, every session's prompts, selections, attended pages and replies equal those of the 48,000-block run. A shorter case (900 against 9,000 blocks) runs in the suite with `observation_query` and in the slow pass (section 2.3) with `full` too; the 32,768-token case takes about 5 minutes and runs in the slow pass. Not measured: the readmission's host-to-device time on the GPU, and how often a session is released again before its next turn under a real workload (least recently used is, under round-robin turns, the session that runs next).

**On the GPU:** MemoryAgentBench job 3397842 (Qwen3.6, runtime `0d37346`, 16 sessions) ran with the option at two pool sizes: 139 releases and 124 readmissions with a 16 GiB pool, 13 and 13 with 24 GiB. All 32 cases were identical between the two pools, and every readmission restored exactly the Pages a completed turn had committed. The job's one failed check (route-b) was a bug of the harness's check, which counted releases in the HTTP tail after the engine had finished the turn; with the corrected check it passes (verified offline). The Terminal-Bench dry run of `full` (section 6.7) released idle sessions' pages without a readmission.

### 4.14 Gemma 4 31B IT (stage 1, not GPU-validated)

This revision holds stage 1 of a second model profile: [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) at revision `842da3794eaa0b77d5f08bae87a17459d91ff475`. It installs; its engine and worker, with the lifecycle of the sliding-window layers, are tested end to end on the CPU and its startup checks on their own; `serve.py --start` starts it. **On the GPU** (job 3397695, runtime `e0df9d5`, one GH200, verified offline) its two-pool layout served with the selective runtime off (plain serving through this build): 16-token pages started (pools of 15,867 and 2,886 blocks, 26.41 GiB of KV), and so did 32-token pages. Against the stock-vLLM replay of job 3364469 (10 trials), 32-token pages were token-exact on 10 of 10 trials; 16-token pages, with and without CUDA graphs (eager), diverged from turn 3 on (the first difference at token 3 to 5; the three servers agree on the first 3 turns), a numerics or tiling difference of the TRITON_ATTN window layout. Selection, the kv-session and chat-text protocols and the MemoryAgentBench path have not run on a GPU yet (jobs queued). The startup checks that depend on the loaded model (the attention backend's cache shape, the final norm, the vision tower's names) fail closed. A Qwen release is unchanged: its vLLM command, startup config, pins and evidence are byte-identical to those of the previous revision (identity probes over 555 + 219 items, 0 differences). The profile also serves the MemoryAgentBench kv-session protocol with Gemma's turn grammar ("MemoryAgentBench" below, section 6.6), on the CPU only.

**Model and selection.**
- 60 decoder layers: 50 sliding-window layers (window 1,024 tokens, 16 KV heads × 256) and 10 global full-attention layers (5, 11, …, 59; 4 KV heads × 512, K and V both cached).
- Pages, budgets, mandatory and recent Pages, vectors and restores apply to the global layers only: 81,920 bytes per token, so a 16-token Page is 1.25 MiB (Qwen: 1 MiB).
- The sliding layers always attend to the exact last 1,023 tokens plus the current one, as in stock vLLM. Their window is not counted in `--active-token-budget`.
- The selection query is the hidden state before the final norm at layer 59 (the last layer, a global one), projected with the same seeded matrix: SHA-256 `852b42e4…a1dfb7` for hidden size 5,376.

**Windows across turns.**
- Each session keeps its sliding window on the GPU between turns: at the end of a turn the engine pins, per sliding group, the blocks the next query reads (at most ⌈1,024 / P⌉ + 1).
- The next turn's phase 1 computes the new tokens on that window plus fresh blocks. Phase 2 starts from the window phase 1 ended in, behind null blocks, so vLLM never allocates the blocks before the window; the older blocks are freed then.
- The worker checks both rows and keeps the window after the computed end as the session's carry; there is no Gated-DeltaNet state. A rebuild or a closed session frees its window.

**Install.** `--model-profile gemma4-31b-it` checks the model directory against the profile's pins in `INSTALL_RELEASE.json` (`model_profiles`: `chat_template.jinja`, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`) and records the profile in the release manifest. The files are read from the model directory, not copied.

```sh
"$VENV/bin/python" -B install.py --site-packages "$VENV/lib/python3.12/site-packages" \
  --model-dir "$MODELS/gemma-4-31b-it" --release-dir "$RELEASE-gemma4" --model-profile gemma4-31b-it
"$VENV/bin/python" -B install.py --site-packages "$VENV/lib/python3.12/site-packages" \
  --model-dir "$MODELS/gemma-4-31b-it" --release-dir "$RELEASE-gemma4" --model-profile gemma4-31b-it --apply
```

**Dry run.**

```sh
python -B serve.py --release-dir "$RELEASE-gemma4" --task /abs/path/to/tasks-g0.json \
  --policy observation_query --active-token-budget 8192 --page-tokens 16 --kv-cache-gib 28 \
  --max-model-len 131072 --max-tokens 4096 --language-model-only --output "$RUNS/gemma-obs" --port 8000
```

What the Gemma profile pins:

| Setting | Value |
| --- | --- |
| attention backend | `TRITON_ATTN` (FlashAttention 3 has no 512-wide heads; FA4, vLLM's default for this model, is not validated with this runtime). Give the stock vLLM control the same backend. |
| text only | `--language-model-only` is required (no vision tower) |
| KV layout | two pools with blocks of P tokens: the global layers (P = 16; `--page-tokens 32` is the fallback) and the sliding-window layers, reserved at startup: per sliding group ⌈1,024 / P⌉ + 1 blocks per session plus one forward's transient, and for `goal_query` the Goal forward's transient, ⌈2,048 / P⌉ blocks (`--additional-config {"remote_ra_kv_layout": {"page_tokens": 16, "state": "sliding", ...}}`, with `"goal_tokens": 2048` for `goal_query` only) |
| one native forward | 8,192 tokens (`--max-num-batched-tokens 8192`); a longer phase 1 runs in forwards of at most 4,096 tokens (below) |
| context | 65,536, 131,072 or 262,144 tokens; 262,144 fits pool 0 under the selective rule (below) or one session at 32 GiB: the strict and release rules need one session at the limit in pool 0 |
| `--kv-cache-gib` | 24, 28 or 32 |
| protocol | Mini-SWE-Agent XML with Gemma's turns, `mswea_xml_gemma4_v1` (below), the kv-session protocol with Gemma's grammar, `--protocol kv-session --kv-session-thinking off` ("MemoryAgentBench" below), or the chat-text protocol with Gemma's turns (`--protocol chat-text`, below) |
| parsers | `gemma4` reasoning and tool-call parsers, `--chat-template-content-format openai` |
| host memory | one GH200's share of a node (`--mem=113560M`); `--host-memory-shares 2` for a job of two shares (`--mem=227120M`, below) |
| refused | `--gdn-device-handoff`, `--phase1-in-step`, `--decode-fastpath3` (Qwen only), `--kv-session-thinking on` and `--goal-rewriter-batch` (it batches `goal_query`'s rewriter calls; not checked with the window layout yet) |

The protocol is that of section 6.1 with Gemma's turns:
- System and task are rendered through the pinned template as OpenAI text parts, as vLLM hands them to this template (one more space token than plain strings).
- A user turn is `<|turn>user\n{content.strip()}<turn|>\n`; the generation header is `<|turn>model\n<|channel>thought\n<channel|>`, the template's prompt with thinking off.
- The empty thought channel stays in front of every reply in the history (append-only; Gemma's own template drops it when it re-renders a conversation, 4 tokens per turn), and a rebuild re-encodes past replies after it.
- The stop ids are 0, 1, 50 and 106 (pad plus `generation_config.json`'s end ids, which include the end of a model turn).

**Chat-text protocol (Terminal-Bench 2.0 with Terminus 2; CPU only).** `--protocol chat-text` (section 6.7) runs on a Gemma release with Gemma's turns (`remote_ra_live_v2/chat_text_protocol.py`, `GEMMA4_CHAT`). The startup config pins `protocol` `chat_text_v1` next to `model_profile`; its `chat_text` record carries Gemma's generation header and template arguments (`header_text` `<|turn>model\n<|channel>thought\n<channel|>`, `template_kwargs` `{"enable_thinking": false}`), and the server checks the pieces below against the pinned template once at startup. Qwen's record and turns are unchanged.
- A fresh or rebuilt session renders the initial segment (the system message, if any, and the first user message) through the pinned template as OpenAI text parts with thinking off: it starts with `<bos>`, and a system message gets one more space, as stock vLLM renders it. Every later message follows as the continuing stream holds it: a user message as `<|turn>user\n{content.strip()}<turn|>\n`, a run of assistant messages as one model turn (the template merges such a run too) after the empty thought channel, `<|turn>model\n<|channel>thought\n<channel|>` + each content after the template's `strip_thinking` + `<turn|>\n`. Then the header.
- A continuing request appends the reply's generated ids, its boundary (`\n` after a stop id; `<turn|>\n` after a reply cut at its bound) and the new user message, then the header. The stop ids are 0, 1, 50 and 106; `finish_reason` is `stop` on a stop id and `length` at the bound; the reply is decoded without special tokens.
- Each prompt is the template's render of its messages with the empty thought channel kept after every past `<|turn>model\n`, and without those channels (`<|channel>thought\n<channel|>`, 4 tokens per past reply) it is stock vLLM's prompt token for token, as long as every reply re-encodes to its own ids. With the pinned tokenizer the chat-text transcript also reproduces the 10 prompts that stock vLLM served in the stage-0 session (job 3364469). A stock Gemma control with vLLM's default template arguments (thinking is off by default in Gemma's template) therefore re-renders each past reply 4 tokens shorter, as Qwen's stock control does (section 6.7).
- Sessions, rebinds, asides, closes, the context-limit refusal and the reply bound are section 6.7's, unchanged; `GET /v1/chat-text/info` adds `model_profile`. Up to 24 sessions are open (K, the window slots of pool 1); `goal_query` is refused (section 6.7).
- Capacity at the Terminal-Bench settings (K = 20 for Harbor's 16 trials, 65,536 tokens, 4,096-token replies, P = 16): pool 1 holds 20 windows and one forward's transient, 9,061 block ids (22.12 GiB). `--kv-cache-gib 28` leaves 4,815 global Pages: `observation_query` and `random_fill` with `--release-idle-pages` and an 8,192-token budget need 2,630 (the selective rule); `full` needs 4,933 (the release rule), so 28 GiB holds 19 `full` sessions and 32 GiB 24. Without `--release-idle-pages` the strict rule needs 4,914 Pages at K = 20 and the idle sessions' Pages would fill pool 0, so the switch is on. By the estimate above, 32 GiB at K = 20 leaves less than 2 GiB of the GH200 (peak about 96,100 MiB, 97,400 with the Qwen runtime's overheads); 28 GiB peaks at about 92,000 MiB (93,300). Host: `--dram-arena-gib 64` holds 52,428 Pages of 1.25 MiB (838,848 tokens; Qwen's 64 GiB holds 1,048,576), 65.0 GiB pinned with the 1 GiB connector pool; without an arena 20 sessions at 65,536 tokens could pin 101.1 GiB, which the launcher refuses.

Terminal-Bench, dry run:

```sh
python -B serve.py --release-dir "$RELEASE-gemma4" --protocol chat-text --policy observation_query \
  --active-token-budget 8192 --release-idle-pages --page-tokens 16 --max-running-sessions 20 \
  --kv-cache-gib 28 --dram-arena-gib 64 --connector-pool-gib 1 --max-model-len 65536 --max-tokens 4096 \
  --cuda-graphs full --decode-fastpath --language-model-only --output "$RUNS/tb-gemma-obs" --port 8000
```

**Capacity.**
- The sliding windows dominate the GPU pool: at P = 16 one session's window is 325 block ids of 2.5 MiB (812.5 MiB), and one 8,192-token forward adds 2,560 block ids (6.25 GiB) once per server.
- The launcher refuses a KV size whose global pool cannot hold one session's worst turn at the context limit. At 131,072 tokens with 4096-token replies, at most 8, 13 and 18 sessions fit 24, 28 and 32 GiB.
- Sixteen sessions (the group's Vanilla runs) need `--release-idle-pages` (section 4.13), which the Gemma profile takes: under GPU pressure the gate releases idle sessions' global Pages and readmits them before the session's next phase 1; the sliding windows always stay on the GPU. With a query policy (`goal_query`, `observation_query` or `random_fill`) and a fixed `--active-token-budget`, the launcher then sizes the global pool by the selective rule: one session's worst selective turn (its resident Pages, one selection of restores, one forward and one reply: 2 × (512 + 769) + 1 = 2,563 Pages at an 8,192-token budget and 4,096-token replies), the other sessions' tails and the 1 % watermark. 16 sessions at 131,072 tokens need 22.13 GiB (18.95 GiB of windows and transient, 3.18 GiB of Pages), so `--kv-cache-gib 24` holds them, below the 26.41 GiB a stock vLLM server gives its KV cache; 24, 28 and 32 GiB hold at most 18, 23 and 28 sessions (`goal_query`: 16, 21 and 26, below). Without the switch the rule above holds.
- With `--release-idle-pages` and no fixed budget (`full`, `random50` or `--active-ratio`) the launcher sizes the global pool by the release rule: one session's worst turn at the context limit, the other sessions' tails and the 1 % watermark. Idle sessions release all their Pages under pressure and readmit them before their next phase 1; for `full` that is the session's whole history. 16 `full` sessions at 131,072 tokens with 4096-token replies need 30.02 GiB (exactly 30,738 MiB; 9,083 Pages against the 10,692 of 32 GiB); 24 and 28 GiB hold at most 8 and 13 (at 65,536 tokens 28 GiB holds 19). `full` runs at `--kv-cache-gib 32`, as Qwen's (`GEMMA4_FULL_KV_CACHE_GIB` in `serve.py`): 1,608 Pages (1.96 GiB) above the rule, and the refusal of a smaller size names 32. `full` never starts the Goal rewriter: it has no selection contract, and only `goal_query` starts one. GPU memory, estimated and not measured: 57.9 GiB of weights and 32 GiB of KV are 92,068 MiB of the GH200's 97,871 MiB. With the stock Gemma server's overheads (READY 87,601 MiB and peak 90,099 MiB at 26.41 GiB of KV: 1,257 MiB at READY beyond weights and KV, 2,498 MiB more at the peak) and about 300 MiB of the runtime's own (phase-1 rows, readmission staging), READY comes to about 93,300 MiB and the peak to about 96,100 MiB, 1.7 GiB below the total; with the Qwen runtime's overheads at 32 GiB (2,215 and 3,092 MiB) the peak is 97,375 MiB, 0.5 GiB below. A memory probe decides: if its peak leaves less than about 2 GiB, `full` falls back to 30.5 GiB (391 Pages above the rule; the launcher takes 24, 28 and 32 only, so that needs a launcher change). Readmission volume, replayed from the Qwen swe16 `full` run (job 3357655, 32 tasks, 16 open at a time, histories scaled to Gemma tokens, each turn taken as an instant with least-recently-used releases): at 32 GiB 18 % of the turns readmit Pages, 1.5 GiB on average (median 1.45 GiB, p90 2.65 GiB, at most 4.1 GiB of host-to-device copies), 278 MiB per turn over all turns; 28 GiB would give 36 % and 546 MiB but holds 16 `full` sessions only below a 104,880-token context.
- vLLM's own startup check asks pool 0 for one request at `--max-model-len` next to pool 1: 16 windows at 131,072 tokens need 28.95 GiB (30.51 GiB for `goal_query`), at 262,144 tokens 38.95 GiB. Before this revision it would therefore have refused 16 sessions at 131,072 tokens below 32 GiB at startup, although the launcher's selective rule accepted them (no 16-session Gemma server has started on a GPU yet), and 16 sessions at 262,144 at every size. Under the selective rule the window layout now pins the rule's figure, `"request_pages": R` in `--additional-config` (`kv_layout.WindowLayout.request_pages`, `window_capacity.pinned_request_pages`, checked by `remote_ra_http/config.py`), and vLLM's check asks pool 0 for R global Pages for one request instead (`vllm/v1/core/kv_cache_utils.py`). The pools, the gate and the refusals are unchanged; the strict and release rules keep vLLM's check.
- Under the selective rule a first turn or a rebuild needs its whole row. One that could not fit even after every idle session is released and every running request has finished is refused with `CAPACITY_EXCEEDED` (not a `CAPACITY_WAIT`, so the client does not retry it) instead of waiting forever.
- How often Pages are released depends on the pool: 16 idle SWE-bench sessions hold about 16 × (512 + the last turn's Pages) at an 8,192-token budget, more than the 4,138 Pages of 24 GiB, so admissions release and readmit Pages there (a few hundred MiB of host-to-device copies per readmission); 28 GiB gives 7,415 Pages and 32 GiB 10,692.
- The engine's admission gate checks both pools. Global Pages follow the Qwen rules; in the sliding pool a turn needs its fresh window blocks in phase 1 and, in phase 2, what lifts the adopted window to its peak, and every running request may still grow to its own peak. A rebuild's sliding need credits the window its render frees, as its global need credits its Pages and tail (fixed in this revision: with every window slot taken a rebuild looked short of pool 1 by its own window, so no idle session was released for its global shortfall, and once admitted with nothing running it waited for global Pages forever). As for Qwen (section 4.4), a turn while nothing runs (and nothing is in flight) is checked when another open session pins global Pages, a tail or a window, and waits (`CAPACITY_WAIT`) when it does not fit after the releases `--release-idle-pages` makes; before integrate-1018 the window layout admitted such a turn unchecked, and its decode could find no block (vLLM preempts, which is fatal here) or wait forever. With no other session pinning anything it is admitted unchecked, as before.
- Host memory: the launcher assumes one GH200's share of a node (`--mem=113560M`) and 30 GiB for the processes, so the connector pool plus the Page arena must stay within 80.9 GiB. The OffloadingConnector's CPU pool is idle in this runtime, so a Gemma server takes `--connector-pool-gib N` (section 4.9) and uses 1 GiB when the option is left out; the pool is always pinned (`cpu_bytes_to_use`, `startup.json` `connector_pool_bytes`) and counted in the budget. At 131,072 tokens 7 sessions fit without `--dram-arena-gib`; the largest arena is 79 GiB, about 86K tokens per session at 12 sessions. Sixteen sessions take `--dram-arena-gib 79` (below). Replayed from the Qwen swe16 runs of the first 32 SWE-bench Verified instances (16 open at a time, five policies), with every history re-tokenized by Gemma's tokenizer (3.7–16.4 % more tokens than Qwen), the 16 open sessions' Pages peak at 26.5–37.75 GiB and the 16 longest histories together take at most 65.25 GiB; the longest single history is 96,336 tokens. One share (`--mem=113560M`) is therefore enough; two shares are not needed. `--host-memory-shares 2` (the Gemma profile only; Qwen keeps its 128 GiB budget) makes the launcher assume a job of two shares (`--mem=227120M`): 191.8 GiB pinned, an arena of at most 190 GiB at 262,144 tokens; MemoryAgentBench's 16 longest cases need it (below).

**Chunked phase 1.** A phase 1 longer than one native forward (8,192 tokens; Mini-SWE-Agent's 10,000-character cap allows about 10.2K tokens of digit-dense output) runs as forwards of at most 4,096 tokens on one lease (`model_profile.GEMMA4_PHASE1_CHUNK_TOKENS`); a phase 1 of at most 8,192 tokens is one forward, as before.
- The global row holds the whole run from the first forward on, and only the first forward zeroes its fresh Pages. Each forward's sliding rows hold the window it reads and its new tokens; the fresh sliding blocks whose tokens left the window are reused for later tokens, so phase 1 holds at most ⌈(1,023 + 4,096) / 16⌉ + 1 = 321 fresh blocks per sliding group, within the one forward's transient (512 at P = 16): pool 1 is unchanged. The lease names the later forwards (`chunks`), which the worker checks (they tile the run, carry the window, never reuse the borrowed window or a block still read) and runs in order: the borrowed window is checked before the first forward, and the window phase 2 starts with is captured after the last. The Page vectors and the observation query are those of the whole run.
- The launcher's rules assume a run of one forward. A longer one is served from the pool's margin (at 24 GiB and 16 sessions 251 global Pages for `goal_query` and 1,518 for `observation_query`, about 4,000 and 24,000 tokens of run) or, when even releasing every idle session cannot make room, refused with `CAPACITY_EXCEEDED`; it never waits forever.
- The computation is split differently from one 10K-token forward (stock vLLM splits such a prefill at 8,192 tokens), so such a turn's outputs may differ from stock vLLM's in the last bits; not measured.

**`goal_query`.**
- The Goal is section 5's, unchanged: its maintenance, the rewriter and its prompts, the gate and the Goal text. The engine builds each turn's Goal context from Gemma's transcript as from Qwen's: the root objective (the task message's `<pr_description>`), the session's previous decoded reply (the generated text after the empty thought channel, without `<turn|>`: what the client echoes back), the latest observation and its `<returncode>`.
- The Goal forward is a separate cold forward of the `<retrieval_query>` text, encoded by Gemma's tokenizer and preceded by `<bos>` (id 2), as every Gemma sequence starts (the PoC encodes the Qwen Goal without special tokens; Qwen has no BOS). Its query vector is the same as the observation's: the hidden state before the final norm at layer 59 of the Goal's last token, projected with the 5,376-wide pin, in the space of the Page vectors.
- It runs after phase 1, while phase 1's fresh sliding blocks are still held, so pool 1 reserves its own transient: ⌈G / P⌉ blocks per sliding group for a Goal of at most G = 2,048 tokens (`model_profile.GEMMA4_GOAL_TOKENS`; Qwen's Goal forwards in the 16-session SWE-bench run, job 3363871, took at most 1,164 tokens). A longer Goal fails its turn with `PHASE1_TOO_LONG` before anything is allocated. The gate counts ⌈G / P⌉ global Pages and sliding blocks per group for a turn whose selection is active.
- Memory: 640 block ids, 1,600 MiB, at either Page size. With 16 sessions at 131,072 tokens, an 8,192-token budget and 4,096-token replies the global pool keeps 2,858 Pages at 24 GiB (`observation_query`: 4,138) and 6,135 at 28 GiB (7,415), against the 2,606 and 2,639 the selective rule needs: 16 `goal_query` sessions need 23.70 GiB (22.13 without the Goal), and 24, 28 and 32 GiB hold at most 16, 21 and 26. A Goal transient of a whole 8,192-token forward (6.25 GiB) would not fit 16 sessions at 24 or 28 GiB. The Goal rewriter (Qwen3-0.6B, about 2 GiB) shares the GPU with the model and the KV pools; that has not been measured for Gemma.
- `--admission-overlap K` (section 4.10) is taken, with the same startup record as for Qwen (`--phase1-in-step` and `--gdn-device-handoff` stay refused). The global pool is reserved as for Qwen. In the sliding pool each admission in flight reserves what its turn keeps once accepted, and the phase-1 transients (fresh blocks, the Goal forward) are reserved once, the largest in flight: phase 1 runs one lease at a time, inside one engine-thread call (an admission parks at its Goal, before its phase 1, and nowhere after it). Without this, each admission's phase-1 peak would hold a forward's worth of pool 1, and with 2,600- to 4,300-token observations only one or two admissions would fit in flight. On the CPU fakes (16 sessions), all 16 turn-1 admissions park together, against 6 with one phase-1 peak reserved per admission. At 24 GiB the global pool still bounds the admissions in flight: each continuing turn reserves its row and its restores, so the idle sessions' Pages are released for them (section 4.13). With 16 sessions: `goal_query --admission-overlap 16 --release-idle-pages --active-token-budget 8192` at 24 or 28 GiB.

Sixteen sessions, dry run:

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

**Serving configuration: CUDA graphs and the decode fast paths (not GPU-validated).** A Gemma server takes `--cuda-graphs piecewise`, `full` or `full-decode`, `--decode-fastpath` and `--decode-fastpath2` with the pins and the startup checks of sections 4.8 and 4.11; without them its vLLM command, startup config and evidence are unchanged. Together they give Gemma the serving configuration of Qwen: `--cuda-graphs full --decode-fastpath --decode-fastpath2`. They combine with every policy, `--admission-overlap K` and `--release-idle-pages` (CPU: `test_gemma_flags`).
- **CUDA graphs.** The global group is the window layout's only FullAttentionSpec group (vLLM's SlidingWindowSpec is not one). With FULL graphs the runner binds it to the physical seq-len buffer at capture and at replay, and it keeps the physical slot mapping; the five sliding groups keep vLLM's logical seq lens, positions and slot mapping. Every buffer a captured graph reads is one of the runner's persistent buffers, rewritten in place before each replay (CPU test: the overlay runner's metadata build on the six groups gives each group the same buffers at capture and at replay). The compiled modes need attention and the KV-cache update as splitting ops (there is no GDN op). Phase 1, its chunks and the Goal forward run eagerly. At 16 sessions the capture sizes are 1, 2, 4, 8, 16, 24 and 32. torch.compile (`piecewise`, `full`) changes bf16 numerics as for Qwen: compare with stock vLLM on `TRITON_ATTN` in the same mode.
- **`--decode-fastpath`.** Qwen's fast paths plus the window owner's incremental observation: a step extends the last observation when the full one would accept the same rows. The full-attention leases grow by the appended blocks, each sliding group's leases by the blocks that entered its window minus those that left it, and the claims and allocation sets move by these differences; the leases of a window that did not move are neither rebuilt nor checked again.
- **`--decode-fastpath2`.** Qwen's one `allocation_batch` RPC per step and the after-forward work behind the logits, plus incremental sliding rows (`remote_ra_live_v2/window_rows.py`): the engine follows each sliding window instead of rebuilding whole rows, a changed request's update carries per KV group only the window's first index, the row's length and the appended (block, incarnation) pairs, an unchanged one carries no rows, and the worker keeps its sliding rows compact. At 131,072 tokens a changed update carried about 41K pairs (five sliding rows of 8,192 entries and the global row); it now carries the appended ones. A request's first update is still whole.
- **vLLM's runner rows.** vLLM's runner only appends a request's new block ids; EngineCore nulls the blocks that left a window in its own rows only. The worker therefore checks a runner's sliding row against EngineCore's on the live window, which holds every read of the step, not the whole row. The stage-1 check compared whole rows and would have refused a real runner at the first window slide; the CPU fakes now keep their rows (and numpy block tables) as vLLM 0.24's runner does.

Sixteen sessions with the serving configuration, dry run:

```sh
python -B serve.py --release-dir "$RELEASE-gemma4" --task /abs/path/to/tasks-16.json \
  --policy observation_query --active-token-budget 8192 --release-idle-pages --page-tokens 16 \
  --kv-cache-gib 24 --dram-arena-gib 79 --max-model-len 131072 --max-tokens 4096 \
  --cuda-graphs full --decode-fastpath --decode-fastpath2 \
  --language-model-only --output "$RUNS/gemma-obs16-fast" --port 8000
```

**To check on the GPU** before the serving configuration is used: startup and capture on `TRITON_ATTN` with the window layout (FULL decode graphs for the six groups, the worker's graph record, capture time and memory next to 24 GiB of KV); token parity of `full-decode` against eager and of `full` against stock vLLM in `full` on the stage-0 append session; one session of `full` and `observation_query` at 131,072 tokens with the serving configuration against the eager server (the fast paths alone must give the eager tokens; graphs may change them, section 4.8); the inter-token latency at 1 and 16 running requests against stock vLLM on Triton (20.6 and 27.0 ms in stage 0) with the step profile.

**MemoryAgentBench: the kv-session protocol (gemma-mab, not GPU-validated).** `--protocol kv-session` (section 6.6) runs on a Gemma release with Gemma's turn grammar; the startup record `kv_session` adds `"grammar": "gemma4"` and `/v1/kv-session/info` names it in its options. The wire format, the slots, the context-limit refusal and the evidence are section 6.6's.
- **Thinking off only** (as decided for MemoryAgentBench; `--kv-session-thinking on` is refused). The generation header after a user turn is `<|turn>model\n<|channel>thought\n<channel|>`, 7 tokens, checked against the pinned template once at startup (`kv_session_protocol.check_gemma4_header`, `gemma4_renderer.Gemma4SourceRenderer.kv_transcripts`).
- **Continuations without a header.** After tool responses Gemma's template continues the model turn and adds no generation header (thinking off), so the client sends `header_tokens` 0. Such a tail ends with `<tool_response|>`, the one token phase 2 computes (e = N − 1); the server checks it. The MemAgent read turns are such turns; a turn after a user message (a format-error nudge, the question) carries the header.
- **Boundary and stops.** The stop ids are 0, 1, 50 and 106. The new messages open at `<|turn>` (`input_ids[boundary_tokens]`, after a text reply's `<turn|>` and `\n`: boundary 2) or right after `<|tool_response>` (`input_ids[boundary_tokens − 1]`): a reply with a tool call stops on `<|tool_response>` (id 50), which is also the first token of the tail that carries its responses, so its boundary is 1. The client restores the previous reply's last sample in front (+1) when it is not the tail's first token (an `<eos>`, pad or length stop, or a tool call that stopped on `<turn|>`), as Qwen's client does for every id but `<|im_end|>`.
- **The client's tail.** Gemma's template renders a reply's tool calls, then the tool responses, then the reply's text, and it drops tool messages that follow a reply without tool calls; Qwen's marker (the newest reply replaced by a text marker) therefore does not find a tool-call reply's tail. The re-implemented client (`test_gemma_kv_session.GemmaKvClient`, the reference of the MemoryAgentBench harness's runner) renders a stand-in instead: the reply's tool calls without text, which the template closes with `<|tool_response>` when nothing follows; the tail goes on from that token. A reply without tool calls takes Qwen's marker.
- **Exactness** (pinned tokenizer and template, `test_gemma_kv_session`): the MemAgent first prompt is 219 tokens (prefix 212 + header 7), and every served prompt equals, token for token, the template's render of its messages with each model turn's empty thought channel kept after `<|turn>model\n` (the header the reply followed; Gemma's template drops it when it re-renders history, 4 tokens per model turn). A stock vLLM server that re-renders the conversation every turn (the Vanilla transport) therefore first differs right after the first `<|turn>model\n`, at prefix + 3; a stock replay of the served prompts (token ids) is the exact control.
- **Memory, 16 sessions at a 32,768-token budget with 4,096-token replies, P = 16** (`--release-idle-pages`, the selective rule): pool 1 holds 18.95 GiB (16 windows of 812.5 MiB, 12.70 GiB, and one 8,192-token forward's transient, 6.25 GiB; `goal_query` 1.56 GiB more for the Goal), pool 0 needs 5,724 global Pages, 7.0 GiB (one selective turn 2 × (2,048 + 769) + 1 = 5,635, the other sessions' tails and the watermark; `goal_query` 5,711): 25.94 GiB, 27.48 GiB with the Goal. 28 GiB holds them (1,690 Pages above the rule, 423 for `goal_query`), 24 GiB does not; nothing has to move the idle sessions' windows to the host (design WP11: 812.5 MiB per window, not built). Idle sessions release their global Pages under pressure and readmit them before their next phase 1; the windows stay in pool 1. A read of two chunks (about 8,300 tokens) runs as a chunked phase 1 from that margin. GPU memory: 57.9 GiB of weights and 28 GiB of KV with the overheads estimated above, about 92,000-93,300 MiB at the peak, 4.5-5.8 GiB below the GH200's 97,871 MiB (2.5-3.7 GiB with `goal_query`'s 2-GiB rewriter); 32 GiB would leave 0.5-1.7 GiB.
- **Context.** The 32 MemAgent32 cases hold 39,336-113,785 Gemma tokens (the first prompt, every chunk in its tool response and the question; 0.97-1.0 of Qwen's count): at 131,072 tokens about 17,300 tokens are left for the 28 reads and the answer of the longest case; 262,144, which the selective rule allows, leaves room for long replies too (the Qwen plans run 262,144). The pinned arena holds every sealed Page of every open session: the 16 longest cases alone hold 1.50 million tokens, 114.5 GiB of global Pages before any reply, more than one share's 80.9 GiB pinned budget, so 16 such sessions need `--host-memory-shares 2` and a two-share job.

```sh
python -B serve.py --release-dir "$RELEASE-gemma4" --protocol kv-session --kv-session-thinking off \
  --policy observation_query --active-token-budget 32768 --page-tokens 16 --kv-cache-gib 28 \
  --max-running-sessions 16 --release-idle-pages --max-tokens 4096 --max-model-len 262144 \
  --dram-arena-gib 150 --host-memory-shares 2 --connector-pool-gib 1 --language-model-only \
  --output "$RUNS/gemma-mab-obs" --port 8000
```

**CPU tests** (`test_gemma_*.py`). With `REMOTE_RA_GEMMA4_METADATA` naming a directory that holds the five pinned metadata files, the rendering tests use the real tokenizer (they need Transformers 5.13 and Tokenizers 0.22.2, as in section 2.1): the transcript reproduces the prompt ids of a stock vLLM 0.24.0 Gemma session on all 10 turns, up to 62,919 tokens. Without the variable those tests are skipped. `test_gemma_e2e` runs the engine and the worker with vLLM's own KV manager (one global group, five sliding groups, a 64-token window) for every policy over three sessions: every forward reads exactly the last 63 tokens in each sliding group, and the gate's figures bound what each turn takes from the sliding pool. `test_gemma_release` serves sixteen sessions together with the global pool at the selective rule's minimum: every turn reads exactly what it reads with a pool that never releases, victims are idle sessions only (least recently used first), readmissions restore exactly the released Pages and their KV, windows and the sliding pool are untouched by releases, and a rebuild that can never fit is refused with `CAPACITY_EXCEEDED`. `test_gemma_goal` covers `goal_query`: the Goal transient's layout, pins and launcher figures, the gate, and sixteen `goal_query` sessions with releases (every turn as with a pool that never releases, each Goal forward in its transient, its query vector the pre-final-norm last row in the Page vectors' projection; without the transient the same pools fail). `test_gemma_chunked` covers chunked phase 1: the arithmetic and the gate's figures, the lease checks, and observations of up to about 2,600 tokens in forwards of 256 (one native forward of 1,024) for `full`, `observation_query` and `goal_query` over three sessions, every turn equal to one forward's and every sliding read exact. `test_gemma_full` covers `full` at sixteen sessions under the release rule (every turn as with a pool that never releases; a readmission brings back every sealed Page with its KV; without the switch the same pool fails). `test_gemma_overlap` covers `--admission-overlap`: the record, the sliding ledger, and sixteen `goal_query` sessions with a rewriter stand-in whose Goals park on their sessions' threads, deciding exactly as the server without the flag, with and without releases. With the real tokenizer, `test_gemma_renderer` checks the Goal's inputs from Gemma's transcript and its Goal probe tokens. `test_gemma_graphs` covers the CUDA-graph pins and checks, the overlay runner's buffer binding on Gemma's six groups and the e2e with graphs attached (only probe passes eager); `test_gemma_fastpath` runs the window e2e with the fast paths off, on and with both (identical RPCs once each batch entry is expanded, evidence files, frames, attended Pages, replies, sliding reads, per-step leases and claims); `test_gemma_window_rows` drives the incremental and the full observation, and the delta rows, over randomized step sequences with corruptions. `test_gemma_flags` runs the options together: 240 launches (`full`, `goal_query` with and without a rewriter, `observation_query` and `random_fill`, each with every combination of `--cuda-graphs`, the first or both decode fast paths, `--release-idle-pages` and `--admission-overlap 4`, at 16 sessions and 131,072 tokens) each add exactly their own startup keys and vLLM arguments, and the startup checks and the window worker accept them all; `--decode-fastpath2` alone, `--decode-fastpath3`, `--goal-rewriter-batch`, `--phase1-in-step`, `--gdn-device-handoff`, the kv-session protocol with thinking on and the chat-text protocol with a `--task` list are refused whatever else is on. On the CPU fakes `goal_query` with `--cuda-graphs full` and both fast paths (three sessions decoding together; only phase 1 and the Goal forward eager), chunked phase 1 with both, `--admission-overlap 8 --release-idle-pages` with both on eight `goal_query` sessions, and the release-rule pressure runs of `full` and `observation_query` with both give the turns of the runs without them, with the same releases and readmissions. The default suite runs these end-to-end tests and `test_gemma_release`'s on six or eight sessions (`goal_query`, `full`, `observation_query`, `random_fill`, the overlap) and chunked phase 1 for `goal_query`; with `REMOTE_RA_SLOW_TESTS=1` they also run sixteen sessions of five turns against a pool that never releases (the 16 of 16 parked turn-1 admissions above) chunked phase 1 for `full` and `observation_query`, and the graph and fast-path runs of every policy and mode, about 6 minutes more. `test_gemma_goal_rewriter` checks that the profile refuses `--goal-rewriter-batch` and that its servers write none of the Goal rewriter's timing fields. `test_gemma_kv_session` covers the kv-session grammar: the re-implemented client, the transcript's geometry and refusals, the record and the header check, the launcher (16 sessions at 262,144 tokens, the refusals, `--host-memory-shares`) and vLLM's startup check with `request_pages`; on the pinned tokenizer every MemAgent turn type (reads, a double read, an invalid call, a format error, the question, restored samples) token-exact against the template. `test_gemma_kv_session_e2e` runs MemAgent sessions of that client through the window layout's engine and worker with `--release-idle-pages` at the selective rule's minimum pool (six sessions of `observation_query` and of `goal_query` in the default suite, sixteen with `REMOTE_RA_SLOW_TESTS=1`, every reply then equal to a pool that never releases, about 1.5 minutes): every prompt is the client's construction, the Gemma geometry holds, the releases and readmissions keep the windows; and three sessions through the packaged route and the frontend capture with a `Gemma4SourceRenderer`. `test_gemma_chat_text` covers the chat-text turns (gemma-tb): the startup record and the template check, Terminus transcripts whose every prompt equals the reference render and, without the thought channels, stock vLLM's prompt (with the pinned tokenizer and template, and on a stand-in), the stage-0 stock session reproduced by the chat-text transcript, and the launcher and startup checks at the Terminal-Bench settings. `test_gemma_chat_text_e2e` serves Gemma chat-text sessions through the packaged router and the chat-text frontend over the window layout's engine and worker fakes with `--release-idle-pages`: sixteen sessions decoding together at K = 20, sixteen on a global pool at the selective rule's minimum (idle sessions released and readmitted), a summary's aside and hand-off rebuild, the reply bound and the context refusal, a cancel and its rebind, closes and evictions that give windows back to pool 1; `REMOTE_RA_SLOW_TESTS=1` adds sixteen sessions of eight episodes with summaries under release pressure, `random_fill` and `full` under pressure, and the comparison with a pool that never releases. `test_gemma_chat_text_replay` (opt-in, `REMOTE_RA_TB_ATIF_DIR`) replays the August Terminus 2 trajectories through Gemma's turns (section 6.7).

```sh
REMOTE_RA_GEMMA4_METADATA=/abs/path/to/gemma-4-31b-it \
  "$VENV/bin/python" -B -m unittest test_gemma_profile test_gemma_renderer test_gemma_kv_pools \
  test_gemma_selection test_gemma_install test_gemma_capacity test_gemma_startup \
  test_gemma_worker test_gemma_engine test_gemma_e2e test_gemma_release test_gemma_goal test_gemma_overlap \
  test_gemma_full test_gemma_chunked test_gemma_goal_rewriter test_gemma_graphs test_gemma_fastpath \
  test_gemma_window_rows test_gemma_flags test_gemma_kv_session test_gemma_kv_session_e2e \
  test_gemma_chat_text test_gemma_chat_text_e2e
```

**Not done yet:** the window record in each `runtime.jsonl` row, windows on the host for the session pool (not needed for MemoryAgentBench at 28 GiB, above), and every GPU validation.

## 5. Page selection for the query policies

`goal_query` and `observation_query` run the Online Selector V2 contract of the dram-backed-kv PoC (`scripts/serve_online_selector_v2.py`, upstream commit `a70701e`). The page size P is a parameter of that contract. At the default P = 784, vLLM restores full-attention KV in 784-token pages, so the PoC's 16-token budgets are converted to whole pages, and each page is represented by its final 16-token sub-page. At P = 16 (`--page-tokens 16`, section 4.6) the pages, budgets and vectors are the PoC's 16-token ones.

Every request after the first of a session runs the PoC's V0 schedule, for every policy:

1. **Phase 1 (observation).** The new observation is prefilled on the pages that stayed resident after the previous turn (its active pages plus the pages sealed while serving it) and on the live tail page, continuing the Gated-DeltaNet state of the previous turn. It yields the vectors of the pages it closes and the observation query: the hidden state at the observation's last token.
2. **Goal.** For `goal_query`, the maintained Goal is built from the previous reply and the latest observation. Its query text runs once as a separate cold forward and is cached by its retrieval-goal hash, as upstream.
3. **Selection** among the pages before the observation (below).
4. **Phase 2 (generation).** The generation header and the decode run on the selected pages plus the observation's KV from phase 1.

Every token is computed exactly once. A page sealed during a turn keeps the vector of that single computation. The first request of a session is one plain prefill.

The selection:

1. Keeps every page active on turn 0 and while the sealed history is at most **8192 tokens** (`reason` `cold_start` / `below_switch_threshold`).
2. Otherwise sets the capacity to `max(1, mandatory pages, floor(0.40 × history tokens) // P)` pages. `--active-token-budget N` replaces the ratio with an absolute budget; `--active-ratio R` changes the ratio.
3. Always keeps the **mandatory** pages: every page touched by the initial system + task prompt.
4. Adds **recent** pages: non-mandatory pages from the two previous turns, newest first, at most ⌈1,024 / P⌉ pages (the PoC's 64 × 16 = 1,024 tokens): 2 pages at P = 784, 64 at P = 16.
5. Fills the rest with the pages whose vectors have the highest cosine to the query, using the upstream `PrefillPageEmbeddingSelectorV2`.

Vectors: the last-token hidden state before the model's final norm (the PoC's last decoder layer), projected with the PoC's seeded 128-d Rademacher matrix (seed 20260904, SHA-256 `20c2e1cf…87aa` for hidden size 5120), L2-normalised and int8-quantised per vector. The Goal query is the PoC's `<retrieval_query>` JSON text of the maintained Goal; `--goal-query-variant candidate_entity_cap8` keeps only the last 8 files, symbols and tests in that text. The maintained Goal uses entity-drift 3 and, with `--goal-rewriter-model`, the PoC's persistent Qwen3-0.6B rewriter. A failed rewrite falls back to deterministic fields for that turn, as upstream, and is recorded; `goals.jsonl` keeps the text of every Goal, Goal query and rewriter exchange (section 8).

`observation_query` does not use the Goal rewriter, because its ranking depends only on the observation vector: the server never starts the rewriter for it, builds its Goal from deterministic fields (evidence only), refuses a startup config that pins one, and records a given `--goal-rewriter-model` as unused in `startup.json` and the dry run (`goal_rewriter`).

**Random-fill control.** `--policy random_fill` is `observation_query` with only step 5 changed: the same threshold, capacity, mandatory and recent pages, the same phase 1 and deterministic Goal (evidence only), no Goal rewriter (a given `--goal-rewriter-model` is recorded as unused, a config that pins one is refused) and no Goal forward. The candidates left after the mandatory and recent pages are put in a random order and the first ones fill the capacity; no query vector, projection or score is used. The order is SplitMix64 with Fisher-Yates (nothing is hashed), keyed by the seed pinned at startup (`--fill-seed`, default 42, the `random_fill` key of the selection contract), the session index and the session's request ordinal. On every active turn `query.selection.fill` records the seed, the session index, the ordinal (`turn`) and the counts of eligible and drawn pages, and `retrieved_pages` lists the drawn pages in draw order. If `goal_query` and `observation_query` do no better than this control, their ranking carries no useful information. Unlike `random50`, it keeps the query policies' mandatory and recent pages and budget.

Known differences from the PoC, and between the two layouts:
- At the default 784-token pages every budget is a whole number of pages and the live tail is one partly filled page. At 16-token pages the budgets, the recent quota and the live tail (under 16 tokens) are the PoC's.
- Mandatory pages cover every page the system + task prompt touches. The PoC keeps only pages lying fully inside it, which at 784-token pages could drop the end of the task; at 16-token pages the two differ by at most one page.
- In the decoupled layout the worker computes the int8 page vectors in one batch on the GPU; at the default layout they are computed per vector on the CPU. The two paths gave identical vectors on the CPU test workload, but on 5,120-wide rows about 0.35 % of vectors differ by ±1 in one element, which can reorder near-ties (`docs/exp-decouple-pages/DESIGN.md`, Integration item 4).
- `random50` keeps its fixed floor(n/2) draw. At 16-token pages it is seeded from a 16-token block hash, so its draws are not comparable with those at 784-token pages.

## 6. Harness interface

The launcher's `--protocol` accepts `mswea-xml` (the default), `kv-session` and `chat-text`. `mswea-xml` reproduces the dram-backed-kv PoC server token for token and is the protocol for the SWE-bench runs (sections 6.1 to 6.5). `kv-session` is the token-level session protocol of the MemoryAgentBench harness (section 6.6). `chat-text` is plain OpenAI chat text for agents such as Terminus 2 on Terminal-Bench 2.0 (section 6.7).

### 6.1 Mini-SWE-Agent XML protocol

The PoC server's contract, used by Mini-SWE-Agent 2.4.2 with single `<mswea_bash_command>` actions:

- **Request.** `POST /v1/chat/completions` with `messages` only: the complete history as `{"role", "content"}` text. It holds the system message, the task message (which must contain the frozen issue), then the agent's assistant and user messages, and it ends with a user message (an observation with `<returncode>`, a format-error prompt, and so on). `tools` is rejected and other message keys are ignored.
- **Sampling.** The server ignores client sampling fields, as the PoC does: greedy decoding, at most the server's reply bound of new tokens (1536, the PoC's; `--max-tokens`, section 4.9), and a stop once the decoded text contains `</mswea_bash_command>` or on EOS/pad. The action marker is detected inside the engine.
- **Rendering.**
  - System and task go through the pinned chat template with `enable_thinking=false`.
  - Every later message is `<|im_start|>{role}\n{content}<|im_end|>\n`.
  - The generation header is `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
  - A previous reply is replaced by the exact token IDs it was generated with, matched after `strip()`, so trimmed echoes are fine.
- **Why thinking off.** Every benchmark of this runtime runs with thinking off, aligned with the PoC: its server (`scripts/serve_online_selector_v2.py` of dram-backed-kv) renders with `enable_thinking=False`, because LiteLLM / Mini-SWE-Agent remove the thinking block from the assistant content before the next request. With thinking on, the next request would no longer extend the KV prefix that the server holds (prefix-unstable, so no KV reuse); with thinking off, the empty think block is part of the prompt that the server renders (the generation header above), and the history stays append-only. A baseline run with thinking on is therefore not comparable with these runs.
- **Response.**
  - `choices[0].message.content` is the decoded text of every generated token, including the marker and anything the final token carries past it.
  - `finish_reason` is always `stop` and there are no tool calls.
  - The boundary `<|im_end|>\n` (`\n` after EOS) enters the server's history but is not returned.
- **Goal-query inputs.** The Goal uses the PR description inside the task message, the previous reply, the latest user message, and its `<returncode>`, as in the PoC.
- **Session.** A request that repeats the previous request, then the reply, then one new user message continues the session (section 5).
  - Any other request starts a fresh cold session, as on the PoC server. Examples are a format-error retry that drops the rejected reply (1 of 5 PoC cohort instances) and compaction.
  - A fresh session re-renders the whole request from text, releases the previous session's pages and resets the selector and Goal state. Its runtime row records `stage_b.rebuild == true`.

### 6.2 Minimal client

`examples/http_client.py` sends one first request (`python -B examples/http_client.py --base-url http://127.0.0.1:8000/v1 --messages examples/messages.json`). It is a smoke client, not an agent. `examples/messages.json` matches `examples/task.json`, so it works against a server started with that task only.

This revision rejects the OpenAI tool-call protocol (`tools`, `tool_calls`, `tool` messages). The previous accepted runtime, commit `4d90616`, serves it and documents it in its README. `examples/http_client.py` still has a `--tools` option, because `package.py` ships that file and a test loads it; do not use it with this runtime, which rejects any request with `tools`.

### 6.3 Errors

Treat any non-2xx status, timeout, disconnect or server exit as the end of that server's run:
1. Preserve the HTTP error body, `failures.jsonl` and the server log.
2. Stop the server with `SIGTERM`.
3. Start a fresh server for any retry.

Do not silently retry a request whose admission is uncertain, do not advance the turn, and do not reuse the session. A contract violation (6.1) is rejected with an HTTP error.

When the engine rejects a request, the HTTP 500 body is empty. The reason is in two places:
- the `chain` and `traceback` fields of the request's row in `failures.jsonl`;
- the server log line `REMOTE_RA_REGISTERED_ADD {"phase": "engine_rejected", ...}`.

A `CAPACITY_WAIT` refusal (section 4.4) writes the same log line, with its traceback, but it is not an error: the frontend retries the turn, and the client sees only the delay.

Unsupported foreign prefix reuse, speculative decoding, async loading, preemption, history mutation and unknown drains are hard failures, never a silent fallback to Full.

### 6.4 Reaching the server from another host

The server listens on `127.0.0.1` only and has no authentication; never expose it beyond loopback. SWE-bench needs Docker, so in the pilots the agents ran on a separate agent host and reached the GPU job through a reverse SSH tunnel (jobs 3299922 and 3300434). Per server, in this order:

1. After READY, the GPU job opens a reverse tunnel whose far end is a **Unix socket** on the agent host:

   ```sh
   ssh -N -R "$AGENT_SOCKET:127.0.0.1:8000" -o ExitOnForwardFailure=yes \
     -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
     -i "$TUNNEL_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes \
     -o UserKnownHostsFile="$KNOWN_HOSTS" "$AGENT_HOST"
   ```

   `AGENT_HOST` is `user@agent-host`, `TUNNEL_KEY` a key that the agent host accepts, `KNOWN_HOSTS` a file that pins its host key, and `AGENT_SOCKET` a new path (one per server, at most 100 bytes) in a `0700` directory there.
2. The job checks the tunnel from the agent host: `curl -sf --unix-socket "$AGENT_SOCKET" http://localhost/v1/models` must list `selective-prefix`.
3. On the agent host, `examples/swebench/relay.py "$AGENT_SOCKET" <port>` listens on `127.0.0.1:<port>` only and relays each connection to the socket; `run_agents.sh` starts it and checks with `ss` that nothing but `127.0.0.1` listens on that port. The agents use `http://127.0.0.1:<port>/v1`.
4. After the agents finish, the job closes the tunnel, then stops the server with `SIGTERM`.

Why a Unix socket: if the agent host's sshd has `GatewayPorts yes`, every TCP remote forward binds on all interfaces whatever address the client asks for, which would expose the unauthenticated server. A Unix-socket forward has no TCP listener, and the relay binds loopback explicitly, so this form stays on loopback whatever the sshd setting. The compute node needs outbound SSH to the agent host; the pilots checked that once per job before the first server.

A simpler TCP form, **not the validated form**: `ssh -N -R 127.0.0.1:<p>:127.0.0.1:8000 <agent-host>`. It stays on loopback only if the agent host's sshd has `GatewayPorts no` (OpenSSH's default). Check with `ss -ltn` on the agent host before any agent runs; `run_agents.sh -` (no socket) performs that check and refuses a non-loopback listener.

### 6.5 Session close (session pool)

On a server started with `--max-running-sessions` (section 4.4), `POST /v1/sessions/close` with the JSON body `{"instance_id": "<startup task id>"}` closes that task's session for good. It is accepted from a loopback peer only (the server binds loopback anyway; through the tunnel of section 6.4 the peer is loopback too), like the chat route it has no HTTP authentication, and the frontend forwards it to the engine as an authenticated frame of the private channel. The response is the engine's close receipt (`"object": "session.close"`, `opened`, `turns`, the released pages and blocks, both pools' free blocks before and after, `open_sessions`); closing a closed task returns the same receipt with `"repeated": true`. Refusals: 403 (not loopback), 404 (not a startup task), 409 (`SESSION_BUSY`: the session has a request queued or in flight; retry after it ends), 400 (no session pool, or a bad body). A task that never opened closes without touching the engine.

### 6.6 MemoryAgentBench kv-session protocol

`--protocol kv-session` serves the token-level session protocol of the MemoryAgentBench harness's KV-session client (method `dram_backed_kv`). The client renders the chat template and parses the replies itself; the server never renders a template. Such a server needs a session pool (`--max-running-sessions K`, section 4.4) and takes no `--task`: its registered tasks are slots, and each client session binds to the next unused slot at its first request (`--kv-session-slots N`, 1 to 64, default 64, so one server serves at most N client sessions over its lifetime). `--kv-session-thinking on|off` (default `on`) pins the generation header the client sends. `/v1/chat/completions` and `/v1/sessions/close` are refused on such a server.

```sh
python -B serve.py --release-dir "$RELEASE" --protocol kv-session --policy observation_query \
  --page-tokens 16 --kv-layout decoupled --active-token-budget 32768 \
  --max-running-sessions 16 --kv-cache-gib 32 --max-tokens 4096 --max-model-len 262144 \
  --dram-arena-gib 88 --release-idle-pages --output "$RUNS/mab-obs" --port 8000 --start
```

- **`GET /v1/kv-session/info`** returns `{"method": "dram_backed_kv", "budget": <tokens or null>, "options": {...}}`: the server's attention budget (its `--active-token-budget`, else `null`) and what it runs (policy, page size, budget mode and selection-contract fields, reply bound, context limit, thinking, sessions).
- **`POST /v1/kv-session/complete`** takes exactly these fields (shortened):

```json
{"method": "dram_backed_kv", "session_id": "<32 lowercase hex>", "input_ids": [248046, 198, 248045, "..."],
 "header_tokens": 5, "prefix_tokens": 423, "boundary_tokens": 2, "budget": 32768, "max_tokens": 4096,
 "temperature": 0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "seed": 42, "messages": ["..."]}
```

  - The first request of a session sends the whole rendered prompt: `boundary_tokens` 0 and `prefix_tokens` + `header_tokens` = the number of `input_ids`.
  - Every later request sends only the new tail: the previous reply's closing delimiter (with the reply's last token in front when it was not EOS), the new messages and the next generation header. `input_ids[0]` must repeat the previous reply's last output id, `prefix_tokens` must not change, and the tail must end with the pinned header. The server's prompt is the session's stream plus `input_ids[1:]`; phase 1 computes from the last reply token to the start of the header (section 5).
  - Sampling: greedy only (`temperature` 0, `top_p` 1, `top_k` −1 or 0, `repetition_penalty` 1); `max_tokens` from 1 to the server's reply bound (`--max-tokens`), clamped to the context left. `budget` and `method` must equal the server's.
  - `messages` (the client's whole history) never makes tokens. Only the messages new since the previous request are used, for the Goal of `goal_query`: their tool and user texts are the observation, and a user message (the question that MemoryAgentBench sends with the last history chunk, or a format-error prompt) becomes the Goal's explicit requirement.
- **Response:** `text` (every generated token decoded, special tokens included), `output_ids` (the last token included), `prompt_tokens` (the prompt's length), `completion_tokens`, `cached_tokens` (`prompt_tokens` minus the number of `input_ids`), `finish_reason` (`stop` or `length`), and, named apart, `kv_slot`, `kv_turn`, `effective_max_tokens`, the waits and the turn's page counts.
- **`POST /v1/kv-session/close`** with `{"session_id": ...}` returns 200 `{"closed": true, ...}`, also for an id the server never saw or already closed (then the first answer with `"repeated": true`); 409 while the session has a request queued or in flight. A closed id is never served again.
- **Errors.** Every failure of a complete request is 400 `{"error": ..., "type": ...}`. A prompt that leaves no room for a reply is refused before any model work with exactly `"Logical RoPE context limit reached"` (`type` `CONTEXT_LIMIT`). Refusals found before admission (the body, the sampling fields, the header, the continuation token, the prefix, the context) leave the session usable; a failure in the engine fails the session, which the client then closes.
- **Evidence.** Every `runtime.jsonl` row adds `kv_session`: the client session id and slot, the token geometry (`header_tokens`, `prefix_tokens`, `boundary_tokens`, `input_tokens`), the client's and the effective `max_tokens`, `finish_reason`, and the Goal's message source and explicit requirement.
- **Gemma 4 (section 4.14).** On a Gemma release the protocol uses Gemma's turn grammar with thinking off: the 7-token header after a user turn, `header_tokens` 0 after tool responses (the tail ends with `<tool_response|>`, which phase 2 computes), the new messages opening at `<|turn>` or after `<|tool_response>`, the stop ids 0, 1, 50 and 106, and the startup record's `"grammar": "gemma4"`.
- **Status.** GPU-validated on Qwen3.6-27B (thinking off, runtime `0d37346`) by MemoryAgentBench jobs 3397844 (G1: one session each of `full` and `observation_query` at 262,144 tokens, every turn token-exact, stock vLLM's eager replay of the `full` prompts token-identical on all 13 turns, the protocol probes passed) and 3397842 (G2: 16 sessions with `--release-idle-pages` at 16 and 24 GiB, section 4.13; the runtime was correct, and the job's one failed check was a bug of the harness's check, corrected and re-verified offline as passing). The real 32-case runs (G4: jobs 3405844 `observation_query`, 3405845 `goal_query`, 3405855 stock, runtime `f9322a6`) served every case without a runtime error; their verdicts and the two corrected harness checks are in [docs/MEMORYAGENTBENCH_BENCHMARK.md](docs/MEMORYAGENTBENCH_BENCHMARK.md); official scoring is pending. The kv-session protocol of the Gemma 4 profile (above, section 4.14) is not GPU-validated. CPU tests (`test_kv_session_protocol.py`, which uses the pinned Qwen3.6 tokenizer when Transformers is installed: a 5-token header with thinking on, and the server's raw token stream equal, token for token, to the template's render with `preserve_thinking`; `test_kv_session_server.py`; `test_kv_session_e2e.py`, MemAgent sessions through the packaged router and the engine fakes).

The configuration of the MemoryAgentBench benchmark run (32 cases, 8 per ability, at most 16 at a time, thinking off, the policies `observation_query` and `goal_query`, and a stock vLLM 0.24.0 control through the harness's Vanilla transport), its validation status and the checklist for aligning baselines with it are in [docs/MEMORYAGENTBENCH_BENCHMARK.md](docs/MEMORYAGENTBENCH_BENCHMARK.md).

### 6.7 Chat text protocol (Terminal-Bench 2.0 with Terminus 2)

`--protocol chat-text` serves plain OpenAI chat text on `/v1/chat/completions` from a server-owned token transcript, for agents that send their whole text history every turn and parse the reply themselves. The target is Harbor 0.21.0's Terminus 2 on Terminal-Bench 2.0, unchanged: JSON actions, no system message, thinking off. Such a server needs a session pool (`--max-running-sessions K`, section 4.4; 1 to 24 on a chat-text server, 1 to 16 on every other) and takes no `--task`: its registered tasks are slots that bind lazily to client sessions (`--chat-text-slots N`, 1 to 1024, default 512). It serves `full`, `random50`, `observation_query` and `random_fill`; `goal_query` is refused, because its Goal is SWE-centric (`<pr_description>`, `<returncode>`). `/v1/sessions/close` and the kv-session routes are refused. A release of the Gemma 4 profile serves the protocol with Gemma's turns (section 4.14): everything below holds for both models except the rendering, the stop ids and the capacity figures.

```sh
python -B serve.py --release-dir "$RELEASE" --protocol chat-text --policy observation_query \
  --page-tokens 16 --kv-layout decoupled --active-token-budget 8192 \
  --max-running-sessions 20 --kv-cache-gib 32 --max-tokens 4096 \
  --connector-pool-gib 1 --dram-arena-gib 64 --output "$RUNS/tb-obs" --port 8000 --start
```

- **Request.** Nonstreaming, one choice, `model` `selective-prefix`. `messages`: an optional system message, then user and assistant text; the first message after the system message is a user message, and so is the last. Content must be a string; `tools`, `tool_calls` and `tool` messages are refused with 400, other message keys are ignored.
- **Sampling.** Greedy; the client's sampling fields are ignored. `max_tokens` (or `max_completion_tokens`) from 1 to the reply bound (`--max-tokens`); without one, the bound clamped to the context left. The reply stops on EOS/pad only (no marker).
- **Rendering.** The reference is the pinned template with `enable_thinking=false` and `preserve_thinking=true`. A fresh or rebuilt session renders the whole conversation through it, then the header `<|im_start|>assistant\n<think>\n\n</think>\n\n`. A continuing request appends the previous reply's exact generated ids and `<|im_end|>\n` (`\n` after EOS/pad), then the new user message as the template renders it, `<|im_start|>user\n{content.strip()}<|im_end|>\n`; startup checks these pieces against the template once. A request continues the session when it repeats the previous request, the previous reply (compared after `strip()`) and one new user message. Anything else, such as Terminus's summary hand-off, an unwound history or a resent request, rebuilds the session (`stage_b.rebuild`). When every reply re-encodes to its own ids, each prompt equals the template's render of its messages.
- **Response.** `choices[0].message.content` is the decoded reply (special tokens skipped), `finish_reason` is `stop` on EOS/pad and `length` at the bound, and `usage` has the prompt, completion and total tokens.
- **Context limit.** A request whose prompt plus requested reply exceeds the context limit (without `max_tokens`: one that leaves no room for one token) is refused with 400 and vLLM's own text, `This model's maximum context length is …`. litellm and Harbor map it to a context-length error. Nothing changes on the server; the session's next request continues normally.
- **Sessions.** A request's key is its initial segment (the system message, if any, and the first user message) and, when the client sends one, its `X-Session-ID` header (litellm also copies it into the body as `session_id`). Terminus 2 sends no header by default, and its first prompt carries the container's host name, so every trial is its own session. Under one header id, a request with another initial segment (Terminus's summary "questions" call, sent without history) is its own session (kind `aside`) and never resets the main one. A key binds to the next unused slot at its first request. A request for a key whose slot has a request queued or in flight waits up to 30 s for it, then gets 409 `SESSION_BUSY`. The key moves to a fresh slot (a rebind: the first request there renders from text, a full prefill) when its slot was failed by a request (a client disconnect or cancel, an engine failure answered with 500), was closed by the server, or reached 256 requests. Slots are never reused; when all are used a new session gets 400 `SESSION_LIMIT`.
- **Closes.** On the default path no close call arrives. Failed, rebound and client-closed slots, and slots at the request bound, close at once. A live session closes only when no spare open session exists, that is when K sessions are open: while a request waits for an open session, the least recently used session idle for `--chat-text-evict-after` seconds (default 30) is closed, asides first, one per waiting request; with nobody waiting, the least recently used session idle for `--chat-text-idle-timeout` seconds (default 600) is closed, one, so that the next new session finds a spare. While a spare exists no live session is closed, however long it has been idle. A client that sends `X-Session-ID` can close: `POST /v1/chat-text/close` with `{"session_id": ...}` (or `{"key_sha256": ...}`, the SHA-256 of the canonical JSON of the initial segment) closes every session of that id. It answers 200, also for an unknown or a closed id (then the first answer with `"repeated": true`), and 409 while one of them has a request in flight; a closed id is refused afterwards. `GET /v1/chat-text/info` returns the options, the counts and the client sessions.
- **Capacity.** At most K sessions are open. For Harbor's `-n 16` the setting is K = 20: four spare sessions take the finished trials (no close call arrives) and Terminus's summary sub-calls, so a running trial in a long command is not evicted. K sizes `max_num_seqs`, the Mamba pool and the pinned page arena (section 4.4). At P = 16 and 32 GiB of KV, K = 20 has 41 Gated-DeltaNet slots, a Mamba pool of 124 blocks (5.93 GiB, 24 blocks or 1.15 GiB more than at K = 16) and 26,699 attention blocks (1,175 fewer). Under the rule of section 4.4 with 4096-token replies, 20 sessions at 131,072 tokens need 24.9 GiB with an 8192-token budget (21.7 at K = 16), 33.1 GiB with a typical last turn (28.2) and 48.9 GiB with the worst last turn (40.7); at 65,536 tokens 20.8, 29.1 and 44.8 GiB. Pinned host memory at 65,536 tokens: 84.1 GiB without an arena (67.5 at K = 16), so `--dram-arena-gib 64` keeps it at 67.9 GiB (64 GiB of captured 1 MiB Pages, for example 20 sessions of 52K tokens); at 131,072 tokens 164.1 GiB, more than the launcher's 128 GiB, so an arena is required. A slot serves 256 requests; one server binds at most N slots (client sessions, asides and rebinds); the lifetime request bound is max(1,024, 256 × N). Idle sessions keep their pages on the GPU between turns, and no close call arrives, so the open sessions' pages can fill the pool (at K = 20, 32 GiB and 65,536 tokens they did in a Terminal-Bench dry run of `full`, section 4.4): run Terminal-Bench servers with `--release-idle-pages` (section 4.13).
- **Evidence.** Every `runtime.jsonl` row adds `chat_text`, and `chat_sessions.jsonl` records every bind, rebind and close (section 8).
- **Status.** On the GPU, `full` passed a 16-task Terminal-Bench 2.0 dry run on one GH200 at runtime `68d19b4` with the settings of [docs/TERMINALBENCH_BENCHMARK.md](docs/TERMINALBENCH_BENCHMARK.md) (K = 20, 32 GiB, `--release-idle-pages`): all 16 trials ended, 333 requests, every request token-exact against its native receipt, no preemption and no capacity stall. `observation_query` (8,192-token budget) passed the same dry run: 384 requests, 18 rebuilds, the selection active on 238 of them, every request token-exact, no preemption and no capacity stall. `random_fill` has not run on Terminal-Bench. CPU tests: `test_chat_text_protocol.py` (with the pinned Qwen3.6 tokenizer when Transformers is installed: JSON replies with escapes, non-ASCII instructions, with and without a system message, all equal to Transformers' template render), `test_chat_text_server.py`, `test_chat_text_e2e.py` (16 sessions decoding together through the packaged router and the engine fakes; cancels, rebinds, idle closes and evictions, the context refusal; `REMOTE_RA_SLOW_TESTS=1` adds 12 sessions × 8 episodes with summaries and asides) and `test_chat_text_replay.py`; `test_integrate_flags.py` runs chat-text sessions with the three decode fast paths and, under pool pressure, with `--release-idle-pages`. The replay test is opt-in (`REMOTE_RA_TB_ATIF_DIR`; the trajectories are not in this repository): the 178 trials of the two August Terminus 2 runs, each replayed twice with the pinned tokenizer (the client's token estimate exact and 20 % low), made 3,363 requests with 83 context refusals, 33 rebuilds and 7 asides, and every served prompt equalled the template's render of its messages. The stock vLLM 0.24.0 control keeps vLLM's default chat template arguments, as in the SWE-bench runs: it re-renders every earlier reply without the empty thought block, so each history reply is 4 tokens shorter there than in this server's stream (`<think>`, `\n\n`, `</think>`, `\n\n`), and it re-encodes reply text where this server keeps the generated ids. Gemma (section 4.14, CPU only): `test_gemma_chat_text.py`, `test_gemma_chat_text_e2e.py` and `test_gemma_chat_text_replay.py`, which replays the same 178 trials twice through Gemma's turns with the pinned Gemma tokenizer and template: 3,359 requests, 126 context refusals, 43 rebuilds and 7 asides, every served prompt equal to the reference render and, without the thought channels, to stock vLLM's prompt (0 differences). The stock Gemma control's prompt is 4 tokens shorter per past reply in the same way (`<|channel>thought\n<channel|>`).

The configuration of the Terminal-Bench 2.0 benchmark run (all 89 tasks, Harbor's Terminus 2 at `-n 16`, the policies `full`, `observation_query` and `random_fill`, and a stock vLLM 0.24.0 control), its validation status and the checklist for aligning baselines with it are in [docs/TERMINALBENCH_BENCHMARK.md](docs/TERMINALBENCH_BENCHMARK.md).

## 7. SWE-bench Verified with Mini-SWE-Agent 2.4.2

The configuration of the SWE-bench Verified benchmark run (the first 32 instances, 16 concurrent agents refilled in list order, the policies `full`, `observation_query`, `goal_query` and `random_fill`, and a stock vLLM 0.24.0 control) and the checklist for aligning baselines with it are in [docs/SWEBENCH_BENCHMARK.md](docs/SWEBENCH_BENCHMARK.md).

This section describes the setup of the multi-session SWE-bench pilots: 3 Astropy instances × `full`/`goal_query`/`observation_query` with 3 sessions per server (job 3299922), and 10 instances × the same 3 policies with 5 sessions per server (job 3300434), on runtime `b1e6988` with 784-token pages. The same 10 instances then ran with 16-token pages × `full`/`observation_query`/`goal_query`/`random_fill`, one job per policy (jobs 3313802, 3314010, 3314319, 3314623, runtime `780fe54`; outcomes in section 4.7), with the agent host unchanged. Every server passed its server-side checks. One run per instance is not benchmark evidence. `random50` was not part of these pilots.

### 7.1 Settings

| Setting | Value |
| --- | --- |
| Agent | Mini-SWE-Agent 2.4.2 (litellm 1.103.1, datasets 5.0.1) |
| Runner | the PoC's `swebench_runner.py` with `--subset verified --split test --workers <sessions> --filter '^(<id>\|<id>…)$' --output <dir>` |
| Config stack | `-c swebench_xml.yaml` (built into Mini-SWE-Agent) `-c swebench_prompt_override.yaml` (the PoC's prompt) `-c model_override.yaml` |
| Model | `examples/swebench/model_override.yaml`: `litellm_textbased`, `openai/selective-prefix`, `api_base` `http://127.0.0.1:<port>/v1`, dummy key, temperature 0.0, `max_tokens` 1536 (`MAX_TOKENS` of `run_agents.sh`; section 4.9), `cost_tracking: ignore_errors` |
| Environment | `docker`, `cwd` `/testbed`, timeout 120 s, `bash -c`, `PAGER`/`MANPAGER` `cat`, `LESS` `-R`, `PIP_PROGRESS_BAR` off, `TQDM_DISABLE` 1 |
| Limits | `step_limit` 90 (`STEP_LIMIT` of `run_agents.sh`; section 4.9), `cost_limit` 0 |
| Process environment | `MSWEA_CTX_PATCH` unset, `AGENTDIET_MODE=off`, `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `MSWEA_GLOBAL_CONFIG_DIR` = an empty directory |
| Servers | one fresh server per policy and group, `--task tasks-g<k>.json`, `--kv-cache-gib 24`, `goal_query` with the 0.6B rewriter; workers = sessions = group size. The 16-token-page pilot added `--page-tokens 16 --kv-layout decoupled` to every server, `full` included, and gave `random_fill` no rewriter option |
| Evaluation | swebench 5.0.2, `SWE-bench/SWE-bench_Verified`, split `test`, `--timeout 1800`, `--max_workers 2` (the pilot script's default) |

`model_override.yaml` with the port filled in is byte-identical to the six files the 10-instance pilot used. The server ignores the client's sampling fields anyway (section 6.1). Mini-SWE-Agent writes `<output>/<instance>/<instance>.traj.json` and `<output>/preds.json`.

**PoC files.** `examples/swebench/upstream/` holds `swebench_runner.py` and `swebench_prompt_override.yaml` from the dram-backed-kv PoC repository (`scripts/` at commit `a70701e`, Git blob bytes with LF; the prompt override equals the copy on the cohort-2 branch `0ab0078`), byte-identical to the copies the pilots ran (`upstream/SOURCE.txt`, `upstream/SHA256SUMS`). `run_agents.sh` uses them by default and refuses files whose SHA-256 differs from these pins:

| File | SHA-256 |
| --- | --- |
| `swebench_runner.py` | `34ca6d9288c72830e61bd00ad8b15790cab4c29515db24b8df30775cfc0880a7` |
| `swebench_prompt_override.yaml` | `1b6a48666ad528775bd11225fa4d5c002721c25c0cdbc05b99cc15b2de31336a` |

Two properties of these files change results, so keep them in mind when you compare with other setups:
- The runner replaces Mini-SWE-Agent's agent class. When the step limit is reached it submits the last `git diff` of at least 3 lines that the agent printed (exit status `LimitsExceeded_InformalSubmission`), and it stops an instance after 10 identical commands in a row (`LoopDetected`, with or without that patch). It also imports an optional `agentdiet_adapter` that is not included; `AGENTDIET_MODE=off` (as in the pilots) leaves it unused.
- The prompt is written for Astropy ("The testbed is an Astropy repository", with Astropy-specific test advice). The pilots ran Astropy instances only; other repositories need an adapted prompt, which is **not part of the validated runs**.

### 7.2 Agent host

The agent host needs Docker, `curl`, `ss`, `python3`, an agent environment and an evaluator environment. The pilots' environments had the versions above; the commands that created them were not recorded, so this is **not part of the validated runs**:

```sh
MINI=/abs/path/to/mini-env
EVAL=/abs/path/to/eval-env
uv venv --python 3.12 "$MINI"
uv pip install --python "$MINI/bin/python" mini-swe-agent==2.4.2 litellm==1.103.1 datasets==5.0.1
uv venv --python 3.12 "$EVAL"
uv pip install --python "$EVAL/bin/python" swebench==5.0.2 datasets==5.0.1
```

The agents ran offline against caches filled beforehand: the Hugging Face dataset `princeton-nlp/SWE-Bench_Verified` (runner and task builder) and one Docker image per instance, `docker.io/swebench/sweb.eval.x86_64.<instance id with "__" replaced by "_1776_", lower case>:latest`. The evaluator loads `SWE-bench/SWE-bench_Verified` (identical statements for all ten pilot instances) and ran without the offline variables. Fill the caches once with network access, for example with `load_dataset(...)` and `docker pull` (not part of the validated runs).

### 7.3 Tasks

Each server's `--task` list must hold exactly the instances its agents run, with `{instance_id, problem_statement}` taken verbatim from SWE-bench Verified. Requests are routed by the problem statement inside the task message, so an agent whose instance is not in the list ends that server's run with an HTTP error (sections 4.4 and 6.3). `examples/swebench/build_tasks.py` builds the files exactly as our harness did: `task-<id>.json` per instance and `tasks-g<k>.json` per server group (the fewest groups of at most `--max-sessions`, sizes differing by at most one). Run it where the dataset is available:

```sh
"$MINI/bin/python" examples/swebench/build_tasks.py --output-dir /abs/path/to/tasks --max-sessions 5 \
  astropy__astropy-12907 astropy__astropy-13236 astropy__astropy-14096
```

Given the same statements, it reproduces the ten pinned pilot `task-*.json` files and the `tasks-g*.json` lists of jobs 3299922 and 3300434 byte for byte, and it refuses a statement that differs from the pilots' SHA-256 pin for those ten instances. Copy the `tasks-g<k>.json` files to the GPU side.

For a session-pool server, `--pool` writes one `tasks-pool.json` with every instance (1 to 64) in the order given instead of groups; `run_agents.sh` refills its agents in that order:

```sh
"$MINI/bin/python" examples/swebench/build_tasks.py --output-dir /abs/path/to/tasks --pool <instance id>...
```

### 7.4 Run one group

Per policy and group: a fresh server with that group's list, the tunnel, then one runner on the agent host (`run_agents.sh <socket> <port> <new output dir> <instance ids of the group>`). With the Slurm script:

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

Notes:
- One server per policy, a fresh one for every group, a fresh output directory and socket each time. The pilots ran every policy over the same groups, one server after another on one GPU.
- `PAGE_TOKENS=16 KV_LAYOUT=decoupled` is the recommended setting for `goal_query`, `observation_query` and `random_fill` (section 4.6); leave both unset for the default 784-token layout. For `random_fill`, set `POLICY=random_fill`; the script passes a rewriter to `goal_query` only.
- Time: in job 3300434 a group of 5 agents took 1,255–2,120 s, and the last of its 6 servers (3 policies × 2 groups) was done 10,851 s after the job started, inside the 10:45:00 allocation. The pilots capped each agent run at 3,300 s (job 3299922) or 5,400 s (job 3300434). In the 16-token-page pilot (one policy and 2 groups per job, agent cap 3,000 s) a group took 1,215–2,045 s and each job 49–68 minutes of its 02:00:00 allocation.
- An SSH drop or timeout does not stop the remote runner. The pilots then killed that runner and removed only the containers it had started (`minisweagent-<8 hex>` names in its log).
- In jobs 3300434 and 3314319 some `goal_query` turns fell back to deterministic Goal fields (finding `GOAL_FALLBACK`; section 5); the servers still passed.
- Session pool (section 4.4): set `TASKS=.../tasks-pool.json` and `MAX_RUNNING=<K>`. The Slurm script passes `--max-running-sessions K` to `serve.py` and `MAX_RUNNING` to `run_agents.sh`, which then runs a pool of K agents, one runner per instance in the list's order, and closes each instance's session when its runner ends (retrying while its last request is still being torn down) before starting the next. It writes `runs/<instance>/`, `logs/<instance>.log`, `closes/<instance>.json`, `timeline.jsonl` (start and end time, runner exit and close status per instance: the wall clock from the first start to the last end) and a merged `preds.json` for `evaluate.sh`. Not part of the validated runs.
- Step limit and reply bound (section 4.9): `STEP_LIMIT=<1-256>` and `MAX_TOKENS=<1536|4096>` set Mini-SWE-Agent's `step_limit` and `max_tokens` in the model override that `run_agents.sh` writes (`render_override.py`); the Slurm script passes both to the agent host and `MAX_TOKENS` (with `MAX_MODEL_LEN` and `DRAM_ARENA_GIB`) to `serve.py`. Unset, the override is byte for byte the pilots' file and `serve.py` gets no new flag. Not part of the validated runs.

### 7.5 Evaluate

After the agents, per policy, on the agent host (all group outputs of that policy in one call):

```sh
EVAL_PYTHON="$EVAL/bin/python" examples/swebench/evaluate.sh my-run-goal_query goal_query \
  /abs/path/to/eval/goal_query /abs/path/to/runs/goal_query-g0 /abs/path/to/runs/goal_query-g1
```

It writes one predictions file with every planned instance (an instance without a prediction gets an empty patch and is reported missing), runs `python -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Verified --split test --predictions_path … --max_workers 2 --timeout 1800 --run_id …` in the new evaluation directory, and prints one line per instance from `logs/evaluation/<run_id>/results.json`. Trajectories, predictions and logs contain prompts and tool output: keep them out of public releases.

## 8. Evidence per server

Each `--output` directory contains:

| File | Content |
| --- | --- |
| `startup.json`, `launch.json`, `chat_template.jinja` | Frozen task/config hashes, exact server command, pinned template |
| `ready.json`, `worker-attached.json` | READY identity (with the selection contract and the pre-norm vector space) and worker attachment receipt |
| `runtime.jsonl` | One row per turn: response `id`, internal request ID, token/order hashes, `candidate_page_count`, `selected_logical_pages`, `prepare`/`restore` H2D bytes, joined purge/drain, latency. `stage_b` holds the turn geometry: the computed start C, the candidate end, the observation span, the phase-1 range, the phase-2 start, `retain`/`restore`/`release`, the computed end C′ = prompt + generated − 1, the sealed end and `rebuild`. Goal/Observation rows add `query.selection` (`reason`, `capacity_pages`, `mandatory_pages`, `recent_pages`, `retrieved_pages`, contract and projection hashes) and `query.goal` (variant, rewriter attempt/success/fallback, cache hit, cold forward); `goal_query` rows add `query.goal_text` (below) |
| `goals.jsonl` | `goal_query` only: per request, the text behind its Goal hashes and the Goal rewriter's input and output (below) |
| `native-<sha256(request id)>.json` | Native plan (`selected`, `retain`/`restore`/`release`, run pages, tail flags, phase-1 lease), worker frames (`prepare`, `phase1`, `phase1_drain`, `reserve`, `allocation_update`, `complete`) and drain receipt |
| `failures.jsonl` | Contract/runtime failures with the exception `chain` and `traceback` (empty on success) |
| `engine-cleanup.json`, `parent-cleanup.json` | Shutdown receipts |

The decoupled layout adds fields; the default layout writes none of them (`docs/exp-decouple-pages/WP4.md`, section 4):
- `ready.json`: `kv_layout` (P, M, K), `num_blocks_by_pool` and `free_blocks_by_pool` (attention pool first, Mamba pool second); `worker-attached.json`: the same as `kv_pools`.
- Every `runtime.jsonl` row and native file: `kv_pools.free_blocks_by_pool` after that request's drain.
- `engine-cleanup.json`: `kv_pools` once every session closed. After a clean run both pools are back at the readiness baseline (every block but each pool's null block free), which the validated runs checked.
- Sealed page records name their tokens by `token_span` [pP, (p+1)P] instead of `token_sha256`, and page vectors use the `compact_int8_v1` form `[[128 int8], scale]`.

A session-pool server (section 4.4) also writes `sessions.jsonl`, one row per session open and close (`schema` `remote-ra-session-event-v1`, the session id, index and task id, `open_sessions`, the pools' free blocks, and for a close its receipt); `ready.json` adds `session_pool` (`max_running_sessions`, `registered`); every `runtime.jsonl` row adds `open_wait_s`; `engine-cleanup.json` lists under `sessions` the sessions still open at shutdown and adds `session_pool` (which sessions the harness closed, which closed without opening, which were open at shutdown and which never opened). A server without the option writes none of these.

A server with `--connector-pool-gib` (section 4.9) records `connector_pool_bytes` in `startup.json` only.

A server with an admission flag (section 4.10) records the pinned `admission` record in `startup.json`, `worker-attached.json` and `ready.json`; with `--gdn-device-handoff` every `runtime.jsonl` row adds `restore.d2d_bytes`. A server without the flags writes none of these. A server with `--goal-rewriter-batch N` records `goal_rewriter_batch` in `startup.json` and `ready.json` and adds `batch_id` and `batch_size` to every `goals.jsonl` exchange.

A server with `--language-model-only` (section 4.12) records `language_model_only` in `startup.json` and its `effective_server_config`, the worker's text-only record in `worker-attached.json`, and both in `ready.json`. A server without the flag writes none of these.

A kv-session server (section 6.6) adds `kv_session` to every `runtime.jsonl` row. A chat-text server (section 6.7) adds `chat_text` to every `runtime.jsonl` row (`client_session`, `kind` session/main/aside, `key_sha256`, `session_id` (the X-Session-ID or null), `slot`, `slot_turn`, `client_turn`, `binding` bind/rebind, `rebind_reason`, `rebuild`, the requested and effective `max_tokens`, `prompt_tokens`, `completion_tokens`, `finish_reason`, `busy_wait_s`) and writes `chat_sessions.jsonl` (`schema` `remote-ra-chat-text-session-v1`: one row per bind, rebind and close, with the slot, the reason, the idle time and the close receipt's turns). A server with `--release-idle-pages` (section 4.13) writes `capacity.jsonl`, adds `idle_release` to every native file and `runtime.jsonl` row and its totals to `engine-cleanup.json`, and records `release_idle_pages` in `startup.json`, `worker-attached.json` and `ready.json`. Other servers write none of these.

**Goal text (`goal_query` only).** `goals.jsonl` (`schema` `remote-ra-goal-text-v1`) has one row per request that reached its native terminal, completed or cancelled, with the text behind the Goal hashes of `runtime.jsonl`:
- `request_id`, `session_id`, `session_index`, `turn`, `local_turn`, `epoch`, `terminal_state`: as in the request's native file.
- `goal`: `text` is the Goal the selector used, the canonical JSON of its selector payload (schema version, root goal, retrieval Goal, policy hint); `sha256` is the row's `query.goal_hash`.
- `query`: `text` is the `<retrieval_query>` text of the Goal query (`variant`, section 5); `sha256` is `query.goal.payload_sha256`.
- `decision`: gate and applied decision, triggers, changed fields, rewriter attempt and success, `fallback_reason`.
- `rewrites`: one entry per call of the Qwen3-0.6B rewriter while the Goal was built: `request` (the exact JSON payload sent), `response` (the raw line the rewriter process answered with; `null` when none arrived) and `error` (`Type: message` of a failed call, equal to `fallback_reason`). A successful line holds the candidate the rewriter parsed from its generation. When the rewriter cannot parse its own generation, its line holds only the error and the token counts: the vendored rewriter script does not return the generated text. A request refused before the call (over its size bounds) has no entry. Each entry also records the call's times, measured with the server's `time.perf_counter`, and the child's own counts:
  - `wait_ms`: from the call until the rewriter took it. The rewriter serves one call at a time under its lock, so this is the time spent behind other sessions' calls.
  - `start_ms`: a start of the rewriter process inside the call; 0 unless the process had exited or an earlier call had timed out (the rewriter then closes it).
  - `call_ms`: from the send to the answer line, or to the failure when none arrived (a timeout runs to the 120-s limit).
  - `input_tokens`, `output_tokens`, `generation_latency_ms`: copied from the answer line (the child's prompt and generated token counts and its `generate` time); `null` when the line has none, for example after a timeout.
  - `batch_id`, `batch_size` (only with `--goal-rewriter-batch N`, section 4.10): the server's batch ordinal and the number of calls sent together. On such a server `wait_ms` is the time queued until the batch formed, `start_ms` a child start and the batch line, `call_ms` the batch's send to its answer, and `generation_latency_ms` the batch's `generate` time; the token counts are the call's own.

The request's native file and its `runtime.jsonl` row name the row as `goal_text` and `query.goal_text`, `{"file": "goals.jsonl", "sha256": …}`, the SHA-256 of the row's bytes without the newline. A verifier checks every `goal_query` row of `runtime.jsonl`: one line of `goals.jsonl` has that SHA-256 and the row's `request_id`; the SHA-256 of `goal.text` (UTF-8) is `query.goal_hash`; the SHA-256 of `query.text` is `query.goal.payload_sha256`. At 16 sessions × 70 turns the file takes about 10–12 MB per server, about 9.5 KB per turn, mostly the root goal repeated in `goal.text` (estimated by replaying the 32 trajectories of job 3363871 with a stand-in rewriter). Recording changes no Goal, selection or rewriter call and checks nothing per request; at startup the server checks that the Goal text hashes to the GoalRecord version hash. The other policies write neither the file nor the reference.

**Goal timing (`goal_query`, sessions decoding together).** When the Goal is built on a Goal thread (section 4.10), the native file records `goal_async_ms` (from parking the admission to resuming it) and `goal_construct_ms` (the construction's wall time). If the construction called the rewriter, the native file adds `goal_rewrite_wait_ms` and `goal_rewrite_call_ms`, the sums of its calls' `wait_ms` and `call_ms`; `goal_construct_ms` covers both. A construction without a call adds neither. A server of another policy writes neither these fields nor the per-call times above, on Qwen or Gemma 4 (`test_gemma_goal_rewriter.py`); a Gemma 4 `goal_query` server with the rewriter (section 4.14) writes them through the same code as a Qwen one, which is not tested separately.

A clean shutdown requires all of the following (`examples/slurm/server_checks.py cleanup` checks the receipts):
- `parent.cleanup_sha256 == SHA256(raw engine-cleanup.json)`;
- `joined == true` in both receipts;
- matching `epoch`/`session_id`;
- `engine.runtime_manifest == ready.runtime_manifest`, `engine.pending_operations == 0`;
- `parent.child_exitcode == 0`, `parent.natural_exit == true`;
- the APIServer process actually exited 0.

Reading the transfer numbers:
- `restore.h2d_bytes` counts every host-to-device copy during the turn's model execution, including the hybrid model's recurrent (Gated-DeltaNet) state, not only attention pages. For Qwen3.6-27B one recurrent state is 153,944,064 bytes in either layout, one 784-token attention page is 51,380,224 bytes and one 16-token attention page 1,048,576 bytes.
- In short histories that fit in GPU KV, no attention page is restored from CPU, so all policies show the same restore bytes.
- `prepare.h2d_bytes` is the phase-1 cost (restoring the recurrent state carried from the previous turn); `restore.h2d_bytes` counts the rest of the turn.
- With `--gdn-device-handoff` (section 4.10) the recurrent state moves on the device: on a continuing turn `restore.h2d_bytes` counts the restored attention pages only, and `restore.d2d_bytes` counts the state copies (one, plus one per Mamba-block crossing); `prepare.h2d_bytes` is still the carry.
- Compare policies only on histories that actually evict pages to CPU, and report these components separately.

These files can contain prompts and tool output. Keep them out of public source releases. Joined transfers establish mechanism behavior, not task quality or speedup. Historical labels inside receipts such as `NOT_SYSTEM_VALID` or `cuda_qualified=false` are deliberately conservative capability strings. Judge a run by its READY, per-turn native evidence, cleanup and task evaluation.

## 9. Release integrity and notices

`SOURCE_SHA256SUMS.txt` and `INSTALL_RELEASE.json` describe the delivered core and its exact mapping to clean upstream. The review-only `patches/v0.24.0.diff` is not regenerated with each runtime (the repository has no generator for it): it lacks several later vLLM edits, including the decoupled layout's, so review against the pinned sources instead. `package.py` builds a deterministic source archive of the core from a fixed file list; the Slurm and SWE-bench examples (and some newer tests) are not in that list, so use a Git checkout or `git archive` for them. `qualification_parent_manifest` in READY points to the historical qualification lineage; it is not a certificate for a new run. The Apache-2.0 [LICENSE](LICENSE) and upstream notices are retained. Model configuration/tokenizer provenance and third-party terms are in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). `examples/swebench/upstream/` contains the PoC's runner and prompt, copied unchanged from the dram-backed-kv repository; they keep that repository's terms.

## History

- **Current runtime: this revision (the current `main`).** It is the runtime below plus three opt-in admission optimizations (section 4.10), the launcher's host-memory budget for one of them, a longer timeout for the release test, a third context choice of 262,144 tokens (section 4.9), `goal_query`'s Goal text evidence (`goals.jsonl`, section 8), an opt-in text-only mode (section 4.12), an opt-in size for the connector's unused CPU pool (section 4.9), three opt-in decode fast paths (section 4.11), the MemoryAgentBench kv-session protocol (section 6.6), the chat-text protocol for Terminal-Bench 2.0 agents (section 6.7), the opt-in release of idle sessions' pages under GPU pressure (section 4.13), the Goal rewriter's per-call times and opt-in batched calls (sections 4.10 and 8) and a stage-1 Gemma 4 31B IT model profile with the five policies, `--admission-overlap K`, CUDA graphs, the first two decode fast paths and the kv-session protocol (section 4.14). Defaults, policies, selection and the default agent protocol are unchanged.
  - `--max-model-len 262144` (was 65536 or 131072) is the model's native context: no YaRN, no new pin. One request's byte and token caps scale with the pinned limit and are set once at startup, only at 262144 (`remote_ra_live_v2/context_limits.py`, `runtime_renderer.py`, `registered_add_protocol.py`, `vllm/v1/engine/core.py`); the launcher's refusal of a too-large `--dram-arena-gib` names the largest arena that fits (`serve.py`). It ran on one GH200 in Slurm job 3360375 (runtime `e3efc22`, before the decode fast path was merged), verified offline token-exact (section 4.9).
  - `--gdn-device-handoff` hands the Gated-DeltaNet state from phase 1 to phase 2 on the device, `--phase1-in-step` runs phase 1 inside a decode step, and `--admission-overlap K` keeps up to K admissions in flight (`remote_ra_live_v2/admission_flags.py`, `gdn_handoff.py`, `phase1_step.py`, `admission_overlap.py`, `capacity.py`, `goal_async.py`, `remote_ra_growing_worker/tensors.py`). With all three off, the startup config, the evidence and the vLLM command gain nothing.
  - The launcher's pinned-host budget counts the second pinned Gated-DeltaNet carry per running session that `--gdn-device-handoff` keeps (`serve.py`, section 4.9).
  - `check_git_release.py` runs the suite with `OMP_NUM_THREADS=1` and allows it 1800 s (was 840 s, then 1200 s until this revision's 1,065 tests took 1,191 s); it is not a runtime file. The default suite runs short variants of the largest CPU end-to-end cases, and `REMOTE_RA_SLOW_TESTS=1` runs every case (the slow pass, section 2.3).
  - `goal_query` servers record the text behind each turn's Goal hashes and the Goal rewriter's input and output in `goals.jsonl`, named from the native file and the `runtime.jsonl` row (section 8; `remote_ra_live_v2/goal_text.py`). Evidence only: Goals, selection and the rewriter are unchanged, and the other policies write the same files as before.
  - Each Goal rewriter call also records how long it waited for the one rewriter process, a restart inside it, how long it took and the child's token counts (`goals.jsonl` `rewrites[]`: `wait_ms`, `start_ms`, `call_ms`, `input_tokens`, `output_tokens`, `generation_latency_ms`), and a Goal built on a Goal thread adds the totals `goal_rewrite_wait_ms` and `goal_rewrite_call_ms` to its native file (section 8; `remote_ra_live_v2/goal_text.py`, `goal_async.py`, `remote_ra_http/evidence.py`). Evidence only: the Goals, the selection and the rewriter calls are unchanged (CPU tests `test_goal_rewriter_timing.py`, `test_goal_text.py`).
  - `--goal-rewriter-batch N` (opt-in, with `--admission-overlap K ≥ N`) sends the Goal rewriter calls that queue together to a child that runs them as one batched `generate` (section 4.10; `remote_ra_live_v2/goal_rewriter_batch.py`, `goal_rewriter_child.py`, `query_selection.py`, `private_endpoint.py`, `goal_text.py`, `serve.py`, `remote_ra_http/config.py`, `remote_ra_http/evidence.py`). The gate, the request, the prompt, the schema, the parsing, the validation and the fallbacks are the upstream ones; batched outputs can differ numerically from unbatched ones. On the GPU in SWE-bench job 3404230 with N = 4 (section 4.10). Without it the launcher, startup, worker and engine outputs are byte-identical (identity probe).
  - `--language-model-only` passes vLLM's own flag: no vision tower and no multimodal warmup, pinned and checked once at startup (`serve.py`, `remote_ra_http/config.py`, `remote_ra_http/evidence.py`, `remote_ra_live_v2/language_model_only.py`, `composition.py`, `worker_composition.py`). Without it the vLLM command, the startup config and the evidence are unchanged. On the GPU only in the Terminal-Bench dry run of `full` (sections 4.12 and 6.7).
  - `--connector-pool-gib N` gives the connector's CPU pool, which this runtime never uses, N GiB instead of 32 and counts N GiB in the launcher's pinned-host budget; pinned and checked once at startup (`serve.py`, `remote_ra_http/config.py`, `remote_ra_live_v2/connector_guard.py`, `composition.py`, `worker_composition.py`). Without it the vLLM command and the startup config are unchanged. On the GPU only in the Terminal-Bench dry run of `full` (sections 4.9 and 6.7).
  - `--decode-fastpath` cuts the per-decode-step host work that grows with the running requests times their Pages (`remote_ra_live_v2/decode_fastpath.py`); it ran on one GH200 in Slurm job 3366128 (runtime `43fd9c3`, section 4.11). `--decode-fastpath2`, which needs it, sends a step's allocation updates as one worker RPC, cuts more per-request host work and runs the after-forward work behind the logits launch (`remote_ra_live_v2/decode_fastpath2.py`, `engine_boundary.py`, `allocation_owner.py`, `worker_composition.py`, `qwen.py`, `generation_guard.py`); not GPU-validated. `--decode-fastpath3`, which needs `--decode-fastpath2`, cuts more of the fixed host work each running request costs in a decode step (`remote_ra_live_v2/decode_fastpath3.py`, `allocation_owner.py`, `qwen.py`, `worker_rows.py`, `engine_boundary.py`); not GPU-validated. Without the flags the startup config, the evidence and the vLLM command are unchanged.
  - `REMOTE_RA_TORCH_PROFILE=<absolute directory>` (a diagnostic, section 4.11) passes vLLM's own `--profiler-config` with the torch profiler and that trace directory, so the server has vLLM's `/start_profile` and `/stop_profile` routes as stock vLLM does (`serve.py`); it is refused together with `REMOTE_RA_STEP_PROFILE=1`. The startup config, its pins and the server's environment are unchanged; unset, the vLLM command is unchanged too (identity probe). CPU only (`test_torch_profile.py`).
  - The admission work, at its own tip (runtime `4eae441`), ran `goal_query` at 5 and 16 sessions on one GH200 (Slurm job 3349504), verified offline token-exact (section 4.10). Job 3349078 (`observation_query`: the flags' parity at one session and their 16-session effect) is verified offline too.
  - The MemoryAgentBench kv-session protocol (`--protocol kv-session`, section 6.6: `remote_ra_live_v2/kv_session_protocol.py`, `remote_ra_http/kv_session_route.py`) and `--release-idle-pages` (section 4.13: `remote_ra_live_v2/idle_release.py`): `--release-idle-pages` released idle sessions' pages on the GPU in the Terminal-Bench dry run of `full` (section 6.7), and the kv-session protocol with releases and readmissions ran in MemoryAgentBench job 3397842 (runtime `0d37346`, section 4.13). With `--protocol mswea-xml` and the option off, the launcher, startup, worker and engine outputs are byte-identical to `ed57675` (identity probe, 549 items).
  - With `--decode-fastpath2`, `--release-idle-pages` and `--connector-pool-gib` off and `--protocol mswea-xml`, the launcher, startup, worker and engine outputs are byte-identical to `4eaf159`, the previous `main` (identity probe: 555 + 219 items, 0 differences).
  - A Gemma 4 31B IT model profile, stage 1 (section 4.14): `install.py --model-profile gemma4-31b-it`, the Gemma launcher path of `serve.py`, the Gemma turn grammar and renderer (`remote_ra_live_v2/mswea_protocol.py`, `gemma4_renderer.py`), the two-pool window layout in vLLM's KV planner (`remote_ra_live_v2/kv_layout.py`, `vllm/v1/core/kv_cache_utils.py`, `vllm/platforms/interface.py`, `vllm/v1/core/sched/scheduler.py`), the 5,376-wide projection pin and the sliding-window and capacity arithmetic (`selection_contract.py`, `model_profile.py`, `sliding_window.py`, `window_capacity.py`), the engine and worker lifecycle of the sliding windows (`remote_ra_native_growing/window_pins.py`, `remote_ra_live_v2/window_phase1.py`, `window_lifecycle.py`, `window_owner.py`, `window_worker.py`), the Gemma startup checks (`remote_ra_http/config.py`), the sliding pool in the admission gate (`capacity.py`), and `--release-idle-pages` with the selective launcher rule for sixteen sessions (`window_worker.py`, `idle_release.py`, `window_capacity.py`, `serve.py`). Qwen releases are unchanged. Not GPU-validated.
  - The Goal rewriter work and the Gemma profile together (`test_gemma_goal_rewriter.py`): the Gemma profile refuses `--goal-rewriter-batch` for every policy (it is not checked with the window layout), in `serve.py` and in the startup check (`remote_ra_http/config.py`); a Gemma server and the policies other than `goal_query` write none of the rewriter's timing fields.
  - The Gemma profile's four policies of the group's runs (section 4.14, CPU only): `goal_query` with the Goal forward's own sliding transient (2,048 tokens, 1,600 MiB of pool 1) and `--admission-overlap K` with the window layout's sliding ledger (`kv_layout.py`, `window_capacity.py`, `window_phase1.py`, `capacity.py`, `idle_release.py`, `admission_overlap.py`, `window_worker.py`, `contract.py`, `remote_ra_http/config.py`, `serve.py`), the release rule for `full` at 16 sessions (`window_capacity.py`, `serve.py`) and phase 1 in forwards of at most 4,096 tokens above one native forward (`window_phase1.py`, `native_probe.py`, `window_owner.py`, `contract.py`, `sliding_window.py`). The Goal's maintenance, rewriter, prompts, gate and content are unchanged.
  - The Gemma profile's serving configuration (section 4.14, CPU only): `--cuda-graphs`, `--decode-fastpath` and `--decode-fastpath2` on the window layout (`cuda_graphs.py`, `window_owner.py`, `window_rows.py`, `window_lifecycle.py`, `window_worker.py`, `engine_boundary.py`, `qwen.py`, `remote_ra_http/config.py`, `serve.py`), with the worker checking vLLM's sliding runner rows on their live window. Qwen releases are unchanged. Not GPU-validated.
  - The chat-text protocol (`--protocol chat-text`, section 6.7: `remote_ra_live_v2/chat_text_protocol.py`, `remote_ra_http/chat_text_route.py`, and its branches in `runtime_renderer.py`, `frontend_registry.py`, `route_runtime.py`, the overlay `api_router.py`, `composition.py`, `controller.py`, `private_endpoint.py`, `capacity.py`, `query_selection.py`, `sessions.py`, `session_authority.py`, `remote_ra_http/config.py`, `evidence.py`, `shutdown.py` and `serve.py`), on the GPU with `full` in a 16-task Terminal-Bench 2.0 dry run (section 6.7), otherwise on the CPU; Qwen only at first (the Gemma profile then served `mswea-xml` and `kv-session`; a later item adds it).
  - The Gemma profile's MemoryAgentBench path (gemma-mab, section 4.14, CPU only): `--protocol kv-session` with Gemma's grammar, thinking off (`remote_ra_live_v2/kv_session_protocol.py`, `gemma4_renderer.py`, `remote_ra_http/kv_session_route.py`, `window_capacity.py`, `remote_ra_http/config.py`, `serve.py`); `--max-model-len 262144` for the profile; `--host-memory-shares 2`; under the selective rule the window layout's `request_pages`, which vLLM's own startup check now asks for one request instead of one request at the context limit (`kv_layout.py`, `vllm/v1/core/kv_cache_utils.py`; it refused 16 Gemma sessions at 131,072 tokens below 32 GiB); and the frontend capture now takes the Gemma profile's source renderer (`frontend_registry.py`: its exact-type check refused `Gemma4SourceRenderer`, so a Gemma server could not have bound its frontend). On a Qwen release the launcher, startup, worker and engine outputs are byte-identical to `227dedb` (identity probes: 555 + 219 mswea-xml items and 482 kv-session and session-pool items, 0 differences).
  - Options that meet for the first time in this revision, CPU only: `test_integrate_flags.py` runs `--decode-fastpath`, `--decode-fastpath2`, `--decode-fastpath3`, `--language-model-only` and `--release-idle-pages` in every combination on the XML, kv-session and chat-text protocols (legacy and decoupled layouts, with `--connector-pool-gib`) through the launcher, the dry run, the startup checks, the worker attach and `ready.json`; at run time, on vLLM-shaped CPU runners, the sixteen-session pressure run with the fast paths and `--release-idle-pages` equals `--release-idle-pages` alone, MemAgent kv-session sessions and sixteen chat-text sessions run with the three fast paths, and chat-text sessions under pool pressure with `--release-idle-pages` and the three fast paths equal a pool that never releases. `test_gemma_flags.py` runs the Gemma options together (section 4.14). Refused whatever else is on: `--decode-fastpath2` without `--decode-fastpath`, `--decode-fastpath3` without `--decode-fastpath2` and `--release-idle-pages` on the legacy layout; for Gemma also `--decode-fastpath3`, `--goal-rewriter-batch`, `--phase1-in-step`, `--gdn-device-handoff` and `--kv-session-thinking on`. `--language-model-only` and Qwen's CUDA graphs are checked at launch and startup only (the CPU fakes do not model them).
  - With `--admission-overlap K` and `--release-idle-pages`, a turn admitted while nothing runs and no admission is in flight now first releases the idle sessions' Pages it needs, as with one admission at a time; before, the ledger gate admitted it unchecked without the release, and its phase 1 could fail with `insufficient native blocks for phase 1` (`remote_ra_live_v2/idle_release.py`, both profiles; found by `test_gemma_flags.py`, the order checked by `test_idle_release.py`; on Qwen, `test_integrate_flags.py` runs eight `goal_query` sessions with `--admission-overlap 4` and `--release-idle-pages` under pool pressure, with and without the three decode fast paths, and such a lone turn, every turn equal to the one-admission server's with a pool that never releases).
  - With nothing running, the capacity gate now also checks a turn when other open sessions pin blocks, and a turn they leave no room for waits (`CAPACITY_WAIT`) until a session closes or `--release-idle-pages` releases idle pages; before, it was admitted unchecked, and on the GPU its decode took the last free attention block and vLLM preempted it, which killed the engine (Slurm job 3398129: Terminal-Bench 2.0 dry run, Qwen3.6 `full` over chat-text, 16 Terminus 2 trials, K = 20, 32 GiB, without `--release-idle-pages`; the idle pins held all but 69 of 26,698 blocks, the turn needed up to 272). The same rule holds with `--admission-overlap K`; the Gemma window layout kept its rule until integrate-1018 (below; `remote_ra_live_v2/capacity.py`, sections 4.4 and 4.13; `test_adm_overlap.py` for both gates, `test_chat_text_e2e.py` for the chat-text turn that preempted before and now waits for a session close, `test_idle_release.py` for the pressure runs without the option, which now stall instead of failing; `test_ms16.py`'s tight pool is 3,200 blocks, since at 3,000 a lone turn fitted only by the unchecked admission).
  - On a Qwen release with `--decode-fastpath3` and `--goal-rewriter-batch` off and `--protocol mswea-xml` or `kv-session`, the launcher, startup, worker and engine outputs are byte-identical to `227dedb`, the previous `main` (identity probes: 555 + 219 mswea-xml items and 482 kv-session and session-pool items, 0 differences); `goal_query` with the Goal rewriter adds only the per-call times and counts in `goals.jsonl` (rewriter probe: 52 items, the two differences are those new keys).
  - The chat-text protocol on the Gemma 4 profile (sections 4.14 and 6.7, CPU only): Gemma's turns in `remote_ra_live_v2/chat_text_protocol.py` (`GEMMA4_CHAT`; Qwen's turns are the previous code as `QWEN_CHAT`), the initial segment and the startup template check in `gemma4_renderer.py`, a Gemma startup config pinning `chat_text_v1` with the profile's record (`remote_ra_http/config.py`, `serve.py`; up to 24 open sessions). Each prompt is the template's render with the empty thought channel kept after every past reply and, without those channels, stock vLLM's prompt; the stage-0 stock session (job 3364469) is reproduced on all 10 turns. A fix for every Gemma protocol: a rebuild's sliding-window need credits the window its render frees (`remote_ra_live_v2/window_capacity.py`; before, with every window slot taken, a rebuild could be admitted without the idle sessions' release its global shortfall needed and wait forever; found by `test_gemma_chat_text_e2e`; it and `test_gemma_release` (the XML protocol) now check it, the latter failing without the fix). On a Qwen release the launcher, startup, worker and engine outputs are byte-identical to `227dedb` (identity probes: 555 + 219 mswea-xml items and 482 kv-session and session-pool items, 0 differences), and the Qwen chat-text launcher, startup config, frontend, worker record, template check and transcripts are byte-identical to `0d37346` (chat-text probe: 977 items, 0 differences).
  - integrate-1018 merges `terminal-bench` (the capacity gate's check of a lone turn under other sessions' pins of `68d19b4`, above, and the Terminal-Bench 2.0 benchmark configuration, [docs/TERMINALBENCH_BENCHMARK.md](docs/TERMINALBENCH_BENCHMARK.md)), `gemma-window-fix` (a rebuild's window credit, above), `gemma-mab` (the Gemma kv-session protocol for MemoryAgentBench, above), `gemma-tb` (the Gemma chat-text protocol for Terminal-Bench, above) and `torch-profile` (`REMOTE_RA_TORCH_PROFILE`, above). The Gemma profile now serves `mswea-xml`, `kv-session` (thinking off) and `chat-text`; under the selective rule its chat-text layout pins `request_pages` too (`serve.py`, `remote_ra_http/config.py`, `gemma4_renderer.py`). Two fixes: the Gemma window layout's gate checks a turn while nothing runs or is in flight when another open session pins global Pages, a tail or a window, and the turn waits (`CAPACITY_WAIT`) when it does not fit after the releases `--release-idle-pages` makes, as `68d19b4` made the Qwen gate do; before, it was admitted unchecked and could run out of blocks (a fatal preemption) or wait in vLLM forever (`remote_ra_live_v2/capacity.py`, `window_capacity.py`, section 4.14; `test_adm_overlap.py` for both gates, `test_gemma_chat_text_e2e.py` for a lone turn that preempted before and now waits for a session close, `test_gemma_goal.py`'s run without the Goal transient, which now waits at the gate instead of failing in phase 1). And the chat-text and kv-session CPU end-to-end harnesses wait for a request as long as the engine makes progress instead of a fixed 20-s or 30-s bound that failed under load, and `test_integrate_flags.py` keys its chat-text sessions by initial segment instead of bind order (tests only). On a Qwen release the launcher, startup, worker and engine outputs are byte-identical to `0d37346` (identity probes: 555 + 219 mswea-xml items, 482 kv-session and session-pool items and 977 chat-text items, 0 differences; the probes do not exercise the gate under other sessions' pins). Not GPU-validated: the Gemma runtime features beyond the two-pool layout of job 3397695 (selective runtime off, section 4.14), `--decode-fastpath3`, `--goal-rewriter-batch` and `REMOTE_RA_TORCH_PROFILE`.
  - It has not had its own clean-install acceptance.
- **Previous runtime: commit `710bfc6718cf36d22f039151fb9c8d15ca2ab0f6` (2026-10-09).** It is the runtime below plus larger servers and new launcher options, each opt-in; the default server is unchanged (sections 4.8 and 4.9).
  - Up to 16 sessions per server (was 8; `MAX_SESSIONS` in `remote_ra_live_v2/sessions.py`) and the session pool: `--max-running-sessions K` registers up to 64 tasks with at most K open, each opened at its first request and closed by the harness (`POST /v1/sessions/close`; sections 4.4 and 6.5).
  - `--cuda-graphs {off,piecewise,full,full-decode}` (section 4.8).
  - `--max-tokens {1536,4096}`, `--max-model-len {65536,131072}` and `--dram-arena-gib`, the lifetime request bound max(1,024, 256 × T) and the launcher's pinned-host budget (section 4.9).
  - Less admission overhead: phase-1 records copied without `deepcopy`, Selector V2 scoring and the random fill vectorized with bit-identical results (`remote_ra_live_v2/fast_selection.py`), and the prompt hashed once per binding (`remote_ra_native_growing/contract.py`).
  - On the reused clean environment of job 3270721 and one GH200, on runtimes of this line (one run each): CUDA graphs in Slurm jobs 3338893 and 3339448 (runtime `f19f464`), 16 sessions in job 3339019 (`afffd2b`), and 16 sessions with `--cuda-graphs full` and the session pool with `full-decode` in job 3343069 (`613a7d5`); sections 4.4 and 4.8. On one GH200, the 131072-token context, 4096-token replies and the arena cap in job 3348705 (`710bfc6`, verified offline token-exact; section 4.9).
  - It has not had its own clean-install acceptance.
- **Earlier runtime: commit `780fe54df8124d75ea76ad752bb6692427a89ed5` (2026-10-05).** It is the runtime below plus the `random_fill` control of the query policies (section 5).
  - `random_fill` is `observation_query` with only the fill step changed: the remaining capacity is filled in a SplitMix64 / Fisher-Yates order keyed by the startup seed (`--fill-seed`, default 42), the session index and the session's request ordinal, instead of by cosine. Only its selection contract carries the `random_fill` key (`remote_ra_live_v2/selection_contract.py`, `remote_ra_live_v2/query_selection.py`, `serve.py`).
  - The four other policies send and write the same bytes as on the runtime below (`test_random_fill_legacy.py`).
  - At 16-token pages, on the reused clean environment of job 3270721, it served the 10-instance SWE-bench pilot with Mini-SWE-Agent 2.4.2, one job per policy: `full`, `observation_query`, `goal_query` and `random_fill` (Slurm jobs 3313802, 3314010, 3314319, 3314623). Every server passed. One run per policy is not benchmark evidence (section 4.7).
  - It has not had its own clean-install acceptance.
- **Earlier runtime: commit `8f1af6a15356a0b83c5e90fa4258e46bea0761e5` (2026-10-04).** It is the runtime below plus the opt-in decoupled KV layout (section 4.6). The default layout, policies, selection and the agent protocol are unchanged.
  - `--page-tokens {784,16,32,64}` and `--kv-layout decoupled` give the full-attention pages and the Gated-DeltaNet states separate block pools (`vllm/platforms/interface.py`, `vllm/v1/core/kv_cache_utils.py`, `kv_cache_coordinator.py`, `kv_cache_manager.py`, `sched/scheduler.py`, `vllm/v1/kv_cache_interface.py`, the offloading connector files, `remote_ra_live_v2/kv_layout.py`). The engine side owns blocks and checks capacity per pool.
  - In that layout the worker compacts page vectors on the GPU, captures and restores attention pages in batches through a pinned arena, and names sealed pages by token span instead of a per-page digest (`remote_ra_live_v2/page_compaction.py`, `remote_ra_growing_worker/tensors.py`, `remote_ra_live_v2/allocation_owner.py`).
  - Without the flags, the vLLM command, startup config, pins and evidence are those of the runtime below (CPU tests against its sources).
  - On the reused clean environment of job 3270721 it passed three GPU smokes: `full` with identical tokens in the default layout, decoupled at 784 and decoupled at 16 (Slurm job 3308015), the four policies at 16-token pages (job 3308414), and 4 sessions at 24 GiB (job 3308482). One run each.
  - It has not had its own clean-install acceptance.
- **Earlier runtime: commit `902690ae2d8e9971bf4eb59ac9d38dc5c613dec0` (2026-10-03).** It is the runtime below plus two changes. Policies, selection and the agent protocol are unchanged.
  - An admission parked at its Goal waits up to 300 s for its acknowledgement (was 60 s), and the request lease is 360 s (was 120 s) (`registered_add_bridge.py`, `registered_add_protocol.py`, `remote_ra_live_v2/lifecycle.py`). Timeouts do not enter the evidence.
  - Nothing on the decode path hashes per step or per request any more: the binding identity is a minted token, render and authenticate no longer digest the prompt, and the worker no longer hashes or signs a row per decode step; only Page-closing hidden rows are copied to the host (`remote_ra_native_growing/contract.py`, `remote_ra_live_v2/qwen.py`, `remote_ra_live_v2/worker_composition.py`). Generated tokens are unchanged.
  - On the reused clean environment of job 3270721, the 4-session synthetic run passed (Slurm job 3300301). Full's decode step was 69.1, 72.0, 72.0 and 74.6 ms at 1–4 concurrent requests, 4.3–7.7 ms above stock vLLM's step measured the same way (job 3291778); one run each.
  - The 300-s admission wait, on the runtime below (commit `b1e6988`), ran the multi-session SWE-bench pilots with Mini-SWE-Agent 2.4.2: 3 instances × 3 policies with 3 sessions per server (Slurm job 3299922), and 10 instances × 3 policies with 5 sessions per server (job 3300434). Every server passed. One run per instance is not benchmark evidence.
  - It has not had its own clean-install acceptance.
- **Earlier runtime: commit `40e8b3382f74f255c12b78181a6bb82c03df75c6` (2026-10-03).** It is the runtime below plus multi-session serving (section 4.4). Policies, selection and the agent protocol of each session are unchanged.
  - One server serves up to 8 sessions, one per startup task (`--task` takes a list). Each request is routed to its session by its task message. Admission runs one request at a time in arrival order; decode is batched across sessions (`max_num_seqs` = sessions). Nothing is preempted: a turn that the GPU KV pool cannot hold waits (`CAPACITY_WAIT`) until running requests free blocks.
  - `--kv-cache-gib {8,16,24,32}` sets the GPU KV pool (default 8, the qualified profile).
  - `observation_query` never starts the Goal rewriter (section 5).
  - `goal_query` builds its Goal on a background thread, so the other sessions keep decoding during a rewrite (`remote_ra_live_v2/goal_async.py`).
  - A request's binding identity is hashed once, not at every scheduler step (`remote_ra_native_growing/contract.py`).
  - On the reused clean environment of job 3270721, synthetic runs with 4 sessions × 6 turns of 6,500-token observations at 24 GiB passed (Slurm jobs 3292739, 3295203, 3299809). In job 3299809 the steady per-token latency with four requests decoding was 82, 82 and 71 ms (full, observation_query, goal_query), against 117, 111 and 102 ms in job 3292739. A stock vLLM 0.24.0 run of the same model takes 67 ms per step while four requests decode (job 3291778, measured the same way; its whole-run mean of 85 ms also counts decode steps stretched by other requests' prefills), so this runtime is still 4–15 ms per step slower at four requests.
  - It has not had its own clean-install acceptance.
- **Earlier runtime: commit `c1ad7fdc8209a370ef229defe44d4832cd40ad85` (2026-10-02).** It is the runtime below with the per-request overhead removed. Policies, selection and the agent protocol are unchanged.
  - File and tokenizer identity are verified once at startup, not on every request (`tokenizer_semantics.py`, `remote_ra_linux/live_identity.py`, `runtime_renderer.py`).
  - The page catalog is no longer re-projected and re-hashed every turn (`remote_ra_live_v2/query_selection.py`).
  - Captured CPU copies of KV pages and Gated-DeltaNet state are tracked by identity token instead of by hashing their bytes (`remote_ra_growing_worker/tensors.py`).
  - On the reused clean environment of job 3270721, the four-policy, ten-turn synthetic run passed (Slurm job 3290762) and was verified token-exact offline:
    - Full's admission at turns 3-9 fell from about 4.0 s (job 3284520) to 1.2 s;
    - Goal-query's extra admission per turn over Full fell from 1.6-1.8 s to 0.11 s.
  - It has not had its own clean-install acceptance.
- **Older runtime: commit `9430c695d60bd90ff4bc588f6316e6613681d3e2` (2026-10-01).** It is the accepted runtime below plus one fix.
  - The fix: the engine's 120-second admission lease also cancelled requests that were still generating, and the frontend never received an output. A reply longer than about 1,160 tokens at ~100 ms per token could hit this; a measurement run did (Slurm job 3276233). Running requests now end only by completion, client cancellation or disconnect (`registered_add_protocol.py`, `remote_ra_live_v2/lifecycle.py`, `test_lease.py`).
  - This runtime has not had its own clean-install acceptance. On the reused clean environment of job 3270721 it passed:
    - a four-policy, ten-turn synthetic run up to about 60K prompt tokens (job 3284520);
    - one real Mini-SWE-Agent 2.4.2 session per policy on one SWE-bench Verified instance (job 3284734), where a 125.6-second request completed.
- **Clean-install acceptance passed (2026-09-30, Slurm job 3270721).** The runtime bytes of commit `a564df36ffcc4194cedc0cd2db66bcfb641a7813` passed on one NVIDIA GH200:
  - installed privately over a fresh official vLLM 0.24.0 aarch64 wheel in a new environment (section 2.1);
  - four fresh servers, one per policy, each served six growing Mini-SWE-Agent XML turns up to about 33K prompt tokens, with native receipts, joined drains and natural shutdown;
  - every token was computed exactly once, and the query policies selected pages on three of the six turns;
  - the source environment was byte-identical before and after.

  This runtime:
  - runs every turn on the dram-backed-kv PoC's V0 schedule (section 5). The observation is prefilled on the pages that stayed resident, then the selector picks the pages for generation, and the Gated-DeltaNet state stays continuous;
  - follows the PoC's Online Selector V2 contract for the query policies;
  - serves the PoC's Mini-SWE-Agent XML protocol (section 6);
  - uses a 64k context profile with 8 GiB of GPU KV (section 1).

  The tested backend profile is in [Runtime environment](#3-runtime-environment). Documentation-only commits after it did not change runtime bytes, up to the fix above; compare `SOURCE_SHA256SUMS.txt`. A GPU dev smoke of the same runtime bytes on a reused environment passed first (Slurm job 3265689).
- **Previous accepted runtime, for the OpenAI tool-call protocol:** commit `4d90616996858d614b3dbfcfcbe8664f7aa0fe7c`, clean-install acceptance on 2026-09-29 (Slurm job 3261646). It served three growing structured tool-history turns per policy. This runtime no longer serves that protocol.
- An earlier runtime revision, before the structured tool-call action extraction in `remote_ra_live_v2/query_selection.py`, passed the same four-policy qualification on a managed installation (Slurm job 3253330).
- **Not established:** SWE-bench task success, task accuracy, speedup or transfer savings, and support outside the tested boundary (section 1). Synthetic HTTP acceptance shows that the mechanism runs; it is not benchmark evidence.
