# Selective-prefix policies for vLLM 0.24.0

English | [简体中文](README.zh-CN.md)

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

- **Commit `780fe54df8124d75ea76ad752bb6692427a89ed5` (2026-10-05):** five policies, the Mini-SWE-Agent XML protocol (section 6), up to 8 sessions per server (section 4.4). It is `902690a` plus the decoupled KV layout (commit `8f1af6a`, section 4.6) and the `random_fill` control.
- **Default configuration unchanged:** 784-token pages in one shared KV pool, as in `902690a`. The decoupled layout with 16-token pages is opt-in. **Recommended for `goal_query`, `observation_query` and `random_fill`: `--page-tokens 16 --kv-layout decoupled`** (section 4.6).
- On the reused clean environment of Slurm job 3270721, the decoupled layout passed three GPU smokes (jobs 3308015, 3308414, 3308482), and `780fe54` with 16-token pages served a 10-instance SWE-bench pilot with Mini-SWE-Agent 2.4.2, one job per policy (jobs 3313802, 3314010, 3314319, 3314623); section 4.7. It has not had its own clean-install acceptance; the last one passed on 2026-09-30 (job 3270721, runtime `a564df3`).
- **Not established:** SWE-bench task success, task accuracy, speedup or transfer savings, and support outside the tested boundary (section 1). One run per instance is not benchmark evidence.
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
- Synchronous, text-only, UniProc executor, TP=PP=DP=1, **one session per startup task, up to 8 per server** (section 4.4). The accepted run used one session per server.
- The launcher pins the qualified server configuration:

  | Setting | Value |
  | --- | --- |
  | `max_model_len` / `max_num_batched_tokens` | 65536 / 16384 |
  | page size and KV layout | 784 tokens in one shared pool (default); `--page-tokens {784,16,32,64}` with `--kv-layout decoupled` for separate attention and Gated-DeltaNet pools (section 4.6). Validated on the GPU: 784 (default) and 16 decoupled; 32 and 64 never ran on a GPU |
  | GPU KV / CPU offload | 8 GiB (`--kv-cache-gib` 16, 24 or 32 for several sessions) / 32 GiB |
  | execution | eager (no CUDA graphs / torch.compile) |
  | Mamba cache | `align` |
  | GDN prefill | Triton |
  | parsers | `qwen3_xml` tool-call, `qwen3` reasoning |
  | served model name | `selective-prefix` |

  **Prompt plus completion must fit in 65536 tokens.** The accepted run (job 3270721) used this profile with prompts up to about 33K tokens. 8 GiB of GPU KV holds the resident working set plus one turn's new tokens. At 64k tokens `full` keeps about 84 attention pages of 51.4 MB (4.3 GB), next to a few 154 MB Gated-DeltaNet state blocks. With 16-token pages the same 64K tokens are 4,096 attention pages of 1 MiB (4 GiB).

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

The suite needs `torch` (CPU is fine; several test modules import it, others skip their tensor tests without it) and `git` on `PATH`. The base environment of section 2.1 has both; job 3270721 ran the public tests there. On a Windows desktop with CPU Torch 2.9.1 and Python 3.12.7, this revision's 512 tests took about 9 minutes. One of them repeats the whole suite on a Git snapshot of the candidate files.

```sh
"$VENV/bin/python" -B -m unittest discover -s . -p 'test_*.py'
"$VENV/bin/python" -B check_git_release.py --ref HEAD
```

`check_git_release.py --ref HEAD` runs the suite on the `git archive` bytes of that commit; it must pass before publishing a commit.

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
- Memory: the model (51.1 GiB) plus 8 GiB KV and the optional 0.6B Goal rewriter (about 2 GiB) fit one GH200 (94.5 GiB free at start); so did 24 GiB KV with five sessions and the rewriter (job 3300434), at 784-token and at 16-token pages (job 3314319). Allow about 32 GiB of host RAM for the CPU KV pool in addition to model loading; the decoupled layout adds a pinned page arena of up to about 4 GiB per session at 16-token pages (section 4.6). The validated jobs requested 200 GiB.

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

`--task` takes a JSON list of 1 to 8 `{instance_id, problem_statement}` objects, one session each (`max_num_seqs` = sessions). Instance ids must be distinct and no problem statement may contain another. Each harness runs one of these tasks against the same endpoint; the server routes every request to the one session whose problem statement its task message (`messages[1]`) contains, and each session behaves as a one-task server (sections 5 and 6). A request that matches no task, or several, is rejected. Admission runs one request at a time in arrival order, then decode is batched across sessions. The sessions share one GPU KV pool and nothing is preempted: a turn whose worst-case peak does not fit waits (`CAPACITY_WAIT`) until running requests free blocks.

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

The pilots capped sessions so that the pool can never run out even when every session sits at the 65,536-token limit: N − 1 idle sessions at I = 85 blocks plus one turn's peak P = 91 blocks plus a margin of 3 must fit the usable blocks U, so N = 1 + ⌊(U − 91 − 3) / 85⌋ (the `full` footprint at 64K from our capacity analysis, used for every policy because a session rebuild re-prefills every page). That gives 1 session at 8 GiB (U = 166) and 5 at 24 GiB (U = 500). The same rule gives 3 at 16 GiB and 7 at 32 GiB; those sizes are **not part of the validated runs**.

The 16-token-page pilot applied the same rule to the attention pool: an idle session at the limit holds I = ⌈65,536 / 16⌉ + 1 = 4,097 blocks, one turn's peak is I + ⌈16,384 / 16⌉ = 5,121, and the margin is one 1,536-token reply (97 blocks) plus vLLM's 1 % watermark (229). Five sessions at 24 GiB need 4 × 4,097 + 5,121 + 326 = 21,835 of the 22,911 usable blocks; six do not fit. The Mamba pool holds 2 slots per session plus a spare (section 4.6). The rule again gives 1 session at 8 GiB, 3 at 16 GiB and 7 at 32 GiB (**not part of the validated runs**).

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

**Host memory.** In the decoupled layout the worker keeps the attention pages it copies to the host in a pinned page arena. Its capacity is S × (⌈65,536 / P⌉ + 1) pages of P × 65,536 bytes, allocated on demand in 256 MiB chunks: up to about 4 GiB per session at P = 16 (`arena_pages` in `remote_ra_live_v2/worker_composition.py`, `PageArena` in `remote_ra_growing_worker/tensors.py`). This comes on top of the 32 GiB CPU pool of section 3.

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

Vectors: the last-token hidden state before the model's final norm (the PoC's last decoder layer), projected with the PoC's seeded 128-d Rademacher matrix (seed 20260904, SHA-256 `20c2e1cf…87aa` for hidden size 5120), L2-normalised and int8-quantised per vector. The Goal query is the PoC's `<retrieval_query>` JSON text of the maintained Goal; `--goal-query-variant candidate_entity_cap8` keeps only the last 8 files, symbols and tests in that text. The maintained Goal uses entity-drift 3 and, with `--goal-rewriter-model`, the PoC's persistent Qwen3-0.6B rewriter. A failed rewrite falls back to deterministic fields for that turn, as upstream, and is recorded.

`observation_query` does not use the Goal rewriter, because its ranking depends only on the observation vector: the server never starts the rewriter for it, builds its Goal from deterministic fields (evidence only), refuses a startup config that pins one, and records a given `--goal-rewriter-model` as unused in `startup.json` and the dry run (`goal_rewriter`).

**Random-fill control.** `--policy random_fill` is `observation_query` with only step 5 changed: the same threshold, capacity, mandatory and recent pages, the same phase 1 and deterministic Goal (evidence only), no Goal rewriter (a given `--goal-rewriter-model` is recorded as unused, a config that pins one is refused) and no Goal forward. The candidates left after the mandatory and recent pages are put in a random order and the first ones fill the capacity; no query vector, projection or score is used. The order is SplitMix64 with Fisher-Yates (nothing is hashed), keyed by the seed pinned at startup (`--fill-seed`, default 42, the `random_fill` key of the selection contract), the session index and the session's request ordinal. On every active turn `query.selection.fill` records the seed, the session index, the ordinal (`turn`) and the counts of eligible and drawn pages, and `retrieved_pages` lists the drawn pages in draw order. If `goal_query` and `observation_query` do no better than this control, their ranking carries no useful information. Unlike `random50`, it keeps the query policies' mandatory and recent pages and budget.

Known differences from the PoC, and between the two layouts:
- At the default 784-token pages every budget is a whole number of pages and the live tail is one partly filled page. At 16-token pages the budgets, the recent quota and the live tail (under 16 tokens) are the PoC's.
- Mandatory pages cover every page the system + task prompt touches. The PoC keeps only pages lying fully inside it, which at 784-token pages could drop the end of the task; at 16-token pages the two differ by at most one page.
- In the decoupled layout the worker computes the int8 page vectors in one batch on the GPU; at the default layout they are computed per vector on the CPU. The two paths gave identical vectors on the CPU test workload, but on 5,120-wide rows about 0.35 % of vectors differ by ±1 in one element, which can reorder near-ties (`docs/exp-decouple-pages/DESIGN.md`, Integration item 4).
- `random50` keeps its fixed floor(n/2) draw. At 16-token pages it is seeded from a 16-token block hash, so its draws are not comparable with those at 784-token pages.

## 6. Harness interface

The launcher's `--protocol` accepts `mswea-xml` only (the default). It reproduces the dram-backed-kv PoC server token for token and is the protocol for the SWE-bench runs.

### 6.1 Mini-SWE-Agent XML protocol

The PoC server's contract, used by Mini-SWE-Agent 2.4.2 with single `<mswea_bash_command>` actions:

- **Request.** `POST /v1/chat/completions` with `messages` only: the complete history as `{"role", "content"}` text. It holds the system message, the task message (which must contain the frozen issue), then the agent's assistant and user messages, and it ends with a user message (an observation with `<returncode>`, a format-error prompt, and so on). `tools` is rejected and other message keys are ignored.
- **Sampling.** The server ignores client sampling fields, as the PoC does: greedy decoding, 1536 new tokens, and a stop once the decoded text contains `</mswea_bash_command>` or on EOS/pad. The action marker is detected inside the engine.
- **Rendering.**
  - System and task go through the pinned chat template with `enable_thinking=false`.
  - Every later message is `<|im_start|>{role}\n{content}<|im_end|>\n`.
  - The generation header is `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
  - A previous reply is replaced by the exact token IDs it was generated with, matched after `strip()`, so trimmed echoes are fine.
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

## 7. SWE-bench Verified with Mini-SWE-Agent 2.4.2

This is the setup of the multi-session SWE-bench pilots: 3 Astropy instances × `full`/`goal_query`/`observation_query` with 3 sessions per server (job 3299922), and 10 instances × the same 3 policies with 5 sessions per server (job 3300434), on runtime `b1e6988` with 784-token pages. The same 10 instances then ran with 16-token pages × `full`/`observation_query`/`goal_query`/`random_fill`, one job per policy (jobs 3313802, 3314010, 3314319, 3314623, runtime `780fe54`; outcomes in section 4.7), with the agent host unchanged. Every server passed its server-side checks. One run per instance is not benchmark evidence. `random50` was not part of these pilots.

### 7.1 Settings

| Setting | Value |
| --- | --- |
| Agent | Mini-SWE-Agent 2.4.2 (litellm 1.103.1, datasets 5.0.1) |
| Runner | the PoC's `swebench_runner.py` with `--subset verified --split test --workers <sessions> --filter '^(<id>\|<id>…)$' --output <dir>` |
| Config stack | `-c swebench_xml.yaml` (built into Mini-SWE-Agent) `-c swebench_prompt_override.yaml` (the PoC's prompt) `-c model_override.yaml` |
| Model | `examples/swebench/model_override.yaml`: `litellm_textbased`, `openai/selective-prefix`, `api_base` `http://127.0.0.1:<port>/v1`, dummy key, temperature 0.0, `max_tokens` 1536, `cost_tracking: ignore_errors` |
| Environment | `docker`, `cwd` `/testbed`, timeout 120 s, `bash -c`, `PAGER`/`MANPAGER` `cat`, `LESS` `-R`, `PIP_PROGRESS_BAR` off, `TQDM_DISABLE` 1 |
| Limits | `step_limit` 90, `cost_limit` 0 |
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
| `runtime.jsonl` | One row per turn: response `id`, internal request ID, token/order hashes, `candidate_page_count`, `selected_logical_pages`, `prepare`/`restore` H2D bytes, joined purge/drain, latency. `stage_b` holds the turn geometry: the computed start C, the candidate end, the observation span, the phase-1 range, the phase-2 start, `retain`/`restore`/`release`, the computed end C′ = prompt + generated − 1, the sealed end and `rebuild`. Goal/Observation rows add `query.selection` (`reason`, `capacity_pages`, `mandatory_pages`, `recent_pages`, `retrieved_pages`, contract and projection hashes) and `query.goal` (variant, rewriter attempt/success/fallback, cache hit, cold forward) |
| `native-<sha256(request id)>.json` | Native plan (`selected`, `retain`/`restore`/`release`, run pages, tail flags, phase-1 lease), worker frames (`prepare`, `phase1`, `phase1_drain`, `reserve`, `allocation_update`, `complete`) and drain receipt |
| `failures.jsonl` | Contract/runtime failures with the exception `chain` and `traceback` (empty on success) |
| `engine-cleanup.json`, `parent-cleanup.json` | Shutdown receipts |

The decoupled layout adds fields; the default layout writes none of them (`docs/exp-decouple-pages/WP4.md`, section 4):
- `ready.json`: `kv_layout` (P, M, K), `num_blocks_by_pool` and `free_blocks_by_pool` (attention pool first, Mamba pool second); `worker-attached.json`: the same as `kv_pools`.
- Every `runtime.jsonl` row and native file: `kv_pools.free_blocks_by_pool` after that request's drain.
- `engine-cleanup.json`: `kv_pools` once every session closed. After a clean run both pools are back at the readiness baseline (every block but each pool's null block free), which the validated runs checked.
- Sealed page records name their tokens by `token_span` [pP, (p+1)P] instead of `token_sha256`, and page vectors use the `compact_int8_v1` form `[[128 int8], scale]`.

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
- Compare policies only on histories that actually evict pages to CPU, and report these components separately.

These files can contain prompts and tool output. Keep them out of public source releases. Joined transfers establish mechanism behavior, not task quality or speedup. Historical labels inside receipts such as `NOT_SYSTEM_VALID` or `cuda_qualified=false` are deliberately conservative capability strings. Judge a run by its READY, per-turn native evidence, cleanup and task evaluation.

## 9. Release integrity and notices

`SOURCE_SHA256SUMS.txt` and `INSTALL_RELEASE.json` describe the delivered core and its exact mapping to clean upstream. The review-only `patches/v0.24.0.diff` is not regenerated with each runtime (the repository has no generator for it): it lacks several later vLLM edits, including the decoupled layout's, so review against the pinned sources instead. `package.py` builds a deterministic source archive of the core from a fixed file list; the Slurm and SWE-bench examples (and some newer tests) are not in that list, so use a Git checkout or `git archive` for them. `qualification_parent_manifest` in READY points to the historical qualification lineage; it is not a certificate for a new run. The Apache-2.0 [LICENSE](LICENSE) and upstream notices are retained. Model configuration/tokenizer provenance and third-party terms are in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt). `examples/swebench/upstream/` contains the PoC's runner and prompt, copied unchanged from the dram-backed-kv repository; they keep that repository's terms.

## History

- **Current runtime: commit `780fe54df8124d75ea76ad752bb6692427a89ed5` (2026-10-05).** It is the runtime below plus the `random_fill` control of the query policies (section 5).
  - `random_fill` is `observation_query` with only the fill step changed: the remaining capacity is filled in a SplitMix64 / Fisher-Yates order keyed by the startup seed (`--fill-seed`, default 42), the session index and the session's request ordinal, instead of by cosine. Only its selection contract carries the `random_fill` key (`remote_ra_live_v2/selection_contract.py`, `remote_ra_live_v2/query_selection.py`, `serve.py`).
  - The four other policies send and write the same bytes as on the runtime below (`test_random_fill_legacy.py`).
  - At 16-token pages, on the reused clean environment of job 3270721, it served the 10-instance SWE-bench pilot with Mini-SWE-Agent 2.4.2, one job per policy: `full`, `observation_query`, `goal_query` and `random_fill` (Slurm jobs 3313802, 3314010, 3314319, 3314623). Every server passed. One run per policy is not benchmark evidence (section 4.7).
  - It has not had its own clean-install acceptance.
- **Previous runtime: commit `8f1af6a15356a0b83c5e90fa4258e46bea0761e5` (2026-10-04).** It is the runtime below plus the opt-in decoupled KV layout (section 4.6). The default layout, policies, selection and the agent protocol are unchanged.
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
