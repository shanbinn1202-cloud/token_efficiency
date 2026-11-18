# Install and uninstall

English | [简体中文](install.zh-CN.md)

Sections 1, 2, 3 and 9 of the README until integrate-1107, with their numbers. Section numbers are unique across the docs pages; the table of the README's [Documentation](../README.md#documentation) section lists the page that holds each.

## 1. Supported boundary

- Linux, Python 3.12, clean upstream **vLLM 0.24.0** installed from its wheel, Transformers **5.13.0**, Tokenizers **0.22.2**.
- Model: local **Qwen3.6-27B** snapshot at revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. Metadata hashes are in `INSTALL_RELEASE.json`; weights are not distributed. The three pinned metadata files must be regular files, not Hugging Face cache symlinks.
- Second model profile: **Gemma 4 31B IT**, a release of its own (`install.py --model-profile gemma4-31b-it`); its revision, pinned files, install commands and settings are in section [4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d). The rest of this page describes a Qwen release.
- Tested hardware: one NVIDIA GH200 120GB on Linux aarch64. Other GPUs, architectures and backends are untested.
- Synchronous by default (`--async-scheduling` is opt-in, sections 4.15 and [docs/async-sched/](async-sched/)), text-only, UniProc executor, TP=PP=DP=1, **one session per startup task, up to 32 per server** (section 4.4; 16 sessions ran on the GPU in jobs 3339019 and 3343069; more than 16: [docs/known-limits.md](known-limits.md) 2.1); with `--max-running-sessions K`, up to 256 registered tasks (64 before integrate-1117) of which at most K (up to 32) run at once (section 4.4; on the GPU: 24 tasks with K = 8 in job 3343069). A chat-text server (section 6.7) runs up to K = 24 sessions at once over up to 1,024 slots (`--chat-text-slots`); a kv-session server (section 6.6) binds up to 64 client sessions (`--kv-session-slots`). The accepted run used one session per server.
- With `--no-speed-defaults` the launcher pins the qualified server configuration below. Without it, the speed defaults of section [4.15](reference.md#415-speed-defaults) also apply: CUDA graphs `full`, the three decode fast paths, `--deferred-completion-io`, `--language-model-only`, a 1-GiB connector pool and, with `--enable-thinking`, 4096-token replies:

  | Setting | Value |
  | --- | --- |
  | `max_model_len` / `max_num_batched_tokens` | 65536 / 16384; `--max-model-len 131072` selects a 131072-token context (GPU-validated in job 3348705, section 4.9); `--max-model-len 262144`, the model's native limit, selects a 262144-token context (GPU-validated in job 3360375, section 4.9) |
  | reply bound (`--max-tokens`) | 1536 new tokens per turn, the PoC's value; with `--enable-thinking` 4096 unless `--no-speed-defaults` (section 4.15); `--max-tokens 4096` (GPU-validated in job 3360375, where 7 replies reached the bound; in job 3348705 6 replies of the 16-session server reached it, section 4.9) |
  | page size and KV layout | 784 tokens in one shared pool (default); `--page-tokens 16`, `32` or `64` always use the decoupled layout, with separate attention and Gated-DeltaNet pools (section 4.6); `--kv-layout decoupled` adds that layout at 784, for validation only. Validated on the GPU: 784 (default) and 16 decoupled; 64 ran once (job 3366128, SPD1_PARTIAL_PASS_VERIFIED: `full` in a decode measurement); 32 never ran on a GPU with Qwen |
  | GPU KV / connector pool | 8 GiB (`--kv-cache-gib` 16, 24, 25 or 32 for several sessions; 25 ran in jobs 3461661 and 3461703) / vLLM's KV connector pool: 1 GiB by default (section 4.15), 32 GiB with `--no-speed-defaults` or `--no-connector-pool-gib`; the runtime never stores to or loads from it |
  | execution | `--cuda-graphs full` by default (section 4.15); eager (no CUDA graphs / torch.compile) with `--cuda-graphs off` or `--no-speed-defaults`, the qualified profile; `--cuda-graphs piecewise`, `full` or `full-decode` is GPU-validated (jobs 3339448 and 3343069; job 3338893 ran the modes but did not complete), but its outputs can differ from eager (section 4.8) |
  | admission | one at a time in arrival order; `--gdn-device-handoff`, `--phase1-in-step` and `--admission-overlap K` are opt-in, GPU-validated with `goal_query` in job 3349504 (section 4.10) |
  | Mamba cache | `align` |
  | GDN prefill | Triton |
  | parsers | `qwen3_xml` tool-call, `qwen3` reasoning |
  | served model name | `selective-prefix` |

  **Prompt plus completion must fit in 65536 tokens** (the context limit; a prompt holds at most the context limit minus the reply bound, 64,000 tokens at the 1536-token bound). The accepted run (job 3270721) used this profile with prompts up to about 33K tokens. 8 GiB of GPU KV holds the resident working set plus one turn's new tokens. At 64k tokens `full` keeps about 84 attention pages of 49 MiB (4.0 GiB), next to a few Gated-DeltaNet states of 147 MiB. With 16-token pages the same 64K tokens are 4,096 attention pages of 1 MiB (4.0 GiB).

## 2. Install and uninstall

Use the same target Python for installation, installed-check and serving. Commands run from this repository with the Quickstart placeholders.

### 2.1 Base environment

The validated environment is the one clean-install acceptance job 3270721 created; every run that [History](history.md#history) lists "on the reused clean environment of job 3270721", and both SWE-bench pilots, used it read-only. It was built like this on Linux aarch64:

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

- **Base interpreter.** Job 3270721 used uv's managed CPython 3.12.13 as `BASE_PYTHON`: its `sys.base_prefix` was `cpython-3.12.13-linux-aarch64-gnu` in uv's default directory `~/.local/share/uv/python/`, and Triton compiled against the headers there (`include/python3.12`). How it was installed is not recorded and not part of the validated runs. With uv, `uv python install 3.12.13` and then `BASE_PYTHON="$(uv python find 3.12.13)"` should give the same interpreter (untested). Any other CPython 3.12 with `venv` and its headers should work but is untested.
- **Wheel.** The official PyPI file, 271,361,241 bytes, SHA-256 as above. The job downloaded that URL once into its job directory; `curl` into `$RUNS`, outside the Git checkout, is only one way to do it. `mkdir -p "$RUNS"` comes first because the pip report goes there too.
- **Disk space.** The environment of job 3270721 holds 66,550 files, 9,071,166,786 bytes (8.4 GiB). The release of section 2.4 copies its installed `vllm/` tree (4,404 files, 613,445,655 bytes, 0.57 GiB) plus the overlay. The checkpoints of section 2.2 add about 51 GiB for Qwen3.6-27B (the server loads 51.1 GiB of weights, section 3) and 1.4 GiB for Qwen3-0.6B (`model.safetensors`, 1,503,300,328 bytes). Put `$VENV`, `$MODELS`, `$RELEASE` and `$RUNS` on a filesystem with room for them.
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

Two checkpoints for a Qwen release, both as local directories (the Gemma 4 31B IT checkpoint and its install: section [4.14](reference.md#414-gemma-4-31b-it-stage-1-gpu-validated-at-cf2b28d)):

| Model | Revision | Used by | Pinned how |
| --- | --- | --- | --- |
| [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` | every server (`--model-dir` of `install.py`) | `install.py` (check-only and `--apply`) verifies the SHA-256 of `config.json`, `tokenizer.json` and `tokenizer_config.json` against `INSTALL_RELEASE.json`; it does not hash the weights |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | `goal_query` with `--goal-rewriter-model` | `serve.py` pins the resolved directory and the SHA-256 of its `config.json` in the selection contract; `install.py` does not check it |

The three pinned Qwen3.6-27B metadata files must be regular files, not Hugging Face cache symlinks (the installer refuses symlinks). The validated runs did not record how the checkpoints were downloaded, so the commands below are **not part of the validated runs**. The base environment of section 2.1 already has the `hf` CLI of huggingface-hub 1.33.0 (`$VENV/bin/hf`). Call it by that path: `$VENV/bin` is on `PATH` only from section 3 on, and nothing else may be installed into `$VENV` (section 2.1). With `--local-dir` it should write regular files; the `find` line checks that:

```sh
"$VENV/bin/hf" download Qwen/Qwen3.6-27B --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --local-dir "$MODELS/qwen3.6-27b"
"$VENV/bin/hf" download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
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

The tests are the modules `tests/test_*.py`, with their helper `tests/gemma4_testing.py` and `tests/testdata/` (since integrate-1107, as in the release repository; before, they were at the root). The suite needs `torch` (CPU is fine; several test modules import it, others skip their tensor tests without it) and `git` on `PATH`. The base environment of section 2.1 has both; job 3270721 ran the public tests there. The number of tests changes with every branch: `run_tests.py --all` prints it (`run` in its summary), and each revision's release check is named in [History](history.md#revisions). On a desktop CPU the sequential `run_tests.py --all` takes about 40 minutes, `check_git_release.py --ref HEAD` with ten workers about 9 to 11 (integrate-1116, 2026-11-16), and the fast default about 5 (`tests/FAST_MODULES.timings.json`). One test, `test_installer`'s `test_git_archive_head_preserves_pins_and_runs_full_suite`, runs `check_git_release` on a Git snapshot of the core package's files (`package.PUBLIC_FILES`, which hold a few of the test modules).

```sh
"$VENV/bin/python" -B run_tests.py --all
"$VENV/bin/python" -B check_git_release.py --ref HEAD
```

`run_tests.py` runs each module as `python -s -B -m unittest <module>` from the repository root, one after the other, with the root and `tests/` on `PYTHONPATH` and one OpenMP thread, and ends with one JSON summary (`status`, `run`, `failures`, `errors`, `skipped`, `failed_modules`). Without arguments it runs the fast default of `tests/FAST_MODULES.txt`: every module that took at most 10 s in the timing run `tests/FAST_MODULES.timings.json`, plus `test_installer` (about a minute, 58 s in that timing run; the CI runs it). With module names it runs those (`"$VENV/bin/python" -B run_tests.py test_http test_lease`), `--timings FILE` writes each module's seconds, and `--fast-from FILE` rewrites the fast list from a new `--all --timings` run.

`check_git_release.py --ref HEAD` runs the suite on the `git archive` bytes of that commit, with one OpenMP thread (`OMP_NUM_THREADS=1`) per process; it must pass before publishing a commit. By default each test module of `tests/` runs in its own process (unittest's discover in `tests/` with that file as the pattern, from the archive root, with the root and `tests/` on `PYTHONPATH`), ten at a time (`--jobs`), the few modules with tight timing bounds alone after them, each module within 1200 s and all within 3600 s; the result line still ends in one `Ran N tests in S` and unittest's own `OK` / `FAILED` line. `--serial` is the former single-process run (one `unittest discover`, 3600 s); `--record` runs the suite in one process like it and, with `--outcomes FILE`, writes every test id's outcome, to compare the two modes.

**Slow pass.** The normal pass (without `REMOTE_RA_SLOW_TESTS`), and with it `check_git_release.py` and `run_tests.py`, runs a short variant of the largest CPU end-to-end cases: one or two (layout, policy) pairs of each layout-by-policy matrix, `full` and `goal_query` in the comparisons with former code, the kv-session MemAgent sessions with `goal_query`, 16 sessions with `observation_query` only, the 10-task session pool, 16 of the 48 random selection catalogs, the Gemma end-to-end cases on 6 or 8 sessions (section 4.14) and the chat-text sessions without summaries and asides, Qwen's and Gemma's (sections 6.7 and 4.14), and 4 Gemma `goal_query` Terminus 2 clients instead of 16 (section 4.14). `REMOTE_RA_SLOW_TESTS=1` runs every case, plus the 32,768-token pressure case of section 4.13. Every validation before a push runs both passes and `check_git_release.py --ref HEAD`:

```sh
REMOTE_RA_SLOW_TESTS=1 OMP_NUM_THREADS=1 "$VENV/bin/python" -B run_tests.py --all
```

**CI.** `.github/workflows/tests.yml` runs `run_tests.py` (the fast default) on GitHub Actions for pushes to `main` and `integrate-*`, pull requests and by hand: ubuntu-24.04, Python 3.12, CPU torch, `transformers==5.13.0`, `tokenizers==0.22.2`, one OpenMP thread, 30 minutes; first it checks the delivered sources against `INSTALL_RELEASE.json` (`install.verify_package`) and `examples/swebench/upstream/SHA256SUMS`.

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
- It creates an independent `<release>/source` tree and pins its dependencies. **It never edits site-packages, installs dependencies, starts a model or enables the feature.** Allow disk space for a full vLLM copy (about 0.6 GiB, section 2.1) and keep the source environment unchanged while copying.
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

The accepted clean-install run used the settings below, and so did the later jobs on its environment. Set them, and the Quickstart placeholders again, in the shell that starts `serve.py` on the GH200 node. The validated jobs ran as Slurm batch jobs (`sbatch`; the benchmark jobs used the GH200 batch partition `ghx4`), and `examples/slurm/serve_gh200.sh` sets this environment itself (section [4.5](running.md#45-gpu-job-slurm)). For an interactive shell on a GPU node, use your cluster's way of allocating one; that is not part of the validated runs. The launcher passes the environment to the server, without every `REMOTE_RA_*` variable (it keeps only the step-profile pair `REMOTE_RA_STEP_PROFILE` and `REMOTE_RA_STEP_PROFILE_EVERY`, and turns `REMOTE_RA_TORCH_PROFILE` into a vLLM flag) and without vLLM's policy runtime-manifest variable. It then sets its own runtime flags and starts the server with `python -I -S -B`.

```sh
# The target environment's bin: its Python and the Ninja used by JIT builds.
export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
# CUDA 13 runtime of the target environment (libcudart.so.13) for the isolated launcher.
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib"
# vLLM binds ZMQ IPC sockets at $TMPDIR/<36-byte uuid>: keep the path short (AF_UNIX limit 107 bytes).
export TMPDIR=/abs/short/tmp
# The directory must exist and be private to you.
mkdir -p -m 700 "$TMPDIR"
# Native top-k/top-p sampler; the FlashInfer sampler needs a JIT toolchain and a warm cache.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Triton compiles launchers with $CC, else gcc/clang; some clusters export CC=cc.
unset CC VLLM_RPC_BASE_PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
```

Notes:
- `len(TMPDIR) + 37` must stay at or below 107 bytes, so `TMPDIR` may hold at most 70 bytes (the acceptance job's 51-byte `TMPDIR` gave 88). A longer path makes the first server exit before READY with `zmq.error.ZMQError: ipc path ... is longer than 107 characters`. The Slurm script uses `$RUN_ROOT/<job id>/tmp`, created under `umask 077`.
- With an empty cache, vLLM 0.24's default FlashInfer sampler JIT-compiles on first use. That needs `nvcc` and Ninja and can take minutes. The native sampler avoids it, and with `temperature=0` the output does not depend on the sampler. The FlashInfer sampler on a cold cache is outside the accepted profile.
- Triton JIT-compiles kernels on the first requests. It needs a working `gcc` (or `clang`) and the Python headers of the target interpreter.
- Separate cache directories per run (`HOME`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, …) are fine. The validated jobs used empty, job-local caches (see `examples/slurm/serve_gh200.sh`).
- Memory: the model (51.1 GiB) plus 8 GiB KV and the optional 0.6B Goal rewriter (about 2.5 GiB with its CUDA context) fit one GH200 (94.5 GiB free at start); so did 24 GiB KV with five sessions and the rewriter (job 3300434), at 784-token and at 16-token pages (job 3314319). Host memory: the launcher refuses a server whose pinned host memory could exceed 128 GiB. That worst case is vLLM's connector pool (1 GiB by default, 32 GiB with `--no-speed-defaults`; the runtime never uses it), one 147-MiB Gated-DeltaNet carry per session (two with `--gdn-device-handoff`) and the host copy of every captured page, about 4 GiB per session at a 65536-token context in either layout (at 16-token pages the decoupled layout's pinned page arena, section 4.6; about 8 GiB at a 131072-token context and 16 GiB at 262144, section 4.9). Allow that plus about 32 GiB for the processes (`examples/slurm/serve_gh200.sh`). The jobs up to 2026-10-13 requested `--mem=200g`: the acceptance 3270721, the SWE-bench pilots and the context jobs 3348705 and 3360375. Since 2026-10-14 the benchmark jobs request one GH200 memory share, `--mem=113560M` (section 4.5). The MemoryAgentBench Qwen servers request two, `--mem=227120M`, from the G4 runs on (jobs 3405844 and 3405845; the earlier jobs 3397842 and 3397844 requested one). A few later measurement jobs kept `200g` by choice.

## 9. Release integrity and notices

`SOURCE_SHA256SUMS.txt` and `INSTALL_RELEASE.json` describe the delivered core and its exact mapping to clean upstream. The review-only `patches/v0.24.0.diff` is not regenerated with each runtime (the repository has no generator for it): it lacks several later vLLM edits, including the decoupled layout's, so review against the pinned sources instead. `package.py` builds a deterministic source archive of the core from a fixed file list; the Slurm and SWE-bench examples (and most test modules) are not in that list, so use a Git checkout or `git archive` for them. `qualification_parent_manifest` in READY points to the historical qualification lineage; it is not a certificate for a new run. The Apache-2.0 [LICENSE](../LICENSE) and upstream notices are retained. Model configuration/tokenizer provenance and third-party terms are in [THIRD_PARTY_NOTICES.txt](../THIRD_PARTY_NOTICES.txt). `examples/swebench/upstream/` contains the PoC's runner and prompt, copied unchanged from the dram-backed-kv repository; they keep that repository's terms.
