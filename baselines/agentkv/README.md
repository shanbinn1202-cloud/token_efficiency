# Introduction

Phase-aware KV-cache eviction for agentic LLM serving, implemented as a plugin inside a
compact SGLang-style inference engine.

## Layout

```
engine/        inference engine (Mini-SGLang) with the AgentKV eviction + compaction path
gorilla/       BFCL benchmark fork (patched with a mini-sglang model handler)
tau2-bench/    tau2-bench fork (patched with a mini-sglang model handler)
experiment/    runners + scripts that drive a benchmark against the engine
```

## Install

`engine/`, `gorilla/`, and `tau2-bench/` are git submodules — forks of upstream
mini-sglang / BFCL / tau2-bench carrying the AgentKV patches — so clone recursively:

```bash
git clone --recursive git@github.com:LiuTaowen-Tony/agentkv.git
# already cloned non-recursively? pull the submodules in:
git submodule update --init --recursive

python -m venv .venv && source .venv/bin/activate
pip install -e engine
```

Requires Linux + an NVIDIA GPU with a recent CUDA toolkit (the engine JIT-compiles CUDA/Triton
kernels). Models are referenced by HuggingFace repo id; set `AGENTKV_MODEL_DIR` to reuse a
local HuggingFace cache. Copy `.env.example` to `.env` for optional API keys.

## Eviction plans

Select a policy with `--eviction-plan` (server) or `EVICTION_PLAN=` (runner), as
`<method>-<budget>`:

- `identity` / `none` — no compression
- `agentkv-<budget>` — AgentKV (phase-aware Stage-Q with SnapKV scoring)
- `rkv-<budget>` — R-KV
- `streamingllm-<budget>` — StreamingLLM
- `triattn-<budget>` — TriAttention (requires external calibration stats; see
  `engine/python/minisgl/eviction/triattention_stats/README.md`)

## License

AgentKV code is Apache-2.0 (see `LICENSE`). Bundled components keep their upstream licenses —
Mini-SGLang (MIT), BFCL (Apache-2.0), tau2-bench (MIT). See `NOTICE`.
