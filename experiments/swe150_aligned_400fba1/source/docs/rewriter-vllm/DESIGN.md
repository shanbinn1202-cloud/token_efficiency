# Goal rewriter on vLLM (`--goal-rewriter-engine vllm`, opt-in)

English | [简体中文](DESIGN.zh-CN.md)

Branch `rewriter-vllm` on `cf2b28d`; design of 2026-10-30 (step 1) with the lm-format-enforcer fix of 2026-11-01,
both on `main` (`cd42995`; `16dcb4d`, `c07d42a`). The engine is part of the stable configuration (with
`--goal-rewriter-batch-invariant` and `--goal-overlap`, job 3576565). Its GPU runs and their verdicts are in
[docs/status.md](../status.md) and [History](../history.md#history); its limits (the child's GPU memory is not
reserved; validated only inside the stable command) are in [docs/known-limits.md](../known-limits.md) 3.1 and 3.2.
What the GPU runs measured for this design is in the sections below. Step 2 is built as `--goal-overlap`
([docs/goal-overlap/DESIGN.md](../goal-overlap/DESIGN.md)).

## Why

On Terminal-Bench (16 sessions, Qwen3.6-27B, `cf2b28d`) a `goal_query` turn spent 3.1 s on average waiting for its
Goal: 1.9 s in the rewriter's generation and 1.2 s queued behind other sessions' calls. On the 75 of 362 turns that
called the rewriter the wait averaged 15.0 s (median 9.7 s, p95 53 s). The rewriter (Qwen3-0.6B) runs Hugging Face
`generate` with lm-format-enforcer in one child process, serving one batch of at most 4 calls at a time under one
lock: 72 ms per generation step alone, 94 ms batched (60.5 ms alone on an idle GPU in the earlier replay).

The aim is to cut the small model's cost and to overlap it with the large model. Step 1 (this document)
moves the generation into a small vLLM engine with continuous batching. Step 2 (later) overlaps phase 1 with the
rewriter call.

## What stays exactly as today

- When the rewriter is called, the request it is sent (`RewriteRequest.to_payload()`), the prompt (the vendored
  `qwen_goal_rewriter.py` system text, the canonical user JSON, `apply_chat_template(..., add_generation_prompt=True,
  enable_thinking=False)`), the JSON schema (`candidate_schema`), greedy decoding, `max_new_tokens` 384, the decode
  (`skip_special_tokens=True`), `parse_exact_candidate`, the parent's validation (`validate_rewrite_candidate`), every
  fallback and its reason, the 120-s call timeout and the answer line format.
- The rewriter object stays a `PersistentCommandGoalRewriter` (its class name is part of the Goal constructor's control
  hash and so of every `selector_config_hash`); the selection contract's `goal_rewriter` pin (`REWRITER_KEYS`) is
  unchanged. As with `--goal-rewriter-batch`, a transport is installed over that instance.
- `goals.jsonl` keeps every field (`request`, `response`, `error`, `wait_ms`, `start_ms`, `call_ms`, `input_tokens`,
  `output_tokens`, `generation_latency_ms`); the native file keeps `goal_rewrite_wait_ms` / `goal_rewrite_call_ms`.
- Without the flag nothing changes: no new startup record, the HF child, its command and every pin as today.

What changes: the model call. vLLM's kernels and its structured-output backend are not the HF ones, so a Goal can
differ from the HF server's (not bit-identical, as with batching). Compare policies only between servers with the
same engine; the replay (below) measures how often the candidates agree.

## Topology

A new child script, `source-overlay/remote_ra_live_v2/goal_rewriter_vllm_child.py`, runs one `AsyncLLM` engine for the
pinned Qwen3-0.6B snapshot in its own process (and its EngineCore subprocess), on the same GPU. Not inside the 27B
engine (one model per vLLM engine), not an offline `LLM` (not safe from several threads), not an HTTP server (no
second port for the harness's port checks to track; no logs mixed into a protocol).

- **Clean vLLM.** The child starts with `sys.path` = the clean environment's site-packages only (not the overlay,
  whose patched `vllm/` and gates would load), the vendored `qwen_goal_rewriter.py` loaded by file path for its prompt,
  schema and parse functions, and an environment scrubbed of `REMOTE_RA_*`, `VLLM_POLICY_*` and `SELECTIVE_*`
  (`REMOTE_RA_HTTP_RUNTIME`, `REMOTE_RA_DYNAMIC_STARTUP`, `VLLM_POLICY_RUNTIME_V1` are set by `serve.py`).
- **stdout carries only protocol lines.** The child duplicates fd 1 for the protocol, then points fd 1 at fd 2 before
  importing vLLM, so vLLM's logger, its EngineCore and any C-level print go to stderr.
- **Process group.** The child starts in its own session; its close kills the group, so no EngineCore is orphaned
  holding GPU memory.
- **Engine arguments** (pinned, recorded in `startup.json`): the pinned snapshot, `local_files_only`, bfloat16,
  `max_num_seqs` 16, `max_model_len` 16,384 (the bounded request is at most ~7.5K prompt tokens in the recorded calls:
  7,544 at most), `max_num_batched_tokens` 8,192, `kv_cache_memory_bytes` from `--goal-rewriter-vllm-kv-gib` (default
  3 GiB, about 28K tokens at 112 KiB per token; vLLM logged 28,080), `gpu_memory_utilization` 0.05 (only vLLM's startup gate once the KV bytes are
  explicit: it needs 4.8 GiB free), prefix caching off, CUDA graphs on with capture sizes 1, 2, 4, 8, 16, the
  structured-output backend from `--goal-rewriter-guided-backend` (below), `generation_config` "vllm".
- **Requests.** The prompt is the token ids the HF tokenizer gives for the upstream prompt text (`TokensPrompt`), so
  `input_tokens` equals today's count. Sampling: temperature 0 (greedy), `max_tokens` 384, `stop_token_ids` the
  tokenizer's `<|im_end|>` and EOS, `structured_outputs=StructuredOutputsParams(json=schema)`, `detokenize=False`,
  `output_kind` FINAL_ONLY. The child decodes the generated ids with the HF tokenizer and `skip_special_tokens=True`,
  exactly as upstream, then `parse_exact_candidate`; a parse failure (including a reply cut at 384 tokens,
  `finish_reason` "length") answers with the upstream error line and its token counts.
- **Concurrency.** One asyncio loop on a dedicated thread owns the engine. A reader thread parses request lines and
  submits each with `run_coroutine_threadsafe`; answers are written, one line each under a write lock, in completion
  order. The line protocol is the upstream one (`{"request_id","request"}` in, `{"request_id","result"}` or
  `{"request_id","error"[,"telemetry"]}` out, the same readiness record); only several calls may be in flight and
  answers may come out of order. An `{"abort": request_id}` line aborts one request.

## Parent side

`source-overlay/remote_ra_live_v2/goal_rewriter_vllm.py` installs over the `PersistentCommandGoalRewriter` instance,
like `BatchedRewriter.install`:

- one child, started **eagerly** once the 27B engine has allocated its KV cache (vLLM's startup gate and the memory
  order need the large engine first), before READY, so the first Goal calls do not pay the engine's start (compile,
  graph capture; the startup timeout stays 300 s). `ready.json` records the child's startup time.
- each `rewrite()` runs the upstream `rewrite` against a ticket (as in `goal_rewriter_batch._Ticket`: faked lock,
  process, `_readline`, `close`) so the upstream envelope, answer checks and validation run unchanged and
  `RewriterTap` stamps `locked` / `sent` / `answered` as today (`wait_ms` becomes the wait for another call's restart
  of the child, near 0; see "Timing" below);
- a reader thread routes answer lines to their tickets by `request_id`;
- a call's timeout aborts that request in the child and fails that call only (today a timeout restarts the child;
  restarting a vLLM engine costs a minute); a child exit fails every call in flight and the next call restarts it;
- shutdown closes the child's process group.

## Flags and pins

- `--goal-rewriter-engine {hf,vllm}` (default `hf`). `vllm` needs `goal_query` with `--goal-rewriter-model` and
  `--admission-overlap K` (K >= 2; otherwise only one call is ever in flight) and excludes `--goal-rewriter-batch`.
- `--goal-rewriter-guided-backend {lm-format-enforcer,xgrammar}` (default `lm-format-enforcer`, the enforcer the HF
  child uses). vLLM 0.24's xgrammar (and guidance, so also `auto`) refuse the schema's `uniqueItems` (not
  re-checked: no record of that test is kept, and every GPU arm sent xgrammar the schema without it); the
  `xgrammar` setting would pass the schema without `uniqueItems` (the parent's validation still rejects duplicates).
  That changes the constraint the model sees, so it is a measurement arm only, for the Goal owner to decide on.
- `--goal-rewriter-vllm-kv-gib G` (default 3) and `--goal-rewriter-batch-invariant` (sets `VLLM_BATCH_INVARIANT=1` in
  the child only: its outputs no longer depend on how many calls run together). Measured in rw3 (job 3499550; from its offline
  verification, not audited): about as fast per call (wall-clock median 947.54 ms alone against 974.89 ms without it; 16,604.77 against 16,373.98 ms with 16 in
  flight, 1.4 % slower), but its engine start took 73,954 ms against 35,427 ms.
- `startup.json` / `ready.json` gain a `goal_rewriter_engine` record (engine, backend, KV bytes, utilization gate,
  batch invariance, engine arguments, startup ms) only with `vllm`. The Gemma 4 profile takes the flag too
  (same child; its GPU memory: next section).

## GPU memory

Nothing reserves room for the rewriter today; the HF child's batch cap leaves about 1 GiB free next to the 27B
server (`serve.py` memory notes). The vLLM child needs weights (~1.2 GB), its KV (3 GiB default), activations at
8,192 batched tokens, the CUDA-graph pool and a CUDA context: about 5-6 GiB. Next to the 25-GiB Qwen server about
12.4-14.1 GiB stay free: of CUDA's 97,362 MiB (the `serve.py` note), the 25-GiB Terminal-Bench run 3461661 peaked at
86,408 MiB with the HF child (1,721 MiB of it at load), so the server alone took at most 84,687 MiB (12,675 MiB
free); the same server without a rewriter peaked at 82,801 MiB, and `serve.py`'s 32-GiB note scaled to 25 GiB leaves
14,428 MiB. Roughly 7 GiB (97,362 - 90,102 MiB) is the 32-GiB server's figure (job 3404230). perf-rwreplay2's
`mem25` and `mem32` phases hold GPU memory so that exactly these 12,675 and 7,260 MiB stay free and start the child
there (its README). Measured in rw3 (job 3499550; not audited): with 12,676 and 7,260 MiB left free the child (3 GiB of KV)
started and answered 16 of 16 calls in each; graph capture took 0.11 GiB, and the child peaked at 5,786 and 5,718 MiB.
Gemma (31B weights) leaves less: in job 3538523 the Gemma server with the child (gvll) peaked at 92,500 MiB against
89,874 MiB for the same server with the Hugging Face child (govl); gvll became READY in 261.1 s, its child started in
about 112 s (from the job's offline verification, not audited).

## Validation

- **CPU** (`test_goal_rewriter_vllm.py`): a fake `vllm` package for the child (a scripted `AsyncLLM` with delays);
  the prompt ids equal the HF `prepare` ids; decode and parse as upstream; 16 concurrent calls answered out of order
  and routed correctly; error, cut-at-384, timeout + abort, child exit; scrubbed environment and `sys.path` (the
  child never imports the overlay's `vllm`); stdout only protocol lines; flag validation and pins; `hf` default:
  startup config, command and evidence byte-identical to today; the control hash unchanged.
- **GPU 1, replay** (modelled on perf-rwreplay rw1): the 682 recorded rewriter calls, arms HF (re-run), vLLM +
  lm-format-enforcer, the same with batch invariance, vLLM + xgrammar (schema without `uniqueItems`): valid and
  fallback rates, candidate agreement with HF (HF's own re-run agreement is the floor: 92 % identical in rw1), per-call
  latency alone and at 16 in flight, ms per generation step, GPU memory.
- **GPU 2**: a Terminal-Bench (or SWE-bench) Goal dry run with `--goal-rewriter-engine vllm`, per-turn breakdown
  against the 10-26 run.

## Step 1 as built (2026-10-30)

Built as above: `remote_ra_live_v2/goal_rewriter_vllm_child.py` (child), `goal_rewriter_vllm.py` (transport, record,
pins), the four flags in `serve.py` for the Qwen and the Gemma 4 profiles, `remote_ra_http/config.py` (load check),
`query_selection.py` / `private_endpoint.py` (the authority starts the child from the pinned record),
`remote_ra_http/evidence.py` (`worker-attached.json` and `ready.json`), [docs/reference.md section 4.10](../reference.md#410-admission-optimizations-opt-in) and [docs/history.md](../history.md#history) (both languages),
release pins (329 sources, 238 payload). Tests: `test_goal_rewriter_vllm.py` (27 then; on `main` it is
`tests/test_goal_rewriter_vllm.py` with 31; the process-group test runs on Linux only), including the CPU list above and an end-to-end run: test_gemma_goal_rewriter's sixteen Gemma 4
`goal_query` sessions with `goal_rewriter_engine` pinned decide exactly as with the Hugging Face child's fakes (the
authority starts the real child from the startup config). With `hf` the launcher, the rewriter command and the startup
evidence equal `cf2b28d`'s (identity probe: 283 items, 0 differences).

Details and deviations from the text above:

- **Eager start point.** No new hook: `compose_engine` builds the `QuerySelectionAuthority` (through
  `SplitPrivateEndpoint`) after `EngineCore.__init__` has profiled, allocated the KV cache and captured its graphs,
  and before the B0 handshake, the input thread and READY; the Hugging Face child starts there today. The vLLM child
  starts in the same place, within the pin's 300-s startup timeout. Its start time travels engine to frontend in
  `worker-attached.json` (`goal_rewriter_engine.startup_ms`, only with `vllm`), which `ready.json` copies.
- **The child loads `goal_rewriter_child.py` by file path too** (its `prepare` and `error_answer`, standard library
  only), so the prompt ids are exactly the Hugging Face `prepare` ids; `torch` is not imported by the child itself
  (the generate's CUDA calls see a host stand-in).
- **Stop ids:** the checkpoint's generation_config eos ids (where Transformers stops), the tokenizer's EOS and
  `<|im_end|>`: {151643, 151645} for Qwen3-0.6B, the Hugging Face set.
- **Engine arguments** add `tokenizer` = the model, `distributed_executor_backend` `uni`, `tensor_parallel_size` 1 and
  `disable_log_stats`; `local_files_only` is the child's `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` (vLLM has no such
  engine argument) and the tokenizer's own `local_files_only`. `--goal-rewriter-vllm-kv-gib` takes 2-16 (2 GiB is the
  smallest that holds one 16,384-token sequence at 112 KiB per token).
- **Abort** cancels the request's task (vLLM's `generate` aborts a cancelled request in the engine); an aborted
  request gets no answer line. A late answer of a call that already failed is dropped by the parent.
- **Timing:** `wait_ms` is the wait for another call's restart of the child, `start_ms` the call's own restart and
  the write of its line, `call_ms` the send to the answer; `generation_latency_ms` (child) runs from the submission
  to the engine to its final output, so it includes the time the request shared the engine with others.
- **Failures:** an engine that reports itself dead (`errored`) answers its call and ends the child, so the next call
  restarts it; the end of stdin cancels what is in flight and shuts the engine down. A prompt over 16,384 - 384 tokens
  is expected to fail its call (vLLM refuses it) where the Hugging Face child would run it; not exercised: the
  largest recorded prompt has 7,544 tokens.
- **Dry run:** the dry-run line also shows `goal_rewriter_engine`.

Open at that date (nothing had run on a GPU or against the real vLLM 0.24 wheel), and what the GPU runs answered
(from the jobs' offline verifications, not audited):

- The vLLM 0.24 names the child uses (`AsyncEngineArgs` fields, `CompilationConfig(cudagraph_capture_sizes=...)`,
  `StructuredOutputsConfig(backend=...)`, `StructuredOutputsParams(json=...)`, `RequestOutputKind.FINAL_ONLY`,
  `TokensPrompt`, `AsyncLLM.from_engine_args`, `errored`, `shutdown`) follow the API notes, unchecked against the
  wheel; the first GPU job checks them. Answered: the replay rw2 (job 3488202, runtime `cd42995`) passed its vLLM
  API probe.
- GPU memory next to the 25- and 32-GiB Qwen servers and the Gemma server; whether the 0.05 utilization gate passes
  after the 27B engine allocated; the child's start time (compile and graph capture) against the 300-s timeout.
  Answered: the child started next to the real 25-GiB server in 111,749 ms (job 3500565); the 32-GiB case ran only
  under a memory blocker (rw3 `mem32`); Gemma: section "GPU memory" (job 3538523).
- Whether vLLM's lm-format-enforcer backend constrains the schema (`uniqueItems` included) as the Hugging Face
  integration does, and what `VLLM_BATCH_INVARIANT=1` needs from the 0.6B engine (backend, eager or graphs).
  The first replay found the backend unusable on vLLM 0.24; see the next section for the fix. With the fix (rw3,
  job 3499550, runtime `c07d42a`) no call had a backend error.
- The CPU fakes decode rows independently, so they cannot show how often vLLM's Goals differ from Hugging Face's:
  that is the replay's agreement measurement. Answered in rw3: lfe.alone's candidates were identical to Hugging Face's
  on 99 of 160 calls (0.6188, below the replay's bar of 0.70); with batch invariance, alone and 16 in flight gave
  identical candidates on 160 of 160 calls. On Gemma (job 3538523) the Goal of the server with the child equalled the
  Hugging Face server's on 22 of 23 lockstep turns.

## lm-format-enforcer on vLLM 0.24 (rewriter-lfe-fix, 2026-11-01; ran on the GPU in rw3, job 3499550)

The first GPU replay (perf-rwreplay2, job 3488202, runtime `cd42995`) ran every lm-format-enforcer arm (`lfe`,
`lfebi`, `mem25`, `mem32`) to a failure on every call in a few milliseconds, while the xgrammar arm answered every
call. The child logs show why, once per request in the EngineCore: `preprocess_add_request` -> `grammar_init` ->
`LMFormatEnforcerBackend.__post_init__` -> `build_vllm_token_enforcer_tokenizer_data` -> vLLM's `LazyLoader` imports
`lmformatenforcer.integrations.vllm`, whose line 4 (lm-format-enforcer 0.11.3, the version installed in the clean
environment) is `from vllm.transformers_utils.tokenizer import MistralTokenizer`. vLLM 0.24 moved its tokenizers to
`vllm/tokenizers/` (`MistralTokenizer` is in `vllm/tokenizers/mistral.py`); the import fails and the integration
re-raises it as `ImportError: vllm is not installed`. The backend is built lazily and never cached after a failure,
so every request fails the same way; the engine stays up and ends each request without a token, so the child
answered every call as a parse failure (`JSONDecodeError`), which the replay counted as a fallback. Another version is not an option: the clean environment is
sealed. The next import would have failed too:
lm-format-enforcer's Transformers integration imports `PreTrainedTokenizerBase` from `transformers.tokenization_utils`,
which Transformers 5 moved (the upstream script installs a process-local alias for it).

The fix, in the child only:

- **The integration module.** Before vLLM loads, the child puts its own module under the name
  `lmformatenforcer.integrations.vllm` in `sys.modules`; vLLM's `LazyLoader` (`importlib.import_module`) then returns
  it. The script's top level does it when it runs as `__main__` (the child) or `__mp_main__` (an EngineCore started
  with the spawn method re-runs the parent's main script under that name); a forked EngineCore inherits it.
- **The same allowed tokens as the Hugging Face child.** The module's `build_vllm_token_enforcer_tokenizer_data` (vLLM
  calls it with `use_bitmask=True` and the model's vocabulary size) returns the data the Hugging
  Face child builds (`build_transformers_prefix_allowed_tokens_fn` -> `build_token_enforcer_tokenizer_data(tokenizer)`:
  `_build_regular_tokens_list` over `len(tokenizer)` ids, its decoder and EOS), with only the bitmask width set to the
  model's vocabulary (vLLM's bitmask has one bit per model id). lm-format-enforcer's own vLLM builder would list every
  model id: the padding ids past the tokenizer decode to "", so they would sit at the root of its token tree and be
  allowed at every step, which the Hugging Face child never allows. The rest of the path is vLLM's backend and the
  same lm-format-enforcer classes as the Hugging Face child (`JsonSchemaParser(schema)`, `TokenEnforcer`): its first
  call (no generated token) is the root state, as the Hugging Face prefix function's first call (the prompt) is, and
  later tokens decode the current word only, so each step's allowed set is the Hugging Face child's for the same
  generated tokens. The process-local Transformers alias is the upstream script's.
- **A startup check.** Once the engine is up and before the readiness record, one guided request (`PROBE_SCHEMA`: an
  object whose only value is `{"probe": "remote-ra-guided-probe", "items": ["a"]}`, with `uniqueItems`, dropped for
  xgrammar as in every request; greedy; 128 tokens; the stop ids; the prompt "Say hello.") must return exactly that
  value, end at a stop token and use only the tokenizer's ids. Otherwise the child logs `refused: the guided decoding
  (...) failed its startup check: ...` and ends without the readiness record, so the server does not start. A
  backend that fails to build fails the check; one that does not constrain at all answers prose. Startup only:
  nothing is checked per call.
- **Tests** (`test_goal_rewriter_vllm.py`, `GuidedBackendTests`): the fake site has vLLM 0.24's layout (no
  `vllm/transformers_utils/tokenizer.py`, `vllm/tokenizers/mistral.py`), lm-format-enforcer 0.11.3's two integration
  modules (their import blocks and builders) and Transformers 5's moved base class; the fake engine builds its backend
  for the first structured request in a separate interpreter that `multiprocessing.spawn.prepare` sets up as a
  spawned EngineCore; a build error is logged and ends the request without a token, as in the replay. With the fix the backend gets the Hugging Face child's data (137 tokenizer ids, none of the 23
  padding ids that lm-format-enforcer's own builder would allow) in a 160-id bitmask. Mutations: no module, or the
  module in the child only (not in a spawned EngineCore), or no alias: the child refuses to start with the
  integration's error; no startup check: every call fails as in the replay; the data over every model id; an
  unguided reply; a stray id.

## Step 2 (not in this change): overlap

Today an admission builds its Goal first and then runs phase 1, because the Goal query's forward runs inside phase 1
(`private_endpoint.reserve_steps`, `phase1_coordinator.run`). Splitting the Goal forward out of phase 1 lets phase 1
(the prefill of the new tokens) run while the rewriter generates, then the Goal forward and the selection. Larger
overlaps (start the rewriter earlier, or use the previous Goal for the current turn) change the Goal's behaviour and
are the Goal owner's decision. Built later as the opt-in `--goal-overlap` ([docs/goal-overlap/DESIGN.md](../goal-overlap/DESIGN.md)).
