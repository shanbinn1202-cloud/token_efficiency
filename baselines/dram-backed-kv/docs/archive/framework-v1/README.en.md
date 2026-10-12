# DRAM-Backed Goal-Conditioned KV Working Set

> Archived framework/recorded-replay README. See the repository root for the current Online Selector V2 entry point.

This repository is a clean-room Stage 0 foundation for experiments in which
complete sealed KV history is backed by CPU DRAM and only a Goal-conditioned
working set is resident in GPU HBM.

The framework deliberately separates three concerns:

```text
TurnContext -> GoalConstructor -> GoalRecord
                                   |
PageCatalog -----------------> Selector -> SelectionPlan
                                              |
                                       RuntimeBackend
```

Goal Constructors and Selectors never manipulate GPU tensors. Runtime
Backends consume a saved `SelectionPlan` and must not embed a retrieval policy.
This makes the same plan replayable in Shadow, semantic replay, and future
DRAM/HBM runtimes.

## Implemented

- Versioned `TurnContext`, `GoalRecord`, `PageCatalog`, and `SelectionPlan`.
- Stable logical Page construction and original-position validation.
- Cross-turn added/retained/removed/reactivated Page accounting.
- Raw-objective and transparent heuristic Goal baselines.
- Session-scoped heuristic Goal maintenance with immutable root anchors,
  deterministic features, phase hysteresis, bounded stuck statistics, and an
  auditable KEEP/PATCH/REWRITE gate.
- Conditional, schema-validated retrieval-Goal rewriting through a deterministic
  fake or a dependency-free command adapter; root/plan-shaped output is rejected
  and failures take a deterministic fallback.
- Goal-aware lexical/Page-metadata selection whose ranking changes when Goal
  entities change.
- Recent-only and offline Oracle Selectors.
- Typed Shadow JSONL traces plus decision/update/generation/cost/root-invariance
  metrics and macro/micro Evidence Recall@Budget.
- JSONL artifacts and a small CLI smoke runner.
- Invariants preventing current/future Page selection and reordered Pages.
- A compatibility adapter that validates new framework records before invoking
  the previously validated Qwen Dense/Reset/Cumulative numerical replay engine.
- A framework-neutral DRAM KV runtime with transactional sealed-Page offload,
  full/selected restore, resident reuse, reactivation, and per-session transfer
  and residency accounting.
- A duck-typed Qwen hybrid-cache adapter that Pages only full-attention K/V,
  preserves exact logical positions, and leaves recurrent/conv state untouched.
- A capability-checked Qwen pre-prefill bridge with separate physical mask/cache
  positions and logical RoPE positions.
- A dependency-free recorded turn loop with stable rendered-token boundaries,
  Page/message provenance, Oracle budget plans, and post-generation sealing.
- A Selector V2 control-plane path with immutable `PageVectorPublisher` /
  `PageVectorIndex` records, direct 16-token physical-Page cosine and hybrid
  ranking, strict vector provenance, deterministic lexical fallback, and a CPU
  hidden-state publisher test double for Shadow integration.
- An offline real-model trajectory extractor that hooks one decoder layer,
  pools/projects Page vectors on GPU, and exports only compact int8 vectors for
  CPU proxy replay. It is teacher-forced and not selection-conditioned.
- A recorded H200 enforcement path in the cumulative runner: V2 plans control
  actual DRAM restore/bind and selection-conditioned Qwen forwards, including
  an HBM-resident live tail for physical Pages crossing turn boundaries.

Not yet implemented: a production serving loop, independent continuous runtime
branches for each experiment policy, block-native attention that avoids
contiguous-cache assembly, or the production GPU/model-runner Page-vector
publisher. Selector V2's CPU/Shadow path establishes control-flow wiring only;
16-token retrieval quality, H2D/HBM behavior, delta-NLL, top-1 agreement, and
selection-conditioned representation drift remain unvalidated.

## Run tests

No third-party dependency is required:

```bash
PYTHONPATH=src python3.12 -m unittest discover -s tests -v
```

Run the Shadow smoke example:

```bash
PYTHONPATH=src python3.12 -m dram_kv.cli \
  --input examples/shadow_input.json \
  --output /tmp/dram-kv-shadow.jsonl \
  --goal heuristic \
  --page-size 128 \
  --bytes-per-token 0
```

## Goal Constructor Shadow experiment

The first offline loop is intentionally separate from DRAM/Qwen execution:

```text
latest action + latest observation
       -> deterministic bounded features
       -> phase/stuck detector -> advisory working-set policy
       -> KEEP / deterministic PATCH / conditional REWRITE
       -> lexical + Page-metadata Selector
       -> SelectionPlan -> ShadowBackend -> Recall@Budget
```

`root_goal` is frozen on the first boundary for each session. A later change to
the task objective, instance identity, or turn ordering fails closed. The SLM
sees only the immutable root, current retrieval Goal, latest bounded action and
observation, extracted features, trigger, and compact counters. It cannot emit
`root_goal`, an execution plan, commands, steps, or arbitrary fields.

Run all four configurations over the checked-in synthetic replay:

```bash
PYTHONPATH=src python3.12 -m dram_kv.cli \
  --input examples/goal_shadow_replay.json \
  --output /tmp/dram-kv-goal-shadow \
  --shadow-config all
```

This writes:

```text
/tmp/dram-kv-goal-shadow/
  raw_objective.jsonl
  stateless_heuristic.jsonl
  heuristic_only.jsonl
  heuristic_slm.jsonl
  comparison.json
```

The four configurations share the exact Page catalog, labels, Page budget, and
selector:

| configuration | Goal behavior | model calls |
|---|---|---|
| `raw_objective` | raw task objective every boundary | none |
| `stateless_heuristic` | existing per-turn regex reconstruction | none |
| `heuristic_only` | frozen root + stateful heuristic maintenance | none; requested rewrites use audited deterministic PATCH/KEEP fallback |
| `heuristic_slm` | same gate plus conditional rewrite | deterministic fake by default; optional one-time init and hard-trigger rewrites only |

Every JSONL file contains one self-contained `shadow_turn` envelope per boundary
and a final `run_summary`. The turn envelope includes bounded context hashes and
excerpts, complete Goal/feature/phase/gate/rewriter state, all selector hashes,
the complete `SelectionPlan`, and Shadow evidence metrics. The summary fixes the
following denominators:

- `maintenance_boundaries`: all turns except the first turn of each session;
- `effective_retrieval_goal_update_rate`: retrieval-goal hash changes divided by
  maintenance boundaries; phase-policy-only changes are reported separately;
- `zero_retrieval_goal_change_rate` and `keep_invariance`: the complementary
  fast-path rate and an explicit check that KEEP never changed retrieval state;
- `slm.maintenance.generation_rate`: actual rewrite attempts divided by
  maintenance boundaries; initialization calls are separate;
- Recall@Budget macro/micro: only boundaries with non-empty evidence labels;
- root invariance: exact per-session root hash plus the constructor diagnostic.

Useful fields in `comparison.json` are:

```text
summaries.<config>.decision_counts
summaries.<config>.trigger_counts
summaries.<config>.effective_retrieval_goal_update_rate
summaries.<config>.slm.{initialization,maintenance}
summaries.<config>.latency_ms
summaries.<config>.token_cost_totals
summaries.<config>.root_invariance
summaries.<config>.shadow_evidence_recall_at_budget
```

The fake rewriter is deterministic and makes tests/replays network-free. The
recommended real `<1B` path is the persistent NDJSON mode: the model loads once,
startup and request timeouts are separate, and `lm-format-enforcer` applies a
JSON Schema token mask during generation. The core strict validator still runs
after generation and every failure retains the deterministic fallback. Torch,
Transformers, and LM Format Enforcer remain optional script-only dependencies:

```bash
PYTHONPATH=src python3.12 -m dram_kv.cli \
  --input examples/goal_shadow_replay.json \
  --output /tmp/qwen-goal-shadow.jsonl \
  --shadow-config heuristic_slm \
  --slm-command 'python3 -u scripts/qwen_goal_rewriter.py --model /path/to/Qwen3-0.6B --local-files-only --serve' \
  --slm-persistent --slm-startup-timeout-seconds 600
```

The Qwen3 chat template explicitly sets `enable_thinking=False`; the same
adapter supports Qwen2.5-0.5B. Non-persistent mode remains compatible with the
original one-shot protocol. A one-shot command may return either a retrieval-Goal JSON object or
`{"candidate": {...}, "input_tokens": N, "output_tokens": N,
"model_cost": X}`. It must not return `root_goal` or execution-plan fields.
Use `--no-slm-init` to isolate maintenance rewrite cost from the optional
one-time initializer call.

Use `scripts/benchmark_goal_rewriter.py` to run both checkpoints against an
identical JSON/JSONL RewriteRequest set, for example
`examples/qwen_rewrite_request.json`, without exposing a trajectory to either
model.

### Real recorded SWE-agent replay (astropy-14995)

The CLI also accepts the supplied trajectory/Oracle zip directly. It reads the
48 real action/observation boundaries and uses the Oracle only as offline
older-evidence labels. The first CPU experiment uses one semantic completed
turn per Page, matching the annotation unit; it is not token-level KV replay.

```bash
PYTHONPATH=src python3.12 -m dram_kv.cli \
  --input /path/to/astropy__astropy-14995_trajectory_oracle_selection_2026-09-02.zip \
  --output /tmp/astropy-14995-goal-shadow \
  --shadow-config all \
  --calibrate-thresholds
```

`calibration.json` scans 12 auditable configurations. On this one trajectory,
the selected candidate was `entity_drift_steps=3`, two-step hysteresis, and the
default stuck thresholds: micro Recall@Budget `0.459`, update rate `0.787`, and
rewrite rate `0.170`. This is per-instance calibration only; Oracle provides
retrieval labels, not ground-truth gate-boundary labels.

Optional EWMA/CUSUM plumbing can be exercised without a model using
`--embedding-drift hashing`. It is opt-in because the first untuned run added
seven semantic-drift triggers and did not improve Recall. A real embedding
backend implements `TextEmbedder` or uses the JSON `CommandTextEmbedder` seam.

The checked-in `scripts/qwen_goal_rewriter.py` is an optional Transformers
adapter. On H200, local Qwen2.5-0.5B loaded successfully. In the final 48-turn
replay, the gate called it at 11/47 maintenance boundaries (23.4%); strict
validation accepted 6/11 outputs (54.5%) and the other five safely fell back
(four produced no JSON and one repeated a supposedly unique entity). SLM
latency was 19.8 s p50 / 35.3 s p95 because the legacy command adapter
cold-started per call. Root invariance remained intact. This is a compatibility
baseline for the old one-shot path; the new persistent/schema-constrained path
has CPU protocol coverage but does not yet have H200 effectiveness results.

Recorded tokenizer spans now propagate `files`, `symbols`, `tests`, and
`errors` into Page metadata. The cumulative Qwen/DRAM runner keeps its Oracle
upper-bound default and adds an online Goal branch:

```bash
python scripts/run_cumulative_oracle_sweep.py \
  --model /path/to/Qwen3.6-27B \
  --bundle /path/to/trajectory_oracle.zip \
  --selection-policy goal --ratios 0.40 0.60 \
  --output /tmp/cumulative-goal.json
```

To enable the same persistent rewriter in the H200 cumulative Goal branch, add
`--slm-command 'python3 -u scripts/qwen_goal_rewriter.py --model /path/to/Qwen3-0.6B --local-files-only --serve' --slm-persistent`.

The Goal path never consumes Oracle selections; the annotation remains present
only so the same runner can retain the `oracle` comparison mode.

The first H200 token-level run completed all 48 turns for full-active and one
Goal-conditioned 40% branch (96 ordered artifacts). The Goal branch averaged
39.98% active tokens and reduced final HBM residency from 2.487 GB to 0.995 GB.
It also exposed the current quality/churn cost: 125 reactivations, mean top-1
agreement 0.9370, and mean delta-NLL +0.1574 versus full-active. Treat this as a
working integration and diagnostic baseline, not a quality win.

### Schema migration notes

`SCHEMA_VERSION` is now `1.2.0`. `GoalRecord` retains its historical positional
fields and mirrors them into a canonical nested `RetrievalGoal`, so existing
runtime/replay code remains source compatible. New records additionally carry
the root anchor, requested/applied gate decision, triggers, changed fields,
working-set policy, SLM/fallback telemetry, and bounded diagnostics. The
selector signal hash is canonical JSON over every field the selector may use:
root, full retrieval Goal, and policy hint. Gate decisions, latency, counters,
and cost do not perturb the hash. Version-1.0 Goal artifacts without nested
fields remain readable.

### Remaining experiments

1. Repeat calibration on many recorded tasks and add human gate-boundary labels.
2. Tune embedding drift on those labels and replace hashing with a real optional
   embedding command/backend.
3. Run Qwen2.5-0.5B and Qwen3-0.6B on one frozen RewriteRequest set to validate
   persistent latency and schema-constrained valid-output rate.
4. Run and compare cumulative Goal branches on H200 and authoritative GH200;
   report numerical quality, H2D churn, HBM, and latency against Oracle/full-active.

The checked-in synthetic and semantic-Page Shadow loops are CPU-only; only the
optional real-SLM and token-level cumulative validations require GPU time.

## Next stages

1. Compare Goal and Selector implementations in Shadow mode against the same
   message/Page evidence labels.
2. Run the strongest online selector cumulatively on multiple trajectories.
3. Integrate block-native attention to remove contiguous-cache assembly copies.
4. Add retention slack/hysteresis to reduce H2D reactivation churn.
5. Extend allocator validation from one session to concurrent serving.

## Legacy drift regression

`scripts/run_legacy_regression.py` migrates the recorded trajectory and Oracle
selection through the new contracts. Budgeting and controlled reactivation are
applied before model execution; the resulting `SelectionPlan` Page IDs are then
used to reconstruct the plan consumed by the numerical engine.

On DeltaAI, submit the complete 13579 compatibility regression with:

```bash
sbatch scripts/run_13579_regression_gh200.sbatch
```

The launcher requests one GH200 for 30 minutes and runs:

- the nine-turn 100% active invariant;
- the full 40% active-budget replay;
- the full Oracle replay with turn-30 reactivated Pages omitted.

`scripts/run_13579_restored_gh200.sbatch` supplies the paired restored-Oracle
baseline for the turn-30 causal comparison.

This backend establishes compatibility with the prior semantic experiment. It
does **not** establish CPU DRAM transfer, physical HBM block release, resident
reuse, or serving throughput; those require the native DRAM/HBM backend.

## Native DRAM runtime foundation

`DramKVRuntime` owns immutable sealed CPU copies and a per-session active GPU
working set. `DramRuntimeBackend.execute()` synchronously restores the
`SelectionPlan` before returning, so model integration follows this order:

```python
result = pipeline.run_turn(context, catalog)  # Goal -> select -> restore
active_kv = dram_backend.prepared_payloads(
    context.session_id,
    context.turn_id,
    preparation_id=result.backend_metrics["preparation_id"],
)
model.prefill(live_suffix, past_key_values=active_kv)
# after generation and Page sealing:
dram_backend.seal_after_generation(session_id, page_id, new_gpu_kv)
```

The runtime currently provides:

- optional pinned CPU backing through a tensor adapter;
- selected and full restore in original logical Page order;
- active-set delta reuse, inactive release, and later reactivation;
- session isolation and transactional state updates on transfer failures;
- DRAM bytes, active HBM bytes, H2D/D2H bytes, counts, and latency metrics.

`release_count` means the runtime has dropped its GPU reference. The caller
must also release its own reference; a model-specific GPU test must verify the
allocator's `memory_allocated`/`memory_reserved` behavior. The generic runtime
does not itself manage recurrent state or prove physical HBM savings.

## Qwen hybrid-cache integration

`QwenKVPageAdapter` stores one Page as the K/V pair for every initialized
full-attention layer. `extract_page()` must run while the source span still has
a contiguous physical-to-logical mapping. After selected restore,
`assemble_pages()` preserves the selected Pages' original logical positions and
`write_active_cache()` returns those positions. The caller must pass them to
the model's RoPE/cache-position path during the following prefill.

Linear-attention recurrent and convolution state is intentionally excluded
from Page payloads and is never overwritten by the adapter. The current
Transformers bridge concatenates restored Pages into contiguous tensors; this
can temporarily coexist with Page tensors and create an assembly peak. A
block-native attention backend is required to remove that copy.

The GH200 allocator validation can be submitted with:

```bash
sbatch scripts/run_qwen_dram_runtime_gh200.sbatch
```

It instantiates the model configuration's real hybrid `DynamicCache` and exact
K/V shapes without loading model weights, then measures dense residency,
DRAM-only backing, full restore, selected restore, reactivation, contiguous
assembly peaks, and final release with CUDA allocator counters. It complements,
rather than replaces, the weight-loaded logit regression above.

The completed GH200 run and allocator table are recorded in
`results/qwen_dram_runtime_gh200_2026-08-31.md`.

## Pre-prefill model bridge

`QwenForwardBridge` makes restore and cache installation a capability-checked
boundary before formal model prefill. It keeps compact physical positions for
the causal mask and cache update while sending the suffix's original logical
positions only to the Qwen full-attention RoPE hook. A superseded preparation,
wrong turn, reordered Page, incorrect logical span, or suffix/history boundary
fails before model execution.

Submit the weight-loaded Dense/full/selected validation with:

```bash
sbatch scripts/run_qwen_forward_bridge_gh200.sbatch
```

The completed Qwen 27B GH200 comparison is documented in
`results/qwen_forward_bridge_gh200_2026-09-01.md`. The 100% restored branch
matched Dense exactly in logits, full-attention K/V, and all recurrent/conv
state tensors. This establishes the first native end-to-end pre-prefill bridge;
it does not yet provide a serving loop or remove contiguous assembly copies.

## Recorded turn loop and Oracle budget pilot

`RecordedTurnLoop` enforces `plan -> restore/bind -> model -> seal` over fixed
rendered tokens. Oracle evidence is mapped through explicit source message IDs,
so an annotation's completed semantic turn includes both its action and later
observation without leaking the current live observation into sealed history.
Mandatory overflow is a hard error; omitted recent/evidence Pages, capacity
fillers, and unused capacity are saved with every plan.

Run the Qwen 27B pilot with:

```bash
# Historical one-off Slurm wrapper removed; the underlying Python tool remains.
```

The completed 48-turn native DRAM replay and six-turn 25%/40%/60% Oracle sweep
are documented in `results/oracle_budget_sweep_gh200_2026-09-01.md`.

## Cumulative independent-runtime experiment

`scripts/run_cumulative_oracle_sweep.py` advances full-active, 40%, and 60%
branches through independent DRAM/HBM lifecycles and continuous recurrent
state. New Pages are sealed from the logically contiguous suffix of each
branch's sparse physical cache and marked selection-conditioned after the
switch.

The completed GH200 experiment is documented in
`results/cumulative_oracle_sweep_gh200_2026-09-02.md`. It confirms the expected
HBM reduction, while also showing that naive selected-set churn raises H2D
traffic by 26.4% at 40% and 16.1% at 60% relative to full-active.
