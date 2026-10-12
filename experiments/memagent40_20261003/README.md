# MemAgent_Bench: fixed 40-question pilot

Results and frozen configuration:
`/u/ssun4/token_efficiency_results/memagent40_20261003`.

## Scope

- Requested data: `cxkjntmngm/MemAgent_Bench`, revision
  `a785de0f8d2511a43b9e524b5326e1105a4d218a`.
- Seven existing adapters: Vanilla, Random Attention, Sliding Window FA3,
  AgentDiet, AgentKV, RepairKV, IntentKV GQA.
- Each method runs the same 40 independent questions at 16K, 12K and 8K:
  21 runs, 840 trajectories. Vanilla and AgentDiet retain their existing
  behavior; their budget labels do not add a new eviction policy.
- Four skills have 10 questions each. Retrieval uses RULER QA1/QA2;
  learning uses the five ICL classification tasks; conflict resolution
  uses SH/MH at 32K/64K source tiers; understanding uses book summaries.
- Selection uses seed 42 without model outputs or answer-based filtering.
  Raw input eligibility is at most 196,608 native Qwen tokens, reserving
  space below the native 262,144 limit for generated turns and wrappers.
  This is a pilot subset, not the complete benchmark or every source tier.
  No selected history is truncated. `manifest.json` records every original
  file, row, question index, context hash and excluded candidate.

## Protocol

The reference works are the [official MemoryAgentBench code](https://github.com/HUST-AI-HYZ/MemoryAgentBench)
and [Mem-alpha](https://github.com/wangyu-ustc/Mem-alpha). They ingest history
incrementally before asking questions. This harness adapts that flow to the
existing tool-based compression methods:

1. Start an independent session without revealing the final question.
2. The model calls the existing `bash` tool schema with `read_next`.
   This deterministic environment returns the next history block. It does
   not execute arbitrary shell commands, search, skip or reread the history.
3. Blocks follow the original order and concatenate byte-for-byte to the
   original history. Boundaries use approximately 4,096 native Qwen tokens.
4. After the last block, append the official source-specific query template.
5. Score only the final answer, never an unfinished thinking trace.

Each QA replays its history from a fresh session. This prevents previous
questions and answers from leaking into later questions, while avoiding a
new cache-fork implementation inside any method. It differs from the
official optimization of ingesting once and querying a memory snapshot
multiple times; ingestion costs are included for every QA in all methods.

AgentDiet uses its existing CoACT hook after tool results. All other methods
use the existing backends unchanged. The harness does not force eviction,
change selection scores or impose a minimum generated length to trigger a
method. The model's actual generated turns determine native trigger events.

## Shared runtime

The confirmed configuration is native 262,144 context and **18 GiB** KV
memory, BF16 Qwen3.6-27B, one GPU, 8 asynchronous requests, native FCFS,
temperature 0, seed 0, thinking enabled, and 4,096 maximum output tokens per
request. Other method options come from the latest SWE configuration.

The initially requested 12 GiB was tested and rejected by native vLLM:
262,144 context requires 16.32 GiB, with a 191,296-token estimate at 12 GiB.
The user then selected 18 GiB. No capacity check, scheduler or algorithm was
modified to bypass that failure. `algorithm_sources.json` freezes the
existing implementation files; every stage verifies their hashes.

## Metrics and judging

- RULER QA and conflict resolution: official substring exact match.
- ICL classification: official exact match and original post-processing.
- Book summaries: official GPT-4o fluency-weighted precision/recall F1.
  Ordinary token overlap or ROUGE is not substituted for this score.
- Report each skill separately. Summary F1 is not a binary accuracy.

The older requested data lacks summary keypoints. `judge.py --prepare-only`
matches all ten selected histories and reference summaries exactly against
official scoring metadata from `ai-hyz/MemoryAgentBench`, revision
`7ea066982b140a19337e17e60d45d4076e042faf`. Only the keypoints are imported;
inference still uses the user-specified data.

To complete GPT-4o scoring after inference, configure `OPENAI_API_KEY` and,
if needed, `OPENAI_BASE_URL` outside chat. `MEMAGENT_JUDGE_MODEL` can name the
available GPT-4o deployment; the default matches the reference code,
`gpt-4o-2024-05-13`. Run `judge.py --bundle <results-directory>`. Judge prompts
and the F1 formula are imported directly from the official repository.
The script also reads `<results-directory>/judge.env`, or a file passed with
`--env-file`. Keys are never written to benchmark logs.

Output throughput is the total main-agent generated tokens, including
thinking and tool calls, divided by batch wall time. Wall time includes
history ingestion, policy work, auxiliary AgentDiet calls and the final
answer. Auxiliary generated tokens are recorded separately and excluded
from the main throughput numerator. Model startup, initial compilation,
warmup and grading are excluded. Independent completed QAs can be resumed
between allocation segments; segment wall times are summed.

## Execution

- `prepare.py`: freeze sampling, full histories and question templates.
- `run.py`: concurrent independent QA trajectories and native scoring.
- `stage.py`: source verification, server lifecycle, transport smoke, run.
- `supervise.py`: sequential runs, allocation renewal and durable progress.
- `judge.py`: matched official references and deferred GPT-4o grading.
- `test_protocol.py`: delayed question, full history, isolation and metric
  checks. All three tests passed before formal inference.

Supervision never reallocates a GPU before one hour of use. It lets active
questions finish and stops dispatching new questions before allocation
expiry. Failures that need inspection stop new stage dispatch. Artifacts
include per-turn responses, usage, policy statistics, per-case progress,
server logs and source hashes.
