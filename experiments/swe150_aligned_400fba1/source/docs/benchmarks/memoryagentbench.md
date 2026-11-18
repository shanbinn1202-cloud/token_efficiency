# MemoryAgentBench (kv-session protocol)

English | [简体中文](memoryagentbench.zh-CN.md)

> Synced from the release repository (release-repo `5b47394`, its benchmark page) in integrate-1107: the client in `examples/memoryagentbench/` and this page are the release's, with the release-only links pointed at this repository's README and benchmark docs. The configuration of record, the runs and their job ids are in [MEMORYAGENTBENCH_BENCHMARK.md](../MEMORYAGENTBENCH_BENCHMARK.md).

## Scope

- 32 cases (MemAgent32), 8 for each of MemoryAgentBench's four abilities (Accurate Retrieval, Test-Time Learning, Long-Range Understanding, Conflict Resolution), at most 16 at a time (8 for the Gemma profile), thinking off in the commands below (the configuration of record; the current runs, thinking on with an 8,192-token budget: [docs/status.md](../status.md#current-configuration)).
- Policies `observation_query` and `goal_query` (and `random_fill`, the control of the two query policies: its servers are under "Server commands"), and a stock vLLM 0.24.0 control: unmodified `vllm.entrypoints.openai.api_server`, driven by the client's vanilla mode (the whole conversation re-rendered every turn and sent to `/v1/completions` as token ids) instead of the kv-session protocol; Qwen3.6-27B and the Gemma 4 profile.
- Protocol: `--protocol kv-session` ([docs/running.md section 6.6](../running.md#66-memoryagentbench-kv-session-protocol)): the client (method `dram_backed_kv`) renders the template itself and sends only the new tail of each session.
- The harness: the read_next harness in [`examples/memoryagentbench/`](../../examples/memoryagentbench/run_cases.py) ([Harness](#harness)). The official scorer is that of the public MemoryAgentBench repository, `HUST-AI-HYZ/MemoryAgentBench` at commit `538026089d1a8a8eff05121d0db89b388f360eba` (MIT), used from your own checkout.
- Scores: the 8 Long-Range Understanding cases (the InfBench summaries, metric `llm_judge_f1`) need MemoryAgentBench's GPT judge, which this release does not run: they stay `judge_pending`, and the composite is reported without them, as the mean over the other three abilities ([step 5](#procedure)). Scoring is deferred for now: the runs so far report verdicts, not scores ([docs/known-limits.md](../known-limits.md) 5.1).

**GPU runs:** the runs that used these loops, prompts, request construction and cases, with each job's runtime commit and verdict, are in [docs/status.md](../status.md). The release repository's `5b47394` is the export of runtime `5466fc2`, which is `cf2b28d` plus serve.py's speed defaults (each with an off switch) and test fixes.

## Harness

The read_next harness is the design of Shanbin Sun: the agent loop and its prompts, and the MemAgent32 case selection and chunking. `examples/memoryagentbench/` holds it as this project runs it:

- `run_cases.py`: the read_next loop. The history arrives in chunks, one per `bash` call with the command `read_next`, as tool messages headed `[History chunk k/N]`; the question follows the last chunk in the same request; a reply without a tool call before the last chunk is a format error, two invalid turns in a row end the case, the first reply without a tool call after the last chunk is the answer, at most 120 turns. It writes one `cases/<id>/result.json` per case, `requests.jsonl`, `run.json` and `summary.json`.
- `kv_client.py`: the clients. The kv-session client for the Qwen3.6 and Gemma 4 turn grammars, and the stock client (`--mode vanilla`) for the stock vLLM control.
- `cases.py`: the cases. MemAgent32 (`MEMAGENT32`: each case's split file, row and question index in its source data, with its chunk count and context tokens), the chunking rule, a builder from source rows, the loader (every case checked by SHA-256 before use) and synthetic cases.
- `score.py`: the official `post_process` of MemoryAgentBench from your checkout. `upstream/SHA256SUMS` pins the two files used (`utils/eval_other_utils.py`, `utils/templates.py`); they are not copied into this repository.

The chunking rule: a case's context is encoded with the Qwen3.6-27B tokenizer (the runtime's pinned copy under `source-overlay/remote_ra_linux/model-evidence/`, no special tokens added) and cut into consecutive windows of 4,096 tokens, each decoded back to text; the chunks concatenate to the context. The same chunks serve both models. The system prompt is MemoryAgentBench's `SYSTEM_MESSAGE` plus the read_next instruction, and the question is MemoryAgentBench's long-context query template for the case's source, filled in. The bash tool and the tool-message format (`<returncode>` / `<output>`) follow mini-swe-agent (MIT).

The data: MemAgent32's case ids name rows of the public Hugging Face dataset [`cxkjntmngm/MemAgent_Bench`](https://huggingface.co/datasets/cxkjntmngm/MemAgent_Bench) at revision `a785de0f8d2511a43b9e524b5326e1105a4d218a` (split file, row, question index), not of `ai-hyz/MemoryAgentBench`; the sources are `ruler_qa2_validation_65536` (Accurate Retrieval), `icl_trec_coarse_6600shot_balance` (Test-Time Learning), `infbench_sum_eng_shots2` (Long-Range Understanding) and `factconsolidation_sh_32k` (Conflict Resolution, one context shared by its 8 questions). No data of either dataset is distributed with this repository, and `cxkjntmngm/MemAgent_Bench` declares no license: download it yourself (step 1). `cases.py build` reads the split files of your download (Parquet, or JSON Lines; rows with `context`, `questions`, `answers` and `metadata.source`), or the files of a selection of your own (`--select`: a JSON list of `{id, split, source, file, row, question_index}`). `cases.py verify` compares a cases directory with MemAgent32's pinned chunk counts and context tokens.

## Server

The servers' commands are under "Server commands" below (with `--kv-session-thinking off`: the protocol's default is on). The Qwen servers (16 sessions, a 124-GiB arena) may pin up to 127.3 GiB of host memory: their job needs two GH200 memory shares, `--mem=227120M`, as the validated runs requested; the Gemma servers (8 sessions) fit in one, `--mem=113560M` ([host memory](../MEMORYAGENTBENCH_BENCHMARK.md#a4-this-runtime-server-flags-and-vllm-arguments), A.4). A host with about 110 GiB of memory (one GH200 memory share, `--mem=113560M`) therefore cannot run the documented Qwen profile, and this release ships no smaller Qwen profile for MemoryAgentBench (none was validated). Main settings: a session pool (`--max-running-sessions`), `--page-tokens 16 --kv-layout decoupled`, an absolute attention budget (`--active-token-budget`), `--max-tokens 4096`, the model's native context (`--max-model-len 262144`), a pinned page arena (`--dram-arena-gib`) and `--release-idle-pages` (idle sessions' pages leave the GPU under pressure and are readmitted on their next turn). The Qwen servers turn off `--decode-fastpath2` and `--deferred-completion-io`: their GPU runs did not use these options (`--decode-fastpath2` was an opt-in then, `--deferred-completion-io` came later), and the commands keep those runs' flags.

The stock control that ran (Qwen), in the [runtime environment](../install.md#3-runtime-environment) plus the two variables that `serve.py` sets for this runtime and `VLLM_NO_USAGE_STATS=1`; `STOCK_CMD` is the command of [MEMORYAGENTBENCH_BENCHMARK.md A.5](../MEMORYAGENTBENCH_BENCHMARK.md#a5-stock-vllm-control-vanilla); the stock branch of the job script in [B.3](../MEMORYAGENTBENCH_BENCHMARK.md#b3-slurm-job-script) is its lifecycle: started in the background, ready when `/health` answers and `/v1/models` lists `selective-prefix`, stopped with `SIGTERM` to its PID (the `serve.py` dry run, `--start`, READY file and cleanup-receipt checks apply to selective releases only; the stock control uses HTTP readiness and process shutdown):

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V2_MODEL_RUNNER=0 VLLM_NO_USAGE_STATS=1
STOCK_CMD=("$VENV/bin/python" -I -B -m vllm.entrypoints.openai.api_server
           --model "$MODELS/qwen3.6-27b" --served-model-name selective-prefix --host 127.0.0.1 --port 8000
           --distributed-executor-backend uni --tensor-parallel-size 1
           --max-num-seqs 16 --max-model-len 262144 --max-num-batched-tokens 16384
           --kv-cache-memory-bytes 34359738368 --seed 42
           --enable-prefix-caching --prefix-caching-hash-algo sha256
           --mamba-cache-mode align --gdn-prefill-backend triton --language-model-only)
```

A Gemma 4 stock-control server recipe is not included; this page does not cover that path.

## Procedure

**1. The cases**, once, on any Linux machine (x86_64 or aarch64) with CPython 3.12 and network access, in the helper environment: a virtual environment of its own (`MAB_VENV`), never the base environment, which stays unchanged ([docs/install.md section 2.1](../install.md#21-base-environment)). `examples/memoryagentbench/requirements.lock` pins, with the SHA-256 of every file, the helper's packages for CPython 3.12 on Linux x86_64 and aarch64: `transformers` 5.13.0 and `tokenizers` 0.22.2 (the release's), `numpy` 2.3.5 and `tiktoken` 0.14.0 (the base environment's versions), `pyarrow` 26.0.0 (the Parquet split files), `nltk` 3.10.4, `rouge-score` 0.1.2 (no wheel on PyPI: pip builds it from its hashed sdist; the build requirements that pip fetches for that build are not locked) and `editdistance` 0.8.1 (the scorer's imports), and their dependencies. The versions beyond the release's and the base environment's are pinned PyPI releases; the GPU runs' scoring environment was not recorded. Synthetic cases (for a smoke run without the dataset: the answer is a code word in the history) need nothing else. For MemAgent32, `ROWS` is your download of the source dataset at the pinned revision and `MAB` a MemoryAgentBench checkout at the pinned commit. The downloads need the network: the `unset` line undoes the [runtime environment](../install.md#3-runtime-environment)'s offline switches, and the last line sets them again:

```sh
MAB_VENV=/abs/path/to/mab-tools       # new: the "venv" line creates it
ROWS=/abs/path/to/MemAgent_Bench      # new: the "hf download" line creates it
MAB=/abs/path/to/MemoryAgentBench     # new: the "git clone" line creates it
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
python3.12 -m venv "$MAB_VENV"
"$MAB_VENV/bin/python" -I -B -m pip --isolated install --require-hashes -r examples/memoryagentbench/requirements.lock
"$MAB_VENV/bin/hf" download cxkjntmngm/MemAgent_Bench --repo-type dataset \
  --revision a785de0f8d2511a43b9e524b5326e1105a4d218a --local-dir "$ROWS"
git clone https://github.com/HUST-AI-HYZ/MemoryAgentBench "$MAB"
git -C "$MAB" checkout 538026089d1a8a8eff05121d0db89b388f360eba
"$MAB_VENV/bin/python" -I -B examples/memoryagentbench/cases.py synthetic --out "$RUNS/mab-synthetic"
"$MAB_VENV/bin/python" -I -B examples/memoryagentbench/cases.py build --rows "$ROWS" --mab "$MAB" \
  --out "$RUNS/mab-cases"
"$MAB_VENV/bin/python" -I -B examples/memoryagentbench/cases.py verify --cases "$RUNS/mab-cases"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

`build` ends with the summary of `verify`: `counts_match` is 32 when every case has MemAgent32's chunk count and context tokens.

**2. The server:** start it ("Server commands" below) and wait for READY. The stock control instead: its `STOCK_CMD` ([Server](#server)), started in the background, then wait until `/health` answers and `/v1/models` lists `selective-prefix` (A.5's readiness; the stock branch of [B.3](../MEMORYAGENTBENCH_BENCHMARK.md#b3-slurm-job-script)).

**3. The cases against the server**, on the same node, in the runtime environment (its `python` has Transformers 5.13): 16 at a time for Qwen (`--workers 16`), 8 for the Gemma profile. The client reads the server's budget, thinking setting and turn grammar from `GET /v1/kv-session/info` and refuses to start when they differ from its own (`--thinking`, default `off`; `--grammar`). The client's output directory, `CLIENT_RUN`, is new for each model, policy and attempt (`POLICY` is the server's `--policy`, `LABEL` the attempt's label, as in the server's `--output`); step 5 scores that same directory. Set `CLIENT_RUN` with the line of the server's model, then run that model's client. Qwen:

```sh
POLICY=observation_query      # the server's --policy: observation_query, goal_query or random_fill
CLIENT_RUN="$RUNS/mab-$POLICY-$LABEL-client"
python -I -B examples/memoryagentbench/run_cases.py --base-url http://127.0.0.1:8000 \
  --cases "$RUNS/mab-cases" --tokenizer "$MODELS/qwen3.6-27b" --workers 16 --out "$CLIENT_RUN"
```

Gemma:

```sh
POLICY=observation_query      # the server's --policy: observation_query, goal_query or random_fill
CLIENT_RUN="$RUNS/mab-gemma-$POLICY-$LABEL-client"
python -I -B examples/memoryagentbench/run_cases.py --base-url http://127.0.0.1:8000 \
  --cases "$RUNS/mab-cases" --tokenizer "$MODELS/gemma-4-31b-it" --grammar gemma4 --workers 8 --out "$CLIENT_RUN"
```

The stock control (`--mode vanilla`, Qwen):

```sh
CLIENT_RUN="$RUNS/mab-stock-$LABEL-client"
python -I -B examples/memoryagentbench/run_cases.py --base-url http://127.0.0.1:8000 --mode vanilla \
  --cases "$RUNS/mab-cases" --tokenizer "$MODELS/qwen3.6-27b" --workers 16 --out "$CLIENT_RUN"
```

It prints one line with the outcome (`COMPLETED`) and the count of each case status: `answered`, `answer_length_limit`, `unfinished_thinking`, `format_error`, `turn_limit`, or `error` (the loop's own status, for example `context_limit`, is the case's `runner_status`). Cases run largest first by default (`--order largest-first`); `--ids` runs a subset.

**4. Stop** the server with `SIGTERM` and check the cleanup receipts ([docs/running.md section 4.1](../running.md#41-one-server)). The stock control writes no receipts: `SIGTERM` to its PID and its exit status (the stock branch of [MEMORYAGENTBENCH_BENCHMARK.md B.3](../MEMORYAGENTBENCH_BENCHMARK.md#b3-slurm-job-script)).

**5. The score**, offline, with the official `post_process`, in the helper environment, of the run in `CLIENT_RUN` (step 3). The checkout's imports are numpy, nltk, tiktoken, rouge_score and editdistance (all in the helper environment); `--stub-unused` replaces the last three, which these sources do not need for their scores:

```sh
"$MAB_VENV/bin/python" -I -B examples/memoryagentbench/score.py --run "$CLIENT_RUN" \
  --cases "$RUNS/mab-cases" --mab "$MAB" --stub-unused
```

It writes each case's metrics and score into its `result.json` and the per-ability means into `accuracy.json`. The 8 Long-Range Understanding cases (the InfBench summaries, metric `llm_judge_f1`) need MemoryAgentBench's GPT judge (an external API service), which this release does not run: `score.py` leaves them `judge_pending` with no score. `accuracy.json` reports the composite without them: `composite_score_without_judge` is the mean over the other three abilities (Accurate Retrieval, Test-Time Learning, Conflict Resolution), once every planned case has run and completed without an infrastructure error (status `error`; `infrastructure_errors` lists them) and each of those three is fully scored; `judge_abilities` names the ability left out. `composite_score`, the mean over all four abilities, stays null while judge cases are pending.

## Server commands

The dry runs of this benchmark's servers, one per policy (add `--start` to serve). `LABEL` is a new label per attempt; the release repository's `tests/test_docs_commands.py` dry-runs these commands.

### Qwen3.6-27B

```sh
SHARED=(--protocol kv-session --kv-session-thinking off --max-running-sessions 16 --kv-cache-gib 32
        --page-tokens 16 --kv-layout decoupled --max-model-len 262144 --max-tokens 4096 --dram-arena-gib 124
        --connector-pool-gib 1 --cuda-graphs full --decode-fastpath --no-decode-fastpath2
        --no-deferred-completion-io --language-model-only --release-idle-pages)
python -B serve.py --release-dir "$RELEASE" --policy observation_query "${SHARED[@]}" \
  --active-token-budget 32768 --output "$RUNS/mab-observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --policy goal_query "${SHARED[@]}" \
  --active-token-budget 32768 --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --admission-overlap 16 \
  --output "$RUNS/mab-goal_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE" --policy random_fill "${SHARED[@]}" \
  --active-token-budget 32768 --output "$RUNS/mab-random_fill-$LABEL" --port 8000
```

### Gemma 4 31B IT

```sh
SHARED=(--protocol kv-session --kv-session-thinking off --max-running-sessions 8 --kv-cache-gib 25
        --page-tokens 16 --kv-layout decoupled --max-model-len 262144 --max-tokens 4096 --dram-arena-gib 60
        --connector-pool-gib 1 --cuda-graphs full --decode-fastpath --decode-fastpath2 --decode-fastpath3
        --deferred-completion-io --language-model-only --release-idle-pages)
python -B serve.py --release-dir "$RELEASE-gemma4" --policy observation_query "${SHARED[@]}" \
  --active-token-budget 32768 --output "$RUNS/mab-gemma-observation_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE-gemma4" --policy goal_query "${SHARED[@]}" \
  --active-token-budget 32768 --goal-rewriter-model "$MODELS/Qwen3-0.6B-c1899de" --admission-overlap 8 \
  --goal-rewriter-batch 4 --output "$RUNS/mab-gemma-goal_query-$LABEL" --port 8000
python -B serve.py --release-dir "$RELEASE-gemma4" --policy random_fill "${SHARED[@]}" \
  --active-token-budget 32768 --output "$RUNS/mab-gemma-random_fill-$LABEL" --port 8000
```
