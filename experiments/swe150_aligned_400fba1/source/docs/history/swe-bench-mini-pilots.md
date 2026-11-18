# SWE-bench pilots: agent host, tasks and evaluation (frozen)

English | [简体中文](swe-bench-mini-pilots.zh-CN.md)

> **Frozen on 2026-11-16** (at `da0058e`): the bodies of sections 7.2, 7.3 and 7.5 of [docs/benchmarks/swe-bench-mini-swe-agent.md](../benchmarks/swe-bench-mini-swe-agent.md), the setup of the multi-session SWE-bench pilots of 2026-09/10 (jobs 3299922 and 3300434), moved here by docs-leftover-1116 unchanged except for their relative links; not updated. Section numbers refer to that page ("the versions above" are those of its section 7.1). Today's agent host, tasks and evaluation are in [swe-bench.md](../benchmarks/swe-bench.md) ([Agent host setup](../benchmarks/swe-bench.md#agent-host-setup), [Tasks](../benchmarks/swe-bench.md#tasks), [Evaluation](../benchmarks/swe-bench.md#evaluation)); job results: [docs/status.md](../status.md).

## From section 7.2: Agent host

**The pilots' setup (2026-09/10), as a record.** Today's scripts refuse parts of it: `images.py` checks every instance image against its pinned digest (no plain `docker pull ...:latest`), `evaluate.sh` needs the SWE-bench evaluator at its pinned commit and evaluates the pinned test split as a local file (not `swebench==5.0.2` from PyPI, not `--dataset_name SWE-bench/SWE-bench_Verified`), and `build_tasks.py` reads the dataset at its pinned revision. For today's agent host, tasks and evaluation see [swe-bench.md](../benchmarks/swe-bench.md): [Agent host setup](../benchmarks/swe-bench.md#agent-host-setup), [Tasks](../benchmarks/swe-bench.md#tasks) and [Evaluation](../benchmarks/swe-bench.md#evaluation).

The agent host needs Docker, `curl`, `ss`, `python3`, an agent environment and an evaluator environment. The pilots' environments had the versions above (of which the evidence records Mini-SWE-Agent's only); the commands that created them were not recorded, so this is **not part of the validated runs**:

```sh
MINI=/abs/path/to/mini-env
EVAL=/abs/path/to/eval-env
uv venv --python 3.12 "$MINI"
uv pip install --python "$MINI/bin/python" mini-swe-agent==2.4.2 litellm==1.103.1 datasets==5.0.1
uv venv --python 3.12 "$EVAL"
uv pip install --python "$EVAL/bin/python" swebench==5.0.2 datasets==5.0.1
```

The agents ran offline against caches filled beforehand: the Hugging Face dataset `princeton-nlp/SWE-Bench_Verified` (runner and task builder) and one Docker image per instance, `docker.io/swebench/sweb.eval.x86_64.<instance id with "__" replaced by "_1776_", lower case>:latest`. The evaluator loads `SWE-bench/SWE-bench_Verified` (identical statements for all ten pilot instances) and ran without the offline variables. Fill the caches once with network access, for example with `load_dataset(...)` and `docker pull` (not part of the validated runs).

## From section 7.3: Tasks

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

## From section 7.5: Evaluate

After the agents, per policy, on the agent host (all group outputs of that policy in one call). `EVAL` here is the evaluator environment of the swe-bench page's [Agent host setup](../benchmarks/swe-bench.md#agent-host-setup) (SWE-bench at commit `02e7a74`): `evaluate.sh` refuses an evaluator that cannot read a local Parquet dataset, and the PyPI `swebench==5.0.2` of the record in section 7.2 was not checked for that.

```sh
EVAL_PYTHON="$EVAL/bin/python" examples/swebench/evaluate.sh my-run-goal_query goal_query \
  /abs/path/to/eval/goal_query /abs/path/to/runs/goal_query-g0 /abs/path/to/runs/goal_query-g1
```

It writes one predictions file with every planned instance (an instance without a prediction gets an empty patch and is reported missing), runs the evaluator (in the pilots: `python -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Verified --split test --predictions_path … --max_workers 2 --timeout 1800 --run_id …`; today on the pinned test split as a local Parquet file, [Evaluation](../benchmarks/swe-bench.md#evaluation)) in the new evaluation directory, and prints one line per instance from `logs/evaluation/<run_id>/results.json`. Trajectories, predictions and logs contain prompts and tool output: keep them out of public releases.
