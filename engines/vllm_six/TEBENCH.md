# Six methods on clean upstream vLLM

Base: official v0.28.0, commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`.
The sibling `../vllm_upstream_028` retains unchanged upstream tracked files.
`git diff HEAD` in this worktree shows the complete adaptation.

## Runtime

All six methods use native `AsyncScheduler`, FCFS, and eight request slots.
The native queue, victim selection, prefill reservations, reset API, Worker,
and output processor retain their upstream behavior. Storage hooks translate
logical token positions to physical cache positions and replay previous
selections after native preemption. A commit fence prevents another batch
from using cache pages while a compaction changes their ownership.

RPCs use the native `worker_extension_cls` interface. AgentKV and RepairKV
turns use ordinary requests and the native KVConnector interface. Original
algorithm dependencies, such as CPU scoring or phase classification, still
synchronize locally. Async scheduling does not imply these operations overlap.

## Policy package

`vllm/tebench` contains storage, query capture, replay, connectors and auditing.
The research repositories under the project `baselines` directory are imported
without modification. Vanilla, Sliding Window and AgentDiet use the standard
completion transport; their selection or reduction happens in the frontend.
AgentDiet uses CoACT's original token counter, prompt, delayed target selection
and acceptance rules.

AgentKV retains its original phase classifier, query ring, per-layer/head
selector and 128-token trigger. Its exact cache, recurrent state and ring are
checkpointed on CPU between ordinary requests so idle turns release GPU pages.
RepairKV retains the full committed history pool and original per-turn base
selection, CPU scoring, burst selection and probe rollback (multi-turn
Algorithm 1). IntentKV calls the original per-query-head selector and stores
the exact selected head unions in native pages.

Native KV pool size is not total method memory. RepairKV's history pool and
IntentKV's physical head unions are recorded separately from logical budgets.

## Validation and experiments

Results and source manifests:
`/u/ssun4/token_efficiency_results/clean_vllm_swe20_20260925`.
The matching official wheel's generated resources are checked against its
RECORD hashes before being linked; no old modified engine Python files are
used as runtime dependencies.

From the project root, use `scripts/run_clean_stage.py`. Source fingerprints
are frozen only after the required CPU and real-model checks pass. Formal
runs verify the frozen sources, original algorithm repos, configuration and
the exact twenty SWE-bench Verified task IDs.
