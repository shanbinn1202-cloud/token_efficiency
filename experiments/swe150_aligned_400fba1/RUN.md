# SWE-bench Verified 150 on local ARM64 testbeds

The job ledger is [local-jobs.json](local-jobs.json). Each of the four independently scheduled ghx4 allocations requests one GPU, 32 physical cores and 145 GiB for five hours. Job outputs are under `runs/<method>/<job_id>/`. The Slurm IDs and dependencies in `revisions/ec2-submissions/` belong to the superseded EC2 path.

The two-hour `ghx4-interactive` validation runs first. The four five-hour production jobs depend on its successful exit. The production entry point also checks a passing interactive report with the same frozen file manifest. Short agent trajectories, synthetic above-budget inputs and reference-patch controls are recorded separately under `interactive-check/runs/<job_id>/` and never enter the 150-task results.

All methods use Qwen3.6-27B, the ordered 22 Astropy and 128 Django tasks in the 400fba1 release, its repository-specific testbed prompt and Mini-SWE-Agent XML tool calls. The server uses 24 concurrent sequences, asynchronous scheduling, a 25 GiB GPU KV pool, 262144-token context and 16384-token batch capacity. The agent uses thinking, temperature 0, seed 42, up to 4096 new tokens and 120 steps, with a 4400-second request timeout. Each allocation checks the frozen code, stages the model on local disk and installs the release with `install.py --apply`.

| Method | Policy |
| --- | --- |
| Random | Native random_fill, 8192 active tokens |
| Sliding Window | Native recent_fill, 8192 active tokens |
| AgentKV | Per-head StageQSnapKV, 8192-token budget, AgentKVConnector |
| Claude Code compression | Official Claude Code 2.1.325 /compact at the start of a turn exceeding 8192 tokens. The summary shares the Qwen service. Auxiliary output is counted separately. |

The native selectors enable idle-page release, 68 GiB host arena and 49 Mamba slots. AgentKV uses its own connector and Claude Code runs ordinary vLLM requests. Both launch with 49 Mamba slots; the native selector's idle-page-release flag and DRAM arena do not govern those requests.

Tool execution and grading use private writable ARM64 Apptainer testbeds on the allocated node. The working tree starts at the official instance commit and each tool command runs via bash -lc with the testbed environment active. Local image/CPU and filesystem costs can change elapsed time relative to runs using the shared x86 EC2 Docker host. The task roster and test specifications come from the same Verified snapshot.

## GPU service and ARM64 container resources

Two concurrent Slurm steps share a GH200 node. Slurm assigns disjoint physical cores and separate memory reservations for the service and the clients. The controller verifies the actual job allocation before starting the service. The step launchers record their affinity, cgroup limits and local scratch device, and the client checks its core set against the service's before starting inference.

| Step | Physical cores | Host memory requested | Workload |
| --- | ---: | ---: | --- |
| GPU service | 16 | 113560 MiB (110.90 GiB) | One GPU and the vLLM service |
| Agent | 16 | 30 GiB | Shared by 24 sessions, one 120-second timeout per tool invocation |
| Grading, after GPU service shutdown | 16 | 30 GiB | 8 concurrent workers, 1800 seconds per instance |

Approximately 4.1 GiB of the 145 GiB allocation remains for the batch controller and overhead. Sandbox writes use the node's NVMe-backed `/tmp`. Grading begins after the service exits and the inference port closes. Inspect `allocation.json`, `engine-placement.json`, `agent-placement.json` and `grade-placement.json` in the run directory for the allocation and observed step limits. CPU and memory isolation are on the same ARM64 host; the remote x86 EC2 data retain a different tool runtime, disk and network path.

The client's `evaluation/metrics.json` gives main output tokens divided by elapsed runner seconds (first task start to last task close, excluding engine startup and grading). `evaluation/result.json` records grading coverage and the resolved fraction once all 150 can be graded. Per-task trajectories, request usage and completion receipts are written during inference. For an interrupted run, `clients/local_grade.py --partial` can grade and summarize its durable outputs.

Local preflight receipts: [image audit](local_images/audit.json), [repository test controls](local_images/representative-tests.json), [client and grading checks](clients/local-preflight.json), [installed engines](engines_adapter/cpu-validation.json). Live engine checks run on the allocated GPU before formal inference.
