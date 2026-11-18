# SWE-bench pilot：agent 主机、任务与评测（冻结）

[English](swe-bench-mini-pilots.md) | 简体中文

> **2026-11-16 冻结**（在 `da0058e`）：[docs/benchmarks/swe-bench-mini-swe-agent.zh-CN.md](../benchmarks/swe-bench-mini-swe-agent.zh-CN.md) 第 7.2、7.3 和 7.5 节的正文，即 2026-09/10 多 session SWE-bench pilot（作业 3299922 和 3300434）的设置，由 docs-leftover-1116 移到这里，除相对链接外不做改动；不再更新。节号指该页（“上面列出的版本”是该页 7.1 节的版本）。现在的 agent 主机、任务和评测见 [swe-bench.zh-CN.md](../benchmarks/swe-bench.zh-CN.md)（[Agent 主机的准备](../benchmarks/swe-bench.zh-CN.md#agent-主机的准备)、[任务](../benchmarks/swe-bench.zh-CN.md#任务)、[评测](../benchmarks/swe-bench.zh-CN.md#评测)）；作业结果见 [docs/status.zh-CN.md](../status.zh-CN.md)。

## 出自 7.2 节：Agent 主机

**pilot 的环境与命令（2026-09/10），仅作记录。** 现在的脚本会拒绝其中一部分：`images.py` 按固定的 digest 检查每个 instance 镜像（不能直接 `docker pull ...:latest`），`evaluate.sh` 需要固定 commit 的 SWE-bench 评测程序，并把固定版本的 test split 作为本地文件评测（不是 PyPI 上的 `swebench==5.0.2`，也不是 `--dataset_name SWE-bench/SWE-bench_Verified`），`build_tasks.py` 读取固定 revision 的数据集。现在的 agent 主机、任务和评测见 [swe-bench.zh-CN.md](../benchmarks/swe-bench.zh-CN.md)：[Agent 主机的准备](../benchmarks/swe-bench.zh-CN.md#agent-主机的准备)、[任务](../benchmarks/swe-bench.zh-CN.md#任务) 和 [评测](../benchmarks/swe-bench.zh-CN.md#评测)。

agent 主机需要 Docker、`curl`、`ss`、`python3`、一个 agent 环境和一个评测环境。pilot 的环境是上面列出的版本（证据中只记录了 Mini-SWE-Agent 的版本）；创建它们的命令没有记录，所以下面的命令**不在已验证的运行范围内**：

```sh
MINI=/abs/path/to/mini-env
EVAL=/abs/path/to/eval-env
uv venv --python 3.12 "$MINI"
uv pip install --python "$MINI/bin/python" mini-swe-agent==2.4.2 litellm==1.103.1 datasets==5.0.1
uv venv --python 3.12 "$EVAL"
uv pip install --python "$EVAL/bin/python" swebench==5.0.2 datasets==5.0.1
```

agent 离线运行，使用事先准备好的缓存：Hugging Face 数据集 `princeton-nlp/SWE-Bench_Verified`（runner 和任务构建脚本用），以及每个 instance 一个 Docker 镜像 `docker.io/swebench/sweb.eval.x86_64.<把 instance id 中的 "__" 换成 "_1776_" 并转成小写>:latest`。评测程序加载 `SWE-bench/SWE-bench_Verified`（十个 pilot instance 的 statement 与前者完全相同），运行时没有设置离线变量。在能联网时准备一次缓存，例如用 `load_dataset(...)` 和 `docker pull`（不在已验证的运行范围内）。

## 出自 7.3 节：任务

每个服务的 `--task` 列表必须恰好包含其 agent 要跑的 instance，`{instance_id, problem_statement}` 原样取自 SWE-bench Verified。请求按任务消息里的 problem statement 路由，所以如果某个 agent 的 instance 不在列表里，该服务的这次运行会以 HTTP 错误结束（4.4 和 6.3 节）。`examples/swebench/build_tasks.py` 按我们 harness 的方式生成这些文件：每个 instance 一个 `task-<id>.json`，每个服务组一个 `tasks-g<k>.json`（分成最少的组，每组最多 `--max-sessions` 个，各组大小最多差一）。在能访问数据集的地方运行：

```sh
"$MINI/bin/python" examples/swebench/build_tasks.py --output-dir /abs/path/to/tasks --max-sessions 5 \
  astropy__astropy-12907 astropy__astropy-13236 astropy__astropy-14096
```

给定相同的 statement，它能逐字节复现 pilot 固定的十个 `task-*.json` 文件，以及作业 3299922 和 3300434 的 `tasks-g*.json` 列表；对这十个 instance，如果 statement 与 pilot 固定的 SHA-256 不同，它会拒绝。把 `tasks-g<k>.json` 复制到 GPU 一侧。

对 session 池服务，`--pool` 按给定顺序生成一个包含全部 instance（1 到 64 个）的 `tasks-pool.json`，不再分组；`run_agents.sh` 按这个顺序补入 agent：

```sh
"$MINI/bin/python" examples/swebench/build_tasks.py --output-dir /abs/path/to/tasks --pool <instance id>...
```

## 出自 7.5 节：评测

agent 结束后，在 agent 主机上按策略评测（同一策略的所有组输出放在一次调用里）。这里的 `EVAL` 是 swe-bench 页面 [Agent 主机的准备](../benchmarks/swe-bench.zh-CN.md#agent-主机的准备) 一节的评测环境（commit `02e7a74` 的 SWE-bench）：`evaluate.sh` 会拒绝不能读取本地 Parquet 数据集的评测程序，而 7.2 节记录里 PyPI 上的 `swebench==5.0.2` 没有为此检查过。

```sh
EVAL_PYTHON="$EVAL/bin/python" examples/swebench/evaluate.sh my-run-goal_query goal_query \
  /abs/path/to/eval/goal_query /abs/path/to/runs/goal_query-g0 /abs/path/to/runs/goal_query-g1
```

它写出一个包含所有计划 instance 的 predictions 文件（没有 prediction 的 instance 用空 patch，并报告为缺失），在新的评测目录里运行评测程序（pilot 中是 `python -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Verified --split test --predictions_path … --max_workers 2 --timeout 1800 --run_id …`；现在使用固定版本 test split 的本地 Parquet 文件，见 [评测](../benchmarks/swe-bench.zh-CN.md#评测)），并根据 `logs/evaluation/<run_id>/results.json` 为每个 instance 输出一行。trajectory、predictions 和日志都包含 prompt 和工具输出，不要放进公开发布。
