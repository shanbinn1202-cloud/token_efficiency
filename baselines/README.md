# Baselines

统一入口：在项目根目录执行 `bash scripts/run.sh --method METHOD`。
公共配置在 [configs/default.yaml](../configs/default.yaml)，参数与后端支持见 [项目说明](../README.md)。

| 目录 | 实现文件 | 职责 |
|---|---|---|
| `vanilla/` | `model.py` | 完整历史选择和模型适配 |
| `sliding_window/` | `model.py` | 固定前缀 + 最新 token，历史足够时精确填满 KV 输入预算 |
| `group/` | `model.py`、`memory.py`、`feedback.py` | 工具接口、scope/实体记忆选择、格式与恢复反馈 |
| `common/` | `history.py`、`model.py` | 共用历史结构和模型请求记录 |
| `common/` | `runner.py`、`harness.py`、`environment.py`、`prepare.py`、`concurrency.py`、`settings.py` | agent 执行、评分、镜像环境与准备、并发、评测配置结构 |

`threshold/`、`intent_memory/`、`generic_scope/` 已删除。
自有 baseline 共 19 个文件（含 README 与包入口，不计固定目录和字节码）。
方法测试统一在 `tests/`，执行 `bash scripts/test.sh`。
各方法原有 `run.py`、`run.sh`、`submit.sh` 等重复入口已删除；请使用 `scripts/run.sh` 和 `scripts/job.sbatch`。
镜像准备入口为 `scripts/prepare.sh`；scope 审计入口为 `scripts/audit_scope_system.py`。

`agentkv/` 和 `dram-backed-kv/` 是固定的外部复制目录，保持不动。
它们的适配代码位于项目根目录下的 `adapters/`。
