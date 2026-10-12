# Token Efficiency

SWE-bench 与 MemoryAgentBench 的 KV 策略实验代码和共享依赖。

## 当前主实验

当前队列使用已冻结的 vLLM 0.24 实验包。**各任务的配置文件是实际运行配置的依据**；本目录保留的历史通用入口和 `configs/default.yaml` 不代表当前队列。

| 内容 | 路径 |
|---|---|
| SWE-bench Verified：全部 500 题、五种方法 | `/projects/bcjw/ssun4/swe500_qwen_20261009/` |
| MemoryAgentBench：固定 32 题、五种方法 | `/projects/bcjw/ssun4/memagent32_qwen_20261010/` |
| 当前主 vLLM 0.24 引擎 | `/projects/bcjw/ssun4/token_efficiency/engines/vllm_024/` |
| 冻结 harness、Python 环境与编译缓存 | `/projects/bcjw/ssun4/vllm024_triton_six_20261008/` |
| 十种任务组合的 GPU 启动验证 | `/projects/bcjw/ssun4/qwen_interactive_validation_20261010/` |

五种方法为 Vanilla、Random、Sliding Window、AgentKV、IntentKV。当前 Qwen 配置为 262144 上下文、25 GiB KV pool、16 并发、16384 批次 token 容量、seed 42、max_tokens 4096、max_turns 120，使用异步调度和 Triton attention/GDN。SWE-bench 压缩预算为 8K，MemoryAgentBench 为 32K；Vanilla 保留完整历史。MemoryAgentBench 使用原有自定义工具框架。

每个方法的配置分别在 `<SWE实验目录>/<方法>/8k.yaml` 和 `<Memory实验目录>/<方法>/memagent/32k.yaml`。已有任务由其控制器管理，请勿重复提交。

只读验证当前配置和源码是否与已通过的 GPU 验证一致：

```bash
python /projects/bcjw/ssun4/qwen_interactive_validation_20261010/gate.py
```

## 目录说明

| 目录 | 用途 |
|---|---|
| `baselines/` | 原方法仓库、公共评测组件及 vLLM 副本 |
| `engines/` | 保留的各版 vLLM 引擎 |
| `experiments/benchmarks32_20261004/` | 当前任务复用的评测、计时、监控代码 |
| `experiments/memagent20_20261003/` | 当前 MemoryAgentBench 复用的逐题执行代码 |
| `experiments/memagent40_20261003/` | 保留仍被引用的预处理代码与说明 |
| `experiments/agentkv_qwen36/` | AgentKV 兼容入口与第三方依赖 |
| `experiments/common_attention_20260927/` | attention 对齐实验的代码快照 |
| `data/` | 数据、Apptainer 镜像和评审参考材料 |
| `external/`、`mini-swe-agent/` | 基准、评分和 agent 的第三方依赖 |
| `tebench/`、`adapters/`、`scripts/`、`configs/`、`tests/` | 保留的通用评测实现、适配层、脚本与测试 |
| `results/` | 本目录内仍需保留的结果和源码快照；新主实验输出位于上表的实验目录 |
| `logs/` | 被旧源码快照引用的日志，保留以避免断链 |

冻结 harness 的 `baselines`、`data`、`experiments`、`external`、`mini-swe-agent` 通过符号链接引用本目录。历史命名中的 20/32/40 不等于当前评测题数，不能仅凭目录名删除共享实现。

## 2026-10-11 路径迁移

实际主目录为 `/projects/bcjw/ssun4/token_efficiency`；主 vLLM 源码已实体放入 `engines/vllm_024/`。

为兼容已提交的 Slurm 任务和冻结记录，保留以下入口：

- `/work/nvme/bcjw/ssun4/token_efficiency` → 当前主目录。
- `/projects/bcjw/ssun4/vllm024_triton_six_20261008/engine` → 当前主目录的 `engines/vllm_024/`。

现有启动器通过 `engine_root()` 解析真实路径，服务端和 worker 使用新目录中的主引擎。原配置、算法文件、评分逻辑及两个 allocation.sh 的最终内容均未修改。Python 环境和模型权重仍使用原位置。

迁移核对记录：`/projects/bcjw/ssun4/token_efficiency_migration_20261011/`。
已有 10 个 Slurm job 的接续控制器：`/projects/bcjw/ssun4/queued_job_recovery_20261011/`，仅连接已有 job，不重新申请分配。

## 2026-10-11 精简记录

清理了旧 KV 张量调试转储、顶层临时缓存、过时的实验入口、未使用的 ReTool 和旧调试结果。旧代码与小型结果在移除前已压缩归档；指向目录外旧结果的链接仅移除链接本身。

所有 vLLM 副本、引擎和源码快照保留。当前主实验依赖、数据镜像，以及此前按“模型 × benchmark × 方法 × budget”选定的最新完整结果和保留短测不作清理。

清单、恢复归档及核对记录：

`/projects/bcjw/ssun4/repository_cleanup_20261011/`

原版 README 同样保存在该目录的归档中，供查阅历史配置。此轮未修改实验算法、运行参数或评分规则。

## 2026-10-11 重新提交的十个主实验

原 3354164--3354173 已失败并归档。当前使用 `/projects/bcjw/ssun4/benchmark_resubmit_20261011/` 持久托管原始 salloc 进程；实时状态见该目录 `status.json`，本项目入口见 `current-benchmark-jobs.json`。不要启动旧的 queued_job_recovery 接续服务或原 pipeline 提交器。

新的 SWE500 job 为 3363635--3363639，MemoryAgentBench32 job 为 3363640--3363644。主引擎来自本项目 `engines/vllm_024`；冻结配置、数据和评分仍位于上表两个实验目录。新服务只修改了分配所有权和监督，算法与配置不变。提交前真实交互验证 3363523 已正常完成（COMPLETED/0:0），验证了主引擎启动、16并发、多轮请求和输出计数。新正式实验仍需等待资源并完成推理和评分。
# token_efficiency
