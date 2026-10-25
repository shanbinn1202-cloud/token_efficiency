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

五种方法为 Vanilla、Random、Sliding Window、AgentKV、IntentKV。已有 Qwen 结果对应的配置为 262144 上下文、25 GiB KV pool、16 并发、16384 批次 token 容量、seed 42、max_tokens 4096、max_turns 120，使用异步调度和 Triton attention/GDN。SWE-bench 压缩预算为 8K，MemoryAgentBench 为 32K；Vanilla 保留完整历史。MemoryAgentBench 使用原有自定义工具框架。

每个方法的配置分别在 `<SWE实验目录>/<方法>/8k.yaml` 和 `<Memory实验目录>/<方法>/memagent/32k.yaml`。已有任务由其控制器管理，请勿重复提交。

只读验证当前配置和源码是否与已通过的 GPU 验证一致：

```bash
python /projects/bcjw/ssun4/qwen_interactive_validation_20261010/gate.py
```

## 新实验批次配置

当前正式实验按模型设置 `max_num_batched_tokens`：Qwen 为 `16384`，Gemma 为 `4096`，同一模型在各 benchmark 上保持一致。上下文为 `262144`，KV pool 为 `25 GiB`，并发为 `16`，`seed=42`，`max_tokens=4096`。已有结果保留实际使用的配置；`configs/benchmarks_batch4k/` 是此前准备的 4K 配置，不代表已完成 Qwen 实验的实际启动参数。

Gemma baseline 的正式配置位于 `experiments/gemma_full_20261017/formal/`。Goal 适配、独立重写模型服务、Terminal-Bench 2.0 的 Terminus-2/ARM64 运行环境及配置位于 `experiments/goal_terminal_alignment_20261023/`。源码、测试和配置随仓库发布；模型权重、Apptainer 镜像、安装环境、数据缓存、运行日志和评测产物保留在服务器，不纳入 Git。

Terminal-Bench 的 `sanitize-git-repo/tests/test_outputs.py` 包含上游故意设置的伪凭据，因 GitHub push protection 不随本仓库发布；服务器上的原文件保持不变，完整测试需从原始 Terminal-Bench 数据集获取。

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

# token_efficiency
