# DRAM-Backed KV：Online Selector V2

本仓库当前以 **Online Selector V2** 为主线：把 Qwen 27B 的历史 KV 固定切成 16-token Page，完整历史落在 CPU DRAM，每轮只恢复选中的工作集到 GPU HBM，并在真实 Mini-SWE-Agent 闭环中继续生成。

当前有三个可直接对照的模式：

| 模式 | 历史 Page | 排序 query | 额外 27B forward |
|---|---|---|---|
| `full_context` | 全部保留 | 无 | 无 |
| `observation` | 按预算选择 | 最新 observation prefill 的深层 hidden state | 无 |
| `goal` | 按预算选择 | 结构化 Retrieval Goal 的深层 hidden state | 仅新 Goal hash；命中缓存时无 |

Page 与 query 都取主模型最后一个 decoder block 的 last-token hidden state，经同一固定投影压到 128 维、L2 归一化和 int8 量化后做 cosine。Selector 只生成 `SelectionPlan`；Backend/Runtime 只执行 KV 生命周期和迁移，Goal Constructor 不直接选择 Page。

## 从这里开始

- 当前架构、数据流和算法：[Online Selector V2 代码架构](docs/current/Online_Selector_V2_Code_Architecture_2026-09-06.zh-CN.md)
- 16-token Page 设计：[Prefill-reused Page Selector](docs/current/Selector_Prefill_Reused_Page_Design_2026-09-04.zh-CN.md)
- 实验入口和分组：[experiments/README.md](experiments/README.md)
- 脚本用途索引：[scripts/README.md](scripts/README.md)
- 结果保存规则：[results/README.md](results/README.md)
- 历史设计与研究材料：[docs/README.md](docs/README.md)

## 运行测试

项目核心测试不需要第三方依赖：

```bash
PYTHONPATH=src python3.12 -m unittest discover -s tests -v
```

## 在线实验

单次 H200 在线运行由 `scripts/run_online_selector_v2_swe_h200.sbatch` 启动。组内复现实验前，复制 `scripts/env.local.sh.example` 为未跟踪的 `scripts/env.local.sh`，填写本机模型与 Python 路径。

当前五实例三模式 cohort 的提交方式为：

```bash
sbatch --export=ALL,EXPERIMENT_MODE=full_context,RUN_GROUP=A scripts/run_online_selector_v2_cohort2_triplet_h200.sbatch
sbatch --export=ALL,EXPERIMENT_MODE=full_context,RUN_GROUP=B scripts/run_online_selector_v2_cohort2_triplet_h200.sbatch
sbatch --export=ALL,EXPERIMENT_MODE=goal,RUN_GROUP=A scripts/run_online_selector_v2_cohort2_triplet_h200.sbatch
sbatch --export=ALL,EXPERIMENT_MODE=goal,RUN_GROUP=B scripts/run_online_selector_v2_cohort2_triplet_h200.sbatch
sbatch --export=ALL,EXPERIMENT_MODE=observation,RUN_GROUP=A scripts/run_online_selector_v2_cohort2_triplet_h200.sbatch
sbatch --export=ALL,EXPERIMENT_MODE=observation,RUN_GROUP=B scripts/run_online_selector_v2_cohort2_triplet_h200.sbatch
```

该 wrapper 固定 `STEP_LIMIT=90`，A 组为 `14369 + 13236 + 14096`，B 组为 `14995 + 14508`；每个 allocation 内最多同时运行两条，避免最慢实例拖住另外一组。

## 当前边界

- `GoalPolicyHint` 仍是日志中的 advisory metadata，尚未控制在线预算。
- block-native attention 尚未实现；Runtime 仍需组装选中的连续 cache。
- `src/dram_kv/selectors/embedding.py`、`hybrid.py`、`retention.py` 是 Selector V1 离线实验代码，不是当前 V2 的主排序路径。
- Goal embedding semantic-drift 旁路因未改善 Recall 已移除；KV-drift 数值兼容回归是另一条保留实验。

## 仓库约定

代码放在 `src/`，测试放在 `tests/`，共享脚本放在 `scripts/`。小型配置、manifest 和总结可以进 Git；完整 trajectory、逐轮日志、向量、中间产物和打包文件不进 Git。根目录不再放日期型设计文档或实验压缩包。
