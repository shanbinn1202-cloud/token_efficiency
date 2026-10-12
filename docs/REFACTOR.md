# 项目结构与本次精简

## 目录边界

固定复制目录保持不动：`baselines/agentkv/`、`baselines/dram-backed-kv/`、`mini-swe-agent/`、`external/`、`experiments/agentkv_qwen36/vendor/`。
适配代码在 `adapters/`；统一配置、推理后端、运行与指标在 `configs/` 和 `tebench/`。

## 本次变更

- 删除 `baselines/threshold/`、`baselines/intent_memory/`、`baselines/generic_scope/`，同时删除方法注册和默认配置项；旧方法名会报错。
- `vanilla`、`sliding_window` 各将 selector 与 model 合并到 `model.py`。
- `group` 保留所有既有策略预设：实体解析、inline scope 与语义 scope 合并到 `memory.py`；格式、恢复和语义反馈合并到 `feedback.py`；模型接口仍在 `model.py`。
- 公共 policy 合并入 `common/history.py`；镜像解析合并入 `common/environment.py`；镜像准备从 vanilla 移至 `common/prepare.py`。
- 移除各方法重复的运行、服务、提交、任务脚本及旧兼容模块；统一使用 `scripts/run.sh`、`scripts/serve.sh`、`scripts/job.sbatch`。环境初始化集中在 `scripts/env.sh`。
- 测试集中到 `tests/`；scope 审计移至 `scripts/audit_scope_system.py`；依赖版本记录移至 `configs/versions.json`；镜像准备使用 `scripts/prepare.sh`。
- 自有 baseline 文件从 131 个减至 19 个（含 README 与包入口，排除固定目录和字节码）。

保留方法的 selector、提示词、反馈规则和默认参数不变。旧的 `python -m baselines.METHOD.run` 入口已移除，使用统一 CLI。

## 统一默认设置

Qwen/Qwen3.6-27B，BF16，thinking=true；max_tokens=8192、max_turns=120、max_window_size=131072、temperature=0。
KV budget=8192；sliding_window 自动保留 budget 内最多的最近完整 assistant turn，无固定轮数上限。
CUDA Graph 与 torch.compile 默认关闭；full attention 使用 SDPA/auto；线性层使用 Transformers 原生 dispatch（当前环境为 FLA）；默认不分块 prefill。
DRAM 使用 goal query、fixed budget、recent_turns=2、recent_quota=64，goal rewriter 默认 null。

vanilla 不裁剪历史；sliding_window 在输入 token budget 内选择最大的最近完整 turn 数 N。用户明确选择 budget 只约束每轮输入，生成期间允许暂时超过；max_tokens=8192 保持不变。
其余方法与后端限制、指标定义见 [README](../README.md)。

## 验证入口

`bash scripts/test.sh` 执行自有方法及框架测试。
`bash scripts/run.sh --list` 查看保留的方法；`--method METHOD --dry-run` 查看完整配置与最终策略。
本次仅做结构精简与 CPU 回归检查，不运行新的 SWE-bench 实验或 vLLM 验证。

本次验证结果：98 项测试通过；五种方法的 dry-run 和 sliding_window 参数覆盖通过；自有 Python / shell 语法检查通过；固定目录中 4,479 个普通文件的 SHA256 与文件列表和修改前一致。

## Sliding window 输入 budget 更新

- `baselines/sliding_window/model.py`：固定前缀 + 输入 budget 内最多的最近连续完整 turn；逐次删除最旧整轮并按实际模板计数，N 自动确定；前缀超预算报错，最新整轮放不下时仅保留前缀。
- `tebench/backends/base.py`：抽出共用 `render`，筛选计数和推理使用同一模板、工具定义、thinking 与 tokenizer；其他方法的生成参数不变。
- `tebench/methods.py`：传入公共 KV budget，并将 `kv_budget`、`budget_scope=input`、`selection=max_recent_turns` 写入最终策略快照。
- `configs/default.yaml`：更新参数注释，默认数值不变；测试与说明同步更新。

正在运行的 vanilla 作业 3133585 保留原执行逻辑，原源码已存于其结果目录的 `source_snapshot/`，用于核对及必要时续跑。固定外部复制目录未改动。


已移除 `methods.sliding_window.options.recent` 以及按固定轮数命名的预设，统一预设为 `sliding_window`。旧轮数参数会报错，避免静默忽略配置；轨迹的 `retained_turns` 记录每轮实际 N。

最大 N 更新验证：107 项测试通过；对 3 条实际轨迹使用本地 Qwen tokenizer，分别检查 4096 / 8192 / 16384 输入预算，共 9 组通过。逐一确认所有更大的连续后缀超预算。例如第一条轨迹在 8192 预算下保留 14 轮、输入 8181 tokens，在 16384 下保留全部 18 轮；输出上限始终保持 8192。未做 vLLM 验证。
