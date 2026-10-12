# AgentDiet（CoACT 实现）

从 `CoACT/src/eval/baselines/agentdiet.py` 接入，不引入 CFQ 或 KV window。

## 保留的算法

- 默认 delay_steps=2、window_before_steps=1、token_threshold=500。
- 在执行工具并追加结果后，选择延迟的 assistant/tool 步骤。
- 原步骤超过阈值才请求反思；节省 token 数严格超过阈值才接受。
- 反思 prompt、XML 解析、tool-call 名称/数量检查、改写及失败回退逻辑保持 CoACT 原样。
- `coact/trajectory_parser.py` 逐字复制；prompt 和应用方法按原文提取；`coact/source_manifest.json` 保存来源 SHA256。
- `coact/agentdiet.py` 只调整依赖导入，以及 `completion(...)` 到可注入的 `self.completion(...)`。不修改原 CoACT 仓库。
- 多工具步骤遵循来源实现：将合并后的 result 应用到第一个 tool 消息；未另改算法。

## 必要的框架适配

反思与 agent 调用同一个后端、同一模型、同一 GPU pool。反思 temperature=0 保留来源算法设定；其余采样、seed、输出上限及上下文检查使用统一配置。主 agent 的 thinking=true 保持对齐；反思专用 `reflection_enable_thinking=false` 显式沿用 CoACT 对 Qwen3.5 反思关闭 thinking 的行为，并用于当前 Qwen3.6。真实 GPU 检查已确认，开启反思 thinking 时，模型在思考中提及 `</step>` 会提前触发停止。此开关独立记录在 policy 中，可显式改为 true。使用原始 `stop=["</step>"]`，vLLM transport 会实际传递 stop。反思响应作为文本解析，不把 XML 内容误当待执行工具调用。

阈值 token 计数直接继承 CoACT 的 `litellm.token_counter`，包括原函数对模型名称的处理；不覆写为服务端 tokenizer，以保持压缩触发与接受条件一致。反思不自动重试，失败请求记录到 trajectory 后沿用来源的保留历史回退；不隐藏额外推理开销。本地服务不计算美元费用，真实 token usage 和时间照常记录。

CoACT 的压缩应用与审计方法保持原文。被压缩消息中的 `agent_erased` 和旧 `reasoning_content` 只存于轨迹，不发送给模型，以免模板重新带回已删除内容。每题拥有自己的策略和统计，线程之间不修改共享全局 completion。

AgentDiet 没有原生 8K/12K/16K 的 KV 限制。`cache.kv_budget` 是未使用的公共标签；不暗中叠加滑窗或按预算调阈值。因此无需为这三个标签重复运行并宣称不同预算策略。

## 对齐已有 SWE-bench 实验

`configs/agentdiet_swe20_5g.yaml` 对齐 `kvpool5g_budget16k_20260914_finalaligned/vanilla/run.json`：Qwen3.6-27B BF16、thinking=true、temperature=0、seed=0、输出4096、上下文65536、120轮、8客户端、5 GiB KV pool、TP=1、prefix caching=true、async scheduling=false、CUDA graph及compile开启、同一批20题与评分配置。不设置额外整题准入，也不指定 max_num_batched_tokens。除了服务端口，公共运行配置相同。

```bash
# 在 token_efficiency 根目录
bash scripts/run.sh --method agentdiet --config configs/agentdiet_swe20_5g.yaml --dry-run
sbatch scripts/job.sbatch --method agentdiet --config configs/agentdiet_swe20_5g.yaml \
  --output results/agentdiet_swe20_5g

# 在同配置 vLLM 服务已启动时验证真实反思和下一轮工具调用
source scripts/env.sh
python -B scripts/validate_agentdiet.py --config configs/agentdiet_swe20_5g.yaml \
  --output results/agentdiet_gpu_smoke.json
```

## 指标

`memory.requests` 分别记录 `kind=agent` 和 `kind=agentdiet`，反思失败也记录。`compression_stats` 记录原 CoACT 累计数据及成功改写审计。反思的 token、请求耗时、排队时间均计入基准开销；反思不计入 agent 的轮数或平均 agent 输入长度。

统一 `batch_output_tokens_per_second` 包括 agent+反思输出；新增 `agent_output_tokens_per_second` 只把 agent 输出放入分子，但分母仍包含反思的全部 wall time。报告吞吐时需注明口径，不能把新增反思输出误认为解题效率收益。另有 `agentdiet_calls`、`agentdiet_output_tokens`、`agentdiet_request_seconds`。失败反思令严格比较资格为 false。
