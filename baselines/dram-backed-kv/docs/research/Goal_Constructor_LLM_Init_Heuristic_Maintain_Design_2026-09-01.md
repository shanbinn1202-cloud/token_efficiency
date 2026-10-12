# Goal Constructor 设计：一次性 LLM 初始化 + 启发式维护 + 状态条件化策略切换

> 状态：研究设计背景。当前代码已移除未带来收益的 embedding semantic-drift 旁路。

> 日期：2026-09-01
> 定位：**收敛版** Goal 构建设计文档。[桥梁文档](./Goal_Keep_Patch_Rewrite_Heuristic_Gate_Literature_Bridge_2026-09-01.md)降级为候选信号池参考，本文是主设计。
> 主骨架参考：PMCoder（arXiv:2608.06811，已全文核实，参数见桥梁文档附录 A）
> 相关：[2pager](./DRAM-Backed_KV_Working_Set_Slack_2pager_EN_2026-08-26.md)；原文提到的外部文献调研稿未纳入本仓库。

## 0. 核心设计决策：判断与生成分离

"如何用 heuristic 构建 goal"这个难题的答案是：**heuristic 不构建 goal**。

| 职责 | 承担者 | 成本 |
|---|---|---|
| goal **是什么**（生成） | 初始化一次 LLM + REWRITE 时 <1B SLM | 全程只有这两类生成点 |
| goal **要不要变**（判断） | 每轮规则 gate | µs–ms |
| 检索/压缩**用什么策略**（切换） | 每轮状态检测器 → 参数映射表 | µs–ms |

三条原则：

1. **昂贵生成只存在两类调用点**：任务初始化（全 session 一次 LLM）与每次生成式 REWRITE 触发时（<1B SLM），绝不每轮例行生成；
2. **正常执行只调策略参数、不碰 goal**——改 goal 是异常路径，不是常规路径；
3. **顶层总目标全程锁死，重写只发生在 selector 使用的 retrieval-goal 层**；Goal Constructor 不承担任务规划或执行调度。

## 1. 总体架构

```text
t = 0（一次性）:
  LLM(issue/task) → root_goal（锁死）+ 初始 current_retrieval_goal G_0

t > 0，每个 turn boundary（主模型 prefill 之前）:
  O_{t+1} 到达
       ↓
  特征抽取（正则/解析器级）: E_{t+1}、return code、diff、归一化命令
       ↓
  ├─ 状态检测器 → state ∈ {探索, 假设, 实现, 验证, 停滞}
  │        → 映射 KV 工作集参数（slack 阈值 / 预算 / 多样性 / 结构权重）【不碰 goal】
  │
  └─ 有效性 gate → KEEP / PATCH / REWRITE
           KEEP    原样复用 G_t，selector query 不变
           PATCH   确定性字段更新（entities / status / evidence_needed）
           REWRITE <1B SLM 只重写 retrieval goal（锚定 root_goal）
```

## 2. 初始化：一次性 LLM goal 构建

输入：issue / task 描述（可选 + repo 结构摘要、README 头部）。

输出（结构化；解析失败回退固定骨架，不静默失败——PMCoder 同款 fallback）：

```yaml
root_goal: 一句话总任务锚点（全程锁死，如"修复 XX 模块的空指针 Issue"）
current_retrieval_goal:      # 只用于 selector，不是执行计划
  description: 当前应检索哪类历史证据
  phase_hint: exploration | hypothesis | implementation | verification
  entities: [文件/符号/测试/错误初值]
  evidence_needed: [当前希望找回的证据类型]
  status: active | blocked | verifying
constraints: 从 issue 提取的硬约束
```

要点：

- 成本：一次 LLM 调用，全 session 摊薄。初始 retrieval goal 的质量是 selector 的重要上限，这一步**不省钱**，可用主模型或中档模型；
- 借用 PMCoder“任务开始只调用一次 LLM、后续规则维护”的成本结构，但**不采用**其 typed subtask/goal stack；本系统的 Goal 只是检索控制信号；
- 这也是与教授各配置共享的初始化阶段（G0 除外），不是独立配置项。

## 3. 状态检测器：切换策略，不碰 goal

输入：归一化后的最后一条命令 + agent reasoning 关键词 + 执行反馈。有序规则（PMCoder 已核实的四类信号，修改动作优先于查看）：

| 信号 | 状态 |
|---|---|
| 文件查看行为、探索性语言 | 探索 |
| 诊断命令、root-cause 语言 | 假设 |
| 文件修改动作（优先级高于查看） | 实现 |
| 跑测试、显式验证语言 | 验证 |
| 停滞四信号任一：失败返回 ≥5 / 单文件编辑 ≥6 / 只读饱和 ≥15 / 归一化 action 重复 ≥8（最小 10 步 floor） | 停滞 |

**滞回**：前向状态转移立即接受；回退/横移需连续 2 步重复证据才生效（debounce，防噪声抖动，PMCoder 核实值）。

状态 → KV 工作集参数映射（初版表，映射系数与换算待标定）：

| state | slack 阈值 | 工作集预算 | 多样性权重 | 结构/实体权重 | 优先保留 |
|---|---|---|---|---|---|
| 探索 | 拉高 | 大（~20k） | 高 | 中 | 任务锚点、recent tail |
| 假设 | 中 | 中（~16k） | 中 | 高 | 诊断输出、stack trace |
| 实现 | 压低 | 小（~12k） | 低 | 高（文件图） | 编辑文件及其依赖 |
| 验证 | 中 | 中（~16k） | 低 | 中 | 返回码、复现记录、测试结果 |
| 停滞 | 拉高 | 大（~20k） | 高 | 中 | 失败尝试、干净基线信息 |

预算列以 PMCoder 的 phase budget（20k/16k/12k/16k/20k）为起点；在本系统中换算为 page 预算与 slack 阈值。

关键性质：

- **正常执行中这是唯一会变的东西**，gate 走 KEEP 快路径，每轮 goal 维护成本 ≈ 特征抽取成本；
- 比"每轮都判断要不要改 goal"稳定得多：状态检测错了，最坏只是参数次优，不影响正确性（advisory only，PMCoder 同款姿态）；
- 停滞状态同时是 gate 的硬触发（§4.3 的 stuck_recovery）。

## 4. 有效性 gate：KEEP / PATCH / REWRITE

### 4.1 KEEP（默认快路径）

- 条件：无硬触发，且实体重叠正常；
- 动作：`G_t` 与 selector query 原样复用，不写入新实体或状态；
- 预期占比：绝大多数 turn。"正常执行零 goal 变更率"本身就是可报告的卖点数字。

### 4.2 PATCH（确定性字段更新，零生成）

- **实体更新**：`E_{t+1}` 并集进 `entities`（可选：连续 k 轮未出现的实体降级为弱候选）；
- **状态迁移**：blocked → active（阻塞命令转绿）；verifying → done；
- **证据需求更新**：根据已验证的 test/verdict，确定性删除已满足的 `evidence_needed`，或加入与新错误直接对应的证据类型；
- 限制：`current_retrieval_goal.description` 只允许经 REWRITE 更换；`constraints` 只允许 user/system 消息添加（agent 自述不是约束来源，防漂移入口）。

### 4.3 REWRITE（retrieval-goal 级，锚定式）

硬约束：**不输出、不修改 `root_goal`**。SLM 只重写 `current_retrieval_goal` 及其字段；输出必须服务历史选择，而不是生成下一步执行计划。

| 子类型 | 触发 | 方式 |
|---|---|---|
| `stuck_recovery` | 停滞状态 | **确定性 PATCH**：把 selector focus 扩展到最近成功状态、失败尝试、验证记录和原始任务锚点；不生成执行恢复计划，不修改工作区 |
| `hypothesis_invalidated` | 新错误类型 / top-frame 变化 | <1B SLM |
| `semantic_drift` | 累积相似度报警（EWMA/CUSUM，持续 m 步） | <1B SLM |
| `user_request` | 用户新需求 | <1B SLM（唯一允许改 constraints 的路径） |
| `goal_invalid` | 当前 retrieval goal 与持续执行证据不再一致 | <1B SLM |

SLM 输入：`root_goal`（只读）、`G_t`、`a_t`、`O_{t+1}`（截断）、`E_{t+1}`、触发类型和紧凑状态统计。
SLM 输出：`current_retrieval_goal.description`、`phase_hint`、`entities`、`constraints`（仅 user_request）和 `evidence_needed`。

因此必须分开报告 **goal-update rate**（含确定性 PATCH）与 **SLM generation rate**；前者可以显著高于后者，因为实体、状态和证据字段更新不需要生成。这是相对“每轮 SLM”（教授配置 (iii)）的直接成本证据。

## 5. 候选信号池（发散区，后续迭代）

初版只跑最小集；其余进池子做消融。完整机制推导见[桥梁文档](./Goal_Keep_Patch_Rewrite_Heuristic_Gate_Literature_Bridge_2026-09-01.md) §2–§3。

| 族 | 候选信号 | 初版启用 |
|---|---|---|
| 执行硬触发 | D1 新错误证据、D2 phase 跳转、D3 阻塞解除、D4 停滞、D5 evidence satisfied、D6 用户新需求、D7 账龄检查点 | **D4、D5、D6** |
| 连续信号 | c1 实体重叠、c2 语义相似度、c3 累积漂移（EWMA/CUSUM）、c4 phase 一致性 | **c1** |
| 决策机制 | 滞回、不对称阈值（false KEEP 代价 ≫ false REWRITE） | **滞回** |

迭代顺序建议：先加 c3（防慢速漂移，对应停滞四信号抓不到的渐变），再加 D7（账龄检查点），最后才是 c2/c4 这类需要 embedding 的信号。

## 6. 成本核算与实验对齐

| 事件 | 生成成本 |
|---|---|
| 任务初始化 | 1 次 LLM 调用 |
| 正常 turn（KEEP / 状态切换） | 0 |
| PATCH | 0 |
| 确定性 retrieval-goal PATCH | 0 |
| 生成式 REWRITE | <1B SLM × 1 |

与教授配置矩阵的关系：本设计 = **(ii) heuristic + (iii) SLM 的条件化组合**，即教授设想的完成形态；G0（无构建）、G2（每轮 SLM）、G3（agent 自报告）保留为对照；**状态条件化策略切换**作为独立的横向消融（映射表开/关）。

关键报告数字：

- goal-update rate vs generation rate（规则短路比例）；
- 正常执行零 goal 变更率；
- 均摊 `T_goal` 的 p50/p95；
- **顶层一致性**：以结构约束（root ID/硬约束是否保持）、goal-boundary 标注准确率和下游 selector recall 为主；`cos(embed(current_retrieval_goal), embed(root_goal))` 仅作辅助诊断，因为合法的检索焦点也可能与总目标表面相似度较低；
- Goal 层质量 → selector Recall@budget → task success 的链条解耦报告。

### 6.1 SLM rewrite 的显存与时延边界

SLM **不读取完整 trajectory，也不需要完整 trajectory 的 KV**。它只读取 `root_goal + current retrieval goal/state + latest action + clipped latest observation + extracted entities + trigger type + compact statistics`。建议把输入硬限制在约 0.8–2.5k tokens，输出限制在约 50–150 tokens；历史事实通过结构化 state/实体和 selector 后续恢复，而不是先灌给 SLM。

对参数量为 `P`、每个权重占 `b_w` 字节的模型，权重显存约为 `P·b_w`：1B BF16 约 2.0 GB、INT8 约 1.0 GB、INT4 约 0.5 GB，另加 runtime workspace。SLM 自身 prompt KV 约为：

`KV_bytes = tokens × 2(K,V) × layers × kv_heads × head_dim × dtype_bytes`。

典型 <1B GQA 模型在 2k-token prompt 下的 KV 通常是几十到约两百 MB，而不是主 Agent 数万 token trajectory 的 KV。实际峰值取决于部署方式：

- 同 GPU 常驻：`main-model resident memory + main active working-set KV + SLM weights/workspace + compact SLM KV`；需预留约 0.7–2.5 GB 量级，依量化和模型结构实测；
- 独立 GPU/CPU/独立服务：主模型 GPU 峰值基本不增加，但会增加通信或 CPU 延迟；
- 同 GPU 按需装载：节省常驻显存但加载权重通常抵消 rare-rewrite 的延迟优势，不建议作为默认实现。

因此成本应按 `amortized T_goal = T_rules + p_gen·T_SLM` 报告，其中 `p_gen` 是生成式 REWRITE 比例。GPU 上短 prompt 的 <1B SLM 通常应按几十至数百毫秒区间预算，CPU 可能达到亚秒至数秒；论文中必须在目标硬件上报告 p50/p95，不能只引用理论 FLOPs。

## 7. 风险与边界

1. **一次性初始化质量会影响早期选择**：初始 retrieval goal 写错会降低 selector recall；兜底是 `goal_invalid` 触发重写，标定期可对比“主模型初始化 vs 中档模型初始化”。
2. **锚定不保证方向正确**：retrieval goal 锚在 root 上仍可能选错证据，需与 G2 对照说明收益来源，不能只靠锚定讲故事。
3. **状态检测的语言规则依赖 reasoning 文本格式**：换基座模型需迁移测试；缓解：命令类信号优先，语言信号只做辅助。
4. **停滞阈值是 PMCoder 的 pilot 值**（SWE-bench 上固定），在我们的轨迹分布上必须重新标定后再当默认值用。
5. **gate 时序约束**：全部输入必须在主模型正式 prefill 前可得，不读主模型 KV（见 survey §5.3 的双-prefill 论证）。

## 8. 文献锚点（最小集）

- **PMCoder**（arXiv:2608.06811）：借用一次性 LLM 初始化、规则状态机、2 步滞回、停滞四信号、phase 条件化预算和 anchor 关键词检索；**不采用**其 subtask/goal stack 与执行恢复规划。核实细节与全部阈值见桥梁文档附录 A。
- **STITCH**（ACL 2026）："默认继承直到边界事件"的决策语义；分割粒度权衡。
- **Goal drift 评测**（arXiv:2505.02709）：顶层锚定、账龄检查点、回归父任务的机制依据。
- **Proactive Memory Agent**：`no_intervention` 作为 KEEP 的一等输出形态。
- 其余信号级来源（主题分割、变点检测、选择性分类等）见桥梁文档 §2 与附录 B。
