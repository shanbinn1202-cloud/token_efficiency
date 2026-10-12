# Goal Keep / Patch / Rewrite 启发式 Gate：文献桥梁与规则构建

> 状态：研究资料，不等同于当前 Gate。文中的 embedding semantic-drift 路径已从代码移除。

> 日期：2026-09-01
> 定位：把外部文献调研稿的 §2.4（四层结构）与 §5（设计建议）桥接到**可执行的启发式规则设计**；该调研稿未纳入本仓库。
> 对应教授的配置 (ii)：heuristic reuse + small model goal construction（双分支：便宜启发式判有效性 → 失效才调 SLM 重写）
> 原始设计：[DRAM-Backed_KV_Working_Set_Slack_2pager_EN_2026-08-26.md](./DRAM-Backed_KV_Working_Set_Slack_2pager_EN_2026-08-26.md)
>
> **导航（2026-09-01 更新）**：收敛版主设计已移至 [Goal_Constructor_LLM_Init_Heuristic_Maintain_Design_2026-09-01.md](./Goal_Constructor_LLM_Init_Heuristic_Maintain_Design_2026-09-01.md)；本文降级为**候选信号池与文献出处参考**（主设计文档 §5 指向本文 §2–§3、附录 A）。

## 0. 要回答的问题

教授方案的分支一是"用便宜启发式抽取重要实体，然后判断 previous goal 是否仍然有效"。本文回答三个问题：

1. **已纳入文献里，"是否更新 goal"这个判断实际上是怎么做的**——每篇论文的具体机制与留下的空白；
2. **survey 之外还有哪些文献族可以直接作为规则来源**——话语主题分割、在线漂移/变点检测、goal drift 评测、循环检测、选择性分类/弃权；
3. **keep/patch/rewrite 的规则怎么写**：特征层、信号层、决策策略、参数标定、验证方式，以及每条规则对应的文献出处。

## 1. 已纳入文献中"判断机制"的具体解剖

这一节从 survey 主表中抽取每篇工作的 *decision mechanism*，因为你要构建的 gate 正是这些机制的低成本组合。

| 工作 | "是否更新"判断机制 | 规则形态 | 可复用点 | 留下的空白（你们填） |
|---|---|---|---|---|
| STITCH | LLM 每步读 `s_t + recent buffer + σ_{t-1}`，判断是否出现 "boundary event (goal-state divergence)"；未检测到边界则 `σ_t` 继承旧 scope | 生成式判断（GPT-5-mini 每步一次） | **决策语义**："常量保持直到发散事件"；粒度消融显示 under-segmentation 最伤 Type 2、over-segmentation 最伤 Type 3——说明阈值本质是**分割粒度旋钮** | 判断本身不便宜；用规则/统计替代每步 LLM，SLM 仅在触发时调用 |
| PMCoder（已全文核实，见附录 A） | 任务开始一次 LLM 拆 typed subtasks（标 target phase，解析失败回退固定骨架）；此后**零成本规则式 phase detector** 每步读最后归一化 shell command + 模型 reasoning：file-inspection/探索性语言→exploration、诊断命令/root-cause 语言→hypothesis、文件修改动作→implementation、跑测试/验证语言→verification（有序规则：修改动作优先于 inspection）；前进立即接受，回退/横移走 **2 步滞回**；10 步 floor 后按四个硬信号判 stuck | 纯规则状态机 | **整套 coding-agent 特征集** + 已核实的阈值表（原文 Table II）+ recovery goal stack（深度上限 16、revert-then-refix 模板、8 步冷却）+ phase→retrieval budget/diversity 映射 + subtask keywords 作检索 anchor | 维护的是 phase/subtask stack，没有输出供检索用的结构化 Goal；原文自认消融未分离 execution grounding 与 edit-integrity recovery 的贡献 |
| Proactive Memory Agent | 每 `N` 步固定周期，并行 memory agent 读原任务+窗口+bank，输出干预提醒或 `<no_intervention/>` | 周期触发 + 生成式判断 | **弃权形态**：`no_intervention` ≈ KEEP，提醒 ≈ PATCH；说明"不更新"应是一等公民输出 | 周期固定、不事件触发；论文自认学习 invocation timing 是未来工作 |
| TSR | 比较 action 前后 screenshots 差分，用 transition-aware verifier 确认/否认 subgoal 进度 | 环境差分 + verifier | **证据原则**：进度确认靠可验证的环境差分，而非 agent 自述 | coding 对应物（return code / test verdict / diff）需自己定义 |
| ContextFlow | stage contract 出现不一致时 scoped update，动词集为 continue / refine / transfer / promote / repair | 显式状态转移语义 | 为三态提供更细的转移语义（见 §3.3） | 面向 embodied 对齐，不服务检索 |
| Goal-Mem | 证据不能与 subgoal unify 时才继续分解/检索 | 证据驱动的判断 | "证据缺口"作为更新触发条件的思想 | 多轮分解太贵，不适合每步 |
| Goal drift 评测（arXiv:2505.02709，survey 未纳入） | 行为度量（持仓变化）+ LLM judge 分类最终目标 | 离线评测而非在线判断 | **漂移机制发现**（见 §2.3） | 它是评测方法，恰好可借来给你们的 gate 做离线标注 |

**结论**：没有任何一篇给出完整的 keep/patch/rewrite 规则表。现有文献分别给了 (a) 特征集（PMCoder）、(b) 决策语义（STITCH 的 boundary、ContextFlow 的转移动词）、(c) 确认证据（TSR 的 diff）、(d) 弃权形态（PMA 的 `no_intervention`）。你们的贡献空间正是把它们组装成一个单一、便宜、可审计的 gate。

## 2. 补充检索：survey 之外的规则来源文献族

### 2.1 话语/主题分割：lexical cohesion 作为边界信号

- **TextTiling**（Hearst, 1997）：对相邻词块计算词汇重叠/余弦相似度，对相似度序列的"低谷"做 depth score，谷深超过阈值即为子主题边界。
- **C99**（Choi, 2000）：对块间相似度矩阵做 rank 变换后再找最优分割，比原始相似度更鲁棒。
- **词汇衔接理论**（Halliday & Hasan 的 cohesion；Grosz & Sidner 的 intentional structure）：段落边界对应衔接链的断裂；近期对话分割工作（如 CobSeg 一类的 coherence boundary modeling）沿用这一范式，只是换成了学习式边界模型。

**翻译为规则**：segment boundary = cohesion drop。对 coding agent，最强的衔接锚点是**精确实体**（文件路径、符号、错误类型）而非通用词汇。因此规则形态是"新 observation 的实体集与当前 Goal 实体集的重叠率骤降"→ 边界证据。这条路比 embedding 便宜，且对 file/symbol 这类 exact handle 更可靠——这同时呼应 survey §8 第 3 条（embedding 相似不等于依赖，lexical baseline 不应弱化）。

### 2.2 在线漂移检测与变点检测：不要单步判断，要累积判断

- **DriftHunter**（intent drift detection）：把意图漂移建模为跨 turn 的**时间转移累积**（可调参数 α, β, γ），而不是逐轮独立分类——说明单步信号噪声大，需要时序积累。
- 短文本流漂移检测（ACL 2023 industry track 有专文做"reliable and interpretable drift detection in streams of short texts"，具体方法细节建议按篇核对）：共性是 embedding 质心漂移 + 可解释触发。
- **经典在线变点检测**（CUSUM / EWMA 族）：监控统计量偏离基线的**累积和**超过阈值才报警；两个超参（漂移率阈值、累积阈值）可以直接控制误报率。

**翻译为规则**：把语义相似度 `s_t = cos(embed(O_{t+1}), embed(G_t))` 当作监控统计量，用 EWMA/CUSUM 而不是单步固定阈值触发 REWRITE。这直接解决"单轮 observation 措辞抖动导致误判"的问题，也让"误重写率"成为可控设计指标而非碰运气。

### 2.3 Goal drift 评测文献：给阈值设计提供机制依据

*Technical Report: Evaluating Goal Drift in Language Model Agents*（arXiv:2505.02709）的关键发现：

1. 漂移随 context 长度与"临时目标阶段"长度**增加**；
2. 显式、强化的 goal 锚定能显著降低漂移；
3. 失败更多来自"**没有撤销旧状态**"（inaction drift），而非做了错误的新动作；
4. 对旧行为的 pattern-matching 比单纯的 token 距离更能解释漂移。

**翻译为规则**：
- (1) → Gate 应有**账龄机制**：Goal 存活越久、轨迹越长，KEEP 的判据越要收紧，或触发周期性重校准（对应 §3.2 的 D7）；
- (2) → root objective 永不被覆盖，且应周期性重新注入（与 survey §2.4 一致，现在有了独立文献依据）；
- (3) → PATCH 必须包含"**回归父任务**"操作：recovery/backtrack 分支完成、证据重新支持原假设时，要能确定性回到父 Goal——这不是新目标生成，而是状态恢复；
- (4) → 实体/状态连续性（而非纯语义相似）应主导 KEEP 判定。

### 2.4 循环 / stuck 检测

- *When Agents Do Not Stop: Uncovering Infinite Agentic Loops in LLM Agents*（arXiv:2607.01641）：提出 IAL-Scan，把根因归结为"循环反馈区缺少有效 bound"。它是静态分析，但运行时对应物就是：**重复调用且无进展信号**。
- 运行时启发式：PMCoder 已给出核实过的四信号版本（失败返回码、单文件过度编辑、只读饱和、归一化 action 重复，见附录 A）——共同点是"重复调用 + 无进展证据"，且都需要最小步数下限与冷却期来防误报和重规划风暴。

**翻译为规则**：stuck 是一个独立的硬触发（§3.2 D4），处理动作是 push recovery goal（PMCoder 的 recovery goal stack），而不是泛泛地"重写"。这给了 REWRITE 一个模板化子类型，无需自由生成。

### 2.5 选择性分类 / 弃权：gate 的成本不对称理论

- **Selective classification**（Geifman & El-Yaniv）：分类器可弃权，用 coverage-risk 曲线刻画质量；
- **Learning to defer**（Madras et al., 2018）：低置信样本转交更强（更贵）的预测器，训练/设计时显式建模转交成本；
- **Conformal prediction**：给出校准后的覆盖率保证。

**翻译为规则**：你们的三态决策存在强不对称成本：

- **false KEEP**（Goal 已失效却复用）→ selector 用错误 Goal 选错历史 page → 任务质量损失，且因 active-only prefill 会把错误编码进新 KV（survey §5.3、two-pager §2.4 的 representation drift），**代价高且会累积**；
- **false REWRITE** → 多一次 <1B SLM forward，几十毫秒级，**代价低**。

因此 gate 的默认偏置必须是"**拿不准就升级**"：KEEP 需要满足较强的联合条件，而 REWRITE 由较弱的任一触发即可。τ 阈值按这个不对称性设置，PMA 的 `<no_intervention/>` 是这一思想在 agent 文献里的形态先例。

### 2.6 会话查询改写：为什么 KEEP 必须是默认态

CONQRR / ZeQR 一类的做法是**每轮都改写**查询——对应到你们这里就是每轮 SLM rewrite（教授的 (iii) 配置），成本高且大量改写是无意义的。面向检索的会话查询编辑工作则研究"**是否**需要编辑"这一前置判断。这条文献支持：默认态是 KEEP（旧 Goal + 追加新实体进 selector query），改写必须由事件触发，而不是周期性发生。

## 3. 规则构建：四层设计

### 3.1 特征抽取层（纯规则，CPU 侧，确定性）

对每个 `O_{t+1}`（连同 `a_t`）抽取：

| 特征类 | 抽取方式 | 文献对应 |
|---|---|---|
| `files` | 路径正则（含 diff 头 `a/`、`b/`、`+++` 行） | PMCoder 特征集；Structured Distillation 的 `regex files_touched` |
| `symbols` | tree-sitter/ctags 标识符；最低配：camelCase/snake_case 标识符 + stack frame 函数名 | Paritok-4B 的 identifier/line-number preservation |
| `errors` | 异常类型（`XxxError:`/`Exception` 行）+ top-3 stack frames | Paritok 的 error 保留；PMCoder reproduction verdict |
| `tests` | 测试名、通过/失败计数 | PMCoder verification phase |
| `commands` | tool name + 归一化命令模板（去掉字面量） | PMCoder 最后 shell command 特征 |
| `verdict` | return code、test pass/fail、是否产生有效 diff | TSR 的环境差分证据原则 |
| 元数据 | turn 类型、距上次 Goal 变更步数（账龄）、plan stack 深度 | 新增；账龄依据 §2.3 |

成本要求：整层必须是正则/解析器级（微秒—毫秒），显著低于 SLM 一次 forward，否则失去"便宜分支"的意义。

### 3.2 信号层：离散触发 + 连续统计

**离散触发（硬规则，直接进决策表）：**

- **D1 新错误证据**：新异常类型或 top-frame 变化（不在 `G.entities.errors` 中）——当前假设可能被证伪；
- **D2 tool phase 转移**：在 `inspect → edit → test → debug` 状态机上，合法后继转移视为延续，跨阶段跳转（如 test→inspect 新文件）触发；
- **D3 阻塞解除**：曾阻塞的命令 return code 归零 / 之前失败的测试转绿；
- **D4 stuck**：PMCoder 原文核实了四个可直接复用的信号——持续非零 return code（失败返回 ≥5）、单一文件被反复编辑（≥6 次）、只读饱和（≥15 次 read 无 edit）、归一化 action 重复（≥8 次）；另加最小步数下限（10 步，防早期误判）与触发后 8 步冷却（防重规划风暴）。卡住时标记当前 subtask 失败，push 模板化 recovery goal 到深度受限（16）的 goal stack，含 **revert-then-refix** 模板：先把高改动文件还原到已提交状态，再从干净基线重修（delta-debugging 原则）；
- **D5 active subtask 完成**：当前 subtask 的 verifier 通过（测试绿、reproduction verdict 成立）→ 从初始化时的 typed subtask stack 取下一个；
- **D6 用户/系统新需求**：交互式场景下新的 requirement 消息；
- **D7 Goal 账龄检查点**：距上次 Goal 变更 ≥ N 步时周期性重校准（PMA 式安全网，防慢速漂移，依据 §2.3 的发现 1）。

**连续信号（各归一到 [0,1]）：**

- **c1 实体重叠**（词汇衔接信号，§2.1）：`|E_{t+1} ∩ G.entities| / |E_{t+1}|`；
- **c2 语义相似度**：`cos(embed(O_{t+1} summary ⊕ E_{t+1}), embed(G.objective ⊕ G.subtask))`，embedding 用 0.1–0.3B 级小模型即可，无 GPU 预算时可整体降级为纯规则+词汇；
- **c3 累积漂移统计量**（§2.2）：对 c2 序列做 EWMA 或 CUSUM；单步不判，报警需持续 m 步；
- **c4 phase 一致性**：当前 tool 类型是否属于当前 subtask 的预期 tool 集合。

### 3.3 决策层：决策表 + 滞回 + 不对称成本

```text
# 优先级从高到低
D6                      → REWRITE(user_request)
D4                      → REWRITE(stuck_recovery)        # 模板化，不生成（§3.5）
D5                      → PATCH(next_subtask_from_stack) # 确定性弹出，不生成
D1 或 D2 或 D3:
    c1 ≥ θ_hi 且 c2 ≥ τ_keep → PATCH(实体/状态更新)
    否则                     → REWRITE(hypothesis_invalidated)
其余:
    c2 ≥ τ_keep 且 c1 ≥ θ_lo 且 c3 无报警 → KEEP（E_{t+1} 追加进 selector query）
    c3 报警（或 c2 < τ_esc）              → REWRITE(semantic_drift)
    其余                                  → PATCH(轻微实体/状态更新)
```

三要素：

1. **滞回（hysteresis）**：`τ_esc < τ_keep`，且 c3 报警需持续 m 步——PMCoder 对回退/横移的滞回 + 变点检测的持续确认，共同防止边界帧上 KEEP/REWRITE 抖动。STITCH 的粒度消融（§1）说明这个旋钮直接决定 under/over-segmentation，必须消融报告。
2. **不对称成本**（§2.5）：KEEP 要求联合条件全满足（偏严），REWRITE 任一触发即可（偏松），因为 false KEEP 代价 ≫ false REWRITE 代价。
3. **ContextFlow 语义细化**（可选报告维度）：continue≈KEEP；refine≈PATCH；transfer/promote≈subtask 级 REWRITE；repair≈stuck_recovery REWRITE。

### 3.4 PATCH 的确定性字段规则

PATCH 必须是字段级确定性更新，不产生自由文本：

- `entities`：并集追加 `E_{t+1}`；可选衰减——连续 k 轮未出现的实体降级为弱候选（不删除）；
- `status`：blocked → active（D3 解除）；verifying → done（verdict 通过）；
- `subtask`：**只允许**经 D5 从 plan stack 弹出或经 REWRITE 更换，禁止自由文本修改；
- `constraints`：**只有** user/system 消息可添加（依据 §2.3 发现 2：agent 自述的"新约束"正是漂移的常见入口）；
- `root_goal`：永不覆盖（survey §2.4 + §2.3 共识）；
- 回归操作：recovery goal 完成后，`parent_goal_id` 指回父 Goal 并恢复其字段快照（依据 §2.3 发现 3 的 inaction drift）。

### 3.5 REWRITE 分支：多数可以绕过生成

REWRITE 触发后按子类型分流：

- `stuck_recovery`、`next_subtask`（若 D5 被归入重写统计）→ **模板化**：从 recovery goal stack / plan stack 直接构造结构化 Goal；
- `hypothesis_invalidated`、`semantic_drift`、`user_request` → 调用 <1B SLM，输入为 `G_t, S_t, a_t, O_{t+1}(截断), E_{t+1}, 触发类型, plan stack 剩余`，输出 survey §5.1 的 YAML schema；要求保留 `parent_goal_id` 并复用既有实体词表（维持 goal-turn 图的连通性）。

这样"rewrite rate"与"generation rate"是两个不同的可报告数字，也是本方案相对教授原始二分法的直接成本优势。

## 4. 规则怎么"构建出来"：标定与验证流程

1. **冷启动标注**：在 SWE-bench 类轨迹上，用 LLM-judge（STITCH 式 scope 标注）或人工标注 goal 边界帧，形成 boundary ground truth；arXiv:2505.02709 的 judge 流程可直接借用；
2. **Shadow mode**：gate 与完整 agent 并行运行，只记录决策不干预（survey §7.3 的 shadow study）；
3. **阈值标定**：按不对称代价 `λ·FN(漏检漂移) + FP(误重写)` 标定 τ/θ，`λ ≫ 1`；c3 的 CUSUM 参数可改用目标误报率反推；
4. **粒度敏感性**：对不同 (τ, θ) 同时报告 gate 的 change-detection F1 **和**下游 turn Recall@budget——用下游指标选工作点，而不是只看 gate 自身 F1（STITCH 粒度消融的教训）；
5. **规则归因消融**：逐族开关规则（D1–D7、c1–c4、滞回、D7 周期检查），否则审稿人会问"为什么是这些规则"；
6. **与教授配置对齐**：G1（纯规则）/ 规则+主模型一次 yes-no（教授原述版本）/ G4（规则+SLM rewrite）/ (iii) 每轮 SLM —— 四档同测。

**默认初值**：连续信号阈值仍取人工起点 `τ_keep≈0.75，τ_esc≈0.45，θ_hi≈0.5，θ_lo≈0.2`；离散阈值直接采用 PMCoder 已核实的配置（附录 A）——滞回 m=2、stuck 四信号（失败返回 ≥5 / 单文件编辑 ≥6 / 只读饱和 ≥15 / action 重复 ≥8）、floor 10 步、冷却 8 步、goal stack 深度 16；D7 的 N=20 暂无文献依据，标定后替换。

## 5. 文献 ↔ 规则映射总表（桥梁本体）

| 规则 / 信号 | 对应文献 | 文献提供了什么 | 本设计的改造 |
|---|---|---|---|
| 特征集（files/symbols/errors/commands/verdict） | PMCoder；Paritok-4B；Structured Distillation | coding-agent 执行特征与实体保留优先级 | 合并为结构化 `E_t`，正则/解析器级抽取 |
| phase 状态机（D2） | PMCoder 的 4-phase；ContextFlow 转移动词 | 阶段语义与确定性维护 | 简化为 tool-phase 状态机 + 决策表 |
| "常量保持直到发散事件"（KEEP 默认） | STITCH | 决策语义；粒度权衡证据 | 用规则+连续统计替代每步 LLM；SLM 仅在触发时出场 |
| 环境差分证据（D3、D5） | TSR | verifier-triggered 进度确认 | coding 版：return code / test verdict / diff |
| 弃权与干预形态 | Proactive Memory Agent | `<no_intervention/>` 一等输出 | KEEP 为默认态 + D7 周期安全网 |
| 实体重叠 c1 | TextTiling；C99；词汇衔接理论 | cohesion drop = boundary | 用 exact entity overlap 代替通用词汇重叠（更便宜、更准） |
| 累积漂移 c3 | DriftHunter；短文本流漂移检测；CUSUM/EWMA | 时序累积变点检测，控制误报 | 对 c2 序列做 EWMA/CUSUM，持续 m 步才报警 |
| 账龄检查点 D7、root 锚定、回归操作 | arXiv:2505.02709 | 漂移随长度增长；锚定有效；inaction drift | D7 周期重校准；root 不可变；PATCH 含父任务回归 |
| stuck → recovery（D4） | PMCoder recovery stack；IAL-Scan（arXiv:2607.01641） | 循环识别 + recovery goal 概念 | 模板化 recovery goal，不自由生成 |
| 不对称阈值、拿不准就升级 | selective classification；learning to defer；conformal prediction | coverage-risk 理论框架 | τ_keep/τ_esc 不对称设置；SLM 升级作为"更贵预测器" |
| 计划栈确定性子任务切换（D5） | PMCoder 的一次性 typed 分解 | 预建计划是可复用的 goal 来源 | subtask 更换零生成成本 |
| KEEP 默认、改写需触发 | 会话查询改写（CONQRR/ZeQR 反例；search-oriented query editing） | 每轮改写的代价证据 | 事件触发制，非周期制 |

## 6. 注意事项与风险

- **时序硬约束**：gate 全部输入必须在 turn boundary 可得，不读主模型 KV（survey §5.3 的双-prefill 论证）；
- **特征抽取成本预算**：若正则/解析层意外变贵（如 tree-sitter 全量解析大 diff），退化为行级正则，宁可漏符号不漏文件；
- **embedding 是可选项**：c2/c3 在没有 GPU 余量时整体砍掉，gate 退化为"硬规则 + 词汇重叠"，这本身是合法的消融配置（也正好回应教授"先单信号再混合"的实验顺序）；
- **阈值不要一步端到端学**：先人工默认值 → 标注后标定 → 最后才考虑把决策表换成学习式分类器（作为上界对照，不作为主方案）；
- **PATCH 与 REWRITE 的统计口径**：主实验按教授的二分法报告 `REUSE={KEEP,PATCH}` vs `REWRITE`，三态作为消融，避免与教授框架错位。

## 7. 一句话总结

keep/patch/rewrite 的启发式规则 = **PMCoder 的执行特征集** × **STITCH 的边界决策语义** × **TSR 的差分证据** × **话语分割的衔接信号** × **变点检测的累积判定** × **选择性分类的不对称升级**。离散触发进决策表，连续统计加滞回防抖，SLM 只在 REWRITE 分支出现，而 REWRITE 的大部分子类型还能被计划栈/恢复模板短路——这就是把教授的 (ii) 配置落到代码层面的完整桥梁。

## 8. 附录 A：PMCoder 全文核实的机制与参数（2026-09-01，arXiv:2608.06811）

以下为从全文逐段核实的机制，替代此前基于 survey 摘要的转述：

**规划器**

- 任务开始**一次** LLM 调用：把 issue 拆为 typed subtasks，每个标注 target phase；解析失败回退到固定骨架（不静默失败）。这是规划器唯一的 LLM 调用；此后逐步 phase 检测、subtask 记账、回溯、重规划全部确定性。
- **Phase detector**（零成本、规则式）：每步对"归一化后的最后 shell command + 模型自述 reasoning"应用有序规则，覆盖四类信号——文件查看行为与探索性语言→exploration；诊断命令或 root-cause 语言→hypothesis；文件修改动作→implementation；测试执行或显式验证语言→verification。有序规则中修改动作优先于查看；一次性诊断执行记为 hypothesis。
- **滞回**：detector 便宜但噪声大，因此前向（phase rank 前进）立即接受，回退/横移需要**重复证据（2 步滞回）**才生效。
- **Stuck 判定**：最小 10 步 floor 之后，任一信号触发——持续非零 return code（失败返回 5 次）、单一文件编辑过多（6 次）、只读饱和（15 次 read 无 edit）、归一化 action 重复超阈（8 次）。多个信号由 memory 子系统以聚合统计形式算好传给规划器。
- **回溯处理**：标记 active subtask 为 failed → push recovery subtasks 进 goal stack（深度上限 16），其中包括 execution grounding 的 **revert-then-refix** 模板（还原被编辑文件到已提交状态，再从干净基线重修；引用 delta-debugging 原则）。**冷却 8 步** + 深度上限防止长轨迹上重规划风暴。
- **完成判定**：非终结 subtask 在检测 phase 越过自身 phase 且证据最小时完成；终结性验证 subtask 只能靠"成功的验证步"完成，且被 execution grounding 加固——return code 为 0 只有在最新 reproduction verdict 不是 fail 时才算证据（问题仍可复现则验证不能通过）。

**Execution grounding**

- 对有已验证 issue 复现脚本的实例：每次文件修改类编辑后重跑复现脚本，verdict 为 pass（不再复现）/fail（仍复现）。
- verdict 有两个消费者：作为当前 tool result 上的 advisory 文本告知模型；作为内部 plan state 更新加固完成谓词。
- Python 文件修改后做编译检查，若不可解析则附加纠错提示，在错误叠加前暴露损坏状态。
- 所有 grounding 信号均为 advisory：不阻塞动作、不回滚编辑、不参与官方评分（harness 只看到提交的 patch）。

**记忆与检索**

- 记忆节点除文本外带结构化元数据：消息角色、位置 ID（=recency）、压缩视图、摘要、前一命令是否编辑文件、触及哪些文件。observation 节点的编辑/文件元数据从**实际执行的前一条 assistant 命令**读取——记忆锚定在 agent 实际跑了什么，而非模型声称跑了什么。图中只存原始 observation；注入内容只改活对话、不改图，避免检索吃回自己的输出。
- **预算化 MMR 检索**：固定 core 永留——task statement、recent tail、**anchor 节点（其查询焦点由 active subtask 关键词播种）**；其余由 beam search 按边际增益加入，`g(v)=λ·rel(v)−(1−λ)·max sim(v,u)`，λ=0.7，相似度为 IDF 加权 Jaccard。
- 相关度融合两路：`rel = w_lex·lexical + w_graph·graph`，权重各 0.5。lexical 是对 anchor 词的 IDF 加权词法分；graph 是**增量构建的代码结构文件图**上的邻近度——同文件共现、Python import 边（AST 解析，失败时保守正则回退）、轨迹邻接。
- **Phase 条件化检索策略**：phase → (预算, 多样性, 图权重)。各 phase 预算为 20k/16k/12k/16k/20k token（exploration/hypothesis/implementation/verification/backtrack）：exploration 与 backtrack 用大预算、偏好多样性（广撒网定位/恢复）；implementation 用小预算、高图权重（紧贴编辑点）。

**注入通道**

- 不改写历史；把紧凑的标记分隔块**追加到最近一条 tool-result 消息**里（plan-state header + top 检索节点，受注入预算约束）——走 tool-calling 模型习惯读的通道，保持 in-distribution；试点观察：tool-result 位置比独立 user-role 提醒更稳定。
- 两个过滤器控增长：**只有 plan state 变化时才注入**；新颖性过滤丢弃已出现过的节点，替换为一行处方式摘要（如失败 subtask 一行总结、文件 churn 计数）。

**结果与消融口径**

- SWE-bench Verified 上平均比 harness 匹配基线多解 25 例；分析显示减少重复失败动作；阈值全部在 pilot 阶段固定，未在最终 headline 上调参。
- RQ4 消融支持 plan–memory coupling，但原文自认**未单独估计 execution grounding 与 edit-integrity recovery 的贡献**——这正是你们做规则归因消融（§4 第 5 条）要避免的口径缺陷。

## 9. 附录 B：本文新增来源（survey 未含）

1. Hearst. *TextTiling: Segmenting Text into Multi-Paragraph Subtopic Passages*, Computational Linguistics, 1997.（主题分割经典）
2. Choi. *Advances in Domain Independent Linear Text Segmentation*（C99）, NAACL 2000.
3. Grosz & Sidner. *Attention, Intentions, and the Structure of Discourse*, Computational Linguistics, 1986.（话语意图结构）
4. *DriftHunter: intent drift detection via differentiable temporal transition accumulation*（GitHub: FDHTJ/DriftHunter）.
5. ACL 2023 Industry Track 短文：*Reliable and Interpretable Drift Detection in Streams of Short Texts*（细节待按篇复核）.
6. *Technical Report: Evaluating Goal Drift in Language Model Agents*, arXiv:2505.02709.
7. *When Agents Do Not Stop: Uncovering Infinite Agentic Loops in LLM Agents*（IAL-Scan）, arXiv:2607.01641.
8. Geifman & El-Yaniv. *Selective Classification for Deep Neural Networks*, NeurIPS 2017.
9. Madras et al. *Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer*, NeurIPS 2018.
10. CobSeg. *Coherence Boundary Modeling for Dialogue Topic Segmentation*, arXiv:2605.30668.（近期学习式对话分割对照）
