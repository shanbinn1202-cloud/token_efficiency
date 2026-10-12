# DRAM-Backed Goal-Conditioned KV Working Set: Core Design Draft

> Status: research rationale retained for reference; this is not the current runbook.

## 1. Core Design

The complete sealed KV history of a session is backed by CPU DRAM. While that history remains below a configured per-session HBM threshold, the system follows the ordinary full-context path. Once the threshold is crossed, the session enters Goal-conditioned mode: before every subsequent turn, it updates the Goal, selects relevant historical Turns/Pages, and restores only their KV to HBM. The suffix prefill and generation use that same active set; newly generated KV is then written back to DRAM for possible selection in later turns.

~~~text
Turn begins: check the full-history KV footprint
        ↓
Goal-conditioned mode already active, or footprint > HBM threshold?
        ├── No  → Full-context: ordinary prefix reuse + prefill/generation
        │                                      ↓
        │                         Seal/write new KV to DRAM
        │                                      ↓
        │                              Check again next turn
        │
        └── Yes → Enter/continue Goal-conditioned mode
                                               ↓
                    Update the Goal from the current observation
                                               ↓
                    Select relevant Turns / KV Pages before prefill
                                               ↓
                              Selected KV: DRAM → HBM
                                               ↓
                         Active-only suffix prefill + generation
                                               ↓
                         Seal new KV/state: HBM → DRAM
                                               ↓
                                  Repeat for every later turn
~~~

The primary benefit is a smaller HBM working-set footprint for each active session, enabling higher concurrency and aggregate throughput under a fixed GPU memory capacity. The method does not promise a substantial reduction in single-request decoding latency.

## 2. Preliminary Module Design

### 2.1 DRAM/HBM KV Hierarchy

- DRAM stores the complete, sealed, and restorable historical KV. Unselected Pages are not deleted and can be restored later when the Goal changes.
- HBM stores only the selected and mandatory historical KV for the current turn, and the current live tail.
- Logical Page IDs, original token positions, and RoPE positions remain unchanged. Restoration changes only the mapping to physical GPU blocks.

Removing Pages from the active block table reduces attention reads but does not reduce HBM allocation unless their KV is backed in DRAM and the corresponding GPU blocks are returned to the shared pool.

The cross-turn lifecycle: new live HBM KV → seal/offload to DRAM → retain mandatory/recent and reused Pages → release other GPU blocks as needed → restore only newly selected Pages from DRAM.


### 2.2 Goal Constructor

After a session crosses the HBM threshold, the Goal Constructor runs at every turn boundary and produces or refreshes the Goal **before** the main model's formal prefill.
If the Goal is constructed after the main-model prefill, the system must either load the complete historical KV into HBM or scan the complete KV from DRAM, reintroducing the memory or transfer overhead that the design is intended to avoid.

The constructor may use the task objective, previous Goal, current observation, and maintained structured state. Potential implementations include heuristics, a small model, Agent self-reporting, or a hybrid; the exact method remains open. The main questions are:

- Who constructs the Goal: a CPU-side component, an independent small model, or the main Agent?
- Must the constructor read the complete raw trajectory every turn, or should it maintain a continuously updated structured Goal/state?
- Is the Goal used only as a selector-side input, or is it also injected into the main-model prompt?

#### 2.2.1 Goal 判断方案与 Action/Observation 时序

记 $a_t$ 为第 $t$ 步生成的 action，$O_{t+1}$ 为环境执行 $a_t$ 后返回的 observation，$G_{t+1}$ 为下一步 Goal，$A_{t+1}=\text{Mandatory}\cup\text{Recent}\cup\text{GoalSelected}$ 为下一步历史工作集。标准时序为：生成 $a_t$ → 获得 $O_{t+1}$ → 判断并更新 $G_{t+1}$ → 选择 $A_{t+1}$ → 保留 $A_t\cap A_{t+1}$、恢复新增 Pages → 在 $KV_{A_{t+1}}$ 上 prefill $O_{t+1}$ → 生成 $a_{t+1}$。当前 observation 是 live suffix，处理完成后才 seal 为可供后续选择的历史 Page。

**方案 A：主 Agent 判断。** 在释放上一轮 HBM working set 前，主 Agent 使用仍驻留的 Goal-selected/recent Pages，即 $KV_{A_t}$，prefill $O_{t+1}$ 和 Goal-validity question，并解码 yes/no。其 observation KV 有两种处理方式：

- **不复用：** 丢弃检查分支的 $KV(O_{t+1}\mid A_t)$；得到 $A_{t+1}$ 后，在新工作集上重新 prefill $O_{t+1}$。该方式保持严格的 selection-before-prefill 语义，但 observation 被 prefill 两次。
- **复用：** 保留 $KV(O_{t+1}\mid A_t)$，更新工作集后直接生成下一 action。这样只 prefill 一次，但新恢复的 Pages 未参与 observation representation，实际语义为 $\operatorname{Generate}(\cdot\mid KV_{A_{t+1}},KV(O_{t+1}\mid A_t))$，需要单独验证其漂移和任务质量。

**方案 B：轻量 Gate 判断。** 使用独立小模型或分类器计算 $z_{t+1}=\operatorname{Gate}(G_t,S_t,a_t,O_{t+1},E_{t+1})$，其中 $S_t$ 是结构化状态，$E_{t+1}$ 是从当前 observation 提取的实体。该方案仍需要短 prompt prefill，但不读取历史 selected-page KV；确定 $G_{t+1}$ 和 $A_{t+1}$ 后，主 Agent 只正式 prefill 一次 $O_{t+1}$。即使复用旧 Goal，$E_{t+1}$ 和 observation 摘要仍应加入 selector query。

#### 2.2.2 基于 PMCoder 的 Goal Constructor：一次初始化、启发式维护、条件式 SLM Rewrite

**核心决策。** 判断与生成分离：任务开始时只调用一次 LLM，从锁定的 `root_goal` 构造供 selector 使用的初始 `current_goal`；后续每轮只用规则维护状态并判断该检索目标是否仍有效。正常执行不改检索目标，只有规则无法处理的语义变化才调用 <1B SLM，且 SLM 不能改 `root_goal`。

```text
t = 0（一次性）:
  LLM(issue/task) → root_goal（锁死）+ 初始 current_goal G_0

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

**状态检测器。** 输入是最后一条规范化命令、Agent reasoning 中的阶段词、return code、diff/文件修改，以及重复失败/编辑/读取/action 的计数。按有序规则判断状态并切换 KV 工作集selection策略：

- `read/list/search` 或探索性语言 → **探索** → 较宽、较多样的工作集；
- 诊断命令、错误/stack trace 或 root-cause 语言 → **假设** → 提高错误、符号和诊断证据权重；
- `edit/write/patch/diff` → **实现** → 聚焦当前文件和依赖邻居；
- 测试命令及 test/reproduction verdict → **验证** → 聚焦测试、修改记录和先前验证证据；
- 重复失败、反复编辑/读取或重复 action → **停滞** → 放宽工作集并提高多样性。

**有效性 Gate。** `KEEP` 原样复用当前 retrieval goal 和 selector query；`PATCH` 只更新结构化字段；`REWRITE` 才调用 SLM。初版 heuristic 由四类 if–then 规则组成：

| 规则类           | 主要信号                                                              | 典型判定                                                                            |
| ------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **实体连续性**     | 当前 observation 与 Goal 中的文件、符号、测试、错误类型、stack frame 的 exact overlap | 核心实体基本延续 → `KEEP`；同一目标下出现相关新实体 → `PATCH entities`；核心实体持续整体切换 → `REWRITE`        |
| **执行与验证证据**   | return code、test/reproduction verdict、有效 diff、错误类型或 top frame 变化  | 证据仅更新进度 → `PATCH status/evidence_needed`；新证据直接推翻当前检索假设 → `REWRITE`              |
| **阶段与循环模式**   | inspect→diagnose→edit→test 的 action 类型，以及重复失败、编辑、读取和 action 计数    | 合法阶段推进 → `KEEP`，只切换 KV 策略；超过停滞阈值 → `PATCH status/evidence_needed`，并切换为更宽的 KV 策略 |
| **持续漂移与来源约束** | 实体重叠/语义相似度的多轮下降、Goal 账龄、显式 user/system 新需求                        | 单轮弱信号不升级；连续多轮越界或 user/system 修改需求 → `REWRITE`；Agent 自述不能修改 `root_goal` 或硬约束     |

**SLM Rewrite。** SLM 不读取完整 trajectory，也不需要在 GPU 上保留完整 trajectory KV。输入严格限制为：`root_goal`、当前 `retrieval_goal/state`、最后 action、截断后的最新 observation、抽取实体、触发类型和紧凑统计；输出是新的结构化 `current_goal`、相关实体和 `evidence_needed`。历史细节仍由后续 selector 从 DRAM-backed memory 中选择和恢复，而不是先灌入 SLM。

##### English Version

**Core design.** At task initialization, a single LLM call constructs an initial `current_goal` for the selector from the immutable `root_goal`. At turn boundaries, heuristic rules maintain the state and determine whether the retrieval Goal remains valid. A <1B SLM is invoked only for semantic changes that cannot be handled by the rules, and it is not allowed to modify the `root_goal`.

```text
t = 0 (one time):
  LLM(issue/task) → immutable root_goal + initial current_goal G_0

t > 0, at each turn boundary (before the main-model prefill):
  O_{t+1} arrives
       ↓
  Rule/parser-based feature extraction:
  E_{t+1}, return code, diff, normalized command
       ↓
  ├─ State detector → state ∈ {exploration, hypothesis, implementation,
  │                            verification, stuck}
  │        → map state to KV working-set parameters
  │          (slack threshold / budget / diversity / structural weight)
  │          [does not modify the Goal]
  │
  └─ Goal-validity gate → KEEP / PATCH / REWRITE
           KEEP    reuse G_t unchanged; keep the selector query unchanged
           PATCH   update entities / status / evidence_needed
           REWRITE use a <1B SLM to rewrite only the retrieval Goal,
                   anchored to the immutable root_goal
```

**State detector.** Its inputs are the latest normalized command, phase-related terms in the Agent's reasoning, return codes, diffs or file mutations. Heuristic rules map these signals to a state and then to a KV working-set policy:

- `read/list/search` or exploratory language → **exploration** → use a broader and more diverse working set;
- diagnostic commands, errors/stack traces→ **hypothesis** → increase the weight of errors and diagnostic evidence;
- `edit/write/patch/diff` → **implementation** → focus on the current files and their dependency neighbors;
- test commands and test/reproduction verdicts → **verification** → focus on tests, edit records, and previous verification evidence;
- repeated failures, edits, reads, or actions → **stuck** → broaden the working set and increase diversity.

**Goal-validity gate.** `KEEP` reuses the current retrieval Goal and selector query unchanged. `PATCH` only updates structured fields. `REWRITE` invokes the SLM to rewrite the retrieval Goal. The initial heuristic consists of four auditable classes of if–then rules:

| Rule class                                  | Main signals                                                                                                          |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Entity continuity**                       | Exact overlap between the current observation and Goal entities: files, symbols, tests, error types, and stack frames |
| **Execution and verification evidence**     | Return codes, test/reproduction verdicts, valid diffs, and changes in error type or top stack frame                   |
| **Phase and loop patterns**                 | Action types along inspect→diagnose→edit→test, plus counters for repeated failures, edits, reads, and actions         |
| **Persistent drift and source constraints** | Multi-turn decline in entity overlap or semantic similarity, Goal age, and explicit new user/system requirements      |

**SLM Rewrite.** The SLM does not read the complete trajectory. Its input is limited to the `root_goal`, current `retrieval_goal/state`, latest action, latest observation, extracted entities... It outputs a new structured `current_goal`, related entities, and `evidence_needed`.

### 2.3 Pre-Prefill Selector

The previous design could use model attention scores after a full-history prefill. Here, selection must occur before restoration and prefill, while attention backends such as FlashAttention do not expose attention matrix. Reconstructing those scores would add a full-history scan, so a pre-prefill proxy is needed.

The online selector is more likely to use signals available on the DRAM/CPU side, such as:

- lexical or metadata matches involving files, symbols, errors, tests, and similar exact handles;
- semantic retrieval between the Goal and Turn/Page summaries;
- `lightweight, precomputed model-derived representations`: when the KV of a Page is first produced, its keys or hidden states can be pooled into a small vector; selection compares only these compact vectors rather than scanning the complete KV.

The design principles are to protect mandatory constraints, use recency as a safety prior, and retain confidence-aware expansion, a recent-context quota, or whole-Turn fallback to reduce the risk of selector misses.

#### 2.3.1 页级语义索引与工作集调度机制

针对页级工作集调度场景，**无需使用 KV 层构建检索目标，嵌入（Embedding）层是最优方案**。 调度的核心决策是「当前查询该加载哪些 KV 页进入 DRAM」，属于粗粒度的集合选择问题，嵌入层的语义区分度完全可以满足页级匹配的精度需求。

**实现路径（适配分页架构，复用 EpiCache 语义匹配思路）**
**离线阶段**：直接构建页级语义索引
1. 将完整上下文按固定 Token 数切分为物理页，每页对应一段连续文本；
2. 用轻量嵌入模型为**每一页单独计算语义嵌入向量**；
3. 构建「页 ID → 单页嵌入向量 → 该页 KV 物理地址」的一对一映射索引，离线存储。
 **在线阶段**：全页级排序加载
4. 查询到来，生成查询嵌入向量；
5. 直接和索引中**所有单页的嵌入向量**计算余弦相似度，得到每页的相关性得分并全局排序；
6. 按得分从高到低依次加载对应 KV 页进入 DRAM，直到占满工作集预算（预留 Slack 空间
**设计依据**：为什么不使用 KV 层做匹配
KV 向量维度高（每层每个 Token 可达数千维）且需按层分别存储，做页级匹配的计算开销远高于嵌入向量。而在话题 / 页级的判断粒度下，文本嵌入与 KV 内部表示的语义相关性高度一致，准确率提升微乎其微。因此嵌入层调度是精度与开销的最优解。

### 2.4 Active-Only Prefill and Its Risks

Single-turn adaptive-KV methods such as SnapKV perform one complete dense prefill. In this multi-turn design, each later suffix is instead prefilled against only the selected active KV:

$$
\operatorname{Prefill}\left(S_t \mid \mathrm{KV}_{A_t}\right)
$$

Compared with KV eviction, this selection method allows a Page to be reactivated from DRAM in a later turn. However, KV produced under the active set differs from dense full-history KV. The method is therefore a reselectable hierarchical KV memory, not lossless recovery of dense-attention state.

**Main risk:** a selector miss affects not only decoding in the current turn, but also the current suffix hidden states and KV, and may continue to affect the future trajectory.
**Key hypothesis to validate:** Whether this selection-conditioned representation drift will accumulate with the number of turns, and whether reactivating an older Page in a later turn can effectively restore the needed capability.

An initial implementation may require:
- a mandatory/recent safety set;
- expansion of the active budget or restoration of whole Turns when selector confidence is low;
- a full-restore turn as a fallback when necessary.

### 2.5 Integration with Inference Systems Such as vLLM

Standard Automatic Prefix Caching retrieves the longest exact prefix and may not directly match our selected set of relevant historical Pages. The system needs four main additions:


1. **APC to Goal-conditioned non-prefix KV retrieval:** Our design can be viewed as retrieving KV from DRAM storage to HBM active working set based on Goal-conditioned score rather than prefix.
$$\text{DRAM KV Store} \xrightarrow{\text{Goal-conditioned retrieval}} \text{HBM Active KV Cache}$$

2. **DRAM-backed KV lifecycle:** a CPU directory maps logical Pages to DRAM KV locations; the scheduler restores selected Pages after reserving HBM, then offloads new KV/state before reclaiming GPU blocks at the turn boundary.
3. **Active logical-sequence mapping:** selected Pages are assembled in their original position order. Suffix prefill and decode read through this active block table, while new KV writes follow the complete logical sequence so that token positions remain unchanged.
4. **Hybrid-state handling:** for models such as Qwen3.6, DeltaNet/Mamba state must be managed and measured separately; reducing full-attention KV alone does not imply an equal HBM or concurrency gain.
5.
Per-turn latency overhead:

$$
T_{\mathrm{extra},t}
=T_{\mathrm{goal},t}+T_{\mathrm{select},t}
+T_{\mathrm{restore},t}+T_{\mathrm{offload},t}
-T_{\mathrm{overlap},t}.
$$

## 3. Questions Requiring Further Confirmation

> Most open questions stem from the requirement to select historical KV before formal prefill in order to reduce HBM residency:

$$\text{Reduce HBM residency} \Rightarrow \text{select before formal prefill} \Rightarrow \begin{cases} \text{full-context model signals are unavailable for selection},\\ \text{suffix prefill attends only to the active set}. \end{cases} $$

> Three methodological questions therefore require confirmation:
1. **Goal construction:** Since it cannot depend on a full-context forward pass of the main model, should it use a small model, heuristics, or existing Agent state, and should it read full history text or maintain an incremental structured Goal?
2. **Pre-prefill selection:** should it rely on metadata/lexical matching, semantic retrieval, compact model-derived representations, or proxy retrieval followed by candidate reranking?
3. **Active-only prefill:** is its semantic drift acceptable, and what mandatory/recent context and whole-Turn or full-restore fallbacks are required?



If the direction is confirmed, the next steps may be a Goal/selector shadow study and a DRAM/HBM feasibility study.
