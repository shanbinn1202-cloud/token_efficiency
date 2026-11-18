# 方案 B（token 聚集）的按 token 打分：runtime 支持的设计选项

[English](PER-TOKEN-OPTIONS.md) | 简体中文

2026-11-15 的调查，只读：没有为它改任何代码，也没有为它跑 GPU 作业；它引用的 GPU 读数来自更早的已核验作业，注明作业号。token 聚集的 GPU 运行见 [docs/status.zh-CN.md](../status.zh-CN.md)。

> **修订（2026-11-15）**：按一次独立审计（2026-11-15）[S34] 修改。审计列出 11 项必改和 20 项建议：11 项必改全部改了，20 项建议全部采纳，没有跳过。凡是改动过的数字，都来自审计的复测（1 线程和 4 线程的测量，以及填充检查）或本说明的重新推导，来源见附录 C。逐项的改动对照见附录 B。

- **基准**：本仓库的 `integrate-1114` 分支（HEAD `a565e84`，代码 `b388033`）；`main` 的文档取自 `f18b8e5`。
- **来源**：每个 `file:line` 都相对 `integrate-1114` 的仓库根目录，除非该行注明了别的提交。每个事实后的 `[S#]` 对应来源表（附录 C）中的一行。
- **数字的三种标注**：
  - **【GPU 实测】**：已核验的 GH200 作业读数，注明作业号。
  - **【本地代理】**：在一台 x86 Windows 机器上测得（Python 3.12.7、numpy 1.26.4；附录 A，[S32]）。它不是 GH200 上的数，只用来比较量级。测量时没有限制线程数，numpy 的 BLAS 默认用了全部 32 个逻辑核。审计在 1 线程和 4 线程下复测 [S34]：262,144 token 的整数打分慢 1.3–2.3 倍，其他量与原值相差在 1.0–1.4 倍以内，只有 1 线程下 30,496 token 的常驻 f32 打分例外（1.8 倍：0.527 对 0.286 ms）。
  - **【估计】**：由上面两类数字算出，算式写在旁边。
- **规则归属**：按 token 打分的确切规则由 goal_query 研究决定，runtime 负责支持它。下文凡是"规则"都是要问算法侧的内容（第 11 节）。
- **共同前提**：在本 runtime 中，所有选项都只作用于 16 个全注意力（FA）层，并保持逻辑 RoPE 位置；48 个 Gated DeltaNet（GDN）层携带全部历史的递归状态，任何选择都不会碰它 [S18 §2][S19 §2]。

---

## 0. 摘要

**缺口。** goal_query 研究在 8,192 token 预算里先保护 prefix 和最近两轮，再"把剩余槽位给 Goal 分数最高的 token"[S28]。runtime 现在每页只有一个向量，剩余预算按页余弦填整页 [S18 §9][S19 §4]。

**主要发现：**

1. **填充逻辑不用改。**
   - token 聚集现有的挑选器接口加 `shared_blocks_v1` 预算已经能表达"剩余槽位给 token"。当 picks 等于 room × 16 个按分数排好的 token 时，fill 只整页取走 16 个 token 全被选中的页，其余 token 全部聚集，pad 为 0。
   - 本说明用 runtime 自己的代码随机验证了 2,000 次 [S33]；审计独立验证了 3,000 次 × 4 种变体（经 `plan` 和 `turn`，包括标记在前的列表和设了 quota 的情况）[S34]。
   - 真正缺的有三样：每个 token 的分数、大批量聚集的吞吐，以及挑选器拿到 room 的途径（挑选器的 `view` 里没有 room，见 2.1）。
2. **哪项代价最大，取决于选项。**
   - **对选项 1**：聚集（每轮 25–141 ms【估计】）远大于打分。
   - **对选项 2**：读 K（54–868 ms【估计】）与聚集同量级或更大。
   - 聚集量：按 token 填满剩余预算时，每轮最多聚集约 5,051 个 token【估计：room 均值 315.7 页 × 16，[S27]】。这是上限，因为 16 个 token 全被选中的页会改为整页 restore。作为对照，markers 现在每轮只聚集 110.6 个【GPU 实测，作业 3603443，[S24]】。
   - H2D 上限：315.7 MiB/轮【估计】。
   - 141 ms 是用 G1 的两个实测点（144 和 1,024 token）线性外推到 5,051 token 得到的，约为最大实测集合的 5 倍，已超出实测范围【估计】。
   - 聚集在第一个 phase-2 step 内同步执行，同批所有会话都要等这一步 [S18 §4][S26][S36]。
   - G1 的 `UVA_ADVISED` 只是阈值规则（C 集 DRAM 聚集中位 > 10 ms），G1 并没有测过 UVA 路径【GPU 实测，作业 3603240，[S23][S38]】。
3. **选项 1：每个 token 一个向量，与页向量同一空间。**
   - **存储与打分速度的搭配**：
     - 只存二进制的 int8 + scale（132 B/token；262,144 token 的会话 33 MiB）时，整数精确余弦每轮从 int8 转换，30,496 / 262,144 token 分别为 3.2 / 28.7 ms【本地代理，32 核】。
     - 再常驻一份 512 B/token 的 float32 副本（典型会话 14.9 MiB，262,144 token 时 128 MiB），则为 0.29 / 4.2 ms【本地代理，32 核】。
     - 在 1–4 线程下，262,144 token 时分别为 37.8–39.1 ms 和 6.2–9.7 ms【审计复测，[S34]】。
   - **可复现性**：给定 int8 的 query 与 token 向量，分数在任何用 int32 或 fp32 累加的实现上逐位相同。但 int8 query 本身是在 engine CPU 上以 fp32 压缩得到的，GH200 与 x86 在少数情况下末位不同；这正是一致性工具需要 `recover_query` 的原因 [S11][S30]。
   - **不能照搬今天页向量的 Python 对象、JSON 和签名路径**：
     - 那样每 token 约 8 KB，262,144 token 时 1.96 GiB/会话；
     - 每个 token 约 190 个 Python 对象 [S34]；
     - 每轮摄入约 71 ms【估计：本地代理 47.3 µs/向量 × 假设每轮 1,500 个新 token】；
     - 上游逐位一致余弦在 262,144 token 时要 179 ms【本地代理】。
4. **选项 2：直接用 K cache 打分。**
   - 不需要新存储，但 Goal 向量是最后一层的隐藏状态，不是任何 FA 层的 query。必须在 Goal 前向里捕获 16 个 FA 层的 q（192 KiB/Goal），并处理 RoPE。
   - 每轮带宽随历史线性增长：读 16 层 K，典型历史约 54–76 ms，262,144 token 约 620–868 ms【估计；按今天 restore 路径的有效吞吐算，设备直读未测】。
   - K 不进证据，离线只能在记录的分数上重放填充。
5. **选项 3：两段式。**
   - 只对选项 2 有意义：把 K 的读取限制在页余弦前 S 页。S = 2 × room 时 16 层约 25–35 ms，与历史长度无关【估计，上界】。
   - 代价是覆盖率。配选项 2 的覆盖率需要 q·k 分数：来自选项 2 的 probe，或作者提供的逐 token 分数。
6. **选项 4：降维或预计算标量。**
   - 缓存未命中时 Goal 会变；而且捕获时，之后各轮的 Goal 还不存在，所以与 Goal 相关的分数不能在捕获时预计算。
   - 降维只在证据体积或要放进 HBM 时才有意义，作为选项 1 的参数保留。
7. **选项 5：契约、回执、证据和一致性。**
   - 照 token 聚集与 exp-subpage-score 的先例：新契约键只在开启时出现；每 token 向量放在签名体之外。代价是这些向量不受 MAC 认证（页向量受认证）；这是有意的取舍，与"完整性检查只在启动时做"一致。
   - 回执只记聚集的 token（已有）、阈值和 int8 query。
   - 不把每 token 的分数写进签名体：262,144 token 时每请求 5.2 MB，构造加序列化至少 184 ms【本地代理】；engine 的签名和预留时的验证还要各加约 110 ms【审计复测】。
   - goal_query 的完整离线重放要扩展一致性工具，因为它今天拒绝 goal_query [S30]。

**推荐（第 10 节）：**

1. **原型 A：选项 1 的影子模式。**
   - 捕获、传输、存储三部分与规则无关；它记录的统计量（top-k、阈值、重合度）则只针对同空间余弦这一种规则。
   - 它能量出：按 token 的同空间余弦选择与今天页填充差多少（重合度、页外 token 数），以及这种分数下两段式的覆盖率。
   - 是否"更好"只能靠真正改变选择的运行来回答。
2. **原型 B1（与规则无关，现在就做）**：大批量聚集吞吐的 GPU probe。用真实的分布：5,051 个 token 分散在 1,000 多页、多个 arena chunk 上；在有并发 decode 的 step 里计时。
3. **原型 B2（在 A 之后做）**：增量聚集的复用率。哪些 token 会跨轮重复取决于规则，所以要用 A 的影子选择（或某条规则的重放）来量，用合成数据量不出来。
4. 规则确认之后再接挑选器。只有当研究的分数基于注意力（q·k）时，才立项选项 2 和选项 3。

---

## 1. 现状

### 1.1 模型几何（Qwen3.6-27B，[S1][S2]）

| 项 | 值 |
| --- | --- |
| hidden | 5,120 |
| 层 | 64 层：16 个 FA 层（每 4 层一个，层 3, 7, …, 63，从 0 开始计）+ 48 个 GDN 层 |
| 注意力头 | 24 个 q 头，4 个 KV 头（每 6 个 q 头共享一个 KV 头），head_dim 256，缩放 1/√256 |
| RoPE | partial_rotary_factor 0.25：每头 256 维中 64 维旋转；MRoPE interleaved [11, 11, 10]；theta 10,000,000 |
| 写 cache 之前 | q 和 k 先做每头 RMSNorm，再做 RoPE；K 以 RoPE 之后的形式存储 |
| 数据类型 / 最长位置 | bf16 / 262,144 |

**派生量【估计：按上表相乘】：**

- 每 token 每 FA 层：K = 4 × 256 × 2 B = 2 KiB；K+V = 4 KiB。
- 每 token 16 层：K = 32 KiB；K+V = 64 KiB。
- 每个 16-token 页：1 MiB。这与 `token_gather.cost` 和 G1 的 `page_bytes` 1,048,576 一致 [S15][S23]。

### 1.2 页向量：在哪里产生、存在哪里、怎么打分

**表示**

- 取最后一个 decoder 层的输出，在 final RMSNorm 之前。hook 拿到的是 norm 之后的行，除以 gain 恢复方向 [S3][S4]。
- 每页取页末 token 的那一行。
- 用 128 维、seed 20260904 的随机 ±1 投影（5,120→128），再做 L2 归一化，再量化成 int8 加一个 fp32 scale [S13][S21]。

**产生（在 GPU 上，只占临时 HBM）**

- **phase 1（观察 [C, e)）**：`ProbeRows` 只保留页末行（设备上的一份 clone）和最后一行（观察 query，以 fp32 拷到 host）[S6]。`PageCompactor.compact` 在设备上做 float64 投影、归一化和量化，然后 `.cpu().tolist()` 拷回 host [S5]。
- **phase 2（生成头、冷 prefill、decode）**：`after_forward` 对本步关闭的页做同样的压缩。decode 每 16 步才有一次 [S7]。

**传输**

- 经 worker 的 phase1 回执和 publish 回执。线格式是 `{page: [[128 int8], scale]}`，约 0.6 KB JSON/页。
- 回执体是签名的：规范 JSON 加 HMAC-SHA256。phase 1 签完立即再验一次 [S8][S9]。

**存储（engine 侧的 host 内存，没有常驻 HBM）**

- `vector_catalog[session]`：页到 CompactVector Python 对象的映射，`unit_values` 做了记忆化。
- `PageColumns`：每个会话一个 128 × n 的 float64 列矩阵，按倍增分配，所以 1,024 B/向量是下限 [S11][S12]。
- 页向量封页时算一次，之后不变 [S11]。

**打分**

- `fast_selection.cosines`：与上游 `compact_cosine` 逐位一致的 float64 补偿求和，按维向量化 [S12]。
- 每页的分数四舍五入到 8 位小数，写进 evidence 的 `scores`。engine 签名的回执体里既有全部 n 页的 `ranking`，也有 `selection`（其中包括全部 n 页的 8 位小数 `scores`）。这个回执体在签发时做一次 HMAC，预留时再验证一次 [S11][S37]。

### 1.3 Goal 向量

- **goal_query**：
  - Goal 文本在 phase-1 租约里做一次冷前向，`start=0`，用自己的块 [S6]。
  - 取最后一行的 pre-final-norm 值并归一化，以 5,120 个 float 放进 phase-1 回执 [S8]。
  - plan 时在 engine CPU 上以 fp32 投影成同一个 128 维 int8 空间 [S11]。
  - 按 `retrieval_goal_hash` 缓存，命中时不做 Goal 前向；回执的 `goal.cache_hit` 记录是否命中 [S11]。
- **observation_query**：用 phase 1 的最后一行，也就是观察的末 token [S8]。
- **结论**：Goal 向量是最后一层残差流上的一个点，与页向量在同一空间。它不是任何 FA 层的 query。

### 1.4 token 聚集的现有机制

**选择**

- **挑选器**：`TokenPicker.picks(view)` 返回候选页（mandatory 和 recent 之外）里的 token 位置，按优先级排序。现有 `markers_v1` 和 `separators_v1`，`register_picker` 可以注册新的 [S15]。
- **`view` 的内容**：`stream`、`end`、`page_tokens`、`eligible`、`state`、`encode`、`separator`、`pin`，**没有 room** [S15]。
- **预算 `shared_blocks_v1`**：
  - room = capacity − mandatory − recent，单位是块；picks 截到 room × 16 个。
  - 按 Selector V2 的余弦序逐页走：如果这一页加上"未被已取页覆盖的 picks"所需的聚集块还放得下，就取这一页。
  - 最后一个块不满时，用最新的、没有被注意的 token 补满（pad）[S15][S18 §3]。

**restore**

- 驻留页上的 token 在设备上索引；DRAM 页上的 token 先在 host 上收集到一个 bounce buffer，再对每个 FA 张量做一次 H2D 和一次 `index_put_`，最后 `drain()` [S16]。
- 这一调用在第一个 phase-2 step 里同步执行 [S36]。每 token 搬 64 KiB [S15]。
- **为什么是精确的**：K 在 RoPE 之后存储，q 用逻辑位置，prefix 内 key 的顺序无关；每个 slot 都是记录过的真实 token [S18 §2]。

**GPU 实测**

- **G1（作业 3603240，GH200）**[S23]：

  | 集合 | token | DRAM 聚集中位 | 其中 host index | restore 这些 token 所在的页 | 驻留页聚集 |
  | --- | --- | --- | --- | --- | --- |
  | E | 144 | 53.40 ms | 49.84 ms | 84 页 9.19 ms | 3.46 ms |
  | C | 1,024 | 69.06 ms | 59.27 ms | 498 页 38.88 ms | 15.41 ms |

  `UVA_ADVISED` 的含义：G1 的规则是"C 集 DRAM 聚集中位 > 10 ms 就给出 `UVA_ADVISED`"，它并没有测 UVA 路径本身 [S38]。
- **gath32ti（作业 3603443，goal_query + markers，32 个任务）**[S24][S26]：
  - 1,210 个选择性请求；每请求 picks 均值 133.8，聚集均值 110.6（p95 317）。
  - DRAM 聚集 token 126,474 个，设备上 14,358 个。
  - admission_engine_ms p50：全部请求 230.2，选择性请求 245.0；latency p50 8.549 s；pool wall 2,845.1 s。
  - 没有逐请求的聚集时间，因为聚集在第一个 phase-2 step 内执行。
- **由 G1 得出的今天 restore 路径的有效吞吐【估计：实测字节 ÷ 中位时间】**：
  - 88,080,384 B ÷ 9.1912 ms = 9.58 GB/s；
  - 522,190,848 B ÷ 38.8763 ms = 13.43 GB/s。
  - 这是现有路径（暂存加每个张量一次 `index_copy_`）的有效速率，不是链路上限。下文的"DRAM → HBM 时间"都用 13.43 到 9.58 GB/s 作区间。

### 1.5 SWE 运行的历史规模

- **gath32ti 的 1,559 次调用**：prompt 长度 < 32K 的 1,257 次，32K–64K 237 次，64K–98K 65 次，最长 97,778 [S25]。每次调用平均生成 358.5 个 token [S26]。
- **预算**：capacity 512 页，fill room 均值 315.7 页 [S27]。
- **并发与资源**：16 个会话同时运行；Slurm `--mem 113560M`、16 CPU、1 GPU [S26]。
- **候选页均值**：goalvllmt 1,906 页（约 30,496 token），obs262t 1,492 页；mandatory 约 140 页，recent 约 48 页 [S28]。注意 goalvllmt 与 gath32ti 不是同一个运行。
- **上限**：262,144 token，即 16,384 页 [S1]。
- **下文的两个规模**：
  - "典型"= 1,906 页 / 30,496 token；
  - "上限"= 16,384 页 / 262,144 token。

### 1.6 缺口在文档里的原文

- main 上的 `docs/token-selection` 第 4 节 "Not covered"：按 token 的 Goal 打分需要每 token 一个向量和一个分数（今天的 16 倍），以及 goal_query 工作的确切打分规则；第 5 节问题 4 [S19]。
- `integrate-1114` 上的 `docs/token-gather` 第 9 节第 1 项 [S18]。
- 研究的设定来自作者 11-11 的周报，经我们 11-13 的笔记转述 [S28]：
  - 保护系统提示和首条 user（1,568–2,688 token），以及最近两轮（最多 1,024 token），三者共享 8,192 的预算；剩余槽位给 Goal 分数最高的 token；
  - 实验在作者自己的原生 Mini-SWE 栈上（H200）进行，不在我们的 runtime 上。
- 策略注册表把 token 级选择列为范围之外，其文档把"per-token Goal scores (one vector per Page)"列为未做项；README 说新的页向量分数和 token 级策略需要更多 runtime 支持 [S17][S20][S22]。
- **我们没有任何文档说明两件事**：
  - "按 token 的 Goal 分数"怎么算（第 11 节问题 1）；
  - 研究在自己的栈里怎么应用一个 token 选择：是否保持原位置，GDN 状态是否仍看到全部历史（第 11 节问题 3）。

  这两点是所有选项的前提。

---

## 2. 所有选项都需要的部分

### 2.1 填充：现有的 `shared_blocks_v1` 已能表达"剩余槽位给 token"

把挑选器换成"候选 token 按分数降序"，取前 room × 16 个。

**推导**：`fill` 只在 taken + 1 + ceil((left − c)/16) ≤ room 时取一页，其中 c 是这一页上的 picks 数 [S15]。初始 left = 16·room。只要被取的页都是 c = 16 的页，就一直有 left = 16·(room − taken)，这时条件化简为 c = 16。

**结论**：

- 只有 16 个 token 全被选中的页才会被整页 restore；
- 其余 picks 全部聚集，总数是 16 的倍数，pad 为 0；
- 选中的 token 集合就是前 room × 16 个 token，这正是研究的规则。

**验证**：

- 本说明用 runtime 自己的 `token_gather.plan` 随机验证了 2,000 次 [S33]。
- 审计独立验证了 3,000 次，每次随机几何与五种分数分布（含并列和"16 个 token 里有 15 个高分"的对抗页）[S34]：
  - `plan` 与 `turn` 都成立；
  - 标记在前的列表也成立；
  - quota 情况下，结果与文档规则的重新实现一致。

**两个附带的好处**：

- picks 少于 room × 16 时（例如设了 quota），剩下的预算按页余弦填整页。"N 个 Goal token + 其余整页"的混合形式不需要额外改动。
- 和 markers 组合时，picks 列表就是"全部标记（最新在前），再接 Goal token"，一个组合挑选器即可。

**runtime 要补的**：

- **挑选器拿到分数**：`view` 要能拿到分数，或者拿到 token 目录加 query。
- **挑选器拿到 room**：`view` 里没有 room（[S15] `token_gather.py:468-474`）。两种做法：
  - 在 `view` 里加 room（或 capacity），挑选器只返回前 room × 16 个；
  - 让挑选器返回全部候选、按分数排好，由 fill 截断。这时 `truncated` 会把整个尾部都算进去，而且 `turn()` 逐个检查 pick 的开销随候选数增长：30,496 / 259,136 个候选分别为 6.7 / 56.5 ms，只返回 room × 16 个时为 4.1 / 7.1 ms【本地代理，审计复测，[S34]】。
- **pin**：规则要 pin 进契约（第 7 节）。

### 2.2 聚集量与吞吐

**每轮的量【估计】**

- 每轮最多聚集约 5,051 个 token（room 均值 315.7 × 16）。被整页选满的页改为整页 restore，H2D 字节数相同，但这样的页能跨轮驻留。markers 目前的均值是 110.6，约为它的 1/46。
- H2D 上限：5,051 × 64 KiB = 315.7 MiB/轮。不论页和 token 怎么混合，上限都是 room × 1 MiB。
- 对照：gath32ti 中每个请求的非 phase-1 H2D 均值是 226.4 MiB，其中包含 146.8 MiB 整数倍的 GDN 状态 [S26]。

**时间【估计】**

| 路径 | 每轮时间 | 算法与可信度 |
| --- | --- | --- |
| 现有 host-index 路径 | 约 141 ms | G1 两点（144、1,024 token）线性外推到 5,051 token，约为最大实测集合的 5 倍。斜率 (69.06 − 53.40)/880 = 0.0178 ms/token，截距 50.8 ms（占 144 token 时间的 95 %）。截距来源不明：审计在 x86 上单独测了 host 索引这一步（16 个 FA 张量、一个未 pin 的 256-slot chunk），144 / 1,024 / 5,051 token 分别为 4.2 / 15.6 / 67.9 ms，呈线性，截距只有约 2 ms【本地代理，[S34]】。所以斜率可信，固定部分无法解释——这是要跑 B1 而不是相信这条直线的又一个理由。 |
| 达到今天 restore 的有效吞吐 | 约 25–35 ms | 331,022,336 B ÷（13.43 到 9.58 GB/s） |
| 设备直读 pinned 内存（UVA） | 未测 | G1 只给了阈值判断 [S38] |

**影响**

- **ITL**：聚集在第一个 phase-2 step 内同步执行 [S18 §4][S26][S36]，这段时间直接计入同批所有正在运行的会话。
  - 按 gath32ti 的节奏算【估计】：1,210 个选择性请求 ÷ 2,845.1 s = 0.425 个/s。
  - 每个请求 141 ms 时，所有运行中的会话约有 6.0 % 的墙钟时间被拖住；25–35 ms 时约 1.0–1.5 %。
- **HBM**【估计】：
  - 一个聚集块是 16 个 token × 64 KiB = 1 MiB（16 个 FA 层）。每个会话约 316 个聚集块，约 316 MiB。轮与轮之间，它们取代了今天驻留的已选页，量级相近。
  - 上一轮的聚集块要到 commit 才释放 [S18 §4]，所以预留期间一个会话最多同时持有约 2 × room 个聚集块，约 632 MiB。
- **容量门**：
  - 在有 gather pin 时，容量门已经按 min(largest, 非驻留 + most) 预留新块（quota 为 0 时 `most = largest`，约 512 页）。这与没有 gather 的长历史用的是同一个最坏情况 [S35]。
  - gath32ti 就是在这个估计下跑的，CAPACITY_WAIT 为 0 次 [S24]。
  - 缺少数据的是：每轮真实分配约 316 个聚集块时的表现。

**对策（都与打分规则无关）**

- 设备直接读取 pinned host 内存来做聚集（未测，B1 的内容）；
- 增量聚集（token 聚集第 9 节第 3 项）：上一轮聚集块里重复出现的 token 不再复制（复用率在 B2 中量）；
- 被选满的页整页 restore：fill 已经自动这样做（2.1）。

### 2.3 规则必须 pin 住

下列内容都要写进契约 pin（第 7 节）：

- 分数的定义；
- 并列时的处理（建议按 (−分数, 位置)）；
- 哪些 token 可以被选（候选页内、mandatory 和 recent 之外，是否包括 reasoning token）；
- 与 markers 的先后顺序。

---

## 3. 选项 1：每个 token 一个向量，与页向量同一空间

### 3.1 做法

对每个 token 产出与页向量相同的表示：pre-final-norm 行 → 128 维投影 → L2 → int8 + scale。

页向量就是页末 token 的向量。实现时照 exp-subpage-score 的做法 [S29]：

- 每页第 16 个 token 直接复用页向量，页末行继续走今天的路径；
- 只为页内另外 15 个 token 新算向量。

**为什么第 16 行要留在原路径上**：压缩结果只在舍入边缘以内与批的形状无关（[S5] `page_compaction.py:14-17`）。页末行不进新批，才能保证页级选择、页向量的字节和摘要都不变。

### 3.2 需要算法侧给出的规则

- token 分数是否就是 cos(Goal 向量, token 向量)？这是同一空间里最直接的定义。还是要加邻域平滑、页分数加权，或者只对某些类别的 token 打分？
- query 用 goal_query 的 Goal 向量，还是 observation 向量？同一机制两者都能支持。
- 并列怎么处理；picks 填满全部 room × 16，还是设 quota、其余给整页。

### 3.3 在哪里捕获

| 阶段 | 今天 | 改为 |
| --- | --- | --- |
| phase 1（观察 [C, e)） | `ProbeRows` 只 clone 页末行 [S6] | 在设备上分块压缩全部行，只回 int8 + scale（132 B/token） |
| phase 2 prefill（冷 prompt、重建、生成头） | `after_forward` 只压缩页末行 [S7] | 同上，处理本步全部行 |
| decode | 页关闭时（每 16 步一次）D2H 一次 [S7] | 每步在设备上把这一行投影进一个 16 × 128 的缓冲，不同步；页关闭时一起量化并 D2H 一次。同步次数与今天相同 |
| 可以跳过 | — | mandatory 页（初始 prompt，永远被注意）不需要 token 向量；`--thinking-history strip` 切点之后的页按现有规则不发布 [S7] |

**显存**【估计：fp32 副本为行数 × 5,120 × 4 B，float64 缓冲为行数 × 5,120 × 8 B】：

- `pre_final_norm` 今天已经为一步的每一行做一份 fp32 副本：Qwen 发布版一步最多 16,384 行，即 320 MiB [S29][S39]。
- 新增的只有压缩时的 float64 临时缓冲：一次处理 16,384 行要 640 MiB，每块 1,024 行时为 40 MiB。分块限制的只是这一部分。

**还要覆盖的路径**：`--phase1-in-step`、`--async-scheduling` 作废的行、重建、取消。`--goal-overlap` 的 Goal 前向不需要 token 向量。

### 3.4 内存

| 表示 | 每 token | 典型 30,496 token | 上限 262,144 token | 16 会话 × 典型 |
| --- | --- | --- | --- | --- |
| 二进制：int8 + fp32 scale（再加 int32 范数平方） | 132 B（136 B） | 3.8 MiB | 33.0 MiB | 61.4 MiB |
| 常驻 float32 副本（快速打分用，另加） | 512 B | 14.9 MiB | 128 MiB | 238 MiB |
| 照搬今天的 Python 表示（CompactVector + 记忆化的 `unit_values` + float64 列） | 7,001 + 1,024 = 8,025 B【本地代理】 | 233 MiB | 1.96 GiB | 3.65 GiB |
| 对照：今天每页一个向量（Python 表示） | — | 1,906 页约 14.6 MiB | — | — |

上表的值都是"每 token 字节 × token 数"【估计】。审计复测的 Python 表示为 6,937 + 1,024 B，差别在 1 % 以内 [S34]。

- **HBM**：没有常驻占用，只有捕获时的临时占用（3.3）。
- **对照 DRAM arena**：16 会话 × 1,906 页 × 1 MiB ≈ 29.8 GiB【估计】。二进制的 token 向量只占 KV 的 132 B / 64 KiB ≈ 0.2 %。
- **为什么不能用 Python 表示**：不是因为字节数——3.65 GiB 只占作业 `--mem 113560M` 的约 3.3 %。真正的原因是：
  - 时间：每轮摄入约 71 ms，上限规模下上游余弦要 179 ms（3.5）；
  - 对象数：每 token 约 190 个 Python 对象，16 个典型会话共约 9.3 × 10^7 个【估计：16 × 30,496 × 190】。每 token 对象数取自审计 [S34]，本说明的拆分与之相符：1 个值元组、约 58 个不在小整数缓存里的 int（−127..−6）、对象本身和它的 `__dict__`、记忆化的元组、128 个 float。
- **结论**：二进制表示可行，常驻 f32 副本是否需要取决于对打分时间的要求。

### 3.5 每轮时间

**每轮新 token 数**：沿用 exp-subpage-score 的典型值 1,500 个【估计】。那是 2026-10、P = 784 时期的数 [S29]。gath32ti（thinking 开）里每次调用平均生成 358.5 个 token [S26]，观察的长度不在本地数据里，所以 1,500 只是一个假设。

| 步骤 | 照搬今天的做法（Python、JSON、签名、上游余弦） | 新表示（二进制、numpy、整数精确余弦） |
| --- | --- | --- |
| 摄入 | 约 71 ms【估计：本地代理 47.3 µs/向量 × 1,500】 | 用 `frombuffer` 直接读二进制，< 1 ms【估计】 |
| 签名 + 验证（phase 1） | 6,500 个向量签一次 45.7 ms【本地代理】；1,500 个约 10.6 ms × 2【估计】 | 放在签名体之外：0 |
| 打分，典型 30,496 | 13.49 ms（上游逐位一致）【本地代理】 | 只存 132 B/token、每轮由 int8 转换：3.22 ms；加常驻 f32 副本（再 +512 B/token）：0.29 ms【本地代理，32 核】 |
| 打分，上限 262,144 | 179.2 ms【本地代理，32 核】；1 线程 207 ms【审计复测】 | 由 int8：28.66 ms；f32 副本：4.25 ms【本地代理，32 核】。1–4 线程时分别为 37.8–39.1 ms 和 6.2–9.7 ms【审计复测】 |
| 取前 5,040（= 315 × 16）个并排序 | — | 0.71 ms（典型），3.33 ms（上限）【本地代理】 |
| 把全部分数写进 evidence | 典型 19.4 ms、0.58 MB；上限 183.6 ms、5.23 MB【本地代理】。如果在签名体内，engine 签名和预留验证还要各加约 110 ms【审计复测】 | 只记 picks、阈值和 int8 query，接近 0 |
| GPU 捕获 | 未测。phase 1 每 4,096 行投影约 5.4 GFLOP（float64）【估计：4,096 × 5,120 × 128 × 2】 | 同左 |
| 聚集（2.2） | 25–141 ms【估计；141 ms 是外推值】 | 同左 |

**整数精确余弦**

- **为什么精确**：128 维 int8 点积的绝对值不超过 128 × 127 × 127 = 2,064,512，小于 2^24。所以用 int32 或 float32 累加，按任意顺序（GPU、BLAS、numpy）都精确等于整数点积；bf16 或 TF32 输入加 fp32 累加也精确，但不能用 fp16 或 bf16 累加。验证情况：
  - 本地在 30,496 和 262,144 个向量上验证过 float32 点积与 int64 点积逐元素相等【本地代理，[S32]】；
  - 审计另外验证了打乱维度顺序、顺序累加和 torch CPU 的情况，bf16 输出的 matmul 不相等 [S34]。
- **分数**：score = dot / sqrt(‖q‖²·‖t‖²)，在 float64 下计算。‖q‖²·‖t‖² 小于 2^53，乘积也是精确的；sqrt 和除法都是正确舍入的。
- **限定条件**：**给定 int8 的 query 与 token 向量**，分数在任何实现上逐位相同。query 的 int8 本身是在 engine CPU 上用 fp32 压缩出来的，GH200 与 x86 之间可能在少数情况下末位不同 [S11][S30]。所以建议在回执里记录 int8 query（第 7 节）。
- **与上游的关系**：
  - 数学上它和上游 `compact_cosine` 是同一个实数（`unit_values` = v/‖v‖，scale 会约掉 [S14]），只是最后几位的舍入不同。
  - 审计实测最大差 1.1e-16：30,496 个时排名完全不变；262,144 个时有一对交换，前 5,040 的集合不变 [S34]。
  - 所以把它作为按 token 打分的新规则单独 pin；页级仍用上游的余弦。

### 3.6 代码改动

**worker 侧**

- `remote_ra_live_v2/page_compaction.py`：压缩全部行，结果留作张量；新增二进制线格式。
- `native_probe.py`：`ProbeRows` 保留每 token 的压缩结果。
- `qwen.py`：`after_forward` 中加 decode 缓冲；`publish_pages`。
- `model_queries.py`、`worker_backend.py`：向量放在签名体之外 [S29]。
- `phase1_coordinator.py`：`Phase1Result` 携带 token 向量；它今天携带页向量、观察向量和 Goal 向量 [S10]。

**接线**（exp-subpage-score 也改了这些）

- `private_endpoint.py`：engine 侧的类选择；
- `worker_composition.py`：worker 侧的类选择；
- `composition.py`：启动记录与一次性的投影检查 [S29]。

**engine 侧**

- `query_selection.py`：每个会话一个 numpy token 目录。commit 时追加，重建或关闭时清空；做与页目录相同的完整性检查；plan 时打分，交给挑选器。
- `token_gather.py`：新挑选器（暂名 `goal_scores_v1`），`view` 增加分数和 room；pin 增加规则字段。
- `selection_contract.py`：pin 校验。
- `serve.py`：启动参数。
- `remote_ra_http/evidence.py`：记录。

**harness**

- `postrun/selection_parity.py`：要加每 token 的重放，并支持 goal_query（第 7 节）。

**规模【估计】**：与 exp-subpage-score 相当，再加一个挑选器和一致性工具的扩展。exp-subpage-score 在 `remote_ra_live_v2` 里改了 6 个文件、新增 1 个模块和 19 个测试，另外改了 `serve.py` 和 README [S29]。

### 3.7 风险

1. **规则不匹配**：如果研究的分数不是隐藏状态余弦（例如是注意力 q·k），选项 1 的分数就和研究不一样。这是最大的风险，只能靠第 11 节问题 1 来回答。
2. **投影精度**：128 维 int8 投影相对 5,120 维余弦有误差。页级今天也用它；如果研究用的是全维度，需要比对。
3. **全维度的代价**：如果规则要求全维度，每 token 存 bf16 5,120 维就是 10 KiB：典型会话约 298 MiB，16 个会话约 4.65 GiB（约为 16 会话 DRAM arena 29.8 GiB 的 16 %），上限会话 2.5 GiB【估计】。这可行但不小（作业 `--mem 113560M` [S26]），而且 D2H 也变成每 token 10 KiB。
4. **语义（未验证）**：每个位置的最后一层表示是模型读到这个 token 为止的状态，页向量（页末 token）也是同一种表示。它作为单 token 的检索键是否合适，目前没有数据。
5. **decode 路径**：fast path、async scheduling 和 cudagraph 的每一步都多一个小 kernel，需要测 ITL。
6. **捕获时的临时显存**：float64 临时缓冲会占 HBM，用分块解决（3.3）。

### 3.8 验证

**CPU 测试**

1. 全部行的压缩结果等于逐行的 float64 参照，与分块方式和批的形状无关。
2. 每个 token 的向量恰好发布一次，覆盖 phase 1、prefill 分块、decode、重建、取消、thinking 切点和 async 作废的行。
3. 页向量的字节不变，复用为第 16 个 token 的向量。
4. 整数精确余弦在 numpy、torch CPU 和打乱求和顺序的情况下逐位相同。
5. 挑选器的 top-k 和并列规则正确；fill 在 picks = room × 16 时只整页取走全部被选中的页（S33/S34 的性质写成测试）。
6. 签名体不变，哈希调用次数不变（沿用 exp-subpage-score 的测试法）。
7. 不开启时 `tools/policy_parity.py` 逐字节一致。
8. 在 `test_token_gather_e2e` 那套解耦端到端框架上跑通。

**GPU smoke**

1. G-probe：测 decode 每步的增量（b = 1..16）、1K–8K token 的 phase 1 捕获时间、D2H 字节数。
2. 1 个任务的影子模式：每个请求的选择应与同状态下不开启时相同；离线从帧重算全部分数和 picks。
3. 接入挑选器之后再跑 32 个任务。

---

## 4. 选项 2：直接用 K cache 打分

### 4.1 做法

用一个 query 对 16 个 FA 层（或其中几层）已存储的 K 做点积，再按规则聚合成每 token 一个分数。

### 4.2 Goal 向量不能直接当 query

**为什么不能**：第 l 层的 K 在每头 256 维的空间里（经过 W_k、k_norm、RoPE）[S2]，而 Goal 向量是最后一层的残差（5,120 维，pre-final-norm）。要得到第 l 层的 query，需要第 l 层的输入以及 W_q^l、q_norm 和 RoPE，从最后一层的向量反推不出来。

**可行的做法**：

- **goal_query**：Goal 冷前向本来就跑完 64 层，在每个 FA 层捕获最后一个 Goal token 的 q（post-norm、post-RoPE）即可：16 × 24 × 256 × 2 B = 192 KiB/Goal【估计】。Goal 缓存命中时不做 Goal 前向 [S11]，所以 q 也要按 `retrieval_goal_hash` 缓存（每个 Goal 192 KiB）。
- **observation_query**：用 phase 1 最后一个观察 token 的 q。这个 q 是在 phase 1 能看到的历史上下文里算出来的。
- **近似做法（不推荐）**：把最后一层的向量直接乘最后一个 FA 层的 W_q。输入所在的层不对，只能算近似。

### 4.3 RoPE

- K 以 RoPE 之后的形式存储，相位对应逻辑位置 p_k [S18 §2]。Goal 冷前向的位置是 0..g−1，所以 Goal 的 q 带的是位置 g−1 的相位。
- 每头只有 64 维被旋转，另外 192 维与位置无关。纯文本时 MRoPE 三段的位置相同，等价于 1-D RoPE [S1][S18 §2]。
- 位置的处理方式本身是规则的一部分，有三种：
  1. **把 q 重新旋转到一个参考位置**（例如当前末尾 L），相当于"把 Goal 当作下一个 query"。旋转可以组合，在实数上是精确的。
  2. **只用 192 个不旋转的维度**，得到与位置无关的分数。
  3. **按每个 token 的位置把 K 反旋转回去**，每个 key 多做一次 64 维旋转。

**还要由规则决定的事**：

- 24 个 q 头怎么聚合（每 6 个共享一个 KV 头）：取 max、取 mean，还是按 softmax 质量；
- 用哪几层；
- 是否做 softmax 归一化：做的话要对全部 key 走两遍，或者在线算 logsumexp。

### 4.4 带宽与时间

非驻留页数近似取 n − 512（512 是 capacity）。

| 历史 | 非驻留页 | 1 层 K 读取量 | 1 层时间 | 16 层 K 读取量 | 16 层时间 |
| --- | --- | --- | --- | --- | --- |
| 典型，1,906 页 | 1,394 | 43.6 MiB | 3.4–4.8 ms | 697 MiB | 54–76 ms |
| 上限，16,384 页 | 15,872 | 496 MiB | 39–54 ms | 7.75 GiB | 620–868 ms |

**这张表怎么算的【估计】**：页数 × 32 KiB/层，再除以 13.43 到 9.58 GB/s。arena 里每页每层的 K 是连续的 32 KiB，步长 64 KiB [S16]。

**这张表的局限**：

- 13.43 到 9.58 GB/s 是今天 restore 路径的有效速率（暂存加每个张量一次 `index_copy_`），不是链路上限。
- 表中还假设这些 32 KiB 的块能达到整页 restore 的速率。
- 设备直读 pinned arena 的路径（B1 要测的另一条路径）没有测过，可能把这些数字大幅改向任一方向。

**其他**：

- **计算量**：每 token 每层 24 × 256 = 6,144 次乘加；上限规模下 16 层共约 2.6 × 10^10 次乘加【估计】，GPU 上没有测过。
- **驻留页**：它们的 K 直接在 HBM 里读；暂存按 `MAX_STAGING_BYTES` 64 MiB 分片 [S16]。也可以在 host CPU 上直接读 pinned arena 来打分，省掉 H2D，但 CPU 的吞吐没有测过。
- **时序**：打分必须在 plan 之前完成。两种放法：
  - **并入 phase-1 的 worker 调用**：Goal 前向本来就在这次调用里 [S6]，打分接在它之后。不加 RPC，但 phase 1 变长。
  - **单独一次 worker pass 加一次 RPC**。

  不论哪种，分数都要回到 engine：每 token 一个 float32，262,144 token 是 1 MiB。对照：选择性请求的 admission_engine_ms p50 是 245 ms [S26]。

### 4.5 内存

- 没有新的常驻存储，只有 Goal q 的缓存（每个缓存中的 Goal 192 KiB）；
- HBM 只有临时占用：暂存的 K（按 64 MiB 分片）和 Goal 的 q。

### 4.6 代码改动

- **模型文件**：`vllm/model_executor/models/qwen3_next.py` 的 overlay，在 `Qwen3NextAttention.forward` 里只在 probe 标志打开时捕获 q。
  - 融合与 eager 两条路径都要覆盖 [S2]，默认路径必须完全不变。这是 vLLM 的模型文件，风险最高。
  - 带 `--phase1-graph-tokens` 时，probe 前向可能在 CUDA graph 下运行 [S4]，所以捕获必须是 graph-safe 的。
- `native_probe.py`：在 Goal 前向或 phase 1 里收集 q。
- **worker 打分器**：新写一个，放在 `tensors.py` 旁边：从驻留块和 arena 读 K，逐层打分。
- `model_queries.py`：分数随回执返回，放在签名体之外。
- **engine 与其他**：`query_selection.py` 接收分数并交给挑选器；`token_gather.py`；契约和启动参数；证据。

### 4.7 风险

- 改动的是每次前向都会经过的模型代码。
- 带宽随历史线性增长；admission 的关键路径或 phase 1 变长。
- **一致性**：K 不在证据里（K+V 每 token 64 KiB，30,496 token 的会话约 1.86 GiB【估计】）。
  - 所以必须把每 token 分数记进证据，才能重放填充：float32 二进制每请求 119 KiB（典型）或 1 MiB（上限）；如果用 JSON，约 0.58 MB 或 5.2 MB【估计 / 本地代理】。
  - 打分器本身只能用 GPU probe 对照 dense 参照来验证。这和 goal_query 今天"只重放 gather、不重放 Goal query"的情况相同 [S31]。

### 4.8 验证

**CPU 测试**

- 在 toy RoPE 注意力层上（`GatheredAttentionTests` 那套）：重新旋转后的 q 在记录的 token 上与 dense 注意力的 logits 一致，误差不超过 1e-12。
- 覆盖部分旋转和 GQA 的头映射。
- 从 arena 读出的 K 与设备上的行相同。

**GPU probe（G1 风格）**

- 在真实模型上捕获 16 层的 q，与 torch 的 dense 参照对比，K 的来源包括驻留页和 DRAM 页两种。
- 在 30K 和 262K token 两种规模下逐层计时；也在 graph 模式下验证。
- 之后跑 1 个任务的影子模式。

---

## 5. 选项 3：两段式（页余弦短名单 → 名单内按 token 打分）

### 5.1 做法

1. 第一段：用今天的页余弦（1,906 页 1.2 ms【本地代理】）取前 S 页。
2. 第二段：只对这 S 页里的 token 打分。picks = 名单内分数最高的 room × 16 个 token；fill 仍用 `shared_blocks_v1`。

### 5.2 什么时候有意义

- **配选项 2**：K 的读取量上限变成 S × 32 KiB/层，与历史长度无关。S = 2 × room ≈ 631 页时，1 层 19.7 MiB（1.5–2.2 ms），16 层 315.5 MiB（25–35 ms）【估计：算法同 4.4】。名单中有一部分页是驻留的，所以这是上界。
- **配选项 1**：只省打分时间，不省存储——向量必须在前向时就捕获，而那时还不知道哪些页会进名单。整数精确打分在上限规模下约为 30–40 ms（由 int8 每轮转换：32 核 28.7 ms，1–4 线程 37.8–39.1 ms【本地代理 / 审计复测】），用了常驻 f32 副本则更少。所以不值得做。

### 5.3 代价：覆盖率

名单外的 token 永远选不到。覆盖率 = 全量规则选出的前 room × 16 个 token 中，落在前 S 页里的比例。

- **配选项 2**（真正有意义的组合）：覆盖率需要 q·k 分数，来自选项 2 的 probe，或作者提供的逐 token 分数（第 11 节问题 7、10）。
- **原型 A（选项 1 的影子模式）**：只能给出同空间余弦这种分数下的覆盖率。

### 5.4 需要算法侧给出的规则

- S 怎么取；
- 第二段用什么分数（同选项 1 或选项 2）；
- 名单是否包括已经整页取走的页。

### 5.5 代码与验证

- **代码**：在选项 1 或 2 的基础上，在 plan 里加一个截断，并记录 S 和名单。
- **CPU 测试**：S = n 时结果与全量逐字节相同。
- **GPU**：同选项 1 或 2。

---

## 6. 选项 4：降维、量化或预计算标量

**更低维的向量**（例如 32 或 64 维 int8，每 token 36 或 68 B，对比 132 B）

- 这是选项 1 的一个参数。二进制的 132 B/token 已经很小（262,144 token 才 33 MiB），所以降维只在证据体积或要放进 HBM 时才有意义。
- 代价是余弦误差变大，需要规则接受。

**每页几个向量**（每页 2 或 4 个，取每个子段的末 token）

- 存储是 2 或 4 倍，而不是 16 倍。
- 只能选到 8 或 4 个 token 的连续子段，选不到单个 token。
- 如果算法侧能接受"短段"，这是页和 token 之间的中间档：把 exp-subpage-score 的思路搬到 16 以下 [S29]。

**捕获时预计算的标量**

- Goal 按 `retrieval_goal_hash` 缓存，命中时复用 [S11]；但缓存未命中时 Goal 会变。更根本的是，捕获时之后各轮的 Goal 还不存在。所以与 Goal 相关的分数不能在捕获时预计算。作为"按 token 的 Goal 分数"的实现方式，**这条不成立，删去。**
- 能预计算的只有与 Goal 无关的先验，例如 token 的类别、它与本页向量的余弦。如果规则是"页的 Goal 余弦 + 页内 token 的先验"，可以用每 token 4 B 实现。可不可用由算法侧决定。

**Goal 不变时缓存分数**

- 命中时旧 token 的分数可以复用，只给新 token 打分。
- 整数精确打分已经很快，这一优化的收益很小。命中率可以从现有证据（回执里的 `goal.cache_hit`）离线统计 [S11]。

---

## 7. 选项 5：契约、回执、证据与一致性（所有选项都需要）

**契约**

- 新键，或者在 `token_gather` 的 pin 里加字段：`{vector, score, query, picker, quota, pad, budget}`，其中 `vector` 为表示与维度，`score` 为规则 id（如 `int_cosine_v1`），`query` 为 `goal|observation`。
- 只在开启时出现，其他契约和摘要逐字节不变。这是 token 聚集和 page_score 的做法 [S15][S29]。

**engine 回执（签名体）**

- 保持 `gather {tokens, pad}`：每轮最多约 5,051 个位置，JSON 约 30–35 KB【估计：每个位置约 6–7 个字节】。
- `ranking` 仍然是页的排名；`selection.scores` 仍是 n 页的分数 [S11]。
- 新增阈值（最后一个被选 token 的分数）、计数，以及 int8 query（132 B）。有了 int8 query，每 token 的重放就不需要恢复 query。
- **不要**把每 token 的分数放进签名体：
  - 262,144 token 时每请求 5.2 MB，构造加序列化至少 184 ms【本地代理】；
  - 签名体在签发时 HMAC 一次、预留时验证一次 [S37]，审计测得这样的 body 做"排序 JSON + HMAC"每次还要约 110 ms【审计复测，[S34]】。

**worker 帧（证据）**

- 每 token 的向量放在签名体之外。先例是 exp-subpage-score 的 `subpage_vectors`：在 HMAC 体之外，不新增每请求的哈希 [S29]。
- **取舍**：这些向量因此不受认证，而页向量受认证。这是有意的取舍，与"完整性检查只在启动时做"一致。
- **格式**：建议用二进制（base64 编码的 int8 + fp32，每 token 约 176 字节【估计：132 × 4/3】），而不是 JSON 列表（456 B/token【本地代理】）。
- **每个会话的证据增量**【估计】：
  - 典型会话 13.9 MB（JSON）或 5.4 MB（base64）；97,778 token 的会话 44.6 MB（JSON）；
  - 每次重建会重新发布（gath32ti 有 35 次重建 [S25]）；
  - 作为对照，一致性工具的注释说，perf-swe16 时单个原生回执最多约 1 MB [S30]。
- **选项 2**：还要记每 token 分数（4.7）。

**一致性与重放**

- **选项 1**：
  - **observation_query**：从帧重建每个会话的 token 目录，用 pin 住的整数精确余弦重算分数，得到 picks，跑 `token_gather.turn`，再与签名的 gather 比较。现有工具对 observation query 已有 `recover_query` 处理 GH200 fp32 末位的差异 [S30]；回执里记录 int8 query 后就不需要恢复。
  - **goal_query**：今天的一致性工具直接拒绝它（`selection_parity.py:167-168`："is not replayed (no rewriter here)"）[S30]；gather harness 也只在签名分数上重放 gather [S31]。要完整重放 goal_query，得先扩展这个工具。Goal 向量本身在 phase-1 帧里，缓存命中时引用更早的帧；有了回执里的 int8 query，可以直接重放。
- **选项 2**：K 不在证据里，只能在记录的分数上重放填充；打分器靠 GPU probe 验证。
- **`tools/policy_parity.py`**：不开启时逐字节一致。

**哈希**：不新增每请求或每个 decode 步的哈希；完整性检查只在启动时做。

---

## 8. 考虑过但不成立的做法

| 做法 | 为什么不成立 |
| --- | --- |
| 更小的页（P < 16） | FA 的 kernel 块最小是 16 [S19 §2] |
| 按 head 选择 | 一个层组共用一张块表 [S18 §9] |
| 事后重算 hidden state | 要用该位置之前的全部上下文（含 GDN 状态）重跑 64 层，等于重做 prefill |
| 让 Goal 前向带着全部历史 KV 做注意力 | 要把全部页 restore 到 HBM，等于不做选择 |
| 捕获时预计算与 Goal 相关的 token 分数 | 捕获时之后各轮的 Goal 还不存在（第 6 节） |

---

## 9. 对比表

| | 选项 1（二进制） | 选项 2（K，16 层） | 选项 3（K + 短名单，S = 631） | 选项 4（降维 / 先验） |
| --- | --- | --- | --- | --- |
| 需要算法侧给出 | 规则是否就是同空间余弦 | 层、头、RoPE、聚合方式 | S，第二段的分数 | 是否接受精度损失或先验 |
| 新增存储（host 内存，每会话） | 132 B/token：典型 3.8 MiB，上限 33 MiB；要快速打分时再加 512 B/token 的 f32 副本：14.9 MiB / 128 MiB | 无（Goal q 缓存 192 KiB/Goal） | 同选项 2 | 36–68 B/token，或 4 B/token |
| 每轮打分，典型 | 只存 132 B：3.2 ms；加 f32 副本：0.3 ms；另加 top-k 0.7 ms【本地代理，32 核】 | K 读取 54–76 ms + 计算（未测）【估计，按今天 restore 路径的速率】 | 25–35 ms（上界）【估计】 | 同选项 1 |
| 每轮打分，上限 | 只存 132 B：28.7 ms（1–4 线程 37.8–39.1）；加 f32 副本：4.2 ms（1–4 线程 6.2–9.7）；另加 top-k 3.3 ms【本地代理 / 审计复测】 | 620–868 ms【估计】 | 25–35 ms【估计】 | 同选项 1 |
| 捕获 | 所有前向：phase 1、prefill、decode | 在 Goal 前向（或 phase 1）中捕获 q | 同选项 2 | 同选项 1 |
| 聚集（各选项共有） | 每轮最多约 5,051 token：现有路径约 141 ms（G1 两点外推，约为实测范围的 5 倍），达到 restore 吞吐时 25–35 ms【估计】 | 同左 | 同左 | 同左 |
| 最大的单项代价 | 聚集 | K 读取（与聚集同量级或更大） | 聚集与 K 读取同量级 | 同选项 1 |
| 改动面 | worker 捕获、engine 目录、挑选器、接线、一致性工具（中） | 模型文件 overlay、worker 打分器、分数回传（大） | 选项 2 + 截断（大） | 选项 1 的参数（小） |
| 离线重放 | 完整；goal_query 需要扩展一致性工具 | 只能在记录的分数上重放填充 | 只能在记录的分数上重放填充 | 完整 |
| 主要风险 | 规则不匹配 | 带宽、改模型前向、验证 | 覆盖率 | 精度 |

---

## 10. 推荐：先原型化什么

1. **现在（不需要 GPU）**
   - 把第 11 节的问题发给 goal_query 研究的作者。
   - 请作者按问题 10 的格式给几条请求的逐 token 分数，将来用来核对我们的实现，也用来算配选项 2 时的覆盖率。
2. **原型 A：选项 1 的影子模式**
   - **做什么**：捕获、二进制传输、numpy 目录、整数精确余弦；只记录 top-k、阈值、与页填充的重合度，以及 S = room、2 × room、4 × room 时的覆盖率。不改选择。
   - **哪些部分与规则无关**：捕获、传输、存储与规则无关；记录下来的统计量只针对同空间余弦这一种规则。
   - **理由**：
     - 它与现有页向量路径同构：页向量就是每页最后一个 token 的向量。
     - 离线可以完整重放（goal_query 需要先扩展一致性工具）。
     - host 内存和 engine 打分的本地代理代价很小（3.4、3.5）；GPU 捕获的代价（phase 1、decode 每步）还没测，正好由 A 测出。
   - **能回答的问题**：按 token 的同空间余弦选择与今天页填充差多少（重合度、页外 token 数），以及这种分数下两段式的覆盖率。
   - **不能回答的问题**：是否"更好"——这只能靠改变选择的运行来回答。
3. **原型 B1：大批量聚集吞吐的 GPU probe（与规则无关，现在做，可与 A 并行）**
   - 把 G1 扩展到 5,051 个 token，并用真实的分布：分散在 1,000 多页、多个 arena chunk 上，而不是 G1 的 100–600 页。
   - 比较现有路径和设备直读 pinned 内存的路径。
   - 在有并发 decode 的 step 里计时。
   - 理由：不论用哪种打分，按 token 填充都可能每轮聚集几千个 token（2.2）。
4. **原型 B2：增量聚集的复用率（在 A 之后）**
   - 哪些 token 会跨轮重复取决于规则，合成数据量不出来。
   - 用 A 记录的影子选择（或某条规则的离线重放）统计逐轮的重复率，再决定是否实现增量聚集。
5. **规则确认之后**
   - 如果规则是同空间余弦，或者能由 token 向量算出来：在原型 A 上加挑选器，接进 token 聚集（可与 markers 组合），先跑 1 个任务的 smoke，再跑 32 个任务。
   - 如果规则基于注意力（q·k）：再立项选项 2。先做 Goal q 捕获加 dense 参照的 probe（同时产出 q·k 分数下的覆盖率），并用选项 3 限制带宽。
6. **选项 4 不单独立项**，需要时作为选项 1 的参数。

---

## 11. 要问 goal_query 研究作者的问题

1. "Ours-token" 里一个 token 的 Goal 分数到底怎么算？
   - (a) Goal 向量与该 token 隐藏状态的余弦：哪一层？final norm 之前还是之后？全维还是投影后的？
   - (b) Goal（或它的最后一个 token）作为 query 对历史 K 的注意力：哪些层、哪些头？怎么聚合？是否 softmax？Goal 放在哪个位置（RoPE）？
   - (c) 其他方式（外部嵌入模型等）？
2. 研究用的 Goal 文本和我们的一致吗（maintained Goal 的 `<retrieval_query>` 文本和 rewriter）？每一轮都重新打分吗？
3. 研究在自己的栈里怎么应用一个 token 选择？
   - 被选 token 是保留原位置（原 RoPE 相位），还是用选中的 token 重新 prefill 一个更短的 prompt？
   - 线性注意力（GDN）层的状态是否仍然看到全部历史？在我们的 runtime 里，选择只作用于 16 个 FA 层、保持逻辑位置，48 个 GDN 层保留全部历史的状态。
   - 另外：只选单个 token，还是有邻域（±k token 平滑、保留短段）？把单个被选 token 的 K/V 单独拿来注意，是否就是研究的设定？（我们做的是单 token 聚集。）
4. 与条件 E、C 的先后关系：顺序是 prefix → recent → markers → Goal token 吗？Goal token 是否排除 markers 和分隔符？
5. 有没有比较过"按 16-token 块的 Goal 分数"（每块一个分数，整块保留）和按 token 打分？如果差别很小，今天的页填充就够了。这和 main 上 token-selection 说明第 5 节问题 3 是同一类问题 [S19]。
6. 精度：128 维 int8 投影可以接受吗，还是必须用全维？全维每 token 10 KiB，代价见 3.7。
7. 能否接受近似：只在页余弦的前 S 页里按 token 选（两段式）？S 取多大？能否在研究自己的栈上量一下覆盖率？
8. 并列时怎么处理？我们需要把它 pin 住，保证确定性。
9. thinking 文本（reasoning token）算不算候选？
10. 能否提供几条请求的逐 token 分数，作为核对数据？每条请求请给出：
    - 请求标识；
    - 所用 tokenizer；
    - 在完整 prompt token id 中的位置；
    - 每个位置的分数；
    - 最终被选中的集合。

    这样才能和我们的分词对齐。

---

## 12. 待讨论的候选做法

这些是已发表方法的一般性描述，不点名。新增引用需要维护者同意。

- **累计注意力作先验**：以最近若干 query 对各个 key 的累计注意力作为每 token 的重要度，选高分 token。这是与 Goal 无关的先验，对应选项 4 的先验，或选项 2 用 observation 的 q。
- **块级上界筛选**：给每个块保存 key 的逐元素最小值和最大值（或一个代表向量），用它算 q·k 的上界来挑块，再在块内选 token。这相当于把选项 3 的第一段换成基于 K 的上界；每页每层要多存 2 × 4 × 256 × 2 B = 4 KiB【估计】。
- **近似检索**：对 key 做聚类或乘积量化，用近似最近邻检索注意力高的 token。对应选项 4 的降维。
- **按层、按头分别选 token**：runtime 做不到，因为一张块表由同组的所有 head 共用 [S18 §9]。

---

## 附录 A：本地 CPU 代理测量

为本说明在一台 x86 Windows 机器上测得（Python 3.12.7、numpy 1.26.4、32 个逻辑核；不是 GH200），只用来比较量级。测量时没有限制线程数，numpy 的 BLAS 用了全部 32 个逻辑核；独立审计（2026-11-15）在同一台机器上以 1 线程和 4 线程复测 [S34]。

- **测了什么** [S32]：按文件路径加载 `integrate-1114` 的 `dram_kv/page_vectors.py` 与 `remote_ra_live_v2/fast_selection.py`（不导入包，不写任何 worktree）：每个向量的线格式、Python 表示及其摄入（`json.loads`、构造与校验、`unit_values`）；对 406 个和 6,500 个向量签名；`fast_selection.cosines` 在 1,906、16,384、30,496 和 262,144 列上的时间；evidence `scores` 字典的构造与序列化；由 int8 和由常驻 f32 副本计算的整数精确余弦，以及前 5,040 排序和全排序；float32 与 int64 点积的比较。
- **填充检查** [S33]：用 runtime 自己的 `token_gather.plan` 验证 2.1 节的性质，随机 2,000 次。

## 附录 B：按独立审计修改的对照 [S34]

**必改（11 项，全部完成）**

| # | 审计要求 | 改在哪里 |
| --- | --- | --- |
| 1 | "聚集比打分大"只对选项 1 成立 | 第 0 节第 2 点改为分选项陈述；第 9 节加"最大的单项代价"一行 |
| 2 | 原型 A 只能量同空间余弦下的差异和覆盖率；q·k 的覆盖率要靠选项 2 的 probe 或作者的分数；删去"好多少" | 第 0 节推荐、5.3、第 10 节第 2 项 |
| 3 | 原型 B 拆开：吞吐 probe 现在做，复用率在 A 之后做 | 第 0 节推荐、2.2 对策、第 10 节第 3、4 项 |
| 4 | 0.29 / 4.2 ms 需要 512 B/token 的常驻 f32 副本 | 第 0 节第 3 点、3.4 新增一行、3.5、第 9 节 |
| 5 | 逐位可复现要以 int8 query 为前提 | 第 0 节第 3 点、3.5 限定条件 |
| 6 | "最多约 1 MB" | 第 7 节 |
| 7 | goal_query 的重放要扩展一致性工具（`selection_parity.py:167-168` 拒绝它） | 第 7 节、3.6、第 9 节；S30 补上这几行 |
| 8 | 第 9 节中 4.3 改为 4.2 | 第 9 节 |
| 9 | 141 ms 标明是两点外推、约为实测范围的 5 倍 | 第 0 节第 2 点、2.2、3.5、第 9 节 |
| 10 | 问研究怎么应用选择、GDN 是否看到全部历史；说明所有选项只作用于 FA 层 | 头部"共同前提"、1.6、第 11 节问题 3 |
| 11 | 写明 BLAS 线程数；"不超过 29 ms"改为约 30–40 ms | 头部、3.5、5.2、第 9 节；S32 补线程说明 |
| — | S29 的提交说明（功能提交 `32f36a1`，分支 HEAD `5a8453f`） | 附录 C 的 S29 |

**建议（20 项，全部采纳，没有跳过）**

| # | 内容 | 改在哪里 |
| --- | --- | --- |
| 1 | 挑选器的 `view` 没有 room；两种做法及 `turn()` 的开销 | 第 0 节第 1 点、1.4、2.1、3.6 |
| 2 | 16 个会话下的 HBM；整批被拖住的时间占比 | 2.2 影响 |
| 3 | 容量门已按最坏情况预留 | 2.2 容量门 |
| 4 | `UVA_ADVISED` 只是阈值规则 | 第 0 节、1.4、2.2 |
| 5 | G1 直线的截距无法解释；审计的 host 索引代理测量 | 2.2 时间表 |
| 6 | 每张表单位统一；Python 路径不可行的真正原因 | 3.4 |
| 7 | `pre_final_norm` 已有 fp32 副本，分块只限制 float64 临时 | 3.3 |
| 8 | 71 ms 和 < 1 ms 标为估计；1,500 的出处 | 第 0 节、3.5 |
| 9 | K 读取时间所依据的速率及其局限 | 第 0 节第 4 点、1.4、4.4 |
| 10 | 选项 2 可并入 phase-1 调用；q 缓存；graph-safe | 4.2、4.4、4.5、4.6 |
| 11 | 5,051 是上限 | 第 0 节、2.2 |
| 12 | "至少 184 ms"，另加签名和验证各约 110 ms | 第 0 节第 7 点、3.5、第 7 节 |
| 13 | 选项 2 要记录分数及其体积 | 4.7、第 7 节 |
| 14 | 改写"Goal 每轮都变"；与"降维"一句对齐 | 第 0 节第 6 点、第 6 节、第 8 节 |
| 15 | 签名体外的向量不受认证 | 第 0 节第 7 点、第 7 节 |
| 16 | 回执里记录 int8 query | 第 0 节、3.5、第 7 节 |
| 17 | 签名回执里也有 `selection.scores` | 1.2、第 7 节 |
| 18 | 原型 A 的措辞（代价是本地代理；GPU 捕获未测；统计量只针对一种规则） | 第 10 节第 2 项 |
| 19 | 接线文件和一致性工具；S29 的提交说明 | 3.6、附录 C 的 S29 |
| 20 | B1 用真实分布、在 step 内计时；问题 10 给出数据格式 | 第 10 节第 3 项、第 11 节问题 10 |

## 附录 C：来源

上文每个事实后面的 `[S#]` 指这里的一行。代码与文档位置除该行另有注明外，都相对 `integrate-1114` 的仓库根目录（HEAD `a565e84`，代码 `b388033`）；`main` 的文档取自 `f18b8e5`。GPU 读数只来自已核验的结果，以作业号注明；本地测量见 S32、S33（附录 A）；独立审计及其复测见 S34。本仓库以外的来源只写名称，不写路径。

修订（2026-11-15，按审计）：补充 S4、S11、S15、S26、S30、S32 的行号或说明；改正 S29 的提交说明；新增 S34–S39（新增的代码位置都已逐行核对）。

**模型与注意力**

| ID | 事实 | 位置 |
| --- | --- | --- |
| S1 | Qwen3.6-27B：`hidden_size` 5120；`num_hidden_layers` 64；`full_attention_interval` 4，`layer_types` 中 16 个 `full_attention`（层 3, 7, …, 63）、48 个 `linear_attention`；`num_attention_heads` 24；`num_key_value_heads` 4；`head_dim` 256；`partial_rotary_factor` 0.25；`mrope_interleaved` true，`mrope_section` [11, 11, 10]；`rope_theta` 10,000,000；`dtype` bfloat16；`max_position_embeddings` 262,144；`attn_output_gate` true | `source-overlay/remote_ra_linux/model-evidence/model-config.json:11, 13, 15, 16, 18, 21-86, 93, 97, 98, 99, 102, 105-110, 112`（FA 层数由 `layer_types` 计数得出） |
| S2 | `Qwen3NextAttention`：`scaling = head_dim**-0.5`（240）；`get_rope`（264）；每头 `q_norm` / `k_norm`（297-298）；融合 qk-norm-RoPE 路径的启用条件（304-309）；`_project_qkv_gate` 返回 post-norm、post-RoPE 的 q、k、v（311-365）；`forward`：`qkv_proj` → `_project_qkv_gate` → `self.attn(q, k, v)` 写入分页 cache（367-377） | `source-overlay/vllm/model_executor/models/qwen3_next.py:240, 264, 297-298, 304-309, 311-365, 367-377` |

**页向量与 Goal 向量**

| ID | 事实 | 位置 |
| --- | --- | --- |
| S3 | 向量取最后一个 decoder 层的输出、final RMSNorm 之前；hook 拿到的是 norm 后的行，除以每通道 gain 恢复方向；只有向量消费者看到这些行 | `source-overlay/remote_ra_live_v2/vector_space.py:1-9, 46-51` |
| S4 | `after_forward` 把 `pre_final_norm` 行交给 bridge 与 probe（40-58）；`force_eager` / `probe_piecewise`：带 `--phase1-graph-tokens` 时，只含 probe 的 pass 可以重放 piecewise CUDA graph（61-77） | `source-overlay/remote_ra_live_v2/hooks.py:40-58, 61-77` |
| S5 | 解耦布局：一次前向里关闭页的行在设备上一起压缩（投影 5120→128、L2、int8 每向量 scale），只回 128 int8 + 1 个 fp32 scale；线格式 `{page: [[128 ints], scale]}`；P = 16 时 6.5K 观察关闭约 406 页，旧的逐页 D2H + `tolist` + Python 归一化约 1.5-5 ms/页；结果与批形状和设备无关，只是在 float64 值落在 int8 或 fp32 舍入边缘的舍入误差内时例外（每页约 1e-9）（14-17）；`compact` 用 float64 linear，`.cpu().tolist()` 回 host；`closing` 用步长视图取页末行 | `source-overlay/remote_ra_live_v2/page_compaction.py:1-25, 97-113, 115-128` |
| S6 | `ProbeRows`：解耦时只保留页末行（设备上一份 clone）和最后一行（host fp32）；legacy 拷全部行，6.5K 观察约 133 MB；Goal 前向从 `start=0` 在自己的租约上冷启动 | `source-overlay/remote_ra_live_v2/native_probe.py:12-35, 184-191, 385` |
| S7 | phase 2（生成头、冷 prefill、decode）每次前向关闭的页：解耦时在设备上一起压缩（`compactor.closing`）；`--thinking-history strip` 切点之后的页不发布；`publish_pages` 发布 [phase2_start, cursor) 关闭的页 | `source-overlay/remote_ra_live_v2/qwen.py:444-466, 485-497` |
| S8 | phase 1：关闭页的向量 = `compactor.compact(rows.closing)`；观察向量 = 最后一行归一化；Goal 向量 = Goal 前向最后一行归一化（5,120 个 float）；回执体含 `page_vectors`、`observation_vector`、`goal_vector`，签名后立刻验证一次 | `source-overlay/remote_ra_live_v2/model_queries.py:109-132, 133-146, 156-176` |
| S9 | `publish_vectors`：解耦时 int8 向量，"~0.6 KB of JSON per Page"，回执 MAC 覆盖这些字节；`sign_receipt`：`json.dumps(sort_keys=True)` + HMAC-SHA256 | `source-overlay/remote_ra_live_v2/worker_backend.py:369-386, 396-402` |
| S10 | engine 的 `Phase1Result` 携带 `page_vectors`、`observation_vector`、`goal_vector` | `source-overlay/remote_ra_live_v2/phase1_coordinator.py:353-374` |
| S11 | `QuerySelectionAuthority`：页向量逐会话缓存（封页后不变，注释记录旧的每页每轮代价）（495-500）；解耦时 worker 已压缩、engine 不投影；numpy 快路径与每会话列矩阵（504-517）；`memo_types` 记住 `unit_values`（183-226）；`page_vector` / `query_vector`（605-618）；plan 先走快路径再 `gathered`（662-668）；`gathered`：按 Selector V2 余弦序调用 `token_gather.turn`（674-703）；`plan_fast`：浮点 Goal 在 CPU 上压缩为 int8（947-953）、`fast_selection.cosines`（966）、按 (-cosine, Page) 排序（972）、evidence `scores` 每页四舍五入到 8 位（1007）；`plan_turn`：Goal 按 `retrieval_goal_hash` 缓存、命中时不做 Goal 前向（1565-1577），签名回执体含全部页的 `ranking` 和 `selection`（即 evidence 字典，包括 8 位小数的 `scores`）（1584-1598），其中 `goal.cache_hit`（1593），以及 `gather`（1599-1602）；`publish_completion`：封页向量只提交一次（1622-1633）；预留时验证回执 MAC（1635-1645）。浮点 query 在 engine CPU 上用 fp32 的 `selection_contract.compact` 压缩（952-953） | `source-overlay/remote_ra_live_v2/query_selection.py`（行号如左） |
| S12 | `fast_selection`：与上游逐位一致的向量化余弦（参照路径约 10 µs/页，8-16K 页时每次 admission 0.3-0.5 s）；`cosines` 是逐维 128 次的补偿求和（与 CPython `sum()` 逐步一致）；`PageColumns`：每会话一个 128×n float64 列矩阵 | `source-overlay/remote_ra_live_v2/fast_selection.py:1-27, 45-87, 90-128` |
| S13 | 契约：128 维、种子 20260904 的随机 ±1 投影（"seeded Rademacher"），L2，int8 每向量量化（14-18）；解耦时 worker 压缩、线格式（20-28）；`PROJECTION_DIMENSION`、`PROJECTION_SEED`、SHA（92-99）；`capacity_pages`（380-391）；`mandatory_pages`（394-397）；`projection` / `compact`（626-648）；`compact_batch`（651-670） | `source-overlay/remote_ra_live_v2/selection_contract.py`（行号如左） |
| S14 | 上游 `CompactVector`：int8 值 + 正的 scale；`unit_values` = 值×scale 再除以其 L2 范数（scale 在数学上约掉）；`compact_cosine` = 两个 unit 向量的点积并夹到 [-1, 1] | `source-overlay/dram_kv/page_vectors.py:94-132, 443-453` |

**token 聚集**

| ID | 事实 | 位置 |
| --- | --- | --- |
| S15 | `token_gather.py`：常量（`PAGE_TOKENS` 16、`QUOTA_LIMIT` 65,536 等，59-72）；`TokenPicker` 接口（`picks(view)` 返回优先级序的位置）（76-100）；`newest`（124-133）；`markers_v1` / `separators_v1`（136-200）；`register_picker`（205-210）；`fill`（shared_blocks_v1，399-423）；`pad`（426-441）；`plan` 与记录（444-459）；`turn`（462-479；挑选器的 `view` 只有 stream、end、page_tokens、eligible、state、encode、separator、pin，没有 room，468-474）；`plan_entry` 的检查（484-497）；`sources`：驻留页在设备上读、其余从 DRAM 页读（506-520）；`cost`：每 token 64 KiB、每页 1 MiB（523-526） | `source-overlay/remote_ra_live_v2/token_gather.py`（行号如左） |
| S16 | `PageArena`：预分配的 pinned host chunk，每个 FA 张量一个 [slots, 去掉块轴的形状] 缓冲（即每页每层 `[2, 16, 4, 256]` 连续，K 在前 32 KiB），`CHUNK_BYTES` 256 MiB（335-377）；`MAX_STAGING_BYTES` 64 MiB（612-618）；`restore_pages`（662-710）；`gather_tokens`：设备行一次索引，host 行在 host 上收集到一个 bounce buffer，每个 FA 张量一次 H2D，一次 `index_put_`，最后一次 join（731-824） | `source-overlay/remote_ra_growing_worker/tensors.py`（行号如左） |
| S17 | 策略注册表：query 插件的 hook、`FillContext`（含 `page_vectors`、`tokens`、`encode`）、`hooked_fill`；"Token-level selection is out of scope" | `source-overlay/policy_registry.py:1-50 (47), 107-155, 173-230` |
| S18 | token 聚集设计说明：第 1 节做什么（7-23）；第 2 节为什么精确：K 在 RoPE 之后存储、q 用逻辑位置、prefix 顺序无关、48 个 GDN 层不受影响、不能屏蔽单个 slot（25-33）；第 3 节预算 shared_blocks_v1、契约、回执、证据（35-43）；第 4 节 phase 2 第一个 step 内聚集、跨轮持有与释放、phase 1、容量门（45-53）；第 5 节代价（55-61）；第 6 节拒绝（63-68）；第 7 节离线回放（70-83）；第 8 节一致性与测试（85-89）；第 9 节待定问题：按 token 的 Goal 打分、按 head 选择、增量聚集（91-96） | [`docs/token-gather/DESIGN.md`](DESIGN.md)（中文版 [DESIGN.zh-CN.md](DESIGN.zh-CN.md) 同节） |
| S19 | main 上的 token 级选择说明：第 1 节研究的设定（14-21）；第 2 节约束：页 16 token、FA 只接受 16/32/64 块、只有每页一个向量（23-33）；第 4 节 "Not covered. Per-token Goal scoring…16 times today's count, and the exact scoring rule of the goal_query work"（66-67）；第 5 节问题 3（"保留标记所在 16-token 块"的消融）与问题 4（75-77） | `docs/token-selection/DESIGN.md` 在 `main` 上（`f18b8e5`；行号按该提交；不在本分支：其内容现在在 [docs/policy-registry.zh-CN.md](../policy-registry.zh-CN.md) 第 6 节） |
| S20 | 策略注册表文档第 6 节：token 级策略的三种做法与代价；第 9 节未做项："per-token Goal scores (one vector per Page)" | `docs/policy-registry.md:249-278, 458`，按 `a565e84`（之后已改写：token 级说明现在是 [docs/policy-registry.zh-CN.md](../policy-registry.zh-CN.md) 第 6 节，未做项列表在第 9 节） |
| S21 | 设计文档：phase 1 产出关闭页的向量和观察 query（19）；向量的定义（pre-final-norm、128 维投影、seed、SHA、int8）（34）；GPU 批量与 CPU 逐向量压缩约 0.35 % 的向量差 1 个元素（47） | `docs/design.md:19, 34, 47` |
| S22 | README：新的页向量分数和 token 级策略需要更多 runtime 支持 | `README.md:192`，按 `a565e84`（README 之后已重写；token 级选择它指向 [docs/policy-registry.zh-CN.md](../policy-registry.zh-CN.md) 第 6 节） |

**GPU 实测（已核验）**

| ID | 事实 | 位置 |
| --- | --- | --- |
| S23 | G1 聚集探针（NVIDIA GH200 120GB，HND）：arena `chunk_slots` 256、`page_bytes` 1,048,576、pinned；`c_dram_gather_median_ms` 69.0626、`c_dram_host_index_median_ms` 59.2726、`decision` `UVA_ADVISED`；E 集 DRAM：144 token、`gather_bytes` 9,437,184、gather 中位 53.4005 ms、host index 49.8355 ms、restore 84 页 9.1912 ms、`restore_bytes` 88,080,384；E 集驻留 73 页：gather 3.4555 ms；C 集 DRAM：1,024 token、gather 69.0626 ms、host index 59.2726 ms、restore 498 页 38.8763 ms、`restore_bytes` 522,190,848；C 集驻留 506 页：15.4129 ms；`threshold_ms` 10；设备 | GPU 作业 3603240：探针的 `gather.json` |
| S24 | gath32ti（goal_query + `--token-gather markers`）：counts：选择性请求 1,210，picks 均值 133.755，gathered 均值 110.581 / p95 317 / max 462，blocks 均值 7.274，host_tokens 126,474，device_tokens 14,358；guardrails：admission_engine_ms p50 230.212，`gather_time` NOT_RECORDED，latency_s p50 8.549，restore_h2d_mib 均值 226.351；capacity waits 0 | GPU 作业 3603443：其 `gather.json` |
| S25 | gath32ti 的 prompt 长度分布：[0, 32,768) 1,257 次、[32,768, 65,536) 237、[65,536, 98,304) 65、更长 0；表 1：1,559 次调用、1,210 个选择性 turn、最大 prompt 97,778 | GPU 作业 3603443：其 prompt 长度表与表 1 |
| S26 | gath32ti 的审计：selection_ms；选择性请求 admission_engine_ms p50 244.99；latency；W1：非 phase-1 H2D = k×146.8 MiB（GDN 状态）+ restore 页×1 MiB + DRAM 聚集 token×64 KiB，聚集占 2.24 %、每请求 5.1 MiB；W5：聚集在第一个 phase-2 step 内执行，没有逐请求聚集时间，每次调用生成 358.5 token；Slurm `--mem 113560M`、16 CPU、1 GPU；pool wall 2,845.1 s；16 个会话同时运行、thinking 开、max_tokens 4096；重建 35、最大 prompt 97,778 | GPU 作业 3603443：其证据的审计读数 |
| S27 | gath32ti：每个请求 capacity 512 页；fill room 均值 315.7 页 | GPU 作业 3603443：由同一审计推导 |

**我们自己的早期笔记与实验**

| ID | 事实 | 位置 |
| --- | --- | --- |
| S28 | 11-13 的 token 级设计笔记：研究的设定（10-23 讨论按 head 选择；11-04 作者在自己的原生 Mini-SWE 栈、H200 上，不是我们的 runtime；11-11 "Ours-token"：8,192 预算、thinking 开，保护系统提示 + 首条 user、最近两轮最多 1,024 token，"remaining slots go to tokens with the highest Goal scores"，条件 B–E）；goalvllmt（作业 3576565）候选页均值 1,906、obs262t 1,492；mandatory 均值 143.5 / 138.9 页、recent 49 / 48、room 319 / 325 | 我们 2026-11-13 的 token 级设计笔记（不在本仓库） |
| S29 | exp-subpage-score（功能提交 `32f36a1`；本地分支 `exp-subpage-score` 的 HEAD 是 `5a8453f`，它只改了 `check_git_release.py` 的超时；未推送，只跑过 CPU；设计写于 2026-10，P = 784）：每个 784-token 页用全部 16-token 子页打分；最后一个子页就是页向量；子页向量在设备上投影、放进 phase1/complete 回执的签名体之外，不新增每请求哈希；为什么在 GPU 上投影（一行 20 KiB、Python 归一化约 2 ms/行）；engine 代价、典型 turn 约 1,500 token、证据增长、"No hashing is added"；一步的 `pre_final_norm` 副本最多 16,384 行；改动的文件：`remote_ra_live_v2` 中 6 个、新模块 `subpage_score.py`、`serve.py`、README；19 个测试；一次本地测量 | 提交 `32f36a1` 与 exp-subpage-score 的设计说明（不在本仓库） |
| S30 | 选择一致性工具：从 worker 帧重建页目录、取 phase-1 帧中的 query、快路径与参照路径都跑并与记录比较，GH200 fp32 末位差异用邻域搜索恢复 query（1-19）；只重放 observation_query 和 random_fill，遇到其他策略（包括 goal_query）直接报错 "is not replayed (no rewriter here)"（163-168）；注释说 perf-swe16 时真实 agent 池服务器约 2,000 个原生回执、每个最多约 1 MB（171-172） | `harnesses/perf-swe32/postrun/selection_parity.py:1-19, 163-168, 171-172` |
| S31 | gather harness 的一致性重放：goal_query 的 Goal query 不重放，只在签名的 8 位小数分数的余弦序上重放 `token_gather.turn`（`gather_only`） | token 聚集 harness 的 `postrun/selection_parity.py`（不在本仓库） |

**本说明的本地测量（x86 Windows，不是 GH200）**

| ID | 事实 | 位置 |
| --- | --- | --- |
| S32 | 本地代理测量（Python 3.12.7、numpy 1.26.4、32 逻辑核；没有限制线程数，numpy 的 BLAS 用默认设置，即全部 32 个逻辑核；1 / 4 线程的复测见 S34）：按路径加载 integrate-1114 的 `dram_kv/page_vectors.py` 与 `remote_ra_live_v2/fast_selection.py`（不导入包）。线格式 455.9 B/向量（紧凑分隔符）；Python 表示 2,800.9 B/向量，加记忆化 `unit_values` 7,001.0 B；`json.loads` 8.46 µs、构造+校验 16.91 µs、`unit_values` 21.95 µs（每向量）；签名 406 个向量 187,620 B / 2.766 ms，6,500 个 3,007,475 B / 45.741 ms；`fast_selection.cosines`：1,906 列 1.195 ms、16,384 列 6.368 ms、30,496 列 13.486 ms、262,144 列 179.195 ms；evidence `scores` 字典（构造 / JSON / 字节）：1,906：0.63 / 0.60 ms / 33,944 B，30,496：10.47 / 8.96 ms / 580,173 B，262,144：106.15 / 77.48 ms / 5,233,706 B；整数精确余弦：30,496 列 int8 3.72 MiB，由 int8 每轮转换 3.221 ms、常驻 f32 副本 0.286 ms，前 5,040 排序 0.71 ms，全排序 1.83 ms；262,144 列 int8 32.0 MiB，28.663 / 4.248 ms，3.33 / 20.98 ms；两种规模 float32 点积 = int64 点积（逐元素相等）；`PageColumns` 1,024 B/向量 | 附录 A |
| S33 | 用 runtime 自己的 `token_gather.plan` 随机验证 2,000 次：picks = 前 room × 16 个 token 时，被取的页正好是 16 个 token 都被选中的页，其余 picks 全部聚集，pad 为 0 | 附录 A（输出 "2000 trials PASS"） |

**独立审计及其复测（2026-11-15）**

| ID | 事实 | 位置 |
| --- | --- | --- |
| S34 | 独立审计：11 项必改、20 项建议。在同一台机器上复测（x86、1 / 4 线程，非 GH200）：线格式 453.7 B/向量；Python 表示 2,737 / 6,937 B（含记忆化）；摄入 54.0 µs（4 线程）/ 60.0 µs（1 线程）；签名 6,500 个向量 53.1 / 63.7 ms；`fast_selection.cosines` 262,144 列 178.96 / 207.15 ms；整数余弦（由 int8）30,496 列 3.60 / 4.02 ms，262,144 列 39.15 / 37.80 ms；常驻 f32 副本 0.269 / 0.527 ms 和 6.152 / 9.655 ms，副本 14.89 MiB 与 128.0 MiB；前 5,040 排序 2.72 / 2.83 ms（262,144）；evidence 分数 262,144 构造 + JSON 118.8 + 87.8 ms，其排序 JSON + HMAC 约 112.0 / 114.1 ms；float32 点积与 int64 点积相等（BLAS、打乱维度、顺序累加、torch f32 与 bf16 输入 + f32 累加），bf16 输出的 matmul 不相等；整数余弦与上游最大差 1.1e-16，30,496 列排名不变，262,144 列 2 个位置（一对交换），前 5,040 集合不变；host 索引代理（16 个 FA 张量、一个未 pin 的 256-slot chunk）144 / 1,024 / 5,051 token 为 4.156 / 15.587 / 67.87 ms；每 token 约 190 个 Python 对象。fill 检查 3,000 次 × 4 种变体（`plan`、`turn` 返回全部候选、标记在前、quota）全部成立；`turn()` 的开销：30,496 个候选 6.7 ms（只给 room × 16 个时 4.08 ms），259,136 个候选 56.47 ms（7.11 ms） | 一次独立审计（2026-11-15）及其复测与填充检查（不在本仓库） |
| S35 | 容量门：有 gather pin 时 `most = largest`（quota 为 0）或 `min(largest, pages(quota, P))`，restores = min(largest, 非驻留页 + most)；注释说明聚集块也是新块、与页共享预算，上一轮的到 commit 才释放 | `source-overlay/remote_ra_live_v2/capacity.py:220-226` |
| S36 | 第一个 phase-2 step 里，restore 之后调用 `self.io.gather_tokens(...)`（同步）；`gather_tokens` 最后调用 `self.drain()` | `source-overlay/remote_ra_live_v2/allocation_owner.py:1281-1290`；`source-overlay/remote_ra_growing_worker/tensors.py:821-824` |
| S37 | engine 的 native authority：`sign` = 对规范摘要 `digest(value)` 做 HMAC-SHA256，`verify` 再算一遍比较 | `source-overlay/remote_ra_native_growing/contract.py:393-399` |
| S38 | G1 探针的决策规则：C 集 DRAM 聚集中位超过 10 ms 时给出 `UVA_ADVISED`，否则 `ARENA_GATHER_OK`；只是测量，不会让探针失败 | GPU 作业 3603240（探针在 token 聚集 harness 中，不在本仓库） |
| S39 | Qwen 发布版保留 16,384 的 max_num_batched_tokens（一次原生前向最多 16,384 个 token） | `serve.py:562-567` |
