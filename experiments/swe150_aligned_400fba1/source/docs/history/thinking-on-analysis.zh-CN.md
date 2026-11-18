# Qwen3.6-27B 开启 thinking：分析（2026-10-30，只读，分支 thinking-on @ cf2b28d）

[English](thinking-on-analysis.md) | 简体中文

> **未审阅。** 一份代理在 2026-10-30 写的分析，按原样保留；没有人逐行审阅过，其中的数字也没有经过审计。**2026-11-16 冻结**（在 `ccee5c6`）：由 docs-reorg-1116 从 `docs/thinking-on/ANALYSIS.zh-CN.md` 移到这里，除这段说明和语言链接外未作改动；不再更新。据此实现的内容：[docs/running.zh-CN.md](../running.zh-CN.md#6-harness-接口) 的 6.8 和 6.9 节、[docs/thinking-cut/DESIGN.zh-CN.md](../thinking-cut/DESIGN.zh-CN.md) 和 [docs/thinking-gemma/DESIGN.zh-CN.md](../thinking-gemma/DESIGN.zh-CN.md)；它目前的限制见 [docs/known-limits.zh-CN.md](../known-limits.zh-CN.md)。

Agent 报告，由 Claude 在内容上原样保存（尚未逐行审阅）。渲染结果用固定的
tokenizer 和 Qwen 模板（sha256 e84f32a2...）检查过，用的是一个不在本仓库中的本地检查脚本。
起初没有在本地核实的是：客户端的行为（Mini-SWE-Agent 2.4.2 及其 litellm；这里没有它们的源码）。
之后，附录中的 stub 探测运行了 Mini-SWE-Agent 2.4.2 和 Terminus 2（Harbor 0.21.0，litellm 1.97.0）。

## 状态（2026-11-08）

- **已实现，在 main 上：** 第 1 阶段 `--enable-thinking`（`94888e1`；Qwen 上的 mswea-xml 和 chat-text），以及方案 (a)，即
  `--thinking-history strip`（`8ca8511`；Qwen 上的 chat-text；[docs/thinking-cut/DESIGN.zh-CN.md](../thinking-cut/DESIGN.zh-CN.md)）。
- **GPU：** `--enable-thinking --thinking-history strip` 的功能性 GPU 检查已通过（作业 3493649、3500646、
  3499759），都在 runtime `8ca8511` 上：作业 3493649 THG1_PASS_VERIFIED（G1），作业 3500646 THG1B_PASS_VERIFIED（G1b），
  以及作业 3499759 TB16_PASS_VERIFIED（一次 Terminal-Bench dry run）。尚无完整的 benchmark。
- **第 1 阶段的 keep 模式**（`--enable-thinking` 不加 `--thinking-history`）只作为作业 3500604 的
  kref / kovl 服务对在 GPU 上运行过（mswea-xml，runtime `c07d42a`）：一次 `--goal-overlap` 检查，CMG1_PASS_VERIFIED，该服务对为 EXACT（
  18 个 lockstep 轮次中 18 个的 Goal 相同）。那是在两侧都用 keep 模式时比较开与不开 `--goal-overlap`；
  keep 模式本身未经验证。使用它的 SWE-bench 运行正在排队（作业 3539242、3539260、3540110 和 3549065）。
  此后（2026-11-15）：作业 3539242（`recent_fill`）和 3539260（`observation_query`）已运行，都在 `60c5d21` 上，
  都为 SWE32_PASS_VERIFIED；作业 3540110 和 3549065 在开始之前就被取消了（开启 thinking 的 `akv_rule`
  在作业 3577833 中运行，SWE32_PARTIAL_VERIFIED）。
- **已确定：** 开启 thinking 时服务端把 reasoning 与 content 分开（content 为 C，`reasoning_content`
  和 `reasoning` 为 R），MarkerStop 忽略 R 中的标记（第 1 阶段）。方案 (a)，即截断并重新 prefill，已实现为
  strip。Gemma 保持关闭 thinking。
- 本页其余部分是 2026-10-30 的分析，基于 `cf2b28d` 的代码。

## 状态（2026-10-30）：第 1 阶段已在分支 thinking-on 上实现，仅经过 CPU 测试

`serve.py --enable-thinking`（需显式开启；Qwen，`mswea-xml` 和 `chat-text`；[docs/running.zh-CN.md 6.8 节](../running.zh-CN.md#68-开启-thinking第一阶段--enable-thinking未经验证)、
`remote_ra_live_v2/thinking.py`、测试 `test_thinking_on.py`）：
开启 thinking 的生成头和模板参数固定在 `startup.json` / `ready.json` 中；响应切分（content 为 C，`reasoning_content` 和 `reasoning` 为 R；
没有 `</think>` 时全部算作 reasoning，content 为 `""`）；MarkerStop 只在 `</think>` 之后启用；
客户端回传 R 时为 append（content 和 reasoning 在 `strip()` 之后比较），否则走现有的重建（第 2 阶段，
截断并重新 prefill，尚未实现）；行中的 `thinking` 记录；Goal 的上一条回复只用 C（已提请
Goal 负责人注意）。未经 GPU 验证；没有跑 benchmark。Gemma 保持关闭；kv-session 不变。

## 附录（2026-10-30，探测结果；取代下文关于可比性的说法）

探测：它的脚本、结果日志和检查笔记是一份本地 scratch 副本，不在本仓库中
（可向维护者索取）。只读地 WebFetch 了合作者的仓库 token_efficiency @ 068efec，并在 agent
主机上运行了一个 stub server。

- **合作者的 baseline 开启 thinking 运行**（来自那次 WebFetch 读取；
  他们的仓库不属于本仓库，这一点也没有对照本地副本重新检查）（配置：enable_thinking true；max_tokens 4096，含
  thinking）。他们自己渲染 prompt（tebench/backends/base.py：带 enable_thinking 的 apply_chat_template；任何地方都没有
  preserve_thinking；生成的文本在 `</think>` 处切分）。AgentKV、RepairKV、Sliding Window 和 Random 使用
  kv-session 后端（追加原始 token，保留 reasoning）。Vanilla 和 IntentKV
  每轮重新渲染整段历史：
  - Terminal-Bench（Terminus 2，interleaved_thinking True，observation 的 role 为 user）：stock 模板丢掉之前的每一条
    reasoning，包括最后一条回复的 -> **R 被丢弃**（stock 默认）。
  - SWE-bench（mini-swe-agent 2.4.6 tool-calling，回传 reasoning_content，observation 的 role 为 tool）：
    模板保留最后一条真实 user 消息之后的 reasoning -> **R 被保留**（除非出现一条格式错误的 user 消息）。
  - MemoryAgentBench：阅读时 R 被保留（chunk 的 role 为 tool），到提问时被丢弃（role 为 user）。
  所以“默认 baseline 丢弃 R”（下文第 3-4 节）只对他们的 Terminal-Bench 运行成立。
- **我们的客户端对 stub 的表现**（A：`R\n</think>\n\nC` 内联在 content 中，即 runtime 在 `cf2b28d` 返回的形式
  （mswea_route.py ~49-52、chat_text_route.py ~176-181 把 reasoning 字段置空；自第 1 阶段起，`--enable-thinking`
  在 content 中返回 C，在 `reasoning_content` 和 `reasoning` 中返回 R）；B：reasoning_content；B2：reasoning）：
  - Mini-SWE-Agent 2.4.2（经由 litellm；探测日志中没有它的版本）：A 把 R 内联保留在 content 中；
    R 中一个完整的 `<mswea_bash_command>` 草稿会引起 FormatError，下一个请求丢掉那条回复（不是
    append）。B / B2：R 作为单独的 reasoning_content 字段回传（litellm 把 `reasoning` 映射为
    reasoning_content）；动作可以解析。
  - Terminus 2（Harbor 0.21.0，litellm 1.97.0）：A 破坏解析（解析器从 R 中取了一个 JSON 对象；第 2 轮执行了
    R 中的一条草稿命令）。B / B2 在 interleaved_thinking 关闭时（我们的默认）：只回传 {role, content}，R
    被丢弃；开启时：reasoning_content 作为单独字段回传。动作可以解析。
  - 两个客户端都不发送 chat_template_kwargs；也都不发送顶层的 `reasoning` 键，而它是我们
    overlay 的 chat_utils 唯一读取的字段（stock vLLM 0.24 会忽略回传的 reasoning_content；没有在上游检查）。
- **由此得出：** 每个方案都需要服务端在响应中把 reasoning 与 content 分开（A 会破坏 Terminus
  的解析，并让 Mini 出现格式错误），而且 MarkerStop 必须忽略 R 中的标记。
  哪种历史语义能让 Ours 得到与合作者的 Vanilla 相同的输入，取决于 benchmark：Terminal-Bench 丢弃 R（需要方案 (a)，
  截断并重新 prefill），用他们的 tool-calling harness 的 SWE-bench 保留 R（reasoning 被分开并回传之后，
  只追加就可行），MemoryAgentBench 在提问之前保留 R。注意他们的 SWE harness（mini 2.4.6 tool-calling）
  与我们的（2.4.2 XML PoC prompt）不同。

## 结论

组里的直觉对了一半：使用 stock 模板时，
下一个 prompt 与上一轮的不同只出现在最后一条回复上，所以更早历史的 KV 不受影响。但本 runtime 目前无法利用这一点：任何不是
append 的请求都会重建整个 session；没有“只截掉最后一条回复”的路径。

## 1. 开启 thinking 的一轮之后的下一个 prompt

- 开启 thinking 的生成头 `<|im_start|>assistant\n<think>\n`（5 个 token；关闭时 7 个）。模型生成 `R\n</think>\n\nC<|im_end|>`
  （R = reasoning，C = 可见的回复）。`<think>`（248068）和 `</think>`（248069）不是特殊 token，所以在
  skip_special_tokens 之后 content 仍然是 `R\n</think>\n\nC`。
- 历史渲染：不加 preserve_thinking 时，最后一条真实 user 消息之前的 assistant 轮次渲染为
  `<|im_start|>assistant\nC<|im_end|>`；即使客户端回传 `R</think>C`，模板也会在 `</think>` 处切掉 R。
  加 preserve_thinking 时（R 在 reasoning_content 中或内联），渲染结果与服务端的 append 流逐 token
  相同（由报告的本地脚本检查，该脚本没有保留；未重新检查）。
- 分歧：与服务端流的公共前缀在最后一条回复的 `<|im_start|>assistant\n` 处结束（生成头起点
  + 3）；被去掉的：`<think>`、`\n`、R、`\n</think>\n\n`（约 |R| + 5 个 token）；C 从文本重新编码（通常是相同的 id，
  但不保证）。更早的回复两边都没有 R（未重新检查）。

| 协议 / harness | 客户端回传什么 | 与服务端流相比 |
| --- | --- | --- |
| mswea-xml + Mini-SWE-Agent | 取决于服务端的响应 | 见下 |
| chat-text + Terminus 2 | 默认只有 C | 上面的分歧 |
| kv-session（MemoryAgentBench） | 只有尾部 | 无 |

- mswea-xml：按 content.strip() 匹配确切的生成 id，而不是按模板渲染。
  如果服务端在 content 中返回 `R</think>C`（如 `cf2b28d` 那样），litellm 很可能原样透传（它只在 content
  以 `<think>` 开头时才切分 reasoning，而它在 prompt 中）-> 仍是只追加，但 R 留在历史中；MarkerStop 会因
  R 中的 `</mswea_bash_command>` 而提前停止，agent 的动作正则也会扫描 R。如果服务端分出
  reasoning_content，agent 只回传 C -> 上面的分歧。
  stock 对照在两种情况下都经由模板丢掉 R。
- chat-text / Terminus 2：content 必须只有 C（JSON 解析器取第一个配平的 `{...}`；
  R 中的花括号会破坏它）。默认 interleaved_thinking=False -> 只有 C -> 分歧。Harbor 0.21.0 有一个 interleaved_thinking kwarg，
  它把 reasoning_content 放进历史（harbor/llms/chat.py）；litellm 的 hosted_vllm 是否转发它尚未核实。
- kv-session：客户端只发送尾部；服务端流保留 R，即 preserve_thinking 的渲染
  （test_kv_session_protocol.py）。只有 Vanilla 对照不同：在提问那一轮它丢掉之前所有的 R。

## 2. runtime 在 cf2b28d 上的做法及其代价

- 在 `cf2b28d` 上，mswea-xml 和 chat-text 无法开启 thinking：生成头和模板 kwargs 是固定的；
  客户端的 chat_template_kwargs 不起作用。自第 1 阶段起，`--enable-thinking` 为 Qwen 上的两种协议固定开启 thinking
  的生成头和模板参数（`remote_ra_live_v2/thinking.py`）。
- 只改生成头、而客户端去掉 R 时：extends() 每轮都失败 -> 每轮都重建（close_session
  释放常驻 Page、tail 和 GDN carry；selector 和 Goal 重置；从 0 开始完整地重新 prefill；local turn 0 -> cold_start
  选页，所有 Page 激活：query 策略每轮都退化为 full）。没有部分截断的路径
  （turn_geometry 要求 T[:C] == computed，否则 SESSION_PREFIX_MISMATCH；[docs/running.zh-CN.md 6.3 节](../running.zh-CN.md#63-错误处理)
  把历史改动列为硬失败）。
- 如果截掉最后一条回复再重新 prefill：每轮重新 prefill |C| + 3 个 token（关闭 thinking 时回复的平均长度：SWE 约 223
  token，作业 3404230：500,619 / 2,245；TB 约 557，作业 3434533：195,484 / 351），
  与几万 token 的 prompt 相比可以忽略。浪费的：R 的 KV 和约 |R|/16 个已封存 Page（每个 1 MiB GPU + 一份 arena 主机副本 + 向量；每轮约 |R|/16
  MiB 的 D2H）。GDN 需要截断点 e 处的状态（阶段 1 结束时的 probe 状态，目前不提交）：
  一个新的回滚 carry，每个 session 锁页约 154 MB。跨越截断点的 tail Page 在 decode 期间已被封存，其向量中含有 obs、
  生成头和 R：要把它解封回 tail（valid_end = e）并丢掉向量。
  选页几何回到不含 R 的历史（与关闭 thinking 时相同）；recent 集合 = obs Page + 重新 prefill 的 C Page
  （归入上一轮）；必保留页不变。
- 如果 R 留在历史中：没有重算或回滚，但每轮多 |R| + 5 个 token：
  Page、候选、arena 和上下文增长得更快（Terminus 更早做 summary -> 更多重建；固定预算下激活比例更低；1024 token 的
  recent 集合大多是上一轮的 reasoning）。
- 所有方案：回复上限只有 1536 / 4096（thinking 需要更大的选项：MAX_TOKENS_CHOICES、
  容量规则 ceil(max_tokens / P)、锁页主机预算、上下文检查）；Goal 的上一条回复应只用 C。

## 3. 方案（按在本 runtime 中实际的侵入程度；只追加是引擎的不变式，所以 (a) 侵入最深）

- (c) 服务端保留 R：最后一条 assistant content 等于存储的 C 时算作延续，
  流保留 R 的确切 id。只改协议层：开启生成头；响应切分为 content + reasoning_content；MarkerStop 只在
  `</think>` 之后启用；新的 extends 规则；新的回复上限选项；证据记录 R；离线参照用
  preserve_thinking 模板渲染。引擎不变。代价：模型的输入与客户端的消息不同。
- (b) 客户端回传 R + preserve_thinking：(c) 加上 Terminus 的 interleaved_thinking: true；TEMPLATE_KWARGS
  {enable_thinking: true, preserve_thinking: true}；extends 比较 (reasoning_content, content)；
  重建的模板渲染等于流。设置相同的 stock baseline 可以逐 token 比较。
  Mini-SWE-Agent 没有这样的开关（R 只能内联在 content 中）。
- (a) 截掉最后一条回复并重新 prefill（strip 语义）：回滚到 e 的几何路径、GDN 回滚 carry、解封、
  Page 释放（GPU、arena、空闲释放、分配所有者）、选页和轮次记录的回滚；证据
  truncate{cut, dropped, reprefill}；verifier 从“每个 token 只计一次”改为
  “每个流版本只计一次”。与 stock 默认的开启 thinking 逐 token 一致（从文本重新编码 C
  甚至会消除关闭 thinking 时看到的每条回复 4 个 token 的差异）。
- 测试：CPU：固定 tokenizer 的模板等价（enable_thinking=True，preserve 开/关）、TB ATIF 178 个 trial
  的回放、一次 Mini-SWE-Agent 回放；(a) 还需要覆盖 fast path、admission overlap、deferred IO、
  空闲释放的引擎 fake。GPU：每个策略一次 SWE 和一次 TB dry run；(a) 还需要与 stock eager 回放做 token 对齐。
- Gemma：它的模板对纯文本轮次总是去掉 thinking（preserve_thinking 只用于 tool_calls），开启 thinking
  会把 `<|think|>` 注入第一个 system 轮次（改变必保留前缀），而且窗口层的回滚不同于
  GDN：(b) 不可能，(c) 没有模板参照，(a) 是单独的工作。建议：Gemma 保持关闭。（2026-11-09 起不再适用：维护者为 Gemma 选择了 (c) 和 (a)，见 [docs/thinking-gemma/DESIGN.md](../thinking-gemma/DESIGN.md)。）
- 可比性：没有一个方案能与过去关闭 thinking 的运行比较。对于合作者开启 thinking 的
  baseline（stock + 默认模板 / 默认 harness，历史丢弃 R），只有 (a) 给出相同的输入；(b) 和 (c)
  只能与 preserve_thinking 的 baseline 比较。

## 4. 工作量（粗估）与建议

| 方案 | 工作量 | 风险 |
| --- | --- | --- |
| kv-session | 不需要代码，1 个 GPU 作业（Vanilla 对照需要 preserve_thinking 才可比） | 低 |
| (c) | 1-2 天 CPU + dry run | 低 |
| (b) | (c) + 0.5-1 天 | 低 |
| (a) | 1.5-3 周 + 若干 GPU 作业 | 最高 |

先做两项不需要 GPU 的检查：(1) 合作者开启 thinking 的 baseline 的渲染和历史设置；(2) 在
agent 主机上用一个 stub server 看 Mini-SWE-Agent / litellm 和开启 interleaved_thinking 的 Harbor 实际回传什么。
然后：要与默认开启 thinking 的 baseline 逐 token 可比 -> (a)；只要证明 KV 复用在开启 thinking 时可行，
或者 baseline 可以换成 preserve -> TB 用 (b)，SWE 用 (c)，MAB 直接做 GPU 检查。
