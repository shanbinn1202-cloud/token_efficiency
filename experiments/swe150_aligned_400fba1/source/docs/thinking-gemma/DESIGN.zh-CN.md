# Gemma 4 的 thinking “与 Qwen 相同”（设计，2026-11-09）

[English](DESIGN.md) | 简体中文

Gemma 4 31B IT profile 开启 thinking 的设计，由维护者于 2026-11-09 决定：完整范围，即 Tier 1（keep：`mswea-xml` 和 `chat-text` 上的
`--enable-thinking`，`--kv-session-thinking on`）和 Tier 2（`chat-text` 上的 `--thinking-history strip`，形式 E）。按实际实现：工作在两个
分支上完成，`gemma-thinking`（本说明；Tier 1 以及 strip 的 transcript 一侧）和 `gemma-thinking-hold`（4.4 节的被保留的切点窗口），于
2026-11-10 在 `gemma-thinking-int`（`3219296`）上集成，自 integrate-1112（`4ebff97`，2026-11-15）起在 `main` 上：Gemma 上的
`--thinking-history strip` 已端到端接通（4.4 节）。它的 GPU 检查（7.2 节）及其判定见 [docs/status.zh-CN.md](../status.zh-CN.md) 和
[历史](../history.zh-CN.md#历史)；Gemma keep 模式仍然成立的限制见 [docs/known-limits.zh-CN.md](../known-limits.zh-CN.md) 1.2。下面两段
“已实现的内容”记录的是 2026-11-10 的分支及其 CPU 测试；本页其余部分是设计记录。

**`gemma-thinking` 上已实现的内容（2026-11-10，CPU 测试）。** Tier 1：Gemma 的 XML 协议和 chat-text 上的 `--enable-thinking`，以及
`--kv-session-thinking on`（launcher、固定值、对照固定模板的启动检查、transcript、路由、MarkerStop 及其异步预测），
`tests/test_gemma_thinking_on.py`。Tier 2，strip 的 transcript 一侧：Gemma 的回复片段和 head（K = |H|）、它的模板检查、在切点处的提交和
重新 prefill、对 `--async-scheduling` 的拒绝，`tests/test_gemma_thinking_cut.py`。thinking 关闭时，以及在每条 Qwen 路径上，启动、记录、
transcript 和响应都与 `ba4d1ba` 相同（覆盖 729 次启动和 35 个协议脚本的 identity probe：0 处差异）。在 Gemma 上 keep 和 strip 的
`chat_text` 记录相同（生成头和模板参数）；`thinking` 记录的 `history` 区分两者。

**`gemma-thinking-int` 上已实现的内容（2026-11-10，CPU 测试）。** 被保留的切点窗口（方案 G）：`remote_ra_live_v2/window_hold.py` 及其引擎、worker 和
容量部分（4.4 节）。`thinking_cut.attach_engine` 和 `attach_worker` 把窗口布局交给它，窗口 worker 接收并回显 strip 记录，启动检查要求
Gemma strip 配置带有 hold slot，`serve.py --window-hold-slots H` 设定它们的数目（不给时：放得下的最多数目）。测试：
`tests/test_gemma_hold.py` 和 `tests/test_gemma_strip_e2e.py`（7.1 节）。集成后代码树的 identity probe，全部 0 处差异：对照 `ba4d1ba`，
`gemma-thinking` 的 launcher probe 的 729 项（Qwen 的 thinking 关、keep 和 strip；Gemma 的 thinking 关；Goal rewriter 的命令；启动证据）、
它的 35 个协议脚本、窗口布局的 10 次引擎运行（122 个文件）以及 `gemma-thinking-hold` 的 launcher probe 的 677 项；对照
`gemma-thinking`，Gemma keep 模式和 `--kv-session-thinking on` 的 278 项（启动和 transcript 步骤）。不开 `--thinking-history strip` 时
不预留任何 hold slot，hold 的代码一行也不运行。

下文的引用：`TPL` 是固定的 Gemma `chat_template.jinja`（SHA-256 `ae53464b...`，固定在 `remote_ra_live_v2/model_profile.py` 中；它不在
代码树里，测试通过 `REMOTE_RA_GEMMA4_METADATA` 读取它），`TCFG` 是它的 `tokenizer_config.json`。runtime 的行号是 `main` `ba4d1ba` 的。

## 0. 决定（2026-11-09）

1. **范围。** Gemma 4 profile 的 Tier 1（`mswea-xml` 和 `chat-text` 上的 keep，`--kv-session-thinking on`）和 Tier 2
   （`chat-text` 上的 `--thinking-history strip`，形式 E）。
2. **Gemma 上 Terminal-Bench 的历史：strip**，即 stock / Vanilla 的输入（1.4 节）。
3. **切点处的窗口：方案 G**，GPU 上的 hold，由 H 个 hold slot 组成的共享预留加一个解码上限（4.3 节；实际上是对在途 strip 轮次的上限）。
   H = 0（不用 strip）时一切逐字节不变；launcher 按选择性规则对照池子设定或拒绝 H。主机拷贝（方案 H）和 32 GiB 的 KV 没有实现。
4. **回复切分：** 由 vLLM 0.24.0 的 `gemma4` reasoning parser 决定（第 2 节）。它与“没有 channel = 只有 content；未闭合的 channel =
   全部是 reasoning”一致，并为不在开头的 channel、没有开启的闭合以及没有 channel 的 `thought` 前缀加了规则；runtime 在这些地方遵循 vLLM。
5. **Baseline：** 合作者仓库中 Gemma 的每个 SWE-bench 和 MemoryAgentBench baseline 配置都设 `enable_thinking: true`（第 5 节）。
6. **MemoryAgentBench：** runtime 一侧（Gemma 的 `--kv-session-thinking on`）属于 Tier 1。MAB 客户端（`examples/memoryagentbench/`）
   是其他人的代码，保留它 thinking 关的 Gemma 尾部。
7. **v1 中的拒绝：** Gemma strip 加 `--async-scheduling`；`mswea-xml` 和 `kv-session` 上的 strip（与 Qwen 相同）。
8. **顺序：** 现在在分支上做 CPU 开发；GPU 检查在 Gemma 的异步运行核验之后；进入 `main` 要在 GPU 验证和维护者批准之后。
9. **不变式：** Qwen 的路径保持逐字节不变，thinking 关的 Gemma 也是（identity probe）。

## 1. Gemma 4 自己的 thinking 格式（固定的模板和 tokenizer）

### 1.1 Token

| Token | Id | 特殊 token |
| --- | --- | --- |
| `<\|think\|>` | 98 | 是（`TCFG` `think_token`） |
| `<\|channel>` / `<channel\|>` | 100 / 101 | 是 |
| `<\|turn>` / `<turn\|>` | 105 / 106 | 是 |
| `<\|tool_call>` / `<tool_call\|>`、`<\|tool_response>` / `<tool_response\|>` | 48 / 49、50 / 51 | 是 |
| `thought` | 45518 | 否（普通 BPE；`thought\n` 是 `[45518, 107]`，`thought\n\n` 是 `[45518, 108]`） |

停止 id 是 {0, 1, 50, 106}；`<channel|>` 不会停止回复。

### 1.2 `enable_thinking` 改变了什么（`TPL:186-196`、`TPL:381-390`）

- **默认关闭**（`enable_thinking | default(false)`）；Qwen 的模板默认开启。stock 的 Gemma 对照要显式要求 thinking
  （`chat_template_kwargs: {"enable_thinking": true}`）。
- **必保留前缀会变。** 开启时，第一个 system 轮次以 `<|think|>\n` 开头。没有 system 消息时（Terminus 2 不发送），模板会加一个完整的
  system 轮次 `<bos><|turn>system\n<|think|>\n<turn|>\n`：多 7 个 token `[105, 9731, 107, 98, 107, 106, 107]`。有 system 消息时多 2 个
  （`<|think|>\n`）。Qwen 的这个开关只改变生成提示。
- **user 轮次之后的生成头：** 关闭时为 `<|turn>model\n<|channel>thought\n<channel|>`（7 个 token，即今天 thinking 关的服务的空 thought
  channel）；开启时为 **`<|turn>model\n`，3 个 token** `[105, 4368, 107]`：由模型自己打开 channel。
- **工具响应之后：** 关闭时没有生成头（model 轮次继续，即今天 kv-session 的 `header_tokens` 0）；开启时为
  **`<|channel>thought\n`，3 个 token** `[100, 45518, 107]`：模板在每个工具响应之后打开 thought channel。

### 1.3 模型输出什么（`TCFG` `response_schema`、`response_template`）

一条 thinking 回复是 `<|channel>thought\n` R `<channel|>` C `<turn|>`；channel 是可选的。两个后果，都与 Qwen 不同：

- **标记是特殊 token。** 跳过特殊 token 解码（chat-text 和 XML 就是这样做的）会丢掉它们：
  `<|channel>thought\nLet me look.<channel|>{"a": 1}` 解码为 `thought\nLet me look.{"a": 1}`。Qwen 的 `</think>` 不是特殊 token。
  Gemma 必须**按 id** 切分（第 2 节）。
- **历史把 reasoning 渲染为 `<|channel>thought\n` + R + `\n<channel|>`**（`TPL:242`），闭合之前有一个换行，而模型的原始输出可能没有。
  重新渲染 R 的客户端（任何 `preserve_thinking` 渲染）可能因为这个 `\n` token 与原始 id 不同；出现的频率要在 GPU 上测量（7.2 节）。

### 1.4 模板如何处理之前的 thought（`TPL:156-166`、`218-243`、`323-349`）

model 消息的 reasoning 只在最后一条 user 消息之后渲染，或在开 `preserve_thinking` 时对带 `tool_calls` 的消息渲染；它的 content 总是经过
`strip_thinking`（去掉每个 `<|channel>...<channel|>`，然后去掉首尾空白）。

| 对话形态 | 默认（`enable_thinking=True`） | `preserve_thinking=True` |
| --- | --- | --- |
| chat-text / 我们的 XML：observation 是 **user** 消息 | 每条过去的回复 `<\|turn>model\n{C}<turn\|>\n`，**R 被丢掉** | 相同（没有 `tool_calls`） |
| 一个 user 轮次内的工具循环（MAB 的 `read_next`，tool-calling 的 SWE harness） | 最后一条 user 消息之后 R 保留 | 相同 |
| 工具循环之后的一条新 user 消息（MAB 的提问） | 之前所有的 R 被丢掉 | R 保留 |

**对文本轮次，Gemma 的自然模式就是 strip**，没有任何模板参数能在那里保留 R。strip 的流与 stock 开启 thinking 的渲染完全相同；今天 thinking
关的流保留了空 channel，每条过去的回复比 stock 多 4 个 token。

## 2. 回复切分（决定 4：vLLM 0.24.0 的 `gemma4` parser）

在 `v0.24.0` tag 上阅读：注册表把 `gemma4` 指向 `gemma4_engine_reasoning_parser.Gemma4ParserReasoningAdapter`，即通用 parser 引擎
（`vllm/parser/engine/`）上 `vllm/parser/gemma4.py` `Gemma4Parser` 的适配器。我们的客户端都不用流式；非流式路径是
`extract_reasoning(model_output)`：

1. thinking 开启时，请求**带**特殊 token 解码（`skip_special_tokens=False`），所以文本中有这些标记。
2. 引擎每次调用都从 CONTENT 开始。CONTENT 中的 `<|channel>` -> REASONING；REASONING 中的 `<channel|>` -> CONTENT；CONTENT 中的
   `<channel|>` 被吞掉；没有转移的终结符（reasoning 中的第二个 `<|channel>`）作为该状态的文本保留（一个待定点，见下）。到结尾仍未闭合的
   reasoning 仍是 reasoning。
3. **注入**（只在第一次输入时）：没有 `<|channel>`、但含 `<channel|>`、或以 `thought\n` 开头、或等于 `thought` 的文本，前面加一个虚拟的
   `<|channel>`。
4. reasoning = reasoning 文本 `rstrip()` 之后（`strip_trailing_reasoning_whitespace`，默认开启），再去掉开头的 `thought\n`（恰好是
   `thought`：没有 reasoning）；为空 -> `None`。content = content 文本；为空 -> `None`。

runtime 在生成的 id 上应用相同的规则（`remote_ra_live_v2/thinking.py`，`Gemma4Split`）：去掉结尾的停止 id（vLLM 的文本不含它），在 100 /
101 处把 id 切成若干段，每段解码（跳过特殊 token，与今天的回复相同），由上面的状态机分配各段。响应带 `content` = C，reasoning 放在
`reasoning_content` 和 `reasoning` 中，与 Qwen 相同。

| 情况 | vLLM | Runtime | 决定 4 的说法 |
| --- | --- | --- | --- |
| 没有 channel，没有 `thought\n` 前缀 | 只有 content | 相同 | 相同 |
| 开头的 channel，已闭合 | reasoning（去掉前缀和末尾空白），之后是 content | 相同 | 相同 |
| channel 始终未闭合 | 全部是 reasoning，content 为 `None` | 全部是 reasoning，content 为 `""` | content 为 `""` |
| 在部分 content 之后出现 channel | 之前和之后的 content，channel 作为 reasoning | 相同 | （未涉及） |
| 有 `<channel\|>` 没有 `<\|channel>` | 它之前是 reasoning | 相同 | 只有 content |
| 没有 channel，文本以 `thought\n` 开头 | reasoning | 相同 | 只有 content |

剩下的差异：vLLM 发送 `null` 的地方 content 和 reasoning 为 `""`（Qwen 的规则，“content 永不为 null”）；runtime 按 id 匹配标记，vLLM
按文本匹配（`<|channel>` 的纯文本写法在 vLLM 中也会切分；模型把标记写成特殊 token）；回复中的其他特殊 token 在我们的文本中被跳过（与今天
相同），reasoning 中的 `<|tool_call>`（vLLM：reasoning 结束）在这些协议上不会出现，因为它们不提供工具。

**待定点：reasoning 中的第二个 `<|channel>`**（11-10 的审查，发现 4）。runtime 把嵌套的开启标记保留为 reasoning 的文本
（`<|channel>thought\nx<|channel>y<channel|>z` 得到 reasoning `x<|channel>y`；`thinking.py`，`Gemma4Split`）。在 `v0.24.0` tag 上对
`vllm/parser/gemma4.py` 的两次概括性阅读（WebFetch，审查者的）显示有一个 (REASONING, `THINK_START`) -> REASONING 的转移，没有输出，还有
一条关于吞掉它的注释，那会得到 `xy`。两次阅读都不是逐字节的，而测试中的 parser 模型（`test_gemma_thinking_on.vllm_gemma4`）是本说明
自己的理解，所以它的随机用例无法定论。只有 reasoning 文本（以及随之的 reasoning token 数）会不同：content，因而每个 strip prompt，两种
情况下都相同。在逐字节读过 vLLM 的 parser 之前（或 GG1 的检查 6 遇到这样的回复之前）保持待定；runtime 不变。

**MarkerStop（XML）。** action 标记只在到目前为止回复的 content 中计数（把上面的切分应用于输出 id）。`MarkerStop` 的增量论证加一条即可
成立：channel id（100 或 101）可以把之前的文本在 reasoning 和 content 之间移动，所以这样的 token 会解码整个输出，与含 `>` 的 token 一样。

## 3. 三个开关

### 3.1 对应关系

| Qwen | Gemma 中的对应 | Gemma 上的模板参照 |
| --- | --- | --- |
| kv-session `--kv-session-thinking on`：客户端渲染，服务追加原始 id，R 保留 | 相同，固定两个生成头：user 轮次之后 `<\|turn>model\n`，工具响应之后 `<\|channel>thought\n` | `enable_thinking=True, preserve_thinking=True`（带 tool call 的消息保留 R） |
| `--enable-thinking` keep（`mswea-xml`、`chat-text`）：流保留 R 的 id；只有 R 回来时追加，否则重建 | 规则相同；生成头 `<\|turn>model\n`；按 id 切分；MarkerStop 在 `<channel\|>` 之后 | 文本轮次没有参照：重建渲染 `<\|turn>model\n<\|channel>thought\n` R `<channel\|>` C `<turn\|>\n` |
| `--thinking-history strip`（`chat-text`）：形式 E，在 X 处提交，重新 prefill 返回的 C | transcript 相同；引擎在 X 处保留滑动窗口（第 4 节） | 默认模板：每条过去的回复 `<\|turn>model\n{strip_thinking(C)}<turn\|>\n` |

### 3.2 与 Qwen 共用（不变）

从原始请求体读取的 reasoning、响应字段、追加规则（content 和 reasoning 在 `strip()` 之后比较）、历史证据的名称、路由、strip 的 `plan` 和
`_commit_cut`、引擎在 X 处对全局池的提交、切点的证据、Goal 的上一条回复 = C，以及回复上限：`--enable-thinking` 对任何 profile 都把
`--max-tokens` 默认设为 4096（reasoning 计入其中，与 baseline 相同）。

### 3.3 Gemma 特有的部分

- **固定值。** `thinking` 记录区分 profile：Gemma 的记录写明生成头 `<|turn>model\n`、模板参数 `{enable_thinking: true}`（没有
  `preserve_thinking`：它对文本轮次不起作用）、channel 标记、`thought\n` 前缀和切分规则。Qwen 的记录逐字节不变。
- **初始渲染。** `enable_thinking=True`：`<|think|>` 的 system 轮次（1.2 节），启动时检查。
- **chat-text keep。** 一条过去的回复是 `<|turn>model\n` +（消息带 reasoning 时为 `<|channel>thought\n` R `<channel|>`）+
  `strip_thinking(C)` + `<turn|>\n`。丢掉 R 的客户端每轮都重建，正好得到 stock 的渲染（strip 的历史）；回传 R 的客户端则追加。
- **XML keep。** 重建从 content 重新编码每条过去的回复（`<|turn>model\n` + `strip_thinking(C)` + `<turn|>\n`，R 不重新渲染，与 Qwen
  相同）；这也是模板自己对我们 XML 形态的渲染。
- **kv-session。** 启动时从模板固定两个生成头；一个请求以其中之一结尾（`<tool_response|>` 之后是工具生成头）；`header_tokens` 为 0 的
  thinking 关路径只用于 thinking 关。
- **strip 片段。** 回复 head `<|turn>model\n` **就是**整个生成头：K = |H| = 3（Qwen 是 3 < 5），所以 strip 的 head 检查在 Gemma 上允许
  K = |H|。`strip_content` 就是模板的 `strip_thinking`。

## 4. 窗口布局中的 strip（形式 E）

### 4.1 几何（chat-text，thinking 开）

生成头 H = `<|turn>model\n`，3 个 token。第 t 轮：prompt N_t，X_t = N_t - 3。下一个 prompt 把回复重新渲染为 `<|turn>model\n` C
`<turn|>\n`；它的前 3 个 token 是旧的生成头，所以稳定结尾是 X + 3，轮次几何与 Qwen 相同（C = X < o = X + 3 < e < N）。阶段 1 重新
prefill `<|turn>model\n` + C + `<turn|>\n` + user 片段；Gemma 的阶段 1 在超过一次 native 前向时本来就分块运行。

### 4.2 为什么切点需要 hold

Gemma 没有 GDN 状态。跨轮次的是滑动窗口：session 在每个滑动组中钉住 [C' - 1023, C') 的 block。阶段 1 结束时 e = X 处的窗口存在，交接把
它交给阶段 2。但阶段 2 不保留它：vLLM 的滑动窗口管理器释放所有 token 都在下一个 query 的窗口之前的 block，所以大约解码 16 个 token 后
X 处窗口最早的 block 开始被释放，大约 1,024 个之后整个窗口都没了。在下一轮重算窗口需要对 [X - 1023, X) 做一次完整的 60 层前向，这还会
写入已封存 Page 的全局 KV；与 Qwen 一样，这个做法被否决。

### 4.3 方案 G：在 GPU 上保留窗口

对交接的窗口 block（延续轮次）或预填超过 X 之后 X 处的窗口（本地第 0 轮）多加一个引用；用保留的引用做 `pin_after(X)`；阶段 2 自己的 block
照常释放。代价：每个正在解码的 strip session 至多多一个窗口槽，65 个 block x 5 个滑动组 x 2.5 MiB = **P = 16 时 812.5 MiB**，在准入时
预留（4,096 token 的回复比 1,024 token 的窗口活得久）。在 25 GiB 的 KV 中只放得下 H 个 hold 的共享预留。下表由 runtime 自己的容量计算得出
（`kv_layout.window_pool_blocks`、`window_capacity.selective_pages` 和 `most_hold_slots`；`tests/test_gemma_hold.py` 的
`CapacityTableTests` 重算每一格）：在 8,192 token 的选择下，超出 launcher 选择性规则之外剩下的 pool 0 Page（负数 = 拒绝），以及放得下的
最多 hold slot 数，即 launcher 的默认值。有 hold slot 时，规则还计入 strip 轮次重新 prefill 的回复（4.4 节，“launcher”），4,096 token
的回复为 256 个 Page：

| 服务（P = 16，4,096 token 的回复，25 GiB） | H = 0（不用 strip） | H = 1 | H = 2 | H = 3 | H = 4 | H = 5 | 最多 H |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TB `observation_query`，K = 20，262,144 token，4,096 token 的前向 | +2,798 | +1,899 | +1,255 | **+612** | -32 | -675 | 3 |
| TB `goal_query`，K = 20 | +1,531 | **+632** | -12 | -655 | -1,299 | -1,942 | 1 |
| SWE `observation_query`，16，131,072，8,192 token 的前向 | +2,330 | +1,430 | +787 | **+143** | -500 | -1,144 | 3 |
| SWE `goal_query`，16 | +1,063 | **+163** | -480 | -1,124 | -1,767 | -2,411 | 1 |

一个 hold slot 是 pool 1 的 325 个 block id（每个滑动组 65 个），812.5 MiB，相当于 650 个全局 Page 的空间：第一行中 pool 1 从 19.00 GiB
（H = 0）增长到 21.38 GiB（H = 3），pool 0 从 4,918 个 Page 减少到 2,968 个。两条 SWE 行只是展示容量计算：strip 只服务 `chat-text`
（4.5 节）。11-09 的研究为第三行列出 H = 2；H = 3 在那里也放得下。它的规则没有重新 prefill 的回复这一项（11-10 的审查）：按它，前两行
各取 4 和 2 个 hold slot，而加上这一项后分别差 32 和 12 个 Page。一个 session 槽和一个 hold slot 占的空间相同，所以少一个 session 就能换回
一个 hold slot（19 个 session 的 `goal_query`：H = 2，+633 个 Page）。第一行用 24 个 session 时（不用 hold slot 时放得下的最多数目，+220
个 Page），连一个 hold slot 都放不下（-679），launcher 在那里拒绝 strip 服务。

**H 限制的是什么。** strip 轮次在准入时拿到 hold slot，在提交时（或失败时）交回：门控（`window_capacity.hold_wait`）计入每个在途或停住的
准入（lifecycle 状态 `rendered`、`reserving` 和 `accepted`）以及引擎中的每个 hold。所以 H 限制的是在途的 strip 轮次，而不只是正在解码的
回复：阶段 1、停在 Goal 处的 `goal_query` 准入和冷 prefill（第一轮、重建）各占一个 slot。门控只等待，从不抢占。开
`--admission-overlap K` 时在途的准入至多 min(K, H) 个，一个 Goal 批次（`--goal-rewriter-batch N`，发送那一刻排队的调用，至多 N 个）至多
H 个调用：在 H = 1 时，也就是 Terminal-Bench `goal_query` 那一行放得下的最多数目，服务从准入到提交一次只跑一个轮次，每个 Goal 批次只有
一个调用。

**H 小一些反而可能服务更多。** 放得下的最多数目（launcher 的默认值）让 pool 0 停在选择性规则的最小值，它能容纳一个最坏情况的轮次（包括其自己的
钉住）加上其他 session 的 tail，而不是 H 个这样的轮次；为它会释放空闲 session 的 Page。25 GiB、K = 20 时 Terminal-Bench 两行的 pool 0，
由代码（`window_capacity.pool_zero_blocks` 和 `selective_pages`）得出，单位为全局 Page，括号中是规则的数值：

| Terminal-Bench 行，K = 20 | H = 0（不用 strip） | H = 1 | H = 2 | H = 3 |
| --- | --- | --- | --- | --- |
| `observation_query` | 4,918（2,119） | 4,268（2,368） | 3,618（2,362） | 2,968（2,355） |
| `goal_query` | 3,638（2,106） | 2,988（2,355） | 拒绝 | 拒绝 |

每个 hold slot 拿走 650 个 Page，它们本来可以容纳恢复、重新准入和同时运行的各轮次的行。上下文超过 8,192 token 的阈值之后，一个轮次让 512
个 Page 的选择常驻，每次运行再加至多一次前向和一条回复（513 个 Page），所以最大的 H 可能让门控在 hold slot 空闲时等待全局 Page
（`CAPACITY_WAIT: FA pool`），而较小的 H 可能同时运行更多轮次。哪个 H 最适合 Terminal-Bench 要靠测量（GG2 按原因统计等待）。主机拷贝
（方案 H，K = 20 时约 16 GiB 锁页内存）和 32 GiB 的 KV 没有被选用。

### 4.4 实现的 hold（`remote_ra_live_v2/window_hold.py`）

**保留的是什么。** 切点 X 处的滑动窗口：五个滑动组中每一组 [X - 1023, X) 的 block（`sliding_window.kept_blocks(X)`），每组至多 65 个
block，一个窗口槽。strip 请求对它们多加一个引用，并在阶段 2 解码越过 X 时保持它；vLLM 照常释放请求自己的行。

| 步骤 | 引擎（`window_pins.WindowPins`、`adapter.EngineCoreAdapter`） | Worker（`window_owner`、`window_lifecycle`） |
| --- | --- | --- |
| 延续轮次，安装时 | `hold(..., 'install')`：对安装的 head，即阶段 1 结束时的窗口（e = X 处的窗口），多加一个引用 | 切点窗口从阶段 2 第一次前向的 native 行读出，并与阶段 1 结束时的窗口核对 |
| 本地第 0 轮 | 在游标到达或越过 X 的第一次分配之前 `hold(..., 'prefill')`；在越过 X 的那一步就结束的单 token 回复从它的行钉住，不用 hold | 切点窗口来自包含 X 或在 X 结束的 prefill 前向的 native 行；不需要有前向恰好在 X 结束，因为无论哪个分块算出窗口的 KV，它都是精确的 |
| 提交 | 从 hold 做 `pin_after(X)`：它的引用成为 session 的钉住；session 像 Qwen 一样在 X 处提交（在 floor(X / P) 处封存，tail 的有效结尾为 X） | drain 在 X 处提交 `WindowCarry` |
| 准入失败、取消、中止 | 释放 hold | 被保留的 block key 在该请求的 drain 时释放 |

**H，hold slot。** `kv_layout.hold_slots` = H。pool 1 多预留 H 个窗口槽：1 + 5 x（S x 65 + H x 65 + 前向的临时空间 + Goal 的临时空间）个
block id。同时至多准入 H 个保留或将要保留窗口的 strip 请求：容量门控（也是 `--admission-overlap` 的 ledger 门控和空闲释放之前的门控）回答
`CAPACITY_WAIT: window hold slots: H of H in use`，轮次等待；不抢占任何内容。每个 strip 轮次的门控数值都带上它的 hold 项，每个滑动组
min(|kept_blocks(X)|, ceil((3 + max_tokens) / P) + 1) 个 block，在 4,096 token 的回复上限下是一个完整的槽。

**launcher。** `serve.py --window-hold-slots H`（1 到运行中的 session 数；只用于带 `--thinking-history strip` 的 Gemma 4 profile）预留 H
个 hold slot。H 按服务的规则（benchmark 各行上的选择性规则）对照 pool 0 检查，剩下的 Page 放不下时拒绝。不给时，launcher 取**放得下的最多
数目**（4.3 节表格的最后一列），连一个都放不下时拒绝该服务。这个数目作为 `kv_layout.hold_slots` 固定在 `startup.json` 和 vLLM 的
`--additional-config` 中，这是 strip 唯一改变的 vLLM 参数；dry run 和 `launch.json` 写明它及其来源（`window_hold`），`ready.json` 带有
worker 回显的布局。默认值给出最多的在途 strip 轮次和规则允许的最小 pool 0；benchmark 计划应当固定 H，以免之后容量计算的改动使它变化。

**选择性规则中重新 prefill 的回复**（`window_capacity.strip_reprefill_pages`；11-10 的审查，发现 8）。规则中最坏的轮次持有两段，各是一次前向
加一条回复：上一轮封存的那段（常驻）和本轮的行。strip 轮次的阶段 1 在 observation 之前再算一遍上一条回复的 content，两者超过一次前向时
分块，所以它的那段至多是一次前向加一个回复上限的 content。在切点处提交丢掉最后一条回复，这为一份这样的 content 留出了空间，但这一轮在它
自己的回复旁边持有两份（上一轮重新 prefill 的 content，在它封存的那段中，以及本轮的）：没有这一项的规则只在这两份 content 合起来不超过一个
回复上限时才覆盖 strip 轮次。两者都达到 4,096 token 上限时，门控对该轮次自己的计数（它的钉住和阶段 2 的需求，即空闲释放在回答
`CAPACITY_EXCEEDED` 之前与 pool 0 比较的量）在 Terminal-Bench 那一行上比规则多 255 个 Page（`StripRuleTests`）。所以有 hold slot 时，
规则在它的两个峰值上都加上 ceil(max_tokens / P) 个 Page，4,096 token 的回复为 256 个；launcher 对 H 的计算、它的拒绝以及它为 vLLM 的启动
检查固定的 `request_pages` 都计入这一项。没有 hold slot 时（除 Gemma strip 服务之外的所有服务）数值不变。

**证据。** native 文件的 `truncate` 增加 `window_hold`：`source`（`install`、`prefill` 或 `row`）、`end`（X）、`first` 和 `last`
（`kept_blocks(X)` 的 block）、`held_at`（保留时的游标）和 `left`（到提交时第一个滑动组中有多少个被保留的 block 已离开该请求的行）。
runtime 行的 `truncate.window_hold` 是同一条记录。

**清理失败的 hold**（11-10 的审查，发现 6）。阶段 2 中的取消或中止如果 drain 失败（worker 的 `cancel_and_drain` 抛出异常），lifecycle 条目会
停在 `drain_pending` 并带有 `cleanup_error`，session 被标为不可用，hold 仍在原处：`after_drain` 中止该请求、让 native 释放解除 hold
（`WindowPins.after_free`），但它只在 drain 成功之后运行。错误交给发起取消的一方（路由的 Cancel；在 vLLM 的中止路径上由 compat lifecycle
的 `free_request` 抛出）。在此之前 hold 占着它的槽，`hold_wait` 计入它：H = 1 时其他每个 strip 轮次都等待（`CAPACITY_WAIT: window hold
slots: 1 of 1 in use`）。它不会永远等下去：`drain_pending` 可以过期（`lifecycle.EXPIRABLE`），所以准入的租约一到期（从准入算起：
`LEASE_SECONDS`，360 s；开 `--admission-overlap K` 时为 360 + 120 (K - 1) s，K = 16 时 2,160 s），引擎循环的 lifecycle 清扫
（`compat.EngineLifecycle.tick`，它调用 `EngineLifecycle.expire`）会重试 drain。重试成功则中止该请求，hold 被释放，等待中的轮次运行；
重试失败则在清扫中抛出 `lease expiry drain failed`，`core.py` 运行清扫时没有处理它，所以 engine core 以这个错误停止。因此清理失败的 hold
占用它的槽至多一个准入租约的时间，绝不会悄悄地更久；不改代码（`test_gemma_hold.py` 的 `CleanupFailureTests` 固定了每一步）。

**交互（CPU 替身，7.1 节）。** `--goal-overlap`（它的修剪只释放比 e 处窗口更旧的 block）、`--admission-overlap`、`--release-idle-pages`
（窗口留在 GPU 上）、三条解码快速路径、`--deferred-completion-io` 以及分块的阶段 1 都能与 hold 组合。没有 strip 历史时 H = 0：布局记录中
没有该字段，不预留 block，门控没有这一项，hold 的代码一行也不运行。

### 4.5 v1 中的拒绝（决定 7）

- Gemma strip 加 `--async-scheduling`（异步 C 会在每个滑动组中作废行）：launcher、启动检查和窗口 worker 都拒绝。
- `mswea-xml` 和 `kv-session` 上的 strip（与 Qwen 相同）。
- 没有 hold slot 的 Gemma strip：`--window-hold-slots 0`、连一个都放不下的行（launcher）、`kv_layout` 中没有 `hold_slots` 的启动配置
  （启动检查）、没有 hold slot 的布局（引擎和 worker 的挂接）。
- 不带 `--thinking-history strip` 的 `--window-hold-slots`，以及 Qwen release 上的它（Qwen 的 strip 提交保留的是 GDN 状态，没有窗口）。

## 5. Benchmark 与 baseline

| Benchmark | 协议 | 开启 thinking 时的自然历史 | Gemma “与 Qwen 相同”的模式 |
| --- | --- | --- | --- |
| Terminal-Bench 2.0（Terminus 2） | chat-text | strip（observation 是 user 消息） | strip（= stock / Vanilla 的输入）；keep 可以用，但没有 stock 对应 |
| SWE-bench Verified（Mini-SWE-Agent 2.4.2 XML） | mswea-xml | 我们的 XML 是 strip；tool-calling harness 是 keep | keep（与 Qwen 相同） |
| MemoryAgentBench（`read_next` 循环） | kv-session | 阅读时 keep；默认在提问时丢掉 R | `--kv-session-thinking on`；模板在每个工具响应之后强制一个 thought |

合作者的仓库（`token_efficiency`，在 `068efec` 读取，2026-10-28）：`experiments/gemma_full_20261017/formal/` 下的每个 Gemma 配置都设
`model.enable_thinking: true`，SWE-bench（`swebench/<method>/8k.yaml`）和 MemoryAgentBench（`memagent/<method>/memagent/32k.yaml`）都是，
五种方法（vanilla、agentkv、intentkv、random_attention、sliding_window）全部如此，`max_tokens: 4096`，反思调用为
`reflection_enable_thinking: false`；Terminal-Bench 的配置在 10-30 读到的是 `enable_thinking: true`。所以 Gemma 的 baseline 是 thinking
开，我们的 Gemma 运行只在 thinking 开时与它们比较。

## 6. 风险

1. **容量（strip）。** 每个被保留的窗口 812.5 MiB；25 GiB 时 H <= 3（TB obs）、1（TB goal）、3（8,192 下的 SWE obs）或 1（SWE goal）：
   门控限制在途的 strip 轮次（4.3 节），吞吐下降。
2. **模型行为。** Gemma 可能跳过 channel，或一直思考到上限；历史渲染中 `<channel|>` 之前的 `\n` 与原始输出的不同，可能让重新渲染的对照在每个
   reasoning 块上差一个 token。
3. **文本轮次的 keep 没有模板参照。** Gemma 的 keep 运行只能与组里的 KV-session 方法（AgentKV、Sliding Window、Random）比较，不能与 Vanilla
   比较。TB 上的 strip 是唯一与 stock 输入完全相同的。
4. **前缀变化。** 初始段多 7 个 token（有 system 消息时多 2 个）；thinking 开的运行与 thinking 关的是不同的配置。
5. **GPU 上的窗口沿革**，涉及快速路径 2 / 3 的紧凑行、incarnation 和 CUDA graph：主要的仅在 GPU 上的风险；GG1 的切点对重建的对比检查它。

## 7. 测试与 GPU 检查

### 7.1 CPU

- `test_gemma_thinking_on.py`（Tier 1）：固定值和拒绝；切分（没有 channel、已闭合、未闭合、C 中的 channel、注入的情况）；`<channel|>`
  之后的 MarkerStop；两个协议上的追加和重建；有 `REMOTE_RA_GEMMA4_METADATA` 时，固定模板带 `<|think|>` 的初始段和各生成头；一条
  MemoryAgentBench 形态的 kv-session 流等于模板在 `enable_thinking=True, preserve_thinking=True` 下的渲染；端到端，Gemma chat-text 服务上的
  Terminus 2 session 和经过路由服务的 kv-session 阅读循环；identity probe（Qwen 和 Gemma thinking 关的记录、生成头和流都不变）。
- `test_gemma_thinking_cut.py`（Tier 2，transcript 一侧）：T1，对随机 content（包括内联的 channel），回复片段等于 stock 的渲染，head =
  生成头；T2，经过 Gemma 轮次、带合成 reasoning 的 Terminal-Bench 形态 session（Terminus 2 的客户端：普通回合、摘要交接、在上限处被截断的
  回复、上下文拒绝），以及带摘要交接的 Terminus 2.0.0 的确切 prompt，每个 prompt 都等于 stock 的渲染（没有 4 个 token 的差异），在
  `REMOTE_RA_GEMMA4_METADATA` 指明时用固定的 tokenizer 和模板；记录下来的 178 个 Terminal-Bench trial 是私有的，所以它们的重放只是可选的
  （`REMOTE_RA_TB_ATIF_DIR`，只计数）。固定值：Gemma strip 配置需要 hold slot，launcher 的 `--window-hold-slots`，dry run 和 `ready.json`
  对 H 的记录。
- `test_gemma_hold.py`（被保留的切点窗口，T3 / T4）：布局的 `hold_slots` 及其固定值；重算 4.3 节的容量表；launcher 的容量计算和拒绝；
  vLLM 滑动窗口管理器之上的 `WindowPins`，W = 1,024（在安装时和本地第 0 轮保留，少于 16、约 1,000 和多于 1,024 token 的回复，取消和丢弃）；
  端到端，在 Gemma chat-text 服务后面的窗口布局引擎和 worker 上，用 runtime 的 Gemma strip transcript 和一个生成头是固定 3 个 token 的
  替身 tokenizer：带保留窗口在 X 处提交、切点之后的轮次、本地第 0 轮、阶段 2 中的取消、切点之后的重建、只有一个 hold slot 时的门控、
  满的池；三条解码快速路径、`--deferred-completion-io`、`--admission-overlap`、`--release-idle-pages`；对 hold 的变异；H = 0 时什么都不运行。
- `test_gemma_strip_e2e.py`（集成）：短于一个 block、略短于一个窗口和长于一个窗口的回复，没有 channel 的回复和在 channel 内被截断的回复；
  带摘要交接的 Terminus 2 session，服务的每个 prompt 都等于 stock 开启 thinking 的渲染；长于一次 native 前向的重新 prefill（分块的阶段 1）；
  goal_query 场景上带 hold 的 `--goal-overlap`（加 `--admission-overlap`、`--release-idle-pages` 和快速路径），每个 session 都等于不带
  `--goal-overlap` 的同一服务；引擎和窗口 worker 之间的 strip 记录；对接线的变异。

### 7.2 GPU（ghx4 batch，一张 GH200；GG1 已运行，GG2 未运行；结果见 docs/status.zh-CN.md）

- **GG1，token 对齐**（与 THG1 相同）：一个 session，`full`，eager 和 `--cuda-graphs full`，一个 strip 服务和一个 keep 服务，对照带
  `--reasoning-parser gemma4` 和 `enable_thinking: true` 的 stock vLLM 0.24.0：每个 prompt 都等于 stock 的（strip 完全相同），回复对照 stock
  自己的波动，切点之后的轮次对照强制的重建，冷探测（(X - 1023) mod 16 为 0、1 和 15；生成头跨越一次前向的边界；少于 16、约 1,000 和多于
  1,024 token 的回复；未闭合的 channel；没有 channel 的回复），一个 kv-session keep session，并测量 `<channel|>` 之前的 `\n`。约 2-2.5 GPU-h。
- **GG2，Terminal-Bench dry run**（与 strip25 相同）：`gstripobs262`、`gstripgoal262` 和一个 stock 开启 thinking 的对照，16 个任务，262,144
  token，25 GiB，K = 20，`observation_query` 用 H = 3，`goal_query` 用 H = 1（放得下的最多数目，4.3 节）。约 2-2.5 GPU-h。
- **冒烟（Tier 1）：** 一次 Gemma SWE keep 冒烟（约 0.9 GPU-h），以及客户端支持之后的一次 Gemma MAB thinking 开冒烟（约 0.75 GPU-h）。
